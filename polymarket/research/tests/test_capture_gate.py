"""Tests for mm_eval/capture_gate.py — the required pre-analysis capture-quality gate.

Synthetic event streams exercise every classification branch (clean / clean_lagged /
mismatch / stale reasons / trade coherence / inferred gaps) plus the threshold verdicts
and the fail-closed ``require_pass``. One integration test round-trips tiny Parquet
shards in the VPS cloud layout through ``run_slice``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mm_engine.events import GapMarker, bba_event, book_event, price_change_event, trade_event
from mm_eval.capture_gate import (
    CaptureQualityError,
    CaptureQualityGate,
    GateThresholds,
    SliceGateResult,
    TokenStats,
    require_pass,
)

TOK = "token1"


def _book(ts, bids, asks, ns=0):
    return book_event(asset_id=TOK, market="m", bids=bids, asks=asks,
                      ts_exchange=ts, ts_local_iso="", ts_monotonic_ns=ns)


def _pc(ts, side, price, size, ns=0):
    return price_change_event(asset_id=TOK, market="m", price=price, side=side, size=size,
                              ts_exchange=ts, ts_local_iso="", ts_monotonic_ns=ns)


def _bba(ts, bid, ask, ns=0):
    return bba_event(asset_id=TOK, market="m", best_bid=bid, best_ask=ask,
                     bid_size=None, ask_size=None, ts_exchange=ts, ts_local_iso="", ts_monotonic_ns=ns)


def _trade(ts, price, side="BUY", size=1.0, ns=0):
    return trade_event(asset_id=TOK, market="m", price=price, side=side, size=size,
                       ts_exchange=ts, ts_local_iso="", ts_monotonic_ns=ns)


BIDS = [[0.40, 10.0], [0.39, 5.0]]
ASKS = [[0.42, 8.0], [0.43, 4.0]]


def test_clean_checkpoint() -> None:
    res = CaptureQualityGate().run_events(iter([
        _book(1000, BIDS, ASKS),
        _bba(2000, 0.40, 0.42),
        _pc(3000, "BUY", 0.40, 9.0),   # later event resolves the pending checkpoint
    ]))
    s = res.overall
    assert (s.checks, s.clean, s.clean_lagged, s.mismatch, s.stale) == (1, 1, 0, 0, 0)


def test_lead_lag_checkpoint_classified_clean_lagged() -> None:
    """The JOIN-1 artifact: BBA frame lands (receive clock) BEFORE its own triggering
    price_change in the same millisecond. Raw scoring calls it a mismatch; the gate
    must resolve it as clean_lagged once same-ms depth updates are applied."""
    res = CaptureQualityGate().run_events(iter([
        _book(1000, BIDS, ASKS, ns=1),
        # same ms=2000: bba (ns=10) claims the NEW bid 0.41 before the pc (ns=20) applies it
        _bba(2000, 0.41, 0.42, ns=10),
        _pc(2000, "BUY", 0.41, 3.0, ns=20),
        _pc(3000, "BUY", 0.41, 4.0, ns=30),   # ts advances -> resolve
    ]))
    s = res.overall
    assert (s.checks, s.clean, s.clean_lagged, s.mismatch) == (1, 0, 1, 0)


def test_genuine_mismatch_survives_same_ms_flush() -> None:
    res = CaptureQualityGate().run_events(iter([
        _book(1000, BIDS, ASKS),
        _bba(2000, 0.99, 0.999),   # never reflected by any depth update
        _pc(3000, "BUY", 0.40, 9.0),
    ]))
    s = res.overall
    assert (s.checks, s.clean, s.clean_lagged, s.mismatch) == (1, 0, 0, 1)


def test_checkpoint_pending_at_stream_end_still_resolves() -> None:
    res = CaptureQualityGate().run_events(iter([
        _book(1000, BIDS, ASKS),
        _bba(2000, 0.40, 0.42),
    ]))
    assert res.overall.clean == 1


def test_stale_no_anchor_and_stale_window() -> None:
    res = CaptureQualityGate().run_events(iter([
        _bba(500, 0.40, 0.42),          # before any snapshot -> no_anchor
        _book(1000, BIDS, ASKS),
        _bba(20_000, 0.40, 0.42),       # >5s after last depth update -> window
    ]))
    s = res.overall
    assert s.stale == 2 and s.stale_no_anchor == 1 and s.stale_window == 1
    assert s.checks == 2 and s.fresh == 0


def test_trade_in_and_out_of_spread() -> None:
    res = CaptureQualityGate().run_events(iter([
        _book(1000, BIDS, ASKS),
        _trade(1500, 0.41),   # inside [0.40, 0.42]
        _trade(1600, 0.55),   # outside
    ]))
    s = res.overall
    assert (s.trades, s.trades_in_spread, s.trades_outside) == (2, 1, 1)


def test_sidecar_gap_marker_invalidates_until_reanchor() -> None:
    res = CaptureQualityGate().run_events(iter([
        _book(1000, BIDS, ASKS),
        GapMarker(reason="capture_gap"),
        _bba(1200, 0.40, 0.42),          # gap pending -> stale even though L1 matches
        _book(1300, BIDS, ASKS),
        _bba(1400, 0.40, 0.42),          # re-anchored -> clean
    ]))
    s = res.overall
    assert res.gaps_sidecar == 1
    assert s.stale == 1 and s.clean == 1


def test_heartbeat_gap_inference() -> None:
    gate = CaptureQualityGate(gap_infer_ms=30_000)
    res = gate.run_events(iter([
        _book(1000, BIDS, ASKS),
        _pc(2000, "BUY", 0.40, 9.0),
        _pc(50_000, "BUY", 0.40, 8.0),   # 48s silence -> inferred gap fires first
        _bba(50_100, 0.40, 0.42),        # book suspect until next full snapshot
        _book(51_000, BIDS, ASKS),
        _bba(51_500, 0.40, 0.42),
    ]))
    assert res.gaps_inferred == 1
    s = res.overall
    assert s.stale == 1 and s.clean == 1


def _stats(checks=10_000, clean=9_700, lagged=0, mismatch=300, stale=0,
           trades=1000, in_spread=990) -> TokenStats:
    s = TokenStats(checks=checks, clean=clean, clean_lagged=lagged, mismatch=mismatch,
                   stale=stale, trades=trades, trades_in_spread=in_spread,
                   trades_outside=trades - in_spread)
    return s


def _result(stats: TokenStats) -> SliceGateResult:
    return SliceGateResult(slice_id="t", thresholds=GateThresholds(), overall=stats,
                           per_token={TOK: stats}, event_counts={})


def test_verdict_ladder() -> None:
    assert _result(_stats()).verdict == "PASS"                                  # 97% clean
    assert _result(_stats(clean=9_000, mismatch=1_000)).verdict == "MARGINAL"   # 90%
    assert _result(_stats(clean=8_000, mismatch=2_000)).verdict == "FAIL"       # 80%
    assert _result(_stats(checks=100, clean=97, mismatch=3)).verdict == "UNDERPOWERED"
    # lead-lag-aware: lagged checkpoints count toward clean
    assert _result(_stats(clean=3_000, lagged=6_700, mismatch=300)).verdict == "PASS"
    # stale cap trips even when fresh checkpoints are clean
    assert _result(_stats(checks=20_000, stale=10_000)).verdict == "FAIL"
    # trade coherence floor
    assert _result(_stats(in_spread=900)).verdict == "MARGINAL"


def test_require_pass_fail_closed() -> None:
    require_pass(_result(_stats()))                                    # PASS -> ok
    marginal = _result(_stats(clean=9_000, mismatch=1_000))
    require_pass(marginal)                                             # MARGINAL ok by default
    with pytest.raises(CaptureQualityError):
        require_pass(marginal, allow_marginal=False)
    with pytest.raises(CaptureQualityError):
        require_pass(_result(_stats(clean=8_000, mismatch=2_000)))     # FAIL


def test_run_slice_parquet_roundtrip(tmp_path: Path) -> None:
    """Tiny shards in the VPS cloud layout replay through run_slice end-to-end."""
    d = tmp_path / "2026-07-01" / "testuni"
    d.mkdir(parents=True)
    base = dict(received_at="2026-07-01T00:00:00.000Z", universe="testuni", market="m")
    pd.DataFrame([
        {**base, "timestamp_ms": 1000, "received_ns": 1, "asset_id": TOK,
         "bids": '[["0.40","10"],["0.39","5"]]', "asks": '[["0.42","8"],["0.43","4"]]'},
    ]).to_parquet(d / "book_testuni_00.parquet")
    pd.DataFrame([
        {**base, "timestamp_ms": 2000, "received_ns": 2, "asset_id": TOK,
         "price": 0.40, "side": "BUY", "size": 9.0, "best_bid": 0.40, "best_ask": 0.42},
    ]).to_parquet(d / "price_change_testuni_00.parquet")
    pd.DataFrame([
        {**base, "timestamp_ms": 2500, "received_ns": 3, "asset_id": TOK,
         "best_bid": 0.40, "best_ask": 0.42, "spread": 0.02},
    ]).to_parquet(d / "bba_testuni_00.parquet")
    pd.DataFrame([
        {**base, "timestamp_ms": 2600, "received_ns": 4, "asset_id": TOK,
         "price": 0.41, "side": "BUY", "size": 1.0, "fee_rate_bps": 0.0, "transaction_hash": "0x0"},
    ]).to_parquet(d / "trades_testuni_00.parquet")

    states = []
    res = CaptureQualityGate().run_slice(d, on_state=lambda ev, st: states.append((ev.type, st.stale)))
    s = res.overall
    assert s.checks == 1 and s.clean == 1
    assert s.trades == 1 and s.trades_in_spread == 1
    assert res.event_counts == {"book": 1, "price_change": 1, "best_bid_ask": 1, "last_trade": 1}
    assert len(states) == 4 and not states[-1][1]   # on_state saw every event; book fresh at end
