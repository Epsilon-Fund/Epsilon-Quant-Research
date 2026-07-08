"""Parquet replay equivalence — converting a JSONL shard to typed Parquet and replaying it
must change NOTHING: byte-identical MarketEvent streams (incl. GapMarkers) and a
source-invariant engine run."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from mm_engine.engine import BACKTEST, run_engine
from mm_engine.events import GapMarker
from mm_engine.fees import FeeModel, FeeSchedule
from mm_engine.feeds.parquet_convert import jsonl_to_parquet
from mm_engine.feeds.replay import replay_feed
from mm_engine.feeds.replay_parquet import replay_parquet
from mm_engine.interfaces import MarketEvent
from mm_engine.latency_models import ConstantLatency
from mm_engine.queue_models import OptimisticQueue
from mm_engine.strategies import SymmetricQuoter

from mm_engine_fixtures import YES, NO, bba_msg, book_msg, pc_msg, trade_msg

BASE = 1_781_000_000_000
PARAMS = {"half_spread": 0.01, "size": 100.0, "tick": 0.001}
FEE = FeeModel(market_schedules={
    YES: FeeSchedule(0.07, 0.20, source="market"),
    NO: FeeSchedule(0.07, 0.20, source="market"),
})


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _rec(msg, received_ms, mono):
    # Controlled receive clocks so gap placement is deterministic and the JSONL <-> Parquet
    # comparison covers ts_local_iso / ts_monotonic_ns too.
    return {
        "received_at": _iso(received_ms),
        "received_monotonic_ns": mono,
        "event_type": msg.get("event_type"),
        "asset_ids": [],
        "assets": [],
        "message": msg,
    }


def _scenario():
    # book(YES) then a fillable trade(YES) FIRST (the engine quotes per-event-token and
    # cancels other tokens' orders on the next event, so the YES fill must happen before any
    # other-token event). Then a 2-asset price_change (same ts+ns -> stresses the ordering
    # tie-break), bba(YES) (L1 cross-check), a gap, a post-gap pc, and a re-anchor book.
    return [
        (book_msg(YES, BASE + 0, [(0.47, 50), (0.46, 300)], [(0.49, 400), (0.50, 600)]), BASE + 0),
        (trade_msg(YES, BASE + 50, "SELL", 0.47, 200), BASE + 50),     # fills our YES BUY (50 ahead)
        (book_msg(NO, BASE + 100, [(0.51, 80)], [(0.53, 200)]), BASE + 100),
        (pc_msg(BASE + 150, [(YES, "BUY", 0.46, 350, 0.47, 0.49),
                             (NO, "SELL", 0.53, 100, 0.51, 0.53)]), BASE + 150),
        (bba_msg(YES, BASE + 200, 0.47, 0.49), BASE + 200),
        (pc_msg(BASE + 300, [(YES, "BUY", 0.47, 560, 0.47, 0.49)]), BASE + 300),
        (book_msg(YES, BASE + 400, [(0.47, 600)], [(0.49, 420)]), BASE + 400),
    ]


def _write_fixture(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shard = run_dir / "shard.jsonl"
    with shard.open("w", encoding="utf-8") as fh:
        for i, (msg, recv_ms) in enumerate(_scenario()):
            fh.write(json.dumps(_rec(msg, recv_ms, mono=i)) + "\n")
    # one disconnect between the trade (BASE+200) and the post-gap pc (BASE+300)
    with (run_dir / "capture_gaps.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": _iso(BASE + 250), "event_type": "disconnect_or_error"}) + "\n")
    return run_dir, shard


def _normalize(stream):
    out = []
    for item in stream:
        if isinstance(item, MarketEvent):
            out.append(("E", item.type, item.token_id, item.ts_exchange,
                        item.ts_local_iso, item.ts_monotonic_ns, item.payload))
        elif isinstance(item, GapMarker):
            out.append(("G", item.reason, item.detail))
        else:
            raise AssertionError(f"unexpected stream item: {item!r}")
    return out


def test_parquet_stream_byte_identical_to_jsonl(tmp_path):
    run_dir, shard = _write_fixture(tmp_path)
    conv = jsonl_to_parquet(shard, tmp_path / "pq", date="2026-06-20", universe="fixture", shard="s0",
                            gaps_path=run_dir / "capture_gaps.jsonl")
    conv_dir = conv["dir"]

    jsonl_stream = _normalize(replay_feed(shard))
    pq_stream = _normalize(replay_parquet(conv_dir))

    assert jsonl_stream == pq_stream                      # 0 diff: events, order, payloads, GapMarkers
    # the scenario actually exercised all the moving parts:
    kinds = [x[0] for x in jsonl_stream]
    assert "G" in kinds                                   # a GapMarker was interleaved
    types = [x[1] for x in jsonl_stream if x[0] == "E"]
    assert {"book", "price_change", "last_trade", "best_bid_ask"} <= set(types)


def test_same_asset_multichange_frame_source_invariant(tmp_path):
    # Two changes for the SAME asset in ONE frame share (ts_exchange, ts_monotonic_ns,
    # token_id, type) -> only the content tie-break disambiguates them. JSONL (record order)
    # and Parquet (table read order) must still produce the identical stream.
    run = tmp_path / "run"; run.mkdir()
    shard = run / "shard.jsonl"
    msgs = [
        (book_msg(YES, BASE + 0, [(0.47, 50)], [(0.49, 400)]), BASE + 0),
        (pc_msg(BASE + 100, [(YES, "BUY", 0.47, 600, 0.47, 0.49),
                             (YES, "BUY", 0.46, 222, 0.47, 0.49)]), BASE + 100),
    ]
    with shard.open("w", encoding="utf-8") as fh:
        for i, (m, ms) in enumerate(msgs):
            fh.write(json.dumps(_rec(m, ms, mono=i)) + "\n")
    conv = jsonl_to_parquet(shard, tmp_path / "pq", date="2026-06-20", universe="fixture", shard="s0")

    assert _normalize(replay_feed(shard)) == _normalize(replay_parquet(conv["dir"]))


def test_engine_output_source_invariant(tmp_path):
    run_dir, shard = _write_fixture(tmp_path)
    conv = jsonl_to_parquet(shard, tmp_path / "pq", date="2026-06-20", universe="fixture", shard="s0",
                            gaps_path=run_dir / "capture_gaps.jsonl")

    def _run(feed):
        # round_trip=0.0 isolates source-invariance from latency tuning (ConstantLatency now
        # defaults to a realistic 200ms); both legs use the same model, so the comparison is fair.
        return run_engine(feed, strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
                          latency_model=ConstantLatency(round_trip=0.0), mode=BACKTEST,
                          params=PARAMS, fee_model=FEE)

    a = _run(replay_feed(shard))
    b = _run(replay_parquet(conv["dir"]))

    assert a.fill_count == b.fill_count and a.fill_count >= 1
    assert a.position == b.position
    assert a.realized_pnl == pytest.approx(b.realized_pnl)
    assert a.gross_pnl == pytest.approx(b.gross_pnl)
    assert a.rebates_earned == pytest.approx(b.rebates_earned)
    assert a.net_with_rebate == pytest.approx(b.net_with_rebate)
    assert a.equity_path == b.equity_path
    assert a.l1_crosscheck == b.l1_crosscheck
