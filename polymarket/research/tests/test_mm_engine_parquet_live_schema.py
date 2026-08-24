"""Live VPS Parquet schema regression — pins the REAL pipeline schema, not the converter's.

JOIN 1 surfaced a latent bug: ``replay_parquet`` hard-coded the ``bba`` column list as
``…, best_bid, best_ask, bid_size, ask_size``, but the **live VPS compression pipeline** ships
a different schema, so the adapter raised ``BinderException`` on the durable, gap-free Parquet it
exists to validate. The "byte-identical to JSONL" equivalence had only ever been tested against
the *local converter's* output (``test_mm_engine_parquet.py``), never the live pipeline's schema.

This test pins the live schema so the bug cannot recur. The schema below is the authoritative
one captured with ``DESCRIBE`` on a real shard
(``r2:epsilon-polymarket-data/parquet/2026-06-30/politics_negrisk/*_12.parquet``):

* every table carries an extra ``universe`` column (right after ``received_ns``);
* ``bba`` ships ``best_bid``/``best_ask``/**``spread``** — there is **no** ``bid_size``/``ask_size``
  (PM's ``best_bid_ask`` frame never carried sizes; the engine never reads them either);
* ``price_change`` carries extra ``best_bid``/``best_ask`` columns (the adapter drops them — the
  canonical ``price_change`` payload omits them by design, see ``events.price_change_event``);
* ``trades`` carry extra ``fee_rate_bps``/``transaction_hash``;
* ``book`` ``bids``/``asks`` are a JSON array of **string-valued dicts**
  ``[{"price":"..","size":".."}, …]`` (the converter writes ``[[price,size], …]`` instead —
  both are accepted by ``events._norm_levels``).

The companion ``test_mm_engine_parquet.py`` (converter path) is kept intact; this file adds the
live-schema coverage the bug slipped through.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mm_engine.engine import BACKTEST, run_engine
from mm_engine.events import GapMarker
from mm_engine.fees import FeeModel, FeeSchedule
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
UNIVERSE = "politics_negrisk"

# --- authoritative live VPS schema (column order matches the real shard) ----------------
_LIVE_SCHEMA = {
    "book": pa.schema([
        ("timestamp_ms", pa.int64()), ("received_at", pa.string()), ("received_ns", pa.int64()),
        ("universe", pa.string()), ("asset_id", pa.string()), ("market", pa.string()),
        ("bids", pa.string()), ("asks", pa.string()),
    ]),
    "trades": pa.schema([
        ("timestamp_ms", pa.int64()), ("received_at", pa.string()), ("received_ns", pa.int64()),
        ("universe", pa.string()), ("asset_id", pa.string()), ("market", pa.string()),
        ("price", pa.float64()), ("size", pa.float64()), ("side", pa.string()),
        ("fee_rate_bps", pa.float64()), ("transaction_hash", pa.string()),
    ]),
    "price_change": pa.schema([
        ("timestamp_ms", pa.int64()), ("received_at", pa.string()), ("received_ns", pa.int64()),
        ("universe", pa.string()), ("asset_id", pa.string()), ("market", pa.string()),
        ("price", pa.float64()), ("side", pa.string()), ("size", pa.float64()),
        ("best_bid", pa.float64()), ("best_ask", pa.float64()),
    ]),
    "bba": pa.schema([
        ("timestamp_ms", pa.int64()), ("received_at", pa.string()), ("received_ns", pa.int64()),
        ("universe", pa.string()), ("asset_id", pa.string()), ("market", pa.string()),
        ("best_bid", pa.float64()), ("best_ask", pa.float64()), ("spread", pa.float64()),
    ]),
}


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _rec(msg, received_ms, mono):
    """One JSONL capture envelope with DETERMINISTIC receive clocks (so the JSONL↔Parquet
    comparison also covers ts_local_iso / ts_monotonic_ns, exactly as the converter test does)."""
    return {
        "received_at": _iso(received_ms),
        "received_monotonic_ns": mono,
        "event_type": msg.get("event_type"),
        "asset_ids": [],
        "assets": [],
        "message": msg,
    }


def _scenario():
    # book(YES) anchors; a fillable SELL trade(YES) hits our resting BUY (must precede any
    # other-token event — the engine cancels a token's quotes on the next other-token event);
    # a 2-asset price_change (same ts+ns -> exercises the content tie-break AND the live
    # price_change.best_bid/ask columns the adapter must DROP); a bba(YES) (L1 cross-check +
    # the missing-sizes NULL-fill); a NO book; a re-anchor book(YES).
    return [
        (book_msg(YES, BASE + 0, [(0.47, 50), (0.46, 300)], [(0.49, 400), (0.50, 600)]), BASE + 0),
        (trade_msg(YES, BASE + 50, "SELL", 0.47, 200), BASE + 50),
        (pc_msg(BASE + 150, [(YES, "BUY", 0.46, 350, 0.47, 0.49),
                             (NO, "SELL", 0.53, 100, 0.51, 0.53)]), BASE + 150),
        (bba_msg(YES, BASE + 200, 0.47, 0.49), BASE + 200),
        (book_msg(NO, BASE + 250, [(0.51, 80)], [(0.53, 200)]), BASE + 250),
        (pc_msg(BASE + 300, [(YES, "BUY", 0.47, 560, 0.47, 0.49)]), BASE + 300),
        (book_msg(YES, BASE + 400, [(0.47, 600)], [(0.49, 420)]), BASE + 400),
    ]


def _write_jsonl(path, recs):
    with path.open("w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")


def _write_live_parquet(out_root, date, universe, shard, recs):
    """Write the SAME records as the live VPS pipeline would — typed tables in the live schema,
    reusing each rec's deterministic clocks so the two sources encode identical underlying data."""
    rows = {t: [] for t in _LIVE_SCHEMA}
    for rec in recs:
        msg = rec["message"]
        ts = int(msg["timestamp"])
        common = {
            "timestamp_ms": ts, "received_at": rec["received_at"],
            "received_ns": int(rec["received_monotonic_ns"]), "universe": universe,
            "market": msg.get("market"),
        }
        et = msg.get("event_type")
        if et == "book":
            rows["book"].append({**common, "asset_id": msg["asset_id"],
                                 "bids": json.dumps(msg["bids"]), "asks": json.dumps(msg["asks"])})
        elif et == "last_trade_price":
            rows["trades"].append({**common, "asset_id": msg["asset_id"],
                                   "price": float(msg["price"]), "size": float(msg["size"]),
                                   "side": msg["side"], "fee_rate_bps": float(msg.get("fee_rate_bps") or 0.0),
                                   "transaction_hash": msg.get("transaction_hash")})
        elif et == "best_bid_ask":
            bb, ba = float(msg["best_bid"]), float(msg["best_ask"])
            rows["bba"].append({**common, "asset_id": msg["asset_id"],
                                "best_bid": bb, "best_ask": ba, "spread": round(ba - bb, 4)})
        elif et == "price_change":
            for ch in msg["price_changes"]:
                rows["price_change"].append({**common, "asset_id": ch["asset_id"],
                                             "price": float(ch["price"]), "side": ch["side"],
                                             "size": float(ch["size"]), "best_bid": float(ch["best_bid"]),
                                             "best_ask": float(ch["best_ask"])})
    out = out_root / date / universe
    out.mkdir(parents=True, exist_ok=True)
    for table, table_rows in rows.items():
        if table_rows:
            pq.write_table(pa.Table.from_pylist(table_rows, schema=_LIVE_SCHEMA[table]),
                           out / f"{table}_{universe}_00.parquet")
    return out


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


@pytest.fixture
def live_fixture(tmp_path):
    recs = [_rec(msg, rm, mono=i) for i, (msg, rm) in enumerate(_scenario())]
    shard = tmp_path / "shard.jsonl"
    _write_jsonl(shard, recs)
    live_dir = _write_live_parquet(tmp_path / "pq", "2026-06-30", UNIVERSE, "00", recs)
    return shard, live_dir


def test_live_schema_reads_without_exception(live_fixture):
    # The whole point: the live bba (no bid_size/ask_size) + extra `universe`/`spread`/
    # price_change best_bid/ask columns must NOT raise (the old hard-coded column list did).
    _shard, live_dir = live_fixture
    stream = list(replay_parquet(live_dir))
    assert stream, "expected a non-empty event stream from the live-schema Parquet"
    types = {e.type for e in stream if isinstance(e, MarketEvent)}
    assert {"book", "price_change", "last_trade", "best_bid_ask"} <= types


def test_live_schema_stream_byte_identical_to_jsonl(live_fixture):
    shard, live_dir = live_fixture
    jsonl_stream = _normalize(replay_feed(shard))
    pq_stream = _normalize(replay_parquet(live_dir))
    assert jsonl_stream == pq_stream            # 0 diff: events, order, payloads (incl. ts fields)


def test_live_schema_missing_sizes_null_fill_cleanly(live_fixture):
    _shard, live_dir = live_fixture
    bbas = [e for e in replay_parquet(live_dir)
            if isinstance(e, MarketEvent) and e.type == "best_bid_ask"]
    assert bbas, "scenario must contain a best_bid_ask event"
    for e in bbas:
        # missing columns NULL-fill to None — not 0.0, not a crash
        assert e.payload["bid_size"] is None
        assert e.payload["ask_size"] is None
        # the columns the live schema DOES have still come through
        assert e.payload["best_bid"] is not None and e.payload["best_ask"] is not None


def test_live_schema_extra_columns_dropped(live_fixture):
    # universe / spread / price_change.best_bid/ask / fee_rate_bps / transaction_hash are not
    # part of the canonical MarketEvent payload — they must be silently dropped, not leaked.
    _shard, live_dir = live_fixture
    for e in replay_parquet(live_dir):
        if not isinstance(e, MarketEvent):
            continue
        assert "universe" not in e.payload and "spread" not in e.payload
        if e.type == "price_change":
            assert "best_bid" not in e.payload and "best_ask" not in e.payload
        if e.type == "last_trade":
            assert "fee_rate_bps" not in e.payload and "transaction_hash" not in e.payload


def test_live_schema_engine_output_source_invariant(live_fixture):
    shard, live_dir = live_fixture

    def _run(feed):
        return run_engine(feed, strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
                          latency_model=ConstantLatency(round_trip=0.0), mode=BACKTEST,
                          params=PARAMS, fee_model=FEE)

    a = _run(replay_feed(shard))
    b = _run(replay_parquet(live_dir))

    assert a.fill_count == b.fill_count and a.fill_count >= 1   # the SELL hit our resting BUY
    assert a.position == b.position
    assert a.realized_pnl == pytest.approx(b.realized_pnl)
    assert a.gross_pnl == pytest.approx(b.gross_pnl)
    assert a.rebates_earned == pytest.approx(b.rebates_earned)
    assert a.net_with_rebate == pytest.approx(b.net_with_rebate)
    assert a.equity_path == b.equity_path
    assert a.l1_crosscheck == b.l1_crosscheck


def test_live_schema_explicit_gaps_interleave_like_jsonl(live_fixture):
    # The live layout ships NO capture_gaps.parquet sidecar, so gaps arrive explicitly. An
    # explicit gap list must interleave identically on both the JSONL and Parquet paths.
    shard, live_dir = live_fixture
    gaps = [BASE + 220]   # between the bba(+200) and the NO book(+250)
    jsonl_stream = _normalize(replay_feed(shard, gaps=gaps))
    pq_stream = _normalize(replay_parquet(live_dir, gaps=gaps))
    assert jsonl_stream == pq_stream
    assert any(x[0] == "G" for x in pq_stream)
