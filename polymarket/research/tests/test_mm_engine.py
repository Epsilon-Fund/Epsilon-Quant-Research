"""Phase 0 tests for the strategy-agnostic MM engine.

Centerpiece is :func:`test_same_code_path_replay_vs_live_shadow` — the build-plan day-one
smoke: the SAME ``SymmetricQuoter`` driven by the SAME ``run_strategy`` produces identical
quote decisions whether the events come from the replay adapter (JSONL on disk) or the
live-shadow adapter (raw WS frames through the identical ``envelope()`` parser). The rest
exercise event conversion, ts_exchange ordering, the capture-gap staleness rule, the
``best_bid_ask``-is-telemetry rule, and the stub strategy/queue Protocol conformance.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mm_engine.book import BookTracker
from mm_engine.events import envelope_to_events
from mm_engine.feeds.live_shadow import FrameTransport, live_shadow_feed
from mm_engine.feeds.replay import replay_feed
from mm_engine.interfaces import (
    BookState,
    LatencyModel,
    MarketEvent,
    Order,
    QueueModel,
    Strategy,
)
from mm_engine.latency_models import DEFAULT_ROUND_TRIP_MS, ConstantLatency, SampledLatency
from mm_engine.queue_models import OptimisticQueue
from mm_engine.runner import run_strategy
from mm_engine.strategies import SymmetricQuoter
from scripts.dali_live_clob_capture import envelope


YES = "100"  # a single market's YES token id (decimal string, like real CLOB token ids)
MARKET = "0x" + "ab" * 20


# --------------------------------------------------------------------------------------
# message + envelope builders (shapes match captured data, see mm_clob_capture_semantics)
# --------------------------------------------------------------------------------------

def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _book_msg(token, ts, bids, asks):
    return {
        "event_type": "book",
        "asset_id": token,
        "market": MARKET,
        "timestamp": str(ts),
        "hash": "h",
        "tick_size": "0.001",
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
    }


def _pc_msg(ts, changes):
    # changes: list of (token, side, price, size, best_bid, best_ask)
    return {
        "event_type": "price_change",
        "market": MARKET,
        "timestamp": str(ts),
        "price_changes": [
            {
                "asset_id": t,
                "side": sd,
                "price": str(p),
                "size": str(s),
                "best_bid": str(bb),
                "best_ask": str(ba),
                "hash": "x",
            }
            for (t, sd, p, s, bb, ba) in changes
        ],
    }


def _trade_msg(token, ts, side, price, size):
    return {
        "event_type": "last_trade_price",
        "asset_id": token,
        "market": MARKET,
        "timestamp": str(ts),
        "side": side,
        "price": str(price),
        "size": str(size),
        "fee_rate_bps": "0",
        "transaction_hash": "0xdeadbeef",
    }


def _bba_msg(token, ts, bb, ba):
    return {
        "event_type": "best_bid_ask",
        "asset_id": token,
        "market": MARKET,
        "timestamp": str(ts),
        "best_bid": str(bb),
        "best_ask": str(ba),
        "spread": str(round(ba - bb, 4)),
    }


def _rec(msg, received_ms, mono):
    """A capture envelope record matching dali_live_clob_capture.envelope() output shape."""
    return {
        "received_at": _iso(received_ms),
        "received_monotonic_ns": mono,
        "event_type": msg.get("event_type"),
        "asset_ids": [],
        "assets": [],
        "message": msg,
    }


# A single-market scenario used by several tests. ts is both exchange ts and (here) recv ts.
def _scenario_messages(base=1_781_000_000_000):
    return [
        _book_msg(YES, base + 0, bids=[(0.47, 500), (0.46, 300)], asks=[(0.49, 400), (0.50, 600)]),
        _pc_msg(base + 200, [(YES, "BUY", 0.47, 600, 0.47, 0.49)]),
        _bba_msg(YES, base + 400, 0.47, 0.49),
        _trade_msg(YES, base + 600, "SELL", 0.47, 200),
        _pc_msg(base + 800, [(YES, "SELL", 0.49, 100, 0.47, 0.49)]),
    ]


# --------------------------------------------------------------------------------------
# the day-one smoke: same code path through both adapters
# --------------------------------------------------------------------------------------

def _sig(ev: MarketEvent):
    """Decision-relevant signature (local receive clocks legitimately differ live vs replay)."""
    return (ev.type, ev.token_id, ev.ts_exchange, ev.payload)


def _write_replay_jsonl(tmp_path: Path, messages) -> Path:
    # Build the replay shard via the REAL capture envelope() — the same function the
    # live-shadow adapter uses to wrap raw frames — so both feeds share the entire
    # envelope -> envelope_to_events path and only the transport (file vs socket) differs.
    # This makes the same-code-path proof complete: an envelope() bug would break parity.
    path = tmp_path / "shard.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for msg in messages:
            for rec in envelope(json.dumps(msg), {}):
                fh.write(json.dumps(rec) + "\n")
    return path


def test_same_code_path_replay_vs_live_shadow(tmp_path):
    messages = _scenario_messages()
    raw_frames = [json.dumps(m) for m in messages]

    # Replay: events come off disk.
    replay_path = _write_replay_jsonl(tmp_path, messages)
    replay_events = [e for e in replay_feed(replay_path) if isinstance(e, MarketEvent)]

    # Live shadow: identical raw frames through the real envelope() + injected transport.
    live_events = [
        e
        for e in live_shadow_feed([YES], connect=lambda *_: FrameTransport(raw_frames))
        if isinstance(e, MarketEvent)
    ]

    # Identical MarketEvents (modulo local receive clocks).
    assert [_sig(e) for e in replay_events] == [_sig(e) for e in live_events]

    # And the SAME strategy code yields byte-identical decisions over each feed.
    params = {"half_spread": 0.01, "size": 100.0, "tick": 0.001}
    replay_decisions = run_strategy(
        replay_feed(replay_path), SymmetricQuoter(), params
    )
    live_decisions = run_strategy(
        live_shadow_feed([YES], connect=lambda *_: FrameTransport(raw_frames)),
        SymmetricQuoter(),
        params,
    )
    assert replay_decisions == live_decisions

    # Sanity: the quoter actually quoted, symmetric around mid 0.48.
    first = replay_decisions[0]
    assert first.event_type == "book" and not first.stale
    sides = {o.side: o for o in first.orders}
    assert sides["BUY"].price == 0.47 and sides["SELL"].price == 0.49
    assert sides["BUY"].size == 100.0


def test_live_shadow_is_read_only_subscription():
    frames = [json.dumps(m) for m in _scenario_messages()]
    transport = FrameTransport(frames)
    list(live_shadow_feed([YES], connect=lambda *_: transport))
    # Exactly one message was sent: the market subscription. No orders, ever.
    assert len(transport.sent) == 1
    sub = json.loads(transport.sent[0])
    assert sub["type"] == "market" and sub["assets_ids"] == [YES]


# --------------------------------------------------------------------------------------
# event conversion
# --------------------------------------------------------------------------------------

def test_envelope_to_events_types_and_split():
    base = 1_781_000_000_000
    book_ev = envelope_to_events(_rec(_book_msg(YES, base, [(0.4, 10)], [(0.5, 10)]), base, 0))
    assert len(book_ev) == 1 and book_ev[0].type == "book" and book_ev[0].token_id == YES

    other = "200"
    pc = envelope_to_events(
        _rec(_pc_msg(base, [(YES, "BUY", 0.4, 5, 0.4, 0.5), (other, "SELL", 0.6, 7, 0.5, 0.6)]), base, 0)
    )
    assert [e.type for e in pc] == ["price_change", "price_change"]
    assert {e.token_id for e in pc} == {YES, other}
    assert all(e.ts_exchange == base for e in pc)  # share the message-level timestamp

    trade = envelope_to_events(_rec(_trade_msg(YES, base, "SELL", 0.47, 200), base, 0))
    assert len(trade) == 1 and trade[0].type == "last_trade"  # normalized from last_trade_price
    assert trade[0].payload["side"] == "SELL"

    bba = envelope_to_events(_rec(_bba_msg(YES, base, 0.47, 0.49), base, 0))
    assert len(bba) == 1 and bba[0].type == "best_bid_ask"


# --------------------------------------------------------------------------------------
# replay ordering + gap handling
# --------------------------------------------------------------------------------------

def test_replay_orders_by_ts_exchange(tmp_path):
    base = 1_781_000_000_000
    # Written out of order: ts 300, 100, 200.
    messages = [
        _book_msg(YES, base + 300, [(0.4, 10)], [(0.5, 10)]),
        _book_msg(YES, base + 100, [(0.4, 10)], [(0.5, 10)]),
        _book_msg(YES, base + 200, [(0.4, 10)], [(0.5, 10)]),
    ]
    path = _write_replay_jsonl(tmp_path, messages)
    events = [e for e in replay_feed(path) if isinstance(e, MarketEvent)]
    assert [e.ts_exchange for e in events] == [base + 100, base + 200, base + 300]


def test_replay_gap_marks_book_stale_until_next_book(tmp_path):
    base = 1_781_000_000_000
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # received_at == exchange ts here so gap placement (by recv time) is unambiguous.
    seq = [
        _book_msg(YES, base + 0, [(0.47, 500)], [(0.49, 400)]),          # e1 anchor
        _pc_msg(base + 1000, [(YES, "BUY", 0.47, 550, 0.47, 0.49)]),     # e2 fresh
        _pc_msg(base + 3000, [(YES, "BUY", 0.47, 560, 0.47, 0.49)]),     # e3 AFTER gap -> stale
        _book_msg(YES, base + 4000, [(0.47, 600)], [(0.49, 420)]),       # e4 re-anchor -> fresh
        _pc_msg(base + 5000, [(YES, "BUY", 0.47, 610, 0.47, 0.49)]),     # e5 fresh
    ]
    with (run_dir / "shard.jsonl").open("w", encoding="utf-8") as fh:
        for i, msg in enumerate(seq):
            fh.write(json.dumps(_rec(msg, int(msg["timestamp"]), mono=i)) + "\n")

    # Disconnect at base+2000 -> marker lands before e3 (recv >= 2000).
    with (run_dir / "capture_gaps.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": _iso(base + 2000), "event_type": "disconnect_or_error"}) + "\n")

    decisions = run_strategy(replay_feed(run_dir), SymmetricQuoter(), {})
    by_ts = {d.ts_exchange: d for d in decisions}

    assert by_ts[base + 1000].stale is False and by_ts[base + 1000].orders  # e2 quotes
    assert by_ts[base + 3000].stale is True and by_ts[base + 3000].orders == ()  # e3 stale -> no quotes
    assert by_ts[base + 4000].stale is False and by_ts[base + 4000].orders  # e4 re-anchored
    assert by_ts[base + 5000].stale is False  # e5 fresh again


def test_replay_gap_placed_when_received_at_unparseable(tmp_path):
    # If received_at is missing/malformed, gap placement must fall back to ts_exchange
    # rather than silently dropping the marker (review finding 1).
    base = 1_781_000_000_000
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def _rec_no_recv(msg, mono):
        return {
            "received_at": "",  # unparseable -> _iso_to_epoch_ms returns None
            "received_monotonic_ns": mono,
            "event_type": msg.get("event_type"),
            "asset_ids": [],
            "assets": [],
            "message": msg,
        }

    seq = [
        _book_msg(YES, base + 0, [(0.47, 500)], [(0.49, 400)]),       # anchor
        _pc_msg(base + 3000, [(YES, "BUY", 0.47, 560, 0.47, 0.49)]),  # after gap -> stale
    ]
    with (run_dir / "shard.jsonl").open("w", encoding="utf-8") as fh:
        for i, msg in enumerate(seq):
            fh.write(json.dumps(_rec_no_recv(msg, mono=i)) + "\n")
    with (run_dir / "capture_gaps.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": _iso(base + 2000), "event_type": "disconnect_or_error"}) + "\n")

    decisions = run_strategy(replay_feed(run_dir), SymmetricQuoter(), {})
    stale_by_ts = {d.ts_exchange: d.stale for d in decisions}
    assert stale_by_ts[base + 3000] is True  # gap marker landed despite empty received_at


def test_book_staleness_window():
    tracker = BookTracker(staleness_ms=5_000)
    base = 1_781_000_000_000
    tracker.apply(MarketEvent("book", YES, base, "", 0, _book_msg(YES, base, [(0.47, 1)], [(0.49, 1)])))
    # A trade 6s later, with no intervening depth update, is beyond the staleness window.
    state = tracker.apply(
        MarketEvent("last_trade", YES, base + 6_000, "", 0, _trade_msg(YES, base + 6_000, "SELL", 0.47, 1))
    )
    assert state.stale is True


# --------------------------------------------------------------------------------------
# best_bid_ask is telemetry-only (never mutates the executable book)
# --------------------------------------------------------------------------------------

def test_best_bid_ask_does_not_mutate_book():
    tracker = BookTracker()
    base = 1_781_000_000_000
    tracker.apply(
        MarketEvent("book", YES, base, "", 0, _book_msg(YES, base, [(0.47, 500)], [(0.49, 400)]))
    )
    # A bba claiming a tighter market must NOT move the executable levels.
    state = tracker.apply(
        MarketEvent("best_bid_ask", YES, base + 100, "", 0, _bba_msg(YES, base + 100, 0.48, 0.485))
    )
    assert state.bids[0] == (0.47, 500.0)
    assert state.asks[0] == (0.49, 400.0)


# --------------------------------------------------------------------------------------
# stub strategy + queue/latency
# --------------------------------------------------------------------------------------

def test_symmetric_quoter_basic():
    q = SymmetricQuoter()
    fresh = BookState(YES, bids=((0.47, 500),), asks=((0.49, 400),), ts_exchange=1, stale=False)
    orders = q.quote(fresh, inventory=0.0, params={"half_spread": 0.01, "size": 50.0})
    by_side = {o.side: o for o in orders}
    assert by_side["BUY"].price == 0.47 and by_side["SELL"].price == 0.49
    assert by_side["BUY"].size == 50.0

    # inventory-agnostic: same quotes regardless of inventory
    assert q.quote(fresh, inventory=999.0, params={"half_spread": 0.01, "size": 50.0}) == orders

    # stale or one-sided book -> no quotes
    stale = BookState(YES, bids=((0.47, 500),), asks=((0.49, 400),), ts_exchange=1, stale=True)
    assert q.quote(stale, 0.0, {}) == []
    one_sided = BookState(YES, bids=((0.47, 500),), asks=(), ts_exchange=1, stale=False)
    assert q.quote(one_sided, 0.0, {}) == []


def test_optimistic_queue_trade_through():
    qm = OptimisticQueue()
    book = BookState(YES, bids=((0.47, 500),), asks=((0.49, 400),), ts_exchange=1, stale=False)
    order = Order(YES, "BUY", 0.47, 100.0)
    base = 1_781_000_000_000

    # First a SELL of 200 at 0.47: seeds queue-ahead from the book (500), consumes 200 -> 300, no fill.
    t1 = MarketEvent("last_trade", YES, base, "", 0, _trade_msg(YES, base, "SELL", 0.47, 200))
    r1 = qm.fill(order, book, t1)
    assert r1.qty == 0.0 and r1.queue_ahead == 300.0

    # Next a SELL of 400: 300 left ahead is cleared, 100 overflow fills our order.
    t2 = MarketEvent("last_trade", YES, base + 1, "", 0, _trade_msg(YES, base + 1, "SELL", 0.47, 400))
    r2 = qm.fill(order, book, t2)
    assert r2.qty == 100.0 and r2.queue_ahead == 0.0

    # A BUY (wrong aggressor side for a resting bid) never fills it.
    t3 = MarketEvent("last_trade", YES, base + 2, "", 0, _trade_msg(YES, base + 2, "BUY", 0.47, 999))
    assert qm.fill(order, book, t3).qty == 0.0


def test_optimistic_queue_get_queue_ahead_seeds_from_cached_book():
    qm = OptimisticQueue()
    book = BookState(YES, bids=((0.47, 500),), asks=((0.49, 400),), ts_exchange=1, stale=False)
    order = Order(YES, "BUY", 0.47, 100.0)
    # on_event caches the book per token so get_queue_ahead can seed without a book arg.
    qm.on_event(MarketEvent("book", YES, 1, "", 0, {}), book)
    assert qm.get_queue_ahead(order) == 500.0


def test_constant_latency():
    lat = ConstantLatency(round_trip=12.5)
    assert lat.round_trip_ms(ts_exchange=1_781_000_000_000) == 12.5


def test_constant_latency_default_is_realistic():
    # default is the realistic ~200ms round-trip, not 0ms (instant fills)
    assert DEFAULT_ROUND_TRIP_MS == 200.0
    assert ConstantLatency().round_trip_ms(ts_exchange=1_781_000_000_000) == 200.0


def test_sampled_latency_constant_when_std_zero():
    lat = SampledLatency(mean=180.0, std=0.0)
    assert lat.round_trip_ms(1) == 180.0 and lat.round_trip_ms(2) == 180.0   # degenerate -> mean


def test_sampled_latency_is_seeded_and_reproducible():
    a = SampledLatency(mean=200.0, std=30.0, seed=7)
    b = SampledLatency(mean=200.0, std=30.0, seed=7)
    seq_a = [a.round_trip_ms(t) for t in range(50)]
    seq_b = [b.round_trip_ms(t) for t in range(50)]
    assert seq_a == seq_b                                   # same seed -> identical draws (deterministic replay)
    assert any(abs(x - 200.0) > 1e-9 for x in seq_a)        # actually dispersed, not constant
    c = SampledLatency(mean=200.0, std=30.0, seed=8)
    assert [c.round_trip_ms(t) for t in range(50)] != seq_a  # different seed -> different draws


def test_sampled_latency_same_ts_is_pure_and_repeatable():
    # round_trip_ms must be a PURE function of ts_exchange: fills.py probes a resting order
    # once per trade-check with the SAME placement_ts, and the order's submit->live latency
    # must be fixed once, not re-rolled on every probe.
    lat = SampledLatency(mean=200.0, std=40.0, seed=2)
    v = lat.round_trip_ms(1_781_000_000_777)
    assert all(lat.round_trip_ms(1_781_000_000_777) == v for _ in range(25))   # repeat -> identical
    assert lat.round_trip_ms(1_781_000_000_777.0) == v                          # keyed on int(ts)
    # call ORDER / history does not perturb a given ts's value (the old sequential-RNG bug)
    other = SampledLatency(mean=200.0, std=40.0, seed=2)
    _ = [other.round_trip_ms(t) for t in (5, 9, 1, 12345)]                      # probe other ts first
    assert other.round_trip_ms(1_781_000_000_777) == v


def test_sampled_latency_disperses_across_ts():
    lat = SampledLatency(mean=200.0, std=40.0, seed=2)
    vals = [lat.round_trip_ms(t) for t in range(1000, 1200)]   # 200 distinct submit times
    assert len(set(vals)) > 150                                # different ts -> different draws
    assert max(vals) - min(vals) > 40.0                        # genuinely spread, not constant


def test_sampled_latency_floor_clamps_nonnegative():
    # tiny mean, wide std, floor 0 -> never returns a negative round-trip
    lat = SampledLatency(mean=1.0, std=100.0, floor_ms=0.0, seed=3)
    assert all(lat.round_trip_ms(t) >= 0.0 for t in range(500))


def test_sampled_latency_from_samples_fits_mean_std():
    lat = SampledLatency.from_samples([100.0, 200.0, 300.0], seed=0)
    assert lat.mean == pytest.approx(200.0)
    assert lat.std == pytest.approx((20000.0 / 3.0) ** 0.5)   # population std of {100,200,300} ≈ 81.65
    # empty samples -> safe default, no crash
    assert SampledLatency.from_samples([]).mean == DEFAULT_ROUND_TRIP_MS


def test_sampled_latency_calibrate_refits_in_place():
    lat = SampledLatency(mean=200.0, std=0.0, seed=1)
    lat.calibrate([50.0, 50.0, 50.0, 50.0])
    assert lat.mean == pytest.approx(50.0) and lat.std == pytest.approx(0.0)
    assert lat.round_trip_ms(1) == pytest.approx(50.0)
    lat.calibrate([])                                       # no samples -> unchanged
    assert lat.mean == pytest.approx(50.0)


def test_stub_protocol_conformance():
    assert isinstance(SymmetricQuoter(), Strategy)
    assert isinstance(OptimisticQueue(), QueueModel)
    assert isinstance(ConstantLatency(), LatencyModel)
    assert isinstance(SampledLatency(), LatencyModel)
