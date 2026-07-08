"""Tests for the three QueueModel variants (queue_models.py) and the engine wiring.

Coverage:
* **Bracketing** — over an identical event stream the realized fills order as
  ``Optimistic >= Prob >= RiskAverse``, both through the real engine (a crafted strict
  case) and across many seeded random streams.
* **FIX 1 — forget() teardown** — a re-quote at a previously-used price re-seeds
  ``queue_ahead`` at the back of the current queue (not a stale value), after a fill, after
  a reconcile-driven cancel (price move), and after a gap cancel.
* **FIX 2 — logical floor** — every cancel advances by at least ``max(0, Δ - back)``: full
  advance when ``back == 0`` (all models), zero for RiskAverse when ``back >= Δ``, exactly
  ``Δ - back`` for RiskAverse when ``0 < back < Δ``, bracket preserved.
* **FIX 3 — order invariance** — a trade and its same-timestamp depletion ``price_change``
  give identical fills AND identical final ``queue_ahead`` regardless of arrival order.
* **Trade-vs-cancel disambiguation**, re-anchor clamp, increase = no change, the power-law
  helper, the ``ProbQueue.f`` knob, protocol conformance, calibrate stub.

Because cancel attribution is **deferred** to the timestamp boundary (FIX 3), a unit test
that reads ``queue_ahead`` right after feeding a cancel ``price_change`` must first call
``flush_pending_cancels()`` (the stream-end finalize the engine also calls).
"""
from __future__ import annotations

import math
import random

import pytest

from mm_engine.engine import BACKTEST, run_engine
from mm_engine.events import GapMarker
from mm_engine.fees import FeeModel, FeeSchedule
from mm_engine.interfaces import BookState, MarketEvent, Order, QueueModel
from mm_engine.latency_models import ConstantLatency
from mm_engine.queue_models import (
    OptimisticQueue,
    ProbQueue,
    RiskAverseQueue,
    _prob_ahead_power,
)
from mm_engine.strategies import SymmetricQuoter

from mm_engine_fixtures import YES, book_msg, events, pc_msg, trade_msg


BASE = 1_781_000_000_000
PRICE = 0.47
PARAMS = {"half_spread": 0.01, "size": 100.0, "tick": 0.001}
FEE = FeeModel(market_schedules={YES: FeeSchedule(0.07, 0.20, source="market")})


def _models():
    """Fresh instance of each model, pessimism-ascending (default ProbQueue f=0.5)."""
    return {"opt": OptimisticQueue(), "prob": ProbQueue(), "ra": RiskAverseQueue()}


# --- direct-driver helpers (replicate the engine's on_event -> (trade) fill order) -------

def _book_state(bid_depth: float, ts: int) -> BookState:
    return BookState(YES, bids=((PRICE, bid_depth),), asks=((0.49, 100.0),), ts_exchange=ts, stale=False)


def _ev_book(ts: int) -> MarketEvent:
    return MarketEvent("book", YES, ts, "", 0, {})


def _ev_pc(ts: int, new_size: float) -> MarketEvent:
    return MarketEvent("price_change", YES, ts, "", 0,
                       {"asset_id": YES, "side": "BUY", "price": PRICE, "size": new_size,
                        "best_bid": PRICE, "best_ask": 0.49})


def _ev_trade(ts: int, size: float) -> MarketEvent:
    return MarketEvent("last_trade", YES, ts, "", 0, {"side": "SELL", "price": PRICE, "size": size})


def _seed(qm, order: Order, init_depth: float) -> None:
    """Cache a book and register the order at the back of ``init_depth`` (queue_ahead = depth)."""
    qm.on_event(_ev_book(0), _book_state(init_depth, 0))
    qm.get_queue_ahead(order)


def _drive(qm, order: Order, init_depth: float, steps: list[tuple[MarketEvent, BookState]]) -> float:
    """Seed, replay ``steps`` like the engine (on_event each; fill on trades), flush, return fills."""
    _seed(qm, order, init_depth)
    filled = 0.0
    remaining = float(order.size)
    for ev, book in steps:
        qm.on_event(ev, book)
        if ev.type == "last_trade" and remaining > 1e-9:
            res = qm.fill(order, book, ev)
            q = min(res.qty, remaining)
            remaining -= q
            filled += q
    qm.flush_pending_cancels()
    return filled


def _qa_at(rep, ts: int, side: str, price: float):
    """queue_ahead for the (side, price) order in the quote snapshot at ts_exchange == ts."""
    for q in rep.quotes:
        if q["ts_exchange"] == ts:
            for o in q["orders"]:
                if o["side"] == side and abs(o["price"] - price) < 1e-9:
                    return o["queue_ahead"]
    return None


# ======================================================================================
# 1. Bracketing — Optimistic >= Prob >= RiskAverse fills
# ======================================================================================

def test_bracket_strict_through_engine():
    """A crafted stream where the three models give a STRICTLY ordered fill quantity.

    Book seeds 50 ahead of our 0.47 bid; +40 joins behind us; two cancels (30 then 20) at
    0.47 with no trade; then a 60-lot SELL trades through. Same message stream + strategy for
    all three — only the queue model differs.
    """
    msgs = [
        book_msg(YES, BASE + 0, bids=[(PRICE, 50)], asks=[(0.49, 100)]),
        pc_msg(BASE + 50, [(YES, "BUY", PRICE, 90, PRICE, 0.49)]),   # +40 joins BEHIND us
        pc_msg(BASE + 100, [(YES, "BUY", PRICE, 60, PRICE, 0.49)]),  # cancel 30 (no trade)
        pc_msg(BASE + 150, [(YES, "BUY", PRICE, 40, PRICE, 0.49)]),  # cancel 20 (no trade)
        trade_msg(YES, BASE + 200, "SELL", PRICE, 60),               # trades through
    ]
    filled = {}
    for name, qm in _models().items():
        rep = run_engine(events(msgs), strategy=SymmetricQuoter(), queue_model=qm,
                         latency_model=ConstantLatency(), mode=BACKTEST, params=PARAMS, fee_model=FEE)
        filled[name] = rep.filled_qty

    # Optimistic clears all 50 ahead -> 60 overflow. RiskAverse advances only the logical
    # floor (10 on the 2nd cancel, where back=10 < Δ=20) -> 40 ahead absorbs 40 -> 20 overflow.
    assert filled["opt"] == pytest.approx(60.0)
    assert filled["ra"] == pytest.approx(20.0)
    assert filled["opt"] > filled["prob"] > filled["ra"]


@pytest.mark.parametrize("seed", range(20))
def test_bracket_random_seeds(seed):
    """Across random cancel/increase/trade streams, fills bracket as Opt >= Prob >= RA."""
    rng = random.Random(seed)
    init_depth = float(rng.randint(20, 60))
    order = Order(YES, "BUY", PRICE, float(rng.randint(5, 30)))

    steps: list[tuple[MarketEvent, BookState]] = []
    depth = init_depth
    ts = 1
    for _ in range(rng.randint(15, 40)):
        ts += rng.randint(1, 5)
        roll = rng.random()
        if roll < 0.45 and depth > 0:                      # cancel: shrink the level
            depth = max(0.0, depth - rng.randint(1, int(depth)))
            steps.append((_ev_pc(ts, depth), _book_state(depth, ts)))
        elif roll < 0.65:                                  # increase: liquidity joins behind us
            depth += rng.randint(1, 20)
            steps.append((_ev_pc(ts, depth), _book_state(depth, ts)))
        else:                                              # trade-through + coincident pc (same ts)
            tsize = float(rng.randint(1, 25))
            steps.append((_ev_trade(ts, tsize), _book_state(depth, ts)))
            depth = max(0.0, depth - tsize)
            steps.append((_ev_pc(ts, depth), _book_state(depth, ts)))

    f_opt = _drive(OptimisticQueue(), order, init_depth, list(steps))
    f_prob = _drive(ProbQueue(), order, init_depth, list(steps))
    f_ra = _drive(RiskAverseQueue(), order, init_depth, list(steps))

    tol = 1e-9
    assert f_opt + tol >= f_prob >= f_ra - tol, (seed, f_opt, f_prob, f_ra)


def test_bracket_random_is_not_vacuous():
    """Sanity: the bracket is not always an equality. With liquidity resting BEHIND us
    (so the logical floor does not force a full advance) and small cancels, Optimistic
    advances through the cancels and out-fills RiskAverse, which holds its place.

    (When ``back == 0`` the floor makes all models advance fully — that is FIX 2 working, not
    a model difference — so divergence requires ``back > 0``, which this stream guarantees.)
    """
    strict = 0
    for seed in range(20):
        rng = random.Random(3000 + seed)
        init_depth = float(rng.randint(36, 44))               # we join at the back of this
        order = Order(YES, "BUY", PRICE, 10_000.0)            # large -> never caps the fill
        steps: list[tuple[MarketEvent, BookState]] = []
        ts = 2
        behind = float(rng.randint(70, 100))                  # a big block joins BEHIND us
        level = init_depth + behind
        steps.append((_ev_pc(ts, level), _book_state(level, ts)))
        cuts = [float(rng.randint(4, 7)) for _ in range(6)]   # each cut < back -> RA floor stays 0
        total_cut = min(sum(cuts), init_depth - 4.0)          # keep Optimistic's queue_ahead >= 4
        remaining_cut = total_cut
        for cut in cuts:
            ts += 2
            cut = min(cut, remaining_cut)
            remaining_cut -= cut
            level = max(1.0, level - cut)
            steps.append((_ev_pc(ts, level), _book_state(level, ts)))
            if remaining_cut <= 0:
                break
        # Optimistic credits every cut ahead -> queue_ahead ~ init_depth - total_cut;
        # RiskAverse holds at init_depth. A trade between the two overflows Opt, not RA.
        opt_qa = init_depth - total_cut
        tsize = (opt_qa + init_depth) / 2.0
        ts += 2
        steps.append((_ev_trade(ts, tsize), _book_state(level, ts)))
        steps.append((_ev_pc(ts, max(0.0, level - tsize)), _book_state(max(0.0, level - tsize), ts)))

        f_opt = _drive(OptimisticQueue(), order, init_depth, list(steps))
        f_ra = _drive(RiskAverseQueue(), order, init_depth, list(steps))
        assert f_opt + 1e-9 >= f_ra
        if f_opt > f_ra + 1e-9:
            strict += 1
    assert strict >= 18                                       # divergence on essentially every stream


# ======================================================================================
# 2. FIX 1 — forget() teardown: a re-quote at a used price re-seeds at the back
# ======================================================================================

def test_forget_after_fill_reseeds_at_back():
    """place -> fill -> re-quote-same-price: the fresh order re-seeds at the current level
    depth, so a later small trade does NOT fill it (it would, with stale queue_ahead = 0)."""
    msgs = [
        book_msg(YES, BASE + 0, bids=[(PRICE, 50)], asks=[(0.49, 100)]),
        trade_msg(YES, BASE + 100, "SELL", PRICE, 200),   # fills our 100; order dropped+forgotten
        trade_msg(YES, BASE + 200, "SELL", PRICE, 30),    # hits the RE-QUOTED order (qa re-seeded 50)
    ]
    # Zero latency: this test isolates forget()/re-seed at ms resolution, not the latency gate.
    rep = run_engine(events(msgs), strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
                     latency_model=ConstantLatency(round_trip=0.0), mode=BACKTEST, params=PARAMS, fee_model=FEE)
    # Only the first trade fills. The re-quoted bid re-seeds at 50 (back of the current level),
    # so the 30-lot second trade cannot reach it. With the bug (stale 0) it would fill 30 more.
    assert rep.fill_count == 1
    assert rep.filled_qty == pytest.approx(100.0)
    assert _qa_at(rep, BASE + 100, "BUY", PRICE) == pytest.approx(50.0)   # re-seeded, not 0


def test_forget_after_cancel_reseeds_at_back():
    """place -> cancel (price move) -> re-quote-same-price: re-seeds at the current depth."""
    msgs = [
        book_msg(YES, BASE + 0, bids=[(PRICE, 50)], asks=[(0.49, 100)]),   # BUY 0.47 seeded 50
        book_msg(YES, BASE + 100, bids=[(0.50, 80)], asks=[(0.52, 100)]),  # mid moves -> BUY 0.47 cancelled
        book_msg(YES, BASE + 200, bids=[(PRICE, 90)], asks=[(0.49, 100)]), # mid back -> BUY 0.47 re-quoted
    ]
    rep = run_engine(events(msgs), strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
                     latency_model=ConstantLatency(), mode=BACKTEST, params=PARAMS, fee_model=FEE)
    # The re-quoted BUY 0.47 must re-seed at the CURRENT depth (90), not the stale 50.
    assert _qa_at(rep, BASE + 200, "BUY", PRICE) == pytest.approx(90.0)


def test_forget_after_gap_reseeds_at_back():
    """place -> gap (cancel_all) -> re-quote-same-price: re-seeds at the current depth."""
    feed = (
        events([book_msg(YES, BASE + 0, bids=[(PRICE, 50)], asks=[(0.49, 100)])])
        + [GapMarker("capture_gap")]
        + events([book_msg(YES, BASE + 100, bids=[(PRICE, 90)], asks=[(0.49, 100)])])
    )
    rep = run_engine(feed, strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
                     latency_model=ConstantLatency(), mode=BACKTEST, params=PARAMS, fee_model=FEE)
    assert _qa_at(rep, BASE + 100, "BUY", PRICE) == pytest.approx(90.0)   # re-seeded post-gap


# ======================================================================================
# 3. FIX 2 — shared logical floor on cancel attribution
# ======================================================================================

def _apply_cancel(qm, order: Order, init_depth: float, behind: float, cancel_to: float):
    """Seed at init_depth, add ``behind`` behind us (ts=10), cancel to ``cancel_to`` (ts=20), flush."""
    _seed(qm, order, init_depth)
    level = init_depth + behind
    if behind > 0:
        qm.on_event(_ev_pc(10, level), _book_state(level, 10))     # increase -> joins behind
    qm.on_event(_ev_pc(20, cancel_to), _book_state(cancel_to, 20)) # the cancel (deferred)
    qm.flush_pending_cancels()


def test_floor_back_zero_all_models_advance_fully():
    """back == 0 (we joined at the very back): a cancel MUST be ahead -> all models advance fully."""
    order = Order(YES, "BUY", PRICE, 10.0)
    qms = _models()
    for qm in qms.values():
        _apply_cancel(qm, order, init_depth=20.0, behind=0.0, cancel_to=12.0)  # cancel 8, back 0
    for qm in qms.values():
        assert qm.get_queue_ahead(order) == pytest.approx(12.0)               # 20 - 8 for ALL


def test_floor_back_ge_delta_riskaverse_advances_zero():
    """back >= Δ: RiskAverse advances 0 (the whole cancel could fit behind us)."""
    order = Order(YES, "BUY", PRICE, 10.0)
    ra = RiskAverseQueue()
    _apply_cancel(ra, order, init_depth=20.0, behind=30.0, cancel_to=44.0)    # level 50->44, cancel 6, back 30
    assert ra.get_queue_ahead(order) == pytest.approx(20.0)


def test_floor_partial_riskaverse_advances_delta_minus_back():
    """0 < back < Δ: RiskAverse advances exactly Δ - back; Opt full; Prob between."""
    order = Order(YES, "BUY", PRICE, 10.0)
    qms = _models()
    for qm in qms.values():
        _apply_cancel(qm, order, init_depth=20.0, behind=5.0, cancel_to=13.0)  # level 25->13, cancel 12, back 5
    assert qms["ra"].get_queue_ahead(order) == pytest.approx(13.0)             # 20 - (12 - 5) = 13
    assert qms["opt"].get_queue_ahead(order) == pytest.approx(8.0)            # 20 - 12
    # Prob: front=20, back=5 -> prob = 20^.5/(20^.5+5^.5); advance clamped up to floor 7
    p = math.sqrt(20) / (math.sqrt(20) + math.sqrt(5))
    expect = 20.0 - max(p * 12.0, 7.0)
    assert qms["prob"].get_queue_ahead(order) == pytest.approx(expect)
    assert (qms["opt"].get_queue_ahead(order)
            <= qms["prob"].get_queue_ahead(order)
            <= qms["ra"].get_queue_ahead(order))


# ======================================================================================
# 4. FIX 3 — order invariance of trade/cancel netting within a timestamp
# ======================================================================================

def _run_invariance(Model, order: Order, init_depth: float, behind: float,
                    trade_size: float, cancel_to: float, trade_first: bool):
    """Set up (seed + optional 'behind'), then a trade + its same-ts depletion pc in the
    given order. Returns (total_filled, final_queue_ahead)."""
    qm = Model()
    _seed(qm, order, init_depth)
    level = init_depth + behind
    if behind > 0:
        qm.on_event(_ev_pc(10, level), _book_state(level, 10))   # joins behind (separate ts)
    trade = _ev_trade(30, trade_size)
    pc = _ev_pc(30, cancel_to)                                   # same ts as the trade
    seq = [trade, pc] if trade_first else [pc, trade]
    filled = 0.0
    remaining = float(order.size)
    for ev in seq:
        qm.on_event(ev, _book_state(level, 30))
        if ev.type == "last_trade" and remaining > 1e-9:
            res = qm.fill(order, _book_state(level, 30), ev)
            q = min(res.qty, remaining)
            remaining -= q
            filled += q
    qm.flush_pending_cancels()
    return filled, qm.get_queue_ahead(order)


def test_order_invariance_trade_then_pc_equals_pc_then_trade():
    """A trade + its same-ts depletion price_change give identical fills AND final queue_ahead
    regardless of arrival order, for all three models."""
    # Scenario A: trade overflows and fills us (level fully depleted; no residual cancel).
    order = Order(YES, "BUY", PRICE, 100.0)
    for Model in (OptimisticQueue, ProbQueue, RiskAverseQueue):
        fa, qa = _run_invariance(Model, order, 20.0, 0.0, trade_size=30.0, cancel_to=0.0, trade_first=True)
        fb, qb = _run_invariance(Model, order, 20.0, 0.0, trade_size=30.0, cancel_to=0.0, trade_first=False)
        assert fa == pytest.approx(fb) == pytest.approx(10.0)   # overflow = 30 - 20
        assert qa == pytest.approx(qb) == pytest.approx(0.0)

    # Scenario B: no fill, a real residual cancel split per model (and back>0 so models differ).
    order_b = Order(YES, "BUY", PRICE, 100.0)
    finals = {}
    for name, Model in (("opt", OptimisticQueue), ("prob", ProbQueue), ("ra", RiskAverseQueue)):
        fa, qa = _run_invariance(Model, order_b, 30.0, 20.0, trade_size=10.0, cancel_to=28.0, trade_first=True)
        fb, qb = _run_invariance(Model, order_b, 30.0, 20.0, trade_size=10.0, cancel_to=28.0, trade_first=False)
        assert fa == pytest.approx(fb) == pytest.approx(0.0)    # 10-lot trade < 30 ahead, no fill
        assert qa == pytest.approx(qb)
        finals[name] = qa
    # level 50 -> 28 = trade(10) + cancel(12); after the trade front=20, back=20, floor=0
    assert finals["opt"] == pytest.approx(8.0)                  # 20 - 12
    assert finals["ra"] == pytest.approx(20.0)                  # cancel ignored (back >= Δ)
    assert finals["prob"] == pytest.approx(20.0 - 0.5 * 12.0)   # prob(20,20)=0.5 -> 14
    assert finals["opt"] < finals["prob"] < finals["ra"]        # models genuinely differ


# ======================================================================================
# 5. Trade-vs-cancel disambiguation (deferred netting; flush before reading)
# ======================================================================================

def _seed_with_back(qm, order: Order, init_depth: float, behind: float):
    _seed(qm, order, init_depth)
    level = init_depth + behind
    qm.on_event(_ev_pc(10, level), _book_state(level, 10))       # +behind joins behind us


def test_pure_cancel_advances_per_model():
    order = Order(YES, "BUY", PRICE, 10.0)
    qms = _models()
    for qm in qms.values():
        _seed_with_back(qm, order, init_depth=20.0, behind=10.0)   # level 30, front 20, back 10
        qm.on_event(_ev_pc(20, 24.0), _book_state(24.0, 20))       # cancel 6, no trade
        qm.flush_pending_cancels()
    assert qms["opt"].get_queue_ahead(order) == pytest.approx(14.0)     # 20 - 6 (all ahead)
    assert qms["ra"].get_queue_ahead(order) == pytest.approx(20.0)      # cancels ignored (back 10 >= 6)
    p = math.sqrt(20) / (math.sqrt(20) + math.sqrt(10))                # prob ahead
    assert qms["prob"].get_queue_ahead(order) == pytest.approx(20.0 - p * 6.0)
    assert (qms["opt"].get_queue_ahead(order)
            < qms["prob"].get_queue_ahead(order)
            < qms["ra"].get_queue_ahead(order))


def test_coincident_trade_not_double_counted_as_cancel():
    """A trade + its same-ts price_change: the trade advances the queue once (in fill); the
    price_change nets it out so it is not re-counted as a cancel."""
    order = Order(YES, "BUY", PRICE, 10.0)
    for qm in _models().values():
        _seed(qm, order, 20.0)
        trade = _ev_trade(30, 5.0)
        qm.on_event(trade, _book_state(20.0, 30))
        res = qm.fill(order, _book_state(20.0, 30), trade)
        assert res.qty == 0.0                                       # 5 < 20 ahead, no overflow
        qm.on_event(_ev_pc(30, 15.0), _book_state(15.0, 30))        # 20 -> 15, fully the trade
        qm.flush_pending_cancels()
        assert qm.get_queue_ahead(order) == pytest.approx(15.0)     # only the 5-lot trade advance


def test_mixed_trade_and_cancel_splits():
    """Same instant: a 5-lot trade plus a 3-lot cancel. The trade (5) advances once via fill;
    only the 3-lot cancel remainder is attributed by the model."""
    order = Order(YES, "BUY", PRICE, 10.0)
    qms = _models()
    for qm in qms.values():
        _seed_with_back(qm, order, init_depth=20.0, behind=10.0)    # level 30, front 20, back 10
        trade = _ev_trade(20, 5.0)
        qm.on_event(trade, _book_state(30.0, 20))
        assert qm.fill(order, _book_state(30.0, 20), trade).qty == 0.0
        qm.on_event(_ev_pc(20, 22.0), _book_state(22.0, 20))        # 30 -> 22 = trade(5)+cancel(3)
        qm.flush_pending_cancels()
    assert qms["opt"].get_queue_ahead(order) == pytest.approx(12.0)  # 15 - 3
    assert qms["ra"].get_queue_ahead(order) == pytest.approx(15.0)   # cancel ignored (back 10 >= 3)
    p = math.sqrt(15) / (math.sqrt(15) + math.sqrt(10))             # front=15 after trade, back=10
    assert qms["prob"].get_queue_ahead(order) == pytest.approx(15.0 - p * 3.0)


# ======================================================================================
# 6. Re-anchor clamp + increase = no change
# ======================================================================================

def test_book_reanchor_clamps_down_never_up():
    """A full ``book`` snapshot lowers queue_ahead to the observed depth but never raises it.

    RiskAverse keeps a belief (50) above the post-cancel truth because back (30) >= the cancel
    (20), so the snapshot must re-anchor it DOWN.
    """
    order = Order(YES, "BUY", PRICE, 10.0)
    ra = RiskAverseQueue()
    _seed_with_back(ra, order, init_depth=50.0, behind=30.0)        # level 80, front 50, back 30
    ra.on_event(_ev_pc(20, 60.0), _book_state(60.0, 20))            # cancel 20 (back 30 >= 20 -> floor 0)
    ra.flush_pending_cancels()
    assert ra.get_queue_ahead(order) == pytest.approx(50.0)         # RA ignored the cancel
    ra.on_event(_ev_book(30), _book_state(40.0, 30))               # snapshot truth: 40 resting
    assert ra.get_queue_ahead(order) == pytest.approx(40.0)         # clamped down
    ra.on_event(_ev_book(40), _book_state(70.0, 40))               # later deeper book
    assert ra.get_queue_ahead(order) == pytest.approx(40.0)         # NOT raised


def test_increase_is_no_change():
    order = Order(YES, "BUY", PRICE, 10.0)
    for qm in _models().values():
        _seed(qm, order, 10.0)
        qm.on_event(_ev_pc(5, 25.0), _book_state(25.0, 5))          # +15 behind us
        qm.flush_pending_cancels()
        assert qm.get_queue_ahead(order) == pytest.approx(10.0)


# ======================================================================================
# 7. Power-law helper, the f knob, protocol conformance, calibrate
# ======================================================================================

def test_prob_ahead_power_bounds_and_form():
    assert _prob_ahead_power(0.0, 5.0, 0.5) == 0.0          # nothing ahead -> can't advance
    assert _prob_ahead_power(5.0, 0.0, 0.5) == 1.0          # nothing behind -> cancel is ahead
    assert _prob_ahead_power(10.0, 10.0, 0.5) == pytest.approx(0.5)   # symmetric -> half
    assert _prob_ahead_power(30.0, 10.0, 1.0) == pytest.approx(0.75)  # f=1 -> front/(front+back)
    for fb in [(3.0, 7.0), (12.0, 1.0), (1.0, 50.0)]:
        assert 0.0 <= _prob_ahead_power(*fb, 0.5) <= 1.0


def test_prob_ahead_power_no_overflow_at_extreme_f_on_deep_books():
    """Scale-free computation: large f over a deep book stays finite and in [0,1]
    (front**n is never formed, so no OverflowError / silent inf -> 0.0)."""
    assert _prob_ahead_power(1202.0, 1202.0, 100.0) == pytest.approx(0.5)   # was 0.0 pre-fix
    assert _prob_ahead_power(5000.0, 10.0, 100.0) == pytest.approx(1.0)     # front >> back
    assert _prob_ahead_power(10.0, 5000.0, 100.0) == pytest.approx(0.0)     # back >> front
    for n in (50.0, 100.0):
        for front, back in [(5000.0, 4000.0), (1e6, 1.0), (1.0, 1e6), (1500.0, 1500.0)]:
            p = _prob_ahead_power(front, back, n)
            assert 0.0 <= p <= 1.0
    # the bracket still holds with an extreme f on a deep, behind-heavy level
    order = Order(YES, "BUY", PRICE, 100.0)
    qms = {"opt": OptimisticQueue(), "prob": ProbQueue(f=100.0), "ra": RiskAverseQueue()}
    for qm in qms.values():
        _apply_cancel(qm, order, init_depth=1500.0, behind=1500.0, cancel_to=2400.0)  # cancel 600
    assert (qms["opt"].get_queue_ahead(order)
            <= qms["prob"].get_queue_ahead(order)
            <= qms["ra"].get_queue_ahead(order))


def test_prob_queue_f_param():
    assert ProbQueue().f == 0.5
    assert ProbQueue(0.8).f == 0.8
    assert ProbQueue(f=1.0).f == 1.0
    for bad in (0.0, -0.5, -1.0):
        with pytest.raises(ValueError):
            ProbQueue(f=bad)                                # power-law exponent must be > 0


def test_protocol_conformance():
    assert isinstance(OptimisticQueue(), QueueModel)
    assert isinstance(RiskAverseQueue(), QueueModel)
    assert isinstance(ProbQueue(), QueueModel)


def test_calibrate_is_a_noop_stub():
    order = Order(YES, "BUY", PRICE, 10.0)
    for qm in _models().values():
        _seed(qm, order, 20.0)
        assert qm.calibrate([{"price": PRICE, "qty": 1.0}]) is None
        assert qm.get_queue_ahead(order) == 20.0
