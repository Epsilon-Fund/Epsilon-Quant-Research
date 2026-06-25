"""Phase 1 tests — the engine machine (book builder, order manager, fill sim, telemetry,
reconciliation) running against the STUB models, in replay and live-shadow."""
from __future__ import annotations

import json

import pytest

from mm_engine.engine import BACKTEST, LIVE_SHADOW, run_engine
from mm_engine.events import GapMarker
from mm_engine.feeds.live_shadow import FrameTransport, live_shadow_feed
from mm_engine.feeds.replay import replay_feed
from mm_engine.fees import FeeModel, FeeSchedule
from mm_engine.latency_models import ConstantLatency
from mm_engine.orders import OrderManager
from mm_engine.queue_models import OptimisticQueue
from mm_engine.reconcile import run_and_reconcile
from mm_engine.strategies import SymmetricQuoter
from mm_engine.telemetry import Telemetry

from mm_engine_fixtures import (
    YES,
    bba_msg,
    book_msg,
    events,
    frames,
    pc_msg,
    trade_msg,
    write_replay_jsonl,
)


BASE = 1_781_000_000_000
PARAMS = {"half_spread": 0.01, "size": 100.0, "tick": 0.001}
# Explicit per-market Crypto schedule for YES (rate 0.07 / rebate 0.20) — deterministic and
# avoids the category-fallback lazy import in these tests.
FEE = FeeModel(market_schedules={YES: FeeSchedule(0.07, 0.20, source="market")})


def _fill_scenario():
    # Small resting size (50) ahead of our 0.47 bid; a 200-lot SELL trades through -> we fill.
    return [
        book_msg(YES, BASE + 0, bids=[(0.47, 50), (0.46, 300)], asks=[(0.49, 400), (0.50, 600)]),
        trade_msg(YES, BASE + 200, "SELL", 0.47, 200),
    ]


# ----------------------------------------------------------------------------------------
# end-to-end: replay + live-shadow
# ----------------------------------------------------------------------------------------

def test_engine_runs_replay_and_live_shadow(tmp_path):
    msgs = _fill_scenario()
    path = write_replay_jsonl(tmp_path / "shard.jsonl", msgs)

    rep = run_engine(replay_feed(path), strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
                     latency_model=ConstantLatency(), mode=BACKTEST, params=PARAMS, fee_model=FEE)
    shadow = run_engine(
        live_shadow_feed([YES], connect=lambda *_: FrameTransport(frames(msgs))),
        strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
        latency_model=ConstantLatency(), mode=LIVE_SHADOW, params=PARAMS, fee_model=FEE,
    )

    assert rep.event_count == shadow.event_count == 2
    assert rep.quote_count == 2 and rep.placed_count >= 2
    assert rep.quotes and rep.orders
    # backtest realizes the fill; live-shadow logs orders only (no fills)
    assert rep.fill_count == 1 and rep.filled_qty == 100.0
    assert shadow.fill_count == 0 and shadow.position == {}


def test_fill_simulator_fills_on_trade_through():
    rep = run_engine(events(_fill_scenario()), strategy=SymmetricQuoter(),
                     queue_model=OptimisticQueue(), latency_model=ConstantLatency(),
                     mode=BACKTEST, params=PARAMS, fee_model=FEE)
    assert rep.fill_count == 1
    fill = rep.fills[0]
    assert fill["side"] == "BUY" and fill["price"] == 0.47 and fill["qty"] == 100.0
    assert fill["queue_ahead"] == 0.0            # 50 ahead cleared by the 200-lot trade
    assert fill["mid_at_fill"] == 0.48
    assert fill["position_after"] == 100.0 and fill["gross_cash_after"] == -47.0
    assert fill["maker_fee"] == 0.0              # passive maker pays no fee
    assert fill["maker_rebate"] == pytest.approx(0.20 * 0.07 * 100 * 0.47 * 0.53)
    assert rep.position[YES] == 100.0
    # PnL split: 0 realized (no offset), 1.0 unrealized (100 * (0.48-0.47))
    assert rep.realized_pnl == 0.0
    assert rep.unrealized_pnl == pytest.approx(1.0)
    assert rep.gross_pnl == pytest.approx(1.0)
    assert rep.net_ex_rebate == pytest.approx(1.0)
    assert rep.net_with_rebate == pytest.approx(1.0 + 0.20 * 0.07 * 100 * 0.47 * 0.53)


def test_latency_gate_blocks_too_fast_trade():
    rep = run_engine(events(_fill_scenario()), strategy=SymmetricQuoter(),
                     queue_model=OptimisticQueue(),
                     latency_model=ConstantLatency(round_trip=10_000_000.0),
                     mode=BACKTEST, params=PARAMS, fee_model=FEE)
    assert rep.fill_count == 0


# ----------------------------------------------------------------------------------------
# order manager
# ----------------------------------------------------------------------------------------

def test_order_manager_idempotent_replace_throttle():
    from mm_engine.interfaces import Order
    om = OrderManager(throttle_ms=1000)

    ops0 = om.reconcile([Order(YES, "BUY", 0.47, 100.0)], ts=0)
    assert [o.op for o in ops0] == ["place"]
    assert om.reconcile([Order(YES, "BUY", 0.47, 100.0)], ts=100) == []  # idempotent
    ops2 = om.reconcile([Order(YES, "BUY", 0.48, 100.0)], ts=200)
    assert [o.op for o in ops2] == ["throttled"]
    assert om.active_orders()[0].order.price == 0.47
    ops3 = om.reconcile([Order(YES, "BUY", 0.48, 100.0)], ts=1500)
    assert [o.op for o in ops3] == ["replace"]
    assert om.active_orders()[0].order.price == 0.48
    ops4 = om.reconcile([], ts=1600)
    assert [o.op for o in ops4] == ["cancel"] and om.active_orders() == []


def test_gap_cancels_resting_orders():
    feed = events([book_msg(YES, BASE, [(0.47, 50)], [(0.49, 400)])]) + [GapMarker("capture_gap")]
    rep = run_engine(feed, strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
                     latency_model=ConstantLatency(), mode=BACKTEST, params=PARAMS, fee_model=FEE)
    ops = [o["op"] for o in rep.orders]
    assert ops.count("place") == 2 and ops.count("cancel") == 2  # BUY+SELL placed then gap-cancelled


# ----------------------------------------------------------------------------------------
# L1 cross-check
# ----------------------------------------------------------------------------------------

def test_l1_crosscheck_reports_match_fraction():
    msgs = [
        book_msg(YES, BASE + 0, [(0.47, 500)], [(0.49, 400)]),
        bba_msg(YES, BASE + 100, 0.47, 0.49),   # matches reconstructed top
        bba_msg(YES, BASE + 200, 0.50, 0.49),   # claims a bid we don't have -> bid mismatch
    ]
    rep = run_engine(events(msgs), strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
                     latency_model=ConstantLatency(), mode=BACKTEST, params=PARAMS, fee_model=FEE)
    cc = rep.l1_crosscheck
    assert cc["compared"] == 2
    assert cc["both_match_frac"] == 0.5
    assert cc["ask_match_frac"] == 1.0


# ----------------------------------------------------------------------------------------
# reconciliation (Join 1) — replay vs live-shadow over the same frames
# ----------------------------------------------------------------------------------------

def test_reconcile_replay_vs_live_shadow_is_clean(tmp_path):
    msgs = _fill_scenario()
    path = write_replay_jsonl(tmp_path / "shard.jsonl", msgs)

    report = run_and_reconcile(
        lambda: replay_feed(path),
        lambda: live_shadow_feed([YES], connect=lambda *_: FrameTransport(frames(msgs))),
        strategy_factory=SymmetricQuoter,
        queue_factory=OptimisticQueue,
        latency_factory=ConstantLatency,
        fee_model_factory=lambda: FEE,
        params=PARAMS,
        tolerance=0.05,
    )
    assert report.passed
    assert report.equity_path_match
    assert all(m.gap == 0.0 for m in report.metrics)
    fc = next(m for m in report.metrics if m.name == "fill_count")
    assert fc.a == 1.0 and fc.b == 1.0
    assert "PASS" in report.render()


# ----------------------------------------------------------------------------------------
# telemetry to disk
# ----------------------------------------------------------------------------------------

def test_telemetry_writes_raw_jsonl(tmp_path):
    tele = Telemetry.to_dir(tmp_path, prefix="rep")
    run_engine(events(_fill_scenario()), strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
               latency_model=ConstantLatency(), mode=BACKTEST, params=PARAMS, fee_model=FEE, telemetry=tele)
    tele.close()

    fills_file = tmp_path / "rep_fills.jsonl"
    orders_file = tmp_path / "rep_orders.jsonl"
    quotes_file = tmp_path / "rep_quotes.jsonl"
    assert fills_file.exists() and orders_file.exists() and quotes_file.exists()
    fill_rows = [json.loads(line) for line in fills_file.read_text().splitlines() if line.strip()]
    assert len(fill_rows) == 1 and fill_rows[0]["qty"] == 100.0
    assert fill_rows[0]["maker_rebate"] > 0.0 and fill_rows[0]["fee_source"] == "market"
    quote_rows = [json.loads(line) for line in quotes_file.read_text().splitlines() if line.strip()]
    assert any("orders" in q for q in quote_rows)
