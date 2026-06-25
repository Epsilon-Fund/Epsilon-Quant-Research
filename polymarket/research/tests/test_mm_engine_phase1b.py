"""Phase 1 realism refinements — fee/maker-rebate model, PnL split + settlement, and the
record->replay reconciliation. All against the FROZEN interface."""
from __future__ import annotations

import json

import pytest

from mm_engine.engine import BACKTEST, run_engine
from mm_engine.feeds.live_shadow import FrameTransport, live_shadow_feed
from mm_engine.fees import FEE_FREE, FeeModel, FeeSchedule
from mm_engine.latency_models import ConstantLatency
from mm_engine.queue_models import OptimisticQueue
from mm_engine.reconcile import reconcile_against_recording
from mm_engine.strategies import SymmetricQuoter

from mm_engine_fixtures import YES, book_msg, events, frames, trade_msg, write_replay_jsonl

BASE = 1_781_000_000_000
PARAMS = {"half_spread": 0.01, "size": 100.0, "tick": 0.001}
FEE = FeeModel(market_schedules={YES: FeeSchedule(0.07, 0.20, source="market")})


def _open_inventory_scenario():
    # One passive BUY fill of 100 @ 0.47, left open (no offsetting trade).
    return [
        book_msg(YES, BASE + 0, [(0.47, 50)], [(0.49, 400)]),
        trade_msg(YES, BASE + 200, "SELL", 0.47, 200),
    ]


def _round_trip_scenario():
    # BUY 100 @ 0.47 then SELL 100 @ 0.49 -> realized round-trip (+0.02/contract).
    return [
        book_msg(YES, BASE + 0, [(0.47, 10)], [(0.49, 10)]),
        trade_msg(YES, BASE + 200, "SELL", 0.47, 200),   # fills our resting BUY 0.47
        trade_msg(YES, BASE + 400, "BUY", 0.49, 200),    # fills our resting SELL 0.49
    ]


# ----------------------------------------------------------------------------------------
# fee + maker-rebate model
# ----------------------------------------------------------------------------------------

def test_fee_schedule_formula():
    s = FeeSchedule(fee_rate=0.07, rebate_rate=0.20)
    # fee = fee_rate * qty * p * (1-p); peak at p=0.5 is 1.75c/share -> $1.75 on 100
    assert s.taker_fee(100, 0.5) == pytest.approx(1.75)
    assert s.maker_rebate(100, 0.5) == pytest.approx(0.35)        # 0.20 * 1.75
    assert s.taker_fee(100, 0.0) == 0.0 and s.taker_fee(100, 1.0) == 0.0
    assert FEE_FREE.taker_fee(100, 0.5) == 0.0 and FEE_FREE.maker_rebate(100, 0.5) == 0.0


def test_fee_field_parse_and_resolution_order():
    # per-market 'fee'/'fd' field preferred
    sched = FeeSchedule.from_fee_field({"fees_enabled": True, "rate": 0.07, "rebateRate": 0.2})
    assert sched.fee_rate == 0.07 and sched.rebate_rate == 0.2 and sched.source == "market"

    m = FeeModel(market_schedules={YES: sched})
    assert m.schedule_for(YES).source == "market"

    # fall back to canonical FEE_BY_CATEGORY (lazy import) for unknown tokens
    cat = FeeModel(token_category={YES: "Crypto"})
    cs = cat.schedule_for(YES)
    assert cs.fee_rate == pytest.approx(0.07) and cs.rebate_rate == pytest.approx(0.20)
    assert cs.source == "category:Crypto"

    # fee_free override
    assert FeeModel.fee_free_model().schedule_for(YES) is FEE_FREE


def test_maker_rebate_credited_and_three_pnl_views():
    rep = run_engine(events(_open_inventory_scenario()), strategy=SymmetricQuoter(),
                     queue_model=OptimisticQueue(), latency_model=ConstantLatency(),
                     mode=BACKTEST, params=PARAMS, fee_model=FEE)
    expected_rebate = 0.20 * 0.07 * 100 * 0.47 * 0.53
    assert rep.rebates_earned == pytest.approx(expected_rebate)
    assert rep.taker_fees_paid == 0.0
    assert rep.gross_pnl == pytest.approx(1.0)                       # unrealized mark-to-mid
    assert rep.net_ex_rebate == pytest.approx(1.0)                   # net_without_rebate discipline
    assert rep.net_with_rebate == pytest.approx(1.0 + expected_rebate)


def test_fee_free_credits_no_rebate():
    rep = run_engine(events(_open_inventory_scenario()), strategy=SymmetricQuoter(),
                     queue_model=OptimisticQueue(), latency_model=ConstantLatency(),
                     mode=BACKTEST, params=PARAMS, fee_model=FeeModel.fee_free_model())
    assert rep.rebates_earned == 0.0
    assert rep.net_with_rebate == pytest.approx(rep.net_ex_rebate)


# ----------------------------------------------------------------------------------------
# PnL split: realized vs unrealized
# ----------------------------------------------------------------------------------------

def test_pnl_split_realized_vs_unrealized():
    rep = run_engine(events(_round_trip_scenario()), strategy=SymmetricQuoter(),
                     queue_model=OptimisticQueue(), latency_model=ConstantLatency(),
                     mode=BACKTEST, params=PARAMS, fee_model=FEE)
    assert rep.fill_count == 2
    # bought 100 @ 0.47, sold 100 @ 0.49 -> realized +2.0, flat -> no open inventory
    assert rep.realized_pnl == pytest.approx(2.0)
    assert rep.open_positions == {}
    assert rep.unrealized_pnl == pytest.approx(0.0)
    assert rep.gross_pnl == pytest.approx(2.0)
    assert abs(rep.position.get(YES, 0.0)) < 1e-9


# ----------------------------------------------------------------------------------------
# settlement hook
# ----------------------------------------------------------------------------------------

def test_settlement_hook_settles_open_inventory():
    rep = run_engine(events(_open_inventory_scenario()), strategy=SymmetricQuoter(),
                     queue_model=OptimisticQueue(), latency_model=ConstantLatency(),
                     mode=BACKTEST, params=PARAMS, fee_model=FEE)
    assert rep.open_positions[YES][0] == pytest.approx(100.0)  # (qty, cost_basis)

    # YES resolves YES (payoff 1.0): settled = 100 * (1.0 - 0.47) = 53.0
    won = rep.settle({YES: 1.0})
    assert won.settled_pnl == pytest.approx(53.0)
    assert won.unrealized_pnl == 0.0 and won.settled_tokens == (YES,)
    assert won.gross_pnl == pytest.approx(53.0)
    assert won.net_with_rebate == pytest.approx(53.0 + rep.rebates_earned)

    # YES resolves NO (payoff 0.0): settled = 100 * (0.0 - 0.47) = -47.0
    lost = rep.settle({YES: 0.0})
    assert lost.settled_pnl == pytest.approx(-47.0)

    # absent from the resolution map -> stays flagged unrealized (marked to mid 0.48)
    none = rep.settle({})
    assert none.settled_pnl == 0.0
    assert none.unrealized_pnl == pytest.approx(1.0) and none.unsettled_tokens == (YES,)


def test_settle_open_token_without_mid_is_entry_marked():
    # An open, unmapped token with NO recorded mid must stay in the unsettled partition AND
    # contribute (entry-marked -> 0), not be silently dropped (partition/sum consistency).
    rep = run_engine(events(_open_inventory_scenario()), strategy=SymmetricQuoter(),
                     queue_model=OptimisticQueue(), latency_model=ConstantLatency(),
                     mode=BACKTEST, params=PARAMS, fee_model=FEE)
    rep.last_mid.clear()                      # simulate a token that never had a two-sided mid
    s = rep.settle({})
    assert s.unsettled_tokens == (YES,)       # still in the partition
    assert s.unrealized_pnl == pytest.approx(0.0)   # entry-marked, not dropped
    assert s.gross_pnl == pytest.approx(s.realized_pnl + s.settled_pnl + s.unrealized_pnl)


# ----------------------------------------------------------------------------------------
# record -> replay reconciliation (real same-code-path / divergence test)
# ----------------------------------------------------------------------------------------

def test_record_replay_reconciliation(tmp_path):
    msgs = _open_inventory_scenario()
    rec_path = tmp_path / "session.jsonl"

    # Run a (shadow-sourced) live session in BACKTEST mode, recording its own frames.
    live = run_engine(
        live_shadow_feed([YES], connect=lambda *_: FrameTransport(frames(msgs)), record_to=rec_path),
        strategy=SymmetricQuoter(), queue_model=OptimisticQueue(),
        latency_model=ConstantLatency(), mode=BACKTEST, params=PARAMS, fee_model=FEE,
    )
    assert live.fill_count == 1                     # non-trivial: the session actually filled
    assert rec_path.exists()
    recorded = [json.loads(l) for l in rec_path.read_text().splitlines() if l.strip()]
    assert len(recorded) == 2                       # book + trade envelopes recorded

    # Replay the session's OWN recording and reconcile — must be byte-identical (0 gap).
    report = reconcile_against_recording(
        live, rec_path,
        strategy_factory=SymmetricQuoter, queue_factory=OptimisticQueue,
        latency_factory=ConstantLatency, fee_model_factory=lambda: FEE,
        params=PARAMS, mode=BACKTEST, tolerance=0.05,
    )
    assert report.passed and report.equity_path_match
    assert all(m.gap == 0.0 for m in report.metrics)
