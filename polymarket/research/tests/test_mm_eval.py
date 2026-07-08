"""Tests for the mm_eval validation layer — markout signs, CIs, scorecard, breakeven, stability.

Pure-function tests on synthetic telemetry (no engine run), plus the breakeven/verdict logic and
the dormant-overfitting guard. Run: ``PYTHONPATH=. uv run pytest tests/test_mm_eval.py``.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from mm_eval.metrics import (CI, block_bootstrap_mean_ci, breakeven_read, build_scorecard,
                             compute_markout, markout_curve, mid_trajectory, verdict_from_bracket,
                             MarkoutCurvePoint)
from mm_eval.stability import temporal_stability
from mm_eval.overfitting_hook import dormant_status, build_oos_split


def _quote(ts, mid, orders=2, stale=False):
    return {"ts_exchange": ts, "token_id": "t", "event_type": "price_change",
            "stale": stale, "best_bid": None, "best_ask": None, "mid": mid,
            "orders": [{} for _ in range(orders)]}


def _fill(ts, side, price, qty, mid_at_fill, position_after, realized_delta=0.0):
    return {"ts_exchange": ts, "token_id": "t", "side": side, "price": price, "qty": qty,
            "queue_ahead": 0.0, "mid_at_fill": mid_at_fill, "maker_rebate": 0.0,
            "realized_delta": realized_delta, "position_after": position_after,
            "trade_price": price, "trade_size": qty}


# ── markout signs + censoring ────────────────────────────────────────────────

def test_markout_signs_buy_favorable_and_adverse():
    # mid trajectory: 0s .. 60s rising 0.46 -> 0.50
    quotes = [_quote(0, 0.46), _quote(5_000, 0.50), _quote(60_000, 0.50)]
    fills = [_fill(0, "BUY", 0.45, 10.0, 0.46, 10.0)]
    mr = compute_markout(fills, quotes, horizons=(5,))
    # BUY filled at 0.45, mid rose to 0.50 -> markout to fill = +0.05; drift = +0.04 (favorable)
    assert mr.markout_to_fill[5][0] == pytest.approx(0.05)
    assert mr.adverse_drift[5][0] == pytest.approx(0.04)


def test_markout_signs_sell_favorable():
    quotes = [_quote(0, 0.54), _quote(5_000, 0.50)]
    fills = [_fill(0, "SELL", 0.55, 10.0, 0.54, -10.0)]
    mr = compute_markout(fills, quotes, horizons=(5,))
    # SELL at 0.55, mid fell to 0.50 -> markout to fill = -1*(0.50-0.55)=+0.05; drift=-1*(0.50-0.54)=+0.04
    assert mr.markout_to_fill[5][0] == pytest.approx(0.05)
    assert mr.adverse_drift[5][0] == pytest.approx(0.04)


def test_markout_adverse_when_price_runs_through():
    # BUY then mid falls (we got picked off): adverse
    quotes = [_quote(0, 0.46), _quote(5_000, 0.40)]
    fills = [_fill(0, "BUY", 0.45, 10.0, 0.46, 10.0)]
    mr = compute_markout(fills, quotes, horizons=(5,))
    assert mr.markout_to_fill[5][0] == pytest.approx(-0.05)   # 0.40 - 0.45
    assert mr.adverse_drift[5][0] == pytest.approx(-0.06)     # 0.40 - 0.46 (adverse)


def test_markout_censored_past_data_edge():
    quotes = [_quote(0, 0.46), _quote(5_000, 0.50)]   # last event at 5s
    fills = [_fill(0, "BUY", 0.45, 10.0, 0.46, 10.0)]
    mr = compute_markout(fills, quotes, horizons=(1, 60))  # 60s is past the 5s data edge
    assert np.isfinite(mr.markout_to_fill[1][0])
    assert math.isnan(mr.markout_to_fill[60][0])
    assert mr.censored[60] == 1 and mr.censored[1] == 0


def test_markout_curve_units_are_cents_and_adverse_rate():
    quotes = [_quote(0, 0.46), _quote(5_000, 0.40), _quote(60_000, 0.40)]
    fills = [_fill(0, "BUY", 0.45, 10.0, 0.46, 10.0)]
    mr = compute_markout(fills, quotes, horizons=(5,))
    cp = markout_curve(mr, n_boot=200)[0]
    assert cp.markout_to_fill_cents.point == pytest.approx(-5.0)   # -0.05 * 100
    assert cp.adverse_rate == pytest.approx(1.0)                    # the single fill is adverse


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def test_block_bootstrap_point_is_weighted_mean_and_brackets():
    vals = np.array([1.0, 1.0, 1.0, 3.0])
    wts = np.array([1.0, 1.0, 1.0, 1.0])
    ci = block_bootstrap_mean_ci(vals, wts, n_boot=1000, seed=1)
    assert ci.point == pytest.approx(1.5)
    assert ci.lo <= ci.point <= ci.hi
    # weighted: heavy weight on the 3.0 raises the point
    ci_w = block_bootstrap_mean_ci(vals, np.array([1, 1, 1, 100.0]), n_boot=500, seed=1)
    assert ci_w.point > 2.5


def test_ci_clears_zero():
    assert CI(0.5, 0.1, 0.9, 10).clears_zero
    assert not CI(0.5, -0.1, 0.9, 10).clears_zero
    assert not CI(0.5, float("nan"), 0.9, 10).clears_zero


# ── breakeven + verdict ──────────────────────────────────────────────────────

def _cp(markout_point, lo, hi, drift_point=None):
    drift_point = markout_point if drift_point is None else drift_point
    return MarkoutCurvePoint(
        horizon_s=30,
        markout_to_fill_cents=CI(markout_point, lo, hi, 50),
        adverse_drift_cents=CI(drift_point, drift_point - 0.5, drift_point + 0.5, 50),
        adverse_rate=0.5, n_fills=50, n_censored=0,
    )


def test_breakeven_clears_when_net_edge_lower_ci_positive():
    cp = _cp(1.0, 0.4, 1.6)  # markout +1c, lower CI +0.4c
    be = breakeven_read(cp, queue="RiskAverse", half_spread=0.01, rebate_per_contract=0.0,
                        fee_mode="no_rebate")
    assert be.net_edge_cents.point == pytest.approx(1.0)
    assert be.clears  # lower CI 0.4 > 0


def test_breakeven_does_not_clear_when_lower_ci_negative():
    cp = _cp(0.5, -0.3, 1.3)
    be = breakeven_read(cp, queue="Optimistic", half_spread=0.01, rebate_per_contract=0.0,
                        fee_mode="no_rebate")
    assert not be.clears


def test_breakeven_rebate_shifts_edge_up():
    cp = _cp(-0.5, -0.9, -0.1)
    be_no = breakeven_read(cp, queue="Optimistic", half_spread=0.01, rebate_per_contract=0.0,
                           fee_mode="no_rebate")
    be_rep = breakeven_read(cp, queue="Optimistic", half_spread=0.01, rebate_per_contract=0.02,
                            fee_mode="representative")
    assert be_rep.net_edge_cents.point == pytest.approx(be_no.net_edge_cents.point + 2.0)  # +0.02*100
    assert be_rep.rebate_cents == pytest.approx(2.0)


def test_verdict_bracket():
    viable_pess = breakeven_read(_cp(1.0, 0.5, 1.5), queue="RiskAverse", half_spread=0.01,
                                 rebate_per_contract=0.0, fee_mode="no_rebate")
    viable_opt = breakeven_read(_cp(1.2, 0.7, 1.7), queue="Optimistic", half_spread=0.01,
                                rebate_per_contract=0.0, fee_mode="no_rebate")
    assert verdict_from_bracket(viable_opt, viable_pess) == "VIABLE"

    dead_pess = breakeven_read(_cp(-1.0, -1.5, -0.5), queue="RiskAverse", half_spread=0.01,
                               rebate_per_contract=0.0, fee_mode="no_rebate")
    frag_opt = breakeven_read(_cp(1.2, 0.7, 1.7), queue="Optimistic", half_spread=0.01,
                              rebate_per_contract=0.0, fee_mode="no_rebate")
    assert verdict_from_bracket(frag_opt, dead_pess) == "FRAGILE"

    dead_opt = breakeven_read(_cp(-0.5, -1.0, -0.1), queue="Optimistic", half_spread=0.01,
                              rebate_per_contract=0.0, fee_mode="no_rebate")
    assert verdict_from_bracket(dead_opt, dead_pess) == "DEAD"


# ── scorecard ────────────────────────────────────────────────────────────────

def _fake_result():
    fills = [
        _fill(0, "BUY", 0.45, 10.0, 0.46, 10.0, realized_delta=0.0),
        _fill(10_000, "SELL", 0.47, 10.0, 0.46, 0.0, realized_delta=0.2),  # closed +0.2
    ]
    quotes = [_quote(0, 0.46), _quote(10_000, 0.46, orders=2), _quote(20_000, 0.46, orders=1, stale=True)]
    equity = [(0, 0.0), (10_000, 0.2), (20_000, 0.1)]
    return SimpleNamespace(
        fills=fills, quotes=quotes, equity_path=equity,
        realized_pnl=0.2, unrealized_pnl=0.0, gross_pnl=0.2,
        net_ex_rebate=0.2, net_with_rebate=0.25, rebates_earned=0.05, filled_qty=20.0,
        fill_count=2, placed_count=4, position={"t": 0.0}, taker_fees_paid=0.0,
        l1_crosscheck={"both_match_frac": 0.99},
    )


def test_scorecard_basic_fields():
    sc = build_scorecard(_fake_result(), days=1.0)
    assert sc.fills == 2 and sc.placed == 4
    assert sc.fill_rate == pytest.approx(0.5)
    assert sc.net_ex_rebate == pytest.approx(0.2)
    assert sc.net_with_rebate == pytest.approx(0.25)
    assert sc.rebate_per_contract == pytest.approx(0.05 / 20.0)
    assert sc.max_inventory == pytest.approx(10.0)
    assert sc.profit_ratio == pytest.approx(float("inf"))  # only a gain, no realized loss
    assert 0.0 <= sc.quote_uptime_two_sided <= 1.0
    assert sc.stale_share == pytest.approx(1 / 3)
    assert sc.max_drawdown_usd == pytest.approx(0.1)   # peak 0.2 -> trough 0.1


# ── temporal stability ───────────────────────────────────────────────────────

def test_temporal_stability_detects_concentration():
    # window 1 (0..50s): favorable fills; window 2 (100..150s): adverse fills of equal weight
    quotes = [_quote(t, 0.50) for t in range(0, 200_001, 1_000)]
    fills = []
    for t in range(0, 51_000, 10_000):     # favorable: BUY then mid up at +30s
        fills.append(_fill(t, "BUY", 0.49, 10.0, 0.50, 10.0))
    for t in range(100_000, 151_000, 10_000):  # adverse: BUY then mid down at +30s
        fills.append(_fill(t, "BUY", 0.51, 10.0, 0.50, 10.0))
    # make mid drop after 100s so the late fills are adverse
    quotes = [_quote(t, 0.50 if t < 130_000 else 0.40) for t in range(0, 200_001, 1_000)]
    st = temporal_stability(fills, quotes, horizon_s=30, n_blocks=8)
    assert st.n_nonempty_blocks >= 2
    assert math.isfinite(st.pooled_edge_cents)


# ── dormant overfitting ──────────────────────────────────────────────────────

def test_dormant_overfitting_haircut_is_zero():
    d = dormant_status(np.array([0.1, -0.2, 0.05, 0.0, 0.3]))
    assert d.wired
    assert d.n_trials == 1
    assert d.sr_star_haircut == pytest.approx(0.0)   # nothing to deflate on 1 trial
    assert not d.pbo_available
    assert d.oos_split_available


def test_oos_split_apparatus_builds():
    ts = np.array([0, 100, 200, 300, 400], dtype=np.int64)
    sp = build_oos_split(ts, is_frac=0.7)
    assert sp.n_is + sp.n_oos == 5
    assert sp.split_ts == 280
