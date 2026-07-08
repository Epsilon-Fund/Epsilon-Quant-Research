"""Task-5 unit tests — InventoryAwareQuoter / ASQuoter / BasketCarryQuoter + the IS/OOS protocol.

Run from polymarket/research/:  PYTHONPATH=. uv run pytest tests/test_mm_task5.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mm_engine.interfaces import BookState
from mm_engine.strategies import (ASQuoter, BasketCarryQuoter, InventoryAwareQuoter,
                                  SymmetricQuoter, microprice, tau_hours)
from mm_eval import protocol as pr


def book(ts=1_000_000, bid=0.47, ask=0.49, bid_sz=100.0, ask_sz=100.0,
         token="tok", stale=False) -> BookState:
    return BookState(token_id=token, bids=((bid, bid_sz),), asks=((ask, ask_sz),),
                     ts_exchange=ts, stale=stale)


BASE = {"half_spread": 0.01, "size": 100.0, "tick": 0.001}


# ── microprice + τ ─────────────────────────────────────────────────────────────

def test_microprice_weighted_mid():
    b = book(bid=0.40, ask=0.50, bid_sz=300.0, ask_sz=100.0)
    # bid-heavy book -> microprice above raw mid, toward the ask
    assert microprice(b) == pytest.approx((0.40 * 100 + 0.50 * 300) / 400)
    assert microprice(b) > 0.45


def test_microprice_fallback_zero_sizes():
    b = book(bid=0.40, ask=0.50, bid_sz=0.0, ask_sz=0.0)
    assert microprice(b) == pytest.approx(0.45)


def test_tau_hours_from_params_anchor():
    b = book(ts=1_000_000)
    assert tau_hours(b, {}) == float("inf")
    assert tau_hours(b, {"end_date_ms": 1_000_000 + 7_200_000}) == pytest.approx(2.0)


# ── InventoryAwareQuoter ───────────────────────────────────────────────────────

def test_v1_zero_inventory_matches_symmetric_center():
    q = InventoryAwareQuoter()
    orders = q.quote(book(), 0.0, {**BASE, "skew_k": 1e-4})
    assert len(orders) == 2
    bid = next(o for o in orders if o.side == "BUY")
    ask = next(o for o in orders if o.side == "SELL")
    assert bid.price == pytest.approx(0.47)   # micro=0.48 (balanced book) − 1c
    assert ask.price == pytest.approx(0.49)


def test_v1_long_inventory_skews_both_quotes_down():
    q = InventoryAwareQuoter()
    orders = q.quote(book(), 400.0, {**BASE, "skew_k": 1e-5})
    bid = next(o for o in orders if o.side == "BUY")
    ask = next(o for o in orders if o.side == "SELL")
    assert bid.price == pytest.approx(0.47 - 0.004, abs=1e-9)
    assert ask.price == pytest.approx(0.49 - 0.004, abs=1e-9)


def test_v1_at_cap_one_sided_reduce_only():
    q = InventoryAwareQuoter()
    orders = q.quote(book(), 500.0, {**BASE, "skew_k": 0.0, "inv_cap": 500.0})
    assert len(orders) == 1 and orders[0].side == "SELL"
    orders = q.quote(book(), -500.0, {**BASE, "skew_k": 0.0, "inv_cap": 500.0})
    assert len(orders) == 1 and orders[0].side == "BUY"


def test_v1_near_expiry_flatten_at_touch():
    q = InventoryAwareQuoter()
    p = {**BASE, "pull_hours": 6.0, "end_date_ms": 1_000_000 + 3_600_000}  # τ = 1h < 6h
    assert q.quote(book(), 0.0, p) == []                       # flat → no opening quotes
    orders = q.quote(book(), 300.0, p)
    assert len(orders) == 1 and orders[0].side == "SELL"
    assert orders[0].price == pytest.approx(0.49)              # joins the touch (best ask)
    assert orders[0].size == pytest.approx(100.0)              # clip-limited
    orders = q.quote(book(), -40.0, p)
    assert orders[0].side == "BUY" and orders[0].size == pytest.approx(40.0)


def test_v1_velocity_gate_pulls_opening_quotes():
    q = InventoryAwareQuoter()
    p = {**BASE, "tox_velocity": True}
    # feed a fast tape: mid jumps 2c inside the 10s window
    q.quote(book(ts=1_000_000, bid=0.47, ask=0.49), 0.0, p)
    orders = q.quote(book(ts=1_005_000, bid=0.49, ask=0.51), 0.0, p)
    assert orders == []                                        # toxic + flat → fully pulled
    orders = q.quote(book(ts=1_006_000, bid=0.49, ask=0.51), 200.0, p)
    assert len(orders) == 1 and orders[0].side == "SELL"       # reduce-only survives
    # same tape with the toggle OFF → two-sided as usual
    q2 = InventoryAwareQuoter()
    q2.quote(book(ts=1_000_000, bid=0.47, ask=0.49), 0.0, BASE)
    assert len(q2.quote(book(ts=1_005_000, bid=0.49, ask=0.51), 0.0, BASE)) == 2


def test_v1_imbalance_gate_separable():
    q = InventoryAwareQuoter()
    lopsided = book(bid_sz=1000.0, ask_sz=10.0)   # |imb| = 0.98 > 0.85
    assert q.quote(lopsided, 0.0, {**BASE, "tox_imbalance": True}) == []
    q2 = InventoryAwareQuoter()
    assert len(q2.quote(lopsided, 0.0, BASE)) == 2


def test_v1_stale_book_never_quoted():
    assert InventoryAwareQuoter().quote(book(stale=True), 0.0, BASE) == []


def test_v1_deterministic():
    a, b_ = InventoryAwareQuoter(), InventoryAwareQuoter()
    seq = [book(ts=1_000_000 + i * 1000, bid=0.47 + (i % 3) * 0.001) for i in range(20)]
    oa = [a.quote(s, float(i), {**BASE, "skew_k": 1e-5}) for i, s in enumerate(seq)]
    ob = [b_.quote(s, float(i), {**BASE, "skew_k": 1e-5}) for i, s in enumerate(seq)]
    assert oa == ob


# ── ASQuoter ───────────────────────────────────────────────────────────────────

def _warm_as(q: ASQuoter, p: dict, n=60, jitter=0.002):
    """Feed alternating mid moves so the σ EWMA warms up."""
    for i in range(n):
        b = book(ts=1_000_000 + i * 10_000, bid=0.47 + (i % 2) * jitter,
                 ask=0.49 + (i % 2) * jitter)
        q.quote(b, 0.0, p)
    return 1_000_000 + n * 10_000


def test_as_rung1_skew_direction_and_gamma_scaling():
    p1 = {**BASE, "as_gamma": 1e-4, "end_date_ms": 1_000_000 + 100 * 3.6e6}
    q1 = ASQuoter()
    ts = _warm_as(q1, p1)
    o_small = q1.quote(book(ts=ts), 500.0, p1)
    assert len(o_small) == 2
    bid1 = next(o for o in o_small if o.side == "BUY").price
    p2 = {**p1, "as_gamma": 1e-3}
    q2 = ASQuoter()
    ts = _warm_as(q2, p2)
    o_big = q2.quote(book(ts=ts), 500.0, p2)
    bid2 = next(o for o in o_big if o.side == "BUY").price
    assert bid2 <= bid1      # bigger γ → bigger long-inventory downshift


def test_as_rung2_spread_floor_and_cap():
    p = {**BASE, "as_gamma": 1e-4, "as_use_spread": True, "as_k_arr": 50.0,
         "end_date_ms": 1_000_000 + 100 * 3.6e6}
    q = ASQuoter()
    ts = _warm_as(q, p)
    orders = q.quote(book(ts=ts), 0.0, p)
    assert len(orders) == 2
    bid = next(o for o in orders if o.side == "BUY").price
    ask = next(o for o in orders if o.side == "SELL").price
    half = (ask - bid) / 2
    assert 0.001 - 1e-9 <= half <= 0.05 + 1e-9


def test_as_flatten_overlay_beats_as_tau():
    # near expiry the flatten fires even though the A-S τ term would shrink the skew
    p = {**BASE, "as_gamma": 1e-4, "pull_hours": 6.0, "end_date_ms": 1_000_000 + 3_600_000}
    q = ASQuoter()
    orders = q.quote(book(), 300.0, p)
    assert len(orders) == 1 and orders[0].side == "SELL" and orders[0].price == pytest.approx(0.49)


# ── BasketCarryQuoter ──────────────────────────────────────────────────────────

def test_basket_complementary_netting_and_balance_skew():
    legs = {"A": {"cond": "c1", "sign": 1}, "B": {"cond": "c1", "sign": -1},
            "C": {"cond": "c2", "sign": 1}}
    p = {**BASE, "skew_k": 1e-5, "basket_legs": legs,
         "half_spread_by_token": {"A": 0.01, "B": 0.01, "C": 0.01}}
    q = BasketCarryQuoter()
    # seed books + inventories: A long 400, B long 400 (complementary → c1 nets to 0), C flat
    q.quote(book(ts=1_000_000, token="A"), 400.0, p)
    q.quote(book(ts=1_001_000, token="B"), 400.0, p)
    orders = q.quote(book(ts=1_002_000, token="C"), 0.0, p)
    by_tok = {}
    for o in orders:
        by_tok.setdefault(o.token_id, {})[o.side] = o.price
    # c1 net = 400·(+1) + 400·(−1) = 0 = c2 → balanced → all legs quote symmetric around micro
    assert by_tok["A"]["BUY"] == pytest.approx(0.47)
    assert by_tok["C"]["BUY"] == pytest.approx(0.47)
    # now A long 800, B flat → c1 over-held (+800), c2 under-held → A sheds (quotes down),
    # C adds (quotes up)
    q.quote(book(ts=1_003_000, token="A"), 800.0, p)
    q.quote(book(ts=1_004_000, token="B"), 0.0, p)
    orders = q.quote(book(ts=1_005_000, token="C"), 0.0, p)
    by_tok = {}
    for o in orders:
        by_tok.setdefault(o.token_id, {})[o.side] = o.price
    assert by_tok["A"]["BUY"] < 0.47      # shed the over-held condition
    assert by_tok["C"]["BUY"] > 0.47      # add the under-held one


def test_basket_no_flatten_near_expiry():
    legs = {"A": {"cond": "c1", "sign": 1}}
    p = {**BASE, "skew_k": 0.0, "basket_legs": legs, "pull_hours": 6.0,
         "end_date_ms": 1_000_000 + 60_000}   # τ ≈ 1 min — v1 would flatten; basket carries
    q = BasketCarryQuoter()
    orders = q.quote(book(token="A"), 300.0, p)
    assert len(orders) == 2                   # still two-sided: carry, don't pull


# ── protocol: windowed costed PnL ─────────────────────────────────────────────

def _mk_fill(ts, side, price, qty, pos_after, basis_after, realized=0.0):
    return {"ts_exchange": ts, "side": side, "price": price, "qty": qty,
            "position_after": pos_after, "cost_basis_after": basis_after,
            "realized_delta": realized, "mid_at_fill": price}


def _mk_quote(ts, bid, ask):
    return {"ts_exchange": ts, "best_bid": bid, "best_ask": ask, "mid": (bid + ask) / 2,
            "orders": [], "stale": False}


def test_windowed_costed_partitions_and_carry():
    S = pr.SPLIT_TS_MS
    t0, t1 = S - 86_400_000, S + 86_400_000
    quotes = [_mk_quote(t, 0.47, 0.49) for t in range(t0, t1, 3_600_000)]
    fills = [
        _mk_fill(t0 + 1000, "BUY", 0.47, 100, 100, 0.47),                    # IS open
        _mk_fill(t0 + 7_200_000, "SELL", 0.49, 100, 0, 0.0, realized=2.0),   # IS round-trip
        _mk_fill(S + pr.EMBARGO_MS + 1000, "BUY", 0.47, 200, 200, 0.47),     # OOS open, carried
    ]
    w = pr.windowed_costed(fills, quotes, span=(t0, t1), n_boot=200)
    assert w["IS"].n_fills == 2 and w["IS"].realized_usd == pytest.approx(2.0)
    assert w["IS"].carry_usd == pytest.approx(0.0)         # flat at both IS edges
    assert w["OOS"].n_fills == 1
    # OOS carry: 200 long @0.47 basis marked to best bid 0.47 → 0 carry, costed 0
    assert w["OOS"].costed_usd == pytest.approx(0.0)
    assert w["OOS"].end_inventory == pytest.approx(200)


def test_windowed_costed_liquidation_at_bid_not_mid():
    S = pr.SPLIT_TS_MS
    t0, t1 = S - 86_400_000, S + 86_400_000
    quotes = ([_mk_quote(t, 0.47, 0.49) for t in range(t0, S, 3_600_000)]
              + [_mk_quote(t, 0.40, 0.44) for t in range(S, t1, 3_600_000)])
    fills = [_mk_fill(S + pr.EMBARGO_MS + 1000, "BUY", 0.41, 100, 100, 0.41)]
    w = pr.windowed_costed(fills, quotes, span=(t0, t1), n_boot=200)
    # long 100 @0.41, final touch bid 0.40 → carry = 100·(0.40−0.41) = −$1 (exit cost, not mid)
    assert w["OOS"].costed_usd == pytest.approx(-1.0)


# ── protocol: group delta + PBO ────────────────────────────────────────────────

def test_group_delta_point_and_beats():
    df = pd.DataFrame({
        "group_id": ["g1", "g1", "g2", "g3", "g4", "g5"],
        "a": [1.0, 1.2, 0.9, 1.1, 1.0, 1.3],
        "b": [0.2, 0.1, 0.3, 0.2, 0.1, 0.4],
    })
    d = pr.group_delta(df, "a", "b", n_boot=500, seed=1)
    assert d.point == pytest.approx(((df.a - df.b).mean()))
    assert d.beats and d.improves


def test_group_delta_noise_does_not_beat():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"group_id": [f"g{i//2}" for i in range(12)],
                       "a": rng.normal(0, 1, 12), "b": rng.normal(0, 1, 12)})
    d = pr.group_delta(df, "a", "b", n_boot=500, seed=1)
    assert not d.beats


def test_group_cscv_pbo_consistent_vs_flipped():
    groups = [f"g{i}" for i in range(6)]
    # cfgA dominates in every group → PBO = 0
    m = pd.DataFrame({"group_id": groups, "cfgA": [1.0] * 6, "cfgB": [0.0] * 6})
    assert pr.group_cscv_pbo(m).pbo == pytest.approx(0.0)
    # anti-symmetric pair: whichever config wins the IS half must lose the OOS half → PBO = 1
    m2 = pd.DataFrame({"group_id": groups,
                       "cfgA": [2.0, 2.0, 2.0, -2.0, -2.0, -2.0],
                       "cfgB": [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]})
    r = pr.group_cscv_pbo(m2)
    assert r.n_splits == 20 and r.pbo == pytest.approx(1.0)


def test_daily_pnl_series_shape():
    S = pr.SPLIT_TS_MS
    t0, t1 = S - 2 * 86_400_000, S + 2 * 86_400_000
    quotes = [_mk_quote(t, 0.47, 0.49) for t in range(t0, t1, 3_600_000)]
    fills = [_mk_fill(S + pr.EMBARGO_MS + 1000, "BUY", 0.47, 100, 100, 0.47, realized=0.0)]
    ser = pr.daily_pnl_series(fills, quotes, span=(t0, t1), window="OOS")
    assert ser.size >= 2 and np.isfinite(ser).all()
