"""Task-5.1 unit tests — NeutralSpikeQuoter (two-lens, graduated) + whole-market CPCV.

Covers the load-bearing redesign mechanics:

* tape injection is causal and engine-independent;
* Lens 1 (VPIN volume clock + sweep weighting) flags one-sided flow and names the
  exposed side; session-relative band, warm-up honored;
* Lens 2 (AS z-score) flags adverse post-fill drift, calm baseline freezes in-regime;
* graduated response: directional-only suspends the exposed side; adverse-only reduces
  size; both → reduce-only at a widened offset; cap → reduce-only; NO calendar flatten;
* asymmetric repricing caps chase speed but withdraws instantly;
* OFI dampening shrinks the pressured side's size only;
* ported CPCV generator: fold/split/path shapes, purge, divisibility guards;
* nested selection never sees a held-out group; cohort folds are balanced;
* costed spans: regime windows partition the lifecycle; accounting matches protocol's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mm_engine.interfaces import BookState, MarketEvent
from mm_engine.strategies import (NSQ_DEFAULTS, NeutralSpikeQuoter, _parse_nsq_cfg)
from mm_eval.cpcv import (assign_cohorts, assign_folds, costed_spans,
                          generate_group_cpcv_splits, group_cluster_delta,
                          nested_outer_estimates, regime_spans)
from mm_eval.tape import TradeTape, tape_feed


def book(ts=1_000_000, bid=0.47, ask=0.49, bid_sz=100.0, ask_sz=100.0, stale=False):
    return BookState("tok", ((bid, bid_sz),), ((ask, ask_sz),), ts, stale)


def base_params(**kw):
    p = {"half_spread": 0.01, "size": 100.0, "tick": 0.001, "skew_k": 2e-5,
         "inv_cap": 500.0}
    p.update(kw)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# tape
# ──────────────────────────────────────────────────────────────────────────────

def _trade_ev(ts, price, size, side="BUY"):
    return MarketEvent("last_trade", "tok", ts, "", 0,
                       {"price": price, "size": size, "side": side})


def _book_ev(ts):
    return MarketEvent("book", "tok", ts, "", 0, {})


def test_tape_records_trades_causally_and_passes_items_through():
    tape = TradeTape()
    items = [_book_ev(1), _trade_ev(2, 0.5, 10), _book_ev(3), _trade_ev(4, 0.51, 5, "SELL")]
    seen = []
    for it in tape_feed(iter(items), tape):
        # at the moment the engine receives a trade event, the tape already has it
        if it.type == "last_trade":
            assert tape.rows[-1][0] == it.ts_exchange
        seen.append(it)
    assert seen == items
    assert [(r[0], r[3]) for r in tape.rows] == [(2, "BUY"), (4, "SELL")]


def test_tape_skips_malformed_payloads():
    tape = TradeTape()
    bad = MarketEvent("last_trade", "tok", 5, "", 0, {"price": None, "size": 3})
    list(tape_feed(iter([bad]), tape))
    assert len(tape) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Lens 1 — VPIN
# ──────────────────────────────────────────────────────────────────────────────

def _feed_one_sided_flow(q, params, n=600, side="BUY", start_ts=1_000_000, px=0.49):
    """Push n aggressive trades of one side through the tape + quote calls."""
    tape = params["trade_tape"]
    ts = start_ts
    for i in range(n):
        ts += 1000
        tape.append(ts, px + (0.002 if side == "BUY" else -0.002), 10.0, side)
        q.quote(book(ts=ts), 0.0, params)
    return ts


def test_lens1_flags_one_sided_flow_and_names_exposed_side():
    tape = TradeTape()
    params = base_params(nsq_lens1=True, trade_tape=tape)
    q = NeutralSpikeQuoter()
    # warm-up: balanced flow → no flag
    ts = 1_000_000
    for i in range(800):
        ts += 1000
        tape.append(ts, 0.49 if i % 2 == 0 else 0.47, 10.0, "BUY" if i % 2 == 0 else "SELL")
        q.quote(book(ts=ts), 0.0, params)
    assert q._vpin.flag_and_side(q._cfg) == (False, None)
    orders = q.quote(book(ts=ts + 1), 0.0, params)
    assert {o.side for o in orders} == {"BUY", "SELL"}
    # now heavy one-sided BUY aggression → directional flag, SELL side exposed
    ts = _feed_one_sided_flow(q, params, n=600, side="BUY", start_ts=ts)
    flag, exposed = q._vpin.flag_and_side(q._cfg)
    assert flag and exposed == "SELL"
    orders = q.quote(book(ts=ts + 1), 0.0, params)
    assert {o.side for o in orders} == {"BUY"}   # ask suspended, bid kept


def test_lens1_warmup_blocks_flag():
    tape = TradeTape()
    params = base_params(nsq_lens1=True, trade_tape=tape)
    q = NeutralSpikeQuoter()
    _feed_one_sided_flow(q, params, n=100)   # far below nsq_vpin_min_buckets of history
    flag, _ = q._vpin.flag_and_side(q._cfg)
    assert not flag


def test_lens1_sweep_weighting_amplifies_through_trades():
    from mm_engine.strategies import _VPINState
    cfg = _parse_nsq_cfg(base_params())
    st = _VPINState()
    st.last_touch = (0.47, 0.49)
    st.typ_size = 10.0
    st._n_size = 1
    tape = TradeTape()
    tape.append(1000, 0.495, 10.0, "BUY")   # 5 ticks through the ask
    st.ingest(tape, book(), cfg)
    swept_buy = st.buy_vol
    st2 = _VPINState()
    st2.last_touch = (0.47, 0.49)
    st2.typ_size = 10.0
    st2._n_size = 1
    tape2 = TradeTape()
    tape2.append(1000, 0.49, 10.0, "BUY")   # touch trade
    st2.ingest(tape2, book(), cfg)
    assert swept_buy > st2.buy_vol * 5      # exp(5·0.5) ≈ 12× the touch weight


# ──────────────────────────────────────────────────────────────────────────────
# Lens 2 — AS z-score
# ──────────────────────────────────────────────────────────────────────────────

def _mature_fill(q, params, ts, inv, mid_px, drift_to, horizon_s=30.0):
    """One fill at ``ts`` and a book ``horizon`` later at ``drift_to`` to mature it."""
    q.quote(book(ts=ts, bid=mid_px - 0.01, ask=mid_px + 0.01), inv, params)
    ts2 = ts + int(horizon_s * 1000) + 1000
    q.quote(book(ts=ts2, bid=drift_to - 0.01, ask=drift_to + 0.01), inv, params)
    return ts2


def test_lens2_flags_adverse_drift_and_reduces_size():
    params = base_params(nsq_lens2=True)
    q = NeutralSpikeQuoter()
    ts = 1_000_000
    inv = 0.0
    # calm baseline: 14 BUY fills with ~zero drift
    for i in range(14):
        inv += 10
        ts = _mature_fill(q, params, ts + 40_000, inv, 0.48, 0.48)
    assert not q._asz.flag(ts, q._cfg)
    orders = q.quote(book(ts=ts), inv, params)
    full_size = orders[0].size
    # now a strongly adverse fill: BUY then mid collapses 5c
    inv += 10
    ts = _mature_fill(q, params, ts + 40_000, inv, 0.48, 0.43)
    assert q._asz.flag(ts, q._cfg)
    orders = q.quote(book(ts=ts), inv, params)
    assert orders and all(o.size == pytest.approx(full_size * 0.5) for o in orders)
    # decay: flag clears after nsq_as_flag_decay_s without new bad drifts
    ts3 = ts + int(NSQ_DEFAULTS["nsq_as_flag_decay_s"] * 1000) + 1000
    assert not q._asz.flag(ts3, q._cfg)


def test_lens2_baseline_freezes_during_regime():
    params = base_params(nsq_lens2=True)
    q = NeutralSpikeQuoter()
    ts = 1_000_000
    inv = 0.0
    for i in range(14):
        inv += 10
        ts = _mature_fill(q, params, ts + 40_000, inv, 0.48, 0.48)
    n_before = len(q._asz.drifts)
    inv += 10
    ts = _mature_fill(q, params, ts + 40_000, inv, 0.48, 0.43)   # flags the regime
    n_at_flag = len(q._asz.drifts)
    # while flagged, matured drifts must NOT enter the calm baseline
    inv += 10
    ts = _mature_fill(q, params, ts + 1_000, inv, 0.43, 0.40)
    assert len(q._asz.drifts) == n_at_flag
    assert n_at_flag <= n_before + 1


# ──────────────────────────────────────────────────────────────────────────────
# graduated response / cap / no flatten
# ──────────────────────────────────────────────────────────────────────────────

def test_both_lenses_widened_reduce_only():
    tape = TradeTape()
    params = base_params(nsq_lens1=True, nsq_lens2=True, trade_tape=tape)
    q = NeutralSpikeQuoter()
    ts = 1_000_000
    inv = 0.0
    for i in range(14):     # calm lens-2 baseline
        inv += 10
        ts = _mature_fill(q, params, ts + 40_000, inv, 0.48, 0.48)
    ts = _feed_one_sided_flow(q, params, n=900, side="BUY", start_ts=ts)   # lens 1 on
    inv += 10
    ts = _mature_fill(q, params, ts + 1_000, inv, 0.48, 0.43)              # lens 2 on
    orders = q.quote(book(ts=ts), inv, params)
    assert len(orders) == 1 and orders[0].side == "SELL"                    # reduce-only (long)
    r = 0.48 - q._cfg["skew_k"] * inv
    assert orders[0].price >= r + 1.9 * q._cfg["half_spread"] - 1e-9        # widened offset
    assert orders[0].size <= inv                                            # never opens


def test_at_cap_reduce_only_and_no_calendar_flatten():
    params = base_params(end_date_ms=1_000_000 + 3_600_000.0)   # τ = 1h — v1 would flatten
    q = NeutralSpikeQuoter()
    orders = q.quote(book(), 600.0, params)                     # above cap 500
    assert len(orders) == 1 and orders[0].side == "SELL"
    # reduce-only rests passively at r + half_spread, NOT at the touch
    r = (0.47 * 100 + 0.49 * 100) / 200 - 2e-5 * 600
    assert orders[0].price == pytest.approx(round(r + 0.01, 3), abs=1e-9)
    # below cap, near expiry: still quotes BOTH sides (no pull_hours behavior at all)
    q2 = NeutralSpikeQuoter()
    orders2 = q2.quote(book(), 100.0, params)
    assert {o.side for o in orders2} == {"BUY", "SELL"}


def test_stale_book_quotes_nothing():
    q = NeutralSpikeQuoter()
    assert q.quote(book(stale=True), 0.0, base_params()) == []


# ──────────────────────────────────────────────────────────────────────────────
# asymmetric repricing
# ──────────────────────────────────────────────────────────────────────────────

def test_asym_repricing_slow_chase_fast_withdraw():
    params = base_params(nsq_asym=True, skew_k=0.0)
    q = NeutralSpikeQuoter()
    ts = 1_000_000
    q.quote(book(ts=ts, bid=0.47, ask=0.49), 0.0, params)
    # market jumps 5c in 2s: chase capped at 0.5 c/s → bid may rise ≤ ~1c
    o2 = q.quote(book(ts=ts + 2000, bid=0.52, ask=0.54), 0.0, params)
    bid2 = [o for o in o2 if o.side == "BUY"][0].price
    assert bid2 <= 0.47 + 0.5 / 100 * 2 + 1e-9       # capped chase
    # market drops back: bid withdraws instantly (down move = away)
    o3 = q.quote(book(ts=ts + 3000, bid=0.40, ask=0.42), 0.0, params)
    bid3 = [o for o in o3 if o.side == "BUY"][0].price
    assert bid3 == pytest.approx(0.40, abs=1e-9)     # r − half_spread immediately


def test_asym_off_reprices_instantly():
    params = base_params(nsq_asym=False, skew_k=0.0)
    q = NeutralSpikeQuoter()
    q.quote(book(ts=1_000_000, bid=0.47, ask=0.49), 0.0, params)
    o2 = q.quote(book(ts=1_002_000, bid=0.52, ask=0.54), 0.0, params)
    assert [o for o in o2 if o.side == "BUY"][0].price == pytest.approx(0.52, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# OFI dampening
# ──────────────────────────────────────────────────────────────────────────────

def test_ofi_dampens_pressured_side_only():
    params = base_params(nsq_damp_coeff=0.6, skew_k=0.0)
    q = NeutralSpikeQuoter()
    ts = 1_000_000
    # sustained buy pressure: bid size grows, ask size shrinks at improving bid prices
    for i in range(200):
        ts += 500
        q.quote(book(ts=ts, bid=0.47 + i * 0.0002, ask=0.49 + i * 0.0002,
                     bid_sz=200.0, ask_sz=20.0), 0.0, params)
    orders = q.quote(book(ts=ts + 500, bid=0.51, ask=0.53, bid_sz=200.0, ask_sz=20.0),
                     0.0, params)
    by_side = {o.side: o.size for o in orders}
    assert by_side["SELL"] < by_side["BUY"]          # ask (exposed to buy flow) dampened
    assert by_side["BUY"] == pytest.approx(100.0)
    assert by_side["SELL"] >= 100.0 * 0.2 - 1e-9     # floor honored


# ──────────────────────────────────────────────────────────────────────────────
# CPCV generator (ported)
# ──────────────────────────────────────────────────────────────────────────────

def test_cpcv_split_shapes_and_paths():
    r = generate_group_cpcv_splits(n_groups=12, n_folds=6, k_test=2)
    assert len(r["folds"]) == 6
    assert len(r["splits"]) == 15                    # C(6,2)
    assert len(r["paths"]) == 15                     # 5!! · 3!!/... = 15 complete pairings
    for sp in r["splits"]:
        test = {g for f in sp["test_groups_by_fold"].values() for g in f}
        train = set(sp["train_groups"].tolist())
        assert not (test & train)
        assert test | train == set(range(12))        # purge_groups=0 → full coverage
    # every path gives every fold exactly one test assignment
    for p in r["paths"]:
        folds = [f for f, _ in p["split_assignments"]]
        assert sorted(folds) == list(range(6))


def test_cpcv_purge_removes_adjacent_training_groups():
    r = generate_group_cpcv_splits(n_groups=12, n_folds=6, k_test=2, purge_groups=1)
    sp = next(s for s in r["splits"] if s["test_fold_indices"] == (0, 1))
    # fold 2 (groups 4,5) is order-adjacent to test fold 1 → group 4 purged
    assert 4 not in sp["train_groups"]
    assert 5 in sp["train_groups"]


def test_cpcv_guards():
    with pytest.raises(ValueError):
        generate_group_cpcv_splits(12, 5, 2)         # 5 % 2 != 0
    with pytest.raises(ValueError):
        generate_group_cpcv_splits(3, 6, 2)          # fewer groups than folds


# ──────────────────────────────────────────────────────────────────────────────
# cohorts + folds + nested honesty
# ──────────────────────────────────────────────────────────────────────────────

def _fake_features(n=12, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "group_id": [f"g{i}" for i in range(n)],
        "t_first": np.arange(n) * 1000,
        "t_last": np.arange(n) * 1000 + 5000,
        "sweep_share": rng.uniform(0, 0.5, n),
        "flow_rate": rng.uniform(10, 1000, n),
    })


def test_fold_assignment_is_cohort_balanced_and_deterministic():
    feat = assign_cohorts(_fake_features())
    f1 = assign_folds(feat, n_folds=6)
    f2 = assign_folds(feat, n_folds=6)
    assert (f1["fold"] == f2["fold"]).all()
    assert set(f1["fold"]) == set(range(6))
    assert f1.groupby("fold").size().max() - f1.groupby("fold").size().min() <= 1


def test_nested_selection_never_sees_test_groups():
    """Config 'cheat' is the best ONLY on group g0; inner selection on splits whose
    training set excludes g0 must not pick it, and g0's honest estimate must come from
    configs selected without seeing g0."""
    groups = [f"g{i}" for i in range(6)]
    usd = pd.DataFrame(index=groups, data={
        "honest_cfg": [10.0] * 6,
        "cheat": [1000.0, -50.0, -50.0, -50.0, -50.0, -50.0],
    })
    qty = pd.DataFrame(index=groups, data={"honest_cfg": [100.0] * 6, "cheat": [100.0] * 6})
    r = generate_group_cpcv_splits(6, 6, 2)
    order_to_group = {i: groups[i] for i in range(6)}
    res = nested_outer_estimates(usd, qty, ["honest_cfg", "cheat"], r["splits"],
                                 order_to_group, rung="t")
    # splits with g0 in TEST have training pooled cheat = -50 → select honest_cfg
    g0 = res.per_group[res.per_group.group_id == "g0"]
    assert g0["honest_c"].iloc[0] == pytest.approx(10.0 / 100.0 * 100.0)
    # splits with g0 in TRAIN may select cheat — and pay for it OOS
    cheat_rows = res.per_split[res.per_split.selected == "cheat"]
    assert (cheat_rows["test_pooled_c"] < 0).all()


def test_group_cluster_delta_gate():
    a = pd.DataFrame({"group_id": [f"g{i}" for i in range(8)],
                      "honest_c": [1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 1.0, 1.1]})
    b = pd.DataFrame({"group_id": [f"g{i}" for i in range(8)],
                      "honest_c": [0.0] * 8})
    d = group_cluster_delta(a, b, n_boot=500, seed=3)
    assert d.beats and d.point == pytest.approx(1.05, abs=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# costed spans / regime windows
# ──────────────────────────────────────────────────────────────────────────────

def _fills_quotes():
    quotes = [{"ts_exchange": t, "best_bid": 0.47, "best_ask": 0.49}
              for t in range(0, 100_001, 1000)]
    fills = [
        {"ts_exchange": 10_000, "qty": 100.0, "realized_delta": 0.0,
         "position_after": 100.0, "cost_basis_after": 0.47},
        {"ts_exchange": 60_000, "qty": 100.0, "realized_delta": 2.0,
         "position_after": 0.0, "cost_basis_after": 0.0},
    ]
    return fills, quotes


def test_costed_spans_matches_realized_plus_carry():
    fills, quotes = _fills_quotes()
    out = costed_spans(fills, quotes, {"full": (0, 100_001), "first": (0, 50_000),
                                       "second": (50_000, 100_001)})
    assert out["full"].costed_usd == pytest.approx(2.0)
    # window additivity: realized+carry over disjoint covering spans sums to full
    assert (out["first"].costed_usd + out["second"].costed_usd
            == pytest.approx(out["full"].costed_usd))
    assert out["first"].carry_usd == pytest.approx(100.0 * (0.47 - 0.47))


def test_regime_spans_partition_and_condition_only():
    span = (0, 200 * 3_600_000)
    end_ms = 100 * 3_600_000.0
    spans = regime_spans(end_ms, span, "politics_negrisk")
    assert spans["full"] == span
    # endgame = last 6h before end; approach 6-48h; midlife everything earlier
    assert spans["tau_endgame"] == (int(end_ms - 6 * 3.6e6), int(end_ms))
    assert spans["tau_approach"] == (int(end_ms - 48 * 3.6e6), int(end_ms - 6 * 3.6e6))
    assert spans["tau_midlife"][0] == 0
    # no end date → conditioning silently absent, full span still evaluated
    assert set(regime_spans(float("nan"), span, "esports")) == {"full"}
