"""Unit tests for the design-input analysis core (:mod:`mm_eval.design_inputs`).

Covers the error-prone numerics: book-JSON best-of-side parsing, backward as-of attach,
rank-AUC separation, token-cluster bootstrap OLS (planted-coefficient recovery + sign),
TTR bucketization edges, and bucketed edge (adverse-sign flip). Pure functions only — no engine,
no I/O.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from mm_eval import design_inputs as di


# ── book parsing ──────────────────────────────────────────────────────────────

def test_parse_book_best_picks_max_bid_min_ask_regardless_of_order():
    bids = json.dumps([{"price": "0.40", "size": "10"}, {"price": "0.44", "size": "5"},
                       {"price": "0.42", "size": "7"}])
    asks = json.dumps([{"price": "0.48", "size": "3"}, {"price": "0.46", "size": "9"},
                       {"price": "0.50", "size": "2"}])
    bb, bs, ab, as_ = di.parse_book_best(bids, asks)
    assert bb == 0.44 and bs == 5.0          # best bid = highest price
    assert ab == 0.46 and as_ == 9.0         # best ask = lowest price


def test_parse_book_best_malformed_is_nan_not_raise():
    bb, bs, ab, as_ = di.parse_book_best("not json", "[]")
    assert all(math.isnan(x) for x in (bb, bs, ab, as_))


def test_book_top_series_imbalance_and_depth():
    df = pd.DataFrame({
        "timestamp_ms": [200, 100],  # deliberately unsorted
        "bids": [json.dumps([{"price": "0.5", "size": "30"}]),
                 json.dumps([{"price": "0.5", "size": "10"}])],
        "asks": [json.dumps([{"price": "0.6", "size": "10"}]),
                 json.dumps([{"price": "0.6", "size": "10"}])],
    })
    top = di.book_top_series(df)
    assert list(top["ts"]) == [100, 200]                     # sorted ascending
    assert top.loc[top.ts == 100, "imbalance"].iloc[0] == 0.0        # 10 vs 10 -> balanced
    # 30 bid vs 10 ask -> (30-10)/40 = +0.5 (bid-heavy)
    assert abs(top.loc[top.ts == 200, "imbalance"].iloc[0] - 0.5) < 1e-9
    assert top.loc[top.ts == 200, "tob_depth"].iloc[0] == 20.0       # (30+10)/2


def test_asof_attach_book_is_backward_and_lookahead_free():
    fills = pd.DataFrame({"ts_exchange": [150, 250, 50], "x": [1, 2, 3]})
    top = pd.DataFrame({"ts": [100, 200], "imbalance": [0.1, 0.9], "tob_depth": [5.0, 9.0]})
    out = di.asof_attach_book(fills, top).sort_values("ts_exchange")
    # fill@50 -> before first snapshot -> NaN; fill@150 -> snapshot@100; fill@250 -> snapshot@200
    vals = dict(zip(out["ts_exchange"], out["imbalance_at_fill"]))
    assert math.isnan(vals[50])
    assert vals[150] == 0.1
    assert vals[250] == 0.9


# ── rank-AUC / feature screen ─────────────────────────────────────────────────

def test_rank_auc_perfect_and_null():
    # VIABLE (1) always scores higher than DEAD (0) -> AUC 1.0
    assert di.rank_auc(np.array([5, 6, 7, 1, 2, 3]), np.array([1, 1, 1, 0, 0, 0])) == 1.0
    # reversed -> 0.0
    assert di.rank_auc(np.array([1, 2, 3, 5, 6, 7]), np.array([1, 1, 1, 0, 0, 0])) == 0.0
    # identical distributions -> 0.5
    assert di.rank_auc(np.array([1, 2, 1, 2]), np.array([1, 1, 0, 0])) == 0.5


def test_screen_feature_diff_sign_matches_medians():
    vals = np.array([10.0, 12.0, 11.0, 2.0, 3.0, 1.0])
    lab = np.array([1, 1, 1, 0, 0, 0])
    fs = di.screen_feature(vals, lab, n_boot=200, seed=1)
    assert fs.auc == 1.0
    assert fs.median_viable > fs.median_dead
    assert fs.diff_point > 0


# ── cluster-bootstrap OLS ─────────────────────────────────────────────────────

def test_cluster_bootstrap_ols_recovers_planted_coefficient():
    rng = np.random.default_rng(0)
    # 20 clusters, ~200 rows each; y = 2.0 * x1 - 1.0 * x2 + cluster noise + fill noise
    rows = []
    for c in range(20):
        cnoise = rng.normal(0, 0.5)
        for _ in range(200):
            x1 = rng.normal(); x2 = rng.normal()
            y = 2.0 * x1 - 1.0 * x2 + cnoise + rng.normal(0, 0.3)
            rows.append({"cluster": c, "x1": x1, "x2": x2, "y": y})
    df = pd.DataFrame(rows)
    raw = di.cluster_bootstrap_ols(df, "y", ["x1", "x2"], "cluster", n_boot=300, seed=2)
    assert raw.attrs["n_clusters"] == 20 and raw.attrs["n_fills"] == 4000
    res = raw.set_index("term")
    # standardized: x1,x2 ~ unit SD so coef ≈ raw slope. Signs must be right + CI-robust.
    assert res.loc["x1", "coef_cents_per_sd"] > 0 and res.loc["x1", "sign_robust"]
    assert res.loc["x2", "coef_cents_per_sd"] < 0 and res.loc["x2", "sign_robust"]


def test_cluster_bootstrap_ols_drops_zero_variance_regressor():
    rng = np.random.default_rng(7)
    rows = []
    for c in range(12):
        for _ in range(80):
            x = rng.normal()
            rows.append({"cluster": c, "x": x, "dead": 0.0, "y": 1.5 * x + rng.normal(0, 0.3)})
    df = pd.DataFrame(rows)
    res = di.cluster_bootstrap_ols(df, "y", ["x", "dead"], "cluster", n_boot=200, seed=1)
    assert "dead" not in set(res["term"])           # constant-zero column dropped, no null row shipped
    assert res.set_index("term").loc["x", "sign_robust"]


def test_cluster_bootstrap_ols_wild_default_recovers_sign():
    rng = np.random.default_rng(8)
    rows = []
    for c in range(20):
        cnoise = rng.normal(0, 0.5)
        for _ in range(150):
            x = rng.normal()
            rows.append({"cluster": c, "x": x, "y": 1.2 * x + cnoise + rng.normal(0, 0.3)})
    df = pd.DataFrame(rows)
    res = di.cluster_bootstrap_ols(df, "y", ["x"], "cluster", n_boot=400, seed=2)  # wild default
    assert res.attrs["method"] == "wild"
    r = res.set_index("term")
    assert r.loc["x", "coef_cents_per_sd"] > 0 and r.loc["x", "sign_robust"]


def test_market_cluster_mean_ci_wider_than_fill_level_when_one_market_dominates():
    # one market's fills are all shifted +2; a fill-level CI would call the mean tightly positive,
    # but with only 2 markets the market-cluster CI must be much wider (few independent units).
    from mm_eval.design_inputs import _market_cluster_mean_ci
    vals = np.concatenate([np.full(500, 2.0) + 0.01, np.full(4, -2.0)])
    w = np.ones_like(vals)
    mp = np.concatenate([np.zeros(500, int), np.ones(4, int)])
    ci = _market_cluster_mean_ci(vals, w, mp, n_boot=500, seed=0)
    assert ci.hi - ci.lo > 1.0        # wide: only 2 independent markets, not 504 fills


def test_screen_feature_guards_tiny_class_ci():
    vals = np.array([10.0, 2.0, 3.0, 1.0, 4.0, 5.0])   # 1 viable vs 5 dead -> degenerate CI
    lab = np.array([1, 0, 0, 0, 0, 0])
    fs = di.screen_feature(vals, lab, n_boot=200, seed=1)
    assert not np.isfinite(fs.auc_lo) and not np.isfinite(fs.diff_lo)   # NaN bounds, not zero-width


def test_cluster_bootstrap_ols_null_feature_not_sign_robust():
    rng = np.random.default_rng(3)
    rows = []
    for c in range(15):
        for _ in range(100):
            x = rng.normal()
            rows.append({"cluster": c, "x": x, "y": rng.normal()})  # y independent of x
    df = pd.DataFrame(rows)
    res = di.cluster_bootstrap_ols(df, "y", ["x"], "cluster", n_boot=300, seed=4).set_index("term")
    assert not res.loc["x", "sign_robust"]           # CI should straddle 0


# ── TTR bucketization + bucketed edge ─────────────────────────────────────────

def test_ttr_bucketize_edges():
    edges = (0.0, 6.0, 24.0, math.inf)
    ttr = np.array([-1.0, 0.0, 5.9, 6.0, 23.0, 24.0, 1000.0, np.nan])
    idx = di.ttr_bucketize(ttr, edges)
    assert list(idx) == [-1, 0, 0, 1, 1, 2, 2, -1]   # negative & NaN excluded; half-open


def test_bucketed_edge_adverse_sign_flip_and_grouping():
    # bucket 0: markout +1c, drift -1c (adverse). bucket 1: markout +0.5c, drift +0.5c (favorable).
    markout = np.array([1.0, 1.0, 0.5, 0.5])
    drift = np.array([-1.0, -1.0, 0.5, 0.5])
    qty = np.array([1.0, 1.0, 1.0, 1.0])
    bidx = np.array([0, 0, 1, 1])
    mkts = np.array(["A", "A", "B", "B"])
    res = di.bucketed_edge(markout, drift, qty, bidx, ["b0", "b1"], market_ids=mkts,
                           n_boot=100, seed=0)
    assert res[0].n_fills == 2 and res[0].n_markets == 1
    assert abs(res[0].markout_cents.point - 1.0) < 1e-9
    assert abs(res[0].adverse_cents.point - 1.0) < 1e-9    # -drift(-1) = +1 adverse
    assert abs(res[1].adverse_cents.point + 0.5) < 1e-9    # -drift(+0.5) = -0.5 (favorable)


def test_within_market_slope_recovers_planted_fixed_effect():
    # each market has its OWN intercept (fixed effect) but a SHARED within slope of +2.0.
    rng = np.random.default_rng(5)
    rows = []
    for m in range(8):
        fe = rng.normal(0, 50)          # large market fixed effect that a pooled OLS would confound
        for _ in range(120):
            x = rng.normal()
            y = fe + 2.0 * x + rng.normal(0, 0.2)
            rows.append({"mkt": m, "x": x, "y": y})
    df = pd.DataFrame(rows)
    res = di.within_market_slope_ci(df, "y", "x", "mkt", n_boot=400, seed=1)
    assert abs(res["slope"] - 2.0) < 0.2 and res["sign_robust"]      # FE removes the intercepts
    assert res["n_markets"] == 8
    assert all(s > 0 for s in res["per_market_slopes"].values())     # every market's within slope +


def test_mid_volatility_zero_for_flat_book():
    bba = pd.DataFrame({"best_bid": [0.4] * 5, "best_ask": [0.6] * 5})
    assert di.mid_volatility(bba) == 0.0
