"""Design-input analysis layer for Task 5 — the *market screen* and the *time-to-resolution regime*.

This module is the pure-numerics half of the two design-input questions that must precede the
Task-5 inventory-managed strategy. It is a **sibling of** :mod:`mm_eval.metrics` — same discipline
(quantity-weighted means, moving-block / cluster bootstrap CIs, cents/contract units), no engine
state, no I/O — so every function is unit-testable in isolation. The runner
``scripts/mm_design_inputs_run.py`` feeds it the engine's raw per-fill telemetry and the captured
Parquet; it never modifies :mod:`mm_engine`.

Two analyses:

1. **Failure attribution → the market screen.** For each token we compare the Task-4
   ``VIABLE`` vs ``DEAD`` verdict against *observable* book features (half-spread, top-of-book
   depth, book imbalance, price extremity, trade intensity, mid volatility) with a rank-AUC + a
   group-difference CI, AND we regress the **per-fill markout** (thousands of points) on those
   features with a **token-clustered bootstrap** so the inference respects that the 24 markets —
   not the ~26k fills — are the independent units.

2. **NegRisk time-to-resolution regime.** Per politics fill we compute time-to-resolution
   (``end_date − fill_ts``) and bucket the per-fill markout + adverse selection by
   time-to-expiry, to test whether mid-life is favorable and near-expiry is toxic — and, crucially,
   to measure **which time-to-resolution regimes the ~11-day capture even covers** per market.

**Sign conventions** are inherited from :mod:`mm_eval.metrics`: ``markout_to_fill`` (>0 accretive,
includes the captured half-spread) and ``adverse_drift`` (<0 = mid moved against us);
``adverse_selection ≡ −adverse_drift``. All public numbers are cents/contract.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mm_eval.metrics import CENTS, CI, block_bootstrap_mean_ci

# ──────────────────────────────────────────────────────────────────────────────
# Order-book snapshot parsing (observable top-of-book depth + imbalance)
# ──────────────────────────────────────────────────────────────────────────────
# The captured `book` table stores `bids`/`asks` as JSON strings of ``[{"price","size"}, ...]``.
# The array order is NOT guaranteed best-first, so best bid = max price, best ask = min price.


def _best_level(cells: list[dict], want_max: bool) -> tuple[float, float]:
    """Return (price, size) of the best level of one side (max-price bid / min-price ask).

    ``want_max=True`` for bids (best = highest price), ``False`` for asks (best = lowest price).
    Empty / malformed side returns (nan, nan).
    """
    best_p = -math.inf if want_max else math.inf
    best_s = float("nan")
    found = False
    for c in cells:
        try:
            p = float(c["price"]); s = float(c["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if (want_max and p > best_p) or (not want_max and p < best_p):
            best_p, best_s, found = p, s, True
    return (best_p, best_s) if found else (float("nan"), float("nan"))


def parse_book_best(bids_json: str, asks_json: str) -> tuple[float, float, float, float]:
    """(best_bid, bid_size, best_ask, ask_size) from a raw book row's JSON strings.

    Any parse failure yields NaNs for that row (never raises) — a corrupt snapshot is dropped,
    not fatal.
    """
    try:
        bids = json.loads(bids_json) if isinstance(bids_json, str) else (bids_json or [])
        asks = json.loads(asks_json) if isinstance(asks_json, str) else (asks_json or [])
    except (json.JSONDecodeError, TypeError):
        return (float("nan"),) * 4
    bp, bs = _best_level(bids, want_max=True)
    ap, as_ = _best_level(asks, want_max=False)
    return bp, bs, ap, as_


def book_top_series(book_df: pd.DataFrame) -> pd.DataFrame:
    """Time series of top-of-book depth + imbalance from raw `book` snapshots.

    Input columns: ``timestamp_ms, bids, asks``. Output (sorted by ts, valid rows only):
    ``ts, best_bid, bid_sz, best_ask, ask_sz, tob_depth, imbalance`` where
    ``tob_depth = (bid_sz + ask_sz)/2`` (contracts) and
    ``imbalance = (bid_sz − ask_sz)/(bid_sz + ask_sz)`` in ``[-1, 1]`` (>0 = bid-heavy).
    """
    if book_df.empty:
        return pd.DataFrame(columns=["ts", "best_bid", "bid_sz", "best_ask", "ask_sz",
                                     "tob_depth", "imbalance"])
    parsed = [parse_book_best(b, a) for b, a in zip(book_df["bids"], book_df["asks"])]
    arr = np.asarray(parsed, dtype=float)
    out = pd.DataFrame({
        "ts": book_df["timestamp_ms"].to_numpy(dtype=np.int64),
        "best_bid": arr[:, 0], "bid_sz": arr[:, 1], "best_ask": arr[:, 2], "ask_sz": arr[:, 3],
    })
    tot = out["bid_sz"] + out["ask_sz"]
    out["tob_depth"] = tot / 2.0
    out["imbalance"] = np.where(tot > 0, (out["bid_sz"] - out["ask_sz"]) / tot, np.nan)
    out = out[np.isfinite(out["best_bid"]) & np.isfinite(out["best_ask"])]
    return out.sort_values("ts").reset_index(drop=True)


def token_book_features(top: pd.DataFrame) -> dict:
    """Per-token observable book features (medians/dispersions) from a ``book_top_series`` frame."""
    if top.empty:
        return {"median_tob_depth": float("nan"), "median_imbalance": float("nan"),
                "median_abs_imbalance": float("nan"), "imbalance_vol": float("nan"),
                "n_book_snaps": 0}
    imb = top["imbalance"].to_numpy()
    imb = imb[np.isfinite(imb)]
    return {
        "median_tob_depth": float(np.median(top["tob_depth"])),
        "median_imbalance": float(np.median(imb)) if imb.size else float("nan"),
        "median_abs_imbalance": float(np.median(np.abs(imb))) if imb.size else float("nan"),
        "imbalance_vol": float(np.std(imb)) if imb.size else float("nan"),
        "n_book_snaps": int(len(top)),
    }


def mid_volatility(bba_df: pd.DataFrame) -> float:
    """Realized volatility of the touch-mid (a proxy for repricing churn / resolution ambiguity).

    Std of successive mid *changes* (cents/contract), from the ``bba`` stream
    (``best_bid, best_ask``). Change-based (not level) so a token parked at a stable price scores
    low regardless of its price level; higher = the mid is repriced more, which co-moves with
    adverse selection.
    """
    if bba_df.empty or len(bba_df) < 3:
        return float("nan")
    b = bba_df["best_bid"].to_numpy(dtype=float)
    a = bba_df["best_ask"].to_numpy(dtype=float)
    ok = np.isfinite(b) & np.isfinite(a) & (a > b)
    mid = (b[ok] + a[ok]) / 2.0
    if mid.size < 3:
        return float("nan")
    return float(np.std(np.diff(mid)) * CENTS)


def asof_attach_book(fills_df: pd.DataFrame, top: pd.DataFrame,
                     *, ts_col: str = "ts_exchange") -> pd.DataFrame:
    """As-of (backward) attach the prevailing top-of-book imbalance/depth to each fill.

    For every fill at ``t`` we take the last book snapshot with ``ts ≤ t`` — lookahead-free (the
    book state a live quoter would have seen). Adds ``imbalance_at_fill`` and ``depth_at_fill``
    columns; fills before the first snapshot get NaN.
    """
    out = fills_df.copy()
    if top.empty or fills_df.empty:
        out["imbalance_at_fill"] = np.nan
        out["depth_at_fill"] = np.nan
        return out
    left = out.sort_values(ts_col)
    right = top.sort_values("ts")
    merged = pd.merge_asof(left, right[["ts", "imbalance", "tob_depth"]],
                           left_on=ts_col, right_on="ts", direction="backward")
    merged = merged.rename(columns={"imbalance": "imbalance_at_fill", "tob_depth": "depth_at_fill"})
    return merged.drop(columns=["ts"]).sort_index()


# ──────────────────────────────────────────────────────────────────────────────
# Task 1a — per-token VIABLE-vs-DEAD separation (rank-AUC + group-difference CI)
# ──────────────────────────────────────────────────────────────────────────────

def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC = P(feature ranks a VIABLE token above a DEAD one) — the Mann-Whitney statistic.

    ``labels`` is 1 (VIABLE) / 0 (DEAD). 0.5 = no separation; >0.5 = higher feature ⇒ more
    likely VIABLE; <0.5 = higher feature ⇒ more likely DEAD. Ties count as half. NaNs dropped.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    s, y = s[m], y[m]
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


@dataclass(frozen=True)
class FeatureScreen:
    feature: str
    auc: float
    auc_lo: float
    auc_hi: float
    median_viable: float
    median_dead: float
    diff_point: float          # median_viable − median_dead
    diff_lo: float
    diff_hi: float
    n_viable: int
    n_dead: int


MIN_CLASS_FOR_CI = 4  # below this per-class count the token-resample CI is degenerate (near zero-width)


def screen_feature(values: np.ndarray, labels: np.ndarray, *,
                   n_boot: int = 5000, seed: int = 0) -> FeatureScreen:
    """One observable feature's VIABLE-vs-DEAD separation: rank-AUC + median difference, both bootstrapped.

    The bootstrap resamples **tokens** (the independent units) with replacement — the only honest
    unit given the tiny per-universe n, but at these counts (~12 tokens) it can only *rank* features,
    not certify a threshold. **Guard:** when either class has fewer than ``MIN_CLASS_FOR_CI`` tokens
    the resample degenerates (every non-trivial draw repeats the same one/two minority values, giving
    a spuriously zero-width CI), so we return NaN bounds rather than a false-certain interval.
    """
    v = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=float)
    m = np.isfinite(v) & np.isfinite(y)
    v, y = v[m], y[m]
    pos, neg = v[y == 1], v[y == 0]
    auc = rank_auc(v, y)
    med_v = float(np.median(pos)) if pos.size else float("nan")
    med_d = float(np.median(neg)) if neg.size else float("nan")
    diff = med_v - med_d
    n = v.size
    rng = np.random.default_rng(seed)
    aucs, diffs = [], []
    if n >= 3 and min(pos.size, neg.size) >= MIN_CLASS_FOR_CI:
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            bv, by = v[idx], y[idx]
            if (by == 1).sum() == 0 or (by == 0).sum() == 0:
                continue
            aucs.append(rank_auc(bv, by))
            diffs.append(np.median(bv[by == 1]) - np.median(bv[by == 0]))
    def _q(a):
        return (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))) if a else (float("nan"),) * 2
    a_lo, a_hi = _q(aucs)
    d_lo, d_hi = _q(diffs)
    return FeatureScreen(feature="", auc=auc, auc_lo=a_lo, auc_hi=a_hi,
                         median_viable=med_v, median_dead=med_d,
                         diff_point=diff, diff_lo=d_lo, diff_hi=d_hi,
                         n_viable=int(pos.size), n_dead=int(neg.size))


def screen_features(df: pd.DataFrame, features: list[str], label_col: str,
                    *, n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Run :func:`screen_feature` over a list of columns; return a tidy DataFrame sorted by |AUC−0.5|."""
    rows = []
    y = df[label_col].to_numpy(dtype=float)
    for i, f in enumerate(features):
        fs = screen_feature(df[f].to_numpy(dtype=float), y, n_boot=n_boot, seed=seed + i)
        rows.append({"feature": f, "auc": fs.auc, "auc_lo": fs.auc_lo, "auc_hi": fs.auc_hi,
                     "median_viable": fs.median_viable, "median_dead": fs.median_dead,
                     "diff": fs.diff_point, "diff_lo": fs.diff_lo, "diff_hi": fs.diff_hi,
                     "separation": abs(fs.auc - 0.5) if np.isfinite(fs.auc) else np.nan,
                     "n_viable": fs.n_viable, "n_dead": fs.n_dead})
    return pd.DataFrame(rows).sort_values("separation", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Task 1b — per-fill markout regression with a token-clustered bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def _ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients via least squares (rank-deficiency tolerant). X includes the intercept col."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _rademacher_signs(rng, k: int) -> np.ndarray:
    """±1 Rademacher weights for ``k`` clusters (one per cluster, mapped to rows by the caller)."""
    return rng.integers(0, 2, size=k).astype(float) * 2.0 - 1.0


def _wild_cluster_ci(X, y, row_cluster_pos, n_clusters, *, n_boot, seed, confidence=0.95):
    """Unrestricted **wild cluster bootstrap** CI for every OLS coefficient (X fixed, Rademacher).

    The pairs cluster bootstrap under-covers badly at small cluster counts (Monte-Carlo: ~15% null
    FP at K=6, ~11% at K=24 vs the nominal 5%). The wild cluster bootstrap (Cameron-Gelbach-Miller)
    is the standard small-K fix: hold ``X`` fixed, resample the residuals with a **single ±1 sign
    per cluster** (so within-cluster dependence is preserved), and recompute the coefficients. With
    ``X`` fixed, ``β* = X⁺ y*`` is exact and cheap. It is *better-covering than pairs at small K but
    not exact* — at K≲6 no bootstrap is nominal, so those CIs stay "directional," see the note.
    """
    pinv = np.linalg.pinv(X)
    beta = pinv @ y
    resid = y - X @ beta
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        signs = _rademacher_signs(rng, n_clusters)[row_cluster_pos]   # one sign per cluster → per row
        y_star = X @ beta + resid * signs
        draws[b] = pinv @ y_star
    alpha = (1 - confidence) / 2
    lo = np.quantile(draws, alpha, axis=0)
    hi = np.quantile(draws, 1 - alpha, axis=0)
    return beta, lo, hi


def cluster_bootstrap_ols(
    df: pd.DataFrame, y_col: str, feature_cols: list[str], cluster_col: str,
    *, n_boot: int = 2000, seed: int = 0, standardize: bool = True, method: str = "wild",
    drop_zero_variance: bool = True,
) -> pd.DataFrame:
    """Per-fill OLS of ``y`` on features, with a **cluster (token) bootstrap** for the CIs.

    Standardizing each feature to unit SD makes coefficients comparable: with ``y`` in
    cents/contract, the coefficient is **"cents/contract per +1 SD of the feature"**. Clustering is
    on the token because fills within a token are dependent (adverse selection is autocorrelated),
    which collapses the effective n to the ~24 markets, so token-level features get honestly-wide
    bands. ``method="wild"`` (default) uses the **wild cluster bootstrap** (Rademacher) — the
    small-K-appropriate method; ``method="pairs"`` is the resample-whole-clusters variant (kept for
    comparison, anti-conservative at small K). Zero-variance regressors (e.g. ``queue_ahead`` ≡ 0
    under the 0-ms pessimistic feed) are dropped by default so no meaningless null row is reported.
    Returns one row per term: coefficient + percentile CI + ``sign_robust`` (CI excludes 0).
    """
    sub = df[[y_col, cluster_col] + feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    y = sub[y_col].to_numpy(dtype=float)
    cols = list(feature_cols)
    Xraw = sub[cols].to_numpy(dtype=float)
    sd0 = Xraw.std(axis=0)
    if drop_zero_variance and np.any(sd0 == 0):
        keep = sd0 > 0
        cols = [c for c, k in zip(cols, keep) if k]
        Xraw = Xraw[:, keep]
    if standardize:
        mu = Xraw.mean(axis=0)
        sd = Xraw.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Xz = (Xraw - mu) / sd
    else:
        Xz = Xraw
    n = Xz.shape[0]
    X = np.column_stack([np.ones(n), Xz])          # intercept first
    names = ["intercept"] + cols

    clusters = sub[cluster_col].to_numpy()
    uniq, row_cluster_pos = np.unique(clusters, return_inverse=True)
    if method == "wild":
        beta, lo, hi = _wild_cluster_ci(X, y, row_cluster_pos, uniq.size, n_boot=n_boot, seed=seed)
    else:
        beta = _ols_beta(X, y)
        idx_by = {i: np.flatnonzero(row_cluster_pos == i) for i in range(uniq.size)}
        rng = np.random.default_rng(seed)
        draws = np.full((n_boot, X.shape[1]), np.nan)
        for b in range(n_boot):
            pick = rng.integers(0, uniq.size, uniq.size)
            rows = np.concatenate([idx_by[i] for i in pick])
            try:
                draws[b] = _ols_beta(X[rows], y[rows])
            except np.linalg.LinAlgError:
                continue
        lo = np.nanquantile(draws, 0.025, axis=0)
        hi = np.nanquantile(draws, 0.975, axis=0)
    rows = []
    for j, nm in enumerate(names):
        rows.append({"term": nm, "coef_cents_per_sd": float(beta[j]),
                     "lo": float(lo[j]), "hi": float(hi[j]),
                     "sign_robust": bool(np.isfinite(lo[j]) and np.isfinite(hi[j])
                                         and (lo[j] > 0 or hi[j] < 0))})
    out = pd.DataFrame(rows)
    out.attrs["n_fills"] = int(n)
    out.attrs["n_clusters"] = int(uniq.size)
    out.attrs["method"] = method
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Task 2 — time-to-resolution bucketing + bucketed edge
# ──────────────────────────────────────────────────────────────────────────────

# Default TTR bucket edges (hours). Fine near expiry (where the delta/gamma spike-zone bites),
# coarse far out (where the durable outrights live). The last, open bucket is "deep mid-life".
DEFAULT_TTR_EDGES_H = (0.0, 6.0, 24.0, 72.0, 168.0, 720.0, math.inf)
DEFAULT_TTR_LABELS = ("<6h", "6-24h", "1-3d", "3-7d", "7-30d", ">30d")


def ttr_bucketize(ttr_hours: np.ndarray, edges=DEFAULT_TTR_EDGES_H) -> np.ndarray:
    """Assign each TTR (hours) to a bucket index ``[0, len(edges)-2]``; NaN/negative → -1 (excluded).

    ``edges`` is monotone increasing (last may be ``inf``); bucket k covers ``[edges[k], edges[k+1])``.
    """
    t = np.asarray(ttr_hours, dtype=float)
    idx = np.full(t.shape, -1, dtype=int)
    edges = np.asarray(edges, dtype=float)
    for k in range(len(edges) - 1):
        sel = np.isfinite(t) & (t >= edges[k]) & (t < edges[k + 1])
        idx[sel] = k
    return idx


def within_market_slope_ci(
    df: pd.DataFrame, y_col: str, x_col: str, market_col: str,
    *, n_boot: int = 3000, seed: int = 0, method: str = "wild",
) -> dict:
    """Market-fixed-effects slope of ``y`` on ``x`` (within-transform) with a cluster bootstrap.

    Demeans ``y`` and ``x`` **within each market** (the fixed-effects / within transformation), then
    fits the through-origin OLS slope on the demeaned variables — so the estimate uses only
    variation *inside* a market and cannot be driven by cross-market composition. ``x`` varies
    within market, so ``method="wild"`` (default, the small-K-appropriate wild cluster bootstrap) is
    the right CI; ``method="pairs"`` (resample whole markets) is provided for transparency (it under-
    covers at small K). Returns the pooled within-slope + CI + ``sign_robust`` + per-market raw
    slopes. This is the clean test of "does toxicity rise as a single market approaches its own
    resolution?"
    """
    sub = df[[y_col, x_col, market_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub[market_col].nunique() < 2 or len(sub) < 10:
        return {"slope": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "sign_robust": False, "n": int(len(sub)), "n_markets": int(sub[market_col].nunique()),
                "per_market_slopes": {}}
    yw = (sub[y_col] - sub.groupby(market_col)[y_col].transform("mean")).to_numpy()
    xw = (sub[x_col] - sub.groupby(market_col)[x_col].transform("mean")).to_numpy()

    def _slope(y, x):
        denom = float(np.sum(x * x))
        return float(np.sum(x * y) / denom) if denom > 0 else float("nan")

    point = _slope(yw, xw)
    mkts = sub[market_col].to_numpy()
    uniq, row_pos = np.unique(mkts, return_inverse=True)
    denom = float(np.sum(xw * xw))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    if method == "wild":
        # Wild cluster bootstrap (Rademacher): through-origin slope, X (=xw) held fixed, residuals
        # resampled with one ±1 sign per market. Small-K-appropriate (pairs under-covers at K≈6).
        resid = yw - point * xw
        for b in range(n_boot):
            signs = _rademacher_signs(rng, uniq.size)[row_pos]
            y_star = point * xw + resid * signs
            draws[b] = float(np.sum(xw * y_star) / denom) if denom > 0 else float("nan")
    else:  # pairs: resample whole markets (shown for transparency; under-covers at small K)
        idx_by = {i: np.flatnonzero(row_pos == i) for i in range(uniq.size)}
        for b in range(n_boot):
            pick = rng.integers(0, uniq.size, uniq.size)
            rows = np.concatenate([idx_by[i] for i in pick])
            draws[b] = _slope(yw[rows], xw[rows])
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    per = {}
    for i, m in enumerate(uniq):
        ii = np.flatnonzero(row_pos == i)
        per[str(m)] = _slope(yw[ii], xw[ii])   # within-market: demeaned already, so this is the raw within slope
    return {"slope": point, "lo": float(lo), "hi": float(hi),
            "sign_robust": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
            "n": int(len(sub)), "n_markets": int(uniq.size), "per_market_slopes": per}


@dataclass(frozen=True)
class BucketEdge:
    label: str
    n_fills: int
    markout_cents: CI
    adverse_cents: CI            # adverse selection = −drift (positive = against)
    adverse_rate: float          # qty-weighted share of fills with markout < 0
    n_markets: int               # distinct markets contributing to this bucket


def _market_cluster_mean_ci(values, weights, market_pos, *,
                            n_boot: int = 2000, seed: int = 0, confidence: float = 0.95) -> CI:
    """Qty-weighted mean with a **market-cluster** bootstrap CI (resample whole markets).

    The independent unit is the market, not the (autocorrelated) fill — a bucket of 800 fills from
    5 markets has ~5 independent observations, not 800. Resampling markets with replacement gives a
    CI that reflects that. NaNs dropped pairwise. Degrades to NaN bounds when <2 markets survive.
    (At the small market counts near expiry this still under-covers — read as directional.)
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mp = np.asarray(market_pos)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w, mp = v[mask], w[mask], mp[mask]
    n = v.size
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    point = float(np.sum(v * w) / np.sum(w))
    uniq = np.unique(mp)
    if uniq.size < 2:
        return CI(point, float("nan"), float("nan"), n)
    idx_by = {m: np.flatnonzero(mp == m) for m in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, uniq.size, replace=True)
        rows = np.concatenate([idx_by[m] for m in pick])
        sw = np.sum(w[rows])
        draws[b] = np.sum(v[rows] * w[rows]) / sw if sw > 0 else np.nan
    lo, hi = np.nanquantile(draws, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return CI(point, float(lo), float(hi), n)


def bucketed_edge(
    markout_cents: np.ndarray, adverse_drift_cents: np.ndarray, qty: np.ndarray,
    bucket_idx: np.ndarray, labels, market_ids=None,
    *, n_boot: int = 2000, seed: int = 0,
) -> list[BucketEdge]:
    """Per-TTR-bucket qty-weighted markout + adverse selection with **market-cluster** CIs.

    ``markout_cents`` / ``adverse_drift_cents`` are already in cents (drift <0 = adverse; the
    reported ``adverse_cents`` flips the sign so positive = against). When ``market_ids`` is given,
    the CI resamples whole markets (the honest independent unit — a near-expiry bucket dominated by
    a few short-dated markets does NOT get a spuriously-tight fill-count CI); each bucket also
    reports how many distinct markets populate it. Without ``market_ids`` it falls back to the
    fill-level moving-block bootstrap.
    """
    mk = np.asarray(markout_cents, dtype=float)
    dr = np.asarray(adverse_drift_cents, dtype=float)
    q = np.asarray(qty, dtype=float)
    mids = np.asarray(market_ids) if market_ids is not None else None
    out = []
    for k, lab in enumerate(labels):
        sel = bucket_idx == k
        n = int(sel.sum())
        if n == 0:
            nan_ci = CI(float("nan"), float("nan"), float("nan"), 0)
            out.append(BucketEdge(lab, 0, nan_ci, nan_ci, float("nan"), 0))
            continue
        if mids is not None:
            mp = mids[sel]
            m_ci = _market_cluster_mean_ci(mk[sel] / CENTS, q[sel], mp, n_boot=n_boot, seed=seed).cents()
            d_ci = _market_cluster_mean_ci(dr[sel] / CENTS, q[sel], mp, n_boot=n_boot, seed=seed).cents()
            nm = int(len(np.unique(mp)))
        else:
            m_ci = block_bootstrap_mean_ci(mk[sel] / CENTS, q[sel], n_boot=n_boot, seed=seed).cents()
            d_ci = block_bootstrap_mean_ci(dr[sel] / CENTS, q[sel], n_boot=n_boot, seed=seed).cents()
            nm = 0
        adverse = CI(-d_ci.point, -d_ci.hi, -d_ci.lo, d_ci.n)         # flip sign + bounds
        valid = np.isfinite(mk[sel]) & np.isfinite(q[sel])
        ww = q[sel][valid]
        rate = float(np.sum(ww[mk[sel][valid] < 0]) / np.sum(ww)) if ww.sum() > 0 else float("nan")
        out.append(BucketEdge(lab, n, m_ci, adverse, rate, nm))
    return out
