"""
core.py — calibration scoring layer (project-agnostic engine)
=============================================================
Score how good probabilistic forecasts are, on top of the forked
superforecasting ledger. Pure scoring/diagnostics: it makes NO judgments and
NEVER writes the ledger (read-only consumer). The ledger's append-only state
machine remains the single source of truth.

This file is duplicated **byte-identical** into each project's package
(``infrastructure/calibration/`` for crypto, ``polymarket/research/lib/calibration/``
for polymarket). The two books share no state — never cross-import. Keep the
two copies in sync.

Conventions reused from the crypto research stack (do not duplicate logic):
  - ``infrastructure/backtester/ml_metrics.py:calibration_table`` — equal-width
    ``pd.cut`` binning, columns ``prob_bucket / mean_pred_prob / actual_freq / n``,
    rounded to 3 dp. ``calibration_table`` below reproduces it exactly.
  - ``infrastructure/ml/walk_forward.py`` — isotonic recalibration via
    ``IsotonicRegression(out_of_bounds='clip')`` fit on a held-out set. The
    sklearn backend here mirrors that; a pure-numpy PAV fallback runs where
    sklearn is absent (e.g. the polymarket venv).

Dependencies: numpy + pandas + stdlib only on every gate-critical path.
matplotlib is imported lazily (plots only); sklearn is optional (recalibration
falls back to numpy). No scipy (Spiegelhalter's Z uses ``math.erf``).

Public API
----------
  Proper scores
    brier_score(prob, label)                      -> float
    log_loss(prob, label)                         -> float
  Decomposition / calibration error
    murphy_decomposition(prob, label, n_bins=None)-> dict   (Brier = REL - RES + UNC [+ residual])
    calibration_table(prob, label, n_bins=10)     -> pd.DataFrame   (reproduces ml_metrics)
    reliability_table(prob, label, n_bins=10)     -> pd.DataFrame   (+ Wilson bands)
    ece(prob, label, n_bins=10)                   -> float
    mce(prob, label, n_bins=10)                   -> float
  Tests / global calibration
    spiegelhalter_z(prob, label)                  -> dict   {z, p_value}
    calibration_in_the_large(prob, label)         -> dict   {mean_pred, base_rate, bias}
  Recalibration  (reuse walk_forward pattern)
    isotonic_recalibrate(p_tr, y_tr, p_apply=None, backend='auto')
    platt_recalibrate(p_tr, y_tr, p_apply=None, backend='auto')
  Markets layer
    implied_prob_decimal(odds) / implied_prob_american(ml)
    devig(probs)                                  -> np.ndarray
    market_edge(model_p, implied_p)               -> np.ndarray
    realized_edge(model_p, implied_p, outcome, ...) -> dict
  Plot / ledger
    reliability_diagram(curves, out_path, ...)    -> Path
    resolve_ledger_dir(book=None)                 -> Path
    load_scored_forecasts(book=None)              -> pd.DataFrame
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

_SQRT2 = math.sqrt(2.0)


# ── input coercion ──────────────────────────────────────────────────────────

def _as_arrays(prob, label) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to 1-D float arrays, drop NaN pairs, validate domain."""
    p = np.asarray(prob, dtype=float).ravel()
    y = np.asarray(label, dtype=float).ravel()
    if p.shape != y.shape:
        raise ValueError(f"prob and label length mismatch: {p.shape} vs {y.shape}")
    mask = ~(np.isnan(p) | np.isnan(y))
    p, y = p[mask], y[mask]
    if p.size == 0:
        raise ValueError("no non-NaN (prob, label) pairs")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    uy = set(np.unique(y).tolist())
    if not uy.issubset({0.0, 1.0}):
        raise ValueError(f"labels must be binary 0/1; got {sorted(uy)}")
    return p, y


# ── proper scores ─────────────────────────────────────────────────────────────

def brier_score(prob, label) -> float:
    """Mean squared error of probabilistic forecasts: mean((p - o)^2). Lower is better."""
    p, y = _as_arrays(prob, label)
    return float(np.mean((p - y) ** 2))


def log_loss(prob, label, eps: float = 1e-15) -> float:
    """Binary cross-entropy (natural log). Lower is better; clips to avoid inf."""
    p, y = _as_arrays(prob, label)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ── Murphy decomposition ───────────────────────────────────────────────────────

def murphy_decomposition(prob, label, n_bins: int | None = None) -> dict:
    """
    Murphy (1973) three-component decomposition of the Brier score:

        Brier = Reliability - Resolution + Uncertainty   (+ residual)

    REL  = (1/N) Σ n_k (f_k - o_k)^2   — how far bucket forecasts sit from the
           bucket's observed frequency (0 = perfectly calibrated; smaller better).
    RES  = (1/N) Σ n_k (o_k - ō)^2     — how much buckets separate from the base
           rate (larger better; a forecaster that always says ō has RES=0).
    UNC  = ō(1 - ō)                    — irreducible variance of the outcome.

    Grouping
    --------
    n_bins=None  → group by UNIQUE forecast values. The identity holds EXACTLY
                   (residual == 0); this is the form used by the reconciliation gate.
    n_bins=int   → equal-width bins. With forecasts that vary inside a bin the
                   three terms no longer sum exactly to Brier, so ``residual`` =
                   Brier - (REL - RES + UNC) is returned and is ~0 when forecasts
                   are near-constant within each bin. ``Brier == REL - RES + UNC +
                   residual`` is therefore always true by construction.

    Returns a dict with brier, reliability, resolution, uncertainty, residual,
    base_rate, n.
    """
    p, y = _as_arrays(prob, label)
    n = p.size
    base = float(np.mean(y))
    brier = float(np.mean((p - y) ** 2))

    if n_bins is None:
        # group by unique forecast value → exact identity
        groups = [np.where(p == v)[0] for v in np.unique(p)]
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        # right-closed bins so p==1.0 lands in the last bin
        idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
        groups = [np.where(idx == k)[0] for k in range(n_bins)]
        groups = [g for g in groups if g.size > 0]

    rel = res = 0.0
    for g in groups:
        nk = g.size
        fk = float(np.mean(p[g]))
        ok = float(np.mean(y[g]))
        rel += nk * (fk - ok) ** 2
        res += nk * (ok - base) ** 2
    rel /= n
    res /= n
    unc = base * (1.0 - base)
    residual = brier - (rel - res + unc)
    return {
        "brier": brier,
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "residual": float(residual),
        "base_rate": base,
        "n": int(n),
    }


# ── calibration table (reproduces ml_metrics.calibration_table) ─────────────────

def calibration_table(prob, label, n_bins: int = 10) -> pd.DataFrame:
    """
    Predicted probability vs observed frequency, per equal-width bin.

    Byte-for-byte reproduction of ``ml_metrics.calibration_table`` (same
    ``pd.cut`` equal-width binning, same ``observed=True`` grouping, same 4
    columns rounded to 3 dp) so research code can swap one for the other with
    no regression. A well-calibrated forecaster has ``mean_pred_prob ≈
    actual_freq`` in every row (points on the diagonal).
    """
    p, y = _as_arrays(prob, label)
    df = pd.DataFrame({"prob": p, "label": y})
    df["bin"] = pd.cut(df["prob"], bins=n_bins)
    rows = []
    for bin_label, grp in df.groupby("bin", observed=True):
        rows.append({
            "prob_bucket": str(bin_label),
            "mean_pred_prob": round(grp["prob"].mean(), 3),
            "actual_freq": round(grp["label"].mean(), 3),
            "n": len(grp),
        })
    return pd.DataFrame(rows)


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (no scipy needed)."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def reliability_table(prob, label, n_bins: int = 10) -> pd.DataFrame:
    """
    ``calibration_table`` plus per-bin Wilson 95% confidence bands on the
    observed frequency (``ci_lo`` / ``ci_hi``) — the bands drawn on the
    reliability diagram. Bins with small ``n`` get visibly wide bands, which is
    the honest reading: an off-diagonal point with a band that still straddles
    the diagonal is not yet evidence of miscalibration.
    """
    p, y = _as_arrays(prob, label)
    df = pd.DataFrame({"prob": p, "label": y})
    df["bin"] = pd.cut(df["prob"], bins=n_bins)
    rows = []
    for bin_label, grp in df.groupby("bin", observed=True):
        k = int(grp["label"].sum())
        nb = len(grp)
        lo, hi = _wilson_interval(k, nb)
        rows.append({
            "prob_bucket": str(bin_label),
            "mean_pred_prob": round(grp["prob"].mean(), 3),
            "actual_freq": round(grp["label"].mean(), 3),
            "n": nb,
            "ci_lo": round(lo, 3),
            "ci_hi": round(hi, 3),
        })
    return pd.DataFrame(rows)


def _bin_stats(p: np.ndarray, y: np.ndarray, n_bins: int):
    """Per-bin (count, mean_pred, obs_freq) over equal-width [0,1] bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for k in range(n_bins):
        g = np.where(idx == k)[0]
        if g.size == 0:
            continue
        out.append((g.size, float(np.mean(p[g])), float(np.mean(y[g]))))
    return out


def ece(prob, label, n_bins: int = 10) -> float:
    """Expected Calibration Error: n-weighted mean |mean_pred - obs_freq| over bins."""
    p, y = _as_arrays(prob, label)
    n = p.size
    return float(sum(cnt * abs(mp - of) for cnt, mp, of in _bin_stats(p, y, n_bins)) / n)


def mce(prob, label, n_bins: int = 10) -> float:
    """Maximum Calibration Error: worst |mean_pred - obs_freq| across bins."""
    p, y = _as_arrays(prob, label)
    stats = _bin_stats(p, y, n_bins)
    return float(max((abs(mp - of) for _, mp, of in stats), default=0.0))


# ── global-calibration statistics ───────────────────────────────────────────────

def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


def spiegelhalter_z(prob, label) -> dict:
    """
    Spiegelhalter's Z test of calibration-in-the-large (1986).

    Z = Σ (o_i - p_i)(1 - 2 p_i) / sqrt( Σ (1 - 2 p_i)^2 p_i (1 - p_i) )

    Under perfect calibration Z ~ N(0, 1). |Z| > 1.96 (two-sided p < 0.05)
    rejects calibration. Returns {z, p_value, n}.
    """
    p, y = _as_arrays(prob, label)
    num = np.sum((y - p) * (1.0 - 2.0 * p))
    den = np.sqrt(np.sum((1.0 - 2.0 * p) ** 2 * p * (1.0 - p)))
    if den == 0:
        return {"z": float("nan"), "p_value": float("nan"), "n": int(p.size)}
    z = float(num / den)
    p_value = float(2.0 * (1.0 - _norm_cdf(abs(z))))
    return {"z": z, "p_value": p_value, "n": int(p.size)}


def calibration_in_the_large(prob, label) -> dict:
    """Mean forecast vs base rate. bias = mean_pred - base_rate (>0 over-forecasting)."""
    p, y = _as_arrays(prob, label)
    mean_pred = float(np.mean(p))
    base = float(np.mean(y))
    return {"mean_pred": mean_pred, "base_rate": base, "bias": mean_pred - base, "n": int(p.size)}


# ── recalibration (reuse walk_forward isotonic pattern) ──────────────────────────

def _pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pool-Adjacent-Violators: fit a non-decreasing step function to (x, y).
    Pure numpy fallback for sklearn's IsotonicRegression. Returns (x_sorted,
    fitted_y) defining a monotone interpolation table.
    """
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order].astype(float)
    n = xs.size
    # weighted PAV
    vals = ys.copy()
    wts = np.ones(n)
    # store as blocks
    block_val = []
    block_wt = []
    for i in range(n):
        v, w = vals[i], wts[i]
        while block_val and block_val[-1] >= v:
            pv, pw = block_val.pop(), block_wt.pop()
            v = (pv * pw + v * w) / (pw + w)
            w = pw + w
        block_val.append(v)
        block_wt.append(w)
    # expand blocks back to per-point fitted values
    fitted = np.empty(n)
    j = 0
    for v, w in zip(block_val, block_wt):
        cnt = int(round(w))
        fitted[j:j + cnt] = v
        j += cnt
    return xs, fitted


def isotonic_recalibrate(p_tr, y_tr, p_apply=None, backend: str = "auto"):
    """
    Isotonic recalibration. Mirrors walk_forward.py's
    ``IsotonicRegression(out_of_bounds='clip')`` fit on a held-out set when
    sklearn is available; otherwise a pure-numpy PAV fit with clip-extrapolation.

    Returns recalibrated probabilities for ``p_apply`` (defaults to ``p_tr``).
    """
    p_tr = np.asarray(p_tr, dtype=float).ravel()
    y_tr = np.asarray(y_tr, dtype=float).ravel()
    p_ap = p_tr if p_apply is None else np.asarray(p_apply, dtype=float).ravel()

    if backend in ("auto", "sklearn"):
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_tr, y_tr)
            return np.asarray(iso.transform(p_ap), dtype=float)
        except ImportError:
            if backend == "sklearn":
                raise
    xs, fitted = _pav(p_tr, y_tr)
    # monotone interpolation with clip (matches out_of_bounds='clip')
    return np.interp(p_ap, xs, fitted, left=fitted[0], right=fitted[-1])


def platt_recalibrate(p_tr, y_tr, p_apply=None, backend: str = "auto", n_iter: int = 100):
    """
    Platt scaling: fit sigmoid(a·f + b) by logistic regression of the label on
    the raw forecast. sklearn ``LogisticRegression`` when available, else a
    pure-numpy IRLS fit. Returns recalibrated probabilities for ``p_apply``.
    """
    p_tr = np.asarray(p_tr, dtype=float).ravel()
    y_tr = np.asarray(y_tr, dtype=float).ravel()
    p_ap = p_tr if p_apply is None else np.asarray(p_apply, dtype=float).ravel()

    if backend in ("auto", "sklearn"):
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1e6, solver="lbfgs")
            lr.fit(p_tr.reshape(-1, 1), y_tr)
            return np.asarray(lr.predict_proba(p_ap.reshape(-1, 1))[:, 1], dtype=float)
        except ImportError:
            if backend == "sklearn":
                raise

    # pure-numpy IRLS for sigmoid(a*f + b)
    X = np.column_stack([p_tr, np.ones_like(p_tr)])
    w = np.zeros(2)
    for _ in range(n_iter):
        eta = X @ w
        mu = 1.0 / (1.0 + np.exp(-eta))
        s = np.clip(mu * (1.0 - mu), 1e-9, None)
        z = eta + (y_tr - mu) / s
        WX = X * s[:, None]
        try:
            w_new = np.linalg.solve(X.T @ WX + 1e-9 * np.eye(2), X.T @ (s * z))
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(w_new - w)) < 1e-10:
            w = w_new
            break
        w = w_new
    Xa = np.column_stack([p_ap, np.ones_like(p_ap)])
    return 1.0 / (1.0 + np.exp(-(Xa @ w)))


# ── markets layer (model-p vs implied-p) ─────────────────────────────────────────

def implied_prob_decimal(odds) -> np.ndarray:
    """Implied probability from decimal odds: 1/odds (still carries the vig)."""
    o = np.asarray(odds, dtype=float)
    if np.any(o <= 1.0):
        raise ValueError("decimal odds must be > 1.0")
    return 1.0 / o


def implied_prob_american(ml) -> np.ndarray:
    """Implied probability from American moneyline odds (vig included)."""
    m = np.asarray(ml, dtype=float)
    out = np.where(m < 0, (-m) / ((-m) + 100.0), 100.0 / (m + 100.0))
    return out.astype(float)


def devig(probs) -> np.ndarray:
    """
    Remove the bookmaker's overround by normalising raw implied probabilities so
    they sum to 1 (proportional / multiplicative method). For a binary market
    pass ``[p_yes, p_no]``; for an N-way market pass all N legs.
    """
    p = np.asarray(probs, dtype=float)
    s = p.sum()
    if s <= 0:
        raise ValueError("implied probabilities must sum to a positive number")
    return p / s


def market_edge(model_p, implied_p) -> np.ndarray:
    """Per-market edge = model probability − (de-vigged) implied probability."""
    return np.asarray(model_p, dtype=float) - np.asarray(implied_p, dtype=float)


def realized_edge(model_p, implied_p, outcome, side_threshold: float = 0.0) -> dict:
    """
    Realized edge over RESOLVED markets. We "back YES" on a market when
    ``model_p - implied_p > side_threshold``. Realized per-bet PnL at the
    implied (fair) price is ``outcome - implied_p`` (unit stake). Returns the
    expected (model) edge, the realized edge, the count of bets and their
    hit rate, plus the Brier of the model on the bet set.
    """
    mp = np.asarray(model_p, dtype=float).ravel()
    ip = np.asarray(implied_p, dtype=float).ravel()
    o = np.asarray(outcome, dtype=float).ravel()
    bet = (mp - ip) > side_threshold
    n_bet = int(bet.sum())
    if n_bet == 0:
        return {"n_markets": int(mp.size), "n_bets": 0, "expected_edge": float("nan"),
                "realized_edge": float("nan"), "hit_rate": float("nan"),
                "model_brier_on_bets": float("nan")}
    return {
        "n_markets": int(mp.size),
        "n_bets": n_bet,
        "expected_edge": float(np.mean(mp[bet] - ip[bet])),
        "realized_edge": float(np.mean(o[bet] - ip[bet])),
        "hit_rate": float(np.mean(o[bet])),
        "model_brier_on_bets": float(np.mean((mp[bet] - o[bet]) ** 2)),
    }


# ── reliability diagram ──────────────────────────────────────────────────────────

def reliability_diagram(curves: dict, out_path, n_bins: int = 10,
                        title: str = "Reliability diagram") -> Path:
    """
    Draw a reliability diagram for one or more forecasters and save a PNG.

    ``curves`` maps a label → (prob, label_array). Each forecaster is plotted as
    observed frequency vs mean forecast per bin, with Wilson 95% bands; the
    dashed 45° line is perfect calibration. A curve that bows BELOW the diagonal
    at high p and ABOVE it at low p is over-confident. ECE is shown in the legend.
    Uses the non-interactive Agg backend so it works headless. Returns the path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfect calibration")
    markers = ["o", "s", "^", "D", "v"]
    for i, (name, (p, y)) in enumerate(curves.items()):
        tbl = reliability_table(p, y, n_bins=n_bins)
        x = tbl["mean_pred_prob"].to_numpy()
        obs = tbl["actual_freq"].to_numpy()
        lo = obs - tbl["ci_lo"].to_numpy()
        hi = tbl["ci_hi"].to_numpy() - obs
        e = ece(p, y, n_bins=n_bins)
        ax.errorbar(x, obs, yerr=[lo, hi], fmt=markers[i % len(markers)] + "-",
                    capsize=3, lw=1.5, label=f"{name} (ECE={e:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean forecast probability (per bin)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ── ledger reader (read-only consumer of the forked superforecasting ledger) ─────

_BOOK_SUBPATHS = {
    "polymarket": ("polymarket", "research", "data", "superforecast"),
    "crypto": ("live_trading", "data", "superforecast"),
}


def _find_repo_root(start: Path | None = None) -> Path:
    """Walk up from this file until a directory containing .git is found."""
    here = (start or Path(__file__)).resolve()
    for parent in [here] + list(here.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("could not locate repo root (.git) above " + str(here))


def resolve_ledger_dir(book: str | None = None) -> Path:
    """
    Resolve the forked superforecasting ledger directory — same contract as
    ``sf.py``: ``$SF_LEDGER_DIR`` wins, else the book (arg or ``$SF_BOOK``) maps
    to its repo-relative ledger. The two books share no state, so this never
    guesses: with neither set it raises.
    """
    override = os.environ.get("SF_LEDGER_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    b = (book or os.environ.get("SF_BOOK", "")).strip().lower()
    if b in _BOOK_SUBPATHS:
        return _find_repo_root().joinpath(*_BOOK_SUBPATHS[b])
    raise SystemExit(
        "calibrate: no ledger selected. Pass book=polymarket|crypto, or set "
        "SF_BOOK=polymarket|crypto (or SF_LEDGER_DIR=<path>). The two books "
        "share no state, so this refuses to guess."
    )


def load_scored_forecasts(book: str | None = None) -> pd.DataFrame:
    """
    Read the ledger's append-only event log and return one row per SCORED
    forecast: ``id, timestamp, prob (=final_probability), label (=outcome),
    brier``. Read-only: it parses ``forecasts/events.jsonl`` and never writes.
    Empty DataFrame (with columns) if there are no scored forecasts yet.
    """
    ledger = resolve_ledger_dir(book)
    events_file = ledger / "forecasts" / "events.jsonl"
    cols = ["id", "timestamp", "prob", "label", "brier"]
    if not events_file.exists():
        return pd.DataFrame(columns=cols)
    rows = []
    for line in events_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if ev.get("type") != "scored":
            continue
        rows.append({
            "id": ev.get("id"),
            "timestamp": ev.get("timestamp"),
            "prob": ev.get("final_probability"),
            "label": ev.get("outcome"),
            "brier": ev.get("brier"),
        })
    return pd.DataFrame(rows, columns=cols)


def score_ledger(book: str | None = None, n_bins: int = 10) -> dict:
    """
    One-call scorecard over a book's scored forecasts: Brier (+ Murphy
    decomposition), log-loss, ECE/MCE, Spiegelhalter Z, calibration-in-the-large.
    Returns {n, ...metrics, table} — table is the reliability table. Raises a
    clear message if the ledger has no scored forecasts.
    """
    df = load_scored_forecasts(book)
    if df.empty:
        raise SystemExit(
            "calibrate: no SCORED forecasts in the ledger yet. Settle some "
            "forecasts with `sf settle` first."
        )
    p = df["prob"].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=float)
    return {
        "n": int(df.shape[0]),
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
        "murphy": murphy_decomposition(p, y, n_bins=n_bins),
        "ece": ece(p, y, n_bins=n_bins),
        "mce": mce(p, y, n_bins=n_bins),
        "spiegelhalter": spiegelhalter_z(p, y),
        "calibration_in_the_large": calibration_in_the_large(p, y),
        "table": reliability_table(p, y, n_bins=n_bins),
    }
