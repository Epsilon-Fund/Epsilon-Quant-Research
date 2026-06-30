"""
offline.py — synthetic series generators + the OFFLINE-ONLY ruptures wrapper.

`ruptures` uses the full series at once (it is a batch segmentation library), so
it is **lookahead-unsafe by construction** and must NEVER be wired into a live
path. It belongs here, for labelling/validation/benchmark ground truth only —
exactly the role the offline HMM plays for the regime-classifier. The online
detectors in detectors.py never import this module.
"""
from __future__ import annotations

import numpy as np

try:
    import ruptures as _rpt
    _HAS_RUPTURES = True
except Exception:  # pragma: no cover - exercised only when ruptures absent
    _rpt = None
    _HAS_RUPTURES = False


# ── synthetic generators (return (series, true_break_indices)) ───────────────
def make_stationary(n: int = 1000, mu: float = 0.0, sigma: float = 1.0,
                    seed: int = 0) -> tuple[np.ndarray, list[int]]:
    """A single stationary Gaussian segment — NO breaks (FPR ground truth)."""
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n), []


def make_noise(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, list[int]]:
    """Pure white noise — alias of stationary; the false-positive-rate series."""
    return make_stationary(n, 0.0, 1.0, seed)


def make_mean_shifts(n: int, breaks: list[int], deltas: list[float],
                     sigma: float = 1.0, seed: int = 0) -> tuple[np.ndarray, list[int]]:
    """Piecewise-constant mean with abrupt level shifts at `breaks`.
    `deltas[i]` is the mean of segment i (len(deltas) == len(breaks)+1)."""
    assert len(deltas) == len(breaks) + 1
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    bounds = [0, *breaks, n]
    for i in range(len(deltas)):
        s, e = bounds[i], bounds[i + 1]
        x[s:e] = rng.normal(deltas[i], sigma, e - s)
    return x, list(breaks)


def make_var_shifts(n: int, breaks: list[int], sigmas: list[float],
                    mu: float = 0.0, seed: int = 0) -> tuple[np.ndarray, list[int]]:
    """Piecewise volatility regime: same mean, abrupt variance shifts."""
    assert len(sigmas) == len(breaks) + 1
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    bounds = [0, *breaks, n]
    for i in range(len(sigmas)):
        s, e = bounds[i], bounds[i + 1]
        x[s:e] = rng.normal(mu, sigmas[i], e - s)
    return x, list(breaks)


def make_markov_switching(n: int = 2000, seed: int = 0,
                          means=(0.4, 0.0, -0.8),
                          sigmas=(0.6, 1.0, 2.2),
                          stay: float = 0.99):
    """A K-state Markov-switching Gaussian process — a controlled stand-in for
    HMM regime labels (with KNOWN transition points) for the Cohen's-κ benchmark.

    Returns (series, regime_labels, true_break_indices) where a break is any bar
    whose regime label differs from the previous bar's.
    """
    rng = np.random.default_rng(seed)
    K = len(means)
    off = (1.0 - stay) / (K - 1)
    P = np.full((K, K), off)
    np.fill_diagonal(P, stay)
    labels = np.empty(n, dtype=int)
    s = 0
    for t in range(n):
        if t > 0:
            s = rng.choice(K, p=P[s])
        labels[t] = s
    x = rng.normal(np.array(means)[labels], np.array(sigmas)[labels])
    breaks = [t for t in range(1, n) if labels[t] != labels[t - 1]]
    return x, labels, breaks


# ── ruptures (OFFLINE ONLY) ──────────────────────────────────────────────────
def ruptures_offline(series, model: str = "rbf", pen: float = 10.0,
                     min_size: int = 10) -> list[int]:
    """Batch changepoint segmentation via ruptures PELT. OFFLINE ONLY — uses the
    whole series at once; NEVER call this in a live/causal path. Returns interior
    break indices (the trailing n is dropped)."""
    if not _HAS_RUPTURES:
        raise RuntimeError(
            "ruptures is not installed. It is an offline-only validation dep; "
            "install with `uv pip install ruptures`. The online detectors "
            "(CUSUM/Page-Hinkley/BOCPD) do not need it.")
    x = np.asarray(series, dtype=float).reshape(-1, 1)
    algo = _rpt.Pelt(model=model, min_size=min_size).fit(x)
    bkps = algo.predict(pen=pen)
    return [b for b in bkps if b < len(x)]


def has_ruptures() -> bool:
    return _HAS_RUPTURES
