"""
evaluate.py — detection-lag / FPR / Cohen's-κ benchmarks for the detectors.

All metrics are causal-aware: a true break at index b is "detected" only by a
flag at index >= b (you cannot detect a break before it happens), within a
tolerance window.
"""
from __future__ import annotations

import numpy as np


def match_breaks(true_breaks: list[int], detected: list[int], tolerance: int = 25):
    """Match each true break to the FIRST detection in [b, b+tolerance].

    Returns dict with:
      matches        : list of (true_b, detected_b, lag)
      misses         : true breaks with no detection in window
      false_alarms   : detections not attributable to any true break window
    """
    detected = sorted(int(d) for d in detected)
    used = [False] * len(detected)
    matches, misses = [], []
    for b in sorted(int(t) for t in true_breaks):
        hit = None
        for j, d in enumerate(detected):
            if used[j]:
                continue
            if b <= d <= b + tolerance:
                hit = (b, d, d - b)
                used[j] = True
                break
        if hit:
            matches.append(hit)
        else:
            misses.append(b)
    false_alarms = [detected[j] for j in range(len(detected)) if not used[j]]
    return {"matches": matches, "misses": misses, "false_alarms": false_alarms}


def detection_metrics(true_breaks, detected, n, tolerance: int = 25) -> dict:
    """Recall, mean/median detection lag, and false-alarm rate (per bar &
    per-1000-bars). On a no-break series, recall is undefined (nan) and only the
    false-alarm rate is meaningful."""
    m = match_breaks(true_breaks, detected, tolerance)
    n_true = len(true_breaks)
    lags = [lag for *_, lag in m["matches"]]
    n_fa = len(m["false_alarms"])
    return {
        "n_true": n_true,
        "n_detected": len(detected),
        "recall": (len(m["matches"]) / n_true) if n_true else float("nan"),
        "mean_lag": float(np.mean(lags)) if lags else float("nan"),
        "median_lag": float(np.median(lags)) if lags else float("nan"),
        "n_false_alarms": n_fa,
        "far_per_bar": n_fa / n if n else float("nan"),
        "far_per_1000": 1000.0 * n_fa / n if n else float("nan"),
    }


def cohens_kappa(a, b) -> float:
    """Cohen's κ between two equal-length label arrays (binary or categorical).
    κ=1 perfect agreement, 0 chance-level, <0 worse than chance."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    labels = np.unique(np.concatenate([a, b]))
    po = float(np.mean(a == b))
    pe = 0.0
    n = a.size
    for k in labels:
        pe += (np.sum(a == k) / n) * (np.sum(b == k) / n)
    if pe >= 1.0:
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def _dilate(binary: np.ndarray, tolerance: int) -> np.ndarray:
    """Forward-dilate a 0/1 indicator by `tolerance` bars: mark [i, i+tol] for
    each set bit. Forward-only keeps the comparison causal (a detection within
    `tolerance` AFTER a transition counts as agreement)."""
    out = np.zeros_like(binary, dtype=bool)
    idx = np.where(binary)[0]
    n = len(binary)
    for i in idx:
        out[i:min(n, i + tolerance + 1)] = True
    return out


def kappa_vs_transitions(regime_labels, cp_flags, tolerance: int = 5) -> dict:
    """Cohen's κ between detector breaks and regime-label transitions.

    `regime_labels` : per-bar regime id (e.g. HMM `regime_online`, or a synthetic
                      Markov-switching label). A transition = label != previous.
    `cp_flags`      : per-bar boolean break flags from a detector.
    Both transition and flag indicators are forward-dilated by `tolerance` so a
    correctly-but-lagged detection counts as agreement. Returns κ + the raw
    agreement components.
    """
    labels = np.asarray(regime_labels)
    flags = np.asarray(cp_flags, dtype=bool)
    n = min(len(labels), len(flags))
    labels, flags = labels[:n], flags[:n]
    trans = np.zeros(n, dtype=bool)
    trans[1:] = labels[1:] != labels[:-1]
    a = _dilate(trans, tolerance)
    b = _dilate(flags, tolerance)
    # interpretable companions to kappa (which is depressed by the rarity of
    # transitions): how many transitions get a flag within tolerance (recall),
    # and how many flags land near a transition (precision).
    t_idx = np.where(trans)[0]
    f_idx = np.where(flags)[0]
    tr_recall = float(np.mean([
        np.any((f_idx >= t) & (f_idx <= t + tolerance)) for t in t_idx
    ])) if len(t_idx) else float("nan")
    fl_prec = float(np.mean([
        np.any((t_idx >= f - tolerance) & (t_idx <= f)) for f in f_idx
    ])) if len(f_idx) else float("nan")
    return {
        "kappa": cohens_kappa(a.astype(int), b.astype(int)),
        "n_transitions": int(trans.sum()),
        "n_flags": int(flags.sum()),
        "tolerance": tolerance,
        "agreement": float(np.mean(a == b)),
        "transition_recall": tr_recall,
        "flag_precision": fl_prec,
    }


def benchmark_detector(factory, *, seeds=range(5), tolerance: int = 25) -> dict:
    """Run a detector across a battery of synthetic series and average metrics.

    `factory` : zero-arg callable returning a FRESH detector each call.
    Returns per-scenario averaged metrics: 'mean_shift', 'var_shift', 'noise'.
    """
    from . import offline
    from .stream import breaks_from_stream, run_detector

    agg: dict[str, list] = {"mean_shift": [], "var_shift": [], "noise": []}
    for seed in seeds:
        # mean-shift: three abrupt level changes
        x, tb = offline.make_mean_shifts(1200, [300, 600, 900], [0, 2.5, -1.5, 1.0], seed=seed)
        s = run_detector(x, factory())
        agg["mean_shift"].append(detection_metrics(tb, _idx(breaks_from_stream(s)), len(x), tolerance))
        # variance-shift
        xv, tbv = offline.make_var_shifts(1200, [400, 800], [0.5, 2.5, 0.8], seed=seed)
        sv = run_detector(xv, factory())
        agg["var_shift"].append(detection_metrics(tbv, _idx(breaks_from_stream(sv)), len(xv), tolerance))
        # pure noise (false-positive rate)
        xn, _ = offline.make_noise(2000, seed=seed)
        sn = run_detector(xn, factory())
        agg["noise"].append(detection_metrics([], _idx(breaks_from_stream(sn)), len(xn), tolerance))

    return {scen: _avg(rows) for scen, rows in agg.items()}


def _idx(breaks):
    """Coerce stream index values (RangeIndex ints) to plain ints."""
    return [int(b) for b in breaks]


def _avg(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r[k] == r[k]]  # drop nan
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
