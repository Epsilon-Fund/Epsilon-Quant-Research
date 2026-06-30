"""
stream.py — run a causal detector over a series, and the thin live hook.

`run_detector` feeds the series one bar at a time (so the output is causal by
construction) and returns the per-bar schema:
    {ts, cp_flag, run_length_mode, change_prob, statistic}

`LiveDetector` is the thin real-time hook: hold one detector, call
`.update(ts, x_t)` as each bar closes, get back one row. `append_changepoints`
persists rows append-only (new shard ⊇ old; never edits history) — the
brain/CODEX.md parquet invariant.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .detectors import StepResult, make_detector

_COLS = ["cp_flag", "run_length_mode", "change_prob", "statistic"]


def causal_standardize(values, warmup: int = 20, min_std: float = 1e-9) -> np.ndarray:
    """Expanding causal z-score: z_t = (x_t - mean(x_1:t)) / std(x_1:t).

    Lookahead-free (uses only data <= t). Brings any series onto a ~unit scale so
    the BOCPD Normal-Gamma prior is appropriate; CUSUM/PH already self-standardise
    so this is near-identity for them. The first `warmup` bars return 0 (baseline
    still forming)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    z = np.zeros(n)
    csum = 0.0
    csum2 = 0.0
    for i in range(n):
        csum += x[i]
        csum2 += x[i] * x[i]
        k = i + 1
        if k > warmup and k > 1:
            mean = csum / k
            var = max(csum2 / k - mean * mean, 0.0)
            std = max(math.sqrt(var), min_std)
            z[i] = (x[i] - mean) / std
    return z


def run_detector(values, detector=None, *, timestamps=None,
                 name: str = "bocpd", **detector_kwargs) -> pd.DataFrame:
    """Run a detector causally over `values`. Returns a DataFrame indexed by `ts`.

    Parameters
    ----------
    values        : 1-D array-like of observations (e.g. standardised log returns)
    detector      : a detector instance; if None, one is built from `name`+kwargs
    timestamps    : optional index (datetimes or ints); defaults to 0..n-1
    name          : detector name when `detector` is None ('cusum'|'page_hinkley'|'bocpd')
    """
    if detector is None:
        detector = make_detector(name, **detector_kwargs)
    x = np.asarray(values, dtype=float)
    n = len(x)
    idx = (pd.Index(timestamps, name="ts") if timestamps is not None
           else pd.RangeIndex(n, name="ts"))

    flags = np.empty(n, dtype=bool)
    rmode = np.empty(n, dtype=np.int64)
    cprob = np.empty(n, dtype=float)
    stat = np.empty(n, dtype=float)
    for i in range(n):
        r: StepResult = detector.update(x[i])
        flags[i] = r.cp_flag
        rmode[i] = r.run_length_mode
        cprob[i] = r.change_prob
        stat[i] = r.statistic

    out = pd.DataFrame(
        {"cp_flag": flags, "run_length_mode": rmode,
         "change_prob": cprob, "statistic": stat},
        index=idx,
    )
    out.attrs["detector"] = getattr(detector, "name", name)
    return out


def breaks_from_stream(stream_df: pd.DataFrame) -> list:
    """Timestamps (index values) where a break fired."""
    return list(stream_df.index[stream_df["cp_flag"].to_numpy()])


class LiveDetector:
    """Thin real-time hook. Construct once, call update(ts, x) per closed bar.

        live = LiveDetector("cusum", k=0.5, h=5.0)
        row = live.update(ts, x_t)   # -> dict {ts, cp_flag, run_length_mode, change_prob, statistic}

    State is causal: the row for bar t depends only on x_1..x_t."""

    def __init__(self, name: str = "bocpd", **detector_kwargs) -> None:
        self.detector = make_detector(name, **detector_kwargs)
        self.name = self.detector.name

    def update(self, ts, x: float) -> dict:
        r = self.detector.update(float(x))
        return {"ts": ts, "cp_flag": bool(r.cp_flag),
                "run_length_mode": int(r.run_length_mode),
                "change_prob": float(r.change_prob), "statistic": float(r.statistic)}


def append_changepoints(df: pd.DataFrame, path: str | Path) -> Path:
    """Append-only persist of a stream DataFrame to parquet. Existing rows are
    preserved; new `ts` rows are added (dedup on ts, keep first-seen). Never
    rewrites history — the brain/CODEX.md append-only invariant."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = df.reset_index()
    if "ts" not in new.columns:
        new = new.rename(columns={new.columns[0]: "ts"})
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts"], keep="first")
    else:
        combined = new
    combined = combined.sort_values("ts").reset_index(drop=True)
    tmp = path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path
