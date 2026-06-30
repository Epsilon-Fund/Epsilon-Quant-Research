"""
integration.py — wire the causal break signal into the three consumers.

1. cpcv_engine embargo  : turn detected break timestamps into bar indices to
                          embargo around each break, to union into CPCV purging.
2. regime-classifier    : a causal `change_prob` / `run_length` feature column to
                          add to Stage-2 (XGBoost) features — it predicts next-day
                          regime, and "a structural break just fired" is a strong,
                          lookahead-free predictor of a regime transition.
3. trend-entry gate     : block new trend entries for a cooldown after a fresh
                          break (the new regime has not yet established its trend).

None of these import the regime-classifier or cpcv_engine — they return plain
arrays/Series/index sets the caller plugs in, keeping this module dependency-free
and the projects decoupled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stream import run_detector


def embargo_indices_from_breaks(index, break_timestamps, embargo_bars: int = 5,
                                symmetric: bool = True) -> np.ndarray:
    """Bar positions to embargo around each break, for CPCV purging.

    `index`            : the dataset's index (DatetimeIndex or RangeIndex)
    `break_timestamps` : iterable of break timestamps (must be in `index`)
    `embargo_bars`     : bars to embargo on each side of a break
    `symmetric`        : embargo both sides (default) or only forward (post-break)

    Returns a sorted unique int array of positions. Feed these as extra purge
    positions alongside `generate_cpcv_splits(..., purge_bars=...)`: a label
    spanning a structural break leaks regime information across the train/test
    boundary, so those bars should be dropped from training.
    """
    index = pd.Index(index)
    pos = []
    for ts in break_timestamps:
        try:
            i = index.get_loc(ts)
        except KeyError:
            continue
        if isinstance(i, slice):          # duplicate timestamps
            i = i.start
        lo = i - embargo_bars if symmetric else i
        hi = i + embargo_bars
        pos.extend(range(max(0, lo), min(len(index), hi + 1)))
    return np.array(sorted(set(pos)), dtype=int)


def changepoint_features(values, *, timestamps=None, name: str = "bocpd",
                         prefix: str = "cp", **detector_kwargs) -> pd.DataFrame:
    """Causal feature columns for regime-classifier Stage 2.

    Returns a DataFrame indexed by ts with `{prefix}_change_prob`,
    `{prefix}_run_length`, `{prefix}_flag`, `{prefix}_bars_since` — all computed
    from data <= t, safe to merge into the Stage-2 feature matrix (which predicts
    NEXT-day regime). Lag them by 1 day in the model if you want strict
    information-at-close semantics, exactly like the existing `regime_lag1`.
    """
    s = run_detector(values, timestamps=timestamps, name=name, **detector_kwargs)
    flags = s["cp_flag"].to_numpy()
    bars_since = np.empty(len(flags), dtype=np.int64)
    c = 10 ** 6
    for i, f in enumerate(flags):
        c = 0 if f else c + 1
        bars_since[i] = c
    return pd.DataFrame({
        f"{prefix}_change_prob": s["change_prob"].to_numpy(),
        f"{prefix}_run_length": s["run_length_mode"].to_numpy(),
        f"{prefix}_flag": flags.astype(int),
        f"{prefix}_bars_since": bars_since,
    }, index=s.index)


def fresh_break_gate(stream_df: pd.DataFrame, cooldown: int = 5) -> pd.Series:
    """Boolean allow-mask for trend entries: False for `cooldown` bars starting at
    each fresh break (and including the break bar), True otherwise.

    A trend-following entry right as a structural break fires is the worst time
    to add risk — the prior trend just ended and the new one has not established.
    Multiply your strategy `position` by this mask (like the regime filter does):

        df['position'] = df['position'] * fresh_break_gate(stream, cooldown=5).reindex(df.index).fillna(1)
    """
    flags = stream_df["cp_flag"].to_numpy()
    n = len(flags)
    allow = np.ones(n, dtype=bool)
    for i in np.where(flags)[0]:
        allow[i:min(n, i + cooldown)] = False
    return pd.Series(allow, index=stream_df.index, name="trend_allow")
