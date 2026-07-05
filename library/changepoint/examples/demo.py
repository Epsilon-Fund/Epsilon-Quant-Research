"""End-to-end demo of rigorkit-changepoint on a synthetic regime-shift series.

Runs in a few seconds with no arguments and no data files:

    python examples/demo.py

What it shows, in order:

1. A synthetic daily-return series with two KNOWN structural breaks
   (calm -> crisis -> recovery), so detections can be scored against truth.
2. Causal standardisation + all three detectors (CUSUM, Page-Hinkley, BOCPD),
   with detection lag / misses / false alarms per detector via `match_breaks`.
3. The three integration helpers on the BOCPD stream:
   - `changepoint_features`      -> causal feature columns for a regime model
   - `fresh_break_gate`          -> trend-entry cooldown mask
   - `embargo_indices_from_breaks` -> purge positions for a purged-CV splitter
4. The `LiveDetector` streaming hook (one bar at a time), verified to agree
   with the batch run — the causal-state guarantee, demonstrated.

Everything is seeded and deterministic. Only numpy + pandas are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rigorkit.changepoint import (
    LiveDetector,
    breaks_from_stream,
    causal_standardize,
    changepoint_features,
    embargo_indices_from_breaks,
    fresh_break_gate,
    match_breaks,
    run_detector,
)

SEED = 7
TRUE_BREAKS = [250, 425]  # bar positions where the generating regime switches
WARMUP = 20               # causal_standardize warmup (its default)
# The causal scaler has not converged during its warmup, so detections in the
# first couple of warmup windows are artifacts — discard them when scoring.
SCORE_FROM = 2 * WARMUP


def make_series(seed: int = SEED) -> pd.Series:
    """Three regimes of synthetic daily returns with known switch points.

    calm      bars   0-249 : mu ~ +4 bps/day, sigma 1.0%/day
    crisis    bars 250-424 : mu ~ -20 bps/day, sigma 3.0%/day  (mean AND variance shift)
    recovery  bars 425-599 : mu ~ +10 bps/day, sigma 1.2%/day
    """
    rng = np.random.default_rng(seed)
    segments = [
        rng.normal(0.0004, 0.010, 250),
        rng.normal(-0.0020, 0.030, 175),
        rng.normal(0.0010, 0.012, 175),
    ]
    returns = np.concatenate(segments)
    ts = pd.date_range("2024-01-01", periods=len(returns), freq="D")
    return pd.Series(returns, index=ts, name="ret")


def detected_positions(stream: pd.DataFrame, index: pd.Index) -> list[int]:
    """Detected break bar positions, excluding the standardisation warmup."""
    pos = [index.get_loc(t) for t in breaks_from_stream(stream)]
    return [p for p in pos if p >= SCORE_FROM]


def main() -> None:
    ret = make_series()
    print(f"synthetic series: {len(ret)} daily returns, "
          f"true breaks at bars {TRUE_BREAKS} "
          f"({[str(ret.index[b].date()) for b in TRUE_BREAKS]})")

    # Detectors expect a roughly unit-scale input; standardise causally so the
    # scaling at bar t uses only bars <= t (no lookahead).
    z = causal_standardize(ret.to_numpy(), warmup=WARMUP)

    # --- 1. run all three detectors and score against the known truth --------
    print(f"\ndetector comparison (tolerance = 25 bars; detections in the "
          f"first {SCORE_FROM} warmup bars discarded):")
    print(f"  {'detector':<14} {'detected':>8} {'matched':>8} {'lags':>12} "
          f"{'missed':>7} {'false':>6}")
    streams = {}
    for name in ("cusum", "page_hinkley", "bocpd"):
        stream = run_detector(z, name=name, timestamps=ret.index)
        streams[name] = stream
        pos = detected_positions(stream, ret.index)
        score = match_breaks(TRUE_BREAKS, pos, tolerance=25)
        lags = [lag for _, _, lag in score["matches"]]
        print(f"  {name:<14} {len(pos):>8} {len(score['matches']):>8} "
              f"{str(lags):>12} {len(score['misses']):>7} "
              f"{len(score['false_alarms']):>6}")

    bocpd = streams["bocpd"]
    break_ts = [ret.index[p] for p in detected_positions(bocpd, ret.index)]

    # --- 2. regime-model features (causal, mergeable into any feature matrix) -
    feats = changepoint_features(z, timestamps=ret.index, name="bocpd")
    crisis_ts = ret.index[TRUE_BREAKS[0]]
    print(f"\nchangepoint_features around the first TRUE break "
          f"({crisis_ts.date()}, calm -> crisis):")
    window = feats.loc[crisis_ts - pd.Timedelta(days=2):
                       crisis_ts + pd.Timedelta(days=3)]
    print(window.round(3).to_string())
    print("  (run_length collapsing + cp_flag firing = the regime model's "
          "'a break just happened' feature)")

    # --- 3. trend-entry gate: stand aside while a fresh break settles ---------
    gate = fresh_break_gate(bocpd, cooldown=5)
    blocked = int((~gate).sum())
    print(f"\nfresh_break_gate(cooldown=5): trend entries blocked on "
          f"{blocked}/{len(gate)} bars "
          f"(multiply your position column by this mask)")

    # --- 4. purged-CV embargo: don't train across a structural break ----------
    purge = embargo_indices_from_breaks(ret.index, break_ts, embargo_bars=5)
    print(f"embargo_indices_from_breaks(embargo_bars=5): {len(purge)} bar "
          f"positions to add to a purged-CV splitter's purge set")

    # --- 5. the same thing, live: one closed bar at a time --------------------
    live = LiveDetector("cusum")
    live_flags = []
    for ts, x in zip(ret.index, z):
        row = live.update(ts, x)
        if row["cp_flag"]:
            live_flags.append(row["ts"])
    batch_flags = list(breaks_from_stream(streams["cusum"]))
    assert live_flags == batch_flags, "live and batch runs disagree!"
    first_real = next(t for t in live_flags
                      if ret.index.get_loc(t) >= SCORE_FROM)
    print(f"\nLiveDetector('cusum') streaming pass: {len(live_flags)} flags, "
          f"bar-for-bar IDENTICAL to the batch run (asserted) — first "
          f"post-warmup break flagged {first_real.date()}. State at t only "
          f"ever sees data <= t, so batch vs live can never disagree.")


if __name__ == "__main__":
    main()
