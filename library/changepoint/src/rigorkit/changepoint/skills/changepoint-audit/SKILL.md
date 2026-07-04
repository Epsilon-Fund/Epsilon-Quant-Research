---
name: changepoint-audit
description: >
  Detect structural breaks / regime shifts in a timestamped series,
  lookahead-free, for gating or purged-CV embargo. Use when you need a causal,
  real-time regime-shift signal (CUSUM / Page-Hinkley / BOCPD) on returns or any
  series — to feed a regime classifier, gate trend entries on a fresh break, or
  hand break timestamps to a purged cross-validation engine as embargo windows.
  Complements (does not replace) batch segmenters like HMM or ruptures.
license: Apache-2.0
---

# Changepoint Audit

A causal, lookahead-free structural-break detector, backed by the
`rigorkit-changepoint` Python package. Its state at time t is a pure function of
data ≤ t, so appending future bars never changes a past output (the no-lookahead
invariant, asserted in the package's tests). It **complements** batch regime
models (HMM/Viterbi, ruptures), which use the full history and cannot run live:
this runs bar by bar, in real time.

## When to use

- You need a **live** regime-shift signal (batch classifiers can't run causally).
- You want a causal `change_prob` feature for a regime-classification model.
- You want to **gate trend entries** right after a structural break (the worst time to add trend risk).
- You need **break timestamps** to embargo around in purged cross-validation.
- A series looks like it changed regime (mean or volatility) and you want it dated, lookahead-free.

## Setup

```bash
pip install rigorkit-changepoint            # or: uv pip install rigorkit-changepoint
# extras: [parquet] for parquet IO, [offline] for the batch ruptures wrapper
```

## Methods

| detector | cost/bar | catches | signal |
|---|---|---|---|
| `cusum` | O(1) — live first line | mean shifts | two-sided CUSUM on causally-standardised input |
| `page_hinkley` | O(1) — live first line | mean shifts | Page-Hinkley cumulative-deviation test |
| `bocpd` | O(rmax), bounded | mean **and** variance shifts | Adams & MacKay (2007) run-length posterior, Student-t predictive; emits a real `change_prob` = P(run just reset) |
| ruptures (offline) | batch | — | **OFFLINE ONLY** (labelling/validation); lookahead-unsafe, never wire it live |

Per-bar output schema: `{ts, cp_flag, run_length_mode, change_prob, statistic}`.

## How to run

```bash
# detect on a parquet series (log-returns), append-only output
rigorkit-changepoint detect prices.parquet --column Close --returns --standardize \
    --detector bocpd --out changepoints/prices_bocpd.parquet

# detection-lag / false-positive-rate benchmark on synthetic series
rigorkit-changepoint benchmark

# Cohen's kappa vs Markov-switching regime transitions
rigorkit-changepoint kappa-demo --tolerance 10
```

In code:

```python
from rigorkit.changepoint import run_detector, LiveDetector

# batch (causal) over a series -> per-bar DataFrame
stream = run_detector(values, name="bocpd", timestamps=index)

# thin real-time hook
live = LiveDetector("cusum", k=0.5, h=5.0)
row = live.update(ts, x_t)   # {ts, cp_flag, run_length_mode, change_prob, statistic}
```

## Integration (the three consumers)

```python
from rigorkit.changepoint import (
    changepoint_features, fresh_break_gate, embargo_indices_from_breaks)

# 1) regime model features — causal columns (merge into your feature matrix)
feats = changepoint_features(log_returns, timestamps=idx, name="bocpd")
#   -> cp_change_prob, cp_run_length, cp_flag, cp_bars_since  (lag by 1 bar for
#      strict information-at-close semantics)

# 2) gate trend entries on a fresh break (multiply into position)
allow = fresh_break_gate(stream, cooldown=5)
df["position"] = df["position"] * allow.reindex(df.index).fillna(1)

# 3) purged-CV embargo — bar positions to purge around each break
emb = embargo_indices_from_breaks(df.index, break_timestamps, embargo_bars=5)
#   union `emb` into your CV splitter's purge set
```

## Benchmarks (synthetic; reproduce with `benchmark` / `kappa-demo`)

- Mean-shift recovery: Page-Hinkley & BOCPD recall ≈ 1.0, median lag 1–4 bars; CUSUM ≈ 0.93.
- Variance-shift recovery: **BOCPD ≈ 1.0; CUSUM/PH ≈ 0.4** (mean detectors are blind to pure variance shifts — use BOCPD for vol regimes).
- False-alarm rate on pure noise: < 2 per 1000 bars for all three (stated bound: < 5/1000).
- κ vs Markov-switching transitions: BOCPD recovers ~67% of transitions within ±10 bars (81% at ±15); it also fires intra-regime breaks a coarse batch model smooths over, so precision/κ are modest by design.

## Notes

- Online path is pure numpy + stdlib `math` — no scipy/sklearn, so the live hook is dependency-light.
- Append-only parquet output: a new shard is a superset of the old; history is never rewritten.
- The no-lookahead invariant is the point: if you replace this detector, keep its test ("prefix run equals full-run prefix").
