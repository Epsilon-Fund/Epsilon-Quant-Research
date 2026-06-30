---
name: changepoint-audit
description: >
  Detect structural breaks / regime shifts in a timestamped series,
  lookahead-free, for gating or CPCV embargo. Prompt-invoke when you need a
  causal, real-time regime-shift signal (CUSUM / Page-Hinkley / BOCPD) on
  returns or any series — to feed the regime-classifier, gate trend entries on a
  fresh break, or hand break timestamps to the CPCV engine as embargo windows.
  Complements (does not replace) the batch HMM→XGBoost regime-classifier.
---

<!--
Source: epsilon-quant-research @ commit 6f6eca0d5e1d46ec388401b304670ae1e6527a9e
(branch justin). First-party skill (NOT vendored). Prompt-invoked (soft) — it is
NOT an auto-trigger safety gate. Code lives in infrastructure/changepoint/
(crypto instance). A Polymarket instance, if needed, gets its OWN copy under
polymarket/research/ — never cross-import (brain/CODEX.md). Packaging mirrors
efficient-fable / data-contract: symlinks in .claude/skills and ~/.codex/skills.
-->

# Changepoint Audit

A causal, lookahead-free structural-break detector. Its state at time t is a
pure function of data ≤ t, so appending future bars never changes a past output
(the no-lookahead invariant, asserted in tests). It **complements**
`topics/regime-classifier/` (batch HMM→XGBoost, which cannot run live): the HMM's
Viterbi labels use the full history; this runs bar by bar, in real time.

## When to prompt-invoke

- You need a **live** regime-shift signal (the regime-classifier is batch-only).
- You want a causal `change_prob` feature for the regime-classifier's Stage-2 model.
- You want to **gate trend entries** right after a structural break (worst time to add trend risk).
- You need **break timestamps** to embargo around in `infrastructure/walkforward/cpcv_engine.py`.
- A series looks like it changed regime (mean or volatility) and you want it dated, lookahead-free.

## Methods

| detector | cost/bar | catches | signal |
|---|---|---|---|
| `cusum` | O(1) — live first line | mean shifts | two-sided CUSUM on causally-standardised input |
| `page_hinkley` | O(1) — live first line | mean shifts | Page-Hinkley cumulative-deviation test |
| `bocpd` | O(rmax), bounded | mean **and** variance shifts | Adams & MacKay (2007) run-length posterior, Student-t predictive; emits a real `change_prob`=P(run just reset) |
| ruptures (offline) | batch | — | **OFFLINE ONLY** (labelling/validation); lookahead-unsafe, never wired live |

Per-bar output schema: `{ts, cp_flag, run_length_mode, change_prob, statistic}`.

## How to run (crypto instance, from the repo root)

```bash
# detect on a parquet series (BTC daily log-returns), append-only output
PYTHONPATH=. .venv/bin/python -m infrastructure.changepoint.cli detect \
    live_trading/cache/daily/BTCUSDT_daily.parquet --column Close --returns --standardize \
    --detector bocpd --out infrastructure/changepoint/changepoints/btc_daily_bocpd.parquet

# detection-lag / false-positive-rate benchmark on synthetic series
PYTHONPATH=. .venv/bin/python -m infrastructure.changepoint.cli benchmark

# Cohen's kappa vs Markov-switching regime transitions (HMM stand-in)
PYTHONPATH=. .venv/bin/python -m infrastructure.changepoint.cli kappa-demo --tolerance 10
```

In code:

```python
from infrastructure.changepoint import run_detector, LiveDetector

# batch (causal) over a series -> per-bar DataFrame
stream = run_detector(values, name="bocpd", timestamps=index)

# thin real-time hook
live = LiveDetector("cusum", k=0.5, h=5.0)
row = live.update(ts, x_t)   # {ts, cp_flag, run_length_mode, change_prob, statistic}
```

## Integration (the three consumers)

```python
from infrastructure.changepoint import (
    changepoint_features, fresh_break_gate, embargo_indices_from_breaks)

# 1) regime-classifier Stage 2 — causal feature columns (merge into the feature matrix)
feats = changepoint_features(log_returns, timestamps=idx, name="bocpd")
#   -> cp_change_prob, cp_run_length, cp_flag, cp_bars_since  (lag by 1 like regime_lag1)

# 2) gate trend entries on a fresh break (multiply into position, like the regime filter)
allow = fresh_break_gate(stream, cooldown=5)
df["position"] = df["position"] * allow.reindex(df.index).fillna(1)

# 3) CPCV embargo — bar positions to purge around each break
emb = embargo_indices_from_breaks(df.index, break_timestamps, embargo_bars=5)
#   union `emb` into the training-side purge in generate_cpcv_splits(...)
```

## Benchmarks (synthetic; reproduce with `cli benchmark` / `kappa-demo`)

- Mean-shift recovery: Page-Hinkley & BOCPD recall ≈ 1.0, median lag 1–4 bars; CUSUM ≈ 0.93.
- Variance-shift recovery: **BOCPD ≈ 1.0; CUSUM/PH ≈ 0.4** (mean detectors are blind to pure variance shifts — use BOCPD for vol regimes).
- False-alarm rate on pure noise: < 2 per 1000 bars for all three (stated bound: < 5/1000).
- κ vs regime transitions: BOCPD recovers ~67% of transitions within ±10 bars (81% at ±15); it also fires intra-regime breaks the coarse HMM smooths over (so precision/κ are modest by design). See the findings note for the full where-it-helps/hurts read.

## Notes

- Online path is pure numpy + stdlib `math` — no scipy/sklearn, so the live hook is dependency-light. `ruptures` (offline only) + scipy are needed solely for the offline validation path.
- Append-only output (`changepoints/*.parquet`): new shard ⊇ old, never edits history (brain/CODEX.md).
- Findings + design rationale: `topics/regime-classifier/changepoint_detector_findings.md`.
