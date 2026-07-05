---
name: calibrate
description: >
  Score forecast/market calibration (Brier + Murphy decomposition, log-loss,
  ECE/MCE, reliability diagrams with Wilson bands, Spiegelhalter's Z,
  calibration-in-the-large) on probability arrays or an append-only forecast
  ledger. Use when asked how well-calibrated a model or forecaster is, to
  compute a Brier/log-loss score, draw a reliability diagram, check
  over/under-confidence, recalibrate probabilities (isotonic / Platt), or
  compare model-p vs market-implied-p and track realized edge over resolved
  markets. Reads any ledger strictly read-only; never writes one.
license: Apache-2.0
---

# Calibrate

A calibration scoring layer backed by the `lemma-calibrate` Python package.
It answers "are these probabilities any good?" — not with an opinion, but with
proper scores, decompositions, and diagnostics. Its optional ledger reader is a
**read-only consumer**: it never writes events and never re-implements a
ledger's state machine.

## When to use

- Someone asks how well-calibrated a model, forecaster, or track record is.
- You need a Brier / log-loss score, a reliability diagram, or an
  over/under-confidence read.
- Probabilities need recalibration (isotonic or Platt) on a held-out set.
- You want model-p vs (de-vigged) market-implied-p and realized edge over
  resolved markets.

## Setup

```bash
# from git (no package registry — the repo is the distribution):
pip install "lemma-calibrate @ git+https://github.com/Epsilon-Fund/lemma.git#subdirectory=calibrate"
# extras: [plot] for reliability-diagram PNGs, [sklearn] for the sklearn
# recalibration backends (a pure-numpy fallback runs without it)
```

## What it computes (as code, not assumed)

| metric | meaning |
|---|---|
| Brier score | mean squared error of forecasts; the headline proper score (lower better) |
| Murphy decomposition | Brier = reliability − resolution + uncertainty (+ residual); separates calibration from discrimination |
| log-loss | binary cross-entropy; punishes confident wrong calls harder than Brier |
| reliability table / diagram | predicted prob vs observed frequency per bin, with Wilson 95% bands |
| ECE / MCE | expected / maximum calibration error across bins |
| Spiegelhalter's Z | hypothesis test of calibration-in-the-large (\|Z\|>1.96 ⇒ reject at p<0.05) |
| calibration-in-the-large | mean forecast vs base rate (systematic over/under-forecasting bias) |
| recalibration | isotonic (sklearn `IsotonicRegression(out_of_bounds='clip')` or pure-numpy PAV) + Platt (logistic / pure-numpy IRLS) |
| markets layer | model-p vs de-vigged implied-p from decimal/American odds; realized edge over resolved markets |

Read the Wilson bands, not just the dots: an off-diagonal point whose band
still straddles the diagonal is **not** yet evidence of miscalibration. The
Murphy decomposition reconciles exactly when grouping by unique forecast value;
binned, it reports the within-bin-variance `residual` rather than hiding it.

## How to run

On arrays:

```python
from lemma.calibrate import (
    brier_score, murphy_decomposition, reliability_table, ece,
    spiegelhalter_z, isotonic_recalibrate, reliability_diagram)

brier_score(prob, label)
murphy_decomposition(prob, label, n_bins=None)   # exact identity form
reliability_table(prob, label, n_bins=10)        # + Wilson ci_lo / ci_hi
p_cal = isotonic_recalibrate(p_train, y_train, p_apply)
reliability_diagram({"model": (prob, label)}, "reliability.png")
```

On a forecast ledger (a directory with `forecasts/events.jsonl`; rows with
`type == "scored"` carry `final_probability` / `outcome` — the
superforecasting-skill convention):

```bash
lemma-calibrate --ledger path/to/ledger score          # scorecard (--json for machine output)
lemma-calibrate --ledger path/to/ledger table          # pred prob vs observed freq
lemma-calibrate --ledger path/to/ledger report --out reliability.png
```

`$SF_LEDGER_DIR` substitutes for `--ledger`. All commands are read-only.

## Markets layer

```python
from lemma.calibrate import implied_prob_decimal, devig, market_edge, realized_edge

ip = devig(implied_prob_decimal([2.10, 1.85]))   # strip the overround
edge = market_edge(model_p, ip)                   # model prob − fair implied prob
realized_edge(model_p, ip, outcome)               # over RESOLVED markets: expected vs realized
```

## Discipline notes

- Proper scores only — no accuracy-at-50% shortcuts.
- Require bands/CIs before calling a forecaster miscalibrated (the reliability
  table ships Wilson bands for exactly this).
- Never write the ledger from a scoring pass; scoring and bookkeeping stay
  separate tools.
