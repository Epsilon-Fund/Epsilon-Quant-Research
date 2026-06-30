---
name: calibrate
description: >
  Score forecast/market calibration (Brier + Murphy decomposition, log-loss,
  ECE/MCE, reliability diagrams with bands, Spiegelhalter's Z,
  calibration-in-the-large) on the forecast ledger. Use when asked how
  well-calibrated a model or forecaster is, to compute a Brier/log-loss score,
  draw a reliability diagram, check over/under-confidence, recalibrate
  probabilities (isotonic / Platt), or compare model-p vs market-implied-p and
  track realized edge over resolved markets. Reads the forked superforecasting
  ledger read-only; never writes it.
---

<!--
Source: epsilon-quant-research @ commit 6f6eca0d5e1d46ec388401b304670ae1e6527a9e
(branch justin). First-party skill (NOT vendored). Built on top of the forked
superforecasting ledger (see .agents/skills/superforecasting/, upstream commit
8913b08). The engine is core.py, duplicated byte-identical into each project's
package; keep both copies in sync. Mirrors the data-contract packaging layout
(symlinks in .claude/skills and ~/.codex/skills).
-->

# Calibrate

A calibration scoring layer on top of the forked superforecasting ledger. It
answers "are these probabilities any good?" — not with an opinion, but with
proper scores, decompositions, and diagnostics. It is a **read-only consumer**
of the ledger: it never writes events and never re-implements the state machine.
The ledger's append-only log remains the single source of truth.

## When this triggers

Prompt-invoked when the task is about scoring or diagnosing probabilistic
forecasts or market prices: a Brier/log-loss score, a reliability diagram, an
over/under-confidence check, recalibration, or a model-vs-market edge read. It
does not fire on unrelated work.

## What it computes (as code, not assumed)

| metric | meaning |
|---|---|
| Brier score | mean squared error of forecasts; the headline proper score (lower better) |
| Murphy decomposition | Brier = reliability − resolution + uncertainty (+ residual); separates calibration from discrimination |
| log-loss | binary cross-entropy; punishes confident wrong calls harder than Brier |
| reliability table / diagram | predicted prob vs observed frequency per bin, with Wilson 95% bands; the calibration picture |
| ECE / MCE | expected / maximum calibration error across bins |
| Spiegelhalter's Z | hypothesis test of calibration-in-the-large (|Z|>1.96 ⇒ reject at p<0.05) |
| calibration-in-the-large | mean forecast vs base rate (systematic over/under-forecasting bias) |
| recalibration | isotonic (PAV) + Platt (logistic); mirrors `infrastructure/ml/walk_forward.py`'s `IsotonicRegression(out_of_bounds='clip')` pattern, with a pure-numpy fallback where sklearn is absent |
| markets layer | model-p vs (de-vigged) implied-p from decimal/American odds; realized edge over resolved markets |

Realism calibration is built in: the reliability table carries per-bin Wilson
bands, so an off-diagonal point whose band still straddles the diagonal is
**not** yet evidence of miscalibration — read the bands, not just the dots. The
Murphy decomposition reconciles exactly when grouping by unique forecast value;
binned, it reports the within-bin-variance `residual` rather than hiding it.

## How to run it (per project — NEVER cross-import the two)

The book selects which forked ledger to read (`--book`, else `$SF_BOOK` /
`$SF_LEDGER_DIR`). The two books share no state.

**Crypto** (`infrastructure.calibration`) — from the repo root, repo `.venv`:

```bash
PYTHONPATH=. uv run --no-project python -m infrastructure.calibration.cli --book crypto score
PYTHONPATH=. uv run --no-project python -m infrastructure.calibration.cli --book crypto table
PYTHONPATH=. uv run --no-project python -m infrastructure.calibration.cli --book crypto report --out docs/assets/crypto_calibration.png
```

**Polymarket** (`lib.calibration`) — from `polymarket/research/`:

```bash
cd polymarket/research
PYTHONPATH=. uv run python -m lib.calibration.cli --book polymarket score
PYTHONPATH=. uv run python -m lib.calibration.cli --book polymarket report --out data/analysis/plots/pm_calibration.png
```

`score` prints the scorecard (add `--json` for machine output); `table` prints
the calibration table; `report` writes a reliability-diagram PNG and prints the
scorecard. All three are read-only on the ledger.

## Using it as a library

```python
from infrastructure.calibration import core as cal      # crypto
# from lib.calibration import core as cal                # polymarket

cal.brier_score(prob, label)
cal.murphy_decomposition(prob, label, n_bins=None)       # exact identity (unique-value grouping)
cal.reliability_table(prob, label, n_bins=10)            # + Wilson bands
cal.spiegelhalter_z(prob, label)
p_cal = cal.isotonic_recalibrate(p_train, y_train, p_apply)   # reuse walk_forward pattern
cal.reliability_diagram({"model": (prob, label)}, "out.png")
cal.score_ledger(book="crypto")                          # one-call scorecard over the ledger
```

`calibration_table(prob, label, n_bins)` is a byte-for-byte reproduction of
`infrastructure/backtester/ml_metrics.py:calibration_table` (same equal-width
binning, same columns) — swap one for the other with no regression.

## Repo invariants (brain/CODEX.md)

uv only, never bare pip; the two projects have separate venvs and never
cross-import (the engine is a byte-identical copy per project, not a shared
import); all metrics lookahead-free; require CI/bands before calling a result
positive — which is why the reliability table ships Wilson bands. Ledger output
is git-ignored append-only runtime data; this layer never writes it.

See `docs/calibration_scoring_layer_findings.md` for design rationale, the
gate results, and the embedded reliability diagram.
