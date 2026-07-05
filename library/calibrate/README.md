# lemma-calibrate

Proper-score calibration diagnostics for probabilistic forecasts: **Brier +
Murphy decomposition** (reliability − resolution + uncertainty), log-loss,
ECE/MCE, **reliability tables/diagrams with Wilson 95% bands**, Spiegelhalter's
Z, isotonic/Platt **recalibration**, and a **markets layer** (model-p vs
de-vigged implied-p, realized edge over resolved markets).

## Why this exists

"Was this forecaster any good?" deserves proper scores, not vibes. Two
disciplines are built in:

1. **Separate calibration from discrimination.** The Murphy decomposition
   splits the Brier score so an over-confident forecaster is punished in the
   *reliability* term while its *resolution* is credited — two different
   failure modes, never conflated. The decomposition reconciles **exactly**
   when grouping by unique forecast value; binned, the within-bin residual is
   reported rather than hidden.
2. **Bands before verdicts.** The reliability table carries per-bin Wilson 95%
   intervals: an off-diagonal point whose band still straddles the diagonal is
   *not yet* evidence of miscalibration. Read the bands, not just the dots.

## Install

No package registry — the repo is the distribution. Two ways in:

**Just the agent skill** (no Python install): copy the bundle into your
agent's skills directory:

```bash
cp -r src/lemma/calibrate/skills/calibrate  .claude/skills/
```

**The engine as a Python package** (deps: numpy + pandas only), straight from
git:

```bash
pip install "lemma-calibrate @ git+https://github.com/Epsilon-Fund/lemma.git#subdirectory=calibrate"
# extras: [plot] matplotlib reliability diagrams · [sklearn] sklearn recalibration backends
```

The recalibrators run without sklearn (pure-numpy PAV / IRLS fallbacks);
Spiegelhalter's Z needs no scipy (stdlib `math.erf`).

## Quick start

```python
from lemma.calibrate import (
    brier_score, murphy_decomposition, reliability_table,
    spiegelhalter_z, isotonic_recalibrate, reliability_diagram)

brier_score(prob, label)
murphy_decomposition(prob, label, n_bins=None)   # exact identity form
reliability_table(prob, label, n_bins=10)        # + Wilson ci_lo / ci_hi
spiegelhalter_z(prob, label)                     # {z, p_value, n}
p_cal = isotonic_recalibrate(p_train, y_train, p_apply)
reliability_diagram({"model": (prob, label)}, "reliability.png")
```

Markets layer:

```python
from lemma.calibrate import implied_prob_decimal, devig, market_edge, realized_edge

fair = devig(implied_prob_decimal([2.10, 1.85]))  # strip the overround
market_edge(model_p, fair)                        # model prob − fair implied prob
realized_edge(model_p, fair, outcome)             # over RESOLVED markets
```

## Forecast-ledger CLI (read-only)

Score an append-only forecast ledger — a directory containing
`forecasts/events.jsonl` where rows with `type == "scored"` carry
`final_probability` / `outcome` (the superforecasting-skill convention; any
tool emitting that shape works):

```bash
lemma-calibrate --ledger path/to/ledger score            # scorecard (--json)
lemma-calibrate --ledger path/to/ledger table            # pred vs observed
lemma-calibrate --ledger path/to/ledger report --out reliability.png
```

`$SF_LEDGER_DIR` substitutes for `--ledger`. The reader **never writes** the
ledger — scoring and bookkeeping stay separate tools, so a probability can
never be rewritten by its own scorer.

## Runnable demo

[`examples/demo.py`](examples/demo.py) is a seeded, self-contained walkthrough —
no data files, a couple of seconds:

```bash
python examples/demo.py
```

It scores a well-calibrated vs a deliberately over-confident synthetic
forecaster (showing the Murphy decomposition separating the two failure
modes), fixes the over-confident one with out-of-sample isotonic
recalibration, exercises the markets layer, and — with `[plot]` installed —
writes a three-curve reliability diagram.

## Agent skill bundle

The package ships a Claude-Code-compatible skill bundle
([Agent Skills spec](https://agentskills.io)) and an installer:

```bash
python -m lemma.calibrate.skills install --project   # ./.claude/skills/
python -m lemma.calibrate.skills install --global    # ~/.claude/skills/
```

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
