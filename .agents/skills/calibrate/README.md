# calibrate (skill)

First-party Claude Code / Codex skill: a **calibration scoring layer** on top of
the forked superforecasting ledger. Brier + Murphy decomposition, log-loss,
ECE/MCE, reliability diagrams with Wilson bands, Spiegelhalter's Z,
calibration-in-the-large, isotonic/Platt recalibration, and a model-vs-market
edge layer. Read-only consumer of the ledger — never writes it.

- **Skill definition:** [`SKILL.md`](SKILL.md) (source commit recorded at top).
- **Engine (byte-identical per project, never cross-imported):**
  - crypto — `infrastructure/calibration/` (`core.py`, `cli.py`, `tests/`)
  - polymarket — `polymarket/research/lib/calibration/` (same files)
- **Symlinks:** `.claude/skills/calibrate` and `~/.codex/skills/calibrate`
  → `.agents/skills/calibrate` (mirrors `data-contract` / `efficient-fable`).

Run (per project):

```bash
# crypto, from repo root
PYTHONPATH=. uv run --no-project python -m infrastructure.calibration.cli --book crypto score
# polymarket, from polymarket/research/
PYTHONPATH=. uv run python -m lib.calibration.cli --book polymarket score
```

Gates (DoD): Brier–Murphy decomposition reconciles on synthetic data; reliability
separates a well- vs over-confident forecaster; reproduces
`ml_metrics.calibration_table` on a shared input; the forked ledger still rejects
post-hoc mutation. Run the suite:

```bash
PYTHONPATH=. uv run --no-project python -m infrastructure.calibration.tests.test_calibration
```

Design rationale + embedded reliability diagram:
`docs/calibration_scoring_layer_findings.md`.
