# data-contract (skill)

First-party Claude Code / Codex skill for the epsilon-quant-research repo. It
auto-triggers before a backtest/research run to validate a dataset's schema,
append-only / lookahead invariants, and drift — failing closed on any
violation.

- **Skill spec:** [`SKILL.md`](SKILL.md) (auto-trigger description + usage).
- **Engine + contracts (code):**
  - crypto: `infrastructure/data/schemas/` (`core.py`, `contracts.py`, `cli.py`, `tests/`)
  - Polymarket: `polymarket/research/data_infra/schemas/` (same layout)
- **Packaging:** symlinked into `.claude/skills/data-contract` and
  `~/.codex/skills/data-contract`, mirroring `efficient-fable` / `cost-mode`.
- **Registered in:** `brain/SKILL_MAP.md` § Runtime efficiency skills (runtime table + prompt pack).
- **Findings:** `polymarket/research/notes/overview/data_quality/data_contract_validation_layer_findings.md`.

Not vendored from upstream — built in-repo. Source commit recorded at the top
of `SKILL.md`.
