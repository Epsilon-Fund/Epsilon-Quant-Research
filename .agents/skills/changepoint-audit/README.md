# changepoint-audit (skill)

First-party, **prompt-invoked** Claude Code / Codex skill for the
epsilon-quant-research repo. A causal, lookahead-free structural-break detector
(CUSUM / Page-Hinkley / BOCPD) that complements — does not replace — the batch
HMM→XGBoost regime-classifier.

- **Skill spec:** [`SKILL.md`](SKILL.md) (prompt-invoke description + usage).
- **Code (crypto instance):** `infrastructure/changepoint/`
  (`detectors.py`, `stream.py`, `offline.py`, `evaluate.py`, `integration.py`, `cli.py`, `tests/`).
- **Packaging:** symlinked into `.claude/skills/changepoint-audit` and
  `~/.codex/skills/changepoint-audit`, mirroring `efficient-fable` / `data-contract`.
- **Registered in:** `brain/SKILL_MAP.md` (runtime skills table + prompt pack).
- **Findings:** `topics/regime-classifier/changepoint_detector_findings.md`.

Not vendored — built in-repo. Source commit recorded at the top of `SKILL.md`.
A Polymarket instance, if ever needed, gets its own copy under
`polymarket/research/` (never cross-import).
