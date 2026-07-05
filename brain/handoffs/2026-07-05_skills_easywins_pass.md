---
title: "Handoff — Skills-lifecycle easy-wins pass: changepoint demo shipped, 4 proposals await go"
created: 2026-07-05
status: complete — builds done, proposals awaiting operator decision
owner: justin
project: infra
para: area
hubs:
  - SKILL_MAP
  - CODEX
  - TODO
tags:
  - handoff
  - skills
  - reflection
  - library
---

# Handoff — Skills-lifecycle easy-wins pass (2026-07-05)

> Hub: [[SKILL_MAP]] · law: [[CODEX]] · backlog: [[candidates]] · prior pass: [[2026-07-04_skills_lifecycle_phase1]]

## Plain-English Summary

- **What this is:** an on-demand exploratory pass over the skills-lifecycle backlog, one day after the kickoff pass — survey what's most worth packaging/developing next, build only the clearly-worthwhile low-risk items, stop at written proposals for anything bigger or public-facing.
- **Built:** (1) a runnable, self-verifying demo for the changepoint library package — the only scrub-approved public entry previously had no executable example (RC-023; 23 tests green, was 22); (2) the RC-009 post-merge hygiene step is now a checklist bullet in [[MERGE_PROTOCOL]] § 2.
- **Proposed, not built (operator go required):** RC-024 `rigorkit-calibrate` extraction (the Calibration Observatory is now a live public consumer — the "thread demands it" Phase-2 trigger), RC-025 `data-contract` extraction (its Python-3.10 blocker verified cleared), RC-026 Obsidian brain starter-kit (L-cost, largest scrub surface of any candidate), RC-027 reflection-prompt + PRD-scaffold skills.
- **Deliberately skipped:** a full discovery re-mining (last pass was yesterday — the weekly Monday pass stays the recurrence cadence) and anything publish-side (PyPI/repo-split/deploy remain human-gated per [[TODO]] § Skills Lifecycle).

## Builds (verified)

1. **RC-023 — changepoint demo.** `library/changepoint/examples/demo.py`: seeded synthetic calm→crisis→recovery daily-return series with two known breaks; scores CUSUM / Page-Hinkley / BOCPD against that truth (the honest read: CUSUM/PH catch the crisis break at lag 0 with zero false alarms but miss the subtler recovery break; BOCPD catches both, at lag [0, 6], at the cost of 6 extra flags); exercises all three integration helpers (`changepoint_features`, `fresh_break_gate`, `embargo_indices_from_breaks`); asserts a bar-by-bar `LiveDetector` run is identical to the batch run. Plus `tests/test_demo.py` (smoke test pinning the output story), a README § Runnable demo, and a SCRUB.md addendum (synthetic data only, literature-default parameters — no new content class; APPROVED verdict stands). Suite: **23 passed**.
2. **RC-009 (first half) — post-merge hygiene check.** [[MERGE_PROTOCOL]] § 2 now says: before pushing a merge (clean or conflicted path), run `python3 tools/brain_hygiene.py`, fix what's trivially merge-attributable, hand anything larger to a normal Janitor prompt. The merge-wrapper script stays planned — build only if the manual step gets skipped in practice.

## Proposals (short form here; full entries in [[candidates]] § 2026-07-05)

- **RC-024 `rigorkit-calibrate` (recommended next package).** Changepoint pattern verbatim: engine (`core.py` + markets layer) → `library/calibrate/`, both projects become same-API shims, `calibrate` SKILL.md bundled in-package, decoupling test, own SCRUB.md. The ledger state machine stays OUT (vendored MIT upstream — re-point, never re-host). Why now: the engine is already byte-identical across both projects (generalization proven), and the Calibration Observatory publicly leans on it.
- **RC-025 `data-contract`.** Blocker cleared (both engines parse under 3.10 grammar, verified via `ast.parse feature_version=(3,10)`; re-verify on a real 3.10 interpreter at packaging time). Sequence after RC-024 — heavier deps (`pandera.polars`) and the PM/crypto contracts need per-instrument genericization.
- **RC-026 Obsidian brain starter-kit.** High outside appeal, but a rewrite-and-generalize (law files embed strategy context), not an extract — L cost, needs its own scoping session.
- **RC-027 reflection-prompt + PRD-scaffold skills.** Prompt-ware generalizations of the reflection engine and the PRD→`/goal` flow; M cost; public-facing → scrub.

## Also verified this pass

- Catalog regen (`tools/skills_catalog.py`) produces timestamp-only drift vs the committed `library/catalog.json` — no content change since 07-04; reverted rather than committing noise.
- Radar pins from the newsagent thread recorded in [[candidates]] (forecasting-tools **Adopt**/MIT; ForecastBench **Borrow-pattern**; FinceptTerminal **Borrow-pattern, design-only**/AGPL) so future forecasting work doesn't re-triage — full table in [[newsagent_repo_data_radar_findings]].

## Next gates

- **Operator:** go/no-go on RC-024 (recommended), sequencing for RC-025→027; the standing release-mechanics gate (library name, PyPI/repo split, Vercel deploy) is unchanged.
- **Engine:** next scheduled reflection pass Monday 2026-07-06 09:31 — it should treat RC-023→027 as logged (never re-propose absent new evidence).
