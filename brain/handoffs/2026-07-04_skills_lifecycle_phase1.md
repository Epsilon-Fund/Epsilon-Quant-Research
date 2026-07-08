---
title: "Skills Lifecycle Phase 1 — reflection engine live, changepoint-audit packaged into library/, catalog emitted"
created: 2026-07-04
status: complete — two human gates open (TCC grant, IP scrub)
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
  - infra
---

# Skills Lifecycle Phase 1 — reflection engine + library packaging proof

> Hub: [[SKILL_MAP]] · law: [[CODEX]] · tasks: [[TODO]] § Skills Lifecycle · backlog: [[candidates]] · prior radar: [[2026-06-28_external_repo_audit]]

## Plain-English Summary

- **What this is:** the kickoff run of the one-funnel skills lifecycle (discover → decide → build → validate → package → publish). Two deliverables on one pipeline: (A) the **reflection engine** — a standing discovery front that mines recent work for recurring pain and turns it into build decisions; (B) the **publish-end proof** — the already-validated `changepoint-audit` extracted into a decoupled top-level `library/` as a pip package + agent-skill bundle, which epsilon now consumes back.
- **Why:** recurring pain was being discovered ad hoc and rejected ideas kept resurfacing; and built-and-validated tooling stayed trapped inside epsilon with no path to the public Epsilon-website catalog.
- **What happened:** the first discovery pass logged 22 scored candidates (3 built the same day), the library Phase 0+1 shipped with all tests green standalone AND through epsilon's shim, a machine-readable catalog + local dashboard now generate from SKILL_MAP + library, and a weekly scheduled pass keeps the engine running.
- **Status:** complete for this run. Two **human** gates are open: the macOS TCC grant that un-blocks the nightly commit agent (RC-005), and the IP/strategy scrub that blocks any public publish.

## A. Reflection engine (discovery front)

**Design.** Weekly scheduled pass (`reflection-weekly`, Mondays 09:31 local, after the Monday hygiene scan so `brain/generated/` reports are fresh; runs while the desktop app is open) + on-demand invocation. Signals are gathered by **read-only subagents** (efficient-fable): git history, `scratch/codex` + `scratch/cowork`, session transcripts (grep-sampled, best-effort), and SKILL_MAP future-skills + generated hygiene/graph reports. Synthesis, scoring, and decisions stay central. Skill: `.agents/skills/reflection-engine/SKILL.md` (symlinked into `.claude/skills/` and `~/.codex/skills/`).

**Scoring rule (recorded, v1):** recurrence (LOW=1 / MED=2–3 / HIGH=4+) × build-cost (S/M/L); HIGH×(S|M)=build now, HIGH×L=plan, MED×S=build now only if it unblocks an automated system, LOW=nothing-with-reason. Full rubric + the practical example live in [[candidates]] § Scoring rubric.

**First pass (2026-07-04) — what the mining found.** Corpus: git log 06-01→07-04 (~8 pain clusters), 7 scratch docs, ~50 transcripts (146 MB, 06-29→07-04), hygiene/graph reports (clean except 4 dead-ends + 1 over-connected hub), 6-seed radar. Headline verified finds:

1. **RC-004 (built):** the nbstripout git filter was configured as `python3 -m nbstripout`, which resolves per-context; under launchd/sandbox PATHs `python3` → `/usr/bin/python3` (no nbstripout) → every notebook-touching git op aborts. This single mis-pin explains the 143 transcript mentions and the daily brief's 07-03 failure line. Fix: `tools/fix_nbstripout_filter.sh` pins the absolute interpreter into local git config (applied + verified on this clone; once per machine).
2. **RC-005 (blocked-human):** the launchd `brain-commit-push` job has failed **daily since ~06-13** — macOS TCC denies `/bin/bash` access to the repo under `~/Desktop` (`Operation not permitted` in `brain/generated/launchd.err`). No agent-side workaround exists for TCC; the human step is in [[TODO]] § Skills Lifecycle. This is the root cause of "brain markdown accumulates uncommitted for weeks".
3. **RC-006 (built):** `infrastructure/backtester/` and `infrastructure/ml/` lacked `__init__.py` (siblings all have one) → the ~20 intermittent ModuleNotFoundError transcript incidents. Added; imports verified.
4. **RC-007 (built):** opt-in guard pre-commit hook (`tools/git_hooks/` + installer): blocks committed merge-conflict markers (the 384151f incident), notebook outputs reaching the index (fires exactly when the nbstripout filter is broken), and CRLF in staged text (the 8-commit LF churn). 5-case test matrix green; installed on this clone. **Bonus find during install:** this clone's local `core.hooksPath` pointed at a nonexistent old-clone path (`~/Desktop/epsilon/github/...`) — git hooks were silently disabled here; the installer now detects and unsets stale hooksPaths.
5. Also observed while debugging: IDE/harness `git status` pollers + ~40 long-dirty notebooks + a required nbstripout filter = constant nbstripout respawn (three at 80–90 % CPU at one point) and transient `index.lock` collisions; a 3-day-stale `index.lock` (timestamped 01:00, the nightly-job window) was removed after confirming no live git process. Root cause is RC-005 (the un-committed pileup), not new machinery — folded into that entry's rationale.

Everything else — 4 planned items, 6 closed-nothing with reasons, and the radar-spawned candidates — is in [[candidates]]. The engine may never re-propose PACKAGING-stage, SKILL_MAP-deferred, or closed-nothing items absent new evidence.

## B. Library packaging proof (publish end)

**What shipped.** `library/changepoint/` = **`rigorkit-changepoint` v0.1.0** (library name *provisional* — final naming at the scrub checkpoint), license **Apache-2.0** (chosen over MIT for the patent grant + NOTICE mechanism — the right default for a company-published methodology library):

- **pip module** (`src/rigorkit/changepoint/`, PEP-420 namespace ready for future `rigorkit.*` siblings): the full engine (detectors / stream / integration / offline / evaluate / cli) extracted from `infrastructure/changepoint/` with epsilon-specific docstrings generalized and **zero changes to code bodies**. Deps: numpy + pandas only; extras `[parquet]`, `[offline]` (ruptures stays offline-only). Console script `rigorkit-changepoint`.
- **Agent-skill bundle shipped inside the package** (radar adoption RC-018/RC-019): `src/.../skills/changepoint-audit/SKILL.md` (agentskills.io-spec frontmatter) + `python -m rigorkit.changepoint.skills install|list|uninstall [--project|--global]` — the gs-quant installer pattern (Apache-2.0, credited in NOTICE). One artifact serves both distribution forms.
- **Tests:** the full 20-test suite (incl. the no-lookahead invariant) adapted + a new ast-based `test_decoupling.py` asserting the non-negotiable rule (no `infrastructure.*` / `polymarket.*` / `live_trading.*` / `topics.*` imports; only declared deps). **21 passed + 1 expected skip in a fresh venv with only the package installed.** CLI + installer smoke-tested end-to-end.
- **Dogfood (the decoupling proof):** `infrastructure/changepoint/` is now a same-API **shim** re-exporting `rigorkit.changepoint` (engine files deleted; epsilon context preserved in the shim docstring; clear ImportError remedy). Epsilon's 20 regression tests pass **through the shim**; the CLI invocation lines in [[SKILL_MAP]] and the findings note are unchanged. One-way dependency proven: epsilon consumes the library, never the reverse. Collaborators: `uv pip install -e "library/changepoint[dev]" --python .venv/bin/python` after pulling.
- **NOT published.** No repo split, no PyPI, no website — hard-blocked on the human IP/strategy scrub. Content reviewed for the scrub: methodology only — no strategy logic, thresholds-as-alpha, data paths, or addresses.

## C. Catalog + dashboard (website feed)

`tools/skills_catalog.py` (stdlib-only) parses `.agents/skills/*/SKILL.md`, `brain/SKILL_MAP.md` tables, and `library/*/pyproject.toml` (+ bundled skills) → **43 entries**:

- `library/catalog.json` — **committed** public-candidate feed for the web colleague; every entry carries `published: false` + `scrub_status: "pending-human-review"`.
- `brain/generated/skills_catalog.json` + `skills_dashboard.html` — full internal catalog + minimal local dashboard (git-ignored, regenerable; polished UX stays with the web colleague).

## D. External-repo radar (6 seeds, 2026-07-04)

Compact verdicts in [[candidates]] § External-repo radar; full evidence:

| seed | contains (verified) | licence | verdict → action taken |
|---|---|---|---|
| **anthropics/skills** | official Agent Skills repo: 17 skills, `spec/agent-skills-spec.md`, `template/`, marketplace layout | Mixed — examples Apache-2.0; **docx/pdf/pptx/xlsx skills proprietary source-available** | **Adopt** — spec-compliant frontmatter used for our library bundle (RC-018, done). Flag: never vendor the doc skills |
| **goldmansachs/gs-quant** | skills ship *inside* the pip wheel + `python -m gs_quant.skills install` CLI (`--global`/`--project`); repo `.claude/skills/` is contributor-only | Apache-2.0 | **Adopt (pattern)** — installer CLI reimplemented in our package (RC-019, done; credited in NOTICE) |
| **BuilderIO/skills** | 10 skills; we already lifted efficient-fable + stay-within-limits | MIT | **Adopt next lifts** — `agent-watchdog` (orchestrator audits implementer) + `efficient-frontier` → RC-020, planned vetting batch |
| **kepano/obsidian-skills** | 5 spec-compliant skills by the Obsidian CEO: obsidian-markdown, bases, json-canvas, obsidian-cli, defuddle | MIT | **Adopt when needed** — obsidian-markdown as syntax authority for brain passes; defuddle for radar capture → RC-021, planned |
| **backtester-mcp (bcosm)** | local-first backtester: `robustness.py` (PBO/CSCV, perturbation-PBO, bootstrap-Sharpe CI, deflated Sharpe), walk-forward w/ Optuna, DuckDB run registry, 13-tool MCP server | Apache-2.0 | **Borrow-pattern** — independent DSR/PBO cross-check vs our `overfitting_audit.py` + DuckDB run-registry idea → RC-022, pair with Phase-3 extraction. 2★/unreviewed: verify math vs López de Prado first |
| **"awesome-claude-fable-5"** | literal-name repos are SEO/promo — skip permanently. Canonical indexes: hesreallyhim/awesome-claude-code (48k★, higher signal) + ComposioHQ/awesome-claude-skills (67k★) | NOASSERTION / none | **Reference (standing radar)** — monthly re-scan wired into the reflection pass; thread-relevant entries pinned in [[candidates]] (Bedrock, Librarian-MCP, AI-Research-Skills, Skill-Seekers, cc-thinking-skills, usage-statusbar) |

No AGPL encountered; the only licence flag is the proprietary Anthropic document skills (reference-read only).

## Decisions taken (recorded)

1. **Rubric v1** as above — revisable by the engine, but every revision must be recorded in [[candidates]].
2. **Apache-2.0** for the library (patent grant + NOTICE); `rigorkit` as *provisional* name — both re-openable at the scrub checkpoint, nowhere else.
3. **Bundle-inside-pip** (gs-quant pattern) over separate bundle/module artifacts.
4. **Sequencing over fusion:** changepoint entered the funnel at PACKAGING as the worked example; the engine is barred from re-proposing it.
5. `candidates.md` is **tracked and authoritative for the backlog**; [[SKILL_MAP]] + `library/` remain canonical for built/published skills.

## Next gates

- **Human:** TCC grant (RC-005) · IP/strategy scrub + final library naming (blocks all publishing).
- **Engine (weekly, automatic):** next pass Monday 09:31; monthly radar re-scan; regenerate catalog each pass.
- **Phase 2 (on demand, not before):** package `calibrate`, then `data-contract`; Phase 3 heavy infra (overfitting harness, CPCV/WF). Vendor vetting batch: RC-013/020/021.
