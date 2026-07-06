---
title: "Reflection Engine — Candidates Backlog"
created: 2026-07-04
status: active
owner: justin
project: infra
para: area
hubs:
  - SKILL_MAP
  - CODEX
  - TODO
tags:
  - reflection
  - skills
  - backlog
  - infra
---

# Reflection Engine — Candidates Backlog

> Hub: [[SKILL_MAP]] · law: [[CODEX]] · tasks: [[TODO]]
> **This file is the authoritative backlog of the skills-lifecycle discovery front.** The reflection engine (weekly pass + on-demand, see [[SKILL_MAP]] § reflection-engine) writes candidates here; accepted candidates graduate along the pipeline below. [[SKILL_MAP]] and `library/` stay canonical for *built* and *published* skills — this file only feeds them.

## Plain-English Summary

- This is the standing backlog where recurring pain observed across agent sessions, git history, scratch lanes, and generated reports is clustered, scored, and turned into a decision: **new skill / automation / fix / nothing**.
- Every entry — including "nothing" — carries a reason, so a rejected idea never re-surfaces as if it were new.
- Populated by the reflection engine's discovery passes; first pass ran 2026-07-04 (git history since 2026-06-01, scratch lanes, ~50 session transcripts since 06-29, hygiene/graph reports, SKILL_MAP future-skills, external-repo radar).
- Current state after the 2026-07-05 rebrand/hardening pass: the library brand is **lemma** (renamed from provisional `rigorkit`; **no PyPI** — distribution is the public repo itself, copy-from-repo bundles + optional git-install). Two packages live (`lemma-changepoint`, `lemma-calibrate` — both scrub-APPROVED) plus two standalone bundles from RC-027 (`reflection-prompt`, `prd-scaffold` — scrubs PENDING). RC-025 data-contract stays next-in-line; RC-026 was reframed to a written setup-guide article and shelved on the docket. Remaining publish steps are human-only: bundle scrub sign-offs, the `Epsilon-Fund/lemma` repo split, the website catalog copy + deploy.

## Scoring rubric (the recorded rule)

Score = **recurrence × build-cost**, decided per cluster (not per raw signal).

| Axis | Bands |
|---|---|
| **Recurrence** | LOW = seen once · MED = 2–3 distinct commits/sessions · HIGH = 4+ distinct commits/sessions (or a scheduled system failing repeatedly) |
| **Build-cost** | S = ≤ ~1 focused agent-hour · M = ~half-day, touches conventions or several files · L = multi-day / new subsystem |

Decision rule used (v1, 2026-07-04 — the engine may revise this rubric; any revision must be recorded here):

- HIGH × (S|M) → **build now**
- HIGH × L → **plan** (schedule, don't start)
- MED × S → **build now** only if it unblocks an automated system or a live thread; else **watch**
- MED × (M|L) → **plan** or **watch**
- LOW × anything → **nothing (watch)** — unless it hard-blocks an automated pipeline, then **fix**
- Items entering at a later lifecycle stage (e.g. already built, awaiting packaging) are **not re-scored** by discovery; they enter the funnel at their stage.

**Practical example:** the nbstripout git-filter failure (RC-004) appeared in 143 transcript lines, the daily brief, and the commit-agent's design-abort path — HIGH recurrence. The fix is pinning one absolute interpreter path in local git config plus a doctor script — S cost. HIGH × S → build now. By contrast, the `PYTHONPATH=. uv run` prefix tax (RC-008) is even more frequent, but the remedy touches the run-environment law in [[CODEX]] and every prompt convention — M/L cost with coordination risk — so it is **plan**, not build-now.

## Graduation pipeline

```
candidate (here) → built (code + passing tests) → registered ([[SKILL_MAP]], with an invocation line)
                → library entry (generalized, decoupled, IP-scrubbed, licensed → library/)
                → published (Epsilon website catalog — BLOCKED on the human IP/strategy scrub)
```

Statuses used below: `proposed` · `accepted` · `in-build` · `built` · `registered` · `packaging` · `library` · `published` · `blocked-human` · `planned` · `watch` · `closed-nothing`.

## Column glossary

| Column | Meaning |
|---|---|
| id | stable candidate id (RC-nnn), never reused |
| pain / cluster | the recurring problem, in plain English |
| evidence | where the signal came from (commits, transcript counts, reports) |
| rec | recurrence band (LOW/MED/HIGH per rubric) |
| cost | build-cost band (S/M/L per rubric) |
| decision | new skill / automation / fix / nothing |
| status | lifecycle status (see pipeline above) |

## Candidates — 2026-07-04 discovery pass

### Lifecycle deliverables (enter at their stage; not re-scored)

| id | title | pain / cluster | evidence | rec | cost | decision | status |
|---|---|---|---|---|---|---|---|
| RC-001 | changepoint-audit → `library/` extraction | Built-and-validated skills stay trapped inside epsilon; no public-library packaging proof exists | skills_library_build_plan (scratch/cowork), 20 tests green in `infrastructure/changepoint/` | — | M | **package** (entered at PACKAGING; discovery must never re-propose it) | **library** (2026-07-04: `library/changepoint/` = `rigorkit-changepoint` v0.1.0, Apache-2.0, 21+1 tests green standalone, epsilon consumes it back via the `infrastructure/changepoint` shim — 20 regression tests green. **Scrub APPROVED same day** — operator-delegated, recorded in `library/changepoint/SCRUB.md`; `/library` page shipped to the epsilon-webs1te repo. Remaining before "published": final name decision + PyPI/repo-split + Vercel deploy. **2026-07-05:** renamed `lemma-changepoint`; name + registry questions RESOLVED — brand `lemma`, NO PyPI, repo-is-the-distribution; remaining = repo split + website copy/deploy) |
| RC-002 | reflection-engine skill + weekly pass | Recurring pain is discovered ad-hoc; no standing discovery front; re-explained context and redone steps across sessions | fable_projects_1_2_plan (scratch/cowork); this very pass | — | M | **new skill** | **registered** (2026-07-04: `.agents/skills/reflection-engine/` + symlinks; scheduled task `reflection-weekly`, Mondays 09:31) |
| RC-003 | machine-readable skills catalog + minimal local dashboard | SKILL_MAP is prose-only; the website colleague needs a machine-readable feed; no single view of skill lifecycle state | GOAL mandate; skills_features_library vision (scratch/cowork) | — | S/M | **automation** | **built** (2026-07-04: `tools/skills_catalog.py` → `library/catalog.json` public-candidate feed + `brain/generated/skills_catalog.json` + `skills_dashboard.html`) |

### Accepted — build now (2026-07-04)

| id | title | pain / cluster | evidence | rec | cost | decision | status |
|---|---|---|---|---|---|---|---|
| RC-004 | Pin nbstripout filter to an absolute interpreter | `filter.nbstripout.clean = python3 -m nbstripout` breaks in every minimal-PATH context (launchd, sandboxed agents): `/usr/bin/python3` has no nbstripout → git operations on notebooks abort; daily brief 2026-07-03 flagged it; commit agent aborts by design | 143 transcript mentions across 5 sessions; daily_brief 07-03; verified live: `/usr/bin/python3 -c "import nbstripout"` → ModuleNotFoundError while framework 3.14 python works | HIGH | S | **fix** (doctor script pins absolute path in local git config) | **built** (2026-07-04: `tools/fix_nbstripout_filter.sh`, applied + verified on this clone; run once per machine) |
| RC-006 | Add missing `__init__.py` to `infrastructure/backtester/` + `infrastructure/ml/` | Implicit-namespace-package gaps cause intermittent ModuleNotFoundError under pytest/tooling; siblings (`validation/`, `walkforward/`, `changepoint/`) all have `__init__.py` | ~20 ModuleNotFoundError transcript incidents; verified missing on disk 2026-07-04 | MED | S | **fix** (unblocks tooling; consistent with siblings) | **built** (2026-07-04: both added, imports verified) |
| RC-007 | Git guard hooks: conflict-marker + notebook-output + CRLF pre-commit check | Committed merge-conflict markers shipped 6 unparseable files (fixed 384151f); LF churn produced 8 normalization commits in 2 weeks; nothing guards against a repeat | commits 384151f, 2c9e31b, c515e4b, 9f3a7af, 4c872c0, 4580634, 4f41307; transcript "merge conflict" ×91 | HIGH | S | **automation** (opt-in `tools/git_hooks/` + installer; hooks are per-clone by design) | **built** (2026-07-04: 5-case test matrix green in isolation; installed on this clone. Bonus find: this clone's local `core.hooksPath` pointed at a nonexistent old-clone path — hooks were silently disabled; installer now detects+unsets that) |

### Human-gated

| id | title | pain / cluster | evidence | rec | cost | decision | status |
|---|---|---|---|---|---|---|---|
| RC-005 | ~~Un-block the launchd `brain-commit-push` job (macOS TCC)~~ | The 01:00 daily job has failed since ~06-13 (`Operation not permitted` under TCC). Originally scored as a fix | `brain/generated/launchd.err`; log last success 2026-06-13 | HIGH | S | **closed — SCRAPPED by operator** (2026-07-04, Justin: "that task has been scrapped a long time ago"). The dead job is intentional, not a bug — never re-propose. The mis-lead is itself a lesson: the daily brief still flags the job as broken; brain sync is manual-cadence now | closed-nothing |

### Planned (real, but not this run)

| id | title | pain / cluster | evidence | rec | cost | decision | status |
|---|---|---|---|---|---|---|---|
| RC-008 | Kill the `PYTHONPATH=. uv run` prefix tax | The per-project run incantation is retyped constantly and mis-remembered across projects | ~968 PYTHONPATH + ~1031 `uv run` transcript mentions | HIGH | M/L | **plan** — remedy touches the run-environment law in [[CODEX]] + both pyprojects (e.g. proper editable installs or Make targets); do it as a deliberate conventions change, not a drive-by | planned |
| RC-009 | Post-merge hygiene validator | Every main-integration merge imports link/frontmatter debt that a later janitor pass mops up | commits 7703de6, cd3bdca, fa8dd20 | MED | M | **plan** — add a `brain_hygiene.py` run to the [[MERGE_PROTOCOL]] merge checklist; candidate for a merge-wrapper script | **half-built** (2026-07-05: the S-cost checklist step is now in [[MERGE_PROTOCOL]] § 2 — post-merge hygiene check before pushing, fix only merge-attributable debt. The merge-wrapper script stays planned; build it only if the manual step proves to be skipped in practice) |
| RC-013 | Vendor reasoning/causal skill packs (causal-inference, critical-thinking, thinking-frameworks subset) | Planned Track-1 installs from 2026-06-18 incorporation scope; never executed | scratch/cowork/2026-06-18_incorporation_scope.md | LOW (planned work, not observed pain) | M each | **plan** — needs the per-file vetting pass (chrome-vetting handoff pattern); batch into a dedicated vendoring session | planned |
| RC-014 | Fold specification-curve + empirical-null into `overfitting_audit.py` | Track-2 method fold; adjacent to existing DSR/PBO/White arsenal | scratch/cowork/2026-06-18_whitespace_build_specs.md | LOW | M | **plan** — only when a live research question needs it (no infra before signal) | planned |

### Closed — nothing (with reasons, so they never re-surface)

| id | title | why nothing |
|---|---|---|
| RC-010 | macOS `timeout` wrapper | The 367 "timeout" hits are dominated by agent-session process management, which the harness's own Bash timeout parameter already covers; a shell wrapper would help only rare human-terminal use. Re-open only if a scheduled job (not an agent) is bitten. |
| RC-011 | Skill-vendoring runner/automation | Vendoring happened twice in a month and REQUIRES human license/security vetting each time by law; automating the mechanical copy saves minutes and risks skipping the vetting. Stays manual. |
| RC-012 | Grep/shell alias pack | Aliases don't transfer into agent contexts (each Bash call is a fresh shell) and agents already prefer dedicated tools; near-zero recurring benefit. |
| RC-015 | R2/VPS sync-script drift | The 5-commit churn (disk fill → per-file verify → expiry gates) converged by 2026-06-30 and is regression-locked; treat as settled. Re-open only on another disk-fill/verification incident (that re-open condition is the watch). |
| RC-016 | Graph dead-ends (4 notes) + over-connected `polymarket_table_dictionary` | Routine hygiene already owned by the existing weekly Janitor pass ([[SKILL_MAP]] § Janitor) — routing it here would duplicate machinery. Flagged to the next Janitor run instead. |
| RC-017 | Sherpa / Indeaverse future skills | Explicitly deferred by design in [[SKILL_MAP]] § Future skills; the reflection engine must not re-propose deferred items absent new evidence of pain. |

### External-repo radar (2026-07-04 pass)

> Radar triage runs inside the reflection pass: per repo — liftable-now vs a live thread, licence, integration cost, verdict per the good-repo criterion ([[2026-06-28_external_repo_audit]]). Awesome-index repos are radar, not adoptees. Full evidence in [[2026-07-04_skills_lifecycle_phase1]].

| seed | verdict | top liftable item | licence | cost |
|---|---|---|---|---|
| anthropics/skills (official) | **Adopt** | `spec/agent-skills-spec.md` + `template/` as our library bundle format; marketplace layout later | Mixed: Apache-2.0 examples; **docx/pdf/pptx/xlsx skills are proprietary source-available — never vendor** | S–M |
| goldmansachs/gs-quant | **Adopt (pattern)** | `gs_quant/skills/__main__.py` — `python -m <pkg>.skills install` CLI for skills shipped inside a pip package | Apache-2.0 | S |
| BuilderIO/skills | **Adopt (next lifts)** | `agent-watchdog` (orchestrator audits implementer), `efficient-frontier` | MIT | S |
| kepano/obsidian-skills | **Adopt (when needed)** | `obsidian-markdown` (syntax authority for brain passes), `defuddle` (web→markdown) | MIT | S |
| bcosm/backtester-mcp | **Borrow-pattern** | `robustness.py` (DSR/PBO/bootstrap-CI) as CPCV cross-checks; DuckDB run-registry; MCP-wrapping-a-backtester | Apache-2.0 | M (verify math first; 2★, unreviewed) |
| "awesome-claude-fable-5" literal repos | **Skip (permanent)** | — SEO/promo lists, not canonical | MIT/none | — |
| hesreallyhim/awesome-claude-code + ComposioHQ/awesome-claude-skills | **Reference (standing radar)** | index entries: Bedrock, Librarian-MCP, AI-Research-Skills, Skill-Seekers, cc-thinking-skills, usage-statusbar | NOASSERTION / none (index-only) | S — re-scan monthly in the reflection pass |

Radar-spawned candidates:

| id | title | pain / cluster | rec | cost | decision | status |
|---|---|---|---|---|---|---|
| RC-018 | Make `library/` bundles agent-skills-spec compliant (anthropics spec + template) | Ad-hoc SKILL.md shapes won't interoperate with marketplaces/other agents | — | S | **accepted — folded into RC-001 this run** | in-build |
| RC-019 | Ship skills *inside* the pip package with a `python -m <pkg>.skills install` CLI (gs-quant pattern) | Bundle + pip module would otherwise be two disconnected artifacts with two install paths | — | S | **accepted — folded into RC-001 this run** | in-build |
| RC-020 | Vendor BuilderIO `agent-watchdog` + `efficient-frontier` | Orchestrator-audits-implementer has no skill support; model-cost routing is ad hoc | LOW | S | **plan** — batch into the RC-013 vetting session (MIT) | planned |
| RC-021 | Vendor kepano `obsidian-markdown` (+ `defuddle`) | Brain passes encode Obsidian syntax rules ad hoc | LOW | S | **plan** — batch into the RC-013 vetting session (MIT; authoritative author) | planned |
| RC-022 | backtester-mcp `robustness.py` cross-check vs our `overfitting_audit.py` + DuckDB run-registry pattern | Our validation stack has no independent implementation to diff against ("non-redundancy by design") | LOW | M | **plan** — pair with the Phase-3 overfitting-harness library extraction | planned |

## Candidates — 2026-07-05 easy-wins pass (on-demand)

> Not a full re-mining — the last discovery pass was the previous day, so no new recurrence evidence could have accumulated. This pass surveyed what's most worth packaging/developing NEXT (candidates + repo + internal skills), built the low-risk wins, and stopped at written proposals for anything bigger or public-facing (per the operator's explicit scoping).

### Built this pass

| id | title | pain / cluster | evidence | rec | cost | decision | status |
|---|---|---|---|---|---|---|---|
| RC-023 | Runnable demo for the changepoint library entry | The only scrub-approved public package had zero runnable example (bundle = SKILL.md only, README 75 lines) — an adopter landing from the site catalog had nothing to execute; a demo is the cheapest credibility upgrade to the publish end | library/changepoint inspection 2026-07-05; catalog entry live on the `/library` page since 07-04 | — (lifecycle deliverable, enters at its stage) | S | **build** — demo for an existing entry | **built** (2026-07-05: `library/changepoint/examples/demo.py` — seeded synthetic calm→crisis→recovery series with two known breaks; scores all 3 detectors vs truth incl. the honest CUSUM-vs-BOCPD trade-off; exercises all 3 integration helpers; asserts live==batch. + `tests/test_demo.py` smoke test pinning the output story; README § Runnable demo; SCRUB.md addendum — no new content class, synthetic data only. **23 tests green** (was 22)) |

### Proposals — bigger / public-facing; STOP at proposal, operator go required

| id | title | proposal (short) | rec | cost | decision | status |
|---|---|---|---|---|---|---|
| RC-024 | Phase-2: extract `calibrate` → `library/calibrate` (`rigorkit-calibrate`) | The natural next package, and the first with a live public consumer: the Calibration Observatory (shipped 2026-07-05, [[strat_news_agent_showcase]]) leans on `calibrate` for its public Brier track record — a public pip package makes that claim reproducible by outsiders. Generalization is already proven: the engine is byte-identical in two projects (`infrastructure/calibration/` · `polymarket/research/lib/calibration/`). Plan = changepoint pattern verbatim: engine (core + markets layer) into `library/calibrate/`, both projects become same-API shims, `calibrate` SKILL.md bundled in-package with the installer CLI, decoupling test, own SCRUB.md. **Exclusion:** the superforecasting ledger state machine stays OUT (vendored MIT fork, upstream `deusyu` — re-point, never re-host); the package is a read-only scorer with a documented ledger-schema adapter | MED (TODO Phase-2 line + the Observatory now demanding it) | M | **build** (operator go 2026-07-05: "build the rest") | **library** (2026-07-05: `library/calibrate/` = `rigorkit-calibrate` v0.1.0, Apache-2.0. 12 tests green in repo venv AND standalone in a fresh venv **without sklearn** (numpy fallbacks proven); epsilon dogfoods it back via same-API shims in both projects — crypto regression 7/7 incl. the ml_metrics byte-for-byte gate, PM 6/6 + expected skip; historical CLI lines unchanged. The epsilon book→ledger-path mapping was deliberately kept OUT of the package (lives in the shims — smaller scrub surface). **SCRUB.md verdict: PENDING HUMAN SIGN-OFF** (checklist executed, no flags; no delegation was given this time). On the `/library` localhost preview with a "review pending" chip; website copy uncommitted, deploy blocked on the scrub. **2026-07-05 (later):** scrub APPROVED by Justin (via Cowork); renamed `lemma-calibrate`; install text rewritten to copy-from-repo + git-install per the no-PyPI decision) |
| RC-025 | Phase-2: extract `data-contract` → library package | Blocker check done this pass: the "Python 3.10 f-string fix on line 856" precondition (skills_library_build_plan § v1 scope, scratch/cowork) is **cleared** — both engines parse under 3.10 grammar (`ast.parse feature_version=(3,10)`, verified 2026-07-05; re-verify with a real 3.10 interpreter at packaging time). Remaining real work: heavier dep surface (`pandera.polars`), and the contracts must be genericized per-instrument (the PM/crypto contract split is the epsilon-specific part; the engine + invariant vocabulary is the public part) | LOW-MED | M | **propose** — sequence AFTER RC-024 (calibrate has a public consumer now; data-contract doesn't) | proposed |
| RC-026 | ~~Obsidian brain starter-kit (public template repo)~~ → **REFRAMED 2026-07-05 (operator): a written SETUP GUIDE / ARTICLE**, not a code extraction | Original framing (public template repo of the brain OS) had the largest scrub surface of any candidate — the law files embed strategy context throughout. Operator reframe: author a clean-from-scratch article — "how we run a multi-agent shared research brain on git + Obsidian" (branch-per-person model, hub/wikilink discipline, hygiene-tooling concepts, agent lanes) — which carries ~zero scrub surface because nothing is extracted | LOW (no observed external pull yet) | M (article, not an L-cost code rewrite) | **shelve on the docket** — schedule deliberately at a future date; do NOT build in a build pass | **shelved-docket** (2026-07-05; the reflection engine must not re-propose — picked up only when deliberately scheduled) |
| RC-027 | `reflection-prompt` + `prd-scaffold` skills (generalized, publishable) | (a) Generalize the reflection-engine SKILL.md into an epsilon-agnostic bundle (mine-your-own-sessions → recurrence × build-cost → decide/log) — the rubric and "nothing-with-reason" discipline are the publishable ideas; (b) codify the PRD → `/goal` co-authoring flow from fable_projects_1_2_plan (scratch/cowork; kickoff prompt → Q&A → emitted goal prompt) as a `prd-scaffold` skill. Both are prompt-ware (no engine). Blocker "no distribution channel" **resolved 2026-07-05** by the no-PyPI / copy-from-repo decision | LOW-MED (PRD flow used twice, 2026-07-04/05, worked both times) | M | **build** (unblocked by the distribution decision; operator go) | **built — scrub pending** (2026-07-05: `library/skills/reflection-prompt/` + `library/skills/prd-scaffold/` — clean-authored SKILL.md + README + fictional worked EXAMPLE.md + LICENSE + SCRUB.md each; no tests by design — prompt-ware has nothing to run, the worked example is the demo. Indexed by the catalog as `library-bundle` entries. **Both SCRUB verdicts: PENDING HUMAN SIGN-OFF**) |

### Radar pins — 2026-07-05 (from the newsagent session's radar; full table in [[newsagent_repo_data_radar_findings]])

> Not a scheduled radar sweep (monthly cadence, last full sweep 07-04). Pinning only the lifecycle-relevant verdicts the newsagent thread produced, so future forecasting work doesn't re-triage:

| seed | verdict | lifecycle relevance | licence |
|---|---|---|---|
| Metaculus/forecasting-tools | **Adopt** (the only Adopt-grade code dep found) | production harness for any future forecasting/ensemble work (N-sample, spend caps) | MIT |
| ForecastBench | **Borrow-pattern** | interim scoring vs prior-day price; freeze-value circularity warning (never anchor the headline number on the market) | MIT code / CC BY-SA data |
| FinceptTerminal | **Borrow-pattern (design only — never read the code)** | news-panel/terminal layout ideas for showcase surfaces | AGPL-3.0 + commercial dual |

## Candidates — 2026-07-06 weekly pass (scheduled)

> First **scheduled weekly** reflection pass (the prior three log entries were on-demand). Window = git/scratch/transcript activity since the last full discovery pass (2026-07-04). Signals gathered via four read-only subagents (efficient-fable): git history, scratch lanes, session transcripts (grep-sampled), and `brain/generated/` reports + [[SKILL_MAP]]. **Short window (~2 days), immediately after the intensive 07-04/07-05 discovery+build week** — so the expected and actual outcome is mostly re-observation of already-logged items plus one self-improvement build. Every lead was verified on the live system before scoring.

### Signals observed and where they landed (the "nothing new" evidence)

- **Clean git history** (18 commits 07-04→07-06): news-agent v3.1/v3.2 iteration, `calibrate` library extraction, MM Task-5 bridge build in progress (untracked: `mm_engine_bridge.py`, `mm_bridge_cli.py`, `order_safety.py`, `mm_eval/`). No reverts, no merge conflicts, one already-shipped nbstripout tooling fix (RC-004). `fv_params.json` / `config.py` churn is intentional calibration, not friction; the MM bridge is forward research/execution work owned by the MM thread, not a tooling pain. → **nothing new**.
- **Scratch lanes**: only `scratch/cowork/fable_projects_1_2_plan.md`; its "ready to run" v3.1/v3.2 news-agent items already shipped per git history. `scratch/codex/` effectively empty. → **nothing new**.
- **Hygiene** clean (0 duplicate basenames, 0 broken links; refreshed 07-06 11:01, graph = 254/258 nodes in one component). Routine Janitor items only (`polymarket/research/newsagent/AUTOMATION.md` orphan + missing frontmatter; one orphan handoff `2026-07-05_newsagent_v32_dashboard_lean.md`) → owned by the weekly Janitor pass per **RC-016**, not a reflection candidate. Daily brief no longer flags the scrapped launchd `brain-commit-push` job (**RC-005** concern resolved). → **flag to Janitor, no RC**.

### Built this pass

| id | title | pain / cluster | evidence | rec | cost | decision | status |
|---|---|---|---|---|---|---|---|
| RC-029 | Harden reflection-engine transcript gathering | This skill's transcript-scan step (§ Procedure step 3) had two accuracy bugs that recur on *every* weekly pass: (a) it greps the **current session's own transcript**, which contains the pass's own pattern list → inflated friction counts; (b) a combined multi-file `grep` over huge single-line JSONL transcripts intermittently returns **false-zero counts** in this environment | surfaced by the 07-06 transcript subagent, which correctly excluded the self-file (`656e41f7`) and switched to a per-file grep loop after hitting the false-zero | MED (bites an automated system — every future weekly pass) | S | **fix** (edit SKILL.md step 3) | **built** (2026-07-06: `.agents/skills/reflection-engine/SKILL.md` step 3 now mandates excluding the current-session transcript + a per-file grep loop. Prompt-ware procedure — nothing to unit-test; the edit is the deliverable) |

### Re-observed — already logged, recorded so they don't re-surface as "new" (never-re-propose rule)

| id | signal this pass | verified on live system | disposition |
|---|---|---|---|
| RC-008 | `ModuleNotFoundError` from running a local-package script without `PYTHONPATH=.` recurred in **3 distinct transcripts** (missing `scripts`, `newsagent`, `overfitting_audit`) | yes — `scripts/` and `newsagent/` ARE packages (`__init__.py` present); scripts do `from scripts import …`; `overfitting_audit.py` lives in `infrastructure/validation/` | **stays PLANNED.** Fresh HIGH-recurrence evidence *raises priority* but the remedy (editable installs / a run-environment conventions change touching [[CODEX]] + both pyprojects) is still M/L with coordination risk; an autonomous scheduled pass must not unilaterally rewrite the run-environment law. Escalate to the operator when the conventions change is deliberately scheduled. |
| RC-010 | `command not found: timeout` recurred in **2 distinct transcripts** | yes — neither `timeout` nor `gtimeout` is on PATH (macOS zsh ships no GNU `timeout`) | **stays CLOSED-NOTHING.** The re-open filter requires a *scheduled job* (not an agent) to be bitten; these are interactive agent sessions. Correct remedy = agent behavior (use the Bash tool's `timeout` parameter the harness provides), not a `timeout` binary — a shim would *encourage* the wrong pattern. No change. |

### New — watch

| id | title | pain / cluster | evidence | rec | cost | decision | status |
|---|---|---|---|---|---|---|---|
| RC-028 | git-lock + nbstripout contention on *interactive* commits | Interactive `git commit` can hang / hit `.git/index.lock` while the nbstripout filter runs on many dirty notebooks (the `git-lock-nbstripout-contention` memory note) | recurred in 2 transcripts (last 48h); verified: nbstripout filter is now pinned to an absolute interpreter (RC-004 holds), and **0** dirty notebooks in the tree right now (no active pileup) | MED | S | **watch** | watch — **already mitigated on both paths**: the automated path (`tools/brain_commit_push.sh`) neutralizes the filter (`-c filter.nbstripout.clean=cat …`) and self-heals a stale `index.lock`; the interactive path has the documented memory workaround. **Re-open to build a shared `git-safe-commit` helper only if it recurs in ≥3 distinct interactive sessions AND the memory workaround proves insufficient** (RC-015-style discipline: don't build tooling for a mitigated/converged issue). |

## Pass log

| date | mode | signals used | outcome |
|---|---|---|---|
| 2026-07-04 | on-demand (lifecycle kickoff) | git log 06-01→07-04 · scratch/codex + scratch/cowork · ~50 session transcripts (06-29→07-04, best-effort) · hygiene/graph reports 07-03 · SKILL_MAP future-skills · 6-seed external radar | 22 candidates logged: 3 lifecycle deliverables, 3 build-now, 1 blocked-human, 7 planned, 6 closed-nothing, 2 radar-adopts folded into RC-001 |
| 2026-07-05 | on-demand (easy-wins / what-next survey) | full backlog + library/ + internal skills state · newsagent session evidence ([[2026-07-05_newsagent_showcase_v0_stop]] + radar findings) · live verification: changepoint suite re-run (22→23 green), catalog regen (timestamp-only drift — reverted, no action), 3.10-grammar parse of both data-contract engines | 2 built (RC-023 demo; RC-009 checklist half), 4 proposals logged for operator go (RC-024→027), 3 radar pins. **No full re-mining** — last discovery pass was the previous day; deliberately skipped so the weekly Monday pass stays the recurrence-evidence cadence |
| 2026-07-05 (2nd, operator go: "build the rest") | build pass, not discovery | RC-024 proposal + operator instruction to build to a localhost-viewable state, skipping anything genuinely unnecessary | RC-024 **built → library** (rigorkit-calibrate, scrub pending); catalog regenerated (2 library entries) + website preview live on localhost with a review-pending chip. Deliberately NOT built, with reasons recorded in the handoff: RC-025 (no public consumer, heavier deps, contract genericization is real API design — after calibrate settles), RC-026 (L-cost rewrite + biggest scrub surface — needs its own scoping session), RC-027 (prompt-ware with no distribution channel until the PyPI/repo-split decision), website commit/deploy (that IS the scrub gate — human-only) |
| 2026-07-05 (3rd, operator: distribution + rebrand + hardening) | decision-application pass, not discovery | Operator decisions: NO PyPI (repo-is-the-distribution, BuilderIO/kepano/gs-quant model); rebrand rigorkit → **lemma**; RC-027 unblocked; RC-026 reframed to an article + shelved | Rename executed repo-wide (git mv namespaces, pyprojects, shims, both venvs reinstalled; all suites green: changepoint 23, calibrate 12, crypto 7/7 + 20 shim, PM 6+skip); all install text → copy-from-repo + optional git-install; lemma index README; **RC-027 built** (2 standalone bundles, scrubs PENDING); catalog tool extended for `library-bundle` entries (47 total; public feed = 2 approved packages + 2 pending bundles); RC-026 → shelved-docket. Website copy + repo split remain human-only |
| 2026-07-06 | **weekly (scheduled)** | git log 07-04→07-06 (all branches) · scratch/codex+cowork · 16 recent session transcripts (grep-sampled, self-file excluded) · brain/generated hygiene+graph+brief (refreshed 07-06 11:01) · [[SKILL_MAP]] future+runtime skills | **1 built** (RC-029 reflection-engine transcript hardening — S, improves the automated weekly pass); **2 re-observed** already-logged (RC-008 stays planned w/ strengthened HIGH-recurrence evidence; RC-010 stays closed — re-open filter requires a scheduled-job hit, not agents); **1 new watch** (RC-028 git-lock interactive contention — already mitigated on both commit paths); routine hygiene flagged to the Janitor (RC-016). **No new build-now beyond the self-improvement** — short 2-day window right after the 07-04/05 discovery+build week. **Radar SKIPPED**: the July external-repo radar already ran 07-04, so 07-06 is not the first pass of the month. |
