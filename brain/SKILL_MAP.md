---
title: Skill Map
created: 2026-06-07
status: active
owner: justin
project: infra
para: area
hubs:
  - VAULT_MAP
  - CODEX
  - COWORK
tags:
  - obsidian
  - brain
  - workflows
  - skills
---

# Skill Map

> Repeatable agent/human workflows so brain hygiene becomes an operating system, not a memory game.
> Each skill is small and triggerable. Prefer running a skill over inventing an ad-hoc process.

Hub links: [[VAULT_MAP]] | [[OPERATING_RHYTHMS]] | [[CODEX]] | [[COWORK]] | [[TODO]]

## How to read this

A "skill" here is a named, repeatable pass — not a heavyweight system. Each row says what triggers it, what it reads, what it produces, and who runs it. Most are a single command or a short focused edit. Start manual; automate via [[OPERATING_RHYTHMS]] once a skill proves out.

There are two different things called "skills":

1. **Brain workflow skills** — the operating passes in this file: Janitor, Rock Tumbler, Chronicler, Cartographer, Librarian, Daily Brief. These are repo-specific habits for keeping the Markdown brain useful.
2. **Agent runtime skills** — Codex/Claude capabilities such as `query`, `read-memories`, `systematic-debugging`, `github`, `browser`, `openai-docs`, `spreadsheets`, `documents`, `presentations`, `imagegen`, `cost-mode`, `efficient-fable`, and `stay-within-limits`. These help an agent do a task, but they do not run unless a prompt or scheduled task calls them — except the two runtime efficiency skills (`efficient-fable`, `stay-within-limits`), which are description-triggered automatically in Claude Code; see § Runtime efficiency skills.

Rule of thumb: brain workflow skills define **what pass should happen**; runtime skills define **which tool the agent should use while doing it**.

## Automation status

| Layer | Status | What runs automatically | What still needs a prompt |
|---|---|---|---|
| Hygiene scan | Automated by scheduled task `brain-hygiene-weekly` | Runs `tools/brain_hygiene.py` and reports counts | The actual Janitor fix pass |
| EOD brief | Automated by scheduled task `brain-eod-brief` | Runs hygiene + graph audit and writes `brain/generated/daily_brief.md` | Any canonical note/TODO changes suggested by the brief |
| Markdown sync | Automated by scheduled task `brain-commit-push` | Commit/push of scoped Markdown paths to the operator's personal branch (never main) | Conflict resolution per [[MERGE_PROTOCOL]], if the task aborts |
| MM capture health | Codex heartbeat `mm-stage-1-capture-health` | Read-only live capture status checks | Any restart, repair, or deployability decision |
| Brain cleanup / synthesis | Manual | Nothing edits canonical notes unattended | Janitor, Rock Tumbler, Chronicler, Cartographer, Librarian |

The scanners are allowed to run unattended because they only write generated reports. Anything that edits durable notes should be a named pass with a human-readable prompt and a reviewable diff.

## Core skills

| Skill | Trigger | Reads | Produces | Owner |
|---|---|---|---|---|
| **Daily Brief** | Start of a working day | changed notes, [[TODO]], recent git log | short "what changed / what matters / what's next" note | either agent |
| **Agent Scratch Log** | Any fast WIP that isn't canonical yet | the task at hand | `scratch/<agent>/YYYY-MM-DD.md` (local-only) | active agent only |
| **Chronicler** | End of a working session, or any decision/closure | the day's scratch + diffs | dated handoff in `brain/handoffs/`, or a decision line in the right hub | assigned chronicler |
| **Rock Tumbler** | A research branch closes or matures | messy scratch + results | one clean canonical `*_findings.md` with links + decision | branch owner |
| **Janitor** | Weekly, or after a burst of new notes | hygiene report | small link/metadata/dedupe fixes (by Codex) | Codex |
| **Cartographer** | New branch, renamed folder, or stale map | folder tree + hubs | refreshed [[VAULT_MAP]] / `generated/GENERATED_INDEX.md` | either agent |
| **Librarian** | New reused shorthand or table column | the term in context | definition in [[glossary]] / [[polymarket_table_dictionary]] | either agent |

## How to call each skill

Use these as lightweight prompts to Cowork or Codex. Keep the pass name in the prompt so everyone knows which mode the agent is in.

| Skill | Ask this | Good moment | Guardrails |
|---|---|---|---|
| **Daily Brief** | "Run a Daily Brief from the generated reports, TODO, and recent commits. Keep it to one screen." | Start of day, or after opening the repo cold | Read-only unless explicitly asked to edit |
| **Agent Scratch Log** | "Use your own scratch lane for WIP; promote only durable results." | Any exploratory agent session | Scratch lives in `scratch/<agent>/` (top-level, local-only, ignored) |
| **Chronicler** | "Run a Chronicler pass for today's decisions and handoff. Update the relevant hub/TODO only if needed." | End of session, after a major decision, before handing to a collaborator | Record why, not every intermediate thought |
| **Rock Tumbler** | "Rock Tumbler this branch into one canonical findings note with summary, evidence, decision, and next gate." | Branch closes, result matures, or messy scratch should become durable | Do not invent conclusions; preserve numbers and links |
| **Janitor** | "Run a Janitor fix pass from `brain/generated/hygiene_report.md`; fix broken links, duplicates, metadata, and summaries in reviewable batches." | Weekly, after a burst of note creation, before sharing with coworker | Scanner finds issues; agent fixes deliberately |
| **Cartographer** | "Run a Cartographer pass: check `VAULT_MAP` active branches / where-to-write against the repo and refresh generated indexes." | New branch, renamed folder, changed ownership, stale map | Fix inaccuracies; avoid broad restructuring |
| **Librarian** | "Run a Librarian pass for new shorthand/columns; define terms in glossary or table dictionary and link them." | New reused acronym, CSV column, bucket label, or strategy nickname | Define once; do not over-document one-off terms |

## Implementation model

The safest workflow is report-first, edit-second:

1. **Generate / gather context.** Run `tools/brain_hygiene.py`, `tools/brain_graph_audit.py`, read [[TODO]], relevant hubs, recent commits, and scratch notes.
2. **Choose one named pass.** Do not mix Janitor, Rock Tumbler, and Cartographer in one giant edit unless the task is tiny.
3. **Work on your personal branch.** All durable-Markdown edits happen on the operator's personal branch ([[MERGE_PROTOCOL]]); concurrent edits surface as merge conflicts, not live collisions.
4. **Edit only the intended surface.** Janitor edits metadata/links/summaries; Rock Tumbler writes canonical findings; Cartographer updates maps; Chronicler records decisions.
5. **Verify.** Re-run the scanner or inspect the target note/map.
6. **Hand off.** Add a short note to `brain/handoffs/` or [[TODO]] when the pass changes state, ownership, or next gates.

### Who should do what

| Pass | Best owner | Why |
|---|---|---|
| Daily Brief | Cowork scheduled task, or either agent manually | Mostly reading and summarizing |
| Janitor | Codex | Good at systematic file edits, link repair, scanner verification |
| Rock Tumbler | Cowork drafts, Codex can polish/verify | Needs strategic judgement plus careful Markdown hygiene |
| Chronicler | Whoever made the decision | Captures intent while it is fresh |
| Cartographer | Either agent | Needs repo inspection and map discipline |
| Librarian | Either agent | Small, local definitions; easiest when noticed immediately |

### Suggested Cowork operating rule

Cowork should not constantly edit canonical hubs while thinking. It should draft in its lane, then trigger a named pass:

- If a session produced decisions: **Chronicler**.
- If a research branch produced durable evidence: **Rock Tumbler**.
- If notes/folders moved or a new branch exists: **Cartographer**.
- If terms/columns became reusable: **Librarian**.
- If reports show link/metadata rot: hand to Codex as **Janitor**.

## Event triggers

These triggers turn the skills into lightweight automation workflows without letting unattended agents rewrite canonical notes. Scheduled jobs may detect events and draft a prompt; durable edits should still run as a named pass with a visible diff.

| Event | Threshold | Triggered skill | Owner | Automation shape |
|---|---:|---|---|---|
| Hygiene report worsens | any broken link, duplicate basename, orphan, or missing hub backlink | `brain-janitor` | Codex | Weekly scan reports issue and suggests Janitor prompt |
| Metadata backlog grows | >10 missing frontmatter or >5 missing findings summaries | `brain-janitor` | Codex | Weekly scan suggests batch cleanup |
| Branch produces durable result | any branch has a conclusion, falsification, promotion, or mature experiment | `brain-rock-tumbler` | Cowork drafts; Codex verifies | Human/Cowork calls pass before branch is considered done |
| Session produces decisions | any material decision, closure, blocker, or ownership change | `brain-chronicler` | Decision owner | EOD brief flags it; owner records handoff/status |
| Repo/map shape changes | new note folder, renamed folder, new hub, closed branch, or map mismatch | `brain-cartographer` | Either agent | EOD/weekly scan flags map refresh |
| Reused term emerges | same shorthand/column/bucket appears in 3+ notes/scripts or enters a data manifest | `brain-librarian` | Either agent | Agent adds glossary/table-dictionary entry |
| Same question repeats | agent asks "where is X?" twice or scans broad folders repeatedly | `brain-cartographer` or `brain-librarian` | Either agent | Add map route or definition so the question dies once |

## Prompt pack

Use these prompts when you want Codex/Cowork to run a skill. The first line is intentionally explicit so the agent loads the right workflow.

### Codex Janitor prompt

```markdown
Use the `brain-janitor` skill.

Before editing, read:
1. `brain/SKILL_MAP.md`
2. `brain/OPERATING_RHYTHMS.md`
3. `brain/VAULT_MAP.md`
4. `brain/CODEX.md`

Run `python3 tools/brain_hygiene.py`, then read `brain/generated/hygiene_report.md`,
`brain/generated/stale_notes.md`, and `brain/GENERATED_INDEX.md`.

Fix only hygiene issues in reviewable batches:
- broken wikilinks
- duplicate basenames
- missing hub backlinks
- missing frontmatter
- findings notes missing `## Summary`
- stale TODOs if the correct action is obvious

Do not alter research conclusions, numbers, CIs, or strategy decisions.
Do not edit `brain/generated/`.
Re-run the scanner and report before/after counts plus anything flagged but not fixed.
```

### Cowork Rock Tumbler prompt

```markdown
Use the `brain-rock-tumbler` skill.

Read `brain/SKILL_MAP.md`, `brain/VAULT_MAP.md`, `brain/COWORK.md`,
`brain/CODEX.md`, `brain/TODO.md`, and the relevant strategy hub.

Turn this branch/session into one canonical findings note:
- plain-English Summary
- research question
- data/experiment covered
- evidence links
- decision: continue / close / promote / watch
- next falsifiable gate

Pull claims only from existing artifacts. Preserve exact numbers and caveats.
If the branch is not mature enough for a canonical note, write a short handoff explaining what is missing.
```

### Chronicler prompt

```markdown
Use the `brain-chronicler` skill.

Read `brain/TODO.md`, `brain/VAULT_MAP.md`, the relevant hub, recent commits, and today's changed notes.

Record only durable session state:
- decisions made
- closures or reopened branches
- blockers
- ownership changes
- next gates
- links to evidence

Write a dated handoff in `brain/handoffs/` or a tight status update in the relevant hub/TODO.
Do not create a transcript. Keep it useful for the next agent.
```

### Cartographer prompt

```markdown
Use the `brain-cartographer` skill.

Read `brain/VAULT_MAP.md`, `brain/SKILL_MAP.md`, `brain/TODO.md`, and relevant hubs.
Inspect the actual repo state with targeted `find`, `rg`, and `git status`.
Run `python3 tools/brain_hygiene.py` and `python3 tools/brain_graph_audit.py`.

Check whether `VAULT_MAP` is accurate for:
- active research branches
- where-to-write paths
- core hubs
- generated reports
- agent lanes

Fix inaccuracies only. Do not restructure folders or invent hub notes.
Flag missing hubs or ambiguous ownership.
```

### Librarian prompt

```markdown
Use the `brain-librarian` skill.

For each reusable term, acronym, CSV/parquet column, bucket label, or strategy nickname:
1. Search with `rg` to see real usage.
2. Decide whether it belongs in `brain/glossary.md`, a table dictionary, or a strategy hub.
3. Add a compact definition and aliases if useful.
4. Link one or two canonical context notes.

Do not guess column meaning from the name alone.
Do not define one-off terms.
```

### Data-contract gate prompt

```markdown
Run the `data-contract` skill on <dataset> before proceeding.

Before any backtest / walk-forward / CPCV / replay / analysis run, validate the
primary dataset's contract and FAIL CLOSED on violation:

- Polymarket (from polymarket/research/):
  PYTHONPATH=. uv run python -m data_infra.schemas.cli validate <pm_trades|pm_closed_positions|pm_traders|pm_l2_*>
- crypto (from repo root):
  PYTHONPATH=. .venv/bin/python -m infrastructure.data.schemas.cli validate <crypto_ohlcv_daily|crypto_ohlcv_hourly>

In code, gate the entry script's bootstrap with `guard_dataset("<dataset>")`
(crypto: `from infrastructure.data.schemas import guard_dataset`; PM:
`from data_infra.schemas import guard_dataset`). It aborts on any schema /
append-only / lowercase-0x / finite / lookahead violation. Drift (PSI+KS) and
missing-bar gaps are warnings, never blockers. Emergency bypass only:
`EPSILON_DATA_CONTRACT=warn|off`. Never bypass for a real run. The plain-English
failure report lands under `…/schemas/data_monitoring/reports/`.
```

### Changepoint-audit prompt

```markdown
Use the `changepoint-audit` skill on <series> (lookahead-free).

Run a causal structural-break detector and use the breaks for gating / embargo:

  PYTHONPATH=. .venv/bin/python -m infrastructure.changepoint.cli detect \
      <parquet> --column Close --returns --standardize --detector bocpd \
      --out infrastructure/changepoint/changepoints/<name>.parquet
  PYTHONPATH=. .venv/bin/python -m infrastructure.changepoint.cli benchmark   # lag/FPR
  PYTHONPATH=. .venv/bin/python -m infrastructure.changepoint.cli kappa-demo  # vs regime transitions

In code: CUSUM/Page-Hinkley = O(1)/bar mean-shift first line; BOCPD = run-length
posterior with a real change_prob (catches mean AND variance shifts). Detector
state at t uses only data ≤ t (no-lookahead, tested). Integration helpers:
`changepoint_features` (causal feature for regime-classifier Stage 2),
`fresh_break_gate` (block trend entries for a cooldown after a break),
`embargo_indices_from_breaks` (purge windows for cpcv_engine). ruptures is
OFFLINE-only (labelling/validation), never wired live. This COMPLEMENTS the
batch HMM→XGBoost regime-classifier — it does not replace it.
```

### Calibrate prompt

```markdown
Use the `calibrate` skill to score how good these probabilities are.

Score calibration on the forked superforecasting ledger (read-only) — pick the
book (the two books share no state):

  crypto (from repo root):
    PYTHONPATH=. uv run --no-project python -m infrastructure.calibration.cli --book crypto score
    PYTHONPATH=. uv run --no-project python -m infrastructure.calibration.cli --book crypto report --out docs/assets/<name>.png
  polymarket (from polymarket/research/):
    PYTHONPATH=. uv run python -m lib.calibration.cli --book polymarket score

Reports Brier + Murphy decomposition (reliability − resolution + uncertainty),
log-loss, ECE/MCE, a reliability diagram with Wilson bands, Spiegelhalter's Z,
and calibration-in-the-large. In code, import the engine
(`from infrastructure.calibration import core` / `from lib.calibration import core`)
for `brier_score`, `murphy_decomposition`, `reliability_table`,
`isotonic_recalibrate` / `platt_recalibrate` (reuse the walk_forward pattern),
and the markets layer (`implied_prob_*`, `devig`, `market_edge`, `realized_edge`).
Read the Wilson bands, not just the dots — an off-diagonal point whose band
straddles the diagonal is not yet miscalibration. `calibration_table` reproduces
`ml_metrics.calibration_table` byte-for-byte. Read-only: it never writes the ledger.
```

## Skill detail

### Daily Brief
Summarize what changed since yesterday, what matters, and the next falsifiable action. Keep it to a screen. Pull recently-changed notes and stale TODOs from `brain/generated/` (see Janitor) rather than re-reading everything.

### Agent Scratch Log
Each agent writes fast, messy work-in-progress to its own scratch (`scratch/codex/` or `scratch/cowork/`, top-level and local-only). This is the collision-avoidance rule: **agents do not draft inside canonical hubs.** Scratch is git-ignored — it exists only on each person's machine and never enters any branch or merge — so per-person WIP can't collide; promote anything durable via Chronicler / Rock Tumbler.

### Chronicler
Capture *why* things changed: decisions, closures, handoffs. Output is a dated `brain/handoffs/<date>_<topic>.md` or a tight status line appended to the relevant hub and [[TODO]]. One chronicler pass per session beats ten scattered edits.

### Rock Tumbler
Turn a messy branch into one canonical findings note that a cold reader can understand: plain-English headline, summary, design, evidence links, decision, next gate. Follows the markdown quality standard in [[CODEX]]. This is how scratch becomes durable knowledge.

### Janitor
Run `tools/brain_hygiene.py` to surface duplicate basenames, broken wikilinks, orphan notes, notes missing hub backlinks, notes missing frontmatter/summary, and stale TODOs. The script **finds** issues; Codex **fixes** them in a focused pass. Never auto-rewrite notes blindly.

### Cartographer
Keep the maps honest. When a branch is added/closed or a folder moves, update [[VAULT_MAP]] § Active research branches and regenerate `brain/generated/GENERATED_INDEX.md` (via `tools/brain_hygiene.py`). The map is the cheapest thing to keep current and the most expensive thing to let rot.

### Librarian
When a shorthand (CSV column, bucket label, code name) gets reused across notes, define it once in [[glossary]] or [[polymarket_table_dictionary]] and link to it. Prevents the "what did `far_absz_ge1` mean again" tax.

## Agent runtime skills

These are not brain workflows by themselves. They are tools an agent can call while performing a workflow skill or implementation task.

| Runtime skill | Useful for | Call it when |
|---|---|---|
| `systematic-debugging` | Debugging failures without guessing | Tests fail, scripts behave unexpectedly, data counts look wrong |
| `query` | DuckDB / SQL over data files | You need counts, joins, distributions, or quick data inspection |
| `read-memories` | Searching past Claude/Codex sessions | You ask "do you remember" or need a prior decision not in the brain |
| `github` / `yeet` / PR skills | PRs, reviews, CI, publishing branches | Work needs GitHub state, review comments, or push/PR flow |
| `browser` | Inspecting local apps or GitHub-rendered pages | Need to verify README/rendered UI/browser behavior |
| `openai-docs` | Current OpenAI API/product guidance | Building with OpenAI APIs or checking latest model/API details |
| `spreadsheets` | CSV/XLSX analysis or workbook creation | Research output needs spreadsheet formatting, formulas, charts |
| `documents` | Word/docx-style artifacts | A written report needs document formatting/export |
| `presentations` | Slide decks | A research or investor update needs PPTX/slides |
| `imagegen` | Bitmap visuals | Need an illustrative/generated image, not a chart/code-native asset |
| `cost-mode` | Concise Claude Code behavior | Budget/token discipline matters; user says cost, budget, tokens, or `/cost-mode` |
| `superforecasting` | Logging a forecast, settling/scoring it (Brier), and reviewing calibration | A decision or claim needs a resolvable, dated, Brier-scored probability — "will X happen", "should I", "how likely", or a calibration review. Set `SF_BOOK=polymarket` or `crypto` first (one ledger per book) |
| `skill-creator` / `skill-installer` | Extending agent capability | You want a new reusable skill or to install one from a repo |
| `changepoint-audit` | Causal, lookahead-free structural-break detection (CUSUM / Page-Hinkley / BOCPD) on a timestamped series | You need a **live** regime-shift signal (the regime-classifier is batch-only), a causal `change_prob` feature for regime Stage 2, a fresh-break gate on trend entries, or break timestamps to embargo in `cpcv_engine`. Crypto instance in `infrastructure/changepoint/`; first-party, prompt-invoked; complements (≠ replaces) `topics/regime-classifier/` |
| `calibrate` | Scoring how good probabilistic forecasts/prices are: Brier + Murphy decomposition, log-loss, ECE/MCE, reliability diagram (Wilson bands), Spiegelhalter's Z, isotonic/Platt recalibration, model-vs-market edge | You ask how well-calibrated a model/forecaster is, want a Brier/reliability read or an over/under-confidence check, to recalibrate probabilities, or to compare model-p vs market-implied-p. Read-only consumer of the forked superforecasting ledger; per-project byte-identical engine (`infrastructure/calibration/` crypto · `polymarket/research/lib/calibration/` PM); set `--book` / `SF_BOOK` |

Runtime skills should be mentioned in prompts only when they matter. Example: "Run a Janitor pass; use `systematic-debugging` if the hygiene scanner output looks inconsistent" is better than asking every skill to load every time.

`superforecasting` is **vendored** (MIT, [`deusyu/superforecasting-skill`](https://github.com/deusyu/superforecasting-skill)) into `.agents/skills/superforecasting/` with symlinks in `.claude/skills/` + `~/.codex/skills/` and a `skills-lock.json` entry (upstream commit `8913b08`). Local fork: the forecast ledger is **repo-controlled and one-per-book** — `SF_BOOK=polymarket` → `polymarket/research/data/superforecast/`, `SF_BOOK=crypto` → `live_trading/data/superforecast/` (git-ignored append-only runtime data, like `trades.json`; the two books share no state). **Guardrail — anti-post-hoc:** the `sf.py` state machine rejects any edit to a forecast once it is settled/scored, so a probability can never be rewritten with hindsight (the engine's `TRANSITIONS` table is the enforcement point). Provenance + the two ledger paths: [[2026-06-29_superforecasting_skill_vendored]].

`calibrate` is **first-party** (built in-repo, NOT vendored) and sits on top of that ledger as a **read-only consumer** — it scores forecasts but never writes the ledger or re-implements its state machine. Packaging mirrors `data-contract`: `.agents/skills/calibrate/` with symlinks in `.claude/skills/` + `~/.codex/skills/` and the source commit at the top of its SKILL.md; the engine is `core.py`, duplicated **byte-identical** into each project (`infrastructure/calibration/` crypto · `polymarket/research/lib/calibration/` PM), never cross-imported. It reuses the crypto stack's conventions rather than duplicating them: `infrastructure/backtester/ml_metrics.py:calibration_table` (reproduced byte-for-byte, gate-checked for no regression) and `infrastructure/ml/walk_forward.py`'s `IsotonicRegression(out_of_bounds='clip')` recalibration pattern (pure-numpy fallback where sklearn is absent). Design + gate results + embedded reliability diagram: [[calibration_scoring_layer_findings]].

### Runtime efficiency skills

Unlike the prompt-invoked brain passes above, these are **description-triggered automatically in Claude Code**: the skill's frontmatter description matches the task and Claude Code loads it without anyone asking. Vendored (adapted) from [BuilderIO/skills](https://github.com/BuilderIO/skills) into `.agents/skills/` with symlinks in `.claude/skills/`; source commit recorded at the top of each SKILL.md. The orchestration variant is not a vendored skill — it is law in [[COWORK]] § Delegation discipline. `data-contract` is also **first-party** (built in-repo, not vendored), but follows the same packaging: `.agents/skills/data-contract/` with symlinks in `.claude/skills/` and `~/.codex/skills/`, source commit recorded at the top of its SKILL.md. Its code lives under each project's `…/schemas/` package (engine + contracts), not in the skill dir.

| Skill | Auto-trigger condition | What it does | Guardrails |
|---|---|---|---|
| `efficient-fable` (Codex / Claude Code: full pattern, auto-triggered) | Token-heavy implementation work: CPCV/walk-forward sweeps across assets, DuckDB scans over polymarket fills/positions, multi-notebook runs, log/capture-output reduction, broad repo or vault scans, repetitive bounded edits | Main agent orchestrates; cheap subagents do bounded heavy lifting; judgment, integration, and final review stay in the main agent | Every handoff packet restates the [[CODEX]] invariants (uv never bare pip, `PYTHONPATH=. uv run` from `polymarket/research/`, lookahead-free metrics, append-only parquet, writes per [[VAULT_MAP]] § Where to write things, commits only to the operator branch); "find prior work" subtasks use gbrain MCP tools, not hub reads; subagent reports are leads, not facts — verify before relying |
| `efficient-fable-orchestration` (Cowork: read-only fan-out, law-driven) | Token-heavy READING in Cowork: vault scans, repo audits, multi-source gathering | Cowork spawns parallel read-only subagents and keeps judgment/synthesis local | Subagents never edit files or run analyses; anything implementation-shaped still becomes a pre-registered Codex prompt — see [[COWORK]] § Delegation discipline |
| `stay-within-limits` | Before/within: CPCV parameter sweeps, per-asset notebook waves, bulk Polymarket reprocessing, any run expected > 30 min or > 2 parallel subagents | Checks 5-hour/weekly usage between waves (`npx -y ccusage@latest blocks --active --json`), pauses new work at 95% of either limit, resumes via self-contained wake prompts; wave throttle 3 | Never interrupt in-flight subagents to save budget; a budget-pause handoff mid-branch follows the Chronicler convention (what ran, what's pending, where results landed) |
| `data-contract` (Codex / Claude Code: first-party, auto-triggered) | About to run a backtest / walk-forward / CPCV / replay / analysis over a known dataset (PM fills/L2 or crypto OHLCV parquet), OR the data looks wrong (NaN/Inf, duplicate/missing bars, non-lowercase 0x addresses, a mutated shard, future-dated rows, a drifted distribution) | Validates schema + invariants (finite, lowercase-0x, monotone-where-real, cadence, lookahead-free, append-only) via `pandera.polars` and computes PSI+KS drift vs a stored reference; a fail-closed `guard_dataset()` aborts the run on violation; writes a plain-English markdown failure report; drift + missing-bar gaps are warnings | Per-project, NEVER cross-import (`infrastructure/data/schemas/` crypto · `polymarket/research/data_infra/schemas/` PM); only imposes invariants the instrument actually has (PM fills carry no monotone clause; the OHLCV refetch cache carries no append-only clause); drift never blocks; coverage caps on huge files are logged; emergency bypass `EPSILON_DATA_CONTRACT=warn\|off` (never for a real run) |

## Future skills (deferred)

From [[OBSIDIAN_INFRA_ROADMAP]] — build only when the basics earn their keep:

- **Sherpa** — route a human/agent to the right context pack for a task.
- **Graph audit** — implemented as `tools/brain_graph_audit.py`; future work is turning recurring graph findings into an automatic Cartographer/Janitor prompt.
- **Indeaverse** — idea-graph + branch registry navigable by concept, not folder (Phase 5).
