---
name: reflection-engine
description: >
  Discovery front of the skills lifecycle: mine recent work for recurring pain
  and turn it into build decisions. Use on the weekly scheduled pass, or
  on-demand when asked to "run a reflection pass", "what should we build/automate
  next", "review recent sessions for friction", or after a burst of repetitive
  manual work. Gathers signals via read-only subagents (git history, scratch
  lanes, session transcripts, hygiene/graph reports), clusters pain, scores
  recurrence × build-cost, logs every decision to brain/reflection/candidates.md
  (including "nothing" with a reason), executes accepted candidates, and runs
  the external-repo radar.
---

<!--
First-party skill (NOT vendored). Built 2026-07-04 as part of the skills
lifecycle (brain/reflection/candidates.md RC-002). Law: brain/CODEX.md.
The backlog it maintains is brain/reflection/candidates.md (tracked,
authoritative). SKILL_MAP + library/ stay canonical for built/published skills.
-->

# Reflection Engine

One pipeline: **discover → decide (recurrence × build-cost) → build → validate →
package → publish**. This skill is the *discover→decide→build* front. It exists
so recurring pain gets found systematically instead of ad hoc, and so rejected
ideas stay rejected (logged with reasons).

## Cadence

- **Weekly pass** — scheduled (`reflection-weekly`, Mondays after the hygiene
  scan so `brain/generated/` reports are fresh).
- **On-demand** — any time the operator asks, or after a session with obvious
  repeated toil.
- **Radar** — the external-repo sweep runs INSIDE this pass, monthly (first
  pass of the month), or when new seed repos are named.

## Procedure

### 1. Gather signals (read-only subagents — efficient-fable pattern)

Spawn parallel read-only subagents; keep synthesis central. Never let a
subagent edit files. Standard four:

1. **git history** — commits since the last pass (see the Pass log in
   candidates.md): churn hotspots, fix/again/rework messages, repeated chores
   (normalization, hygiene, vendoring), merge pain, test churn. Evidence =
   hashes + dates + counts.
2. **scratch lanes** — `scratch/codex/` + `scratch/cowork/`: explicit
   complaints, planned-but-unbuilt items, deferred chores.
3. **session transcripts (best-effort)** — `~/.claude/projects/<this repo>/`:
   grep-sample friction markers (errors, retries, repeated commands,
   corrections). Sampling only; never full reads; state confidence honestly.
   Two gotchas (learned the 2026-07-06 pass — RC-029; gotcha #1 broadened the
   2026-07-17 pass — RC-030; gotcha #1's *detection method* corrected the
   2026-07-21 pass — RC-033), both mandatory for an accurate scan:
   - **Exclude EVERY reflection-pass transcript, not just the current one.** A
     reflection pass writes this SKILL's text, the gathering prompt's pattern
     list, and past candidates-table incident counts into its own `.jsonl`, so
     grepping any of them back inflates friction counts with the engine's own
     vocabulary. Dropping only the current-session file (the original RC-029
     fix) is insufficient — prior weekly-pass transcripts accumulate and each
     one re-contaminates every future scan. Before counting, drop **(a)** the
     newest/self file (by name or mtime) AND **(b)** any transcript that is
     itself a reflection output.

     **Detecting (b) — use the RELIABLE discriminators, NOT a bare path match.**
     Do **not** drop a file just because it matches the backlog path
     `brain/reflection/candidates.md`. That path is embedded verbatim in this
     skill's own catalog `description:`, which Claude Code loads into the system
     context of **every** session in this repo — so a `grep` for it fires ~1×
     as boilerplate on *genuine work* sessions too (verified RC-033: 1 hit on a
     real audit session vs 49 on the reflection pass). A `≥1` path threshold
     therefore wrongly DROPS real work sessions — an under-count, the opposite
     failure mode from RC-030's over-count. Instead, per file, drop it as a
     reflection output if **either**:
     - **(b1) Opening-prompt signal** — the transcript's first records contain
       the reflection scheduled-task header. `head -c 4000 "$f" | grep -qE
       'reflection-weekly|scheduled-task name=.reflection|WEEKLY pass'`. This is
       the strongest, cleanest signal (RC-033: fired on the pass file, silent on
       both work files). Or:
     - **(b2) Trio co-occurrence fingerprint** — the file matches **all three**
       of `gtimeout` AND `nbstripout` AND `index.lock` (a trio a normal dev
       session rarely names together). RC-033: reflection pass = 45/112/62;
       genuine work = 8 nbstripout but **0** gtimeout and **0** index.lock, so
       the AND correctly did not fire. Require all three, not any one.
     - (Optional b3) a HIGH-VOLUME path threshold (e.g. `candidates.md` ≥5) can
       corroborate, but never as the sole criterion — the boilerplate hit makes
       `≥1` unsafe.

     Report which files you dropped and by which signal. (Evidence RC-030: on
     the 2026-07-17 pass, contamination made `gtimeout` and "still failing" 100%
     artifacts and inflated ModuleNotFoundError 13→39, index.lock 34→60,
     nbstripout 125→207 across just two prior-pass files. Evidence RC-033: on
     2026-07-21 the bare path-sentinel fired on 4/4 in-scope files including both
     genuine work sessions; b1+b2 separated the 2 real from the 2 reflection
     files cleanly.)
   - **Grep per file, not multi-file.** A combined `grep -c pattern $(find …)`
     over these very-long single-line JSONL transcripts intermittently returns
     false-zero counts here; a `while read f; do grep … "$f"; done` loop (one
     grep invocation per file) is the reliable pattern.
4. **maps + reports** — [[SKILL_MAP]] § Future skills, `brain/generated/`
   hygiene/graph/daily-brief reports, `brain/TODO.md` blockers.

### 2. Cluster and verify

Cluster raw signals into named pains ACROSS sessions (one incident ≠ a
cluster). Subagent reports are leads, not facts — **verify on the live system
before accepting evidence** (e.g. the 2026-07-04 pass found "nbstripout
missing" was actually a PATH-context bug plus a TCC-blocked launchd job;
verification changed the fix entirely).

### 3. Score and decide

Use the rubric in `brain/reflection/candidates.md` § Scoring rubric
(recurrence LOW/MED/HIGH × cost S/M/L). You MAY revise the rubric without
asking, but RECORD the rule used in that section. Per cluster decide:
**new skill / automation / fix / nothing**.

### 4. Log — everything

Append to `brain/reflection/candidates.md`: new RC-nnn ids, evidence,
score, decision, status; update the Pass log. Rules:

- "nothing" gets a **reason** (so it never re-surfaces as new).
- Never re-propose: items at a later lifecycle stage (e.g. PACKAGING),
  SKILL_MAP-deferred items (Sherpa/Indeaverse), or closed-nothing entries —
  absent genuinely NEW evidence (then say what changed).
- Radar verdicts: Adopt / Borrow-pattern / Reference / Skip, with licence +
  integration cost (AGPL → reimplement, never vendor).

### 5. Execute accepted candidates end-to-end

Build-now items get built in the same session where feasible: code + passing
tests, registered in [[SKILL_MAP]] with an invocation line. Human-gated items
get status `blocked-human` with the exact step the human must take.

### 6. Graduate

candidate → built (tests green) → SKILL_MAP (registered) → `library/` entry
(generalized, decoupled, IP-scrubbed, licensed). The **human IP/strategy scrub
blocks any public publish** — no exceptions. Everything lands on the operator's
personal branch, never main.

## Guardrails

- No infra before signal; no ML before a rule-based baseline; no cross-import
  polymarket ↔ crypto (brain/CODEX.md — always in force).
- Token discipline: gathering is subagent work; a reflection pass should not
  read the vault end-to-end centrally.
- The engine proposes and builds *tooling*; it never edits research
  conclusions, numbers, or strategy decisions.
