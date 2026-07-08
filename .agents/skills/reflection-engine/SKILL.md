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
   Two gotchas (learned the 2026-07-06 pass — RC-029), both mandatory for an
   accurate scan:
   - **Exclude the current session's own transcript.** The live pass writes
     this SKILL's text and the gathering prompt's pattern list into its own
     `.jsonl`, so grepping it back inflates every friction count with the
     pass's own vocabulary. Drop the newest/self file (by name or mtime) first.
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
