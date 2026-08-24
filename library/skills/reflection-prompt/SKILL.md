---
name: reflection-prompt
description: >
  Mine your own recent work for recurring pain and turn it into build
  decisions. Use on a recurring cadence (e.g. weekly) or on demand when asked
  "what should we build/automate next", "review recent sessions for friction",
  or after a burst of repetitive manual work. Gathers signals from git
  history, working notes, and session transcripts; clusters pain ACROSS
  sessions; scores recurrence × build-cost; decides per cluster (new skill /
  automation / fix / nothing) and logs every decision — including "nothing",
  with a reason — to a persistent backlog so rejected ideas stay rejected.
license: Apache-2.0
---

# Reflection Prompt

A standing discovery loop for agent-assisted projects: instead of noticing
friction ad hoc (and re-proposing the same rejected ideas forever), mine the
recent record systematically, decide deliberately, and **write every decision
down**. The backlog file is the product; the builds are a side effect.

## The backlog file

Maintain ONE persistent backlog (suggested: `reflection/candidates.md` at your
project root). Every candidate gets a stable id (`RC-001`, `RC-002`, … never
reused), the evidence, the score, the decision, and a lifecycle status. The
file is append-mostly: statuses change, history does not.

## Procedure

### 1. Gather signals (read-only)

Sweep whatever record your project keeps — cheap and parallel where your
harness allows:

- **git history** since the last pass: churn hotspots, "fix"/"again"/"rework"
  messages, repeated chores, merge pain, test churn. Evidence = hashes +
  dates + counts.
- **working notes / scratch files**: explicit complaints, planned-but-unbuilt
  items, deferred chores.
- **session transcripts** (if your agent tooling stores them): grep-sample for
  friction markers — errors, retries, repeated commands, corrections. Sample;
  never read everything; state your confidence honestly.
- **project docs**: deferred-item lists, TODO blockers, generated reports.

### 2. Cluster and verify

Cluster raw signals into named pains ACROSS sessions — one incident is not a
cluster. Treat gathered reports as **leads, not facts**: verify on the live
system before accepting evidence (a "missing tool" lead often turns out to be
a PATH bug; the fix changes entirely).

### 3. Score: recurrence × build-cost

| Axis | Bands |
|---|---|
| **Recurrence** | LOW = seen once · MED = 2–3 distinct commits/sessions · HIGH = 4+ (or an automated system failing repeatedly) |
| **Build-cost** | S = ≤ ~1 focused agent-hour · M = ~half-day, touches conventions or several files · L = multi-day / new subsystem |

Decision rule (v1 — you MAY revise it, but record the rule you used in the
backlog so scores stay comparable):

- HIGH × (S|M) → **build now**
- HIGH × L → **plan** (schedule, don't start)
- MED × S → **build now** only if it unblocks an automated system or a live
  thread; else **watch**
- MED × (M|L) → **plan** or **watch**
- LOW × anything → **nothing (watch)** — unless it hard-blocks an automated
  pipeline, then **fix**

### 4. Log everything

Append to the backlog: new ids, evidence, score, decision, status. Two rules
that make the loop converge instead of thrash:

- **"Nothing" gets a reason.** A rejected idea with a recorded reason never
  re-surfaces as if it were new. Include the re-open condition if there is one
  ("re-open only if X happens again").
- **Never re-propose** items at a later lifecycle stage, explicitly-deferred
  items, or closed-nothing entries — absent genuinely NEW evidence (then say
  what changed).

### 5. Execute accepted candidates

Build-now items get built in the same session where feasible: code + passing
tests + registered wherever your project catalogs its tooling. Items needing
a human decision get a status naming the exact step the human must take.

## Guardrails

- Gathering is read-only; the pass never edits project conclusions, numbers,
  or decisions — it builds *tooling*.
- Don't burn your context reading the whole record centrally; sample, and
  delegate bulk reading to subagents if your harness has them.
- Keep the backlog authoritative for DISCOVERY state only; your project's
  real catalog/docs stay canonical for what's built and shipped.

## Worked example

See [EXAMPLE.md](EXAMPLE.md) — a fictional two-pass backlog showing the
rubric applied, a build-now entry, a plan entry, and a closed-nothing entry
whose recorded reason stops it from coming back.
