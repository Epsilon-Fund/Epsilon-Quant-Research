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

## Core skills

| Skill | Trigger | Reads | Produces | Owner |
|---|---|---|---|---|
| **Daily Brief** | Start of a working day | changed notes, [[TODO]], recent git log | short "what changed / what matters / what's next" note | either agent |
| **Agent Scratch Log** | Any fast WIP that isn't canonical yet | the task at hand | `brain/agents/<agent>/scratch/YYYY-MM-DD.md` | active agent only |
| **Chronicler** | End of a working session, or any decision/closure | the day's scratch + diffs | dated handoff in `brain/handoffs/`, or a decision line in the right hub | assigned chronicler |
| **Rock Tumbler** | A research branch closes or matures | messy scratch + results | one clean canonical `*_findings.md` with links + decision | branch owner |
| **Janitor** | Weekly, or after a burst of new notes | hygiene report | small link/metadata/dedupe fixes (by Codex) | Codex |
| **Cartographer** | New branch, renamed folder, or stale map | folder tree + hubs | refreshed [[VAULT_MAP]] / `generated/GENERATED_INDEX.md` | either agent |
| **Librarian** | New reused shorthand or table column | the term in context | definition in [[glossary]] / [[polymarket_table_dictionary]] | either agent |

## Skill detail

### Daily Brief
Summarize what changed since yesterday, what matters, and the next falsifiable action. Keep it to a screen. Pull recently-changed notes and stale TODOs from `brain/generated/` (see Janitor) rather than re-reading everything.

### Agent Scratch Log
Each agent writes fast, messy work-in-progress to its own lane (`brain/agents/codex/scratch/` or `brain/agents/cowork/scratch/`). This is the collision-avoidance rule: **agents do not draft inside canonical hubs.** Scratch is git-ignored by default; promote anything durable via Chronicler / Rock Tumbler.

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

## Future skills (deferred)

From [[OBSIDIAN_INFRA_ROADMAP]] — build only when the basics earn their keep:

- **Sherpa** — route a human/agent to the right context pack for a task.
- **Graph audit** — Graphify-style orphan/hub/dead-end/stale-branch report (Phase 3).
- **Indeaverse** — idea-graph + branch registry navigable by concept, not folder (Phase 5).
