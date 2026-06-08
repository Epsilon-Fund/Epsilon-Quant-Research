---
title: Operating Rhythms
created: 2026-06-07
status: active
owner: justin
project: infra
para: area
hubs:
  - VAULT_MAP
  - SKILL_MAP
tags:
  - obsidian
  - brain
  - cadence
  - hygiene
---

# Operating Rhythms

> Hygiene should not depend on memory or motivation. Run small, boring checks on a schedule.
> Generated reports are Markdown inside the vault so humans and agents read them — they don't regenerate them by hand.

Hub links: [[VAULT_MAP]] | [[SKILL_MAP]] | [[TODO]] | [[CODEX]] | [[COWORK]]

## Cadence

| Cadence | Skill | What it does | Output |
|---|---|---|---|
| Daily (start) | Daily Brief | Read changed notes, [[TODO]], recent git state | screen-length brief |
| During active work | Agent Scratch Log | Keep Codex/Cowork WIP separate until promoted | `scratch/<agent>/` (local-only) |
| Daily (end) | Chronicler | Record what changed and why | `brain/handoffs/` or hub status line |
| On branch close | Rock Tumbler | Convert messy branch work into a canonical findings note | `*_findings.md` |
| Weekly | Janitor | Link / metadata / duplicate / stale-task cleanup | `brain/generated/hygiene_report.md` |
| Weekly | Cartographer | Refresh [[VAULT_MAP]] + `brain/generated/GENERATED_INDEX.md` | updated maps |
| Monthly | Archive Review | Move closed branches to archive state; refresh roadmaps | updated status frontmatter |

## The weekly hygiene pass (concrete)

This is the one rhythm worth not skipping. It is one command plus a focused fix pass.

```bash
# 1. Generate the reports (finds issues; changes nothing)
python tools/brain_hygiene.py

# 2. Read the output
#    brain/GENERATED_INDEX.md          — navigation index by folder
#    brain/generated/hygiene_report.md — duplicates, broken links, orphans, missing frontmatter
#    brain/generated/stale_notes.md    — old / unlinked notes + stale TODOs

# 3. Hand the report to Codex for a focused Janitor fix pass.
#    The script never auto-edits notes — cleanup is a deliberate, reviewable step.
```

### Why a script, not discipline

The brain is meant to grow fast. Manual hygiene scales with note count and dies the first busy week. The script makes the cost of finding rot ~constant; the only human/agent judgement left is which fixes to apply. Schedule it (e.g. a weekly task) once you trust the output — until then run it by hand.

## Scheduling (when ready)

When you want this automated rather than manual, wire `tools/brain_hygiene.py` into a weekly scheduled task. The script is side-effect-light: it only writes under `brain/GENERATED_INDEX.md` and `brain/generated/` (both regenerable, and `generated/` is git-ignored). A schedule that just refreshes the reports — leaving the actual fixes to a human/Codex review — is safe to run unattended.

## Source-of-truth rules (Relay + Git)

- Edit shared Markdown in Obsidian/Relay first; use Git for snapshots, code, and audit history.
- Don't have two people edit the same canonical note simultaneously across Relay and Git.
- New `brain/**/*.md` is tracked automatically (gitignore was inverted 2026-06-07). Only `brain/generated/`, `brain/agents/locks/*.lock.md`, top-level `local_agents/`, and top-level `scratch/` are ignored.
- Keep generated data and large artifacts out of the note-collaboration layer.
- Invite keys go directly to collaborators, never into committed notes.
