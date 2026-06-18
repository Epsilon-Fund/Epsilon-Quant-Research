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
| Daily (start) | Canon Check | Find the most-updated canon (`justin` vs `origin/main` after any `alvaro`/`justin` merge), catch up via [[MERGE_PROTOCOL]], and confirm LF normalization before any work | clean, up-to-date, LF-normalized branch |
| During active work | Agent Scratch Log | Keep Codex/Cowork WIP separate until promoted | `scratch/<agent>/` (local-only) |
| Daily (end) | Chronicler | Record what changed and why | `brain/handoffs/` or hub status line |
| On branch close | Rock Tumbler | Convert messy branch work into a canonical findings note | `*_findings.md` |
| Weekly | Janitor | Link / metadata / duplicate / stale-task cleanup | `brain/generated/hygiene_report.md` |
| Weekly | Cartographer | Refresh [[VAULT_MAP]] + `brain/generated/GENERATED_INDEX.md` | updated maps |
| Weekly | gbrain re-index | Re-import the vault + extract links so semantic search / graph traversal stay current (commands in [[gbrain_retrieval_layer]]) | refreshed gbrain index |
| Monthly | Archive Review | Move closed branches to archive state; refresh roadmaps | updated status frontmatter |

## The weekly hygiene pass (concrete)

This is the one rhythm worth not skipping. It is one command plus a focused fix pass.

```bash
# 1. Generate the reports (finds issues; changes nothing)
python tools/brain_hygiene.py

# 2. Read the output
#    brain/generated/GENERATED_INDEX.md — navigation index by folder
#    brain/generated/hygiene_report.md — duplicates, broken links, orphans, missing frontmatter
#    brain/generated/stale_notes.md    — old / unlinked notes + stale TODOs

# 3. Hand the report to Codex for a focused Janitor fix pass.
#    The script never auto-edits notes — cleanup is a deliberate, reviewable step.
```

### Why a script, not discipline

The brain is meant to grow fast. Manual hygiene scales with note count and dies the first busy week. The script makes the cost of finding rot ~constant; the only human/agent judgement left is which fixes to apply. Schedule it (e.g. a weekly task) once you trust the output — until then run it by hand.

## Scheduling (when ready)

When you want this automated rather than manual, wire `tools/brain_hygiene.py` into a weekly scheduled task. The script is side-effect-light: it only writes under `brain/generated/` (regenerable and git-ignored, including `brain/generated/GENERATED_INDEX.md`). A schedule that just refreshes the reports — leaving the actual fixes to a human/Codex review — is safe to run unattended.

## Source-of-truth rules (git branches)

- Git is the only sync layer. Each collaborator edits on their **personal branch** (named by GitHub handle); `main` is the integration branch — no direct commits to it. Merges follow [[MERGE_PROTOCOL]].
- Commit/push EOD to your own branch (the per-person `brain-commit-push` task); merge main into your branch at session start and after anyone merges to main.
- New `brain/**/*.md` is tracked automatically (gitignore was inverted 2026-06-07). Only `brain/generated/`, top-level `local_agents/`, and top-level `scratch/` are ignored — those last two are per-person state and never participate in any branch or merge.
- Keep generated data and large artifacts out of the note layer; `brain/generated/` is regenerable and never committed.
- **Line endings are LF in the index, cross-platform.** Mac (`justin`) and Windows (`alvaro`) share `main`; `.gitattributes` (`* text=auto eol=lf`) keeps history LF so merges never produce whitespace-only conflicts. Set `core.autocrlf false` + `core.eol lf` + `merge.renormalize true` per clone and let `.gitattributes` be the single source of truth — see [[MERGE_PROTOCOL]] § 6.
- **Daily canon check (start of day).** Before working, find the most-updated canon — usually `origin/main` after an `alvaro`/`justin` merge, or your own branch if you pushed last — then `git checkout <handle> && git merge main` ([[MERGE_PROTOCOL]] § 1) and `git add --renormalize .` to confirm LF before continuing.
