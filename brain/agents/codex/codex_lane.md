---
title: Codex Lane
created: 2026-06-07
status: active
owner: justin
project: infra
para: area
hubs:
  - VAULT_MAP
  - CODEX
tags:
  - agent-lane
  - codex
---

# Implementation-Agent Shared Role Convention (Codex / Claude Code)

> Shared implementation-role convention for all collaborators' coding agents — **Codex and Claude Code** both follow it.
> This is not a private per-person instruction file.
> Canonical knowledge lives in the shared brain — the implementation agent does **not** use canonical hubs as scratchpads.

Hub links: [[VAULT_MAP]] | [[CODEX]] | [[TODO]] | [[cowork_lane]]

## Role

The implementation agent (Codex *or* Claude Code) does long-running computation, producing CSVs / findings docs / scripts. Full operating contract is in [[CODEX]]. This shared role convention holds Codex-type startup rules and team conventions. Personal style/preferences live in `local_agents/codex.md`; per-person WIP lives in local scratch.

## Startup (every session)

1. Run the Agent Bootstrap — canonical copy in [[VAULT_MAP]] § Agent Bootstrap (seeds `local_agents/codex.md` from [[codex.local.template.md|codex.local.template]] if missing).
2. Read [[CODEX]] (shared implementation law, invariants, run env).
3. Read [[TODO]] (active threads, open tasks, blockers).
4. Read [[VAULT_MAP]] for where-to-write, then the relevant strategy hub.
5. For data-heavy work, read the relevant manifest before scanning raw folders.
6. Write WIP to `scratch/codex/YYYY-MM-DD.md` (top-level, local-only); promote durable results to the right `*_findings.md` and link the hub.

## Lane rules

- **You operate on a personal branch named after the operator's GitHub handle. Commit/push only to that branch — never main. Merge main into your branch at session start; all merges follow [[MERGE_PROTOCOL]].**
- Scratch in `scratch/codex/` is git-ignored, local-only, and Codex-owned on each machine. Fast and messy is fine.
- Your personal, per-person instructions live in `local_agents/codex.md` (local, private) — read it first.
- Promote findings out of scratch via the Rock Tumbler / Chronicler skills ([[SKILL_MAP]]).
- Edits to canonical files ride your personal branch like everything else; genuine overlaps surface as merge conflicts and are resolved per [[MERGE_PROTOCOL]].
- Never put code in `brain/`; scripts go under the relevant project (or `tools/` for repo-level tooling).

## Scratch

`scratch/codex/YYYY-MM-DD.md` (top-level) — create per session. Local-only: git-ignored, never on any branch, so it cannot collide with a collaborator's WIP. (The lane operating doc you are reading IS shared via Git; only scratch is private.)
