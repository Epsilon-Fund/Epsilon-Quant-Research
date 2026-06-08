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

# Codex Shared Role Convention

> Shared Codex-role convention for all collaborators' Codex agents.
> This is not a private per-person instruction file.
> Canonical knowledge lives in the shared brain — Codex does **not** use canonical hubs as scratchpads.

Hub links: [[VAULT_MAP]] | [[CODEX]] | [[TODO]] | [[cowork_lane]]

## Role

Codex is the **implementation agent**: long-running computation, producing CSVs / findings docs / scripts. Full operating contract is in [[CODEX]]. This shared role convention holds Codex-type startup rules and team conventions. Personal style/preferences live in `local_agents/codex.md`; per-person WIP lives in local scratch.

## Startup (every session)

1. Run Agent Bootstrap: seed/read `local_agents/codex.md` from [[codex.local.template.md|codex.local.template]] if missing.
2. Read [[CODEX]] (shared implementation law, invariants, run env).
3. Read [[TODO]] (active threads, open tasks, blockers).
4. Read [[VAULT_MAP]] for where-to-write, then the relevant strategy hub.
5. For data-heavy work, read the relevant manifest before scanning raw folders.
6. Write WIP to `scratch/codex/YYYY-MM-DD.md` (top-level, local-only); promote durable results to the right `*_findings.md` and link the hub.
7. Before editing shared durable Markdown, acquire a lock with `python3 tools/brain_edit_guard.py acquire --agent codex --path <path.md> --intent "<short reason>"`.

## Lane rules

- Scratch in `scratch/codex/` is git-ignored, Relay-excluded, and Codex-owned on each machine. Fast and messy is fine.
- Your personal, per-person instructions live in `local_agents/codex.md` (local, private) — read it first.
- Promote findings out of scratch via the Rock Tumbler / Chronicler skills ([[SKILL_MAP]]).
- If a canonical file needs editing, acquire an edit lock first. If Cowork already owns the lock, stop and ask Justin which surface should be edited.
- Cross-lane edits are allowed only when Justin explicitly asks for them; use `--allow-cross-lane` so the override is visible.
- Never put code in `brain/`; scripts go under the relevant project (or `tools/` for repo-level tooling).

## Scratch

`scratch/codex/YYYY-MM-DD.md` (top-level) — create per session. Local-only: git-ignored and never Relay-shared, so it cannot collide with a collaborator's WIP. (The lane operating doc you are reading IS shared via Relay/Git; only scratch is private.)
