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

# Codex Lane

> Codex's own operating surface. Codex may edit anything in this lane freely.
> Canonical knowledge lives in the shared brain — Codex does **not** use canonical hubs as scratchpads.

Hub links: [[VAULT_MAP]] | [[CODEX]] | [[TODO]] | [[cowork_lane]]

## Role

Codex is the **implementation agent**: long-running computation, producing CSVs / findings docs / scripts. Full operating contract is in [[CODEX]]. This lane holds Codex-specific prompting, style notes, and scratch.

## Startup (every session)

1. Read [[CODEX]] (implementation README, invariants, run env).
2. Read [[TODO]] (active threads, open tasks, blockers).
3. Read [[VAULT_MAP]] for where-to-write, then the relevant strategy hub.
4. For data-heavy work, read the relevant manifest before scanning raw folders.
5. Write WIP to `brain/agents/codex/scratch/YYYY-MM-DD.md`; promote durable results to the right `*_findings.md` and link the hub.

## Lane rules

- Scratch here (`scratch/`) is git-ignored and Codex-owned. Fast and messy is fine.
- Promote findings out of scratch via the Rock Tumbler / Chronicler skills ([[SKILL_MAP]]).
- If a canonical file needs editing and Cowork might also touch it, take ownership for the edit or leave a linked scratch note.
- Never put code in `brain/`; scripts go under the relevant project (or `tools/` for repo-level tooling).

## Scratch

`brain/agents/codex/scratch/YYYY-MM-DD.md` — create per session. Not tracked in Git by design.
