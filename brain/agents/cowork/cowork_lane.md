---
title: Cowork Lane
created: 2026-06-07
status: active
owner: justin
project: infra
para: area
hubs:
  - VAULT_MAP
  - COWORK
tags:
  - agent-lane
  - cowork
---

# Cowork Shared Role Convention

> Shared Cowork-role (orchestration) convention for all collaborators' strategic agents. Coding agents (Codex / Claude Code) use [[codex_lane]] instead.
> This is not a private per-person instruction file.
> Canonical knowledge lives in the shared brain — Cowork does **not** use canonical hubs as scratchpads.

Hub links: [[VAULT_MAP]] | [[COWORK]] | [[TODO]] | [[codex_lane]]

## Role

Cowork is the **strategic agent**: framing questions, drafting pre-registered Codex prompts, interpreting Codex outputs, and maintaining the living docs in `brain/` and `polymarket/research/notes/`. Full operating contract is in [[COWORK]]. This shared role convention holds Cowork-type startup rules and team conventions. Personal style/preferences live in `local_agents/cowork.md`; per-person WIP lives in local scratch.

## Startup (every session)

1. Run the Agent Bootstrap — canonical copy in [[VAULT_MAP]] § Agent Bootstrap (seeds `local_agents/cowork.md` from [[cowork.local.template.md|cowork.local.template]] if missing).
2. Read [[COWORK]] (shared strategic law, active clusters, prompt discipline).
3. Read [[CODEX]], [[TODO]], [[POLYMARKET_BRAIN]] before drafting prompts.
4. Read [[VAULT_MAP]] for where-to-write and current branch status.
5. Write WIP / draft prompts to `scratch/cowork/YYYY-MM-DD.md` (top-level, local-only); keep finished Codex prompts in chat, not in repo files.

## Lane rules

- **You operate on a personal branch named after the operator's GitHub handle. Commit/push only to that branch — never main. Merge main into your branch at session start; all merges follow [[MERGE_PROTOCOL]].**
- Scratch in `scratch/cowork/` is git-ignored, local-only, and Cowork-owned on each machine.
- Your personal, per-person instructions live in `local_agents/cowork.md` (local, private) — read it first.
- Every Cowork-authored Codex prompt starts with the read-order preamble in [[COWORK]] § Cowork prompt discipline.
- Interpret Codex output through [[CODEX]] § Realism calibration before deciding CLOSE / enhance / reopen.
- Update [[TODO]] and the relevant hub via short, intentional Chronicler passes — not constant churn.
- Edits to canonical files ride your personal branch like everything else; genuine overlaps surface as merge conflicts and are resolved per [[MERGE_PROTOCOL]].

## Scratch

`scratch/cowork/YYYY-MM-DD.md` (top-level) — create per session. Local-only: git-ignored, never on any branch, so it cannot collide with a collaborator's WIP. (The lane operating doc you are reading IS shared via Git; only scratch is private.)
