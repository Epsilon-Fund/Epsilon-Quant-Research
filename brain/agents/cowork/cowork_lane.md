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

# Cowork Lane

> Cowork / Claude Code's own operating surface. Edit anything in this lane freely.
> Canonical knowledge lives in the shared brain — Cowork does **not** use canonical hubs as scratchpads.

Hub links: [[VAULT_MAP]] | [[COWORK]] | [[TODO]] | [[codex_lane]]

## Role

Cowork is the **strategic agent**: framing questions, drafting pre-registered Codex prompts, interpreting Codex outputs, and maintaining the living docs in `brain/` and `polymarket/research/notes/`. Full operating contract is in [[COWORK]]. This lane holds Cowork-specific prompting, style notes, and scratch.

## Startup (every session)

1. Read [[COWORK]] (strategic README, active clusters, prompt discipline).
2. Read [[CODEX]], [[TODO]], [[POLYMARKET_BRAIN]] before drafting prompts.
3. Read [[VAULT_MAP]] for where-to-write and current branch status.
4. Write WIP / draft prompts to `brain/agents/cowork/scratch/YYYY-MM-DD.md`; keep finished Codex prompts in chat, not in repo files.

## Lane rules

- Scratch here (`scratch/`) is git-ignored and Cowork-owned.
- Every Cowork-authored Codex prompt starts with the read-order preamble in [[COWORK]] § Cowork prompt discipline.
- Interpret Codex output through [[CODEX]] § Realism calibration before deciding CLOSE / enhance / reopen.
- Update [[TODO]] and the relevant hub via short, intentional Chronicler passes — not constant churn.
- If a canonical file needs editing and Codex might also touch it, take ownership or leave a linked scratch note.

## Scratch

`brain/agents/cowork/scratch/YYYY-MM-DD.md` — create per session. Not tracked in Git by design.
