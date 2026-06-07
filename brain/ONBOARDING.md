---
title: Onboarding — Workspace Structure
created: 2026-06-07
status: active
owner: justin
project: infra
para: area
hubs:
  - VAULT_MAP
tags:
  - onboarding
  - collaboration
  - brain
---

# Onboarding — How This Workspace Is Organized

> For a new collaborator **or their agent**. Read this once, then work from [[VAULT_MAP]].
> Written to be agent-agnostic: nothing here assumes a specific tool. If you are an AI agent
> opening this repo, treat this as your orientation contract.

Hub links: [[VAULT_MAP]] | [[CODEX]] | [[COWORK]] | [[TODO]] | [[OPERATING_RHYTHMS]]

## 1. What this repo is

A two-project quant research monorepo. The projects **share no code** — separate venvs, separate dependencies, never cross-import.

- **Polymarket alpha** — `polymarket/` — prediction-market microstructure, copytrade, MM, OD, dali lineage.
- **Crypto systematic** — `live_trading/` + `topics/` + `infrastructure/` — momentum / stat-arb / breakout.

Full map: [[VAULT_MAP]].

## 2. The brain (how knowledge is organized)

`brain/` is an Obsidian vault. Knowledge is **shared and canonical**; fast-moving scratch is **per-agent and private**.

| Layer | What it is | Who edits |
|---|---|---|
| **Canonical hubs** | Durable truth: [[VAULT_MAP]], [[TODO]], [[POLYMARKET_BRAIN]], strategy hubs | one owner at a time, via short intentional passes |
| **Findings notes** | Research results under `polymarket/research/notes/<cluster>/` | the author, promoted from scratch |
| **Agent lanes** | Per-agent operating docs + scratch: [[codex_lane]], [[cowork_lane]] | that agent only |
| **Generated reports** | `brain/generated/` — regenerable, **not committed** | nobody (a script writes them) |

**The one rule that prevents collisions:** agents never use a canonical hub as a scratchpad. Draft in your own lane (`brain/agents/<agent>/scratch/`), then promote durable results into the canonical note and link it.

## 3. Where to write things

See [[VAULT_MAP]] § "Where to write things" for the full table. Short version: findings go in the matching `notes/<cluster>/` folder with a hub backlink; cross-thread context goes in `brain/handoffs/<date>_<topic>.md`; task updates edit [[TODO]] directly; **code never goes in `brain/`**.

Every durable note gets YAML frontmatter and a plain-English `## Summary` near the top — the standard is in [[CODEX]] § Markdown quality standard.

## 4. Sync model: Relay + Git

- **Relay** syncs the Markdown brain **live** between machines. Edit notes in Obsidian/Relay; both people see changes in real time.
- **Git** is the snapshot/audit layer for code, data, and history. `brain/**/*.md` is tracked (so notes also version in Git); only `brain/generated/` and `brain/agents/*/scratch/` are ignored.
- **Don't** edit the same canonical note simultaneously through both Relay and Git. For code/data (not Relay-synced), use normal Git discipline: pull before you push.
- Invite keys are sent directly between collaborators — never committed.

## 5. Starting a session (human or agent)

1. Read [[VAULT_MAP]] — the start-here surface.
2. Read your lane: [[codex_lane]] or [[cowork_lane]].
3. Read [[TODO]] — the authoritative live task list — before suggesting next actions.
4. For data-heavy work, read the relevant manifest (see [[VAULT_MAP]]) before scanning raw folders.
5. Write WIP to your scratch lane; promote durable results to the right note and link the hub.

## 6. Keeping it clean

Hygiene is a script, not a chore: `python tools/brain_hygiene.py` regenerates the index and flags duplicates, broken links, orphans, and missing metadata into `brain/generated/`. It **finds** issues; a person or agent fixes them in a reviewed pass. Cadence is in [[OPERATING_RHYTHMS]]; a weekly scan runs automatically.
