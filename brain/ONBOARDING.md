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

Hub links: [[VAULT_MAP]] | [[START_RESEARCH_IDEA]] | [[CODEX]] | [[COWORK]] | [[TODO]] | [[OPERATING_RHYTHMS]]

## Agent Bootstrap (do this before anything else)

1. Determine your role: Codex -> `local_agents/codex.md`; Cowork/Claude Code -> `local_agents/cowork.md`.
2. If that file does not exist, create `local_agents/` and copy the matching template from `brain/agents/templates/<role>.local.template.md` into it, then tell the user "seeded your local <role> overlay - edit it to set your personal preferences."
3. Read your `local_agents/<role>.md` overlay (personal style), then the shared law `brain/CODEX.md` or `brain/COWORK.md`, then `brain/VAULT_MAP.md`, then `brain/TODO.md`.

Precedence: personal overlay = voice/preferences; shared `CODEX`/`COWORK` + repo invariants = law (always win).

## 1. What this repo is

A two-project quant research monorepo. The projects **share no code** — separate venvs, separate dependencies, never cross-import.

- **Polymarket alpha** — `polymarket/` — prediction-market microstructure, copytrade, MM, OD, dali lineage.
- **Crypto systematic** — `live_trading/` + `topics/` + `infrastructure/` — momentum / stat-arb / breakout.

Full map: [[VAULT_MAP]].

## 2. The brain (how knowledge is organized)

`brain/` is an Obsidian vault. Knowledge is **shared and canonical**; per-person overlays and fast-moving scratch are **local and private** when Relay is scoped as described below.

| Layer | What it is | Who edits |
|---|---|---|
| **Canonical hubs** | Durable truth: [[VAULT_MAP]], [[TODO]], [[POLYMARKET_BRAIN]], strategy hubs | one owner at a time, via short intentional passes |
| **Findings notes** | Research results under `polymarket/research/notes/<cluster>/` | the author, promoted from scratch |
| **Agent lanes** | Shared role conventions [[codex_lane]], [[cowork_lane]]; per-person overlays in `local_agents/<agent>.md` and scratch in `scratch/<agent>/` (local-only) | shared conventions by all; local overlays/scratch by that person |
| **Generated reports** | `brain/generated/` — regenerable, **not committed** (synced via Relay so both see them) | nobody (a script writes them) |

**The one rule that prevents collisions:** agents never use a canonical hub as a scratchpad. Draft in your own scratch (`scratch/<agent>/`, local to your machine), then promote durable results into the canonical note and link it.

**Relay shares the whole `brain/` folder** (and the research folders), so [[VAULT_MAP]], [[TODO]], the hubs, `brain/handoffs/`, generated reports, shared lane docs, templates, and the edit locks are all **live for both collaborators**. The things Relay does **not** share are top-level `local_agents/` overlays and top-level `scratch/`; both are local-only and never committed. Relay's boundary is its shared-folder list, not `.gitignore`; those folders simply stay off that list.

For shared durable Markdown, use the cooperative edit guard before editing:

```bash
python3 tools/brain_edit_guard.py acquire --agent <codex|cowork|justin> --path <path.md> --intent "<short reason>"
# edit the file
python3 tools/brain_edit_guard.py release --agent <codex|cowork|justin> --path <path.md>
```

If the guard says another agent owns the file, stop and ask Justin which surface should be edited. If the agent is unsure whether it is Codex or Cowork, it should not edit canonical files yet; read the lane docs and ask for confirmation. Cross-lane edits are allowed only when Justin explicitly asks for them; use `--allow-cross-lane` so the intent is visible.

## 3. Where to write things

See [[VAULT_MAP]] § "Where to write things" for the full table. Short version: findings go in the matching `notes/<cluster>/` folder with a hub backlink; cross-thread context goes in `brain/handoffs/<date>_<topic>.md`; task updates edit [[TODO]] directly; **code never goes in `brain/`**.

Every durable note gets YAML frontmatter and a plain-English `## Summary` near the top — the standard is in [[CODEX]] § Markdown quality standard.

## 4. Sync model: Relay + Git

- **Relay** syncs Markdown **live** between machines. Shared: the research folders (`Attachments/`, `archive/`, `docs/`, `infrastructure/`, `live_trading/`, `meetings/`, `midas/`, `newsletters/`, `polymarket/`, `topics/`) and **all of `brain/`** — maps, hubs, [[TODO]], handoffs, generated reports, and edit locks. So the control-plane is live for both of you.
- **Not shared on Relay:** top-level `local_agents/<agent>.md` (per-person instruction overlay) and top-level `scratch/<agent>/` (per-person WIP). Both stay local to each machine. They must not be added to Relay's shared-folder list.
- **Shared templates:** `brain/agents/templates/` is shared under `brain/` so every machine can seed its private overlay.
- **Git** is the snapshot/audit layer. `brain/**/*.md` is tracked; only `brain/generated/`, `brain/agents/locks/*.lock.md`, `local_agents/`, and `scratch/` are ignored.
- **Two agents, same file:** before editing a shared canonical note, acquire an edit lock with `tools/brain_edit_guard.py`. Locks sync live via Relay, so the other agent sees the claim and waits. Relay's CRDT avoids hard merge conflicts; the lock stops two agents logically clobbering the same section.
- **Code/data** (not Relay-synced) uses normal Git discipline: pull before you push.
- Invite keys are sent directly between collaborators — never committed.

## 5. Starting a session (human or agent)

1. Run the Agent Bootstrap above.
2. Read your shared role convention: [[codex_lane]] or [[cowork_lane]].
3. Read [[TODO]] — the authoritative live task list — before suggesting next actions.
4. For data-heavy work, read the relevant manifest (see [[VAULT_MAP]]) before scanning raw folders.
5. Write WIP to your scratch (`scratch/<agent>/`, local-only); promote durable results to the right note and link the hub.

## 6. Starting a new research idea

Use [[START_RESEARCH_IDEA]] as the set guide. In short: start from [[VAULT_MAP]], read the relevant lane and project hub, check nearby existing notes, draft in scratch/chat first, then promote durable outputs to the right project note folder with hub backlinks. The "brain" means the linked Markdown universe, not only the `brain/` folder.

## 7. Keeping it clean

Hygiene is a script, not a chore: `python tools/brain_hygiene.py` regenerates the index and flags duplicates, broken links, orphans, and missing metadata into `brain/generated/`. It **finds** issues; a person or agent fixes them in a reviewed pass. Cadence is in [[OPERATING_RHYTHMS]]; a weekly scan runs automatically.
