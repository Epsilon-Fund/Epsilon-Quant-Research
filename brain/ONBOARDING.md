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

Hub links: [[VAULT_MAP]] | [[MERGE_PROTOCOL]] | [[START_RESEARCH_IDEA]] | [[CODEX]] | [[COWORK]] | [[TODO]] | [[OPERATING_RHYTHMS]]

## Agent Bootstrap (do this before anything else)

Run the Agent Bootstrap — canonical copy in [[VAULT_MAP]] § Agent Bootstrap.

## 1. What this repo is

A two-project quant research monorepo. The projects **share no code** — separate venvs, separate dependencies, never cross-import.

- **Polymarket alpha** — `polymarket/` — prediction-market microstructure, copytrade, MM, OD, dali lineage.
- **Crypto systematic** — `live_trading/` + `topics/` + `infrastructure/` — momentum / stat-arb / breakout.

Full map: [[VAULT_MAP]].

## 2. The brain (how knowledge is organized)

`brain/` is an Obsidian vault inside the git repo. Knowledge is **shared and canonical**; per-person overlays and fast-moving scratch are **local and private** (git-ignored, so they never appear on any branch).

| Layer | What it is | Who edits |
|---|---|---|
| **Canonical hubs** | Durable truth: [[VAULT_MAP]], [[TODO]], [[POLYMARKET_BRAIN]], strategy hubs | anyone, on their own branch; changes reach `main` via [[MERGE_PROTOCOL]] |
| **Findings notes** | Research results under `polymarket/research/notes/<cluster>/` | the author, promoted from scratch |
| **Agent lanes** | Shared role conventions [[codex_lane]], [[cowork_lane]]; per-person overlays in `local_agents/<agent>.md` and scratch in `scratch/<agent>/` (local-only) | shared conventions by all; local overlays/scratch by that person |
| **Generated reports** | `brain/generated/` — regenerable, **not committed**; regenerate locally with `python tools/brain_hygiene.py` | nobody (a script writes them) |

**The one rule that prevents collisions:** agents never use a canonical hub as a scratchpad. Draft in your own scratch (`scratch/<agent>/`, local to your machine), then promote durable results into the canonical note and link it. Concurrent edits are safe because each person works on their own branch; genuine overlaps surface as merge conflicts and are resolved per [[MERGE_PROTOCOL]].

## 3. Where to write things

See [[VAULT_MAP]] § "Where to write things" for the full table. Short version: findings go in the matching `notes/<cluster>/` folder with a hub backlink; cross-thread context goes in `brain/handoffs/<date>_<topic>.md`; task updates edit [[TODO]] directly; **code never goes in `brain/`**.

Every durable note gets YAML frontmatter and a plain-English `## Summary` near the top — the standard is in [[CODEX]] § Markdown quality standard.

## 4. Collaboration model (git branches)

Git is the only sync layer. The model is **one personal branch per collaborator, named by their GitHub handle** (e.g. `justin`, `alvaro`). `main` is the shared integration branch.

- **No direct commits to main.** Nobody — human or agent — commits or pushes to `main` directly. Merges into main are deliberate, agent-assisted steps per [[MERGE_PROTOCOL]]. Recommended: a GitHub branch-protection rule on `main` blocking direct pushes, if the plan supports it.
- **Create your branch ONCE off main:** `git checkout main && git pull && git checkout -b <handle>`.
- **EOD: commit + push to YOUR branch only.** The per-person `brain-commit-push` scheduled task does this; it now refuses to run on main, never force-pushes, and does not pull or merge main.
- **Session start, AND after anyone merges to main:** catch up with `git checkout <handle> && git merge main` — resolve conflicts via [[MERGE_PROTOCOL]]. Branches that don't pull main in regularly drift and make merges painful.
- **Merge to main in small, frequent units** (a finished note, a completed task) via [[MERGE_PROTOCOL]] — not big infrequent dumps. After a merge, everyone pulls main into their branch.
- **Personal state never participates.** `local_agents/<handle>.md` and `scratch/<handle>/` are git-ignored — they exist only on your machine and can never conflict in any branch or merge.
- **Shared templates:** `brain/agents/templates/` is tracked under `brain/` so every machine can seed its private overlay.
- **Tracked vs ignored:** `brain/**/*.md` is tracked; only `brain/generated/`, top-level `local_agents/`, and top-level `scratch/` are ignored.

### Onboarding a new collaborator

Exact first-run steps (copy-pasteable; replace `<handle>` with your GitHub handle):

1. **Clone the repo:**
   ```bash
   git clone <repo-url> epsilon-quant-research
   cd epsilon-quant-research
   ```
2. **Open the folder as an Obsidian vault** (Obsidian → "Open folder as vault" → the repo root). Nothing to install — no sync plugin; git is the sync layer.
3. **Run the Agent Bootstrap** ([[VAULT_MAP]] § Agent Bootstrap) in your first agent session — it seeds `local_agents/<role>.md` from `brain/agents/templates/` for you. These overlay files are git-ignored and personal; put your "current focus / waiting-on" state there.
4. **Create your personal branch once:**
   ```bash
   git checkout main && git pull && git checkout -b <handle>
   git push -u origin <handle>
   ```
5. **Set up your daily commit/push task** (scheduled task or cron) targeting **your own branch, never main**. It should: stage only `brain/`, `polymarket/research/notes/`, `docs/`, `README.md`, `tools/` Markdown; commit; pull your own remote branch (`git pull --no-rebase origin <handle>`); push to `<handle>`; abort and report on any conflict or error; never force-push; never pull or merge main. This task is per-person — Justin's is configured on his machine; you configure your own.
6. **Every session start (and after any merge to main):** `git checkout <handle> && git merge main`. Conflicts → [[MERGE_PROTOCOL]].
7. **Where things go:** durable notes follow [[VAULT_MAP]] § Where to write things (findings into `polymarket/research/notes/<cluster>/` etc., with hub backlinks); personal WIP goes in `scratch/<handle>/` and personal preferences in `local_agents/` (both git-ignored, never shared).

## 5. Starting a session (human or agent)

1. Catch up from main: `git checkout <handle> && git merge main` (conflicts → [[MERGE_PROTOCOL]]).
2. Run the Agent Bootstrap ([[VAULT_MAP]] § Agent Bootstrap).
3. Read your shared role convention: [[codex_lane]] or [[cowork_lane]].
4. Read [[TODO]] — the authoritative live task list — before suggesting next actions.
5. For data-heavy work, read the relevant manifest (see [[VAULT_MAP]]) before scanning raw folders.
6. Write WIP to your scratch (`scratch/<agent>/`, local-only); promote durable results to the right note and link the hub.

## 6. Starting a new research idea

Use [[START_RESEARCH_IDEA]] as the set guide. In short: start from [[VAULT_MAP]], read the relevant lane and project hub, check nearby existing notes, draft in scratch/chat first, then promote durable outputs to the right project note folder with hub backlinks. The "brain" means the linked Markdown universe, not only the `brain/` folder.

## 7. Keeping it clean

Hygiene is a script, not a chore: `python tools/brain_hygiene.py` regenerates the index and flags duplicates, broken links, orphans, and missing metadata into `brain/generated/`. It **finds** issues; a person or agent fixes them in a reviewed pass. Cadence is in [[OPERATING_RHYTHMS]]; a weekly scan runs automatically.
