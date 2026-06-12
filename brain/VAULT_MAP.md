---
title: Vault Map
created: 2026-06-07
status: active
owner: justin
project: infra
para: area
hubs:
  - CODEX
  - COWORK
tags:
  - obsidian
  - brain
  - navigation
---

# Vault Map

> The single start-here surface for humans and agents. Read this before scanning folders.
> If you only read one file to orient, read this one, then the hub it points you to.

Hub links: [[CODEX]] | [[COWORK]] | [[TODO]] | [[ONBOARDING]] | [[MERGE_PROTOCOL]] | [[START_RESEARCH_IDEA]] | [[POLYMARKET_BRAIN]] | [[glossary]] | [[SKILL_MAP]] | [[OPERATING_RHYTHMS]] | [[OBSIDIAN_INFRA_ROADMAP]] | [[COWORK_MIGRATION]]

## What this repo is

A two-project quantitative research monorepo. The projects **share no code** — separate venvs, separate dependencies, never cross-import.

| Project | Root | What it does |
|---|---|---|
| **Polymarket alpha** | `polymarket/research/` + `polymarket/execution/` (Midas) | Prediction-market trading: copytrade, market-making (MM), options-delta (OD), and the dali research lineage |
| **Crypto live trading** | `live_trading/` + `topics/` + `infrastructure/` | Momentum / stat-arb / BB-breakout on Binance; walk-forward + CPCV research engine |

## Agent Bootstrap (do this before anything else)

> This is the **canonical copy** of the Agent Bootstrap. Every other doc points here — do not duplicate the block.

1. Determine your role: **implementation agent (Codex *or* Claude Code)** -> `local_agents/codex.md`; **orchestration agent (Cowork)** -> `local_agents/cowork.md`.
2. If that file does not exist, create `local_agents/` and copy the matching template from `brain/agents/templates/<role>.local.template.md` into it, then tell the user "seeded your local <role> overlay - edit it to set your personal preferences."
3. Read your `local_agents/<role>.md` overlay (personal style), then the shared law `brain/CODEX.md` or `brain/COWORK.md`, then `brain/VAULT_MAP.md`, then `brain/TODO.md`.
4. You operate on a personal branch named after the operator's GitHub handle. Commit/push only to that branch — never main. Merge main into your branch at session start; all merges follow [[MERGE_PROTOCOL]].

Precedence: personal overlay = voice/preferences; shared `CODEX`/`COWORK` + repo invariants = law (always win).

## Start here (reading order)

After bootstrap:

1. **This file** — orientation + where things live.
2. Your shared role convention: [[codex_lane]] (implementation: Codex / Claude Code) or [[cowork_lane]] (orchestration: Cowork).
3. For a new idea: [[START_RESEARCH_IDEA]] — fresh-agent workflow, idea-card shape, and where first durable notes go.
4. For Polymarket work: [[POLYMARKET_BRAIN]] → the relevant strategy hub.
5. For data-heavy work: the relevant manifest (see Generated Reports + Data Manifests below) before scanning raw folders.

## Top-level folder map

| Path | Owns |
|---|---|
| `brain/` | Git-tracked context hub: maps, hubs, task list, agent lanes, handoffs (this folder) |
| `local_agents/` | Local-only per-person agent instruction overlays; git-ignored, never on any branch |
| `polymarket/research/` | Polymarket research code, notes, data manifests |
| `polymarket/execution/` (+ `midas/`) | Polymarket execution bot |
| `live_trading/` | Unified Streamlit live-trading app + dashboards |
| `topics/` | Crypto strategy research (momentum, stat-arb, BB-breakout, CPCV) |
| `infrastructure/` | Walk-forward + CPCV engines for crypto |
| `docs/` | Crypto strategy + data references |
| `tools/` | Repo-level tooling (e.g. `brain_hygiene.py`) |
| `Attachments/` | Obsidian attachment default (images/PDFs) |
| `archive/` | Closed / historical material |

## Core hubs

| Hub | Owns | Read when |
|---|---|---|
| [[CODEX]] | Implementation-agent README: invariants, run env, where to write | Every implementation-agent (Codex / Claude Code) session start |
| [[COWORK]] | Strategic-agent README: active clusters, prompt discipline | Every Cowork session start |
| [[codex.local.template.md|codex.local.template]] | Shared template for seeding `local_agents/codex.md` | Fresh Codex / Claude Code machine / missing local overlay |
| [[cowork.local.template.md|cowork.local.template]] | Shared template for seeding `local_agents/cowork.md` | Fresh Cowork machine / missing local overlay |
| [[TODO]] | Authoritative live task list | Before any "what's next" |
| [[MERGE_PROTOCOL]] | Branch-per-person git model + merge-to-main procedure + smart-merge prompt | Session start catch-up, any merge to main, any merge conflict |
| [[ONBOARDING]] | Workspace structure + collaboration model + new-collaborator setup | First session on a new machine / new collaborator |
| [[START_RESEARCH_IDEA]] | Fresh-agent guide for framing and launching new research ideas | Before starting a new branch or asking another agent to frame one |
| [[POLYMARKET_BRAIN]] | Obsidian map of Polymarket strategy clusters | Any Polymarket work |
| [[SKILL_MAP]] | Repeatable agent workflows + when to run them | Setting up / running a hygiene or chronicler pass |
| [[OPERATING_RHYTHMS]] | Daily/weekly/monthly hygiene cadence | Deciding what maintenance is due |
| [[glossary]] | Cross-project term definitions | Any unfamiliar shorthand |
| [[OBSIDIAN_INFRA_ROADMAP]] | The infra plan this map implements | Planning further brain work |

## Where to write things

> This is the **canonical** where-to-write table. [[CODEX]] and [[COWORK]] point here and carry only their role-specific deltas.

| Content type | Path | Required backlink |
|---|---|---|
| Polymarket MM findings | `polymarket/research/notes/market_making/<topic>_findings.md` | [[strat_market_making]] · [[COWORK]] |
| Polymarket OD findings | `polymarket/research/notes/options_delta/<topic>_findings.md` | [[strat_options_delta]] · [[COWORK]] |
| copytrade findings | `polymarket/research/notes/copytrade/<topic>_findings.md` | [[COWORK]] |
| dali lineage findings | `polymarket/research/notes/dali/<topic>_findings.md` | [[COWORK]] |
| Polymarket synthesis / explainers | `polymarket/research/notes/overview/synthesis/<topic>.md` | [[POLYMARKET_BRAIN]] · relevant strategy hub or [[COWORK]] |
| Polymarket foundations / deep research | `polymarket/research/notes/overview/foundations/<topic>.md` | [[POLYMARKET_BRAIN]] |
| Polymarket data quality / methodology | `polymarket/research/notes/overview/data_quality/<topic>.md` | [[POLYMARKET_BRAIN]] |
| Polymarket market maps / screens | `polymarket/research/notes/overview/market_maps/<topic>.md` | [[POLYMARKET_BRAIN]] |
| Polymarket CSV report outputs | `polymarket/research/data/analysis/csv_outputs/<cluster>/<topic>.csv` — explain non-obvious columns in the linked `.md` | the linked findings note |
| Polymarket chart outputs | `polymarket/research/data/analysis/plots/<cluster>/<topic>.<png/svg>`, unless an existing script owns a local plot folder | the embedding note |
| Polymarket data manifests | update [[polymarket_data_manifest]] for durable data-family changes; don't link every shard | — |
| Crypto/live-trading data manifests | update [[docs/CRYPTO_DATA_MANIFEST|crypto data manifest]] for durable cache/pickle/CSV/JSONL family changes | — |
| Crypto strategy / WF findings | `topics/<strategy>/research/` or `docs/` | `docs/STRATEGY_REFERENCE.md` |
| Live trading architecture | `live_trading/CLAUDE.md` (append) | — |
| Cross-thread snapshots | `brain/handoffs/<YYYY-MM-DD>_<topic>.md` | relevant hub |
| Task list updates | `brain/TODO.md` (edit directly; keep "done" pruned) | — |
| Agent personal overlay | `local_agents/<agent>.md` (top-level, local-only) | never link from canonical notes except setup docs |
| Agent scratch (WIP) | `scratch/<agent>/YYYY-MM-DD.md` (top-level, local-only) | own lane only |
| Code / scripts | under the relevant project — **never** in `brain/` | — |

Concurrent-edit safety comes from the branch model: each collaborator edits on their own branch and conflicts are resolved at merge time per [[MERGE_PROTOCOL]]. There are no per-file edit locks.

## Active research branches

| Branch | Folder | Status | Hub |
|---|---|---|---|
| copytrade | `polymarket/research/notes/copytrade/` | active — first real-money smoke pending | [[COWORK]] § copytrade |
| MM — market-making | `polymarket/research/notes/market_making/` | active live-measurement track; old single-venue K2/K-PEG path closed | [[strat_market_making]] |
| OD — options-delta | `polymarket/research/notes/options_delta/` | standalone pricing closed; selected sizing/execution diagnostics feed MM | [[strat_options_delta]] |
| dali / research lineage | `polymarket/research/notes/dali/` | not globally closed; individual branches falsified | [[COWORK]] § dali |
| crypto momentum (live) | `live_trading/dashboards/momentum/` | live (6-asset universe) | `docs/STRATEGY_REFERENCE.md` |

> Closed/falsified branch anchors live in [[POLYMARKET_BRAIN]] § Falsification And Redesign Anchors. The single high-level map of the whole arc is [[2026-06-04_state_of_the_arc_and_novelty_frontier]].

## Generated reports + data manifests

Generated reports live in `brain/generated/` (git-ignored, regenerable). Refresh with `tools/brain_hygiene.py` — see [[OPERATING_RHYTHMS]].

| Report | Source | Cadence |
|---|---|---|
| `brain/generated/GENERATED_INDEX.md` | `tools/brain_hygiene.py` | on demand / weekly |
| `brain/generated/hygiene_report.md` | `tools/brain_hygiene.py` | weekly |
| `brain/generated/stale_notes.md` | `tools/brain_hygiene.py` | weekly |

> Generated reports are **not committed** (they live in the ignored `brain/generated/`). They are one command away — `python tools/brain_hygiene.py` — and the durable, agnostic map is this file ([[VAULT_MAP]]), which any human or agent reads first.

Data manifests (read before scanning raw data folders):

- [[polymarket_data_manifest]] — Parquet/CSV/JSONL/DuckDB/raw Polymarket data
- [[polymarket_csv_output_audit]] — result-table layout
- [[polymarket_plot_gallery_index]] — generated figures
- [[polymarket_table_dictionary]] — compact column / bucket definitions
- [[docs/CRYPTO_DATA_MANIFEST|crypto data manifest]] — crypto/live-trading artifacts
- [[storage_consolidation_audit_2026_06_05]] — disk-pressure / storage-layout decisions

## Agent lanes

Shared knowledge brain, shared role conventions, **separate per-person overlays and scratch**. Agents never use a canonical hub as a scratchpad.

| Role | Shared role convention | Local private overlay | Scratch |
|---|---|---|---|
| Implementation — Codex / Claude Code | [[codex_lane]] (`brain/agents/codex/codex_lane.md`) | `local_agents/codex.md` seeded from [[codex.local.template.md|codex.local.template]] | `scratch/codex/YYYY-MM-DD.md` |
| Orchestration — Cowork | [[cowork_lane]] (`brain/agents/cowork/cowork_lane.md`) | `local_agents/cowork.md` seeded from [[cowork.local.template.md|cowork.local.template]] | `scratch/cowork/YYYY-MM-DD.md` |

Collaboration model: each person works on a **personal git branch named after their GitHub handle**; `main` is the shared integration branch and is only written by deliberate, agent-assisted merges — see [[MERGE_PROTOCOL]] and [[ONBOARDING]] § Collaboration model. Top-level `local_agents/` and `scratch/` are git-ignored, so each person's agent overlays and WIP never appear in any branch or merge. (The former live-sync + edit-lock mechanism was retired 2026-06-10 — see [[2026-06-10_relay_retirement_branch_model]].)

## Deeper inspection

```bash
# Refresh navigation index + hygiene reports (finds issues; does not auto-fix)
python tools/brain_hygiene.py

# List tracked brain markdown
git ls-files 'brain/**/*.md'
```

For "find prior work on X" / "what do we already know about Y", prefer the local **gbrain MCP tools** (semantic `search`, `traverse_graph`, `get_backlinks`) over reading hubs end-to-end — it indexes this vault locally and turns `[[basename]]` links into a queryable graph. Retrieval only (synthesis stays in-agent); re-import after big vault changes. Setup/teardown + caveats: [[gbrain_retrieval_layer]] (`docs/tooling/gbrain_retrieval_layer.md`).
