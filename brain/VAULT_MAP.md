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

Hub links: [[CODEX]] | [[COWORK]] | [[TODO]] | [[ONBOARDING]] | [[START_RESEARCH_IDEA]] | [[POLYMARKET_BRAIN]] | [[glossary]] | [[SKILL_MAP]] | [[OPERATING_RHYTHMS]] | [[OBSIDIAN_INFRA_ROADMAP]] | [[COWORK_MIGRATION]]

## What this repo is

A two-project quantitative research monorepo. The projects **share no code** — separate venvs, separate dependencies, never cross-import.

| Project | Root | What it does |
|---|---|---|
| **Polymarket alpha** | `polymarket/research/` + `polymarket/execution/` (Midas) | Prediction-market trading: copytrade, market-making (MM), options-delta (OD), and the dali research lineage |
| **Crypto live trading** | `live_trading/` + `topics/` + `infrastructure/` | Momentum / stat-arb / BB-breakout on Binance; walk-forward + CPCV research engine |

## Agent Bootstrap (do this before anything else)

1. Determine your role: Codex -> `local_agents/codex.md`; Cowork/Claude Code -> `local_agents/cowork.md`.
2. If that file does not exist, create `local_agents/` and copy the matching template from `brain/agents/templates/<role>.local.template.md` into it, then tell the user "seeded your local <role> overlay - edit it to set your personal preferences."
3. Read your `local_agents/<role>.md` overlay (personal style), then the shared law `brain/CODEX.md` or `brain/COWORK.md`, then `brain/VAULT_MAP.md`, then `brain/TODO.md`.

Precedence: personal overlay = voice/preferences; shared `CODEX`/`COWORK` + repo invariants = law (always win).

## Start here (reading order)

After bootstrap:

1. **This file** — orientation + where things live.
2. Your shared role convention: [[codex_lane]] (Codex) or [[cowork_lane]] (Cowork/Claude Code).
3. For a new idea: [[START_RESEARCH_IDEA]] — fresh-agent workflow, idea-card shape, and where first durable notes go.
4. For Polymarket work: [[POLYMARKET_BRAIN]] → the relevant strategy hub.
5. For data-heavy work: the relevant manifest (see Generated Reports + Data Manifests below) before scanning raw folders.

## Top-level folder map

| Path | Owns |
|---|---|
| `brain/` | Git-tracked context hub: maps, hubs, task list, agent lanes, handoffs (this folder) |
| `local_agents/` | Local-only per-person agent instruction overlays; git-ignored and not Relay-shared |
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
| [[CODEX]] | Implementation-agent README: invariants, run env, where to write | Every Codex session start |
| [[COWORK]] | Strategic-agent README: active clusters, prompt discipline | Every Cowork session start |
| [[codex.local.template.md|codex.local.template]] | Shared template for seeding `local_agents/codex.md` | Fresh Codex machine / missing local overlay |
| [[cowork.local.template.md|cowork.local.template]] | Shared template for seeding `local_agents/cowork.md` | Fresh Cowork/Claude Code machine / missing local overlay |
| [[TODO]] | Authoritative live task list | Before any "what's next" |
| [[START_RESEARCH_IDEA]] | Fresh-agent guide for framing and launching new research ideas | Before starting a new branch or asking another agent to frame one |
| [[POLYMARKET_BRAIN]] | Obsidian map of Polymarket strategy clusters | Any Polymarket work |
| [[SKILL_MAP]] | Repeatable agent workflows + when to run them | Setting up / running a hygiene or chronicler pass |
| [[OPERATING_RHYTHMS]] | Daily/weekly/monthly hygiene cadence | Deciding what maintenance is due |
| [[glossary]] | Cross-project term definitions | Any unfamiliar shorthand |
| [[OBSIDIAN_INFRA_ROADMAP]] | The infra plan this map implements | Planning further brain work |

## Where to write things

| Content type | Path | Required backlink |
|---|---|---|
| Polymarket MM findings | `polymarket/research/notes/market_making/<topic>_findings.md` | [[strat_market_making]] · [[COWORK]] |
| Polymarket OD findings | `polymarket/research/notes/options_delta/<topic>_findings.md` | [[strat_options_delta]] · [[COWORK]] |
| copytrade findings | `polymarket/research/notes/copytrade/<topic>_findings.md` | [[COWORK]] |
| dali lineage findings | `polymarket/research/notes/dali/<topic>_findings.md` | [[COWORK]] |
| Polymarket synthesis / explainers | `polymarket/research/notes/overview/synthesis/<topic>.md` | [[POLYMARKET_BRAIN]] · relevant strategy hub or [[COWORK]] |
| Crypto strategy / WF findings | `topics/<strategy>/research/` or `docs/` | `docs/STRATEGY_REFERENCE.md` |
| Live trading architecture | `live_trading/CLAUDE.md` (append) | — |
| Cross-thread snapshots | `brain/handoffs/<YYYY-MM-DD>_<topic>.md` | relevant hub |
| Task list updates | `brain/TODO.md` (edit directly; keep "done" pruned) | — |
| Agent personal overlay | `local_agents/<agent>.md` (top-level, local-only) | never link from canonical notes except setup docs |
| Agent scratch (WIP) | `scratch/<agent>/YYYY-MM-DD.md` (top-level, local-only) | own lane only |
| Code / scripts | under the relevant project — **never** in `brain/` | — |

Before editing any shared durable Markdown file, acquire a cooperative edit lock:

```bash
python3 tools/brain_edit_guard.py acquire --agent <codex|cowork|justin> --path <path.md> --intent "<short reason>"
python3 tools/brain_edit_guard.py release --agent <codex|cowork|justin> --path <path.md>
```

If the lock is held by another agent, stop and report instead of editing around it.

Cross-lane edits are not banned, but they require explicit instruction. Use `--allow-cross-lane` only when Justin asked one agent to edit the other agent's lane.

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
| Codex | [[codex_lane]] (`brain/agents/codex/codex_lane.md`) | `local_agents/codex.md` seeded from [[codex.local.template.md|codex.local.template]] | `scratch/codex/YYYY-MM-DD.md` |
| Cowork / Claude Code | [[cowork_lane]] (`brain/agents/cowork/cowork_lane.md`) | `local_agents/cowork.md` seeded from [[cowork.local.template.md|cowork.local.template]] | `scratch/cowork/YYYY-MM-DD.md` |

Temporary cooperative lock files live under `brain/agents/locks/`; see [[agent_edit_locks]]. They are managed by `tools/brain_edit_guard.py` and sync live via Relay so each agent sees the other's active claim.

Relay scope: it shares the research folders (`Attachments/`, `archive/`, `docs/`, `infrastructure/`, `live_trading/`, `meetings/`, `midas/`, `newsletters/`, `polymarket/`, `topics/`) and **all of `brain/`** — so [[VAULT_MAP]], [[TODO]], the hubs, handoffs, generated reports, shared lane docs, templates, and edit locks are live for both collaborators. Top-level `local_agents/` and `scratch/` are deliberately kept off Relay and ignored by Git, so each person has private agent overlays and WIP. See [[ONBOARDING]] § Sync model.

## Deeper inspection

```bash
# Refresh navigation index + hygiene reports (finds issues; does not auto-fix)
python tools/brain_hygiene.py

# List tracked brain markdown
git ls-files 'brain/**/*.md'
```
