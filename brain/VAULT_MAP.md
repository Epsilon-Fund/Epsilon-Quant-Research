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

Hub links: [[CODEX]] | [[COWORK]] | [[TODO]] | [[POLYMARKET_BRAIN]] | [[glossary]] | [[SKILL_MAP]] | [[OPERATING_RHYTHMS]] | [[OBSIDIAN_INFRA_ROADMAP]]

## What this repo is

A two-project quantitative research monorepo. The projects **share no code** — separate venvs, separate dependencies, never cross-import.

| Project | Root | What it does |
|---|---|---|
| **Polymarket alpha** | `polymarket/research/` + `polymarket/execution/` (Midas) | Prediction-market trading: copytrade, market-making (MM), options-delta (OD), and the dali research lineage |
| **Crypto live trading** | `live_trading/` + `topics/` + `infrastructure/` | Momentum / stat-arb / BB-breakout on Binance; walk-forward + CPCV research engine |

## Start here (reading order)

1. **This file** — orientation + where things live.
2. Your agent lane: [[codex_lane]] (Codex) or [[cowork_lane]] (Cowork/Claude Code).
3. [[TODO]] — authoritative live task list. Read before suggesting next actions.
4. For Polymarket work: [[POLYMARKET_BRAIN]] → the relevant strategy hub.
5. For data-heavy work: the relevant manifest (see Generated Reports + Data Manifests below) before scanning raw folders.

## Top-level folder map

| Path | Owns |
|---|---|
| `brain/` | Shared context hub: maps, hubs, task list, agent lanes, handoffs (this folder) |
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
| [[TODO]] | Authoritative live task list | Before any "what's next" |
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
| Agent scratch (WIP) | `brain/agents/<agent>/scratch/YYYY-MM-DD.md` | own lane only |
| Code / scripts | under the relevant project — **never** in `brain/` | — |

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

Shared knowledge brain, **separate agent operating surfaces**. Agents never use a canonical hub as a scratchpad.

| Lane | Operating doc | Scratch |
|---|---|---|
| Codex | [[codex_lane]] (`brain/agents/codex/codex_lane.md`) | `brain/agents/codex/scratch/YYYY-MM-DD.md` |
| Cowork / Claude Code | [[cowork_lane]] (`brain/agents/cowork/cowork_lane.md`) | `brain/agents/cowork/scratch/YYYY-MM-DD.md` |

## Deeper inspection

```bash
# Refresh navigation index + hygiene reports (finds issues; does not auto-fix)
python tools/brain_hygiene.py

# List tracked brain markdown
git ls-files 'brain/**/*.md'
```
