---
title: "Pre-migration vault (May 2026) — archive index"
created: 2026-09-09
status: archive — historical only, nothing here is current
owner: justin
project: infra
para: archive
hubs:
  - VAULT_MAP
  - COWORK
tags:
  - archive
  - pre-migration-vault
---

# Pre-migration vault (May 2026) — archive index

> **ARCHIVED — HISTORICAL ONLY.** Everything in this folder is a snapshot of how the work looked in **May 2026**, before this repo became the brain. **Nothing here is current.** Do not build on any note in this folder, do not quote its numbers, and do not treat its checklists as live task lists. Every framework described here has since been superseded — several of them explicitly falsified.

Hub links: [[VAULT_MAP]] | [[COWORK]] | [[TODO]] | [[MERGE_PROTOCOL]]

## What this is

Before this repo carried the brain, Justin kept a separate Obsidian vault at `~/Documents/Claude/Projects/Epsilon` — a "lab notebook" alongside the "lab". It held a daily journal, a per-project README/TODO/progress trio, ADR-style decision logs, and an inbox where each Claude chat dropped an end-of-day summary for a nightly ingest task to route.

That vault was migrated into this repo through mid-2026. Most of its research content came across under repo-native names and is **already live** in `polymarket/research/notes/` — see § Already migrated below. What remained was the scaffolding and the May-era status prose, which had no repo home. It was imported here on **2026-09-09** so that no Epsilon material lives only in a chat or a folder outside the repo.

## Why it is worth keeping

Not for its conclusions — for its trail. The May notes record *what was believed at the time* and, in several places, the moment a belief broke:

- The **cohort-membership thesis** ("cohorts > individuals") is stated confidently in `01-execution` / `02-data` / ADR 0001, then contradicted in place by the 2026-05-16 audit pivot (`inbox/_archive/2026-05/2026-05-16_pm-data.md`) — n=7 per-leader audits, Spearman ρ = −0.21 between lifetime PnL and copyability. Both halves are here.
- The **"crypto-momentum is parked" error** and its same-day correction are both preserved, in `premig_crypto-momentum_progress.md` and `premig_dashboard-status.md`.
- Two **methodology-lessons ledgers** were written explicitly to outlive the strategies that produced them. They are archived with everything else, but they are the most reusable thing in this folder.

## Status of each line of work, today

| Thread | Status now | Live surface |
|---|---|---|
| market-making | **the only ACTIVE research thread** | [[strat_market_making]] → [[mm_model]] |
| copytrade | DEPRIORITISED — execution/signing infra still live and shared with MM | [[COWORK]] § copytrade cluster |
| crypto momentum | book is live; the May snapshots here are not its state | `docs/STRATEGY_REFERENCE.md`, [[TODO]] § Crypto |
| dali | PARKED lineage | `polymarket/research/notes/dali/` |
| macro newsletter | operational, low priority | this folder only |

## Contents


### Vault root

- [[premig_vault_README]] — Pre-migration vault — root README (how the standalone vault worked)


### `_shared/` — cross-project reference

- [[stack]] — Stack inventory (May 2026)


### `journal/` — daily working log

- [[2026-05-09]] — Daily journal — 2026-05-09
- [[2026-05-10]] — Daily journal — 2026-05-10
- [[2026-05-11]] — Daily journal — 2026-05-11
- [[2026-05-15]] — Daily journal — 2026-05-15
- [[2026-05-16]] — Daily journal — 2026-05-16
- [[2026-05-18]] — Daily journal — 2026-05-18


### `inbox/` — EOD chat summaries and the nightly ingest

- [[2026-05-11_cc-crypto]] — Inbox EOD — 2026-05-11 cc-crypto
- [[2026-05-11_pm-data]] — Inbox EOD — 2026-05-11 pm-data
- [[2026-05-15_pm-data]] — Inbox EOD — 2026-05-15 pm-data
- [[2026-05-16_pm-data]] — Inbox EOD — 2026-05-16 pm-data (the audit-framework pivot)
- [[premig_inbox_README]] — Inbox — EOD ingest conventions (May 2026)


### `polymarket-copytrade/` — the copy-trading thread as it stood in May

- [[0001-leader-rankings-schema]] — ADR 0001 — leader_rankings.parquet schema (superseded framework)
- [[0002-cowork-vs-claude-code-routing]] — ADR 0002 — Cowork vs Claude Code routing for Phase 5
- [[premig_copytrade_decisions_00-INDEX]] — Polymarket copytrade — decisions index (May 2026)
- [[premig_copytrade_README]] — Polymarket copytrade — north star + status (May 2026)
- [[premig_copytrade_TODO]] — Polymarket copytrade — todos (May 2026)
- [[premig_copytrade_progress]] — Polymarket copytrade — progress log (May 2026)
- [[01-execution]] — Copytrade 01 — execution layer state dump (2026-05-09)
- [[02-data]] — Copytrade 02 — data layer state dump (2026-05-09)
- [[03-system-design]] — Copytrade 03 — end-to-end system design (superseded framing)
- [[premig_copytrade_methodology-lessons]] — Polymarket copytrade — methodology lessons ledger (May 2026)
- [[premig_copytrade_research_00-INDEX]] — Polymarket copytrade — research index (May 2026)


### `crypto-momentum/` — the crypto momentum book as it stood in May

- [[premig_crypto-momentum_decisions_00-INDEX]] — Crypto momentum — decisions index (May 2026)
- [[premig_crypto-momentum_README]] — Crypto momentum — north star + status (May 2026)
- [[premig_crypto-momentum_TODO]] — Crypto momentum — todos (May 2026)
- [[premig_crypto-momentum_progress]] — Crypto momentum — progress log (May 2026)
- [[premig_crypto-momentum_methodology-lessons]] — Crypto momentum — methodology lessons ledger (May 2026)
- [[premig_dashboard-status]] — Live trading dashboard status snapshot (2026-05-11, ~80% accurate)
- [[premig_xs-momentum-recap]] — XS momentum thread recap (May 2026, likely scrapped)
- [[premig_crypto-momentum_research_00-INDEX]] — Crypto momentum — research index (May 2026)


### `dali/` — the OFI/microstructure lineage as it stood in May

- [[premig_dali_README]] — Dali — north star + status (May 2026)
- [[premig_dali_TODO]] — Dali — roadmap checklist (May 2026)
- [[premig_dali_research_00-INDEX]] — Dali — research index (May 2026)


### `macro-newsletter/` — the n8n newsletter pipeline

- [[premig_macro-newsletter_README]] — Macro newsletter — n8n pipeline (May 2026)


## Already migrated — deliberately **not** re-imported

These vault files came across under repo-native names before this archive was made. The repo copy is authoritative; the vault copy is a stale duplicate.

| Vault path | Live in the repo as |
|---|---|
| `_shared/glossary.md` | [[glossary]] |
| `dali/research/01-literature-synthesis.md` | [[dali_literature_synthesis]] |
| `dali/research/02-factor-construction.md` | [[dali_factor_construction]] |
| `dali/research/notes/block_a0_runbook.md` | [[block_a0_runbook]] |
| `dali/research/notes/block_b_findings.md` | [[block_b_findings]] |
| `dali/research/notes/dali_tfi_baseline_results.md` | [[dali_tfi_baseline_results]] |
| `dali/research/notes/historical_sign_convention_audit.md` | [[historical_sign_convention_audit]] |
| `dali/research/notes/sign_convention_findings.md` | [[sign_convention_findings]] |
| `polymarket-copytrade/research/notes/data-readme.md` | `polymarket/research/README.md` |
| `polymarket-copytrade/research/notes/formulas.md` | `polymarket/research/docs/METRICS_REFERENCE.md` |
| `polymarket-copytrade/research/notes/research-findings.md` | `polymarket/research/RESEARCH_FINDINGS.md` |
| `research_v1_audit/*` | `polymarket/research/docs/audit_2026-09/` (md5-identical) |

Nine `journal/2026-05-*.md` files and the two vault-root `2026-05-*.md` files were **dropped**: they are empty section headers with zero content.

## Filenames

Directory structure is preserved. Basenames that would have collided with a live repo note (`README.md`, `TODO.md`, `progress.md`, `00-INDEX.md`, `methodology-lessons.md`) carry a `premig_` prefix, because Obsidian resolves `[[wikilinks]]` by basename and basenames must stay unique across this vault.
