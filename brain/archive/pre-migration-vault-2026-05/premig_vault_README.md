---
title: "Pre-migration vault — root README (how the standalone vault worked)"
created: 2026-05-09
archived: 2026-09-09
status: archived — pre-migration vault (May 2026); historical, not current
owner: justin
project: infra
para: archive
hubs:
  - pre-migration-vault-2026-05
tags:
  - archive
  - pre-migration-vault
  - infra
---

> **ARCHIVED — PRE-MIGRATION VAULT (May 2026).** Historical record, imported 2026-09-09 from the standalone Cowork vault that predates this repo becoming the brain. Index: [[pre-migration-vault-2026-05]].
> **PARKED (2026-08-25).** Working practice from the standalone-vault era. The live equivalents are [[COWORK]], [[VAULT_MAP]] and [[MERGE_PROTOCOL]] — do not follow this note's process.

# Epsilon

Cowork notebook for Epsilon Fund Group quant research. Code lives in `github.com/Epsilon-Fund/epsilon-quant-research`. This folder is the lab notebook — journal, research notes, todos, decisions. The repo is the lab.

## Active subprojects

- **polymarket-copytrade** — copy trading bot for Polymarket (exec + data work streams). Leader audit framework, per-leader deployable cells, weather FTC TP passive strategy.
- **dali** — short-horizon OFI/microstructure ML for Polymarket. Block A0 24h capture starts 2026-05-28. Block B (TFI deep-dive) and historical sign convention complete.
- **crypto-momentum** — daily momentum on BTC/ETH/SOL/BNB. WF + CPCV infra. Parked while polymarket is in flight.
- **macro-newsletter** — n8n pipeline auto-publishing weekly research digest to `epsilon-fund.github.io`.

## How this folder works

- `journal/YYYY-MM-DD.md` — daily working log, one file per day, sectioned by subproject.
- `<project>/README.md` — north star + current status.
- `<project>/TODO.md` — live checklist.
- `<project>/progress.md` — append-only weekly rollup of what shipped.
- `<project>/research/` — numbered deep-dive reports / explainers.
- `<project>/decisions/` — ADR-style decision logs.
- `_shared/` — glossary, stack inventory, cross-project reference.

## Working principles

- Minimal targeted edits, no surprise refactors.
- Plain-language explanations alongside technical content.
- Commit infra changes to feature branches, not main.
- Crypto: 1-bar forward shift on regime filters; never blanket `dropna()`; `sqrt(periods_per_year)` for annualisation; Calmar uses annualised return.
