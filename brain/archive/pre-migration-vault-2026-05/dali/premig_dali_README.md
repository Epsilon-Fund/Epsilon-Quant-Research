---
title: "Dali — north star + status (May 2026)"
created: 2026-05-23
archived: 2026-09-09
status: archived — pre-migration vault (May 2026); historical, not current
owner: justin
project: polymarket
para: archive
hubs:
  - pre-migration-vault-2026-05
tags:
  - archive
  - pre-migration-vault
  - dali
---

> **ARCHIVED — PRE-MIGRATION VAULT (May 2026).** Historical record, imported 2026-09-09 from the standalone Cowork vault that predates this repo becoming the brain. Index: [[pre-migration-vault-2026-05]].
> **PARKED (2026-08-25).** This research lineage is closed/parked. Any concept from it that the active market-making project needs is explained inline in [[strat_market_making]] / [[mm_model]] — do not build on this note.

# Dali — Short-Horizon ML for Polymarket

Short-horizon directional and market-making strategies for Polymarket using order flow imbalance (OFI) signals and microstructure-based ML models.

## North star

Determine whether OFI-based signals produce a tradeable edge on Polymarket's CLOB. If taker signal validates → build rule-based baseline, then ML layer. If maker signal validates → Avellaneda-Stoikov framework on neglected markets.

## Current status (2026-05-23)

**Phase 1 — Signal Validation.** Block A (live OFI capture) is the immediate focus. TFI baseline from historical fills produced weak, tail-driven signals across all 4 market families. Sign convention unverified. Live OFI capture is the decisive next test.

Related project: **polymarket-copytrade** (Midas bot) — separate strategy class (copy trading); shares Polymarket infrastructure context.

## How this folder works

- `README.md` — this file; north star + current status
- `TODO.md` — live roadmap checklist (Blocks A–J + research gaps)
- `progress.md` — append-only weekly rollup of what shipped
- `research/` — numbered deep-dive reports
- `decisions/` — ADR-style decision logs

## Key references

- `research/01-literature-synthesis.md` — academic foundations, Dali findings, block structure
- `research/02-factor-construction.md` — factor catalog, trade-the-price formulation, model architecture
- `_shared/glossary.md` — OFI, TFI, CKS, AS-MM, and other shared terms
