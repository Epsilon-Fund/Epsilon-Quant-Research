---
title: "Dali — research index (May 2026)"
created: 2026-05-27
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

# research index — dali

Numbered deep-dives. Promote new docs here as they're written.

- `01-literature-synthesis.md` — academic foundations (microstructure, OFI, prediction markets), Dali current findings (TFI baseline results, open questions), Block A–J roadmap, reading priorities
- `02-factor-construction.md` — factor catalog, trade-the-price formulation, feature engineering stack, model architecture (LightGBM), strategy critical analysis, monetisation-focused roadmap
- `03-block-a-results.md` — *(planned, post 2026-05-28 capture)* live OFI capture results: R², hit-rate-by-magnitude, per-market taker/maker classification

## notes/

Mirrors of authoritative repo notes from `polymarket/research/notes/`. Re-mirror after repo updates.

- `block_a0_runbook.md` — full runbook for Block A0 24h capture (start commands, tmux/laptop/VPS modes, audit steps). **Mirrored 2026-05-27.**
- `block_b_findings.md` — Block B decision matrix (OUTCOME 3: Mixed Results). Resolution sweep, per-market heterogeneity, walk-forward, volume interaction, sports league breakdown, operator filter. **Mirrored 2026-05-27.**
- `historical_sign_convention_audit.md` — empirical verification of `maker_side` semantics across 4 families. Decision: `historical_to_aggressor()` correct, no rerun needed. **Mirrored 2026-05-27.**
- `dali_tfi_baseline_results.md` — TFI baseline results: crypto (contaminated by resolution window), equity-index, AI/product. Raw signal metrics. **Mirrored 2026-05-27.**
- `sign_convention_findings.md` — live sign convention status: 1 `last_trade_price` event, 0 classifiable. Normalization not established. **Mirrored 2026-05-27.**
