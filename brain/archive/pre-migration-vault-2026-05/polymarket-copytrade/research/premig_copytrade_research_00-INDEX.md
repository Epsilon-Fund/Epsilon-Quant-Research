---
title: "Polymarket copytrade — research index (May 2026)"
created: 2026-05-09
archived: 2026-09-09
status: archived — pre-migration vault (May 2026); historical, not current
owner: justin
project: copytrade
para: archive
hubs:
  - pre-migration-vault-2026-05
tags:
  - archive
  - pre-migration-vault
  - copytrade
---

> **ARCHIVED — PRE-MIGRATION VAULT (May 2026).** Historical record, imported 2026-09-09 from the standalone Cowork vault that predates this repo becoming the brain. Index: [[pre-migration-vault-2026-05]].
> **DEPRIORITISED (2026-08-25).** The copy-trading thread is not the active research thread (that is [[strat_market_making]]). It is not archived either — its execution/signing infrastructure is live and shared with the market-making machinery. Pick this thread back up only with Justin.

# research index — polymarket copytrade

Numbered deep-dives. Promote new docs here as they're written. Notes that age out into reference can stay numbered; experimental drafts can use `XX-` prefix until promoted.

- `01-execution.md` — exec layer: live bot architecture, RTDS → watcher → signal → risk → mirror, journal-replay state model, latency budget, sizing/NegRisk callouts
- `02-data.md` — data layer: cohort research stack, Phase 1/2/3 outputs (1.064B fills → 270M closed positions → 2.58M traders), schema, known limitations
- `03-system-design.md` — end-to-end stitch of 01 + 02; division of labour; integration gaps (bankroll, time windows, NegRisk, multi-leader)
- `04-cohort-pools.md` — six pool spec + archetype commentary + exec-readiness filter + candidate shortlist with exec colour. Selection criteria verbatim from `notes/formulas.md` §3.
- `05-evaluation.md` — backtest design + live metrics vs source wallet *(planned, Phase 5)*

## notes/

Free-form notes / authoritative reference docs.

- `formulas.md` — **METRICS_REFERENCE mirror**. Every column with formula, file:line ref, edge cases, trustworthiness rating. Re-mirror after data refresh.
- `research-findings.md` — **RESEARCH_FINDINGS mirror**. Phase 1–4 analytical results, distributions, candidate shortlist with full per-candidate dossiers. Re-mirror when phases 5+ produce new findings.
- `data-readme.md` — **data-side README mirror**. Repo layout, dataset inventory, build steps, conventions. Reference for "what scripts produce what" and "how long does the X build take".
- `validation_report.md` — *(pending pull from repo)* data-side Phase 1 validation findings
- `api_reconciliation_v1.md` — *(pending pull from repo)* Polymarket API cross-check (10 hand-picked traders)
- `profile_domah.md` — *(pending pull from repo)* sample `profile_trader()` output, useful as a template for what due-diligence dossiers look like
- (others as they accrue)
