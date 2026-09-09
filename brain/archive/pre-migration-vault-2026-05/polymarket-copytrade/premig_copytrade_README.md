---
title: "Polymarket copytrade — north star + status (May 2026)"
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

# polymarket copytrade

Cohort-based copy trading on Polymarket. Identify groups of skilled traders, validate edge historically, push a ranked list to an execution bot that mirrors their positions. Inspired by the Tatv "cohorts > individuals" thesis (refs in `research/02-data.md`).

Repo: `Epsilon-Quant-Research/polymarket/` — two siblings:
- `polymarket/research/` — cohort research stack (where most cowork notes live).
- `polymarket/execution/` — live copy-trading bot (single-leader PoC, engineering complete, fake-venue verified).

## Status

| Track | State | Live-money? | Blocker |
|---|---|---|---|
| **data / research** | Phase 3 done · Phase 4 next | n/a | spec the six cohort pools before materialising |
| **execution** | engineering complete · 214 tests · fake-venue verified | wired but never run | three operational tasks: PLAN.md sync, creds, smoke |
| **integration** | not begun | no | exec hardcoded to one address; `leader_rankings.parquet` schema spec'd in `decisions/0001-leader-rankings-schema.md` |

Detailed state: `research/01-execution.md`, `research/02-data.md`. End-to-end design: `research/03-system-design.md`.

## North star

A live copy-trading bot on Polymarket that mirrors a **ranked cohort** of historically-validated leaders, with point-in-time–correct sizing, walk-forward + CPCV-validated edge, and resilient execution that handles NegRisk markets, partial fills, and resolution events.

## Interface contract (research → execution)

`leader_rankings.parquet`. Atomically written by research, consumed by exec on a refresh cycle. Schema and gaps captured in `decisions/0001-leader-rankings-schema.md`.

Key gaps to close before multi-leader live:
- **Bankroll**: exec wants point-in-time, data ships lifetime peak. Phase 5 work.
- **Time windows**: exec wants 30d-rolling on most metrics, data ships lifetime. Phase 4 prep should add 30d windows.
- **NegRisk**: gap on both sides — data flags via `phantom_position_score`/`negrisk_volume_share`, exec assumes binary YES/NO position keying. Mitigation in v1: prefer leaders with low `negrisk_volume_share`, use market-level PnL for ranking.

## Sizing rule (division of labour)

```
leader_fraction = leader_trade_usd / research.estimated_bankroll_usd     ← research input
my_bet_usd      = leader_fraction × my_strategy_capital                  ← exec config
my_bet_usd      = min(my_bet_usd, max_per_trade_cap, available_balance)  ← exec risk
```

Research decides bankroll (the *who* and *how big*); exec applies fraction × capital with caps (the *how to place*).

## Open questions (still live)

- **Cohort selection spec.** What's the criteria for each of the six pools? Promote to `research/04-cohort-pools.md` before materialising.
- **Refresh cadence.** Daily? Hourly? How often is `leader_rankings.parquet` regenerated, and what's the data-refresh upstream cadence?
- **Multi-leader risk semantics.** Kill-switch on the cohort or per-leader? Daily-loss attribution? Out of scope until single-leader smoke is stable.
- **Geo / VPS.** UK-blocked for order submission. VPS region + provider not yet picked.
- **Success metric.** Tracking error vs cohort, net PnL after fees + slippage, or hit rate on resolved markets?
