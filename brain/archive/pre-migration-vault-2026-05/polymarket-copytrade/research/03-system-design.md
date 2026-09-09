---
title: "Copytrade 03 — end-to-end system design (superseded framing)"
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

# 03 — system design (end-to-end)

*Stitches `01-execution.md` (live bot) and `02-data.md` (cohort research) into a single end-to-end picture. Reads as a one-pager you can hand a new collaborator.*

> **Framework note (2026-05-18).** The data-side framing in this doc reflects the cohort-research paradigm (six pools → `leader_rankings.parquet` → exec). That paradigm has been **superseded** by per-trader copy-execution audits. Empirical finding (n=7 audits): cohort membership does not predict copyability; copyability is per-(trader × market-family × maker-or-taker × hour-of-day). The diagram and contract sections below remain useful as the *integration shape* between research and exec, but the "Phase 4 cohort selection" → "leader_rankings.parquet" pipeline should be read as one possible upstream producer for the interface, not the only one. The per-leader audit (`scripts/domah_copy_audit.py`) is the current upstream. See `inbox/_archive/2026-05/2026-05-16_pm-data.md` for the pivot rationale and `research/notes/methodology-lessons.md` 2026-05-16 entries for the underlying lessons.

---

## End-to-end flow

```
                    ┌─────────────────────────────────────┐
                    │       polymarket/research/          │
                    │       (cohort research stack)       │
                    │                                     │
   bulk seed ──┐    │  Phase 1  data infra (DuckDB +      │
   warproxxx   │    │           Parquet, views.sql)       │
   CSV         ├──► │  Phase 2  closed_positions.parquet  │
                │   │           (270M rows, $0 self-cons) │
   delta ──────┘    │  Phase 3  traders.parquet           │ ◄── current state
   Goldsky          │           (2.58M addresses)         │     (Phase 3 done)
   subgraph         │                                     │
                    │  Phase 4  cohort selection          │ ◄── next
                    │           (six pools + diagnostics) │
                    │                                     │
                    │  Phase 5  walk-forward + CPCV       │ ◄── deferred
                    │           backtesting               │
                    │                                     │
                    │  output:  leader_rankings.parquet ──┼──┐
                    │           (atomic write, periodic)  │  │
                    └─────────────────────────────────────┘  │
                                                             │
                              ┌──────────────────────────────┘
                              │ file-based interface
                              │ (research writes, exec reads,
                              │  never code-imported)
                              ▼
                    ┌─────────────────────────────────────┐
                    │      polymarket/execution/          │
                    │      (live copy-trading bot)        │
                    │                                     │
   Polymarket ──┐   │  watcher  RTDS WebSocket → fill_q   │
   RTDS         ├──►│  signal   classify ENTRY/EXIT,      │
   firehose     │   │           target_size_shares        │
                │   │  risk     7 breakers, fail-fast     │ ◄── current state
                │   │  mirror   build CandidateOrder,     │     (eng complete,
                │   │           submit, poll for fills    │      fake-venue
                │   │  journal  append-only JSONL,        │      verified, never
                │   │           source of truth           │      run real money)
                │   │                                     │
                │   │  output:  Polymarket CLOB orders ──┐│
                │   │           (via py-clob-client)      │
                │   └─────────────────────────────────────┘
                │                                          │
                └──────────────────────────────────────────┘
                  same firehose: bot also watches its own
                  orders' fills via polling thread
```

---

## Tracks at a glance

| Track | Status | Live-money? | Blocker |
|---|---|---|---|
| **research / data** | Phase 3 done, Phase 4 starting | n/a | spec the six cohort pools before materialising |
| **execution** | engineering complete, fake-venue verified | no — but wired | three operational tasks (PLAN.md sync + snapshot, creds, smoke) |
| **integration** | not begun | no | exec is hardcoded to one address; `leader_rankings.parquet` schema is now spec'd in `decisions/0001-leader-rankings-schema.md` |

---

## Division of labour (the contract)

Codified informally in the exec dump and worth pinning here:

- **Research decides who, how big, why.** Cohort selection, bankroll estimation, edge ranking, pricing-mode recommendation — all research's call. Outputs flow into `leader_rankings.parquet`.
- **Execution decides how to place the order.** Given a sizing decision and a leader address, exec handles RTDS subscription, classification, risk gates, order submission, fill reconciliation. No PnL maths in exec beyond what risk needs.
- **Sizing math sits across the line:**
  ```
  leader_fraction = leader_trade_usd / research.estimated_bankroll_usd     ← research input
  my_bet_usd      = leader_fraction × my_strategy_capital                  ← exec config
  my_bet_usd      = min(my_bet_usd, max_per_trade_cap, available_balance)  ← exec risk
  ```

---

## Known integration gaps (will need resolving before multi-leader live)

### 1. Bankroll mismatch — point-in-time vs lifetime peak

- **Exec wants**: `estimated_bankroll_usd` for proportional sizing math, *as of the time the bot is acting*.
- **Data has**: `est_bankroll_usd_30d_max_approx` — lifetime peak deployed. Useful for "this trader operates at $X scale" descriptive ranking; *not* honest for sizing because it implicitly looks ahead.
- **Resolution**: data Phase 5 plan already calls for point-in-time bankroll computation. Integration is gated on this.

### 2. Time-window mismatch — 30d-rolling vs lifetime

Exec interface fields are mostly `*_30d` (winrate, PnL, maker-taker ratio, position count, hold duration). Data side currently ships lifetime metrics on `traders.parquet`. Either:
- Research adds 30d-rolling versions of style + PnL metrics (probably best — matches honest backtesting), or
- Exec accepts lifetime as proxy and accepts the staleness risk (fine for the PoC, weak for live).

### 3. NegRisk — gap on both sides

- **Data**: `phantom_position_score` flags affected traders; position-level PnL conflated; market-level metrics robust. Merge/split events not indexed (v2 work).
- **Exec**: position keying is `(condition_id, asset_id)` assuming binary YES/NO; multi-outcome rebalances may look like independent trades. Action: avoid mirroring NegRisk-heavy leaders for now; flag in PLAN.md.
- **Composite rule of thumb**: until both sides handle NegRisk, prefer leaders with low `negrisk_volume_share` and use *market-level* PnL metrics for ranking.

### 4. Multi-leader orchestration

Today's bot watches a single `POLYMARKET_LEADER_ADDRESS`. Cohort copy-trading needs:
- Watcher subscribes to N proxy addresses (RTDS supports — currently filtered client-side).
- Per-leader weight from `rank_score` or `rank_position`.
- Per-leader bankroll → per-leader fraction → aggregate target size, capped per market and per total deployed.
- Risk semantics under cohort: kill-switch on the cohort or per-leader? Daily-loss attribution?

Not in scope until the PoC is operationally complete and stable.

---

## Resilience / state model

The execution bot has no database. Position state, dedup, in-flight orders are all rebuilt by replaying the journal (today + yesterday) at startup. This is deliberate and matches the "the journal is the database" pattern.

For research, state lives in versioned Parquet shards (append-only, schema-uniform). DuckDB views provide the analytical surface; no Postgres.

Both sides converge on **file-based, atomically-written, schema-uniform** persistence. The integration interface (`leader_rankings.parquet`) is the same pattern: write to `.tmp`, rename atomically, exec reads on a refresh cycle (e.g. every 4 hours), refuses to act on stale data.

---

## Reference reading

- `01-execution.md` — exec layer detail
- `02-data.md` — data layer detail
- `decisions/0001-leader-rankings-schema.md` — interface contract ADR
- [Tatv — Polymarket Copy-Trading Field Manual](https://tatv.ai/article/the-polymarket-copy-trading-field-manual)
- [Tatv — From Prediction to Allocation: a Cohort Copy-Trading System for Polymarket](https://tatv.ai/article/from-prediction-to-allocation-a-cohort-copy-trading-system-for-polymarket)
