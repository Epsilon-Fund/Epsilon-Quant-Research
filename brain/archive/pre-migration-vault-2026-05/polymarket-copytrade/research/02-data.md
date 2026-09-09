---
title: "Copytrade 02 — data layer state dump (2026-05-09)"
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

# 02 — data layer (cohort research infrastructure)

*Source: state dump from `copytrade-data` chat, 2026-05-09. Repo: `Epsilon-Quant-Research/polymarket/research/`. Sibling: `polymarket/execution/` — see `01-execution.md`.*

---

## TL;DR

Cohort-based copy-trading research infra, Tatv-inspired ("cohorts > individuals"). **Phase 3 complete**: 2.58M-trader parquet with PnL, style, and bankroll metrics, built from 1.064B raw fills (warproxxx CSV seed Dec 2022 → Oct 2025 + Goldsky GraphQL pull Oct 2025 → Apr 23 2026). Self-consistency holds (sum of realised PnL across 270M closed positions = $0.00). Ready for **Phase 4 — stratified cohort selection**. Interface to execution is a future `leader_rankings.parquet`; schema TBD, not yet integrated.

Two parallel tracks: research (this) and execution (separate). Execution is currently running its own POC against a hardcoded trader; research is nowhere near integration.

---

## What this project is

Build cohort-based copy-trading research infrastructure for Polymarket. Goal: identify groups of skilled traders to follow, validate their edge historically, eventually feed a ranked list to the execution module.

---

## What exists in the repo

### Data infrastructure (Phase 1, done)

- `data_infra/goldsky.py` — GraphQL client for Polymarket subgraph
- `data_infra/gamma.py` — Polymarket markets metadata API client
- `data_infra/views.py` — DuckDB connection + SQL view loader
- `data_infra/operator_denylist.py` — 12 deny-listed addresses (relayers, MM bots, HFT)
- `sql/views.sql` — canonical view definitions: `raw_trades`, `trader_actions`, `trader_actions_orphan`, `traders_raw`, `traders_filtered`

### Datasets (gitignored, ~30 GB total)

- `data/trades/*.parquet` — 1.064B raw fills, 13-column schema, validated
  - Bulk seed from warproxxx CSV (Dec 2022 → Oct 2025), ~151M rows
  - Goldsky GraphQL pull (Oct 2025 → Apr 23 2026), ~913M rows
  - Schema-uniform across seed and delta shards
- `data/markets/markets_2026-05-06.parquet` — Polymarket markets metadata from Gamma API (`clob_token_ids`, `outcome_prices`, `condition_id`, `neg_risk`, etc.)
- `data/closed_positions.parquet` — Phase 2 output, 270M rows
  - One row per `(address, market_id, outcome_index)` for closed markets
  - Realised PnL synthesised from trades + market resolution
  - Self-consistency check: total realised_pnl across all 270M rows = $0.00
- `data/traders.parquet` — Phase 3 output, 2.58M rows (one per address)
  - Activity, position-level PnL metrics, market-level PnL metrics
  - Style profile (maker:taker, hold duration, sub-second %, role balance)
  - Phantom position score (NegRisk arb detection)
  - `is_operator_like` flag
  - `est_bankroll_usd_30d_max_approx` (lifetime peak deployed, not point-in-time)

---

## Key data facts

- 2.58M unique addresses across all activity
- 808k markets touched, 96% closed/resolved
- 78% regular markets, 22% NegRisk multi-outcome
- Operator addresses (12 deny-listed) account for 38% of total fills
- Median trader has 16 closed positions; p99 = 1,644
- ~5,000 traders have 100+ closed positions; ~1,500 have 500+
- Self-consistency: sum of all realised PnL across all closed positions = $0 (winners' payouts = losers' losses, modulo open positions and small fee leakage)

---

## Known limitations / documented gaps

1. **Merge/split blindspot on NegRisk markets.** Trades data doesn't capture on-chain split/merge events. NegRisk-active traders' position-level PnL is conflated (winners look bigger, losers look bigger, total nets correctly). `phantom_position_score` flags affected traders. Market-level metrics are robust to this; position-level are not.
2. **Open positions excluded from PnL.** No mark-to-market layer in v1. Only resolved markets contribute to realised PnL.
3. **Sharpe annualisation is naive.** `sqrt(N/years_active)` scaling produces numerical artifacts at the tail (p99 > 10, max = 1.66×10¹⁵). Phase 4 cohort filters must include guards: `n_closed_positions ≥ 200`, `active_days ≥ 90`, `pos_std_pnl > $1`. Treat Sharpe as diagnostic, not primary ranker. Profit factor and dollar-weighted win rate are more honest.
4. **Bankroll is lifetime peak, not point-in-time.** Useful for descriptive "this trader operates at $X scale" but **not** for historical position sizing. Phase 5 backtesting needs a separate point-in-time bankroll computation.
5. **External reconciliation is order-of-magnitude only.** Polymarket's UI shows lifetime PnL; the public API only exposes current portfolio value and `realisedPnL` on currently-open positions. Our numbers are internally consistent but cannot be fully validated against UI without scraping.
6. **Data tail lag ~16 days.** Last sync ended Apr 23 2026. Refresh deferred until before Phase 5.
7. **`peak_position_size` is a proxy.** True running cumulative max OOM'd; substituted with `peak_fill_abs_token` (largest single fill). Strict lower bound on actual peak. Good enough for v1.

---

## Schema reference

### `traders.parquet` (one row per address)

**Activity**
- `address`, `n_closed_positions`, `n_distinct_markets`, `n_fills_total`
- `total_volume_usd`, `first_activity_ts`, `last_activity_ts`, `active_days`

**Position-level PnL** *(suspect on NegRisk-heavy traders)*
- `pos_total_pnl`, `pos_winners`, `pos_losers`, `pos_win_rate`, `pos_dollar_win_rate`
- `pos_avg_win_usd`, `pos_avg_loss_usd`, `pos_profit_factor`
- `pos_sharpe`, `pos_sortino`, `pos_kelly_fraction`, `pos_max_drawdown_usd`

**Market-level PnL** *(NegRisk-robust, prefer for ranking)*
- `mkt_total_pnl`, `mkt_n_markets_traded`, `mkt_winners`, `mkt_losers`, `mkt_win_rate`
- `mkt_dollar_win_rate`, `mkt_avg_win_usd`, `mkt_avg_loss_usd`, `mkt_profit_factor`
- `mkt_sharpe`, `mkt_sortino`, `mkt_kelly_fraction`, `mkt_max_drawdown_usd`

**Style profile**
- `style_maker_fill_count`, `style_taker_fill_count`, `style_maker_taker_ratio`
- `style_role_balance` (0=pure taker, 1=pure maker)
- `style_avg_fill_size_usd`, `style_max_fill_size_usd`
- `style_buy_sell_symmetry`, `style_pct_sub_second`
- `style_avg_holding_hours`, `style_median_holding_hours`

**Flags & misc**
- `phantom_position_score` (>>1 = NegRisk arb), `negrisk_volume_share`
- `is_operator_like` (boolean)
- `est_bankroll_usd_30d_max_approx` (lifetime peak, descriptive only)

### `closed_positions.parquet` (one row per address × market × outcome)

- `address`, `market_id`, `outcome_index`, `neg_risk`
- `realised_pnl`, `realised_cash_flow`, `redemption_value`, `final_token_position`
- `total_bought_usd`, `total_sold_usd`, `gross_token_volume`, `gross_usd_volume`
- `n_fills`, `first_fill_ts`, `last_fill_ts`, `resolution_ts`
- `holding_duration_hours`, `peak_fill_abs_token`, `is_held_to_resolution`

---

## What's next

### Phase 4 — Exploration & cohort selection (immediate)

Materialise six stratified cohort pools as parquet:
- High Sharpe directional (with annualisation guards)
- High profit factor with size
- NegRisk specialists
- Sports/event directional fast
- Patient accumulators
- High Kelly edge

Plus cross-pool diagnostics (overlap, sensitivity, correlation matrices), plus a per-trader `profile_trader()` function for due diligence on candidates.

### Phase 5 — Rigorous backtesting (deferred)

- Walk-forward: pick date, compute cohort from pre-date data, simulate following them post-date.
- CPCV (combinatorial purged cross-validation) for robustness.
- Refresh data first (full Goldsky pull from Apr 23 2026 to current).
- Recompute bankroll point-in-time (not lifetime peak).

### Future v2 work (deferred)

- Index merge/split events from Polygon for true NegRisk PnL accuracy.
- Mark-to-market open positions (needs CLOB price history).
- True running peak position size.
- Refresh markets parquet pinned to latest snapshot per analysis.

---

## Workflow conventions

- `uv` for packaging; DuckDB over Parquet glob; no Postgres.
- All metrics lookahead-free (filter by timestamp before aggregating).
- Append-only Parquet shards, schema-uniform across shards.
- Lowercase 0x-prefixed addresses; `(transaction_hash, log_index)` as unique fill identifier.
- Gitignored: `data/`, `*.parquet`, `.env`.
- Reference: `warproxxx/poly_data` (GPL-3.0, patterns only — don't copy code).

---

## Reference reading

- [Tatv — Polymarket Copy-Trading Field Manual](https://tatv.ai/article/the-polymarket-copy-trading-field-manual)
- [Tatv — From Prediction to Allocation: a Cohort Copy-Trading System for Polymarket](https://tatv.ai/article/from-prediction-to-allocation-a-cohort-copy-trading-system-for-polymarket)
- `polymarket/research/notes/validation_report.md`
- `polymarket/research/notes/api_reconciliation_v1.md`

---

## Cross-refs / open items

- **Interface contract**: `leader_rankings.parquet` schema is TBD. Define after `01-execution.md` is seeded so the contract reflects what exec actually consumes.
- **Phase 4 spec**: each of the six cohort pools deserves its own selection criteria + sanity-check plan. Promote into a dedicated note (e.g. `04-cohort-pools.md`) before materialising.
- **Phase 5 prep**: walk-forward + CPCV will reuse patterns from crypto-momentum's `wf_engine.py`. Worth pulling that into the planning when we get there.
- **Notes pull**: pull `validation_report.md` + `api_reconciliation_v1.md` into `polymarket-copytrade/research/notes/` if you want them living alongside the rest.
