# Polymarket Copy-Trading: Data Infrastructure
## Repo context
This module lives inside the Epsilon-Quant-Research repo, alongside an
existing crypto momentum trading framework. The two projects are independent
— separate pyproject.toml, separate venv. NEVER modify files outside
polymarket/. NEVER import from the quant code (wf_engine, etc). The crypto
and prediction-market work do not share code or data.
## Goal
Build historical trade & trader data infrastructure for cohort-based copy
trading on Polymarket. This phase is offline-only — no live trading, no
WebSockets, no execution. Just historical data + trader ranking.

## Data Sources
- Bulk historical seed: warproxxx/poly_data orderFilled_complete.csv.xz
  from https://polydata-archive.s3.us-east-1.amazonaws.com/
- Incremental updates: Goldsky subgraph (GraphQL)
- Market metadata: Polymarket Gamma API (https://gamma-api.polymarket.com)
- No direct Polygon RPC needed for v1.

## Architecture
- data_infra/ — clients for Goldsky and Gamma, parquet I/O helpers
- data/trades/ — parquet shards, append-only, never edit in place
- data/markets/ — market metadata snapshots
- data/traders.parquet — computed trader stats output
- DuckDB is the query layer over parquet; we don't put data into Postgres
  or run a database server.

## Conventions
- Minimal, targeted code changes. Don't refactor unrelated things.
- Don't add features that weren't asked for.
- Prefer DuckDB SQL over pandas/polars for aggregations on trades.
- All metrics must be lookahead-free — filter by timestamp before
  aggregating, never assume current-state data.
- Storage: append-only Parquet shards. New data → new shard. Reads union
  shards via DuckDB's glob support.
- Reference repo for patterns (DO NOT copy code, GPL-3.0 license):
  https://github.com/warproxxx/poly_data
  Use it to understand the Goldsky query shape and trade-processing logic,
  then write your own.

## Polymarket-specific facts
- Trades emit OrderFilled events on the CTF Exchange contract (regular and
  NegRisk variants). Both must be indexed.
- asset_id == 0 means USDC. Non-zero asset_ids are outcome tokens.
- Prices are decimals 0.0–1.0 (NOT cents like Kalshi).
- Amounts use 6 decimals (USDC convention).
- Per warproxxx convention: when filtering for a specific user's trades,
  filter by the `maker` column. This is how Polymarket emits events at
  the contract level.

## What this phase does NOT do
- No live trading, no WebSockets, no order placement.
- No FPMM legacy trades (pre-2022 AMM). CTF Exchange only.
- No splits/merges/redemptions yet — trades only is fine for v1.
- No category tagging from Gamma yet — defer until basic ranking works.

## What "trader stats" means in v1
For each address that appears as `maker` in the trades table:
- total trades
- total USD volume
- win rate (per-position, computed only on resolved markets)
- realised PnL
- simple Sharpe (over per-position returns)
- max drawdown
Output as data/traders.parquet.

## Workflow rules
- Never run `pip install` without `--break-system-packages` or an active venv.
- Don't run full backfills in interactive sessions — they take hours.
  Use the warproxxx CSV snapshot as the historical seed.
- Don't add a database server (Postgres, etc.). DuckDB over parquet only.
