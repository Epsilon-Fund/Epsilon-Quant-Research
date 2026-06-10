# Polymarket Research

## What this is

Prediction-market research on Polymarket. The data spine is cohort-based copy-trading: identify groups of skilled traders (not individuals), validate their edge across historical data, and eventually feed a ranked list of cohorts to the [execution module](../execution/README.md). This folder is the offline research half — historical data ingestion, trader ranking, cohort exploration, and the strategy notes of every research branch that grew out of that dataset. No live trading happens here.

Copy-trading strategy framing follows the Tatv articles linked at the bottom of this file: cohort-level allocation rather than single-leader copy.

## Status — copy-trading data pipeline

| phase | description | outcome |
|---|---|---|
| **1. Data infrastructure** | Bulk warproxxx CSV seed (2022-11 → 2025-10) + Goldsky subgraph delta (2025-10 → 2026-04) + Gamma markets snapshot. Validation pass on data quality. | **1 064 500 317 fills** ingested, 2 576 698 distinct addresses, 1 036 708 markets in snapshot. Validation found 0.38 % orphan rate, 905 self-trades (early-protocol noise), 12 operator-shaped addresses to deny-list. See [`notes/overview/data_quality/validation_report.md`](notes/overview/data_quality/validation_report.md). |
| **2. Closed-position reconstruction** | For each `(address, market_id, outcome_index)` on resolved markets: aggregate trades + synthesise redemption at resolution price. | **269 974 929 positions** reconstructed. **Σ realised_pnl = $0.00** across all positions (zero-sum self-consistency invariant). 96.4 % held to resolution. |
| **3. Trader metric panel** | Per-address activity, position-level + market-level PnL stats, style profile, drawdown, bankroll, operator flag. | **2 576 698 traders** with full metric panel; 2 572 665 after operator deny-list filter. Σ pos_total_pnl == Σ mkt_total_pnl confirmed to the cent. |
| **4. Cohort exploration** | Six stratified pools with defensive guards (Sharpe artifact bounds, sample-size minimums). | 9 788 unique addresses qualify for ≥ 1 pool; 947 for 3+ pools; 0 for 5+ (pools are designed to span independent dimensions). 5 manual-inspection candidates surfaced — see [`RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md). |
| **5. Pending — backtesting** | Walk-forward + CPCV on cohort-selection rules. Point-in-time bankroll. Out-of-sample edge validation. | Not started. |
| **execution-side integration** | Separate module at [`polymarket/execution/`](../execution/README.md). Will consume a leader_rankings parquet produced here. | Not started. |

## Beyond copy-trading — the wider research arc

The same fills dataset and capture infrastructure power a much larger body of strategy research, documented branch by branch under [`notes/`](notes/) — including the falsified branches, which stay in the record so nobody rediscovers a dead idea:

<p align="center">
  <img src="../../docs/assets/notes_by_cluster.png" width="640" alt="Documented Polymarket research notes by strategy cluster" />
</p>

- [`notes/market_making/`](notes/market_making/) — passive quoting economics: spread capture, carry-to-resolution, adverse selection, queue position, incumbent-maker concentration, NegRisk accounting, and live-loop designs.
- [`notes/options_delta/`](notes/options_delta/) — binary prices vs external references: realized-vol fair values, longshot/vol overpricing, hedge overlays, settlement alignment, and the realism audits that closed the standalone pricing branches.
- [`notes/dali/`](notes/dali/) — order-flow microstructure lineage: CLOB/L2 capture, OFI/TOB signals, taker/maker execution tests under non-overlap math, and the ML tier that confirmed the structural diagnosis.
- [`notes/copytrade/`](notes/copytrade/) — leader audits, copyability metrics, directionality classification, and smoke-deployment planning.
- [`notes/overview/`](notes/overview/) — cross-branch synthesis, methodology and data-quality notes, market maps, and shared table/term definitions.

The folder-level map is [`notes/INDEX.md`](notes/INDEX.md); shared column/bucket definitions live in [`notes/overview/data_quality/polymarket_table_dictionary.md`](notes/overview/data_quality/polymarket_table_dictionary.md).

## Repo layout

```
polymarket/research/
├── README.md                       — this file
├── RESEARCH_FINDINGS.md            — analytical results across phases 1-4
├── CLAUDE.md                       — project goals + conventions
├── pyproject.toml                  — uv-managed dependencies
├── docs/
│   └── METRICS_REFERENCE.md        — every column, every formula, file:line refs
├── data_infra/
│   ├── duck.py                     — DuckDB connection helper
│   ├── views.py                    — load_views(con) — registers SQL views
│   ├── operator_denylist.py        — 12 hardcoded operator addresses + heuristic
│   ├── trader_profile.py           — profile_trader(addr) — per-address dossier
│   ├── goldsky.py                  — Goldsky GraphQL client
│   └── gamma.py                    — Polymarket Gamma API client
├── sql/
│   └── views.sql                   — raw_trades, joined_fills, trader_actions, traders_filtered
├── scripts/
│   ├── inspect_seed.py             — sanity check on warproxxx CSV before build
│   ├── build_trades_table.py       — seed CSV → trades_seed.parquet
│   ├── sync_trades_delta_parallel.py  — Goldsky → trades_delta_shard*.parquet
│   ├── build_markets_table.py      — Gamma → markets_<date>.parquet
│   ├── check_progress.py           — sync progress monitor
│   ├── smoke_test_views.py         — sanity tests for views.sql
│   ├── build_closed_positions.py   — closed_positions.parquet (Phase 2)
│   ├── build_traders_table.py      — traders.parquet (Phase 3)
│   ├── build_cohorts.py            — data/cohorts/*.parquet (Phase 4)
│   ├── api_reconciliation.py       — Polymarket API cross-check (10 traders)
│   ├── sanity_phase3.py            — pre-Phase-4 sanity stdout
│   ├── cohort_diagnostics.py       — Phase 4 cross-pool diagnostic stdout
│   └── validation/
│       └── 01_..07_*.py            — per-validation-check scripts
├── notebooks/
│   ├── README.md                   — notebook index
│   └── cohort_exploration.ipynb    — Phase 4 interactive diagnostics
├── notes/
│   ├── INDEX.md                    — markdown note-folder map
│   ├── overview/                   — synthesis/foundations/data-quality/market-map subfolders
│   ├── market_making/              — MM / maker notes
│   ├── options_delta/              — OD / digital-option notes
│   ├── copytrade/                  — copytrade notes and profiles
│   └── dali/                       — dali / Polymarket research lineage
└── data/                           — gitignored
    ├── trades/                     — parquet shards, raw fills
    ├── markets/                    — Gamma snapshots
    ├── closed_positions.parquet    — Phase 2 output
    ├── traders.parquet             — Phase 3 output
    ├── cohorts/                    — Phase 4 output (6 parquets)
    └── _traders_build.duckdb       — Phase 3 work file (cached intermediates)
```

## Datasets (gitignored)

| dataset | rows | size | built by | consumed by |
|---|---:|---:|---|---|
| `data/trades/trades_seed.parquet` | 151 053 823 | 5.8 GB | `scripts/build_trades_table.py` (warproxxx CSV seed) | `views.sql` (`raw_trades`) |
| `data/trades/trades_delta_shard*.parquet` | ~913 M (28 shards) | ~33 GB total | `scripts/sync_trades_delta_parallel.py` (Goldsky GraphQL) | `views.sql` (`raw_trades`) |
| `data/markets/markets_2026-05-06.parquet` | 1 036 708 | 137 MB | `scripts/build_markets_table.py` (Gamma API) | `views.sql` (`markets_tokens`) |
| `data/closed_positions.parquet` | 269 974 929 | 27 GB | `scripts/build_closed_positions.py` | `build_traders_table.py`, `build_cohorts.py`, `trader_profile.py` |
| `data/traders.parquet` | 2 576 698 | 555 MB | `scripts/build_traders_table.py` | `build_cohorts.py`, `trader_profile.py`, notebook |
| `data/cohorts/*.parquet` | 113–7 703 each | 48 KB–2.3 MB | `scripts/build_cohorts.py` | notebook, `cohort_diagnostics.py`, `trader_profile.py` |

## Schema reference

Every column in every derived parquet — including verbatim formula, source `file:line`, edge cases, and trustworthiness rating — lives in [`docs/METRICS_REFERENCE.md`](docs/METRICS_REFERENCE.md). Read that document before relying on any column for decisions.

For the family-level map of Parquet shards, findings-support CSVs, JSONL checkpoints, and DuckDB scratch artifacts, see [the Polymarket data manifest](notes/overview/data_quality/polymarket_data_manifest.md).

Quick-ref column groups (full list in METRICS_REFERENCE):

- **`closed_positions.parquet`** (22 cols): identity (`address`, `market_id`, `outcome_index`), activity (`n_fills`, `first_fill_ts`, `resolution_ts`, `holding_duration_hours`), volume (`gross_*_volume`, `total_bought_usd`, `total_sold_usd`), PnL (`final_token_position`, `realised_cash_flow`, `redemption_value`, `realised_pnl`, `resolution_price`), diagnostic (`peak_fill_abs_token`, `is_held_to_resolution`).
- **`traders.parquet`** (50 cols): activity (`n_closed_positions`, `n_distinct_markets`, `total_volume_usd`, `active_days`), position-level PnL (`pos_*` family), market-level PnL (`mkt_*` family — NegRisk-robust), `phantom_position_score`, style profile (`style_*`), drawdown, bankroll, `is_operator_like`.
- **`data/cohorts/*.parquet`**: all `traders.parquet` columns + `pos_std_pnl` + `mkt_std_pnl` (the std columns used as artifact-blowup guards).

## How to reproduce from scratch

Assumes a checkout of the repo and an empty `polymarket/research/data/` directory. Times below were measured on the dev machine (M-series Mac, 32 GB RAM, ~14 GB/s sequential SSD). Network steps depend on Goldsky / Gamma availability.

1. `cd polymarket/research && uv sync` — installs DuckDB, gql, httpx, pyarrow, pandas (Python ≥ 3.14).
2. **Download warproxxx seed.**
   ```
   curl -O https://polydata-archive.s3.us-east-1.amazonaws.com/orderFilled_complete.csv.xz
   xz -d orderFilled_complete.csv.xz   # → data/raw/orderFilled_complete.csv (~7 GB)
   ```
   Sha256: cross-check against the warproxxx archive page; the file is ~5.5 GB compressed, ~33 GB uncompressed.
3. **Build markets snapshot.** `PYTHONPATH=. uv run python scripts/build_markets_table.py` (~5 min). Output: `data/markets/markets_<today>.parquet`.
4. **Build seed parquet.** `PYTHONPATH=. uv run python scripts/build_trades_table.py` (~10 min). Output: `data/trades/trades_seed.parquet`. Joins to the markets snapshot from step 3 to enrich `market_id` / `condition_id` / `neg_risk`.
5. **Backfill Goldsky delta.** `PYTHONPATH=. uv run python scripts/sync_trades_delta_parallel.py` (**~14 hours**, 15 parallel workers, restart-safe via per-shard cursor files in `data/trades/_inprog/`). Output: `data/trades/trades_delta_shard*.parquet`. Monitor via `scripts/check_progress.py`.
6. **Smoke-test views.** `PYTHONPATH=. uv run python scripts/smoke_test_views.py` (~3 min). Confirms row counts, sign-symmetry on a sample, domah reconciliation.
7. **Build closed_positions.** `PYTHONPATH=. uv run python scripts/build_closed_positions.py` (~12 min). Output: `data/closed_positions.parquet` (~27 GB).
8. **Build traders panel.** `PYTHONPATH=. uv run python scripts/build_traders_table.py` (~18 min on first run; ~6 min on reruns thanks to the cached `_traders_build.duckdb`). Output: `data/traders.parquet`.
9. **Build cohort pools.** `PYTHONPATH=. uv run python scripts/build_cohorts.py` (~2 min). Output: `data/cohorts/*.parquet` (6 files).

End state: `traders.parquet` and the 6 cohort pools queryable via DuckDB, `data_infra.views.load_views(con)` registers the standard SQL views, `data_infra.trader_profile.profile_trader(address)` produces a per-trader dossier.

## Key conventions

- **uv** for packaging and run isolation. Never `pip install` directly.
- **DuckDB over Parquet** as the query layer. No Postgres, no other DB server.
- **Lookahead-free metrics**: filter by timestamp before aggregating; never assume current-state data when reconstructing historical points.
- **Append-only Parquet shards.** New data → new shard. Never edit shards in place. Reads union shards via DuckDB's glob support.
- **Lowercase 0x-prefixed addresses** everywhere — already so in source data, never re-cased.
- **Fill identification:** `transaction_hash` carries the on-chain tx; multi-fill transactions are common (~30 % of fills). Phase 2 currently does not collapse same-bucket fills (see METRICS_REFERENCE §1.5 caveat). A composite (`transaction_hash`, fill-order-within-tx) key would be needed for strict per-fill identity but is not currently materialised.
- **Sign convention** in `trader_actions`: from each address's POV. `token_delta > 0` ⇒ received outcome tokens; `usd_delta > 0` ⇒ received USDC.
- **Gitignored**: everything in `data/`, every `*.parquet` (defence in depth), `.env`.

## Known limitations

1. **Merge/split blindspot on NegRisk markets.** Trader bookkeeping that mints USDC into YES+NO and merges back is not an `OrderFilled` event. `phantom_position_score > 1` flags these traders; their `pos_*` per-position metrics are inflated/deflated, but `mkt_*` metrics aggregate correctly.
2. **Open positions excluded from PnL.** No mark-to-market on currently-open positions. Phase 2 is closed-markets-only.
3. **Sharpe annualisation is naive** (`sqrt(N / years_active)`); it produces absurdly inflated values at the artifact tail (~15 orders of magnitude). Cohort filters require simultaneous `n_closed_positions > 200`, `active_days > 90`, `mkt_std_pnl > 1.0` guards. Sharpe is a DIAGNOSTIC, never a primary ranker.
4. **Bankroll is lifetime peak**, not point-in-time. Descriptive only — using it for forward-looking sizing leaks future capacity into past decisions.
5. **External reconciliation against Polymarket UI is order-of-magnitude only.** The public API exposes only currently-open mark-to-market and partial-realized within open positions; lifetime PnL lives only on the UI profile pages and isn't scriptable.
6. **Data tail lag ~16 days** as of 2026-05-10. Goldsky's subgraph indexer trails ~9 days; the last sync added ~7 days more. Walk-forward backtests should set test cutoffs ≥ 9 days behind run date.
7. **`peak_fill_abs_token` is a proxy**, not a true running peak. The real cumulative-max would have required sorting a 1.4 B-row collapsed table (OOM at 300 GB temp during the build).
8. **`style_median_fill_size_usd` ships as NULL.** `approx_quantile` over the 2 B-row CROSS JOIN'd stream silently exited the build process.

Full caveat list with source `file:line` references in [`docs/METRICS_REFERENCE.md`](docs/METRICS_REFERENCE.md) §"Known limitations".

## Reference reading

- **Strategy thesis**: [Tatv — The Polymarket Copy Trading Field Manual](https://tatv.ai/article/the-polymarket-copy-trading-field-manual) and [From Prediction to Allocation: A Cohort Copy Trading System for Polymarket](https://tatv.ai/article/from-prediction-to-allocation-a-cohort-copy-trading-system-for-polymarket).
- **Data quality findings**: [`notes/overview/data_quality/validation_report.md`](notes/overview/data_quality/validation_report.md). Generated by `scripts/validation/*.py`.
- **External reconciliation**: [`notes/overview/data_quality/api_reconciliation_v1.md`](notes/overview/data_quality/api_reconciliation_v1.md). 10 hand-picked traders, Polymarket Data API cross-check.
- **Sample profile output**: [`notes/copytrade/profile_domah.md`](notes/copytrade/profile_domah.md). Generated by `data_infra.trader_profile.profile_trader('0x9d84ce…')`.
- **Schema + formulas**: [`docs/METRICS_REFERENCE.md`](docs/METRICS_REFERENCE.md).
- **Analytical findings across phases**: [`RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md).
- **Notebook bridge**: [`notebooks/README.md`](notebooks/README.md).

Reference repo (DO NOT copy code, GPL-3.0): https://github.com/warproxxx/poly_data — used to understand Goldsky query shape and trade-processing logic.

## What's next

**Phase 5 backlog**:
- Point-in-time bankroll (rolling 30-day-prior max, not lifetime peak) so backtests don't leak future capacity.
- Data refresh pipeline (incremental Goldsky pull + closed_positions delta + traders panel rebuild). Currently full rebuilds.
- Walk-forward backtesting on cohort-selection rules. Pick a date, rank with pre-date data only, simulate copying qualifying traders' next-period trades, compound across windows.
- CPCV (combinatorial purged cross-validation) on the same selection rules. Same discipline as the existing crypto-momentum framework in repo root (do **not** import from there — separate venv).
- Out-of-sample edge validation against the 5 manual-inspection candidates surfaced in [`RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md).

**Future v2** (deferred):
- Merge/split indexer (CTF events outside `OrderFilled`) to remove the NegRisk blindspot.
- Mark-to-market on open positions (requires current Polymarket prices — Gamma API has these).
- True running-peak position size (`peak_position_size`, replacing the v1 `peak_fill_abs_token` proxy).
- Exact `n_distinct_counterparties` (replacing HLL approximation), if cohort-selection precision warrants it.
- `style_median_fill_size_usd` via sampled pass.

Vault hub: brain/POLYMARKET_BRAIN.md · brain/COWORK.md · table terms: notes/overview/data_quality/polymarket_table_dictionary.md
