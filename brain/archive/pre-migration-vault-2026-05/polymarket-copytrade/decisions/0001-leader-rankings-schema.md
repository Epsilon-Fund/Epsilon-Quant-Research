---
title: "ADR 0001 — leader_rankings.parquet schema (superseded framework)"
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

# 0001 — leader_rankings.parquet schema (research → execution interface)

- **date**: 2026-05-09 (v0.1); 2026-05-10 (v0.2 patched against `formulas.md`); 2026-05-11 (v0.3 — bankroll gap closed); 2026-05-18 (v0.4 status note — framework supersession flagged)
- **status**: **proposed but framework superseded.** v0.4 note: as of the 2026-05-16 pm-data audit pivot, the *cohort-membership-predicts-copyability* thesis underlying this schema's `cohort_pool` / `rank_score` design is empirically wrong (n=7 per-leader audits, weak cross-leader signal). The schema's STRONG passthrough columns (`address`, `negrisk_volume_share`, `phantom_position_score`, `last_updated_utc`, `bankroll_method`, `estimated_bankroll_usd` post-bankroll-rebuild) all remain useful; the *ranking* layer needs rework against per-leader audit output rather than cohort-pool membership. See `inbox/_archive/2026-05/2026-05-16_pm-data.md` for the audit framework. Specific reframings needed in a v0.5: (1) `pricing_mode_recommended` heuristic is **inverted** (high `style_role_balance` doesn't favour `leader_fill` — copying a maker-conviction leader as a maker is adversely selected); (2) `phantom_position_score < 2.0` is partially obsolete — authoritative arb-exclusion now lives in `traders_directionality.parquet` and the new `split_position_signature` flag; (3) `rank_score` semantics need to be redesigned around per-leader deployable-cell count + recency, not cohort membership × PnL.

## Context

The polymarket copytrade project has two work streams: research (cohort selection from 2.58M traders) and execution (live mirror bot, currently single-leader hardcoded). Today they're decoupled by hardcoded `POLYMARKET_LEADER_ADDRESS` in exec's env config; integration is a future deliverable.

Exec's state dump (`research/01-execution.md`) explicitly enumerated the fields it would need from a future `leader_rankings.parquet`. Research's `traders.parquet` (`research/02-data.md`) ships most of these as lifetime metrics. We need to lock the contract before research starts producing it, and before exec tries to consume it.

Constraints in play:
- **File-based, never code-imported**: research writes, exec reads. Decouples deploy cadence and language stacks.
- **Atomically written**: write to `.tmp`, rename → exec never reads a partial file.
- **Refresh cycle**: exec reads on startup + every ~4h; staleness ≥24h disqualifies the file (exec falls back to env-hardcoded leader).
- **Time windows matter**: exec wants 30d-rolling for sizing/risk inputs; data currently ships lifetime. Mismatch is the load-bearing gap.

## Decision

`leader_rankings.parquet`, atomically written by research, consumed by execution. One row per leader address.

### Schema (proposed)

| Column | Type | Source (data side) | Purpose |
|---|---|---|---|
| `proxy_address` | str (lowercase 0x, 42 chars) | `traders.address` | canonical leader ID; matched against RTDS `proxyWallet` |
| `rank_score` | float64 | Phase 4 cohort selection | opaque score; exec uses for "trader N from list" or "above threshold X" |
| `rank_position` | int32 | Phase 4 cohort selection | 1-indexed rank within the file |
| `cohort_pool` | str | Phase 4 (which of 6 pools) | which pool the trader was selected from; exec may filter by pool |
| `last_updated_utc` | timestamp | run timestamp | exec refuses to act if `now - last_updated_utc > 24h` |
| **sizing inputs** | | | |
| `estimated_bankroll_usd` | float64 | **Phase 5 point-in-time** (gap — see below) | feeds `leader_fraction` math |
| `bankroll_method` | str | research config (e.g. `"rolling_max_30d_v1"`) | so exec can detect algorithm changes |
| **risk inputs (journal context, not halting)** | | | |
| `typical_position_count_30d` | int32 | derived from `traders_*` (gap — see below) | sanity check unusual activity |
| `typical_hold_duration_hours_30d` | float64 | derived (currently `style_*_holding_hours` lifetime) | exit-tracker fallback timeout |
| `winrate_30d` | float64 | derived (gap — currently lifetime) | journal context |
| `pnl_usd_30d` | float64 | derived (gap — currently lifetime) | journal context |
| **pricing mode** | | | |
| `pricing_mode_recommended` | str enum (`leader_fill` \| `current_book`) | derived from `style_maker_taker_ratio_30d` | exec respects if present |
| `maker_taker_ratio_30d` | float64 | currently `style_maker_taker_ratio` lifetime (gap) | supporting evidence |
| **NegRisk caveats** | | | |
| `negrisk_volume_share` | float64 | `traders.negrisk_volume_share` | exec downweights or skips heavy NegRisk leaders until handling lands |
| `phantom_position_score` | float64 | `traders.phantom_position_score` | sanity flag |

### Format

- Parquet, snappy compression.
- Atomic write: `tmp/leader_rankings.parquet.tmp` → `mv` → `output/leader_rankings.parquet`.
- Path: `polymarket/research/output/leader_rankings.parquet` (research repo).
- Exec consumes via configured path (`POLYMARKET_LEADER_RANKINGS_PATH`); falls back to env address if missing/stale.

## Alternatives considered

- **JSON instead of Parquet** — rejected. Parquet matches research's existing tooling (DuckDB over Parquet glob); cohort sizes (~hundreds to low-thousands of leaders) are well within JSON range, but consistency wins.
- **Database (sqlite/Postgres)** — rejected. Both sides explicitly avoid stateful databases (exec's "journal is the database"; research's "no Postgres"). File-based is the house style.
- **Code-imported (research as a Python package exec depends on)** — rejected. Coupling deploy cadences. The bot needs to be restartable independent of research re-runs.

## Consequences

**Good**
- Clear contract: research and exec can iterate independently as long as the schema holds.
- Honest separation of concerns: research decides the *who* and *how big*, exec decides the *how to place*.
- Fallback to env-hardcoded leader means exec is never blocked by integration regression.

**Cost / known debt**
- **Time-window gap** — most exec inputs want 30d-rolling, data currently ships lifetime. Either research adds 30d versions to `traders.parquet` (and downstream cohort pools), or exec accepts lifetime-as-proxy. Recommend: research adds 30d windows during Phase 5, since walk-forward backtesting wants those windows anyway.
- **~~Bankroll gap~~ — CLOSED in v0.3** (2026-05-11). `rolling_bankroll_usd_30d` ships point-in-time, built from `bankroll_timeseries.parquet`. Multi-leader integration with proportional sizing is **unblocked from the data side.** Rebuild also fixed a Phase 3 bug (placeholder-date filter inconsistency inflated whale peaks 5-7×).
- **NegRisk gap** — schema flags it via `negrisk_volume_share` + `phantom_position_score`, but neither side handles it natively. Composite handling (data: index merge/split events; exec: multi-outcome position keying) deferred. **Mitigation in v1**: prefer leaders with `negrisk_volume_share < 0.3` and `phantom_position_score < 2` (low end of normal — see phantom-baseline correction below). The `sports_directional_fast` cohort encodes this filter (`formulas.md` §3).
- **Phantom score baseline correction (v0.3, 2026-05-11)** — `formulas.md` §2.4 says "1.0 = clean directional, ≫1.0 = NegRisk arb / split-merge behaviour." Empirically, all six cohort pools sit at 1.4–1.8 phantom baseline — the theoretical "1.0 = pure directional" is rarely observed. Use phantom as **relative signal** (1.7 normal vs 8+ heavy arb), not absolute threshold. The `< 2.0` filter still works as a coarse arb-screen but now reads as "low end of normal," not "pure directional."
- **Sharpe rank-score warning** — if `rank_score` is derived from `mkt_sharpe`, the **SUSPECT** rating per `formulas.md` §2.3 propagates. Naive annualisation: `mean_pnl / std_pnl × sqrt(N / years_active)` blows up at the tail (p99 = 11.17, max = 107.20 on `mkt_sharpe`). Cohort guards (`n_closed_positions > 200 AND active_days > 90 AND mkt_std_pnl > 1.0`) mitigate but don't fix. Prefer `mkt_profit_factor` and `mkt_dollar_win_rate` for ranking, treat Sharpe as diagnostic.

## Source-column mapping (v0.2)

Cross-references against `research/notes/formulas.md`. All `traders.*` references map to `traders.parquet` columns; `cohorts.*` are `data/cohorts/*.parquet` columns.

| Schema column | Data-side source | Trustworthiness | Notes |
|---|---|---|---|
| `proxy_address` | `traders.address` | STRONG | already lowercase 0x; matches RTDS `proxyWallet` directly |
| `rank_score` | derived (Phase 4) | depends on what's used | if from `mkt_profit_factor` / `mkt_dollar_win_rate`: STRONG. If from `mkt_sharpe`: SUSPECT — diagnostic only |
| `rank_position` | derived | STRONG | trivial sort |
| `cohort_pool` | `cohorts/*.parquet` filename | STRONG | one of the 6; address can appear in multiple |
| `last_updated_utc` | run timestamp | STRONG | |
| `estimated_bankroll_usd` | `traders.rolling_bankroll_usd_30d` (v0.3 — point-in-time, shipped 2026-05-11) | **STRONG for sizing** | rebuilt as point-in-time from `bankroll_timeseries.parquet` (429M rows, address-day). Old `est_bankroll_usd_30d_max_approx` is now `est_bankroll_lifetime_peak_deprecated` — do not use for sizing. The rebuild also surfaced/fixed a Phase 3 bug (inconsistent placeholder-date filtering inflated whale peaks 5-7×). |
| `bankroll_method` | research config string | STRONG | now `"point_in_time_30d_v2"` (was `"lifetime_max_v1"`) |
| `typical_position_count_30d` | **gap** — data ships lifetime `n_closed_positions` | gap | needs 30d-rolling version during Phase 4 prep |
| `typical_hold_duration_hours_30d` | **gap** — data ships lifetime `style_avg_holding_hours` / `style_median_holding_hours` | gap | needs 30d-rolling version |
| `winrate_30d` | **gap** — data ships lifetime `mkt_dollar_win_rate` (preferred) or `mkt_win_rate` | gap | journal context only, not halting |
| `pnl_usd_30d` | **gap** — data ships lifetime `mkt_total_pnl` | gap | journal context only |
| `pricing_mode_recommended` | derived from `style_role_balance` (or `style_maker_taker_ratio`) | MODERATE heuristic | `style_role_balance > 0.7` → `leader_fill` (conviction maker); `< 0.3` → `current_book` (taker speed); middle → global default |
| `maker_taker_ratio_30d` | **gap** — data ships lifetime `traders.style_maker_taker_ratio` | gap | needs 30d window |
| `negrisk_volume_share` | `traders.negrisk_volume_share` | STRONG | direct passthrough |
| `phantom_position_score` | `traders.phantom_position_score` | STRONG (diagnostic) | direct passthrough; ~1.0 = clean, ≫1 = arb |

**Net of mapping (v0.3)**: 7 of 14 columns are STRONG (was 6 — bankroll graduated from SUSPECT to STRONG); 5 are gaps awaiting 30d-rolling versions; 1 is SUSPECT-context-dependent (`rank_score` if Sharpe-derived); 1 is a heuristic (`pricing_mode_recommended`). The bankroll gap that gated multi-leader integration is **closed.**

## Status / next actions

- [ ] data-track: review the source-column mapping above — confirm each cell, flag any I got wrong
- [ ] data-track: add 30d-rolling versions of `n_closed_positions`, `style_avg_holding_hours`, `mkt_dollar_win_rate`, `mkt_total_pnl`, `style_maker_taker_ratio` during Phase 4 prep — closes 5 of 5 gap rows
- [ ] data-track: Phase 5 — point-in-time bankroll computation; replace `estimated_bankroll_usd` source from lifetime peak to point-in-time
- [ ] exec-track: confirm `rank_score` semantics — single scalar or per-pool? If we want pool-aware ranking, schema may need `rank_score_per_pool` map column
- [ ] exec-track: add `POLYMARKET_LEADER_RANKINGS_PATH` env var and read-on-refresh logic (deferred until single-leader smoke is live)
- [ ] revisit ADR after first cohort pool is materialised and a non-trivial leader candidate is picked from it — concrete numbers may shift the schema
