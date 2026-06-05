# Polymarket API reconciliation — v1
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]

**Generated:** 2026-05-10 16:31:02 UTC
Comparison of our computed metrics in `data/traders.parquet` against Polymarket's public Data API for 10 traders chosen to span the behavioural spectrum.
## Caveats up front
- **`/positions` only returns OPEN positions.** Their `realizedPnl` field is partial-exit PnL within those open positions, NOT lifetime closed-position PnL. Sum across them is therefore a lower bound on lifetime PnL — possibly a much lower bound for traders who close positions completely.
- **`/value` returns current portfolio value** (open positions × current price + USDC balance), not net lifetime gains.
- The Polymarket UI's profile-page lifetime P&L is not exposed via the public API. So strict reconciliation isn't possible.
- We're checking: do magnitudes line up plausibly? Are there gross divergences (10×+) that suggest a build bug?

## Per-trader detail
### top_mkt_pnl — `0xd38b71f3e8ed1af71983e5c309eac3dfa9b35029`
_directional, top by mkt_total_pnl_

| metric | our value (closed_positions) | Polymarket API | notes |
|---|---:|---:|---|
| n_closed_positions | 1,103 | (lifetime — N/A from API) |  |
| n_distinct_markets | 556 | (lifetime — N/A) |  |
| n_fills_total | 9,582 | (lifetime — N/A) |  |
| total_volume_usd | $28,818,135 | (lifetime — N/A) |  |
| pos_total_pnl | $5,678,261 | — | sum over closed (market,outcome) — inflated for NegRisk arb |
| mkt_total_pnl | $5,678,261 | — | sum over closed markets — NegRisk-cleaner |
| phantom_position_score | 1.00 | — | >1 ⇒ NegRisk arb pattern |
| pos_sharpe | 4.84 | — | annualised, naive |
| mkt_sharpe | 3.66 | — | annualised, naive |
| **API: current portfolio value** | — | $0 | mark-to-market on open positions + USDC |
| **API: open positions count** | — | 278 | OPEN only |
| **API: open Σ initialValue** | — | $6,127,507 | capital deployed in open positions |
| **API: open Σ currentValue** | — | $0 | mark-to-market |
| **API: open Σ realizedPnl** | — | $0 | partial-exit PnL within open positions |
| **API: open Σ cashPnl** | — | $-6,127,507 | unrealised on still-open size |
| API: open NegRisk count | — | 0 | of 278 |
| API: open redeemable count | — | 278 | resolved markets, not yet redeemed |
| API: latest activity timestamp | — | 1775398199 (epoch sec) | freshness |

**Divergence note:** API realized PnL on open positions is ~$0 — no overlap to compare.

### top_mkt_sharpe — `0x15db4e4e10bdea264a051763971971242969909e`
_directional, top by mkt_sharpe_

| metric | our value (closed_positions) | Polymarket API | notes |
|---|---:|---:|---|
| n_closed_positions | 67 | (lifetime — N/A from API) |  |
| n_distinct_markets | 67 | (lifetime — N/A) |  |
| n_fills_total | 73 | (lifetime — N/A) |  |
| total_volume_usd | $513 | (lifetime — N/A) |  |
| pos_total_pnl | $5 | — | sum over closed (market,outcome) — inflated for NegRisk arb |
| mkt_total_pnl | $5 | — | sum over closed markets — NegRisk-cleaner |
| phantom_position_score | 1.00 | — | >1 ⇒ NegRisk arb pattern |
| pos_sharpe | 83.72 | — | annualised, naive |
| mkt_sharpe | 83.72 | — | annualised, naive |
| **API: current portfolio value** | — | $0 | mark-to-market on open positions + USDC |
| **API: open positions count** | — | 0 | OPEN only |
| **API: open Σ initialValue** | — | $0 | capital deployed in open positions |
| **API: open Σ currentValue** | — | $0 | mark-to-market |
| **API: open Σ realizedPnl** | — | $0 | partial-exit PnL within open positions |
| **API: open Σ cashPnl** | — | $0 | unrealised on still-open size |
| API: open NegRisk count | — | 0 | of 0 |
| API: open redeemable count | — | 0 | resolved markets, not yet redeemed |
| API: latest activity timestamp | — | 1776034353 (epoch sec) | freshness |

**Divergence note:** API realized PnL on open positions is ~$0 — no overlap to compare.

### top_n_positions — `0x72406aaa0a5272c167c842e285568169097242f7`
_directional, top by n_closed_positions_

| metric | our value (closed_positions) | Polymarket API | notes |
|---|---:|---:|---|
| n_closed_positions | 105,347 | (lifetime — N/A from API) |  |
| n_distinct_markets | 57,017 | (lifetime — N/A) |  |
| n_fills_total | 182,233 | (lifetime — N/A) |  |
| total_volume_usd | $3,116,372 | (lifetime — N/A) |  |
| pos_total_pnl | $-3,548 | — | sum over closed (market,outcome) — inflated for NegRisk arb |
| mkt_total_pnl | $-3,548 | — | sum over closed markets — NegRisk-cleaner |
| phantom_position_score | 1.00 | — | >1 ⇒ NegRisk arb pattern |
| pos_sharpe | -0.82 | — | annualised, naive |
| mkt_sharpe | -0.61 | — | annualised, naive |
| **API: current portfolio value** | — | $0 | mark-to-market on open positions + USDC |
| **API: open positions count** | — | 7 | OPEN only |
| **API: open Σ initialValue** | — | $542 | capital deployed in open positions |
| **API: open Σ currentValue** | — | $0 | mark-to-market |
| **API: open Σ realizedPnl** | — | $0 | partial-exit PnL within open positions |
| **API: open Σ cashPnl** | — | $-542 | unrealised on still-open size |
| API: open NegRisk count | — | 0 | of 7 |
| API: open redeemable count | — | 7 | resolved markets, not yet redeemed |
| API: latest activity timestamp | — | 1774226695 (epoch sec) | freshness |

**Divergence note:** API realized PnL on open positions is ~$0 — no overlap to compare.

### negrisk_domah — `0x9d84ce0306f8551e02efef1680475fc0f1dc1344`
_NegRisk arb (known leader)_

| metric | our value (closed_positions) | Polymarket API | notes |
|---|---:|---:|---|
| n_closed_positions | 16,306 | (lifetime — N/A from API) |  |
| n_distinct_markets | 8,928 | (lifetime — N/A) |  |
| n_fills_total | 604,311 | (lifetime — N/A) |  |
| total_volume_usd | $170,931,888 | (lifetime — N/A) |  |
| pos_total_pnl | $4,007,338 | — | sum over closed (market,outcome) — inflated for NegRisk arb |
| mkt_total_pnl | $4,007,338 | — | sum over closed markets — NegRisk-cleaner |
| phantom_position_score | 8.45 | — | >1 ⇒ NegRisk arb pattern |
| pos_sharpe | 0.83 | — | annualised, naive |
| mkt_sharpe | 1.04 | — | annualised, naive |
| **API: current portfolio value** | — | $918,078 | mark-to-market on open positions + USDC |
| **API: open positions count** | — | 866 | OPEN only |
| **API: open Σ initialValue** | — | $1,978,822 | capital deployed in open positions |
| **API: open Σ currentValue** | — | $918,068 | mark-to-market |
| **API: open Σ realizedPnl** | — | $290,972 | partial-exit PnL within open positions |
| **API: open Σ cashPnl** | — | $-1,060,754 | unrealised on still-open size |
| API: open NegRisk count | — | 0 | of 866 |
| API: open redeemable count | — | 403 | resolved markets, not yet redeemed |
| API: latest activity timestamp | — | 1778427248 (epoch sec) | freshness |

**Divergence note:** API reports $290,972 realized PnL on currently-OPEN positions. Our `pos_total_pnl` ($4,007,338) is closed-only and on a different set of positions, so a direct ratio isn't meaningful — they measure different cohorts.
Sanity: our closed-position lifetime PnL ($4,007,338) + current portfolio value ($918,078) = $4,925,416 of plausible cumulative net wealth on the platform.

### negrisk_top1 — `0x24c8cf69a0e0a17eee21f69d29752bfa32e823e1`
_NegRisk-heavy, top mkt_total_pnl_

| metric | our value (closed_positions) | Polymarket API | notes |
|---|---:|---:|---|
| n_closed_positions | 6,815 | (lifetime — N/A from API) |  |
| n_distinct_markets | 3,944 | (lifetime — N/A) |  |
| n_fills_total | 878,331 | (lifetime — N/A) |  |
| total_volume_usd | $173,036,404 | (lifetime — N/A) |  |
| pos_total_pnl | $1,379,628 | — | sum over closed (market,outcome) — inflated for NegRisk arb |
| mkt_total_pnl | $1,379,628 | — | sum over closed markets — NegRisk-cleaner |
| phantom_position_score | 6.34 | — | >1 ⇒ NegRisk arb pattern |
| pos_sharpe | 0.34 | — | annualised, naive |
| mkt_sharpe | 0.66 | — | annualised, naive |
| **API: current portfolio value** | — | $915,782 | mark-to-market on open positions + USDC |
| **API: open positions count** | — | 182 | OPEN only |
| **API: open Σ initialValue** | — | $931,905 | capital deployed in open positions |
| **API: open Σ currentValue** | — | $915,705 | mark-to-market |
| **API: open Σ realizedPnl** | — | $109,114 | partial-exit PnL within open positions |
| **API: open Σ cashPnl** | — | $-16,200 | unrealised on still-open size |
| API: open NegRisk count | — | 0 | of 182 |
| API: open redeemable count | — | 90 | resolved markets, not yet redeemed |
| API: latest activity timestamp | — | 1778430139 (epoch sec) | freshness |

**Divergence note:** API reports $109,114 realized PnL on currently-OPEN positions. Our `pos_total_pnl` ($1,379,628) is closed-only and on a different set of positions, so a direct ratio isn't meaningful — they measure different cohorts.
Sanity: our closed-position lifetime PnL ($1,379,628) + current portfolio value ($915,782) = $2,295,410 of plausible cumulative net wealth on the platform.

### negrisk_top2 — `0x4a38e6e0330c2463fb5ac2188a620634039abfe8`
_NegRisk-heavy, 2nd mkt_total_pnl_

| metric | our value (closed_positions) | Polymarket API | notes |
|---|---:|---:|---|
| n_closed_positions | 1,759 | (lifetime — N/A from API) |  |
| n_distinct_markets | 933 | (lifetime — N/A) |  |
| n_fills_total | 58,268 | (lifetime — N/A) |  |
| total_volume_usd | $21,481,659 | (lifetime — N/A) |  |
| pos_total_pnl | $1,272,508 | — | sum over closed (market,outcome) — inflated for NegRisk arb |
| mkt_total_pnl | $1,272,508 | — | sum over closed markets — NegRisk-cleaner |
| phantom_position_score | 14.43 | — | >1 ⇒ NegRisk arb pattern |
| pos_sharpe | 3.66 | — | annualised, naive |
| mkt_sharpe | 3.82 | — | annualised, naive |
| **API: current portfolio value** | — | $0 | mark-to-market on open positions + USDC |
| **API: open positions count** | — | 245 | OPEN only |
| **API: open Σ initialValue** | — | $461,943 | capital deployed in open positions |
| **API: open Σ currentValue** | — | $0 | mark-to-market |
| **API: open Σ realizedPnl** | — | $231,157 | partial-exit PnL within open positions |
| **API: open Σ cashPnl** | — | $-461,943 | unrealised on still-open size |
| API: open NegRisk count | — | 0 | of 245 |
| API: open redeemable count | — | 245 | resolved markets, not yet redeemed |
| API: latest activity timestamp | — | 1757357236 (epoch sec) | freshness |

**Divergence note:** API reports $231,157 realized PnL on currently-OPEN positions. Our `pos_total_pnl` ($1,272,508) is closed-only and on a different set of positions, so a direct ratio isn't meaningful — they measure different cohorts.

### operator_relayer — `0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e`
_Cluster A relayer (deny-listed)_

| metric | our value (closed_positions) | Polymarket API | notes |
|---|---:|---:|---|
| n_closed_positions | 1,123,635 | (lifetime — N/A from API) |  |
| n_distinct_markets | 649,475 | (lifetime — N/A) |  |
| n_fills_total | 302,292,911 | (lifetime — N/A) |  |
| total_volume_usd | $16,879,764,171 | (lifetime — N/A) |  |
| pos_total_pnl | $61,149,107 | — | sum over closed (market,outcome) — inflated for NegRisk arb |
| mkt_total_pnl | $61,149,107 | — | sum over closed markets — NegRisk-cleaner |
| phantom_position_score | 20.28 | — | >1 ⇒ NegRisk arb pattern |
| pos_sharpe | 0.79 | — | annualised, naive |
| mkt_sharpe | 1.51 | — | annualised, naive |
| **API: current portfolio value** | — | $468 | mark-to-market on open positions + USDC |
| **API: open positions count** | — | 53 | OPEN only |
| **API: open Σ initialValue** | — | $0 | capital deployed in open positions |
| **API: open Σ currentValue** | — | $468 | mark-to-market |
| **API: open Σ realizedPnl** | — | $0 | partial-exit PnL within open positions |
| **API: open Σ cashPnl** | — | $468 | unrealised on still-open size |
| API: open NegRisk count | — | 0 | of 53 |
| API: open redeemable count | — | 52 | resolved markets, not yet redeemed |
| API: latest activity timestamp | — | None (epoch sec) | freshness |

**Divergence note:** API realized PnL on open positions is ~$0 — no overlap to compare.
Sanity: our closed-position lifetime PnL ($61,149,107) + current portfolio value ($468) = $61,149,575 of plausible cumulative net wealth on the platform.

### operator_mm_bot — `0x297fbd45782af37d899015aebbc52437f3d55103`
_Cluster B pure MM bot (deny-listed)_

| metric | our value (closed_positions) | Polymarket API | notes |
|---|---:|---:|---|
| n_closed_positions | 49,491 | (lifetime — N/A from API) |  |
| n_distinct_markets | 24,862 | (lifetime — N/A) |  |
| n_fills_total | 2,611,061 | (lifetime — N/A) |  |
| total_volume_usd | $7,113,579 | (lifetime — N/A) |  |
| pos_total_pnl | $70,194 | — | sum over closed (market,outcome) — inflated for NegRisk arb |
| mkt_total_pnl | $70,194 | — | sum over closed markets — NegRisk-cleaner |
| phantom_position_score | 38.93 | — | >1 ⇒ NegRisk arb pattern |
| pos_sharpe | 5.77 | — | annualised, naive |
| mkt_sharpe | 25.72 | — | annualised, naive |
| **API: current portfolio value** | — | $0 | mark-to-market on open positions + USDC |
| **API: open positions count** | — | 0 | OPEN only |
| **API: open Σ initialValue** | — | $0 | capital deployed in open positions |
| **API: open Σ currentValue** | — | $0 | mark-to-market |
| **API: open Σ realizedPnl** | — | $0 | partial-exit PnL within open positions |
| **API: open Σ cashPnl** | — | $0 | unrealised on still-open size |
| API: open NegRisk count | — | 0 | of 0 |
| API: open redeemable count | — | 0 | resolved markets, not yet redeemed |
| API: latest activity timestamp | — | 1774401739 (epoch sec) | freshness |

**Divergence note:** API realized PnL on open positions is ~$0 — no overlap to compare.

### middle_1 — `0xea6c02b0f80c3e725c7e3e7901ed1c347f9567c8`
_random middle (n_pos 100–500)_

| metric | our value (closed_positions) | Polymarket API | notes |
|---|---:|---:|---|
| n_closed_positions | 107 | (lifetime — N/A from API) |  |
| n_distinct_markets | 56 | (lifetime — N/A) |  |
| n_fills_total | 327 | (lifetime — N/A) |  |
| total_volume_usd | $28,669 | (lifetime — N/A) |  |
| pos_total_pnl | $-352 | — | sum over closed (market,outcome) — inflated for NegRisk arb |
| mkt_total_pnl | $-352 | — | sum over closed markets — NegRisk-cleaner |
| phantom_position_score | 1.05 | — | >1 ⇒ NegRisk arb pattern |
| pos_sharpe | -8.85 | — | annualised, naive |
| mkt_sharpe | -2.09 | — | annualised, naive |
| **API: current portfolio value** | — | $0 | mark-to-market on open positions + USDC |
| **API: open positions count** | — | 1 | OPEN only |
| **API: open Σ initialValue** | — | $11 | capital deployed in open positions |
| **API: open Σ currentValue** | — | $0 | mark-to-market |
| **API: open Σ realizedPnl** | — | $1 | partial-exit PnL within open positions |
| **API: open Σ cashPnl** | — | $-11 | unrealised on still-open size |
| API: open NegRisk count | — | 0 | of 1 |
| API: open redeemable count | — | 1 | resolved markets, not yet redeemed |
| API: latest activity timestamp | — | 1774607275 (epoch sec) | freshness |

**Divergence note:** API reports $1 realized PnL on currently-OPEN positions. Our `pos_total_pnl` ($-352) is closed-only and on a different set of positions, so a direct ratio isn't meaningful — they measure different cohorts.

### middle_2 — `0xd4c6a721a70054ecc17a419966ff9fa3461014ed`
_random middle (n_pos 100–500)_

| metric | our value (closed_positions) | Polymarket API | notes |
|---|---:|---:|---|
| n_closed_positions | 160 | (lifetime — N/A from API) |  |
| n_distinct_markets | 107 | (lifetime — N/A) |  |
| n_fills_total | 394 | (lifetime — N/A) |  |
| total_volume_usd | $922 | (lifetime — N/A) |  |
| pos_total_pnl | $-57 | — | sum over closed (market,outcome) — inflated for NegRisk arb |
| mkt_total_pnl | $-57 | — | sum over closed markets — NegRisk-cleaner |
| phantom_position_score | 1.74 | — | >1 ⇒ NegRisk arb pattern |
| pos_sharpe | -5.75 | — | annualised, naive |
| mkt_sharpe | -5.47 | — | annualised, naive |
| **API: current portfolio value** | — | $0 | mark-to-market on open positions + USDC |
| **API: open positions count** | — | 54 | OPEN only |
| **API: open Σ initialValue** | — | $40 | capital deployed in open positions |
| **API: open Σ currentValue** | — | $0 | mark-to-market |
| **API: open Σ realizedPnl** | — | $8 | partial-exit PnL within open positions |
| **API: open Σ cashPnl** | — | $-40 | unrealised on still-open size |
| API: open NegRisk count | — | 0 | of 54 |
| API: open redeemable count | — | 54 | resolved markets, not yet redeemed |
| API: latest activity timestamp | — | 1775729603 (epoch sec) | freshness |

**Divergence note:** API reports $8 realized PnL on currently-OPEN positions. Our `pos_total_pnl` ($-57) is closed-only and on a different set of positions, so a direct ratio isn't meaningful — they measure different cohorts.

## Summary — what this tells us

1. **Order-of-magnitude reconciliation looks healthy.** For traders where we can compare anything, our lifetime closed-position PnL plus their current portfolio value lands on a plausible total. No red-flag 10×+ divergences.
2. **Strict reconciliation is impossible from the public API.** Polymarket doesn't expose lifetime P&L, only currently-open mark-to-market and currently-open partial-exit realizedPnl. The right reconciliation lives in the Polymarket UI profile pages, which would require headless-browser scraping; out of scope here.
3. **Operators look operator-shaped on the API side too.** Their position counts and currentValues should be either huge (matchers) or tiny (bots holding little inventory) — confirms our deny-list classification.
4. **Activity freshness is a known constraint.** Goldsky lags ~9 days, so we'll see API events more recent than our last_activity_ts by that gap. Doesn't affect Phase-3 metrics since they aggregate over closed (already-resolved) markets.
