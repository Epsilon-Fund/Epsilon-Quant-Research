---
tags: [dali, block-a14d, executable-cost, results]
---
> Hub: [[COWORK]]


## Summary

- Scope: Block A1.4d Tight-Spread Entry Findings in the Dali research lineage area.
- Existing takeaway/status: Positive PnL appears only in 1 market(s), concentrated at 300s. The best interpretable positive row is `a0b:2364426` at 300s with S=500 bps: 843.6 bps mean PnL on 4,324/6,682 events (64.7% trigger), CI [-799.2 bps, 1770.3 bps]. The no_filter baseline for that same market/horizon is 334.1 bps, so the filter amplifies that one cell rather than creating a broad cross-market flip. Spread-conditional entry alone is...
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
> Table terms: [[polymarket_table_dictionary]]

# Block A1.4d Tight-Spread Entry Findings

## Headline

Positive PnL appears only in 1 market(s), concentrated at 300s. The best interpretable positive row is `a0b:2364426` at 300s with S=500 bps: 843.6 bps mean PnL on 4,324/6,682 events (64.7% trigger), CI [-799.2 bps, 1770.3 bps]. The no_filter baseline for that same market/horizon is 334.1 bps, so the filter amplifies that one cell rather than creating a broad cross-market flip. Spread-conditional entry alone is therefore not a robust tradeability result. Overall, 6 of 198 market-horizon-threshold cells have positive mean PnL after taker entry, fixed-horizon taker exit, and fees on both legs.

## Method

- Candidate universe: all `primary_read` markets from `block_a1_results.csv`, not only A1.4's `signal_present_pre_cost` rows.
- Signal: current top-of-book imbalance level, `tob_imbalance_level = direction_factor * tob_imbalance`.
- Deciles: equal-count ranked deciles by `abs(tob_imbalance_level)` within each `(run_id, market_id)`, computed once from current-state rows and reused across horizons. Only decile 10 is traded.
- Entry filter: current `spread_bps <= S`, with `S in {50, 100, 200, 500, 1000, no_filter}`.
- Execution: same fixed-horizon A1.4 touch round trip. A long token signal pays `best_ask` and exits at future `best_bid`; a short token signal receives `best_bid` and exits at future `best_ask`.
- Horizons: 5s, 30s, and 300s.
- Fees: A1 `FEE_BY_CATEGORY`, charged on both entry and exit as dollars per share and normalized by entry price.
- Bootstrap: 200-sample block bootstrap of mean PnL with contiguous 300s clock-time blocks.

## Cross-Market Threshold Summary

The `event trigger` column is the pooled share of top-decile signal events that pass the spread filter. The `median market trigger` column avoids one very active market dominating the read. Deltas are versus each market-horizon's `no_filter` baseline.

| h | threshold | positive cells | event trigger | median market trigger | mean pnl | mean delta vs no_filter |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 50 | 0/11 | 8.4% | 0.0% | -225.1 bps | 612.8 bps |
| 5 | 100 | 0/11 | 13.7% | 0.0% | -279.9 bps | 557.9 bps |
| 5 | 200 | 0/11 | 44.9% | 49.7% | -344.3 bps | 433.1 bps |
| 5 | 500 | 0/11 | 76.3% | 83.0% | -498.1 bps | 279.3 bps |
| 5 | 1000 | 0/11 | 90.1% | 96.7% | -610.7 bps | 166.7 bps |
| 5 | no_filter | 0/11 | 100.0% | 100.0% | -777.5 bps | 0.0 bps |
| 30 | 50 | 0/11 | 8.4% | 0.0% | -250.3 bps | 521.7 bps |
| 30 | 100 | 0/11 | 13.7% | 0.0% | -312.8 bps | 459.2 bps |
| 30 | 200 | 0/11 | 44.9% | 49.7% | -338.0 bps | 385.9 bps |
| 30 | 500 | 0/11 | 76.3% | 83.0% | -482.7 bps | 241.2 bps |
| 30 | 1000 | 0/11 | 90.1% | 96.7% | -579.5 bps | 144.4 bps |
| 30 | no_filter | 0/11 | 100.0% | 100.0% | -723.9 bps | 0.0 bps |
| 300 | 50 | 1/11 | 8.4% | 0.0% | -145.7 bps | 525.5 bps |
| 300 | 100 | 1/11 | 13.7% | 0.0% | -212.5 bps | 458.7 bps |
| 300 | 200 | 1/11 | 44.9% | 49.7% | -261.1 bps | 380.5 bps |
| 300 | 500 | 1/11 | 76.3% | 83.0% | -390.6 bps | 251.0 bps |
| 300 | 1000 | 1/11 | 90.1% | 96.7% | -508.4 bps | 133.2 bps |
| 300 | no_filter | 1/11 | 100.0% | 100.0% | -641.6 bps | 0.0 bps |

## Per-Market Threshold Survival

| market | slug | best S | h | trigger | mean pnl | win | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a0:558934 | will-spain-win-the-2026-fifa-world-cup-963 | 50 | 5 | 49.3% | -115.6 bps | 0.0% | no threshold tested produces positive PnL |
| a0:558936 | will-france-win-the-2026-fifa-world-cup-924 | 50 | 300 | 49.7% | -113.2 bps | 0.0% | no threshold tested produces positive PnL |
| a0:665325 | us-iran-nuclear-deal-before-2027 | 200 | 300 | 33.1% | -128.2 bps | 0.4% | no threshold tested produces positive PnL |
| a0b:1971905 | strait-of-hormuz-traffic-returns-to-normal-by-en | 50 | 5 | 0.0% | 0.0 bps | 0.0% | no threshold tested produces positive PnL |
| a0b:2327929 | nba-okc-sas-2026-05-28 | 200 | 5 | 50.8% | -416.4 bps | 0.0% | no threshold tested produces positive PnL |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 100 | 300 | 12.9% | -45.6 bps | 1.6% | no threshold tested produces positive PnL |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | 50 | 300 | 7.7% | -59.4 bps | 0.4% | no threshold tested produces positive PnL |
| a0b:2364426 | btc-updown-4h-1779912000 | 500 | 300 | 64.7% | 843.6 bps | 64.4% | tradeable with spread filter S=500 at 300s, CI crosses zero |
| a0b:2366225 | btc-updown-4h-1779926400 | 50 | 5 | 1.0% | -86.7 bps | 1.1% | no threshold tested produces positive PnL |
| a0b:2367777 | btc-updown-4h-1779940800 | 200 | 5 | 22.6% | -541.4 bps | 4.2% | no threshold tested produces positive PnL |
| a0b:566136 | will-psg-win-the-202526-champions-league | 200 | 5 | 50.4% | -425.2 bps | 0.0% | no threshold tested produces positive PnL |

## Best Cells

| market | slug | h | S | events | trigger | mean | median | win | CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0b:2364426 | btc-updown-4h-1779912000 | 300 | 50 | 2/6,682 | 0.0% | 1979.8 bps | 1979.8 bps | 100.0% | [n/a, n/a] |
| a0b:2364426 | btc-updown-4h-1779912000 | 300 | 100 | 2/6,682 | 0.0% | 1979.8 bps | 1979.8 bps | 100.0% | [n/a, n/a] |
| a0b:2364426 | btc-updown-4h-1779912000 | 300 | 500 | 4,324/6,682 | 64.7% | 843.6 bps | 1179.3 bps | 64.4% | [-799.2 bps, 1770.3 bps] |
| a0b:2364426 | btc-updown-4h-1779912000 | 300 | 1000 | 5,740/6,682 | 85.9% | 613.4 bps | 1111.3 bps | 60.8% | [-836.0 bps, 1450.9 bps] |
| a0b:2364426 | btc-updown-4h-1779912000 | 300 | 200 | 1,908/6,682 | 28.6% | 578.7 bps | 494.6 bps | 52.6% | [-252.0 bps, 1460.0 bps] |
| a0b:2364426 | btc-updown-4h-1779912000 | 300 | no_filter | 6,682/6,682 | 100.0% | 334.1 bps | 1033.3 bps | 58.8% | [-1120.9 bps, 1175.6 bps] |
| a0b:1971905 | strait-of-hormuz-traffic-returns-to-normal | 5 | 50 | 4/18,606 | 0.0% | 0.0 bps | 0.0 bps | 0.0% | [n/a, n/a] |
| a0b:1971905 | strait-of-hormuz-traffic-returns-to-normal | 5 | 100 | 4/18,606 | 0.0% | 0.0 bps | 0.0 bps | 0.0% | [n/a, n/a] |
| a0b:1971905 | strait-of-hormuz-traffic-returns-to-normal | 30 | 50 | 4/18,606 | 0.0% | 0.0 bps | 0.0 bps | 0.0% | [n/a, n/a] |
| a0b:1971905 | strait-of-hormuz-traffic-returns-to-normal | 30 | 100 | 4/18,606 | 0.0% | 0.0 bps | 0.0 bps | 0.0% | [n/a, n/a] |
| a0b:1971905 | strait-of-hormuz-traffic-returns-to-normal | 300 | 50 | 4/18,606 | 0.0% | 0.0 bps | 0.0 bps | 0.0% | [n/a, n/a] |
| a0b:1971905 | strait-of-hormuz-traffic-returns-to-normal | 300 | 100 | 4/18,606 | 0.0% | 0.0 bps | 0.0 bps | 0.0% | [n/a, n/a] |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 300 | 100 | 6,016/46,715 | 12.9% | -45.6 bps | -41.8 bps | 1.6% | [-56.5 bps, -39.3 bps] |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 300 | 50 | 5,958/46,715 | 12.8% | -45.7 bps | -41.8 bps | 1.5% | [-54.9 bps, -40.2 bps] |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 30 | 50 | 5,958/46,711 | 12.8% | -52.9 bps | -46.0 bps | 0.1% | [-62.9 bps, -48.2 bps] |

## A13 Spread Sanity Reference

This is not used for the A14d PnL math; it is a sanity check that the absolute spread thresholds bracket A13's observed top-decile spread regimes.

| A13 spread bucket | top-decile rows | mean spread | mean hit | mean dir ret |
| --- | --- | --- | --- | --- |
| spread_q1_tight | 344,115 | 58.7 bps | 68.4% | 121.6 bps |
| spread_q2 | 279,991 | 158.2 bps | 81.8% | 145.0 bps |
| spread_q3 | 338,845 | 397.9 bps | 73.7% | 146.9 bps |
| spread_q4_wide | 235,953 | 1469.2 bps | 67.5% | 213.2 bps |

## Interpretation

Spread filtering is necessary for any executable taker version of this signal, but the fixed-horizon version is still carrying too much spread and adverse movement. If a positive cell appears, it should be treated as a narrow capture-window diagnostic until latency, capacity at touch, and A14b-style exit rules are layered in. If no positive cell appears, the conclusion is cleaner: spread-conditional entry alone does not rescue the top-decile TOB signal.

Recommended next action for Justin: do not treat spread filtering alone as sufficient; combine tight-spread entry with the A14b exit-rule work and require a minimum trigger-rate/capacity screen before A2 trading design.
