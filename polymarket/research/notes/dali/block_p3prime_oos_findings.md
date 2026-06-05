---
tags: [dali, p3prime, oos, micro-price, reversion, results]
---

# P3' A0c OOS Reversion Replication Findings

> Table terms: [[polymarket_table_dictionary]]

## Headline

Fails OOS: no A0c deep-book geopolitics cell meets CI lower > 0, n >= 30, and fill rate >= 2%. Under the preregistered rule, the local Dali microstructure signal is closed; go to P6.

Best OOS row by CI lower bound: `a0c_geopolitics_deep_market` / `a0c:1962237` / `bottom_decile`, W=10s, target `micro_price`, timeout=10s, n=9, fill 1.6%, mean -25.3 bps, CI [-80.9 bps, 30.4 bps], verdict `fails_n_lt_30`.

## Decision Rule

- Preregistered replication bar: CI lower > 0, n >= 30, and fill rate >= 2.0% on A0c-only deep-book geopolitics cells.
- Rows clearing CI lower > 0: 0
- Rows clearing CI lower > 0 and n >= 30: 0
- Rows clearing CI lower > 0, n >= 30, and fill >= 2.0%: 0

## OOS Top Rows

| segment | market | slug | tail | fill | target | timeout | signals | exec | fill rate | mean | CI | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market | a0c:1962237 | us-x-iran-permanent-peace-deal-by. | bottom_decile | W=10 | micro price | 10s | 556 | 9 | 1.6% | -25.3 bps | [-80.9 bps, 30.4 bps] | fails_n_lt_30 |
| market | a0c:1962237 | us-x-iran-permanent-peace-deal-by. | bottom_decile | W=10 | half to micro price | 10s | 556 | 9 | 1.6% | -25.3 bps | [-80.9 bps, 30.4 bps] | fails_n_lt_30 |
| all | ALL | ALL | both_tails | W=1 | micro price | 10s | 4,507 | 20 | 0.4% | -32.4 bps | [-87.9 bps, 23.0 bps] | fails_n_lt_30 |
| all | ALL | ALL | both_tails | W=1 | half to micro price | 10s | 4,507 | 20 | 0.4% | -32.4 bps | [-87.9 bps, 23.0 bps] | fails_n_lt_30 |
| market | a0c:1962237 | us-x-iran-permanent-peace-deal-by. | both_tails | W=10 | micro price | 10s | 861 | 14 | 1.6% | -33.7 bps | [-90.0 bps, 22.7 bps] | fails_n_lt_30 |
| market | a0c:1962237 | us-x-iran-permanent-peace-deal-by. | both_tails | W=10 | half to micro price | 10s | 861 | 14 | 1.6% | -33.7 bps | [-90.0 bps, 22.7 bps] | fails_n_lt_30 |
| all | ALL | ALL | both_tails | W=1 | micro price | 30s | 4,507 | 19 | 0.4% | -26.9 bps | [-90.8 bps, 37.0 bps] | fails_n_lt_30 |
| all | ALL | ALL | both_tails | W=1 | half to micro price | 30s | 4,507 | 19 | 0.4% | -26.9 bps | [-90.8 bps, 37.0 bps] | fails_n_lt_30 |
| market | a0c:1962237 | us-x-iran-permanent-peace-deal-by. | bottom_decile | W=1 | micro price | 10s | 556 | 8 | 1.4% | -28.4 bps | [-92.0 bps, 35.2 bps] | fails_n_lt_30 |
| market | a0c:1962237 | us-x-iran-permanent-peace-deal-by. | bottom_decile | W=1 | half to micro price | 10s | 556 | 8 | 1.4% | -28.4 bps | [-92.0 bps, 35.2 bps] | fails_n_lt_30 |
| market | a0c:1962237 | us-x-iran-permanent-peace-deal-by. | both_tails | W=1 | micro price | 10s | 861 | 13 | 1.5% | -36.2 bps | [-97.9 bps, 25.4 bps] | fails_n_lt_30 |
| market | a0c:1962237 | us-x-iran-permanent-peace-deal-by. | both_tails | W=1 | half to micro price | 10s | 861 | 13 | 1.5% | -36.2 bps | [-97.9 bps, 25.4 bps] | fails_n_lt_30 |
| all | ALL | ALL | top_decile | W=1 | micro price | 30s | 2,272 | 10 | 0.4% | -6.9 bps | [-98.9 bps, 85.2 bps] | fails_n_lt_30 |
| all | ALL | ALL | top_decile | W=1 | half to micro price | 30s | 2,272 | 10 | 0.4% | -6.9 bps | [-98.9 bps, 85.2 bps] | fails_n_lt_30 |
| all | ALL | ALL | top_decile | W=10 | micro price | 30s | 2,272 | 10 | 0.4% | -6.9 bps | [-98.9 bps, 85.2 bps] | fails_n_lt_30 |
| all | ALL | ALL | top_decile | W=10 | half to micro price | 30s | 2,272 | 10 | 0.4% | -6.9 bps | [-98.9 bps, 85.2 bps] | fails_n_lt_30 |

## Method

- Discovery feature panel: `data/analysis/block_a1_features.parquet` with run IDs `a0, a0b`. Confirmed A0c is absent.
- A0c feature panel: `data/analysis/block_a0c_features.parquet`. This is append-only separate output; `block_a1_features.parquet` was not mutated.
- A0c rows loaded: 4,536,389 across 17 markets. A0c deep geopolitics tail events tested: 4,507.
- Active A0c capture processes observed at replay time: 0. If nonzero, the feature parquet is a snapshot of available shards at replay time.
- Deep-book filter: A0/A0b discovery relative-depth q90 = `2.17338`; A0c rows must satisfy `relative_depth >= q90`.
- Signal: `ofi_5s = direction_factor * rolling_sum(ofi_combined_event, 5s) / market_mean_depth`.
- Trigger: per-market OFI bottom/top deciles; the preregistered decision focuses on `bottom_decile` and `both_tails`. The grid is not retuned on A0c.
- Execution: passive fade at the touch, P2 fill proxy, entry maker rebate, taker exit to bid/ask, non-overlap after actual fill.
- Grid: W in (1, 10), target in ('micro_price', 'half_to_micro_price'), timeout in (5, 10, 30, 60).
- CI: normal interval over contiguous 300s block means of non-overlap executed PnL.

## Output

- `data/analysis/csv_outputs/dali/p3prime_oos_replication.csv`
