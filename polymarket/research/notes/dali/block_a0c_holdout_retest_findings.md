---
tags: [dali, a0c, holdout, oos, retest, results]
title: A0c Holdout Retest Findings
created: 2026-05-30
status: closed
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
---

# A0c Holdout Retest Findings

> Hub: [[COWORK]]

> Table terms: [[polymarket_table_dictionary]]


## Summary

This note retests A0c as holdout data for the Dali binary-bet, passive deep-book fade, and TOB hit-rate diagnostics. Retest A fails OOS, Retest B confirms the passive deep-book artifact, and Retest C confirms the A1.3 73.7% TOB point estimate did not replicate. The conclusion closes the local Dali signal with no remaining needs-more-data caveat.

## Headline

- Retest A binary-bet / 4h-boundary: **failed-OOS** under CI lower > 0, n >= 30, and >= 3 windows.
- Retest B passive deep-book fade: **artifact-confirmed** under CI lower > 0, n >= 30, and fill >= 2.0%.
- Retest C TOB hit-rate diagnostic: **artifact-confirmed** versus the A1.3 73.7% point estimate.
- Retest A and B both fail; the local Dali signal is closed with no remaining needs-more-data caveat.
- Retest A/C crypto-roll universe after the >=300-trade gate: 6 windows (btc). ETH/SOL peer windows were captured but did not meet the preregistered trade-count gate.

## Retest A Top Rows

| scope | slug | signal | tail | hold/target | n | fill | mean | CI | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | ALL | ofi_5s | bottom_decile | fixed_300s | 149 |  | -2341.5 bps | [-3222.4 bps, -1667.9 bps] | failed_OOS |
| pooled | ALL | tob_imbalance_level | bottom_decile | fixed_300s | 164 |  | -2621.5 bps | [-3602.4 bps, -1770.7 bps] | failed_OOS |
| pooled | ALL | ofi_5s | top_decile | fixed_300s | 151 |  | -2804.0 bps | [-4424.3 bps, -1580.8 bps] | failed_OOS |
| pooled | ALL | tob_imbalance_level | top_decile | fixed_300s | 195 |  | -2830.7 bps | [-4903.6 bps, -1451.5 bps] | failed_OOS |
| pooled | ALL | tob_imbalance_level | bottom_decile | boundary_4h | 6 |  | 1322.5 bps | [-7786.4 bps, 13318.7 bps] | failed_OOS |
| pooled | ALL | ofi_5s | bottom_decile | boundary_4h | 6 |  | -542.8 bps | [-13610.9 bps, 11369.7 bps] | failed_OOS |
| pooled | ALL | tob_imbalance_level | top_decile | boundary_4h | 6 |  | -3181.4 bps | [-14915.8 bps, 6064.4 bps] | failed_OOS |
| pooled | ALL | ofi_5s | top_decile | boundary_4h | 6 |  | -3985.1 bps | [-20217.6 bps, 8091.9 bps] | failed_OOS |
| window | btc-updown-4h-1780099200 | ofi_5s | top_decile | fixed_300s | 28 |  | -607.3 bps | [-941.6 bps, -336.3 bps] | failed_OOS |
| window | btc-updown-4h-1780099200 | tob_imbalance_level | top_decile | fixed_300s | 23 |  | -505.3 bps | [-946.4 bps, -59.3 bps] | failed_OOS |
| window | btc-updown-4h-1780099200 | ofi_5s | bottom_decile | fixed_300s | 29 |  | -1242.1 bps | [-1521.8 bps, -954.8 bps] | failed_OOS |
| window | btc-updown-4h-1780113600 | tob_imbalance_level | top_decile | fixed_300s | 31 |  | -872.6 bps | [-1827.8 bps, 209.0 bps] | failed_OOS |

## Retest B Top Rows

| scope | slug | signal | tail | hold/target | n | fill | mean | CI | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market | us-x-iran-permanent-peace-deal-by-june-30-2026-. | ofi_5s | bottom_decile | half_to_micro_price | 34 | 2.3% | -26.7 bps | [-66.1 bps, 0.0 bps] | artifact_confirmed |
| market | us-x-iran-permanent-peace-deal-by-june-30-2026-. | ofi_5s | bottom_decile | micro_price | 34 | 2.3% | -26.7 bps | [-69.5 bps, 0.0 bps] | artifact_confirmed |
| market | us-x-iran-permanent-peace-deal-by-june-30-2026-. | ofi_5s | bottom_decile | half_to_micro_price | 35 | 2.4% | -34.3 bps | [-81.7 bps, 0.0 bps] | artifact_confirmed |
| market | us-x-iran-permanent-peace-deal-by-june-30-2026-. | ofi_5s | bottom_decile | micro_price | 35 | 2.4% | -34.3 bps | [-82.0 bps, 0.0 bps] | artifact_confirmed |
| market | us-x-iran-permanent-peace-deal-by-june-30-2026-. | ofi_5s | bottom_decile | micro_price | 30 | 2.0% | -39.3 bps | [-85.1 bps, 0.0 bps] | artifact_confirmed |
| market | us-x-iran-permanent-peace-deal-by-june-30-2026-. | ofi_5s | bottom_decile | half_to_micro_price | 30 | 2.0% | -39.3 bps | [-85.9 bps, 0.0 bps] | artifact_confirmed |
| market | us-x-iran-permanent-peace-deal-by-june-30-2026-. | ofi_5s | bottom_decile | micro_price | 27 | 1.8% | -43.7 bps | [-93.5 bps, 0.0 bps] | artifact_confirmed |
| market | us-x-iran-permanent-peace-deal-by-june-30-2026-. | ofi_5s | bottom_decile | half_to_micro_price | 27 | 1.8% | -43.7 bps | [-96.8 bps, 0.0 bps] | artifact_confirmed |
| market | nhl-mon-car-2026-05-29-spread-home-1pt5 | ofi_5s | bottom_decile | micro_price | 5 | 0.9% | -111.1 bps | [-112.0 bps, -110.2 bps] | artifact_confirmed |
| market | nhl-mon-car-2026-05-29-spread-home-1pt5 | ofi_5s | bottom_decile | micro_price | 5 | 0.9% | -111.1 bps | [-112.5 bps, -110.2 bps] | artifact_confirmed |
| market | nhl-mon-car-2026-05-29-spread-home-1pt5 | ofi_5s | bottom_decile | micro_price | 5 | 0.9% | -111.1 bps | [-112.5 bps, -110.2 bps] | artifact_confirmed |
| market | nhl-mon-car-2026-05-29-spread-home-1pt5 | ofi_5s | bottom_decile | half_to_micro_price | 5 | 0.9% | -111.1 bps | [-112.5 bps, -110.2 bps] | artifact_confirmed |

## Retest C Rows

| scope | slug | tail | n | hit | CI | delta vs A1.3 | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | ALL | top_abs_decile | 3143 | 36.0% | [32.8%, 39.2%] | -37.7% | artifact_confirmed |
| window | btc-updown-4h-1780056000 | top_abs_decile | 735 | 43.5% | [36.5%, 50.8%] | -30.2% | artifact_confirmed |
| window | btc-updown-4h-1780099200 | top_abs_decile | 89 | 46.1% | [35.0%, 58.7%] | -27.6% | artifact_confirmed |
| window | btc-updown-4h-1780070400 | top_abs_decile | 777 | 41.1% | [33.9%, 47.2%] | -32.6% | artifact_confirmed |
| window | btc-updown-4h-1780113600 | top_abs_decile | 227 | 43.2% | [32.6%, 53.1%] | -30.5% | artifact_confirmed |
| window | btc-updown-4h-1780084800 | top_abs_decile | 615 | 28.1% | [23.1%, 33.7%] | -45.6% | artifact_confirmed |
| window | btc-updown-4h-1780041600 | top_abs_decile | 700 | 25.6% | [20.6%, 30.5%] | -48.1% | artifact_confirmed |

## Feature Panel Append Check

| run | feature rows | markets | first | last |
| --- | --- | --- | --- | --- |
| a0 | 2,563,417 | 12 | 2026-05-27 12:07:28 | 2026-05-28 12:07:27 |
| a0b | 1,761,469 | 9 | 2026-05-27 23:19:13 | 2026-05-28 11:19:13 |
| a0c | 4,536,389 | 17 | 2026-05-29 11:01:30 | 2026-05-30 11:01:30 |
| a0c_roll | 2,445,782 | 38 | 2026-05-29 11:09:20 | 2026-05-30 11:09:21 |

Raw JSONL event counts match the final A0c notes:

| capture | event | final note | raw scan | match |
| --- | --- | --- | --- | --- |
| main_a0c | book | 26,289 | 26,289 | yes |
| main_a0c | price_change | 2,150,145 | 2,150,145 | yes |
| main_a0c | best_bid_ask | 197,022 | 197,022 | yes |
| main_a0c | last_trade_price | 12,788 | 12,788 | yes |
| crypto_roll | book | 13,273 | 13,273 | yes |
| crypto_roll | price_change | 1,068,165 | 1,068,165 | yes |
| crypto_roll | best_bid_ask | 290,489 | 290,489 | yes |
| crypto_roll | last_trade_price | 5,690 | 5,690 | yes |

## Discovery Thresholds

Deep-book relative-depth q90 from A0/A0b discovery rows: `2.13135`.

| scope | signal | q10 | q90 | abs q90 | n |
| --- | --- | --- | --- | --- | --- |
| crypto_4h_up_down | ofi_5s | -0.999911 | 0.999911 | 1.8944 | 450,994 |
| crypto_4h_up_down | tob_imbalance_level | -0.913043 | 0.8 | 0.937422 | 450,994 |
| geopolitics | ofi_5s | -0.000704083 | 0.00207177 | 0.0148222 | 983,122 |
| geopolitics | tob_imbalance_level | -0.733458 | 0.885763 | 0.923157 | 983,122 |
| all | ofi_5s | -0.219829 | 0.154145 | 0.606206 | 3,087,512 |
| all | tob_imbalance_level | -0.867963 | 0.898364 | 0.945358 | 3,087,512 |

## Method

- A0/A0b are discovery only; A0c main and A0c crypto_roll are strict holdout rows.
- A0c main is tagged `a0c`; crypto_roll is tagged `a0c_roll` in `data/analysis/block_a1_features.parquet`.
- `will-jd-vance` is excluded from all retests.
- Event ordering and horizons use `exchange_ts` when present, falling back to `received_at`.
- Entry and exit quote states require `is_book_state_complete` and `book_staleness_seconds <= 5`.
- Confidence intervals are 300s clock-block bootstrap intervals.
- Retest A uses taker entry/exit, net of `FEE_BY_CATEGORY`, with non-overlap per market.
- Retest B uses the P2 passive fill proxy with W=1s, maker entry rebate, taker exit, and non-overlap after fill.
- Retest C is PnL-independent and non-overlap at the 5s horizon.

## Outputs

- `data/analysis/csv_outputs/dali/a0c_holdout_retest_surface.csv`
- `notes/dali/block_a0c_holdout_retest_findings.md`
