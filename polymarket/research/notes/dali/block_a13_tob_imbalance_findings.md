---
tags: [dali, block-a13, tob-imbalance, results]
title: Block A1.3 TOB Imbalance Level Deep-Dive
created: 2026-05-28
status: archived
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
---

# Block A1.3 TOB Imbalance Level Deep-Dive

> Hub: [[COWORK]]

> Table terms: [[polymarket_table_dictionary]]


## Summary

A1.3 tests `tob_imbalance_level` as a standing top-of-book state variable on the Block A1 feature panel. The top-decile current-level signal shows a strong 5s hit rate of 73.7% with caveats around market-specific persistence and execution costs. The note promotes TOB imbalance as a primary candidate signal at this diagnostic stage, while later notes test and narrow its executable use.

## Headline

current-level 5s top decile hit 73.7% (CI [67.6%, 77.7%], n=299,864) with 72.9 bps directional return. When TOB and OFI agree at 5s, TOB hit is 61.0%; when they disagree, inspect the joint table rather than assuming OFI adds edge. Conditional TFI in high-TOB rows hits 62.6% at 5s. The result supports promoting `tob_imbalance_level` to a primary A2 candidate, with two caveats: it is a standing state variable rather than flow, and the executable-cost question still needs bid/ask entry-exit tests rather than mid-return diagnostics.

## Method

This sidecar uses `data/analysis/block_a1_features.parquet` only. `tob_imbalance_level = (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size)` is sign-normalized by `direction_factor`, so positive means imbalance favors the market-direction outcome. Two variants are reported:

- `current_level`: the standing top-of-book imbalance at signal time `t`; this is the primary A1.3 signal because the variable is a level/state, not an order-flow sum.
- `window_mean_level`: the rolling mean of the level over the horizon; this reconciles to A1.1's component sweep.

Deciles are global equal-count buckets within each horizon and variant, using absolute signal magnitude. Decile 10 is the largest absolute imbalance, not the most bullish imbalance. Hit rate is `sign(signal) == sign(future_directional_mid_return_bps)`.

## A1.1 Reconciliation

| h | A13 window hit | A11 hit | diff | A13 dir ret | A11 dir ret |
| --- | --- | --- | --- | --- | --- |
| 1 | 71.3% | 71.3% | +0.0000 | 23.1 bps | 23.1 bps |
| 5 | 70.7% | 70.7% | +0.0000 | 30.9 bps | 30.9 bps |
| 30 | 67.6% | 67.6% | +0.0000 | 49.4 bps | 49.4 bps |
| 300 | 71.6% | 71.6% | +0.0000 | 25.3 bps | 25.3 bps |

## Decile Aggregate

### Current Level

| h | hit | CI | dir ret | n | top family | share |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 70.6% | [61.8%, 78.1%] | 32.7 bps | 299,898 | daily_crypto_up_down | 46.8% |
| 5 | 73.7% | [67.6%, 77.7%] | 72.9 bps | 299,864 | daily_crypto_up_down | 46.8% |
| 30 | 74.1% | [70.3%, 78.8%] | 146.6 bps | 299,676 | daily_crypto_up_down | 46.9% |
| 300 | 69.4% | [63.4%, 76.4%] | 356.8 bps | 299,466 | daily_crypto_up_down | 46.9% |

Read: the current standing imbalance is not a weaker proxy for the A1.1 result; it is stronger at 5s and 30s in this panel. That is consistent with a book-state signal rather than a decaying flow signal.

### A1.1-Compatible Window Mean

| h | hit | CI | dir ret | n | top family | share |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 71.3% | [65.3%, 78.5%] | 23.1 bps | 308,231 | daily_crypto_up_down | 38.9% |
| 5 | 70.7% | [64.5%, 77.2%] | 30.9 bps | 304,208 | daily_crypto_up_down | 35.3% |
| 30 | 67.6% | [61.6%, 74.3%] | 49.4 bps | 308,260 | geopolitics_policy | 35.1% |
| 300 | 71.6% | [61.1%, 80.9%] | 25.3 bps | 304,797 | geopolitics_policy | 37.5% |

## Persistence

The persistence test measures how long the sign-normalized TOB imbalance sign lasts before flipping. This distinguishes a durable state from a rapidly changing flow-like feature.

| market | family | runs | median flip | p90 flip | <=5s | <=30s |
| --- | --- | --- | --- | --- | --- | --- |
| will-anthropic-have-the-best-ai-model-at-the-end-of-june-202 | ai_product | 8,960 | 0.4s | 3.7s | 91.1% | 95.9% |
| will-china-invade-taiwan-by-december-31-2027 | geopolitics_policy | 74 | 1.7m | 1.2h | 38.9% | 44.4% |
| bitcoin-up-or-down-on-may-28-2026 | daily_crypto_up_down | 6,084 | 1.1s | 32.6s | 67.7% | 89.0% |
| ethereum-up-or-down-on-may-28-2026 | daily_crypto_up_down | 11,070 | 0.5s | 14.9s | 78.2% | 95.2% |
| us-iran-nuclear-deal-before-2027 | geopolitics_policy | 3,922 | 2.4s | 24.1s | 58.2% | 90.6% |
| will-openai-announce-earbuds-or-headphones-in-2026 | ai_product | 9,818 | 4.5s | 42.1s | 52.2% | 80.0% |
| nato-x-russia-military-clash-by-december-31-2026-244 | geopolitics_policy | 882 | 1.2s | 26.8s | 87.5% | 90.2% |
| strait-of-hormuz-traffic-returns-to-normal-by-end-of-june | geopolitics_policy | 88 | 3.3m | 45.2m | 20.9% | 30.2% |
| btc-updown-4h-1779926400 | crypto_4h_up_down | 5,998 | 0.5s | 10.3s | 82.3% | 96.7% |
| btc-updown-4h-1779940800 | crypto_4h_up_down | 5,376 | 0.5s | 14.7s | 76.7% | 95.0% |
| strait-of-hormuz-traffic-returns-to-normal-by-july-31 | geopolitics_policy | 82 | 4.3m | 51.6m | 12.5% | 22.5% |
| will-mojtaba-khamenei-be-head-of-state-in-iran-end-of-2026 | geopolitics_policy | 180 | 3.6m | 45.4m | 18.0% | 30.3% |

Read: persistence is market-specific rather than family-uniform. Crypto and several AI/geopolitics books flip on sub-second-to-seconds state-run horizons, while China/Hormuz-style geopolitical books can keep the same imbalance sign for minutes to hours. That means `tob_imbalance_level` is sometimes a fast dynamic state and sometimes a slow book descriptor, so A2 should carry persistence controls.

## Spread And Depth Controls

Bucket labels are reconstructed with the same quantile-bucket method used in A1.1. The A1.1 CSV does not persist numeric bucket boundaries, so A1.3 reconstructs the buckets from the same feature table and code path.

### Spread Bucket, 5s Top Decile

| spread_bucket | 5s hit | CI | dir ret | n |
| --- | --- | --- | --- | --- |
| spread_q2 | 86.2% | [79.5%, 91.2%] | 73.5 bps | 70,039 |
| spread_q3 | 74.3% | [67.2%, 80.8%] | 55.5 bps | 84,773 |
| spread_q4_wide | 70.9% | [66.3%, 75.4%] | 117.9 bps | 59,015 |
| spread_q1_tight | 68.9% | [60.6%, 77.7%] | 58.6 bps | 86,037 |

### Relative Depth Bucket, 5s Top Decile

| relative_depth_bucket | 5s hit | CI | dir ret | n |
| --- | --- | --- | --- | --- |
| depth_q4_deep | 81.8% | [77.5%, 85.6%] | 91.5 bps | 142,748 |
| depth_q3 | 76.3% | [71.9%, 81.5%] | 89.9 bps | 68,098 |
| depth_q1_shallow | 66.4% | [54.2%, 74.5%] | 57.3 bps | 35,555 |
| depth_q2 | 54.1% | [38.6%, 71.6%] | 11.8 bps | 53,463 |

### Spread x Depth, 5s Top Decile

| spread_x_depth | 5s hit | CI | dir ret | n |
| --- | --- | --- | --- | --- |
| spread_q2|depth_q2 | 93.1% | [85.8%, 98.8%] | 42.6 bps | 7,425 |
| spread_q2|depth_q4_deep | 87.2% | [82.0%, 92.0%] | 80.0 bps | 41,111 |
| spread_q1_tight|depth_q4_deep | 85.7% | [79.1%, 91.9%] | 101.3 bps | 29,678 |
| spread_q2|depth_q1_shallow | 84.0% | [75.9%, 94.7%] | 72.0 bps | 7,044 |
| spread_q3|depth_q4_deep | 81.4% | [74.0%, 87.3%] | 73.8 bps | 43,052 |
| spread_q2|depth_q3 | 81.1% | [71.2%, 91.4%] | 71.4 bps | 14,459 |
| spread_q1_tight|depth_q3 | 78.0% | [70.3%, 84.9%] | 95.4 bps | 16,580 |
| spread_q4_wide|depth_q4_deep | 75.6% | [70.3%, 80.6%] | 124.0 bps | 28,907 |
| spread_q3|depth_q3 | 74.9% | [67.6%, 84.2%] | 61.8 bps | 20,918 |
| spread_q4_wide|depth_q3 | 74.4% | [66.7%, 80.8%] | 137.3 bps | 16,141 |
| spread_q4_wide|depth_q1_shallow | 71.9% | [61.4%, 79.0%] | 118.7 bps | 7,055 |
| spread_q3|depth_q1_shallow | 62.0% | [39.4%, 83.2%] | 26.9 bps | 11,077 |
| spread_q1_tight|depth_q1_shallow | 58.4% | [46.5%, 72.6%] | 38.1 bps | 10,379 |
| spread_q4_wide|depth_q2 | 54.7% | [43.3%, 70.9%] | 46.3 bps | 6,912 |
| spread_q1_tight|depth_q2 | 48.8% | [33.1%, 72.5%] | 2.1 bps | 29,400 |
| spread_q3|depth_q2 | 48.2% | [28.9%, 81.3%] | -6.9 bps | 9,726 |

Read: the 5s top-decile signal survives every spread quartile, so it is not just a tight-spread artifact. It is much stronger in deep relative-depth cells and weak or negative in several `depth_q2` cells, so depth conditioning should travel into A2.

## Joint TOB x OFI Signal

The joint table compares sign agreement between current TOB imbalance and depth-normalized OFI. `same_sign` means both predictors point the same way; `disagree` means they conflict; `imbalance_only` and `ofi_only` mean one signal is zero.

| joint bin | TOB hit | OFI hit | TOB dir | OFI dir | n |
| --- | --- | --- | --- | --- | --- |
| imbalance_only | 60.6% | n/a | 4.2 bps | 0.0 bps | 1,512,830 |
| same_sign | 61.0% | 61.0% | 51.9 bps | 51.9 bps | 846,566 |
| disagree | 57.2% | 42.8% | 23.1 bps | -23.1 bps | 701,966 |
| ofi_only | n/a | 51.6% | 0.0 bps | -15.7 bps | 23,061 |

Read: OFI does not show a clean incremental sign edge over current TOB imbalance in this run. When TOB and OFI agree, the hit rate is identical by construction; when they disagree, TOB remains positive and OFI points the wrong way in the 5s aggregate.

## Conditional TFI

This checks whether trade-flow imbalance becomes more useful inside high-TOB-imbalance regimes.

| h | slice | TFI hit | CI | dir ret | n | TFI nonzero |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | all_tfi_nonzero | 56.4% | [53.8%, 58.5%] | 43.0 bps | 180,349 | 100.0% |
| 5 | tob_top_decile_all | 62.6% | [56.6%, 68.2%] | 8.6 bps | 299,864 | 7.2% |
| 5 | tob_top_decile_tfi_nonzero | 62.6% | [56.7%, 66.7%] | 119.8 bps | 21,528 | 100.0% |
| 30 | all_tfi_nonzero | 53.9% | [51.8%, 56.7%] | 16.4 bps | 504,646 | 100.0% |
| 30 | tob_top_decile_all | 54.2% | [49.5%, 60.6%] | 1.7 bps | 299,676 | 19.1% |
| 30 | tob_top_decile_tfi_nonzero | 54.2% | [49.2%, 59.2%] | 9.1 bps | 57,256 | 100.0% |

Read: TFI becomes more interesting inside high-TOB-imbalance rows at 5s, but the effect is much weaker by 30s. Treat this as a conditional 5s interaction candidate, not a standalone replacement for the TOB level signal.

## Plots

![](data/analysis/block_a13_plots/block_a13_tob_decile_hit_current-level.png)
![](data/analysis/block_a13_plots/block_a13_tob_decile_hit_window-mean-level.png)
![](data/analysis/block_a13_plots/block_a13_tob_persistence_hist_by_market.png)
![](data/analysis/block_a13_plots/block_a13_spread_bucket_top_decile_hit_heatmap.png)
![](data/analysis/block_a13_plots/block_a13_spread_bucket_top_decile_n_heatmap.png)
![](data/analysis/block_a13_plots/block_a13_relative_depth_bucket_top_decile_hit_heatmap.png)
![](data/analysis/block_a13_plots/block_a13_relative_depth_bucket_top_decile_n_heatmap.png)
![](data/analysis/block_a13_plots/block_a13_spread_x_depth_top_decile_hit_heatmap.png)
![](data/analysis/block_a13_plots/block_a13_spread_x_depth_top_decile_n_heatmap.png)
![](data/analysis/block_a13_plots/block_a13_tob_ofi_joint_hit_heatmap.png)
![](data/analysis/block_a13_plots/block_a13_conditional_tfi_hit_rate.png)

## Outputs

- `data/analysis/csv_outputs/dali/block_a13_tob_decile_aggregate.csv`
- `data/analysis/csv_outputs/dali/block_a13_a11_reconciliation.csv`
- `data/analysis/csv_outputs/dali/block_a13_tob_persistence_runs.csv`
- `data/analysis/csv_outputs/dali/block_a13_tob_persistence_by_market.csv`
- `data/analysis/csv_outputs/dali/block_a13_tob_control_buckets.csv`
- `data/analysis/csv_outputs/dali/block_a13_tob_ofi_joint_signal.csv`
- `data/analysis/csv_outputs/dali/block_a13_tob_conditional_tfi.csv`
- `data/analysis/block_a13_plots/`

Recommended next action for Justin: make `tob_imbalance_level` a primary A2 feature candidate alongside OFI, but require executable bid/ask cost tests before treating it as tradable edge.
