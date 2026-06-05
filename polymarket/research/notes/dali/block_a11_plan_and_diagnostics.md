---
tags: [dali, block-a1, a1-1, diagnostics]
---
> Hub: [[COWORK]]


> Table terms: [[polymarket_table_dictionary]]

# Block A1.1 Segment And L2 Proxy Diagnostics

## Read

I agree with the A1.1 direction: run segmentation, side-aware cost QA, and OFI component checks before approving A2. The L2 work should start in A1.1 because A0/A0b already contain maintained top-5 depth snapshots. This sidecar uses those snapshots to test whether the A1 signal is a touch-only effect, a broader top-5 depth/imbalance effect, or a spread/depth/regime artifact. It does not replace true multi-level OFI replay; that remains part of A2 or a deeper A1.1 follow-up.

## Outputs

- `data/analysis/csv_outputs/dali/block_a11_segment_surface.csv`: segment x horizon x absolute-OFI decile diagnostics.
- `data/analysis/csv_outputs/dali/block_a11_ofi_component_sweep.csv`: combined, bid-only, ask-only, instant-depth, top-5 pressure, L1 imbalance, and top-5 imbalance sweep.
- `data/analysis/block_a11_plots/`: heatmaps by family, market, spread bucket, relative depth bucket, run, resolution status, time-to-resolution, and component.

## What L2 Means Here

Current A1 canonical OFI is L1/top-of-book OFI computed from an L2-maintained executable book. A1.1 adds L2 proxies available in `block_a1_features.parquet`: top-5 bid/ask shares, top-5 notional, and top-5 book imbalance. The `top5_depth_pressure_mean_depth` component is a rolling change in top-5 bid depth minus rolling change in top-5 ask depth, normalized by each market's mean touch depth. Because the A1 parquet does not persist every per-level previous/new pair, this is not full MLOFI.

## Component Sweep

| component | h | hit | dir ret | n | top family | share |
| --- | --- | --- | --- | --- | --- | --- |
| tob_imbalance_level | 300 | 71.6% | 25.3 bps | 304,797 | geopolitics_policy | 37.5% |
| tob_imbalance_level | 1 | 71.3% | 23.1 bps | 308,231 | daily_crypto_up_down | 38.9% |
| tob_imbalance_level | 10 | 71.2% | 41.1 bps | 308,585 | daily_crypto_up_down | 32.2% |
| tob_imbalance_level | 5 | 70.7% | 30.9 bps | 304,208 | daily_crypto_up_down | 35.3% |
| top5_depth_pressure_mean_depth | 1 | 68.7% | 72.7 bps | 222,283 | daily_crypto_up_down | 49.2% |
| tob_imbalance_level | 30 | 67.6% | 49.4 bps | 308,260 | geopolitics_policy | 35.1% |
| ofi_combined_mean_depth | 10 | 65.4% | 122.8 bps | 180,602 | daily_crypto_up_down | 51.8% |
| ofi_combined_mean_depth | 5 | 64.1% | 94.9 bps | 156,638 | daily_crypto_up_down | 50.3% |
| ofi_ask_mean_depth | 10 | 63.1% | 127.4 bps | 128,524 | daily_crypto_up_down | 49.2% |
| ofi_bid_mean_depth | 10 | 63.1% | 127.5 bps | 128,700 | daily_crypto_up_down | 49.2% |
| ofi_ask_mean_depth | 5 | 62.5% | 97.9 bps | 109,609 | crypto_4h_up_down | 49.1% |
| ofi_bid_mean_depth | 5 | 62.3% | 97.7 bps | 109,361 | crypto_4h_up_down | 49.0% |
| top5_imbalance_level | 300 | 62.3% | -46.6 bps | 304,797 | ai_product | 62.0% |
| top5_depth_pressure_mean_depth | 10 | 61.1% | 105.7 bps | 258,905 | daily_crypto_up_down | 44.4% |
| ofi_combined_instant_depth | 300 | 59.9% | 367.6 bps | 275,587 | daily_crypto_up_down | 53.1% |
| top5_depth_pressure_mean_depth | 5 | 59.8% | 50.3 bps | 246,119 | daily_crypto_up_down | 46.2% |
| top5_depth_pressure_mean_depth | 30 | 58.9% | 136.1 bps | 280,120 | daily_crypto_up_down | 42.2% |
| top5_imbalance_level | 30 | 57.3% | 7.3 bps | 308,049 | ai_product | 64.6% |

## Segment Snapshots

### Family

| family | 5s hit | 5s dir ret | n | spread bps | rel depth |
| --- | --- | --- | --- | --- | --- |
| daily_crypto_up_down | 71.4% | 158.4 bps | 78,828 | 694.6 | 1.69 |
| crypto_4h_up_down | 62.3% | 44.1 bps | 67,214 | 1149.7 | 1.52 |
| geopolitics_policy | 55.2% | 3.8 bps | 1,512 | 457.1 | 2.04 |
| ai_product | 42.4% | -65.6 bps | 9,071 | 1927.3 | 1.52 |

### Market

| market | 5s hit | 5s dir ret | n | spread bps | rel depth |
| --- | --- | --- | --- | --- | --- |
| bitcoin-up-or-down-on-may-28-2026 | 74.5% | 180.6 bps | 47,898 | 411.5 | 1.80 |
| ethereum-up-or-down-on-may-28-2026 | 68.5% | 124.1 bps | 30,930 | 1132.9 | 1.51 |
| will-anthropic-have-the-best-ai-model-at-the-end-of-june-2026 | 68.3% | 1.4 bps | 1,498 | 125.7 | 3.74 |
| btc-updown-4h-1779926400 | 63.3% | 37.8 bps | 30,071 | 1367.2 | 1.63 |
| btc-updown-4h-1779912000 | 61.6% | 187.6 bps | 8,669 | 869.0 | 1.32 |
| btc-updown-4h-1779940800 | 61.4% | 7.2 bps | 28,474 | 1005.5 | 1.46 |
| us-iran-nuclear-deal-before-2027 | 59.8% | 4.5 bps | 1,341 | 491.6 | 2.08 |
| will-openai-announce-earbuds-or-headphones-in-2026 | 39.4% | -79.7 bps | 7,481 | 2302.9 | 1.09 |
| strait-of-hormuz-traffic-returns-to-normal-by-end-of-june | n/a | 0.0 bps | 127 | 215.3 | 1.87 |

### Spread Bucket

| spread_bucket | 5s hit | 5s dir ret | n | spread bps | rel depth |
| --- | --- | --- | --- | --- | --- |
| spread_q1_tight | 70.8% | 215.4 bps | 19,802 | 90.3 | 1.87 |
| spread_q2 | 70.2% | 91.1 bps | 29,222 | 153.4 | 1.72 |
| spread_q3 | 66.2% | 66.5 bps | 44,397 | 470.9 | 1.50 |
| spread_q4_wide | 60.3% | 78.9 bps | 63,217 | 1946.2 | 1.55 |

### Relative Depth Bucket

| relative_depth_bucket | 5s hit | 5s dir ret | n | spread bps | rel depth |
| --- | --- | --- | --- | --- | --- |
| depth_q4_deep | 69.0% | 127.9 bps | 85,040 | 972.0 | 2.61 |
| depth_q2 | 61.2% | 84.7 bps | 16,146 | 829.2 | 0.40 |
| depth_q3 | 59.5% | 86.4 bps | 22,395 | 1264.8 | 0.84 |
| depth_q1_shallow | 58.1% | 20.9 bps | 33,057 | 781.6 | 0.14 |

### Run

| run_id | 5s hit | 5s dir ret | n | spread bps | rel depth |
| --- | --- | --- | --- | --- | --- |
| a0b | 65.8% | 105.7 bps | 146,169 | 903.5 | 1.61 |
| a0 | 43.5% | -56.2 bps | 10,469 | 1733.8 | 1.60 |

### Resolved In Capture

| resolved_in_capture | 5s hit | 5s dir ret | n | spread bps | rel depth |
| --- | --- | --- | --- | --- | --- |
| resolved | 65.0% | 53.3 bps | 10,189 | 1293.4 | 1.86 |
| unresolved | 64.0% | 97.8 bps | 146,449 | 935.7 | 1.59 |

### Time To Resolution

| time_to_resolution_bucket | 5s hit | 5s dir ret | n | spread bps | rel depth |
| --- | --- | --- | --- | --- | --- |
| resolved_5_30m | 67.0% | 43.4 bps | 4,071 | 1619.9 | 2.26 |
| resolved_30m_2h | 64.1% | 59.9 bps | 6,118 | 1076.1 | 1.59 |
| unresolved | 64.0% | 97.8 bps | 146,449 | 935.7 | 1.59 |

## Heatmaps

![](data/analysis/block_a11_plots/block_a11_family_top_decile_hit_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_family_top_decile_n_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_market_top_decile_hit_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_market_top_decile_n_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_spread_bucket_top_decile_hit_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_spread_bucket_top_decile_n_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_relative_depth_bucket_top_decile_hit_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_relative_depth_bucket_top_decile_n_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_run_id_top_decile_hit_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_run_id_top_decile_n_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_resolved_in_capture_top_decile_hit_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_resolved_in_capture_top_decile_n_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_time_to_resolution_bucket_top_decile_hit_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_time_to_resolution_bucket_top_decile_n_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_component_top_decile_hit_heatmap.png)
![](data/analysis/block_a11_plots/block_a11_component_top_decile_directional_return_heatmap.png)

## A2 Plan

1. Capture design: run a 1-2 week VPS capture with active-market reselection, retaining raw `book`, `price_change`, `last_trade_price`, lifecycle, and telemetry messages unchanged. Keep crypto 4h up/down, separate pre-game from in-game sports, include fee-free geopolitics, and add a targeted AI/product sleeve selected by live trade-rate.
2. L2 feature design: replay raw JSONL into a new A2 feature table with per-level OFI columns `bid_ofi_l1..l10`, `ask_ofi_l1..l10`, `combined_ofi_l1..l10`, plus depth-weighted, exponentially weighted, and integrated/PCA-style OFI variants. Keep the current L1 CKS OFI as the baseline.
3. Cost design: replace the old mid-cost overlay with executable scenarios: enter at ask and exit at bid, enter at ask and mark exit at mid, inventory-reduction/sell-at-bid diagnostics, and paired YES/NO complement routes for clean binary markets.
4. Segment design: preserve the A1.1 segment surface schema so A2 can answer which families, spread/depth regimes, and resolution windows actually carry signal after executable costs.
5. Gate: approve capture budget only after A1.1 says whether crypto 5s/10s alpha survives executable entry/exit, whether 300s low-OFI is composition leakage, whether ask-only/top-5 components improve cost-adjusted signal, and which families deserve the panel slots.

Recommended next action for Justin: review the A1.1 segment/component heatmaps, then choose whether to run the deeper raw MLOFI replay before provisioning A2.
