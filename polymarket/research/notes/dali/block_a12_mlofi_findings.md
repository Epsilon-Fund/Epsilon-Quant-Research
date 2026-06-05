---
tags: [dali, block-a12, mlofi, results]
---
> Hub: [[COWORK]]


> Table terms: [[polymarket_table_dictionary]]

# Block A1.2 MLOFI Findings

## Headline

A1.2 replayed the existing A0/A0b raw captures into true top-10 per-level OFI and compared MLOFI variants against the current L1 CKS baseline. The load-bearing result is that MLOFI does not beat L1 where A1's signal is cleanest: L1 wins 1s, 5s, and 30s top-decile hit rate, including the 5s headline cell. Best MLOFI-vs-L1 hit-rate deltas by horizon: 1s exp_decay_alpha_0p5 -19.35pp; 5s exp_decay_alpha_0p5 -15.57pp; 30s exp_decay_alpha_0p5 -5.46pp; 300s depth_weighted_l1_l10 +1.27pp. The only positive MLOFI delta is the 300s depth-weighted cell, which is modest and sits in the horizon A1 already treated as composition-sensitive. Treat this as a feature-family sniff test, not optimization or a deployment result.

## Outputs

- `data/analysis/block_a12_mlofi_features.parquet`: per-event top-10 MLOFI feature sidecar.
- `data/analysis/csv_outputs/dali/block_a12_mlofi_decile_aggregate.csv`: variant x horizon x absolute-signal decile aggregate.
- `data/analysis/csv_outputs/dali/block_a12_mlofi_market_panel.csv`: per-market x horizon x variant panel with A1 reporting guards.
- `data/analysis/csv_outputs/dali/block_a12_mlofi_comparison.csv`: top-decile side-by-side table vs L1 baseline.
- `data/analysis/block_a12_plots/`: decile hit-rate plots and top-decile delta heatmap.

## Method

Per-level OFI compares previous and new `(price, size)` at each book rank `k=1..10` using the same CKS rules as A1 L1 OFI: bid price up contributes new bid size, bid price down contributes negative previous bid size, unchanged bid contributes size delta; ask price down contributes negative new ask size, ask price up contributes previous ask size, unchanged ask contributes previous minus new size. `combined_ofi_lk = bid_ofi_lk + ask_ofi_lk`.

The tested variants are:

- `l1_cks`: `combined_ofi_l1`, the A1 baseline.
- `integrated_l1_l10`: sum of `combined_ofi_l1..l10`.
- `depth_weighted_l1_l10`: sum of `combined_ofi_lk * level_depth_lk` divided by sum of `level_depth_lk`, using current bid+ask size at level `k`.
- `exp_decay_alpha_0p1`, `0p3`, `0p5`: sum of `combined_ofi_lk * exp(-alpha * (k - 1))`.

Each event-level variant is rolled over 1s, 5s, 30s, and 300s, flipped into YES/NO market direction with A1's `direction_factor`, then normalized by each market's mean touch depth. Deciles are global equal-count buckets within each `(variant, horizon)` based on absolute normalized signal magnitude; decile 10 is largest absolute signal, not most bullish.

Replay intentionally mirrors A1's shard-local anchoring: each JSONL shard starts from its first full `book` snapshot for an asset, then applies `price_change` updates. That is why the L1 baseline check below is the key comparability test.

## L1 Baseline Check

This is the A1.2 L1 replay minus the existing A1 decile CSV. It matches exactly at all four horizons, so the MLOFI sidecar is comparable to the A1 baseline.

| h | hit delta | dir ret delta |
| --- | --- | --- |
| 1 | +0.00 pp | 0.0 bps |
| 5 | +0.00 pp | 0.0 bps |
| 30 | +0.00 pp | 0.0 bps |
| 300 | +0.00 pp | 0.0 bps |

## Horizon Winners

| h | winner | hit | hit vs L1 | dir ret | dir ret vs L1 |
| --- | --- | --- | --- | --- | --- |
| 1 | l1_cks | 56.4% | +0.00 pp | 20.5 bps | 0.0 bps |
| 5 | l1_cks | 64.1% | +0.00 pp | 94.9 bps | 0.0 bps |
| 30 | l1_cks | 57.0% | +0.00 pp | 66.4 bps | 0.0 bps |
| 300 | depth_weighted_l1_l10 | 53.1% | +1.27 pp | 158.4 bps | 69.1 bps |

## Full Top-Decile Comparison

| variant | h | top hit | hit CI | top dir ret | pooled R2 | hit delta pp | dir delta | top n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| l1_cks | 1 | 56.4% | [51.4%, 61.7%] | 20.5 bps | 0.00024 | +0.00 | 0.0 bps | 122,872 |
| exp_decay_alpha_0p5 | 1 | 37.0% | [33.0%, 40.5%] | -54.6 bps | 0.00822 | -19.35 | -75.1 bps | 263,251 |
| exp_decay_alpha_0p3 | 1 | 32.8% | [29.1%, 36.4%] | -67.8 bps | 0.01215 | -23.58 | -88.3 bps | 262,875 |
| depth_weighted_l1_l10 | 1 | 31.9% | [28.7%, 35.6%] | -54.4 bps | 0.00008 | -24.47 | -74.9 bps | 274,773 |
| exp_decay_alpha_0p1 | 1 | 31.0% | [27.7%, 34.3%] | -68.6 bps | 0.01151 | -25.35 | -89.1 bps | 262,314 |
| integrated_l1_l10 | 1 | 30.5% | [27.3%, 33.6%] | -69.0 bps | 0.01011 | -25.81 | -89.5 bps | 253,824 |
| l1_cks | 5 | 64.1% | [59.8%, 68.6%] | 94.9 bps | 0.00779 | +0.00 | 0.0 bps | 156,638 |
| exp_decay_alpha_0p5 | 5 | 48.5% | [45.0%, 51.9%] | 3.8 bps | 0.00005 | -15.57 | -91.1 bps | 279,582 |
| exp_decay_alpha_0p3 | 5 | 44.3% | [41.0%, 47.8%] | -15.0 bps | 0.00197 | -19.79 | -109.9 bps | 279,426 |
| exp_decay_alpha_0p1 | 5 | 41.4% | [38.6%, 44.4%] | -26.9 bps | 0.00393 | -22.62 | -121.8 bps | 279,187 |
| integrated_l1_l10 | 5 | 40.6% | [37.6%, 43.3%] | -32.5 bps | 0.00401 | -23.52 | -127.4 bps | 273,282 |
| depth_weighted_l1_l10 | 5 | 39.3% | [36.0%, 42.5%] | -41.5 bps | 0.00003 | -24.72 | -136.4 bps | 286,571 |
| l1_cks | 30 | 57.0% | [51.8%, 63.2%] | 66.4 bps | 0.00005 | +0.00 | 0.0 bps | 218,378 |
| exp_decay_alpha_0p5 | 30 | 51.5% | [46.7%, 55.6%] | -27.1 bps | 0.00305 | -5.46 | -93.5 bps | 298,312 |
| exp_decay_alpha_0p3 | 30 | 48.0% | [44.2%, 52.3%] | -90.1 bps | 0.00455 | -8.92 | -156.5 bps | 297,902 |
| depth_weighted_l1_l10 | 30 | 45.9% | [41.1%, 51.5%] | -1.8 bps | 0.00595 | -11.07 | -68.2 bps | 301,032 |
| exp_decay_alpha_0p1 | 30 | 44.0% | [39.2%, 48.4%] | -151.9 bps | 0.00424 | -12.97 | -218.3 bps | 297,683 |
| integrated_l1_l10 | 30 | 42.2% | [38.0%, 46.2%] | -173.3 bps | 0.00351 | -14.74 | -239.7 bps | 295,042 |
| depth_weighted_l1_l10 | 300 | 53.1% | [43.3%, 63.8%] | 158.4 bps | 0.00897 | +1.27 | 69.1 bps | 304,139 |
| exp_decay_alpha_0p5 | 300 | 52.1% | [39.2%, 62.2%] | 138.6 bps | 0.00475 | +0.27 | 49.3 bps | 303,755 |
| exp_decay_alpha_0p1 | 300 | 52.1% | [41.1%, 60.3%] | 105.7 bps | 0.00215 | +0.25 | 16.5 bps | 303,248 |
| l1_cks | 300 | 51.8% | [40.4%, 62.7%] | 89.3 bps | 0.00309 | +0.00 | 0.0 bps | 275,591 |
| exp_decay_alpha_0p3 | 300 | 51.6% | [40.5%, 60.3%] | 72.8 bps | 0.00405 | -0.26 | -16.5 bps | 303,830 |
| integrated_l1_l10 | 300 | 50.5% | [39.9%, 59.2%] | 42.9 bps | 0.00132 | -1.32 | -46.4 bps | 303,616 |

## Reportable Per-Market Snapshot

Top reportable 5s rows across variants:

| run | market | variant | family | n class | label | hit | dir ret | top n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | nato-x-russia-military-clash-by-december-31-2026-244 | exp_decay_alpha_0p1 | geopolitics_policy | 32 | thin_wide_CI | 99.7% | 203.8 bps | 336 |
| a0 | nato-x-russia-military-clash-by-december-31-2026-244 | integrated_l1_l10 | geopolitics_policy | 32 | thin_wide_CI | 99.7% | 204.1 bps | 324 |
| a0 | nato-x-russia-military-clash-by-december-31-2026-244 | exp_decay_alpha_0p3 | geopolitics_policy | 32 | thin_wide_CI | 99.4% | 206.4 bps | 318 |
| a0b | strait-of-hormuz-traffic-returns-to-normal-by-july-31 | integrated_l1_l10 | geopolitics_policy | 131 | thin_wide_CI | 98.2% | 139.7 bps | 56 |
| a0 | nato-x-russia-military-clash-by-december-31-2026-244 | exp_decay_alpha_0p5 | geopolitics_policy | 32 | thin_wide_CI | 97.5% | 199.9 bps | 316 |
| a0b | strait-of-hormuz-traffic-returns-to-normal-by-july-31 | depth_weighted_l1_l10 | geopolitics_policy | 131 | thin_wide_CI | 96.8% | 131.2 bps | 62 |
| a0b | strait-of-hormuz-traffic-returns-to-normal-by-july-31 | exp_decay_alpha_0p1 | geopolitics_policy | 131 | thin_wide_CI | 96.5% | 135.8 bps | 57 |
| a0 | nato-x-russia-military-clash-by-december-31-2026-244 | depth_weighted_l1_l10 | geopolitics_policy | 32 | thin_wide_CI | 95.9% | 186.7 bps | 319 |
| a0b | strait-of-hormuz-traffic-returns-to-normal-by-july-31 | exp_decay_alpha_0p3 | geopolitics_policy | 131 | thin_wide_CI | 93.1% | 129.3 bps | 58 |
| a0b | strait-of-hormuz-traffic-returns-to-normal-by-july-31 | exp_decay_alpha_0p5 | geopolitics_policy | 131 | thin_wide_CI | 89.5% | 120.3 bps | 57 |
| a0b | bitcoin-up-or-down-on-may-28-2026 | l1_cks | daily_crypto_up_down | 1,096 | primary_read | 82.2% | 891.0 bps | 8,736 |
| a0b | strait-of-hormuz-traffic-returns-to-normal-by-july-31 | l1_cks | geopolitics_policy | 131 | thin_wide_CI | 80.4% | 104.2 bps | 46 |
| a0b | ethereum-up-or-down-on-may-28-2026 | depth_weighted_l1_l10 | daily_crypto_up_down | 248 | primary_read | 79.7% | 249.0 bps | 15,113 |
| a0b | ethereum-up-or-down-on-may-28-2026 | integrated_l1_l10 | daily_crypto_up_down | 248 | primary_read | 79.6% | 290.5 bps | 15,008 |
| a0 | us-iran-nuclear-deal-before-2027 | integrated_l1_l10 | geopolitics_policy | 206 | primary_read | 77.6% | 10.4 bps | 1,744 |
| a0b | ethereum-up-or-down-on-may-28-2026 | exp_decay_alpha_0p1 | daily_crypto_up_down | 248 | primary_read | 77.3% | 278.4 bps | 15,079 |
| a0b | bitcoin-up-or-down-on-may-28-2026 | exp_decay_alpha_0p5 | daily_crypto_up_down | 1,096 | primary_read | 75.2% | 648.9 bps | 9,285 |
| a0 | us-iran-nuclear-deal-before-2027 | depth_weighted_l1_l10 | geopolitics_policy | 206 | primary_read | 75.1% | 28.7 bps | 1,779 |

These are per-market diagnostics, not the headline. Some high-hit rows are `thin_wide_CI`; the pooled depth-normalized comparison above is the result to cite.

## Plots

![](data/analysis/block_a12_plots/block_a12_mlofi_decile_hit_rate_1s.png)
![](data/analysis/block_a12_plots/block_a12_mlofi_decile_hit_rate_5s.png)
![](data/analysis/block_a12_plots/block_a12_mlofi_decile_hit_rate_30s.png)
![](data/analysis/block_a12_plots/block_a12_mlofi_decile_hit_rate_300s.png)
![](data/analysis/block_a12_plots/block_a12_mlofi_top_decile_delta_heatmap.png)

## Interpretation

Current A0/A0b evidence says keep L1 CKS as the primary A2 OFI signal. Logging top-10 per-level OFI in A2 is still useful if storage/CPU cost is acceptable, because it preserves optionality for later family/regime work, but it should not replace or distract from the simpler L1 feature. The 300s depth-weighted improvement is not strong enough to drive design by itself.

Recommended next action for Justin: keep L1 CKS as the A2 headline signal, and log top-10 MLOFI as an optional sidecar only if capture resources are comfortable.
