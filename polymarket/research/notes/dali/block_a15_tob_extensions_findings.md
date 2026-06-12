---
tags: [dali, block-a15, tob-extensions, results]
---
> Hub: [[COWORK]]


## Summary

- Scope: Block A1.5 TOB Extensions Findings in the Dali research lineage area.
- Existing takeaway/status: Best multi-level variant is `exp_decay_imbalance_alpha_0p5` at 300s, 74.3%, +4.92 pp vs L1 TOB. TOB survives the micro-price target check; hit-rate deltas are not a material collapse. Best micro-change cell is 5s, 54.2%; it does not beat A1.3 TOB where TOB is strongest. Signal characterization only: no execution simulation, no ML, and no parameter optimization.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
> Table terms: [[polymarket_table_dictionary]]

# Block A1.5 TOB Extensions Findings

## Headline

Best multi-level variant is `exp_decay_imbalance_alpha_0p5` at 300s, 74.3%, +4.92 pp vs L1 TOB. TOB survives the micro-price target check; hit-rate deltas are not a material collapse. Best micro-change cell is 5s, 54.2%; it does not beat A1.3 TOB where TOB is strongest. Signal characterization only: no execution simulation, no ML, and no parameter optimization.

## Calibration Gate

The A1.3 current-level TOB decile aggregate was recomputed on the A1.5 code path from the A1.2 per-level parquet. The gate requires zero top-decile hit-rate delta and zero directional-return delta at all four horizons.

| h | A15 n | A13 n | hit delta | dir delta | pass |
| --- | --- | --- | --- | --- | --- |
| 1 | 299,898 | 299,898 | +0.00 pp | 0.0 bps | yes |
| 5 | 299,864 | 299,864 | +0.00 pp | 0.0 bps | yes |
| 30 | 299,676 | 299,676 | +0.00 pp | 0.0 bps | yes |
| 300 | 299,466 | 299,466 | +0.00 pp | 0.0 bps | yes |

## Sub-Experiment 1: Multi-Level Imbalance Signals

Per-level imbalance is `(bid_size_k - ask_size_k) / (bid_size_k + ask_size_k)` with missing side sizes treated as zero and both-missing levels set to zero. The TOB baseline uses A1.3's stricter L1 convention for exact reconciliation.

| variant | h | hit | CI | vs L1 TOB | dir ret | n | top family | share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tob_current_level | 1 | 70.6% | [61.8%, 78.7%] | +0.00 pp | 32.7 bps | 299,898 | daily_crypto_up_down | 46.8% |
| exp_decay_imbalance_alpha_0p5 | 1 | 60.5% | [54.9%, 66.9%] | -10.11 pp | 11.1 bps | 309,021 | geopolitics_policy | 28.0% |
| exp_decay_imbalance_alpha_0p3 | 1 | 55.5% | [50.3%, 61.7%] | -15.16 pp | 7.0 bps | 309,021 | geopolitics_policy | 27.1% |
| depth_weighted_imbalance_l1_l10 | 1 | 50.7% | [47.1%, 54.8%] | -19.96 pp | 1.8 bps | 309,021 | geopolitics_policy | 33.4% |
| exp_decay_imbalance_alpha_0p1 | 1 | 46.8% | [41.7%, 52.2%] | -23.80 pp | -3.7 bps | 309,021 | geopolitics_policy | 27.4% |
| integrated_imbalance_l1_l10 | 1 | 40.2% | [36.0%, 44.5%] | -30.39 pp | -11.6 bps | 309,020 | geopolitics_policy | 24.7% |
| tob_current_level | 5 | 73.7% | [68.4%, 78.7%] | +0.00 pp | 72.9 bps | 299,864 | daily_crypto_up_down | 46.8% |
| exp_decay_imbalance_alpha_0p5 | 5 | 64.3% | [60.3%, 68.1%] | -9.39 pp | 24.2 bps | 308,872 | geopolitics_policy | 28.0% |
| exp_decay_imbalance_alpha_0p3 | 5 | 58.6% | [53.7%, 63.0%] | -15.10 pp | 17.2 bps | 308,801 | geopolitics_policy | 27.1% |
| depth_weighted_imbalance_l1_l10 | 5 | 50.1% | [47.3%, 54.0%] | -23.59 pp | -4.6 bps | 308,879 | geopolitics_policy | 33.4% |
| exp_decay_imbalance_alpha_0p1 | 5 | 48.8% | [45.0%, 53.9%] | -24.92 pp | -8.6 bps | 308,856 | geopolitics_policy | 27.4% |
| integrated_imbalance_l1_l10 | 5 | 43.3% | [39.2%, 47.1%] | -30.45 pp | -18.5 bps | 308,879 | geopolitics_policy | 24.7% |
| tob_current_level | 30 | 74.1% | [69.4%, 78.6%] | +0.00 pp | 146.6 bps | 299,676 | daily_crypto_up_down | 46.9% |
| exp_decay_imbalance_alpha_0p5 | 30 | 68.8% | [64.2%, 72.9%] | -5.31 pp | 63.9 bps | 308,276 | geopolitics_policy | 28.0% |
| exp_decay_imbalance_alpha_0p3 | 30 | 61.7% | [56.5%, 67.4%] | -12.48 pp | 43.1 bps | 308,133 | geopolitics_policy | 27.1% |
| exp_decay_imbalance_alpha_0p1 | 30 | 51.7% | [47.6%, 56.3%] | -22.47 pp | -14.1 bps | 308,254 | geopolitics_policy | 27.4% |
| depth_weighted_imbalance_l1_l10 | 30 | 50.4% | [44.4%, 56.5%] | -23.74 pp | 12.4 bps | 308,271 | geopolitics_policy | 33.3% |
| integrated_imbalance_l1_l10 | 30 | 45.8% | [41.7%, 50.8%] | -28.29 pp | -36.7 bps | 308,274 | geopolitics_policy | 24.7% |
| exp_decay_imbalance_alpha_0p5 | 300 | 74.3% | [67.4%, 80.9%] | +4.92 pp | 181.0 bps | 304,796 | geopolitics_policy | 28.2% |
| tob_current_level | 300 | 69.4% | [64.3%, 74.7%] | +0.00 pp | 356.8 bps | 299,466 | daily_crypto_up_down | 46.9% |
| exp_decay_imbalance_alpha_0p3 | 300 | 66.8% | [57.3%, 74.5%] | -2.60 pp | 148.8 bps | 303,998 | geopolitics_policy | 27.4% |
| exp_decay_imbalance_alpha_0p1 | 300 | 54.1% | [45.8%, 63.2%] | -15.28 pp | 10.0 bps | 304,796 | geopolitics_policy | 27.6% |
| depth_weighted_imbalance_l1_l10 | 300 | 53.5% | [43.6%, 65.9%] | -15.94 pp | 74.4 bps | 304,773 | geopolitics_policy | 33.5% |
| integrated_imbalance_l1_l10 | 300 | 50.1% | [41.5%, 56.9%] | -19.33 pp | -24.9 bps | 304,796 | geopolitics_policy | 25.0% |

Verdict: multi-level imbalance does not beat L1 TOB at 1s, 5s, or 30s. The only extension that beats L1 is `exp_decay_imbalance_alpha_0p5` at 300s, where it hits 74.3% versus L1's 69.4%, but that is the horizon already most exposed to composition and persistence effects.

## Sub-Experiment 2: Micro-Price Target

Micro-price is `mid + 0.5 * spread * tob_imbalance_level`. The micro-price target replaces `future_directional_mid` with `future_directional_micro_price` while keeping the current directional mid denominator, matching the prompt's target substitution.

| h | mid hit | micro hit | delta | mid dir | micro dir | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 70.6% | 97.3% | +26.68 pp | 32.7 bps | 333.6 bps | 299,898 |
| 5 | 73.7% | 95.3% | +21.60 pp | 72.9 bps | 340.0 bps | 299,864 |
| 30 | 74.1% | 90.0% | +15.82 pp | 146.6 bps | 363.0 bps | 299,676 |
| 300 | 69.4% | 76.9% | +7.49 pp | 356.8 bps | 516.1 bps | 299,466 |

Verdict: the TOB signal does not collapse when the future target is micro-price. It gets stronger mechanically because future micro-price still contains book imbalance; treat this as a robustness/audit target, not a tradability result.

## Sub-Experiment 3: Micro-Price Change Signal

`micro_change_hs` is `direction_factor * (micro_price_t - micro_price_{t-h}) / mean_depth_at_touch`, binned by global absolute signal deciles within each horizon.

| h | micro-change hit | CI | vs TOB | vs L1 OFI | micro dir | TOB dir | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 49.9% | [47.5%, 52.6%] | -20.71 pp | -6.45 pp | -36.4 bps | 32.7 bps | 123,029 |
| 5 | 54.2% | [52.3%, 56.7%] | -19.54 pp | -9.91 pp | 5.8 bps | 72.9 bps | 155,235 |
| 30 | 53.2% | [50.0%, 56.3%] | -20.95 pp | -3.78 pp | -11.6 bps | 146.6 bps | 206,277 |
| 300 | 44.1% | [36.6%, 52.2%] | -25.34 pp | -7.78 pp | -362.0 bps | 356.8 bps | 270,989 |

Verdict: micro-price change is not competitive with L1 TOB or L1 OFI. Its best top-decile hit rate is 54.2% at 5s, far below TOB's 73.7% at the same horizon.

## Outputs

- `data/analysis/block_a15_features.parquet`
- `data/analysis/csv_outputs/dali/block_a15_imbalance_variants_decile.csv`
- `data/analysis/csv_outputs/dali/block_a15_microprice_target_comparison.csv`
- `data/analysis/csv_outputs/dali/block_a15_microprice_signal_decile.csv`
- `data/analysis/csv_outputs/dali/block_a15_baseline_check.csv`

Recommended next action for Justin: carry L1 `tob_imbalance_level` and `exp_decay_imbalance_alpha_0p5` into A2, with L1 as the primary 1s/5s/30s feature and exp-decay imbalance as a 300s sidecar; keep micro-price as an audit target and do not promote micro-price-change as a primary signal yet.
