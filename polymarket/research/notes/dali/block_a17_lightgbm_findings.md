---
tags: [dali, block-a17, lightgbm, results]
---

# Block A1.7 LightGBM Findings

> Table terms: [[polymarket_table_dictionary]]

## Headline

No: no market-threshold cell beat both rule-based baselines with CI lower bound above zero.

A1.7 used a single pooled LightGBM classifier across all `primary_read` markets, with `market_id` as a categorical feature and walk-forward per-market splits of 2/3 train, 1/6 validation, and 1/6 test. Hyperparameters were fixed at `max_depth=6`, `num_leaves=31`, `learning_rate=0.05`, `n_estimators=200`, and `early_stopping_rounds=20`; no Optuna or random shuffling was used. Test-set deployment used only `model_prob > P` for P in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8), a 5s hold, touch round-trip entry and exit, taker fees on both legs, and one non-overlapping position per market.

## Per-Market Verdict

| market | slug | best P | test events | eligible | trades | ML mean | ML CI | TOB base | OFI base | delta TOB | delta OFI | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0:558934 | will-spain-win-the-2026-fifa-world-cu. | 0.55 | 2 | 2 | 0 | n/a | [n/a, n/a] | n/a | n/a | n/a | n/a | insufficient_test_data |
| a0:558936 | will-france-win-the-2026-fifa-world-c. | 0.55 | 4 | 4 | 0 | n/a | [n/a, n/a] | n/a | n/a | n/a | n/a | insufficient_test_data |
| a0:665325 | us-iran-nuclear-deal-before-2027 | 0.55 | 2,492 | 68 | 7 | -261.3 bps | [-281.7 bps, -138.9 bps] | -239.9 bps | -263.0 bps | -21.4 bps | 1.7 bps | no executable edge |
| a0b:1971905 | strait-of-hormuz-traffic-returns-to-n. | 0.55 | 89 | 29 | 0 | n/a | [n/a, n/a] | n/a | n/a | n/a | n/a | insufficient_test_data |
| a0b:2327929 | nba-okc-sas-2026-05-28 | 0.55 | 2 | 2 | 0 | n/a | [n/a, n/a] | n/a | n/a | n/a | n/a | insufficient_test_data |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 0.60 | 14,551 | 2,310 | 84 | -551.6 bps | [-708.3 bps, -418.1 bps] | -2356.3 bps | -2312.8 bps | 1804.7 bps | 1761.1 bps | no executable edge |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | 0.70 | 23,363 | 130 | 28 | -536.3 bps | [-1207.0 bps, -147.4 bps] | -139.4 bps | -238.5 bps | -397.0 bps | -297.8 bps | no executable edge |
| a0b:2364426 | btc-updown-4h-1779912000 | 0.80 | 3,298 | 2 | 1 | -1397.2 bps | [n/a, n/a] | -5007.9 bps | -8319.3 bps | 3610.7 bps | 6922.1 bps | no executable edge |
| a0b:2366225 | btc-updown-4h-1779926400 | 0.70 | 14,356 | 282 | 21 | -807.0 bps | [-1082.1 bps, -571.0 bps] | -927.1 bps | -2111.7 bps | 120.0 bps | 1304.6 bps | no executable edge |
| a0b:2367777 | btc-updown-4h-1779940800 | 0.80 | 14,136 | 9 | 1 | -418.2 bps | [n/a, n/a] | -1887.5 bps | -2604.0 bps | 1469.3 bps | 2185.7 bps | no executable edge |
| a0b:566136 | a0b:566136 | 0.55 | 0 | 0 | 0 | n/a | [n/a, n/a] | n/a | n/a | n/a | n/a | insufficient_test_data |

## Feature Importance

| feature | gain | split count |
| --- | --- | --- |
| tob_imbalance_level_instant | 145743.2 | 190 |
| ofi_5s | 134004.9 | 141 |
| market_id | 32869.6 | 76 |
| depth_relative | 27758.4 | 78 |
| depth_at_touch | 27289.6 | 75 |
| spread_bps_instant | 22805.1 | 59 |
| tob_imbalance_level_5s_mean | 20096.4 | 53 |
| spread_bps_5s_mean | 17024.2 | 48 |

## Probability Calibration Check

| pred prob bin | n | mean predicted | actual hit | mean abs OFI |
| --- | --- | --- | --- | --- |
| <0.50 | 23,635 | 43.7% | 40.4% | 444.61 |
| 0.50-0.55 | 17,380 | 52.6% | 52.6% | 290.64 |
| 0.55-0.60 | 15,364 | 57.5% | 55.7% | 479.94 |
| 0.60-0.65 | 8,580 | 62.3% | 64.9% | 615.61 |
| 0.65-0.70 | 4,941 | 67.1% | 69.5% | 1099.54 |
| 0.70-0.75 | 773 | 72.5% | 56.3% | 792.03 |
| 0.75-0.80 | 1,537 | 78.0% | 64.5% | 785.72 |
| >=0.80 | 83 | 81.0% | 69.9% | 3404.49 |

The calibration table is computed on pooled test-set events only. `actual hit` is the realized `direction_correct` rate, so threshold interpretation is meaningful only where the bin has enough samples and actual hit rate increases with predicted probability.

## Recommendation

no Tier 2 edge found.
