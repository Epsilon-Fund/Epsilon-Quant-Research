---
title: Block K2 v2 Defensive Maker Findings
created: 2026-06-05
status: closed
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
  - strat_market_making
tags:
  - market-making
  - block-k
  - defensive-maker
  - binance-anchor
  - research
---

# Block K2 v2 Defensive Maker Findings

> **Strat:** [[strat_market_making]] (Market-Making). Sibling: [[strat_options_delta]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

Generated: 2026-05-31T11:26:36Z

## Summary

K2 v2 checks whether Binance-anchored defensive maker rules can rescue the crypto maker branch. No bucket clears zero after costs, and the pull/widen defense trigger appears in less than 0.1% of simulated fills. The note keeps the maker thesis closed for this crypto universe.

## Headline

No Binance-anchored defensive maker bucket clears zero after costs; the maker thesis remains closed.

Best full-sample config: `base100_L1_m1_no_defense`, n=1669, mean -4316.2 bps, CI [-4845.2 bps, -3762.9 bps], defense trigger 0.0%.

Best bucket: `near_absz_lt0.25|early_gt2h` under `base750_L10_m1_no_defense`, n=23, mean -224.3 bps, CI [-507.8 bps, 56.2 bps].

Bucket tests: 2816; raw CI-positive buckets: 0; robust clears with n>=30: 0.

Defense opportunity was scarce: 32 fills triggered the pull/widen rule across the whole grid (<0.1% of simulated fills), the largest per-config trigger rate was 0.1%, and the largest observed pre-fill adverse fair move was 0.10c. Config labels round the smallest toxicity band to `tox0bp`; use the numeric `tox_band_prob` column in the CSV for exact thresholds.

## Best Configs

| config | fills | mkts | mean | 95% CI | win | defense | spread | rebate | exit_fee |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base100_L1_m1_no_defense | 1669 | 25 | -4316.2 bps | [-4845.2 bps, -3762.9 bps] | 8.0% | 0.0% | 607.4 bps | 73.5 bps | 540.4 bps |
| base100_L1_m1_widen_tox0bp | 1669 | 25 | -4316.2 bps | [-4845.2 bps, -3762.9 bps] | 8.0% | 0.1% | 607.4 bps | 73.5 bps | 540.4 bps |
| base100_L1_m1_widen_tox1bp | 1669 | 25 | -4316.2 bps | [-4845.2 bps, -3762.9 bps] | 8.0% | 0.1% | 607.4 bps | 73.5 bps | 540.4 bps |
| base100_L1_m1_widen_tox2bp | 1669 | 25 | -4316.2 bps | [-4845.2 bps, -3762.9 bps] | 8.0% | 0.1% | 607.4 bps | 73.5 bps | 540.4 bps |
| base100_L1_m1_widen_tox5bp | 1669 | 25 | -4316.2 bps | [-4845.2 bps, -3762.9 bps] | 8.0% | 0.1% | 607.4 bps | 73.5 bps | 540.4 bps |
| base100_L1_m1_pull_tox10bp | 1669 | 25 | -4316.2 bps | [-4845.2 bps, -3762.9 bps] | 8.0% | 0.0% | 607.4 bps | 73.5 bps | 540.4 bps |
| base100_L1_m1_widen_tox10bp | 1669 | 25 | -4316.2 bps | [-4845.2 bps, -3762.9 bps] | 8.0% | 0.0% | 607.4 bps | 73.5 bps | 540.4 bps |
| base100_L5_m1_no_defense | 1631 | 25 | -4302.3 bps | [-4875.7 bps, -3717.8 bps] | 8.3% | 0.0% | 673.3 bps | 73.1 bps | 540.0 bps |
| base100_L5_m1_pull_tox0bp | 1631 | 25 | -4302.3 bps | [-4875.7 bps, -3717.8 bps] | 8.3% | 0.0% | 673.3 bps | 73.1 bps | 540.0 bps |
| base100_L5_m1_widen_tox0bp | 1631 | 25 | -4302.3 bps | [-4875.7 bps, -3717.8 bps] | 8.3% | 0.0% | 673.3 bps | 73.1 bps | 540.0 bps |

## Best Buckets

| config | |z| | tau | fills | mean | 95% CI | clears |
| --- | --- | --- | --- | --- | --- | --- |
| base750_L10_m1_no_defense | near_absz_lt0.25 | early_gt2h | 23 | -224.3 bps | [-507.8 bps, 56.2 bps] | no |
| base750_L10_m1_pull_tox0bp | near_absz_lt0.25 | early_gt2h | 23 | -224.3 bps | [-507.8 bps, 56.2 bps] | no |
| base750_L10_m1_pull_tox10bp | near_absz_lt0.25 | early_gt2h | 23 | -224.3 bps | [-507.8 bps, 56.2 bps] | no |
| base750_L10_m1_pull_tox1bp | near_absz_lt0.25 | early_gt2h | 23 | -224.3 bps | [-507.8 bps, 56.2 bps] | no |
| base750_L10_m1_pull_tox2bp | near_absz_lt0.25 | early_gt2h | 23 | -224.3 bps | [-507.8 bps, 56.2 bps] | no |
| base750_L10_m1_pull_tox5bp | near_absz_lt0.25 | early_gt2h | 23 | -224.3 bps | [-507.8 bps, 56.2 bps] | no |
| base750_L10_m1_widen_tox0bp | near_absz_lt0.25 | early_gt2h | 23 | -224.3 bps | [-507.8 bps, 56.2 bps] | no |
| base750_L10_m1_widen_tox10bp | near_absz_lt0.25 | early_gt2h | 23 | -224.3 bps | [-507.8 bps, 56.2 bps] | no |
| base750_L10_m1_widen_tox1bp | near_absz_lt0.25 | early_gt2h | 23 | -224.3 bps | [-507.8 bps, 56.2 bps] | no |
| base750_L10_m1_widen_tox2bp | near_absz_lt0.25 | early_gt2h | 23 | -224.3 bps | [-507.8 bps, 56.2 bps] | no |
| base750_L10_m1_widen_tox5bp | near_absz_lt0.25 | early_gt2h | 23 | -224.3 bps | [-507.8 bps, 56.2 bps] | no |
| base750_L10_m2_no_defense | near_absz_lt0.25 | early_gt2h | 23 | -224.3 bps | [-507.8 bps, 56.2 bps] | no |

## Family Split

Selected best config only.

| family | fills | markets | mean | 95% CI | rebate | exit_fee |
| --- | --- | --- | --- | --- | --- | --- |
| crypto_4h_up_down | 1368 | 23 | -3770.4 bps | [-4412.5 bps, -3184.1 bps] | 72.7 bps | 506.3 bps |
| daily_crypto_up_down | 301 | 2 | -6796.6 bps | [-8268.5 bps, -5253.2 bps] | 76.8 bps | 695.3 bps |

## Method

- Universe: A1 feature panel, pooled `a0b+a0c_roll`, families `crypto_4h_up_down` and `daily_crypto_up_down`; no holdout.
- Reservation price: Binance-implied European digital fair `N(z)`, not Polymarket mid.
- 4h surface: reused `data/analysis/cache/k3v3h_panel_features.parquet`.
- Daily surface: rebuilt from Binance 1s spot candles using Gamma `eventStartTime/endDate`; cache `data/analysis/cache/k2v2_daily_model_surface.parquet`.
- Defensive rule: if Binance token fair at `fill_time - reaction_latency` moves away from the resting side by more than `tox_band`, either pull the quote or widen that side before the Polymarket taker print.
- Fill proxy: A1.4h style, one-share passive fill only when a real taker print crosses the modeled quote within 5s of the quote state.
- Exit: taker after 60s or before `window_end - 60s`; entry maker rebate and exit taker fee use the K1 crypto fee/rebate table.
- Hard flatten: no quotes in `abs_z < 0.25` and `tau <= 30m`.
- CI: 500 bootstrap samples over fill-time market blocks.

## Diagnostics

- Quote states with model: 3,617,046
- Candidate taker prints with fresh prior quote: 9,631
- Simulated fills across all configs: 469,283
- Daily markets modeled: 2

## Outputs

- Summary CSV: `data/analysis/csv_outputs/market_making/k2v2_defensive_maker.csv`
- Fill ledger: `data/analysis/k2v2_defensive_maker_fills.parquet`
- Repro script: `scripts/dali_block_k2v2_defensive_maker.py`
