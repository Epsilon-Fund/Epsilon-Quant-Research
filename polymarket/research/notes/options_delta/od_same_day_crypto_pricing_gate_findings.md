---
title: "Same-Day Crypto Touch And Terminal-Ladder Pricing Gate"
created: 2026-06-05
status: closed
owner: justin
project: polymarket
para: project
hubs:
  - strat_options_delta
  - COWORK
tags:
  - research
  - options-delta
---
# Same-Day Crypto Touch And Terminal-Ladder Pricing Gate

> Hub: [[strat_options_delta]] · [[COWORK]]
> Related: [[od_cross_asset_gate0_universe_map_findings]] · [[od_conditional_prob_calibration_findings]] · [[od_pricing_model_form_findings]] · [[od_strategy_a_v3_pnl_risk_findings]] · [[block_k4_arb_scan_findings]] · [[mm_deployable_cells_findings]] · [[block_k5_findings]] · [[block_k5_stress_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Summary

- Scope: Same-Day Crypto Touch And Terminal-Ladder Pricing Gate in the OD/options-delta area.
- Existing takeaway/status: Final verdicts after Arm T Tier-1 extension: **Arm T CLOSE** and **Arm E CLOSE**.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Headline

Final verdicts after Arm T Tier-1 extension: **Arm T CLOSE** and **Arm E CLOSE**.

Arm T Tier-1 reason: best registered held-out Tier-1 cell pos_z_ge_3/1_4h has 5% haircut EV 0.17c with CI [0.12c, 0.27c] and BH capacity q=0.0003; mean is below the pre-registered 0.25c materiality bar.

Arm E reason: terminal ladder power improves the sample, but no cell clears both net-EV lower CI and incremental-vs-structural lower CI.

This is the same-day daily crypto gate only. Multi-week/monthly `will hit X in <month>` barriers are out of scope. The fair values are external Binance 1m path/close truth tables; no PM ladder monotonicity or internal-consistency arb is used.

## Design And Pre-Registered Gates

Arm T contains same-day touch/running-extreme markets that resolve if Binance BTC/ETH/SOL reaches a level anytime in the day. Arm E contains same-day terminal above-X ladders that resolve on the Binance 1m close at the resolution timestamp. Ambiguous resolution text is quarantined.

The pre-registered Arm T pass bar is a defensible z/time-left cell with selected executable fills whose market-cluster bootstrap lower CI is above zero after PM taker fees and spread-crossing entry. The pre-registered Arm E pass bar is stricter: net-EV lower CI above zero **and** K5-haircut incremental-vs-structural lower CI above zero, matching the closed-line bar in [[od_conditional_prob_calibration_findings]] and [[od_pricing_model_form_findings]].

Touch example: `will-bitcoin-dip-to-67k-on-june-2` has level $67,000, spot $67,719, YES ask 67.10c, empirical touch 74.45c, and model edge 5.80c before future resolution.

Terminal example: `bitcoin-above-66k-on-june-3-2026` has strike $66,000, spot $67,719, YES ask 80.80c, empirical terminal 80.19c, and model edge -1.69c; it still must beat the structural baseline after capacity.

## Step 1: Same-Day Classification

| market_slug | asset | level | resolution_ts_utc | arm | resolution_class | decision_quote | volume_for_share_usd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bitcoin-above-68k-on-june-2-2026 | BTC | $68,000 | 2026-06-02T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for BTC/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $673,324 |
| bitcoin-above-66k-on-june-2-2026 | BTC | $66,000 | 2026-06-02T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for BTC/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $521,897 |
| bitcoin-above-72k-on-june-2-2026 | BTC | $72,000 | 2026-06-02T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for BTC/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $426,862 |
| bitcoin-above-70k-on-june-2-2026 | BTC | $70,000 | 2026-06-02T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for BTC/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $393,667 |
| bitcoin-above-74k-on-june-2-2026 | BTC | $74,000 | 2026-06-02T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for BTC/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $333,762 |
| ethereum-above-1900-on-june-2-2026 | ETH | $1,900 | 2026-06-02T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for ETH/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $191,266 |
| bitcoin-above-66k-on-june-3-2026 | BTC | $66,000 | 2026-06-03T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for BTC/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $151,268 |
| bitcoin-above-76k-on-june-2-2026 | BTC | $76,000 | 2026-06-02T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for BTC/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $141,155 |
| ethereum-above-2100-on-june-2-2026 | ETH | $2,100 | 2026-06-02T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for ETH/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $129,729 |
| bitcoin-above-76k-on-june-3-2026 | BTC | $76,000 | 2026-06-03T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for BTC/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $123,989 |
| will-bitcoin-reach-71k-on-june-2 | BTC | $71,000 | 2026-06-03T04:00:00+00:00 | T | barrier_touch | This market will immediately resolve to "Yes" if any Binance 1-minute candle for Bitcoin (BTC/USDT) on the date specified in the title, between 12:00 AM ET and 11:59 PM ET has a final "High" price equal to or greater than the price specified in the title. | $104,469 |
| bitcoin-above-68k-on-june-3-2026 | BTC | $68,000 | 2026-06-03T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for BTC/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $96,695 |
| bitcoin-above-78k-on-june-3-2026 | BTC | $78,000 | 2026-06-03T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for BTC/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $93,780 |
| will-bitcoin-dip-to-67k-on-june-2 | BTC | $67,000 | 2026-06-03T04:00:00+00:00 | T | barrier_touch | This market will immediately resolve to "Yes" if any Binance 1 minute candle for Bitcoin (BTC/USDT) on the date specified in the title, between 12:00 AM ET and 11:59 PM ET has a final "Low" price equal to or lower than the price specified in the title. | $89,905 |
| bitcoin-above-72k-on-june-3-2026 | BTC | $72,000 | 2026-06-03T16:00:00+00:00 | E | terminal_close | This market will resolve to "Yes" if the Binance 1 minute candle for BTC/USDT 12:00 in the ET timezone (noon) on the date specified in the title has a final "Close" price higher than the price specified in the title. | $87,709 |
| will-bitcoin-dip-to-68k-on-june-2 | BTC | $68,000 | 2026-06-03T04:00:00+00:00 | T | barrier_touch | This market will immediately resolve to "Yes" if any Binance 1 minute candle for Bitcoin (BTC/USDT) on the date specified in the title, between 12:00 AM ET and 11:59 PM ET has a final "Low" price equal to or lower than the price specified in the title. | $81,579 |

Read: classification is by Gamma resolution text, not slug. `Close` at a timestamp goes to Arm E; `High`/`Low` anytime in the day goes to Arm T; unclear rows stay quarantined.

| arm | markets | volume_usd | volume_share |
| --- | --- | --- | --- |
| E | 66 | $4,181,480 | 0.8960567705218091 |
| T | 40 | $485,055 | 0.10394322947819078 |

## Step 2: External Fair-Value Calibration

Arm T uses Binance 1m highs/lows for first-passage labels. Arm E uses Binance 1m terminal closes. Both empirical estimators are expanding-time CV: each validation month uses only prior completed data.

![Arm T calibration](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_same_day_crypto_touch_calibration.png)

![Arm E calibration](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_same_day_crypto_terminal_calibration.png)

| arm | prob_bucket | rows | mean_pred | observed | obs_minus_pred |
| --- | --- | --- | --- | --- | --- |
| E | 0_5c | 71027 | 1.93% | 2.07% | 0.14% |
| E | 5_10c | 33989 | 6.84% | 7.09% | 0.25% |
| E | 10_25c | 67004 | 16.72% | 16.19% | -0.53% |
| E | 25_50c | 42270 | 37.17% | 36.07% | -1.10% |
| E | 50_75c | 51286 | 63.57% | 62.06% | -1.50% |
| E | 75_90c | 56853 | 84.06% | 83.19% | -0.88% |
| E | 90_95c | 33406 | 92.53% | 91.77% | -0.76% |
| E | 95_100c | 72745 | 98.02% | 97.53% | -0.49% |
| T | 0_5c | 90214 | 2.65% | 2.90% | 0.25% |
| T | 5_10c | 65726 | 7.40% | 7.86% | 0.46% |
| T | 10_25c | 95640 | 16.13% | 16.21% | 0.08% |
| T | 25_50c | 82838 | 35.95% | 36.36% | 0.40% |
| T | 50_75c | 31262 | 58.18% | 59.29% | 1.10% |
| T | 75_90c | 62972 | 79.39% | 79.24% | -0.15% |

Read: calibration is checked before PM application. If this table were badly off, the gate would be invalid regardless of PM PnL.

## Step 3: Net-Of-Cost Gates

Selected fills are actual PM taker buys of the side that the empirical model says is underpriced after fee. `realized_net_ev` is payoff minus executed price minus taker fee. Arm E also reports `incremental_vs_structural`, which applies the K5 non-incumbent haircut and subtracts the v4 structural queue baseline of 1.98c.

| arm | sample | fills | markets | mean_price | mean_empirical_prob | mean_model_edge | mean_realized_net_ev | realized_net_ev_ci_lo | mean_incremental_vs_structural | incremental_ci_lo | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E | all_executable_fills | 1662754 | 5726 | 54.14c | 53.89c | -0.88c | -0.37c | [-0.48c, -0.27c] | -2.00c | [-2.00c, -1.99c] | 54.41% |
| E | empirical_edge_positive | 688902 | 5569 | 30.92c | 36.05c | 4.46c | 0.23c | [-0.44c, -0.07c] | -1.97c | [-2.00c, -1.98c] | 31.83% |
| T | all_executable_fills | 110231 | 663 | 69.49c | 69.79c | -0.05c | -0.73c | [-0.96c, 0.13c] | n/a | [n/a, n/a] | 69.10% |
| T | empirical_edge_positive | 64848 | 620 | 76.61c | 80.01c | 3.07c | -0.18c | [-2.32c, 0.04c] | n/a | [n/a, n/a] | 76.76% |

Power read: Arm E's same-day ladder supplies 25.8 independent strike markets per day in the local historical sample, versus the old OD 4h OOS far-|z| gate's n=6 markets. That improves power, but power does not matter unless the incremental-vs-structural lower CI clears.

| arm | z_bucket | tau_bucket | fills | markets | mean_pm_price | mean_empirical_prob | mean_terminal_or_naive | mean_brownian_touch | mean_model_edge | mean_realized_net_ev | realized_net_ev_ci_lo | mean_incremental_vs_structural | incremental_ci_lo | passes_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T | neg_z_2_3 | 15m_1h | 35 | 7 | 55.03c | 56.87c | 1.04c | 2.07c | 1.64c | 1.91c | [0.36c, 2.97c] | n/a | [n/a, n/a] | True |
| T | pos_z_ge_3 | 1_4h | 860 | 145 | 33.23c | 34.27c | 0.01c | 0.03c | 0.96c | 0.64c | [0.05c, 2.34c] | n/a | [n/a, n/a] | True |
| T | neg_z_ge_3 | 15m_1h | 324 | 71 | 82.67c | 83.19c | 0.00c | 0.01c | 0.47c | 0.61c | [0.28c, 0.57c] | n/a | [n/a, n/a] | True |
| T | pos_z_ge_3 | 15m_1h | 254 | 67 | 51.37c | 51.95c | 0.01c | 0.02c | 0.52c | 0.54c | [0.18c, 0.67c] | n/a | [n/a, n/a] | True |
| T | neg_z_1_1p5 | 15m_1h | 5 | 1 | 78.84c | 88.72c | 11.95c | 23.90c | 8.73c | 20.01c | [20.01c, 20.01c] | n/a | [n/a, n/a] | False |
| T | pos_z_0_0p5 | 1_4h | 79 | 6 | 41.69c | 57.05c | 37.44c | 74.87c | 14.05c | 13.97c | [-30.06c, 23.04c] | n/a | [n/a, n/a] | False |
| E | pos_z_0_0p5 | 4_12h | 17667 | 355 | 46.73c | 53.67c | 49.27c | n/a | 5.26c | 8.15c | [-4.24c, 3.94c] | -1.57c | [-2.19c, -1.79c] | False |
| T | neg_z_1p5_2 | 12_24h | 462 | 87 | 26.44c | 38.30c | 4.35c | 8.70c | 11.06c | 7.39c | [-5.47c, 6.78c] | n/a | [n/a, n/a] | False |
| T | pos_z_1_1p5 | 15m_1h | 52 | 7 | 45.69c | 52.73c | 10.09c | 20.19c | 6.24c | 7.36c | [-3.07c, 16.91c] | n/a | [n/a, n/a] | False |
| T | neg_z_1p5_2 | 4_12h | 655 | 86 | 42.05c | 49.47c | 4.11c | 8.22c | 6.69c | 6.99c | [-5.21c, 4.72c] | n/a | [n/a, n/a] | False |
| E | pos_z_0_0p5 | lt_15m | 1248 | 61 | 35.61c | 49.78c | 42.34c | n/a | 12.62c | 6.83c | [-8.28c, 11.92c] | -1.64c | [-2.39c, -1.38c] | False |
| E | pos_z_0p5_1 | 4_12h | 13067 | 382 | 44.11c | 50.66c | 48.38c | n/a | 5.25c | 6.26c | [-3.36c, 3.69c] | -1.67c | [-2.15c, -1.80c] | False |
| E | pos_z_0p5_1 | lt_15m | 1410 | 68 | 19.53c | 49.66c | 24.04c | n/a | 29.07c | 6.15c | [-7.01c, 10.75c] | -1.67c | [-2.31c, -1.43c] | False |
| E | neg_z_0p5_1 | 15m_1h | 3130 | 161 | 40.28c | 47.29c | 44.59c | n/a | 5.90c | 5.63c | [-4.30c, 6.16c] | -1.70c | [-2.19c, -1.69c] | False |
| E | neg_z_1_1p5 | 4_12h | 9856 | 379 | 41.70c | 46.50c | 44.62c | n/a | 3.87c | 4.31c | [-2.45c, 4.82c] | -1.76c | [-2.11c, -1.74c] | False |
| E | neg_z_0_0p5 | 4_12h | 16989 | 346 | 39.97c | 46.76c | 40.94c | n/a | 5.15c | 3.78c | [-5.40c, 2.81c] | -1.79c | [-2.25c, -1.83c] | False |
| E | neg_z_1p5_2 | 1_4h | 6276 | 296 | 22.55c | 28.14c | 22.89c | n/a | 5.05c | 3.68c | [-0.56c, 7.44c] | -1.80c | [-2.01c, -1.61c] | False |
| T | pos_z_1p5_2 | 4_12h | 704 | 101 | 29.26c | 37.16c | 4.04c | 8.09c | 7.35c | 3.58c | [-7.03c, 4.64c] | n/a | [n/a, n/a] | False |
| E | neg_z_1_1p5 | 1_4h | 8184 | 305 | 23.92c | 30.60c | 25.25c | n/a | 5.87c | 3.24c | [1.85c, 10.16c] | -1.82c | [-1.89c, -1.46c] | False |
| T | neg_z_ge_3 | 1_4h | 2579 | 158 | 72.89c | 75.05c | 0.02c | 0.03c | 1.93c | 2.80c | [-1.86c, 0.24c] | n/a | [n/a, n/a] | False |

![Selected-fill EV](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_same_day_crypto_bucket_ev.png)

Caption: selected executable fills by arm and z/time-left cell. Error bars are market-cluster bootstrap CIs. Arm E must also clear the structural-incremental CI.

![Behavioral gap](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_same_day_crypto_behavioral_gap.png)

Caption: PM executed price versus the terminal/naive benchmark and empirical external probability. Arm T would support the retail-touch-underpricing thesis if PM sat near terminal while empirical touch sat near Brownian and materially higher.

Read: these are the gate tables. Arm T has lower-CI-positive defensible same-day touch cells, concentrated in short-tau/far-|z| buckets. Arm E has many ladder strikes, but the incremental-vs-structural bar remains the binding failure.

## Step 4: OFI

OFI is not promoted to a standalone result. The same-day historical fills have Binance path state and PM trades, but not synchronized PM order-book/OFI capture for the exact touch markets. Current CLOB is a one-shot quote snapshot with no realized label. Under A15b/A1.7 discipline, OFI remains untested here rather than treated as alpha.

## Step 5: Capacity

| sample | arm | markets | matched_markets | missing_markets | volume_usd | weighted_top3_share | weighted_non_top3_share | missing_market_slugs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_gamma | E | 66 | 0 | 66 | $4,181,480 | n/a | n/a | bitcoin-above-68k-on-june-2-2026; bitcoin-above-66k-on-june-2-2026; bitcoin-above-72k-on-june-2-2026; bitcoin-above-70k-on-june-2-2026; bitcoin-above-74k-on-june-2-2026; ethereum-above-1900-on-june-2-2026; bitcoin-above-66k-on-june-3-2026; bitcoin-above-76k-on-june-2-2026; ethereum-above-2100-on-june-2-2026; bitcoin-above-76k-on-june-3-2026 |
| current_gamma | T | 40 | 0 | 40 | $485,055 | n/a | n/a | will-bitcoin-reach-71k-on-june-2; will-bitcoin-dip-to-67k-on-june-2; will-bitcoin-dip-to-68k-on-june-2; will-bitcoin-dip-to-69k-on-june-2; will-bitcoin-dip-to-66k-on-june-2; will-ethereum-dip-to-1900-on-june-2; will-bitcoin-dip-to-65k-on-june-2; will-bitcoin-dip-to-63k-on-june-2; will-bitcoin-dip-to-64k-on-june-2; will-ethereum-dip-to-1850-on-june-2 |
| historical_local | E | 5916 | 3330 | 2586 | $1,136,528,654 | 54.69% | 45.31% | bitcoin-above-120k-on-september-12; solana-above-250-on-september-12; ethereum-above-4200-on-september-13; bitcoin-above-118k-on-september-14; bitcoin-above-120k-on-september-14; bitcoin-above-122k-on-september-14; bitcoin-above-104k-on-september-15; bitcoin-above-108k-on-september-15; bitcoin-above-118k-on-september-15; bitcoin-above-120k-on-september-15 |
| historical_local | T | 683 | 335 | 348 | $32,689,557 | 77.32% | 22.68% | will-bitcoin-dip-to-68000-on-march-6; will-bitcoin-dip-to-63k-on-march-9; will-bitcoin-dip-to-67k-on-march-12; will-bitcoin-dip-to-70k-on-march-15; will-solana-dip-to-85-on-march-15; will-bitcoin-dip-to-67k-on-march-16; will-bitcoin-dip-to-67k-on-march-17; will-bitcoin-dip-to-69k-on-march-18; will-bitcoin-reach-78k-on-march-20; will-bitcoin-dip-to-63k-on-march-26 |

Read: same-day current markets are newer than the K5 cache and are often missing exact concentration. Missing rows inherit no proof of headroom. Historical matched rows still show the usual top-maker concentration problem.

## Decision

**Arm T:** CLOSE. best registered held-out Tier-1 cell pos_z_ge_3/1_4h has 5% haircut EV 0.17c with CI [0.12c, 0.27c] and BH capacity q=0.0003; mean is below the pre-registered 0.25c materiality bar.

**Arm E:** CLOSE. terminal ladder power improves the sample, but no cell clears both net-EV lower CI and incremental-vs-structural lower CI.

Concrete next step: close same-day Arm T as a standalone strategy; fold the first-passage/HAR-Kou IV-gap flag into MM as a caution feature. Keep Arm E closed.

## Outputs

- Classification CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_classification.csv`
- Historical markets CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_history_markets.csv`
- Arm summary CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_arm_summary.csv`
- Bucket summary CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_bucket_summary.csv`
- Calibration CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_calibration.csv`
- Current quote CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_current_quotes.csv`
- Capacity CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_capacity.csv`
- Fill sample parquet: `data/analysis/od_same_day_crypto_pricing_fills.parquet`
- Script: `scripts/od_same_day_crypto_pricing_gate.py`

## Confirmation Pass

This confirmation pass closes the earlier asymmetry: Arm T is now checked with capacity haircut, time-embargo OOS, and multiple-comparison correction, while Arm E is not rerun. The pass uses the already-saved PM fill sample and Binance-derived fair values from this note; no new market capture is added.

Pre-registered split rule: sort Arm T resolution dates, use the first 70% as train/pre-embargo, embargo the next resolution date, and evaluate only the remaining held-out dates. In this run train is 2026-03-07 to 2026-04-20 (57418 fills / 436 markets), the embargo date is 2026-04-21 (401 fills), and held-out is 2026-04-22 to 2026-05-07 (7029 fills / 169 markets).

Capacity rule: Arm T now reports raw net EV and 5% K5-style non-incumbent-capacity haircut EV. I could not derive a touch-specific passive structural/MM baseline from the saved fill sample alone because it has executed taker fills but not synchronized resting quote/queue states. The 1.98c structural baseline is therefore shown as a borrowed crypto-4h diagnostic, not the primary touch gate. The Arm T confirmation pass bar is held-out raw net-EV lower CI > 0, held-out capacity-haircut lower CI > 0, and BH-adjusted q <= 0.05 across all held-out defensible cells.

Train registered 3 cells:

| z_bucket | tau_bucket | fills | markets | mean_realized_net_ev | realized_net_ev_ci_lo | mean_capacity_haircut_ev | capacity_haircut_ci_lo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pos_z_1p5_2 | 12_24h | 357 | 65 | 11.17c | [2.41c, 20.59c] | 0.56c | [0.11c, 1.02c] |
| neg_z_ge_3 | 15m_1h | 213 | 48 | 0.45c | [0.29c, 0.61c] | 0.02c | [0.01c, 0.03c] |
| pos_z_ge_3 | 15m_1h | 125 | 43 | 0.35c | [0.12c, 0.62c] | 0.02c | [0.01c, 0.03c] |

Read: these cells are selected using train/pre-embargo data only. Held-out performance below decides the verdict.

Held-out registered cells after BH across 39 defensible held-out cells:

| z_bucket | tau_bucket | fills | markets | mean_realized_net_ev | realized_net_ev_ci_lo | realized_net_ev_q | mean_capacity_haircut_ev | capacity_haircut_ci_lo | capacity_haircut_q | mean_borrowed_structural_ev | borrowed_structural_ci_lo | passes_confirmation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neg_z_ge_3 | 15m_1h | 106 | 19 | 0.44c | [0.15c, 0.80c] | 0.0130 | 0.02c | [0.01c, 0.04c] | 0.0065 | -1.96c | [-1.97c, -1.94c] | True |
| pos_z_ge_3 | 15m_1h | 119 | 21 | 0.47c | [-0.01c, 1.08c] | 0.1634 | 0.02c | [-0.00c, 0.05c] | 0.2154 | -1.96c | [-1.98c, -1.92c] | False |
| pos_z_1p5_2 | 12_24h | 220 | 25 | -10.39c | [-21.47c, 0.32c] | 0.9820 | -0.52c | [-1.12c, 0.03c] | 0.9820 | -2.50c | [-3.06c, -1.98c] | False |

Read: `mean_capacity_haircut_ev` is raw realized net EV multiplied by 0.05. `mean_borrowed_structural_ev` subtracts the old 1.98c crypto-4h structural baseline after that haircut; it is included to show the severity of the structural bar, but it is borrowed and not touch-specific.

Mechanism check on survivors:

| z_bucket | tau_bucket | fills | markets | mean_pm_price | mean_naive_terminal | mean_brownian_touch | mean_empirical_prob | mean_realized_net_ev | capacity_haircut_ci_lo | mechanism_read |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neg_z_ge_3 | 15m_1h | 106 | 19 | 83.16c | 0.00c | 0.01c | 83.86c | 0.44c | [0.01c, 0.04c] | fragile calibration residual: PM is close to empirical first-passage, not near terminal |

Read: the survivor count is 1. The deciding number is the best survivor's held-out capacity-haircut lower CI if any survivor exists; otherwise the deciding number is the maximum held-out capacity-haircut lower CI among train-registered cells.

**Revised Arm T verdict:** MERITS-BUILD. best confirmed cell neg_z_ge_3/15m_1h has held-out net CI [0.15c, 0.80c] and 5% capacity-haircut CI [0.01c, 0.04c]; BH capacity q=0.0065; mechanism is fragile calibration residual: PM is close to empirical first-passage, not near terminal.

Confirmation CSVs: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_confirmation_train_cells.csv` and `data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_confirmation_heldout_cells.csv`.

## Tier-1 Edge-Concentration Extension

This extension asks whether a mechanism-backed filter can lift same-day Arm T above the confirmation pass's economically trivial ~0.02c/contract capacity-haircut floor. It uses only the saved PM fill sample plus cached Binance 1m bars; no new market capture, Deribit feed, Kronos model, or PM consistency scan is introduced.

Pre-registered Tier-1 rule: start with the original external empirical-edge-positive Arm T fills, invert each PM touch YES price to Brownian first-passage implied vol, compare it with a causal HAR-RV-J forecast plus directional Kou jump variance, and keep only fills where `PM touch IV - HAR/Kou sigma` is at least 0.25 annualized vol in the side-implied direction. Rich touch IV means fade by buying NO; cheap touch IV means buy YES. The materiality bar is 0.25c per contract after the 5% non-incumbent capacity haircut, with held-out lower CI > 0 and BH q <= 0.05.

OOS realism rule: use an embargoed held-out split only when the saved sample has at least 20 resolution dates and spans at least 30 days. Otherwise the script treats the sample as train-only rather than manufacturing a powerless split. The OOS guard is active and this sample passes it: 52 resolution dates over 62 days. Train is 2026-03-07 to 2026-04-20 (42347 Tier-1 fills / 163 markets), embargo is 2026-04-21 (5 fills), and held-out is 2026-04-22 to 2026-05-07 (232 fills / 51 markets).

Touch-specific structural baseline attempt: the saved sample has executed taker fills and realized outcomes, but not synchronized passive quote/queue states. A touch-market passive MM baseline cannot be derived offline from these fields. The old 1.98c crypto-4h structural baseline remains a borrowed diagnostic only, not an Arm T kill switch. The Tier-1 gate is raw net EV + 5% capacity-haircut EV + OOS/BH + materiality.

Directional HAR/Kou snapshot:

| asset | forecast_session_date | sigma_har_annualized | sigma_har_kou_up_annualized | sigma_har_kou_down_annualized | kou_up_lambda_per_year | kou_down_lambda_per_year | kou_up_avg_abs_jump | kou_down_avg_abs_jump |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC | 2026-06-02 | 0.422 | 0.431 | 0.435 | 149.638 | 157.492 | 0.007 | 0.008 |
| ETH | 2026-06-02 | 0.597 | 0.614 | 0.625 | 165.345 | 191.801 | 0.011 | 0.014 |
| SOL | 2026-06-02 | 0.678 | 0.696 | 0.703 | 104.581 | 113.675 | 0.015 | 0.017 |

Read: the Kou extension is a cheap moment-matched directional jump variance add-on, not a full jump-diffusion pricer. It is enough for the gate question: does PM touch IV look mechanically rich/cheap versus a causal vol-and-jump forecast?

![Tier-1 IV gap](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_same_day_crypto_arm_t_tier1_iv_gap.png)

Caption: PM touch implied vol minus causal HAR-RV-J/Kou forecast vol for empirical-edge-positive Arm T fills. Dashed lines are the pre-registered +/-25 vol-point selection threshold.

Train-registered Tier-1 cells:

| z_bucket | tau_bucket | fills | markets | mean_abs_z_har_kou | mean_pm_touch_iv | mean_sigma_har_kou | mean_iv_gap_har_kou | mean_realized_net_ev | realized_net_ev_ci_lo | mean_capacity_haircut_ev | capacity_haircut_ci_lo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neg_z_2_3 | 4_12h | 51 | 11 | 2.76 | 0.916 | 0.564 | 0.351 | 9.09c | [6.67c, 11.99c] | 0.45c | [0.32c, 0.60c] |
| neg_z_ge_3 | 4_12h | 766 | 26 | 3.98 | 1.072 | 0.570 | 0.502 | 3.57c | [3.07c, 4.20c] | 0.18c | [0.15c, 0.21c] |
| neg_z_ge_3 | 1_4h | 468 | 23 | 4.98 | 1.137 | 0.468 | 0.669 | 2.87c | [2.48c, 3.28c] | 0.14c | [0.12c, 0.16c] |
| pos_z_ge_3 | 1_4h | 110 | 19 | 6.02 | 1.939 | 0.728 | 1.212 | 2.70c | [2.31c, 3.07c] | 0.13c | [0.12c, 0.15c] |
| neg_z_ge_3 | 15m_1h | 91 | 26 | 6.77 | 1.417 | 0.582 | 0.835 | 0.71c | [0.50c, 0.94c] | 0.04c | [0.02c, 0.05c] |

Read: these cells are chosen from pre-embargo data only. Held-out cells below decide the verdict.

Held-out registered cells after BH across 2 defensible Tier-1 cells:

| z_bucket | tau_bucket | fills | markets | mean_abs_z_har_kou | mean_entry_price | mean_pm_yes_price | mean_empirical_yes_prob | mean_naive_terminal_yes_prob | mean_pm_touch_iv | mean_sigma_har_kou | mean_iv_gap_har_kou | mean_realized_net_ev | realized_net_ev_ci_lo | realized_net_ev_q | mean_capacity_haircut_ev | capacity_haircut_ci_lo | capacity_haircut_q | passes_materiality | passes_tier1 | mechanism_read |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pos_z_ge_3 | 1_4h | 31 | 9 | 5.98 | 96.16c | 3.84c | 2.12c | 0.01c | 1.755 | 0.603 | 1.152 | 3.49c | [2.29c, 5.43c] | 0.0003 | 0.17c | [0.12c, 0.27c] | 0.0003 | False | False | calibration residual: PM touch price is already close to empirical first-passage |

![Tier-1 held-out EV](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_same_day_crypto_arm_t_tier1_heldout_ev.png)

Caption: held-out 5% capacity-haircut EV per contract. The red dashed line is the 0.25c materiality threshold.

Mechanism check on Tier-1 survivors:

No Tier-1 cell survived OOS + capacity + BH + materiality.

Read: the survivor count is 0. A survivor would merit a constrained live **MEASUREMENT** loop, not a trading system; no survivor means same-day touch is efficient net of realistic costs at this offline resolution.

Assumption-vs-live ledger:

| Ledger bucket | Items |
|---|---|
| Modeled assumptions | 5% non-incumbent capacity share; taker entry at executed price plus PM fee; empirical first-passage base rates from prior Binance history; HAR-RV-J and directional Kou jump parameters from causal Binance 1m history; OOS/BH market-cluster CIs. |
| Live-only unknowns | Passive fill rate; real non-incumbent headroom on these exact same-day touch markets; touch-specific passive/MM baseline; adverse selection around barrier touches; persistence versus the K3 ~10s/54s lead-lag decay; quote/queue behavior during touch jumps. |

**Tier-1 Arm T verdict:** CLOSE. best registered held-out Tier-1 cell pos_z_ge_3/1_4h has 5% haircut EV 0.17c with CI [0.12c, 0.27c] and BH capacity q=0.0003; mean is below the pre-registered 0.25c materiality bar.

Tier-1 outputs: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_tier1_train_cells.csv`, `data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_tier1_heldout_cells.csv`, `data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_tier1_kou_params.csv`, and `data/analysis/od_same_day_crypto_arm_t_tier1_fills.parquet`.
