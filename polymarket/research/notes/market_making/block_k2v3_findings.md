---
title: Block K2 v3 Digital-Anchored Maker Mechanism Test
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
  - digital-anchor
  - maker-exit
  - research
---

# Block K2 v3: digital-anchored maker mechanism test

> **Strat:** [[strat_market_making]] (Market-Making). Sibling: [[strat_options_delta]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Summary

K2 v3 replaces the old Polymarket-mid fair with a lagged Binance digital fair and evaluates taker exits, passive maker exits, and hold-to-resolution variants. No robust bucket clears zero across those exit policies. The result closes the digital-anchored maker mechanism on the tested a0b+a0c_roll crypto-4h universe.

**Verdict:** No robust bucket clears zero across taker, maker-exit, or hold-to-resolution exits.

This run keeps the K2/K-PEG guardrails: causal quote inputs, one-share passive fill proxy, non-overlapping inventory by exit policy, net-of-rebate and net-of-exit-cost PnL, and block bootstrap CIs. The digital fair used for quoting is explicitly lagged by the swept latency `L`, so the quote never gets contemporaneous Binance information. The test universe is a0b+a0c_roll crypto-4h; daily crypto rows were present in A1 but excluded from the digital-anchor simulation because the K3/Binance surface only has the 4h strike/window panel.

## Headline

Least-negative digital config under the taker-exit maker loop: `digital_b100_L5_m1_flat0z0m`. The Sharpe-like objective pick was `digital_b750_L10_m2_flat0.75z30m`; both remain CI-negative.
No digital search config had positive taker-exit mean; the least-negative mean was -3412.0 bps, CI [-4223.7 bps, -2760.0 bps], n=1621.
Taker-exit selected result: -3412.0 bps, CI [-4223.7 bps, -2760.0 bps], n=1621.
Best maker-exit window among the selected digital config: -2216.2 bps, CI [-3396.1 bps, -1278.7 bps], maker-exit fill rate 55.5%, window=1800s, offset=5 ticks.
Hold-to-resolution: 242.5 bps, CI [-1148.3 bps, 1640.6 bps], n=602.
Shadow entry-delta hedge variant: 311.0 bps, CI [-803.5 bps, 1384.9 bps], mean hedge cost 578.3 bps.

## What Was Tested

The old K2 quote used Polymarket mid as fair value. K2 v3 instead estimates the probability that the crypto window settles Up from Binance spot, the window strike, time left, and a causal realized-volatility estimate. Quotes are widened by the digital option's spot delta, so quotes automatically get wider when a small Binance move can change the settlement probability a lot.

A fill happens only if a real trade print would have crossed our modeled passive quote within the A1.4h freshness window. For digital configs, the Binance anchor, delta, z, and theta are read from `quote_time - L`; the local book spread is still read at quote time because that is the venue state being quoted. After entry, the same fills are evaluated three ways: force taker exit after 60s, post passive maker exit after 30s with longer resting windows, or hold to actual Gamma settlement.

## Latency Guardrail

| anchor lag L | best config at L | fills | mean | 95% CI |
| --- | --- | --- | --- | --- |
| 1s | digital_b100_L1_m2_flat0z0m | 1599 | -3429.2 bps | [-4246.6 bps, -2654.1 bps] |
| 5s | digital_b100_L5_m1_flat0z0m | 1621 | -3412.0 bps | [-4223.7 bps, -2760.0 bps] |
| 10s | digital_b100_L10_m1_flat0z0m | 1560 | -3538.3 bps | [-4334.7 bps, -2813.5 bps] |

## Policy Summary

| exit | fills | mean | 95% CI | win | fills/day | fills/active-hr |
| --- | --- | --- | --- | --- | --- | --- |
| hold_resolution | 602 | 242.5 bps | [-1148.3 bps, 1640.6 bps] | 49.7% | 241.5 | 3.84 |
| hold_resolution_delta_hedged | 602 | 311.0 bps | [-803.5 bps, 1384.9 bps] | 49.5% | 241.5 | 3.84 |
| maker_exit_post30_win30_off0 | 1614 | -3162.3 bps | [-3744.1 bps, -2584.0 bps] | 13.9% | 647.4 | 10.31 |
| maker_exit_post30_win30_off2 | 1606 | -3242.0 bps | [-3985.4 bps, -2669.1 bps] | 13.5% | 644.2 | 10.26 |
| maker_exit_post30_win30_off5 | 1612 | -3232.9 bps | [-3881.6 bps, -2681.8 bps] | 12.9% | 646.6 | 10.29 |
| maker_exit_post30_win60_off0 | 1465 | -2983.2 bps | [-3581.9 bps, -2459.6 bps] | 16.5% | 587.6 | 9.36 |
| maker_exit_post30_win60_off2 | 1451 | -3059.4 bps | [-3741.8 bps, -2528.2 bps] | 16.7% | 582.0 | 9.27 |
| maker_exit_post30_win60_off5 | 1461 | -2986.7 bps | [-3557.4 bps, -2454.0 bps] | 15.9% | 586.0 | 9.33 |
| maker_exit_post30_win120_off0 | 1275 | -2952.4 bps | [-3508.7 bps, -2448.3 bps] | 19.1% | 511.4 | 8.14 |
| maker_exit_post30_win120_off2 | 1260 | -2974.5 bps | [-3668.9 bps, -2345.3 bps] | 21.2% | 505.4 | 8.05 |
| maker_exit_post30_win120_off5 | 1246 | -2888.0 bps | [-3440.4 bps, -2379.3 bps] | 20.2% | 499.8 | 7.96 |
| maker_exit_post30_win300_off0 | 1033 | -2939.7 bps | [-3654.1 bps, -2269.9 bps] | 25.2% | 414.3 | 6.60 |
| maker_exit_post30_win300_off2 | 996 | -2877.2 bps | [-3735.3 bps, -2163.7 bps] | 29.0% | 399.5 | 6.36 |
| maker_exit_post30_win300_off5 | 977 | -2675.0 bps | [-3420.3 bps, -2119.6 bps] | 29.0% | 391.9 | 6.24 |
| maker_exit_post30_win600_off0 | 893 | -3016.5 bps | [-4178.7 bps, -2141.9 bps] | 29.6% | 358.2 | 5.70 |
| maker_exit_post30_win600_off2 | 857 | -2522.0 bps | [-3450.1 bps, -1786.3 bps] | 36.9% | 343.7 | 5.47 |
| maker_exit_post30_win600_off5 | 839 | -2401.3 bps | [-3090.1 bps, -1777.7 bps] | 37.3% | 336.5 | 5.36 |
| maker_exit_post30_win1800_off0 | 737 | -2569.1 bps | [-3503.3 bps, -1790.2 bps] | 35.0% | 295.6 | 4.71 |
| maker_exit_post30_win1800_off2 | 698 | -2398.2 bps | [-3666.4 bps, -1469.4 bps] | 47.4% | 280.0 | 4.46 |
| maker_exit_post30_win1800_off5 | 679 | -2216.2 bps | [-3396.1 bps, -1278.7 bps] | 53.2% | 272.3 | 4.34 |
| taker_60s | 1621 | -3412.0 bps | [-4223.7 bps, -2760.0 bps] | 11.5% | 650.2 | 10.35 |

## Maker Exit Windows

| window s | offset ticks | fills | exit fill | mean | 95% CI |
| --- | --- | --- | --- | --- | --- |
| 30 | 0 | 1614 | 13.6% | -3162.3 bps | [-3744.1 bps, -2584.0 bps] |
| 30 | 2 | 1606 | 7.7% | -3242.0 bps | [-3985.4 bps, -2669.1 bps] |
| 30 | 5 | 1612 | 3.0% | -3232.9 bps | [-3881.6 bps, -2681.8 bps] |
| 60 | 0 | 1465 | 21.0% | -2983.2 bps | [-3581.9 bps, -2459.6 bps] |
| 60 | 2 | 1451 | 13.8% | -3059.4 bps | [-3741.8 bps, -2528.2 bps] |
| 60 | 5 | 1461 | 5.9% | -2986.7 bps | [-3557.4 bps, -2454.0 bps] |
| 120 | 0 | 1275 | 29.9% | -2952.4 bps | [-3508.7 bps, -2448.3 bps] |
| 120 | 2 | 1260 | 20.6% | -2974.5 bps | [-3668.9 bps, -2345.3 bps] |
| 120 | 5 | 1246 | 10.1% | -2888.0 bps | [-3440.4 bps, -2379.3 bps] |
| 300 | 0 | 1033 | 44.6% | -2939.7 bps | [-3654.1 bps, -2269.9 bps] |
| 300 | 2 | 996 | 34.6% | -2877.2 bps | [-3735.3 bps, -2163.7 bps] |
| 300 | 5 | 977 | 21.1% | -2675.0 bps | [-3420.3 bps, -2119.6 bps] |
| 600 | 0 | 893 | 55.8% | -3016.5 bps | [-4178.7 bps, -2141.9 bps] |
| 600 | 2 | 857 | 46.8% | -2522.0 bps | [-3450.1 bps, -1786.3 bps] |
| 600 | 5 | 839 | 31.3% | -2401.3 bps | [-3090.1 bps, -1777.7 bps] |
| 1800 | 0 | 737 | 72.7% | -2569.1 bps | [-3503.3 bps, -1790.2 bps] |
| 1800 | 2 | 698 | 66.5% | -2398.2 bps | [-3666.4 bps, -1469.4 bps] |
| 1800 | 5 | 679 | 55.5% | -2216.2 bps | [-3396.1 bps, -1278.7 bps] |

## Best Buckets

`clears` means CI lower > 0 **and** at least 30 fills. Multiple-testing scan: 5 raw CI-positive bucket-policy cells; 0 robust clears.

The pre-registered bucket is mid-|z| / mid-tau / moderate-spread. Other cells are exploratory and should be read against the multiple-testing count, not as standalone discoveries.

| exit | fills | mean | 95% CI | clears |
| --- | --- | --- | --- | --- |
| hold_resolution | 14 | 1142.6 bps | [458.6 bps, 2470.3 bps] | no |
| hold_resolution_delta_hedged | 14 | -543.3 bps | [-2868.0 bps, 1182.5 bps] | no |
| maker_exit_post30_win1800_off5 | 19 | -259.5 bps | [-1010.4 bps, 178.7 bps] | no |
| taker_60s | 50 | -2000.8 bps | [-3087.7 bps, -1182.9 bps] | no |

| exit | |z| | tau | spread | fills | mean | 95% CI | clears |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hold_resolution | mid_0.5-1.5 | very_early_180m+ | moderate_<=1925bps | 1 | 25539.8 bps | [n/a, n/a] | no |
| hold_resolution | mid_0.5-1.5 | very_early_180m+ | tight_<=765bps | 3 | 20287.6 bps | [n/a, n/a] | no |
| hold_resolution_delta_hedged | mid_0.5-1.5 | very_early_180m+ | moderate_<=1925bps | 1 | 17311.5 bps | [n/a, n/a] | no |
| hold_resolution_delta_hedged | mid_0.5-1.5 | very_early_180m+ | tight_<=765bps | 3 | 13538.6 bps | [n/a, n/a] | no |
| hold_resolution_delta_hedged | far_|z|>1.5 | late_0-15m | wide_>1925bps | 7 | 12250.9 bps | [-309.6 bps, 29192.5 bps] | no |
| hold_resolution | far_|z|>1.5 | early_60-180m | wide_>1925bps | 8 | 10923.4 bps | [-198396.1 bps, 105024.2 bps] | no |
| hold_resolution_delta_hedged | far_|z|>1.5 | mid_15-60m | wide_>1925bps | 16 | 8117.8 bps | [1759.3 bps, 15354.3 bps] | no |
| hold_resolution_delta_hedged | far_|z|>1.5 | early_60-180m | wide_>1925bps | 8 | 5976.2 bps | [-153058.2 bps, 75670.9 bps] | no |
| hold_resolution_delta_hedged | far_|z|>1.5 | late_0-15m | moderate_<=1925bps | 2 | 5832.2 bps | [n/a, n/a] | no |
| hold_resolution | far_|z|>1.5 | mid_15-60m | wide_>1925bps | 16 | 5591.5 bps | [3271.5 bps, 9772.4 bps] | no |
| hold_resolution | mid_0.5-1.5 | early_60-180m | wide_>1925bps | 33 | 4952.5 bps | [-10409.8 bps, 24468.2 bps] | no |
| hold_resolution_delta_hedged | mid_0.5-1.5 | early_60-180m | wide_>1925bps | 33 | 3486.7 bps | [-5761.6 bps, 15843.0 bps] | no |

## OOS Guardrail

No robust in-sample bucket cleared, so the held-out confirmation step was not triggered. If a bucket clears in a future rerun, the script emits frozen a0b/a0c_roll confirmation rows before the note can call it deployable.

_No rows._

## Spread Surface

| time since open | mean spread | mean |z| | active hrs |
| --- | --- | --- | --- |
| 0-15m | 1560.1 bps | 0.34 | 9.5 |
| 15-30m | 1911.9 bps | 0.39 | 9.1 |
| 30-60m | 2213.0 bps | 0.45 | 18.6 |
| 60-120m | 1997.2 bps | 0.90 | 39.5 |
| 120-240m | 2887.7 bps | 1.16 | 79.9 |
| 240m+ | 2226.9 bps | 0.01 | 0.0 |

## Figures

![k2v3 spread surface](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/k2v3_plots/k2v3_spread_surface.png)
![k2v3 pnl taker 60s](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/k2v3_plots/k2v3_pnl_taker_60s.png)
![k2v3 pnl maker exit post30 win1800 off5](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/k2v3_plots/k2v3_pnl_maker_exit_post30_win1800_off5.png)
![k2v3 pnl hold resolution](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/k2v3_plots/k2v3_pnl_hold_resolution.png)
![k2v3 pnl hold resolution delta hedged](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/k2v3_plots/k2v3_pnl_hold_resolution_delta_hedged.png)

## Mechanism Checks

Same-sample mid-anchor K2-style config had mean 30s adverse-selection cost 144.5 bps; the best digital-anchor config had 324.7 bps. This is the direct anchor comparison, not the old pooled-all-markets K2 result.
Multiple-testing guardrail: 681 bucket-policy cells were evaluated; 5 had raw CI lower > 0, and 0 survived the 30-fill robust-clear rule. Pre-registered robust clears: 0.
K-PEG selected crypto-4h fills were 36.8% past the Binance-implied fair, with mean distance 3.53 ticks. This checks whether the chase sign-flip sits at the external fair rather than the local micro-price.
Resolution-source check: 1/24 crypto-4h markets had Binance proxy direction disagree with Gamma settlement. This is a direction check, not a tick-level Chainlink-vs-Binance basis estimate.

## Simple Conclusion

Digital anchoring is the right way to test the adverse-selection mechanism, but the neutral maker loop still has to pay exit costs. If taker and maker-exit rows remain CI-negative while hold-to-resolution is positive, the result is not a single-venue market-making edge; it is a Track-A entry-and-carry/hedge question.

Daily crypto note: the A1 panel contained daily crypto markets, but they were not included in the digital quote search because the available K3 fair-value panel covers crypto-4h only. A daily extension should parse Gamma eventStartTime/endDate and replay a matching Binance surface before pooling daily rows.

## Files

- `data/analysis/csv_outputs/market_making/k2v3_digital_maker.csv`
- `notes/market_making/block_k2v3_findings.md`
- Input markets: 43 crypto markets in A1 a0b/a0c_roll; daily quote-state rows observed: 789,238
