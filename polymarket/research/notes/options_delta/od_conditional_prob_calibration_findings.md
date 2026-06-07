---
title: "OD Conditional Resolution-Probability Calibration: Binance-Only Reopen-Or-Close Test"
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
# OD Conditional Resolution-Probability Calibration: Binance-Only Reopen-Or-Close Test

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior OD notes: [[od_v4_calibration_gate_findings]] · [[od_v4_queue_replay_findings]] · [[od_strategy_a_v3_pnl_risk_findings]]
> ML discipline gate: [[2026-05-31_kronos_hermes_eval]]
> Table terms: [[polymarket_table_dictionary]]

## Summary

- Scope: OD Conditional Resolution-Probability Calibration: Binance-Only Reopen-Or-Close Test in the OD/options-delta area.
- Existing takeaway/status: Final verdict: **CLOSE**.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Headline

Final verdict: **CLOSE**.

OD **CLOSES as a standalone strategy** in this time-boxed test. Arm B does not prove that OD fair-value richness adds robust incremental edge beyond the structural/source quote-selection baseline.

Arm B, the Binance-only empirical conditional probability model, predicts 52.22% ITM on the same 23-fill v4 far-|z| strict-rich short set. The tokens actually paid 39.13%. The mean Arm-B model edge is 3.83c, CI [1.94c, 5.73c]; realized net EV is 17.05c, CI [-1.84c, 26.49c]. After the K5 top-3 maker capacity haircut, the market-level lower CI is -0.13c versus a 0c baseline and -2.11c after subtracting the borrowed v4 structural queue baseline.

Plain-English read: Binance history says the far-|z| states are not absurd longshots. In the exact PM set, the empirical conditional probability is close enough to the traded price that the OD richness gap is not independently decisive. The positive realized cents are still small-sample/concentration-sensitive, and the structural 0c-edge replay remains the better explanation than a standalone OD valuation edge.

## Design

This test uses **no new Polymarket capture**. It downloads/caches Binance spot 5-minute klines for BTC/ETH/SOL from `2021-01` through `2026-05` and builds synthetic 4h UP/DOWN windows aligned to UTC 4h boundaries. For every in-window 5-minute state, it computes:

```text
K = Binance open at the 4h window start
tau = seconds to the 4h close
z = ln(S_t / K) / (causal_EWMA_sigma_t * sqrt(tau))
outcome = 1 if Binance close > K else 0
```

Arm A is the old RV physical-probability control: `N(z)`. Arm B is model-free: expanding-time CV estimates empirical `P(resolve UP | signed z bucket, time-left bucket)` from prior Binance history. For the PM token side, UP uses that probability directly and DOWN uses `1 - P(resolve UP)`.

Historical sample: 1,234,539 Binance 5-minute states, 26,268 asset-window 4h outcomes across 8,756 UTC time slots, assets `BTC, ETH, SOL`. Expanding-CV validation rows: 927,075.

### Granularity Caveat: What The 5-Minute History Is And Is Not

The 2021-to-2026 Binance history is a **broad base-rate calibration**, not a live-execution-quality reconstruction of the exact Polymarket episodes. It answers: "In thousands of historical 4h crypto windows, what is the empirical resolution frequency for this signed moneyness/time-left state?" That is useful for detecting whether the Gaussian `N(z)` model is wildly wrong.

It does **not** replace the captured-window data. For the actual PM windows we also have much richer, more relevant evidence:

- `data/analysis/block_a0c_roll_features.parquet`: crypto-4h Polymarket LOB/WS capture with top-of-book, depth, trade flow, OFI, and exchange timestamps across 38 crypto-4h slugs on 2026-05-29 to 2026-05-30.
- `data/analysis/block_a0c_features.parquet`: A0c targeted capture with crypto-4h plus daily crypto rows on 2026-05-29.
- `data/analysis/cache/k2v2_daily_binance_1s.parquet` and `data/analysis/cache/k2v2_daily_model_surface.parquet`: 1s Binance/model surface for the daily BTC/ETH crypto capture on 2026-05-27 to 2026-05-28.

Plain-English read: the 5-minute historical panel is a good truth-table prior; the captured 1s Binance + Polymarket LOB windows are the right place to ask whether the **live market state** had jumps, order-flow pressure, OFI, liquidity depletion, or source-basis risk that the broad 5-minute table cannot see.

For the pricing-model-form reopen test, the preferred ordering should be:

1. Use the historical 5-minute panel only as the background calibration/control for `P(resolve | z, tau)`. 2. For the actual PM validation rows, rebuild the state at 1s granularity from the captured windows where available. 3. Estimate jump/OFI features from the local live window around each fill, not only from multi-year unconditional Binance history. 4. Treat Deribit BTC/ETH as an illustrative market-IV anchor for the same captured days, not as a powered gate.

This caveat does not overturn this note's close verdict. It narrows what kind of evidence would be allowed to reopen OD: a stronger pricing-model-form test should show residual EV on the **captured-window/live 1s panel**, not merely on a smoother 5-minute historical lookup.

## Arm B Binance Reliability

![Arm B calibration](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_conditional_prob_arm_b_calibration.png)

The diagonal is perfect calibration. Points near the diagonal mean the empirical conditional table is learning the actual Binance resolution frequency for that probability range.

| Arm B probability bucket | rows | mean predicted UP | observed UP | observed - predicted |
| --- | --- | --- | --- | --- |
| 0_5c | 88329 | 1.85% | 1.70% | -0.14% |
| 5_10c | 21963 | 7.03% | 6.31% | -0.72% |
| 10_25c | 102494 | 16.73% | 16.75% | 0.02% |
| 25_50c | 248689 | 39.23% | 39.08% | -0.14% |
| 50_75c | 248938 | 61.45% | 62.38% | 0.93% |
| 75_90c | 97435 | 82.71% | 83.37% | 0.66% |
| 90_95c | 32017 | 92.63% | 93.50% | 0.88% |
| 95_100c | 87210 | 98.29% | 98.47% | 0.18% |

Read: this is the truth-teller independent of the tiny Polymarket sample. If Arm B were badly calibrated here, it would be an invalid reopen signal. If Arm B is calibrated but removes the OD edge on PM fills, the old `N(z)` gap was mostly our model/specification, not a robust Polymarket mispricing.

## PM v4 Far-|z| Short Set

| label | fills | markets | mean_short_price | mean_pred_itm_prob | realized_itm_rate | obs - pred | mean_model_edge | edge CI | mean net EV | net EV CI | after-top3 market net | after-top3 CI | incremental vs 0c | inc vs 0c CI | borrowed baseline | incremental vs borrowed | inc vs borrowed CI | primary lookup share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_a_ewma_nz_original_set | 23 | 8 | 56.05c | 52.53% | 39.13% | -13.40% | 3.53c | [2.45c, 4.34c] | 17.05c | [-1.84c, 26.49c] | 2.45c | [-0.13c, 6.76c] | 2.45c | [-0.13c, 6.76c] | 1.98c | 0.48c | [-2.11c, 4.79c] | 100.00% |
| arm_b_empirical_conditional_original_set | 23 | 8 | 56.05c | 52.22% | 39.13% | -13.09% | 3.83c | [1.94c, 5.73c] | 17.05c | [-1.84c, 26.49c] | 2.45c | [-0.13c, 6.76c] | 2.45c | [-0.13c, 6.76c] | 1.98c | 0.48c | [-2.11c, 4.79c] | 100.00% |
| arm_b_empirical_conditional_rich_ge_1c | 15 | 6 | 36.81c | 30.71% | 26.67% | -4.04% | 6.10c | [3.52c, 7.19c] | 10.29c | [2.28c, 14.52c] | 1.29c | [0.17c, 2.55c] | 1.29c | [0.17c, 2.55c] | 1.98c | -0.69c | [-1.80c, 0.58c] | 100.00% |
| arm_b_empirical_conditional_rich_ge_5c | 10 | 4 | 15.32c | 7.47% | 0.00% | -7.47% | 7.85c | [7.34c, 8.21c] | 15.50c | [14.54c, 17.20c] | 1.94c | [0.81c, 3.07c] | 1.94c | [0.81c, 3.07c] | 1.98c | -0.04c | [-1.17c, 1.09c] | 100.00% |

Column read: `mean_model_edge` is `short price - predicted ITM probability`. `mean net EV` is the actual resolution PnL per fill after maker rebate. `after-top3 market net` applies the 5% non-incumbent capacity haircut used in v4. `incremental vs 0c` is the raw capacity-adjusted read. `incremental vs borrowed` subtracts the best v4 0c-edge queue replay baseline. Live-measured baseline is not filled in this run because there is no separate live queue baseline artifact for the same validation rows yet.

![PM EV by arm](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_conditional_prob_pm_ev.png)

## Arm B PM Reliability Buckets

| Arm B probability bucket | fills | markets | mean predicted ITM | observed ITM | observed - predicted | mean short price | mean edge | mean net EV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5_10c | 8 | 3 | 6.20% | 0.00% | -6.20% | 13.65c | 7.45c | 13.81c |
| 10_25c | 3 | 2 | 10.44% | 0.00% | -10.44% | 18.00c | 7.56c | 18.21c |
| 75_90c | 4 | 2 | 89.22% | 25.00% | -64.22% | 88.00c | -1.22c | 63.15c |
| 90_95c | 4 | 3 | 92.59% | 100.00% | 7.41% | 94.06c | 1.47c | -5.86c |
| 95_100c | 4 | 1 | 98.22% | 100.00% | 1.78% | 99.45c | 1.23c | -0.54c |

Read: the PM set is tiny, so this table is diagnostic rather than decisive. The decisive comparison is the Binance-CV reliability plus the PM realized/incremental gate above.

## Decision

OD **CLOSES as a standalone strategy** in this time-boxed test. Arm B does not prove that OD fair-value richness adds robust incremental edge beyond the structural/source quote-selection baseline.

Arms C/D were not run because Arm B already decided the gate. That follows the Kronos discipline: do not escalate to HAR/Kronos when the model-free conditional probability does not reopen the standalone signal.

Clarification for future reopen prompts: the skipped Arms C/D here refer to the **conditional-probability task's** HAR/Kronos-style forward-vol arms. A separate pricing-model-form diagnostic may still run a small jump-model/Deribit extension, but it should be anchored on the captured 1s/LOB windows. In that framing, Merton/Kou jump diffusion is the cheap first extension; Bates/Variance-Gamma is only worth running if Merton/Kou leaves residual lower-CI-positive EV; Deribit is BTC/ETH-only and illustrative.

Operational next step: fold the source/richness information back into [[strat_market_making]] as a weak quote-selection or caution feature. Do not build new OD queue infrastructure from this result.

## Outputs

- Summary CSV: `data/analysis/csv_outputs/options_delta/od_conditional_prob_calibration_summary.csv`
- Calibration bins CSV: `data/analysis/csv_outputs/options_delta/od_conditional_prob_calibration_bins.csv`
- PM fills parquet: `data/analysis/od_conditional_prob_pm_fills.parquet`
- Binance history parquet: `data/analysis/od_conditional_prob_binance_history.parquet`
- Binance CV parquet: `data/analysis/od_conditional_prob_binance_cv.parquet`
