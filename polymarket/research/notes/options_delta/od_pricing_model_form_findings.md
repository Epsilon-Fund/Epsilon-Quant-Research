# OD Pricing-Model-Form Test: Do Jumps Explain The Far-OTM Longshot Result?

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior notes: [[od_v4_calibration_gate_findings]] · [[od_v4_queue_replay_findings]] · [[od_conditional_prob_calibration_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Headline

Final verdict: **CLOSE remains**.

This run tested the prompt's model-form hypothesis: maybe the old Gaussian `N(z)` digital underpriced short-dated OTM tail probability because it had no jump mass. The test repriced the same v4 far-|z| strict-rich short set with causal jump parameters from the captured 1s Binance/K3 panel, then checked whether any residual OD model edge survives the v4 structural baseline.

It does **not** reopen OD. The live 1s jump and higher-moment models do add tail mass in the OTM shape diagnostic, but on the actual 23-fill PM set they do not create a deployable incremental edge. Merton original-set model edge is 1.03c, CI [-2.84c, 2.59c]; Kou original-set model edge is 0.85c, CI [-3.39c, 2.53c]; Edgeworth higher-moment model edge is 3.47c, CI [2.40c, 4.31c]. The best after-top3 incremental lower CI across tested pricing-model rows is -1.07c, and Deribit's best illustrative after-top3 lower CI is -2.45c, so the branch still fails the structural-baseline bar.

Plain-English read: jumps make the tail model more honest, but they do not turn the OD rich-short signal into a standalone trade. The apparent upside is still explained better as source/structure/queue selection with tiny capacity, not a distinct pricing-model edge.

## What Changed Versus The 5-Minute Conditional-Probability Test

The prior [[od_conditional_prob_calibration_findings]] note used years of Binance 5-minute data as a broad truth table. This script keeps that result as background, but the validation here is anchored on the captured windows:

- **1s Binance/K3 panel:** `data/analysis/cache/k3v3h_panel_features.parquet`, covering all 23 PM fills.
- **Polymarket LOB/WS capture:** `block_a0c_roll_features` plus `block_a0c_features`, covering 95.65% of PM fills.
- **Jump detection:** Lee-Mykland-style standardized 1s returns, flagged when `|r_1s| > 8.0 * sigma_1s` and at least 10.0 bps in one second, with BNS-style 5-minute realized-minus-bipower jump ratio reported as a diagnostic.
- **Deribit:** public BTC/ETH DVOL pulled as an external IV anchor. This fetched 110 hourly DVOL rows. It is illustrative only because it is BTC/ETH-only, DVOL is a 30-day index, and the local artifacts do not contain a clean historical Deribit per-option IV surface for the PM fills.

The 5-minute history is the prior/control. The captured 1s panel is the actual live-window test.

## Design

Unit of observation for the PM table is one v4 far-|z| strict-rich short fill. The model asks: if we short the token at price `p`, what does each model think the token's probability of paying `$1` is?

```text
model edge = short price - model P(token pays $1)
realized EV = short price - realized payoff + maker rebate
```

The arms:

- **Arm A, Gaussian control:** the old EWMA `N(z)` digital.
- **Arm B, Merton:** compound-Poisson normal jumps fitted causally from prior captured 1s Binance returns.
- **Arm B, Kou-style:** asymmetric up/down exponential jump moments, also fitted causally from prior captured 1s returns. This is a moment-matched Kou-style approximation, used because the repo environment has NumPy/Pandas but not SciPy for full closed-form calibration.
- **Arm C, higher moments:** run as a cheap higher-moment / Edgeworth extension. Full Bates or calibrated VG were still not attempted because the environment lacks SciPy/Arch/Torch and the PM validation set is only 23 fills, but this arm directly tests whether causal 1s skew/kurtosis changes the OTM digital probability enough to reopen the gate.
- **Arm D, Deribit DVOL:** BTC/ETH-only illustrative anchor. It is now reported with the same MM-incremental columns, but it is still not gate-grade because DVOL is a 30-day index rather than a historical 4h option surface.

CI columns are market-cluster bootstraps. `incremental vs structural` applies the K5 non-incumbent 5% capacity haircut and subtracts the best v4 structural queue baseline of 1.98c per market. That is the MM integration check: a pricing model can have positive raw realized EV and still fail if, after realistic non-incumbent capacity, it does not beat the already-known MM/structural quote-selection result.

## PM Far-|z| Short Set Results

| arm / subset | fills | markets | price | model P(ITM) | realized ITM | model edge | edge CI | realized net EV | realized CI | incremental vs structural | incremental CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_a_ewma_nd2_original_set | 23 | 8 | 56.05c | 52.53% | 39.13% | 3.53c | [2.49c, 4.34c] | 17.05c | [-1.50c, 26.62c] | 0.47c | [-2.08c, 4.84c] |
| arm_b_merton_live1s_original_set | 23 | 8 | 56.05c | 55.02% | 39.13% | 1.03c | [-2.84c, 2.59c] | 17.05c | [-1.50c, 26.62c] | 0.47c | [-2.08c, 4.84c] |
| arm_b_kou_live1s_original_set | 23 | 8 | 56.05c | 55.20% | 39.13% | 0.85c | [-3.39c, 2.53c] | 17.05c | [-1.50c, 26.62c] | 0.47c | [-2.08c, 4.84c] |
| arm_c_edgeworth_higher_moment_original_set | 23 | 8 | 56.05c | 52.59% | 39.13% | 3.47c | [2.40c, 4.31c] | 17.05c | [-1.50c, 26.62c] | 0.47c | [-2.08c, 4.84c] |
| arm_b_merton_live1s_rich_ge_1c | 14 | 5 | 55.31c | 51.99% | 50.00% | 3.32c | [1.78c, 4.06c] | 5.42c | [-4.49c, 9.57c] | -1.22c | [-2.29c, 0.35c] |
| arm_b_kou_live1s_rich_ge_1c | 14 | 5 | 61.73c | 58.52% | 57.14% | 3.21c | [1.64c, 4.23c] | 4.69c | [-3.94c, 9.36c] | -1.32c | [-2.30c, 0.04c] |
| arm_c_edgeworth_higher_moment_rich_ge_1c | 22 | 8 | 54.06c | 50.48% | 36.36% | 3.58c | [2.63c, 4.33c] | 17.83c | [-1.68c, 26.75c] | 0.47c | [-2.08c, 4.84c] |
| arm_b_merton_live1s_rich_ge_5c | 2 | 1 | 55.05c | 49.60% | 50.00% | 5.45c | [5.45c, 5.45c] | 5.18c | [5.18c, 5.18c] | -1.46c | [-1.46c, -1.46c] |
| arm_b_kou_live1s_rich_ge_5c | 4 | 2 | 74.56c | 69.16% | 75.00% | 5.40c | [5.39c, 5.42c] | -0.34c | [-8.76c, 2.47c] | -2.01c | [-2.42c, -1.61c] |
| arm_c_edgeworth_higher_moment_rich_ge_5c | 5 | 2 | 15.60c | 9.45% | 0.00% | 6.15c | [5.89c, 7.20c] | 15.78c | [15.18c, 18.21c] | -0.01c | [-1.07c, 1.06c] |

Read: a positive `model edge` means the model says the token is overpriced at our short price. `realized net EV` is what happened on this tiny PM sample. `incremental vs structural` is the MM integration check. To reopen OD, the residual after the top-maker haircut also had to beat the structural queue baseline with lower-CI > 0. It does not.

![PM model edge](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_pricing_model_form_pm_edge.png)

## Captured-Window 1s Reliability

This table uses the K3 captured 1s panel downsampled to one row per minute per market, not the multi-year 5-minute history. It is still overlapping within a 4h window, so read it as a live-window calibration diagnostic rather than a powered OOS gate.

| arm | probability bucket | rows | markets | mean pred | observed | obs - pred |
| --- | --- | --- | --- | --- | --- | --- |
| arm_a_ewma_nd2 | 0_5c | 409 | 17 | 1.46% | 14.67% | 13.21% |
| arm_a_ewma_nd2 | 5_10c | 153 | 16 | 7.69% | 49.02% | 41.32% |
| arm_a_ewma_nd2 | 10_25c | 592 | 17 | 17.67% | 40.54% | 22.87% |
| arm_a_ewma_nd2 | 75_90c | 416 | 14 | 82.59% | 45.67% | -36.91% |
| arm_a_ewma_nd2 | 90_95c | 143 | 10 | 92.36% | 51.75% | -40.61% |
| arm_a_ewma_nd2 | 95_100c | 405 | 10 | 98.70% | 90.62% | -8.09% |
| arm_b_merton_live1s | 0_5c | 476 | 17 | 1.62% | 19.75% | 18.12% |
| arm_b_merton_live1s | 5_10c | 233 | 16 | 7.60% | 57.94% | 50.34% |
| arm_b_merton_live1s | 10_25c | 667 | 18 | 17.70% | 54.72% | 37.02% |
| arm_b_merton_live1s | 75_90c | 454 | 16 | 81.16% | 40.53% | -40.63% |
| arm_b_merton_live1s | 90_95c | 100 | 9 | 92.57% | 85.00% | -7.57% |
| arm_b_merton_live1s | 95_100c | 321 | 9 | 98.47% | 97.20% | -1.28% |
| arm_b_kou_live1s | 0_5c | 434 | 17 | 1.73% | 18.20% | 16.48% |
| arm_b_kou_live1s | 5_10c | 224 | 16 | 7.45% | 48.21% | 40.76% |
| arm_b_kou_live1s | 10_25c | 671 | 18 | 17.81% | 59.61% | 41.80% |
| arm_b_kou_live1s | 75_90c | 427 | 16 | 81.28% | 42.62% | -38.66% |
| arm_b_kou_live1s | 90_95c | 109 | 9 | 92.49% | 86.24% | -6.25% |
| arm_b_kou_live1s | 95_100c | 289 | 9 | 98.25% | 96.89% | -1.36% |
| arm_c_edgeworth_higher_moment | 0_5c | 409 | 17 | 1.44% | 14.67% | 13.23% |
| arm_c_edgeworth_higher_moment | 5_10c | 153 | 16 | 7.67% | 49.02% | 41.35% |
| arm_c_edgeworth_higher_moment | 10_25c | 588 | 17 | 17.65% | 40.48% | 22.83% |
| arm_c_edgeworth_higher_moment | 75_90c | 416 | 14 | 82.58% | 45.67% | -36.90% |
| arm_c_edgeworth_higher_moment | 90_95c | 149 | 10 | 92.40% | 51.01% | -41.40% |
| arm_c_edgeworth_higher_moment | 95_100c | 399 | 10 | 98.75% | 91.48% | -7.28% |

Read: the live-window panel says the jump models move probabilities, but they do not reveal a clean PM-specific residual edge. The important thing is that this is now using the same captured windows and live-state granularity as the actual fills.

## Delta, Gamma, And Tail Shape

The prompt's mechanism question is whether jumps change the digital shape, not just the level of volatility. The table below uses a representative PM horizon and sigma, then compares probability mass and numerical Greeks around the strike.

| arm | lambda/4h | P_up z=-2 | P_up z=-1.5 | P_up z=0 | delta | gamma | avg OTM tail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian | 3.890 | 2.28% | 6.68% | 50.00% | 98.04 | -98.02 | 6.68% |
| merton | 3.890 | 2.11% | 6.18% | 47.39% | 95.89 | 1310.33 | 7.14% |
| kou | 3.890 | 2.29% | 6.51% | 47.53% | 94.32 | 1104.33 | 7.52% |
| edgeworth | 3.890 | 2.17% | 6.59% | 50.25% | 97.85 | -546.93 | 6.69% |

`P_up z=-2` and `P_up z=-1.5` are OTM-up probabilities. `avg OTM tail` averages the two symmetric far-tail directions. A bigger value means the model assigns more chance to a far OTM token finishing ITM.

![Tail shape](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_pricing_model_form_tail_shape.png)

Read: the jump forms do what they are supposed to do mechanically: they redistribute some mass into the OTM tail and change the near-strike Greek shape. That is useful knowledge, but the PM residual-EV gate still fails.

## Deribit Anchor

Deribit is BTC/ETH only. The public historical anchor used here is DVOL, extrapolated down to the PM horizon by using the annualized DVOL level inside the same digital formula. That is deliberately labeled illustrative: it is a 30-day options-market IV index, not a 4h binary-resolution surface, and SOL has no Deribit analogue.

| row | n | markets | price | Deribit P/fair | edge/diff | CI / p95 | realized net EV / read | incremental vs structural | incremental CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_d_deribit_dvol_btc_eth_original_set | 15 | 5 | 55.23c | 51.89% | 3.33c | [-2.78c, 5.06c] | 22.04c | 1.33c | [-2.45c, 8.25c] |
| arm_d_deribit_dvol_btc_eth_rich_ge_1c | 8 | 3 | 32.67c | 22.11% | 10.56c | [6.47c, 11.68c] | 7.82c | -0.94c | [-2.57c, 2.18c] |
| daily_24h_deribit_illustrative_btc | 82801 | n/a | n/a | 14.38% | 1.18c | p95 abs 8.86c | illustrative | illustrative | illustrative |
| daily_24h_deribit_illustrative_eth | 82801 | n/a | n/a | 11.35% | 0.55c | p95 abs 11.86c | illustrative | illustrative | illustrative |

![Deribit anchor](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_pricing_model_form_deribit.png)

Read: Deribit is helpful as a sanity anchor, not a decision gate. The BTC/ETH Deribit-rich subset has a positive model edge, but it is only 8 fills / 3 markets and its structural-incremental CI is still not a deployable lower-CI-positive OD result. A real Deribit option-surface comparison would need historical per-instrument IV/mark snapshots aligned to the PM fills; this run only uses the public DVOL index plus the local captured PM/Binance rows.

## Decision

OD stays **closed as a standalone strategy**. The pricing-model-form hypothesis is informative but not enough: Merton/Kou jump-aware pricing and the higher-moment Edgeworth extension do not leave a lower-CI-positive residual that beats the structural queue baseline, and the Deribit anchor is too small/indirect to reopen the branch.

Recommended routing: fold the useful pieces back into [[strat_market_making]] as weak quote-selection features. In practice that means: avoid pretending far-|z| Gaussian richness is alpha by itself; prefer source-clean, liquid, queue-realistic cells; and use live 1s jump/OFI flags as caution filters around tail states.

## Outputs

- Summary CSV: `data/analysis/csv_outputs/options_delta/od_pricing_model_form_summary.csv`
- Captured-window reliability CSV: `data/analysis/csv_outputs/options_delta/od_pricing_model_form_reliability.csv`
- Greek/tail-shape CSV: `data/analysis/csv_outputs/options_delta/od_pricing_model_form_shape.csv`
- Deribit CSV: `data/analysis/csv_outputs/options_delta/od_pricing_model_form_deribit.csv`
- PM fill parquet: `data/analysis/od_pricing_model_form_pm_fills.parquet`
- Live 1s panel parquet: `data/analysis/od_pricing_model_form_live1s_panel.parquet`
