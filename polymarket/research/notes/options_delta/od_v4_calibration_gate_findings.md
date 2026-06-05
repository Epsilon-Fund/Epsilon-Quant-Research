# OD v4 Calibration Gate: Is The Rich Longshot Signal Real Enough To Build Queue Replay?

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior note: [[od_strategy_a_v3_pnl_risk_findings]]
> MM benchmark notes: [[mm_deployable_cells_findings]] · [[block_k5_stress_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Phase 0 Verdict

Phase 0 **FAIL**. The full-panel strict-rich far-|z| short set has 23 fills across 8 markets. Mean short price is 56.05c, realized ITM is 39.13%, gross EV `p - P_itm` is 16.92c with market-cluster CI [-1.43c, 26.38c]; net of maker rebate it is 17.05c, CI [-1.49c, 26.80c].

Phase 1 queue-aware replay was **not run**. The calibration/EV signal is too small, too concentrated, and not lower-CI positive enough to justify building execution infra.

The largest positive market is `btc-updown-4h-1780056000` with 11 fills and 338.85c net PnL. It contributes 86.42% of total net PnL.

Leave-one-market-out sensitivity: the weakest remaining mean gross EV is 4.33c after dropping `2378289`.

Plain-English read: the scary `39.13%` realized ITM number is not literally "we shorted sub-39c longshots that hit 39% of the time." This set mixes low-price longshot shorts and high-price rich-token shorts. But the gate still does not green-light infra: the full-panel sample is only 23 fills / 8 markets, the confidence interval is wide, and the profit is highly concentrated in one BTC window.

## What This Gate Tests

The primitive economics of a short binary token are:

```text
short EV before rebate = short price p - probability token pays $1
net realized short PnL = p - realized_payoff + maker_rebate
```

If `p - P_itm` is not convincingly positive on the full panel, a queue-aware replay cannot rescue the OD thesis. Queue replay can improve fill realism; it cannot turn a negative-resolution bet into an edge.

## Overall EV And Calibration

| label | fills | markets | mean_short_price | mean_pred_itm_prob | realized_itm_rate | mean_model_edge | mean_gross_ev | gross_ev_ci_lo | mean_net_ev | net_ev_ci_lo | trailing_far_rich_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_far_strict_rich_short | 23 | 8 | 56.05c | 52.53% | 39.13% | 3.53c | 16.92c | [-1.43c, 26.38c] | 17.05c | [-1.49c, 26.80c] | 91.30% |
| trailing_vol_confirmed_far_strict_rich | 21 | 7 | 56.73c | 53.10% | 38.10% | 3.62c | 18.63c | [0.30c, 28.69c] | 18.75c | [0.09c, 28.81c] | 100.00% |

`mean_pred_itm_prob` is the OD model's probability that the token we shorted pays $1. `realized_itm_rate` is how often it actually paid. `mean_model_edge` is the edge claimed at entry (`price - predicted probability`). `mean_gross_ev` is the realized primitive short EV (`price - realized payoff`). The CI is market-clustered so repeated fills inside one 4h market do not create fake precision.

![Calibration curve](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_v4_calibration_curve.png)

The diagonal is perfect calibration. Points above the diagonal mean the token paid more often than OD predicted, which is bad for a short; points below mean OD overpredicted ITM.

| predicted probability bin | fills | markets | mean pred | observed freq | obs - pred | mean price | mean gross EV | EV CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5_10c | 7 | 2 | 8.62% | 0.00% | -8.62c | 13.60c | 13.60c | [13.00c, 13.70c] |
| 10_25c | 4 | 3 | 12.32% | 0.00% | -12.32c | 17.00c | 17.00c | [14.00c, 18.00c] |
| 75_90c | 5 | 3 | 86.40% | 40.00% | -46.40c | 88.63c | 48.63c | [-12.00c, 88.00c] |
| 90_95c | 3 | 2 | 92.00% | 100.00% | 8.00c | 95.03c | -4.97c | [-6.00c, -4.45c] |
| 95_100c | 4 | 1 | 97.64% | 100.00% | 2.36c | 99.45c | -0.55c | [-0.55c, -0.55c] |

## Short-Price Buckets

This is the bucket table that resolves the `39%` confusion. Low-price shorts and high-price shorts are different bets. Shorting a 14c token that pays 0 has very different risk from shorting a 99c token that pays 1.

![EV by short price](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_v4_ev_by_short_price.png)

| short price bucket | fills | markets | mean price | predicted ITM | realized ITM | gross EV | EV CI | net EV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p_lt_10c | 1 | 1 | 10.00c | 7.45% | 0.00% | 10.00c | [10.00c, 10.00c] | 10.13c |
| p_10_25c | 10 | 4 | 15.32c | 10.21% | 0.00% | 15.32c | [14.37c, 17.00c] | 15.50c |
| p_75_90c | 4 | 2 | 88.00c | 86.36% | 25.00% | 63.00c | [-12.00c, 88.00c] | 63.15c |
| p_90_100c | 8 | 4 | 96.75c | 94.14% | 100.00% | -3.25c | [-7.43c, -1.11c] | -3.20c |

Read: the low-price buckets are positive in this tiny sample because those tokens did not resolve ITM. The high-price buckets can still be positive in cents if price was above realized rate, but they are not "longshot shorts"; they are rich high-probability token shorts, economically equivalent to buying cheap complements.

## Is The `far-|z|` Bucket A Vol-Sizing Artifact?

`abs_z` is computed from Binance moneyness divided by causal EWMA vol and time left. If EWMA vol is too low, ordinary states can be mislabeled as far-|z| longshots. As a cheap causal proxy for the requested HAR/Kronos question, this gate recomputes the same classification using trailing realized vol already in the panel. This is not a full HAR/Kronos bake-off; it is only an artifact screen.

| abs_z bucket | fills | markets | EWMA abs_z | trailing abs_z | realized ITM | gross EV | EV CI | trailing far/rich share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| z_1_1p25 | 8 | 5 | 1.110 | 1.152 | 25.00% | 37.14c | [-4.12c, 70.50c] | 87.50% |
| z_1p25_1p5 | 11 | 4 | 1.369 | 1.309 | 27.27% | 8.57c | [-1.00c, 13.50c] | 90.91% |
| z_1p5_2 | 2 | 1 | 1.832 | 1.739 | 100.00% | -1.00c | [-1.00c, -1.00c] | 100.00% |
| z_2_3 | 2 | 1 | 2.235 | 2.250 | 100.00% | -0.10c | [-0.10c, -0.10c] | 100.00% |

The trailing-vol proxy keeps 21 of 23 fills as far and rich. Its gross EV is 18.63c, CI [0.30c, 28.69c]. That does not meet the prompt's condition for proceeding: no corrected forward-vol model here flips the gate into a lower-CI-positive result, and the actual HAR/Kronos path model remains gated rather than run.

## Concentration / Small-Sample Diagnosis

| market_id | asset | fills | mean price | predicted ITM | realized ITM | gross EV | net PnL | PnL share | trailing far/rich share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2378289 | BTC | 11 | 48.85c | 44.96% | 18.18% | 30.66c | 338.85c | 86.42% | 90.91% |
| 2379681 | SOL | 3 | 16.33c | 12.47% | 0.00% | 16.33c | 49.57c | 12.64% | 100.00% |
| 2380737 | ETH | 1 | 18.00c | 10.77% | 0.00% | 18.00c | 18.21c | 4.64% | 100.00% |
| 2384176 | SOL | 1 | 14.00c | 10.28% | 0.00% | 14.00c | 14.17c | 3.61% | 100.00% |
| 2378275 | SOL | 4 | 99.45c | 97.64% | 100.00% | -0.55c | -2.17c | -0.55% | 100.00% |
| 2378276 | ETH | 1 | 94.00c | 92.49% | 100.00% | -6.00c | -5.92c | -1.51% | 100.00% |
| 2379677 | BTC | 1 | 91.13c | 86.53% | 100.00% | -8.87c | -8.76c | -2.23% | 100.00% |
| 2364426 | BTC | 1 | 88.00c | 85.53% | 100.00% | -12.00c | -11.85c | -3.02% | 0.00% |

The `net PnL share` column is why this remains a gate fail even though the fill-weighted point estimate is positive. A narrow Phase 1 queue replay would mostly be replaying whether we could capture the same concentrated windows, not proving a broad OD mispricing.

## Decision

Phase 1 queue-aware replay is skipped in this run. The OD richness signal remains useful as a **selection feature** to carry back into MM/source-filtered lifecycle analysis, but it is not strong enough as a standalone longshot-EV edge to justify new execution infrastructure.

Required condition to reopen Phase 1: a corrected causal fair-value model, preferably HAR-RV or Kronos forward paths per [[2026-05-31_kronos_hermes_eval]], must show `p - P_itm` lower-CI > 0 on the full far-|z| strict-rich set or a pre-registered sub-bucket with enough independent markets.

## Outputs

- CSV summary: `data/analysis/csv_outputs/options_delta/od_v4_calibration_gate_summary.csv`
- CSV calibration bins: `data/analysis/csv_outputs/options_delta/od_v4_calibration_gate_bins.csv`
- Contract parquet: `data/analysis/od_v4_calibration_gate_contracts.parquet`

## User-Override Exploratory Queue Replay

After the official Phase 0 fail, an exploratory Phase 1 queue replay was run by user override. That run is documented in [[od_v4_queue_replay_findings]]. It does not change the Phase 0 gate verdict; it only shows what execution would look like if the branch were pursued anyway.
