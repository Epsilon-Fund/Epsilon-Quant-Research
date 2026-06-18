---
title: "OD Replacement-Fair Sensitivity"
created: 2026-06-07
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
# OD Replacement-Fair Sensitivity

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior notes: [[od_strategy_a_v3_findings]] · [[od_strategy_a_v3_pnl_risk_findings]] · [[od_conditional_prob_calibration_findings]] · [[od_pricing_model_form_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Summary

- Scope: OD Replacement-Fair Sensitivity in the OD/options-delta area.
- Existing takeaway/status: Verdict: **does not improve OD**.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Headline

Verdict: **does not improve OD**.

This is the no-remake sensitivity: it reuses the existing 23-fill v4 far-|z| strict-rich short artifact and asks which fills would still pass if Strategy A used a better fair definition than the original RV physical probability. The sources are RV physical probability, Arm B empirical conditional probability, Merton, Kou, and Edgeworth.

The answer is mostly a hygiene confirmation, not a rescue. RV keeps 23/23 fills at the 1c rich-short gate. The best non-RV replacement by borrowed-baseline lower CI is `arm_b_empirical_conditional`, with 15/23 fills surviving and borrowed-baseline after-top3 lower CI -1.61c. The best row overall is `arm_b_empirical_conditional`, and its borrowed-baseline lower CI is -1.61c. None clear the standalone reopen bar.

Plain-English read: changing fair definitions changes which fills are still "rich," especially for Merton/Kou. But using better fair definitions does not create a clean, capacity-adjusted residual edge over the structural MM baseline. The branch remains useful as a quote-selection/caution feature, not as standalone OD.

## Scope

- Input: `data/analysis/od_pricing_model_form_pm_fills.parquet`.
- No upstream dataset remake: this script consumes the conditional/model-form fill artifact already written by the prior runs.
- Candidate set: 23 v4 far-|z| strict-source rich-short fills across 8 markets.
- Sizing: `flat_1_contract` and `replacement_edge_scaled`, where edge-scaled size is `clip(edge / 5c, 0.25, 3.0)` after the 1c/5c gate.
- PnL read: token-short resolution PnL, `entry_price - payoff + maker_rebate`, aggregated by market. The after-top3 lens multiplies by 5.00% and then compares to either 0c or the borrowed 1.98c v4 structural queue baseline.

## 1c Survival With Edge-Scaled Sizing

| fair source | fills survive | markets survive | mean edge | edge CI | mean size | total size | after-top3 net | after-top3 CI | borrowed-baseline CI lo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rv_physical_prob | 23/23 | 8/8 | 3.53c | [2.41c, 4.27c] | 0.71 | 16.28 | 1.35c | [0.01c, 3.30c] | -1.97c |
| arm_b_empirical_conditional | 15/23 | 6/8 | 6.10c | [3.37c, 7.22c] | 1.22 | 18.31 | 2.07c | [0.37c, 3.83c] | -1.61c |
| merton | 14/23 | 5/8 | 3.32c | [1.76c, 4.05c] | 0.67 | 9.33 | 0.59c | [-0.20c, 1.75c] | -2.18c |
| kou | 14/23 | 5/8 | 3.21c | [1.63c, 4.26c] | 0.64 | 9.01 | 0.42c | [-0.23c, 1.39c] | -2.21c |
| edgeworth | 22/23 | 8/8 | 3.58c | [2.61c, 4.47c] | 0.72 | 15.83 | 1.33c | [-0.03c, 3.25c] | -2.01c |

Surviving fill counts at the 1c gate: `arm_b_empirical_conditional` 15, `edgeworth` 22, `kou` 14, `merton` 14, `rv_physical_prob` 23.

## Full Summary

| fair | gate | sizing | fills | markets | edge | total size | market net | market net CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rv_physical_prob | original_set_repriced | flat_1_contract | 23 | 8 | 3.53c | 23.00 | 49.01c | [-2.63c, 134.77c] |
| rv_physical_prob | rich_ge_01c | flat_1_contract | 23 | 8 | 3.53c | 23.00 | 49.01c | [-2.62c, 133.53c] |
| rv_physical_prob | rich_ge_01c | replacement_edge_scaled | 23 | 8 | 3.53c | 16.28 | 27.02c | [0.25c, 68.11c] |
| rv_physical_prob | rich_ge_05c | flat_1_contract | 5 | 2 | 6.26c | 5.00 | 39.46c | [18.21c, 60.71c] |
| rv_physical_prob | rich_ge_05c | replacement_edge_scaled | 5 | 2 | 6.26c | 6.26 | 50.08c | [26.34c, 73.82c] |
| arm_b_empirical_conditional | original_set_repriced | flat_1_contract | 23 | 8 | 3.83c | 23.00 | 49.01c | [-2.20c, 135.09c] |
| arm_b_empirical_conditional | rich_ge_01c | flat_1_contract | 15 | 6 | 6.10c | 15.00 | 25.71c | [3.44c, 50.42c] |
| arm_b_empirical_conditional | rich_ge_01c | replacement_edge_scaled | 15 | 6 | 6.10c | 18.31 | 41.48c | [10.14c, 81.15c] |
| arm_b_empirical_conditional | rich_ge_05c | flat_1_contract | 10 | 4 | 7.85c | 10.00 | 38.75c | [16.19c, 61.32c] |
| arm_b_empirical_conditional | rich_ge_05c | replacement_edge_scaled | 10 | 4 | 7.85c | 15.69 | 61.48c | [24.76c, 110.69c] |
| merton | original_set_repriced | flat_1_contract | 23 | 8 | 1.03c | 23.00 | 49.01c | [-3.09c, 138.81c] |
| merton | rich_ge_01c | flat_1_contract | 14 | 5 | 3.32c | 14.00 | 15.17c | [-6.29c, 46.53c] |
| merton | rich_ge_01c | replacement_edge_scaled | 14 | 5 | 3.32c | 9.33 | 11.84c | [-3.85c, 33.63c] |
| merton | rich_ge_05c | flat_1_contract | 2 | 1 | 5.45c | 2.00 | 10.37c | [10.37c, 10.37c] |
| merton | rich_ge_05c | replacement_edge_scaled | 2 | 1 | 5.45c | 2.18 | 12.97c | [12.97c, 12.97c] |
| kou | original_set_repriced | flat_1_contract | 23 | 8 | 0.85c | 23.00 | 49.01c | [-2.27c, 135.80c] |
| kou | rich_ge_01c | flat_1_contract | 14 | 5 | 3.21c | 14.00 | 13.13c | [-6.30c, 40.47c] |
| kou | rich_ge_01c | replacement_edge_scaled | 14 | 5 | 3.21c | 9.01 | 8.47c | [-4.54c, 29.31c] |
| kou | rich_ge_05c | flat_1_contract | 4 | 2 | 5.40c | 4.00 | -0.68c | [-8.76c, 7.41c] |
| kou | rich_ge_05c | replacement_edge_scaled | 4 | 2 | 5.40c | 4.32 | -0.67c | [-9.50c, 8.16c] |
| edgeworth | original_set_repriced | flat_1_contract | 23 | 8 | 3.47c | 23.00 | 49.01c | [-1.57c, 135.44c] |
| edgeworth | rich_ge_01c | flat_1_contract | 22 | 8 | 3.58c | 22.00 | 49.02c | [-2.26c, 135.57c] |
| edgeworth | rich_ge_01c | replacement_edge_scaled | 22 | 8 | 3.58c | 15.83 | 26.66c | [0.38c, 64.24c] |
| edgeworth | rich_ge_05c | flat_1_contract | 5 | 2 | 6.15c | 5.00 | 39.46c | [18.21c, 60.71c] |
| edgeworth | rich_ge_05c | replacement_edge_scaled | 5 | 2 | 6.15c | 6.15 | 49.23c | [26.23c, 72.23c] |

## Interpretation

Use this as the Strategy A fair-source guardrail:

- `rv_physical_prob` is the original RV physical-probability fair; it is a control, not option-market IV fair.
- `arm_b_empirical_conditional` is the most model-free replacement fair. If this does not reopen OD, escalating model form should be treated skeptically.
- `merton` and `kou` add jump mass and generally reduce apparent far-tail richness.
- `edgeworth` can preserve more RV-like richness, but it still fails the MM-integrated borrowed-baseline lens here.

## Outputs

- Summary CSV: `data/analysis/csv_outputs/options_delta/od_replacement_fair_sensitivity_summary.csv`
- Fill parquet: `data/analysis/od_replacement_fair_sensitivity_fills.parquet`
