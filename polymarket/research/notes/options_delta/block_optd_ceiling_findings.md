---
title: OPT-D Directional Ceiling Search Findings
created: 2026-06-05
status: candidate
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
  - strat_options_delta
tags: [dali, optd, optuna, ceiling, in-sample, results]
---

# OPT-D Directional Ceiling Search Findings

> Hub: [[COWORK]] · [[strat_options_delta]]
> Table terms: [[polymarket_table_dictionary]]

## Summary

OPT-D searches an in-sample directional ceiling with Optuna over pooled Dali runs. The best trial has positive mean, but the confidence interval is wide and no trial among the minimum-size set has a lower CI above zero. The note is a candidate lead only, not a validated result or permission to trade.

## Headline

Best in-sample net-of-cost config: trial `319` with mean `18.6 bps`, CI `[-68.9 bps, 101.6 bps]`, n `46`, fill `0.2%`.

Verdict: **INCONCLUSIVE/POTENTIAL lead only: optimized IS ceiling does not close below zero**.

Among trials with n >= 30: `41` had positive mean, `0` had CI lower > 0, and `140` had CI upper < 0.

An in-sample positive is a lead, not a result. Only an optimized ceiling with CI upper < 0 would close the directional family rigorously.

## Best Config

- Params: `rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/300s, target_micro_price, family=Geopolitics, spread=wide_q50, depth=deep_q50, q=0.9`
- Raw entry fill rate: `0.2%`
- Unweighted mean PnL: `18.6 bps`
- Win rate: `47.8%`
- Mean hold: `201.4s`
- Markets executed: `5`
- Top exit reason: `fixed_timeout`

## Best By CI Lower

Trial `327`: `0.3 bps` CI `[0.0 bps, 0.9 bps]`, n `48`, params `rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/900s, target_micro_price, family=Geopolitics, spread=tight_q20, depth=deep_q50, q=0.9`.

## Top Trials

| trial | mean | CI | n | fill | sharpe | params |
| --- | --- | --- | --- | --- | --- | --- |
| 319 | 18.6 bps | [-68.9 bps, 101.6 bps] | 46 | 0.2% | 0.42 | rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/300s, target_micro_price, family=Geopol. |
| 327 | 0.3 bps | [0.0 bps, 0.9 bps] | 48 | 0.1% | 1.00 | rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/900s, target_micro_price, family=Geopol. |
| 330 | 0.3 bps | [0.0 bps, 0.9 bps] | 48 | 0.1% | 1.00 | rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/900s, target_micro_price, family=Geopol. |
| 315 | 0.3 bps | [0.0 bps, 0.9 bps] | 49 | 0.1% | 1.00 | rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/900s, target_micro_price, family=Geopol. |
| 318 | 0.3 bps | [0.0 bps, 0.9 bps] | 49 | 0.1% | 1.00 | rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/900s, target_micro_price, family=Geopol. |
| 324 | 0.3 bps | [0.0 bps, 0.9 bps] | 49 | 0.1% | 1.00 | rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/900s, target_micro_price, family=Geopol. |
| 325 | 0.3 bps | [0.0 bps, 1.2 bps] | 49 | 0.1% | 1.00 | rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/900s, target_micro_price, family=Geopol. |
| 342 | 0.3 bps | [0.0 bps, 0.9 bps] | 50 | 0.1% | 1.00 | rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/900s, target_micro_price, family=Geopol. |
| 286 | 0.2 bps | [0.0 bps, 0.7 bps] | 65 | 0.3% | 1.00 | rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/300s, target_micro_price, family=Geopol. |
| 107 | 0.2 bps | [0.0 bps, 0.9 bps] | 67 | 0.2% | 1.00 | rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/300s, target_micro_price, family=Geopol. |
| 241 | 0.2 bps | [0.0 bps, 0.6 bps] | 67 | 0.2% | 1.00 | rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/300s, target_micro_price, family=Geopol. |
| 294 | 0.2 bps | [0.0 bps, 0.8 bps] | 68 | 0.2% | 1.00 | rrank_weighted_mid_edge_bps, continuous_rank_sizing, fade, passive_maker, fixed/300s, target_micro_price, family=Geopol. |

## Data

| run | rows loaded | markets |
| --- | --- | --- |
| a0 | 1,399,032 | 12 |
| a0b | 1,698,775 | 9 |
| a0c | 4,254,020 | 17 |
| a0c_roll | 2,445,782 | 38 |

- Valid quote rows searched: `9,747,343`.
- Optuna trials: `350` with TPE sampler.
- Candidate cap applied in `2.9%` of trials; capped trials keep the strongest `250,000` signal rows before non-overlap.

## Method

- Pooled `a0`, `a0b`, `a0c`, and `a0c_roll` as one in-sample set; no holdout and no train/test split.
- Signal set: OFI L1, OFI 5s, TOB imbalance level, weighted-mid edge, and 300s rolling-rank variants of each.
- Search dimensions include mapping, direction, fixed/boundary horizons, exits, spread/depth/family gates, taker versus passive maker execution, and passive fill window.
- Event ordering uses exchange timestamps when present, falling back to receive timestamps.
- Quote states require complete books and `book_staleness_seconds <= 5`.
- PnL is net of `FEE_BY_CATEGORY`; passive maker entries use the A1.4h/P2 trade-through fill proxy and maker rebate.
- Non-overlap is enforced per market. CI bars use 300s clock-block bootstrap over executed trade PnL.

## Outputs

- `data/analysis/csv_outputs/options_delta/optd_ceiling_search.csv`
- `notes/options_delta/block_optd_ceiling_findings.md`
