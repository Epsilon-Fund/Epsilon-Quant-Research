---
title: Block K2 Logit-Space Quoting Findings
created: 2026-06-05
status: closed
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
  - strat_market_making
tags: [dali, block-k2, logit-as, optuna, maker-sim]
---

# Block K2 Logit-Space Quoting Findings

> **Strat:** [[strat_market_making]] (Market-Making). Sibling: [[strat_options_delta]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Summary

K2 tests an optimized logit-space Avellaneda-Stoikov maker ceiling on pooled in-sample data. Both the zero-skew and small-contrarian stages have confidence intervals entirely below zero after modeled fills and exit costs. Under the pre-commit rule, this kills the maker thesis on that universe.

## Headline

The optimized logit-space A-S ceiling is negative in pooled IS. Both zero-skew and small-contrarian stages have 95% CIs entirely below zero, so the K2 pre-commit rule kills the maker thesis on this universe.

Stage B's fixed small contrarian fade has incremental total PnL of 15764.4 bps units and incremental mean PnL of 18.1 bps versus stage A. Pre-commit gate result: **maker thesis dead on this universe**.

## Selected Optuna Fits

| stage | gamma | base half-spread | cap | widen | fills | mean | 95% CI | Sharpe-like | clears |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero_skew | 0.49336 | 397.0 bps | 1 | 0.3317 | 809 | -1126.5 bps | [-1499.4 bps, -747.8 bps] | -6.31 | no |
| contrarian | 0.47287 | 500.0 bps | 2 | 0.1633 | 808 | -1108.3 bps | [-1511.9 bps, -675.2 bps] | -6.05 | no |

## Category Breakdown

| stage | segment | category | fills | mean | 95% CI | Sharpe-like | clears |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contrarian | fee_enabled | Crypto | 779 | -1092.2 bps | [-1517.8 bps, -686.5 bps] | -5.78 | no |
| contrarian | fee_enabled | Finance | 1 | -265.3 bps | [n/a, n/a] | n/a | no |
| contrarian | fee_enabled | Sports | 17 | -2554.7 bps | [-4664.6 bps, -806.3 bps] | -2.93 | no |
| contrarian | fee_free | Geopolitics | 11 | -93.2 bps | [-239.0 bps, 88.7 bps] | -0.90 | no |
| zero_skew | fee_enabled | Crypto | 766 | -1141.1 bps | [-1535.9 bps, -727.7 bps] | -6.08 | no |
| zero_skew | fee_enabled | Finance | 1 | -369.3 bps | [n/a, n/a] | n/a | no |
| zero_skew | fee_enabled | Sports | 16 | -2202.2 bps | [-4026.5 bps, -888.3 bps] | -2.72 | no |
| zero_skew | fee_enabled | Tech | 1 | -109.5 bps | [n/a, n/a] | n/a | no |
| zero_skew | fee_free | Geopolitics | 25 | -60.9 bps | [-254.7 bps, 120.4 bps] | -0.80 | no |

## Method

- Data: `data/analysis/block_a1_features.parquet`, pooled `a0, a0b, a0c, a0c_roll` as one in-sample set. No holdout split.
- Exclusion: slugs containing `will-jd-vance`.
- Candidate fills: A1.4h-style passive proxy with a 5s quote freshness window. A bid fills only on a real SELL print at or below our modeled bid; an ask fills only on a real BUY print at or above our modeled ask.
- Logit A-S quote: `x=logit(p)`, `rx=x-q*gamma*sigma^2*tau`, half-spread in logit units is a probability-floor-derived base term plus `0.5*gamma*sigma^2*tau`, then bid/ask are mapped back with sigmoid.
- Spread floor: optimized `base_spread_bps`, interpreted as displayed half-spread floor in bps of entry mid.
- Inventory cap: optimized integer cap, per asset. Quotes that would add beyond the cap are disabled.
- Crypto-4h widening: optimized `widening_strength * near_50(p) * near_resolution(tau)` added to the logit half-spread for `crypto_4h` markets.
- Volatility: rolling 300s logit-mid realized variance per second from quote-state updates.
- Exits: each fill is closed as taker after 60s or before resolution minus a 10s buffer. Taker exit fee is charged. Entry maker rebate uses the K1 fee table.
- Stage A: no directional skew.
- Stage B: same parameter search but with a fixed small contrarian skew of 0.03 logit units when the A1 current TOB signal is in its market-level top absolute decile.
- Objective: pooled fill-level `mean(net_bps) / std(net_bps) * sqrt(n)`, with fewer than 30 fills penalized.
- CI: 500 bootstrap resamples over contiguous 300s fill-time blocks within market.

## Interpretation

Unlike K1's generous midpoint spread-capture decomposition, the logit A-S quote has to earn fills away from mid and then pay taker exit costs. That optimized ceiling is still below zero. The small contrarian fade is mildly less bad, but it does not rescue the thesis.

The simulator is still generous: no queue priority, no partial fill sizing, no quote cancellation latency, and one-unit fills. Because even this optimized ceiling is negative, adding realistic queue and latency costs would only worsen it.

Inputs precomputed: 24,586 candidate trade rows across 56 markets.

CSV: `data/analysis/csv_outputs/market_making/k2_quoting_sim.csv`.
