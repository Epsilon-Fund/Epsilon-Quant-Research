---
title: Block K3 v3h2 Persistence-Gated Findings
created: 2026-06-05
status: closed
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
  - strat_options_delta
tags:
  - options-delta
  - block-k
  - persistence-gate
  - hedged-basis
  - research
---

# Block K3 v3h2 Persistence-Gated Findings

> **Strat:** [[strat_options_delta]] (Options-Delta). Sibling: [[strat_market_making]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

Generated: 2026-05-31T01:50:20Z

## Summary

K3 v3h2 is the persistence-gated falsifier for the moderate-delta hedged RV idea. It restricts entries to mid-sized dynamic gaps, requires same-signed persistence, applies strict source filters, and excludes far/late/tail rows. The best hedged and naked regimes remain negative after costs, closing this rescue variant.

## Headline

The moderate-delta persistence-gated hedged RV does not clear zero after costs. Best hedged regime is still strictly negative: latency 2s, entry 0.35, exit 0.10, k=30s, max_hold=300s, mean -17.33c CI [-18.19c, -13.64c].

Best naked counterpart: latency 10s, entry 0.50, exit 0.25, k=5s, max_hold=300s, mean -8.15c CI [-10.25c, -2.82c].

Selected hedged hold distribution: median 66.0s, p90 172.2s, p95 190.8s. Across all H2 trades, median hold is 37.0s and p95 is 177.0s. This is minute-scale on the selected median.

This is the intended falsifier for K3 v3-H: it **only** enters `mid_absz_0.25_1|mid_30m_2h` (`|z|` in [0.25, 1.00], tau in 30m-2h), requires the same-signed dynamic logit gap to persist for k in (5, 15, 30), excludes far/late/tail rows entirely, applies the strict source-basis filter, and excludes large static basis at entry. Costs include Polymarket taker on entry+exit, Binance hedge turnover at 6.0bp, and funding at 1.0bp per 8h.

## Best Hedged Regimes

| lat | entry | exit | k_s | max_s | trades | mkts | hedged_mean | hedged_CI | naked_mean | naked_CI | med_hold | p95_hold | conv | top_mkt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.35 | 0.10 | 30 | 300 | 14 | 4 | -17.33c | [-18.19c, -13.64c] | -8.18c | [-11.24c, -6.01c] | 66.0 | 190.8 | 21.43% | 57.14% |
| 2 | 0.35 | 0.10 | 30 | 600 | 14 | 4 | -17.33c | [-18.19c, -13.64c] | -8.18c | [-11.24c, -6.01c] | 66.0 | 190.8 | 21.43% | 57.14% |
| 2 | 0.35 | 0.25 | 30 | 300 | 14 | 4 | -17.35c | [-18.19c, -13.79c] | -8.32c | [-11.24c, -6.65c] | 64.5 | 190.8 | 21.43% | 57.14% |
| 2 | 0.35 | 0.25 | 30 | 600 | 14 | 4 | -17.35c | [-18.19c, -13.79c] | -8.32c | [-11.24c, -6.65c] | 64.5 | 190.8 | 21.43% | 57.14% |
| 2 | 0.35 | 0.10 | 30 | 180 | 15 | 4 | -17.38c | [-18.24c, -14.24c] | -8.48c | [-11.51c, -5.46c] | 57.0 | 176.4 | 20.00% | 53.33% |
| 2 | 0.35 | 0.25 | 30 | 180 | 15 | 4 | -17.40c | [-18.24c, -14.36c] | -8.60c | [-11.51c, -6.27c] | 54.0 | 176.4 | 20.00% | 53.33% |
| 1 | 0.35 | 0.10 | 30 | 300 | 14 | 4 | -17.56c | [-18.49c, -13.84c] | -8.54c | [-11.24c, -6.08c] | 66.0 | 190.8 | 21.43% | 57.14% |
| 1 | 0.35 | 0.10 | 30 | 600 | 14 | 4 | -17.56c | [-18.49c, -13.84c] | -8.54c | [-11.24c, -6.08c] | 66.0 | 190.8 | 21.43% | 57.14% |
| 1 | 0.35 | 0.25 | 30 | 300 | 14 | 4 | -17.67c | [-18.49c, -14.46c] | -8.81c | [-11.37c, -7.35c] | 64.5 | 190.8 | 21.43% | 57.14% |
| 1 | 0.35 | 0.25 | 30 | 600 | 14 | 4 | -17.67c | [-18.49c, -14.46c] | -8.81c | [-11.37c, -7.35c] | 64.5 | 190.8 | 21.43% | 57.14% |

## Best Naked Regimes

| lat | entry | exit | k_s | max_s | trades | mkts | hedged_mean | hedged_CI | naked_mean | naked_CI | med_hold | p95_hold | conv | top_mkt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.50 | 0.25 | 5 | 300 | 12 | 3 | -18.04c | [-21.19c, -10.11c] | -8.15c | [-10.25c, -2.82c] | 63.5 | 202.6 | 25.00% | 58.33% |
| 10 | 0.50 | 0.25 | 5 | 600 | 12 | 3 | -18.04c | [-21.19c, -10.11c] | -8.15c | [-10.25c, -2.82c] | 63.5 | 202.6 | 25.00% | 58.33% |
| 10 | 0.50 | 0.10 | 5 | 300 | 12 | 3 | -18.16c | [-21.34c, -10.26c] | -8.30c | [-10.34c, -3.08c] | 63.5 | 202.6 | 16.67% | 58.33% |
| 10 | 0.50 | 0.10 | 5 | 600 | 12 | 3 | -18.16c | [-21.34c, -10.26c] | -8.30c | [-10.34c, -3.08c] | 63.5 | 202.6 | 16.67% | 58.33% |
| 1 | 0.50 | 0.10 | 5 | 300 | 11 | 4 | -17.53c | [-20.11c, -10.95c] | -8.25c | [-10.69c, -3.07c] | 79.0 | 203.0 | 18.18% | 54.55% |
| 1 | 0.50 | 0.10 | 5 | 600 | 11 | 4 | -17.53c | [-20.11c, -10.95c] | -8.25c | [-10.69c, -3.07c] | 79.0 | 203.0 | 18.18% | 54.55% |
| 1 | 0.50 | 0.25 | 5 | 300 | 11 | 4 | -17.66c | [-20.11c, -11.44c] | -8.60c | [-10.69c, -4.34c] | 79.0 | 203.0 | 18.18% | 54.55% |
| 1 | 0.50 | 0.25 | 5 | 600 | 11 | 4 | -17.66c | [-20.11c, -11.44c] | -8.60c | [-10.69c, -4.34c] | 79.0 | 203.0 | 18.18% | 54.55% |
| 2 | 0.35 | 0.25 | 15 | 300 | 29 | 5 | -17.48c | [-19.91c, -13.57c] | -9.65c | [-11.08c, -6.13c] | 24.0 | 186.6 | 31.03% | 41.38% |
| 2 | 0.35 | 0.25 | 15 | 600 | 29 | 5 | -17.48c | [-19.91c, -13.57c] | -9.65c | [-11.08c, -6.13c] | 24.0 | 186.6 | 31.03% | 41.38% |

## Latency Robustness

Same selected hedged parameter set across action latencies.

For the selected hedged parameter set, hedged mean changes from -17.56c at 1s to -18.93c at 10s; naked changes from -8.54c to -8.92c.

| latency_s | trades | hedged_mean | hedged_CI | naked_mean | naked_CI | med_hold_s | p95_hold_s | hedge_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 14 | -17.56c | [-18.49c, -13.84c] | -8.54c | [-11.24c, -6.08c] | 66.0 | 190.8 | 7.40c |
| 2 | 14 | -17.33c | [-18.19c, -13.64c] | -8.18c | [-11.24c, -6.01c] | 66.0 | 190.8 | 7.41c |
| 5 | 14 | -19.74c | [-22.07c, -13.08c] | -10.27c | [-11.54c, -5.63c] | 66.0 | 190.8 | 7.46c |
| 10 | 13 | -18.93c | [-22.49c, -14.70c] | -8.92c | [-11.48c, -7.09c] | 75.0 | 193.2 | 7.65c |

Same selected naked parameter set across action latencies.

| latency_s | trades | hedged_mean | hedged_CI | naked_mean | naked_CI | med_hold_s | p95_hold_s | hedge_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 11 | -17.66c | [-20.11c, -11.44c] | -8.60c | [-10.69c, -4.34c] | 79.0 | 203.0 | 7.32c |
| 2 | 12 | -18.54c | [-20.92c, -11.37c] | -9.13c | [-11.09c, -4.05c] | 63.5 | 202.6 | 7.72c |
| 5 | 11 | -20.91c | [-24.48c, -9.91c] | -10.54c | [-12.86c, -2.57c] | 79.0 | 203.0 | 8.13c |
| 10 | 12 | -18.04c | [-21.19c, -10.11c] | -8.15c | [-10.25c, -2.82c] | 63.5 | 202.6 | 7.75c |

## Cost Components

| selection | trades | pm_pnl | hedge_pnl | hedge_cost | funding | pm_fees | hedged_net | naked_net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_hedged | 14 | -8.18c | -1.73c | 7.41c | 0.00c | 2.09c | -17.33c | -8.18c |
| best_naked | 12 | -8.15c | -2.14c | 7.75c | 0.00c | 2.09c | -18.04c | -8.15c |

## Exit Mix

Selected hedged regime exit reasons:

| exit_reason | share |
| --- | --- |
| converged | 21.43% |
| regime_exit | 78.57% |

## Source And Regime Filter

Rows available: 48,946 in the forced moderate/mid regime, 32,087 after strict Chainlink-vs-Binance source filter, and 29,692 after excluding large static basis at the row level. Static-large share inside target+strict is 7.46%.

The strict source filter is the same hard filter as v3-H: no Chainlink-vs-Binance direction disagreement and Binance settlement margin >= 10.0bp. Source-risk windows are not counted as alpha.

## Method

- Signal: `dynamic_logit_gap = (pm_logit - fair_logit) - causal_static_logit_gap`.
- Entry: buy `UP` when the gap is below `-entry_band`; buy `DOWN` when above `entry_band`; require the same sign to persist for `k` sampled seconds before the signal.
- Exit: convergence to the exit band, max hold, or forced flatten when the row leaves the moderate/mid regime.
- No overlap: one open trade per market/config; the next entry search resumes after the exit fill.
- Hedge: optional Binance digital-delta hedge is rebalanced every second; naked and hedged PnL are both reported.
- CI: cluster bootstrap by market, 500 samples.

## Outputs

- H2 trade ledger: `data/analysis/csv_outputs/options_delta/k3v3h2_persistence_trades.csv`
- H2 summary grid: `data/analysis/csv_outputs/options_delta/k3v3h2_persistence_summary.csv`
- Extended v3-H ledger: `data/analysis/csv_outputs/options_delta/k3v3h_hedged_basis_trades_ext.csv`
- Feature cache reused: `data/analysis/cache/k3v3h_panel_features.parquet`
- Repro script: `scripts/dali_block_k3v3h2_persistence.py`
