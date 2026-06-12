---
title: Block K6 Strategy A Static Hedge
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
  - static-hedge
  - strategy-a
  - research
---

# Block K6 Strategy A Static Hedge

> **Strat:** [[strat_options_delta]] (Options-Delta). Sibling: [[strat_market_making]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Summary

This note runs the missing static-hedge Strategy A gate on K-PEG/K6 crypto fills. The strict OOS far/late cell has positive mean net, but its confidence interval crosses below zero. Under the preregistered gate, that fails to unblock the Kronos/HAR/EWMA forward-vol bake-off.

## Headline

Gate fails: do not unblock Kronos/HAR/EWMA forward-vol bake-off.

Strict OOS far/late: n=11, mean net 1.07c, CI [-1.04c, 3.99c].

This is the missing static-hedge gate: passive K-PEG maker entries, no Polymarket exit, one Binance hedge set at entry delta and closed at resolution. The primary table uses strict source filtering, K5-style two-sided book eligibility, and excludes the late near-50c spike zone.

## Construction

- Entry source: `data/analysis/kpeg_robustness_fills.parquet`, restricted to crypto-4h fills that overlap the K6/K3 panel.
- Pricing surface: `data/analysis/k6_vol_gap_panel.parquet`.
- Entry is passive maker fill: maker fee is zero; maker rebate is `20% * 0.07 * p * (1-p)`.
- Static hedge: `hedge_units = -entry_token_position * entry_digital_delta`; held unchanged until Binance window close.
- Binance cost: entry plus settlement notional at `6.0bp` each way.
- Spike filter: exclude rows with <=15m to expiry and token price in `[0.40, 0.60]`, plus K6 toxic near-strike/near-expiry rows.
- Non-overlap: each bucket cell is a restricted strategy and takes at most the first eligible fill per market, held to resolution.

Input counts:

- overlapping K-PEG/K6 crypto fills: `370`
- K5-style eligible fills before non-overlap: `362`
- strict-source eligible fills before non-overlap: `241`

## OOS Bucket Table

Strict source, OOS holdout only.

| bucket | n | net | net CI | unhedged | PM taker fee | maker rebate | hedge cost | win | median hold min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| far_absz_ge1|late_lt30m | 11 | 1.07c | [-1.04c, 3.99c] | 0.52c | 0.00c | 0.03c | 2.18c | 27.27% | 10.3 |
| far_absz_ge1|early_gt2h | 1 | 58.02c | [58.02c, 58.02c] | 84.19c | 0.00c | 0.19c | 2.52c | 100.00% | 147.9 |
| far_absz_ge1|mid_30m_2h | 8 | 11.15c | [0.96c, 21.41c] | 17.61c | 0.00c | 0.12c | 4.22c | 62.50% | 51.6 |
| near_absz_lt0.25|mid_30m_2h | 5 | -1.43c | [-15.21c, 10.87c] | -9.26c | 0.00c | 0.34c | 9.28c | 60.00% | 78.9 |
| mid_absz_0.25_1|mid_30m_2h | 6 | -5.92c | [-20.99c, 12.46c] | -10.06c | 0.00c | 0.30c | 9.10c | 33.33% | 70.3 |
| mid_absz_0.25_1|late_lt30m | 3 | -6.40c | [-23.48c, 7.57c] | 25.58c | 0.00c | 0.26c | 17.78c | 33.33% | 22.5 |
| mid_absz_0.25_1|early_gt2h | 3 | 11.04c | [-37.15c, 35.90c] | 41.71c | 0.00c | 0.26c | 4.46c | 66.67% | 202.3 |
| near_absz_lt0.25|early_gt2h | 3 | -25.86c | [-45.04c, -7.94c] | -35.91c | 0.00c | 0.30c | 5.12c | 0.00% | 233.2 |

## IS Bucket Table

Strict source, discovery only. This is a lead, not the result.

| bucket | n | net | net CI | unhedged | PM taker fee | maker rebate | hedge cost | win | median hold min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| far_absz_ge1|late_lt30m | 3 | -0.10c | [-0.10c, -0.08c] | -0.10c | 0.00c | 0.00c | 0.03c | 0.00% | 22.5 |
| near_absz_lt0.25|early_gt2h | 1 | 69.52c | [69.52c, 69.52c] | -42.66c | 0.00c | 0.34c | 9.60c | 100.00% | 161.4 |
| mid_absz_0.25_1|early_gt2h | 1 | 63.55c | [63.55c, 63.55c] | -33.69c | 0.00c | 0.31c | 8.95c | 100.00% | 151.9 |
| far_absz_ge1|early_gt2h | 1 | -4.69c | [-4.69c, -4.69c] | -11.85c | 0.00c | 0.15c | 5.32c | 0.00% | 139.3 |

## Far/Late Split

Decision bucket: `far_absz_ge1|late_lt30m`.

| sample | n | net | net CI | unhedged | hedge cost | win |
| --- | --- | --- | --- | --- | --- | --- |
| is_discovery | 3 | -0.10c | [-0.10c, -0.08c] | -0.10c | 0.03c | 0.00% |
| oos_holdout | 11 | 1.07c | [-1.04c, 3.99c] | 0.52c | 2.18c | 27.27% |
| pooled | 14 | 0.82c | [-0.80c, 3.16c] | 0.39c | 1.72c | 21.43% |

## Decision

Gate fails: do not unblock Kronos/HAR/EWMA forward-vol bake-off.

The static hedge removes the continuous-turnover problem from K6, but the OOS far/late gate is judged only on the strict-source, non-overlap bucket lower CI. This note does not run Kronos or any forward-vol model.

Outputs:

- `data/analysis/csv_outputs/options_delta/k6_strategy_a_static_hedge.csv`
- `data/analysis/k6_strategy_a_static_hedge_trades.parquet`
