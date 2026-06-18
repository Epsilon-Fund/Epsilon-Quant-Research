---
title: "OD Strategy A v3 PnL/Risk Deep-Dive"
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
# OD Strategy A v3 PnL/Risk Deep-Dive

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior OD note: [[od_strategy_a_v3_findings]]
> MM benchmark notes: [[mm_deployable_cells_findings]] · [[block_k5_stress_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Summary

- Scope: OD Strategy A v3 PnL/Risk Deep-Dive in the OD/options-delta area.
- Existing takeaway/status: OD Strategy A v3 **FAIL** under the new strict gate: survive the priced-in left tail **and** beat the same-market MM/structural baseline with lower-CI > 0.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Headline

OD Strategy A v3 **FAIL** under the new strict gate: survive the priced-in left tail **and** beat the same-market MM/structural baseline with lower-CI > 0.

Primary per-asset concurrent row: n=7 markets / 22 fills, mean 57.71c, median 14.17c, mean ROC -25.34%, daily run-rate $4.85 on peak concurrent capital $5.71.

Full-panel far-|z| strict-rich shorts had empirical ITM rate 39.13%; the simulated priced-in tail mean is 49.60c, CI [-13.72c, 114.85c].

Same-market strict-rich minus strict-source incremental mean is -40.47c, CI [-120.93c, 1.49c].

My read: the per-asset capital framing makes the raw OD row look economically interesting, but the strict pass/fail is still blocked by sample size and by the incremental-over-MM test. The OD richness filter is useful selection, but in this replay it does not yet prove an independent edge beyond the structural/source-filtered lifecycle that MM already highlighted.

## What Changed From v3

This note changes the capital model, not the trade idea. v3 kept the old global one-slot embargo as the official gate, which meant same-time BTC, ETH, and SOL markets competed for one risk slot. This deep-dive adopts **per-asset concurrent** as official: BTC, ETH, and SOL 4h markets can be held at the same time, because they are separate Polymarket markets and separate risk sleeves. The global one-slot row remains below as a conservative sensitivity.

This is Polymarket-only. There is no Binance hedge sweep here. The hedge was already demoted by v3 because it cut little variance in the far-|z| gold-mine bucket and consumed premium. Risk control here is capital sizing, market capture, and diversification across assets.

## Why `n=4` Does Not Mean Four Crypto Windows Worked

`n` is the number of selected **market episodes** after the row's filters, not the number of available crypto 4h windows and not the number of winners. A market episode is one resolved Polymarket market such as one BTC, ETH, or SOL 4h UP/DOWN contract, with all accepted fills inside it aggregated into one PnL observation.

The `n=4` row is the old **global sensitivity**: OOS + far-|z| + strict Chainlink/Binance source filter + rich-short edge >= 1c + one global active episode at a time. That global one-slot rule discards same-time ETH/SOL/BTC opportunities after one of them is selected. Under the new official per-asset concurrent capital model, the same OD filter has `n=7`, because BTC, ETH, and SOL can be active at the same time. The broader bare per-asset far-|z| row has `n=17`, because it does not require strict source plus rich-short richness.

So `n=4` means: four non-overlapping global-time episodes survived a narrow filter. It does **not** mean only four crypto 4h windows existed, nor that the other windows "failed"; most were outside OOS, had no eligible K-PEG fill, were not far-|z| at entry, failed the source filter, lacked a rich-short edge, or were removed by the conservative global embargo.

## Capital Definition

Every fill is converted into a synthetic long claim. A long UP is a long UP claim costing `price`; a short UP is a long DOWN claim costing `1 - price`; a short DOWN is a long UP claim costing `1 - price`. That is the cleanest way to understand shorting binary tokens: selling a token is economically the same as buying its complement.

`capital_at_risk` in the return tables is the conservative **peak complement capital tied up** inside the episode. The parquet also stores `peak_two_sided_risk_capital`, which nets paired UP/DOWN complete sets, and `peak_abs_dollar_delta`, which is the directional OD exposure. ROC uses the conservative complement-capital denominator.

## Task 1 — Capital And Return

| row | markets | fills | mean net | median net | CI | mean ROC | median ROC | ROC CI | $/day | peak concurrent capital |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_rich_short_per_asset | 7 | 22 | 57.71c | 14.17c | [0.48c, 151.00c] | -25.34% | 16.48% | [-75.97%, 20.05%] | $4.85 | $5.71 |
| strict_rich_short_global_sensitivity | 4 | 16 | 105.20c | 33.89c | [16.19c, 258.69c] | 29.66% | 20.98% | [17.91%, 50.71%] | $5.05 | $5.63 |
| official_strict_source_per_asset | 11 | 149 | 61.93c | -0.20c | [-0.65c, 175.78c] | -40.78% | -98.60% | [-77.40%, -2.65%] | $6.81 | $8.84 |
| bare_lifecycle_per_asset | 17 | 211 | 85.56c | -0.20c | [-1.92c, 204.62c] | -34.39% | -12.12% | [-62.48%, -3.69%] | $14.55 | $31.35 |

Read: `strict_rich_short_per_asset` is the official OD row. The global sensitivity shows the old one-slot assumption. `official_strict_source_per_asset` is the structural/source-filtered lifecycle without the OD richness cut.

The positive cents/episode and negative mean ROC can coexist because the ROC denominator is tiny for near-99c shorts. A loss of 6c on 6c complement capital is roughly -99% ROC, while a large winning episode contributes more cents but a less extreme percentage. That is why the table reports both mean and median.

## Task 2 — Capture / Capacity Proxy

Capacity is a proxy, not a full queue replay. I as-of joined each rich-short fill to `block_a1_features.parquet`, used same-side touch/top-5 depth plus 300s observed trade flow, then applied the K5 incumbent reality that top-3 wallets capture roughly 95% of each market. The `realistic` columns therefore assume only 5% of the observed opportunity is actually accessible to us without winning the incumbent queue.

| target contracts/fill | gross units | realistic units | realistic capital | realistic expected EV | realistic realized PnL | realistic expected EV/day |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 22.0 | 1.10 | $0.50 | $0.04 | $0.20 | $0.05 |
| 2 | 44.0 | 2.20 | $1.00 | $0.08 | $0.40 | $0.10 |
| 5 | 110.0 | 5.50 | $2.50 | $0.20 | $1.01 | $0.24 |
| 10 | 212.7 | 10.64 | $4.67 | $0.40 | $1.98 | $0.47 |
| 25 | 489.7 | 24.49 | $9.90 | $0.81 | $4.54 | $0.97 |
| 50 | 878.1 | 43.91 | $16.28 | $1.16 | $8.19 | $1.39 |
| 100 | 1486.5 | 74.33 | $24.35 | $1.29 | $14.67 | $1.54 |

Read: the gross book can look deep, but the incumbent haircut collapses deployable dollars. This is the OD version of the MM capacity lesson: crypto 4h may show high bps, yet practical non-incumbent dollars are small unless we solve queue/capture.

## Task 3 — Sizing Policies

All sizing policies use the same OOS far-|z| strict-rich/source-filtered fill set.

| policy | markets | mean net | median net | CI | mean ROC | PnL std | CVaR 5% | worst | weighted fills |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rv_edge_scaled | 7 | 31.72c | 10.55c | [1.70c, 73.21c] | -29.82% | 57.87c | -8.07c | -8.07c | 2.25 |
| dollar_delta_cap_50 | 7 | 21.68c | 12.31c | [0.71c, 51.45c] | 71.28% | 40.54c | -5.92c | -5.92c | 1.40 |
| flat_1_contract | 7 | 57.71c | 14.17c | [0.48c, 151.00c] | -25.34% | 125.58c | -8.76c | -8.76c | 3.14 |
| fractional_kelly_25pct | 7 | 2.73c | 1.10c | [-0.49c, 7.12c] | -30.15% | 6.03c | -1.78c | -1.78c | 1.26 |

Sizing definitions: `flat_1_contract` buys one synthetic complement per accepted fill. `rv_edge_scaled` sizes roughly proportional to edge versus RV physical-probability fair, capped at 3x. `dollar_delta_cap_50` clips fills once running episode dollar-delta reaches $50. `fractional_kelly_25pct` uses a quarter-Kelly binary-contract proxy from RV physical-probability fair, also capped at 3x.

## Task 4 — Left Tail

| row | markets | mean | median | worst | CVaR 5% | max drawdown | empirical ITM | stress CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed_primary | 7 | 57.71c | 14.17c | -8.76c | -8.76c | 8.76c | n/a | - |
| stress_empirical_far_short_itm | 7 | 49.60c | 43.42c | n/a | n/a | 160.32c | 39.13% | [-13.72c, 114.85c] |
| observed_concurrent_slot_portfolio | 7 | 100.99c | 29.51c | 14.17c | 14.17c | 0.00c | n/a | - |

The bad event is simple: we short an apparently overpriced longshot token and that token resolves ITM. The observed OOS set is too small to trust by itself, so the stress row prices disasters with the full-panel empirical ITM rate for far-|z| strict-rich shorts. Diversification helps only if the bad longshots are not all the same macro event; the slot portfolio row is the first check of same-time BTC/ETH/SOL aggregation.

## Task 5 — Incremental Over MM

MM benchmark used here: K5-STRESS crypto_4h structured-non-top3 median `2.4 bps`; aggregate structured-non-top3 `189 bps` with CI lower `21.8 bps`; deployable crypto 4h cells in `mm_deployable_cells_findings` round to about `$0/day` after incumbent capacity.

| variant | markets | mean net | median net | CI | mean ROC | lower ROC bps | lower bps minus MM median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bare_lifecycle_per_asset | 17 | 85.56c | -0.20c | [-1.92c, 204.62c] | -34.39% | -6248.0 | -6250.4 |
| official_strict_source_per_asset | 11 | 61.93c | -0.20c | [-0.65c, 175.78c] | -40.78% | -7739.9 | -7742.3 |
| rich_short_no_source_per_asset | 12 | 83.25c | 6.00c | [-28.79c, 240.69c] | -35.63% | -6931.5 | -6933.9 |
| strict_rich_short_per_asset | 7 | 57.71c | 14.17c | [0.48c, 151.00c] | -25.34% | -7597.5 | -7599.9 |
| strict_rich_short_global_sensitivity | 4 | 105.20c | 33.89c | [16.19c, 258.69c] | 29.66% | 1790.7 | 1788.3 |
| strict_rich_short_minus_strict_source_same_markets | 7 | -40.47c | 0.59c | [-120.93c, 1.49c] | - | - | - |

The key line is `strict_rich_short_minus_strict_source_same_markets`. If it is not lower-CI positive, OD has not proven that RV-richness adds EV beyond simply selecting the same source-filtered/structural markets. In that case this should fold back into MM as a quote-selection feature rather than stand alone as a separate OD edge.

## Gate Verdict

Pre-registered pass condition: per-asset concurrent, OOS, net-of-cost, far-|z| family, priced-in left tail survives with lower-CI > 0, and incremental-over-MM lower-CI > 0. This run does **not pass** that strict gate.

CSV: `data/analysis/csv_outputs/options_delta/od_strategy_a_v3_pnl_risk.csv`

Episode parquet: `data/analysis/od_strategy_a_v3_pnl_risk_episodes.parquet`

Distribution parquet: `data/analysis/od_strategy_a_v3_pnl_risk_distributions.parquet`
