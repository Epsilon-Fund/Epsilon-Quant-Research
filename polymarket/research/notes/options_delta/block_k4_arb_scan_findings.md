---
title: Block K4 Intra-Polymarket Arb Scan
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
  - arbitrage
  - rebalancing
  - research
---

# Block K4 — Intra-Polymarket Arb Scan

> **Strat:** [[strat_options_delta]] (Options-Delta). Sibling: [[strat_market_making]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Summary

K4 scans the owned universe for intra-Polymarket rebalancing and combinatorial arbitrage intervals. One violation interval is detected, but none survives the p99-latency and minimum-size executability gates. The result closes this owned-universe arb scan while leaving the broader literature thread as a future watch item.

## Headline

**Is this a real parallel thread? No. On this owned universe, there were no p99-latency-capturable, minimum-size executable arb intervals after complete-book/fresh-book filtering.**

Frequency × size: `1` violation intervals were detected in total, `0` survived the primary p99 latency cut (`900ms`) with top-of-book size at least `5` shares, and `0` intervals were both p99-capturable and executable long/mint bundles. The largest observed gap was `2.000` cents per bundle unit; largest displayed size was `16.90` shares.

## Method

- Source: `data/analysis/block_a1_features.parquet`; this refreshed A1 feature parquet contains `a0`, `a0b`, `a0c`, and `a0c_roll`.
- Book controls: snapshot events only (`book`, `price_change`, `best_bid_ask`), exchange timestamps, complete book state, `book_staleness_seconds <= 5`, valid positive top-of-book sizes, and no crossed same-outcome books.
- Rebalancing rule: binary YES/NO ask sum `< $1` or bid sum `> $1`, collapsed into contiguous event-time intervals.
- Combinatorial rule: conservative owned-universe logical sets only. Mutually exclusive sets scan both buy-all-NO asks and sell-all-YES bids; the Strait of Hormuz date pair scans the June-implies-July implication.
- Capturable rule: primary headline uses `900ms` action latency from `configs/backtest_default.yaml` p99 and top-of-book bundle size `>= 5` shares. A `300ms` median-latency flag is also in the CSV.
- Literature anchor: Saguillo et al., _Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets_ ([arXiv:2508.03474](https://arxiv.org/abs/2508.03474)), distinguishing market rebalancing and combinatorial arb.

## Opportunity Summary

| type | direction | execution | intervals | p99 capturable | max gap c | max size | max gross $ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rebalancing | mint_sell_yes_no_to_bids | mint_then_sell_executable | 1 | 0 | 2.000 | 16.90 | 0.3380 |

## By Family

| family | intervals | p99 capturable | max gap c | max size | interval sec |
| --- | --- | --- | --- | --- | --- |
| sports_game_lines | 1 | 0 | 2.000 | 16.90 | 0.1 |

## Segment Sensitivity

| spread | depth | time to resolution | intervals | p99 capturable | max gap c |
| --- | --- | --- | --- | --- | --- |
| wide_>5c | small_5_25 | unknown | 1 | 0 | 2.000 |

## Clock-Time Sensitivity

| UTC hour | intervals | p99 capturable | max gap c |
| --- | --- | --- | --- |
| 1 | 1 | 0 | 2.000 |

## Market-Balanced View

| run | group | slugs | intervals | p99 capturable | max gap c | max size |
| --- | --- | --- | --- | --- | --- | --- |
| a0c | binary_yes_no_parity | nhl-mon-car-2026-05-29-spread-home-1pt5 | 1 | 0 | 2.000 | 16.90 |

## Largest Intervals

| type | direction | run | slugs | duration ms | max gap c | max size | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rebalancing | mint_sell_yes_no_to_bids | a0c | nhl-mon-car-2026-05-29-spread-home-1pt5 | 84 | 2.000 | 16.90 | False |

## Row-Count Heatmap

| run | family | markets | state rows | first state | last state |
| --- | --- | --- | --- | --- | --- |
| a0 | ai_product | 4 | 266,161 | 2026-05-27 10:05:10.835000+00:00 | 2026-05-28 10:07:23.201000+00:00 |
| a0 | daily_equity_index | 1 | 368 | 2026-05-27 09:46:00.159000+00:00 | 2026-05-28 07:01:58.153000+00:00 |
| a0 | daily_single_stock | 1 | 1,530 | 2026-05-27 10:07:24.071000+00:00 | 2026-05-28 01:06:34.004000+00:00 |
| a0 | geopolitics_policy | 4 | 297,671 | 2026-05-27 10:07:24.984000+00:00 | 2026-05-28 10:07:28+00:00 |
| a0 | sports_game_lines | 2 | 75,506 | 2026-05-27 10:07:09.610000+00:00 | 2026-05-28 10:07:27.001000+00:00 |
| a0b | crypto_4h_up_down | 3 | 196,585 | 2026-05-27 21:13:03.360000+00:00 | 2026-05-28 07:57:37.462000+00:00 |
| a0b | daily_crypto_up_down | 2 | 356,856 | 2026-05-27 21:18:55.106000+00:00 | 2026-05-28 09:19:12.271000+00:00 |
| a0b | geopolitics_policy | 2 | 127,161 | 2026-05-27 21:19:12.330000+00:00 | 2026-05-28 09:19:13.589000+00:00 |
| a0b | sports_game_lines | 1 | 46,946 | 2026-05-27 21:19:13.057000+00:00 | 2026-05-28 09:19:12.005000+00:00 |
| a0b | sports_neg_risk_outright | 1 | 15,212 | 2026-05-27 21:19:11.115000+00:00 | 2026-05-28 09:18:59.668000+00:00 |
| a0c | crypto_4h_up_down | 6 | 371,369 | 2026-05-29 08:59:39.371000+00:00 | 2026-05-29 15:58:09.117000+00:00 |
| a0c | daily_crypto_up_down | 3 | 284,630 | 2026-05-29 09:01:22.815000+00:00 | 2026-05-29 15:59:30.836000+00:00 |
| a0c | geopolitics_policy | 2 | 285,116 | 2026-05-29 09:01:24.992000+00:00 | 2026-05-30 09:01:30.205000+00:00 |
| a0c | politics_neg_risk_outright | 1 | 576,270 | 2026-05-29 09:01:30.319000+00:00 | 2026-05-30 09:01:29.937000+00:00 |
| a0c | sports_game_lines | 3 | 327,333 | 2026-05-29 09:00:55.756000+00:00 | 2026-05-30 09:01:29.238000+00:00 |
| a0c | sports_neg_risk_outright | 2 | 47,149 | 2026-05-29 09:01:18.911000+00:00 | 2026-05-30 09:01:23.046000+00:00 |
| a0c_roll | crypto_4h_up_down | 38 | 1,009,510 | 2026-05-29 08:02:49.382000+00:00 | 2026-05-30 09:09:21.897000+00:00 |

## Run Coverage

| run | raw rows | raw markets | state rows | state markets | first state | last state |
| --- | --- | --- | --- | --- | --- | --- |
| a0 | 2,563,417 | 12 | 641,236 | 12 | 2026-05-27 09:46:00.159000+00:00 | 2026-05-28 10:07:28+00:00 |
| a0b | 1,761,469 | 9 | 742,760 | 9 | 2026-05-27 21:13:03.360000+00:00 | 2026-05-28 09:19:13.589000+00:00 |
| a0c | 4,536,389 | 17 | 1,891,867 | 17 | 2026-05-29 08:59:39.371000+00:00 | 2026-05-30 09:01:30.205000+00:00 |
| a0c_roll | 2,445,782 | 38 | 1,009,510 | 38 | 2026-05-29 08:02:49.382000+00:00 | 2026-05-30 09:09:21.897000+00:00 |

## Logical Groups Considered

| run | type | group | relation | slugs |
| --- | --- | --- | --- | --- |
| a0 | exclusive | ai_best_model_june_2026 | mutually_exclusive_non_exhaustive | will-anthropic-have-the-best-ai-model-at-.; will-google-have-the-best-ai-model-at-the. |
| a0 | exclusive | fifa_world_cup_2026_winner | mutually_exclusive_non_exhaustive | will-france-win-the-2026-fifa-world-cup-9.; will-spain-win-the-2026-fifa-world-cup-963 |
| a0b | implication | strait_hormuz_june_implies_july | market_0_yes_implies_market_1_yes | strait-of-hormuz-traffic-returns-to-norma.; strait-of-hormuz-traffic-returns-to-norma. |

## Read

This scan is deterministic and has no IS/OOS split. The key guardrail is that interval duration is measured from exchange-time book states, so single sparse ticks do not become fake capturable trades. Rebalancing rows are directly executable as complete-set buy/redeem or mint/sell bundles; combinatorial rows tagged `requires_short_or_inventory` are diagnostic unless we already hold the short leg inventory or can source it without leg risk.
