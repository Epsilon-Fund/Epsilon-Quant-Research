---
title: Block K3 Lead-Lag and Digital-Option Basis Findings
created: 2026-06-05
status: candidate
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
  - strat_options_delta
tags:
  - options-delta
  - block-k
  - lead-lag
  - basis
  - research
---

# Block K3 Lead-Lag + Digital-Option Basis Findings

> **Strat:** [[strat_options_delta]] (Options-Delta). Sibling: [[strat_market_making]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

Generated: 2026-05-30T21:08:03Z

## Summary

K3 tests crypto 4h lead-lag and a digital-option basis using Binance fair value versus Polymarket mids. The in-sample post-fee basis survives as a diagnostic lead, but the note explicitly does not validate a hedgeable edge. The main takeaway is that basis rows need causal latency, source, execution, and OOS tests before becoming a deployable OD branch.

## Headline

In-sample post-fee basis survives, but it is not yet a validated hedgeable edge.

Pooled A0b + A0c crypto-roll in-sample has 24 in-window 4h contracts (BTC, ETH, SOL) and 29,683 10-second panel rows. The strongest cross-correlation has external Binance fair value leading Polymarket by -10s with corr 0.408; window bootstrap CI for the best lag is [-10, -10] seconds and corr-at-lag CI is [0.316, 0.531]. HY-style asynchronous overlap peaks at -10s with corr 0.408.

Median mid basis is -0.35c (mean -0.61c; 95th percentile absolute basis 9.67c). Best post-fee taker/complement edge is 26.29c on btc-updown-4h-1780113600 at 2026-05-30 05:05:10+00:00; 35.05% of rows are barely positive after fee and 22.49% clear a 1c buffer.

## Step 0 - Fee Gate

4h dynamic anti-arb fee present: **NO**.

All captured BTC/ETH/SOL 4h markets expose the standard Crypto CLOB fee schedule: taker-only, rate 0.07, exponent 1, rebateRate 0.2. That implies a peak taker fee of 1.75c at 50c. I did not observe a 4h fee rate consistent with the cited 15m anti-arb fee peak of about 3.15c. A live 15m probe during this run returned rate 0.070 / peak 1.75c, so the dynamic-fee claim is either not currently represented in the CLOB `fd` field or was not active on the sampled market.

| asset | market | runs | fee_rate | peak_fee | taker_only |
| --- | --- | --- | --- | --- | --- |
| BTC | btc-updown-4h-1779912000 | a0b | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1779926400 | a0b | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1779940800 | a0b | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780041600 | a0c_roll | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780056000 | a0c_roll | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780070400 | a0c_roll | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780084800 | a0c_roll | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780099200 | a0c_roll | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780113600 | a0c_roll | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780128000 | a0c_roll | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780142400 | a0c_roll | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780156800 | a0c_roll | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780171200 | a0c_roll | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780185600 | a0c_roll | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780200000 | a0c_roll | 0.070 | 1.75c | true |
| BTC | btc-updown-4h-1780214400 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780041600 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780056000 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780070400 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780084800 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780099200 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780113600 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780128000 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780142400 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780156800 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780171200 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780185600 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780200000 | a0c_roll | 0.070 | 1.75c | true |
| ETH | eth-updown-4h-1780214400 | a0c_roll | 0.070 | 1.75c | true |
| SOL | sol-updown-4h-1780041600 | a0c_roll | 0.070 | 1.75c | true |

## Data And Method

- Local CLOB: `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527` and `data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning`.
- External dependency: Binance spot 1s klines and Binance USDT-M perpetual 1m klines were fetched live by `scripts/dali_block_k3_leadlag_basis.py`.
- Only quotes inside the 4h contract window are used. Pre-open quotes are excluded because the strike is the window-open price.
- Strike is Binance spot at the window open. Digital fair value is `N(d2)` using the full-window realized spot volatility, annualized from 1m log returns. This is an ex-post fair-value normalization, not a tradable implied-vol forecast.
- Executable edge uses taker asks only: `buy_up = fair - up_ask - fee(up_ask)` and complement route `buy_down = 1 - fair - down_ask - fee(down_ask)`.
- Resolution-source caveat: Polymarket crypto up/down resolves from Chainlink streams, while Binance is the hedge/reference venue. The measured basis includes this source basis risk.

## Lead-Lag And Persistence

- Cross-correlation best lag: -10s, corr 0.408, n=29,635.
- HY-style 10s panel-overlap proxy best lag: -10s, corr 0.408.
- Basis autocorr: 60s 0.822 (n=29,539); 300s 0.696 (n=28,963); AR(1) half-life 54.4s.
- Positive post-fee edge run count: 1419; median run 20.0s; max run 4870.0s.

## Post-Fee Executable Basis

| asset | market | runs | max_edge | positive_rows | median_basis | p95_abs_basis |
| --- | --- | --- | --- | --- | --- | --- |
| BTC | btc-updown-4h-1780113600 | a0c_roll | 26.29c | 57.92% | 3.80c | 9.98c |
| SOL | sol-updown-4h-1780113600 | a0c_roll | 18.66c | 5.62% | -7.03c | 20.74c |
| ETH | eth-updown-4h-1780041600 | a0c_roll | 18.30c | 5.88% | -0.37c | 5.53c |
| BTC | btc-updown-4h-1780041600 | a0c_roll | 15.91c | 59.30% | -4.72c | 8.61c |
| BTC | btc-updown-4h-1779940800 | a0b | 15.60c | 70.62% | -0.63c | 10.67c |
| ETH | eth-updown-4h-1780056000 | a0c_roll | 13.18c | 54.46% | -4.82c | 11.12c |
| ETH | eth-updown-4h-1780113600 | a0c_roll | 11.75c | 12.15% | -0.26c | 6.83c |
| ETH | eth-updown-4h-1780099200 | a0c_roll | 10.80c | 14.79% | 0.64c | 7.91c |

The executable screen is materially weaker than the midpoint basis screen. Much of the apparent basis is eaten by the ask plus the convex taker fee, and the surviving rows are clustered enough that this should be treated as an in-sample executable screen rather than a durable hedgeable edge until it clears OOS with stricter latency/source controls.

## Output

- CSV panel: `data/analysis/csv_outputs/options_delta/k3_leadlag_basis.csv`
- Repro script: `scripts/dali_block_k3_leadlag_basis.py`
