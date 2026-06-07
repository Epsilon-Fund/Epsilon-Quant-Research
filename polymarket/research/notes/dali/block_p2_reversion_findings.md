---
tags: [dali, p2, micro-price, reversion, results]
title: P2 Reversion-To-Microprice Findings
created: 2026-05-31
status: archived
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
---

# P2 Reversion-To-Microprice Findings

> Hub: [[COWORK]]

> Table terms: [[polymarket_table_dictionary]]


## Summary

P2 tests an explicit passive fade-to-microprice framing for the local Dali signal. It finds eight tail-aware passive survivor rows with CI lower > 0, but none survive the stronger n >= 30 robustness bar and the combined both-tails frontier is empty. The note reopens the signal only as a fragile tail-specific anomaly, not as a clean reusable edge.

## Headline

Yes, but only narrowly: 8 tail-aware passive market-regime-timeout rows clear CI lower > 0 under non-overlap, with 0 still clearing at n >= 30. The combined `both_tails` frontier has 0 CI-positive rows, so this reopens the local signal only as a fragile tail-specific anomaly rather than a clean reusable edge.

Best passive row by CI lower bound is `ofi_5s` / `market_depth_decile` / `a0b:1971905` / `depth_d10_deep`, W=1s, target `micro_price`, timeout=5s, 11 executions, fill rate 1.4%, mean 10.8 bps, CI [1.1 bps, 20.5 bps].

## Open Question Answer

The narrow test was: does **any passive fade-to-microprice cell** survive non-overlap, fill-rate drag, and net execution costs? Counting `market_spread_decile` and `market_depth_decile` rows with at least 5 non-overlap executions, the answer is:

- Passive tail-aware survivor rows with CI lower > 0: 8
- Passive tail-aware robust survivor rows with CI lower > 0 and n >= 30: 0
- Passive combined-`both_tails` survivor rows with CI lower > 0: 0
- Passive combined-`both_tails` robust survivor rows with CI lower > 0 and n >= 30: 0
- Any-segment passive survivor rows, for context only: 8
- Taker fade survivor rows, for execution-control context: 0

## Passive Frontier

| signal | segment | market | slug | regime | tail | fill | target | timeout | signals | exec | fill rate | mean | CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ofi_5s | market_depth_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | depth_d10_deep | bottom_decile | W=1 | micro price | 5s | 771 | 11 | 1.4% | 10.8 bps | [1.1 bps, 20.5 bps] |
| ofi_5s | market_depth_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | depth_d10_deep | bottom_decile | W=1 | micro price | 10s | 771 | 11 | 1.4% | 10.8 bps | [1.1 bps, 20.5 bps] |
| ofi_5s | market_depth_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | depth_d10_deep | bottom_decile | W=1 | half to micro price | 5s | 771 | 11 | 1.4% | 10.8 bps | [1.1 bps, 20.5 bps] |
| ofi_5s | market_depth_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | depth_d10_deep | bottom_decile | W=1 | half to micro price | 10s | 771 | 11 | 1.4% | 10.8 bps | [1.1 bps, 20.5 bps] |
| ofi_5s | market_depth_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | depth_d10_deep | bottom_decile | W=1 | micro price | 30s | 771 | 10 | 1.3% | 11.9 bps | [0.8 bps, 23.0 bps] |
| ofi_5s | market_depth_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | depth_d10_deep | bottom_decile | W=1 | micro price | 60s | 771 | 10 | 1.3% | 11.9 bps | [0.8 bps, 23.0 bps] |
| ofi_5s | market_depth_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | depth_d10_deep | bottom_decile | W=1 | half to micro price | 30s | 771 | 10 | 1.3% | 11.9 bps | [0.8 bps, 23.0 bps] |
| ofi_5s | market_depth_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | depth_d10_deep | bottom_decile | W=1 | half to micro price | 60s | 771 | 10 | 1.3% | 11.9 bps | [0.8 bps, 23.0 bps] |
| ofi_5s | market_spread_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | spread_d05 | bottom_decile | W=10 | micro price | 5s | 277 | 14 | 5.1% | 0.0 bps | [0.0 bps, 0.0 bps] |
| ofi_5s | market_spread_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | spread_d05 | bottom_decile | W=10 | micro price | 10s | 277 | 14 | 5.1% | 0.0 bps | [0.0 bps, 0.0 bps] |
| ofi_5s | market_spread_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | spread_d05 | bottom_decile | W=10 | micro price | 30s | 277 | 14 | 5.1% | 0.0 bps | [0.0 bps, 0.0 bps] |
| ofi_5s | market_spread_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | spread_d05 | bottom_decile | W=10 | micro price | 60s | 277 | 14 | 5.1% | 0.0 bps | [0.0 bps, 0.0 bps] |
| ofi_5s | market_spread_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | spread_d05 | bottom_decile | W=10 | half to micro price | 5s | 277 | 14 | 5.1% | 0.0 bps | [0.0 bps, 0.0 bps] |
| ofi_5s | market_spread_decile | a0b:1971905 | strait-of-hormuz-traffic-returns-. | spread_d05 | bottom_decile | W=10 | half to micro price | 10s | 277 | 14 | 5.1% | 0.0 bps | [0.0 bps, 0.0 bps] |

## Tight-Spread Passive Rows

| signal | segment | market | slug | regime | tail | fill | target | timeout | signals | exec | fill rate | mean | CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tob_imbalance_level | family_spread_decile | ALL | ALL | spread_d01_tight | both_tails | W=1 | micro price | 5s | 12,224 | 18 | 0.1% | -1.6 bps | [-3.8 bps, 0.7 bps] |
| tob_imbalance_level | market_spread_decile | a0:1469737 | will-mojtaba-khamenei-be-head-of-. | spread_d01_tight | both_tails | W=1 | micro price | 5s | 12,224 | 18 | 0.1% | -1.6 bps | [-3.8 bps, 0.7 bps] |
| tob_imbalance_level | family_spread_decile | ALL | ALL | spread_d01_tight | both_tails | W=1 | micro price | 10s | 12,224 | 18 | 0.1% | -1.6 bps | [-3.8 bps, 0.7 bps] |
| tob_imbalance_level | market_spread_decile | a0:1469737 | will-mojtaba-khamenei-be-head-of-. | spread_d01_tight | both_tails | W=1 | micro price | 10s | 12,224 | 18 | 0.1% | -1.6 bps | [-3.8 bps, 0.7 bps] |
| tob_imbalance_level | family_spread_decile | ALL | ALL | spread_d01_tight | both_tails | W=1 | half to micro price | 5s | 12,224 | 18 | 0.1% | -1.6 bps | [-3.8 bps, 0.7 bps] |
| tob_imbalance_level | market_spread_decile | a0:1469737 | will-mojtaba-khamenei-be-head-of-. | spread_d01_tight | both_tails | W=1 | half to micro price | 5s | 12,224 | 18 | 0.1% | -1.6 bps | [-3.8 bps, 0.7 bps] |
| tob_imbalance_level | family_spread_decile | ALL | ALL | spread_d01_tight | both_tails | W=1 | half to micro price | 10s | 12,224 | 18 | 0.1% | -1.6 bps | [-3.8 bps, 0.7 bps] |
| tob_imbalance_level | market_spread_decile | a0:1469737 | will-mojtaba-khamenei-be-head-of-. | spread_d01_tight | both_tails | W=1 | half to micro price | 10s | 12,224 | 18 | 0.1% | -1.6 bps | [-3.8 bps, 0.7 bps] |
| tob_imbalance_level | family_spread_decile | ALL | ALL | spread_d01_tight | both_tails | W=5 | micro price | 5s | 12,224 | 18 | 0.1% | -1.6 bps | [-3.8 bps, 0.7 bps] |
| tob_imbalance_level | market_spread_decile | a0:1469737 | will-mojtaba-khamenei-be-head-of-. | spread_d01_tight | both_tails | W=5 | micro price | 5s | 12,224 | 18 | 0.1% | -1.6 bps | [-3.8 bps, 0.7 bps] |

## Taker Fade Control

| signal | segment | market | slug | regime | tail | fill | target | timeout | signals | exec | fill rate | mean | CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ofi_5s | market_spread_decile | a0b:2364426 | btc-updown-4h-1779912000 | spread_d01_tight | both_tails | W=0 | micro price | 5s | 13 | 8 | 61.5% | -12.2 bps | [-12.2 bps, -12.2 bps] |
| ofi_5s | market_spread_decile | a0b:2364426 | btc-updown-4h-1779912000 | spread_d01_tight | bottom_decile | W=0 | micro price | 5s | 13 | 8 | 61.5% | -12.2 bps | [-12.2 bps, -12.2 bps] |
| ofi_5s | market_spread_decile | a0b:2364426 | btc-updown-4h-1779912000 | spread_d01_tight | both_tails | W=0 | micro price | 10s | 13 | 8 | 61.5% | -12.2 bps | [-12.2 bps, -12.2 bps] |
| ofi_5s | market_spread_decile | a0b:2364426 | btc-updown-4h-1779912000 | spread_d01_tight | bottom_decile | W=0 | micro price | 10s | 13 | 8 | 61.5% | -12.2 bps | [-12.2 bps, -12.2 bps] |
| ofi_5s | market_spread_decile | a0b:2364426 | btc-updown-4h-1779912000 | spread_d01_tight | both_tails | W=0 | micro price | 30s | 13 | 8 | 61.5% | -12.2 bps | [-12.2 bps, -12.2 bps] |
| ofi_5s | market_spread_decile | a0b:2364426 | btc-updown-4h-1779912000 | spread_d01_tight | bottom_decile | W=0 | micro price | 30s | 13 | 8 | 61.5% | -12.2 bps | [-12.2 bps, -12.2 bps] |
| ofi_5s | market_spread_decile | a0b:2364426 | btc-updown-4h-1779912000 | spread_d01_tight | both_tails | W=0 | micro price | 60s | 13 | 8 | 61.5% | -12.2 bps | [-12.2 bps, -12.2 bps] |
| ofi_5s | market_spread_decile | a0b:2364426 | btc-updown-4h-1779912000 | spread_d01_tight | bottom_decile | W=0 | micro price | 60s | 13 | 8 | 61.5% | -12.2 bps | [-12.2 bps, -12.2 bps] |
| ofi_5s | market_spread_decile | a0b:2364426 | btc-updown-4h-1779912000 | spread_d01_tight | both_tails | W=0 | half to micro price | 5s | 13 | 8 | 61.5% | -12.2 bps | [-12.2 bps, -12.2 bps] |
| ofi_5s | market_spread_decile | a0b:2364426 | btc-updown-4h-1779912000 | spread_d01_tight | bottom_decile | W=0 | half to micro price | 5s | 13 | 8 | 61.5% | -12.2 bps | [-12.2 bps, -12.2 bps] |

## Signal Event Counts

| signal | top/bottom-decile events |
| --- | --- |
| ofi_5s | 323,663 |
| tob_imbalance_level | 628,873 |

## Method

- Input: `data/analysis/block_a1_features.parquet`.
- Signals: `tob_imbalance_level = direction_factor * tob_imbalance` and a 5s OFI sidecar, `ofi_5s = direction_factor * rolling_sum(ofi_combined_event, 5s) / mean_depth`.
- Trigger: per-market signed top and bottom deciles. The CSV reports `top_decile`, `bottom_decile`, and `both_tails`.
- Trade direction: fade the signal. If continuation would buy a token, P2 sells it; if continuation would sell, P2 buys it.
- Target: current weighted mid/microprice, `weighted_mid = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)`, or half-way from entry price to that weighted mid.
- Mode T: immediate taker fade at the opposite executable touch, with taker fee on entry and exit.
- Mode P: post at the touch on the fade side. A long bid fills on a SELL print at or below the bid; a short ask fills on a BUY print at or above the ask, within W in (1, 5, 10). Entry gets the maker rebate; exit pays taker fee at bid/ask.
- Exit: first of target reached, one-spread adverse stop, or timeout in (5, 10, 30, 60). Exits are marked to bid for long and ask for short.
- Non-overlap: one open position per market per grid cell. Taker signals block from entry to exit. Passive signals block only after an actual fill; candidates are considered in fill-time order and are skipped if either signal time or fill time falls inside an open interval.
- CI: normal interval over contiguous 300s block means of non-overlap executed PnL.
- Segments: all, family, market, spread decile, depth decile, family x spread decile, market x spread decile, and market x depth decile.

## Outputs

- `data/analysis/csv_outputs/dali/p2_reversion_surface.csv`
- `data/analysis/csv_outputs/dali/p2_reversion_passive_fillfrontier.csv`
