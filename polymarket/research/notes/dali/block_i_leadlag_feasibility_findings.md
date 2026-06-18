---
title: "Block I Lead-Lag Feasibility Findings"
created: 2026-06-05
status: closed
owner: justin
project: polymarket
para: project
hubs:
  - COWORK
tags:
  - research
  - dali
---
# Block I Lead-Lag Feasibility Findings

> Hub: [[COWORK]]

## Summary

- Scope: Block I Lead-Lag Feasibility Findings in the Dali research lineage area.
- Existing takeaway/status: Block I lead-lag feasibility closes the cross-market lead-lag branch. It checks timestamp alignment, predictive lead-lag, and executable gating, with the recorded verdict: CONFIRM-CLOSE.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
Links: [[block_a1x_external_note_reconciliation]] (#21), [[block_k3_leadlag_findings]], [[block_k3v2_findings]], [[block_k5b_findings]].

Verdict: **CONFIRM-CLOSE**.

Deciding number: L=60s H=5s latency=1s threshold=20bp OOS mean -21.580c, cluster CI [-31.731c, -10.857c]; even the best OOS 1s diagnostic was L=2s H=1s latency=1s threshold=0bp at -9.478c, CI [-12.571c, -7.232c].

This is a feasibility gate for an external-underlying lag edge, not a reopening of the closed local Dali continuation signal. The cached artifacts contain Binance spot/return history and PM crypto CLOB quotes; they do **not** contain Binance/OKX order-book OFI. I therefore gate the available version: Binance return impulse leads PM direction markets.

## Timestamp Alignment

| segment | n | p50 | p95 | p99 | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| panel_1s_gaps | 296721 | 1000ms | 1000ms | 1000ms | 1000ms | 1000ms |
| all_panel_rows | 296745 | -31ms | 55ms | 1987902ms | -99ms | 2804728ms |
| changed_quote_rows | 47242 | -29ms | 12ms | 65ms | -99ms | 2804728ms |
| changed_quote_rows_a0b | 4417 | -22ms | 17ms | 62ms | -87ms | 605ms |
| changed_quote_rows_a0c_roll | 42825 | -29ms | 9ms | 65ms | -99ms | 2804728ms |

Alignment read: the one-second panel itself is exactly spaced. Fresh PM quote-change rows have small capture-latency dispersion, so 1-60s lead-lag measurement is clean enough. The gate is **not** clean for true sub-second measurement because the cached Binance side is one-second klines and the PM panel is one-second resampled; latency `0s` is therefore a diagnostic for same-second/sub-second race conditions, not deployable evidence.

## Predictive Lead-Lag

Top OOS correlations at the official 1s entry-latency proxy:

| cell | markets | rows | corr | cluster CI | sign hit |
| --- | --- | --- | --- | --- | --- |
| L=60s H=60s | 12 | 140631 | 0.0431 | [0.0237, 0.0625] | 53.65% |
| L=10s H=10s | 12 | 141831 | 0.0326 | [0.0164, 0.0550] | 58.18% |
| L=30s H=60s | 12 | 140991 | 0.0241 | [0.0090, 0.0441] | 53.61% |
| L=10s H=30s | 12 | 141591 | 0.0233 | [0.0097, 0.0388] | 55.40% |
| L=60s H=30s | 12 | 140991 | 0.0225 | [0.0083, 0.0388] | 53.39% |
| L=1s H=1s | 12 | 142047 | 0.0221 | [-0.0014, 0.0449] | 53.47% |

Top pooled correlations at the official 1s entry-latency proxy:

| cell | markets | rows | corr | cluster CI | sign hit |
| --- | --- | --- | --- | --- | --- |
| L=60s H=60s | 24 | 293841 | 0.0835 | [0.0372, 0.1179] | 54.17% |
| L=30s H=60s | 24 | 294561 | 0.0574 | [0.0274, 0.0819] | 53.87% |
| L=10s H=10s | 24 | 296241 | 0.0529 | [0.0358, 0.0786] | 57.74% |
| L=60s H=30s | 24 | 294561 | 0.0524 | [0.0268, 0.0741] | 53.85% |
| L=5s H=10s | 24 | 296361 | 0.0490 | [0.0325, 0.0743] | 57.71% |
| L=10s H=30s | 24 | 295761 | 0.0487 | [0.0362, 0.0666] | 55.58% |

The predictive signal exists at slow horizons but is small: PM logit moves in the same direction as the prior Binance return impulse only weakly once measured at a deployable 1s proxy. Rows with `H=60s` are boundary diagnostics around K3's 54s basis half-life; the best inside-half-life OOS correlation is the `L=10s H=10s` row above.

## Executable Gate

Configuration was selected on train markets at `latency_s=1` by highest market-cluster CI lower bound among rows with at least 3 active markets and 30 trades, then evaluated on OOS markets.

| row | cell | markets | trades | mean | cluster CI | win | worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train-selected train | L=60s H=5s latency=1s threshold=20bp | 10 | 651 | -8.272c | [-9.871c, -6.183c] | 0.46% | -90.659c |
| train-selected OOS | L=60s H=5s latency=1s threshold=20bp | 5 | 56 | -21.580c | [-31.731c, -10.857c] | 1.79% | -52.065c |
| train-selected all | L=60s H=5s latency=1s threshold=20bp | 15 | 707 | -9.326c | [-12.564c, -7.153c] | 0.57% | -90.659c |

Count of OOS `latency_s=1` rows with market-cluster lower CI > 0: `0`.

Best OOS diagnostics at the deployable 1s proxy:

| cell | markets | trades | mean | cluster CI | win |
| --- | --- | --- | --- | --- | --- |
| L=2s H=1s latency=1s threshold=0bp | 12 | 36826 | -9.478c | [-12.571c, -7.232c] | 0.05% |
| L=2s H=2s latency=1s threshold=0bp | 12 | 28423 | -9.553c | [-12.657c, -7.092c] | 0.06% |
| L=1s H=2s latency=1s threshold=0bp | 12 | 25404 | -9.544c | [-12.726c, -7.197c] | 0.08% |
| L=1s H=1s latency=1s threshold=0bp | 12 | 31628 | -9.468c | [-12.777c, -7.188c] | 0.04% |
| L=5s H=2s latency=1s threshold=0bp | 12 | 34616 | -9.733c | [-12.951c, -6.989c] | 0.05% |
| L=5s H=10s latency=1s threshold=0bp | 12 | 11474 | -9.747c | [-12.971c, -7.363c] | 0.21% |

Best OOS diagnostics at same-second latency `0s` (not deployable from this artifact):

| cell | markets | trades | mean | cluster CI | win |
| --- | --- | --- | --- | --- | --- |
| L=1s H=1s latency=0s threshold=0bp | 12 | 31629 | -9.502c | [-12.550c, -7.223c] | 0.04% |
| L=5s H=2s latency=0s threshold=0bp | 12 | 34618 | -9.736c | [-12.583c, -7.285c] | 0.07% |
| L=60s H=2s latency=0s threshold=0bp | 12 | 44641 | -9.858c | [-12.670c, -7.256c] | 0.06% |
| L=1s H=10s latency=0s threshold=0bp | 12 | 10278 | -9.764c | [-12.717c, -7.324c] | 0.30% |
| L=30s H=2s latency=0s threshold=0bp | 12 | 42678 | -9.813c | [-12.727c, -7.542c] | 0.06% |
| L=30s H=1s latency=0s threshold=0bp | 12 | 62486 | -9.819c | [-12.784c, -7.486c] | 0.02% |

The spread/fee headwind dominates. The best 1s-latency OOS rows are still negative after entering at the PM ask and exiting at the bid. Same-second rows do not rescue the gate either; even if they had, this artifact would not establish a 100-800ms edge.

## Assumption Ledger

Modeled assumptions:

- Binance spot return impulse proxies the external lead. True Binance/OKX perp OFI is unavailable in the saved artifacts and remains untested here; OKX public klines were not added because klines alone do not supply the missing OFI and would not change the PM executable spread gate.
- Official action latency is the next one-second panel row (`latency_s=1`), used as a conservative proxy for a 100-800ms path. `latency_s=0` is only a clock-artifact/same-second diagnostic.
- Entry buys the direction token implied by the prior Binance move at PM ask; exit sells at PM bid after a fixed hold. Taker fee is charged on entry and exit.
- Non-overlap is one open position per market/config; chronological OOS split is first 12 markets train, last 12 OOS.
- Cluster CI resamples active market clusters; same-window cross-asset dependence is a residual limitation, so this is a scope gate rather than a deployment claim.

Live-only unknowns:

- Real decision-to-order-to-fill latency and quote survival after the Binance move.
- Whether Binance/OKX perp order-flow imbalance adds information beyond spot returns without turning the edge into a sub-second race.
- PM order rejection, stale-book handling, and slippage when touching the book live.
- Chainlink/source-basis behavior near expiry versus Binance spot/perp moves.

## Data

| sample | run | asset | rows | markets | first UTC | last UTC |
| --- | --- | --- | --- | --- | --- | --- |
| oos | a0c_roll | BTC | 47,362 | 4 | 2026-05-29 20:00 | 2026-05-30 08:00 |
| oos | a0c_roll | ETH | 47,362 | 4 | 2026-05-29 20:00 | 2026-05-30 08:00 |
| oos | a0c_roll | SOL | 47,359 | 4 | 2026-05-29 20:00 | 2026-05-30 08:00 |
| train | a0b | BTC | 38,447 | 3 | 2026-05-27 20:00 | 2026-05-28 04:00 |
| train | a0c_roll | BTC | 38,738 | 3 | 2026-05-29 08:00 | 2026-05-29 16:00 |
| train | a0c_roll | ETH | 38,736 | 3 | 2026-05-29 08:00 | 2026-05-29 16:00 |
| train | a0c_roll | SOL | 38,741 | 3 | 2026-05-29 08:00 | 2026-05-29 16:00 |

## Outputs

- `data/analysis/csv_outputs/dali/block_i_leadlag_alignment.csv`
- `data/analysis/csv_outputs/dali/block_i_leadlag_signal_summary.csv`
- `data/analysis/csv_outputs/dali/block_i_leadlag_executable_market.csv`
- `data/analysis/csv_outputs/dali/block_i_leadlag_executable_summary.csv`
- `data/analysis/csv_outputs/dali/block_i_leadlag_selected_trades.csv`

Elapsed: `22.6s`.
