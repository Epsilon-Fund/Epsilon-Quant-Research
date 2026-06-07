---
tags: [dali, block-a15b, micro-price, results]
title: Block A1.5b Decoupled Micro-Price Target Findings
created: 2026-05-28
status: archived
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
---

# Block A1.5b Decoupled Micro-Price Target Findings

> Hub: [[COWORK]]

> Table terms: [[polymarket_table_dictionary]]


## Summary

A15b checks whether OFI/TFI signal performance is contaminated by using a target that shares TOB imbalance terms. OFI's 5s hit rate is essentially unchanged against future micro-price versus future mid, while 1s improves and longer horizons weaken; TFI degrades more clearly. The note diagnoses the signal as mostly mean-reversion to micro-price rather than multi-second continuation alpha.

## Headline

OFI's 5s top-decile hit rate is essentially unchanged when the target is future micro-price instead of future mid: 64.2% vs 64.1%, +0.09 pp. The one clear micro-target hit-rate improvement is at 1s (61.8% vs 56.4%, +5.45 pp). The decoupled test does not fully clear the mean-reversion concern for OFI across horizons: hit-rate improves at 1s, is flat at 5s, and weakens at 30s/300s, even though directional return bps are larger against the micro-price target. TFI does not show the same clean micro-target improvement pattern. This removes the A1.5 contamination path because TOB imbalance is target-only here; entries are driven by OFI or TFI.

## Calibration Gate

OFI with `mid_target` recomputes A1's depth-normalized decile aggregate from `block_a1_features.parquet`. The gate checks every decile row; the table below shows the top-decile rows.

| h | A15b n | A1 n | hit delta | dir delta | pass |
| --- | --- | --- | --- | --- | --- |
| 1 | 122,872 | 122,872 | +0.00 pp | 0.0 bps | yes |
| 5 | 156,638 | 156,638 | +0.00 pp | 0.0 bps | yes |
| 30 | 218,378 | 218,378 | +0.00 pp | 0.0 bps | yes |
| 300 | 275,591 | 275,591 | +0.00 pp | 0.0 bps | yes |

## OFI Signal

| h | mid hit | micro hit | delta | mid dir | micro dir | dir delta | mid n | micro n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 56.4% | 61.8% | +5.45 pp | 20.5 bps | 135.6 bps | 115.1 bps | 122,872 | 122,872 |
| 5 | 64.1% | 64.2% | +0.09 pp | 94.9 bps | 171.0 bps | 76.1 bps | 156,638 | 156,638 |
| 30 | 57.0% | 54.8% | -2.14 pp | 66.4 bps | 92.1 bps | 25.7 bps | 218,378 | 218,378 |
| 300 | 51.8% | 50.8% | -1.04 pp | 89.3 bps | 159.1 bps | 69.9 bps | 275,591 | 275,591 |

## TFI Signal

| h | mid hit | micro hit | delta | mid dir | micro dir | dir delta | mid n | micro n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 65.9% | 53.3% | -12.57 pp | 109.5 bps | 42.1 bps | -67.4 bps | 7,028 | 7,028 |
| 5 | 65.7% | 52.4% | -13.32 pp | 284.7 bps | 262.9 bps | -21.8 bps | 17,996 | 17,996 |
| 30 | 51.6% | 47.1% | -4.49 pp | -60.2 bps | -145.8 bps | -85.6 bps | 50,412 | 50,412 |
| 300 | 55.5% | 51.7% | -3.77 pp | 425.7 bps | 378.1 bps | -47.7 bps | 158,943 | 158,943 |

## Method

- `signal_ofi_h = direction_factor * rolling_sum(ofi_combined_event, h) / mean_depth_at_touch`.
- `signal_tfi_h = direction_factor * rolling_sum(signed_live_trade_size, h) / mean_depth_at_touch`.
- `mid_target` uses A1's future directional mid return.
- `micro_target` uses `future_micro_price = future_mid + 0.5 * future_spread * future_tob_imbalance`, direction-adjusted, with current directional mid as the denominator.
- Deciles are global equal-count buckets within each `(signal, target, horizon)` and are based on absolute signal magnitude.

## Outputs

- `data/analysis/csv_outputs/dali/block_a15b_decoupled_results.csv`
- `data/analysis/csv_outputs/dali/block_a15b_baseline_check.csv`

Recommended next action for Justin: carry OFI-vs-micro-price-target as an A2 audit column alongside L1 TOB, and treat TFI as a conditional sidecar rather than a primary signal.
