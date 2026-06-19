---
tags: [dali, block-a14f, executable-cost, results]
---
> Hub: [[COWORK]]


## Summary

- Scope: Block A1.4f Combined Refined-Exit + Tight-Spread Findings in the Dali research lineage area.
- Existing takeaway/status: A1.4f tests all primary-read markets with per-market top-decile current TOB imbalance, spread-filtered entry, and refined exits. 5 of 660 market-horizon-threshold-exit cells crossed zero by mean PnL; 4 had at least 30 signal events, 0 had a bootstrap CI lower bound above zero, and 0 used a refined dynamic exit. The positive cells are all the same 300s fixed-horizon BTC 4h market from A14d, so the combination does...
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
> Table terms: [[polymarket_table_dictionary]]

# Block A1.4f Combined Refined-Exit + Tight-Spread Findings

## Headline

A1.4f tests all primary-read markets with per-market top-decile current TOB imbalance, spread-filtered entry, and refined exits. 5 of 660 market-horizon-threshold-exit cells crossed zero by mean PnL; 4 had at least 30 signal events, 0 had a bootstrap CI lower bound above zero, and 0 used a refined dynamic exit. The positive cells are all the same 300s fixed-horizon BTC 4h market from A14d, so the combination does not create a new refined-exit winner. 84 cells improved versus both available baselines on paper, but only 32 beat A14d by more than 10 bps. Best cell: `a0b:2364426` h=300s, S=500, `cfg_fixed_5s` with 844.0 bps on 4,326 signal events.

## Method

- Candidate universe: all `primary_read` markets from `block_a1_results.csv`.
- Signal: per-market top decile by `abs(tob_imbalance_level)`, with `tob_imbalance_level = direction_factor * tob_imbalance`.
- Entry filter: `spread_bps <= S`, where `S in {100, 200, 500, 1000, no_filter}` and `spread_bps = (best_ask - best_bid) / mid * 10_000`.
- Exit configs: `cfg_fixed_5s`, `cfg_signal_reversal`, `cfg_take_profit_500bps`, and `cfg_stop_loss_combined`.
- Horizon handling: the inherited `cfg_fixed_5s` label is kept for comparability, but the row's `horizon` controls the fixed hold length. Dynamic exits close on their event trigger or `min(p90 persistence, horizon)`.
- Execution: touch round trip with taker fees on both legs; no partial fills, queue, size capacity, or latency model.
- Bootstrap: 200-sample contiguous 300s block bootstrap on mean PnL.
- Baselines: `a14b_mean_pnl_bps` is the no-spread refined-exit baseline where available; `a14d_mean_pnl_bps` is the fixed-horizon tight-spread baseline for the same market/horizon/spread threshold.

## Best Configuration Cells

| h | S | exit | positive | positive n>=30 | mean pnl | delta vs A14d | hold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | 100 | cfg_fixed_5s | 1/11 | 0/11 | -212.5 bps | -0.0 bps | 300.00s |
| 300 | 200 | cfg_fixed_5s | 1/11 | 1/11 | -261.0 bps | 0.1 bps | 300.00s |
| 300 | 500 | cfg_fixed_5s | 1/11 | 1/11 | -390.6 bps | 0.0 bps | 300.00s |
| 300 | 1000 | cfg_fixed_5s | 1/11 | 1/11 | -508.3 bps | 0.0 bps | 300.00s |
| 300 | no_filter | cfg_fixed_5s | 1/11 | 1/11 | -641.6 bps | 0.0 bps | 300.00s |
| 5 | 100 | cfg_fixed_5s | 0/11 | 0/11 | -279.9 bps | -0.0 bps | 5.00s |
| 5 | 100 | cfg_take_profit_500bps | 0/11 | 0/11 | -280.8 bps | -0.9 bps | 4.44s |
| 300 | 100 | cfg_take_profit_500bps | 0/11 | 0/11 | -294.0 bps | -81.5 bps | 111.19s |
| 30 | 100 | cfg_take_profit_500bps | 0/11 | 0/11 | -294.0 bps | 18.8 bps | 21.22s |
| 5 | 100 | cfg_signal_reversal | 0/11 | 0/11 | -302.5 bps | -22.6 bps | 3.59s |
| 5 | 100 | cfg_stop_loss_combined | 0/11 | 0/11 | -302.5 bps | -22.6 bps | 3.55s |
| 30 | 100 | cfg_signal_reversal | 0/11 | 0/11 | -304.8 bps | 7.9 bps | 17.07s |
| 30 | 100 | cfg_stop_loss_combined | 0/11 | 0/11 | -304.9 bps | 7.9 bps | 17.01s |
| 300 | 100 | cfg_signal_reversal | 0/11 | 0/11 | -309.3 bps | -96.8 bps | 90.76s |
| 300 | 100 | cfg_stop_loss_combined | 0/11 | 0/11 | -309.3 bps | -96.8 bps | 90.69s |
| 30 | 100 | cfg_fixed_5s | 0/11 | 0/11 | -312.8 bps | 0.0 bps | 30.00s |
| 30 | 200 | cfg_fixed_5s | 0/11 | 0/11 | -337.9 bps | 0.0 bps | 30.00s |
| 5 | 200 | cfg_fixed_5s | 0/11 | 0/11 | -344.3 bps | -0.0 bps | 5.00s |

## Best Individual Cells

| market | slug | h | S | exit | events | fillable | mean | median | win | CI | dB | dD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0b:2364426 | btc-updown-4h-1779912000 | 300 | 500 | cfg_fixed_5s | 4,326 | 100.0% | 844.0 bps | 1179.3 bps | 64.4% | [-611.0 bps, 1741.8 bps] | 1957.0 bps | 0.4 bps |
| a0b:2364426 | btc-updown-4h-1779912000 | 300 | 1000 | cfg_fixed_5s | 5,742 | 100.0% | 613.7 bps | 1111.3 bps | 60.8% | [-604.9 bps, 1433.4 bps] | 1726.7 bps | 0.3 bps |
| a0b:2364426 | btc-updown-4h-1779912000 | 300 | 200 | cfg_fixed_5s | 1,910 | 99.9% | 579.3 bps | 494.6 bps | 52.6% | [-446.2 bps, 1330.1 bps] | 1692.3 bps | 0.6 bps |
| a0b:2364426 | btc-updown-4h-1779912000 | 300 | no_filter | cfg_fixed_5s | 6,684 | 100.0% | 334.4 bps | 1033.3 bps | 58.8% | [-1322.5 bps, 1142.8 bps] | 1447.4 bps | 0.2 bps |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 300 | 100 | cfg_fixed_5s | 6,016 | 100.0% | -45.7 bps | -41.8 bps | 1.6% | [-55.2 bps, -40.2 bps] | 901.1 bps | -0.1 bps |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 30 | 100 | cfg_take_profit_500bps | 6,016 | 100.0% | -53.1 bps | -46.0 bps | 0.1% | [-66.7 bps, -48.7 bps] | 856.0 bps | 0.1 bps |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 300 | 100 | cfg_take_profit_500bps | 6,016 | 100.0% | -53.2 bps | -46.0 bps | 0.1% | [-62.3 bps, -49.2 bps] | 855.9 bps | -7.6 bps |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 30 | 100 | cfg_fixed_5s | 6,016 | 100.0% | -53.2 bps | -46.0 bps | 0.1% | [-64.5 bps, -48.7 bps] | 893.5 bps | -0.0 bps |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 30 | 100 | cfg_signal_reversal | 6,016 | 100.0% | -53.9 bps | -46.0 bps | 0.0% | [-64.9 bps, -49.2 bps] | 866.9 bps | -0.6 bps |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 30 | 100 | cfg_stop_loss_combined | 6,016 | 100.0% | -53.9 bps | -46.0 bps | 0.0% | [-65.7 bps, -49.4 bps] | 873.6 bps | -0.6 bps |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 300 | 100 | cfg_signal_reversal | 6,016 | 100.0% | -54.0 bps | -46.0 bps | 0.0% | [-65.8 bps, -49.6 bps] | 866.7 bps | -8.4 bps |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 300 | 100 | cfg_stop_loss_combined | 6,016 | 100.0% | -54.0 bps | -46.0 bps | 0.0% | [-64.6 bps, -49.6 bps] | 873.5 bps | -8.4 bps |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 5 | 100 | cfg_take_profit_500bps | 6,016 | 100.0% | -54.1 bps | -46.0 bps | 0.0% | [-69.4 bps, -49.4 bps] | 855.0 bps | 0.0 bps |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 5 | 100 | cfg_fixed_5s | 6,016 | 100.0% | -54.1 bps | -46.0 bps | 0.0% | [-66.9 bps, -49.1 bps] | 892.6 bps | -0.0 bps |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 5 | 100 | cfg_signal_reversal | 6,016 | 100.0% | -54.4 bps | -46.0 bps | 0.0% | [-68.7 bps, -49.9 bps] | 866.4 bps | -0.2 bps |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 5 | 100 | cfg_stop_loss_combined | 6,016 | 100.0% | -54.4 bps | -46.0 bps | 0.0% | [-67.2 bps, -49.6 bps] | 873.2 bps | -0.2 bps |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | 300 | 100 | cfg_fixed_5s | 4,519 | 100.0% | -82.7 bps | -71.8 bps | 0.2% | [-109.2 bps, -61.9 bps] | 1298.2 bps | -0.0 bps |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | 5 | 100 | cfg_take_profit_500bps | 4,519 | 100.0% | -87.0 bps | -71.8 bps | 0.0% | [-111.5 bps, -71.3 bps] | 1298.5 bps | 0.2 bps |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | 5 | 100 | cfg_signal_reversal | 4,519 | 100.0% | -87.2 bps | -71.8 bps | 0.0% | [-110.1 bps, -71.3 bps] | 1295.6 bps | -0.0 bps |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | 5 | 100 | cfg_fixed_5s | 4,519 | 100.0% | -87.3 bps | -71.8 bps | 0.1% | [-115.1 bps, -72.2 bps] | 1293.6 bps | -0.1 bps |

## Per-Market Combined Verdict

| market | slug | h | S | exit | events | mean | dB | dD | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0:558934 | will-spain-win-the-2026-fifa-world-cup-963 | 5 | 100 | cfg_fixed_5s | 8,001 | -339.0 bps | n/a | 0.0 bps | still wiped |
| a0:558936 | will-france-win-the-2026-fifa-world-cup-924 | 300 | 100 | cfg_fixed_5s | 8,015 | -335.4 bps | n/a | 0.0 bps | still wiped |
| a0:665325 | us-iran-nuclear-deal-before-2027 | 300 | 200 | cfg_fixed_5s | 8,663 | -128.2 bps | 257.0 bps | 0.0 bps | still wiped |
| a0b:1971905 | strait-of-hormuz-traffic-returns-to-normal-b | 5 | 100 | cfg_fixed_5s | 4 | 0.0 bps | n/a | 0.0 bps | still wiped |
| a0b:2327929 | nba-okc-sas-2026-05-28 | 5 | 200 | cfg_fixed_5s | 5,364 | -416.4 bps | n/a | 0.0 bps | still wiped |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 300 | 100 | cfg_fixed_5s | 6,016 | -45.7 bps | 901.1 bps | -0.1 bps | still wiped |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | 300 | 100 | cfg_fixed_5s | 4,519 | -82.7 bps | 1298.2 bps | -0.0 bps | still wiped |
| a0b:2364426 | btc-updown-4h-1779912000 | 300 | 500 | cfg_fixed_5s | 4,326 | 844.0 bps | 1957.0 bps | 0.4 bps | crosses zero, CI wide |
| a0b:2366225 | btc-updown-4h-1779926400 | 5 | 100 | cfg_fixed_5s | 266 | -111.6 bps | 1461.1 bps | 0.0 bps | still wiped |
| a0b:2367777 | btc-updown-4h-1779940800 | 5 | 200 | cfg_fixed_5s | 4,085 | -541.5 bps | 698.5 bps | -0.1 bps | still wiped |
| a0b:566136 | will-psg-win-the-202526-champions-league | 5 | 200 | cfg_fixed_5s | 1,660 | -425.2 bps | n/a | 0.0 bps | still wiped |

## Interpretation

The combined filter is the first place where the test can ask whether A14b's exit discipline and A14d's tight-spread entry stack constructively. Positive cells should still be read through event count, CI width, and whether they beat both sidecar baselines. A sparse positive row is not a strategy; a repeated positive row across neighboring thresholds/horizons is the thing worth carrying into A2.

Recommended next action for Justin: keep TOB plus tight-spread/refined-exit as an A2 diagnostic, but require repeated positive cells with CI support and add latency/capacity before any taker design.
