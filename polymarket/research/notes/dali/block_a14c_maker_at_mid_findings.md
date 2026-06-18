---
tags: [dali, block-a14c, maker-thesis, results]
title: Block A1.4c Maker-at-Mid Findings
created: 2026-05-28
status: archived
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
---

# Block A1.4c Maker-at-Mid Findings

> Hub: [[COWORK]]

> Table terms: [[polymarket_table_dictionary]]


## Summary

A14c tests a generous maker-at-mid counterfactual for the TOB imbalance signal, assuming full priority inside the spread. It finds one promising BTC-4h cell with +554.9 bps mean PnL, 9.0% fill rate, and 248.1 bps 5s adverse selection, but flags this as a best-case maker assumption. The note is preserved as the positive clue that later required the non-overlap retest in [[block_a14h_maker_non_overlap_findings]].

## Headline

The maker-at-mid counterfactual is materially more interesting than taker execution, but it is a best-case maker assumption. The CSV has 768 rows because it preserves the four A1.3 current-level horizon labels; collapsing those duplicate labels leaves 192 unique market/grid cells, of which 3 have positive mean PnL. The best cell is `a0b:2364426` / `btc-updown-4h-1779912000` with W=10s, H=30s, `exit_symmetric_maker`, mean 554.9 bps, fill rate 9.0%, and 5s signed adverse selection 248.1 bps. Per-market verdicts: 1 maker-thesis-live, 5 fills-too-rare, 10 adverse-selection-wipes-rebate.

## Method

- Universe: A1 markets labeled `primary_read` or `thin_wide_CI` at the 5s horizon.
- Signal: A1.3 current-level TOB imbalance, `direction_factor * tob_imbalance`, using per-market top absolute decile.
- Signal horizon: carried from A1.3 current-level horizons. Because current-level TOB is a state variable, the signal rows are horizon-invariant; repeated horizon labels are included for traceability.
- Entry: post at current mid on the signal-favorable token side. Long token posts bid at mid; short token posts ask at mid.
- Fill windows: 1s, 5s, 10s. Unfilled signal events are retained in `n_unfilled` and `fill_rate`.
- Exit conventions: `exit_forced_taker` closes at the opposite touch after H; `exit_symmetric_maker` posts opposite-side at mid after H and, if not filled within H, forces a taker close at t_fill + 2H.
- Fees/rebates: entry maker rebate is credited once. Taker fee is charged on forced-taker exits and on symmetric-maker fallback exits.
- Queue model: none. Inside-spread quotes are assumed to have full priority, so these are best-case maker numbers.
- Bootstrap: 200 resamples over contiguous 300s blocks on filled PnL.

## Grid Winners

| fill W | hold H | exit | mean pnl | median pnl | fill rate | fills | positive cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 30 | exit_symmetric_maker | -199.1 bps | -202.2 bps | 5.2% | 13,569 | 1/16 |
| 1 | 30 | exit_symmetric_maker | -221.2 bps | -232.2 bps | 2.0% | 4,798 | 0/16 |
| 5 | 30 | exit_symmetric_maker | -228.0 bps | -216.3 bps | 4.0% | 10,341 | 0/16 |
| 1 | 30 | exit_forced_taker | -247.7 bps | -219.6 bps | 2.0% | 4,798 | 0/16 |
| 10 | 30 | exit_forced_taker | -256.6 bps | -202.2 bps | 5.2% | 13,569 | 1/16 |
| 5 | 30 | exit_forced_taker | -261.4 bps | -216.3 bps | 4.0% | 10,341 | 1/16 |
| 5 | 5 | exit_symmetric_maker | -286.0 bps | -221.0 bps | 4.0% | 10,341 | 0/16 |
| 1 | 5 | exit_symmetric_maker | -292.2 bps | -216.3 bps | 2.0% | 4,798 | 0/16 |
| 10 | 5 | exit_symmetric_maker | -296.5 bps | -232.2 bps | 5.2% | 13,569 | 0/16 |
| 1 | 5 | exit_forced_taker | -298.3 bps | -216.3 bps | 2.0% | 4,798 | 0/16 |
| 5 | 5 | exit_forced_taker | -300.1 bps | -232.2 bps | 4.0% | 10,341 | 0/16 |
| 10 | 5 | exit_forced_taker | -312.1 bps | -232.2 bps | 5.2% | 13,569 | 0/16 |

## Top Cells

| market | slug | signal | W | H | exit | signals | fills | fill rate | mean pnl | 5s adverse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0b:2364426 | btc-updown-4h-1779912000 | current | 10 | 30 | symmetric_maker | 6,556 | 588 | 9.0% | 554.9 bps | 248.1 bps |
| a0b:2364426 | btc-updown-4h-1779912000 | current | 10 | 30 | forced_taker | 6,556 | 588 | 9.0% | 301.9 bps | 248.1 bps |
| a0b:2364426 | btc-updown-4h-1779912000 | current | 5 | 30 | forced_taker | 6,556 | 306 | 4.7% | 17.4 bps | 147.8 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in- | current | 10 | 5 | symmetric_maker | 7,652 | 81 | 1.1% | -11.2 bps | 0.0 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in- | current | 10 | 5 | forced_taker | 7,652 | 81 | 1.1% | -11.2 bps | 0.0 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in- | current | 10 | 30 | symmetric_maker | 7,652 | 81 | 1.1% | -11.2 bps | 0.0 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in- | current | 10 | 30 | forced_taker | 7,652 | 81 | 1.1% | -11.2 bps | 0.0 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in- | current | 5 | 5 | symmetric_maker | 7,652 | 50 | 0.7% | -13.8 bps | 0.0 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in- | current | 5 | 5 | forced_taker | 7,652 | 50 | 0.7% | -13.8 bps | 0.0 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in- | current | 5 | 30 | symmetric_maker | 7,652 | 50 | 0.7% | -13.8 bps | 0.0 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in- | current | 5 | 30 | forced_taker | 7,652 | 50 | 0.7% | -13.8 bps | 0.0 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in- | current | 1 | 5 | symmetric_maker | 7,652 | 37 | 0.5% | -16.1 bps | 0.0 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in- | current | 1 | 5 | forced_taker | 7,652 | 37 | 0.5% | -16.1 bps | 0.0 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in- | current | 1 | 30 | symmetric_maker | 7,652 | 37 | 0.5% | -16.1 bps | 0.0 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in- | current | 1 | 30 | forced_taker | 7,652 | 37 | 0.5% | -16.1 bps | 0.0 bps |
| a0b:2176262 | strait-of-hormuz-traffic-returns-to-normal | current | 5 | 5 | symmetric_maker | 12,544 | 37 | 0.3% | -19.8 bps | 86.1 bps |

## Per-Market Verdicts

- `a0:1090496` (nato-x-russia-military-clash-by-december-31-2026-244): fills too rare; best cell W=10s, H=5s, exit_symmetric_maker, mean -135.1 bps, fill 0.9%, 5s adverse 0.0 bps.
- `a0:1469737` (will-mojtaba-khamenei-be-head-of-state-in-iran-end-of-): adverse selection wipes rebate; best cell W=10s, H=5s, exit_symmetric_maker, mean -11.2 bps, fill 1.1%, 5s adverse 0.0 bps.
- `a0:558934` (will-spain-win-the-2026-fifa-world-cup-963): adverse selection wipes rebate; best cell W=1s, H=30s, exit_symmetric_maker, mean -181.4 bps, fill 4.6%, 5s adverse 0.0 bps.
- `a0:558936` (will-france-win-the-2026-fifa-world-cup-924): adverse selection wipes rebate; best cell W=10s, H=30s, exit_symmetric_maker, mean -173.1 bps, fill 4.1%, 5s adverse 0.0 bps.
- `a0:631139` (will-google-have-the-best-ai-model-at-the-end-of-june-): fills too rare; best cell W=1s, H=5s, exit_symmetric_maker, mean -453.9 bps, fill 0.4%, 5s adverse 0.0 bps.
- `a0:631140` (will-anthropic-have-the-best-ai-model-at-the-end-of-ju): fills too rare; best cell W=1s, H=30s, exit_symmetric_maker, mean -171.7 bps, fill 0.1%, 5s adverse 1.1 bps.
- `a0:665325` (us-iran-nuclear-deal-before-2027): adverse selection wipes rebate; best cell W=5s, H=5s, exit_forced_taker, mean -291.5 bps, fill 2.0%, 5s adverse -60.3 bps.
- `a0b:1971905` (strait-of-hormuz-traffic-returns-to-normal-by-end-of-j): adverse selection wipes rebate; best cell W=5s, H=30s, exit_symmetric_maker, mean -80.1 bps, fill 4.2%, 5s adverse 5.9 bps.
- `a0b:2176262` (strait-of-hormuz-traffic-returns-to-normal-by-july-31): fills too rare; best cell W=5s, H=5s, exit_symmetric_maker, mean -19.8 bps, fill 0.3%, 5s adverse 86.1 bps.
- `a0b:2327929` (nba-okc-sas-2026-05-28): adverse selection wipes rebate; best cell W=10s, H=30s, exit_symmetric_maker, mean -216.9 bps, fill 4.1%, 5s adverse 0.0 bps.
- `a0b:2362124` (bitcoin-up-or-down-on-may-28-2026): adverse selection wipes rebate; best cell W=1s, H=30s, exit_symmetric_maker, mean -229.8 bps, fill 1.3%, 5s adverse 88.4 bps.
- `a0b:2362186` (ethereum-up-or-down-on-may-28-2026): fills too rare; best cell W=1s, H=30s, exit_symmetric_maker, mean -88.2 bps, fill 0.6%, 5s adverse 84.4 bps.
- `a0b:2364426` (btc-updown-4h-1779912000): maker thesis lives; best cell W=10s, H=30s, exit_symmetric_maker, mean 554.9 bps, fill 9.0%, 5s adverse 248.1 bps.
- `a0b:2366225` (btc-updown-4h-1779926400): adverse selection wipes rebate; best cell W=1s, H=30s, exit_symmetric_maker, mean -420.1 bps, fill 1.9%, 5s adverse -80.4 bps.
- `a0b:2367777` (btc-updown-4h-1779940800): adverse selection wipes rebate; best cell W=1s, H=30s, exit_symmetric_maker, mean -383.1 bps, fill 11.6%, 5s adverse 17.9 bps.
- `a0b:566136` (will-psg-win-the-202526-champions-league): adverse selection wipes rebate; best cell W=5s, H=30s, exit_symmetric_maker, mean -212.5 bps, fill 3.3%, 5s adverse 0.0 bps.

## Interpretation

Positive maker-at-mid cells mean the spread capture plus entry rebate beat adverse selection under full-priority inside-spread posting. They do not imply deployability: queue priority, quote cancellation, touch size, latency, and inventory constraints are still absent. Low fill-rate rows should be treated as opportunity diagnostics, not scalable PnL estimates.

Recommended next action for Justin: A2 should include a maker-focused branch, but only with queue/latency instrumentation and a minimum fill-rate screen rather than a pure OFI taker thesis.
