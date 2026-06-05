---
tags: [dali, block-a14, executable-cost, results]
---
> Hub: [[COWORK]]


> Table terms: [[polymarket_table_dictionary]]

# Block A1.4 Executable Taker Findings

## Headline

A1.4 replaces the A1 mid-return cost overlay with an executable touch-to-touch round trip on the six 5s primary-read pre-cost candidates. 0 of 12 market-horizon cells have positive mean executable PnL after crossing the spread on both entry and exit. At the original A1 5s horizon, 0 of 6 candidates survive; the mean gap versus A1's 5s overlay is -560.8 bps, where positive means executable round-trip PnL was better than the A1 overlay. In this run, the top-decile mid-mid signal is real as a descriptive pattern but is not close to surviving a simple taker round trip.

## Method

- Candidate universe: markets with `verdict = signal_present_pre_cost` and `sample_size_label = primary_read` in `block_a1_results.csv` at the 5s horizon.
- Signal: top decile by absolute `OFI_scaled` per market and horizon, where `OFI_scaled = direction_factor * rolling_OFI / mean_depth_at_touch`.
- Horizons: 5s and 10s. A1 has a cost-overlay comparison at 5s; 10s is reported as executable-only because A1 did not produce a 10s cost-overlay row.
- Entry/exit: after the YES/NO direction flip, the action is converted back to the current asset's token side. A long token signal pays `best_ask` at entry and receives future `best_bid`; a short token signal receives `best_bid` at entry and pays future `best_ask`.
- Fees: Polymarket taker fee is applied at entry and exit using the A1 `FEE_BY_CATEGORY` table as dollars per share, then normalized by entry price.
- Fill model: full size at touch, no partial fills, no queue model, and no latency layer.
- Exit quote: last observed book state at or before `t + horizon`, matching A1's forward-state convention.
- Confidence intervals: 200-sample block bootstrap of mean PnL using contiguous 300s clock-time blocks.

## Gap Table

| market | slug | h | events | unfillable | fillable | mean pnl | median pnl | win | mean CI | A1 overlay | gap | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0:665325 | us-iran-nuclear-deal-before-2027 | 5 | 12,541 | 0 | 100.0% | -493.6 bps | -411.0 bps | 0.0% | [-533.7 bps, -455.5 bps] | -97.1 bps | -396.5 bps | wiped by spread |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 5 | 40,082 | 0 | 100.0% | -996.5 bps | -610.1 bps | 3.2% | [-1083.8 bps, -929.5 bps] | -123.5 bps | -873.0 bps | wiped by spread |
| a0b:2364426 | btc-updown-4h-1779912000 | 5 | 4,131 | 280 | 93.2% | -1294.0 bps | -863.6 bps | 10.8% | [-1753.5 bps, -952.1 bps] | -993.7 bps | -300.3 bps | wiped by spread |
| a0b:2367777 | btc-updown-4h-1779940800 | 5 | 15,485 | 206 | 98.7% | -1605.4 bps | -1308.7 bps | 1.6% | [-1871.8 bps, -1420.6 bps] | -788.7 bps | -816.7 bps | wiped by spread |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | 5 | 27,363 | 0 | 100.0% | -1773.6 bps | -734.3 bps | 0.3% | [-2086.9 bps, -1505.8 bps] | -1805.9 bps | 32.3 bps | wiped by spread |
| a0b:2366225 | btc-updown-4h-1779926400 | 5 | 15,220 | 305 | 98.0% | -2150.0 bps | -1337.1 bps | 0.0% | [-2644.7 bps, -1676.6 bps] | -1139.5 bps | -1010.5 bps | wiped by spread |
| a0:665325 | us-iran-nuclear-deal-before-2027 | 10 | 15,644 | 0 | 100.0% | -462.6 bps | -400.0 bps | 0.0% | [-496.6 bps, -431.6 bps] | n/a | n/a | wiped by spread |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | 10 | 42,578 | 93 | 99.8% | -1017.2 bps | -574.1 bps | 5.1% | [-1103.2 bps, -944.8 bps] | n/a | n/a | wiped by spread |
| a0b:2364426 | btc-updown-4h-1779912000 | 10 | 5,315 | 380 | 92.9% | -1413.9 bps | -848.7 bps | 3.3% | [-1781.0 bps, -1222.9 bps] | n/a | n/a | wiped by spread |
| a0b:2367777 | btc-updown-4h-1779940800 | 10 | 18,386 | 220 | 98.8% | -1420.3 bps | -1189.5 bps | 2.9% | [-1675.3 bps, -1248.5 bps] | n/a | n/a | wiped by spread |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | 10 | 29,112 | 0 | 100.0% | -1763.0 bps | -589.6 bps | 1.5% | [-2267.5 bps, -1395.1 bps] | n/a | n/a | wiped by spread |
| a0b:2366225 | btc-updown-4h-1779926400 | 10 | 18,096 | 377 | 97.9% | -1919.7 bps | -1282.2 bps | 0.0% | [-2416.3 bps, -1515.8 bps] | n/a | n/a | wiped by spread |

## Per-Market Verdicts

- `a0:665325` (us-iran-nuclear-deal-before-2027): wiped by spread; 5s -493.6 bps, 10s -462.6 bps.
- `a0b:2362124` (bitcoin-up-or-down-on-may-28-2026): wiped by spread; 5s -996.5 bps, 10s -1017.2 bps.
- `a0b:2362186` (ethereum-up-or-down-on-may-28-2026): wiped by spread; 5s -1773.6 bps, 10s -1763.0 bps.
- `a0b:2364426` (btc-updown-4h-1779912000): wiped by spread; 5s -1294.0 bps, 10s -1413.9 bps.
- `a0b:2366225` (btc-updown-4h-1779926400): wiped by spread; 5s -2150.0 bps, 10s -1919.7 bps.
- `a0b:2367777` (btc-updown-4h-1779940800): wiped by spread; 5s -1605.4 bps, 10s -1420.3 bps.

## Interpretation

This is the executable-cost QA that A1's overlay could not provide. A positive row means the observed mid-mid OFI move was large enough to overcome both entry and exit spread crossing plus taker fees under the simplified touch-fill assumption. A negative row means the apparent mid-return alpha was wiped by executable round-trip cost in this capture window.

The 10s rows should be read as an A1.4 extension, not a direct A1 overlay audit, because the original A1 result table only contains 1s, 5s, 30s, and 300s horizons.

Recommended next action for Justin: do not advance the A1 top-decile signal as a taker edge; make A2 a longer signal-characterization capture with an explicit tight-spread executable-cost screen.
