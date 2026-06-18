---
tags: [dali, block-a14b, executable-cost, results]
---
> Hub: [[COWORK]]


## Summary

- Scope: Block A1.4b Refined-Exit Taker Findings in the Dali research lineage area.
- Existing takeaway/status: A1.4b swaps A1.4's OFI trigger for A1.3's current top-of-book imbalance level and tests refined exits on the same six candidate markets. 0 of 36 market-config rows have positive mean executable PnL. The best single row is `a0:665325` with `cfg_signal_reversal` at -348.6 bps. Averaged across markets, the configuration ranking is:
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
> Table terms: [[polymarket_table_dictionary]]

# Block A1.4b Refined-Exit Taker Findings

## Headline

A1.4b swaps A1.4's OFI trigger for A1.3's current top-of-book imbalance level and tests refined exits on the same six candidate markets. 0 of 36 market-config rows have positive mean executable PnL. The best single row is `a0:665325` with `cfg_signal_reversal` at -348.6 bps. Averaged across markets, the configuration ranking is:

| config | markets | mean pnl | mean median | delta vs A1.4 | mean hold | win |
| --- | --- | --- | --- | --- | --- | --- |
| cfg_signal_reversal | 6 | -1059.7 bps | -754.2 bps | 325.8 bps | 16.16s | 1.5% |
| cfg_stop_loss_combined | 6 | -1102.8 bps | -779.2 bps | 282.7 bps | 15.01s | 1.4% |
| cfg_fixed_5s | 6 | -1106.4 bps | -782.1 bps | 279.1 bps | 5.00s | 2.2% |
| cfg_take_profit_500bps | 6 | -1110.3 bps | -778.9 bps | 275.2 bps | 18.40s | 0.3% |
| cfg_take_profit_200bps | 6 | -1143.4 bps | -836.2 bps | 242.1 bps | 16.25s | 0.1% |
| cfg_take_profit_100bps | 6 | -1154.9 bps | -843.5 bps | 230.6 bps | 14.70s | 0.0% |

## Method

- Candidate universe: same as A1.4, primary-read markets with `signal_present_pre_cost` at 5s in `block_a1_results.csv`.
- Signal: per-market top decile by absolute current TOB imbalance level, where `tob_imbalance_level = direction_factor * tob_imbalance`.
- Entry: instantaneous taker at the current asset touch. Long token signals pay `best_ask`; short token signals receive `best_bid`.
- Exits: fixed 5s, signal reversal or p90 time stop, take-profit at +100/+200/+500 bps or p90 time stop, and signal reversal plus -100 bps stop loss or p90 time stop.
- Take-profit and stop-loss checks use mid-price movement in the token-side trade direction. Executable PnL uses bid/ask touch on both entry and exit.
- Time stops use `p90_time_until_flip_sec` from `block_a13_tob_persistence_by_market.csv`.
- Fees: taker fee applied at entry and exit using A1's `FEE_BY_CATEGORY`.
- Confidence intervals: 200-sample block bootstrap of mean PnL using contiguous 300s clock-time blocks.

## Four-Config Gap Table

| market | slug | config | events | unfillable | fillable | mean pnl | median pnl | win | mean CI | mean hold | A1.4 base | delta | exit reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0:665325 | us-iran-nuclear-deal-before-2027 | cfg_fixed_5s | 25,274 | 28 | 99.9% | -385.2 bps | -344.8 bps | 0.0% | [-439.5 bps, -346.3 bps] | 5.00s | -493.6 bps | 108.3 bps | signal_reversal=0;take_profit=0;stop_loss=0;time_stop=25246 |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | cfg_fixed_5s | 46,643 | 0 | 100.0% | -946.7 bps | -614.7 bps | 0.6% | [-981.6 bps, -912.5 bps] | 5.00s | -996.5 bps | 49.7 bps | signal_reversal=0;take_profit=0;stop_loss=0;time_stop=46643 |
| a0b:2364426 | btc-updown-4h-1779912000 | cfg_fixed_5s | 5,399 | 0 | 100.0% | -1113.0 bps | -899.9 bps | 7.9% | [-1647.8 bps, -871.7 bps] | 5.00s | -1294.0 bps | 181.0 bps | signal_reversal=0;take_profit=0;stop_loss=0;time_stop=5399 |
| a0b:2367777 | btc-updown-4h-1779940800 | cfg_fixed_5s | 18,195 | 0 | 100.0% | -1239.9 bps | -1139.6 bps | 2.1% | [-1350.8 bps, -1139.3 bps] | 5.00s | -1605.4 bps | 365.5 bps | signal_reversal=0;take_profit=0;stop_loss=0;time_stop=18195 |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | cfg_fixed_5s | 31,937 | 0 | 100.0% | -1380.9 bps | -607.6 bps | 1.1% | [-1529.7 bps, -1251.6 bps] | 5.00s | -1773.6 bps | 392.7 bps | signal_reversal=0;take_profit=0;stop_loss=0;time_stop=31937 |
| a0b:2366225 | btc-updown-4h-1779926400 | cfg_fixed_5s | 18,090 | 4 | 100.0% | -1572.7 bps | -1086.3 bps | 1.3% | [-1900.3 bps, -1332.1 bps] | 5.00s | -2150.0 bps | 577.3 bps | signal_reversal=0;take_profit=0;stop_loss=0;time_stop=18086 |
| a0:665325 | us-iran-nuclear-deal-before-2027 | cfg_signal_reversal | 25,274 | 38 | 99.8% | -348.6 bps | -344.8 bps | 0.0% | [-384.1 bps, -320.6 bps] | 21.61s | -493.6 bps | 145.0 bps | signal_reversal=4476;take_profit=0;stop_loss=0;time_stop=20760 |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | cfg_signal_reversal | 46,643 | 0 | 100.0% | -920.7 bps | -618.1 bps | 0.0% | [-960.8 bps, -888.5 bps] | 22.81s | -996.5 bps | 75.7 bps | signal_reversal=20780;take_profit=0;stop_loss=0;time_stop=25863 |
| a0b:2364426 | btc-updown-4h-1779912000 | cfg_signal_reversal | 5,399 | 0 | 100.0% | -928.2 bps | -790.8 bps | 5.6% | [-1244.0 bps, -731.0 bps] | 22.53s | -1294.0 bps | 365.8 bps | signal_reversal=1888;take_profit=0;stop_loss=0;time_stop=3511 |
| a0b:2367777 | btc-updown-4h-1779940800 | cfg_signal_reversal | 18,195 | 0 | 100.0% | -1216.8 bps | -1093.3 bps | 2.5% | [-1325.0 bps, -1117.0 bps] | 10.16s | -1605.4 bps | 388.6 bps | signal_reversal=8541;take_profit=0;stop_loss=0;time_stop=9654 |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | cfg_signal_reversal | 31,937 | 26 | 99.9% | -1382.8 bps | -625.6 bps | 0.0% | [-1510.6 bps, -1273.6 bps] | 11.90s | -1773.6 bps | 390.8 bps | signal_reversal=9251;take_profit=0;stop_loss=0;time_stop=22660 |
| a0b:2366225 | btc-updown-4h-1779926400 | cfg_signal_reversal | 18,090 | 4 | 100.0% | -1561.0 bps | -1052.6 bps | 1.1% | [-1879.3 bps, -1295.3 bps] | 7.94s | -2150.0 bps | 589.0 bps | signal_reversal=6706;take_profit=0;stop_loss=0;time_stop=11380 |
| a0:665325 | us-iran-nuclear-deal-before-2027 | cfg_stop_loss_combined | 25,274 | 38 | 99.8% | -357.8 bps | -344.8 bps | 0.0% | [-399.1 bps, -325.7 bps] | 20.98s | -493.6 bps | 135.8 bps | signal_reversal=4049;take_profit=0;stop_loss=1059;time_stop=20128 |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | cfg_stop_loss_combined | 46,643 | 0 | 100.0% | -927.5 bps | -618.1 bps | 0.0% | [-967.4 bps, -896.0 bps] | 22.16s | -996.5 bps | 68.9 bps | signal_reversal=20557;take_profit=0;stop_loss=1414;time_stop=24672 |
| a0b:2364426 | btc-updown-4h-1779912000 | cfg_stop_loss_combined | 5,399 | 0 | 100.0% | -1023.3 bps | -822.2 bps | 5.2% | [-1360.5 bps, -796.9 bps] | 19.36s | -1294.0 bps | 270.8 bps | signal_reversal=1709;take_profit=0;stop_loss=1004;time_stop=2686 |
| a0b:2367777 | btc-updown-4h-1779940800 | cfg_stop_loss_combined | 18,195 | 0 | 100.0% | -1258.6 bps | -1155.1 bps | 2.5% | [-1354.3 bps, -1158.4 bps] | 9.15s | -1605.4 bps | 346.9 bps | signal_reversal=6427;take_profit=0;stop_loss=3367;time_stop=8401 |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | cfg_stop_loss_combined | 31,937 | 26 | 99.9% | -1414.5 bps | -677.9 bps | 0.0% | [-1530.2 bps, -1295.2 bps] | 10.94s | -1773.6 bps | 359.1 bps | signal_reversal=8577;take_profit=0;stop_loss=3361;time_stop=19973 |
| a0b:2366225 | btc-updown-4h-1779926400 | cfg_stop_loss_combined | 18,090 | 2 | 100.0% | -1635.4 bps | -1057.2 bps | 1.0% | [-1914.1 bps, -1388.2 bps] | 7.47s | -2150.0 bps | 514.6 bps | signal_reversal=5269;take_profit=0;stop_loss=2172;time_stop=10647 |
| a0:665325 | us-iran-nuclear-deal-before-2027 | cfg_take_profit_100bps | 25,274 | 38 | 99.8% | -361.5 bps | -344.8 bps | 0.0% | [-409.6 bps, -325.7 bps] | 19.86s | -493.6 bps | 132.1 bps | signal_reversal=0;take_profit=5750;stop_loss=0;time_stop=19486 |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | cfg_take_profit_100bps | 46,643 | 0 | 100.0% | -984.4 bps | -657.1 bps | 0.0% | [-1013.6 bps, -961.9 bps] | 23.51s | -996.5 bps | 12.1 bps | signal_reversal=0;take_profit=17602;stop_loss=0;time_stop=29041 |
| a0b:2364426 | btc-updown-4h-1779912000 | cfg_take_profit_100bps | 5,399 | 0 | 100.0% | -1244.8 bps | -1099.4 bps | 0.0% | [-1696.0 bps, -1096.3 bps] | 17.36s | -1294.0 bps | 49.2 bps | signal_reversal=0;take_profit=2883;stop_loss=0;time_stop=2516 |
| a0b:2367777 | btc-updown-4h-1779940800 | cfg_take_profit_100bps | 18,195 | 0 | 100.0% | -1323.2 bps | -1189.5 bps | 0.0% | [-1433.1 bps, -1219.9 bps] | 8.88s | -1605.4 bps | 282.3 bps | signal_reversal=0;take_profit=9575;stop_loss=0;time_stop=8620 |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | cfg_take_profit_100bps | 31,937 | 26 | 99.9% | -1393.7 bps | -656.2 bps | 0.0% | [-1507.2 bps, -1302.4 bps] | 11.83s | -1773.6 bps | 380.0 bps | signal_reversal=0;take_profit=9042;stop_loss=0;time_stop=22869 |
| a0b:2366225 | btc-updown-4h-1779926400 | cfg_take_profit_100bps | 18,090 | 4 | 100.0% | -1622.1 bps | -1114.2 bps | 0.0% | [-1924.6 bps, -1364.3 bps] | 6.77s | -2150.0 bps | 527.9 bps | signal_reversal=0;take_profit=8469;stop_loss=0;time_stop=9617 |
| a0:665325 | us-iran-nuclear-deal-before-2027 | cfg_take_profit_200bps | 25,274 | 38 | 99.8% | -359.7 bps | -344.8 bps | 0.0% | [-391.3 bps, -326.8 bps] | 22.70s | -493.6 bps | 133.8 bps | signal_reversal=0;take_profit=2178;stop_loss=0;time_stop=23058 |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | cfg_take_profit_200bps | 46,643 | 3 | 100.0% | -970.5 bps | -657.1 bps | 0.1% | [-1000.6 bps, -942.7 bps] | 26.45s | -996.5 bps | 25.9 bps | signal_reversal=0;take_profit=12435;stop_loss=0;time_stop=34205 |
| a0b:2364426 | btc-updown-4h-1779912000 | cfg_take_profit_200bps | 5,399 | 0 | 100.0% | -1218.8 bps | -1093.3 bps | 0.0% | [-1598.6 bps, -1082.6 bps] | 18.08s | -1294.0 bps | 75.2 bps | signal_reversal=0;take_profit=2740;stop_loss=0;time_stop=2659 |
| a0b:2367777 | btc-updown-4h-1779940800 | cfg_take_profit_200bps | 18,195 | 0 | 100.0% | -1312.3 bps | -1155.1 bps | 0.2% | [-1429.5 bps, -1201.5 bps] | 10.40s | -1605.4 bps | 293.1 bps | signal_reversal=0;take_profit=7555;stop_loss=0;time_stop=10640 |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | cfg_take_profit_200bps | 31,937 | 26 | 99.9% | -1390.6 bps | -660.4 bps | 0.0% | [-1508.1 bps, -1288.9 bps] | 12.50s | -1773.6 bps | 383.0 bps | signal_reversal=0;take_profit=7006;stop_loss=0;time_stop=24905 |
| a0b:2366225 | btc-updown-4h-1779926400 | cfg_take_profit_200bps | 18,090 | 4 | 100.0% | -1608.3 bps | -1106.3 bps | 0.0% | [-1875.5 bps, -1362.4 bps] | 7.35s | -2150.0 bps | 541.7 bps | signal_reversal=0;take_profit=7219;stop_loss=0;time_stop=10867 |
| a0:665325 | us-iran-nuclear-deal-before-2027 | cfg_take_profit_500bps | 25,274 | 38 | 99.8% | -356.8 bps | -344.8 bps | 0.0% | [-397.4 bps, -322.6 bps] | 24.04s | -493.6 bps | 136.7 bps | signal_reversal=0;take_profit=337;stop_loss=0;time_stop=24899 |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | cfg_take_profit_500bps | 46,643 | 3 | 100.0% | -909.2 bps | -657.1 bps | 0.4% | [-949.7 bps, -877.4 bps] | 29.37s | -996.5 bps | 87.3 bps | signal_reversal=0;take_profit=6563;stop_loss=0;time_stop=40077 |
| a0b:2364426 | btc-updown-4h-1779912000 | cfg_take_profit_500bps | 5,399 | 0 | 100.0% | -1155.9 bps | -828.0 bps | 0.0% | [-1627.4 bps, -1000.5 bps] | 22.00s | -1294.0 bps | 138.1 bps | signal_reversal=0;take_profit=2005;stop_loss=0;time_stop=3394 |
| a0b:2367777 | btc-updown-4h-1779940800 | cfg_take_profit_500bps | 18,195 | 0 | 100.0% | -1294.0 bps | -1121.0 bps | 1.0% | [-1411.5 bps, -1189.4 bps] | 12.65s | -1605.4 bps | 311.5 bps | signal_reversal=0;take_profit=3842;stop_loss=0;time_stop=14353 |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | cfg_take_profit_500bps | 31,937 | 26 | 99.9% | -1385.5 bps | -656.2 bps | 0.6% | [-1498.3 bps, -1276.8 bps] | 13.72s | -1773.6 bps | 388.1 bps | signal_reversal=0;take_profit=3841;stop_loss=0;time_stop=28070 |
| a0b:2366225 | btc-updown-4h-1779926400 | cfg_take_profit_500bps | 18,090 | 4 | 100.0% | -1560.3 bps | -1066.0 bps | 0.1% | [-1872.8 bps, -1256.9 bps] | 8.64s | -2150.0 bps | 589.7 bps | signal_reversal=0;take_profit=3977;stop_loss=0;time_stop=14109 |

## Per-Market Verdicts

- `a0:665325` (us-iran-nuclear-deal-before-2027): improves but stays negative; best `cfg_signal_reversal` at -348.6 bps, delta vs A1.4 145.0 bps.
- `a0b:2362124` (bitcoin-up-or-down-on-may-28-2026): improves but stays negative; best `cfg_take_profit_500bps` at -909.2 bps, delta vs A1.4 87.3 bps.
- `a0b:2362186` (ethereum-up-or-down-on-may-28-2026): improves but stays negative; best `cfg_fixed_5s` at -1380.9 bps, delta vs A1.4 392.7 bps.
- `a0b:2364426` (btc-updown-4h-1779912000): improves but stays negative; best `cfg_signal_reversal` at -928.2 bps, delta vs A1.4 365.8 bps.
- `a0b:2366225` (btc-updown-4h-1779926400): improves but stays negative; best `cfg_take_profit_500bps` at -1560.3 bps, delta vs A1.4 589.7 bps.
- `a0b:2367777` (btc-updown-4h-1779940800): improves but stays negative; best `cfg_signal_reversal` at -1216.8 bps, delta vs A1.4 388.6 bps.

## Interpretation

The refined exits move the needle mostly by shortening exposure when the TOB state is unstable. That improves every market relative to the OFI fixed-horizon A1.4 baseline, but positive executable taker PnL is absent in this run. This is still a touch-fill, no-latency, no-size-capacity diagnostic rather than a tradeable verdict.

Recommended next action for Justin: carry TOB imbalance into A2, but gate any taker version on tight-spread/deep-book cells and require the refined-exit executable test to stay positive after latency.
