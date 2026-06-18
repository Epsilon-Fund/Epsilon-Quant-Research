---
tags: [dali, block-a14i, pyramiding, results]
---
> Hub: [[COWORK]]


## Summary

- Scope: Block A1.4i Pyramiding K-Cap Sweep in the Dali research lineage area.
- Existing takeaway/status: Calibration passed: K=1 matched block_a14h_maker_non_overlap_results.csv; K=inf matched block_a14c_maker_at_mid_results.csv. Across 960 market/K/grid cells, 3 had positive mean PnL and no cell cleared the robustness bar at any K. The best cell was K=inf on `a0b:2364426` / `btc-updown-4h-1779912000` with W=10s, H=30s, `exit_symmetric_maker`, 588 fills, mean 554.9 bps, and CI [-329.5 bps, 1054.1 bps].
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
> Table terms: [[polymarket_table_dictionary]]

# Block A1.4i Pyramiding K-Cap Sweep

## Headline

Calibration passed: K=1 matched block_a14h_maker_non_overlap_results.csv; K=inf matched block_a14c_maker_at_mid_results.csv. Across 960 market/K/grid cells, 3 had positive mean PnL and no cell cleared the robustness bar at any K. The best cell was K=inf on `a0b:2364426` / `btc-updown-4h-1779912000` with W=10s, H=30s, `exit_symmetric_maker`, 588 fills, mean 554.9 bps, and CI [-329.5 bps, 1054.1 bps].

## Method

- Universe: same as A14c/A14h, A1 markets labeled `primary_read` or `thin_wide_CI`.
- Signal: per-market top absolute decile of current-level TOB imbalance.
- Entry/fill/exit/rebate: identical to A14c, using A14h's explicit exit timestamps for capacity accounting.
- K rule: at most K concurrent open maker positions per market. A candidate fill is accepted only if fewer than K positions are open at both signal time and fill time.
- K sweep: `1`, `2`, `3`, `5`, `inf`. K=1 reproduces A14h; K=inf reproduces A14c.
- Bootstrap: 200 resamples over contiguous 300s blocks on filled PnL.

## K Summary

| K | positive cells | robust+ | best market | slug | best grid | fills | fill rate | mean | CI | max conc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0/192 | 0 | a0:1469737 | will-mojtaba-khamenei-be-head-of-s | W=1, H=5, symmetric_maker | 9 | 0.1% | -15.3 bps | [-28.4 bps, -7.2 bps] | 1 |
| 2 | 0/192 | 0 | a0:1469737 | will-mojtaba-khamenei-be-head-of-s | W=5, H=5, symmetric_maker | 17 | 0.2% | -15.4 bps | [-29.3 bps, -7.6 bps] | 2 |
| 3 | 0/192 | 0 | a0:1469737 | will-mojtaba-khamenei-be-head-of-s | W=5, H=5, symmetric_maker | 25 | 0.3% | -15.4 bps | [-33.5 bps, -7.4 bps] | 3 |
| 5 | 0/192 | 0 | a0:1469737 | will-mojtaba-khamenei-be-head-of-s | W=5, H=5, symmetric_maker | 39 | 0.5% | -15.6 bps | [-32.2 bps, -7.3 bps] | 5 |
| inf | 3/192 | 0 | a0b:2364426 | btc-updown-4h-1779912000 | W=10, H=30, symmetric_maker | 588 | 9.0% | 554.9 bps | [-329.5 bps, 1054.1 bps] | 153 |

## BTC-4h A14c Winner Curve

This is the original A14c positive cell, `a0b:2364426`, W=10s, H=30s, `exit_symmetric_maker`.

| K | fills | fill rate | mean | CI | win | max conc. |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 13 | 0.2% | -451.3 bps | [-960.3 bps, -9.7 bps] | 30.8% | 1 |
| 2 | 25 | 0.4% | -469.3 bps | [-1001.2 bps, 15.7 bps] | 32.0% | 2 |
| 3 | 36 | 0.5% | -468.4 bps | [-1020.9 bps, 34.8 bps] | 33.3% | 3 |
| 5 | 58 | 0.9% | -367.3 bps | [-982.4 bps, 247.4 bps] | 34.5% | 5 |
| inf | 588 | 9.0% | 554.9 bps | [-329.5 bps, 1054.1 bps] | 59.5% | 153 |

## PnL-vs-K Curve Per Market

Each cell shows the best mean PnL for that market and K, with executed fill count in parentheses.

| market | slug | K=1 | K=2 | K=3 | K=5 | K=inf |
| --- | --- | --- | --- | --- | --- | --- |
| a0:1090496 | nato-x-russia-military-clash-by-de | -135.5 bps (7) | -135.5 bps (14) | -135.9 bps (19) | -136.6 bps (27) | -135.1 bps (88) |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-s | -15.3 bps (9) | -15.4 bps (17) | -15.4 bps (25) | -15.6 bps (39) | -11.2 bps (81) |
| a0:558934 | will-spain-win-the-2026-fifa-world | -182.5 bps (75) | -182.5 bps (150) | -182.6 bps (221) | -181.8 bps (314) | -181.4 bps (368) |
| a0:558936 | will-france-win-the-2026-fifa-worl | -173.3 bps (39) | -170.9 bps (74) | -170.1 bps (109) | -172.1 bps (181) | -173.1 bps (323) |
| a0:631139 | will-google-have-the-best-ai-model | -332.8 bps (2) | -332.8 bps (4) | -332.8 bps (6) | -358.2 bps (9) | -453.9 bps (17) |
| a0:631140 | will-anthropic-have-the-best-ai-mo | -107.6 bps (6) | -108.0 bps (11) | -108.1 bps (16) | -116.7 bps (22) | -171.7 bps (23) |
| a0:665325 | us-iran-nuclear-deal-before-2027 | -208.2 bps (42) | -207.1 bps (82) | -213.0 bps (119) | -227.7 bps (149) | -291.5 bps (498) |
| a0b:1971905 | strait-of-hormuz-traffic-returns-t | -80.5 bps (15) | -80.5 bps (30) | -80.5 bps (45) | -78.1 bps (70) | -80.1 bps (785) |
| a0b:2176262 | strait-of-hormuz-traffic-returns-t | -53.1 bps (4) | -36.9 bps (7) | -33.8 bps (9) | -22.2 bps (12) | -19.8 bps (37) |
| a0b:2327929 | nba-okc-sas-2026-05-28 | -217.8 bps (17) | -216.8 bps (33) | -216.4 bps (49) | -217.8 bps (85) | -216.9 bps (414) |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | -284.4 bps (51) | -286.7 bps (99) | -292.7 bps (115) | -290.9 bps (182) | -229.8 bps (629) |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | -430.0 bps (29) | -437.3 bps (55) | -418.9 bps (78) | -409.7 bps (119) | -88.2 bps (176) |
| a0b:2364426 | btc-updown-4h-1779912000 | -133.6 bps (14) | -138.5 bps (27) | -125.0 bps (39) | -21.0 bps (63) | 554.9 bps (588) |
| a0b:2366225 | btc-updown-4h-1779926400 | -876.0 bps (46) | -930.7 bps (84) | -900.6 bps (92) | -726.8 bps (138) | -420.1 bps (344) |
| a0b:2367777 | btc-updown-4h-1779940800 | -457.2 bps (111) | -458.7 bps (219) | -456.1 bps (324) | -444.0 bps (521) | -383.1 bps (2101) |
| a0b:566136 | will-psg-win-the-202526-champions- | -202.7 bps (12) | -203.6 bps (23) | -202.6 bps (33) | -200.2 bps (50) | -212.5 bps (108) |

## Cross-Market Verdict

Intermediate K values explain the overlap artifact rather than rescuing it. If a cell only becomes attractive at high K or infinite K, the edge is capacity-dependent repeated exposure to the same episode, not a clean one-position maker edge. The robustness bar is mean > 0, CI lower > 0, and at least 5 fills.

## Recommendation

Recommended next action for Justin: Dali microstructure is closed on A0/A0b unless A2 is explicitly reframed as episode discovery rather than deployable maker/taker PnL.
