---
tags: [dali, block-a14h, maker-thesis, results]
title: Block A1.4h Maker Non-Overlap Findings
created: 2026-05-28
status: closed
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
---

# Block A1.4h Maker Non-Overlap Findings

> Hub: [[COWORK]]

> Table terms: [[polymarket_table_dictionary]]


## Summary

A14h retests the A14c maker-at-mid clue with one open position per market and non-overlapping executed fills. The BTC-4h winner collapses from +554.9 bps overlap math to -451.3 bps with a 0.2% fill rate, and no unique market/grid cell remains robust-positive. The conclusion is that the remaining maker clue was an overlap artifact, so queue/latency would be autopsy rather than likely rescue.

## Headline

The A14c BTC-4h winner does **not** survive the non-overlap check. The same `a0b:2364426` / `btc-updown-4h-1779912000` cell that was +554.9 bps in A14c overlap math is -451.3 bps after non-overlap, with CI [-960.3 bps, -9.7 bps], fill rate 0.2%, and 13 executed fills. This puts it in the same bucket as A14f: the positive cell was mostly overlap-position math, not a deployable one-position-at-a-time maker thesis.

## Method

- Universe: same as A14c, A1 markets labeled `primary_read` or `thin_wide_CI` at the 5s horizon.
- Signal: per-market top absolute decile of current-level TOB imbalance, `direction_factor * tob_imbalance`.
- Entry: at signal time, post at current mid on the signal-favorable token side. Long signal posts a bid at mid; short signal posts an ask at mid.
- Fill model: same as A14c. Long-entry bid fills on a SELL print at or below mid; short-entry ask fills on a BUY print at or above mid, within W in {1s, 5s, 10s}.
- Exit model: same two A14c conventions. `exit_forced_taker` closes at opposite touch after H. `exit_symmetric_maker` posts opposite-side at mid after H and forces a taker fallback at t_fill + 2H if no exit fill arrives.
- Non-overlap rule: unfilled signals do not block. Only actual fills create a blocking interval, and candidate fills are greedily selected in fill-time order so no executed fill or signal occurs inside an already-open interval.
- PnL: A14c direction-adjusted mid-maker entry PnL plus entry maker rebate, minus taker fee on forced exits.
- Bootstrap: 200 resamples over contiguous 300s blocks on non-overlap filled PnL.

## Side-by-Side Per-Market Table

| market | slug | A14c best grid | A14c mean | A14c fill | A14h same mean | A14h same fill | same delta | A14h best | A14h best CI | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0:1090496 | nato-x-russia-military-clash-by-dece | W=10, H=5, symmetric_maker | -135.1 bps | 0.9% | -135.5 bps | 0.1% | -0.4 bps | -135.5 bps | [-172.6 bps, -105.4 bps] | fills too rare even in best case |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-sta | W=10, H=5, symmetric_maker | -11.2 bps | 1.1% | -15.3 bps | 0.1% | -4.1 bps | -15.3 bps | [-28.4 bps, -7.2 bps] | fills too rare even in best case |
| a0:558934 | will-spain-win-the-2026-fifa-world-c | W=1, H=30, symmetric_maker | -181.4 bps | 4.6% | -182.5 bps | 0.9% | -1.0 bps | -182.5 bps | [-205.5 bps, -154.9 bps] | fills too rare even in best case |
| a0:558936 | will-france-win-the-2026-fifa-world- | W=10, H=30, symmetric_maker | -173.1 bps | 4.1% | -173.3 bps | 0.5% | -0.1 bps | -173.3 bps | [-200.0 bps, -141.4 bps] | fills too rare even in best case |
| a0:631139 | will-google-have-the-best-ai-model-a | W=1, H=5, symmetric_maker | -453.9 bps | 0.4% | -332.8 bps | 0.0% | 121.1 bps | -332.8 bps | [n/a, n/a] | fills too rare even in best case |
| a0:631140 | will-anthropic-have-the-best-ai-mode | W=1, H=30, symmetric_maker | -171.7 bps | 0.1% | -111.1 bps | 0.0% | 60.6 bps | -107.6 bps | [-185.7 bps, -56.2 bps] | fills too rare even in best case |
| a0:665325 | us-iran-nuclear-deal-before-2027 | W=5, H=5, forced_taker | -291.5 bps | 2.0% | -220.9 bps | 0.1% | 70.6 bps | -208.2 bps | [-282.9 bps, -138.9 bps] | fills too rare even in best case |
| a0b:1971905 | strait-of-hormuz-traffic-returns-to- | W=5, H=30, symmetric_maker | -80.1 bps | 4.2% | -80.5 bps | 0.1% | -0.3 bps | -80.5 bps | [-111.6 bps, -34.5 bps] | fills too rare even in best case |
| a0b:2176262 | strait-of-hormuz-traffic-returns-to- | W=5, H=5, symmetric_maker | -19.8 bps | 0.3% | -53.1 bps | 0.0% | -33.3 bps | -53.1 bps | [n/a, n/a] | fills too rare even in best case |
| a0b:2327929 | nba-okc-sas-2026-05-28 | W=10, H=30, symmetric_maker | -216.9 bps | 4.1% | -217.8 bps | 0.2% | -0.9 bps | -217.8 bps | [-252.7 bps, -161.4 bps] | fills too rare even in best case |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | W=1, H=30, symmetric_maker | -229.8 bps | 1.3% | -310.8 bps | 0.1% | -81.1 bps | -284.4 bps | [-460.8 bps, -169.7 bps] | fills too rare even in best case |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | W=1, H=30, symmetric_maker | -88.2 bps | 0.6% | -539.2 bps | 0.1% | -450.9 bps | -430.0 bps | [-728.4 bps, -137.1 bps] | fills too rare even in best case |
| a0b:2364426 | btc-updown-4h-1779912000 | W=10, H=30, symmetric_maker | 554.9 bps | 9.0% | -451.3 bps | 0.2% | -1006.2 bps | -133.6 bps | [-590.4 bps, 279.1 bps] | overlap artifact like A14f |
| a0b:2366225 | btc-updown-4h-1779926400 | W=1, H=30, symmetric_maker | -420.1 bps | 1.9% | -1294.8 bps | 0.1% | -874.7 bps | -876.0 bps | [-1234.6 bps, -11.9 bps] | fills too rare even in best case |
| a0b:2367777 | btc-updown-4h-1779940800 | W=1, H=30, symmetric_maker | -383.1 bps | 11.6% | -462.6 bps | 0.3% | -79.5 bps | -457.2 bps | [-524.1 bps, -379.6 bps] | fills too rare even in best case |
| a0b:566136 | will-psg-win-the-202526-champions-le | W=5, H=30, symmetric_maker | -212.5 bps | 3.3% | -202.7 bps | 0.4% | 9.8 bps | -202.7 bps | [-233.2 bps, -158.4 bps] | fills too rare even in best case |

## Cross-Market Verdict

Collapsing A14c's duplicated current-level signal-horizon labels, A14c had 3/192 positive unique market/grid cells. A14h keeps 0/192 positive unique cells, with 0 clearing the robustness bar of mean > 0, CI lower > 0, and at least 5 fills. In the raw CSV rows, A14h has 0/768 positive rows and 0 robust-positive rows.

Verdict counts by market: fills too rare even in best case: 15, overlap artifact like A14f: 1.

## Top Non-Overlap Cells

| market | slug | W | H | exit | signals | fills | fill rate | mean | CI | A14c | delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state | 1 | 5 | symmetric_maker | 7,652 | 9 | 0.1% | -15.3 bps | [-28.4 bps, -7.2 bps] | -16.1 bps | 0.8 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state | 1 | 5 | forced_taker | 7,652 | 9 | 0.1% | -15.3 bps | [-28.4 bps, -7.2 bps] | -16.1 bps | 0.8 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state | 1 | 30 | symmetric_maker | 7,652 | 9 | 0.1% | -15.3 bps | [-28.4 bps, -7.9 bps] | -16.1 bps | 0.8 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state | 1 | 30 | forced_taker | 7,652 | 9 | 0.1% | -15.3 bps | [-28.5 bps, -7.9 bps] | -16.1 bps | 0.8 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state | 5 | 5 | symmetric_maker | 7,652 | 9 | 0.1% | -15.3 bps | [-28.5 bps, -7.9 bps] | -13.8 bps | -1.5 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state | 5 | 5 | forced_taker | 7,652 | 9 | 0.1% | -15.3 bps | [-29.2 bps, -7.9 bps] | -13.8 bps | -1.5 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state | 5 | 30 | symmetric_maker | 7,652 | 9 | 0.1% | -15.3 bps | [-28.4 bps, -7.9 bps] | -13.8 bps | -1.5 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state | 5 | 30 | forced_taker | 7,652 | 9 | 0.1% | -15.3 bps | [-27.6 bps, -7.9 bps] | -13.8 bps | -1.5 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state | 10 | 5 | symmetric_maker | 7,652 | 9 | 0.1% | -15.3 bps | [-29.2 bps, -7.2 bps] | -11.2 bps | -4.1 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state | 10 | 5 | forced_taker | 7,652 | 9 | 0.1% | -15.3 bps | [-29.2 bps, -7.9 bps] | -11.2 bps | -4.1 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state | 10 | 30 | symmetric_maker | 7,652 | 9 | 0.1% | -15.3 bps | [-29.2 bps, -7.9 bps] | -11.2 bps | -4.1 bps |
| a0:1469737 | will-mojtaba-khamenei-be-head-of-state | 10 | 30 | forced_taker | 7,652 | 9 | 0.1% | -15.3 bps | [-27.6 bps, -7.9 bps] | -11.2 bps | -4.1 bps |

## Per-Market Verdict

The per-market verdict table above is the diagnostic read. `maker thesis survives` requires a positive non-overlap best cell with CI lower bound above zero. `overlap artifact like A14f` means the market's A14c best overlap cell was positive but the same cell collapsed or lost more than 250 bps under non-overlap. `fills too rare even in best case` means the best non-overlap cell had fewer than 30 fills or sub-1% fill rate. The remaining negative cells are classified as adverse-selection/rebate failure.

## Interpretation

This is still a generous model: full priority at mid, no queue, no latency, no quote-cancel risk, and no partial fills. Since the last positive maker clue disappears before adding those frictions, queue+latency would be an execution autopsy rather than a likely rescue.

Recommended next action for Justin: run A14e queue+latency only if you want to falsify deployability formally; otherwise pivot away from Dali microstructure on A0/A0b because non-overlap already kills the remaining maker clue.
