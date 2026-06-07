---
title: MM Deployable Cells Findings
created: 2026-06-05
status: candidate
owner: justin
project: polymarket
para: project
hubs:
  - COWORK
  - strat_market_making
tags:
  - market-making
  - deployable-cells
  - paper-trading
  - k5-stress
  - research
---

# MM Deployable Cells Findings

> Hub: [[strat_market_making]] · [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]

## Summary

This note converts the K5-STRESS structured-maker result into paper-trading sub-cells. Eight sub-cells qualify for MM paper-trading, with 2026 run-rate headroom and base EV estimated from historical structured non-top3 behavior. It is the MM-track replacement for a draft K9 prompt and deliberately avoids OD valuation or directional skew.

## Plain-English Headline

**8 sub-cells qualify for MM paper-trading.** Historical median-based open-headroom EV is **$20,470** on **$6,841,265** capturable flow, but the deployable run-rate should use 2026 flow: **$78/active day** on **$26,166/active day** estimated headroom (~**$2,331** per 30 active days). Mean-based historical optimistic EV is **$111,717**.

This is the MM-track replacement for the draft "K9" prompt. It does **not** introduce OD valuation, Binance
fair value, vol, or directional skew. It only asks where the K5-STRESS real-maker playbook can be paper-tested:
two-sided passive making, carry-to-resolution, spike-zone avoidance, and no incumbent top-3 dependence.

Important window note: this screen does **not** use the A1/A0 order-book capture panel (roughly 48h in the
maker-sim work). It uses the K5-STRESS historical raw fills plus closed/open position reconstruction. Local raw
fills currently span 2022-11-21 to 2026-05-26. Because flow is much larger now than in the early history, the
paper-trade run-rate uses **2026 raw flow/day** as the baseline. Historical `base EV` is kept in the CSV for
auditability, but table ranking is by `base_ev_2026_per_active_day_usd`.

## Design

Input: `data/analysis/csv_outputs/market_making/k5_stress.csv` plus the cached full marked wallet-market table
`data/analysis/k5_stress_wallet_market_full.parquet`.

The unit of observation in the output CSV is one deployable MM sub-cell. A sub-cell is a category split such as
`sports:nba:moneyline`, `culture:oscars`, or `crypto_4h:btc:12_18utc`.

For every sub-cell I recomputed the K5-STRESS structured non-top3 result:

- structured playbook: two-sided USD share >= 60%, carry-token share >= 50%, spike-zone share <= 2%
- non-incumbent: exclude each market's global top-3 maker wallets
- paper gate: CI lower > 0, median structured wallet > 0, capacity open, positive in >= 2/3 active months, net without rebate > 0, and sub-cell edge above the parent average

## How To Read The Main Table

- `net bps`: aggregate structured non-top3 PnL per gross dollar in the cell.
- `CI`: market-block bootstrap confidence interval.
- `median`: the base-case EV rate; this avoids letting a few right-tail wallets define expected returns.
- `capacity`: open/constrained/captured based on observed top-3 maker share, active structured makers, and non-top3 flow.
- `2026 flow/day`: raw sub-cell fill notional per active 2026 day.
- `2026 headroom/day`: 2026 flow/day times the historical capturable-headroom share.
- `2026 base EV/day`: 2026 headroom/day times the median structured maker bps.
- `growth`: 2026 raw flow/day divided by the full-history raw flow/day.

| rank | sub-cell | paper? | net bps | CI | median | capacity | 2026 flow/day | 2026 headroom/day | 2026 base EV/day | growth | flow conf | +months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | other:misc_other | YES | 150 bps | [85.6 bps, 215 bps] | 29.0 bps | open | $32,763,605 | $24,298 | $70 | 5.1x | medium | 76.0% |
| 2 | other:crypto_misc | YES | 206 bps | [90.5 bps, 355 bps] | 23.0 bps | open | $2,580,978 | $1,503 | $3 | 2.3x | medium | 84.8% |
| 3 | sports:nba:outright | YES | 420 bps | [257 bps, 590 bps] | 90.3 bps | open | $468,542 | $311 | $3 | 1.7x | high | 68.8% |
| 4 | sports:soccer:market_other | YES | 579 bps | [76.8 bps, 1,283 bps] | 160 bps | open | $50,775 | $29 | $0 | 2.0x | medium | 75.0% |
| 5 | sports:ufc:outright | YES | 1,161 bps | [437 bps, 1,806 bps] | 743 bps | open | $4,469 | $4 | $0 | 1.0x | medium | 100.0% |
| 6 | culture:oscars | YES | 1,135 bps | [497 bps, 1,810 bps] | 81.4 bps | open | $15,000 | $13 | $0 | 0.9x | medium | 84.2% |
| 7 | crypto_4h:btc:00_06utc | YES | 548 bps | [265 bps, 880 bps] | 275 bps | open | $16,224 | $3 | $0 | 1.5x | medium | 87.5% |
| 8 | crypto_4h:btc:06_12utc | YES | 313 bps | [7.6 bps, 627 bps] | 143 bps | open | $32,509 | $5 | $0 | 1.4x | medium | 75.0% |
| 9 | sports:ufc:player_prop | NO | 22,090 bps | [n/a, n/a] | 22,090 bps | captured | n/a | n/a | n/a | n/a | medium | 100.0% |
| 10 | sports:soccer:player_prop | NO | 4,881 bps | [-1,702 bps, 5,454 bps] | 1,121 bps | constrained | $1,329 | $0 | $0 | 0.6x | medium | 66.7% |
| 11 | sports:mlb:outright | NO | 2,300 bps | [99.0 bps, 4,772 bps] | 463 bps | constrained | $2,280 | $0 | $0 | 0.6x | medium | 50.0% |
| 12 | sports:nfl:spread | NO | 1,506 bps | [-879 bps, 4,204 bps] | 120 bps | constrained | $591,394 | $0 | $0 | 1.0x | medium | 85.7% |
| 13 | sports:nfl:moneyline | NO | 1,424 bps | [-217 bps, 1,995 bps] | 119 bps | constrained | $15,702 | $0 | $0 | 0.1x | medium | 100.0% |
| 14 | sports:nfl:total | NO | 1,188 bps | [-1,208 bps, 3,531 bps] | 422 bps | constrained | $178,250 | $0 | $0 | 1.3x | medium | 80.0% |
| 15 | sports:ufc:moneyline | NO | 1,091 bps | [-276 bps, 2,606 bps] | 1,008 bps | open | $7,755 | $13 | $1 | 0.1x | medium | 78.9% |

## Paper-Qualified Cells

| rank | sub-cell | median bps | 2026 flow/day | 2026 headroom/day | 2026 base EV/day | 30d base EV | growth | top3 share | structured wallets | cluster tag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | other:misc_other | 29.0 bps | $32,763,605 | $24,298 | $70 | $2,113 | 5.1x | 46.3% | 645 | misc_other:2026-03 |
| 2 | other:crypto_misc | 23.0 bps | $2,580,978 | $1,503 | $3 | $104 | 2.3x | 40.7% | 216 | crypto_misc:2025-06 |
| 3 | sports:nba:outright | 90.3 bps | $468,542 | $311 | $3 | $84 | 1.7x | 27.5% | 89 | nba:will-the-indiana-pacers-win-the-2025-nba-finals |
| 4 | sports:soccer:market_other | 160 bps | $50,775 | $29 | $0 | $14 | 2.0x | 45.5% | 34 | soccer:will-tottenham-be-relegated-from-the-english-premier-league-after-the-202 |
| 5 | sports:ufc:outright | 743 bps | $4,469 | $4 | $0 | $8 | 1.0x | 47.8% | 12 | ufc:will-sean-strickland-be-the-ufc-middleweight-champion-on-december-31-2026 |
| 6 | culture:oscars | 81.4 bps | $15,000 | $13 | $0 | $3 | 0.9x | 40.5% | 135 | oscars:2026-04 |
| 7 | crypto_4h:btc:00_06utc | 275 bps | $16,224 | $3 | $0 | $3 | 1.5x | 51.4% | 75 | btc:2025-12-24 |
| 8 | crypto_4h:btc:06_12utc | 143 bps | $32,509 | $5 | $0 | $2 | 1.4x | 56.4% | 104 | btc:2026-03-06 |

## Interpretation

The median-based EV is intentionally much smaller than the category-level K5-STRESS dollar totals. That is the
point: entering the market changes capacity, and the historical non-top3 aggregate is an upper bound, not our
expected fill share. The practical sizing number is the **2026 base EV/day** column, not the older historical
total.

The cells that survive are the ones where the edge is not merely "the parent category was good." They have
positive market-block CI, positive typical structured wallet, enough non-incumbent flow, and month stability.
Cells with high mean bps but captured capacity are not paper targets.

## Guardrails

- This remains historical research, not a live bot statement.
- Capacity uses realized maker-flow proxies because historical order/cancel queues are not in the owned fill layer.
- The flow-confidence label is diagnostic, not a causal proof of retail uninformed flow.
- Any paper test must log live quotes, cancels, queue position, and missed fills; otherwise we cannot tell whether
failure is capacity, speed, or bad quoting.

## Output

CSV: `data/analysis/csv_outputs/market_making/mm_deployable_cells.csv`
