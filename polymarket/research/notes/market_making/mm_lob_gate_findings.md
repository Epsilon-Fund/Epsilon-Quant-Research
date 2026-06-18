---
title: MM LOB Gate Findings
created: 2026-06-05
status: closed
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
  - strat_market_making
tags:
  - market-making
  - lob
  - queue
  - paper-trading
  - research
---

# MM LOB Gate Findings

> Hub: [[strat_market_making]] · [[mm_deployable_cells_findings]]

## Summary

This note checks whether the captured LOB panel can support a historical aggression surface for MM deployable cells. The target coverage gate fails, especially for `other:misc_other`, and the available queue proxy is only an upper bound. The conclusion is to route aggression measurement to live paper quoting rather than build a historical LOB backtest from this panel.

## Headline

**DO NOT BUILD a historical LOB aggression surface from this panel. Route aggression measurement to live paper quoting.**

`other:misc_other` has 3 LOB-covered market(s), 6 trade print(s), and represents 0.01% of its 2026 raw flow. That fails the main target coverage gate.

Exact same-market K5 overlap exists for $142,964 of maker notional, but it is still not queue-position calibration.

## Coverage Gate

| sub-cell | coverage ok? | markets | assets | LOB trades | active hrs | 2026 flow represented | LOB trade notional | 2026 cell flow/day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| other:misc_other | NO | 3 | 6 | 6 | 25 | 0.01% | $243 | $32,763,605 |
| other:crypto_misc | NO | 0 | 0 | 0 | 0 | 0.00% | $0 | $2,580,978 |
| sports:nba:outright | NO | 0 | 0 | 0 | 0 | 0.00% | $0 | $468,542 |
| sports:soccer:market_other | NO | 0 | 0 | 0 | 0 | 0.00% | $0 | $50,775 |
| sports:ufc:outright | NO | 0 | 0 | 0 | 0 | 0.00% | $0 | $4,469 |
| culture:oscars | NO | 0 | 0 | 0 | 0 | 0.00% | $0 | $15,000 |
| crypto_4h:btc:00_06utc | NO | 3 | 6 | 1397 | 32 | 0.00% | $20,554 | $16,224 |
| crypto_4h:btc:06_12utc | NO | 6 | 12 | 3472 | 37 | 0.00% | $40,635 | $32,509 |

## Full-Priority Proxy Calibration

These rows are **upper bounds**, not deployable fills. The proxy assumes our quote has full queue priority and
fills on any trade-through at our modeled price.

| sub-cell | policy | proxy fills | proxy notional | fills/hr | spread-to-mid | 60s markout | 60s adverse | proxy/K5 overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_4h:btc:00_06utc | join | 788 | $8,549 | 24.62 | 473 bps | -88.2 bps | 88.2 bps | n/a |
| crypto_4h:btc:00_06utc | improve_1 | 872 | $10,477 | 27.25 | 236 bps | -113 bps | 113 bps | n/a |
| crypto_4h:btc:00_06utc | improve_2 | 917 | $11,415 | 28.66 | 88.4 bps | -106 bps | 106 bps | n/a |
| crypto_4h:btc:06_12utc | join | 1999 | $18,469 | 54.03 | 374 bps | -218 bps | 218 bps | n/a |
| crypto_4h:btc:06_12utc | improve_1 | 2124 | $21,766 | 57.41 | 210 bps | -224 bps | 224 bps | n/a |
| crypto_4h:btc:06_12utc | improve_2 | 2197 | $23,028 | 59.38 | 71.0 bps | -243 bps | 243 bps | n/a |

## Queue Limitation

The captured panel has best bid/ask states and trade prints, but not our historical queue rank, order age,
cancel/replace priority, or maker identity at the touch. Therefore a join-best historical fill rate cannot
distinguish "first in queue" from "last in queue." That is first-order for deployability.

## Decision

Do **not** build a standalone historical MM quote-aggression backtest from this panel. The only honest use of
this LOB data is as an upper-bound diagnostic for cells it actually covers, mainly crypto/sports/active markets.
The main MM cell, `other:misc_other`, is not materially covered. Quote aggression should be measured in live
paper trading, with logs for quote price, quote size, best bid/ask, post/cancel time, queue proxy if available,
fills, missed trade-throughs, markouts, settlement, and inventory cluster.

## Output

CSV: `data/analysis/csv_outputs/market_making/mm_lob_coverage_proxy_calibration.csv`
