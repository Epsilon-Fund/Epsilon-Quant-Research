---
title: Block K5b Top-Maker Dominance Decomposition
created: 2026-06-05
status: active
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
  - strat_market_making
tags:
  - market-making
  - block-k
  - top-maker
  - capacity
  - research
---

# Block K5b Top-Maker Dominance Decomposition

> **Strat:** [[strat_market_making]] (Market-Making). Sibling: [[strat_options_delta]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Summary

K5b decomposes top-maker dominance into capital, scale, structure, and fill-share proxies. It does not prove a Rust-speed moat because quote/cancel data and true queue position are absent. The practical takeaway is to prioritize structured quoting, inventory control, market selection, and live telemetry before optimizing for low-level speed.

## Verdict

The moat is **capital/scale plus structure**, not a proven Rust-speed moat. The historical files show fills, not quotes/cancels, so true requote latency and cancel frequency are unobservable here. The fill-based queue proxy says the top 3 have higher market fill share than ranks 4-23, but the stronger story is that they operate a larger, more two-sided, more professionally structured book.

Can a disciplined new maker earn a positive net share? **not proven after excluding the top3**. Below the global top 3, the K5 crypto maker universe earns **29.4 bps** with CI **[-173 bps, 260 bps]**. That means the market is not mathematically closed, but the best dollars are crowded: the top3 tier earns **$49,446** versus **$18,905** for the next20 tier.

The practical implication for Strategy A: do not start by rewriting Midas in Rust. Start with the right structure: two-sided quoting, inventory/carry control, market selection, and capacity-aware sizing. Rust only matters after we can observe live quote/cancel latency and prove we are losing queue to speed rather than to capital, risk limits, or quoting policy.

## What Is Observable

- Observable: maker fills, fill timing, fill size, realized PnL, settlement carry, two-sidedness, market coverage, and fill-share proxies.
- Not observable in `data/trades/*.parquet`: quote placements, cancels, queue position at the touch, and time between a book move and a wallet's requote.
- Therefore all speed conclusions below are **fill-based proxies**, not direct latency measurements.
- Top3/next20 are ranked by realized crypto-4h net PnL after a material-maker screen: at least **1,000** maker fills and **50** markets. This avoids one-off high-PnL wallets being mislabeled as dominant market makers.

## Top Wallets

| rank | tier | wallet | markets | maker fills | net PnL | bps | carry | two-sided | spike-zone | max carry markets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | top3 | 0x7881a918... | 287 | 6265 | $28,877 | 557 bps | 55.9% | 96.6% | 0.6% | 3 |
| 2 | top3 | 0x6fdc6877... | 824 | 83715 | $12,985 | 194 bps | 77.3% | 99.7% | 1.5% | 6 |
| 3 | top3 | 0xfb1c3c1a... | 809 | 18337 | $7,583 | 267 bps | 98.5% | 97.0% | 1.8% | 3 |
| 4 | next20 | 0xa9650fe4... | 598 | 1722 | $7,019 | 1,585 bps | 100.0% | 66.5% | 0.0% | 21 |
| 5 | next20 | 0xba016b05... | 666 | 28861 | $2,814 | 199 bps | 74.4% | 99.7% | 1.1% | 6 |
| 6 | next20 | 0x686c0816... | 2042 | 22102 | $2,645 | 2,034 bps | 99.9% | 12.4% | 0.0% | 4 |
| 7 | next20 | 0x8301ba6b... | 1803 | 3808 | $1,724 | 6,931 bps | 98.7% | 27.0% | 0.0% | 3 |
| 8 | next20 | 0x260b8cd6... | 610 | 2385 | $1,181 | 4,712 bps | 98.2% | 26.1% | 0.0% | 3 |
| 9 | next20 | 0x3329cfc2... | 923 | 4818 | $467 | 2,012 bps | 100.0% | 0.3% | 0.0% | 3 |
| 10 | next20 | 0x46b35366... | 1415 | 3211 | $425 | 6,060 bps | 100.0% | 3.1% | 0.0% | 3 |
| 11 | next20 | 0x73edfe80... | 52 | 20799 | $403 | 91.6 bps | 99.6% | 100.0% | 0.6% | 1 |
| 12 | next20 | 0x229c0a7d... | 379 | 3210 | $307 | 1,834 bps | 100.0% | 20.2% | 0.0% | 2 |
| 13 | next20 | 0x4acf2696... | 1239 | 8424 | $279 | 551 bps | 99.5% | 23.3% | 3.3% | 6 |
| 14 | next20 | 0x8162fa34... | 2226 | 3941 | $272 | 3,430 bps | 99.8% | 5.8% | 0.0% | 4 |
| 15 | next20 | 0x50b97739... | 2436 | 3089 | $252 | 3,893 bps | 98.8% | 1.4% | 0.0% | 3 |
| 16 | next20 | 0x9f16feb6... | 271 | 1121 | $206 | 3,697 bps | 23.4% | 6.3% | 6.8% | 3 |
| 17 | next20 | 0x7552bbb9... | 815 | 1762 | $185 | 926 bps | 96.9% | 20.5% | 2.7% | 3 |
| 18 | next20 | 0xafcdbf1f... | 1554 | 3984 | $173 | 131 bps | 58.8% | 44.4% | 0.0% | 3 |
| 19 | next20 | 0xf70c7732... | 337 | 1003 | $151 | 392 bps | 71.0% | 54.2% | 0.0% | 10 |
| 20 | next20 | 0x8f44b737... | 60 | 1083 | $129 | 460 bps | 19.2% | 73.0% | 0.0% | 3 |
| 21 | next20 | 0xcd85e175... | 54 | 2121 | $117 | 125 bps | 13.3% | 45.9% | 1.3% | 1 |
| 22 | next20 | 0x88be9e29... | 229 | 1876 | $83 | 834 bps | 48.5% | 0.0% | 0.0% | 3 |
| 23 | next20 | 0x195d28a9... | 1008 | 1348 | $74 | 3,675 bps | 100.0% | 2.1% | 0.0% | 3 |

## Tier Comparison

| tier | wallets | markets | gross volume | net PnL | bps | CI | carry | two-sided | fill share proxy | avg fill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top3 | 3 | 1739 | $1,473,433 | $49,446 | 336 bps | [135 bps, 516 bps] | 74.4% | 98.2% | 20.6% | $7.73 |
| next20 | 20 | 3223 | $291,521 | $18,905 | 648 bps | [312 bps, 1,065 bps] | 91.2% | 77.6% | 5.4% | $1.95 |
| below_top3 | 829 | 3278 | $2,690,648 | $7,898 | 29.4 bps | [-173 bps, 260 bps] | 81.2% | 45.9% | 19.2% | $5.69 |
| all | 832 | 3278 | $4,164,081 | $57,344 | 138 bps | [-4.9 bps, 293 bps] | 79.6% | 63.4% | 19.7% | $6.24 |

## Top3 Minus Next20

| metric | factor | top3 | next20 | diff | diff CI |
| --- | --- | --- | --- | --- | --- |
| net_pnl_bps | profit_rate | 336 bps | 648 bps | -313 bps | [-597 bps, -18.1 bps] |
| resolution_carry_token_share | risk | 74.4% | 91.2% | -16.8% | [-19.2%, -14.6%] |
| two_sided_usd_share | structure | 98.2% | 77.6% | 20.6% | [18.2%, 23.7%] |
| early15_usd_share | speed_queue_proxy | 3.9% | 24.0% | -20.1% | [-23.6%, -17.3%] |
| late180_240_usd_share | risk_timing | 30.5% | 26.1% | 4.4% | [2.2%, 6.6%] |
| spike_zone_usd_share | risk_timing | 1.4% | 0.7% | 0.7% | [0.4%, 1.1%] |
| market_fill_share | queue_proxy | 20.6% | 5.4% | 15.2% | [12.1%, 17.2%] |
| avg_fill_usd | capital_scale | 7.73 | 1.95 | 5.78 | [4.09, 12.74] |

## Profit Edge Attribution

- Observed dollar gap, top3 minus next20: **$30,541**.
- Rate edge on top3 volume: **$-46,103**.
- Scale edge at next20 rate: **$76,645**.
- Carry-share difference: **-16.8%**.
- Average maker fill-size difference: **$5.78**.
- Market fill-share proxy difference: **15.2%**.

Read this cautiously: speed/queue can only be proxied by realized fill share and early-window fill share. Those proxies do not isolate technology from better quoting, more capital, or simply being willing to sit in more markets.

## Capacity Below Top3

The buckets below group markets by the global top3's share of raw maker USD, then show PnL for everyone else in the K5 crypto maker universe.

| top3 saturation bucket | markets | below-top3 PnL | below-top3 bps | CI | below-top3 maker USD | fills |
| --- | --- | --- | --- | --- | --- | --- |
| top3_0_10pct | 1032 | $6,009 | 47.8 bps | [-136 bps, 259 bps] | $769,927 | 132081 |
| no_top3 | 1545 | $1,980 | 17.3 bps | [-374 bps, 423 bps] | $742,919 | 105930 |
| top3_60_100pct | 45 | $-580 | -719 bps | [-1,734 bps, -24.3 bps] | $2,938 | 1693 |
| top3_30_60pct | 170 | $-2,805 | -423 bps | [-1,420 bps, 247 bps] | $35,089 | 21609 |
| top3_10_30pct | 486 | $3,294 | 155 bps | [-298 bps, 627 bps] | $113,984 | 31084 |

## Time-Window Fill Share

| phase | tier | maker USD | share of phase maker USD |
| --- | --- | --- | --- |
| 00_15m | next20 | $63,875 | 1.8% |
| 00_15m | non_k5_or_untagged | $3,200,304 | 92.1% |
| 00_15m | rest | $176,947 | 5.1% |
| 00_15m | top3 | $35,107 | 1.0% |
| 15_60m | next20 | $39,829 | 1.5% |
| 15_60m | non_k5_or_untagged | $2,277,947 | 87.3% |
| 15_60m | rest | $161,696 | 6.2% |
| 15_60m | top3 | $130,086 | 5.0% |
| 180_240m | next20 | $88,121 | 1.0% |
| 180_240m | non_k5_or_untagged | $7,735,627 | 90.2% |
| 180_240m | rest | $476,947 | 5.6% |
| 180_240m | top3 | $271,819 | 3.2% |
| 60_180m | next20 | $129,995 | 1.4% |
| 60_180m | non_k5_or_untagged | $8,526,927 | 88.9% |
| 60_180m | rest | $475,641 | 5.0% |
| 60_180m | top3 | $464,382 | 4.8% |
| after_close | next20 | $1,185 | 0.1% |
| after_close | non_k5_or_untagged | $587,425 | 71.0% |
| after_close | rest | $238,834 | 28.9% |
| after_close | top3 | $18 | 0.0% |

## Conclusion

K5b does not support "we need Rust because the edge is a pure latency race." It supports a more mundane but harder conclusion: the best makers win through scale, two-sided structure, queue/fill share, and willingness to carry settlement risk while avoiding the worst spike-zone flow. A Python/Midas maker can be good enough for research and initial deployment if it implements those structural choices. The real live question is whether we can get enough priority and fill share without joining the top-wallet arms race.

Caveat: this inherits K5's closed-position survivorship issue. Open/unresolved inventory is excluded, so realized profitability may overstate live deployable economics if losing risk remains open.
