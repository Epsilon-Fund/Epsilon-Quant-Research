---
title: Block K5 Real Maker PnL Reality Check
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
  - real-maker-pnl
  - wallets
  - research
---

# Block K5 Real Maker PnL Reality Check

> **Strat:** [[strat_market_making]] (Market-Making). Sibling: [[strat_options_delta]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Summary

K5 compares real maker-wallet PnL and structure against the earlier single-leg maker simulations. The note finds real maker behavior is not captured by the K simulations alone, with top makers showing scale and two-sided structure differences. It becomes a durable reference for deployability and incumbent-capacity questions rather than a standalone trade green-light.

## Verdict

Real maker wallets are not obviously described by the single-leg K simulations. In the top-analysis-wallet closed-position sample, pooled realized net PnL is **$189,844,130** on **$13,106,216,230** gross closed volume, or **145 bps** with CI **[85.2 bps, 210 bps]**.

For **crypto-4h**, real maker-heavy wallets are **positive with a CI above zero**: **$67,894**, **171 bps**, CI **[33.8 bps, 327 bps]**, across **256** wallets and **3268** markets.

This does not vindicate the single-leg maker simulations, because those strategies intentionally enter one leg and then pay an exit. Real makers are running a continuous book, often two-sided, and often carrying residual exposure to resolution. The direct correction versus K2v3 is shown in `data/analysis/csv_outputs/market_making/k5_real_maker_pnl.csv`: crypto-4h real-maker bps minus the K2v3 taker-exit, maker-exit, and hold-resolution baselines.

The fee-cushion story mostly survives the model-free check: fee-enabled categories have positive point estimates, while fee-free geopolitics is negative on point estimate and does not clear zero. The crypto-4h result is real in aggregate, but capacity is tight: the top 3 wallets capture about **95%** of positive crypto-4h maker profit by market.

## Method

- Maker wallets were seeded from `data/traders.parquet`, then recomputed from raw `data/trades/*.parquet` with `EXCHANGE_INTERNAL_LEG_V1/_V2` removed on both maker and taker slots.
- A wallet qualifies as a maker when corrected maker share is at least **70%** with at least **1,000** passive fills.
- Realized PnL comes from `data/closed_positions.parquet`, joined to market metadata and adjusted for K1-style maker rebates and taker fees.
- CI bars bootstrap market blocks, so one huge market does not get treated as hundreds of independent observations.
- Caveat: `closed_positions.parquet` is closed-only. Open inventory, unresolved positions, and any missing market metadata are outside this result.

Qualified makers found: **9,121**. Analysis cohort: **588** wallets, chosen from top overall makers, top crypto-4h makers, and top per-category makers.

## Category Rank

| category | wallets | markets | maker fills | net PnL | bps | CI | settlement carry token share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| politics_negrisk | 300 | 6044 | 5153958 | $78,822,302 | 443 bps | [116 bps, 883 bps] | 35.2% |
| tech | 407 | 16189 | 1226152 | $3,907,120 | 184 bps | [-142 bps, 539 bps] | 62.4% |
| crypto_4h | 256 | 3268 | 360997 | $67,894 | 171 bps | [33.8 bps, 327 bps] | 78.8% |
| other | 562 | 334474 | 31862365 | $74,273,610 | 147 bps | [81.3 bps, 207 bps] | 62.7% |
| daily_crypto | 457 | 160390 | 106740139 | $11,670,782 | 83.2 bps | [71.0 bps, 94.0 bps] | 68.5% |
| sports | 397 | 64205 | 10062722 | $26,224,728 | 73.6 bps | [-7.0 bps, 155 bps] | 73.5% |
| geopolitics | 430 | 11726 | 3172407 | $-5,122,305 | -46.8 bps | [-163 bps, 69.3 bps] | 52.6% |

## Crypto-4h Makers

| wallet | markets | maker fills | net PnL | bps | $/maker fill | settlement carry | late near-50c share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0x6fdc6877... | 824 | 83715 | $12,985 | 194 bps | $0.1551 | 77.3% | 1.5% |
| 0xba016b05... | 666 | 28861 | $2,814 | 199 bps | $0.0975 | 74.4% | 1.1% |
| 0x686c0816... | 2042 | 22102 | $2,645 | 2,034 bps | $0.1197 | 99.9% | 0.0% |
| 0x73edfe80... | 52 | 20799 | $403 | 91.6 bps | $0.0194 | 99.6% | 0.6% |
| 0xfb1c3c1a... | 809 | 18337 | $7,583 | 267 bps | $0.4135 | 98.5% | 1.8% |
| 0x5eb2c2e2... | 31 | 18192 | $455 | 125 bps | $0.0250 | 99.7% | 1.3% |
| 0x4ffe49ba... | 1019 | 13120 | $-2,502 | -360 bps | $-0.1907 | 73.5% | 0.9% |
| 0x4acf2696... | 1239 | 8424 | $279 | 551 bps | $0.0331 | 99.5% | 3.3% |

## True Markouts

Positive markout means the maker fill improved after entry. Negative markout is adverse selection.

| horizon | covered fills | coverage | mean markout | CI | adverse cost | positive rate | avg print lag sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_4h_5s | 263936 | 90.3% | 494 bps | [204 bps, 955 bps] | -494 bps | 36.8% | 37.9 |
| crypto_4h_30s | 257211 | 88.0% | 684 bps | [239 bps, 1,597 bps] | -684 bps | 41.2% | 43.6 |
| crypto_4h_60s | 252557 | 86.4% | 616 bps | [212 bps, 1,286 bps] | -616 bps | 42.7% | 47.5 |
| crypto_4h_resolution | 292430 | 100.0% | 249 bps | [-513 bps, 1,318 bps] | -249 bps | 44.2% | n/a |

K2v3 assumed adverse-selection costs around 145-325 bps. The realized markout table is the empirical check. Here, **negative adverse cost means maker-favorable reversion**, because adverse cost is reported as `-markout`.

## Behavior

The profitable-maker behavior is different from the tested single-leg scripts:

- **Two-sidedness:** crypto-4h maker-volume share in markets where the wallet made both outcomes is **64.2%**.
- **Carry/settle:** crypto-4h residual token share carried to settlement is **78.8%**; the complement is the rough intraday round-trip share.
- **Spike zone:** crypto-4h maker-dollar share in the late-window near-50c zone is **0.8%**.
- **Fees:** crypto-4h net fee/rebate adjustment is **$2,414**, so rebates help but are not the whole story.

## Capacity

| category | markets | median maker wallets/market | top1 share of positive profit | top3 share of positive profit |
| --- | --- | --- | --- | --- |
| crypto_4h | 3268 | 11.0 | 78.8% | 95.1% |
| daily_crypto | 160390 | 11.0 | 58.3% | 83.7% |
| geopolitics | 11726 | 5.0 | 54.0% | 81.5% |
| other | 334474 | 3.0 | 62.7% | 88.4% |
| politics_negrisk | 6044 | 6.0 | 37.5% | 68.7% |
| sports | 64205 | 3.0 | 60.1% | 87.2% |
| tech | 16189 | 4.0 | 55.0% | 78.7% |

High top-1/top-3 concentration means the PnL exists but may be captured by a few established wallets. Low concentration means the category is less obviously locked up.

## My Read

K5 is the right reality check: if real makers earn money while the single-leg sim loses thousands of bps, the sim is measuring an intentionally bad lifecycle, not the economics of market making. The result should be read with two guardrails. First, closed-position survivorship can overstate realized profitability if losing inventory remains open or unresolved. Second, maker-share is not profit: the only result that matters is closed, net-of-cost PnL with CI.

For crypto-4h, the practical question is whether the real-maker edge is large enough and unconcentrated enough to copy. Here the CI clears zero, but capacity looks captured. My conclusion is: the single-leg maker thesis stays closed; a real-maker version is only salvageable as a continuous two-sided/hold-to-resolution design with explicit capacity and queue-priority assumptions.
