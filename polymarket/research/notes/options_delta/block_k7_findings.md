# Block K7 Cross-Category Longshot Premium

> **Strat:** [[strat_options_delta]] (Options-Delta). Sibling: [[strat_market_making]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Headline

No category clears all K7 gates. The longshot-premium harvest is not deployable from this model-free pass; effort should return to copytrade unless a narrower category is pre-registered and retested.

The strongest warning is capacity and tail clustering, not calibration. Several categories show low-price tokens that are overpriced at last-traded prices, and some actual maker-sell tail fills are profitable net of maker rebates. But the categories with enough flow are often either capacity-captured, cluster-risky, or have CIs that do not clear zero.

I split the premium test into two sub-gates: broad calibration and actual maker-sell overlay. That is why the rank table reports gates out of five. In this pass, the strong overlay categories do **not** also clear the calibration gate; `culture` clears calibration, but its tradeable overlay CI does not clear zero.

## Method

- Universe: resolved binary markets in the owned trade window, `DATA_START=2025-08-01`, `DATA_END=2026-05-31`.
- Calibration price: last executed trade before fixed time-to-resolution (`1h`, `4h`, `1d`, `7d`, `30d`) with a staleness filter. This is not a full order-book quote.
- Longshot tail: outcome-token price between **0.005** and **0.20**.
- Tradeable overlay: historical maker sells of tail contracts, held to settlement. Net PnL is `sell_price - resolution_payout + maker_rebate`; no exit spread and no external model.
- CIs bootstrap by market.

## Final Rank

| rank | category | clears | gates /5 | calib net gap | calib CI | overlay bps | overlay CI | fills/day | $/day | top3 cap | cluster loss x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | economics | no | 4 | 0.97c | [-1.29c, 3.20c] | 9,825 bps | [9,204 bps, 10,048 bps] | 1,263.28 | $29,099 | 34.6% | 0.00 |
| 2 | politics_negrisk | no | 4 | -1.63c | [-3.13c, -0.25c] | 7,446 bps | [3,067 bps, 8,883 bps] | 1,026.75 | $12,515 | 27.7% | 0.03 |
| 3 | tech | no | 4 | -0.51c | [-1.03c, 0.05c] | 5,355 bps | [2,165 bps, 7,578 bps] | 1,495.44 | $10,094 | 40.8% | 0.02 |
| 4 | culture | no | 4 | 1.59c | [0.09c, 2.93c] | 4,407 bps | [-4,896 bps, 9,052 bps] | 237.92 | $1,348 | 34.1% | 0.02 |
| 5 | other | no | 4 | -0.85c | [-1.04c, -0.68c] | 1,934 bps | [307 bps, 3,257 bps] | 18,269.09 | $144,061 | 48.7% | 0.03 |
| 6 | daily_crypto | no | 4 | -0.12c | [-0.41c, 0.19c] | 588 bps | [103 bps, 1,075 bps] | 21,798.71 | $44,522 | 66.7% | 0.01 |
| 7 | sports | no | 3 | -3.23c | [-3.94c, -2.58c] | 1,092 bps | [-1,637 bps, 4,003 bps] | 1,893.58 | $32,810 | 63.9% | 0.05 |
| 8 | geopolitics | no | 3 | -0.62c | [-1.45c, 0.26c] | 476 bps | [-7,196 bps, 6,101 bps] | 3,544.92 | $42,367 | 31.2% | 0.27 |
| 9 | weather | no | 3 | -0.28c | [-0.66c, 0.03c] | -0.8 bps | [-1,182 bps, 1,045 bps] | 2,814.81 | $2,945 | 58.2% | 0.03 |
| 10 | finance | no | 3 | -2.73c | [-3.61c, -1.89c] | -591 bps | [-4,633 bps, 3,307 bps] | 1,508.34 | $6,208 | 55.9% | 0.24 |
| 11 | crypto_4h | no | 2 | 0.67c | [-0.49c, 1.74c] | 1,149 bps | [-4,147 bps, 4,779 bps] | 109.88 | $386 | 94.2% | 0.18 |

## Calibration Gate

Tail calibration is the average low-price gap: traded price minus realized outcome frequency, plus maker rebate for the net-gap column.

| category | obs | markets | avg price | realized | gap | gap CI | net gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| culture | 995 | 654 | 4.96c | 3.4% | 1.54c | [0.04c, 2.88c] | 1.59c |
| economics | 461 | 204 | 5.47c | 4.6% | 0.91c | [-1.34c, 3.15c] | 0.97c |
| crypto_4h | 1963 | 1955 | 8.21c | 7.6% | 0.57c | [-0.59c, 1.64c] | 0.67c |
| daily_crypto | 16101 | 13819 | 3.09c | 3.2% | -0.16c | [-0.45c, 0.15c] | -0.12c |
| weather | 32464 | 19928 | 5.34c | 5.7% | -0.33c | [-0.70c, -0.02c] | -0.28c |
| tech | 7695 | 5446 | 5.07c | 5.6% | -0.56c | [-1.08c, -0.00c] | -0.51c |
| geopolitics | 6837 | 3802 | 6.01c | 6.6% | -0.63c | [-1.46c, 0.24c] | -0.62c |
| other | 116770 | 90654 | 6.02c | 6.9% | -0.91c | [-1.10c, -0.74c] | -0.85c |
| politics_negrisk | 3225 | 1621 | 6.10c | 7.8% | -1.69c | [-3.18c, -0.31c] | -1.63c |
| finance | 6502 | 4538 | 7.58c | 10.4% | -2.81c | [-3.69c, -1.97c] | -2.73c |
| sports | 9094 | 7056 | 6.60c | 9.9% | -3.28c | [-3.99c, -2.63c] | -3.23c |

## Tradeable Overlay

Actual historical maker-sell tail fills, held to settlement.

| category | markets | fills | sell USD | net PnL | bps | CI | fill win rate | worst market |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| economics | 304 | 344828 | $7,942,947 | $7,803,847 | 9,825 bps | [9,204 bps, 10,048 bps] | 99.6% | $-119,374 |
| politics_negrisk | 2355 | 286390 | $3,490,806 | $2,599,194 | 7,446 bps | [3,067 bps, 8,883 bps] | 95.6% | $-165,950 |
| tech | 5966 | 418031 | $2,821,503 | $1,510,862 | 5,355 bps | [2,165 bps, 7,578 bps] | 96.3% | $-241,099 |
| culture | 1015 | 66131 | $374,670 | $165,100 | 4,407 bps | [-4,896 bps, 9,052 bps] | 96.0% | $-119,549 |
| other | 77976 | 5109210 | $40,288,618 | $7,791,974 | 1,934 bps | [307 bps, 3,257 bps] | 93.7% | $-1,805,858 |
| crypto_4h | 2345 | 22419 | $78,691 | $9,039 | 1,149 bps | [-4,147 bps, 4,779 bps] | 91.4% | $-12,939 |
| sports | 7613 | 528640 | $9,159,768 | $1,000,663 | 1,092 bps | [-1,637 bps, 4,003 bps] | 95.0% | $-398,206 |
| daily_crypto | 134430 | 6098418 | $12,455,495 | $732,228 | 588 bps | [103 bps, 1,075 bps] | 90.0% | $-97,391 |
| geopolitics | 5061 | 990223 | $11,834,730 | $563,788 | 476 bps | [-7,196 bps, 6,101 bps] | 94.6% | $-2,490,464 |
| weather | 28460 | 786542 | $822,989 | $-63 | -0.8 bps | [-1,182 bps, 1,045 bps] | 94.1% | $-19,127 |
| finance | 42700 | 421942 | $1,736,734 | $-102,569 | -591 bps | [-4,633 bps, 3,307 bps] | 90.7% | $-267,245 |

## Flow

This is the demand available to sell into: takers buying the longshot from a maker.

| category | fills | markets | fills/day | fills/active day | $/day | takers | top3 taker share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| daily_crypto | 6098418 | 134430 | 21,798.71 | 21,858.13 | $44,522 | 238841 | 7.2% |
| other | 5109210 | 77976 | 18,269.09 | 18,312.58 | $144,061 | 477427 | 3.0% |
| geopolitics | 990223 | 5061 | 3,544.92 | 3,549.19 | $42,367 | 164467 | 3.0% |
| weather | 786542 | 28460 | 2,814.81 | 2,819.15 | $2,945 | 40171 | 3.1% |
| sports | 528640 | 7613 | 1,893.58 | 1,894.77 | $32,810 | 125112 | 7.1% |
| finance | 421942 | 42700 | 1,508.34 | 1,512.34 | $6,208 | 55758 | 3.1% |
| tech | 418031 | 5966 | 1,495.44 | 1,498.32 | $10,094 | 106543 | 1.9% |
| economics | 344828 | 304 | 1,263.28 | 1,267.75 | $29,099 | 107701 | 3.2% |
| politics_negrisk | 286390 | 2355 | 1,026.75 | 1,030.18 | $12,515 | 75016 | 4.7% |
| culture | 66131 | 1015 | 237.92 | 239.61 | $1,348 | 19277 | 5.8% |
| crypto_4h | 22419 | 2345 | 109.88 | 110.44 | $386 | 4986 | 28.1% |

## Tail And Cluster Risk

Worst resolution-day cluster by category.

| category | worst day | cluster PnL | markets | fills | loss / positive profit |
| --- | --- | --- | --- | --- | --- |
| geopolitics | 2026-04-07 00:00:00 | $-2,661,104 | 20 | 7510 | 0.27 |
| finance | 2025-11-21 00:00:00 | $-290,420 | 117 | 8177 | 0.24 |
| crypto_4h | 2026-02-25 00:00:00 | $-12,482 | 16 | 280 | 0.18 |
| sports | 2026-02-13 00:00:00 | $-398,145 | 44 | 1124 | 0.05 |
| other | 2025-11-01 00:00:00 | $-1,048,883 | 252 | 26939 | 0.03 |
| weather | 2025-11-26 00:00:00 | $-18,729 | 17 | 223 | 0.03 |
| politics_negrisk | 2025-10-19 00:00:00 | $-86,620 | 14 | 1830 | 0.03 |
| culture | 2025-10-10 00:00:00 | $-7,692 | 8 | 840 | 0.02 |
| tech | 2025-12-14 00:00:00 | $-55,566 | 8 | 607 | 0.02 |
| daily_crypto | 2025-11-20 00:00:00 | $-107,003 | 353 | 8458 | 0.01 |
| economics | 2026-01-31 00:00:00 | $-4,094 | 12 | 3123 | 0.00 |

## Capacity

Per-market positive maker profit concentration for the tail-selling overlay.

| category | markets | median wallets/market | top1 positive share | top3 positive share | top3 CI |
| --- | --- | --- | --- | --- | --- |
| politics_negrisk | 2355 | 8.00 | 15.0% | 27.7% | [22.9%, 40.6%] |
| geopolitics | 5061 | 9.00 | 15.6% | 31.2% | [27.2%, 35.9%] |
| culture | 1015 | 4.00 | 17.7% | 34.1% | [28.6%, 45.8%] |
| economics | 304 | 13.00 | 16.4% | 34.6% | [30.2%, 39.9%] |
| tech | 5966 | 4.00 | 23.3% | 40.8% | [34.4%, 47.1%] |
| other | 77976 | 3.00 | 28.5% | 48.7% | [47.1%, 50.4%] |
| finance | 42700 | 2.00 | 34.4% | 55.9% | [51.0%, 62.7%] |
| weather | 28460 | 6.00 | 31.0% | 58.2% | [56.9%, 59.4%] |
| sports | 7613 | 3.00 | 38.3% | 63.9% | [56.1%, 70.9%] |
| daily_crypto | 134430 | 5.00 | 40.9% | 66.7% | [66.3%, 67.1%] |
| crypto_4h | 2345 | 2.00 | 72.1% | 94.2% | [93.1%, 95.1%] |

## Conclusion

K7 does **not** green-light a broad cross-category longshot-premium maker. A few categories are interesting research leads, but the combined gate is strict: positive calibrated tail gap, positive net maker overlay with CI, enough buy-flow, not captured by the top makers, and tolerable cluster risk. On this pass, no category clears every gate.

The best next move is not to build another maker around this yet. Either narrow the hypothesis to a specific category/window and retest out-of-sample, or return effort to copytrade, which is the stated fallback.
