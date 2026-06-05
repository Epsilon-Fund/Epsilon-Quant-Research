# Block K-PEG Robustness Review

> **Strat:** [[strat_market_making]] (Market-Making). Sibling: [[strat_options_delta]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Verdict

| claim | verdict | read |
| --- | --- | --- |
| 1. V0 faithfully reproduces K-PEG | CONFIRM | The rerun gives V0 pooled 759.6 bps and Crypto 692.9 bps; canonical `kpeg.simulate` in the audit rerun prints the same values within rounding. |
| 2. No arithmetic bug / no future in entry | MOSTLY CONFIRM | The identity error max is 7.28e-12 bps. I do not see future-mid leakage into quote formation. However, 58/409 30s lookups and 36/409 60s lookups fall back to a stale pre-fill state, so the "future_mid" cost is sometimes not actually future. |
| 3. Taker round-trip flips sign | PARTIALLY CONFIRM / OVERSTATED | The audit's entry-time half-spread proxy gives Crypto -753.4 bps, confirming its calculation. But repricing at the actual pre-resolution +60s book touch is only possible for 32/403 Crypto fills and is +2555.6 bps on that eligible subset. The stronger finding is that most K-PEG fills are too close to resolution for a K2-style +60s taker exit to be feasible. |
| 4. Spread structure / late fills | PARTIALLY CONFIRM | The audit's candidate-time spread table reproduces, and 341/370 (92.2%) of crypto-4h fills are in 120-240m. On the full book-state panel, spreads also widen late, but the exact levels differ from the audit because the audit used trade-candidate states, not every book state. |
| 5. K-PEG positive is mark-to-mid artifact | PARTIALLY CONFIRM | The neutral maker-loop version is not established: the positive V0/V1 depends on mark-to-mid and many fills have no feasible post-entry pre-resolution exit. But hold-to-resolution via Gamma settlement is positive in-sample, so the result is better described as "not a standalone neutral maker edge" rather than pure nonsense. |

## Reproduced Numbers

| metric | scope | n | mean | 95% CI | win |
| --- | --- | --- | --- | --- | --- |
| v0_reproduce | pooled | 409 | 759.6 bps | [415.7 bps, 1109.1 bps] | 79.0% |
| v2_entry_halfspread_proxy | pooled | 409 | -742.4 bps | [-1232.1 bps, -347.0 bps] | 26.2% |
| actual_taker_exit_30s | pooled | 38 | 41152.6 bps | [2187.2 bps, 124188.7 bps] | 89.5% |
| actual_taker_exit_60s | pooled | 38 | 41108.4 bps | [2109.1 bps, 124191.9 bps] | 89.5% |
| v0_reproduce | Crypto | 403 | 692.9 bps | [370.1 bps, 1078.9 bps] | 78.7% |
| v2_entry_halfspread_proxy | Crypto | 403 | -753.4 bps | [-1167.9 bps, -380.3 bps] | 26.3% |
| actual_taker_exit_30s | Crypto | 32 | 2608.1 bps | [1674.2 bps, 3328.8 bps] | 90.6% |
| actual_taker_exit_60s | Crypto | 32 | 2555.6 bps | [1608.2 bps, 3379.1 bps] | 90.6% |

## Timestamp And Identity Checks

- Max absolute V0 identity error: `7.27596e-12` bps.
- Future-state ordering:

| horizon | n | future state before fill | median lag to target | p95 lag to target |
| --- | --- | --- | --- | --- |
| 30 | 409 | 58 | 0.879s | 176.841s |
| 60 | 409 | 36 | 0.784s | 117.109s |

## Clean Null

The category-shuffle placebo in the audit is not a valid null: it mixes price levels across markets, so it can create huge artificial PnL when entry price and settlement level are correlated. A cleaner per-market/per-asset circular shift of `future_mid_30` gives median mean `5191.4 bps` with 95% randomization range `[4036.7 bps, 6028.6 bps]`. That does not prove an entry leak; it says the category shuffle was confounded.

## Exit-Cost Fairness

The original V2 used entry-time half-spread as the exit spread. Actual +60s exit half-spread is lower on average for the 32 eligible Crypto fills (`270.5 bps` vs audit proxy `1636.0 bps`), so V2 is pessimistic in magnitude. But this is not a clean rescue: the actual pre-resolution +60s exit exists for only 32/403 Crypto fills because most selected fills are too close to resolution.

## Maker Exit Stress

Re-running the existing maker-exit script on the current full-panel artifacts gives lower fill than the addendum: mid-offset Crypto maker-exit fill is about `5.0%`, not 12%. In the independent extension below, I cap exits before resolution; that leaves only 32 eligible Crypto fills for post-entry maker-exit testing. The positive long-window cells are therefore not comparable to all 403 K-PEG Crypto fills.

Mid-offset Crypto by exit window:

| window s | maker fill | mean | 95% CI | win |
| --- | --- | --- | --- | --- |
| 30 | 15.6% | 2649.4 bps | [1633.4 bps, 3416.3 bps] | 90.6% |
| 60 | 31.2% | 2641.5 bps | [1570.9 bps, 3382.8 bps] | 90.6% |
| 120 | 40.6% | 2692.7 bps | [1766.0 bps, 3319.8 bps] | 93.8% |
| 300 | 46.9% | 2347.5 bps | [1476.6 bps, 2712.2 bps] | 93.8% |
| 600 | 50.0% | 2313.0 bps | [1278.8 bps, 2729.3 bps] | 93.8% |
| 900 | 53.1% | 2254.3 bps | [1241.2 bps, 2749.4 bps] | 93.8% |
| 1800 | 56.2% | 2287.3 bps | [1299.8 bps, 2739.7 bps] | 93.8% |
| 3600 | 81.2% | 3152.5 bps | [2330.6 bps, 4121.3 bps] | 93.8% |
| 7200 | 90.6% | 3154.4 bps | [2490.1 bps, 3932.1 bps] | 93.8% |

Best Crypto maker-exit cells tested:

| offset | window s | maker fill | mean | 95% CI |
| --- | --- | --- | --- | --- |
| 5 | 7200 | 87.5% | 3632.9 bps | [2886.3 bps, 4458.7 bps] |
| 5 | 3600 | 78.1% | 3581.0 bps | [2670.0 bps, 4558.5 bps] |
| 3 | 7200 | 90.6% | 3467.2 bps | [2732.1 bps, 4311.9 bps] |
| 3 | 3600 | 78.1% | 3410.8 bps | [2536.3 bps, 4382.1 bps] |
| 2 | 7200 | 90.6% | 3363.0 bps | [2650.0 bps, 4182.7 bps] |
| 2 | 3600 | 81.2% | 3329.1 bps | [2472.8 bps, 4300.4 bps] |
| 1 | 7200 | 90.6% | 3258.7 bps | [2570.4 bps, 4044.0 bps] |
| 1 | 3600 | 81.2% | 3240.8 bps | [2410.9 bps, 4214.3 bps] |

The first mid-offset eligible-subset cell at or above 40% fill is window=120s, but again this is 13 maker fills out of 32 eligible fills, not 40% of the full 403-fill Crypto set.

## Spread Phase

Full-panel crypto-4h book-state spread, not just candidate trade states:

| phase | states | mean spread | median spread |
| --- | --- | --- | --- |
| 0-15m | 630 | 344.2 bps | 400.0 bps |
| 15-30m | 867 | 365.0 bps | 400.0 bps |
| 30-60m | 1,505 | 359.8 bps | 400.0 bps |
| 60-120m | 16,829 | 795.0 bps | 606.1 bps |
| 120-240m | 1,246,656 | 1612.9 bps | 1052.6 bps |
| 240m+ | 74,250 | 2002.9 bps | 1138.2 bps |

The slug epoch parse checks out against Gamma metadata: for 21 crypto-4h markets, Gamma `endDate` equals `slug_epoch + 4h` exactly.

## Fill Model Stress

| scope | fills | strict through | inside improve | join touch | queue survive | survive rate | median trade size | median queue ahead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 409 | 57 | 134 | 0 | 166 | 40.6% | nan | 100.00 |
| Crypto | 403 | 56 | 133 | 0 | 165 | 40.9% | nan | 100.00 |

Interpretation: full-priority fills are generous. A strict-through-only assumption would keep only 56/403 Crypto fills. The `queue survive` count is mostly strict-through plus inside-spread improvement; exact trade-size matching was not reliable in this panel join, and there were no join-touch fills in the selected set. This does not kill every fill, but it materially reduces capacity before any exit-cost critique.

## Hold To Resolution

Gamma lookups succeeded for 25/25 crypto markets. Hold-to-resolution over Crypto fills gives 1846.4 bps, CI [620.2 bps, 3265.1 bps], win rate 38.0%, n=403. This is the Track-A style that avoids Polymarket exit spread, but it converts the strategy into a directional/resolution-risk book, not a neutral maker exit.

## Denominator / Capacity

The K-PEG headline fill rate is fills divided by quote-state opportunities times two, so it is repricing-inflated. The deployable volume read is simpler: 403 Crypto fills over 2.43 days, about 166 fills/day. Using the full span, that is about 6.9 Crypto fills per active clock hour; per active market-hour is higher because only a subset of crypto-4h markets are live at a time, but still one-share scale in this simulation.

## Independent Opinion

K-PEG is not a standalone market-making edge as currently specified. The positive V0/V1 result is a mark-to-mid / mean-reversion-to-micro diagnostic. Once I require an exit, the thesis becomes much more conditional: the audit's forced V2 is negative, but it overstates the round-trip case by using entry-time spread and by treating +60s exit as feasible for fills that are already too close to resolution. Queue/size assumptions also cut the apparent fill set.

The salvage path is not K-PEG as a neutral maker loop. The only plausible salvage is Track A: enter passively when the quote is favorable, then hold/hedge to resolution using a separate Binance/Chainlink digital-value model. That is no longer "baseline maker PnL"; it is a directional/resolution-risk strategy with maker entry alpha, and it needs fresh OOS validation, capital/risk limits, and hedge accounting before it earns the word edge.
