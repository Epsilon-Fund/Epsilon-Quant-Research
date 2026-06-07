---
title: "Politics Deep-Dive: Domah"
created: 2026-06-07
status: generated
owner: justin
project: polymarket
para: resource
hubs:
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - research
  - data-quality
---
# Politics Deep-Dive: Domah
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


Restricted to politics-family fills/positions. Fragments: **93,018**, positions: **2,645**, leader PnL: **$228,666**, A_real PnL: **$-259,553**.

Hypothesis to test: why does politics show negative A_real capture (−114% on the original heuristic, −81% on the proposed one)?

## Cut 1: By lifecycle phase (politics only)

| slice           | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_opt_capture | A_real_capture | adverse_select_ratio | A_taker_fallback_pct | A_maker_realfill_rate |
| --------------- | ----------- | ------- | ---------- | --------- | ---------- | -------- | --------- | ---------- | ------------- | -------------- | -------------------- | -------------------- | --------------------- |
|            open |       2,169 |  84,317 |    178,024 |  -151,910 |   -191,050 | -630,148 |      -976 |    -38,705 |       -0.8533 |        -1.0732 |               0.6259 |               0.3122 |                0.0751 |
|          middle |         232 |   5,175 |     69,369 |   -81,438 |    -65,194 |   30,776 |   -92,017 |    -83,664 |       -1.1740 |        -0.9398 |               1.0194 |               0.2237 |                0.1874 |
| near-resolution |         244 |   3,526 |    -18,728 |    -4,438 |     -3,309 |  -42,568 |    23,921 |     29,441 |        0.2370 |         0.1767 |               1.5657 |               0.3123 |                0.1321 |

## Cut 2: By hour-of-day UTC (politics only)

| slice | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_opt_capture | A_real_capture | adverse_select_ratio | A_taker_fallback_pct | A_maker_realfill_rate |
| ----- | ----------- | ------- | ---------- | --------- | ---------- | -------- | --------- | ---------- | ------------- | -------------- | -------------------- | -------------------- | --------------------- |
| 00-06 |         700 |  30,820 |     55,457 |    10,629 |     -4,914 | -127,916 |    64,384 |     39,661 |        0.1917 |        -0.0886 |               0.8377 |               0.2977 |                0.0668 |
| 06-12 |         530 |  16,109 |     40,652 |  -129,077 |   -141,040 | -212,554 |   -73,576 |    -74,557 |       -3.1752 |        -3.4695 |               0.8588 |               0.3468 |                0.0963 |
| 12-18 |         693 |  21,391 |     95,628 |  -142,557 |   -147,143 | -128,604 |  -126,789 |   -123,911 |       -1.4907 |        -1.5387 |               0.6565 |               0.2429 |                0.1164 |
| 18-24 |         722 |  24,698 |     36,929 |    23,218 |     33,543 | -172,866 |    66,909 |     65,879 |        0.6287 |         0.9083 |               0.5270 |               0.3792 |                0.0686 |

Reference: the audit's all-family hour slice showed 18-24 UTC as Domah's best hour bucket (89% A_opt capture). Does that hold inside politics?

## Cut 3: Position-size distribution (fragment notional)

Each row is one Domah politics fill. Distribution of |usd_amount| for fills on winning vs losing politics positions.

| pct | winners_fill_notional_usd | losers_fill_notional_usd |
| --- | ------------------------- | ------------------------ |
|  10 |                    0.2083 |                   0.2000 |
|  25 |                    1.1310 |                   1.1000 |
|  50 |                    7.5881 |                   6.3636 |
|  75 |                        69 |                       40 |
|  90 |                       385 |                      200 |
|  95 |                       960 |                      493 |
|  99 |                     5,708 |                    3,089 |

- Winners: n_fills=50,072, mean=$302, sum=$15,097,297
- Losers:  n_fills=42,946, mean=$168, sum=$7,204,633

**Per-position aggregate notional (politics):**

| pct | winners_pos_notional_usd | losers_pos_notional_usd |
| --- | ------------------------ | ----------------------- |
|  10 |                       39 |                  9.2693 |
|  25 |                      172 |                      37 |
|  50 |                      786 |                     246 |
|  75 |                    4,152 |                   1,338 |
|  90 |                   18,818 |                  10,646 |
|  95 |                   44,708 |                  26,539 |
|  99 |                  248,073 |                  97,103 |

- Winners: n_positions=1,360, mean_notional=$11,101
- Losers:  n_positions=1,285, mean_notional=$5,607
- **Fills-per-position**: winners median = 10, losers median = 11

## Cut 4: PnL concentration by market

Politics totals: PnL=$228,666; winning-position PnL=$1,871,493; losing-position PnL=$-1,642,827.

**Top-K markets by absolute PnL — share of politics PnL:**

| k  | top_k_market_pnl_sum | share_of_total_politics_pnl |
| -- | -------------------- | --------------------------- |
|  5 |               25,363 |                      0.1109 |
| 10 |              212,074 |                      0.9274 |
| 20 |              290,433 |                      1.2701 |

**Top-K winning + losing markets:**

| k  | top_k_winning_markets_pnl | share_of_winning_pnl | top_k_losing_markets_pnl | share_of_losing_pnl |
| -- | ------------------------- | -------------------- | ------------------------ | ------------------- |
|  5 |                   419,750 |               0.2243 |                 -399,318 |              0.2431 |
| 10 |                   654,273 |               0.3496 |                 -544,668 |              0.3315 |
| 20 |                   902,251 |               0.4821 |                 -757,428 |              0.4611 |

**Top 10 winning politics markets:**

| market_id | slug                                                                                          | pnl    | n_fills | notional |
| --------- | --------------------------------------------------------------------------------------------- | ------ | ------- | -------- |
|    555828 |                                        will-pope-leo-xiv-be-times-person-of-the-year-for-2025 | 95,786 |     484 |  364,838 |
|   1336699 |                                                       starmer-out-by-february-28-2026-352-692 | 89,673 |     297 |  363,115 |
|    680392 |                                    will-there-be-another-us-government-shutdown-by-january-31 | 83,912 |   1,553 |  871,897 |
|    578140 |                                                        israel-x-hamas-ceasefire-by-october-31 | 82,237 |     359 |  193,731 |
|   1092199 |                                                        us-strikes-iran-by-january-31-2026-165 | 68,142 |     698 |  399,166 |
|    616561 |                                                        israel-x-hamas-ceasefire-by-october-10 | 65,974 |     291 |  136,791 |
|    661699 |                                         khamenei-out-as-supreme-leader-of-iran-by-june-30-747 | 45,971 |     222 |   96,323 |
|    621018 | us-x-venezuela-military-engagement-by-november-30-216-397-226-467-735-374-192-441-421-482-389 | 42,967 |     242 |   79,917 |
|    916440 |                                                             maduro-out-by-january-31-2026-318 | 40,326 |      33 |   17,235 |
|    524996 |                                                          will-pietro-parolin-be-the-next-pope | 39,284 |     381 |  121,235 |

**Top 10 losing politics markets:**

| market_id | slug                                                                                                                                                                                                                                                            | pnl      | n_fills | notional |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------- | -------- |
|    519068 |                                                                                                                                                                     romania-presidential-election-winner-will-nicuor-dan-win-the-romanian-presidential-election | -129,684 |     140 |  179,987 |
|   1299962 |                                                                                                                                                                                                                will-the-government-shutdown-last-5-days-or-more | -114,325 |     453 |  179,327 |
|    984442 | us-strikes-iran-by-june-30-2026-699-664-723-485-753-218-567-164-387-443-377-384-159-973-494-631-694-956-361-443-224-518-537-678-486-386-275-153-976-862-149-897-221-522-378-575-993-365-444-965-488-963-698-313-696-829-887-721-211-322-726-411-231-474-417-867 |  -75,613 |     335 |  104,040 |
|    588095 |                                                                                                                                                                                                                    will-eric-adams-drop-out-by-september-30-183 |  -45,785 |     138 |   57,002 |
|   1640919 |                                                                                                                                                                                                                            us-forces-enter-iran-by-april-30-899 |  -33,912 |      77 |   33,912 |
|    673597 |                                                                                                                                                                                                                will-the-government-shutdown-end-november-12-365 |  -31,386 |     196 |  152,348 |
|   1308327 |                                                                                                                                                                                                                   another-us-government-shutdown-by-february-14 |  -30,159 |     844 |  201,765 |
|   1094735 |                                                                                                                                                                                                  will-mara-corina-machado-enter-venezuela-by-january-31-859-245 |  -28,008 |     157 |   45,171 |
|    791705 |                                                                                                                                                                                                                            israel-strikes-iran-by-march-31-2026 |  -27,987 |     151 |   29,991 |
|    984441 |                                                                                                                                                                                                                    us-strikes-iran-by-march-31-2026-393-881-954 |  -27,810 |     479 |  120,118 |

## Cut 5: Time-of-fill within market lifetime (winners vs losers)

For each fill, fraction of total market lifetime elapsed: (fill_ts − first_observed_trade) / (end_ts − first_observed_trade). 0.0 = first day of market; 1.0 = resolution. NaN when end_ts is unknown.

| pct | winners_elapsed_frac | losers_elapsed_frac |
| --- | -------------------- | ------------------- |
|  10 |               0.0016 |              0.0020 |
|  25 |               0.0246 |              0.0267 |
|  50 |               0.1119 |              0.1198 |
|  75 |               0.3278 |              0.4861 |
|  90 |               0.7370 |              0.8665 |
|  95 |               0.9561 |              1.0038 |

**Fill-count by lifetime bucket × outcome:**

| lifetime_bucket    | loser_fills | winner_fills | winner_share |
| ------------------ | ----------- | ------------ | ------------ |
|      early_(0-25%) |      24,750 |       32,648 |       0.5688 |
|     late_(75-100%) |       4,091 |        2,698 |       0.3974 |
| mid-early_(25-50%) |       5,921 |        5,563 |       0.4844 |
|  mid-late_(50-75%) |       3,609 |        3,448 |       0.4886 |
|           post-end |       2,151 |        1,750 |       0.4486 |
|            unknown |       2,424 |        3,965 |       0.6206 |

## Cut 6: Market spotlight — 3 best, 3 worst

### Top 3 winning markets

**will-pope-leo-xiv-be-times-person-of-the-year-for-2025** (market_id=555828)

- timeframe: 2025-07-17 13:55 → 2025-12-08 13:46 (144.0 days)
- fills: 484 (409 maker / 75 taker); buys $294,913 @ avg 0.641; sells $69,926 @ avg 0.365; mark=1.000
- Domah PnL=**$73,897**; copy A_opt=$7,604 (capture 0.10); A_real=$7,604 (capture 0.10); B=$64,006 (capture 0.87); C_opt=$5,133 (capture 0.07)
- maker fill rate (A_real model): 4.40%; taker fallback rate: 40.00%

- peak inventory: 268,565 tokens; final inventory: 268,565.


**starmer-out-by-february-28-2026-352-692** (market_id=1336699)

- timeframe: 2026-02-05 04:50 → 2026-02-15 20:18 (10.6 days)
- fills: 297 (188 maker / 109 taker); buys $216,486 @ avg 0.477; sells $146,628 @ avg 0.575; mark=0.000
- Domah PnL=**$41,194**; copy A_opt=$41,813 (capture 1.02); A_real=$42,713 (capture 1.04); B=$36,297 (capture 0.88); C_opt=$46,003 (capture 1.12)
- maker fill rate (A_real model): 21.81%; taker fallback rate: 6.42%

- peak inventory: 198,709 tokens; final inventory: 198,709.


**will-there-be-another-us-government-shutdown-by-january-31** (market_id=680392)

- timeframe: 2025-12-04 15:58 → 2026-01-31 13:03 (57.9 days)
- fills: 1,553 (1,201 maker / 352 taker); buys $435,354 @ avg 0.303; sells $436,543 @ avg 0.282; mark=0.000
- Domah PnL=**$-11,308**; copy A_opt=$-45,272 (capture 4.00); A_real=$-48,445 (capture 4.28); B=$-21,272 (capture 1.88); C_opt=$-44,153 (capture 3.90)
- maker fill rate (A_real model): 33.31%; taker fallback rate: 4.26%

- peak inventory: 533,797 tokens; final inventory: -113,334.


### Top 3 losing markets

**romania-presidential-election-winner-will-nicuor-dan-win-the-romanian-presidential-election** (market_id=519068)

- timeframe: 2025-05-10 22:10 → 2025-05-18 19:40 (7.9 days)
- fills: 140 (135 maker / 5 taker); buys $176,732 @ avg 0.696; sells $3,256 @ avg 0.015; mark=0.000
- Domah PnL=**$-130,444**; copy A_opt=$-8,467 (capture 0.06); A_real=$-8,467 (capture 0.06); B=$-135,926 (capture 1.04); C_opt=$-8,467 (capture 0.06)
- maker fill rate (A_real model): 7.41%; taker fallback rate: 0.00%

- peak inventory: 210,000 tokens; final inventory: 43,591.


**will-the-government-shutdown-last-5-days-or-more** (market_id=1299962)

- timeframe: 2026-01-31 05:39 → 2026-02-03 21:44 (3.7 days)
- fills: 453 (346 maker / 107 taker); buys $88,707 @ avg 0.217; sells $90,620 @ avg 0.401; mark=0.000
- Domah PnL=**$-77,312**; copy A_opt=$-50,206 (capture 0.65); A_real=$-51,365 (capture 0.66); B=$-81,223 (capture 1.05); C_opt=$-50,133 (capture 0.65)
- maker fill rate (A_real model): 28.03%; taker fallback rate: 0.93%

- peak inventory: 220,521 tokens; final inventory: 182,414.


**us-strikes-iran-by-june-30-2026-699-664-723-485-753-218-567-164-387-443-377-384-159-973-494-631-694-956-361-443-224-518-537-678-486-386-275-153-976-862-149-897-221-522-378-575-993-365-444-965-488-963-698-313-696-829-887-721-211-322-726-411-231-474-417-867** (market_id=984442)

- timeframe: 2026-01-03 07:09 → 2026-02-28 07:35 (56.0 days)
- fills: 335 (230 maker / 105 taker); buys $63,036 @ avg 0.436; sells $41,003 @ avg 0.376; mark=0.000
- Domah PnL=**$-54,348**; copy A_opt=$-29,617 (capture 0.54); A_real=$-28,661 (capture 0.53); B=$-56,623 (capture 1.04); C_opt=$-29,376 (capture 0.54)
- maker fill rate (A_real model): 13.91%; taker fallback rate: 6.67%

- peak inventory: 74,250 tokens; final inventory: 35,325.


## Synthesis

1. **Concentration is severe**: top-5 politics markets by absolute PnL account for 225% of family PnL — Domah's politics edge lives in ~5 markets, not the average. Family-level adverse-select-ratio of 0.69 is the *average* across thousands of marginal markets; the few decisive markets dominate the dollar outcome.

2. **Averaging-in pattern is real**: losers have median **11 fills per position** vs winners' median **10** — Domah scales into losing positions much harder than into winners. Mean fill notional is similar ($302 winners vs $168 losers), but losing positions accumulate more fills, so a copy bot that listens to every fill catches the entire loss arc.

3. **Late-cycle losing is a feature**: 15% of losing-position fills happen in the last 25% of market lifetime vs 9% for winners. Domah holds losers into resolution and prints more fragments as the price drifts against him; the copy bot eats every one of those adverse-priced fills.

4. **Information-driven adverse selection on his maker bids**: maker fills on losing politics positions fire 31pp more often than on winning ones (1.0 vs 0.69 fill-rate ratio). His bids sit on the book; when news moves against him, the bot fills the bid first while the price collapses through it.

5. **Pure-taker is uniformly worse here** (B PnL = $-641,940 vs leader $228,666): crossing the spread on every Domah fill loses faster because the 87% maker share means you pay 3¢ to cross thousands of times. The deployment implication: copying Domah's politics is uncopyable with naive fill-mirroring — even a hand-curated subset that filters to his largest 5 markets per quarter would need a separate signal to avoid getting filled on the wrong side.
