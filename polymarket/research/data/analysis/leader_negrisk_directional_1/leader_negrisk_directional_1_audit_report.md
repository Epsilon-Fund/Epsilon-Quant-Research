---
title: "Copy-Execution Audit: leader_negrisk_directional_1"
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
# Copy-Execution Audit: leader_negrisk_directional_1
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


_Generated 2026-05-18 16:32 UTC_

Read-only diagnostic. Trade data covers 2025-01-02 → 2026-04-24 (last available shard; not today 2026-05-16).

## Universe summary
- **label**: leader_negrisk_directional_1
- **address**: 0x629bc4a1e53e1d475beb7ea3d388791e96dd995a
- **window_start**: 2025-01-02
- **window_end_exclusive**: 2026-04-25
- **n_fills**: 38,774
- **dollar_volume_usd**: 12,309,933.68
- **n_positions**: 1,064
- **n_resolved_positions**: 982
- **n_families_covered**: 5
- **leader_pnl_calc_usd**: 994,599.05
- **leader_pnl_closed_positions_usd**: 741,240.84

## Primary family table

`A_opt` / `A_real`: role-mirrored, optimistic vs realistic maker-fill model. `B`: pure taker. `C_opt` / `C_real`: pure maker. `adverse_select_ratio` = A_real maker-fill rate on winning vs losing positions (N/A if either bucket <30 maker fills). Deployable cells have **high capture AND adverse_select_ratio close to 1.0**.

| family   | n_fills | n_positions | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl   | C_opt_pnl | C_real_pnl | A_opt_capture | A_real_capture | A_taker_fallback_pct | A_maker_realfill_rate | adverse_select_ratio | adv_sel_n_win | adv_sel_n_lose |
| -------- | ------- | ----------- | ---------- | --------- | ---------- | ------- | --------- | ---------- | ------------- | -------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| politics |  28,316 |         806 |    340,534 |   -35,154 |    -20,265 | 110,058 |    50,855 |     64,199 |       -0.1032 |        -0.0595 |               0.4343 |                0.1380 |               0.8950 |        11,073 |         10,418 |
|    other |   9,740 |         233 |    645,563 |   450,572 |    406,770 | 565,406 |   402,296 |    365,620 |        0.6980 |         0.6301 |               0.4107 |                0.2054 |               1.6455 |         4,174 |          3,114 |
|    macro |     361 |          16 |      2,260 |     6,914 |      4,321 |    -959 |     3,307 |      2,061 |        3.0597 |         1.9121 |               0.5690 |                0.1155 |               0.3022 |           150 |            153 |
|   crypto |     309 |           6 |     16,112 |     3,819 |      2,098 |  10,367 |     5,562 |      3,840 |        0.2371 |         0.1302 |               0.9011 |                0.0459 |                      |           218 |              0 |
|   sports |      48 |           3 |     -9,870 |    -5,899 |     -4,823 | -10,694 |    -1,191 |       -115 |        0.5977 |         0.4887 |               0.9091 |                0.0769 |                      |             0 |             26 |

### Secondary slices

#### By market lifecycle phase (hours-to-resolution at fill time)
| slice           | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl   | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| --------------- | ----------- | ------- | ---------- | --------- | ---------- | ------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
|            open |         668 |  31,301 |    912,264 |   349,671 |    322,417 | 628,500 |   385,367 |    366,009 |               0.4500 |               0.1622 |                0.1323 |               1.1227 |        0.3833 |         0.3534 |
| near-resolution |         205 |   3,002 |    109,211 |    96,905 |     91,783 |  92,716 |   106,726 |    100,208 |               0.3077 |               0.3903 |                0.3413 |               1.5689 |        0.8873 |         0.8404 |
|          middle |         191 |   4,471 |    -26,876 |   -26,324 |    -26,100 | -47,038 |   -31,265 |    -30,612 |               0.4191 |               0.2193 |                0.1817 |               0.8503 |        0.9795 |         0.9711 |

#### By hour-of-day (UTC)
| slice | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl   | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | ------- | ---------- | --------- | ---------- | ------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| 00-06 |          58 |   1,883 |     -2,953 |   -45,095 |    -52,054 | -22,246 |   -46,774 |    -52,650 |               0.5407 |               0.1475 |                0.1066 |               1.6347 |       15.2731 |        17.6302 |
| 12-18 |         386 |  14,822 |    425,480 |   266,498 |    230,244 | 311,147 |   263,977 |    259,670 |               0.4207 |               0.1799 |                0.1442 |               0.9150 |        0.6263 |         0.5411 |
| 06-12 |         221 |   9,231 |    257,193 |    92,301 |    106,806 | 187,980 |    85,984 |     86,924 |               0.4481 |               0.1825 |                0.1541 |               0.9661 |        0.3589 |         0.4153 |
| 18-24 |         399 |  12,838 |    314,878 |   106,548 |    103,104 | 197,296 |   157,642 |    141,660 |               0.4340 |               0.2015 |                0.1713 |               1.3192 |        0.3384 |         0.3274 |

#### By leader's role on the originating fill
| slice | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl   | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | ------- | ---------- | --------- | ---------- | ------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| maker |         610 |  25,016 |    568,306 |   201,342 |    147,196 | 391,677 |   217,289 |    151,738 |               0.4137 |               0.1645 |                0.1357 |               1.0965 |        0.3543 |         0.2590 |
| taker |         454 |  13,758 |    426,293 |   218,910 |    240,903 | 282,500 |   243,539 |    283,867 |               0.4477 |               0.2441 |                0.2024 |               1.1472 |        0.5135 |         0.5651 |

### Sensitivity: H1 2025 vs 2026-YTD

Flag families with `|H1 cap − 2026 cap| > 30pp` AND `n_fills > 200` as 'structure-shifting': historical capture does not generalise.

| family   | A_opt_capture_2025-H1 | A_opt_capture_2026-YTD | A_real_capture_2025-H1 | A_real_capture_2026-YTD | n_fills_2025-H1 | n_fills_2026-YTD |
| -------- | --------------------- | ---------------------- | ---------------------- | ----------------------- | --------------- | ---------------- |
|   crypto |                0.2089 |                        |                 0.0501 |                         |             265 |                  |
|    macro |               -1.2898 |                 1.7376 |                -0.5941 |                  2.9078 |             182 |               46 |
|    other |                0.7155 |                 0.3346 |                 0.6464 |                  0.2319 |           4,958 |            3,325 |
| politics |                0.2963 |                 0.3028 |                 0.4570 |                  0.2390 |          10,095 |           11,805 |
|   sports |                       |                 0.0000 |                        |                  0.0000 |                 |                7 |

### Sensitivity: fallback cents (taker-leg only)

| family   | A_real_pnl_1c | A_real_pnl_2c | A_real_pnl_3c | A_real_pnl_5c | B_pnl_1c | B_pnl_2c | B_pnl_3c | B_pnl_5c |
| -------- | ------------- | ------------- | ------------- | ------------- | -------- | -------- | -------- | -------- |
|   crypto |         3,030 |         2,564 |         2,098 |         1,166 |   14,398 |   12,382 |   10,367 |    6,335 |
|    macro |         4,848 |         4,584 |         4,321 |         3,794 |    1,254 |      147 |     -959 |   -3,173 |
|    other |       417,282 |       412,026 |       406,770 |       396,258 |  616,158 |  590,782 |  565,406 |  514,655 |
| politics |        11,248 |        -4,509 |       -20,265 |       -51,779 |  254,466 |  182,262 |  110,058 |  -34,351 |
|   sports |        -4,592 |        -4,708 |        -4,823 |        -5,054 |  -10,145 |  -10,419 |  -10,694 |  -11,242 |

### Per-family diagnostics

**politics** — n_fills=28,316, taker=6,825, maker=21,491
- taker_fallback_pct: 43.4%; maker_no-fill_optimistic: 83.1%; maker_no-fill_realistic: 86.2%
- taker lag percentiles (s): p10=2.0, p25=6.0, p50=26.0, p75=102.0, p90=200.0
- any-next-fill cumulative share: <30s=19.3%, <60s=26.3%, <120s=33.5%, <300s=45.4%, none-in-window=54.6%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=3,861: p10=-2.00, p25=-1.00, p50=0.00, p75=1.00, p90=2.00
- crossed-market share: 17.3%

**other** — n_fills=9,740, taker=2,452, maker=7,288
- taker_fallback_pct: 41.1%; maker_no-fill_optimistic: 75.8%; maker_no-fill_realistic: 79.5%
- taker lag percentiles (s): p10=2.0, p25=8.0, p50=26.0, p75=92.0, p90=174.0
- any-next-fill cumulative share: <30s=23.0%, <60s=29.8%, <120s=39.1%, <300s=52.6%, none-in-window=47.3%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=1,445: p10=-1.87, p25=-0.50, p50=0.00, p75=1.00, p90=1.86
- crossed-market share: 23.7%

**macro** — n_fills=361, taker=58, maker=303
- taker_fallback_pct: 56.9%; maker_no-fill_optimistic: 87.5%; maker_no-fill_realistic: 88.4%
- taker lag percentiles (s): p10=30.0, p25=30.0, p50=36.0, p75=88.0, p90=166.0
- any-next-fill cumulative share: <30s=6.4%, <60s=13.9%, <120s=21.6%, <300s=33.2%, none-in-window=66.8%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=25: p10=-2.00, p25=0.00, p50=1.00, p75=1.00, p90=3.00
- crossed-market share: 14.7%

**crypto** — n_fills=309, taker=91, maker=218
- taker_fallback_pct: 90.1%; maker_no-fill_optimistic: 94.0%; maker_no-fill_realistic: 95.4%
- taker lag percentiles (s): p10=40.0, p25=40.0, p50=40.0, p75=42.0, p90=42.0
- any-next-fill cumulative share: <30s=1.9%, <60s=6.8%, <120s=7.1%, <300s=8.7%, none-in-window=91.3%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=9: p10=-8.00, p25=-8.00, p50=-8.00, p75=-7.00, p90=-7.00
- crossed-market share: 1.9%

## Sanity checks
- A_total_pnl_match_within_10pct: **FAIL**
- B_fill_count_subset_invariants: **PASS**
- C_pnl_monotonicity_warnings: **PASS**
  - leader PnL (replay calc) = 994,599; closed_positions sum = 741,241 (34.2% drift). Difference largely from marking unresolved positions to last fill price; closed_positions does not.

PnL-monotonicity informationals (NOT bugs — see check description):
  - politics: A_real PnL (-20,265) > A_opt PnL (-35,154) — the maker fills filtered by the realistic model were net-losing, so dropping them raised PnL.
  - politics: C_real PnL > C_opt PnL — same dynamic on the pure-maker branch.
  - sports: A_real PnL (-4,823) > A_opt PnL (-5,899) — the maker fills filtered by the realistic model were net-losing, so dropping them raised PnL.
  - sports: C_real PnL > C_opt PnL — same dynamic on the pure-maker branch.
  - politics (maker_share=76%): pure-taker B PnL (110,058) > role-mirrored A_real (-20,265) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.
  - other (maker_share=75%): pure-taker B PnL (565,406) > role-mirrored A_real (406,770) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.
  - crypto (maker_share=71%): pure-taker B PnL (10,367) > role-mirrored A_real (2,098) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.

## Interpretation
Leader `leader_negrisk_directional_1` is **76% maker** on 38,774 fills with leader realised PnL of **$994,599** in this window, so the audit hinges on the maker-leg fill model and adverse selection more than on taker slippage.

**Deployable family today: other** — A_real capture ≥40%, leader PnL positive, and adverse_select_ratio not destructive.

Adverse selection is destructive for: **macro** (maker-leg fill rate is significantly higher on losing positions than winning ones). This is the central economic blocker — even free maker rebates don't fix this.

Pure-taker (Branch B) is uniformly worse than role-mirroring across the maker-dominant families — paying 3¢ to cross spread on his entire flow burns capital faster than the leader earns from maker rebates.

**Structurally unstable**: other have >30pp capture swings between H1 2025 and 2026 — historical capture there does not generalise; only a live A/B test will answer it.