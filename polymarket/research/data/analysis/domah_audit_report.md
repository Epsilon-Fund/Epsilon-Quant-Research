---
title: "Domah Copy-Execution Audit"
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
# Domah Copy-Execution Audit
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


_Generated 2026-05-16 15:57 UTC_

Read-only diagnostic. Trade data covers 2025-01-02 → 2026-04-24 (last available shard; not today 2026-05-16).

## Universe summary
- **address**: 0x9d84ce0306f8551e02efef1680475fc0f1dc1344
- **window_start**: 2025-01-02
- **window_end_exclusive**: 2026-04-25
- **n_fills**: 170,005
- **dollar_volume_usd**: 40,465,013.78
- **n_positions**: 5,890
- **n_resolved_positions**: 5,050
- **n_families_covered**: 6
- **leader_pnl_calc_usd**: 941,122.63
- **leader_pnl_closed_positions_usd**: 888,540.20

## Primary family table

`A_opt` / `A_real`: role-mirrored, optimistic vs realistic maker-fill model. `B`: pure taker. `C_opt` / `C_real`: pure maker. `adverse_select_ratio` = A_real maker-fill rate on winning vs losing positions (N/A if either bucket <30 maker fills). Deployable cells have **high capture AND adverse_select_ratio close to 1.0**.

| family   | n_fills | n_positions | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_opt_capture | A_real_capture | A_taker_fallback_pct | A_maker_realfill_rate | adverse_select_ratio | adv_sel_n_win | adv_sel_n_lose |
| -------- | ------- | ----------- | ---------- | --------- | ---------- | -------- | --------- | ---------- | ------------- | -------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| politics |  93,018 |       2,645 |    228,666 |  -237,787 |   -259,553 | -641,940 |   -69,072 |    -92,928 |       -1.0399 |        -1.1351 |               0.3064 |                0.0833 |               0.6915 |        43,258 |         37,462 |
|    other |  44,862 |       2,354 |    352,904 |    64,742 |     71,341 |   62,300 |   107,994 |     87,287 |        0.1835 |         0.2022 |               0.4664 |                0.0574 |               0.9766 |        21,431 |         16,358 |
|    macro |  25,172 |         444 |    420,407 |   304,844 |    274,311 |   60,500 |   344,078 |    305,422 |        0.7251 |         0.6525 |               0.2845 |                0.1298 |               1.4116 |        13,486 |          9,197 |
|   sports |   5,323 |         342 |    -46,596 |   -60,323 |    -69,873 |  -82,178 |   -56,024 |    -66,066 |        1.2946 |         1.4995 |               0.5790 |                0.0750 |               0.3066 |         2,795 |          1,870 |
|   crypto |     817 |          65 |    -21,817 |   -14,703 |    -14,733 |  -33,548 |      -201 |       -360 |        0.6739 |         0.6753 |               0.6041 |                0.0677 |               0.6339 |           336 |            284 |
|  weather |     813 |          40 |      7,558 |       944 |        641 |    2,799 |       375 |        107 |        0.1249 |         0.0848 |               0.8443 |                0.0077 |               0.0000 |           415 |            231 |

### Secondary slices

#### By market lifecycle phase (hours-to-resolution at fill time)
| slice           | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| --------------- | ----------- | ------- | ---------- | --------- | ---------- | -------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
|            open |       4,914 | 154,216 |    929,002 |   235,863 |    152,436 | -500,368 |   481,399 |    352,002 |               0.3742 |               0.0971 |                0.0764 |               0.8478 |        0.2539 |         0.1641 |
| near-resolution |         499 |   6,671 |    -10,714 |   -50,318 |    -39,594 |  -66,214 |   -18,341 |      3,800 |               0.2825 |               0.2067 |                0.1561 |               1.0588 |        4.6964 |         3.6955 |
|          middle |         477 |   9,118 |     22,835 |  -127,827 |   -110,708 |  -65,485 |  -135,908 |   -122,340 |               0.3302 |               0.1883 |                0.1468 |               0.9409 |       -5.5979 |        -4.8482 |

#### By hour-of-day (UTC)
| slice | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | ------- | ---------- | --------- | ---------- | -------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| 18-24 |       1,542 |  49,435 |    369,634 |   331,414 |    299,877 |  -96,762 |   386,295 |    350,872 |               0.3936 |               0.1122 |                0.0914 |               1.0090 |        0.8966 |         0.8113 |
| 00-06 |       1,571 |  52,986 |    202,538 |    15,415 |        467 | -181,614 |   122,020 |     69,388 |               0.3784 |               0.0885 |                0.0660 |               0.6671 |        0.0761 |         0.0023 |
| 12-18 |       1,727 |  43,664 |    289,123 |  -146,021 |   -136,449 | -125,734 |  -102,118 |   -110,398 |               0.3246 |               0.1157 |                0.0917 |               0.7451 |       -0.5050 |        -0.4719 |
| 06-12 |       1,050 |  23,920 |     79,827 |  -143,090 |   -161,761 | -227,957 |   -79,047 |    -76,398 |               0.4041 |               0.1155 |                0.0885 |               0.9790 |       -1.7925 |        -2.0264 |

#### By Domah's role on the originating fill
| slice | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | ------- | ---------- | --------- | ---------- | -------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| taker |       1,281 |  39,736 |    238,061 |   -85,571 |    -93,468 | -195,885 |    28,431 |    -24,815 |               0.3616 |               0.1002 |                0.0826 |               0.3405 |       -0.3594 |        -0.3926 |
| maker |       4,609 | 130,269 |    703,061 |   143,288 |     95,602 | -436,181 |   298,719 |    258,279 |               0.3726 |               0.1077 |                0.0833 |               1.0677 |        0.2038 |         0.1360 |

### Sensitivity: H1 2025 vs 2026-YTD

Flag families with `|H1 cap − 2026 cap| > 30pp` AND `n_fills > 200` as 'structure-shifting': historical capture does not generalise.

| family   | A_opt_capture_2025-H1 | A_opt_capture_2026-YTD | A_real_capture_2025-H1 | A_real_capture_2026-YTD | n_fills_2025-H1 | n_fills_2026-YTD |
| -------- | --------------------- | ---------------------- | ---------------------- | ----------------------- | --------------- | ---------------- |
|   crypto |                0.5639 |                -7.6072 |                 0.5639 |                 -7.6072 |              67 |              171 |
|    macro |                0.4427 |                 0.6993 |                 0.4421 |                  0.5409 |             265 |           11,610 |
|    other |               -2.1123 |                -0.0489 |                -0.0243 |                 -0.1612 |             212 |           13,953 |
| politics |               -0.2784 |               -10.1507 |                -0.2091 |                -11.3531 |           1,412 |           42,550 |
|   sports |               15.5000 |                 1.1979 |                15.5000 |                  1.1564 |               3 |            1,574 |
|  weather |                       |                 0.0060 |                        |                  0.0000 |                 |               61 |

### Sensitivity: fallback cents (taker-leg only)

| family   | A_real_pnl_1c | A_real_pnl_2c | A_real_pnl_3c | A_real_pnl_5c | B_pnl_1c | B_pnl_2c | B_pnl_3c | B_pnl_5c   |
| -------- | ------------- | ------------- | ------------- | ------------- | -------- | -------- | -------- | ---------- |
|   crypto |       -12,588 |       -13,660 |       -14,733 |       -16,879 |  -26,717 |  -30,132 |  -33,548 |    -40,379 |
|    macro |       298,877 |       286,594 |       274,311 |       249,744 |  268,456 |  164,478 |   60,500 |   -147,455 |
|    other |        96,943 |        84,142 |        71,341 |        45,739 |  210,871 |  136,585 |   62,300 |    -86,271 |
| politics |      -159,563 |      -209,558 |      -259,553 |      -359,544 | -187,650 | -414,795 | -641,940 | -1,096,230 |
|   sports |       -64,369 |       -67,121 |       -69,873 |       -75,376 |  -60,662 |  -71,420 |  -82,178 |   -103,694 |
|  weather |         1,161 |           901 |           641 |           121 |    5,847 |    4,323 |    2,799 |       -248 |

### Per-family diagnostics

**politics** — n_fills=93,018, taker=12,298, maker=80,720
- taker_fallback_pct: 30.6%; maker_no-fill_optimistic: 89.4%; maker_no-fill_realistic: 91.7%
- taker lag percentiles (s): p10=2.0, p25=4.0, p50=10.0, p75=36.0, p90=104.0
- any-next-fill cumulative share: <30s=33.4%, <60s=39.0%, <120s=44.2%, <300s=52.0%, none-in-window=47.9%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=8,530: p10=-0.40, p25=0.00, p50=0.38, p75=1.00, p90=3.00
- crossed-market share: 16.8%

**other** — n_fills=44,862, taker=7,073, maker=37,789
- taker_fallback_pct: 46.6%; maker_no-fill_optimistic: 91.8%; maker_no-fill_realistic: 94.3%
- taker lag percentiles (s): p10=2.0, p25=4.0, p50=10.0, p75=36.0, p90=118.0
- any-next-fill cumulative share: <30s=26.3%, <60s=31.7%, <120s=35.9%, <300s=42.1%, none-in-window=57.8%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=3,774: p10=-0.30, p25=0.00, p50=1.00, p75=2.00, p90=5.00
- crossed-market share: 10.3%

**macro** — n_fills=25,172, taker=2,489, maker=22,683
- taker_fallback_pct: 28.4%; maker_no-fill_optimistic: 85.0%; maker_no-fill_realistic: 87.0%
- taker lag percentiles (s): p10=2.0, p25=6.0, p50=14.0, p75=52.0, p90=128.0
- any-next-fill cumulative share: <30s=30.7%, <60s=38.4%, <120s=45.2%, <300s=54.9%, none-in-window=45.1%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=1,781: p10=-0.20, p25=0.00, p50=0.10, p75=0.60, p90=1.50
- crossed-market share: 14.0%

**sports** — n_fills=5,323, taker=658, maker=4,665
- taker_fallback_pct: 57.9%; maker_no-fill_optimistic: 90.2%; maker_no-fill_realistic: 92.5%
- taker lag percentiles (s): p10=2.0, p25=6.0, p50=8.0, p75=32.0, p90=70.0
- any-next-fill cumulative share: <30s=24.8%, <60s=29.2%, <120s=32.7%, <300s=37.6%, none-in-window=62.3%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=277: p10=-0.30, p25=0.00, p50=0.10, p75=1.00, p90=5.34
- crossed-market share: 9.6%

**crypto** — n_fills=817, taker=197, maker=620
- taker_fallback_pct: 60.4%; maker_no-fill_optimistic: 92.3%; maker_no-fill_realistic: 93.2%
- taker lag percentiles (s): p10=2.0, p25=2.0, p50=2.0, p75=6.0, p90=20.0
- any-next-fill cumulative share: <30s=21.9%, <60s=24.1%, <120s=25.1%, <300s=27.7%, none-in-window=72.3%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=78: p10=0.41, p25=0.80, p50=1.00, p75=5.00, p90=35.60
- crossed-market share: 13.0%

**weather** — n_fills=813, taker=167, maker=646
- taker_fallback_pct: 84.4%; maker_no-fill_optimistic: 96.6%; maker_no-fill_realistic: 99.2%
- taker lag percentiles (s): p10=6.0, p25=11.5, p50=16.0, p75=32.5, p90=141.0
- any-next-fill cumulative share: <30s=8.4%, <60s=10.2%, <120s=11.2%, <300s=14.3%, none-in-window=85.7%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=26: p10=0.00, p25=0.20, p50=1.00, p75=1.18, p90=2.50
- crossed-market share: 5.7%

## Sanity checks
- A_total_pnl_match_within_10pct: **PASS**
- B_fill_count_subset_invariants: **PASS**
- C_pnl_monotonicity_warnings: **PASS**
  - leader PnL (replay calc) = 941,123; closed_positions sum = 888,540 (5.9% drift). Difference largely from marking unresolved positions to last fill price; closed_positions does not.

PnL-monotonicity informationals (NOT bugs — see check description):
  - other: A_real PnL (71,341) > A_opt PnL (64,742) — the maker fills filtered by the realistic model were net-losing, so dropping them raised PnL.
  - weather (maker_share=79%): pure-taker B PnL (2,799) > role-mirrored A_real (641) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.

## Interpretation
Domah is **87% maker** on 170,005 fills with leader realised PnL of **$941,123** in this window, so the audit hinges on the maker-leg fill model and adverse selection more than on taker slippage.

**Deployable family today: macro** — A_real capture ≥40%, leader PnL positive, and adverse_select_ratio not destructive.

Adverse selection is destructive for: **politics, sports, crypto, weather** (maker-leg fill rate is significantly higher on losing positions than winning ones). This is the central economic blocker — even free maker rebates don't fix this.

Pure-taker (Branch B) is uniformly worse than role-mirroring across the maker-dominant families — paying 3¢ to cross spread on his entire flow burns capital faster than the leader earns from maker rebates.

**Structurally unstable**: politics, sports have >30pp capture swings between H1 2025 and 2026 — historical capture there does not generalise; only a live A/B test will answer it.