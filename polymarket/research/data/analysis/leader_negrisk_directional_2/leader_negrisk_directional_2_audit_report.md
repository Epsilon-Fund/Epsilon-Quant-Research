---
title: "Copy-Execution Audit: leader_negrisk_directional_2"
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
# Copy-Execution Audit: leader_negrisk_directional_2
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


_Generated 2026-05-18 16:33 UTC_

Read-only diagnostic. Trade data covers 2025-01-02 → 2026-04-24 (last available shard; not today 2026-05-16).

## Universe summary
- **label**: leader_negrisk_directional_2
- **address**: 0x5bffcf561bcae83af680ad600cb99f1184d6ffbe
- **window_start**: 2025-01-02
- **window_end_exclusive**: 2026-04-25
- **n_fills**: 21,449
- **dollar_volume_usd**: 58,715,423.63
- **n_positions**: 500
- **n_resolved_positions**: 466
- **n_families_covered**: 5
- **leader_pnl_calc_usd**: 975,270.23
- **leader_pnl_closed_positions_usd**: 817,521.18

## Primary family table

`A_opt` / `A_real`: role-mirrored, optimistic vs realistic maker-fill model. `B`: pure taker. `C_opt` / `C_real`: pure maker. `adverse_select_ratio` = A_real maker-fill rate on winning vs losing positions (N/A if either bucket <30 maker fills). Deployable cells have **high capture AND adverse_select_ratio close to 1.0**.

| family   | n_fills | n_positions | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_opt_capture | A_real_capture | A_taker_fallback_pct | A_maker_realfill_rate | adverse_select_ratio | adv_sel_n_win | adv_sel_n_lose |
| -------- | ------- | ----------- | ---------- | --------- | ---------- | -------- | --------- | ---------- | ------------- | -------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
|    other |   8,292 |         249 |   -316,170 |    35,330 |   -395,256 | -615,336 |    80,588 |   -587,037 |       -0.1117 |         1.2501 |               0.2859 |                0.2203 |               0.4274 |         3,959 |          2,346 |
| politics |   8,189 |         147 |    727,188 |   441,478 |    305,421 |  336,850 |   382,119 |    226,321 |        0.6071 |         0.4200 |               0.3684 |                0.0907 |               0.2475 |         5,798 |            510 |
|    macro |   3,761 |          61 |    339,806 |   188,400 |    156,074 |  165,364 |   214,056 |    175,816 |        0.5544 |         0.4593 |               0.1798 |                0.1739 |               0.5370 |         2,487 |            451 |
|   crypto |     874 |          31 |    201,780 |   164,699 |    231,017 |  177,518 |   129,606 |    187,843 |        0.8162 |         1.1449 |               0.0918 |                0.4352 |               0.7830 |           421 |             50 |
|   sports |     333 |          12 |     22,666 |   -20,552 |    -21,027 |  -45,302 |     9,405 |      3,153 |       -0.9067 |        -0.9277 |               0.1858 |                0.6727 |               1.1778 |           150 |             70 |

### Secondary slices

#### By market lifecycle phase (hours-to-resolution at fill time)
| slice           | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| --------------- | ----------- | ------- | ---------- | --------- | ---------- | -------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
|            open |         210 |  13,549 |  1,305,355 |   750,483 |    633,003 |  587,682 |   686,209 |    527,948 |               0.3439 |               0.1235 |                0.1041 |               0.4238 |        0.5749 |         0.4849 |
| near-resolution |         207 |   4,445 |   -373,242 |  -422,994 |   -436,399 | -495,696 |  -382,186 |   -388,976 |               0.0977 |               0.3392 |                0.3025 |               0.6935 |        1.1333 |         1.1692 |
|          middle |          83 |   3,455 |     43,157 |   481,865 |     79,626 |  -72,892 |   511,751 |   -132,877 |               0.1103 |               0.2589 |                0.2431 |               0.3185 |       11.1653 |         1.8450 |

#### By hour-of-day (UTC)
| slice | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | ------- | ---------- | --------- | ---------- | -------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| 00-06 |         152 |   4,724 |   -207,877 |  -257,121 |   -210,805 | -402,876 |  -188,356 |   -143,216 |               0.3605 |               0.1981 |                0.1722 |               0.7029 |        1.2369 |         1.0141 |
| 06-12 |         147 |   5,488 |    443,793 |   289,855 |    243,351 |  224,402 |   226,835 |    180,111 |               0.1675 |               0.2342 |                0.2144 |               0.4505 |        0.6531 |         0.5483 |
| 12-18 |         185 |  10,297 |    663,952 |   741,287 |    209,043 |  193,809 |   713,036 |    -85,720 |               0.3162 |               0.1722 |                0.1519 |               0.2457 |        1.1165 |         0.3148 |
| 18-24 |          16 |     940 |     75,403 |    35,332 |     34,641 |    3,759 |    64,258 |     54,919 |               0.1696 |               0.2642 |                0.2000 |               1.1090 |        0.4686 |         0.4594 |

#### By leader's role on the originating fill
| slice | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | ------- | ---------- | --------- | ---------- | -------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| maker |         263 |  11,700 |    -82,141 |   281,760 |   -216,914 | -538,835 |   343,829 |   -392,747 |               0.2456 |               0.2332 |                0.2031 |               0.6064 |       -3.4302 |         2.6407 |
| taker |         237 |   9,749 |  1,057,411 |   527,594 |    493,144 |  557,929 |   471,944 |    398,842 |               0.3033 |               0.1418 |                0.1301 |               0.1117 |        0.4989 |         0.4664 |

### Sensitivity: H1 2025 vs 2026-YTD

Flag families with `|H1 cap − 2026 cap| > 30pp` AND `n_fills > 200` as 'structure-shifting': historical capture does not generalise.

| family   | A_opt_capture_2025-H1 | A_opt_capture_2026-YTD | A_real_capture_2025-H1 | A_real_capture_2026-YTD | n_fills_2025-H1 | n_fills_2026-YTD |
| -------- | --------------------- | ---------------------- | ---------------------- | ----------------------- | --------------- | ---------------- |
|   crypto |                0.8198 |                        |                 0.8255 |                         |              98 |                  |
|    macro |                0.4503 |                 0.4181 |                 0.1695 |                  0.4155 |             178 |            1,618 |
|    other |                0.2893 |                 0.0390 |                 0.1719 |                  0.7972 |           2,649 |            4,035 |
| politics |                0.5471 |                 0.9965 |                 0.1965 |                  0.9165 |           2,131 |            1,589 |
|   sports |                0.0000 |      -178,623,648.9615 |                 0.0000 |       -178,623,648.9615 |               1 |              102 |

### Sensitivity: fallback cents (taker-leg only)

| family   | A_real_pnl_1c | A_real_pnl_2c | A_real_pnl_3c | A_real_pnl_5c | B_pnl_1c | B_pnl_2c | B_pnl_3c | B_pnl_5c |
| -------- | ------------- | ------------- | ------------- | ------------- | -------- | -------- | -------- | -------- |
|   crypto |       231,997 |       231,507 |       231,017 |       230,038 |  190,637 |  184,078 |  177,518 |  164,400 |
|    macro |       175,882 |       165,978 |       156,074 |       136,267 |  270,579 |  217,971 |  165,364 |   60,148 |
|    other |      -359,049 |      -377,152 |      -395,256 |      -431,462 | -418,113 | -516,725 | -615,336 | -812,559 |
| politics |       378,054 |       341,737 |       305,421 |       232,787 |  577,274 |  457,062 |  336,850 |   96,426 |
|   sports |        -1,116 |       -11,071 |       -21,027 |       -40,937 |     -425 |  -22,863 |  -45,302 |  -90,178 |

### Per-family diagnostics

**other** — n_fills=8,292, taker=1,987, maker=6,305
- taker_fallback_pct: 28.6%; maker_no-fill_optimistic: 75.7%; maker_no-fill_realistic: 78.0%
- taker lag percentiles (s): p10=2.0, p25=2.0, p50=8.0, p75=52.0, p90=162.0
- any-next-fill cumulative share: <30s=39.2%, <60s=44.7%, <120s=51.3%, <300s=61.6%, none-in-window=38.3%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=1,419: p10=-0.64, p25=0.00, p50=0.00, p75=0.20, p90=1.00
- crossed-market share: 24.0%

**politics** — n_fills=8,189, taker=1,881, maker=6,308
- taker_fallback_pct: 36.8%; maker_no-fill_optimistic: 88.8%; maker_no-fill_realistic: 90.9%
- taker lag percentiles (s): p10=2.0, p25=4.0, p50=14.0, p75=44.0, p90=138.6
- any-next-fill cumulative share: <30s=26.9%, <60s=35.5%, <120s=43.5%, <300s=53.9%, none-in-window=46.0%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=1,188: p10=-1.00, p25=0.00, p50=0.00, p75=0.40, p90=1.00
- crossed-market share: 25.7%

**macro** — n_fills=3,761, taker=823, maker=2,938
- taker_fallback_pct: 18.0%; maker_no-fill_optimistic: 80.4%; maker_no-fill_realistic: 82.6%
- taker lag percentiles (s): p10=4.0, p25=6.0, p50=12.0, p75=41.0, p90=176.0
- any-next-fill cumulative share: <30s=43.3%, <60s=54.2%, <120s=68.0%, <300s=84.7%, none-in-window=15.2%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=675: p10=-0.10, p25=-0.10, p50=0.00, p75=0.10, p90=1.00
- crossed-market share: 24.2%

**crypto** — n_fills=874, taker=403, maker=471
- taker_fallback_pct: 9.2%; maker_no-fill_optimistic: 51.6%; maker_no-fill_realistic: 56.5%
- taker lag percentiles (s): p10=3.0, p25=8.0, p50=18.0, p75=54.0, p90=98.0
- any-next-fill cumulative share: <30s=59.5%, <60s=73.0%, <120s=80.1%, <300s=86.5%, none-in-window=13.4%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=366: p10=-1.00, p25=-0.10, p50=0.00, p75=1.00, p90=1.00
- crossed-market share: 44.2%

**sports** — n_fills=333, taker=113, maker=220
- taker_fallback_pct: 18.6%; maker_no-fill_optimistic: 31.8%; maker_no-fill_realistic: 32.7%
- taker lag percentiles (s): p10=6.0, p25=6.0, p50=6.0, p75=8.5, p90=18.0
- any-next-fill cumulative share: <30s=70.0%, <60s=81.1%, <120s=84.7%, <300s=87.7%, none-in-window=12.3%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=92: p10=0.00, p25=0.10, p50=0.10, p75=0.10, p90=0.10
- crossed-market share: 49.8%

## Sanity checks
- A_total_pnl_match_within_10pct: **FAIL**
- B_fill_count_subset_invariants: **PASS**
- C_pnl_monotonicity_warnings: **PASS**
  - leader PnL (replay calc) = 975,270; closed_positions sum = 817,521 (19.3% drift). Difference largely from marking unresolved positions to last fill price; closed_positions does not.

PnL-monotonicity informationals (NOT bugs — see check description):
  - crypto: A_real PnL (231,017) > A_opt PnL (164,699) — the maker fills filtered by the realistic model were net-losing, so dropping them raised PnL.
  - crypto: C_real PnL > C_opt PnL — same dynamic on the pure-maker branch.
  - politics (maker_share=77%): pure-taker B PnL (336,850) > role-mirrored A_real (305,421) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.
  - macro (maker_share=78%): pure-taker B PnL (165,364) > role-mirrored A_real (156,074) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.

## Interpretation
Leader `leader_negrisk_directional_2` is **76% maker** on 21,449 fills with leader realised PnL of **$975,270** in this window, so the audit hinges on the maker-leg fill model and adverse selection more than on taker slippage.

**No family clears the deployable bar** (A_real capture ≥40% on positive leader PnL AND adverse_select_ratio ≥0.85). Closest call: **crypto** with A_real capture 114%, leader PnL $201,780, adv-sel 0.78.

Adverse selection is destructive for: **other, politics, macro** (maker-leg fill rate is significantly higher on losing positions than winning ones). This is the central economic blocker — even free maker rebates don't fix this.

Pure-taker (Branch B) is uniformly worse than role-mirroring across the maker-dominant families — paying 3¢ to cross spread on his entire flow burns capital faster than the leader earns from maker rebates.

**Structurally unstable**: other, politics have >30pp capture swings between H1 2025 and 2026 — historical capture there does not generalise; only a live A/B test will answer it.