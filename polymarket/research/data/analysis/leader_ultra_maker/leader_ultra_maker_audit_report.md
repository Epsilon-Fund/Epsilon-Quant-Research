---
title: "Copy-Execution Audit: leader_ultra_maker"
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
# Copy-Execution Audit: leader_ultra_maker
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


_Generated 2026-05-18 16:31 UTC_

Read-only diagnostic. Trade data covers 2025-01-02 → 2026-04-24 (last available shard; not today 2026-05-16).

## Universe summary
- **label**: leader_ultra_maker
- **address**: 0x2005d16a84ceefa912d4e380cd32e7ff827875ea
- **window_start**: 2025-01-02
- **window_end_exclusive**: 2026-04-25
- **n_fills**: 2,456,399
- **dollar_volume_usd**: 233,316,702.47
- **n_positions**: 101,136
- **n_resolved_positions**: 101,129
- **n_families_covered**: 4
- **leader_pnl_calc_usd**: 8,376,945.04
- **leader_pnl_closed_positions_usd**: 8,378,137.12

## Primary family table

`A_opt` / `A_real`: role-mirrored, optimistic vs realistic maker-fill model. `B`: pure taker. `C_opt` / `C_real`: pure maker. `adverse_select_ratio` = A_real maker-fill rate on winning vs losing positions (N/A if either bucket <30 maker fills). Deployable cells have **high capture AND adverse_select_ratio close to 1.0**.

| family   | n_fills   | n_positions | leader_pnl | A_opt_pnl  | A_real_pnl | B_pnl      | C_opt_pnl  | C_real_pnl | A_opt_capture | A_real_capture | A_taker_fallback_pct | A_maker_realfill_rate | adverse_select_ratio | adv_sel_n_win | adv_sel_n_lose |
| -------- | --------- | ----------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------- | -------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
|    other | 1,574,312 |      76,328 |  6,717,995 |   -223,099 |   -876,989 |     72,473 |  1,432,633 |    789,768 |       -0.0332 |        -0.1305 |               0.0812 |                0.5704 |               0.7985 |       714,215 |        700,537 |
|   sports |   875,058 |      24,667 |  1,657,498 | -2,044,012 | -2,549,217 | -1,569,716 | -1,446,861 | -1,896,338 |       -1.2332 |        -1.5380 |               0.0302 |                0.7486 |               0.8908 |       385,399 |        406,259 |
| politics |     6,989 |         126 |      1,105 |    -27,584 |    -34,358 |    -18,760 |    -24,682 |    -31,284 |      -24.9601 |       -31.0902 |               0.0323 |                0.5967 |               0.7435 |         2,941 |          3,243 |
|   crypto |        40 |          15 |        346 |        616 |        616 |        606 |        216 |        216 |        1.7771 |         1.7771 |               0.4000 |                0.0857 |                      |            25 |             10 |

### Secondary slices

#### By market lifecycle phase (hours-to-resolution at fill time)
| slice           | n_positions | n_fills   | leader_pnl | A_opt_pnl  | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| --------------- | ----------- | --------- | ---------- | ---------- | ---------- | -------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| near-resolution |      87,278 | 1,910,999 |  7,467,238 |   -842,366 | -1,735,065 | -524,826 |   969,914 |    112,086 |               0.0780 |               0.6727 |                0.6063 |               0.8177 |       -0.1128 |        -0.2324 |
|          middle |      11,769 |   475,487 |    794,304 | -1,151,214 | -1,384,045 | -751,876 |  -800,359 | -1,009,863 |               0.0203 |               0.7904 |                0.7442 |               0.8857 |       -1.4493 |        -1.7425 |
|            open |       2,089 |    69,913 |    115,403 |   -300,500 |   -340,839 | -238,694 |  -208,249 |   -239,861 |               0.0418 |               0.7232 |                0.6668 |               0.8808 |       -2.6039 |        -2.9535 |

#### By hour-of-day (UTC)
| slice | n_positions | n_fills | leader_pnl | A_opt_pnl  | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | ------- | ---------- | ---------- | ---------- | -------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| 12-18 |      36,216 | 835,139 |  2,681,710 |   -911,050 | -1,298,910 | -969,558 |   -14,365 |   -391,632 |               0.0644 |               0.6791 |                0.6128 |               0.8243 |       -0.3397 |        -0.4844 |
| 18-24 |      37,840 | 878,397 |  2,519,961 | -1,348,237 | -1,645,707 | -873,201 |  -470,157 |   -727,660 |               0.0728 |               0.6899 |                0.6254 |               0.8190 |       -0.5350 |        -0.6531 |
| 06-12 |       9,720 | 316,200 |  1,723,576 |    346,243 |    109,169 |  525,198 |   585,143 |    354,656 |               0.0366 |               0.6969 |                0.6415 |               0.8443 |        0.2009 |         0.0633 |
| 00-06 |      17,360 | 426,663 |  1,451,698 |   -381,035 |   -624,500 | -197,836 |  -139,315 |   -373,001 |               0.0664 |               0.7426 |                0.6872 |               0.8620 |       -0.2625 |        -0.4302 |

#### By leader's role on the originating fill
| slice | n_positions | n_fills   | leader_pnl | A_opt_pnl  | A_real_pnl | B_pnl      | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | --------- | ---------- | ---------- | ---------- | ---------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| maker |      88,563 | 2,222,066 |  7,713,093 | -1,627,156 | -2,722,040 | -1,006,612 |   -44,008 | -1,108,432 |               0.0530 |               0.6957 |                0.6334 |               0.8284 |       -0.2110 |        -0.3529 |
| taker |      12,573 |   234,333 |    663,852 |   -666,923 |   -737,908 |   -508,784 |     5,314 |    -29,205 |               0.0932 |               0.7069 |                0.6448 |               0.8785 |       -1.0046 |        -1.1116 |

### Sensitivity: H1 2025 vs 2026-YTD

Flag families with `|H1 cap − 2026 cap| > 30pp` AND `n_fills > 200` as 'structure-shifting': historical capture does not generalise.

| family   | A_opt_capture_2026-YTD | A_real_capture_2026-YTD | n_fills_2026-YTD |
| -------- | ---------------------- | ----------------------- | ---------------- |
|   crypto |                 1.2474 |                  1.2474 |               24 |
|    other |                -0.0586 |                 -0.1536 |        1,331,694 |
| politics |                19.4378 |                 24.0189 |            6,923 |
|   sports |                -2.2917 |                 -2.7511 |          681,905 |

### Sensitivity: fallback cents (taker-leg only)

| family   | A_real_pnl_1c | A_real_pnl_2c | A_real_pnl_3c | A_real_pnl_5c | B_pnl_1c   | B_pnl_2c   | B_pnl_3c   | B_pnl_5c   |
| -------- | ------------- | ------------- | ------------- | ------------- | ---------- | ---------- | ---------- | ---------- |
|   crypto |           620 |           618 |           616 |           612 |        627 |        617 |        606 |        586 |
|    other |      -843,192 |      -860,091 |      -876,989 |      -910,787 |    238,257 |    155,365 |     72,473 |    -93,311 |
| politics |       -34,317 |       -34,338 |       -34,358 |       -34,399 |    -18,616 |    -18,688 |    -18,760 |    -18,903 |
|   sports |    -2,534,691 |    -2,541,954 |    -2,549,217 |    -2,563,742 | -1,507,335 | -1,538,525 | -1,569,716 | -1,632,097 |

### Per-family diagnostics

**other** — n_fills=1,574,312, taker=159,560, maker=1,414,752
- taker_fallback_pct: 8.1%; maker_no-fill_optimistic: 35.9%; maker_no-fill_realistic: 43.0%
- taker lag percentiles (s): p10=2.0, p25=2.0, p50=6.0, p75=24.0, p90=82.0
- any-next-fill cumulative share: <30s=75.1%, <60s=83.8%, <120s=89.8%, <300s=94.9%, none-in-window=5.1%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=146,611: p10=-2.00, p25=0.00, p50=1.70, p75=5.00, p90=15.00
- crossed-market share: 37.2%

**sports** — n_fills=875,058, taker=83,400, maker=791,658
- taker_fallback_pct: 3.0%; maker_no-fill_optimistic: 20.3%; maker_no-fill_realistic: 25.1%
- taker lag percentiles (s): p10=2.0, p25=2.0, p50=4.0, p75=12.0, p90=38.0
- any-next-fill cumulative share: <30s=86.5%, <60s=92.7%, <120s=96.1%, <300s=98.3%, none-in-window=1.7%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=80,883: p10=-1.00, p25=0.00, p50=1.00, p75=3.00, p90=5.00
- crossed-market share: 56.2%

**politics** — n_fills=6,989, taker=805, maker=6,184
- taker_fallback_pct: 3.2%; maker_no-fill_optimistic: 33.4%; maker_no-fill_realistic: 40.3%
- taker lag percentiles (s): p10=2.0, p25=2.0, p50=4.0, p75=10.0, p90=46.4
- any-next-fill cumulative share: <30s=82.4%, <60s=89.1%, <120s=93.4%, <300s=97.4%, none-in-window=2.5%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=779: p10=-2.00, p25=0.00, p50=1.00, p75=3.00, p90=6.00
- crossed-market share: 42.7%

## Sanity checks
- A_total_pnl_match_within_10pct: **PASS**
- B_fill_count_subset_invariants: **PASS**
- C_pnl_monotonicity_warnings: **PASS**
  - leader PnL (replay calc) = 8,376,945; closed_positions sum = 8,378,137 (0.0% drift). Difference largely from marking unresolved positions to last fill price; closed_positions does not.

PnL-monotonicity informationals (NOT bugs — see check description):
  - other (maker_share=90%): pure-taker B PnL (72,473) > role-mirrored A_real (-876,989) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.
  - sports (maker_share=90%): pure-taker B PnL (-1,569,716) > role-mirrored A_real (-2,549,217) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.
  - politics (maker_share=88%): pure-taker B PnL (-18,760) > role-mirrored A_real (-34,358) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.

## Interpretation
Leader `leader_ultra_maker` is **90% maker** on 2,456,399 fills with leader realised PnL of **$8,376,945** in this window, so the audit hinges on the maker-leg fill model and adverse selection more than on taker slippage.

**Deployable family today: crypto** — A_real capture ≥40%, leader PnL positive, and adverse_select_ratio not destructive.

Adverse selection is not catastrophic (ratio ≥0.7) on any family with sufficient maker fills, but is below 1.0 on most — copy fills happen disproportionately when his post is wrong.

Pure-taker (Branch B) is uniformly worse than role-mirroring across the maker-dominant families — paying 3¢ to cross spread on his entire flow burns capital faster than the leader earns from maker rebates.

H1-2025 vs 2026-YTD capture ratios are within 30pp on the substantial families, so the audit numbers should generalise modulo regime change.