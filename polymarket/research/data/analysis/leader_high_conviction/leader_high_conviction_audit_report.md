---
title: "Copy-Execution Audit: leader_high_conviction"
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
# Copy-Execution Audit: leader_high_conviction
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


_Generated 2026-05-18 16:27 UTC_

Read-only diagnostic. Trade data covers 2025-01-02 → 2026-04-24 (last available shard; not today 2026-05-16).

## Universe summary
- **label**: leader_high_conviction
- **address**: 0x204f72f35326db932158cba6adff0b9a1da95e14
- **window_start**: 2025-01-02
- **window_end_exclusive**: 2026-04-25
- **n_fills**: 5,127,237
- **dollar_volume_usd**: 565,945,336.98
- **n_positions**: 167,238
- **n_resolved_positions**: 167,135
- **n_families_covered**: 4
- **leader_pnl_calc_usd**: 10,637,398.70
- **leader_pnl_closed_positions_usd**: 10,636,907.14

## Primary family table

`A_opt` / `A_real`: role-mirrored, optimistic vs realistic maker-fill model. `B`: pure taker. `C_opt` / `C_real`: pure maker. `adverse_select_ratio` = A_real maker-fill rate on winning vs losing positions (N/A if either bucket <30 maker fills). Deployable cells have **high capture AND adverse_select_ratio close to 1.0**.

| family   | n_fills   | n_positions | leader_pnl | A_opt_pnl  | A_real_pnl | B_pnl     | C_opt_pnl | C_real_pnl | A_opt_capture | A_real_capture | A_taker_fallback_pct | A_maker_realfill_rate | adverse_select_ratio | adv_sel_n_win | adv_sel_n_lose |
| -------- | --------- | ----------- | ---------- | ---------- | ---------- | --------- | --------- | ---------- | ------------- | -------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
|    other | 3,491,032 |     128,479 |  8,996,991 |  4,190,514 |  3,107,420 | 7,075,520 | 5,617,262 |  4,798,950 |        0.4658 |         0.3454 |               0.3075 |                0.3471 |               0.8465 |     1,101,294 |        968,909 |
|   sports | 1,633,854 |      38,449 |  1,647,741 | -1,919,087 | -2,482,693 |  -341,241 |   -25,047 |   -293,467 |       -1.1647 |        -1.5067 |               0.1235 |                0.5581 |               0.9226 |       443,283 |        401,614 |
| politics |     1,276 |          76 |     -8,110 |     -6,846 |     -4,527 |    -7,420 |    -4,387 |     -1,132 |        0.8441 |         0.5581 |               0.3813 |                0.3004 |               0.9876 |           265 |            434 |
|   crypto |     1,075 |         234 |        777 |       -130 |       -650 |       289 |      -326 |     -1,170 |       -0.1675 |        -0.8360 |               0.7070 |                0.2130 |               0.3090 |           433 |            229 |

### Secondary slices

#### By market lifecycle phase (hours-to-resolution at fill time)
| slice           | n_positions | n_fills   | leader_pnl | A_opt_pnl  | A_real_pnl | B_pnl     | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| --------------- | ----------- | --------- | ---------- | ---------- | ---------- | --------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
|            open |       7,022 |   364,845 |  1,823,144 |    886,151 |    771,096 | 1,290,431 | 1,307,373 |  1,201,882 |               0.1108 |               0.6172 |                0.5298 |               0.9102 |        0.4861 |         0.4229 |
|          middle |      42,499 | 2,176,290 |  6,192,569 |  2,531,198 |  1,557,198 | 4,131,617 | 3,812,298 |  3,405,623 |               0.2423 |               0.5089 |                0.4189 |               0.8830 |        0.4087 |         0.2515 |
| near-resolution |     117,717 | 2,586,102 |  2,621,686 | -1,152,898 | -1,708,743 | 1,305,100 |   467,830 |   -104,325 |               0.2616 |               0.4861 |                0.3829 |               0.8540 |       -0.4398 |        -0.6518 |

#### By hour-of-day (UTC)
| slice | n_positions | n_fills   | leader_pnl | A_opt_pnl  | A_real_pnl | B_pnl     | C_opt_pnl  | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | --------- | ---------- | ---------- | ---------- | --------- | ---------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| 00-06 |      36,532 | 1,194,407 |    124,813 | -1,899,583 | -2,042,229 |  -845,022 | -1,007,328 |   -927,854 |               0.2076 |               0.5437 |                0.4486 |               0.8702 |      -15.2195 |       -16.3624 |
| 18-24 |      52,466 | 1,516,182 |  3,712,663 |  1,179,129 |    480,527 | 2,500,017 |  2,222,950 |  1,629,539 |               0.2411 |               0.5049 |                0.4082 |               0.8762 |        0.3176 |         0.1294 |
| 12-18 |      47,498 | 1,278,777 |  4,074,471 |  1,990,318 |  1,706,756 | 3,199,769 |  2,715,143 |  2,693,499 |               0.2561 |               0.4911 |                0.3927 |               0.8714 |        0.4885 |         0.4189 |
| 06-12 |      30,742 | 1,137,871 |  2,725,453 |    994,586 |    474,496 | 1,872,383 |  1,656,737 |  1,107,996 |               0.2654 |               0.4804 |                0.3848 |               0.8764 |        0.3649 |         0.1741 |

#### By leader's role on the originating fill
| slice | n_positions | n_fills   | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl     | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | --------- | ---------- | --------- | ---------- | --------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| taker |      67,944 | 2,082,268 |  4,903,260 | 1,526,652 |    845,974 | 3,139,873 | 3,161,759 |  2,801,483 |               0.2747 |               0.5598 |                0.4590 |               0.9310 |        0.3114 |         0.1725 |
| maker |      99,294 | 3,044,969 |  5,734,139 |   737,798 |   -226,424 | 3,587,274 | 2,425,743 |  1,701,698 |               0.2087 |               0.4773 |                0.3829 |               0.8448 |        0.1287 |        -0.0395 |

### Sensitivity: H1 2025 vs 2026-YTD

Flag families with `|H1 cap − 2026 cap| > 30pp` AND `n_fills > 200` as 'structure-shifting': historical capture does not generalise.

| family   | A_opt_capture_2026-YTD | A_real_capture_2026-YTD | n_fills_2026-YTD |
| -------- | ---------------------- | ----------------------- | ---------------- |
|   crypto |                 0.8451 |                  1.4007 |            1,055 |
|    other |                 0.4978 |                  0.3666 |        2,700,921 |
| politics |                 1.0547 |                  0.6278 |              818 |
|   sports |                13.7351 |                 16.3787 |        1,185,457 |

### Sensitivity: fallback cents (taker-leg only)

| family   | A_real_pnl_1c | A_real_pnl_2c | A_real_pnl_3c | A_real_pnl_5c | B_pnl_1c  | B_pnl_2c  | B_pnl_3c  | B_pnl_5c   |
| -------- | ------------- | ------------- | ------------- | ------------- | --------- | --------- | --------- | ---------- |
|   crypto |          -436 |          -543 |          -650 |          -864 |       871 |       580 |       289 |       -294 |
|    other |     4,219,242 |     3,663,331 |     3,107,420 |     1,995,598 | 9,947,047 | 8,511,283 | 7,075,520 |  4,203,994 |
| politics |        -4,274 |        -4,400 |        -4,527 |        -4,780 |    -6,912 |    -7,166 |    -7,420 |     -7,929 |
|   sports |    -2,202,752 |    -2,342,722 |    -2,482,693 |    -2,762,634 |   378,723 |    18,741 |  -341,241 | -1,061,204 |

### Per-family diagnostics

**other** — n_fills=3,491,032, taker=1,420,829, maker=2,070,203
- taker_fallback_pct: 30.8%; maker_no-fill_optimistic: 55.4%; maker_no-fill_realistic: 65.3%
- taker lag percentiles (s): p10=2.0, p25=4.0, p50=16.0, p75=60.0, p90=154.0
- any-next-fill cumulative share: <30s=38.2%, <60s=47.7%, <120s=56.5%, <300s=67.2%, none-in-window=32.7%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=983,914: p10=-1.00, p25=-1.00, p50=0.00, p75=1.00, p90=2.00
- crossed-market share: 17.8%

**sports** — n_fills=1,633,854, taker=788,957, maker=844,897
- taker_fallback_pct: 12.3%; maker_no-fill_optimistic: 35.2%; maker_no-fill_realistic: 44.2%
- taker lag percentiles (s): p10=2.0, p25=4.0, p50=10.0, p75=34.0, p90=104.0
- any-next-fill cumulative share: <30s=57.9%, <60s=68.6%, <120s=77.0%, <300s=85.6%, none-in-window=14.4%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=691,522: p10=-2.00, p25=-1.00, p50=0.00, p75=2.00, p90=4.00
- crossed-market share: 39.8%

**politics** — n_fills=1,276, taker=577, maker=699
- taker_fallback_pct: 38.1%; maker_no-fill_optimistic: 60.7%; maker_no-fill_realistic: 70.0%
- taker lag percentiles (s): p10=4.0, p25=8.0, p50=24.0, p75=74.0, p90=150.8
- any-next-fill cumulative share: <30s=36.8%, <60s=47.6%, <120s=55.6%, <300s=66.1%, none-in-window=33.8%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=357: p10=-5.00, p25=-2.00, p50=0.00, p75=1.00, p90=3.40
- crossed-market share: 18.2%

**crypto** — n_fills=1,075, taker=413, maker=662
- taker_fallback_pct: 70.7%; maker_no-fill_optimistic: 73.9%; maker_no-fill_realistic: 78.7%
- taker lag percentiles (s): p10=2.0, p25=6.0, p50=26.0, p75=76.0, p90=242.0
- any-next-fill cumulative share: <30s=16.3%, <60s=24.9%, <120s=28.7%, <300s=33.1%, none-in-window=66.8%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=121: p10=-1.12, p25=0.00, p50=1.00, p75=3.00, p90=6.00
- crossed-market share: 13.0%

## Sanity checks
- A_total_pnl_match_within_10pct: **PASS**
- B_fill_count_subset_invariants: **PASS**
- C_pnl_monotonicity_warnings: **PASS**
  - leader PnL (replay calc) = 10,637,399; closed_positions sum = 10,636,907 (0.0% drift). Difference largely from marking unresolved positions to last fill price; closed_positions does not.

PnL-monotonicity informationals (NOT bugs — see check description):
  - politics: A_real PnL (-4,527) > A_opt PnL (-6,846) — the maker fills filtered by the realistic model were net-losing, so dropping them raised PnL.
  - politics: C_real PnL > C_opt PnL — same dynamic on the pure-maker branch.
  - other (maker_share=59%): pure-taker B PnL (7,075,520) > role-mirrored A_real (3,107,420) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.
  - sports (maker_share=52%): pure-taker B PnL (-341,241) > role-mirrored A_real (-2,482,693) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.
  - crypto (maker_share=62%): pure-taker B PnL (289) > role-mirrored A_real (-650) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.

## Interpretation
Leader `leader_high_conviction` is **57% maker** on 5,127,237 fills with leader realised PnL of **$10,637,399** in this window, so the audit hinges on the maker-leg fill model and adverse selection more than on taker slippage.

**No family clears the deployable bar** (A_real capture ≥40% on positive leader PnL AND adverse_select_ratio ≥0.85). Closest call: **other** with A_real capture 35%, leader PnL $8,996,991, adv-sel 0.85.

Adverse selection is destructive for: **crypto** (maker-leg fill rate is significantly higher on losing positions than winning ones). This is the central economic blocker — even free maker rebates don't fix this.

Pure-taker (Branch B) is uniformly worse than role-mirroring across the maker-dominant families — paying 3¢ to cross spread on his entire flow burns capital faster than the leader earns from maker rebates.

H1-2025 vs 2026-YTD capture ratios are within 30pp on the substantial families, so the audit numbers should generalise modulo regime change.