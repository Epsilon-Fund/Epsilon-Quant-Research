---
title: "Copy-Execution Audit: leader_top_leaderboard"
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
# Copy-Execution Audit: leader_top_leaderboard
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


_Generated 2026-05-18 16:21 UTC_

Read-only diagnostic. Trade data covers 2025-01-02 → 2026-04-24 (last available shard; not today 2026-05-16).

## Universe summary
- **label**: leader_top_leaderboard
- **address**: 0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee
- **window_start**: 2025-01-02
- **window_end_exclusive**: 2026-04-25
- **n_fills**: 184,268
- **dollar_volume_usd**: 171,609,696.75
- **n_positions**: 4,365
- **n_resolved_positions**: 4,359
- **n_families_covered**: 3
- **leader_pnl_calc_usd**: 14,951,978.39
- **leader_pnl_closed_positions_usd**: 14,952,586.45

## Primary family table

`A_opt` / `A_real`: role-mirrored, optimistic vs realistic maker-fill model. `B`: pure taker. `C_opt` / `C_real`: pure maker. `adverse_select_ratio` = A_real maker-fill rate on winning vs losing positions (N/A if either bucket <30 maker fills). Deployable cells have **high capture AND adverse_select_ratio close to 1.0**.

| family   | n_fills | n_positions | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl      | C_opt_pnl | C_real_pnl | A_opt_capture | A_real_capture | A_taker_fallback_pct | A_maker_realfill_rate | adverse_select_ratio | adv_sel_n_win | adv_sel_n_lose |
| -------- | ------- | ----------- | ---------- | --------- | ---------- | ---------- | --------- | ---------- | ------------- | -------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
|   sports | 180,861 |       4,062 | 11,307,668 | 7,087,797 |  7,056,737 | 10,034,043 | 6,970,457 |  7,137,992 |        0.6268 |         0.6241 |               0.0411 |                0.4731 |               0.6992 |        87,875 |         63,538 |
|    other |   2,977 |         297 |  3,710,078 |   878,428 |  1,086,825 |  3,002,308 |   974,281 |  1,126,601 |        0.2368 |         0.2929 |               0.0175 |                0.2850 |               0.5105 |         1,472 |          1,047 |
| politics |     430 |           6 |    -65,768 |   -44,438 |    -41,980 |    -71,104 |   -44,570 |    -41,898 |        0.6757 |         0.6383 |               0.0000 |                0.2610 |                      |             3 |            384 |

### Secondary slices

#### By market lifecycle phase (hours-to-resolution at fill time)
| slice           | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl      | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| --------------- | ----------- | ------- | ---------- | --------- | ---------- | ---------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
|          middle |         349 |  14,753 |   -394,837 |  -374,081 |   -328,374 |   -404,226 |  -352,205 |   -284,957 |               0.0364 |               0.7127 |                0.6545 |               0.8099 |        0.9474 |         0.8317 |
| near-resolution |       3,892 | 151,253 | 13,423,501 | 6,714,898 |  7,021,016 | 11,485,868 | 6,739,074 |  6,774,052 |               0.0407 |               0.5298 |                0.4916 |               0.7822 |        0.5002 |         0.5230 |
|            open |         124 |  18,262 |  1,923,315 | 1,580,970 |  1,408,939 |  1,883,605 | 1,513,300 |  1,733,600 |               0.0468 |               0.1945 |                0.1723 |               0.2541 |        0.8220 |         0.7326 |

#### By hour-of-day (UTC)
| slice | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl     | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | ------- | ---------- | --------- | ---------- | --------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| 18-24 |       1,631 |  57,479 |  9,069,877 | 4,528,891 |  5,175,506 | 8,182,517 | 3,895,510 |  4,499,066 |               0.0363 |               0.5912 |                0.5507 |               0.8432 |        0.4993 |         0.5706 |
| 00-06 |       2,027 |  78,747 |    915,744 |  -529,379 |   -901,429 |   194,996 |   -70,763 |   -547,547 |               0.0370 |               0.5718 |                0.5298 |               0.7347 |       -0.5781 |        -0.9844 |
| 12-18 |         650 |  42,222 |  4,234,763 | 3,531,331 |  3,427,557 | 3,911,541 | 3,689,865 |  3,887,363 |               0.0499 |               0.3125 |                0.2825 |               0.4644 |        0.8339 |         0.8094 |
| 06-12 |          57 |   5,820 |    731,595 |   390,945 |    399,948 |   676,193 |   385,557 |    383,814 |               0.0754 |               0.2369 |                0.2161 |               0.8584 |        0.5344 |         0.5467 |

#### By leader's role on the originating fill
| slice | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl     | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | ------- | ---------- | --------- | ---------- | --------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| maker |       2,938 | 129,337 |  8,398,424 | 2,747,943 |  3,125,101 | 6,569,100 | 2,925,068 |  3,253,639 |               0.0563 |               0.5101 |                0.4719 |               0.6723 |        0.3272 |         0.3721 |
| taker |       1,427 |  54,931 |  6,553,555 | 5,173,845 |  4,976,480 | 6,396,147 | 4,975,100 |  4,969,056 |               0.0316 |               0.4997 |                0.4618 |               0.7976 |        0.7895 |         0.7594 |

### Sensitivity: H1 2025 vs 2026-YTD

Flag families with `|H1 cap − 2026 cap| > 30pp` AND `n_fills > 200` as 'structure-shifting': historical capture does not generalise.

| family   | A_opt_capture_2025-H1 | A_opt_capture_2026-YTD | A_real_capture_2025-H1 | A_real_capture_2026-YTD | n_fills_2025-H1 | n_fills_2026-YTD |
| -------- | --------------------- | ---------------------- | ---------------------- | ----------------------- | --------------- | ---------------- |
|    other |                       |                 0.2048 |                        |                  0.2619 |                 |            2,795 |
| politics |                       |                 2.1122 |                        |                  0.3547 |                 |               15 |
|   sports |                0.1222 |                 0.6193 |                 0.0416 |                  0.6717 |             175 |           98,843 |

### Sensitivity: fallback cents (taker-leg only)

| family   | A_real_pnl_1c | A_real_pnl_2c | A_real_pnl_3c | A_real_pnl_5c | B_pnl_1c   | B_pnl_2c   | B_pnl_3c   | B_pnl_5c  |
| -------- | ------------- | ------------- | ------------- | ------------- | ---------- | ---------- | ---------- | --------- |
|    other |     1,087,974 |     1,087,399 |     1,086,825 |     1,085,676 |  3,003,775 |  3,003,042 |  3,002,308 | 3,000,842 |
| politics |       -41,980 |       -41,980 |       -41,980 |       -41,980 |    -67,739 |    -69,421 |    -71,104 |   -74,470 |
|   sports |     7,091,888 |     7,074,312 |     7,056,737 |     7,021,586 | 10,263,139 | 10,148,591 | 10,034,043 | 9,804,947 |

### Per-family diagnostics

**sports** — n_fills=180,861, taker=29,448, maker=151,413
- taker_fallback_pct: 4.1%; maker_no-fill_optimistic: 48.9%; maker_no-fill_realistic: 52.7%
- taker lag percentiles (s): p10=2.0, p25=2.0, p50=6.0, p75=18.0, p90=62.0
- any-next-fill cumulative share: <30s=74.2%, <60s=82.7%, <120s=88.8%, <300s=94.8%, none-in-window=5.2%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=28,238: p10=-1.00, p25=0.00, p50=0.00, p75=1.00, p90=1.00
- crossed-market share: 35.7%

**other** — n_fills=2,977, taker=458, maker=2,519
- taker_fallback_pct: 1.7%; maker_no-fill_optimistic: 68.7%; maker_no-fill_realistic: 71.5%
- taker lag percentiles (s): p10=2.0, p25=2.0, p50=4.0, p75=6.0, p90=8.0
- any-next-fill cumulative share: <30s=97.3%, <60s=98.2%, <120s=99.2%, <300s=99.6%, none-in-window=0.4%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=450: p10=-2.00, p25=-0.00, p50=0.00, p75=1.00, p90=1.00
- crossed-market share: 22.7%

**politics** — n_fills=430, taker=43, maker=387
- taker_fallback_pct: 0.0%; maker_no-fill_optimistic: 68.2%; maker_no-fill_realistic: 73.9%
- taker lag percentiles (s): p10=2.0, p25=3.0, p50=4.0, p75=21.0, p90=93.6
- any-next-fill cumulative share: <30s=30.9%, <60s=43.3%, <120s=61.4%, <300s=78.8%, none-in-window=20.9%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=43: p10=-1.00, p25=-0.10, p50=0.00, p75=0.10, p90=0.20
- crossed-market share: 14.4%

## Sanity checks
- A_total_pnl_match_within_10pct: **PASS**
- B_fill_count_subset_invariants: **PASS**
- C_pnl_monotonicity_warnings: **PASS**
  - leader PnL (replay calc) = 14,951,978; closed_positions sum = 14,952,586 (0.0% drift). Difference largely from marking unresolved positions to last fill price; closed_positions does not.

PnL-monotonicity informationals (NOT bugs — see check description):
  - sports: C_real PnL > C_opt PnL — same dynamic on the pure-maker branch.
  - other: A_real PnL (1,086,825) > A_opt PnL (878,428) — the maker fills filtered by the realistic model were net-losing, so dropping them raised PnL.
  - other: C_real PnL > C_opt PnL — same dynamic on the pure-maker branch.
  - politics: A_real PnL (-41,980) > A_opt PnL (-44,438) — the maker fills filtered by the realistic model were net-losing, so dropping them raised PnL.
  - politics: C_real PnL > C_opt PnL — same dynamic on the pure-maker branch.
  - sports (maker_share=84%): pure-taker B PnL (10,034,043) > role-mirrored A_real (7,056,737) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.
  - other (maker_share=85%): pure-taker B PnL (3,002,308) > role-mirrored A_real (1,086,825) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.

## Interpretation
Leader `leader_top_leaderboard` is **84% maker** on 184,268 fills with leader realised PnL of **$14,951,978** in this window, so the audit hinges on the maker-leg fill model and adverse selection more than on taker slippage.

**No family clears the deployable bar** (A_real capture ≥40% on positive leader PnL AND adverse_select_ratio ≥0.85). Closest call: **sports** with A_real capture 62%, leader PnL $11,307,668, adv-sel 0.70.

Adverse selection is destructive for: **sports, other** (maker-leg fill rate is significantly higher on losing positions than winning ones). This is the central economic blocker — even free maker rebates don't fix this.

Pure-taker (Branch B) is uniformly worse than role-mirroring across the maker-dominant families — paying 3¢ to cross spread on his entire flow burns capital faster than the leader earns from maker rebates.

**Structurally unstable**: sports have >30pp capture swings between H1 2025 and 2026 — historical capture there does not generalise; only a live A/B test will answer it.