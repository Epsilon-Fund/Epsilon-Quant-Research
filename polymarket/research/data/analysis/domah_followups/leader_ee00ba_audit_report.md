---
title: "Copy-Execution Audit: leader_ee00ba"
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
# Copy-Execution Audit: leader_ee00ba
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


_Generated 2026-05-16 16:28 UTC_

Read-only diagnostic. Trade data covers 2025-01-02 → 2026-04-24 (last available shard; not today 2026-05-16).

## Universe summary
- **label**: leader_ee00ba
- **address**: 0xee00ba338c59557141789b127927a55f5cc5cea1
- **window_start**: 2025-01-02
- **window_end_exclusive**: 2026-04-25
- **n_fills**: 93,653
- **dollar_volume_usd**: 141,886,029.01
- **n_positions**: 8,723
- **n_resolved_positions**: 8,696
- **n_families_covered**: 5
- **leader_pnl_calc_usd**: 4,066,602.15
- **leader_pnl_closed_positions_usd**: 4,126,657.52

## Primary family table

`A_opt` / `A_real`: role-mirrored, optimistic vs realistic maker-fill model. `B`: pure taker. `C_opt` / `C_real`: pure maker. `adverse_select_ratio` = A_real maker-fill rate on winning vs losing positions (N/A if either bucket <30 maker fills). Deployable cells have **high capture AND adverse_select_ratio close to 1.0**.

| family   | n_fills | n_positions | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_opt_capture | A_real_capture | A_taker_fallback_pct | A_maker_realfill_rate | adverse_select_ratio | adv_sel_n_win | adv_sel_n_lose |
| -------- | ------- | ----------- | ---------- | --------- | ---------- | -------- | --------- | ---------- | ------------- | -------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
|   sports |  59,003 |       5,639 |  2,208,974 |   178,480 |   -616,181 | -379,082 |   813,006 |    -11,422 |        0.0808 |        -0.2789 |               0.3436 |                0.2144 |               0.3882 |        34,788 |         10,319 |
|    other |  32,302 |       2,872 |  1,473,325 |   966,782 |    746,228 |  519,042 | 1,179,076 |    923,856 |        0.6562 |         0.5065 |               0.3820 |                0.2105 |               0.3876 |        19,196 |          7,561 |
| politics |   1,134 |          77 |    117,816 |    29,871 |     24,589 |  100,126 |    15,189 |      8,731 |        0.2535 |         0.2087 |               0.6176 |                0.0301 |               0.1317 |           813 |            185 |
|   crypto |   1,133 |         127 |    264,110 |   139,604 |    134,978 |  177,444 |   155,306 |    149,983 |        0.5286 |         0.5111 |               0.1647 |                0.2118 |               0.1896 |           930 |            118 |
|    macro |      81 |           8 |      2,377 |     1,588 |      1,588 |     -649 |     2,453 |      2,453 |        0.6684 |         0.6684 |               0.4545 |                0.0857 |                      |            62 |              8 |

### Secondary slices

#### By market lifecycle phase (hours-to-resolution at fill time)
| slice           | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl    | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| --------------- | ----------- | ------- | ---------- | --------- | ---------- | -------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
|          middle |       3,134 |  29,994 |  1,807,492 |   843,164 |    604,341 |  494,531 | 1,346,920 |    985,004 |               0.2878 |               0.2152 |                0.1885 |               0.3822 |        0.4665 |         0.3344 |
| near-resolution |       4,209 |  44,520 |  1,716,734 |   268,069 |   -312,526 |  323,226 |   610,497 |    109,687 |               0.3167 |               0.2699 |                0.2506 |               0.3663 |        0.1562 |        -0.1820 |
|            open |       1,380 |  19,139 |    542,376 |   205,093 |       -613 | -400,875 |   207,615 |    -21,090 |               0.5225 |               0.1605 |                0.1412 |               0.5172 |        0.3781 |        -0.0011 |

#### By hour-of-day (UTC)
| slice | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl   | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | ------- | ---------- | --------- | ---------- | ------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| 00-06 |       3,594 |  35,800 |  1,596,969 |   703,010 |    383,271 |  30,515 | 1,359,041 |  1,011,210 |               0.2530 |               0.2892 |                0.2625 |               0.4824 |        0.4402 |         0.2400 |
| 18-24 |       2,847 |  28,379 |  1,193,332 |   326,325 |   -187,359 |  98,719 |   377,276 |    -82,424 |               0.4060 |               0.1799 |                0.1586 |               0.2832 |        0.2735 |        -0.1570 |
| 12-18 |       1,588 |  17,936 |    873,717 |   201,651 |    105,479 | 217,531 |   302,763 |     88,485 |               0.4912 |               0.1535 |                0.1389 |               0.3327 |        0.2308 |         0.1207 |
| 06-12 |         694 |  11,538 |    402,585 |    85,340 |    -10,189 |  70,116 |   125,951 |     56,330 |               0.3820 |               0.3023 |                0.2850 |               0.4960 |        0.2120 |        -0.0253 |

#### By leader's role on the originating fill
| slice | n_positions | n_fills | leader_pnl | A_opt_pnl | A_real_pnl | B_pnl   | C_opt_pnl | C_real_pnl | A_taker_fallback_pct | A_maker_optfill_rate | A_maker_realfill_rate | adverse_select_ratio | A_opt_capture | A_real_capture |
| ----- | ----------- | ------- | ---------- | --------- | ---------- | ------- | --------- | ---------- | -------------------- | -------------------- | --------------------- | -------------------- | ------------- | -------------- |
| taker |       3,453 |  29,150 |  1,528,075 |   648,960 |    614,083 | 182,192 | 1,350,379 |  1,208,537 |               0.4025 |               0.2175 |                0.1965 |               0.6731 |        0.4247 |         0.4019 |
| maker |       5,270 |  64,503 |  2,538,527 |   667,365 |   -322,881 | 234,689 |   814,652 |   -134,936 |               0.2262 |               0.2354 |                0.2138 |               0.3279 |        0.2629 |        -0.1272 |

### Sensitivity: H1 2025 vs 2026-YTD

Flag families with `|H1 cap − 2026 cap| > 30pp` AND `n_fills > 200` as 'structure-shifting': historical capture does not generalise.

| family   | A_opt_capture_2025-H1 | A_opt_capture_2026-YTD | A_real_capture_2025-H1 | A_real_capture_2026-YTD | n_fills_2025-H1 | n_fills_2026-YTD |
| -------- | --------------------- | ---------------------- | ---------------------- | ----------------------- | --------------- | ---------------- |
|   crypto |                0.4938 |                        |                 0.4758 |                         |             697 |                  |
|    macro |                0.4802 |                        |                 0.4802 |                         |               8 |                  |
|    other |                0.7829 |                 1.0856 |                 0.7059 |                  1.1477 |          10,197 |            6,784 |
| politics |                0.6304 |                 0.0785 |                 0.6306 |                  0.0260 |             176 |               34 |
|   sports |                0.3590 |                 0.9857 |                 0.0584 |                  0.9015 |          32,247 |            6,790 |

### Sensitivity: fallback cents (taker-leg only)

| family   | A_real_pnl_1c | A_real_pnl_2c | A_real_pnl_3c | A_real_pnl_5c | B_pnl_1c | B_pnl_2c | B_pnl_3c | B_pnl_5c   |
| -------- | ------------- | ------------- | ------------- | ------------- | -------- | -------- | -------- | ---------- |
|   crypto |       135,049 |       135,014 |       134,978 |       134,907 |  211,752 |  194,598 |  177,444 |    143,137 |
|    macro |         2,189 |         1,889 |         1,588 |           988 |    1,212 |      282 |     -649 |     -2,510 |
|    other |       801,945 |       774,087 |       746,228 |       690,511 |  994,427 |  756,735 |  519,042 |     43,658 |
| politics |        26,174 |        25,381 |        24,589 |        23,003 |  108,855 |  104,491 |  100,126 |     91,396 |
|   sports |      -442,934 |      -529,558 |      -616,181 |      -789,429 |  793,994 |  207,456 | -379,082 | -1,552,157 |

### Per-family diagnostics

**sports** — n_fills=59,003, taker=13,896, maker=45,107
- taker_fallback_pct: 34.4%; maker_no-fill_optimistic: 76.0%; maker_no-fill_realistic: 78.6%
- taker lag percentiles (s): p10=2.0, p25=6.0, p50=16.0, p75=48.0, p90=126.0
- any-next-fill cumulative share: <30s=26.8%, <60s=33.1%, <120s=38.5%, <300s=44.2%, none-in-window=55.8%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=9,121: p10=-1.00, p25=0.00, p50=0.00, p75=1.00, p90=5.00
- crossed-market share: 19.6%

**other** — n_fills=32,302, taker=5,545, maker=26,757
- taker_fallback_pct: 38.2%; maker_no-fill_optimistic: 77.4%; maker_no-fill_realistic: 78.9%
- taker lag percentiles (s): p10=2.0, p25=4.0, p50=12.0, p75=42.0, p90=127.4
- any-next-fill cumulative share: <30s=27.3%, <60s=31.7%, <120s=36.5%, <300s=42.7%, none-in-window=57.2%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=3,427: p10=-1.00, p25=0.00, p50=0.00, p75=1.00, p90=4.00
- crossed-market share: 17.0%

**politics** — n_fills=1,134, taker=136, maker=998
- taker_fallback_pct: 61.8%; maker_no-fill_optimistic: 96.0%; maker_no-fill_realistic: 97.0%
- taker lag percentiles (s): p10=2.2, p25=9.5, p50=48.0, p75=120.0, p90=158.8
- any-next-fill cumulative share: <30s=9.3%, <60s=16.4%, <120s=22.3%, <300s=34.1%, none-in-window=65.7%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=52: p10=-1.92, p25=0.00, p50=1.00, p75=4.23, p90=8.00
- crossed-market share: 15.8%

**crypto** — n_fills=1,133, taker=85, maker=1,048
- taker_fallback_pct: 16.5%; maker_no-fill_optimistic: 75.9%; maker_no-fill_realistic: 78.8%
- taker lag percentiles (s): p10=2.0, p25=2.0, p50=18.0, p75=64.0, p90=140.0
- any-next-fill cumulative share: <30s=24.3%, <60s=27.2%, <120s=30.0%, <300s=33.3%, none-in-window=66.7%
- slip on real taker next-fills (signed cents, +ve = bot worse off), n=71: p10=-0.20, p25=0.00, p50=1.00, p75=2.00, p90=3.00
- crossed-market share: 15.8%

## Sanity checks
- A_total_pnl_match_within_10pct: **PASS**
- B_fill_count_subset_invariants: **PASS**
- C_pnl_monotonicity_warnings: **PASS**
  - leader PnL (replay calc) = 4,066,602; closed_positions sum = 4,126,658 (1.5% drift). Difference largely from marking unresolved positions to last fill price; closed_positions does not.

PnL-monotonicity informationals (NOT bugs — see check description):
  - sports (maker_share=76%): pure-taker B PnL (-379,082) > role-mirrored A_real (-616,181) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.
  - politics (maker_share=88%): pure-taker B PnL (100,126) > role-mirrored A_real (24,589) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.
  - crypto (maker_share=92%): pure-taker B PnL (177,444) > role-mirrored A_real (134,978) — A_real misses too many of Domah's maker positions; pure-taker buys all of them at the next-fill price.

## Interpretation
Leader `leader_ee00ba` is **79% maker** on 93,653 fills with leader realised PnL of **$4,066,602** in this window, so the audit hinges on the maker-leg fill model and adverse selection more than on taker slippage.

**Deployable family today: macro** — A_real capture ≥40%, leader PnL positive, and adverse_select_ratio not destructive.

Adverse selection is destructive for: **sports, other, politics, crypto** (maker-leg fill rate is significantly higher on losing positions than winning ones). This is the central economic blocker — even free maker rebates don't fix this.

Pure-taker (Branch B) is uniformly worse than role-mirroring across the maker-dominant families — paying 3¢ to cross spread on his entire flow burns capital faster than the leader earns from maker rebates.

**Structurally unstable**: other, sports have >30pp capture swings between H1 2025 and 2026 — historical capture there does not generalise; only a live A/B test will answer it.

## Cross-leader analysis: Domah vs `0xee00ba…`

Both audits re-bucketed with the proposed FAMILY_KEYWORDS (Task 1). Cross-leader deployable threshold: **A_real capture > 30% AND adverse_select_ratio > 0.85 AND leader_pnl > 0**.

### Family-level side-by-side

| family   | n_fills_domah | leader_pnl_domah | A_real_capture_domah | adverse_select_ratio_domah | n_fills_ee00ba | leader_pnl_ee00ba | A_real_capture_ee00ba | adverse_select_ratio_ee00ba | deployable_domah | deployable_ee00ba | both_deployable |
| -------- | ------------- | ---------------- | -------------------- | -------------------------- | -------------- | ----------------- | --------------------- | --------------------------- | ---------------- | ----------------- | --------------- |
|   crypto |         1,839 |           -2,865 |               1.1421 |                     0.7202 |          1,133 |           264,110 |                0.5111 |                      0.1896 |            False |             False |           False |
|    macro |        27,082 |          412,419 |               0.6342 |                     1.3575 |             81 |             2,377 |                0.6684 |                             |             True |             False |           False |
|    other |        37,735 |          283,051 |               0.1768 |                     1.0141 |         32,302 |         1,473,325 |                0.5065 |                      0.3876 |            False |             False |           False |
| politics |        96,833 |          291,169 |              -0.8117 |                     0.7010 |          1,134 |           117,816 |                0.2087 |                      0.1317 |            False |             False |           False |
|   sports |         5,703 |          -50,211 |               1.4042 |                     0.3096 |         59,003 |         2,208,974 |               -0.2789 |                      0.3882 |            False |             False |           False |
|  weather |           813 |            7,558 |               0.0848 |                     0.0000 |                |                   |                       |                             |            False |             False |           False |

### 1. Does Domah's macro-only finding generalise?

- Domah macro: A_real capture **63%** on $412,419 leader PnL, adv-sel **1.36** → deployable.
- 0xee00ba macro: A_real capture **67%** on $2,377 leader PnL (n_fills=81).
- **Macro does NOT meaningfully apply to 0xee00ba** — he barely trades it (81 fills vs Domah's 27,082). Not a generalising signal; it's a Domah-specific specialisation.

- 0xee00ba's own deployable families: **none**.

### 2. Where do they diverge?

- **Deployable for Domah only**: macro.
- Deployable for neither: crypto, other, politics, sports, weather.

Brief explanations for the divergences:

  - **crypto**: Domah A_real cap +114% (adv-sel 0.72), 0xee00ba A_real cap +51% (adv-sel 0.19) ↓. Both leaders are positive but small samples; capture rough but consistent in sign.
  - **politics**: Domah A_real cap -81% (adv-sel 0.70), 0xee00ba A_real cap +21% (adv-sel 0.13) ↑. Politics destroys Domah (he averages in on losers); 0xee00ba's tiny politics footprint (~1k fills) means the leader PnL is dominated by a small number of bets — high variance, not a stable signal.
  - **sports**: Domah A_real cap +140% (adv-sel 0.31), 0xee00ba A_real cap -28% (adv-sel 0.39) ↓. Domah's sports footprint is tiny; 0xee00ba is sports-heavy. Sports A_real capture for 0xee00ba is negative because his maker fills suffer adverse selection on his big sports book.

### 3. Cross-leader deployable intersection (family × role × hour)

For each (family, role, hour_bucket) cell, compute A_real capture + adverse_select_ratio separately for each leader. Deployable iff both have A_real capture > 30% AND adverse_select_ratio > 0.85 AND leader_pnl > 0.

**No (family × role × hour) cell is deployable for both leaders.**

This means: there is no single execution profile that captures both leaders' edge under a copy-mirroring strategy. Either each leader needs a per-leader execution policy, or a multi-leader strategy requires a different mechanism (e.g. consensus signal across leaders rather than per-fill mirroring).


**Cells deployable for AT LEAST ONE leader (5):**

| family   | role  | hour_bucket | n_fills_d | leader_pnl_d | cap_d   | advsel_d | n_fills_e | leader_pnl_e | cap_e  | advsel_e | dep_d | dep_e |
| -------- | ----- | ----------- | --------- | ------------ | ------- | -------- | --------- | ------------ | ------ | -------- | ----- | ----- |
|   crypto | maker |       12-18 |       545 |       15,927 |  0.6056 |   1.8595 |       603 |       53,275 | 0.7484 |          |  True | False |
|    macro | maker |       18-24 |    10,139 |      177,397 |  1.0158 |   1.5679 |        18 |          579 | 0.3008 |          |  True | False |
|    other | taker |       00-06 |     3,578 |      113,392 |  0.4869 |   0.7762 |     1,055 |      226,080 | 1.0095 |   1.2839 | False |  True |
|    other | taker |       06-12 |       482 |       18,696 | -0.1687 |   1.0169 |     2,676 |      168,169 | 0.4589 |   1.9576 | False |  True |
| politics | maker |       00-06 |    26,810 |      189,220 |  0.4496 |   0.9686 |        58 |      -19,742 | 1.0368 |          |  True | False |

### Synthesis

The two leaders' deployable cells lie on **opposite execution profiles**: Domah's are exclusively **maker** (he is 87% maker; the bot succeeds when it copies his bids), while 0xee00ba's are exclusively **taker** (he is 69% maker but his selection edge transfers when the bot crosses spread on his behalf). The cross-leader intersection is empty because there is no family × hour cell where both leaders' edge survives copy execution in the same direction; copy-trading them is two separate strategies, not one. Macro is Domah-specific (0xee00ba has 81 fills); sports is 0xee00ba's only large family but his sports maker fills suffer adverse selection (0.39). A multi-leader strategy would need to combine the signals upstream (consensus of leaders before execution), not merge fills downstream of each leader.
