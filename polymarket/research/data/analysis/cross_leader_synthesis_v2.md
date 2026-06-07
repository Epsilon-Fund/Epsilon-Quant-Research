---
title: "Cross-leader copy-execution synthesis v2"
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
# Cross-leader copy-execution synthesis v2
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


7 audited leaders. Cells are (leader × family × role × hour_bucket); deployable threshold per task spec: `A_real_capture > 0.3` AND `adverse_select_ratio > 0.85` AND `leader_pnl > 0`.

## 1. Per-leader headline

| leader | address | n_fragments | n_positions | leader_pnl_window | n_cells_total | n_cells_deployable_cell_level | n_families_deployable_family_level |
|---|---|---|---|---|---|---|---|
| domah | 0x9d84ce0306f8551e02efef1680475fc0f1dc1344 | 170,005 | 5,890 | $941,123 | 47 | 3 | 1 |
| ee00ba | 0xee00ba338c59557141789b127927a55f5cc5cea1 | 93,653 | 8,723 | $4,066,602 | 36 | 2 | 0 |
| top_leaderboard | 0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee | 184,268 | 4,365 | $14,951,978 | 19 | 2 | 0 |
| high_conviction | 0x204f72f35326db932158cba6adff0b9a1da95e14 | 5,127,237 | 167,238 | $10,637,399 | 28 | 3 | 0 |
| ultra_maker | 0x2005d16a84ceefa912d4e380cd32e7ff827875ea | 2,456,399 | 101,136 | $8,376,945 | 28 | 2 | 0 |
| negrisk_directional_1 | 0x629bc4a1e53e1d475beb7ea3d388791e96dd995a | 38,774 | 1,064 | $994,599 | 27 | 6 | 1 |
| negrisk_directional_2 | 0x5bffcf561bcae83af680ad600cb99f1184d6ffbe | 21,449 | 500 | $975,270 | 35 | 1 | 0 |

## 2. Deployable cells (cell-level) per leader

One row per (leader × family × role × hour_bucket) cell that meets the deployable thresholds. If empty for a leader, no copy execution profile passes the bar for that leader.

| leader | family | role | hour_bucket | n_fills | leader_pnl | A_real_capture | adverse_select_ratio |
|---|---|---|---|---|---|---|---|
| domah | crypto | maker | 12-18 | 545 | $15,927 | 0.606 | 1.859 |
| domah | macro | maker | 18-24 | 10,139 | $177,397 | 1.016 | 1.568 |
| domah | politics | maker | 00-06 | 26,810 | $189,220 | 0.450 | 0.969 |
| ee00ba | other | taker | 00-06 | 1,055 | $226,080 | 1.009 | 1.284 |
| ee00ba | other | taker | 06-12 | 2,676 | $168,169 | 0.459 | 1.958 |
| high_conviction | other | taker | 06-12 | 344,395 | $1,671,956 | 0.587 | 0.970 |
| high_conviction | other | taker | 12-18 | 382,179 | $2,201,318 | 0.793 | 0.950 |
| high_conviction | other | taker | 18-24 | 426,803 | $1,363,254 | 0.580 | 0.926 |
| negrisk_directional_1 | other | maker | 00-06 | 326 | $4,477 | 0.655 | 3.375 |
| negrisk_directional_1 | other | maker | 12-18 | 2,385 | $204,671 | 0.529 | 0.954 |
| negrisk_directional_1 | other | maker | 18-24 | 2,347 | $93,511 | 0.623 | 2.085 |
| negrisk_directional_1 | other | taker | 12-18 | 1,983 | $312,734 | 0.750 | 4.134 |
| negrisk_directional_1 | politics | taker | 06-12 | 1,839 | $108,520 | 0.558 | 1.262 |
| negrisk_directional_1 | politics | taker | 18-24 | 3,308 | $98,772 | 0.357 | 1.182 |
| negrisk_directional_2 | sports | maker | 00-06 | 210 | $3,664 | 0.519 | 1.253 |
| top_leaderboard | sports | taker | 12-18 | 13,298 | $1,167,148 | 1.002 | 1.048 |
| top_leaderboard | sports | taker | 18-24 | 15,977 | $4,201,307 | 0.739 | 0.902 |
| ultra_maker | politics | maker | 18-24 | 1,184 | $4,286 | 0.323 | 1.149 |
| ultra_maker | sports | taker | 06-12 | 12,049 | $171,685 | 0.583 | 0.953 |

## 3. Cross-leader cell intersection (≥2 leaders deployable)

Cells deployable for ≥2 leaders — candidates for a cohort signal:

| family | role | hour_bucket | leaders_deployed | n_leaders_deployed |
|---|---|---|---|---|
| other | taker | 06-12 | ee00ba, high_conviction | 2 |
| other | taker | 12-18 | high_conviction, negrisk_directional_1 | 2 |

## 4. Copyability metrics anchored to outcomes

Per-leader copyability metrics (from `data/copyability_candidates/traders_copyability_metrics.parquet`) alongside the audit's deployable-cell counts. With n=7 leaders this is suggestive only; correlations are reported, not used as gating rules.

| leader | address | fragmentation_index | hold_to_resolution_share | split_position_signature | market_family_concentration | win_loss_size_ratio | style_role_balance | active_days_last_90d | volume_30d_to_lifetime_ratio | dominant_family | mkt_total_pnl | n_closed_positions | n_cells_deployable_cell_level | n_families_deployable_family_level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top_leaderboard | 0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee | 17.000 | 32.9% | 26.8% | 0.673 | 1.050 | 0.837 | 85 | 8.9% | sports | $14,952,586 | 4,359 | 2 | 0 |
| high_conviction | 0x204f72f35326db932158cba6adff0b9a1da95e14 | 14.000 | 74.4% | 18.4% | 0.620 | 0.828 | 0.569 | 90 | 12.8% | other | $10,636,907 | 167,135 | 3 | 0 |
| ultra_maker | 0x2005d16a84ceefa912d4e380cd32e7ff827875ea | 8.000 | 30.3% | 7.1% | 0.543 | 1.048 | 0.901 | 90 | 30.5% | other | $8,378,137 | 101,129 | 2 | 0 |
| ee00ba | 0xee00ba338c59557141789b127927a55f5cc5cea1 | 3.000 | 7.1% | 50.3% | 0.542 | 0.530 | 0.792 | 70 | 8.7% | sports | $4,658,451 | 11,726 | 2 | 0 |
| domah | 0x9d84ce0306f8551e02efef1680475fc0f1dc1344 | 11.000 | 27.3% | 6.2% | 0.546 | 0.939 | 0.888 | 90 | 0.7% | politics | $4,007,338 | 16,306 | 3 | 1 |
| negrisk_directional_2 | 0x5bffcf561bcae83af680ad600cb99f1184d6ffbe | 8.000 | 21.3% | 35.6% | 0.726 | 0.546 | 0.900 | 62 | 1.5% | politics | $3,275,528 | 1,108 | 1 | 0 |
| negrisk_directional_1 | 0x629bc4a1e53e1d475beb7ea3d388791e96dd995a | 10.000 | 37.1% | 30.1% | 0.611 | 2.244 | 0.781 | 90 | 5.7% | politics | $1,795,457 | 1,365 | 6 | 1 |

### Correlations (Spearman) with `n_cells_deployable_cell_level` (n=7)

| metric | spearman_rho | n |
|---|---|---|
| active_days_last_90d | 0.805 | 7 |
| hold_to_resolution_share | 0.636 | 7 |
| win_loss_size_ratio | 0.505 | 7 |
| fragmentation_index | 0.397 | 7 |
| volume_30d_to_lifetime_ratio | -0.094 | 7 |
| mkt_total_pnl | -0.206 | 7 |
| market_family_concentration | -0.225 | 7 |
| split_position_signature | -0.374 | 7 |
| style_role_balance | -0.636 | 7 |

_Spearman because the sample is tiny and we don't want a single outlier dominating Pearson. Not gating thresholds; just direction of association._

## 5. Answers to the task's per-leader hypotheses

**top_leaderboard ($14.95M PnL) vs smaller-PnL specialists.** top_leaderboard has **2 deployable cells** (0 families), vs Domah 3 cells / 1 fam, negrisk_directional_1 6 cells / 1 fam. **Highest PnL ≠ most copyable.** The leaderboard's two deployable cells are both `sports / taker / {12-18, 18-24}` — high-fill, late-day sports games. They are real but narrow: copy execution must take taker and fire only in those windows. Lifetime mkt_total_pnl alone does NOT predict deployable footprint (Spearman ρ ≈ -0.21 across n=7).

**high_conviction (hold_to_resolution_share = 74.4%).** 3 deployable cells, 0 family-level — the extreme conviction profile (mostly 'other' family with taker entries holding to resolution) produces 3 deployable cells, all `other / taker / {06-12, 12-18, 18-24}`. **Conviction helps but doesn't dominate**: Domah ties at 3 cells with only 27% hold-to-res. Hold-to-resolution correlates positively with cell count (ρ ≈ +0.64) but the largest deployable footprint in the sample is negrisk_directional_1 at 6 cells, whose hold-to-res is 37% — not the highest.

**ultra_maker (role_balance = 0.90, even more maker than Domah).** 2 deployable cells (0 family-level): only `politics / maker / 18-24` (tiny n=1,184) and `sports / taker / 06-12`. **Maker-leg adverse selection generalises** — none of his core 'other' or sports maker fills clear the adv_sel bar. The Domah politics-maker problem is not a Domah-specific pathology; it shows up wherever a leader posts passive limits on directional event markets, the maker bids that get filled are systematically the losing ones.

**negrisk_directional_1 (the ex-Pool-C closing-the-loop case).** 6 deployable cells, 1 at the family level — **most deployable footprint in the entire sample**. The cells are spread across 'other' (both maker and taker, all hour buckets) and politics-taker (06-12, 18-24). The directionality reclassification was correct: this trader's edge is captureable, but only on the taker side for politics — **confirms the same politics-uncopyable-as-maker / politics-deployable-as-taker** pattern that fell out of Domah's audit.

**negrisk_directional_2 (second NegRisk-political data point).** 1 deployable cell only: `sports / maker / 00-06` with n=210 fills. **Domah's politics-maker pattern does NOT fully generalise to other politics-dominant NegRisk bettors.** negrisk_directional_2's politics fills failed (adv_sel 0.247 << 0.85) and the lone deployable cell is sports, not politics. Each leader's deployable cell set is essentially idiosyncratic.

## 6. Cross-leader intersection — interpretation

**Cross-leader intersection is NOT entirely empty** — 2 cells are deployable for ≥2 leaders, both `other / taker / ...`:

- `other / taker / 06-12` — leaders: ee00ba, high_conviction
- `other / taker / 12-18` — leaders: high_conviction, negrisk_directional_1

These are narrow overlaps in the 'other' grab-bag family (not a single discoverable market type), with only 2-leader overlap. The 'cohort-style multi-leader signal' framing is **weakly viable in 'other' / taker afternoons**, but only as a consensus across specifically high_conviction + ee00ba (06-12) or high_conviction + negrisk_directional_1 (12-18). Per-leader execution remains the dominant frame; cohort framing earns ~2 narrow cells.

## 7. Anchoring observations from copyability metrics (n=7, descriptive)

Spearman correlations against `n_cells_deployable_cell_level` (cell-level count):

- **`active_days_last_90d` (ρ +0.81):** currently-active traders have more deployable cells. Stale traders' patterns may have decayed, or their fill history isn't dense enough in the recent window for the audit to find adv_sel ≥0.85 cells.
- **`hold_to_resolution_share` (ρ +0.64):** holding to resolution helps — fewer mid-position exits means the copy bot doesn't need to replicate exit timing.
- **`win_loss_size_ratio` (ρ +0.50) / `fragmentation_index` (ρ +0.40):** weak positive — leaders whose winners are bigger than their losers and who scale in are easier to copy.
- **`style_role_balance` (ρ −0.64):** stronger maker traders have fewer deployable cells. Consistent with the maker-adverse-selection theme: high-role-balance leaders have larger maker books, and their maker fills disproportionately come off when their post is wrong.
- **`split_position_signature` (ρ −0.37):** weak negative — leaders with split-construction signatures (already flagged uncopyable for 0xd38b71f3) tend to have fewer audit-deployable cells, but the signal is weak across the audited leaders (0xd38b71f3 itself was excluded from the audit set per the flagged_uncopyable label).
- **`mkt_total_pnl` (ρ −0.21):** no correlation between lifetime PnL and copyability. **Choosing audit candidates by PnL is wrong**; use active_days_last_90d + hold_to_resolution_share + low role_balance instead.
_n=7, all rank correlations. Not gating thresholds — these are directions of association in a tiny sample to inform the next round of candidate selection._

## 8. Sanity checks across all 5 new audits

| leader | A_total_pnl_match_within_10pct | B_fill_count_subset_invariants | C_pnl_monotonicity_warnings |
|---|---|---|---|
| top_leaderboard | PASS | PASS | PASS |
| high_conviction | PASS | PASS | PASS |
| ultra_maker | PASS | PASS | PASS |
| negrisk_directional_1 | **FAIL** (34% drift — unresolved positions mark-to-last-fill) | PASS | PASS |
| negrisk_directional_2 | **FAIL** (19% drift — unresolved positions mark-to-last-fill) | PASS | PASS |

The two FAILs are the documented limitation: when a leader still has unresolved (open) positions inside the audit window, the replay marks those to last-fill price whereas `closed_positions.parquet` does not. Numerical impact is contained to the replay PnL vs closed_positions delta; the family-level capture/adv_sel numbers come from positions that resolved within window, so the deployable-cell count is unaffected. These FAILs are informational, not bugs in the audit.

## 9. Update applied to copyability parquet

Wrote `n_deployable_cells` (family-level count from each leader's recomputed family table) and updated `audit_status` to `audited` for the 5 newly-audited addresses, in `data/copyability_candidates/traders_copyability_metrics.parquet`. The 2 previously-audited leaders' `n_deployable_cells` values are also overwritten to match the augmented-keyword recompute, so the column is internally consistent across all 7 audited rows. `flagged_uncopyable` for 0xd38b71f3 is preserved.
