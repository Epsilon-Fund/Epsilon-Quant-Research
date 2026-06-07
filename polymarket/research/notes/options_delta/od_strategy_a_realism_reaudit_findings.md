---
title: "OD Strategy A Realism Re-Audit: Passive Longshot/Vol Harvest Was Killed Too Harshly"
created: 2026-06-05
status: watching
owner: justin
project: polymarket
para: project
hubs:
  - strat_options_delta
  - COWORK
tags:
  - research
  - options-delta
---
# OD Strategy A Realism Re-Audit: Passive Longshot/Vol Harvest Was Killed Too Harshly

> Hub: [[strat_options_delta]] · [[COWORK]]
> Related: [[od_strategy_a_v2_lifecycle_findings]] · [[od_strategy_a_v3_findings]] · [[od_strategy_a_v3_pnl_risk_findings]] · [[block_k6_strategy_a_static_hedge_findings]] · [[block_k5_stress_findings]] · [[mm_deployable_cells_findings]] · [[od_conditional_prob_calibration_findings]] · [[od_pricing_model_form_findings]] · [[od_same_day_crypto_pricing_gate_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Summary

- Scope: OD Strategy A Realism Re-Audit: Passive Longshot/Vol Harvest Was Killed Too Harshly in the OD/options-delta area.
- Existing takeaway/status: Final re-audit verdict: **MERITS-LIVE-MEASUREMENT-LOOP** for the source-clean/rich-short Strategy A longshot harvest at small capital, not a production trading system and not a full standalone OD reopen.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Headline

Final re-audit verdict: **MERITS-LIVE-MEASUREMENT-LOOP** for the source-clean/rich-short Strategy A longshot harvest at small capital, not a production trading system and not a full standalone OD reopen.

The prior hard close was too harsh in two specific ways: it treated the one-position-global embargo as a law rather than a capital assumption, and it used the borrowed 1.98c crypto-4h structural queue baseline as a kill switch in later OD close notes. Under a realistic $10-$100 small-capital deployment, concurrent BTC/ETH/SOL sleeves are legitimate. The clean per-asset strict-rich design has **18.36c/filled contract** with market-cluster CI **[0.25c, 27.43c]** across **7 markets / 22 fills**. A $50 dollar-delta cap keeps **15.49c/weighted contract**, CI **[0.48c, 47.54c]**.

The close is still fair for the **bare one-position-global far-|z| design**: 10.42c/contract, CI **[-2.02c, 26.95c]**, only 6 markets. That design remains **CONFIRM-CLOSE**. The surviving Strategy A form is narrower: source-clean, rich-short, passive maker lifecycle, carried to resolution, with live instrumentation for fill share, queue, adverse selection, and capacity.

No new data capture was used. Inputs were the existing `kpeg_robustness_fills`, `k6_vol_gap_panel`, `od_strategy_a_v3_*`, and `k5_stress_*` artifacts. Outputs for this audit:

- `data/analysis/csv_outputs/options_delta/od_strategy_a_realism_reaudit_designs.csv`
- `data/analysis/csv_outputs/options_delta/od_strategy_a_realism_reaudit_k6_far_family.csv`
- `data/analysis/csv_outputs/options_delta/od_strategy_a_realism_reaudit_same_day_touch_capacity.csv`
- `data/analysis/csv_outputs/market_making/mm_deployable_cells_capacity_sensitivity.csv`
- `data/analysis/csv_outputs/market_making/k5_stress_gate_rows_for_realism_reaudit.csv`

## Rule 1 - Powerless OOS Split

Where the old close overreached:

- [[od_strategy_a_v2_lifecycle_findings]] correctly said the OOS far-|z| gate failed at n=6, but the decision language still let a powerless 6-market split close the lifecycle gate.
- [[od_strategy_a_v3_findings]] made the one-position-global embargo the official power assumption and demoted per-asset concurrency to diagnostic. That was coherent as a pre-registration, but not a law of the instrument.
- [[od_strategy_a_v3_pnl_risk_findings]] later made per-asset concurrency official, which implicitly admits the prior global power failure was a capital assumption, not evidence that the edge was gone.

The re-audit reports both capital views. PnL is settlement PnL plus maker rebate, no mark-to-mid and no Polymarket exit. Per-contract edge is total PnL divided by filled contracts, with market-cluster bootstrap CIs.

| design | capital view | markets | contracts | episode mean | per-contract edge | 5% visible-opportunity diagnostic | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bare far all-tau | one-position global | 6 | 68 | 118.08c `[-17.14c, 328.66c]` | 10.42c `[-2.02c, 26.95c]` | 0.52c `[-0.10c, 1.35c]` | **CONFIRM-CLOSE** for this capital model |
| Bare far all-tau | concurrent BTC/ETH/SOL | 17 | 211 | 85.56c `[-1.24c, 205.70c]` | 6.89c `[-0.15c, 13.93c]` | 0.34c `[-0.01c, 0.70c]` | underpowered, not no-edge |
| Bare widened longshot `abs_z >= 0.75` | one-position global | 6 | 116 | 291.52c `[3.96c, 627.72c]` | 15.08c `[-0.10c, 24.21c]` | 0.75c `[-0.01c, 1.21c]` | positive episode diagnostic, not per-contract proof |
| Strict-source far all-tau | one-position global | 6 | 84 | 115.78c `[4.38c, 319.23c]` | 8.27c `[0.30c, 22.39c]` | 0.41c `[0.02c, 1.12c]` | source-clean lifecycle survives |
| Strict-source rich-short >=1c far all-tau | one-position global | 4 | 16 | 105.20c `[16.19c, 258.69c]` | 26.30c `[15.35c, 30.32c]` | 1.32c `[0.77c, 1.52c]` | survives but n=4 |
| Strict-rich per-asset, flat one contract | concurrent BTC/ETH/SOL | 7 | 22 | 57.71c `[0.35c, 155.68c]` | 18.36c `[0.25c, 27.43c]` | 0.92c `[0.01c, 1.37c]` | **MERITS-LIVE-MEASUREMENT-LOOP** |
| Strict-rich per-asset, $50 dollar-delta cap | concurrent BTC/ETH/SOL | 7 | 9.80 weighted | 21.68c `[0.69c, 53.77c]` | 15.49c `[0.48c, 47.54c]` | 0.77c `[0.02c, 2.38c]` | **MERITS-LIVE-MEASUREMENT-LOOP** |
| Strict-rich per-asset, fair-value scaled | concurrent BTC/ETH/SOL | 7 | 15.78 weighted | 31.72c `[1.93c, 77.09c]` | 14.07c `[1.94c, 17.63c]` | 0.70c `[0.10c, 0.88c]` | **MERITS-LIVE-MEASUREMENT-LOOP** |
| Strict-rich per-asset, 25% Kelly proxy | concurrent BTC/ETH/SOL | 7 | 8.79 weighted | 2.73c `[-0.46c, 7.46c]` | 2.18c `[-0.24c, 13.44c]` | 0.11c `[-0.01c, 0.67c]` | **CONFIRM-CLOSE** for this sizing |

Read: for the actual intended deployment scale, a one-global-position embargo is more conservative than the capital constraint requires. The realistic small-capital view is concurrent BTC/ETH/SOL, judged per asset-market episode. That view does not turn the bare lifecycle into a clean statistical pass, but it does show real, economically visible edge once the source/rich-short filter is applied.

## Rule 2 - Borrowed Baseline As Kill Switch

Where the old close overreached:

- [[od_conditional_prob_calibration_findings]] closed Arm B partly because after the K5 top-3 haircut, incremental lower CI versus the v4 structural queue baseline was negative. The table used **1.98c** as `structural baseline`.
- [[od_pricing_model_form_findings]] repeated that same structural-incremental bar and reported the best after-top3 incremental lower CI as negative.
- [[strat_options_delta]] then summarized those rows as "CLOSE remains" after the K5 top-maker haircut and v4 structural queue baseline.

Under [[CODEX]] Rule 2, that is too harsh. The 1.98c number is a useful warning from a crypto-4h queue replay, but it is not a gate-grade baseline for this passive longshot-harvest carry form. The saved Strategy A artifacts do not include synchronized resting quote placement, queue position, cancels, or missed fills, so a true passive longshot-harvest structural baseline is **live-only unknown**.

The closest offline same-instrument diagnostic is source-only lifecycle versus strict-rich on the same markets. [[od_strategy_a_v3_pnl_risk_findings]] reports `strict_rich_short_minus_strict_source_same_markets` at **-40.47c/episode**, CI **[-120.93c, 1.49c]**. That is a serious diagnostic: the source/lifecycle filter is doing a lot of the work. It is not a hard kill because OD and MM are now explicitly two layers of one strategy, and n=7 is not enough to separate "valuation edge" from "source-clean lifecycle edge" offline.

New gate language: do not subtract the borrowed 1.98c as the official kill switch. Use it only as a labeled diagnostic until a live passive quote/queue baseline exists.

## Rule 4 - Statistical Survival Versus Economic Materiality

The prior notes sometimes decided on episode-level lower CI or a single bucket. The re-audit requires absolute cents per contract and market count.

The narrow Strategy A survivors are economically nontrivial on a per-filled-contract basis: 14c to 26c/contract point estimates, with lower CIs from 0.25c to 15.35c depending on the design. Even the 5% visible-opportunity diagnostic is materially larger than the same-day Arm T anchor's ~0.02c/contract, though it is still based on tiny samples.

The K6 static-hedge far-|z| family also should not be decided only by far/late. Recomputed strict-source OOS cells:

| K6 static-hedge far cell | markets | contracts | net per contract | unhedged | hedge cost | read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `far_absz_ge1|late_lt30m` | 11 | 11 | 1.07c `[-1.04c, 3.99c]` | 0.52c | 2.18c | original gate misses |
| `far_absz_ge1|mid_30m_2h` | 8 | 8 | 11.15c `[0.96c, 21.41c]` | 17.61c | 4.22c | clears |
| `far_absz_ge1|early_gt2h` | 1 | 1 | 58.02c `[58.02c, 58.02c]` | 84.19c | 2.52c | n=1 only |
| First far fill per market, all tau | 11 | 11 | 9.58c `[0.43c, 21.39c]` | 12.42c | 2.86c | family-level diagnostic clears |

Read: far/late alone was an arbitrary hard gate for a family that is directionally positive across time buckets. This does **not** unblock Kronos/HAR immediately; it says the static-hedge/longshot family belongs in the same live measurement loop as Strategy A, with the hedge treated as an overlay rather than the edge.

## Same-Day Touch-Fade Cluster Member

This does **not** relabel the standalone same-day Arm T line. [[od_same_day_crypto_pricing_gate_findings]] remains **CLOSE** as a standalone strategy because the pre-registered Tier-1 materiality bar was 0.25c/contract after the 5% capacity haircut, and the best held-out Tier-1 cell reached only 0.17c at that assumption.

For the Strategy A realism cluster, however, that same held-out cell belongs beside the longshot/short-vol harvest family. The trade is the same economic mechanism: the unlikely touch side is rich, so the fade buys the high-priced NO / equivalently sells the rich low-priced YES and carries to resolution. In the saved Tier-1 artifact, the registered held-out cell is `pos_z_ge_3/1_4h`: **31 fills / 9 markets**, raw net **3.49c/contract**, CI **[2.29c, 5.43c]**, BH q **0.0003**. Mean entry was a **96.16c NO** against mean PM YES price 3.84c and empirical YES probability 2.12c.

Capacity sensitivity using the same assumption ladder as the Strategy A/MM re-audit:

| capacity assumption | deployable cents/contract | CI | standalone 0.25c bar? | cluster read |
| --- | ---: | ---: | --- | --- |
| 5% modeled non-incumbent share | 0.17c | `[0.11c, 0.27c]` | fails on mean | standalone Arm T close remains fair |
| 10% small-cap improved queue share | 0.35c | `[0.23c, 0.54c]` | clears mean, lower CI below 0.25c | measurement-loop candidate |
| 22.68% historical Arm T non-top3 ceiling | 0.79c | `[0.52c, 1.23c]` | clears | measurement-loop candidate if live queue supports it |

Left-tail warning: this is the most dangerous member of the cluster. The held-out sample has only **9 market clusters**, and all 31 observed fills were NO winners. That under-samples adverse touch days. A single bad touch on a 96c NO can lose nearly the whole contract value, so the positive cluster CI is not a full tail model. The correct live loop must explicitly log barrier distance, jump/OFI state, quote exposure, same-day macro/event context, and stop/cancel behavior around touch acceleration.

## Rule 3 - Capacity As Assumption, Not Hard Cap

Where the old close overreached:

- [[mm_deployable_cells_findings]] called MM standalone sub-scale at about **$78/active day** using cell-specific entrant-capture assumptions, median structured-wallet bps, and 2026 active-day flow.
- That base case is a fair conservative model, but the capacity share, median-vs-mean choice, and 2026 run-rate are modeled assumptions. They should not become a hard law about what a better live entrant can capture.

Sensitivity on the 8 paper-qualified MM cells:

| capacity/run-rate scenario | median EV/day | mean EV/day | modeled headroom/day |
| --- | ---: | ---: | ---: |
| Script base, cell-specific capture, 2026 flow | $78 | $412 | $26,166 |
| Fixed 0.25% of non-top3 flow, 2026 flow | $76 | $406 | $26,024 |
| Fixed 1.00% of non-top3 flow, 2026 flow | $304 | $1,622 | $104,095 |
| Fixed 5.00% of non-top3 flow, 2026 flow | $1,521 | $8,112 | $520,477 |
| Fixed 0.25% of non-top3 flow, full-history flow | $17 | $92 | $5,542 |
| Fixed 5.00% of non-top3 flow, full-history flow | $332 | $1,833 | $110,836 |

Read: the base sub-scale conclusion is fair for the modeled median case, but not a hard cap. Real fill share and queue are live-only. MM standalone still does not justify a dedicated bot from the offline base case, but its structured non-top3 cells merit a **capacity measurement loop**, especially because Strategy A needs the same execution layer.

K5-STRESS category status also needs one wording fix: `politics_negrisk` is **blocked-pending-accounting**, not closed. Its structured-non-top3 row is huge (**1,177 bps**, CI **[502 bps, 2,174 bps]**, median **9.8 bps**), but NegRisk merge/split/redemption accounting is unresolved. The correct disposition is not "NO alpha"; it is "do not trade until accounting is reconstructed."

## Assumption Versus Live Ledger

| Design | Modeled assumptions | Live-only unknowns | Re-audit verdict | Deciding number |
| --- | --- | --- | --- | --- |
| Bare Strategy A far, one-position global | Global one-active-episode capital slot; K-PEG fills as passive-fill proxy; one contract per fill; carry to resolution; maker rebate; short OOS holdout. | Whether one global slot is necessary at actual capital; live fill share; queue; missed fills; edge persistence. | **CONFIRM-CLOSE** for this capital model only. | per-contract CI lower **-2.02c** |
| Source-clean/rich-short Strategy A, concurrent BTC/ETH/SOL | Per-asset capital sleeves; strict source filter; rich-short >=1c; one contract per fill or simple sizing; no PM exit; no mark-to-mid; 5% opportunity haircut only diagnostic. | Passive maker fill rate; real non-incumbent queue share; adverse selection of rich-longshot quotes; source-basis events; capacity at $10-$100 scale; whether source filter or valuation filter owns the edge. | **MERITS-LIVE-MEASUREMENT-LOOP**. | flat per-contract CI lower **+0.25c**; $50 cap lower **+0.48c**; rv-edge-scaled lower **+1.94c** |
| Same-day Arm T Tier-1 touch fade, `pos_z_ge_3/1_4h` | Saved Tier-1 held-out cell; PM touch IV rich versus HAR/Kou; buy high-priced NO / sell unlikely touch; capacity share tested at 5%, 10%, and historical 22.68% non-top3 ceiling. | Real same-day touch queue; adverse jump days; barrier-touch acceleration; whether a 9-cluster all-winner sample survives live adverse days; passive fill share near a 96c NO. | Standalone **CLOSE** remains; as a cluster member **MERITS-LIVE-MEASUREMENT-LOOP**. | raw **3.49c** CI **[2.29c, 5.43c]**; deployable **0.17c / 0.35c / 0.79c** at 5% / 10% / 22.68% |
| K6 static-hedge far family | Strict source; first eligible far fill per market; one static Binance hedge at entry delta; 6 bp Binance cost; hold hedge to resolution. | Hedge execution/slippage; fill/queue; whether far/mid survives more markets; whether live IV/vol state persists without overfitting. | **MERITS-LIVE-MEASUREMENT-LOOP** as overlay evidence, not Kronos unblock. | far all-tau lower **+0.43c**; far/mid lower **+0.96c** |
| MM structured non-top3 capacity | Structured-playbook thresholds; top-3 exclusion; entrant capture formula; median structured-wallet bps; 2026 active-day run-rate. | Real quote/cancel queue; missed fills; actual entrant share; speed versus structure; whether `other:misc_other` is reproducible; NegRisk accounting. | **MERITS-LIVE-MEASUREMENT-LOOP** for capacity/moat measurement; MM standalone base case remains sub-scale. | base **$78/day**; 1% non-top3 2026 sensitivity **$304/day** |
| politics_negrisk maker cells | Token-level PnL is modeled, but linked NegRisk accounting is missing. | Merge/split/redemption reconstruction and account-level settlement accounting. | **Blocked-pending-accounting**, not closed. | structured non-top3 **1,177 bps**, CI **[502 bps, 2,174 bps]** |

## Final Disposition

The original Strategy A closure should be narrowed:

- **CONFIRM-CLOSE:** bare far-|z| one-position-global lifecycle; 25% Kelly sizing; any claim that the current offline results are enough for a trading system.
- **MERITS-LIVE-MEASUREMENT-LOOP:** source-clean/rich-short Strategy A at small capital with concurrent BTC/ETH/SOL sleeves, the same-day Arm T Tier-1 touch fade as a cluster member, and the K6 far-family static-hedge overlay as diagnostic support.
- **Demote to diagnostic:** the borrowed 1.98c structural queue baseline.
- **Live instrumentation required:** quote placements, cancels, queue position, missed fills, fill share by top-maker rank, source-basis events, realized adverse selection, and per-market capacity at 1-contract to 10-contract size.

This re-audit does not reopen the out-of-scope closed branches: dali microstructure continuation, terminal-digital same-day Arm T/E, K2/K2v2/K2v3 quoting, or K4 arb.

## Tail & Sizing Robustness

Follow-up outputs:

- `data/analysis/csv_outputs/options_delta/od_strategy_a_tail_sample_growth.csv`
- `data/analysis/csv_outputs/options_delta/od_strategy_a_tail_adverse_regimes.csv`
- `data/analysis/csv_outputs/options_delta/od_strategy_a_tail_sizing_robustness.csv`
- `data/analysis/csv_outputs/options_delta/od_strategy_a_tail_same_day_touch.csv`
- `data/analysis/csv_outputs/options_delta/od_strategy_a_tail_source_vs_valuation.csv`
- `data/analysis/csv_outputs/options_delta/od_strategy_a_tail_terminal_daily_boundary.csv`
- `data/analysis/od_strategy_a_tail_extended_binance_history.parquet`
- `data/analysis/od_strategy_a_tail_sizing_episodes.parquet`
- `data/analysis/od_strategy_a_tail_sizing_weighted_fills.parquet`

This pass does **not** re-litigate whether the longshot-harvest finding exists. It asks how much of it survives an explicit left-tail model.

### Sample Growth

The honest split is PM-fill sample versus Binance tail sample. Binance history can grow the tail probability estimate; it cannot manufacture passive maker fills or source-clean PM episodes.

| sample | design | independent episodes/windows | markets | fills/states | read |
| --- | --- | ---: | ---: | ---: | --- |
| PM replay | bare far per-asset | 17 | 17 | 211 | broader lifecycle power diagnostic |
| PM replay | source-clean far per-asset | 11 | 11 | 149 | source-clean lifecycle grows beyond n=6/7, but still short |
| PM replay | strict-rich far per-asset | 7 | 7 | 22 | no offline PM-fill growth beyond the current Strategy A survivor |
| PM + Binance tail probabilities | strict-rich OOS Arm B | 7 | 7 | 22 | same PM fills, better tail probability estimate |
| extended cached Binance history | crypto 4h BTC/ETH/SOL, train cutoff 2026-05-27 | 35,466 | 35,466 | 1,666,845 | tail/stress base rate only, not fill evidence |
| same-day touch PM replay | Tier-1 `pos_z_ge_3/1_4h`, held-out | 6 resolution dates | 9 | 31 | original OOS cluster member |
| same-day touch PM replay | Tier-1 `pos_z_ge_3/1_4h`, all saved selected rows | 17 resolution dates | 28 | 141 | tail/sizing support; not a new OOS split |

Cached Binance 5m files contained 2025-2026 rows with microsecond-scale timestamps stored in ms-typed parquet columns; the robustness rebuild corrected those timestamps locally and used only cached data, no new capture. The 4h tail calibration remains lookahead-free by training before the 2026-05-29 PM OOS fills.

Daily terminal rows are deliberately not folded into the verdict. The available daily terminal artifact is the previously closed terminal-digital Arm E surface, not the passive Strategy A maker lifecycle. As a boundary check, the selected terminal-close rows are huge but tail-flat: **688,902 fills / 5,569 markets**, model EV **4.46c**, realized EV **0.23c**, worst fill **-99.44c**. That supports the existing terminal-digital close rather than reopening daily terminal pricing.

### Left-Tail Stress

For the 4h strict-rich short set, the parametric bad event is "the token we shorted resolves ITM." Arm B uses historical `P(resolve | signed z, tau)` from cached Binance history. The stress row takes the worst of Arm B, top-decile EWMA-vol history, and top-decile absolute 4h move history for each fill's bucket.

| 4h tail regime | fills | markets | mean bad-event probability | mean unit EV | worst unit EV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arm B conditional history | 22 | 7 | 50.58% | 4.14c | -1.41c |
| High-vol top decile | 22 | 7 | 50.59% | 4.14c | -4.50c |
| Large-4h-move top decile | 22 | 7 | 49.89% | 4.83c | -8.82c |
| Stress max per fill | 22 | 7 | 52.99% | 1.74c | -8.82c |

The observed flat replay is **not** all-winner: 3 of 7 4h episodes lose, and the worst observed flat episode is **-8.76c**. But that is still not a complete left-tail model. If every shorted token in the flat sleeve resolved ITM, the worst episode would be **-561.15c**. The stress row is therefore a tail-adjusted EV estimate, not a ruin bound.

### Tail-Aware Sizing

All rows use the same 7-market / 22-fill strict-rich source-clean PM set. "Tail EV" uses Arm B conditional probabilities. "Stress EV" uses the adverse-regime max described above. CIs are market-cluster bootstrap and net of costs/rebate; there is no mark-to-mid.

| sizing rule | weighted fills | tail EV / contract | stress EV / contract | stress EV / day | stress verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Flat one contract | 22.00 | 4.14c `[2.45c, 6.48c]` | 1.74c `[-0.39c, 5.07c]` | $0.46 `[-$0.07, $1.04]` | lower CI crosses zero |
| $50 dollar-delta cap | 9.80 | 3.03c `[1.02c, 6.70c]` | 0.55c `[-3.79c, 6.00c]` | $0.06 `[-$0.39, $0.51]` | lower CI crosses zero |
| Fair-value scaled | 15.78 | 5.79c `[3.46c, 7.05c]` | **4.19c `[0.62c, 6.66c]`** | **$0.79 `[$0.09, $1.72]`** | **survives stress** |
| 10% Kelly proxy | 3.59 | 1.56c `[0.72c, 6.17c]` | 0.32c `[-0.26c, 4.81c]` | $0.01 `[-$0.02, $0.05]` | lower CI crosses zero |
| 25% Kelly proxy | 8.79 | 1.58c `[0.73c, 6.17c]` | 0.33c `[-0.26c, 4.81c]` | $0.03 `[-$0.05, $0.12]` | lower CI crosses zero |
| 50% Kelly proxy | 12.20 | 2.16c `[1.00c, 6.17c]` | 0.52c `[-0.36c, 4.81c]` | $0.08 `[-$0.08, $0.23]` | lower CI crosses zero |

Deciding number: the largest 4h sizing rule whose adverse-regime tail-adjusted lower CI stays above zero is **fair-value scaled**, at **4.19c/stress-adjusted weighted contract**, CI **[0.62c, 6.66c]**, with stress-adjusted run-rate **$0.79/day**, CI **[$0.09/day, $1.72/day]** on the tiny 0.83-day PM replay window. Flat one-contract is positive under Arm B but does **not** survive the adverse-regime stress lower CI. The 25% Kelly row still behaves like the warning that motivated this audit: once the left tail is made explicit, its stress lower CI is below zero.

### Same-Day Touch Tail Member

The same-day touch member remains standalone **CLOSE**, but it is still part of the shared live measurement cluster. Tail-adjusting the registered `pos_z_ge_3/1_4h` cell by empirical first-passage probability instead of all-winner realized payoff gives:

| sample | fills | markets | tail EV / contract | bad-touch probability | loss given touch | 5% capacity | 10% capacity | 22.68% capacity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Held-out registered cell | 31 | 9 | 1.47c `[0.78c, 1.86c]` | 2.12% | 96.41c | 0.07c | 0.15c | 0.33c |
| All saved registered-cell rows | 141 | 28 | 1.02c `[0.86c, 1.41c]` | 1.42% | 97.55c | 0.05c | 0.10c | 0.23c |

The empirical loss rate is still **0%** in both samples, so the market-cluster CI under-samples adverse touch days. The parametric tail is the real warning: a single bad touch loses roughly **96c-98c** per contract.

### Source Versus Valuation

The source-vs-valuation test did **not** grow offline. The source-clean lifecycle only exists inside the KPEG/K6/v3 PM overlap artifacts, so the same-market diagnostic remains:

| diagnostic | markets | mean incremental | CI | read |
| --- | ---: | ---: | ---: | --- |
| `strict_rich_short - strict_source` on same 4h markets | 7 | -40.47c/episode | `[-120.93c, 1.49c]` | OD valuation filter is not proven incremental |

Arm B and the extended Binance history improve tail probabilities for the strict-rich fills, but they do not create additional PM source-clean fills. The honest ownership read is unchanged: the source-clean passive maker lifecycle/MM execution layer owns most of the proven edge; OD valuation is useful for quote selection and sizing, not yet an independently proven alpha layer.

### Tail Verdict

**CONFIRM MERITS-LIVE-MEASUREMENT-LOOP**, narrowed by sizing. The cluster still has real per-contract edge after explicit tail adjustment, but not at every size. The 4h sleeve should enter live measurement as **rv-edge-scaled sizing only**, with flat one-contract, $50 cap, and Kelly rows logged as shadow policies until they survive adverse-regime lower-CI tests. The same-day touch member can share the loop, but its live risk budget must be tiny and explicitly event/touch-gated because the offline sample has zero adverse touches.

## OD Pure-Taker Attribution

Follow-up outputs:

- `data/analysis/csv_outputs/options_delta/od_pure_taker_attribution_summary.csv`
- `data/analysis/csv_outputs/options_delta/od_pure_taker_attribution_spread_regime.csv`
- `data/analysis/od_pure_taker_attribution_trades.parquet`
- `scripts/od_pure_taker_attribution.py`

This pass removes the K-PEG/passive maker lifecycle entirely. On the captured crypto-4h one-second K6/K3 panel, the replay recomputes the v3 causal `N(z)` digital fair value, crosses the displayed best ask as a taker when the selected UP or DOWN token is cheap after the Polymarket taker fee, then carries to resolution. Non-overlap is first qualifying entry per market because the position is held to resolution. The captured panel spans only **14 strict-source crypto-4h market episodes**, so this is an attribution gate over the owned capture window, not a powered OOS production backtest.

The stale-vs-contemporaneous Binance control did not remove any candidates: the stored K6 `p_model` is already built from the one-second Binance state at the PM panel timestamp, and recomputing `N(z)` from contemporaneous `binance_spot`, strike, EWMA sigma, and seconds-to-expiry matched to displayed precision. Stale-only rejected rows were **0** in every reported cell. That means the result below is not being rescued or killed by a hidden stale-fair-value correction inside this artifact.

Primary attribution rows:

| pure-taker cell | selected markets | selected trades | mean PnL / contract | CI | median selected edge | stale-only rejects | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Strict source, any z/tau, edge > 0 | 14 | 14 | 1.95c | `[-26.04c, 26.73c]` | 0.52c | 0 | **CONFIRM-CLOSE** |
| Strict source, far `abs_z >= 1`, edge > 0 | 14 | 14 | 0.80c | `[-27.00c, 27.92c]` | 1.81c | 0 | **CONFIRM-CLOSE** |
| Strict source, far `abs_z >= 1`, edge >= 1c | 14 | 14 | 1.17c | `[-26.07c, 28.74c]` | 1.81c | 0 | **CONFIRM-CLOSE** |
| Strict source, far/late, edge > 0 | 14 | 14 | 3.01c | `[0.02c, 6.50c]` | 0.36c | 0 | statistical diagnostic only |
| Strict source, far/late, edge >= 1c | 10 | 10 | 3.23c | `[-0.97c, 7.98c]` | 1.58c | 0 | **CONFIRM-CLOSE** |

Read: the broad pure-taker OD rule and the Strategy-A-aligned far-|z| rule do **not** clear. The only pre-specified regime family with a positive lower CI is the zero-threshold far/late row, and its deciding edge is tiny: the median selected model edge is only **0.36c** and the lower CI is **+0.02c**. Once the edge threshold is raised to a minimal 1c, far/late crosses back below zero. Under [[CODEX]] Rule 4, that is "statistically cute, economically hair-thin", not a standalone OD revival.

Spread/cost diagnostics explain the shape. Far/late really is tighter than early/mid on an absolute basis: median selected-side spread is **1.00c** in `far_absz_ge1|late_lt30m` versus **3.00c** in far/mid and **7.00c** in far/early. But the executable net edge is also tiny in far/late: candidate rows have median net edge **0.17c**, and taker cost consumes about **62%** of the contemporaneous mid-edge. So the spread tightening is factored in; it leaves a small late-resolution dust edge, not enough proof that OD valuation beats the spread as a standalone strategy.

An unregistered diagnostic slice, `selected ask <= 30c`, is strongly positive in this short capture (**36.09c**, CI **[9.06c, 61.66c]** across 14 markets). It is recorded but not used as the attribution verdict because it was not the pre-registered pure-taker gate, it mixes mid-|z| and far states, and it is exactly the kind of threshold family that needs a fresh pre-registered holdout before it can reopen OD. If revisited, it should be a separate "pure-taker cheap-token/longshot" gate, not retrofitted into this attribution pass.

Assumption ledger:

| Modeled assumptions | Live-only unknowns |
| --- | --- |
| Captured one-second K6/K3 panel; displayed best ask is executable for one contract; Polymarket taker fee is paid on entry; carry to resolution with no mark-to-mid and no exit; first qualifying signal per market; `source_ok_strict` is used as a clean-label/source-risk quarantine inherited from v3. | Real order-to-fill latency and quote survival after signal; whether a tiny far/late dust edge survives live clock skew and order routing; whether a causal source-basis monitor can replace the ex-post strict-source quarantine; whether the unregistered ask<=30c diagnostic survives a fresh pre-registered sample. |

Branch fired: **CONFIRM-CLOSE for pure-taker OD attribution**. OD valuation does not currently prove it beats the spread alone on the captured crypto-4h LOB. The surviving Strategy A cluster remains a passive source-clean maker/carry measurement loop, with OD valuation as sizing/selection unless future live or fresh-capture evidence proves incremental taker edge.
