---
title: "Neutral Spike-Avoidance Market-Making under Whole-Market Nested CPCV (Task 5.1) — the Redesigned Split + Controller, Re-Run Ladder"
created: 2026-07-07
status: active
owner: justin
project: polymarket
para: project
tags:
  - market-making
  - inventory-management
  - vpin
  - adverse-selection
  - cpcv
  - overfitting
  - backtesting
---

# Neutral Spike-Avoidance Market-Making under Whole-Market Nested CPCV (Task 5.1)

> Hubs: [[strat_market_making]] · [[mm_backtesting_methodology_explainer]] · supersedes the evaluation methodology and the inventory controller of [[mm_task5_inventory_quoter_findings]] (Task 5); keeps its frozen engine, costed-PnL machinery, ladder discipline, and {Optimistic, Prob, RiskAverse} queue bracketing · PRD: [[2026-07-07_mm_task5_1_redesign_prd_reference]] · controller grounding: the LOTECH post-event study (Alvaro Fernandez, 2025-04-27 LO-USDT session) · definitions: [[glossary]] · [[polymarket_table_dictionary]]
>
> **No profitability claim until Join 2. No live trading.** Every number below is bracketed and conditional on the modeled queue; the real fill rate is a live-calibration unknown.

## Plain-English Summary

- **What this is.** Task 5 shipped "damage control, no edge" — but its verdict rested on a broken evaluation split (one calendar cut through the *same* markets, so in-sample = calm mid-life and out-of-sample = the toxic pre-resolution endgame) and a controller that answers the wrong question (a near-expiry *flatten* that pays the spread to exit, when the goal is to *carry balanced inventory but refuse one-sided stacking during informed flow*). Task 5.1 fixes both and re-runs the gated ladder on the full 17.9-day R2 capture.
- **Fix 1 — the split.** The unit of the train/test split is now the **whole market** (NegRisk event group / match): a market's entire lifecycle — calm mid-life *and* endgame — sits on one side. Cross-validation is **nested combinatorial purged CV (CPCV)** over those groups: knobs are chosen on training groups only (inner loop) and scored once on held-out groups (outer loop), across every C(6,2)=15 fold combination — many honest backtest paths instead of Task 5's single cut. Cohort features (aggressiveness, liquidity) come **only from each market's lead-in window**, so fold assignment can't peek at the outcome. τ (time-to-resolution) *conditions* the reported surface; it never splits.
- **Fix 2 — the controller.** `NeutralSpikeQuoter` keeps the microprice inventory skew (`r = microprice − k·q`) and the tight cap, drops the calendar flatten entirely, and adds a **two-lens toxicity gate** from the LOTECH study: Lens 1 = **VPIN** order-flow toxicity (volume-clock buy/sell imbalance with sweep-distance weighting, session-relative band); Lens 2 = **adverse-selection z-score** (post-fill drift vs the market's own calm baseline, z < −2, statistical not tuned). The response is **graduated**: directional flow only → stop adding on the exposed side; own fills deteriorating only → shrink size; both → widen/pull and unwind *passively*. Plus asymmetric repricing (slow to chase, fast to withdraw) and OFI-keyed continuous size dampening — each component a separately gated ladder rung.
- **The verdict.** (1) **The regime confound is confirmed and fixed**: under the whole-market split every config transfers ~1:1 from training to held-out groups (the Task-5 IS→OOS collapse is gone), and the politics baseline's honest loss shrinks from −0.96¢ to −0.13¢/contract. (2) **Politics: directionally right, unprovable** — every controller component improves the honest point estimate monotonically (full NSQ stack +0.65¢ vs baseline, the only configuration positive on every CPCV path), but at K=11 groups no delta clears the 95% group-CI: nothing ships, honestly for *power* reasons this time, not regime confusion. (3) **Esports: the first CI-certified kept rung in this research line** — A-S rung 2 (`γ=1e-4, cap=200`, derived optimal spread) beats the baseline +6.66¢/contract with group-CI [+1.77, +11.81], uniform across all 16 groups and across the whole queue bracket; its own level is ≈ 0¢ volume-pooled, so the certified claim is *avoided bleed*, not profit. (4) The overfitting apparatus now separates cleanly: PBO 0.00 (selection transfers), DSR fails (19 daily obs vs 67 trials — sample length), White's RC p ≈ 0.03 both categories (best configs are unlikely to be pure snooping).
- **Why it matters.** This is the honest re-measurement of whether inventory-managed Polymarket market-making has any edge once the regime confound is removed — and the shipped per-cohort configs are the Join-2 live-measurement candidates, not a trading system.

---

## 1. What Task 5 got wrong (and how the audit knew)

**Defect 1 — a calendar cut through the same markets.** Task 5 split at 2026-06-24 00:00 UTC: everything a market traded before was IS, after was OOS — the same markets on both sides. Because those markets marched toward late-June/July resolutions, IS was their calm mid-life and OOS their endgame (the most toxic regime, per v0's −1.01¢ near-expiry read). The tell, in `mm_task5_is_selection.csv`: the IS→OOS drop was **near-uniform across configs and scaled with the inventory cap** — an overfit signature would punish the *selected* config harder than its siblings; instead the whole surface shifted down together, and configs that carried more inventory into the endgame dropped more. That is a **regime confound**, not knob-overfitting: edge existed mid-life and was eaten near expiry, and the split happened to make OOS = the endgame. It never tested generalization to unseen markets.

**Defect 2 — flatten answers the wrong question.** Task 5's `pull_hours` exits all inventory near expiry (paying the spread), and `pull=off` carries blindly. What the maker actually needs (per the LOTECH study's course-of-action section) is to **carry balanced/small inventory to resolution while refusing to stack a one-sided book during spikes** — keep providing two-sided liquidity, and when flow turns informed, stop feeding the exposed side rather than dump. Task-5 politics never even ran `pull=off`, so flatten was never A/B'd against carrying.

*Practical example (the confound in one market).* A Musk-tweet-count token trades 06-19 → 06-30 and resolves 06-30. Under Task 5's cut, its 06-19→23 stretch (quiet, wide two-sided flow) trained the knobs; its 06-24→30 stretch (position-jockeying into resolution) judged them. *Every* config looked worse in the judgment window because the window itself was categorically more toxic — regardless of knobs. Under Task 5.1, that token's whole lifecycle is one unit: either it helps choose the knobs, or it judges them — never both, and the judgment set contains complete lifecycles of *unseen* markets.

## 2. Sample (code-verified before any run)

Full R2 clone `~/epsilon_l2_full` (`r2:epsilon-polymarket-data/parquet`), pulled fresh this session including the 7 extra days beyond 06-30:

| universe | span (UTC) | days | tokens | markets | trades |
|---|---|---|---|---|---|
| politics_negrisk | 2026-06-19 12:46 → 2026-07-07 10:59 | 17.9 | 1,261 | 663 | 541,673 |
| esports | 2026-06-19 12:46 → 2026-07-07 10:59 | 17.9 | 4,942 | 2,640 | 1,132,590 |

Eval universe = the top-32 most-traded *quotable* tokens per category (≥150 prints, average price 5–95¢ — the Task-4 rule), which map to **11 politics event groups and 16 esports groups** (vs Task 5's 6 and 7 — the added days and width nearly double the independent units). One token is excluded by a **replay-feasibility cap** declared ex-ante (>8M L2 events; the Netanyahu outright at 23.2M events needs ~10 GB RAM and ~10 CPU-hours per config sweep); its group keeps its other legs, so the split-unit count is unaffected. fee = 0 / rebate = 0 (captured truth), latency 0 ms (isolates the queue gate; esports numbers stay latency-naive upper bounds).

![Split diagnosis: whole-market folds vs the old calendar cut](../../data/analysis/plots/market_making/mm_task5_1_split_diagnosis.png)

*Read this chart:* each bar is one event group's observed lifecycle (x = calendar time); the black head is its leakage-safe lead-in window; colors are CPCV folds; the red dashed line is Task 5's 06-24 calendar cut. **What to notice:** most politics lifecycles straddle the red line — the old split put the same market's calm phase in IS and its endgame in OOS. The new folds keep every bar whole, and each fold mixes early- and late-starting, aggressive- and benign-cohort groups.

## 3. Fix 1 — the evaluation methodology (whole-market nested CPCV)

**Split unit = the whole market.** Group = the NegRisk event (politics) or match (esports), from Gamma metadata (`negRiskMarketID` per event — never assumed from siblings; resolved markets re-queried with `closed=true`).

**Nested CPCV over groups** (`mm_eval/cpcv.py`, the split generator ported from `infrastructure/walkforward/cpcv_engine.py::generate_cpcv_splits` — ported, not imported, because that module loads the crypto backtester and cross-importing is an invariant violation):

- Groups are dealt into **6 cohort-balanced folds**; every C(6,2)=15 combination of 2 held-out folds is a split; the 15 complete paths give every group exactly one honest out-of-sample estimate per path — the path distribution Task 5's single cut could never produce.
- **Inner-select / outer-estimate:** within each split, the rung's knobs are chosen by argmax of the volume-pooled costed net over the **training groups only**, and that chosen config is scored **once** on the held-out groups. A config that cheats on one group (see the unit test `test_nested_selection_never_sees_test_groups`) can win only the splits that trained on that group — and pays for it in *their* held-out scores.
- **Purge/embargo, stated honestly:** the labels here (30 s markouts, per-group costed PnL) never span group boundaries — every group is a separate engine run, and markout windows are purged at window edges inside each run. What CPCV's bar-purge protects against therefore cannot occur across groups. What *does* remain is **calendar concurrency**: politics groups trade simultaneously, so shared macro news is a residual cross-market channel that no group split can purge away. It is quantified per split in `mm_task5_1_overlap_<universe>.parquet` (mean fraction of each test group's lifespan overlapping ≥1 training group) instead of being pretended away.
- **Leakage-safe cohorts:** per group, structural features — flow rate, book depth, **sweep share** (fraction of trade volume executing beyond the touch — participant aggressiveness), spread, 1-minute mid-vol, average price, queue-turnover proxy — are computed **only from the lead-in window** (first min(24 h, 25% of observed span)). Cohort axes: `cohort_aggr` (aggressive/benign by category-median sweep share — the primary surface axis) and `cohort_liq` (thick/thin by flow). Folds are dealt stratified-snake within cohort strata, so every fold carries a comparable mix.
- **τ conditions, never splits:** each run's costed PnL is also sliced into τ-regime windows (politics: mid-life >48 h / approach 6–48 h / endgame <6 h; esports: pre >6 h / in-play <6 h) for the (cohort × τ-regime) surface. The slices share the whole-lifecycle accounting (realized + liquidation-marked carry at the executable touch, longs→bid shorts→ask, never mark-to-mid).
- **Session-aware baselines:** every toxicity reference in v0 and in the controller is calibrated per market on its own lead-in/calm window (z-scores), retiring Task-5's fixed-threshold caveat.

**Overfitting apparatus (shared `infrastructure/validation/overfitting_audit.py`, reused wholesale):** the real **CSCV PBO** (`pbo_cscv`, Bailey–Borwein–López de Prado–Zhu) with event-groups as blocks — replacing Task 5's hand-rolled approximation; **DSR** deflated by **effective** trial count (`effective_n_trials` on the config×daily-PnL matrix — fixes Task 5's crude raw-count deflation); and **White's Reality Check** (`whites_reality_check`, Hansen-studentised) added as the data-snooping gate Task 5 lacked.

*Worked example (one split).* Split #3 holds out folds {1, 4} — say groups {Fed-July, Musk-week-2} and {Iran-meeting, Lula-2026}. The inner argmax over the other 7 politics groups picks `lens[k=2e-5,cap=200,w=20]`. That config — chosen without ever seeing the four held-out groups — is then scored on their complete lifecycles. Group Fed-July's "honest ¢/contract" is the average of such scores over all splits that held it out; the keep-gate compares those per-group honest series between rungs.

## 4. Fix 2 — the controller (`NeutralSpikeQuoter`, LOTECH-grounded)

New strategy alongside the frozen protocol (`mm_engine/strategies.py`; `interfaces.py` untouched — τ, knobs, and the public-trade tape all injected via `params`). The frozen `Strategy` interface hides `last_trade` events, so the eval runner wraps the feed with `mm_eval/tape.py::tape_feed`: every public print is appended to a `TradeTape` **at the moment it is yielded** — exactly the stream a live quoter's websocket delivers, strictly causal (unit-tested).

Components, each a separately gated rung:

1. **Core (kept from v1):** reservation `r = microprice − k·q`, tight cap with passive reduce-only quoting at cap. The LOTECH study independently validates microprice as fair value (MAE 0.75 ticks vs mid's 8.82 — ~12× closer, holding through spike and post-sweep regimes). **No calendar behavior at all** — the near-expiry flatten is gone.
2. **Lens 1 — VPIN order-flow toxicity (leading).** Volume clock: bucket volume = EWMA typical trade size × 25 (session-relative by construction); per closed bucket, buy/sell imbalance `|B−S|/(B+S)`; VPIN = mean over the last `w` buckets (`w` ∈ {20, 50}, a tuned knob). Trades are weighted by the LOTECH sweep rule — `exp(min(dist_ticks, 6) × 0.5)`, a bounded variant of the study's `exp(dist/tick)` — so a print 5 ticks through the book counts ~12× a touch trade. The elevated band is the **session's own 0.90-quantile** (plus a 0.30 absolute floor, below the literature's 0.4 "elevated"), never a fixed cross-market constant. Direction: net signed flow over the window names the **exposed side** (net buying → our asks are being consumed).
3. **Lens 2 — adverse-selection z-score (confirming).** On each own fill (detected from the inventory delta the frozen interface already passes), measure the post-fill mid drift at 30 s; `signed_drift = side·(mid_{t+30s} − mid_at_fill)` (negative = adverse); z against the rolling mean/std of the last 50 calm-period drifts; **flag at z < −2 — a statistical constant, never tuned**. The baseline only updates while no flag is active, so a spike cannot normalize itself into its own reference.
4. **Graduated response (the anti-"sell-sell-sell" core):** directional only → **suspend adding on the exposed side** (the other side keeps quoting, which also passively reduces exposed inventory); adverse only → halve opening size; **both** → pull opening quotes and rest a reduce-only quote at 2× the half-spread from the reservation — **passive unwind in measured tranches, never a dump at the touch, never crossing**. Balanced inventory may carry to resolution.
5. **Asymmetric repricing (LOTECH idea 1):** opening quotes chase a moving market at ≤0.5 ¢/s but withdraw instantly — chasing a pump with a passive bid is the adverse-selection trap; the fill you exit fast is the one in a reversing market.
6. **OFI size-dampening (LOTECH idea 2, keyed on Cont-style order-flow imbalance):** the pressured side's opening size shrinks continuously with `tanh(OFI_ewma/3·scale)` (coefficient ∈ {0.3, 0.6}, tuned; floor 0.2×) — exposure fades during one-sided flow without any hard threshold.

**Knob discipline:** tuned = `k`, `cap`, VPIN window `w`, damp coefficient (and the A-S γ on its rungs). Everything else — z-threshold, sweep weighting, bands, chase rate, widen multiple — is a declared constant listed in `NSQ_DEFAULTS`.

*Practical example (a spike).* Buyers start sweeping an outcome token: prints land 3–6 ticks through the ask, volume buckets close fast, VPIN spikes over its session band → `directional_flag`, exposed side = SELL. The quoter stops re-arming its ask (no more short-stacking into the pump) but keeps its bid resting below (slow-chasing at ≤0.5 ¢/s under asymmetry). If our few fills then mark out badly (drift z < −2), both lenses agree → the remaining quote widens to reduce-only and inventory unwinds passively as the book comes back. Task 5's v1 would instead have quoted both sides until the cap, then (politics) dumped everything at the touch 2 h before expiry.

## 5. v0 re-grounded — where is toxicity actually net-negative?

Re-derived on the full sample, whole-market view, session-aware baselines (`scripts/mm_task5_1_v0_attribution.py`). Baseline fills (symmetric quoter, RiskAverse, 0 ms) mapped per (cohort_aggr × τ-regime) cell with market-cluster CIs and leave-one-market-out re-pools.

**Pre-registered wiring rule (written before the numbers ran):** the two-lens gate + size-dampening enter a category's gated ladder iff ≥1 cell has qty-weighted markout(30 s) point < 0 AND LOMO-negative ≥ 50%. Failing cells stay reported as UNSUPPORTED and the shipped per-cohort config withholds the defensive knobs there.

**Toxicity surface** (baseline fills; markout(30 s) qty-weighted ¢/contract; CI = market-cluster bootstrap; cells with <10 fills omitted):

| universe | cohort | τ-regime | fills | markets | markout ¢ | 95% CI | adverse ¢ | LOMO-neg | net-negative? |
|---|---|---|---|---|---|---|---|---|---|
| politics | benign | **endgame (<6 h)** | 1,359 | 11 | **−0.64** | [−1.02, **−0.05**] | −1.20 | **100%** | **YES — CERTIFIED** |
| politics | benign | approach (6–48 h) | 3,859 | 17 | +0.13 | [−0.03, +0.26] | −0.33 | 0% | no |
| politics | benign | mid-life (>48 h) | 2,619 | 16 | +0.35 | [+0.29, +0.42] | −0.06 | 0% | no |
| politics | aggressive | mid-life (>48 h) | 8,384 | 13 | +0.32 | [+0.20, +0.46] | −0.25 | 0% | no |
| esports | aggressive | in-play (<6 h) | 26,737 | 22 | +0.20 | [−0.00, +0.40] | −0.36 | 0% | no |
| esports | aggressive | pre (>6 h) | 554 | 18 | +0.41 | [+0.29, +0.62] | −0.12 | 0% | no |
| esports | benign | in-play (<6 h) | 7,734 | 10 | +0.20 | [−0.39, +0.76] | −0.47 | 0% | no |
| esports | benign | pre (>6 h) | 124 | 9 | +0.46 | [+0.35, +0.61] | −0.12 | 0% | no |

*Column glossary:* `markout ¢` = net per-contract edge to mid at 30 s (spread captured + post-fill drift); `adverse ¢` = the pure post-fill drift component (negative = the market moved against the fill); `LOMO-neg` = share of leave-one-market-out re-pools where the cell's markout stays negative; `net-negative?` = the pre-registered wiring criterion (point < 0 AND LOMO ≥ 50%).

**Wiring outcome:** politics → `wire_defensive_knobs = TRUE` (one supported cell: **benign × endgame**); esports → `FALSE` (no supported cell — every cell positive). The esports lens/asym/damp rungs therefore ran as **diagnostics outside the gate** and cannot ship, exactly as pre-registered.

![v0 toxicity surface](../../data/analysis/plots/market_making/mm_task5_1_v0_surface.png)

**Lens-aligned counterfactual rescue** (per token, recompute qty-weighted markout excluding flagged fills; positive delta = the flag isolates toxic fills):

| universe | signal | mean rescue ¢ | median ¢ | qty flagged | note |
|---|---|---|---|---|---|
| politics | **as_z < −2 (Lens 2)** | **+0.22** | +0.15 | **7%** | best rescue per unit flagged |
| politics | mid-vel (Task-5's gate) | +0.11 | +0.02 | 22% | works, but blunt |
| politics | sweep_rel (Lens 1) | +0.07 | +0.02 | 27% | weak alone |
| politics | aggressor size | −0.04 | +0.01 | 40% | still fails as a skip-gate |
| esports | **as_z < −2 (Lens 2)** | **+0.59** | +0.65 | **6%** | strongest signal anywhere |
| esports | mid-vel | +0.14 | +0.02 | 18% | |
| esports | sweep_rel (Lens 1) | −0.05 | −0.07 | 22% | in-play sweeps are benign flow |
| esports | aggressor size | +0.07 | +0.07 | 39% | |

**Read.** Three things ground the controller design. (1) The Task-5 near-expiry story survives the whole-market re-derivation and gets *sharper*: politics toxicity is concentrated in the **benign-cohort endgame** — the quiet weeklies (Musk-tweet-type) whose expiries get picked off — and is now CI-certified, not merely directional. The aggressive/thick cohort (the long-dated Fed/Iran outrights) shows no net-negative cell in this capture (its endgames mostly lie beyond the window — stated, not hidden). (2) Esports is positive in every cell — an in-play book is not "toxic near expiry", confirming Task-5's pre-registered withholding with cohort granularity. (3) The **AS z-score (Lens 2) is the most efficient toxicity flag ever measured on this telemetry** — +0.22¢/+0.59¢ mean rescue while flagging only ~6–7% of volume, beating Task-5's shipped velocity gate on both power and precision, and it is exactly the session-aware, statistically-thresholded signal the controller's confirming lens implements. Lens 1 alone is weak (politics) or mildly harmful (esports in-play) as a *skip* rule — supporting its LOTECH role as the *leading/direction-naming* lens in a graduated response rather than a standalone gate.

## 6. The ladder (nested-CPCV honest OOS, bracketed)

**How to read the table.** One row per rung per category. `honest_pooled_c` = the volume-pooled costed ¢/contract over every group's held-out estimates (each group scored only by configs selected without seeing it, pessimistic queue). `path mean [p10, p90]` = the distribution of that number over the 15 complete CPCV paths — the spread Task 5's single split could not show. `modal config` = the config most often inner-selected across splits (what would ship); its pooled ¢ re-run under all three queue models is the bracket. `Δ vs prev kept [95% CI]` = the keep-gate: per-group paired honest delta vs the previous kept rung, bootstrap over groups. A rung marked DIAGNOSTIC ran outside the gate (v0-unsupported defensive knobs) and cannot ship.

### politics_negrisk (11 whole-market groups; defensive rungs GATED per v0)

| rung | modal config | honest pooled ¢ | path mean [p10, p90] | Δ vs baseline ¢ [95% CI] | keep |
|---|---|---|---|---|---|
| baseline | symmetric | −0.13 | −0.19 [−0.31, −0.07] | — | KEPT (baseline) |
| NSQ core | core[k=5e-6,cap=500] | −0.14 | −0.14 [−0.18, −0.10] | +0.31 [−0.82, +1.34] | DROP — FRAGILE |
| + two-lens | lens[k=5e-6,cap=500,w=50] | −0.06 | −0.07 [−0.13, −0.01] | +0.36 [−0.77, +1.37] | DROP — FRAGILE |
| + asymmetry | asym[k=5e-6,cap=500,w=20] | **+0.27** | +0.25 [+0.13, +0.32] | +0.64 [−0.52, +1.71] | DROP — FRAGILE |
| + size-dampen | damp[k=5e-6,cap=500,w=20,d=0.6] | **+0.29** | +0.27 [+0.13, +0.35] | **+0.65 [−0.49, +1.69]** | DROP — FRAGILE |
| A-S rung 1 | rung1[γ=1e-5,cap=500] | −0.23 | −0.22 [−0.42, −0.11] | +0.21 [−0.83, +1.11] | DROP — FRAGILE |
| A-S rung 2 | rung2[γ=1e-5,cap=500] | −0.19 | −0.19 [−0.23, −0.16] | +0.30 [−0.78, +1.22] | DROP — FRAGILE |
| A-S rung 3 | rung3[γ=1e-5,cap=200] | −0.10 | −0.10 [−0.13, −0.07] | +0.16 [−1.21, +1.31] | DROP — FRAGILE |
| basket-carry | basket[k=5e-6,cap=200] | −0.21 | −0.18 [−0.25, −0.11] | +0.30 [−0.87, +1.34] | DROP — FRAGILE |

### esports (16 whole-market groups; defensive rungs DIAGNOSTIC per v0)

| rung | modal config | honest pooled ¢ | path mean [p10, p90] | Δ vs prev kept ¢ [95% CI] | keep |
|---|---|---|---|---|---|
| baseline | symmetric | −3.45 | −3.34 [−3.68, −2.84] | — | KEPT (baseline) |
| NSQ core | core[k=2e-5,cap=200] | −0.03 | −0.04 [−0.05, −0.03] | +4.70 [−1.21, +10.80] | DROP — FRAGILE |
| + two-lens | lens[k=5e-6,cap=200,w=20] | −0.04 | −0.04 [−0.05, −0.04] | +4.70 [−1.20, +10.76] | DIAGNOSTIC (v0-unsupported) |
| + asymmetry | asym[k=2e-5,cap=500,w=20] | +0.27 | +0.27 [+0.22, +0.32] | +5.33 [−0.76, +11.69] | DIAGNOSTIC (v0-unsupported) |
| + size-dampen | damp[k=2e-5,cap=500,w=20,d=0.3] | +0.34 | +0.34 [+0.28, +0.41] | +5.53 [−0.68, +12.04] | DIAGNOSTIC (v0-unsupported) |
| A-S rung 1 | rung1[γ=1e-4,cap=200] | −0.05 | −0.06 [−0.07, −0.06] | +4.71 [−1.20, +10.78] | DROP — FRAGILE |
| **A-S rung 2** | **rung2[γ=1e-4,cap=200]** | **+0.03** | +0.02 [−0.00, +0.03] | **+6.66 [+1.77, +11.81]** | **KEPT — beats baseline, certified** |
| A-S rung 3 | rung3[γ=1e-5,cap=200] | +0.09 | +0.09 [+0.06, +0.12] | +0.69 [+0.06, +1.44] vs rung2 | DIAGNOSTIC (v0-unsupported)¹ |
| basket-carry | basket[k=2e-5,cap=200] | +0.06 | +0.06 [+0.05, +0.06] | −1.83 [−4.92, +0.15] vs rung2 | DROP — worse than rung2 |

¹ Pre-registered tension, reported not shipped: the toxicity overlay *would* add +0.69¢ [+0.06, +1.44] on top of rung 2 — a certified-looking increment — but the v0 gate found no net-negative toxicity cell in esports, so the defensive knob stays withheld. This mirrors Task-5's rescue-table tension and is a Join-2 live question, not a backtest claim.

**Queue bracket (modal configs, pooled ¢/contract; RA = RiskAverse):** the bracket is tight everywhere — nothing above is a queue-assumption artifact. Politics: baseline −0.13/−0.05/−0.10 (RA/Prob/Opt), NSQ damp **+0.34/+0.34/+0.34**, NSQ asym +0.32/+0.32/+0.32, lens +0.01/+0.02/+0.01, A-S rung 2 −0.15/−0.15/−0.15. Esports: baseline −3.45/−3.23/−3.19, **rung 2 +0.04/+0.04/+0.04**, damp +0.42/+0.41/+0.42, rung 3 +0.12/+0.12/+0.12. Full table: `mm_task5_1_ladder_table.csv`.

**Read (what the ladder says).**

1. **The regime confound is gone, and the baseline verdict changes with it.** Under whole-market honest OOS the politics symmetric baseline loses only −0.13¢/contract (path spread [−0.31, −0.07]) — Task 5's "−0.96¢ OOS" was the endgame window, not the market. Esports baseline still bleeds (−3.45¢): its inventory-bet failure mode is real in every split design.
2. **Politics: the controller improves the point estimate monotonically, exactly in component order** — core +0.31 → +lenses +0.36 → +asymmetry +0.64 → +damping +0.65¢ vs baseline — and the full NSQ stack is the only configuration with *positive* honest OOS (+0.29¢ pooled; every CPCV path positive, p10 +0.13). But at K=11 groups no delta clears the 95% group-CI: **all rungs read FRAGILE, nothing ships.** The honest conclusion is "directionally right, unprovable at this group count" — the power problem Task 5 had, now measured on a split that could in principle have certified it.
3. **Esports: the ladder keeps A-S rung 2 — the first CI-certified rung in this research line.** The derived optimal spread (γσ²τ/2 + arrival term, k_arr from the lead-in) quotes wider and more selectively than the hand-tuned quoters; its improvement is *uniform across all 16 groups* (+6.66¢ mean delta, CI [+1.77, +11.81]), where the raw skew quoter's larger-looking +4.70¢ was whale-group-skewed and failed the CI. **This flips Task-5's "A-S certified worse" verdict** — that result was an artifact of the calendar split (wide quotes barely filled inside the calm IS window; whole-lifecycle accounting rewards their selectivity).
4. **Level vs delta, stated plainly:** rung 2's own honest level is +0.03¢/contract volume-pooled (≈ zero economically) and +1.8¢ equal-weighted per group — the certified claim is *damage avoidance vs the baseline* (+6.7¢/contract of avoided bleed), plus a small positive carry on the thin groups. **No profitability claim; Join-2 measures the level live.**
5. **Basket-carry loses to rung 2 head-to-head again** (−1.83¢ [−4.92, +0.15]) — partial-partition carry remains a slower inventory bet, consistent with Task 5.

![Kept-rung performance surface](../../data/analysis/plots/market_making/mm_task5_1_surface_heatmap.png)

*Read this chart:* costed ¢/contract of the kept rung's modal config per (cohort × τ-regime) cell, pessimistic queue, pooled over groups; cell labels show the group count. **What to notice — politics** (kept = baseline, so this is the *market's* regime structure): the benign cohort earns **+2.66¢ in the approach window (6–48 h out)** and gives back **−3.13¢ in the endgame (<6 h)** — the Task-5 thesis ("edge in mid-life, eaten near expiry") drawn directly from held-out data; the aggressive cohort (long-dated outrights) is mildly negative mid-life (−1.11¢) with no endgame in this capture. This is exactly the surface the single pooled OOS number of Task 5 hid, and it maps where a live loop should and should not quote. **Esports** (kept = A-S rung 2): small positives concentrate in the aggressive cohort pre-match (+0.39¢); everything else ≈ 0 — the certified improvement is bleed-avoidance, not concentrated regime alpha.

![Inner→outer transfer by cap](../../data/analysis/plots/market_making/mm_task5_1_inner_outer_by_cap.png)

*Read this chart:* each dot is one config — x = its pooled ¢/contract on CPCV training groups (mean over splits), y = the same on held-out groups; color = inventory cap. **What to notice:** every dot sits on the diagonal in both categories — the near-uniform, cap-scaled IS→OOS collapse that indicted Task 5's split (`mm_task5_is_selection.csv`) has vanished once whole markets are held out together. Configs generalize ~1:1; the split, not the strategies, made Task-5's numbers fall apart. The baseline (yellow) sits below the diagonal — the one configuration whose inventory bet makes even honest transfer noisy.

## 7. Overfitting audit (shared `overfitting_audit` infra; pessimistic grid, final)

| gate | politics | esports | read |
|---|---|---|---|
| **PBO** (real CSCV, blocks = group-runs) | 0.00 (4 blocks, sens {4: 0.00}) | 0.00 (8 blocks, sens {4: 0.00, 6: 0.05, 8: 0.00}) | IS-best configs stay top-half OOS — selection is not overfitting at the config level. Politics' 4-block version has few combinations (directional); esports' is solid. |
| **DSR** (kept config, 19 daily obs, deflated by n_eff of 67 trials) | fails by construction (kept = baseline, SR −1.06) | **fails**: SR_ann +2.92 vs SR* 18.2, n_eff 43.2 → dsr_p ≈ 0.0002 | 19 daily observations against a ~43-effective-trial haircut cannot clear — the honest cost of a 67-config search on 18 days, same shape as Task 5's DSR verdict. |
| **White's RC** (studentised, daily PnL, 2000 boots) | **p = 0.032** (raw 0.49) | **p = 0.036** (raw 0.59) | The *best* config's true mean daily PnL is > 0 after the data-snooping correction in BOTH categories — the first snooping-adjusted positive this line has produced. Caveat: daily obs are cross-market correlated (not group-clustered), so this is supporting evidence, not the gate. |
| **Concurrency overlap diagnostic** | mean test-lifespan overlap with train = **1.00** | 0.60 | The politics residual channel is maximal (all groups concurrent — shared macro news cannot be purged by any market split); esports lifecycles are ~40% disjoint. Stated per realism rule 3, not hidden. |

*Read:* the apparatus now separates cleanly: **PBO says the selection transfers** (unlike Task 5's coin-flip 0.5), **DSR says the sample is still too short to certify a Sharpe** after deflating for 67 trials, and **White's RC says the best configs are unlikely to be pure snooping**. Together with the group-CI gate: config-level selection skill is real but the *economic level* remains uncertified — exactly the "merits live measurement, not a trading system" disposition.

## 8. The spike, visually (charts 4–5)

![Toxicity trace through the spike](../../data/analysis/plots/market_making/mm_task5_1_toxicity_trace.png)

*Read this chart:* the sample's largest politics spike (mid 0.18 → 0.75 in minutes, ~22:50 UTC); top = mid, middle = Lens 1 (volume-clock VPIN, red dots = `directional_flag` firing), bottom = Lens 2 (post-fill drift z vs the calm baseline, red dashed = the −2 statistical threshold). **What to notice:** VPIN jumps from ~0 to 0.57 at ~22:20 — **half an hour before the main move** — and stays elevated (the leading property the VPIN literature claims); Lens 2 confirms during the move itself (z reaching −13). The conjunction — the full defensive response — fires exactly through the toxic window, replicating the LOTECH post-event finding ("both lenses diverging simultaneously and monotonically... well before the sweep") on Polymarket data. One illustrative episode, not evidence — the statistics live in the ladder table.

![Inventory path: baseline vs NeutralSpikeQuoter](../../data/analysis/plots/market_making/mm_task5_1_inventory_path.png)

*Read this chart:* net position (left axis) through the same spike; grey = mid (right axis). **What to notice:** the symmetric baseline (red) sells into the entire rally — stacking from −750 to **−1,800 contracts short** as the price runs from 0.18 to 0.75, the exact "one-sided book into informed flow" failure mode the LOTECH session documented. The NeutralSpikeQuoter (blue, lenses on) holds between −270 and +250 and crosses ~flat through the move — it suspends the exposed side, keeps quoting the other, and carries a balanced book to the far side of the spike. Damage avoidance by construction, not by calendar.

## 9. Assumption ledger (brain/CODEX.md realism rules)

**Modeled assumptions:** queue bracket {Optimistic, Prob(0.5), RiskAverse} stands in for the unknown fill rate (grid selection pessimistic-only; modal configs bracketed); latency 0 ms (fair politics, optimistic esports); liquidation marks at the last observed touch (depth-of-exit unmodeled — flatters *large* terminal inventories, i.e. the baseline, not the controlled configs); NSQ declared constants (`NSQ_DEFAULTS`: volume-clock ×25, sweep λ=0.5 cap 6 ticks, session band q=0.90 floor 0.30, z<−2, 50-fill calm baseline, 0.5 ¢/s chase, 2× widen, 0.2 size floor) grounded on LOTECH/VPIN literature, not tuned per config; the replay-feasibility cap (>8M events, 1 token); `end_date` = nominal Gamma deadline (mid-life τ "large" for the long outrights); cohort medians computed within-category on lead-in windows; A-S `k_arr` calibrated per token on the lead-in window only.

**Live-only unknowns (Join 2):** true passive fill rate + queue position; whether VPIN/AS-z fire early enough at live latency; real capacity at the touch; adverse selection of the *controller's own* (post-gate) fills; persistence of the cohort structure out of this 17.9-day capture.

**Statistical honesty:** K = 11/16 groups → group-bootstrap CIs are approximate/directional; PBO at these K uses few CSCV combinations (politics especially) and is directional; the concurrency overlap diagnostic shows the residual cross-market channel the split cannot remove (politics 1.00 — fully concurrent; esports 0.60).

**Materiality (rule 4):** the deployable per-contract claim is **nil in both categories**. Politics' best configuration nets +0.29¢/contract honest-pooled — statistically uncertified and, at Task-5-scale fill volumes, single-digit dollars per day before any capacity haircut. Esports' certified rung nets ≈ +0.03¢/contract volume-pooled (its +1.8¢ equal-weighted mean lives in thin, low-capacity groups); the certified quantity is the **+6.7¢/contract of avoided baseline bleed**, which is a risk-control result, not revenue. Statistical survival ≠ economic materiality — both categories remain "merits a live MEASUREMENT loop," not "merits a trading system."

## 10. Decision and next step

**Gate outcome, per category:**

- **politics_negrisk — no rung ships.** Every controller component improves the honest OOS point estimate in design order (the full NSQ stack, `damp[k=5e-6, cap=500, w=20, d=0.6]`, is the only configuration positive on every CPCV path: +0.29¢ pooled, bracket-tight +0.34¢ across all three queue models), but at K=11 groups no delta certifies. **Join-2 live-measurement candidate: the full NSQ stack**, quoting per the surface — the benign cohort's approach window (+2.66¢) is where the edge lives; the benign endgame (−3.13¢, certified toxic in v0) is where the two-lens gate must prove itself live.
- **esports — A-S rung 2 ships as the measurement config:** `rung2[γ=1e-4, cap=200, A-S spread, k_arr from lead-in]`, the first CI-certified kept rung in this research line (+6.66¢ vs baseline, [+1.77, +11.81], uniform across 16 groups and the whole queue bracket). Its own level ≈ 0¢ volume-pooled: what is certified is bleed-avoidance. **Live A/B to run at Join 2: rung 2 vs rung 2 + two-lens overlay** — the overlay's +0.69¢ [+0.06, +1.44] increment is pre-registeredly withheld (v0 found no net-negative toxicity cell in esports) and only live fills can resolve that tension.

**What changed in the map.** (1) Task-5's "no edge" verdict is *re-attributed*: the IS→OOS collapse was the split, not the strategies — under whole-market holdout every config transfers ~1:1 and the tell (drop scaling with cap) is gone. Task 5's costed-lens conclusion stands (never quote the uncontrolled baseline), but its A-S condemnation is **reversed** for esports and its politics numbers are superseded by the surface. (2) The methodology is now durable: whole-market nested CPCV + leakage-safe cohorts + the real audit stack (`mm_eval/cpcv.py`) is the standing evaluation harness for any future Polymarket MM work — nothing about it is specific to this controller. (3) The LOTECH mechanics are validated on PM data at diagnostic level: microprice-anchored quoting, VPIN's lead property, the AS z-score's efficiency (v0's best rescuer), and the one-sided-stacking failure mode are all visible in this capture. (4) The near-expiry problem is now *located*: benign-cohort endgames, −3.13¢/contract — a flow-triggered gate has a concrete target, and a calendar flatten remains the wrong tool.

**Concrete next step:** Join-2 one-contract live loop — politics NSQ-stack on VIABLE benign-cohort markets (quote mid-life/approach; the gate handles the endgame), esports rung-2 vs rung-2+overlay A/B — to collapse the queue bracket with real fill rates, calibrate `k_arr`/VPIN bands on live sessions, and measure the level the backtest cannot certify. **No sizing decision from this note.**

## Adversarial self-check (where would this be wrong?)

1. **The group-independence assumption is the weakest politics claim.** The overlap diagnostic reads 1.00 — every held-out politics group trades concurrently with training groups, and shared macro shocks (a Musk news day moves several groups) violate the bootstrap's independence assumption. That makes the politics CIs, if anything, too *narrow* — which strengthens the FRAGILE (don't-ship) verdicts but would also erode a future politics "keep". Esports (overlap 0.60, sequential matches) is structurally safer, which is part of why its certification is more credible.
2. **The esports certified keep could be a baseline-badness result, not a rung-2-goodness result.** The +6.66¢ delta is measured against a baseline that demonstrably bleeds; any inventory-controlled config gets most of that delta (core: +4.70¢). What distinguishes rung 2 is *uniformity* (per-group consistency tightening the CI), which is real but subtler than the headline. Counter-consideration: rung 2 also beat the previously-kept chain, not just the baseline, and basket-carry lost to it head-to-head.
3. **Reflexivity of the v0 wiring rule.** Toxicity was mapped on *baseline* fills; a different quoter changes which fills occur. The withheld esports defensive knobs might be supported under the NSQ's own fill distribution — rung 3's +0.69¢ [+0.06, +1.44] increment over rung 2 hints exactly at this. Pre-registered discipline kept it out; the Join-2 A/B is the honest resolution.
4. **`k_arr` calibration leans on thin lead-ins.** Rung 2's spread needs the arrival decay; esports lead-in windows are pre-match (low flow), so live `k_arr` could differ materially from the lead-in estimate. Live recalibration is mandatory before trusting the derived spread.
5. **Latency 0 ms flatters esports most.** The certified esports rung is a latency-naive upper bound; an in-play snipe faster than our cancel could erase it. This was true of every Task-4/5 esports number and remains a Join-2 unknown.
6. **Lens-1 warm-up is a real hole.** The session-relative VPIN band needs ~30 buckets; a spike in a market's first hour is unprotected (the LOTECH event itself happened minutes into a session). Live mitigation — seeding the band from the lead-in window — is designed but untested.
7. **White's RC borrows significance from correlated days.** Its p≈0.03 treats 19 daily observations as exchangeable; same-day cross-market correlation inflates the effective evidence. It is supporting color, never the gate — the group-CI is.
8. **The replay-feasibility cap removed one whale.** Declared ex-ante and group-preserving, but if 20M-event markets differ systematically (deeper incumbency, tighter spreads), the universe tilts toward smaller books. One token of 64; the direction of any bias is unknown.
9. **Restart hygiene:** the run was killed and resumed four times (memory management + an overnight pause); determinism was verified — the resumed aggregation reproduced identical verdicts from cache — and every cell is hash-keyed by (token, config, queue), so no stale-config contamination is possible.

## Reproduce / artifacts

- **Setup:** `PYTHONPATH=. uv run python scripts/mm_task5_1_setup.py --top-k 32` (coverage verification, token selection, Gamma metadata with the closed-markets retry, lead-in cohorts, folds).
- **v0:** `… scripts/mm_task5_1_v0_attribution.py --workers 6` → surface, wiring rule, lens-aligned rescue.
- **Ladder:** `… scripts/mm_task5_1_ladder_run.py --workers 7` (cells cached per token×config×queue — crash-resumable; `--prewarm` fills the cache without aggregating).
- **Charts:** `… scripts/mm_task5_1_charts.py`.
- **Tests:** `PYTHONPATH=. uv run pytest tests/test_mm_task5_1.py` (21) — tape causality, both lenses, graduated response, no-flatten, asymmetric reprice caps, OFI damping, ported CPCV shapes/purge/paths, nested-selection honesty, cohort-balanced folds, costed-span additivity, regime windows. Existing suites unaffected (`test_mm_task5.py` 22, `test_mm_eval.py` 15).
- **Code:** `mm_engine/strategies.py::NeutralSpikeQuoter` (+`NSQ_DEFAULTS`), `mm_eval/tape.py`, `mm_eval/cpcv.py`, the three `mm_task5_1_*` scripts. `mm_engine/interfaces.py` untouched.
- **Artifacts** (`data/analysis/csv_outputs/market_making/`): row-heavy machine data is **Parquet** — `mm_task5_1_{groups, groups_honest, splits, paths, surface, v0_rescue, audit, overlap_politics_negrisk, overlap_esports}.parquet`; small human-read summaries stay **CSV** — `mm_task5_1_{ladder_table, v0_surface}.csv`; plus `mm_task5_1_v0_wiring.json`. (Parquet identifiers — `token_id`, `group_id`, `market` — are stored as strings, matching the engine's verbatim-CLOB-id convention; read with `duckdb`/`pd.read_parquet`.)
- **Plots:** `data/analysis/plots/market_making/mm_task5_1_{split_diagnosis, v0_surface, surface_heatmap, inner_outer_by_cap, toxicity_trace, inventory_path}.png`.
- Deterministic: seeded bootstraps, deterministic engine, event-stream-only strategy state.

## Cross-links

Supersedes (methodology + controller): [[mm_task5_inventory_quoter_findings]]. Baseline/verdicts: [[mm_symmetric_quoter_validation_findings]] (Task 4). Design inputs: [[mm_market_screen_and_ttr_regime_findings]]. Engine: [[mm_backtesting_methodology_explainer]] · [[mm_join1_reconciliation_findings]] · [[mm_engine_queue_models]]. NegRisk structure: [[mm_politics_negrisk_accounting_findings]]. Live loop this feeds: [[mm_politics_negrisk_live_loop_design]] · Join-2 calibration. Hub: [[strat_market_making]] · [[COWORK]].
