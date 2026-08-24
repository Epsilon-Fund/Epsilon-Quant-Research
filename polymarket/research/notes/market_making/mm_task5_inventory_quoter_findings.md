---
title: "Inventory-Managed Market-Making on Polymarket (Task 5 + 5.1) — From a Confounded Ladder to Whole-Market Nested CPCV"
created: 2026-07-07
updated: 2026-07-08
status: active
owner: justin
project: polymarket
para: project
hubs:
  - strat_market_making
  - mm_backtesting_methodology_explainer
tags:
  - market-making
  - inventory-management
  - avellaneda-stoikov
  - vpin
  - adverse-selection
  - cpcv
  - overfitting
  - backtesting
---

# Inventory-Managed Market-Making on Polymarket (Task 5 + 5.1)

> Hubs: [[strat_market_making]] · [[mm_backtesting_methodology_explainer]]. Builds on [[mm_symmetric_quoter_validation_findings]] (Task 4 baseline) and [[mm_market_screen_and_ttr_regime_findings]] (screen + τ-regime). Engine lock: [[mm_join1_reconciliation_findings]]. **Full Task-5.1 technical record:** [[mm_task5_1_neutral_quoter_cpcv_findings]]. PRDs: [[2026-07-07_mm_task5_prd_reference]] · [[2026-07-07_mm_task5_1_redesign_prd_reference]]. Next steps + manual audit: [[2026-07-08_mm_task5_next_steps_and_manual_audit]].
>
> This note consolidates the whole arc as one paper: Task 5 (first inventory-managed quoter + costed gated ladder) and Task 5.1 (the redesign that fixed a broken evaluation split and the controller). Explanations use **STAR** (Situation · Task · Action · Result). **No profitability claim — measurement only until Join 2. No live trading.**

---

## Summary

Plain-English, before the dense abstract below:

- **What this note is.** The consolidated write-up of MM Task 5 and Task 5.1 — building a market-making quoter for Polymarket that actively manages inventory, and testing it honestly (real exit costs, no mark-to-mid inflation).
- **Why it was written.** Task 4 had shown the earlier symmetric fixed-spread quoter's apparent "profit" was really a **directional inventory bet**, not spread capture. Task 5 built the fix (skew against inventory, cap it, flatten politics before expiry); Task 5.1 then fixed a **broken evaluation split** and redesigned the controller into `NeutralSpikeQuoter` (a two-lens VPIN + adverse-selection toxicity gate).
- **What it covers.** A costed, gated ladder of configurations judged under **whole-market nested combinatorial-purged cross-validation**, on politics-NegRisk markets (11 groups) and esports markets (16 groups).
- **One-line takeaway.** **Politics** is directionally right but not certifiable at this sample (OOS group-CI spans zero); **esports** A-S rung 2 is the first CI-certified rung in this line (**+6.66¢ [+1.77, +11.81]** vs baseline) but its own level ≈ 0¢, so what is certified is **bleed-avoidance, not profit**. Deployable per-contract edge is **nil on both sides** → this **merits a live measurement loop, not a trading system**. No profitability claim, no live trading until Join 2.

---

## Abstract

Task 4 showed the symmetric fixed-spread quoter's "profit" is a **directional inventory bet**. Task 5 built the fix — a quoter that skews against inventory, caps it, and (politics) flattens before expiry — and judged it on a **costed gated ladder** (realized round-trips + inventory marked at the *executable touch*, never mark-to-mid). Charging exit costs, the baseline **loses** out-of-sample in both categories and inventory control is the only rung kept, but with **no provable edge**. That verdict rested on a **broken split** — a single calendar cut through the *same* markets, so in-sample was calm mid-life and out-of-sample the toxic pre-resolution endgame (tell: the IS→OOS drop was near-uniform across configs and scaled with the cap → a regime confound, not overfitting). **Task 5.1 fixed the test and the controller.** The test is now **whole-market nested combinatorial-purged CV**; the controller, `NeutralSpikeQuoter`, drops the flatten for a **two-lens toxicity gate** (VPIN + adverse-selection z-score) driving a **graduated response**, grounded in the LOTECH study.

**Headline results.** (1) Confound gone — configs now transfer ~1:1 IS→OOS, and the politics baseline's honest loss falls from **−0.96¢ to −0.13¢/contract**. (2) **Politics: directionally right, not certifiable on this sample** — every controller component improves the in-sample estimate monotonically (full stack **+0.29¢ pooled, +0.65¢ vs baseline**), but at 11 groups the OOS group-CI spans zero. (3) **Esports: the first CI-certified rung in this line** — Avellaneda-Stoikov **rung 2** beats baseline **+6.66¢ [+1.77, +11.81]**, uniform across 16 groups — but its own level ≈ 0¢, so what's certified is **bleed-avoidance, not profit**. (4) **PBO 0.00** (selection transfers), **DSR fails** (19 obs vs 67 trials — sample length), **White's RC p≈0.03**. Deployable per-contract edge is **nil** both sides → *merits a live measurement loop, not a trading system*.

---

## Glossary (read this first)

| term | plain meaning |
|---|---|
| **rung** | one configuration on the ordered ladder; each must beat the previous *kept* rung to survive (baseline → NSQ core → +two-lens → +asymmetry → +size-dampen → A-S 1/2/3 → basket). |
| **IS / OOS** | in-sample (data used to *choose* knobs) vs out-of-sample (held-out data used to *judge* them). |
| **costed PnL** | realized round-trips + leftover inventory marked at the price you could actually exit at (bid if long, ask if short) — never mark-to-mid. |
| **cohort** | a market's type, tagged from its *lead-in window only*: aggressive/benign (sweep share), thick/thin (flow). |
| **τ-regime** | time-to-resolution bucket (mid-life / approach / endgame; esports pre / in-play) — a slice we *report*, never a split axis. |
| **VPIN** (Lens 1) | volume-clock order-flow toxicity — buy/sell imbalance per equal-volume bucket; leading, names the *exposed* side. |
| **AS-z** (Lens 2) | post-fill adverse-selection z-score — did the mid drift against our fill vs a calm baseline; confirming. |
| **rung / A-S** | "A-S" = Avellaneda-Stoikov, the closed-form optimal-quote theory tested as rungs 1–3. |
| **PBO** | Probability of Backtest Overfitting — how often the in-sample-best config is below-median out-of-sample (0.5 = luck). |
| **DSR** | Deflated Sharpe — is the Sharpe real after penalising for how many configs were tried. |
| **concurrency overlap** | fraction of a held-out market's life that overlaps *in calendar time* with training markets (politics 1.00, esports 0.60). See § Concurrency. |

## The arc in STAR

| phase | Situation | Task | Action | Result |
|---|---|---|---|---|
| **Task 4 legacy** | symmetric quoter's PnL looked like edge | is it real? | decompose the PnL | it's a **directional inventory bet**, not spread capture |
| **Task 5 — controller** | need inventory control | build + honestly judge a skewing/capping quoter | `InventoryAwareQuoter` + costed gated ladder | baseline loses costed; v1 the only kept rung; **no provable edge** |
| **Task 5 — diagnosis** | "no edge" verdict | is it the strategy or the test? | inspect the IS→OOS drop pattern | drop is **uniform + cap-scaled** → regime confound (OOS = endgame), not overfit |
| **5.1 — fix the split** | confounded calendar cut | test generalisation to *unseen markets* | whole-market **nested CPCV** + leakage-safe cohorts | configs transfer **~1:1**; baseline honest loss −0.96¢ → **−0.13¢** |
| **5.1 — fix the controller** | flatten answers the wrong question | carry balanced inventory, refuse one-sided stacking | `NeutralSpikeQuoter` (VPIN + AS-z + graduated + asym + OFI) | through the biggest spike, baseline stacked **−1,800**; NSQ held **±270, ~flat** |
| **5.1 — v0 gate** | don't wire defences blindly | where is toxicity actually net-negative? | pre-registered (cohort × τ) surface | **politics benign-endgame certified toxic**; esports withhold; **AS-z the best flag** |
| **5.1 — ladder** | which rung ships? | gate on honest OOS group-CI | run the full ladder, bracketed | politics **all FRAGILE** (full stack +0.29¢, CI spans 0); **esports A-S rung 2 CERTIFIED** (+6.66¢, bleed-avoidance) |
| **5.1 — audit** | is the selection overfit? | PBO / DSR / White's RC | shared `overfitting_audit` | **PBO 0.00** (transfers), **DSR fails** (short sample), **RC p≈0.03** |

---

## Methods

**Data.** JOIN-1-locked `mm_engine` replay of the R2 L2 capture, politics_negrisk + esports. Task 5: ~11 days, 13 groups. Task 5.1: full **17.9 days** (+7), top-32 quotable tokens/category → **11 politics + 16 esports whole-market groups**. fee = 0, rebate = 0; latency 0 ms (esports figures are latency-naive upper bounds).

**Split — Task 5 (superseded):** single walk-forward, IS 06-19→23 / OOS 06-24→30, *same markets both sides*. **Split — Task 5.1 (authoritative):** whole-market **nested CPCV** (`mm_eval/cpcv.py`) — split unit = the NegRisk event / match so a market's whole lifecycle stays on one side; 6 cohort-balanced folds, every C(6,2)=15 held-out combination a path; **inner-select / outer-estimate**; **leakage-safe cohorts** from the lead-in window only; τ conditions, never splits; overfitting stats reuse `infrastructure/validation/overfitting_audit.py`.

**Keep-gate:** a rung is kept only if its per-group paired OOS delta vs the previous kept config has a group-cluster bootstrap **lower CI > 0** (pessimistic queue). Point-up with CI-through-zero = FRAGILE.

**Controllers.** Task 5: `InventoryAwareQuoter` (microprice skew + cap + flatten + toxicity subset), `ASQuoter` (A-S rungs 1–3), `BasketCarryQuoter`. Task 5.1: `NeutralSpikeQuoter` — microprice core + tight cap, **no flatten**, + **Lens 1 VPIN**, **Lens 2 AS-z (z<−2)**, **graduated response** (directional→suspend exposed side; adverse→shrink; both→widen + passive unwind), **asymmetric repricing** and **OFI size-dampening**. `interfaces.py` frozen; τ, knobs, trade-tape via `params`.

![How the split changed: whole-market folds vs the old calendar cut](../../data/analysis/plots/market_making/mm_task5_1_split_diagnosis.png)

*Each bar = one market's lifecycle; red dashed = Task 5's 06-24 cut. Old split put a market's calm phase in IS and its endgame in OOS; the new folds keep every market whole.*

---

## Results (method → result, with the evidence)

**R1 · Inventory control works mechanically.** Baseline ran a 14,053-contract one-sided book (ended 12,948 short); capped v1 peaked at 295, ended flat. In 5.1's biggest spike the baseline stacked to −1,800 short; NSQ held ±270 and crossed ~flat.

![Inventory path: baseline stacks one-sided; NeutralSpikeQuoter stays flat](../../data/analysis/plots/market_making/mm_task5_1_inventory_path.png)

**R2 · Costed lens kills the baseline.** Charging exits, the symmetric quoter loses OOS everywhere — Task 5: politics −0.96¢ (≈−$2.2k), esports −36.4¢ (≈−$24.9k; one token carried 33.7k contracts to resolution for −$17.3k). Task 5.1 whole-market: politics −0.13¢, esports −3.45¢. *Never quote uncontrolled; never cite naive PnL.*

**R3 · The diagnosis (why Task 5's "no edge" was the test, not the strategy).** In `mm_task5_is_selection.csv` the IS→OOS drop was near-uniform across all configs and *scaled with the cap* — overfitting would punish the selected config hardest; instead the whole surface fell together and configs carrying more inventory into the endgame fell more. Regime confound: edge existed mid-life, was eaten near expiry, and the cut made OOS = the endgame.

**R4 · The fix works: ~1:1 transfer.** Under whole-market nested CPCV every config sits on the diagonal (training ≈ held-out); the cap-scaled collapse is gone, and the baseline re-attributes to −0.13¢.

![Inner→outer transfer by cap: configs now generalise ~1:1](../../data/analysis/plots/market_making/mm_task5_1_inner_outer_by_cap.png)

**R5 · v0 toxicity gate — where defences are allowed.** Pre-registered rule (before the numbers): wire the defensive knobs for a category iff a (cohort × τ) cell is point-negative AND leave-one-market-out-negative ≥50%.

| universe | cohort × τ | markout ¢ | 95% CI | LOMO-neg | net-negative? |
|---|---|---|---|---|---|
| politics | **benign × endgame** | **−0.64** | [−1.02, −0.05] | **100%** | **YES — certified** |
| politics | benign × approach | +0.13 | [−0.03, +0.26] | 0% | no |
| politics | benign × mid-life | +0.35 | [+0.29, +0.42] | 0% | no |
| politics | aggressive × mid-life | +0.32 | [+0.20, +0.46] | 0% | no |
| esports | (every cell) | +0.20…+0.46 | — | 0% | no → **withhold** |

The **AS-z score (Lens 2) is the most efficient toxicity flag measured here** — +0.22¢/+0.59¢ rescue while flagging only 6–7% of volume, beating Task 5's velocity gate. VPIN (Lens 1) is the leading, direction-naming signal, weak as a standalone skip-rule.

![v0 toxicity surface (baseline fills, cohort × τ)](../../data/analysis/plots/market_making/mm_task5_1_v0_surface.png)

**R6 · The ladder (authoritative, nested-CPCV honest OOS).**

*politics_negrisk (11 groups; defensive rungs gated):*

| rung | honest ¢ | path [p10,p90] | Δ vs baseline [95% CI] | keep |
|---|---|---|---|---|
| baseline | −0.13 | [−0.31,−0.07] | — | KEPT (baseline) |
| NSQ core | −0.14 | [−0.18,−0.10] | +0.31 [−0.82,+1.34] | FRAGILE |
| + two-lens | −0.06 | [−0.13,−0.01] | +0.36 [−0.77,+1.37] | FRAGILE |
| + asymmetry | +0.27 | [+0.13,+0.32] | +0.64 [−0.52,+1.71] | FRAGILE |
| **+ size-dampen (full stack)** | **+0.29** | [+0.13,+0.35] | **+0.65 [−0.49,+1.69]** | FRAGILE |
| A-S rung 1/2/3 | −0.23/−0.19/−0.10 | — | span 0 | FRAGILE |
| basket-carry | −0.21 | [−0.25,−0.11] | +0.30 [−0.87,+1.34] | FRAGILE |

*esports (16 groups; defensive rungs diagnostic):*

| rung | honest ¢ | Δ vs prev kept [95% CI] | keep |
|---|---|---|---|
| baseline | −3.45 | — | KEPT (baseline) |
| NSQ core | −0.03 | +4.70 [−1.21,+10.80] | FRAGILE |
| **A-S rung 2** | **+0.03** | **+6.66 [+1.77,+11.81]** | **KEPT — certified** |
| basket-carry | +0.06 | −1.83 [−4.92,+0.15] vs rung2 | worse than rung2 |

Politics: the full stack is the only config positive on every path (+0.29¢), improving monotonically in component order — but no delta certifies at K=11. Esports: A-S rung 2 (derived optimal spread) is the first CI-certified rung; its edge is *uniform across all 16 groups* (the raw skew quoter's larger +4.70¢ was whale-skewed and failed the CI). Its level ≈ 0¢ → certified quantity is **avoided bleed**, not revenue. This **flips Task 5's "A-S certified worse"** — that was a calendar-split artifact.

**R7 · The (cohort × τ) surface (held-out).** Politics benign markets earn **+2.66¢ in the approach window** and give back **−3.13¢ in the endgame** — "edge in mid-life, eaten near expiry" drawn straight from held-out data; it maps *where a live loop should and should not quote*.

![Kept-rung performance surface (cohort × τ)](../../data/analysis/plots/market_making/mm_task5_1_surface_heatmap.png)

**R8 · The spike, and the leading signal.** VPIN jumped from ~0 to 0.57 ~30 min *before* the main move; AS-z confirmed during it (z→−13). The conjunction fired the full defensive response through the toxic window.

![Toxicity trace: VPIN leads, AS-z confirms](../../data/analysis/plots/market_making/mm_task5_1_toxicity_trace.png)

**R9 · Overfitting audit.**

| gate | politics | esports | read |
|---|---|---|---|
| **PBO** | 0.00 | 0.00 | IS-best stays top-half OOS — selection transfers (vs Task 5's 0.5) |
| **DSR** | fails | fails (p≈2e-4) | 19 daily obs vs a 67-config / 43-effective-trial haircut — sample too short |
| **White's RC** | p≈0.032 | p≈0.036 | best configs unlikely pure snooping (supporting, not the gate) |
| **Concurrency overlap** | 1.00 | 0.60 | see § Concurrency — a CI caveat + a *strategy opportunity*, not a leakage to purge |

---

## Concurrency — what it means, and why it is not a problem to purge

**Definition.** For each held-out market, the fraction of its lifetime overlapping *in calendar time* with training markets. Politics = 1.00 (all politics markets trade at once); esports = 0.60 (matches are more sequential).

**Why we do not purge it.** Classic CPCV purge/embargo removes *training samples whose label window overlaps the test window* — the time-series leakage where a training label secretly contains test-period information. Here each market's costed PnL is computed **inside its own lifecycle in a separate engine run**; markets share no samples, so there is nothing for the purge to remove (the group-axis purge is "symbolic" — stated as such in `mm_eval/cpcv.py`). Purging concurrency would mean deleting nearly every group, which is both impossible and pointless.

**The one real statistical consequence.** Concurrent markets share news, so their PnLs are correlated → the group-bootstrap (which resamples groups as if independent) has a smaller *effective* sample than its nominal K → **the OOS confidence intervals are somewhat overconfident (too narrow)**. That is a reason to lean *less* on OOS certification and more on IS + mechanism at this data size — consistent with the decision to go to live measurement rather than chase a backtest CI.

**Concurrency is a strategy positive, not just a caveat.** Correlated markets trading together let a maker **net inventory across them** — cross-market / portfolio inventory control that hedges correlated exposure and permits tighter, larger quoting. This is the natural extension of `BasketCarryQuoter` beyond a single NegRisk event to *correlated events*, and a genuine future edge lever (flagged for the Join-2 roadmap). The current evaluation treats each market independently and therefore *leaves this on the table*.

**IS vs OOS at this sample size.** With 11–16 groups neither window is well-powered; OOS guards against overfitting but is noisy and (per the above) its CI is optimistic. The honest weight is on the **in-sample monotone component-ordering + PBO 0.00 (selection transfers) + the mechanistic spike behaviour**, with OOS as a directional check. "Nothing certifies" therefore means **"not enough independent data to certify," not "no signal"** — which is exactly why the next step is a live loop, where real fills (not backtest CIs) decide.

---

## Materiality & verdict

Deployable per-contract claim is **nil both categories.** Politics' best config nets +0.29¢/contract honest — uncertified and single-digit dollars/day at these fill volumes. Esports' certified rung nets ≈ +0.03¢ volume-pooled; the certified quantity is **+6.7¢/contract of avoided baseline bleed** — risk control, not revenue. **Statistical survival ≠ economic materiality.** Both: *merits a live measurement loop, not a trading system.*

**What changed in the map.** (1) Task 5's "no edge" is **re-attributed to the split** — under whole-market holdout configs transfer ~1:1. The costed-lens conclusion stands; the **A-S condemnation is reversed for esports**; the politics numbers are superseded by the surface. (2) The methodology (`mm_eval/cpcv.py`) is now **durable and controller-agnostic** — the standing eval harness for any future PM MM work (a lemma-library candidate). (3) LOTECH mechanics validated on PM data (microprice fair value, VPIN's lead, AS-z efficiency, one-sided-stacking failure). (4) The near-expiry problem is **located** (benign-cohort endgames) — a flow-triggered gate has a concrete target. (5) **Concurrency is an unexploited edge lever** (cross-market inventory netting).

## Honest caveats (condensed)

1. **OOS CIs are somewhat overconfident** (concurrency → effective N < K), so lean on IS + mechanism at this data size.
2. **The esports keep could be baseline-badness, not rung-2-goodness** — any inventory-controlled config gets most of the +6.66¢; rung 2's distinction is per-group *uniformity* (real, subtler). It beat the prior kept chain and basket head-to-head.
3. **v0-wiring reflexivity** — toxicity mapped on *baseline* fills; the controller changes which fills occur, so the withheld esports defensive knobs might be supported under NSQ's own fills (rung 3's +0.69¢ increment hints). Kept out by pre-registration; a Join-2 A/B.
4. **`k_arr` (A-S rung 2 spread) leans on thin pre-match lead-ins** → live recalibration mandatory. **Latency 0 ms flatters esports most.**
5. **DSR/PBO directional at this K; White's RC borrows significance from correlated days** — the group-CI is the gate, the rest is colour.

## Decision & next step

- **politics — no rung ships.** Join-2 candidate = the **full NSQ stack** (`damp[k=5e-6, cap=500, w=20, d=0.6]`), quoting the benign cohort's mid-life/approach (where +2.66¢ lives); the two-lens gate must prove itself in the certified-toxic benign endgame.
- **esports — A-S rung 2 ships as the measurement config** (`rung2[γ=1e-4, cap=200]`). Live A/B: rung 2 vs rung 2 + two-lens overlay.
- **Concrete next step:** the **Join-2 one-contract live loop** — collapse the queue bracket with real fill rates, calibrate `k_arr`/VPIN bands live, and *test the level the backtest cannot certify*. Explore **cross-market inventory netting** on correlated concurrent politics markets. **No sizing decision from this note.**

---

## Reproduce / artifacts

- **Task 5.1 (authoritative):** `mm_engine/strategies.py::NeutralSpikeQuoter` (+`NSQ_DEFAULTS`), `mm_eval/tape.py`, `mm_eval/cpcv.py`; `scripts/mm_task5_1_{setup,v0_attribution,ladder_run,charts}.py`; `tests/test_mm_task5_1.py`. CSVs/plots: `data/analysis/{csv_outputs,plots}/market_making/mm_task5_1_*`. Full narrative + all charts: [[mm_task5_1_neutral_quoter_cpcv_findings]].
- **Task 5 (superseded methodology):** `mm_engine/strategies.py` (`InventoryAwareQuoter`, `ASQuoter`, `BasketCarryQuoter`), `mm_eval/protocol.py`; `scripts/mm_task5_{fetch_meta,v0_attribution,ladder_run}.py`; `tests/test_mm_task5.py`. CSVs/plots: `…/mm_task5_*`.
- `interfaces.py` frozen throughout; deterministic (seeded bootstraps, event-stream-only strategy state).

## Cross-links

Full 5.1 record: [[mm_task5_1_neutral_quoter_cpcv_findings]]. Baseline: [[mm_symmetric_quoter_validation_findings]] (Task 4). Design inputs: [[mm_market_screen_and_ttr_regime_findings]]. Engine: [[mm_backtesting_methodology_explainer]] · [[mm_join1_reconciliation_findings]] · [[mm_engine_queue_models]]. NegRisk structure: [[mm_politics_negrisk_accounting_findings]] · [[mm_negrisk_consistency_scanner_findings]]. Live loop this feeds: [[mm_politics_negrisk_live_loop_design]] · Join-2 calibration. Hub: [[strat_market_making]].
