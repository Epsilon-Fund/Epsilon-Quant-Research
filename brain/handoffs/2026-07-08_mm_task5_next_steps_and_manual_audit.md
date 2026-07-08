---
title: "MM Task 5/5.1 — next-steps + manual-audit discussion handoff"
created: 2026-07-08
status: active
owner: justin
project: polymarket-mm
hubs:
  - strat_market_making
  - COWORK
tags:
  - handoff
  - market_making
  - discussion
---

# MM Task 5 / 5.1 — where we are, and what to think about next

> **Purpose.** A discussion handoff for a fresh chat. This is **not a build prompt** — the goal is to decide next steps and, mainly, to set up a **manual audit** (Justin reads the logic himself). No new heavy runs are needed to make progress here; the value now is *understanding*, not more compute. Join 2 (the live loop) is the eventual next build, but there is learning to bank first.

## Read first (in order)

1. `polymarket/research/notes/market_making/mm_task5_inventory_quoter_findings.md` — the consolidated Task 5 + 5.1 paper (STAR, charts, matrices, glossary). Start here.
2. `…/mm_task5_1_neutral_quoter_cpcv_findings.md` — the full 5.1 technical record (all ablations/charts).
3. Code to read for the manual audit: `mm_engine/strategies.py` (`NeutralSpikeQuoter`, `InventoryAwareQuoter`, `ASQuoter`, `BasketCarryQuoter`), `mm_eval/cpcv.py` (the split), `mm_eval/tape.py`.
4. Design inputs: [[mm_market_screen_and_ttr_regime_findings]] · live loop target [[mm_politics_negrisk_live_loop_design]].

## State in one paragraph

Task 5.1 fixed Task 5's broken evaluation split (whole-market nested CPCV) and its controller (`NeutralSpikeQuoter`: microprice skew + tight cap + two-lens VPIN/AS-z gate + graduated response + asymmetric repricing + OFI dampening, no calendar flatten). The regime confound is gone (configs transfer ~1:1). **Politics: directionally right, not certifiable at 11 groups** (full stack +0.29¢ pooled, +0.65¢ vs baseline, positive on every path; no OOS delta clears the CI). **Esports: A-S rung 2 is the first CI-certified rung** (+6.66¢ vs baseline, uniform across 16 groups) but its level ≈ 0 → certified quantity is *bleed-avoidance*, not profit. PBO 0.00 / DSR fails (short sample) / White's RC p≈0.03. Deployable edge nil both → live measurement, not sizing.

## The manual-audit questions worth understanding (highest value first)

1. **Why does Avellaneda-Stoikov (A-S) seem to make things worse — except esports rung 2?** In Task 5 every A-S rung was *certified worse* than the hand-tuned linear skew; in 5.1 A-S rung 2 became the only certified esports keep. Understand: (a) A-S assumes *no adverse selection*, constant σ, a meaningful τ-horizon — which of these does each market violate? (b) rung 2's "optimal spread" quotes wider/off-the-touch → fills collapse; why does that *help* esports (selectivity avoids the resolution jump) but not politics? (c) is the esports win really rung-2 skill or just "any wide-quoting config beats a bleeding baseline"? (caveat 2 in the paper). This is the single richest thing to reason through by hand.
2. **Why does the adverse-selection z-score (AS-z / Lens 2) "rescue" markout but stay withheld?** v0 showed AS-z is the most efficient toxicity flag (biggest markout rescue per unit volume flagged), yet the esports defensive knobs are pre-registered *off* (no net-negative cell), and rung 3's toxicity overlay would add a certified-looking +0.69¢ but is withheld. Understand the reflexivity: toxicity was mapped on *baseline* fills; the controller changes which fills happen. Is the withholding right, or too conservative? (This is a Join-2 A/B, but reason through the logic first.)
3. **IS vs OOS trust at this data size.** With 11–16 groups, is the OOS group-CI even the right gate, or should IS + mechanism + PBO carry more weight? The paper now argues the latter. Pressure-test that: where could leaning on IS bite us?
4. **Concurrency as an opportunity, not a caveat.** Politics markets all trade at once (overlap 1.00). For *evaluation* that only means the OOS CIs are a bit overconfident (not a leakage to purge). For the *strategy* it's an edge lever: **cross-market inventory netting** on correlated concurrent markets (hedge correlated exposure, quote tighter/bigger) — the extension of basket-carry beyond one NegRisk event to correlated events. Worth scoping as a future controller upgrade.
5. **Where is the edge actually located?** The (cohort × τ) surface says: politics benign markets earn +2.66¢ in the approach window and lose −3.13¢ in the endgame; aggressive/long-dated outrights are mildly negative mid-life with no endgame in-sample. Does that survive as a *market-selection* rule (quote benign approach windows, avoid benign endgames), independent of the controller?

## Next-step options (to decide with Justin)

- **A — Wait for Join 2 (live 1-contract loop) as the next real test.** The backtest cannot certify the level; live fills can. Configs are chosen: politics = full NSQ stack (quote mid-life/approach); esports = A-S rung 2, A/B vs rung 2 + two-lens overlay. This is the pre-registered plan.
- **B — Bank understanding first (recommended before any build).** Run the manual audit above; write a short "what we learned about A-S / AS-z / concurrency" note. Cheap, high-value, de-risks the Join-2 design.
- **C — Package the harness.** `mm_eval/cpcv.py` (whole-market nested CPCV + leakage-safe cohorts + the audit stack) is durable and controller-agnostic — the Phase-3 "CPCV as a pip module" item. Extract to `library/lemma` when convenient.
- **D — Scope cross-market inventory netting** (question 4) as a v-next controller — the one idea in this arc that could turn "bleed-avoidance" into an actual edge, and it *uses* the concurrency the eval currently treats as a nuisance.
- **Methodological ceiling to remember:** certifying *politics* via OOS needs a longer, less-concurrent calendar span (more independent time blocks) — not more knobs or folds. If certification matters, it's a data-collection problem.

## Housekeeping (in flight)

A cache/data cleanup pass is being run separately (reduce the 101 GB `data/` footprint + the derived run-caches; revert the pointless Task-5 csv→parquet churn). It does not affect any result or the code above.

## Cross-links

Paper: [[mm_task5_inventory_quoter_findings]]. Full 5.1: [[mm_task5_1_neutral_quoter_cpcv_findings]]. Live loop: [[mm_politics_negrisk_live_loop_design]]. Hub: [[strat_market_making]] · [[COWORK]].
