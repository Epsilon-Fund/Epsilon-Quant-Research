---
title: "Strat — Market-Making (MM)"
tags: [strat, market-making, block-k]
created: 2026-06-01
status: single-venue maker CLOSED; real-maker playbook validated (K5); politics_negrisk standing live-loop anchor with event-aware sizing
identifier: "MM"
relationship: >
  One of the two strats carved out of the former "Block K" umbrella. MM = earn the spread + maker
  rebate by quoting, fighting adverse selection. Its sibling is [[strat_options_delta]] (digital-option
  pricing / vol / delta hedge). Shared origin: [[block_k_maker_options_research]] (foundation) and
  [[block_k_plain_english_synthesis]] (full arc). "Block K" is now the historical name for the joint arc;
  prompts should target MM or OD specifically.
---

# Strat — Market-Making (MM)


## Summary

- Scope: Strat — Market-Making (MM) in the MM/market-making area.
- Existing takeaway/status: Market-making is the execution/lifecycle layer: quote for spread plus rebate while controlling adverse selection, inventory, and resolution risk. Current active value is live measurement around politics NegRisk and related slow-market neutral-MM questions; older single-venue Polymarket maker variants remain closed.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.

## One-line thesis

Stop trading direction; **quote around fair value** and earn `spread + rebate − adverse selection −
inventory/resolution risk`. On Polymarket, makers are paid (rebate) and takers charged (fee), so liquidity
provision can be net-positive *before* any view on direction.

## Scope (what belongs here)

In: maker economics, Avellaneda-Stoikov / logit quoting, defensive quoting, adverse-selection measurement,
rebate capture, and the model-free study of real profitable maker wallets. Out (→ [[strat_options_delta]]):
anything whose edge is digital-option mispricing, vol richness, basis, or an external delta hedge. The
**bridge** is K-PEG (maker-fill entry) and K2v3 (digital-anchored quoting) — they live here but feed OD's
Strategy A.

## Current state (2026-06-03)

**2026-06-04 directional-vs-neutral decomposition + novelty reframe:** [[mm_structural_maker_directional_decomposition_findings]] — across sports, residual-misc, equities AND politics, the clean neutral (`arb_like`) structured-maker subset is empty/negative; the historical edge sits in `two_sided_directional` wallets. Two consequences. (1) The directional pick is the copyable alpha, the maker wrapper is incidental → feed the directional-carrier wallets into copytrade (sports-first; running). (2) **Crucial framing (the novelty point):** wallet-absence of clean neutral makers is NOT proof neutral MM has no edge — in crypto neutral MM was *directly* tested and died (K2/K2v2/K2v3, adverse selection), but in politics/sports it was only *screened*, never run by us. So neutral structured MM in slow (politics/sports) markets is an **open novelty question** — capture the spread the directional players leave — answerable only live. The politics live loop is best understood as a novel neutral-MM test, not a reproduce-the-winners test. Full map: [[2026-06-04_state_of_the_arc_and_novelty_frontier]].

**2026-06-03 equities up/down structural-maker scope:** [[mm_equities_updown_structural_scope_findings]] is **THIN-DATA / live collector needed**, not an OD pricing reopen. The preferred close-style SPX/NDX structured-non-top3 settled cut is point-positive (**+246 bps**, median wallet **+269 bps**, ex rebate **+206 bps**) but underpowered and CI-crossing (**[-403, +838] bps**) on only **$84.8k** gross. NDX is negative; SPX alone is positive but only 32 target markets; QQQ's tiny positive diagnostic has only 8 markets and must not be promoted. ES/MES adverse-selection drift cannot be reconstructed locally because fill-aligned futures states and PM quote/queue history are missing. Next useful step is a live SPX close-style up/down maker collector with ES/MES post-fill drift, not another OD fair-value pass.

**2026-06-03 politics NegRisk accounting:** [[mm_politics_negrisk_accounting_findings]] unblocks the K5-STRESS `politics_negrisk` structured-non-top3 cell. After indexing 2.10M NegRisk activity rows for 284 structured wallets and decoding 125,937/125,937 relevant tx-wallet receipts, the corrected-carry re-cut is **+2,290 bps**, CI **[1,020, 3,621]**, median wallet **14.5 bps**, net ex rebate **+2,276 bps**. Verdict: **MERITS-LIVE-MEASUREMENT-LOOP**, not alpha-close. Capacity remains Rule-3-as-assumption: at 0.25% / 1% / 5% of non-top3 2026 flow, mean EV/day is about **$1.5k / $6.0k / $29.9k**, but median-wallet EV/day is only **$9 / $38 / $189**. Live unknowns are fill share, queue, missed fills, and operational merge/split/redeem inventory handling.

**Phase-1 execution bridge:** [[polymarket/execution/maker/README|Politics NegRisk maker module]] documents the measurement-loop executor surface; [[2026-06-03_politics_negrisk_phase1_review]] records the first review and targeted follow-up fixes.

**Politics persistence/capacity check:** the cell is a **standing live-loop anchor with event-aware sizing**, not a one-off election-season artifact. 2026 non-top3 politics-NegRisk maker flow is **$381.1M** through 2026-05-26, active on **146/146 observed days**; every observed 2026 day had at least **$250k** flow. Flow mix is broad: 2028 US presidential outrights **36%**, Trump personnel/policy **24%**, non-US elections **19%**, other politics **16%**, 2026 US races/midterms **5%**. Settled-only, no-mark-to-mid edge checks stay positive outside 2024: 2025 close-year **+1,402 bps**, CI **[387, 2,919]**; 2026 settled **+1,356 bps**, CI **[744, 2,507]**. But median-bps monthly EV at 1% capture is only **$0.4k-$1.6k** in observed 2026 months; the mean-bps version is right-tail **$66k-$254k/month**. Treat the loop as continuous measurement, with sizing expanded only when live fills validate event buckets.

**2026-06-02 realism re-audit:** [[od_strategy_a_realism_reaudit_findings]] keeps MM standalone sub-scale under the base median/cell-specific capture model (**$78/active day**), but demotes that capacity result from hard cap to modeled assumption. A 1% fixed capture of non-top3 2026 flow would be about **$304/day** median EV across the 8 deployable cells, while 5% would be about **$1,521/day**. Treat structured non-top3 MM as a live capacity/moat measurement loop and the execution layer for OD Strategy A, not a standalone green-lit bot. The prior `politics_negrisk` block is now superseded by the accounting audit above.

**Our own single-venue Polymarket maker is CLOSED across every anchor tried:**
- K1 economics gate "passes" only by marking to mid (fee-free geopolitics also "passes" → not a real edge).
- K2 logit A-S, optimized, real exit: **−1,126 bps**, CI<0.
- K2v3 Binance/digital-anchored: **0/681 buckets clear**; the anchor *raised* adverse selection (325 vs 145 bps).
- K2v2 defensive (pull/widen on Binance move): **−4,316 bps**; defense fired <0.1% → **adverse selection is
  structural, not a dodgeable latency race.**

**K5 breakthrough (model-free, closed-position realized PnL): real maker-heavy wallets ARE profitable** —
crypto-4h **+171 bps, CI [34, 327]** (256 wallets); pooled top-maker **+145 bps, CI [85, 210]**. Winners'
playbook: **64% two-sided, 78.8% carry-to-resolution, 0.8% in the late near-50¢ spike zone.** **Capacity
warning: top-3 wallets capture ~95% of positive crypto-4h profit per market.** Geopolitics negative (no
rebate). Survivorship caveat (closed positions only). K5b decomposes the dominance; K5-STRESS stresses it.

**Capacity nuance:** K5/K9 do not prove a new entrant can never steal top3/next20 share. They show historical
non-incumbent headroom under observed fills, while historical files lack quote placements, cancels, queue
position, and missed fills. Any claim that better pricing or speed can win share needs live-paper
instrumentation, and belongs in the OD/cross-asset expansion branch rather than another single-venue crypto-4h
maker rerun. See [[od_cross_asset_updown_scoping]].

**Net:** our single-leg quoting is dead; the validated profitable design is *two-sided + carry-to-resolution
+ spike-avoidance + (for us) an external hedge* — and that hedge moves it into [[strat_options_delta]]'s
Strategy A. So MM's live value is now (a) explaining the maker moat and (b) copy/learning the winners.

## Member notes

- Economics & quoting: [[block_k1_maker_economics_findings]], [[block_k2_quoting_findings]], [[block_k2v2_findings]], [[block_k2v3_findings]]
- Real-maker reality check: [[block_k5_findings]], [[block_k5_stress_findings]], [[block_k5b_findings]], [[mm_deployable_cells_findings]], [[od_strategy_a_realism_reaudit_findings]], [[mm_politics_negrisk_accounting_findings]], [[mm_equities_updown_structural_scope_findings]], [[mm_lob_gate_findings]]
- Deterministic arb gates: [[mm_negrisk_consistency_scanner_findings]] (NegRisk basket price-consistency — CLOSE for a small player: real & offline-persistent, but net-negative after fees, dust depth, ~4s live close on liquid baskets)
- Live data and collection semantics: [[mm_clob_capture_semantics]], [[block_a0_runbook]], [[dali_live_l2_capture_plan]]
- Maker-fill entry (bridge to OD Strategy A): [[block_kpeg_findings]], [[block_kpeg_robustness_findings]], [[block_kpeg_robustness_review]]

## Open tasks (authoritative list in `brain/TODO.md` § MM)

- [ ] **Decompose why top-3 makers dominate** (speed/queue vs capital vs carry-risk) from K5 wallet data —
  answers capacity + whether the moat is speed (→ Rust) or structure (→ Python/Midas enough). Codex.
- [x] **Unblock `politics_negrisk` NegRisk accounting** — position-level merge/split/redeem reconstruction survives and merits live measurement. See [[mm_politics_negrisk_accounting_findings]].
- [x] **Check `politics_negrisk` persistence and deployable-capacity translation** — standing measurement anchor, event-aware sizing; median economics remain small until live capture is proven.
- [x] **Equities index up/down as structural MM scope** — completed; THIN-DATA, not OD pricing. Close-style SPX/NDX structured-non-top3 is point-positive but CI-crossing and only $84.8k gross; ES/MES adverse-selection drift needs live capture. See [[mm_equities_updown_structural_scope_findings]].
- [ ] **SPX close-style up/down live MM collector** — quote/fill/queue/cancel telemetry plus ES/MES post-fill drift for 30-50 sessions. Measurement loop only; no sizing ladder until the live structured-maker edge clears.
- [ ] **Strategy B — copy/learn the profitable makers** (re-merges with the copytrade thread).
- [ ] Paper-trade on Midas only after a design clears OOS. Rust only if dominance is speed-based.
- [ ] Do NOT spawn new single-venue quoting/anchor/continuous-hedge variants — those families are exhausted.

## Cross-links

Sibling strat: [[strat_options_delta]]. Foundation: [[block_k_maker_options_research]]. Full arc + glossary:
[[block_k_plain_english_synthesis]]. Pivot handoff: [[2026-05-30_maker_options_delta_pivot]]. Copytrade
re-merge: [[profile_domah]], [[copytrade_relayer_implications]].
