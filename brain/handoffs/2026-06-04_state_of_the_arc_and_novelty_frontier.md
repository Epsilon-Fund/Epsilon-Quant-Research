---
title: "State of the arc + the novelty frontier (consolidation)"
tags: [handoff, consolidation, synthesis, state-of-the-world, novelty, live-transition]
created: 2026-06-04
status: living consolidation — the single high-level map of where the Polymarket research stands and why the frontier is now live testing + novelty, not more historical mining.
purpose: >
  One place that captures everything we know so far: what is closed and how, what survives, the
  methodological limit we hit (historical screening is blind to novelty), and the live + novelty
  frontier. Read this to orient fast; follow the wikilinks for detail.
relationship: >
  Sits above the hubs [[strat_market_making]] and [[strat_options_delta]] and the live handoff
  [[2026-06-03_politics_negrisk_live_loop]]. Methodology spine: [[CODEX]] § Realism calibration and
  [[COWORK]] § The research loop.
---

# State of the arc + the novelty frontier

## Where we are in one line

Offline PM research is comprehensively exhausted. **PM financial-binary pricing is efficient net-of-cost; the durable edges are execution + copy, and the remaining questions are live-only.** The frontier is now (a) live measurement of the execution edge and (b) deliberately novel hypotheses that historical wallet screening *cannot* validate.

## What is CLOSED (and how — so we don't relitigate)

- **PM financial-binary PRICING is efficient net-of-cost, across crypto AND equities.** 5+ independent tests: crypto terminal ([[od_conditional_prob_calibration_findings]], [[od_pricing_model_form_findings]]), same-day touch + terminal ([[od_same_day_crypto_pricing_gate_findings]]), SPX close N(z) ([[od_equities_index_pricing_scope_findings]]). PM ≈ fair everywhere. Do not propose a pricing/forecasting strategy.
- **OD-as-standalone is closed three ways** — source-vs-valuation (−40.47c), passive-only survival, pure-taker not clearing ([[od_strategy_a_realism_reaudit_findings]]). OD survives only as tiny sizing/selection inside the MM lifecycle.
- **dali local microstructure signal is dead across all framings** — directional continuation (A1.4–A1.7), passive reversion-to-microprice ([[block_a18_passive_reversion_findings]]), cross-market lead-lag ([[block_i_leadlag_feasibility_findings]]). The 73.7% TOB hit was real but un-tradeable (reversion inside the spread; passive route fills ~0.1%).
- **Neutral market-making in CRYPTO is directly-tested dead** — K2/K2v2/K2v3 lost −1,126 to −4,316 bps to *structural* adverse selection (defensive pull fired <0.1%). This is a real experiment, not wallet-absence.
- **Intra-PM arb ≈ zero** on the owned universe ([[block_k4_arb_scan_findings]]); a latency race we'd lose.

## What SURVIVES / is promising

- **The historical structured-maker edge is REAL but DIRECTIONAL-carried, not neutral.** [[mm_structural_maker_directional_decomposition_findings]]: in sports, residual-misc, equities, AND politics, the clean neutral (`arb_like`) subset is empty/negative; the edge sits in `two_sided_directional` wallets. Two exploit paths follow, and **neither is "copying neutral MM":**
  1. **Copy the directional winners** → the directional pick is the (copyable) alpha, the maker wrapper is incidental and hard to copy. Feed the directional-carrier wallets into copytrade (sports-first, fast resolution). *(In progress.)*
  2. **Be the novel neutral maker** → nobody runs disciplined neutral MM in politics/sports, so the spread the directional players leave is uncaptured. Only testable live.
- **politics_negrisk is the proven anchor** — 2,290 bps, accounting-confirmed, standing flow ([[mm_politics_negrisk_accounting_findings]]). Its live measurement loop is the separate-chat handoff [[2026-06-03_politics_negrisk_live_loop]]. NOTE: the directional decomposition shows this edge is also directional-carried, so that loop is best understood as a **novel neutral-MM test**, not a reproduce-the-winners test.
- **copytrade is the standing live track** — Midas exec + the safety harness, the shared plumbing every live idea rides.
- **The maker exec stack mostly already exists** — `maker/negrisk_inventory.py`, `resolution_handler.py`, `event_calendar.py`, `maker_engine.py`, `clob_http_client` with `set_neg_risk`, + tests ([[mm_maker_infra_audit_findings]]). Going live is closer than the TODO wording implies.

## The methodological limit we hit (the important part)

Our method mines **historical wallet behavior** for edge. It is excellent at **falsification** (should-show-up-and-doesn't → dead) and at **finding copyable winners** (who's profitable → copy). It is **structurally blind to novelty**: an edge nobody exploits yet leaves *no historical signal*, so the method reads "no edge" — a false negative on exactly the opportunities worth most. **Novel edge can only be tested forward/live.** So the live transition is also the *novelty* transition: we stop asking "who already does this profitably?" and start asking "what edge *should* exist that nobody runs yet?" — and test it with a small live measurement loop, not a historical screen.

## The live + novelty frontier

Standing/live (separate lanes):
- **Politics neutral-MM live loop** — [[2026-06-03_politics_negrisk_live_loop]] (separate chat).
- **copytrade** — directional-carrier feed + existing leader-audit backlog.

Novelty directions (cannot be validated historically — forward-test only; rank/scope before spend):
- **Disciplined neutral structured MM in slow markets** (politics/sports) — capture the spread directional players leave; the central live experiment.
- **Systematic NegRisk-basket consistency at scale** — most participants mis-account merge/split/redeem; the $34.6M accounting gap ([[mm_politics_negrisk_accounting_findings]]) is itself a potential novelty edge (K4 was narrow-universe and pre-accounting).
- **Resolution-criteria + news LLM scanning (Block J)** — read resolution text + news to flag mispricings before resolution; never attempted; needs a news/LLM pipeline.
- **First-mover liquidity in new/thin markets** — by definition new markets have no historical winners; provide liquidity before incumbents set up.
- **Cross-PLATFORM (Block I, the real version)** — PM-vs-Kalshi cost-structure differences; PM-PM lead-lag closed, cross-platform never tested.

## Discipline that governs all of it

The research loop + its live terminus ([[COWORK]]), § Realism calibration (statistical vs economic, borrowed baselines, capacity-as-assumption, the reopen filter; [[CODEX]]). Carry into the live phase: measurement-first, median-not-mean sizing, kill cheaply, and — new — **treat absence-of-precedent as an open novelty question, not a closed verdict, unless we directly tested it ourselves.**

## 2026-06-05 update — cross-book novelty hunt

A deep literature + repo sweep across BOTH books (this PM frontier + the crypto live stack + cross-pollination + brand-new domains) landed two new notes: [[2026-06-05_novelty_deep_research]] (cited evidence base) and [[2026-06-05_novelty_frontier_map]] (ranked candidates + pre-registered gates + ready-to-paste Codex prompts). Headlines: (1) **before scaling the crypto momentum book, run an overfitting/Monte-Carlo audit** (Deflated Sharpe, PBO/CSCV, regime-aware MC) — the 2.24 Sharpe is the max of a 400-trial search; (2) **Kalshi macro repricing forecasts crypto vol orthogonally to fed funds futures / Deribit IV** ([arXiv 2604.01431](https://arxiv.org/abs/2604.01431)) → a vol throttle that de-risks momentum using the PM pipeline we already own; (3) the **NegRisk-basket consistency** edge is independently corroborated at $40M ([arXiv 2508.03474](https://arxiv.org/abs/2508.03474)), matching our $34.6M accounting gap; (4) on PM forecasting, **calibration ≠ profit** (Prophet Arena GPT-5 beats the market's Brier yet loses net of spread) — reinforces this map's efficient-pricing verdict, with LLM edge surviving only on neglected/fine-print markets (Block J).
