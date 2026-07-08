---
title: "MM Task 5a — design readjustment (companion to the Task 5 PRD goal handoff)"
created: 2026-07-07
status: complete
owner: justin
project: polymarket-mm
para: project
hubs:
  - strat_market_making
  - COWORK
tags:
  - handoff
  - market_making
---
# Task 5a — design readjustment (companion to `2026-07-07_mm_task5_prd_goal_handoff.md`)

**Date:** 2026-07-07 | **Branch:** `justin` → `main` | **Status:** confirmed positions after design discussion; feed these into the `prd-scaffold` Q&A.

This supersedes the *leans* in §4 of the main handoff. Where the handoff said "lean X," this doc records the **confirmed v1 position** plus the reasoning and the explicit build-up path. Guardrails (≤3–4 knobs in v1, OOS/DSR/CPCV gating, bracket every number across {Optimistic, Prob, RiskAverse}, no profitability claim until Join 2) are unchanged.

---

## Decision 1 — skew mechanism: linear skew v1, A-S as a gated build-up

**v1 ships:** hand-tuned **linear skew around the microprice**. One knob: skew slope `k` (cents of reservation shift per contract of inventory). Reservation `r = microprice − k·q`.

**Reference price = microprice (weighted mid), not raw mid.** `weighted_mid = (bid·ask_size + ask·bid_size)/(bid_size+ask_size)`. Zero-parameter, martingale-flavored, directly reduces adverse selection by centering quotes on an imbalance-aware fair value. This is the same information as the book-imbalance toxicity signal, expressed as a price shift. Compute raw-mid and microprice both; quote around microprice; report the markout difference as a free microstructure result.

**Why not full A-S in v1:** A-S is closed-form *optimal only under its assumptions* — it does **not** remove parameters, it restructures them (a free slope becomes `γσ²τ`, with `γ`, `σ` now the knobs) and it assumes **no adverse selection**, which our Step-0 findings contradict. "Derived slope" ≠ "true value": it is optimal *conditional on the model + estimated σ + chosen γ*. So A-S is the backbone to build *toward*, one rung at a time, each rung kept only if it beats the previous rung **out-of-sample**.

### A-S build-up ladder — assumptions imported / estimates without hard evidence, per rung

| Rung | What it adds | Assumptions imported | Estimated without clear evidence | Gate |
|---|---|---|---|---|
| **0 (v1)** | Linear skew `r = microprice − k·q` | Linear inventory response is adequate; skew should pull inventory → 0 | `k` (skew slope) — but ONE interpretable, OOS-tuned knob | ship if beats symmetric baseline OOS |
| **1** | Replace `k` with A-S derived slope `γσ²τ` | Mid ~ arithmetic Brownian motion, **constant σ** over horizon; well-defined horizon τ | `σ` (which estimator? window? regime-dependent), `γ` (risk aversion — **no ground truth**, a risk-budget dial), τ choice | keep only if derived slope beats hand-tuned `k` OOS |
| **2** | A-S optimal spread `≈ γσ²τ/2 + (1/γ)ln(1+γ/k_arr)` | Poisson arrivals `λ(δ)=A·e^{−k_arr·δ}`; fills **independent of information** (no adverse selection) | `A`, `k_arr` (arrival intensity / liquidity decay) — noisy, regime-dependent; the exponential form itself is an assumption | keep only if it beats a fixed/empirical spread OOS |
| **3** | Adverse-selection overlay (the piece A-S omits) — widen/pull on toxicity signals (Glosten-Milgrom flavor) | Informed flow is detectable ex-ante from microprice divergence / imbalance / velocity / τ-proximity | signal thresholds & weights — **highest overfit risk**, informed by Step-0 | strict OOS/DSR/CPCV; ship the smallest subset that survives |

**Rule:** never jump rungs. Each addition must earn its parameter against the DSR/CPCV budget. Full A-S + overlay is 4–5 knobs — over the v1 budget by design; it is the *destination*, not v1.

---

## Decision 2 — time-to-resolution (τ): flatten near expiry (an adverse-selection overlay, NOT the A-S τ term)

**v1 ships:** as τ→0, widen and actively pull to flatten naked inventory before the toxic near-expiry regime. τ injected via `params` (runner computes it from each market's `end_date`; no interface change).

**Critical sign warning:** this is **not** the A-S τ. In A-S the skew *shrinks* as τ→0 (less time = less variance risk = safe to hold). Our concern is that toxicity *grows* as resolution approaches (informed flow + redemption-floor pull). So our near-expiry pull is an **adverse-selection overlay with the opposite sign** to the A-S inventory-variance term — do not wire them through the same τ or they fight.

**Effect on esports (confirmed correct):** flattening clips *both* terminal tails — the few lucky wins and the few big reversals — because both come from holding naked inventory through the resolution jump. That is the right MM stance: our edge is the spread, not predicting the outcome; the terminal jump is a coin flip we have no edge on and near expiry it is negative-EV (adverse). The forgone "few high profits" were compensated gambling, not skill. Step-0 must confirm the near-expiry regime is net-negative; the exit cost of flattening is captured in 5b's costed eval.

---

## Decision 3 — position cap: v1 ships a single TIGHT cap

**v1 ships:** one tight cap; one-sided quoting when at cap (stop adding to the losing side). No sweep in the shipped config.

**Understanding runs (separate from ship):** (a) one **uncapped diagnostic** to characterize the natural inventory excursion (tells us where sane caps sit); (b) a small cap sweep for cost/benefit understanding. These are diagnostics only. **You cannot post-hoc clip a loose-cap run down to a tight cap** — the cap changes the fill path itself (a tight cap refuses fills a loose run took), so it is path-dependent; each cap value needs its own run. Ship-tight is the v1 decision; the sweep is context, and any shipped value is chosen by OOS, never best-in-sample.

---

## Decision 4 — toxicity gate: proposed AND executed inside the Fable prompt, AFTER Step-0

**v1 ships:** the gate is designed and run by the Fable prompt *after it receives Step-0 results*, so the signal set is grounded in which signals actually flagged the Task-4 failures. Seed with the base we discussed (velocity + book imbalance + conservative hardcoded near-expiry pull), then let Step-0 refine.

**Build it as separable, individually-toggleable signals** (velocity, book imbalance / microprice divergence, τ-proximity, one-sided flow, depth evaporation) so we can ablate — gate off / each signal alone / combined — and **attribute** the adverse-selection reduction to each. This doubles as the microstructure research we care about (which signal predicts toxic flow, and does it differ politics vs esports).

**Discipline:** measuring N signals' effects is research (report all configs); *shipping* N tuned signals is overfitting. Keep the measured set and the shipped set separate; shipped subset is the smallest that survives OOS/DSR/CPCV.

---

## Decision 5 — carry vs basket: v1 per-token carry + τ-flatten; basket netting is v2

**v1 ships:** per-token inventory, carried within the tight cap, flattened near expiry.

**Clarification (these were half-merged):** basket-balancing is **not** the τ machinery — it is the *alternative* to it. Two coherent stances:

- **(A) Per-token, flatten near expiry** — naked single-token inventory, exit before toxicity. This is decisions 2+3+4 and is v1. It *avoids* needing the redemption floor.
- **(B) Basket-balanced carry** — hold the complementary NegRisk basket (sums to ~$1) so you redeem $1 at resolution regardless of outcome; carrying is only *safe* because the basket is balanced. This is **v2**.

The redemption floor lives in (B). So basket netting is the machinery for *carry-to-resolution* (v2) — the thing v1 deliberately skips by flattening. It is the v2 upgrade that would let us carry safely instead of flattening, not part of the v1 τ handling.

---

## Net v1 knob budget (check against ≤3–4)

1. skew slope `k`  2. tight cap size  3. near-expiry pull threshold/rate  4. toxicity gate (smallest OOS-surviving subset). Microprice reference = 0-param. A-S rungs 1–3 are **out of v1** by design.
