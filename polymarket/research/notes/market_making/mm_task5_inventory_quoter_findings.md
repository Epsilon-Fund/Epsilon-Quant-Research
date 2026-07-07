---
title: "Inventory-Managed Market-Making (Task 5) — Gated Ladder vs the Symmetric Baseline, Costed & Bracketed (Politics vs Esports)"
created: 2026-07-07
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
  - adverse-selection
  - overfitting
  - backtesting
  - engine
---

# Inventory-Managed Market-Making (Task 5) — Gated Ladder vs the Symmetric Baseline, Costed & Bracketed

> Hubs: [[strat_market_making]] · [[mm_backtesting_methodology_explainer]] · builds directly on [[mm_symmetric_quoter_validation_findings]] (Task 4 — the symmetric baseline + per-token verdicts) and [[mm_market_screen_and_ttr_regime_findings]] (the market screen + time-to-resolution regime) · engine lock: [[mm_join1_reconciliation_findings]] · data limits: [[mm_clob_capture_semantics]] · definitions: [[glossary]] · [[polymarket_table_dictionary]]
>
> **This is Task 5** — the first *inventory-managed* market-making strategy in this research line, evaluated as a **gated ladder** against the Task-4 symmetric baseline with **costs** (realized round-trips + liquidation-marked inventory carry, never mark-to-mid) and the overfitting apparatus (DSR / group-CPCV / PBO) now **live**, because for the first time there are parameters to overfit. New code: `mm_engine/strategies.py` (`InventoryAwareQuoter`, `ASQuoter`, `BasketCarryQuoter` — the frozen `interfaces.py` is untouched), `mm_eval/protocol.py` (the pinned IS/OOS protocol), runners `scripts/mm_task5_v0_attribution.py` + `scripts/mm_task5_ladder_run.py`, metadata `scripts/mm_task5_fetch_meta.py`, tests `tests/test_mm_task5.py`. **No profitability claim — measurement only until Join 2. No live trading.**

## Plain-English Summary

- **What this is.** Task 4 showed the fixed-spread symmetric quoter's PnL is a *directional inventory bet* — it accumulates thousands of one-sided contracts and its "profit" is whichever way the market drifted. Task 5 builds the thing Task 4 said was the gating problem: a quoter that **skews its prices against its own inventory** (sell into strength when long), **caps** its position, and **flattens before the toxic near-expiry regime** — then asks, honestly, whether that beats the symmetric baseline **out-of-sample, with exit costs charged**, on the same ~11-day politics-NegRisk + esports L2 capture.
- **How it is judged (the whole point).** Every config is a rung on a **gated ladder** (symmetric baseline → v1 linear-skew quoter → Avellaneda-Stoikov rungs 1–3 → basket-balanced carry). Knobs are chosen on the **in-sample window only** (2026-06-19→23); the verdict is the **out-of-sample window** (06-24→30), reported as a **bracket across three queue models** {Optimistic, Prob(0.5), RiskAverse} with event-group-clustered bootstrap CIs. A rung is **kept only if it beats the previous kept config OOS on the pessimistic-queue lower CI** — improving the point estimate is not enough. PnL is **costed**: realized round-trips plus the inventory carry marked at the *executable touch* (longs to the bid, shorts to the ask) — the exit cost Task-4's mark-to-mid ignored.
- **v0 gate (pre-registered, answered before any strategy ran).** The near-expiry flatten and the toxicity gate were only allowed into the design where the data says the near-expiry regime is net-negative. **Politics: YES** (last-6h markout −1.01¢/contract, negative in 100% of leave-one-market-out re-pools; directional, not CI-certified — 5 markets). **Esports: NO** (+0.25¢ — an in-play market's whole life is "near expiry" and it is not net-negative there), so for esports those two knobs were **held off and reported as unsupported**, exactly as pre-registered.
- **Inventory control works mechanically.** On the same Musk-tweet token, the baseline ran a 14,053-contract one-sided position and ended the capture 12,948 short; v1 with a 200-contract cap peaked at 295 and ended **flat**, converting the resolution coin-flip into spread capture. That is the designed behavior, not the verdict.
- **The verdict (bracketed, small-sample honest).** Once you charge exit costs, the Task-4 baseline is exposed: it **loses** out-of-sample in both categories (politics −0.96¢/contract pooled ≈ −$2.2k; esports **−36.4¢/contract ≈ −$24.9k** — the "best gem" token alone carried a 33.7k-contract position into match resolution for −$17.3k). The **v1 `InventoryAwareQuoter` is the only rung the ladder keeps**, in both categories: it beats the baseline OOS on the pre-registered group-cluster gate (politics: token-mean per-contract improvement +13.3¢, CI [+0.4, +32.4]; esports +33.7¢ [+25.6, +41.8] — but on just 2 OOS groups) and ends every market ~flat. **Every A-S rung and the basket-carry alternative DROP** — politics rung 1 and basket-carry are *certified worse* than v1 (delta upper CI < 0), not just unproven. **But v1's own OOS level is not a positive edge:** politics +0.03¢/contract point across the whole queue bracket with a CI through zero (strict verdict: DEAD-as-labelled, honestly "positive point, unproven"); esports −1.2¢ (still negative — do not deploy); PBO ≈ 0.5 (IS-selection transfer is a coin flip at K≤7 groups) and DSR fails hard (7 daily obs vs a 29-trial deflation). **Inventory control fixes the failure mode; it does not, on 11 days, manufacture a provable maker edge.**
- **No profitability claim.** 11 days × 2 categories is thin; several rungs read FRAGILE/underpowered by construction, PBO/DSR are approximate at K≤7 event groups, and the true fill rate is a Join-2 live-calibration unknown. Every number below is a conditional, bracketed range.

---

## What this builds on, and the honest sample

**Inputs.** (a) The Task-4 per-token verdict + symmetric baseline ([[mm_symmetric_quoter_validation_findings]]): politics 8/12 VIABLE / esports 5/12 on the spread-capture precondition, with the naive PnL exposed as an inventory bet. (b) The design-inputs map ([[mm_market_screen_and_ttr_regime_findings]]): certified fill-level toxicity signals (large aggressor, book imbalance at fill), the ~50¢-parked-market warning, and the near-expiry toxicity read this note's v0 re-tests. (c) The same **~11-day R2 capture** (`~/epsilon_l2_full`, 2026-06-19 → 06-30, politics_negrisk + esports), replayed through the JOIN-1-locked `mm_engine` — same top-12 tokens per universe as Task 4, so the A/B is on identical markets. (d) **Gamma metadata** per condition id (`data/markets/mm_task5_market_meta.json` — local cache under the gitignored data tree; regenerate offline-stable via `scripts/mm_task5_fetch_meta.py`): `end_date` (the τ anchor), the **NegRisk event id** (the split unit), and resolution payoffs for in-window resolvers.

**Sample truth.** fee = 0 / rebate = 0 (captured), latency 0 ms (isolates the queue gate — same as Task 4; esports numbers remain latency-naive upper bounds). 24 tokens over 17 conditions in **13 event groups** (politics 6: Bennett, Fed, Iran, Petro+Starmer *one* NegRisk event, two Musk-week groups; esports 7 matches). The 11-day window supports a single temporal walk-forward split and group-level cross-validation — nothing finer; CIs at K≤7 groups are **approximate/directional** and labelled so.

## The five design decisions (closed before the build)

Per the PRD (`brain/handoffs/2026-07-07_mm_task5_prd_reference.md`, superseding leans via the design-readjustment handoff):

1. **Skew = linear around the microprice** — reservation `r = microprice − k·q`, `q` = inventory, one knob `k`. **Reference = microprice** (`(bid·ask_size + ask·bid_size)/(bid_size+ask_size)`), 0-parameter and imbalance-aware, not the raw mid. Full A-S is *not* v1 — it restructures the free slope into `γσ²τ` and assumes no adverse selection; it is built as a gated ladder (below).
2. **Near-expiry flatten** — as τ→0 stop opening and rest a reduce-only quote *at the touch* until flat. **Opposite sign to the A-S τ term** (A-S skew shrinks toward expiry; our toxicity grows), so it is wired as a separate overlay, never through the same τ. Deliberately clips *both* esports terminal tails — the edge is the spread, not the resolution coin-flip.
3. **One tight position cap; one-sided reduce-only quoting at cap.** No cap sweep in the ship config; an uncapped diagnostic + a small sweep are reported as understanding-only (a tight cap changes the fill path, so each cap value needs its own run — no post-hoc clipping).
4. **Toxicity gate — separable, individually-toggleable signals, grounded by v0.** Velocity / book-imbalance / depth-evaporation as independent toggles (the strategy-observable subset; aggressor-size is not visible through the frozen `Strategy` interface and v0 showed it does not rescue as a skip-gate anyway). Thresholds are fixed declared constants; **the knob is the subset**, chosen on IS, shipped only if it survives OOS.
5. **Per-token carry + τ-flatten is v1; basket-balanced carry is the v2 *alternative*** (hold the complementary NegRisk basket, redeem regardless of outcome) — compared head-to-head on the same protocol, not stacked.

**τ plumbing (the one interface subtlety):** the frozen `BookState` carries no time-to-resolution. The eval runner knows each market's Gamma `end_date` and injects `end_date_ms` via the `params` dict already passed to `Strategy.quote(book, inventory, params)`; the strategy derives per-event τ from `book.ts_exchange`. `interfaces.py` is untouched.

---

## v0 — the gate: failure attribution + the near-expiry regime (data-only)

v0 re-dumps the Task-4 per-fill telemetry (same setup: RiskAverse pessimistic queue, 0 ms) with three additions: **pre-fill mid velocity** (10 s, strategy-observable), **depth-evaporation ratio** (top depth vs trailing median), and **time-to-resolution for both universes** (esports end-dates were never pulled before).

### The pre-registered near-expiry check (decision rule written before the numbers ran)

*Rule:* wire the τ-flatten + toxicity gate (decisions 2 & 4) for a universe **iff** the qty-weighted markout(30 s) point estimate at TTR < 6 h is negative AND stays negative in ≥50% of leave-one-market-out re-pools. CI status reported alongside (certified iff the market-cluster CI upper bound < 0).

| universe | near-expiry fills | markets | markout(30s) ¢ | market-cluster CI | LOMO-negative | **wire 2 & 4?** |
|---|---|---|---|---|---|---|
| politics_negrisk | 799 | 5 | **−1.01** | [−1.34, +0.10] | 100% | **YES (directional)** |
| esports | 17,380 | 7 | **+0.25** | [−0.05, +0.57] | 0% | **NO — unsupported** |

**Read.** Politics near-expiry is toxic exactly as [[mm_market_screen_and_ttr_regime_findings]] found — the point estimate is −1¢/contract and no single market drives it, though with only 5 (all Musk-type) markets the interval still grazes zero: **directional, not certified**. Esports **fails the pre-registration**: an in-play book lives near expiry by construction, and pooled across all 12 tokens that regime is *positive* (+0.25¢) — the viable esports gems earn their reversion bonus precisely in-play. Consequence, honored throughout the ladder: **esports configs carry no near-expiry pull and no toxicity gate** — those two decisions are reported as unsupported for esports rather than silently shipped. (Tension noted honestly: *per-token*, skipping near-expiry fills would have rescued 3 of 7 DEAD esports tokens — see the rescue table — but the pooled pre-registered criterion is the gate, and it says no.)

### Failure attribution — which observable signal flags the toxic fills?

Counterfactual-rescue test, per token × signal: recompute the qty-weighted markout(30 s) *excluding* fills flagged by the signal (at the token's 80th percentile; full-sample percentiles — research attribution, not a shipped rule). `rescue_delta` > 0 = the gate would have helped.

| universe · verdict | best rescuer | mean rescue ¢/ct | 2nd best | mean ¢ |
|---|---|---|---|---|
| politics DEAD (4) | **pre-fill mid velocity** | **+0.39** | near-expiry (<6h) | +0.32 |
| politics VIABLE (8) | mid velocity | +0.02 (≈0, good) | — | — |
| esports DEAD (7) | **pre-fill mid velocity** | **+0.53** | near-expiry | +0.27 |
| esports VIABLE (5) | (nothing helps; all ≤ 0) | — | — | — |

**Read.** The strongest *strategy-observable* toxicity signal is **pre-fill mid velocity** — a fast tape just before the fill marks the fill as toxic; skipping those fills flips 2 of 4 DEAD politics and 3 of 7 DEAD esports tokens positive (per-token table in `mm_task5_v0_rescue.csv`), while barely touching VIABLE tokens (politics +0.02¢ — the gate is nearly free where the market is benign). Book imbalance is second (politics DEAD +0.20¢). **Aggressor trade size — the design-inputs note's certified per-fill signal — does NOT work as a skip-gate** (politics DEAD −0.27¢): it flags ~40% of quantity and throws away good fills with the bad; it predicts *per-fill* markout but is too blunt as a binary skip. This grounded the v1 toxicity subset ordering: velocity first, imbalance second, depth third.

![Task-5 v0: markout & adverse selection vs time-to-resolution, both universes](../../data/analysis/plots/market_making/mm_task5_v0_ttr_regime.png)

*Read this chart:* x = time-to-resolution bucket (near expiry → mid-life; note esports uses hour-scale buckets — matches live in hours); blue = net markout(30 s), red = adverse selection, ¢/contract, market-cluster CIs; labels show fills and distinct markets per bucket. **What to notice:** politics reproduces the monotone toxicity gradient (−1.0¢ inside 6 h → positive mid-life); esports is *flat-to-positive* across its whole (short) TTR range — its toxicity is bimodal across *markets*, not concentrated at expiry. That asymmetry is exactly why the pre-registered gate wired the flatten for politics only.

---

## The IS/OOS protocol (pinned — this is what "OOS" means everywhere below)

- **Unit of observation = the event / NegRisk group, never the token.** Complementary legs share one resolution fingerprint (Petro+Starmer are literally the same NegRisk event; two Task-4 esports tokens are two outcomes of the same match), so token-level splits leak. All cross-config CIs resample **groups** (politics K=6, esports K=7 — approximate at this K, labelled directional).
- **Primary ship gate = temporal walk-forward (carry-forward).** IS = 2026-06-19 → 06-23 (5 days); OOS = 06-24 → 06-30 (7 days), with a 1 h embargo after the boundary and markout purged where its 30 s window crosses an edge. Inventory carries across the boundary (what live deployment faces). The boundary deliberately puts both Musk-group expiries (06-26, 06-30) in OOS — the strategy *faces* the near-expiry regime out-of-sample. **Esports thinness stated:** 5 of 7 esports events finished inside IS, so the esports OOS rests on 2 groups — reported as underpowered, not hidden.
- **Knobs selected on IS only** (pessimistic queue); **reported on OOS as the bracket** {Optimistic, Prob(0.5), RiskAverse}. Categories are gated separately; cross-regime transfer is a robustness read.
- **Costed PnL** per window = realized round-trip PnL of fills inside the window **+** the change in inventory value **marked at the executable touch** (longs → best bid, shorts → best ask) across the window edges. Per-contract = window costed $ ÷ window filled contracts, in ¢. This charges the carry/exit that mark-to-mid hides. A settle-at-actual-resolution diagnostic (Gamma payoffs) is reported for resolved tokens as a column, not the gate.
- **Keep rule (pre-registered):** a rung is KEPT iff its paired per-token OOS costed-net delta vs the previous kept config has group-cluster lower CI > 0 under the pessimistic queue. Point-improves-with-CI-through-zero = FRAGILE/underpowered → reported, **not** kept.
- **Overfitting apparatus live:** **PBO** via group-CSCV (combinatorial half-splits of event groups; C(6,3)=20 splits politics) over *every trial config*; **DSR** deflates the shipped config's pooled OOS daily-PnL Sharpe by the full number of configs tried (diagnostic runs included — harsher, honest).

### Worked example (one token, the mechanism in action)

Musk-tweet-count token `24890562…` (politics, resolved 06-26, avg price 0.17, VIABLE in Task 4 with +$3,674 naive PnL — *directional luck*):

- **Baseline (symmetric):** accumulates up to **14,053** one-sided contracts, ends the capture **12,948 short**; its whole-capture "profit" was the resolution bet paying off.
- **v1 (k=2e-5, cap 200, pull 6 h, velocity gate — a diagnostic configuration; the shipped politics config is k=5e-6/cap=500/pull=2h):** peak inventory **295** (cap + one clip of overshoot), position at expiry **0** — the τ-flatten worked it flat before the toxic final hours; costed net ≈ **+0.24¢/contract in both windows** — spread capture surviving costs, with the coin-flip surgically removed.

---

## Selection (IS window, pessimistic queue — what was chosen and why)

40 configs were tried in total (12+6 v1 grid, 5 toxicity subsets, 2×5 cap-sweep diagnostics, 3+3 γ grid, ladder + basket; the full per-config table is `mm_task5_is_selection.csv`). Per category, on IS pooled costed ¢/contract:

- **politics:** `v1[k=5e-6, cap=500, pull=2h]` at **+0.34¢** (baseline: **−1.73¢**). Toxicity ablation then picked **velocity alone** (+0.36¢) — the same signal v0's rescue test certified. Gentle skew won: k=8e-5 over-skews (−0.16 to −0.24¢ IS).
- **esports:** `v1[k=2e-5, cap=200]` at **+0.11¢** — but the **baseline beat every v1 config in-sample (+1.00¢)**: in the IS window the uncapped inventory bet *paid*. The ladder's whole point is that this is exactly the number you must not trust — and OOS it reversed to −36.4¢.
- **rung 1 γ:** 1e-5 selected in both categories (the gentlest derived slope; larger γ over-skews the same way large k does).

## The ladder table (the deliverable)

**How to read it.** One row per config, per category. `IS ¢` / `OOS ¢` = pooled per-contract **costed** net (Σ window costed $ ÷ Σ window filled contracts, in cents) — qty-weighted, so high-volume tokens dominate. The OOS bracket spans the three queue models (**RA** = RiskAverse, the pessimistic fill bound; Opt = Optimistic, the upper bound). `Δ vs prev kept` = the **gate**: mean per-token paired delta vs the previous KEPT config (equal token weights — hence its cents scale differs from the pooled columns: a token where the baseline lost 123¢/contract and v1 lost 5¢ contributes +118), with the group-cluster bootstrap 95% CI (politics K=6 / esports K=2 *effective* OOS groups — approximate, directional). `verdict` = strict Task-4 bracket semantics on the **level** (VIABLE iff even the pessimistic queue's group-cluster lower CI > 0; DEAD = not even the optimistic clears) — note a config with a *positive point estimate* still reads DEAD when its CI spans zero; the keep/drop column is the ladder decision.

### politics_negrisk (6 event groups; OOS = 06-24 → 06-30 incl. both Musk-group expiries)

| config | knobs | IS ¢ | OOS ¢ RA / Prob / Opt | OOS fills (RA) | Δ vs prev kept ¢ [95% CI] | verdict (level) | keep? |
|---|---|---|---|---|---|---|---|
| baseline (symmetric) | 0 | −1.73 | −0.96 / −0.93 / −1.10 | 9,372 | — | DEAD | KEPT (baseline) |
| **v1 [k=5e-6, cap=500, pull=2h, tox=vel]** | 4 | +0.36 | **+0.03 / +0.04 / +0.04** | 5,114 | **+13.34 [+0.40, +32.43]** | DEAD¹ | **KEPT — beats baseline OOS** |
| rung 1 [γ=1e-5] | γ replaces k | −0.09 | −0.01 / −0.01 / +0.01 | 4,734 | −0.62 [−1.17, **−0.03**] | DEAD | DROP — **certified worse than v1** |
| rung 2 [γ=1e-5, A-S spread] | +A, k_arr | −0.25 | −0.08 / −0.08 / −0.07 | 1,866 | −1.85 [−5.15, +0.12] | DEAD | DROP |
| rung 3 [γ=1e-5, A-S spread, tox=vel] | +overlay | −0.01 | +0.06 / +0.06 / +0.07 | 1,770 | −0.86 [−2.54, +0.09] | DEAD | DROP |
| basket-carry [k=5e-6, cap=500, no flatten] | 2 | +0.31 | −0.25 / −0.24 / −0.23 | 2,809 | −1.33 [−2.66, **−0.16**] | DEAD | DROP — **certified worse than v1** |

¹ The strict level-verdict reads DEAD because v1's group-cluster CI spans zero at K=6 — the honest translation is "**positive point across the whole queue bracket, unproven at 95%**", not "loses money".

### esports (7 event groups, but only 2 with OOS fills — the walk-forward gate is thin here, stated not hidden)

| config | knobs | IS ¢ | OOS ¢ RA / Prob / Opt | OOS fills (RA) | Δ vs prev kept ¢ [95% CI] | verdict (level) | keep? |
|---|---|---|---|---|---|---|---|
| baseline (symmetric) | 0 | **+1.00** | **−36.38 / −36.41 / −36.49** | 3,238 | — | DEAD | KEPT (baseline) |
| **v1 [k=2e-5, cap=200, pull=off², tox=off²]** | 2 | +0.11 | −1.22 / −1.22 / −1.22 | 868 | **+33.73 [+25.63, +41.82]³** | DEAD | **KEPT — beats baseline OOS** |
| rung 1 [γ=1e-5] | γ replaces k | +0.09 | −1.66 / −1.66 / −1.36 | 662 | −0.48 [−0.79, **−0.17**] | DEAD | DROP — **certified worse than v1** |
| rung 2 [γ=1e-5, A-S spread] | +A, k_arr | +0.25 | −3.78 / −3.78 / −3.78 | 153 | −2.82 [−3.47, **−2.17**] | DEAD | DROP — **certified worse** |
| basket-carry [k=2e-5, cap=200, no flatten] | 2 | +0.17 | −2.05 / −2.05 / −2.04 | 304 | −0.94 [−1.86, **−0.02**] | DEAD | DROP — **certified worse** |

² Per the v0 pre-registered gate (esports near-expiry regime NOT net-negative), the flatten and toxicity knobs were held OFF for esports, and rung 3 (the toxicity overlay) was **omitted** there rather than silently shipped.
³ On 2 effective OOS groups the group-cluster CI is close to degenerate — treat as **very directional**. The direction itself is mechanically unambiguous (a 200-contract cap cannot lose −36¢/contract on inventory), but the interval should not be quoted as a calibrated 95%.

![Task-5 ladder — IS selection vs OOS queue bracket](../../data/analysis/plots/market_making/mm_task5_ladder_oos.png)

*Read this chart:* one row per ladder config (KEPT/DROP tagged); filled markers = the OOS pooled costed net under the three queue models (the bracket — note it is tight everywhere, so nothing here is a queue-assumption artifact); the hollow square = the same config's IS value (the selection window). **What to notice:** (1) the esports panel's x-axis is dominated by the baseline's −36¢ OOS — the inventory bet blowing up is *the* finding; (2) politics baseline sits far left of its own IS value too (−1.7¢ IS → −0.96¢ OOS, negative in both); (3) v1 is the only config near-or-above zero OOS in both panels; (4) IS squares sit systematically right of OOS markers — in-sample optimism made visible.

**Read (what the ladder says).**

1. **The costed lens kills the baseline everywhere.** Task 4's naive mark-to-mid made the symmetric quoter look like a coin-flip (median-negative, some winners). Charging the exit — liquidation at the executable touch, or the actual resolution payoff, which agree to within ~1% on the in-window resolvers — makes it *unambiguously* negative: politics −$2.2k OOS on ~226k filled contracts, esports −$24.9k on ~68k. The single largest hit: token `98484419` (the Task-4 study's *best* per-contract market, +1.77¢ markout) carried −33,718 contracts into its match resolution for **−$17.3k**. Per-contract precondition and portfolio outcome are different animals — this is the cleanest demonstration in the research line so far.
2. **v1 fixes the failure mode.** Same tokens, same windows: politics ends every book ≤370 contracts from flat and posts **+$39** OOS; esports ends ~flat and posts **−$270** (vs −$24.9k). The keep-gate passes in both categories. That is what "inventory control is the gating problem" (Task 4's conclusion) looks like when solved at v1 level.
3. **v1 is damage control, not yet edge.** Politics OOS is +0.03¢/contract — positive at every queue assumption, but ~$6/day across 12 markets and CI-through-zero. Esports OOS is still −1.2¢: with pull/tox correctly disallowed by the v0 gate, skew+cap alone doesn't make in-play esports quoting positive (the cap-sweep confirms tighter is better there: cap 100 → −0.37¢, uncapped → −2.71¢ — every step of inventory freedom costs money OOS).
4. **The A-S ladder loses to the hand-tuned knob at every rung, on this sample.** Rung 1 (γσ²τ replaces k) is *certified worse* than v1 in both categories — the derived slope is a worse inventory controller than a directly-tuned one when its assumptions (constant σ, no adverse selection, meaningful τ-horizon) don't hold. Rung 2's optimal spread (≈2¢ half-spreads from the fitted arrival decay) prices the quoter off the touch — fills collapse ~65% and economics worsen. Rung 3's toxicity overlay claws some back (politics OOS +0.06, the best A-S row) but never beats v1. **No rung was jumped; every rung earned its evaluation and failed its gate.**
5. **Basket-carry loses to flattening head-to-head.** The v2 alternative (carry the NegRisk basket, skew legs toward balance, no flatten) is certified worse than v1 in both categories (politics −1.33¢ [−2.66, −0.16]; esports −0.94¢ [−1.86, −0.02]). On this token set the basket is a *partial* partition (2–3 of ~10 buckets), so "balanced" still carries resolution risk — the redemption floor mostly can't bind. Carrying without the full basket is just a slower inventory bet; decision 5's stance A (flatten) wins as implemented.

## Toxicity ablation + cap sweep (research tables, pessimistic queue)

**Toxicity subsets** (politics, at the selected k=5e-6/cap=500/pull=2h; esports excluded by the v0 gate). The knob was the subset; thresholds were fixed declared constants:

| subset | IS ¢ | OOS ¢ | note |
|---|---|---|---|
| off | +0.34 | −0.05 | |
| **vel** (IS-selected) | **+0.36** | **+0.03** | v0's strongest rescuer; the shipped subset |
| imb | +0.28 | −0.06 | |
| depth | +0.37 | −0.04 | |
| vel+imb | +0.33 | **+0.09** | OOS-best — but selecting it would be OOS-peeking |
| all | +0.36 | +0.07 | |

*Read:* the velocity gate is the only single signal that improves both IS and OOS — consistent with v0. That the OOS-best subset (vel+imb) differs from the IS-selected one (vel) is PBO ≈ 0.5 made concrete: at this sample size, fine subset ranking does not transfer. Shipping "vel" (the IS choice) is the discipline; "vel+imb" is a candidate for Join-2 live measurement, not a backtest claim.

**Cap sweep** (understanding-only diagnostics; each cap is its own run — fill paths are not clippable post-hoc):

| cap (contracts) | politics OOS ¢ | esports OOS ¢ |
|---|---|---|
| 100 | −0.02 | **−0.37** |
| 200 | +0.01 | −1.22 |
| 500 | −0.05 | −1.25 |
| 1000 | −0.02 | −1.77 |
| uncapped | +0.06 | −2.71 |

*Read:* **esports is monotone — every contract of inventory freedom costs money OOS** (and even uncapped-with-skew, −2.7¢, is 13× better than the unskewed baseline's −36¢: the microprice skew alone does most of the damage control). Politics is non-monotone noise at ±0.05¢ — the cap barely binds there because the 2h flatten and the skew keep excursions moderate anyway (uncapped politics peaked well below the esports excursions). No cap value is distinguishable from the shipped one at this power.

## Cross-regime robustness (not a gate)

The v0 gate made the config spaces category-specific (politics carries pull/tox; esports must not), so the exact shipped configs don't transfer verbatim; comparing the shared (k, cap) cores under each category's own pull policy: the politics-selected core (k=5e-6, cap=500) evaluated on esports posts **−2.73¢ OOS** (vs −1.22 for the esports-selected core); the esports-selected core (k=2e-5, cap=200) on politics posts **−0.04¢ OOS** (vs +0.03). **Knobs do not transfer across regimes** — consistent with the two markets' different toxicity structure, and a warning against pooling them in any future fit.

## Overfitting apparatus (live for the first time)

| gate | politics | esports | read |
|---|---|---|---|
| **PBO** (group-CSCV, P(IS-best is below-median OOS)) | **0.50** (20 splits, 6 groups, 29 configs) | **0.46** (35 splits, 7 groups, 17 configs) | IS-ranking transfer is a **coin flip** at this K — config selection on this capture carries no demonstrated skill beyond the baseline-vs-v1 gap |
| **DSR** (shipped config, pooled OOS daily PnL, deflated by trials) | dsr_p = 0.06 (SR_ann 3.0 vs SR* 15.6, 7 obs, 29 trials) | dsr_p = 0.01 (SR_ann −11.2) | **fails**, as pre-warned: 7 daily observations against a 29-trial haircut cannot clear; this is the honest cost of trying 40 configs on 11 days |

*Read:* the apparatus, now that it has parameters to bite on, says exactly what the power caveat predicted it would: **nothing config-level is certified**. The one result that survives is structural, not selected — *any* inventory-controlled variant beats the uncontrolled baseline OOS, while the fine ranking among controlled variants is noise. The pre-registered keep-gate (paired delta vs baseline) is the only claim being made.

---

## Assumption ledger (brain/CODEX.md realism rule 3)

**Modeled assumptions (knobs we set):** queue bracket {Optimistic, Prob(0.5), RiskAverse} stands in for the unknown fill rate (grid/tox/γ selection ran pessimistic-only — conservative and consistent; final configs bracketed); latency 0 ms (fair for politics, optimistic for esports — event-snipe unmeasured, Join-2); liquidation marks at the last *observed* touch (a real exit of hundreds of contracts would walk the book — depth-of-exit unmodeled, biases costed PnL *up* for large terminal inventories, i.e. flatters the **baseline**, not the inventory-managed configs); toxicity thresholds fixed declared constants (0.5¢/10 s velocity, |imbalance| 0.85, depth-EWMA ratio 0.4) grounded on v0, not tuned per config; A-S τ capped at 168 h and σ = 1 h-half-life EWMA (declared); `end_date` = nominal Gamma deadline (placeholder for Dec-2026 outrights — mid-life τ is "large", which only the τ-cap touches); basket-carry covers only the evaluated legs of each event (partial partitions — the redemption floor is weak on this token set and is labelled a diagnostic).

**Live-only unknowns (Join 2):** true passive fill rate + queue position; near-expiry behavior of the durable political outrights (still unobserved — the flatten threshold ships as a live-calibrated rule); esports latency snipe; whether the IS-selected knobs persist out of this 11-day window; real capacity at the touch.

**Materiality (rule 4):** the shipped politics config's OOS economics are **+0.03¢/contract ≈ +$39 over 7 days across 12 markets (~$6/day)** at unthrottled 100-contract clips — statistically unproven *and* economically ~zero before any capacity haircut. Esports is **−1.2¢/contract** — negative, do not deploy. The deployable claim is therefore **nil**; the value of Task 5 is (a) the demonstrated ~$2.2k/$24.9k OOS loss *avoided* vs the naive baseline on the same tokens, and (b) a v1 configuration worth *measuring* live, not a system worth sizing.

---

## Decision and next step

**Gate outcome.** The ladder keeps exactly one rung: **v1 `InventoryAwareQuoter`** — politics `[k=5e-6, cap=500, pull=2h, tox=velocity]`, esports `[k=2e-5, cap=200]` (pull/tox correctly withheld by the v0 gate). It beats the symmetric baseline OOS on the pre-registered group-cluster gate in both categories; every A-S rung and the basket-carry alternative drop, several *certified worse*. The shipped config's strict bracket verdict on its own OOS level is **not VIABLE** (politics: positive point at every queue assumption, CI through zero; esports: negative) — per the framing, this is a **"merits a live MEASUREMENT loop," not "merits a trading system"** disposition.

**What changed in the map.** (1) The costed lens retires the Task-4 naive-PnL ambiguity for good: the uncontrolled symmetric quoter *loses* out-of-sample in both categories once exits are charged — never quote uncontrolled, and never cite naive PnL again. (2) Inventory control at v1 level (microprice skew + tight cap + politics τ-flatten + velocity gate) is **necessary and sufficient to stop the bleeding**, and is the right *default execution layer* for any future Polymarket maker edge. (3) A-S theory earns nothing here yet: its assumptions (no adverse selection, constant σ, meaningful τ-horizon) are the exact things this market violates; revisit only with live-calibrated σ/arrival inputs at Join 2, never as a backtest fit. (4) Basket-carry needs a *full* partition to mean anything — with partial coverage it is a slower inventory bet; only reconsider if the quoted universe ever spans whole NegRisk events.

**Concrete next step (unchanged from Task 4, now with a config to test):** the **Join-2 one-contract live loop on the top VIABLE politics markets**, running v1-politics as the measurement configuration — collapse the queue bracket with real fill rates, measure the near-expiry pull threshold on durable outrights (still unobserved), and A/B the velocity vs velocity+imbalance gate live (the backtest cannot separate them). Esports quoting stays closed pending latency instrumentation. **No sizing decision from this note.**

---

## Reproduce / artifacts

- **Strategies:** `mm_engine/strategies.py` (`InventoryAwareQuoter`, `ASQuoter`, `BasketCarryQuoter`, `microprice`, `tau_hours`) — frozen `interfaces.py` untouched; τ + all knobs via `params`.
- **Protocol:** `mm_eval/protocol.py` (event-group meta, windowed costed PnL, group-cluster deltas, group-CSCV PBO, DSR wiring). Runner τ injection: `mm_eval/runner.py` (`end_date_ms`).
- **Runners** (from `polymarket/research/`): `PYTHONPATH=. uv run python scripts/mm_task5_fetch_meta.py` (Gamma metadata → `data/markets/mm_task5_market_meta.json`), `… scripts/mm_task5_v0_attribution.py` (v0), `… scripts/mm_task5_ladder_run.py --workers 7` (the full ladder; per-run disk cache → crash-resumable).
- **Tests:** `PYTHONPATH=. uv run pytest tests/test_mm_task5.py` (22) — skew signs, cap one-sidedness, flatten-at-touch, toxicity separability, basket netting, windowed-costed accounting, group-delta/PBO logic, determinism. Existing suites unaffected (`tests/test_mm_eval.py` 15 green).
- **CSVs:** `data/analysis/csv_outputs/market_making/mm_task5_{v0_ttr_regime, v0_preregistered_check, v0_rescue, is_selection, ladder_table, pbo, dsr, cross_regime, ab_tokens}.csv`.
- **Plots:** `data/analysis/plots/market_making/mm_task5_v0_ttr_regime.png`, `mm_task5_ladder_oos.png`.
- Deterministic: seeded bootstraps, deterministic engine, event-stream-only strategy state.

## Cross-links

Baseline + verdicts: [[mm_symmetric_quoter_validation_findings]] (Task 4). Design inputs: [[mm_market_screen_and_ttr_regime_findings]]. Engine: [[mm_backtesting_methodology_explainer]] · [[mm_join1_reconciliation_findings]] · [[mm_engine_queue_models]]. Concepts: [[mm_concepts_and_strategy_buildup]] (Layer 2 inventory skew, Layer 4 spike-zone). NegRisk structure: [[mm_politics_negrisk_accounting_findings]] · [[mm_negrisk_consistency_scanner_findings]]. Live loop this feeds: [[mm_politics_negrisk_live_loop_design]] · Join-2 calibration. Hub: [[strat_market_making]].
