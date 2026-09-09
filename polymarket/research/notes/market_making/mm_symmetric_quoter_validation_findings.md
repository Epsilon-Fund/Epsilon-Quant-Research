---
title: "Symmetric-Quoter Market-Making Validation — Per-Market Breakeven, Markout & Adverse Selection, Bracketed (Politics vs Esports)"
created: 2026-07-01
status: active
owner: justin
project: polymarket
para: project
hubs:
  - strat_market_making
  - mm_backtesting_methodology_explainer
tags:
  - market-making
  - backtesting
  - validation
  - adverse-selection
  - markout
  - breakeven
  - engine
---

> **HISTORICAL EVIDENCE (2026-08-25).** Detailed findings behind the active market-making project, kept for verification. Read the de-jargoned canon surface first — [[strat_market_making]] + [[mm_model]] — and treat every number here as preliminary per the hub reliability ledger.


# Symmetric-Quoter Market-Making Validation — Per-Market Breakeven, Markout & Adverse Selection (Politics vs Esports, Queue-Bracketed)

> Hubs: [[strat_market_making]] · [[mm_backtesting_methodology_explainer]] (§6 — the engine this consumes) · prereqs: [[mm_join1_reconciliation_findings]] (JOIN-1 lock) · [[2026-06-23_mm_engine_phase01_buildplan]] (Task 4 brief) · data limits: [[mm_clob_capture_semantics]] · definitions: [[glossary]]
> This is **Task 4** — the validation / eval layer that turns the JOIN-1-locked engine's raw logs into a **decision**. It writes **no engine code**; it orchestrates `mm_engine` (the `SymmetricQuoter` over the VPS Parquet, under each queue model) and consumes the bracketed logs. New code: package `polymarket/research/mm_eval/`, runner `scripts/mm_validation_run.py`, tests `tests/test_mm_eval.py`.

## Plain-English Summary

- **What this is.** A standardized, decision-oriented report on whether a *fixed-spread* maker (the placeholder `SymmetricQuoter`) could plausibly survive per market, on the **~11-day** R2 capture (2026-06-19→06-30, 269 h) of real politics-NegRisk + esports L2, with **every number bracketed** under the optimistic (`OptimisticQueue`) and pessimistic (`RiskAverseQueue`) queue models and carried with a **block-bootstrap CI** — never a point estimate. It is the direct input to **Task 5** (where/how to quote).
- **The one distinction that drives everything (do not conflate).** There are **two** economic readings and they answer different questions. (1) **Per-contract markout / breakeven** (short horizon, 1–60 s): *does the half-spread cover the adverse selection per fill?* — this is the "is there spread to capture" precondition → the **viable / fragile / dead** headline. (2) **The naive quoter's net PnL** (whole capture): **high-variance and median-negative**, because the inventory-blind symmetric quoter accumulates large one-sided inventory whose mark-to-mid swamps the spread — some tokens even land net-*positive* by directional luck (politics median −$234, range [−$5.5k, +$3.7k]; esports median −$5.8k, range [−$28k, +$27k]). So the naive PnL is essentially a **directional inventory bet**, not spread capture — which is exactly why (1), the per-contract markout, is the clean signal. (1) being VIABLE does **not** mean the symmetric quoter profits — it means the market has the spread-capture *precondition* Task 5 needs, *conditional on inventory control*.
- **The captured truth: fees = 0 → rebate = 0.** Every captured trade has `fee_rate_bps = 0`, so the maker has **no rebate cushion**: `net_ex_rebate ≡ net_with_rebate`. The methodology's `0.07/0.20` schedule was a *representative assumption*; the data says zero. We report fee=0 as **primary** and the canonical category rebate as a **labeled sensitivity**.
- **Headline contrast (the point).** On the spread-capture precondition (30 s markout, no rebate, pessimistic queue), **politics-NegRisk is the more reliably-viable maker venue — 8/12 markets VIABLE**, thin ~0.5¢ half-spreads but **low, uniform** adverse selection (median 0.34¢) usually covered, and **temporally stable** (sign holds across ≥75% of daily blocks in 9/12). **Esports is bimodal — only 5/12 VIABLE**: a handful of deep, liquid, balanced-book tokens are the **best markets in the whole study** (favorable adverse selection, net edge up to +1.8¢/contract), but the majority are **toxic** (adverse selection > spread, one at +2.9¢) and the reads are **shakier in time** (5/12 flip sign when one day is removed). Net edges everywhere are **small — sub-1¢/contract even where viable** (materiality caveat below).
- **No profitability claim.** Until Join-2 live calibration collapses the queue bracket toward a live-measured fill rate (and a real strategy adds inventory control), every figure is a conditional range. The overfitting apparatus (Deflated-Sharpe / CPCV) is **wired but dormant** — vacuous on a single-config, 0-parameter quoter.

---

## What this feeds, and the honest sample

This report exists to give Task 5 a per-market read: **which markets have spread left over after adverse selection (so an inventory-managed maker could plausibly survive), and which are dead before you even start.** The contrast between a **slow** market (politics-NegRisk) and a **fast / event-driven** one (esports) is the point — it tells Task 5 *where* to quote.

**Sample (be strict, per [[CODEX]] realism rule 1).** The capture is the **full ~11-day R2 clone** (`r2:epsilon-polymarket-data/parquet`, **2026-06-19 → 2026-06-30**, **269 h ≈ 11.2 days**) for each of politics-NegRisk and esports — ~6.7 GiB and ~5.5 GiB of typed Parquet respectively. (An earlier draft used only the 2-day local `l2_data` slice; that was a partial mirror — corrected to the full R2 sample, which *is* the ~10 days the brief anticipated.) Consequence: 11 days is enough to ask the **temporal-stability** question with real teeth — split into ~daily blocks and check whether the per-market read is driven by one day. It is **not**, however, grounds for an overfitting OOS verdict: a strict OOS split only *bites* once something is **fit** (`ProbQueue.f` at Join 2, or strategy params at Task 5); on a fixed, 0-parameter quoter an IS/OOS gap is sampling, not overfitting (see the dormant-apparatus section).

Per universe (full ~11 days): **politics-NegRisk** = 565 markets / 1,074 traded tokens / 369,969 trades / **351.2 M** price-change events; **esports** = 1,643 markets / 3,067 traded tokens / 650,741 trades / **314.5 M** price-change events. We evaluate the **top-12 quotable tokens** per universe (most-traded over the 11 days, average price in [0.05, 0.95] so it is a genuine two-sided book, ≥150 trades). The unit of observation is **one token** ("market"); a binary or NegRisk market may contribute several tokens. Each token is replayed over its full active span (busiest ≈ 1.6 M events); markout/breakeven pool all of its fills, and the scorecard reflects the continuous-inventory naive quoter (no daily flatten).

---

## How the validation runs (it uses the engine; it never modifies it)

The module drives `mm_engine.run_engine` — the JOIN-1-locked machine — and consumes its raw fill/quote telemetry:

1. **Per-token replay.** The engine's `OrderManager` cancels a token's quotes on the next *other-token* event (Check C, [[mm_join1_reconciliation_findings]]), so each market is replayed **alone** over a small filtered Parquet dir. The quoter rests **at the touch** (`half_spread = median touch spread / 2`), where queue position binds and the models can diverge.
2. **Queue bracket.** Each token is run under **Optimistic** (upper bound on fills), **Prob(f=0.5)** (middle), and **RiskAverse** (lower bound). Latency is held at **0 ms** to isolate the queue gate (latency is ~immaterial for slow politics and is itself a Join-2 calibration target; see the latency caveat below for esports).
3. **Markout + adverse selection** are computed from the fills and the per-event mid trajectory the engine logs. **Scorecard** (PnL three ways, ND-PnL, PnLMAP, fill rate, max inventory, quote uptime, …) is computed per run.
4. **Breakeven + verdict** from the spread + the measured markout, bracketed.
5. **Temporal stability** (CPCV-style) and the **wired-but-dormant** overfitting apparatus are attached.

The A/B framework runs with **one arm** (the symmetric quoter) — the baseline. There is no second strategy until Task 5; the scaffolding (`mm_eval/ab.py`) accepts a second `Arm` unchanged.

---

## The two readings you must not conflate

This is the most important thing in the note.

- **Per-contract markout / breakeven (short horizon).** For each fill we measure the signed move of the mid over the next 1/5/30/60 s. `markout_to_fill(T) = side · (mid_{t+T} − fill_price)` is the realized per-contract edge to mid — it **includes** the half-spread captured at the touch and **nets** the post-fill drift. The **signed post-fill drift** is `drift(T) = side · (mid_{t+T} − mid_at_fill)` (negative ⇒ the market moved against the fill); **adverse selection** `A(T) ≡ −drift(T)` (positive ⇒ against — the sign used in the tables and glossary). Per fill `markout_to_fill = realized_half_spread + drift`, so the net per-contract edge is `E(T) = markout_to_fill(T) + rebate`. The verdict **gates directly on the bootstrap lower-CI of `E(T)`**; the tidy statement "breakeven ⇔ `A(T) ≤ A* = half_spread + rebate`" holds *only approximately* (ignoring intra-order mid drift between quote and fill, and tick rounding — `A*` uses the *quoted* half-spread, `E(T)` the *realized* one), so read `A*` vs `A(T)` as intuition and `E(T)` as the decision. This answers: *is there spread left after adverse selection?* — the **precondition** for a viable maker.
- **Naive-quoter net PnL (whole capture).** The symmetric quoter has **no inventory skew and no position limit**. On a trending book it gets one side filled repeatedly, accumulating large one-sided inventory (median max |inventory| **7.1k contracts politics / 32.7k esports**, peaks 32k / 63k) marked to a drifting mid. The engine's `net_ex_rebate` is therefore **dominated by the unrealized inventory mark**, so it is **high-variance and median-negative — a directional inventory bet, not spread capture**: politics median −$234 (range [−$5,453, +$3,674]), esports median −$5,819 (range [−$27,791, +$27,152]). A token lands net-positive only when the market happened to trend the way the quoter accumulated — luck, not edge.

**These do not contradict — they answer different questions.** A market can be "VIABLE" on the per-contract precondition (the half-spread covers 30 s adverse selection) yet post a large positive-or-negative naive PnL purely from where the price drifted (e.g. the best politics token below: +0.93¢/contract net edge, but the naive quoter still lost −$398). The breakeven verdict is the input to Task 5 (*which markets are worth building an inventory-managed quoter for*); the net-PnL/inventory columns are the reality check (*why the inventory-blind baseline is not deployable, and why inventory control is the gating design problem*). We report **both, side by side**, and never let "VIABLE" stand in for "profitable."

### Why "breakeven *fill rate*" becomes "breakeven *adverse selection*" here

The task frames the headline as a "breakeven fill rate." On this instrument it collapses to a per-contract sign test, and here is the honest reason. For a **passive maker with zero quoting cost and zero rebate** (the captured reality), the expected PnL per fill is `E = half_spread + rebate − adverse_selection`, which **does not depend on fill *volume*** — scaling a *fixed fill mix* up or down just scales total PnL, it does not move the per-contract sign. (It *does* depend on *which* fills you get: a more/less optimistic queue model grants a different fill mix and hence different adverse selection — which is exactly why we **bracket Optimistic↔RiskAverse** instead of assuming a single fill rate. That selection channel is handled by the bracket, not by a fill-rate threshold.) So there is **no interior fill-rate threshold** where PnL crosses zero; the breakeven is the **adverse-selection level** `A* = half_spread + rebate`, and the *fill model* (optimistic vs pessimistic queue) enters by changing *which* and *how many* fills you get, not by scaling a fixed cost. We therefore report, per market: the **breakeven adverse selection** `A*` (= the half-spread, no-rebate), the **measured adverse selection** with CI, bracketed by queue model, and map the optimistic↔pessimistic bracket onto the verdict exactly as the brief asks (DEAD if even the optimistic queue can't clear; FRAGILE if only the optimistic clears; VIABLE if even the pessimistic clears). An interior fill-rate breakeven only re-appears once there is a per-quote cost or a binding inventory/exit cost — i.e. at Task 5, with a real strategy.

---

## Metric definitions (column glossary)

Some scorecard names in the build plan were never defined in the vault; the definitions below are **operational choices made here** and flagged as such. See also [[glossary]].

| metric | definition | unit |
|---|---|---|
| **half-spread** | median touch spread (`best_ask − best_bid` from the `bba` stream) ÷ 2 — the quoter rests here | ¢/contract |
| **markout-to-fill(T)** | `side·(mid_{t+T} − fill_price)`; realized per-contract edge to mid at horizon T (incl. half-spread) | ¢/contract |
| **adverse selection(T)** | `−side·(mid_{t+T} − mid_at_fill)`; positive = the mid moved against the fill | ¢/contract |
| **adverse rate** | qty-weighted share of fills with `markout_to_fill < 0` | fraction |
| **breakeven adverse `A*`** | `half_spread + rebate` — the max adverse selection the spread+rebate can absorb | ¢/contract |
| **net edge `E(T)`** | `markout_to_fill(T) + rebate`; `>0` (lower-CI) = clears breakeven | ¢/contract |
| **ND-PnL** | *Normalized Daily PnL* (operational defn) = net PnL ÷ capture-days (≈11.2, the 269 h capture) | $/day |
| **PnLMAP** | *PnL per unit Mean Absolute Position* (operational defn) = net PnL ÷ time-weighted mean \|inventory\| — capital/inventory efficiency | $ / contract-held |
| **profit ratio** | gross realized profit ÷ \|gross realized loss\| over offsetting round-trips | ratio |
| **fill rate** | `fills ÷ placements` (the engine/reconcile convention) — **can exceed 1**: a sticky resting quote is hit by several trades between re-quotes | ratio |
| **interval Sharpe** | Sharpe of 5-min-bucketed equity increments, **un-annualised**, **diagnostic only** (the quoter's equity is inventory-mark-dominated, so this is noisy) | — |
| **max inventory** | `max(\|position\|)` over the run | contracts |
| **quote uptime** | share of events with a resting two-sided quote (and the stale share) | fraction |

**Fee modes.** *no-rebate* (PRIMARY) = the captured `fee_rate_bps = 0` reality (rebate 0). *representative* (SENSITIVITY, **borrowed** per [[CODEX]] rule 2) = the canonical `FEE_BY_CATEGORY` schedule (Politics 0.04/0.25, Sports 0.03/0.25) — what a rebate *would* add if PM turned it on.

---

## Worked example (one fill)

Take the strongest politics market, token `11216749…` (avg price ≈ 0.21, a wide ~2.6¢ touch → half-spread **1.30¢**). Our BUY rests at the bid = `mid − 1.30¢`. A SELL aggressor trades through and fills us — we are now long 1 contract at `mid − 1.30¢`. Over the next **30 s** the mid drifts **−0.37¢** against us (this token's measured adverse selection, pessimistic queue).

- **markout-to-fill(30s)** = `(mid_{+30s} − fill_price)` = `(mid − 0.37¢) − (mid − 1.30¢)` = **+0.93¢/contract** — the 1.30¢ half-spread more than covers the 0.37¢ adverse drift.
- **Breakeven rule:** `A* = half_spread = 1.30¢` (no rebate); measured adverse `0.37¢ < 1.30¢` ⇒ the net edge is positive, and its pessimistic-queue **lower CI is +0.51¢ > 0** ⇒ **VIABLE**.
- **But the naive quoter still lost −$398 on this token** over the 11 days: it kept re-quoting the bid, accumulated up to 2,336 contracts of long inventory, and the mark-to-mid on that inventory (a directional bet) swamped the +0.93¢/contract spread capture. That gap — positive per-contract precondition, negative naive PnL — *is* the Task-5 mandate: capture the +0.93¢ **while controlling the inventory**.

---

## Headline — per-market bracketed verdict

**Verdict semantics:** the verdict is the **spread-capture precondition** (per-contract `E(30s)` net of adverse selection, no rebate), bracketed by queue. **VIABLE** = even the pessimistic queue's net-edge lower-CI > 0; **FRAGILE** = only the optimistic clears; **DEAD** = even the optimistic can't clear. *It is necessary, not sufficient — see the naive net-PnL column.*

**Table columns:** `net edge RA (lo)` = per-contract net edge (¢) under the **pessimistic** RiskAverse queue with its bootstrap lower-CI (no rebate); `adverse RA` = measured adverse selection ¢ (pessimistic); `naive net $` = the inventory-blind quoter's whole-capture `net_ex_rebate` (the *directional-bet* reality, **not** the verdict); `max inv` = peak |inventory| (contracts); `stability` = daily-block sign-stability (0–1), or **flip** if removing one day flips the pooled sign. Sorted by net edge.

### politics-NegRisk (slow market) — 8/12 VIABLE

| token | avg px | ½-spread ¢ | net edge RA (lo) ¢ | adverse RA ¢ | naive net $ | max inv | stability | verdict |
|---|---|---|---|---|---|---|---|---|
| 11216749… | 0.21 | 1.30 | +0.93 (+0.51) | +0.37 | −398 | 2,336 | 1.00 | **VIABLE** |
| 39343707… | 0.70 | 1.00 | +0.62 (+0.47) | +0.38 | −2,154 | 7,085 | 0.89 | **VIABLE** |
| 24890562… | 0.17 | 0.50 | +0.43 (+0.33) | +0.07 | +3,674 | 14,053 | 1.00 | **VIABLE** |
| 11160441… | 0.79 | 0.50 | +0.42 (+0.35) | +0.08 | −71 | 3,464 | 1.00 | **VIABLE** |
| 38128511… | 0.20 | 0.70 | +0.39 (+0.13) | +0.31 | +1,777 | 3,996 | 0.91 | **VIABLE** |
| 39409206… | 0.86 | 0.50 | +0.32 (+0.29) | +0.18 | −1,301 | 32,344 | 1.00 | **VIABLE** |
| 96718127… | 0.51 | 0.40 | +0.29 (+0.01) | +0.11 | −3,489 | 6,741 | 0.88 | **VIABLE** |
| 58945936… | 0.15 | 0.50 | +0.28 (+0.25) | +0.22 | +873 | 23,579 | 1.00 | **VIABLE** |
| 68177837… | 0.17 | 0.25 | −0.16 (−0.42) | +0.41 | +2,989 | 14,009 | 0.30 | **DEAD** |
| 24577050… | 0.83 | 0.70 | −0.17 (−0.56) | +0.87 | −2,852 | 7,048 | flip | **DEAD** |
| 33312145… | 0.40 | 0.50 | −0.22 (−1.64) | +0.72 | −5,453 | 11,107 | flip | **DEAD** |
| 77865029… | 0.28 | 0.65 | −0.93 (−2.50) | +1.59 | +1,735 | 4,047 | 0.83 | **DEAD** |

**Read.** The dividing line is clean and mechanical: **VIABLE iff measured adverse selection < the half-spread**, with the pessimistic lower-CI clearing 0. The 8 viable politics markets have adverse selection 0.07–0.38¢ against 0.4–1.3¢ half-spreads — spread to spare. The 4 dead ones have adverse selection 0.41–1.59¢ that overruns thin 0.25–0.7¢ spreads (token `77865029…` is the worst: 1.59¢ adverse on a 0.65¢ half-spread → −0.93¢/contract). The optimistic and pessimistic net edges are **nearly identical** (e.g. +0.32 vs +0.32) — at the touch on a slow book the per-contract markout barely depends on the queue model, so the **bracket is tight and the verdict is not a queue-assumption artifact**. Note the `naive net $` column swings both signs (+$3,674 to −$5,453) independently of the verdict — that is the directional inventory bet, not the edge.

### esports (fast / event-driven market) — 5/12 VIABLE

| token | avg px | ½-spread ¢ | net edge RA (lo) ¢ | adverse RA ¢ | naive net $ | max inv | stability | verdict |
|---|---|---|---|---|---|---|---|---|
| 98484419… | 0.46 | 1.00 | +1.77 (+0.60) | −0.77 | −17,329 | 33,813 | 0.80 | **VIABLE** |
| 11376685… | 0.55 | 0.50 | +1.01 (+0.68) | −0.51 | +15,015 | 26,958 | 0.89 | **VIABLE** |
| 23751721… | 0.42 | 0.50 | +0.73 (+0.11) | −0.23 | +27,152 | 61,334 | 0.58 | **VIABLE** |
| 55785151… | 0.27 | 0.50 | +0.53 (+0.37) | −0.03 | +5,082 | 19,966 | 0.86 | **VIABLE** |
| 94930362… | 0.49 | 0.50 | +0.45 (−0.00) | +0.05 | +22,516 | 50,912 | 0.70 | **DEAD** |
| 11308897… | 0.73 | 0.50 | +0.23 (+0.06) | +0.27 | −4,165 | 16,626 | 1.00 | **VIABLE** |
| 58796774… | 0.50 | 0.50 | +0.18 (−0.21) | +0.32 | −11,363 | 20,265 | 0.67 | **DEAD** |
| 24800977… | 0.51 | 0.50 | +0.02 (−0.72) | +0.48 | −15,585 | 31,503 | flip | **DEAD** |
| 10178948… | 0.58 | 0.50 | −0.04 (−0.59) | +0.54 | −27,791 | 63,046 | flip | **DEAD** |
| 55252861… | 0.50 | 0.50 | −0.10 (−1.13) | +0.60 | +17,372 | 39,537 | flip | **DEAD** |
| 78005024… | 0.53 | 0.50 | −0.12 (−1.06) | +0.62 | −20,499 | 44,427 | flip | **DEAD** |
| 47686579… | 0.68 | 1.00 | −1.87 (−6.39) | +2.87 | −7,473 | 19,333 | flip | **DEAD** |

**Read.** Esports is **bimodal**. The top 4 are the **best markets in the study** — deep, liquid, balanced books where the post-fill mid *reverts* (adverse selection is **negative**, i.e. favorable: −0.03 to −0.77¢), so the maker keeps the whole half-spread *plus* a reversion bonus (net edge up to +1.77¢/contract). But the other 8 are toxic: adverse selection 0.05–2.87¢ that meets or beats the uniform 0.5¢ half-spread. Two are borderline-DEAD only on the CI (`94930362…` net +0.45 but lower-CI −0.00; `58796774…` +0.18 / −0.21) — the point estimate is positive but the pessimistic bootstrap can't rule out zero, so we do not call them viable. The worst, `47686579…`, pays **2.87¢** adverse on a 1.0¢ half-spread (−1.87¢/contract). Crucially, **DEAD esports tokens are overwhelmingly `flip`** — their (already-negative) reads are also window-concentrated, and even the viable `23751721…` is only 0.58 stable — so esports viability is **fragile in time**, unlike politics.

---

## Markout curve + adverse-selection profile

![Per-contract markout to mid vs horizon (universe mean, queue-bracketed)](../../data/analysis/plots/market_making/mm_validation_markout_curves.png)

*Read this chart:* x = markout horizon (1/5/30/60 s); y = the **mean across the 12 tokens** of each token's quantity-weighted markout-to-fill, ¢/contract (the realized per-contract edge to mid, **including** the half-spread; the cited numbers are this cross-token mean — qty-weighting *across* tokens shifts them slightly, e.g. esports 30 s +0.26¢ vs +0.23¢); the three lines are the queue bracket (Optimistic / Prob(0.5) / RiskAverse). Above 0 = fills are accretive at that horizon; the slope shows whether adverse selection grows with horizon. **What to notice:** both universes start near **+0.45¢** at +1 s (the half-spread, barely touched yet), but then diverge — **politics decays monotonically** (+0.44 → +0.30 → +0.18 → **+0.07¢** by 60 s) as adverse drift grows steadily (−0.19 → −0.55¢), i.e. the slow-market maker is in a race between banking the spread and the mid grinding away; **esports plateaus** (+0.45 → +0.28 → **+0.23 → +0.25¢**) — adverse drift saturates near −0.35¢ rather than growing, but a larger *share* of fills are adverse (adverse-rate 0.13 → 0.39). The three queue lines sit almost on top of each other — the per-contract markout is nearly queue-invariant at the touch.

![Spread vs adverse selection — below the line survives, above is dead](../../data/analysis/plots/market_making/mm_validation_breakeven_scatter.png)

*Read this chart:* x = half-spread ¢ (= the no-rebate breakeven adverse selection `A*`); y = measured adverse selection ¢ under the pessimistic queue; the dashed `y = x` line is breakeven. Points **below** the line (adverse < spread) clear the precondition; **above** (adverse > spread) are dead. Colour = universe. **What to notice:** politics (blue) clusters at thin half-spreads (0.25–1.3¢) with adverse selection mostly *below* the line; esports (red) sits on a 0.5–1.0¢ half-spread column but **splits vertically** — a few points at *negative* adverse selection (favorable, far below the line = the liquid gems) and a scatter *above* the line (the toxic majority, up to 2.87¢).

**Read.** The two lenses agree: a maker "survives" a market when the spread it can rest at exceeds the adverse selection it eats. Politics wins that trade-off *broadly but thinly* (many markets clear, none by much); esports wins it *narrowly but richly* (few markets clear, but those few clear by a lot). Neither offers a fat edge — the best sustained per-contract number in the study is esports `98484419…` at +1.77¢, and the median viable market is well under 1¢/contract.

---

## The slow-vs-fast contrast (the point)

This contrast is the direct input to Task 5's "where to quote" decision.

| | politics-NegRisk (slow) | esports (fast / event-driven) |
|---|---|---|
| median half-spread | 0.50¢ | 0.50¢ |
| adverse selection (median, pess. 30 s) | **+0.34¢** (low, uniform) | **+0.30¢** but **bimodal** (−0.77 … +2.87¢) |
| markout vs horizon | decays to ~0 by 60 s | plateaus ~+0.25¢ |
| VIABLE (no-rebate) | **8/12** | **5/12** |
| median net edge (Opt/RA) | +0.31 / +0.30¢ | +0.23 / +0.20¢ |
| best single market | +0.93¢ (`11216749…`) | **+1.77¢** (`98484419…`) |
| temporal stability (sign) | median **0.90**, 2/12 flip | median **0.62**, 5/12 flip |
| profit ratio (realized round-trips) | **1.37** (>1) | 0.83 (<1) |

**The read.** For a small, non-colocated maker, **politics is the safer, broader venue**: thin spreads but low and *uniform* adverse selection, most markets clear the precondition, its realized round-trip leg is not loss-making on the sample (profit ratio 1.37 — a no-CI diagnostic that excludes the open-inventory mark), and the verdicts are **robust across the 11 days**. **Esports is a stock-picker's venue**: the median market is *worse* (more toxic, profit ratio 0.83, 7/12 dead) and the reads are **fragile in time** (5/12 flip on one day), but the few deep, liquid, balanced-book tokens carry the **richest edge in the study** (favorable reversion, up to +1.77¢/contract). Task-5 implication: **quote politics broadly with light selection; quote esports only on a hard liquidity/toxicity screen** (and re-measure per event, since match dynamics move the toxicity).

> **LATENCY CAVEAT (load-bearing for esports).** Every number here is at **0 ms latency**, which isolates the queue gate. That is fair for slow politics but **optimistic for esports**: esports is event-driven, and the methodology ([[mm_backtesting_methodology_explainer]] §2) is explicit that in-play markets snipe resting quotes around scores/rounds — adverse selection a 0-ms replay *cannot see*. So the esports "favorable adverse selection" on the liquid tokens is a **latency-naive upper bound**; a real maker's resting quote would be picked off on the game events that move these books. This is a **live-only unknown** (Join 2), and it means the esports gems deserve *more* skepticism than their point estimates suggest, not less.

---

## Full scorecard

The full scorecard (all tokens × queue models) is in `data/analysis/csv_outputs/market_making/mm_validation_scorecard.csv`. Universe **medians** under the pessimistic (RiskAverse) queue:

| metric | politics-NegRisk | esports |
|---|---|---|
| ND-PnL (net-ex-rebate, $/day) | **−20.9** | **−518.7** |
| PnLMAP ($ per unit mean-\|inventory\|) | −0.045 | −0.649 |
| profit ratio (realized round-trips) | **1.37** | 0.83 |
| interval Sharpe (5-min, un-annualised, diagnostic) | −0.008 | −0.054 |
| max drawdown ($) | 1,999 | 17,229 |
| fill rate (fills/placement) | 0.59 | 1.32 |
| quote uptime (two-sided) | 1.00 | 0.98 |
| stale share | 0.000 | 0.001 |
| `l1_both_match_frac` (telemetry artifact) | 0.40 | 0.68 |

**Read.** The scorecard is the **naive-baseline reality check, not a profitability statement**. ND-PnL and PnLMAP are negative (esports much more so — ~−$519/day and −0.65 $/contract-held vs politics ~−$21/day and −0.05) because the inventory-blind quoter carries huge inventory that marks against it; PnLMAP being ~13× worse in esports quantifies the inventory-efficiency gap the fast market punishes. Two things cut the other way and matter for Task 5. Politics's **profit ratio 1.37 vs esports's 0.83** — read as a *diagnostic, not a profitability result* (it is a bare universe-median of realized offsetting round-trips: no CI, and it deliberately **excludes the open-inventory mark that dominates total PnL**) — indicates the *realized* spread-capture leg is not itself loss-making in politics (>1), while esports's <1 says even realized round-trips lose on the median toxic token. So in politics it is the *open inventory* that bleeds, not the spread leg; in esports both do. Quote **uptime ≈ 1.0** and **stale ≈ 0** confirm the quoter was continuously in the book on the clean, gap-free capture. The low `l1_both_match_frac` (0.40 / 0.68) is the **known intra-ms `best_bid_ask` ordering artifact** from JOIN-1, *not* a book error — it does not touch fills or the executable mid (see [[mm_join1_reconciliation_findings]] §B).

---

## Temporal stability (CPCV-style — is the read driven by one window?)

The ~11-day capture is split into ~daily time blocks (`--n-blocks 12`); per token we recompute the per-contract markout in each block and test (a) **sign stability** (share of blocks agreeing with the pooled sign) and (b) **leave-one-block-out** (does removing any single day flip the pooled sign?). With 11 days this has real teeth — it catches a "verdict" that is actually one day's burst. It is still **not** an overfitting verdict (nothing is fit) — it is the robustness-in-time check.

| universe (pessimistic queue) | median sign-stability | LOBO sign-flips | block notes |
|---|---|---|---|
| politics-NegRisk | **0.90** | **2/12** | 9 stable · 1 mixed · 2 flip |
| esports | **0.62** | **5/12** | 4 stable · 3 mixed · 5 flip |

**Read.** Politics is **temporally robust** — the per-contract sign holds across ≥75% of daily blocks in 9/12 markets, and only 2/12 flip when a single day is dropped (and those 2 are already DEAD). So the politics viable/dead verdicts are **not artifacts of one day**. Esports is **materially shakier**: median sign-stability 0.62 and **5/12 flip** on one day — including some in the VIABLE set (`23751721…` at 0.58). This reinforces the contrast: esports viability is real on the sample but **window-dependent**, so any esports quoting must be re-measured continuously rather than trusted from an 11-day snapshot.

---

## Wired-but-dormant: the overfitting apparatus (no verdict)

The Deflated-Sharpe / CPCV / OOS apparatus (`mm_eval/overfitting_hook.py`, importing the shared `infrastructure/validation/overfitting_audit.py`) is **wired but reports no verdict**, because it is **vacuous on a single-config, 0-parameter quoter**:

- **Deflated Sharpe** deflates the best-of-N selected Sharpe by the expected max under the null of N trials. With **N = 1** (one fixed config), the haircut `SR* = expected_max_sharpe_null(1, ·) = 0` — *there is nothing to deflate*. Demonstrated numerically (haircut = 0.000) rather than asserted.
- **PBO / CSCV** needs a `T × N` candidate-return matrix (`N ≥ 2`); one config → undefined.
- A strict **OOS split** only *bites* once a parameter is **fit**; on a fixed quoter an IS/OOS gap is sampling noise.

It **becomes live at Task 5** (a parameterized strategy selected across trials) and at **Join 2** (fitting `ProbQueue.f`). Stating an overfitting "pass/fail" on the symmetric quoter now would be meaningless.

---

## Assumption ledger ([[CODEX]] realism rule 3)

**Modeled assumptions (knobs we set):** the queue bracket (Optimistic↔RiskAverse) stands in for the unknown true fill rate; latency = 0 ms (fair for politics, **optimistic for esports**); half-spread = median touch spread/2 (the quoter rests at the touch); fee = 0 / rebate = 0 from the capture (primary), category schedule as a labeled sensitivity; markout to the engine's reconstructed mid (faithful ~99% per JOIN-1, despite the low `l1_both_match_frac` telemetry artifact).

**Live-only unknowns (only the 1-contract loop / capture can resolve):** our true passive fill rate net of adverse selection; the real per-market rebate policy; **esports latency / event-snipe adverse selection** (the 0-ms read cannot see it); whether the favorable short-horizon markout survives once inventory is actually carried; edge persistence.

**Materiality (rule 4).** Per-contract edges are reported in **cents/contract** absolute, and they are **small**. The median viable market clears breakeven by only **~0.3¢/contract** (politics) to **~0.5¢/contract** (esports viable subset); the single best sustained market is **+1.77¢/contract** (esports `98484419…`, latency-naive). On a $0.50 contract that is a ~0.6%–3.5% gross edge *before* any capacity haircut, before the inventory cost the naive baseline shows is real, and before esports's latency snipe. **This is a "statistically-positive, economically-thin" precondition** — enough to justify a Join-2 live *measurement* loop on the strongest politics markets, **not** enough to green-light a sized bot. Rebate would help but is **0 in the captured data**; the representative-schedule sensitivity adds only a fraction of a cent (Politics 0.04/0.25, Sports 0.03/0.25 on `p(1−p)` prices) — small, but **enough to flip two marginal esports verdicts**: `94930362…` DEAD→VIABLE and `58796774…` DEAD→FRAGILE (politics unchanged). So the rebate is not decision-neutral on the toxic-but-borderline esports tokens, even though it changes nothing in politics.

---

## Decision and next step

**Per-market verdict (bracketed, ~11 days, no-rebate, 0-ms latency):** politics-NegRisk **8/12 VIABLE, 4 DEAD**; esports **5/12 VIABLE, 7 DEAD**. "VIABLE" = the half-spread covers the measured 30 s adverse selection with the pessimistic-queue lower-CI above zero — the **spread-capture precondition**, necessary but not sufficient. The optimistic/pessimistic bracket is **tight** (per-contract markout is nearly queue-invariant at the touch), so these verdicts are not queue-assumption artifacts.

**What Task 5 should build from this:**
1. **Where to quote — politics broadly, esports selectively.** Politics is the reliable venue: low uniform adverse selection, most markets clear, its realized round-trip leg is not loss-making on the sample (profit ratio 1.37, a no-CI diagnostic), and it is **temporally stable** (2/12 window-flips). Esports is a stock-picker's venue: the median token is toxic and the reads are **fragile in time** (5/12 flip), but the few deep/liquid/balanced-book tokens carry the study's richest edge — gate them on a hard liquidity/toxicity screen and re-measure per event.
2. **The gating design problem is inventory control, not spread.** The naive symmetric quoter is a **directional inventory bet** (net PnL median-negative, high-variance, sign uncorrelated with the verdict) precisely because it has no inventory skew or position cap. Every "VIABLE" precondition is only realizable by a quoter that skews on inventory and caps position — that is Task 5's core.
3. **Economics are thin.** Best sustained edge +1.77¢/contract, median viable ~0.3–0.5¢ — statistically positive, economically marginal, with **no rebate** in the captured data and **latency snipe unmeasured** for esports.

This is a **live-measurement precondition map, not a trading system** ([[CODEX]] rule 3): the concrete next step is a **Join-2 1-contract live loop on the top politics markets** (e.g. `11216749…`, `39343707…`) to measure the true passive fill rate, collapse the queue bracket, and confirm the short-horizon markout survives real inventory carry. Esports gems (`98484419…`, `11376685…`) are worth a *separate* latency-instrumented measurement, not a bot. Neither, on this evidence, merits a sized deployment yet.

> **Design-input follow-up (2026-07-03) → [[mm_market_screen_and_ttr_regime_findings]].** Two analyses that turn this per-token verdict into Task-5 inputs, on the same capture: (1) a **failure-driver market screen** — the certified real-time toxicity signals are *large aggressor trade* + *book imbalance at fill*; the pre-quote gate is *avoid ~50¢-parked markets, prefer calm low-mid-vol books*; and **half-spread is refuted as a safety selector** (toxic books are the *wider* ones). (2) A **time-to-resolution regime** with a **load-bearing amendment to this note's politics read**: the top-12 politics tokens are a mix of horizons, and the favorable read is carried by **durable Dec-2026 outrights observed only ~6 months out (deep mid-life)**; the *only* near-expiry politics we observe (short-dated Musk-tweet markets resolving in-window) is **toxic** (markout −1.0¢ inside the last 6 h, adverse selection +1.5¢ CI-clears; within-market log-TTR slope positive under fixed effects). So **"politics is the safe venue" is a mid-life statement — near-expiry toxicity for the durable outrights is unobserved (a live-only risk), and Task 5 must pull quotes near expiry as a live-calibrated rule.**

---

## Reproduce / artifacts

- **Module:** `polymarket/research/mm_eval/` (`markets.py`, `metrics.py`, `runner.py`, `stability.py`, `overfitting_hook.py`, `ab.py`, `report.py`).
- **Runner:** `PYTHONPATH=. uv run python scripts/mm_validation_run.py --top-k 12 --horizon 30 --n-boot 2000 --n-blocks 12 --l2-root <R2-clone> --cache <cache>` (from `polymarket/research/`). The ~11-day R2 clone (`r2:epsilon-polymarket-data/parquet`) was pulled with `rclone copy`; `--l2-root` points at it. Determinism: seeded bootstrap + deterministic engine → reproducible.
- **Tests:** `tests/test_mm_eval.py` (markout signs, censoring, block-bootstrap CI, breakeven/verdict, scorecard, stability, dormant overfitting) — `PYTHONPATH=. uv run pytest tests/test_mm_eval.py`.
- **CSVs:** `data/analysis/csv_outputs/market_making/mm_validation_{scorecard,markout,breakeven,verdict,stability}.csv`.
- **Plots:** `data/analysis/plots/market_making/mm_validation_{markout_curves,breakeven_scatter}.png`.

## Cross-links

Engine + methodology: [[mm_backtesting_methodology_explainer]] (§6) · [[mm_join1_reconciliation_findings]] (JOIN-1) · [[mm_engine_queue_models]] (the queue bracket) · [[2026-06-23_mm_engine_phase01_buildplan]] (Task 4 brief). Design inputs derived from this verdict: [[mm_market_screen_and_ttr_regime_findings]] (the market screen + the time-to-resolution regime). Hub: [[strat_market_making]]. Data: [[mm_clob_capture_semantics]] · [[polymarket_l2_ingestion]]. Next: Join-2 1-contract live calibration; Task 5 v1 strategy (inventory-managed quoter on the VIABLE markets, screened + time-to-resolution-gated).
