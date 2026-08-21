---
title: "MM Join-2 model-fragility audit — does the politics maker sign survive the two suspect models?"
created: 2026-07-09
status: active
owner: justin
project: polymarket-mm
hubs:
  - strat_market_making
  - COWORK
tags:
  - market_making
  - validation
  - execution
  - join2
  - fragility
---

# MM Join-2 model-fragility audit — does the politics maker sign survive the two suspect models?

> Hub: [[strat_market_making]] · [[COWORK]]
> Compact-column definitions: [[polymarket_table_dictionary]]. Prior context: [[mm_task5_1_neutral_quoter_cpcv_findings]] · [[mm_task5_inventory_quoter_findings]] · [[mm_engine_join2_machinery_findings]] · [[mm_engine_join2a_bridge_findings]] · [[mm_join1_reconciliation_findings]] · PRD [[2026-07-09_mm_join2_validation_prd_reference]].

## Plain-English Summary

- **What this is.** Join-2 Part A: an *offline* fragility test of the two models the whole politics market-making verdict rests on — (1) the **queue-fill model** (how much of a resting quote fills, `ProbQueue.f` + latency) and (2) the **executable-touch costing convention** (how open inventory is marked). If either is mis-specified, the razor-thin politics "sign" flips. This note asks: under which perturbations does it flip, under which does it hold, and are the models trustworthy enough to justify spending a few real dollars to calibrate them.
- **The "sign" at stake.** The full neutral-spike quoter stack (NSQ `damp[k=5e-6,cap=500,w=20,d=0.6]`) shows a **positive** costed edge on the politics capture (~**+0.29¢/contract** honest, **+0.34¢** even under the pessimistic queue) versus a **losing** symmetric baseline (−0.13¢). But the formal *beats-baseline* gate spans zero at K=11 event groups. Is that positive direction real or an artifact of the two models?
- **What I could and could not run.** The raw politics L2 capture (`~/epsilon_l2_full`), the materialized token parquets, and the raw per-run rows were all in a prior session's scratchpad and have been **cleaned** — only the aggregated Task-5.1 CSVs survive. So **A1 (model report) and A2 (baseline-luck) are fully computed offline; A3 (latency sweep) and A4 (costing-convention sweep) are delivered as *structural/analytical* reasoning from the frozen code, NOT measured numbers.** This was an explicit operator decision (2026-07-09) to deliver Part A offline rather than re-clone from R2. Every analytical claim is labelled as such.
- **Headline verdict.** The politics maker sign is **directionally robust and not baseline-luck** (A2: positive under every resampling, carried by 8/11 groups, leave-one-group-out stable). The **costing convention is conservative, not inflating** (A1: it marks to the executable touch, never mid — the historical K-PEG bias is provably absent in the code). The **queue-fill uncertainty is bracketed and sign-consistent** (the number is positive across the whole Optimistic→Prob→RiskAverse bracket). The **one genuinely unvalidated input that could still move the thin edge is latency**, because the entire result was computed at **0 ms** (`ConstantLatency(0.0)`, hard-coded) and 0 ms flatters. → **The models are trustworthy enough to justify a tiny, bounded, gated live *measurement* loop** (measure latency, get first fills), and **not yet enough to call it a trading system.** This is a "merits a live MEASUREMENT loop," per `brain/CODEX.md` realism rule 3.

---

## Worked example (so the abstract stays concrete)

One politics event group is, e.g., a NegRisk market "who wins the 2026 X primary" with several outcome tokens. We rest a two-sided quote (bid just below, ask just above the touch) on each liquid token for the market's whole life. **A fill** = a public trade prints through our resting price and the queue ahead of us is exhausted. **Costed PnL for that group** = the realized round-trips we closed + the change in the *liquidation-marked* value of whatever inventory we're still holding at the window edges (a long marked at the best *bid* — what we'd actually get if we dumped it now). The question of this note: if we'd assumed a slower fill (higher latency) or marked that leftover inventory differently (mid instead of touch), would this group's + turn into a −, and would the politics book as a whole flip from "the NSQ stack beats a losing baseline" to "no"?

---

## A1 — What the four models actually compute (read, not changed)

> **Path correction (the PRD flagged its paths as guesses).** The PRD named `mm_eval/tape.py` as the costing/marking module. It is **not** — `tape.py` is the *causal public-trade injection* tape (feeds `last_trade` prints to the toxicity lens without touching the frozen `Strategy` interface). The **executable-touch costing/marking lives in `mm_eval/protocol.py`** (`_liq_mark`, `_touch_series`, `windowed_costed`) and its whole-lifecycle/τ-regime twin in `mm_eval/cpcv.py` (`costed_spans`, `daily_series`). Reported so the next agent doesn't audit the wrong file.

### (1) Costing / marking — `mm_eval/protocol.py::_liq_mark` + `windowed_costed`; `cpcv.py::costed_spans`

**What it computes.** Window costed PnL = `realized_delta` summed over fills inside the window (the round-trips we actually closed) **+** the change in the *liquidation mark* of open inventory across the window edges:
`costed = realized + (liq_mark(b) − liq_mark(a))`.
`_liq_mark(t)` reconstructs position `q` and cost basis at `t` from the fills up to `t`, then marks at the **executable touch**: a **long → best bid**, a **short → best ask** (`exit_px`), i.e. `q·(exit_px − basis)`. The touch series (`best_bid`/`best_ask`) comes from the engine's own quotes log.

**Fees / rebates.** In `fees.py`, `fee = fee_rate·qty·p·(1−p)` and `rebate = rebate_rate·fee`. The captured politics markets are **fee-free** (`fee_rate = 0`), so the primary costed number is **fee/rebate-agnostic**: rebate = 0, and it is reported separately from the costed PnL (never folded in). A "representative" schedule (Politics `0.04/0.25`) is a *borrowed* sensitivity only (CODEX realism rule 2), and a rebate can only **add** to a maker's edge, never subtract.

**Is anything marked to mid?** **No.** `_liq_mark` marks strictly to the touch. This is the deliberate fix for the historical **K-PEG mark-to-mid artifact** (a maker holding inventory looks profitable at mid but pays half the spread to actually exit). Verified by reading the code: there is no mid-marking path in the costed pipeline. **The convention is conservative, not inflating.**

**Where it could be wrong.** (a) It assumes the *entire* open inventory can be liquidated at the *top-of-book* touch with zero walk-down / market impact. For a 5-share live inventory that is realistic; for the large inventories the *symmetric baseline* carried to resolution it is optimistic on depth. (b) It marks at the *last-observed* touch as-of the window edge — a stale book at the edge yields a stale mark. (c) Nothing here models slippage beyond L1.

### (2) Gating — `mm_eval/cpcv.py` (+ the older single-cut `protocol.py`)

**Split unit = the whole market (NegRisk event group), never the token.** A group's entire τ arc (mid-life + endgame) stays on one side of every split, so complementary legs of one event can't leak across the IS/OOS boundary. τ (hours-to-resolution) is a *conditioning* axis of the reported surface, never a split axis. This is the fix for Task-5's broken single-calendar cut (which put the same markets on both sides → IS=calm, OOS=endgame regime confound → the spurious `PBO≈0.5`).

**Group-CPCV** (`generate_group_cpcv_splits`) is a port of the crypto `cpcv_engine` (López de Prado combinatorial purged CV) with the unit changed bars→groups; folds are **cohort-balanced** (aggressiveness × liquidity, from a leakage-safe lead-in window only). **Nested**: knobs are inner-selected on the outer-training groups only, then scored once on held-out groups (kills the selection seam). **PBO** = real CSCV with event-groups as the exchangeable blocks; **DSR** = Sharpe deflated by the *effective* number of trials; **White's Reality Check** over the config set's daily costed PnL.

**Why PBO ≈ 0.00.** Selection transfers ~1:1 IS→OOS under the whole-market split (the IS-best config is the OOS-best), so the probability of backtest overfitting is ~0. This is a *selection-stability* statement, not a *magnitude* one.

**K and why the CIs are overconfident.** Politics has **K = 11 event groups** (esports 16), over the 2026-06-19→07-07 span. The honest caveat baked into the module (`fold_overlap_diagnostic`, and stated in the docstrings): politics groups trade **concurrently** (calendar overlap ≈ 1.00), so a group bootstrap that treats the 11 groups as independent **overstates** the information — the effective independent K is well below 11, and shared macro-news is a residual leakage channel the module *reports* rather than purges (an ordering-purge between simultaneous markets is symbolic). → OOS CIs are **news-correlated / overconfident**; the discipline is to lean on IS + mechanism + PBO and read OOS as *directional*.

### (3) Queue models — `mm_engine/queue_models.py`

Three variants behind one frozen protocol, differing in exactly one thing — how a **cancel** is attributed between the size ahead of us and behind us:

- **`OptimisticQueue`** — every cancel is *ahead* → we advance fully → **upper bound on fills**.
- **`RiskAverseQueue`** — cancels *behind* except a logical floor (`max(0, Δ−back)`, the volume that provably can't fit behind us) → **lower bound on fills** (hftbacktest's `RiskAverseQueueModel`, made logically tight).
- **`ProbQueue(f)`** — power-law split: fraction taken from ahead = `frontᶠ/(frontᶠ+backᶠ)` → **between the bounds**.

The realized fills provably bracket **Optimistic ≥ Prob ≥ RiskAverse**. Two correctness properties are baked in: the **logical floor** (a cancel advances fully when nobody is behind us; keeps `queue_ahead ≤ depth`) and **order-invariant trade-vs-cancel netting** within a timestamp (a trade's depletion `price_change` is netted against the trade so the same event stream fills identically regardless of arrival order).

**How `f` is currently set.** Every eval/gate run uses **`ProbQueue(0.5)` — the fixed default, unfitted.** `calibrate(live_fills)` is a **Phase-2 stub** (public anonymous L2 can't reveal our own fill rate). The machinery note's "synthetic-recovery fit" only proved the *fitter recovers a known f* on synthetic fills — it did **not** set the eval's `f`. So the shipped gate uses the **pessimistic** bound (RiskAverse) as its keep-gate and reports `Prob(0.5)` as the middle. The **live fills from Part C (C6) are what would first fit `f` to reality.**

### (4) Latency — `mm_engine/latency_models.py` + the fill gate in `fills.py`

`fills.py` gates every fill: a resting order can only fill a trade once it could have *landed* — `placement_ts + round_trip_ms(placement_ts) ≤ trade.ts_exchange`. `ConstantLatency` defaults to a realistic **~200 ms**, and `SampledLatency` is a Phase-2 placeholder drawing from a fitted `(mean, std)`.

**Critical fact for A3:** the entire Task-5.1 eval **and** the bridge default run at **`ConstantLatency(0.0)`** (hard-coded in `mm_eval/runner.py`, `scripts/mm_task5_1_ladder_run.py`, and `BridgeConfig.latency_ms=0.0`). So **the surviving politics number is the 0 ms result** — the optimistic end. 0 ms means *any* order placed on an earlier event is live; a real round-trip removes fills a late quote would miss. Latency is market-stratified: ~immaterial for slow politics, essential for fast crypto/in-play sports.

### Join-1 lock — what "locked to real tape" does and does not mean

Per [[mm_join1_reconciliation_findings]] / [[mm_engine_join2a_bridge_findings]]: Join-1 established that the engine, replayed against recorded tape, reproduces the recorded run with a **0% record→replay gap**, the **Optimistic ≥ Prob ≥ RiskAverse bracket holds**, and there are **0 `queue_ahead > depth` violations**. That proves **consistency / determinism** — the fill machinery is internally correct and reproducible. It does **not** prove **real-fill realism** — whether our modeled fills match what a live resting order would actually get. Real-fill realism is exactly what Part C begins to measure (latency probe + first fills → C6 calibration).

---

## A2 — Baseline-luck robustness (MEASURED, offline)

**What A2 tests.** Is the politics "NSQ beats a losing baseline" result carried by one lucky group / one partition, or is it robust to resampling? **Data note / limit:** the strongest form (re-running *nested selection under alternate CPCV fold partitions*) needs the raw config×group matrix, which is gone. What survives is each rung's **nested-CPCV honest per-group series** (already OOS estimates) in `mm_task5_1_groups_honest.csv`. So A2 stresses the paired per-group **(rung − baseline)** honest delta — exactly the driver's keep-gate (`group_cluster_delta`) — plus the **absolute level** of the stack, under mean-bootstrap, volume-pooled bootstrap, and leave-one-group-out (LOO) jackknife. Artifact: `data/analysis/csv_outputs/market_making/mm_join2_a2_baseline_luck.csv` (gitignored/regenerable).

**Column meanings.** `mean_delta_c` = equal-weight mean over the 11 groups of (rung − baseline) honest ¢/contract, with `[lo,hi]` = 95% group-bootstrap CI. `pooled_delta_c` = volume-weighted (Σ$/Σqty) delta, same CI. `groups_beat_base` = how many of 11 groups have rung > baseline. `loo_min/max` = the mean delta's range as each single group is dropped; `loo_all_pos` = does it stay positive dropping *any* one group.

| rung | mean Δ¢ | mean 95% CI | pooled Δ¢ | pooled 95% CI | groups>base | LOO range | LOO all + |
|---|---|---|---|---|---|---|---|
| nsq_core | +0.314 | [−0.86, +1.35] | −0.011 | [−1.37, +1.17] | 8/11 | [+0.06, +0.68] | yes |
| nsq_lens | +0.360 | [−0.81, +1.38] | +0.075 | [−1.28, +1.24] | 8/11 | [+0.11, +0.73] | yes |
| nsq_asym | +0.644 | [−0.56, +1.72] | +0.400 | [−1.02, +1.64] | 8/11 | [+0.38, +1.02] | yes |
| **nsq_damp** (shipped) | **+0.651** | **[−0.53, +1.71]** | **+0.425** | **[−0.97, +1.65]** | **8/11** | **[+0.39, +1.01]** | **yes** |
| as_rung1 | +0.209 | [−0.86, +1.11] | −0.101 | [−1.35, +0.97] | 8/11 | [+0.06, +0.54] | yes |
| as_rung2 | +0.296 | [−0.79, +1.24] | −0.062 | [−1.36, +1.08] | 8/11 | [+0.09, +0.63] | yes |
| basket | +0.297 | [−0.90, +1.35] | −0.082 | [−1.45, +1.11] | 8/11 | [+0.03, +0.66] | yes |

**Absolute honest level (the sign of the number itself, not vs baseline):**

| rung | mean ¢ | mean 95% CI | pooled ¢ | groups>0 |
|---|---|---|---|---|
| baseline | −0.369 | [−1.46, +0.83] | −0.131 | 3/11 |
| nsq_asym | +0.275 | [+0.07, +0.48] | +0.269 | 8/11 |
| **nsq_damp** | **+0.282** | **[+0.075, +0.494]** | **+0.294** | **8/11** |

**Read.**
- **Not baseline-luck.** Every NSQ/A-S rung beats baseline on the *point* estimate, in **8/11 groups**, and — decisively — the mean delta stays **positive under every leave-one-group-out** (dropping the single most-favorable group never flips it). The direction is not one-group-driven.
- **Level lower-CI clears zero (naively).** The absolute level of the NSQ stack has a mean-bootstrap **lower CI above zero (+0.075¢)** — a stronger statement than the *delta-vs-baseline* gate, which spans zero only because **baseline itself is noisy** ([−1.46, +0.83], just 3/11 groups positive). The formal driver gate drops damp on the *delta* CI; the *level* is what actually decides "does this make money," and it is positive with a lower CI above zero.
- **Two honest deflators (why this is NOT a certification).** (a) **Concurrency**: the group bootstrap treats 11 concurrent politics groups as independent; overlap ≈1.00 means the effective K is far smaller → the level's lower-CI-above-zero is **overconfident** (CODEX rule; do not upgrade to "certified"). (b) **Materiality/capacity**: the edge is **weaker volume-weighted than equal-weighted** (mean Δ +0.651 vs pooled +0.425; baseline mean −0.369 vs pooled −0.131) → the edge concentrates in **lower-volume groups**, a capacity flag (CODEX rule 4: statistical survival ≠ economic materiality).

**A2 verdict: the politics sign HOLDS under baseline-luck perturbation — directionally robust, LOO-stable, not a partition artifact — but remains a *thin, concurrency-deflated* signal, i.e. a data-collection problem, not a knobs problem.**

---

## A3 — Latency sensitivity (ANALYTICAL — blocked on data, not measured)

> **Cannot be run offline.** Latency gates fills *inside the engine* (`fills.py`); there is no way to recompute fills at latency > 0 from the aggregated CSVs. Re-running needs the L2 capture (gone). The following is the **mechanism-level prediction** from the frozen code + the known tape structure, explicitly unmeasured. Part C's `--mode mm_latency` probe + C6 fill calibration are what settle it.

**Mechanism.** Raising the round-trip `L` means a quote placed at `t` can only fill trades at `t' ≥ t + L`. For a **passive maker** this cuts two ways: it removes **benign** fills (we weren't live yet) *and* protective-removes **toxic** fills (a fast adverse sweep right after we (re)quote hits before we're live → we dodge it). This is the opposite sign to a taker, for whom latency is pure cost.

**Prediction for politics (slow market).** The latency model's own docstring calls latency "a minor refinement for slow politics." Politics inter-trade times are long relative to plausible round-trips (200 ms–~1 s), so a realistic 200 ms should remove **very few** fills → the politics **level** is expected to be **latency-robust in [0, ~1 s]**, and the sign should **not** flip from latency alone in that range.

**Where it could still bite (the residual risk the 0 ms number hides):**
1. **The delta, not the level.** Latency partly *substitutes* for the NSQ toxicity gate (both dodge fast adverse fills). Adding latency helps the **defenseless symmetric baseline** more than the already-gated NSQ stack → it **shrinks the damp−baseline delta** even if the NSQ level barely moves. Since the delta already spans zero, latency erodes it further.
2. **Fills on fresh requotes.** If a non-trivial fraction of NSQ fills land on *just-repriced* quotes (the stack reprices asymmetrically), a real 200 ms+ latency voids those specific fills — and if they are the edge-carrying ones, the thin level moves.
3. **Measured ≫ assumed.** If the real CLOB/cloud round-trip is ≫ 200 ms (rate-limit backoff, cloud RTT), *and* politics trades cluster more tightly than assumed, the removed-fill fraction is larger than predicted.

**A3 verdict (analytical): sign EXPECTED to HOLD for slow politics at realistic latency; the damp−baseline *delta* is the vulnerable quantity, not the *level*; a flip requires measured latency ≫ assumed AND tight trade clustering. This is UNMEASURED and is precisely why Part C measures latency first (C4) and calibrates on real fills (C6).**

---

## A4 — Costing-convention sensitivity (ANALYTICAL — blocked on data, not measured)

> **Cannot be run offline.** Re-costing under alternative marks needs the raw per-fill `realized_delta` + the touch series (both gone with the L2 data). Reasoning from `_liq_mark` + the Task-5 inventory findings; unmeasured.

**The lever.** Costed PnL = realized round-trips + Δ(liquidation mark of open inventory). Only the **inventory-mark** term is convention-dependent, and it scales with **|end inventory| × (mark − touch gap)**. Conventions, ordered generous→conservative:
- **mark-to-mid** (generous; the K-PEG bias): overstates by ~½-spread × |inventory| at each edge.
- **spread-fraction** (`touch ± φ·spread`): between mid and touch.
- **executable-touch** (current, conservative): long→bid, short→ask.
- **harsher-than-touch** (walk-the-book / slippage for large inventory; or a *borrowed* structural baseline like the crypto-4h 1.98¢): a warning, not a legitimate gate for politics unless re-derived (CODEX rule 2).

**Direction of the effect on the sign.** The Task-5 finding is that the **symmetric baseline carries large inventory** to resolution (its whole loss is carry cost), while the **NSQ stack ends near-flat** (tight cap). Because the mark-convention term scales with end inventory:
- Switching executable-touch → **mid** inflates the **inventory-heavy baseline** far more than the near-flat NSQ → it **shrinks the damp−baseline delta** (baseline looks less bad). The **NSQ level** barely moves (little inventory to re-mark).
- So the current **executable-touch is the convention most *favorable to the NSQ advantage*** — it maximizes the delta by honestly penalizing the baseline's carry. A *more generous* mark can only **shrink the advantage**, and the NSQ **level** stays positive (near-flat book ⇒ low mark-sensitivity).
- **Fee/rebate on/off cannot flip the sign negative:** captured fee = 0, and a representative rebate only **adds** to a maker's per-contract edge.

**A4 verdict (analytical): the NSQ *level* is EXPECTED robust to the costing convention (near-flat book ⇒ low mark-sensitivity); the damp−baseline *delta* is convention-sensitive but the current executable-touch is the *conservative* end (it does not inflate — it is the historical de-biasing fix); the only conventions that flip the sign are *harsher-than-touch* ones that are borrowed/unwarranted for politics per CODEX rule 2. UNMEASURED; a small re-cost sweep on restored data would confirm.**

---

## Sign-flip map (the deliverable)

| Perturbation | Status | Does the politics sign flip? |
|---|---|---|
| **A2** alternate resampling / LOO / pooled-vs-mean | **MEASURED** | **No — holds.** Positive under every resample; 8/11 groups; LOO all-positive. Not baseline-luck. Deflated by concurrency + concentrated in low-volume groups → thin, not certified. |
| **A3** latency 0 → ~1 s (slow politics) | **ANALYTICAL** | **Expected no** for the *level*; latency **erodes the damp−baseline *delta*** (helps the defenseless baseline). Flip needs measured latency ≫ 200 ms + tight trade clustering. UNMEASURED — Part C. |
| **A4** executable-touch → mid / spread-frac | **ANALYTICAL** | **Expected no** for the *level* (near-flat NSQ book). Touch is the *conservative* end; generous marks only **shrink the advantage**, not flip the level. UNMEASURED. |
| **A4** fee/rebate on/off | **ANALYTICAL** | **No — cannot flip negative** (fee=0 captured; rebate only adds). |
| **Queue bracket** Optimistic/Prob/RiskAverse | **MEASURED** | **No — holds across the whole bracket** (damp level +0.336 / +0.339 / +0.336¢; sign-consistent). `f=0.5` is unfitted but the bracket is narrow *and* positive throughout. |
| **Harsher-than-touch mark / borrowed structural baseline** | n/a | Would flip it, but is *borrowed/unwarranted* for politics (CODEX rule 2) — a warning, not a gate, unless re-derived on politics. |

---

## Verdict — are the two models trustworthy enough to justify spending real money to calibrate them?

**Yes — trustworthy enough for a tiny, bounded, gated live *measurement* loop; not yet enough to call it a trading system.**

- **Costing model (executable-touch):** trustworthy. It is the *conservative* convention; the historical mark-to-mid inflation is provably absent in the code; it does not manufacture the edge (if anything it under-credits the inventory-heavy baseline). No red flag.
- **Queue-fill model — attribution (`ProbQueue.f`):** trustworthy *as bracketed*. `f=0.5` is unfitted, but the sign is positive across the entire Optimistic→RiskAverse bracket, and the gate already uses the pessimistic bound. The attribution uncertainty is bounded and sign-consistent.
- **Queue-fill model — latency:** **the one genuinely unvalidated input.** The whole result is the **0 ms** number, which flatters. Analysis says slow politics should be robust, but *"should be"* is not *"is,"* and 0 ms is exactly the assumption most likely to be wrong in the direction that matters (it inflates fills and shrinks the case for the toxicity gate). **The only way to retire it is to measure real latency and get real fills** — which is what a few dollars buys.
- **A2 anchors the decision:** the sign is not luck. The direction is real; it is *thin and concurrency-deflated*, i.e. a **data-collection** problem. Spending ~$3–5 (bounded by the 5-share order size and the inventory cap) to obtain (a) a **measured latency** (Part C, C4) and (b) a **first handful of real fills** (C5→C6 `QueueModel.calibrate`) directly retires the single unvalidated model input and replaces `f=0.5`/`0 ms` assumptions with measured values. That is a **live MEASUREMENT loop, not a trading system** (CODEX rule 3) — the exit state the PRD asks for.

**Caveat that outlives this note:** this verdict is contingent on the analytical A3/A4 being confirmed numerically once the L2 data is restored. If a future session re-clones politics from R2, run the latency sweep + costing sweep to replace the two ANALYTICAL rows above with measured numbers before any scale decision.

---

## Part B — the measurement instrument (confirmed)

The C5 loop runs the **naive join-the-touch quoter** as the measurement instrument, and **no config change is needed beyond order size**. Concretely: `mm_engine_bridge.py` drives `mm_engine.strategies.SymmetricQuoter` (quotes `mid ± half_spread`, which sits *at* the touch when `half_spread` = the prevailing half-spread — the "two-sided join-the-touch quote" the runbook describes), with size taken from `MAKER_SIZE_CONTRACTS`. For C5 that is set to **5 shares** (the venue minimum — verify against py-clob-client at pre-flight); the quoter, queue model, and latency plumbing are otherwise unchanged. Its virtue as an instrument is that it has **no skew / no inventory logic / no toxicity gate**, so strategy behavior does not confound the "does the fill model predict reality" read.

The **NSQ / inventory-aware Task-5 stack** (microprice skew + one-sided cap + near-expiry flatten + two-lens toxicity gate, politics config `damp[k=5e-6, cap=500, w=20, d=0.6]`) is the **later alpha-certification candidate** and is **explicitly out of scope** here — no NSQ wiring, no strategy pre-registration in this pass. It is what a *future* Phase-2 loop A/B-tests once the fill/latency models are grounded by the join-the-touch measurement.

---

## Status / next

- **A:** fragility report complete — A1 (four-model report + path correction), A2 (measured, sign holds), A3/A4 (analytical, sign expected to hold; latency = the one unvalidated input), sign-flip map + trustworthiness verdict. `mm_engine` + `mm_eval` suites **172 passed**. Artifacts: `mm_join2_a2_baseline_luck.csv`.
- **B:** join-the-touch confirmed as the C5 instrument (SymmetricQuoter via the bridge, size = 5 via `MAKER_SIZE_CONTRACTS`, no other config change); NSQ recorded out-of-scope.
- **C (LIVE — gated):** awaiting operator go-ahead. First step C1 = authed read-only balance probe. **Nothing committed; no orders placed.**
- **Data debt:** A3/A4 are analytical because the politics L2 capture was cleaned from scratch. To make them numeric, re-clone `parquet/*/politics_negrisk/` (06-19→07-07) from R2 and re-run a latency sweep + costing sweep via new scratch scripts (the frozen engine/eval are untouched and ready).
