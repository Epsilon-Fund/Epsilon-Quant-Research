---
title: "MM Engine — Three Queue Models (Optimistic / RiskAverse / Prob) Behind the Frozen Protocol"
created: 2026-06-29
status: active
owner: justin
project: polymarket
para: resource
hubs:
  - strat_market_making
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - market-making
  - backtesting
  - engine
  - queue-position
---

# MM Engine — Three Queue Models (Optimistic / RiskAverse / Prob)

> Hubs: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · [[COWORK]]
> Engine plan + handoff: [[2026-06-23_mm_engine_phase01_buildplan]] · why queue position is the binding unknown: [[mm_backtesting_methodology_explainer]] §1 · what the public feed can/can't prove (trade-vs-cancel): [[mm_clob_capture_semantics]] · sibling engine note: [[mm_engine_fee_pnl_settlement]].
> Code: `polymarket/research/mm_engine/queue_models.py` · tests: `polymarket/research/tests/test_mm_queue_models.py`.

## Summary

- **What:** the three realism **queue models** that plug into the strategy-agnostic `mm_engine` behind the **frozen** `QueueModel` protocol (`interfaces.py` was *not* touched). They answer the one question a replay backtest can't read off the tape: *given the size resting ahead of our quote, would a trade actually have reached us?*
- **Why:** queue position is the single largest source of backtest over-optimism on anonymous L2 ([[mm_backtesting_methodology_explainer]] §1). We can only ever *bound* it offline, so we ship an **optimistic upper bound, a pessimistic lower bound, and a probabilistic middle**, and report the range until our own live fills calibrate it.
- **The three models** differ in exactly ONE thing — how a **cancel** (a level shrinking with no trade) advances our queue: `OptimisticQueue` credits the whole cancel as ahead of us (upper bound), `RiskAverseQueue` ignores cancels entirely and advances only on real trades (lower bound), `ProbQueue` splits each cancel via hftbacktest's power-law `front^f/(front^f+back^f)` (middle; `f` default 0.5).
- **Takeaway / status:** all three are runnable drop-ins for the engine. Over any one event stream the realized fills bracket as **Optimistic ≥ Prob ≥ RiskAverse** (verified through the real engine and across 20 seeded random streams). Trade-vs-cancel disambiguation and the book re-anchor clamp are unit-tested. `calibrate(live_fills)` is a Phase-2 stub on all three. **60 `mm_engine` tests green (29 prior + 31 new); no regression.** This is engine machinery, **not** a strategy verdict.

## Background: why a queue model at all

A replay backtest cannot move the market — it replays the recorded book + trade events and asks "would my resting order have filled?" The answer hinges on **size-ahead**: a passive maker at the best bid fills only after the size *ahead* of it trades through. If 25 contracts rest ahead and 20 trade, you don't fill. Our public Polymarket L2 is **anonymous** (aggregate price levels, no order IDs, no queue) — see [[mm_clob_capture_semantics]] — so we can never *know* our queue position offline; we can only model it and report optimistic/pessimistic bounds. That is exactly what these three models are.

## The frozen contract (unchanged)

Each model implements `QueueModel` from `interfaces.py`:

- `on_event(ev, book)` — called for **every** market event so the model can track size-ahead.
- `fill(order, book, trade) -> FillResult(qty, queue_ahead)` — called when a trade could touch our resting order; returns the filled quantity and the size still ahead after the trade.
- `get_queue_ahead(order) -> float` — the current size-ahead estimate (snapshotted per quote, no trade needed).
- `calibrate(live_fills)` — Phase-2 hook to tune the model from our own fills.

**On join, queue_ahead = the depth at our price** (we assume we sit at the **back** of the queue). The backtest book excludes our own order, so that depth is purely other makers.

## How `queue_ahead` moves (shared across all three)

`on_event` receives the **post-event** book (the engine applies the event to the book *then* notifies the model). It updates `queue_ahead` like this:

| event at our price | effect on `queue_ahead` |
|---|---|
| **book** snapshot | `queue_ahead = min(queue_ahead, depth_at_our_price)` — re-anchor DOWN to the truth, never raise (size we now see beyond our belief is assumed to have joined behind us). |
| **last_trade** | recorded for coincident-cancel netting (see below); the fill itself is realized in `fill()`. |
| **price_change DECREASE** with no coincident trade = **cancel** | the ONE model-dependent case — attributed per model (next section). |
| **price_change INCREASE** | new liquidity joins **behind** us → no change. |

**Trade-vs-cancel disambiguation (order-invariant).** A trade does not mutate the book directly; PM reports its depletion as a separate (usually same-`ts_exchange`) `price_change`. So when a level shrinks, part of the shrink may be a *trade-through* and part a *cancel*. Within a timestamp the model only **accumulates** the net depth-decrease and the total trade qty per `(token, side, price)`, and attributes the cancel — `max(0, net_decrease − trade_qty)` — **once at the timestamp boundary** (or on `flush_pending_cancels()` at stream end). Because the cancel is never applied before the trade's `fill()`, the fill and the final `queue_ahead` are identical regardless of whether the trade or its `price_change` is processed first. This is the executable, order-invariant version of the capture-semantics rule: same-instant trade + depletion is a fill/consumption, not a pull ([[mm_clob_capture_semantics]] § "What We Can Infer"). See [[mm_queue_model_audit_fixes_findings]] for why the earlier immediate-attribution form was order-dependent and how often the coincidence occurs on real L2.

**Why the trade advance lives in `fill()`, not `on_event`.** The engine calls `on_event(trade)` *then* `fill(trade)` on the same trade (and in live-shadow it never calls `fill` at all). Advancing the queue on the trade in *both* places would double-count it. So `fill()` is the single source of truth for the trade-through advance (it needs the *pre-trade* `queue_ahead` to compute the overflow that becomes our fill), and `on_event(last_trade)` only *records* the trade for the netting above. One honest consequence: in **live-shadow** mode (no `fill` calls) the trade-through advance is not reflected in the `get_queue_ahead` telemetry — a Join-1 reconciliation detail, not a flaw in the model math.

## The three models — differ ONLY in cancel attribution

Let `Δ` = the cancel size (after netting any coincident trade), `front` = current `queue_ahead`, `back` = size resting behind us.

All three apply a shared **logical floor** first: of a cancel of size `Δ`, at most `back` could have rested behind us, so at least `floor = max(0, Δ − back)` MUST have been ahead and is removed regardless of model. The model's raw attribution is then clamped into `[floor, Δ]`: `advance = min(max(raw, floor), Δ)`.

| model | raw cancel attribution | role |
|---|---|---|
| **OptimisticQueue** | `Δ` (the whole cancel is assumed ahead) | **upper bound** on fills |
| **RiskAverseQueue** | `0` → clamped up to the floor `max(0, Δ − back)` | **lower bound** on fills |
| **ProbQueue** | `prob_ahead · Δ`, `prob_ahead = front^f / (front^f + back^f)` → clamped into `[floor, Δ]` | **middle** (hftbacktest power-law; `f` default 0.5) |

`prob_ahead` is hftbacktest's `PowerProbQueueFunc` (the function behind `ProbQueueModel`/`ProbQueueModel2`). At `f=1` it is the plain `front/(front+back)` ratio; at the default `f=0.5` a symmetric queue (`front==back`) attributes half the cancel ahead. It is always in `[0,1]` (nothing ahead → 0; nothing behind → 1), and the floor/clamp keeps every advance in `[floor, Δ]`, which is precisely what guarantees ProbQueue sits between the two bounds. The floor also makes a cancel advance **fully** when nobody is behind us (`back == 0` ⇒ the cancel *must* be ahead — all three models agree) and keeps `queue_ahead ≤ level depth`. `f` is the knob `calibrate()` will fit to our own live fills; until then it is a stated assumption, and ProbQueue results should be read as the *middle of a bound*, not a point estimate. (See [[mm_queue_model_audit_fixes_findings]] for the floor's full justification and the other two engine-wiring fixes.)

### Worked example (inspect the divergence)

Our bid rests at 0.47 with **50 contracts ahead** of us. Then:

1. **+40 join behind us** (level 50 → 90). `queue_ahead` stays 50 for all three; `back` is now 40.
2. **cancel 30** at 0.47, no trade (level 90 → 60). `back=40 ≥ 30` so `floor=0`. Optimistic: 50 → **20**. RiskAverse: floor 0 → stays **50**. Prob (`f=0.5`): `prob_ahead = √50/(√50+√40) ≈ 0.53` → 50 − 0.53·30 ≈ **34**.
3. **cancel 20** at 0.47, no trade (level 60 → 40). Now `back=10 < 20` so `floor = 20 − 10 = 10`. Optimistic: 20 → **0**. RiskAverse: advances the floor 10 → **40**. Prob: ≈ **23**.
4. A **60-lot SELL trades through** at 0.47. `fill()` consumes our `queue_ahead` first, the overflow fills us:
   - Optimistic: 0 ahead → **60 fills**.
   - RiskAverse: 40 ahead absorbs 40 → **20 fills**.
   - Prob: ~23 ahead absorbed → **~37 fills**.

So on the same tape: **Optimistic 60 ≥ Prob 37 ≥ RiskAverse 20.** The Optimistic/RiskAverse spread (60 vs 20) is the honest width of our offline ignorance about cancellations; ProbQueue is our best single guess inside it. The RiskAverse advance of 10 in step 3 is the *logical floor* — a cancel of 20 with only 10 resting behind us forces at least 10 to have been ahead, even for the most pessimistic model. (This is the exact stream in `test_bracket_strict_through_engine`.)

## What was tested

- **Bracketing (the headline sanity check):** `Optimistic ≥ Prob ≥ RiskAverse` realized fills — proven both through the real engine on the crafted stream above (strict 60/≈37/20) and across **20 seeded random** cancel/increase/trade streams driven through the engine's exact `on_event → fill` order. A non-vacuous check (liquidity behind us, small cancels) confirms the models genuinely diverge, not just tie.
- **Logical floor (FIX 2):** `back == 0` ⇒ all three advance fully; `back ≥ Δ` ⇒ RiskAverse advances 0; `0 < back < Δ` ⇒ RiskAverse advances exactly `Δ − back`.
- **Order invariance (FIX 3):** a trade and its same-`ts_exchange` depletion `price_change`, fed in both orders, yield identical fills **and** identical final `queue_ahead` for all three models (both an overflow-fill case and a residual-cancel case).
- **forget() teardown (FIX 1):** a re-quote at a used price re-seeds `queue_ahead` at the back of the current level — verified through the engine after a fill, after a reconcile-driven cancel (price move), and after a gap cancel — instead of inheriting the stale (over-optimistic) value.
- **Trade-vs-cancel disambiguation:** pure cancel (per-model), coincident trade not double-counted, mixed trade+cancel split.
- **Re-anchor clamp:** a full `book` snapshot lowers `queue_ahead` to the observed depth (RiskAverse's stale belief 50 → 40) and never raises it.
- **Edge cases:** size-increase = no change; `prob_ahead` bounds/form; the `ProbQueue.f` knob; `QueueModel` protocol conformance; `calibrate` no-op stub.

**Result:** **68 `mm_engine` tests green** (29 engine + 39 queue), no regression. Three audit fixes (forget wiring, logical floor, order-invariant netting) plus a ProbQueue power-law hardening are detailed in [[mm_queue_model_audit_fixes_findings]], which also records a six-lens adversarial review (verdict: SHIP).

## Read / interpretation

- A "positive" maker result must be reported as a **band**, not a point: run Optimistic and RiskAverse as the bracket and ProbQueue as the central estimate. If a strategy only looks good under Optimistic, that is the over-optimism the methodology explainer warns about, not an edge.
- The width Optimistic − RiskAverse is itself a diagnostic: in cancel-heavy books (lots of posted-and-pulled liquidity) the band is wide and the fill rate is genuinely uncertain offline; in trade-dominated books it is narrow.
- These models do **not** resolve adverse selection or our *true* fill rate — those remain **live-only** unknowns ([[mm_backtesting_methodology_explainer]] §1b). The models bound queue position; they don't manufacture the live signal.

## Decision and next step

- **Done / shippable now:** the three models are drop-in `QueueModel` implementations; any engine run can swap them to get the fill-rate band. No interface change, no strategy claim.
- **Join 1 (reconciliation):** run replay + live-shadow over the same dates with each model and confirm same-code-path consistency; note the live-shadow `get_queue_ahead` caveat above when comparing queue telemetry.
- **Join 2 (calibration):** with real 1-contract fills, implement `ProbQueue.calibrate(live_fills)` to fit `f` so modeled fill rate matches observed — this is when the bound *collapses* toward the live-measured rate and the backtest earns "reliable." Until then ProbQueue is a stated assumption, not a measurement.
- **Do NOT** treat any maker result as deployable on the Optimistic model alone, and do NOT add latency gating inside the queue models — that is the engine's job (`fills.py`).
