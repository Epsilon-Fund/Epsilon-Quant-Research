---
title: "MM Queue Models — Three Audit Fixes (forget() teardown, logical floor, order-invariant netting) + L2 Coincidence Measurement"
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

# MM Queue Models — Three Audit Fixes + L2 Coincidence Measurement

> Hubs: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · [[COWORK]]
> Parent note (the three models in full): [[mm_engine_queue_models]] · engine plan: [[2026-06-23_mm_engine_phase01_buildplan]] · why queue position binds: [[mm_backtesting_methodology_explainer]] §1 · trade-vs-cancel data limits: [[mm_clob_capture_semantics]].
> Code: `mm_engine/queue_models.py`, `mm_engine/orders.py`, `mm_engine/engine.py` · tests: `tests/test_mm_queue_models.py` · measurement: `scripts/mm_queue_coincidence_measure.py`.

## Summary

- **What:** three correctness fixes to the MM-engine queue models + their engine wiring, found in an audit of the [[mm_engine_queue_models|three-queue-model]] build. The frozen `interfaces.py` was **not** touched.
- **Why:** each fix removed a way the backtest could *over-state* fills (the exact over-optimism the methodology explainer warns about) or behave inconsistently depending on event order.
- **The three fixes:** (1) **`forget()` teardown** — per-order queue state is now torn down when an order is cancelled/replaced/filled, so a re-quote at a used price re-seeds at the back of the queue instead of inheriting a stale `queue_ahead`; (2) **logical floor** — every cancel advances by at least `max(0, Δ − back)`, the volume that cannot fit behind us, which makes RiskAverse logically tight and forces a full advance when nobody is behind us; (3) **order-invariant netting** — trade-vs-cancel netting is deferred to the timestamp boundary so a trade and its same-`ts_exchange` depletion `price_change` give identical results regardless of arrival order.
- **Measurement:** on the captured L2 in `./l2_data/` (175,260 trades), **1.33%** of trades share a timestamp with a same-(token, price, resting-side) `price_change` — the exact case fix (3) makes order-invariant — rising to **2.09%** in faster esports markets vs **0.16%** in slow politics markets. The bug fix (3) prevents is real but bounded, and concentrated where the methodology explainer says realism matters most: fast markets.
- **Status / takeaway:** **68 `mm_engine` tests green** (29 engine + 39 queue), no regression; a six-lens adversarial review returned SHIP (see below). Engine machinery only — no strategy verdict.

## Background

The MM engine asks, for each resting quote, "would a real trade have reached us?" given the size resting **ahead** of us (`queue_ahead`). Public Polymarket L2 is anonymous, so `queue_ahead` can only be *modelled* and reported as an optimistic/pessimistic band ([[mm_engine_queue_models]]). The audit found three places where the model or its wiring could silently inflate fills or depend on event order. All three are now fixed and tested.

---

## FIX 1 — `forget()` teardown so re-quotes re-seed at the back

**The bug.** Queue state is keyed by `(token, side, price)`. `QueueModel.forget()` existed but was **never called**. So when an order was cancelled, replaced, or fully filled and the strategy later re-quoted at the **same** price, the new order inherited the old order's `queue_ahead` — which after a fill is `0`. A fresh quote would therefore look like it was already at the front of the queue and fill on the very next trade-through: a pure over-optimism artifact.

**The fix.**
- `OrderManager.reconcile()` and `.cancel_all()` now return `(ops, removed)` where `removed` is every `ActiveOrder` that left the book this tick — a cancel, or the **old side** of a replace. `.drop_filled()` now returns the removed `ActiveOrder`s (not just client ids).
- `engine.py` calls `queue_model.forget(ao.order)` for every removed order after reconcile, drop-filled, and gap cancel-all.

**Worked example.** We rest BUY 0.47 behind 50 contracts. A 200-lot SELL fills our 100 and the order is dropped → `forget` clears the `(YES, BUY, 0.47)` queue state. We immediately re-quote BUY 0.47; it re-seeds at the **current** level depth (50, back of the queue). A later 30-lot SELL then does **not** reach us (50 ahead absorbs it). With the bug, the re-quote would have inherited `queue_ahead = 0` and wrongly filled 30 more. Tested through the engine for the fill path, the reconcile-cancel path (price move), and the gap path.

---

## FIX 2 — shared logical floor on cancel attribution

**The issue.** RiskAverse previously "ignored every cancel" (advance only on real trades). But if a level shrinks by `Δ` and only `back` was resting behind us, then at least `Δ − back` of that cancel **must** have been ahead of us — ignoring it is not just pessimistic, it is logically impossible and lets `queue_ahead` exceed the level depth.

**The fix (all models).** In the deferred flush, after computing the cancel `Δ`, `front = queue_ahead`, and `back`:

```
floor   = max(0, Δ − back)                 # cancel volume that cannot fit behind us
advance = min(max(model_raw_attribution, floor), Δ)
queue_ahead -= advance
```

- Optimistic raw = `Δ` → `advance = Δ` (unchanged).
- RiskAverse raw = `0` → `advance = floor` (logically tight, not naive zero).
- ProbQueue raw = `prob_ahead · Δ` → clamped into `[floor, Δ]`.

**Properties (all unit-tested):** `back == 0` ⇒ every model advances fully (a cancel with nobody behind us must be ahead); `back ≥ Δ` ⇒ RiskAverse advances 0; `0 < back < Δ` ⇒ RiskAverse advances exactly `Δ − back`; and `floor ≤ advance ≤ Δ` always, so the bracket **Optimistic ≥ Prob ≥ RiskAverse** is preserved while `queue_ahead ≤ level depth` is guaranteed.

---

## FIX 3 — order-invariant trade-vs-cancel netting

**The bug.** A trade does not mutate the book; PM reports its depletion as a separate `price_change` that usually shares the trade's `ts_exchange`. The old code attributed cancels *immediately*, netting out the trade only if the `last_trade` happened to be processed **before** its `price_change`. In the reverse order the full depth-decrease was treated as a cancel — double-counting the trade as a cancel and advancing the queue too far.

**The fix.** Within a `ts_exchange` the model only **accumulates**, per `(token, side, price)`: the net depth-decrease (`depth_start − depth_end`) and the total trade qty. Nothing is attributed to `queue_ahead` until the timestamp completes — at the next event whose `ts_exchange` is larger, or on `flush_pending_cancels()` at stream end (the engine calls it post-loop). The cancel volume is then `max(0, net_decrease − trade_qty)`, attributed once via the FIX-2 floor. Trade-through **fills** still happen the instant the trade is processed, against the pre-cancel `queue_ahead`; because the deferred cancel never touches `queue_ahead` before that fill, the fill and the final `queue_ahead` are **identical** no matter which order the trade and its `price_change` arrive in.

**Tested:** the same trade + same-`ts` depletion `price_change`, fed in both orders, gives identical fills and identical final `queue_ahead` for all three models — in both an overflow-fill case and a residual-cancel case where the models genuinely differ (8 / 14 / 20).

**Scope (stated honestly):** the netting is *net-within-timestamp*. A single trade + its single depletion `price_change` (the real-world case) is exactly order-invariant. Multiple distinct absolute-size updates to the **same** level within one millisecond are best-effort, matching whatever the book builder reconstructs — the capture-semantics note already treats same-ms bursts as inherently ambiguous.

---

## How often does the FIX-3 coincidence actually happen? (L2 measurement)

`scripts/mm_queue_coincidence_measure.py` runs DuckDB over the captured L2 in `./l2_data/` (politics_negrisk + esports, 2026-06-23/24) and, for each trade, maps the aggressor side to the resting side it consumes (SELL→BUY bids, BUY→SELL asks — the queue model's own mapping, validated below) and asks whether a `price_change` shares its `(timestamp_ms, asset_id, price, resting_side)`.

**Column meanings.** *trades* = total `last_trade` prints. *exact-coincident* = trades sharing a `timestamp_ms` with a `price_change` at the same token, price, and mapped resting side — the precise event FIX 3 nets. *any-coincident* = trades whose millisecond carries *any* `price_change` at that token (a "busy-millisecond" upper bound).

| universe | trades | exact-coincident | any-coincident |
|---|---:|---:|---:|
| esports (fast) | 106,315 | 2,219 (**2.09%**) | 7,673 (7.22%) |
| politics_negrisk (slow) | 68,945 | 107 (**0.16%**) | 1,153 (1.67%) |
| **all** | 175,260 | 2,326 (**1.33%**) | 8,826 (5.04%) |

**Side-mapping validation.** For esports trades that have a same-`ts`/same-price `price_change`, **2,219** fall on the mapped resting side vs only **185** on the opposite side (~12:1) — confirming both the measurement and the queue model's `_resting_side` logic.

**Read.** The exact coincidence is **real but modest overall (~1.33%)** and, as expected, **an order of magnitude higher in fast markets** (esports 2.09% vs politics 0.16%). So FIX 3 corrects a genuine source of order-dependent error, concentrated exactly where the methodology explainer ([[mm_backtesting_methodology_explainer]] §2) says queue/latency realism bites — fast, CEX-led or in-play markets — and largely immaterial in the slow politics markets the live loop currently targets. It is cheap, correct, and removes a class of "depends on capture order" non-determinism regardless of frequency.

---

## Adversarial review outcome

A six-lens adversarial review (bracket invariant, order invariance, `forget()` wiring, floor math, determinism/frozen-interface/lookahead, measurement correctness) returned **SHIP**: every audited property held under heavy fuzzing (4000+ floor streams, 15k single- + 9k two-trade order-invariance configs, 0 violations; results byte-identical across `PYTHONHASHSEED` 0/12345; `interfaces.py`/`book.py`/`fills.py` byte-unchanged). The two order-dependent behaviors it surfaced are the explicitly carved-out cases — a full `book` snapshot interleaved within one millisecond (an absolute re-sync that supersedes incremental state by design) and a level emptied to zero (handled above the queue model) — not defects.

It also flagged one **pre-existing** robustness nit (independent of the three fixes): `_prob_ahead_power` formed `front**n` directly, which could `OverflowError` or silently overflow to `inf` (returning `0.0`) at an operator-chosen extreme power `f` (e.g. 100) over a deep book (>~1200 contracts). It cannot break the bracket (the bad `0.0` stays in `[0,1]`, so the floor/clamp keeps Prob in `[floor, Δ]`), but it was hardened anyway: the probability is now computed **scale-free** as `1/(1+(b/a)^n)` (the larger leg never raised directly), `ProbQueue` rejects `f ≤ 0`, and a regression test covers `f ∈ {50,100}` over books >1200. (68 tests green after the hardening.)

## Decision and next step

- **Done / shippable:** all three fixes are in `mm_engine` behind the frozen interface; 67 tests green; no regression. The models remain drop-in `QueueModel`s and still produce the Optimistic/Prob/RiskAverse fill band.
- **Carry into Join 1 (reconciliation):** the deferred netting means a per-event `get_queue_ahead` telemetry snapshot reflects cancels through the *previous* timestamp (current-ts cancels finalize at the boundary) — a one-ts lag to keep in mind when diffing replay vs live-shadow queue telemetry; fills are unaffected.
- **Carry into Join 2 (calibration):** none of these fixes resolve the live-only unknowns (true fill rate, adverse selection). They make the *offline bound* honest and order-independent so that, once `ProbQueue.calibrate(f)` is fit on real fills, the bound collapses onto a trustworthy rate.
- **Do NOT** treat any maker result as deployable on the Optimistic model alone; run the band.
