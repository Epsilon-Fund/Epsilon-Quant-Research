---
title: "MM Backtest Infra Rebuild — The Plan (Justin builds the machine, Alvaro builds the models)"
created: 2026-06-15
updated: 2026-06-15
status: prep
owner: justin
project: polymarket
para: resource
hubs:
  - strat_market_making
  - COWORK
tags:
  - meeting
  - market-making
  - backtesting
  - infra
  - planning
---

# MM Backtest Infra Rebuild — The Plan

> The single planning doc (supersedes the earlier "work-division options" note). Why-it-works detail lives in [[mm_backtesting_methodology_explainer]].

## North star

**A reliable backtesting machine + a live-validation infra, built as ONE strategy-agnostic engine.** "Reliable" isn't something the backtester proves on its own — for passive MM the binding unknowns (queue position, fill rate) are live-only, so the backtest is trustworthy only once its numbers are **reconciled against live fills on the same dates**. We build the machine now; a real strategy plugs in later. If it doubles as the foundation for the eventual bot, good — that's the intent.

## The key idea: the bot is strategy-agnostic

The "strategy" is one small pluggable function — `quote(book_state, inventory, params) -> orders`. Everything else is strategy-independent. So we build the whole machine with a **placeholder symmetric quoter** (fixed-width bid/ask around mid) — which is also the standard A/B baseline. That placeholder is enough to prove the machine works (sane replay fills, same code runs live in shadow, backtest fill-rate matches live). A real strat slots into the same function when it's ready. **We do not need a concrete strategy to build the machine.**

## The two halves

**Justin — the machine (bot-building):**
- replay + live **feed adapters** (both read the same VPS-captured L2; emit identical event objects)
- **book builder** (reconstruct top-N from book + price_change; gap/staleness ≤5s)
- **order manager** (place/cancel/replace, idempotent, throttled)
- the **fill-simulator slot** (calls a queue model + latency model — see plug-points) and **live/shadow execution**
- **telemetry** plumbing (fill share, markout, own round-trip) + the **reconciliation harness** (backtest vs live)
- the **placeholder symmetric quoter**

**Alvaro — the models (research):**
- the **queue model** (estimate fills from aggregate L2 — optimistic / pessimistic / ProbQueue, with a `calibrate(live_fills)` hook)
- the **latency model** (constant round-trip; refine from measured own round-trip)
- the **reconstruction audit** (how cleanly our L2 rebuilds — clean vs ambiguous %)
- the **validation / gates** (A/B vs the symmetric quoter; Deflated Sharpe / CPCV) and the **breakeven-fill-rate** read
- eventually, the **real quoting strategy**

Rationale: Justin wants the executable machine and the live infra; Alvaro takes the modelling/stats. The machine is strategy- and model-agnostic; Alvaro's models drop into it.

## Where you meet — the ~4 plug-points (agree these first, ~an afternoon)

This is all "the interface" means:
1. **Event format** — the shape of a captured event (book / price_change / last_trade / our-order / fill). It's basically whatever the VPS capture already writes; just write it down so replay and live are identical.
2. **`queue_model(book_state, our_order) -> fill?`** — Alvaro's model, called by Justin's fill simulator.
3. **`latency_model() -> round_trip`** — same.
4. **`quote(book_state, inventory, params) -> orders`** — the strategy slot; Justin stubs it (symmetric quoter), Alvaro fills it later.

Plus one agreement: **which reconciliation numbers must match and how close** (fill rate, position path, equity; target <5% fill-rate gap).

Once these are set, the two halves are **highly separable**: Justin builds the full machine against stub models; Alvaro builds/tests his models on the VPS data in isolation; integration is swapping the stubs for the real functions.

## The VPS

A shared cloud box collecting live L2, both have access. It's just the data source — replay reads its files, the live arm connects to the same stream. Nothing more to engineer here; it already fixes the old gap problem.

## Timeline — parallel tracks, with the only three joins marked

```
JOIN 0  (both, ~an afternoon):  agree the 4 plug-points + reconciliation metrics.

PARALLEL  (~1–2 wks):
  Justin: build the machine end-to-end against STUB models; get the
          placeholder quoter running in replay AND live-shadow.
  Alvaro: build the queue + latency models and the reconstruction
          audit on the ~1 week of L2; stand up the gates.

JOIN 1  (reconciliation): run the SAME code in backtest and live-shadow
          over the same dates; swap stubs → real models; tune queue+latency
          until backtest fills/positions/equity match live (<5% gap).
          ← this is the moment the backtest becomes "reliable."

JOIN 2  (calibration): 1-contract real quoting on one market → real fills
          calibrate the queue model f + latency constant → re-run calibrated.

then PARALLEL: accumulate fills, re-validate, report breakeven vs measured.
```

Only **Join 0, Join 1, Join 2** can't be parallelized. Everything else runs independently.

## What "robust" means (acceptance criteria for the machine)

- Reconstructed L1 matches native `best_bid_ask` on a high, reported fraction of intervals; clean/ambiguous % reported.
- Gaps handled — no fills simulated across a gap; ≤5s staleness enforced.
- **Same code path verified** — identical decisions in backtest vs live-shadow on identical event streams.
- **Backtest fill-rate matches live** within tolerance (the reconciliation passes).
- Queue reported as **optimistic/pessimistic bounds** until calibrated; the calibrate hook exercised.
- Overfitting gates wired; deterministic/seeded replay.

Until the reconciliation passes, a backtest number is a **breakeven fill rate**, never a profitability claim. (The ~1 week of L2 is enough to **build and test** the machine and characterize book shape — *not* to call an edge.)

## First steps this week

1. **Both:** agree the 4 plug-points + reconciliation metrics (Join 0).
2. **Justin:** scaffold the engine + placeholder quoter; get one market replaying through it and the same code connecting live-shadow to the VPS stream. That smoke *is* the proof the same-code-path works.
3. **Alvaro:** run the reconstruction audit on the week of L2 (clean/ambiguous %) and draft the queue + latency models against the stub interface.

## Splits we considered (for the record)

Before settling on machine-vs-models, we weighed: by market regime (politics vs crypto/sports — rejected: lopsided, fast markets mostly already closed) and engine-owner vs strategy-owner (rejected: too sequential). Machine-vs-models keeps both people productive in parallel and matches who-wants-what.

## Cross-links

[[mm_backtesting_methodology_explainer]] (the why: queue/latency, same-code-path, realism pitfalls) · [[mm_concepts_and_strategy_buildup]] (the strat, for when the real `quote()` is written) · [[mm_clob_capture_semantics]] (what our L2 contains) · [[mm_politics_negrisk_live_loop_design]] (the gates the calibrated backtest eventually feeds). Consolidates the old replay/sim scripts: `dali_clob_replay_features.py`, `od_v4_queue_replay.py`, `dali_block_k2_quoting_sim.py`, `dali_paper_backtest.py`.
