---
title: "MM Engine — Fee/Rebate, Three-Way PnL, Settlement, and Record→Replay Reconciliation"
created: 2026-06-25
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
  - fees
  - pnl
---

# MM Engine — Fee/Rebate, Three-Way PnL, Settlement, Record→Replay

> Hubs: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · [[COWORK]]
> Engine plan: [[2026-06-23_mm_engine_phase01_buildplan]] · fee source: `scripts/dali_block_a1_analyze.py` `FEE_BY_CATEGORY` · discipline: [[block_k5_stress_findings]] (`net_without_rebate`) · fee gate fact: [[block_k3_leadlag_findings]] (crypto `fd`: rate 0.07 / rebateRate 0.2).

## Summary

- **What:** machine-realism refinements to the strategy-agnostic `mm_engine` (against the **frozen** interface): a fee + maker-rebate model, PnL reported **three ways**, a realized/unrealized/**settled** split with a `settle()` hook, and a **record→replay** reconciliation. No strategy verdict here — this makes the engine's accounting honest so a *future* strategy can be judged.
- **Why:** so we never assume away the two things that decided every prior MM result — the **maker rebate** (is the "edge" rebate-only?) and **settlement** (how much PnL rides on open inventory carried to resolution vs actually realized?).
- **Takeaway:** fills now credit the rebate per the captured/category schedule; `EngineResult` exposes `gross` / `net_with_rebate` / `net_ex_rebate` and `realized` / `unrealized` / `settled`; record→replay reconciles a live-shadow session against a replay of its own recorded frames at **0% gap**. 28 `mm_engine` tests green.

## What was added (all in `mm_engine/`, interface untouched)

1. **Fee + maker-rebate model** (`fees.py`). Reuses the canonical `FEE_BY_CATEGORY` (lazy-imported from `dali_block_a1_analyze` — single source of truth, no duplicated numbers). Formula `fee = fee_rate · qty · p · (1−p)`; **maker rebate = `rebate_rate · fee`**. Resolution order: **per-market `fee` field** (`FeeSchedule.from_fee_field`, from the captured `fd`: `fees_enabled`/`rate`/`rebateRate`) → **category fallback** (`FEE_BY_CATEGORY`) → **`fee_free`** override. We quote passively → pay **0 maker fee**, *earn* the rebate; the taker-fee path is modeled (`taker_fee_ref` in the fill log) for any future crossing leg.
2. **Three-way PnL** in `EngineResult` + telemetry: `gross_pnl`, `net_ex_rebate` (the `net_without_rebate` discipline — a rebate-only edge shows up as `net_ex_rebate ≤ 0`), `net_with_rebate`.
3. **PnL split + settlement.** Per-token average-cost ledger → `realized_pnl` (offsetting round-trips) and `unrealized_pnl` (open inventory marked-to-mid, flagged UNREALIZED). `EngineResult.settle(resolution_map)` carries open inventory to the resolved payoff (`qty · (payoff − cost_basis)`, matching the prior `entry − payoff + rebate` accounting); tokens absent from the map **stay unrealized** rather than being assumed settled.
4. **Record → replay reconciliation.** `live_shadow_feed(..., record_to=...)` writes the session's own envelope frames to a replayable JSONL shard; `reconcile_against_recording(live_result, recording)` replays that exact shard and diffs fill-rate / realized / gross / net / position / equity-path. A real same-code-path / determinism test, not two replays of pre-captured data.

## Worked example (one passive lot — inspect the accounting)

A passive **BUY 100 @ 0.47** fills (Crypto schedule, `fee_rate 0.07`, `rebate 0.20`); mid is 0.48.

| quantity | value | note |
|---|---|---|
| maker fee | **$0.00** | passive — we never pay the taker fee |
| maker rebate | **+$0.349** | `0.20 · 0.07 · 100 · 0.47 · 0.53` (a taker would pay `$1.744`) |
| realized | $0.00 | no offsetting trade yet |
| unrealized (mark-to-mid) | **+$1.00** | `100 · (0.48 − 0.47)` — flagged UNREALIZED |
| `net_ex_rebate` / `net_with_rebate` | $1.00 / **$1.349** | rebate is the whole difference |
| `settle({YES: 1.0})` | **+$53.0** | `100 · (1.0 − 0.47)` |
| `settle({YES: 0.0})` | **−$47.0** | `100 · (0.0 − 0.47)` |
| `settle({})` | unrealized +$1.00 | absent → stays flagged, **not** assumed settled |

**Read:** for a single open lot, realized is $0 and mark-to-mid is only +$1, but settlement swings to **+$53 / −$47** — i.e. essentially *all* the economics ride on open inventory carried to resolution. That is exactly the quantity the split makes visible: we can now *measure* whether settlement is small for a given strategy instead of assuming it (the mark-to-mid inflation trap that killed K-PEG).

## Decision / next step

This is engine realism, not a trade verdict — the running strategy is still the stub `SymmetricQuoter` and the queue/latency models are still stubs (Alvaro's lane). Next: Alvaro's real `QueueModel`/`LatencyModel` drop in via the frozen interface; then **Join 2** bridges to `execution/maker` for the 1-contract live calibration, at which point the rebate, settlement, and fill-rate numbers here become live-anchored rather than assumption-conditional.
