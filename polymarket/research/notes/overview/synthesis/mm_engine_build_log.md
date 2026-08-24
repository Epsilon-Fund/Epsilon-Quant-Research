---
title: "MM Engine — Build Log & Component Map (Phase 0 → Join 1, the strategy-agnostic backtest↔live machine)"
created: 2026-07-01
status: active
owner: justin
project: polymarket
para: resource
hubs:
  - strat_market_making
  - POLYMARKET_BRAIN
tags:
  - market-making
  - backtesting
  - engine
  - build-log
  - recap
---

# MM Engine — Build Log & Component Map

> Hubs: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · Plan + status: [[2026-06-23_mm_engine_phase01_buildplan]] · How/why it works: [[mm_backtesting_methodology_explainer]] §6 · Research roadmap: [[mm_backtest_research_roadmap]]

## Plain-English Summary

- **What this is.** A one-stop, navigable **recap** of the `polymarket/research/mm_engine/` build — the strategy-agnostic, event-driven engine that runs the *same quoting code* in replay and live-shadow. Three parts: a chronological **build log**, a **component map** (file → what it is, with links to the deep docs), and a **where-we-are** snapshot. It links to the detailed notes; it does **not** re-explain them.
- **Why it exists.** The work spans seven commits and ~ten notes across two lanes (machine = Justin, models = Alvaro). This page is the index so a cold reader can see *what was built, in what order, and why* without reassembling it from git.
- **Status in one line.** **Phase 0 + Phase 1 are built and Join 1 (reconciliation) is LOCKED** (2026-06-30, independently re-verified): same-code-path reconciles at **0% gap**, the book reconstructs ~99% faithfully, the queue-model fill bracket holds on real VPS Parquet, runs are deterministic. The engine is trustworthy **as a consistency/determinism machine — not yet as a real-fill predictor** (that is Join 2). Now in parallel: **Task 4** (validation gates, Alvaro) ‖ **Join 2** (live 1-contract calibration, Justin).
- **The one caveat that governs every number.** Until Join-2 live calibration, every backtest figure is a **bracketed range** (optimistic vs pessimistic queue), never a point estimate.

---

## Part 1 — Build log / checklist (chronological)

Each row is one task. Commit hashes are the engine-code commits under `polymarket/research/mm_engine/` (`git log --oneline -- polymarket/research/mm_engine/`). "✓" = done; "▶" = in progress / not yet built.

| # | done | what was built — and why (one line) | detailed note | commit |
|---|---|---|---|---|
| 1 | ✓ | **Phase 0 scaffold + same-code-path proof** — `mm_engine/` package with stub `SymmetricQuoter`/`OptimisticQueue`/`ConstantLatency` and the replay + live-shadow feed adapters, so one quoter can run through both feeds; the VPS capture was documented to ground the replay path. | [[2026-06-23_mm_engine_phase01_buildplan]] · [[mm_vps_capture_setup]] | `a7194cc` (06-23) |
| 2 | ✓ | **Interface freeze (Join 0)** — froze `interfaces.py` (the `MarketEvent`/`BookState`/`Order`/`FillResult` shapes + `QueueModel`/`LatencyModel`/`Strategy` protocols) so the machine and models lanes could build independently against one contract. | [[2026-06-23_mm_engine_phase01_buildplan]] § shared interface | `2252804` (06-24) |
| 3 | ✓ | **Phase 1 machine** — the `BookTracker`, `OrderManager` (idempotent place/cancel/replace, deterministic IDs), `FillSimulator` (latency × queue gate against the real trade tape), telemetry journal, and reconciliation harness — the lookahead-safe event loop itself. | [[mm_backtesting_methodology_explainer]] §6 | `f09fdb0` (06-24) |
| 4 | ✓ | **Economics: fees / 3-way PnL / settlement** — maker-rebate model off the canonical `FEE_BY_CATEGORY`, PnL reported `gross`/`net_ex_rebate`/`net_with_rebate`, a realized/unrealized/`settle()` split, and a record→replay reconciliation at 0% gap — so the accounting is honest (rebate-only "edges" and mark-to-mid inflation are made visible). | [[mm_engine_fee_pnl_settlement]] | `97e1505` (06-25) |
| 5 | ✓ | **Parquet adapter + converter** — `feeds/replay_parquet.py` replays the durable typed VPS Parquet (byte-identical to the JSONL adapter on shared shards) plus a JSONL→Parquet fixture converter (gaps preserved), making the gap-free VPS Parquet replayable. | [[mm_vps_capture_setup]] §3.1 | `2267391` (06-30) |
| 6 | ✓ | **Models lane: queue + latency + data-contract + calibrate hooks** — the three queue models (`Optimistic`/`RiskAverse`/`Prob`), `Constant`/`Sampled` latency, the fail-closed data-contract gate, and `calibrate()` stubs — all behind the frozen interface; **68 mm_engine tests green**. | [[mm_engine_queue_models]] · [[mm_queue_model_audit_fixes_findings]] · [[mm_latency_measurement_spec]] · [[data_contract_validation_layer_findings]] | `c578140` (06-30) |
| 7 | ✓ | **JOIN 1 — reconciliation, LOCKED** — swapped Alvaro's real models in and proved 0%-gap record→replay per model, the `Opt ≥ Prob ≥ RA` bracket with **0** `queue_ahead>depth` violations, ~99% book accuracy, and determinism on real gap-free VPS Parquet. Caught + fixed the live-Parquet schema bug and **regression-locked** it; **Check A′** validated the live WS JSON path on real frames (0 parse errors / 11k+ frames). Suite **85 green**. | [[mm_join1_reconciliation_findings]] · [[mm_reconstruction_audit_findings]] | `b6dbc1b` (06-30) |
| 8 | ▶ | **Task 4 ‖ Join 2 (now, in parallel)** — **Task 4** (Alvaro): validation-gate/eval layer (A/B vs symmetric quoter, ND-PnL/markout/adverse-selection, DSR/CPCV, breakeven-fill-rate) reading engine logs, no new engine code. **Join 2** (Justin): bridge to `execution/maker` safety+signing + the pre-registered 1-contract live loop that fits `ProbQueue.f`/the latency constant from real fills. | [[mm_politics_negrisk_live_loop_design]] · [[mm_latency_measurement_spec]] | *(none yet)* |

**Commit-coverage caveat (knowledge gap).** The git history scoped to `mm_engine/` records the *engine-code* deliverables only. The Join-1 reconciliation runs and the Check A/A′/C illustrative tables were produced by **scratch scripts that were not committed** (per [[mm_join1_reconciliation_findings]] § Independent verification), so row 7's hash is the live-Parquet *fix + regression test*, not the reconciliation run itself; the standalone reconstruction-audit script lives under `scripts/`, not `mm_engine/`. Their *properties* are pinned by committed tests (`test_record_replay_reconciliation`, `test_mm_queue_models`, `test_mm_engine_parquet_live_schema`).

---

## Part 2 — Component map

Every `mm_engine` module + the model implementations + the strategy layer, each 1–2 lines. Deep explanations live in the linked docs — not duplicated here.

| component | what it is / what it's made of | deep doc |
|---|---|---|
| `interfaces.py` | The frozen contract both lanes build against: `MarketEvent`, `BookState`, `Order`, `FillResult` + `QueueModel`/`LatencyModel`/`Strategy` protocols. Unchanged since Join 0. | [[2026-06-23_mm_engine_phase01_buildplan]] |
| `events.py` | Event + `GapMarker` types and the `envelope → events` normalization shared by both feeds (canonicalizes payloads to the Parquet-schema fields). | [[mm_backtesting_methodology_explainer]] §6.1 |
| `feeds/replay.py` | JSONL replay adapter — reads captured `*.jsonl[.gz]` shards, emits the `MarketEvent` stream in `ts_exchange` order, honors `capture_gaps.jsonl`. | [[mm_vps_capture_setup]] §3 |
| `feeds/replay_parquet.py` | Parquet replay adapter — same `MarketEvent` stream from the typed VPS Parquet; byte-identical to JSONL on shared shards; schema-drift-tolerant column projection. | [[mm_vps_capture_setup]] §3.1 |
| `feeds/live_shadow.py` | Read-only live-WS adapter (no orders); `record_to` lets a live session record its own frames for record→replay. | [[mm_join1_reconciliation_findings]] Check A′ |
| `book.py` | `BookTracker` — top-N reconstruction from `book`+`price_change`, staleness/gap flagging, L1 cross-check vs native `best_bid_ask` (telemetry-only). | [[mm_reconstruction_audit_findings]] |
| `orders.py` | `OrderManager` — place/cancel/replace, idempotent, cancel-replace throttle, **deterministic client IDs** (reproducible replay). | [[mm_backtesting_methodology_explainer]] §6.1 |
| `fills.py` | `FillSimulator` — latency gate (quote must have landed) × queue gate (`QueueModel.fill`) against the **real** trade tape; no independent price simulation. | [[mm_backtesting_methodology_explainer]] §6.1 |
| `fees.py` | `FeeModel`/`FeeSchedule` — per-market `fee` field → category `FEE_BY_CATEGORY` → `fee_free`; passive = 0 maker fee + earned rebate; taker-fee path wired but unused. | [[mm_engine_fee_pnl_settlement]] |
| `engine.py` | `run_engine` loop + `EngineResult` — 3-way PnL (`gross`/`net_ex_rebate`/`net_with_rebate`), realized/unrealized split, `settle(resolution_map)`. | [[mm_engine_fee_pnl_settlement]] |
| `reconcile.py` | The Join-1 artifact — `reconcile_against_recording` replays a recorded session and diffs fill-rate/PnL/position/equity-path vs tolerance. | [[mm_join1_reconciliation_findings]] |
| `telemetry.py` | Raw append-only fill / order / per-quote queue-snapshot logs; metrics are computed downstream from these. | [[mm_backtesting_methodology_explainer]] §6.1 |
| `strategies.py` | `SymmetricQuoter` — the placeholder A/B-baseline quoter (the real `quote()` comes later); exposes the `Strategy` swap point. | [[mm_concepts_and_strategy_buildup]] |
| `queue_models.py` → **OptimisticQueue** | Credits the whole cancel as ahead of us → **upper bound** on fills. | [[mm_engine_queue_models]] |
| `queue_models.py` → **RiskAverseQueue** | Ignores cancels, advances only on real trades (+ the logical floor) → **lower bound** on fills. | [[mm_engine_queue_models]] |
| `queue_models.py` → **ProbQueue** | Splits each cancel by hftbacktest's power-law `front^f/(front^f+back^f)` → **middle** estimate; `f` (default 0.5) is the `calibrate()` knob. | [[mm_engine_queue_models]] · [[mm_queue_model_audit_fixes_findings]] |
| `latency_models.py` → **ConstantLatency** | Fixed submit→ack round-trip (placeholder `200ms`); sufficient for slow politics. | [[mm_latency_measurement_spec]] |
| `latency_models.py` → **SampledLatency** | Deterministic `Normal(mean,std)` draw per order from measured samples (`from_samples`/`calibrate`); the dispersion bites in fast crypto/in-play. | [[mm_latency_measurement_spec]] |
| **Strategy layers** (design, not engine code) | The eventual real quoter is classic A-S inventory MM + a stack: slow-market selection, two-sided book, carry-to-resolution, spike-zone avoidance, non-incumbent cell selection, NegRisk redemption floor. | [[mm_concepts_and_strategy_buildup]] Part 2 |

---

## Part 3 — Where we are

**Join 1 — LOCKED (2026-06-30).** All four reconciliation checks pass on real gap-free VPS Parquet (politics-NegRisk + esports) and a recorded session: (A) record→replay **0% gap per model**, (B) book reconstruction ~99% faithful once the intra-millisecond `best_bid_ask` ordering artifact is accounted for, (C) the `Optimistic ≥ Prob ≥ RiskAverse` fill bracket with **0** `queue_ahead > depth` violations, (D) byte-identical determinism + JSONL↔Parquet source-invariance. Independently re-verified (all cited numbers reconcile, two strengthened). The "68% mismatch" headline was **proven a metric-ordering artifact, not a book/fill error** (fills byte-identical regardless of `best_bid_ask` ordering). The live-Parquet schema bug was caught → fixed → regression-locked; **Check A′** validated the live WS JSON path on real frames (0 parse errors over 11k+ frames, 0%-gap record→replay). Verdict: the engine is trustworthy **as a consistency/determinism machine — not yet real-fill-realistic.**

**Task 4 ‖ Join 2 — in parallel (now).**

- **Task 4 (Alvaro lane)** — the validation-gate / eval layer: A/B vs the symmetric quoter, net-of-cost PnL / markout / adverse-selection, Deflated-Sharpe / CPCV, breakeven-fill-rate per market. Reads engine logs; **no new engine code**.
- **Join 2 (Justin lane)** — the execution-safety bridge to `execution/maker` (signing + safety) plus the pre-registered **1-contract live-calibration loop**: real fills fit `ProbQueue.f` and the latency constant, collapsing the optimistic/pessimistic bracket toward the live-measured rate. This is when the backtest earns "reliable."

**The one standing caveat.** Until Join-2 calibration lands, **every backtest number is bracketed** (optimistic vs pessimistic queue) and conditional — never a point estimate. Join 1 proves the engine is *consistent and deterministic*; only Join 2 makes its *fill rate* real.

> **Numbering note (knowledge gap).** This recap uses the build-plan's join numbering — **Join 0** = freeze interface, **Join 1** = reconciliation, **Join 2** = live calibration ([[2026-06-23_mm_engine_phase01_buildplan]]). The research roadmap ([[mm_backtest_research_roadmap]]) numbers differently: its **Join 2** = engine validation (this note's Join 1) and its **Join 3** = calibration (this note's Join 2), and it attributes the models lane to "Carlos" where the build-plan and the model notes say "Alvaro." Same milestones, different labels — reconcile by milestone, not number.

---

## Cross-links

Plan + live status block: [[2026-06-23_mm_engine_phase01_buildplan]]. Why the engine is shaped this way: [[mm_backtesting_methodology_explainer]] (§6 walks the built engine). Deep component docs: [[mm_engine_fee_pnl_settlement]] · [[mm_engine_queue_models]] · [[mm_queue_model_audit_fixes_findings]] · [[mm_latency_measurement_spec]] · [[data_contract_validation_layer_findings]]. Reconciliation + data quality: [[mm_join1_reconciliation_findings]] · [[mm_reconstruction_audit_findings]] · [[mm_clob_capture_semantics]] · [[mm_vps_capture_setup]]. Strategy the real quoter will implement: [[mm_concepts_and_strategy_buildup]]. The live loop Join 2 feeds: [[mm_politics_negrisk_live_loop_design]]. Research checklist: [[mm_backtest_research_roadmap]]. Hubs: [[strat_market_making]] · [[POLYMARKET_BRAIN]].
