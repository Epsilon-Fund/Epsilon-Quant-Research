---
title: "MM Backtest Research Roadmap"
tags: [market-making, backtesting, roadmap, queue-model, calibration]
created: 2026-06-24
status: active — Phase 0 not yet started
identifier: "mm_backtest_research_roadmap"
relationship: >
  Operational skeleton for the MM backtesting research process. Companion to
  [[mm_backtesting_methodology_explainer]] (the "why") — this doc is the "what, in what order."
  Feeds into [[strat_market_making]] MM Path B. Depends on L2 data from [[polymarket_l2_ingestion]].
  Justin's infrastructure plan (MM_Backtest_Infra_Plan.pdf, June 2026) is the implementation
  counterpart — this roadmap is the research/modelling counterpart from Carlos's side.
---

# MM Backtest Research Roadmap

> Hub: [[strat_market_making]] · [[COWORK]]

## What this document is

The step-by-step skeleton for building a calibrated MM backtester on Polymarket, then iteratively
researching strategies through it. Each phase has a gate — work does not advance until the gate
passes. The methodology and academic grounding live in [[mm_backtesting_methodology_explainer]];
this doc is the operational checklist.

## The big picture — the iterative research loop

The research process is NOT "build backtester → run strategy → done." It is a calibration loop:

```
Build backtester
     ↓
Live-test simplest strategy (placeholder quoter, $1 orders)
     ↓
Calibrate fill model to those live results
     ↓
Validate calibration out-of-sample
     ↓
Formulate hypothesis → backtest strategy v1
     ↓
Live-test strategy v1 small → recalibrate strategy-specific parameters
     ↓
Repeat with v2, v3... scaling as confidence grows
     ↓
Kill gate: if no strategy viable after N iterations → shelve MM
```

### What transfers between strategies and what doesn't

The calibration has two layers:

**Market dynamics layer (transfers):** queue drain rates, trade arrival intensity, cancellation
patterns, adverse selection profile, latency distribution. These are properties of the venue (Polymarket)
and transfer across strategies. Calibrated once with the placeholder, roughly valid for any strategy.

**Strategy behavior layer (does NOT transfer):** where you quote, how often you cancel/replace,
your order size relative to queue. A symmetric quoter that posts and sits has different fill dynamics
than an algo that adjusts every 500ms. Each new strategy requires its own live recalibration cycle,
but it's faster than the first because the market dynamics layer is already known.

Consequence: the first backtest of a new strategy is directional (good enough for go/no-go), not
precise. Live testing tightens the strategy-specific parameters.

---

## Phase 0 — Interface Agreement

**Who:** Carlos + Justin, together.
**Duration:** Half a day.
**Objective:** Lock down the shared contract so both sides can build independently.
**Gate:** Four plug-point stub signatures exist; reconciliation thresholds defined with concrete numbers.

### Decision 1 — Event format

The shape of one row of data. Every market event (book update, trade, cancel) is one event.
Must match what the L2 capture pipeline already writes (see [[polymarket_l2_ingestion]]).

What to agree:

- [ ] Event types: at minimum `book_update`, `trade`, `our_order_placed`, `our_order_cancelled`, `our_fill`
- [ ] Fields per type: timestamp (precision — ms or μs?), price, size, side, event_type
- [ ] For trades: aggressor side (buy or sell initiated)
- [ ] For book updates: `prev_qty` AND `new_qty` at the level (queue model needs the delta)
- [ ] Snapshots vs deltas: does each event carry full book or just the change?
- [ ] Depth levels: how many levels per side? (5-10 recommended by queue-reactive literature)
- [ ] How the existing per-shard Parquet output maps to this format

Connection to data pipeline: the L2 capture already writes `book`, `trades`, `price_change`,
`best_bid_ask` event types in per-shard Parquet. The event format here should be a thin
standardization layer on top of that, not a redesign.

### Decision 2 — Queue model interface (CRITICAL)

Justin's doc says `queue_model(book_state, our_order) → fill?`. Research shows this must be
event-driven with separate handlers. The model needs to know WHAT happened, not just the
current snapshot.

What to agree:

- [ ] Event-driven interface with separate methods:
  - `on_new_order(our_order, book_state)` → initialize queue position estimate
  - `on_trade(our_order, trade_event, book_state)` → advance position, check fill
  - `on_depth_change(our_order, prev_qty, new_qty, book_state)` → probabilistic position update
  - `is_filled(our_order, book_state)` → (filled?, fill_qty)
- [ ] What `book_state` contains (see Decision 4)
- [ ] Whether the model returns just fill yes/no or also fill quality (adverse/benign classification)
- [ ] Who maintains queue position state — the model (stateful) or the engine (passes position in)?

Research basis: hftbacktest uses exactly this pattern. See [[mm_backtesting_methodology_explainer]]
§ Queue model research findings.

### Decision 3 — Latency model scope

`latency_model() → round_trip` — but what counts as latency?

What to agree:

- [ ] Scope: network RTT only, or full pipeline (signal → compute → send → exchange ack)?
- [ ] Fixed constant vs distributional (mean + jitter)? Literature says distributional matters.
- [ ] Separate latency for different actions (place vs cancel vs amend)?
- [ ] How to measure: instrument the live path during Phase 2 to collect real latency samples

For slow politics markets a fixed constant is probably sufficient initially. But define the
interface as distributional now (return a sample from a distribution) so it doesn't need
refactoring for faster markets later.

### Decision 4 — What "book_state" contains

Every plug-point receives book_state. Define exactly once.

What to agree:

- [ ] Minimum: best bid/ask price and size
- [ ] Depth at N levels per side (agree on N)
- [ ] Derived quantities: imbalance, spread, microprice — computed by engine or by model?
- [ ] Recent trade list / rolling trade arrival rate (window size?)
- [ ] Whether book_state is a snapshot (copied) or a live reference (mutable)

Recommendation: engine computes derived quantities (imbalance, microprice, trade rate) once
and passes them in. Avoids duplicate computation and ensures consistency.

### Decision 5 — Reconciliation thresholds

The pass/fail definition for "backtest matches live." Without concrete numbers, Join 2 and
Join 3 become subjective debates.

What to agree:

- [ ] **Book reconstruction (Join 2):** replayed book matches live book. Expect exact match on mid-price and BBO sizes.
- [ ] **Decision match (Join 2):** placeholder quoter makes identical place/cancel decisions in backtest and live-shadow. Expect 100% match (same code path = same decisions).
- [ ] **Fill rate (Join 3):** backtest fill count vs live fill count on same dates. Tolerance: ±__% (suggest starting with ±20%, tighten as data grows).
- [ ] **Fill timing (Join 3):** distribution of time-to-fill. Kolmogorov-Smirnov test, p > 0.05?
- [ ] **Equity path (Join 3):** end-of-period PnL within ±__% of each other.
- [ ] **Out-of-sample requirement:** calibrate on first N days, validate on held-out days. Thresholds must hold on held-out data.

### Decision 6 — Adverse selection tracking

Not in Justin's original doc. Research says 66-89% of fills are adverse. Must be a first-class
output.

What to agree:

- [ ] **Post-fill markout windows:** price at fill+1s, fill+5s, fill+30s, fill+60s
- [ ] **Who computes:** engine logs fill timestamp + price; markout computed in telemetry/analysis
- [ ] **Classification:** fill is "adverse" if mid-price moved against our position by >X bps within Ns
- [ ] **This feeds the kill criterion:** if adverse fill % is >Y% across all strategies tested, venue-level MM may not be viable

---

## Phase 1 — Parallel Build

**Who:** Justin (engine) + Carlos (models), independently.
**Duration:** 1-2 weeks.
**Prerequisite:** Phase 0 complete.
**Gate (Join 2):** Placeholder quoter backtest = live-shadow (same decisions on same data).

### Justin builds

- [ ] Feed adapters (replay historical L2 + connect to live VPS stream — same code)
- [ ] Order book builder (reconstruct book state from L2 events)
- [ ] Order manager (place / cancel / replace, calls queue_model interface)
- [ ] Fill-simulator slot (calls queue model on each event, maintains order lifecycle)
- [ ] Telemetry + reconciliation harness (logs for comparing backtest vs live)
- [ ] Placeholder quoter (symmetric, fixed spread around mid, no logic)

### Carlos builds

- [ ] Queue model v0 — three variants behind the agreed interface:
  - Optimistic (front of queue — overstates fills)
  - Pessimistic (back of queue — understates fills)
  - Probabilistic (power-law or log model, à la hftbacktest ProbQueueModel)
- [ ] Latency model v0 — initial fixed constant, distributional interface
- [ ] Validation + overfitting gates:
  - Out-of-sample split enforced in all backtest runs
  - Confidence intervals on all reported metrics (no point estimates without CIs)
  - Bracketing: every strategy tested under both optimistic and pessimistic queue models
- [ ] Draft real strategy on paper (quoting logic, not code yet)

### Join 2 — Engine validation

Run placeholder quoter through:
1. Historical replay of captured L2 data
2. Live-shadow on the same market (observing, no real orders)
3. Compare: do they produce identical decisions?

Expected result: exact match on decisions (same code path). Book reconstruction should
match within floating-point tolerance. If not → debug the engine, not the models.

---

## Phase 2 — Calibrate with Real Fills

**Who:** Justin (live execution) + Carlos (model fitting).
**Duration:** 2-4 weeks (gated by fill accumulation rate).
**Prerequisite:** Join 2 passed.
**Gate (Join 3, MAKE-OR-BREAK):** Queue model can be calibrated to match live fills.

### Execution

- [ ] Justin: live path places real $1 quotes on ONE slow politics market
- [ ] Telemetry logs: queue rank estimate, fill events, post-fill price at markout windows, round-trip latency per order

### Calibration

- [ ] Carlos: fit queue model parameters to observed fills
  - Tune the probabilistic model's power parameter until backtest fill rate ≈ live fill rate
  - Fit latency distribution from observed round-trip times
  - Measure adverse selection profile (markout distribution)
- [ ] Validate out-of-sample: calibrate on first N days, test on held-out days
- [ ] Run backtest on same dates as live → compare fill rate, equity, adverse selection %

### Join 3 — Calibration gate

Compare backtest predictions vs live actuals on held-out dates:
- Fill rate within agreed tolerance
- Fill timing distribution statistically similar
- Equity path within agreed tolerance
- Adverse selection % within ±10 percentage points

**If Join 3 passes:** the backtester is now reliable. Proceed to Phase 3.
**If Join 3 fails:** diagnose. Is the L2 data too coarse? Is the queue model structurally wrong? Is Polymarket's matching behavior non-standard? Decision point: fix and retry, or shelve MM.

### Polymarket-specific items to verify during Phase 2

- [ ] Confirm price-time priority (FIFO) matching — expected based on docs, verify empirically
- [ ] Measure off-chain matching latency characteristics (hybrid architecture: match off-chain, settle on-chain)
- [ ] Check whether L2 capture granularity is sufficient or if L3/MBO data is needed

---

## Phase 3 — Real Strategy Iteration

**Who:** Carlos (strategy design + backtest) + Justin (live execution).
**Duration:** Ongoing.
**Prerequisite:** Join 3 passed (calibrated backtester).

### The iteration loop

Each strategy version follows this cycle:

```
1. Formulate hypothesis
   "Strategy vN addresses problem X by doing Y. Expected improvement: Z."

2. Backtest in calibrated simulator
   Run under BOTH optimistic and pessimistic queue models.
   Results are directional (market dynamics calibrated, strategy behavior layer is new).

3. Gate: backtest promising?
   Profitable under pessimistic model → high confidence, go live.
   Profitable under optimistic only → probably not real, revise hypothesis.
   Unprofitable under both → dead, next hypothesis.

4. Live test at small size
   Same market, small orders. Compare fills vs backtest predictions.

5. Recalibrate strategy-specific parameters
   The strategy behavior layer (cancel/replace frequency, quote placement distribution)
   now has live data. Tighten the model.

6. Compare recalibrated backtest vs live
   Does the tighter model still show edge? If yes → scale up cautiously.
   If the edge disappeared after recalibration → the "edge" was a calibration artifact.

7. Repeat or scale
```

### Strategy pipeline (tentative, to be refined)

- [ ] **v0 — Placeholder quoter:** symmetric, fixed spread. NOT expected to be profitable. Purpose: calibration data only.
- [ ] **v1 — Simple asymmetric quoter:** skew quotes based on inventory. First real strategy.
- [ ] **v2 — Imbalance-aware quoter:** widen/tighten spread based on book imbalance signal (the 95% R² predictor from literature).
- [ ] **v3+ — To be defined** based on what v1/v2 reveal about the market.

### Kill criterion

Define upfront to prevent sunk-cost drift:

- [ ] If no strategy achieves breakeven fill rate under the pessimistic queue model after **3 full iteration cycles** (≈ 2-3 months of Phase 3), reassess venue-level viability of passive MM on Polymarket.
- [ ] If adverse selection % exceeds **85%** across all strategies and markets tested, the spread is structurally insufficient to compensate — shelve MM and redirect resources.
- [ ] Review and potentially update these thresholds after Phase 2 data gives a baseline adverse selection measurement.

---

## How the pieces connect

```
L2 Data Pipeline (VPS)          This Roadmap              Justin's Engine
─────────────────────          ───────────────           ─────────────────
 Captures raw events    ──→    Event format (D1)    ──→  Feed adapters
 Parquet per shard             Queue model (D2)     ──→  Fill simulator slot
                               Latency model (D3)   ──→  Order manager
                               Book state (D4)      ──→  Order book builder
                               Thresholds (D5)      ──→  Reconciliation harness
                               Adverse sel. (D6)    ──→  Telemetry

                                      ↓
                               Placeholder quoter
                                      ↓
                               Calibration loop
                                      ↓
                               Real strategy slot ←── Carlos's quoting logic
```

## Data dependencies

| What | Source | Status | Needed by |
|------|--------|--------|-----------|
| Raw L2 events | VPS capture pipeline | OPERATIONAL since 2026-06-19 | Phase 1 (replay engine) |
| Parquet conversion | Compression pipeline | OPERATIONAL (OOM fix 2026-06-22) | Phase 1 (data loading) |
| Cloud backup (R2) | rclone sync | PENDING Justin's config | Phase 1 (redundancy) |
| Live fill logs | Justin's engine, live path | NOT YET BUILT | Phase 2 (calibration) |
| Latency measurements | Instrumented live path | NOT YET BUILT | Phase 2 (latency model fit) |
| Markout data | Telemetry, post-fill price tracking | NOT YET BUILT | Phase 2 (adverse selection) |

## Timeline (rough)

| Phase | Target | Dependency |
|-------|--------|------------|
| Phase 0 | Next session with Justin (TBD) | Needs Justin availability |
| Phase 1 | 1-2 weeks after Phase 0 | Parallel work |
| Join 2 | End of Phase 1 | Both sides deliver |
| Phase 2 | 2-4 weeks after Join 2 | Gated by fill accumulation |
| Join 3 | End of Phase 2 | Enough fills for calibration |
| Phase 3 | Ongoing after Join 3 | Iterative |

## Open questions

- [ ] Minimum number of live fills needed for statistically confident calibration at Join 3?
- [ ] How thin are politics NegRisk markets — what's the expected fills/day for a $1 symmetric quoter?
- [ ] Should we start with a single market or multiple markets in Phase 2?
- [ ] How to handle Polymarket's hybrid off-chain/on-chain architecture in the latency model?

## Cross-links

- Methodology + research: [[mm_backtesting_methodology_explainer]]
- Strategy hub: [[strat_market_making]] (MM Path B section)
- L2 data pipeline: [[polymarket_l2_ingestion]]
- L2 capture semantics: [[mm_clob_capture_semantics]]
- Justin's infra plan: `MM_Backtest_Infra_Plan.pdf` (June 2026)
- Queue model academic references: see [[mm_backtesting_methodology_explainer]] § Queue model research findings
