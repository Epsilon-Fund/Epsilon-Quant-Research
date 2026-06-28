---
title: "MM Backtesting Methodology Explainer"
tags: [market-making, backtesting, queue-model, methodology]
created: 2026-06-24
status: active — Phase 0 decisions pending
identifier: "mm_backtesting_methodology_explainer"
relationship: >
  Documents the professional market-making backtesting approach for Polymarket. Feeds into
  [[strat_market_making]] as the methodology layer. Depends on captured L2 data from
  [[polymarket_l2_ingestion]]. Queue model research from deep-research pass (2026-06-20).
---

# MM Backtesting Methodology Explainer

## Summary

Professional market-making backtesting cannot use simple historical replay like taker strategies. The core problem: when you post a limit order, whether it fills depends on your queue position — something not observable in L2 data. This document captures the methodology decisions, queue model research, and the phased roadmap agreed upon for building a reliable MM backtester on Polymarket captured L2 data.

## Why MM backtesting is different from taker backtesting

Taker strategies hit existing orders — if a price existed in historical data, you could have traded at it. Market making is the opposite: you *post* limit orders and wait. Your profit depends on two unobservable things: queue position (are you 1st or 20th in line?) and adverse selection (did price move against you right after the fill?).

A naive backtest that assumes "if the price touched my order, I was filled" is wildly optimistic. Professional firms calibrate queue models against real fills, then simulate.

## Queue model research findings (deep-research pass 2026-06-20)

The Phase 0 plug-point `queue_model(book_state, our_order) → fill?` is an oversimplification. Research findings:

### What queue models actually consume (richer than book_state + our_order)

1. **Triggering event type** — trade at your level, depth change (cancel/add), or new order placement. Each triggers different queue-position logic. In hftbacktest, separate methods: `trade(order, trade_qty, depth)` and `depth(order, prev_qty, new_qty, depth)`.

2. **Market context** — queue imbalance (bid size / ask size), recent volatility, trade arrival rate. One paper found three variables — near-side queue, opposite-side queue, and book imbalance — explain 95% of fill probability variance.

3. **Cancellation dynamics** — when someone cancels ahead of you, your position improves. But L2 data only shows total depth changed, not *who* cancelled. The model must probabilistically estimate "cancel ahead or behind me?" This is the core L2 queue model problem, and every implementation tends to be overly optimistic.

4. **Adverse selection signal** — across CME futures studies, 66-89% of all limit order fills happen because price moved *through* the order (guaranteed fill, guaranteed loss). A model that just says "fill? yes/no" without tracking fill toxicity misses the whole game.

### Main academic models

- **Cont-Stoikov-Talreja (2010):** stochastic model for order book dynamics, arrival/cancellation rates as Poisson processes
- **Moallemi-Yuan (2016):** queue position valuation, what a position at depth D is worth given execution probability
- **Huang-Lehalle-Rosenbaum (2015):** queue-reactive model — event-driven, models how the queue reacts to trades/cancels
- **hftbacktest implementation:** L3 (exact position known) vs L2 (probabilistic). Three L2 models: ProbQueueModel1 (uniform), ProbQueueModel2 (power-law, more realistic), ProbQueueModel3 (ML-based)

### Key sources

- hftbacktest docs: Probability Queue Models tutorial
- Moallemi & Yuan: Queue Position Valuation in a Limit Order Book (2016)
- DeLise: The Negative Drift of a Limit Order Fill (2024)
- Lalor & Swishchuk: Market Simulation under Adverse Selection (2024)
- Fabre & Ragel: Interpretable ML for HF Execution (2024)

## The chicken-and-egg problem and its resolution

**The cycle:** need a strategy to get fills → need fills to calibrate → need calibration to trust strategy tests.

**Resolution — the placeholder quoter:** a dumb symmetric quoter (fixed spread around mid, no edge) generates fills to calibrate the queue model. The fill model is mostly strategy-agnostic — queue dynamics at a given price level are driven by the market, not your strategy. Once calibrated, plug in the real strategy.

**Bracketing before calibration:** run any strategy under optimistic and pessimistic queue models simultaneously. If profitable under pessimistic → probably real. If only under optimistic → probably not. This gives bounds before spending any money.

## Phased roadmap (agreed with Justin)

> **Operational companion:** [[mm_backtest_research_roadmap]] has the full phase-by-phase checklist,
> iteration loop, calibration philosophy (what transfers between strategies vs what doesn't),
> kill criteria, and data dependency table. This section below is the summary; the roadmap is
> the working document.

### Phase 0 — Interface agreement (half-day session)

Six decisions to lock down:

1. **Event format** — exact fields per event row (timestamp, price, size, side, type). Must match L2 capture schema.
2. **Queue model interface** (CRITICAL) — must be event-driven, not monolithic. Separate handlers for trade/depth-change/order-placement events. Return fill probability + toxicity signal.
3. **Latency model interface** — round-trip distribution per action (place, cancel, amend). Strategy-agnostic.
4. **Strategy interface** — `quote(book_state, inventory) → orders`. Placeholder first, real algo later.
5. **Reconciliation thresholds** — concrete numbers for "backtest matches live" (e.g., fill rate within ±15%, PnL per fill within ±20%).
6. **Adverse selection tracking** (addition to Justin's doc) — post-fill price movement must be a first-class output of the engine.

### Phase 1 — Build (parallel work)

Justin builds engine + placeholder quoter end-to-end. Carlos builds queue model, latency model, validation gates, drafts real strategy.

**Gate (Join 2):** placeholder quoter backtest matches live-shadow. Machine is trustworthy.

### Phase 2 — Calibrate (real money, tiny)

$1 quotes on one slow politics market. Collect fill logs. Fit queue model until backtest reproduces live.

**Gate (Join 3, make-or-break):** queue model can be calibrated to match live fills. If not → rethink approach.

### Phase 3 — Real strategy iteration

Plug in real strategy. Backtest → live small → compare → tweak → repeat, scaling as confidence grows.

## Relationship to L2 data pipeline

The L2 capture pipeline running on the VPS (see [[polymarket_l2_ingestion]]) feeds this backtester. It captures all 4 event types needed: book snapshots, trades, price_change (with order-level detail), and best bid/ask. The per-shard parquet output structure works directly with `pd.read_parquet("directory/")` for loading into the backtester.

Current data: capturing since 2026-06-19, ~5 days accumulated, 4 universes (politics_negrisk, esports, crypto, unknown). Phase 4 analysis (first spread/adverse-selection snapshots) targets ~2026-07-03 after 2+ weeks of data.

## Related documents

- [[mm_backtest_research_roadmap]] — operational skeleton: phases, gates, iteration loop, kill criteria
- [[strat_market_making]] — MM strategy hub (MM Path B section)
- [[polymarket_l2_ingestion]] — L2 data pipeline architecture

## Open questions for Phase 0

- Exact Polymarket queue priority rules (price-time? pro-rata? hybrid?) — need to verify from exchange docs
- Whether L2 capture granularity (event-level, not order-level) is sufficient for the queue model, or if we need L3
- How to handle the per-shard parquet structure (many small files per day) in the backtester's data loader
- Minimum number of fill events needed to calibrate the queue model with statistical confidence
