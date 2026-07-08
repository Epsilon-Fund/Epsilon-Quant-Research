---
title: "MM Latency Measurement Spec — Measuring Our Own Submit→Ack Round-Trip with Unexecutable Probe Orders (Phase 2)"
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
  - latency
  - data-quality
  - measurement
---

# MM Latency Measurement Spec — Submit→Ack Round-Trip via Unexecutable Probe Orders

> Hubs: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · [[COWORK]]
> Why latency matters + the model ladder: [[mm_backtesting_methodology_explainer]] §2 · what the public feed exposes + timestamp semantics: [[mm_clob_capture_semantics]] · the stubs this feeds: `mm_engine/latency_models.py` (`ConstantLatency`, `SampledLatency`) · the queue side of the same calibration: [[mm_engine_queue_models]] / [[mm_queue_model_audit_fixes_findings]].

## Plain-English Summary

- **What:** the Phase-2 protocol for measuring *our own* order round-trip latency on Polymarket and turning it into the engine's latency model. We submit deep **unexecutable** orders (BUY at 0.001, SELL at 0.999), time **submit→ack**, cancel immediately, and fit a `(mean, std)` per market category.
- **Why:** the backtest's fill gate only lets a resting quote fill once it could have *landed* (`placement_ts + round_trip ≤ trade_ts`). The round-trip is a free knob that fabricates or denies fills — hftbacktest reports the *same* strategy swinging Sharpe −0.20 → +1.54 → −0.38 by changing only the latency model ([[mm_backtesting_methodology_explainer]] §2). We currently *assume* ~200ms (`ConstantLatency` default); this spec replaces the assumption with a measurement.
- **How it lands:** the fitted mean → `ConstantLatency(round_trip=mean)`; the fitted `(mean, std)` → `SampledLatency.from_samples(...)` so the backtest sees latency *dispersion*, not just its mean. Both are already wired behind the frozen `LatencyModel` protocol.
- **Stratify:** politics (slow — latency ~immaterial, a coarse constant is fine) vs crypto-4h / in-play sports (fast — latency *material*, the dispersion bites). Same venue, so the wire round-trip is similar; what differs is how much the gate *matters*, which decides which model to apply where.
- **Status:** spec only. This is a **live measurement loop** (Phase 2 / Join 2), not yet run — it needs a funded, signing-capable account (bridge to `execution/maker/`). Until then the engine runs on the assumed ~200ms constant.

## Why measure latency at all

A replay backtest cannot move the market; it asks "would my resting order have filled?" Two unknowns decide the answer: queue position (size ahead — see [[mm_engine_queue_models]]) and **latency** (was my quote even live when the trade printed?). The fill simulator (`mm_engine/fills.py`) encodes the latency leg as a gate:

> a resting order can fill a trade only if `placement_ts + round_trip_ms(placement_ts) ≤ trade.ts_exchange`.

Set the round-trip too low (0ms = instant) and the backtest fills quotes you could never have placed in time — pure over-optimism. Set it too high and you deny real fills. So the round-trip must be **measured**, not guessed, and — because it bites in proportion to how fast a market moves — **measured per market regime**.

## The protocol

### 1. Probe with unexecutable orders

Submit an order priced so far out of the market that it **cannot fill**, then cancel it the moment it is acknowledged. It never rests long enough, and never at a fillable price, to take on real exposure — its only purpose is to time the round-trip. This is exactly how hftbacktest's `IntpOrderLatency` is fed: round-trips "collected by submitting unexecutable orders regularly" ([[mm_backtesting_methodology_explainer]] §2).

- **BUY probe:** price **0.001** (the lowest tick on PM's 0.001 grid). It rests far below any ask, so it cannot match unless someone *sells at 0.001* — only possible in a near-resolved/degenerate book.
- **SELL probe:** price **0.999** (the highest tick). It rests far above any bid, so it cannot match unless someone *buys at 0.999*.
- **Size:** 1 contract (minimum), to bound the worst case.
- **Safety guard (avoid the one fill path):** only probe a market whose book is strictly inside the probe prices — `best_ask < 0.999` for a SELL probe, `best_bid > 0.001` for a BUY probe — i.e., skip markets trading within a tick of 0/1 (near-resolved). If both ends are safe, prefer the side further from the touch.

### 2. Time submit→ack on the *local* clock

- Stamp `t_submit = time.monotonic_ns()` immediately before the signed order request leaves us.
- Stamp `t_ack = time.monotonic_ns()` when the venue acknowledges the order — the REST response and/or the order `PLACEMENT` confirmation on the authenticated **user** channel (the user channel is the only place our own order lifecycle is visible — [[mm_clob_capture_semantics]] § Public Market Websocket / authenticated user channel).
- `round_trip_ms = (t_ack − t_submit) / 1e6`.
- **Cancel** the probe right after the ack (a second round-trip we may also log separately as cancel latency).

Use the **monotonic** clock on both ends — this is a *local* submit→ack interval, so it sidesteps the exchange-vs-local clock skew that can make absolute one-way latency look negative ([[mm_clob_capture_semantics]] § Timestamp Semantics). It measures **order-entry + order-response** latency; **feed latency** (market-data delay) is a separate term in the tick-to-trade taxonomy and is out of scope for this probe (approximate or ignore it for now — note it in the ledger).

### 3. Accumulate, then fit

- Probe on a fixed cadence (e.g. every 30–60s per market) until at least **K ≈ 200–500** samples per market category, across multiple sessions/times-of-day.
- Fit per market, then aggregate to category: report **mean, std, and p50/p90/p99** (the tail matters — a fat upper tail means more missed fills than the mean implies). Handle outliers explicitly: a few multi-second network stalls should be winsorized or trimmed, and **both raw and trimmed fits reported** (an honest-in-both-directions gate — don't let one stall inflate the whole model, but don't hide a genuinely fat tail either).

### 4. Stratify: politics vs crypto / sports

Take the measurement **in each market's live conditions** and keep separate fits:

| category | speed | latency's role | model to apply |
|---|---|---|---|
| **politics NegRisk** | slow (days–weeks; seconds-scale book) | ~immaterial — books don't move fast enough to snipe a 200ms-stale quote | a coarse `ConstantLatency(mean)` is sufficient |
| **crypto-4h / daily up-down** | fast (a CEX leads the PM) | **material** — a stale quote is sniped; this is the K2-family adverse selection | `SampledLatency(mean, std)` — the dispersion changes fill counts |
| **in-play / live sports** | fast (jumps on goals) | **material** — seconds-to-minutes stale-quote window after a surprise | `SampledLatency(mean, std)` |

The wire round-trip to the *same* CLOB is similar across categories; what differs is **how much the gate matters**, which decides whether the mean alone (politics) or the full distribution (crypto/sports) is worth modelling.

## How the fit lands in the engine

Both models are already behind the frozen `LatencyModel` protocol in `mm_engine/latency_models.py`:

- **Constant (politics, or a first pass anywhere):** `ConstantLatency(round_trip=fitted_mean_ms)`. The current default is a placeholder `DEFAULT_ROUND_TRIP_MS = 200.0` — this spec's measurement *replaces* that number.
- **Sampled (crypto / sports):** `SampledLatency.from_samples(measured_round_trips_ms, floor_ms=0.0)` computes `(mean, std)` and draws `Normal(mean, std)` clamped to `≥ 0`. The draw is a **pure function of the submit timestamp** — keyed on `(seed, int(ts_exchange))` — so a given order's round-trip is fixed once at placement (not re-rolled each time the fill simulator probes it against a later trade), and replay stays deterministic across processes. `SampledLatency.calibrate(new_samples)` refits `(mean, std)` in place as more data arrives.
- **Eventual successor:** hftbacktest's `IntpOrderLatency` (interpolate *measured* round-trips by time-of-day / regime). `SampledLatency` is the placeholder until enough data justifies the richer model.

### Worked example (one probe → one sample → the fit)

A SELL probe is submitted at 0.999 on a politics market whose book is 0.41 / 0.43 (safely inside). `t_submit` is stamped; the venue acks **187ms** later (`t_ack`); the probe is cancelled. That is **one** sample: `round_trip_ms = 187`. Repeat ~300 times across two sessions → mean **203ms**, std **41ms**, p99 **460ms**. For politics we ship `ConstantLatency(round_trip=203)`; if we later probe a crypto-4h market and get mean 210ms / std 55ms / a fatter p99, we ship `SampledLatency(mean=210, std=55)` there so the backtest feels the late-quote misses, not just the average.

## Assumption ledger (be explicit about what this does and doesn't measure)

- **Modeled / measurable here:** order-entry + order-response round-trip (submit→ack), per market category, mean + dispersion + tail.
- **Out of scope of this probe:** *feed* latency (how stale our market data is when we decide to quote) — a distinct tick-to-trade term; and any colocation/hardware effects (we are retail/cloud, not co-located). Note these in any result rather than folding them silently into the order-entry number.
- **Live-only prerequisites:** a funded, signing-capable account and the order/cancel path (bridge to `execution/maker/` safety + signing — the same Join-2 step the queue calibration needs). The probe places *real* (if unexecutable) orders, so it cannot run from the read-only research venv alone.

## Gates / acceptance

1. ≥ K (≈200–500) clean samples per category, across ≥2 sessions and times-of-day.
2. Mean stable session-to-session (no regime drift) before freezing a per-category fit; if it drifts, keep `SampledLatency.calibrate` live and re-fit.
3. Report mean, std, p50/p90/p99, and both raw and trimmed fits.
4. Apply the stratified rule above (constant for politics; sampled for crypto/sports).

## Decision and next step

- **This spec is Phase 2 / Join 2** — a *live measurement loop*, not a trading system. It is the latency twin of the queue model's `calibrate(f)`: both turn assumed parameters into measured ones so the offline fill-rate bound collapses onto a trustworthy rate ([[mm_backtesting_methodology_explainer]] §1 punchline — the 1-contract live loop is the instrument that makes the backtest reliable).
- **Next concrete step when the live loop opens:** wire a tiny probe-and-cancel routine through `execution/maker/`, collect ≥200 samples on one politics and one crypto market, fit, and replace the `DEFAULT_ROUND_TRIP_MS = 200.0` placeholder with the measured per-category models. Until then the engine runs on the documented ~200ms constant, clearly labelled an assumption.
