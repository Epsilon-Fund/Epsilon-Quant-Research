---
title: "MM Engine JOIN 2a — Live Bridge: run the backtest's quoting code against the real venue (DRY-RUN/mock)"
created: 2026-07-01
status: active
owner: justin
project: polymarket
para: project
hubs:
  - strat_market_making
  - POLYMARKET_BRAIN
tags:
  - market-making
  - backtesting
  - engine
  - execution
  - bridge
  - live-loop
---

> **HISTORICAL EVIDENCE (2026-08-25).** Detailed findings behind the active market-making project, kept for verification. Read the de-jargoned canon surface first — [[strat_market_making]] + [[mm_model]] — and treat every number here as preliminary per the hub reliability ledger.


# MM Engine JOIN 2a — Live Bridge (same-code-path → real venue, DRY-RUN/mock)

> Hub: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · Build plan: [[2026-06-23_mm_engine_phase01_buildplan]] · Reconciliation it builds on: [[mm_join1_reconciliation_findings]] · Live-loop gates: [[mm_politics_negrisk_live_loop_design]] · Live maker safety/signing reused: [[mm_maker_infra_audit_findings]] · Queue/latency realism: [[mm_backtesting_methodology_explainer]]

## Plain-English Summary

- **What this is.** JOIN 2a is the *machine bridge* between the research market-making backtest engine (`polymarket/research/mm_engine/`) and the live execution stack (`polymarket/execution/maker/`). It lets the **exact same** strategy + queue/latency models + book builder that ran in the [[mm_join1_reconciliation_findings|JOIN-1 backtest]] place orders through the **existing** NegRisk-aware signer + venue adapter + safety harness. **Reuse, not rebuild.**
- **Why it was written.** The build plan ([[2026-06-23_mm_engine_phase01_buildplan]] § NOW) splits Join 2 into a Justin lane (this execution-safety bridge + the eventual 1-contract calibration loop) and an Alvaro lane (Task-4 gates). This note is Join 2a: **build + prove the bridge in DRY-RUN / against a MOCK venue — NO real orders.** Calibration (fitting `ProbQueue.f` / the latency constant from real fills) is Join 2b/2c and is operator-driven.
- **The one-line design.** *The only thing that swaps is the fill path.* In backtest, fills are realized by the `FillSimulator` against the recorded trade tape; in the bridge, the `OrderManager`'s place/cancel/replace ops are routed to the real venue and fills come back **from the venue** — fed into the **same** PnL accounting (`_apply_fill`) and the **same** telemetry schema. The quoting code (`strategy.quote → OrderManager.reconcile → BookTracker`) is byte-for-byte the code from the backtest.
- **Venv-integration choice (the decision this note documents).** The bridge runs in the **execution venv** and imports `mm_engine` as a **path dependency**. This is the least-duplicating integration: the execution side keeps its irreplaceable, independently-audited signer + safety, and the pure/stdlib `mm_engine` quoting stack drops in unmodified. No signing is copied into `mm_engine`; no `mm_engine` code is copied into execution.
- **Status / takeaway.** Bridge runs end-to-end against a mock venue; the same `SymmetricQuoter` + `OptimisticQueue` that drove JOIN-1 drive it; the full safety harness (`MAX_REAL_ORDERS`, `REQUIRE_OPERATOR_CONFIRM`, kill-switch, per-trade/market/deployed/daily caps) is in the order path; telemetry is a superset of the backtest's. An adversarial multi-lens review (10 agents) surfaced 3 live-path bugs — all fixed (see § Adversarial review). A **pre-2b hardening pass** then removed phantom resting orders (rollback-on-reject + periodic venue-state reconcile), wired `fill_share_this_market` in backtest + bridge, and smoke-tested the live fill ingestion against the real data-api (see § Join-2a hardening). **Now: full execution suite 289 green; mm_engine research suite 85 green.** No real orders placed; `interfaces.py` + Alvaro's queue/latency models + `_kernel` untouched.

## What JOIN 2a is (and is not)

Per [[2026-06-23_mm_engine_phase01_buildplan]] the joins are: **Join 0** = agree the frozen `interfaces.py`; **Join 1** = reconciliation (same-code-path 0% gap, done + locked); **Join 2** = calibration. Join 2a is the **execution-safety bridge** that Join 2b/2c calibration will drive. It proves the *plumbing* — same quoting code, real order path, real safety — but makes **no profitability claim and places no real orders**. Real-fill realism (collapsing the optimistic/pessimistic queue bracket toward the live-measured rate) is Join 2b/2c.

## The design — one swap, everything else reused

### What is identical to the backtest (imported unmodified from `mm_engine`)

| Component | Module | Role in the bridge |
|---|---|---|
| Book builder | `mm_engine.book.BookTracker` | reconstructs top-N + staleness from the *same* `MarketEvent` stream |
| Strategy | `mm_engine.strategies.SymmetricQuoter` | `quote(book, inventory, params)` — the identical A/B quoting logic |
| Order reconciliation | `mm_engine.orders.OrderManager` | diffs desired vs resting → place/cancel/replace/throttled ops |
| Queue / latency models | `mm_engine.queue_models` / `latency_models` | drive `queue_ahead` + `own_round_trip_ms` **telemetry** (Alvaro's models, FROZEN) |
| Telemetry | `mm_engine.telemetry.Telemetry` | the same append-only fills/orders/quotes JSONL schema |
| Fee/rebate + PnL | `mm_engine.fees.FeeModel`, `engine._apply_fill`/`_Pos`/`EngineResult` | the same average-cost accounting + 3-way PnL + `.settle()` |
| Market-data feed | `mm_engine.feeds.live_shadow.live_shadow_feed` | the same read-only public-WS `MarketEvent` source |

The per-event loop in `MMEngineBridge.on_market_event` is a line-for-line mirror of `run_engine`'s steps 2–4 (mark equity → re-quote → reconcile → queue snapshot). It emits the telemetry **before** routing to the venue, so on a fills-free feed the bridge's `orders` and `quotes` streams are **byte-identical** to the backtest's (a test asserts this).

### The only swap: the fill path

| | Backtest (`run_engine`) | Live bridge (`MMEngineBridge`) |
|---|---|---|
| Order placement | ops update the `OrderManager`, logged only | ops routed to the venue via `VenueOrderRouter` (real signer + adapter + safety) |
| Fills | `FillSimulator.simulate()` realizes fills on `last_trade` events (queue × latency gated) | `FillSource.poll()` returns **venue-reported** fills; `ingest_fill()` books them |
| Fill PnL/telemetry | `_apply_fill` + fills-stream dict | **the same** `_apply_fill` + a **superset** of the same fills-stream dict |

Nothing in the bridge simulates fills. The queue model is still driven (`on_event`, `get_queue_ahead`, `forget`) so `queue_ahead` and `own_round_trip_ms` are logged as the model's live **estimates** — the telemetry Join-2d calibration needs — but the fill *quantity and price are the venue's ground truth*.

### Practical example (one bridge cycle)

A `book` event arrives: best bid 0.47 / best ask 0.49. `SymmetricQuoter` (half-spread 0.01) emits BUY 0.47 + SELL 0.49; `OrderManager.reconcile` returns two `place` ops; the bridge logs both to the `orders` stream, then `VenueOrderRouter.place` runs the safety gate (fake venue in DRY-RUN → gate is a no-op) and calls `venue.submit_order(...)` — the exact same signing/venue path the copytrade and maker loops use. Later the venue reports a fill of 50 contracts @ 0.47 on our resting bid; `ingest_fill` books `+50` via `_apply_fill` (position 50, cost-basis 0.47, `gross_cash −23.50`), computes `queue_ahead` from the queue model, and emits a fills record carrying every backtest key plus `source="venue"`, `own_round_trip_ms`, `news_proximate`, `fill_share_this_market`. `result().settle({YES: 1.0})` then pays `50 × (1.0 − 0.47) = +26.5`.

## The venv-integration choice (documented decision)

**Decision: run the bridge in the execution venv; import `mm_engine` as a path dependency** (a `sys.path` insert of `polymarket/research` in `ensure_mm_engine_importable()`).

Why this direction and not the reverse:

- **The execution side owns the irreplaceable pieces.** The NegRisk-aware `ClobSigner`, the venue adapter chain (`cli.build_venue_adapter`), the safety harness, resolution/redemption, `negrisk_inventory`, and the journal are security-sensitive and independently audited ([[mm_maker_infra_audit_findings]]). Duplicating any of them into the research venv is exactly what the task forbids, and the research venv has no `py-clob-client` at all.
- **The `mm_engine` quoting stack is pure/stdlib**, so it drops into the execution venv unchanged. Verified empirically: `lib.clob_book` (the book builder's only external dep) imports only `dataclasses`/`typing`; every `mm_engine` leaf (`interfaces`, `events`, `book`, `orders`, `strategies`, `telemetry`, `queue_models`, `latency_models`, `fees`, `engine`) imports and runs under the execution runner. `fees.py` lazily imports pandas **only** on a category fee-table fallback — the bridge defaults to `FeeModel.fee_free_model()`, so that import never fires. `live_shadow` lazily imports `websocket` **only** on a real connection.
- **No boundary leak, one direction only.** A test scans every `mm_engine` source file and asserts it references no `py_clob_client` / `polymarket.execution` / `ClobSigner`; another asserts the bridge module imports no `_kernel`. Execution imports mm_engine; mm_engine never imports execution.

**Operational note (Join 2b/2c).** A *real* market-data feed uses `mm_engine.feeds.live_shadow`, whose transport needs `websocket-client` (present in the research venv, not the execution venv). The DRY-RUN/mock path in this task never touches it. For a real run the operator adds it: `uv run --with websocket-client python -m polymarket.execution --mode mm_bridge`.

## The safety harness in the order path (reused, not duplicated)

Every real submit passes, fail-fast, through:

1. **kill-switch** — `risk.check_kill_switch` (the pure breaker), checked on *every* venue so a kill-switch file stops even a dry run; trips `halted` + a `RiskHalt`.
2. **per-trade / per-market / deployed / daily caps** — `risk.check_size_cap` / `check_market_cap` / `check_deployed_cap` / `check_daily_loss` (the same pure breakers the copytrade bot uses), fed a per-quote `CandidateOrder` + `RiskState`.
3. **`MAX_REAL_ORDERS` + `REQUIRE_OPERATOR_CONFIRM`** — the shared `RealOrderGate`, a **no-op on fake venues**, so DRY-RUN placements flow freely and only real venues are gated.

**"Do not duplicate" — the one refactor.** The `MAX_REAL_ORDERS`/operator-confirm gate previously lived inline in `MakerEngine._safety_block_reason`. It was extracted to `maker/order_safety.py::RealOrderGate` (single source of truth) and `MakerEngine` now delegates to it — a **behavior-preserving** change (identical `RiskHalt` reasons + counter semantics; full maker suite stays green). The bridge composes that same gate with the reused `risk` breakers. The signer + venue adapter are reused with **zero** duplication via `build_venue_adapter`.

## Telemetry parity (so Join-2d consumes live and backtest logs identically)

- **orders / quotes streams:** identical to the backtest (same `OrderOp.as_dict()`, same per-event quote-snapshot dict). A test asserts `bridge.orders == backtest.orders` and `bridge.quotes == backtest.quotes` on a fills-free feed.
- **fills stream:** a **strict superset** of `run_engine`'s fills record. Every backtest key is present (`ts_exchange, token_id, side, price, qty, queue_ahead, mid_at_fill, maker_fee, maker_rebate, taker_fee_ref, fee_rate, rebate_rate, fee_source, realized_delta, position_after, cost_basis_after, gross_cash_after, client_id, trade_ts, trade_price, trade_size`), plus the live-measurement fields the task enumerates: `source`, `venue_order_id`, `transaction_hash`, `matched`, `own_round_trip_ms`, `news_proximate`, `fill_share_this_market`. (Post-fill *markout* is not a stored field in either — it is derived downstream from `mid_at_fill` + the quotes-stream mid trajectory, exactly as in the backtest.) **One value-semantics caveat, made explicit:** the backtest's `trade_size` is the *aggressor trade's total print size* (can exceed our fill); a venue fill only reports *our* fill, so the aggressor total is unknown live and `trade_size` is emitted as **`None`** rather than falsely equated to our fill qty (our fill size is carried in `qty`). This keeps the two logs value-comparable for Join-2d instead of biasing any `qty/trade_size` ratio to 1.0. A test asserts `set(backtest_keys) ⊆ set(bridge_keys)`; another pins `trade_size is None` on the live path.
- **result shape:** `MMEngineBridge.result()` returns a real `mm_engine.engine.EngineResult` (mode `"live_bridge"`), so reconciliation / settlement / Join-2d treat a live run and a backtest run through identical code.

## Files delivered

| File | What |
|---|---|
| `polymarket/execution/maker/mm_engine_bridge.py` | `MMEngineBridge` (the loop), `VenueOrderRouter` (fill-path swap + composed safety), `VenueFill`/`FillSource`, `DataApiFillSource` (live fill glue reusing the maker's `DataApiTradeClient`), `BridgeConfig`, `ensure_mm_engine_importable()` |
| `polymarket/execution/maker/order_safety.py` | shared `RealOrderGate` (MAX_REAL_ORDERS + operator-confirm), now used by both the maker loop and the bridge |
| `polymarket/execution/maker/mm_bridge_cli.py` | operator entry (`--mode mm_bridge`); DRY-RUN default, 1-contract, wires the JOIN-1 models + read-only feed + `DataApiFillSource`; `build_bridge()` seam for tests |
| `polymarket/execution/maker/maker_engine.py` | refactored to delegate to `RealOrderGate` (behavior-preserving) |
| `polymarket/execution/__main__.py` | additive `--mode mm_bridge` dispatch |
| `polymarket/execution/tests/maker/test_mm_engine_bridge.py` | 19 tests (below) |

## Tests / verification

Unit of observation for the safety tests = one `MMEngineBridge` fed one/few hand-built `MarketEvent`s with a mock venue (`is_real_venue()` toggled). No test touches the network.

- **Same-code-path:** bridge uses the same strategy/queue objects; on a fills-free feed `orders`/`quotes` are byte-identical to `run_engine`.
- **Fill-path swap:** a venue fill books position/PnL via `_apply_fill`; unmatched funder fills are still booked (rank `None`); `result()` is an `EngineResult` and `.settle()` pays the resolution payoff.
- **Telemetry parity:** backtest fill-record keys ⊆ bridge fill-record keys; live-only fields present.
- **Safety in path:** real venue + `MAX_REAL_ORDERS=0` → no submit + `RiskHalt`/skip; operator-confirm declined → `operator_aborted`, no submit; accepted → submits; kill-switch file → no submit + halt; per-trade cap → `size_cap` skip; gap → venue cancels.
- **Venv boundary:** mm_engine imports no signing/execution; bridge imports no `_kernel`; `build_bridge` wires the JOIN-1 models; `BridgeConfig.from_env` parses `.env`.
- **Determinism:** two runs → identical `orders`/`quotes`.
- **DataApiFillSource:** matches funder fills to resting quotes; ignores others; idempotent; **skips pre-session history**; **emits all fills of a multi-fill tx**.

Suite results (after the hardening pass): **bridge tests 30 green**; **full execution suite 289 green** (was 259; +30); **mm_engine research suite 85 green**. The frozen `interfaces.py`, Alvaro's queue/latency models, and `_kernel` are **unmodified**; the hardening added small additive changes to `engine.py` (`fill_share` + tape counter) and `orders.py` (`drop_order`) — both of Justin's machine, not frozen — with the mm_engine suite (incl. the determinism + record→replay reconciliation tests) staying green.

## Adversarial review (10-agent workflow) — 3 confirmed bugs, all fixed

After the first-pass tests were green, a multi-lens adversarial review (5 finder lenses × independent verify pass, opus) audited the bridge. It confirmed **3 real bugs — all in the *live* `DataApiFillSource` / fill-record path** (the Join-2b/2c surface, not the DRY-RUN acceptance path) — and correctly dismissed 2 as non-issues (the OrderManager-vs-venue divergence on a safety-skipped place, already documented as a 2b/2c concern; and an ISO-timestamp edge that `data-api`'s numeric timestamps never hit). All three fixed + regression-tested:

1. **(HIGH) Pre-session history booked as phantom fills.** `DataApiFillSource.poll()` declared `session_start` but never used it, so on the first poll every historical funder trade with an unseen tx would be booked into position/PnL. The maker loop it mirrors guards this (`ts < session_start → skip`); the guard was dropped. **Fix:** skip rows older than `session_start`, reusing the maker's `_row_ts` parser. Regression: `test_data_api_fill_source_skips_pre_session_trades`.
2. **(MEDIUM) `trade_size` value-incomparability** — see the telemetry-parity caveat above. **Fix:** emit `None` on the live path. Regression: `test_bridge_fill_record_trade_size_is_none_on_live_path`.
3. **(MEDIUM) Multi-fill transactions dropped.** Dedup keyed on `transactionHash` alone drops every fill after the first when one tx sweeps several of our levels (understating position — the dangerous direction). Repo invariant is `(tx, log_index)`; `data-api` exposes no `log_index`. **Fix:** composite dedup key `(tx, asset, side, price, size, ts)`. Regression: `test_data_api_fill_source_emits_all_fills_of_multi_fill_tx`.

These bugs did not affect the DRY-RUN/mock acceptance path (which uses the mock venue's own fill source), but they would have corrupted the Join-2b/2c live measurement — exactly the kind of silent-accounting error the review exists to catch. The fixes tighten `DataApiFillSource` to faithfully mirror `MakerEngine.process_fills_once`.

## Join-2a hardening (pre-2b) — three items, DRY-RUN/mock only

A follow-up pass hardened the bridge before any real order. Still **no real orders** (mock/injected venues + read-only public endpoints only); the frozen `interfaces.py`, Alvaro's queue/latency models, and `_kernel` were **not** touched. `engine.py` and `orders.py` (Justin's machine, not frozen) gained small additive changes noted below. Suites after: **execution 289 green** (was 281; +8 hardening tests), **mm_engine 85 green** (additive `engine.py`/`orders.py` changes preserve determinism/reconciliation).

### 1. No phantom resting orders — the engine's resting set always equals the venue's

The bridge previously left a safety-skipped/rejected place in the `OrderManager` (it had optimistically recorded it), so the next reconcile saw an idempotent no-op and never re-proposed the intent — a phantom the venue never took. Two-part fix:

- **(a) Roll back on skip/reject.** `VenueOrderRouter.place` now returns a `PlaceOutcome(accepted, venue_order_id, reason)`; `accepted=True` only when the order is actually resting at the venue. On any not-accepted (gate skip, cap veto, venue NACK, ambiguous, or transport exception) the bridge calls the new **`OrderManager.drop_order(client_id)`** (additive primitive in `mm_engine/orders.py`, unused by the backtest) to remove the intent, so the next reconcile **re-proposes** it. Tests: `test_rejected_place_rolls_back_and_is_reproposed_next_event` (venue NACKs first, then accepts → order rests on the retry); `test_gate_skipped_place_is_reproposed_each_event_not_silenced` (4 skips over 2 events, not 2 — proves no silent phantom).
- **(b) Periodic venue-state reconcile.** New `MMEngineBridge.reconcile_with_venue()` reads the venue's ACTUAL open orders via the **existing** read-only `reconcile_open_orders(set())` (→ `venue_open_client_order_ids`; `get_open_orders()` fallback) — no new venue code, submits nothing — and syncs both directions: drops any engine order the venue no longer has (external cancel / missed ack / filled-away → `MakerQuoteCanceled reason="venue_reconcile_missing"`), and cancels any venue orphan the engine doesn't track (`venue_reconcile_orphan`). After it runs, `{venue_coid(o) for o in om.active_orders()} == venue open set`. Driven every `POLYMARKET_MM_BRIDGE_RECONCILE_EVERY` market events in `run()`. Tests: `test_reconcile_drops_order_the_venue_no_longer_has`, `test_reconcile_cancels_orphan_venue_order`, `test_resting_set_equals_venue_after_reconcile_both_directions`, `test_run_invokes_periodic_reconcile_at_interval`.
- **(c) Exercised on the real-venue code path** via a mock venue with `is_real_venue()==True` that simulates NACKs + external cancels + orphans — no real orders.

### 2. Read-only smoke of `DataApiFillSource` against the REAL data-api

`polymarket/execution/tests/probes/mm_bridge_dataapi_smoke.py` — funder ADDRESS only (from `.env`), no private key, places nothing (public `data-api/trades` GET). Result (2026-07-01):

- **Our funder is DORMANT** — 0 recent trades on data-api across all markets ⇒ no pre-session history that the session filter would need to exclude (and nothing to phantom-book). Expected pre-2b.
- Because our funder is dormant, the ingestion battery also ran against the **busiest real wallet** on a live top-volume politics-NegRisk market (`0xf426…`, the reality proxy): **100 feed rows parsed cleanly**; that wallet's **3 fills emitted**; **matched by (side, price)** — 1 synthesised resting level matched all 3 real fills; **session filter** with `session_start=now` emitted **0** (its history is correctly pre-session); **idempotent** re-poll emitted **0**. Multi-fill txs: none in that wallet's 3-row sample (the composite-key multi-fill emit is covered by the unit test instead).
- **Read:** the live fill-ingestion path parses, matches, session-filters, and dedups correctly against reality, not just fixtures. The only dimension not observed live was a multi-fill tx (small real sample) — unit-tested.

### 3. `fill_share_this_market` wired in backtest AND bridge

`fill_share = our fills / total market trades on the tape` (per token). Both sides count the **last_trade tape** as the denominator so the two fills streams carry a comparable ratio:

- **Backtest (`engine.py`):** added a per-token `market_trades` counter (every `last_trade` event) + `our_fills_by_token`, and the fills record now carries `fill_share_this_market`. Test: `test_fill_share_backtest_equals_our_over_total_on_synthetic_tape` (1/2 then 2/3 over a synthetic tape).
- **Bridge:** same counters; the fills record + `MakerFillTelemetry` now carry the live ratio (was `None`). Test: `test_fill_share_bridge_equals_our_over_total_on_synthetic_tape`.
- **Markout stays DERIVED, no code change** — computed downstream from `mid_at_fill` + the quotes-stream mid trajectory (documented in § Telemetry parity), identically for backtest and live.

## Decision and next step

**JOIN 2a PASSES (plumbing only), now hardened.** The backtest's quoting code drives real orders through the existing safety + signing; the only swap is the fill path; the engine's resting set is kept equal to the venue's (rollback-on-reject + periodic venue reconcile); `fill_share` is populated in both backtest and bridge; the live fill-ingestion path is verified against the real data-api. Nothing real was placed; the frozen `interfaces.py` / queue-latency models / `_kernel` were not touched.

Concrete next steps (not done here, by instruction):

1. **Join 2b — first real 1-contract quoting** on one politics-NegRisk market via `--mode mm_bridge` with `POLYMARKET_VENUE=real`, `MAX_REAL_ORDERS=1`, `REQUIRE_OPERATOR_CONFIRM=true`, per the operator runbook and the pre-registered gates in [[mm_politics_negrisk_live_loop_design]]. Add `websocket-client` to the run venv for the live feed, and set `POLYMARKET_MM_BRIDGE_RECONCILE_EVERY` so the periodic venue reconcile runs.
2. **Join 2c — calibration:** feed the live fills through `QueueModel.calibrate()` to fit `ProbQueue.f` / the latency constant and collapse the optimistic/pessimistic bracket toward the live-measured rate; re-run the backtest and re-reconcile.
3. ~~`fill_share_this_market` populate~~ — **DONE** (this hardening pass; backtest + bridge). Real markout stays derived downstream (no code needed).
4. ~~Retry/reconcile of a safety-skipped intent~~ — **DONE** (this hardening pass: rollback-on-reject re-proposes; periodic venue reconcile syncs the resting set to the venue).
