---
title: "MM Engine JOIN 2 (b/c/d) — Live-Execution Machinery: latency harness, live-run wiring, calibration pipeline (DRY-RUN/mock)"
created: 2026-07-08
status: active
owner: justin
project: polymarket
para: project
hubs:
  - strat_market_making
  - POLYMARKET_BRAIN
tags:
  - market-making
  - engine
  - execution
  - latency
  - calibration
  - live-loop
---

# MM Engine JOIN 2 — Live-Execution Machinery (2b + 2c + 2d, DRY-RUN/mock)

> Hub: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · Build plan: [[2026-06-23_mm_engine_phase01_buildplan]] · Bridge it extends: [[mm_engine_join2a_bridge_findings]] · Gates: [[mm_politics_negrisk_live_loop_design]] · Latency spec: [[mm_latency_measurement_spec]] · PRD: [[2026-07-07_mm_join2_build_prd_reference]] · Operator procedure: [[MM_JOIN2_RUNBOOK]]

## Plain-English Summary

- **What this is.** The Join-2 **live-execution machinery** for the MM engine, built and proven **dry-run/mock only**: (2b) a latency-measurement harness that times our own submit→ack round-trip with unexecutable probe orders; (2c) the live-run wiring for `--mode mm_bridge` — the pre-registered 5-screen market selection, 1-contract caps, a new **hard inventory cap in contracts**, the measured-latency hand-off, and the venue-reconcile cadence; (2d) a calibration pipeline that fits `ProbQueue.f` + the latency constant from fills and emits a **bracket-collapse report**. Plus the operator runbook.
- **Why.** Join 2a proved the bridge plumbing (same quoting code, real order path, real safety). What remained before the operator's first real 1-contract run was the measurement instrumentation: latency (a free knob that fabricates or denies backtest fills), calibration (collapsing the Optimistic/RiskAverse fill bracket toward a live-measured rate), and a safe, written procedure.
- **What it is NOT.** No real order was placed; nothing was committed; the live run itself, calibration on real fills, and the 2e ≥30-market gate read-out are later, human-driven steps. `interfaces.py`, Alvaro's queue/latency models, `fills.py`/`engine.py`, and `_kernel` are untouched (verified by git diff — the only mm_engine diffs are the pre-existing, documented Join-2a hardening in `engine.py`/`orders.py`).
- **Status.** Execution suite **331 green** (was 289; +42 new tests), mm_engine research suite **98 green** (85 untouched + 13 new screen tests). The no-real-order property is proven as tests: a venue submit requires (real venue) ∧ (`MAX_REAL_ORDERS` budget) ∧ (operator confirm) — each conjunct knocked out separately blocks everything, all three together allow exactly the budgeted submits. A real-feed dry run (fake venue + read-only public WS on a live screened market) ran the full operator path cleanly. An adversarial opus review of the live-order path confirmed 3 bugs (inventory-cap overshoot, un-cancelled ambiguous probe, ambiguous-submit orphan) — **all fixed + regression-tested** (see the review section).

## v0 ⛔ gate — env↔account cross-check (2026-07-08, recorded)

Pre-registered pass = operator confirms (a) funder == `@jamonator` (~$109), (b) positions clean, (c) caps tiny. Result: **PASS.**

- `.env` (whitelisted fields only; secrets confirmed SET, never printed): funder `0xe2ef…89e5`, chain 137, sig-type 1, caps $50/$20/$30 (max/trade/market), sizing $5, `MAX_REAL_ORDERS=1`, `REQUIRE_OPERATOR_CONFIRM=true`, venue=fake.
- Read-only public cross-check: Gamma public-profile for the funder returns `name: "jamonator"` → identity **confirmed**. Open positions: exactly one dead ECB June-2026 YES×40 (resolved worthless, $0). data-api `/value` = $0 and on-chain USDC (bridged + native) = $0.
- **Operator resolution of the $0-vs-$109 discrepancy:** the ~$109 is present in the account; public surfaces can't see Polymarket-custodied cash (data-api `/value` is positions-only; deposits are not held as on-chain USDC in the proxy). Consequence baked into the machinery: the runbook's funding pre-flight uses the **authed read-only CLOB balance-allowance probe** (`tests/probes/mm_join2_balance_probe.py`), never a public read — and the bridge got a **hard inventory cap** that bounds exposure in contracts *independently of any balance read* (operator directive).

## What was built (v1 deliverables)

| # | Deliverable | Files (all left uncommitted for review) |
|---|---|---|
| 2b | Latency probe harness + `--mode mm_latency` | `polymarket/execution/maker/mm_latency_harness.py` |
| 2c | Bridge wiring: `latency_ms` + hard `max_inventory_contracts` (env `POLYMARKET_MM_BRIDGE_LATENCY_MS` / `_MAX_INVENTORY`) + startup warnings | `maker/mm_engine_bridge.py`, `maker/mm_bridge_cli.py` (additive edits) |
| 2c | 5-screen politics-NegRisk market selection (research venv) | `polymarket/research/scripts/mm_join2_market_screen.py` |
| 2c | Ready-to-source live-run env template | `polymarket/execution/scripts/mm_join2_live.env.example` |
| 2d | Calibration pipeline + `--mode mm_calibrate` | `polymarket/execution/maker/mm_calibration.py` |
| — | Authed read-only balance pre-flight probe | `polymarket/execution/tests/probes/mm_join2_balance_probe.py` |
| — | Operator runbook | `polymarket/execution/maker/MM_JOIN2_RUNBOOK.md` ([[MM_JOIN2_RUNBOOK]]) |
| — | Tests (42 new execution + 13 new research) | `tests/maker/test_mm_latency_harness.py`, `test_mm_calibration.py`, `test_mm_join2_no_real_order_proof.py`, `test_mm_bridge_cli_e2e.py`, additions to `test_mm_engine_bridge.py`; `research/tests/test_mm_join2_market_screen.py` |

### 2b — latency harness (design per [[mm_latency_measurement_spec]])

Probes are **unexecutable by construction**: BUY at 0.001 / SELL at 0.999 (the extreme ticks), 1 contract, cancelled the moment the venue acks. The safety guard refuses to probe a book within a tick of 0/1 (the only fill path), and prefers the side further from the touch. Timing is `time.monotonic_ns()` stamped around the venue call only (a transparent `TimingVenue` wrapper), so our own bookkeeping never inflates the measurement. Samples fit to mean/std/p50/p90/p99, **raw and winsorized** (one network stall must not set the model, but the honest tail stays reported). The trimmed mean is printed as the exact `POLYMARKET_MM_BRIDGE_LATENCY_MS=<N>` line the bridge consumes; the `(mean, std)` pair loads `SampledLatency` for fast markets. **Crucially, a probe is gated exactly like a quote**: it goes through the same `VenueOrderRouter` (kill-switch → USD caps → `RealOrderGate`), consumes `MAX_REAL_ORDERS` budget on real venues, and honors per-order operator confirm.

*Practical example:* on a politics book of 0.41/0.43, the harness submits SELL 1 @ 0.999; the venue acks in 187ms; the probe is cancelled. Repeated ~200 times over two sessions → trimmed mean 203ms → the operator sets `POLYMARKET_MM_BRIDGE_LATENCY_MS=203` and the backtest's fill gate stops assuming.

### 2c — live-run wiring

- **Hard inventory cap (new, operator-directed):** the reused risk breakers bound USD notional only, and the account's cash is invisible to the bridge — so `POLYMARKET_MM_BRIDGE_MAX_INVENTORY` now caps |position| in **contracts**, as a true worst-case bound: each side's clip is clamped to its remaining band (`BUY ≤ cap − inventory`, `SELL ≤ cap + inventory`; a side with < 1 contract of room is withheld and journals `MakerQuoteSkipped(inventory_cap)`), so even a full fill stops exactly at ±cap (the adversarial review caught the original side-withholding version overshooting by up to a clip). With the cap disabled or slack the orders/quotes streams stay **byte-identical** to the backtest (tested).
- **5-screen selection** (pre-registered in [[mm_politics_negrisk_live_loop_design]] Decision 1): `negRisk=true` from Gamma → bucket ∈ {non-US elections, Trump personnel/policy, other politics, 2026 US races} with 2028 outrights excluded (keyword classifier, transparent per-candidate) → ≥5% historical non-top3 maker share from the corrected-carry cache (UNKNOWN when no history — new markets can't hard-pass a historical screen) → top-3 makers' volume-weighted directionality below the politics-top3 median (preference) → scheduled objective resolution inside the horizon (flag). Live read-only run (2026-07-08): 599 active negRisk rows → **165 candidates**; the top of the table is the 2026 Brazilian presidential basket (headroom 6–22%, clarity ✓), then the 2026 US midterm control markets.
- **Rate-limit-safe reconcile:** `POLYMARKET_MM_BRIDGE_RECONCILE_EVERY=100` events (politics books tick slowly → minutes between the read-only open-orders GETs). The CLI now **warns at startup** if the reconcile interval or the inventory cap is unset.

### 2d — calibration pipeline

`bracket_collapse_report(feed, observed_fills, latency_samples)` runs: the latency fit → `QueueModel.calibrate()` invoked end-to-end (the frozen hook — a **no-op stub by design**, so the actual fit is an **external grid search** over `ProbQueue(f)` replays through the real `run_engine`; no frozen file modified) → `SampledLatency.calibrate()` (a real in-place refit) → Optimistic/RiskAverse/fitted-Prob replays side by side. The synthetic ground-truth test builds a stream where 50 contracts join *behind* our order before a 60-contract cancel (the one case where attribution matters: Opt fills 80, RA 30), generates "observed" fills with a hidden `f_true`, and the fit **recovers `f_true` exactly** (parametrized 0.2/0.75/2.0); modeled fills are monotone in `f` and the bracket always holds.

*Practical example (the report on the synthetic set):* bracket before calibration `[30.0, 80.0]` contracts (width 50); observed 57.6; fitted `ProbQueue(f=0.75)` → 57.6, gap 0.00 — the bracket collapses onto the measured rate, which is precisely what the operator's real fills will do to the backtest's uncertainty.

## The no-real-order proof (v1 success criterion)

`tests/maker/test_mm_join2_no_real_order_proof.py` demonstrates, for **both** order-originating paths (bridge quotes and latency probes), that a venue submit requires the conjunction **(real venue) ∧ (budget remaining) ∧ (operator confirms)**:

| Conjunct knocked out | Result (asserted) |
|---|---|
| venue not real | placements flow to the fake adapter only; the real-order budget is untouched; operator never prompted |
| `MAX_REAL_ORDERS=0` | zero submits reach the venue; `RiskHalt(max_real_orders)` journaled |
| operator declines | zero submits; `RiskHalt(operator_aborted)` |
| **all three true** | submits flow and stop **exactly at the budget** (2), proving the blocks came from the gate |

Additionally, a real-feed dry run (fake venue + the read-only public WS on the top screened market) exercised the full operator command path: Gamma token resolution, live book frames, two-sided 1-contract quoting, staleness-conservative cancel/replace, clean shutdown, session lifecycle journaled — no secret anywhere in stdout or the journal.

## Adversarial opus review of the live-order path — 3 confirmed bugs, all fixed + regression-tested

A targeted adversarial review (opus) audited the live-order path — the latency harness's submit, the 2c wiring, and the `RealOrderGate` — hunting unintended real-order paths, secret leaks, gate bypasses, and accounting errors. It **confirmed 3 real bugs** (all fixed, each with a regression test) and flagged 2 plausible hardening items (both applied):

1. **(HIGH, confirmed) The "hard" inventory cap was a quoting band, not a position bound.** With cap 5 and clip 10, the BUY side was only withheld once `inventory ≥ cap` — so one full-clip fill landed position **10**, a 100% overshoot the USD breakers don't catch (10 contracts ≈ $4.70, far under every cap). **Fix:** the cap now clamps each side's **order size to the remaining band** (`BUY ≤ cap − inventory`, `SELL ≤ cap + inventory`; a side with < 1 contract of room is withheld) — so even a full fill stops exactly at ±cap. Regressions: `test_inventory_cap_clamps_clip_so_a_full_fill_cannot_overshoot` (including the asymmetric short-position case: inventory −3 → BUY band 8 / SELL band 2).
2. **(HIGH, confirmed) An *ambiguous* probe submit left an un-cancelled GTC resting at the venue.** On the venue-timeout path (`SubmitResult(ambiguous=True)` — the order *may have landed*), `probe_once` returned before its cancel block, leaving a far-tick GTC resting unattended — executable exactly if the market later runs toward resolution. **Fix:** on `ambiguous_submit`/submit-exception outcomes the harness fires a best-effort cancel-by-coid before reporting the block. Regression: `test_ambiguous_probe_submit_fires_best_effort_cancel`.
3. **(MEDIUM, confirmed) An ambiguous *bridge* submit orphaned the order:** the engine rolled the intent back (correct for a clean NACK) while the venue might hold it — and the ambiguous-submit **halt** breaks the loop before the next periodic venue reconcile could catch the orphan. **Fix:** the rollback now also fires a best-effort cancel-by-coid on ambiguous/exception outcomes, and `run()` performs one final read-only `reconcile_with_venue()` (cancels only, never submits) whenever it exits halted. Regression: `test_ambiguous_bridge_submit_fires_best_effort_cancel_and_final_reconcile`.
4. **(PLAUSIBLE, hardened) Balance-probe exception repr could reach stderr un-scrubbed** from the third-party client built with raw credentials. Now every secret value is redacted from the exception text and the output truncated.
5. **(PLAUSIBLE-LOW, hardened) A funder trade row with an unparseable timestamp bypassed the pre-session filter** (and collapsed the dedup key to `ts_ms=0`). The live fill path now treats `ts=None` as pre-session and skips it. Regression: `test_data_api_fill_source_skips_rows_with_no_timestamp`.

**Verified clean by the review:** the gate conjunction (no ungated submit path anywhere, counter semantics exact); the `TimingVenue` wrapper cannot fake venue realness (`is_real_venue` explicitly delegated); no secret can surface through the execution-side exception/journal paths; `ActiveOrder` identity accounting; kill-switch-first fail-fast ordering on every venue; calibration purity (fresh model per replay, no network/secret).

Post-fix suites: **execution 331 green** (+4 regressions), **mm_engine research 98 green**.

## Assumption ledger (CODEX § Realism calibration rule 3)

- **Modeled/mock here:** venue ack timing (mock returns instantly — the dry-run fit is plumbing proof, not a latency number); synthetic fills for the 2d fit; keyword bucket classification (operator overrides at selection time).
- **Live-only unknowns (unchanged by this build):** real submit→ack distribution; passive fill rate / queue reality (the whole point of 2b/2c); whether SELL-side probes are accepted without token inventory; adverse selection; the pre-registered gates' outcomes.

## Decision and next step

**v1 machinery COMPLETE, dry-run/mock-proven; nothing committed; no real order possible from this build.** The next step is entirely operator-driven, per [[MM_JOIN2_RUNBOOK]]: pre-flight (authed balance probe + `--check-auth`) → 5-screen pick → 2b latency measurement (~200 probes) → the 1-contract loop under per-order confirm → 2d calibration on the first real fills → the pre-registered gates over ≥30 settled markets (2e, later).
