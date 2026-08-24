---
title: "MM Join-2 C5 live session — real-venue rebuild, measured latency, and the two Step-1 redesign findings"
created: 2026-07-12
status: active
owner: justin
project: polymarket-mm
hubs:
  - COWORK
  - strat_market_making
tags:
  - market_making
  - execution
  - join2
  - live-loop
  - handoff
  - redesign
---
# MM Join-2 C5 live session — real-venue rebuild, measured latency, and the two Step-1 redesign findings

> Hub: [[strat_market_making]] · [[COWORK]] · PRD: [[2026-07-09_mm_join2_validation_prd_reference]] · Part-A verdict: [[mm_join2_model_fragility_findings]] · Runbook: [[MM_JOIN2_RUNBOOK]] · Operator guide (now superseded-pending-redesign): [[MM_JOIN2_C5_LIVE_GUIDE]]

## Plain-English Summary

- **What this session did (implementation lane).** Ran Join-2 Part C (live measurement) on the Brazil-presidential NegRisk event. Two real artifacts were produced: **(1) a measured order latency — 160 ms** (30 clean unexecutable probes), and **(2) the first-ever real resting order placed by this machinery** (a 5-share BUY that rested, then auto-cancelled). It also uncovered that the real-venue order path had **never actually worked** (Join-2a was fake-venue-proven only) and rebuilt it.
- **Why it stopped before a fill (operator call).** Two design findings surfaced that make the current Step-1 instrument wrong, and they're strategic (instrument design), so Justin paused live work for a **Cowork consolidation + Step-1 redesign** rather than pushing to a fill.
- **The two findings.** (A) **You cannot rest an ask on a token you don't own** — a no-inventory maker is *bid-only*; true two-sided event MM means resting a **BID on YES *and* a BID on NO** (the complement), not both sides of one token. (B) **The bid rested for only ~1–2 s** — the quoter churns (cancels + re-quotes on every mid tick) and with a tiny order budget went flat immediately; the order never persisted long enough to be seen or filled. A supporting finding: (C) on a deep book (Lula, ~51k ahead) a 5-share passive bid has **near-zero fill probability** — queue depth, not spread, is the binding constraint for getting a calibration fill.
- **Status.** All code staged, **nothing committed**, `_kernel` untouched, suites green (execution 359 + gateway 7 + market-metadata). Account clean: 0 open orders, 0 new positions, **$0 spent**. Next actor: **Cowork** (consolidate + redesign Step 1), then back to the implementation lane to build the redesigned instrument.

---

## 1. What was measured / achieved

- **Latency (Part C / 2b): `POLYMARKET_MM_BRIDGE_LATENCY_MS = 160`.** 30 unexecutable BUY probes on the Flávio market, all accepted: mean 160.3 ms, std 21.8, p50 155, p90 179, p99 233. Validates the model's ~200 ms `ConstantLatency` assumption (Part-A A3 said slow-politics latency should be immaterial and ~200 ms — confirmed). Measures submit→ack round-trip on the local monotonic clock; excludes feed latency. BUY-only (no token inventory for SELL probes) and from one ~1-min window → a first point estimate, not a distribution.
- **First real order (Part C / 2c):** `BUY 5 @ 0.60` on Lula, venue id `0x5d3b299bdf…709d4c`. Venue lifecycle confirmed **`status: CANCELED, original_size 5, size_matched 0, associate_trades []`** — it genuinely rested then was cancelled by our own reconcile on the next mid move. **Not a phantom; the V2 path places real resting orders.**
- **Probes + order left 0 resting, 0 fills, $0 net.** Account: only the pre-existing dead ECB dust (40 YES @ 0.096, redeemable ~$0).

## 2. The real-venue path was never exercised — rebuilt this session

Join-2a was **fake-venue-proven only**; the first real API contact (the latency probe) surfaced a chain of latent bugs, and Polymarket had **archived `py-clob-client`** (venue now rejects V1 orders: "invalid order version, please use the latest clob-client"). Fixes (all in `mirror/` + `execution/maker/`; frozen `_kernel` untouched):

1. **Probe size** hardcoded to 1 < venue min 5 → env-configurable (`MAKER_SIZE_CONTRACTS`), default preserved.
2. **Tick** read from Gamma keys `tick_size`/`tickSize`, but live Gamma ships `orderPriceMinTickSize` (0.001) → defaulted to 0.01 → valid prices rejected AND quotes would collapse. Now reads the live key (`mirror/market_metadata.py`).
3. **price/size** passed as strings; py-clob `OrderArgs` wants floats → `TypeError: '>=' str vs float`. Coerced (`mirror/clob_signer.py`).
4. **tick_size** passed as float; py-clob wants the `Literal['0.001',…]` string → `KeyError: 0.001`. Mapped to the literal.
5. **V2 SDK migration:** placement now goes through the successor SDK (`polymarket-client`, GitHub `Polymarket/py-sdk`) running in an **isolated child process** (`mirror/pysdk_gateway.py` + `mirror/pysdk_order_gateway.py`), because the SDK's top-level `polymarket` package shadows this repo's `polymarket/`. Parent↔child speak JSON-lines; child failures map onto the kernel's ambiguity model (timeout→TimeoutError, transport→OSError, clean reject→NACK). Wired into `cli.build_venue_adapter` (real path, fail-closed). Legacy V1 wire path retained only for the gateway-less fake/test path.

Adversarial review of the V2 path (2026-07-12) → fixed: **HIGH** timeout/head-of-line hazard (parent now waits > child's SDK timeout so a rollback cancel is never queued behind a wedged post); **MEDIUM** secrets inherited into the child env (now stripped — stdin `init` only); a **stale-response race** in the parent read loop (rewritten to a reader-thread + queue, race-free); added the missing direct gateway tests (timeout/EOF/stale-id/scrub). Remaining accepted items: `cancel_all` blast-radius assumes single-bot-per-funder (true here); NegRisk/tick side-channel is dead on the gateway path (SDK resolves internally — safe, but the independently-audited NegRisk logic no longer participates).

Suites after rebuild: **execution 359 + gateway 7 green**; mm_engine untouched (172 green from Part A).

## 3. The two Step-1 redesign findings (for Cowork)

### Finding A — no-inventory maker is bid-only; two-sided event MM = YES-bid + NO-bid

`SELL 5 @ 0.61` was rejected: `not enough balance / allowance … balance: 0, order amount: 5000000`. A SELL (ask) on the YES token must escrow 5 YES tokens; we hold none. **You cannot quote an ask on a token you don't own.** So the current instrument (SymmetricQuoter resting bid+ask on ONE token) can only ever rest the bid until it fills.

**The design consequence:** two-sided liquidity provision on a Polymarket *event*, without pre-holding inventory, is done by resting a **BID on YES** and a **BID on NO** (the complementary token). Since YES + NO = $1 at resolution, a resting bid on NO at price `p` is economically a resting **ask on YES at `1 − p`**. "Buy NO cheap" == "sell YES dear." This reshapes the quoter, the queue-fill accounting, and the costing convention around the **complementary pair**, not a single token. The whole Part-A trust question (queue-fill model + executable-touch costing) has to be re-examined in the YES/NO-pair frame.

Open questions for Cowork:
- Does the Join-2 measurement instrument become a **paired YES-bid / NO-bid quoter**? (The mm_engine `SymmetricQuoter` and the bridge currently quote one token two-sided.)
- How does inventory/exposure net across the pair (a NO fill = short-YES exposure); how does the 5-contract inventory cap apply per-token vs per-event?
- Costing/marking (`mm_eval/protocol._liq_mark`) and the queue models were all derived single-token — do they hold under paired quoting, or does the executable-touch convention need a pair-aware restatement?

### Finding B — the bid never persisted (churn + tiny budget)

The BUY rested ~1–2 s then our own reconcile cancelled it when the mid ticked (desired price changed → cancel the old, and with `MAX_REAL_ORDERS=2` spent, no replacement → flat). Justin never saw it on the site. **The Step-1 instrument churns the quote away instantly** rather than resting it.

Open questions for Cowork:
- Step 1 should probably be **"place one bid and HOLD it"** (rest a persistent order, observe it live, measure whether/when it fills) — not the continuous cancel-on-every-tick re-quote loop. Does the measurement want a *static resting probe-order* mode distinct from the *quoting* mode?
- Budget/`MAX_REAL_ORDERS` semantics: rejected SELLs consume budget; a churning quoter burns budget on every tick. A redesign should decouple "keep one order resting" from "re-quote on every event."

### Finding C (supporting) — deep queue ⇒ near-zero passive fill

Joining the touch on Lula puts a 5-share bid behind ~51k shares → it will not fill before the price moves. For *getting a calibration fill*, **queue depth beats spread**: thinner books (Flávio ~591 ahead; some longshots <50) fill sooner but have no spread to measure. There's a real tension between "wide spread to measure edge" and "thin queue to actually get filled" — the redesign should pick the market for the *measurement goal* (fills to calibrate the queue model), which favors thin queues.

## 4. Status / what's staged / next

- **Committed:** nothing. All changes staged on Justin's working tree. `_kernel` untouched.
- **New/changed files (this session):** `mirror/pysdk_gateway.py`, `mirror/pysdk_order_gateway.py` (new); `mirror/clob_http_client.py`, `mirror/clob_signer.py`, `mirror/market_metadata.py`, `maker/mm_latency_harness.py`, `maker/mm_engine_bridge.py`, `cli.py` (V2 path + wire fixes); `tests/mirror/test_pysdk_gateway.py` (new) + edits to `tests/test_clob_http_client.py`, `tests/test_market_metadata.py`, `tests/maker/test_negrisk_signature.py`, `tests/maker/test_mm_latency_harness.py`; `tests/probes/mm_join2_positions_probe.py` (new); runbook + C5 guide + this handoff.
- **Account:** clean — 0 open orders, 0 new positions, $0 spent. Latency 160 ms captured.
- **Next actor: Cowork** — consolidate this session and redesign Step 1 around Findings A (YES/NO paired quoting) and B (persistent-rest vs churn), choosing the measurement market for fill-ability (Finding C). Then hand the redesigned instrument back to the implementation lane to build + a fresh gated live pass.
- **C6 (calibration) is NOT done** — it needs real fills, which the redesign is a prerequisite for. The measured latency (160 ms) is ready to feed it when fills exist.

## 5. Where this could be wrong (adversarial self-check)

- The V2 gateway placed exactly **one** real order (then cancelled) — signing/placement is proven, but the **fill path, cancel-by-id under load, and reconcile of a genuinely-resting order over time are still unexercised** against the real venue. The adversarial review's HIGH (timeout/HOL) fix is reasoned + unit-tested, not yet observed under a real slow post.
- 160 ms is BUY-only, one window, n=30 — treat as a first point estimate.
- Finding A's "YES-bid + NO-bid = two-sided" is the standard NegRisk maker structure, but the exact netting/cap/costing implications under *our* models are unverified — that's the redesign's job, not a settled result.
