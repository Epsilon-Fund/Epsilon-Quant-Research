---
title: "MM Join-2 Operator Runbook — the LOCAL 1-contract live measurement run"
created: 2026-07-07
status: active
owner: justin
project: polymarket-mm
para: project
hubs:
  - strat_market_making
  - COWORK
tags:
  - market_making
  - runbook
  - execution
---
# MM Join-2 Operator Runbook — the LOCAL 1-contract live measurement run

> Hub: [[strat_market_making]] · [[COWORK]] · Machinery findings: [[mm_engine_join2_machinery_findings]] · Bridge: [[mm_engine_join2a_bridge_findings]] · Gates: [[mm_politics_negrisk_live_loop_design]] · Latency spec: [[mm_latency_measurement_spec]]

## Summary

- **What this is.** The step-by-step operator procedure for the Join-2b/2c **live measurement run**: measure our own order latency with unexecutable probes, then run the MM-engine bridge 1-contract quoting loop on ONE screened politics-NegRisk market, with per-order confirmation. It is a **measurement loop, not a trading system** — the strategy config is the Task-5 v1-politics damage-control quoter lineage, and the pre-registered gates in [[mm_politics_negrisk_live_loop_design]] decide everything downstream.
- **Where it runs.** **Locally, on your own machine, to your own account (`@jamonator`) — NOT the VPS.** Your own IP is permitted; no VPN.
- **Money at risk.** 1 contract per order, `MAX_REAL_ORDERS` raised deliberately run-by-run, hard inventory cap of 5 contracts, USD caps ≤ $50. Worst-case exposure is a few dollars.
- **Who does what.** Everything in this note is **human-driven**. The machinery (latency harness, bridge wiring, calibration pipeline) was built and dry-run-proven agent-side; no agent places real orders.

## 0. One-time sanity — what the v0 gate already established (2026-07-08)

- `POLYMARKET_FUNDER` = `0xe2ef99558fc5170fcf5c9f73087e6c96ae3389e5` **is** the `@jamonator` proxy (Gamma public-profile match). Secrets in `.env` are SET.
- **Public reads show $0 for this account by design** — data-api `/value` counts open-position value only, and deposited cash is NOT held as on-chain USDC in the proxy. Do not use public endpoints to check funding. The authoritative check is step 1.2 below.
- One dead position exists (ECB June-2026 YES ×40, resolved worthless, redeemable for $0). Expected; nothing to do.

## 1. Pre-flight checklist (every session, in order)

Run everything from the **repo root**, execution venv, with `polymarket/execution/.env` holding the secrets. The non-secret run env lives in `polymarket/execution/scripts/mm_join2_live.env.example` — source it on top.

1. **Funder identity.** `.env → POLYMARKET_FUNDER` ends `…89e5` and equals the wallet on your Polymarket account page. (Confirmed at the v0 gate; re-check after any .env edit.)
2. **Funding present — authoritative balance read (read-only, no order):**
   ```bash
   PYTHONPATH=. uv run --no-project --with py-clob-client \
       python polymarket/execution/tests/probes/mm_join2_balance_probe.py
   ```
   PASS iff the printed collateral balance matches the UI (~$109). $0 here with cash in the UI ⇒ the credentials/funder don't belong to this account — **STOP**.
3. **Auth + open-orders read (read-only):**
   ```bash
   PYTHONPATH=. uv run --no-project --with py-clob-client \
       python -m polymarket.execution --mode maker --check-auth
   ```
   Exit 0 and an open-order count of 0 expected. Exit 5 ⇒ credentials/geo problem — **STOP**.
4. **Caps are tiny.** In the run env: `MAKER_SIZE_CONTRACTS=1`, `POLYMARKET_MAX_REAL_ORDERS=1` (first run), `POLYMARKET_REQUIRE_OPERATOR_CONFIRM=true`, `POLYMARKET_MM_BRIDGE_MAX_INVENTORY=5`, `PER_TRADE/PER_MARKET/MAX_CAPITAL` at their small .env values. The bridge banner prints all of these at startup — read it before confirming any order.
5. **Kill-switch path is clear** (`POLYMARKET_KILLSWITCH_PATH`, default `/tmp/polymarket_killswitch` — must not exist). To halt at any time: `touch /tmp/polymarket_killswitch`. Cancels still go out; no new orders do.
6. **Reconcile interval set.** `POLYMARKET_MM_BRIDGE_RECONCILE_EVERY=100` (events) — keeps the engine's resting set equal to the venue's open orders (external cancels / missed acks). Startup WARNs if unset.
7. **websocket-client present for the live feed:** run live commands with `uv run --with websocket-client --with py-clob-client …` (the read-only market WS needs it; DRY-RUN does not).
8. **⛔ On-chain allowances (first live session only).** The first real order needs USDC allowance for the NegRisk exchange (`0xC5d563A36AE78145C45a50134d48A1215220f80a`). If step 2's probe prints a zero/missing allowance, set the allowance on the Polymarket UI side (any deposit/first trade normally does this) and re-run the probe. Do not proceed while allowance is 0.

## 2. Select the market — the pre-registered 5-screen filter

From `polymarket/research/` (research venv; read-only):

```bash
PYTHONPATH=. uv run python scripts/mm_join2_market_screen.py --top 15
PYTHONPATH=. uv run python scripts/mm_join2_market_screen.py --emit-env 1   # → env lines
```

The screens (locked in [[mm_politics_negrisk_live_loop_design]] Decision 1): negRisk=true → bucket in {non-US elections, Trump personnel/policy, other politics, 2026 US races} (2028 outrights excluded) → ≥5% historical non-top3 maker share (UNKNOWN if the market has no history) → uninformed top-3 flow preferred → scheduled objective resolution. Pick from the top of the table, sanity-check the market on the Polymarket page (real book, sane spread, no pending UMA dispute), and paste the emitted `POLYMARKET_MAKER_CONDITION_ID` into the run env. **Prefer a mid-life market; near-expiry politics is toxic** ([[mm_market_screen_and_ttr_regime_findings]]).

## 3. Step (a) — latency measurement (2b)

Dry-run rehearsal first (fake venue, no network orders):

```bash
POLYMARKET_VENUE=fake POLYMARKET_MM_LATENCY_SAMPLES=5 POLYMARKET_MM_LATENCY_CADENCE_S=0 \
PYTHONPATH=. uv run --no-project --with py-clob-client \
    python -m polymarket.execution --mode mm_latency
```

Then the real measurement (unexecutable probes: BUY@0.001 / SELL@0.999, 1 contract, cancelled on ack; each probe consumes one unit of `MAX_REAL_ORDERS` and prompts for confirm):

```bash
POLYMARKET_VENUE=real POLYMARKET_MAX_REAL_ORDERS=200 \
POLYMARKET_MM_LATENCY_SAMPLES=200 POLYMARKET_MM_LATENCY_CADENCE_S=30 \
PYTHONPATH=. uv run --no-project --with py-clob-client --with websocket-client \
    python -m polymarket.execution --mode mm_latency
```

Notes: the harness refuses to probe near-resolved books (within a tick of 0/1); per-probe confirm at K=200 is tedious — `POLYMARKET_REQUIRE_OPERATOR_CONFIRM=false` is acceptable **for the probe session only** (probes are unexecutable by construction + budget-capped); restore `true` before step 4. Output: samples JSONL + fit JSON in the journal dir, and a stdout line `→ set POLYMARKET_MM_BRIDGE_LATENCY_MS=<N>` — put that number in the run env. Sanity: mean ≈ 100–500ms; a fat p99 (multi-second) on politics is fine (latency ~immaterial there — the spec's stratification).

## 4. Step (b) — the 1-contract quoting loop (2c), per-order confirm

Dry-run rehearsal first (`POLYMARKET_VENUE=fake`, same command minus `real`). Then:

```bash
POLYMARKET_VENUE=real POLYMARKET_MAX_REAL_ORDERS=2 \
PYTHONPATH=. uv run --no-project --with py-clob-client --with websocket-client \
    python -m polymarket.execution --mode mm_bridge
```

- Read the startup banner: venue, caps, `latency_ms`, `max_inventory`, `reconcile_every` — all as intended, **no WARNING lines**.
- The first reconcile proposes a two-sided join-the-touch quote (1 contract each side). You confirm each order on stdin. `MAX_REAL_ORDERS=2` = exactly one two-sided quote set; nothing more can flow this session.
- Later sessions: raise `MAX_REAL_ORDERS` deliberately (e.g. 10–20 to allow refreshes/replaces), keep per-order confirm on until you trust the loop, keep `MAX_INVENTORY=5`.
- **Stop conditions** (any one ⇒ `touch /tmp/polymarket_killswitch`, then Ctrl-C):
  - any journal `RISK_HALT` you did not expect (`ambiguous_submit` especially — check the UI, reconcile manually before restarting);
  - inventory pinned at the cap (banner side withheld) for more than ~an hour — the book is one-way toxic;
  - fills clustering right after news (`news_proximate=true` in telemetry);
  - anything that looks like a mis-priced quote at the venue (UI shows an order you don't recognize — the periodic reconcile should cancel orphans, but verify).
- **Session end:** Ctrl-C (cancels resting quotes), then check the Polymarket UI for zero open orders.

### First-fill mini-audit (do this on the day of the first real fill)

1. Journal `MAKER_FILL_TELEMETRY` row exists and matches the data-api trade (price/size/side/tx) — the same cross-check the copytrade PoC used (data-api is authoritative).
2. The fills-stream record carries `queue_ahead`, `own_round_trip_ms`, `news_proximate`, `fill_share_this_market` (None allowed where documented).
3. Position math: `position_after` = ±qty, `cost_basis_after` = fill price; the UI position agrees.
4. `DataApiFillSource` dedup: restart the loop; the same fill must NOT book twice (session filter + composite key).
5. Note the fill in the measurement log (market, bucket, time-to-resolution, book state at fill).

## 5. Step (c) — calibration (2d) + the pre-registered gates

After each session with fills, run the calibration pipeline over the session's recorded events + fills (paths from the journal/telemetry):

```bash
POLYMARKET_MM_CALIBRATE_EVENTS=<events.jsonl> POLYMARKET_MM_CALIBRATE_FILLS=<fills.jsonl> \
POLYMARKET_MM_CALIBRATE_LATENCY_SAMPLES=<latency_samples.jsonl> \
POLYMARKET_MM_CALIBRATE_OUT=<report.json> \
PYTHONPATH=. uv run --no-project --with py-clob-client \
    python -m polymarket.execution --mode mm_calibrate
```

It fits `ProbQueue.f` (external grid — the frozen `calibrate()` is a stub by design) + the latency constant and prints the **bracket-collapse report**: the [RiskAverse, Optimistic] fill bound and where the fitted point lands. Early sessions have tiny fill counts — treat the fit as directional until tens of fills exist.

**The gates that decide everything** (pre-registered in [[mm_politics_negrisk_live_loop_design]] Decision 3 — do not re-derive): fill share > 0% in ≥5 markets; post-fill 60s drift lower CI > −500 bps; news-proximate adverse fills < 50%; net-of-cost lower CI > 0 over **≥30 settled markets** (or 90 days, whichever first); resolution drag < 10%. Quoting parameters are **frozen** for the duration — log, don't tune. The 2e gate read-out over ≥30 settled markets is a separate later step (reuses the Task-4/5 metric machinery).

## Appendix — exact env reference

Non-secret template: `polymarket/execution/scripts/mm_join2_live.env.example`. Secrets stay in `polymarket/execution/.env` (never echo them; the tools here never print them). Journal + latency/calibration artifacts land in `POLYMARKET_JOURNAL_DIR`.

| Knob | First live session | Why |
|---|---|---|
| `POLYMARKET_VENUE` | `real` (after a fake rehearsal) | the only thing that arms the gate |
| `MAKER_SIZE_CONTRACTS` | `1` | 1-contract measurement loop |
| `POLYMARKET_MAX_REAL_ORDERS` | `2` (loop) / `200` (probe session) | hard submit budget, per process |
| `POLYMARKET_REQUIRE_OPERATOR_CONFIRM` | `true` (loop) | you approve every order |
| `POLYMARKET_MM_BRIDGE_MAX_INVENTORY` | `5` | hard contracts cap — clip sizes clamp to the remaining band, so worst-case position is exactly ±cap |
| `POLYMARKET_MM_BRIDGE_RECONCILE_EVERY` | `100` | resting set == venue's open orders |
| `POLYMARKET_MM_BRIDGE_LATENCY_MS` | from step 3 | the measured submit→ack constant |
