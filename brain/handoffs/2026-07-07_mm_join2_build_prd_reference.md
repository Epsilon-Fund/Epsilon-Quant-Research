---
title: "MM Join 2 — PRD reference (build the live-execution machinery: 2b + 2c + 2d)"
created: 2026-07-07
status: active
owner: justin
project: polymarket-mm
hubs:
  - COWORK
  - strat_market_making
tags:
  - prd
  - market_making
  - handoff
  - execution
---

# MM Join 2 (build) — PRD reference

> **What this file is.** The full, unlimited-length context for the Join-2 **build** (the live-execution machinery only). The work is launched by a `/goal` prompt (fable / Claude Code, ≤4000 chars) that reads this file **in full first**. Everything load-bearing lives here; the `/goal` is the short pointer and is **not committed** (lives in chat).
>
> **Role split.** You (the implementation agent) build and dry-run-prove the machinery. You do **not** go live, place real orders, or commit. Design is closed — if something is genuinely underspecified, make the smallest reversible choice and flag it. **Whenever a step needs the operator to confirm something (the env↔account cross-check) or to check something on the Polymarket website, STOP and wait for their explicit reply — do not proceed past a ⛔ GATE on your own.**

---

## Mission (one line)

Build the **Join-2 live-execution machinery** for the MM engine — the **2b** latency-measurement harness, the **2c** live-run wiring, and the **2d** calibration pipeline — as **DRY-RUN / mock code only**. It is the plumbing the operator will later drive live to their own account; it is **NOT** the live run, it places **NO** real orders, and it commits nothing.

## Read first, in order

1. Agent Bootstrap per `brain/VAULT_MAP.md` (seed/read `local_agents/codex.md`, then `brain/CODEX.md` as **law** — this is an execution-lane build, follow CODEX not COWORK — then `brain/TODO.md`, then `brain/COWORK.md` § Active threads).
2. This file (the PRD).
3. `brain/handoffs/2026-06-23_mm_engine_phase01_buildplan.md` — the build-plan + live status tracker (Join 1 DONE+LOCKED; Join 2 in progress).
4. `polymarket/research/notes/market_making/mm_engine_join2a_bridge_findings.md` — Join-2a: what the bridge already is and what the hardening pass fixed (phantom orders, session filter, dedup, reconcile).
5. `polymarket/research/notes/market_making/mm_politics_negrisk_live_loop_design.md` — the 5-screen market selection + the pre-registered live-loop gates.
6. `polymarket/research/notes/overview/data_quality/mm_latency_measurement_spec.md` — the latency-harness spec (unexecutable-order round-trip).
7. Execution package: `polymarket/execution/PLAN.md`, `polymarket/execution/CLAUDE.md`, `polymarket/execution/maker/README.md`; then the code you extend: `polymarket/execution/maker/mm_engine_bridge.py`, `polymarket/execution/maker/order_safety.py` (the `RealOrderGate`: `MAX_REAL_ORDERS` + `REQUIRE_OPERATOR_CONFIRM`), `polymarket/execution/maker/mm_bridge_cli.py`, `polymarket/execution/__main__.py` (`--mode mm_bridge`).
8. The **frozen** research engine the bridge imports (do **NOT** modify): `polymarket/research/mm_engine/interfaces.py`, `queue_models.py`, `latency_models.py`, `fills.py`, `engine.py`.

## Framing (prevents the predictable misreads)

- **The predictable misread is to "finish Join 2" by going live. Do NOT.** This task builds and dry-run-proves the machinery. The real 1-contract run is a **separate operator step**, run **locally to the operator's own account** (`@jamonator`, ~$109) — **not on the VPS** for now (the operator's own IP is permitted, so no VPS and no VPN are involved). Real money, real orders, live 2d calibration on real fills, and the 2e gate read-out are all **later and human-driven**.
- **The real safety net is not the prompt's gate — it is the code path.** Everything here runs against a **mock venue** with `REQUIRE_OPERATOR_CONFIRM=true` and a tiny `MAX_REAL_ORDERS`. A real order is impossible without a real venue **and** a raised cap **and** per-order operator confirmation. Preserve that; never weaken it.
- **Secrets are never printed.** You may read and display the **funder address** (public) and the risk caps. The private key / API secret / passphrase are confirmed **SET only** — never echoed to stdout, logs, or files.
- **Our funder is dormant.** Own-fill ingestion (`DataApiFillSource`) was validated read-only via a proxy wallet; it is only fully exercised on the operator's first *real* fill (2b/2c, out of scope here). Build and test the path against synthetic/mock fills.

## GOALS (observable)

1. **2b latency harness** — submits **unexecutable** orders (far-from-touch, immediately cancelled), times submit→ack, aggregates the round-trip, and sets the latency model (`ConstantLatency` for slow politics). Dry-run/mock-tested; real submission is gated behind the live venue + operator-confirm and is **not exercised here**.
2. **2c live-run wiring** for `--mode mm_bridge` against a real venue: the 5-screen politics-NegRisk market selection; 1-contract caps; `REQUIRE_OPERATOR_CONFIRM=true`; `POLYMARKET_MM_BRIDGE_RECONCILE_EVERY` set to a rate-limit-safe interval; target = a **local** operator run to the operator's account (not the VPS). Verified end-to-end in **dry-run/mock**.
3. **2d calibration pipeline** — wire `QueueModel.calibrate()` end-to-end on **synthetic** fills: fit `ProbQueue`'s power-law `f` and the latency constant, and emit the **"bracket collapse" report** (Optimistic/RiskAverse bounds → toward the fitted rate). Tested on synthetic fills; real fills come later.
4. **Operator runbook** (a note in `polymarket/execution/`) for the operator's local live run: the exact env, the pre-flight checklist, and the ordered steps — (a) latency measurement, (b) the 1-contract loop with per-order confirm, (c) stop conditions + the pre-registered gates + the first-fill mini-audit hook.

## NON-GOALS (explicit exclusions)

- **Placing any real order.** Everything is dry-run or mock. No exceptions.
- **The live execution itself** (operator's 2b/2c run) — a separate staged human step.
- **2d calibration on real fills** and the **2e gate read-out** (needs ≥30 real settled markets; reuses the Task-4/Task-5 metric machinery — coordinate, do not duplicate).
- **VPS deployment** — local-to-account only for now.
- **Any change to `interfaces.py`, Alvaro's `queue_models.py`/`latency_models.py`, or `_kernel`** — frozen and reconciled.
- **Committing anything** — leave all work staged for the operator's review. The `/goal` prompt is not committed.
- **Printing/logging any secret value.**

## PHASING (validate before build — v0 GATES v1)

- **v0 ⛔ GATE — env↔account cross-check (runs FIRST, before building anything):**
  - Read `polymarket/execution/.env`. Print **only**: `POLYMARKET_FUNDER` (address), `POLYMARKET_CHAIN_ID`, `POLYMARKET_SIGNATURE_TYPE`, and the caps (`MAX_CAPITAL_USD`, `PER_TRADE_CAP_USD`, `PER_MARKET_CAP_USD`, `SIZING_USD`, `MAX_OPEN_POSITIONS`, `MAX_REAL_ORDERS`). Confirm the private key / API secret / passphrase are **SET** but do **NOT** print them.
  - Cross-check **read-only**: query the public data-api positions/value endpoint for `POLYMARKET_FUNDER` (no key, no order) and print the account's current value + any open positions.
  - **STOP.** Ask the operator to confirm: (a) `POLYMARKET_FUNDER` == their `@jamonator` wallet (~$109), (b) the open positions shown are expected/clean, (c) the caps are the tiny 1-contract values they intend. **Pre-registered pass = all three confirmed.** On fail or no confirmation: **STOP, build nothing, report why.**
- **v1 (the gated slice):** GOALS 1–4, dry-run/mock, no real orders.
- **v2 (deferred — operator/human-driven, NOT in this build):** the operator's local live run → account; 2d calibration on real fills; the 2e gate read-out over ≥30 settled markets.

## SUCCESS (checkable, per phase)

- **v0:** the operator's funder/account/caps confirmation, recorded.
- **v1:** the full execution + mm_engine suites are **green**; a dry-run **proves** no real order is possible without a real venue **AND** raised `MAX_REAL_ORDERS` **AND** operator-confirm; the **opus adversarial review** of the live-order path returns clean (confirmed bugs fixed + regression-tested); the calibration pipeline emits a **bracket-collapse report** on synthetic fills; the **runbook** exists with the pre-flight checklist.

## GUARDRAILS

- **DRY-RUN / mock only, zero real orders, zero spend.** `REQUIRE_OPERATOR_CONFIRM=true` and tiny `MAX_REAL_ORDERS` preserved throughout.
- **Never print/log secrets** — funder address + caps only.
- **Do not touch** `interfaces.py` / Alvaro's `queue_models.py`·`latency_models.py` / `_kernel`.
- **Commit nothing** — leave staged for operator review; the `/goal` lives in chat, not the repo.
- **Repo invariants:** run from `polymarket/execution/` in its venv (the bridge does a `sys.path` insert of `research/` via `ensure_mm_engine_importable()`); deterministic/seeded; record→replay parity preserved.
- **Any step needing the operator to verify something on the Polymarket website is a hard ⛔ STOP-and-ask gate** (GATE 2) — e.g. confirming the funder's on-chain USDC allowance/approvals, or a chosen market's tradability. Print exactly what to check; do not proceed until they reply.
- **Self-QA MUST include an adversarial opus review of the live-order path** (see below).

---

## The two operator gates (do not proceed past either without a reply)

- **⛔ GATE 1 = v0 env↔account cross-check** (above). First thing, before building.
- **⛔ GATE 2 = any Polymarket-page verification.** If the build surfaces anything the operator must check on the Polymarket UI/account (on-chain allowance/approvals, market tradability, funding state), STOP and print the exact thing to check.

## Self-QA + adversarial review (v1 exit criteria)

1. Run the full execution + mm_engine test suites; report green/red.
2. Dry-run the entire path end-to-end against the mock venue; demonstrate that no real network order can be placed without (real venue) ∧ (raised `MAX_REAL_ORDERS`) ∧ (operator confirm).
3. **Spawn an adversarial opus review** targeted at the **live-order path specifically** (the latency harness's real-order submit + the 2c wiring + the `RealOrderGate`): hunt for any unintended real-order path, any secret leak, any safety-gate bypass, any accounting/dedup error (recall the 2a review caught phantom fills, trade-size incomparability, and dropped multi-fill tx). Fix confirmed bugs, add regression tests, and report the findings.

## Runbook scope (GOAL 4)

A note under `polymarket/execution/` giving the operator, for a **local** run to their account (not VPS): the exact env values; the pre-flight checklist (funder matches `@jamonator`, funding present, caps tiny, kill-switch, reconcile interval); then the ordered steps — (a) latency measurement via unexecutable orders, (b) the 1-contract loop under per-order confirm, (c) stop conditions, the pre-registered gates from the live-loop design doc, and the first-fill mini-audit.

## On return (operator requirement)

Audit before trusting: a plain-English explanation of what was built, an adversarial self-check (where would this be wrong?), then the deliverables — before any "it works." Nothing committed; no real orders.
