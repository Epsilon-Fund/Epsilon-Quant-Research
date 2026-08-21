---
title: "MM Join 2 — Model-Validation + Gated-Live PRD reference (2026-07-09)"
created: 2026-07-09
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
  - validation
  - live-loop
---

# MM Join 2 — Model-Validation + Gated-Live PRD reference

> **What this file is.** The full, unlimited-length context for the next Join-2 pass. It is launched by a short **interactive** Claude Code prompt (NOT a `/goal` — this pass has human decision/confirm gates in the middle, so it is not run-to-completion) that reads this file **in full first**. Everything load-bearing lives here; the launch prompt is the pointer and is **not** committed (lives in chat).
>
> **⚠️ What this pass IS and IS NOT.** It **is** a measurement + info-gathering pass: fragility-test two models offline, then collect a first **latency number** and a **first handful of real fills** to replace assumed values with measured ones. It is **NOT** a deployed/working strategy, **NOT** the 90-day / 30-settled-market Phase-2 gate, and **NOT** a decision to run the NSQ stack. The exit state is *"we now know how trustworthy the models are and have real latency + first fills to calibrate them,"* nothing more.
>
> **Order size — venue minimum is 5 shares.** Polymarket's CLOB rejects orders below **5 shares**, so "1 contract" is NOT a valid order. The live loop quotes at the **venue-minimum size = 5 shares per order** (`MAX_REAL_ORDERS` caps the order *count*, not the size). Tick size varies per market (0.01 vs 0.001) — confirm both min-size and tick-size for the chosen market before quoting. Verify the current min-size against py-clob-client / the market metadata at pre-flight; do not hard-code from this doc.
>
> **Money & time at risk.** Account is ~$109 total (`@jamonator`). A 5-share order is 5 × ($0–$1 price) ≈ **~$2.50 at a typical mid, ~$5 worst case**; total is bounded by the contracts inventory cap (a small multiple of 5) → realistically **a few dollars**. The absolute-minimum first test is a *single* resting 5-share order (~$2.50). Latency probes risk **$0** (unexecutable by construction). Exact caps + size + tick are printed and operator-confirmed at the C2/C3 gates before any order can fire. Time: Part A = one offline session; Part C latency probe = minutes; the quoting loop = a **short, manually-stopped** session to get first fills — not a standing deployment.
>
> **Why this pass exists.** The machinery (2a bridge + 2b/2c/2d) is already built and dry-run-proven ([[mm_engine_join2_machinery_findings]], [[mm_engine_join2a_bridge_findings]]). What is *not* done is **trusting it**. A second orchestration read flagged that the distrust localizes to **two models** — the queue-fill model and the executable-touch costing convention — and that **if either is mis-specified, the politics sign flips.** So Join 2's first-order value is **grounding those two models against reality**, which ranks **above** any MM-alpha claim as the reason to run it. This pass does that offline, then — gated — stands up the first real 1-contract run that produces the fills the models need.

---

## Mission (one line)

Run the **offline model-validation pass** for MM Join 2 (fragility-test the two suspect models), settle the **quoter-scope pre-registration** (join-the-touch vs the NSQ/inventory-aware stack), and then — behind **hard operator gates** — stand up the **first real minimum-size (5-share) quoting run** whose fills feed 2d calibration. Offline parts place no orders and commit nothing; every live boundary STOPS, prints exactly what the operator must check, and runs only on explicit "confirm".

## Read first, in order

1. Agent Bootstrap per `brain/VAULT_MAP.md` (seed/read `local_agents/codex.md`), then `brain/CODEX.md` as **law** (this is an execution-lane build — CODEX, not COWORK), then `brain/TODO.md`, then `brain/COWORK.md` § Active threads.
2. This file (the PRD).
3. `brain/handoffs/2026-06-23_mm_engine_phase01_buildplan.md` — the build-plan / status tracker (Join 1 DONE+LOCKED; Join 2 machinery built).
4. `brain/handoffs/2026-07-07_mm_join2_build_prd_reference.md` — the machinery build PRD (the guardrails, the two operator gates, the no-real-order conjunction).
5. `polymarket/research/notes/market_making/mm_engine_join2a_bridge_findings.md` — the bridge (the one swap = the fill path; telemetry parity; the 3 fixed live-path bugs).
6. `polymarket/research/notes/market_making/mm_engine_join2_machinery_findings.md` — the 2b/2c/2d machinery (latency harness, live wiring, calibration; the self-check caveats).
7. `polymarket/research/notes/market_making/mm_task5_inventory_quoter_findings.md` — the Task-5 verdict + the shipping candidate configs; **read the corrected 5.1 split** (PBO, per-path politics numbers).
8. `polymarket/research/notes/market_making/mm_join1_reconciliation_findings.md` — where the engine was last locked against real tape (the natural baseline for "does the fill model function").
9. The four files this pass audits: `polymarket/research/mm_eval/tape.py` (executable-touch costing/marking), `polymarket/research/mm_eval/cpcv.py` (group-CSCV / PBO / DSR gating), the `mm_engine` queue models (`polymarket/research/mm_engine/queue_models.py` — `OptimisticQueue` / `RiskAverseQueue` / `ProbQueue` power-law `f`), and `polymarket/research/mm_engine/latency_models.py`.

> Paths for `mm_eval/*` are the best current guess; if the modules live elsewhere, **locate them by symbol** (costing/marking, CSCV/PBO/DSR, `ProbQueue`) and report the real paths rather than assuming.

## Framing (prevents the predictable misreads)

- **"Standalone MM = no edge" is stale/overstated as of the 5.1 corrected split.** Under the corrected split unit (NegRisk **event** group), PBO is reported ~**0.00** (not ~0.5), and the politics **full stack** is positive on all measured paths (~+0.29¢ / +0.65¢ vs baseline). "Uncertifiable" means **too few independent time-blocks** at K≈11 with news-correlated (overconfident) CIs — **not** a zero point estimate. **Certifying politics is a data-collection problem**, which is exactly what the gated live run begins to solve. Do not repeat the old "no edge" headline uncritically.
- **The distrust is correct and it localizes to two models, not the strategy.** (a) the **queue-fill model** (`ProbQueue.f` + latency) — how much of a resting quote fills, given the tape; (b) the **executable-touch costing convention** in `tape.py` — costing exits/marks at the executable touch rather than mid. If either is mis-specified the sign flips. **Model validation is the point of this pass.**
- **The probe is not the quoter.** The 2b latency harness submits deliberately **unexecutable** orders (far from touch, cancelled on ack) purely to time submit→ack. That is a stopwatch, not the strategy — it is *supposed* to never fill. The **quoter** is what determines real fills; that is the Part B decision.
- **Offline first, live gated.** Parts A and B are pure offline analysis — no venue, no orders, nothing committed. Part C touches the real venue and is fenced by hard STOP-and-confirm gates and the existing `RealOrderGate` conjunction. The real safety net is the **code path** (real venue ∧ raised `MAX_REAL_ORDERS` ∧ per-order confirm), never the prompt text — preserve it, never weaken it.

---

## PART A — Offline model verification & fragility (no gates, no orders)

**Goal:** answer "how fragile is the politics sign to the two suspect models?" with numbers, before a dollar is spent.

### A1 — Inventory & plain-English report (read, don't change)

Read and report, in plain English, each of the following — what it actually computes, its key assumptions, and where it could be wrong:

- **`tape.py` costing/marking:** how are entries, exits, and marks priced? Confirm whether it costs at the **executable touch** (crossing the spread) vs mid, how fees/rebates enter, and whether any step marks-to-mid anywhere (the historical bias source). State the exact convention.
- **`cpcv.py` gating:** how the split unit (NegRisk **event** group), purge/embargo, group-CSCV, **PBO**, and **DSR** are computed. Confirm the corrected 5.1 split is what produces PBO≈0.00, and state exactly how many independent time-blocks (K) politics has and why the CIs are called news-correlated / overconfident.
- **`mm_engine` queue models:** the mechanics of `OptimisticQueue` / `RiskAverseQueue` / `ProbQueue` (the power-law `f`), the provable bracket ordering, and how `ProbQueue.f` is currently set (the machinery entry references a synthetic-recovery fit — report the current value/source).
- **Join-1 lock:** from `mm_join1_reconciliation_findings`, restate the sense in which the engine is "locked to real tape" (record→replay 0% gap, bracket holds, 0 `queue_ahead>depth`) — this is the baseline meaning of "the fill model functions structurally." Be explicit that Join 1 proves **consistency/determinism**, not real-fill realism (that's what Part C begins).

### A2 — Baseline-luck robustness audit

Re-gate the politics result across **resampled / alternate OOS splits** (e.g. alternate group-CSCV partitions, block-bootstrap over event groups, leave-one-event-group-out). Report how the politics point estimate and CI move, and whether the "beats baseline on all paths" property survives alternative splits or is an artifact of one partition. Lookahead-free; CI, not point estimates.

### A3 — Latency sensitivity sweep

Sweep the latency constant (0 ms → a plausible politics upper bound; the machinery notes 0 ms flatters a razor-thin result). Report the politics point estimate + CI **as a function of latency**, and identify the latency at which the sign / certification flips. This directly tells the operator how much the eventual **measured** latency (Part C, 2b) matters.

### A4 — Costing-convention sensitivity

Re-run the politics result under alternative costing conventions (executable-touch as-is vs a spread-fraction vs mid, plus a fee/rebate on/off variant). Report which conventions preserve the sign and which flip it. This isolates how much the verdict rides on the `tape.py` convention specifically.

### A-OUTPUT — Fragility report

A single findings note (`polymarket/research/notes/market_making/mm_join2_model_fragility_findings.md`, hub `[[strat_market_making]] · [[COWORK]]`, opening `## Plain-English Summary`) that states: **under which model perturbations does the politics sign flip, and under which does it hold.** End with a one-line verdict: *are the two models trustworthy enough to justify spending real money to calibrate them?* Artifacts (CSVs) under `data/analysis/csv_outputs/market_making/`, gitignored/regenerable.

---

## PART B — Confirm the measurement instrument (offline, short)

This pass is **not** choosing a strategy, so this is deliberately small. The first live run uses the **naive join-the-touch quoter** (join best bid / best ask, no skew, no inventory logic — already built + dry-run-proven) purely as the **measurement instrument**: it is the simplest thing that produces real fills, which keeps strategy behavior from confounding the "does the fill model predict reality" read.

Do only: (1) confirm join-the-touch is what the C5 loop will run and that no config change is needed to use it; (2) in **two or three sentences** in the fragility findings note, record that the NSQ / inventory-aware Task-5 stack (microprice skew + one-sided cap + near-expiry flatten + toxicity gate, politics config `[k=5e-6, cap=500, pull=2h, tox=velocity]`) is the **later** alpha-certification candidate and is explicitly **out of scope** here. No separate memo, no NSQ wiring, no strategy pre-registration.

---

## PART C — First real 1-contract run (LIVE — HARD GATES)

Only after Part A's verdict and the operator's explicit go-ahead to proceed to live. **Every live boundary below is a hard ⛔ STOP:** print the exact thing the operator must check/do, then wait for an explicit "confirm" before running the step. No step self-proceeds.

- **⛔ C1 — Pre-flight.** Run the authed **read-only CLOB balance probe** (`polymarket/execution/tests/probes/mm_join2_balance_probe.py`, secret-scrubbed) + `--check-auth`. Print the authenticated balance and open positions. STOP: operator confirms funder == `@jamonator`, ~$109 present, positions clean, caps tiny (5-share order size = venue minimum). (Public data-api shows $0 for custodied cash — expected; the authed read is the real check.)
- **⛔ C2 — Env↔account + Polymarket-page verification.** Print `POLYMARKET_FUNDER`, chain id, signature type, and the caps (secrets SET-only, never echoed). STOP: operator verifies on the Polymarket UI the chosen market's tradability and any on-chain USDC allowance/approvals. Print exactly what to check.
- **⛔ C3 — Market pick.** Run the 5-screen politics-NegRisk selection; present the top candidates. For the chosen market, print its **tick size** and **minimum order size** (from py-clob-client / market metadata) and the resulting 5-share notional at the current touch. STOP: operator eyeballs the market on the site (the bucket classifier is keyword-based) and confirms the condition id, tick, and size.
- **⛔ C4 — Latency measurement (2b).** Run `--mode mm_latency` (unexecutable probes, gated exactly like quotes, refuses any book within a tick of 0/1). Print the fitted `POLYMARKET_MM_BRIDGE_LATENCY_MS`. STOP: operator confirms before it is consumed. (Note the known unknown: SELL probes may be rejected without token inventory; BUY probes carry no such constraint.)
- **⛔ C5 — The 1-contract loop.** Run `--mode mm_bridge` with the **join-the-touch** quoter, **order size = 5 shares (venue minimum)**, `POLYMARKET_VENUE=real`, `REQUIRE_OPERATOR_CONFIRM=true`, `MAX_REAL_ORDERS` starting at **1** (one resting 5-share order for the absolute-minimum first fill; operator may raise it modestly for a two-sided / re-quote session), the hard contracts inventory cap (≥5), the measured latency, and a rate-limit-safe `POLYMARKET_MM_BRIDGE_RECONCILE_EVERY`. Per-order confirm on **every** submit. Print the pre-registered stop conditions and the first-fill mini-audit hook from the runbook. Add `websocket-client` to the run venv for the live feed.
- **C6 — Calibrate on the first real fills (2d).** Feed real fills through `QueueModel.calibrate()` → fit `ProbQueue.f` + the latency constant → emit the **bracket-collapse report** (Optimistic/RiskAverse bounds → toward the live-measured rate). Cross-check the live-measured fill/latency against Part A's fragility bands: **did reality land where the models assumed?** This is the actual answer to the distrust.

The pre-registered Phase-2 gates ([[mm_politics_negrisk_live_loop_design]]) still govern any scale decision: fill share >0% in ≥5 markets; 60s adverse-selection drift lower CI > −500 bps; news-proximate <50%; net-of-cost lower CI >0 over ≥30 settled markets; resolution drag <10%. This pass only *begins* that sample.

---

## GUARDRAILS (never break)

- **No real order without real venue ∧ raised `MAX_REAL_ORDERS` ∧ per-order operator confirm.** Preserve the `RealOrderGate` conjunction; never weaken it. Parts A/B place nothing.
- **Never print/log secrets** — funder address + caps only; keys/secret/passphrase confirmed SET, never echoed.
- **Do NOT modify frozen files:** `interfaces.py`, `queue_models.py`, `latency_models.py`, `fills.py`, `engine.py`, `_kernel`. Part A **reads** them; any needed analysis runs in new scratch/eval scripts, not by editing frozen code.
- **Commit nothing.** Leave findings notes + artifacts staged for operator review on the operator's branch. The `/goal` is not committed.
- **Repo invariants:** run from the correct venv (`PYTHONPATH=. uv run …` in `polymarket/research/` for eval; the execution venv for the bridge), deterministic/seeded, lookahead-free, CI not point estimates, append-only Parquet.

## SUCCESS (checkable, per part)

- **A:** the fragility report exists with the sign-flip map across A2/A3/A4 and a trustworthiness verdict; `mm_engine` + `mm_eval` suites green.
- **B:** join-the-touch confirmed as the C5 instrument (no config change needed); NSQ recorded as out-of-scope in 2–3 sentences. No NSQ wiring.
- **C:** each gate recorded with the operator's confirmation; the 1-contract loop ran under per-order confirm with zero unintended orders; a first latency number + first real fills captured; the bracket-collapse report compares live-measured vs Part-A-assumed. **Not** a working strategy, **not** the Phase-2 sample.

## Self-QA + adversarial review (exit criteria)

1. Run the full execution + `mm_engine` + `mm_eval` suites; report green/red.
2. For any code touched on the live-order path, **spawn an adversarial opus review** targeted at that path (unintended real-order path, secret leak, safety-gate bypass, accounting/dedup error) before declaring done. Fix confirmed bugs, add regression tests, report findings.
3. On return: plain-English "what I did", an adversarial self-check ("where would this be wrong?"), then the deliverables — before any "it works". Nothing committed; no real orders outside the gated, operator-confirmed C5.
