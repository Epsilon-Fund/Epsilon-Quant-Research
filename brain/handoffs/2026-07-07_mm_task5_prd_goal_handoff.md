---
title: "Handoff — build a PRD to run MM Task 5a + 5b as one /goal prompt"
created: 2026-07-07
status: complete
owner: justin
project: polymarket-mm
para: project
hubs:
  - strat_market_making
  - COWORK
tags:
  - handoff
  - market_making
---
# Handoff — Fresh chat: build a PRD to run Task 5a + 5b as one `/goal` prompt

**Date:** 2026-07-07
**From:** Cowork orchestration (MM engine thread)
**To:** Fresh chat (orchestration) — you will draft a PRD, then hand it to the implementation agent (Claude Code / Codex) as a single `/goal` prompt covering 5a **and** 5b.
**Repo:** `epsilon-quant-research` | **Branch discipline:** work on `justin`, merge to `main` (Justin is solo on this lane now — took over Alvaro's Models lane).

---

## 0. The `/goal` + PRD resource — READ THIS FIRST

Both jobs (co-author the PRD, then emit the `/goal` prompt at the end) are **one skill bundle**:

**`library/skills/prd-scaffold/`** (relative to repo root; private quant repo only, not yet split to the public `lemma` repo). Note it lives under `library/skills/` — **not** `library/changepoint/` or `library/calibrate/`, which are the two engine packages (different shape).

Contents:

- `SKILL.md` — the actual guide/logic: the Q&A structure, what it forces you to decide, how it emits the `/goal` prompt at the end. **Read this before drafting.**
- `README.md` — install + usage overview.
- `EXAMPLE.md` — worked dialogue → emitted `/goal` prompt → the gate firing, end to end. **Read this for the exact output format.**
- `SCRUB.md`, `LICENSE` — packaging metadata (scrub-approved, Apache-2.0).

**> ACTION: read `library/skills/prd-scaffold/SKILL.md` + `EXAMPLE.md` first, then run its Q&A over §§4–7 below to co-author the PRD and emit the `/goal` prompt.** Everything below is the domain content that goes *into* that structure.

---

## 1. What this is (the goal of the `/goal`)

Build **Task 5** — the first *real*, inventory-managed MM strategy — and evaluate it honestly. Two sub-tasks, to be run as one `/goal`:

- **5a — `InventoryAwareQuoter`**: a strategy that skews/withdraws quotes based on current inventory (and time-to-resolution), replacing the symmetric zero-parameter quoter from Task 4.
- **5b — Costed evaluation**: realized PnL with inventory-carry and exit costs, A/B vs the symmetric baseline, with the overfitting apparatus (DSR/CPCV/PBO) now *live* because 5a introduces parameters.

The `/goal` should produce: the strategy code, the eval/runner wiring, a findings doc, and pass all repo invariants (below).

---

## 2. What is already built, verified, and LOCKED (do not rebuild)

**Engine (Machine lane — DONE, Join 1 reconciled + locked):**

- `mm_engine/interfaces.py` — **FROZEN. Do not modify.** Defines `MarketEvent`, `BookState` (frozen; `bids/asks: tuple[tuple[float,float],...]`), `FillResult(qty, queue_ahead)`, and protocols `QueueModel`, `LatencyModel`, `Strategy.quote(book, inventory, params) -> list[Order]`.
- `mm_engine/queue_models.py` — 3 provably-ordered models: **Optimistic** (upper bound) ≥ **Prob** (power-law, à la hftbacktest ProbQueueModel2) ≥ **RiskAverse** (lower bound, logically tight via floor). Hardened: `forget()` wired on cancel/replace; order-invariant coincident trade/cancel netting deferred to timestamp boundary; RiskAverse floor `max(0, cancel-back)`.
- `mm_engine/latency_models.py` — `ConstantLatency(200ms default)`, `SampledLatency` (Normal(mean,std), **pure function of ts** — keyed on `(seed, int(ts))`, Phase-2 calibration stub).
- `mm_engine/fills.py` — latency gate (`placement_ts + round_trip <= trade.ts`) then queue gate.
- `mm_eval/metrics.py` — `block_bootstrap_mean_ci` (moving block, ~√n, 2000 resamples, quantity-weighted, 2.5/97.5 pct), `compute_markout` (markout-to-fill + adverse A(T)), `breakeven_read`, `verdict_from_bracket` (VIABLE = pessimistic/RA net-edge lower-CI > 0; FRAGILE = only optimistic clears; DEAD = even optimistic fails).
- `mm_eval/runner.py` — currently uses `ConstantLatency(0.0)` deliberately to isolate the queue gate; queue set = {Optimistic, Prob(0.5), RiskAverse}.

**Task 4 findings (symmetric quoter characterization — DONE):**
`polymarket/research/notes/market_making/mm_symmetric_quoter_validation_findings.md`

- Politics(negrisk): **8/12 VIABLE**; esports: **5/12**. fee = 0, **no rebate** for either market (confirmed).
- The symmetric quoter's naive PnL is essentially a **directional inventory bet** — this is exactly why 5a (inventory control) is the gating problem.
- esports is bimodal/fragile and **latency-naive** (runner used 0ms); politics is thin/stable/reliable.
- Overfitting apparatus (DSR/CPCV/PBO) was **dormant** on a 0-parameter quoter — it "wakes up" for 5a.

---

## 3. The one interface subtlety for 5a

The strategy needs **time-to-resolution (τ)**, but the frozen `BookState` does **not** carry it. **Do not change the interface.** Inject τ via the `params` dict that `runner.py` already passes to `Strategy.quote(...)` — the runner knows each market's `end_date`, so it computes τ and puts it in `params`. This is a runner change, not an interface change.

---

## 4. Task 5a — open design decisions (Justin to confirm; my leans noted)

Keep v1 to **~3–4 knobs total** (overfitting discipline). The five decisions:

1. **Skew mechanism — Avellaneda-Stoikov reservation price vs simple linear "lean" skew.**
   *Lean:* linear lean skew for v1 (fewer knobs; A-S models inventory not adverse selection, and adds γ/σ estimation). A-S can be a v2 comparison.
2. **Time-to-resolution (τ) in the quote — auto-flatten / widen near expiry?**
   *Lean:* yes — carry τ in `params`, widen/pull as τ→0 (NegRisk markets turn toxic near expiry; redemption floor logic).
3. **Position-cap tightness.**
   *Lean:* conservative — hard cap, one-sided quoting when at cap (stop adding to the losing side).
4. **Toxicity gate scope** — velocity + book-imbalance + a conservative hardcoded near-expiry pull.
   *Lean:* start now with a simple gate; refine after **Step-0** (see §5).
5. **Per-token carry+cap+near-expiry-pull (v1) vs full NegRisk-basket inventory management (v2).**
   *Lean:* per-token for v1; basket-level netting is v2.

---

## 5. Step-0 — prerequisite or part of 5a? (Justin's open question)

**Step-0** = (a) failure attribution — *why* the 4/12 politics and 7/12 esports tokens failed in Task 4 — and (b) the **NegRisk time-to-resolution regime** (markets mean-revert mid-life, turn toxic near expiry). Task 4 pooled across each token's whole life, so the near-expiry regime may be **unobserved** in the current sample.

This should **ground the 5a screen and toxicity gate**. Decision for the fresh chat to settle with Justin: run Step-0 as an explicit first stage *inside* the `/goal` (recommended — its output tunes decisions 2 & 4), or as a separate pre-task. Either way its findings feed the toxicity gate and the near-expiry pull.

---

## 6. Task 5b — costed evaluation scope

- **Realized PnL** with inventory-carry + exit/liquidation costs (not mark-to-mid; the Task-4 naive PnL is a directional bet, not tradeable edge).
- **Breakeven-fill-rate reappears** as a real metric (it was a degenerate sign-test on the costless symmetric setup).
- **A/B vs the symmetric baseline** — 5a must beat Task 4's symmetric quoter on the same tokens.
- **DSR / CPCV / PBO now live** — 5a is parameterized, so the overfitting gates activate. Include an **OOS split**.
- Report every number as a **bracket across {Optimistic, Prob, RiskAverse}** queue models — never a point estimate.

---

## 7. Standing discipline / guardrails (carry into the PRD)

- **No profitability claim** until Join 2 (live 1-contract calibration). Every backtest number is a bracketed (optimistic/pessimistic queue) range.
- **Lookahead-free, non-overlapping, deterministic/seeded.** CIs before any "positive" verdict.
- Run from `polymarket/research/` with `PYTHONPATH=. uv run ...`. DuckDB over Parquet.
- **~3–4 knobs max** in v1.
- Branch `justin` → merge `main`.

---

## 8. Data state

- Full **11-day** L2 sample on R2: `r2:epsilon-polymarket-data/parquet` (2026-06-19 → 06-30), markets `politics_negrisk` + `esports`.
- Local mount currently holds a 2-day slice (`l2_data/2026-06-23/`, `2026-06-24/`) — pull the full sample from R2 for 5b.
- fee = 0, no rebate, both markets (confirmed).

---

## 9. What the fresh chat should do, in order

1. **Ask Justin for the `/goal` resource** (§0) and confirm the five 5a decisions (§4) + Step-0 placement (§5).
2. Draft the **PRD** in the `/goal` format, covering 5a + 5b as one goal, embedding §3 (params injection), §6 (costed eval), §7 (guardrails).
3. Hand the PRD to the implementation agent as the `/goal` prompt.
4. On return: audit before trusting (Justin's standing requirement — plain-English explanation, adversarial self-check, then the findings doc).
