---
title: "MM Task 5 — PRD reference (5a InventoryAwareQuoter + 5b costed eval)"
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
---

# MM Task 5 — PRD reference

> **What this file is.** The full, unlimited-length context for the Task 5 build. The actual work is launched by a `/goal` prompt (Claude Code, ≤4000 chars) that reads this file first. Everything load-bearing lives here; the `/goal` is the short pointer. Authored by the Cowork planning chat via the `prd-scaffold` skill from `brain/handoffs/2026-07-07_mm_task5_prd_goal_handoff.md` + `..._mm_task5a_design_readjustment.md` (the readjustment supersedes §4 leans of the main handoff — read it if any conflict).
>
> **Role split.** You (the implementation agent) build. Design is closed — do not reopen the five decisions below; if something is genuinely underspecified, make the smallest reversible choice and flag it in the findings doc.

---

## Mission (one line)

Build **Task 5** — the first inventory-managed market-making strategy — and evaluate it honestly against the Task-4 symmetric baseline with costs and the overfitting apparatus now live. Two sub-tasks, one goal: **5a** = `InventoryAwareQuoter`; **5b** = costed A/B evaluation.

## Read first, in order

1. Agent Bootstrap per `brain/VAULT_MAP.md` (seed/read `local_agents/codex.md`, then `brain/CODEX.md` as **law**, then `brain/TODO.md`, then `brain/COWORK.md` § Active threads).
2. This file (the PRD).
3. `brain/handoffs/2026-07-07_mm_task5_prd_goal_handoff.md` and `brain/handoffs/2026-07-07_mm_task5a_design_readjustment.md` — the domain source ([[2026-07-07_mm_task5_prd_goal_handoff]], [[2026-07-07_mm_task5a_design_readjustment]]).
4. Task-4 characterization: `polymarket/research/notes/market_making/mm_symmetric_quoter_validation_findings.md`.
5. The frozen engine: `polymarket/research/mm_engine/interfaces.py` (do NOT modify), then `queue_models.py`, `latency_models.py`, `fills.py`; and the eval side `polymarket/research/mm_eval/` (`runner.py`, `metrics.py`, `ab.py`, `overfitting_hook.py`, `stability.py`).

## Framing (prevents the predictable misreads)

- Task-4's symmetric quoter's naive PnL is a **directional inventory bet**, not a tradeable maker edge. 5a's whole point is inventory control; 5b's whole point is to price the carry/exit that Task-4 ignored. **Do not report mark-to-mid PnL as edge.**
- Task 4 ran **latency-naive** (`ConstantLatency(0.0)`) to isolate the queue gate. The overfitting apparatus (DSR/CPCV/PBO) was **dormant** on a 0-parameter quoter — it "wakes up" now because 5a introduces parameters.
- **No profitability claim until Join 2** (live 1-contract calibration). Every backtest number is a **bracket across {Optimistic, Prob, RiskAverse}** queue models — never a point estimate.

## GOALS (observable)

1. `InventoryAwareQuoter` in `mm_engine/strategies.py` implementing the confirmed v1 design (§ Confirmed decisions), conforming to the frozen `Strategy.quote(book, inventory, params) -> list[Order]` protocol.
2. τ (time-to-resolution) injected via `params` by the eval runner (see § Interface subtlety) — **no interface change**.
3. Eval/runner wiring for costed, bracketed A/B vs the Task-4 symmetric baseline on the same tokens, with the IS/OOS protocol (§ below) and DSR/CPCV/PBO live.
4. Execute the full **comparison arc** — symmetric baseline → v1 → A-S rungs 1–3 → basket-carry alternative — each on the *same* IS/OOS protocol, and ship the best OOS-surviving config.
5. A findings doc `polymarket/research/notes/market_making/mm_task5_inventory_quoter_findings.md` (plain-English summary first; define every term/table; bracket every number; include the comparison/ladder table).

## NON-GOALS (explicit exclusions)

- **Building any config *ahead of its OOS gate*.** The ladder is executed, but no rung/architecture is *kept or shipped* unless it beats the previous config OOS (see § IS/OOS protocol). "Execute the ladder" ≠ "ship every rung."
- **Jumping ladder rungs.** Rung 2 is not attempted before rung 1 has been evaluated; each rung earns its parameters against the DSR/CPCV budget or is dropped.
- **A standalone Step-0 task** — Step-0 is folded in as **v0** (the gate), not a separate deliverable.
- **Any interface change to `BookState`/`interfaces.py`** — frozen and reconciled.
- **Profitability claims / live trading** — measurement only until Join 2.

## PHASING (validate before build — each stage GATES the next; a later config is *kept* only if it beats the previous config OOS on the bracketed net-edge lower-CI)

- **v0 (the gate — folded Step-0, runs first, cheap, data-only, no strategy code):** (a) attribute *why* the Task-4 tokens failed (politics 4/12, esports 7/12); (b) characterize the **NegRisk near-expiry regime** (markets mean-revert mid-life, turn toxic near expiry — Task 4 pooled across each token's whole life so this regime may be under-observed). **Pre-registered read:** confirm the near-expiry regime is net-negative before wiring the τ-flatten + toxicity gate. If it is NOT net-negative on the sample, say so and treat decisions 2 & 4 as unsupported (do not silently ship them). This grounds the toxicity signal set and the near-expiry pull.
- **v1 (ships iff it beats the symmetric baseline OOS):** `InventoryAwareQuoter` per the confirmed decisions, ≤4 knobs (below), then 5b costed A/B.
- **v2 (executed AND compared under the *same* IS/OOS protocol — not deferred):**
  - **A-S ladder** rung 1 (`γσ²τ`) → rung 2 (A-S spread) → rung 3 (adverse-selection overlay). Never jump rungs; each kept only if it beats the previous rung OOS.
  - **Basket-balanced carry + redemption floor** — the architecture **alternative** to v1's per-token flatten (decision 5, stance B); compared head-to-head with v1 on the same OOS, not stacked on top of it.
  - **Ship = the single best OOS-surviving config**, chosen by the OOS bracket + PBO/DSR, never best-in-sample. The deliverable includes the full comparison/ladder table (§ SUCCESS).

## SUCCESS (checkable, per phase)

- **v0:** a documented failure-attribution + near-expiry-regime read, with the pre-registered net-negative check answered yes/no.
- **v1 / 5a:** a running `InventoryAwareQuoter` that skews around the microprice, respects the tight cap (one-sided at cap), flattens near expiry, and applies the (v0-grounded) toxicity gate. Beats the symmetric baseline OOS.
- **v1 / 5b:** realized (costed) PnL A/B vs the symmetric baseline on the same tokens, every metric a bracket across {Optimistic, Prob, RiskAverse}, with block-bootstrap CIs, the IS/OOS split, and DSR/CPCV/PBO reported. Verdict per `verdict_from_bracket` (VIABLE / FRAGILE / DEAD).
- **v2 / comparison:** the full **comparison/ladder table** — one row per config (symmetric baseline → v1 → rung 1 → rung 2 → rung 3 → basket-carry), columns = knobs · IS read · **OOS bracket** across {Optimistic, Prob, RiskAverse} · PBO/DSR · **keep/drop verdict** (kept iff it beats the previous config OOS). Plus the single **shipped config** and its `verdict_from_bracket`. **Power caveat honored:** 11 days × 2 categories is thin — rungs that read FRAGILE/underpowered are reported as such; do not fabricate significance.

## GUARDRAILS

- Frozen `mm_engine/interfaces.py` — do not modify. τ via `params` only.
- **≤3–4 knobs in v1** (budget below). Reference price = microprice = **0-param**.
- Lookahead-free, non-overlapping, deterministic/seeded. CIs before any "positive" verdict. **Select on IS, report on OOS** (§ IS/OOS protocol); never jump ladder rungs; keep a config only if it beats the previous OOS.
- Run from `polymarket/research/` with `PYTHONPATH=. uv run ...`. DuckDB over Parquet.
- Bracket every number across {Optimistic, Prob, RiskAverse}. No profitability claim until Join 2.
- Branch `justin` → merge `main` per `brain/MERGE_PROTOCOL.md`. Findings + code committed; the `/goal` prompt itself is not committed (lives in chat).

---

## Interface subtlety (the one thing to get right)

The strategy needs **time-to-resolution (τ)**, but the frozen `BookState` does not carry it. **Do not change the interface.** The eval runner knows each market's `end_date`, so it computes τ and puts it in the `params` dict already passed to `Strategy.quote(...)`. This is a runner change (`mm_eval/runner.py`), not an interface change.

## Confirmed decisions (v1) — design is CLOSED

These are the resolved positions from `..._mm_task5a_design_readjustment.md`; they supersede the "leans" in §4 of the main handoff.

**1. Skew — linear skew around the microprice (1 knob `k`).** Reservation `r = microprice − k·q`, where `q` = inventory. **Reference = microprice (weighted mid), not raw mid:** `weighted_mid = (bid·ask_size + ask·bid_size)/(bid_size+ask_size)` — 0-param, imbalance-aware, reduces adverse selection. Compute raw-mid and microprice both; quote around microprice; **report the markout difference as a free microstructure result.** Full A-S is NOT v1 — it restructures the free slope into `γσ²τ` (adds γ, σ knobs) and assumes *no adverse selection*, which our market violates. A-S is a gated build-up ladder (each rung kept only if it beats the previous OOS):

| Rung | Adds | Assumptions imported | Estimated w/o hard evidence | Gate |
|---|---|---|---|---|
| **0 (v1)** | Linear skew `r = microprice − k·q` | linear inventory response adequate; skew pulls inventory→0 | `k` (one interpretable OOS-tuned knob) | ship if beats symmetric baseline OOS |
| **1** | Replace `k` with A-S slope `γσ²τ` | mid ~ ABM, **constant σ**; well-defined τ | σ (estimator/window/regime), γ (no ground truth), τ | keep only if beats hand-tuned `k` OOS |
| **2** | A-S optimal spread `≈ γσ²τ/2 + (1/γ)ln(1+γ/k_arr)` | Poisson arrivals `λ(δ)=A·e^{−k_arr·δ}`; fills info-independent | A, k_arr (noisy, regime-dependent) | keep only if beats fixed/empirical spread OOS |
| **3** | Adverse-selection overlay (widen/pull on toxicity) | informed flow detectable ex-ante | signal thresholds/weights (highest overfit risk) | strict OOS/DSR/CPCV; ship smallest surviving subset |

Never jump rungs. Full A-S + overlay = 4–5 knobs, over the v1 budget **by design**.

**2. Time-to-resolution — flatten near expiry (adverse-selection overlay, OPPOSITE sign to the A-S τ term).** As τ→0, widen and actively pull to flatten naked inventory before the toxic near-expiry regime. **Sign warning:** in A-S skew *shrinks* as τ→0; our toxicity *grows* as τ→0. Do NOT wire the near-expiry pull and any future A-S τ term through the same τ — they fight. Flattening intentionally clips *both* esports terminal tails (lucky wins AND big reversals) — correct MM stance: the edge is the spread, not the resolution coin-flip, which is negative-EV (adverse) near expiry. The exit cost of flattening is captured in 5b's costed eval.

**3. Position cap — single TIGHT cap; one-sided quoting at cap.** Stop adding to the losing side. **No sweep in the ship config.** Separate understanding-only runs: (a) one uncapped diagnostic (characterizes natural inventory excursion → where sane caps sit); (b) a small cap sweep for cost/benefit. **You cannot post-hoc clip a loose-cap run to a tight cap** — the cap changes the fill path (a tight cap refuses fills a loose run took), so each cap value needs its own run. Any shipped cap value chosen by OOS, never best-in-sample.

**4. Toxicity gate — propose AND execute it here, AFTER Stage-0 results.** Base = velocity + book-imbalance/microprice-divergence + a conservative hardcoded near-expiry pull; Stage-0 refines which signals actually flagged the Task-4 failures. **Build signals as separable, individually-toggleable** (velocity, book imbalance/microprice divergence, τ-proximity, one-sided flow, depth evaporation) so you can ablate (gate off / each signal alone / combined) and **attribute** the adverse-selection reduction per signal — this is the microstructure research. Discipline: **measuring N signals = research (report all configs); shipping N tuned signals = overfitting.** Keep measured vs shipped sets separate; ship the smallest subset that survives OOS/DSR/CPCV.

**5. Carry vs basket — per-token carry + τ-flatten (v1); basket netting is v2.** Two coherent stances: **(A) per-token, flatten near expiry** (v1 = decisions 2+3+4; avoids needing the redemption floor) vs **(B) basket-balanced carry** (hold the complementary NegRisk basket summing ~$1, redeem $1 regardless of outcome; the redemption floor lives here) = **v2**. Basket netting is the carry-to-resolution machinery v1 deliberately skips by flattening — it is the v2 upgrade, not part of v1 τ handling.

## v1 knob budget (check ≤4)

1. skew slope `k`  2. tight cap size  3. near-expiry pull threshold/rate  4. toxicity gate (smallest OOS-surviving subset). Microprice reference = 0-param. A-S rungs 1–3 out of v1.

## IS / OOS protocol (PINNED — not left to gamble)

The split is the load-bearing methodology; get it wrong and every "OOS" number leaks. Feature-similar baskets carried forward is *half* right — but "similar features" is a **stratifier**, not the split axis, and the split must be leak-safe.

**Unit of observation = the event / NegRisk group, NOT the individual token.** Complementary legs of one NegRisk event sum to ~$1 and share a single resolution fingerprint (no basis risk between them — see `mm_negrisk_consistency_scanner_findings` / SPCX S8), so they are highly correlated. A token-level split can put a token's own complement on the other side of the IS/OOS line and leak. **All legs of an event go to the same fold.**

**Stratify within category.** politics_negrisk and esports are split and evaluated **separately** (esports is bimodal/fragile, politics thin/stable — pooling hides the regime). Cross-regime transfer (fit politics → test esports, and vice-versa) is reported as a **robustness read, not the ship gate.**

**Feature-matched baskets balance the folds, they don't define the split.** Liquidity, participant mix, and τ-regime are used to make IS and OOS baskets *comparable* (else an IS-easy / OOS-hard basket masquerades as overfitting). They are covariates for balancing, not the partition.

**Two OOS axes, both pinned:**

1. **Temporal walk-forward (carry-forward) — the primary ship gate.** Select knobs on the earlier window, evaluate on the later window (what live deployment faces). **Purge + embargo** around the boundary so a token's near-expiry rows never leak into a fold holding its own mid-life. Given the 11-day sample, use an expanding/rolling split at the event-group level.
2. **Cross-sectional held-out events — the generalization axis.** **CPCV** (combinatorial purged CV) across event-groups with purge+embargo → many backtest paths → **PBO** (probability the IS-best config is OOS-suboptimal). **DSR** deflates the Sharpe for the number of configs/rungs tried.

**Selection vs reporting:** knobs chosen on **IS folds only**; the reported verdict is the **OOS bracket** across {Optimistic, Prob, RiskAverse}. Never tune on the OOS fold. `verdict_from_bracket` is applied to the **OOS** net-edge lower-CI.

**Power caveat (state it, don't hide it):** 11 days × 2 categories is thin for CPCV; with ~5–6 configs (ladder + basket alt) DSR deflation is harsh and CPCV paths are limited. Expect some rungs to read FRAGILE/underpowered — that is the honest outcome, not a bug. **Pull the full 11-day R2 sample** (not the 2-day local slice) — it is load-bearing for statistical power.

## 5b — costed evaluation scope

- **Realized PnL** with inventory-carry + exit/liquidation costs (not mark-to-mid).
- **Breakeven-fill-rate** reappears as a real metric (was a degenerate sign-test on the costless symmetric setup).
- **A/B vs the symmetric baseline** on the same tokens; 5a must beat it OOS.
- **The full comparison ladder** (§ SUCCESS) evaluated on the *same* IS/OOS protocol — every config apples-to-apples.
- **DSR / CPCV / PBO live** (parameterized configs) per the pinned **§ IS/OOS protocol** (select on IS, report on OOS).
- Report every number as a **bracket across {Optimistic, Prob, RiskAverse}**; use the existing `metrics.py` machinery (`block_bootstrap_mean_ci`, `compute_markout`, `breakeven_read`, `verdict_from_bracket`).

## Data state

- Full **11-day** L2 sample on R2: `r2:epsilon-polymarket-data/parquet` (2026-06-19 → 06-30), markets `politics_negrisk` + `esports`.
- Local mount currently holds a 2-day slice (`l2_data/2026-06-23/`, `2026-06-24/`) — **pull the full sample from R2 for 5b.**
- **fee = 0, no rebate**, both markets (confirmed).

## On return (operator requirement)

Audit before trusting: plain-English explanation of what was built, an adversarial self-check (where would this be wrong?), then the findings doc — before any "it works."
