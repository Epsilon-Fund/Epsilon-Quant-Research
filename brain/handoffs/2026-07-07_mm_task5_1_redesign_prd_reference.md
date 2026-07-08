---
title: "MM Task 5.1 — PRD reference (characteristic-cohort split + neutral spike-avoidance controller)"
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

# MM Task 5.1 — PRD reference (redesign)

> **What this file is.** The full context for the Task 5 **redesign**. The build is launched by a ≤4000-char `/goal` (Claude Code) that reads this first. This supersedes the eval methodology and the inventory controller of Task 5 (`2026-07-07_mm_task5_prd_reference.md`); the frozen engine, costed-PnL machinery, ladder discipline, and bracketing are **kept**.
>
> **Why a redesign, not a diagnostic.** Task 5 shipped "damage control, no edge," but the audit found the verdict rested on a **broken IS/OOS split** and a controller (near-expiry *flatten*) that answers the wrong question. Diagnosing the old split learns nothing new. We fix the measurement and the controller together, then re-run the ladder.

---

## 0. What Task 5 got wrong (the two defects this redesign fixes)

**Defect 1 — the IS/OOS split is a single calendar cut through the same markets.** `SPLIT_TS_MS = 2026-06-24 00:00 UTC`: everything a market traded before is IS, after is OOS — *the same markets on both sides*. Because these markets march toward late-June/July resolutions, **IS = calm mid-life and OOS = the pre-resolution endgame** (the most toxic period, per v0's −1.01¢ near-expiry read). Evidence it's a regime confound and not knob-overfit: in `mm_task5_is_selection.csv` the IS→OOS drop is **near-uniform across all configs and scales with cap** (politics cap=200/500 barely move OOS; cap=1000/inf and the baseline blow out) — an overfit signature would punish the *selected* config harder than its siblings; instead the whole surface shifts down together and the drop scales with how much inventory a config carries into the endgame. So "IS good / OOS bad" ≈ **edge exists in mid-life, is eaten in the endgame, and the split happens to be the endgame.** This does not test generalization.

**Defect 2 — the controller *flattens*; that is not what we want.** `pull_hours` exits inventory near expiry (pays spread); `pull=off` carries blindly. The real objective (per Justin + Alvaro's LOTECH study) is to **carry balanced/small inventory to resolution while refusing to stack a one-sided book during spikes/toxic flow** — provide two-sided liquidity, and when flow turns informed, *stop feeding the exposed side* rather than dump. Note: Task-5 politics never ran `pull=off`, so flatten was never even A/B'd.

---

## 1. Mission (one line)

Rebuild Task 5's **evaluation methodology** (characteristic-cohort split over whole markets, all R2 data) **and** its **inventory controller** (a neutral spike-avoidance quoter grounded in the LOTECH study), then re-run the OOS-gated ladder and report where edge actually lives.

## 2. Read first, in order

1. Agent Bootstrap per `brain/VAULT_MAP.md` (`local_agents/codex.md`, then `brain/CODEX.md` as **law**, `brain/TODO.md`, `brain/COWORK.md` § Active threads).
2. This file.
3. Task-5 lineage: `brain/handoffs/2026-07-07_mm_task5_prd_reference.md` + `..._mm_task5a_design_readjustment.md`; the Task-5 findings `polymarket/research/notes/market_making/mm_task5_inventory_quoter_findings.md`; and the artifacts `data/analysis/csv_outputs/market_making/mm_task5_*.csv` (esp. `is_selection` — the drop-scales-with-cap evidence).
4. **The LOTECH study** — `LOTECH Report - Alvaro Fernandez.pdf` (Justin will provide; ask if absent). It is an *example* of the target research, not gospel — treat as one grounding, corroborate with general institutional/academic method.
5. Engine/eval (frozen unless noted): `polymarket/research/mm_engine/` (`interfaces.py` = DO NOT MODIFY; `strategies.py`), `polymarket/research/mm_eval/` (`protocol.py`, `runner.py`, `metrics.py`).

## 3. Plain-English: what "IS optimization" is (carry this into the doc)

The controller has a few dials; "optimising IS" = try many dial-settings, score each by *cents per contract netted in the IS window*, keep the argmax, then judge it on OOS. Dials ↔ equations:

- **k (skew slope)** in the quote-center `r = microprice − k·q` (`microprice` = liquidity-weighted fair value; `q` = inventory). Long → shift quotes down to sell more/buy less, pulling inventory to 0.
- **inv_cap** — hard limit on `|q|`; at cap, quote one side only.
- **(old) pull_hours** — near-expiry flatten trigger. *Replaced this redesign.*
- **toxicity subset** — which skip-signals are on.
- **γ (A-S rungs)** in `r = mid − q·γσ²τ`, `spread ≈ γσ²τ + (2/γ)ln(1+γ/k_arr)`.

**Objective maximised on IS:** pooled per-contract costed net ¢ = (Σ realized PnL + inventory marked at the executable touch) ÷ Σ contracts × 100, pessimistic (RiskAverse) queue. It is a bare `argmax`, greedy phase-by-phase — **no CI, no grouping.** That is the *selection* step. The **gate** is separate and on OOS: a group-clustered bootstrap CI on the OOS delta vs the previous kept config, plus PBO/DSR. Task-5's failure is precisely that seam: selection found calm-regime winners; the gate was the storm.

## 4. Fix 1 — the IS/OOS methodology (characteristic cohorts, whole-market holdout)

**Split unit = the whole market** (NegRisk event group for politics; the match/condition for esports). A market is entirely in IS or entirely in OOS — its **full τ arc (mid-life + endgame) stays together**. This kills the "same market on both sides / OOS = endgame" confound: OOS now contains the *complete lifecycles of unseen markets*.

**Cohort by structural characteristics that travel across markets** (Justin-confirmed set + additions):

- **Liquidity / depth** — book depth and flow volume (drives fill rate + adverse-selection likelihood).
- **Queue speed / fill dynamics** — queue turnover (how realized fills relate to the queue models).
- **Participant aggressiveness** — sweep propensity: how far through the book flow executes, how fast/hard it responds near a move (see LOTECH Lens 1).
- **Distance-to-resolution regime (τ)** — a **conditioning dimension, not the split axis**, so mid-life vs endgame is represented on both sides.
- *Additions to consider:* realized mid-vol / spike propensity; spread level; category / resolution-clarity (politics bucket type per `mm_politics_negrisk_live_loop_design`).

**Leakage-safe cohort features (fixes the subtlest trap).** Cohort/characteristic features must be computed **only from a pre-registered lead-in window** of each market (e.g. its first N hours of book/flow), never from the evaluation window — otherwise the fold assignment itself peeks at the OOS endgame (a market gets labelled "toxic" *using* the toxicity we're trying to predict OOS). Features are structural (liquidity/depth, queue turnover, early sweep/aggressiveness, spread), fixed before scoring.

**Cross-validation: nested Combinatorial Purged CV (CPCV) over whole markets** — this is the load-bearing methodology fix, and it addresses two defects the Task-5 single split had:

- **Combinatorial + purged (López de Prado):** partition the whole-market groups into K folds; test on every C(K, k) combination of held-out groups with **purge (markout horizon) + embargo** at the boundaries. This yields *many* backtest paths from few groups — the power Task-5 lacked (7 daily obs / 6 groups) — and the path distribution is exactly what feeds **PBO** and the **Deflated Sharpe Ratio** honestly. CPCV is documented to lower PBO and raise DSR vs walk-forward/k-fold.
- **Nested (inner-select / outer-estimate):** knobs are selected in an **inner** CV loop **on the outer-fold training groups only**; the chosen config is scored **once** on the outer held-out groups. This is the fix for Task-5's selection seam — non-nested selection biases the generalization estimate optimistically because the same data tunes and judges.
- Balance the group→fold assignment on the leakage-safe cohort features so folds are characteristic-comparable, and **report a (cohort × τ-regime) performance surface**, not one pooled number, so you can *see* where edge lives (calm mid-life) vs dies (spike/endgame).
- **Session-aware baselines** (the z-score below) calibrated per market on its own lead-in/calm window — no fixed thresholds leaking across folds (retires Task-5's fixed-threshold caveat).

**Data:** pull the **entire** R2 sample `r2:epsilon-polymarket-data/parquet` — including the ~7 extra days now available beyond 06-30 — and **verify coverage by code** (print date range + market count per category before running). More markets = more units for the cohort CV = the power Task-5 lacked (its DSR had 7 daily obs, PBO 6–7 groups).

## 4b. REUSE the packaged overfitting/CPCV infra — do NOT rebuild it

The repo already ships a validated overfitting harness and a CPCV split generator. Reuse them; `mm_eval/protocol.py` **already** imports the first for DSR (via `_load_audit()`).

- **`infrastructure/validation/overfitting_audit.py` — reuse wholesale** (engine-agnostic; every fn takes plain returns arrays/matrices). Use `deflated_sharpe_ratio` (Bailey–López de Prado, skew/kurtosis-corrected), `pbo_cscv` (Bailey-Borwein-LdP-Zhu CSCV — this is the real PBO; Task-5 hand-rolled a `group_cscv_pbo` approximation), `whites_reality_check` (White 2000 data-snooping — add it), `effective_n_trials` (deflate by *effective* not raw trial count — fixes Task-5's crude "40 trials"), `build_trial_returns_matrix`, `run_overfitting_audit` + `OverfittingVerdict.to_markdown()/.plot()`, and `make_null_ohlcv`/`NullMCResult` for a synthetic-null MC gate. Feed it the per-config **per-group** return series so the CSCV blocks are event-groups (not time) — matching the whole-market split.
- **`infrastructure/walkforward/cpcv_engine.py::generate_cpcv_splits(n_bars, N, k, purge_bars)` — PORT the pure function, do not import the module.** It correctly encodes the combinatorial train/test enumeration + purge, but the module imports the crypto backtester (`optuna`, `engine`, `wf_engine`) at load — importing it drags crypto **strategy** code into the PM venv, violating the separate-venv / never-cross-import invariant. Lift the ~80-line generator into `mm_eval`, adapt the unit from bars → **event groups** (embargo in group/time terms).
- **Do NOT reuse `run_cpcv` / `cpcv_portfolio.py`** — bound to the crypto backtester + per-asset return pkls. `mm_eval/runner.py` produces the per-config/per-group return matrices; those feed the audit functions above.
- Boundary: `overfitting_audit` is sanctioned shared *validation* infra (already imported via `sys.path` insert of repo root); the crypto *backtester/strategy* code is not — never import it.

## 5. Fix 2 — the controller: a neutral spike-avoidance quoter (LOTECH-grounded)

New strategy (suggested `NeutralSpikeQuoter`, alongside — not replacing — the frozen protocol). Keep what worked, replace flatten with graduated neutrality. Components:

1. **Microprice skew (kept).** `r = microprice − k·q`. LOTECH independently validates microprice as fair value (MAE 0.75 tick vs mid 8.82 — ~12× closer, holds through spike/post-sweep). Report raw-mid vs microprice markout diff as before.
2. **Two-lens toxicity detector (replaces the ad-hoc velocity/imbalance/depth trio), from LOTECH §5:**
   - **Lens 1 — order-flow toxicity (leading):** ground this in **VPIN** (volume-synchronized probability of informed trading; Easley–López de Prado–O'Hara) — the established order-flow-toxicity / adverse-selection early-warning: slice flow on a **volume clock** into equal-volume buckets and score the persistent buy/sell imbalance per bucket. Enhance with the LOTECH sweep-distance weighting (weight trades by how far past the touch they execute). `directional_flag` when toxicity is elevated (VPIN literature ≈ 0.4 elevated / 0.6 high, used **session-relative**, not as hard constants); net buy vs sell gives the exposed side.
   - **Lens 2 — adverse-selection z-score (confirming):** on each maker fill, measure post-fill mid drift at horizon T; sign it (adverse = negative); rolling calm baseline mean/std; `as_z = (signed_drift − mean)/std`; `as_flag = as_z < −2`. **Session-aware** → no per-instrument hand-tuning (this is the fix for leaking fixed thresholds).
3. **Graduated response (the anti-"sell-sell-sell" core, LOTECH combined trigger):**
   - `directional_flag` only → **suspend adding to the exposed side** (stop one-sided stacking); keep quoting the other side; monitor `as_z`.
   - `as_flag` only → **reduce size** on deep layers; monitor direction.
   - **both** → widen/pull deep layers (defensive), alert.
   - This *prevents* the one-sided book rather than dumping it — carry the (now balanced/small) inventory.
4. **Asymmetric repricing (LOTECH idea 1):** deep quotes reprice **slower to chase price away, faster to withdraw when price comes toward them** — chasing a move with a passive quote is the adverse-selection trap; the fill in a reversing market is the one to exit fast.
5. **Continuous imbalance size-dampening (LOTECH idea 2), keyed on OFI:** feed **order-flow imbalance** (OFI; Cont et al. — the dominant driver of short-horizon price moves, and the same information microprice already tilts on) into size so exposure shrinks *continuously* during one-sided flow, complementing (not replacing) a cap.
6. **Carry-to-resolution + passive unwind, never flatten-dump (LOTECH §4):** post-spike, halt new quoting on the exposed side and unwind **passively in measured tranches, letting the book come to you** — damage limitation, not chasing. Balanced/small inventory may ride to resolution.

**Knob discipline:** keep TUNED knobs small — skew `k`, size-dampening coefficient, VPIN bucket/window + elevated-band. The AS z-threshold (−2) and calm-baseline are *statistical/standard*, not tuned (keeps the knob count + leakage down). Build the components **separable/toggleable** so the ladder can ablate and attribute each (Lens 1 alone, Lens 2 alone, both, +asymmetry, +size-dampen).

## 5b. Methodology references (desk research, 2026-07-07)

- **CPCV / purging / embargo / PBO / DSR** — López de Prado, *Advances in Financial Machine Learning*; CPCV shown to lower PBO and raise DSR vs walk-forward/k-fold. [purged CV overview](https://en.wikipedia.org/wiki/Purged_cross-validation) · [backtest-overfitting method comparison](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)
- **Nested CV (inner-select / outer-estimate)** — separates hyperparameter selection from generalization estimation; non-nested selection is optimistically biased. [scikit-learn: nested vs non-nested](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) · [Raschka, model-eval survey](https://arxiv.org/pdf/1811.12808)
- **VPIN (order-flow toxicity)** — Easley, López de Prado & O'Hara; volume-clock buy/sell imbalance as adverse-selection early-warning (~0.4 elevated / ~0.6 high). [VPIN paper](https://www.quantresearch.org/VPIN.pdf) · [overview](https://www.visualhft.com/post/volume-synchronized-probability-of-informed-trading-vpin)
- **OFI (order-flow imbalance)** — Cont et al.; net best-bid/ask order-flow drives short-horizon price change; state variable for MM/execution. [OFI overview](https://www.emergentmind.com/topics/order-flow-imbalance)

## 6. The ladder (kept discipline, new rungs)

Per category, gated on the cohort-CV OOS (group-cluster CI lower bound > 0 vs previous kept config; PBO over cohort-folds; DSR deflated by all configs). Never jump rungs.

`baseline (symmetric)` → `NeutralSpikeQuoter` core (microprice + skew + cap) → `+ two-lens gate` → `+ asymmetric repricing` → `+ imbalance size-dampen` → `A-S rungs 1–3` → `basket-carry (v2 alternative)`. Ship the best OOS-surviving config **per cohort × regime**. Every number bracketed across {Optimistic, Prob, RiskAverse}.

## 7. v0 (re-grounded)

Recompute the failure attribution + near-expiry/flow-regime read **under the new whole-market cohort split** and with session-aware calm baselines. Confirm (or overturn) that politics near-expiry is net-negative and esports "near expiry" (in-play) is benign. This grounds where the two-lens gate and size-dampening should bite. Pre-registered: if a cohort×regime shows no net-negative toxicity, the defensive knobs are withheld there and reported unsupported.

## 8. SUCCESS

- A **(cohort × τ-regime) performance surface** per category showing where the controller nets positive vs negative — the thing Task-5's single pooled OOS number hid.
- The **ladder/comparison table** (baseline → controller ablations → A-S → basket), OOS-gated on the new cohort CV, bracketed, with PBO (cohort-folds) + DSR.
- An honest verdict per cohort×regime, and the shipped config(s). **No profitability claim until Join 2.**
- Charts (see §10) that let a human audit the split and the diagnosis visually.

## 9. GUARDRAILS

`interfaces.py` frozen (τ + all knobs via `params`). Lookahead-free, non-overlapping, deterministic/seeded. **Split unit = whole market; τ conditions results, never splits them.** Session-aware baselines (no fixed-threshold leakage). Select on IS folds, report on OOS folds; never jump rungs. Run from `polymarket/research/` with `PYTHONPATH=. uv run`; DuckDB over Parquet. Pull + code-verify the FULL R2 sample. fee=0, no rebate. Bracket every number. Branch `justin` → merge `main`.

## 10. Charts to produce (for the findings + the human audit)

1. **Split diagnosis:** market lifecycle timeline showing the old calendar cut vs the new whole-market cohort holdout (why OOS was all endgame).
2. **(cohort × τ-regime) performance heatmap** — per-contract costed net, so edge-location is visible.
3. **IS→OOS scatter colored by cap** — reproduce the "uniform drop, scales with cap" evidence on the new split.
4. **Toxicity trace** — sweep-score (Lens 1) + AS z-score (Lens 2) through a spike, with the graduated-response firing points (LOTECH-style).
5. **Inventory path** — one-sided stacking (baseline) vs neutral controller, into a spike.

## 11. On return

Plain-English explanation + adversarial self-check (where would this be wrong?) before "it works", then the findings doc and the charts.
