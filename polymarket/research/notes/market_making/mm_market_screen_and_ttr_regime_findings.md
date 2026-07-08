---
title: "Market-Making Design Inputs — the Failure-Driver Market Screen + the NegRisk Time-to-Resolution Regime (feeds Task 5)"
created: 2026-07-03
status: active
owner: justin
project: polymarket
para: project
hubs:
  - strat_market_making
  - mm_backtesting_methodology_explainer
tags:
  - market-making
  - market-selection
  - adverse-selection
  - time-to-resolution
  - negrisk
  - design-input
  - engine
---

# Market-Making Design Inputs — the Failure-Driver Market Screen + the NegRisk Time-to-Resolution Regime

> Hubs: [[strat_market_making]] · [[mm_backtesting_methodology_explainer]] · builds directly on [[mm_symmetric_quoter_validation_findings]] (Task 4 — the per-token VIABLE/DEAD verdict this consumes) · concept lineage: [[mm_concepts_and_strategy_buildup]] §Layer 4 (spike-zone avoidance), [[block_k5_findings]] (late-spike avoidance) · data limits: [[mm_clob_capture_semantics]] · definitions: [[glossary]] · [[polymarket_table_dictionary]]
> This is the **design-input analysis that precedes Task 5** (the inventory-managed strategy). It writes **no engine code** — it drives the JOIN-1-locked `mm_engine` exactly as Task 4 did (`SymmetricQuoter`, RiskAverse pessimistic queue, 0-ms latency) and adds a new analysis-layer module `polymarket/research/mm_eval/design_inputs.py` + runner `scripts/mm_design_inputs_run.py` + tests `tests/test_mm_design_inputs.py`.

## Plain-English Summary

- **Why this exists.** Task 4 told us *which* of the 24 evaluated tokens have the spread-capture precondition (politics **8/12 VIABLE**, esports **5/12**). It did **not** tell Task 5 the two things it needs to actually build a strategy: (1) **what observable feature of a market predicts that it will be accretive vs toxic** — so we can *screen new markets we haven't quoted yet* — and (2) **how the edge behaves as a market approaches resolution** — so we know whether the favorable politics read is durable or a mid-life mirage. This note answers both, on the **same ~11-day capture** and the **same tokens**.
- **What "the screen" is (Task 1).** We compare each token's Task-4 VIABLE/DEAD label against *observable* book features (half-spread, top-of-book depth, book imbalance, price extremity, trade intensity, mid volatility) and — with far more power — regress the **per-fill markout** (thousands of fills) on those features with a **token-clustered bootstrap**. Headline: **no single *token-level* observable is individually certified on 12 tokens/universe, but two separate and one myth dies.** (1) **Price extremity** — markets *parked at ~50¢* (the coin-flip / delta-gamma spike zone) are the toxic ones (esports rank-AUC 0.83; 5 of 7 dead esports tokens sit within 0.05 of 50¢). (2) **Low mid-volatility** is the only pooled feature whose CI clears chance (rank-AUC 0.21 ⇒ calm books are viable). The myth: **a wider half-spread is NOT safer** — dead politics markets have the *wider* spreads (0.58¢ vs 0.50¢), because toxic books widen. At the *fill* level (thousands of fills, real power) two signals are CI-robust: a **larger aggressor trade** and a **more imbalanced book at the moment of fill** both predict a toxic fill.
- **What "the time-to-resolution regime" is (Task 2).** For every politics fill we compute time-to-resolution from the market's Gamma `end_date`, then bucket the per-fill markout and adverse selection by time-to-expiry. Headline: **per-fill markout climbs monotonically with time-to-resolution — from −1.0¢/contract inside the last 6 h to a +0.9¢ peak at 7–30 days, settling +0.3¢ deep mid-life.** The near-expiry toxicity is anchored by a **CI-certified +1.5¢ adverse-selection spike inside 6 h** (survives clustering on the 5 markets); the within-short-dated log-TTR slope is sign-robust under the small-K-appropriate wild cluster bootstrap (pooled +0.49, market-fixed-effects +0.35 ¢/log-hr). **Near expiry is toxic; mid-life is favorable.**
- **The load-bearing caveat (do not skip).** The top-12 politics tokens are a **mix of resolution horizons**, and that mix is not incidental — it *is* the finding. The near-expiry regime is observed **only for short-dated Musk-tweet-count + one diplomatic market** that resolve *inside* the capture; the durable political outrights that carry Task-4's "politics is the safe venue" verdict (Bennett/Petro/Starmer) are observed **only ~6 months from resolution — deep mid-life.** So Task-4's favorable politics read is a **mid-life read for the durable markets**, and near-expiry toxicity for *them* is **unobserved — a live-only risk.** The strategy must **pull quotes as any market approaches resolution**, on a live-calibrated clock.
- **No profitability claim.** Everything here is bracketed (pessimistic queue, block/cluster-bootstrap CIs) and is a **precondition/regime map**, not an edge. It tells Task 5 *where* to quote and *when to pull* — not that quoting makes money.

---

## What feeds this, and the honest sample

**Inputs.** (a) The **Task-4 per-token verdict** ([[mm_symmetric_quoter_validation_findings]]): the VIABLE/DEAD label + per-contract net edge + measured adverse selection for each of the 24 tokens (12 politics-NegRisk + 12 esports). (b) The **same ~11-day R2 capture** (`~/epsilon_l2_full`, `r2:epsilon-polymarket-data/parquet`, **2026-06-19 → 06-30**) replayed through `mm_engine` to dump the raw **per-fill telemetry** (mid-at-fill, queue-ahead, aggressor trade size, inventory, realized half-spread) that the aggregated Task-4 CSVs don't carry. (c) **Gamma market metadata** (`end_date`, resolution status) for each token's condition id, pulled once and cached (`scratchpad/*_market_meta.json`) so the run is offline-reproducible.

**Queue model = RiskAverse (pessimistic).** We dump fills under the single pessimistic queue, matching the Task-4 verdict's honest lower bound. Task 4 established the per-contract markout is **near queue-invariant at the touch** (the Optimistic/RiskAverse bracket is tight), so using one queue for the per-fill work costs almost nothing in the sign structure while keeping the read conservative. The engine re-run **reproduced the Task-4 RiskAverse fill counts token-for-token** (a determinism check), so the per-fill dump is faithful. Pooled we dump **26,361 fills across 24 tokens** (politics **8,568** / esports **17,793**); the regression uses **26,256** of them — the ~105 fills within 30 s of the capture edge are right-censored (no 30 s markout) and drop out, exactly as `compute_markout` censors them everywhere else.

**Power, stated honestly (CODEX rule 1).** The **per-token** screen (Task 1a) has only **12 tokens per universe** — enough to *rank* features and read a direction, not to certify a threshold; every AUC/median-difference carries a token-resample bootstrap CI and most will be wide (and are set to NaN, not a false-certain zero-width interval, if a class ever has <4 tokens). The **per-fill** regression (Task 1b) has thousands of fills but only **24 independent markets**, so we cluster the bootstrap on the token: fill-level features (aggressor size, imbalance, inventory) are identified *within* a market and get tight bands; token-level features (spread, depth, …) are identified only *across* the 24 markets and get honestly-wide bands. We never dress a 24-cluster coefficient as if it had thousands of degrees of freedom.

> **Small-K bootstrap caveat (load-bearing for every "sign-robust" flag).** A naïve pairs cluster bootstrap *under-covers* when the number of clusters is small — Monte-Carlo false-positive rates of ~11% at K=24, ~15% at K=6 vs the nominal 5%. So we use the **wild cluster bootstrap** (Rademacher) for within-cluster effects (fill-level regressors, the log-TTR slope) — the standard small-K fix — and the **pairs** bootstrap only where it is the right tool (cluster-constant token-level features, where the wild bootstrap is itself anti-conservative). The market/token counts that matter most — the K=6 TTR gradient and the ~5-market near-expiry buckets — are still only *approximate* even under the wild bootstrap; those results are labelled **directional**, and the load-bearing conclusions (near-expiry toxicity, the durable-outright mid-life-only caveat) are stated as **live-only risks to measure**, not certified backtest edges.

---

## Task 1 — the failure-driver market screen

### What we mean by "observable feature" and why the screen matters

The Task-4 verdict is mechanical: **VIABLE iff measured adverse selection < the half-spread**. But *adverse selection is the outcome you're trying to avoid* — you only learn it **after** you've quoted and been picked off. A screen has to run on quantities you can see **before** committing quotes to a new market. So we test which pre-quote **observables** predict the post-fill outcome:

| feature (per token, observable pre-quote) | definition | why it might separate accretive from toxic |
|---|---|---|
| **half-spread** ¢ | median touch spread ÷ 2 (the cushion the maker rests behind) | more cushion should absorb more adverse selection |
| **top-of-book depth** | median of `(best_bid_size + best_ask_size)/2` from `book` snapshots (contracts) | a deep, liquid book has more uninformed flow and less pick-off |
| **book imbalance** | `(bid_sz − ask_sz)/(bid_sz + ask_sz)` at the touch — median \|·\| and its volatility | a persistently one-sided / churning book signals a trending, informed tape |
| **price extremity** | `\|avg_price − 0.5\|` | near 50¢ is the delta/gamma spike zone (Layer 4) — most toxic; extreme prices are calmer |
| **trade intensity** | trades per active hour | fast tape ⇒ more information arriving ⇒ more adverse selection |
| **mid volatility** ¢ | std of successive touch-mid *changes* (proxy for repricing churn / resolution ambiguity) | a mid that gets repriced hard co-moves with adverse selection |

`mid volatility` and `price extremity` are the two **L2-observable proxies** for the "resolution-clarity" gate the live-loop design ([[mm_politics_negrisk_live_loop_design]]) calls for; the *semantic* clarity of a resolution criterion (is "will X be confirmed" objective?) is a **metadata** gate, not an L2 quantity, and is flagged as such — it is not something this 11-day tape can measure.

### Task 1a — per-token separation (rank-AUC + median difference, bootstrapped over tokens)

**How to read `rank-AUC`.** It is the probability that a VIABLE token scores *higher* on the feature than a DEAD one (0.5 = the feature is useless for the screen; >0.5 = higher value ⇒ more likely VIABLE; <0.5 = higher value ⇒ more likely DEAD). The median-difference row is `median(VIABLE) − median(DEAD)` with a token-resample CI.

**Pooled (24 tokens), sorted by separation:**

| feature | rank-AUC (95% CI) | median VIABLE | median DEAD | median diff (95% CI) | clears chance? |
|---|---|---|---|---|---|
| **mid volatility ¢** | 0.21 [0.05, 0.42] | 0.53 | 0.66 | −0.12 [−0.40, −0.03] | **yes** — low vol ⇒ viable |
| price extremity | 0.71 [0.48, 0.92] | 0.23 | 0.09 | +0.15 [−0.04, +0.28] | nearly — far from 50¢ ⇒ viable |
| trade intensity /h | 0.32 [0.12, 0.56] | 24.9 | 191.1 | −166 [−402, +103] | no (dir: calmer ⇒ viable) |
| median \|imbalance\| | 0.59 [0.34, 0.82] | 0.68 | 0.64 | +0.03 [−0.08, +0.12] | no |
| imbalance volatility | 0.57 [0.32, 0.80] | 0.66 | 0.67 | −0.01 [−0.05, +0.08] | no |
| half-spread ¢ | 0.52 [0.31, 0.73] | 0.50 | 0.50 | 0.00 [−0.15, +0.25] | no |
| top-of-book depth (log) | 0.50 [0.26, 0.73] | 6.64 | 7.14 | −0.50 [−2.37, +2.66] | no |

**Per-universe standouts (the pooled signal is not uniform):** in **esports**, `price_extremity` is the single best separator (**AUC 0.83 [0.50, 1.00]**) — the dead tokens sit right at 50¢ (**5 of 7 within 0.05 of the coin-flip**; medians dead 0.012 vs viable 0.082). In **politics**, the directional order is `log-depth` (0.73, deeper ⇒ viable: median depth **324 vs 117 contracts**), `imbalance_vol` (0.78), and low `mid_vol` (0.23) — but **no politics feature's CI clears chance on 12 tokens.** The cross-universe myth-buster: **half-spread does not separate, and in politics the DEAD tokens have the *wider* spreads (0.58¢ vs 0.50¢)** — toxic books widen, so "quote the wide markets" is exactly backwards.

![Observable-feature separation of VIABLE vs DEAD](../../data/analysis/plots/market_making/mm_design_screen_auc.png)

*Read this chart:* each row is one observable feature; the dot is its rank-AUC over the pooled 24 tokens with a token-resample 95% CI; the dashed line at 0.5 is "no separation." A dot far from 0.5 with a CI that clears 0.5 is a feature the screen can lean on. **The only one that clears is low mid-volatility (bottom row, CI entirely below 0.5); price-extremity (top row) is the strongest positive-direction feature but its CI grazes 0.5.** *Read:* on 12 tokens/universe **nothing is certified**, but the only pooled feature whose CI excludes chance is **low mid-volatility** (calm, rarely-repriced books are viable), and the strongest *directional* feature — decisive in esports — is **price extremity**: the toxic markets are the ones **parked at ~50¢**, the delta/gamma coin-flip spike zone the whole strat is built to avoid ([[mm_concepts_and_strategy_buildup]] Layer 4). Two intuitions are refuted: wider spread is not safer, and top-of-book depth alone does not save you.

### Task 1b — per-fill markout regression (token-clustered bootstrap)

This is the higher-power view: regress each fill's **markout(30s)** (cents/contract; >0 = accretive) on the observables, standardizing every feature so the coefficient reads as **"cents/contract per +1 SD of the feature."** Inference is **cluster-bootstrapped on the token**, with the resampling scheme matched to the regressor type (see the small-K note below): fill-level regressors (which vary *within* a market) use the **wild cluster bootstrap** — the small-K-appropriate method; token-level regressors (constant within a market) use the **pairs cluster bootstrap** — which correctly reflects their between-market sampling variability (the wild bootstrap is anti-conservative for cluster-constant regressors). `queue-ahead` is dropped: it is identically 0 under the RiskAverse 0-ms feed, so it carries no information.

**Pooled, model = fill-level (26,256 fills, 24 token clusters; wild cluster bootstrap):**

| regressor (per fill) | Δ markout(30s) ¢ per +1 SD | 95% CI | robustly signed? |
|---|---|---|---|
| **log aggressor trade size** | **−0.10** | [−0.18, −0.03] | **yes — bigger taker ⇒ toxic** |
| **book imbalance at fill** | **−0.20** | [−0.34, −0.05] | **yes — lopsided book ⇒ toxic** |
| fill-price extremity | +0.05 | [−0.18, +0.27] | no |
| log \|inventory\| before | −0.06 | [−0.20, +0.09] | no |
| is-buy | +0.04 | [−0.19, +0.26] | no |
| log depth at fill | −0.01 | [−0.14, +0.13] | no |

The two robust drivers **survive both the wild and the pairs cluster bootstrap** (pairs: −0.10 [−0.19, −0.02] and −0.19 [−0.31, −0.03]) — the strongest evidence in the note. Adding the 7 token-level observables (model = *all*, pairs cluster bootstrap) leaves the same two fill-level signals robust and **none of the token-level features individually sign-robust** — expected, since they are identified only across the 24 markets (e.g. `median_abs_imbalance` +3.6¢/SD but CI [−2.5, +7.0]). So the token-level screen (Task 1a) and the fill-level regression agree: the *certified* signal is at the fill, not the token.

![Per-fill markout drivers (token-clustered bootstrap)](../../data/analysis/plots/market_making/mm_design_fill_regression.png)

*Read this chart:* horizontal bars are each regressor's effect on per-fill markout(30s) in ¢/contract per +1 SD, with token-clustered CIs; red = the CI excludes zero (a robustly-signed driver). *Read:* at the fill level — where we have thousands of points and real power — **two observables robustly flag a toxic fill: a large aggressor trade and a lopsided book at the instant of the fill.** Both are things a live quoter *sees in real time*, so they are the certified, actionable half of the screen: **thin/pull quotes when the aggressor is large or the top-of-book is imbalanced.** Queue position, inventory, side, and price level do not individually predict per-fill markout here.

### The screen (the deliverable Task 5 asked for)

Split by what the data can and cannot certify:

- **Certified (fill-level, real-time, CI-robust):** *shade wider or skip re-quoting when the aggressing trade is large or the top-of-book is imbalanced* — both robustly lower per-fill markout. These are the levers Task 5 can act on tick-by-tick.
- **Strong directional (token-level, use as a pre-quote gate, not a certified threshold):** *do not quote markets parked at ~50¢* (the coin-flip/spike zone — decisive in esports, where 5 of 7 dead tokens sit within 0.05 of 50¢), and *prefer low-mid-volatility, deep books* (low mid-vol is the only pooled CI-clearing separator; depth is directional in politics: median 324 vs 117 contracts).
- **Refuted (do not screen on it):** **half-spread.** A wider touch is *not* a safety cushion — in politics the dead markets have the *wider* spreads. Selection must be on toxicity proxies (price zone, mid-vol, aggressor size, imbalance), never on spread width.
- **Not L2-observable (defer to metadata / live):** semantic resolution-clarity (is the criterion objective?) — the 11-day tape cannot measure it; keep it as the metadata gate the live-loop design specifies ([[mm_politics_negrisk_live_loop_design]]).

Concretely for Task 5: **quote away from 50¢ in calm, deep, low-mid-vol markets; react to large aggressors and book imbalance by widening/pulling; ignore spread width as a selector.** All token-level thresholds are directional (12 tokens/universe) and must be re-fit once live fills accumulate.

---

## Task 2 — the NegRisk time-to-resolution regime

### The natural experiment hiding in the top-12 politics tokens

Time-to-resolution (**TTR**) = `end_date − fill_ts`. The critical realism question ([[CODEX]] rule 1/3) is **does the 11-day capture even span the near-expiry regime for these markets?** Answering it required pulling each politics market's resolution date, and the answer reframes Task 4. The top-12 politics tokens split into three resolution horizons — and the split maps almost perfectly onto viability:

| horizon (Gamma `end_date`) | markets (tokens) | verdicts | what the capture sees |
|---|---|---|---|
| **short-dated, resolves IN-window** | 5 Elon-Musk tweet-count NegRisk markets (Jun-26 / Jun-30) | 2 VIABLE / 3 DEAD | their **full ~2-week life down to TTR ≈ 0** — the *only* near-expiry politics we observe |
| **near-term deadline** | Iran-meeting-by-Jun-30 (1), Fed-no-change-Jul (1) | both VIABLE | Iran only at **~10 days out**; Fed at **~28–39 days** — mid-life, not final hours |
| **durable outrights (Dec-2026)** | Bennett-PM, Petro-out, Starmer-out (5 tokens) | 4 VIABLE / 1 DEAD | **only ~183–194 days from resolution — deep mid-life**; near-expiry **never observed** |

*Practical example.* The strongest politics market in Task 4, `11216749…` (Iran meeting, +0.93¢/contract, VIABLE), is observed by us at **TTR ≈ 9–10 days** — its fills all sit 225–251 h before its own resolution; we never see its final hours. The four durable outrights that anchor Task-4's "politics is the safe venue" verdict are all ~6 months from resolving. **The favorable politics read is, structurally, a mid-life read.**

### Per-fill markout & adverse selection by time-to-expiry

We bucket every politics fill by TTR (fine near expiry, coarse far out) and compute the qty-weighted markout(30s) + adverse selection per bucket with **market-cluster** bootstrap CIs (the honest independent unit is the market, not the autocorrelated fill — a bucket of 800 fills from 5 markets has ~5 effective observations, not 800). Each bucket also reports **how many distinct markets populate it**.

| TTR bucket | fills | distinct markets | net markout(30s) ¢ (95% CI) | adverse selection ¢ (95% CI) | adverse rate |
|---|---|---|---|---|---|
| **<6 h** | 800 | 5 (all Musk) | −1.01 [−1.34, +0.10] | **+1.53 [+0.27, +1.93]** | 0.40 |
| 6–24 h | 697 | 5 | +0.10 [−0.15, +0.32] | +0.36 [+0.13, +0.58] | 0.30 |
| 1–3 d | 1,581 | 5 | +0.24 [−0.10, +0.43] | +0.18 [+0.06, +0.43] | 0.22 |
| 3–7 d | 150 | 5 | +0.44 [+0.41, +0.52] | +0.06 [−0.05, +0.10] | 0.08 |
| 7–30 d | 448 | 2 (Iran, Fed) | +0.91 [+0.19, +0.93] | +0.37 [+0.31, +0.37] | 0.18 |
| >30 d | 4,892 | 4 (3 outrights + Fed tail) | +0.31 [+0.16, +0.59] | +0.29 [+0.20, +0.55] | 0.17 |

![Politics-NegRisk markout & adverse selection vs time-to-resolution](../../data/analysis/plots/market_making/mm_design_ttr_regime.png)

*Read this chart:* x = TTR bucket from near-expiry (left) to mid-life (right); blue = net markout(30s), red = adverse selection, both ¢/contract with market-cluster CIs; the `n=…(k mkt)` labels show fills and distinct markets per bucket. *Read:* the markout **point estimates** trace exactly the hypothesized shape — **−1.0¢/contract inside the last 6 hours** (40% of fills adverse), improving steadily to a **+0.9¢ peak at 7–30 days** and settling at **+0.3¢ deep mid-life**. Under the honest market-cluster CIs the near-expiry markout point is negative but its interval *includes zero* (only 5 markets), while the **near-expiry adverse-selection spike is CI-certified** (+1.53¢ [+0.27, +1.93], lower bound clears zero even clustering on the 5 markets) and mid-life markout (3–7 d, 7–30 d, >30 d) is CI-certified positive. But look at the *distinct-markets* column: the toxic near-expiry buckets are **entirely the 5 short-dated Musk markets**, and the benign far buckets are **entirely the durable outrights** — different market *populations*, not one population aging. That is the confound the next test controls for.

### Does toxicity rise as a market approaches its own resolution?

The pooled buckets confound expiry with market identity (the near-expiry bucket is *different markets* than the mid-life bucket). The clean within-market test regresses markout(30s) on `log(TTR)` **restricted to the in-window resolvers** (the 6 markets observed across a range of TTR) — pooled, and after removing each market's own mean (fixed effects), reported under both bootstraps:

| specification | method | slope of markout(30s) on log(TTR hrs) ¢ | 95% CI | robustly signed? |
|---|---|---|---|---|
| pooled (short-dated set) | wild cluster | +0.49 | [+0.27, +0.70] | **yes** |
| + market fixed effects | wild cluster | +0.35 | [+0.12, +0.58] | **yes** |
| + market fixed effects | pairs cluster | +0.35 | [−0.00, +0.50] | no (touches zero) |

Restricted to the 6 in-window resolvers (3,668 fills), markout **rises with time-to-resolution.** Under the small-K-appropriate **wild cluster bootstrap**, both the pooled slope and the stricter **market-fixed-effects** slope (which removes each market's own mean, so it cannot be driven by cross-market composition) are **sign-robust** (+0.49 and +0.35 ¢/log-hr); under the plain **pairs** cluster bootstrap the FE lower bound just touches zero — so the FE result is method-sensitive at the margin. The effect is carried by the 5 Musk markets, each watched across its full ~2-week life (**4 of 5 have positive within-market slopes** — reproducible in `mm_design_ttr_per_market_slope.csv`); the Iran market is uninformative here (its fills span only a 26 h TTR window at ~10 days out, so its within-slope is noise). At **K=6 markets even the wild bootstrap is only approximate**, so read the gradient as **directional-to-robust support**. The most-certified piece of the near-expiry-toxicity case is the **market-clustered adverse-selection spike inside 6 h** (CI clears zero); the markout gradient (points + slope) corroborates it.

### The mid-life-only caveat (the Task-4 amendment)

Putting the coverage and the gradient together yields the amendment Task 4 needs:

1. **Task-4's "politics is the safer, broader venue" is a MID-LIFE statement.** The durable outrights that make it true (Bennett/Petro/Starmer, 4/5 VIABLE) are observed *only* ~6 months from resolution. Their low, uniform adverse selection is a mid-life property.
2. **The only near-expiry politics we observe is toxic** — and it comes from a peculiar market *type* (Musk tweet-counts: fast, countable, high-information-flow near a hard deadline). So two things are tangled: **expiry proximity** and **market type.** The within-market fixed-effects slope says expiry proximity matters *on its own*; but we cannot cleanly separate it from "tweet-count markets are just toxic," because the durable outrights never enter the near-expiry buckets.
3. **Therefore near-expiry toxicity of the durable outrights is UNOBSERVED — a live-only risk.** Nothing in this 11-day capture tells us how a Bennett/Petro/Starmer book behaves in its final hours. Layer-4 spike-zone theory, K5's 0.8% late-spike behavior ([[block_k5_findings]]), and the short-dated within-market gradient all point one way: **it will get toxic.**
4. **Design consequence:** Task 5 must treat **time-to-resolution as a first-class quoting gate** — *pull quotes as any market approaches resolution* (a time generalization of the price-based spike-zone rule), with the pull threshold **calibrated live**, not read off this capture. The strategy carries this rule by default and relaxes it only when live near-expiry fills prove benign.

---

## Assumption ledger ([[CODEX]] realism rule 3)

**Modeled assumptions (knobs we set):** pessimistic RiskAverse queue + 0-ms latency (the honest lower bound; markout is near queue-invariant at the touch per Task 4, but the 0-ms read is still latency-naive for esports — event snipe unmeasured); half-spread = median touch spread/2 (rest at the touch); markout to the engine's reconstructed mid (JOIN-1 faithful ~99%); `end_date` = the nominal Gamma resolution deadline (for the open-ended Dec-2026 outrights this is a placeholder deadline, not a hard resolution instant — so their true TTR is "large and uncertain," which only *strengthens* the mid-life-only reading); the screen features are the observables a live quoter can compute pre-commit; **inference uses cluster bootstraps matched to regressor type** (wild cluster bootstrap for within-cluster effects, pairs for cluster-constant token-level features), and at the small cluster counts that carry the TTR story (K=6 gradient, ~5-market near-expiry buckets) even the wild bootstrap is approximate — those results are labelled directional, not certified.

**Live-only unknowns (only the 1-contract loop / capture can resolve):** the true passive fill rate net of adverse selection; whether near-expiry toxicity for the **durable** political outrights matches the short-dated proxy (unobserved in this window — see the caveat); the real per-market rebate policy (captured fee = 0); esports event-snipe adverse selection; whether the screen thresholds hold out-of-sample once Task 5 fits a strategy (a strict OOS/overfitting split only bites once something is fit — dormant here, per Task 4).

**Materiality (rule 4).** This note reports **preconditions and regimes**, not deployable edge. Even where a feature separates cleanly, the underlying per-contract edges are the sub-1¢/contract numbers Task 4 already flagged as "statistically positive, economically thin." The screen's value is **avoiding the toxic tail**, not manufacturing size.

---

## Decision and next step

**This is a design-input map, not a trading system** ([[CODEX]] rule 3). Two concrete outputs for Task 5:

1. **The market screen (Task 1).** Select on *toxicity proxies*, not spread. **Certified (fill-level, real-time):** widen/skip on large aggressor trades and on top-of-book imbalance (both CI-robust drivers of toxic fills). **Directional (pre-quote gate):** avoid ~50¢-parked markets (the coin-flip spike zone — decisive in esports), prefer calm (low-mid-vol) deep books. **Refuted:** do not use half-spread as a safety selector (toxic books are the *wider* ones). Token-level thresholds are directional on 12 tokens/universe and must be re-fit live.
2. **The time-to-resolution gate (Task 2).** Per-fill markout is monotone in time-to-resolution (−1.0¢ in the last 6 h → +0.9¢ at 7–30 d), the near-expiry adverse-selection spike is **CI-confirmed even clustering on the 5 markets** (+1.5¢ [+0.27, +1.93] inside 6 h), and the within-short-dated log-TTR slope is sign-robust under the wild cluster bootstrap (pooled and fixed-effects; K=6, so directional-to-robust). **But the near-expiry regime is observed only for short-dated Musk-type markets; the durable outrights that carry Task-4's favorable politics verdict are observed mid-life only.** So: **quote the durable outrights mid-life; pull as resolution approaches, on a live-calibrated clock** — near-expiry toxicity for those markets is a live-only unknown to be measured, not assumed benign.

**Next step:** feed both into the Task-5 inventory-managed quoter as (a) a market-selection filter and (b) a time-to-resolution pull rule; the Join-2 1-contract live loop is where the near-expiry pull threshold and the token-level screen thresholds get calibrated. No sizing before that loop reads out.

---

## Reproduce / artifacts

- **Module:** `polymarket/research/mm_eval/design_inputs.py` (book parsing, as-of attach, rank-AUC + group screen, token-cluster bootstrap OLS, TTR bucketization + bucketed edge). Pure, tested.
- **Runner:** `PYTHONPATH=. uv run python scripts/mm_design_inputs_run.py` (from `polymarket/research/`; `--force` rebuilds the materialize+replay cache, `--quick` smoke). Drives `mm_engine` unchanged; stage-A per-fill dump is cached in scratchpad for instant re-analysis.
- **Tests:** `tests/test_mm_design_inputs.py` — `PYTHONPATH=. uv run pytest tests/test_mm_design_inputs.py`.
- **CSVs:** `data/analysis/csv_outputs/market_making/mm_design_{screen,token_features,fill_regression,ttr_span,ttr_buckets,ttr_gradient,ttr_per_market_slope}.csv` (the `fill_regression` and `ttr_gradient` CSVs carry a `method` column: wild vs pairs cluster bootstrap).
- **Plots:** `data/analysis/plots/market_making/mm_design_{screen_auc,fill_regression,ttr_regime}.png`.
- **Metadata cache:** `scratchpad/politics_market_meta.json`, `scratchpad/esports_market_meta.json` (Gamma `end_date` per condition id).

## Cross-links

Task 4 (the verdict this consumes): [[mm_symmetric_quoter_validation_findings]]. Engine/methodology: [[mm_backtesting_methodology_explainer]] · [[mm_join1_reconciliation_findings]] · [[mm_engine_queue_models]]. Concept lineage: [[mm_concepts_and_strategy_buildup]] (Layer 1 slow-market selection, Layer 4 spike-zone avoidance) · [[block_k5_findings]] (late-spike avoidance) · [[block_k5_stress_findings]] (category adverse-selection table). Live loop this feeds: [[mm_politics_negrisk_live_loop_design]]. Hub: [[strat_market_making]].
