---
title: "Polymarket copytrade — methodology lessons ledger (May 2026)"
created: 2026-05-19
archived: 2026-09-09
status: archived — pre-migration vault (May 2026); historical, not current
owner: justin
project: copytrade
para: archive
hubs:
  - pre-migration-vault-2026-05
tags:
  - archive
  - pre-migration-vault
  - copytrade
---

> **ARCHIVED — PRE-MIGRATION VAULT (May 2026).** Historical record, imported 2026-09-09 from the standalone Cowork vault that predates this repo becoming the brain. Index: [[pre-migration-vault-2026-05]].
> **DEPRIORITISED (2026-08-25).** The copy-trading thread is not the active research thread (that is [[strat_market_making]]). It is not archived either — its execution/signing infrastructure is live and shared with the market-making machinery. Pick this thread back up only with Justin.

# methodology lessons — polymarket copytrade

Generalisable rules learned from concrete incidents. Append-only at the top (newest first). Each block ties a principle to a concrete trigger and (where applicable) a fix pattern. Mirror of similar pattern in `crypto-momentum/research/notes/methodology-lessons.md`.

---

## 2026-05-19 — scheduled-ingest stamp (no new lessons)

**Principle.** Nightly inbox ingest fired against `2026-05-15_pm-data.md` and `2026-05-16_pm-data.md`. All Section 3 lessons from both source files are already captured below (9 from 2026-05-15 pm-data v1, 2 from the same-day v2 revision, 6 from 2026-05-16 pm-data). No new lesson blocks added to avoid duplicating curated entries already present.

**Concrete trigger.** Both source files had been hand-ingested into this file prior to the scheduled task running. Files remained in `inbox/` and were processed + archived by the nightly task on 2026-05-19.

**Source.** `inbox/_archive/2026-05/2026-05-15_pm-data.md`, `inbox/_archive/2026-05/2026-05-16_pm-data.md`.

---

## 2026-05-16 — Copyability is per-(trader × family × role × hour), not per-cohort

**Principle.** "Pick a cohort, copy everyone in it" assumes cohort membership predicts transferable copy-behaviour. Empirically, it doesn't. The right unit of analysis for copy-trading is a 4-tuple per leader: `(trader × market-family × maker-or-taker × hour-of-day)`. Each leader has 1-6 deployable cells across this product space; cross-leader overlap at this granularity is thin.

**Concrete trigger.** Per-leader audit framework (`scripts/domah_copy_audit.py`, ~90s per leader). 7 leaders audited; only 2 narrow shared cells across the set. Cohort framing in `04-cohort-pools.md` had no concept of deployable-cell density per leader — the right metric for picking who to follow.

**Fix pattern.** Audit individual leaders first; defer cohort-signal investigation until ≥12 audits available (below that, signal-vs-noise discrimination is too thin). Threshold-setting for any layer-1 cohort metric needs 15-20 audits before it's meaningful.

**Reframes earlier work.** `04-cohort-pools.md` six-pool framework remains factually correct (counts, overlap matrix, WHERE clauses) but its *thesis* — that cohort membership predicts copyability — is wrong. Preserve as historical reference; add supersession note.

**Source.** `inbox/2026-05-16_pm-data.md`.

---

## 2026-05-16 — Highest lifetime PnL does NOT predict copyability

**Principle.** Lifetime PnL is the wrong selection heuristic for copy-trading. Empirically: Spearman ρ between `mkt_total_pnl` and copyable-cell count = −0.21 (n=7). The leaders with the most lifetime PnL often have it because their strategy has elements (split-construction, deep queue priority, unique market access) that can't be mirrored.

**Concrete trigger.** Per-leader audit results show `0x6a72f61820b2…` (top of leaderboard at $14.95M) has only 2 deployable cells — Domah ($59M lifetime bankroll, NegRisk arb) has 3, despite being explicitly flagged as do-not-copy in v1.

**Fix pattern.** Use `active_days_last_90d` (ρ = +0.81) and `hold_to_resolution_share` (ρ = +0.64) as primary predictors. Apply `split_position_signature > 60%` exclusion to drop architecturally-uncopyable leaders before any ranking. **Reframes `04-cohort-pools.md`'s "smoke against `0x6a72f61820b2…`" recommendation** — the selection heuristic that produced it (PnL × pool-membership × exec-readiness-filter) was optimising the wrong quantity.

**Source.** `inbox/2026-05-16_pm-data.md`.

---

## 2026-05-16 — Recency dominates copyability prediction

**Principle.** Lifetime metrics are deceptive when copying execution. The relevant question is "would I have copied them in the last 90 days?", not "are they good over their lifetime?". Recency captures the current liquidity/style regime — which is what the bot would actually be operating in.

**Concrete trigger.** `active_days_last_90d` ρ = +0.81 vs copyable-cell count. Strongest single predictor across the 7-audit sample.

**Fix pattern.** When ranking candidate leaders, sort by recency × hold-to-resolution rate (× exclusion filters for split_position_signature and operator-like flags). Lifetime PnL is a sanity floor at best, not a ranking driver.

**Cross-reference.** Echoes ADR 0001 v0.2 finding that exec wants 30d-rolling metrics, not lifetime. Same lesson at the trader-selection layer.

**Source.** `inbox/2026-05-16_pm-data.md`.

---

## 2026-05-16 — Maker-style traders are adversely-selected if you copy them as makers

**Principle.** The leader's natural execution role isn't necessarily your best execution role when copying. For high-`style_role_balance` (maker-conviction) leaders, posting passive limits *alongside* their fills is adversely selected — your maker order tends to fill when their thesis is wrong (someone else is paying for the privilege of being on the other side).

**Concrete trigger.** Per-leader audit: `style_role_balance` ρ = −0.64 vs copyable-cell count. Confirms the queue-priority lesson from 2026-05-15 at the per-leader granularity.

**Fix pattern.** When copying a maker-conviction leader, consider crossing the spread (taker mode) rather than mirroring their maker mode. The leader earns the spread; the copier doesn't necessarily.

**Reframes ADR 0001.** The `pricing_mode_recommended` heuristic (high `style_role_balance` → `leader_fill` pricing mode) is **inverted**. The correct mapping is more like: high role_balance → cross-the-spread to follow their maker fills (taker execution to match speed); low role_balance → mirror as taker (their natural mode). ADR needs v0.4 patch.

**Source.** `inbox/2026-05-16_pm-data.md`.

---

## 2026-05-16 — Architectural uncopyability via splitPosition

**Principle.** Some traders' positions change outside `OrderFilled` events entirely. They use CTF-level operations (`splitPosition`, `mergePositions`, `redeem`) to construct or unwind positions. **No execution improvement can fix this** — your bot's WebSocket subscription wouldn't see the construction event at all. These leaders are architecturally uncopyable, full stop.

**Concrete trigger.** `0xd38b71f3` re-investigated under the per-leader audit framework. Previously framed in `04-cohort-pools.md` as the "atypical taker, role_balance=0.22, defer to second smoke" candidate. Audit reveals the taker role balance was a *consequence* of `splitPosition` construction (which produces taker-shaped fills on the natural side), not a strategy choice. 10 of top-50 traders by lifetime PnL fall into this category.

**Fix pattern.** New trader-level metric: `split_position_signature` (in `traders_directionality.parquet`). Apply as a hard-exclusion filter before any cohort or audit work. Generalises: when picking leaders for an execution-bound strategy, audit *how* they construct positions, not just what they end up holding.

**Cross-reference.** Connects to the "execution-model is first-class" lesson from 2026-05-15 — both are about treating mechanism-of-action as a primary design variable.

**Source.** `inbox/2026-05-16_pm-data.md`.

---

## 2026-05-16 — `phantom_position_score` retired as primary arb filter

**Principle.** When an arb-detection metric flags <100% of arb-shaped traders, it's not "tunable" — it's structurally missing a class of arb construction. The right fix is a more direct signature of the construction mechanism, not a threshold tweak.

**Concrete trigger.** `phantom_position_score` (the previous arb-detection metric) found Pool C (NegRisk specialists) was **91% contaminated** under fill-concentration analysis in the new `traders_directionality.parquet`. The phantom score's "≫1.0 = arb-shaped" framing had already needed an empirical-baseline calibration on 2026-05-11 (all six pools sit at 1.4-1.8 baseline). This week's finding is the next layer down — for the specific subset where you'd most expect the metric to fire, it was missing 91% of cases.

**Fix pattern.** Replaced by fill-concentration metrics in `traders_directionality.parquet`. Phantom score still useful as a coarse arb-direction signal but no longer the authoritative arb filter.

**Reframes ADR 0001.** The `phantom_position_score < 2.0` filter is partially obsolete — keep as a coarse screen, but the authoritative arb-exclusion now comes from `traders_directionality.parquet` (and `split_position_signature` for the architectural case).

**Source.** `inbox/2026-05-16_pm-data.md`.

---

## 2026-05-15 — Execution-model assumption is a first-class research input

**Principle.** The execution model — taker / passive / hybrid / quote-driven — determines what every downstream metric and filter *means*. Two readings of the same data under different execution models can flip a deployability verdict without a single data point changing. Document the execution model alongside any deployability conclusion. Treat "what execution model are we assuming?" as a question you ask *before* defining the slippage proxy, not after.

**Concrete trigger.** Weather FTC TP same-day pivot: initial Proposal-B analysis concluded "shelve at canonical parameterisation" under taker (cross-the-spread) execution. Subsequent WS-passive (post limits at the touch via CLOB WebSocket) analysis showed `(p_in=0.50, p_out=0.90)` is deployable at +6% ROI per filled trade, 2,283 fills/yr. The next-fill price proxy was measuring what other takers paid; it doesn't apply to passive execution at all. Same data, different model, opposite verdict.

**Cascading implications.**

- "Data-supported subsets are more expensive" (Proposal B) is a **taker** finding. Under passive execution, the "data-supported subset" is just the subset where fills happen — by definition the trading set.
- "Constant slippage with labelling" applies to next-fill *trade-print* proxies (which are taker-shaped). It does not apply to quote-based or fill-rate-based metrics.
- The optimal grid cell can change between execution models — under taker the canonical (0.60, 0.90) was the focus; under passive the best cell is (0.50, 0.90) because lower entry price gives more cross events and more headroom to TP.

**Fix pattern.** When evaluating a strategy, run the execution-model assumption up to the front of the spec. List the candidate execution models (taker, passive, hybrid, quote-driven). Derive the meaningful metric *per model*. Only then do you have a deployability verdict that's a verdict-about-the-strategy and not a verdict-about-the-implicit-execution-model.

**Source.** `inbox/2026-05-15_pm-data.md` revision.

---

## 2026-05-15 — Queue priority is the dominant deployability question for passive strategies

**Principle.** For any passive (limit-posting) strategy, the gap between *optimistic-exit* and *strict-exit* assumptions tells you the entire deployable-vs-not range. The actual fill mix sits somewhere in between, determined by your real queue priority. There is no offline data analysis that closes this gap — only small-live measurement does.

**Concrete trigger.** Weather WS-passive canonical (0.60, 0.90): +0.87% ROI under optimistic exit (assume our `p_out` ask lifted whenever price reached `p_out`), −2.5% ROI under strict exit (require a real aggressive-buy print). 20pp gap in TP rate of filled trades (0.445 vs 0.246). Real performance is determined by queue priority on the exit ask.

**Fix pattern.** Pre-deployment test: small-live 2-4 weeks at canonical params with sizes small enough to absorb worst-case drawdown. Log every fill with its queue-position-at-arrival (where available) plus realised exit lag. Reconcile fill mix against the optimistic vs strict bracket to pin down true edge. Only scale after the bracket has been bounded.

**Cross-reference.** Connects to the polymarket exec side's `pricing_mode_recommended` from `style_role_balance` heuristic in ADR 0001 — the same maker-vs-taker-conviction question shows up there. Worth carrying the queue-priority framing into ADR 0001 when we revisit it for the live cohort copy-trading bot.

**Source.** `inbox/2026-05-15_pm-data.md` revision.

---

## 2026-05-15 — Compounding bug: equity convention mismatch is silently 4-order-of-magnitude wrong

**Principle.** When sharing capital across trades, decide *once* whether you're modelling compounding or flat-bankroll, and make sure the equity curve and the per-trade PnL series use the same convention. Mathematically incompatible conventions produce wealth-trajectory numbers that diverge by orders of magnitude while leaving Sharpe and bucket probabilities looking fine.

**Concrete trigger.** Weather FTC TP backtest had `equity = cumprod(1 + ret)` (compounding) while `pnl_pct` was computed on flat $1 (no compounding). Headline reported total_return = 2,272,346% (two million percent). Real number: 628.6%. Sharpe and bucket probs unaffected — they're scale-invariant.

**Why it's dangerous.** Scale-invariant diagnostics don't catch it. You pass every sanity check that doesn't explicitly compare end-of-period wealth against a hand-computed expectation.

**Fix pattern.** Pick a single equity model up front. Either equity is realised + open-position MTM at flat sizing (no `cumprod`), or per-trade PnL is fed into a compounding curve (`size_t = equity_{t-1} × pct`). Never mix.

**Source.** `inbox/2026-05-15_pm-data.md`.

---

## 2026-05-15 — Constant-slippage-with-labelling: imputed values masquerading as measurements

**Principle.** Any imputed value can be *reported* as a measurement; the test of whether it actually is one is whether the answer would change if you swapped the imputation rule. If not, you have a constant with extra labelling.

**Concrete trigger.** Weather slippage diagnostic: `fallback_pct > 50%` on three of four legs. The resulting "slippage estimate" was the `fallback_cents=3¢` fallback constant with sample variance from the minority real-fill rows. Reported edge of −1.42¢/share was mostly the fallback constant.

**Fix pattern.** Diagnostic emits literal string `"constant_slippage_with_labelling"` when fallback_pct > 50% to flag the issue at the metric layer. Generalise: any imputation-heavy metric should self-discredit when imputation share crosses a threshold. The right follow-up is a sensitivity sweep on the imputation rule — if the answer moves materially, you're measuring the imputation, not the world.

**Reusability.** Apply this pattern to other slippage modules in Phase 5 cohort backtests.

**Source.** `inbox/2026-05-15_pm-data.md`.

---

## 2026-05-15 — Data-supported subsets can be more expensive, not cleaner

**Principle.** When proposing a "filter to data-supported subset" cleanup, ask: what does the *existence* of supporting data correlate with? In thin markets with episodic activity, it correlates with adverse selection. The subset that has the data you wanted is often the subset where the underlying conditions made the trade harder.

**Concrete trigger.** Weather FTC TP — restricting to crosses where a real next-fill existed within 5 min (29% of anchors) made edge *worse*, not better. The markets that print follow-up fills are markets where price is actively drifting (hence the follow-up). Observed entry slip 5-7¢ on the "data-supported" subset vs the 3¢ assumed on the rest.

**Fix pattern.** Before filtering to data-supported rows, run a between-subsets comparison on a proxy for the underlying difficulty (e.g. observed slippage, observed drift). If the subset that has data is also the subset where conditions are harder, the filter is selecting on adverse selection.

**Source.** `inbox/2026-05-15_pm-data.md`.

---

## 2026-05-15 — Wider lookup windows contaminate slippage proxies with drift

**Principle.** When a slippage proxy gets *worse* as you give it more time, the proxy is contaminated with directional drift. Later fills carry drift information, not execution-cost information. The fix isn't more time — it's a structurally better fill model.

**Concrete trigger.** Weather analysis: extending next-fill window from 5min → 10min dropped fallback by 5-10pp but made edges worse (same_dir −0.69¢, opp_dir −0.32¢).

**Fix pattern.** When the slippage proxy is drift-contaminated, the real fix is quotes (bid/ask) instead of prints. Trade-based next-fill proxies are upper-bounded by the contamination.

**Source.** `inbox/2026-05-15_pm-data.md`.

---

## 2026-05-15 — Event-market structural thinness dominates realisable edge

**Principle.** For execution-cost-bounded strategies on event markets, structural illiquidity at the cross/anchor times dominates realisable edge. This is a market-class property — generalises to any sub-second-or-near-real-time edge strategy on event markets.

**Concrete trigger.** Weather markets — 43% of crosses had NO follow-up fill within 30 min. Conditional lag percentiles (any side): p10/p25/p50/p75/p90/p99 = 30/86/298/788/1315/1752 seconds.

**Practical reading.** For any future cohort/strategy targeting event-market execution, *liveness of the post-signal moment* is a first-class diagnostic. If most crosses are followed by silence, no fill model will close the gap.

**Cross-reference.** Reinforces the polymarket exec-side "96.4% held to resolution" finding — most positions don't close via on-book sells. Same market-class property surfacing in two contexts.

**Source.** `inbox/2026-05-15_pm-data.md`.

---

## 2026-05-15 — Bookkeeping: defaulted-NaN coercion is a latent landmine

**Principle.** Defaulted-label coercion of NaN inputs is among the most dangerous silent failures in financial code. Always raise or explicitly drop; never assume the default category is benign.

**Concrete trigger.** Weather backtest: NaN resolution silently became `0` (chop). Zero affected rows in the current dataset, but the latent landmine was there until caught.

**Fix pattern.** `assert .notna().all()` or explicit `dropna(subset=critical_cols)` at the top of any function that reads a categorical input. Make the failure mode noisy.

**Source.** `inbox/2026-05-15_pm-data.md`.

---

## 2026-05-15 — Dedup on the outcome-partition timestamp, not the convenience timestamp

**Principle.** When deduping bets, dedup on the field that defines the *outcome partition*, not the field that defines convenience. For event-resolution betting that's `end_ts`, not `entry_ts`.

**Concrete trigger.** Weather `one_trade_per_city` was keying on `entry_ts.floor("D")`; should have been `end_ts.floor("D")`. Two-way drift — caught some legitimate duplicates and missed others (same-day-entry-different-resolution-day siblings).

**Source.** `inbox/2026-05-15_pm-data.md`.

---

## 2026-05-15 — Default param drift between code paths

**Principle.** When multiple call sites pass the same param, the function's default is dead code at best and a misalignment bomb at worst. Either remove the default (require callers to pass explicitly) or move it to a module-level config constant that all call sites read from.

**Concrete trigger.** Weather `backtest()` default was `max_notional_pct=0.05`; `grid_backtest` was passing `0.02`; docstring said `0.02`. Direct calls to `backtest()` inflated `avg_notional_per_trade` from 1.98% to 3.69%.

**Source.** `inbox/2026-05-15_pm-data.md`.

---

## 2026-05-15 — Universe-constant annualisation across grid cells

**Principle.** When comparing across configurations of the same backtest, the time axis must be a universe-wide constant, not derived per-config. Otherwise you're comparing different things.

**Concrete trigger.** `grid_backtest` was computing Sharpe/CAGR with per-cell annualisation spans (derived from each cell's first/last trade). Cells with shorter spans had inflated annualised numbers. Fix: one constant `annualization_days=345` across all cells.

**Source.** `inbox/2026-05-15_pm-data.md`.

---

## 2026-05-11 — Indiscriminate copy drowns in slippage

**Principle.** Signal-volume and edge-magnitude have to be matched against expected per-signal cost. When cohort-membership cardinality is large, Top-K is the canonical answer.

**Concrete trigger.** Stage 1 backtests copied 270+ leaders → 80k signals/run → cost dominated returns.

**Fix pattern.** Switch from "everyone in the qualified pool gets copied" to Top-K=10 selection within the qualified pool. Stage 1.5 spec uses percentile-based qualification + absolute floors + Top-K=10.

**Source.** `inbox/2026-05-11_pm-data.md` (archived to `_archive/2026-05/`).

---

## 2026-05-11 — In-sample cohort thresholds don't generalise across time

**Principle.** Any absolute-threshold selection rule needs an OOS sanity-test before being applied historically. Thresholds calibrated against the most recent snapshot will be uncalibrated for older periods.

**Concrete trigger.** Phase 4 thresholds were calibrated against the 2026 snapshot. Applied to 2024 cohorts they nearly emptied: B = 11 leaders/month, BC = 0 signals.

**Fix pattern.** Replace absolute thresholds with `percentile + floor + Top-K`. Percentile self-calibrates per period; floor prevents nonsense in degenerate periods; Top-K caps cardinality.

**Source.** `inbox/2026-05-11_pm-data.md` (archived to `_archive/2026-05/`).

---

## 2026-05-11 — Document the empirical floor when the theoretical floor is rarely observed

**Principle.** When a metric's theoretical floor is rarely observed, document the empirical floor *before* downstream code uses the theoretical one as a threshold. Otherwise the threshold encodes a wrong intuition.

**Concrete trigger.** `phantom_position_score` was framed as "1.0 = pure directional" — idealised. All six cohort pools sit at 1.4–1.8 baseline. The metric is useful as relative signal (1.7 normal vs 8+ heavy arb) but not as an absolute threshold.

**Fix pattern.** Keep the `< 2.0` filter as a coarse arb-screen but reframe interpretation as "low end of normal," not "pure directional." Update `formulas.md` §2.4 calibration note.

**Source.** `inbox/2026-05-11_pm-data.md` (archived to `_archive/2026-05/`).

---

## 2026-05-11 — Date-filter consistency across CTE branches

**Principle.** When a metric depends on date-filter conditions, every CTE/branch must apply the same filter. Silent inconsistency between branches biases the tail (whales, in this case).

**Concrete trigger.** Phase 3 bankroll bug — inconsistent placeholder-date filtering across CTEs inflated whale traders' lifetime peaks by 5-7×. Surfaced only when the bankroll point-in-time rebuild reconciled against the timeseries.

**Fix pattern.** Centralise date-filter logic in a single CTE / view; downstream branches read from it. When a metric exists in two implementations (e.g. lifetime vs point-in-time), reconcile them and treat divergence as a bug signal.

**Source.** `inbox/2026-05-11_pm-data.md` (archived to `_archive/2026-05/`).

---
