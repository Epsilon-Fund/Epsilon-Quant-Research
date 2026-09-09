---
title: "XS momentum thread recap (May 2026, likely scrapped)"
created: 2026-05-11
archived: 2026-09-09
status: archived — pre-migration vault (May 2026); historical, not current
owner: justin
project: crypto
para: archive
hubs:
  - pre-migration-vault-2026-05
tags:
  - archive
  - pre-migration-vault
  - crypto
---

> **ARCHIVED — PRE-MIGRATION VAULT (May 2026).** Historical record, imported 2026-09-09 from the standalone Cowork vault that predates this repo becoming the brain. Index: [[pre-migration-vault-2026-05]].
> **PARKED (2026-08-25).** The crypto momentum book is live, but this note is a May-2026 snapshot of it, not its current state. The live record is `docs/STRATEGY_REFERENCE.md` and [[TODO]] § Crypto — do not build on this note or quote its numbers as current.

*State recap of the XS momentum research thread. Captured 2026-05-11. **Status: likely to be scrapped — empirical results not promising.** Preserved here for the methodology lessons + perturbation-robustness findings, which survive the strategy decision.*

---

# Crypto Momentum — XS thread status summary

## 1. Active Research Thread

Cross-sectional (XS) momentum strategy as a complement to the existing time-series (TS) momentum stack. The goal is a long/short XS strategy that diversifies the TS strategies' bear-regime drawdowns (specifically 2022 and 2026 periods where TS systematically loses) and ultimately deploys via Hyperliquid vault.

The architecture is intentionally separate from the TS work — independent strategy families combined at the portfolio level, not embedded into a single hybrid strategy. The two families share `wf_engine.py` / `wf_visualizer.py` infrastructure but live in their own folder structure under `momentum/xs_momentum/`.

Champion signal candidate: residual Sharpe (Blitz/Huij/Martens 2011 construction adapted to crypto). Beta-strip BTC from each coin's returns, then take the Sharpe of the residual time series over the formation window. This is rank-equivalent to the residual t-statistic — well-grounded in equity literature with documented robustness properties.

## 2. Built and Shipped vs In-Flight

**Shipped to the engine:**

- New folder structure `momentum/xs_momentum/` with notebook template adapted from existing TS pattern
- Multi-asset data layer pulling perp OHLCV from Binance for the universe
- Universe cache + filter system: parquet-based cache of perp data with incremental daily augmentation; lightweight filter function reads from cache with no API calls
- Universe filter parameters: TOP_N=40, MIN_VOLUME=$100M, MIN_AGE=180d, VOLUME_WINDOW=30d
- Three signal variants tested through the full WF + perturbation pipeline: raw returns, rolling Sharpe, residual alpha
- Portfolio construction: long top N / short bottom N with configurable long/short weight split (currently 50/50 baseline, designed for asymmetric extension)
- Universe size handling: hard floor `MIN_UNIVERSE_SIZE` below which strategy goes flat
- XS-specific diagnostics replacing the trade-level stats: universe size over time, turnover per rebalance, long vs short leg attribution, spread (long minus short return), rank autocorrelation, basket-level hit rate

**In-flight (designed, not yet built):**

- Composite signal: 50/50 average rank of rolling Sharpe and residual alpha. Theoretical case is strong (two robust components combine multiplicatively); empirical test pending.
- Asymmetric leg sizing architecture: Layer 1 (BTC vs MA-200 directional binary), Layer 1.5 (universe breadth as continuous magnitude), Layer 2 (cross-sectional dispersion as gross exposure gate). Designed but not implemented.
- Block bootstrap for OOS confidence intervals — designed, lightweight to add.

**Explicitly deferred (v3+ or contingent):**

- CPCV — defer until champion strategy is locked and we're at deployment-readiness validation
- Variable K driven by dispersion — defer; theoretical case is weaker, infrastructure cost is higher
- Leg-asymmetric K (K_short < K_long) — contingent on diagnostic showing asymmetric leg decay
- BTC ADX as Layer 1.6 — contingent on breadth alone showing residual unexplained losses
- DVS overlay — defer; CVS captures most of the benefit, DVS requires return forecasting that adds estimation noise
- Multi-factor residualisation (BTC + ETH) — defer until single-factor version is validated

## 3. Recent Bug Fixes and Learnings

**Universe construction:**

- Issue: Initial design pulled spot data for ranking and perp data for execution as two parallel caches. Fix: Consolidated to perps-only — for large-cap liquid assets the spot/perp price difference is negligible on daily bars, and dual-pipeline complexity wasn't justified. Lesson: Theoretical purity (spot for clean prices) doesn't always justify infrastructure complexity when the practical difference is sub-basis-point.
- Issue: Volume filter initially considered spot volume. Fix: Perp volume only. Lesson: Filter on the data dimension you'll actually execute in — spot liquidity doesn't predict perp liquidity reliably.
- Issue: Initially considered fetching universe filter live from Binance API at each rebalance inside the WF loop. Fix: Two-layer system — offline cache script + lightweight in-notebook filter reading from cache. Lesson: API-bound operations inside hot loops are a performance and rate-limit trap; cache once, filter many times.

**Signal construction:**

- Issue: Misconception that residual momentum could be computed with a single regression over the J-bar window producing J residuals, then Sharpe of those residuals. Fix: Confirmed the correct construction uses rolling beta (each bar gets its own β from its own J-bar window), giving one scalar residual per bar; the Sharpe is then a second J-bar window over those residuals. Warmup is 2J as a result. Lesson: "Compounded rolling windows" multiply in data hunger but the standard residual momentum literature (BHM) uses this construction precisely because constant-beta-within-window misrepresents time-varying market exposure.

**Parameter discipline:**

- Issue: Considered putting universe filter parameters (TOP_N, MIN_VOLUME, MIN_AGE, VOLUME_WINDOW) into the Optuna parameter space. Fix: Kept them as fixed infrastructure parameters, sensitivity-tested manually at tight/base/loose configurations. Lesson: Infrastructure parameters (what data is trustworthy) and strategy parameters (how to use that data) are categorically different — mixing them in Optuna is a subtle but severe overfitting vector.
- Issue: Considered optimising regime parameters (SMA window, ADX threshold) for leg sizing. Fix: Same principle — regime parameters are fixed, sensitivity-tested, not Optuna candidates.

Constraint: Established that J_max must be `≤ UF_MIN_AGE_DAYS - 30` (buffer for post-listing settling). Universe filter constrains parameter lookback, not vice versa.

**Empirical robustness findings:**

- Rolling Sharpe has notably less parameter fragility than raw returns and substantially less perturbation degradation
- Residual alpha has less fragility than raw returns; debatable vs rolling Sharpe on fragility, but the strongest perturbation result of the three
- The perturbation result for residual alpha is the most important finding — it's evidence the signal is capturing something structural, not parameter-specific

## 4. Open Questions / Blocking Decisions

**Q1:** Does the composite signal beat either component alone on robustness? The theoretical case is strong (two robust components, different mechanisms — vol normalisation × beta neutralisation). The empirical test is pending. If the composite shows lower perturbation degradation than either component, it becomes the champion. If it ties or underperforms residual alpha, residual alpha becomes the champion.

**Q2:** Do leg returns show asymmetric decay over the holding period? The hypothesis is that short-leg signals decay faster than long-leg signals (losers collapse in air pockets, winners grind). This is the deciding diagnostic for whether leg-asymmetric K is worth building. Until this is checked, leg-specific K stays on the backlog.

**Q3:** Do the strategy's bad periods cluster in any particular regime quadrant? Cross-tab daily PnL against (BTC above/below MA-200) × (universe dispersion high/low). If losses cluster cleanly, regime filtering is well-motivated and the layered architecture should work. If losses spread across all quadrants, the layered architecture won't help much and a different approach is needed.

**Q4:** Does universe breadth add real value over BTC-vs-MA alone? Diagnostic: compute both signals daily, find the ~25–30% of periods where they disagree (decoupling regimes), check strategy performance in those periods. If breadth-disagreement periods show systematic strategy underperformance, breadth as Layer 1.5 is justified. If not, Layer 1 alone may suffice.

**Q5:** What's the right long/short weight split for the baseline? Currently 50/50 for clean attribution. Research suggests 60/40 long-biased may be the natural landing spot given the asymmetric reliability of long vs short signals in crypto. Decision deferred until composite signal is locked in, since the weight split should be informed by leg attribution under the champion signal, not the baseline.

## 5. Immediate Next 3 TODOs When We Resume

*(Captured for completeness — superseded if XS is scrapped.)*

**TODO 1:** Build and test the composite signal. Implement the 50/50 average-rank composite of rolling Sharpe and residual alpha. Run through the existing WF + perturbation pipeline. Compare IS/OOS Sharpe, fragility, and perturbation degradation against both individual components. Output: champion signal selected.

**TODO 2:** Run the regime diagnostic. Compute (a) BTC vs MA-200 binary, (b) universe breadth (% of universe above MA-200), (c) cross-sectional dispersion (rolling percentile rank) over the full backtest period. Overlay against daily strategy PnL. Output: empirical evidence on whether the layered leg-sizing architecture addresses real failure modes, and which layers add information vs which are incremental noise.

**TODO 3:** Build block bootstrap utility for OOS confidence intervals. Quick utility (~50 lines) that bootstrap-resamples blocks of consecutive OOS returns to produce a confidence interval on OOS Sharpe. This gives statistical grounding to "is the composite genuinely better than residual alpha alone, or are they within noise?" — which is the question that decides TODO 1's output. Output: confidence intervals replacing point estimates in signal comparison.

After these three, the path forward is: build Layer 1 + 1.5 asymmetric leg sizing if TODO 2 supports it → add Layer 2 dispersion gate → run full WF on the v2 architecture → evaluate against v1 baseline → if v2 wins materially, implement CVS overlay as the final v2.5 risk management layer → then CPCV as deployment validation gate.

---

## Why this is likely being scrapped

*(2026-05-11.)* Empirical results not promising. Specific reasons TBD — capture here when scrapping is finalised. Distilled lessons from this thread live in `methodology-lessons.md` so they survive the strategy decision.
