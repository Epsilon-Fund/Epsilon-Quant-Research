---
title: "Momentum Overfitting Audit — does the live book's 2.24 Sharpe survive a selection haircut?"
created: 2026-06-10
status: active
owner: justin
project: crypto
para: resource
hubs:
  - STRATEGY_REFERENCE
tags:
  - crypto
  - momentum
  - validation
  - overfitting
---

# Live crypto momentum book — overfitting audit (DSR / PBO / Reality Check)

> Hub: [[STRATEGY_REFERENCE]] · Harness: `infrastructure/validation/overfitting_audit.py` (reference: [[STRATEGY_REFERENCE]] § H; plain-English explainer of the tests: [[OVERFITTING_VALIDATION]]) · Runner: `run_overfitting_audit.py` (this folder) · Raw results: `oos/overfitting_audit_2026-06-10.pkl`, `overfitting_audit_summary.csv` · Pre-registration trail: [[2026-06-05_novelty_frontier_map]] Prompt 1, [[2026-06-05_novelty_deep_research]] § A

## Plain-English Summary

- **What this is:** the first run of the new reusable overfitting-validation harness, applied to the live 6-asset daily momentum book (ADA, AVAX, BTC, ETH, SOL, XRP on Binance, long-only trend-following, equal weight). The question: how much of the headline 2.24 portfolio Sharpe is an artifact of picking the best of 400 Optuna trials per optimisation fold?
- **Why it was written:** the 2.24 number is the mean over CPCV paths of search-selected configs; nothing in the stack previously measured selection bias. This was the top never-run item on the novelty frontier list.
- **What ran:** for each asset, a same-design 400-trial TPE search was replayed on the exact sample behind the artifacts (prices reconstructed from the pickles, not refetched), giving the full candidate-return matrix the original search effectively chose from. From that: Deflated Sharpe Ratio, PBO via CSCV, and White's Reality Check, with bootstrap CIs.
- **One-line takeaway:** **the edge is real but the optimiser isn't adding anything** — the book's Sharpe survives the selection haircut comfortably (post-haircut ≈ 1.94, conservative lower bound ≈ 1.20, never below 0.84 even under the harshest haircut variant), but PBO is 0.57–0.92 across assets (mean 0.805), meaning the in-sample-best parameter set is *systematically not* the out-of-sample-best. The pre-registered gate therefore fires: **FLAG-FOR-REVIEW** — keep the book live, re-baseline expectations to ~1.6–1.9 portfolio Sharpe, and stop trusting per-fold optimiser winners over plateau-centre (consensus) parameters.

## What was tested and why these three statistics

The momentum stack selects parameters with Optuna TPE, `n_trials = 400` per walk-forward fold / CPCV split ([[STRATEGY_REFERENCE]] § B.2, § G.5). Selecting the best of 400 trials inflates the winner's backtest performance even on pure noise. Three complementary statistics quantify this:

1. **Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado 2014. Computes the Sharpe a 400-trial search would be *expected to find on zero-edge data* (call it SR\*, the "selection haircut"), then asks whether the observed OOS Sharpe clears it, accounting for sample length and return skew/kurtosis. "Deflated Sharpe" below = observed Sharpe − SR\*, in annualised units.
2. **Probability of Backtest Overfitting (PBO)** via CSCV — Bailey, Borwein, López de Prado, Zhu 2014. Takes the matrix of *all 400 candidate configs'* return series, splits time into 16 blocks, and for each of the 12,870 balanced in-sample/out-of-sample block partitions asks: does the config ranked **best in-sample** land in the **bottom half out-of-sample**? PBO = fraction of partitions where it does. 0.5 = in-sample ranking carries zero information; above 0.5 = selection is actively anti-informative.
3. **White's Reality Check** — White 2000, studentised per Hansen 2005. Bootstrap p-value for "the best config's mean return beats zero *after* accounting for having searched 400 configs." Guards against the whole family being noise.

**Practical example of what PBO measures here:** take BTC's 400 replayed configs. Hide 2023–2024, rank all 400 on the remaining years, and pick the winner — say `ema_span=17, atr_stop=22, …`. Now reveal 2023–2024 and rank all 400 on it. If the hidden-period rank of that winner is 287th of 400 (bottom half), that partition counts toward PBO. For BTC this happened in **84.9%** of the 12,870 partitions.

## Design — how the numbers were produced

- **Sample fidelity.** OHLCV per asset was reconstructed by unioning the per-group OOS frames inside `oos/{sym}usdt_cpcv.pkl` (the 8 CPCV groups tile the full sample), so the audit runs on byte-identical data to the artifacts behind the 2.24 headline (≈2,030–2,170 daily bars per asset, mid-2020 → 2026-04). A live Binance refetch would have shifted the sample.
- **Trial matrix via same-design replay.** The engines historically discarded each Optuna study after extracting `best_params`, so the original 28×400 trials per asset are unrecoverable. Instead, one fresh 400-trial TPE study per asset (TPESampler seed 42, the asset's exact `param_defs`/`fixed_params` from its pickle, the notebook's exact strategy function, SCORE_FN and REJECT_FN) was run on the full sample, capturing every explored config's net returns through the standard backtest (1-bar execution lag, 0.1%/leg cost, realised sizing). This is the candidate set a 400-trial selection event sees — same design, not a bit-identical recovery. *(Fixed structurally going forward: `run_cpcv`/`walk_forward` now take `collect_trials=True`; the CPCV template sets it.)*
- **Selected-OOS series.** Per asset: the **median-Sharpe CPCV path's** equity-curve returns — one representative pure-OOS realisation. Averaging the 105 paths would shrink variance and flatter the Sharpe.
- **Effective trial count.** Trials are correlated (most configs trade the same trends), so the DSR haircut uses N_eff = p̄ + (1−p̄)·400 where p̄ is the average pairwise correlation of trial returns (pre-registered formula; distinct from the engine's path-overlap N_eff = 7 — different objects). Raw N=400 and the full-search N=11,200 are reported as harsher sensitivity variants.
- **Portfolio haircut.** Selection bias inflates each sleeve's *mean return* by ≈ SR\*ᵢ·σᵢ. Means average linearly across the 6 equal-weight sleeves while portfolio volatility shrinks with diversification, so the portfolio Sharpe haircut is SR\*_p = mean(SR\*ᵢ·σᵢ)/σ_p — **diversification does not diversify away selection bias**, and this correctly makes the portfolio haircut (0.30) larger than the average per-asset haircut.

### Pre-registered gate (set before results were seen)

> PASS requires **all** of: portfolio deflated Sharpe > 0, **and** mean per-asset PBO < 0.5, **and** post-haircut lower-CI Sharpe > 0.25 (materiality bar). Reality-Check p < 0.05 is supporting evidence, not gated. Anything else ⇒ FLAG-FOR-REVIEW.

## Results

### Per-asset table

| Asset | CPCV path Sharpe (mean) | Selected OOS Sharpe [95% CI] | N_eff /400 | SR\* haircut | Deflated Sharpe | DSR prob | PBO (S=16) | P(OOS loss) | RC p |
|---|---|---|---|---|---|---|---|---|---|
| ADA | 1.38 | 1.29 [0.45, 1.99] | 21 | 0.27 | **1.02** | 0.997 | **0.773** | 0.011 | 0.009 |
| AVAX | 1.41 | 1.40 [0.42, 2.21] | 12 | 0.13 | **1.28** | 1.000 | **0.818** | 0.000 | 0.009 |
| BTC | 1.40 | 1.43 [0.47, 2.32] | 14 | 0.18 | **1.25** | 0.999 | **0.849** | 0.001 | 0.003 |
| ETH | 1.24 | 1.24 [0.40, 2.00] | 12 | 0.14 | **1.10** | 0.998 | **0.920** | 0.000 | 0.001 |
| SOL | 1.72 | 1.71 [0.80, 2.59] | 9 | 0.12 | **1.59** | 1.000 | **0.900** | 0.001 | 0.002 |
| XRP | 1.30 | 1.30 [0.53, 1.99] | 22 | 0.24 | **1.06** | 0.999 | **0.570** | 0.000 | 0.014 |

**Column glossary.** *CPCV path Sharpe (mean):* mean annualised Sharpe over that asset's 105 CPCV paths (the per-asset number behind the portfolio headline). *Selected OOS Sharpe:* the median-Sharpe CPCV path, with a stationary-bootstrap 95% CI — this is the series the DSR is computed on. *N_eff:* effective independent trials out of 400 after correlation shrinkage (configs are highly correlated — most of a 400-trial search is the same bet retried). *SR\* haircut:* expected max annualised Sharpe a search of N_eff null trials would find; the amount to subtract for selection. *Deflated Sharpe:* selected Sharpe − SR\*. *DSR prob:* probability the true Sharpe exceeds SR\* given sample length and non-normality (gate: > 0.5; strong bar: ≥ 0.95). *PBO:* fraction of 12,870 CSCV partitions where the IS-best config ranks bottom-half OOS (gate: < 0.5; block-count sensitivity S=8/12 moved no asset materially, range ±0.07). *P(OOS loss):* fraction of partitions where the IS-best config's OOS Sharpe is actually negative. *RC p:* studentised Reality-Check p-value (B=2,000, stationary bootstrap, MC standard error ≤ ±0.005).

**Read:** two opposite signals, both strong. Every asset clears DSR at ≥ 0.997 with deflated Sharpe ≥ 1.0 — at the actual trial count, selection bias explains only 0.12–0.27 of Sharpe per asset. Every asset also clears the Reality Check at p ≤ 0.014 — the family's returns are not noise. But **every asset fails PBO**, four of six at 0.8+: the config the optimiser picks in-sample is *systematically in the worse half* out-of-sample. Crucially, P(OOS loss) ≈ 0 everywhere: the IS-best config underperforms its *siblings*, not cash.

### Why both can be true — the parameter plateau

The replayed trial Sharpes are extraordinarily tight per asset (full-sample, TPE-explored):

| Asset | Trial Sharpe median | IQR | Max |
|---|---|---|---|
| ADA | 1.53 | 1.42–1.60 | 1.67 |
| AVAX | 1.54 | 1.48–1.59 | 1.64 |
| BTC | 1.85 | 1.77–1.89 | 1.93 |
| ETH | 1.59 | 1.53–1.63 | 1.68 |
| SOL | 1.94 | 1.86–1.98 | 2.00 |
| XRP | 1.59 | 1.48–1.64 | 1.75 |

*(Unit of observation: one row per asset over its 400 replayed trial configs; each value is that config's full-sample annualised Sharpe, net of costs. TPE concentrates sampling near good regions, so this is the search's candidate population, not a uniform draw over the parameter space.)*

**Read:** the search space — after the prior stability passes froze many params ([[STRATEGY_REFERENCE]] § G.5–G.6 discipline) — is a **plateau**: nearly any candidate config earns within ~0.2 Sharpe of any other. On a plateau, in-sample ranking among near-identical siblings is pure noise, so PBO ≈ 0.5 *at best* and drifts above it when the IS winner is the one that got luckiest in-sample (mild mean reversion of luck). The same flatness keeps cross-trial Sharpe variance tiny, which is exactly why the DSR haircut is small. High PBO + near-zero P(OOS loss) + large deflated Sharpe = **real family edge, zero marginal selection skill**. BTC's engine-native IS/OOS efficiency (mean 0.677) corroborates: OOS delivers ~⅔ of IS Sharpe — degradation, not collapse.

### Portfolio verdict vs the 2.24 headline

| Quantity | Value | Note |
|---|---|---|
| Headline portfolio Sharpe | **2.238** | mean over 2,000 CPCV portfolio paths, equal-weight 6 assets, OOS 2020-10 → 2026-04 |
| Engine adjusted 95% CI | [1.50, 2.97] | overlap-adjusted (the honest CI the stack already reported) |
| Portfolio SR\* haircut | **0.30** | at N_eff per asset; sensitivity: 0.50 at raw N=400, **0.66** at full-search N=11,200 |
| **Post-haircut Sharpe** | **1.94** | 1.74 / **1.58** under the two harsher haircut variants |
| Post-haircut lower CI | **1.20** | 1.00 / **0.84** under harsher variants — never near zero |
| Portfolio DSR prob | 1.000 | PSR of the median portfolio path at benchmark SR\*_p |
| Mean per-asset PBO | **0.805** | gate requires < 0.5 — **fails** |
| **Pre-registered gate** | **FLAG-FOR-REVIEW** | deflated > 0 ✓ · PBO < 0.5 ✗ · post-haircut lower-CI > 0.25 ✓ |

![[overfitting_audit_sharpe_haircut.png]]
*Chart read: per asset and for the portfolio — blue = observed OOS Sharpe, red = the selection haircut SR\* (what a same-size search would find on zero-edge data), green = the difference (deflated Sharpe). The dashed line is the pre-registered 0.25 materiality bar. Everything green sits far above it; the problem this audit surfaces is not in this chart — it is the PBO column in the table above.*

**Read:** the answer to the headline question — *does the live book's edge survive a proper overfitting haircut?* — is **yes in level terms**: a realistic forward baseline is ≈ 1.9 portfolio Sharpe (≈ 1.6 under the harshest haircut), with a conservative lower bound of ≈ 1.2 (0.84 harshest). The 2.24 headline should no longer be quoted unhaircut. The gate still fires, correctly, because PBO reveals a process problem: **the optimisation step is buying nothing** — a median (plateau-centre) config would be expected to do as well as or better OOS than the freshly-optimised winner.

## Assumption ledger (realism calibration, [[CODEX]] § Realism calibration)

**Modeled assumptions (could move the numbers):**
- The trial matrix is a *same-design replay* (one 400-trial TPE on the full sample per asset), not the original discarded per-split studies; cross-trial variance and correlations are estimated from it. The original selection events optimised on ~75% train slices — the replay's objective sees the full sample.
- DSR's N_eff = p̄ + (1−p̄)·M (pre-registered) — raw-400 and 11,200-trial variants reported throughout; no variant changes any conclusion.
- Portfolio haircut aggregates sleeve drags in return units (assumes sleeve selection biases are additive in means); 0.2% round-trip cost as everywhere in the stack; the representative OOS series is the median-Sharpe path.
- PBO's 12,870 CSCV partitions are mutually dependent — no formal CI exists; block-count sensitivity (S=8/12/16) spans ±0.07 around each estimate and never approaches the 0.5 gate from above.
- This whole sample is 2020-2026 crypto — structurally bull-heavy for a long-only momentum family. The audit measures selection bias *within* this sample; it cannot rule out regime dependence of the family edge itself.

**Live-only unknowns (no backtest can resolve):**
- Realised slippage/fees vs the 0.1%/leg assumption at live size; execution at the daily 08:00 UTC bar.
- Whether the family edge persists out of the 2020-2026 regime — the post-haircut 1.9 is a backtest-calibrated expectation, not a forecast.

## Decision and next steps

**Gate outcome: FLAG-FOR-REVIEW (pre-registered; PBO leg failed).** Concretely:

1. **Keep the book live.** Nothing here says the edge is fake: deflated Sharpe is materially positive under every haircut variant, Reality Check clears at p < 0.015 on all six assets, and the IS-best config almost never loses money OOS. There is no capital-destruction signal.
2. **Re-baseline expectations.** Plan around **≈ 1.6–1.9 portfolio Sharpe** (not 2.24), with drawdown sizing against the post-haircut conservative lower bound **≈ 0.84–1.2**. Any future note quoting this book should quote the post-haircut number.
3. **Change the re-optimisation policy.** PBO ≈ 0.8 says fresh per-fold optimiser winners are *worse* bets than plateau-centre parameters. At the next scheduled re-optimisation, prefer `consensus_params`/median-of-stable-params (the stack already computes these) over the newest `study.best_params`, and freeze any param whose stability CV clears the existing 0.15 bar. Do not add trials or search iterations — G.6's overfit-to-the-path-distribution warning is now empirically visible.
4. **Structural fix is in place:** all new searches run with `collect_trials=True` (CPCV template updated), so future audits use the *actual* trials, not a replay; the audit gate is a required template step before any strategy goes live ([[START_RESEARCH_IDEA]] step 6, [[STRATEGY_REFERENCE]] § H).
5. **Do not** reopen the search space or re-run CPCV to "fix" the PBO number — on a plateau that is exactly the iteration-cycle overfitting G.6 caps.
