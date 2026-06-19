---
title: "Overfitting Validation — what DSR, PBO, and the Reality Check actually mean"
created: 2026-06-10
status: active
owner: justin
project: crypto
para: resource
hubs:
  - STRATEGY_REFERENCE
tags:
  - crypto
  - validation
  - overfitting
  - methodology
---

# Overfitting Validation — DSR, PBO, and White's Reality Check explained

> Hub: [[STRATEGY_REFERENCE]] (§ H is the terse API/gate reference; this note is the plain-English *why*) · Harness: `infrastructure/validation/overfitting_audit.py` · First worked application: [[momentum_overfitting_audit_findings]] · Doctrine: [[CODEX]] § Realism calibration

## Plain-English Summary

- **What this note is:** a from-scratch explainer of the four statistics our overfitting harness computes (DSR, PBO, Reality Check, synthetic-null Monte Carlo), written for someone who has never seen them. If you only want the function signatures and the pass/fail gate, read `docs/STRATEGY_REFERENCE.md` § H instead; if you want a real example with numbers, read [[momentum_overfitting_audit_findings]].
- **Why it exists:** every strategy in the stack picks its parameters by searching hundreds of Optuna trials and keeping the best. Keeping the best of many tries makes a backtest look good *even when there is no real edge*. These three tests measure how much of a backtest's performance is that illusion.
- **The one idea behind all three:** a single impressive backtest is meaningless without knowing **how many were tried to find it**. Forty coin-flippers will produce one who "called" 5 heads in a row — that is selection, not skill. DSR, PBO, and the Reality Check are three different ways of pricing in the search.
- **Takeaway:** run them all; they fail in different ways and a strategy needs to survive all of them. The pre-registered gate is **deflated Sharpe > 0 AND PBO < 0.5 AND post-haircut lower-CI Sharpe > 0.25**; the synthetic-null MC adds an empirical **real > null-95th-percentile** check that doesn't lean on DSR's distributional assumptions. Statistical survival is necessary but not sufficient — a survivor can still be too small to trade ([[CODEX]] § Realism calibration).

## The problem: searching inflates backtests

Our walk-forward and CPCV engines run **400 Optuna trials per fold/split** ([[STRATEGY_REFERENCE]] § B.2, § G.5) and report the *best* one's out-of-sample Sharpe. The trouble is that the maximum of many noisy estimates is biased upward.

**Toy example.** Suppose a strategy family has **zero true edge** — every parameter set is really a coin flip. Backtest 400 of them on a few years of data. By pure luck, the *luckiest* of the 400 will show a Sharpe of maybe 0.8–1.0, not 0.0. If you report only that winner, the backtest "discovers" an edge that does not exist. The more configs you try, the higher the lucky winner's Sharpe climbs. So **"OOS Sharpe = 1.4" means nothing until you know it was the best of 400 tries** — that is the single fact these three tests are built around.

---

## Test 1 — Deflated Sharpe Ratio (DSR)

**Question it answers:** *Does the winning Sharpe beat what a search of this size would produce on pure noise?*

**Intuition.** Compute the Sharpe a search of N trials would be *expected to find by luck alone* on zero-edge data — call it **SR\*** (the "selection haircut"). It grows with the number of trials and with how much the trial Sharpes vary. Then subtract: **deflated Sharpe = observed Sharpe − SR\***. If the observed Sharpe doesn't clear the haircut, the "edge" is just the luckiest draw of the search.

The DSR also reports a **probability** (`dsr_prob`): the chance the strategy's *true* Sharpe exceeds SR\*, accounting for sample length and fat tails (returns with negative skew / high kurtosis are riskier than a normal distribution, so they need a higher observed Sharpe to clear the same bar). Gate: **deflated Sharpe > 0** (equivalently `dsr_prob > 0.5`); `dsr_prob ≥ 0.95` is the "strong" bar.

**Mini-example.** Observed OOS Sharpe = 1.43, and a 400-trial null search would be expected to find SR\* = 0.18. Deflated Sharpe = 1.43 − 0.18 = **1.25** → clears the haircut comfortably → the edge is not a search artifact. (This is the real BTC momentum number.)

**One subtlety — effective trials.** 400 trials of nearly-identical parameter sets are not 400 *independent* tries. The harness shrinks the count to an "effective N" using the average correlation between trial return series, so the haircut isn't overstated. Raw-400 and full-search counts are reported alongside as harsher sensitivity variants.

---

## Test 2 — Probability of Backtest Overfitting (PBO), via CSCV

**Question it answers:** *Is the config that looks best in-sample actually good out-of-sample, or does picking the in-sample winner systematically hurt you?*

**Intuition.** Take the return series of **all** the candidate configs (not just the winner). Split the timeline into blocks, and for every balanced way of cutting the blocks into an in-sample half and an out-of-sample half:

1. Find the config that ranks **best in the in-sample half**.
2. Look at where that same config ranks **in the out-of-sample half**.
3. If it lands in the **bottom half** out-of-sample, this split is "overfit."

**PBO = the fraction of splits where the in-sample winner is bottom-half out-of-sample.** (CSCV = Combinatorially Symmetric Cross-Validation — it just means doing this over *every* symmetric in/out block partition rather than one arbitrary split.)

- **PBO ≈ 0.5** → in-sample ranking carries **no information** about out-of-sample; picking the IS-best is a coin flip.
- **PBO < 0.5** (gate; < 0.2 is "good") → the IS-best tends to stay good OOS; the selection process works.
- **PBO > 0.5** → selection is **actively anti-informative**: the IS-best tends to be *worse* OOS than its peers.

**Mini-example.** Hide 2023–2024 of BTC's data, rank all 400 configs on the rest, pick the winner. Reveal 2023–2024 and re-rank. If the winner now sits 287th of 400 (bottom half), that split counts toward PBO. Across all 12,870 such splits this happened **85%** of the time → PBO = 0.85 → the optimiser's pick is no better than a random config.

**Important companion number — `P(IS-best loses OOS)`.** A high PBO is alarming only if the IS-best also *loses money* OOS. If PBO is high but `P(OOS loss) ≈ 0`, the winner is merely worse than its **siblings**, not unprofitable — the symptom of a **flat parameter plateau** where every config is about equally good and ranking among them is noise. That is exactly the live momentum book's situation (real edge, useless optimiser step), and it is a very different disease from "the whole family is noise."

---

## Test 3 — White's Reality Check

**Question it answers:** *After accounting for the whole search, is the best config's return distinguishable from zero (cash)?*

**Intuition.** DSR asks whether the winner beats the *search-luck haircut*; the Reality Check asks the more basic question of whether the **best of the whole family beats doing nothing**, with the search built into the null. It's a bootstrap: resample the return histories many times (using a *block* bootstrap that preserves short-term autocorrelation — the stationary bootstrap), and ask how often a no-edge world would produce a best-config mean return as large as the one observed. That frequency is the **p-value**.

- **p < 0.05** (our supporting bar) → unlikely the family's best is a fluke vs cash.
- We use the **studentised** form (Hansen 2005's refinement): scaling each config by its own volatility stops a single noisy high-variance config from dominating the maximum.

This test is reported as **supporting evidence, not gated** — it confirms the family isn't pure noise, which DSR + PBO already largely cover.

---

## How the three fit together — the interpretation matrix

The tests fail in different ways, so read **DSR and PBO together**:

| DSR (haircut) | PBO (selection) | What it means | Action |
|---|---|---|---|
| Pass | Pass (< 0.5) | Real edge **and** the optimiser's pick generalises | **Trust** — proceed to live, sized off the post-haircut CI |
| **Pass** | **Fail (> 0.5)** | Real family edge, but a **flat plateau** — fresh per-fold winners are noise | **Flag for review** — use plateau-centre/consensus params, don't re-optimise harder *(← the live momentum book)* |
| Fail | Pass | Rare: ranking is stable but nothing clears the haircut — usually a tiny, uniform edge | Re-validate; likely too small to trade |
| Fail | Fail | The winner is just the luckiest of the search; no durable edge | **Close** — do not deploy |

*Table read: the unit is one strategy/asset. "Pass/Fail" is against the pre-registered gate (DSR: deflated Sharpe > 0; PBO < 0.5). The second row is the most common and most misread outcome — a strong backtest that is genuinely real yet whose **optimisation step adds nothing**, because the parameter surface is flat. The fix there is process (anchor parameters), not abandonment.*

The full gate also requires the **post-haircut lower-CI Sharpe to clear a materiality bar (0.25)** — because a strategy can pass DSR and PBO and still be economically trivial after a realistic haircut. Statistical survival ≠ economic materiality ([[CODEX]] § Realism calibration); always quote the *post-haircut* Sharpe and its conservative lower bound, never the raw search-best.

## Combating overfitting — what to do when a result is bad

The cure depends on **which test failed** (read alongside the matrix above). The wrong reflex is to keep re-running the search until the numbers improve — that is itself overfitting, and [[STRATEGY_REFERENCE]] § G.6 caps iteration cycles at 3–4 for exactly this reason.

**High PBO — selection is the problem (the live momentum book's case).**
The in-sample-best config doesn't generalise, but the family edge may be real. Don't abandon — fix the *selection step*:
- **Deploy plateau-centre params, not the per-fold winner.** Use `consensus_params` / median-of-stable-params (`cpcv_parameter_analysis` already computes these). On a flat plateau the median config is a better OOS bet than the freshly-optimised winner.
- **Freeze stable params.** Anything with `stability_df` CV < 0.15 → fix it. Fewer free knobs = lower selection bias and a lower DSR haircut.
- **Collapse redundancy.** The parameter correlation matrix and tercile comparison reveal params that don't independently matter; merge or drop them — the strategy has fewer real degrees of freedom than it looks.
- **Ensemble over the plateau** instead of betting on one winner.

**Failed DSR — the winning Sharpe is within search-luck range.**
- **Shrink the search.** Fewer free params and fewer trials *lower SR\* itself* (you're "discovering" less by luck), so the same observed Sharpe can clear a smaller haircut. Counterintuitive but correct.
- **If the deflated Sharpe stays near zero across reasonable search sizes, the signal is genuinely too weak.** More trials cannot manufacture edge — you need a *different* signal, not more tuning.

**Failed Reality Check — the whole family ≈ cash.**
- Not just the winner but the entire candidate set is indistinguishable from zero. Redesign or abandon the signal; parameter work won't rescue it.

**General hygiene that lowers all three (do regardless of which failed):**
- **More data / more regimes** — the single biggest lever. Selection bias shrinks as the sample lengthens and spans more conditions; it also directly attacks the regime-risk caveat below.
- **Cost-stress the survivor.** Re-run `cost_stress_test` (in `wf_engine.py`); an edge that survives 2–3× the modelled per-leg cost is far more trustworthy.
- **Hold out a final slice in time** the search never touched, and check the gate there too.
- **Pre-register the gate before looking** (we do) — stops the subtle overfitting of nudging the bar to fit the result.
- **Re-baseline expectations to the post-haircut Sharpe** and size risk off the conservative lower bound, never the headline.

## Test 4 — Synthetic-null Monte Carlo (the empirical cross-check)

**Question it answers:** *could the entire search pipeline have manufactured the real result on data with no exploitable signal?* This is the **empirical version of the DSR haircut**: instead of an analytic expected-max formula (which assumes normality and independent trials), it builds synthetic no-signal markets and runs the same best-of-N selection on each one.

**Intuition.** Chop the real price history into short blocks (stationary block bootstrap of bar-level relative bars) and re-chain them: the synthetic market keeps the real drift, fat tails, short-range volatility clustering, and (when resampled jointly) cross-asset correlation — but the longer-range trend structure a momentum rule exploits is destroyed. Evaluate **all N candidate configs** on each synthetic market, take the max Sharpe, and repeat for hundreds of paths. That's the distribution of "best search result in a world where the strategy's premise is false." **Gate (pre-registered): real statistic > 95th percentile of the null distribution.**

**Key design choice — the block length defines the null.** Short blocks (5–10 bars) destroy the structure a multi-week strategy trades; long blocks (~20+) *preserve* month-scale runs, i.e. the null then partially **contains the phenomenon being tested** and the bar rises mechanically. Run the pre-registered primary (block 10 for daily momentum) plus block-length and demeaned (drift-removed) sensitivities, and read long-block rows as *horizon localisation* of the edge, not falsification. The momentum-book run showed exactly this shape: decisive pass on the primary and harsher nulls, marginal per-asset results only against month-run-preserving nulls — see [[momentum_overfitting_audit_findings]] § Synthetic-null Monte Carlo.

**How to run it.** `make_null_ohlcv(df_or_dict, mean_block_len, rng, demean)` generates paths (pass a dict of aligned frames for joint multi-asset resampling); `synthetic_null_mc(df, eval_fn, ...)` is the serial driver; for big runs parallelise over paths in a file-based script (macOS spawn deadlocks on heredoc/stdin scripts) — pattern in `topics/momentum/strategies/momentum_cpcv/run_synthetic_null_mc.py`, including the verified fast-strategy-port trick that makes ~10⁶ backtests tractable. Regime-stratified nulls (HMM labels, `topics/regime-classifier/` — currently BTC-only research) are the natural next refinement.

**Not in this harness — trade-shuffle / drawdown MC.** Permuting trade order to build a max-drawdown distribution is a *position-sizing / path-risk* tool, not an overfitting test; naive shuffling also scatters regime-clustered losing streaks and so understates tail drawdowns. The CPCV path max-DD distribution (already produced by every CPCV run) is the better drawdown object; a block-permutation DD tool belongs in risk tooling if ever needed.

## What these tests do NOT tell you

- **Regime risk.** They measure selection bias *within* the sample. They cannot tell you the edge will survive a regime the sample never contained (e.g. our 2020–2026 crypto sample is bull-heavy for long-only momentum).
- **Live frictions.** Real slippage, fees beyond the modelled per-leg cost, capacity, and fill quality are live-only unknowns — ship them in the assumption ledger.
- **Whether the strategy logic is sound.** A strategy can be statistically clean and still be a bad idea for reasons no backtest sees.

## How to run it

**In a CPCV notebook (preferred).** Run the search with `collect_trials=True` (the [[STRATEGY_REFERENCE|cpcv_template]] sets this), then the template's final **"Overfitting check (required before live)"** cell calls `audit_cpcv_run(...)` and renders the verdict box inline. Paste the printed raw-markdown block into the strategy's findings note.

**Standalone.**

```python
from overfitting_audit import run_overfitting_audit
verdict = run_overfitting_audit(
    selected_oos_returns=oos_returns,   # the chosen config's OOS net returns
    trial_returns=trial_matrix,         # T x N matrix of every candidate's returns
    n_trials=400,
    periods_per_year=365,               # daily crypto
    label='BTCUSDT momentum',
)
print(verdict.to_markdown())            # the formatted verdict box
print(verdict.verdict)                  # 'PASS' | 'FLAG-FOR-REVIEW' | 'INSUFFICIENT-DATA'
```

## Reading a verdict box

Each row is one check with its value, the gate it must clear, and a pass/fail. The header line gives the overall verdict. Worked example with the real numbers and a line-by-line read: [[momentum_overfitting_audit_findings]] § Results.

## Pointers

- **Code + docstrings:** `infrastructure/validation/overfitting_audit.py` (every function documents its formula and source).
- **Reference (API, gate, integration):** [[STRATEGY_REFERENCE]] § H.
- **Worked application (the live momentum book):** [[momentum_overfitting_audit_findings]].
- **Original research spec + library survey:** [[2026-06-05_novelty_deep_research]] § A, [[2026-06-05_novelty_frontier_map]] Prompt 1.
- **Primary sources:** Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"; Bailey, Borwein, López de Prado & Zhu (2014) "The Probability of Backtest Overfitting"; White (2000) "A Reality Check for Data Snooping"; Hansen (2005) "A Test for Superior Predictive Ability."
