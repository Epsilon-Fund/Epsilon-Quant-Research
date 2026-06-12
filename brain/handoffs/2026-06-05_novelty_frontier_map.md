---
title: "Novelty frontier map — ranked cross-book strategy candidates + pre-registered gates + Codex prompts"
tags: [handoff, novelty, map, ranked, codex-prompts, crypto, polymarket, cross-pollination]
created: 2026-06-05
status: living map — the actionable, ranked, gated version of the novelty hunt across both books. Each idea has an EV/readiness rating, a pre-registered gate, and (for the top 5) a ready-to-paste Codex prompt.
purpose: >
  Turn the deep-research foundation into a decision surface: what to test, in what order, with what
  pre-registered acceptance criterion, and the exact Codex prompt to run it. Built to be acted on, not
  just read.
relationship: >
  Evidence base + citations: [[2026-06-05_novelty_deep_research]]. Polymarket state map this extends:
  [[2026-06-04_state_of_the_arc_and_novelty_frontier]]. Crypto hub: [[STRATEGY_REFERENCE]]. PM hubs:
  [[strat_market_making]], [[strat_options_delta]], [[POLYMARKET_BRAIN]]. Discipline:
  [[CODEX]] § Realism calibration, [[COWORK]] § The research loop. Codex prompts follow [[COWORK]] § Cowork prompt discipline.
---

# Novelty frontier map

## Summary

- This is the **act-on-it** companion to [[2026-06-05_novelty_deep_research]] (the cited evidence). It ranks ~12 novel candidates across all four battlefields Justin chose — crypto live, Polymarket frontier, cross-pollination, brand-new domains — weighting foundation-models + classical + raw EV.
- **Top priority is not a new strategy — it's a robustness audit.** Before trusting/scaling the live momentum book (2.24 Sharpe, but the max of a 400-trial Optuna search), run the **overfitting + Monte-Carlo layer** (Deflated Sharpe, PBO/CSCV on the trials, stationary-block bootstrap, Hansen SPA, regime-aware MC). Cheap, uses data we own, directly answers "how overfitted is my stuff." This is **Codex Prompt 1**.
- **Highest-EV genuinely-new bets:** (crypto) a vol-managed overlay on the live trend book = near-free Sharpe; a funding-carry sleeve; (cross-book) **Kalshi macro repricing → crypto vol throttle** that de-risks the momentum book using the PM pipeline we already have; (Polymarket) **NegRisk-basket consistency**, the strongest structural edge, now corroborated by an independent $40M academic study.
- **Everything is gated on net-of-cost realized PnL on a holdout, never R²/Brier** — Prophet Arena's GPT-5 beats the market on calibration yet loses net of spread; the Briola caveat binds. See [[CODEX]] § Realism calibration.
- **Five ready-to-paste Codex prompts below** (§ Codex Prompts). Three more (PM touch/barrier, Block J LLM two-arm, copy lag-decay) are scoped in the table and draftable on request.

---

## Ranked candidate menu

Rating key — **EV**: expected edge if it works. **Ready**: shovel-readiness (data on hand, low build). **Novelty**: how un-tested this is for us. Scores ◐ low ● med ★ high.

| # | Candidate | Book | EV | Ready | Novelty | Pre-registered gate (primary endpoint) | Lit anchor |
|---|---|---|---|---|---|---|---|
| 1 | **Overfitting + MC robustness layer** (DSR, PBO/CSCV, block-bootstrap CIs, SPA, regime-aware MC) | crypto | ★ | ★ | ● | Momentum book passes **DSR ≥ 0.95 on effective-N**, **PBO < 0.2**, real Sharpe beyond 95th pct of regime-aware MC null. Any sleeve failing → do not scale. | LdP DSR/PBO; Bailey; Hansen SPA |
| 2 | **Vol-managed overlay on live trend book** (σ-target replaces/【augments】ATR sizing) | crypto | ★ | ★ | ● | OOS Sharpe & Calmar of σ-targeted book **> current ATR-sized book**, lower max-DD, **after the § 1 audit**. | Moreira–Muir; crypto risk-managed momentum |
| 3 | **Funding-carry sleeve + funding-extreme de-risk gate** | new domain | ★ | ● | ★ | Rules-based long-spot/short-perp book entered on funding>fee-hurdle, exited on sign-flip earns **positive net Sharpe after fees/slippage/borrow across a CPCV split spanning ≥1 funding-regime reversal**; gate sub-test: fading top-funding-decile longs improves live book max-DD. | BIS Crypto Carry (WP 1087) |
| 4 | **Kalshi macro → crypto vol-target / regime throttle** | cross-poll | ★ | ● | ★ | Adding KXRECSSNBER/KXCPI daily Δprob to HAR-RV vol forecast **lowers OOS MSFE (Clark-West p<0.05)** for BTC/ETH 5-day RV; vol-targeting on augmented forecast **raises momentum Sharpe** vs HAR-only throttle. | Mohanty–Krishnamachari (2604.01431) |
| 5 | **NegRisk-basket consistency live scanner + persistence measurement** | PM | ★ | ● | ● | On executed baskets where (1 − Σ best-ask YES) > threshold net of gas+2% fee, **realized return > 0**; secondary: gap **persistence measured in minutes** on multi-condition sets (the mis-accounting thesis). | Arb in Prediction Markets (2508.03474) |
| 6 | **Residual momentum, formalized** (BTC+ETH beta strip, rank residual-ret/residual-vol) | crypto | ● | ★ | ◐ | Residual-momentum L/S **beats current residual-Sharpe ranking OOS** (Sharpe, turnover-adjusted), after § 1 audit. | Blitz–Huij–Martens |
| 7 | **Meta-labeling sizer** (secondary model sizes/filters the rule signals) | crypto | ● | ● | ● | Triple-barrier-labeled, purged-CV meta-model **improves net Sharpe & cuts false-positive trades** vs raw signal OOS; **purged CV mandatory** (leakage makes this look magic). | López de Prado meta-labeling |
| 8 | **Neutral MM in slow politics/sports** (boundary-aware A-S + Glosten-Milgrom toxicity pull) | PM | ● | ◐ | ★ | Live measurement loop: **spread+rewards − realized adverse-selection PnL > 0** over ≥30 settled markets; A-S-vs-symmetric-quoter diff as endpoint. (Standing live loop.) | A-S 1605.01862; Bayesian PM MM |
| 9 | **Block J — LLM resolution-criteria + slow-news scan** (two arms) | PM | ● | ◐ | ★ | (A) fine-print arm: hit-rate on flagged mismatches **>60% & PnL>0** net cost; (B) slow-news arm beats zero on **thin/slow** markets while the same pipeline on liquid markets does **not** (contrast = endpoint). | Halawi 2402.18563; FutureSearch; Prophet Arena |
| 10 | **Deribit variance-risk-premium harvest** (regime-gated short-vol) | new domain | ● | ◐ | ● | Regime-gated delta-hedged short-straddle (sell only when IV/RV>thr AND vol-of-vol calm) **positive net Sharpe across a CPCV split incl. ≥1 vol spike**, beating always-on. | BTC VRP ~66% (2410.15195) |
| 11 | **Vol-path model → PM same-day touch/barrier** (Kronos/HAR first-passage vs PM digital) | cross-poll | ● | ◐ | ● | On same-day BTC/ETH touch markets, jump-augmented HAR-RV/Kronos first-passage price **beats PM mid & is net-profitable on a held-out CPCV split** vs Black-Scholes-σ. NOT the 5-min market. | First-passage jump-diffusion; Kronos |
| 12 | **Copy the informed minority** (skilled-wallet mirror, lag-decay diagnostic) | PM | ● | ● | ◐ | Copy-portfolio **Sharpe > 0 net of copy-lag slippage**; primary diagnostic = edge-vs-seconds-of-delay decay curve. Re-merges with copytrade thread. | Gómez-Cram (SSRN 6617059) |

**Sequencing recommendation.** Do **#1 first** (it gates the trust you place in everything crypto). Then the near-free crypto upgrades (#2, #6) and the de-risking cross-poll (#4) in parallel — all reuse owned data. #3 (funding carry) is the best standalone *new sleeve*. On the PM side, #5 (NegRisk measurement) is the cleanest deterministic edge; #8 is already the standing live loop; #9/#11/#12 are research bets to queue behind those.

**What I deliberately did NOT propose** (stay-closed per [[CODEX]] § Realism calibration reopen filter): any PM *terminal* crypto/equity pricing strategy (efficient, robustly closed); continuous/banded gamma-scalp (turnover); single-venue neutral crypto MM (structural adverse selection, directly tested); the PM 5-minute market (execution/oracle-latency game, not forecasting); pure CEX-DEX latency arb and DeFi AMM LP (latency-oligopoly / LVR-dominated).

---

## Codex Prompts

> Paste inline in chat per [[COWORK]] § Cowork prompt discipline. Each starts with the required read-order preamble. Outputs are `*_findings.md` notes linked from the relevant hub — not committed prompt files. **Crypto** prompts substitute `docs/STRATEGY_REFERENCE.md` for the PM brain in the read order.

### Prompt 1 — Crypto backtest-overfitting + Monte-Carlo robustness layer (DO FIRST)

```markdown
Before doing anything else, read, in order:
1. `brain/CODEX.md`
2. `brain/TODO.md`
3. `brain/COWORK.md`
4. `docs/STRATEGY_REFERENCE.md`
5. `brain/handoffs/2026-06-05_novelty_deep_research.md` § A (the robustness toolkit) and `brain/handoffs/2026-06-05_novelty_frontier_map.md` (candidate #1).

GOAL: build a reusable backtest-overfitting + Monte-Carlo robustness module for the crypto WF/CPCV
engine and run it on the LIVE momentum book first. This is a measurement/validation task, not a new
strategy. Respect the repo invariants (separate venv, non-overlap math, CI required, lookahead-free).

DELIVERABLES (code under `infrastructure/validation/` or `topics/momentum/research/`, findings note at
`docs/` or `topics/momentum/research/robustness_overfitting_findings.md`):

1. PERSIST OPTUNA TRIALS. Confirm/extend the WF+CPCV pipeline to save EVERY trial's OOS return series
   (not just the winner) for at least the live momentum configs — CSCV/PBO needs the full N-config × T
   matrix. If not currently persisted, add it.

2. DEFLATED SHARPE RATIO + effective-N. Implement (or vendor `rubenbriones/Probabilistic-Sharpe-Ratio`
   logic): σ_SR non-normal (skew/kurtosis), PSR, expected-max-Sharpe across trials, and effective
   independent N via average pairwise trial-return correlation (N_eff = p + (1−p)·M). Report DSR for the
   live momentum book and each per-asset config. GATE: DSR ≥ 0.95 to "trust"; flag anything below.

3. PBO via CSCV. Run combinatorially-symmetric CV on the persisted trial matrix (S=16 blocks). Report
   PBO, performance-degradation slope, and probability-of-loss. GATE: PBO < 0.2 good, < 0.5 acceptable.

4. STATIONARY BLOCK BOOTSTRAP. Using `arch` (StationaryBootstrap + optimal_block_length), bootstrap the
   live book's returns 10k× → report 5th-pct Sharpe and 95th-pct max-DD (autocorrelation-respecting CIs),
   replacing point estimates.

5. HANSEN SPA / StepM. With `arch.bootstrap.SPA` and `StepM`, test the momentum config family vs a
   buy-and-hold benchmark; report SPA p-value and the StepM-superior set. GATE: SPA p < 0.05.

6. REGIME-AWARE MONTE-CARLO. Using the existing HMM regime labels, block-bootstrap BARS within-regime,
   re-run the full signal pipeline on each synthetic series → null Sharpe/Calmar distribution. Report the
   percentile of the real Sharpe. GATE: real Sharpe beyond the 95th percentile of the synthetic null.

7. MinBTL sanity check: report 2·ln(N_eff)/E[max]² and compare to the live track length.

For each test, report the statistic, the gate verdict, and a plain-English read. Stratify all results BY
HMM REGIME, not just pooled. End with a per-strategy CLOSE/TRUST/RE-VALIDATE verdict and explicitly state
which sleeves should NOT be scaled until they pass. Confidence intervals on every headline; non-overlap by
default. Add the note to the STRATEGY_REFERENCE hub with a wikilink.
```

### Prompt 2 — Vol-managed overlay on the live trend book

```markdown
Before doing anything else, read, in order:
1. `brain/CODEX.md`
2. `brain/TODO.md`
3. `brain/COWORK.md`
4. `docs/STRATEGY_REFERENCE.md`
5. `brain/handoffs/2026-06-05_novelty_deep_research.md` § E.1 (vol-managed portfolios) and the map note candidate #2.

GOAL: test whether volatility-targeting (Moreira–Muir) improves the LIVE momentum book. Do NOT touch
signal generation — only sizing. PREREQUISITE: this is only meaningful after Prompt-1's robustness audit;
note in the findings whether the baseline book passed DSR/PBO first.

DESIGN: replace (and separately, augment) the current ATR-based `position_size_raw` with an inverse-
realized-variance scaler targeting a constant annualized vol (sweep target ∈ {40,60,80}% on a 30-day EWMA
of realized vol; also test a 20-day). Keep entries/exits/stops identical. Walk-forward + CPCV exactly as
the existing momentum notebooks, same cost convention (per-leg 0.001).

REPORT: OOS Sharpe, Calmar, max-DD, turnover, and the left-tail (worst 5% of monthly returns) for
(a) current ATR sizing, (b) pure σ-target, (c) σ-target × ATR hybrid — per asset and portfolio. GATE:
σ-target or hybrid must beat current sizing on OOS Sharpe AND max-DD with CIs (CPCV effective-N). Respect
the Cederburg et al. robustness caveat — report whether the improvement survives a realistic
implementation lag (size on t-1 realized vol, not t). Findings → `topics/momentum/research/
vol_managed_overlay_findings.md`, linked to STRATEGY_REFERENCE.
```

### Prompt 3 — Funding-carry sleeve + funding-extreme de-risk gate

```markdown
Before doing anything else, read, in order:
1. `brain/CODEX.md`
2. `brain/TODO.md`
3. `brain/COWORK.md`
4. `docs/STRATEGY_REFERENCE.md`
5. `brain/handoffs/2026-06-05_novelty_deep_research.md` § D.1 (crypto carry) and the map note candidate #3.

GOAL: validate a delta-neutral perpetual funding-carry sleeve as a new, low-correlation book, AND test
funding-extreme as a de-risk overlay on the existing momentum book. For data artifacts, use the relevant
data/artifact manifests; do not relink raw shards one by one. If funding-rate history is not already
cached, scope the cheapest way to pull Binance (and optionally Bybit/OKX) funding + spot/perp marks first.

SLEEVE DESIGN: long spot / short perp equal notional on the liquid universe; enter when trailing-7d mean
funding > a fee+slippage hurdle, exit on funding sign-flip; inverse-vol weight across coins; weekly
rebalance. Model fees, taker slippage, and the 8h-vs-other settlement intervals. Validate with WF + CPCV
over a window spanning ≥1 funding-regime reversal (must include a deleveraging event). GATE: positive net
Sharpe after all costs with CPCV effective-N CI; report correlation to the momentum book conditional on
drawdown deciles (expose tail co-movement, do not just quote the unconditional matrix).

OVERLAY SUB-TEST: does suppressing/fading momentum longs when their funding is in the top decile improve
the live book's max-DD without killing CAGR? Report as an overlay on the existing momentum OOS path.
Findings → `topics/<new funding-carry folder>/funding_carry_findings.md` (or `docs/`), linked to
STRATEGY_REFERENCE. CIs on every headline; non-overlap by default.
```

### Prompt 4 — Kalshi macro repricing → crypto vol-target / regime throttle

```markdown
Before doing anything else, read, in order:
1. `brain/CODEX.md`
2. `brain/TODO.md`
3. `brain/COWORK.md`
4. `docs/STRATEGY_REFERENCE.md`
5. `brain/handoffs/2026-06-05_novelty_deep_research.md` § C.1 (Kalshi→crypto-vol, arXiv 2604.01431) and the map note candidate #4.

GOAL: test whether Kalshi macro-contract repricing forecasts crypto realized vol out-of-sample for OUR
universe, and whether using it as a vol-target / regime throttle de-risks the momentum book. This is a
forecast-as-feature task; we do NOT trade Kalshi directly. The replicated paper finds the signal is
orthogonal to Fed funds futures / Treasuries / Deribit IV, with channel specificity (Fed→BTC but
regime-dependent; recession KXRECSSNBER→BTC durable OOS; CPI KXCPI→altcoins). Treat the Fed channel as
regime-dependent; prioritize KXRECSSNBER and KXCPI.

STEPS: (1) pull daily Kalshi prices+volume for KXFED, KXCPI, KXRECSSNBER (and the paper's other series)
via the Kalshi API; build the daily volume-weighted Δprobability signal. (2) Build a baseline HAR-RV
5-day-ahead realized-vol forecast for BTC/ETH/SOL. (3) Add the Kalshi Δprob as an exogenous regressor;
test OOS MSFE improvement with Clark–West (GATE: p < 0.05) and orthogonality vs fed funds futures /
Treasury / Deribit IV controls. (4) Feed the augmented vol forecast into a vol-target throttle on the live
momentum book; GATE: augmented-throttle OOS Sharpe > HAR-only-throttle and > no-throttle, with CPCV CIs.
Report the regime-dependence of the Fed channel explicitly. Findings →
`topics/momentum/research/kalshi_macro_vol_throttle_findings.md`, linked to STRATEGY_REFERENCE.
```

### Prompt 5 — Polymarket NegRisk-basket consistency live scanner + persistence measurement

```markdown
Before doing anything else, read, in order:
1. `brain/CODEX.md`
2. `brain/TODO.md`
3. `brain/COWORK.md`
4. `brain/POLYMARKET_BRAIN.md`
5. `polymarket/research/notes/market_making/strat_market_making.md` (MM hub), plus `brain/handoffs/2026-06-05_novelty_deep_research.md` § B.1 and `mm_politics_negrisk_accounting_findings`.

GOAL: build a NegRisk-basket consistency scanner and MEASURE the persistence of basket mispricings — the
strongest deterministic PM edge, corroborated by the $40M study (arXiv 2508.03474) and our own $34.6M
accounting gap. This is a measurement loop, not a trading-system build. For data artifacts, use
[[polymarket_data_manifest]] etc.; do not relink raw shards one by one. Repo invariants: uv, DuckDB over
Parquet, lowercase 0x addresses, lookahead-free.

STEPS: (1) Stream/replay all current NegRisk sets via Gamma; for each set compute Σ best-ask(YES) and the
NO-merge-implied basket cost net of gas + the 2% effective fee, using the canonical merge/split/convert
semantics in `Polymarket/neg-risk-ctf-adapter`. (2) For each detected gap (1 − Σ ask > threshold), measure
how long it PERSISTS (snapshot the book at fine intervals) — the mis-accounting thesis predicts persistence
in minutes, not ms, on multi-condition sets. (3) Estimate realized capturable return after gas/slippage at
small size, separating market-rebalancing (single-set) from combinatorial (cross-set) opportunities, and
report the share of dollars in the slow-persistence bucket a small non-latency player could realistically
take. GATE (pre-registered): a non-trivial frequency of gaps with post-cost return > 0 AND median
persistence > [X] seconds. Findings → `polymarket/research/notes/market_making/
negrisk_basket_consistency_findings.md`, linked to [[strat_market_making]]. CIs on every headline.
```

> **Draftable on request (scoped in the table, prompts not yet written):** #11 PM same-day touch/barrier with a Kronos/HAR first-passage pricer (OD reopen with a better vol input); #9 Block J LLM two-arm (fine-print arb vs slow-news, using the Halawi/FutureSearch harness); #12 copy-the-informed-minority lag-decay diagnostic. Ask and I'll write any of them.

## Cross-links

Evidence base: [[2026-06-05_novelty_deep_research]]. Extends: [[2026-06-04_state_of_the_arc_and_novelty_frontier]]. Hubs: [[STRATEGY_REFERENCE]], [[strat_market_making]], [[strat_options_delta]], [[POLYMARKET_BRAIN]], [[COWORK]], [[CODEX]].
