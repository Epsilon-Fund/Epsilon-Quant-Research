---
title: "Novelty deep research — cross-book literature + repo foundation (crypto + Polymarket)"
tags: [handoff, deep-research, foundations, novelty, literature, crypto, polymarket, cross-pollination]
created: 2026-06-05
status: living foundation note — the cited literature + open-source-repo base that the novelty map is built on. Read this for the "why / evidence"; read [[2026-06-05_novelty_frontier_map]] for the "what to do / ranked + gated + Codex prompts".
purpose: >
  A single deep-research artifact feeding the brain: the academic + quant-trading literature and
  GitHub repos behind every novel-strategy candidate across both books. Each section is mechanism +
  evidence (with URLs) + honest caveat, so a future Cowork/Codex session can act without re-searching.
relationship: >
  Sits beside [[2026-06-04_state_of_the_arc_and_novelty_frontier]] (the Polymarket state map) and is the
  evidence base for [[2026-06-05_novelty_frontier_map]] (ranked ideas + pre-registered gates + Codex prompts).
  Crypto hub: [[STRATEGY_REFERENCE]]. PM hubs: [[strat_market_making]], [[strat_options_delta]], [[POLYMARKET_BRAIN]].
  Methodology spine: [[CODEX]] § Realism calibration, [[COWORK]] § The research loop.
---

# Novelty deep research — cross-book literature + repo foundation

## Summary

- This note is the **evidence base** for the novelty hunt across both books (crypto live trading + Polymarket). It is organized by research area; each entry is **mechanism → evidence (URLs) → honest caveat**. The ranked, gated, actionable version with Codex prompts is the sibling note [[2026-06-05_novelty_frontier_map]].
- **Four high-stakes citations were independently web-verified** before writing (NegRisk $40M arb, Gómez-Cram informed minority, Kalshi→crypto-vol, Prophet Arena). They are real and accurately summarized. Other citations are canonical (López de Prado, Moreira–Muir, Liu–Tsyvinski–Wu, Avellaneda–Stoikov, BIS Crypto Carry, Halawi, Schoenegger).
- **The single most important crypto finding:** before trusting/scaling the live momentum book (reported 2.24 Sharpe over a 400-trial Optuna search), apply a **backtest-overfitting + Monte-Carlo robustness layer** (Deflated Sharpe, PBO/CSCV, stationary-block bootstrap, Hansen SPA, regime-aware MC). This is cheap, uses data we already have, and directly answers Justin's "how overfitted is my stuff" question. See § A.
- **The single most important cross-book finding:** Kalshi macro-contract repricing **forecasts crypto realized volatility orthogonally to Fed funds futures / Treasuries / Deribit IV** ([arXiv 2604.01431](https://arxiv.org/abs/2604.01431)). We already pull prediction-market data — this becomes a vol-target/regime throttle that *de-risks the momentum book*. See § C.
- **The strongest Polymarket structural edge** is deterministic, not predictive: **NegRisk-basket / combinatorial consistency** persists because participants mis-account merge/split/redeem — exactly the $34.6M gap we found locally, now corroborated by an independent **$40M** academic study ([arXiv 2508.03474](https://arxiv.org/abs/2508.03474)). See § B.
- **Discipline that governs all of it (unchanged):** calibration/forecast-accuracy ≠ net-of-cost profit (Briola caveat + Prophet Arena's GPT-5 sub-break-even result); every candidate is gated on **realized net PnL on a holdout**, not R²/Brier. See [[CODEX]] § Realism calibration.

---

## A. Crypto backtest-overfitting + Monte-Carlo robustness toolkit

**Why this is section A.** The live momentum book reports ~77% CAGR / 2.24 Sharpe, but it is the **max of a 400-trial-per-fold Optuna search** over a survivor-selected 6-coin universe. None of the existing validation (WF, CPCV) tests *whether the selection itself is overfit*. This toolkit does, cheaply, on data we already have. Framing point that resolves the "isn't CPCV enough?" question: **CPCV gives the OOS distribution of one already-chosen config; CSCV/PBO audits the selection process.** They are complementary ([CPCV vs CSCV, MQL5](https://www.mql5.com/en/articles/21603)).

### A.1 Deflated Sharpe Ratio (DSR) + Probabilistic Sharpe Ratio (PSR)
Bailey & López de Prado. Deflates the observed Sharpe by the **expected maximum Sharpe achievable by luck across N trials**, correcting for short track length, skew, and fat tails.
- SR estimator stdev (non-normal): `σ_SR = sqrt( (1 − γ₃·SR + ((γ₄−1)/4)·SR²) / (T−1) )` (γ₃ skew, γ₄ kurtosis, T obs).
- `PSR(SR*) = Φ((SR − SR*) / σ_SR)`; DSR uses `SR* = E[max]` across trials.
- `E[max] ≈ σ_SR_trials · [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(Ne))]`, γ = Euler–Mascheroni ≈ 0.5772.
- Crucial input: the **stdev of Sharpes across the Optuna trials**, and **effective independent N** (TPE trials are highly correlated — use `N_eff = p + (1−p)·M` from avg pairwise trial-return correlation `p`, or ONC clustering). 400 naive trials over-deflates.
- Gate: DSR ≥ 0.95. **MinTRL** falls out of the same module ("have I observed enough?").
- Papers: [DSR PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) · [SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551). Code: [rubenbriones/Probabilistic-Sharpe-Ratio](https://github.com/rubenbriones/Probabilistic-Sharpe-Ratio) (best reference impl), [esvhd/pypbo](https://github.com/esvhd/pypbo), `mlfinlab`.

### A.2 Probability of Backtest Overfitting (PBO) via CSCV
Bailey, Borwein, López de Prado, Zhu. Probability the in-sample-best config ranks **below median out-of-sample** — the direct measure of "is my Optuna selection overfit?" Requires keeping **every trial's return series** (not just the winner). Logit of OOS relative rank; PBO = fraction of CSCV splits with negative logit. Gate: PBO < 0.5 (ideally < 0.2).
- Paper: [SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253). Code: [esvhd/pypbo](https://github.com/esvhd/pypbo) (`pbo.pbo(...)`), R [mrbcuda/pbo](https://github.com/mrbcuda/pbo).

### A.3 Monte-Carlo / data-snooping tests
- **Hansen SPA / White Reality Check / StepM / MCS** — does the *best* config among the whole search beat buy-hold after accounting for the search? SPA > White's RC (re-centered, more powerful); StepM identifies *which* configs are genuinely superior (FWER). **Most production-ready library:** `arch` (`arch.bootstrap.SPA`, `StepM`, `MCS`) — [docs](https://bashtage.github.io/arch/multiple-comparison/multiple-comparison_examples.html), [repo](https://github.com/bashtage/arch). Paper: [Hansen SPA / Hsu-Kuan, SSRN 685361](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=685361).
- **Stationary / block bootstrap** (Politis–Romano) — CIs for Sharpe/Calmar/maxDD respecting autocorrelation. `arch` (`StationaryBootstrap`, `optimal_block_length`); [recombinator](https://github.com/InvestmentSystems/recombinator).
- **Monte-Carlo synthetic price paths (Masters bar permutation)** — shuffle/block-resample log returns, rebuild synthetic OHLC, re-run the strategy → null Sharpe distribution; p = fraction of synthetic runs beating real. **Crypto-correct version: regime-aware block bootstrap** — resample bars *within HMM regimes* (we already classify them), since GBM surrogates understate crypto tails. [Build Alpha MCPT](https://www.buildalpha.com/monte-carlo-permutation/), [Masters PDF](https://evidencebasedta.com/montedoc12.15.06.pdf).
- **Training-process / signal-permutation test (Masters)** — permute the target/signal then **re-run the whole Optuna selection** on permuted data; the distribution of best-permuted Sharpes is the honest null that accounts for the search itself. Highest fidelity, highest engineering cost. Book: [Masters, *Permutation and Randomization Tests for Trading System Development*](https://www.amazon.com/Permutation-Randomization-Trading-System-Development/dp/B084QLXFKW).

### A.4 Minimum Backtest Length + multiple-testing haircut
- **MinBTL** ≈ `2·ln(N_eff) / E[max Sharpe]²` years. Headline: with ~5 years of data you should try **no more than ~45 independent configs** or an IS Sharpe ≈ 1 maps to OOS ≈ 0. 400 trials is far past 45 *unless* effective-N is small — so compute effective-N. Paper: [Pseudo-Mathematics, AMS](https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf). (Verify the constant against the primary PDF before hard-coding; some write-ups present it as an upper bound.)
- **Harvey–Liu haircut** — the Sharpe haircut for multiple testing is **nonlinear** (high Sharpes lightly penalized, marginal ones gutted); use **BHY (FDR)**, not a flat 50%. Feed **effective-N**, not 400. Paper: [SSRN 2345489](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2345489) · [Duke code](https://people.duke.edu/~charvey/backtesting/).

### A.5 Crypto-specific overfitting gotchas
- **Regime non-stationarity is the dominant risk** — stratify CPCV folds + MC blocks by HMM regime; check Sharpe stability per-regime, not pooled.
- **Survivorship in the coin universe** — dead-coin exclusion can inflate backtested returns 200–400%; the 6-coin universe is implicitly survivor-selected. Backtest on a point-in-time universe or at least acknowledge the inflation. ([StratBase](https://stratbase.ai/en/blog/survivorship-bias-crypto)).
- **Look-ahead in funding/OI features** — funding/OI are timestamped at settlement; lag ≥1 settlement interval; verify with the signal-permutation test.
- **Optuna overfitting the CPCV path distribution itself** — if the objective is computed over the 105 CPCV paths, 400 trials fit *their* noise. Defenses: a separate regime-distinct holdout Optuna never touches; run the CSCV/PBO audit on the trials; training-process permutation.

### A.6 Library summary
| Repo / package | Implements |
|---|---|
| [bashtage/arch](https://github.com/bashtage/arch) | Hansen SPA, White RC, StepM, MCS, StationaryBootstrap, optimal_block_length |
| [esvhd/pypbo](https://github.com/esvhd/pypbo) | PBO/CSCV, PSR, MinTRL, MinBTL, DSR |
| [rubenbriones/Probabilistic-Sharpe-Ratio](https://github.com/rubenbriones/Probabilistic-Sharpe-Ratio) | PSR, σ_SR (non-normal), expected-max-SR, num_independent_trials, DSR |
| `mlfinlab` (Hudson & Thames) | DSR, haircut Sharpe, MinTRL, CPCV, CSCV |
| [recombinator](https://github.com/InvestmentSystems/recombinator) | IID/block/stationary bootstrap + optimal block length |
| Masters book (C++) | training-process permutation, MCPT, bar permutation |

---

## B. Polymarket novelty frontier

### B.1 NegRisk-basket / combinatorial consistency at scale (strongest structural edge)
**Mechanism.** In a NegRisk set exactly one of N mutually-exclusive YES resolves to $1, so Σ YES ≈ 1. Two deterministic arbs: (i) **convert/dutch-book** when Σ YES < 1 (buy basket < $1, guaranteed $1); (ii) **NO-merge** via the NegRiskAdapter (burn N−1 NO tokens → collateral + complementary YES). Prediction-market analog of sportsbook overround harvesting / Hanson LMSR coherence. Collateral-efficient: NegRisk needs only 1.0 collateral vs Σ-prices.
**Evidence (verified).** [Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets, arXiv 2508.03474](https://arxiv.org/abs/2508.03474) (AFT 2025): **~$40M extracted Apr-2024→Apr-2025 across 17,218 conditions; market-rebalancing arb = 73% of profit from 8.6% of opportunities**; persistence attributed to participants mis-modeling the CTF convert path — corroborates our **$34.6M** local accounting gap ([[mm_politics_negrisk_accounting_findings]]). Mechanics: [Polymarket neg-risk-ctf-adapter docs](https://github.com/Polymarket/neg-risk-ctf-adapter). Independent replication: [Navnoor Bawa, "$29M extracted"](https://medium.com/@navnoorbawa/negrisk-market-rebalancing-how-29m-was-extracted-from-multi-condition-prediction-markets-2f1f91644c5b). Live detector repo: [ivanzzeth/polymarket-go-gamma-client (find-negrisk-opportunities)](https://pkg.go.dev/github.com/ivanzzeth/polymarket-go-gamma-client/examples/find-negrisk-opportunities).
**Caveat.** This is partly a latency game on liquid politics; the durable small-player slice is **multi-condition sets where mis-accounting makes gaps persist in minutes, not ms**, and the residual edge after gas + 2% effective fee. Our local edge belongs to *measuring persistence*, not winning a ms race.

### B.2 Neutral market-making in slow (politics/sports) markets
**Mechanism.** Adapt Avellaneda–Stoikov to a [0,1] bounded, resolution-terminating contract: σ must collapse near the boundaries (scale by p(1−p)); a terminal jump to {0,1} replaces Brownian diffusion; inventory penalties asymmetric near boundaries. Add Glosten–Milgrom-style toxicity inference — *pull quotes when hit by large/informed flow* — since you can't diffuse, only learn from who's trading. Polymarket pays explicit **liquidity rewards** (depth-near-mid × uptime) on top of spread; rebates can flip MM economics positive in low-toxicity venues.
**Why it might live where crypto MM died.** Our crypto neutral MM died to *structural* adverse selection (fast informed flow). Slow markets have the opposite microstructure: flow is ~84-96% uninformed (see B.4), news arrives in watchable chunks (debates, games, court dates), and first/only makers capture an outsized reward share.
**Evidence.** [Avellaneda–Stoikov / Guéant survey, arXiv 1605.01862](https://arxiv.org/pdf/1605.01862); [Optimal Quoting under Adverse Selection & Price Reading, arXiv 2508.20225](https://arxiv.org/pdf/2508.20225); [Bayesian Market Maker for prediction markets (Brahma/Das/Magdon-Ismail/Sanmay)](https://people.cs.vt.edu/~sanmay/papers/bmm-ec.pdf); [Comparing Prediction Market Structures (Othman & Sandholm)](https://people.cs.vt.edu/~sanmay/papers/predmarkets.pdf); [Optimal make-take fees, arXiv 1805.02741](https://arxiv.org/pdf/1805.02741). Repos: [warproxxx/poly-maker](https://github.com/warproxxx/poly-maker) (two-sided AS-style PM MM bot), [PolyScripts/polymarket-market-maker-bot](https://github.com/PolyScripts/polymarket-market-maker-bot/). This is the existing standing live loop [[2026-06-03_politics_negrisk_live_loop]], reframed as a *novel neutral-MM test* not a reproduce-the-winners test.

### B.3 Block J — LLM resolution-criteria + slow-news scanning (never attempted)
**Mechanism & the central finding.** Retrieval-augmented LLM forecasters. The honest 2025-26 state of the art directly answers "is this dead too?":
- **Calibration is solved; profitability is not.** [Prophet Arena, arXiv 2510.17638](https://arxiv.org/abs/2510.17638) (verified): even **GPT-5 beats the market's Brier/calibration yet fails break-even (avg return < 1)** net of spread — the cleanest confirmation of our efficient-pricing thesis.
- **But edge survives where prices are stale or fine-print-dependent.** [Halawi et al. 2024, arXiv 2402.18563](https://arxiv.org/abs/2402.18563) (code: [dannyallover/llm_forecasting](https://github.com/dannyallover/llm_forecasting)) find LLM forecasts most additive on **slow-moving, news-light, under-traded** questions. [FutureSearch "Can AI Beat Kalshi?"](https://futuresearch.ai/blog/kalshi-trader-case-study/) trades only 24/153 markets with ≥2% edge — a working template. [Schoenegger & Park, *Wisdom of the Silicon Crowd*, Science Advances 2024](https://www.science.org/doi/10.1126/sciadv.adp1528) (12-LLM ensemble rivals 925-human crowd).
**The under-exploited angle = resolution-criteria reading, not forecasting.** Most mispricing on neglected markets is because nobody read the exact resolution source. An LLM that ingests resolution fine print + primary source can catch *deterministic* mismatches — closer to arbitrage than prediction.
**Caveat.** LLMs still lose to elite human superforecasters (o3 Brier 0.135 vs SF 0.023, [arXiv 2507.04562](https://arxiv.org/html/2507.04562v3)); the edge is *vs the market price on neglected questions*, never vs liquid prices.

### B.4 Copy the informed minority (on-chain)
**Mechanism & strongest single citation.** [Prediction Market Accuracy: Crowd Wisdom or Informed Minority? (Gómez-Cram, Guo, Jensen, Kung), SSRN 6617059](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6617059) (verified; $13.76B / 1.72M accounts / ~210k markets): **~3.14% of traders generate the bulk of price discovery and >30% of profits; the majority funds them.** Decisive for copy-viability: **skilled Polymarket traders persist OOS 44%** vs only 10% for skilled mutual funds — skill is far stickier here. A 1pp rise in skilled net buying → ~8bp rise in P(correct outcome).
**Caveat (make-or-break).** You can't get their fill price — the edge must survive **copy-lag slippage**; the primary diagnostic is the edge-vs-seconds-of-delay decay curve. Leaderboards also rotate 30-40% per 60-90 days. Tooling: [Polycool](https://polymark.et/product/polycool). This re-merges with our standing copytrade thread.

### B.5 Cross-platform Polymarket vs Kalshi
**Mechanism.** Same event, two venues, different prices + different cost structures (Kalshi: CFTC, USD, per-contract fees; PM: USDC, gas + ~2%). **Polymarket leads Kalshi in price discovery** ([Price Discovery and Trading in Modern Prediction Markets, SSRN 5331995](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331995)), so read PM's move and trade the lagging Kalshi quote.
**Caveat.** ICE's 2025 Polymarket investment compressed the window toward ms on liquid contracts; the durable small-player edge is in **less-liquid/slower pairs** and **resolution-criteria mismatches** (the two venues sometimes word "the same" market differently — a structural, not speed, edge). Repo: [ImMike/polymarket-arbitrage](https://github.com/ImMike/polymarket-arbitrage).

### B.6 First-mover liquidity in new/thin markets
**Mechanism.** Be the first resting maker in a freshly-listed market: set the spread, earn it on early (least-informed) flow, capture an outsized share of liquidity rewards before competitors notice. Already scoped MERITS-LIVE-STAGE-1 ([[mm_first_mover_liquidity_scope_findings]]); ties to B.2's boundary-aware quoting for the early-repricing risk. Evidence on new-listing MM/rebate economics: [Informed Trading and Maker-Taker Fees](https://www.researchgate.net/publication/280320141_Informed_Trading_and_Maker-Taker_Fees_in_a_Low-Latency_Limit_Order_Market).

> **Cross-cutting through-line for all PM edges:** prices are made accurate by a ~3% informed minority funded by a ~84-96% uninformed majority. Every surviving edge is a way to sit on the informed side of that flow *without forecasting better than the price*: MM/first-mover monetize uninformed flow as counterparty; NegRisk/cross-platform monetize structural mis-accounting + lead-lag (deterministic, hence persistent); copy/LLM-fine-print *become* the informed minority cheaply.

---

## C. Cross-pollination (highest-novelty, reuses existing infra)

### C.1 Kalshi macro repricing → crypto vol-target / regime throttle (best cross-book fit)
**Mechanism & verified anchor.** [Do Prediction Markets Forecast Cryptocurrency Volatility? Evidence from Kalshi Macro Contracts (Mohanty & Krishnamachari), arXiv 2604.01431](https://arxiv.org/abs/2604.01431): **daily volume-weighted probability changes in Kalshi macro contracts forecast 5-day-ahead crypto realized vol, orthogonal to Fed funds futures, Treasuries, and Deribit IV.** Channel specificity: Fed-rate (KXFED) → BTC vol (t=3.63 in-sample, but **regime-dependent** — gains in the 2024-25 cutting cycle, reverse after); **recession (KXRECSSNBER) → BTC, most reliable OOS (MSFE ratio 0.979, Clark-West p=0.020)**; **CPI (KXCPI) → altcoin vol (ETH MSFE 0.959 p=0.010, SOL p=0.048)**. First-stage R² vs conventional instruments only 2.3%/7.5% → genuinely new information. The paper's own "Practical Implications" frames it as a **volatility-managed-portfolio input** — exactly our use case. Corroboration: the [Fed's own Kalshi study](https://www.federalreserve.gov/econres/feds/files/2026010pap.pdf) (Kalshi beats fed funds futures + Bloomberg consensus on rate + CPI).
**Use.** Not traded directly — a macro-uncertainty feature in the HMM regime classifier and a vol-target throttle that cuts momentum gross when KXRECSSNBER/KXCPI repricing predicts a vol spike. Reuses the PM data pipeline; **de-risks the existing book** (highest-value use of a forecast you can't trade).
**Caveat (paper's own).** Fed channel is regime-dependent — don't hard-code it; recession/CPI channels are the durable ones.

### C.2 Superior vol-path model → PM same-day touch / barrier (the one residual PM crypto pricing edge)
**Mechanism.** PM crypto *terminal* digitals (daily up/down, price-target) are efficiently priced — we proved that. But **path-dependent** structures (one-touch / first-passage / "will BTC touch $X intraday") depend on the full vol path and jump intensity, not the terminal distribution. A genuinely better intraday vol/path model (jump-augmented HAR-RV, regime-conditional σ from the HMM, or Kronos synthetic K-line paths) → Monte-Carlo'd first-passage probability vs the PM digital. First-passage under jump-diffusion is well-developed ([Double-barrier first-passage, jump-diffusion](https://mediatum.ub.tum.de/doc/1114133/1114133.pdf)).
**Honest read.** **Not** the 5-min market — that's Chainlink-snapshot-resolved, thin, and dominated by last-5-second order-flow/oracle-latency bots (an execution game, not forecasting). The edge is in **multi-hour to same-day touch/barrier markets**. Narrow capacity. This is the principled reopen of the same-day Arm-T / touch cluster ([[od_same_day_crypto_pricing_gate_findings]]) with a *better vol input*, gated exactly as before.

---

## D. Brand-new domains that fit the WF/CPCV engine

### D.1 Crypto perpetual funding / basis carry (standalone sleeve)
**Mechanism.** Long spot / short perp of equal notional; PnL ≈ funding collected + basis convergence − fees. Persistent positive funding (structural in bull phases) is harvested by the short-perp leg. Funding extremes also double as a **contrarian de-risk gate** overlaying the directional book.
**Evidence.** [Schmeling, Schrimpf, Todorov, "Crypto Carry," BIS WP 1087 / *Management Science* 2024](https://www.bis.org/publ/work1087.pdf): carry is large, volatile, **not** explained by fundamentals → a genuine segmentation/limits-to-arbitrage premium. [The Crypto Carry Trade (Christin et al.)](https://www.andrew.cmu.edu/user/azj/files/CarryTrade.v1.0.pdf). [Risk and Return of Funding Rate Arbitrage on CEX/DEX](https://www.sciencedirect.com/science/article/pii/S2096720925000818): ~10-20% APY favorable / 0-5% neutral; cross-venue divergence adds 3-5% annualized with occasional 20%+ spikes.
**Caveat.** Risk is **funding-regime risk, not direction** — funding flips negative exactly in violent deleveraging (when momentum also bleeds), so it diversifies in normal regimes but is **not a tail hedge**; one study found forced exits in ~95% of opportunities before full convergence → sizing/unwind discipline is the whole game. Settlement-interval mismatch (8h CEX vs 1h Hyperliquid) must be modeled.

### D.2 Deribit variance-risk-premium harvest (regime-gated)
**Mechanism.** Bitcoin VRP averages **~66%/yr — far above equities/FX/commodities**. Systematic short-variance (delta-hedged straddle/strangle), risk-reversal selling (call-skew in bull phases), or calendar/term-structure trades. VRP *spikes before* big moves → itself a regime signal feeding the trend book.
**Evidence.** [Risk Premia in the Bitcoin Market, arXiv 2410.15195](https://arxiv.org/pdf/2410.15195); [The Bitcoin VIX and Its VRP (Alexander & Imeraj)](https://www.pm-research.com/content/iijaltinv/23/4/84); [Du 2025, *Pricing Crypto Options With Vol-of-Vol*, J. Futures Markets](https://onlinelibrary.wiley.com/doi/10.1002/fut.70029) (vol-of-vol makes the VRP sign time-varying → harvest must be *conditional*, which is where the regime classifier earns its keep); [Deribit Insights: four years of vol regimes](https://insights.deribit.com/industry/bitcoin-options-finding-edge-in-four-years-of-volatility-regimes/).
**Caveat.** Short-vol co-crashes with carry and partially with momentum drawdowns; cap allocation, size for fat tails, gate on vol-of-vol regime.

### D.3 What's NOT your game (documented dead-ends)
- **Pure CEX-DEX / latency arb** — a latency/MEV oligopoly (~$233.8M to 19 searchers, Aug-2023→Mar-2025; sub-second subslot games). [arXiv 2507.13023](https://arxiv.org/pdf/2507.13023). The only survivable variant is **cross-venue funding stat-arb** (hours-to-days), which collapses into D.1.
- **DeFi AMM LP yield** — dominated by **Loss-Versus-Rebalancing** (a closed-form short-gamma cost scaling with σ²; [Milionis et al., arXiv 2208.06046](https://arxiv.org/abs/2208.06046)). Net edge only in stable-stable / low-σ high-fee pairs. Mostly a short-straddle dressed as yield.

---

## E. Foundation models & modern ML — honest, repo-by-repo

**Governing principle (Briola caveat).** These are **forecast engines, not strategies**; high forecast accuracy ≠ tradeable net of cost ([Briola et al., *Deep LOB Forecasting*, arXiv 2403.09267](https://arxiv.org/abs/2403.09267)). Gate every FM output through a cost-aware backtest / meta-label sizer before it touches sizing.

| Model | What it is | Crypto-relevance / honest read | Repo |
|---|---|---|---|
| **Kronos** | Decoder-only FM pre-trained natively on OHLCV K-lines (tokenized candlesticks). Only FM built for candlesticks. Emits P(next-24h vol > recent) and **synthetic K-line paths**. | **Most relevant.** Author-reported +93% RankIC / 58-65% directional; demo has no costs/sizing (their own README). High ceiling *if* fine-tuned + hard-gated. Best use = **path generator for C.2** + vol-regime feature, not directional black box. | [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) · [arXiv 2508.02739](https://arxiv.org/abs/2508.02739) |
| **TabPFN-TS** | Univariate forecasting as tabular regression via pretrained TabPFN-v2. <20M params, CPU-fast, zero-shot. | Ranked #1 GIFT-EVAL (Jan 2025); tiny, no-GPU. **The cheap strong baseline Kronos must beat.** | [PriorLabs/tabpfn-time-series](https://github.com/PriorLabs/tabpfn-time-series) |
| **Chronos / Chronos-Bolt** | T5 tokenized TS, probabilistic. | Consistently near-best on noisy-periodic; good vol baseline. | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) |
| **TimesFM** (Google) | Patch decoder-only. | Strong general zero-shot; no native OHLCV/microstructure. Baseline to beat. | [google-research/timesfm](https://github.com/google-research/timesfm) |
| **Moirai-2** (Salesforce) | Masked-encoder, native multivariate. | Multivariate attractive for cross-asset crypto; mid-pack on noisy data. | [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) |
| **Lag-Llama** | Decoder-only probabilistic. | Good intervals, weaker point accuracy; useful for vol/quantiles. | [time-series-foundation-models/lag-llama](https://github.com/time-series-foundation-models/lag-llama) |
| **PatchTST / TimeMixer** | Supervised SOTA (you train them). | Best fine-tuned on your own bars. | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) |

**Recommended FM pipeline:** FM produces return/vol forecast → feed as a *feature* into a meta-label sizer (López de Prado), never as a direct signal → cost-aware purged-CV backtest measuring net PnL → only then size. Start with **Kronos fine-tuned on our universe**, benchmark vs **TabPFN-TS / Chronos**; kill it if it can't beat a vol-scaled EMA net of costs.

**Meta-labeling (López de Prado).** Keep the interpretable rules as the **primary** model (decides side); train a **secondary** classifier on {ADX, breadth, dispersion-confidence, funding extreme, realized vol, DVOL, on-chain flow, FM forecast} to predict P(primary trade correct) → size & filter. Reduces turnover, fuses all gates into one sizing layer. Insist on **purged/embargoed CV** to avoid the leakage that makes meta-labeling look magic. [Meta-Labeling overview](https://en.wikipedia.org/wiki/Meta-Labeling) · [counterpoint: not a silver bullet](https://www.quantconnect.com/forum/discussion/14706/why-meta-labeling-is-not-a-silver-bullet/).

**RL & LLM-agents — trust map.** RL for **your own execution** (order placement / impact minimization) is mature and low-hype ([arXiv 2411.06389](https://arxiv.org/abs/2411.06389)). **LLM-agent *alpha* claims are mostly hype** — StockBench / LiveTradeBench / AI-Trader find LLM agents fail to beat simple baselines OOS, with benchmark contamination conflating reasoning with recall ([arXiv 2605.28359](https://arxiv.org/html/2605.28359v1)). Use LLMs for **research automation** (news/event extraction feeding the PM + macro pipelines), not autonomous decisions.

### E.1 Classical crypto-factor literature (build-on-momentum)
- **Vol-managed portfolios** (Moreira–Muir, *JF* 2017, [NBER w22208](https://www.nber.org/system/files/working_papers/w22208/w22208.pdf)) — largest alpha is on *momentum*; crypto-specific risk-managed momentum lifts Sharpe ~1.12→1.42 ([FRL 2025](https://www.sciencedirect.com/science/article/abs/pii/S1544612325011377)). Robustness caveat: [Cederburg et al., *JFE* 2020](https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X).
- **Residual / idiosyncratic momentum** (Blitz, Huij, Martens) — strip BTC(+ETH) beta, rank on residual-return/residual-vol; ~doubles risk-adjusted profit, avoids crashes. We're ~70% there in `xs_momentum`.
- **Crypto factor zoo** (Liu, Tsyvinski, Wu, *JF* 2022, [NBER w25882](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)) — 3-factor (market, size, momentum) prices the cross-section; **CTREND** trend factor survives costs in liquid coins ([JFQA 2024](https://jfqa.org/2024/09/20/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns/)). Short-term reversal works *only in illiquid coins*.
- **On-chain — robust subset only.** Skip MVRV/SOPR (tiny N, overfit-prone cycle-timers). Keeper: **stablecoin/exchange netflows** predict intraday return + vol ([arXiv 2411.06327](https://arxiv.org/abs/2411.06327)) — use as a **vol/risk gate**, needs a vendor (Glassnode/CryptoQuant).

### E.2 Repos worth mining
- [freqtrade](https://github.com/freqtrade/freqtrade) + freqtrade-strategies + FreqAI; [Hummingbot](https://github.com/hummingbot/hummingbot) + Quants Lab; [microsoft/qlib](https://github.com/microsoft/qlib) (ships a Kronos fine-tune+backtest example); [awesome-systematic-trading](https://github.com/wangzhe3224/awesome-systematic-trading); [awesome-quant](https://github.com/wilsonfreitas/awesome-quant).

---

## F. Verification log

Web-verified before writing (real + accurately summarized): NegRisk $40M arb ([2508.03474](https://arxiv.org/abs/2508.03474)); Gómez-Cram informed minority ([SSRN 6617059](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6617059)); Kalshi→crypto-vol ([2604.01431](https://arxiv.org/abs/2604.01431)); Prophet Arena ([2510.17638](https://arxiv.org/abs/2510.17638)). Canonical/unverified-but-standard: López de Prado DSR/PBO, Moreira–Muir, Liu–Tsyvinski–Wu, Avellaneda–Stoikov, BIS Crypto Carry, Halawi, Schoenegger, Kronos (already in-brain). One numeric caveat: the MinBTL constant `2·ln N / E[max]²` is quoted as an approximate bound — verify against the primary AMS PDF before hard-coding.

## Sources

Inline throughout. Primary anchors: [arXiv 2508.03474](https://arxiv.org/abs/2508.03474) · [SSRN 6617059](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6617059) · [arXiv 2604.01431](https://arxiv.org/abs/2604.01431) · [arXiv 2510.17638](https://arxiv.org/abs/2510.17638) · [BIS WP 1087](https://www.bis.org/publ/work1087.pdf) · [NBER w25882](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf) · [NBER w22208](https://www.nber.org/system/files/working_papers/w22208/w22208.pdf) · [DSR/SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) · [PBO/SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253).
