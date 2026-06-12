---
title: "ML for Prediction Markets — Literature Synthesis and Strategy Layout"
tags: [literature, polymarket, microstructure, ofi, market-making, research]
created: 2026-05-23
status: active
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
audience: cowork-handoff
related: [[dali_factor_construction]], [[block_k_maker_options_research]], [[2026-06-05_novelty_deep_research]], [[2026-06-05_novelty_frontier_map]]
parent: dali
---

# ML for Prediction Markets — Literature Synthesis and Strategy Layout

> Hub: [[COWORK]]
> **Purpose.** Consolidated handoff document covering academic foundations
> for Polymarket short-horizon ML strategies, current findings on the Dali
> project, and the structured roadmap for next-stage research. Designed for
> a deep-research follow-up agent to extend with primary-source verification
> and additional gap analysis.
> Table terms: [[polymarket_table_dictionary]]

## Summary

This synthesis collects the academic and practical foundations for Polymarket short-horizon ML, maker, and OFI-style strategy work. It separates taker directional prediction from maker liquidity provision and anchors both in microstructure literature. The note remains an active research reference for strategy layout and literature-backed next steps.

---

## 1. Strategy Universe and Where Edge Lives

Two distinct strategy classes have emerged as candidates for Polymarket:

**Taker strategies** rely on directional prediction. You consume liquidity
(pay taker fees) when your signal is confident enough to cross the spread.
Edge source: better prediction than market consensus. Critical constraints:
fee economics by category (0% geopolitics to 1.8% crypto), signal must be
strong enough to overcome round-trip costs.

**Maker strategies** rely on liquidity provision. You rest passive orders
(pay 0% fees, may earn rebates) and profit from capturing spread minus
adverse selection. Edge source: providing immediacy to uninformed flow
while avoiding informed counterparties. Critical constraints: adverse
selection management, inventory risk, quote staleness during news arrival.

The two are not competing — they're complementary strategies addressing
different market opportunities. Maker strategies generally open more
markets (cost economics don't constrain by category) but require more
sophisticated risk management.

---

## 2. Foundational Literature

### 2.1 Microstructure Theory — Why These Strategies Should Work

**Stoll (1978).** Market maker's spread = inventory cost + adverse selection
cost + order processing cost. Establishes that liquidity provision is a
risk-bearing service with definable components. Underlies all market making
research.

**Glosten & Milgrom (1985).** Optimal spread = compensation for adverse
selection from informed traders. Shows that bid-ask spread is the market
maker's protection against being picked off by participants with private
information. Critical insight for Polymarket: spread width tells you about
information asymmetry.

**O'Hara (1995).** *Market Microstructure Theory.* Cambridge University
Press. Foundational textbook synthesizing the field.

**Avellaneda & Stoikov (2008).** *High-Frequency Trading in a Limit Order
Book.* Quantitative Finance. The canonical closed-form solution for
optimal market making with inventory aversion. Gives concrete formulas
for setting bid/ask given current inventory, volatility, risk aversion,
time horizon. If pursuing maker strategy on Polymarket seriously, this
is the framework to start from.

### 2.2 Order Flow Imbalance — The Core Signal

**Cont, Kukanov & Stoikov (2014).** *The Price Impact of Order Book Events.*
Journal of Financial Econometrics. The foundational empirical paper on OFI.

Key findings:
- OFI explains short-term price changes with R² around 65% on equity data
- TFI (trade-only signal) achieves only ~32% R² on same data
- The relationship is near-linear: `Δprice ≈ β × OFI`
- Coefficient β is stable across stocks when normalized by tick size and depth
- OFI captures information unavailable to fill-only analysis (limit order
  placements and cancellations carry directional information)

Critical implication for Dali: weak TFI does not condemn OFI. They are
qualitatively different signals; TFI is a degraded subset of OFI.

**Cont (2011).** *Statistical Modeling of High-Frequency Financial Data.*
IEEE Signal Processing Magazine. Earlier theoretical work establishing
how to model order book dynamics statistically.

**Lipton, Pesavento & Sotiropoulos (2013).** *Trade arrival dynamics and
quote imbalance in a limit order book.* Extended OFI with depth-weighted
formulations showing further improvement.

### 2.3 Deep Learning on Order Books

**Sirignano & Cont (2019).** *Universal features of price formation in
financial markets: Perspectives from deep learning.* Quantitative Finance.

Key findings:
- Studied 115 Nasdaq stocks at granular order book level
- A single universal model trained on the pool generalizes across stocks
- Order-flow-to-price relationship is structural and stationary
- Implication: insights transfer across instruments — relevant for
  applying equity research to Polymarket

**Kolm, Turiel & Westray (2023).** *Deep Order Flow Imbalance.*
Mathematical Finance.

Key findings:
- Neural nets trained on OFI features outperform models trained on raw
  order books
- "Hand-crafted features that are proven useful" + tree/NN models > raw
  data + complex models
- Multiple-horizon forecasting works; horizons 1s-30s show clearest signal

Implication for Dali: feature engineering matters more than model
complexity. Start with linear/logistic models on well-constructed OFI
features.

**Zhang, Zohren & Roberts (2019).** *DeepLOB: Deep Convolutional Neural
Networks for Limit Order Books.* IEEE TSP. Established the CNN+LSTM
architecture as a strong baseline. Often cited as the academic state
of the art before transformer-based approaches.

**Briola, Bartolucci & Aste (2024).** *Deep Limit Order Book Forecasting:
A Microstructural Guide.* arXiv:2403.09267. Released the LOBFrame
open-source code.

**Critical caveat from this paper:** high forecasting accuracy does NOT
automatically translate to profitable trading signals after costs. Even
state-of-the-art models achieving 70%+ directional accuracy show zero or
negative net returns in many conditions after realistic transaction
costs and execution modeling.

This is the most important paper to cite when discussing the gap between
ML research and tradeable strategies.

### 2.4 Prediction Markets — Theoretical Foundation

**Storkey (2011).** *Machine Learning Markets.* arXiv:1106.4509. Shows
that prediction markets with appropriate utility functions implement
product-of-experts and mixture-of-experts model combination as equilibrium
pricing. Inference in probabilistic models is equivalent to market
dynamics. Foundational: it establishes that beating a prediction market
means having a better posterior than the collective.

**Hu & Storkey (2014).** *Multi-period Trading Prediction Markets with
Connections to Machine Learning.* arXiv:1403.0648. Extends Storkey's
framework to risk-measure-based agents and market makers. Shows that
even selfish agents collectively optimize a global probabilistic objective.

**Abernethy, Chen & Wortman Vaughan (2011).** *An optimization-based
framework for automated market-making.* Regret bounds for market makers,
foundation for LMSR analysis.

**Hanson (2003).** *Combinatorial Information Market Design.* Origin of
the Logarithmic Market Scoring Rule (LMSR). Polymarket uses CLOB rather
than LMSR; this informs the contrast.

### 2.5 Prediction Markets — Empirical

**Reichenbach & Walther (2025).** *Exploring Decentralized Prediction
Markets: Accuracy, Skill, and Bias on Polymarket.* SSRN 5910522.
Analyzed 124 million Polymarket trades. Empirical baseline for accuracy
and bias profiles.

**Clinton & Huang (2025).** Vanderbilt working paper on Polymarket/Kalshi
accuracy critique. Useful counterweight to overly optimistic accuracy
claims.

**brier.fyi** and **Calibration City** — ongoing public tracking of
Polymarket calibration. Brier scores around 0.09 aggregate, 0.18 in some
analyses. Reference points for assessing model improvements.

### 2.6 Behavioral / Documented Biases

**Snowberg & Wolfers (2010).** *Explaining the Favorite-Longshot Bias:
Is It Risk-Love or Misperceptions?* Journal of Political Economy. Markets
underweight low-probability events; same pattern in horse racing and
prediction markets.

**Thaler & Ziemba (1988).** *Anomalies: Parimutuel Betting Markets.*
JEP. Classic survey of betting market inefficiencies. Direct transferable
to Polymarket.

---

## 3. The Gap Between Literature and Tradeable Strategy

Academic papers establish:
- OFI is the right signal primitive ✓
- Statistical predictive relationships exist ✓
- Signals generalize across instruments ✓
- Multiple horizons show clearest information at 1s-300s ✓

Academic papers do NOT solve:
- Trading rules from the signal
- Position sizing
- Cost-adjusted edge after fees, spread, slippage
- Entry/exit logic
- Execution latency handling
- Risk management

Approximately 60% of the work is foundational research. The remaining
40% is strategy construction, and it's not in the papers. This is the
gap Dali is attempting to bridge for Polymarket specifically.

The Briola et al. (2024) caveat is the single most important point:
papers showing R² of 0.6+ or accuracy of 70%+ often produce zero or
negative Sharpe after costs. Forecasting accuracy ≠ trading profit.

---

## 4. Polymarket-Specific Considerations

### 4.1 Cost Structure (Updated March 2026)

Polymarket uses a dynamic taker-fee model with the formula:
`fee = C × 0.03 × p × (1-p)` where C is the category coefficient.

Peak effective fees (at 50¢ price point):

| Category | Peak Fee |
|---|---|
| Geopolitics | 0% (free) |
| Sports | 0.75% |
| Politics | 1.00% |
| Tech | 1.00% |
| Finance | 1.00% |
| Culture | 1.25% |
| Weather | 1.25% |
| Mentions | 1.56% |
| Economics | 1.50% |
| Crypto | 1.80% |

Makers pay 0% and may receive 20-25% of taker fees as rebates (50% in
Finance category).

### 4.2 Structural Differences from Equity Markets

- **Large tick size regime.** 1¢ on a $1 instrument is 100bps relative.
  Gould & Bonart showed queue imbalance signals are strongest in
  large-tick instruments — Polymarket structurally suits OFI signals.
- **CLOB on Polygon.** Settlement is on-chain. Latency budget is
  100-800ms for normal infrastructure (per Dali infrastructure audit),
  vs microseconds for HFT in equities. Sweet spot for ML strategies
  too fast for manual but too slow for traditional HFT.
- **Bounded prices.** [0, 1] range introduces mean-reversion dynamics
  near boundaries. Momentum strategies must be regime-conditioned on
  price level.
- **Resolution dynamics.** Markets converge to truth at resolution.
  Pre-resolution windows have distinct dynamics that contaminate
  microstructure analysis.

### 4.3 Polymarket-Specific Empirical Patterns

From prior research:
- Favorite-longshot bias confirmed: markets <10% resolve YES ~14% of the time
- 72-hour overcorrection windows after major news
- US-centric pricing leaves foreign events underweighted
- 2-5% liquidity premium in thinly-traded markets
- Resolution-criteria ambiguity creates persistent mispricings

---

## 5. Dali Project — Current Findings (as of 2026-05-23)

### 5.1 What Has Been Built

Production research infrastructure including:
- Market universe screening pipeline
- Historical fill TFI baseline (multi-family)
- Live CLOB capture pipeline (book, price_change, best_bid_ask,
  last_trade_price events)
- Replay parser computing CKS-style OFI with maintained book state
- Sign normalization library with empirical placeholders
- Executable-price backtest engine (no spread double-counting)
- Hit-rate-by-magnitude diagnostic across 4 families

### 5.2 Findings from Historical TFI Analysis

Hit-rate-by-magnitude analysis across AI/product, daily crypto, daily
equity index, and sports families produced:

- **Tail-driven, sub-50% hit rate** across most family/horizon combinations
- **AI/product and equity-index:** asymmetric payoff rather than clean
  directional signal — real but hard to trade
- **Crypto:** stronger 300s maker-side behavior but contaminated by
  external price dynamics
- **No family** showed the clean "hit rate rises with magnitude" pattern
  that indicates a robust short-horizon directional signal

Interpretation: fill-only TFI does not clearly support a tradeable
short-horizon strategy on Polymarket. Further analysis required to
exhaust the data; live OFI is the decisive next test.

### 5.3 Open Questions

**Sign convention:** Historical maker_side semantics assumed from logic
("maker is passive, aggressor is inverse") but not empirically verified.
If interpretation is wrong, all TFI results are inverted. Empirical
audit pending.

**Live sign convention:** Only 1 trade in current capture; cannot
establish whether `last_trade_price.side` represents aggressor, maker, or
some other convention. Requires longer captures (50+ classifiable trades).

**OFI on Polymarket:** No data yet. Block A is designed to collect 24-48h
of live OFI on shortlisted markets to test the CKS thesis on Polymarket
specifically.

**Maker thesis:** Polymarket fee structure heavily favors makers. Existing
Midas infrastructure aligns with maker strategies. Avellaneda-Stoikov
framework is directly applicable. Not yet tested.

---

## 6. Current Block Structure and Roadmap

See `../TODO.md` for the full actionable checklist. Summary of blocks:

### Block A — Live OFI Capture with Dual Taker/Maker Analysis (current)

Capture 24-48h of OFI data on 8-12 diverse markets. Analyze for both
taker thesis (CKS-style R² and hit rate) and maker thesis (counterfactual
fill quality, adverse selection). Output classifies each market as
tradeable/ambiguous/not tradeable for each strategy class.

### Block B — TFI Deep-Dive on Existing Fill Data (in parallel)

Exhaust fill-only signal analysis through resolution-contamination
sweeps, per-market heterogeneity, sports-family explicit analysis, and
TFI × {volume, time-of-day} interactions. Output either salvages a
specific signal subset or concludes fill data is exhausted.

### Block C — Sign Convention Resolution (blocker)

Empirically verify historical maker_side semantics through price-action
correlation. If inverted, re-run all prior TFI analyses with corrected
sign. Document Goldsky source semantics if available.

### Block D — Backtest Engine Extensions (deferred until signal validated)

Multi-strategy evaluation, realistic order rejection, per-category fee
model, maker/taker classification at execution, walk-forward validation.

Trigger: at least one strategy class shows tradeable signal in Block A.

### Block E — Wallet/Competition Analysis (parallel research)

Cluster historical wallets by behavior. Identify systematic vs
discretionary, retail vs sophisticated. Filter target markets by
competition intensity. Informs market selection for Block A and future
captures.

### Block F — Parameter Search (deferred per Task 5 triggers)

Optuna-based parameter search over rule-based strategies. Trigger
conditions: 3+ families, 24h+ per family, 200+ events, sign resolved.

### Future Blocks (Phase 2+)

After Phase 1 signal validation:
- Block G: Live deployment infrastructure
- Block H: ML model training (LightGBM first, NNs later) — rule-based baseline must show edge first
- Block I: Cross-platform arbitrage (Polymarket vs Kalshi, options)
- Block J: Resolution-criteria edge scanning (LLM-driven)

---

## 7. Areas for Cowork Deep-Research Extension

1. **Avellaneda-Stoikov adaptation for bounded prices.** Paper assumes Gaussian / unbounded dynamics. What adaptations for [0,1] range? Published applications to prediction markets?

2. **Maker rebate optimization.** 20-25% of taker fees (50% Finance). Optimal quoting under rebate is non-trivial.

3. **Adverse selection in prediction markets.** Most literature is equity. Empirical studies on adverse selection specifically in prediction market flow (retail-heavy categories)?

4. **Cross-platform price discovery.** Polymarket vs Kalshi vs Manifold on overlapping questions. Lead-lag analysis. Who leads, who follows, when does the leader change?

5. **LLM-based prediction market forecasters.** Halawi et al. (2024); Schoenegger & Park (2024) GPT-4 vs Metaculus. State of play as of late 2025.

6. **Polymarket-specific market manipulation literature.** Whale moves, coordinated activity, resolution gaming. Sparse but growing.

7. **Optimal capture protocol design.** CKS used months of data; what's the minimum useful sample size for OFI analysis?

8. **Bias-corrected calibration metrics.** Beyond Brier score — what metrics best detect systematic mispricings exploitable for hold-to-resolution strategies?

> **2026-06-05 extension (cross-book novelty sweep).** Several of the items above are now substantially answered with primary sources in [[2026-06-05_novelty_deep_research]] — read it as the extension of this section:
> - **(1) A-S for bounded prices + (3) adverse selection in PM flow** → § B.2: boundary-aware A-S (σ scaled by p(1−p), terminal-jump mark, Glosten-Milgrom toxicity pull), Bayesian PM market-maker (Brahma/Das), Othman-Sandholm PM structures, make-take rebate economics. Plus the structural read: PM flow is ~84-96% uninformed funding a ~3% informed minority ([Gómez-Cram, SSRN 6617059](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6617059)).
> - **(4) cross-platform price discovery** → § B.5: Polymarket *leads* Kalshi ([SSRN 5331995](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331995)); window compressing toward ms on liquid pairs post-ICE, so the durable edge is slow/thin pairs + resolution-criteria mismatches.
> - **(5) LLM forecasters, state of play 2026** → § B.3: calibration is solved but **profitability is not** — GPT-5 beats the market's Brier yet loses net of spread ([Prophet Arena, 2510.17638](https://arxiv.org/abs/2510.17638)); LLM edge survives only on neglected/slow/fine-print markets (Halawi; FutureSearch). This is the "Block J" two-arm test.
> - **NEW structural edge not previously catalogued:** NegRisk-basket / combinatorial consistency, independently corroborated at **$40M** ([2508.03474](https://arxiv.org/abs/2508.03474)), matching our $34.6M accounting gap → § B.1.
> - **(8) bias-corrected metrics:** the through-line is to gate on *net-of-cost realized PnL on a holdout*, never Brier/calibration (Briola caveat, reinforced by Prophet Arena).
> Ranked candidates + pre-registered gates + Codex prompts: [[2026-06-05_novelty_frontier_map]].

---

## 8. Reading Priorities

If picking three papers to read first:

1. **Cont, Kukanov & Stoikov (2014)** — establishes OFI as the canonical signal and the linear price-impact relationship.

2. **Briola, Bartolucci & Aste (2024)** — the gap between forecasting accuracy and trading profit. Most important honesty check.

3. **Avellaneda & Stoikov (2008)** — if pursuing maker strategies, the foundational closed-form framework.

For the prediction-market context specifically, **Storkey (2011)** is the foundational bridge between ML and prediction markets.

---

## Appendix — Reference Implementation Resources

- **LOBFrame** (Briola et al.) — open-source LOB forecasting framework. Reference for OFI feature pipelines.
- **DeepLOB** (Zhang et al.) — CNN+LSTM reference architecture.
- **AS-MM** — various GitHub implementations of Avellaneda-Stoikov market making; useful starting points.

These are reference implementations, not direct plug-ins. Polymarket context (bounded prices, on-chain settlement, CLOB structure, fee model) requires adaptation.
