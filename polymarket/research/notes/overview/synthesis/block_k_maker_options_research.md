---
title: "Block K Deep Research — Market Making & Binary-Option/Delta Strategies for Polymarket"
tags: [block-k, maker, avellaneda-stoikov, digital-options, delta-hedge, price-discovery, deep-research]
created: 2026-05-30
status: active
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
  - strat_market_making
  - strat_options_delta
method: deep-research harness (5 parallel search angles → fetch → claim extraction → confidence triage → synthesis)
relationship: Feeds brain/handoffs/2026-05-30_maker_options_delta_pivot.md. Built AFTER the local Dali directional signal was closed (P3' OOS + A0c holdout retest).
confidence_legend: "[H] high / primary source; [M] medium; [L] low or practitioner-only; [contested]/[unverified] flagged inline"
---

# Block K — Market Making & Binary-Option/Delta Strategies (Deep Research)

> Hubs: [[COWORK]] | [[strat_market_making]] | [[strat_options_delta]]
> Table terms: [[polymarket_table_dictionary]]

## Summary

The deep-research pass confirms the Block K pivot rests on solid academic foundations, with three findings that
should reshape how we build it:

1. **Avellaneda-Stoikov is an inventory-only framework — it needs no directional alpha.** This is the right tool
   precisely *because* our directional signal is dead. The quoting asymmetry comes from inventory risk, not a
   price forecast. [H]
2. **A bounded-price [0,1] adaptation now exists in the literature** (arXiv:2510.15205, Oct 2025): quote in
   **logit/log-odds space**, where A-S formulas carry over and spreads auto-compress near 0/1, with explicit
   handling of the resolution jump. This fills the exact theory gap flagged in the handoff — but it is an
   unrefereed preprint with **no empirical backtest**, so we validate it ourselves. [H source / unvalidated method]
3. **In the one realistic costed backtest found, A-S profit came almost entirely from maker rebates, not from
   the skew** — and naive A-S *lost badly* (Sharpe ≈ −246) until skew was dampened. This is the empirical
   anchor for our whole thesis: the edge is the rebate + spread, and the directional term is a small,
   dangerous add-on. [H]

The hard constraint, unchanged and now triple-confirmed by theory (Glosten-Milgrom), our own data (A1.4h: 248 bps
adverse selection, fills flow-capped 9%→0.2%), and the empirical Polymarket microstructure literature: **a maker
profits from uninformed flow and loses to informed flow.** The entire question reduces to whether
`rebate + spread capture > adverse selection + inventory/resolution risk` on a given market.

Two refinements to the handoff fall straight out of the fee research:
- **The rebate cushion does NOT exist in geopolitics** (taker fee = 0 there → no rebate to redistribute). So
  Track B "pure maker rebate" is mis-named for geopolitics; on fee-free categories you have *only* spread
  capture vs adverse selection, no rebate backstop. The rebate cushion lives in crypto (20%), and
  sports/politics/finance/etc. (25%). Re-target Track B accordingly.
- **The cleanest documented near-model-free edge on Polymarket is not maker or delta at all — it's
  intra-platform combinatorial/rebalancing arbitrage (~$40M realized, 2024-25).** Flagged as a high-credibility
  alternative thread.

---

## Part I — Optimal market-making theory (Avellaneda-Stoikov family)

**Avellaneda & Stoikov (2008).** Single dealer maximizing CARA utility of terminal wealth over horizon T; mid-price
is arithmetic Brownian motion (Gaussian, **unbounded**). Order arrivals Poisson with intensity λ(δ)=A·e^(−κδ).
Reservation price `r = s − q·γ·σ²·(T−t)` (inventory q skews the center); optimal total spread
`δᵃ+δᵇ = γσ²(T−t) + (2/γ)ln(1+γ/κ)`, which collapses to zero as t→T. The model is **inventory-only**: no drift/alpha
term. [H] — Quantitative Finance 8(3):217-224; https://www.tandfonline.com/doi/abs/10.1080/14697680701381228 ;
formulas via https://hummingbot.org/blog/technical-deep-dive-into-the-avellaneda--stoikov-strategy/

**Guéant, Lehalle & Fernandez-Tapia (2013), "Dealing with the inventory risk."** Keeps the A-S setup but solves it
under **explicit inventory bounds** q∈[−Q,Q] and reduces the HJB to linear ODEs, yielding **closed-form asymptotic
(time-stationary) quotes** — this is the version practitioners deploy, because it doesn't collapse as t→T.
Practical closed form: `half-spread = c₁ + (Δ/2)σc₂`, `skew = σc₂` with
`c₁=(1/(ξΔ))ln(1+ξΔ/k)`, `c₂=√[(γ/(2AΔk))(1+ξΔ/k)^(k/(ξΔ)+1)]`. [H] — arXiv:1105.3115 ;
https://hftbacktest.readthedocs.io/en/latest/tutorials/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.html

**Guéant (2017), "Optimal market making."** Generalizes GLFT to general intensities and adds multi-asset closed
forms; demonstrated on corporate bonds and two credit indices (a real application). Multi-asset extension with
correlation in Bergault-Guéant (arXiv:1810.04383). [H] — arXiv:1605.01862

**Bounded-price / resolution-jump adaptation (the key gap — now filled).** "Toward Black-Scholes for Prediction
Markets" (arXiv:2510.15205, Oct 2025) **explicitly adapts A-S to bounded [0,1] prices by quoting in logit space**
x=logit(p): reservation `rₓ = xₜ − qₜγσ̄²(T−t)`, spread `2δₓ ≈ γσ̄²(T−t)+(2/k)ln(1+γ/k)`, mapped back via
p=sigmoid(x). Displayed half-spread `δ_p ≈ p(1−p)·δₓ` **auto-compresses near p→0/1**; imposes a spread floor and an
inventory cap that tightens near boundaries. Prices first-passage/threshold notes with an **absorbing boundary** and
handles resolution via terminal conditions in a PIDE (models the settlement jump). [H source]
**Caveat [unverified method]:** preprint v1, partly heuristic, **no backtest** — treat the formulas as a principled
starting point, not a proven rule. https://arxiv.org/html/2510.15205v1

**Costed implementation reality.** hftbacktest GLFT on ETHUSDT (Binance Futures) with realistic fees (−0.005% maker
rebate, 0.07% taker): naive GLFT → **Sharpe ≈ −246, −2.06%** (skew so strong ~2 units of inventory wipes the
half-spread). With skew-dampening (adj2=0.05): single day → Sharpe ≈ 1.2 (breakeven, *"profit mainly from
rebates"*); 5-day → Sharpe ≈ 16; +grid overlay → Sharpe ≈ 20, +5.6%. γ (risk aversion) is set arbitrarily — a known
weakness. [H] — same hftbacktest URL. RL-tuned A-S (Falces Marin et al., PLOS ONE 2022) beats baseline on Sharpe but
shows occasional heavy drawdowns. [M] — https://doi.org/10.1371/journal.pone.0277042

> **Dali read:** The rebate-not-skew result is the single most important transfer. It says our thesis ordering is
> correct: prove the rebate+spread baseline with **zero directional skew first**, then add skew only if it earns
> incremental edge. And it warns that aggressive skew (which a noisy contrarian signal would produce) is how A-S
> blows up.

---

## Part II — Adverse selection: why this is the whole ballgame

**Glosten-Milgrom (1985).** A positive bid-ask spread arises *purely from information asymmetry* even with a
risk-neutral, zero-profit dealer: ask = E[value | buy arrives], bid = E[value | sell arrives]; the spread widens
with the probability the counterparty is informed. The MM **profits from uninformed flow and loses to informed
flow**; the competitive quote is the break-even where they offset. [H] — https://www.sciencedirect.com/science/article/pii/0304405X85900443

**Kyle (1985).** Strategic informed trader vs noise traders; linear price impact `ΔP = λ·(order flow)`, Kyle's λ =
price impact, 1/λ = market depth. Noise trading "camouflages" the informed and provides depth. [H] —
https://personal.utdallas.edu/~nina.baranchuk/Fin7310/papers/Kyle1985.pdf

**Flow toxicity / VPIN.** VPIN (Easley-López de Prado-O'Hara) measures order-flow imbalance over equal-*volume*
buckets as a proxy for probability of informed trading; MMs widen/pull when it's high. **[contested]** Andersen &
Bondarenko show VPIN is largely a mechanical artifact of volume and *peaked after, not before*, the Flash Crash;
usefulness is disputed. Use it, if at all, as one input, not gospel. [H for the critique] — VPIN
https://www.quantresearch.org/VPIN.pdf ; critique https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1881731

**RL / deep market making.** Spooner et al. (2018) and Ganesh et al. (2019, JPMorgan) show RL MMs that learn
**inventory skew and hedging**, with PnL decomposed into Spread − Hedge Cost + Inventory (transaction costs built
in). **All results are simulator-only; latency is the cited barrier to live use.** [H sources / sim-only] —
arXiv:1804.04216 ; arXiv:1911.05892

**Forecasting ≠ profit (Briola, Bartolucci & Aste 2024).** SOTA deep LOB models forecast NASDAQ mid-moves, but high
forecasting power *"does not necessarily correspond to actionable trading signals"*; standard accuracy/F1 fail, and
whether DL even works depends on a stock's microstructure (tick size, liquidity). [H — verbatim abstract; specific
net-loss magnitude **unverified**] — arXiv:2403.09267

> **Dali read:** Our A1.4h finding (248 bps adverse selection on the one filled cell; fills flow-capped 9%→0.2%) is
> the Glosten-Milgrom informed-flow tax made concrete, and the 73.7%→36% OOS collapse is a textbook Briola result.
> Nothing here contradicts our closure; it explains it and tells us the maker version must be *measured*, not assumed.

---

## Part III — Prediction-market & betting-exchange market making

**LMSR (Hanson).** Cost function `C(q)=b·log Σexp(qᵢ/b)`; price `Pᵢ=exp(qᵢ/b)/Σexp(qₖ/b)` (sums to 1, = probability).
Bounded worst-case loss `b·log(n)` (binary: b·ln2). The b knob is **fixed liquidity**: it never profits, offers **no
adverse-selection protection**, and moves price equally in thin and thick markets. [H] —
https://gnosis-pm-js.readthedocs.io/en/v1.3.0/lmsr-primer.html ; http://mason.gmu.edu/~rhanson/mktscore.pdf

**Liquidity-sensitive AMMs (Othman, Pennock, Reeves & Sandholm 2010/2013).** Cost-function MM that is
liquidity-sensitive (impact shrinks as volume grows) **and can run at a profit**, fixing LMSR's two flaws by letting
the effective spread (Σprices − 1) grow with volume. [H] — https://www.cs.cmu.edu/~sandholm/liquidity-sensitive%20market%20maker.EC10.pdf

**Polymarket is a CLOB, not an AMM** (off-chain order book, on-chain CTF settlement). Fee structure (official docs):
**maker fee = 0**; **taker fee = C·feeRate·p·(1−p)** — symmetric around 50¢, →0 at the extremes. feeRate by category:
**Crypto 0.07, Econ/Culture/Weather/Other 0.05, Finance/Politics/Tech 0.04, Sports 0.03, Geopolitics 0**. **Maker
rebates** are funded *from collected taker fees*: **20% (Crypto) / 25% (other fee-enabled)** redistributed daily in
USDC pro-rata to maker liquidity, $1 minimum. CLOB v2 launched 28 Apr 2026 with a ~$1M liquidity-rewards program for
pro MMs. [H official / $1M figure M-secondary] — https://docs.polymarket.com/market-makers/maker-rebates ;
https://docs.polymarket.com/trading/orderbook

> **Critical refinement:** because rebates are funded by taker fees, **geopolitics (0 fee) pays no rebate**. The
> rebate cushion exists only where takers are charged. Crypto-4h (feeRate 0.07, 20% rebate) is the best-cushioned
> category we care about; sports/politics get 25% of a smaller fee.

**Kalshi.** Same convex taker-fee shape (`0.07·p·(1−p)`, peak ~$0.0175 at 50¢); makers usually 0 (flat ~0.25% on
marquee events [M]). Runs a **formal designated-market-maker program** (two-sided quoting commitments for fee/limit
benefits; incentive payouts reported ~$35k/day [M-unverified]). [H program / M figures] —
https://kalshi.com/docs/kalshi-fee-schedule.pdf ; https://help.kalshi.com/en/articles/13823819-how-to-become-a-market-maker-on-kalshi

**Betfair / sports exchanges.** MMs "trade out / green up" — back and lay the same selection at different odds to
lock a guaranteed profit (a "green book"), profiting from price movement rather than holding to settlement. In-play
is the main venue; edge is information-speed + deep liquidity to exit, i.e. a latency game, not passive spread. [H
mechanics / M practitioner] — https://betting.betfair.com/guides/how-to-trade-on-betfair/

**Empirical Polymarket microstructure (arXiv:2604.24366).** Strong **longshot spread premium**: median quoted
half-spread ~400 bps mid-book, **~1,818 bps in the 0-10¢ decile vs ~53 bps in the 90-100¢ decile**. Depth is
**layered, not top-heavy** (median L1/L10 depth ratio 0.137). Maker liquidity is **decentralized** (~32 effective
makers/market; thin-market tail dominated by 1-3 wallets). A separate working paper finds **makers systematically
favored at extreme prices** (at ~1¢, takers win ~0.43% vs makers ~1.57%) and that post-2024 volume drew in pro LPs
capturing spread. [H primary / M for the 1¢ asymmetry] — https://arxiv.org/html/2604.24366v1 ;
https://www.jbecker.dev/research/prediction-market-microstructure. Favorite-longshot bias is the most robust
prediction-market regularity. [H] — https://arxiv.org/pdf/1805.04225

---

## Part IV — Binary/digital options & delta-hedging the crypto-4h contract

**Digital pricing.** European cash-or-nothing digital call = `e^(−rτ)·N(d₂)` = discounted risk-neutral P(S_T>K). A
Polymarket binary that settles $1/$0 **is** a digital; its price = market-implied probability, and for short τ the
discount ≈1 so **price ≈ P(event)**. YES+NO=$1 is digital call+put parity. [H] —
https://www.quantpie.co.uk/bsm_bin_c_formula/bs_bin_c_summary.php ;
https://navnoorbawa.substack.com/p/the-math-of-prediction-markets-binary

**The delta blow-up.** Digital delta `= φ·e^(−rτ)·n(d₂)/(S·σ·√τ)`; gamma carries `1/τ`. As τ→0 at/near the strike,
the **1/√τ factor sends delta →∞** and gamma diverges faster — delta is a tall spike at K, ~0 elsewhere. Continuous
delta-hedging near strike/expiry is **cost-explosive** (pin/barrier risk). [H] — quantpie + https://medium.com/@fc.cortinovis/digital-options-9c01cb4a86dc

**Hedging approach.** Practitioner standard is a **leveraged vanilla call-spread** (long α at K−H/2, short α at
K+H/2, α=1/H), sized as a conservative over-hedge rather than exact replication to bound risk. Peer-reviewed result
(Springer 2023): delta-hedging an ATM digital near maturity *with the wrong vol can increase risk vs not hedging at
all* — favors a static bull-spread hedge. For crypto, the hedge instrument is **spot or perpetual futures**;
static call-spread preferred precisely because continuous futures rebalancing around the strike is costly. [H / M] —
https://link.springer.com/article/10.1007/s11009-023-10013-6

**0DTE transfer.** A 4-hour binary behaves like a 0DTE option: vega ≈ irrelevant, value dominated by **expected
realized vol over the window** and by **gamma/theta**, not term-structure IV. Dealer gamma sign governs whether
hedging dampens or amplifies the underlying. [H] — https://menthorq.com/guide/understanding-0dte-gamma-exposure/

**Gap [unverified]:** no peer-reviewed paper found pricing 4-hour crypto event contracts specifically; nearest
rigorous work is the Springer ATM-digital-near-maturity paper. The framing (digital on S_open, value = expected
window realized vol) is standard and safe; the *empirical* edge is ours to find.

> **Dali read:** The delta blow-up locates the danger precisely: a crypto-4h binary is most adverse near 50¢ late in
> the window (S≈S_open, τ→0). That's exactly where a maker accumulates toxic, hard-to-hedge inventory. Quoting must
> widen / cap inventory as p→50¢ and τ→0 — which is the *opposite* of the logit-space auto-compression (that
> compresses near 0/1, i.e. it helps at the extremes, not at the dangerous middle). Combine: logit-space spread for
> the boundary behavior + an explicit τ-and-distance-to-50¢ widening term for the gamma spike.

---

## Part V — Cross-venue price discovery & arbitrage

**Methodology.** Hasbrouck (1995) Information Shares: each venue's share of the variance of the common efficient-price
innovation, from a cointegrated VECM; ordering-dependent (upper/lower bounds). NYSE ~92.7% IS of the Dow. [H] —
https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1995.tb04054.x. Gonzalo-Granger component shares are the
permanent-transitory alternative (which venue others adjust toward); IS and GG differ unless residuals uncorrelated;
Information Leadership Share (Putniņš/Yan-Zivot) is the order-robust combination. [H] —
https://faculty.washington.edu/ezivot/research/DynamicsOfPriceDiscovery20050418.pdf

**Crypto lead-lag.** Price discovery has migrated toward derivatives (perps/CME lead spot when their volume
dominates), though venue/period-dependent. Recent HF work (arXiv:2506.08718) applies IS/CS/IL/ILS + Hayashi-Yoshida
across ~20 spot/26 futures venues. Cross-exchange latency-arb windows are short (~100 ms cited) and shrinking. [M /
practitioner-L for the ms figures] — arXiv:2506.08718

**Prediction market vs underlying — the live one.** Polymarket short-dated crypto markets **demonstrably lagged
Binance/Coinbase** (~30-90s cited [L]), and bots arbed it. **Polymarket killed the clean version by introducing a
dynamic taker fee on 15-minute crypto markets (~3.15% at 50¢), deliberately set above the typical arb margin.** This
is direct evidence the gross edge existed and is now largely fee-eaten on the shortest markets. [H for the fee
response] — https://www.financemagnates.com/cryptocurrency/polymarket-introduces-dynamic-fees-to-curb-latency-arbitrage-in-short-term-crypto-markets/

**The cleanest documented edge is intra-Polymarket arb.** Saguillo et al., "Unravelling the Probabilistic Forest"
(AFT 2025, arXiv:2508.03474), using on-chain order books Apr 2024-Apr 2025, documents **Market Rebalancing**
(YES+NO≠$1) and **Combinatorial** (logically related markets) arbitrage across 7,000+ markets with **~$40M realized
profit** — model-free, guaranteed when it appears. Cross-platform Polymarket↔Kalshi arb is real but friction-bound
(fee asymmetry, UMA-oracle resolution freezes of 3-14 days, capital lockup, leg risk, ~2-7s gaps). [H for the $40M /
M-L for cross-platform frictions] — https://arxiv.org/abs/2508.03474

> **Dali read:** The Polymarket-lags-Binance edge (our prompt P6) is real but the 15-min markets are now
> fee-protected. Two implications: (1) check whether the **4-hour** crypto markets carry the same dynamic fee — if
> not, the delta/basis edge may still be open at the 4h horizon where we have data; (2) the $40M intra-Polymarket
> combinatorial/rebalancing arb is a higher-credibility, model-free alternative thread that reuses our order-book
> infra and sidesteps adverse selection entirely. Worth a scoping look in parallel.

---

## Part VI — SYNTHESIS: strategy classes mapped to Dali findings

| Strategy class | Academic basis | Survives our structural problems? | First validation test (data we own) |
|---|---|---|---|
| **A-S / GLFT inventory maker, NO directional skew** | A-S 2008; GLFT 2013; logit-space 2510.15205 | **Best fit.** Needs no directional alpha (which is dead). Lives/dies on rebate+spread > adverse selection. The hftbacktest result says rebate is the real edge. | Maker-economics decomposition (below). Then logit-space A-S sim with zero skew. |
| **+ contrarian inventory/flow skew** | Glosten-Milgrom (lean away from informed side) | **Maybe, small.** Our signal is 36% OOS = contrarian/reversion. Skew *against* the extreme is theory-consistent, but A-S blows up on aggressive skew. Start at zero; add tiny. | Incremental-PnL test of contrarian skew over the no-skew baseline, OOS on A0c. |
| **Crypto-4h maker + external delta hedge** | digital Greeks; Springer 2023 static hedge; 0DTE gamma | **Conditional.** Delta spike near 50¢/expiry is the toxic zone; static call-spread/perp hedge needed. Edge requires the 4h market NOT be dynamic-fee-killed like the 15-min. | (1) Check 4h dynamic-fee status. (2) Digital-basis test: N(d₂) from Binance vs Polymarket mid; measure lag + post-fee edge. |
| **Pure-maker rebate (fee-enabled cats)** | A-S; Polymarket rebate 20/25% | **Conditional.** Rebate cushion real ONLY where takers are charged. Crypto/sports/politics yes; **geopolitics no**. | Decomposition per category; rank by (rebate+spread − adverse selection). |
| **Pure-maker on geopolitics** | — | **Weak.** No rebate (0 taker fee). Only spread capture vs adverse selection, on slow-flow markets. Our A1.4h flow-cap hits hardest here. | Same decomposition; expect negative — treat as control. |
| **Longshot-spread harvesting** | longshot bias; 2604.24366 (1818 bps spreads at 0-10¢); jbecker maker-win asymmetry | **Intriguing, untested.** Huge spreads + makers favored at extremes + digital delta smallest there (easy hedge). But thin flow and resolution/jump risk at the tails. | Decomposition restricted to 0-10¢ / 90-100¢ deciles; measure realized capture vs tail-jump losses. |
| **Cross-venue lead-lag (P6)** | Hasbrouck IS; crypto price discovery | **Conditional, time-decaying.** Real but fee-protected on 15-min; unknown on 4h; latency-gated. | IS/lead-lag of Binance vs Polymarket 4h on owned A0c crypto-roll; post-fee survival. |
| **Intra-Polymarket combinatorial/rebalancing arb** | Saguillo 2025 ($40M) | **Strongest model-free edge, untouched by our findings.** No adverse selection, no directional bet. Competition + latency are the constraints. | Scan owned captures for YES+NO≠$1 and logically-linked-market violations; measure frequency × size × our latency. |
| **LMSR / cost-function AMM** | Hanson; Othman-Sandholm | **Not applicable** (Polymarket is a CLOB; we'd be a taker of liquidity-provision design, not an operator). | n/a — reference only. |
| **RL market making** | Spooner; Ganesh | **Premature.** Sim-only in literature; Briola caveat; no rule-based maker edge yet. Latency barrier. | Deferred until a rule-based maker baseline clears zero. |

### The decision spine

Everything routes through one number, exactly as Glosten-Milgrom and our A1.4h both say:
**`net maker PnL = rebate + spread capture − adverse selection − inventory/resolution risk`.**
Prove that is positive *before any skew*, on data we own, or the pivot doesn't start.

---

## Recommended first validation tests (all on A0/A0b/A0c — no new capture)

1. **Maker-economics decomposition (the ballgame).** Per market, per category, reconstruct passive fills with the
   A1.4h fill proxy and compute: realized half-spread captured, rebate using the confirmed formula
   `C·feeRate·p·(1−p)` × {0.20 crypto, 0.25 other, 0 geopolitics}, minus adverse selection at 5/30/60s, minus an
   inventory/resolution-risk charge. **Output the sign of baseline maker PnL with zero skew, split fee-enabled vs
   fee-free.** This is the gate for the entire thread.
2. **Logit-space A-S sim** (arXiv:2510.15205 formulas) fed by our fill proxy, **no skew first**; add a τ-and-
   distance-to-50¢ widening term for the crypto gamma zone; measure PnL vs test 1's baseline. Then add a small
   contrarian skew and measure *incremental* edge OOS on A0c only.
3. **Crypto-4h digital-basis + fee check.** (a) Confirm whether 4h crypto markets carry the dynamic anti-arb taker
   fee. (b) Compute N(d₂) fair value from Binance spot + window realized vol vs Polymarket mid on the A0c crypto-roll
   windows; measure lead-lag (the 30-90s claim) and whether any basis survives the fee. Folds in prompt P6.
4. **Combinatorial-arb scan (parallel, high-credibility).** Scan owned captures for YES+NO≠$1 and logically-linked
   market inconsistencies; quantify frequency × size and whether our latency could have captured them. Cheap, and
   it's the one edge the literature shows is real and model-free.

### Sequencing
Test 1 first — it's a few hours of replay and it gates 2. Run 3 and 4 in parallel (independent). Do **not** build a
quoter or provision infra until test 1's baseline is positive on at least one category. Carry the standing
guardrails: non-overlap, net-of-cost, CI bars; rebate ≠ free money if fills are adversely selected; no RL/ML before a
rule-based maker edge exists (Briola).

---

## Confidence & contested-claims ledger

- **[contested] VPIN** as a volatility/toxicity predictor — Andersen-Bondarenko rebut the original Flash-Crash claim;
  treat as one weak input.
- **[unverified method] arXiv:2510.15205** bounded-price A-S — strong, directly relevant, but unrefereed preprint with
  no backtest. Use as scaffolding, validate empirically.
- **[unverified] Briola** specific net-of-cost loss magnitude — the "accuracy ≠ actionable" framing is verbatim;
  the dollar figure is not in the abstract.
- **[L / practitioner]** all latency numbers (30-90s Polymarket lag, 100ms CEX windows), the $271k/30-day UI-lag bot,
  Kalshi ~$35k/day MM incentive, $1M v2 rewards pool — directionally consistent across sources, not peer-reviewed.
- **[M]** the 1¢ maker-win asymmetry and per-category effective-spread signs (2604.24366 flags measurement noise).
- **[H]** A-S/GLFT/Guéant formulas, Glosten-Milgrom/Kyle, digital Greeks + delta blow-up, Hasbrouck IS, Polymarket
  official fee/rebate structure, Saguillo $40M intra-platform arb.

## References (primary, deduped)
- Avellaneda & Stoikov (2008), Quantitative Finance 8(3) — tandfonline 10.1080/14697680701381228
- Guéant, Lehalle & Fernandez-Tapia (2013) — arXiv:1105.3115
- Guéant (2017), Optimal market making — arXiv:1605.01862 ; Bergault-Guéant multi-asset — arXiv:1810.04383
- "Toward Black-Scholes for Prediction Markets" (2025) — arXiv:2510.15205
- Glosten & Milgrom (1985) — JFE 14(1) ; Kyle (1985) — Econometrica 53(6)
- Easley, López de Prado, O'Hara — VPIN (quantresearch.org) ; Andersen & Bondarenko critique — SSRN 1881731
- Spooner et al. (2018) — arXiv:1804.04216 ; Ganesh et al. (2019, JPMorgan) — arXiv:1911.05892
- Briola, Bartolucci & Aste (2024) — arXiv:2403.09267
- Hanson LMSR — mason.gmu.edu/~rhanson/mktscore.pdf ; Othman-Sandholm liquidity-sensitive MM — CMU EC2010
- Polymarket maker-rebate & order-book docs — docs.polymarket.com
- Kalshi fee schedule & DMM program — kalshi.com
- Polymarket microstructure — arXiv:2604.24366 ; favorite-longshot — arXiv:1805.04225
- Digital option Greeks — quantpie.co.uk ; ATM-digital static hedge — Springer 10.1007/s11009-023-10013-6
- Hasbrouck (1995) information shares — JF 50(4) ; crypto price discovery — arXiv:2506.08718
- Saguillo et al. (2025), intra-Polymarket arbitrage — arXiv:2508.03474
