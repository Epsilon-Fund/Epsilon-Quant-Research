---
title: "Block K — Plain-English Synthesis: Concepts, Findings, and the Two Live Strategies"
tags: [block-k, synthesis, glossary, maker, options, plain-english]
created: 2026-05-31
audience: "anyone picking this up cold — assumes no quant-finance background"
relationship: >
  Synthesises the whole Block K maker/options-delta arc (K1, K2, K2v2, K2v3, K-PEG + robustness +
  maker-exit, K3, K4, K6) into one readable document. Source notes remain authoritative for numbers;
  this is the orientation + glossary + strategy explainer.
---

# Block K — Plain-English Synthesis


## Summary

- Scope: Block K — Plain-English Synthesis in the research area.
- Existing takeaway/status: We spent Block K asking one question: **can we make money by providing liquidity ("market making") on Polymarket's short-dated crypto markets, instead of betting on direction?** The directional bet was already closed in earlier work; this was the pivot.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
> **Now split into two strats.** This doc remains the shared origin + glossary for the whole arc, but
> active work is tracked under two hubs: **[[strat_market_making]]** (MM — quote for spread + rebate) and
> **[[strat_options_delta]]** (OD — digital-option vol/basis + delta hedge). "Block K" = the historical
> joint arc. Foundation: [[block_k_maker_options_research]].
> Table terms: [[polymarket_table_dictionary]]

## 0. TL;DR (read this if nothing else)

We spent Block K asking one question: **can we make money by providing liquidity ("market making") on
Polymarket's short-dated crypto markets, instead of betting on direction?** The directional bet was
already closed in earlier work; this was the pivot.

After ~ten tests, the answer for *a maker operating on Polymarket alone* is **no** — robustly, for a
specific and now well-understood reason. But two real, measurable edges survived the wreckage:

1. **A passive entry genuinely catches a real price reversion** (we proved the entry signal is consistent,
   not luck).
2. **Polymarket systematically overprices "longshot" outcomes / volatility** (a real, large mispricing).

Both are blocked by the *same* thing: the cost of getting *out* of the position on Polymarket eats the
edge. That single fact points to the two strategies still worth pursuing (Section 6):
- **Strategy A:** capture the entry, then *hold to resolution and hedge the direction externally on
  Binance with a static hedge* — so you never pay Polymarket's exit cost.
- **Strategy B:** stop trying to be the maker; *find the traders who are already profitably making these
  markets and copy/learn from them* (re-merges with the copytrade project).

---

## 1. The setup: what "market making" even means here

A prediction market like Polymarket is a place where people buy and sell "shares" that pay **$1 if an
event happens and $0 if it doesn't.** A share trading at $0.66 means the market thinks there's a ~66%
chance. Because the payoff is $1 or $0, the price *is* a probability.

There are always two kinds of participants:

- **Takers** cross the spread to trade *now*. They pay a fee and accept a worse price for immediacy.
- **Makers** post resting orders ("I'll buy at $0.64, I'll sell at $0.68") and *wait* to be filled. They
  get a better price and, on Polymarket, a small **rebate**. They provide the liquidity takers consume.

Our whole pivot was: **be the maker, not the taker.** The appeal is structural — Polymarket *pays* makers
(rebate) and *charges* takers (fee), so in principle a maker can be profitable before having any view on
which way the price goes.

### The core trade-off (memorise this one equation)

```
maker profit per trade  =  spread captured  +  rebate  −  adverse selection  −  inventory/resolution risk
```

Everything in Block K is a fight over the right-hand side. The first two terms are what you earn; the last
two are what can kill you. **Adverse selection** is the killer, so it gets its own section.

---

## 2. Glossary — every term we used, in plain English

### Order-book basics
- **Bid / ask (the "touch"):** the highest price anyone will currently buy at (bid) and the lowest anyone
  will sell at (ask). The gap between them is the **spread**.
- **Tick:** the smallest price step. On Polymarket it's **1 cent ($0.01).**
- **Mid:** the simple midpoint, `(bid + ask) / 2`.
- **Micro-price:** a smarter "fair value" than mid. It leans toward whichever side has *less* size
  resting, because that side is more likely to get hit next. If there's a huge bid and a tiny ask, the
  true value is closer to the ask. It's `(ask·bid_size + bid·ask_size) / (bid_size + ask_size)`.
- **Reservation price:** in market-making theory, the price you center your quotes around — usually fair
  value, *adjusted* for how much inventory you're already holding (if you're long, you shift down so you're
  keener to sell than buy).

### Fees (Polymarket specifics — these are real and confirmed)
- **Maker fee = 0.** Makers are never charged.
- **Taker fee = `C · feeRate · p · (1−p)`.** It's biggest at 50¢ and shrinks to zero near 0¢/100¢.
  `feeRate` depends on category: Crypto 0.07, Sports 0.03, Geopolitics **0** (no fee at all).
- **Maker rebate:** funded *out of* collected taker fees and paid to makers — 20% (Crypto), 25% (other
  fee-enabled categories), **0% for Geopolitics** (because no taker fee is collected there, there's nothing
  to rebate). This is why "earn the rebate" is a real cushion in crypto but a *myth* in geopolitics.

### The killer: adverse selection
- **Adverse selection:** the bad case where the person who fills your resting quote knows something (or is
  faster than) you. You posted a bid at $0.64; someone sells to you at $0.64 *because* the price is about to
  drop to $0.60. You "won" the trade and immediately lost money. A maker **profits from uninformed flow and
  loses to informed flow** — this is the 1985 Glosten-Milgrom result, and it's the single most important
  idea in the whole block.
- **Toxic fill:** a fill that was adversely selected. Lots of toxic fills = the rebate and spread can't save
  you.
- **Markout:** how the mid moved in the seconds *after* your fill. It's how we *measure* adverse selection
  (e.g. "30-second markout" = mid 30s later minus your entry).

### Inventory and the binary jump
- **Inventory risk:** while you hold a position, the price can move against you before you offload it.
- **Resolution / settlement:** when the market closes, every share snaps to exactly $1 or $0. Holding a
  position *into* resolution is a coin-flip-like jump, not a smooth move. This is unique to binary markets
  and it's why holding inventory to the end is high-variance.
- **Hold-to-resolution:** deliberately *not* trading out, and instead letting the position settle at $1/$0.
  This avoids paying to exit, but you take the full binary jump (unless you hedge it — see Strategy A).

### The market-making theory we tried to use
- **Avellaneda-Stoikov (A-S):** the standard math for optimal maker quoting. Its key feature: it needs **no
  view on direction** — it tells you how wide to quote and how to skew based purely on *inventory risk* and
  *volatility*. That's why it was attractive: our directional edge was already dead.
- **Inventory skew:** A-S says if you're holding too much, shift your quotes to dump it (quote cheaper to
  sell, less keen to buy). The skew comes from inventory, not a price forecast.
- **Logit-space adaptation:** plain A-S assumes prices can wander anywhere; Polymarket prices are trapped in
  [0,1] and jump at resolution. The fix (a 2025 paper) is to do the A-S math in "log-odds" space, which
  naturally compresses spreads near 0/1. We tested it (K2) — see findings.

### How we judge results (methodology terms — these are why we trust/distrust a number)
- **In-sample (IS) vs out-of-sample (OOS):** IS = the data you tuned the strategy on. OOS = fresh data it's
  never seen. **An IS result is a lead, not a result** — almost every strategy looks good on the data it was
  optimised on. This project has been burned repeatedly by IS positives that died OOS.
- **Optimisation ceiling / overfitting:** if you try 360 parameter combos and report the best one, you've
  cherry-picked noise. The "best" is an upper bound (a *ceiling*), not what you'd actually earn.
- **Overlap vs non-overlap:** if two simulated trades overlap in time they double-count the same price move,
  manufacturing fake profit. **Non-overlap** (only one position at a time) is the honest way. Overlap math
  produced essentially every false positive in the earlier directional work.
- **Confidence interval (CI) / bootstrap:** a range around the average. If the *whole* CI is above zero,
  the edge is probably real; if the CI straddles zero, it's noise. "Bootstrap" = re-sampling the data many
  times to build that range. "**Clears**" in our tables means: mean > 0, CI lower bound > 0, and ≥30 trades.
- **Fill rate vs absolute fills:** "fill rate" can be misleading — K-PEG's "0.04%" counted every 1-second
  *re-quote* as an order opportunity, so it looked tiny. The honest metric is **absolute fills per day**
  (~166/day for crypto here), which is plenty.
- **Mark-to-mid vs realizable PnL:** *the* distinction of this whole block. **Mark-to-mid** = "value my open
  position at the current mid." **Realizable** = "what I'd actually get if I closed it." On a wide-spread
  venue these are wildly different, because closing means crossing the spread. A pretty mark-to-mid profit
  can be a real loss the moment you try to bank it.

### The options lens (because a 4-hour crypto market *is* an option)
- **Digital / binary / cash-or-nothing option:** an option that pays a fixed $1 if the underlying is above
  a strike at expiry, else $0. A Polymarket "Will BTC be up over the next 4 hours?" contract **is exactly
  this** — strike = the price at the window's start, expiry = 4 hours later.
- **N(d₂) / implied probability:** the textbook formula for a digital's fair value ≈ the probability the
  event happens. We compute it from Binance's BTC price + how volatile BTC has been + time left, and compare
  it to Polymarket's price. A gap = a potential mispricing.
- **Moneyness (we call it `z`):** how far the current price is from the strike, scaled by volatility and
  time. `z ≈ 0` means "right at the strike, genuine coin-flip." Large `|z|` means "far from the strike, the
  outcome looks fairly decided" (a *longshot* on one side).
- **τ (tau):** time left until the market resolves.
- **Delta:** how much the option's fair value moves when the underlying (BTC) moves $1. For a digital, delta
  is a tall **spike right at the strike** and ~0 far away.
- **Gamma:** how fast delta changes. For a digital near expiry it **blows up** (mathematically ∝ 1/√τ). This
  is why hedging a digital near the strike, near expiry, is brutally expensive — the hedge needs constant,
  large rebalancing. This "delta/gamma spike zone" (near 50¢, late in the window) is the danger area.
- **Theta:** time decay — how the option bleeds value as the clock runs, all else equal.
- **Delta hedging:** neutralising direction by holding an offsetting BTC position on Binance. **Continuous**
  hedging rebalances constantly (accurate but expensive near the spike). A **static hedge** (e.g. a fixed
  call-spread) is set once and left — cheaper, less precise. The literature says static beats continuous for
  digitals near expiry; we have *not* tested static yet.
- **Implied vol vs realized vol:** "implied" = the volatility the market's price *assumes*; "realized" = how
  volatile BTC actually was. If Polymarket's implied vol > realized, Polymarket is *overpricing
  uncertainty* — which is exactly what K6 found.
- **Gamma scalping:** the classic way to harvest an implied-vs-realized vol gap — buy/sell the option and
  continuously delta-hedge, capturing the difference. K6 tested it; the hedge cost killed it.
- **Longshot premium / favorite-longshot bias:** the most reliable regularity in all betting/prediction
  markets — unlikely outcomes are systematically *overpriced* (people overpay for lottery-ticket bets). On
  Polymarket this shows up as huge spreads and overpriced vol far from the strike.

### Cross-venue terms
- **Lead-lag / price discovery:** which venue moves first. We found **Binance leads Polymarket by ~10
  seconds** on crypto — Polymarket is the slow follower.
- **Combinatorial / rebalancing arbitrage:** risk-free profit from price inconsistencies (e.g. YES + NO ≠
  $1, or logically linked markets disagreeing). K4 scanned for it.

---

## 3. What each test found (the story in order)

| block | what it tested | result | one-line takeaway |
|---|---|---|---|
| **K1** | Baseline maker economics — is spread+rebate > adverse selection, before any skew? | Several categories "clear" | But **fee-free Geopolitics also cleared**, proving K1 is just a *generous* spread-capture gate (it marks to mid and doesn't pay to exit), not proof of a real edge. A permission slip to keep testing, nothing more. |
| **K2** | Proper Avellaneda-Stoikov maker in logit space, optimised, *with a real (taker) exit* | **−1,126 bps, CI<0. Dead.** | Once you make the maker actually earn fills away from mid *and pay to close*, it loses badly. |
| **K-PEG** | A "chase" maker that follows the quote, heavily optimised (360 combos) | **+759 bps, looked great** | The standout positive — which triggered the whole audit below. |
| **K-PEG robustness** (us) | Is K-PEG real? Lookahead? Calc bugs? Realizable? | **No bug, no lookahead — but it never pays to exit.** | The +759 is **mark-to-mid**. Force a realistic exit and it flips to **−753 bps**. The exit cost (~1,635 bps half-spread) is bigger than the entire edge. |
| **K-PEG maker-exit** (us) | What if you exit passively (as a maker) instead of crossing? | **−569 bps; only ~12% of exits fill passively; you'd need ~40%.** | The exit is flow-starved just like the entry. Passive exit doesn't rescue it. |
| **Codex independent audit** | Fact-check our audit, adversarially | **Confirmed our core read.** | Refined it: our taker-exit was *too harsh in size*, but most fills are *too close to resolution to exit on Polymarket at all*. Hold-to-resolution was +1,846 bps IS (not robust). |
| **Shape analysis** (us) | Is the edge broad-based or a few lucky fills? | **Mark-to-mid edge is broad-based & consistent (79% win, survives dropping the best 5%).** | This *proves the entry signal is real* — it's not luck. The realizable version is a clean, consistent *loss*. So: real signal, unrealizable on one venue. |
| **K3** | Is the 4h market arb-able vs Binance? | 4h has **no anti-arb fee**; Binance **leads ~10s**; but post-fee basis is weak/IS-only. | The hedge leg is *feasible* (no fee block), but the raw cross-venue edge is thin. |
| **K4** | Risk-free intra-Polymarket arbitrage on our data | **Essentially none** (1 interval, 0 capturable). | Not a live thread on the universe we captured. |
| **K2v3** | Anchor the maker to *Binance* fair value + widen by delta + 3 exit types | **No bucket clears. Binance anchor *increased* adverse selection (325 vs 145 bps).** | The "anchor to the fast venue to stop being picked off" idea **failed its own mechanism test.** |
| **K2v2** | Add a *defensive* rule: pull/widen the quote when Binance moves against you | **−4,316 bps. Defense fired on <0.1% of fills.** | There was nothing to defend against — the toxic fills are **not** preceded by a visible Binance move. Adverse selection here is *structural*, not latency-arb you can dodge. |
| **K6** | Is Polymarket's vol overpriced, and can you gamma-scalp it? | **Vol IS overpriced (+3.7 pts avg, +24 pts far/late). But the scalp is −9.4c, of which 9.56c is hedge cost.** | The signal is real; **continuous** delta-hedging the digital gamma is what kills it. Static hedge untested. |

---

## 4. The two lessons that everything converged on

**Lesson 1 — Mark-to-mid is not money.** K-PEG's whole apparent edge was the difference between valuing a
position at the mid and actually closing it. On a market with 15–30¢ spreads, "the price reverted in my
favour" means nothing if collecting that reversion requires giving most of it back to cross the spread. Our
shape analysis nailed this: the *same fills* are a smooth, broad-based **+$4.60** marked-to-mid and a smooth,
broad-based **−$8.27** when you actually exit. The signal is real; the realizability is not — on one venue.

**Lesson 2 — The adverse selection here is structural, not a latency race.** We kept hoping the toxicity was
"Binance moves, then a fast bot picks off our stale quote 10 seconds later" — because that you can dodge
(anchor to Binance, pull your quote). K2v2 tested exactly that and the defensive trigger almost never fired:
the bad fills are *not* preceded by an observable Binance move. The toxicity comes from the spread itself,
the thin late-window flow, and the resolution jump. You can't out-react it; you can only avoid being there.

Both lessons point the same way: **the problem is always the exit / the carry, never the entry.** So the
surviving ideas are the ones that change how you *exit*, not how you *quote*.

---

## 5. What is now closed (do not re-run)

- Single-venue Polymarket maker, every anchor we tried: **mid-anchored (K2), digital/Binance-anchored
  (K2v3), defensively-anchored (K2v2).** All dead, all for the same structural-adverse-selection reason.
- **Continuous-delta-hedged gamma scalping (K6).** Dead on hedge cost.
- **Intra-Polymarket arbitrage (K4)** on the captured universe.
- (From earlier) the entire **directional** microstructure thesis.

We should **not** keep spawning new quoting/anchor/continuous-hedge variants. Those families are exhausted.

---

## 6. The two strategies still worth pursuing

Both are built on the two things that *survived*: a real passive-entry reversion, and a real
overpriced-longshot/vol premium. Both attack the actual problem — the exit — rather than re-optimising the
quote.

### Strategy A — "Capture and carry": passive maker entry, held to resolution, statically hedged on Binance

**The plain-English idea.** Get filled passively on Polymarket exactly the way K-PEG did (post a resting
quote in the rich, far-from-strike / late-window zone where Polymarket overpays). But then **do not try to
trade back out on Polymarket** — that's the move that loses money. Instead, **hold the position until the
4-hour market resolves to $1 or $0**, and **hedge the price direction on Binance** so you're not exposed to
which way BTC goes. Crucially, use a **static hedge** (set a fixed offsetting position / call-spread once),
*not* a continuously-rebalanced delta hedge — because K6 proved continuous rebalancing near expiry is what
bleeds you dry.

**Why it could work when everything else failed.**
- It never pays Polymarket's exit spread (the thing that turned K-PEG's +$4.60 into −$8.27).
- It harvests *two* real edges at once: the entry reversion (proven broad-based and consistent) **and** the
  overpriced vol/longshot premium (K6's +24 vol points in the far/late bucket).
- It avoids the continuous-hedge cost that killed K6 by hedging statically.
- The hedge leg is feasible: K3 confirmed the 4h market has no anti-arb fee and Binance leads, so you can
  put the hedge on the venue that moves first.

**What you're actually exposed to (the honest risks).**
- **Binary/resolution variance:** even hedged, a digital near the strike is jumpy. The static hedge reduces
  but doesn't eliminate this — you're trading some precision for far lower cost.
- **Basis risk:** Polymarket crypto resolves off Chainlink/Pyth, your hedge is on Binance. If those two
  prices disagree at the wire, the hedge isn't perfect. K3 flagged this; K6 saw 1/24 markets where the
  settlement source disagreed in direction.
- **Fill rate:** you still only get the passive entry fills you get (~166/day crypto) — fine in absolute
  terms, but capacity is bounded.
- **Not yet validated:** hold-to-resolution has only ever shown **positive-mean-but-not-robust** results
  (wide CIs from the binary jump). It has **never been tested with a static hedge**, which is the specific
  variant that could tighten those CIs by removing direction risk cheaply.

**What would prove or kill it (one clean test):** on the K6 panel, take the passive entries in the
far-|z|/late bucket, hold to actual resolution, apply a **static call/put-spread hedge** sized once at
entry, and measure net PnL with non-overlap and CIs. If even this is negative or non-robust, the whole
single-underlying crypto-4h maker/options program is closed and we stop. If it clears, it then needs an
out-of-sample confirmation before any real money.

### Strategy B — "Don't compete, copy": find and mirror the makers who are already profitable

**The plain-English idea.** Instead of building our own maker and discovering it loses, **use the 9 months
of Polymarket trader data we already own to find the wallets that are actually making money market-making
these markets** — then either copy them or reverse-engineer *how* they do it (Are they quoting both sides
continuously? Do they hold to resolution? Which categories? Do they avoid the late-window spike zone?). This
re-merges Block K with the **copytrade** project, which is the repo's nominal primary thread anyway.

**Why it's the right reality check (and possibly the better strategy).**
- Our sims all model a *single-leg, enter-then-exit* maker. A real professional maker runs a **continuous
  two-sided book** — a buy fill is offset by a *later sell fill on the other side*, never by crossing the
  spread. Our accounting structurally bakes in the exit cost a real two-sided maker never pays. So our −3,000
  bps numbers may be too harsh *for that style*, and the only way to know is to look at who's actually
  winning.
- It's **model-free**: realized PnL from settled positions doesn't depend on any of our fill-proxy or
  exit assumptions.
- It works across **all** categories (not just where we have a Binance surface), so it also answers "are
  geopolitics / sports makers profitable?"

**The honest risks / blind spots.**
- **Hedged makers look like losers on Polymarket alone.** If a pro is delta-hedging on Binance (i.e. running
  Strategy A for real), their Polymarket leg can show a *loss* while their combined book profits — we'd only
  see the losing half and wrongly conclude "no edge." Strategy B can identify *unhedged* profitable makers
  cleanly, but can't see hedged ones' true PnL.
- **Concentration/capacity:** if the profit in a market is captured by 1–3 wallets (which the literature
  says is common in thin markets), there may be no room for us even if an edge exists.
- **Attribution care:** we have to correctly identify the *active* wallet (the relayer work established it's
  in the `maker` field, with exchange-internal legs filtered) and not confuse "quotes a lot" with "profits."

**What would prove or kill it:** the K5 analysis — rank real maker wallets by realized, fee-adjusted PnL on
crypto-4h (and by category), measure their behaviour (two-sided? hold-to-resolution? spike-zone avoidance?),
and their true post-fill markout (the *real* adverse selection, to check whether our sims were too harsh).
If profitable makers exist and we can see how, we copy/adopt. If even the best are flat-to-negative, that
*confirms* our sims and closes the maker thesis empirically, model-free.

### How the two relate
They're complementary, not competing. **Strategy B tells us whether a maker edge exists at all and what it
looks like** (cheap, model-free, uses data we have). **Strategy A is our specific bet on the one mechanism
we can construct ourselves** (entry reversion + overpriced vol, carried to resolution, statically hedged).
Run B first — it's cheaper and it might reveal that the real winners are doing exactly Strategy A (hold +
hedge), which would massively de-risk building it.

---

## 7. Suggested next steps (in order)

1. **Strategy B / K5 first** (cheapest, model-free, highest information): do real makers profit, in which
   categories, and how? This either reveals the playbook or empirically closes the thesis.
2. **Strategy A clean test** (the static-hedge carry-to-resolution on the K6 far/late bucket): the last
   crypto-4h maker/options corner worth one rigorous, OOS-gated test.
3. If both fail, the maker/options pivot is honestly closed and effort returns fully to **copytrade**, the
   primary thread.

Standing guardrails on all of the above: non-overlap, net-of-cost, confidence intervals on every headline,
out-of-sample confirmation before any in-sample positive is called deployable, and never confuse
mark-to-mid with realizable PnL.

---

## 8. UPDATE (2026-05-31): K5 — real makers ARE profitable, and they hand us the playbook

K5 was the empirical reality check behind Strategy B, and it's the most important single result in the
block. Instead of simulating a maker, it measured the **realized, closed-position PnL of real maker wallets**
from 9 months of data — model-free, no fill-proxy or exit assumptions.

**Headline:** real maker-heavy wallets are **profitably** making crypto-4h: **+171 bps, CI [34, 327]**
(clears zero — the first robust positive in the whole block), across 256 wallets / 3,268 markets. Pooled
across categories the top-maker cohort is +145 bps, CI [85, 210].

**It confirms both of our lessons and proves the single-leg sims were measuring the wrong lifecycle.** The
profitable makers do exactly the three things our sims didn't:
- **64% two-sided** — they run a continuous book and offset a buy with a later sell on the other side,
  *never crossing the spread to exit* (the cost that turned K-PEG's +$4.60 into −$8.27).
- **78.8% carried to settlement** — they *hold to resolution* (Strategy A's core move, now validated).
- **0.8% in the late near-50¢ spike zone** — they *avoid the gamma-spike danger area* we identified.

And the real-world markout (true adverse selection) on this cohort is maker-*favorable* on average at 5–60s
— i.e. the entry reversion is real, exactly as our shape analysis showed.

**The big caveat — capacity.** Top-3 wallets capture **~95%** of positive crypto-4h maker profit per market.
So profit exists but is **winner-take-most**: a few established wallets dominate. The live question is no
longer "is there an edge?" (yes) but **"can we be a top-3 maker, or are we structurally outcompeted?"** Also
note the survivorship caveat: closed positions only, so open/unresolved losing inventory is excluded (biases
the number up). And fee-free **geopolitics is negative** (−47 bps, doesn't clear) — the rebate-cushion story
survives the model-free check.

**What this does to the two strategies:**
- **Strategy A is de-risked and re-specified.** We're no longer guessing the structure — the winners' design
  is now known: two-sided + carry-to-resolution + spike-avoidance + (for us) a static external hedge. The
  backtest should replicate *that exact lifecycle*, not the single-leg one.
- **Strategy B is half-answered.** "Do makers profit?" → yes. The remaining question is **why the top-3
  dominate** (speed/queue-position vs capital vs risk appetite to carry at scale) — because that determines
  whether we can compete, and whether execution speed (e.g. a Rust quoter) is even the moat. K2v2 already
  hinted the edge is *structural, not a latency race*, which would mean Python/Midas is enough.

**Revised next steps:** (1) decompose the top-3's source of dominance from the K5 wallet data — this answers
the capacity question and the "do we need Rust" question at once; (2) backtest the winners' playbook as
Strategy A (two-sided + carry + spike-avoid + static hedge, OOS-gated); (3) only then paper-trade on Midas,
and consider Rust *only if* step 1 shows the moat is speed.

---

## 9. Source notes (Obsidian links)

This note is the Block K hub. The findings it summarises, as wikilinks:

- Foundation: [[block_k_maker_options_research]]
- Economics & quoting: [[block_k1_maker_economics_findings]] · [[block_k2_quoting_findings]] · [[block_k2v2_findings]] · [[block_k2v3_findings]]
- Chase + robustness audits: [[block_kpeg_findings]] · [[block_kpeg_robustness_findings]] · [[block_kpeg_robustness_review]]
- Cross-venue, options & vol: [[block_k3_leadlag_findings]] · [[block_k3v2_findings]] · [[block_k3v3h_findings]] · [[block_k3v3h2_findings]] · [[block_k6_vol_findings]]
- Arb scan: [[block_k4_arb_scan_findings]]
- Real-maker reality check: [[block_k5_findings]] · [[block_k5b_findings]]
- Directional falsifier: [[block_optd_ceiling_findings]]
- Upstream closure this builds on: [[block_p3prime_oos_findings]] · [[block_a0c_holdout_retest_findings]] · [[block_a14h_maker_non_overlap_findings]]
- Hubs: [[COWORK]] · [[TODO]] · handoff [[2026-05-30_maker_options_delta_pivot]]
