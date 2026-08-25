---
title: "The market-making model — fundamentals first, additions second"
created: 2026-08-25
status: active
owner: justin
project: polymarket-mm
hubs:
  - strat_market_making
tags:
  - market_making
  - methodology
  - canon
---

# The market-making model — fundamentals first, additions second

> Hub: [[strat_market_making]] (read that first). This note is the canonical description of *what the model is*. The rule that organises it: **the fundamentals are the model; everything else is an addition that must individually earn its place on top of working fundamentals.** The additions are listed last on purpose.

## The fundamentals (the model)

**1. The quote.** Rest a buy order and a sell order around fair value on a chosen market and wait to be filled. Fair value is the *size-weighted* midpoint (it leans toward the side of the book with less resting size, because a thin side signals the likely next move) — measured on our data to track the next price far better than the plain midpoint. In an event market you do not need to own tokens to quote both sides: an order to buy YES plus an order to buy NO is economically a bid and an ask, since buying NO at p is selling YES at 1−p.

**2. Honest accounting.** P&L = completed round-trips **plus the change in value of leftover inventory priced at what you could actually exit at** — the best bid if you're long, the best ask if you're short. Never the midpoint: mid-pricing books half the spread you haven't earned and historically manufactured phantom profits in this repo. This convention is deliberately conservative and is non-negotiable.

**3. Simulated fills are a bracket, not a number.** Public Polymarket book data is anonymous — you cannot see your own queue position, so a backtest must *model* when your resting order would fill. We bound this with two extremes (every book cancellation happens ahead of you → most fills; behind you → fewest fills) and a middle model, and report all three. Any conclusion that only holds at one end of the bracket is not a conclusion. The bracket's middle is an assumption, not a measurement — only real live fills can calibrate it (that is a primary purpose of the live lane).

**4. The evaluation split.** Test generalisation by holding out **whole markets** — a market's entire life, from calm mid-life to violent endgame, stays on one side of the split. Splitting by calendar date instead is a trap this project fell into once: it quarantines the calm regime in training and the violent regime in testing, and makes every strategy look like it fails out-of-sample for reasons that have nothing to do with the strategy. Additional honesty checks on top: does the configuration chosen in training remain the best out of sample (here: yes, almost one-for-one under the whole-market split), and are results deflated for the number of configurations tried.

**5. Market selection.** The clearest empirical structure found so far, on held-out data: **calm politics markets earn during the weeks approaching resolution and give it back in the final days before resolution.** Calmness is judged only from a market's early-life window (so the label can never leak information from the scored period). The first-order profit decision is therefore *which markets, and when in their life, to quote at all* — before any cleverness about how.

**Status of the fundamentals: the design is settled and trustworthy; the profit numbers it produces are preliminary** (fill model and latency unvalidated against reality — see the hub's reliability ledger).

## The main open question: P&L spike anatomy

The fundamental risk in this strategy is the sudden P&L jump. What one dissected episode shows: a market moved 18¢ → 75¢ in minutes; an undefended quoter sold into the entire move and stacked ~1,800 contracts the wrong way, while a defended one crossed the episode roughly flat. Two measurements bracket such episodes — a volume-clocked order-flow imbalance that rose ~30 minutes *before* the move, and post-fill adverse price drift that confirmed *during* it. This is one episode plus aggregate maps, not a causal model. **Understanding when and why these spikes occur — and how early they are detectable — is the project's main open research path.**

## The additions (each one a separate claim, added only on top of working fundamentals)

Each addition was introduced one at a time and re-evaluated, so any improvement is attributable to a specific mechanism. They are defenses, not the edge. In order:

1. **Inventory skew.** Shift the quote centre against your accumulated position (long → quote lower, encouraging sells to you to stop; symmetric for short). Bounds inventory, but the right strength is regime-dependent — a fixed skew reacts too slowly in violent episodes, which is why the later additions exist.
2. **Hard position cap.** Beyond a set inventory, stop adding — only quotes that reduce the position remain. Mechanically truncates the disaster tail (it is what prevents the 1,800-contract stack). Cost: also caps upside on genuinely balanced flow.
3. **Two warning signals with a graduated response.** Signal A (flow imbalance, leads) → suspend new quotes on the exposed side. Signal B (post-fill drift, confirms) → shrink quote size. Both, or at the cap → withdraw and rest a single passive reduce-only quote at a wider price; never dump at the touch. Signal B was the most efficient single flag measured.
4. **Asymmetric repricing.** Chase a moving market slowly; withdraw from it instantly.
5. **Size dampening.** Continuously shrink quoted size on the pressured side as flow imbalance builds. Helped point estimates; never independently certified.

**Preliminary read on the additions:** on politics, the full defended stack improved on the undefended quoter in most held-out markets and was the only configuration positive on every evaluation path — but with too few independent markets (and simultaneous markets sharing news) to certify. Treat as promising, unproven.

## Additions that were tested and rejected (kept as concepts so nobody re-imports them)

- **A classical academic quoting model** (closed-form optimal spread/skew from the equities literature). Its assumptions are wrong for prediction markets: it presumes smoothly diffusing prices and uninformed flow, while Polymarket prices jump to 0/1 and informed flow near resolution is the dominant risk; its spread also *narrows* toward expiry — exactly backwards here. Where it looked good (esports), the certified quantity was *avoided losses from quoting wide*, not profit. Verdict: do not adopt; if a derived-spread formula is ever wanted, it must be re-derived for resolving markets.
- **Basket carry** (quote all outcomes of an event jointly and carry the balanced basket to resolution, exploiting that outcomes sum to $1). Structurally safe, empirically no edge over the simpler defenses in our tests. The *pair-quoting insight* it rests on (buy-YES + buy-NO ≡ bid + ask; merge/split routes) is live and folded into fundamental #1 — it's the strategy variant that's parked, not the arithmetic.

## What would change this note

Real fills. The live lane's next step (one persistent order on a market chosen for fillability) calibrates the fill model and replaces the instant-order assumption with the measured 160 ms. Until then, every number here is a modelled number.
