# OD Cross-Asset Up/Down Expansion Scoping

> Hub: [[strat_options_delta]] · [[strat_market_making]] · [[POLYMARKET_BRAIN]]
> Related: [[od_pricing_model_form_findings]] · [[od_rv_deribit_daily_scoping_findings]] · [[block_k5b_findings]] · [[mm_deployable_cells_findings]]

## Thesis

The 2026-06-02 OD close is narrow: our standalone fair-value model did not prove a deployable edge on PM crypto-4h digitals after better probability calibration, jump tails, structural queue baseline, and top-maker capacity haircuts. It does **not** prove that PM up/down markets are untradeable, that top-maker share is impossible to steal, or that OD cannot work on other reference assets with better external price/option APIs.

The live reopen candidate is broader and more market-vs-market:

```text
PM binary / price-band market
vs
external reference price, option surface, futures, or internal negative-risk consistency
```

This is a different class from "our crypto-4h model vs PM." It asks whether PM financial binaries are stale, mispriced, or internally inconsistent relative to more liquid reference markets.

## What The Old Failures Do And Do Not Say

- K5/K9 capacity does not prove we can never take top3/next20 fill share. It says historical positive maker dollars were crowded under observed fills, and median non-incumbent headroom was small. A faster maker, a better-priced maker, or a maker quoting less crowded markets could still capture share. Historical files lack quote placements, cancels, queue position, and missed fills, so speed/queue can only be proxied.
- K2v2 did not find a simple Binance-staleness dodge in crypto-4h: the defensive pull/widen rule fired on <0.1% of fills. That weakens the pure "Binance moved, PM stale" story for the captured crypto set, but it does not test equities, indexes, price-band markets, or neg-risk baskets.
- OD v4/model-form closed the absolute-pricing version on crypto-4h. It does not close external-surface pricing, daily horizons, equity/index underlyings, or PM-internal consistency.

## Candidate Market Families

1. **Crypto daily vs Deribit 1-day options.** Already scoped separately. Needs forward PM+Deribit dual capture because DVOL and expired-instrument OHLC are not enough.
2. **Index up/down and price bands.** Examples: S&P 500 / Nasdaq-style PM markets, SPY/SPX/ES/NQ reference prices, and listed option/futures surfaces. Digital fair can be approximated from tight call spreads or option-implied CDFs.
3. **Single-stock up/down and close-above/close-below markets.** Examples: liquid names with live equity quotes and listed options. These may have cleaner APIs than PM crypto settlement, but require careful exchange-hours, halt, dividend, and official-close handling.
4. **Close at/above price-level markets.** These are direct digital or call-spread claims. They fit OD better than generic direction because the strike is explicit.
5. **Negative-risk / mutually exclusive baskets.** These need PM-internal consistency checks: outcome sums, monotonicity across price bands, and merge/split accounting. This is an MM/OD hybrid because fair value may come from the basket itself plus an external reference.

## Proposed Gates

### Gate 0: Universe Map

Build a current PM financial-binary universe:

- family: crypto daily, index, single-stock, price band, neg-risk
- resolution source and exact settlement timestamp
- available external API/reference source
- listed option/futures availability
- fees/rebates and tick size
- volume, spread, top3 maker share, and incumbent concentration

### Gate 1: Model-Free Historical Calibration

For resolved markets, check whether low-price tails, favorites, and price bands were miscalibrated without using any external options model. This is K7-style and should not be enough to trade by itself.

### Gate 2: External Fair-Value Test

For markets with liquid references, compare PM prices to causal external fair values:

- option-surface digital via call spreads or implied CDF
- futures/spot reference for lead-lag and settlement basis
- bid/ask-aware cost floors, not midpoint-only basis

### Gate 3: Execution And Capacity

Replay or paper-test passive quotes:

- quote just inside the incumbent when external fair leaves edge
- log missed fills, queue position proxies, cancels, and latency
- estimate whether better pricing can steal top3/next20 share without becoming adverse-selected

### Gate 4: OOS Deployability

Only call it reopened if a pre-registered family clears:

- net of PM fees/rebates, external hedge/option costs, and reference-basis risk
- cluster-bootstrap lower CI > 0
- enough independent markets
- realistic non-incumbent fill share
- capacity above a minimum daily EV threshold

## Claude/Cowork Prompt Seed

Use this if discussing with Claude/Cowork:

```text
Please evaluate a narrow reopen of OD/MM, not the closed crypto-4h absolute-pricing branch.

Existing evidence:
- PM crypto-4h model-vs-market OD is closed after conditional probability, jump-model, and structural-incremental tests.
- K5 proves real makers profit, but K5b/K9 say historical profit is top-maker/capacity concentrated.
- That capacity result does not prove a faster/better-priced entrant cannot steal fill share, because historical data lacks quotes, cancels, queue position, and missed fills.

Question:
Should we scope a new cross-asset PM financial-binary branch: PM daily crypto / index / single-stock up-down / close-above-price / neg-risk markets vs external APIs and option surfaces?

Focus:
1. Which PM financial binary families have external reference APIs or listed options/futures good enough for causal fair value?
2. Is this market-vs-market relative value rather than another homemade fair-value-model bake-off?
3. How should we test whether better pricing or speed can steal top3/next20 fill share?
4. What pre-registered gates should prevent us from relitigating the closed crypto-4h branch?
```

## Decision Status

Open as a **scoping candidate**, not a reopened strategy. The next useful work is a universe/API/capacity map, not another crypto-4h pricing model.
