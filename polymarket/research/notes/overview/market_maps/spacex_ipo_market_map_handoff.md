---
title: "SpaceX IPO Cross-Market Handoff"
tags: [spacex, ipo, polymarket, hyperliquid, tradingview, market-map, handoff]
created: 2026-06-07
audience: "Cowork/Codex sessions building SpaceX IPO data agents and EV screens"
status: "live market map; refresh prices before trading decisions"
---

# SpaceX IPO Cross-Market Handoff

> Hub: [[COWORK]] · [[POLYMARKET_BRAIN]]
> Table terms: [[polymarket_table_dictionary]]
> Companion addendum: [[spacex_ipo_coworker_addendum]]
> Offline go/no-go gate for the long-IPO / short-perp convergence trade: [[spcx_convergence_calc_findings]]

## Plain-English Summary

- SpaceX exposure is split across different instruments that look linked but are not the same trade: official stock after IPO, Polymarket event contracts, Hyperliquid synthetic perpetuals, TradingView chart/order-routing, and proxy funds or tokenized products.
- As of 2026-06-07 15:18 UTC, the market complex prices a June 2026 IPO as near-certain and centers first-day closing market cap around roughly $1.5T-$2.5T. Polymarket's threshold curve gives a rough expected closing market cap near $2.19T if treated as a survival curve, but that is an approximation, not a true risk-neutral fair.
- The main research hypothesis is bearish: SpaceX may be overvalued. To get high conviction, the team must normalize every venue into the same unit, compare implied market cap to S-1 share counts and official IPO terms, and track funding, liquidity, settlement rules, and post-listing conversion mechanics.
- The best next build is not one monolithic trading bot. Build small agents that collect official filings, Polymarket CDF/order books, Hyperliquid perp state, proxy fund marks/NAVs, and post-listing TradingView/broker data, then feed a single valuation-normalization and EV layer.
- A companion note, [[spacex_ipo_coworker_addendum]], incorporates the new DOCX/PNG angle: Class A vs Class B, `xyz:SPCX` vs `vntl:SPACEX`, Trade Republic vs TradingView workflow, and the coworker's PCHIP distribution.

This is research context, not investment advice. Many products below are restricted by geography, KYC status, or venue terms.

## Repo Context

This note was written after pulling `origin/main` on 2026-06-07. The pull fast-forwarded the repo from `ef24478` to `126228c` and preserved the existing local dirty worktree via Git autostash.

The newest pulled commits are broad Midas/Polymarket execution infrastructure, not a SpaceX-specific trade package. Existing repo data already contains SpaceX Polymarket market rows in generated screens, but the repo did not yet have a structured cross-market SpaceX handoff.

## Current Tradable Surfaces

| route | what you are actually trading | current read | source links |
|---|---|---|---|
| Official IPO / brokers | SpaceX Class A common stock after the registration statement is effective and the stock lists under `SPCX`. | SpaceX announced a roadshow on 2026-06-04 for 555,555,555 Class A shares at an expected $135 IPO price, with Nasdaq/Nasdaq Texas ticker `SPCX`. Sales still require SEC effectiveness. | [SpaceX IPO release](https://content.spacex.com/cms-assets/FINAL_Documents%20and%20Updates/6.4.26_SpaceX_Announces_IPO_US.pdf), [SEC S-1/A index](https://www.sec.gov/Archives/edgar/data/1181412/000162828026040364/0001628280-26-040364-index.htm) |
| Polymarket / PM | Event contracts: timing, first-day closing market cap thresholds, and valuation brackets. You are trading a resolution rule, not shares. | PM prices June IPO as near-certain. The most liquid cap-threshold surface is `SpaceX IPO closing market cap above ___ ?`, about $6.95M volume. | [PM cap-above event](https://polymarket.com/event/spacex-ipo-closing-market-cap-above), [PM market data docs](https://docs.polymarket.com/market-data/overview) |
| Hyperliquid `xyz:SPCX` | Cash-settled pre-IPO perpetual via trade[XYZ]/HIP-3, not equity and not IPO allocation. | 2026-06-07 15:59 UTC API snapshot: 5x max leverage, strict isolated margin, mark $173.53, oracle $173.35, about $16.38M 24h notional, OI 353,134.78 SPCX. | [Hyperliquid info endpoint](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint), [trade[XYZ] IPOP docs](https://docs.trade.xyz/asset-directory/pre-ipo-perpetuals-ipops.md), [trade[XYZ] IPOP spec index](https://docs.trade.xyz/consolidated-resources/specification-index-ipops.md) |
| Hyperliquid `vntl:SPACEX` | Separate Ventuals synthetic valuation perp on Hyperliquid. It is a different namespace/product from `xyz:SPCX`. | 2026-06-07 15:59 UTC API snapshot: 3x max leverage, strict isolated margin, mark 2059.5, oracle 1866.6, about $165k 24h notional, OI 1,178.548. Ventuals defines private-company price as valuation divided by $1B, so the mark is about $2.059T. | [Ventuals private-company docs](https://docs.ventuals.com/perp-specifications/private-companies.md), [Ventuals API](https://docs.ventuals.com/developers/api) |
| TradingView | Charting, alerts, Pine scripts, and broker order routing. TradingView is not the economic exposure unless routed through a connected broker. | Pre-listing, it is mostly a watch/alert layer. After listing, chart `NASDAQ:SPCX` if available and trade only through a broker that supports it. Trade Republic should be treated as a separate in-app IPO allocation workflow unless a TradingView integration appears. | [TradingView trading](https://www.tradingview.com/trading/), [TradingView brokers](https://www.tradingview.com/brokers/), [Trade Republic IPO access notice](https://www.tradingview.com/news/eqs%3Ac0a999269094b%3A0-starting-today-trade-republic-gives-european-retail-investors-direct-access-to-ipos/) |
| Proxy funds / baskets | Indirect public-market exposure to SpaceX through funds that hold private shares, SPVs, or baskets. | Useful for sentiment and NAV/premium comparison, not clean short-term SpaceX exposure. Examples: XOVR, DXYZ/Tech100, BPTRX. | [XOVR](https://entrepreneurshares.com/xovr-etf/), [D/XYZ FAQ](https://destiny.xyz/faq), [Baron BPTRX](https://www.baroncapitalgroup.com/product-detail/baron-partners-fund-bptrx) |
| Other crypto pre-IPO venues | Tokenized or synthetic pre-IPO products such as BingX and Pionex. Mechanics differ by venue. | Useful as extra price feeds, but each product has its own settlement/conversion terms and regional restrictions. | [BingX SpaceX Pre-IPO](https://bingx.com/en/support/articles/15802997358735), [Pionex SPCX disclaimer](https://www.pionex.com/blog/spcx-pre-ipo-introduction-and-disclaimer/) |
| Private secondaries | Forge/EquityZen/Hiive-style private share transactions. | Forge now says SpaceX trading is no longer permitted after the S-1, so this is mostly historical/private-market context. | [Forge SpaceX](https://forgeglobal.com/spacex_stock/) |

## Current Market Snapshot

Refresh these numbers before acting. They are a 2026-06-07 15:18 UTC snapshot.

### Official IPO Anchor

- SpaceX release: roadshow launched 2026-06-04; expected IPO price $135; 555,555,555 Class A shares offered; Nasdaq Global Select Market and Nasdaq Texas application under `SPCX`; registration statement not yet effective.
- Latest S-1/A index checked: accession `0001628280-26-040364`, filed 2026-06-03.
- S-1/A share-count anchor after offering, no underwriter option: 7,380,196,910 Class A shares plus 5,695,668,265 Class B shares, or 13,075,865,175 total Class A/Class B shares.
- Therefore, using the no-option share count, $135 implies about $1.765T market cap. If the underwriters exercise the option, the share base changes. Do not mix share-count conventions.
- Class A is the listed/public IPO class and carries one vote per share. Class B carries 10 votes per share, is not the listed public class, and generally converts into Class A on transfer subject to permitted-transfer exceptions.
- Do not confuse the 555,555,555 Class A shares offered in the IPO with the full post-offering Class A count. Public float/offered shares are much smaller than total outstanding Class A plus Class B.

### Polymarket Timing

| PM event | current price / read |
|---|---:|
| IPO by 2026-06-15 | 95.65% YES |
| IPO by 2026-06-30 | 99.35% YES |
| IPO by 2026-12-31 | 99.65% YES |
| IPO in June 2026 | 98.8% YES |
| Fail to IPO by 2026-12-31 | 0.6% YES |

Plain English: PM is no longer pricing "will there be an IPO?" as the main uncertainty. The live debate is mostly "where does the first-day close land?"

### Polymarket First-Day Closing Market Cap

Bracket market:

| closing market cap bucket | YES price |
|---|---:|
| less than $1.0T | 0.85% |
| $1.0T-$1.5T | 3.85% |
| $1.5T-$2.0T | 38.0% |
| $2.0T-$2.5T | 42.5% |
| $2.5T-$3.0T | 12.2% |
| $3.0T-$3.5T | 3.55% |
| at least $3.5T | 1.05% |
| no IPO by 2027-12-31 | 0.35% |

Threshold market:

| threshold | PM probability close is above threshold |
|---|---:|
| > $1.6T | 91.8% |
| > $1.8T | 77.5% |
| > $2.0T | 63.5% |
| > $2.2T | 46.5% |
| > $2.4T | 29.5% |
| > $2.6T | 14.5% |
| > $2.8T | 10.5% |
| > $3.0T | 6.5% |
| > $3.2T | 3.55% |
| > $3.4T | 2.5% |
| > $3.6T | 1.9% |
| > $3.8T | 1.6% |
| > $4.0T | 1.1% |

Approximation: treating the threshold prices as a survival curve `S(K)=P(V>K)` and adding a simple tail to zero after $4.5T gives an approximate expected closing cap of $2.19T, or about $167/share on 13.076B shares. This is a working diagnostic only: PM order books have spreads, independent markets can disagree, and event-contract prices include risk premia and fees/slippage.

### Hyperliquid Synthetic Perps

| market | mark | oracle | rough normalized read | 24h notional | OI | leverage / margin |
|---|---:|---:|---|---:|---:|---|
| `xyz:SPCX` | $173.53 | $173.35 | Share-like trade[XYZ] IPOP; mark implies about $2.269T on 13.076B shares. | $16.38M | 353,134.78 SPCX | 5x, strict isolated |
| `vntl:SPACEX` | 2059.5 | 1866.6 | Ventuals valuation perp; mark means about $2.059T because price is valuation divided by $1B. | $165k | 1,178.548 | 3x, strict isolated |

Plain English: `xyz:SPCX` is the active/high-volume share-like synthetic price. `vntl:SPACEX` is a separate, thinner valuation product and should not be treated as the same quote.

## How The Markets Link

### Plain English

The official IPO is the anchor. It tells us the expected IPO price, ticker, share count, listing venue, and legal status.

Polymarket turns the IPO into yes/no questions. A PM share at 64 cents on `SpaceX IPO closing market cap above $2T?` is roughly the market saying "about 64% chance this resolves YES," before adjusting for spread, slippage, and liquidity.

Hyperliquid turns the same story into a live leveraged price. Instead of waiting for an event to resolve, traders can long or short a synthetic SpaceX valuation number 24/7. This makes it fast and scalpable, but also exposes the trader to funding, liquidation, oracle/mark behavior, and venue risk.

TradingView is the screen and workflow layer. It can help chart the listed stock later, run alerts, or route orders through a connected broker, but it is not a separate SpaceX trade by itself.

Proxy funds are slower indirect exposures. They tell us how public wrappers mark or price SpaceX exposure, but they include fund discounts/premiums, other holdings, fees, and stale private valuations.

### Technical Map

Normalize everything into first-day closing market cap:

```text
market_cap_trillions = share_price * shares_outstanding / 1e12
share_price = market_cap_trillions * 1e12 / shares_outstanding
```

Using the S-1/A no-option count of 13,075,865,175 shares:

| share price | implied market cap |
|---:|---:|
| $135.00 official IPO price | $1.765T |
| $150.00 reference price | $1.961T |
| $167.30 rough PM threshold-curve EV | $2.188T |
| $173.53 `xyz:SPCX` mark | $2.269T |

For Polymarket:

```text
S(K) = price of YES on "closing market cap above K"
P(a < V <= b) ~= S(a) - S(b)
EV[V] ~= integral_0_infinity S(K) dK
```

For Hyperliquid:

```text
perp_mark ~= oracle/reference price + market premium
PnL is cash-settled in stablecoin
funding transfers value between longs and shorts
liquidation depends on leverage, margin mode, and mark/oracle behavior
```

For relative value:

```text
perp_implied_cap - PM_implied_cap = possible richness/cheapness signal
```

But this is not a clean arbitrage. PM resolves to a specific event rule, usually first-day closing market cap. Perps are continuous contracts with funding and liquidation. Proxy funds have NAV and wrapper effects. Tokenized venues may convert or settle based on their own benchmarks. TradingView does not create exposure.

## What The Market Currently Implies

The public IPO price at $135 and roughly 13.076B post-offering shares anchors around $1.765T.

PM says the closing cap is very likely above the IPO-price-implied cap. Interpolating the threshold curve around $1.765T gives roughly 80% probability above the IPO price, though that interpolation should be refreshed and stress-tested.

PM's biggest mass is $1.5T-$2.5T. The market is not mainly pricing a disaster IPO; it is pricing a large first-day premium as plausible but not guaranteed.

`xyz:SPCX` currently looks richer than the rough PM EV if normalized share-like: $173.53 implies about $2.269T, while the rough PM EV diagnostic is about $2.19T. That spread is not automatically tradable, but it is exactly the kind of basis a SpaceX agent should monitor.

Our bearish hypothesis should be framed as:

> SpaceX may be worth less than the cross-market complex is implying, especially if first-day scarcity/retail reflexivity and synthetic-perp momentum are overpricing fundamentals.

That hypothesis needs evidence, not vibes. High-conviction evidence would include weak S-1 fundamentals versus valuation, weakening IPO demand/order-book reports, falling secondary/proxy marks, PM threshold compression, negative funding-adjusted perp basis, and lower-liquidity/high-leverage unwind risk.

## Candidate Trade Expressions

| expression | how it makes money | main risk |
|---|---|---|
| Buy PM NO on high thresholds, e.g. `NO >$2.4T` or `NO >$2.6T` | Wins if first-day closing cap lands below that threshold. Clean event settlement, no liquidation. | Capital tied until resolution; event-rule risk; PM liquidity/spread; cannot exit if market dries up. |
| PM bracket trades | Buy underpriced bucket, sell/avoid overpriced buckets. | Multi-market consistency is imperfect; independent liquidity; resolution wording must be checked. |
| Short `xyz:SPCX` perp | Wins if synthetic share-like price falls. Faster and more liquid than PM. | Liquidation, funding, weekend/24h squeeze, oracle/mark behavior, no guarantee convergence to PM. |
| Short `vntl:SPACEX` perp | Similar bearish synthetic expression on a valuation-like product. | Much thinner, separate oracle/product, possible venue-specific shock. |
| Relative value: short rich perp, long PM upside tail | Attempts to isolate synthetic richness versus event distribution. | Different settlement, leverage, funding, timing, and tail behavior; hedge can fail when volatility matters most. |
| Proxy-fund/NAV discount watch | Uses DXYZ/XOVR/BPTRX as public sentiment and mark references. | Not pure SpaceX; fund wrapper effects can dominate. |
| Post-listing broker trade | Trade actual `SPCX` stock after listing/effectiveness. | IPO open/close volatility, borrow availability for shorts, broker restrictions, halts. |

For max EV, the initial screen should compare `edge = model_probability - market_price` for PM contracts and `edge = model_fair_cap - venue_implied_cap - funding/liquidity cost` for perps.

## Agent Build Plan

### 1. Official Filing Agent

Purpose: maintain the legal/reference anchor.

Pull:

- SpaceX press releases and `spacexipo.com`.
- SEC EDGAR filings for CIK/entity `1181412`, especially S-1/A, EFFECT, 424B, 8-A12B.
- Nasdaq listing/ticker status after effectiveness.

Store:

- Filing timestamp.
- Offered shares.
- IPO price/range.
- Class A/Class B post-offering shares.
- Lockup/transfer language.
- Related-party and xAI/Starlink disclosures that might affect valuation.

### 2. Polymarket CDF Agent

Purpose: turn PM into a live first-day closing cap distribution.

Pull:

- Gamma event slugs:
  - `spacex-ipo-by`
  - `in-which-month-will-spacex-ipo`
  - `spacex-ipo-closing-market-cap`
  - `spacex-ipo-closing-market-cap-above`
- CLOB token IDs, best bid/ask, midpoint, spread, liquidity, volume, recent trades.

Compute:

- Monotone threshold curve.
- Bracket probabilities.
- Approximate EV, median, and percentiles.
- Cross-market inconsistency: `bracket(a,b)` versus `S(a)-S(b)`.
- Best EV contracts under the team's model.

### 3. Hyperliquid Perp Agent

Purpose: monitor fast synthetic price discovery and liquidation/funding risk.

Pull through Hyperliquid `/info` and websockets:

- `metaAndAssetCtxs` for `dex:"xyz"` and `dex:"vntl"`.
- `xyz:SPCX` and `vntl:SPACEX` mark, oracle, mid, impact prices, OI, funding, volume.
- Order book depth and trade prints.
- Liquidations if available through the chosen data path.

Compute:

- `xyz:SPCX` implied cap using selected share count.
- `vntl:SPACEX` normalized cap after confirming units.
- Perp basis versus PM EV and versus official IPO price.
- Funding-adjusted carry for shorts/longs.
- Squeeze risk: OI / depth, impact spread, liquidation clusters.

### 4. TradingView / Listed-Stock Agent

Purpose: charting and post-listing workflow, not primary pre-IPO data.

Use:

- TradingView alerts/Pine for human-readable levels after `NASDAQ:SPCX` exists.
- Broker APIs directly for execution/fills when possible.
- Do not rely on scraping TradingView as the canonical data source if direct exchange/broker feeds are available.

### 5. Proxy/NAV Agent

Purpose: measure public wrapper sentiment and private-market marks.

Pull:

- XOVR holdings/NAV/premium and SpaceX SPV exposure notes.
- BPTRX holdings, especially SpaceX weight.
- DXYZ/Tech100 holdings and NAV/premium if available.
- Forge SpaceX page as secondary-price context; Forge showed $128.90 and $1.53T valuation on 2026-06-07 while saying trading is no longer permitted.
- Pionex/BingX SpaceX product pages as extra synthetic venue references.

Compute:

- Proxy-implied SpaceX mark, if extractable.
- Premium/discount to NAV.
- Lag between proxy marks and PM/perp changes.

## Research Questions For High Conviction

1. What is the latest defensible share count: basic, diluted, no-option, full-option, or venue-specific convention?
2. Does the PM CDF imply an EV materially above what S-1 fundamentals can support?
3. Is `xyz:SPCX` persistently rich versus PM-implied EV after funding and spread?
4. Are high thresholds mispriced because traders anchor on "biggest IPO ever" headlines rather than share-count math?
5. Does proxy-fund pricing confirm or contradict the synthetic/perp premium?
6. Is short interest/borrow available after listing, or is the clean bearish expression pre-listing PM NO/perp short?
7. What is the worst squeeze/liquidation path if the bearish thesis is right but timing is wrong?

## Do Not Make These Mistakes

- Do not compare `vntl:SPACEX` 2059.5 directly to `xyz:SPCX` 173.53 without normalizing units. The first is valuation divided by $1B; the second is share-like.
- Do not compare PM first-day-close contracts to a perp mark as if they settle on the same thing.
- Do not use the Claude artifact's 13.1B share count without noting it is a rounded convention. The current S-1/A no-option Class A/Class B count is 13.075865175B.
- Do not treat 555.6M offered Class A shares as the full share count for market-cap normalization.
- Do not treat TradingView as an exposure venue.
- Do not assume Trade Republic can be fixed by TradingView broker routing. Use TradingView for charts/alerts and Trade Republic separately for IPO subscription unless a supported broker integration appears.
- Do not assume tokenized products convert into real stock unless the venue's settlement document says so and the user's jurisdiction is eligible.
- Do not confuse "SpaceX is overvalued" with "short now at any price." The trade needs timing, carry, liquidation, and exit analysis.

## Source Links

- [SpaceX IPO release, 2026-06-04](https://content.spacex.com/cms-assets/FINAL_Documents%20and%20Updates/6.4.26_SpaceX_Announces_IPO_US.pdf)
- [SEC S-1/A index, accession 0001628280-26-040364](https://www.sec.gov/Archives/edgar/data/1181412/000162828026040364/0001628280-26-040364-index.htm)
- [Polymarket IPO-by event](https://polymarket.com/event/spacex-ipo-by)
- [Polymarket IPO month event](https://polymarket.com/event/in-which-month-will-spacex-ipo)
- [Polymarket closing-cap bracket event](https://polymarket.com/event/spacex-ipo-closing-market-cap)
- [Polymarket cap-above event](https://polymarket.com/event/spacex-ipo-closing-market-cap-above)
- [Polymarket market data docs](https://docs.polymarket.com/market-data/overview)
- [Polymarket geographic restrictions](https://help.polymarket.com/en/articles/13364163-geographic-restrictions)
- [Hyperliquid info endpoint docs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint)
- [Hyperliquid `xyz:SPCX` trade page](https://app.hyperliquid.xyz/trade/xyz:SPCX)
- [Hyperliquid `vntl:SPACEX` trade page](https://app.hyperliquid.xyz/trade/vntl:SPACEX)
- [trade[XYZ] API docs](https://docs.trade.xyz/api/api)
- [trade[XYZ] Pre-IPO Perpetuals docs](https://docs.trade.xyz/asset-directory/pre-ipo-perpetuals-ipops.md)
- [trade[XYZ] IPOP specification index](https://docs.trade.xyz/consolidated-resources/specification-index-ipops.md)
- [trade[XYZ] IPOP risk disclaimer](https://docs.trade.xyz/legal-and-disclaimers/pre-ipo-perpetual-markets-risks-and-disclaimers.md)
- [trade[XYZ] perp mechanics](https://docs.trade.xyz/perp-mechanics/overview)
- [trade[XYZ] access/geography note](https://docs.trade.xyz/about-trade-xyz/introduction)
- [Ventuals overview](https://docs.ventuals.com/)
- [Ventuals private-company perp docs](https://docs.ventuals.com/perp-specifications/private-companies.md)
- [Ventuals API docs](https://docs.ventuals.com/developers/api)
- [TradingView trading overview](https://www.tradingview.com/trading/)
- [TradingView broker directory](https://www.tradingview.com/brokers/)
- [TradingView unsupported-broker help](https://www.tradingview.com/support/solutions/43000479602-the-broker-i-want-to-trade-through-is-not-supported-can-you-add-it/)
- [Trade Republic IPO access announcement on TradingView News](https://www.tradingview.com/news/eqs%3Ac0a999269094b%3A0-starting-today-trade-republic-gives-european-retail-investors-direct-access-to-ipos/)
- [Forge SpaceX stock page](https://forgeglobal.com/spacex_stock/)
- [XOVR ETF](https://entrepreneurshares.com/xovr-etf/)
- [D/XYZ FAQ](https://destiny.xyz/faq)
- [Baron Partners Fund BPTRX](https://www.baroncapitalgroup.com/product-detail/baron-partners-fund-bptrx)
- [BingX SpaceX Pre-IPO page](https://bingx.com/en/support/articles/15802997358735)
- [Pionex SPCX Pre-IPO disclaimer](https://www.pionex.com/blog/spcx-pre-ipo-introduction-and-disclaimer/)
