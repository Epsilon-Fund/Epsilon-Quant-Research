---
title: "SpaceX IPO Coworker Addendum"
tags: [spacex, ipo, polymarket, hyperliquid, trade-republic, tradingview, addendum]
created: 2026-06-07
audience: "Cowork/Codex sessions using the coworker DOCX and PDF chart as execution-facing inputs"
status: "companion addendum; verify execution claims before trading"
source_files:
  - "/Users/justiniturregui/Downloads/SpaceX_IPO_Research_Handoff_v2.docx"
  - "/Users/justiniturregui/Downloads/spacex_pdf_analysis.png"
---

# SpaceX IPO Coworker Addendum

> Hub: [[COWORK]] · [[POLYMARKET_BRAIN]]
> Companion: [[spacex_ipo_market_map_handoff]]
> Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

- The coworker files add an execution-facing angle to the main market map: Trade Republic IPO allocation, day-1 sell/pro-rata unknowns, a richer Polymarket PCHIP distribution, Hyperliquid `SPCX` IPOP mechanics, and the important capital-structure correction.
- Keep this note separate from the primary handoff for now because some claims are venue-support questions, not facts yet: Trade Republic day-1 sale capability, allocation size, flipping penalties, and exact operational timing for perp conversion.
- The biggest immediate corrections are simple: SpaceX has two common stock classes after the IPO, `xyz:SPCX` and `vntl:SPACEX` are different Hyperliquid products with different units, and Trade Republic is not currently a TradingView broker-routing fix.

## What The Coworker Files Add

| topic | import into our working model | confidence |
|---|---|---|
| Capital structure | Two-class common-stock structure is the right frame. Public investors receive Class A, one vote per share. Class B carries 10 votes per share and is not the listed IPO class. | High, but use SEC counts instead of the DOCX simplification. |
| Polymarket distribution | The PCHIP/Fritsch-Carlson fit is more useful than a rough trapezoid because it gives mean, median, mode, percentiles, skew, and bucket-consistency checks. | Medium-high, but refresh the 16-strike PM data before trading. |
| PM bucket edge | The coworker distribution says the `$1.5T-$2.0T` bucket is materially richer than the continuous threshold curve implies. | Interesting, but must rebuild from live order books. |
| Hyperliquid `xyz:SPCX` | trade[XYZ] docs classify SpaceX `SPCX` as a live pre-IPO perpetual (`IPOP`) referencing expected Class A price per share, not shares or IPO allocation rights. | High from trade[XYZ] docs. |
| Hyperliquid `vntl:SPACEX` | Ventuals docs say private-company perps trade valuation divided by $1B. A displayed `2059.5` is about $2.059T valuation, not $2,059/share. | High from Ventuals docs. |
| Trade Republic | Trade Republic announced in-app IPO subscriptions, official allocation price, pro-rata allocation, and a 1 EUR settlement fee. | Medium-high, but day-1 sell is still unconfirmed. |
| TradingView | TradingView is charting/alerts/order-routing only where a broker integration exists. Trade Republic did not appear on the checked TradingView broker list. | High enough for workflow design. |

## Class A vs Class B

The coworker DOCX is right to correct the earlier mental model away from "three classes." The relevant post-IPO common-stock structure is Class A and Class B.

Use this distinction:

| concept | Class A | Class B |
|---|---:|---:|
| Listed IPO class | Yes, `SPCX` | No |
| Votes per share | 1 | 10 |
| Offered to public in IPO | 555,555,555 shares | No |
| Official post-offering count, no option | 7,380,196,910 | 5,695,668,265 |
| Aggregate voting power after offering | about 11.5% | about 88.5% |
| Transfer behavior | Listed/tradable class | Generally converts to Class A on transfer, subject to permitted transfers |

Important nuance: "555.6M Class A shares offered to the public" is not the same as "only 555.6M Class A shares exist." The S-1/A post-offering table shows about 7.38B Class A shares and 5.70B Class B shares outstanding after the offering, before the underwriter option. Public float is much smaller than total Class A outstanding because insiders/conversions/lockups can sit inside the Class A count.

Why this matters for market work:

- Polymarket day-1 market-cap resolution should be normalized to total outstanding shares, not only public float.
- The official IPO price of $135 implies about $1.765T using the S-1/A no-option total of 13.075865175B Class A plus Class B shares.
- The coworker artifact used about 13.091B shares. That rounded convention is close enough for visual exploration, but production agents should store the exact share-count convention beside every valuation.
- Musk's personal voting control is a separate number from aggregate Class B voting power. The S-1/A indicates roughly 82.4% personal voting power after offering at the expected IPO price, while aggregate Class B voting power is about 88.5%.

## `xyz:SPCX` vs `vntl:SPACEX`

These are not interchangeable tickers.

| field | Hyperliquid `xyz:SPCX` | Hyperliquid `vntl:SPACEX` |
|---|---|---|
| Deployer / namespace | trade[XYZ] / `xyz` | Ventuals / `vntl` |
| Product type | Pre-IPO perpetual (`IPOP`) that references expected SpaceX Class A share price | Private-company valuation perpetual |
| Unit | Dollars per expected Class A share | Company valuation divided by $1B |
| Live API snapshot, 2026-06-07 15:59 UTC | mark $173.53, oracle $173.35, 5x max, strict isolated, about $16.38M 24h notional | mark 2059.5, oracle 1866.6, 3x max, strict isolated, about $165k 24h notional |
| Normalized read | $173.53 x 13.075865175B = about $2.269T implied cap | 2059.5 means about $2.059T valuation |
| Main risk | It can gap when internal pre-listing pricing converts to external equity pricing; 5x leverage makes timing risk brutal. | It is thinner, uses Ventuals' valuation oracle/settlement rules, and settles/resolves as a valuation product. |

trade[XYZ] docs explicitly say IPOPs are cash-settled derivatives, not shares, not tokenized equity, and not IPO allocation rights. For `SPCX`, the IPOP spec says the market is live, initial price $150, discovery bound +/-20%, max leverage 5x, strict isolated margin, open-interest cap $150M, and anticipated listing date June 12, 2026. The generic IPOP docs say pre-listing oracle pricing uses a 30-minute EWMA/internal mechanism, pre-conversion funding samples are 1% of standard XYZ perp funding (`0.005` multiplier), and the market is expected to convert to a normal equity perp after the underlying begins regular trading and external data is sufficient.

Ventuals docs say private-company perps trade valuations, not shares: `price = company valuation / 1,000,000,000`. For `vntl:SPACEX`, that means a displayed mark near `2059.5` is roughly a $2.059T valuation quote. Ventuals also states its pre-IPO markets resolve when the company goes public, with the resolution price based on total valuation from basic shares outstanding times the first-day closing share price.

Plain-English translation: `xyz:SPCX` asks, "Where will the expected Class A share price trade?" `vntl:SPACEX` asks, "What total SpaceX valuation should the private-company perp show?" They can disagree because their units, oracle construction, liquidity, funding, settlement, and deployers differ.

## TradingView And Trade Republic

The right fix is a workflow split, not an integration assumption.

| job | best current tool |
|---|---|
| IPO subscription/allocation | Trade Republic app, if eligible in the user's country/account |
| Day-1 allocation questions | Trade Republic support/app docs, not TradingView |
| Chart levels and alerts | TradingView |
| Direct chart trading | Only through a TradingView-supported broker, if that broker offers `SPCX` |
| Post-listing manual execution | Broker app/API that actually supports the listed stock and the user's jurisdiction |

Trade Republic announced that European customers can subscribe to selected IPOs directly in-app before trading starts, with allocation at the official allocation price, pro-rata allocation if demand is high, and a 1 EUR settlement fee. That is not the same thing as being an integrated TradingView broker.

Checked TradingView sources imply:

- TradingView can route orders only through supported broker integrations.
- TradingView's help center says unsupported brokers need to integrate through TradingView's API process.
- Trade Republic did not appear on the checked TradingView broker directory page during this pass.

Practical setup:

- Use TradingView for watchlists, alerts, and levels: $135 IPO price, about $164 coworker median, about $167 coworker mean, about $173 live `xyz:SPCX` mark, and about $187 coworker P75.
- Use Trade Republic separately for IPO subscription and any allocated-share workflow.
- Before sizing any IPO-allocation strategy, confirm with Trade Republic support: can allocated shares be sold on day 1, are there flipping limits or penalties, when do shares appear in the account, and can orders be placed before Nasdaq open?

## Coworker PCHIP Distribution

The PNG and DOCX use Polymarket's 16-strike "closing market cap above" surface, then fit a monotone cubic PCHIP/Fritsch-Carlson curve. This treats threshold YES prices as a survivor function:

```text
S(K) = P(first-day close market cap > K)
PDF(K) = -dS(K)/dK
EV[V] = integral S(K) dK
```

The coworker share convention is 13.091B shares. Using their convention, the distribution summary is:

| statistic | market cap | share price |
|---|---:|---:|
| Mean | $2.185T | $166.9 |
| Median | $2.149T | $164.2 |
| Mode | $2.112T | $161.3 |
| Standard deviation | $0.533T | $40.7 |
| P5 | $1.481T | $113.2 |
| P10 | $1.629T | $124.4 |
| P25 | $1.836T | $140.2 |
| P75 | $2.453T | $187.3 |
| P90 | $2.820T | $215.4 |
| P95 | $3.079T | $235.2 |

The most useful cross-market diagnostic in the coworker files is the bucket-vs-continuous comparison:

| bucket | continuous-implied | PM bucket market | difference | read |
|---|---:|---:|---:|---|
| less than $1.0T | 1.1% | 0.6% | +0.5pp | small |
| $1.0T-$1.5T | 4.3% | 3.4% | +1.0pp | small |
| $1.5T-$2.0T | 31.0% | 38.2% | -7.2pp | possible bucket overpricing |
| $2.0T-$2.5T | 42.6% | 41.7% | +0.9pp | aligned |
| $2.5T-$3.0T | 14.3% | 11.7% | +2.6pp | modest |
| $3.0T-$3.5T | 4.4% | 3.3% | +1.0pp | small |
| $3.5T+ | 2.0% | 1.0% | +1.0pp | small |

Working implication: if fresh data still shows the `$1.5T-$2.0T` bucket materially overpriced versus the threshold curve, a NO expression on that bucket may have cleaner EV than a generic "SpaceX overvalued" short. That needs a full payoff matrix including fees, bid/ask, capital lock, and independent market liquidity.

## Ambiguity To Reconcile

The DOCX says buying IPO allocation at $135 has about 80% probability of profit and about +$32/share expected gain if the first-day closing distribution mean is near $166.9/share.

The PNG summary panel, however, says:

```text
IPO at 135(~1.77T):
  P(win): 79.9%
  EV: $-3.3/share
```

That EV sign does not match the visible mean-minus-entry arithmetic. It may be using a different payoff convention, a hedged trade-entry convention, an allocation/pro-rata/cost adjustment, or it may simply be a chart bug. Do not import that EV line into production screens until the calculation code is checked.

## Agent Implications

Build the SpaceX agent stack around shared normalization:

```text
official_share_count = 13,075,865,175  # S-1/A no-option Class A + Class B
cap_from_share_price = share_price * official_share_count
share_from_cap = cap / official_share_count
ventuals_cap = displayed_spacex_price * 1,000,000,000
```

Recommended agent modules:

| module | purpose | critical fields |
|---|---|---|
| Official filing agent | Keep share count, IPO price, effect date, listing status, and lockup mechanics current. | S-1/A, EFFECT, 424B, Nasdaq page, share-count convention |
| PM distribution agent | Rebuild threshold CDF/PDF and compare against buckets. | token IDs, bid/ask, spreads, liquidity, CDF monotonicity, bucket gap |
| Hyperliquid agent | Track `xyz:SPCX` and `vntl:SPACEX` separately. | mark, oracle, mid, OI, funding, depth, unit conversion |
| Trade Republic workflow agent | Track IPO allocation mechanics and operational constraints. | eligibility, subscription cutoff, pro-rata, day-1 sell, fee, flipping rules |
| TradingView watch agent | Make levels/alerts visible after `NASDAQ:SPCX` exists. | symbol availability, levels, alert triggers, broker routing availability |

For strategy discussion, separate three questions:

1. Is SpaceX fundamentally overvalued relative to a defensible intrinsic model?
2. Is the day-1 market distribution too optimistic relative to realistic demand/float/lockup mechanics?
3. Which expression has the best EV after settlement rules, leverage, funding, borrow/access, slippage, and operational constraints?

The answer can differ by venue. A bearish fundamental view does not automatically mean shorting `xyz:SPCX` at 5x is the best trade; a PM bucket or threshold NO may be cleaner if the edge is in event settlement rather than intraday price path.

## Verify Before Trading

- Refresh the Polymarket 16-strike threshold data and bucket market on June 11 and June 12.
- Confirm Trade Republic day-1 sell capability, flipping limits, allocation timing, and whether orders can be staged before Nasdaq open.
- Monitor the Nasdaq official listing page for the market-cap/share-count field PM may use for resolution.
- Reconcile the PNG's `-$3.3/share` IPO EV line against the underlying calculation.
- Treat `xyz:SPCX` conversion timing and gap risk as a liquidation problem, not only a valuation problem.
- Treat `vntl:SPACEX` as a separate valuation-settlement product, not a thinner quote for the same share-like market.

## Source Links

- Local coworker DOCX: `/Users/justiniturregui/Downloads/SpaceX_IPO_Research_Handoff_v2.docx`
- Local coworker PNG: `/Users/justiniturregui/Downloads/spacex_pdf_analysis.png`
- [SpaceX IPO release, 2026-06-04](https://content.spacex.com/cms-assets/FINAL_Documents%20and%20Updates/6.4.26_SpaceX_Announces_IPO_US.pdf)
- [SEC S-1/A index, accession 0001628280-26-040364](https://www.sec.gov/Archives/edgar/data/1181412/000162828026040364/0001628280-26-040364-index.htm)
- [SEC S-1/A HTML filing](https://www.sec.gov/Archives/edgar/data/1181412/000162828026040364/spaceexplorationtechnologib.htm)
- [trade[XYZ] Pre-IPO Perpetuals docs](https://docs.trade.xyz/asset-directory/pre-ipo-perpetuals-ipops.md)
- [trade[XYZ] IPOP specification index](https://docs.trade.xyz/consolidated-resources/specification-index-ipops.md)
- [trade[XYZ] IPOP risk disclaimer](https://docs.trade.xyz/legal-and-disclaimers/pre-ipo-perpetual-markets-risks-and-disclaimers.md)
- [trade[XYZ] oracle price docs](https://docs.trade.xyz/perp-mechanics/oracle-price.md)
- [trade[XYZ] funding docs](https://docs.trade.xyz/perp-mechanics/funding.md)
- [Ventuals private-company perp docs](https://docs.ventuals.com/perp-specifications/private-companies.md)
- [Trade Republic IPO access announcement on TradingView News](https://www.tradingview.com/news/eqs%3Ac0a999269094b%3A0-starting-today-trade-republic-gives-european-retail-investors-direct-access-to-ipos/)
- [TradingView broker directory](https://www.tradingview.com/brokers/)
- [TradingView unsupported-broker help](https://www.tradingview.com/support/solutions/43000479602-the-broker-i-want-to-trade-through-is-not-supported-can-you-add-it/)
