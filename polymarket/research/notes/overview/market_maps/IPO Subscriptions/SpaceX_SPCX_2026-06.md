---
title: "IPO Subscription Log — SpaceX (SPCX)"
type: ipo-subscription-log
deal: "SpaceX"
ticker: "SPCX"
status: "planned"
created: "2026-06-09"
tags: [ipo, subscription, log, spacex, spcx]
---

# IPO Subscription Log — SpaceX (SPCX)

> System: [[eu_ipo_broker_subscription_model]] · Hub: [[POLYMARKET_BRAIN]] · SpaceX cluster: [[spacex_ipo_market_map_handoff]] · [[spacex_ipo_coworker_addendum]] · [[spcx_convergence_calc_findings]] · [[spacex_pdf_construction_audit]]
> Recommender: `polymarket/research/scripts/eu_ipo_capital_split.py` · Folder guide: `README.md` (this folder)

## Plain-English Summary

- **Deal:** SpaceX IPO, Class A common stock, ticker **`SPCX`** (Nasdaq Global Select Market / Nasdaq Texas). Roadshow launched **2026-06-04**; first trading day anticipated **2026-06-12**.
- **The retail-subscription question this log tracks:** how much to subscribe at each EU broker, what fraction actually filled, and what it cost — so the next deals (OpenAI, Anthropic) inherit a real per-broker fill-rate track record.
- **Status:** `planned`. Prospectus terms are pre-filled from the vault. Subscription and fill fields are **TBD** until eligibility/availability are confirmed and allocation happens.
- **The one number:** expected IPO price **$135** (⇒ ≈ **$1.765T** at the ~13.076B-share convention). The audited day-1 distribution puts **P(close > $135) ≈ 80%, mean ≈ $168, median ≈ $164** — see [[spacex_pdf_construction_audit]]. Ignore the colleague PNG's `-$3.3/share` EV line (flagged inconsistent in [[spacex_ipo_coworker_addendum]]).

## Prospectus Terms

| field | value | source |
|---|---|---|
| Deal | SpaceX (Space Exploration Technologies Corp.) | [[spacex_ipo_market_map_handoff]] |
| Ticker / listing venue | `SPCX` / Nasdaq Global Select Market + Nasdaq Texas | SpaceX IPO release 2026-06-04 |
| Expected IPO price | **$135** | SpaceX IPO release 2026-06-04 |
| Max / strike price (if a max-price order applies) | **UNKNOWN** (no max-price term documented; confirm at the broker) | — |
| Tranche / shares offered | **555,555,555 Class A** shares (public offering) | SpaceX IPO release 2026-06-04 |
| Retail tranche / free float (press) | Up to **~30% reserved for retail**, but **free float ~5%** ⇒ expect heavy oversubscription / low fill | boerse-online, telepolis (web, 2026-06-09) |
| Total shares outstanding (post-offering, no underwriter option) | **13,075,865,175** (7,380,196,910 Class A + 5,695,668,265 Class B) | S-1/A accession 0001628280-26-040364 |
| Implied market cap at expected price | **≈ $1.765T** ($135 × 13,075,865,175) | [[spacex_ipo_market_map_handoff]] |
| Registration / effectiveness status | **Not yet effective** as of 2026-06-07; latest S-1/A accession `0001628280-26-040364`, filed 2026-06-03 | [SEC S-1/A index](https://www.sec.gov/Archives/edgar/data/1181412/000162828026040364/0001628280-26-040364-index.htm) |
| Subscription deadline (per broker) | **UNKNOWN per broker** — Trade Republic subscriptions close *before trading starts*; exact cutoff UNKNOWN. Refresh June 11 evening / June 12 morning. | [[spacex_ipo_coworker_addendum]] |
| Anticipated first trading day | **2026-06-12** | trade[XYZ] IPOP spec via [[spacex_ipo_market_map_handoff]] |
| Audited day-1 distribution | P(close > $135) ≈ 80%; mean ≈ $168; median ≈ $164 (single-hump lognormal, best-ask) | [[spacex_pdf_construction_audit]] |

## Per-Broker Plan (fill in BEFORE the deadline)

> **CANDIDATE SET (user, 2026-06-09): Revolut + Trade Republic.** DEGIRO and Interactive Brokers are out of scope. Revolut (user facts): offers SPCX, **no fee**, **$500 min** (USD), **no max**, **no day-1 sell limit**. Trade Republic (vault + web): pro-rata by subscription volume, official price, 1 EUR fee, **no documented max or day-1 restriction**. Remaining UNKNOWNs: TR eligibility for Justin's account, TR **min**, exact deadline, TR cash-block-during-book-build; the USD/EUR reconciliation.
> **Reference split** from the recommender (`--capital C --tilt 1.0`): **Revolut 50% · Trade Republic 50%**, with **expected fill E[F] ≈ 5% of your subscription** (~€500 per €10k). Researching fill rates (web, 2026-06-09) found **no per-broker fill difference** (TR's IPO product is days old; Revolut has no public allocation history), so there is **no fill-rate reason to favour either broker** — see [[eu_ipo_broker_subscription_model]] § "The researched prior". Any Revolut lean (e.g. 60/40) is an **operational** choice (no fee, $500 min, confirmed day-1 sell), not a fill edge. The $500 Revolut floor is USD — convert against your budget currency; below ~$500 the tool flags INFEASIBLE.
> **Expect LOW fill:** press puts ~30% of SPCX reserved for retail but only ~5% free float, and a worked example of subscribe €10,000 → ~€500 or nothing. Heavy oversubscription is likely.

| broker | eligible? | offers SPCX subscription? | subscription deadline | planned subscription | requested_shares | min/max | fee | notes |
|---|---|---|---|---:|---:|---|---|---|
| Revolut | YES (user) | YES (user) | UNKNOWN | TBD | TBD | min $500 (USD) / **max none** (user) | none | Day-1 sell allowed (user). Highest-trust broker. |
| Trade Republic | UNKNOWN (EU retail; Justin's account UNKNOWN) | YES — TR press names it an SPCX distribution partner | UNKNOWN (before trading starts) | TBD | TBD | min UNKNOWN / **max none documented** | 1 EUR | Pro-rata by subscription volume (web-confirmed); no documented day-1 limit; ~5% float ⇒ low fill; confirm cash-block. |

## Realised Fill (per broker) — fill in AFTER allocation

> `fill_fraction` = filled_shares / requested_shares. Leave `TBD` until allocated (the tool skips TBD rows).

| broker | requested_shares | filled_shares | fill_fraction | effective_price | notes |
|---|---:|---:|---:|---:|---|
| Revolut | TBD | TBD | TBD | TBD | |
| Trade Republic | TBD | TBD | TBD | TBD | |

## Day-1 and P&L (fill in after listing)

| field | value |
|---|---|
| Day-1 open | TBD |
| Day-1 close | TBD |
| Total filled shares (across brokers) | TBD |
| Blended effective entry price | TBD |
| Realised P&L (if sold) / mark-to-market (if held) | TBD |
| Sold day 1? where? | TBD |

## Notes / linked analysis

- **Convergence trade (separate from the subscription):** a long-IPO / short-perp basis trade is gated and analysed in [[spcx_convergence_calc_findings]] — TRADE-ABLE only **unlevered**; do not lever the Hyperliquid short. That is a different decision from *how much to subscribe*, but the IPO-allocation long leg is shared, so keep the two logs consistent.
- **Day-1 sell:** Revolut has **no day-1 sell limit** (user); Trade Republic has **no documented day-1/flipping restriction** (web search 2026-06-09 + user) — the 30-day flipping bans found online are US-broker (e.g. Robinhood) policies, not TR. Still worth a final check against TR's listing-week terms, which TR said are coming.

## Post-mortem / lessons

- (After allocation) what each broker's realised `fill_fraction` implies; update `default_brokers()` maturity priors if a broker over/under-delivered.
