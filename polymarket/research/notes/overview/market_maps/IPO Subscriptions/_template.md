---
title: "IPO Subscription Log — <DEAL NAME> (<TICKER>)"
type: ipo-subscription-log
deal: "<DEAL NAME>"
ticker: "<TICKER>"
status: "planned | subscribed | allocated | traded | closed"
created: "<YYYY-MM-DD>"
tags: [ipo, subscription, log]
---

# IPO Subscription Log — <DEAL NAME> (<TICKER>)

> System: [[eu_ipo_broker_subscription_model]] · Hub: [[POLYMARKET_BRAIN]] · Folder guide: `README.md` (this folder)
> Recommender: `polymarket/research/scripts/eu_ipo_capital_split.py`
>
> HOW TO USE: copy this file within this folder to `<Deal>_<TICKER>_<YYYY-MM>.md`, fill the
> prospectus terms and your planned subscription BEFORE the deadline, then fill the realised-fill
> table AFTER allocation. Keep the `Realised Fill (per broker)` table's column names exactly as
> below — the track-record tool parses them. Use `TBD` (not 0) for anything unresolved so it is
> skipped, not counted as a zero fill.

## Plain-English Summary

- One-line description of the deal and why you are (or aren't) subscribing.
- Status and the one number that matters (e.g. expected price, your total subscription).

## Prospectus Terms

| field | value |
|---|---|
| Deal | <DEAL NAME> |
| Ticker / listing venue | <TICKER> / <exchange> |
| Expected IPO price | <price or range> |
| Max / strike price (if a max-price order applies) | <price or UNKNOWN> |
| Tranche / shares offered | <shares offered to retail/public> |
| Total shares outstanding (post-offering, share-count convention noted) | <count + convention> |
| Implied market cap at expected price | <cap> |
| Registration / effectiveness status | <effective? S-1/A accession> |
| Subscription deadline (per broker) | see per-broker plan below |
| Anticipated first trading day | <date> |
| Source(s) | <links to filings / vault notes> |

## Per-Broker Plan (fill in BEFORE the deadline)

Subscription amounts come from the recommender. Record the per-broker deadline because cutoffs differ.
Candidate set (user, 2026-06-09): **Revolut + Trade Republic**. Add a row only if a future deal uses another broker.

| broker | eligible? | offers this deal? | subscription deadline | planned subscription | requested_shares | min/max | fee | notes |
|---|---|---|---|---:|---:|---|---|---|
| Revolut | UNKNOWN | UNKNOWN | UNKNOWN | 0 | 0 | min $500 (USD) / max none | none | day-1 sell allowed (user) |
| Trade Republic | UNKNOWN | UNKNOWN | UNKNOWN | 0 | 0 | min UNKNOWN / max none documented | 1 EUR | pro-rata by subscription volume; no documented day-1 limit |

## Realised Fill (per broker) — fill in AFTER allocation

> Column names are load-bearing: the track-record tool keys on `broker` and `fill_fraction`.
> `fill_fraction` = filled_shares / requested_shares (0–1). Leave `TBD` until the broker allocates.
> If you leave `fill_fraction` blank but enter both share columns, the tool computes it.

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

## Post-mortem / lessons

- What the realised fills said about each broker (feeds the rolling track record).
- Any operational surprises (eligibility, FX, day-1 sell blocks, timing).
- Whether the maturity priors in `default_brokers()` should be updated.
