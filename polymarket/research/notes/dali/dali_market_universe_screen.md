---
title: Dali Market Universe Screen
created: 2026-05-23
status: archived
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
tags:
  - dali
  - market-universe
  - screening
  - research
---

# Dali Market Universe Screen

> Hub: [[COWORK]]


## Summary

This screen chooses an initial Dali market universe for the first trade-the-price proof of concept. It recommends daily crypto up/down for pipeline baseline, SPX/SPY/QQQ style markets for the external-data track, and selected AI/product/geopolitics markets for Polymarket-native flow. The purpose is universe selection and bad-universe avoidance before TFI/OFI testing, not final edge validation.

Generated: 2026-05-23

Purpose: choose a small, repeatable market universe for Dali's first
"trade the price" PoC. This note uses the execution/factor research docs plus
the local historical `OrderFilled` data. The goal is not to find the final
edge yet; it is to avoid bad universes before we test TFI/OFI.

## Recommendation

Start with three tracks:

1. **Daily crypto up/down markets** as the data/pipeline baseline.
   They have the cleanest free external feed, enough repeated history, and many
   nearly identical market instances. Expect competition; use these first to
   prove ingestion, labeling, and paper execution, not because they are
   obviously the highest edge.

2. **SPX/SPY/QQQ up/down and opens up/down markets** as the user-priority
   external-data track.
   These are attractive conceptually, but free live equity data is weaker than
   free crypto data. Paper only until the feed/latency problem is solved.

3. **AI/product and selected geopolitics/policy markets** as the thinner
   Polymarket-native flow track.
   These are better candidates for pure TFI/OFI "trade the price" because the
   external reference price is less mechanically obvious, the markets are still
   active, and the likely edge may live inside Polymarket flow itself.

Do **not** start with broad Twitter/news/LLM ingestion. Use market-native
trade-flow/order-book features first, then add external sources only where the
market family has a named feed.

## What Was Run

Created efficient screen tooling:

- `scripts/dali_market_universe_screen.py`
- `data/analysis/dali_market_fill_stats_basic.parquet`
- `data/analysis/csv_outputs/dali/dali_forward_viable_market_screen.csv`
- `data/analysis/csv_outputs/dali/dali_open_recent_market_screen.csv`
- `data/analysis/csv_outputs/dali/dali_closed_backtest_market_screen.csv`
- `data/analysis/csv_outputs/dali/dali_family_screen_summary.csv`
- `data/analysis/csv_outputs/dali/dali_family_shortlist_examples.csv`
- `data/analysis/dali_forward_candidate_fills.parquet`
- `data/analysis/csv_outputs/dali/dali_forward_candidate_audit.csv`
- `data/analysis/csv_outputs/dali/dali_forward_candidate_competition_audit.csv`
- `data/analysis/csv_outputs/dali/dali_gamma_current_future_candidate_markets.csv`

The script avoids repeated full-table brute force. It builds one narrow
market-level aggregate from raw fills, then all screens run from the small
aggregate plus Gamma metadata. Candidate-level drilldowns materialize only the
shortlisted fills.

Historical scope:

- Local fills: 1,064,500,317 total rows, 2022-11-21 to 2026-04-24.
- Matched market aggregate: 808,856 markets, 1,060,467,502 matched fills,
  $55.76B matched volume.
- Local tail: 2026-04-24 00:00:00.
- Goldsky public subgraph probe latest: 2026-04-28 11:00:40 UTC.

So Goldsky itself is not a month stale; the local delta is stale by about one
month versus today, and Goldsky currently only gets us about four more days.

## Shortlist

### 1. Daily Crypto Up/Down

Use for: first backtest harness, market-second TFI labels, live capture smoke
tests, paper trading.

Historical screen:

- 21,096 eligible closed markets since 2026-01-01.
- 17.56M fills in liquid-baseline bucket.
- 18.44M fills in thin-alive bucket.
- Latest local examples include BTC/ETH/XRP 5-minute and 15-minute up/down
  markets on 2026-04-23.

Why this is good:

- Repeated market template.
- High historical sample count.
- Free public crypto data is genuinely usable: Binance streams provide
  real-time trade/kline/depth updates.
- Clear external label: BTC/ETH/SOL/XRP price over the same window.

Issues:

- Likely the most competed short-horizon universe.
- Many markets resolve within minutes, so operational reliability matters.
- Current Gamma active/open list includes stale-ended crypto markets; live
  discovery needs stricter `endDate >= now`, `closed=false`, and real CLOB
  subscription checks.

Verdict: **include as baseline, not assumed alpha.**

### 2. SPX/SPY/QQQ Up/Down, Opens Up/Down

Use for: external reference price experiments, especially "trade the price"
around open/close uncertainty.

Historical screen:

- 246 eligible closed equity-index markets since 2026-01-01.
- Recent examples:
  - `S&P 500 (SPX) Up or Down on April 23?`
  - `SPY (SPY) Up or Down on April 23?`
  - `QQQ (QQQ) Up or Down on April 23?`
  - `S&P 500 (SPX) Opens Up or Down on April 23?`

Why this is good:

- Clean market question.
- External causal variable is obvious.
- User already likes this idea.
- Open/close framing may create retail lag and confusion.

Issues:

- Free real-time equities data is the weak point. Alpaca free data is IEX-only,
  not SIP. FRED has official S&P close history, but it is daily-close data, not
  a live execution feed. Stooq is useful for historical data, but not a robust
  live trading feed.
- If we use delayed/free data, the strategy can become a latency trap.
- For live trading, prefer paid SIP/futures data eventually; for PoC, paper
  with Alpaca IEX/RTDS and measure whether the signal arrives early enough.

Verdict: **include as a focused research track, paper only at first.**

### 3. AI/Product Markets

Use for: thinner "Polymarket-native price path" tests; external data later via
leaderboards/news/social.

Historical screen:

- 68 eligible open markets and 214+ eligible closed markets across thin/liquid
  buckets.
- Current Gamma future examples:
  - `Will Anthropic have the best AI model at the end of June 2026?`
  - `Will Google/OpenAI/Anthropic/xAI have the best AI model at the end of May 2026?`
  - `OpenAI IPO closing market cap above $1T?`

Why this is good:

- Active but not purely mechanical.
- Natural catalysts: model releases, benchmark/leaderboard changes, product
  announcements.
- Better fit for TFI/OFI first, then richer features later.

Issues:

- External data is messier. Public leaderboards and news sources can change
  format; Twitter/social ingestion adds noise.
- Some prices are already near boundaries, so price-level regime filters matter.

Verdict: **include as one of the best thinner PoC universes.**

### 4. Geopolitics/Policy Catalysts

Use for: Polymarket-native flow test and later news/social overlays.

Historical/current examples:

- Iran regime / US-Iran / Israel-Hezbollah / Strait of Hormuz markets.
- Ukraine/Russia ceasefire and policy markets.
- Election/governor/presidential primary markets.

Why this is good:

- Huge current and historical volume.
- Retail attention and information shocks are plausible.
- Flow may lead price on thinner related markets.

Issues:

- News competition is real.
- Causality can be hard to prove.
- Long-dated politics markets with micro-prices can be liquidity/rebate games
  rather than useful price-path targets.

Verdict: **include only curated near-catalyst markets, not broad politics.**

### 5. Weather Daily

Use for: separate resolution/fundamental strategy, not first price-flow PoC.

Historical screen:

- 6,022 eligible closed daily weather markets in the thin-alive bucket.

Why this is good:

- Repeated template.
- Free external weather data exists.
- Prior research already explored weather/tail behavior.

Issues:

- This is more "predict resolution" than "trade the price."
- Many city/weather APIs are adequate for resolution modeling, but the edge may
  not be short-horizon microstructure.

Verdict: **defer for Dali price-path PoC; keep for a separate weather strategy.**

### 6. Sports Lines

Use for: later, if we get reliable odds/score feeds.

Historical screen:

- Very large closed market count.
- Strong examples in NBA/NHL/MLB/tennis/esports.

Issues:

- Free odds feeds are usually poor or delayed.
- Paid sportsbook/odds APIs may be needed for a serious strategy.
- Highly time-sensitive around games.

Verdict: **defer unless we decide to buy or build reliable sports feeds.**

## Robustness Checks

### Timestamp Granularity

Historical fill timestamps are second-level only. Multiple fills in the same
second are preserved as separate rows, but true within-second ordering is not
available. The validation report recommends collapsing
`(address, transaction_hash, timestamp, market_id, outcome_index)` into a
single atomic effect for position/accounting logic.

For Dali's first TFI test:

- Aggregate to market-second or token-second bars.
- Do not pretend to know within-second sequence.
- Preserve `transaction_hash` and row count so tied buckets remain auditable.

### Signed Flow Caveat

The historical `maker_side` field describes the maker's side, not necessarily
the aggressor's economic direction. Before treating signed flow as alpha, audit
the sign convention against live CLOB `last_trade_price` events and/or known
book state.

Recommended first labels:

- `signed_maker_usd = +usd` when maker side is BUY, `-usd` when maker side is SELL.
- Also test the inverse as a sanity check.
- Prefer live CLOB trade events for true short-horizon execution research once
  capture is running.

### Existing Competition

The 40 forward candidate markets had:

- 80,923 raw fills.
- $3.42M historical volume.
- Two relay/operator addresses accounted for a large share of raw fills.
- After excluding known operator/HFT addresses, most markets retained roughly
  46-66% of fills and 27-68% of USD volume.
- Top non-operator actor concentration is often tolerable, but some markets
  are concentrated. Example: the Mohamed Salah market had a high top-actor
  USD share after operator exclusion.

Interpretation:

- There are already bots/operators in these markets.
- This is not fatal for a price-path strategy, but it means market selection
  should penalize extreme top-actor concentration and pure relay-dominated flow.

Suggested guards:

- Reject markets where post-operator fill retention is too low.
- Reject markets where top non-operator actor > 40% of USD volume.
- Reject markets with too few distinct seconds/trades for the intended horizon.
- Reject markets where current spread/depth makes a one-tick edge impossible.

### Current Market Discovery

Gamma is useful but needs filtering. A current Gamma active/open pull fetched
10,100 unique markets before the API returned a 422 offset limit. After filtering
for future end date, volume >= $10k, and liquidity >= $500:

- Geopolitics/policy: 1,043 candidates.
- Sports: 224.
- AI/product: 156.
- Daily single-stock-like: 8, mostly not true daily stocks.
- Daily equity index: 2, mostly long-dated S&P markets.
- Daily crypto up/down: none future-valid in that snapshot because short-lived
  markets age out quickly and some stale markets remain active/open in Gamma.

Conclusion: static Gamma scans are good for durable markets; daily crypto/equity
up/down needs live discovery near market creation.

## External Data Sources

### Polymarket

- Gamma API: current market discovery. Public, no auth.
  `https://gamma-api.polymarket.com/markets`
- CLOB API: public orderbook/prices, authenticated order operations.
  `https://clob.polymarket.com`
- CLOB WebSocket market channel: live L2/orderbook, price changes, trade events.
  `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- RTDS: Polymarket real-time data socket for comments, crypto prices, and equity
  prices. Useful candidate feed, but must be validated against the market
  resolution source.

Sources:

- https://docs.polymarket.com/api-reference/introduction
- https://docs.polymarket.com/quickstart
- https://docs.polymarket.com/market-data/websocket/overview
- https://docs.polymarket.com/market-data/websocket/rtds
- https://docs.polymarket.com/trading/orderbook

### Crypto

- Binance public WebSocket/REST for BTC/ETH/SOL/XRP trades, klines, and depth.
- Coinbase can be a backup reference.

Source:

- https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md

### Equities / Indexes

- Alpaca free market data can support PoC, but it is IEX-only, not consolidated
  SIP. That can be materially different from the real market.
- FRED has official S&P 500 close history, useful for daily validation, not live
  execution.
- Stooq is useful for free historical data but not a robust live execution feed.
- Alpha Vantage has a free API, but rate limits and latency make it more useful
  for research than live execution.

Sources:

- https://docs.alpaca.markets/us/docs/market-data-faq
- https://fred.stlouisfed.org/series/SP500
- https://stooq.com/db/h/
- https://www.alphavantage.co/documentation/

### Weather

- Open-Meteo / NWS-style public feeds can support resolution/fundamental
  modeling, but this is probably a separate strategy track.

### Sports/Odds

- Free scores are feasible; free real-time odds are usually not robust enough.
  Treat sports as a later track unless we add a serious odds feed.

## Next Research Tasks

1. Build a market-second table for `daily_crypto_up_down` closed markets:
   `market_id`, `second`, `signed_flow`, `gross_flow`, `n_fills`, `last_price`,
   `future_return_30s`, `future_return_2m`, `future_return_5m`.

2. Run a no-ML TFI test:
   does signed flow predict future price direction/magnitude after costs?

3. Start live CLOB capture on:
   - current BTC/ETH up/down markets as they appear,
   - current SPX/SPY/QQQ up/down/open markets as they appear,
   - 10-20 AI/product and geopolitics markets from the current Gamma screen.

4. Add current-market filters:
   `endDate >= now`, `closed=false`, `enableOrderBook=true` if available,
   `clobTokenIds` present, spread <= threshold, liquidity >= threshold.

5. Re-run candidate competition audit after each universe update.

## Bottom Line

The clean first PoC is:

1. Daily crypto up/down for pipeline and statistical baseline.
2. SPX/SPY/QQQ up/down/open for the user's preferred external-data track.
3. AI/product and selected geopolitics markets for thinner Polymarket-native
   TFI/OFI.

The first model should still be boring: TFI/OFI plus price state and cost
filters. Add external data and LightGBM only after the simple flow baseline
beats noise.
