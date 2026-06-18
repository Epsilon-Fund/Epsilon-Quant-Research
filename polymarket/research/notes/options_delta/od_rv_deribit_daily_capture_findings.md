---
title: "OD-RV Daily Settlement Check and 08:00 UTC Aligned Capture"
created: 2026-06-05
status: watching
owner: justin
project: polymarket
para: project
hubs:
  - strat_options_delta
  - COWORK
tags:
  - research
  - options-delta
---
# OD-RV Daily Settlement Check and 08:00 UTC Aligned Capture

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior notes: [[od_rv_deribit_daily_scoping_findings]] · [[od_pricing_model_form_findings]]
> Status: original PM daily-vs-Deribit daily is parked; 08:00 UTC aligned PM 4h/hourly capture is implemented.

## Summary

- Scope: OD-RV Daily Settlement Check and 08:00 UTC Aligned Capture in the OD/options-delta area.
- Existing takeaway/status: Park the original PM daily-vs-Deribit daily branch.** Polymarket daily BTC/ETH resolves at 12:00 ET, which is 16:00 UTC during the checked June 2026 markets, from Binance 1 minute candle close prices. Deribit BTC/ETH option instruments returned by the public API all expire at 08:00 UTC. That leaves an 8-hour tail where the Polymarket daily binary can flip after the Deribit option has already expired.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Phase 0 Verdict

**Park the original PM daily-vs-Deribit daily branch.** Polymarket daily BTC/ETH resolves at 12:00 ET, which is 16:00 UTC during the checked June 2026 markets, from Binance 1 minute candle close prices. Deribit BTC/ETH option instruments returned by the public API all expire at 08:00 UTC. That leaves an 8-hour tail where the Polymarket daily binary can flip after the Deribit option has already expired.

The scoping note was right to flag this as structural, not a data-cleanliness nuisance. A clean 16:00 UTC Deribit synthetic was not found. Using the 08:00 UTC option plus spot/perp/next-expiry overlays does not recreate a cash-or-nothing payoff at 16:00 UTC; it reintroduces an unhedged directional/gamma leg over the exact 8 hours the hedge was supposed to eliminate.

**Do proceed only with the aligned 08:00 UTC alternative.** Polymarket has BTC/ETH 4h and hourly markets that end at the same 08:00 UTC timestamp as the Deribit daily/weekly/monthly option expiry:

| pair | PM market example checked | PM end | PM source | Deribit match | verdict |
| --- | --- | --- | --- | --- | --- |
| Daily BTC/ETH | `bitcoin-up-or-down-on-june-2-2026`, `ethereum-up-or-down-on-june-2-2026` | 16:00 UTC | Binance 1m close at 12:00 ET vs prior 12:00 ET | no 16:00 Deribit option expiry | park |
| 4h BTC/ETH | `btc-updown-4h-1780459200`, `eth-updown-4h-1780459200` | 2026-06-03 08:00 UTC | Chainlink Data Streams BTC/USD, ETH/USD | `BTC-3JUN26-*`, `ETH-3JUN26-*` | aligned timestamp, source mismatch remains |
| Hourly BTC/ETH | `bitcoin-up-or-down-june-3-2026-3am-et`, `ethereum-up-or-down-june-3-2026-3am-et` | 2026-06-03 08:00 UTC | Binance 1H open/close for the 03:00 ET candle | `BTC-3JUN26-*`, `ETH-3JUN26-*` | aligned timestamp, cleaner PM strike source |

Read: the daily product is dead for this RV form, but a narrower 08:00 UTC product exists. The 4h product is closer to prior OD work but uses Chainlink as PM's official source. The hourly product is cleaner for PM strike reconstruction because its official rule uses Binance 1H candles. Both still have Deribit-index versus PM-reference source mismatch, so settlement-reference snapshots remain mandatory.

## Source Checks

Public checks used on 2026-06-02:

- Polymarket daily BTC/ETH Gamma metadata: [BTC daily](https://gamma-api.polymarket.com/markets?slug=bitcoin-up-or-down-on-june-2-2026), [ETH daily](https://gamma-api.polymarket.com/markets?slug=ethereum-up-or-down-on-june-2-2026). The market text specifies Binance BTC/USDT or ETH/USDT 1 minute candle close at 12:00 ET, and Gamma reports `endDate=2026-06-02T16:00:00Z`.
- Deribit active option instruments: [BTC instruments](https://www.deribit.com/api/v2/public/get_instruments?currency=BTC&kind=option&expired=false), [ETH instruments](https://www.deribit.com/api/v2/public/get_instruments?currency=ETH&kind=option&expired=false). The returned active BTC/ETH option expiries all had `expiration_timestamp` at 08:00 UTC, with settlement periods `day`, `week`, or `month`.
- Polymarket aligned 4h examples: [BTC 4h](https://gamma-api.polymarket.com/markets?slug=btc-updown-4h-1780459200), [ETH 4h](https://gamma-api.polymarket.com/markets?slug=eth-updown-4h-1780459200). Gamma reports `endDate=2026-06-03T08:00:00Z` and Chainlink Data Streams as the resolution source.
- Polymarket aligned hourly examples: [BTC hourly](https://gamma-api.polymarket.com/markets?slug=bitcoin-up-or-down-june-3-2026-3am-et), [ETH hourly](https://gamma-api.polymarket.com/markets?slug=ethereum-up-or-down-june-3-2026-3am-et). Gamma reports `endDate=2026-06-03T08:00:00Z` and Binance 1H candles as the resolution source.
- Pyth feed IDs are pulled from the public Pyth price-feed list: [Pyth Price Feed IDs](https://docs.pyth.network/price-feeds/core/price-feeds/price-feed-ids?search=btc).

## Implemented Collector

Script: `polymarket/research/scripts/od_rv_deribit_aligned_capture.py`.

The collector is read-only. It discovers the next selected Deribit 08:00 UTC expiry, finds the matching PM BTC/ETH 4h and hourly markets, and writes JSONL snapshots under `polymarket/research/data/live_clob/od_rv_deribit_aligned/<run_id>/`.

Each snapshot stores:

- **PM executable state:** UP/DOWN token IDs, outcome index, top-of-book bid/ask/size, top-5 book depth from CLOB `/book`, recent taker-only trades from `data-api.polymarket.com/trades`, slug, condition id, resolution source, window start/end, and fee schedule.
- **Deribit executable state:** per PM market slug, the selected aligned expiry's calls and puts over five strikes below/at and five strikes above the PM strike or fallback center. Each instrument stores raw instrument name, top book, mark price, mark IV, bid/ask IV, index price, underlying price, estimated delivery price, volume stats, OI, and Greeks from Deribit `/public/get_order_book`.
- **Settlement diagnostics:** Binance spot ticker, Binance candle-open probe for PM strike, Deribit index price, Pyth Hermes latest BTC/ETH prices, and Chainlink stream URLs. For PM 4h, Chainlink is recorded as the official source but the script currently marks raw Chainlink price parsing as unavailable; the Binance 4h candle open is only a fallback center, not the official PM strike.
- **Cadence metadata:** 30s base cadence, 5s in the final hour or when any PM UP mid is between 40c and 60c.

Launch from `polymarket/research`:

```bash
PYTHONPATH=. uv run python scripts/od_rv_deribit_aligned_capture.py --dry-run
```

```bash
PYTHONPATH=. uv run python scripts/od_rv_deribit_aligned_capture.py --once
```

```bash
PYTHONPATH=. uv run python scripts/od_rv_deribit_aligned_capture.py \
  --assets BTC,ETH \
  --families 4h,hourly
```

If `--duration-hours` is omitted, the full run continues until the selected Deribit expiry plus 10 minutes. Pass `--expiry-date YYYY-MM-DD` to pin a specific Deribit date, and pass a unique `--run-id` when running repeated smoke tests so JSONL rows do not append to a prior run.

## Smoke Check

Smoke command run:

```bash
PYTHONPATH=. uv run python scripts/od_rv_deribit_aligned_capture.py \
  --expiry-date 2026-06-03 \
  --run-id od_rv_deribit_aligned_smoke_20260603 \
  --once \
  --trade-limit 5
```

It discovered four aligned PM markets:

- `btc-updown-4h-1780459200`
- `bitcoin-up-or-down-june-3-2026-3am-et`
- `eth-updown-4h-1780459200`
- `ethereum-up-or-down-june-3-2026-3am-et`

The one-row smoke snapshot wrote two PM token books per market and 20 Deribit option books per PM market slug, i.e. 10 selected strikes times call/put. It also returned Pyth data without an error and selected 5s cadence because PM UP was in the 40c-60c band.

## Validation Design

Count **one asset-day per expiry timestamp as one independent window**, not one row per snapshot and not both 4h and hourly as independent if they share the same asset and 08:00 UTC expiry. BTC 2026-06-03 08:00 and ETH 2026-06-03 08:00 are two independent asset-day windows. The 4h and hourly markets for the same asset-day are nested candidate products and must be reported separately or pre-registered as one primary family before computing a powered CI.

Target sample remains about **60-100 independent asset-days**, roughly **30-50 calendar days** for BTC+ETH at one 08:00 UTC aligned window per day. Inside-window snapshots are basis path observations, not independent PnL observations.

Evaluation must include both directions:

- short PM / long Deribit call-spread digital when PM is rich
- long PM / short Deribit call-spread digital when PM is cheap

For each direction, compute a bid/ask-aware cost floor using PM executable bid/ask, Deribit option bid/ask, fees, call-spread width/curvature error, and leg latency. Then compute settlement-mismatch PnL using Binance, Chainlink, Pyth, and Deribit index snapshots. Any apparent edge that disappears after source mismatch is not OD-RV alpha; it is reference-basis tail risk.

## Decision

The original daily capture requested in [[od_rv_deribit_daily_scoping_findings]] should be **parked**, because PM daily 16:00 UTC settlement is not aligned to Deribit 08:00 UTC expiry and no clean 16:00 UTC option synthetic was found.

The branch can continue only as **OD-RV 08:00 aligned BTC/ETH**, using PM 4h/hourly markets that end at Deribit expiry. Treat this as fresh live data-gathering, not as a reopening of the closed crypto-4h absolute-pricing model. The next useful action is to run the collector across 30-50 calendar days, then evaluate net-of-cost and settlement-mismatch PnL at the asset-day level.
