---
title: "Goldsky Incremental Freshness Note"
created: 2026-06-05
status: active
owner: justin
project: polymarket
para: project
hubs:
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - research
  - data-quality
---
# Goldsky Incremental Freshness Note
> Hub: [[COWORK]]


## Summary

- Scope: Goldsky Incremental Freshness Note in the Polymarket data-quality area.
- Existing takeaway/status: Local parquet tail before this freshness work: `2026-04-24 00:00:00 UTC`. Latest public Goldsky indexed `OrderFilled` observed in the freshness probe: `2026-04-28 11:00:40 UTC`. Goldsky incremental shard written: `data/trades/trades_delta_shardinc_20260424T000000Z_20260428T110040Z.parquet`. Goldsky shard validation: 18,842,358 rows, min timestamp `2026-04-24 00:00:02 UTC`, max timestamp `2026-04-27 00:10:40 UTC`,
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
Generated: 2026-05-23
Updated: 2026-05-26

Purpose: keep the local historical `OrderFilled` parquet store fresh up to the
latest Goldsky-indexed on-chain fill without treating Goldsky as a current/live
feed.

## Current Check

- Local parquet tail before this freshness work:
  `2026-04-24 00:00:00 UTC`.
- Latest public Goldsky indexed `OrderFilled` observed in the freshness probe:
  `2026-04-28 11:00:40 UTC`.
- Goldsky incremental shard written:
  `data/trades/trades_delta_shardinc_20260424T000000Z_20260428T110040Z.parquet`.
- Goldsky shard validation: 18,842,358 rows, min timestamp
  `2026-04-24 00:00:02 UTC`, max timestamp `2026-04-27 00:10:40 UTC`,
  null `market_id` rate `0.001165565371`.
- Direct Polygon shards written:
  `data/trades/trades_delta_shardpolygon_*.parquet`.
- Polygon shard validation: 30 shards, 185,150,029 rows, min timestamp
  `2026-04-27 00:10:42 UTC`, max timestamp `2026-05-26 19:57:58 UTC`,
  null `market_id` rate `0.000021431269`.
- Combined delta shard validation: 61 shards, 1,117,438,881 rows, max
  timestamp `2026-05-26 19:57:58 UTC`, null `market_id` rate
  `0.002432273520`.
- `data/trades/_inprog` was clean after the final validation.

## Scripts

`scripts/sync_goldsky_incremental.py`

Default dry-run:

```bash
cd polymarket/research
PYTHONPATH=. uv run python scripts/sync_goldsky_incremental.py
```

Write one append-only shard:

```bash
cd polymarket/research
PYTHONPATH=. uv run python scripts/sync_goldsky_incremental.py --write
```

The shard written by the 2026-05-26 run:

```text
data/trades/trades_delta_shardinc_20260424T000000Z_20260428T110040Z.parquet
```

Direct Polygon catch-up:

```bash
cd polymarket/research
PYTHONPATH=. uv run python scripts/sync_polygon_order_fills.py --write --replace-inprogress --block-chunk 100 --workers 6
```

Metadata repair for direct Polygon shards:

```bash
cd polymarket/research
PYTHONPATH=. uv run python scripts/repair_polygon_market_ids.py --write --replace-inprogress --gamma-batch-size 20
```

Safe smoke check:

```bash
cd polymarket/research
PYTHONPATH=. uv run python scripts/sync_goldsky_incremental.py --max-regular-pages 2
```

## Safety Rules

- The Goldsky and Polygon fetchers write to `data/trades/_inprog/` first, then
  atomically rename the completed parquet into `data/trades/`.
- The Goldsky script never rewrites existing parquet shards.
- The Polygon fetcher writes daily/partial shards and skips existing final
  shard files.
- The Polygon metadata repair is intentionally metadata-only: it does not
  refetch logs or change fill amounts/prices; it fills null
  `market_id`/`condition_id` from Gamma token metadata.
- If an in-progress temp file exists, scripts refuse to proceed unless run with
  `--replace-inprogress`.
- The Goldsky script caps every run at the current latest indexed Goldsky
  timestamp discovered before fetching, so it does not chase a moving live tail.
- The Polygon script caps every run at the latest block minus
  `--finality-blocks` and defaults to 256 finality blocks.
- The Goldsky script fetches exact timestamp boundary rows separately so a page
  split cannot drop fills that share the same `timestamp`.
- Refresh `data/markets/markets_*.parquet` before large writes if many new
  markets were created after the current market snapshot; otherwise unmapped
  token IDs will be written with null `market_id`/`condition_id`.
- Use `POLYGON_RPC_URL` for a private Polygon endpoint. Without it, the direct
  Polygon script falls back to a public RPC and should be run with modest
  concurrency.

## API Lane

Use Goldsky as the append-only historical settlement source. It is appropriate
for backfilling on-chain `OrderFilled` events into parquet shards, but it can
lag real time and should not be the live trading clock.

For the current public Goldsky subgraph used by this repo, the latest observed
indexed fill in the probe was `2026-04-28 11:00:40 UTC`. The safe incremental
Goldsky shard materialized rows through `2026-04-27 00:10:40 UTC`. Do not rely
on this public Goldsky endpoint alone for current research freshness.

To prevent research from staying month-lagged, supplement Goldsky with direct
Polygon log ingestion for the CTF Exchange and Neg Risk CTF Exchange contracts.
That should use the same downstream parquet schema/enrichment path: decode
`OrderFilled`, normalize USDC/token amounts into `usd_amount`, `token_amount`,
`price`, `maker_side`, then enrich `market_id`, `condition_id`, and `neg_risk`
from Gamma token metadata. Include `closed=true` Gamma lookups for resolved
markets; recent sports and short-dated markets often require it. Hosted SQL
providers such as CryptoHouse, Dune, or Allium are useful cross-checks and
exports, but direct logs are the cleanest repeatable backfill source for
complete recent fills.

Use Gamma for discovery and metadata. Official docs describe Gamma as the
primary API for markets/events/tags/series/search, and the active market path is
`events?active=true&closed=false` with pagination. For Dali live discovery, use
Gamma to keep `slug`, `conditionId`, `clobTokenIds`, `active`, `closed`,
`endDate`, `enableOrderBook`, tick/min-size, and NegRisk metadata fresh.

Use CLOB REST/WebSocket for market-native live prices and order book state.
Official docs put public orderbook/price endpoints on `https://clob.polymarket.com`
and the market WebSocket on
`wss://ws-subscriptions-clob.polymarket.com/ws/market`. Subscribe by token ID for
`book`, `price_change`, `last_trade_price`, and, with custom features enabled,
`best_bid_ask`, `new_market`, and `market_resolved`.

Use RTDS for reference feeds and platform context, not as the canonical
historical fill source. Current official RTDS docs list crypto price, Chainlink
crypto price, equity/ETF/commodity/forex price, and comment streams at
`wss://ws-live-data.polymarket.com`. That is useful for crypto/equity
up/down labels and current context. If relying on any undocumented
`activity/trades` stream observed by local probes, treat it as opportunistic and
keep Goldsky/CLOB reconciliation around it.

Optional audit supplement: Data API `/trades` can help inspect recent user or
market trade history, but it is pagination/rate-limit constrained and should
not replace Goldsky for complete parquet history.

Sources:

- https://docs.polymarket.com/resources/blockchain-data
- https://docs.polymarket.com/market-data/overview
- https://docs.polymarket.com/market-data/fetching-markets
- https://docs.polymarket.com/market-data/websocket/market-channel
- https://docs.polymarket.com/market-data/websocket/rtds
- https://docs.polymarket.com/api-reference/rate-limits
