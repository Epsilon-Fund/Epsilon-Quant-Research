# L2 Ingestion Pipeline

Captures broad Polymarket L2 order book data (book, price_change, last_trade_price, best_bid_ask) across many market categories for the MM Path B research line. See [[polymarket_l2_ingestion]] for the full architecture spec.

## Three components

1. **discovery/** — Polls the Gamma API every 15 min, filters to our target universes (politics, sports, culture, crypto control), and writes `live_universe.json` (the list of asset IDs to capture).
2. **capture/** — A 24/7 daemon that subscribes to the CLOB WebSocket for every asset in `live_universe.json`, stamps each message with arrival timestamps, and appends raw events to hourly-rotated `*.jsonl.gz` files.
3. **compression/** — An hourly job that converts completed JSONL.gz files to typed Parquet, validates row counts, and hands off to `sync/` (rclone → Cloudflare R2).

## How they communicate — files on disk, not imports

The three components never call each other directly. They coordinate through files:
- discovery **writes** `live_universe.json`; capture **reads** it (hot-reload, no restart).
- capture **writes** `data/raw/{date}/{universe}_{hour}.jsonl.gz`; compression **reads** completed ones.
- compression **writes** Parquet; `sync/` uploads it to the cloud.

This decoupling means any component can crash or restart without taking the others down. `deploy/` holds the systemd units/timers that keep each one alive. `monitoring/` checks the whole chain is healthy.

## Operational docs

- [[DEPLOY]] — step-by-step deployment of the pipeline on the Hetzner VPS.
- [[R2_HANDOVER]] — Cloudflare R2 cloud-backup setup + handover notes.
