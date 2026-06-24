---
title: "MM Data Ingestion — Architecture & Roadmap"
created: 2026-06-16
status: active
owner: alvaro
project: mm
para: project
hubs:
  - strat_market_making
tags:
  - data
  - infrastructure
  - market-making
---

# MM Data Ingestion — Architecture & Roadmap

> Hub: [[strat_market_making]] · [[POLYMARKET_BRAIN]]

## Why this exists

We're restarting the MM research line from scratch. The thesis (Path B) is:

> **Profitability through structural edge — wide spreads + uninformed taker flow — rather than informational edge (better fair value model).**

Testing this requires broad L2 order book data across many Polymarket categories over weeks/months. We need to measure spread, depth, adverse selection, and toxicity across market types, then identify where spread > adverse selection + costs.

All previous MM research used tiny data windows (~48h), full-priority fill assumptions (only 14% survive a strict model), and no out-of-sample testing. This time we build the data foundation first.

---

## What we're capturing

Polymarket CLOB WebSocket (`wss://ws-subscriptions-clob.polymarket.com/ws/market`) emits 4 event types:

| Event | What it gives us | Why we need it |
|---|---|---|
| `book` | Full L2 snapshot (all price levels, both sides) | Spread, depth, queue estimation |
| `price_change` | Incremental L2 delta (price/side/size) | Book reconstruction between snapshots |
| `last_trade_price` | Trade prints (price, side, size, timestamp) | Adverse selection, toxicity (VPIN), fill modeling |
| `best_bid_ask` | L1 top-of-book updates | Microprice, fast spread tracking |

Each message gets stamped with `received_at` (wall clock) and `received_monotonic_ns` (monotonic) on arrival for latency analysis.

### Market universes to capture

| Universe | Why | Approx. markets |
|---|---|---|
| Politics NegRisk | Best historical structural-maker edge (+2,290 bps). Live-loop anchor. | ~200-400 active legs |
| Sports / eSports | Low adverse selection (232 bps historical), wide spreads, high uninformed flow | ~100-300 active |
| Culture / Other | K5-STRESS gate pass, decent structural maker returns | ~50-100 active |
| Crypto (control) | High adverse selection (1,886 bps). Negative control — we should NOT be profitable here | ~20-50 active |

Total: ~500-800 asset IDs across active markets. The WS supports up to ~500 per connection, so we may need 2 connections.

---

## System architecture

Three independent components, each simple and restartable:

### 1. Market Discovery Daemon

**What it does:** Periodically polls the Polymarket Gamma API to discover active markets in our target universes. Outputs a `live_universe.json` that the capture daemon reads.

```
Every 15 minutes:
  GET https://gamma-api.polymarket.com/markets?active=true&...
  → Filter by category/tag (politics, sports, esports, culture, crypto)
  → Extract condition_id + asset_ids (YES/NO tokens)
  → Write live_universe.json (atomic rename)
```

**Why separate:** Market creation/resolution is slow (hours). Decoupling means capture restarts don't need API calls, and discovery failures don't kill capture.

### 2. Capture Daemon

**What it does:** Subscribes to the WS for all asset IDs in `live_universe.json`. Writes raw events as append-only JSONL with hourly file rotation.

```
Data flow:
  WS messages → stamp received_at + received_monotonic_ns → append to JSONL

File structure:
  data/raw/{date}/{universe}_{hour}.jsonl.gz
  
  Example:
  data/raw/2026-06-17/politics_negrisk_14.jsonl.gz
  data/raw/2026-06-17/sports_14.jsonl.gz
```

On hourly rotation: close current file, gzip it, open new file. Also writes `capture_gaps.jsonl` for health monitoring (any WS disconnects, reconnects, message gaps).

**Hot-reload:** Watches `live_universe.json` for changes. When discovery adds new markets, capture subscribes without restart. When markets resolve, capture unsubscribes.

**Existing code:** `polymarket/research/scripts/mm_stage1_live_control.py` already captures all 4 WS event types in this exact format with hourly rotation, `received_at`/`received_monotonic_ns` timestamps, and `capture_gaps.jsonl`. What it lacks: daemonization, dynamic subscriptions from a universe file, and 24/7 reliability (auto-reconnect, watchdog).

### 3. Compression Pipeline (hourly cron)

**What it does:** Converts completed JSONL.gz files to columnar Parquet, then syncs to cloud.

```
Every hour (cron):
  1. Find JSONL.gz files older than 1 hour (completed, not being written)
  2. Parse → Parquet with typed columns
  3. Validate row counts match
  4. Sync to cloud (rclone → Cloudflare R2)
  5. Optionally delete local JSONL.gz after confirmed cloud upload
```

**Parquet schema** (one table per event type):

```
book.parquet columns:
  timestamp_ms    INT64     — PM server timestamp
  received_at     STRING    — ISO wall clock  
  received_ns     INT64     — monotonic nanoseconds
  asset_id        STRING    — token ID
  market          STRING    — human-readable slug
  bids            STRING    — JSON array of [price, size] levels
  asks            STRING    — JSON array of [price, size] levels

trades.parquet columns:
  timestamp_ms, received_at, received_ns, asset_id, market,
  price FLOAT64, size FLOAT64, side STRING

price_change.parquet columns:
  timestamp_ms, received_at, received_ns, asset_id, market,
  price FLOAT64, side STRING, size FLOAT64

bba.parquet columns:
  timestamp_ms, received_at, received_ns, asset_id, market,
  best_bid FLOAT64, best_ask FLOAT64, bid_size FLOAT64, ask_size FLOAT64
```

**Directory structure on cloud:**

```
r2://epsilon-polymarket-data/
  raw/                          ← JSONL.gz backups (optional, delete after 30d)
  parquet/
    {date}/
      {universe}/
        book.parquet
        trades.parquet
        price_change.parquet
        bba.parquet
  metadata/
    universes/                  ← daily universe snapshots
    health/                     ← daily capture health reports
```

---

## Infrastructure

### VPS: Hetzner CX32

- **Specs:** 4 vCPU, 8 GB RAM, 80 GB disk — way more than needed
- **Cost:** ~€6.50/month
- **Location:** Nuremberg or Helsinki (low latency to Polymarket's WS servers)
- **Why Hetzner:** Cheapest reliable European VPS provider. The workload is I/O-light (~1-5 MB/s raw WS traffic)

**Disk math:** At ~50 MB/hour raw JSONL across all universes → ~1.2 GB/day raw. Parquet compresses ~4x → ~300 MB/day processed. 80 GB disk holds ~60 days raw + processed before cleanup needed. Cloud sync means local is just a buffer.

### Cloud storage: Cloudflare R2

- **Free tier:** 10 GB storage, 1M Class A ops, 10M Class B ops/month
- **After free tier:** $0.015/GB/month (no egress fees — this is the key advantage over S3)
- **Access:** S3-compatible API, works with rclone out of the box

**Cost estimate:** ~9 GB/month parquet data → stays within free tier for months. Even at 6 months accumulated (~54 GB), cost is ~$0.66/month.

**Total monthly cost: ~$7/month** (VPS + essentially free storage).

### Sync tool: rclone

Standard CLI tool for cloud sync. Config is one-time:

```bash
rclone config
# Type: s3
# Provider: Cloudflare
# Access key + secret from R2 dashboard
# Endpoint: https://<account-id>.r2.cloudflarestorage.com
```

Then the cron job runs: `rclone sync /data/parquet/ r2:epsilon-polymarket-data/parquet/`

---

## What exists vs. what needs building

| Component | Status | Work needed |
|---|---|---|
| WS capture logic | ✅ Exists (`mm_stage1_live_control.py`) | Refactor into a daemon class |
| 4 event types + timestamps | ✅ Exists | None |
| Hourly rotation + gzip | ✅ Exists | None |
| Health monitoring (gaps) | ✅ Exists | None |
| Market discovery | ❌ New | ~100 lines: Gamma API poll + filter + write JSON |
| Dynamic subscribe/unsubscribe | ❌ New | ~50 lines: watch universe file, diff, send WS commands |
| Auto-reconnect + watchdog | ❌ New | ~80 lines: exponential backoff, systemd service |
| JSONL → Parquet conversion | ❌ New | ~150 lines: parse + typed columns + validate |
| Cloud sync (rclone cron) | ❌ New | ~20 lines: bash script + crontab entry |
| Post-capture analysis | ✅ Exists (`mm_stage1_analyze_capture.py`) | Adapt to new parquet schema |

**Estimate: ~400-500 lines of new Python + a systemd unit + a cron script.**

---

## Roadmap — implementation session tomorrow

### Phase 1: VPS setup (~30 min)

1. Provision Hetzner CX32
2. SSH setup (key-only auth, firewall: SSH only)
3. Install Python 3.11+, pip, rclone, systemd
4. Clone repo (or just copy the scripts we need)

### Phase 2: Market discovery daemon (~1 hour)

1. Write `mm_discovery_daemon.py` — Gamma API poller
2. Define universe configs in YAML (which categories, filters)
3. Test: run once, check `live_universe.json` output
4. Set up as systemd timer (every 15 min)

### Phase 3: Capture daemon (~2 hours)

1. Refactor `mm_stage1_live_control.py` into `mm_capture_daemon.py`
   - Read universe from JSON instead of hardcoded config
   - Add hot-reload (inotify or poll `live_universe.json` mtime)
   - Add auto-reconnect with exponential backoff
   - Add systemd service with restart-on-failure
2. Test: subscribe to a small universe (~10 markets), verify all 4 event types flowing
3. Deploy as systemd service

### Phase 4: Compression + cloud sync (~1 hour)

1. Write `mm_compress_pipeline.py` — JSONL.gz → Parquet
2. Set up Cloudflare R2 bucket + rclone config
3. Write `mm_sync_cloud.sh` — rclone sync + cleanup
4. Set up as hourly cron jobs
5. Test: process a few hours of captured data, verify parquet + cloud upload

### Phase 5: Monitoring + validation (~30 min)

1. Simple health check script: are files being written? Any gaps? Cloud in sync?
2. Adapt `mm_stage1_analyze_capture.py` to read parquet and produce a quick spread/depth/toxicity snapshot
3. Let it run 24h, check next day

**Total estimate: ~5-6 hours for a working 24/7 pipeline.**

---

## How to use the captured data (later)

Once we have weeks of broad L2 data, the Path B analysis is:

1. **Spread map:** For each market category × time-of-day, what's the typical quoted spread?
2. **Adverse selection measurement:** Using trade prints + subsequent book moves, measure realized adverse selection per category.
3. **Toxicity (VPIN):** Volume-synchronized probability of informed trading per market.
4. **Fill simulation:** Using L2 depth + hftbacktest queue models, simulate realistic fill rates.
5. **The test:** Find markets where `spread − adverse_selection − costs > 0` with statistical confidence.

This is the data that will tell us whether Path B works, and where.
