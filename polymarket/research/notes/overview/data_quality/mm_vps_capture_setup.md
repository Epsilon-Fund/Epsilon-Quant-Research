---
title: "MM L2 Capture — VPS → Cloudflare R2 Setup (cloud path, shard layout, format, how to sync)"
created: 2026-06-23
status: active
owner: justin
project: polymarket
para: resource
hubs:
  - strat_market_making
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - data
  - data-quality
  - market-making
  - infrastructure
  - capture
---

# MM L2 Capture — VPS → Cloudflare R2 Setup

> Hubs: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · [[COWORK]]
> Capture semantics (what the feed can/can't prove): [[mm_clob_capture_semantics]] · Where all the data lives: [[polymarket_data_manifest]] · Architecture/roadmap: [[polymarket_l2_ingestion]] · Engine that replays it: [[mm_engine_phase01_buildplan]]

## Plain-English Summary

- **What this is.** The MM research line now has a 24/7 Polymarket L2 order-book capture running on a small rented Linux server (VPS), backing its data up to cloud object storage (Cloudflare R2). This note is the operational map: **where the data lives in the cloud, how the files are laid out, what format they're in, and the exact commands to pull/sync a copy locally.**
- **Why it was written.** The strategy-agnostic MM engine's **replay adapter** reads captured CLOB JSONL, and Alvaro's reconstruction audit reads the same data — both need to know how to fetch it. The local `polymarket/research/data/live_clob/` shards were uploaded to R2 and **pruned locally to free 73 GB**, so any code that "reads `live_clob`" must now pull from R2 first. This note unblocks that.
- **What it covers.** The deployed VPS pipeline (discovery → capture → compress → sync → expire), the R2 bucket layout, the two on-disk formats (raw **envelope JSONL** vs typed **Parquet**), the tiered retention rules, the `rclone` setup including the one gotcha that bit us, and a worked "pull one day and replay it" example.
- **One-line takeaway.** Live capture is **up and backing up to `r2:epsilon-polymarket-data`**; raw envelope JSONL (what the replay adapter consumes) lives under `raw/` (rolling) and the historical ~1-week test fixture under `research-live-clob/` (R2-only); typed Parquet under `parquet/` is the permanent keeper. Pull with `rclone copy` using the pre-configured `r2:` remote.

---

## 1. The pipeline (what runs where)

Three decoupled components on the VPS, each restartable, mirroring the design in [[polymarket_l2_ingestion]]:

1. **Discovery** — polls the Gamma API for active markets in the target universes (politics NegRisk, sports/eSports, culture/other, crypto-as-control) and writes `live_universe.json`.
2. **Capture daemon** — subscribes to the public market WebSocket (`wss://ws-subscriptions-clob.polymarket.com/ws/market`) for all asset IDs in the universe file and writes raw events as **append-only JSONL**, rotated hourly and gzipped, with a `capture_gaps.jsonl` health log. This is the same capture-envelope format produced by `scripts/dali_live_clob_capture.py` (`envelope()`), so everything in [[mm_clob_capture_semantics]] applies unchanged.
3. **Compression + sync + expiry** — hourly compression turns completed `*.jsonl.gz` into typed Parquet; a periodic sync pushes both to R2; a periodic expiry trims the cloud raw archive once Parquet is confirmed.

**Host / paths (deployed).** Hetzner VPS, host `ubuntu-4gb-hel1-1` (~40 GB disk, 3.7 GB RAM), pipeline rooted at `/opt/epsilon/l2_ingestion` on the VPS. The repo copies of the **sync/expiry scripts and systemd units** live at `infrastructure/data/l2_ingestion/` (`sync/sync_cloud.sh`, `sync/expire_r2_raw.sh`, `deploy/expire-raw.{service,timer}`). The capture/compression daemons themselves run on the VPS (the planned Hetzner CX32 in [[polymarket_l2_ingestion]] was provisioned as a smaller 4 GB box — the workload is I/O-light).

> Practical note: the VPS is the colleague's machine; **Justin pays for the R2 bucket** (it sits on Justin's personal Cloudflare account). See § "Account & access" below.

---

## 2. Cloud layout — `r2:epsilon-polymarket-data`

All backups go to one Cloudflare R2 bucket, **`epsilon-polymarket-data`**, on Justin's personal Cloudflare account (account ID `da772cd9e4f67e78c41afa24337805eb`). The `rclone` remote is named **`r2`** (configured on both the VPS and Justin's Mac).

| R2 prefix | what's there | format | retention |
|---|---|---|---|
| `raw/{date}/{universe}_{hour}.jsonl.gz` | gzipped **capture-envelope JSONL** — the raw WS events, one record per market message | envelope JSONL (gzip) — § 3 | **disposable:** an R2 raw shard is deleted as soon as a non-empty Parquet for it is confirmed (no age gate) |
| `parquet/{date}/{universe}/{table}_{shard}.parquet` | typed columnar tables, one file per (table, source shard); `table` ∈ `book` / `trades` / `price_change` / `bba` | Parquet (typed) — § 3 | **kept forever** — this is the permanent keeper |
| `research-live-clob/...` | the irreplaceable **historical ~1-week capture** (877 raw shards, gzipped 79 GB → 6 GB) used as the gappy backtest test fixture | envelope JSONL (gzip) | kept; **R2-only** (local copy was pruned) |
| `metadata/universes/`, `metadata/health/` | daily universe snapshots + capture-health reports | JSON | as written |

**Column meaning** (so the Parquet tables are self-describing): every table carries `timestamp_ms` (PM server timestamp, ms epoch — the lookahead-free ordering key), `received_at` (ISO wall clock), `received_ns` (monotonic ns), `asset_id`, `market`. Then `book` adds `bids`/`asks` (JSON arrays of `[price, size]` levels); `trades` adds `price`/`size`/`side`; `price_change` adds `price`/`side`/`size`; `bba` adds `best_bid`/`best_ask`/`bid_size`/`ask_size`. Full schema: [[polymarket_l2_ingestion]] § "Parquet schema".

---

## 3. The two formats — and which one the engine reads

There are **two** on-disk representations of the same events; do not confuse them:

- **Envelope JSONL** (`raw/` and `research-live-clob/`): one JSON record per WS message, shape `{received_at, received_monotonic_ns, event_type, asset_ids, assets, message}` where `message` is the raw PM payload. The four `event_type`s are `book`, `price_change`, `last_trade_price`, `best_bid_ask`. **This is what the MM engine's replay adapter consumes** (`mm_engine/feeds/replay.py` reads `*.jsonl` and `*.jsonl.gz`, so the gzipped shards work directly), via the shared `envelope_to_events()` parser. Capture gaps come from the sibling `capture_gaps.jsonl`.
- **Typed Parquet** (`parquet/`): the columnar tables above, for fast spread/depth/toxicity scans and the reconstruction audit. The Phase-0 replay adapter does **not** read Parquet yet — a Parquet→`MarketEvent` adapter is future work; for now, replay off the envelope JSONL.

Why both: JSONL is the faithful, replayable record (every field, exact ordering); Parquet is the compact analytical surface (~4× smaller, columnar). The producer is **write-once-per-path** — raw shards are append-only then closed; each Parquet is written exactly once to a unique `{table}_{shard}.parquet` — which is the invariant that makes the size-based prune (§ 5) safe.

---

## 4. Retention — tiered, so nothing fills the disk and nothing is lost

| layer | raw (`*.jsonl.gz`) | parquet (`*.parquet`) |
|---|---|---|
| **VPS local disk** | pruned after `RAW_RETENTION_DAYS=3` (`find -mtime +3` keeps ~4 days) | pruned after `PARQUET_RETENTION_DAYS=7` (keeps ~8 days) |
| **R2 cloud** | deleted once a confirmed non-empty Parquet exists for the shard (`expire_r2_raw.sh`) | **kept forever** |

The decision (Justin + colleague): **Parquet is the keeper, raw is disposable after a verified parse** — we won't be inspecting the JSON by hand once the typed tables exist. The `unknown`-universe captures produce no Parquet (tiny/near-empty), so their raw is never expired (harmless). Local pruning is **per-file verified** (§ 5): a local file is deleted only when R2 already holds the same relative path at the same byte size, so an in-progress current-hour shard or a failed upload is always kept.

---

## 5. `rclone` setup + the gotcha that bit us

The `r2` remote is an S3-compatible Cloudflare remote, configured once on both machines:

```bash
rclone config
# name: r2 | type: s3 | provider: Cloudflare
# access_key_id / secret_access_key: from the R2 dashboard (Object Read & Write token)
# endpoint: https://da772cd9e4f67e78c41afa24337805eb.r2.cloudflarestorage.com
# no_check_bucket: true        # ← REQUIRED, see below
```

**The gotcha — `no_check_bucket = true` is mandatory.** An Object-Read&Write R2 token *cannot* `CreateBucket`, which `rclone` attempts on the first upload to a bucket root → a `403 AccessDenied` on the PUT. Setting `no_check_bucket = true` on the remote skips that probe and uploads succeed. (R2 also intermittently returns `501 NotImplemented` on PUT; `rclone`'s retries recover — not a real failure.) On Justin's Mac there is also a pre-existing `ofdrive:` remote — **don't clobber it**; the `r2:` remote is additive.

**Sync (`sync_cloud.sh`, every 6h on the VPS).** Both raw and parquet upload with **`rclone copy`, never `rclone sync`** — `copy` only ever adds/updates on the remote, while `sync` would mirror local deletions onto R2 and destroy the archive. Pruning is **per-file verified, not gated on a global "did the whole upload succeed" flag** — because a 24/7 capture always has a current-hour shard being appended that `rclone` can't copy ("source file is being updated"), a global gate evaluated false on every run and the disk filled to 99%. The fix deletes a local file only when R2 holds the same relative path at the same byte size (sound because the producer is write-once-per-path). A `df ≥ 90%` backstop exits nonzero so monitoring surfaces a stuck prune.

**Expiry (`expire_r2_raw.sh`, every 6h via `expire-raw.timer`).** Deletes an R2 `raw/` shard only when a **non-empty** Parquet for that shard exists in R2; keeps (and WARN-logs) anything unverified; never touches `parquet/` or `research-live-clob/`. Listing failures abort with nothing deleted (fail-safe).

---

## 6. How to pull / sync data locally

The `r2` remote already exists on Justin's Mac. From anywhere:

```bash
# 1. The historical ~1-week test fixture (what the replay adapter + reconstruction audit use).
#    It is R2-only — the local copy was pruned. Pull what you need into the expected path:
rclone copy r2:epsilon-polymarket-data/research-live-clob/ \
    polymarket/research/data/live_clob/ --progress

# 2. A specific day of the live rolling raw capture (gzipped envelope JSONL):
rclone copy r2:epsilon-polymarket-data/raw/2026-06-23/ \
    polymarket/research/data/live_clob/raw_2026-06-23/ --progress

# 3. The typed Parquet for a day/universe (for spread/depth/toxicity scans, not the replay adapter):
rclone copy r2:epsilon-polymarket-data/parquet/2026-06-23/politics_negrisk/ \
    polymarket/research/data/l2_parquet/2026-06-23/politics_negrisk/ --progress

# Inspect without downloading:
rclone lsf -R r2:epsilon-polymarket-data/raw/ | head
rclone size r2:epsilon-polymarket-data/parquet/
```

The replay adapter is gzip-aware, so a pulled `*.jsonl.gz` shard (or a whole run directory with its `capture_gaps.jsonl`) can be replayed in place — no decompression step needed.

---

## 7. Practical example — pull one capture and replay it through the engine

Concretely, to drive the symmetric quoter over a real captured market:

```bash
cd polymarket/research
# pull a small slice of the historical fixture
rclone copy r2:epsilon-polymarket-data/research-live-clob/<run_dir>/ \
    data/live_clob/<run_dir>/ --progress
```

```python
# then replay it (envelope JSONL -> MarketEvent stream -> quotes), lookahead-free by ts_exchange
from mm_engine import SymmetricQuoter, run_strategy
from mm_engine.feeds import replay_feed

decisions = run_strategy(
    replay_feed("data/live_clob/<run_dir>"),   # honors capture_gaps.jsonl automatically
    SymmetricQuoter(),
    {"half_spread": 0.01, "size": 100.0},
)
```

A `book` event anchors the book and clears staleness; events after a recorded gap are marked `stale=True` (so the quoter emits nothing) until the next full `book` re-anchors — exactly the gap rule the capture pipeline's `capture_gaps.jsonl` was built to support.

---

## 8. Account & access (operational facts to keep straight)

- **Bucket:** `epsilon-polymarket-data` · **account ID:** `da772cd9e4f67e78c41afa24337805eb` · **endpoint:** `https://da772cd9e4f67e78c41afa24337805eb.r2.cloudflarestorage.com`.
- **Who pays:** Justin (personal Cloudflare account). Within the R2 free tier for months; ~$0.66/month even at ~54 GB accumulated (no egress fees).
- **rclone remote name:** `r2` (both VPS and Justin's Mac), with `no_check_bucket = true`.
- **Repo gotcha when committing capture/infra changes:** `nbstripout` is configured as a git filter for `*.ipynb` but isn't installed locally, so commits can fail until bypassed (`-c filter.nbstripout.clean=cat -c filter.nbstripout.smudge=cat -c filter.nbstripout.required=false`).

---

## 9. Decision / next step

**State:** continuous VPS→R2 L2 capture is live and verified; the backup/prune/expire loop is per-file safe and disk-bounded. The MM engine's replay adapter consumes the envelope JSONL (raw + research-live-clob) directly, gzip and gaps included.

**Next:**
1. A **Parquet→`MarketEvent` replay adapter** so the engine can also replay the compact typed tables (not just JSONL) once weeks of data accumulate.
2. The **reconstruction audit** (Alvaro) over the rolling capture — clean-vs-ambiguous interval % per market/category — per [[mm_clob_capture_semantics]] § Required Reconstruction Audit; it reads the same R2 data documented here.
3. Keep this note in sync if the bucket layout, retention, or `rclone` config changes. The raw expiry recently dropped its age gate (delete as soon as Parquet is confirmed) — re-check the deployed `expire_r2_raw.sh` before relying on any age assumption.
