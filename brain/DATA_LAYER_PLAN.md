# Epsilon — Polymarket data layer build plan

Version 1 · 2026-08-25 · written for Alvaro, Justin and Gonzalo

The goal, in one sentence: **turn 71 GiB of hourly capture shards into something where "show me this market" takes seconds, and where a new person is productive on day one.**

---

## The shape of it

Three layers, built in order. Each is useful on its own.

| Layer | What it is | Lives | Built by |
|---|---|---|---|
| **0 · Archive** | What exists today. Raw JSONL + hourly Parquet, immutable, append-only. Never queried directly for research. | R2 `raw/`, `parquet/` | Already done |
| **1 · Research dataset** | The same data reshaped for reading: partitioned, sorted, book exploded into numbers, plus a catalog. | Built locally → published to R2 `research/v1/` → mirrored locally by each person | Phase B |
| **2 · Tools** | The loader (`get_data`) and the viewer (plots). | The repo | Phases C, D |

**Storage decision, already made:** R2 stays the home. It is the distribution channel, not the query engine — queries always run against a local copy, because a backtest streaming millions of events is throughput work and nothing beats a local SSD at that. Adding ~20 GB of derived data takes the bill from about $0.92 to about $1.30 a month. No new vendor, no new service, nothing to administer. Revisit only if the archive outgrows a laptop, at which point MotherDuck or ClickHouse becomes the right conversation.

**No database server.** DuckDB is a library, not a service — it reads Parquet off disk from inside your Python process. Nothing to install, run or migrate.

---

## Phase A — Know what you have

**Goal:** answer "what data do we actually have, and how much of it is trustworthy" with numbers, not impressions. Nothing is built yet; this is the survey that shapes everything after it.

Not blocked by the pipeline recovery — it runs against the R2 archive as it stands.

### A1 · Coverage manifest (30 minutes, no download)

```
rclone lsd r2:epsilon-polymarket-data                > prefixes.txt
rclone lsl r2:epsilon-polymarket-data/parquet        > parquet_manifest.txt
rclone lsl r2:epsilon-polymarket-data/raw            > raw_manifest.txt
rclone lsl r2:epsilon-polymarket-data/research-live-clob > fixture_manifest.txt
```

From the listings alone, produce: a day-by-day coverage calendar per universe; every missing hour; bytes per day and how it trends; the balance between politics and esports; and what the older 79 GB `research-live-clob` fixture actually contains.

**Done when:** you can point at a calendar and say which days are complete, which are partial, and which are missing.

### A2 · Look at real rows (1 hour)

Pull a single hour of one universe. Verify that the four tables match what `mm_engine/feeds/replay_parquet.py` expects, column by column, and confirm the known schema drift (`bba` ships `spread`, not sizes). Then plot it, roughly, just to see what a market looks like.

**Done when:** the schema is confirmed against real data rather than against the code, and you have seen an order book move.

### A3 · The catalog (half a day)

The most valuable single artefact in this plan. One row per **(asset_id, date)**, covering:

- **Identity** — `asset_id`, `condition_id`, `market` slug, `universe`, `neg_risk`, `end_date`
- **Coverage** — `first_ts`, `last_ts`, hours present, hours missing, gap seconds (sidecar and inferred, kept separate)
- **Activity** — event counts by type, trade count, traded volume, distinct minutes with activity
- **Microstructure** — median and p25/p75 spread in cents, median depth at touch, median depth to 5 levels, median mid, price range
- **Quality** — the `mm_eval/capture_gate.py` verdict (PASS / MARGINAL / FAIL / UNDERPOWERED), clean-checkpoint fraction, stale fraction, trade-in-spread fraction
- **Lifecycle** — `first_seen`, resolution date if known, and a `full_lifecycle` flag: was this market captured from before it became liquid through to resolution, or did we join partway?

Plus a companion `markets.parquet` — one row per `asset_id` with the static metadata.

**Why the lifecycle flag matters:** discovery only subscribes a market once it clears its volume floor, so some markets are missing their early life. "Full lifecycle" has to be measured, not assumed.

**The number that matters most:** how many event groups (politics events, esports matches) have their complete lifecycle inside the capture window. Every confidence interval in the existing research was strangled by having only 11 politics groups. This number tells you whether re-running the ladder on 67 days can certify what 18 days couldn't.

**Done when:** `catalog.parquet` exists, and a one-page summary answers: how many usable market-days, how many complete lifecycles, where the holes are, and what the dataset can and cannot support.

---

## Phase B — Build the research dataset

**Goal:** the same data, reshaped so reading it is fast. This is where minutes become seconds.

### Why the current layout hurts

One Parquet file per hour per table per universe means ~12,000 files; reading a month of politics trades touches roughly 1,600 of them. Worse, the `book` table stores bids and asks as **JSON strings**, so getting depth means parsing JSON on the largest table you own.

### B1 · The layout

```
research/v1/
  _manifest.json                  # version, build date, source range, row counts, checksums
  catalog.parquet                 # one row per (asset_id, date)
  markets.parquet                 # one row per asset_id
  capture_health.parquet          # one row per (universe, date, hour)
  book/     universe=…/date=…/part-*.parquet
  quotes/   universe=…/date=…/part-*.parquet      # from bba
  trades/   universe=…/date=…/part-*.parquet
  changes/  universe=…/date=…/part-*.parquet      # from price_change
```

Rules:

- **Hive partitioning** by `universe` and `date` — so a query for one day touches one directory.
- **Sorted by `(asset_id, timestamp_ms)`** within every file, so Parquet's footer statistics let a reader skip whole files when you ask for one market.
- **Book exploded into numbers**: `bid_px_1..10`, `bid_sz_1..10`, `ask_px_1..10`, `ask_sz_1..10`. No JSON at read time.
- **Convenience columns** alongside: `best_bid`, `best_ask`, `mid`, `microprice`, `spread_c`, `imbalance`.
- **zstd compression**, target 64–128 MB per file.

**One hard rule about the convenience columns:** they exist for viewing and screening only. The engine always recomputes from the raw levels. If a plot and a backtest ever disagree, we need to know it's not because two different pieces of code computed the mid.

### B2 · The builder

A script that reads Layer 0 and writes Layer 1. Idempotent, resumable, one day at a time so it never needs more memory than a day. Run rarely, by one person.

### B3 · Validation

- Row counts per table per day match the source exactly.
- One market spot-checked end to end against the raw JSONL.
- `capture_gate.py` run over the new layout returns the same verdicts as over the old.
- Convenience columns match a fresh recomputation from levels.

### B4 · Publish

`rclone copy` to `r2:epsilon-polymarket-data/research/v1/`, with the manifest. **Built once, by one person; everyone else downloads the finished product.** When the shape changes, cut `v2` rather than mutating `v1` — so nobody's analysis silently changes underneath them.

**Done when:** `research/v1/` is on R2, validated, and roughly a third the size and a fraction of the file count of the source.

---

## Phase C — The loader

**Goal:** `get_data()`. One import, three functions, works identically on three machines.

### C1 · The module

```python
catalog(universe=None, gate="PASS", min_days=None, full_lifecycle=None) -> DataFrame
load(table, asset_id=None, universe=None, start=None, end=None)         -> DataFrame
events(asset_id, day)                                                    -> Iterator[MarketEvent]
```

- **Config**: a local data directory, the R2 remote, the dataset version. Environment variable with a sensible default, so nobody hardcodes a path.
- **Lazy local cache**: when you ask for a slice it doesn't have, the loader downloads that slice, caches it, and reads from disk. Every subsequent call is local-disk speed. It *feels* like an API; it's a cache. A new colleague clones the repo and queries immediately rather than waiting on a 20 GB download.
- **`sync()`** for bulk pre-download when you know you want everything.
- **Direct-from-R2 mode** as an explicit fallback (DuckDB over S3), never the default.

### C2 · The anti-drift rule

`events()` is **the only** function in the codebase that converts stored files into a `MarketEvent` stream. The engine uses it. Research uses it. Everything else — every screen, plot and summary — is analysis layered on top.

**Test that enforces it:** for one shared day, `events()` must produce a stream byte-identical to today's `replay_parquet`. If that test ever fails, research and backtest have diverged, which is the most expensive class of bug in this field.

**Done when:** all three functions work against both local and R2, the parity test passes, and a first-time user can go from `git clone` to a dataframe of real order-book data without asking anyone a question.

---

## Phase D — The viewer

**Goal:** look at a market in seconds. Build a feel for how these things move, and see immediately where the data is not to be trusted.

### D1 · Plotting functions (pure: data in, figure out)

No UI, so they work in a notebook, a script, or any app built later. The core panel, stacked on one shared time axis:

1. **Price with the spread as a band** — best bid and best ask, so the band's width *is* the spread and you watch it breathe.
2. **Trade prints** on the same axis — marks sized by volume, coloured by aggressor side. Book underneath, tape on top: this pairing is the entire physical basis of every simulated fill, so being able to look at it is how judgement gets built.
3. **Spread** in cents.
4. **Depth at the touch**, and to five levels — the queue reality. Fifty-one thousand shares ahead of you should *look* like something.
5. **Volume** per bucket.
6. **Book imbalance** — one of only two signals the research CI-certified as predicting a toxic fill.
7. **Post-trade markout** — how the mid moved 30 s after each trade. Adverse selection, made visible.
8. **Capture-health strip** along the bottom — gaps, stale book, reconnects. So you are never reading a chart without knowing whether to believe it.

### D2 · Multi-market views

- **Small multiples** — the same panel for several markets at once, for scanning.
- **The screen scatter** — markets plotted on spread against depth, or spread against adverse selection. This is the market screen rendered as a picture: which of these is even quotable becomes obvious rather than a table lookup.

### D3 · The overlay hook — build this now, populate it later

Every plotting function takes **optional** strategy data: quote ladder, fills, inventory, PnL, gate state as background shading. Yours renders without it; Gonzalo's renders with it.

Get this right now and his job is supplying a data source. Get it wrong and he writes a parallel plotting stack, the two drift, and you spend your time debugging charts instead of strategies.

### D4 · The app

A thin local app over the functions — pick a market, pick a window, see the panel. Roughly a hundred lines. The functions are the durable thing; the app is what makes "check this token, this token, this token" take seconds.

**Done when:** you can open any market from the catalog and see all eight panels in under ten seconds, and the same call with strategy data attached renders the overlay.

---

## Phase E — Gonzalo

**Goal:** productive in a day, then building the layer above.

### E1 · Getting started

A short README: clone, set one environment variable, `catalog()`, first plot. Under an hour, no help needed. Alongside it, the three orientation pages already written — the field map, the model anatomy, the data atlas.

### E2 · His first piece of work

The strategy-debugging views, on top of D3's overlay hook: where did we quote, where did we get filled, what did inventory do, when did the gate fire, and — the question that actually matters — **when are we losing on this quoting, and does it coincide with price running?** That is a chart, and once it exists it will teach you more about the strategy than another confidence interval will.

---

## Sequence and rough effort

| Phase | Effort | Depends on |
|---|---|---|
| A1 manifest | 30 min | nothing |
| A2 real rows | 1 h | A1 |
| A3 catalog | half a day | A2 |
| B dataset + publish | 1–2 days | A3 |
| C loader | half a day | B |
| D viewer | 1–2 days | C |
| E onboarding | half a day | D |

About a week of Claude Code-assisted work. **A1 and A2 can start immediately** — they need no build and no recovered pipeline.

---

## Decisions already made

- R2 stays the storage. Distribution, not query.
- No database server. Parquet on disk plus DuckDB.
- The derived dataset is versioned and immutable: `v1`, then `v2`, never a mutated `v1`.
- Layer 0 raw is never touched, so any derived layer can be rebuilt from scratch.
- One conversion from files to `MarketEvent`, shared by research and engine.
- Convenience columns are for viewing only; the engine recomputes.

## Still open

- **Whether to keep capturing.** Decide after A3 tells you how many complete lifecycles the window contains. Cost is a few euros a month, and the stream is irreplaceable — so do not switch it off before you know.
- **Whether to re-enable the crypto control universe.** Eleven commented-out lines in `universes.yaml`. It is the falsification instrument and it is currently off.
- **Where the loader module lives** in the repo, and matching the naming of the existing crypto `get_data`.
- **The research-tooling API** beyond the viewer — screens, sweeps, markout tooling. Its own design conversation, better held once we have both looked at real markets.
