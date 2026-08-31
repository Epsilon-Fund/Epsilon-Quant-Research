# epsilon_data — the Polymarket research library (v1)

The loader for the reshaped Polymarket order-book archive (`research/v1`). Everything a
dashboard, notebook, or analysis should touch goes through these functions — **never raw
parquet paths.** If a panel needs something not here, extend this library and document it.

```python
import epsilon_data as ed
ed.search("fed")                          # find a market among 15,386
aid = ed.resolve("politics/fed-decision-in-july-181/…/yes")
l1  = ed.load_l1(aid)                      # its L1 tape (UTC-indexed)
```

## Quickstart (from `git clone`)

```bash
cd polymarket/research
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # (uv users: `uv sync`)
python scripts/fetch_data.py             # download the data (~1.09 GB, no rclone; needs 3 R2 env vars)
#  point at the data (choose ONE):
#    local (fast, what fetch_data.py just wrote):  set EPSILON_DATA_ROOT=…\polymarket\research\data\research_v1
#    straight from R2 (nothing to download, slower first touch):  see "Getting the data" below
python scripts/check_setup.py            # verifies env + data; tells you what to fix in a sentence
.venv/Scripts/streamlit run dashboard/app.py
```

Both data paths work with **only `EPSILON_DATA_ROOT` changed** — no code edit. See
"Getting the data" and "Troubleshooting" below, and `CONTRIBUTING.md` to add a panel.

## The tree (four levels)

```
universe          politics_negrisk | esports
└── event         "Fed Decision in July?"          event_id, event_slug, event_end_date
    └── market    "Will the Fed cut 25bps?"        condition_id, market_slug, question
        └── token YES | NO  (or team A | team B)   asset_id  ← the order book lives here
```

The order book exists only at **token** level. The two tokens of a market are complementary:
in a healthy binary market `mid(A) + mid(B) ≈ 1`.

## Where the data lives

Local `research/v1/` (default `polymarket/research/data/research_v1/`), overridable with the
`EPSILON_DATA_ROOT` environment variable — so the same code reads an R2 mirror later with no
edits. Files:

| file | what |
|---|---|
| `tokens.parquet` | the tree — one row per token (30,772) |
| `l1/universe=…/month=…/*.parquet` | L1 tape, 101,051,502 rows, deduped to touch-moving rows |
| `trades/universe=…/month=…/*.parquet` | trade prints, 7,227,528 rows |
| `obs_stats.parquet` | per-asset observation stats (feeds `tokens`) |
| `exclusions.csv` | operator-edited exclusions, applied at load; **currently empty** |
| `pc_file_manifest.txt` | archive file list (drives `coverage()`) |

## Public functions

| function | returns |
|---|---|
| `catalog(universe, event, resolved, min_trades, min_days, apply_exclusions=True)` | one row per token: identity + observation + check columns. How you find markets. |
| `events(universe, apply_exclusions=True)` | event rollup: n_markets, n_tokens, date span, total trades, neg_risk |
| `search(text, limit=50)` | free-text over event_title, question, both slugs |
| `resolve(ref)` | `asset_id \| path \| market_slug` → `asset_id` (so nobody types a 77-digit number) |
| `load_l1(ref, start, end)` | one token's L1 tape |
| `load_trades(ref, start, end)` | one token's trade prints |
| `load_pair(condition_id, start, end)` | both sides, mids time-aligned (the YES/NO mirror) |
| `load_event(event_slug, start, end)` | every token in an event, mids aligned (NegRisk sum-to-1) |
| `coverage(universe)` | per (universe,date): active / quiet / gap hours (the calendar) |
| `reconciliation()` | the identities that must hold (catalog sums vs table row counts) |
| `activity_by_time(universe)` | trade counts & volume by UTC hour-of-day and weekday |
| `markout(ref, horizons=(10,30,60))` | per-trade adverse-selection markout from the maker's view (see units) |
| `negrisk_sum(event_slug)` | instantaneous YES-sum for a NegRisk event (common timestamp, never a sum of medians) |
| `audit_market(ref)` | run every cheap sanity check; returns a verdict + evidence + recommendation (never writes) |
| `write_exclusion(asset_id, scope, reason, who)` | append to exclusions.csv — **explicit action only**, never called by audit |

- **Exclusions apply by default.** `apply_exclusions=False` is deliberate — use it only to
  *show* excluded/flagged tokens (marked), never to silently analyse them back in. Nothing is
  excluded automatically; `exclusions.csv` is an operator instrument.
- **`start`/`end`** accept a UTC datetime (or anything `pandas.Timestamp` parses) or epoch-ms.
- **Load per token.** `l1` is 101 M rows; `load_l1`/`load_trades` read one token's partitions
  and let parquet stats skip the rest. Do not read the whole table.

## Tables & columns — type, unit, meaning

### `tokens.parquet` (via `catalog` / `search`)
Identity: `universe` (str), `event_id`/`event_title`/`event_slug` (str), `event_end_date` (str
ISO), `neg_risk` (bool), `condition_id` (str, = the on-chain 0x hash), `question` (str),
`market_slug` (str), `outcome_label` (str, **verbatim** — "Yes"/"No" or a team name),
`outcome_index` (int 0/1), `outcome` (str `YES`/`NO` **or null**, see traps), `resolved_outcome`
(str, winning label or null), `closed` (bool), `closed_time` (str ISO or null), `asset_id`
(str, the 77-digit token id), `complement_asset_id` (str, the other side), `identity_status`
(str), `path` (str, unique — `universe/event_slug/market_slug/outcome_label`).

Observation: `first_seen`/`last_seen` (int epoch-ms), `n_days` (int), `n_l1_events` (int),
`n_trades` (int), **`median_mid` (float, DOLLARS 0–1)**, **`last_mid` (float, DOLLARS)**,
**`median_spread` (float, DOLLARS)**, **`median_spread_cents` (float, CENTS)**,
`hours_from_last_seen_to_close` (float hours or null).

Verification: `check1_roundtrip` (bool), `check2_pairing` (bool), `check3_negrisk`
(bool/null), `check4_status` (str categorical), `check5_indep` (bool/null). Plus `excluded`
(bool) added by `catalog`.

### `l1` (via `load_l1` / `load_pair` / `load_event`)
`ts` (datetime UTC, added by the loader), `timestamp_ms` (int, exchange ms), `received_ns`
(int, local monotonic capture clock — the tiebreak), `asset_id` (str), `best_bid`/`best_ask`/
`mid` (float, **DOLLARS** 0–1), `spread_c` (float, **CENTS**). Sorted `(timestamp_ms,
received_ns)`.

### `trades` (via `load_trades`)
`ts` (datetime UTC), `timestamp_ms`, `received_ns`, `asset_id` (str), `price` (float, DOLLARS),
`size` (float, contracts), `side` (str, BUY/SELL), `fee_rate_bps` (float),
`transaction_hash` (str).

## Traps — read these

- **ids are strings, always.** A 77-digit token id becomes a float silently and every join
  then fails invisibly. The loader keeps them strings; you must too.
- **`outcome` is null for esports.** Those markets are *team A vs team B* (and some Yes/No
  props); `outcome_label` holds the verbatim string. Only ~4,836 tokens (all politics + some
  esports props) carry `YES`/`NO`.
- **`check4_status` is categorical, and `near_half` is the biggest bucket and is NOT a
  failure.** Values: `converged_correct` (6,650), `near_half` (21,522 across both universes —
  esports 21,082 + politics 440; settled but the book never traded to the extreme),
  `no_convergence` (328), `not_resolved` (2,148),
  `no_price` (40), `inverted` (14 — genuine upsets: the underdog won; the mapping is still
  correct, confirmed by checks 1/2). `identity_status='unresolved'` tokens have it null.
- **`l1` is deduped to touch-moving rows — it is NOT every message.** A row exists only where
  `best_bid`/`best_ask` changed. Between rows the touch is unchanged (carry the last value
  forward). Raw price_change was 3.83 B rows; l1 keeps the 2.64% that move the touch.
- **units (post-H0):** mids and `median_spread` are DOLLARS; `spread_c`/`median_spread_cents`
  are CENTS. `mid ± spread/2` is only valid with the dollar spread.
- **`identity_status`:** `resolved` (Gamma named it), `unresolved` (70 tokens — 35 conditions
  Gamma couldn't resolve, kept with identity null and a synthetic `path`, never dropped).

## Known gaps (from `coverage()` and the capture log)

- **The one real outage: 2026-06-22 15:00 → 06-23 08:00 UTC, ~17 h, both universes** (an OOM
  crash). No book, no trades — a true gap, distinct from a quiet market.
- **2026-06-19 h00-11** both universes: capture started midday — not a loss.
- **2026-08-21 h21-23** both universes: reboot tail (deliberately skipped compression).
- **esports quiet hours:** ~24 hours across 14 esports days have a book snapshot but no
  price_change — a *quiet market*, not a gap (e.g. 2026-07-24 h12, 2026-08-21 h11-12).
  `coverage()` separates `quiet_hours` (book, no pc) from `missing_hours` (no book = true gap).

## Reconciliation identities (must always hold)

`sum(catalog.n_l1_events) == l1 rows == 101,051,502` and
`sum(catalog.n_trades) == trades rows == 7,227,528`. `reconciliation()` checks them live; the
dashboard shows them. If either breaks, something downstream has drifted.

## Using it

From `polymarket/research/` with the package importable (`PYTHONPATH=.` or an installed venv):

```python
import epsilon_data as ed
```

The anti-drift test (`tests/test_loader.py::test_anti_drift_l1`) proves the loader returns
exactly what a raw parquet read returns. Run: `PYTHONPATH=. python -m pytest tests/ -q`.

See `notebooks/epsilon_data_examples.ipynb` for five worked tasks end to end.

## Getting the data — two ways, `EPSILON_DATA_ROOT` picks

Both work with **no code change** — only the env var differs.

1. **Fetch it locally** (recommended — fast browsing, best for real work). One command, no rclone,
   nothing to install beyond `requirements.txt`:
   ```
   python scripts/fetch_data.py
   set EPSILON_DATA_ROOT=<repo>/polymarket/research/data/research_v1
   ```
   `fetch_data.py` is a pure-Python (boto3) downloader: it reads the **same three credentials the
   loader uses** (below), byte-copies all 792 files (~1.09 GB) in parallel, is **resumable** (re-run
   to finish a partial fetch — it skips files already present at the right size), and is **read-only
   against R2** (it only lists and gets — no `put`/`delete`/`copy`). After the copy it verifies file
   count, total bytes, `tokens.parquet` row count, and that L1+trades are non-empty, then writes
   `_fetch_receipt.json`. Add `--dest <dir>` to fetch elsewhere, `--dry-run` to list without
   transferring. It fetches into the loader's default dir, so a bare `python scripts/fetch_data.py`
   followed by the `set` above is the whole setup. A full fetch takes ~1–2 min on a normal
   connection; the byte counter can appear to sit still while several large L1 files finish in
   parallel — the *files* counter keeps climbing, so it's working, not hung.

2. **Read straight from R2** (nothing to download; slower first touch — good for a quick look):
   ```
   set EPSILON_DATA_ROOT=s3://epsilon-polymarket-data/research/v1
   ```
   DuckDB httpfs with predicate pushdown fetches only the bytes a single-token read needs.

*(Alternative to (1), if you already have rclone configured:
`rclone copy r2:epsilon-polymarket-data/research/v1 <yourdir>/research_v1 -P` — **`copy` only**, see
the warning below. `fetch_data.py` needs no rclone and is the supported path.)*

### Credentials

Both the loader (R2 path) and `fetch_data.py` read the **same** R2 credentials — from environment
variables `EPSILON_R2_KEY_ID`, `EPSILON_R2_SECRET`, `EPSILON_R2_ENDPOINT`, falling back to your
local `rclone.conf` `[r2]` remote if those aren't set. So a newcomer needs only those three env
vars — no rclone install required. Ask the operator for them and put them in a `.env` (gitignored)
or the rclone config file — **never in the repo, a committed config, or a command string.** The
key is read/write today, so treat it accordingly.

> ### ⚠️ The R2 key can write and delete. Only ever *copy*.
> **Never `sync`, `delete`, `purge` or `move` with `r2:` as the target** (and `fetch_data.py` never
> does — it's read-only by construction). The 71 GB raw archive is not backed up anywhere else — a
> mistyped `sync` destroys it permanently. (A read-only token scoped to the research prefix is the
> eventual fix; ask the operator.)

## Troubleshooting

| symptom | fix |
|---|---|
| `EPSILON_DATA_ROOT is not set` | set it (see Quickstart) — local dir or `s3://…`. |
| `no tokens.parquet under <dir>` | fetch the data: `python scripts/fetch_data.py` (re-run to resume a partial fetch). |
| `no R2 credentials found` (s3 root) | set `EPSILON_R2_KEY_ID/_SECRET/_ENDPOINT` or configure rclone `[r2]`. |
| `No module named streamlit` / plotly | `pip install -r requirements.txt` (or `uv sync`); make sure the venv is activated. |
| `pip: No module named pip` (uv venv) | `python -m ensurepip --upgrade` then `pip install -r requirements.txt`, or use `uv sync`. |
| dashboard shows a blank chart | it shouldn't — empty states say why ("No trades in this window"). If truly blank, check `python scripts/check_setup.py`. |
| wrong Python | need ≥ 3.10 (the venv targets 3.14). |
| unsure what's wrong | run `python scripts/check_setup.py` — it names the problem in a sentence. |

## What this library is for — and what it is NOT for

This library is for **looking at data**: screening markets, plotting them, auditing quality,
forming intuitions. It is **not** the backtester's feed.

**The research library does NOT feed the existing backtester.** `mm_engine/feeds/replay_parquet.py`
replays *raw events* and needs order-book **depth** for queue position; the library is deduped L1
with identity attached, for *viewing*. Do not assume a backtest can read `research_v1` today — it
cannot. **Step F (`book`) is the bridge.** Details and the scoped task are below.

### Backtest adapter — the next person's first task (scoped, not built)

`mm_engine/feeds/replay_parquet.py` reads the **raw** capture layout:
`parquet/{date}/{universe}/{table}_{shard}.parquet`, the four tables `book`/`trades`/
`price_change`/`bba`, with `market` and `received_at` columns and a `capture_gaps.parquet`
sidecar. It builds a `MarketEvent` stream ordered by exchange time.

A library-fed replay would need:
- **`market` and `received_at`** — not in `l1`/`trades` today. `market` (= `condition_id`) is
  **recoverable** by joining `asset_id → condition_id` from `tokens.parquet`; `received_at` (the
  local ISO receive time) is **not** in the library — only `received_ns` (monotonic) and
  `timestamp_ms` (exchange) — so it needs a rebuild of `l1`/`trades` to carry it, or the engine
  adapted to use `timestamp_ms`/`received_ns`.
- **`book`** — does **not** exist yet. This is **Step F** and it is the real dependency: queue
  position cannot be modelled from L1 alone.
- **`l1` is deduped to touch-moving rows** — it is neither `price_change` nor `bba`. It can
  legitimately produce `best_bid_ask`-style L1 events (the touch over time); it **cannot**
  reproduce full `price_change` depth deltas or a `book` snapshot stream.
- **the gap sidecar** — that information now lives in `coverage()` (active/quiet/gap per hour) and
  the documented outage; an adapter would synthesize `GapMarker`s from it rather than a
  `capture_gaps.parquet`.

**Why do it:** replaying from the library means one clean, verified, documented dataset instead
of 71 GB of raw JSON-derived parquet with no identity attached — the whole reason the library
exists. Scope: build `book` (Step F), then a `research_v1 → MarketEvent` adapter; the anti-drift
discipline (parity vs a known-good replay) applies.

## The two-book finding (counter-intuitive — read before you "discover" it)

- We have **both** tokens' quotes and trades. Nothing is missing.
- **NO quotes are the exact complement of YES quotes**: `NO_bid = 1 − YES_ask`, at the 1st and
  99th percentile across 1.29 M observations. They carry **no independent information**.
- This is **Polymarket's design, not our capture**: the daemon subscribes to both tokens and
  stores `best_bid`/`best_ask` verbatim; nothing computes a complement. The matching engine mints
  a complete set from a YES buyer and a NO buyer, so a bid on NO *is* an ask on YES — two order
  ledgers over one pool of liquidity.
- **Consequence:** plotting both mids is redundant by construction. **Trades are not** — those are
  two genuine streams of intent (someone bearish must *buy NO*, they cannot short YES).
- **Honest limit:** with no independent NO quotes, this dataset **cannot** answer whether
  cross-book arbitrage ever existed. That is a limit, not a finding.

## Known gaps & a data caveat

- **Outage:** 2026-06-22 15:00 → 06-23 08:29 UTC (~17 h, both universes) — no book, no trades.
- **Capture start ramp:** 2026-06-19 h00-11 (both) — not a loss.
- **Reboot tail:** 2026-08-21 h21-23 (both).
- **esports quiet hours:** book present, no trading (e.g. 07-24 h12, 08-21 h11-12) — **not gaps**.
  `coverage()` separates `quiet_hours` from `missing_hours` (true gap = no book).
- **Caveat:** `median_mid − median_spread/2` goes **negative for 71 tokens** — two independently
  taken medians cannot rebuild a book. Use `l1` for a real bid/ask at a time, not the medians.
