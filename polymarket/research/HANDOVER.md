# Handover — the Polymarket research library

For Gonzalo (and anyone new). You are capable; this assumes you know nothing about this project.
You should not need to read the step reports to start. Read this page, run the dashboard, then
your first task is at the bottom.

## What this is, in a page

Over ~64 days (2026-06-19 → 08-21) a daemon captured the Polymarket order book for two universes
— **politics** (NegRisk election/economics markets) and **esports** (match markets) — as raw
JSONL, then hourly Parquet: ~3.8 billion price-change events, 71 GiB. That raw archive is great
for a machine and miserable for a person: no market names, one file per hour per table, the book
stored as JSON strings.

This **research library (`research_v1`, ~1.09 GB)** is that data reshaped for *looking at it*:

- **`tokens.parquet`** — one row per token (30,772), fully named: universe → event → market →
  YES/NO (or team A/B), with identity, per-token stats, and five verification columns.
- **`l1/`** — the L1 tape (best bid/ask/mid/spread), **deduped to touch-moving rows** (101 M rows,
  the 2.64% of raw events that actually moved the touch).
- **`trades/`** — every trade print (7.2 M), taker side.
- plus `obs_stats`, `coverage`, and an (empty) `exclusions.csv`.

Every token was **identity-verified** against Gamma and against the price data itself (round-trip
ids, pair-sums-to-1, NegRisk orientation, resolution convergence, an independent snapshot). The
mapping is trustworthy — see the checks columns and the `audit_market()` tool.

You never touch raw parquet: one library, `epsilon_data`, is the only way in.

## Get it running (≈10 minutes)

```bash
cd polymarket/research
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # (uv users: `uv sync`)
```

Then get the data and point `EPSILON_DATA_ROOT` at it — **no code change either way**:

- **Fetch it locally** (recommended — no rclone): `python scripts/fetch_data.py`, then
  `set EPSILON_DATA_ROOT=<repo>/polymarket/research/data/research_v1`. Pure Python, ~1.09 GB in
  ~1–2 min, resumable (re-run to finish), read-only against R2, verifies itself and writes
  `_fetch_receipt.json`. Needs only the three R2 env vars below.
- **Straight from R2** (nothing to download): `set EPSILON_DATA_ROOT=s3://epsilon-polymarket-data/research/v1`.

Both need R2 credentials — the **same three env vars** for the fetch script and the loader:
`EPSILON_R2_KEY_ID`, `EPSILON_R2_SECRET`, `EPSILON_R2_ENDPOINT` (or a local `rclone.conf` `[r2]`
remote). Ask the operator; put them in a gitignored `.env` — never in a command string or the repo.

```bash
python scripts/check_setup.py                      # tells you what's wrong, in a sentence
.venv/Scripts/streamlit run dashboard/app.py       # search "fed july", then Explore / Audit
```

> ### ⚠️ The R2 key can write AND delete. Only ever *copy*.
> **Never `sync` / `delete` / `purge` / `move` with `r2:` as the target** (`fetch_data.py` is
> read-only and never does). The 71 GB raw archive has no other backup — a mistyped `sync` destroys
> it permanently. If you use rclone instead of the script, it's `rclone copy … -P`, **copy only**.

## The tree, tables, units, traps (the short version; full detail in `epsilon_data/README.md`)

- **Tree:** universe / event / market / token. The order book is at token level. Two tokens of a
  market are complementary (`mid_A + mid_B ≈ 1`).
- **Units:** mids and `median_spread` are **dollars** (0–1); `spread_c` / `median_spread_cents`
  are **cents**. `timestamp_ms` is exchange time; `received_ns` is the local tiebreak; the loader
  gives you a UTC `ts`.
- **Traps:** ids are **strings** always (77 digits). `outcome` is null for esports (team names —
  use `outcome_label`). `l1` is deduped touch-moves, not every message. `check4_status='near_half'`
  is the biggest bucket and is **not** a failure.

## What is known wrong or missing

- One real **outage**: 2026-06-22 15:00 → 06-23 08:29 (both universes). Capture-start ramp on
  06-19 and a reboot tail on 08-21. esports has quiet hours (book, no trading) that are **not**
  gaps — `coverage()` tells them apart.
- **No independent NO quotes**: NO_bid = 1 − YES_ask exactly (Polymarket's design). So both mids
  are redundant by construction; **trades** are the two real streams of intent. The dataset
  therefore **cannot** answer cross-book arbitrage questions.
- `median_mid − median_spread/2` goes negative for 71 tokens — don't rebuild a book from two
  medians; use `l1`.

## Open questions worth your judgment

- **NegRisk sums:** measured correctly (instantaneously) they centre on 1, but a right tail
  remains (Elon-tweet-range events ~2.35 with tight books) — mid-overstatement + ffill of quiet
  candidates, and possibly non-mutually-exclusive bundled markets. See the NegRisk panel.
- **The stale-book cohort:** 21,082 esports tokens settled 1/0 but the book never left ~0.5,
  watched to settlement. Artefact, or money left on the table?
- **esports quiet hours:** how much of esports is genuinely quiet vs thinly captured.

## Your first task — feed the backtester from the library

The library does **not** feed `mm_engine`'s backtester today, and that is deliberate: the library
is for *viewing*; the backtester replays *raw events* and needs order-book **depth** for queue
position. Making it read the library is the next piece of work, and it is **scoped for you** in
`epsilon_data/README.md` → "Backtest adapter". The short version:

- `market` (= condition_id) is recoverable from `tokens`; `received_at` needs a rebuild or an
  engine tweak to use `received_ns`/`timestamp_ms`.
- **`book` does not exist yet — that is Step F, and it is the real dependency** (no queue position
  from L1 alone).
- `l1` can produce `best_bid_ask`-style events, not `price_change` depth or `book` snapshots.

Why bother: one clean, verified, named dataset instead of 71 GB of raw JSON-derived parquet. That
is the whole reason the library exists.

## Where the detailed record lives

`brain/handoff/reports/step{1..J}.md` and `brain/handoff/LOG.md` — the full build history, findable
if you want it, not required reading. `CONTRIBUTING.md` — how to add a panel or a loader function.
