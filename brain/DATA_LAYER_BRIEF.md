# Data Layer — brief for Claude Code

**Read this before doing anything.** Standing context and working protocol. Supersedes the 2026-08-26 version.

Last updated 2026-08-26. Companion: `brain/DATA_LAYER_PLAN.md` (background), `brain/handoffs/2026-08-25_cowork_data_layer_session.md` (how we got here).

---

## 0 · The goal

Turn 64 days of captured Polymarket order-book data into **a clean, navigable dataset** that three people can use without asking each other questions.

The finished thing is: **one table describing every token, four tables holding every time series.** You navigate by filtering the first and reading the others. Nothing else.

Everything in this document serves that. Quality grading, backtests, plotting tools — later, and out of scope here.

---

## 1 · How we work

**One step → report → wait.** The operator (Alvaro) is judging each result, not delegating. Auditable beats fast.

- **Never chain steps.** Finishing three at once is worse than finishing none, because it can't be checked.
- **Read-only first.** Before anything that writes, deletes or uploads: run the counting or `--dry-run` version, show the numbers, say what the real run will do, then wait.
- **Anything over five minutes gets announced first.** Before starting it, post: what it does, why it's needed, what it will produce, and your time estimate. Then wait for a go-ahead. The point is to consider a cheaper route before spending the time — several times already the expensive option turned out to be avoidable.
- **If something is already running, report it** — what it is, why, how long, and what you'll do with the result.
- **Show real numbers, not summaries of them.** "Row counts match" is worthless; "source 4,182,993 / output 4,182,993" is the point. For large output: counts plus five concrete examples. Never dump thousands of lines.
- **Visualise when a picture is clearer than a table** — coverage, distributions, a market's behaviour, anything with a shape. Write reports and figures to `C:\Users\alvar\OneDrive\Documentos\Claude\Projects\Epsilon - Research\_reports\<step>\` so they can be reviewed outside your session.
- **If an instruction is wrong, say so and stop.** Exception: if following it literally would *destroy data*, do the safe thing, then flag loudly and stop. Otherwise propose and wait — do not silently improve.
- **No new dependencies without asking.**
- **A surprise is the report.** An unexpected number is worth more than a completed step.

---

## 2 · What we are building

### The tree

```
universe          politics_negrisk | esports
└── event         "Brazil Presidential Election 2026"     event_id, end_date
    └── market    "Will Lula win?"                        condition_id
        └── token YES | NO                                asset_id  ← the order book
```

The order book exists only at token level — that is the venue's shape. YES and NO of one market are not independent: a bid on NO at 40¢ is an ask on YES at 60¢.

### The tables

```
research/v1/
  _manifest.json
  tokens.parquet                           the tree — one row per token
  l1/       universe=…/month=…/*.parquet   1.7 GB   light
  trades/   universe=…/month=…/*.parquet   0.7 GB   light
  book/     universe=…/date=…/*.parquet    ~5 GB    light
  changes/  universe=…/date=…/*.parquet    60 GB    full tier
  exclusions.csv                           hand-edited, applied at load
```

**`tokens`** — ~3,000 rows, fully denormalised so every query is single-table:
`universe`, `event_id`, `event_title`, `event_end_date`, `neg_risk`, `condition_id`, `question`, `outcome` (YES/NO), `resolved_outcome`, `complement_asset_id`, `asset_id`, `slug`, `first_seen`, `last_seen`, `n_days`, plus the mapping-verification columns from §4.

`slug` is the human path, e.g. `politics/brazil-pres-2026/lula/yes`. It labels every chart.

**All four data tables begin with the same three columns** — `asset_id`, `timestamp_ms`, `received_ns` — and are sorted `(asset_id, timestamp_ms, received_ns)`. One key, one order, everywhere.

| table | extra columns | built from |
|---|---|---|
| `l1` | `best_bid`, `best_ask`, `mid`, `spread_c` | `price_change`, deduped to rows where the touch changed |
| `trades` | `price`, `size`, `side`, `fee_rate_bps`, `transaction_hash` | `trades` |
| `book` | `side`, `level_idx`, `price`, `size` — **long form, one row per level** | `book`, JSON exploded |
| `changes` | `price`, `side`, `size`, `best_bid`, `best_ask` | `price_change`, all rows |

`level_idx = 1` is always the touch, counting outward. **The raw feed stores levels the other way round — the build flips them.**

`asset_id` and `condition_id` are strings. Always. A token id is 77 digits and overflows every integer type.

### Decided, do not relitigate

- R2 is distribution, not a query engine. Queries run against a local copy.
- No database server. Parquet on disk plus DuckDB.
- `research/v1/` is immutable. Adding a new table is fine; changing a published table means `v2`.
- Layer 0 raw is never touched, so anything derived can be rebuilt.
- Convenience columns (`mid`, `spread_c`) are for viewing. The engine recomputes from levels.
- Nothing is deleted silently. Exclusion is a human edit to `exclusions.csv` with a reason and a date.
- **The 2026-08-22 → 08-26 raw days are out of scope for v1.** The archive is 64 days, 19 June to 21 August.

---

## 3 · Facts already established — do not re-derive

**Archive.** 11,984 files, 71.08 GiB, 64 dates 2026-06-19 → 2026-08-21, universes `politics_negrisk` and `esports`. Layout `{date}/{universe}/{table}_{universe}_{HH}.parquet`. Coverage 3,008 of 3,048 possible book-hours (98.7%).

**Holes.** 2026-06-19 hours 0–11 (capture started midday). **2026-06-22 15:00 → 06-23 08:00 — 17 hours, both universes, the only real loss.** 2026-08-21 hours 21–23.

**Sizes.** `price_change` 60.05 GiB (84.5%) · `book` 7.87 · `bba` 2.45 · `trades` 0.71. By universe: esports 44.53, politics 26.54.

**Traps confirmed by inspection:**

- `book.bids`/`asks` are JSON strings: array of `{"price","size"}` with **values as strings**, ordered extreme→touch so **the best level is last**. Level counts range 1 to 140.
- `price_change` carries `best_bid`/`best_ask`, contradicting the comment in `events.py`. **Only ~4% of rows move the touch** — deduping gives a full-resolution L1 tape of 1.68 GB for the whole archive.
- **56% of `price_change` rows share `timestamp_ms` with the previous row.** `received_ns` is the real tiebreak. Within a single asset, ordering on `received_ns` is clean (zero regressions).
- **Book snapshots are a ~15-minute periodic anchor, not hour-aligned** — median 13 minutes into each hour before the first one arrives. **A per-hour builder that does not carry state across hour boundaries produces a silent 13-minute blind spot every hour.**
- 50 hours have a `book` file but no `trades`/`price_change` file. **A missing table means zero rows, not missing data.**
- Two legacy whole-day files, `2026-06-19/esports/{bba,book}.parquet`, with no hour suffix. A `{table}_*` glob skips them; a `*.parquet` glob double-counts. Handle explicitly.
- A stray `_processed.txt` sits under `parquet/` in R2. Ignore. Delete nothing from R2.
- `bba` is absent for 7–16% of assets, but those assets carry **zero trades** — the quiet tail.
- The `market` column **is** the condition id (0x, 32-byte hash).
- **Identity above condition id is not in the capture at all** — no event, title, end date, question, outcome label, or `negRisk`.
- `capture_gate` costs ~110 µs/event; over the full window that is ~100 hours of CPU. **It is not a build step.** Keep it as an on-demand check for one market-day.

**Open question, low priority:** in esports, `price_change.best_bid/ask` disagrees with the reconstructed book touch 29% of the time by more than a cent (politics: 0.3%). `bba` is a third independent source and can settle which is right. Worth answering before the viewer treats either as truth.

---

## 4 · The steps

Do **one** per session. Each has a gate.

### Step A — Rescue the VPS metadata  *(time-sensitive)*

`root@89.167.68.98` is stopped and heading for decommission. These exist nowhere else:
`/opt/epsilon/l2_ingestion/data/live_universe.json` and `.../capture_gaps.jsonl`.

Copy both to `_reports/stepA/`. Report `live_universe.json`'s structure and exactly which identity fields it carries per asset — whatever it has is Gamma work we don't have to do. For `capture_gaps.jsonl`: size, date range, event-type distribution. It is the only ground truth about feed downtime.

**Gate:** both files are off that machine.

### Step B — Scope the Gamma fetch  *(scope only, do not fetch)*

How many distinct `condition_id` values across all 64 days (cheap metadata scan). Which Gamma endpoint returns, for a condition: `event id`, `event title/slug`, `question`, `outcomes` labels, `clobTokenIds`, `end_date`, `negRisk`, and resolution outcome. Rate limits. Does it still serve closed markets. Estimated total time. Proposed cache layout: **one raw JSON per condition on disk, unmodified**, so a rerun costs nothing and the mapping stays auditable.

**Gate:** a written plan with a time estimate, approved before any fetching.

### Step C — Fetch and build `tokens.parquet`

Fetch per the approved plan. Then build the identity table.

**Mechanical requirements:**
- Extract token ids **as text before any numeric parsing touches them.** A 77-digit id silently becomes a float in a careless parser and every join then fails invisibly.
- Cache every response raw and unmodified. Never re-fetch what is cached.
- Anything Gamma cannot resolve is kept with identity fields null and `identity_status = 'unresolved'` — never dropped silently.

### Step D — **Verify the mapping**  *(the most important gate in this build)*

Getting YES and NO the wrong way round inverts a price series — 0.98 becomes 0.02 — and on a longshot's chart that looks entirely normal. Nothing crashes. Every downstream number is wrong. **Assume the mapping is wrong until the price data proves otherwise.**

Write this as a **separate program** from Step C, so a bug in the mapper cannot also write the verifier. Four checks, each recorded per token in `tokens`:

1. **Round-trip on ids.** The token ids Gamma returns for a condition must be exactly the set observed in the capture for that condition — compared as strings, set-equal. Catches id corruption and wrong-market responses.
2. **Pairing.** The two tokens of a market must be complementary in our own data: `mid_A + mid_B ≈ 1` across the window. Validates the pairing, not the orientation.
3. **Orientation via NegRisk.** Across a politics event, the YES mids of all its markets must sum to ≈ 1. One candidate oriented backwards breaks this loudly. Politics is entirely NegRisk.
4. **Orientation via resolution.** For any market that resolved inside the window, the token labelled with the winning outcome must converge toward 1.00 and its complement toward 0.00. Decisive where it applies.

Report pass rates for each check, per universe. Anything failing goes to a **quarantine file with the reason** — not into `tokens`, and not dropped silently either. Then print **ten mapped markets** with question, outcome label, slug and a price sample, for human eyeball.

**Gate:** every check reported with a pass rate, quarantine list produced, ten examples eyeballed and approved. **No data table is built before this gate passes** — building on a wrong mapping wastes everything after it.

### Step E — Build `l1` and `trades`

2.4 GB combined, and **this is the milestone that matters**: with `tokens` plus these two, a market can be opened and looked at. Everything after is enrichment.

Builder requirements: **one day at a time, streaming** (download a day, build it, delete the source, next day — never more than a couple of GB on disk); **idempotent** with a per-day marker so a rerun skips completed days; **validated per day, not at the end** (row counts against source, immediately); resumable after a crash at a cost of one day.

### Step F — Build `book`

Long form, one row per level, `level_idx = 1` at the touch. **Carry book state across hour boundaries.**

### Step G — Publish `research/v1/` light tier to R2

With `_manifest.json`: version, build date, source range, per-table row counts, checksums.

### Later, not now

`changes` (60 GB, full tier) when a backtest needs queue depth. Then the loader, then the viewer.

---

## 5 · Hard rules

- **R2 is a source, never a deletion target.** No `rclone sync`/`delete`/`purge`/`move` with `r2:` as target. `rclone copy` with the bucket on the right is the only permitted write. 71 GiB, no second copy.
- **Never delete a local file** unless the same path is verified in R2 at the same byte size. Log every deletion.
- Do not touch `polymarket/execution/` — trading code, out of scope.
- Do not restart the capture pipeline or re-enable the VPS services. Capture was stopped deliberately on 2026-08-26.
- Do not edit `universes.yaml`.
- Never commit anything under `data/`. Never print or copy `rclone.conf`.
- Match existing repo conventions — find the crypto-side data loader and follow its naming rather than inventing a second dialect.

---

## 6 · Report format

```
STEP <x> — <name>   [DONE | BLOCKED | STOPPED FOR DECISION]

WHAT I RAN
WHAT I FOUND            actual numbers; five examples where output is large
WHAT SURPRISED ME
WHAT I DID NOT DO
FILES CHANGED           paths, or none
FIGURES                 path to any report/figures produced
PROPOSED NEXT           one step: what it reads, what it writes, time estimate
```

Then **stop**. Do not begin the proposed next step.

Where a build step is involved, add three short answers in the reader's language, not a systems engineer's: why this way and not the obvious alternative; what breaks first if it's wrong, and whether it breaks loudly or silently; and what a new person would misunderstand about it.
