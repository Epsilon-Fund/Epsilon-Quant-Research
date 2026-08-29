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
- **`check4_status` is categorical, and `near_half` is the biggest bucket (21,522) and is NOT a
  failure.** Values: `converged_correct` (6,650), `near_half` (21,522 — settled but the book
  never traded to the extreme, mostly esports), `no_convergence` (328), `not_resolved` (2,148),
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
