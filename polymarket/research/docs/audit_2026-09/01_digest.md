# research_v1 — audit digest

**Auditor:** Claude (for Justin) · **Date:** 2026-09-01
**Repo state:** `alvaro` @ `9810c2b` (fast-forwarded from `4268744` this session)
**Data:** `polymarket/research/data/research_v1`, 792 files / 1,167,809,048 bytes, fetched 2026-09-01T17:43:12Z, all four self-checks PASS (`_fetch_receipt.json`)
**Environment:** `.venv` Python 3.14.0, boto3 1.43.85 installed this session, `check_setup.py` all-green, `pytest tests/test_loader.py` 10/10 pass

Paths are relative to `polymarket/research/` unless stated. Everything below was re-derived from the parquet, not copied from the docs; where I could not verify something I say so.

---

## 0. Headline for an auditor

The **token-level numbers in the docs are accurate** — I re-derived every count in `epsilon_data/README.md` from `tokens.parquet` and they match exactly, to the row. The identity/verification story holds up.

The problems are **not in the counts**. They are in three places:

1. **`negrisk_sum()` does not do what its docstring says**, and the size of the gap is large (median quote being summed is **20.2 hours stale**). Every NegRisk number in the docs, including the *replacement* for the known-invalid figures, inherits this. §8.1.
2. **Two of the seven documents you told me to read describe a different dataset** (the copytrade `closed_positions`/`traders` pipeline, ~1.06 B fills). Nothing in `docs/METRICS_REFERENCE.md` or `RESEARCH_FINDINGS.md` documents `research_v1`. §8.2.
3. **A scatter of doc-vs-code contradictions**, one of which (the outage window) is wrong in a way that leaves a 57-minute hole in anyone's filter. §8.

The known-invalid NegRisk figures (`median 1.007, p5 0.081, p95 1.579, 59 events above 1.05`) **do not appear anywhere** in the library, its docs, the dashboard, or the notebooks. That sweep is clean — but the number that *replaced* them is also not reproducible. §9.

---

## 1. What the dataset contains

### Provenance
64 capture days, **2026-06-19 → 2026-08-21** (verified: 64 distinct dates in both `l1` and `trades`, both universes). Two universes: `politics_negrisk` (NegRisk election/economics) and `esports` (match markets). Source archive was 3,832,521,222 raw `price_change` rows / 71 GiB; `research_v1` is the reshaped 1.09 GiB view of it (`HANDOVER.md:9-15`, `_manifest.json` `daemon_counter_correction`).

Built by commit `043d64f03dc9d20f7adaee93369a4f2ef012c8f2` (`_manifest.json`).

### Tables, grain, verified row counts

| file | grain | rows (doc) | rows (**I verified**) |
|---|---|---:|---:|
| `tokens.parquet` | one row per token | 30,772 | **30,772** ✓ |
| `l1/universe=…/month=…/*.parquet` (779 files) | one row per **touch-moving** L1 event | 101,051,502 | **101,051,502** ✓ |
| `trades/universe=…/month=…/*.parquet` (6 files) | one row per trade print | 7,227,528 | **7,227,528** ✓ |
| `coverage.parquet` | one row per (universe, date) | — | **128** (64 dates × 2) |
| `obs_stats.parquet` | per-asset observation stats | — | shipped, **never read by any code** (§8.6) |
| `pc_file_manifest.txt` | archive file list | — | shipped, **never read by any code** (§8.6) |
| `e2_counts.csv` | per (universe, month) raw/kept counts | — | 6 rows, **undocumented** (§8.6) |
| `exclusions.csv` | operator exclusions | empty | **empty** ✓ (header + 2 comment lines only) |

### Universe split (verified, not in any doc at this granularity)

| universe | tokens | markets | events | l1 rows | trade rows |
|---|---:|---:|---:|---:|---:|
| esports | 27,918 | 13,959 | 4,170 | 87,215,904 | 5,632,001 |
| politics_negrisk | 2,854 | 1,427 | 273 | 13,835,598 | 1,595,527 |
| **total** | **30,772** | **15,386** | **4,443** | **101,051,502** | **7,227,528** |

esports is **91% of the tokens and 86% of the L1 volume.** Every dataset-wide distribution in the dashboard is therefore an esports statistic wearing a general label. Politics is the small half.

The "15,386" in `epsilon_data/README.md:9` and `catalog.py:53` is correct — it is the distinct `condition_id` count.

### Dedup ratio (verified against `e2_counts.csv`)
Raw price_change 3,832,521,222 → L1 kept 101,051,502 = **2.637%** (docs say 2.64% ✓). The per-month rows in `e2_counts.csv` sum exactly to both published totals.

---

## 2. What a "token" is, and the linkage

Four levels (`epsilon_data/README.md:30-40`):

```
universe          politics_negrisk | esports
└── event         event_id, event_slug, event_title, event_end_date   (4,443)
    └── market    condition_id, market_slug, question                 (15,386)
        └── token asset_id  ← THE ORDER BOOK LIVES ONLY HERE          (30,772)
```

- A **token** is one side of one market: YES/NO for politics, team A/team B for esports. `asset_id` is a 77-digit uint256 string.
- **token → market**: `asset_id → condition_id` (the on-chain CTF hash). Exactly 2 tokens per market (30,772 = 2 × 15,386, exact).
- **market → event**: `condition_id → event_id`. Politics events are NegRisk baskets of many candidate markets (max observed: 52 markets in `presidential-election-winner-2028`).
- **The complement**: `complement_asset_id` points at the other side. `outcome_index ∈ {0,1}` orders them.
- **`path`** (`universe/event_slug/market_slug/outcome_label`) is the human-readable unique key and what `resolve()` accepts, so nobody types a 77-digit id.

**Linkage is verified, and the verification is real** — five check columns per token, described at `epsilon_data/README.md:100-102`: `check1_roundtrip`, `check2_pairing`, `check3_negrisk`, `check4_status`, `check5_indep`. 30,702 of 30,772 tokens are `identity_status='resolved'`; **70 tokens across 35 conditions are `unresolved`** (Gamma could not name them) and are kept with null identity and a synthetic path rather than dropped (`epsilon_data/README.md:133-134`, verified: 70 tokens / 35 conditions).

---

## 3. Units and type conventions — the things that silently corrupt

These are the traps that produce a wrong answer *without an error*. All verified.

| trap | detail | evidence |
|---|---|---|
| **ids are strings, always** | 77-digit `asset_id` becomes a float silently and every join then fails invisibly. The loader casts `asset_id`, `condition_id`, `event_id`, `complement_asset_id` to `string`. You must too. | `_internal.py:23`, `:87-89`; `README:117-118` |
| **dollars vs cents in the same row** | `median_mid`, `last_mid`, `median_spread`, `best_bid/ask/mid`, `price` are **DOLLARS (0–1)**. `median_spread_cents` and `spread_c` are **CENTS**. `mid ± spread/2` is only valid with the dollar spread. | `README:131-132`, `_manifest.json.units` |
| ↳ **how bad the mistake is** | Using the cents spread by mistake: **27,333 of 30,772 tokens** produce a negative bid. Using the correct dollar spread: **71 tokens** (the documented caveat). A 385× difference in the error rate — the wrong-unit version looks catastrophically broken, which is at least loud. | verified |
| **`outcome` is null for esports** | Only **4,836 of 30,772** tokens carry `YES`/`NO` (politics 2,854 + esports props 1,982). Use `outcome_label` (verbatim team name) for the rest. Anything keyed on `outcome == 'YES'` silently drops 84% of the dataset. | `README:119-121`, verified exactly |
| **`l1` is NOT every message** | A row exists only where the touch moved. Between rows the touch is unchanged — **carry the last value forward** is the correct read, not an approximation. Any per-row average over `l1` is an average over *touch-moves*, not over time; it needs time-weighting. | `README:128-130` |
| **`near_half` is not a failure** | 21,522 of 30,772 tokens (70%) — the **largest** bucket. Settled markets whose book never traded to the extreme. Filtering it out as "bad data" deletes most of the dataset. | `README:122-127`, verified |
| **`spread_c` = 0 and = 100¢ are normal** | Locked/one-tick and one-sided books. `audit.py:114` treats them as informational, deliberately. | `audit.py:114-117` |
| **`first_seen`/`last_seen` are DOUBLE, not int** | README:95 says "int epoch-ms"; the parquet column type is `DOUBLE`. Harmless at these magnitudes but it will surprise a strict schema check. | verified vs `README:95` |

### `check4_status` — verified counts

| value | doc | verified | meaning |
|---|---:|---:|---|
| `near_half` | 21,522 | **21,522** ✓ | settled, book never left ~0.5 (esports 21,082 + politics 440 — both verified) |
| `converged_correct` | 6,650 | **6,650** ✓ | winner → ~1, loser → ~0 |
| `not_resolved` | 2,148 | **2,148** ✓ | still open |
| `no_convergence` | 328 | **328** ✓ | |
| `no_price` | 40 | **40** ✓ | (these are exactly the 40 rows with null `median_spread`) |
| `inverted` | 14 | **14** ✓ | genuine upsets — mapping still correct |
| null | 70 | **70** ✓ | the `unresolved` identity tokens |

Every published figure in this table is exact. This is the strongest part of the dataset.

---

## 4. The loader's public API

14 functions, `epsilon_data/__init__.py:20-26`. Timings are wall-clock on this machine, local data root, warm OS cache — they are *relative* guidance, not benchmarks.

| function | returns | cost (measured) | notes / traps |
|---|---|---|---|
| `catalog(universe, event, resolved, min_trades, min_days, apply_exclusions=True)` | 30,772-row DataFrame, 34 columns + `excluded` | **109 ms cold, 0.8 ms warm** (`lru_cache(4)` on the whole table) | the entry point. `apply_exclusions=False` shows excluded rows *marked*, which is what the dashboard does |
| `events(universe, apply_exclusions=True)` | event rollup | **25 ms** | returns **4,444 rows for 4,443 event_ids** — the 70 null-identity tokens form one extra `NaN` group (`groupby(dropna=False)`, `catalog.py:41`). Docstring says "one row per event" |
| `search(text, limit=50)` | matching token rows | **27 ms** | case-insensitive substring over `event_title`, `question`, `market_slug`, `event_slug`. **Does not apply exclusions** — it marks `excluded` but never filters, unlike `catalog()`. `limit` truncates *after* matching, so it is not "top N by relevance", it is "first 50 in table order" |
| `resolve(ref)` | `asset_id` | ~ms | accepts asset_id \| path \| market_slug. A `market_slug` returns the **outcome_index-0 side only** (`_internal.py:159-161`) |
| `load_l1(ref, start, end)` | one token's L1 tape, UTC `ts` prepended | **150 ms** for the busiest token (305,761 rows); **69 ms** for a 2.4k-row token | reads only that token's universe partitions; parquet stats skip the rest |
| `load_trades(ref, start, end)` | one token's prints | **33 ms** (1,114 rows) | |
| `load_pair(condition_id, start, end)` | both sides' mids, time-aligned | **312 ms** (62,841 aligned rows) | **1-second buckets + unbounded forward-fill** (§8.1). Raises if the market doesn't have exactly 2 tokens |
| `load_event(event_slug, start, end)` | every token in an event, mids aligned | seconds for a 100-token event | same ffill caveat, larger |
| `coverage(universe)` | per (universe,date) active/quiet/gap hours | **13 ms** | reads `coverage.parquet` — **not** `pc_file_manifest.txt` as the README claims |
| `reconciliation()` | 2-row identity check | **67 ms** (counts both tables) | both identities hold: 101,051,502 and 7,227,528 ✓ |
| `activity_by_time(universe)` | trades by UTC hour × weekday | **69 ms cold, 0 ms warm** | full 7.2 M-row trades scan, cached. weekday 0=Sunday |
| `markout(ref, horizons=(10,30,60), start, end)` | per-trade maker-view markout | **202 ms** | **loads the token's entire L1 tape regardless of `start`/`end`** (`tape.py:61` passes no bounds) — so a 1-hour window on a busy token still pays the full-tape cost. Negative = adversely selected |
| `audit_market(ref)` | verdict + evidence + recommended exclusion lines | **1.67 s** | the slowest call by 5×: it loads L1 + trades + the aligned pair. Never writes |
| `write_exclusion(asset_id, scope, reason, who)` | the line written | ms | **the only writer.** Refuses an s3 root. Append-only |

**Cost model in one line:** everything token-scoped is 30–300 ms; `catalog`/`coverage`/`activity` are cached to ~0; `audit_market` is 1.7 s; nothing here reads the whole 101 M-row L1 table, and you should not either (`README:81-82`).

---

## 5. Dashboard panels

Auto-discovered from `dashboard/panels/` — any file not starting with `_` self-registers via `@panel(...)` (`panels/__init__.py:13-16`). Adding a panel is adding a file; `app.py` names none of them.

### Audit section (dataset-wide, no market selected) — all **data-quality**

| panel | file | question it answers | verdict |
|---|---|---|---|
| **Reconciliation** (order 10) | `reconciliation.py` | "Do catalog sums still equal table row counts?" | **data-quality.** The drift tripwire. 2 metrics, MATCH/MISMATCH |
| **Coverage calendar** (20) | `coverage.py` | "Which hours are active / quiet / a true gap?" | **data-quality.** Blue=active, grey=quiet (book, no trades — *not* a gap), red=true gap |
| **Distributions** (30) | `distributions.py` | "What do spread, trades/token, days alive, median mid look like across the universe?" | **data-quality**, but see §8.5 — it is 91% esports and unlabelled as such; the clipping (`≤50¢`, `≤500` trades) is silent |
| **Outliers** (40) | `outliers.py` | "Show me the tails: widest spreads, zero-trade, fully empty, inverted, unresolved, busiest" | **data-quality.** Six tabs, every row carries its `path` so you can open it |
| **Activity by time** (50) | `activity.py` | "When is there flow to capture?" (UTC hour × weekday heatmap, per universe) | **strategy-ideas** dressed as audit — "when is there flow" is a session/opportunity question. The caption itself says the politics/esports contrast "is itself a finding" |
| **Stale-book cohort** (60) | `stale_cohort.py` | "21,082 esports tokens settled 1/0 but the book never left ~0.5 — artefact or money left on the table?" | **strategy-ideas.** This is an alpha hypothesis with a data-quality framing. It is the most interesting panel in the app |

### Explore section (needs a selected market)

| panel | file | question | verdict |
|---|---|---|---|
| **Market** (10) | `market.py` | "What did this market do?" — price + bid/ask band + trade prints, signed volume, order-flow imbalance + cumulative signed, spread, realised vol, on one linked time axis | **strategy-ideas.** The OFI/cumulative-signed subplot is explicitly an adverse-selection read ("are we being run over") |
| **Markout (adverse selection)** (20) | `markout.py` | "Was the resting quote picked off? Negative markout = adversely selected" | **strategy-ideas.** This is a market-making viability panel |
| **NegRisk** (30) | `negrisk.py` | "Do the candidates' YES mids sum to 1?" | **both** — presented as data-quality, but a basket sum ≠ 1 is an arb signal, and the caption is careful to say it is "a diagnostic, not a clean arb signal". See §8.1: the sum it plots is not instantaneous |

**Split: 4 clean data-quality, 3 clean strategy-ideas, 2 mixed.** The Audit/Explore division does *not* map onto the quality/ideas division — `stale_cohort` and `activity` are idea panels living in the Audit tab.

---

## 6. What this library is explicitly NOT for

Stated plainly by the authors, and I agree with all of it:

- **It is not the backtester's feed.** `mm_engine/feeds/replay_parquet.py` replays *raw events* and needs order-book **depth** for queue position. `research_v1` is deduped L1 with identity attached, for *viewing*. "Do not assume a backtest can read `research_v1` today — it cannot." (`epsilon_data/README.md:224-232`, `HANDOVER.md:92-101`.)
- **It cannot answer cross-book arbitrage questions.** No independent NO quotes exist (§7). "That is a limit, not a finding." (`README:272-273`.)
- **It cannot produce `price_change` depth deltas or `book` snapshots** — only `best_bid_ask`-style touch events. `book` does not exist; that is Step F and it is the real dependency (`README:247-251`).
- **The medians are not a book.** `median_mid − median_spread/2` is meaningless (negative for 71 tokens). Use `l1` for a real bid/ask at a time (`README:282-283`).
- **`exclusions.csv` is not an automatic filter.** Nothing auto-excludes, ever. Verification results are columns to look at, never filters that fire on their own (`CONTRIBUTING.md:7-8`).

---

## 7. Every gap and caveat the authors admit, in one list

| # | admitted caveat | where |
|---|---|---|
| 1 | **The one real outage**: 2026-06-22 15:00 → 06-23 08:29 UTC (~17 h, both universes, OOM crash) — no book, no trades | `HANDOVER.md:72`, `epsilon_data/README.md:138`, `:277`, `_manifest.json`, `audit.py:17`, `catalog.py:88` — **and they contradict each other, §8.3** |
| 2 | **Capture-start ramp** 2026-06-19 h00-11, both universes — not a loss | `README:140`, `:278` |
| 3 | **Reboot tail** 2026-08-21 h21-23, both universes | `README:141`, `:279` |
| 4 | **esports quiet hours**: book present, no trading — *not* gaps. `coverage()` separates them | `README:142-144`, `:280-281`, `HANDOVER.md:74` |
| 5 | **No independent NO quotes**: `NO_bid = 1 − YES_ask` exactly. Both mids redundant by construction; trades are the two real streams of intent | `README:261-273`, `HANDOVER.md:75-77` |
| 6 | **Consequence**: the dataset cannot answer cross-book arbitrage | `README:272-273`, `HANDOVER.md:77` |
| 7 | **`median_mid − median_spread/2` negative for 71 tokens** — two independent medians cannot rebuild a book | `README:282-283`, `HANDOVER.md:78`, `_manifest.json.notes` |
| 8 | **NegRisk right tail**: sums centre on 1 but a right tail remains (Elon-tweet-range ~2.35), attributed to mid-overstatement + ffill of quiet candidates + possibly non-mutually-exclusive bundles | `HANDOVER.md:83-85` — **not reproducible, §8.1/§9** |
| 9 | **NegRisk sum ffill caveat**: "forward-fills quiet candidates, so it can overstate… read it as a diagnostic, not a clean arb signal" | `negrisk.py:38-42` (panel caption **only** — not in the loader docstring, §8.1) |
| 10 | **v1 does not store Gamma's full listed-candidate count** — a sum < 1 may just be missing candidates | `catalog.py:143-146`, `negrisk.py:41-42` |
| 11 | **The stale-book cohort is an open question**: 21,082 esports tokens settled 1/0 with the book at ~0.5 — "artefact, or money left on the table?" | `HANDOVER.md:86-87`, `stale_cohort.py:21-23` |
| 12 | **esports quiet vs thinly-captured is unresolved** | `HANDOVER.md:88` |
| 13 | **`near_half` is the biggest bucket and is NOT a failure** | `README:122-127`, `HANDOVER.md:68` |
| 14 | **70 unresolved-identity tokens** kept with null identity + synthetic path | `README:133-134` |
| 15 | **14 `inverted` tokens** — genuine upsets, mapping still correct | `README:126-127`, `audit.py:177-178` |
| 16 | **Library does not feed the backtester**; `book` (Step F) is the missing dependency | `HANDOVER.md:92-101`, `README:224-259` |
| 17 | **`received_at` is not in the library** — only `received_ns` (monotonic) and `timestamp_ms` (exchange); a library-fed replay needs a rebuild or an engine tweak | `README:243-246`, `HANDOVER.md:97-98` |
| 18 | **The R2 key is read/write and the 71 GB raw archive has no backup**; a read-only scoped token is "the eventual fix" | `README:205-209`, `HANDOVER.md:54-57` |
| 19 | **Exclusions are empty today** and are an operator instrument only | `README:54`, `:77-79`, `CONTRIBUTING.md:7-8` |
| 20 | **Anti-drift test is load-bearing**: "if the loader ever silently diverges, everything built on it is wrong and nothing else would catch it" | `CONTRIBUTING.md:68-71`, `tests/test_loader.py:6-8` — **its actual coverage is narrower than claimed, §8.7** |

---

## 8. What I found wrong

Ordered by how much it would cost you.

### 8.1 `negrisk_sum()` is not instantaneous — the median quote it sums is 20.2 hours stale ⚠️ **most important**

**The claim.** `catalog.py:139-146`: "The NegRisk YES-sum for an event computed **at a common timestamp** — never a sum of per-candidate medians (which is invalid…)". Returns `n_live` = "**how many candidates were quoting then**". `.attrs['note']` (`catalog.py:157`): "missing candidates pull the sum down **only**." Panel subtitle (`negrisk.py:25`): "YES sum (instantaneous, common timestamp)".

**The code.** `negrisk_sum` → `load_event` → `align_mids` (`_internal.py:219-231`), which floors `ts` to 1-second buckets, takes the last value per bucket, outer-joins, and **`.ffill()` with no limit**.

**Measured** on `elon-musk-of-tweets-july-21-july-28` (25 YES legs, 135,931 one-second buckets):

| | value |
|---|---:|
| mean legs *actually updating* in a given 1s bucket | **1.31 of 25** |
| median age of the quote being summed | **72,836 s = 20.2 hours** |
| p90 age | 486,100 s = 135 hours |
| max age | 874,533 s = **10.1 days** |
| fraction of summed values older than 1 hour | **67.7%** |
| yes_sum median, **as `negrisk_sum` returns it** | **4.528** |
| yes_sum median, **true instantaneous (no ffill)** | **0.270** |

Same pattern on `presidential-election-winner-2028` (52 legs): `n_live` averages **44.15**, but only **1.01 legs** actually update per second; ffilled median 0.964 vs true-instantaneous median 0.090.

**Three separate defects:**
1. **"at a common timestamp" is a misnomer.** It is a sum of last-known mids at a median staleness of 20 hours — the *same class of error* as the sum-of-medians the docstring explicitly condemns, just with a finer stale-clock.
2. **`n_live` is documented wrong.** It is `notna().sum()` *after* ffill, so it counts candidates that have **ever** quoted by that instant, not candidates quoting then. Off by 44× vs 1× on the 2028 event.
3. **`.attrs['note']` is directionally wrong.** It says missing candidates pull the sum down *only*. ffill makes dead candidates pull it **up** — which is precisely the right tail the authors are puzzling over in `HANDOVER.md:83-85`.

The panel caption (`negrisk.py:38-42`) *does* admit the ffill overstatement. The loader docstring and the `.attrs` note — what a notebook user sees — do not. The honest read: **for these events the instantaneous NegRisk sum is not measurable from this dataset**, because fewer than 2 of 25 legs quote in any given second.

### 8.2 Two of the seven prescribed documents are about a different dataset

- **`docs/METRICS_REFERENCE.md`** (1,098 lines) documents `raw_trades` (1,064,500,317 fills), `closed_positions.parquet` (269,974,929 rows), `traders.parquet` (2,576,698 rows), `bankroll_timeseries`, and `cohorts/` — the copytrade pipeline. Snapshot date **2026-05-10**, coverage 2022-11-21 → 2026-04-24. It contains **zero** columns of `research_v1`.
- **`RESEARCH_FINDINGS.md`** (308 lines) is the phases 1–4 log for that same pipeline. Also zero `research_v1` content.

`research_v1` is a 2026-06-19 → 08-21 order-book capture. The two datasets share a domain and nothing else — no shared table, column, grain, or time range. Anyone following the prescribed reading order will arrive at `research_v1` believing `phantom_position_score`, `mkt_*`/`pos_*`, and the operator deny-list are part of it. They are not.

Worse: the repo's own canon audit (`brain/POLYMARKET_BRAIN.md`, "Pre-Alvaro pipeline trust audit 2026-07-21") records that **the historical position pipeline is condemned pending bundle-aware regeneration** and that A17's calibration table is "condemned as evidence". `METRICS_REFERENCE.md` still carries `status: closed` with no such banner, and `RESEARCH_FINDINGS.md` presents the cohort/candidate tables at face value. I did not re-audit those numbers — out of scope here — but a reader of `01_digest` should know they are already contested elsewhere in this repo.

### 8.3 The documented outage window is wrong at both ends

| source | stated window |
|---|---|
| `_manifest.json` `known_outage` | 06-22 **15:00** → 06-23 **08:29** |
| `HANDOVER.md:72` | 06-22 15:00 → 06-23 08:29 |
| `epsilon_data/README.md:277` | 06-22 15:00 → 06-23 08:29 |
| `epsilon_data/README.md:138` | 06-22 15:00 → 06-23 **08:00**, "~17 h" |
| `catalog.py:88` (the `coverage()` docstring) | 06-22 15:00 → 06-23 **08:00** |
| `audit.py:17` (`_OUTAGE`, used to explain gaps) | 06-22 15:00 → 06-23 08:29 |

**Measured from the parquet:**
- last L1 event before the gap: **2026-06-22 14:03:30 UTC** (both universes, within 0.5 s of each other)
- first L1 event after: **2026-06-23 08:29:10 UTC** (both universes, within 0.01 s)
- rows between 14:03:31 and 15:00 on 06-22: **0 L1, 0 trades**

So the true gap is **18 h 26 min starting 14:03**, not ~17 h starting 15:00. The "15:00" comes from `coverage.parquet` hour-granularity (hour 14 is marked present because it has *some* data), and every doc inherited it.

**Why it matters concretely:** `audit.py:141-144` uses `_OUTAGE` to decide whether a >12 h gap in a token's tape is "explained". A token whose gap starts at 14:03 is still flagged as an unexplained gap only if it doesn't overlap — it does overlap, so it passes. The real cost is downstream: anyone who filters `15:00 ≤ t < 08:29` to exclude the outage leaves **57 minutes of dead capture in their data**, looking exactly like a quiet market. The two files that say `08:00` also under-state the end by 29 minutes.

Additionally the overlap test at `audit.py:143` is loose: *any* long gap that merely touches the outage window counts as explained, so a three-month hole overlapping 06-22 would be silently excused.

### 8.4 `check_setup.py` cannot see `.env`, but every doc tells you to use one

`HANDOVER.md:47` and `epsilon_data/README.md:201` both say: put the credentials "in a gitignored `.env`". `requirements.txt:9` ships `python-dotenv`.

**Nothing in `epsilon_data/`, `check_setup.py`, or `fetch_data.py` ever calls `load_dotenv()`.** `check_setup.py:46` reads `os.environ.get("EPSILON_DATA_ROOT")` directly; `config.py:17` and `_r2_creds()` (`_internal.py:42-43`) likewise. The only `load_dotenv` calls in the repo are in unrelated SPCX scripts (`scripts/spcx_pm_pdf_monitor.py:645`, `scripts/spcx_rehearsal.py:79`).

Demonstrated this session: I appended `EPSILON_DATA_ROOT` to `.env` exactly as `HANDOVER` instructs, re-ran `check_setup.py`, and it still reported `EPSILON_DATA_ROOT is not set`. It only passed once I exported the variable in the shell. A newcomer following the documented path hits a failure the troubleshooting table (`README:215`) answers with "set it (see Quickstart)" — the advice that just failed.

Related: `README:34` and `HANDOVER.md:34` give the install as `.venv/Scripts/pip` (Windows) / `.venv/bin/pip`. This `.venv` is uv-created and **has no `pip` binary at all**, so the documented command fails outright. The README anticipates this at `:219` for a different symptom. `uv pip install -r requirements.txt` worked; only `boto3` was missing, as expected.

### 8.5 The Distributions panel is an esports panel labelled as a universe panel

`distributions.py:22-27` plots `cat_all()` — all 30,772 tokens, 91% esports — with no universe split and no caption saying so, under the heading "Distributions (counts, not just shapes)". The two clips (`median_spread_cents ≤ 50`, `n_trades ≤ 500`) are applied silently in `_hist(..., hi=)` with only the axis title recording them, so the rightmost bar is a pile-up of everything beyond the clip, not a real bin. Every other audit panel either splits by universe (`activity.py:13`) or offers a radio (`coverage.py:14`). This one doesn't, and politics — the half of the dataset anyone is actually going to trade — is invisible in it.

### 8.6 Three shipped files that nothing reads, and one that no doc mentions

- `pc_file_manifest.txt` (650 KB). `epsilon_data/README.md:55` says it "**drives `coverage()`**". It does not: `catalog.py:73-79` reads `coverage.parquet`. `grep` across `epsilon_data/`, `dashboard/`, `scripts/`, `tests/` finds **no code reference at all**.
- `obs_stats.parquet` (2.6 MB). Documented as "per-asset observation stats (feeds `tokens`)" — feeds the *build*, not the library. Never read at runtime.
- `e2_counts.csv`. **Not in `epsilon_data/README.md`'s file table and not in `_manifest.json`'s `tables` block.** It is the per-(universe,month) raw→kept ledger and it is the only place the 3.83 B → 101 M dedup is reconcilable per partition. It is the most auditable file in the dataset and it is undocumented.
- Conversely, `coverage.parquet` **is** in `_manifest.json` but is **missing from the README file table** — which lists `pc_file_manifest.txt` in the slot where `coverage.parquet` belongs.

### 8.7 The anti-drift test proves less than the docs claim

`CONTRIBUTING.md:68-71` and `README:160-161`: it "**proves the loader returns exactly what a raw parquet read returns**", and "if the loader ever silently diverges from the data, everything built on it is wrong and **nothing else would catch it**".

What `test_anti_drift_l1` (`tests/test_loader.py:45-61`) actually does: samples **5 tokens of 30,772** (the 4 busiest + 1 smallest, `_sample_assets:21-26`), compares row count, first/last row on 5 columns, and an md5 checksum over **`mid` only**. `best_bid`, `best_ask`, `spread_c`, `asset_id`, and every trades column are checksummed nowhere. `load_trades` has **no anti-drift test at all**. A drift confined to `best_ask`, or to any token outside those 5, passes green.

It is a good smoke test. It is not the proof the docs say it is — and the docs lean on it as the single thing standing between the library and silent corruption.

Two smaller test issues:
- `tests/test_loader.py:117` — `d = ed.to_dict() if hasattr(ed, "to_dict") else None` under the comment "audit NEVER writes". `ed` is the module; this asserts nothing and is dead. (The real coverage is `test_audit_never_writes` at `:120`.)
- `test_write_exclusion_roundtrip` (`:131-148`) takes a `tmp_path` fixture and ignores it — it writes to the **real** `data/research_v1/exclusions.csv` and restores it in a `finally`. It passes, but a hard kill mid-test leaves a mutated data file.

### 8.8 `markout()` carries the last touch forward without bound

`tape.py:68-70` takes the mid at `trade_ts + Δ` via `merge_asof(direction="backward")` with no tolerance. Measured on the three busiest politics tokens, **6.6% / 68.1% / 88.0%** of trades have no new L1 row inside a 60 s horizon.

**This is mostly correct, not a bug**: under L1 dedup semantics the absence of a row *means* the touch didn't move, so carrying forward is the right read. I want to be precise about that rather than inflate it.

The narrow real defect: there is **no staleness bound**, so where the tape genuinely ends — a token's last quote, the 18 h outage, end of capture — `markout` silently returns a number computed from a *pre-trade* mid instead of `NaN`. Measured: 7–10 trades per busy token fall after their token's last L1 row (~0.03%). Small, but it means the markout panel's own guard — "Trades exist but no post-trade mid within the horizon" (`markout.py:24`) — can essentially never fire, because `merge_asof` backward almost always returns something. The panel reports a mean and a "% of fills with negative markout" over a window that may be mostly carried-forward mids, with no staleness indicator.

### 8.9 `HANDOVER.md:108` points at a directory that doesn't exist

"`brain/handoff/reports/step{1..J}.md` and `brain/handoff/LOG.md` — the full build history, findable if you want it."

There is no `brain/handoff/`. The directory is `brain/handoffs/` (plural), it contains 40 dated handoff notes, and **no `step*.md` and no `LOG.md` anywhere in `brain/`**. The full build record for the dataset I just audited is, as far as I can find, not in this repo. I did not locate it elsewhere — flagging as unknown rather than guessing.

### 8.10 Smaller items

- `events()` returns 4,444 rows for 4,443 events (§4) — the null-identity group. Cosmetic, but a `.set_index("event_id")` on it will carry a `NaN` key.
- `search()` marks but does not drop excluded tokens, while `catalog()` drops them by default. Two functions, same concept, opposite defaults, neither README line says so (`README:61-63`, `:77-79`).
- `README:9` — "find a market among 15,386" — is correct, but the same README's tree section says the order book is at token level (30,772). Both true; easy to misread as one number being wrong.
- `README:265-266` cites "1.29 M observations" for the NO_bid = 1 − YES_ask finding. I could not reproduce that exact sample (my 5-market politics sample gave 982,153 paired observations). **The finding itself replicates** — see §9 — but the sample size is unverifiable from what's shipped.

---

## 9. Numbers that look carried forward from an earlier draft

**The condemned NegRisk figures — `median 1.007, p5 0.081, p95 1.579, 59 events above 1.05` — appear nowhere.** Sweep performed over: all `*.md` in the repo (excluding `.git`), all `*.py` and `*.json` under `polymarket/research/`, all 143 notebooks under `notebooks/`, and `brain/`. The only hits on those digit-strings are unrelated coincidences (a TFI notebook's `0.081`, a lead-lag CI bound `0.0819`, an MLB row `40.0814`). **Clean.**

**But the number that replaced them does not reproduce.** `HANDOVER.md:83-85` states: NegRisk sums "measured correctly (instantaneously) they centre on 1, but a right tail remains (**Elon-tweet-range events ~2.35** with tight books)".

Measured with the library's **own** `negrisk_sum()` across all **173** politics events with ≥2 markets:

| statistic | value |
|---|---:|
| median of per-event median yes_sum | **0.995** ("centre on 1" ✓) |
| p95 of per-event median | 2.496 |
| max of per-event median | **4.528** |
| events with median > 1.05 | **28 of 173** |
| events with max > 1.05 | 86 of 173 |

The ten Elon tweet-range events have medians of **1.99 – 4.53** and maxima to **6.06** — roughly **2× the documented "~2.35"**, not a small drift. And per §8.1, the whole quantity is a 20-hour-stale composite; computed truly instantaneously the same event's median is **0.270**.

**Verdict:** treat `~2.35` as a stale draft figure of exactly the same family as the four condemned ones. It is not reproducible from the shipped library under either computation. The safe statement today is: *the ffilled NegRisk YES-sum centres near 1 with a heavy right tail concentrated in many-leg range events; the instantaneous sum is not measurable from this capture.*

Other figures I checked for staleness and found **sound**: every `check4_status` count, 30,772 / 15,386 / 101,051,502 / 7,227,528, 21,082 + 440 near_half, 4,836 YES/NO tokens, 70 unresolved / 35 conditions, 14 inverted, 71 negative-bid tokens, 2.64% dedup, 24 esports quiet hours across 14 days, 64 capture days. The `21,522` figure is correctly flagged in `stale_cohort.py:21` as having been a both-universes number misread as esports-only in an early draft — that correction is already made and is consistent everywhere I looked.

---

## 10. What I did not check

- I did not re-audit the copytrade pipeline behind `METRICS_REFERENCE.md` / `RESEARCH_FINDINGS.md` (different dataset, out of scope — and already contested in `brain/`).
- I did not verify `check1/2/3/5` against Gamma or an independent snapshot; I verified their *distributions*, not their *correctness*. Re-running the identity verification needs the raw archive and Gamma access.
- I did not open the Streamlit app; panels were read as source. Nothing in them is hard to predict from the code, but I have not seen them render.
- I could not locate the build history (`step*.md`, `LOG.md`) referenced by `HANDOVER.md:108`, so the "why" behind build decisions is un-audited.
- `README:265` "1.29 M observations" — sample not reproducible from shipped data (§8.10).

---

## Appendix — `_fetch_receipt.json`

```json
{
  "fetched_at_utc": "2026-09-01T17:43:12Z",
  "source": "s3://epsilon-polymarket-data/research/v1/",
  "file_count": 792,
  "total_bytes": 1167809048,
  "checks": {
    "file_count":        { "expect": 792,       "got": 792,       "ok": true },
    "total_bytes":       { "expect": 1167809048,"got": 1167809048,"ok": true },
    "tokens_rows":       { "expect": 30772,     "got": 30772,     "ok": true },
    "l1_trades_nonempty":{ "l1_rows": 101051502,"trades_rows": 7227528, "ok": true }
  }
}
```
