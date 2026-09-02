# research_v1 — independent audit plan and results

**Audit part 4** · 2026-09-02 · repo `alvaro` @ `9810c2b` · data `polymarket/research/data/research_v1`
Scripts: `polymarket/research/scripts/audit_checks/` (scratch, git-excluded via `.git/info/exclude` — not in the index)
Raw outputs: `./04_results/*.txt` · Prior context: `01_digest.md`, `02_dashboard_walkthrough.md`, `03_numbers.json`

**R2 usage this session, for the record:** `list_objects_v2` ×~15, `GetObject` ×19 (157 MB, all under `parquet/`), **zero** put/delete/copy. Nothing touched the 71 GB archive beyond those reads.

---

## 0. The two answers

| question | verdict | one line |
|---|---|---|
| **Is the token ↔ identity mapping right?** | **Sound on every sample; the authors' own evidence for it is not.** | 60/60 random tokens + 30/30 deliberately chosen `check3`-failed/passed tokens match two fresh oracles on all 8 fields; 20/20 resolved markets end on the oracle's winner. But the code that produced the five check columns was never committed, `check3` is undefined and 72% False, `check5` covered 0.38% of tokens, and the "pair sums to 1" check is an algebraic identity. |
| **Was anything silently dropped between raw and library?** | **No drops found; one undocumented *addition* found.** | On 5 hourly shards across both universes, every trade is present (hash sets identical, 15,313 trades) and `l1` reproduces from raw under a rule the docs never state. The shipped reconciliation is circular. And the dedup resets at part-file boundaries, injecting **195,984 rows (0.19%) that are not touch-moves** — 0.40% of esports August — contradicting `README:129`. |

---

## 1. Mapping: token ↔ market ↔ event

### 1.1 How the authors claim it was verified — in code, not prose

| artefact | what it actually is | file:line |
|---|---|---|
| `check1_roundtrip`, `check2_pairing`, `check3_negrisk`, `check4_status`, `check5_indep` | five columns in `tokens.parquet`. **No code in this repo computes them.** `git log --all -S check3_negrisk` finds one commit, `bd12630` (Step H, the loader that *reads* the column). `_manifest.json.built_by_commit = 043d64f` is the Step I *dashboard* commit; every `build_*.py` present at that commit is the copytrade pipeline (`build_closed_positions.py`, `build_traders_table.py`, …). The "Step D mapping gate" exists only in `HANDOVER.md` prose. | `epsilon_data/README.md:100-102`; `_manifest.json`; `git ls-tree 043d64f` |
| `audit_market()` "identity" check | reads the three boolean columns and reports which are `False`; checks `complement_asset_id` exists in `tokens`. **It re-reads the verdict; it does not re-verify anything.** | `audit.py:69-79` |
| `audit_market()` "pair sum≈1" check | `median(|mid_A + mid_B − 1|)` from `load_pair`. **Tautological**: `NO_bid = 1−YES_ask` and `NO_ask = 1−YES_bid` are Polymarket's quoting design, so `mid_NO ≡ 1 − mid_YES`; worst deviation observed 1.1e-16 over 162,041 points (`03_numbers.json → l1_sanity.pair_sum_is_tautological`). Cannot fail. | `audit.py:119-131` |
| `test_load_pair_two_sides_sum_near_one` | same identity, asserted to ±0.05. Cannot fail. | `tests/test_loader.py:84-96` |
| dashboard header "checks · c1=… c5=…" | displays the columns. | `app.py:82-84` |
| `HANDOVER.md:24-26` | *"Every token was identity-verified against Gamma and against the price data itself (round-trip ids, pair-sums-to-1, NegRisk orientation, resolution convergence, an independent snapshot). The mapping is trustworthy."* | prose |

**Measured coverage of that sentence** (from `03_numbers.json → identity.check_coverage`): c1 evaluated on 99.77% of tokens, c2 98.60% (and vacuous), **c3 8.59%** (2,642 tokens, of which **1,900 = 71.9% are False**), **c5 0.38%** (116 tokens). `check4` is resolution convergence, not identity, and 70% of tokens are `near_half` by design.

So before this turn the honest statement was: *the mapping has never been independently verified in any way that is reproducible from this repo.*

### 1.2 Independent checks designed and run

Neither check touches the five columns, `audit_market()`, or any loader logic. `tokens.parquet` is the thing under test; the oracles are live services.

#### M1 — two-oracle mapping check · `check_mapping_oracle.py` · **ran, 19.7 s**
- **Oracles.** CLOB `GET https://clob.polymarket.com/markets/{condition_id}` → `tokens[{token_id, outcome, winner}]`, `market_slug`, `question`, `neg_risk`, `closed` (the venue the feed came from; works for closed markets). Gamma `GET /markets?condition_ids=&closed=true` → `events[0].{slug,id}`.
  *Trap discovered on the way:* Gamma's `condition_ids` filter **silently returns `[]` for closed markets** unless `closed=true` is passed — 27 of my first 30 came back "not found" for that reason (`04_results/` keeps the superseded `check_mapping_gamma.py` run). Anyone re-verifying against Gamma without that flag will conclude the ids are wrong.
- **Compared per token:** `asset_id ∈ CLOB token_ids` **and** `CLOB outcome(asset_id) == outcome_label` (that pair *is* the mapping), plus `market_slug`, `question`, `neg_risk`, `closed`, `event_slug`, `event_id`.
- **Sample A — random:** 30 politics + 30 esports, seed 7, `identity_status='resolved'`.
- **Sample B — adversarial:** 20 tokens with `check3_negrisk=False` + 10 with `=True` (all politics — the only universe c3 was evaluated on).

| | asset+outcome | slug | question | neg_risk | closed | event_slug | event_id | CLOB winner == our `resolved_outcome` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A random, n=60 | **60/60** | 60/60 | 60/60 | 60/60 | 60/60 | 60/60 | 60/60 | **25/25** |
| B `check3=False`, n=20 | **20/20** | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | (11/11 across B) |
| B `check3=True`, n=10 | **10/10** | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | |

- **Expected:** 100% on asset+outcome; anything less is a price series on the wrong token.
- **Result:** 90/90 clean. **`check3=False` tokens map exactly as well as `check3=True` tokens.** Whatever `check3` measured, it was not the mapping. That downgrades the biggest finding in `03_dataset_overview.html §3` from "possible broken mapping" to "broken or mis-defined check, definition lost."
- **What a failure would have meant:** a single `asset+outcome` miss = at least one token's entire L1/trades series attributed to the wrong side or wrong market; every downstream number touching that token wrong; the 71.9% `check3` failure rate would have been the true error rate.

#### M2 — resolution convergence against the oracle's winner · `check_resolution_convergence.py` · **ran, 8.5 s**
- **Oracle:** CLOB `tokens[].winner` (independent of our `resolved_outcome`). **Tape:** direct DuckDB read of `l1` (not the loader): median of the last 20 touch-moves for the winner token and the loser token.
- **Samples:** 10 markets with `check4_status='converged_correct'` (the tape carries the signal) + 10 random resolved markets (descriptive — 70% of the dataset is `near_half`).
- **Mapping-error signature:** winner's series ends near 0 and loser's near 1.

| sample | `resolved_outcome` == CLOB winner | winner token present | winner ends > 0.9 | loser ends < 0.1 | **inverted** |
|---|---:|---:|---:|---:|---:|
| converged_correct, n=10 | **10/10** | 10/10 | 8/10 | 8/10 | **0** |
| random resolved, n=10 | **10/10** | 10/10 | 8/10 | 8/10 | **0** |

The four non-converging: three are `near_half` stale books (e.g. BESTIA Academy ends 0.505) — the documented phenomenon; one (`Iron Wing`, `converged_correct`) converged only in its last four touch-moves (`last_mid` 0.9995, previous rows ~0.51), which tells you `check4` is a function of the single last row, not of a window. Not a mapping issue.

- **What a failure would have meant:** an inverted pair = the price series is attached to the complement token — the same catastrophe as M1, seen through the price data instead of the id.

### 1.3 Mapping verdict
The mapping is **sound on 110 independently-checked tokens/markets across both universes and both oracles**, including every category the authors' own columns flag as failed. The authors' evidence for it, however, is unreproducible (no build code), partly vacuous (c2 / pair-sum), partly absent (c5 on 0.38%), and partly contradictory (c3). Recommend: keep the mapping, **delete or redefine `check3_negrisk`** and stop citing `check2` and the audit pair-sum as evidence of anything.

---

## 2. Reconciliation: raw capture → built library

### 2.1 How the authors claim it was verified — in code

| artefact | what it actually is | file:line |
|---|---|---|
| `reconciliation()` | `sum(tokens.n_l1_events) == COUNT(*) l1` and `sum(tokens.n_trades) == COUNT(*) trades`, computed live | `catalog.py:96-112` |
| `_manifest.json.reconciliation` | the same two identities, frozen at build time | `_manifest.json` |
| Reconciliation panel | displays `reconciliation()` | `panels/reconciliation.py:8-16` |
| `test_reconciliation_holds` | asserts `reconciliation()` matches | `tests/test_loader.py:105-107` |
| `daemon_counter_correction` | a sentence: "True archive: 3,832,521,222 price_change rows → l1 101,051,502 (2.64%); trades 7,227,528" | `_manifest.json` |
| `e2_counts.csv` | per (universe, month): `l1_raw`, `l1_kept`, `trades`, `l1_parts`. **Undocumented** (not in the README file table, not in the manifest) | `data/research_v1/e2_counts.csv` |

**The shipped reconciliation is circular — proven.** `tokens.n_l1_events == obs_stats.n_l1_events` for all 30,732 assets, and `obs_stats.n_l1_events == COUNT(*) FROM l1 GROUP BY asset_id` for all 30,732 (`03_numbers.json` this turn; query in transcript). `reconciliation()` therefore verifies that *a column derived from `l1` equals a count of `l1`*. It detects post-build drift between `tokens.parquet` and `l1/`. It **cannot** detect anything dropped, duplicated, or mis-deduped before `l1` was written. The only raw→library claims are the manifest sentence and `e2_counts.csv`, and there is no code behind either.

The **dedup rule itself is nowhere documented.** `README:128-130` says "A row exists only where best_bid/best_ask changed" — but changed relative to what, ordered how, and with what state at partition boundaries is unstated. I had to discover it.

### 2.2 Independent checks designed and run

#### R1 — raw hourly shard → library, l1 and trades · `check_reconcile_raw_hour.py <date> <universe> <hh>` · **ran ×5, 0.4 s each from cache (+1–21 s first download)**
- **Raw side:** one hourly `price_change_{uni}_{HH}.parquet` (+ its `bba`/`trades` sidecars) via a fresh `boto3 get_object` (read-only). Raw schema turns out to carry `best_bid`/`best_ask` on every fanned-out row, plus `received_at` and `market` (which the docs correctly say the library dropped).
- **Rule tested:** keep a raw row iff `(best_bid, best_ask)` differs from the same asset's previous row, ordered `(timestamp_ms, received_ns)`. Two variants: first row of the shard counted / not counted (the shard can't see the previous hour's state).
- **Library side:** direct DuckDB read of `l1/` and `trades/` in the shard's own `[min, max] timestamp_ms` window. **Not the loader.**
- **Trades:** row count **and** set equality on `transaction_hash|asset_id|size`.

| shard | raw pc rows | rule: touch changed (excl. first row) | library l1 | assets exact / total | trades raw = lib | tx-key sets identical | keep ratio this hour | universe-month baseline |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 2026-08-10 politics h03 | 572,158 | 4,756 | **4,756** | **158/158** | 321 = 321 | ✓ | 0.86% | 1.16% |
| 2026-07-06 politics h14 | 1,093,656 | 8,120 | 8,124 | 274/278 (rest ±1) | 929 = 929 | ✓ | 0.77% | 1.20% |
| 2026-06-20 politics h20 | 2,564,148 | 5,246 | 5,250 | 460/464 (rest ±1) | 2,832 = 2,832 | ✓ | 0.22% | 0.46% |
| 2026-07-15 esports h10 | 2,955,886 | 70,536 | 70,588 | 230/282 (rest ±1) | 7,518 = 7,518 | ✓ | 2.40% | 3.66% |
| 2026-08-17 esports h12 | 1,645,342 | 76,922 (77,220 incl. first row) | **77,220** | **298/298** (first-row variant) | 3,713 = 3,713 | ✓ | 4.69% | 3.53% |

- **Expected:** every asset within ±1 of the rule (the ±1 is the hour boundary); trades exact with identical hash sets.
- **Result:** all five within tolerance; **15,313 trades, zero missing, zero extra.** `l1` is *not* the `bba` table (5,274 bba rows vs 4,756 l1 rows in h03). The rule above is the rule.
- **On the "2.64%":** it is a blend, not a per-universe truth. From `e2_counts.csv`, politics keeps **1.00%** overall (0.46/1.20/1.16% by month) and esports **3.56%** (3.29/3.66/3.53%); the sampled hours land in their own universe's band, with the hour-to-hour spread you'd expect. The sample supports the ledger; it does not support quoting 2.64% for either universe alone.
- **What a failure would have meant:** a missing tx-key = a silently dropped trade; a systematic per-asset shortfall = a dedup that drops real touch-moves; a surplus = duplicated rows.

The 08-17 row is the anomaly that led to R2.

#### R2 — l1 dedup integrity, library-wide · `check_l1_dedup_integrity.py` · **ran, 13 s**
Direct scan of every `l1` partition: count rows whose `(best_bid, best_ask)` equals the same asset's previous row. Under `README:129` this must be **zero**.

| universe | month | rows | **non-touch-move rows** | share | at a part-file boundary | parts |
|---|---|---:|---:|---:|---:|---:|
| politics | 06 / 07 / 08 | 1.6M / 8.4M / 3.8M | 8 / 10 / 0 | 0.000% | 0 | 1 each |
| esports | 2026-06 | 10,611,146 | 1,414 | 0.013% | 1,264 | 12 (daily) |
| esports | 2026-07 | 44,684,318 | 65,924 | 0.148% | 65,732 | 283 (daily → hourly) |
| esports | 2026-08 | 31,920,440 | **128,628** | **0.403%** | 128,628 | **481 (hourly)** |
| **total** | | 101,051,502 | **195,984** | **0.194%** | 195,624 | |

- **Mechanism (confirmed on 08-17 h12):** `l1` for esports is cut into hourly `part-YYYY-MM-DD-HH.parquet` files from late July; the dedup's "previous row" state **resets per part**, so every asset's first row in every part is kept whether or not the touch moved. 274 of the 298 assets in that hour had a quote in the previous hour; all 298 first rows were kept. Politics is one part per month and shows 18 such rows in 14M.
- **What it means:** `l1` is *almost* the touch-move tape the docs describe. `n_l1_events` for esports tokens is inflated by roughly (#hourly parts the token was quoted in); any "touch-move rate" or "quotes per hour" statistic on esports Jul/Aug is biased up by 0.15–0.4%, concentrated at :00 of every hour; the anti-drift and audit tools would never notice because they compare the library to itself. Not a data-loss problem — an *addition* — but it is the kind of undocumented artefact that becomes a "finding" in someone's notebook.
- **Fix:** carry LAG state across parts in the build, or drop rows where `(bb,ba)` equals the previous row post hoc (195,984 rows).

#### R3 — `e2_counts.csv` ledger vs library · `check_e2_counts_vs_library.py` · **ran, 0.6 s**
All six (universe, month) rows match the library exactly on both `l1_kept` and `trades`. The ledger's library side is verified; its `l1_raw` side (3,832,521,222) is the only number in the reconciliation chain that still rests on the authors' word, and verifying it needs the 71 GB archive (§4).

### 2.3 Reconciliation verdict
**Nothing dropped on any sampled hour**, trades provably complete there, and the ledger is internally consistent. The authors' reconciliation, however, proves nothing about raw→library (it is circular), the dedup rule is undocumented, and the build injects 0.19% non-touch-move rows at part boundaries. Recommend: document the rule; add R1 (five random shards) and R2 to the build as gates; fix the part-boundary reset.

---

## 3. What the anti-drift test does and does not prove

`tests/test_loader.py::test_anti_drift_l1` (`:45-61`), which `CONTRIBUTING.md:68-71` calls the thing that "proves the loader returns exactly what a raw parquet read returns" and without which "nothing else would catch" divergence.

**What it does:**
- Picks **5 tokens** — the 4 busiest by `n_l1_events` plus the single smallest (`_sample_assets`, `:21-26`). Deterministic; the same 5 every run.
- For each, reads the **library** `l1` parquet directly with DuckDB (`_raw_l1`, `:29-38`) and compares to `ed.load_l1()`: row count; first and last row on `timestamp_ms, received_ns, best_bid, best_ask, mid`; an MD5 over the `mid` column only.

**What it proves:** that `load_l1()` returns the same rows, in the same order, as a plain read of the same file, for those 5 tokens, on the `mid` column. That the loader is a faithful pass-through. Nothing more.

**What it does not prove:**
- Anything about `best_bid`, `best_ask`, `spread_c`, `asset_id`, or `received_ns` beyond the first and last row.
- Anything about **`load_trades`** — there is no trades anti-drift test at all.
- Anything about the other 30,767 tokens.
- Anything about whether the **parquet itself is right** — the "raw" side of the test is the library, not the capture. It cannot see dropped trades, a wrong dedup rule, part-boundary duplicates (R2), or a mapping error. It is a loader test mislabelled as a data test.

**Widened version, run this turn** (`check_anti_drift_full.py`, 5.7 s): 40 seeded-random tokens (20/20 by universe), **every column**, `l1` **and** `trades` → 40/40 and 40/40. So the property the test wants to establish does hold broadly; the shipped test just under-covers it. Cheap upgrade: sample randomly, hash every column, add trades.

Other things in the file worth knowing: `test_load_pair_two_sides_sum_near_one` (`:84-96`) asserts an identity; `test_audit_market_structure:117` is a dead line (`ed.to_dict()` on the module); `test_write_exclusion_roundtrip` (`:131-148`) ignores its `tmp_path` and mutates the **real** `exclusions.csv`, restored in `finally`; `test_search_finds_fed` (`:99-102`) tests `"fed"` while every doc tells the user to type `"fed july"`, which returns nothing (`02_dashboard_walkthrough.md §1`). The shipped suite: 10 passed in 2.7 s.

---

## 4. The plan — checks ranked by damage-if-failed

All scripts in `scripts/audit_checks/`, run from `polymarket/research` with `EPSILON_DATA_ROOT` exported and `PYTHONPATH=.`. Run ✓ = executed this turn, output in `04_results/`.

| # | check | script | if it fails, it means | expected | runtime | result |
|---|---|---|---|---|---|---|
| 1 | **Two-oracle mapping** (asset_id ↔ CLOB token/outcome; Gamma event) on random + `check3`-stratified tokens | `check_mapping_oracle.py random 30 7` / `check3 20 10` | price series attached to the wrong token/market/event; everything downstream on that token wrong | 100% on asset+outcome | 20 s | ✓ **90/90** |
| 2 | **Resolution convergence** vs CLOB `winner` | `check_resolution_convergence.py 11` | inverted pair = series on the complement token | 0 inverted; `resolved_outcome` == winner | 9 s | ✓ **0 inverted, 20/20** |
| 3 | **Raw→library trades**, hash-set equality per hourly shard | `check_reconcile_raw_hour.py D U HH` | silently dropped/duplicated trades | sets identical | 0.4 s cached / ≤21 s cold | ✓ **5/5 shards, 15,313 trades** |
| 4 | **Raw→library l1** under the touch-change rule per shard | same script | dedup drops real moves, or rule ≠ documented | every asset within ±1 | same | ✓ **5/5 within ±1** |
| 5 | **l1 dedup integrity**, library-wide | `check_l1_dedup_integrity.py` | rows that aren't touch-moves → biased rates, inflated `n_l1_events` | 0 | 13 s | ✗ **195,984 (0.19%)** — part-boundary reset |
| 6 | **e2 ledger vs library** | `check_e2_counts_vs_library.py` | the shipped ledger doesn't describe the shipped data | 6/6 exact | 0.6 s | ✓ |
| 7 | **Anti-drift, all columns, l1 + trades, random tokens** | `check_anti_drift_full.py 40 3` | loader alters data | 40/40 both | 6 s | ✓ |
| 8 | Full CLOB sweep of all 15,386 markets (item 1 at population scale) | `check_mapping_oracle.py` with a `--all` mode (not written) | as #1, but finds the rare miss | 100% | ~30–60 min at 0.15 s/req; rate-limit risk | **not run** — exceeds the 10-min budget; recommended before any external publication |
| 9 | Raw trades total = 7,227,528 | new script: sum rows over all 3,072 `trades_*` shards | drops outside the 5 sampled hours | exact | 3,072 `get_object` (~450 MB) | **not run** — beyond "a handful" of reads; cheap in bytes, expensive in calls; do it once, in the build |
| 10 | Raw `price_change` total = 3,832,521,222 | scan the archive | `l1_raw` in the ledger is wrong | exact | 71 GB | **not run** — only sensible as a build-time gate |
| 11 | Recover `check3_negrisk`'s definition | ask Alvaro; nothing in repo | unknowable which of the 1,900 are informative | — | — | **blocked on the author** |
| 12 | NegRisk basket membership: every `event_id`'s markets in `tokens` == Gamma event's markets | new script | missing legs explain sums < 1 (and the docs already say this) | list of missing legs per event | ~5 min (320 events) | **not run** — proposed; would turn the "missing candidates" caveat into a number |

---

## 5. Is Alvaro's brief complete?

`01_digest.md §7` lists the **20 gaps the authors admit**. Below is everything found in code or data across all four audit turns that the docs do **not** mention. Bold = would change a conclusion someone draws from this data.

| # | found, not admitted | where established |
|---|---|---|
| 1 | **The code that built `tokens.parquet` and its five checks is not in the repo** at any commit; `built_by_commit` points at the dashboard commit | this doc §1.1 |
| 2 | **`check3_negrisk` is False on 71.9% of the tokens it covers and has no definition**; the mapping is fine regardless (M1-B) | `03 §3`; this doc §1.2 |
| 3 | **`check5_indep` covers 116 tokens (0.38%)**; `check3` covers 8.59% — "every token was identity-verified" is not true of those two | `03 §3` |
| 4 | **`check2_pairing`, the audit pair-sum, and `test_load_pair…` are an algebraic identity** and cannot fail | `03 §4`; this doc §1.1 |
| 5 | **The shipped reconciliation is circular** (`n_l1_events` is `COUNT(l1)` by construction) | this doc §2.1 |
| 6 | **The l1 dedup rule is undocumented**; it is "(bb,ba) differs from previous row of the same asset" | this doc §2.2 R1 |
| 7 | **195,984 l1 rows are not touch-moves** (0.19%; esports Aug 0.40%) from the dedup resetting per part file — `README:129` is false at the margin; `n_l1_events` inflated for esports | this doc §2.2 R2 |
| 8 | **`l1` partitioning is inconsistent** — monthly for politics, daily for esports Jun/early-Jul, hourly from late Jul — and undocumented | this doc §2.2 |
| 9 | **`negrisk_sum()` is a 20-hour-stale composite**, `n_live` counts ever-quoted legs, `.attrs['note']` states the wrong direction | `01 §8.1` |
| 10 | **HANDOVER's replacement NegRisk figure (~2.35) doesn't reproduce** (Elon events 2.0–4.5); 147 of the 320 NegRisk events are esports, which the NegRisk panel text says don't exist | `01 §9`; `02 §9`; `03 §6` |
| 11 | **The outage is 14:03→08:29 (18h26m), not 15:00→08:29**; two files also say 08:00 | `01 §8.3` |
| 12 | **Gamma's `condition_ids` filter hides closed markets** unless `closed=true` — anyone re-verifying the mapping the obvious way gets 90% "not found" | this doc §1.2 M1 |
| 13 | **`audit_market()` has no plausibility check** — passes a 579,445-buy / 8-sell market at 0.1¢ as "looks fine" | `02 §6` |
| 14 | **`search("fed july")` returns nothing**; it is the documented example in four places | `02 §1` |
| 15 | `check_setup.py` and the loader never read `.env`; docs say to use one | `01 §8.4` |
| 16 | 24 crossed-book rows (spread to −65.9¢) — undocumented; assume-non-negative code trips | `02 §3` |
| 17 | `METRICS_REFERENCE.md` and `RESEARCH_FINDINGS.md` describe a different dataset, one the repo's own canon audit condemns, with no banner | `01 §8.2` |
| 18 | `brain/handoff/reports/` (the build history) does not exist | `01 §8.9` |
| 19 | `pc_file_manifest.txt` does not drive `coverage()` (nothing reads it); `e2_counts.csv` — the only raw ledger — is undocumented; `coverage.parquet` missing from the README file table | `01 §8.6` |
| 20 | The anti-drift test covers 5 tokens, one column, no trades; the property holds but the test doesn't show it | this doc §3 |
| 21 | NegRisk tab ignores the window selector; 25 s render; audit 12–14 s | `02 §8-9` |
| 22 | `events()` returns a NaN event row; `search()` doesn't apply exclusions while `catalog()` does; `first_seen`/`last_seen` are DOUBLE not int; `received_at` and `market` exist in raw (docs right about that) | `01 §8.10`, `§3` |

**Answer: no, the brief is not complete.** Of the 22 items above, the authors' 20-item gap list covers none. Items 1–8 are the ones that matter for trusting the dataset, and they cut both ways: **the data is better than its documentation** (mapping and completeness both survived independent checks the authors never ran), and **the documentation's evidence is worse than it reads** (three of the five identity checks are unreproducible, vacuous, or 0.38%-coverage, and the reconciliation checks the library against itself).

What I would ask Alvaro for, in order: the build code (or a statement that it's gone); the definition of `check3_negrisk`; whether the part-boundary reset is known; and permission to run #8 and #9 once inside the build so the two "not run" rows become gates rather than trust.

---

## Appendix — scripts and outputs

```
polymarket/research/scripts/audit_checks/
  check_mapping_oracle.py          M1  (random | check3 modes)       -> 04_results/mapping_oracle.txt
  check_mapping_gamma.py           superseded: Gamma-only; documents the closed-market trap
  check_resolution_convergence.py  M2                                -> 04_results/resolution_convergence.txt
  check_reconcile_raw_hour.py      R1  (per shard; caches raw in scratchpad, GetObject only)
                                                                     -> 04_results/reconcile_raw_hours.txt
  check_l1_dedup_integrity.py      R2                                -> 04_results/l1_dedup_integrity.txt
  check_e2_counts_vs_library.py    R3                                -> 04_results/e2_counts_vs_library.txt
  check_anti_drift_full.py         §3                                -> 04_results/anti_drift.txt
```
All excluded from git via `.git/info/exclude` (local only; nothing staged). Raw shards cached under the session scratchpad, not the repo.
