# `scripts/audit_checks/` — independent verification scripts

One-off checks written for the **research_v1 audit, 2026-09-02** (repo `alvaro` @ `9810c2b`).
They exist to verify `data/research_v1` **without** using the library's own evidence: none of them
call `audit_market()`, read the five `check*` columns as truth, or (except where noted) go through
`epsilon_data`'s loaders. The oracles are live Polymarket services and direct DuckDB reads of the
parquet.

**READ-ONLY against R2.** `check_reconcile_raw_hour.py` is the only script that touches the archive
bucket, and only via `GetObject`; every other script reads the locally-fetched `research_v1` copy.

Full write-up: [`../../docs/audit_2026-09/`](../../docs/audit_2026-09/) — `04_audit_plan.md` is the
plan and results table; `04_results/*.txt` are these scripts' raw stdout.

## How to run

From `polymarket/research/`, with the data present and credentials available:

```bash
EPSILON_DATA_ROOT=data/research_v1 PYTHONPATH=. uv run python scripts/audit_checks/<script>.py [args]
```

(`EPSILON_DATA_ROOT` can now come from `polymarket/research/.env` — see `epsilon_data/README.md`.)
`check_reconcile_raw_hour.py` caches downloaded raw shards in `$AUDIT_RAW_CACHE`, defaulting to a
system temp dir — never inside the repo.

## The scripts

| script | backs | what it checks | expected | runtime |
|---|---|---|---|---|
| `check_mapping_oracle.py [random N seed \| check3 n_false n_true]` | plan §1.2 M1, check #1 | `asset_id` ↔ CLOB `token_id`/`outcome` (**the mapping**) plus slug, question, `neg_risk`, `closed`, and Gamma `event_slug`/`event_id`, on a seeded-random sample and on a `check3_negrisk`-stratified sample | 100% on asset+outcome | ~20 s |
| `check_mapping_gamma.py [n seed]` | plan §1.2 M1 (superseded) | Gamma-only ancestor of the above. Kept because it documents the trap: Gamma's `condition_ids` filter silently returns `[]` for **closed** markets unless `closed=true` is passed | — | ~20 s |
| `check_resolution_convergence.py [seed]` | plan §1.2 M2, check #2 | does each token's tape end where the CLOB `winner` says it should? An inverted pair means the series is attached to the complement token | 0 inverted | ~9 s |
| `check_reconcile_raw_hour.py DATE UNIVERSE HH` | plan §2.2 R1, checks #3 and #4 | one raw hourly shard → library: `l1` under the touch-change rule (per asset, ±1 for the hour boundary) and `trades` by row count **and** `transaction_hash` set equality | every asset ±1; trade sets identical | 0.4 s cached, ≤21 s cold |
| `check_l1_dedup_integrity.py` | plan §2.2 R2, check #5 | library-wide: rows whose `(best_bid, best_ask)` equals the same asset's previous row. Under `epsilon_data/README.md` this must be 0 | 0 | ~13 s |
| `check_e2_counts_vs_library.py` | plan §2.2 R3, check #6 | `e2_counts.csv` ledger vs actual per-partition `l1`/`trades` counts | 6/6 exact | ~1 s |
| `check_anti_drift_full.py [n seed]` | plan §3, check #7 | the widened anti-drift property the shipped test under-covers: seeded-random tokens, **every** column, `l1` **and** `trades` | n/n both | ~6 s |

## Known results (2026-09-02)

All pass except `check_l1_dedup_integrity.py`, which finds **195,984 rows (0.19%)** that are not
touch-moves — the dedup's "previous row" state resets at part-file boundaries. See plan §2.2 R2.
