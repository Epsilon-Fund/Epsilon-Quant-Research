---
title: "Storage Consolidation Audit - 2026-06-05"
created: 2026-06-05
status: active
owner: justin
project: polymarket
para: project
hubs:
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - research
  - data-quality
---
# Storage Consolidation Audit - 2026-06-05

> Hub: [[polymarket_data_manifest]] · [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

- This note records the repo disk audit for the 60G `./.tmp` spill, 109G Polymarket data body, 8.7G crypto momentum folder, and 76G `.git` object store.
- Safe cleanup has been applied: stale DuckDB temp spill, Git temp-pack garbage, and zero-byte data/log files were removed.
- Same-path Parquet recompression has been applied where it was strictly smaller; rewritten files remain directly readable by DuckDB/PyArrow and do not require a decompress-to-use workflow.
- The remaining durable consolidation path is a hot/cold dataset layout: keep canonical raw shards, avoid blind rewrites, compress or convert live JSONL only after reader support, and introduce integer-keyed hot marts only with stable views/docs.

## Observed Footprint

| area | size | read |
|---|---:|---|
| repo total | 256G | Working tree plus `.git`. |
| `./.tmp` | 60G | DuckDB temp spill files from 2026-05-31, plus tiny graph previews. |
| `.git` | 76G | 27G real pack files plus about 46G `tmp_pack_*` garbage reported by `git count-objects -vH`. |
| `polymarket/research/data` | 109G | Main Polymarket datasets. |
| `topics/momentum` | 8.7G | Mostly BB CPCV notebooks and pickle outputs. |
| `live_trading` | 15M | Not material for this cleanup. |

Filesystem state during audit: 926Gi disk, 824Gi used, 72Gi available.

## Completed Cleanup

Completed on 2026-06-05:

| action | before | after | result |
|---|---:|---:|---|
| Delete `./.tmp/duckdb_temp_storage_*.tmp` | 60G | 0 files | Removed abandoned DuckDB spill. `./.tmp` is now about 1.4M. |
| Delete `.git/objects/pack/tmp_pack_*` | 46G | 0 files | `git count-objects -vH` now reports `garbage: 0`, `size-garbage: 0 bytes`. |
| Delete zero-byte data/log files | 3 files | 0 files | Removed `dali_market_fill_stats.parquet` and two empty logs. |
| Recompress smaller SNAPPY Parquets to ZSTD | 159 candidates / 5.05GiB | 89 rewritten, 70 kept | Saved 1.535GiB with row-count/schema validation and same filenames. |

Post-cleanup footprint:

| area | size | note |
|---|---:|---|
| repo total | 149G | Down from 256G. |
| `./.tmp` | 1.4M | Only tiny graph-preview/zip artifacts remain. |
| `.git` | 30G | 27.39GiB real pack plus 2.60GiB loose objects; no Git garbage temp packs remain. |
| `polymarket/research/data` | 108G | Down from 109G after Parquet recompression. |
| `topics/momentum` | 8.7G | Unchanged; still dominated by BB CPCV notebooks/pickles. |

Current Parquet codec state under `polymarket/research/data`: 350 ZSTD files / 89.116GiB, 70 SNAPPY files / 0.372GiB, 0 bad Parquets.

## Immediate Cleanup Candidates

These are cleanup candidates, not source-data consolidation. They should only be executed after explicit approval and a quick no-active-process check.

| candidate | size | status | recommended action |
|---|---:|---|---|
| `./.tmp/duckdb_temp_storage_*.tmp` | about 60G | Looks like abandoned DuckDB spill from 2026-05-31. No DuckDB/Python process was running. The Codex app VM had read handles during the audit. | Delete only after confirming no active DuckDB/Codex data job needs it. Expected savings: about 60G. |
| `.git/objects/pack/tmp_pack_*` | about 46G | `git count-objects -vH` reports 47 garbage files, `size-garbage: 45.97 GiB`. | Remove via a cautious Git cleanup when no git process is running, then run a connectivity check. Expected savings: about 46G. |
| zero-byte data/log files | negligible | Found `dali_market_fill_stats.parquet` and two empty log files. | Delete only as hygiene; not meaningful for disk pressure. |

## Data Artifact Breakdown

Across Polymarket data, crypto momentum, and live trading artifacts:

| type | count | size | note |
|---|---:|---:|---|
| Parquet | 456 | 91.04G | Main storage mass. Mostly already ZSTD. |
| JSONL | 139 | 9.89G | Mostly live CLOB capture logs and checkpoints. |
| Pickle | 116 | 6.16G | Mostly crypto CPCV outputs. |
| XZ | 1 | 5.81G | Raw `orderFilled_complete.csv.xz`. |
| DuckDB | 1 | 1.26G | Copytrade relayer scratch DB. |
| CSV | 232 | 1.12G | Findings/report support tables. |
| Zip | 282 | 0.24G | Small enough to ignore for this cleanup. |

## What Not To Consolidate Blindly

- `polymarket/research/data/trades/trades_seed.parquet` and `trades_delta_shard*.parquet` are timestamp-contiguous, not obvious duplicates. The seed ends at 2025-10-07 and gap/delta/polygon shards continue forward.
- The trade shards are already ZSTD and append-oriented. Repartitioning them may improve query ergonomics, but should not be expected to save tens of GB by itself.
- `closed_positions.parquet` is intentionally materialized. Existing docs note it costs about 27G but avoids recomputing a 270M-row resolved-position table for downstream work.
- Do not blindly dictionary-rewrite `closed_positions.parquet`: an 80-row-group sample grew by 5.6% despite dictionary encoding the string columns, so same-path dictionary rewrite is currently rejected.
- Do not compress live CLOB JSONL in place until readers are patched. Many replay/audit scripts glob `*.jsonl` and call `Path.open()` directly; renaming to `.jsonl.zst` or `.jsonl.gz` would make old scripts silently miss data.

## Real Consolidation Levers

1. **Hot/cold Polymarket marts.** Keep full `closed_positions.parquet` and `bankroll_timeseries.parquet` as cold canonical tables, but add smaller hot marts for common analysis paths. Then update scripts to read hot marts by default and full tables only when needed.
2. **Integer-key dimensions.** `bankroll_timeseries.parquet` spends about 8.59G on repeated `address` strings. `closed_positions.parquet` spends about 5.93G on `outcome_token_id`, 4.10G on `condition_id`, and 3.94G on `address`. Replacing repeated strings with `address_id`, `condition_id`, and `outcome_token_id` dimension joins could save meaningful space while keeping readable views in DuckDB.
3. **Compress or convert live JSONL.** Live CLOB JSONL is about 7.9G under `data/live_clob`, plus about 1.9G JSONL under `data/analysis`. Convert raw capture logs to compressed JSONL/ZSTD or Parquet only after adding shared reader support for `.jsonl`, `.jsonl.gz`/`.jsonl.zst`, and/or Parquet sidecars.
4. **Recompress SNAPPY analysis Parquets.** Mostly done. Remaining SNAPPY files were kept because ZSTD was not smaller or the file was too small to matter.
5. **Crypto CPCV cold storage.** `topics/momentum/strategies/bb_cpcv` is about 7.9G, dominated by a 3.7G `portfolio_cpcv_paths.pkl`, duplicated OOS/WF pickles, and large notebooks with embedded outputs. Strip notebook outputs only after preserving useful galleries/results notes; cold-archive or regenerate bulky pickles.
6. **Raw seed policy.** `orderFilled_complete.csv.xz` is 5.81G. Keep it if local reproducibility matters; otherwise treat it as a cold raw payload because the query surface is `trades_seed.parquet` plus delta shards.

## Safe Execution Order

1. Done: free non-source space first via DuckDB temp cleanup and Git garbage cleanup.
2. Done: re-run disk audit and record before/after sizes.
3. Done where safe: same-path Parquet recompression for files that became smaller under ZSTD.
4. Next: add shared reader support before compressing/converting JSONL, so old scripts do not silently miss archived captures.
5. Next: prototype hot marts for `closed_positions` and `bankroll_timeseries` using integer-key dimensions and DuckDB views.
6. Then: update readers/scripts to target shared views, not one-off direct file paths.
7. Only after validation, move full cold tables or bulky crypto pickles out of the hot working set.
