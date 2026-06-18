---
title: "Crypto Data And Artifact Manifest"
created: 2026-06-05
status: closed
owner: justin
project: crypto
para: resource
hubs:
  - STRATEGY_REFERENCE
tags:
  - crypto
  - research
---
# Crypto Data And Artifact Manifest

> Hub: [[STRATEGY_REFERENCE]] · [[topics/momentum/README|momentum README]]

## Plain-English Summary

- This note documents the non-Markdown data artifacts for the crypto momentum, live-trading, and stat-arb side of the repo.
- Parquet and pickle files are documented by family because they are generated datasets, not narrative notes.
- CSV and JSONL files that support user-facing outputs are linked directly when they are small enough and meaningful enough to inspect.
- Python bytecode caches are not documented here; they are disposable and should stay deleted/ignored.

## Parquet Families

| family | paths | meaning | handling |
|---|---|---|---|
| live trading daily cache | `live_trading/cache/daily/*_daily.parquet` | Binance daily OHLCV cache for dashboard and optimizer reads. | Rebuild/refetch cache, not a research conclusion. |
| live trading hourly cache | `live_trading/cache/hourly/*_hourly.parquet` | Binance hourly OHLCV cache for live dashboards and walk-forward support. | Rebuild/refetch cache. |
| XS momentum universe cache | [[topics/momentum/xs_momentum/universe/cache/close.parquet|XS close cache]], [[topics/momentum/xs_momentum/universe/cache/volume.parquet|XS volume cache]], [[topics/momentum/xs_momentum/universe/cache/meta.parquet|XS metadata cache]] | Cross-sectional momentum universe cache. | Keep with [[topics/momentum/xs_momentum/README|XS momentum README]]. |

## Pickle Families

Pickles are generated strategy artifacts, usually pandas DataFrames or result dictionaries from walk-forward/CPCV notebooks. They are not portable schemas; read them with the notebook/code that created them.

| family | paths | owner |
|---|---|---|
| live momentum CPCV | `topics/momentum/strategies/momentum_cpcv/**/*.pkl` | [[topics/momentum/strategies/momentum_cpcv/README|momentum CPCV README]] |
| BB breakout CPCV | `topics/momentum/strategies/bb_cpcv/**/*.pkl` | [[topics/momentum/strategies/bb_cpcv/README|BB CPCV README]] |
| BB breakout walk-forward | `topics/momentum/strategies/bb_breakout_wf/**/*.pkl` | [[topics/momentum/strategies/bb_breakout_wf/README|BB breakout WF README]] |
| first-generation walk-forward | `topics/momentum/strategies/wf_testing/**/*.pkl`, `topics/momentum/strategies/wf_testing_2/**/*.pkl` | [[topics/momentum/strategies/wf_testing/README|wf_testing README]], [[topics/momentum/strategies/wf_testing_2/README|wf_testing_2 README]] |
| portfolio result pickles | `topics/momentum/results/oos/*.pkl` | [[topics/momentum/results/README|momentum results README]] |
| cross-sectional momentum | `topics/momentum/xs_momentum/oos/*.pkl`, `topics/momentum/strategies/xs_cpcv/oos/*.pkl` | [[topics/momentum/xs_momentum/README|XS momentum README]], [[topics/momentum/strategies/xs_cpcv/README|XS CPCV README]] |
| stat-arb OOS exports | `topics/statistical-arbitrage/strategies/testing/*_oos.pkl` | [[topics/statistical-arbitrage/strategies/testing/README|pairs trading README]] |

## CSV Support Files

- [[topics/momentum/outputs/wf_fold_results.csv|momentum walk-forward fold results]]
- [[output/screening/pairs_screen_v7_0_full.csv|pairs screen v7 full output]]
- [[output/screening/pairs_screen_v7_0_wf_input.csv|pairs screen v7 walk-forward input]]

## JSONL Support Files

- [[journal_logs/execution-2026-05-12.jsonl|execution journal 2026-05-12]]

## Ignored Or Disposable

- `__pycache__/` and `*.pyc` are disposable interpreter caches; do not link them.
- Zip archives are intentionally ignored as raw payload/transport containers.
- Live dashboard JSON state (`trades.json`, `positions.json`, `mae_cache.json`, `signals_cache.json`) is runtime state and is gitignored; document the workflow in [[live_trading/CLAUDE|live trading guide]], not as graph nodes.
