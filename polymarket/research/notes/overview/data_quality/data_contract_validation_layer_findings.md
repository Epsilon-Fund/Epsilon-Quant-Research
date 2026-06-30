---
title: "Executable Data-Contract + Drift Validation Layer (the `data-contract` skill)"
created: 2026-06-29
status: active
owner: justin
project: polymarket
para: resource
hubs:
  - POLYMARKET_BRAIN
  - CODEX
  - SKILL_MAP
tags:
  - research
  - data-quality
  - infrastructure
  - validation
---
# Executable Data-Contract + Drift Validation Layer (the `data-contract` skill)

Hubs: [[POLYMARKET_BRAIN]] · [[CODEX]] · [[SKILL_MAP]] · [[strat_market_making]] · data manifests: [[polymarket_data_manifest]] · [[docs/CRYPTO_DATA_MANIFEST|crypto data manifest]]

## Plain-English Summary

- **What:** a code-first layer that checks a dataset is *trustworthy* before any backtest/research run reads it — schema, the repo's hard data invariants, and distribution **drift** — and **fails closed** (aborts the run) when a contract is broken. Tooling is [pandera](https://pandera.readthedocs.io) on its polars backend; no service, no DB.
- **Why:** the repo's invariants (lookahead-free, append-only parquet, lowercase-0x addresses, monotone timestamps, no dup/missing bars, finite values) were *assumed* by convention. This makes them *enforced as code* at the point a run starts, so a corrupt or mutated shard can't silently poison a result.
- **Where:** instantiated **per project, never cross-imported** — `infrastructure/data/schemas/` (crypto) and `polymarket/research/data_infra/schemas/` (Polymarket). The engine `core.py` is byte-identical in both; the dataset contracts in `contracts.py` differ.
- **Packaged as the auto-invoked `data-contract` skill** (`.agents/skills/data-contract/`, symlinked into `.claude/skills/` and `~/.codex/skills/`), registered in [[SKILL_MAP]], and **wired as a hard gate** into two real entry scripts.
- **Status / takeaway:** built, tested (44 passing tests across the two projects), and verified end-to-end. Known-good production shards pass with ~0 false positives in milliseconds; deliberately broken shards abort the run with a plain-English report. The intellectually load-bearing work was **realism calibration** — only enforcing the invariants each dataset actually has.

## What was built

Three capabilities, one engine, two instantiations:

1. **Schema / contract validators** — column presence, dtype, nullability, and per-column ranges via `pandera.polars`, plus cross-column row rules (e.g. OHLC sanity, `best_ask ≥ best_bid`).
2. **Invariant checks** — finite (no NaN/Inf), lowercase-0x addresses (`^0x[0-9a-f]+$`), monotone timestamps, cadence (no duplicate / sub-step / missing bars), lookahead-free (no future-dated rows; for captured streams, server-ts not ahead of receive-ts), and append-only (sharded parquet never mutates in place; a growing file is a row-superset of its prior self).
3. **Drift** — Population Stability Index (PSI) + two-sample Kolmogorov–Smirnov (KS) of each monitored column vs a stored reference window, appended append-only to `data_monitoring/drift_<dataset>.parquet`.

On failure the engine writes a plain-English markdown report (this note's standard) under `…/schemas/data_monitoring/reports/`.

### Architecture (per-project, never cross-import)

```
infrastructure/data/schemas/                 polymarket/research/data_infra/schemas/
  core.py        ← engine (BYTE-IDENTICAL)     core.py        ← engine (BYTE-IDENTICAL)
  contracts.py   ← crypto OHLCV contracts       contracts.py   ← PM trades / positions / L2
  cli.py         ← `python -m …schemas.cli`      cli.py
  tests/         ← 26 tests                      tests/         ← 18 tests
  data_monitoring/ (gitignored, regenerable)     data_monitoring/ (gitignored, regenerable)
```

`core.py` is duplicated deliberately: the two projects have separate venvs and **must not import each other** ([[CODEX]] invariant). The engine is project-agnostic; only `contracts.py` + the package `__init__.py` path-resolution differ. If you edit `core.py`, edit both copies (a `diff -q` check is in the build log).

## The contracts (what each dataset actually promises)

| dataset | key clauses | calibration note |
|---|---|---|
| `crypto_ohlcv_daily` / `_hourly` | OHLC>0, Volume≥0, High≥Low, High≥max(O,C), Low≤min(O,C); `Time` strictly increasing; cadence 1d/1h; finite; no future bars | **No append-only** — the cache is a rebuildable refetch, not a shard family. Drift on Volume only. |
| `pm_trades` (seed + delta shards) | schema; price∈[0,1]; usd/token≥0; lowercase-0x `condition_id/maker/taker/transaction_hash`; finite; **append-only**; no future fills | **No monotone-ts clause** — fills are not stored strictly time-sorted. |
| `pm_closed_positions` | schema; `resolution_price`∈[0,1]; `first_fill_ts ≤ last_fill_ts`; n_fills≥0; lowercase-0x `address/condition_id`; finite; no future fills | 270M-row single file → row scan capped (see Coverage). |
| `pm_traders` | schema; `pos_win_rate`∈[0,1]; volume≥0; lowercase-0x `address`; finite | one row per wallet. |
| `pm_l2_book/_trades/_price_change/_bba` | schema; price/bid/ask∈[0,1]; size≥0; `best_ask ≥ best_bid` & `spread = ask−bid` (when both sides present); lowercase-0x `market`(+`transaction_hash`); **append-only** hourly capture; ordering on `received_ns`; no future + server-not-ahead-of-receive | ordering is on the monotonic **receive** clock, not `timestamp_ms`. |

### Realism calibration — the part that matters

Gates must be honest in both directions ([[CODEX]] § Realism calibration). Four decisions, each made by inspecting the real parquet, not by assuming:

1. **Polymarket fills are NOT strictly time-ordered.** A sampled delta shard has ≈3,305 / 6.1M micro-scale out-of-order timestamps (66 / 5M even in the seed) — fills land in coarse chronological / block-log order with interleaved markets. A `strict`/`non_decreasing` monotone clause would *fail clean production data*, so `pm_trades` carries **no** monotone clause. Ordering IS enforced where it is real: crypto OHLCV (strict) and L2 capture (`received_ns`, 0 inversions observed).
2. **The crypto OHLCV cache is rebuildable, not append-only.** Imposing an append-only clause on a refetch cache would be the wrong invariant; it is omitted.
3. **Missing bars are a warning, not a kill.** Binance hourly history has rare maintenance gaps (BTCUSDT: 25 missing bars across 12 gaps = 0.047% over 53,696 rows). Duplicate / sub-step / off-grid intervals are hard errors (corruption); clean missing bars under a tolerance (default 1%) are **warnings** — they do not abort a backtest.
4. **Drift never blocks.** PSI/KS are a watch signal, surfaced and logged, but a distribution shift is never a fail-closed gate (statistical significance ≠ a data-integrity break). KS is also hyper-sensitive at large n, so the KS *statistic* (effect size) is reported alongside the p-value.

### Practical example

A backtest as-of bootstrap calls `guard_dataset("pm_trades")`. The engine globs the seed + all 62 delta shards, confirms every shard still exposes the 13 contract columns, checks the append-only manifest (no shard's bytes changed, none vanished), then row-scans the 2 most-recent shards (capped at 3M rows) for lowercase-0x addresses, price∈[0,1], finite values, and no future fills. If someone had hand-edited a shard to "fix" a price, the manifest hash mismatch fires; if a fill carried an uppercase checksummed address, the address check fires; either way the run **aborts before the first backtest cell** with a report naming the offending rows.

## The hard gate (fail-closed) — wired, not just description-matched

Description-matching alone is not a safety gate, so the validator is wired into the bootstrap of two real entry scripts as a guarded, fail-closed call:

- **crypto:** [build_dataset.py](../../../../../infrastructure/ml/features/build_dataset.py) — `guard_dataset("crypto_ohlcv_daily", symbols=present)` at the top of `__main__`, before any feature build. Only symbols whose cache exists are gated (absent optional symbols stay the script's own concern).
- **Polymarket:** [run_stage1.py](../../../scripts/backtest/run_stage1.py) — `guard_dataset("pm_trades")` + `guard_dataset("pm_closed_positions")` at the top of `main()`, before the 72-run matrix.

On any blocking violation the guard raises `DataContractError` and the process exits non-zero. Emergency bypass: `EPSILON_DATA_CONTRACT=warn` (logs, proceeds) or `=off` (skips) — never for a real run. Default is `enforce`.

## Verification (Definition of Done)

| DoD gate | result |
|---|---|
| Negative tests catch schema/dtype/range/monotonicity/append-only/finite/address/lookahead | **PASS** — 26 crypto + 18 PM tests, all green (`pytest`, ~0.6s each suite) |
| Synthetic drift detected | **PASS** — Volume×3 (crypto) and a price-distribution shift (PM) both flag `large`/`ks_significant` |
| ~0 false positives on a known-good shard | **PASS** — all 6 live daily OHLCV symbols, all 4 L2 tables, traders (2.5M), closed_positions clean |
| Runs under a stated time budget | **PASS** — OHLCV ≈0.005–0.08s/symbol; L2 price_change (1.1M) 0.53s; pm_trades (156M across 62 shards) ≈3.8s; closed_positions (270M, lazy tail) ≈1.8s @ ~2GB peak |
| Hard-gate abort fires on a deliberately broken shard | **PASS** — `build_dataset.py --symbol <broken>` exits 1 with `DATA-CONTRACT FAILURE`; PM guard surfaces the exact injected violations |
| Skill loads in Claude Code AND Codex; appears in SKILL_MAP; triggers on intended description | **PASS** — appears in the Claude Code skill list with the trigger description; symlinks resolve in `.claude/skills/` and `~/.codex/skills/`; registered in [[SKILL_MAP]] |

### Coverage honesty (no silent caps)

Huge files are not fully row-scanned at gate time. The engine logs exactly what it did: e.g. for `pm_trades` — *"schema + append-only on all 62 shards; row-level invariants on the 2 most-recent shards, capped at 3,000,000 / 156,548,998 rows (most recent)."* Schema/presence is metadata-only and covers every shard (O(1)); append-only fingerprints every shard. Row-level invariants are bounded for speed, and the coverage line says so.

## Read & decision

- **Read:** the layer does what was asked and is honest about its limits. The known-good pass rate (0 false positives) and the deliberately-broken abort both check out; the realism calibration means it will not cry wolf on production data (the most common reason such gates get disabled). pandera 0.32.1 works cleanly on the repo's bleeding-edge stack (Python 3.14 / pandas 3.0 / polars 1.40) — a real compatibility risk that was verified, not assumed.
- **Decision:** **adopt.** The two wired gates are live; the skill auto-triggers before backtest/research runs.
- **Next steps (cheap, optional):**
  1. Wire `guard_dataset("pm_l2_*")` into the MM Path B analysis entry once that pipeline lands (the L2 contracts are ready and calibrated).
  2. Add a contract for the Goldsky delta-shard *freshness* (last-shard end-ts vs now) — complements append-only with a staleness check (see [[goldsky_incremental_freshness]]).
  3. The append-only disappearance check is directory-scoped so partial / `--paths` validation does not false-flag the rest of a family; revisit if a family ever spans multiple directories.
  4. Drift references are auto-baselined on first run and gitignored; re-baseline deliberately with `set-reference` after an intentional data regeneration.

## How to run

```bash
# Polymarket (from polymarket/research/)
PYTHONPATH=. uv run python -m data_infra.schemas.cli validate pm_trades
PYTHONPATH=. uv run python -m data_infra.schemas.cli list

# crypto (from repo root)
PYTHONPATH=. .venv/bin/python -m infrastructure.data.schemas.cli validate crypto_ohlcv_daily

# tests
PYTHONPATH=. uv run python -m pytest data_infra/schemas/tests -q          # PM (from polymarket/research/)
PYTHONPATH=. .venv/bin/python -m pytest infrastructure/data/schemas/tests -q   # crypto (from root)
```

> Column glossary for this note: **PSI** = Population Stability Index (binned distribution-shift score; `<0.10` stable, `0.10–0.25` moderate, `>0.25` large). **KS** = two-sample Kolmogorov–Smirnov test (statistic = max CDF gap; p-value flags a shift). **append-only** = parquet shards are never edited in place; new data → new shard ([[CODEX]] invariant). **lookahead-free** = no row stamped after the as-of point-in-time. **0x-lowercase** = addresses/hashes match `^0x[0-9a-f]+$`. **coverage cap** = the bounded row count actually scanned on a huge file, always logged.
