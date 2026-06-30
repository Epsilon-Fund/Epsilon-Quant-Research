---
name: data-contract
description: >
  Validate a dataset's schema, append-only/lookahead invariants, and drift
  before any backtest or research run. Use when about to run a backtest,
  walk-forward/CPCV sweep, replay, or analysis over Polymarket fills/L2 or
  crypto OHLCV parquet — or when data looks wrong (NaN/Inf, duplicate or
  missing bars, non-lowercase 0x addresses, a mutated shard, future-dated
  rows, a distribution that shifted). Pandera-backed, code-first, no service.
---

<!--
Source: epsilon-quant-research @ commit 6f6eca0d5e1d46ec388401b304670ae1e6527a9e
(branch justin). First-party skill (NOT vendored). The engine is core.py,
duplicated byte-identical into each project's schemas package; keep both copies
in sync. Mirrors the efficient-fable / cost-mode packaging layout (symlinks in
.claude/skills and ~/.codex/skills).
-->

# Data Contract

An executable data-contract + drift validation layer. Before any backtest or
research run touches a dataset, validate that it satisfies its contract — and
**fail closed** (abort) if it does not. This is a safety gate, not advice.

## When this triggers

Auto-loads when the task is "run a backtest / walk-forward / CPCV / replay /
analysis" over a known dataset, or when the data itself looks suspect. Run the
contract on the dataset **before proceeding** with the run.

## What it enforces (as code, not assumed)

| invariant | meaning | severity |
|---|---|---|
| schema | column presence, dtype, nullability, per-column ranges (pandera.polars) | error |
| finite | no NaN / Inf in numeric columns | error |
| lowercase 0x | address / hash columns match `^0x[0-9a-f]+$` | error |
| monotone ts | strictly-increasing (bars) / non-decreasing (streams) — only where the dataset is actually ordered | error |
| cadence | duplicate / sub-step / off-grid intervals (corruption) | error |
| cadence gaps | missing bars under a tolerance (e.g. exchange maintenance) | **warn** |
| lookahead-free | no row time-stamped after `as_of` (default now); for streams, server-ts not ahead of receive-ts | error |
| append-only | sharded parquet never mutates in place / a growing file is a row-superset of its prior self | error |
| drift | PSI + two-sample KS vs a stored reference window | **never blocks** (watch signal) |

Realism calibration is built in: invariants are only applied where they are real
(e.g. PM fills are NOT strictly time-sorted, so they carry no monotone clause;
the crypto OHLCV cache is rebuildable, so it carries no append-only clause).
Drift and missing-bar gaps are warnings, never gates.

## How to run it (per project — NEVER cross-import the two)

**Polymarket** (`data_infra.schemas`) — run with uv from `polymarket/research/`:

```bash
cd polymarket/research
PYTHONPATH=. uv run python -m data_infra.schemas.cli list
PYTHONPATH=. uv run python -m data_infra.schemas.cli validate pm_trades
PYTHONPATH=. uv run python -m data_infra.schemas.cli validate pm_l2_bba --report
```
Datasets: `pm_trades`, `pm_closed_positions`, `pm_traders`, `pm_l2_book`,
`pm_l2_trades`, `pm_l2_price_change`, `pm_l2_bba`.

**Crypto** (`infrastructure.data.schemas`) — run from the repo root with the
crypto venv:

```bash
PYTHONPATH=. .venv/bin/python -m infrastructure.data.schemas.cli validate crypto_ohlcv_daily
PYTHONPATH=. .venv/bin/python -m infrastructure.data.schemas.cli validate crypto_ohlcv_hourly --symbols BTCUSDT
```
Datasets: `crypto_ohlcv_daily`, `crypto_ohlcv_hourly`.

A markdown failure report (plain-English, per the CODEX markdown standard) is
written under each project's `…/schemas/data_monitoring/reports/` on failure
(`--report` also writes one on pass). Drift rows append to
`…/data_monitoring/drift_<dataset>.parquet`.

## The hard gate (fail-closed) — in code

The gate is already wired into the bootstrap of the canonical entry scripts:
- crypto: `infrastructure/ml/features/build_dataset.py` → `guard_dataset("crypto_ohlcv_daily", …)`
- Polymarket: `polymarket/research/scripts/backtest/run_stage1.py` → `guard_dataset("pm_trades")` + `guard_dataset("pm_closed_positions")`

To gate a new entry script, call the guard at the top of its bootstrap:

```python
# Polymarket
from data_infra.schemas import guard_dataset
guard_dataset("pm_trades")            # raises DataContractError (aborts) on any violation

# crypto
from infrastructure.data.schemas import guard_dataset
guard_dataset("crypto_ohlcv_daily", symbols=["BTCUSDT", "ETHUSDT"])
```

Escape hatch (never for a real run): `EPSILON_DATA_CONTRACT=warn` logs but does
not abort; `=off` skips entirely. Default is `enforce`.

## Adding a dataset

Edit the project's `contracts.py` (`infrastructure/data/schemas/contracts.py`
or `polymarket/research/data_infra/schemas/contracts.py`): declare a `Contract`
with a `pandera.polars` schema plus the `TimestampRule` / `LookaheadRule` /
`AppendOnlyRule` / `RowRule` / `address_columns` / `drift_columns` that are real
for that dataset, then register it in `CONTRACTS` and add a path spec in the
package `__init__.py`. Calibrate value sets against the real parquet first
(don't impose invariants the instrument lacks). Add negative tests in
`tests/test_data_contract.py`.

## Repo invariants (brain/CODEX.md)

uv only, never bare pip; the two projects have separate venvs and never
cross-import; all metrics lookahead-free; parquet shards append-only; addresses
lowercase 0x; require CI before calling a result positive. The engine lives in
`core.py` (identical copy per project); project contracts live in `contracts.py`.

See `data_contract_validation_layer_findings.md` for design rationale and the
realism-calibration decisions.
