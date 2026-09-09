---
title: "Crypto momentum — north star + status (May 2026)"
created: 2026-05-11
archived: 2026-09-09
status: archived — pre-migration vault (May 2026); historical, not current
owner: justin
project: crypto
para: archive
hubs:
  - pre-migration-vault-2026-05
tags:
  - archive
  - pre-migration-vault
  - crypto
---

> **ARCHIVED — PRE-MIGRATION VAULT (May 2026).** Historical record, imported 2026-09-09 from the standalone Cowork vault that predates this repo becoming the brain. Index: [[pre-migration-vault-2026-05]].
> **PARKED (2026-08-25).** The crypto momentum book is live, but this note is a May-2026 snapshot of it, not its current state. The live record is `docs/STRATEGY_REFERENCE.md` and [[TODO]] § Crypto — do not build on this note or quote its numbers as current.

# crypto momentum

Daily/hourly timeframe momentum strategies on a 6-coin perp universe. **Live on Binance via VPS-hosted Streamlit dashboard.** Long-term deployment target: Hyperliquid vault.

## Status (2026-05-11)

**Live trading.** Momentum strategy active, 4 open positions, real money in flight. VPS-hosted Streamlit dashboard at `live_trading/app.py` — three pages (Dashboards, Trade Log, Portfolio). Operational state recap in `research/notes/dashboard-status.md`.

**Testing infrastructure is stable** — walk-forward (`wf_engine.py`) + CPCV (`cpcv_engine.py`, canonical entry point for new research) + portfolio aggregation (`portfolio_metrics.py` bar-level, `cpcv_portfolio.py` path-level). Conventions, metric formulas, and trustworthiness ratings documented in `research/notes/strategy-reference.md`.

**Active research direction:** parameter robustness — 18 free params in current momentum WF is confirmed too many (9/15 returning N/A in plateau, 40–55% perturbation degradation). Parameter derivation (collapse stop multipliers into 2–3 derived params) is the planned fix, deferred during dashboard build.

XS momentum was a parallel thread, likely scrapped (see `notes/xs-momentum-recap.md`).

## Active universe (live)

From `live_trading/dashboards/momentum/config.py` (`ACTIVE_ASSETS`):
`ADAUSDT`, `AVAXUSDT`, `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `XRPUSDT` (6 coins).

**BNB diagnostic still open**: present in `momentum_swing` strategy registry but absent from `ACTIVE_ASSETS`. Resolve.

Wider research universes (BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, MATIC) are case-by-case for backtest validation; BB breakout adds DOT / LINK / MATIC / NEAR; XS used TOP_N=40 by volume.

## Strategies

- **`momentum_swing`** (live) — ETH, SOL, BNB, ADA, XRP. Long-only EMA trend + caution gate + volume filter + ATR-sized risk-budget + trailing stop.
- **`momentum_no_vol`** (live) — AVAX, BTC. Same as `momentum_swing` minus volume filter.
- **`BBBreakout`** (dashboard shell wired, **not yet actively trading**) — 4H setup (two big same-colour candles + BB expansion + MA slope) → 1H pullback entry. Strategy function + `optimise.py` built; awaiting production optimisation run with full `n_trials`.
- **`make_xs_strategy`** (likely scrapped) — cross-sectional L/S factory; residual-Sharpe + rolling-Sharpe variants tested.
- **Stat arb** — empty shell at `live_trading/dashboards/statarb/`, no strategy implementation.

**Deprecated / superseded:** J5 / J6 / J7 EMA+ADX swing variants, Supertrend pullback, EMA crossover. The current `momentum_swing` / `momentum_no_vol` are the descendants.

## Live trading stack

- `live_trading/app.py` — unified Streamlit app, 3 pages (Dashboards, Trade Log, Portfolio)
- `live_trading/dashboards/momentum/` — momentum dashboard, signals, decisions, active positions
- `live_trading/dashboards/bbbreakout/` — BB breakout shell (wired, not trading)
- `live_trading/dashboards/statarb/` — empty shell
- `cache_manager.py` / `data_loader.py` — parquet OHLCV cache (daily, `update_cache.py` cron at 00:05 UTC)
- `positions.json` — FIFO position tracking, keys `{SYMBOL}_{YYYYMMDD}_{seq:03d}`
- `realised_capital.json` — updates on exit only
- Capital snapshot consistency: `size_usd` / `coin_capital` / `capital_total` / `coin_weight` **frozen at entry** in `trades.json`, never recomputed from live config

## Research stack

- `infrastructure/walkforward/wf_engine.py` — walk-forward Optuna TPE engine
- `infrastructure/walkforward/cpcv_engine.py` — CPCV engine (canonical for new strategies)
- `infrastructure/walkforward/cpcv_portfolio.py` — path-level portfolio bootstrap
- `infrastructure/walkforward/xs_strategy.py` — XS strategy factory
- `infrastructure/backtester/engine.py` — single-asset backtester
- `infrastructure/backtester/performance_metrics.py` — metric formulas
- `infrastructure/backtester/portfolio_metrics.py` — bar-level multi-sleeve aggregation
- Per-coin WF notebooks (`momentumETH_wf.ipynb` etc.) at `topics/momentum/strategies/wf_testing/` — **source of production `live_params.json`**
- Per-coin CPCV notebooks at `topics/momentum/strategies/momentum_cpcv/` — canonical new-research path

## Key principles

*(authoritative versions in `notes/strategy-reference.md` §G; lessons in `notes/methodology-lessons.md`)*

- **1-bar forward shift** on signals and regime filters.
- **Never blanket `dropna()`** — `dropna(subset=indicator_cols)` only.
- **`sqrt(periods_per_year)` annualisation** for Sharpe; **Calmar uses annualised return** (CAGR / |max DD|).
- **`_default_score` is a weighted composite**, not Sharpe: 50% Sharpe + 30% Calmar + 20% Return, clipped.
- **≥10–20 Optuna trials per free parameter.** Momentum WF currently uses 18 free params with `n_trials=400` (~22/param — at the lower bound; parameter derivation is the planned fix).
- **Cap iteration cycles at 3–4** before declaring convergence.
- **Infrastructure params ≠ strategy params** — never mix in Optuna.
- **Capital + position state snapshotted at entry** — never recomputed from live config (load-bearing for honest historical P&L).
- **Disk + session state must mutate atomically** — any disk write that changes position state must also update the in-memory session_state copy.
- **Cost convention**: per-leg; round-trip = 2 × cost. Default 0.001.
- Commit infra changes to `justin/wf-updates`, not main.

## Open production issues

1. **Stop loss state staleness on VPS** — fix shipped locally; needs verification end-to-end on live VPS after corrected JSON files are scp'd. Only remaining known production bug.
2. **BB Breakout no production params** — `live_params.json` is test/empty; full `optimise.py` run pending.
3. **VPS security** — running on raw port, no HTTPS, no auth. Anyone who knows the IP can see positions.
4. **Execution-hour P&L** — infrastructure built, toggle exists, never verified against real trades.

## Cross-platform note

Dimitris is on Windows. Use `os.path.join` with commented-out root alternatives for paths.

## Reference docs

- `notes/strategy-reference.md` — full repo audit
- `notes/dashboard-status.md` — operational state of the VPS-deployed live trading dashboard
- `notes/methodology-lessons.md` — durable methodology lessons (append-only)
- `notes/xs-momentum-recap.md` — XS thread recap (likely-scrapped)
