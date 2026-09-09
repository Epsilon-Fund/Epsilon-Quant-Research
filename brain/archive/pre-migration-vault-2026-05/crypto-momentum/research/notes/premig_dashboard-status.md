---
title: "Live trading dashboard status snapshot (2026-05-11, ~80% accurate)"
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

*State recap of the live trading dashboard. Captured 2026-05-11 from the user's dashboard chat. User flagged this is "not fully correct either" — treat as ~80% accurate operational snapshot, cross-check with `strategy-reference.md` for anything code-shaped.*

**Reality flag:** crypto-momentum is **NOT parked**. Live trading on VPS, momentum strategy active, real money in flight. The earlier "parked while polymarket is in flight" framing in cowork was wrong.

---

# Epsilon Fund — Momentum Strategy Status

## 1. Active Research Thread

Walk-forward validation of momentum swing strategies across a diversified crypto universe (BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, MATIC — prioritised by volume and history depth). Primary research notebook is `momentumETH_wf.ipynb` with per-coin variants.

Current research focus is parameter robustness — the walk-forward engine (`wf_engine.py`) has plateau analysis, perturbation testing, and cost stress testing built in. The known issue going into live trading is that 18 free parameters is too many for Optuna TPE to converge meaningfully — the heuristic is 10–20 trials per free parameter, and unused parameters dilute search budget. The planned fix (parameter derivation, collapsing related stops into 2–3 derived params) has not been executed yet — it was deferred when live dashboard build began.

OOS windows are currently 137-day. The plan to switch to 275-day windows for final validation has not happened yet — also deferred.

BB Breakout walk-forward research exists in `topics/bb_breakout/strategies/wf_testing/bb_breakout_wf/` for BTC, ETH, BNB, AVAX, XRP and was used as the source of truth for the live dashboard implementation.

## 2. What's Built and Shipped vs In-Flight

**Shipped and running on VPS:**

*Research infrastructure:*
- `wf_engine.py` — walk-forward optimisation with Optuna TPE, plateau analysis, perturbation test, cost stress test
- `wf_visualizer.py` — Plotly visualisation suite including plateau and portfolio OOS charts
- `engine.py` — custom backtester
- `portfolio.py` — dynamic portfolio module auto-discovering `*_oos.pkl` files

*Live trading infrastructure:*
- Single Streamlit app (`live_trading/app.py`) running on VPS with three pages: Dashboards, Trade Log, Portfolio
- Momentum dashboard fully operational: signals, decisions, active positions with live WebSocket prices, trade logging forms, stop loss confirmation workflow
- BB Breakout dashboard shell built, strategy function and `optimise.py` implemented, wired into unified app
- Shared infrastructure: `cache_manager.py` (parquet OHLCV cache), `data_loader.py`, `charts.py`, `trade_log_components.py`, `portfolio_components.py`, `styles.py`
- FIFO position tracking (`positions.json` keyed by `{SYMBOL}_{YYYYMMDD}_{seq:03d}`)
- Capital snapshot consistency (`size_usd` frozen at entry, never recomputed from live config)
- Realised capital tracking (`realised_capital.json` updates on exit only)
- Live WebSocket price streaming with REST fallback, 5-second fragment refresh
- Session state caching for signal computation (`run_dashboard` once per session)
- Daily OHLCV cache via parquet files, `update_cache.py` cron job at 00:05 UTC

**In-flight / partially built:**
- BB Breakout live trading: strategy function and `optimise.py` done, dashboard shell wired in, but not yet actively trading — no live positions, no confirmed params in production
- Execution-hour cumulative P&L: infrastructure built (hourly cache, `execution_cumulative` column in `build_equity_curve()`), toggle exists in portfolio page, but not verified end-to-end against real trades
- Portfolio page prompts 5 (per-coin tab) and 6 (fund tab) — Prompt 6 ran but fund equity chart theoretical toggle was intentionally skipped; full fund portfolio not yet verified with real multi-strategy data
- Trade log page: built and structured but limited real trade history to verify statistics sections meaningfully
- Stat arb: empty shell only, no strategy implementation

**Not yet started:**
- Parameter derivation to reduce free param count in momentum walk-forward
- 275-day OOS window validation
- VPS nginx + SSL setup (currently running on raw port, no HTTPS)
- Website integration (GitHub Pages portfolio summary push)
- Stat arb strategy implementation

## 3. Bugs Fixed and Lessons Learned

**Stop loss state staleness** (most recent, highest impact). Auto-ratchet wrote `pending_stop` to disk but didn't update the shared `pos` dict in memory. The live-price fragment re-renders from `session_state` every 30 seconds so it displayed the pre-ratchet value until next full page reload. Fixed by also setting `pos['pending_stop'] = sugg_stop` on the in-memory dict immediately after the disk write. **Lesson: any disk write that mutates position state must also mutate the session_state copy — disk and memory must stay in sync atomically.**

**VPS positions.json stale structure.** VPS was deployed with old JSON files from before the FIFO position ID refactor. The code expected `{SYMBOL}_{YYYYMMDD}_{seq:03d}` keys but found legacy symbol-only keys. This caused silent failures in stop loss display and decision logic. Fix: manual JSON rewrite with correct structure. **Lesson: data file schema migrations need an explicit migration script run at deployment time, not just code changes.**

**Capital snapshot retroactive recalculation.** Adding a new coin to `ACTIVE_ASSETS` or changing `CAPITAL` was recalculating historical trade P&L because `build_trade_pairs()` was calling `get_coin_capital()` which read live config. Fixed by freezing `size_usd`, `coin_capital`, `capital_total`, `coin_weight` at entry time in `trades.json` and reading them back directly in `build_trade_pairs()`. **Lesson: any value that affects historical P&L must be snapshotted at the time of the event, never derived from current config.**

**Calmar ratio bug (research engine).** Raw total return was used instead of annualised return in the denominator. Fixed in `wf_engine.py`. **Lesson: Calmar on short backtests is misleading if not annualised — a 6-month backtest with 20% total return is not 20% annualised.**

**Sharpe computation with flat periods.** Zero-return flat bars between trades compressed std dev and inflated Sharpe. Resolved as intentional and consistent — computed over all bars including flat periods, annualised with `sqrt(periods_per_year)`. Documented explicitly so future changes don't accidentally remove flat periods from the calculation.

**`dropna` discipline.** Blanket `dropna()` before passing data to the backtest engine was dropping valid rows. Fixed to `dropna(subset=[indicator_cols])` with `fillna(0)` on the position column. **Lesson: never blanket dropna on financial time series — missing indicators are different from missing price data.**

**Lookahead bias in BBBreakout 4H filter.** 4H `ffill` alignment on Binance open-labelled bars required `.shift(1)` before reindex to prevent lookahead. Fixed. **Lesson: any regime filter on a different timeframe than the signal requires explicit forward-shift before alignment.**

**`risk_per_trade` typo in FIXED_PARAMS.** Caused permanent max-leverage positioning — the parameter was never being read correctly. Fixed by auditing dead parameters and removing `vol_ma_period` from strategy logic entirely. **Lesson: dead parameters in Optuna dilute search budget and act as weak regularisation — correct fix is removal and proportional reduction of `n_trials`.**

**Flask journal server port conflict.** `journal_server.py` on port 5001 conflicted with prior processes. Resolved by eliminating Flask entirely and consolidating all trade logging directly into Streamlit. **Lesson: minimise the number of running processes — every additional server is a failure point and operational burden.**

**Max drawdown showing 0.00%.** Division by peak equity which starts at zero produced `inf` or `nan`, defaulting to zero. Fixed by computing dollar drawdown first then expressing as percentage of peak at the point of maximum drawdown, with explicit `peak == 0` guard. **Lesson: drawdown calculations on equity curves that start at zero must use absolute dollar drawdown to find the worst point before converting to percentage.**

## 4. Open Questions and Decisions Blocking Progress

**Parameter count in momentum walk-forward.** 18 free parameters is confirmed too many. The plateau analysis showed 9/15 free parameters returning N/A and 40–55% score degradation at ±10% perturbation. Parameter derivation (collapsing related stops into 2–3 derived params) is the agreed fix but hasn't been executed. **This is blocking a clean final re-optimisation for momentum** — current `live_params.json` was generated with the bloated parameter set.

**BB Breakout production readiness.** Strategy function and `optimise.py` are built but no production optimisation has been run with full `n_trials`. The current `live_params.json` for bbbreakout either has test params or is empty. Decision needed: run full optimisation before activating BB Breakout as a live strategy.

**OOS window for final validation.** Plan was 275-day OOS for final validation after iteration phase. Not done. Decision needed: is the current momentum strategy stable enough to consider the iteration phase complete, or does parameter derivation need to happen first?

**Execution-hour P&L verification.** The infrastructure is built but the toggle hasn't been verified against real trades. With 4 open positions and some closed history now available, this can be tested. Blocking factor: need to confirm hourly cache is populated for the entry dates of current positions (May 6–10).

**VPS security.** Dashboard is currently accessible on a raw port with no authentication and no HTTPS. Anyone who knows the IP can see live positions and trade history. Decision needed: implement nginx + SSL + basic auth before sharing the URL more widely or treating this as production.

**Stat arb strategy.** Completely unstarted. No timeline set. Not blocking anything currently.

## 5. Immediate Next 3 Todos When We Resume

**Todo 1 — Fix VPS positions.json and verify stop loss end-to-end.** Your mate needs to `scp` the four corrected JSON files (positions, realised_capital for momentum + clean state for statarb and bbbreakout) onto the VPS. After that, verify the stop loss confirm button persists correctly across page refreshes on the live VPS — not just locally. This is the only remaining known production bug.

**Todo 2 — Run full BB Breakout optimisation and activate.** Run `python3 dashboards/bbbreakout/optimise.py` for each coin in the BB Breakout universe with production `n_trials`. Commit the resulting `live_params.json`. Push to VPS. Verify BB Breakout dashboard shows correct signals. This makes the second strategy live.

**Todo 3 — VPS security hardening.** Set up nginx as a reverse proxy in front of Streamlit, add SSL certificate via Let's Encrypt, add basic auth. This is a single focused Claude Code prompt covering `nginx.conf`, `docker-compose.yml` update, and certbot setup. Required before treating the dashboard as production-grade or sharing access.

---

## Reconciliation against `strategy-reference.md`

Some details in this chat summary differ from the Claude-Code audit. Where they conflict, the audit (which read the actual source) wins:

| Topic | Dashboard chat | STRATEGY_REFERENCE (audit) | Resolution |
|---|---|---|---|
| Streamlit entry | `live_trading/app.py` (single unified app, 3 pages) | `live_trading/dashboards/momentum/streamlit_app.py` | **Reconcile by reading code** — possibly both exist, or app structure refactored since one was written |
| Primary research notebooks | `momentumETH_wf.ipynb` + per-coin under `wf_testing/` | "older WF-only research, superseded by CPCV folders" | wf_testing/ is the source of `live_params.json` (production), CPCV is the new-research path. Both exist concurrently. |
| BB Breakout path | `topics/bb_breakout/strategies/wf_testing/bb_breakout_wf/` | `topics/momentum/strategies/bb_breakout_wf/strategy_design/bb_breakout.py` | One of the paths is wrong. Audit's path comes from the file system; chat's from memory. **Audit wins.** |
| BB Breakout status | "dashboard shell built, not yet actively trading, no live positions, no confirmed params" | "research only, not yet wired to a live dashboard" + "stub at `live_trading/dashboards/bbbreakout/`" | Consistent — not live yet. Chat has more detail on what's wired vs trading. |
| Stat arb | "empty shell only" | "stub" | Consistent. |
| Active universe (live) | implied by trade history | ACTIVE_ASSETS = ADA/AVAX/BTC/ETH/SOL/XRP (6 coins, no BNB) | Audit is authoritative. BNB exists in strategy registry but not ACTIVE_ASSETS — same diagnostic flagged earlier. |
| Parameter count | 18 free params | Engine default `n_trials = 400` (covers ~10 free at 40 trials each) | Both true. 18 params × 40 trials minimum = 720 trials minimum — current 400 is below the heuristic. **Real over-fitting risk flagged.** |
| OOS windows | 137-day current, 275-day planned | not specified in audit | Chat is authoritative for this. |
