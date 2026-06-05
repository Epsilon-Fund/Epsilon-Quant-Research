# Epsilon Fund — live_trading/ Architecture & Strategy Guide

> Data/artifact map: [[docs/CRYPTO_DATA_MANIFEST|crypto data manifest]]

## Purpose of this file
This memory gives a new Claude Code session full context on how `live_trading/` is
structured and exactly what to build when adding a new strategy (Stat Arb, BB Breakout,
or any future strategy).

---

## Directory layout

```
live_trading/
├── app.py                        ← Fund landing page (4 summary metrics)
├── pages/
│   ├── 1_Dashboards.py           ← Live signal + trade-log view (one tab per strategy)
│   ├── 2_Trade_Log.py            ← Trade journal (one tab per strategy)
│   └── 3_Portfolio.py            ← Equity/drawdown/deployment charts (one tab per strategy)
├── shared/                       ← Single source of truth — all reusable logic lives here
│   ├── __init__.py
│   ├── data_loader.py            ← All file I/O and data computation
│   ├── charts.py                 ← Plotly chart builders (equity, drawdown, deployment)
│   ├── styles.py                 ← Global CSS injector (call apply_styles() once per page)
│   ├── trade_log_components.py   ← render_fund_tab(), render_strategy_tab()
│   ├── portfolio_components.py   ← render_strategy_portfolio(), render_fund_portfolio()
│   └── binance_utils.py          ← MAE / price fetch helpers
└── dashboards/
    ├── momentum/                 ← COMPLETE — reference implementation
    │   ├── config.py             ← CAPITAL, ACTIVE_ASSETS, COIN_WEIGHTS, WF_CONFIG, EXECUTION_HOUR
    │   ├── strategies.py         ← STRATEGY_REGISTRY dict of callable strategy functions
    │   ├── dashboard.py          ← run_dashboard(), apply_decision(), fetch_live_price()
    │   ├── optimise.py           ← Walk-forward optimiser; writes live_params.json
    │   ├── streamlit_app.py      ← Standalone entry point (exec'd by 1_Dashboards.py)
    │   ├── trades.json           ← Append-only trade log  [ {ENTRY}, {EXIT}, … ]
    │   ├── positions.json        ← Live open positions     { position_id: { … } }
    │   ├── live_params.json      ← Optimised params        { symbol: { params, fixed_params, … } }
    │   └── mae_cache.json        ← MAE cache               { position_id: mae_pct }
    ├── statarb/                  ← STUB — needs strategies.py, dashboard.py, optimise.py, streamlit_app.py
    │   ├── config.py             ← exists (ACTIVE_ASSETS is empty [])
    │   ├── trades.json           ← exists (empty [])
    │   ├── positions.json        ← exists (empty {})
    │   └── live_params.json      ← exists (empty {})
    └── bbbreakout/               ← STUB — same state as statarb/
        ├── config.py
        ├── trades.json
        ├── positions.json
        └── live_params.json
```

---

## How data flows (read this first)

```
Binance API
    ↓
dashboard.py::run_dashboard()     ← expensive; cached 5 min in streamlit_app.py
    ↓
streamlit_app.py                  ← exec'd by 1_Dashboards.py via _exec_page()
    → writes trades.json          (append-only via _write_trade())
    → writes positions.json       (open/close via _write_position_entry/exit())

trades.json + positions.json
    ↓
shared/data_loader.py
    load_trades(data_dir)         → normalised trade list
    load_positions(data_dir)      → open position dict
    build_trade_pairs(data_dir)   → { "closed": [...], "open": [...] }
    build_equity_curve(data_dir)  → daily P&L DataFrame
    build_capital_deployment()    → daily deployment DataFrame
    load_config(data_dir)         → { capital, active_assets, coin_weights, execution_hour }
    ↓
shared/trade_log_components.py
    render_strategy_tab(data_dir, prefix, strategy_keys, …)
    render_fund_tab(dashboard_dirs, prefix)
    ↓ called by pages/2_Trade_Log.py

shared/portfolio_components.py
    render_strategy_portfolio(data_dir, prefix, strategy_name)
    render_fund_portfolio(dashboard_dirs, prefix)
    ↓ called by pages/3_Portfolio.py

app.py — calls load_trades/build_trade_pairs/load_config directly
          aggregates across all three dashboard dirs for the summary cards
```

---

## The three entry-point pages and how they wire in strategies

### `pages/1_Dashboards.py` — Live signals + trade forms
Uses `_exec_page(path)` to embed `dashboards/<strategy>/streamlit_app.py` inside a tab.
`set_page_config` is no-op'd; `_SUPPRESS_H1=True` is injected to suppress the inner H1.

To add a strategy:
1. Build `dashboards/<strategy>/streamlit_app.py` (see momentum as reference).
2. Add its path to `_DASHBOARDS_` and add a tab:
```python
# In pages/1_Dashboards.py
_STATARB_APP = os.path.join(_LT_DIR, 'dashboards', 'statarb', 'streamlit_app.py')

tab_momentum, tab_statarb, tab_bb = st.tabs(["Momentum", "Stat Arb", "BB Breakout"])

with tab_statarb:
    _exec_page(_STATARB_APP, extra={'_SUPPRESS_H1': True})
```

### `pages/2_Trade_Log.py` — Trade journal
Calls `render_strategy_tab(data_dir, prefix, strategy_keys, display_name)` from shared.

To activate a strategy tab:
```python
# Replace the st.info stub in pages/2_Trade_Log.py:
with tab_statarb:
    render_strategy_tab(
        data_dir=DASHBOARD_DIRS["Stat Arb"],
        prefix="statarb",
        strategy_keys=["statarb_pairs"],   # must match strategy field in trades.json
        display_name="Stat Arb",
        show_strategy_col=True,
    )
```

### `pages/3_Portfolio.py` — Equity curves + drawdown
Calls `render_strategy_portfolio(data_dir, prefix, strategy_name)` from shared.

To activate a strategy tab:
```python
# Replace the st.info stub in pages/3_Portfolio.py:
with tab_statarb:
    render_strategy_portfolio(
        data_dir=DASHBOARD_DIRS["Stat Arb"],
        prefix="statarb",
        strategy_name="Stat Arb",
    )
```

---

## `shared/data_loader.py` — key functions

| Function | What it does |
|---|---|
| `load_trades(data_dir)` | Reads trades.json, normalises v1/v2/v3 schemas, sorts by date |
| `load_positions(data_dir)` | Reads positions.json as-is |
| `load_live_params(data_dir)` | Reads live_params.json as-is |
| `load_config(data_dir)` | Imports config.py via importlib; returns plain dict |
| `build_trade_pairs(data_dir)` | FIFO-matches ENTRY→EXIT by symbol; returns {closed, open} |
| `build_equity_curve(data_dir)` | Daily cumulative P&L from closed pairs; cached |
| `build_capital_deployment(data_dir)` | Daily deployed $ using frozen size_usd; cached |
| `compute_mae(...)` | MAE% from Binance daily OHLC; cached in mae_cache.json |

`data_dir` is always the **absolute path** to the strategy's dashboard directory
(e.g. `dashboards/momentum/`). Every function is directory-agnostic — the same code
serves all strategies.

---

## `config.py` — what every strategy dashboard must export

```python
CAPITAL          = 100_000     # total strategy capital in USD
EXECUTION_HOUR   = 8           # UTC hour for theoretical execution price
INDICATOR_WARMUP = 100         # bars to discard for indicator warm-up
COIN_WEIGHTS     = {}          # optional; empty = equal-weight across ACTIVE_ASSETS
ACTIVE_ASSETS    = ["ETHUSDT"] # list of symbols this strategy trades
WF_CONFIG        = {           # walk-forward optimiser settings
    "train_bars": 1050,
    "test_bars":  137,
    "burnin":     100,
    "n_trials":   400,
    "cost":       0.001,
}
```

`shared/data_loader.py::load_config()` reads exactly these five keys.
Missing any of them will cause a KeyError.

---

## `trades.json` — trade record schema (v3, current)

Every `_write_trade()` call in `streamlit_app.py` appends one record.

**ENTRY record:**
```json
{
  "timestamp": "2026-04-15T08:00:00",
  "date":      "2026-04-15",
  "position_id": "ETHUSDT_20260415_001",
  "action":    "ENTRY",
  "strategy":  "momentum_swing",
  "actual_price":         3200.00,
  "theoretical_price":    3195.00,
  "slippage_pct":         0.0016,
  "actual_leverage":      1.5,
  "theoretical_leverage": 1.5,
  "theoretical_stop":     2950.00,
  "discretion_note":      "",
  "direction":            "long",
  "entry_type":           "Strategy",
  "entry_close":          3195.00,
  "signal_snapshot":      { … },
  "coin_capital":         20000.0,
  "size_usd":             30000.0,
  "capital_total":        100000,
  "coin_weight":          null
}
```

**EXIT record:**
```json
{
  "timestamp": "2026-04-22T08:00:00",
  "date":      "2026-04-22",
  "position_id": "ETHUSDT_20260415_001",
  "action":    "EXIT",
  "strategy":  "momentum_swing",
  "actual_price":         3400.00,
  "theoretical_price":    3395.00,
  "actual_leverage":      1.5,
  "theoretical_leverage": 1.5,
  "exit_type":            "full",
  "exit_leverage":        1.5,
  "exit_close":           3395.00,
  "exit_reason":          "Strategy"
}
```

`position_id` format: `{SYMBOL}_{YYYYMMDD}_{seq:03d}` — this is how FIFO matching works.

---

## `positions.json` — open position schema

```json
{
  "ETHUSDT_20260415_001": {
    "position_id":         "ETHUSDT_20260415_001",
    "symbol":              "ETHUSDT",
    "strategy":            "momentum_swing",
    "in_position":         true,
    "entry_date":          "2026-04-15",
    "entry_price":         3200.00,
    "leverage_multiplier": 1.5,
    "pending_stop":        2980.00,
    "current_stop":        2950.00,
    "direction":           "long",
    "partial_exits":       0,
    "exit_price":          null,
    "exit_date":           null,
    "discretion_note":     "",
    "coin_capital":        20000.0,
    "size_usd":            30000.0
  }
}
```

`in_position: false` after exit — record is kept for history.
`pending_stop` vs `current_stop`: two-state ratchet; confirm button in the dashboard promotes pending → current.

---

## Building a new strategy dashboard — step by step

### Files to create inside `dashboards/<strategy>/`

1. **`strategies.py`** — define signal functions + STRATEGY_REGISTRY
   ```python
   def my_strategy(df_slice: pd.DataFrame, params: dict) -> dict:
       # df_slice: OHLCV DataFrame for one asset, pre-warmed
       # params:   dict from live_params.json for this symbol
       # return:   signal dict — must include at minimum:
       #   decision, leverage_multiplier, size_usd, theoretical_stop,
       #   close, entry_long, current_stop, stop_updated, stop_detail
       ...

   STRATEGY_REGISTRY = {"my_strategy": my_strategy}
   ```

2. **`dashboard.py`** — computation layer (no Streamlit)
   Must export:
   - `run_dashboard(coin_symbols, live_params, positions) → dict`
   - `apply_decision(sig, open_positions, exec_price, coin_capital) → dict`
   - `fetch_live_price(symbol) → float | None`
   - `get_coin_capital(symbol) → float`
   - `get_open_positions(symbol, positions) → list`

   Mirror momentum/dashboard.py structure — it fetches OHLCV from Binance,
   runs each symbol through the strategy function, and returns a list of
   `coin_rows` dicts (one per asset).

3. **`optimise.py`** — walk-forward optimiser
   Must write `live_params.json` with structure:
   ```json
   {
     "ETHUSDT": {
       "params":            { "ema_period": 20, … },
       "fixed_params":      { "ema_period": 20 },
       "fixed_param_keys":  ["ema_period"],
       "optimised_on":      "2026-04-15",
       "strategy":          "my_strategy"
     }
   }
   ```
   Mirror momentum/optimise.py. Run it once per asset before the dashboard
   will show any signals: `python3 dashboards/<strategy>/optimise.py --asset ETHUSDT`

4. **`streamlit_app.py`** — the live dashboard UI
   Start with this at the top:
   ```python
   # Standalone entry point — primary app is live_trading/app.py
   ```
   Mirror momentum/streamlit_app.py — it:
   - Calls `load_all()` (cached) → runs run_dashboard()
   - Applies decisions (uncached, reads positions.json fresh)
   - Renders: portfolio summary card → active positions → decisions table
     → trade log forms → entry conditions → caution flags → stop details → params
   - Handles all _write_trade / _write_position_* calls on form submit
   - Uses `_SUPPRESS_H1` guard so the H1 is hidden when exec'd inside a tab

5. **`config.py`** — already exists; just populate `ACTIVE_ASSETS` after optimising

6. **Data files** — already exist as empty defaults:
   `trades.json` (`[]`), `positions.json` (`{}`),
   `live_params.json` (`{}`), `mae_cache.json` (`{}`)

---

## Wiring the new strategy into the three pages

### `pages/1_Dashboards.py`
```python
# Add path
_STATARB_APP = os.path.join(_LT_DIR, 'dashboards', 'statarb', 'streamlit_app.py')

# Activate tab (replace st.info stub)
with tab_statarb:
    _exec_page(_STATARB_APP, extra={'_SUPPRESS_H1': True})
```

### `pages/2_Trade_Log.py`
```python
# strategy_keys must match the 'strategy' field written in trades.json
with tab_statarb:
    render_strategy_tab(
        data_dir=DASHBOARD_DIRS["Stat Arb"],
        prefix="statarb",
        strategy_keys=["statarb_pairs"],
        display_name="Stat Arb",
        show_strategy_col=True,
    )
```

### `pages/3_Portfolio.py`
```python
with tab_statarb:
    render_strategy_portfolio(
        data_dir=DASHBOARD_DIRS["Stat Arb"],
        prefix="statarb",
        strategy_name="Stat Arb",
    )
```

### `app.py` — no changes needed
`DASHBOARD_DIRS` already includes `"Stat Arb"` and `"BB Breakout"`.
`_load_summary()` skips dirs with empty `trades.json`, so they show zero
until the first real trade is logged.

---

## Key conventions — follow these exactly

| Convention | Detail |
|---|---|
| `data_dir` is always absolute | Use `os.path.dirname(os.path.abspath(__file__))` inside each dashboard |
| Widget keys are prefixed | Every `st.widget(key=...)` must use `f"{prefix}_..."` to prevent DuplicateWidgetID |
| No module-level Streamlit calls in shared/ | `render_*` functions are called inside tab contexts; they must not call `st.set_page_config` |
| `@st.cache_data` args must be hashable | Pass `tuple(sorted(dict.items()))` instead of raw dicts |
| `trades.json` is append-only | Never rewrite it; only append new records |
| `positions.json` is mutable | Read fresh (uncached) on every Streamlit render cycle |
| `live_params.json` drives the dashboard | Dashboard shows no signals until optimise.py has run for at least one asset |
| Capital snapshot is frozen at ENTRY time | `coin_capital` and `size_usd` stored in the ENTRY trade record; never recomputed from config for historical records |

---

## Running the app

```bash
# Primary unified app (all strategies, all pages)
streamlit run live_trading/app.py

# Momentum standalone (bypasses unified app)
streamlit run live_trading/dashboards/momentum/streamlit_app.py

# Optimise a new asset before adding it to ACTIVE_ASSETS
python3 live_trading/dashboards/statarb/optimise.py --asset ETHUSDT
```

---

## What's currently live vs stub

| Component | Momentum | Stat Arb | BB Breakout |
|---|---|---|---|
| `config.py` | ✅ configured | ✅ exists, ACTIVE_ASSETS=[] | ✅ exists, ACTIVE_ASSETS=[] |
| `strategies.py` | ✅ complete | ❌ not written | ❌ not written |
| `dashboard.py` | ✅ complete | ❌ not written | ❌ not written |
| `optimise.py` | ✅ complete | ❌ not written | ❌ not written |
| `streamlit_app.py` | ✅ complete | ❌ not written | ❌ not written |
| `trades.json` | ✅ has real data | ✅ empty `[]` | ✅ empty `[]` |
| `positions.json` | ✅ has open pos | ✅ empty `{}` | ✅ empty `{}` |
| `live_params.json` | ✅ has params | ✅ empty `{}` | ✅ empty `{}` |
| 1_Dashboards tab | ✅ exec'd | ❌ info stub | ❌ info stub |
| 2_Trade_Log tab | ✅ render_strategy_tab | ❌ info stub | ❌ info stub |
| 3_Portfolio tab | ✅ render_strategy_portfolio | ❌ info stub | ❌ info stub |
| app.py summary | ✅ counted | ✅ skipped (no trades) | ✅ skipped (no trades) |

---

## Adding notes to this cluster

- Live-trading notes belong here when they describe app architecture, dashboard wiring, shared component contracts, trade/position schemas, Streamlit behavior, or operational patterns for `live_trading/`.
- Prefer appending durable architecture notes to this file. If a standalone note is clearer, place it under `live_trading/` with a name like `live_<area>_<topic>.md`, `dashboard_<strategy>_<topic>.md`, or `ops_<workflow>_runbook.md`.
- Add `> Hub: [[live_trading/CLAUDE|CLAUDE]]` near the top of each standalone live-trading note, and add a path wikilink to that note in the relevant section of this file.
