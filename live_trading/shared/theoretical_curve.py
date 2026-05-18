"""
Theoretical equity curve — pure-strategy backtest over history.

This module produces a daily P&L series that depends ONLY on:
    - the strategy code in dashboards/<strategy>/strategies.py
    - the live_params history in dashboards/<strategy>/live_params_history.json
    - the cached daily OHLC for each ACTIVE_ASSET
    - the dashboard's CAPITAL / COIN_WEIGHTS / TRADING_COST_PCT

It does NOT read trades.json — the click timing of real trades cannot
affect it.  This is the contract: the Theoretical line on the Portfolio
page reflects what the strategy itself produced, regardless of when the
user actually pressed Entry/Exit.

Execution model: signal at close of day T → fill at close of day T
(same-bar). Closed trades only — open positions at the end of the
window are dropped (not marked-to-market).
"""
from __future__ import annotations

import importlib.util
import json
import os
from datetime import date, datetime, timedelta

import pandas as pd

# ── Sentinel covering "from the dawn of time" for seed entries ────────────────
_EPOCH = date(1970, 1, 1)


# ── strategies.py loader (per-dashboard-dir) ──────────────────────────────────

_STRATEGY_REGISTRY_CACHE: dict = {}


def _load_strategy_registry(data_dir: str) -> dict:
    """Import strategies.py from data_dir and return its STRATEGY_REGISTRY."""
    abs_dir = os.path.abspath(data_dir)
    if abs_dir in _STRATEGY_REGISTRY_CACHE:
        return _STRATEGY_REGISTRY_CACHE[abs_dir]
    path = os.path.join(abs_dir, 'strategies.py')
    if not os.path.exists(path):
        _STRATEGY_REGISTRY_CACHE[abs_dir] = {}
        return {}
    spec = importlib.util.spec_from_file_location(
        f'_strats_{abs_dir.replace(os.sep, "_")}', path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    reg = getattr(mod, 'STRATEGY_REGISTRY', {})
    _STRATEGY_REGISTRY_CACHE[abs_dir] = reg
    return reg


# ── Params history (load + auto-seed) ─────────────────────────────────────────

def _seed_history_from_live_params(data_dir: str) -> list:
    """Bootstrap live_params_history.json from the current live_params.json.

    Each seed entry covers "from the epoch onwards" so the current params
    are treated as default for any historical date until a real
    optimisation event appends a later record above it.  We also stash
    the symbol's actual ``optimised_on`` on the seed so the no-trades
    window-start fallback can use it — see build_theoretical_curve.
    """
    live_path = os.path.join(data_dir, 'live_params.json')
    if not os.path.exists(live_path):
        return []
    with open(live_path) as f:
        live = json.load(f)
    seed = []
    for symbol, entry in live.items():
        seed.append({
            'symbol':           symbol,
            'effective_from':   str(_EPOCH),
            'strategy':         entry.get('strategy'),
            'params':           entry.get('params', {}),
            'fixed_param_keys': entry.get('fixed_param_keys', []),
            'optimised_on':     entry.get('optimised_on'),
            '_seeded':          True,
        })
    hist_path = os.path.join(data_dir, 'live_params_history.json')
    with open(hist_path, 'w') as f:
        json.dump(seed, f, indent=2)
    return seed


def _most_recent_optimised_date(history: list, symbol: str) -> "date | None":
    """Per-symbol date that the currently-effective params became live.

    Used as the no-trades-window start.  Walks the symbol's history
    newest-first and returns the first usable date — `effective_from`
    for real reoptimise entries, `optimised_on` for the seeded fallback.
    """
    sym_records = sorted(
        [h for h in history if h.get('symbol') == symbol],
        key=lambda r: _parse_date(r['effective_from']),
        reverse=True,
    )
    for rec in sym_records:
        if not rec.get('_seeded'):
            try:
                return _parse_date(rec['effective_from'])
            except Exception:
                continue
        if rec.get('optimised_on'):
            try:
                return _parse_date(rec['optimised_on'])
            except Exception:
                continue
    return None


def _load_history(data_dir: str) -> list:
    """Load live_params_history.json (seeding it from live_params.json if missing)."""
    path = os.path.join(data_dir, 'live_params_history.json')
    if not os.path.exists(path):
        return _seed_history_from_live_params(data_dir)
    with open(path) as f:
        history = json.load(f)
    if not history:
        return _seed_history_from_live_params(data_dir)
    return history


def _parse_date(s: str) -> date:
    return datetime.strptime(s, '%Y-%m-%d').date()


def _segments_for_symbol(history: list, symbol: str,
                         window_start: date, window_end: date) -> list:
    """Return [(seg_start, seg_end, strategy, params), ...] covering the window.

    Each segment's [seg_start, seg_end] inclusive is the range of dates on
    which the given params were the latest active record.  Segments are
    clipped to [window_start, window_end].
    """
    sym_records = sorted(
        [h for h in history if h.get('symbol') == symbol],
        key=lambda r: _parse_date(r['effective_from']),
    )
    if not sym_records:
        return []

    segments = []
    for i, rec in enumerate(sym_records):
        seg_start = _parse_date(rec['effective_from'])
        if i + 1 < len(sym_records):
            seg_end = _parse_date(sym_records[i + 1]['effective_from']) - timedelta(days=1)
        else:
            seg_end = window_end

        # Clip to window
        seg_start = max(seg_start, window_start)
        seg_end   = min(seg_end,   window_end)
        if seg_start > seg_end:
            continue

        segments.append((seg_start, seg_end, rec.get('strategy'), rec.get('params', {})))
    return segments


# ── Per-symbol backtest ───────────────────────────────────────────────────────

def _first_entry_date_per_symbol(trades_list: list) -> dict:
    """Min ENTRY date per symbol (used only to determine window start)."""
    out: dict = {}
    for t in trades_list:
        if t.get('action') != 'ENTRY':
            continue
        pid = t.get('position_id') or ''
        # Extract symbol from FIFO position_id; fall back to pid as-is
        import re
        m = re.match(r'^([A-Z0-9]+)_\d{8}_\d{3}$', pid)
        sym = m.group(1) if m else pid
        d = t.get('date')
        if not d or not sym:
            continue
        if sym not in out or d < out[sym]:
            out[sym] = d
    return out


def _closed_trades_for_symbol(
    symbol: str,
    ohlc_df: pd.DataFrame,
    segments: list,
    strategy_registry: dict,
    coin_capital: float,
    cost_pct: float,
) -> list:
    """Run each param-segment's strategy on ohlc_df, stitch positions, extract
    closed trades (long AND short), return list of dicts with entry/exit
    date+close, leverage, pnl_usd (net of round-trip cost).

    Same-bar close execution: position 0 → non-zero at bar i means we
    enter at close[i].  Non-zero → 0 at bar j means we exit at close[j].
    Direction is sign(position_at_entry): +1 long, -1 short.

    Trades whose entry is inside a window but whose exit lies beyond it are
    dropped — "only count closed theoretical trades".

    ``ohlc_df`` can be daily OR hourly — the strategy decides what to do
    with it.  exit_date is the calendar date of the exit bar's timestamp,
    so hourly bars within the same day collapse into one daily P&L bucket
    when aggregated upstream.
    """
    if ohlc_df is None or ohlc_df.empty or not segments:
        return []

    # Strategy may be stateful (state machine inside the function), but each
    # call returns positions for every bar in ohlc_df.  We run the strategy
    # once per segment with that segment's params, then keep only the
    # positions/sizes inside the segment's date range.
    position = pd.Series(0, index=ohlc_df.index, dtype=int)
    pos_size = pd.Series(0.0, index=ohlc_df.index, dtype=float)

    for seg_start, seg_end, strategy, params in segments:
        fn = strategy_registry.get(strategy)
        if fn is None:
            continue
        try:
            out_df, _ = fn(ohlc_df, params)
        except Exception as e:
            print(f"  theoretical_curve {symbol} seg {seg_start}..{seg_end}: {e}")
            continue
        # Bars whose calendar date falls inside the segment
        idx_dates = out_df.index.map(lambda ts: ts.date()
                                     if hasattr(ts, 'date') else ts)
        mask = pd.Series(
            [(seg_start <= d <= seg_end) for d in idx_dates],
            index=out_df.index,
        )
        position.loc[mask] = out_df.loc[mask, 'position'].astype(int).values
        pos_size.loc[mask] = out_df.loc[mask, 'position_size'].astype(float).values

    # ── Extract trades from position transitions ──────────────────────────────
    # Sign convention: position is +1 (long), -1 (short), or 0 (flat).
    # P&L direction follows the sign of the entry position.
    closed = []
    entry_sign = 0
    entry_ts   = None
    entry_lev  = 0.0

    pos_values  = position.values
    size_values = pos_size.values
    ts_index    = position.index

    for i in range(len(position)):
        ts  = ts_index[i]
        pos = int(pos_values[i])
        sz  = float(size_values[i])

        if entry_sign == 0 and pos != 0:
            entry_sign = 1 if pos > 0 else -1
            entry_ts   = ts
            entry_lev  = sz
        elif entry_sign != 0 and pos == 0:
            entry_close = float(ohlc_df.loc[entry_ts, 'Close'])
            exit_close  = float(ohlc_df.loc[ts, 'Close'])
            gross_ret   = (
                (exit_close / entry_close - 1.0) * entry_sign
                if entry_close else 0.0
            )
            size_usd    = coin_capital * entry_lev
            gross_pnl   = size_usd * gross_ret
            cost_usd    = size_usd * cost_pct * 2.0
            closed.append({
                'symbol':       symbol,
                'direction':    'long' if entry_sign > 0 else 'short',
                'entry_date':   entry_ts.date() if hasattr(entry_ts, 'date') else entry_ts,
                'exit_date':    ts.date() if hasattr(ts, 'date') else ts,
                'entry_close':  entry_close,
                'exit_close':   exit_close,
                'leverage':     entry_lev,
                'size_usd':     size_usd,
                'pnl_usd':      gross_pnl - cost_usd,
                'gross_pnl':    gross_pnl,
                'cost_usd':     cost_usd,
            })
            entry_sign = 0
            entry_ts   = None
            entry_lev  = 0.0
        elif entry_sign != 0 and pos != 0 and ((pos > 0) != (entry_sign > 0)):
            # Direct flip without a flat bar in between — close current,
            # open new on the same bar.  Rare but possible for short-on-
            # long-stop or vice versa.
            entry_close = float(ohlc_df.loc[entry_ts, 'Close'])
            exit_close  = float(ohlc_df.loc[ts, 'Close'])
            gross_ret   = (
                (exit_close / entry_close - 1.0) * entry_sign
                if entry_close else 0.0
            )
            size_usd    = coin_capital * entry_lev
            gross_pnl   = size_usd * gross_ret
            cost_usd    = size_usd * cost_pct * 2.0
            closed.append({
                'symbol':       symbol,
                'direction':    'long' if entry_sign > 0 else 'short',
                'entry_date':   entry_ts.date() if hasattr(entry_ts, 'date') else entry_ts,
                'exit_date':    ts.date() if hasattr(ts, 'date') else ts,
                'entry_close':  entry_close,
                'exit_close':   exit_close,
                'leverage':     entry_lev,
                'size_usd':     size_usd,
                'pnl_usd':      gross_pnl - cost_usd,
                'gross_pnl':    gross_pnl,
                'cost_usd':     cost_usd,
            })
            entry_sign = 1 if pos > 0 else -1
            entry_ts   = ts
            entry_lev  = sz

    return closed


# ── Public API ────────────────────────────────────────────────────────────────

def build_theoretical_curve(data_dir: str) -> pd.DataFrame:
    """Pure-strategy daily P&L curve.

    Returns DataFrame with columns:
        date | theoretical_pnl | theoretical_cumulative

    Window: min(entry_date in trades.json) → today.  Returns an empty
    DataFrame when there are no live trades to anchor the window or when
    no strategy can be loaded.
    """
    # Lazy imports avoid a circular dependency with data_loader.
    from shared.data_loader import (
        load_trades, load_config, _load_config, get_coin_capital,
    )
    from shared.cache_manager import get_daily_ohlcv_range, get_hourly_ohlcv

    cfg          = load_config(data_dir)
    cost_pct     = float(cfg.get('trading_cost_pct', 0.0))
    active       = list(cfg.get('active_assets', []))
    data_freq    = str(cfg.get('data_frequency', 'daily')).lower()
    if not active:
        return pd.DataFrame(columns=['date', 'theoretical_pnl', 'theoretical_cumulative'])

    history             = _load_history(data_dir)
    strategy_registry   = _load_strategy_registry(data_dir)

    trades       = load_trades(data_dir)
    first_dates  = _first_entry_date_per_symbol(trades)
    window_end   = date.today()
    if first_dates:
        window_start = min(first_dates.values())
    else:
        # No live trades yet — fall back to "since most recent optimised_on
        # per symbol" so a brand-new strategy (e.g. BB Breakout pre-first-
        # trade) still shows a theoretical curve.  The user can spot
        # missed entries this way.  Window start = the EARLIEST of the
        # active assets' most-recent optimisation dates so every symbol's
        # signals from then onwards render.
        opt_dates = [
            d for d in (_most_recent_optimised_date(history, s) for s in active)
            if d is not None
        ]
        if not opt_dates:
            return pd.DataFrame(columns=['date', 'theoretical_pnl', 'theoretical_cumulative'])
        window_start = min(opt_dates)

    # Indicator warmup — pull extra history before window_start so the
    # strategy's indicators are warm by the time we reach the window.
    # INDICATOR_WARMUP is in BARS (daily for momentum, hourly for BB),
    # which we convert to a calendar-day buffer here.  Add a healthy
    # safety margin so multi-timeframe strategies (BB resamples 1H→4H)
    # never run short of history at the boundary.
    try:
        warmup_bars = int(getattr(_load_config(data_dir), 'INDICATOR_WARMUP', 100))
    except Exception:
        warmup_bars = 100
    if data_freq == 'hourly':
        warmup_days = max(30, warmup_bars // 24 + 30)
    else:
        warmup_days = warmup_bars + 30
    fetch_start = window_start - timedelta(days=warmup_days)

    all_closed: list = []
    for symbol in active:
        segments = _segments_for_symbol(history, symbol, window_start, window_end)
        if not segments:
            continue
        try:
            if data_freq == 'hourly':
                ohlc_df = get_hourly_ohlcv(symbol, fetch_start, window_end)
            else:
                ohlc_df = get_daily_ohlcv_range(symbol, fetch_start, window_end)
        except Exception as e:
            print(f"build_theoretical_curve: {symbol} fetch failed — {e}")
            continue
        if ohlc_df is None or ohlc_df.empty:
            continue

        coin_cap = get_coin_capital(symbol, data_dir)
        closed = _closed_trades_for_symbol(
            symbol=symbol,
            ohlc_df=ohlc_df,
            segments=segments,
            strategy_registry=strategy_registry,
            coin_capital=coin_cap,
            cost_pct=cost_pct,
        )
        all_closed.extend(closed)

    # ── Aggregate daily P&L on exit date, bucketed over [window_start, today] ─
    pnl_by_date: dict = {}
    for t in all_closed:
        d = t['exit_date']
        # Only keep theoretical trades whose exit falls inside the live window
        if d < window_start or d > window_end:
            continue
        pnl_by_date[d] = pnl_by_date.get(d, 0.0) + t['pnl_usd']

    rows = []
    cur  = window_start
    while cur <= window_end:
        rows.append({'date': cur, 'theoretical_pnl': pnl_by_date.get(cur, 0.0)})
        cur += timedelta(days=1)

    df = pd.DataFrame(rows)
    df['theoretical_cumulative'] = df['theoretical_pnl'].cumsum()
    return df
