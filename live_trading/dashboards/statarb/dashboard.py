"""
Epsilon Fund — Stat Arb computation module.

No display, no HTML, no print statements (except fetch confirmations).
All trade logging is done in streamlit_app.py.

Exported public API
-------------------
  run_dashboard(pair_keys, live_params, positions) → dict
  fetch_pair_data(symbol_y, symbol_x, lookback_bars) → DataFrame
  fetch_hourly_recent(symbol, days)                  → DataFrame
  fetch_live_price(symbol)                           → float | None
  compute_pair_signal(pair_df, params, strategy)     → dict
  apply_decision(sig, open_positions, pair_capital)  → dict
  get_open_positions(pair_key, positions)            → dict
  get_pair_capital(pair_key)                         → float
  get_execution_price(hourly_df, signal_date, hour)  → float | None
  load_live_params()                                 → dict
  load_positions()                                   → dict
"""

import os
import sys
import json
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_LT_DIR  = os.path.abspath(os.path.join(_DASHBOARD_DIR, '..', '..'))
ROOT     = os.path.dirname(_LT_DIR)
_INFRA   = os.path.join(ROOT, 'infrastructure', 'data')
for _p in (_DASHBOARD_DIR, _LT_DIR, _INFRA):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from binance_client import get_binance_client
from strategies import STRATEGY_REGISTRY
from config import ACTIVE_ASSETS, INDICATOR_WARMUP, EXECUTION_HOUR, CAPITAL, COIN_WEIGHTS

LIVE_PARAMS_PATH = os.path.join(_DASHBOARD_DIR, 'live_params.json')
POSITIONS_PATH   = os.path.join(_DASHBOARD_DIR, 'positions.json')


# ══════════════════════════════════════════════════════════════════════════════
#  Data fetching
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def fetch_live_price(symbol: str):
    """Current market price for a single symbol. Returns float or None."""
    try:
        client = get_binance_client()
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"  fetch_live_price({symbol}) failed: {e}")
        return None


def fetch_pair_data(symbol_y: str, symbol_x: str, lookback_bars: int = 300):
    """
    Return daily OHLCV for both legs, strip the incomplete bar, inner-join.
    Returns a DataFrame with Close_Y and Close_X indexed by datetime.

    Reads from the local parquet cache (live_trading/cache/daily/).
    Falls back to a live Binance fetch on cache miss (first run or after
    running backfill_cache.py).
    """
    from shared.cache_manager import get_daily_ohlcv

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=24)

    def _fetch(sym):
        df = get_daily_ohlcv(sym, warmup_bars=lookback_bars + 10)
        if df.empty:
            raise RuntimeError(
                f'fetch_pair_data: no daily data for {sym}. '
                'Run backfill_cache.py first.'
            )
        last_naive = (
            df.index[-1].tz_convert('UTC').tz_localize(None)
            if df.index.tz is not None else df.index[-1]
        )
        return df.iloc[:-1] if last_naive > cutoff else df

    df_y = _fetch(symbol_y)
    df_x = _fetch(symbol_x)

    y = df_y[['Close']].rename(columns={'Close': 'Close_Y'})
    x = df_x[['Close']].rename(columns={'Close': 'Close_X'})
    merged = y.join(x, how='inner').dropna()

    last_bar = merged.index[-1]
    print(f"  {symbol_y}/{symbol_x}  last bar: {last_bar.date()}"
          f"  Y={float(merged['Close_Y'].iloc[-1]):.4f}"
          f"  X={float(merged['Close_X'].iloc[-1]):.4f}")
    return merged


@st.cache_data(ttl=3600)
def fetch_hourly_recent(symbol: str, days: int = 3):
    """
    Return the last `days` days of 1h OHLCV for a symbol.

    Reads from the local parquet cache (live_trading/cache/hourly/).
    Returns a UTC-aware DatetimeIndex DataFrame to match get_execution_price().
    """
    from shared.cache_manager import get_hourly_ohlcv
    from datetime import date as _date, timedelta

    start = _date.today() - timedelta(days=days + 1)
    end   = _date.today() + timedelta(days=1)

    df = get_hourly_ohlcv(symbol, start, end)

    if df.empty:
        return df

    # Cache stores tz-naive timestamps; localise to UTC to match get_execution_price().
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize('UTC')

    return df


def get_execution_price(hourly_df: pd.DataFrame, signal_date, hour_utc: int):
    """
    Return the Open price at hour_utc UTC on signal_date + 1 day.
    Returns None if the bar is in the future or missing.
    """
    exec_dt = (
        pd.Timestamp(signal_date, tz='UTC')
        + pd.Timedelta(days=1)
    ).replace(hour=hour_utc, minute=0, second=0, microsecond=0)

    if exec_dt > pd.Timestamp(datetime.now(timezone.utc)):
        return None
    return float(hourly_df.loc[exec_dt, 'Open']) if exec_dt in hourly_df.index else None


# ══════════════════════════════════════════════════════════════════════════════
#  Signal computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_pair_signal(pair_df: pd.DataFrame, params: dict, strategy_name: str) -> dict:
    """
    Run the strategy on pair_df and extract the last-bar signal state.
    Returns a dict of indicator values used by apply_decision and the UI.
    """
    strategy_fn = STRATEGY_REGISTRY[strategy_name]
    result_df, _ = strategy_fn(pair_df.copy(), params)
    last = result_df.iloc[-1]

    def _safe(v):
        return float(v) if v is not None and not pd.isna(v) else None

    return {
        'z':        _safe(last['z']),
        'spread':   _safe(last['spread']),
        'beta':     _safe(last['beta']),
        'close_y':  float(last['Close_Y']),
        'close_x':  float(last['Close_X']),
        'last_pos': int(last['position']),
        'params':   params,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Decision logic
# ══════════════════════════════════════════════════════════════════════════════

def get_open_positions(pair_key: str, positions: dict) -> dict:
    """Return {position_id: pos} for all open positions of pair_key."""
    return {
        pid: p for pid, p in positions.items()
        if p.get('symbol', pid) == pair_key and p.get('in_position') is True
    }


def apply_decision(sig: dict, open_positions: dict, pair_capital: float) -> dict:
    """
    Apply stat arb decision rules:
      - Flat + |z| > entry  → ENTRY_LONG or ENTRY_SHORT
      - In position + |z| > stop_z → STOP
      - In position + bars_held >= max_holding → EXIT
      - In position + |z| < exit_z → EXIT
      - Otherwise in position → HOLD
    """
    z      = sig.get('z')
    params = sig['params']
    entry_z  = float(params['entry'])
    exit_z   = float(params['exit_z'])
    stop_z   = float(params['stop_z'])
    max_days = int(params['max_holding'])

    in_position = bool(open_positions)

    base = {
        'size_usd':            pair_capital,
        'leverage_multiplier': 1.0,
        'z_at_decision':       z,
    }

    if z is None:
        return {**base, 'decision': 'FLAT', 'size_usd': None,
                'direction': None, 'exit_reason': None}

    if not in_position:
        if z > entry_z:
            return {**base, 'decision': 'ENTRY_SHORT', 'direction': 'short',
                    'exit_reason': None,
                    'entry_reason': f'z={z:.2f} > entry_z={entry_z:.2f} → short spread'}
        if z < -entry_z:
            return {**base, 'decision': 'ENTRY_LONG', 'direction': 'long',
                    'exit_reason': None,
                    'entry_reason': f'z={z:.2f} < -entry_z={-entry_z:.2f} → long spread'}
        return {**base, 'decision': 'FLAT', 'size_usd': None,
                'direction': None, 'exit_reason': None}

    primary = next(iter(open_positions.values()), {})

    # Hard stop
    if abs(z) > stop_z:
        return {**base, 'decision': 'STOP', 'direction': primary.get('direction'),
                'exit_reason': f'Stop loss: |z|={abs(z):.2f} > stop_z={stop_z:.2f}'}

    # Max holding
    entry_date_str = primary.get('entry_date', '')
    if entry_date_str:
        try:
            held_days = (date.today() - datetime.strptime(entry_date_str, '%Y-%m-%d').date()).days
            if held_days >= max_days:
                return {**base, 'decision': 'EXIT', 'direction': primary.get('direction'),
                        'exit_reason': f'Max holding: {held_days}d ≥ {max_days}d'}
        except Exception:
            pass

    # Mean-reversion exit
    if abs(z) < exit_z:
        return {**base, 'decision': 'EXIT', 'direction': primary.get('direction'),
                'exit_reason': f'Mean reverted: |z|={abs(z):.2f} < exit_z={exit_z:.2f}'}

    return {**base, 'decision': 'HOLD', 'direction': primary.get('direction'),
            'exit_reason': None}


# ══════════════════════════════════════════════════════════════════════════════
#  Capital allocation
# ══════════════════════════════════════════════════════════════════════════════

def get_pair_capital(pair_key: str) -> float:
    """Dollar capital allocated to this pair."""
    if pair_key in COIN_WEIGHTS:
        weight = COIN_WEIGHTS[pair_key]
    else:
        allocated    = sum(COIN_WEIGHTS.values())
        remaining    = 1.0 - allocated
        n_unweighted = sum(1 for a in ACTIVE_ASSETS if a not in COIN_WEIGHTS)
        weight       = remaining / n_unweighted if n_unweighted > 0 else 0.0
    return CAPITAL * weight


# ══════════════════════════════════════════════════════════════════════════════
#  File I/O
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_live_params() -> dict:
    if not os.path.exists(LIVE_PARAMS_PATH):
        return {}
    with open(LIVE_PARAMS_PATH) as f:
        return json.load(f)


def load_positions() -> dict:
    if not os.path.exists(POSITIONS_PATH):
        return {}
    with open(POSITIONS_PATH) as f:
        content = f.read().strip()
    return json.loads(content) if content else {}


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_dashboard(pair_keys: list, live_params: dict, positions: dict) -> dict:
    """
    Run the full computation pipeline for all pairs.

    apply_decision() is NOT called here — decisions must stay outside any
    cache in streamlit_app.py so positions.json changes take effect immediately.
    """
    pair_rows   = []
    signal_date = None

    for pair_key in pair_keys:
        lp_entry  = live_params[pair_key]
        symbol_y  = lp_entry['symbol_y']
        symbol_x  = lp_entry['symbol_x']
        params    = lp_entry['params']
        strategy  = lp_entry.get('strategy', 'stat_arb_spread')
        fixed_keys = set(lp_entry.get('fixed_param_keys', []))

        pair_df     = fetch_pair_data(symbol_y, symbol_x, INDICATOR_WARMUP)
        signal_date = pair_df.index[-1].date()
        sig         = compute_pair_signal(pair_df, params, strategy)

        hourly_y   = fetch_hourly_recent(symbol_y, days=3)
        hourly_x   = fetch_hourly_recent(symbol_x, days=3)
        exec_y     = get_execution_price(hourly_y, signal_date, EXECUTION_HOUR)
        exec_x     = get_execution_price(hourly_x, signal_date, EXECUTION_HOUR)

        pair_cap    = get_pair_capital(pair_key)
        pair_weight = COIN_WEIGHTS.get(pair_key, pair_cap / CAPITAL)

        pair_rows.append({
            'pair_key':     pair_key,
            'symbol_y':     symbol_y,
            'symbol_x':     symbol_x,
            'strategy':     strategy,
            'sig':          sig,
            'exec_y':       exec_y,
            'exec_x':       exec_x,
            'all_params':   params,
            'fixed_keys':   fixed_keys,
            'optimised_on': lp_entry.get('optimised_on', 'unknown'),
            'pair_capital': pair_cap,
            'pair_weight':  pair_weight,
        })

    return {
        'signal_date':  signal_date,
        'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'pairs':        pair_rows,
    }
