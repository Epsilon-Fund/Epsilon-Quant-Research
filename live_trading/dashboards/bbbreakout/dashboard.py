"""
Epsilon Fund — Pure computation module for the BB Breakout live trading dashboard.

No display, no HTML, no print statements (except fetch confirmations and errors).
All trade logging is done in streamlit_app.py.

The BB Breakout strategy runs on 1H OHLCV data (resampled to 4H internally).
fetch_ohlcv() therefore reads from the hourly cache rather than the daily cache.

Exported public API
-------------------
  run_dashboard(coin_symbols, live_params, positions) → dict
  fetch_ohlcv(symbol, warmup_bars)                    → DataFrame  (hourly)
  fetch_hourly_recent(symbol, days)                   → DataFrame
  fetch_live_price(symbol)                            → float | None
  compute_signals(df, params, strategy)               → dict
  apply_decision(sig, open_positions, exec_price, capital) → dict
  get_open_positions(symbol, positions)               → dict
  get_execution_price(hourly_df, signal_date, hour_utc)   → float | None
  get_coin_capital(symbol)                            → float
  load_live_params()                                  → dict
  load_positions()                                    → dict
"""

import os
import sys
import json
from datetime import datetime, timezone, date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))           # dashboards/bbbreakout/
_LT_DIR  = os.path.abspath(os.path.join(_DASHBOARD_DIR, '..', '..')) # live_trading/
ROOT     = os.path.dirname(_LT_DIR)                                    # repo root
_INFRA   = os.path.join(ROOT, 'infrastructure', 'data')
for _p in (_DASHBOARD_DIR, _LT_DIR, _INFRA):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from binance_client import get_binance_client
from strategies import STRATEGY_REGISTRY
from config import ACTIVE_ASSETS, INDICATOR_WARMUP, CAPITAL, COIN_WEIGHTS

LIVE_PARAMS_PATH = os.path.join(_DASHBOARD_DIR, 'live_params.json')
POSITIONS_PATH   = os.path.join(_DASHBOARD_DIR, 'positions.json')


# ══════════════════════════════════════════════════════════════════════════════
#  Data fetching
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def fetch_live_price(symbol):
    """
    Fetch the current market price for a single symbol via the ticker endpoint.
    Very lightweight — one REST call, no OHLCV data fetched.
    Returns a float, or None on error.
    """
    try:
        client = get_binance_client()
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"  fetch_live_price({symbol}) failed: {e}")
        return None


def fetch_ohlcv(symbol, warmup_bars=INDICATOR_WARMUP):
    """
    Return hourly OHLCV — enough bars to warm up BB Breakout indicators.

    warmup_bars is interpreted as days of history to fetch (matches the
    momentum convention where warmup_bars == number of daily bars).

    Reads from the local parquet cache (live_trading/cache/hourly/).
    The last bar is stripped when its open timestamp is less than 1 h old
    (current incomplete candle). Strategy always runs on fully-closed bars.
    """
    from shared.cache_manager import get_hourly_ohlcv

    end   = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=warmup_bars + 10)

    df = get_hourly_ohlcv(symbol, start, end)

    if df.empty:
        raise RuntimeError(
            f'fetch_ohlcv: no hourly data for {symbol}. '
            'Run backfill_cache.py first.'
        )

    # Strip incomplete bar: open timestamp < 1 h ago → bar not yet closed
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=1)
    last_utc_naive = (
        df.index[-1].tz_convert('UTC').tz_localize(None)
        if df.index.tz is not None
        else df.index[-1]
    )
    if last_utc_naive > cutoff:
        df = df.iloc[:-1]

    last_bar   = df.index[-1]
    last_close = float(df['Close'].iloc[-1])
    print(f"  {symbol} last completed bar: {last_bar}  close: {last_close:,.2f}")

    return df


@st.cache_data(ttl=300)
def fetch_hourly_recent(symbol, days=3):
    """
    Return the last `days` days of 1h OHLCV for a symbol.

    Reads from the local parquet cache (live_trading/cache/hourly/).
    Used by get_execution_price() to look up the T+1 1H bar open.
    TTL matches load_all (5 min) so the next bar is visible as soon as it opens.

    Returns a DataFrame indexed by UTC-aware timestamps.
    """
    from shared.cache_manager import get_hourly_ohlcv

    start = date.today() - timedelta(days=days + 1)
    end   = date.today() + timedelta(days=1)

    df = get_hourly_ohlcv(symbol, start, end)

    if df.empty:
        return df

    # Localise to UTC to match get_execution_price's tz-aware lookups
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize('UTC')

    return df


def get_execution_price(hourly_df, signal_date):
    """
    Return the Open price of the T+1 1H bar immediately following signal_date.

    Signal fires on the close of bar T (signal_date).
    Execution at the open of bar T+1 — exactly one hour later.
    Matches the 1-bar execution lag the backtester applies on hourly data.

    Returns None if the T+1 bar hasn't opened yet or is missing from the cache.
    """
    exec_dt = pd.Timestamp(signal_date) + pd.Timedelta(hours=1)

    # Align tz to hourly_df index
    if hourly_df.index.tz is not None and exec_dt.tz is None:
        exec_dt = exec_dt.tz_localize('UTC')
    elif hourly_df.index.tz is None and exec_dt.tz is not None:
        exec_dt = exec_dt.tz_localize(None)

    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    if hourly_df.index.tz is None:
        now_utc = now_utc.tz_localize(None)

    if exec_dt > now_utc:
        return None  # next bar hasn't opened yet

    return float(hourly_df.loc[exec_dt, 'Open']) if exec_dt in hourly_df.index else None


# ══════════════════════════════════════════════════════════════════════════════
#  Signal computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_signals(hourly_df, params, strategy_name):
    """
    Run the BB Breakout strategy on hourly_df and extract the last-bar signal.

    BB-specific signal_detail fields
    ---------------------------------
    close          : last bar close price
    sma            : 1H SMA value (entry zone anchor)
    bb_mid         : BB centre (rolling mean over bb_period)
    bb_upper       : bb_mid + 2 × rolling std
    bb_lower       : bb_mid − 2 × rolling std
    bb_width       : (bb_upper − bb_lower) / bb_mid
    position       : -1 / 0 / 1 from the strategy state machine
    position_size  : leverage multiplier from risk-based sizing
    entry_long     : True if strategy just entered long on this bar
    entry_short    : True if strategy just entered short on this bar
    in_long        : True if position == 1
    in_short       : True if position == -1
    stop           : current trailing stop level (0.0 when flat)
    leverage_multiplier : alias of position_size for downstream compatibility
    """
    strategy_fn      = STRATEGY_REGISTRY[strategy_name]
    result_df, _     = strategy_fn(hourly_df.copy(), params)
    last             = result_df.iloc[-1]
    prev             = result_df.iloc[-2] if len(result_df) > 1 else last

    close         = float(last['Close'])
    sma           = float(last['SMA'])   if not pd.isna(last['SMA'])           else float('nan')
    position      = int(last['position'])
    position_size = float(last['position_size'])
    stop          = float(last['stop_loss'])

    # Bollinger Bands — computed over bb_period (same window as 4H setup uses
    # but applied here on 1H closes for display purposes)
    bb_period = int(params['bb_period'])
    bb_mid_s  = result_df['Close'].rolling(bb_period).mean()
    bb_std_s  = result_df['Close'].rolling(bb_period).std()
    bb_mid    = float(bb_mid_s.iloc[-1])  if not bb_mid_s.iloc[-1] != bb_mid_s.iloc[-1] else float('nan')
    bb_upper  = float((bb_mid_s + 2 * bb_std_s).iloc[-1])
    bb_lower  = float((bb_mid_s - 2 * bb_std_s).iloc[-1])
    bb_width  = (bb_upper - bb_lower) / bb_mid if bb_mid != 0 else float('nan')

    prev_position = int(prev['position'])
    entry_long    = (prev_position == 0 and position == 1)
    entry_short   = (prev_position == 0 and position == -1)
    in_long       = (position == 1)
    in_short      = (position == -1)

    return {
        'close':               close,
        'sma':                 sma,
        'bb_mid':              bb_mid,
        'bb_upper':            bb_upper,
        'bb_lower':            bb_lower,
        'bb_width':            bb_width,
        'position':            position,
        'position_size':       position_size,
        'entry_long':          entry_long,
        'entry_short':         entry_short,
        'in_long':             in_long,
        'in_short':            in_short,
        'stop':                stop,
        'leverage_multiplier': position_size,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Decision logic
# ══════════════════════════════════════════════════════════════════════════════

def get_open_positions(symbol, positions):
    """
    Return {position_id: pos} for all open positions of a symbol.

    Compatible with both old-format keys (key == symbol, no 'symbol' field)
    and new FIFO-format keys (e.g. "BTCUSDT_20260415_001", has 'symbol' field).
    """
    return {
        pid: p for pid, p in positions.items()
        if p.get('symbol', pid) == symbol and p.get('in_position') is True
    }


def apply_decision(sig, open_positions, exec_price, capital):
    """
    Apply BB Breakout decision rules.

    The strategy state machine determines position on every bar, so the
    signal's `position` field is the single source of truth:

      position != 0 and no open position  → ENTRY  (long if +1, short if -1)
      position == 0 and open position     → EXIT
      position != 0 and open position     → HOLD   (stop updated to sig['stop'])
      position == 0 and no open position  → FLAT

    open_positions: dict of {position_id: pos} from get_open_positions().
    """
    in_position       = bool(open_positions)
    primary           = next(iter(open_positions.values()), None)
    position_signal   = sig['position']
    close             = sig['close']
    current_stop      = sig['stop']
    leverage_mult     = sig['leverage_multiplier']

    # ── ENTRY ─────────────────────────────────────────────────────────────────
    if not in_position and position_signal != 0:
        price_for_units = exec_price if exec_price is not None else close
        size_usd        = leverage_mult * capital
        size_units      = size_usd / price_for_units if price_for_units else None
        return {
            'decision':            'ENTRY',
            'direction':           position_signal,   # +1 long, -1 short
            'current_stop':        current_stop,
            'stop_updated':        False,
            'leverage_multiplier': leverage_mult,
            'size_usd':            size_usd,
            'size_units':          size_units,
        }

    # ── FLAT ──────────────────────────────────────────────────────────────────
    if not in_position:
        return {
            'decision':            'FLAT',
            'direction':           0,
            'current_stop':        None,
            'stop_updated':        False,
            'leverage_multiplier': None,
            'size_usd':            None,
            'size_units':          None,
        }

    # ── In position branch ────────────────────────────────────────────────────
    confirmed_stop_raw = primary.get('current_stop')
    confirmed_stop     = float(confirmed_stop_raw) if confirmed_stop_raw is not None else None
    trade_direction    = primary.get('direction', 1)
    stored_lev         = primary.get('leverage_multiplier') or primary.get('size_pct', 0)
    size_usd           = stored_lev * capital

    # ── EXIT — strategy state machine exited the trade ────────────────────────
    if position_signal == 0:
        print(f"  [EXIT trigger] strategy position=0  confirmed_stop={confirmed_stop}")
        return {
            'decision':            'EXIT',
            'direction':           trade_direction,
            'current_stop':        confirmed_stop,
            'stop_updated':        False,
            'leverage_multiplier': stored_lev,
            'size_usd':            size_usd,
            'size_units':          None,
        }

    # ── HOLD — ratchet trailing stop from strategy ────────────────────────────
    old_stop    = confirmed_stop if confirmed_stop else 0.0
    new_stop    = current_stop   # strategy already ratcheted this bar
    stop_updated = (
        (trade_direction ==  1 and new_stop > old_stop) or
        (trade_direction == -1 and new_stop < old_stop and old_stop != 0.0)
    )
    return {
        'decision':            'HOLD',
        'direction':           trade_direction,
        'current_stop':        new_stop,
        'stop_updated':        stop_updated,
        'old_stop':            old_stop,
        'leverage_multiplier': stored_lev,
        'size_usd':            size_usd,
        'size_units':          None,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Capital allocation
# ══════════════════════════════════════════════════════════════════════════════

def get_coin_capital(symbol, data_dir=None):
    """
    Return the dollar capital allocated to this coin.

    Uses realised_capital (updated after every EXIT) rather than the static
    config CAPITAL.  Deployed positions are never subtracted — open positions
    for the same coin always see the full realised allocation.
    """
    if data_dir is None:
        data_dir = _DASHBOARD_DIR
    from shared.data_loader import load_realised_capital
    realised = load_realised_capital(data_dir)
    if symbol in COIN_WEIGHTS:
        weight = COIN_WEIGHTS[symbol]
    else:
        allocated    = sum(COIN_WEIGHTS.values())
        remaining    = 1.0 - allocated
        n_unweighted = sum(1 for a in ACTIVE_ASSETS if a not in COIN_WEIGHTS)
        weight       = remaining / n_unweighted if n_unweighted > 0 else 0.0
    return realised * weight


# ══════════════════════════════════════════════════════════════════════════════
#  File I/O
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_live_params():
    """Load live_params.json. Returns {} if file is missing or empty."""
    if not os.path.exists(LIVE_PARAMS_PATH):
        return {}
    with open(LIVE_PARAMS_PATH) as f:
        return json.load(f)


def load_positions():
    """Load positions.json. Returns {} if file is missing or empty."""
    if not os.path.exists(POSITIONS_PATH):
        return {}
    with open(POSITIONS_PATH) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_dashboard(coin_symbols, live_params, positions):
    """
    Run the full computation pipeline for all assets.

    NOTE: apply_decision() is NOT called here — decisions must stay outside
    any cache in streamlit_app.py because positions.json changes when trades
    are logged and must be read fresh every render.
    """
    from shared.data_loader import load_realised_capital
    _realised = load_realised_capital(_DASHBOARD_DIR)

    coin_rows   = []
    signal_date = None

    for symbol in coin_symbols:
        lp_entry = live_params[symbol]
        params   = lp_entry['params']
        strategy = lp_entry['strategy']

        fixed_keys = set(lp_entry.get('fixed_param_keys', []))

        hourly_df   = fetch_ohlcv(symbol)
        signal_date = hourly_df.index[-1]

        sig = compute_signals(hourly_df, params, strategy)

        hourly_recent = fetch_hourly_recent(symbol, days=3)
        exec_price    = get_execution_price(hourly_recent, signal_date)

        coin_cap = get_coin_capital(symbol, _DASHBOARD_DIR)

        coin_rows.append({
            'symbol':       symbol,
            'strategy':     strategy,
            'sig':          sig,
            'exec_price':   exec_price,
            'all_params':   params,
            'fixed_keys':   fixed_keys,
            'optimised_on': lp_entry.get('optimised_on', 'unknown'),
            'coin_capital': coin_cap,
            'coin_weight':  COIN_WEIGHTS.get(symbol, coin_cap / _realised if _realised > 0 else 0.0),
        })

    return {
        'signal_date':  signal_date,
        'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'assets':       coin_rows,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Smoke test — python dashboard.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import pprint

    # Minimal test params matching the NEAR notebook fixed + mid-range search params
    TEST_PARAMS = {
        'NEARUSDT': {
            'strategy': 'bb_breakout',
            'params': {
                'bb_period':         20,
                'bb_exp_window':     10,
                'atr_period':        5,
                'breakout_pct':      0.7113,
                'breakout_lookback': 40,
                'h4_ma_period':      20,
                'slope_epsilon':     0.001,
                'h1_ma_period':      10,
                'entry_zone_bps':    50,
                'overshoot_bps':     99,
                'max_1h_bars':       23,
                'pullback_atr_mult': 2.717,
                'trail_atr_mult':    2.7646,
                'adx_period':        14,
                'adx_strong':        30.0,
                'trend_ma_period':   220,
                'risk_per_trade':    0.03,
                'max_leverage':      2.5,
            },
            'fixed_param_keys': [
                'atr_period', 'breakout_pct', 'overshoot_bps',
                'pullback_atr_mult', 'trail_atr_mult', 'max_1h_bars',
                'trend_ma_period', 'risk_per_trade', 'max_leverage',
            ],
        }
    }

    print("Fetching hourly data for NEARUSDT …")
    result = run_dashboard(['NEARUSDT'], TEST_PARAMS, {})

    print(f"\nsignal_date : {result['signal_date']}")
    print(f"generated_at: {result['generated_at']}")
    print("\n── signal_detail ──────────────────────────────────────────")
    pprint.pprint(result['assets'][0]['sig'])
