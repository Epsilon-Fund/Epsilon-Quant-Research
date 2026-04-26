"""
Epsilon Fund — Pure computation module for the live trading dashboard.

No display, no HTML, no print statements (except fetch confirmations and errors).
All trade logging is done in streamlit_app.py.

Exported public API
-------------------
  run_dashboard(coin_symbols, live_params, positions) → dict
  fetch_ohlcv(symbol, warmup_bars)     → DataFrame
  fetch_hourly_recent(symbol, days)    → DataFrame
  fetch_live_price(symbol)             → float | None
  compute_signals(df, params, strategy)→ dict
  apply_decision(sig, open_positions, exec_price, capital) → dict
  get_open_positions(symbol, positions)→ dict
  get_execution_price(hourly_df, signal_date, hour_utc)    → float | None
  get_coin_capital(symbol)             → float
  load_live_params()                   → dict
  load_positions()                     → dict
"""

import os
import sys
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))          # dashboards/momentum/
_LT_DIR  = os.path.abspath(os.path.join(_DASHBOARD_DIR, '..', '..'))  # live_trading/
ROOT     = os.path.dirname(_LT_DIR)                                    # repo root
_INFRA   = os.path.join(ROOT, 'infrastructure', 'data')
for _p in (_DASHBOARD_DIR, _LT_DIR, _INFRA):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from binance_client import get_binance_client, get_data
from strategies import STRATEGY_REGISTRY
from config import ACTIVE_ASSETS, INDICATOR_WARMUP, EXECUTION_HOUR, CAPITAL, COIN_WEIGHTS

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
    Return daily OHLCV — enough bars to warm up indicators.

    Reads from the local parquet cache (live_trading/cache/daily/).
    Falls back to a live Binance fetch on cache miss (first run or after
    running backfill_cache.py).

    The last bar is stripped when its open timestamp is less than 24 h old
    (current incomplete candle).  Strategy always runs on fully-closed bars.
    """
    from shared.cache_manager import get_daily_ohlcv
    df = get_daily_ohlcv(symbol, warmup_bars=warmup_bars + 10)

    if df.empty:
        raise RuntimeError(
            f'fetch_ohlcv: no daily data for {symbol}. '
            'Run backfill_cache.py first.'
        )

    # Strip incomplete bar: open timestamp < 24 h ago → bar not yet closed
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=24)
    last_utc_naive = (
        df.index[-1].tz_convert('UTC').tz_localize(None)
        if df.index.tz is not None
        else df.index[-1]
    )
    if last_utc_naive > cutoff:
        df = df.iloc[:-1]

    last_bar   = df.index[-1]
    last_close = float(df['Close'].iloc[-1])
    print(f"  {symbol} last completed bar: {last_bar.date()}  close: {last_close:,.2f}")

    return df


@st.cache_data(ttl=3600)
def fetch_hourly_recent(symbol, days=3):
    """
    Return the last `days` days of 1h OHLCV for a symbol.

    Reads from the local parquet cache (live_trading/cache/hourly/).
    The cache is updated daily by update_cache.py; intra-day freshness is
    not critical since exec_price is only needed for signal display.

    Returns a DataFrame indexed by UTC-aware timestamps (same contract as
    the original Binance-fetching implementation) so get_execution_price()
    works without changes.
    """
    from shared.cache_manager import get_hourly_ohlcv
    from datetime import date, timedelta

    start = date.today() - timedelta(days=days + 1)
    end   = date.today() + timedelta(days=1)

    df = get_hourly_ohlcv(symbol, start, end)

    if df.empty:
        return df

    # Cache stores tz-naive timestamps; localise to UTC to match the original
    # function's contract (get_execution_price looks up tz-aware exec_dt).
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize('UTC')

    return df


def get_execution_price(hourly_df, signal_date, hour_utc):
    """
    Return the Open price at hour_utc UTC on signal_date + 1 day.

    Signal fires on the close of signal_date (day T).
    Execution is at HH:00 UTC on day T+1.

    Returns None if the T+1 HH:00 bar is in the future or missing.
    """
    exec_dt = (
        pd.Timestamp(signal_date, tz='UTC')
        + pd.Timedelta(days=1)
    ).replace(hour=hour_utc, minute=0, second=0, microsecond=0)

    if exec_dt > pd.Timestamp(datetime.now(timezone.utc)):
        return None  # bar not yet open

    return float(hourly_df.loc[exec_dt, 'Open']) if exec_dt in hourly_df.index else None


# ══════════════════════════════════════════════════════════════════════════════
#  Signal computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_signals(daily_df, params, strategy_name):
    """
    Run the strategy function on daily_df and extract the last-bar signal state.

    Returns a dict of indicator values, condition flags, and derived quantities
    (theoretical stop, position size). Separates signal computation from display
    so the dict can also be used for downstream logging.
    """
    strategy_fn       = STRATEGY_REGISTRY[strategy_name]
    result_df, _      = strategy_fn(daily_df.copy(), params)
    last              = result_df.iloc[-1]

    close     = float(last['Close'])
    ema       = float(last['EMA'])
    adx       = float(last['ADX_14'])
    atr_stp   = float(last['ATR_Stp'])
    swing_stp = float(last['Swing_Hi_Stp'])

    close_above_ema = bool(close > ema)
    adx_threshold   = int(params['adx_override'])
    adx_strong      = bool(adx > adx_threshold)
    caution_long    = bool(last['Caution_Long'])
    caution_obv     = bool(last['Caution_OBV'])
    caution_short   = bool(last['Caution_Short'])
    entry_long      = bool(last['Entry_Long'])

    stop_atr_scale = float(params['stop_atr_scale'])

    # ── Entry stop path ───────────────────────────────────────────────────────
    if caution_long and caution_short:
        entry_path = 'ent_both';    entry_mult = float(params['stop_mult_ent_both'])
    elif caution_long:
        entry_path = 'ent_caution'; entry_mult = float(params['stop_mult_ent_caution'])
    else:
        entry_path = 'ent_normal';  entry_mult = float(params['stop_mult_ent_normal'])
    entry_stop       = swing_stp - atr_stp * entry_mult * stop_atr_scale
    theoretical_stop = entry_stop   # alias kept for downstream compatibility

    # ── Hold stop path (Day 2+ ratchet candidate) ─────────────────────────────
    if caution_long:
        hold_path = 'pos_caution'; hold_mult = float(params['stop_mult_pos_caution'])
    else:
        hold_path = 'pos_normal';  hold_mult = float(params['stop_mult_pos_normal'])
    hold_stop_candidate = swing_stp - atr_stp * hold_mult * stop_atr_scale

    leverage_multiplier = float(last['position_size_raw'])

    entry_reasons = []
    if not entry_long:
        if not close_above_ema:
            entry_reasons.append('Close below EMA')
        if caution_long and not adx_strong:
            entry_reasons.append('Caution Long')
        if caution_obv and not adx_strong:
            entry_reasons.append('Caution OBV')
        if 'Vol_MA' in result_df.columns and not bool(last['Volume'] > last['Vol_MA']):
            entry_reasons.append('Volume below MA')
        if not entry_reasons:
            entry_reasons.append('Caution active')
    else:
        if (caution_long or caution_obv) and adx_strong:
            entry_reasons.append('ADX override')

    sig = {
        'close':               close,
        'ema':                 ema,
        'close_ema_ratio':     close / ema if ema != 0 else None,
        'close_above_ema':     close_above_ema,
        'adx':                 adx,
        'adx_threshold':       adx_threshold,
        'adx_strong':          adx_strong,
        'has_vol_ma':          'Vol_MA' in result_df.columns,
        'caution_long':        caution_long,
        'caution_obv':         caution_obv,
        'caution_short':       caution_short,
        'entry_long':          entry_long,
        'entry_reasons':       entry_reasons,
        'theoretical_stop':    theoretical_stop,
        'leverage_multiplier': leverage_multiplier,
        'stop_detail': {
            'swing_hi_stp':          swing_stp,
            'atr_stp':               atr_stp,
            'stop_atr_scale':        stop_atr_scale,
            'entry_path':            entry_path,
            'entry_multiplier':      entry_mult,
            'entry_stop':            entry_stop,
            'hold_path':             hold_path,
            'hold_multiplier':       hold_mult,
            'hold_stop_candidate':   hold_stop_candidate,
            'hold_stop_previous':    None,
            'hold_stop_final':       None,
            'hold_stop_updated':     False,
        },
    }

    if sig['has_vol_ma']:
        vol_ma = float(last['Vol_MA'])
        volume = float(last['Volume'])
        sig['vol_ma']           = vol_ma
        sig['vol_vol_ma_ratio'] = volume / vol_ma if vol_ma > 0 else None
        sig['vol_above_ma']     = bool(volume > vol_ma)

    return sig


# ══════════════════════════════════════════════════════════════════════════════
#  Decision logic
# ══════════════════════════════════════════════════════════════════════════════

def get_open_positions(symbol, positions):
    """
    Return {position_id: pos} for all open positions of a symbol.

    Compatible with both old-format keys (key == symbol, no 'symbol' field)
    and new FIFO-format keys (e.g. "ETHUSDT_20260415_001", has 'symbol' field).
    """
    return {
        pid: p for pid, p in positions.items()
        if p.get('symbol', pid) == symbol and p.get('in_position') is True
    }


def apply_decision(sig, open_positions, exec_price, capital):
    """
    Apply decision rules that exactly match the strategy's backtest loop.

    open_positions: dict of {position_id: pos} from get_open_positions().
                    Pass {} (empty dict) when not in position.
    """
    in_position = bool(open_positions)
    primary = next(iter(open_positions.values()), None)

    entry_long       = sig['entry_long']
    close            = sig['close']
    theoretical_stop = sig['theoretical_stop']
    stop_detail      = dict(sig['stop_detail'])

    if not in_position and entry_long:
        price_for_units     = exec_price if exec_price is not None else close
        leverage_multiplier = sig['leverage_multiplier']
        size_usd            = leverage_multiplier * capital
        size_units          = size_usd / price_for_units
        return {
            'decision':            'ENTRY',
            'current_stop':        theoretical_stop,
            'stop_updated':        False,
            'leverage_multiplier': leverage_multiplier,
            'size_usd':            size_usd,
            'size_units':          size_units,
            'stop_detail':         stop_detail,
        }

    if not in_position:
        return {
            'decision':            'FLAT',
            'current_stop':        None,
            'stop_updated':        False,
            'leverage_multiplier': None,
            'size_usd':            None,
            'size_units':          None,
            'stop_detail':         stop_detail,
        }

    leverage_multiplier = (primary.get('leverage_multiplier')
                           or primary.get('size_pct', 0))
    size_usd = leverage_multiplier * capital

    confirmed_stop_raw = primary.get('current_stop')
    confirmed_stop     = float(confirmed_stop_raw) if confirmed_stop_raw is not None else None

    if confirmed_stop and close < confirmed_stop:
        print(f"  [EXIT trigger] close={close:,.2f} < confirmed_stop={confirmed_stop:,.2f}")
        return {
            'decision':            'EXIT',
            'current_stop':        confirmed_stop,
            'stop_updated':        False,
            'leverage_multiplier': leverage_multiplier,
            'size_usd':            size_usd,
            'size_units':          None,
            'stop_detail':         stop_detail,
        }

    old_stop     = confirmed_stop if confirmed_stop else 0.0
    candidate    = stop_detail['hold_stop_candidate']
    new_stop     = max(old_stop, candidate)
    stop_updated = new_stop > old_stop
    stop_detail['hold_stop_previous'] = old_stop
    stop_detail['hold_stop_final']    = new_stop
    stop_detail['hold_stop_updated']  = stop_updated
    return {
        'decision':            'HOLD',
        'current_stop':        new_stop,
        'stop_updated':        stop_updated,
        'old_stop':            old_stop,
        'leverage_multiplier': leverage_multiplier,
        'size_usd':            size_usd,
        'size_units':          None,
        'stop_detail':         stop_detail,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Capital allocation
# ══════════════════════════════════════════════════════════════════════════════

def get_coin_capital(symbol, data_dir=None):
    """
    Return the dollar capital allocated to this coin.

    Uses realised_capital (updated after every EXIT) rather than the static
    config CAPITAL.  Deployed positions are never subtracted — both open
    positions for the same coin always see the same full realised allocation.
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
#  File I/O  (reads from this dashboard's own directory)
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

        daily_df    = fetch_ohlcv(symbol)
        signal_date = daily_df.index[-1].date()
        sig         = compute_signals(daily_df, params, strategy)

        hourly_df  = fetch_hourly_recent(symbol, days=3)
        exec_price = get_execution_price(hourly_df, signal_date, EXECUTION_HOUR)

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
