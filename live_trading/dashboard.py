import os
import sys
import json
import webbrowser
from datetime import datetime, timezone

import pandas as pd
from jinja2 import Environment, FileSystemLoader

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LT_DIR = os.path.dirname(os.path.abspath(__file__))   # live_trading/
sys.path.insert(0, _LT_DIR)
sys.path.append(os.path.join(ROOT, 'infrastructure', 'data'))
# ─────────────────────────────────────────────────────────────────────────────

from binance_client import get_binance_client, get_data
from strategies import STRATEGY_REGISTRY
from config import ACTIVE_ASSETS, INDICATOR_WARMUP, EXECUTION_HOUR, CAPITAL, COIN_WEIGHTS

LIVE_PARAMS_PATH  = os.path.join(_LT_DIR, 'live_params.json')
POSITIONS_PATH    = os.path.join(_LT_DIR, 'positions.json')
TEMPLATES_DIR     = os.path.join(_LT_DIR, 'templates')
OUTPUT_HTML       = os.path.join(_LT_DIR, 'outputs', 'dashboard.html')


# ══════════════════════════════════════════════════════════════════════════════
#  Data fetching
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(symbol, warmup_bars=INDICATOR_WARMUP):
    """
    Fetch daily OHLCV — enough bars to warm up indicators.

    The last bar returned by Binance is often the *current* incomplete candle
    (opened at 00:00 UTC today, not yet closed).  Strip it so the strategy
    always runs on fully-closed bars only.

    A daily bar is only complete if its open timestamp is more than 24 hours
    in the past.  Comparing against a 24-hour cutoff is timezone-safe and does
    not depend on matching calendar dates.
    """
    client = get_binance_client()
    df     = get_data(client, symbol, interval='1d', lookback=warmup_bars + 10)

    # ── Strip incomplete bar: open timestamp < 24 h ago means bar not yet closed
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


def fetch_hourly_recent(symbol, days=3):
    """
    Fetch the last `days` days of 1h OHLCV for a symbol.
    Uses client.get_historical_klines directly, matching the approach in
    topics/momentum/results/portfolio_2.ipynb (cell: "fetch 1h data").

    Returns a DataFrame indexed by UTC-aware datetime with float OHLCV columns.
    3 days = ~72 bars — enough buffer if the latest hourly bar isn't available yet.
    """
    client = get_binance_client()
    klines = client.get_historical_klines(symbol, '1h', f'{days} days ago UTC')

    df = pd.DataFrame(klines, columns=[
        'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_volume', 'Trades', 'Taker_base', 'Taker_quote', 'Ignore'
    ])
    df['Time'] = pd.to_datetime(df['Time'], unit='ms', utc=True)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].astype(float)
    return df.set_index('Time')


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

    position_size = float(last['position_size_raw'])

    # Identify all failing conditions when entry is blocked
    entry_reasons = []
    if not entry_long:
        if not close_above_ema:
            entry_reasons.append('Close below EMA')
        if caution_long and not adx_strong:
            entry_reasons.append('ADX below threshold')
        if 'Vol_MA' in result_df.columns and not bool(last['Volume'] > last['Vol_MA']):
            entry_reasons.append('Volume below MA')
        if not entry_reasons:
            entry_reasons.append('Caution active')

    sig = {
        'close':            close,
        'ema':              ema,
        'close_ema_ratio':  close / ema if ema != 0 else None,
        'close_above_ema':  close_above_ema,
        'adx':              adx,
        'adx_threshold':    adx_threshold,
        'adx_strong':       adx_strong,
        'has_vol_ma':       'Vol_MA' in result_df.columns,
        'caution_long':     caution_long,
        'caution_obv':      caution_obv,
        'caution_short':    caution_short,
        'entry_long':       entry_long,
        'entry_reasons':    entry_reasons,   # empty list when entry fires
        'theoretical_stop': theoretical_stop,
        'position_size':    position_size,   # capital multiplier (e.g. 2.25x)
        'stop_detail': {
            # raw inputs
            'swing_hi_stp':          swing_stp,
            'atr_stp':               atr_stp,
            'stop_atr_scale':        stop_atr_scale,
            # entry stop (Day 1)
            'entry_path':            entry_path,
            'entry_multiplier':      entry_mult,
            'entry_stop':            entry_stop,
            # hold stop (Day 2+ ratchet candidate)
            'hold_path':             hold_path,
            'hold_multiplier':       hold_mult,
            'hold_stop_candidate':   hold_stop_candidate,
            # populated by apply_decision when in position
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
#
#  Pure function — no file I/O, no side effects. positions.json is loaded once
#  in main() and passed in; it is never written here. dashboard.py is read-only
#  with respect to positions.json. Only journal_server.py (Layer 5) writes back
#  after a trade is actually logged. This makes the dashboard fully idempotent.
#
#  To test HOLD and EXIT paths without live positions, temporarily add an entry
#  to positions.json by hand, run dashboard.py, then remove it.
#  The key is position_id (== symbol for single-asset strategies):
#    echo '{"ETHUSDT": {"in_position": true, "entry_price": 2000, \
#           "entry_date": "2026-04-01", "current_stop": 1800, "size": 0.5}}' \
#         > live_trading/positions.json
# ══════════════════════════════════════════════════════════════════════════════

def apply_decision(sig, position, exec_price, capital):
    """
    Apply the four decision rules and compute sizing.

    Parameters
    ----------
    sig        : dict from compute_signals()
    position   : dict from positions.json for this position_id, or None
    exec_price : float | None — theoretical execution price (T+1 HH:00 UTC)
    capital    : float — CAPITAL from config.py

    Returns a dict to be merged into sig:
        decision, current_stop, stop_updated,
        size_pct, size_usd, size_units

    Decision rules (applied in order):
        1. NOT in position AND entry_long  → ENTRY
        2. IN position  AND entry_long     → HOLD  (ratchet stop only upward)
        3. IN position  AND NOT entry_long → EXIT
        4. NOT in position AND NOT entry   → FLAT
    """
    in_position      = bool(position.get('in_position', False)) if position else False
    entry_long       = sig['entry_long']
    theoretical_stop = sig['theoretical_stop']
    stop_detail      = dict(sig['stop_detail'])   # shallow copy — we'll update hold fields

    # ── ENTRY ─────────────────────────────────────────────────────────────────
    if not in_position and entry_long:
        price_for_units = exec_price if exec_price is not None else sig['close']
        size_pct   = sig['position_size']
        size_usd   = size_pct * capital
        size_units = size_usd / price_for_units
        return {
            'decision':     'ENTRY',
            'current_stop': theoretical_stop,
            'stop_updated': False,
            'size_pct':     size_pct,
            'size_usd':     size_usd,
            'size_units':   size_units,
            'stop_detail':  stop_detail,            # hold fields remain None/False
        }

    # ── HOLD ──────────────────────────────────────────────────────────────────
    if in_position and entry_long:
        old_stop     = float(position['current_stop'])
        candidate    = stop_detail['hold_stop_candidate']
        new_stop     = max(old_stop, candidate)     # never move stop down
        stop_updated = new_stop > old_stop
        size_pct     = position['size']
        size_usd     = size_pct * capital
        stop_detail['hold_stop_previous'] = old_stop
        stop_detail['hold_stop_final']    = new_stop
        stop_detail['hold_stop_updated']  = stop_updated
        return {
            'decision':     'HOLD',
            'current_stop': new_stop,
            'stop_updated': stop_updated,
            'old_stop':     old_stop,               # kept for display diff
            'size_pct':     size_pct,
            'size_usd':     size_usd,
            'size_units':   None,                   # already in position
            'stop_detail':  stop_detail,
        }

    # ── EXIT ──────────────────────────────────────────────────────────────────
    if in_position and not entry_long:
        return {
            'decision':     'EXIT',
            'current_stop': None,
            'stop_updated': False,
            'size_pct':     None,
            'size_usd':     None,
            'size_units':   None,
            'stop_detail':  stop_detail,
        }

    # ── FLAT ──────────────────────────────────────────────────────────────────
    return {
        'decision':     'FLAT',
        'current_stop': None,
        'stop_updated': False,
        'size_pct':     None,
        'size_usd':     None,
        'size_units':   None,
        'stop_detail':  stop_detail,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Print helpers
# ══════════════════════════════════════════════════════════════════════════════

def _b(flag):
    return 'TRUE' if flag else 'FALSE'


def _fv(v):
    """Format a param value — ints as int, floats to 4 significant figures."""
    if isinstance(v, float):
        return f'{v:.4g}'
    return str(v)


def _param_lines(param_dict, per_line=3, indent='    '):
    """Format a param dict as pipe-separated lines, `per_line` entries each."""
    items = [f'{k}: {_fv(v)}' for k, v in param_dict.items()]
    lines = []
    for i in range(0, len(items), per_line):
        lines.append(indent + '  |  '.join(items[i:i + per_line]))
    return '\n'.join(lines)


def _print_coin_block(symbol, sig, exec_price, lp_entry, CAPITAL, EXECUTION_HOUR):
    """Print the full signal block for one coin, including decision."""
    W = 58
    decision = sig['decision']

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n── {symbol} {'─' * (W - len(symbol) - 4)}")

    exec_str = (
        f"{exec_price:>12,.2f}  ({EXECUTION_HOUR:02d}h UTC T+1)"
        if exec_price is not None
        else f"{'pending':>12}  ({EXECUTION_HOUR:02d}h UTC T+1 — not yet available)"
    )
    print(f"  Close:       {sig['close']:>12,.2f}")
    print(f"  Exec price:  {exec_str}")

    # ── Decision (most actionable — shown before indicator detail) ────────────
    print(f"\n  Decision:    ► {decision}")
    if decision == 'ENTRY':
        exec_tag = f"${exec_price:,.2f}" if exec_price is not None else "pending"
        print(f"    Size:      {sig['size_pct']:.2f}x  (${sig['size_usd']:,.0f})  "
              f"|  Units: {sig['size_units']:.4f}")
        print(f"    Stop:      ${sig['current_stop']:,.2f}  |  Exec: {exec_tag}")

    elif decision == 'HOLD':
        if sig['stop_updated']:
            print(f"    Stop:      ${sig['old_stop']:,.2f}  →  ${sig['current_stop']:,.2f}  (ratcheted up)")
        else:
            print(f"    Stop:      ${sig['current_stop']:,.2f}  (unchanged)")
        print(f"    Carry size: {sig['size_pct']:.2f}x  (${sig['size_usd']:,.0f})")

    elif decision == 'EXIT':
        print(f"    Stop:      None  —  close position at exec price")

    else:  # FLAT
        print(f"    —  no position, no signal")

    # ── Entry conditions ──────────────────────────────────────────────────────
    print(f"\n  Entry conditions:")
    ratio = f"{sig['close_ema_ratio']:.3f}" if sig['close_ema_ratio'] else "N/A"
    print(f"    EMA:        {sig['ema']:>12,.2f}  |  Close/EMA: {ratio:<8}  |  Above:  {_b(sig['close_above_ema'])}")
    print(f"    ADX:        {sig['adx']:>12.1f}  |  Threshold: {sig['adx_threshold']:<8}  |  Strong: {_b(sig['adx_strong'])}")

    if sig['has_vol_ma']:
        ratio_str = f"{sig['vol_vol_ma_ratio']:.2f}" if sig['vol_vol_ma_ratio'] is not None else "N/A"
        print(f"    Vol/VolMA:  {ratio_str:>12}                           |  Above:  {_b(sig['vol_above_ma'])}")

    reason_str = f"  ({', '.join(sig['entry_reasons'])})" if sig['entry_reasons'] else ''
    print(f"    Entry_Long: {_b(sig['entry_long'])}{reason_str}")

    # ── Caution flags ─────────────────────────────────────────────────────────
    print(f"\n  Caution flags:")
    print(f"    Caution_Long:  {_b(sig['caution_long'])}  |  "
          f"Caution_Short: {_b(sig['caution_short'])}  |  "
          f"Caution_OBV: {_b(sig['caution_obv'])}")

    # ── Stop detail ───────────────────────────────────────────────────────────
    sd       = sig['stop_detail']
    notional = sig['position_size'] * CAPITAL
    print(f"\n  Stop detail:")
    print(f"    Inputs:       Swing_Hi_Stp: {sd['swing_hi_stp']:,.2f}  |  "
          f"ATR_Stp: {sd['atr_stp']:,.2f}  |  "
          f"stop_atr_scale: {sd['stop_atr_scale']:.4g}")
    print(f"    Entry stop:   path: {sd['entry_path']:<12}  "
          f"mult: {sd['entry_multiplier']:.4g}  |  "
          f"stop: {sd['entry_stop']:,.2f}")
    print(f"    Hold stop:    path: {sd['hold_path']:<12}  "
          f"mult: {sd['hold_multiplier']:.4g}  |  "
          f"candidate: {sd['hold_stop_candidate']:,.2f}", end='')
    if sd['hold_stop_previous'] is not None:
        updated_tag = '  (ratcheted up)' if sd['hold_stop_updated'] else '  (unchanged)'
        print(f"\n                  previous: {sd['hold_stop_previous']:,.2f}  →  "
              f"final: {sd['hold_stop_final']:,.2f}{updated_tag}", end='')
    print()
    print(f"    Position size: {sig['position_size']:.2f}x  (${notional:,.0f})")

    # ── Params ────────────────────────────────────────────────────────────────
    all_params   = lp_entry['params']
    optimised_on = lp_entry.get('optimised_on', 'unknown')
    fixed_keys   = set(lp_entry.get('fixed_param_keys', []))

    if fixed_keys:
        fixed_dict = {k: all_params[k] for k in all_params if k in fixed_keys}
        optim_dict = {k: all_params[k] for k in all_params if k not in fixed_keys}
        print(f"\n  Fixed params:")
        print(_param_lines(fixed_dict))
        print(f"\n  Optimised params (last run: {optimised_on}):")
        print(_param_lines(optim_dict))
    else:
        # fixed_param_keys absent — re-run optimise.py to get the split
        print(f"\n  Params (last run: {optimised_on}):")
        print(_param_lines(all_params))


# ══════════════════════════════════════════════════════════════════════════════
#  Capital allocation
# ══════════════════════════════════════════════════════════════════════════════

def get_coin_capital(symbol):
    """
    Return the dollar capital allocated to this coin.

    Uses COIN_WEIGHTS from config.py.  Coins not listed there share the
    remaining unallocated weight equally.
    """
    if symbol in COIN_WEIGHTS:
        weight = COIN_WEIGHTS[symbol]
    else:
        allocated     = sum(COIN_WEIGHTS.values())
        remaining     = 1.0 - allocated
        n_unweighted  = sum(1 for a in ACTIVE_ASSETS if a not in COIN_WEIGHTS)
        weight        = remaining / n_unweighted if n_unweighted > 0 else 0.0
    return CAPITAL * weight


# ══════════════════════════════════════════════════════════════════════════════
#  HTML rendering
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_param(v):
    """Jinja2 filter: format a param value for HTML tables — max 2dp."""
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f'{v:.2f}'
    return str(v)


def render_html(coin_rows, signal_date, generated_at):
    """
    Render the Jinja2 dashboard template and write outputs/dashboard.html.

    coin_rows: list of dicts, one per active coin, each containing:
        symbol, sig, exec_price, fixed_params, optim_params, optimised_on,
        coin_capital, coin_weight
    Per-coin capital allocation is carried in coin_rows — no global needed.
    """
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=True)
    env.filters['fmt_param'] = _fmt_param
    tmpl = env.get_template('dashboard.html')

    # Union of fixed param keys across all coins (sorted for stable column order)
    all_fixed_keys = sorted({
        k for c in coin_rows for k in c['fixed_params']
    })
    # Union of optimised param keys across all coins
    all_optim_keys = sorted({
        k for c in coin_rows for k in c['optim_params']
    })

    html = tmpl.render(
        coins          = coin_rows,
        signal_date    = signal_date,
        generated_at   = generated_at,
        execution_hour = EXECUTION_HOUR,
        capital        = CAPITAL,
        all_fixed_keys = all_fixed_keys,
        all_optim_keys = all_optim_keys,
    )

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nHTML dashboard written → {OUTPUT_HTML}")
    webbrowser.open(f'file://{OUTPUT_HTML}')


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def load_live_params():
    """Load live_params.json. Returns {} if file is missing or empty."""
    if not os.path.exists(LIVE_PARAMS_PATH):
        return {}
    with open(LIVE_PARAMS_PATH) as f:
        return json.load(f)


def load_positions():
    """
    Load positions.json. Returns {} if file is missing or empty.

    Read-only — only journal_server.py (Layer 5) writes back.
    Keys are position_ids (== symbol for single-asset strategies).
    """
    if not os.path.exists(POSITIONS_PATH):
        return {}
    with open(POSITIONS_PATH) as f:
        return json.load(f)


def main():
    live_params = load_live_params()

    if not live_params:
        print("live_params.json is empty — run optimise.py first")
        return

    # ── Load current positions (read-only — journal_server.py writes back) ────
    positions = load_positions()

    # Warn about active assets with no params
    for asset in ACTIVE_ASSETS:
        if asset not in live_params:
            base = asset.replace("USDT", "")
            print(f"WARNING: {asset} is active but has no params — run optimise.py --asset {base}")

    # Process only coins present in both ACTIVE_ASSETS and live_params
    active_set = set(ACTIVE_ASSETS)
    coins = [sym for sym in live_params if sym in active_set]

    coin_rows    = []
    summary_rows = []
    signal_date  = None

    for symbol in coins:
        lp_entry   = live_params[symbol]
        params     = lp_entry['params']
        strategy   = lp_entry['strategy']
        fixed_keys = set(lp_entry.get('fixed_param_keys', []))

        # ── Daily OHLCV & signals ─────────────────────────────────────────────
        daily_df    = fetch_ohlcv(symbol)
        signal_date = daily_df.index[-1].date()
        sig         = compute_signals(daily_df, params, strategy)

        # ── Hourly & execution price ──────────────────────────────────────────
        hourly_df  = fetch_hourly_recent(symbol, days=3)
        exec_price = get_execution_price(hourly_df, signal_date, EXECUTION_HOUR)

        # ── Decision logic ────────────────────────────────────────────────────
        position_id = symbol                       # for single-asset strategies, position_id == symbol
        position = positions.get(position_id)     # None if no entry in positions.json
        coin_cap      = get_coin_capital(symbol)
        decision_dict = apply_decision(sig, position, exec_price, coin_cap)
        sig.update(decision_dict)                 # merge decision fields into sig

        # ── Per-coin print block ──────────────────────────────────────────────
        _print_coin_block(symbol, sig, exec_price, lp_entry, CAPITAL, EXECUTION_HOUR)

        # ── Accumulate for HTML and terminal summary ──────────────────────────
        all_params   = lp_entry['params']
        fixed_params = {k: all_params[k] for k in all_params if k in fixed_keys}
        optim_params = {k: all_params[k] for k in all_params if k not in fixed_keys}

        coin_rows.append({
            'symbol':       symbol,
            'sig':          sig,
            'exec_price':   exec_price,
            'fixed_params': fixed_params,
            'optim_params': optim_params,
            'optimised_on': lp_entry.get('optimised_on', 'unknown'),
            'coin_capital': coin_cap,
            'coin_weight':  COIN_WEIGHTS.get(symbol, coin_cap / CAPITAL),
        })
        summary_rows.append({
            'symbol':       symbol,
            'decision':     sig['decision'],
            'size_pct':     sig['size_pct'],
            'size_usd':     sig['size_usd'],
            'current_stop': sig['current_stop'],
            'old_stop':     sig.get('old_stop'),
            'stop_updated': sig['stop_updated'],
            'exec_price':   exec_price,
        })

    # ── Terminal decision summary ─────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("DECISION SUMMARY")
    print(f"{'═' * 72}")
    for r in summary_rows:
        decision = r['decision']
        ep_str   = f"${r['exec_price']:,.2f}" if r['exec_price'] is not None else "pending"

        if decision == 'ENTRY':
            detail = (
                f"size: {r['size_pct']:.2f}x (${r['size_usd']:,.0f})  "
                f"|  stop: ${r['current_stop']:,.2f}  "
                f"|  exec: {ep_str}"
            )
        elif decision == 'HOLD':
            if r['stop_updated']:
                stop_str = f"${r['old_stop']:,.2f} → ${r['current_stop']:,.2f} (updated)"
            else:
                stop_str = f"${r['current_stop']:,.2f} (unchanged)"
            detail = (
                f"stop: {stop_str}  "
                f"|  carry size: {r['size_pct']:.2f}x (${r['size_usd']:,.0f})"
            )
        elif decision == 'EXIT':
            detail = f"stop: None  |  exec: {ep_str}"
        else:  # FLAT
            detail = "—"

        print(f"  {r['symbol']:<10}  {decision:<5}  {detail}")

    # ── HTML render ───────────────────────────────────────────────────────────
    generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    render_html(coin_rows, str(signal_date), generated_at)


if __name__ == "__main__":
    main()
