"""
Centralised data loading for all Streamlit dashboards.

Every public function accepts a ``data_dir`` argument — the directory that
contains the dashboard's trades.json, positions.json, live_params.json,
mae_cache.json, and config.py.  Pass it as:

    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    trades   = load_trades(DATA_DIR)

This lets the same shared module serve momentum/, statarb/, bbbreakout/,
and any future dashboard without duplicating file-read or config logic.
"""

import importlib.util
import json
import os
import re
import sys
from datetime import date, datetime, timedelta

import pandas as pd

# ── Path setup (for binance_client in compute_mae) ────────────────────────────
_SHARED_DIR = os.path.dirname(os.path.abspath(__file__))   # live_trading/shared/
_LT_DIR     = os.path.dirname(_SHARED_DIR)                  # live_trading/
_ROOT       = os.path.dirname(_LT_DIR)                      # repo root
_INFRA_DATA = os.path.join(_ROOT, 'infrastructure', 'data')
if _INFRA_DATA not in sys.path:
    sys.path.append(_INFRA_DATA)

# ── FIFO position_id regex ─────────────────────────────────────────────────────
_FIFO_PID_RE = re.compile(r'^([A-Z0-9]+)_\d{8}_\d{3}$')


def _pid_to_symbol(pid: str) -> str:
    """Extract symbol from a FIFO position_id ('BTCUSDT_20260415_001' → 'BTCUSDT').
    If pid is already a plain symbol returns it unchanged."""
    m = _FIFO_PID_RE.match(pid)
    return m.group(1) if m else pid


# ── Config loader (importlib, cached per data_dir) ────────────────────────────

_CONFIG_CACHE: dict = {}


def _load_config(data_dir: str):
    """Load and cache config.py from data_dir, returning the module object."""
    abs_dir = os.path.abspath(data_dir)
    if abs_dir not in _CONFIG_CACHE:
        config_path = os.path.join(abs_dir, 'config.py')
        spec = importlib.util.spec_from_file_location(
            f'_cfg_{abs_dir.replace(os.sep, "_")}', config_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CONFIG_CACHE[abs_dir] = mod
    return _CONFIG_CACHE[abs_dir]


# ── Low-level file helpers ────────────────────────────────────────────────────

def _read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        raw = f.read().strip()
    return json.loads(raw) if raw else default


def _save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# ── Schema normalisation ──────────────────────────────────────────────────────

def _leverage(t: dict, key_new: str, key_old: str) -> float:
    """Return leverage from new key, falling back to old key. Never returns None."""
    for k in (key_new, key_old):
        v = t.get(k)
        if v is not None:
            return float(v)
    return 0.0


def _normalize_trade(t: dict) -> dict:
    """
    Unify three historical schemas into one consistent dict:
      v1 — no position_id, uses 'asset'; size in 'actual_size_pct'
      v2 — has position_id; size still in 'actual_size_pct'
      v3 — has position_id; size in 'actual_leverage' (current _write_trade)
    """
    position_id = t.get('position_id') or t.get('asset', '')

    date_str = t.get('date', '')
    try:
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except Exception:
        parsed_date = None

    return {
        'position_id':          position_id,
        'action':               t.get('action', ''),
        'strategy':             t.get('strategy', ''),
        'date':                 parsed_date,
        'actual_price':         float(t.get('actual_price', 0)),
        'theoretical_price':    float(t.get('theoretical_price', 0)),
        'actual_leverage':      _leverage(t, 'actual_leverage',      'actual_size_pct'),
        'theoretical_leverage': _leverage(t, 'theoretical_leverage', 'theoretical_size_pct'),
        'slippage_pct':         float(t.get('slippage_pct', 0)),
        'theoretical_stop':     t.get('theoretical_stop'),
        'discretion_note':      t.get('discretion_note'),
        # ── new fields (null for legacy records) ──────────────────────────────
        'entry_close':          t.get('entry_close'),
        'exit_close':           t.get('exit_close'),
        'exit_reason':          t.get('exit_reason'),
        'signal_snapshot':      t.get('signal_snapshot'),
        # ── entry classification (explicit or derived) ────────────────────────
        'entry_type':           t.get('entry_type'),   # 'Strategy' | 'Discretionary' | None
        # ── capital snapshot (frozen at entry time; None for legacy records) ───
        'coin_capital':         t.get('coin_capital'),
        'size_usd':             t.get('size_usd'),
        'capital_total':        t.get('capital_total'),
        'coin_weight':          t.get('coin_weight'),
    }


# ── Public loaders ────────────────────────────────────────────────────────────

def load_trades(data_dir: str) -> list:
    """
    Return all trade entries as a list of normalised dicts,
    sorted by date ascending (oldest first).
    """
    path = os.path.join(data_dir, 'trades.json')
    raw  = _read_json(path, [])
    trades = [_normalize_trade(t) for t in raw]
    trades.sort(key=lambda t: t['date'] or datetime.min.date())
    return trades


def load_positions(data_dir: str) -> dict:
    """Return positions.json as a dict keyed by position_id."""
    return _read_json(os.path.join(data_dir, 'positions.json'), {})


def load_live_params(data_dir: str) -> dict:
    """Return live_params.json as a dict."""
    return _read_json(os.path.join(data_dir, 'live_params.json'), {})


def load_config(data_dir: str) -> dict:
    """Return dashboard config as a plain dict."""
    cfg = _load_config(data_dir)
    return {
        'capital':        cfg.CAPITAL,
        'active_assets':  cfg.ACTIVE_ASSETS,
        'coin_weights':   cfg.COIN_WEIGHTS,
        'execution_hour': cfg.EXECUTION_HOUR,
    }


def get_coin_capital(symbol: str, data_dir: str) -> float:
    """Dollar capital allocated to this coin, using config from data_dir."""
    cfg          = _load_config(data_dir)
    capital      = cfg.CAPITAL
    coin_weights = cfg.COIN_WEIGHTS
    active       = cfg.ACTIVE_ASSETS

    if symbol in coin_weights:
        weight = coin_weights[symbol]
    else:
        allocated    = sum(coin_weights.values())
        remaining    = 1.0 - allocated
        n_unweighted = sum(1 for a in active if a not in coin_weights)
        weight       = remaining / n_unweighted if n_unweighted > 0 else 0.0
    return capital * weight


# ── Entry reason derivation ───────────────────────────────────────────────────

def derive_entry_reason(signal_snapshot, discretion_note=None) -> str:
    """
    Classify entry as 'Strategy' or 'Discretionary'.

    Priority order:
      1. If signal_snapshot was captured: use entry_long flag directly.
      2. If no snapshot: fall back to discretion_note presence.
    """
    if signal_snapshot is not None:
        return 'Strategy' if signal_snapshot.get('entry_long') else 'Discretionary'
    return 'Discretionary' if discretion_note else 'Strategy'


# ── MAE computation with file cache ──────────────────────────────────────────

def compute_mae(position_id: str, symbol: str, entry_date, exit_date,
                entry_price: float, data_dir: str):
    """
    Maximum Adverse Excursion (%) for a trade.

    MAE % = (min(Low) - entry_price) / entry_price * 100

    Closed trades (exit_date is not None) are cached in mae_cache.json inside
    data_dir.  Open trades (exit_date is None) are computed fresh — not cached.

    Returns None on fetch error or if fewer than 1 bar is available.
    """
    from datetime import date as _date

    mae_cache_path = os.path.join(data_dir, 'mae_cache.json')
    is_closed      = (exit_date is not None)

    if is_closed:
        cache = _read_json(mae_cache_path, {})
        if position_id in cache:
            return cache[position_id]

    try:
        import pandas as pd
        from binance_client import get_binance_client, get_data

        today     = _date.today()
        end_date  = exit_date if exit_date is not None else today
        days_back = max((today - entry_date).days + 5, 10)

        client = get_binance_client()
        df     = get_data(client, symbol, '1d', days_back)

        start_ts = pd.Timestamp(str(entry_date))
        end_ts   = pd.Timestamp(str(end_date)) + pd.Timedelta(days=1)
        period   = df[(df.index >= start_ts) & (df.index < end_ts)]

        if len(period) < 1:
            return None

        min_low = float(period['Low'].min())
        mae_pct = (min_low - entry_price) / entry_price * 100

        if is_closed:
            cache[position_id] = mae_pct
            _save_json(mae_cache_path, cache)

        return mae_pct

    except Exception as e:
        print(f"  compute_mae({position_id}): {e}")
        return None


# ── Trade pair builder ────────────────────────────────────────────────────────

def build_trade_pairs(data_dir: str) -> dict:
    """
    Match ENTRY and EXIT trades chronologically by symbol.

    Groups by symbol (extracted from position_id) so that old-format entries
    (pid == plain symbol, e.g. 'BTCUSDT') correctly match new-format exits
    (pid == FIFO format, e.g. 'BTCUSDT_20260415_001').

    Returns:
        {
          "closed": list of closed-pair dicts,
          "open":   list of unmatched ENTRY dicts (open positions)
        }
    """
    trades = load_trades(data_dir)

    from collections import defaultdict
    entries_by_sym: dict = defaultdict(list)
    exits_by_sym:   dict = defaultdict(list)

    for t in trades:
        sym = _pid_to_symbol(t['position_id'])
        if t['action'] == 'ENTRY':
            entries_by_sym[sym].append(t)
        elif t['action'] == 'EXIT':
            exits_by_sym[sym].append(t)

    closed:      list = []
    open_trades: list = []

    for sym, entry_list in entries_by_sym.items():
        exit_list = exits_by_sym.get(sym, [])

        for i, entry in enumerate(entry_list):
            if i < len(exit_list):
                ex = exit_list[i]

                act_entry  = entry['actual_price']
                act_exit   = ex['actual_price']
                theo_entry = entry['theoretical_price']
                theo_exit  = ex['theoretical_price']
                act_lev    = entry['actual_leverage']
                theo_lev   = entry['theoretical_leverage']

                # ── Capital snapshot (frozen at entry time) ──────────────────
                # Use values stored in the ENTRY record when available.
                # For legacy records that pre-date the snapshot, fall back to
                # current config and mark the pair as legacy so the UI can warn.
                coin_cap_snap = entry.get('coin_capital')
                size_usd_snap = entry.get('size_usd')
                is_legacy     = (coin_cap_snap is None or size_usd_snap is None)

                if is_legacy:
                    coin_cap     = get_coin_capital(sym, data_dir)
                    act_size_usd = coin_cap * act_lev
                else:
                    coin_cap     = coin_cap_snap
                    act_size_usd = size_usd_snap   # frozen at entry time

                theo_size_usd = coin_cap * theo_lev

                # ── Close-based P&L ─────────────────────────────────────────
                eff_entry_close = entry.get('entry_close') or theo_entry or act_entry
                eff_exit_close  = ex.get('exit_close')  or theo_exit  or act_exit
                has_close_data  = (entry.get('entry_close') is not None
                                   and ex.get('exit_close') is not None)

                close_to_close_pct = (
                    (eff_exit_close / eff_entry_close - 1) * 100
                    if eff_entry_close else 0.0
                )

                pnl_usd      = act_size_usd * (close_to_close_pct / 100)
                act_ret_pct  = (act_exit  / act_entry  - 1) * 100 if act_entry  else 0.0
                theo_ret_pct = (theo_exit / theo_entry - 1) * 100 if theo_entry else 0.0
                act_pnl_usd  = act_size_usd  * (act_ret_pct  / 100)
                theo_pnl_usd = theo_size_usd * (theo_ret_pct / 100)

                # ── Slippage vs prior-day close ──────────────────────────────
                entry_slip = (
                    (act_entry - eff_entry_close) / eff_entry_close * 100
                    if eff_entry_close else 0.0
                )
                exit_slip = (
                    (act_exit - eff_exit_close) / eff_exit_close * 100
                    if eff_exit_close else 0.0
                )

                holding_days = (
                    (ex['date'] - entry['date']).days
                    if ex['date'] and entry['date'] else 0
                )

                # ── Signal metadata ──────────────────────────────────────────
                signal_snapshot = entry.get('signal_snapshot')
                entry_type = (
                    entry.get('entry_type')
                    or derive_entry_reason(signal_snapshot, entry.get('discretion_note'))
                )
                entry_reason = entry_type   # kept for backward compat

                # ── MAE (cached for closed trades) ───────────────────────────
                mae_pct = compute_mae(
                    entry['position_id'], sym,
                    entry['date'], ex['date'],
                    act_entry, data_dir,
                )

                closed.append({
                    'position_id':            entry['position_id'],
                    'symbol':                 sym,
                    'strategy':               entry['strategy'],
                    'entry_date':             entry['date'],
                    'exit_date':              ex['date'],
                    'holding_days':           holding_days,
                    'entry_close':            entry.get('entry_close'),
                    'exit_close':             ex.get('exit_close'),
                    'has_close_data':         has_close_data,
                    'close_to_close_pct':     close_to_close_pct,
                    'actual_entry':           act_entry,
                    'actual_exit':            act_exit,
                    'theoretical_entry':      theo_entry,
                    'theoretical_exit':       theo_exit,
                    'entry_slippage_pct':     entry_slip,
                    'exit_slippage_pct':      exit_slip,
                    'slippage_pct':           (entry_slip + exit_slip) / 2,
                    'leverage':               act_lev,
                    'actual_size_pct':        act_lev,
                    'theoretical_size_pct':   theo_lev,
                    'coin_capital':           coin_cap,
                    'size_usd':               act_size_usd,
                    'actual_size_usd':        act_size_usd,
                    'theoretical_size_usd':   theo_size_usd,
                    'pnl_usd':                pnl_usd,
                    'actual_pnl_usd':         act_pnl_usd,
                    'actual_return_pct':      act_ret_pct,
                    'theoretical_return_pct': theo_ret_pct,
                    'theoretical_pnl_usd':    theo_pnl_usd,
                    'mae_pct':                mae_pct,
                    'exit_reason':            ex.get('exit_reason') or '—',
                    'signal_snapshot':        signal_snapshot,
                    'entry_reason':           entry_reason,
                    'entry_type':             entry_type,
                    'discretion_note':        entry.get('discretion_note'),
                    # True when coin_capital/size_usd were missing from the ENTRY
                    # record and had to be recomputed from current config.
                    'legacy':                 is_legacy,
                })
            else:
                open_trades.append(entry)

    return {'closed': closed, 'open': open_trades}


# ── Equity curve builders ─────────────────────────────────────────────────────

_EQUITY_COLS = ['date', 'actual_pnl', 'theoretical_pnl',
                'actual_cumulative', 'theoretical_cumulative']


def _equity_df_from_closed(closed: list, start_date: date) -> pd.DataFrame:
    """Shared helper: build daily equity step-function from a list of closed pairs."""
    pnl_by_date: dict = {}
    for p in closed:
        d = p['exit_date']
        if d is None:
            continue
        if d not in pnl_by_date:
            pnl_by_date[d] = {'actual_pnl': 0.0, 'theoretical_pnl': 0.0}
        pnl_by_date[d]['actual_pnl']      += p['pnl_usd']
        pnl_by_date[d]['theoretical_pnl'] += p['theoretical_pnl_usd']

    rows = []
    current = start_date
    today   = date.today()
    while current <= today:
        day = pnl_by_date.get(current, {'actual_pnl': 0.0, 'theoretical_pnl': 0.0})
        rows.append({'date': current, **day})
        current += timedelta(days=1)

    df = pd.DataFrame(rows, columns=['date', 'actual_pnl', 'theoretical_pnl'])
    df['actual_cumulative']      = df['actual_pnl'].cumsum()
    df['theoretical_cumulative'] = df['theoretical_pnl'].cumsum()
    return df


def build_equity_curve(data_dir: str) -> pd.DataFrame:
    """
    Daily step-function equity curve from closed trade pairs.

    Columns: date | actual_pnl | theoretical_pnl |
             actual_cumulative | theoretical_cumulative
    """
    pairs  = build_trade_pairs(data_dir)
    closed = pairs.get('closed', [])

    if not closed:
        return pd.DataFrame(columns=_EQUITY_COLS)

    exit_dates = [p['exit_date'] for p in closed if p['exit_date']]
    if not exit_dates:
        return pd.DataFrame(columns=_EQUITY_COLS)

    return _equity_df_from_closed(closed, min(exit_dates))


def build_coin_equity_curves(data_dir: str) -> dict:
    """
    Per-symbol equity curves.

    Returns dict[symbol -> DataFrame] with the same columns as build_equity_curve,
    filtered to trades for that coin only.
    """
    from collections import defaultdict

    pairs  = build_trade_pairs(data_dir)
    closed = pairs.get('closed', [])

    if not closed:
        return {}

    by_symbol: dict = defaultdict(list)
    for p in closed:
        by_symbol[p['symbol']].append(p)

    result = {}
    for symbol, sym_closed in by_symbol.items():
        exit_dates = [p['exit_date'] for p in sym_closed if p['exit_date']]
        if not exit_dates:
            continue
        result[symbol] = _equity_df_from_closed(sym_closed, min(exit_dates))

    return result


# ── Capital deployment builder ────────────────────────────────────────────────

def build_capital_deployment(data_dir: str) -> pd.DataFrame:
    """
    Daily capital deployment series.

    Uses size_usd frozen in the ENTRY record; positions without a stored
    size_usd (legacy records) are excluded to preserve snapshot integrity.

    Columns: date | deployed_usd | deployment_pct
    """
    trades    = load_trades(data_dir)
    positions = load_positions(data_dir)
    capital   = float(_load_config(data_dir).CAPITAL)

    windows = []
    for t in trades:
        if t['action'] != 'ENTRY' or t.get('size_usd') is None:
            continue

        pid        = t['position_id']
        entry_date = t['date']
        size_usd   = float(t['size_usd'])

        pos          = positions.get(pid, {})
        exit_date_s  = pos.get('exit_date')
        exit_date    = None
        if exit_date_s:
            try:
                exit_date = datetime.strptime(exit_date_s, '%Y-%m-%d').date()
            except Exception:
                pass

        windows.append({
            'entry_date': entry_date,
            'exit_date':  exit_date,
            'size_usd':   size_usd,
        })

    if not windows:
        return pd.DataFrame(columns=['date', 'deployed_usd', 'deployment_pct'])

    entry_dates = [w['entry_date'] for w in windows if w['entry_date']]
    if not entry_dates:
        return pd.DataFrame(columns=['date', 'deployed_usd', 'deployment_pct'])

    start_date = min(entry_dates)
    today      = date.today()

    rows    = []
    current = start_date
    while current <= today:
        deployed = sum(
            w['size_usd']
            for w in windows
            if w['entry_date'] and w['entry_date'] <= current
            and (w['exit_date'] is None or w['exit_date'] >= current)
        )
        rows.append({
            'date':           current,
            'deployed_usd':   deployed,
            'deployment_pct': deployed / capital * 100 if capital else 0.0,
        })
        current += timedelta(days=1)

    return pd.DataFrame(rows)
