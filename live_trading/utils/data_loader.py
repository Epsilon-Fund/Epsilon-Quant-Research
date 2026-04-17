"""
Centralised data loading for all Streamlit pages.
No page should duplicate file reads or config imports.
"""

import json
import os
import sys
from datetime import datetime

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_LT_DIR    = os.path.dirname(_UTILS_DIR)
if _LT_DIR not in sys.path:
    sys.path.insert(0, _LT_DIR)

from config import ACTIVE_ASSETS, COIN_WEIGHTS, CAPITAL, EXECUTION_HOUR

_TRADES_PATH    = os.path.join(_LT_DIR, 'trades.json')
_POSITIONS_PATH = os.path.join(_LT_DIR, 'positions.json')
_PARAMS_PATH    = os.path.join(_LT_DIR, 'live_params.json')


# ── Low-level file readers ────────────────────────────────────────────────────

def _read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        raw = f.read().strip()
    return json.loads(raw) if raw else default


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
    }


# ── Public loaders ────────────────────────────────────────────────────────────

def load_trades() -> list:
    """
    Return all trade entries as a list of normalised dicts,
    sorted by date ascending (oldest first).
    """
    raw = _read_json(_TRADES_PATH, [])
    trades = [_normalize_trade(t) for t in raw]
    trades.sort(key=lambda t: t['date'] or datetime.min.date())
    return trades


def load_positions() -> dict:
    """Return positions.json as a dict keyed by position_id."""
    return _read_json(_POSITIONS_PATH, {})


def load_live_params() -> dict:
    """Return live_params.json as a dict."""
    return _read_json(_PARAMS_PATH, {})


def load_config() -> dict:
    return {
        'capital':        CAPITAL,
        'active_assets':  ACTIVE_ASSETS,
        'coin_weights':   COIN_WEIGHTS,
        'execution_hour': EXECUTION_HOUR,
    }


def get_coin_capital(symbol: str) -> float:
    """Dollar capital allocated to this coin, using the same logic as dashboard.py."""
    if symbol in COIN_WEIGHTS:
        weight = COIN_WEIGHTS[symbol]
    else:
        allocated    = sum(COIN_WEIGHTS.values())
        remaining    = 1.0 - allocated
        n_unweighted = sum(1 for a in ACTIVE_ASSETS if a not in COIN_WEIGHTS)
        weight       = remaining / n_unweighted if n_unweighted > 0 else 0.0
    return CAPITAL * weight


# ── Trade pair builder ────────────────────────────────────────────────────────

def build_trade_pairs() -> dict:
    """
    Match ENTRY and EXIT trades by position_id in FIFO order.

    Returns:
        {
          "closed": list of closed-pair dicts,
          "open":   list of unmatched ENTRY dicts (open positions)
        }
    """
    trades = load_trades()

    from collections import defaultdict
    entries_by_pid: dict = defaultdict(list)
    exits_by_pid:   dict = defaultdict(list)

    for t in trades:
        if t['action'] == 'ENTRY':
            entries_by_pid[t['position_id']].append(t)
        elif t['action'] == 'EXIT':
            exits_by_pid[t['position_id']].append(t)

    closed:      list = []
    open_trades: list = []

    for pid, entry_list in entries_by_pid.items():
        exit_list = exits_by_pid.get(pid, [])

        for i, entry in enumerate(entry_list):
            if i < len(exit_list):
                ex = exit_list[i]

                coin_cap     = get_coin_capital(pid)
                act_entry    = entry['actual_price']
                act_exit     = ex['actual_price']
                theo_entry   = entry['theoretical_price']
                theo_exit    = ex['theoretical_price']
                act_lev      = entry['actual_leverage']
                theo_lev     = entry['theoretical_leverage']

                act_size_usd  = coin_cap * act_lev
                theo_size_usd = coin_cap * theo_lev

                act_ret_pct  = (act_exit  / act_entry  - 1) * 100 if act_entry  else 0.0
                theo_ret_pct = (theo_exit / theo_entry - 1) * 100 if theo_entry else 0.0

                act_pnl_usd  = act_size_usd  * (act_ret_pct  / 100)
                theo_pnl_usd = theo_size_usd * (theo_ret_pct / 100)

                holding_days = (
                    (ex['date'] - entry['date']).days
                    if ex['date'] and entry['date'] else 0
                )
                avg_slip = (entry['slippage_pct'] + ex['slippage_pct']) / 2

                closed.append({
                    'position_id':            pid,
                    'strategy':               entry['strategy'],
                    'entry_date':             entry['date'],
                    'exit_date':              ex['date'],
                    'holding_days':           holding_days,
                    'actual_entry':           act_entry,
                    'actual_exit':            act_exit,
                    'theoretical_entry':      theo_entry,
                    'theoretical_exit':       theo_exit,
                    'actual_size_pct':        act_lev,
                    'theoretical_size_pct':   theo_lev,
                    'coin_capital':           coin_cap,
                    'actual_size_usd':        act_size_usd,
                    'theoretical_size_usd':   theo_size_usd,
                    'actual_return_pct':      act_ret_pct,
                    'theoretical_return_pct': theo_ret_pct,
                    'actual_pnl_usd':         act_pnl_usd,
                    'theoretical_pnl_usd':    theo_pnl_usd,
                    'slippage_pct':           avg_slip,
                    'entry_slippage_pct':     entry['slippage_pct'],
                    'exit_slippage_pct':      ex['slippage_pct'],
                    'discretion_note':        entry.get('discretion_note'),
                })
            else:
                open_trades.append(entry)

    return {'closed': closed, 'open': open_trades}
