# Standalone entry point — primary app is live_trading/app.py
"""
Epsilon Fund — Stat Arb Dashboard (Streamlit)

Imports all computation from dashboard.py — no logic is duplicated here.
All trade logging is done via direct file writes in this module.

Run:
    streamlit run live_trading/dashboards/statarb/streamlit_app.py
"""

import json
import re
import sys
import os
import numpy as np
from datetime import datetime, timezone, date as _date_cls
from html import escape

# ── Path setup ────────────────────────────────────────────────────────────────
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_LT_DIR        = os.path.abspath(os.path.join(_DASHBOARD_DIR, '..', '..'))
_ROOT          = os.path.dirname(_LT_DIR)
_INFRA         = os.path.join(_ROOT, 'infrastructure', 'data')
for _p in (_DASHBOARD_DIR, _LT_DIR, _INFRA):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR       = _DASHBOARD_DIR
POSITIONS_PATH = os.path.join(DATA_DIR, 'positions.json')
TRADES_PATH    = os.path.join(DATA_DIR, 'trades.json')

import streamlit as st

from dashboard import (
    run_dashboard,
    fetch_live_price,
    apply_decision,
    get_pair_capital,
    get_open_positions,
    load_live_params,
    load_positions,
)
from shared.data_loader import (
    build_trade_pairs,
    load_realised_capital,
    update_realised_capital,
    invalidate_trade_caches,
)
from shared.binance_utils import get_live_prices as _shared_live_prices
from config import ACTIVE_ASSETS, EXECUTION_HOUR, CAPITAL, COIN_WEIGHTS
import config as _cfg_mod
TRADING_COST_PCT = getattr(_cfg_mod, 'TRADING_COST_PCT', 0.001)
from optimise import ASSET_CONFIG

_PID_RE = re.compile(r'^\w+_\d{8}_\d{3}$')

# Per-pair fixed-param key sets (ASSET_CONFIG is the authoritative source).
_PAIR_FIXED_KEYS = {a['pair_key']: set(a['fixed_params'].keys()) for a in ASSET_CONFIG}


def _next_position_id(pair_key, positions):
    """Return the next available FIFO position_id for pair_key on today's UTC date."""
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    seq   = 1
    while f"{pair_key}_{today}_{seq:03d}" in positions:
        seq += 1
    return f"{pair_key}_{today}_{seq:03d}"


# ── File I/O helpers ──────────────────────────────────────────────────────────

def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        raw = f.read().strip()
    return json.loads(raw) if raw else default


def _save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _write_trade(position_id, action, strategy, theoretical_price, actual_price,
                 size_usd, discretion_note,
                 direction=None, exit_reason=None,
                 entry_y_price=None, entry_x_price=None,
                 exit_y_price=None, exit_x_price=None,
                 pair_capital=None, pair_weight=None,
                 signal_snapshot=None):
    slippage = ((actual_price - theoretical_price) / theoretical_price * 100
                if theoretical_price else 0.0)
    record = {
        'timestamp':            datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'),
        'date':                 datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'position_id':          position_id,
        'action':               action,
        'strategy':             strategy,
        'theoretical_price':    round(theoretical_price, 6),
        'actual_price':         round(actual_price, 6),
        'slippage_pct':         round(slippage, 4),
        'actual_leverage':      1.0,
        'theoretical_leverage': 1.0,
        'theoretical_stop':     None,
        'discretion_note':      discretion_note,
    }
    if action == 'ENTRY':
        record['direction']       = direction or 'long'
        record['entry_type']      = 'Strategy'
        record['entry_y_price']   = round(entry_y_price, 6) if entry_y_price else None
        record['entry_x_price']   = round(entry_x_price, 6) if entry_x_price else None
        record['signal_snapshot'] = signal_snapshot
        record['coin_capital']    = round(pair_capital, 4) if pair_capital  else None
        record['size_usd']        = round(size_usd, 4)     if size_usd      else None
        record['capital_total']   = load_realised_capital(DATA_DIR)
        record['coin_weight']     = pair_weight
    if action == 'EXIT':
        record['exit_type']     = 'full'
        record['exit_leverage'] = 1.0
        record['exit_reason']   = exit_reason or 'Strategy'
        record['exit_y_price']  = round(exit_y_price, 6) if exit_y_price else None
        record['exit_x_price']  = round(exit_x_price, 6) if exit_x_price else None
    trades = _load_json(TRADES_PATH, [])
    trades.append(record)
    _save_json(TRADES_PATH, trades)


def _write_position_entry(position_id, pair_key, symbol_y, symbol_x,
                          strategy, direction, size_usd, pair_capital,
                          entry_y_price, entry_x_price, beta, discretion_note):
    positions = _load_json(POSITIONS_PATH, {})
    positions[position_id] = {
        'position_id':         position_id,
        'symbol':              pair_key,
        'strategy':            strategy,
        'in_position':         True,
        'entry_date':          datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'entry_price':         1.0,           # synthetic normalised spread price
        'leverage_multiplier': 1.0,
        'direction':           direction,
        'partial_exits':       0,
        'exit_price':          None,
        'exit_date':           None,
        'discretion_note':     discretion_note,
        'coin_capital':        pair_capital,
        'size_usd':            size_usd,
        # Stat arb specific — needed to compute spread P&L at exit
        'symbol_y':            symbol_y,
        'symbol_x':            symbol_x,
        'entry_y_price':       round(entry_y_price, 6) if entry_y_price else None,
        'entry_x_price':       round(entry_x_price, 6) if entry_x_price else None,
        'beta':                round(beta, 6) if beta is not None else None,
    }
    _save_json(POSITIONS_PATH, positions)


def _write_position_exit(position_id, spread_pnl_pct, discretion_note):
    positions = _load_json(POSITIONS_PATH, {})
    if position_id in positions:
        positions[position_id]['in_position'] = False
        positions[position_id]['exit_price']  = round(1.0 + spread_pnl_pct, 6)
        positions[position_id]['exit_date']   = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        positions[position_id]['discretion_note'] = discretion_note
    _save_json(POSITIONS_PATH, positions)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Epsilon Fund — Stat Arb",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  html, body, [class*="css"] { font-size: 13px !important; }
  .stApp            { background-color: #f8f8f7 !important; }
  .block-container  { background-color: #f8f8f7 !important; padding-top: 5rem; padding-bottom: 2rem; }
  .streamlit-expanderHeader {
      font-size: 11px !important; font-weight: 400 !important;
      letter-spacing: 0 !important; text-transform: none !important; color: #888780 !important;
  }
  .dashboard-card {
      background: white; border: 1px solid #d3d1c7; border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06); padding: 0;
      margin-bottom: 14px; overflow: hidden;
  }
  table { font-size: 13px !important; }
  .dash-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  .dash-table th {
      font-size: 11px !important; font-weight: 500; letter-spacing: 0.05em;
      text-transform: uppercase; color: #444441;
      padding: 7px 12px 5px; text-align: center !important;
      border-bottom: 1px solid #d3d1c7; white-space: nowrap;
  }
  .dash-table td {
      padding: 8px 12px; border-bottom: 1px solid #e4e4e1;
      vertical-align: middle; text-align: center !important;
  }
  .dash-table tr:last-child td { border-bottom: none; }
  .dash-table td, .dash-table th { border-left: none !important; border-right: none !important; }
  .dash-table-labeled td:first-child {
      background: #fafaf9 !important; font-weight: 500; border-right: 1px solid #d3d1c7 !important;
  }
  .dash-table td.field-label {
      font-size: 12px; font-weight: 500; color: #888780; white-space: nowrap;
      background: #fafaf9; border-right: 1px solid #d3d1c7 !important;
      width: 180px; min-width: 180px; max-width: 180px; text-align: center !important;
  }
  .divider-row td {
      background: #f5f5f3; font-size: 10px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.07em;
      color: #888780; padding: 4px 12px; border-bottom: 1px solid #d3d1c7;
  }
  .asset-name  { font-weight: 500; font-size: 12px; }
  .asset-alloc { font-size: 11px; color: #888780; }
  .row-ENTRY td { background: #f0f8e8 !important; }
  .row-EXIT  td { background: #fdeaea !important; }
  .badge {
      display: inline-block; padding: 2px 10px; border-radius: 4px;
      font-size: 11px; font-weight: 500; white-space: nowrap;
  }
  .badge-ENTRY { background: #EAF3DE; color: #3B6D11; }
  .badge-HOLD  { background: #FAEEDA; color: #854F0B; }
  .badge-EXIT  { background: #FCEBEB; color: #A32D2D; }
  .badge-FLAT  { background: #F1EFE8; color: #5F5E5A; }
  .t { color: #1a5c2a; }
  .f { color: #8a1a1a; }
  .entry-t { background: #EAF3DE; color: #3B6D11; font-weight: 600; }
  .entry-f { background: #FCEBEB; color: #A32D2D; font-weight: 600; }
  .badge-fixed {
      background: #e4e4e1; color: #5F5E5A; font-size: 9px;
      padding: 1px 4px; border-radius: 3px; margin-left: 5px;
      display: inline-block; vertical-align: middle;
  }
  .dash-meta { font-size: 12px; color: #888780; margin-bottom: 18px; line-height: 1.8; }
  .dash-meta strong { color: #444441; font-weight: 500; }
  .table-scroll { overflow-x: auto; }
  .row-total td {
      background: #f0efea !important; font-weight: 600;
      border-top: 2px solid #d3d1c7 !important;
  }
  .form-card-header {
      font-size: 12px; font-weight: 600; color: #444441;
      margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Controls")
    if st.button("↻ Refresh data", key="sa_refresh"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data auto-refreshes every 5 minutes.")


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Fetching market data…")
def load_all():
    """
    Cache only the expensive Binance API calls and signal computation.
    Positions and decision logic stay outside — they must recompute fresh
    every render so changes to positions.json take effect immediately.
    """
    live_params = load_live_params()
    if not live_params:
        return [], None
    active_set = set(ACTIVE_ASSETS)
    pair_keys  = [k for k in live_params if k in active_set]
    result     = run_dashboard(pair_keys, live_params, positions={})
    pair_rows  = result['pairs']
    sig_date   = result['signal_date']
    # ASSET_CONFIG is authoritative for fixed params.
    for row in pair_rows:
        row['fixed_keys'] = _PAIR_FIXED_KEYS.get(row['pair_key'], row['fixed_keys'])
    return pair_rows, sig_date


def load_live_prices(symbols: tuple):
    """
    Look up current prices for the requested symbols.

    Backed by shared.binance_utils.fetch_all_live_prices() — one batched
    REST call shared across all dashboards, cached 120 s.
    """
    return _shared_live_prices(symbols)


pair_rows, signal_date = load_all()

if not pair_rows:
    st.error("live_params.json is empty or ACTIVE_ASSETS is empty — run `optimise.py` first.")
    st.stop()

# Apply decisions OUTSIDE the cache so positions.json changes take effect immediately.
# Single fresh read used by decisions, header portfolio summary, and trade forms.
_positions_now           = load_positions()
_positions_for_decisions = _positions_now

# Refresh pair_capital OUTSIDE the load_all cache so the decisions table reflects
# the latest realised_capital.json the moment a trade closes (otherwise the cached
# pair_capital could lag by up to TTL=60s after an EXIT updates realised capital).
for _r in pair_rows:
    _r['pair_capital'] = get_pair_capital(_r['pair_key'])

for _r in pair_rows:
    _open_pos = get_open_positions(_r['pair_key'], _positions_for_decisions)
    _r['sig'].update(apply_decision(_r['sig'], _open_pos, _r['pair_capital']))

# ── Auto-log theoretical trades ───────────────────────────────────────────────
# Reads positions/trades fresh each render to guard against duplicate writes.
# EXIT/STOP: fires even when exec prices are None (uses close prices as fallback).
# ENTRY_LONG/ENTRY_SHORT: fires only once both exec_y and exec_x are available.
_auto_raw     = _load_json(TRADES_PATH, [])
_auto_exited  = {t['position_id'] for t in _auto_raw if t.get('action') == 'EXIT'}
_auto_pos_now = _load_json(POSITIONS_PATH, {})
_auto_open_pks = {
    p.get('symbol', pid) for pid, p in _auto_pos_now.items() if p.get('in_position')
}
_did_auto = False

for _ar in pair_rows:
    _apk    = _ar['pair_key']
    _asig   = _ar['sig']
    _adec   = _asig['decision']
    _astrat = _ar['strategy']
    _aey    = _ar['exec_y']
    _aex    = _ar['exec_x']

    if _adec in ('EXIT', 'STOP'):
        _apos_file = _load_json(POSITIONS_PATH, {})
        for _apid, _apos in list(_apos_file.items()):
            if (
                _apos.get('symbol', _apid) == _apk
                and _apos.get('in_position')
                and _apid not in _auto_exited
            ):
                _adir   = _apos.get('direction', 'long')
                _ads    = 1 if _adir == 'long' else -1
                _asz    = float(_apos.get('size_usd') or _apos.get('coin_capital') or 0)
                _apos_ey = float(_apos.get('entry_y_price') or 0)
                _apos_ex = float(_apos.get('entry_x_price') or 0)
                _abeta   = float(_apos.get('beta') or 0)
                _axity   = _aey if _aey is not None else _asig.get('close_y', 1.0)
                _axitx   = _aex if _aex is not None else _asig.get('close_x', 1.0)
                if _apos_ey > 0 and _apos_ex > 0 and _axity > 0 and _axitx > 0:
                    _aspread_pnl = _ads * (
                        np.log(_axity / _apos_ey) - _abeta * np.log(_axitx / _apos_ex)
                    )
                else:
                    _aspread_pnl = 0.0
                _write_trade(
                    position_id=_apid, action='EXIT', strategy=_astrat,
                    theoretical_price=1.0 + _aspread_pnl * _ads,
                    actual_price=1.0 + _aspread_pnl * _ads,
                    size_usd=_asz, discretion_note='',
                    exit_reason=_adec,
                    exit_y_price=_axity, exit_x_price=_axitx,
                )
                _write_position_exit(_apid, _aspread_pnl, '')
                if _asz > 0:
                    update_realised_capital(DATA_DIR, _aspread_pnl * _asz, _apid)
                _auto_exited.add(_apid)
                _did_auto = True

    elif _adec in ('ENTRY_LONG', 'ENTRY_SHORT') and _apk not in _auto_open_pks:
        if _aey is not None and _aex is not None:
            _adir    = 'long' if _adec == 'ENTRY_LONG' else 'short'
            _apos_fresh = _load_json(POSITIONS_PATH, {})
            _anpid   = _next_position_id(_apk, _apos_fresh)
            _acap    = get_pair_capital(_apk)
            _abeta   = float(_asig.get('beta') or 0)
            _write_trade(
                position_id=_anpid, action='ENTRY', strategy=_astrat,
                theoretical_price=1.0, actual_price=1.0,
                size_usd=_acap, discretion_note='',
                direction=_adir,
                entry_y_price=_aey, entry_x_price=_aex,
                pair_capital=_acap,
            )
            _write_position_entry(
                _anpid, _apk, _ar['symbol_y'], _ar['symbol_x'],
                _astrat, _adir, _acap, _acap,
                _aey, _aex, _abeta, '',
            )
            _auto_open_pks.add(_apk)
            _did_auto = True

if _did_auto:
    invalidate_trade_caches()

generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt(v):
    if v is None: return '—'
    if isinstance(v, int): return str(v)
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f'{v:.4f}'
    return escape(str(v))


def _badge_css(d):
    if d in ('ENTRY_LONG', 'ENTRY_SHORT'): return 'badge-ENTRY'
    if d == 'HOLD':                        return 'badge-HOLD'
    if d in ('EXIT', 'STOP'):              return 'badge-EXIT'
    return 'badge-FLAT'


def _badge(d):
    return f'<span class="badge {_badge_css(d)}">{escape(d)}</span>'


def _row_cls(d):
    if d in ('ENTRY_LONG', 'ENTRY_SHORT'): return 'class="row-ENTRY"'
    if d in ('EXIT', 'STOP'):              return 'class="row-EXIT"'
    return ''


def _tf(flag):
    if flag is None:
        return '<td>—</td>'
    cls = 't' if flag else 'f'
    return f'<td><span class="{cls}">{"TRUE" if flag else "FALSE"}</span></td>'


# ── Load positions and live prices ────────────────────────────────────────────
# _positions_now was already read once near the top of this render.
_open          = {pid: p for pid, p in _positions_now.items() if p.get('in_position')}

_all_syms    = tuple(sorted({r['symbol_y'] for r in pair_rows} | {r['symbol_x'] for r in pair_rows}))
_live_prices = load_live_prices(_all_syms) if _all_syms else {}

# Realized P&L from closed trade pairs
_trade_pairs      = build_trade_pairs(DATA_DIR)
_realized_pnl_usd = sum(p['actual_pnl_usd'] for p in _trade_pairs.get('closed', []))

# Realised capital — updated after every EXIT; drives all sizing going forward
_realised_capital = load_realised_capital(DATA_DIR)
_capital_delta    = _realised_capital - CAPITAL

# Unrealized P&L across all open positions
_total_size_usd     = 0.0
_unrealized_pnl_usd = 0.0
_total_pos_value    = 0.0
_total_has_live     = False

for _pid0, _pos0 in _open.items():
    _size0 = _pos0.get('size_usd') or _pos0.get('coin_capital', 0)
    _total_size_usd += _size0
    _ey0  = _pos0.get('entry_y_price')
    _ex0  = _pos0.get('entry_x_price')
    _b0   = _pos0.get('beta')
    _sy0  = _pos0.get('symbol_y')
    _sx0  = _pos0.get('symbol_x')
    _ds0  = 1 if _pos0.get('direction', 'long') == 'long' else -1
    _ly0  = _live_prices.get(_sy0)
    _lx0  = _live_prices.get(_sx0)
    if all(v is not None for v in [_ey0, _ex0, _b0, _ly0, _lx0]):
        _pnl0 = _ds0 * (np.log(_ly0 / _ey0) - _b0 * np.log(_lx0 / _ex0)) * _size0
        _unrealized_pnl_usd += _pnl0
        _total_pos_value    += _size0 + _pnl0
        _total_has_live      = True
    else:
        _total_pos_value += _size0

_total_pnl_usd = _realized_pnl_usd + _unrealized_pnl_usd


# ── Page header ───────────────────────────────────────────────────────────────

if not globals().get('_SUPPRESS_H1', False):
    st.markdown("""
<h1 style="font-size:35px;font-weight:700;letter-spacing:-0.01em;margin-bottom:10px">
  Epsilon Fund — Stat Arb Dashboard
</h1>
""", unsafe_allow_html=True)

_optim_dates  = [r['optimised_on'] for r in pair_rows if r.get('optimised_on') and r['optimised_on'] != 'unknown']
_last_optim   = max(_optim_dates) if _optim_dates else None
if _last_optim:
    _days_since = (_date_cls.today() - datetime.strptime(_last_optim, '%Y-%m-%d').date()).days
    _optim_note = f"{_days_since}d ago"
else:
    _optim_note = '—'

st.markdown(f"""
<div class="dash-meta">
  <strong>Signal date:</strong> {signal_date or '—'} &nbsp;&nbsp;
  <strong>Generated:</strong> {generated_at} UTC &nbsp;&nbsp;
  <strong>Execution hour:</strong> {EXECUTION_HOUR}h UTC (T+1) &nbsp;&nbsp;
  <strong>Last optimised:</strong> {_optim_note}
</div>
""", unsafe_allow_html=True)


# ── Portfolio summary ─────────────────────────────────────────────────────────

_port_val      = CAPITAL + _realized_pnl_usd + (_unrealized_pnl_usd if _total_has_live else 0)
_port_val_note = '' if _total_has_live else '<span style="font-size:10px;color:#888780"> (excl. open)</span>'
_pnl_pct       = _total_pnl_usd / CAPITAL * 100 if CAPITAL else 0.0
_pnl_cls       = 'entry-t' if _pnl_pct >= 0 else 'entry-f'
_pnl_sign      = '+' if _pnl_pct >= 0 else ''

_delta_color = '#1a5c2a' if _capital_delta >= 0 else '#a32d2d'
_delta_sign  = '+' if _capital_delta >= 0 else ''
_cap_cell    = (
    f'${_realised_capital:,.2f}'
    f'<br><span style="font-size:10px;color:{_delta_color}">'
    f'{_delta_sign}${_capital_delta:,.0f} vs initial</span>'
)

st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Capital</th><th>Invested</th><th>Portfolio value</th><th>Total P&amp;L</th>
    </tr></thead>
    <tbody><tr>
      <td>{_cap_cell}</td>
      <td>${_total_size_usd:,.0f}</td>
      <td>${_port_val:,.0f}{_port_val_note}</td>
      <td class="{_pnl_cls}">{_pnl_sign}{_pnl_pct:.2f}%</td>
    </tr></tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Active Positions ──────────────────────────────────────────────────────────

st.markdown("#### ACTIVE POSITIONS")

if not _open:
    st.markdown('<p style="font-size:12px;color:#888780;margin-bottom:14px">No open positions</p>',
                unsafe_allow_html=True)
else:
    _pos_rows_html = ''
    _tot_size_r    = 0.0
    _tot_pnl_r     = 0.0
    _tot_val_r     = 0.0
    _tot_live_r    = False

    for pid, pos in _open.items():
        pair_key   = pos.get('symbol', pid)
        symbol_y   = pos.get('symbol_y', '—')
        symbol_x   = pos.get('symbol_x', '—')
        direction  = pos.get('direction', 'long')
        entry_date = pos.get('entry_date', '—')
        try:
            days_held = (_date_cls.today() - _date_cls.fromisoformat(entry_date)).days
        except Exception:
            days_held = '—'

        size_usd = pos.get('size_usd') or pos.get('coin_capital', 0)
        beta     = pos.get('beta')
        ey       = pos.get('entry_y_price')
        ex_p     = pos.get('entry_x_price')
        ly       = _live_prices.get(symbol_y)
        lx       = _live_prices.get(symbol_x)
        dsign    = 1 if direction == 'long' else -1
        _tot_size_r += size_usd

        if all(v is not None for v in [ey, ex_p, beta, ly, lx]):
            pnl_pct_r   = dsign * (np.log(ly / ey) - beta * np.log(lx / ex_p))
            unr_pnl_usd = pnl_pct_r * size_usd
            pos_val     = size_usd + unr_pnl_usd
            _tot_pnl_r += unr_pnl_usd
            _tot_val_r += pos_val
            _tot_live_r = True
            pnl_cls     = 'entry-t' if unr_pnl_usd >= 0 else 'entry-f'
            pnl_sgn     = '+' if pnl_pct_r >= 0 else ''
            pnl_pct_td  = f'<td class="{pnl_cls}">{pnl_sgn}{pnl_pct_r*100:.2f}%</td>'
            pnl_usd_td  = f'<td class="{pnl_cls}">{pnl_sgn}{unr_pnl_usd:,.0f}</td>'
            pos_val_td  = f'<td>{pos_val:,.0f}</td>'
        else:
            _tot_val_r += size_usd
            pnl_pct_td  = '<td>—</td>'
            pnl_usd_td  = '<td>—</td>'
            pos_val_td  = f'<td>{size_usd:,.0f}</td>'

        beta_td = f'<td>{beta:.3f}</td>' if beta is not None else '<td>—</td>'
        ly_td   = f'<td>{ly:,.4f}</td>' if ly is not None else '<td>—</td>'
        lx_td   = f'<td>{lx:,.4f}</td>' if lx is not None else '<td>—</td>'
        dir_lbl = 'Long' if direction == 'long' else 'Short'

        _pos_rows_html += f"""
        <tr>
          <td class="asset-name">{escape(pair_key)}</td>
          <td style="font-size:11px">{escape(symbol_y)} / {escape(symbol_x)}</td>
          <td>{dir_lbl}</td>
          <td>{escape(entry_date)}</td>
          <td>{days_held}</td>
          {beta_td}
          <td>{size_usd:,.0f}</td>
          {ly_td}
          {lx_td}
          {pnl_pct_td}
          {pnl_usd_td}
          {pos_val_td}
        </tr>"""

    # Total row
    if _tot_live_r and _tot_size_r > 0:
        _tot_pct     = _tot_pnl_r / _tot_size_r * 100
        _tot_cls     = 'entry-t' if _tot_pct >= 0 else 'entry-f'
        _tot_sgn     = '+' if _tot_pct >= 0 else ''
        _tot_pct_td  = f'<td class="{_tot_cls}">{_tot_sgn}{_tot_pct:.2f}%</td>'
        _tot_pusd_td = f'<td class="{_tot_cls}">{_tot_sgn}{_tot_pnl_r:,.0f}</td>'
    else:
        _tot_pct_td  = '<td>—</td>'
        _tot_pusd_td = '<td>—</td>'

    _total_row = f"""
        <tr class="row-total">
          <td>Total</td>
          <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
          <td>{_tot_size_r:,.0f}</td>
          <td>—</td><td>—</td>
          {_tot_pct_td}
          {_tot_pusd_td}
          <td>{_tot_val_r:,.0f}</td>
        </tr>"""

    st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Pair</th><th>Y / X</th><th>Direction</th>
      <th>Entry date</th><th>Days</th><th>Beta</th><th>Size ($)</th>
      <th>Live Y</th><th>Live X</th>
      <th>P&amp;L (%)</th><th>P&amp;L ($)</th><th>Position ($)</th>
    </tr></thead>
    <tbody>{_pos_rows_html}{_total_row}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Decisions ─────────────────────────────────────────────────────────────────

st.markdown("#### DECISIONS")
st.caption(f"Sizes based on realised capital: ${_realised_capital:,.2f}")

_dec_rows_html = ''
for r in pair_rows:
    sig      = r['sig']
    d        = sig['decision']
    pair_key = r['pair_key']

    alloc_note  = f"{r['pair_capital']:,.0f} alloc ({r['pair_weight']*100:.0f}% of ${_realised_capital:,.0f})"
    z_str       = f"{sig['z']:.3f}"    if sig.get('z')      is not None else '—'
    spread_str  = f"{sig['spread']:.4f}" if sig.get('spread') is not None else '—'
    size_str    = f"${sig['size_usd']:,.0f}" if sig.get('size_usd') is not None else '—'
    close_y_html = f"{sig['close_y']:,.4f}" if sig.get('close_y') is not None else '—'
    close_x_html = f"{sig['close_x']:,.4f}" if sig.get('close_x') is not None else '—'
    exec_y_html = (f"{r['exec_y']:,.4f}" if r['exec_y'] is not None
                   else '<span style="color:#888780">pending</span>')
    exec_x_html = (f"{r['exec_x']:,.4f}" if r['exec_x'] is not None
                   else '<span style="color:#888780">pending</span>')
    ly      = _live_prices.get(r['symbol_y'])
    lx      = _live_prices.get(r['symbol_x'])
    ly_html = f"{ly:,.4f}" if ly is not None else '—'
    lx_html = f"{lx:,.4f}" if lx is not None else '—'

    _dec_rows_html += f"""
    <tr {_row_cls(d)}>
      <td>
        <div class="asset-name">{escape(pair_key)}</div>
        <div class="asset-alloc">{escape(r['symbol_y'])} / {escape(r['symbol_x'])}</div>
        <div class="asset-alloc">{escape(alloc_note)}</div>
      </td>
      <td>{_badge(d)}</td>
      <td>{z_str}</td>
      <td>{spread_str}</td>
      <td>{size_str}</td>
      <td>{close_y_html}</td>
      <td>{close_x_html}</td>
      <td>{exec_y_html}</td>
      <td>{exec_x_html}</td>
      <td>{ly_html}</td>
      <td>{lx_html}</td>
    </tr>"""

st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Pair</th><th>Decision</th>
      <th>z-score</th><th>Spread</th><th>Size ($)</th>
      <th class="r">Last Daily Close Y ($)</th><th class="r">Last Daily Close X ($)</th>
      <th>Exec Y ({EXECUTION_HOUR}h)</th><th>Exec X ({EXECUTION_HOUR}h)</th>
      <th>Live Y</th><th>Live X</th>
    </tr></thead>
    <tbody>{_dec_rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Trade Log Forms ───────────────────────────────────────────────────────────

st.markdown("#### TRADE LOG")

_positions_for_forms = _positions_now   # reuse the single render-time read
_form_cols = st.columns(len(pair_rows))

for _fi, _r in enumerate(pair_rows):
    with _form_cols[_fi]:
        _pk       = _r['pair_key']
        _sig      = _r['sig']
        _d        = _sig['decision']
        _strategy = _r['strategy']
        _sym_y    = _r['symbol_y']
        _sym_x    = _r['symbol_x']
        _beta_sig = _sig.get('beta')

        _open_for_pair = {
            pid: p for pid, p in _positions_for_forms.items()
            if p.get('symbol', pid) == _pk and p.get('in_position')
        }
        _primary_pos  = next(iter(_open_for_pair.values()), {})
        _has_position = bool(_open_for_pair)

        _default_y = _r['exec_y'] if _r['exec_y'] is not None else _sig.get('close_y', 1.0)
        _default_x = _r['exec_x'] if _r['exec_x'] is not None else _sig.get('close_x', 1.0)

        st.markdown(
            f'<div class="form-card-header">'
            f'{escape(_pk)}&nbsp;<span class="badge {_badge_css(_d)}">{escape(_d)}</span>'
            f'<br><span style="font-size:10px;font-weight:400;color:#888780">'
            f'{escape(_sym_y)} / {escape(_sym_x)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Action toggle (outside form -> live rerender)
        _action_key  = f'sa_action_{_pk}'
        _action_opts = ['ENTRY', 'EXIT'] if _has_position else ['ENTRY']
        _def_idx     = 1 if (_has_position and _d in ('HOLD', 'EXIT', 'STOP')) else 0
        _action      = st.radio(
            '',
            _action_opts,
            index=min(_def_idx, len(_action_opts) - 1),
            key=_action_key,
            horizontal=True,
            label_visibility='collapsed',
        )
        _is_exit = (_action == 'EXIT')

        # Position selectbox for EXIT
        _selected_pid = None
        _selected_pos = {}
        if _is_exit:
            if not _open_for_pair:
                st.caption('No open positions to exit.')
            elif len(_open_for_pair) == 1:
                _selected_pid = next(iter(_open_for_pair))
                _selected_pos = _open_for_pair[_selected_pid]
            else:
                _pid_labels = {
                    pid: f"{pid}  —  {pos.get('entry_date','?')}  {pos.get('direction','?')}"
                    for pid, pos in _open_for_pair.items()
                }
                _selected_pid = st.selectbox(
                    'Position to exit',
                    options=list(_pid_labels.keys()),
                    format_func=lambda k: _pid_labels[k],
                    key=f'sa_exit_pid_{_pk}',
                )
                _selected_pos = _open_for_pair[_selected_pid]

        with st.form(key=f'sa_trade_form_{_pk}'):
            if not _is_exit:
                # ── ENTRY form ────────────────────────────────────────────────
                _dir_radio = st.radio(
                    'Direction',
                    ['Long spread', 'Short spread'],
                    index=0 if _d != 'ENTRY_SHORT' else 1,
                    horizontal=True,
                    key=f'sa_dir_{_pk}',
                )
                _price_y = st.number_input(
                    f'Y leg price  ({_sym_y})',
                    value=float(round(_default_y, 4)) if _default_y else 0.0,
                    min_value=0.0,
                    format='%.4f',
                    key=f'sa_ent_y_{_pk}',
                )
                _price_x = st.number_input(
                    f'X leg price  ({_sym_x})',
                    value=float(round(_default_x, 4)) if _default_x else 0.0,
                    min_value=0.0,
                    format='%.4f',
                    key=f'sa_ent_x_{_pk}',
                )
            else:
                # ── EXIT form ─────────────────────────────────────────────────
                _exit_reason_radio = st.radio(
                    'Exit reason',
                    ['Strategy', 'Discretionary'],
                    horizontal=True,
                    key=f'sa_exit_reason_{_pk}',
                )
                _price_y = st.number_input(
                    f'Y leg price  ({_sym_y})',
                    value=float(round(_default_y, 4)) if _default_y else 0.0,
                    min_value=0.0,
                    format='%.4f',
                    key=f'sa_ext_y_{_pk}',
                )
                _price_x = st.number_input(
                    f'X leg price  ({_sym_x})',
                    value=float(round(_default_x, 4)) if _default_x else 0.0,
                    min_value=0.0,
                    format='%.4f',
                    key=f'sa_ext_x_{_pk}',
                )

            _disc      = st.text_input('Discretion note', key=f'sa_note_{_pk}')
            _submitted = st.form_submit_button('Log trade', use_container_width=True)

        _stored_note = _primary_pos.get('discretion_note') or ''
        if _stored_note:
            with st.expander('stored note'):
                st.caption(_stored_note)

        # ── Submit handler ────────────────────────────────────────────────────
        if _submitted:
            _action_final  = st.session_state.get(_action_key, 'ENTRY')
            _is_exit_final = (_action_final == 'EXIT')

            if _is_exit_final:
                _sub_sig = ('EXIT', round(_price_y, 6), round(_price_x, 6), _selected_pid)
            else:
                _dir_val_r = st.session_state.get(f'sa_dir_{_pk}', 'Long spread')
                _sub_sig   = ('ENTRY', round(_price_y, 6), round(_price_x, 6), _dir_val_r)

            _guard_key = f'_sa_last_trade_{_pk}'
            if st.session_state.get(_guard_key) != _sub_sig:
                st.session_state[_guard_key] = _sub_sig

                if _is_exit_final and _selected_pid:
                    _pid_to_exit = st.session_state.get(f'sa_exit_pid_{_pk}', _selected_pid)
                    if _pid_to_exit is None and _open_for_pair:
                        _pid_to_exit = next(iter(_open_for_pair))
                    _pos_exit    = _open_for_pair.get(_pid_to_exit, _selected_pos)
                    _ey_st       = _pos_exit.get('entry_y_price')
                    _ex_st       = _pos_exit.get('entry_x_price')
                    _beta_st     = _pos_exit.get('beta')
                    _dir_st      = _pos_exit.get('direction', 'long')
                    _dsign_ex    = 1 if _dir_st == 'long' else -1
                    _size_st     = _pos_exit.get('size_usd') or _pos_exit.get('coin_capital', 0)

                    # Compute spread P&L from actual leg prices
                    if all(v is not None for v in [_ey_st, _ex_st, _beta_st]):
                        _spread_pnl = _dsign_ex * (
                            np.log(_price_y / _ey_st) - _beta_st * np.log(_price_x / _ex_st)
                        )
                    else:
                        _spread_pnl = 0.0

                    # Theoretical exit P&L from T+1 execution prices
                    _exec_y_r = _r['exec_y']
                    _exec_x_r = _r['exec_x']
                    if all(v is not None for v in [_exec_y_r, _exec_x_r, _ey_st, _ex_st, _beta_st]):
                        _theo_spread = _dsign_ex * (
                            np.log(_exec_y_r / _ey_st) - _beta_st * np.log(_exec_x_r / _ex_st)
                        )
                    else:
                        _theo_spread = _spread_pnl  # fall back to actual

                    _exit_reason_val = st.session_state.get(f'sa_exit_reason_{_pk}', 'Strategy')
                    _write_trade(
                        position_id=_pid_to_exit,
                        action='EXIT',
                        strategy=_strategy,
                        theoretical_price=1.0 + _theo_spread * _dsign_ex,
                        actual_price=1.0 + _spread_pnl * _dsign_ex,
                        size_usd=_size_st,
                        discretion_note=_disc,
                        exit_reason=_exit_reason_val,
                        exit_y_price=_price_y,
                        exit_x_price=_price_x,
                    )
                    _write_position_exit(_pid_to_exit, _spread_pnl, _disc)

                    # ── Update realised capital with exit P&L ─────────────────
                    _pnl_usd_exit = _spread_pnl * _size_st
                    _old_rc = load_realised_capital(DATA_DIR)
                    _new_rc = update_realised_capital(DATA_DIR, _pnl_usd_exit, _pid_to_exit)
                    print(f"Capital updated: ${_old_rc:.2f} -> ${_new_rc:.2f} "
                          f"(trade: {_pid_to_exit}, P&L: ${_pnl_usd_exit:+.2f})")

                elif not _is_exit_final:
                    _dir_k    = st.session_state.get(f'sa_dir_{_pk}', 'Long spread')
                    _dir_val  = 'long' if _dir_k == 'Long spread' else 'short'
                    _pos_fresh = load_positions()
                    _new_pid   = _next_position_id(_pk, _pos_fresh)
                    _pair_cap  = _r['pair_capital']
                    _snap      = {
                        'z':        float(_sig['z'])      if _sig.get('z')      is not None else None,
                        'spread':   float(_sig['spread']) if _sig.get('spread') is not None else None,
                        'beta':     float(_sig['beta'])   if _sig.get('beta')   is not None else None,
                        'close_y':  float(_sig['close_y']),
                        'close_x':  float(_sig['close_x']),
                        'decision': _d,
                    }
                    _write_trade(
                        position_id=_new_pid,
                        action='ENTRY',
                        strategy=_strategy,
                        theoretical_price=1.0,
                        actual_price=1.0,
                        size_usd=_pair_cap,
                        discretion_note=_disc,
                        direction=_dir_val,
                        entry_y_price=_price_y,
                        entry_x_price=_price_x,
                        pair_capital=_pair_cap,
                        pair_weight=_r['pair_weight'],
                        signal_snapshot=_snap,
                    )
                    _write_position_entry(
                        _new_pid, _pk, _sym_y, _sym_x,
                        _strategy, _dir_val, _pair_cap, _pair_cap,
                        _price_y, _price_x, _beta_sig, _disc,
                    )

                invalidate_trade_caches()
                st.rerun()


# ── Signal Conditions ─────────────────────────────────────────────────────────

col_title, col_help = st.columns([1, 6])
with col_title:
    st.markdown("#### SIGNAL CONDITIONS")
with col_help:
    with st.expander("Explanation"):
        st.markdown("""
**How the signal is built**

**Step 1 — Hedge ratio (β)**
Rolling OLS regression of log prices over a `lookback` window:
`log_Y = α + β × log_X`
β is how many units of X hedge one unit of Y. Shifted 1 bar forward to avoid lookahead bias.

**Step 2 — Spread**
`spread = log_Y − β × log_X`
The stationary residual of the cointegrated pair. When the pair is in equilibrium this hovers around its rolling mean.

**Step 3 — Z-score**
`z = (spread − rolling_mean) / rolling_std`
Normalised over a `z_lookback` window — decoupled from the OLS window so hedge-ratio stability and signal responsiveness can be tuned independently.

**Entry / Exit rules**

| Condition | Action |
|---|---|
| `z > +entry_z` | **ENTRY SHORT** — spread overvalued; short Y, long X×β |
| `z < −entry_z` | **ENTRY LONG** — spread undervalued; long Y, short X×β |
| `\|z\| < exit_z` | **EXIT** — spread reverted to mean, take profit |
| `\|z\| > stop_z` | **STOP** — spread diverging, relationship may be breaking down |
| days held ≥ `max_holding` | **EXIT** — time-based exit, avoid stale positions |
""")


_cond_rows_html = ''
for r in pair_rows:
    sig    = r['sig']
    params = r['all_params']
    z      = sig.get('z')
    entry  = float(params.get('entry',     0))
    exit_z = float(params.get('exit_z',    0))
    stop_z = float(params.get('stop_z',    0))
    mh     = int(params.get('max_holding', 0))

    z_str = f"{z:.3f}" if z is not None else '—'
    b_str = f"{sig.get('beta'):.3f}" if sig.get('beta') is not None else '—'
    s_str = f"{sig.get('spread'):.4f}" if sig.get('spread') is not None else '—'

    if z is not None:
        entry_td = _tf(abs(z) > entry)
        exit_td  = _tf(abs(z) < exit_z)
        stop_td  = _tf(abs(z) > stop_z)
    else:
        entry_td = exit_td = stop_td = '<td>—</td>'

    # Days held for primary open position
    _op_pr = next(
        (p for p in _positions_for_decisions.values()
         if p.get('symbol') == r['pair_key'] and p.get('in_position')), {}
    )
    if _op_pr.get('entry_date'):
        try:
            _dh = (_date_cls.today() - _date_cls.fromisoformat(_op_pr['entry_date'])).days
            held_td = f'<td>{_dh}d / {mh}d</td>'
        except Exception:
            held_td = '<td>—</td>'
    else:
        held_td = '<td>—</td>'

    _cond_rows_html += f"""
    <tr>
      <td class="asset-name">{escape(r['pair_key'])}</td>
      <td>{b_str}</td>
      <td>{s_str}</td>
      <td>{z_str}</td>
      {entry_td}
      {exit_td}
      {stop_td}
      {held_td}
      <td>{_badge(sig['decision'])}</td>
    </tr>"""

st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table dash-table-labeled">
    <thead><tr>
      <th>Pair</th><th>Beta</th><th>Spread</th><th>z-score</th>
      <th>|z|&gt;entry</th><th>|z|&lt;exit</th><th>|z|&gt;stop</th>
      <th>Held / Max</th><th>Decision</th>
    </tr></thead>
    <tbody>{_cond_rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Spread Details ────────────────────────────────────────────────────────────

with st.expander("SPREAD DETAILS"):
    st.markdown("""
**S1 Spread construction:** `spread = log(Close_Y) − β × log(Close_X)` · β from rolling OLS over `lookback` bars
**S2 Z-score normalisation:** `z = (spread − mean) / std` · rolling over `z_lookback` bars
**S3 Thresholds:** Entry at `±entry_z` · Exit at `±exit_z` · Stop at `±stop_z`
""")
    _n  = len(pair_rows)
    _cw = f'calc((100% - 180px) / {_n})'

    _pair_ths_sd = ''.join(
        f'<th style="text-align:right;padding-right:24px">{escape(r["pair_key"])}'
        f'<br><span style="font-size:10px;font-weight:400;text-transform:none;'
        f'letter-spacing:0;color:#888780">'
        f'{escape(r["symbol_y"])} / {escape(r["symbol_x"])}</span></th>'
        for r in pair_rows
    )

    def _sd_row(label, cells):
        tds = ''.join(
            f'<td style="text-align:right;padding-right:24px">{cell}</td>'
            for cell in cells
        )
        return f'<tr><td class="field-label">{escape(label)}</td>{tds}</tr>'

    def _sd_divider(label):
        return f'<tr class="divider-row"><td colspan="{_n+1}">{escape(label)}</td></tr>'

    def _fmt4(v):
        return f'{v:.4f}' if v is not None else '—'

    def _fmt3(v):
        return f'{v:.3f}' if v is not None else '—'

    def _fmti(v):
        return str(int(v)) if v is not None else '—'

    _sd_html = (
        _sd_divider('Hedge ratio') +
        _sd_row('OLS lookback (bars)',  [_fmti(r['all_params'].get('lookback'))   for r in pair_rows]) +
        _sd_row('β (current)',          [_fmt3(r['sig'].get('beta'))               for r in pair_rows]) +
        _sd_divider('Spread') +
        _sd_row('z lookback (bars)',    [_fmti(r['all_params'].get('z_lookback'))  for r in pair_rows]) +
        _sd_row('Spread (current)',     [_fmt4(r['sig'].get('spread'))             for r in pair_rows]) +
        _sd_row('Rolling mean',         [_fmt4(r['sig'].get('spread_mean'))        for r in pair_rows]) +
        _sd_row('Rolling std',          [_fmt4(r['sig'].get('spread_std'))         for r in pair_rows]) +
        _sd_row('Z-score (current)',    [_fmt3(r['sig'].get('z'))                  for r in pair_rows]) +
        _sd_divider('Thresholds') +
        _sd_row('Entry |z| >',          [_fmt3(r['all_params'].get('entry'))       for r in pair_rows]) +
        _sd_row('Exit  |z| <',          [_fmt3(r['all_params'].get('exit_z'))      for r in pair_rows]) +
        _sd_row('Stop  |z| >',          [_fmt3(r['all_params'].get('stop_z'))      for r in pair_rows]) +
        _sd_row('Max holding (days)',   [_fmti(r['all_params'].get('max_holding')) for r in pair_rows])
    )

    _pair_cols_sd = ''.join(f'<col style="width:{_cw}">' for _ in pair_rows)
    _colgroup_sd  = f'<col style="width:180px">{_pair_cols_sd}'

    st.markdown(f"""
<div class="dashboard-card">
  <div class="table-scroll">
  <table class="dash-table" style="table-layout:fixed;width:100%">
    <colgroup>{_colgroup_sd}</colgroup>
    <thead><tr>
      <th style="text-align:left">Metric</th>{_pair_ths_sd}
    </tr></thead>
    <tbody>{_sd_html}</tbody>
  </table>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Parameters ────────────────────────────────────────────────────────────────

all_param_keys = sorted({k for r in pair_rows for k in r['all_params']})
n     = len(pair_rows)
col_w = f'calc((100% - 180px) / {n})'

pair_ths = ''.join(
    f'<th style="text-align:right;padding-right:24px">{escape(r["pair_key"])}'
    f'<br><span style="font-size:10px;font-weight:400;text-transform:none;'
    f'letter-spacing:0;color:#888780">{escape(r["optimised_on"])}</span></th>'
    for r in pair_rows
)

_param_rows_html = ''
for key in all_param_keys:
    tds = ''
    for r in pair_rows:
        val      = r['all_params'].get(key)
        is_fixed = key in r['fixed_keys']
        f_badge  = ' <span class="badge-fixed">F</span>' if is_fixed else ''
        tds     += f'<td style="text-align:right;padding-right:24px">{fmt(val)}{f_badge}</td>'
    _param_rows_html += f'<tr><td class="field-label">{escape(key)}</td>{tds}</tr>'

pair_cols = ''.join(f'<col style="width:{col_w}">' for _ in pair_rows)
colgroup  = f'<col style="width:180px">{pair_cols}'
_optim_str = '  ·  '.join(f"{escape(r['pair_key'])}: {escape(r['optimised_on'])}" for r in pair_rows)

with st.expander("PARAMETERS"):
    st.caption(f"Last optimised — {_optim_str}")
    st.markdown(f"""
<div class="dashboard-card">
  <div class="table-scroll">
  <table class="dash-table" style="table-layout:fixed;width:100%">
    <colgroup>{colgroup}</colgroup>
    <thead><tr>
      <th style="text-align:left">Param</th>{pair_ths}
    </tr></thead>
    <tbody>{_param_rows_html}</tbody>
  </table>
  </div>
  <div style="padding:6px 16px 8px;font-size:11px;color:#888780;border-top:1px solid #e4e4e1">
    <span class="badge-fixed">F</span> &nbsp;fixed by stability analysis
  </div>
</div>
""", unsafe_allow_html=True)
