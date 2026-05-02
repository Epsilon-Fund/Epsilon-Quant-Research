# Standalone entry point — primary app is live_trading/app.py
"""
Epsilon Fund — BB Breakout Dashboard (Streamlit)

Imports all computation from dashboard.py — no logic is duplicated here.
dashboard.py is the sole computation layer. All trade logging is done via
direct file writes in this module — no external server required.

Run:
    streamlit run live_trading/dashboards/bbbreakout/streamlit_app.py
"""

import hashlib
import json
import math
import re
import sys
import os
import time
from datetime import datetime, timezone, timedelta, date
from html import escape

import pandas as pd

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
    fetch_live_prices,
    apply_decision,
    get_coin_capital,
    get_open_positions,
)
from shared.data_loader import (
    load_positions,
    load_trades,
    load_live_params,
    build_trade_pairs,
    load_realised_capital,
    update_realised_capital,
    invalidate_trade_caches,
)
from shared.binance_utils import get_live_prices as _shared_live_prices
from config import ACTIVE_ASSETS, CAPITAL, COIN_WEIGHTS
import config as _cfg_mod
TRADING_COST_PCT = getattr(_cfg_mod, 'TRADING_COST_PCT', 0.0)
from strategies import ASSET_CONFIG

_PID_RE = re.compile(r'^\w+_\d{8}_\d{3}$')


# ── FIFO helpers ──────────────────────────────────────────────────────────────

def _next_position_id(symbol, positions):
    """Return the next available position_id for symbol on today's UTC date."""
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    seq   = 1
    while f"{symbol}_{today}_{seq:03d}" in positions:
        seq += 1
    return f"{symbol}_{today}_{seq:03d}"


def _migrate_positions():
    """
    One-time migration: convert old symbol-keyed positions.json entries to the
    new FIFO position_id format ({symbol}_{YYYYMMDD}_{seq:03d}).
    Idempotent — already-migrated entries are left untouched.
    """
    positions = _load_json(POSITIONS_PATH, {})
    old_keys  = [k for k in positions if not _PID_RE.match(k)]
    if not old_keys:
        return

    migrated = {}
    for key, pos in positions.items():
        if _PID_RE.match(key):
            migrated[key] = pos
            continue
        symbol     = key
        entry_date = pos.get('entry_date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        date_str   = entry_date.replace('-', '')
        seq = 1
        new_key = f"{symbol}_{date_str}_{seq:03d}"
        while new_key in migrated:
            seq += 1
            new_key = f"{symbol}_{date_str}_{seq:03d}"
        new_pos = dict(pos)
        new_pos['position_id'] = new_key
        new_pos['symbol']      = symbol
        new_pos.setdefault('partial_exits', 0)
        migrated[new_key] = new_pos

    _save_json(POSITIONS_PATH, migrated)
    print(f"[bb positions migration] migrated {len(old_keys)} entr{'y' if len(old_keys)==1 else 'ies'}")


# ── File write helpers ────────────────────────────────────────────────────────

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
                 theoretical_leverage, actual_leverage, theoretical_stop, discretion_note,
                 direction=None, exit_type=None, exit_leverage=None,
                 entry_close=None, exit_close=None, exit_reason=None, signal_snapshot=None,
                 entry_type=None,
                 coin_capital=None, size_usd=None, capital_total=None, coin_weight=None):
    slippage = (actual_price - theoretical_price) / theoretical_price * 100 if theoretical_price else 0.0
    entry = {
        'timestamp':            datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'),
        'date':                 datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'position_id':          position_id,
        'action':               action,
        'strategy':             strategy,
        'theoretical_price':    round(theoretical_price, 4),
        'actual_price':         round(actual_price, 4),
        'slippage_pct':         round(slippage, 4),
        'theoretical_leverage': theoretical_leverage,
        'actual_leverage':      actual_leverage,
        'theoretical_stop':     theoretical_stop,
        'discretion_note':      discretion_note,
    }
    if action == 'ENTRY':
        entry['direction']       = direction or 'long'
        entry['entry_type']      = entry_type or 'Strategy'
        entry['entry_close']     = round(float(entry_close), 4) if entry_close is not None else None
        entry['signal_snapshot'] = signal_snapshot
        entry['coin_capital']    = round(coin_capital, 4) if coin_capital is not None else None
        entry['size_usd']        = round(size_usd,     4) if size_usd     is not None else None
        entry['capital_total']   = capital_total
        entry['coin_weight']     = coin_weight
    if action == 'EXIT':
        entry['exit_type']     = exit_type or 'full'
        entry['exit_leverage'] = exit_leverage
        entry['exit_close']    = round(float(exit_close), 4) if exit_close is not None else None
        entry['exit_reason']   = exit_reason or 'Strategy'
    trades = _load_json(TRADES_PATH, [])
    trades.append(entry)
    _save_json(TRADES_PATH, trades)


def _write_position_entry(position_id, symbol, actual_price, actual_leverage,
                          theoretical_stop, strategy, discretion_note, direction='long',
                          coin_capital=None, size_usd=None,
                          take_profit=None, regime_at_entry=None):
    positions = _load_json(POSITIONS_PATH, {})
    positions[position_id] = {
        'position_id':         position_id,
        'symbol':              symbol,
        'strategy':            strategy,
        'in_position':         True,
        'entry_date':          datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'entry_price':         actual_price,
        'leverage_multiplier': actual_leverage,
        'pending_stop':        theoretical_stop,
        'current_stop':        None,
        'direction':           direction,
        'partial_exits':       0,
        'exit_price':          None,
        'exit_date':           None,
        'discretion_note':     discretion_note,
        'coin_capital':        coin_capital,
        'size_usd':            size_usd,
        'take_profit':         take_profit,        # None ⇒ trail-only (strong-bull at entry)
        'regime_at_entry':     regime_at_entry,    # 'strong_bull' or 'chop_bear'
    }
    _save_json(POSITIONS_PATH, positions)


def _write_position_exit(position_id, actual_price, discretion_note):
    positions = _load_json(POSITIONS_PATH, {})
    if position_id in positions:
        positions[position_id]['in_position'] = False
        positions[position_id]['exit_price']  = actual_price
        positions[position_id]['exit_date']   = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        positions[position_id]['discretion_note'] = discretion_note
    _save_json(POSITIONS_PATH, positions)


def _write_position_partial_exit(position_id, exit_leverage, actual_price, discretion_note):
    positions = _load_json(POSITIONS_PATH, {})
    if position_id in positions:
        pos         = positions[position_id]
        current_lev = pos.get('leverage_multiplier') or pos.get('size_pct', 0)
        remaining   = round(max(current_lev - exit_leverage, 0), 4)
        pos['leverage_multiplier'] = remaining
        pos['partial_exits']       = pos.get('partial_exits', 0) + 1
        pos['discretion_note']     = discretion_note
        pos.pop('size_pct', None)
        coin_cap = pos.get('coin_capital')
        if coin_cap is not None:
            pos['size_usd'] = round(coin_cap * remaining, 4)
        # If the partial exit drained the entire leverage, auto-close the
        # position — otherwise the dashboard ends up with an in_position=True
        # row that has size_usd=0 and dashes out every P&L cell.
        if remaining <= 0:
            pos['in_position'] = False
            pos['exit_price']  = actual_price
            pos['exit_date']   = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    _save_json(POSITIONS_PATH, positions)


def _write_pending_stop(position_id, new_stop):
    positions = _load_json(POSITIONS_PATH, {})
    if position_id in positions:
        positions[position_id]['pending_stop'] = new_stop
    _save_json(POSITIONS_PATH, positions)


def _confirm_stop(position_id):
    positions = _load_json(POSITIONS_PATH, {})
    if position_id in positions:
        pending = positions[position_id].get('pending_stop')
        if pending is not None:
            positions[position_id]['current_stop'] = pending
    _save_json(POSITIONS_PATH, positions)


_migrate_positions()

_hint_popover = getattr(st, 'popover', None) or st.expander

# Per-coin fixed-param key sets from ASSET_CONFIG
_ASSET_FIXED_KEYS = {a['symbol']: set(a['fixed_params'].keys()) for a in ASSET_CONFIG}

for _sym in ('BTCUSDT', 'ETHUSDT', 'AVAXUSDT', 'ADAUSDT', 'NEARUSDT'):
    if _sym in _ASSET_FIXED_KEYS:
        print(f"BB {_sym} fixed params: {sorted(_ASSET_FIXED_KEYS[_sym])}")


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Epsilon Fund — BB Breakout",
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
      box-shadow: 0 1px 3px rgba(0,0,0,0.06); padding: 0; margin-bottom: 14px; overflow: hidden;
  }
  .section-label {
      font-size: 11px !important; font-weight: 500; letter-spacing: 0.06em;
      text-transform: uppercase; color: #444441; margin: 0;
  }
  .section-note { font-size: 11px; color: #888780; }
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
  [data-testid="stExpander"] .dash-table td.field-label {
      border-right: 1px solid #d3d1c7 !important;
  }
  .dash-table-labeled td:first-child {
      background: #fafaf9 !important; font-weight: 500;
      border-right: 1px solid #d3d1c7 !important;
  }
  .asset-name  { font-weight: 500; font-size: 12px; }
  .asset-alloc { font-size: 11px; color: #888780; }
  .row-ENTRY td { background: #f0f8e8 !important; }
  .row-EXIT  td { background: #fdeaea !important; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 4px; font-size: 11px; font-weight: 500; white-space: nowrap; }
  .badge-ENTRY { background: #EAF3DE; color: #3B6D11; }
  .badge-HOLD  { background: #FAEEDA; color: #854F0B; }
  .badge-EXIT  { background: #FCEBEB; color: #A32D2D; }
  .badge-FLAT  { background: #F1EFE8; color: #5F5E5A; }
  .badge-LONG  { background: #EAF3DE; color: #3B6D11; }
  .badge-SHORT { background: #FCEBEB; color: #A32D2D; }
  .t { color: #1a5c2a; }
  .f { color: #8a1a1a; }
  .entry-t { background: #EAF3DE; color: #3B6D11; font-weight: 600; }
  .entry-f { background: #FCEBEB; color: #A32D2D; font-weight: 600; }
  .stop-up   { color: #1a5c2a; font-size: 11px; font-weight: 600; }
  .stop-prev { color: #888780; font-size: 11px; }
  .dash-table td.field-label {
      font-size: 12px; font-weight: 500; color: #888780;
      white-space: nowrap; background: #fafaf9;
      border-right: 1px solid #d3d1c7;
      width: 180px; min-width: 180px; max-width: 180px;
      text-align: center !important;
  }
  .divider-row td {
      background: #f5f5f3; font-size: 10px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.07em;
      color: #888780; padding: 4px 12px; border-bottom: 1px solid #d3d1c7;
  }
  .badge-fixed {
      background: #e4e4e1; color: #5F5E5A; font-size: 9px;
      padding: 1px 4px; border-radius: 3px; margin-left: 5px;
      display: inline-block; vertical-align: middle;
  }
  .upd-yes { color: #1a5c2a; font-weight: 600; }
  .upd-no  { color: #888780; }
  .dash-meta {
      font-size: 12px; color: #888780; margin-bottom: 18px; line-height: 1.8;
  }
  .dash-meta strong { color: #444441; font-weight: 500; }
  .table-scroll { overflow-x: auto; }
  .row-total td {
      background: #f0efea !important; font-weight: 600;
      border-top: 2px solid #d3d1c7 !important;
  }
  .form-card {
      background: white; border: 1px solid #d3d1c7; border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06); padding: 14px 16px 10px; margin-bottom: 14px;
  }
  .form-card-header {
      font-size: 12px; font-weight: 600; color: #444441;
      margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
  }
  .bb-band-above { color: #1a5c2a; font-weight: 600; }
  .bb-band-below { color: #8a1a1a; font-weight: 600; }
  .bb-band-mid   { color: #888780; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Controls")
    if st.button("↻ Refresh data", key="bb_refresh_data"):
        st.cache_data.clear()
        st.session_state.pop("bb_dashboard_data", None)
        st.session_state.pop("bb_dashboard_cache_key", None)
        st.rerun()
    if st.button("↻ Refresh signals", key="bb_refresh_signals"):
        st.session_state.pop("bb_dashboard_data", None)
        st.session_state.pop("bb_dashboard_cache_key", None)
        st.rerun()
    st.caption("Prices refresh every 30s. Signals refresh on new 1H bar.")


# ── Signal cache helpers ───────────────────────────────────────────────────────

def _signal_cache_key(coin_symbols, live_params):
    payload = json.dumps(
        {"assets": sorted(coin_symbols), "params": live_params},
        sort_keys=True, default=str
    )
    return hashlib.md5(payload.encode()).hexdigest()


def _current_signal_date() -> str:
    """Returns a string identifying the current expected 1H signal bar (YYYYmmddHH)."""
    return datetime.now(timezone.utc).strftime('%Y%m%d%H')


# ── Data loading — session_state signal cache ─────────────────────────────────

def load_live_prices(symbols: tuple):
    """
    Look up current prices for the requested symbols.

    Backed by shared.binance_utils.fetch_all_live_prices() — one batched
    REST call shared across all dashboards, cached 300 s.
    """
    return _shared_live_prices(symbols)


_live_params_raw = load_live_params(DATA_DIR)
if not _live_params_raw:
    st.error("live_params.json is empty — run `optimise.py` first.")
    st.stop()

_active_set   = set(ACTIVE_ASSETS)
_coin_symbols = [sym for sym in _live_params_raw if sym in _active_set]
_cache_key    = _signal_cache_key(_coin_symbols, _live_params_raw)

# Auto-refresh when a new 1H bar is available
_expected_signal_date = _current_signal_date()
if st.session_state.get("bb_signal_date") != _expected_signal_date:
    st.session_state.pop("bb_dashboard_data", None)
    st.session_state.pop("bb_dashboard_cache_key", None)

if ("bb_dashboard_data" not in st.session_state or
        st.session_state.get("bb_dashboard_cache_key") != _cache_key):
    with st.spinner("Loading BB Breakout signals..."):
        _result = run_dashboard(_coin_symbols, _live_params_raw, positions={})
        _rows   = _result['assets']
        _sdate  = _result['signal_date']
        # Capture the actual data-fetch time so the "Generated:" header
        # shows when run_dashboard() last fired, not the current render
        # clock.  Persisted in session_state alongside the data so
        # subsequent cache-hit renders display the same value.
        _gen_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        for _row in _rows:
            _row['fixed_keys'] = _ASSET_FIXED_KEYS.get(_row['symbol'], _row['fixed_keys'])
        st.session_state.bb_dashboard_data      = (_rows, _sdate, _gen_at)
        st.session_state.bb_dashboard_cache_key = _cache_key
        st.session_state.bb_signal_date         = _expected_signal_date

coin_rows, signal_date, generated_at = st.session_state.bb_dashboard_data

# Apply decisions OUTSIDE the cache so positions.json is always fresh.
# Single fresh read used by decisions, header portfolio summary, and trade forms.
_positions_now           = load_positions(DATA_DIR)
_positions_for_decisions = _positions_now

# Refresh coin_capital OUTSIDE the load_all cache so the decisions table reflects
# the latest realised_capital.json the moment a trade closes (otherwise the cached
# coin_capital could lag by up to TTL=60s after an EXIT updates realised capital).
for _c in coin_rows:
    _c['coin_capital'] = get_coin_capital(_c['symbol'])

for _c in coin_rows:
    _open_pos = get_open_positions(_c['symbol'], _positions_for_decisions)
    _c['sig'].update(
        apply_decision(_c['sig'], _open_pos, _c['exec_price'], _c['coin_capital'])
    )

# `generated_at` was already loaded from session_state above (the actual
# time run_dashboard() last fired, not the current render clock).

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt(v):
    if v is None: return '—'
    if isinstance(v, int): return str(v)
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f'{v:.4f}'.rstrip('0').rstrip('.')
    return escape(str(v))


def _fmt_price(v):
    """Adaptive price formatter — matches decimal places to price magnitude."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    if v >= 100:
        return f'{v:,.2f}'
    elif v >= 1:
        return f'{v:,.3f}'
    else:
        return f'{v:,.4f}'


def _bg_box(flag):
    cls = 'entry-t' if flag else 'entry-f'
    return f'<td class="c {cls}">{"TRUE" if flag else "FALSE"}</td>'


def _tf(flag):
    txt_cls = 't' if flag else 'f'
    return f'<td class="c"><span class="{txt_cls}">{"TRUE" if flag else "FALSE"}</span></td>'


def _dir_str(direction_int):
    """Convert +1/-1 integer to 'long'/'short' string."""
    if direction_int == 1:  return 'long'
    if direction_int == -1: return 'short'
    return '—'


# ── Live prices + portfolio state ─────────────────────────────────────────────
# _positions_now was already read once near the top of this render.
_open            = {pid: p for pid, p in _positions_now.items() if p.get('in_position')}
st.session_state.bb_open_positions = _open
_coin_sig_by_sym = {c['symbol']: c['sig'] for c in coin_rows}

_all_syms    = tuple(sorted({c['symbol'] for c in coin_rows} | {p.get('symbol', pid) for pid, p in _open.items()}))
_live_prices = load_live_prices(_all_syms) if _all_syms else {}

_trade_pairs        = build_trade_pairs(DATA_DIR)
_realized_pnl_usd   = sum(p['actual_pnl_usd'] for p in _trade_pairs.get('closed', []))

# Realised capital — updated after every EXIT; drives all sizing going forward
_realised_capital = load_realised_capital(DATA_DIR)
_capital_delta    = _realised_capital - CAPITAL

_total_size_usd      = 0.0
_unrealized_pnl_usd  = 0.0
_total_pos_value     = 0.0
_total_has_live      = False

for _pid0, _pos0 in _open.items():
    _sym0 = _pos0.get('symbol', _pid0)
    _lp0  = _live_prices.get(_sym0)
    _ep0  = _pos0.get('entry_price', 0)
    _lev0 = _pos0.get('leverage_multiplier') or _pos0.get('size_pct', 0)
    _sz0  = _pos0.get('size_usd') or (get_coin_capital(_sym0) * _lev0)
    _dir0 = _pos0.get('direction', 'long')
    _total_size_usd += _sz0
    if _lp0 and _ep0:
        _cost0 = _sz0 * TRADING_COST_PCT * 2
        # P&L direction: long profits from price up, short from price down
        _move  = (_lp0 - _ep0) / _ep0 if _dir0 == 'long' else (_ep0 - _lp0) / _ep0
        _pnl0  = _move * _sz0 - _cost0
        _unrealized_pnl_usd += _pnl0
        _total_pos_value    += _sz0 + _pnl0
        _total_has_live      = True
    else:
        _total_pos_value += _sz0

_total_pnl_usd = _realized_pnl_usd + _unrealized_pnl_usd


# ── Page header ───────────────────────────────────────────────────────────────

_optim_dates      = [c['optimised_on'] for c in coin_rows if c.get('optimised_on')]
_last_optim       = max(_optim_dates) if _optim_dates else None
_days_since_optim = (
    (datetime.today().date() - datetime.strptime(_last_optim, '%Y-%m-%d').date()).days
    if _last_optim else None
)

if not globals().get('_SUPPRESS_H1', False):
    st.markdown("""
<h1 style="font-size:35px;font-weight:700;letter-spacing:-0.01em;margin-bottom:10px">
  Epsilon Fund — BB Breakout Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="dash-meta">
  <strong>Last 1H bar:</strong> {signal_date} UTC &nbsp;&nbsp;
  <strong>Generated:</strong> {generated_at} UTC
</div>
""", unsafe_allow_html=True)


# ── Portfolio summary ─────────────────────────────────────────────────────────

_port_val      = CAPITAL + _realized_pnl_usd + (_unrealized_pnl_usd if _total_has_live else 0)
_port_val_note = '' if _total_has_live else '<span style="font-size:10px;color:#888780"> (excl. open)</span>'
_port_val_str  = f"${_port_val:,.0f}{_port_val_note}"
_pnl_pct    = _total_pnl_usd / CAPITAL * 100 if CAPITAL else 0.0
_pnl_cls    = 'entry-t' if _pnl_pct >= 0 else 'entry-f'
_pnl_sign   = '+' if _pnl_pct >= 0 else ''
_pnl_str    = f"{_pnl_sign}{_pnl_pct:.2f}%"

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
      <td>{_port_val_str}</td>
      <td class="{_pnl_cls}">{_pnl_str}</td>
    </tr></tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Active Positions — live-price fragment ────────────────────────────────────

@st.fragment(run_every=30)
def _bb_positions_fragment():
    """Re-renders active positions HTML table with fresh live prices. No external calls besides fetch_live_prices."""
    _t0 = time.time()
    open_positions = st.session_state.get("bb_open_positions", {})
    if not open_positions:
        st.markdown(
            '<p style="font-size:12px;color:#888780;margin-bottom:14px">No open positions</p>',
            unsafe_allow_html=True,
        )
        return

    symbols     = list({p.get("symbol", pid) for pid, p in open_positions.items()})
    live_prices = fetch_live_prices(symbols)
    elapsed     = time.time() - _t0
    print(f"Fragment refresh: {elapsed:.2f}s ({len(symbols)} price fetches)")

    _bb_rows, _, _ = st.session_state.get("bb_dashboard_data", ([], None, None))
    coin_sig_by_sym = {c['symbol']: c['sig'] for c in _bb_rows} if _bb_rows else {}

    from datetime import date as _date
    _pos_rows       = ''
    _tot_size_usd_r = 0.0
    _tot_pnl_usd_r  = 0.0
    _tot_pos_val_r  = 0.0
    _tot_has_live_r = False

    for pid, pos in open_positions.items():
        _sym_for_pos = pos.get('symbol', pid)
        live_price   = live_prices.get(_sym_for_pos)
        entry_price  = pos.get('entry_price', 0) or 0
        pos_dir      = pos.get('direction', 'long')

        leverage_multiplier = pos.get('leverage_multiplier') or pos.get('size_pct', 0)
        size_usd = pos.get('size_usd') or (get_coin_capital(_sym_for_pos) * leverage_multiplier)
        _tot_size_usd_r += size_usd

        if live_price and entry_price:
            _cost0   = size_usd * TRADING_COST_PCT * 2
            _move    = (live_price - entry_price) / entry_price if pos_dir == 'long' else (entry_price - live_price) / entry_price
            _pnl0    = _move * size_usd - _cost0
            _ret_pct = _pnl0 / size_usd * 100
            _pos_val = size_usd + _pnl0
            _tot_pnl_usd_r  += _pnl0
            _tot_pos_val_r  += _pos_val
            _tot_has_live_r  = True
            _pnl_cls0  = 'entry-t' if _pnl0 >= 0 else 'entry-f'
            _sign0     = '+' if _ret_pct >= 0 else ''
            live_td    = f'<td>{_fmt_price(live_price)}</td>'
            pnl_pct_td = f'<td class="{_pnl_cls0}">{_sign0}{_ret_pct:.2f}%</td>'
            pnl_usd_td = f'<td class="{_pnl_cls0}">{_sign0}{_pnl0:,.0f}</td>'
            pos_val_td = f'<td>{_pos_val:,.0f}</td>'
        else:
            _tot_pos_val_r += size_usd
            live_td    = '<td>—</td>'
            pnl_pct_td = '<td>—</td>'
            pnl_usd_td = '<td>—</td>'
            pos_val_td = '<td>—</td>'

        _dir_badge = (
            '<span class="badge badge-LONG">LONG</span>' if pos_dir == 'long'
            else '<span class="badge badge-SHORT">SHORT</span>'
        )

        conf_stop    = pos.get('current_stop')
        pending_stop = pos.get('pending_stop')
        _needs_confirm = pending_stop is not None and pending_stop != conf_stop

        if conf_stop is not None:
            if _needs_confirm:
                stop_td = (f'<td>{_fmt_price(conf_stop)}'
                           f'<br><span class="stop-up">-> {_fmt_price(pending_stop)}</span></td>')
            else:
                stop_td = f'<td>{_fmt_price(conf_stop)}</td>'
        elif pending_stop is not None:
            stop_td = (f'<td><span style="color:#888780">{_fmt_price(pending_stop)}</span>'
                       f'<br><span style="font-size:10px;color:#888780">unconfirmed</span></td>')
        else:
            stop_td = '<td style="color:#888780">—</td>'

        # Take-profit cell — re-derived at the user-selected R:R ratio for
        # display only.  Stored TP in positions.json is always the entry-time
        # 6:1 target; this just shows what a different ratio would price at.
        coin_sig       = coin_sig_by_sym.get(_sym_for_pos, {})
        _stored_tp     = pos.get('take_profit')
        _entry_regime  = pos.get('regime_at_entry')
        _regime_now    = bool(coin_sig.get('regime_strong_bull', False))

        # Back-derive entry-time stop distance from the stored 6:1 TP, then
        # apply the user-chosen ratio.  Falls back to current trail_dist for
        # legacy positions that lack a stored TP but aren't strong-bull.
        _entry_stop_dist = None
        if _stored_tp is not None:
            _entry_stop_dist = abs(float(_stored_tp) - entry_price) / 6.0
        elif _entry_regime is None:
            # Legacy position — use the live trail_dist as a best-effort proxy
            _td = coin_sig.get('trail_dist')
            if _td and _td > 0:
                _entry_stop_dist = float(_td)

        # Strong-bull entries (TP=None and regime_at_entry='strong_bull')
        # genuinely have no fixed TP — keep the trail-only display.
        if _entry_regime == 'strong_bull' and _stored_tp is None:
            _tp_for_display = None
        elif _entry_stop_dist is None or _entry_stop_dist <= 0:
            _tp_for_display = None
        else:
            _tp_for_display = ((entry_price + _tp_ratio_choice * _entry_stop_dist)
                               if pos_dir == 'long'
                               else (entry_price - _tp_ratio_choice * _entry_stop_dist))

        if _tp_for_display is None:
            tp_td = ('<td><span style="color:#888780">—</span>'
                     '<br><span style="font-size:10px;color:#888780">trail only</span></td>')
        else:
            if live_price and live_price > 0:
                _tp_move = ((_tp_for_display - live_price) / live_price * 100
                            if pos_dir == 'long'
                            else (live_price - _tp_for_display) / live_price * 100)
                _tp_sign = '+' if _tp_move >= 0 else ''
                tp_sub   = f'{_tp_sign}{_tp_move:.2f}% to TP ({_tp_ratio_choice}:1)'
            else:
                tp_sub = f'{_tp_ratio_choice} × stop dist'
            tp_td = (f'<td>{_fmt_price(_tp_for_display)}'
                     f'<br><span style="font-size:10px;color:#888780">{tp_sub}</span></td>')

        # Regime badge cell — current live regime
        if _regime_now:
            regime_td = '<td><span class="badge badge-LONG">STRONG BULL</span></td>'
        else:
            regime_td = ('<td><span class="badge" style="background:#f1efe8;color:#5F5E5A">'
                         'CHOP / BEAR</span></td>')

        try:
            days_held = (_date.today() - _date.fromisoformat(pos['entry_date'])).days
        except Exception:
            days_held = '—'

        _pos_rows += f"""
        <tr>
          <td class="asset-name">{escape(_sym_for_pos)}</td>
          <td>{_dir_badge}</td>
          <td>{escape(pos.get('entry_date','—'))}</td>
          <td>{days_held}</td>
          <td>{_fmt_price(entry_price)}</td>
          {live_td}
          {stop_td}
          {tp_td}
          {regime_td}
          <td>{leverage_multiplier:.2f}x</td>
          <td>{size_usd:,.0f}</td>
          {pnl_pct_td}
          {pnl_usd_td}
          {pos_val_td}
        </tr>"""

    if _tot_has_live_r and _tot_size_usd_r > 0:
        _tot_pct        = _tot_pnl_usd_r / _tot_size_usd_r * 100
        _tot_cls        = 'entry-t' if _tot_pct >= 0 else 'entry-f'
        _tot_sign       = '+' if _tot_pct >= 0 else ''
        _tot_pnl_pct_td = f'<td class="{_tot_cls}">{_tot_sign}{_tot_pct:.2f}%</td>'
        _tot_pnl_usd_td = f'<td class="{_tot_cls}">{_tot_sign}{_tot_pnl_usd_r:,.0f}</td>'
        _tot_pos_td     = f'<td>{_tot_pos_val_r:,.0f}</td>'
    else:
        _tot_pnl_pct_td = '<td>—</td>'
        _tot_pnl_usd_td = '<td>—</td>'
        _tot_pos_td     = f'<td>{_tot_pos_val_r:,.0f}</td>'

    _total_row = f"""
        <tr class="row-total">
          <td>Total</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
          <td>—</td><td>—</td><td>—</td><td>—</td>
          <td>{_tot_size_usd_r:,.0f}</td>
          {_tot_pnl_pct_td}{_tot_pnl_usd_td}{_tot_pos_td}
        </tr>"""

    st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Symbol</th><th>Dir</th><th>Entry date</th><th>Days</th>
      <th>Entry ($)</th><th>Live ($)</th>
      <th>Stop ($)</th><th>TP ($)</th><th>Regime</th>
      <th>Leverage</th><th>Size ($)</th>
      <th>P&amp;L (%)</th><th>P&amp;L ($)</th><th>Position ($)</th>
    </tr></thead>
    <tbody>{_pos_rows}{_total_row}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Active Positions ──────────────────────────────────────────────────────────

st.markdown("#### ACTIVE POSITIONS")

# Discretionary TP-ratio explorer.  Display-only: lets the trader see what
# the take-profit target would be at lower R:R ratios for each open position
# (useful when looking at the chart and considering a partial / early exit).
# The actual stored TP is always the entry-time 6:1 — this never writes any
# state, just changes the rendered value in the TP column.
_tp_ratio_choice = st.radio(
    "Display TP at ratio",
    options=[3, 4, 5, 6],
    index=3,                 # default = 6:1, the strategy's actual target
    horizontal=True,
    key="bb_active_tp_ratio",
    help=("Display-only: re-derives the take-profit target at the chosen "
          "reward:risk ratio so you can eyeball where a 3:1 / 4:1 / 5:1 "
          "discretionary exit would sit relative to the live price.  Stored "
          "TP and exit logic are unaffected."),
)
# Auto-ratchet: propose improved stops based on latest signal (writes positions.json).
# Direction-aware — long stops only move up, short stops only move down.
_confirm_items = []
for pid, pos in _open.items():
    _sym_for_pos = pos.get('symbol', pid)
    pos_dir      = pos.get('direction', 'long')
    conf_stop    = pos.get('current_stop')
    pending_stop = pos.get('pending_stop')
    coin_sig     = _coin_sig_by_sym.get(_sym_for_pos, {})
    sugg_stop    = coin_sig.get('current_stop') if coin_sig.get('decision') == 'HOLD' else None
    if sugg_stop is not None and sugg_stop != pending_stop:
        _improves = (
            (pos_dir == 'long'  and sugg_stop > (conf_stop or 0)) or
            (pos_dir == 'short' and (conf_stop is None or sugg_stop < conf_stop))
        )
        if _improves:
            pending_stop = sugg_stop
            _write_pending_stop(pid, sugg_stop)
    _needs_confirm = pending_stop is not None and pending_stop != conf_stop
    if _needs_confirm:
        _confirm_items.append((pid, _sym_for_pos, pending_stop))

# Live-price table — rerenders every 30s independently of the rest of the page
_bb_positions_fragment()

# Confirm stop buttons — outside fragment so they trigger full page reruns
if _confirm_items:
    _conf_cols = st.columns(len(_confirm_items))
    for _col, (_pid, _sym, _pend) in zip(_conf_cols, _confirm_items):
        with _col:
            if st.button(f"✓ Confirm stop {_sym}: {_fmt_price(_pend)}",
                         key=f"bb_conf_stop_{_pid}"):
                _confirm_stop(_pid)
                invalidate_trade_caches()
                st.rerun()


# ── Section 1: Decisions ──────────────────────────────────────────────────────
#
# Heading + manual-arm toggle render BEFORE the row loop so each row can
# decide between (a) the strategy's real decision and (b) a simulated ENTRY
# in the user's chosen direction.  The toggle's session-state key is shared
# with the 1H Setup table further down so a single click drives both views.

st.markdown("#### DECISIONS")
st.caption(f"Sizes based on realised capital: ${_realised_capital:,.2f}")

_tog_dec1, _tog_dec2, _tog_dec3 = st.columns([3, 3, 6])
with _tog_dec1:
    _force_armed = st.checkbox(
        "Manually arm 4H setup",
        key="bb_force_armed",
        help=("Fill in a simulated ENTRY for any coin that's currently IDLE "
              "(no real position, no real arm) so you can see what the "
              "direction, leverage, $ size, stop, and take-profit would be "
              "if you placed a discretionary trade now.  Also drives the "
              "1H Setup table below.  Real positions and real strategy arms "
              "are NEVER overridden — they always show their genuine "
              "direction.  Display only — no real trade is placed."),
    )
_force_dir = None
with _tog_dec2:
    if _force_armed:
        _force_dir = st.radio(
            "Direction",
            ["Long", "Short"],
            key="bb_force_armed_dir",
            horizontal=True,
            label_visibility="collapsed",
        )

rows_html = ''
for c in coin_rows:
    sig = c['sig']
    d   = sig['decision']

    alloc_note = f"{c['coin_capital']:,.0f} alloc ({c['coin_weight']*100:.0f}% of ${_realised_capital:,.0f})"

    # ── Decide if we're rendering a prospective ENTRY for this coin ───────
    # Mirrors the 1H Setup table's effective-state logic so both tables
    # always agree on STATE and direction.  The decision badge follows the
    # same vocabulary as 1H Setup's State column:
    #
    #   Strategy state              1H Setup state   Decisions badge
    #   ──────────────────────────  ───────────────  ──────────────────────
    #   Real position open          IN POSITION x    HOLD / EXIT  (apply_decision)
    #   Strategy just fired entry   IN POSITION x    ENTRY        (apply_decision)
    #   Real arm, 1H entry FALSE    ARMED x          ARMED x
    #   Real arm, 1H entry TRUE     would transition (handled above as ENTRY)
    #   Sim arm, 1H entry FALSE     ARMED x (sim)    ARMED x (sim)
    #   Sim arm, 1H entry TRUE      ARMED x (sim)    ENTRY x (sim)
    #   Idle, no toggle             IDLE             FLAT
    _real_setup_active = bool(sig.get('setup_active', False))
    _real_setup_dir    = int(sig.get('setup_direction', 0))
    _is_real_armed     = (d == 'FLAT' and _real_setup_active and _real_setup_dir != 0)
    _is_toggle_sim     = (d == 'FLAT' and _force_armed and not _is_real_armed)
    _sim_entry         = (_is_real_armed or _is_toggle_sim)

    if _sim_entry:
        _sim_dir_int    = (_real_setup_dir if _is_real_armed
                           else (1 if _force_dir == 'Long' else -1))
        _sim_close      = sig.get('close', 0.0)
        _sim_prev_close = sig.get('prev_close', 0.0)
        _sim_trail_dist = sig.get('trail_dist') or 0.0

        # Prospective leverage = risk_per_trade / (ATR / close), clipped to
        # [0.1, max_leverage].  Always computed inline from the (still-fresh)
        # h1_atr and close in `sig`, regardless of whether the cached
        # `prospective_leverage` field exists — that field can persist as 0
        # or None in stale `load_all` caches and we don't want to trust it.
        _params  = c.get('all_params', {})
        _risk    = float(_params.get('risk_per_trade', 0.03))
        _max_lev = float(_params.get('max_leverage',   2.5))
        _h1_atr  = sig.get('h1_atr')
        if _h1_atr and _h1_atr > 0 and _sim_close > 0:
            _raw_lev = _risk / (_h1_atr / _sim_close)
            _sim_lev = max(0.1, min(_max_lev, _raw_lev))
        else:
            # No usable ATR — neutral default rather than depending on
            # `prospective_leverage` from the (possibly stale) cache.
            _sim_lev = 1.0
        # Final defence — never let `_sim_lev` be ≤ 0 for any code path
        # reason.  This is what kept rendering as "0.00x" before.
        if not _sim_lev or _sim_lev <= 0:
            _sim_lev = 1.0

        _sim_coin_cap = c.get('coin_capital', 0.0)
        _sim_size_usd = _sim_lev * _sim_coin_cap
        _sim_stop = ((_sim_close - _sim_trail_dist) if _sim_dir_int == 1
                     else (_sim_close + _sim_trail_dist)) if _sim_trail_dist > 0 else None
        # TP — None if currently in strong-bull regime (trail-only path).
        _sim_tp_dist = sig.get('tp_distance')
        if _sim_tp_dist is None or _sim_trail_dist <= 0:
            _sim_tp = None
        else:
            _sim_tp = ((_sim_close + 6.0 * _sim_trail_dist) if _sim_dir_int == 1
                       else (_sim_close - 6.0 * _sim_trail_dist))

        # ── Direction-aware 1H entry-condition recompute ─────────────────
        # Same logic as the 1H Setup loop; tells us whether ALL P1+P2
        # conditions actually fire for this direction.  Real arms only
        # reach this branch when the entry HASN'T fired (otherwise
        # apply_decision would have returned 'ENTRY' already), so for
        # real arms _entry_would_fire is effectively always False here.
        _sim_sma_off  = sig.get('sma_offset_bps')
        _sim_over_thr = sig.get('overshoot_bps', 0.0)
        if _sim_dir_int == 1:
            _sim_overshoot   = (_sim_sma_off is not None
                                and not (isinstance(_sim_sma_off, float) and math.isnan(_sim_sma_off))
                                and _sim_sma_off < -_sim_over_thr)
            _sim_momentum_ok = _sim_close > _sim_prev_close
        else:
            _sim_overshoot   = (_sim_sma_off is not None
                                and not (isinstance(_sim_sma_off, float) and math.isnan(_sim_sma_off))
                                and _sim_sma_off > _sim_over_thr)
            _sim_momentum_ok = _sim_close < _sim_prev_close
        _sim_in_zone     = bool(sig.get('in_zone', False))
        _sim_pullback_ok = bool(sig.get('pullback_ok', False))
        _entry_would_fire = bool(_sim_in_zone and _sim_momentum_ok
                                 and _sim_pullback_ok and not _sim_overshoot)

    # Decision badge cell — match the 1H Setup state vocabulary so the
    # two tables never disagree on whether something is ARMED vs ENTRY.
    if _is_real_armed:
        _dir_lbl = 'LONG' if _real_setup_dir == 1 else 'SHORT'
        badge = f'<span class="badge badge-LONG">ARMED {_dir_lbl}</span>'
    elif _is_toggle_sim:
        _dir_lbl = 'LONG' if _sim_dir_int == 1 else 'SHORT'
        if _entry_would_fire:
            # All 1H conditions align in the simulated direction → entry
            # would fire if the strategy were really armed this way.
            badge = (f'<span class="badge" style="background:#FAEEDA;color:#854F0B">'
                     f'ENTRY {_dir_lbl} (sim)</span>')
        else:
            # Armed but at least one 1H condition is failing.
            badge = (f'<span class="badge" style="background:#FAEEDA;color:#854F0B">'
                     f'ARMED {_dir_lbl} (sim)</span>')
    else:
        badge = f'<span class="badge badge-{d}">{d}</span>'

    # Direction
    if _sim_entry:
        dir_html = ('<span class="badge badge-LONG">LONG</span>' if _sim_dir_int == 1
                    else '<span class="badge badge-SHORT">SHORT</span>')
    else:
        _dir_int = sig.get('direction', 0)
        if _dir_int == 1:
            dir_html = '<span class="badge badge-LONG">LONG</span>'
        elif _dir_int == -1:
            dir_html = '<span class="badge badge-SHORT">SHORT</span>'
        else:
            dir_html = '—'

    # Size (leverage)
    if _sim_entry:
        _size_sub = 'on entry' if _is_real_armed else 'if armed'
        size_pct = (f"{_sim_lev:.2f}x"
                    f'<br><span style="font-size:11px;color:#888780">{_size_sub}</span>')
    else:
        lev = sig.get('leverage_multiplier')
        # Treat lev == 0 the same as missing — a stored leverage of 0
        # means a broken / drained-to-zero position record, not a real
        # zero-leverage hold.  Renders "—" instead of misleading "0.00x".
        size_pct = f"{lev:.2f}x" if lev is not None and lev > 0 else '—'
        if d == 'HOLD' and lev is not None and lev > 0:
            size_pct += '<br><span style="font-size:11px;color:#888780">held</span>'
        elif d == 'EXIT' and lev is not None and lev > 0:
            size_pct += '<br><span style="font-size:11px;color:#888780">to exit</span>'

    # Size ($)
    if _sim_entry:
        size_usd = f"{_sim_size_usd:,.0f}"
    else:
        size_usd = f"{sig['size_usd']:,.0f}" if sig.get('size_usd') is not None else '—'

    # Stop ($)
    if _sim_entry:
        stop_html = _fmt_price(_sim_stop) if _sim_stop is not None else '—'
    else:
        _stop_val = sig.get('current_stop')
        if _stop_val is not None and _stop_val > 0:
            stop_html = _fmt_price(_stop_val)
            if d == 'HOLD' and sig.get('stop_updated'):
                stop_html += f' <span class="stop-up">↑</span><br><span class="stop-prev">was {_fmt_price(sig["old_stop"])}</span>'
        else:
            stop_html = '—'

    # TP ($) — new column
    if _sim_entry:
        if _sim_tp is None:
            tp_html = ('<span style="color:#888780">—</span>'
                       '<br><span style="font-size:10px;color:#888780">trail only</span>')
        else:
            tp_html = (f'{_fmt_price(_sim_tp)}'
                       f'<br><span style="font-size:10px;color:#888780">6× stop dist</span>')
    else:
        # Real position TP: read from positions.json via _coin_sig_by_sym → primary
        _open_for_coin = [p for _pid, p in _open.items() if p.get('symbol', _pid) == c['symbol']]
        _stored_tp     = _open_for_coin[0].get('take_profit') if _open_for_coin else None
        if _stored_tp is None:
            tp_html = '<span style="color:#888780">—</span>'
        else:
            tp_html = _fmt_price(float(_stored_tp))

    _lp       = _live_prices.get(c['symbol'])
    live_html = _fmt_price(_lp) if _lp is not None else '<span style="color:#888780">—</span>'

    # Row tint — keep neutral for FLAT and simulated rows; use the existing
    # row-{decision} class for real ENTRY/HOLD/EXIT.
    row_cls = f'class="row-{d}"' if d not in ('FLAT',) and not _sim_entry else ''

    rows_html += f"""
    <tr {row_cls}>
      <td>
        <div class="asset-name">{escape(c['symbol'])}</div>
        <div class="asset-alloc">{escape(alloc_note)}</div>
      </td>
      <td>{badge}</td>
      <td>{dir_html}</td>
      <td class="r">{size_pct}</td>
      <td class="r">{size_usd}</td>
      <td class="r">{stop_html}</td>
      <td class="r">{tp_html}</td>
      <td class="r">{_fmt_price(sig['close'])}</td>
      <td class="r">{live_html}</td>
    </tr>"""

st.markdown(f"""
<div class="dashboard-card">
  <div class="table-scroll">
  <table class="dash-table" style="width:auto;min-width:100%;white-space:nowrap">
    <thead><tr>
      <th>Asset</th><th>Decision</th><th>Direction</th>
      <th class="r">Size (leverage)</th><th class="r">Size ($)</th>
      <th class="r">Stop ($)</th><th class="r">TP ($)</th>
      <th class="r">Last 1H Close ($)</th>
      <th class="r">Live ($)</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Trade Log Forms ───────────────────────────────────────────────────────────

st.markdown("#### TRADE LOG")

_positions_for_forms = _positions_now   # reuse the single render-time read
_form_cols = st.columns(max(len(coin_rows), 1))

for _fi, _c in enumerate(coin_rows):
    with _form_cols[_fi]:
        _sym      = _c['symbol']
        _sig      = _c['sig']
        _decision = _sig['decision']
        _strategy = _c['strategy']

        _default_price = _c['exec_price'] if _c['exec_price'] is not None else _sig['close']

        _open_for_sym = {
            pid: p for pid, p in _positions_for_forms.items()
            if p.get('symbol', pid) == _sym and p.get('in_position')
        }
        _primary_pos  = next(iter(_open_for_sym.values()), {})
        _has_position = bool(_open_for_sym)

        if _decision == 'ENTRY':
            _default_size = _sig.get('leverage_multiplier') or 0.0
        elif _has_position:
            _default_size = (
                _primary_pos.get('leverage_multiplier')
                or _primary_pos.get('size_pct')
                or _sig.get('leverage_multiplier')
                or 0.0
            )
        else:
            _default_size = 0.0

        _stored_note = _primary_pos.get('discretion_note') or ''

        st.markdown(
            f'<div class="form-card-header">'
            f'{escape(_sym)} &nbsp;<span class="badge badge-{_decision}">{_decision}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        _action_key     = f'bb_action_radio_{_sym}'
        _action_options = ['ENTRY', 'EXIT'] if _has_position else ['ENTRY']
        _action_default = 1 if (_has_position and _decision in ('HOLD', 'EXIT')) else 0
        _action_radio   = st.radio(
            '',
            _action_options,
            index=min(_action_default, len(_action_options) - 1),
            key=_action_key,
            horizontal=True,
            label_visibility='collapsed',
        )
        _is_exit = (_action_radio == 'EXIT')

        _selected_pid = None
        _selected_pos = {}
        if _is_exit:
            if len(_open_for_sym) == 0:
                st.caption('No open positions to exit.')
            elif len(_open_for_sym) == 1:
                _selected_pid = next(iter(_open_for_sym))
                _selected_pos = _open_for_sym[_selected_pid]
            else:
                _pid_labels = {
                    pid: f"{pid}  —  {pos.get('entry_date','?')} @ ${pos.get('entry_price',0):,.4f}"
                    for pid, pos in _open_for_sym.items()
                }
                _selected_pid = st.selectbox(
                    'Position to exit',
                    options=list(_pid_labels.keys()),
                    format_func=lambda k: _pid_labels[k],
                    key=f'bb_exit_pid_{_sym}',
                )
                _selected_pos = _open_for_sym[_selected_pid]
            _held_lev = float(
                _selected_pos.get('leverage_multiplier')
                or _selected_pos.get('size_pct')
                or _default_size
                or 1.0
            )
        else:
            _held_lev = float(_default_size) if _default_size > 0 else 1.0

        with st.form(key=f'bb_trade_form_{_sym}'):
            if not _is_exit:
                # ── ENTRY form ────────────────────────────────────────────────
                _ent_type_default = 0 if _sig.get('entry_long') or _sig.get('entry_short') else 1
                st.radio(
                    'Entry type',
                    ['Strategy', 'Discretionary'],
                    index=_ent_type_default,
                    horizontal=True,
                    key=f'bb_entry_type_{_sym}',
                )
                # Default direction from strategy signal
                _sig_dir_int = _sig.get('direction', 0)
                _dir_default = 0 if (_sig.get('entry_long') or _sig_dir_int == 1) else 1
                _direction = st.radio(
                    'Direction',
                    ['Long', 'Short'],
                    index=_dir_default,
                    horizontal=True,
                    key=f'bb_direction_{_sym}',
                )
                _size = st.number_input(
                    'Actual leverage (x)',
                    value=float(round(max(_default_size, 0.01), 2)),
                    min_value=0.01,
                    format='%.2f',
                    key=f'bb_size_{_sym}',
                )
                _price = st.number_input(
                    'Actual price',
                    value=float(round(_default_price, 2)),
                    step=0.01,
                    format='%.6f',
                    key=f'bb_price_{_sym}',
                )
                _exit_lev = None
            else:
                # ── EXIT form ─────────────────────────────────────────────────
                _exit_reason_widget = st.radio(
                    'Exit reason',
                    ['Strategy', 'Discretionary'],
                    key=f'bb_exit_reason_{_sym}',
                    horizontal=True,
                )
                _exit_lev = st.number_input(
                    'Exit leverage (x)',
                    value=float(round(_held_lev, 2)),
                    min_value=0.01,
                    max_value=_held_lev,
                    format='%.2f',
                    key=f'bb_exit_lev_{_sym}',
                )
                _price = st.number_input(
                    'Actual price',
                    value=float(round(_default_price, 2)),
                    step=0.01,
                    format='%.6f',
                    key=f'bb_price_{_sym}',
                )
                _direction = None
                _size      = _held_lev

            _disc = st.text_input('Discretion note', key=f'bb_note_{_sym}')
            _submitted = st.form_submit_button('Log trade', use_container_width=True)

        if _stored_note:
            with st.expander('📋 stored note'):
                st.caption(_stored_note)

        if _submitted:
            _action_final  = st.session_state.get(_action_key, 'ENTRY')
            _is_exit_final = (_action_final == 'EXIT')

            # The strategy only writes a non-zero `stop` on entry-fire / HOLD
            # bars.  For discretionary entries (which fire on any bar) and
            # trade-log defaults, derive the theoretical stop from the user's
            # actual entry price and the live trail distance.
            _trail_dist = _sig.get('trail_dist', 0.0) or 0.0
            if _is_exit_final:
                _theo_stop = _sig.get('stop', 0.0) or 0.0
            else:
                _dir_for_stop = (_direction or 'Long').lower()
                if _trail_dist > 0 and _price > 0:
                    _theo_stop = ((_price - _trail_dist) if _dir_for_stop == 'long'
                                  else (_price + _trail_dist))
                else:
                    _theo_stop = 0.0

            if _is_exit_final:
                _exit_amount   = _exit_lev if _exit_lev is not None else _held_lev
                _is_full_exit  = (_exit_amount >= _held_lev * 0.999)
                _exit_type_log = 'full' if _is_full_exit else 'partial'
                _pid_to_exit   = st.session_state.get(f'bb_exit_pid_{_sym}', _selected_pid)
                if _pid_to_exit is None and _open_for_sym:
                    _pid_to_exit = next(iter(_open_for_sym))
                _sub_sig = ('EXIT', round(_price, 6), round(_exit_amount, 4), _pid_to_exit)
            else:
                # Reject zero-leverage / zero-price entries before any
                # record is written.  Zero-leverage entries used to slip
                # through and create an in_position=True row with size_usd=0,
                # which then dashed out the P&L cells and crashed the divide.
                if not _size or _size <= 0:
                    st.error(f"{_sym}: leverage must be > 0 to log an entry.")
                    st.stop()
                if not _price or _price <= 0:
                    st.error(f"{_sym}: price must be > 0 to log an entry.")
                    st.stop()
                _dir_val = (_direction or 'Long').lower()
                _sub_sig = ('ENTRY', round(_price, 6), round(_size, 4), _dir_val)

            _guard_key = f'bb_last_trade_{_sym}'
            if st.session_state.get(_guard_key) != _sub_sig:
                st.session_state[_guard_key] = _sub_sig

                if _is_exit_final and _pid_to_exit:
                    _write_trade(
                        position_id=_pid_to_exit,
                        action='EXIT',
                        strategy=_strategy,
                        theoretical_price=_default_price,
                        actual_price=_price,
                        theoretical_leverage=_held_lev,
                        actual_leverage=_exit_amount,
                        theoretical_stop=_theo_stop,
                        discretion_note=_disc,
                        exit_type=_exit_type_log,
                        exit_leverage=_exit_amount,
                        exit_close=_sig['close'],
                        exit_reason=st.session_state.get(f'bb_exit_reason_{_sym}', 'Strategy exit'),
                    )
                    if _is_full_exit:
                        _write_position_exit(_pid_to_exit, _price, _disc)
                    else:
                        _write_position_partial_exit(_pid_to_exit, _exit_amount, _price, _disc)
                    # ── Update realised capital with exit P&L ─────────────────
                    _exited_pos   = _open_for_sym.get(_pid_to_exit, {})
                    _entry_p      = _exited_pos.get('entry_price', 0)
                    _full_sz_usd  = (_exited_pos.get('size_usd')
                                     or get_coin_capital(_sym) * _held_lev)
                    _frac_closed  = _exit_amount / _held_lev if _held_lev > 0 else 1.0
                    _dir_sign     = 1 if _exited_pos.get('direction', 'long') == 'long' else -1
                    _pnl_usd_exit = (
                        _dir_sign * (_price - _entry_p) / _entry_p * _full_sz_usd * _frac_closed
                        if _entry_p > 0 else 0.0
                    )
                    _old_rc = load_realised_capital(DATA_DIR)
                    _new_rc = update_realised_capital(DATA_DIR, _pnl_usd_exit, _pid_to_exit)
                    print(f"Capital updated: ${_old_rc:.2f} -> ${_new_rc:.2f} "
                          f"(trade: {_pid_to_exit}, P&L: ${_pnl_usd_exit:+.2f})")
                elif not _is_exit_final:
                    _positions_fresh = load_positions(DATA_DIR)
                    _new_pid         = _next_position_id(_sym, _positions_fresh)
                    _signal_snapshot = {
                        'close':         float(_sig['close']),
                        'position':      int(_sig.get('position', 0)),
                        'entry_long':    bool(_sig.get('entry_long', False)),
                        'entry_short':   bool(_sig.get('entry_short', False)),
                        'bb_upper':      float(_sig.get('bb_upper', 0)),
                        'bb_mid':        float(_sig.get('bb_mid', 0)),
                        'bb_lower':      float(_sig.get('bb_lower', 0)),
                        'bb_width':      float(_sig.get('bb_width', 0)),
                        'sma':           float(_sig.get('sma', 0)),
                        'stop':          float(_sig.get('stop', 0)),
                    }
                    _snap_coin_cap = get_coin_capital(_sym)
                    _snap_size_usd = _snap_coin_cap * _size
                    _snap_weight   = COIN_WEIGHTS.get(_sym)
                    _write_trade(
                        position_id=_new_pid,
                        action='ENTRY',
                        strategy=_strategy,
                        theoretical_price=_default_price,
                        actual_price=_price,
                        theoretical_leverage=_default_size,
                        actual_leverage=_size,
                        theoretical_stop=_theo_stop,
                        discretion_note=_disc,
                        direction=_dir_val,
                        entry_close=_sig['close'],
                        signal_snapshot=_signal_snapshot,
                        entry_type=st.session_state.get(f'bb_entry_type_{_sym}', 'Strategy'),
                        coin_capital=_snap_coin_cap,
                        size_usd=_snap_size_usd,
                        capital_total=load_realised_capital(DATA_DIR),
                        coin_weight=_snap_weight,
                    )
                    # Compute and persist take-profit at entry time (regime-
                    # aware): no TP if currently in strong-bull regime, else
                    # 6× stop distance from the user's entry price.
                    _regime_at_entry = bool(_sig.get('regime_strong_bull', False))
                    _stop_dist       = abs(_price - _theo_stop) if _theo_stop else 0.0
                    if _regime_at_entry or _stop_dist <= 0:
                        _tp_at_entry  = None
                        _regime_label = 'strong_bull'
                    else:
                        _tp_at_entry  = (_price + 6 * _stop_dist) if _dir_val == 'long' \
                                        else (_price - 6 * _stop_dist)
                        _regime_label = 'chop_bear'

                    _write_position_entry(
                        _new_pid, _sym, _price, _size, _theo_stop,
                        _strategy, _disc, direction=_dir_val,
                        coin_capital=_snap_coin_cap,
                        size_usd=_snap_size_usd,
                        take_profit=_tp_at_entry,
                        regime_at_entry=_regime_label,
                    )
            invalidate_trade_caches()
            st.rerun()


# ── Section 2: Entry Conditions ───────────────────────────────────────────────
#
# Split into two tables that mirror the strategy's two-stage state machine:
#   1. Stage 1 — Engine Room (4H setup arms the trade)
#   2. Stage 2 — Buy the Dip (1H trigger fires + spoilers that drop the watch)


def _fv(v, dp=4):
    """Format a float; '—' for NaN/None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    return f'{v:,.{dp}f}'


def _bool_td(flag):
    """Coloured TRUE / FALSE cell — matches momentum's _bg_box."""
    cls = 'entry-t' if flag else 'entry-f'
    return f'<td class="c {cls}">{"TRUE" if flag else "FALSE"}</td>'


def _pass_td(passed, label_ok='OK', label_bad='VIOLATED'):
    """Coloured pass/fail cell with custom labels."""
    cls = 'entry-t' if passed else 'entry-f'
    return f'<td class="c {cls}">{label_ok if passed else label_bad}</td>'


def _muted_td(text='—', cls='c'):
    return f'<td class="{cls}" style="color:#888780">{escape(str(text))}</td>'


def _value_threshold_td(value, threshold, passed, value_fmt, threshold_prefix='≤'):
    """
    Right-aligned cell rendering "value / ≤ threshold" with green/red tint
    on the value depending on `passed`.  Used for in-zone / pullback / etc.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return _muted_td()
    val_cls   = 'stat-pos' if passed else 'stat-neg'
    val_color = '#3B6D11' if passed else '#A32D2D'
    return (
        f'<td class="r" style="color:{val_color};font-weight:600">'
        f'{value_fmt.format(value)} '
        f'<span style="font-size:10px;color:#888780;font-weight:400">'
        f'/ {threshold_prefix} {value_fmt.format(threshold)}</span></td>'
    )


# ── Section 2 header ──────────────────────────────────────────────────────────

st.markdown("#### ENTRY CONDITIONS")


# ── Table 1: 4H Engine Room ───────────────────────────────────────────────────

engine_rows_html = ''
for c in coin_rows:
    sig = c['sig']

    h4_dir = sig.get('h4_dir', 'mixed')
    if sig.get('h4_two_big_green'):
        dir_label = 'Both green'
    elif sig.get('h4_two_big_red'):
        dir_label = 'Both red'
    else:
        dir_label = 'Mixed'

    # ── Plain numeric cell (no green/red tint — matches momentum's pattern) ──
    def _num_td(v, fmt='{:.2f}'):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return '<td class="r" style="color:#888780">—</td>'
        return f'<td class="r">{fmt.format(v)}</td>'

    def _num(v, fmt='{:.2f}'):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return '—'
        return fmt.format(v)

    # C1 — Range ratio shown for BOTH 4H bars (current / prior) so the
    # "both must be big" requirement is visible at a glance.
    c1_pass        = bool(sig.get('c1_two_big_same_dir', False))
    _c1_ratio_curr = sig.get('c1_range_ratio')
    _c1_ratio_prev = sig.get('c1_range_ratio_prev')
    c1_ratio_td    = (f'<td class="r">{_num(_c1_ratio_curr)} '
                      f'<span style="font-size:10px;color:#888780">/ '
                      f'{_num(_c1_ratio_prev)}</span></td>')

    # 4H bar direction indicator — last three bars' colours, ordered
    # oldest → newest (left → right).  Direction inference for slope/setup
    # still uses only the LAST TWO bars (curr + prev); the third is shown
    # for context only.
    def _dir_letter(d):
        if d == 'green': return '<span style="color:#3B6D11;font-weight:700">G</span>'
        if d == 'red':   return '<span style="color:#A32D2D;font-weight:700">R</span>'
        # Doji / missing: bolded so the slot is unmistakably present in
        # the row even when sig is from a stale cache and lacks the field.
        return '<span style="color:#888780;font-weight:700">−</span>'
    bars_dir_td = (f'<td class="c">'
                   f'{_dir_letter(sig.get("h4_prev2_dir", "doji"))} '
                   f'{_dir_letter(sig.get("h4_prev_dir",  "doji"))} '
                   f'{_dir_letter(sig.get("h4_curr_dir",  "doji"))}'
                   f'</td>')

    # C1 boolean
    c1_td = _bool_td(c1_pass)

    # C2 — BB expansion
    c2_pass     = bool(sig.get('c2_bb_expanding', False))
    c2_ratio_td = _num_td(sig.get('c2_bb_ratio'))
    c2_td       = _bool_td(c2_pass)

    # C3 — slope OK.  Show raw slope_norm with the per-asset epsilon
    # threshold inline (eps varies between assets — making it explicit
    # lets the user verify the cutoff).  Direction context lives in the
    # Bars column to the left, so the boolean cell stays narrow.
    c3_pass    = bool(sig.get('c3_slope_ok', False))
    slope_norm = sig.get('h4_slope_norm', 0.0)
    slope_eps  = sig.get('slope_epsilon', 0.0)
    c3_ratio_td = (f'<td class="r">{slope_norm:+.5f} '
                   f'<span style="font-size:10px;color:#888780">'
                   f'/ ε±{slope_eps:.5f}</span></td>')
    c3_td      = _bool_td(c3_pass)

    # 4H ADX — value and adx_strong threshold in a single cell, then the
    # boolean leg.  Matches the +DI / −DI two-value-one-cell pattern.
    adx_val      = sig.get('h4_adx',     0.0)
    adx_thresh   = sig.get('adx_strong', 0.0)
    adx_strong_p = bool(sig.get('adx_strong_pass', False))
    adx_td       = (f'<td class="r">{adx_val:.2f} '
                    f'<span style="font-size:10px;color:#888780">/ {adx_thresh:.2f}</span></td>')
    adx_pass_td  = _bool_td(adx_strong_p)

    # +DI / −DI combined into a single cell to reduce clutter.  The
    # "+DI > −DI" boolean still has its own column for the verdict.
    plus_di    = sig.get('h4_plus_di',  0.0)
    minus_di   = sig.get('h4_minus_di', 0.0)
    di_bull    = bool(sig.get('plus_di_dominant', False))
    di_vals_td = (f'<td class="r">{plus_di:.2f} '
                  f'<span style="font-size:10px;color:#888780">/ {minus_di:.2f}</span></td>')
    di_pass_td = _bool_td(di_bull)

    # Bull-regime veto — only matters for shorts.  For longs / mixed, show muted "—".
    bull_veto = bool(sig.get('bull_veto_active', False))
    if h4_dir == 'short':
        # When direction is short: TRUE = vetoes the short = caution colour
        if bull_veto:
            veto_td = '<td class="c caution">VETOES SHORT</td>'
        else:
            veto_td = '<td class="c entry-t">CLEAR</td>'
    else:
        veto_td = _muted_td('n/a (long dir)')

    # Setup-would-arm column — based on the LAST 4H bar's conditions
    if c1_pass and c2_pass and c3_pass:
        if h4_dir == 'long':
            setup_td = '<td class="c entry-t">ARMS LONG</td>'
        elif h4_dir == 'short' and not bull_veto:
            setup_td = '<td class="c entry-t">ARMS SHORT</td>'
        elif h4_dir == 'short' and bull_veto:
            setup_td = '<td class="c caution">SHORT VETOED</td>'
        else:
            setup_td = _muted_td('—')
    else:
        setup_td = _muted_td('—')

    engine_rows_html += f"""
    <tr>
      <td class="asset-name">{escape(c['symbol'])}</td>
      {c1_ratio_td}
      {bars_dir_td}
      {c1_td}
      {c2_ratio_td}
      {c2_td}
      {c3_ratio_td}
      {c3_td}
      {adx_td}
      {adx_pass_td}
      {di_vals_td}
      {di_pass_td}
      {veto_td}
      {setup_td}
    </tr>"""

col_title, col_help = st.columns([1, 6])
with col_title:
    st.markdown('<div class="sub-section-label">4H Setup</div>', unsafe_allow_html=True)
with col_help:
    with st.expander("Explanation"):
        st.markdown("""
The 4H "Engine Room" arms the trade when **C1 + C2 + C3** all pass.
Values are from the **last fully-closed 4H bar**.

**C1 — 2 Big Same Dir:** Range/thresh `> 1.00` for **both** bars AND
Bars are `G G` (→ long) or `R R` (→ short).

**C2 — BB Expanding:** BB / mean `> 1.00`.

**C3 — Slope OK:** direction inferred from the **last 4H candle's colour only**.
- last bar green → slope `≥ −ε`  (long-direction check)
- last bar red   → slope `≤ +ε`  (short-direction check)
- doji           → symmetric `−ε ≤ slope ≤ +ε`
ε is per-asset and fixed.

**Bull veto (shorts only):** rejects a short setup if
`(close > 1H trend MA) OR (ADX > adx_strong AND +DI > −DI)`.
Don't fade strong bulls.

**After exit:** must wait for two new big 4H bars before re-arming.

---

#### What to look for

| Column | Pass when |
|---|---|
| Range / thresh | both ratios `> 1.00` (current and prior bar) |
| Bars | last 3 candle colours (oldest → newest); only the **last two** drive C1 — `G G` or `R R` to fire, mixed fails |
| BB / mean | `> 1.00` |
| Slope / ε | last-candle-aware: green ⇒ slope `≥ −ε`; red ⇒ slope `≤ +ε`; doji ⇒ symmetric |
| 4H ADX / strong | first number `>` second — ADX above its threshold |
| +DI / −DI | `+DI > −DI` ⇒ bullish bias |
| Bull veto | FALSE = clear · TRUE = blocks shorts |
| Setup | `ARMS LONG` / `ARMS SHORT` = all three conditions fired |
""")

st.markdown(f"""
<div class="dashboard-card">
  <div class="table-scroll">
  <table class="dash-table dash-table-labeled" style="width:auto;min-width:100%;white-space:nowrap">
    <thead><tr>
      <th>Asset</th>
      <th class="r">Range / thresh</th>
      <th class="c">Bars</th>
      <th class="c">C1: 2 Big</th>
      <th class="r">BB / mean</th>
      <th class="c">C2: BB Exp</th>
      <th class="r">Slope / ε</th>
      <th class="c">C3: Slope OK</th>
      <th class="r">4H ADX / strong</th>
      <th class="c">ADX strong</th>
      <th class="r">+DI / −DI</th>
      <th class="c">+DI > −DI</th>
      <th class="c">Bull veto</th>
      <th class="c">Setup</th>
    </tr></thead>
    <tbody>{engine_rows_html}</tbody>
  </table>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Table 2: 1H Buy the Dip + Spoilers ────────────────────────────────────────
#
# Heading + explanation + manual-arm toggle live BEFORE the row loop because
# the toggle controls the per-row rendering logic.

col_title, col_help = st.columns([1, 6])
with col_title:
    st.markdown('<div class="sub-section-label">1H Setup</div>', unsafe_allow_html=True)
with col_help:
    with st.expander("Explanation"):
        st.markdown("""
Once the 4H Engine Room arms a setup, the strategy switches to the 1H
timeframe and waits for the perfect pullback.  The watch lasts up to
`max_1h_bars` 1H candles.

**Timing** — Every value in this table is taken from the **last fully-closed
1H bar** (open + 1h ≤ now).  Incomplete bars are stripped before any
computation runs.

**Entry conditions (both must hold)**
- **P1 — Entry zone:** \\|close − 1H SMA\\| ≤ `entry_zone_bps`
- **P2 — Momentum resumption:** long ⇒ close > prev close;
  short ⇒ close < prev close

**Spoilers (any one drops the armed setup)**
- **E1 — Time decay:** more than `max_1h_bars` 1H bars elapse with no entry
- **E2 — Volatility spike:** 1H range > `pullback_atr_mult` × ATR
  (the pullback was too violent — likely a continuation, not a dip)
- **E3 — Structural break:** close overshoots the 1H SMA by more than
  `overshoot_bps` (price has gone through the level we wanted to buy)

**State** — `IDLE` (no 4H setup) · `ARMED LONG/SHORT` (watching 1H) ·
`IN POSITION LONG/SHORT` (already entered).  Spoiler / entry columns are
greyed when no setup is armed.

**Manual-arm toggle** — Useful for examining the 1H conditions on demand.
When enabled, every coin without an actual position is treated as ARMED
in the chosen direction; spoiler / entry / momentum cells are recomputed
against that direction.  No effect on real trading or stored state — it's
purely a visualisation aid.  Cells from a simulated arm carry a *(sim)*
tag in the State column so you don't confuse them with a real setup.
""")

# Manual-arm state is set by the toggle in the Decisions section above.
# This table just reads the same session_state values so a single click
# drives both views.
# (_force_armed and _force_dir are already in scope from the Decisions section.)

# Build a {symbol → manual direction} lookup so the 1H Setup table's
# IN POSITION badge reflects the user's real positions.json — not the
# strategy's internal `sig['position']`, which is also non-zero for
# shadow trades the user never logged manually.
_manual_open_dirs = {p.get('symbol', pid): p.get('direction', 'long')
                     for pid, p in _open.items() if p.get('in_position')}

dip_rows_html = ''
for c in coin_rows:
    sig = c['sig']

    state_badge_real = sig.get('state_badge', 'IDLE')
    setup_active     = bool(sig.get('setup_active', False))
    setup_direction  = int(sig.get('setup_direction', 0))
    # `in_position` here means a MANUAL position is open for this coin.
    # Strategy-shadow trades (sig['position'] != 0 with no manual record)
    # are NOT IN POSITION from the user's perspective — they fall through
    # to ARMED / ARMED (sim) / IDLE depending on the toggle.
    _manual_dir      = _manual_open_dirs.get(c['symbol'])
    in_position      = _manual_dir is not None

    # ── Determine effective state (real or simulated) ─────────────────────
    # Real arm always wins over the toggle.  In-position rows ignore the toggle.
    if in_position or setup_active:
        eff_active     = setup_active and not in_position
        eff_direction  = setup_direction if eff_active else 0
        eff_simulated  = False
    elif _force_armed:
        eff_active     = True
        eff_direction  = 1 if _force_dir == 'Long' else -1
        eff_simulated  = True
    else:
        eff_active     = False
        eff_direction  = 0
        eff_simulated  = False

    # ── Recompute direction-aware spoilers under the effective direction ──
    close_v        = sig.get('close', 0.0)
    prev_close_v   = sig.get('prev_close', 0.0)
    sma_off        = sig.get('sma_offset_bps')
    over_thr       = sig.get('overshoot_bps', 0.0)
    if eff_active:
        if eff_direction == 1:
            eff_overshoot   = (sma_off is not None and not (isinstance(sma_off, float) and math.isnan(sma_off))
                               and sma_off < -over_thr)
            eff_momentum_ok = close_v > prev_close_v
        else:
            eff_overshoot   = (sma_off is not None and not (isinstance(sma_off, float) and math.isnan(sma_off))
                               and sma_off > over_thr)
            eff_momentum_ok = close_v < prev_close_v
        eff_in_zone     = bool(sig.get('in_zone', False))
        eff_pullback_ok = bool(sig.get('pullback_ok', False))
        eff_entry_fires = bool(eff_in_zone and eff_momentum_ok and eff_pullback_ok and not eff_overshoot)
    else:
        eff_overshoot   = False
        eff_momentum_ok = False
        eff_in_zone     = False
        eff_pullback_ok = False
        eff_entry_fires = False

    # ── State badge cell — based on MANUAL position, not strategy state ────
    if in_position:
        if _manual_dir == 'long':
            state_td = '<td class="c entry-t">IN POSITION LONG</td>'
        elif _manual_dir == 'short':
            state_td = '<td class="c entry-f">IN POSITION SHORT</td>'
        else:
            state_td = _muted_td('IN POSITION')
    elif eff_active and eff_simulated:
        _label = 'ARMED LONG (sim)' if eff_direction == 1 else 'ARMED SHORT (sim)'
        state_td = f'<td class="c caution">{_label}</td>'
    elif eff_active:
        _label = 'ARMED LONG' if eff_direction == 1 else 'ARMED SHORT'
        state_td = f'<td class="c entry-t">{_label}</td>'
    else:
        state_td = _muted_td('IDLE')

    # ── Bars-to-expiry — only meaningful when truly armed (not simulated) ──
    if eff_active and not eff_simulated:
        bars_left = int(sig.get('bars_until_expiry') or 0)
        max_bars  = int(sig.get('max_1h_bars', 0))
        ratio     = bars_left / max_bars if max_bars > 0 else 0
        cell_color = '#854F0B' if ratio <= 0.25 else '#3B6D11'
        expiry_td = (f'<td class="r" style="color:{cell_color};font-weight:600">'
                     f'{bars_left}h '
                     f'<span style="font-size:10px;color:#888780;font-weight:400">'
                     f'/ {max_bars}h</span></td>')
    elif eff_active and eff_simulated:
        # Simulated arms have no real expiry; show the param as full-window context.
        max_bars = int(sig.get('max_1h_bars', 0))
        expiry_td = (f'<td class="r" style="color:#888780">— '
                     f'<span style="font-size:10px">/ {max_bars}h</span></td>')
    else:
        expiry_td = _muted_td()

    # ── Plain numeric cell helper (no colour — matches momentum's pattern) ──
    def _plain_num(v, fmt='{:.2f}'):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return _muted_td()
        return f'<td class="r">{fmt.format(v)}</td>'

    # Δ SMA / entry zone — plain "X bps / ≤ Y bps", momentum-style.
    if eff_active:
        _v = sig.get('dist_sma_bps'); _t = sig.get('entry_zone_bps', 0.0)
        if _v is None or (isinstance(_v, float) and math.isnan(_v)):
            zone_td = _muted_td()
        else:
            zone_td = (f'<td class="r">{_v:.0f} bps '
                       f'<span style="font-size:10px;color:#888780">/ ≤ {_t:.0f}</span></td>')
    else:
        zone_td = _muted_td()

    # Pullback range vs ATR — plain "X× / ≤ Y×".
    if eff_active:
        _v = sig.get('pullback_ratio'); _t = sig.get('pullback_atr_mult', 0.0)
        if _v is None or (isinstance(_v, float) and math.isnan(_v)):
            pull_td = _muted_td()
        else:
            pull_td = (f'<td class="r">{_v:.2f}× '
                       f'<span style="font-size:10px;color:#888780">/ ≤ {_t:.2f}×</span></td>')
    else:
        pull_td = _muted_td()

    # Close − SMA (signed bps) — plain numeric.
    sma_off_td = _plain_num(sig.get('sma_offset_bps'), fmt='{:+.0f} bps') if eff_active else _muted_td()

    # Overshoot status (boolean cell — coloured), uses effective direction
    if eff_active:
        overshoot_td = ('<td class="c entry-f">OVERSHOT</td>' if eff_overshoot
                        else '<td class="c entry-t">OK</td>')
    else:
        overshoot_td = _muted_td()

    # Δ close % vs prev close — plain numeric
    mom_pct_td = _plain_num(sig.get('momentum_pct'), fmt='{:+.2f}%') if eff_active else _muted_td()

    # Momentum boolean (effective direction)
    momentum_td = _bool_td(eff_momentum_ok) if eff_active else _muted_td()

    # Final entry trigger (effective direction).  When FALSE, list which
    # of the four 1H legs is blocking — same idea as momentum's
    # "Entry Long" cell that lists `entry_reasons` under FALSE.
    if eff_active:
        if eff_entry_fires:
            entry_td = _bool_td(True)
        else:
            _reasons = []
            if not eff_in_zone:     _reasons.append('out of zone')
            if not eff_momentum_ok: _reasons.append('no momentum')
            if not eff_pullback_ok: _reasons.append('vol spike')
            if eff_overshoot:       _reasons.append('overshot')
            _reason_html = (f'<br><span style="font-size:10px;font-weight:400">'
                            f'{escape(", ".join(_reasons))}</span>') if _reasons else ''
            entry_td = (f'<td class="c entry-f">FALSE{_reason_html}</td>')
    else:
        entry_td = _muted_td()

    dip_rows_html += f"""
    <tr>
      <td class="asset-name">{escape(c['symbol'])}</td>
      {state_td}
      {expiry_td}
      {zone_td}
      {pull_td}
      {sma_off_td}
      {overshoot_td}
      {mom_pct_td}
      {momentum_td}
      {entry_td}
    </tr>"""

st.markdown(f"""
<div class="dashboard-card">
  <div class="table-scroll">
  <table class="dash-table dash-table-labeled" style="width:auto;min-width:100%;white-space:nowrap">
    <thead><tr>
      <th>Asset</th>
      <th class="c">State</th>
      <th class="r">Bars left</th>
      <th class="r">Δ SMA <span style="font-weight:400;font-size:10px;color:#888780">(P1)</span></th>
      <th class="r">Pullback / ATR <span style="font-weight:400;font-size:10px;color:#888780">(E2)</span></th>
      <th class="r">Close − SMA</th>
      <th class="c">Overshoot <span style="font-weight:400;font-size:10px;color:#888780">(E3)</span></th>
      <th class="r">Δ close %</th>
      <th class="c">Momentum <span style="font-weight:400;font-size:10px;color:#888780">(P2)</span></th>
      <th class="c">Entry fires</th>
    </tr></thead>
    <tbody>{dip_rows_html}</tbody>
  </table>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Section 3: Exit details ───────────────────────────────────────────────────
#
# Per-coin breakdown of stop-loss math, regime detection, and the regime-aware
# take-profit target.  Mirrors the section structure of momentum's
# EXIT DETAILS table (Inputs -> Stop -> Take profit -> Regime).

with st.expander("EXIT DETAILS"):
    st.markdown("""
**Trailing ratchet stop** — Initial stop = entry ± `trail_atr_mult` × ATR_1H.
Ratchets in favour only on every closed 1H bar — never moves backward.

**Regime-aware take-profit**
- *Strong-bull regime* (close > 1H trend MA AND 4H ADX > `adx_strong` AND +DI > −DI): no TP, ride the trail.
- *Chop / Bear regime*: fixed 6:1 reward-to-risk from entry (TP = entry ± 6 × stop distance).
""")
    n        = len(coin_rows)
    col_w    = f'calc((100% - 180px) / {n})'
    coin_ths = ''.join(
        f'<th style="text-align:right;padding-right:24px">{escape(c["symbol"])}</th>'
        for c in coin_rows
    )

    def _row(label, cells):
        tds = ''.join(
            f'<td style="text-align:right;padding-right:24px">{cell}</td>'
            for cell in cells
        )
        return f'<tr><td class="field-label">{escape(label)}</td>{tds}</tr>'

    def _divider(label):
        return f'<tr class="divider-row"><td colspan="{n+1}">{escape(label)}</td></tr>'

    def _fnum(v, dp=2, suffix=''):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return '—'
        return f'{v:,.{dp}f}{suffix}'

    def _bool_cell(passed):
        # Green TRUE / red FALSE — matches momentum's _bg_box pattern.
        cls = 'entry-t' if passed else 'entry-f'
        return f'<span class="{cls}" style="padding:2px 7px;border-radius:3px;font-size:11px;font-weight:600">{"TRUE" if passed else "FALSE"}</span>'

    def _regime_badge(strong_bull):
        if strong_bull:
            return '<span class="badge badge-LONG">STRONG BULL</span>'
        return '<span class="badge" style="background:#f1efe8;color:#5F5E5A">CHOP / BEAR</span>'

    # Per-coin pre-derived cell strings, used by the row helpers below.
    def _entry_price_for(c):
        # If we hold an open position on this coin, use its entry price for
        # the TP target; otherwise fall back to the current close (i.e. "what
        # would TP look like if we entered now").
        sym  = c['symbol']
        for _pid, _pos in _open.items():
            if _pos.get('symbol', _pid) == sym and _pos.get('in_position'):
                return float(_pos['entry_price']), _pos.get('direction', 'long'), True
        return float(c['sig'].get('close', 0.0)), 'long', False

    # Build all the per-section cell lists in one pass to keep the table assembly readable.
    inputs_atr        = []
    inputs_close      = []
    inputs_trail_mult = []
    inputs_trend_ma   = []

    stop_trail_dist   = []
    stop_current      = []
    stop_dist_close   = []
    stop_ratcheted    = []

    tp_regime         = []
    tp_target         = []
    tp_dist_close     = []
    tp_rr             = []

    reg_close_vs_ma   = []
    reg_adx           = []
    reg_di            = []

    upd_yes = '<span class="upd-yes">↑ Yes</span>'
    upd_no  = '<span class="upd-no">No</span>'

    for c in coin_rows:
        sig = c['sig']
        h1_atr        = sig.get('h1_atr',         float('nan'))
        close_v       = sig.get('close',          float('nan'))
        trend_ma      = sig.get('h1_trend_ma',    float('nan'))
        trail_mult    = sig.get('trail_atr_mult', 0.0)
        trail_dist    = sig.get('trail_dist',     float('nan'))
        tp_dist       = sig.get('tp_distance')
        cur_stop      = sig.get('stop',           0.0) or 0.0
        regime_bull   = bool(sig.get('regime_strong_bull', False))
        adx_v         = sig.get('h4_adx',     0.0)
        plus_di       = sig.get('h4_plus_di', 0.0)
        minus_di      = sig.get('h4_minus_di',0.0)
        adx_strong_t  = sig.get('adx_strong', 0.0)

        # Inputs
        inputs_atr.append(_fnum(h1_atr, 2))
        inputs_close.append(_fmt_price(close_v))
        inputs_trail_mult.append(_fnum(trail_mult, 2, '×'))
        inputs_trend_ma.append(_fmt_price(trend_ma) if not math.isnan(trend_ma) else '—')

        # Stop block
        stop_trail_dist.append(_fnum(trail_dist, 2))
        if cur_stop and not math.isnan(cur_stop):
            stop_current.append(_fmt_price(cur_stop))
        else:
            stop_current.append('—')
        if cur_stop and not math.isnan(cur_stop) and not math.isnan(close_v) and close_v > 0:
            sd_pct = abs(close_v - cur_stop) / close_v * 100
            stop_dist_close.append(f'{sd_pct:.2f}%')
        else:
            stop_dist_close.append('—')
        stop_ratcheted.append(upd_yes if sig.get('stop_updated') else upd_no)

        # Take profit block
        tp_regime.append(_regime_badge(regime_bull))
        entry_p, dirn, has_pos = _entry_price_for(c)
        if regime_bull or tp_dist is None:
            tp_target.append('<span style="color:#888780">trail only</span>')
            tp_dist_close.append('—')
            tp_rr.append('<span style="color:#888780">n/a</span>')
        else:
            tp_t = entry_p + tp_dist if dirn == 'long' else entry_p - tp_dist
            tp_target.append(
                _fmt_price(tp_t) +
                (' <span style="font-size:10px;color:#888780">'
                 '(entry + 6× stop dist)</span>' if has_pos else '')
            )
            if not math.isnan(close_v) and close_v > 0:
                tp_pct = (tp_t - close_v) / close_v * 100 if dirn == 'long' \
                         else (close_v - tp_t) / close_v * 100
                sign = '+' if tp_pct >= 0 else ''
                tp_dist_close.append(f'{sign}{tp_pct:.2f}%')
            else:
                tp_dist_close.append('—')
            tp_rr.append('6 : 1')

        # Regime breakdown — collapse each check to a single ratio.
        # Pass = ratio > 1.0 (green); fail = ratio ≤ 1.0 (red).
        def _ratio_cell(ratio):
            if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)):
                return '<span style="color:#888780">—</span>'
            cls = 'stat-pos' if ratio > 1.0 else 'stat-neg'
            return f'<span class="{cls}" style="font-weight:600">{ratio:.2f}</span>'

        # close / 1H trend MA
        if not math.isnan(trend_ma) and trend_ma != 0:
            reg_close_vs_ma.append(_ratio_cell(close_v / trend_ma))
        else:
            reg_close_vs_ma.append(_ratio_cell(None))

        # 4H ADX / adx_strong threshold
        if adx_strong_t > 0:
            reg_adx.append(_ratio_cell(adx_v / adx_strong_t))
        else:
            reg_adx.append(_ratio_cell(None))

        # +DI / −DI
        if minus_di > 0:
            reg_di.append(_ratio_cell(plus_di / minus_di))
        else:
            reg_di.append(_ratio_cell(None))

    rows_html_exit = (
        _divider('Inputs') +
        _row('Close ($)',           inputs_close)      +
        _row('1H ATR ($)',          inputs_atr)        +
        _row('trail_atr_mult',      inputs_trail_mult) +
        _row('1H trend MA ($)',     inputs_trend_ma)   +

        _divider('Stop loss') +
        _row('Stop distance ($)',   stop_trail_dist)   +
        _row('Current stop ($)',    stop_current)      +
        _row('% from close',        stop_dist_close)   +
        _row('Ratcheted this bar',  stop_ratcheted)    +

        _divider('Take profit') +
        _row('Regime',              tp_regime)         +
        _row('TP target ($)',       tp_target)         +
        _row('% from close',        tp_dist_close)     +
        _row('Reward : risk',       tp_rr)             +

        _divider('Regime checks (all ratios must be > 1.00 for STRONG BULL)') +
        _row('Close / 1H trend MA', reg_close_vs_ma)   +
        _row('4H ADX / adx_strong', reg_adx)           +
        _row('+DI / −DI',           reg_di)
    )

    coin_cols = ''.join(f'<col style="width:{col_w}">' for _ in coin_rows)
    colgroup  = f'<col style="width:180px">{coin_cols}'

    st.markdown(f"""
<div class="dashboard-card">
  <div class="table-scroll">
  <table class="dash-table" style="table-layout:fixed;width:100%">
    <colgroup>{colgroup}</colgroup>
    <thead><tr>
      <th style="text-align:left">Field</th>{coin_ths}
    </tr></thead>
    <tbody>{rows_html_exit}</tbody>
  </table>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Section 4: Parameters ─────────────────────────────────────────────────────

all_param_keys = sorted({k for c in coin_rows for k in c['all_params']})
n     = len(coin_rows)
col_w = f'calc((100% - 180px) / {n})'

optim_dates = '  ·  '.join(
    f"{escape(c['symbol'])}: {escape(c['optimised_on'])}" for c in coin_rows
)

coin_ths = ''.join(
    f'<th style="text-align:right;padding-right:24px">{escape(c["symbol"])}'
    f'<br><span style="font-size:10px;font-weight:400;text-transform:none;'
    f'letter-spacing:0;color:#888780">{escape(c["optimised_on"])}</span></th>'
    for c in coin_rows
)

rows_html_params = ''
for key in all_param_keys:
    tds = ''
    for c in coin_rows:
        val      = c['all_params'].get(key)
        is_fixed = key in c['fixed_keys']
        f_badge  = ' <span class="badge-fixed">F</span>' if is_fixed else ''
        tds     += f'<td style="text-align:right;padding-right:24px">{fmt(val)}{f_badge}</td>'
    rows_html_params += f'<tr><td class="field-label">{escape(key)}</td>{tds}</tr>'

coin_cols = ''.join(f'<col style="width:{col_w}">' for _ in coin_rows)
colgroup  = f'<col style="width:180px">{coin_cols}'

with st.expander("PARAMETERS"):
    st.caption(f"Last optimised — {optim_dates}")
    st.markdown(f"""
<div class="dashboard-card">
  <div class="table-scroll">
  <table class="dash-table" style="table-layout:fixed;width:100%">
    <colgroup>{colgroup}</colgroup>
    <thead><tr>
      <th style="text-align:left">Param</th>{coin_ths}
    </tr></thead>
    <tbody>{rows_html_params}</tbody>
  </table>
  </div>
  <div style="padding:6px 16px 8px;font-size:11px;color:#888780;border-top:1px solid #e4e4e1">
    <span class="badge-fixed">F</span> &nbsp;fixed by stability analysis
  </div>
</div>
""", unsafe_allow_html=True)
