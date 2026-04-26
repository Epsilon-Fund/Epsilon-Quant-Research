# Standalone entry point — primary app is live_trading/app.py
"""
Epsilon Fund — BB Breakout Dashboard (Streamlit)

Imports all computation from dashboard.py — no logic is duplicated here.
dashboard.py is the sole computation layer. All trade logging is done via
direct file writes in this module — no external server required.

Run:
    streamlit run live_trading/dashboards/bbbreakout/streamlit_app.py
"""

import json
import re
import sys
import os
from datetime import datetime, timezone
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
                          coin_capital=None, size_usd=None):
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
    if st.button("↻ Refresh data", key="bb_refresh"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data auto-refreshes every minute.")


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner="Fetching BB Breakout market data…")
def load_all():
    live_params = load_live_params(DATA_DIR)
    if not live_params:
        return [], None

    active_set   = set(ACTIVE_ASSETS)
    coin_symbols = [sym for sym in live_params if sym in active_set]

    result = run_dashboard(coin_symbols, live_params, positions={})

    coin_rows   = result['assets']
    signal_date = result['signal_date']

    for row in coin_rows:
        row['fixed_keys'] = _ASSET_FIXED_KEYS.get(row['symbol'], row['fixed_keys'])

    return coin_rows, signal_date


def load_live_prices(symbols: tuple):
    """
    Look up current prices for the requested symbols.

    Backed by shared.binance_utils.fetch_all_live_prices() — one batched
    REST call shared across all dashboards, cached 120 s.
    """
    return _shared_live_prices(symbols)


coin_rows, signal_date = load_all()

if not coin_rows:
    st.error("live_params.json is empty — run `optimise.py` first.")
    st.stop()

# Apply decisions OUTSIDE the cache so positions.json is always fresh.
# Single fresh read used by decisions, header portfolio summary, and trade forms.
_positions_now           = load_positions(DATA_DIR)
_positions_for_decisions = _positions_now
for _c in coin_rows:
    _open_pos = get_open_positions(_c['symbol'], _positions_for_decisions)
    _c['sig'].update(
        apply_decision(_c['sig'], _open_pos, _c['exec_price'], _c['coin_capital'])
    )

generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt(v):
    if v is None: return '—'
    if isinstance(v, int): return str(v)
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f'{v:.4f}'.rstrip('0').rstrip('.')
    return escape(str(v))


def _fmt_price(v):
    """Adaptive price formatter — matches decimal places to price magnitude."""
    import math as _m
    if v is None or (isinstance(v, float) and _m.isnan(v)):
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


# ── Active Positions ──────────────────────────────────────────────────────────

st.markdown("#### ACTIVE POSITIONS")
if not _open:
    st.markdown('<p style="font-size:12px;color:#888780;margin-bottom:14px">No open positions</p>',
                unsafe_allow_html=True)
else:
    from datetime import date as _date
    _pos_rows       = ''
    _tot_size_usd_r = 0.0
    _tot_pnl_usd_r  = 0.0
    _tot_pos_val_r  = 0.0
    _tot_has_live_r = False
    _confirm_items  = []

    for pid, pos in _open.items():
        _sym_for_pos = pos.get('symbol', pid)
        live_price   = _live_prices.get(_sym_for_pos)
        entry_price  = pos['entry_price']
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

        # Direction badge
        _dir_badge = (
            '<span class="badge badge-LONG">LONG</span>' if pos_dir == 'long'
            else '<span class="badge badge-SHORT">SHORT</span>'
        )

        # Two-state stop
        conf_stop    = pos.get('current_stop')
        pending_stop = pos.get('pending_stop')

        # Auto-ratchet: suggest strategy's trailing stop as new pending
        coin_sig  = _coin_sig_by_sym.get(_sym_for_pos, {})
        sugg_stop = coin_sig.get('current_stop') if coin_sig.get('decision') == 'HOLD' else None
        if sugg_stop is not None and sugg_stop != pending_stop:
            # Direction-aware improvement check
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

        if conf_stop is not None:
            if _needs_confirm:
                stop_td = (f'<td>{_fmt_price(conf_stop)}'
                           f'<br><span class="stop-up">→ {_fmt_price(pending_stop)}</span></td>')
            else:
                stop_td = f'<td>{_fmt_price(conf_stop)}</td>'
        elif pending_stop is not None:
            stop_td = (f'<td><span style="color:#888780">{_fmt_price(pending_stop)}</span>'
                       f'<br><span style="font-size:10px;color:#888780">unconfirmed</span></td>')
        else:
            stop_td = '<td style="color:#888780">—</td>'

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
          <td>Total</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
          <td>{_tot_size_usd_r:,.0f}</td>
          {_tot_pnl_pct_td}{_tot_pnl_usd_td}{_tot_pos_td}
        </tr>"""

    st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Symbol</th><th>Dir</th><th>Entry date</th><th>Days</th>
      <th>Entry ($)</th><th>Live ($)</th><th>Stop ($)</th>
      <th>Leverage</th><th>Size ($)</th>
      <th>P&amp;L (%)</th><th>P&amp;L ($)</th><th>Position ($)</th>
    </tr></thead>
    <tbody>{_pos_rows}{_total_row}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

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

rows_html = ''
for c in coin_rows:
    sig = c['sig']
    d   = sig['decision']

    alloc_note = f"{c['coin_capital']:,.0f} alloc ({c['coin_weight']*100:.0f}% of ${_realised_capital:,.0f})"
    badge      = f'<span class="badge badge-{d}">{d}</span>'

    lev = sig.get('leverage_multiplier')
    size_pct = f"{lev:.2f}x" if lev is not None else '—'
    if d == 'HOLD' and lev is not None:
        size_pct += '<br><span style="font-size:11px;color:#888780">held</span>'
    elif d == 'EXIT' and lev is not None:
        size_pct += '<br><span style="font-size:11px;color:#888780">to exit</span>'
    size_usd = f"{sig['size_usd']:,.0f}" if sig.get('size_usd') is not None else '—'

    _stop_val = sig.get('current_stop')
    if _stop_val is not None and _stop_val > 0:
        stop_html = _fmt_price(_stop_val)
        if d == 'HOLD' and sig.get('stop_updated'):
            stop_html += f' <span class="stop-up">↑</span><br><span class="stop-prev">was {_fmt_price(sig["old_stop"])}</span>'
    else:
        stop_html = '—'

    # Direction badge for ENTRY/HOLD
    _dir_int = sig.get('direction', 0)
    if _dir_int == 1:
        dir_html = '<span class="badge badge-LONG">LONG</span>'
    elif _dir_int == -1:
        dir_html = '<span class="badge badge-SHORT">SHORT</span>'
    else:
        dir_html = '—'

    _lp       = _live_prices.get(c['symbol'])
    live_html = _fmt_price(_lp) if _lp is not None else '<span style="color:#888780">—</span>'

    row_cls = f'class="row-{d}"' if d != 'FLAT' else ''

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
      <td class="r">{_fmt_price(sig['close'])}</td>
      <td class="r">{live_html}</td>
    </tr>"""

st.markdown("#### DECISIONS")
st.caption(f"Sizes based on realised capital: ${_realised_capital:,.2f}")
st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Asset</th><th>Decision</th><th>Direction</th>
      <th class="r">Size (leverage)</th><th class="r">Size ($)</th>
      <th class="r">Stop ($)</th><th class="r">Last 1H Close ($)</th>
      <th class="r">Live ($)</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
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
                    value=float(round(_default_size, 2)),
                    min_value=0.0,
                    format='%.2f',
                    key=f'bb_size_{_sym}',
                )
                _price = st.number_input(
                    'Actual price',
                    value=float(round(_default_price, 6)),
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
                    value=float(round(_default_price, 6)),
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
            _theo_stop     = _sig.get('stop', 0.0)

            if _is_exit_final:
                _exit_amount   = _exit_lev if _exit_lev is not None else _held_lev
                _is_full_exit  = (_exit_amount >= _held_lev * 0.999)
                _exit_type_log = 'full' if _is_full_exit else 'partial'
                _pid_to_exit   = st.session_state.get(f'bb_exit_pid_{_sym}', _selected_pid)
                if _pid_to_exit is None and _open_for_sym:
                    _pid_to_exit = next(iter(_open_for_sym))
                _sub_sig = ('EXIT', round(_price, 6), round(_exit_amount, 4), _pid_to_exit)
            else:
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
                    print(f"Capital updated: ${_old_rc:.2f} → ${_new_rc:.2f} "
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
                    _write_position_entry(
                        _new_pid, _sym, _price, _size, _theo_stop,
                        _strategy, _disc, direction=_dir_val,
                        coin_capital=_snap_coin_cap,
                        size_usd=_snap_size_usd,
                    )
            invalidate_trade_caches()
            st.rerun()


# ── Section 2: Entry Conditions ───────────────────────────────────────────────

rows_html = ''
for c in coin_rows:
    sig = c['sig']

    close     = sig['close']
    bb_upper  = sig.get('bb_upper', float('nan'))
    bb_mid    = sig.get('bb_mid',   float('nan'))
    bb_lower  = sig.get('bb_lower', float('nan'))
    bb_width  = sig.get('bb_width', float('nan'))
    sma       = sig.get('sma',      float('nan'))

    import math
    def _fv(v, dp=4):
        return f'{v:,.{dp}f}' if not math.isnan(v) else '—'

    # Price position relative to BB bands
    if not math.isnan(bb_upper) and not math.isnan(bb_lower) and not math.isnan(close):
        if close >= bb_upper:
            band_pos_td = '<td class="c bb-band-above">above upper</td>'
        elif close <= bb_lower:
            band_pos_td = '<td class="c bb-band-below">below lower</td>'
        else:
            _pct = (close - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
            band_pos_td = f'<td class="c bb-band-mid">{_pct:.0f}% in band</td>'
    else:
        band_pos_td = '<td class="c">—</td>'

    # SMA proximity
    if not math.isnan(sma) and sma > 0:
        sma_bps = abs(close - sma) / sma * 10000
        sma_td  = f'<td class="r">{_fv(sma)} <span style="font-size:10px;color:#888780">({sma_bps:.0f} bps)</span></td>'
    else:
        sma_td = f'<td class="r">{_fv(sma)}</td>'

    entry_cls_l = 'entry-t' if sig.get('entry_long')  else 'entry-f'
    entry_cls_s = 'entry-t' if sig.get('entry_short') else 'entry-f'

    _pos_val = sig.get('position', 0)
    if _pos_val == 1:
        pos_td = '<td class="c entry-t">LONG</td>'
    elif _pos_val == -1:
        pos_td = '<td class="c entry-f">SHORT</td>'
    else:
        pos_td = '<td class="c" style="color:#888780">FLAT</td>'

    rows_html += f"""
    <tr>
      <td class="asset-name">{escape(c['symbol'])}</td>
      <td class="r">{_fv(bb_upper)}</td>
      <td class="r">{_fv(bb_mid)}</td>
      <td class="r">{_fv(bb_lower)}</td>
      <td class="r">{_fv(bb_width, 4)}</td>
      {band_pos_td}
      {sma_td}
      <td class="r">{_fv(close)}</td>
      <td class="c {entry_cls_l}">{"TRUE" if sig.get("entry_long") else "FALSE"}</td>
      <td class="c {entry_cls_s}">{"TRUE" if sig.get("entry_short") else "FALSE"}</td>
      {pos_td}
    </tr>"""

col_title, col_help = st.columns([1, 6])
with col_title:
    st.markdown("#### ENTRY CONDITIONS")
with col_help:
    with st.expander("Explanation"):
        st.markdown("""
**BB Bands** — Computed on 1H close prices over `bb_period` bars.

**Band position** — Where the close sits relative to the bands:
- *above upper*: price has broken above upper band
- *below lower*: price has broken below lower band
- *n% in band*: how far price sits between lower and upper

**SMA** — 1H moving average used as the pullback entry anchor.
A signal fires when price pulls back within `entry_zone_bps` of this SMA
after a two-candle 4H breakout, and momentum resumes in the breakout direction.

**Entry Long / Entry Short** — True when the strategy state machine entered on the last bar.

**Position** — Current state: LONG / SHORT / FLAT (from the 1H state machine).
""")

st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table dash-table-labeled">
    <thead><tr>
      <th>Asset</th>
      <th class="r">BB Upper ($)</th><th class="r">BB Mid ($)</th><th class="r">BB Lower ($)</th>
      <th class="r">BB Width</th><th class="c">Band position</th>
      <th class="r">SMA ($)</th><th class="r">Close ($)</th>
      <th class="c">Entry Long</th><th class="c">Entry Short</th>
      <th class="c">Position</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Section 3: Stop details ───────────────────────────────────────────────────

with st.expander("STOP LOSS DETAILS"):
    with st.expander("Explanation"):
        st.markdown("""
**Trailing ratchet stop** — Initialised at `trail_atr_mult × ATR_1H` from entry price.
Ratchets in favour only (up for longs, down for shorts) on every bar.
In strong-bull regime (price > trend MA, ADX strong, +DI > -DI): no TP — trailing stop only.
Otherwise: 6:1 fixed TP from entry.
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

    import math as _math

    def _stop_cell(c):
        s = c['sig'].get('stop', 0.0)
        return _fmt_price(s) if s and not _math.isnan(s) else '—'

    def _trail_mult_cell(c):
        v = c['all_params'].get('trail_atr_mult')
        return fmt(v)

    upd_yes = '<span class="upd-yes">↑ Yes</span>'
    upd_no  = '<span class="upd-no">No</span>'
    upd_tds = ''.join(
        f'<td style="text-align:right;padding-right:24px">'
        f'{upd_yes if c["sig"].get("stop_updated") else upd_no}</td>'
        for c in coin_rows
    )

    rows_html_stop = (
        _divider('Trailing stop') +
        _row('trail_atr_mult',   [_trail_mult_cell(c) for c in coin_rows]) +
        _row('Current stop ($)', [_stop_cell(c)       for c in coin_rows]) +
        f'<tr><td class="field-label">Ratcheted this bar</td>{upd_tds}</tr>'
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
    <tbody>{rows_html_stop}</tbody>
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
