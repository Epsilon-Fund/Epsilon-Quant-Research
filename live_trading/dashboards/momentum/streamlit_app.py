# Standalone entry point — primary app is live_trading/app.py
"""
Epsilon Fund — Momentum Dashboard (Streamlit)

Imports all computation from dashboard.py — no logic is duplicated here.
dashboard.py is the sole computation layer. All trade logging is done via
direct file writes in this module — no external Flask server required.

Run:
    streamlit run live_trading/dashboards/momentum/streamlit_app.py
"""

import json
import re
import sys
import os
import webbrowser
from datetime import datetime, timezone
from html import escape

# Open the browser once when the process starts.
# The env-var guard ensures this runs exactly once — Streamlit reruns the script
# on every interaction, but the process (and its env) persists across reruns.

# ── Path setup ────────────────────────────────────────────────────────────────
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))          # dashboards/momentum/
_LT_DIR        = os.path.abspath(os.path.join(_DASHBOARD_DIR, '..', '..'))  # live_trading/
_ROOT          = os.path.dirname(_LT_DIR)                             # repo root
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
from config   import ACTIVE_ASSETS, EXECUTION_HOUR, CAPITAL, COIN_WEIGHTS
import config as _cfg_mod
TRADING_COST_PCT = getattr(_cfg_mod, 'TRADING_COST_PCT', 0.0)
from optimise import ASSET_CONFIG

_PID_RE = re.compile(r'^\w+_\d{8}_\d{3}$')


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
    Prints a summary of what was changed.
    """
    positions = _load_json(POSITIONS_PATH, {})
    old_keys  = [k for k in positions if not _PID_RE.match(k)]
    if not old_keys:
        return   # nothing to migrate

    migrated = {}
    log      = []
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
        log.append((key, new_key))

    _save_json(POSITIONS_PATH, migrated)
    print(f"[positions migration] migrated {len(log)} entr{'y' if len(log)==1 else 'ies'}:")
    for old, new in log:
        print(f"  {old}  ->  {new}")


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
        # ── Capital snapshot — frozen at submission time ───────────────────
        # These values are never recomputed from config for historical records.
        entry['coin_capital']  = round(coin_capital, 4)  if coin_capital  is not None else None
        entry['size_usd']      = round(size_usd, 4)      if size_usd      is not None else None
        entry['capital_total'] = capital_total
        entry['coin_weight']   = coin_weight
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
        'pending_stop':        theoretical_stop,  # proposed — not yet binding
        'current_stop':        None,              # null until manually confirmed
        'direction':           direction,
        'partial_exits':       0,
        'exit_price':          None,
        'exit_date':           None,
        'discretion_note':     discretion_note,
        # ── Capital snapshot — frozen at entry, used for all P&L calcs ────
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
    """Reduce held leverage by exit_leverage; keep in_position = True."""
    positions = _load_json(POSITIONS_PATH, {})
    if position_id in positions:
        pos         = positions[position_id]
        current_lev = pos.get('leverage_multiplier') or pos.get('size_pct', 0)
        remaining   = round(max(current_lev - exit_leverage, 0), 4)
        pos['leverage_multiplier'] = remaining
        pos['partial_exits']       = pos.get('partial_exits', 0) + 1
        pos['discretion_note']     = discretion_note
        # Remove legacy size_pct so only leverage_multiplier is used going forward
        pos.pop('size_pct', None)
        # Keep size_usd consistent with the new leverage using the frozen coin_capital
        coin_cap = pos.get('coin_capital')
        if coin_cap is not None:
            pos['size_usd'] = round(coin_cap * remaining, 4)
    _save_json(POSITIONS_PATH, positions)


def _write_pending_stop(position_id, new_stop):
    """Propose a new stop. Does not bind until _confirm_stop() is called."""
    positions = _load_json(POSITIONS_PATH, {})
    if position_id in positions:
        positions[position_id]['pending_stop'] = new_stop
    _save_json(POSITIONS_PATH, positions)


def _confirm_stop(position_id):
    """Promote pending_stop -> current_stop, making it binding for EXIT decisions."""
    positions = _load_json(POSITIONS_PATH, {})
    if position_id in positions:
        pending = positions[position_id].get('pending_stop')
        if pending is not None:
            positions[position_id]['current_stop'] = pending
    _save_json(POSITIONS_PATH, positions)


# Run once at startup — converts old symbol-keyed entries to FIFO position_id format.
_migrate_positions()

# st.popover() requires Streamlit ≥ 1.31. Fall back to st.expander() on older versions.
_hint_popover = getattr(st, 'popover', None) or st.expander

# Build per-coin fixed-param key sets from ASSET_CONFIG (authoritative source).
_ASSET_FIXED_KEYS = {a['symbol']: set(a['fixed_params'].keys()) for a in ASSET_CONFIG}

# Startup diagnostics — confirm per-coin fixed params are loading correctly.
for _sym in ('BTCUSDT', 'ETHUSDT'):
    if _sym in _ASSET_FIXED_KEYS:
        print(f"{_sym} fixed params: {sorted(_ASSET_FIXED_KEYS[_sym])}")


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Epsilon Fund — Live Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Base font */
  html, body, [class*="css"] { font-size: 13px !important; }

  /* Page background */
  .stApp            { background-color: #f8f8f7 !important; }
  .block-container  { background-color: #f8f8f7 !important; padding-top: 5rem; padding-bottom: 2rem; }

  /* Expander header typography — compact hint style */
  .streamlit-expanderHeader {
      font-size: 11px !important;
      font-weight: 400 !important;
      letter-spacing: 0 !important;
      text-transform: none !important;
      color: #888780 !important;
  }

  /* Card wrapper */
  .dashboard-card {
      background: white;
      border: 1px solid #d3d1c7;
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      padding: 0;
      margin-bottom: 14px;
      overflow: hidden;
  }
  .card-header {
      padding: 10px 16px 8px;
      border-bottom: 1px solid #d3d1c7;
      display: flex; align-items: center; gap: 12px;
  }
  .section-label {
      font-size: 11px !important; font-weight: 500;
      letter-spacing: 0.06em; text-transform: uppercase;
      color: #444441; margin: 0;
  }
  .section-note { font-size: 11px; color: #888780; }

  /* Base table */
  table { font-size: 13px !important; }
  .dash-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  .dash-table th {
      font-size: 11px !important; font-weight: 500; letter-spacing: 0.05em;
      text-transform: uppercase; color: #444441;
      padding: 7px 12px 5px; text-align: center !important;
      border-bottom: 1px solid #d3d1c7; white-space: nowrap;
  }
  .dash-table th.r { text-align: center !important; }
  .dash-table th.c { text-align: center !important; }
  .dash-table td {
      padding: 8px 12px; border-bottom: 1px solid #e4e4e1;
      vertical-align: middle; text-align: center !important;
  }
  .dash-table tr:last-child td { border-bottom: none; }
  .dash-table td.r { text-align: center !important; }
  .dash-table td.c { text-align: center !important; }
  .dash-table td[style] { text-align: center !important; }
  .dash-table th[style] { text-align: center !important; }

  /* Remove vertical column lines from all tables */
  .dash-table td, .dash-table th {
      border-left: none !important;
      border-right: none !important;
  }
  /* Restore field-label right border inside expanders only */
  [data-testid="stExpander"] .dash-table td.field-label {
      border-right: 1px solid #d3d1c7 !important;
  }

  /* Grey shading on first column for entry conditions and caution flags tables */
  .dash-table-labeled td:first-child {
      background: #fafaf9 !important;
      font-weight: 500;
      border-right: 1px solid #d3d1c7 !important;
  }

  /* Asset cell */
  .asset-name  { font-weight: 500; font-size: 12px; }
  .asset-alloc { font-size: 11px; color: #888780; }

  /* Decision row tints */
  .row-ENTRY td { background: #f0f8e8 !important; }
  .row-EXIT  td { background: #fdeaea !important; }

  /* Decision badges — match Jinja2 exactly */
  .badge {
      display: inline-block; padding: 2px 10px; border-radius: 4px;
      font-size: 11px; font-weight: 500; white-space: nowrap;
  }
  .badge-ENTRY { background: #EAF3DE; color: #3B6D11; }
  .badge-HOLD  { background: #FAEEDA; color: #854F0B; }
  .badge-EXIT  { background: #FCEBEB; color: #A32D2D; }
  .badge-FLAT  { background: #F1EFE8; color: #5F5E5A; }

  /* Bool text — used in condition columns (white cell background) */
  .t { color: #1a5c2a; }
  .f { color: #8a1a1a; }

  /* Caution Long TRUE cell — light amber only, all other condition cells white */
  .caution { background: #FAEEDA; color: #854F0B; font-weight: 500; }

  /* Entry Long result cell — light shades matching decision badges */
  .entry-t { background: #EAF3DE; color: #3B6D11; font-weight: 600; }
  .entry-f { background: #FCEBEB; color: #A32D2D; font-weight: 600; }

  /* Stop ratchet */
  .stop-up   { color: #1a5c2a; font-size: 11px; font-weight: 600; }
  .stop-prev { color: #888780; font-size: 11px; }

  /* Transposed table label column */
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
      color: #888780; padding: 4px 12px;
      border-bottom: 1px solid #d3d1c7;
  }

  /* Fixed param "F" badge (inline in value cell) */
  .badge-fixed {
      background: #e4e4e1; color: #5F5E5A; font-size: 9px;
      padding: 1px 4px; border-radius: 3px; margin-left: 5px;
      display: inline-block; vertical-align: middle;
  }

  /* Path badges — match decision badge palette */
  .badge-ent_normal  { background: #EAF3DE; color: #3B6D11; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-ent_caution { background: #FCEBEB; color: #A32D2D; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-ent_both    { background: #FAEEDA; color: #854F0B; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-pos_normal  { background: #EAF3DE; color: #3B6D11; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-pos_caution { background: #FCEBEB; color: #A32D2D; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }

  /* Updated cell */
  .upd-yes { color: #1a5c2a; font-weight: 600; }
  .upd-no  { color: #888780; }

  /* Page header meta */
  .dash-meta {
      font-size: 12px; color: #888780;
      margin-bottom: 18px; line-height: 1.8;
  }
  .dash-meta strong { color: #444441; font-weight: 500; }

  /* Scroll wrapper for wide transposed tables */
  .table-scroll { overflow-x: auto; }

  /* Total row in active positions */
  .row-total td {
      background: #f0efea !important;
      font-weight: 600;
      border-top: 2px solid #d3d1c7 !important;
  }

  /* Trade log form card */
  .form-card {
      background: white; border: 1px solid #d3d1c7; border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06); padding: 14px 16px 10px;
      margin-bottom: 14px;
  }
  .form-card-header {
      font-size: 12px; font-weight: 600; color: #444441;
      margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: refresh ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Controls")
    if st.button("↻ Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data auto-refreshes every 5 minutes.")


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Fetching market data…")
def load_all():
    """
    Cache only the expensive Binance API calls and signal computation.
    Positions and decision logic are intentionally excluded — they must
    recompute fresh every render so changes to positions.json take effect
    immediately without waiting for the 5-minute cache expiry.
    """
    live_params = load_live_params(DATA_DIR)
    if not live_params:
        return [], None

    active_set   = set(ACTIVE_ASSETS)
    coin_symbols = [sym for sym in live_params if sym in active_set]

    # run_dashboard() does all the expensive Binance calls + signal computation.
    # apply_decision() is NOT called inside — it stays outside the cache below.
    result = run_dashboard(coin_symbols, live_params, positions={})

    coin_rows   = result['assets']
    signal_date = result['signal_date']

    # Use ASSET_CONFIG as the authoritative source for fixed params;
    # run_dashboard() uses live_params.json's fixed_param_keys as a fallback,
    # but ASSET_CONFIG takes precedence here.
    for row in coin_rows:
        row['fixed_keys'] = _ASSET_FIXED_KEYS.get(row['symbol'], row['fixed_keys'])

    return coin_rows, signal_date


def load_live_prices(symbols: tuple):
    """
    Look up current market prices for the requested symbols.

    Backed by shared.binance_utils.fetch_all_live_prices() — a single
    batched REST call shared across all dashboards, cached 120 s.
    """
    return _shared_live_prices(symbols)


coin_rows, signal_date = load_all()

if not coin_rows:
    st.error("live_params.json is empty — run `optimise.py` first.")
    st.stop()

# Apply decisions OUTSIDE the cache so positions.json is read fresh every render.
# st.cache_data returns deserialized copies, so mutating coin_rows here is safe.
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

generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


# ── Helper: format param value ────────────────────────────────────────────────

def fmt(v):
    if v is None: return '—'
    if isinstance(v, int): return str(v)
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f'{v:.2f}'
    return escape(str(v))


# ── Load positions + live prices early (needed for header portfolio summary) ──
# _positions_now was already read once near the top of this render.
_open            = {pid: p for pid, p in _positions_now.items() if p.get('in_position')}
_coin_sig_by_sym = {c['symbol']: c['sig'] for c in coin_rows}

# Fetch live prices for all active coins (positions + decisions)
_all_syms    = tuple(sorted({c['symbol'] for c in coin_rows} | {p.get('symbol', pid) for pid, p in _open.items()}))
_live_prices = load_live_prices(_all_syms) if _all_syms else {}

# Pre-compute portfolio totals for the header summary card
# Realized P&L: sum actual_pnl_usd from all closed trade pairs in the journal
_trade_pairs      = build_trade_pairs(DATA_DIR)
_realized_pnl_usd = sum(p['actual_pnl_usd'] for p in _trade_pairs.get('closed', []))

# Realised capital — updated after every EXIT; drives all sizing going forward
_realised_capital = load_realised_capital(DATA_DIR)
_capital_delta    = _realised_capital - CAPITAL

# Unrealized P&L: open positions marked to live prices
_total_size_usd    = 0.0
_unrealized_pnl_usd = 0.0
_total_pos_value   = 0.0
_total_has_live    = False
for _pid0, _pos0 in _open.items():
    _sym0 = _pos0.get('symbol', _pid0)
    _lp0  = _live_prices.get(_sym0)
    _ep0  = _pos0.get('entry_price', 0)
    _lev0 = _pos0.get('leverage_multiplier') or _pos0.get('size_pct', 0)
    # Use frozen size_usd from positions.json; fall back to live config for legacy entries
    _sz0  = _pos0.get('size_usd') or (get_coin_capital(_sym0) * _lev0)
    _total_size_usd += _sz0
    if _lp0 and _ep0:
        # Deduct round-trip cost (entry leg already paid + estimated exit leg)
        # so unrealized P&L is consistent with how closed trades report P&L.
        _cost0               = _sz0 * TRADING_COST_PCT * 2
        _pnl0                = (_lp0 - _ep0) / _ep0 * _sz0 - _cost0
        _unrealized_pnl_usd += _pnl0
        _total_pos_value    += _sz0 + _pnl0
        _total_has_live      = True
    else:
        _total_pos_value += _sz0

_total_pnl_usd = _realized_pnl_usd + _unrealized_pnl_usd

# ── Page header ───────────────────────────────────────────────────────────────

_optim_dates = [c['optimised_on'] for c in coin_rows if c.get('optimised_on')]
_last_optim  = max(_optim_dates) if _optim_dates else None
_days_since_optim = (
    (datetime.today().date() - datetime.strptime(_last_optim, '%Y-%m-%d').date()).days
    if _last_optim else None
)
_optim_note = (f"{_days_since_optim}d ago" if _days_since_optim is not None else "—")

if not globals().get('_SUPPRESS_H1', False):
    st.markdown("""
<h1 style="font-size:35px;font-weight:700;letter-spacing:-0.01em;margin-bottom:10px">
  Epsilon Fund — Live Trading Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="dash-meta">
  <strong>Signal date:</strong> {signal_date} &nbsp;&nbsp;
  <strong>Generated:</strong> {generated_at} UTC &nbsp;&nbsp;
  <strong>Execution hour:</strong> {EXECUTION_HOUR}h UTC (T+1)
</div>
""", unsafe_allow_html=True)

# ── Portfolio summary ─────────────────────────────────────────────────────────

# Portfolio value: always show realized component; add unrealized only when live prices available
_port_val      = CAPITAL + _realized_pnl_usd + (_unrealized_pnl_usd if _total_has_live else 0)
_port_val_note = '' if _total_has_live else '<span style="font-size:10px;color:#888780"> (excl. open)</span>'
_port_val_str  = f"${_port_val:,.0f}{_port_val_note}"
# P&L: always show (realized is always available; unrealized requires live prices)
_pnl_pct    = _total_pnl_usd / CAPITAL * 100 if CAPITAL else 0.0
_pnl_cls    = 'entry-t' if _pnl_pct >= 0 else 'entry-f'
_pnl_sign   = '+' if _pnl_pct >= 0 else ''
_pnl_str    = f"{_pnl_sign}{_pnl_pct:.2f}%"
_pnl_td_cls = f' class="{_pnl_cls}"'

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
      <td{_pnl_td_cls}>{_pnl_str}</td>
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
    _pos_rows        = ''
    # Totals already computed above; re-accumulate per-row P&L for the total row display
    _tot_size_usd_r  = 0.0
    _tot_pnl_usd_r   = 0.0
    _tot_pos_val_r   = 0.0
    _tot_has_live_r  = False

    _confirm_items = []   # [(pid, pending_stop)] needing a confirm button

    for pid, pos in _open.items():
        _sym_for_pos = pos.get('symbol', pid)
        live_price   = _live_prices.get(_sym_for_pos)
        entry_price  = pos['entry_price']

        leverage_multiplier = (pos.get('leverage_multiplier') or pos.get('size_pct', 0))
        # Use frozen size_usd from positions.json; fall back to live config for legacy entries
        size_usd = pos.get('size_usd') or (get_coin_capital(_sym_for_pos) * leverage_multiplier)

        _tot_size_usd_r += size_usd
        if live_price and entry_price:
            # Gross price move (for context — how much has the price moved)
            _gross_pnl         = (live_price - entry_price) / entry_price * size_usd
            # Net P&L: deduct round-trip cost (entry leg paid + estimated exit leg)
            # Consistent with how closed trades report P&L in the journal.
            _trade_cost        = size_usd * TRADING_COST_PCT * 2
            unrealised_pnl_usd = _gross_pnl - _trade_cost
            net_return_pct     = unrealised_pnl_usd / size_usd * 100
            position_value     = size_usd + unrealised_pnl_usd
            _tot_pnl_usd_r    += unrealised_pnl_usd
            _tot_pos_val_r    += position_value
            _tot_has_live_r    = True
            pnl_cls    = 'entry-t' if unrealised_pnl_usd >= 0 else 'entry-f'
            pnl_sign   = '+' if net_return_pct >= 0 else ''
            live_td    = f'<td>{live_price:,.2f}</td>'
            pnl_pct_td = f'<td class="{pnl_cls}">{pnl_sign}{net_return_pct:.2f}%</td>'
            pnl_usd_td = f'<td class="{pnl_cls}">{pnl_sign}{unrealised_pnl_usd:,.0f}</td>'
            pos_val_td = f'<td>{position_value:,.0f}</td>'
        else:
            _tot_pos_val_r += size_usd
            live_td    = '<td>—</td>'
            pnl_pct_td = '<td>—</td>'
            pnl_usd_td = '<td>—</td>'
            pos_val_td = '<td>—</td>'

        # ── Two-state stop ────────────────────────────────────────────────────
        conf_stop    = pos.get('current_stop')          # binding — used for EXIT
        pending_stop = pos.get('pending_stop')          # proposed — shown only

        # Auto-ratchet: update pending_stop if strategy suggests a higher stop.
        # Never touches current_stop — confirm button does that.
        coin_sig  = _coin_sig_by_sym.get(_sym_for_pos, {})
        sugg_stop = (coin_sig.get('current_stop')
                     if coin_sig.get('decision') == 'HOLD'
                     else coin_sig.get('theoretical_stop'))
        if sugg_stop is not None and sugg_stop > (conf_stop or 0) and sugg_stop != pending_stop:
            pending_stop = sugg_stop
            _write_pending_stop(pid, sugg_stop)

        # Determine whether this position needs a confirm button
        _needs_confirm = (
            pending_stop is not None
            and pending_stop != conf_stop
        )
        if _needs_confirm:
            _confirm_items.append((pid, _sym_for_pos, pending_stop))

        # Render stop cell
        if conf_stop is not None:
            if _needs_confirm:
                # Confirmed stop exists; ratchet proposes a higher one
                stop_td = (f'<td>{conf_stop:,.2f}'
                           f'<br><span class="stop-up">-> {pending_stop:,.2f}</span></td>')
            else:
                stop_td = f'<td>{conf_stop:,.2f}</td>'
        elif pending_stop is not None:
            # No confirmed stop yet — new position, day-of-entry
            stop_td = (f'<td><span style="color:#888780">{pending_stop:,.2f}</span>'
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
          <td>{escape(pos.get('entry_date','—'))}</td>
          <td>{days_held}</td>
          <td>{entry_price:,.2f}</td>
          {live_td}
          {stop_td}
          <td>{leverage_multiplier:.2f}x</td>
          <td>{size_usd:,.0f}</td>
          {pnl_pct_td}
          {pnl_usd_td}
          {pos_val_td}
        </tr>"""

    # ── Total row ─────────────────────────────────────────────────────────────
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
          <td>Total</td>
          <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
          <td>{_tot_size_usd_r:,.0f}</td>
          {_tot_pnl_pct_td}
          {_tot_pnl_usd_td}
          {_tot_pos_td}
        </tr>"""

    st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Symbol</th><th>Entry date</th><th>Days</th><th>Entry ($)</th>
      <th>Live ($)</th><th>Stop ($)</th>
      <th>Leverage</th><th>Size ($)</th>
      <th>P&amp;L (%)</th><th>P&amp;L ($)</th><th>Position ($)</th>
    </tr></thead>
    <tbody>{_pos_rows}{_total_row}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

    # ── Confirm stop buttons ──────────────────────────────────────────────────
    if _confirm_items:
        _conf_cols = st.columns(len(_confirm_items))
        for _col, (_pid, _sym, _pend) in zip(_conf_cols, _confirm_items):
            with _col:
                if st.button(f"✓ Confirm stop {_sym}: {_pend:,.2f}",
                             key=f"conf_stop_{_pid}"):
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
    size_usd = f"{sig['size_usd']:,.0f}" if sig['size_usd'] is not None else '—'

    if sig['current_stop'] is not None:
        stop_html = f"{sig['current_stop']:,.2f}"
        if d == 'HOLD' and sig.get('stop_updated'):
            stop_html += f' <span class="stop-up">↑</span><br><span class="stop-prev">was {sig["old_stop"]:,.2f}</span>'
    else:
        stop_html = '—'

    exec_html = f"{c['exec_price']:,.2f}" if c['exec_price'] is not None else '<span style="color:#888780">pending</span>'

    _lp = _live_prices.get(c['symbol'])
    live_html = f"{_lp:,.2f}" if _lp is not None else '<span style="color:#888780">—</span>'

    row_cls = f'class="row-{d}"' if d != 'FLAT' else ''

    rows_html += f"""
    <tr {row_cls}>
      <td>
        <div class="asset-name">{escape(c['symbol'])}</div>
        <div class="asset-alloc">{escape(alloc_note)}</div>
      </td>
      <td>{badge}</td>
      <td class="r">{size_pct}</td>
      <td class="r">{size_usd}</td>
      <td class="r">{stop_html}</td>
      <td class="r">{sig['close']:,.2f}</td>
      <td class="r">{exec_html}</td>
      <td class="r">{live_html}</td>
    </tr>"""

st.markdown("#### DECISIONS")
st.caption(f"Sizes based on realised capital: ${_realised_capital:,.2f}")
st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Asset</th><th>Decision</th>
      <th class="r">Size (leverage)</th><th class="r">Size ($)</th>
      <th class="r">Stop ($)</th><th class="r">Last Daily Close ($)</th>
      <th class="r">Today's {EXECUTION_HOUR}h ($)</th><th class="r">Live ($)</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Trade Log Forms ───────────────────────────────────────────────────────────

st.markdown("#### TRADE LOG")

_positions_for_forms = _positions_now   # reuse the single render-time read
_form_cols = st.columns(len(coin_rows))

for _fi, _c in enumerate(coin_rows):
    with _form_cols[_fi]:
        _sym      = _c['symbol']
        _sig      = _c['sig']
        _decision = _sig['decision']
        _strategy = _c['strategy']
        _pos      = _positions_for_forms.get(_sym, {})

        # Default price: exec_price if available, else close
        _default_price = _c['exec_price'] if _c['exec_price'] is not None else _sig['close']

        # Get all open positions for this symbol (new FIFO format)
        _open_for_sym = {
            pid: p for pid, p in _positions_for_forms.items()
            if p.get('symbol', pid) == _sym and p.get('in_position')
        }
        # Primary position for defaults (oldest open)
        _primary_pos  = next(iter(_open_for_sym.values()), {})
        _has_position = bool(_open_for_sym)

        # Default leverage: signal value for ENTRY, held value for HOLD/EXIT
        if _decision == 'ENTRY':
            _default_size = _sig.get('leverage_multiplier') or 0.0
        elif _has_position:
            _default_size = (_primary_pos.get('leverage_multiplier')
                             or _primary_pos.get('size_pct')
                             or _sig.get('leverage_multiplier')
                             or 0.0)
        else:
            _default_size = 0.0

        _stored_note = _primary_pos.get('discretion_note') or ''

        st.markdown(
            f'<div class="form-card-header">'
            f'{escape(_sym)} &nbsp;<span class="badge badge-{_decision}">{_decision}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Action toggle (outside form -> live rerender) ──────────────────────
        _action_key     = f'action_radio_{_sym}'
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

        # ── Position selectbox for EXIT (outside form -> rerender updates leverage max) ──
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
                    pid: f"{pid}  —  {pos.get('entry_date','?')} @ ${pos.get('entry_price',0):,.2f}"
                    for pid, pos in _open_for_sym.items()
                }
                _selected_pid = st.selectbox(
                    'Position to exit',
                    options=list(_pid_labels.keys()),
                    format_func=lambda k: _pid_labels[k],
                    key=f'exit_pid_{_sym}',
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

        with st.form(key=f'trade_form_{_sym}'):
            if not _is_exit:
                # ── ENTRY form ────────────────────────────────────────────────
                _ent_type_default = 0 if _sig.get('entry_long') else 1
                st.radio(
                    'Entry type',
                    ['Strategy', 'Discretionary'],
                    index=_ent_type_default,
                    horizontal=True,
                    key=f'entry_type_{_sym}',
                )
                _direction = st.radio(
                    'Direction',
                    ['Long', 'Short'],
                    horizontal=True,
                    key=f'direction_{_sym}',
                )
                _size = st.number_input(
                    'Actual leverage (x)',
                    value=float(round(_default_size, 2)),
                    min_value=0.0,
                    format='%.2f',
                    key=f'size_{_sym}',
                )
                _price = st.number_input(
                    'Actual price',
                    value=float(round(_default_price, 2)),
                    format='%.2f',
                    key=f'price_{_sym}',
                )
                _exit_lev = None
            else:
                # ── EXIT form ─────────────────────────────────────────────────
                _exit_reason_widget = st.radio(
                    'Exit reason',
                    ['Strategy', 'Discretionary'],
                    key=f'exit_reason_{_sym}',
                    horizontal=True,
                )
                _exit_lev = st.number_input(
                    'Exit leverage (x)',
                    value=float(round(_held_lev, 2)),
                    min_value=0.01,
                    max_value=_held_lev,
                    format='%.2f',
                    key=f'exit_lev_{_sym}',
                )
                _price = st.number_input(
                    'Actual price',
                    value=float(round(_default_price, 2)),
                    format='%.2f',
                    key=f'price_{_sym}',
                )
                _direction = None
                _size      = _held_lev   # held leverage, for trade log reference

            _disc = st.text_input(
                'Discretion note',
                key=f'note_{_sym}',
            )
            _submitted = st.form_submit_button('Log trade', use_container_width=True)

        if _stored_note:
            with st.expander('📋 stored note'):
                st.caption(_stored_note)

        if _submitted:
            _action_final  = st.session_state.get(_action_key, 'ENTRY')
            _is_exit_final = (_action_final == 'EXIT')

            if _is_exit_final:
                _exit_amount   = _exit_lev if _exit_lev is not None else _held_lev
                _is_full_exit  = (_exit_amount >= _held_lev * 0.999)
                _exit_type_log = 'full' if _is_full_exit else 'partial'
                # Resolve selected position_id (from selectbox or only open position)
                _pid_to_exit   = st.session_state.get(f'exit_pid_{_sym}', _selected_pid)
                if _pid_to_exit is None and _open_for_sym:
                    _pid_to_exit = next(iter(_open_for_sym))
                _sub_sig = ('EXIT', round(_price, 4), round(_exit_amount, 4), _pid_to_exit)
            else:
                _dir_val = (_direction or 'Long').lower()
                _sub_sig = ('ENTRY', round(_price, 4), round(_size, 4), _dir_val)

            _guard_key = f'_last_trade_{_sym}'
            if st.session_state.get(_guard_key) != _sub_sig:
                st.session_state[_guard_key] = _sub_sig
                _theo_stop = _sig['theoretical_stop']

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
                        exit_reason=st.session_state.get(f'exit_reason_{_sym}', 'Strategy exit'),
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
                    _pnl_usd_exit = (
                        (_price - _entry_p) / _entry_p * _full_sz_usd * _frac_closed
                        if _entry_p > 0 else 0.0
                    )
                    _old_rc = load_realised_capital(DATA_DIR)
                    _new_rc = update_realised_capital(DATA_DIR, _pnl_usd_exit, _pid_to_exit)
                    print(f"Capital updated: ${_old_rc:.2f} -> ${_new_rc:.2f} "
                          f"(trade: {_pid_to_exit}, P&L: ${_pnl_usd_exit:+.2f})")
                elif not _is_exit_final:
                    # Generate new FIFO position_id
                    _positions_fresh = load_positions(DATA_DIR)
                    _new_pid         = _next_position_id(_sym, _positions_fresh)
                    _signal_snapshot = {
                        'close':         float(_sig['close']),
                        'entry_long':    bool(_sig.get('entry_long', False)),
                        'adx':           float(_sig.get('adx', 0)),
                        'adx_threshold': int(_sig.get('adx_threshold', 0)),
                        'adx_strong':    bool(_sig.get('adx_strong', False)),
                        'caution_long':  bool(_sig.get('caution_long', False)),
                        'caution_short': bool(_sig.get('caution_short', False)),
                        'caution_obv':   bool(_sig.get('caution_obv', False)),
                        'entry_reasons': list(_sig.get('entry_reasons', [])),
                        'entry_path':    str(_sig.get('stop_detail', {}).get('entry_path', '')),
                    }
                    # ── Capital snapshot — frozen at submission time ───────────
                    _snap_coin_cap = get_coin_capital(_sym)
                    _snap_size_usd = _snap_coin_cap * _size   # actual leverage × coin capital
                    _snap_weight   = COIN_WEIGHTS.get(_sym)   # None if evenly split
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
                        entry_type=st.session_state.get(f'entry_type_{_sym}', 'Strategy'),
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


# ── Section 2: Entry conditions ───────────────────────────────────────────────
rows_html = ''
for c in coin_rows:
    sig = c['sig']

    # New helper for full-box background coloring (Green/Red)
    def _bg_box(flag):
        cls = 'entry-t' if flag else 'entry-f'
        return f'<td class="c {cls}">{"TRUE" if flag else "FALSE"}</td>'

    def _tf(flag, amber=False):
        """White background with coloured text; Caution Long TRUE uses amber cell."""
        if amber and flag:
            return '<td class="c caution">TRUE</td>'
        txt_cls = 't' if flag else 'f'
        return f'<td class="c"><span class="{txt_cls}">{"TRUE" if flag else "FALSE"}</span></td>'

    if sig['has_vol_ma']:
        vol_ratio = f"{sig['vol_vol_ma_ratio']:.2f}" if sig.get('vol_vol_ma_ratio') is not None else '—'
        vol_td    = f'<td class="r">{vol_ratio}</td>'
        # Box background for Vol above
        vol_ab_td = _bg_box(sig['vol_above_ma'])
    else:
        vol_td    = '<td class="c" style="color:#888780">—</td>'
        vol_ab_td = '<td class="c" style="color:#888780">—</td>'

    entry_cls = 'entry-t' if sig['entry_long'] else 'entry-f'
    reason    = (f'<br><span style="font-size:10px;font-weight:400">'
                 f'{escape(", ".join(sig["entry_reasons"]))}</span>') if sig['entry_reasons'] else ''
    entry_td  = f'<td class="c {entry_cls}">{"TRUE" if sig["entry_long"] else "FALSE"}{reason}</td>'

    # Box background for ADX override
    adx_override_td = _bg_box(sig['adx_strong'])

    ratio_str = f"{sig['close_ema_ratio']:.2f}" if sig['close_ema_ratio'] is not None else '—'

    rows_html += f"""
    <tr>
      <td class="asset-name">{escape(c['symbol'])}</td>
      <td class="r">{sig['ema']:,.2f}</td>
      <td class="r">{ratio_str}</td>
      {_bg_box(sig['close_above_ema'])}
      <td class="r">{sig['adx']:.1f}</td>
      <td class="r">{sig['adx_threshold']}</td>
      {adx_override_td}
      {vol_td}
      {vol_ab_td}
      {_bg_box(sig['caution_obv'])}
      {_tf(sig['caution_long'], amber=True)}
      {entry_td}
    </tr>"""

col_title, col_help = st.columns([1, 6])
with col_title:
    st.markdown("#### ENTRY CONDITIONS")
with col_help:
    with st.expander("Explanation"):
        st.markdown("""

 **E1 — EMA Filter:**
- The price must be trending up. Specifically, Close > EMA.

**E2 — Volume Validation:**
- Volume > Vol_MA (not used for BTC/AVAX)

**E3 — Caution:**
- **Caution_Long** blocks entry if True.

**Power Clause:** ADX > adx_override, allows entry, betting on pure parabolic momentum
""")
st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table dash-table-labeled">
    <thead><tr>
      <th>Asset</th><th class="r">EMA ($)</th><th class="r">Close/EMA</th>
      <th class="c">EMA cross</th><th class="r">ADX</th><th class="r">Threshold</th>
      <th class="c">ADX override</th><th class="r">Vol/VolMA</th><th class="c">Vol above</th>
      <th class="c">Caution OBV</th><th class="c">Caution Long</th><th class="c">Entry Long</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Section 3: Caution flags ──────────────────────────────────────────────────

rows_html = ''
for c in coin_rows:
    sig = c['sig']

    def _caution_td(flag):
        if flag:
            return '<td class="c caution">TRUE</td>'
        return f'<td class="c"><span class="f">FALSE</span></td>'

    rows_html += f"""
    <tr>
      <td class="asset-name">{escape(c['symbol'])}</td>
      {_caution_td(sig['caution_long'])}
      {_caution_td(sig['caution_short'])}
      {_caution_td(sig['caution_obv'])}
    </tr>"""

col_title, col_help = st.columns([1, 6])
with col_title:
    st.markdown("#### CAUTION FLAGS")
with col_help:
    with st.expander("Explanation"):
        st.markdown("""
**C1 — Caution Long (Overextension)**
- **OBV Divergence:** Price is rising, but OBV < OBV_MA.
- **Price Stretch:** Swing_High_Cau - Low > 1.5 × ATR, mean-reversion risk.

**C2 — Caution Short (Bear Exhaustion)**
- **Trend Breach:** Close > EMA
- **Downside Stretch:** High - Swing_Lo_Cau > 1.5 × ATR.
- Indicating a shift back to bullish/neutral bias.

**C3 — Caution "Both" (Volatility/Chop)**
- Triggered when **C1 and C2** are active simultaneously, stop multiplier to survive market noise.
""")
st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table dash-table-labeled">
    <thead><tr>
      <th>Asset</th>
      <th class="c">Caution Long</th><th class="c">Caution Short</th><th class="c">Caution OBV</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Section 3b: Stop loss calculations ───────────────────────────────────────

def _path_badge(p):
    return f'<span class="badge-{escape(p)}">{escape(p)}</span>'

def _dash_or(v, fmt_fn):
    return '—' if v is None else fmt_fn(v)

with st.expander("STOP LOSS DETAILS"):
    st.markdown("""
**S1 Entry Stop:** Stop = Swing_Hi_Stp − (ATR_Stp × Mult × Scale) · path: ent_normal / ent_caution / ent_both
**S2 Trailing Ratchet:** Stop can only move up — Stop = max(prev, new) using OS multiplier
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

    def _badge_row(label, getter):
        tds = ''.join(
            f'<td style="text-align:center">{_path_badge(getter(c))}</td>'
            for c in coin_rows
        )
        return f'<tr><td class="field-label">{escape(label)}</td>{tds}</tr>'

    upd_yes = '<span class="upd-yes">↑ Yes</span>'
    upd_no  = '<span class="upd-no">No</span>'
    upd_tds = ''.join(
        f'<td style="text-align:right;padding-right:24px">'
        f'{upd_yes if c["sig"]["stop_detail"]["hold_stop_updated"] else upd_no}</td>'
        for c in coin_rows
    )

    rows_html = (
        _divider('Inputs') +
        _row('Swing Hi Stp ($)', [f"{c['sig']['stop_detail']['swing_hi_stp']:,.2f}"   for c in coin_rows]) +
        _row('ATR Stp ($)',      [f"{c['sig']['stop_detail']['atr_stp']:,.2f}"         for c in coin_rows]) +
        _row('Stop ATR scale',   [f"{c['sig']['stop_detail']['stop_atr_scale']:.2f}"   for c in coin_rows]) +
        _divider('Entry stop (Day 1)') +
        _badge_row('Path', lambda c: c['sig']['stop_detail']['entry_path']) +
        _row('Multiplier',       [f"{c['sig']['stop_detail']['entry_multiplier']:.2f}" for c in coin_rows]) +
        _row('Entry stop ($)',   [f"{c['sig']['stop_detail']['entry_stop']:,.2f}"       for c in coin_rows]) +
        _divider('Hold stop (Day 2+ ratchet)') +
        _badge_row('Path', lambda c: c['sig']['stop_detail']['hold_path']) +
        _row('Multiplier',       [f"{c['sig']['stop_detail']['hold_multiplier']:.2f}"  for c in coin_rows]) +
        _row('Hold candidate ($)', [f"{c['sig']['stop_detail']['hold_stop_candidate']:,.2f}" for c in coin_rows]) +
        _row('Previous stop ($)', [_dash_or(c['sig']['stop_detail']['hold_stop_previous'], lambda v: f'{v:,.2f}') for c in coin_rows]) +
        _row('Final stop ($)',    [_dash_or(c['sig']['stop_detail']['hold_stop_final'],    lambda v: f'{v:,.2f}') for c in coin_rows]) +
        f'<tr><td class="field-label">Updated</td>{upd_tds}</tr>'
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
    <tbody>{rows_html}</tbody>
  </table>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Parameters ────────────────────────────────────────────────────────────────

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

rows_html = ''
for key in all_param_keys:
    tds = ''
    for c in coin_rows:
        val      = c['all_params'].get(key)
        is_fixed = key in c['fixed_keys']
        f_badge  = ' <span class="badge-fixed">F</span>' if is_fixed else ''
        tds     += (f'<td style="text-align:right;padding-right:24px">'
                    f'{fmt(val)}{f_badge}</td>')
    rows_html += f'<tr><td class="field-label">{escape(key)}</td>{tds}</tr>'

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
    <tbody>{rows_html}</tbody>
  </table>
  </div>
  <div style="padding:6px 16px 8px;font-size:11px;color:#888780;border-top:1px solid #e4e4e1">
    <span class="badge-fixed">F</span> &nbsp;fixed by stability analysis
  </div>
</div>
""", unsafe_allow_html=True)
