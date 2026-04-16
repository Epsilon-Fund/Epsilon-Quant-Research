"""
Epsilon Fund — Live Trading Dashboard (Streamlit)

Imports all computation from dashboard.py — no logic is duplicated here.
dashboard.py is the sole computation layer. All trade logging is done via
direct file writes in this module — no external Flask server required.

Run:
    streamlit run live_trading/streamlit_app.py
"""

import json
import sys
import os
from datetime import datetime, timezone
from html import escape

_LT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_LT_DIR)
sys.path.insert(0, _LT_DIR)
sys.path.append(os.path.join(_ROOT, 'infrastructure', 'data'))

import streamlit as st

from dashboard import (
    compute_signals,
    load_live_params,
    load_positions,
    fetch_ohlcv,
    fetch_hourly_recent,
    fetch_live_price,
    get_execution_price,
    apply_decision,
    get_coin_capital,
)
from config   import ACTIVE_ASSETS, EXECUTION_HOUR, CAPITAL, COIN_WEIGHTS
from optimise import ASSET_CONFIG

POSITIONS_PATH = os.path.join(_LT_DIR, 'positions.json')
TRADES_PATH    = os.path.join(_LT_DIR, 'trades.json')


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
                 theoretical_size_pct, actual_size_pct, theoretical_stop, discretion_note):
    slippage = (actual_price - theoretical_price) / theoretical_price * 100 if theoretical_price else 0.0
    entry = {
        'timestamp':              datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'),
        'date':                   datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'position_id':            position_id,
        'action':                 action,
        'strategy':               strategy,
        'theoretical_price':      round(theoretical_price, 4),
        'actual_price':           round(actual_price, 4),
        'slippage_pct':           round(slippage, 4),
        'theoretical_size_pct':   theoretical_size_pct,
        'actual_size_pct':        actual_size_pct,
        'theoretical_stop':       theoretical_stop,
        'discretion_note':        discretion_note,
    }
    trades = _load_json(TRADES_PATH, [])
    trades.append(entry)
    _save_json(TRADES_PATH, trades)


def _write_position_entry(position_id, actual_price, actual_size_pct,
                          theoretical_stop, strategy, discretion_note):
    positions = _load_json(POSITIONS_PATH, {})
    positions[position_id] = {
        'in_position':    True,
        'entry_price':    actual_price,
        'entry_date':     datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'current_stop':   theoretical_stop,
        'size_pct':       actual_size_pct,
        'strategy':       strategy,
        'exit_price':     None,
        'exit_date':      None,
        'discretion_note': discretion_note,
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


def _write_stop_update(position_id, new_stop):
    positions = _load_json(POSITIONS_PATH, {})
    if position_id in positions:
        positions[position_id]['current_stop'] = new_stop
    _save_json(POSITIONS_PATH, positions)


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
  .block-container  { background-color: #f8f8f7 !important; padding-top: 1.5rem; padding-bottom: 2rem; }

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
      padding: 7px 12px 5px; text-align: left;
      border-bottom: 1px solid #d3d1c7; white-space: nowrap;
  }
  .dash-table th.r { text-align: right; }
  .dash-table th.c { text-align: center; }
  .dash-table td {
      padding: 8px 12px; border-bottom: 1px solid #e4e4e1;
      vertical-align: middle;
  }
  .dash-table tr:last-child td { border-bottom: none; }
  .dash-table td.r { text-align: right; }
  .dash-table td.c { text-align: center; }

  /* Asset cell */
  .asset-name  { font-weight: 500; font-size: 12px; }
  .asset-alloc { font-size: 11px; color: #888780; }

  /* Decision row tints */
  .row-ENTRY td { background: #f5faf0 !important; }
  .row-HOLD  td { background: #fdf6ec !important; }
  .row-EXIT  td { background: #fef5f5 !important; }

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
  .t { color: #1a5c2a; font-weight: 600; }
  .f { color: #8a1a1a; font-weight: 600; }

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
  }
  .divider-row td {
      background: #f5f5f3; font-size: 10px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.07em;
      color: #888780; padding: 4px 12px;
      border-bottom: 1px solid #d3d1c7;
  }

  /* Fixed param "F" badge (inline in value cell) */
  .badge-fixed {
      background: #3d3d3a; color: #fff; font-size: 9px;
      padding: 1px 4px; border-radius: 3px; margin-left: 5px;
      display: inline-block; vertical-align: middle;
  }

  /* Path badges — high contrast */
  .badge-ent_normal  { background: #3d3d3a; color: #fff; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-ent_caution { background: #8a4f00; color: #fff; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-ent_both    { background: #8a1a1a; color: #fff; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-pos_normal  { background: #3d3d3a; color: #fff; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-pos_caution { background: #8a4f00; color: #fff; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }

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
    live_params = load_live_params()
    if not live_params:
        return [], None

    positions  = load_positions()
    active_set = set(ACTIVE_ASSETS)
    coins      = [sym for sym in live_params if sym in active_set]

    coin_rows   = []
    signal_date = None

    for symbol in coins:
        lp_entry = live_params[symbol]
        params   = lp_entry['params']
        strategy = lp_entry['strategy']

        # Use ASSET_CONFIG as the authoritative source for fixed params;
        # fall back to live_params.json's fixed_param_keys if not found.
        fixed_keys = _ASSET_FIXED_KEYS.get(
            symbol,
            set(lp_entry.get('fixed_param_keys', []))
        )

        daily_df    = fetch_ohlcv(symbol)
        signal_date = daily_df.index[-1].date()
        sig         = compute_signals(daily_df, params, strategy)

        hourly_df  = fetch_hourly_recent(symbol, days=3)
        exec_price = get_execution_price(hourly_df, signal_date, EXECUTION_HOUR)

        position    = positions.get(symbol)
        coin_cap    = get_coin_capital(symbol)
        sig.update(apply_decision(sig, position, exec_price, coin_cap))

        coin_rows.append({
            'symbol':       symbol,
            'strategy':     strategy,
            'sig':          sig,
            'exec_price':   exec_price,
            'all_params':   params,
            'fixed_keys':   fixed_keys,
            'optimised_on': lp_entry.get('optimised_on', 'unknown'),
            'coin_capital': coin_cap,
            'coin_weight':  COIN_WEIGHTS.get(symbol, coin_cap / CAPITAL),
        })

    return coin_rows, signal_date


@st.cache_data(ttl=60, show_spinner=False)
def load_live_prices(symbols: tuple):
    """Fetch current market price for each symbol. Cached for 60 s."""
    return {sym: fetch_live_price(sym) for sym in symbols}


coin_rows, signal_date = load_all()

if not coin_rows:
    st.error("live_params.json is empty — run `optimise.py` first.")
    st.stop()

generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


# ── Helper: format param value ────────────────────────────────────────────────

def fmt(v):
    if v is None: return '—'
    if isinstance(v, int): return str(v)
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f'{v:.2f}'
    return escape(str(v))


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(f"""
<h1 style="font-size:35px;font-weight:700;letter-spacing:-0.01em;margin-bottom:10px">
  Epsilon Fund — Live Trading Dashboard
</h1>
<div class="dash-meta">
  <strong>Signal date:</strong> {signal_date} &nbsp;&nbsp;
  <strong>Generated:</strong> {generated_at} UTC &nbsp;&nbsp;
  <strong>Execution hour:</strong> {EXECUTION_HOUR}h UTC (T+1) &nbsp;&nbsp;
  <strong>Capital ($):</strong> {CAPITAL:,}
</div>
""", unsafe_allow_html=True)


# ── Active Positions ──────────────────────────────────────────────────────────

_positions_now   = load_positions()   # fresh read — not cached
_open            = {pid: p for pid, p in _positions_now.items() if p.get('in_position')}
_coin_sig_by_sym = {c['symbol']: c['sig'] for c in coin_rows}

# Live prices for open positions — 60 s cache, very lightweight ticker calls
_open_syms   = tuple(sorted(_open.keys()))
_live_prices = load_live_prices(_open_syms) if _open_syms else {}

st.markdown("#### ACTIVE POSITIONS")
if not _open:
    st.markdown('<p style="font-size:12px;color:#888780;margin-bottom:14px">No open positions</p>',
                unsafe_allow_html=True)
else:
    from datetime import date as _date
    _pos_rows = ''
    _update_stop_items = []   # (pid, confirmed_stop, suggested_stop) for coins needing update

    for pid, pos in _open.items():
        close       = _live_prices.get(pid)   # current market price, ~1 min fresh
        entry_price = pos['entry_price']
        conf_stop   = pos.get('current_stop')   # confirmed stop from positions.json
        size_pct    = pos.get('size_pct', 0)
        size_usd    = size_pct * CAPITAL

        # Suggested stop: the ratcheted value from apply_decision for HOLD coins
        coin_sig      = _coin_sig_by_sym.get(pid, {})
        coin_decision = coin_sig.get('decision')
        sugg_stop     = coin_sig.get('current_stop') if coin_decision == 'HOLD' else None

        pnl_pct = ((close - entry_price) / entry_price * 100) if close else None
        try:
            days_held = (_date.today() - _date.fromisoformat(pos['entry_date'])).days
        except Exception:
            days_held = '—'

        if pnl_pct is None:
            pnl_td = '<td class="r">—</td>'
        else:
            pnl_cell_cls = 'entry-t' if pnl_pct >= 0 else 'entry-f'
            pnl_sign     = '+' if pnl_pct >= 0 else ''
            pnl_td = f'<td class="r {pnl_cell_cls}">{pnl_sign}{pnl_pct:.2f}%</td>'

        sugg_html = '—'
        if sugg_stop is not None:
            if conf_stop is not None and sugg_stop > conf_stop:
                sugg_html = f'<span class="stop-up">{sugg_stop:,.2f} ↑</span>'
                _update_stop_items.append((pid, conf_stop, sugg_stop))
            else:
                sugg_html = f"{sugg_stop:,.2f}"

        close_html = f"{close:,.2f}" if close else '—'

        _pos_rows += f"""
        <tr>
          <td class="asset-name">{escape(pid)}</td>
          <td class="r">{entry_price:,.2f}</td>
          <td class="r">{close_html}</td>
          <td class="r">{sugg_html}</td>
          <td class="r">{size_pct:.2f}%</td>
          <td class="r">{size_usd:,.0f}</td>
          {pnl_td}
          <td class="r">{days_held}</td>
        </tr>"""

    st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Asset</th>
      <th class="r">Entry ($)</th><th class="r">Live price ($)</th>
      <th class="r">Sugg. stop ($)</th>
      <th class="r">Size (%)</th><th class="r">Size ($)</th>
      <th class="r">Unreal. P&amp;L</th><th class="r">Days held</th>
    </tr></thead>
    <tbody>{_pos_rows}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

    # Update stop buttons — shown below table for any HOLD coins with ratcheted stop
    if _update_stop_items:
        _upd_cols = st.columns(len(_update_stop_items))
        for _col, (_pid, _conf, _sugg) in zip(_upd_cols, _update_stop_items):
            with _col:
                if st.button(
                    f"Update stop {_pid}: {_conf:,.2f} → {_sugg:,.2f}",
                    key=f"upd_stop_{_pid}",
                ):
                    _write_stop_update(_pid, _sugg)
                    st.cache_data.clear()
                    st.rerun()


# ── Section 1: Decisions ──────────────────────────────────────────────────────

rows_html = ''
for c in coin_rows:
    sig = c['sig']
    d   = sig['decision']

    alloc_note = f"{c['coin_capital']:,.0f} alloc ({c['coin_weight']*100:.0f}% of {CAPITAL:,})"
    badge      = f'<span class="badge badge-{d}">{d}</span>'

    size_pct = f"{sig['size_pct']:.2f}x" if sig['size_pct'] is not None else '—'
    if d == 'HOLD' and sig['size_pct'] is not None:
        size_pct += '<br><span style="font-size:11px;color:#888780">held</span>'
    size_usd   = f"{sig['size_usd']:,.0f}" if sig['size_usd'] is not None else '—'
    size_units = f"{sig['size_units']:.4f}" if sig['size_units'] is not None else '—'
    if d == 'HOLD' and sig['size_units'] is None:
        size_units = '<span style="font-size:11px;color:#888780">in position</span>'

    if sig['current_stop'] is not None:
        stop_html = f"{sig['current_stop']:,.2f}"
        if d == 'HOLD' and sig.get('stop_updated'):
            stop_html += f' <span class="stop-up">↑</span><br><span class="stop-prev">was {sig["old_stop"]:,.2f}</span>'
    else:
        stop_html = '—'

    exec_html = f"{c['exec_price']:,.2f}" if c['exec_price'] is not None else '<span style="color:#888780">pending</span>'

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
      <td class="r">{size_units}</td>
      <td class="r">{stop_html}</td>
      <td class="r">{sig['close']:,.2f}</td>
      <td class="r">{exec_html}</td>
    </tr>"""

st.markdown("#### DECISIONS")
st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Asset</th><th>Decision</th>
      <th class="r">Size (%)</th><th class="r">Size ($)</th><th class="r">Size (units)</th>
      <th class="r">Stop ($)</th><th class="r">Close ($)</th><th class="r">Exec price ($)</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Trade Log Forms ───────────────────────────────────────────────────────────

st.markdown("#### TRADE LOG")

_positions_for_forms = load_positions()   # fresh read
_form_cols = st.columns(len(coin_rows))

for _fi, _c in enumerate(coin_rows):
    with _form_cols[_fi]:
        _sym      = _c['symbol']
        _sig      = _c['sig']
        _decision = _sig['decision']
        _strategy = _c['strategy']
        _pos      = _positions_for_forms.get(_sym, {})

        # Default action based on signal decision
        _default_action_idx = 1 if _decision == 'EXIT' else 0

        # Default price: exec_price if available, else close
        _default_price = _c['exec_price'] if _c['exec_price'] is not None else _sig['close']

        # Default size: signal size for ENTRY, held size for HOLD, held size for EXIT, 0 for FLAT
        if _decision == 'ENTRY':
            _default_size = _sig['size_pct'] or 0.0
        elif _decision in ('HOLD', 'EXIT'):
            _default_size = _pos.get('size_pct', _sig.get('size_pct') or 0.0)
        else:
            _default_size = 0.0

        _stored_note = _pos.get('discretion_note') or ''

        st.markdown(
            f'<div class="form-card-header">'
            f'{escape(_sym)} &nbsp;<span class="badge badge-{_decision}">{_decision}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.form(key=f'trade_form_{_sym}'):
            _action = st.selectbox(
                'Action',
                ['ENTRY', 'EXIT'],
                index=_default_action_idx,
                key=f'action_{_sym}',
            )
            _price = st.number_input(
                'Actual price',
                value=float(round(_default_price, 2)),
                format='%.2f',
                key=f'price_{_sym}',
            )
            _size = st.number_input(
                'Actual size (%)',
                value=float(round(_default_size, 4)),
                min_value=0.0,
                format='%.4f',
                key=f'size_{_sym}',
            )
            _disc = st.text_input(
                'Discretion note',
                key=f'note_{_sym}',
            )
            _submitted = st.form_submit_button('Log trade', use_container_width=True)

        if _stored_note:
            with st.expander('📋 stored note'):
                st.caption(_stored_note)

        if _submitted:
            _theo_stop = _sig['theoretical_stop']
            _write_trade(
                position_id=_sym,
                action=_action,
                strategy=_strategy,
                theoretical_price=_default_price,
                actual_price=_price,
                theoretical_size_pct=_default_size,
                actual_size_pct=_size,
                theoretical_stop=_theo_stop,
                discretion_note=_disc,
            )
            if _action == 'ENTRY':
                _write_position_entry(_sym, _price, _size, _theo_stop, _strategy, _disc)
            elif _action == 'EXIT':
                _write_position_exit(_sym, _price, _disc)
            st.cache_data.clear()
            st.rerun()


# ── Section 2: Entry conditions ───────────────────────────────────────────────

rows_html = ''
for c in coin_rows:
    sig = c['sig']

    def _tf(flag, amber=False):
        """White background with coloured text; Caution Long TRUE uses amber cell."""
        if amber and flag:
            return '<td class="c caution">TRUE</td>'
        txt_cls = 't' if flag else 'f'
        return f'<td class="c"><span class="{txt_cls}">{"TRUE" if flag else "FALSE"}</span></td>'

    if sig['has_vol_ma']:
        vol_ratio = f"{sig['vol_vol_ma_ratio']:.2f}" if sig.get('vol_vol_ma_ratio') is not None else '—'
        vol_td    = f'<td class="r">{vol_ratio}</td>'
        vol_ab_td = _tf(sig['vol_above_ma'])
    else:
        vol_td    = '<td class="c" style="color:#888780">—</td>'
        vol_ab_td = '<td class="c" style="color:#888780">—</td>'

    entry_cls = 'entry-t' if sig['entry_long'] else 'entry-f'
    reason    = (f'<br><span style="font-size:10px;font-weight:400">'
                 f'{escape(", ".join(sig["entry_reasons"]))}</span>') if sig['entry_reasons'] else ''
    entry_td  = f'<td class="c {entry_cls}">{"TRUE" if sig["entry_long"] else "FALSE"}{reason}</td>'

    adx_strong = sig['adx_strong']
    adx_td     = f'<td class="c"><span class="{"t" if adx_strong else "f"}">{"TRUE" if adx_strong else "FALSE"}</span></td>'

    ratio_str = f"{sig['close_ema_ratio']:.2f}" if sig['close_ema_ratio'] is not None else '—'

    rows_html += f"""
    <tr>
      <td class="asset-name">{escape(c['symbol'])}</td>
      <td class="r">{sig['ema']:,.2f}</td>
      <td class="r">{ratio_str}</td>
      {_tf(sig['close_above_ema'])}
      <td class="r">{sig['adx']:.1f}</td>
      <td class="r">{sig['adx_threshold']}</td>
      {_tf(sig['adx_strong'])}
      {vol_td}
      {vol_ab_td}
      {_tf(sig['caution_long'], amber=True)}
      {adx_td}
      {entry_td}
    </tr>"""

col_title, col_help = st.columns([1, 6])
with col_title:
    st.markdown("#### ENTRY CONDITIONS")
with col_help:
    with st.expander("Explanation"):
        st.markdown("""
 **E1 — EMA Filter:**
- The price must be trending up. Specifically, `Close > EMA`.

**E2 — Volume Validation**
— `Volume > Vol_MA` (not used for BTC/AVAX)

**E3 - Caution Gates**
- Entry is blocked if `Caution_Long` is `True`.
- **Power Clause:** ADX > adx_override, betting on pure parabolic momentum
""")
st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Asset</th><th class="r">EMA ($)</th><th class="r">Close/EMA</th>
      <th class="c">EMA cross</th><th class="r">ADX</th><th class="r">Threshold</th>
      <th class="c">ADX strong</th><th class="r">Vol/VolMA</th><th class="c">Vol above</th>
      <th class="c">Caution Long</th><th class="c">ADX override</th><th class="c">Entry Long</th>
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
- **OBV Divergence:** Price is rising, but `OBV < OBV_MA`.
- **Price Stretch:** Swing_High_Cau - Low > 1.5 × ATR, mean-reversion risk.

**C2 — Caution Short (Bear Exhaustion)**
- **Trend Breach:** `Close > EMA`
- **Downside Stretch:** `High - Swing_Lo_Cau > 1.5 × ATR.`
- Indicating a shift back to bullish/neutral bias.

**C3 — Caution "Both" (Volatility/Chop)**
- Triggered when **C1 and C2** are active simultaneously, stop multiplier to survive market noise.
""")
st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
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
    with st.expander("Explanation"):
        st.markdown("""
**S1 Entry Stop:**
  - **Stop** = `Swing_Hi_Stp − (ATR_Stp × Mult × Scale)`
  - Multiplier path: `ent_normal / ent_caution / ent_both`

**S2 Trailing Ratchet:**
  - **Stop can only move up, never down**, using OS multiplier
  - Stop = max(prev, new)
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
