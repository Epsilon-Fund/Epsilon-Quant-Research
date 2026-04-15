"""
Epsilon Fund — Live Trading Dashboard (Streamlit)

Imports all computation from dashboard.py — no logic is duplicated here.
dashboard.py is the sole computation layer.

Run:
    cd live_trading
    streamlit run streamlit_app.py
"""

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
    get_execution_price,
    apply_decision,
    get_coin_capital,
)
from config import ACTIVE_ASSETS, EXECUTION_HOUR, CAPITAL, COIN_WEIGHTS


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Epsilon Fund — Live Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Card wrapper */
  .dashboard-card {
      background: white;
      border: 1px solid #e4e4e1;
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      padding: 0;
      margin-bottom: 14px;
      overflow: hidden;
  }
  .card-header {
      padding: 10px 16px 8px;
      border-bottom: 1px solid #e4e4e1;
      display: flex;
      align-items: center;
      gap: 12px;
  }
  .card-header .section-label {
      font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
      text-transform: uppercase; color: #5a5a55; margin: 0;
  }
  .card-header .section-note {
      font-size: 11px; color: #8a8a85;
  }

  /* Base table */
  .dash-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
  .dash-table th {
      font-size: 11px; font-weight: 500; letter-spacing: 0.05em;
      text-transform: uppercase; color: #8a8a85;
      padding: 7px 12px 5px; text-align: left;
      border-bottom: 1px solid #e4e4e1; white-space: nowrap;
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
  .asset-name { font-weight: 500; font-size: 12px; }
  .asset-alloc { font-size: 11px; color: #8a8a85; }

  /* Decision badges */
  .badge {
      display: inline-block; padding: 2px 8px; border-radius: 3px;
      font-size: 11px; font-weight: 600; letter-spacing: 0.04em; white-space: nowrap;
  }
  .badge-ENTRY { background: #EAF3DE; color: #3B6D11; }
  .badge-HOLD  { background: #FAEEDA; color: #854F0B; }
  .badge-EXIT  { background: #FCEBEB; color: #A32D2D; }
  .badge-FLAT  { background: #F1EFE8; color: #5F5E5A; }

  /* Bool text */
  .t { color: #3B6D11; font-weight: 500; }
  .f { color: #A32D2D; }

  /* Coloured cells */
  .cond-t  { background: #f3f9ed; }
  .cond-f  { background: #fdf1f1; }
  .caution { background: #FAEEDA; color: #854F0B; font-weight: 500; }
  .adx-ov  { background: #f3f9ed; }
  .entry-t { background: #EAF3DE; color: #3B6D11; font-weight: 600; }
  .entry-f { background: #FCEBEB; color: #A32D2D; font-weight: 600; }

  /* Stop ratchet */
  .stop-up   { color: #3B6D11; font-size: 11px; }
  .stop-prev { color: #8a8a85; font-size: 11px; }

  /* Transposed tables */
  .dash-table td.field-label {
      font-size: 12px; font-weight: 500; color: #8a8a85;
      white-space: nowrap; background: #fafaf9;
      border-right: 1px solid #e4e4e1;
      width: 160px; min-width: 160px; max-width: 160px;
  }
  .divider-row td {
      background: #f5f5f3; font-size: 10px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.07em;
      color: #8a8a85; padding: 4px 12px;
      border-bottom: 1px solid #e4e4e1;
  }

  /* Path badges */
  .badge-ent_normal  { background: #f0f0ee; color: #6a6a65; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-ent_caution { background: #FAEEDA; color: #854F0B; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-ent_both    { background: #FCEBEB; color: #A32D2D; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-pos_normal  { background: #f0f0ee; color: #6a6a65; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-pos_caution { background: #FAEEDA; color: #854F0B; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }

  /* Updated cell */
  .upd-yes { color: #3B6D11; font-weight: 500; }
  .upd-no  { color: #8a8a85; }

  /* Page header meta */
  .dash-meta {
      font-size: 12px; color: #8a8a85;
      margin-bottom: 18px; line-height: 1.8;
  }
  .dash-meta strong { color: #5a5a55; font-weight: 500; }
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
        lp_entry   = live_params[symbol]
        params     = lp_entry['params']
        strategy   = lp_entry['strategy']
        fixed_keys = set(lp_entry.get('fixed_param_keys', []))

        daily_df    = fetch_ohlcv(symbol)
        signal_date = daily_df.index[-1].date()
        sig         = compute_signals(daily_df, params, strategy)

        hourly_df  = fetch_hourly_recent(symbol, days=3)
        exec_price = get_execution_price(hourly_df, signal_date, EXECUTION_HOUR)

        position_id   = symbol
        position      = positions.get(position_id)
        coin_cap      = get_coin_capital(symbol)
        decision_dict = apply_decision(sig, position, exec_price, coin_cap)
        sig.update(decision_dict)

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

    return coin_rows, signal_date


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
<h1 style="font-size:20px;font-weight:600;letter-spacing:-0.01em;margin-bottom:10px">
  Epsilon Fund — Live Trading Dashboard
</h1>
<div class="dash-meta">
  <strong>Signal date:</strong> {signal_date} &nbsp;&nbsp;
  <strong>Generated:</strong> {generated_at} UTC &nbsp;&nbsp;
  <strong>Execution hour:</strong> {EXECUTION_HOUR}h UTC (T+1) &nbsp;&nbsp;
  <strong>Capital:</strong> ${CAPITAL:,}
</div>
""", unsafe_allow_html=True)


# ── Section 1: Decisions ──────────────────────────────────────────────────────

rows_html = ''
for c in coin_rows:
    sig = c['sig']
    d   = sig['decision']

    alloc_note = f"${c['coin_capital']:,.0f} alloc ({c['coin_weight']*100:.0f}% of ${CAPITAL:,})"
    badge      = f'<span class="badge badge-{d}">{d}</span>'

    size_pct  = f"{sig['size_pct']:.2f}x" if sig['size_pct'] is not None else '—'
    if d == 'HOLD' and sig['size_pct'] is not None:
        size_pct += '<br><span style="font-size:11px;color:#8a8a85">held</span>'
    size_usd   = f"${sig['size_usd']:,.0f}" if sig['size_usd'] is not None else '—'
    size_units = f"{sig['size_units']:.4f}" if sig['size_units'] is not None else '—'
    if d == 'HOLD' and sig['size_units'] is None:
        size_units = '<span style="font-size:11px;color:#8a8a85">in position</span>'

    if sig['current_stop'] is not None:
        stop_html = f"${sig['current_stop']:,.2f}"
        if d == 'HOLD' and sig.get('stop_updated'):
            stop_html += f' <span class="stop-up">↑</span><br><span class="stop-prev">was ${sig["old_stop"]:,.2f}</span>'
    else:
        stop_html = '—'

    exec_html = f"${c['exec_price']:,.2f}" if c['exec_price'] is not None else '<span style="color:#8a8a85">pending</span>'

    rows_html += f"""
    <tr>
      <td>
        <div class="asset-name">{escape(c['symbol'])}</div>
        <div class="asset-alloc">{escape(alloc_note)}</div>
      </td>
      <td>{badge}</td>
      <td class="r">{size_pct}</td>
      <td class="r">{size_usd}</td>
      <td class="r">{size_units}</td>
      <td class="r">{stop_html}</td>
      <td class="r">${sig['close']:,.2f}</td>
      <td class="r">{exec_html}</td>
    </tr>"""

st.markdown(f"""
<div class="dashboard-card">
  <div class="card-header">
    <span class="section-label">Decisions</span>
    <span class="section-note">Execution at {EXECUTION_HOUR}h UTC on T+1</span>
  </div>
  <table class="dash-table">
    <thead><tr>
      <th>Asset</th><th>Decision</th>
      <th class="r">Size (%)</th><th class="r">Size ($)</th><th class="r">Size (units)</th>
      <th class="r">Stop</th><th class="r">Close</th><th class="r">Exec price</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Section 2: Entry conditions ───────────────────────────────────────────────

hdr_col, hint_col = st.columns([6, 1])
with hdr_col:
    st.markdown('<p class="section-label" style="margin:0 0 6px">Entry conditions</p>', unsafe_allow_html=True)
with hint_col:
    with st.expander("How entry works"):
        st.markdown("""
- **E1 EMA** — Close > EMA (uptrend filter)
- **E2 Volume** — Volume > Vol_MA *(not used for BTC/AVAX)*
- **E3 Caution gate** — Caution_Long blocks entry. **ADX override:** ADX > threshold bypasses caution.
- **E4 Validity** — Swing highs and ATRs must be non-NaN.
""")

rows_html = ''
for c in coin_rows:
    sig = c['sig']
    adx_ov = sig['adx_strong']

    def _tf(flag, amber=False):
        if amber and flag:
            return f'<td class="c caution"><span class="t">TRUE</span></td>'
        cls = 'cond-t' if flag else 'cond-f'
        txt_cls = 't' if flag else 'f'
        return f'<td class="c {cls}"><span class="{txt_cls}">{"TRUE" if flag else "FALSE"}</span></td>'

    def _num_tf(val, flag):
        cls = 'cond-t' if flag else 'cond-f'
        return f'<td class="r {cls}">{escape(str(val))}</td>'

    if sig['has_vol_ma']:
        vol_ratio = f"{sig['vol_vol_ma_ratio']:.2f}" if sig.get('vol_vol_ma_ratio') is not None else '—'
        vol_td    = f'<td class="r {"cond-t" if sig["vol_above_ma"] else "cond-f"}">{vol_ratio}</td>'
        vol_ab_td = _tf(sig['vol_above_ma'])
    else:
        vol_td    = '<td class="c" style="color:#8a8a85">—</td>'
        vol_ab_td = '<td class="c" style="color:#8a8a85">—</td>'

    entry_cls = 'entry-t' if sig['entry_long'] else 'entry-f'
    reason    = f'<br><span style="font-size:10px;font-weight:400">{escape(", ".join(sig["entry_reasons"]))}</span>' if sig['entry_reasons'] else ''
    entry_td  = f'<td class="c {entry_cls}">{"TRUE" if sig["entry_long"] else "FALSE"}{reason}</td>'
    adx_td    = f'<td class="c {"adx-ov" if adx_ov else ""}"><span class="{"t" if adx_ov else "f"}">{"TRUE" if adx_ov else "FALSE"}</span></td>'

    ratio_str = f"{sig['close_ema_ratio']:.2f}" if sig['close_ema_ratio'] is not None else '—'

    rows_html += f"""
    <tr>
      <td class="asset-name">{escape(c['symbol'])}</td>
      <td class="r">${sig['ema']:,.2f}</td>
      {f'<td class="r {"cond-t" if sig["close_above_ema"] else "cond-f"}">{ratio_str}</td>'}
      {_tf(sig['close_above_ema'])}
      {_num_tf(f"{sig['adx']:.1f}", sig['adx_strong'])}
      <td class="r">{sig['adx_threshold']}</td>
      {_tf(sig['adx_strong'])}
      {vol_td}
      {vol_ab_td}
      {_tf(sig['caution_long'], amber=True)}
      {adx_td}
      {entry_td}
    </tr>"""

st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Asset</th><th class="r">EMA</th><th class="r">Close/EMA</th>
      <th class="c">EMA cross</th><th class="r">ADX</th><th class="r">Threshold</th>
      <th class="c">ADX strong</th><th class="r">Vol/VolMA</th><th class="c">Vol above</th>
      <th class="c">Caution Long</th><th class="c">ADX override</th><th class="c">Entry Long</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Section 3: Caution flags ──────────────────────────────────────────────────

hdr_col, hint_col = st.columns([6, 1])
with hdr_col:
    st.markdown('<p class="section-label" style="margin:0 0 6px">Caution flags</p>', unsafe_allow_html=True)
with hint_col:
    with st.expander("How caution works"):
        st.markdown("""
- **C1 OBV divergence** — Price at highs but OBV below OBV_MA. Unsupported move.
- **C2 Overextension** — (Swing_Hi_Cau − Low) > 1.5 × ATR_Cau. Range stretched too wide.
- Either flag sets **Caution_Long = True**, blocking entries unless ADX override is active.
""")

rows_html = ''
for c in coin_rows:
    sig = c['sig']
    def _caution_td(flag):
        if flag:
            return '<td class="c caution"><span class="t">TRUE</span></td>'
        return '<td class="c"><span class="f">FALSE</span></td>'
    rows_html += f"""
    <tr>
      <td class="asset-name">{escape(c['symbol'])}</td>
      {_caution_td(sig['caution_long'])}
      {_caution_td(sig['caution_short'])}
      {_caution_td(sig['caution_obv'])}
    </tr>"""

st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table">
    <thead><tr>
      <th>Asset</th>
      <th class="c">Caution Long</th><th class="c">Caution Short</th><th class="c">Caution OBV</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <div style="padding:7px 16px 9px;font-size:11px;color:#8a8a85;border-top:1px solid #e4e4e1">
    Caution flags suppress entry unless ADX &gt; adx_override threshold.
  </div>
</div>
""", unsafe_allow_html=True)


# ── Section 3b: Stop loss calculations ───────────────────────────────────────

def _path_badge(p):
    return f'<span class="badge-{escape(p)}">{escape(p)}</span>'

def _dash_or(v, fmt_fn):
    return '—' if v is None else fmt_fn(v)

with st.expander("Stop loss calculations — click to expand"):
    coin_ths = ''.join(f'<th style="text-align:right;padding-right:24px">{escape(c["symbol"])}</th>' for c in coin_rows)
    n = len(coin_rows)
    col_w = f'calc((100% - 160px) / {n})'

    def _row(label, cells):
        tds = ''.join(f'<td style="text-align:right;padding-right:24px">{c}</td>' for c in cells)
        return f'<tr><td class="field-label">{escape(label)}</td>{tds}</tr>'

    def _divider(label):
        return f'<tr class="divider-row"><td colspan="{n+1}">{escape(label)}</td></tr>'

    def _badge_row(label, getter):
        tds = ''.join(f'<td style="text-align:center">{_path_badge(getter(c))}</td>' for c in coin_rows)
        return f'<tr><td class="field-label">{escape(label)}</td>{tds}</tr>'

    def _updated_row():
        upd_yes = '<span class="upd-yes">↑ Yes</span>'
        upd_no  = '<span class="upd-no">No</span>'
        tds = ''.join(
            f'<td style="text-align:right;padding-right:24px">{upd_yes if c["sig"]["stop_detail"]["hold_stop_updated"] else upd_no}</td>'
            for c in coin_rows
        )
        return f'<tr><td class="field-label">Updated</td>{tds}</tr>'

    rows_html = (
        _divider('Inputs') +
        _row('Swing Hi Stp',  [f"${c['sig']['stop_detail']['swing_hi_stp']:,.2f}" for c in coin_rows]) +
        _row('ATR Stp',       [f"${c['sig']['stop_detail']['atr_stp']:,.2f}" for c in coin_rows]) +
        _row('Stop ATR scale',[f"{c['sig']['stop_detail']['stop_atr_scale']:.2f}" for c in coin_rows]) +
        _divider('Entry stop (Day 1)') +
        _badge_row('Path', lambda c: c['sig']['stop_detail']['entry_path']) +
        _row('Multiplier',    [f"{c['sig']['stop_detail']['entry_multiplier']:.2f}" for c in coin_rows]) +
        _row('Entry stop',    [f"${c['sig']['stop_detail']['entry_stop']:,.2f}" for c in coin_rows]) +
        _divider('Hold stop (Day 2+ ratchet)') +
        _badge_row('Path', lambda c: c['sig']['stop_detail']['hold_path']) +
        _row('Multiplier',    [f"{c['sig']['stop_detail']['hold_multiplier']:.2f}" for c in coin_rows]) +
        _row('Hold candidate',[f"${c['sig']['stop_detail']['hold_stop_candidate']:,.2f}" for c in coin_rows]) +
        _row('Previous stop', [_dash_or(c['sig']['stop_detail']['hold_stop_previous'], lambda v: f'${v:,.2f}') for c in coin_rows]) +
        _row('Final stop',    [_dash_or(c['sig']['stop_detail']['hold_stop_final'],    lambda v: f'${v:,.2f}') for c in coin_rows]) +
        _updated_row()
    )

    st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table" style="table-layout:fixed;width:100%">
    <colgroup>
      <col style="width:160px">
      {''.join(f'<col style="width:{col_w}">' for _ in coin_rows)}
    </colgroup>
    <thead><tr>
      <th style="text-align:left">Field</th>
      {coin_ths}
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Section 4: Fixed params ───────────────────────────────────────────────────

with st.expander("Fixed params — anchored by stability analysis"):
    all_fixed_keys = sorted({k for c in coin_rows for k in c['fixed_params']})
    n = len(coin_rows)
    col_w = f'calc((100% - 160px) / {n})'
    coin_ths = ''.join(f'<th style="text-align:right;padding-right:24px">{escape(c["symbol"])}</th>' for c in coin_rows)

    rows_html = ''
    for key in all_fixed_keys:
        tds = ''.join(
            f'<td style="text-align:right;padding-right:24px">{fmt(c["fixed_params"].get(key))}</td>'
            for c in coin_rows
        )
        rows_html += f'<tr><td class="field-label">{escape(key)}</td>{tds}</tr>'

    st.markdown(f"""
<div class="dashboard-card">
  <div style="padding:7px 16px;font-size:11px;color:#8a8a85;border-bottom:1px solid #e4e4e1;font-style:italic">
    Anchored by stability analysis. Should not change between optimise.py runs.
  </div>
  <table class="dash-table" style="table-layout:fixed;width:100%">
    <colgroup>
      <col style="width:160px">
      {''.join(f'<col style="width:{col_w}">' for _ in coin_rows)}
    </colgroup>
    <thead><tr>
      <th style="text-align:left">Param</th>
      {coin_ths}
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── Section 5: Optimised params ───────────────────────────────────────────────

with st.expander("Optimised params"):
    all_optim_keys = sorted({k for c in coin_rows for k in c['optim_params']})
    n = len(coin_rows)
    col_w = f'calc((100% - 160px) / {n})'
    coin_ths = ''.join(
        f'<th style="text-align:right;padding-right:24px">{escape(c["symbol"])}'
        f'<br><span style="font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;color:#8a8a85">{escape(c["optimised_on"])}</span>'
        f'</th>'
        for c in coin_rows
    )

    rows_html = ''
    for key in all_optim_keys:
        tds = ''.join(
            f'<td style="text-align:right;padding-right:24px">{fmt(c["optim_params"].get(key))}</td>'
            for c in coin_rows
        )
        rows_html += f'<tr><td class="field-label">{escape(key)}</td>{tds}</tr>'

    st.markdown(f"""
<div class="dashboard-card">
  <table class="dash-table" style="table-layout:fixed;width:100%">
    <colgroup>
      <col style="width:160px">
      {''.join(f'<col style="width:{col_w}">' for _ in coin_rows)}
    </colgroup>
    <thead><tr>
      <th style="text-align:left">Param</th>
      {coin_ths}
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)
