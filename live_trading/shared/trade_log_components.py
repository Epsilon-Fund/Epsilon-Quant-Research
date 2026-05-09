"""
shared/trade_log_components.py
==============================
Shared rendering logic for all Trade Journal pages.

Public API
----------
render_fund_tab(dashboard_dirs, prefix)
    Aggregate trades from multiple dashboard directories and render the
    flat cross-strategy fund log with summary metrics.

render_strategy_tab(data_dir, prefix, strategy_keys, display_name, show_strategy_col)
    Load trades from a single dashboard directory and render the full
    per-strategy view: coin sub-tabs, open positions, performance table,
    and execution detail table.

Both functions accept ``prefix`` which is prepended to every Streamlit
widget key, preventing DuplicateWidgetID errors when multiple views are
rendered on the same page.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import streamlit as st

# ── Resolve live_trading/ root so shared imports always work ──────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))   # shared/
_LT_DIR = os.path.dirname(_HERE)                        # live_trading/
if _LT_DIR not in sys.path:
    sys.path.insert(0, _LT_DIR)

from shared.data_loader import (
    build_trade_pairs,
    derive_entry_reason,
    get_coin_capital,
    load_config,
    load_trades,
    _pid_to_symbol,
)
from shared.styles import apply_styles


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_STRATEGY_LABELS: Dict[str, str] = {
    'momentum_swing':  'Swing',
    'momentum_no_vol': 'No Vol',
}

_DATE_OPTIONS = ["All time", "Last 30 days", "Last 90 days", "Year to date", "Custom"]


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=120, show_spinner="Loading trade data…")
def _load_single(data_dir: str):
    """Load trades + pairs from one dashboard directory."""
    trades = load_trades(data_dir)
    pairs  = build_trade_pairs(data_dir)
    return trades, pairs


@st.cache_data(ttl=120, show_spinner="Loading trade journal…")
def _load_all(dashboard_dirs_items: tuple):
    """
    Aggregate trades + pairs from multiple dashboard directories.
    Accepts a tuple of (name, path) pairs so the result is cacheable.
    """
    merged_trades = []
    merged_closed = []
    merged_open   = []

    for dashboard_name, data_dir in dashboard_dirs_items:
        trades = load_trades(data_dir)
        for t in trades:
            t['_data_dir']  = data_dir
            t['_dashboard'] = dashboard_name
        merged_trades.extend(trades)

        pairs = build_trade_pairs(data_dir)
        for p in pairs.get('closed', []):
            p['_data_dir']  = data_dir
            p['_dashboard'] = dashboard_name
        for t in pairs.get('open', []):
            t['_data_dir']  = data_dir
            t['_dashboard'] = dashboard_name
        merged_closed.extend(pairs.get('closed', []))
        merged_open.extend(pairs.get('open', []))

    merged_trades.sort(key=lambda t: t['date'] or datetime.min.date())
    merged_closed.sort(key=lambda p: p['entry_date'] or datetime.min.date())

    return (
        merged_trades,
        {'closed': merged_closed, 'open': merged_open},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PURE COMPUTE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_exit_reason(r) -> str:
    r = (r or '').strip()
    if not r or r == '—':
        return '—'
    if 'Discretion' in r:
        return 'Discretionary'
    return 'Strategy'


def _compute_stats(pairs: list) -> dict | None:
    n = len(pairs)
    if n == 0:
        return None

    pnls    = [p['actual_pnl_usd'] for p in pairs]
    # Use net return (after round-trip costs) so stats are consistent with P&L ($).
    # Falls back to gross actual_return_pct for legacy records missing the field.
    returns = [p.get('actual_net_return_pct', p['actual_return_pct']) for p in pairs]
    wins    = [p for p in pairs if p['actual_pnl_usd'] > 0]
    losses  = [p for p in pairs if p['actual_pnl_usd'] <= 0]
    maes    = [p['mae_pct'] for p in pairs if p['mae_pct'] is not None]
    days    = [p['holding_days'] for p in pairs]

    total_pnl     = sum(pnls)
    win_rate      = len(wins) / n * 100
    sum_wins      = sum(p['actual_pnl_usd'] for p in wins)
    sum_losses    = abs(sum(p['actual_pnl_usd'] for p in losses))
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else float('inf')

    avg_win_usd  = sum_wins / len(wins) if wins else 0
    avg_loss_usd = sum(p['actual_pnl_usd'] for p in losses) / len(losses) if losses else 0
    avg_win_pct  = sum(p.get('actual_net_return_pct', p['actual_return_pct']) for p in wins)   / len(wins)   if wins   else 0
    avg_loss_pct = sum(p.get('actual_net_return_pct', p['actual_return_pct']) for p in losses) / len(losses) if losses else 0
    avg_ret_pct  = sum(returns) / n

    largest_win  = max(pairs, key=lambda p: p['actual_pnl_usd']) if wins  else None
    largest_loss = min(pairs, key=lambda p: p['actual_pnl_usd']) if losses else None

    # Sort chronologically before computing streaks and drawdown so the result
    # is identical regardless of how pairs were ordered by the caller
    # (fund tab sorts by entry_date; momentum tab preserves build_trade_pairs order).
    sorted_pairs = sorted(pairs, key=lambda p: p['exit_date'] or p['entry_date'])
    results = ['W' if p['actual_pnl_usd'] > 0 else 'L' for p in sorted_pairs]
    max_win_streak = max_loss_streak = cur_w = cur_l = 0
    for r in results:
        if r == 'W':
            cur_w += 1; cur_l = 0
            max_win_streak  = max(max_win_streak, cur_w)
        else:
            cur_l += 1; cur_w = 0
            max_loss_streak = max(max_loss_streak, cur_l)

    cur_result = results[-1]
    cur_streak = 1
    for r in reversed(results[:-1]):
        if r == cur_result:
            cur_streak += 1
        else:
            break

    cum    = np.cumsum([p['actual_pnl_usd'] for p in sorted_pairs])
    peak   = np.maximum.accumulate(cum)
    max_dd = float((cum - peak).min())

    loss_rate       = 1 - win_rate / 100
    expectancy      = (win_rate / 100 * avg_win_usd) + (loss_rate * avg_loss_usd)
    recovery_factor = total_pnl / abs(max_dd) if max_dd < 0 else None

    n_strategy   = sum(1 for p in pairs if (p.get('entry_type') or 'Strategy') == 'Strategy')
    n_disc       = sum(1 for p in pairs if (p.get('entry_type') or 'Strategy') == 'Discretionary')
    n_strat_exit = sum(1 for p in pairs if _normalize_exit_reason(p.get('exit_reason')) == 'Strategy')
    n_disc_exit  = sum(1 for p in pairs if _normalize_exit_reason(p.get('exit_reason')) == 'Discretionary')

    # ── Time in market ────────────────────────────────────────────────────────
    # Union of all [entry_date, exit_date] intervals, divided by total date range.
    time_in_market_pct = None
    _intervals = [
        (p['entry_date'], p['exit_date'])
        for p in pairs
        if p['entry_date'] and p['exit_date']
    ]
    if _intervals:
        _min_date = min(s for s, _ in _intervals)
        _max_date = max(e for _, e in _intervals)
        _total_calendar_days = (_max_date - _min_date).days
        if _total_calendar_days > 0:
            # Merge overlapping intervals to count unique calendar days in market
            _merged = sorted(_intervals)
            _unique_days = 0
            _cur_s = _cur_e = None
            for _s, _e in _merged:
                if _cur_s is None:
                    _cur_s, _cur_e = _s, _e
                elif _s <= _cur_e:
                    _cur_e = max(_cur_e, _e)
                else:
                    _unique_days += (_cur_e - _cur_s).days
                    _cur_s, _cur_e = _s, _e
            if _cur_s is not None:
                _unique_days += (_cur_e - _cur_s).days
            time_in_market_pct = _unique_days / _total_calendar_days * 100

    return {
        'n': n, 'total_pnl': total_pnl, 'win_rate': win_rate,
        'profit_factor': profit_factor, 'avg_ret_pct': avg_ret_pct,
        'avg_win_pct': avg_win_pct, 'avg_loss_pct': avg_loss_pct,
        'avg_win_usd': avg_win_usd, 'avg_loss_usd': avg_loss_usd,
        'largest_win': largest_win, 'largest_loss': largest_loss,
        'max_win_streak': max_win_streak, 'max_loss_streak': max_loss_streak,
        'cur_result': cur_result, 'cur_streak': cur_streak,
        'expectancy': expectancy, 'max_dd': max_dd,
        'recovery_factor': recovery_factor,
        'avg_days_all':    sum(days) / n if days else 0,
        'avg_days_wins':   sum(p['holding_days'] for p in wins)   / len(wins)   if wins   else 0,
        'avg_days_losses': sum(p['holding_days'] for p in losses) / len(losses) if losses else 0,
        'avg_mae_all':    sum(maes) / len(maes) if maes else None,
        'avg_mae_wins':   (sum(p['mae_pct'] for p in wins   if p['mae_pct'] is not None) /
                           len([w for w in wins   if w['mae_pct'] is not None]))
                          if any(w['mae_pct'] is not None for w in wins)   else None,
        'avg_mae_losses': (sum(p['mae_pct'] for p in losses if p['mae_pct'] is not None) /
                           len([l for l in losses if l['mae_pct'] is not None]))
                          if any(l['mae_pct'] is not None for l in losses) else None,
        'n_strategy': n_strategy, 'n_disc': n_disc,
        'n_strat_exit': n_strat_exit, 'n_disc_exit': n_disc_exit,
        'time_in_market_pct': time_in_market_pct,
    }


def _compute_exec_stats(pairs: list) -> dict | None:
    n = len(pairs)
    if n == 0:
        return None

    entry_slips    = [p['entry_slippage_pct'] for p in pairs]
    exit_slips     = [p['exit_slippage_pct']  for p in pairs]
    avg_entry_slip = sum(entry_slips) / n
    avg_exit_slip  = sum(exit_slips)  / n
    total_slip_cost = sum(
        abs(p['entry_slippage_pct']) * (p.get('size_usd') or 0) / 100 +
        abs(p['exit_slippage_pct'])  * (p.get('size_usd') or 0) / 100
        for p in pairs
    )
    best  = min(pairs, key=lambda p: p['entry_slippage_pct'])
    worst = max(pairs, key=lambda p: p['entry_slippage_pct'])
    strat_pairs   = [p for p in pairs if _normalize_exit_reason(p.get('exit_reason')) == 'Strategy']
    disc_pairs    = [p for p in pairs if _normalize_exit_reason(p.get('exit_reason')) == 'Discretionary']
    avg_pnl_strat = sum(p['actual_pnl_usd'] for p in strat_pairs) / len(strat_pairs) if strat_pairs else None
    avg_pnl_disc  = sum(p['actual_pnl_usd'] for p in disc_pairs)  / len(disc_pairs)  if disc_pairs  else None

    return {
        'avg_entry_slip': avg_entry_slip, 'avg_exit_slip': avg_exit_slip,
        'total_slip_cost': total_slip_cost, 'best': best, 'worst': worst,
        'avg_pnl_strat': avg_pnl_strat, 'avg_pnl_disc': avg_pnl_disc,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HTML RENDERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _pct(v, dp=2):
    return f"{v:+.{dp}f}%" if v is not None else "—"


def _usd(v, dp=2):
    if v is None:
        return "—"
    sign = "+" if v > 0 else ""
    return f"${sign}{v:,.{dp}f}"


def _stat_row(label, value_html, cls=""):
    val_cls = f"stat-value {cls}".strip() if cls else "stat-value"
    return (f'<tr><td class="stat-label">{label}</td>'
            f'<td class="{val_cls}">{value_html}</td></tr>')


def _render_stats_html(s: dict, n_total: int, skip_summary: bool = False) -> str:
    if s is None:
        return '<p class="no-data-msg">No closed trades yet.</p>'
    pf_val     = s['profit_factor']
    pf_str     = "∞" if pf_val == float('inf') else f"{pf_val:.3f}"
    pf_cls     = 'stat-pos' if pf_val >= 1 else 'stat-neg'
    dd_str     = f"${s['max_dd']:,.3f}" if s['max_dd'] < 0 else "—"
    rf_val     = s['recovery_factor']
    rf_str     = f"{rf_val:.3f}" if rf_val is not None else "—"
    rf_cls     = ('stat-pos' if rf_val is not None and rf_val >= 1
                  else 'stat-neg' if rf_val is not None else '')
    streak_cls = 'stat-pos' if s['cur_result'] == 'W' else 'stat-neg'
    streak_str = (f"<span class='{streak_cls}'>{s['cur_streak']} "
                  f"{'win' if s['cur_result']=='W' else 'loss'}"
                  f"{'s' if s['cur_streak'] > 1 else ''}</span>")
    lw = s['largest_win'];  ll = s['largest_loss']
    lw_str = f"${lw['actual_pnl_usd']:,.0f} ({lw['symbol']}, {lw['exit_date']})" if lw else "—"
    ll_str = f"${ll['actual_pnl_usd']:,.0f} ({ll['symbol']}, {ll['exit_date']})" if ll else "—"
    wr_cls = 'stat-pos' if s['win_rate'] >= 50 else 'stat-neg'
    ar_cls = 'stat-pos' if s['avg_ret_pct'] >= 0 else 'stat-neg'
    ex_cls = 'stat-pos' if s['expectancy'] >= 0 else 'stat-neg'
    summary_rows = [] if skip_summary else [
        _stat_row("Total trades",           f"{n_total}"),
        _stat_row("Win rate",               f"{s['win_rate']:.1f}%",          wr_cls),
        _stat_row("Profit factor",          pf_str,                            pf_cls),
        _stat_row("Total P&L",              _usd(s['total_pnl']),
                  'stat-pos' if s['total_pnl'] >= 0 else 'stat-neg'),
    ]
    rows = "".join(summary_rows + [
        _stat_row("Avg return / trade",     _pct(s['avg_ret_pct']),            ar_cls),
        _stat_row("Avg win",                _pct(s['avg_win_pct']),            'stat-pos'),
        _stat_row("Avg loss",               _pct(s['avg_loss_pct']),           'stat-neg'),
        _stat_row("Largest win",            lw_str,                            'stat-pos'),
        _stat_row("Largest loss",           ll_str,                            'stat-neg'),
        _stat_row("Max consecutive wins",   str(s['max_win_streak']),          'stat-pos'),
        _stat_row("Max consecutive losses", str(s['max_loss_streak']),         'stat-neg'),
        _stat_row("Current streak",         streak_str),
        _stat_row("Expectancy / trade",     _usd(s['expectancy']),             ex_cls),
        _stat_row("Max drawdown (P&amp;L)", dd_str,
                  'stat-neg' if s['max_dd'] < 0 else ''),
        _stat_row("Recovery factor",        rf_str,                            rf_cls),
    ])
    return f'<table class="stats-table">{rows}</table>'


def _render_hold_html(s: dict) -> str:
    if s is None:
        return ''
    mae_all = f"{s['avg_mae_all']:.3f}%"    if s['avg_mae_all']    is not None else "—"
    mae_w   = f"{s['avg_mae_wins']:.3f}%"   if s['avg_mae_wins']   is not None else "—"
    mae_l   = f"{s['avg_mae_losses']:.3f}%" if s['avg_mae_losses'] is not None else "—"
    mae_all_cls = 'stat-neg' if s['avg_mae_all']    is not None else ''
    mae_w_cls   = 'stat-neg' if s['avg_mae_wins']   is not None else ''
    mae_l_cls   = 'stat-neg' if s['avg_mae_losses'] is not None else ''
    tim = s.get('time_in_market_pct')
    tim_str = f"{tim:.1f}%" if tim is not None else "—"
    n   = s['n']
    pct = lambda v: f"{v / n * 100:.0f}%" if n > 0 else "—"
    rows = "".join([
        _stat_row("Avg holding days (all)",    f"{s['avg_days_all']:.1f}"),
        _stat_row("Avg holding days (wins)",   f"{s['avg_days_wins']:.1f}"),
        _stat_row("Avg holding days (losses)", f"{s['avg_days_losses']:.1f}"),
        _stat_row("Time in market",            tim_str),
        _stat_row("Avg MAE (all)",             mae_all,  mae_all_cls),
        _stat_row("Avg MAE (wins)",            mae_w,    mae_w_cls),
        _stat_row("Avg MAE (losses)",          mae_l,    mae_l_cls),
        _stat_row("Strategy entries",          f"{s['n_strategy']} ({pct(s['n_strategy'])})"),
        _stat_row("Discretionary entries",     f"{s['n_disc']} ({pct(s['n_disc'])})"),
        _stat_row("Strategy exits",            f"{s['n_strat_exit']} ({pct(s['n_strat_exit'])})"),
        _stat_row("Discretionary exits",       f"{s['n_disc_exit']} ({pct(s['n_disc_exit'])})"),
    ])
    return f'<table class="stats-table">{rows}</table>'


def _render_exec_html(e: dict, pairs: list) -> str:
    if e is None:
        return '<p class="no-data-msg">No closed trades yet.</p>'
    b = e['best'];  w = e['worst']
    # Slippage: negative = favourable (bought below close), positive = adverse
    es_cls = 'stat-pos' if e['avg_entry_slip'] <= 0 else 'stat-neg'
    xs_cls = 'stat-pos' if e['avg_exit_slip']  <= 0 else 'stat-neg'
    ps_cls = ('stat-pos' if e['avg_pnl_strat'] is not None and e['avg_pnl_strat'] >= 0
              else 'stat-neg' if e['avg_pnl_strat'] is not None else '')
    pd_cls = ('stat-pos' if e['avg_pnl_disc']  is not None and e['avg_pnl_disc']  >= 0
              else 'stat-neg' if e['avg_pnl_disc']  is not None else '')
    rows = "".join([
        _stat_row("Avg entry slippage",          f"{e['avg_entry_slip']:+.3f}%",  es_cls),
        _stat_row("Avg exit slippage",           f"{e['avg_exit_slip']:+.3f}%",   xs_cls),
        _stat_row("Total slippage cost",         f"${e['total_slip_cost']:,.3f}", 'stat-neg'),
        _stat_row("Best execution trade",
                  f"{b['entry_slippage_pct']:+.3f}% ({b['symbol']}, {b['entry_date']})" if b else "—",
                  'stat-pos'),
        _stat_row("Worst execution trade",
                  f"{w['entry_slippage_pct']:+.3f}% ({w['symbol']}, {w['entry_date']})" if w else "—",
                  'stat-neg'),
        _stat_row("Avg P&amp;L — strategy exit",
                  _usd(e['avg_pnl_strat']) if e['avg_pnl_strat'] is not None else "—", ps_cls),
        _stat_row("Avg P&amp;L — discretionary",
                  _usd(e['avg_pnl_disc'])  if e['avg_pnl_disc']  is not None else "—", pd_cls),
    ])
    return f'<table class="stats-table">{rows}</table>'


# ══════════════════════════════════════════════════════════════════════════════
#  PANDAS STYLER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _style_return(val):
    try:
        v = float(val)
        if v > 0: return "color: #3B6D11; font-weight: 600"
        if v < 0: return "color: #A32D2D; font-weight: 600"
    except Exception:
        pass
    return ""


def _style_mae(val):
    try:
        v = float(val)
        if v < -10: return "color: #A32D2D; font-weight: 600"
        if v < -5:  return "color: #854F0B; font-weight: 600"
    except Exception:
        pass
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  FILTER WIDGET RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _render_filters_and_compute(tab_key: str, all_trades: list, all_pairs: dict):
    """
    Render date-range selector and coin multiselect.
    Returns (coin_sel, filt_trades, filt_closed, filt_open).
    ``tab_key`` is prepended to every widget key.
    """
    today     = date.today()
    ytd_start = date(today.year, 1, 1)
    all_syms  = sorted({_pid_to_symbol(t['position_id']) for t in all_trades
                        if t['position_id']})

    ff1, ff2 = st.columns([2, 2])
    with ff1:
        date_sel = st.selectbox("Date range", _DATE_OPTIONS, index=0,
                                key=f"{tab_key}_date_sel")
        if date_sel == "Last 30 days":
            f_start, f_end = today - timedelta(days=30), today
        elif date_sel == "Last 90 days":
            f_start, f_end = today - timedelta(days=90), today
        elif date_sel == "Year to date":
            f_start, f_end = ytd_start, today
        elif date_sel == "Custom":
            c1, c2 = st.columns(2)
            f_start = c1.date_input("From", value=today - timedelta(days=90),
                                    key=f"{tab_key}_from")
            f_end   = c2.date_input("To",   value=today, key=f"{tab_key}_to")
        else:  # All time
            _all_dates = [t['date'] for t in all_trades if t['date']]
            f_start = min(_all_dates) if _all_dates else today - timedelta(days=365)
            f_end   = today
    with ff2:
        sel = st.multiselect("Coin filter", options=all_syms, default=[],
                             placeholder="All coins", key=f"{tab_key}_coins")
    coin_sel = sel if sel else all_syms

    def _in(d):
        return d is None or f_start <= d <= f_end

    filt_trades = [t for t in all_trades
                   if _pid_to_symbol(t['position_id']) in coin_sel and _in(t['date'])]
    filt_closed = [p for p in all_pairs.get('closed', [])
                   if p['symbol'] in coin_sel and _in(p['entry_date'])]
    filt_open   = [t for t in all_pairs.get('open', [])
                   if _pid_to_symbol(t['position_id']) in coin_sel]
    return coin_sel, filt_trades, filt_closed, filt_open


# ══════════════════════════════════════════════════════════════════════════════
#  COIN SUB-TAB RENDERER  (shared by both public functions)
# ══════════════════════════════════════════════════════════════════════════════

def _render_coin_tabs(strat_closed: list, strat_open: list,
                      coin_sel: list, tab_prefix: str, prefix: str,
                      show_strategy_col: bool, data_dir: str,
                      skip_summary_stats: bool = False):
    """
    Render 'All coins' + per-coin sub-tabs with open positions,
    trade performance table, and execution detail table.
    """
    traded_syms      = sorted({p['symbol'] for p in strat_closed} |
                               {_pid_to_symbol(t['position_id']) for t in strat_open})
    active_in_filter = [s for s in traded_syms if s in coin_sel]
    coin_tab_labels  = ["All coins"] + active_in_filter
    coin_tabs        = st.tabs(coin_tab_labels)

    for ct_idx, coin_label in enumerate(coin_tab_labels):
        with coin_tabs[ct_idx]:
            coin_key = f"{tab_prefix}_{coin_label.replace(' ', '_')}"

            if coin_label == "All coins":
                tab_closed = strat_closed
                tab_open   = strat_open
            else:
                tab_closed = [p for p in strat_closed if p['symbol'] == coin_label]
                tab_open   = [t for t in strat_open
                              if _pid_to_symbol(t['position_id']) == coin_label]

            search = st.text_input(
                "Search notes", key=f"{prefix}_search_{coin_key}",
                placeholder="Filter by note content…", label_visibility="collapsed",
            )
            if search:
                tab_closed = [p for p in tab_closed
                              if search.lower() in (p.get('discretion_note') or '').lower()]
                tab_open   = [t for t in tab_open
                              if search.lower() in (t.get('discretion_note') or '').lower()]

            if not tab_closed and not tab_open:
                st.markdown('<p class="no-data-msg">No trades match current filters.</p>',
                            unsafe_allow_html=True)
                continue

            # ── Wallet analysis (trade + holding + execution stats) ───────────
            if tab_closed:
                s = _compute_stats(tab_closed)
                e = _compute_exec_stats(tab_closed)
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("**Trade statistics**")
                    st.markdown(_render_stats_html(s, len(tab_closed),
                                                   skip_summary=skip_summary_stats),
                                unsafe_allow_html=True)
                with col_b:
                    st.markdown("**Holding & entry/exit mix**")
                    st.markdown(_render_hold_html(s), unsafe_allow_html=True)
                with col_c:
                    st.markdown("**Execution detail**")
                    st.markdown(_render_exec_html(e, tab_closed), unsafe_allow_html=True)
                st.divider()

            # ── Open positions ────────────────────────────────────────────────
            if tab_open:
                st.caption(f"{len(tab_open)} open position{'s' if len(tab_open)!=1 else ''}")
                open_rows = []
                for t in tab_open:
                    sym        = _pid_to_symbol(t['position_id'])
                    entry_date = t['date']
                    days_open  = (date.today() - entry_date).days if entry_date else None
                    etype      = (t.get('entry_type') or
                                  derive_entry_reason(t.get('signal_snapshot'),
                                                      t.get('discretion_note')))
                    _ddir      = t.get('_data_dir', data_dir)
                    open_size  = (
                        t.get('size_usd')
                        or (t.get('coin_capital') or
                            (get_coin_capital(sym, _ddir) if _ddir else 0.0))
                           * t['actual_leverage']
                    )
                    row = {
                        'Coin':        sym,
                        'Entry date':  entry_date,
                        'Days held':   days_open,
                        'Entry price': t['actual_price'],
                        'Leverage':    t['actual_leverage'],
                        'Size ($)':    open_size,
                        'Entry type':  etype,
                    }
                    if t.get('discretion_note'):
                        row['Note'] = (t['discretion_note'] or '')[:40]
                    if show_strategy_col:
                        row['Strategy'] = _STRATEGY_LABELS.get(
                            t.get('strategy', ''), t.get('strategy', '—'))
                    open_rows.append(row)

                df_open = pd.DataFrame(open_rows)
                try:
                    styled_open = df_open.style.format(
                        {'Entry price': '{:,.3f}', 'Leverage': '{:.3f}x', 'Size ($)': '{:,.0f}'},
                        na_rep='—',
                    )
                except Exception:
                    styled_open = df_open
                open_col_cfg = {'Entry date': st.column_config.DateColumn("Entry date")}
                if 'Note' in df_open.columns:
                    open_col_cfg['Note'] = st.column_config.TextColumn("Note",
                                                                        help="Full text on hover")
                st.dataframe(styled_open, use_container_width=True, hide_index=True,
                             column_config=open_col_cfg)

            # ── Closed: Table 1 — Trade performance ───────────────────────────
            if tab_closed:
                sorted_closed = sorted(tab_closed,
                                       key=lambda x: x['entry_date'] or date.today())
                perf_rows = []
                for p in sorted_closed:
                    entry_type_disp  = p.get('entry_type') or 'Strategy'
                    exit_reason_disp = _normalize_exit_reason(p.get('exit_reason'))
                    strat_label      = _STRATEGY_LABELS.get(
                        p.get('strategy', ''), p.get('strategy', '—'))
                    size_usd  = p.get('actual_size_usd', p.get('size_usd'))
                    is_legacy = p.get('legacy', False)
                    if size_usd is None:
                        size_display = '—'
                    elif is_legacy:
                        size_display = f'⚠ {size_usd:,.0f}'
                    else:
                        size_display = f'{size_usd:,.0f}'

                    row = {'Coin': p['symbol']}
                    if show_strategy_col:
                        row['Strategy'] = strat_label
                    row['Entry date']  = p['entry_date']
                    row['Exit date']   = p['exit_date']
                    row['Days']        = p['holding_days']
                    row['Size ($)']    = size_display
                    # Net return (after round-trip costs) — consistent with P&L ($)
                    row['Return %']    = p.get('actual_net_return_pct', p['actual_return_pct'])
                    row['P&L ($)']     = p['actual_pnl_usd']
                    row['MAE %']       = p.get('mae_pct')
                    row['Entry type']  = entry_type_disp
                    row['Exit reason'] = exit_reason_disp
                    perf_rows.append(row)

                st.caption(f"{len(tab_closed)} closed trade{'s' if len(tab_closed)!=1 else ''}")
                df_perf = pd.DataFrame(perf_rows)
                try:
                    fmt_perf: dict = {'Return %': '{:+.3f}', 'P&L ($)': '{:+,.3f}'}
                    if 'MAE %' in df_perf.columns:
                        fmt_perf['MAE %'] = lambda v: f'{v:.3f}' if pd.notna(v) else '—'
                    style_perf = df_perf.style.applymap(_style_return, subset=['Return %'])
                    style_perf = style_perf.applymap(_style_return, subset=['P&L ($)'])
                    if 'MAE %' in df_perf.columns:
                        style_perf = style_perf.applymap(_style_mae, subset=['MAE %'])
                    styled_perf = style_perf.format(fmt_perf, na_rep='—')
                except Exception:
                    styled_perf = df_perf
                st.markdown("**Trade performance**")
                st.dataframe(styled_perf, use_container_width=True, hide_index=True,
                             column_config={
                                 'Entry date': st.column_config.DateColumn("Entry date"),
                                 'Exit date':  st.column_config.DateColumn("Exit date"),
                             })
                st.caption(
                    "MAE % — Maximum Adverse Excursion: the furthest the price moved "
                    "against the position from entry, expressed as a % of entry price."
                )

                # ── Closed: Table 2 — Execution detail ───────────────────────
                exec_rows = []
                for p in sorted_closed:
                    sz = p.get('actual_size_usd', p.get('size_usd')) or 0
                    strat_entry = (p['entry_close'] if p.get('entry_close') is not None
                                   else p['theoretical_entry'])
                    strat_exit  = (p['exit_close']  if p.get('exit_close')  is not None
                                   else p['theoretical_exit'])
                    e_slip_usd = p['entry_slippage_pct'] / 100 * sz if sz else None
                    x_slip_usd = p['exit_slippage_pct']  / 100 * sz if sz else None

                    exec_rows.append({
                        'Coin': p['symbol'], 'Date': p['entry_date'], 'Action': 'Entry',
                        'Actual price': p['actual_entry'], 'Strat close': strat_entry,
                        'Slip %': p['entry_slippage_pct'], 'Slip $': e_slip_usd,
                        'Leverage': p['leverage'], 'Theo Lev': p['theoretical_size_pct'],
                        'Note': (p.get('discretion_note') or '')[:60],
                    })
                    exec_rows.append({
                        'Coin': p['symbol'], 'Date': p['exit_date'], 'Action': 'Exit',
                        'Actual price': p['actual_exit'], 'Strat close': strat_exit,
                        'Slip %': p['exit_slippage_pct'], 'Slip $': x_slip_usd,
                        'Leverage': p['leverage'], 'Theo Lev': p['theoretical_size_pct'],
                        'Note': _normalize_exit_reason(p.get('exit_reason')),
                    })

                df_exec = pd.DataFrame(exec_rows)
                try:
                    styled_exec = df_exec.style.format(
                        {'Actual price': '{:,.3f}', 'Strat close': '{:,.3f}',
                         'Slip %': '{:+.3f}', 'Slip $': '{:+,.3f}',
                         'Leverage': '{:.3f}x', 'Theo Lev': '{:.3f}x'},
                        na_rep='—',
                    )
                except Exception:
                    styled_exec = df_exec
                st.markdown("**Execution detail**")
                st.dataframe(styled_exec, use_container_width=True, hide_index=True,
                             column_config={
                                 'Date': st.column_config.DateColumn("Date"),
                                 'Note': st.column_config.TextColumn("Note",
                                                                      help="Full text on hover"),
                             })


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def render_strategy_tab(
    data_dir: str,
    prefix: str,
    strategy_keys: Union[str, List[str]],
    display_name: str = "Momentum",
    show_strategy_col: bool = True,
) -> None:
    """
    Render a complete per-strategy trade journal tab.

    Parameters
    ----------
    data_dir        : absolute path to the strategy's dashboard directory
                      (e.g. live_trading/dashboards/momentum)
    prefix          : unique string prepended to all Streamlit widget keys
    strategy_keys   : one or more strategy identifiers that match the
                      ``strategy`` field on each trade record
                      (e.g. ["momentum_swing", "momentum_no_vol"])
    display_name    : human-readable name shown in empty-state messages
    show_strategy_col : whether to include a Strategy column in the tables
    """
    if isinstance(strategy_keys, str):
        strategy_keys = [strategy_keys]

    all_trades, all_pairs = _load_single(data_dir)

    cfg      = load_config(data_dir)
    cost_pct = float(cfg.get('trading_cost_pct', 0.0))

    # Reserve header slot — filled after filters so metrics reflect the filter
    _hdr = st.container()

    coin_sel, _, filt_closed, filt_open = _render_filters_and_compute(
        f"{prefix}_strat", all_trades, all_pairs
    )

    strat_closed = [p for p in filt_closed if (p.get('strategy') or '') in strategy_keys]
    strat_open   = [t for t in filt_open   if (t.get('strategy') or '') in strategy_keys]

    if not strat_closed and not strat_open:
        st.markdown(
            f'<p class="no-data-msg">No {display_name} trades in the current filter window.</p>',
            unsafe_allow_html=True,
        )
        return

    # ── Strategy-level header metrics (rendered into reserved slot above filters)
    _total_rpnl  = sum(p['actual_pnl_usd'] for p in strat_closed)
    _wins        = [p for p in strat_closed if p['actual_pnl_usd'] > 0]
    _losses      = [p for p in strat_closed if p['actual_pnl_usd'] <= 0]
    _win_rate    = len(_wins) / len(strat_closed) * 100 if strat_closed else None
    _sum_wins    = sum(p['actual_pnl_usd'] for p in _wins)
    _sum_losses  = abs(sum(p['actual_pnl_usd'] for p in _losses)) if _losses else 0
    _pf          = _sum_wins / _sum_losses if _sum_losses > 0 else None
    _avg_win     = _sum_wins / len(_wins) if _wins else None
    _avg_loss    = (sum(p['actual_pnl_usd'] for p in _losses) / len(_losses)
                    if _losses else None)
    _wl          = (_avg_win / abs(_avg_loss)
                    if _avg_win is not None and _avg_loss is not None and _avg_loss != 0
                    else None)
    _pnl_sign    = (f"+${_total_rpnl:,.3f}" if _total_rpnl >= 0
                    else f"-${abs(_total_rpnl):,.3f}")

    with _hdr:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Closed trades",  len(strat_closed))
        m2.metric("Open positions", len(strat_open))
        m3.metric("Total P&L",      _pnl_sign)
        m4.metric("Win rate",       f"{_win_rate:.1f}%" if _win_rate is not None else "—")
        m5.metric("Profit factor",  f"{_pf:.3f}" if _pf is not None else "—")
        m6.metric("Avg win / loss", f"{_wl:.3f}x" if _wl is not None else "—")
        if cost_pct > 0:
            st.caption(
                f"P&L figures include trading costs: "
                f"**{cost_pct * 100:.3f}%** per leg · "
                f"**{cost_pct * 2 * 100:.3f}%** round-trip"
            )
        st.markdown("---")

    _render_coin_tabs(
        strat_closed=strat_closed,
        strat_open=strat_open,
        coin_sel=coin_sel,
        tab_prefix=f"{prefix}_strat",
        prefix=prefix,
        show_strategy_col=show_strategy_col,
        data_dir=data_dir,
        skip_summary_stats=True,
    )


def render_fund_tab(
    dashboard_dirs: Dict[str, str],
    prefix: str,
) -> None:
    """
    Render the aggregated cross-strategy fund log tab.

    Parameters
    ----------
    dashboard_dirs : mapping of display-name -> absolute data_dir path,
                     e.g. {"Momentum": ".../dashboards/momentum"}
    prefix         : unique string prepended to all Streamlit widget keys
    """
    # _load_all requires a hashable arg
    dirs_tuple = tuple(sorted(dashboard_dirs.items()))
    all_trades, all_pairs = _load_all(dirs_tuple)

    # Collect trading cost config per strategy (for the cost note)
    _cost_pcts = {
        name: float(load_config(data_dir).get('trading_cost_pct', 0.0))
        for name, data_dir in dirs_tuple
    }

    # Reserve header slot so metrics float above the filter widgets
    _hdr = st.container()
    coin_sel, filt_trades, filt_closed, filt_open = _render_filters_and_compute(
        f"{prefix}_fund", all_trades, all_pairs
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_rpnl  = sum(p['actual_pnl_usd'] for p in filt_closed)
    _wins       = [p for p in filt_closed if p['actual_pnl_usd'] > 0]
    _losses     = [p for p in filt_closed if p['actual_pnl_usd'] <= 0]
    _win_rate   = len(_wins) / len(filt_closed) * 100 if filt_closed else None
    _sum_wins   = sum(p['actual_pnl_usd'] for p in _wins)
    _sum_losses = abs(sum(p['actual_pnl_usd'] for p in _losses)) if _losses else 0
    _pf         = _sum_wins / _sum_losses if _sum_losses > 0 else None
    _avg_win    = _sum_wins / len(_wins) if _wins else None
    _avg_loss   = (sum(p['actual_pnl_usd'] for p in _losses) / len(_losses)
                   if _losses else None)

    with _hdr:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total trades",   len(filt_trades))
        m2.metric("Open positions", len(filt_open))
        delta_sign = (f"+${total_rpnl:,.3f}" if total_rpnl >= 0
                      else f"-${abs(total_rpnl):,.3f}")
        m3.metric("Total P&L",      delta_sign)
        m4.metric("Win rate",       f"{_win_rate:.1f}%" if _win_rate is not None else "—")
        m5.metric("Profit factor",  f"{_pf:.3f}" if _pf is not None else "—")
        _wl = (_avg_win / abs(_avg_loss)
               if _avg_win is not None and _avg_loss is not None and _avg_loss != 0
               else None)
        m6.metric("Avg win / loss", f"{_wl:.3f}x" if _wl is not None else "—")
        _nonzero = {n: v for n, v in _cost_pcts.items() if v > 0}
        if _nonzero:
            _unique = set(_nonzero.values())
            if len(_unique) == 1:
                _c = next(iter(_unique))
                st.caption(
                    f"P&L figures include trading costs: "
                    f"**{_c * 100:.3f}%** per leg · "
                    f"**{_c * 2 * 100:.3f}%** round-trip"
                )
            else:
                _parts = [f"{n}: {v * 100:.3f}%" for n, v in _nonzero.items()]
                st.caption(
                    "P&L includes per-leg trading costs — " + ", ".join(_parts)
                )
        st.markdown("---")

    # ── Fund analysis ─────────────────────────────────────────────────────────
    # Full stats across all wallets for the current date-range + coin filter.
    if filt_closed:
        st.markdown("#### Fund analysis")
        s_fund = _compute_stats(filt_closed)
        e_fund = _compute_exec_stats(filt_closed)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Trade statistics**")
            st.markdown(_render_stats_html(s_fund, len(filt_closed), skip_summary=True),
                        unsafe_allow_html=True)
        with col_b:
            st.markdown("**Holding & entry/exit mix**")
            st.markdown(_render_hold_html(s_fund), unsafe_allow_html=True)
        with col_c:
            st.markdown("**Execution detail**")
            st.markdown(_render_exec_html(e_fund, filt_closed), unsafe_allow_html=True)
        st.divider()

    # ── Flat log table ────────────────────────────────────────────────────────
    sorted_closed = sorted(filt_closed, key=lambda x: x['entry_date'] or date.today())
    fund_rows = []
    for p in sorted_closed:
        entry_type_disp  = p.get('entry_type') or 'Strategy'
        exit_reason_disp = _normalize_exit_reason(p.get('exit_reason'))
        strat_label      = _STRATEGY_LABELS.get(p.get('strategy', ''), p.get('strategy', '—'))
        size_usd  = p.get('actual_size_usd', p.get('size_usd'))
        is_legacy = p.get('legacy', False)
        if size_usd is None:
            size_display = '—'
        elif is_legacy:
            size_display = f'⚠ {size_usd:,.0f}'
        else:
            size_display = f'{size_usd:,.0f}'

        fund_rows.append({
            'Wallet':      p.get('_dashboard', '—'),
            'Coin':        p['symbol'],
            'Strategy':    strat_label,
            'Entry date':  p['entry_date'],
            'Exit date':   p['exit_date'],
            'Days':        p['holding_days'],
            'Size ($)':    size_display,
            # Net return (after round-trip costs) — consistent with P&L ($)
            'Return %':    p.get('actual_net_return_pct', p['actual_return_pct']),
            'P&L ($)':     p['actual_pnl_usd'],
            'Entry type':  entry_type_disp,
            'Exit reason': exit_reason_disp,
        })

    if not fund_rows:
        st.markdown('<p class="no-data-msg">No closed trades match current filters.</p>',
                    unsafe_allow_html=True)
        return

    df_fund = pd.DataFrame(fund_rows)
    try:
        fmt_fund   = {'Return %': '{:+.3f}', 'P&L ($)': '{:+,.3f}'}
        style_fund = df_fund.style.applymap(_style_return, subset=['Return %'])
        style_fund = style_fund.applymap(_style_return, subset=['P&L ($)'])
        styled_fund = style_fund.format(fmt_fund, na_rep='—')
    except Exception:
        styled_fund = df_fund

    st.dataframe(
        styled_fund,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Entry date': st.column_config.DateColumn("Entry date"),
            'Exit date':  st.column_config.DateColumn("Exit date"),
        },
    )
