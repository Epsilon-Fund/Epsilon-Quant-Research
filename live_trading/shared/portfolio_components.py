"""
Shared portfolio rendering components.

Public API
----------
render_strategy_portfolio(data_dir, prefix, strategy_name)
    Renders the full portfolio view for a single strategy directory:
    header, warning, Overview tab (4 metrics + 3 charts + toggles),
    Per coin tab stub.

render_fund_portfolio(dashboard_dirs, prefix)
    Stub — cross-strategy fund portfolio (not yet built).

All widget keys are namespaced by `prefix` to prevent DuplicateWidgetID
errors when multiple callers render in the same Streamlit session.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# ── Ensure shared/ siblings are importable ────────────────────────────────────
_SHARED_DIR = os.path.dirname(os.path.abspath(__file__))
_LT_DIR     = os.path.dirname(_SHARED_DIR)
for _p in (_SHARED_DIR, _LT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.data_loader import (
    build_equity_curve,
    build_capital_deployment,
    build_trade_pairs,
    load_config,
)
from shared.charts import (
    equity_chart,
    drawdown_chart,
    capital_deployment_chart,
)
from shared.styles import apply_styles

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_TRADES = 30


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner="Loading portfolio data…")
def _load_portfolio(data_dir: str):
    """Load all data needed for a single strategy portfolio page."""
    curve      = build_equity_curve(data_dir)
    deployment = build_capital_deployment(data_dir)
    pairs      = build_trade_pairs(data_dir)
    cfg        = load_config(data_dir)
    return curve, deployment, pairs, cfg


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(curve_df: pd.DataFrame, window_days: int, capital: float) -> dict:
    """
    Compute portfolio statistics from a daily equity curve DataFrame.

    Parameters
    ----------
    curve_df    : output of build_equity_curve()
    window_days : rolling-Sharpe window in trading days
    capital     : initial capital in USD (from config)

    Returns a dict with all keys set to None (except initial_capital)
    when curve_df has fewer than 5 rows.
    """
    empty = {
        'total_return_pct':             None,
        'theoretical_total_return_pct': None,
        'sharpe':                       None,
        'theoretical_sharpe':           None,
        'rolling_sharpe':               None,
        'max_drawdown_pct':             None,
        'theoretical_max_drawdown_pct': None,
        'initial_capital':              capital,
    }

    if curve_df is None or curve_df.empty or len(curve_df) < 5:
        return empty

    actual_cum = curve_df['actual_cumulative']
    theo_cum   = curve_df['theoretical_cumulative']
    actual_pnl = curve_df['actual_pnl']
    theo_pnl   = curve_df['theoretical_pnl']

    # ── Total return (vs initial capital) ────────────────────────────────────
    total_return_pct      = actual_cum.iloc[-1] / capital * 100 if capital else None
    theo_total_return_pct = theo_cum.iloc[-1]   / capital * 100 if capital else None

    # ── Annualised Sharpe (includes zero days — conservative) ────────────────
    def _sharpe(pnl: pd.Series):
        std = pnl.std()
        if std == 0 or math.isnan(std):
            return None
        return float(pnl.mean() / std * math.sqrt(252))

    sharpe      = _sharpe(actual_pnl)
    theo_sharpe = _sharpe(theo_pnl)

    # ── Rolling Sharpe ────────────────────────────────────────────────────────
    roll_mean = actual_pnl.rolling(window_days).mean()
    roll_std  = actual_pnl.rolling(window_days).std()
    rs_values = np.where(roll_std != 0,
                         roll_mean / roll_std * math.sqrt(252),
                         np.nan)
    rolling_sharpe = pd.Series(rs_values, index=curve_df.index)

    # ── Max drawdown % ────────────────────────────────────────────────────────
    def _max_drawdown(cum: pd.Series) -> float:
        peak = cum.cummax()
        dd   = np.where(peak != 0, (cum - peak) / peak.abs() * 100, 0.0)
        return float(np.min(dd))

    max_dd      = _max_drawdown(actual_cum)
    theo_max_dd = _max_drawdown(theo_cum)

    return {
        'total_return_pct':             total_return_pct,
        'theoretical_total_return_pct': theo_total_return_pct,
        'sharpe':                       sharpe,
        'theoretical_sharpe':           theo_sharpe,
        'rolling_sharpe':               rolling_sharpe,
        'max_drawdown_pct':             max_dd,
        'theoretical_max_drawdown_pct': theo_max_dd,
        'initial_capital':              capital,
    }


# ── Formatting helper ─────────────────────────────────────────────────────────

def _fmt(value, fmt_str: str, suffix: str = '') -> str:
    """Return formatted value string, or em dash when value is None."""
    if value is None:
        return '—'
    return f'{value:{fmt_str}}{suffix}'


# ── Public API ────────────────────────────────────────────────────────────────

def render_strategy_portfolio(
    data_dir:      str,
    prefix:        str,
    strategy_name: str = "Momentum Swing",
) -> None:
    """
    Render the full portfolio view for a single strategy.

    Parameters
    ----------
    data_dir      : absolute path to the strategy's dashboard directory
    prefix        : unique string prepended to all widget keys
    strategy_name : display name shown in the page header
    """
    apply_styles()

    curve, deployment, pairs, cfg = _load_portfolio(data_dir)

    capital      = float(cfg.get('capital', 0))
    closed_count = len(pairs.get('closed', []))

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
<h1 style="font-size:35px;font-weight:700;letter-spacing:-0.01em;margin-bottom:10px">
  Portfolio
</h1>
<div class="dash-meta">
  <strong>{strategy_name}</strong> &nbsp;·&nbsp;
  Capital: <strong>${capital:,.0f}</strong> &nbsp;·&nbsp;
  Allocated per coin: shown in per-coin tab
</div>
""", unsafe_allow_html=True)

    # ── Minimum trade count warning ───────────────────────────────────────────
    if closed_count < MIN_TRADES:
        st.warning(
            f"⚠ {closed_count} closed trades — statistics require "
            f"{MIN_TRADES}+ for meaningful results. "
            f"Sharpe and rolling Sharpe may be unreliable."
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_overview, tab_coins = st.tabs(["Overview", "Per coin"])

    # ── Overview tab ──────────────────────────────────────────────────────────
    with tab_overview:

        # Read window from session_state before computing metrics so the first
        # render uses the correct window (subsequent renders pick up user choice).
        _window  = st.session_state.get(f'{prefix}_rolling_window', 30)
        metrics  = compute_metrics(curve, window_days=_window, capital=capital)

        col1, col2, col3, col4 = st.columns(4)

        # col1 — Total return
        with col1:
            act  = metrics['total_return_pct']
            theo = metrics['theoretical_total_return_pct']
            delta = (
                f"{act - theo:+.2f}% vs theoretical"
                if act is not None and theo is not None
                else None
            )
            st.metric("Total Return (Actual)", _fmt(act, '.2f', '%'), delta=delta)
            if theo is not None:
                st.caption(f"Theoretical: {theo:.2f}%")

        # col2 — Sharpe ratio
        with col2:
            sharpe      = metrics['sharpe']
            theo_sharpe = metrics['theoretical_sharpe']
            st.metric("Sharpe Ratio (Actual)", _fmt(sharpe, '.2f'))
            if theo_sharpe is not None:
                st.caption(f"Theoretical: {theo_sharpe:.2f}")
            st.caption(f"Annualised · {closed_count} closed trades")

        # col3 — Rolling Sharpe + window selector
        with col3:
            rs_series = metrics['rolling_sharpe']
            if rs_series is not None:
                valid     = rs_series.dropna()
                rs_latest = float(valid.iloc[-1]) if not valid.empty else None
            else:
                rs_latest = None

            st.metric("Rolling Sharpe (Actual)", _fmt(rs_latest, '.2f'))
            window = st.selectbox(
                "Window (days)", [30, 60, 90],
                key=f'{prefix}_rolling_window',
            )
            st.caption(f"Rolling {window}d")

        # col4 — Max drawdown
        with col4:
            dd      = metrics['max_drawdown_pct']
            theo_dd = metrics['theoretical_max_drawdown_pct']
            st.metric("Max Drawdown (Actual)", _fmt(dd, '.2f', '%'))
            if theo_dd is not None:
                st.caption(f"Theoretical: {theo_dd:.2f}%")

        st.divider()

        # ── Chart toggles ─────────────────────────────────────────────────────
        col_t1, col_t2, col_t3 = st.columns([2, 2, 2])

        with col_t1:
            show_theoretical = st.checkbox(
                "Show theoretical",
                value=True,
                key=f'{prefix}_show_theoretical_overview',
            )

        with col_t2:
            show_execution = st.checkbox(
                "Show execution hour",
                value=False,
                key=f'{prefix}_show_execution_overview',
            )
            if show_execution:
                st.caption("Fetches hourly Binance data — may be slow")

        # col_t3 reserved for future toggles

        # ── Equity chart ──────────────────────────────────────────────────────
        fig_equity = equity_chart(
            curve,
            show_theoretical=show_theoretical,
            show_execution_hour=show_execution,
            title=f"Cumulative P&L — {strategy_name}",
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # ── Drawdown chart ────────────────────────────────────────────────────
        st.markdown("#### Drawdown from peak")
        fig_dd = drawdown_chart(curve, title="")
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Capital deployment chart ──────────────────────────────────────────
        st.markdown("#### Capital deployment")
        fig_dep = capital_deployment_chart(deployment, title="")
        st.plotly_chart(fig_dep, use_container_width=True)

    # ── Per coin tab ──────────────────────────────────────────────────────────
    with tab_coins:
        st.info("Per coin charts — coming in next build step")


def render_fund_portfolio(dashboard_dirs: dict, prefix: str) -> None:
    """
    Stub: cross-strategy fund portfolio view.

    Will aggregate equity curves across all strategies once each strategy
    has sufficient trade history.
    """
    st.info("Fund portfolio — coming soon")
