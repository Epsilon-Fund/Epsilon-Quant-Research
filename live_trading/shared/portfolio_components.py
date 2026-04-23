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
from datetime import datetime

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
    build_coin_equity_curves,
    build_trade_pairs,
    load_config,
)
from shared.charts import (
    equity_chart,
    drawdown_chart,
    capital_deployment_chart,
    coin_equity_chart,
    fund_equity_chart,
    correlation_heatmap,
)
from shared.styles import apply_styles

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_TRADES = 30


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner="Loading portfolio data…")
def _load_portfolio(data_dir: str):
    """Load all data needed for a single strategy portfolio page."""
    curve       = build_equity_curve(data_dir)
    deployment  = build_capital_deployment(data_dir)
    pairs       = build_trade_pairs(data_dir)
    cfg         = load_config(data_dir)
    coin_curves = build_coin_equity_curves(data_dir)
    return curve, deployment, pairs, cfg, coin_curves


@st.cache_data(ttl=120, show_spinner="Loading fund data…")
def _load_fund_data(dashboard_dirs_tuple: tuple):
    """
    Load equity curves, configs, deployment, and trade pairs for every
    strategy.  Accepts a sorted tuple of (name, data_dir) pairs so the
    result is hashable and cacheable.

    Returns four dicts keyed by strategy name:
        strategy_curves      : {name -> equity DataFrame}
        strategy_cfgs        : {name -> config dict}
        strategy_deployments : {name -> deployment DataFrame}
        strategy_pairs       : {name -> {'closed': [...], 'open': [...]}}
    """
    strategy_curves      = {}
    strategy_cfgs        = {}
    strategy_deployments = {}
    strategy_pairs       = {}

    for name, data_dir in dashboard_dirs_tuple:
        strategy_curves[name]      = build_equity_curve(data_dir)
        strategy_cfgs[name]        = load_config(data_dir)
        strategy_deployments[name] = build_capital_deployment(data_dir)
        strategy_pairs[name]       = build_trade_pairs(data_dir)

    return strategy_curves, strategy_cfgs, strategy_deployments, strategy_pairs


@st.cache_data(ttl=3600, show_spinner="Loading BTC benchmark from cache…")
def _fetch_btc_series(start_date: str, end_date: str) -> pd.Series:
    """
    Return daily BTC/USDT closing prices between start_date and end_date
    (both as 'YYYY-MM-DD' strings).

    Reads from the local parquet cache (live_trading/cache/daily/).
    Falls back to a live Binance fetch on cache miss.
    Returns an empty Series on any error.
    """
    try:
        from shared.cache_manager import get_daily_ohlcv_range  # noqa: PLC0415
        df = get_daily_ohlcv_range('BTCUSDT', start_date, end_date)
        if df is None or df.empty:
            raise ValueError("No BTC data in cache for requested range")
        return df['Close'].astype(float)
    except Exception as exc:
        st.warning(f"Could not load BTC benchmark: {exc}")
        return pd.Series(dtype=float)


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
        'annualised_return_pct':        None,
        'calmar':                       None,
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
    # The equity curve is cumulative P&L (starts at 0), not portfolio value.
    # peak_at_mdd is therefore a small P&L number, not the full portfolio value.
    # Dividing by it produces nonsensical percentages (e.g. -175%).
    # Correct denominator: capital + peak_at_mdd = portfolio value at the peak.
    def _max_drawdown(cum: pd.Series, base_capital: float) -> tuple[float, float, object]:
        """Return (max_dd_pct, max_dd_usd, occurred_at_index)."""
        peak            = cum.cummax()
        drawdown        = cum - peak           # always <= 0, in dollars
        max_dd_usd      = float(drawdown.min())
        max_dd_idx      = drawdown.idxmin()
        peak_at_mdd     = float(peak.loc[max_dd_idx])
        portfolio_peak  = base_capital + peak_at_mdd   # portfolio value at high-water mark
        if portfolio_peak <= 0:
            return 0.0, max_dd_usd, max_dd_idx
        return max_dd_usd / portfolio_peak * 100, max_dd_usd, max_dd_idx

    max_dd_pct, max_dd_usd_val, max_dd_idx_val = _max_drawdown(actual_cum, capital)
    theo_max_dd_pct, _, _                       = _max_drawdown(theo_cum, capital)
    max_dd      = max_dd_pct
    theo_max_dd = theo_max_dd_pct

    print(f"Max drawdown:     {max_dd_pct:.2f}%")
    print(f"Max drawdown USD: {max_dd_usd_val:.2f}")
    print(f"Occurred at:      {max_dd_idx_val}")

    # ── Calmar ratio ──────────────────────────────────────────────────────────
    # Annualise the total return over the actual number of calendar days active.
    # Calmar = annualised_return_pct / |max_drawdown_pct|.
    days_active = (curve_df['date'].iloc[-1] - curve_df['date'].iloc[0]).days
    if total_return_pct is not None and days_active > 0:
        annualised_return_pct = (
            ((1 + total_return_pct / 100) ** (365.0 / days_active) - 1) * 100
        )
    else:
        annualised_return_pct = None

    calmar = None
    if annualised_return_pct is not None and max_dd not in (0, None):
        calmar = annualised_return_pct / abs(max_dd)

    return {
        'total_return_pct':             total_return_pct,
        'theoretical_total_return_pct': theo_total_return_pct,
        'sharpe':                       sharpe,
        'theoretical_sharpe':           theo_sharpe,
        'rolling_sharpe':               rolling_sharpe,
        'max_drawdown_pct':             max_dd,
        'theoretical_max_drawdown_pct': theo_max_dd,
        'annualised_return_pct':        annualised_return_pct,
        'calmar':                       calmar,
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

    curve, deployment, pairs, cfg, coin_curves = _load_portfolio(data_dir)

    capital      = float(cfg.get('capital', 0))
    closed_count = len(pairs.get('closed', []))

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
        cost_pct  = float(cfg.get('trading_cost_pct', 0.0))
        cost_note = (
            f" · Trading costs: **{cost_pct * 100:.3f}%** per leg"
            f" ({cost_pct * 2 * 100:.3f}% round-trip)"
            if cost_pct > 0 else ""
        )
        st.caption(
            f"Capital: **${capital:,.0f}** total"
            f" · per-coin allocation in Per coin tab{cost_note}"
        )

        # Read window from session_state before computing metrics so the first
        # render uses the correct window (subsequent renders pick up user choice).
        _window  = st.session_state.get(f'{prefix}_rolling_window', 30)
        metrics  = compute_metrics(curve, window_days=_window, capital=capital)

        col1, col2, col3, col4, col5, col6 = st.columns(6)

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

        # col5 — Calmar ratio
        with col5:
            calmar = metrics.get('calmar')
            ann    = metrics.get('annualised_return_pct')
            st.metric("Calmar Ratio", _fmt(calmar, '.2f'))
            if ann is not None:
                st.caption(f"Ann. return: {ann:.2f}%")
            st.caption("Ann. return / |max DD %|")

        # col6 — Capital efficiency
        with col6:
            _total_pnl_usd = (
                float(curve['actual_cumulative'].iloc[-1])
                if curve is not None and not curve.empty else None
            )
            _avg_dep_usd = (
                float(deployment['deployed_usd'].mean())
                if deployment is not None and not deployment.empty else None
            )
            efficiency = (
                _total_pnl_usd / _avg_dep_usd
                if _total_pnl_usd is not None and _avg_dep_usd and _avg_dep_usd > 0
                else None
            )
            st.metric("Capital Efficiency", _fmt(efficiency, '.2f'))
            st.caption("Total P&L / avg deployed $")

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
                _has_exec = (
                    curve is not None
                    and not curve.empty
                    and 'execution_cumulative' in curve.columns
                )
                if _has_exec:
                    st.caption("Execution hour prices loaded from local cache")
                else:
                    st.warning(
                        "Execution hour data unavailable — "
                        "run `python3 live_trading/backfill_cache.py` first"
                    )

        # col_t3 reserved for future toggles

        # ── Equity chart ──────────────────────────────────────────────────────
        fig_equity = equity_chart(
            curve,
            show_theoretical=show_theoretical,
            show_execution_hour=show_execution,
            title=f"Cumulative P&L — {strategy_name}",
        )
        st.plotly_chart(fig_equity, use_container_width=True, key=f'{prefix}_equity')

        # ── Drawdown chart ────────────────────────────────────────────────────
        st.markdown("#### Drawdown from peak")
        fig_dd = drawdown_chart(curve, title="")
        st.plotly_chart(fig_dd, use_container_width=True, key=f'{prefix}_drawdown')

        # ── Capital deployment chart ──────────────────────────────────────────
        st.markdown("#### Capital deployment")
        fig_dep = capital_deployment_chart(deployment, title="")
        st.plotly_chart(fig_dep, use_container_width=True, key=f'{prefix}_deployment')

    # ── Per coin tab ──────────────────────────────────────────────────────────
    with tab_coins:
        available_coins = sorted(coin_curves.keys()) if coin_curves else []
        closed_all      = pairs.get('closed', [])

        # ── Build coin_capitals lookup (used by chart normalisation + table) ──
        coin_capitals: dict = {}
        for sym in available_coins:
            sym_pairs = [p for p in closed_all if p['symbol'] == sym]
            cap = next(
                (p['coin_capital'] for p in sym_pairs if p['coin_capital']),
                capital,
            )
            coin_capitals[sym] = float(cap)

        # ── Coin selector + normalise toggle ─────────────────────────────────
        col_sel, col_norm = st.columns([3, 1])
        with col_sel:
            coins_selected = st.multiselect(
                "Coins",
                options=available_coins,
                default=[],
                key=f'{prefix}_coins_selected',
            )
        with col_norm:
            show_pct = st.checkbox(
                "Show as % return",
                value=False,
                key=f'{prefix}_normalise_coins',
            )

        if not available_coins:
            st.info("No closed trades yet — per-coin data unavailable.")
        else:
            shown_coins = coins_selected if coins_selected else available_coins

            # ── Combined portfolio equity chart ───────────────────────────────
            # Sum selected coins' actual_cumulative on a shared daily axis.
            _frames = []
            for _sym in shown_coins:
                _df = coin_curves[_sym][['date', 'actual_cumulative']].copy()
                _df = _df.set_index('date').rename(
                    columns={'actual_cumulative': _sym})
                _frames.append(_df)

            if _frames:
                _wide = pd.concat(_frames, axis=1).sort_index().ffill().fillna(0)
                _combined = _wide.sum(axis=1).reset_index()
                _combined.columns = ['date', 'actual_cumulative']
                # equity_chart expects theoretical_cumulative too — supply zeros
                _combined['theoretical_cumulative'] = 0.0
                _combined['actual_pnl']             = _combined['actual_cumulative'].diff().fillna(0)
                _combined['theoretical_pnl']        = 0.0

                st.markdown("#### Combined portfolio — selected coins")
                fig_combined = equity_chart(
                    _combined,
                    show_theoretical=False,
                    show_execution_hour=False,
                    title="Combined cumulative P&L",
                )
                st.plotly_chart(fig_combined, use_container_width=True,
                                key=f'{prefix}_combined_equity')

            # ── Per-coin equity chart ─────────────────────────────────────────
            st.markdown("#### Per coin cumulative P&L")
            fig_coin_eq = coin_equity_chart(
                coin_curves,
                coins_to_show=coins_selected if coins_selected else None,
                normalised=show_pct,
                coin_capitals=coin_capitals,
                show_combined=False,
                title="",
            )
            st.plotly_chart(fig_coin_eq, use_container_width=True, key=f'{prefix}_coin_equity')

            # ── Per coin statistics table ─────────────────────────────────────
            st.markdown("#### Per coin statistics")

            closed = closed_all

            def _coin_max_dd(symbol: str) -> float | None:
                """Max drawdown % for a single coin's equity curve."""
                df = coin_curves.get(symbol)
                if df is None or df.empty:
                    return None
                coin_cap    = coin_capitals.get(symbol, capital)
                cum         = df['actual_cumulative']
                peak        = cum.cummax()
                dd          = cum - peak
                max_dd_usd  = float(dd.min())
                max_dd_idx  = dd.idxmin()
                peak_at_mdd = float(peak.loc[max_dd_idx])
                port_peak   = coin_cap + peak_at_mdd
                if port_peak <= 0:
                    return 0.0
                return max_dd_usd / port_peak * 100

            display_coins = shown_coins
            table_rows = []
            for sym in display_coins:
                sym_pairs = [p for p in closed if p['symbol'] == sym]
                if not sym_pairs:
                    continue
                n_closed  = len(sym_pairs)
                total_pnl = sum(p['actual_pnl_usd'] for p in sym_pairs)
                coin_cap  = coin_capitals.get(sym, capital)
                ret_pct   = total_pnl / coin_cap * 100 if coin_cap else None
                wins      = [p for p in sym_pairs if p['actual_pnl_usd'] > 0]
                win_rate  = len(wins) / n_closed * 100
                avg_hold  = sum(p['holding_days'] for p in sym_pairs) / n_closed
                max_dd    = _coin_max_dd(sym)

                table_rows.append({
                    'Coin':          sym,
                    'Closed trades': n_closed,
                    'Total P&L':     total_pnl,
                    'Return %':      ret_pct,
                    'Win rate':      win_rate,
                    'Avg hold (d)':  avg_hold,
                    'Max DD %':      max_dd,
                })

            if table_rows:
                df_stats = pd.DataFrame(table_rows)

                def _colour_pnl(val):
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        return ''
                    colour = '#16a34a' if val >= 0 else '#dc2626'
                    return f'color: {colour}; font-weight: 600'

                def _fmt_pnl(v):
                    return f'${v:,.0f}' if v is not None else '—'

                def _fmt_pct(v):
                    return f'{v:.2f}%' if v is not None else '—'

                styled = (
                    df_stats.style
                    .applymap(_colour_pnl, subset=['Total P&L', 'Return %'])
                    .format({
                        'Total P&L':    _fmt_pnl,
                        'Return %':     _fmt_pct,
                        'Win rate':     lambda v: f'{v:.1f}%',
                        'Avg hold (d)': lambda v: f'{v:.1f}',
                        'Max DD %':     _fmt_pct,
                    })
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.info("No closed trades for the selected coins.")

            # ── Capital deployment (filtered to selected coins) ────────────────
            st.markdown("#### Capital deployment — selected coins")
            dep_coins     = tuple(shown_coins) if shown_coins != available_coins else None
            fig_dep_coins = capital_deployment_chart(
                build_capital_deployment(data_dir, coins=dep_coins),
                title="",
            )
            st.plotly_chart(fig_dep_coins, use_container_width=True, key=f'{prefix}_deployment_coins')


def render_fund_portfolio(dashboard_dirs: dict, prefix: str) -> None:
    """
    Cross-strategy fund portfolio view.

    Parameters
    ----------
    dashboard_dirs : {display_name -> absolute data_dir path}
                     e.g. {"Momentum": ".../dashboards/momentum", ...}
    prefix         : unique string prepended to all widget keys
    """
    apply_styles()

    dirs_tuple = tuple(sorted(dashboard_dirs.items()))
    strategy_curves, strategy_cfgs, strategy_deployments, strategy_pairs = \
        _load_fund_data(dirs_tuple)

    # ── Header ────────────────────────────────────────────────────────────────
    n_strategies  = len(dashboard_dirs)
    total_capital = sum(float(cfg.get('capital', 0))
                        for cfg in strategy_cfgs.values())
    st.markdown(
        f'<div class="dash-meta">'
        f'Epsilon Fund &nbsp;·&nbsp; {n_strategies} strategies &nbsp;·&nbsp;'
        f' Total capital: <strong>${total_capital:,.0f}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Data warnings ─────────────────────────────────────────────────────────
    warnings: list[str] = []
    for name in dashboard_dirs:
        n_closed = len(strategy_pairs.get(name, {}).get('closed', []))
        if n_closed == 0:
            warnings.append(f"**{name}** — no trades yet")
        elif n_closed < MIN_TRADES:
            warnings.append(
                f"**{name}** — {n_closed} closed trades "
                f"({MIN_TRADES}+ needed for reliable statistics)"
            )
    if warnings:
        with st.expander("⚠ Data warnings — click to expand"):
            for w in warnings:
                st.markdown(w)

    # ── Strategy selector + toggles ───────────────────────────────────────────
    active_default = [
        name for name in dashboard_dirs
        if not strategy_curves.get(name, pd.DataFrame()).empty
    ]

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        strategies_selected = st.multiselect(
            "Strategies",
            options=list(dashboard_dirs.keys()),
            default=active_default,
            key=f"{prefix}_strategies",
        )
    with col2:
        fund_normalised = st.checkbox(
            "Normalise to 100", value=False,
            key=f"{prefix}_normalised",
        )
    with col3:
        show_btc = st.checkbox(
            "BTC benchmark", value=False,
            key=f"{prefix}_btc",
        )

    if not strategies_selected:
        st.info("Select at least one strategy above.")
        return

    # ── Build combined curve (sum of selected strategy actual_cumulative) ─────
    curves_to_combine = {
        n: strategy_curves[n]
        for n in strategies_selected
        if n in strategy_curves and not strategy_curves[n].empty
    }

    combined_curve_df   = None
    total_capital_sel   = sum(
        float(strategy_cfgs[n].get('capital', 0))
        for n in strategies_selected if n in strategy_cfgs
    )

    if curves_to_combine:
        _frames = []
        for _n, _c in curves_to_combine.items():
            _f = (_c[['date', 'actual_cumulative']]
                  .set_index('date')
                  .rename(columns={'actual_cumulative': _n}))
            _frames.append(_f)
        _wide       = pd.concat(_frames, axis=1).sort_index().ffill().fillna(0)
        _combined_s = _wide.sum(axis=1)
        _combined_p = _combined_s.diff().fillna(0)
        combined_curve_df = pd.DataFrame({
            'date':                   _combined_s.index,
            'actual_cumulative':      _combined_s.values,
            'theoretical_cumulative': _combined_s.values,   # not used
            'actual_pnl':             _combined_p.values,
            'theoretical_pnl':        _combined_p.values,
        }).reset_index(drop=True)

    # ── Fund metrics ──────────────────────────────────────────────────────────
    _window = st.session_state.get(f'{prefix}_rolling_window', 30)

    if combined_curve_df is not None and total_capital_sel > 0:
        fund_metrics = compute_metrics(
            combined_curve_df, window_days=_window,
            capital=total_capital_sel,
        )
    else:
        fund_metrics = {k: None for k in [
            'total_return_pct', 'sharpe', 'rolling_sharpe', 'max_drawdown_pct',
        ]}

    # Current capital deployed (today's row across selected strategies)
    dep_today_usd = 0.0
    for _n in strategies_selected:
        _dep = strategy_deployments.get(_n)
        if _dep is not None and not _dep.empty:
            dep_today_usd += float(_dep['deployed_usd'].iloc[-1])
    dep_today_pct = (dep_today_usd / total_capital_sel * 100
                     if total_capital_sel else None)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Return",
                  _fmt(fund_metrics.get('total_return_pct'), '.2f', '%'))
    with col2:
        st.metric("Sharpe Ratio",
                  _fmt(fund_metrics.get('sharpe'), '.2f'))
        st.caption("Annualised")
    with col3:
        _rs = fund_metrics.get('rolling_sharpe')
        _rs_latest = None
        if _rs is not None:
            _valid = _rs.dropna()
            _rs_latest = float(_valid.iloc[-1]) if not _valid.empty else None
        st.metric("Rolling Sharpe", _fmt(_rs_latest, '.2f'))
        _window = st.selectbox(
            "Window (days)", [30, 60, 90],
            key=f'{prefix}_rolling_window',
        )
    with col4:
        st.metric("Max Drawdown",
                  _fmt(fund_metrics.get('max_drawdown_pct'), '.2f', '%'))
    with col5:
        st.metric("Capital Deployed",
                  _fmt(dep_today_pct, '.1f', '%'))

    # ── Trading cost note ─────────────────────────────────────────────────────
    _sel_cost_pcts = {
        n: float(strategy_cfgs[n].get('trading_cost_pct', 0.0))
        for n in strategies_selected if n in strategy_cfgs
    }
    _nonzero_costs = {n: v for n, v in _sel_cost_pcts.items() if v > 0}
    if _nonzero_costs:
        _unique_costs = set(_nonzero_costs.values())
        if len(_unique_costs) == 1:
            _cc = next(iter(_unique_costs))
            st.caption(
                f"P&L figures include trading costs: "
                f"**{_cc * 100:.3f}%** per leg · "
                f"**{_cc * 2 * 100:.3f}%** round-trip"
            )
        else:
            _cparts = [f"{n}: {v * 100:.3f}%" for n, v in _nonzero_costs.items()]
            st.caption("P&L includes per-leg trading costs — " + ", ".join(_cparts))

    st.divider()

    # ── BTC benchmark series ──────────────────────────────────────────────────
    btc_series = None
    if show_btc and combined_curve_df is not None and len(combined_curve_df) > 0:
        _start = str(combined_curve_df['date'].iloc[0])
        _end   = str(combined_curve_df['date'].iloc[-1])
        btc_series = _fetch_btc_series(_start, _end)

    # ── Fund equity chart ─────────────────────────────────────────────────────
    fig_fund = fund_equity_chart(
        strategy_curves,
        strategies_to_show=strategies_selected,
        normalised=fund_normalised,
        benchmark_series=btc_series,
        total_capital=total_capital_sel,
        title="Epsilon Fund — Combined equity",
    )
    st.plotly_chart(fig_fund, use_container_width=True,
                    key=f'{prefix}_fund_equity')

    # ── Drawdown chart ────────────────────────────────────────────────────────
    st.markdown("#### Drawdown from peak")
    if combined_curve_df is not None and not combined_curve_df.empty:
        fig_dd = drawdown_chart(combined_curve_df, title="")
        st.plotly_chart(fig_dd, use_container_width=True,
                        key=f'{prefix}_fund_drawdown')
    else:
        st.info("No combined curve — select at least one strategy with trades.")

    # ── Capital deployment (combined across selected strategies) ──────────────
    st.markdown("#### Capital deployment")
    _dep_frames = []
    for _n in strategies_selected:
        _dep = strategy_deployments.get(_n)
        if _dep is not None and not _dep.empty:
            _f = (_dep[['date', 'deployed_usd']]
                  .set_index('date')
                  .rename(columns={'deployed_usd': _n}))
            _dep_frames.append(_f)

    if _dep_frames:
        _dep_wide  = pd.concat(_dep_frames, axis=1).sort_index().ffill().fillna(0)
        _dep_total = _dep_wide.sum(axis=1)
        combined_dep = pd.DataFrame({
            'date':           _dep_total.index,
            'deployed_usd':   _dep_total.values,
            'deployment_pct': (
                _dep_total.values / total_capital_sel * 100
                if total_capital_sel else 0.0
            ),
        }).reset_index(drop=True)
        fig_dep = capital_deployment_chart(combined_dep, title="")
        st.plotly_chart(fig_dep, use_container_width=True,
                        key=f'{prefix}_fund_deployment')
    else:
        st.info("No deployment data for selected strategies.")

    # ── Per-strategy breakdown table ──────────────────────────────────────────
    st.markdown("#### Per strategy")
    table_rows = []
    for name in strategies_selected:
        _curve  = strategy_curves.get(name)
        _cfg    = strategy_cfgs.get(name, {})
        _pairs  = strategy_pairs.get(name, {})
        _dep    = strategy_deployments.get(name)
        _cap    = float(_cfg.get('capital', 0))
        _closed = _pairs.get('closed', [])

        if _curve is not None and not _curve.empty and _cap > 0:
            _m = compute_metrics(_curve, window_days=_window, capital=_cap)
        else:
            _m = None

        _total_pnl = (float(_curve['actual_cumulative'].iloc[-1])
                      if _curve is not None and not _curve.empty else None)

        _avg_dep_usd = (
            float(_dep['deployed_usd'].mean())
            if _dep is not None and not _dep.empty else None
        )
        _efficiency = (
            _total_pnl / _avg_dep_usd
            if _total_pnl is not None and _avg_dep_usd and _avg_dep_usd > 0
            else None
        )

        table_rows.append({
            'Strategy':      name,
            'Capital':       _cap,
            'Closed trades': len(_closed),
            'Total P&L ($)': _total_pnl,
            'Return %':      _m['total_return_pct']  if _m else None,
            'Sharpe':        _m['sharpe']             if _m else None,
            'Max DD %':      _m['max_drawdown_pct']   if _m else None,
            'Calmar':        _m.get('calmar')          if _m else None,
            'Efficiency':    _efficiency,
        })

    if table_rows:
        df_strat = pd.DataFrame(table_rows)

        def _colour_signed(val):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return ''
            return ('color: #16a34a; font-weight: 600'
                    if val >= 0 else 'color: #dc2626; font-weight: 600')

        styled_strat = (
            df_strat.style
            .applymap(_colour_signed, subset=['Total P&L ($)', 'Return %'])
            .format({
                'Capital':       lambda v: f'${v:,.0f}',
                'Total P&L ($)': lambda v: f'${v:,.0f}' if v is not None else '—',
                'Return %':      lambda v: f'{v:.2f}%'  if v is not None else '—',
                'Sharpe':        lambda v: f'{v:.2f}'   if v is not None else '—',
                'Max DD %':      lambda v: f'{v:.2f}%'  if v is not None else '—',
                'Calmar':        lambda v: f'{v:.2f}'   if v is not None else '—',
                'Efficiency':    lambda v: f'{v:.2f}'   if v is not None else '—',
            })
        )
        st.dataframe(styled_strat, use_container_width=True, hide_index=True)

    # ── Strategy correlation heatmap ──────────────────────────────────────────
    st.markdown("#### Strategy correlation")
    _strats_with_trades = [
        n for n in strategies_selected
        if len(strategy_pairs.get(n, {}).get('closed', [])) > 0
    ]
    if len(_strats_with_trades) < 2:
        st.caption(
            "Select 2 or more strategies with closed trades to see return correlation."
        )
    else:
        # Build daily P&L series per strategy from equity curves
        _pnl_frames: dict = {}
        for _n in _strats_with_trades:
            _c = strategy_curves.get(_n)
            if _c is not None and not _c.empty and 'actual_pnl' in _c.columns:
                _s = _c[['date', 'actual_pnl']].set_index('date')['actual_pnl']
                _pnl_frames[_n] = _s

        if len(_pnl_frames) >= 2:
            _pnl_df = (
                pd.DataFrame(_pnl_frames)
                .sort_index()
                .ffill()
                .fillna(0)
            )
            _corr = _pnl_df.corr()
            fig_corr = correlation_heatmap(
                _corr,
                title="Daily P&L correlation across strategies",
            )
            st.plotly_chart(fig_corr, use_container_width=False,
                            key=f'{prefix}_corr_heatmap')
        else:
            st.caption("Insufficient curve data to compute correlation.")
