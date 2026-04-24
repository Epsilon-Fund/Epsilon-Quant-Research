"""
Shared Plotly chart builders. Each function returns a figure — pages call st.plotly_chart().
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Portfolio-page style constants ────────────────────────────────────────────
_C_ACTUAL      = '#3b82f6'   # blue
_C_THEORETICAL = '#94a3b8'   # muted grey-blue
_C_EXEC_HOUR   = '#f59e0b'   # amber
_C_BENCHMARK   = '#f97316'   # orange
_C_ZERO_LINE   = '#cbd5e1'   # light grey

# Plotly qualitative Safe palette for per-coin / per-strategy lines
_SAFE_PALETTE = px.colors.qualitative.Safe

_COIN_PALETTE = {
    'BTCUSDT':  '#F7931A',
    'ETHUSDT':  '#627EEA',
    'SOLUSDT':  '#9945FF',
    'XRPUSDT':  '#00AAE4',
    'AVAXUSDT': '#E84142',
    'BNBUSDT':  '#F3BA2F',
    'ADAUSDT':  '#0033AD',
}
_DEFAULT_COLOUR = '#888780'


def _coin_colour(pid: str) -> str:
    return _COIN_PALETTE.get(pid, _DEFAULT_COLOUR)


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        paper_bgcolor='white',
        plot_bgcolor='#f8f8f7',
        font=dict(family='Inter, sans-serif', size=12, color='#444441'),
    )
    return fig


def _base_layout(fig: go.Figure, title: str, xaxis_title='', yaxis_title='') -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        paper_bgcolor='white',
        plot_bgcolor='#f8f8f7',
        font=dict(family='Inter, sans-serif', size=12, color='#444441'),
        legend=dict(
            bgcolor='white', bordercolor='#d3d1c7', borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode='x unified',
    )
    fig.update_xaxes(showgrid=False, linecolor='#d3d1c7', tickfont=dict(size=11))
    fig.update_yaxes(gridcolor='#e4e4e1', linecolor='#d3d1c7', tickfont=dict(size=11))


# ── Equity curve ──────────────────────────────────────────────────────────────

def equity_curve_chart(trade_pairs: dict, benchmark_df=None) -> go.Figure:
    """
    Cumulative P&L chart: Actual vs Theoretical lines.
    Points are placed at each trade's exit_date.
    benchmark_df: optional DataFrame with a 'Close' column indexed by date (BTC buy-and-hold).
    """
    closed = sorted(
        trade_pairs.get('closed', []),
        key=lambda x: x['exit_date'] or x['entry_date'],
    )

    if not closed:
        return _empty_fig('Equity Curve (no closed trades yet)')

    dates      = [t['exit_date'] for t in closed]
    actual_cum = list(np.cumsum([t['actual_pnl_usd'] for t in closed]))
    theo_cum   = list(np.cumsum([t['theoretical_pnl_usd'] for t in closed]))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=actual_cum,
        name='Actual',
        mode='lines+markers',
        line=dict(color='#3B6D11', width=2),
        marker=dict(size=6),
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=theo_cum,
        name='Theoretical',
        mode='lines+markers',
        line=dict(color='#888780', width=2, dash='dash'),
        marker=dict(size=6),
    ))

    if benchmark_df is not None and not benchmark_df.empty and len(closed) >= 2:
        start = closed[0]['entry_date']
        end   = closed[-1]['exit_date']
        bdf   = benchmark_df.loc[
            (benchmark_df.index >= str(start)) & (benchmark_df.index <= str(end))
        ]
        if not bdf.empty:
            bdf_close     = bdf['Close'].astype(float)
            bdf_normed    = (bdf_close / bdf_close.iloc[0] - 1) * float(
                sum(t['actual_size_usd'] for t in closed) / len(closed)
            )
            fig.add_trace(go.Scatter(
                x=bdf.index, y=bdf_normed,
                name='BTC B&H',
                mode='lines',
                line=dict(color='#F7931A', width=1.5, dash='dot'),
                opacity=0.7,
            ))

    _base_layout(fig, 'Equity Curve — Cumulative P&L', xaxis_title='Date', yaxis_title='Cumulative P&L ($)')
    fig.add_hline(y=0, line_color='#d3d1c7', line_width=1)
    return fig


# ── Drawdown ──────────────────────────────────────────────────────────────────

def drawdown_chart(trade_pairs: dict) -> go.Figure:
    """Running drawdown from peak on the actual cumulative P&L series."""
    closed = sorted(
        trade_pairs.get('closed', []),
        key=lambda x: x['exit_date'] or x['entry_date'],
    )

    if not closed:
        return _empty_fig('Drawdown (no closed trades yet)')

    dates  = [t['exit_date'] for t in closed]
    cum_pnl = np.cumsum([t['actual_pnl_usd'] for t in closed])

    running_max = np.maximum.accumulate(cum_pnl)
    drawdown    = cum_pnl - running_max   # always <= 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=list(drawdown),
        name='Drawdown',
        mode='lines',
        fill='tozeroy',
        line=dict(color='#A32D2D', width=1.5),
        fillcolor='rgba(163,45,45,0.12)',
    ))

    _base_layout(fig, 'Drawdown from Peak', xaxis_title='Date', yaxis_title='Drawdown ($)')
    fig.add_hline(y=0, line_color='#d3d1c7', line_width=1)
    return fig


# ── Slippage ──────────────────────────────────────────────────────────────────

def slippage_chart(trade_pairs: dict) -> go.Figure:
    """Bar chart of entry slippage (%) per closed trade, coloured by coin."""
    closed = sorted(
        trade_pairs.get('closed', []),
        key=lambda x: x['exit_date'] or x['entry_date'],
    )

    if not closed:
        return _empty_fig('Slippage (no closed trades yet)')

    labels  = [f"{t['position_id']} {t['entry_date']}" for t in closed]
    slips   = [t['entry_slippage_pct'] for t in closed]
    colours = [_coin_colour(t['position_id']) for t in closed]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=slips,
        marker_color=colours,
        name='Entry slippage (%)',
        text=[f"{s:.3f}%" for s in slips],
        textposition='outside',
    ))

    _base_layout(fig, 'Entry Slippage per Trade', xaxis_title='Trade', yaxis_title='Slippage (%)')
    fig.add_hline(y=0, line_color='#444441', line_width=1)
    return fig


# ── Per-coin return comparison ────────────────────────────────────────────────

def per_coin_chart(trade_pairs: dict) -> go.Figure:
    """Grouped bar chart: actual vs theoretical return (%) per closed trade, grouped by coin."""
    closed = sorted(
        trade_pairs.get('closed', []),
        key=lambda x: x['exit_date'] or x['entry_date'],
    )

    if not closed:
        return _empty_fig('Per-coin returns (no closed trades yet)')

    labels    = [f"{t['position_id']} {t['entry_date']}" for t in closed]
    # Use net return (after round-trip costs) so bars reflect actual P&L impact.
    # Falls back to gross return for legacy records missing the net field.
    actual    = [t.get('actual_net_return_pct',      t['actual_return_pct'])      for t in closed]
    theo      = [t.get('theoretical_net_return_pct', t['theoretical_return_pct']) for t in closed]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Actual',
        x=labels,
        y=actual,
        marker_color='#3B6D11',
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name='Theoretical',
        x=labels,
        y=theo,
        marker_color='#888780',
        opacity=0.85,
    ))

    fig.update_layout(barmode='group')
    _base_layout(fig, 'Return per Trade: Actual vs Theoretical', xaxis_title='Trade', yaxis_title='Return (%)')
    fig.add_hline(y=0, line_color='#d3d1c7', line_width=1)
    return fig


# ── Portfolio-summary chart helpers ───────────────────────────────────────────

def _portfolio_layout(fig: go.Figure, title: str,
                      xaxis_title='', yaxis_title='') -> None:
    """Base layout for all portfolio-page charts."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='system-ui, sans-serif', size=12, color='#334155'),
        legend=dict(
            bgcolor='white', bordercolor='#e2e8f0', borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=64, r=24, t=52, b=48),
        hovermode='x unified',
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor='#e2e8f0',
        tickfont=dict(size=11),
    )
    fig.update_yaxes(
        gridcolor='#f1f5f9',
        linecolor='#e2e8f0',
        tickfont=dict(size=11),
    )


def _dollar_tickformat() -> str:
    return '$,.0f'


# ── 1. equity_chart ───────────────────────────────────────────────────────────

def equity_chart(curve_df: pd.DataFrame,
                 show_theoretical: bool = True,
                 show_execution_hour: bool = False,
                 title: str = '') -> go.Figure:
    """
    Cumulative P&L line chart (actual, optional theoretical, optional execution-hour).

    curve_df columns: date | actual_cumulative | theoretical_cumulative
                      (optional) execution_cumulative
    """
    fig = go.Figure()

    if curve_df.empty:
        _portfolio_layout(fig, title or 'Equity Curve',
                          xaxis_title='Date', yaxis_title='Cumulative P&L ($)')
        return fig

    dates = curve_df['date']

    fig.add_trace(go.Scatter(
        x=dates,
        y=curve_df['actual_cumulative'],
        name='Actual',
        mode='lines',
        line=dict(color=_C_ACTUAL, width=2),
        hovertemplate='%{x|%b %d, %Y}<br>Actual: $%{y:,.0f}<extra></extra>',
    ))

    if show_theoretical and 'theoretical_cumulative' in curve_df.columns:
        fig.add_trace(go.Scatter(
            x=dates,
            y=curve_df['theoretical_cumulative'],
            name='Theoretical',
            mode='lines',
            line=dict(color=_C_THEORETICAL, width=1.5, dash='dash'),
            hovertemplate='%{x|%b %d, %Y}<br>Theoretical: $%{y:,.0f}<extra></extra>',
        ))

    if (show_execution_hour
            and 'execution_cumulative' in curve_df.columns):
        fig.add_trace(go.Scatter(
            x=dates,
            y=curve_df['execution_cumulative'],
            name='Execution-hour',
            mode='lines',
            line=dict(color=_C_EXEC_HOUR, width=1.5, dash='dot'),
            hovertemplate='%{x|%b %d, %Y}<br>Exec-hour: $%{y:,.0f}<extra></extra>',
        ))

    fig.add_hline(y=0, line_color=_C_ZERO_LINE, line_width=1)
    _portfolio_layout(fig, title or 'Equity Curve — Cumulative P&L',
                      xaxis_title='Date', yaxis_title='Cumulative P&L ($)')
    fig.update_yaxes(tickformat=_dollar_tickformat())
    return fig


# ── 2. drawdown_chart ─────────────────────────────────────────────────────────

def drawdown_chart(curve_df: pd.DataFrame, title: str = '') -> go.Figure:
    """
    Drawdown from peak on actual_cumulative, in dollars.

    drawdown = actual_cumulative - running_peak  (always <= 0)
    No division by peak — avoids inf/nan when the curve starts at 0.
    """
    fig = go.Figure()

    if curve_df.empty or 'actual_cumulative' not in curve_df.columns:
        _portfolio_layout(fig, title or 'Drawdown from Peak',
                          xaxis_title='Date', yaxis_title='Drawdown ($)')
        return fig

    cum      = curve_df['actual_cumulative']
    peak     = cum.cummax()
    drawdown = cum - peak   # always <= 0, in dollars

    fig.add_trace(go.Scatter(
        x=curve_df['date'],
        y=drawdown,
        name='Drawdown',
        mode='lines',
        fill='tozeroy',
        line=dict(color='#ef4444', width=1.5),
        fillcolor='rgba(254,226,226,0.6)',
        hovertemplate='%{x|%b %d, %Y}<br>Drawdown: $%{y:,.0f}<extra></extra>',
    ))

    fig.add_hline(y=0, line_color=_C_ZERO_LINE, line_width=1)
    _portfolio_layout(fig, title or 'Drawdown from Peak',
                      xaxis_title='Date', yaxis_title='Drawdown ($)')
    fig.update_yaxes(tickformat=_dollar_tickformat())
    return fig


# ── 3. capital_deployment_chart ───────────────────────────────────────────────

def capital_deployment_chart(deployment_df: pd.DataFrame,
                              title: str = '') -> go.Figure:
    """Bar chart of daily capital deployment %."""
    fig = go.Figure()

    if deployment_df.empty:
        _portfolio_layout(fig, title or 'Capital Deployment',
                          xaxis_title='Date', yaxis_title='% Deployed')
        return fig

    customdata = np.stack([
        deployment_df['deployed_usd'],
        deployment_df['deployment_pct'],
    ], axis=1)

    fig.add_trace(go.Bar(
        x=deployment_df['date'],
        y=deployment_df['deployment_pct'],
        name='Deployed %',
        marker_color=_C_ACTUAL,
        opacity=0.6,
        customdata=customdata,
        hovertemplate=(
            '%{x|%b %d, %Y}<br>'
            'Deployed: $%{customdata[0]:,.0f}<br>'
            'Deployment: %{customdata[1]:.1f}%'
            '<extra></extra>'
        ),
    ))

    fig.add_hline(y=100, line_color=_C_ZERO_LINE, line_width=1,
                  annotation_text='100%', annotation_position='top right')
    _portfolio_layout(fig, title or 'Capital Deployment',
                      xaxis_title='Date', yaxis_title='% of Capital Deployed')
    fig.update_yaxes(range=[0, max(110, deployment_df['deployment_pct'].max() * 1.1)])
    return fig


# ── 4. coin_equity_chart ──────────────────────────────────────────────────────

def coin_equity_chart(coin_curves_dict: dict,
                      coins_to_show: list = None,
                      normalised: bool = False,
                      coin_capitals: dict = None,
                      show_combined: bool = False,
                      title: str = '') -> go.Figure:
    """
    Multi-line cumulative P&L (or % return) per coin, with optional combined
    portfolio line.

    normalised=False → Y axis in USD cumulative P&L.
    normalised=True  → Y axis as % return on allocated coin capital
                       (actual_cumulative / coin_capital * 100).
                       coin_capitals dict {symbol: float} must be supplied.
    show_combined    → overlay a thick dashed line summing all shown coins:
                       total P&L in USD mode, blended % return in normalised mode.
    """
    fig = go.Figure()

    symbols = list(coin_curves_dict.keys())
    if coins_to_show is not None:
        symbols = [s for s in symbols if s in coins_to_show]

    if not symbols:
        yaxis_title = '% Return on allocated capital' if normalised else 'Cumulative P&L ($)'
        _portfolio_layout(fig, title or 'Per-Coin Equity',
                          xaxis_title='Date', yaxis_title=yaxis_title)
        return fig

    # ── Individual coin lines ─────────────────────────────────────────────────
    for i, symbol in enumerate(symbols):
        df    = coin_curves_dict[symbol]
        color = _SAFE_PALETTE[i % len(_SAFE_PALETTE)]
        cum   = df['actual_cumulative']

        if normalised and coin_capitals and symbol in coin_capitals:
            cap = float(coin_capitals[symbol])
            y   = cum / cap * 100 if cap else cum
            hover_tmpl = (
                f'%{{x|%b %d, %Y}}<br>{symbol}: %{{y:.2f}}%<extra></extra>'
            )
        else:
            y          = cum
            hover_tmpl = (
                f'%{{x|%b %d, %Y}}<br>{symbol}: $%{{y:,.0f}}<extra></extra>'
            )

        fig.add_trace(go.Scatter(
            x=df['date'],
            y=y,
            name=symbol,
            mode='lines',
            line=dict(color=color, width=2),
            hovertemplate=hover_tmpl,
        ))

    # ── Combined portfolio line ───────────────────────────────────────────────
    if show_combined and len(symbols) > 1:
        frames = []
        for sym in symbols:
            df_s = coin_curves_dict[sym][['date', 'actual_cumulative']].copy()
            df_s = df_s.set_index('date').rename(
                columns={'actual_cumulative': sym})
            frames.append(df_s)

        wide     = pd.concat(frames, axis=1).sort_index().ffill().fillna(0)
        combined = wide.sum(axis=1)

        if normalised and coin_capitals:
            total_cap = sum(float(coin_capitals.get(s, 0)) for s in symbols)
            y_comb    = combined / total_cap * 100 if total_cap else combined
            hover_comb = '%{x|%b %d, %Y}<br>Combined: %{y:.2f}%<extra></extra>'
        else:
            y_comb     = combined
            hover_comb = '%{x|%b %d, %Y}<br>Combined: $%{y:,.0f}<extra></extra>'

        fig.add_trace(go.Scatter(
            x=wide.index,
            y=y_comb,
            name='Combined',
            mode='lines',
            line=dict(color=_C_ACTUAL, width=2.5, dash='dash'),
            hovertemplate=hover_comb,
        ))

    fig.add_hline(y=0, line_color=_C_ZERO_LINE, line_width=1)

    yaxis_title = '% Return on allocated capital' if normalised else 'Cumulative P&L ($)'
    _portfolio_layout(fig, title or 'Per-Coin Equity',
                      xaxis_title='Date', yaxis_title=yaxis_title)
    if not normalised:
        fig.update_yaxes(tickformat=_dollar_tickformat())
    return fig


# ── 5. fund_equity_chart ──────────────────────────────────────────────────────


def fund_equity_chart(strategy_curves_dict: dict,
                      strategies_to_show: list = None,
                      normalised: bool = False,
                      benchmark_series: pd.Series = None,
                      total_capital: float = 0.0,
                      title: str = '') -> go.Figure:
    """
    Combined fund equity curve, summing actual_cumulative across strategies.

    strategy_curves_dict : {strategy_name: equity_curve_df}
    benchmark_series     : pd.Series of BTC daily closes, date-indexed.
    total_capital        : sum of capital across selected strategies; used to
                           index the curve correctly and to convert BTC to
                           equivalent dollar P&L when not normalised.
    normalised           : if True, plot portfolio value indexed to 100 at the
                           start date.  Requires total_capital > 0.
    """
    fig = go.Figure()

    names = list(strategy_curves_dict.keys())
    if strategies_to_show is not None:
        names = [n for n in names if n in strategies_to_show]

    if not names:
        yaxis_title = '% Return (indexed to 100)' if normalised else 'Cumulative P&L ($)'
        _portfolio_layout(fig, title or 'Fund Equity Curve',
                          xaxis_title='Date', yaxis_title=yaxis_title)
        return fig

    # Align all strategy curves on a common date index, fill gaps with ffill
    frames = []
    for name in names:
        df = strategy_curves_dict[name][['date', 'actual_cumulative']].copy()
        df = df.set_index('date').rename(columns={'actual_cumulative': name})
        frames.append(df)

    combined_wide = pd.concat(frames, axis=1).sort_index()
    combined_wide = combined_wide.ffill().fillna(0)
    combined_sum  = combined_wide.sum(axis=1)   # cumulative P&L series

    dates = combined_sum.index

    if normalised and total_capital > 0:
        # Portfolio value = capital + cumulative P&L, then index to 100.
        # combined_sum starts at 0 (no trades closed yet), so day-1 value
        # is exactly total_capital → index = 100.0 ✓
        portfolio_value = total_capital + combined_sum
        y_plot      = portfolio_value / total_capital * 100
        hover_tmpl  = '%{x|%b %d, %Y}<br>Fund: %{y:.2f}<extra></extra>'
        ref_line    = 100
        yaxis_title = 'Portfolio value (indexed to 100)'
    else:
        y_plot      = combined_sum
        hover_tmpl  = '%{x|%b %d, %Y}<br>Fund: $%{y:,.0f}<extra></extra>'
        ref_line    = 0
        yaxis_title = 'Cumulative P&L ($)'

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_plot,
        name='Fund',
        mode='lines',
        line=dict(color=_C_ACTUAL, width=2.5),
        hovertemplate=hover_tmpl,
    ))

    if benchmark_series is not None and not benchmark_series.empty:
        # Align benchmark to combined curve start date
        start = dates[0]
        bm    = benchmark_series[benchmark_series.index >= str(start)]
        if not bm.empty:
            if normalised and total_capital > 0:
                # Index BTC to 100 — same scale as the fund index
                bm_y   = bm / bm.iloc[0] * 100
                bm_lbl = 'BTC (indexed to 100)'
                bm_tmpl = '%{x|%b %d, %Y}<br>BTC: %{y:.2f}<extra></extra>'
            elif total_capital > 0:
                # Convert BTC to equivalent dollar P&L:
                # "if you had invested total_capital in BTC instead"
                bm_y   = (bm / bm.iloc[0] - 1) * total_capital
                bm_lbl = 'BTC equiv. P&L ($)'
                bm_tmpl = '%{x|%b %d, %Y}<br>BTC: $%{y:,.0f}<extra></extra>'
            else:
                # No capital info — index to 100 as fallback
                bm_y   = bm / bm.iloc[0] * 100
                bm_lbl = 'BTC (indexed)'
                bm_tmpl = '%{x|%b %d, %Y}<br>BTC: %{y:.2f}<extra></extra>'

            fig.add_trace(go.Scatter(
                x=bm.index,
                y=bm_y,
                name=bm_lbl,
                mode='lines',
                line=dict(color=_C_BENCHMARK, width=1.5, dash='dot'),
                opacity=0.75,
                hovertemplate=bm_tmpl,
            ))

    fig.add_hline(y=ref_line, line_color=_C_ZERO_LINE, line_width=1)
    _portfolio_layout(fig, title or 'Fund Equity Curve',
                      xaxis_title='Date', yaxis_title=yaxis_title)
    if not normalised:
        fig.update_yaxes(tickformat=_dollar_tickformat())
    return fig


# ── 6. correlation_heatmap ────────────────────────────────────────────────────

def correlation_heatmap(corr_matrix: pd.DataFrame, title: str = '') -> go.Figure:
    """
    Strategy (or coin) correlation heatmap with annotated cells.

    corr_matrix : square DataFrame with strategy names as index and columns,
                  values in [-1, 1].
    """
    fig = go.Figure()

    if corr_matrix is None or corr_matrix.empty:
        _portfolio_layout(fig, title or 'Strategy Correlation')
        return fig

    labels = list(corr_matrix.columns)
    z      = corr_matrix.values
    text   = [[f'{v:.2f}' for v in row] for row in z]

    fig.add_trace(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        text=text,
        texttemplate='%{text}',
        colorscale='RdYlGn',
        zmin=-1,
        zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text='Correlation', side='right'),
            thickness=12,
            len=0.8,
        ),
        hovertemplate='%{y} × %{x}: %{z:.2f}<extra></extra>',
    ))

    n = len(labels)
    fig.update_layout(
        title=dict(text=title or 'Strategy Correlation', font=dict(size=14)),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='system-ui, sans-serif', size=12, color='#334155'),
        margin=dict(l=64, r=24, t=52, b=80),
        width=max(400, n * 120),
        height=max(360, n * 100),
        xaxis=dict(
            showgrid=False,
            linecolor='#e2e8f0',
            tickfont=dict(size=11),
            side='bottom',
        ),
        yaxis=dict(
            showgrid=False,
            linecolor='#e2e8f0',
            tickfont=dict(size=11),
            autorange='reversed',
        ),
    )
    return fig
