"""
Shared Plotly chart builders. Each function returns a figure — pages call st.plotly_chart().
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
        title=dict(text=title, font=dict(size=14, weight=700)),
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
    actual    = [t['actual_return_pct']      for t in closed]
    theo      = [t['theoretical_return_pct'] for t in closed]

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
