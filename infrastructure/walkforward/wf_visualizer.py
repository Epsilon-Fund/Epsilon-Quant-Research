"""
Walk-Forward Visualizer
──────────────────────────────────────────────────────────────────────────────
Companion to infrastructure/backtester/visualizer.py
Same Plotly style, same color palette, same template.

Main entry point
----------------
    from walk_forward.wf_visualizer import plot_walk_forward_results

    plot_walk_forward_results(results, show=True, save_html='wf_results.html')
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── shared style constants (mirrors visualizer.py) ─────────────────────────────
_BLUE       = '#3b82f6'
_AMBER      = '#f59e0b'
_GREEN      = '#22c55e'
_RED        = '#ef4444'
_RED_FILL   = 'rgba(239, 68, 68, 0.3)'
_GREY       = '#64748b'
_GRID       = '#f1f5f9'
_BORDER     = '#cbd5e1'
_BG_BOX     = 'rgba(255, 255, 255, 0.9)'
_FONT_MONO  = 'monospace'
_TEMPLATE   = 'plotly_white'


# ──────────────────────────────────────────────────────────────────────────────
#  Fold performance bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_fold_performance(results_df, show=True, save_html=None):
    """
    Side-by-side IS / OOS bars for return, Sharpe, and max drawdown per fold.
    Mirrors the annotation + layout style of visualizer.py.
    """
    labels = [f'Fold {r}' for r in results_df['fold']]

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)'),
        vertical_spacing=0.10,
        row_heights=[0.35, 0.35, 0.30],
    )

    # ── Return ──────────────────────────────────────────────────────────────
    for col, name, color in [
        ('train_return', 'IS Return',  _BLUE),
        ('test_return',  'OOS Return', _GREEN),
    ]:
        vals = results_df[col].fillna(0) * 100
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=name,
            marker_color=color, opacity=0.85,
            hovertemplate=f'<b>{name}</b><br>%{{x}}<br>%{{y:.2f}}%<extra></extra>',
        ), row=1, col=1)

    # ── Sharpe ───────────────────────────────────────────────────────────────
    for col, name, color in [
        ('train_sharpe', 'IS Sharpe',  _BLUE),
        ('test_sharpe',  'OOS Sharpe', _GREEN),
    ]:
        vals = results_df[col].fillna(0)
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=name,
            marker_color=color, opacity=0.85,
            showlegend=False,
            hovertemplate=f'<b>{name}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>',
        ), row=2, col=1)

    # ── Drawdown ─────────────────────────────────────────────────────────────
    for col, name, color in [
        ('train_drawdown', 'IS Drawdown',  _BLUE),
        ('test_drawdown',  'OOS Drawdown', _RED),
    ]:
        vals = results_df[col].fillna(0) * 100
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=name,
            marker_color=color, opacity=0.85,
            showlegend=False,
            hovertemplate=f'<b>{name}</b><br>%{{x}}<br>%{{y:.2f}}%<extra></extra>',
        ), row=3, col=1)

    fig.update_layout(
        height=900,
        barmode='group',
        hovermode='x unified',
        template=_TEMPLATE,
        title=dict(
            text='<b>Walk-Forward: IS vs OOS Performance by Fold</b>',
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
        legend=dict(
            yanchor='top', y=0.99, xanchor='right', x=0.99,
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1,
        ),
    )

    for row in range(1, 4):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=_GRID, row=row, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=_GRID, row=row, col=1)

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved fold performance chart → {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Parameter evolution across folds
# ──────────────────────────────────────────────────────────────────────────────

def plot_parameter_evolution(results_df, param_defs, fixed_params=None,
                             show=True, save_html=None):
    """
    Line chart showing how each free parameter moved across folds.
    Fixed parameters are excluded — they don't evolve.
    """
    if fixed_params is None:
        fixed_params = {}

    free_params = [k for k in param_defs if k not in fixed_params]
    n = len(free_params)
    if n == 0:
        print('No free parameters to plot.')
        return None

    cols  = 2
    rows  = int(np.ceil(n / cols))
    titles = free_params

    v_spacing = min(0.08, round(0.9 / max(rows - 1, 1), 3))

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        vertical_spacing=v_spacing,
        horizontal_spacing=0.10,
    )

    fold_labels = [f'Fold {r}' for r in results_df['fold']]
    fold_nums   = list(results_df['fold'])

    for idx, name in enumerate(free_params):
        row = idx // cols + 1
        col = idx %  cols + 1
        vals = results_df[f'param_{name}'].values

        # shaded ±1 std band
        med = np.median(vals)
        std = np.std(vals)
        fig.add_trace(go.Scatter(
            x=fold_nums + fold_nums[::-1],
            y=list(np.full(len(fold_nums), med + std)) +
              list(np.full(len(fold_nums), med - std)),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.12)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=(idx == 0),
            name='±1 std',
            hoverinfo='skip',
        ), row=row, col=col)

        # median line
        fig.add_trace(go.Scatter(
            x=fold_nums,
            y=np.full(len(fold_nums), med),
            mode='lines',
            line=dict(color=_AMBER, dash='dash', width=1),
            showlegend=(idx == 0),
            name='Median',
            hoverinfo='skip',
        ), row=row, col=col)

        # actual value per fold
        fig.add_trace(go.Scatter(
            x=fold_nums, y=vals,
            mode='lines+markers',
            line=dict(color=_BLUE, width=2),
            marker=dict(size=7),
            showlegend=False,
            name=name,
            hovertemplate=f'<b>{name}</b><br>Fold %{{x}}<br>Value: %{{y:.4g}}<extra></extra>',
        ), row=row, col=col)

    fig.update_layout(
        height=max(400, rows * 280),
        template=_TEMPLATE,
        title=dict(
            text='<b>Walk-Forward: Parameter Evolution Across Folds</b>',
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
        legend=dict(
            yanchor='top', y=0.99, xanchor='right', x=0.99,
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1,
        ),
    )

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved parameter evolution chart → {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Combined OOS equity curve (uses same layout as visualizer.plot_results)
# ──────────────────────────────────────────────────────────────────────────────

def plot_oos_equity(oos_metrics, oos_combined_df, fold_boundaries=None,
                   benchmark_data=None, show=True, save_html=None,
                   show_trades=False):
    """
    Full equity + drawdown chart for the stitched OOS period.
    Layout matches visualizer.plot_results exactly so output looks consistent.

    Parameters
    ----------
    oos_metrics     : dict returned by engine.backtest on combined OOS df
    oos_combined_df : the stitched OOS strategy dataframe
    fold_boundaries : list of dates to draw vertical fold-separation lines
    benchmark_data  : optional DataFrame with 'Close' for buy-and-hold comparison
    """
    # Clip equity at 0 — (1+r).cumprod() can go below 0 when a single bar has
    # r < -1 (e.g. a shorted memecoin pumps several hundred percent). In reality
    # liquidation would have ended the position at 0; the cumprod math doesn't
    # know that. Cosmetic only — engine math / oos_metrics numbers untouched.
    equity_curve = oos_metrics['equity_curve'].clip(lower=0)

    benchmark_equity = None
    if benchmark_data is not None:
        br = benchmark_data['Close'].pct_change()
        benchmark_equity = (1 + br).cumprod().fillna(1.0).clip(lower=0)
        # align to OOS window
        benchmark_equity = benchmark_equity.reindex(equity_curve.index, method='nearest')
        #renormalise
        benchmark_equity = benchmark_equity / benchmark_equity.iloc[0]
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            'OOS Equity Curve (Net of Costs)',
            'OOS Drawdown (Net of Costs)',
        ),
        vertical_spacing=0.08,
    )

    # equity
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        mode='lines', name='OOS Strategy',
        line=dict(color=_BLUE, width=2),
        hovertemplate='<b>OOS Strategy</b><br>Date: %{x}<br>Equity: %{y:.4f}<extra></extra>',
    ), row=1, col=1)

    if benchmark_equity is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_equity.index, y=benchmark_equity.values,
            mode='lines', name='Buy and Hold Benchmark',
            line=dict(color=_AMBER, width=2),
            hovertemplate='<b>Buy and Hold Benchmark</b><br>Date: %{x}<br>Equity: %{y:.4f}<extra></extra>',
        ), row=1, col=1)

    # drawdown
    running_max = equity_curve.cummax()
    drawdown    = (equity_curve - running_max) / running_max

    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        mode='lines', name='Drawdown',
        fill='tozeroy', fillcolor=_RED_FILL,
        line=dict(color=_RED, width=1),
        hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>%{y:.2f}%<extra></extra>',
        showlegend=False,
    ), row=2, col=1)

    # show charts
    if show_trades and oos_metrics.get('trades') is not None:
        trades = oos_metrics['trades']
        if len(trades) > 0:
            fig.add_trace(go.Scatter(
                x=trades['entry_time'], y=trades['entry_price'],
                mode='markers', name='Entry',
                marker=dict(symbol='triangle-up', size=9, color=_GREEN),
                hovertemplate='<b>Entry</b><br>%{x}<br>$%{y:.2f}<extra></extra>',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=trades['exit_time'], y=trades['exit_price'],
                mode='markers', name='Exit',
                marker=dict(symbol='triangle-down', size=9, color=_RED),
                hovertemplate='<b>Exit</b><br>%{x}<br>$%{y:.2f}<extra></extra>',
            ), row=1, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='#94a3b8', line_width=1, row=2, col=1)

    # fold boundary lines
    if fold_boundaries is not None and len(fold_boundaries) > 0:
        for date in fold_boundaries:
            for r in [1, 2]:
                fig.add_vline(
                    x=str(date), line_dash='dot',
                    line_color='rgba(100,116,139,0.5)', line_width=1,
                    row=r, col=1,
                )

    # metrics annotation (same style as visualizer.py)
    m = oos_metrics
    calmar = m['total_return'] / abs(m['max_drawdown']) if m['max_drawdown'] != 0 else 0
    main_text = (
        f"<b>OOS Performance</b><br>"
        f"Total Return: <b>{m['total_return']*100:.2f}%</b><br>"
        f"Sharpe Ratio: <b>{m['sharpe_ratio']:.2f}</b><br>"
        f"Max Drawdown: <b>{m['max_drawdown']*100:.2f}%</b><br>"
        f"Calmar Ratio: <b>{calmar:.2f}</b><br>"
        f"Profit Factor: <b>{m['profit_factor']:.2f}</b>"
    )
    trade_text = (
        f"<b>Trade Statistics</b><br>"
        f"Total Trades: <b>{m['num_trades']}</b><br>"
        f"Win Rate: <b>{m['win_rate']*100:.2f}%</b><br>"
        f"Avg Win/Loss: <b>{m['avg_win_loss_ratio']:.2f}</b>"
    )

    yearly_ret_text = '<b>Yearly Returns:</b><br>'
    for year, ret in sorted(m['yearly_returns'].items()):
        yearly_ret_text += f'{year}: <b>{ret*100:.2f}%</b><br>'

    yearly_sharpe_text = '<b>Yearly Sharpe:</b><br>'
    for year, sh in sorted(m['yearly_sharpe'].items()):
        yearly_sharpe_text += f'{year}: <b>{sh:.2f}</b><br>'

    for text, x, y in [
        (main_text,         0.01, 0.98),
        (yearly_ret_text,   0.01, 0.65),
        (yearly_sharpe_text,0.20, 0.65),
        (trade_text,        0.01, 0.35),
    ]:
        fig.add_annotation(
            xref='x domain', yref='y domain',
            x=x, y=y, xanchor='left', yanchor='top',
            text=text, showarrow=False,
            font=dict(size=10, family=_FONT_MONO),
            align='left',
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1, borderpad=8,
            row=1, col=1,
        )

    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified',
        template=_TEMPLATE,
        title=dict(
            text='<b>Walk-Forward: Combined OOS Results</b>',
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
        legend=dict(
            yanchor='top', y=0.99, xanchor='right', x=0.99,
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1,
        ),
    )

    fig.update_xaxes(title_text='Date', showgrid=True, gridwidth=1, gridcolor=_GRID, row=1, col=1)
    fig.update_xaxes(title_text='Date', showgrid=True, gridwidth=1, gridcolor=_GRID, row=2, col=1)
    fig.update_yaxes(title_text='Equity', showgrid=True, gridwidth=1, gridcolor=_GRID, row=1, col=1)
    fig.update_yaxes(title_text='Drawdown (%)', showgrid=True, gridwidth=1, gridcolor=_GRID, row=2, col=1)

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved OOS equity chart → {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Convenience wrapper — call this from notebooks
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
#  Plateau analysis — 1-D parameter sweep curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_plateau_analysis(sweep_results, consensus_params, param_defs,
                          fixed_params=None, threshold=0.20,
                          show=True, save_html=None):
    """
    Plotly grid of 1-D sensitivity sweeps from plateau_analysis().
    Each subplot shows score vs parameter value, with the consensus value
    marked and the plateau region (within `threshold` of peak) shaded.
    """
    if fixed_params is None:
        fixed_params = {}

    free_params = [k for k in param_defs if k not in fixed_params
                   if k in sweep_results]
    n = len(free_params)
    if n == 0:
        print('No sweep results to plot.')
        return None

    cols = 3
    rows = int(np.ceil(n / cols))

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=free_params,
        vertical_spacing=min(0.10, round(0.9 / max(rows - 1, 1), 3)),
        horizontal_spacing=0.08,
    )

    for idx, name in enumerate(free_params):
        r = idx // cols + 1
        c = idx %  cols + 1

        sdf = sweep_results[name].dropna(subset=['score'])
        if len(sdf) == 0:
            continue

        peak   = sdf['score'].max()
        cutoff = peak * (1 - threshold)

        # plateau shading — filled area where score >= cutoff
        above = sdf[sdf['score'] >= cutoff]
        if len(above) > 0:
            fig.add_trace(go.Scatter(
                x=pd.concat([above['value'], above['value'][::-1]]),
                y=pd.concat([above['score'], pd.Series([cutoff] * len(above))]),
                fill='toself',
                fillcolor='rgba(34, 197, 94, 0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=(idx == 0),
                name='Plateau region',
                hoverinfo='skip',
            ), row=r, col=c)

        # cutoff line
        fig.add_trace(go.Scatter(
            x=[sdf['value'].min(), sdf['value'].max()],
            y=[cutoff, cutoff],
            mode='lines',
            line=dict(color=_GREEN, dash='dot', width=1),
            showlegend=False,
            hoverinfo='skip',
        ), row=r, col=c)

        # score curve
        fig.add_trace(go.Scatter(
            x=sdf['value'], y=sdf['score'],
            mode='lines+markers',
            line=dict(color=_BLUE, width=2),
            marker=dict(size=5),
            showlegend=False,
            name=name,
            hovertemplate=(
                f'<b>{name}</b><br>'
                'Value: %{x:.4g}<br>'
                'Score: %{y:.4f}<br>'
                '<extra></extra>'
            ),
        ), row=r, col=c)

        # consensus value vertical line
        cv = consensus_params.get(name)
        if cv is not None:
            fig.add_trace(go.Scatter(
                x=[cv, cv],
                y=[sdf['score'].min() * 0.95, sdf['score'].max() * 1.02],
                mode='lines',
                line=dict(color=_RED, dash='dash', width=1.5),
                showlegend=(idx == 0),
                name='Consensus',
                hoverinfo='skip',
            ), row=r, col=c)

    fig.update_layout(
        height=max(400, rows * 300),
        template=_TEMPLATE,
        title=dict(
            text='<b>Plateau Analysis — 1-D Parameter Sweeps</b>',
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
        legend=dict(
            yanchor='top', y=0.99, xanchor='right', x=0.99,
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1,
        ),
    )

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=_GRID,
                             row=row, col=col)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=_GRID,
                             title_text='score', row=row, col=col)

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved plateau analysis chart → {save_html}')
    if show:
        fig.show()
    return fig

def plot_walk_forward_results(
    results,
    param_defs,
    fixed_params   = None,
    benchmark_data = None,
    show           = True,
    save_html_dir  = None,
    show_fold_perf  = True,
    show_param_evol = True,
    show_oos_equity = True,
    show_trades     = False,
):
    if fixed_params is None:
        fixed_params = {}

    results_df   = results['results_df']
    oos_metrics  = results['oos_metrics']
    oos_combined = results['oos_combined_df']

    def _html(name):
        if save_html_dir is None:
            return None
        import os
        os.makedirs(save_html_dir, exist_ok=True)
        return os.path.join(save_html_dir, name)

    if show_fold_perf:
        plot_fold_performance(
            results_df,
            show=show,
            save_html=_html('wf_fold_performance.html'),
        )

    if show_param_evol:
        plot_parameter_evolution(
            results_df, param_defs, fixed_params,
            show=show,
            save_html=_html('wf_parameter_evolution.html'),
        )

    if show_oos_equity:
        if oos_metrics is not None and oos_combined is not None:
            fold_boundaries = pd.to_datetime(results_df['test_start'].values)
            plot_oos_equity(
                oos_metrics, oos_combined,
                fold_boundaries=fold_boundaries,
                benchmark_data=benchmark_data,
                show=show,
                save_html=_html('wf_oos_equity.html'),
                show_trades=show_trades,
            )
        else:
            print('No valid combined OOS data — skipping equity chart.')

def plot_portfolio_oos(
    coin_dfs:    dict,
    weights:     dict = None,
    show_coins:  list = None,
    benchmark:   dict = None,
    cost:        float = 0.0,
    show:        bool  = True,
    save_html:   str   = None,
):
    """
    Stitch OOS dataframes from multiple coins into a combined portfolio
    and run through backtest() for consistent metrics + chart.

    Parameters
    ----------
    coin_dfs    : {'BTC': oos_df, 'ETH': oos_df, ...}
    weights     : {'BTC': 0.4, 'ETH': 0.3, ...} — auto equal-weights if None
    show_coins  : subset of coin_dfs keys to include — None = all
    benchmark   : {'BTC': oos_df} or multi-coin dict for B&H benchmark.
                  None = equal-weight B&H of show_coins.
                  Single coin e.g. {'BTC': btc_df} = BTC B&H only.
    cost        : per-leg trading cost fraction applied at each position
                  change (entry and exit separately), so effective
                  round-trip cost = 2 × cost.
                  e.g. cost=0.001 → 0.1% per leg → 0.2% round-trip.
                  Defaults to 0.0 because per-coin costs are typically
                  already embedded via walk_forward / _run_backtest.
                  Only set non-zero here to add a portfolio-level
                  rebalancing cost on top.
    """
    import sys, os
    # resolve backtester so backtest() is importable from here
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backtester'))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from engine import backtest

    # ── defaults ──────────────────────────────────────────────────────────────
    if show_coins is None:
        show_coins = list(coin_dfs.keys())

    raw_weights = {k: weights.get(k, 1/len(show_coins)) if weights else 1/len(show_coins)
                   for k in show_coins}
    total = sum(raw_weights.values())
    w = {k: v / total for k, v in raw_weights.items()}   # normalise

    # ── per-coin bar returns ──────────────────────────────────────────────────
    def _strat_ret(df):
        r    = df['Close'].pct_change().fillna(0)
        pos  = df['position'].shift(1).fillna(0)      if 'position'      in df.columns else pd.Series(1,   index=df.index)
        size = df['position_size'].shift(1).fillna(0) if 'position_size' in df.columns else pd.Series(1.0, index=df.index)
        # cost is per-leg: fires once on entry (0→1) and once on exit (1→0),
        # so total deduction per round-trip = 2 × cost. ✓ 2-way.
        trade_cost = df['position'].diff().abs().fillna(0) * cost if 'position' in df.columns else pd.Series(0.0, index=df.index)
        return r * pos * size - trade_cost

    aligned = pd.concat(
        [_strat_ret(coin_dfs[k]).rename(k) for k in show_coins],
        axis=1,
    ).fillna(0)

    combined_ret = sum(aligned[k] * w[k] for k in show_coins)

    # ── build synthetic OHLCV the engine can consume ──────────────────────────
    equity = (1 + combined_ret).cumprod()
    port_df = pd.DataFrame({
        'Close':         equity,
        'Open':          equity.shift(1).bfill(),
        'High':          equity,
        'Low':           equity,
        'Volume':        1.0,
        'position':      1,
        'position_size': 1.0,
        'stop_loss':     0.0,
    }, index=combined_ret.index)

    # ── benchmark ─────────────────────────────────────────────────────────────
    if benchmark is None:
        # equal-weight B&H of show_coins
        bh_aligned = pd.concat(
            [coin_dfs[k]['Close'].pct_change().fillna(0).rename(k) for k in show_coins],
            axis=1,
        ).fillna(0)
        bh_ret    = sum(bh_aligned[k] * w[k] for k in show_coins)
        bh_equity = (1 + bh_ret).cumprod()
        bench_df  = pd.DataFrame({'Close': bh_equity}, index=bh_equity.index)
    else:
        # caller supplied benchmark — single or multi-coin
        bench_coins = list(benchmark.keys())
        bw = {k: 1/len(bench_coins) for k in bench_coins}
        bh_aligned = pd.concat(
            [benchmark[k]['Close'].pct_change().fillna(0).rename(k) for k in bench_coins],
            axis=1,
        ).fillna(0)
        bh_ret    = sum(bh_aligned[k] * bw[k] for k in bench_coins)
        bh_equity = (1 + bh_ret).cumprod()
        bench_df  = pd.DataFrame({'Close': bh_equity}, index=bh_equity.index)

    # normalise benchmark to same start as portfolio
    # align tz before reindex — apply_scenario strips tz from coin_dfs but the
    # benchmark dict may still be tz-aware; mismatched tz → silent all-NaN reindex
    if bench_df.index.tz is not None and port_df.index.tz is None:
        bench_df.index = bench_df.index.tz_convert(None).normalize()
    elif bench_df.index.tz is None and port_df.index.tz is not None:
        bench_df.index = bench_df.index.tz_localize(port_df.index.tz)
    bench_df = bench_df.reindex(port_df.index, method='nearest')
    bench_df['Close'] = bench_df['Close'] / bench_df['Close'].iloc[0]

    # ── title suffix ──────────────────────────────────────────────────────────
    weight_str = '  |  '.join(f'{k} {round(w[k]*100, 1)}%' for k in show_coins)

    # ── compute per-coin trade stats ─────────────────────────────────────────
    # port_df always has position=1 so backtest() finds zero trade transitions.
    # compute trade stats directly from each coin's df and aggregate.
    import sys as _sys, os as _os
    _bt_dir = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', 'backtester'))
    if _bt_dir not in _sys.path:
        _sys.path.insert(0, _bt_dir)
    from performance_metrics import identify_trades
    from visualizer import plot_results as _plot_results

    coin_trade_stats = {}
    all_trades_list  = []
    for coin in show_coins:
        df_c = coin_dfs[coin].copy()
        df_c['position_change'] = df_c['position'].diff().abs().fillna(0)
        t = identify_trades(df_c)
        coin_trade_stats[coin] = t
        if len(t) > 0:
            all_trades_list.append(t)

    if all_trades_list:
        agg          = pd.concat(all_trades_list, ignore_index=True)
        n_trades     = len(agg)
        wins         = agg[agg['pnl'] > 0]
        losses       = agg[agg['pnl'] < 0]
        win_rate     = len(wins) / n_trades if n_trades else 0.0
        gross_profit = wins['pnl'].sum()
        gross_loss   = abs(losses['pnl'].sum())
        pf           = gross_profit / gross_loss if gross_loss > 0 else 0.0
        avg_w        = wins['pnl'].mean()        if len(wins)   > 0 else 0.0
        avg_l        = abs(losses['pnl'].mean()) if len(losses) > 0 else 0.0
        awl          = avg_w / avg_l             if avg_l       > 0 else 0.0
    else:
        n_trades = 0
        win_rate = pf = awl = 0.0

    # ── run backtest for equity / sharpe / drawdown (no chart yet) ───────────
    metrics = backtest(
        data           = port_df,
        cost           = cost,
        show_plot      = False,
        save_html      = None,
        show_trades    = False,
        benchmark_data = bench_df,
    )

    # override trade stats with the correctly aggregated values
    metrics['num_trades']         = n_trades
    metrics['win_rate']           = win_rate
    metrics['profit_factor']      = pf
    metrics['avg_win_loss_ratio'] = awl
    metrics['coin_trade_stats']   = coin_trade_stats

    # ── per-year trade statistics ─────────────────────────────────────────────
    yearly_trade_stats = {}
    if all_trades_list:
        agg['year'] = pd.to_datetime(agg['entry_time']).dt.year
        for yr, grp in agg.groupby('year'):
            n   = len(grp)
            w   = grp[grp['pnl'] > 0]
            l   = grp[grp['pnl'] < 0]
            wr  = len(w) / n if n else 0.0
            gp  = w['pnl'].sum()
            gl  = abs(l['pnl'].sum())
            pfy = gp / gl if gl > 0 else 0.0
            aw  = w['pnl'].mean()        if len(w) > 0 else 0.0
            al  = abs(l['pnl'].mean())   if len(l) > 0 else 0.0
            awly = aw / al              if al > 0 else 0.0
            yearly_trade_stats[yr] = {
                'trades': n, 'win_rate': wr,
                'profit_factor': pfy, 'avg_win_loss': awly,
            }
    metrics['yearly_trade_stats'] = yearly_trade_stats

    # render chart with corrected metrics
    if show or save_html:
        _plot_results(
            metrics        = metrics,
            benchmark_data = bench_df,
            show           = show,
            save_html      = save_html,
        )

    # ── print granular yearly trade table ────────────────────────────────────
    if yearly_trade_stats:
        print(f'\n── Yearly Trade Statistics ──')
        print(f'  {"Year":<6} {"Trades":>7} {"Win Rate":>10} {"Prof.Factor":>13} {"Avg W/L":>9}')
        print(f'  {"─"*6} {"─"*7} {"─"*10} {"─"*13} {"─"*9}')
        for yr in sorted(yearly_trade_stats):
            s = yearly_trade_stats[yr]
            pf_s  = f'{s["profit_factor"]:>13.2f}' if s["profit_factor"] > 0 else f'{"—":>13}'
            awl_s = f'{s["avg_win_loss"]:>9.2f}'   if s["avg_win_loss"]  > 0 else f'{"—":>9}'
            print(f'  {yr:<6} {s["trades"]:>7} {s["win_rate"]*100:>9.1f}% {pf_s} {awl_s}')

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
#  Closed-Trade (realized) equity curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_closed_trade_equity(
    position_dfs: dict,
    weights:      dict  = None,
    cost:         float = 0.001,
    bar_returns:  dict  = None,
    show:         bool  = True,
    save_html:    str   = None,
):
    """
    Step-function equity curve that only moves on **trade exits** — realized P&L,
    no intra-trade mark-to-market.  Read alongside the full MTM chart, not instead of it.

    Parameters
    ----------
    position_dfs : {label: df}
        Each df must have a ``'position'`` column for entry/exit detection.
        Bar returns are resolved in this priority order:
          1. ``bar_returns[label]`` if supplied
          2. ``df['net_returns']`` if the column exists  (stat-arb pkls)
          3. Recomputed via  ``Close.pct_change() * pos.shift(1) * size.shift(1) - cost * |pos.diff()|``
             (momentum pkls that carry Close / position / position_size)
    weights     : {label: float}  — auto-normalised; None = equal weight
    cost        : per-side transaction cost used only in path 3 above
    bar_returns : {label: pd.Series}  — optional pre-computed per-bar returns (path 1)
    show        : render Plotly chart inline
    save_html   : optional file path to write chart HTML

    Returns
    -------
    closed_eq : pd.Series  indexed by exit timestamps, starting at 1.0
    """
    if not position_dfs:
        print('  [closed-trade] no sleeves supplied — skipping.')
        return pd.Series([1.0])

    # normalise weights
    n = len(position_dfs)
    w = {k: v / sum(weights.values()) for k, v in weights.items()} \
        if weights else {k: 1 / n for k in position_dfs}

    def _bar(label, df):
        if bar_returns is not None and label in bar_returns:
            return bar_returns[label]
        if 'net_returns' in df.columns:
            return df['net_returns'].fillna(0)
        r    = df['Close'].pct_change().fillna(0)
        pos  = df['position'].shift(1).fillna(0)
        size = df['position_size'].shift(1).fillna(0) if 'position_size' in df.columns \
               else pd.Series(1.0, index=df.index)
        to   = df['position'].diff().abs().fillna(0)
        return r * pos * size - cost * to

    trades = []
    for label, df in position_dfs.items():
        pos  = df['position'].fillna(0)
        prev = pos.shift(1).fillna(0)
        entry = (prev == 0) & (pos != 0)
        exit_ = (prev != 0) & ((pos == 0) | (pos != prev))
        if pos.iloc[0] != 0:
            entry.iloc[0] = True

        bar     = _bar(label, df)
        entries = list(df.index[entry])
        exits   = list(df.index[exit_])
        wt      = w.get(label, 0)

        for e in entries:
            nxt = [x for x in exits if x > e]
            if not nxt:
                break
            x = nxt[0]
            trades.append((x, wt * ((1 + bar.loc[e:x]).prod() - 1)))

    if not trades:
        closed_eq = pd.Series([1.0])
    else:
        all_times = sorted({t for t, _ in trades})
        deltas    = pd.Series(0.0, index=pd.Index(all_times))
        for t, r in trades:
            deltas.loc[t] += r
        _start    = min(df.index[0] for df in position_dfs.values())
        closed_eq = pd.concat([pd.Series([1.0], index=[_start]), (1 + deltas).cumprod()])

    realized = (closed_eq.iloc[-1] - 1) * 100 if len(closed_eq) else 0.0
    print(f'  {len(trades)} trade exits  |  realized return: {realized:.2f}%')
    print('  Hides intra-trade MTM drawdowns — read alongside the MTM chart above.')

    if show or save_html:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=closed_eq.index, y=closed_eq.values,
            mode='lines', line_shape='hv',
            name='Closed-trade equity',
            line=dict(width=2, color=_BLUE),
            hovertemplate='<b>Realized equity</b><br>%{x}<br>%{y:.4f}x<extra></extra>',
        ))
        fig.update_layout(
            title=f'<b>Closed-Trade Equity</b>  —  realized P&L only  ({len(trades)} exits)',
            xaxis_title='Date',
            yaxis_title='Equity (1.0 = start)',
            template=_TEMPLATE,
            height=420,
            hovermode='x unified',
        )
        if save_html:
            fig.write_html(save_html)
            print(f'✓ Saved closed-trade chart → {save_html}')
        if show:
            fig.show()

    return closed_eq