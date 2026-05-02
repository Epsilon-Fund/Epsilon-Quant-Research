"""
CPCV Visualizer
──────────────────────────────────────────────────────────────────────────────
Plotly visualisation suite for Combinatorial Purged Cross-Validation results.
Companion to infrastructure/walkforward/cpcv_engine.py.

Identical colour palette, template, annotation conventions, and figure layout
as wf_visualizer.py so all outputs look like one visual family.

Main entry point
----------------
    from cpcv_visualizer import plot_cpcv_results

    plot_cpcv_results(cpcv_results, analysis=analysis, wf_sharpe=1.4)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── shared style constants (mirrors wf_visualizer.py exactly) ─────────────────
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


# ── internal helpers ───────────────────────────────────────────────────────────

def _infer_ppy(index):
    """Estimate periods-per-year from a DatetimeIndex."""
    if len(index) < 2:
        return 252
    deltas = pd.Series(index.to_list()).diff().dropna()
    median_days = deltas.median().days if hasattr(deltas.median(), 'days') else float(deltas.median()) / 86_400_000_000_000
    if median_days <= 1.5:
        return 252
    if median_days <= 8:
        return 52
    if median_days <= 35:
        return 12
    return 252


def _yearly_metrics_from_curve(curve, ppy=252):
    """
    Compute per-year return, Sharpe, and max DD from an equity curve Series.
    Returns three dicts keyed by integer year.
    """
    if curve is None or len(curve) < 2:
        return {}, {}, {}
    returns = curve.pct_change().dropna()
    yr_ret, yr_sh, yr_dd = {}, {}, {}
    for y in sorted(returns.index.year.unique()):
        yr_r = returns[returns.index.year == y]
        c_yr = curve[curve.index.year == y]
        if len(yr_r) < 2 or len(c_yr) < 2:
            continue
        yr_ret[y] = float((c_yr.iloc[-1] - c_yr.iloc[0]) / c_yr.iloc[0])
        yr_sh[y]  = float(yr_r.mean() / yr_r.std() * np.sqrt(ppy)) if yr_r.std() > 0 else 0.0
        run_max   = c_yr.cummax()
        yr_dd[y]  = float(((c_yr - run_max) / run_max).min())
    return yr_ret, yr_sh, yr_dd


def _aggregate_trade_stats(cpcv_results):
    """
    Aggregate n_trades, win_rate, and profit_factor across all OOS group
    evaluations in split_results.  win_rate is weighted by trade count.
    profit_factor is the simple mean of per-group values (best approximation
    without access to raw trade PnL).
    Returns (total_trades, win_rate, profit_factor) — nan when unavailable.
    """
    total_trades = 0
    weighted_wr  = 0.0
    pf_vals      = []
    for sr in cpcv_results.get('split_results', []):
        for gr in sr.get('group_results', {}).values():
            m = (gr or {}).get('metrics') or {}
            nt = m.get('n_trades') or 0
            if nt > 0:
                total_trades += nt
                weighted_wr  += nt * (m.get('win_rate') or 0.0)
            pf = m.get('profit_factor')
            if pf is not None and np.isfinite(pf):
                pf_vals.append(pf)
    win_rate      = weighted_wr / total_trades if total_trades > 0 else float('nan')
    profit_factor = float(np.mean(pf_vals)) if pf_vals else float('nan')
    return total_trades, win_rate, profit_factor


def _split_oos_sharpes(cpcv_results):
    """Return per-split OOS Sharpe as a float array (mean across k test groups)."""
    oos = []
    for sr in cpcv_results['split_results']:
        vals = [
            gr['metrics']['sharpe']
            for gr in sr['group_results'].values()
            if gr['metrics'] and gr['metrics']['sharpe'] is not None
        ]
        oos.append(float(np.mean(vals)) if vals else float('nan'))
    return np.array(oos, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
#  Function 1: plot_path_equity_curves
# ──────────────────────────────────────────────────────────────────────────────

def _downsample(arr, max_pts: int):
    """Return a thinned copy of arr with at most max_pts elements."""
    if len(arr) <= max_pts:
        return arr
    step = max(1, len(arr) // max_pts)
    return arr[::step]


def plot_path_equity_curves(cpcv_results, wf_sharpe=None, title=None,
                            show=True, save_html=None,
                            max_paths: int = 30,
                            max_pts:   int = 1500):
    """
    CPCV path equity curves with a drawdown panel below — matching the layout
    and annotation style of infrastructure/backtester/visualizer.py.

    Row 1 (equity, 70 %):
      - Min–max envelope band, all individual paths (semi-transparent), mean path.
      - Group boundary dotted vlines with G1…GN labels.
      - Annotation boxes: Portfolio Performance, Yearly Returns, Yearly Sharpe,
        OOS Trade Stats (aggregated across all group evaluations).

    Row 2 (drawdown of mean path, 30 %):
      - Red filled drawdown of the mean equity curve.
      - Yearly Max DD box (top-right).

    All statistics are computed from the mean path equity curve or aggregated
    across split group results — no single-path cherry-picking.
    """
    paths   = cpcv_results['paths']
    bounds  = cpcv_results['group_boundaries']
    config  = cpcv_results['config']
    N, k    = config['N'], config['k']

    valid = [p for p in paths if p['equity_curve'] is not None]
    if not valid:
        print('[plot_path_equity_curves] No valid equity curves — nothing to plot.')
        return None

    # ── sample paths for rendering (stats still use all valid paths) ──────────
    plot_paths = valid
    if len(valid) > max_paths:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(valid), size=max_paths, replace=False)
        idx.sort()
        plot_paths = [valid[i] for i in idx]

    # ── align sampled curves to a common sorted index ─────────────────────────
    common_idx = plot_paths[0]['equity_curve'].index
    for p in plot_paths[1:]:
        common_idx = common_idx.union(p['equity_curve'].index)
    common_idx = common_idx.sort_values()

    aligned = pd.DataFrame(index=common_idx)
    for p in plot_paths:
        aligned[p['path_id']] = p['equity_curve'].reindex(common_idx).ffill()

    # full-resolution curves — used for annotation stats and yearly breakdowns
    mean_curve = aligned.mean(axis=1)
    min_curve  = aligned.min(axis=1)
    max_curve  = aligned.max(axis=1)

    # ── downsample time axis for all rendered traces ──────────────────────────
    ds_idx        = _downsample(common_idx, max_pts)
    aligned_ds    = aligned.reindex(ds_idx).ffill()
    mean_curve_ds = mean_curve.reindex(ds_idx).ffill()
    min_curve_ds  = min_curve.reindex(ds_idx).ffill()
    max_curve_ds  = max_curve.reindex(ds_idx).ffill()

    x_fwd = list(ds_idx)
    x_rev = list(ds_idx[::-1])

    # ── annotation stats: sourced from path distribution (matches histogram) ────
    path_sharpes = [p['sharpe']       for p in valid if p.get('sharpe')       is not None]
    path_calmars = [p['calmar']       for p in valid if p.get('calmar')       is not None]
    path_maxdds  = [p['max_dd']       for p in valid if p.get('max_dd')       is not None]
    path_rets    = [p['total_return'] for p in valid if p.get('total_return') is not None]
    sharpe  = float(np.mean(path_sharpes)) if path_sharpes else 0.0
    calmar  = float(np.mean(path_calmars)) if path_calmars else 0.0
    max_dd  = float(np.mean(path_maxdds))  if path_maxdds  else 0.0
    tot_ret = float(np.mean(path_rets))    if path_rets    else 0.0

    # CAGR from mean equity curve — for display alongside total return
    n_years = (common_idx[-1] - common_idx[0]).days / 365.25
    cagr    = float((1 + tot_ret) ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0

    # ── mean-path curve (for drawdown panel and yearly breakdowns only) ────────
    ppy      = _infer_ppy(common_idx)
    yr_ret, yr_sh, yr_dd = _yearly_metrics_from_curve(mean_curve, ppy)
    total_trades, win_rate, profit_factor = _aggregate_trade_stats(cpcv_results)

    # ── 2-row subplot ─────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.06,
        shared_xaxes=True,
    )

    # ── Row 1: envelope band ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fwd + x_rev,
        y=list(max_curve_ds.values) + list(min_curve_ds.values[::-1]),
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Min–Max Envelope',
        hoverinfo='skip',
    ), row=1, col=1)

    # ── Row 1: individual paths (sampled subset) ──────────────────────────────
    for i, p in enumerate(plot_paths):
        fig.add_trace(go.Scatter(
            x=x_fwd, y=aligned_ds[p['path_id']].values,
            mode='lines',
            line=dict(color=_BLUE, width=0.7),
            opacity=0.20,
            name='Paths',
            showlegend=(i == 0),
            hoverinfo='skip',
        ), row=1, col=1)

    # ── Row 1: mean path ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fwd, y=mean_curve_ds.values,
        mode='lines',
        line=dict(color=_AMBER, width=2.5),
        name='Mean Path',
        hovertemplate='<b>Mean Path</b><br>%{x}<br>Equity: %{y:.4f}<extra></extra>',
    ), row=1, col=1)

    # ── Row 2: mean path drawdown ─────────────────────────────────────────────
    run_max_ds = mean_curve_ds.cummax()
    dd_pct_ds  = (mean_curve_ds - run_max_ds) / run_max_ds
    fig.add_trace(go.Scatter(
        x=x_fwd, y=(dd_pct_ds.values * 100),
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        fillcolor=_RED_FILL,
        line=dict(color=_RED, width=1),
        hovertemplate='<b>Drawdown</b><br>%{x}<br>%{y:.2f}%<extra></extra>',
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash='dash', line_color='#94a3b8',
                  line_width=1, row=2, col=1)

    # ── group boundary vlines + labels (span full height, label on row 1) ────
    for i, (start, _) in enumerate(bounds):
        fig.add_vline(
            x=str(start),
            line_dash='dot',
            line_color='rgba(100,116,139,0.55)',
            line_width=1.5,
        )
        fig.add_annotation(
            x=str(start), xref='x',
            y=1.0,         yref='y domain',
            text=f'G{i+1}',
            showarrow=False,
            font=dict(size=9, color=_GREY, family=_FONT_MONO),
            xanchor='left', yanchor='bottom',
            row=1, col=1,
        )

    # ── annotation box 1: Portfolio Performance (top-left, row 1) ────────────
    date_start = pd.Timestamp(common_idx[0]).strftime('%Y-%m-%d')
    date_end   = pd.Timestamp(common_idx[-1]).strftime('%Y-%m-%d')
    perf_lines = [
        '<b>Portfolio Performance</b>',
        f'Period:        <b>{date_start} → {date_end}</b>',
        f'CAGR:          <b>{cagr*100:.2f}%</b>',
        f'Total Return:  <b>{tot_ret*100:.2f}%</b>',
        f'Sharpe Ratio:  <b>{sharpe:.2f}</b>',
        f'Max Drawdown:  <b>{max_dd*100:.2f}%</b>',
        f'Calmar Ratio:  <b>{calmar:.2f}</b>',
        f'N Paths:       <b>{len(valid)}</b> (showing {len(plot_paths)})',
    ]
    if wf_sharpe is not None:
        perf_lines.append(f'WF Sharpe:     <b>{wf_sharpe:.2f}</b>')
    fig.add_annotation(
        xref='x domain', yref='y domain',
        x=0.01, y=0.99, xanchor='left', yanchor='top',
        text='<br>'.join(perf_lines),
        showarrow=False,
        font=dict(size=10, family=_FONT_MONO),
        align='left',
        bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1, borderpad=8,
        row=1, col=1,
    )

    # ── annotation box 2: Yearly Returns (mid-left, row 1) ───────────────────
    if yr_ret:
        yr_ret_txt = '<b>Yearly Returns</b><br>' + '<br>'.join(
            f'{y}:  <b>{r*100:+.2f}%</b>' for y, r in sorted(yr_ret.items())
        )
        fig.add_annotation(
            xref='x domain', yref='y domain',
            x=0.01, y=0.60, xanchor='left', yanchor='top',
            text=yr_ret_txt,
            showarrow=False,
            font=dict(size=9, family=_FONT_MONO),
            align='left',
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1, borderpad=8,
            row=1, col=1,
        )

    # ── annotation box 3: Yearly Sharpe (mid, row 1) ─────────────────────────
    if yr_sh:
        yr_sh_txt = '<b>Yearly Sharpe</b><br>' + '<br>'.join(
            f'{y}:  <b>{s:.2f}</b>' for y, s in sorted(yr_sh.items())
        )
        fig.add_annotation(
            xref='x domain', yref='y domain',
            x=0.22, y=0.60, xanchor='left', yanchor='top',
            text=yr_sh_txt,
            showarrow=False,
            font=dict(size=9, family=_FONT_MONO),
            align='left',
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1, borderpad=8,
            row=1, col=1,
        )

    # ── annotation box 4: Trade Stats (low-left, row 1) ──────────────────────
    if total_trades > 0:
        trade_lines = [
            '<b>OOS Trade Stats</b>',
            f'Total Trades:  <b>{total_trades}</b>',
            f'Win Rate:      <b>{win_rate*100:.2f}%</b>',
        ]
        if np.isfinite(profit_factor):
            trade_lines.append(f'Profit Factor: <b>{profit_factor:.2f}</b>')
        fig.add_annotation(
            xref='x domain', yref='y domain',
            x=0.01, y=0.32, xanchor='left', yanchor='top',
            text='<br>'.join(trade_lines),
            showarrow=False,
            font=dict(size=10, family=_FONT_MONO),
            align='left',
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1, borderpad=8,
            row=1, col=1,
        )

    # ── annotation box 5: Yearly Max DD (top-right, row 2) ───────────────────
    if yr_dd:
        yr_dd_txt = '<b>Yearly Max DD</b><br>' + '<br>'.join(
            f'{y}:  <b>{d*100:.2f}%</b>' for y, d in sorted(yr_dd.items())
        )
        fig.add_annotation(
            xref='x2 domain', yref='y2 domain',
            x=0.99, y=0.98, xanchor='right', yanchor='top',
            text=yr_dd_txt,
            showarrow=False,
            font=dict(size=9, family=_FONT_MONO),
            align='left',
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1, borderpad=8,
            row=2, col=1,
        )

    # ── layout ────────────────────────────────────────────────────────────────
    plot_title = title or (
        f'<b>CPCV Path Equity Curves</b>'
        f'  —  {len(valid)} paths  (N={N}, k={k})'
    )
    fig.update_layout(
        height=900,
        template=_TEMPLATE,
        title=dict(text=plot_title, font=dict(size=22, color='#1E293B'),
                   x=0.5, xanchor='center'),
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99,
                    bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1),
        hovermode='x unified',
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=_GRID,
                     dtick='M12', tickformat='%Y', row=1, col=1)
    fig.update_xaxes(title_text='Date', showgrid=True, gridwidth=1,
                     gridcolor=_GRID, dtick='M12', tickformat='%Y', row=2, col=1)
    fig.update_yaxes(title_text='Equity (1.0 = start)', showgrid=True,
                     gridwidth=1, gridcolor=_GRID, row=1, col=1)
    fig.update_yaxes(title_text='Drawdown (%)', showgrid=True,
                     gridwidth=1, gridcolor=_GRID, row=2, col=1)

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved path equity curves → {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Function 2: plot_path_distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_path_distribution(cpcv_results, wf_sharpe=None, ci_results=None,
                           show=True, save_html=None):
    """
    Histogram of path-level Sharpe ratios (one value per complete OOS path).

    - 15 bins regardless of path count (wide enough to read clearly).
    - Solid amber vertical line at the mean.
    - If wf_sharpe is provided: solid grey vertical line for the WF baseline.
    - If ci_results is provided (output of cpcv_confidence_intervals):
        - Adjusted CI: semi-transparent blue shaded band labelled in the legend
          as "Adjusted XX% CI (N_eff=YY.Y)". Only the overlap-adjusted interval
          is shown — the naive interval is anticonservative and omitted.
        - Dashed red vertical line at the conservative lower bound.
        - Annotation box: N paths, N effective, Mean Sharpe, adjusted CI.
    - Without ci_results the annotation box shows mean, std, min, max.
    """
    paths = cpcv_results['paths']
    sharpes = np.array([p['sharpe'] for p in paths if p['sharpe'] is not None],
                       dtype=float)

    if len(sharpes) == 0:
        print('[plot_path_distribution] No valid path Sharpes — nothing to plot.')
        return None

    mean_sh = float(np.mean(sharpes))
    std_sh  = float(np.std(sharpes))
    min_sh  = float(np.min(sharpes))
    max_sh  = float(np.max(sharpes))

    _CI_FILL  = 'rgba(59, 130, 246, 0.18)'   # muted blue, low opacity
    _CI_LINE  = 'rgba(59, 130, 246, 0.60)'   # same blue, stronger border

    fig = go.Figure()

    # ── adjusted CI shaded band + legend entry (drawn below bars) ─────────────
    if ci_results is not None:
        sh_ci = ci_results.get('sharpe')
        if sh_ci is not None:
            lo_a, hi_a = sh_ci['adjusted_ci']
            n_eff       = sh_ci['n_effective']
            conf_pct    = int(round(ci_results.get('confidence', 0.95) * 100))
            ci_label    = f'Adjusted {conf_pct}% CI (N_eff={n_eff:.1f})'

            # add_shape instead of add_vrect: Plotly 6.x add_vrect sets
            # yref='y domain' which breaks full-height coverage; add_shape
            # with yref='paper' correctly fills only the plot area.
            fig.add_shape(
                type='rect',
                x0=lo_a, x1=hi_a,
                y0=0, y1=1,
                xref='x', yref='paper',
                fillcolor=_CI_FILL,
                line=dict(color=_CI_LINE, width=1),
                layer='below',
            )
            # dummy invisible scatter so the CI appears in the legend
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=12, color=_CI_FILL,
                            symbol='square', line=dict(color=_CI_LINE, width=1)),
                name=ci_label,
                showlegend=True,
            ))


    # ── histogram ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Histogram(
        x=sharpes,
        nbinsx=15,
        marker_color=_BLUE,
        opacity=0.75,
        name='Path Sharpe',
        hovertemplate='Sharpe: %{x:.2f}<br>Count: %{y}<extra></extra>',
    ))

    # ── mean — solid amber vertical line ──────────────────────────────────────
    fig.add_vline(
        x=mean_sh,
        line_dash='solid', line_color=_AMBER, line_width=2,
        annotation=dict(
            text=f'Mean: {mean_sh:.2f}',
            font=dict(size=10, family=_FONT_MONO, color=_AMBER),
            bgcolor=_BG_BOX, bordercolor=_AMBER, borderwidth=1, borderpad=4,
        ),
        annotation_position='top right',
    )

    # ── WF Sharpe baseline — solid grey vertical line ─────────────────────────
    if wf_sharpe is not None:
        fig.add_vline(
            x=wf_sharpe,
            line_dash='solid', line_color=_GREY, line_width=2,
            annotation=dict(
                text=f'WF: {wf_sharpe:.2f}',
                font=dict(size=10, family=_FONT_MONO, color=_GREY),
                bgcolor=_BG_BOX, bordercolor=_GREY, borderwidth=1, borderpad=4,
            ),
            annotation_position='top left',
        )

    # ── annotation box ────────────────────────────────────────────────────────
    if ci_results is not None and ci_results.get('sharpe') is not None:
        sh_ci      = ci_results['sharpe']
        lo_a, hi_a = sh_ci['adjusted_ci']
        conf_pct   = int(round(ci_results.get('confidence', 0.95) * 100))
        ann = (
            f'<b>Path Distribution</b><br>'
            f'N paths:     <b>{sh_ci["n_paths"]}</b><br>'
            f'N effective: <b>{sh_ci["n_effective"]:.1f}</b><br>'
            f'Mean Sharpe: <b>{mean_sh:.2f}</b><br>'
            f'Adj {conf_pct}% CI:  <b>[{lo_a:.2f}, {hi_a:.2f}]</b>'
        )
    else:
        ann = (
            f'<b>Path Distribution</b><br>'
            f'N paths: <b>{len(sharpes)}</b><br>'
            f'Mean:    <b>{mean_sh:.2f}</b><br>'
            f'Std:     <b>{std_sh:.2f}</b><br>'
            f'Min:     <b>{min_sh:.2f}</b><br>'
            f'Max:     <b>{max_sh:.2f}</b>'
        )
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.99, y=0.99, xanchor='right', yanchor='top',
        text=ann, showarrow=False,
        font=dict(size=10, family=_FONT_MONO),
        align='left',
        bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1, borderpad=8,
    )

    config = cpcv_results['config']
    fig.update_layout(
        height=480,
        template=_TEMPLATE,
        title=dict(
            text=(f'<b>CPCV Path Sharpe Distribution</b>'
                  f'  —  {len(sharpes)} paths  '
                  f'(N={config["N"]}, k={config["k"]})'),
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
        legend=dict(yanchor='bottom', y=0.01, xanchor='left', x=0.01,
                    bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1),
        xaxis=dict(title='Sharpe Ratio', showgrid=True,
                   gridwidth=1, gridcolor=_GRID),
        yaxis=dict(title='Count', showgrid=True, gridwidth=1, gridcolor=_GRID),
        bargap=0.05,
    )

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved path distribution → {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Function 3: plot_parameter_distributions
# ──────────────────────────────────────────────────────────────────────────────

def plot_parameter_distributions(cpcv_results, analysis=None,
                                 show=True, save_html=None):
    """
    Grid of strip plots — one subplot per free parameter.

    Each subplot shows:
    - Jittered dots of the N_splits optimised values, coloured by OOS Sharpe
      (diverging RdYlGn: red = low Sharpe, green = high Sharpe).
    - IQR shaded region.
    - Median horizontal line.
    - If analysis is provided: verdict (Stable / Moderate / Scattered) and
      recommended action as subplot subtitle.
    """
    param_dist   = cpcv_results['param_distributions']
    free_params  = list(param_dist.columns)
    n            = len(free_params)

    if n == 0:
        print('[plot_parameter_distributions] No free parameters.')
        return None

    oos_sharpes = _split_oos_sharpes(cpcv_results)
    sharpe_min  = float(np.nanmin(oos_sharpes)) if not np.all(np.isnan(oos_sharpes)) else -1.0
    sharpe_max  = float(np.nanmax(oos_sharpes)) if not np.all(np.isnan(oos_sharpes)) else 1.0
    if sharpe_min == sharpe_max:
        sharpe_min -= 0.5
        sharpe_max += 0.5

    cols = 3
    rows = int(np.ceil(n / cols))

    subplot_titles = []
    for p in free_params:
        if analysis is not None:
            ds = analysis['distribution_stats']
            cr = analysis['consensus_ranges']
            cv = float(ds.loc[p, 'cv']) if p in ds.index else float('nan')
            if not np.isnan(cv):
                verdict = 'Stable' if cv < 0.10 else ('Moderate' if cv < 0.25 else 'Scattered')
            else:
                verdict = ''
            action = cr.loc[p, 'action'] if p in cr.index else ''
            subplot_titles.append(f'<b>{p}</b><br><i>{verdict} — {action}</i>')
        else:
            subplot_titles.append(f'<b>{p}</b>')

    v_spacing = min(0.15, round(1.0 / max(rows, 1), 3))
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=v_spacing,
        horizontal_spacing=0.08,
    )

    rng = np.random.default_rng(42)

    for idx, p in enumerate(free_params):
        r = idx // cols + 1
        c = idx %  cols + 1

        vals = param_dist[p].values.astype(float)
        valid_vals = vals[~np.isnan(vals)]
        if len(valid_vals) == 0:
            continue

        median = float(np.median(valid_vals))
        q25    = float(np.percentile(valid_vals, 25))
        q75    = float(np.percentile(valid_vals, 75))

        jitter = rng.uniform(-0.30, 0.30, len(vals))

        # IQR shaded rectangle
        fig.add_shape(
            type='rect',
            x0=-0.55, x1=0.55, y0=q25, y1=q75,
            fillcolor='rgba(59, 130, 246, 0.12)',
            line=dict(color='rgba(0,0,0,0)'),
            row=r, col=c,
        )

        # median line
        fig.add_shape(
            type='line',
            x0=-0.45, x1=0.45, y0=median, y1=median,
            line=dict(color=_AMBER, width=2, dash='dash'),
            row=r, col=c,
        )

        # dots coloured by OOS Sharpe
        fig.add_trace(go.Scatter(
            x=jitter, y=vals,
            mode='markers',
            marker=dict(
                size=8,
                color=oos_sharpes,
                colorscale='RdYlGn',
                showscale=(idx == 0),
                colorbar=dict(
                    title=dict(text='OOS<br>Sharpe', font=dict(size=11)),
                    len=0.5, x=1.03, thickness=14,
                ) if idx == 0 else None,
                cmin=sharpe_min,
                cmax=sharpe_max,
                line=dict(color='white', width=0.5),
            ),
            showlegend=False,
            hovertemplate=(
                f'<b>{p}</b><br>'
                'Value: %{y:.4g}<br>'
                'OOS Sharpe: %{marker.color:.2f}'
                '<extra></extra>'
            ),
        ), row=r, col=c)

        fig.update_xaxes(
            showticklabels=False, showgrid=False, zeroline=False,
            range=[-0.65, 0.65], row=r, col=c,
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor=_GRID, row=r, col=c,
        )

    n_splits = cpcv_results['n_splits']
    fig.update_layout(
        height=max(420, rows * 310),
        template=_TEMPLATE,
        title=dict(
            text=f'<b>CPCV Parameter Distributions</b>  —  {n_splits} splits',
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99,
                    bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1),
        margin=dict(r=80),
    )

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved parameter distributions → {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Function 4: plot_parameter_correlation_matrix
# ──────────────────────────────────────────────────────────────────────────────

def plot_parameter_correlation_matrix(analysis, show=True, save_html=None):
    """
    Annotated heatmap of cross-parameter Pearson correlations.

    - Diverging RdBu colourmap centred at 0.
    - Lower triangle only (upper triangle left blank).
    - Cell text shows the correlation value to 2 d.p.
    """
    corr   = analysis['cross_param_corr'].copy()
    params = list(corr.columns)
    n      = len(params)

    if n < 2:
        print('[plot_parameter_correlation_matrix] Need ≥ 2 free parameters.')
        return None

    # lower triangle: keep i >= j, blank i < j
    z    = corr.values.astype(float)
    text = []
    for i in range(n):
        row_text = []
        for j in range(n):
            if j > i:
                z[i, j] = float('nan')
                row_text.append('')
            else:
                v = z[i, j]
                row_text.append(f'{v:.2f}' if not np.isnan(v) else '')
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=params,
        y=params,
        colorscale='RdBu',
        zmid=0,
        zmin=-1, zmax=1,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=10, family=_FONT_MONO),
        hovertemplate='<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>',
        colorbar=dict(
            title=dict(text='Pearson r', font=dict(size=12)),
            len=0.8, thickness=16,
        ),
    ))

    fig.update_layout(
        height=max(420, n * 55 + 160),
        template=_TEMPLATE,
        title=dict(
            text='<b>CPCV Parameter Correlation Matrix</b>',
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
        xaxis=dict(side='bottom', tickangle=-40,
                   showgrid=False, zeroline=False),
        yaxis=dict(autorange='reversed', showgrid=False, zeroline=False),
    )

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved correlation matrix → {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Function 5: plot_split_performance_heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_split_performance_heatmap(cpcv_results, show=True, save_html=None):
    """
    N × N heatmap where cell (i, j) with i < j shows the OOS Sharpe of the
    CPCV split that held out groups i and j (k=2 only).

    Diverging RdYlGn colourmap centred at 0: red = negative, green = positive.
    Cells where i >= j are blank.
    """
    config        = cpcv_results['config']
    split_results = cpcv_results['split_results']
    N, k          = config['N'], config['k']

    if k != 2:
        print(f'[plot_split_performance_heatmap] Heatmap requires k=2 '
              f'(current k={k}). Skipping.')
        return None

    # per-split OOS Sharpe (mean of the two group Sharpes)
    split_sh = {}
    for sr in split_results:
        sharpes = [
            gr['metrics']['sharpe']
            for gr in sr['group_results'].values()
            if gr['metrics'] and gr['metrics']['sharpe'] is not None
        ]
        split_sh[sr['split_id']] = float(np.mean(sharpes)) if sharpes else float('nan')

    # build upper-triangle N×N matrix
    z = np.full((N, N), float('nan'))
    for sr in split_results:
        gi, gj = sr['test_group_indices']
        i, j   = (gi, gj) if gi < gj else (gj, gi)
        z[i, j] = split_sh[sr['split_id']]

    labels = [f'G{i+1}' for i in range(N)]

    text = []
    for i in range(N):
        row_text = []
        for j in range(N):
            v = z[i, j]
            row_text.append(f'{v:.2f}' if not np.isnan(v) else '')
        text.append(row_text)

    valid = z[~np.isnan(z)]
    vmax  = max(abs(float(np.nanmin(valid))), abs(float(np.nanmax(valid)))) \
            if len(valid) > 0 else 1.0

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-vmax, zmax=vmax,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=11, family=_FONT_MONO),
        hovertemplate=(
            'Held-out: <b>%{y}</b> + <b>%{x}</b><br>'
            'OOS Sharpe: %{z:.3f}'
            '<extra></extra>'
        ),
        colorbar=dict(
            title=dict(text='OOS<br>Sharpe', font=dict(size=12)),
            len=0.8, thickness=16,
        ),
    ))

    fig.update_layout(
        height=max(420, N * 55 + 160),
        template=_TEMPLATE,
        title=dict(
            text='<b>CPCV Split Performance Heatmap</b>  —  OOS Sharpe by Group Pair',
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
        xaxis=dict(
            title='Held-out Group (j)',
            side='top',
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            title='Held-out Group (i)',
            autorange='reversed',
            showgrid=False, zeroline=False,
        ),
    )

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved split performance heatmap → {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Function 6: plot_tercile_comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_tercile_comparison(cpcv_results, analysis, show=True, save_html=None):
    """
    Grid of box plots — one subplot per free parameter.

    Each subplot shows parameter values from top-tercile splits (high OOS Sharpe,
    green) vs bottom-tercile splits (low OOS Sharpe, red). The separation score
    from cpcv_parameter_analysis is shown in the subplot subtitle.
    """
    param_dist    = cpcv_results['param_distributions']
    free_params   = list(param_dist.columns)
    n             = len(free_params)
    tercile_comp  = analysis.get('tercile_comparison', {})

    if n == 0 or not tercile_comp:
        print('[plot_tercile_comparison] No tercile data — nothing to plot.')
        return None

    oos_sharpes = _split_oos_sharpes(cpcv_results)
    valid_mask  = ~np.isnan(oos_sharpes)
    n_valid     = int(valid_mask.sum())
    # Use valid-split count so each tercile is exactly 1/3 of successful splits.
    n_tercile   = max(1, n_valid // 3)

    if n_valid < 3:
        print('[plot_tercile_comparison] Fewer than 3 valid splits — cannot split terciles.')
        return None

    valid_idx  = np.where(valid_mask)[0]
    sorted_pos = np.argsort(oos_sharpes[valid_idx])
    top_idx    = valid_idx[sorted_pos[-n_tercile:]]
    bottom_idx = valid_idx[sorted_pos[:n_tercile]]

    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    subplot_titles = []
    for p in free_params:
        sep = tercile_comp.get(p, {}).get('separation', float('nan'))
        sep_str = f'{sep:.2f}' if not np.isnan(sep) else 'N/A'
        subplot_titles.append(f'<b>{p}</b><br><i>Separation: {sep_str}</i>')

    v_spacing = min(0.15, round(1.0 / max(rows, 1), 3))
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=v_spacing,
        horizontal_spacing=0.10,
    )

    for idx, p in enumerate(free_params):
        r = idx // cols + 1
        c = idx %  cols + 1

        vals     = param_dist[p].values.astype(float)
        top_vals = vals[top_idx]
        bot_vals = vals[bottom_idx]

        fig.add_trace(go.Box(
            y=top_vals,
            name='Top tercile',
            marker_color=_GREEN,
            line_color=_GREEN,
            boxmean=True,
            showlegend=(idx == 0),
            hovertemplate=f'<b>{p} — Top tercile</b><br>%{{y:.4g}}<extra></extra>',
        ), row=r, col=c)

        fig.add_trace(go.Box(
            y=bot_vals,
            name='Bottom tercile',
            marker_color=_RED,
            line_color=_RED,
            boxmean=True,
            showlegend=(idx == 0),
            hovertemplate=f'<b>{p} — Bottom tercile</b><br>%{{y:.4g}}<extra></extra>',
        ), row=r, col=c)

        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=_GRID, row=r, col=c)
        fig.update_xaxes(showgrid=False, row=r, col=c)

    fig.update_layout(
        height=max(420, rows * 320),
        template=_TEMPLATE,
        title=dict(
            text='<b>CPCV Tercile Comparison</b>  —  Top vs Bottom Splits by OOS Sharpe',
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
        legend=dict(
            yanchor='top', y=0.99, xanchor='right', x=0.99,
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1,
        ),
        boxmode='group',
    )

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved tercile comparison → {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Function 7: plot_cpcv_results  (convenience wrapper)
# ──────────────────────────────────────────────────────────────────────────────

def plot_cpcv_results(cpcv_results, wf_sharpe=None, ci_results=None,
                      analysis=None, show=True, save_html_dir=None):
    """
    Convenience wrapper for path-level charts only.

    Calls in sequence:
      1. plot_path_equity_curves  — all paths + mean + envelope + group lines
      2. plot_path_distribution   — histogram of path Sharpes (with CI bands if
                                    ci_results is provided)

    Parameter plots (plot_parameter_distributions, plot_parameter_correlation_matrix,
    plot_split_performance_heatmap, plot_tercile_comparison) are intentionally
    excluded so they can be called once with the analysis object in the
    Parameter Analysis section rather than rendering twice.

    Parameters
    ----------
    cpcv_results  : dict returned by run_cpcv()
    wf_sharpe     : float walk-forward Sharpe for comparison annotation (optional)
    ci_results    : dict returned by cpcv_confidence_intervals() (optional).
                    Passed through to plot_path_distribution to render CI bands.
    analysis      : accepted for API compatibility; not used by this wrapper.
    show          : render each figure inline
    save_html_dir : if provided, save each chart as an HTML file in this dir
    """
    import os

    def _html(name):
        if save_html_dir is None:
            return None
        os.makedirs(save_html_dir, exist_ok=True)
        return os.path.join(save_html_dir, name)

    plot_path_equity_curves(
        cpcv_results, wf_sharpe=wf_sharpe,
        show=show,
        save_html=_html('cpcv_equity_curves.html'),
    )

    plot_path_distribution(
        cpcv_results, wf_sharpe=wf_sharpe, ci_results=ci_results,
        show=show,
        save_html=_html('cpcv_path_distribution.html'),
    )
