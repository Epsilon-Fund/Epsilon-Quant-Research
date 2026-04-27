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


# ── internal helper ────────────────────────────────────────────────────────────

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

def plot_path_equity_curves(cpcv_results, wf_sharpe=None, title=None,
                            show=True, save_html=None):
    """
    All CPCV path equity curves on one chart.

    - Each path: semi-transparent blue line.
    - Mean path: bold amber line.
    - Envelope: shaded band between min and max path equity at every bar.
    - Group boundaries: vertical dashed lines with group labels.
    - If wf_sharpe is provided: annotation box comparing WF vs CPCV mean ± std.
    """
    paths   = cpcv_results['paths']
    bounds  = cpcv_results['group_boundaries']
    config  = cpcv_results['config']
    N, k    = config['N'], config['k']

    valid = [p for p in paths if p['equity_curve'] is not None]
    if not valid:
        print('[plot_path_equity_curves] No valid equity curves — nothing to plot.')
        return None

    # align all curves to a common sorted index
    common_idx = valid[0]['equity_curve'].index
    for p in valid[1:]:
        common_idx = common_idx.union(p['equity_curve'].index)
    common_idx = common_idx.sort_values()

    aligned = pd.DataFrame(index=common_idx)
    for p in valid:
        aligned[p['path_id']] = p['equity_curve'].reindex(common_idx).ffill()

    mean_curve = aligned.mean(axis=1)
    min_curve  = aligned.min(axis=1)
    max_curve  = aligned.max(axis=1)

    x_fwd = list(common_idx)
    x_rev = list(common_idx[::-1])

    fig = go.Figure()

    # ── envelope band ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fwd + x_rev,
        y=list(max_curve.values) + list(min_curve.values[::-1]),
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Min–Max Envelope',
        hoverinfo='skip',
    ))

    # ── individual paths ──────────────────────────────────────────────────────
    for i, p in enumerate(valid):
        curve = aligned[p['path_id']]
        fig.add_trace(go.Scatter(
            x=x_fwd, y=curve.values,
            mode='lines',
            line=dict(color=_BLUE, width=0.7),
            opacity=0.20,
            name='Paths',
            showlegend=(i == 0),
            hoverinfo='skip',
        ))

    # ── mean path ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fwd, y=mean_curve.values,
        mode='lines',
        line=dict(color=_AMBER, width=2.5),
        name='Mean Path',
        hovertemplate='<b>Mean Path</b><br>%{x}<br>Equity: %{y:.4f}<extra></extra>',
    ))

    # ── group boundary lines + labels ─────────────────────────────────────────
    for i, (start, _) in enumerate(bounds):
        fig.add_vline(
            x=str(start),
            line_dash='dot',
            line_color='rgba(100,116,139,0.55)',
            line_width=1.5,
        )
        fig.add_annotation(
            x=str(start), y=1.0, yref='paper',
            text=f'G{i}',
            showarrow=False,
            font=dict(size=9, color=_GREY, family=_FONT_MONO),
            xanchor='left', yanchor='bottom',
        )

    # ── WF Sharpe comparison annotation ──────────────────────────────────────
    if wf_sharpe is not None:
        valid_sh = [p['sharpe'] for p in valid if p['sharpe'] is not None]
        mean_sh  = float(np.mean(valid_sh)) if valid_sh else float('nan')
        std_sh   = float(np.std(valid_sh))  if valid_sh else float('nan')
        ann = (
            f'<b>Sharpe Comparison</b><br>'
            f'WF Sharpe:   <b>{wf_sharpe:.2f}</b><br>'
            f'CPCV Mean:   <b>{mean_sh:.2f}</b><br>'
            f'CPCV ±Std:   <b>{std_sh:.2f}</b>'
        )
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.01, y=0.99, xanchor='left', yanchor='top',
            text=ann, showarrow=False,
            font=dict(size=10, family=_FONT_MONO),
            align='left',
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1, borderpad=8,
        )

    plot_title = title or (
        f'<b>CPCV Path Equity Curves</b>'
        f'  —  {len(valid)} paths  (N={N}, k={k})'
    )
    fig.update_layout(
        height=600,
        template=_TEMPLATE,
        title=dict(text=plot_title, font=dict(size=22, color='#1E293B'),
                   x=0.5, xanchor='center'),
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99,
                    bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1),
        xaxis=dict(title='Date', showgrid=True, gridwidth=1, gridcolor=_GRID),
        yaxis=dict(title='Equity (1.0 = start)', showgrid=True,
                   gridwidth=1, gridcolor=_GRID),
        hovermode='x unified',
    )

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

    labels = [f'G{i}' for i in range(N)]

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
    n_splits    = len(cpcv_results['split_results'])
    n_tercile   = max(1, n_splits // 3)
    valid_mask  = ~np.isnan(oos_sharpes)

    if valid_mask.sum() < 3:
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
