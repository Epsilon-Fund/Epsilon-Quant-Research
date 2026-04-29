"""
CPCV Portfolio Visualizer
──────────────────────────────────────────────────────────────────────────────
Plotly visualisation suite for portfolio-level CPCV results.
Companion to cpcv_portfolio.py and cpcv_visualizer.py.

Matches the visual style of cpcv_visualizer.py exactly: same template, colour
palette, annotation boxes, and figure layout.

Typical usage
─────────────
    from cpcv_portfolio import (
        load_asset_cpcv, sample_portfolio_paths,
        portfolio_confidence_intervals, per_asset_split_heatmaps,
    )
    from cpcv_portfolio_visualizer import plot_portfolio_results

    assets   = load_asset_cpcv({'BTC': 'btcusdt_cpcv.pkl', ...})
    weights  = {'BTC': 0.40, 'ETH': 0.35, 'SOL': 0.25}
    paths    = sample_portfolio_paths(assets, weights, n_samples=2000)
    ci       = portfolio_confidence_intervals(paths, assets)
    heatmaps = per_asset_split_heatmaps(paths, assets)
    plot_portfolio_results(paths, assets, ci_results=ci, weights=weights)
"""

import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── shared style constants (mirrors cpcv_visualizer.py exactly) ───────────────
_BLUE         = '#3b82f6'
_AMBER        = '#f59e0b'
_GREEN        = '#22c55e'
_RED          = '#ef4444'
_RED_FILL     = 'rgba(239, 68, 68, 0.3)'
_GREY         = '#64748b'
_GRID         = '#f1f5f9'
_BORDER       = '#cbd5e1'
_BG_BOX       = 'rgba(255, 255, 255, 0.9)'
_FONT_MONO    = 'monospace'
_TEMPLATE     = 'plotly_white'

# per-asset colour palette (distinct from blue/amber reserved for portfolio)
_ASSET_PALETTE = [
    '#8b5cf6', '#06b6d4', '#84cc16', '#f97316',
    '#ec4899', '#14b8a6', '#a16207', '#6366f1',
]


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _asset_medians(asset_results: dict) -> dict:
    """Return {asset: median_sharpe} from each asset's CPCV path Sharpes."""
    out = {}
    for asset, results in asset_results.items():
        vals = [p['sharpe'] for p in results['paths'] if p.get('sharpe') is not None]
        out[asset] = float(np.median(vals)) if vals else float('nan')
    return out


def _fmt(val, pct=False, dp=2):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 'N/A'
    if pct:
        return f'{val * 100:.{dp}f}%'
    return f'{val:.{dp}f}'


def _annotation_box(fig, lines, x=0.01, y=0.99, xanchor='left', yanchor='top'):
    """Add a semi-transparent annotation box (matches cpcv_visualizer.py style)."""
    fig.add_annotation(
        xref='paper', yref='paper',
        x=x, y=y,
        xanchor=xanchor, yanchor=yanchor,
        text='<br>'.join(lines),
        showarrow=False,
        font=dict(size=11, family=_FONT_MONO),
        align='left',
        bgcolor=_BG_BOX,
        bordercolor=_BORDER,
        borderwidth=1,
        borderpad=6,
    )


def _save_and_show(fig, show, save_html):
    if save_html:
        fig.write_html(save_html)
        print(f'  Saved → {save_html}')
    if show:
        fig.show()
    return fig


def _infer_ppy(index):
    """Estimate periods-per-year from a DatetimeIndex."""
    if len(index) < 2:
        return 365
    deltas = pd.Series(index.to_list()).diff().dropna()
    med = deltas.median()
    hours = med.total_seconds() / 3600 if hasattr(med, 'total_seconds') else float(med) / 3_600_000_000_000
    if hours <= 1:
        return 8760
    if hours <= 4:
        return 2190
    if hours <= 24:
        return 365
    if hours <= 168:
        return 52
    return 12


# ──────────────────────────────────────────────────────────────────────────────
#  Function 1: plot_portfolio_equity_curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_portfolio_equity_curves(portfolio_paths, ci_results=None, n_display=200,
                                 title=None, show=True, save_html=None):
    """
    Portfolio equity curves — 2-panel layout matching the backtester chart style.

    Row 1 (equity, 70%):
      - Min–max envelope band, individual paths (semi-transparent), mean path.
      - Annotation boxes: Portfolio Performance (top-left), Yearly Returns
        (mid-left), Yearly Sharpe (mid), CI stats folded into Performance box.

    Row 2 (drawdown of mean path, 30%):
      - Red filled drawdown of the mean equity curve.
      - Yearly Max DD annotation (top-right).

    Parameters
    ----------
    portfolio_paths : list of dicts — output of sample_portfolio_paths()
    ci_results      : dict, optional — output of portfolio_confidence_intervals()
    n_display       : int — max individual paths to draw (random subsample)
    title           : str, optional
    show            : bool
    save_html       : str, optional
    """
    valid = [p for p in portfolio_paths if p.get('equity_curve') is not None]
    if not valid:
        print('[plot_portfolio_equity_curves] No valid equity curves.')
        return None

    # subsample for display
    rng = np.random.default_rng(0)
    if len(valid) > n_display:
        display_paths = [valid[i] for i in rng.choice(len(valid), n_display, replace=False)]
    else:
        display_paths = valid

    # build common index (union of display paths)
    common_idx = display_paths[0]['equity_curve'].index
    for p in display_paths[1:]:
        common_idx = common_idx.union(p['equity_curve'].index)
    common_idx = common_idx.sort_values()

    mat = np.array([
        p['equity_curve'].reindex(common_idx).ffill().values
        for p in display_paths
    ], dtype=float) / 100.0   # equity_curve is 100-based; display as 1-based

    env_min     = np.nanmin(mat, axis=0)
    env_max     = np.nanmax(mat, axis=0)
    mean_ec     = np.nanmean(mat, axis=0)
    mean_series = pd.Series(mean_ec, index=common_idx)

    dates = common_idx.tolist()

    # ── annotation stats: sourced from path distribution (matches histogram) ────
    path_sharpes = [p['sharpe'] for p in valid if p.get('sharpe') is not None]
    path_calmars = [p['calmar'] for p in valid if p.get('calmar') is not None]
    path_maxdds  = [p['max_dd'] for p in valid if p.get('max_dd')  is not None]
    path_rets    = [p['total_return'] for p in valid if p.get('total_return') is not None]
    sharpe  = float(np.mean(path_sharpes)) if path_sharpes else 0.0
    calmar  = float(np.mean(path_calmars)) if path_calmars else 0.0
    max_dd  = float(np.mean(path_maxdds))  if path_maxdds  else 0.0
    tot_ret = float(np.mean(path_rets))    if path_rets    else 0.0

    # ── mean-path curve stats (for drawdown panel and yearly breakdowns only) ──
    ppy       = _infer_ppy(common_idx)
    mean_rets = mean_series.pct_change().dropna()
    run_max   = mean_series.cummax()
    dd_pct    = (mean_series - run_max) / run_max   # negative fractions

    # ── yearly metrics from mean equity curve ─────────────────────────────────
    yr_ret, yr_sh, yr_dd = {}, {}, {}
    for y in sorted(mean_series.index.year.unique()):
        yr_r = mean_rets[mean_rets.index.year == y]
        c_yr = mean_series[mean_series.index.year == y]
        if len(yr_r) < 2 or len(c_yr) < 2:
            continue
        yr_ret[y] = float((c_yr.iloc[-1] - c_yr.iloc[0]) / c_yr.iloc[0])
        yr_sh[y]  = (float(yr_r.mean() / yr_r.std() * np.sqrt(ppy))
                     if yr_r.std() > 0 else 0.0)
        rm_yr     = c_yr.cummax()
        yr_dd[y]  = float(((c_yr - rm_yr) / rm_yr).min())

    # ── CI values for annotation ──────────────────────────────────────────────
    n_eff  = float('nan')
    adj_lo = float('nan')
    adj_hi = float('nan')
    if ci_results and 'sharpe' in ci_results:
        sh_ci  = ci_results['sharpe']
        n_eff  = sh_ci.get('n_effective', float('nan'))
        adj    = sh_ci.get('adjusted_ci', (float('nan'), float('nan')))
        adj_lo, adj_hi = adj

    # ── 2-row subplot ─────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.06,
        shared_xaxes=True,
    )

    # envelope band
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=list(env_max) + list(env_min[::-1]),
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.10)',
        line=dict(width=0),
        hoverinfo='skip',
        showlegend=True,
        name='Envelope',
    ), row=1, col=1)

    # individual paths
    for ec_vals in mat:
        fig.add_trace(go.Scatter(
            x=dates, y=ec_vals.tolist(),
            mode='lines',
            line=dict(color=_BLUE, width=0.7),
            opacity=0.20,
            showlegend=False,
            hoverinfo='skip',
        ), row=1, col=1)

    # mean path
    fig.add_trace(go.Scatter(
        x=dates, y=mean_ec.tolist(),
        mode='lines',
        line=dict(color=_AMBER, width=2.5),
        name='Mean path',
        hovertemplate='<b>Mean Path</b><br>%{x}<br>Equity: %{y:.4f}<extra></extra>',
    ), row=1, col=1)

    # drawdown of mean path
    fig.add_trace(go.Scatter(
        x=dates, y=(dd_pct.values * 100).tolist(),
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

    # ── annotation box 1: Portfolio Performance (top-left, row 1) ────────────
    perf_lines = [
        '<b>Portfolio Performance</b>',
        f'Total Return:  <b>{tot_ret * 100:.2f}%</b>',
        f'Sharpe Ratio:  <b>{sharpe:.2f}</b>',
        f'Max Drawdown:  <b>{max_dd * 100:.2f}%</b>',
        f'Calmar Ratio:  <b>{calmar:.2f}</b>',
        f'N Paths:       <b>{len(valid)}</b>',
    ]
    if not math.isnan(n_eff):
        perf_lines.append(f'N effective:   <b>{n_eff:.1f}</b>')
    if not math.isnan(adj_lo):
        perf_lines.append(
            f'Adj 95% CI:    <b>[{_fmt(adj_lo)}, {_fmt(adj_hi)}]</b>'
        )
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
            f'{y}:  <b>{r * 100:.2f}%</b>' for y, r in sorted(yr_ret.items())
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

    # ── annotation box 4: Yearly Max DD (top-right, row 2) ───────────────────
    if yr_dd:
        yr_dd_txt = '<b>Yearly Max DD</b><br>' + '<br>'.join(
            f'{y}:  <b>{d * 100:.2f}%</b>' for y, d in sorted(yr_dd.items())
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

    fig.update_layout(
        height=900,
        template=_TEMPLATE,
        title=dict(
            text=title or 'Portfolio CPCV — Sampled Equity Curves',
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99,
                    bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1),
        hovermode='x unified',
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=_GRID,
                     dtick='M12', tickformat='%Y', row=1, col=1)
    fig.update_xaxes(title_text='Date', showgrid=True, gridwidth=1,
                     gridcolor=_GRID, dtick='M12', tickformat='%Y', row=2, col=1)
    fig.update_yaxes(title_text='Portfolio Equity (base = 1)', showgrid=True,
                     gridwidth=1, gridcolor=_GRID, row=1, col=1)
    fig.update_yaxes(title_text='Drawdown (%)', showgrid=True,
                     gridwidth=1, gridcolor=_GRID, row=2, col=1)

    return _save_and_show(fig, show, save_html)


# ──────────────────────────────────────────────────────────────────────────────
#  Function 2: plot_portfolio_distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_portfolio_distribution(portfolio_paths, ci_results=None, asset_results=None,
                                wf_sharpes=None, show=True, save_html=None):
    """
    Histogram of sampled portfolio Sharpe ratios with CI band overlay.

    - CI band : blue shaded rect between adjusted CI bounds (via add_shape).
    - Histogram: blue, nbinsx=15, opacity=0.75.
    - Mean vline: solid amber.
    - Conservative floor vline: dashed red.
    - Per-asset CPCV median Sharpes: dotted coloured vlines (from _ASSET_PALETTE).
    - WF Sharpes (optional): dash-dot grey vlines.
    - Annotation box (top-right): N paths, N effective, Mean Sharpe, Adj CI.

    Parameters
    ----------
    portfolio_paths : list of dicts
    ci_results : dict, optional
    asset_results : dict, optional
        If provided, per-asset CPCV median Sharpe vlines are drawn.
    wf_sharpes : dict, optional
        {asset: float} or {label: float} — walk-forward Sharpes as reference lines.
    show : bool
    save_html : str, optional
    """
    sharpes = [p['sharpe'] for p in portfolio_paths if p.get('sharpe') is not None]
    if not sharpes:
        print('[plot_portfolio_distribution] No valid Sharpe values.')
        return None

    arr      = np.array(sharpes, dtype=float)
    mean_sh  = float(np.mean(arr))

    fig = go.Figure()

    # CI band (via add_shape — matches cpcv_visualizer.py pattern)
    adj_lo, adj_hi = float('nan'), float('nan')
    floor_val      = float('nan')
    n_eff          = float('nan')
    if ci_results and 'sharpe' in ci_results:
        sh_ci    = ci_results['sharpe']
        n_eff    = sh_ci.get('n_effective', float('nan'))
        adj      = sh_ci.get('adjusted_ci', (float('nan'), float('nan')))
        adj_lo, adj_hi = adj
        floor_val = sh_ci.get('conservative_lower_bound', float('nan'))

        if not (math.isnan(adj_lo) or math.isnan(adj_hi)):
            fig.add_shape(
                type='rect',
                xref='x', yref='paper',
                x0=adj_lo, x1=adj_hi,
                y0=0, y1=1,
                fillcolor='rgba(59, 130, 246, 0.18)',
                line=dict(width=0),
                layer='below',
            )
            # dummy scatter so CI band appears in legend
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color=_BLUE, width=8),
                opacity=0.4,
                name='Adj 95% CI',
            ))

    # histogram
    fig.add_trace(go.Histogram(
        x=arr.tolist(),
        nbinsx=15,
        marker_color=_BLUE,
        opacity=0.75,
        name='Portfolio Sharpe',
    ))

    # mean vline
    fig.add_vline(
        x=mean_sh,
        line=dict(color=_AMBER, width=2, dash='solid'),
        annotation_text=f'Mean {_fmt(mean_sh)}',
        annotation_position='top right',
        annotation_font=dict(color=_AMBER, size=11),
    )

    # conservative floor vline
    if not math.isnan(floor_val):
        fig.add_vline(
            x=floor_val,
            line=dict(color=_RED, width=1.5, dash='dash'),
            annotation_text=f'Floor {_fmt(floor_val)}',
            annotation_position='bottom left',
            annotation_font=dict(color=_RED, size=11),
        )

    # per-asset CPCV median vlines
    if asset_results:
        medians = _asset_medians(asset_results)
        alt_pos = ['top left', 'bottom left', 'top right', 'bottom right']
        for i, (asset, med_sh) in enumerate(medians.items()):
            if math.isnan(med_sh):
                continue
            col = _ASSET_PALETTE[i % len(_ASSET_PALETTE)]
            fig.add_vline(
                x=med_sh,
                line=dict(color=col, width=1.2, dash='dot'),
                annotation_text=asset,
                annotation_position=alt_pos[i % len(alt_pos)],
                annotation_font=dict(color=col, size=10),
            )

    # WF Sharpes vlines
    if wf_sharpes:
        for label, val in wf_sharpes.items():
            fig.add_vline(
                x=float(val),
                line=dict(color=_GREY, width=1.2, dash='dashdot'),
                annotation_text=f'WF {label}',
                annotation_position='top left',
                annotation_font=dict(color=_GREY, size=10),
            )

    # annotation box (top-right)
    ann_lines = [
        f'N paths    : {len(arr)}',
        f'N effective: {_fmt(n_eff, dp=1)}',
        f'Mean Sharpe: {_fmt(mean_sh)}',
    ]
    if not math.isnan(adj_lo):
        ann_lines.append(f'Adj 95% CI : [{_fmt(adj_lo)}, {_fmt(adj_hi)}]')

    _annotation_box(fig, ann_lines, x=0.99, y=0.99, xanchor='right')

    fig.update_layout(
        template=_TEMPLATE,
        title='Portfolio CPCV — Sharpe Distribution',
        xaxis_title='Sharpe Ratio',
        yaxis_title='Count',
        height=480,
        bargap=0.05,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    return _save_and_show(fig, show, save_html)


# ──────────────────────────────────────────────────────────────────────────────
#  Function 3: plot_per_asset_heatmaps
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_asset_heatmaps(heatmap_data, asset_results, show=True, save_html=None):
    """
    Grid of heatmaps — one per asset — showing mean portfolio Sharpe for each
    (group_i, group_j) split pair used by that asset.

    Colorscale is centred at the global mean of all mean_portfolio_sharpe values
    (centering at 0 is not useful since all portfolio Sharpes are positive).

    Parameters
    ----------
    heatmap_data : dict
        Output of per_asset_split_heatmaps() from cpcv_portfolio.py.
        {asset: DataFrame with columns [group_i, group_j, mean_portfolio_sharpe, count]}
    asset_results : dict
        Output of load_asset_cpcv() — used to determine N (number of groups).
    show : bool
    save_html : str, optional
    """
    assets = list(heatmap_data.keys())
    if not assets:
        print('[plot_per_asset_heatmaps] No heatmap data.')
        return None

    # determine global colorscale center
    all_vals = []
    for df in heatmap_data.values():
        if df is not None and len(df):
            all_vals.extend(df['mean_portfolio_sharpe'].dropna().tolist())
    global_center = float(np.mean(all_vals)) if all_vals else 0.0
    global_min    = float(np.min(all_vals))  if all_vals else -1.0
    global_max    = float(np.max(all_vals))  if all_vals else  1.0

    # build symmetric colorscale around center
    half_range = max(abs(global_max - global_center), abs(global_center - global_min), 1e-6)
    zmin = global_center - half_range
    zmax = global_center + half_range

    n_assets = len(assets)
    cols     = min(3, n_assets)
    rows     = math.ceil(n_assets / cols)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=assets,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, asset in enumerate(assets):
        row = idx // cols + 1
        col = idx %  cols + 1
        is_last = (idx == n_assets - 1)

        df = heatmap_data.get(asset)
        if df is None or len(df) == 0:
            continue

        # infer N from asset_results
        N = asset_results[asset]['config']['N']
        z_mat  = np.full((N, N), float('nan'))
        text_mat = [[''] * N for _ in range(N)]

        for _, row_data in df.iterrows():
            gi = int(row_data['group_i'])
            gj = int(row_data['group_j'])
            v  = row_data['mean_portfolio_sharpe']
            if 0 <= gi < N and 0 <= gj < N:
                z_mat[gi][gj] = v
                text_mat[gi][gj] = _fmt(v)

        group_labels = [str(i) for i in range(N)]

        fig.add_trace(
            go.Heatmap(
                z=z_mat,
                x=group_labels,
                y=group_labels,
                text=text_mat,
                texttemplate='%{text}',
                colorscale='RdYlGn',
                zmin=zmin,
                zmax=zmax,
                showscale=is_last,
                colorbar=dict(title='Mean Portfolio<br>Sharpe') if is_last else None,
                hovertemplate='Group i=%{y}<br>Group j=%{x}<br>Sharpe=%{z:.3f}<extra></extra>',
            ),
            row=row, col=col,
        )

        fig.update_yaxes(autorange='reversed', row=row, col=col)

    fig.update_layout(
        template=_TEMPLATE,
        title='Portfolio CPCV — Mean Portfolio Sharpe by Split Pair (per asset)',
        height=max(420, rows * 280),
    )

    return _save_and_show(fig, show, save_html)


# ──────────────────────────────────────────────────────────────────────────────
#  Function 4: plot_portfolio_vs_assets
# ──────────────────────────────────────────────────────────────────────────────

def plot_portfolio_vs_assets(portfolio_paths, asset_results, weights=None,
                             show=True, save_html=None):
    """
    Box plots: per-asset CPCV Sharpe distributions + portfolio Sharpe distribution.

    - Per-asset boxes: blue, sorted by median Sharpe descending.
    - Portfolio box: amber, shown last.
    - boxmean=True so the mean marker is visible.

    Parameters
    ----------
    portfolio_paths : list of dicts
        Output of sample_portfolio_paths().
    asset_results : dict
        Output of load_asset_cpcv().
    weights : dict, optional
        {asset: weight} — shown in x-axis labels if provided.
    show : bool
    save_html : str, optional
    """
    # collect per-asset individual path Sharpes
    asset_sharpes = {}
    for asset, results in asset_results.items():
        vals = [p['sharpe'] for p in results['paths'] if p.get('sharpe') is not None]
        if vals:
            asset_sharpes[asset] = np.array(vals, dtype=float)

    # sort assets by median descending
    sorted_assets = sorted(
        asset_sharpes.keys(),
        key=lambda a: float(np.median(asset_sharpes[a])),
        reverse=True,
    )

    portfolio_sharpes = np.array(
        [p['sharpe'] for p in portfolio_paths if p.get('sharpe') is not None],
        dtype=float,
    )

    fig = go.Figure()

    for asset in sorted_assets:
        label = asset
        if weights and asset in weights:
            label = f'{asset} ({weights[asset]*100:.0f}%)'
        fig.add_trace(go.Box(
            y=asset_sharpes[asset].tolist(),
            name=label,
            marker_color=_BLUE,
            fillcolor='rgba(59, 130, 246, 0.25)',
            line_color=_BLUE,
            boxmean=True,
            showlegend=False,
        ))

    # portfolio box (amber)
    fig.add_trace(go.Box(
        y=portfolio_sharpes.tolist(),
        name='Portfolio',
        marker_color=_AMBER,
        fillcolor='rgba(245, 158, 11, 0.25)',
        line_color=_AMBER,
        boxmean=True,
        showlegend=False,
    ))

    fig.update_layout(
        template=_TEMPLATE,
        title='Portfolio CPCV — Asset vs Portfolio Sharpe Distributions',
        xaxis_title='Asset',
        yaxis_title='Sharpe Ratio',
        height=500,
    )

    return _save_and_show(fig, show, save_html)


# ──────────────────────────────────────────────────────────────────────────────
#  Function 5: plot_portfolio_results  (master entry point)
# ──────────────────────────────────────────────────────────────────────────────

def plot_portfolio_results(portfolio_paths, asset_results, ci_results=None,
                           weights=None, wf_sharpes=None,
                           show=True, save_html_dir=None):
    """
    Convenience wrapper: calls plot_portfolio_equity_curves, plot_portfolio_distribution,
    and plot_portfolio_vs_assets in sequence.

    Note: plot_per_asset_heatmaps is NOT called here because it requires
    heatmap_data computed separately via per_asset_split_heatmaps().

    Parameters
    ----------
    portfolio_paths : list of dicts
    asset_results : dict
    ci_results : dict, optional
    weights : dict, optional
    wf_sharpes : dict, optional
        Passed to plot_portfolio_distribution.
    show : bool
    save_html_dir : str, optional
        Directory in which to write:
          - portfolio_equity_curves.html
          - portfolio_distribution.html
          - portfolio_vs_assets.html
    """
    def _html_path(name):
        if save_html_dir:
            os.makedirs(save_html_dir, exist_ok=True)
            return os.path.join(save_html_dir, name)
        return None

    figs = {}

    figs['equity_curves'] = plot_portfolio_equity_curves(
        portfolio_paths,
        ci_results=ci_results,
        show=show,
        save_html=_html_path('portfolio_equity_curves.html'),
    )

    figs['distribution'] = plot_portfolio_distribution(
        portfolio_paths,
        ci_results=ci_results,
        asset_results=asset_results,
        wf_sharpes=wf_sharpes,
        show=show,
        save_html=_html_path('portfolio_distribution.html'),
    )

    figs['vs_assets'] = plot_portfolio_vs_assets(
        portfolio_paths,
        asset_results,
        weights=weights,
        show=show,
        save_html=_html_path('portfolio_vs_assets.html'),
    )

    return figs


# ──────────────────────────────────────────────────────────────────────────────
#  Function 6: plot_diversification_benefit
# ──────────────────────────────────────────────────────────────────────────────

def plot_diversification_benefit(div_benefit_df, weights,
                                  show=True, save_html=None):
    """
    Two subplots:

    Left  — histogram of diversification_benefit across all paths.
            Bars below 0 in red, bars above 0 in green.  Vertical
            lines at 0 (break-even) and at the mean.  Annotation box
            with mean benefit and % paths positive.

    Right — scatter of weighted_avg_component_sharpe (x) vs
            portfolio_sharpe (y), one point per path, coloured by
            diversification_benefit (RdYlGn scale).  Diagonal y = x
            line marks where portfolio == weighted-average component.
            Points above the line = true diversification benefit.

    Parameters
    ----------
    div_benefit_df : pd.DataFrame  returned by diversification_benefit()
    weights        : dict  {"ASSET": weight, ...}
    show / save_html : display / export controls
    """
    benefits  = div_benefit_df['diversification_benefit'].dropna()
    wa_col    = div_benefit_df['weighted_avg_component_sharpe']
    ps_col    = div_benefit_df['portfolio_sharpe']
    div_col   = div_benefit_df['diversification_benefit']
    mean_b    = float(benefits.mean())
    pct_pos   = float((benefits > 0).mean())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Diversification Benefit Distribution',
            'Portfolio vs Weighted-Avg Component Sharpe',
        ],
        horizontal_spacing=0.14,
    )

    # ── left: histogram split by sign ────────────────────────────────────────
    neg_vals = benefits[benefits < 0]
    pos_vals = benefits[benefits >= 0]

    if len(neg_vals):
        fig.add_trace(go.Histogram(
            x=neg_vals.tolist(), nbinsx=15,
            marker_color='rgba(239, 68, 68, 0.65)',
            name='Below 0',
        ), row=1, col=1)

    if len(pos_vals):
        fig.add_trace(go.Histogram(
            x=pos_vals.tolist(), nbinsx=15,
            marker_color='rgba(34, 197, 94, 0.65)',
            name='Above 0',
        ), row=1, col=1)

    # break-even and mean vlines (via shapes for subplot precision)
    for x_val, color, dash, label in [
        (0,      _GREY,  'dash',  None),
        (mean_b, _AMBER, 'solid', f'Mean {_fmt(mean_b)}'),
    ]:
        fig.add_shape(
            type='line', xref='x', yref='y domain',
            x0=x_val, x1=x_val, y0=0, y1=1,
            line=dict(color=color, width=1.8, dash=dash),
            row=1, col=1,
        )
        if label:
            fig.add_annotation(
                x=x_val, y=1, xref='x', yref='y domain',
                text=label, showarrow=False,
                font=dict(color=color, size=10, family=_FONT_MONO),
                xanchor='left', yanchor='top',
                row=1, col=1,
            )

    # annotation box — left panel (paper coords)
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.01, y=0.97, xanchor='left', yanchor='top',
        text=(f'Mean benefit : {_fmt(mean_b)}<br>'
              f'% paths > 0  : {pct_pos * 100:.1f}%'),
        showarrow=False,
        font=dict(size=11, family=_FONT_MONO),
        bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1, borderpad=6,
    )

    # ── right: scatter coloured by benefit ───────────────────────────────────
    cmin = float(div_col.min())
    cmax = float(div_col.max())
    # centre colorscale at 0 when there are negative values
    if cmin < 0:
        half = max(abs(cmin), abs(cmax))
        cmin, cmax = -half, half

    fig.add_trace(go.Scatter(
        x=wa_col.tolist(), y=ps_col.tolist(),
        mode='markers',
        marker=dict(
            color=div_col.tolist(),
            colorscale='RdYlGn',
            cmin=cmin, cmax=cmax,
            size=6, opacity=0.75,
            colorbar=dict(title='Div benefit', len=0.55, x=1.02),
            showscale=True,
        ),
        name='Paths', showlegend=False,
    ), row=1, col=2)

    # y = x diagonal reference line
    lo = min(float(wa_col.min()), float(ps_col.min()))
    hi = max(float(wa_col.max()), float(ps_col.max()))
    pad = (hi - lo) * 0.06
    fig.add_trace(go.Scatter(
        x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
        mode='lines',
        line=dict(color=_GREY, dash='dash', width=1),
        name='y = x',
    ), row=1, col=2)

    fig.update_layout(
        template=_TEMPLATE,
        title='Portfolio CPCV — Diversification Benefit Analysis',
        height=500,
        barmode='overlay',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
    )
    fig.update_xaxes(title_text='Diversification Benefit', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_xaxes(title_text='Weighted-Avg Component Sharpe', row=1, col=2)
    fig.update_yaxes(title_text='Portfolio Sharpe', row=1, col=2)

    return _save_and_show(fig, show, save_html)


# ──────────────────────────────────────────────────────────────────────────────
#  Function 7: plot_correlation_structure
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation_structure(corr_results, show=True, save_html=None):
    """
    Two subplots:

    Left  — violin per asset pair showing the distribution of pairwise
            Pearson correlations across all sampled paths.  Violins are
            coloured by median correlation (green < 0.3, amber 0.3–0.5,
            red > 0.5).  Horizontal line at 0.7 marks the high-correlation
            threshold used by asset_correlation_structure().

    Right — symmetric N×N heatmap of mean pairwise correlations.
            Each cell is annotated with mean ± std.  Diverging RdYlGn
            colorscale centred at 0.

    Parameters
    ----------
    corr_results : dict  returned by asset_correlation_structure()
    show / save_html : display / export controls
    """
    pair_df    = corr_results['pair_correlations']
    summary_df = corr_results['summary']

    if len(pair_df) == 0:
        print('[plot_correlation_structure] No correlation data.')
        return None

    # derive ordered asset list from pairs
    all_assets = sorted(set(summary_df['asset_a'].tolist()
                            + summary_df['asset_b'].tolist()))

    pairs = [
        (r['asset_a'], r['asset_b'])
        for _, r in summary_df.iterrows()
    ]
    n_pairs = len(pairs)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Correlation Distribution by Asset Pair',
            'Mean Pairwise Correlation Heatmap',
        ],
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.12,
    )

    # ── left: violin per pair ─────────────────────────────────────────────────
    def _pair_color(median_corr):
        if median_corr < 0.3:
            return _GREEN
        if median_corr < 0.5:
            return _AMBER
        return _RED

    for _, row_data in summary_df.iterrows():
        pair_label = f"{row_data['asset_a']}-{row_data['asset_b']}"
        mask       = (
            (pair_df['asset_a'] == row_data['asset_a']) &
            (pair_df['asset_b'] == row_data['asset_b'])
        )
        vals  = pair_df.loc[mask, 'correlation'].dropna().tolist()
        color = _pair_color(row_data['median_corr'])

        fig.add_trace(go.Violin(
            x=[pair_label] * len(vals),
            y=vals,
            name=pair_label,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color.replace('#', 'rgba(').rstrip(')') if False else color,
            line_color=color,
            opacity=0.65,
            showlegend=False,
        ), row=1, col=1)

    # high-correlation threshold line
    fig.add_shape(
        type='line', xref='x', yref='y',
        x0=-0.5, x1=n_pairs - 0.5,
        y0=0.7, y1=0.7,
        line=dict(color=_RED, width=1.5, dash='dash'),
        row=1, col=1,
    )
    fig.add_annotation(
        x=n_pairs - 1, y=0.7, xref='x', yref='y',
        text='0.7 threshold', showarrow=False,
        font=dict(color=_RED, size=10, family=_FONT_MONO),
        xanchor='right', yanchor='bottom',
        row=1, col=1,
    )

    # ── right: mean correlation heatmap ──────────────────────────────────────
    n_assets = len(all_assets)
    z_mat    = np.ones((n_assets, n_assets))   # diagonal = 1
    text_mat = [[''] * n_assets for _ in range(n_assets)]

    for _, row_data in summary_df.iterrows():
        i = all_assets.index(row_data['asset_a'])
        j = all_assets.index(row_data['asset_b'])
        v    = row_data['mean_corr']
        s    = row_data['std_corr']
        z_mat[i][j] = v
        z_mat[j][i] = v
        cell_text      = f'{v:.2f}<br><sub>±{s:.2f}</sub>'
        text_mat[i][j] = cell_text
        text_mat[j][i] = cell_text

    for k in range(n_assets):
        text_mat[k][k] = '1.00'

    fig.add_trace(go.Heatmap(
        z=z_mat,
        x=all_assets, y=all_assets,
        text=text_mat,
        texttemplate='%{text}',
        colorscale='RdYlGn',
        zmin=-1, zmax=1,
        colorbar=dict(title='Mean corr', len=0.55, x=1.02),
        hovertemplate='%{y} vs %{x}: %{z:.3f}<extra></extra>',
    ), row=1, col=2)

    fig.update_yaxes(autorange='reversed', row=1, col=2)

    fig.update_layout(
        template=_TEMPLATE,
        title='Portfolio CPCV — Asset Correlation Structure',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
    )
    fig.update_xaxes(title_text='Asset Pair', row=1, col=1)
    fig.update_yaxes(title_text='Pearson Correlation', row=1, col=1)

    return _save_and_show(fig, show, save_html)


# ──────────────────────────────────────────────────────────────────────────────
#  Function 8: plot_drawdown_decomposition
# ──────────────────────────────────────────────────────────────────────────────

def plot_drawdown_decomposition(dd_results, weights, max_display=20,
                                show=True, save_html=None):
    """
    Two subplots:

    Left  — horizontal stacked bar chart, up to max_display worst paths sorted
            by portfolio max_dd severity (worst at top).  Each segment =
            one asset's absolute % contribution to that path's drawdown.
            Segments are coloured from _ASSET_PALETTE.  The primary driver
            asset is labelled.

    Right — box plot per asset showing the distribution of absolute
            pct_of_dd values across all worst paths.  A horizontal line
            at each asset's weight marks the "proportional share" level —
            boxes above the line indicate disproportionate tail risk.

    Parameters
    ----------
    dd_results  : dict  returned by worst_drawdown_decomposition()
    weights     : dict  {"ASSET": weight, ...}
    max_display : int   max bars to show in the stacked bar chart (default 20)
    show / save_html : display / export controls
    """
    worst_df       = dd_results['worst_paths']
    summary_df     = dd_results['summary']
    primary_driver = dd_results.get('primary_driver')

    if len(worst_df) == 0:
        print('[plot_drawdown_decomposition] No drawdown data.')
        return None

    # infer asset list from summary
    assets = summary_df['asset'].tolist() if len(summary_df) else list(weights.keys())
    colors = {a: _ASSET_PALETTE[i % len(_ASSET_PALETTE)]
              for i, a in enumerate(assets)}

    # sort by severity (most negative max_dd first = worst on top), cap display
    worst_sorted = worst_df.sort_values('portfolio_max_dd').reset_index(drop=True)
    n_total  = len(worst_sorted)
    worst_sorted = worst_sorted.head(max_display)
    n_paths  = len(worst_sorted)
    path_labels  = [
        f"P{int(r['path_id'])} ({r['portfolio_max_dd']*100:.1f}%)"
        for _, r in worst_sorted.iterrows()
    ]

    bar_title = (
        f'Asset Contribution to Drawdown (worst {n_paths} of {n_total} paths)'
        if n_total > n_paths else
        f'Asset Contribution to Drawdown (worst {n_paths} paths)'
    )
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[bar_title, 'Drawdown Contribution Distribution by Asset'],
        horizontal_spacing=0.12,
        column_widths=[0.60, 0.40],
    )

    # ── left: horizontal stacked bar (abs pct_of_dd) ─────────────────────────
    for asset in assets:
        col  = f'{asset}_pct_of_dd'
        vals = worst_sorted[col].abs().tolist() if col in worst_sorted.columns \
               else [0.0] * n_paths
        label = f'{asset} ★' if asset == primary_driver else asset

        fig.add_trace(go.Bar(
            x=vals,
            y=path_labels,
            name=label,
            orientation='h',
            marker_color=colors[asset],
            opacity=0.85,
        ), row=1, col=1)

    # ── right: box plot per asset ─────────────────────────────────────────────
    for asset in assets:
        col  = f'{asset}_pct_of_dd'
        vals = worst_df[col].abs().dropna().tolist() \
               if col in worst_df.columns else []

        fig.add_trace(go.Box(
            y=vals,
            name=asset,
            marker_color=colors[asset],
            fillcolor=colors[asset].replace(
                '#', 'rgba(').rstrip(')') if False else colors[asset],
            line_color=colors[asset],
            opacity=0.7,
            boxmean=True,
            showlegend=False,
        ), row=1, col=2)

    # horizontal line at each asset's weight (proportional contribution)
    # use a single mean weight line to avoid clutter; or per-asset annotations
    for i, asset in enumerate(assets):
        w = weights.get(asset, 1.0 / len(assets))
        fig.add_shape(
            type='line', xref='x2', yref='y2',
            x0=i - 0.45, x1=i + 0.45,
            y0=w, y1=w,
            line=dict(color=colors[asset], width=2, dash='dot'),
        )

    fig.update_layout(
        template=_TEMPLATE,
        title='Portfolio CPCV — Drawdown Decomposition',
        height=max(400, min(n_paths * 28 + 150, 800)),
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
    )
    fig.update_xaxes(title_text='|% of Drawdown|', row=1, col=1)
    fig.update_xaxes(tickformat='.0%', row=1, col=1)
    fig.update_yaxes(title_text='Path', row=1, col=1)
    fig.update_yaxes(title_text='|% of Drawdown|', row=1, col=2)
    fig.update_yaxes(tickformat='.0%', row=1, col=2)

    return _save_and_show(fig, show, save_html)


# ──────────────────────────────────────────────────────────────────────────────
#  Function 9: plot_weight_optimisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_weight_optimisation(optimal_results, asset_names,
                              show=True, save_html=None):
    """
    Three subplots:

    Top-left  — scatter of mean_sharpe (x) vs conservative_sharpe_floor (y)
                for all weight configs.  Points coloured by mean_max_dd
                (red = high DD, green = low DD).  Four optimal configs
                highlighted with large star markers; equal-weight config
                with a diamond.

    Top-right — scatter of mean_sharpe (x) vs mean_max_dd (y, axis inverted
                so lower DD appears higher).  Same colouring.  Traces the
                Sharpe vs drawdown efficiency frontier.

    Bottom    — grouped horizontal bar chart showing the per-asset weights
                for each of the four optimal configs and the equal-weight
                baseline.  Assets on y-axis; one bar group per objective.

    Parameters
    ----------
    optimal_results : dict  returned by optimal_weights()
    asset_names     : list  ordered list of asset names
    show / save_html : display / export controls
    """
    all_df      = optimal_results['all_configs']
    optimal     = optimal_results['optimal']
    equal_res   = optimal_results['equal_weight_result']

    if len(all_df) == 0:
        print('[plot_weight_optimisation] No weight config data.')
        return None

    # ── color scale: map mean_max_dd → [0,1] for RdYlGn ─────────────────────
    dd_vals    = all_df['mean_max_dd'].fillna(0)
    dd_min     = float(dd_vals.min())
    dd_max     = float(dd_vals.max())   # closest to 0 = best

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'bar', 'colspan': 2}, None]],
        subplot_titles=[
            'Mean Sharpe vs Conservative Floor',
            'Mean Sharpe vs Max Drawdown',
            'Optimal Portfolio Weights by Objective',
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    # ── top-left and top-right scatter (background all configs) ──────────────
    scatter_kwargs = dict(
        mode='markers',
        marker=dict(
            color=dd_vals.tolist(),
            colorscale='RdYlGn',
            cmin=dd_min, cmax=dd_max,
            size=5, opacity=0.55,
            colorbar=dict(title='Mean DD', len=0.35, x=1.02, y=0.78),
            showscale=True,
        ),
        showlegend=False,
        hovertemplate=(
            'Mean Sharpe: %{x:.2f}<br>'
            'Value: %{y:.3f}<br>'
            '<extra></extra>'
        ),
    )

    fig.add_trace(go.Scatter(
        x=all_df['mean_sharpe'].tolist(),
        y=all_df['conservative_sharpe_floor'].tolist(),
        name='All configs',
        **scatter_kwargs,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=all_df['mean_sharpe'].tolist(),
        y=all_df['mean_max_dd'].tolist(),
        name='All configs',
        **{**scatter_kwargs, 'marker': {**scatter_kwargs['marker'],
                                        'showscale': False, 'colorbar': None}},
    ), row=1, col=2)

    # ── highlight optimal configs ─────────────────────────────────────────────
    obj_labels = {
        'max_conservative_sharpe':    'Max cons. Sharpe',
        'max_sharpe_calmar':          'Max Sharpe×Calmar',
        'min_drawdown':               'Min drawdown',
        'max_equal_weight_comparison': 'Equal weight',
    }
    obj_colors  = [_AMBER, _BLUE, _GREEN, _GREY]
    obj_symbols = ['star', 'star', 'star', 'diamond']

    for idx_obj, (obj_name, res) in enumerate(optimal.items()):
        label  = obj_labels.get(obj_name, obj_name)
        color  = obj_colors[idx_obj % len(obj_colors)]
        symbol = obj_symbols[idx_obj % len(obj_symbols)]
        ms     = res.get('mean_sharpe', float('nan'))
        cs     = res.get('conservative_sharpe_floor', float('nan'))
        dd     = res.get('mean_max_dd', float('nan'))

        for r, c, y_val in [(1, 1, cs), (1, 2, dd)]:
            fig.add_trace(go.Scatter(
                x=[ms], y=[y_val],
                mode='markers+text',
                marker=dict(color=color, size=14, symbol=symbol,
                            line=dict(color='white', width=1.5)),
                text=[label],
                textposition='top right',
                textfont=dict(size=9, color=color),
                name=label if (r == 1 and c == 1) else None,
                showlegend=(r == 1 and c == 1),
            ), row=r, col=c)

    # mean_max_dd is negative: less-negative (lower DD) is naturally higher on
    # the y-axis without any reversal — no autorange flip needed here.

    # ── bottom: grouped horizontal bar chart of weights ───────────────────────
    # Include all objectives + equal weight baseline
    groups_to_plot: list = []
    for obj_name, res in optimal.items():
        groups_to_plot.append((obj_labels.get(obj_name, obj_name), res['weights']))

    # deduplicate equal weight if it's already in optimal
    if 'max_equal_weight_comparison' not in optimal and equal_res:
        ew = equal_res.get('weights', {})
        groups_to_plot.append(('Equal weight', ew))

    for i_asset, asset in enumerate(asset_names):
        asset_color  = _ASSET_PALETTE[i_asset % len(_ASSET_PALETTE)]
        bar_x = [grp_w.get(asset, 0.0) for _, grp_w in groups_to_plot]
        bar_y = [grp_label for grp_label, _ in groups_to_plot]

        fig.add_trace(go.Bar(
            x=bar_x, y=bar_y,
            name=asset,
            orientation='h',
            marker_color=asset_color,
            opacity=0.85,
        ), row=2, col=1)

    fig.update_layout(
        template=_TEMPLATE,
        title='Portfolio CPCV — Weight Optimisation Results',
        height=700,
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
    )
    fig.update_xaxes(title_text='Mean Sharpe', row=1, col=1)
    fig.update_yaxes(title_text='Conservative Sharpe Floor', row=1, col=1)
    fig.update_xaxes(title_text='Mean Sharpe', row=1, col=2)
    fig.update_yaxes(title_text='Mean Max DD', row=1, col=2)
    fig.update_xaxes(title_text='Weight', tickformat='.0%', row=2, col=1)

    return _save_and_show(fig, show, save_html)


# ──────────────────────────────────────────────────────────────────────────────
#  Function 10: plot_portfolio_full_results  (extended wrapper)
# ──────────────────────────────────────────────────────────────────────────────

def plot_portfolio_full_results(
    portfolio_paths,
    asset_results,
    ci_results       = None,
    weights          = None,
    wf_sharpes       = None,
    div_benefit_df   = None,
    corr_results     = None,
    dd_results       = None,
    optimal_results  = None,
    show             = True,
    save_html_dir    = None,
):
    """
    Extended convenience wrapper.  Calls all available plot functions in
    sequence.  Conditional arguments are silently skipped when None so
    partial calls work cleanly.

    Always called:
        plot_portfolio_equity_curves
        plot_portfolio_distribution
        plot_portfolio_vs_assets

    Called when the corresponding argument is not None:
        plot_diversification_benefit  (div_benefit_df, weights)
        plot_correlation_structure    (corr_results)
        plot_drawdown_decomposition   (dd_results, weights)
        plot_weight_optimisation      (optimal_results, asset_names)

    Parameters
    ----------
    portfolio_paths  : list returned by sample_portfolio_paths()
    asset_results    : dict returned by load_asset_cpcv()
    ci_results       : dict, optional — output of portfolio_confidence_intervals()
    weights          : dict, optional — {"ASSET": weight, ...}
    wf_sharpes       : dict, optional — passed to plot_portfolio_distribution
    div_benefit_df   : pd.DataFrame, optional — output of diversification_benefit()
    corr_results     : dict, optional — output of asset_correlation_structure()
    dd_results       : dict, optional — output of worst_drawdown_decomposition()
    optimal_results  : dict, optional — output of optimal_weights()
    show             : bool
    save_html_dir    : str, optional — directory for HTML exports
    """
    def _html(name):
        if save_html_dir:
            os.makedirs(save_html_dir, exist_ok=True)
            return os.path.join(save_html_dir, name)
        return None

    figs = {}

    figs['equity_curves'] = plot_portfolio_equity_curves(
        portfolio_paths, ci_results=ci_results,
        show=show, save_html=_html('portfolio_equity_curves.html'),
    )
    figs['distribution'] = plot_portfolio_distribution(
        portfolio_paths, ci_results=ci_results,
        asset_results=asset_results, wf_sharpes=wf_sharpes,
        show=show, save_html=_html('portfolio_distribution.html'),
    )
    figs['vs_assets'] = plot_portfolio_vs_assets(
        portfolio_paths, asset_results, weights=weights,
        show=show, save_html=_html('portfolio_vs_assets.html'),
    )

    if div_benefit_df is not None and weights is not None:
        figs['diversification_benefit'] = plot_diversification_benefit(
            div_benefit_df, weights,
            show=show, save_html=_html('portfolio_diversification.html'),
        )

    if corr_results is not None:
        figs['correlation_structure'] = plot_correlation_structure(
            corr_results,
            show=show, save_html=_html('portfolio_correlations.html'),
        )

    if dd_results is not None and weights is not None:
        figs['drawdown_decomposition'] = plot_drawdown_decomposition(
            dd_results, weights,
            show=show, save_html=_html('portfolio_drawdowns.html'),
        )

    if optimal_results is not None:
        asset_names = list(asset_results.keys())
        figs['weight_optimisation'] = plot_weight_optimisation(
            optimal_results, asset_names,
            show=show, save_html=_html('portfolio_weights.html'),
        )

    return figs


# ──────────────────────────────────────────────────────────────────────────────
#  Function 11: plot_yearly_performance
# ──────────────────────────────────────────────────────────────────────────────

def plot_yearly_performance(trade_stats, trade_cis, show=True, save_html=None):
    """
    Two-panel yearly performance chart.

    Row 1 — Bar chart of mean portfolio return per calendar year (green =
      positive, red = negative).  Error bars show the overlap-adjusted 95% CI
      bounds.  A secondary y-axis overlays mean yearly Sharpe as a line with
      markers.

    Row 2 — Heatmap of per-path annual returns (up to 100 sampled paths on the
      y-axis, calendar years on the x-axis).  Diverging RdYlGn colourscale
      centred at 0 reveals the distribution of outcomes per year visually.

    Parameters
    ----------
    trade_stats : dict returned by extract_portfolio_trade_stats()
    trade_cis   : dict returned by trade_stats_confidence_intervals()
    show / save_html : display / export controls
    """
    yr_df      = trade_stats.get('yearly_stats', pd.DataFrame())
    yearly_cis = trade_cis.get('yearly_cis', {})

    if len(yr_df) == 0:
        print('[plot_yearly_performance] No yearly stats available.')
        return None

    years     = sorted(yr_df['year'].unique())
    yr_means  = yr_df.groupby('year')['year_return'].mean()
    yr_sharpe = yr_df.groupby('year')['year_sharpe'].mean()

    # ── CI error bars ─────────────────────────────────────────────────────────
    err_lo = []
    err_hi = []
    for y in years:
        yci = yearly_cis.get(int(y), {}).get('return')
        if yci:
            lo, hi = yci['adjusted_ci']
            mean_r  = float(yr_means[y])
            err_lo.append(mean_r - lo)   # distance below mean
            err_hi.append(hi - mean_r)   # distance above mean
        else:
            err_lo.append(0.0)
            err_hi.append(0.0)

    mean_rets   = [float(yr_means[y])  for y in years]
    mean_sharpe = [float(yr_sharpe[y]) for y in years]
    bar_colors  = [_GREEN if r >= 0 else _RED for r in mean_rets]

    # ── 2-row subplot; row 1 has secondary y-axis ─────────────────────────────
    from plotly.subplots import make_subplots as _msp
    fig = _msp(
        rows=2, cols=1,
        specs=[[{'secondary_y': True}], [{'secondary_y': False}]],
        row_heights=[0.52, 0.48],
        vertical_spacing=0.10,
        subplot_titles=['Yearly Return (mean ± adj CI)', 'Return Distribution by Year'],
    )

    # ── Row 1: return bars ────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=[str(y) for y in years],
        y=[r * 100 for r in mean_rets],
        error_y=dict(
            type='data',
            symmetric=False,
            array=[e * 100 for e in err_hi],
            arrayminus=[e * 100 for e in err_lo],
            color=_GREY,
            thickness=1.5,
            width=6,
        ),
        marker_color=bar_colors,
        opacity=0.85,
        name='Mean Return',
        text=[f'{r * 100:.1f}%' for r in mean_rets],
        textposition='outside',
        textfont=dict(size=9, family=_FONT_MONO),
        hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>',
    ), row=1, col=1, secondary_y=False)

    # ── Row 1: Sharpe line (secondary y-axis) ─────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[str(y) for y in years],
        y=mean_sharpe,
        mode='lines+markers',
        name='Mean Sharpe',
        line=dict(color=_AMBER, width=2),
        marker=dict(size=7, symbol='circle', color=_AMBER),
        hovertemplate='<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>',
    ), row=1, col=1, secondary_y=True)

    fig.add_hline(y=0, line_dash='dash', line_color='#94a3b8',
                  line_width=1, row=1, col=1)

    # ── Row 2: heatmap (subsample ≤100 paths) ────────────────────────────────
    rng         = np.random.default_rng(0)
    all_path_ids = sorted(yr_df['path_id'].unique())
    if len(all_path_ids) > 100:
        sample_ids = sorted(rng.choice(all_path_ids, 100, replace=False).tolist())
    else:
        sample_ids = all_path_ids

    # build (n_paths × n_years) matrix
    yr_pivot = yr_df.pivot(index='path_id', columns='year', values='year_return')
    yr_pivot = yr_pivot.reindex(index=sample_ids, columns=years)

    z_vals    = (yr_pivot.values * 100).tolist()   # convert to %
    y_labels  = [f'P{pid}' for pid in sample_ids]
    x_labels  = [str(y) for y in years]

    # symmetric colour range centred at 0
    abs_max = float(np.nanmax(np.abs(yr_pivot.values))) * 100
    abs_max = max(abs_max, 1.0)

    fig.add_trace(go.Heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-abs_max,
        zmax=abs_max,
        colorbar=dict(title='Return %', thickness=12, len=0.45, y=0.22),
        hovertemplate='<b>%{x}</b><br>Path %{y}<br>Return: %{z:.2f}%<extra></extra>',
        showscale=True,
    ), row=2, col=1)

    fig.update_layout(
        height=800,
        template=_TEMPLATE,
        title=dict(
            text='Portfolio CPCV — Yearly Performance',
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99,
                    bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1),
        hovermode='x unified',
        bargap=0.25,
    )
    fig.update_yaxes(title_text='Return (%)',       row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text='Sharpe',           row=1, col=1, secondary_y=True,
                     showgrid=False)
    fig.update_yaxes(title_text='Sampled Path',     row=2, col=1,
                     showticklabels=False)
    fig.update_xaxes(title_text='Year',             row=2, col=1)

    return _save_and_show(fig, show, save_html)


# ──────────────────────────────────────────────────────────────────────────────
#  Function 12: plot_trade_statistics
# ──────────────────────────────────────────────────────────────────────────────

def plot_trade_statistics(trade_stats, trade_cis, show=True, save_html=None):
    """
    2 × 2 grid of histograms for the four key aggregate trade statistics.

    Each panel shows:
      - Histogram of the per-path distribution (blue, nbinsx=20)
      - Shaded CI band (blue, adjusted 95%) via add_shape
      - Mean vline (amber, solid)
      - Conservative floor vline (red, dashed) — lower bound of adjusted CI

    Panels:
      Top-left:  Win rate distribution
      Top-right: Avg trade return distribution
      Bot-left:  Profit factor distribution
      Bot-right: Avg holding period (bars) distribution

    Parameters
    ----------
    trade_stats : dict returned by extract_portfolio_trade_stats()
    trade_cis   : dict returned by trade_stats_confidence_intervals()
    show / save_html : display / export controls
    """
    agg_df  = trade_stats.get('aggregate_stats', pd.DataFrame())
    agg_cis = trade_cis.get('aggregate_cis', {})

    if len(agg_df) == 0:
        print('[plot_trade_statistics] No aggregate stats available.')
        return None

    panels = [
        (1, 1, 'win_rate',               'Win Rate',                True),
        (1, 2, 'avg_trade_return',        'Avg Trade Return',        True),
        (2, 1, 'profit_factor',           'Profit Factor',           False),
        (2, 2, 'avg_holding_period_bars', 'Avg Holding Period (bars)', False),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[p[3] for p in panels],
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    for row, col, key, label, as_pct in panels:
        if key not in agg_df.columns:
            continue
        vals = (agg_df[key]
                .replace([np.inf, -np.inf], float('nan'))
                .dropna()
                .values
                .astype(float))
        if len(vals) == 0:
            continue

        ci_data  = agg_cis.get(key, {})
        mean_val = ci_data.get('mean', float(np.mean(vals)))
        adj_lo, adj_hi = ci_data.get('adjusted_ci', (float('nan'), float('nan')))
        floor    = ci_data.get('conservative_lower_bound', adj_lo)

        scale   = 100.0 if as_pct else 1.0
        # scale already converts to %; use plain float format and append % manually
        num_fmt = '.1f' if as_pct else '.2f'
        suffix  = '%'  if as_pct else ''
        # hovertemplate uses Plotly d3 format — '.1%' multiplies by 100, so
        # pass the raw (unscaled) value and use '.1%'; or pass scaled with '.1f'
        hover_fmt = '.1f' if as_pct else '.2f'

        def _label(v):
            return f'{v * scale:{num_fmt}}{suffix}'

        # histogram (x values already scaled to %)
        fig.add_trace(go.Histogram(
            x=(vals * scale).tolist(),
            nbinsx=20,
            marker_color=_BLUE,
            opacity=0.75,
            name=label,
            showlegend=False,
            hovertemplate=f'%{{x:{hover_fmt}}}{suffix}: %{{y}}<extra></extra>',
        ), row=row, col=col)

        # CI band (add_shape with paper coordinates relative to subplot)
        if not (math.isnan(adj_lo) or math.isnan(adj_hi)):
            xref = 'x' if (row == 1 and col == 1) else f'x{(row - 1) * 2 + col}'
            fig.add_shape(
                type='rect',
                xref=xref, yref='paper',
                x0=adj_lo * scale, x1=adj_hi * scale,
                y0=0, y1=1,
                fillcolor='rgba(59, 130, 246, 0.18)',
                line=dict(width=0),
                layer='below',
            )

        # mean vline
        if not math.isnan(mean_val):
            fig.add_vline(
                x=mean_val * scale,
                line_color=_AMBER, line_width=2, line_dash='solid',
                row=row, col=col,
                annotation_text=f'Mean<br>{_label(mean_val)}',
                annotation_position='top right',
                annotation_font=dict(size=9, family=_FONT_MONO),
            )

        # floor vline
        if not math.isnan(floor) and floor != mean_val:
            fig.add_vline(
                x=floor * scale,
                line_color=_RED, line_width=1.5, line_dash='dash',
                row=row, col=col,
                annotation_text=f'Floor<br>{_label(floor)}',
                annotation_position='top left',
                annotation_font=dict(size=9, family=_FONT_MONO, color=_RED),
            )

        x_title = label + (' (%)' if as_pct else '')
        fig.update_xaxes(title_text=x_title, row=row, col=col,
                         showgrid=True, gridwidth=1, gridcolor=_GRID)
        fig.update_yaxes(title_text='Count', row=row, col=col,
                         showgrid=True, gridwidth=1, gridcolor=_GRID)

    fig.update_layout(
        height=700,
        template=_TEMPLATE,
        title=dict(
            text='Portfolio CPCV — Trade Statistics Distribution',
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
    )

    return _save_and_show(fig, show, save_html)


# ──────────────────────────────────────────────────────────────────────────────
#  Function 13: plot_asset_yearly_heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_asset_yearly_heatmap(breakdown, show=True, save_html=None):
    """
    Three stacked heatmaps — assets on y-axis, calendar years on x-axis.

    Top    — Mean annual return per (asset, year).  Diverging RdYlGn centred
             at 0.  A final 'Portfolio' row shows portfolio mean return.
             Cells annotated with return as percentage.

    Middle — % paths positive per (asset, year).  RdYlGn centred at 50%.
             Cells annotated with percentage.

    Bottom — Asset rank by year (1 = best return that year). Reversed RdYlGn
             so rank 1 is green.  Cells annotated with rank number.

    Parameters
    ----------
    breakdown : dict returned by per_asset_yearly_breakdown()
    show / save_html : display / export controls
    """
    asset_summary = breakdown.get('asset_year_summary', pd.DataFrame())
    port_summary  = breakdown.get('portfolio_year_summary', pd.DataFrame())
    asset_ranks   = breakdown.get('asset_ranks', pd.DataFrame())

    if len(asset_summary) == 0:
        print('[plot_asset_yearly_heatmap] No data.')
        return None

    # ── pivot tables ──────────────────────────────────────────────────────────
    ret_piv  = asset_summary.pivot(index='asset', columns='year',
                                   values='mean_return')
    pos_piv  = asset_summary.pivot(index='asset', columns='year',
                                   values='pct_paths_positive')
    rank_piv = asset_ranks.pivot(index='asset', columns='year', values='rank')

    years     = sorted(ret_piv.columns.tolist())
    assets    = ret_piv.index.tolist()
    n_assets  = len(assets)
    year_strs = [str(y) for y in years]

    # ── Top panel: return matrix (assets + Portfolio row) ─────────────────────
    ret_mat  = ret_piv[years].values
    ret_text = [[f'{v * 100:.1f}%' if not np.isnan(v) else ''
                 for v in row] for row in ret_mat]

    port_row = np.array([
        float(port_summary.loc[port_summary['year'] == y, 'mean_return'].iloc[0])
        if y in port_summary['year'].values else float('nan')
        for y in years
    ])
    port_text = [f'{v * 100:.1f}%' if not np.isnan(v) else '' for v in port_row]

    top_z    = np.vstack([ret_mat, port_row])
    top_text = ret_text + [port_text]
    top_ys   = assets + ['Portfolio']

    all_ret  = top_z[~np.isnan(top_z)]
    abs_max  = max(float(np.abs(all_ret).max()), 0.01) if len(all_ret) else 0.5

    # ── Middle panel: pct_paths_positive ─────────────────────────────────────
    pos_mat  = pos_piv[years].values
    pos_text = [[f'{v * 100:.0f}%' if not np.isnan(v) else ''
                 for v in row] for row in pos_mat]

    # ── Bottom panel: rank ────────────────────────────────────────────────────
    rank_mat  = rank_piv[years].values.astype(float)
    rank_text = [[str(int(v)) if not np.isnan(v) else ''
                  for v in row] for row in rank_mat]
    rank_max  = int(np.nanmax(rank_mat)) if not np.all(np.isnan(rank_mat)) else n_assets

    # ── subplots — 3 rows, shared x-axis ─────────────────────────────────────
    # row heights proportional to cell count so each cell is equal size
    n_top   = n_assets + 1   # assets + Portfolio
    n_mid   = n_assets
    n_bot   = n_assets
    n_total = n_top + n_mid + n_bot
    rh = [n_top / n_total, n_mid / n_total, n_bot / n_total]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            'Mean Annual Return by Asset',
            '% Paths Positive',
            'Asset Rank by Year',
        ],
        row_heights=rh,
    )

    # ── Row 1: return heatmap ─────────────────────────────────────────────────
    fig.add_trace(go.Heatmap(
        z=top_z,
        x=year_strs,
        y=top_ys,
        text=top_text,
        texttemplate='%{text}',
        textfont=dict(size=10, family=_FONT_MONO),
        colorscale='RdYlGn',
        zmid=0,
        zmin=-abs_max,
        zmax=abs_max,
        showscale=True,
        colorbar=dict(
            title='Return',
            tickformat='.0%',
            thickness=12,
            len=rh[0],
            y=1 - rh[0] / 2,
            yanchor='middle',
        ),
        hovertemplate='<b>%{y} — %{x}</b><br>Return: %{z:.2%}<extra></extra>',
    ), row=1, col=1)

    fig.update_yaxes(autorange='reversed', row=1, col=1)

    # ── Row 2: pct_paths_positive heatmap ────────────────────────────────────
    fig.add_trace(go.Heatmap(
        z=pos_mat,
        x=year_strs,
        y=assets,
        text=pos_text,
        texttemplate='%{text}',
        textfont=dict(size=10, family=_FONT_MONO),
        colorscale='RdYlGn',
        zmid=0.5,
        zmin=0.0,
        zmax=1.0,
        showscale=True,
        colorbar=dict(
            title='% Positive',
            tickformat='.0%',
            thickness=12,
            len=rh[1],
            y=rh[2] + rh[1] / 2,
            yanchor='middle',
        ),
        hovertemplate='<b>%{y} — %{x}</b><br>Positive: %{z:.1%}<extra></extra>',
    ), row=2, col=1)

    fig.update_yaxes(autorange='reversed', row=2, col=1)

    # ── Row 3: rank heatmap ───────────────────────────────────────────────────
    fig.add_trace(go.Heatmap(
        z=rank_mat,
        x=year_strs,
        y=assets,
        text=rank_text,
        texttemplate='%{text}',
        textfont=dict(size=10, family=_FONT_MONO),
        colorscale='RdYlGn',
        reversescale=True,       # rank 1 (best) → green
        zmin=1,
        zmax=rank_max,
        showscale=True,
        colorbar=dict(
            title='Rank',
            thickness=12,
            len=rh[2],
            y=rh[2] / 2,
            yanchor='middle',
        ),
        hovertemplate='<b>%{y} — %{x}</b><br>Rank: %{z:.0f}<extra></extra>',
    ), row=3, col=1)

    fig.update_yaxes(autorange='reversed', row=3, col=1)
    fig.update_xaxes(title_text='Year', row=3, col=1)

    # height: ~38px per cell row + margins
    cell_px = 38
    height  = max(500, (n_top + n_mid + n_bot) * cell_px + 220)

    fig.update_layout(
        height=height,
        template=_TEMPLATE,
        title=dict(
            text='Portfolio CPCV — Per-Asset Yearly Performance',
            font=dict(size=22, color='#1E293B'),
            x=0.5, xanchor='center',
        ),
        margin=dict(l=80, r=120, t=80, b=60),
    )

    return _save_and_show(fig, show, save_html)
