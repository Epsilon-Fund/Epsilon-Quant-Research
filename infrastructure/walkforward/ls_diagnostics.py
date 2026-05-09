"""
ls_diagnostics.py — Long/short attribution diagnostics for walk-forward output.

Designed for cross-sectional and pair-trading strategies that produce explicit
long_ret / short_ret columns alongside the standard strategy_returns and turnover.
Trade-centric metrics (num_trades, win_rate, profit_factor) are meaningless for
fixed-schedule rebalanced portfolios; these diagnostics replace them.

Conventions (the strategy must produce these columns):
    long_ret       : per-bar arithmetic return of the long-leg basket
                     (mean of long coins' bar returns; positive when longs go up)
    short_ret      : per-bar return of the short-leg basket
                     (mean of short coins' bar returns; you LOSE this when held short)
    turnover       : fractional portfolio turnover at each rebalance bar (0 elsewhere)
    universe_size  : number of coins available at each rebalance (optional)

A 50/50 dollar-neutral spread strategy earns:
    strategy_returns = 0.5 * long_ret - 0.5 * short_ret

Public API:
    compute_attribution(oos_df, ann=365)  →  pd.Series of metrics
    plot_attribution(oos_df, ...)         →  4-panel diagnostic chart
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── helpers ───────────────────────────────────────────────────────────────────

def _sharpe(returns, ann=365):
    """Annualised Sharpe of a return series. Drops NaN; returns 0 if std=0."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    s = r.std()
    return float(r.mean() / s * np.sqrt(ann)) if s > 0 else 0.0


def _per_rebalance_spread(oos_df, spread):
    """
    Aggregate the spread series into per-rebalance-period values.
    A rebalance is identified as a bar where turnover > 0.
    Each rebalance's spread = sum of `spread` from this rebalance up to (but not
    including) the next rebalance.
    """
    if 'turnover' not in oos_df.columns:
        return pd.Series(dtype=float)

    rb_idx = oos_df.index[oos_df['turnover'] > 0]
    if len(rb_idx) == 0:
        return pd.Series(dtype=float)

    out = []
    for i, dt in enumerate(rb_idx):
        end = rb_idx[i + 1] if i + 1 < len(rb_idx) else oos_df.index[-1] + pd.Timedelta(seconds=1)
        out.append(spread.loc[dt:end].sum())
    return pd.Series(out, index=rb_idx, name='spread_per_rebalance')


# ── attribution metrics ───────────────────────────────────────────────────────

def compute_attribution(oos_df, ann=365):
    """
    Long/short attribution metrics from a combined OOS dataframe.

    Parameters
    ----------
    oos_df : pd.DataFrame — typically `results['oos_combined_df']` from walk_forward.
             Must contain `long_ret` and `short_ret` columns (per-bar basket returns).
             Optional: `turnover`, `universe_size`, `net_returns`.
    ann    : annualisation factor for Sharpe (365 for daily crypto, 252 for stocks).

    Returns
    -------
    pd.Series with index:
      long_sharpe        — Sharpe of being 100% long the long basket
      short_sharpe       — Sharpe of being 100% short the short basket  (Sharpe of -short_ret)
      spread_sharpe      — Sharpe of (long_ret - short_ret), i.e. dollar-neutral spread (1× leverage)
      net_sharpe         — Sharpe of `net_returns` (post-cost, includes 0.5/0.5 weighting)
      long_total_return  — cumulative long-leg return
      short_basket_total — cumulative return of being SHORT the short basket
      spread_total       — cumulative spread return (long − short, dollar-neutral)
      hit_rate           — fraction of rebalances where spread sum > 0
      n_rebalances       — number of rebalance events
      mean_turnover      — average turnover per rebalance (range 0–2)
      avg_universe_size  — mean coins available at rebalance (if column present)
      min/max_universe_size — range of universe sizes
    """
    for col in ('long_ret', 'short_ret'):
        if col not in oos_df.columns:
            raise ValueError(
                f"oos_df missing '{col}'. Available columns: {list(oos_df.columns)}"
            )

    long_ret  = oos_df['long_ret'].fillna(0)
    short_ret = oos_df['short_ret'].fillna(0)
    spread    = long_ret - short_ret

    metrics = {
        'long_sharpe'       : _sharpe(long_ret,    ann),
        'short_sharpe'      : _sharpe(-short_ret,  ann),
        'spread_sharpe'     : _sharpe(spread,      ann),
        'long_total_return' : float((1 + long_ret).prod()  - 1),
        'short_basket_total': float((1 - short_ret).prod() - 1),
        'spread_total'      : float((1 + spread).prod()    - 1),
    }

    if 'net_returns' in oos_df.columns:
        metrics['net_sharpe'] = _sharpe(oos_df['net_returns'].fillna(0), ann)

    sp_per_rb = _per_rebalance_spread(oos_df, spread)
    if len(sp_per_rb) > 0:
        metrics['hit_rate']     = float((sp_per_rb > 0).mean())
        metrics['n_rebalances'] = int(len(sp_per_rb))
        metrics['mean_turnover'] = float(oos_df.loc[oos_df['turnover'] > 0, 'turnover'].mean())

    if 'universe_size' in oos_df.columns:
        u = oos_df['universe_size'][oos_df['universe_size'] > 0]
        if len(u) > 0:
            metrics['avg_universe_size'] = float(u.mean())
            metrics['min_universe_size'] = int(u.min())
            metrics['max_universe_size'] = int(u.max())

    return pd.Series(metrics, name='value')


# ── chart ─────────────────────────────────────────────────────────────────────

def plot_attribution(oos_df, benchmark_data=None,
                     show=True, save_html=None, title='Long/Short Attribution'):
    """
    4-panel diagnostic chart:

      1. Cumulative equity by leg — long basket, short basket (held short), combined net
      2. Cumulative ALPHA per leg vs benchmark (excess return over equal-weight basket)
         — strips market beta out and shows pure ranking-signal contribution
      3. Turnover per rebalance bar (bar chart at rebalance points)
      4. Universe size over time  (only included if 'universe_size' column exists)

    The alpha panel (panel 2) replaces the older spread panel.  Cumulative spread
    is geometrically equivalent to your strategy equity at higher leverage and was
    redundant with the wf_visualizer OOS chart.  Alpha-vs-benchmark answers a
    different and more useful question: how much of each leg's return comes from
    the *ranking signal* vs from market direction?

      long_alpha  = (long_ret  − benchmark_ret).cumsum()    rising ⇒ top picks beat the universe
      short_alpha = (benchmark_ret − short_ret).cumsum()    rising ⇒ bottom picks lose to the universe
      both rising  ⇒ ranking works on both ends (real alpha)
      one rising   ⇒ signal only works on one side
      both flat    ⇒ no ranking edge; returns are pure beta

    Parameters
    ----------
    oos_df         : pd.DataFrame — `results['oos_combined_df']`
    benchmark_data : pd.DataFrame with 'Close' column, OR pd.Series of prices.
                     Typically the equal-weight universe basket.  If None, the
                     alpha panel is skipped (panel becomes a 3-panel chart).
    show           : display the chart inline (Jupyter)
    save_html      : optional path to write a standalone HTML file
    title          : chart title
    """
    has_uni  = 'universe_size' in oos_df.columns
    has_to   = 'turnover'      in oos_df.columns
    has_bm   = benchmark_data  is not None

    panels = ['Cumulative equity by leg (long, −short, combined)']
    if has_bm:  panels.append('Cumulative alpha vs equal-weight basket (beta stripped)')
    if has_to:  panels.append('Turnover per rebalance')
    if has_uni: panels.append('Universe size at each rebalance')
    n_rows = len(panels)

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        subplot_titles=panels, vertical_spacing=0.06)

    long_ret  = oos_df['long_ret'].fillna(0)
    short_ret = oos_df['short_ret'].fillna(0)

    # Clip equity at 0 — (1+r).cumprod() can go below 0 when a single bar has
    # r < -1 (e.g. a shorted memecoin pumps several hundred percent). In reality
    # liquidation would have ended the position at 0; the cumprod math doesn't
    # know that. Cosmetic only — does not affect attribution metrics.
    long_eq  = (1 + long_ret).cumprod().clip(lower=0)
    short_eq = (1 - short_ret).cumprod().clip(lower=0)   # equity from being short the short basket

    # ── Panel 1: cumulative equity by leg ────────────────────────────────────
    fig.add_trace(go.Scatter(x=long_eq.index,  y=long_eq.values,
                              name='Long leg',          line=dict(color='green')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=short_eq.index, y=short_eq.values,
                              name='Short leg (held short)', line=dict(color='red')),
                  row=1, col=1)
    if 'net_returns' in oos_df.columns:
        net_eq = (1 + oos_df['net_returns'].fillna(0)).cumprod().clip(lower=0)
        fig.add_trace(go.Scatter(x=net_eq.index, y=net_eq.values,
                                  name='Combined (net of costs)',
                                  line=dict(color='black', width=2)),
                      row=1, col=1)
    fig.add_hline(y=1.0, line=dict(color='gray', dash='dash'), row=1, col=1)

    next_row = 2

    # ── Panel 2: cumulative alpha vs benchmark ───────────────────────────────
    if has_bm:
        # Accept either a DataFrame with 'Close' or a price Series
        if isinstance(benchmark_data, pd.DataFrame) and 'Close' in benchmark_data.columns:
            bm_close = benchmark_data['Close']
        else:
            bm_close = benchmark_data
        bm_ret = bm_close.pct_change().reindex(oos_df.index, method='nearest').fillna(0)

        long_alpha  = (long_ret  - bm_ret).cumsum()
        short_alpha = (bm_ret    - short_ret).cumsum()

        fig.add_trace(go.Scatter(x=long_alpha.index,  y=long_alpha.values,
                                  name='Long alpha (vs basket)',
                                  line=dict(color='green', dash='dot')),
                      row=next_row, col=1)
        fig.add_trace(go.Scatter(x=short_alpha.index, y=short_alpha.values,
                                  name='Short alpha (vs basket)',
                                  line=dict(color='red',   dash='dot')),
                      row=next_row, col=1)
        fig.add_hline(y=0.0, line=dict(color='gray', dash='dash'), row=next_row, col=1)
        next_row += 1

    # ── Panel 3: turnover ────────────────────────────────────────────────────
    if has_to:
        rb = oos_df[oos_df['turnover'] > 0]
        fig.add_trace(go.Bar(x=rb.index, y=rb['turnover'],
                              name='Turnover', marker_color='orange',
                              showlegend=False),
                      row=next_row, col=1)
        next_row += 1

    # ── Panel 4: universe size ───────────────────────────────────────────────
    if has_uni:
        u = oos_df['universe_size'].replace(0, np.nan)
        fig.add_trace(go.Scatter(x=u.index, y=u.values,
                                  name='Universe size', line=dict(color='purple'),
                                  mode='lines+markers', showlegend=False),
                      row=next_row, col=1)

    fig.update_layout(height=280 * n_rows, hovermode='x unified', title=title,
                       legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                   xanchor='left',  x=0))
    fig.update_yaxes(title_text='Equity',           row=1, col=1)
    panel_idx = 2
    if has_bm:
        fig.update_yaxes(title_text='Cum. alpha',   row=panel_idx, col=1); panel_idx += 1
    if has_to:
        fig.update_yaxes(title_text='Turnover',     row=panel_idx, col=1); panel_idx += 1
    if has_uni:
        fig.update_yaxes(title_text='# coins',      row=panel_idx, col=1)

    if save_html:
        fig.write_html(save_html)
    if show:
        fig.show()

    return fig
