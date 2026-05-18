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

    # Per-leg coin counts (only present in newer strategy_fn outputs)
    if 'n_long_held' in oos_df.columns:
        nl = oos_df['n_long_held'][oos_df['n_long_held'] > 0]
        if len(nl) > 0:
            metrics['avg_n_long'] = float(nl.mean())
            metrics['min_n_long'] = int(nl.min())
            metrics['max_n_long'] = int(nl.max())

    if 'n_short_held' in oos_df.columns:
        ns = oos_df['n_short_held'][oos_df['n_short_held'] > 0]
        if len(ns) > 0:
            metrics['avg_n_short'] = float(ns.mean())
            metrics['min_n_short'] = int(ns.min())
            metrics['max_n_short'] = int(ns.max())

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

    # ── Panel 3: turnover (stacked long/short if available) ──────────────────
    if has_to:
        rb_mask     = oos_df['turnover'] > 0
        has_leg_to  = ('long_turnover' in oos_df.columns and 'short_turnover' in oos_df.columns)
        if has_leg_to:
            fig.add_trace(go.Bar(
                x=oos_df.index[rb_mask], y=oos_df.loc[rb_mask, 'long_turnover'],
                name='Long turnover',  marker_color='green',
            ), row=next_row, col=1)
            fig.add_trace(go.Bar(
                x=oos_df.index[rb_mask], y=oos_df.loc[rb_mask, 'short_turnover'],
                name='Short turnover', marker_color='red',
            ), row=next_row, col=1)
        else:
            fig.add_trace(go.Bar(x=oos_df.index[rb_mask], y=oos_df.loc[rb_mask, 'turnover'],
                                  name='Turnover', marker_color='orange',
                                  showlegend=False),
                          row=next_row, col=1)
        next_row += 1

    # ── Panel 4: universe size + per-leg coin counts ─────────────────────────
    if has_uni:
        u = oos_df['universe_size'].replace(0, np.nan)
        fig.add_trace(go.Scatter(x=u.index, y=u.values,
                                  name='Universe size', line=dict(color='purple'),
                                  mode='lines+markers'),
                      row=next_row, col=1)
        # Overlay per-leg coin counts if present (newer strategy_fn outputs)
        if 'n_long_held' in oos_df.columns and 'n_short_held' in oos_df.columns:
            nl = oos_df['n_long_held'].replace(0, np.nan)
            ns = oos_df['n_short_held'].replace(0, np.nan)
            fig.add_trace(go.Scatter(x=nl.index, y=nl.values,
                                      name='n_long held',
                                      line=dict(color='green', dash='dot'),
                                      mode='lines'),
                          row=next_row, col=1)
            fig.add_trace(go.Scatter(x=ns.index, y=ns.values,
                                      name='n_short held',
                                      line=dict(color='red', dash='dot'),
                                      mode='lines'),
                          row=next_row, col=1)

    fig.update_layout(height=280 * n_rows, hovermode='x unified', title=title,
                       barmode='stack',   # long/short turnover bars stack on top of each other
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


# ── XS-specific regime diagnostics ───────────────────────────────────────────

def bear_hedge_diagnostic(oos_df, btc_close, ann=365, verbose=True):
    """
    Conditional-regime performance check for dollar-neutral L/S strategies.

    Splits OOS bars by sign of BTC's daily return and computes per-regime alphas
    plus strategy Sharpe.  A genuine bear hedge satisfies:
      short_α(down) > short_α(up)   AND   strat_Sharpe(down) > 0

    Parameters
    ----------
    oos_df    : combined OOS DataFrame (must contain long_ret, short_ret,
                strategy_returns).  Typically `results['oos_combined_df']`.
    btc_close : pd.Series of BTC Close prices, OR a panel containing a 'BTC'
                column.  The series is aligned to oos_df.index via reindex.
    ann       : annualisation factor (365 daily crypto, 252 stocks).
    verbose   : print the formatted summary (default True).

    Returns
    -------
    dict with keys:
        up_long_alpha, down_long_alpha, all_long_alpha
        up_short_alpha, down_short_alpha, all_short_alpha
        up_strat_sharpe, down_strat_sharpe, all_strat_sharpe
        n_up, n_down, n_flat
        corr_strategy_btc, hedge_works
    """
    if isinstance(btc_close, pd.DataFrame):
        if 'BTC' in btc_close.columns:
            btc_close = btc_close['BTC']
        elif 'Close' in btc_close.columns:
            btc_close = btc_close['Close']
        else:
            raise ValueError("btc_close DataFrame must have 'BTC' or 'Close' column")

    btc_ret = btc_close.pct_change().reindex(oos_df.index).fillna(0)

    up_mask   = btc_ret > 0
    flat_mask = btc_ret == 0
    down_mask = btc_ret < 0

    long_ret  = oos_df['long_ret']
    short_ret = oos_df['short_ret']
    strat     = oos_df['strategy_returns']

    def _row(mask):
        n = int(mask.sum())
        if n == 0:
            return dict(n=0, long_alpha=0.0, short_alpha=0.0,
                        strat_mean=0.0, strat_std=0.0, strat_sharpe=0.0)
        long_alpha   = (long_ret  - btc_ret)[mask].mean() * ann
        short_alpha  = (btc_ret   - short_ret)[mask].mean() * ann
        strat_mean   = strat[mask].mean() * ann
        strat_std    = strat[mask].std()  * np.sqrt(ann)
        strat_sharpe = strat_mean / strat_std if strat_std > 0 else 0.0
        return dict(n=n, long_alpha=long_alpha, short_alpha=short_alpha,
                    strat_mean=strat_mean, strat_std=strat_std,
                    strat_sharpe=strat_sharpe)

    up_stats   = _row(up_mask)
    down_stats = _row(down_mask)
    all_stats  = _row(pd.Series(True, index=oos_df.index))

    corr = float(strat.corr(btc_ret))
    sa_up   = up_stats['short_alpha']
    sa_down = down_stats['short_alpha']
    hedge_works = (sa_down > sa_up) and (sa_down > 0)

    out = {
        'up_long_alpha':   up_stats['long_alpha'],
        'down_long_alpha': down_stats['long_alpha'],
        'all_long_alpha':  all_stats['long_alpha'],
        'up_short_alpha':  up_stats['short_alpha'],
        'down_short_alpha': down_stats['short_alpha'],
        'all_short_alpha': all_stats['short_alpha'],
        'up_strat_sharpe':   up_stats['strat_sharpe'],
        'down_strat_sharpe': down_stats['strat_sharpe'],
        'all_strat_sharpe':  all_stats['strat_sharpe'],
        'n_up':   up_stats['n'],
        'n_down': down_stats['n'],
        'n_flat': int(flat_mask.sum()),
        'corr_strategy_btc': corr,
        'hedge_works': bool(hedge_works),
    }

    if verbose:
        print('═' * 86)
        print('BEAR-HEDGE DIAGNOSTIC — Performance by BTC regime  (annualised)')
        print('═' * 86)
        print(f'{"Regime":<20}{"Bars":>6}  {"Long alpha vs BTC":>18}  '
              f'{"Short alpha vs BTC":>20}  {"Strat Sharpe":>14}')
        print('─' * 86)
        for label, st in [('BTC up days', up_stats),
                          ('BTC down days', down_stats),
                          ('All bars', all_stats)]:
            if st['n'] == 0:
                print(f'  {label:<18}  bars=   0  (no data)')
                continue
            print(f'  {label:<18}  bars={st["n"]:>4}  '
                  f'long_α={st["long_alpha"]*100:>+7.1f}%  '
                  f'short_α={st["short_alpha"]*100:>+7.1f}%  '
                  f'strat_Sharpe={st["strat_sharpe"]:>+5.2f}')

        if abs(corr) < 0.15:
            corr_label = 'dollar-neutral holding'
        elif corr > 0:
            corr_label = 'POSITIVE BTC beta leakage'
        else:
            corr_label = 'NEGATIVE BTC tilt (net short bias)'
        print(f'\ncorr(strategy_returns, BTC) = {corr:+.3f}   ({corr_label})')

        verdict = ('✓ short leg behaves like a bear hedge'
                   if hedge_works else
                   '✗ short leg does NOT add value in BTC-down regimes')
        print(f'\nVerdict: {verdict}')
        print(f'         (short_α: BTC-up = {sa_up*100:+.1f}%, '
              f'BTC-down = {sa_down*100:+.1f}%)')

    return out


def regime_quadrant_diagnostic(oos_df, btc_close,
                                ma_window=200, er_window=14, ann=365,
                                verbose=True, plot=True, show=True,
                                save_html=None):
    """
    Performance split by BTC trend (above/below MA) × trend-strength
    (Kaufman Efficiency Ratio above/below median).

    The four quadrants:
        Bull / trending   Bull / chop
        Bear / trending   Bear / chop

    Reveals where the strategy makes its money vs where it bleeds — a
    prerequisite to designing regime-conditional filters.

    Parameters
    ----------
    oos_df    : combined OOS DataFrame (must contain strategy_returns).
    btc_close : pd.Series of BTC Close prices, OR DataFrame containing 'BTC'
                or 'Close' column.
    ma_window : MA window for the bull/bear discriminator (200 = 200d).
    er_window : Kaufman Efficiency Ratio lookback (14 = standard).
    ann       : annualisation factor (365 daily crypto, 252 stocks).
    verbose   : print the per-quadrant table + verdict.
    plot      : produce the equity + regime-timeline chart.
    show      : display the chart inline (Jupyter).
    save_html : optional path to write a standalone HTML file.

    Returns
    -------
    dict with keys:
        quadrant      : pd.Series of quadrant labels (oos_df-aligned)
        stats         : dict {quadrant: {pct, n, ann_ret, sharpe, total_return}}
        er_threshold  : float (median ER over OOS window)
        verdict       : str
        figure        : plotly Figure or None
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if isinstance(btc_close, pd.DataFrame):
        if 'BTC' in btc_close.columns:
            btc_close = btc_close['BTC']
        elif 'Close' in btc_close.columns:
            btc_close = btc_close['Close']
        else:
            raise ValueError("btc_close DataFrame must have 'BTC' or 'Close' column")

    btc_ma     = btc_close.rolling(ma_window).mean()
    bull_mask  = (btc_close > btc_ma).reindex(oos_df.index).fillna(False)

    direction  = btc_close.diff(er_window).abs()
    path_total = btc_close.diff().abs().rolling(er_window).sum()
    er_full    = (direction / path_total.replace(0, np.nan)).fillna(0)
    er         = er_full.reindex(oos_df.index)

    er_pos     = er[er > 0]
    er_thresh  = float(er_pos.median()) if len(er_pos) > 0 else 0.3
    trend_mask = er >= er_thresh

    quadrant = pd.Series('—', index=oos_df.index, dtype='object')
    quadrant.loc[ bull_mask &  trend_mask] = 'Bull / trending'
    quadrant.loc[ bull_mask & ~trend_mask] = 'Bull / chop'
    quadrant.loc[~bull_mask &  trend_mask] = 'Bear / trending'
    quadrant.loc[~bull_mask & ~trend_mask] = 'Bear / chop'

    quad_names = ['Bull / trending', 'Bull / chop', 'Bear / trending', 'Bear / chop']
    stats = {}
    for q in quad_names:
        mask = quadrant == q
        n    = int(mask.sum())
        if n == 0:
            stats[q] = dict(pct=0.0, n=0, ann_ret=0.0, sharpe=0.0, total_return=0.0)
            continue
        pct      = n / len(oos_df) * 100
        rets     = oos_df['strategy_returns'][mask]
        ann_ret  = rets.mean() * ann
        ann_vol  = rets.std()  * np.sqrt(ann)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0
        tot_ret  = float((1 + rets).prod() - 1)
        stats[q] = dict(pct=pct, n=n, ann_ret=ann_ret, sharpe=sharpe,
                        total_return=tot_ret)

    n_neg = sum(1 for s in stats.values() if s['total_return'] < 0)
    if n_neg == 0:
        verdict = '✓ strategy positive in all four quadrants — no regime filter needed'
    elif n_neg == 1:
        bad = next(q for q, s in stats.items() if s['total_return'] < 0)
        verdict = f'regime filter candidate — turn strategy OFF in {bad!r}'
    elif n_neg == 2:
        bad = [q for q, s in stats.items() if s['total_return'] < 0]
        verdict = f'strategy struggles in {bad}; consider signal redesign over filter'
    else:
        verdict = 'strategy is fragile across regimes — filter alone unlikely to help'

    if verbose:
        print('═' * 86)
        print('REGIME QUADRANT DIAGNOSTIC — performance by BTC trend × trend-strength')
        print('═' * 86)
        print(f'BTC trend split:      bull = above {ma_window}d MA   bear = below {ma_window}d MA')
        print(f'Trend-strength split: median Efficiency Ratio over OOS = {er_thresh:.3f}')
        print()
        print(f'{"Quadrant":<22}{"% OOS":>9}{"Bars":>7}{"Ann return":>14}'
              f'{"Sharpe":>10}{"Total return":>16}')
        print('─' * 86)
        for q in quad_names:
            s = stats[q]
            if s['n'] == 0:
                print(f'  {q:<20}{"—":>9}{0:>7}{"—":>14}{"—":>10}{"—":>16}')
                continue
            print(f'  {q:<20}{s["pct"]:>8.1f}%{s["n"]:>7}'
                  f'{s["ann_ret"]*100:>+13.1f}%{s["sharpe"]:>+10.2f}'
                  f'{s["total_return"]*100:>+15.1f}%')
        print()
        print(f'Verdict: {verdict}')

    fig = None
    if plot:
        equity = (1 + oos_df['strategy_returns'].fillna(0)).cumprod().clip(lower=0)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.62, 0.38], vertical_spacing=0.08,
                            subplot_titles=('Strategy equity (clipped at 0)',
                                            'BTC regime quadrant per OOS bar'))

        fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name='Strategy',
                                  line=dict(color='black', width=2), showlegend=False),
                      row=1, col=1)
        fig.add_hline(y=1.0, line=dict(color='gray', dash='dash'), row=1, col=1)

        quadrant_levels = {'Bull / trending': 4, 'Bull / chop': 3,
                           'Bear / trending': 2, 'Bear / chop': 1}
        quadrant_colors = {'Bull / trending': '#1abc9c', 'Bull / chop':     '#a3e4b8',
                           'Bear / trending': '#c0392b', 'Bear / chop':     '#f5a8a0'}
        for q, lvl in quadrant_levels.items():
            mask = quadrant == q
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=oos_df.index[mask], y=[lvl] * int(mask.sum()),
                    mode='markers', marker=dict(color=quadrant_colors[q], size=5),
                    name=q, hovertemplate=f'{q}<br>%{{x|%Y-%m-%d}}<extra></extra>',
                ), row=2, col=1)

        fig.update_yaxes(title_text='Equity', row=1, col=1)
        fig.update_yaxes(title_text='Regime', row=2, col=1,
                         tickvals=[1, 2, 3, 4],
                         ticktext=['Bear / chop', 'Bear / trending',
                                   'Bull / chop',  'Bull / trending'])
        fig.update_layout(height=620, hovermode='x unified',
                          title='Strategy equity vs BTC regime quadrants',
                          legend=dict(orientation='h', yanchor='bottom',
                                      y=1.02, xanchor='left', x=0))
        if save_html:
            fig.write_html(save_html)
        if show:
            fig.show()

    return {
        'quadrant':     quadrant,
        'stats':        stats,
        'er_threshold': er_thresh,
        'verdict':      verdict,
        'figure':       fig,
    }
