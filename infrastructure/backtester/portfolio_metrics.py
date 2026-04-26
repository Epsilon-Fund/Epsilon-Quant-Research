"""
portfolio_metrics.py
====================
Shared helpers for combined-portfolio analysis notebooks.

Covers the three cross-cutting concerns that appear in both
epsilon_portfolio.ipynb and combined_portfolio.py:

  1. PKL loading      — numpy-compat unpickler
  2. Return building  — unified sleeve schema, momentum bar returns
  3. Weighting        — normalisation, stat-arb inverse-vol, momentum 3-level
  4. Simulations      — realized-sizing equity curve

Public API
----------
  load_pkl(path)                               → pd.DataFrame
  mom_bar_returns(df, cost)                    → pd.Series
  wrap_as_sleeve(bar_returns)                  → pd.DataFrame
  sleeve_freq(df)                              → 'hourly' | 'daily'
  norm_weights(d)                              → dict
  sa_inverse_vol_weights(dfs, method, window)  → dict
  build_momentum_weights(mom_dfs, mom_sel,
                         strat_weights,
                         coin_weights)         → dict
  build_sleeve_weights(sa_dfs, sa_w,
                       mom_dfs, mom_w,
                       strategy_weights)       → dict
  build_realized_equity(position_dfs, weights,
                        cost)                  → pd.Series
  sweep_top_level(sa_equity, mom_equity,
                  step, current_sa_weight)     → list[dict]
  sweep_momentum_strategy(mom_dfs, mom_sel,
                          strat_weights_grid,
                          coin_weights, cost)  → list[dict]
"""

from __future__ import annotations

import contextlib
import io
import pickle

import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
#  1. PKL loading
# ══════════════════════════════════════════════════════════════════════════════

class _NumpyCompat(pickle.Unpickler):
    """Unpickler that remaps numpy._core → numpy.core for older PKLs."""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_pkl(path: str) -> pd.DataFrame:
    """Load a PKL file, handling numpy._core compatibility for older files."""
    try:
        return pd.read_pickle(path)
    except Exception:
        with open(path, 'rb') as f:
            return _NumpyCompat(f).load()


# ══════════════════════════════════════════════════════════════════════════════
#  2. Return building
# ══════════════════════════════════════════════════════════════════════════════

def mom_bar_returns(df: pd.DataFrame, cost: float) -> pd.Series:
    """
    Per-bar net return for a momentum OOS PKL.

    Exact replica of wf_visualizer._strat_ret — kept here so notebooks and
    combined_portfolio.py can import a single canonical implementation.

    df must have: Close, position.  position_size is optional (default 1.0).
    """
    pos  = df['position'].shift(1).fillna(0)
    size = (df['position_size'].shift(1).fillna(0)
            if 'position_size' in df.columns
            else pd.Series(1.0, index=df.index))
    ret  = df['Close'].pct_change().fillna(0)
    to   = df['position'].diff().abs().fillna(0)
    return ret * pos * size - cost * to


def sleeve_freq(df: pd.DataFrame) -> str:
    """Return 'hourly' or 'daily' based on the average bar density of df."""
    n_days = max((df.index[-1] - df.index[0]).days, 1)
    return 'hourly' if len(df) / n_days > 1.5 else 'daily'


def wrap_as_sleeve(bar_returns: pd.Series) -> pd.DataFrame:
    """
    Wrap a bar-return Series into the unified sleeve schema expected by
    plot_portfolio_oos: {Close = cumprod(1+r), position=1, position_size=1}.

    plot_portfolio_oos._strat_ret recovers bar returns via Close.pct_change(),
    so costs must already be baked into bar_returns before calling this.
    """
    eq = (1 + bar_returns).cumprod()
    return pd.DataFrame(
        {'Close': eq, 'position': 1, 'position_size': 1.0},
        index=bar_returns.index,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  3. Weighting
# ══════════════════════════════════════════════════════════════════════════════

def build_momentum_weights(
    mom_dfs: dict,
    mom_selection: dict,
    strat_weights: dict | None,
    coin_weights: dict | None,
) -> dict:
    """
    Build within-momentum sleeve weights (three-level: strategy × coin).

    mom_dfs       : {label: df}  resolved momentum sleeves
    mom_selection : {label: (tag, coin)}
    strat_weights : {tag: weight} or None for equal across strategies
    coin_weights  : {tag: {coin: weight}} or None for equal within each strategy

    Returns {label: weight} where weights sum to 1.0 across all sleeves.
    """
    tags = {mom_selection[s][0] for s in mom_dfs}
    msw  = norm_weights(strat_weights or {t: 1 for t in tags})
    out  = {}
    for tag in tags:
        sleeves = [s for s in mom_dfs if mom_selection[s][0] == tag]
        cw      = (coin_weights or {}).get(tag)
        if cw is None:
            cw_n = {mom_selection[s][1]: 1 / len(sleeves) for s in sleeves}
        else:
            cw_n = norm_weights({mom_selection[s][1]: cw.get(mom_selection[s][1], 0)
                                  for s in sleeves})
        for s in sleeves:
            out[s] = msw.get(tag, 0) * cw_n[mom_selection[s][1]]
    return out


def norm_weights(d: dict) -> dict:
    """Normalise a weight dict so values sum to 1.0."""
    s = sum(d.values())
    return {k: v / s for k, v in d.items()} if s > 0 else dict(d)


def sa_inverse_vol_weights(
    dfs: dict,
    method: str = 'in_market',
    window: int | None = None,
) -> dict:
    """
    Inverse-volatility weights for stat arb sleeves.

    method : 'in_market'  — vol computed only on bars where position != 0
             'full'       — vol computed over all bars
    window : None = full history | int = last N bars
    """
    def _vol(df):
        r, p = df['net_returns'].fillna(0), df['position'].fillna(0)
        if window:
            r, p = r.iloc[-window:], p.iloc[-window:]
        return r[p != 0].std() if method == 'in_market' else r.std()

    vols = {k: _vol(dfs[k]) for k in dfs}
    inv  = {k: 1 / v if v > 0 else 0.0 for k, v in vols.items()}
    return norm_weights(inv)


def build_sleeve_weights(
    sa_dfs: dict,
    sa_w: dict,
    mom_dfs: dict,
    mom_w: dict,
    strategy_weights: dict,
) -> dict:
    """
    Combine within-bucket weights with the top-level bucket split.

    Returns a flat dict {sleeve_label: final_weight} ready for plot_portfolio_oos.
    """
    sw = norm_weights(strategy_weights)
    out = {}
    for k in sa_dfs:
        out[k] = sw['statarb']  * sa_w[k]
    for s in mom_dfs:
        out[s] = sw['momentum'] * mom_w[s]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  4. Simulations
# ══════════════════════════════════════════════════════════════════════════════

def build_realized_equity(
    position_dfs: dict,
    weights: dict,
    cost: float = 0.001,
) -> 'pd.Series':
    """
    Realized-sizing equity curve.

    Entries are sized at  w_k × realized_equity  (not MTM equity).
    Realized equity only increments on trade close — open unrealized P&L from
    concurrent positions does not inflate future entry notionals.

    Algorithm (per bar):
      1. Earn bar returns for all positions open at the start of the bar.
      2. Exits  → realize P&L into realized_equity, reset cum_mult.
      3. Entries → entry_notional = weights[k] × realized_equity (post-exits).
      4. Portfolio equity = realized_equity + Σ(entry_notional × unrealized_ret).

    SA sleeves  → bar returns from 'net_returns' column (cost already baked).
    Mom sleeves → bar returns via mom_bar_returns(df, cost).

    Returns a pd.Series normalised to 1.0 at the first bar.
    """
    labels = list(position_dfs.keys())

    # per-sleeve bar returns (cost already baked for SA)
    bar_rets = {}
    for k, df in position_dfs.items():
        if 'net_returns' in df.columns:
            bar_rets[k] = df['net_returns'].fillna(0)
        else:
            bar_rets[k] = mom_bar_returns(df, cost)

    # union index across all sleeves
    all_idx = bar_rets[labels[0]].index
    for k in labels[1:]:
        all_idx = all_idx.union(bar_rets[k].index)

    # pre-convert to dicts for O(1) per-bar lookup (fast over 26k+ bars)
    bar_d = {k: bar_rets[k].reindex(all_idx).fillna(0).to_dict() for k in labels}
    pos_d = {k: position_dfs[k]['position'].fillna(0)
                .reindex(all_idx).ffill().fillna(0).to_dict()
             for k in labels}

    prev_pos        = {k: 0.0 for k in labels}
    cum_mult        = {k: 1.0 for k in labels}   # (1+bar_ret).cumprod since entry
    entry_notional  = {k: 0.0 for k in labels}
    realized_equity = 1.0
    eq_out          = []

    for t in all_idx:
        curr_pos = {k: pos_d[k][t] for k in labels}

        # 1 — earn bar returns for positions open at start of this bar
        for k in labels:
            if prev_pos[k] != 0:
                cum_mult[k] *= 1.0 + bar_d[k][t]

        # 2 — exits: realize P&L
        for k in labels:
            op, np_ = prev_pos[k], curr_pos[k]
            if op != 0 and (np_ == 0 or np_ != op):
                realized_equity  += entry_notional[k] * (cum_mult[k] - 1.0)
                entry_notional[k] = 0.0
                cum_mult[k]       = 1.0

        # 3 — entries: size against realized equity post-exits this bar
        for k in labels:
            op, np_ = prev_pos[k], curr_pos[k]
            if np_ != 0 and (op == 0 or np_ != op):
                entry_notional[k] = weights[k] * realized_equity
                cum_mult[k]       = 1.0

        # 4 — portfolio equity = realized + all open unrealized P&L
        unrealized = sum(entry_notional[k] * (cum_mult[k] - 1.0) for k in labels)
        eq_out.append(realized_equity + unrealized)
        prev_pos = curr_pos

    s = pd.Series(eq_out, index=all_idx)
    return s / s.iloc[0]


# ══════════════════════════════════════════════════════════════════════════════
#  5. Weight sweeps
# ══════════════════════════════════════════════════════════════════════════════

def sweep_top_level(
    sa_equity: pd.Series,
    mom_equity: pd.Series,
    step: int = 5,
    current_sa_weight: float | None = None,
) -> list[dict]:
    """
    Sweep the top-level stat-arb / momentum split from 0% to 100% in `step`% increments.

    sa_equity / mom_equity : bucket equity curves (from plot_portfolio_oos).

    Bar returns from the two buckets are combined at their native frequency via
    `concat + fillna(0)` — the exact alignment used by `plot_portfolio_oos` —
    and fed through `engine.backtest` so Sharpe, max-drawdown and (annualised)
    Calmar are produced by the same code path as the equity chart. This
    guarantees the sweep row at the configured split matches the chart numbers.

    Returns a list of dicts with keys:
      sa_pct, mom_pct, sharpe_ratio, total_return, max_drawdown, calmar_ratio, is_current
    """
    from engine import backtest

    sa_r = sa_equity.pct_change().fillna(0).rename('sa')
    mo_r = mom_equity.pct_change().fillna(0).rename('mo')
    aligned = pd.concat([sa_r, mo_r], axis=1).fillna(0)

    rows = []
    for sa_pct in range(0, 101, step):
        mo_pct = 100 - sa_pct
        comb   = aligned['sa'] * (sa_pct / 100) + aligned['mo'] * (mo_pct / 100)
        equity = (1 + comb).cumprod()
        port_df = pd.DataFrame(
            {'Close': equity, 'position': 1, 'position_size': 1.0},
            index=comb.index,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            m = backtest(port_df, cost=0.0, show_plot=False)
        rows.append({
            'sa_pct':        sa_pct,
            'mom_pct':       mo_pct,
            'sharpe_ratio':  m['sharpe_ratio'],
            'total_return':  m['total_return'],
            'max_drawdown':  m['max_drawdown'],
            'calmar_ratio':  m['calmar_ratio'],
            'is_current':    (current_sa_weight is not None
                              and abs(sa_pct / 100 - current_sa_weight) < 1e-9),
        })
    return rows


def sweep_momentum_strategy(
    mom_dfs: dict,
    mom_selection: dict,
    strat_weights_grid: list[dict],
    coin_weights: dict | None,
    cost: float,
) -> list[dict]:
    """
    Sweep momentum strategy-level weights (e.g. bb vs wf2 split) and return
    portfolio metrics for each combination.

    mom_dfs           : {label: df}  resolved momentum sleeves
    mom_selection     : {label: (tag, coin)}  maps labels to strategy/coin
    strat_weights_grid: list of dicts, each is a candidate MOM_STRAT_WEIGHTS
                        e.g. [{'bb': 0.0, 'wf2': 1.0}, {'bb': 0.1, 'wf2': 0.9}, ...]
    coin_weights      : {tag: {coin: weight}} or None for equal within each strategy
    cost              : per-leg trading cost for momentum

    Returns list of dicts with keys:
      strat_weights, sleeve_weights, sharpe_ratio, total_return, max_drawdown, calmar_ratio
    """
    from engine import backtest

    def _build_mom_w(msw_raw):
        msw    = norm_weights(msw_raw)
        tags   = {mom_selection[s][0] for s in mom_dfs}
        result = {}
        for tag in tags:
            sleeves = [s for s in mom_dfs if mom_selection[s][0] == tag]
            cw      = (coin_weights or {}).get(tag)
            if cw is None:
                cw_n = {mom_selection[s][1]: 1 / len(sleeves) for s in sleeves}
            else:
                cw_n = norm_weights({mom_selection[s][1]: cw.get(mom_selection[s][1], 0)
                                     for s in sleeves})
            for s in sleeves:
                result[s] = msw.get(tag, 0) * cw_n[mom_selection[s][1]]
        return result

    rows = []
    for msw_raw in strat_weights_grid:
        mw = _build_mom_w(msw_raw)
        # Use pd.concat + fillna(0) to align mixed hourly/daily series — same
        # approach as plot_portfolio_oos so sweep numbers are directly comparable.
        aligned = pd.concat(
            [mom_bar_returns(mom_dfs[s], cost).rename(s) for s in mom_dfs],
            axis=1,
        ).fillna(0)
        all_ret = sum(aligned[s] * mw[s] for s in mom_dfs)
        df_bt = pd.DataFrame(
            {'strategy_returns': all_ret, 'position': 1},
            index=all_ret.index,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            m = backtest(df_bt, cost=0.0, show_plot=False)
        rows.append({
            'strat_weights': msw_raw,
            'sleeve_weights': mw,
            'sharpe_ratio':  m['sharpe_ratio'],
            'total_return':  m['total_return'],
            'max_drawdown':  m['max_drawdown'],
            'calmar_ratio':  m['calmar_ratio'],
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  6. Print helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_weight_audit(
    sa_dfs: dict,
    sa_w: dict,
    mom_dfs: dict,
    mom_w: dict,
    sw: dict,
    sleeve_weights: dict,
    statarb_method: str = 'inverse_vol',
) -> None:
    """Print the three-level weight breakdown (buckets → SA sleeves → momentum sleeves)."""
    W = 70
    print('═' * W)
    print('  Bucket split')
    print('─' * W)
    for k, v in sw.items():
        print(f'  {k:<12}  {v*100:>6.2f}%')

    print('\n' + '─' * W)
    print(f'  Stat arb sleeves  (method={statarb_method})')
    print('─' * W)
    print(f'  {"Pair":<12}  {"within":>8}   {"bucket":>8}   {"final":>8}')
    for k in sa_dfs:
        print(f'  {k.replace("_","/"):<12}  {sa_w[k]*100:>7.2f}%'
              f'   {sw["statarb"]*100:>7.2f}%   {sleeve_weights[k]*100:>7.2f}%')

    print('\n' + '─' * W)
    print('  Momentum sleeves')
    print('─' * W)
    print(f'  {"Sleeve":<12}  {"within":>8}   {"bucket":>8}   {"final":>8}')
    for s in mom_dfs:
        print(f'  {s:<12}  {mom_w[s]*100:>7.2f}%'
              f'   {sw["momentum"]*100:>7.2f}%   {sleeve_weights[s]*100:>7.2f}%')
    print('═' * W)


def print_bucket_comparison(m_sa, m_mom, m_comb) -> None:
    """Print StatArb / Momentum / Combined metrics side by side."""
    def _f(m, key, fmt, scale=1, sfx=''):
        return f'{m[key] * scale:{fmt}}{sfx}' if m is not None else '—'

    W = 62
    print('═' * W)
    print(f'  {"":20}  {"StatArb":>10}  {"Momentum":>10}  {"Combined":>10}')
    print('─' * W)
    for lbl, key, fmt, scale, sfx in [
        ('Total Return', 'total_return', '.1f', 100, '%'),
        ('Sharpe Ratio', 'sharpe_ratio', '.2f', 1,   '' ),
        ('Max Drawdown', 'max_drawdown', '.1f', 100, '%'),
        ('Calmar Ratio', 'calmar_ratio', '.2f', 1,   '' ),
    ]:
        print(f'  {lbl:<20}  {_f(m_sa,   key, fmt, scale, sfx):>10}'
              f'  {_f(m_mom,  key, fmt, scale, sfx):>10}'
              f'  {_f(m_comb, key, fmt, scale, sfx):>10}')
    print('═' * W)


def print_weight_sweep(sweep_results: list) -> None:
    """Print the top-level SA / Momentum split sensitivity table."""
    W = 62
    print('═' * W)
    print('  Weight sensitivity — top-level split')
    print('─' * W)
    print(f'  {"SA%":>5}  {"MOM%":>5}  {"Sharpe":>7}  {"Return":>8}  {"MaxDD":>7}  {"Calmar":>7}')
    print(f'  {"-"*5}  {"-"*5}  {"-"*7}  {"-"*8}  {"-"*7}  {"-"*7}')
    for r in sweep_results:
        marker = '  <-- current' if r.get('is_current') else ''
        print(f'  {r["sa_pct"]:>5}  {r["mom_pct"]:>5}  {r["sharpe_ratio"]:>7.2f}  '
              f'{r["total_return"]*100:>7.1f}%  {r["max_drawdown"]*100:>7.1f}%  '
              f'{r["calmar_ratio"]:>7.2f}{marker}')
    print('═' * W)


def print_momentum_sweep(sweep_results: list, current_bb_weight: float | None = None) -> None:
    """Print the bb vs wf2 strategy-split sensitivity table."""
    best_sh = max(sweep_results, key=lambda r: r['sharpe_ratio'])
    best_ca = max(sweep_results, key=lambda r: r['calmar_ratio'])

    W = 72
    print('═' * W)
    print('  Momentum strategy sweep  (bb vs wf2, coin weights fixed)')
    print('─' * W)
    print(f'  {"bb%":>5}  {"wf2%":>5}  {"Sharpe":>7}  {"Return":>9}  {"MaxDD":>7}  {"Calmar":>7}')
    print(f'  {"-"*5}  {"-"*5}  {"-"*7}  {"-"*9}  {"-"*7}  {"-"*7}')
    for r in sweep_results:
        bb_pct  = round(r['strat_weights'].get('bb',  0) * 100)
        wf2_pct = round(r['strat_weights'].get('wf2', 0) * 100)
        tags = []
        if current_bb_weight is not None and abs(r['strat_weights'].get('bb', 0) - current_bb_weight) < 1e-9:
            tags.append('current')
        if r is best_sh:
            tags.append('best Sharpe')
        if r is best_ca and r is not best_sh:
            tags.append('best Calmar')
        suffix = f'  <-- {", ".join(tags)}' if tags else ''
        print(f'  {bb_pct:>5}  {wf2_pct:>5}  {r["sharpe_ratio"]:>7.2f}  '
              f'{r["total_return"]*100:>8.1f}%  {r["max_drawdown"]*100:>7.1f}%  '
              f'{r["calmar_ratio"]:>7.2f}{suffix}')
    print('═' * W)


def print_per_coin_stats(mom_dfs: dict, mom_selection: dict, cost: float) -> None:
    """Print per-momentum-sleeve isolation metrics, sorted by Sharpe."""
    from engine import backtest

    rows = []
    for s, df in mom_dfs.items():
        tag, _ = mom_selection[s]
        bar_r  = mom_bar_returns(df, cost)
        df_bt  = pd.DataFrame({'strategy_returns': bar_r, 'position': 1}, index=bar_r.index)
        with contextlib.redirect_stdout(io.StringIO()):
            m = backtest(df_bt, cost=0.0, show_plot=False)
        rows.append({'sleeve': s, 'tag': tag, 'sharpe': m['sharpe_ratio'],
                     'ret': m['total_return'], 'dd': m['max_drawdown'],
                     'calmar': m['calmar_ratio']})

    rows.sort(key=lambda r: r['sharpe'], reverse=True)

    W = 68
    print('═' * W)
    print('  Per-coin analysis  (each sleeve run in isolation, sorted by Sharpe)')
    print('─' * W)
    print(f'  {"Sleeve":<12}  {"Strategy":>8}  {"Sharpe":>7}  {"Return":>9}  {"MaxDD":>8}  {"Calmar":>7}')
    print(f'  {"-"*12}  {"-"*8}  {"-"*7}  {"-"*9}  {"-"*8}  {"-"*7}')
    for r in rows:
        print(f'  {r["sleeve"]:<12}  {r["tag"]:>8}  {r["sharpe"]:>7.2f}  '
              f'{r["ret"]*100:>8.1f}%  {r["dd"]*100:>8.1f}%  {r["calmar"]:>7.2f}')
    print('═' * W)
