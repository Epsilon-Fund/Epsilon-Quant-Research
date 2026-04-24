"""
portfolio_metrics.py
====================
Shared helpers for combined-portfolio analysis notebooks.

Covers the three cross-cutting concerns that appear in both
epsilon_portfolio.ipynb and combined_portfolio.py:

  1. PKL loading      — numpy-compat unpickler
  2. Return building  — unified sleeve schema, momentum bar returns
  3. Weighting        — normalisation, stat-arb inverse-vol, momentum 3-level

Public API
----------
  load_pkl(path)                               → pd.DataFrame
  mom_bar_returns(df, cost)                    → pd.Series
  wrap_as_sleeve(bar_returns)                  → pd.DataFrame
  norm_weights(d)                              → dict
  sa_inverse_vol_weights(dfs, method, window)  → dict
  build_sleeve_weights(sa_dfs, sa_w,
                       mom_dfs, mom_w,
                       strategy_weights)       → dict
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
#  4. Weight sweeps
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

    Returns a list of dicts with keys:
      sa_pct, mom_pct, sharpe_ratio, total_return, max_drawdown, calmar_ratio, is_current
    """
    from engine import backtest

    all_idx = sa_equity.index.union(mom_equity.index)
    sa_r    = sa_equity.pct_change().fillna(0).reindex(all_idx).fillna(0)
    mo_r    = mom_equity.pct_change().fillna(0).reindex(all_idx).fillna(0)

    rows = []
    for sa_pct in range(0, 101, step):
        mo_pct  = 100 - sa_pct
        comb    = sa_r * (sa_pct / 100) + mo_r * (mo_pct / 100)
        df_bt   = pd.DataFrame({'strategy_returns': comb, 'position': 1}, index=comb.index)
        with contextlib.redirect_stdout(io.StringIO()):
            m = backtest(df_bt, cost=0.0, show_plot=False)
        calmar = (m['total_return'] / abs(m['max_drawdown'])
                  if m['max_drawdown'] != 0 else 0.0)
        rows.append({
            'sa_pct':        sa_pct,
            'mom_pct':       mo_pct,
            'sharpe_ratio':  m['sharpe_ratio'],
            'total_return':  m['total_return'],
            'max_drawdown':  m['max_drawdown'],
            'calmar_ratio':  calmar,
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
        mw      = _build_mom_w(msw_raw)
        all_ret = sum(
            mom_bar_returns(mom_dfs[s], cost) * mw[s]
            for s in mom_dfs
        )
        df_bt = pd.DataFrame(
            {'strategy_returns': all_ret, 'position': 1},
            index=all_ret.index,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            m = backtest(df_bt, cost=0.0, show_plot=False)
        calmar = (m['total_return'] / abs(m['max_drawdown'])
                  if m['max_drawdown'] != 0 else 0.0)
        rows.append({
            'strat_weights': msw_raw,
            'sleeve_weights': mw,
            'sharpe_ratio':  m['sharpe_ratio'],
            'total_return':  m['total_return'],
            'max_drawdown':  m['max_drawdown'],
            'calmar_ratio':  calmar,
        })
    return rows
