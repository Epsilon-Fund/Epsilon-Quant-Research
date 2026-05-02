"""
ml_metrics.py
=============
Performance evaluation for the ML prediction walk-forward output.

Takes the predictions DataFrame from walk_forward.run_walk_forward() and
computes all metrics needed for the research pitch.

Public API
----------
  sharpe(returns, freq)                -> float
  max_drawdown(equity)                 -> float
  hit_rate(preds_df, tau)              -> float
  equity_curve(preds_df, cost_bps)     -> pd.Series
  sharpe_by_year(preds_df, cost_bps)   -> pd.DataFrame
  hit_rate_by_bucket(preds_df)         -> pd.DataFrame
  calibration_table(preds_df, n_bins)  -> pd.DataFrame
  feature_importance(folds_meta)       -> pd.DataFrame
  summary(preds_df, folds_meta,
          cost_bps)                    -> dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Building blocks ───────────────────────────────────────────────────────────

def sharpe(returns: pd.Series, freq: int = 252) -> float:
    """Annualised Sharpe ratio."""
    if returns.std() == 0 or len(returns) < 5:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(freq))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max.replace(0, np.nan)
    return float(dd.min())


# ── Signal extraction ─────────────────────────────────────────────────────────

def _active_returns(preds_df: pd.DataFrame, cost_bps: float) -> pd.Series:
    """
    Daily P&L for all active signals (pred = 1 → long, pred = 0 → short).
    Flat rows (pred = NaN) are excluded.
    """
    active = preds_df[preds_df['pred'].notna()].copy()
    direction = np.where(active['pred'] == 1, 1.0, -1.0)
    gross     = direction * active['next_ret']
    net       = gross - cost_bps / 10_000
    return pd.Series(net, index=active.index)


def equity_curve(preds_df: pd.DataFrame, cost_bps: float = 6) -> pd.Series:
    """
    Cumulative equity curve (starts at 1.0) from active signals.
    Aggregates across all symbols by summing daily P&L then cumsum.
    """
    rets = _active_returns(preds_df, cost_bps)
    daily = rets.groupby(level=0).mean()   # average across same-day signals
    return (1 + daily).cumprod()


# ── Core metrics ──────────────────────────────────────────────────────────────

def hit_rate(preds_df: pd.DataFrame, tau: float | None = None) -> float:
    """Fraction of signals where pred matches label."""
    df = preds_df[preds_df['pred'].notna()].copy()
    if tau is not None:
        df = df[df['confidence'] >= tau]
    if df.empty:
        return np.nan
    return float((df['pred'] == df['label']).mean())


def sharpe_by_year(preds_df: pd.DataFrame, cost_bps: float = 6) -> pd.DataFrame:
    """
    Annualised Sharpe per calendar year.
    The key slide for a fund pitch — shows consistency across regimes.
    """
    rets       = _active_returns(preds_df, cost_bps)
    daily      = rets.groupby(level=0).mean()
    active_idx = pd.DatetimeIndex(preds_df[preds_df['pred'].notna()].index)
    rows = []
    for year, grp in daily.groupby(daily.index.year):
        # Reindex to every calendar day in the year so flat days count as 0
        full_idx  = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
        grp_full  = grp.reindex(full_idx).fillna(0)
        n_signals = int((active_idx.year == year).sum())
        rows.append({
            'year':      year,
            'sharpe':    round(sharpe(grp_full, freq=365), 2),
            'ret_pct':   round(grp.sum() * 100, 1),
            'n_signals': n_signals,
            'n_days':    len(grp),
        })
    return pd.DataFrame(rows).set_index('year')


def hit_rate_by_bucket(preds_df: pd.DataFrame,
                        buckets: list = None) -> pd.DataFrame:
    """
    Hit rate broken down by confidence bucket.
    Validates calibration: higher confidence should → higher hit rate.
    """
    if buckets is None:
        buckets = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01]

    df   = preds_df[preds_df['pred'].notna()].copy()
    rows = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        sub = df[(df['confidence'] >= lo) & (df['confidence'] < hi)]
        if sub.empty:
            continue
        rows.append({
            'confidence_bucket': f'{lo:.0%}–{hi:.0%}',
            'n_signals':         len(sub),
            'hit_rate':          round((sub['pred'] == sub['label']).mean(), 3),
            'avg_confidence':    round(sub['confidence'].mean(), 3),
        })
    return pd.DataFrame(rows)


def calibration_table(preds_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Calibration check: predicted probability vs actual frequency of label=1.
    A well-calibrated model's points should lie on the diagonal.
    """
    df = preds_df[preds_df['pred'].notna()].copy()
    df['bin'] = pd.cut(df['prob'], bins=n_bins)
    rows = []
    for bin_label, grp in df.groupby('bin', observed=True):
        rows.append({
            'prob_bucket':   str(bin_label),
            'mean_pred_prob': round(grp['prob'].mean(), 3),
            'actual_freq':   round(grp['label'].mean(), 3),
            'n':             len(grp),
        })
    return pd.DataFrame(rows)


def feature_importance(folds_meta: list[dict],
                        top_n: int = 20) -> pd.DataFrame:
    """
    Average feature importance across all folds.
    Shows which signals actually drove predictions.
    """
    all_imp: dict[str, list] = {}
    for fold in folds_meta:
        for feat, imp in fold['importance'].items():
            all_imp.setdefault(feat, []).append(imp)

    rows = [{'feature': f, 'importance': np.mean(v)}
            for f, v in all_imp.items()]
    df = pd.DataFrame(rows).sort_values('importance', ascending=False)
    return df.head(top_n).reset_index(drop=True)


# ── Summary ───────────────────────────────────────────────────────────────────

def summary(preds_df: pd.DataFrame,
            folds_meta: list[dict],
            cost_bps: float = 6) -> dict:
    """
    One-dict summary of all key metrics — the pitch scorecard.
    """
    rets   = _active_returns(preds_df, cost_bps)
    daily  = rets.groupby(level=0).mean()
    eq     = (1 + daily).cumprod()

    # Honest Sharpe: reindex to every calendar day, flat days earn 0
    full_idx   = pd.date_range(eq.index[0], eq.index[-1], freq='D')
    daily_full = daily.reindex(full_idx).fillna(0)

    years   = (eq.index[-1] - eq.index[0]).days / 365.25
    ann_ret = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    mdd     = max_drawdown(eq)

    active = preds_df[preds_df['pred'].notna()]

    return {
        'n_folds':         len(folds_meta),
        'n_signals':       int(active['pred'].notna().sum()),
        'hit_rate':        round(hit_rate(preds_df), 3),
        'sharpe_net':      round(sharpe(daily_full, freq=365), 2),
        'calmar':          round(ann_ret / abs(mdd), 2) if mdd != 0 else np.nan,
        'max_drawdown':    round(mdd, 3),
        'ann_return':      round(ann_ret, 3),
        'total_return':    round(float(eq.iloc[-1] - 1), 3) if len(eq) else np.nan,
        'avg_tau':         round(np.mean([f['best_tau'] for f in folds_meta]), 3),
        'cost_bps':        cost_bps,
    }
