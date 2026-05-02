"""
walk_forward.py
===============
Rolling walk-forward engine for the ML prediction model.

Each fold:
  1. RobustScaler fitted on train features only
  2. XGBoost trained on scaled train set
  3. IsotonicRegression calibrator fitted on val set
  4. Optimal confidence threshold tau found on val set (max Sharpe net of fees)
  5. Test set scored with calibrated probabilities

Default window config:
  train : 1095 days (3 years)
  val   :   90 days (3 months) — threshold selection
  test  :   30 days (1 month)  — OOS predictions
  step  :   30 days            — roll forward each month

Public API
----------
  run_walk_forward(df, config)  -> (predictions_df, folds_meta)

  df      : pooled feature DataFrame from build_dataset.build_all()
            must have columns: label, next_ret, symbol, + feature cols
  config  : dict with keys train_days, val_days, test_days, step_days,
            cost_bps, tau_grid (optional)

Returns
-------
  predictions_df : DataFrame with columns:
      date, symbol, prob, pred, label, next_ret, fold, tau, confidence
  folds_meta     : list of dicts — one per fold with train/val/test dates,
                   val Sharpe per tau, chosen tau, n_features used
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    'train_days': 1095,   # 3 years
    'val_days':     90,   # 3 months
    'test_days':    30,   # 1 month
    'step_days':    30,   # roll 1 month each fold
    'cost_bps':      6,   # round-trip transaction cost
    'tau_grid': [0.80],
    'xgb_params': {
        'n_estimators':    300,
        'max_depth':         4,
        'learning_rate':  0.03,
        'subsample':       0.8,
        'colsample_bytree':0.8,
        'min_child_weight':  5,
        'eval_metric':  'logloss',
        'verbosity':         0,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sharpe(returns: pd.Series, cost_bps: float) -> float:
    """Annualised Sharpe of a daily return series after transaction costs."""
    if len(returns) < 5:
        return -np.inf
    net = returns - cost_bps / 10_000
    if net.std() == 0:
        return 0.0
    return float(net.mean() / net.std() * np.sqrt(252))


def _signals_to_returns(probs: np.ndarray, labels: np.ndarray,
                         next_rets: np.ndarray, tau: float) -> pd.Series:
    """
    Convert calibrated probabilities to a daily P&L series.
    Long when prob > tau, short when prob < (1 - tau), flat otherwise.
    """
    pos = np.where(probs > tau, 1.0,
          np.where(probs < 1 - tau, -1.0, 0.0))
    # next_rets is log return of the coin; position is +1/-1
    pnl = pos * next_rets
    return pd.Series(pnl)


def _best_tau(probs: np.ndarray, labels: np.ndarray,
              next_rets: np.ndarray, tau_grid: list[float],
              cost_bps: float) -> tuple[float, dict]:
    """Find tau that maximises Sharpe on the validation set."""
    results = {}
    for tau in tau_grid:
        rets = _signals_to_returns(probs, labels, next_rets, tau)
        active = (np.abs(np.where(probs > tau, 1, np.where(probs < 1-tau, -1, 0))) > 0)
        n_trades = int(active.sum())
        if n_trades < 5:
            results[tau] = {'sharpe': -np.inf, 'n_trades': n_trades}
            continue
        results[tau] = {
            'sharpe':   _sharpe(rets[active], cost_bps),
            'n_trades': n_trades,
        }
    best_tau = max(results, key=lambda t: results[t]['sharpe'])
    return best_tau, results


# ── Core fold ─────────────────────────────────────────────────────────────────

def _run_fold(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    feat_cols: list[str],
    config:    dict,
    fold_idx:  int,
) -> tuple[pd.DataFrame, dict]:
    """Train, calibrate, threshold-select, and score one fold."""

    cost_bps = config['cost_bps']
    tau_grid = config['tau_grid']

    # ── Prepare arrays ────────────────────────────────────────────────────────
    def _xy(df):
        X = df[feat_cols].copy()
        y = df['label'].values
        r = df['next_ret'].values
        return X, y, r

    X_tr, y_tr, _      = _xy(train_df)
    X_va, y_va, r_va   = _xy(val_df)
    X_te, y_te, r_te   = _xy(test_df)

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr.fillna(X_tr.median()))
    X_va_s = scaler.transform(X_va.fillna(X_va.median()))
    X_te_s = scaler.transform(X_te.fillna(X_te.median()))

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    model = XGBClassifier(**config['xgb_params'])
    model.fit(X_tr_s, y_tr)

    # ── Calibrate on val ──────────────────────────────────────────────────────
    raw_probs_val = model.predict_proba(X_va_s)[:, 1]
    calibrator    = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probs_val, y_va)

    cal_probs_val = calibrator.transform(raw_probs_val)

    # ── Find best tau on val ──────────────────────────────────────────────────
    best_tau, tau_results = _best_tau(
        cal_probs_val, y_va, r_va, tau_grid, cost_bps
    )

    # ── Score test set ────────────────────────────────────────────────────────
    raw_probs_te = model.predict_proba(X_te_s)[:, 1]
    cal_probs_te = calibrator.transform(raw_probs_te)

    preds = pd.DataFrame({
        'symbol':     test_df['symbol'].values,
        'prob':       cal_probs_te,
        'label':      y_te,
        'next_ret':   r_te,
        'fold':       fold_idx,
        'tau':        best_tau,
        'confidence': np.maximum(cal_probs_te, 1 - cal_probs_te),
        'pred':       np.where(cal_probs_te > best_tau, 1,
                      np.where(cal_probs_te < 1 - best_tau, 0, np.nan)),
    }, index=test_df.index)

    # ── Feature importance ────────────────────────────────────────────────────
    importance = dict(zip(feat_cols, model.feature_importances_))

    meta = {
        'fold':        fold_idx,
        'train_start': train_df.index[0].date(),
        'train_end':   train_df.index[-1].date(),
        'val_start':   val_df.index[0].date(),
        'val_end':     val_df.index[-1].date(),
        'test_start':  test_df.index[0].date(),
        'test_end':    test_df.index[-1].date(),
        'best_tau':    best_tau,
        'tau_results': tau_results,
        'n_train':     len(train_df),
        'n_val':       len(val_df),
        'n_test':      len(test_df),
        'importance':  importance,
        'model':       model,       # kept for SHAP analysis
        'scaler':      scaler,      # kept for SHAP analysis
        'calibrator':  calibrator,  # kept for SHAP analysis
        'feat_cols':   feat_cols,   # kept for SHAP analysis
    }

    return preds, meta


# ── Main entry ────────────────────────────────────────────────────────────────

def run_walk_forward(
    df:     pd.DataFrame,
    config: dict | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run full walk-forward over df.

    df must be indexed by date, contain columns: label, next_ret, symbol,
    and all feature columns. Only labelled rows (label not NaN) are used.

    Returns (predictions_df, folds_meta).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Labelled rows only, sorted by date
    data = df[df['label'].notna()].sort_index()

    feat_cols = [c for c in data.columns
                 if c not in ('label', 'next_ret', 'symbol')]

    dates  = data.index.unique().sort_values()
    n      = len(dates)
    tr_d   = cfg['train_days']
    va_d   = cfg['val_days']
    te_d   = cfg['test_days']
    step_d = cfg['step_days']

    min_start_idx = tr_d + va_d
    if n < min_start_idx + te_d:
        raise ValueError(
            f'Not enough data: need {min_start_idx + te_d} labelled days, have {n}.'
        )

    all_preds = []
    all_meta  = []
    fold_idx  = 0

    # Walk from first valid test window to end of data
    start = min_start_idx
    while start + te_d <= n:
        tr_dates  = dates[start - tr_d - va_d : start - va_d]
        va_dates  = dates[start - va_d         : start       ]
        te_dates  = dates[start                : start + te_d]

        train_df = data.loc[data.index.isin(tr_dates)]
        val_df   = data.loc[data.index.isin(va_dates)]
        test_df  = data.loc[data.index.isin(te_dates)]

        if len(train_df) < 200 or len(val_df) < 20 or len(test_df) < 5:
            start += step_d
            continue

        print(f'  Fold {fold_idx:02d}  '
              f'train {tr_dates[0].date()}–{tr_dates[-1].date()}  '
              f'test {te_dates[0].date()}–{te_dates[-1].date()}',
              end='  ')

        preds, meta = _run_fold(
            train_df, val_df, test_df, feat_cols, cfg, fold_idx
        )
        all_preds.append(preds)
        all_meta.append(meta)

        n_signals = int((preds['pred'].notna()).sum())
        print(f'τ={meta["best_tau"]:.2f}  signals={n_signals}')

        fold_idx += 1
        start    += step_d

    if not all_preds:
        raise RuntimeError('No folds completed — check data length and config.')

    predictions_df = pd.concat(all_preds).sort_index()
    print(f'\nDone. {fold_idx} folds, {len(predictions_df)} test rows, '
          f'{predictions_df["pred"].notna().sum():.0f} signals.')

    return predictions_df, all_meta
