"""
build_dataset.py
================
Compute feature matrix and labels for the ML prediction project.

Joins all raw parquet sources for each symbol, computes ~40 features,
and applies a deadband label:
  1  if next-day log return >  DEADBAND
  0  if next-day log return < -DEADBAND
  NaN otherwise (sample discarded at training time)

Public API
----------
  build_features(symbol)  -> pd.DataFrame   features + label for one symbol
  build_all()             -> pd.DataFrame   all symbols stacked, with 'symbol' column
  save_all()              -> None           writes features/{SYMBOL}_features.parquet

Usage:
  python build_dataset.py                   # build and save all symbols
  python build_dataset.py --symbol BTCUSDT  # single symbol
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parents[3]
_OHLCV_DIR    = _ROOT / 'live_trading' / 'cache' / 'daily'
_ML_RAW       = _ROOT / 'topics' / 'ml-prediction' / 'data' / 'raw'
_FEATURES_DIR = _ROOT / 'topics' / 'ml-prediction' / 'data' / 'features'

UNIVERSE      = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT',
                 'XRPUSDT', 'DOGEUSDT', 'LINKUSDT', 'AVAXUSDT']
DERIBIT_COINS = ['BTC', 'ETH']

DEADBAND = 0.003   # 30 bps — samples with |return| < this are dropped


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def _close(symbol: str) -> pd.Series:
    df = _load(_OHLCV_DIR / f'{symbol}_daily.parquet')
    if df is None:
        raise FileNotFoundError(f'No OHLCV cache for {symbol}. Run backfill_cache.py first.')
    s = df['Close'].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index()


# ── Feature groups ────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def price_features(close: pd.Series) -> pd.DataFrame:
    log_ret = np.log(close / close.shift(1))

    f = pd.DataFrame(index=close.index)
    for d in [1, 3, 7, 14, 30]:
        f[f'ret_{d}d'] = np.log(close / close.shift(d))
    for w in [7, 14, 30]:
        f[f'rvol_{w}d'] = log_ret.rolling(w).std()
    f['zscore_ma20'] = (close - close.rolling(20).mean()) / close.rolling(20).std()
    f['zscore_ma60'] = (close - close.rolling(60).mean()) / close.rolling(60).std()
    f['rsi_14']      = _rsi(close)
    f['skew_30d']    = log_ret.rolling(30).skew()
    f['kurt_30d']    = log_ret.rolling(30).kurt()
    return f


def funding_features(symbol: str) -> pd.DataFrame:
    df = _load(_ML_RAW / 'funding' / f'{symbol}_funding.parquet')
    if df is None:
        return pd.DataFrame()

    s = df['funding_sum_1d']
    f = pd.DataFrame(index=s.index)
    f['funding_1d']      = s
    f['funding_avg_7d']  = s.rolling(7).mean()
    rolling_mean         = s.rolling(30).mean()
    rolling_std          = s.rolling(30).std()
    f['funding_zscore']  = (s - rolling_mean) / rolling_std.replace(0, np.nan)
    f['funding_extreme'] = (s.abs() > 0.0015).astype(float)  # |daily sum| > 0.15%
    return f


def oi_features(symbol: str) -> pd.DataFrame:
    df = _load(_ML_RAW / 'oi' / f'{symbol}_oi.parquet')
    if df is None:
        return pd.DataFrame()

    s = df['oi_usd']
    f = pd.DataFrame(index=s.index)
    f['oi_chg_1d'] = s.pct_change(1)
    f['oi_chg_7d'] = s.pct_change(7)
    return f


def ls_features(symbol: str) -> pd.DataFrame:
    df = _load(_ML_RAW / 'longshort' / f'{symbol}_ls.parquet')
    if df is None:
        return pd.DataFrame()

    s = df['ls_ratio']
    f = pd.DataFrame(index=s.index)
    f['ls_ratio']   = s
    f['ls_chg_7d']  = s.diff(7)
    return f


def dvol_features(symbol: str) -> pd.DataFrame:
    coin = symbol.replace('USDT', '')
    if coin not in DERIBIT_COINS:
        return pd.DataFrame()

    df = _load(_ML_RAW / 'deribit' / f'{coin}_dvol.parquet')
    if df is None:
        return pd.DataFrame()

    s = df['dvol']
    f = pd.DataFrame(index=s.index)
    f['dvol']        = s
    f['dvol_chg_7d'] = s.diff(7)
    return f


def cross_asset_features(symbol: str, close: pd.Series) -> pd.DataFrame:
    log_ret = np.log(close / close.shift(1))
    f = pd.DataFrame(index=close.index)

    if symbol != 'BTCUSDT':
        btc_close   = _close('BTCUSDT').reindex(close.index)
        btc_ret     = np.log(btc_close / btc_close.shift(1))
        f['btc_corr_30d'] = log_ret.rolling(30).corr(btc_ret)
        cov = log_ret.rolling(30).cov(btc_ret)
        var = btc_ret.rolling(30).var()
        f['btc_beta_30d'] = cov / var.replace(0, np.nan)
    else:
        f['btc_corr_30d'] = np.nan
        f['btc_beta_30d'] = np.nan

    return f


def macro_features() -> pd.DataFrame:
    df = _load(_ML_RAW / 'macro' / 'macro_daily.parquet')
    if df is None:
        return pd.DataFrame()

    f = pd.DataFrame(index=df.index)
    for col, name in [('dxy_close', 'dxy'), ('spx_close', 'spx'), ('gold_close', 'gold')]:
        s = df[col]
        f[f'{name}_ret_1d'] = np.log(s / s.shift(1))
        f[f'{name}_ret_7d'] = np.log(s / s.shift(7))

    # Lag by 1 day — US markets close after crypto daily bar
    f = f.shift(1)
    return f


def onchain_features() -> pd.DataFrame:
    df = _load(_ML_RAW / 'onchain' / 'onchain_daily.parquet')
    if df is None:
        return pd.DataFrame()

    f = pd.DataFrame(index=df.index)
    f['tvl_chg_7d']        = df['defi_tvl_usd'].pct_change(7)
    f['stablecoin_chg_7d'] = df['stablecoin_mcap_usd'].pct_change(7)
    return f


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=index)
    f['day_of_week'] = index.dayofweek
    f['is_weekend']  = (index.dayofweek >= 5).astype(float)
    return f


# ── Label ─────────────────────────────────────────────────────────────────────

def make_label(close: pd.Series, deadband: float = DEADBAND) -> pd.Series:
    """Next-day log return, clipped to {0, 1, NaN} with deadband filter."""
    next_ret = np.log(close.shift(-1) / close)
    label    = pd.Series(np.nan, index=close.index, name='label')
    label[next_ret >  deadband] = 1.0
    label[next_ret < -deadband] = 0.0
    return label


# ── Main builder ──────────────────────────────────────────────────────────────

def build_features(symbol: str) -> pd.DataFrame:
    """Build full feature matrix + label for one symbol."""
    close = _close(symbol)

    frames = [
        price_features(close),
        funding_features(symbol),
        oi_features(symbol),
        ls_features(symbol),
        dvol_features(symbol),
        cross_asset_features(symbol, close),
        macro_features(),
        onchain_features(),
        calendar_features(close.index),
    ]

    # Align everything on close's index
    feat = pd.concat(
        [f.reindex(close.index) for f in frames if not f.empty],
        axis=1,
    )

    feat['next_ret'] = np.log(close.shift(-1) / close)
    feat['label']    = make_label(close)
    feat['symbol']   = symbol

    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat.sort_index()


def build_all() -> pd.DataFrame:
    frames = []
    for sym in UNIVERSE:
        print(f'  Building {sym}...')
        try:
            frames.append(build_features(sym))
        except Exception as e:
            print(f'  WARNING: {sym} failed — {e}')
    return pd.concat(frames).sort_index()


def save_all() -> None:
    _FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for sym in UNIVERSE:
        print(f'  {sym}...', end=' ', flush=True)
        try:
            df = build_features(sym)
            path = _FEATURES_DIR / f'{sym}_features.parquet'
            df.to_parquet(path, engine='pyarrow')
            valid = df['label'].notna().sum()
            total = len(df)
            print(f'{total} rows, {valid} labelled ({100*valid/total:.0f}% survive deadband)')
        except Exception as e:
            print(f'ERROR — {e}')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default=None)
    args = parser.parse_args()

    _FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.symbol:
        sym  = args.symbol
        df   = build_features(sym)
        path = _FEATURES_DIR / f'{sym}_features.parquet'
        df.to_parquet(path, engine='pyarrow')
        valid = df['label'].notna().sum()
        print(f'{sym}: {len(df)} rows, {valid} labelled, {len(df.columns)-3} features')
    else:
        print('Building feature matrix for all symbols...')
        save_all()
        print('\nDone.')
