"""
Stat arb pairs strategy: rolling OLS spread → z-score → mean-reversion signals.

Exported: STRATEGY_REGISTRY
"""

import numpy as np
import pandas as pd


def stat_arb_spread(df_slice: pd.DataFrame, params: dict):
    """
    Pairs mean-reversion strategy.

    df_slice must contain Close_Y and Close_X (inner-joined on datetime).

    Signal loop (exactly mirrors the research notebooks):
      - Enter when |z| > entry (short spread if z > +entry, long if z < -entry)
      - Exit when |z| < exit_z  OR  |z| > stop_z  OR  bars_held >= max_holding

    Returns (result_df, indicator_cols) to match the momentum strategy signature.
    """
    df = df_slice.copy()

    lookback    = int(params['lookback'])
    z_lookback  = int(params['z_lookback'])
    entry       = float(params['entry'])
    exit_z      = float(params['exit_z'])
    stop_z      = float(params['stop_z'])
    max_holding = int(params['max_holding'])

    ln_y = np.log(df['Close_Y'])
    ln_x = np.log(df['Close_X'])

    beta  = ln_y.rolling(lookback).cov(ln_x) / ln_x.rolling(lookback).var()
    alpha = ln_y.rolling(lookback).mean() - beta * ln_x.rolling(lookback).mean()

    df['beta']        = beta
    df['alpha']       = alpha
    df['spread']      = ln_y - (alpha + beta * ln_x)
    df['spread_mean'] = df['spread'].rolling(z_lookback).mean()
    df['spread_std']  = df['spread'].rolling(z_lookback).std()
    df['z']           = (df['spread'] - df['spread_mean']) / df['spread_std']

    r_y = df['Close_Y'].pct_change()
    r_x = df['Close_X'].pct_change()
    df['spread_return'] = r_y - beta.shift(1) * r_x

    position      = np.zeros(len(df), dtype=int)
    holding_count = 0

    for i in range(1, len(df)):
        z_curr = df['z'].iloc[i]
        prev   = position[i - 1]

        if np.isnan(z_curr):
            position[i] = 0
            continue

        if prev != 0:
            holding_count += 1
            if abs(z_curr) > stop_z:
                position[i]   = 0
                holding_count = 0
                continue
            if holding_count >= max_holding:
                position[i]   = 0
                holding_count = 0
                continue
            if abs(z_curr) < exit_z:
                position[i]   = 0
                holding_count = 0
                continue
            position[i] = prev
            continue

        if z_curr > entry:
            position[i]   = -1   # short spread (sell Y, buy X)
            holding_count = 0
        elif z_curr < -entry:
            position[i]   = 1    # long spread (buy Y, sell X)
            holding_count = 0
        else:
            position[i] = 0

    df['position']         = position
    df['position_size']    = np.where(position != 0, 1.0, 0.0)
    df['stop_loss']        = 0.0
    df['strategy_returns'] = df['spread_return']

    for col in ('position', 'position_size', 'stop_loss', 'strategy_returns'):
        df[col] = df[col].fillna(0)
    df['position'] = df['position'].astype(int)

    return df, ['beta', 'alpha', 'spread', 'z']


STRATEGY_REGISTRY = {'stat_arb_spread': stat_arb_spread}
