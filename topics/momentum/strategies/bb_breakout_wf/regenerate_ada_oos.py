"""
Regenerate ada_oos.pkl with correct walk-forward windows.

Root cause: ADA.ipynb was left with TEST_BARS=1 (one 1H bar per fold) and
TRAIN_BARS=51600 (larger than the available data), producing a single fold
with a 1-row OOS pkl.

Fix: restore TRAIN_BARS=17520 (2y), TEST_BARS=4380 (6m) as documented in the
notebook comment. All consensus params are placed in FIXED_PARAMS so Optuna
has nothing to search — N_TRIALS=1 runs each fold in seconds.
"""

import sys
import os
import numpy.core

# ── numpy compat shim (pkl files saved with numpy 2.x) ───────────────────────
sys.modules.setdefault('numpy._core', numpy.core)
for _sub in ['numeric', 'multiarray', 'umath', 'fromnumeric',
             'function_base', 'arrayprint', 'shape_base']:
    try:
        sys.modules.setdefault(
            f'numpy._core.{_sub}',
            __import__(f'numpy.core.{_sub}', fromlist=[_sub])
        )
    except ImportError:
        pass

ROOT = r'C:\Users\user\Documents\Epsilon Fund\Epsilon-Quant-Research'
sys.path.append(os.path.join(ROOT, 'infrastructure', 'data'))
sys.path.append(os.path.join(ROOT, 'infrastructure', 'walkforward'))
sys.path.append(os.path.join(ROOT, 'infrastructure', 'backtester'))

import pandas as pd
import numpy as np
from binance_client import get_binance_client, get_data
from wf_engine import walk_forward

# ── all consensus params fixed → N_TRIALS=1 is sufficient ────────────────────
FIXED_PARAMS = {
    'bb_period':         32,
    'bb_exp_window':     7,
    'atr_period':        7,
    'breakout_pct':      0.5199,
    'breakout_lookback': 100,
    'h4_ma_period':      41,
    'slope_epsilon':     0.0005,
    'h1_ma_period':      10,
    'entry_zone_bps':    49,
    'overshoot_bps':     20,
    'max_1h_bars':       13,
    'pullback_atr_mult': 2.9524,
    'trail_atr_mult':    3.7041,
    'adx_period':        14,
    'adx_strong':        50.7566,
    'trend_ma_period':   244,
    'risk_per_trade':    0.03,
    'max_leverage':      2.5,
}

PARAM_DEFS = {
    'bb_period':         ('int',   10, 40),
    'bb_exp_window':     ('int',   2,  20),
    'atr_period':        ('int',   5,  20),
    'breakout_pct':      ('float', 0.50, 0.85),
    'breakout_lookback': ('int',   20, 100),
    'h4_ma_period':      ('int',   10, 50),
    'slope_epsilon':     ('float', 0.0, 0.003),
    'h1_ma_period':      ('int',   5,  17),
    'entry_zone_bps':    ('int',   5,  100),
    'overshoot_bps':     ('int',   5,  110),
    'max_1h_bars':       ('int',   12, 48),
    'pullback_atr_mult': ('float', 1.0, 3.0),
    'trail_atr_mult':    ('float', 0.5, 4.0),
    'adx_period':        ('int',   7, 21),
    'adx_strong':        ('float', 20.0, 60.0),
    'trend_ma_period':   ('int',   150, 300),
    'risk_per_trade':    ('float', 0.01, 0.05),
    'max_leverage':      ('float', 1.0, 3.0),
}


def my_strategy(df_slice: pd.DataFrame, params: dict):

    df = df_slice.copy()
    h1 = df.rename(columns=str.lower)

    h4 = h1.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()

    def _atr(d, period):
        hi, lo, cl = d['high'], d['low'], d['close']
        prev_cl = cl.shift(1)
        tr = pd.concat([(hi - lo), (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    def _sma(d, period):
        return d['close'].rolling(period).mean()

    def _bb_width(d, period):
        mid = d['close'].rolling(period).mean()
        std = d['close'].rolling(period).std()
        return (std * 2) / mid.replace(0, np.nan)

    def _ma_slope(d, period):
        ma = d['close'].rolling(period).mean()
        return ma - ma.shift(1)

    def _candle_range(d):
        return d['high'] - d['low']

    def _adx(d, period):
        hi, lo, cl = d['high'], d['low'], d['close']
        prev_hi = hi.shift(1)
        prev_lo = lo.shift(1)
        prev_cl = cl.shift(1)
        tr = pd.concat([(hi - lo), (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1).max(axis=1)
        plus_dm  = (hi - prev_hi).clip(lower=0).where((hi - prev_hi) > (prev_lo - lo), 0.0)
        minus_dm = (prev_lo - lo).clip(lower=0).where((prev_lo - lo) > (hi - prev_hi), 0.0)
        alpha    = 1.0 / period
        atr_w    = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_w.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_w.replace(0, np.nan)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx      = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        return adx, plus_di, minus_di

    h4_atr   = _atr(h4, params['atr_period'])
    h4_range = _candle_range(h4)
    h4_bw    = _bb_width(h4, params['bb_period'])
    h4_slope = _ma_slope(h4, params['h4_ma_period'])

    h4_green = h4['close'] > h4['open']
    h4_red   = h4['close'] < h4['open']

    brk_threshold = h4_range.rolling(int(params['breakout_lookback'])).quantile(params['breakout_pct'])
    big           = h4_range > brk_threshold

    two_big_green = big & big.shift(1) & h4_green & h4_green.shift(1)
    two_big_red   = big & big.shift(1) & h4_red   & h4_red.shift(1)

    bb_exp        = h4_bw > h4_bw.rolling(params['bb_exp_window']).mean()

    h4_slope_norm = h4_slope / h4['close'].replace(0, np.nan)
    slope_eps     = params['slope_epsilon']
    h4_long  = two_big_green & bb_exp & (h4_slope_norm >= -slope_eps)
    h4_short = two_big_red   & bb_exp & (h4_slope_norm <=  slope_eps)

    h4_adx, h4_plus_di, h4_minus_di = _adx(h4, params['adx_period'])

    h1_atr      = _atr(h1,  params['atr_period'])
    h1_range    = _candle_range(h1)
    h1_sma      = _sma(h1,  params['h1_ma_period'])
    trend_period = int(params['trend_ma_period'])
    h1_trend_ma  = h1['close'].rolling(trend_period).mean()

    h1_pos_size = (
        params['risk_per_trade'] / (h1_atr / h1['close'])
    ).clip(0.1, params['max_leverage'])

    h4_adx_1h      = h4_adx.shift(1).reindex(h1.index,      method='ffill').fillna(0.0)
    h4_plus_di_1h  = h4_plus_di.shift(1).reindex(h1.index,  method='ffill').fillna(0.0)
    h4_minus_di_1h = h4_minus_di.shift(1).reindex(h1.index, method='ffill').fillna(0.0)

    h4_long_1h  = h4_long.shift(1).reindex(h1.index,  method='ffill').fillna(False)
    h4_short_1h = h4_short.shift(1).reindex(h1.index, method='ffill').fillna(False)

    long_setup_fires  = h4_long_1h  & ~h4_long_1h.shift(1).fillna(False)
    short_setup_fires = h4_short_1h & ~h4_short_1h.shift(1).fillna(False)

    close_arr     = h1['close'].to_numpy()
    sma_arr       = h1_sma.to_numpy()
    range_arr     = h1_range.to_numpy()
    atr_arr       = h1_atr.to_numpy()
    pos_size_arr  = h1_pos_size.to_numpy()
    trend_ma_arr  = h1_trend_ma.to_numpy()
    adx_arr       = h4_adx_1h.to_numpy()
    plus_di_arr   = h4_plus_di_1h.to_numpy()
    minus_di_arr  = h4_minus_di_1h.to_numpy()
    long_fire     = long_setup_fires.to_numpy()
    short_fire    = short_setup_fires.to_numpy()

    max_1h_bars       = params['max_1h_bars']
    pullback_atr_mult = params['pullback_atr_mult']
    entry_zone_bps    = params['entry_zone_bps']
    overshoot_bps     = params['overshoot_bps']

    n             = len(h1)
    position      = np.zeros(n, dtype=int)
    position_size = np.ones(n)
    stop_loss     = np.zeros(n)

    setup_active    = False
    setup_direction = 0
    bars_since      = 0
    in_trade        = False
    trade_direction = 0
    trade_stop      = 0.0
    trade_tp        = 0.0
    trade_size      = 1.0

    for i in range(1, n):
        if in_trade:
            close      = close_arr[i]
            h1_at_i    = atr_arr[i]
            trail_mult = params['trail_atr_mult']

            if not np.isnan(h1_at_i):
                if trade_direction == 1:
                    trade_stop = max(trade_stop, close - trail_mult * h1_at_i)
                else:
                    trade_stop = min(trade_stop, close + trail_mult * h1_at_i)

            stop_hit = (
                (trade_direction ==  1 and trade_stop > 0 and close <= trade_stop) or
                (trade_direction == -1 and trade_stop > 0 and close >= trade_stop)
            )
            tp_hit = (
                (trade_direction ==  1 and trade_tp > 0 and close >= trade_tp) or
                (trade_direction == -1 and trade_tp > 0 and close <= trade_tp)
            )

            if stop_hit or tp_hit:
                in_trade = False
            else:
                position[i]      = trade_direction
                position_size[i] = trade_size
            continue

        if not setup_active:
            if long_fire[i]:
                setup_active, setup_direction, bars_since = True, 1, 0
            elif short_fire[i]:
                trend_ma_i = trend_ma_arr[i]
                adx_i      = adx_arr[i]
                plus_di_i  = plus_di_arr[i]
                minus_di_i = minus_di_arr[i]
                adx_strong = params['adx_strong']

                above_ma   = not np.isnan(trend_ma_i) and close_arr[i] > trend_ma_i
                bull_trend = adx_i > adx_strong and plus_di_i > minus_di_i
                if not above_ma and not bull_trend:
                    setup_active, setup_direction, bars_since = True, -1, 0

        if not setup_active:
            continue

        bars_since += 1
        close  = close_arr[i]
        s_ma   = sma_arr[i]
        h1_rng = range_arr[i]
        h1_at  = atr_arr[i]

        if np.isnan(s_ma) or np.isnan(h1_at) or s_ma == 0:
            continue

        if bars_since > max_1h_bars:
            setup_active = False
            continue

        if h1_rng > pullback_atr_mult * h1_at:
            setup_active = False
            continue

        if setup_direction == 1:
            if close < s_ma - (s_ma * overshoot_bps / 10000):
                setup_active = False
                continue
        else:
            if close > s_ma + (s_ma * overshoot_bps / 10000):
                setup_active = False
                continue

        bps_from_sma = abs(close - s_ma) / s_ma * 10000
        in_zone = bps_from_sma <= entry_zone_bps

        prev_close  = close_arr[i - 1]
        momentum_ok = (
            (setup_direction ==  1 and close > prev_close) or
            (setup_direction == -1 and close < prev_close)
        )

        if in_zone and momentum_ok:
            raw_size   = pos_size_arr[i]
            sz         = raw_size if not np.isnan(raw_size) else 1.0
            trail_mult = params['trail_atr_mult']

            initial_stop_dist = trail_mult * h1_at
            ts_val = (close - initial_stop_dist) if setup_direction == 1 else (close + initial_stop_dist)

            in_strong_bull = (
                close > trend_ma_arr[i]
                and adx_arr[i] > params['adx_strong']
                and plus_di_arr[i] > minus_di_arr[i]
            )

            if in_strong_bull or np.isnan(trend_ma_arr[i]):
                tp_val = 0.0
            else:
                tp_dist = 6 * initial_stop_dist
                tp_val  = (close + tp_dist) if setup_direction == 1 else (close - tp_dist)

            position[i]      = setup_direction
            position_size[i] = sz
            stop_loss[i]     = ts_val

            in_trade        = True
            trade_direction = setup_direction
            trade_stop      = ts_val
            trade_tp        = tp_val
            trade_size      = sz
            setup_active    = False

    indicator_cols      = ['SMA']
    df['SMA']           = h1_sma.to_numpy()
    df['position']      = position
    df['position_size'] = position_size
    df['stop_loss']     = stop_loss

    return df, indicator_cols


def score_fn(m):
    SHARPE_MAX = 3
    CALMAR_MAX = 10.0
    RETURN_MAX = 100.0
    calmar = m['total_return'] / abs(m['max_drawdown']) if m['max_drawdown'] != 0 else 0
    s = np.clip(m['sharpe_ratio']  / SHARPE_MAX, 0, 1)
    c = np.clip(calmar             / CALMAR_MAX, 0, 1)
    r = np.clip(m['total_return']  / RETURN_MAX, 0, 1)
    return 0.50 * s + 0.30 * c + 0.20 * r


def reject_fn(m):
    if m is None:                   return True
    if m['num_trades']   < 15:      return True
    if m['win_rate']     < 0.4:     return True
    if m['max_drawdown'] < -0.6:    return True
    if m['profit_factor'] < 0.5:   return True
    return False


if __name__ == '__main__':
    print('Fetching ADAUSDT 1H data...')
    client = get_binance_client()
    df = get_data(client, 'ADAUSDT', '1h', 2151)
    print(f'Data: {df.index[0].date()} to {df.index[-1].date()}  ({len(df)} bars)')

    results = walk_forward(
        df           = df,
        strategy_fn  = my_strategy,
        param_defs   = PARAM_DEFS,
        fixed_params = FIXED_PARAMS,
        train_bars   = 17520,
        test_bars    = 4380,
        burnin_bars  = 200,
        n_trials     = 1,
        cost         = 0.001,
        score_fn     = score_fn,
        reject_fn    = reject_fn,
    )

    oos_df = results['oos_combined_df']
    if oos_df is None or len(oos_df) == 0:
        print('\nERROR: No OOS data produced. Check reject_fn filters.')
        sys.exit(1)

    out_path = os.path.join(ROOT, 'topics', 'momentum', 'strategies',
                            'bb_breakout_wf', 'oos', 'ada_oos.pkl')
    oos_df.to_pickle(out_path)
    print(f'\nSaved: {out_path}')
    print(f'OOS rows: {len(oos_df)}')
    print(f'OOS period: {oos_df.index[0].date()} to {oos_df.index[-1].date()}')