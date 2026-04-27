"""
CPCV Engine — end-to-end validation script
=========================================
Tests generate_cpcv_splits, run_cpcv, cpcv_parameter_analysis, cpcv_summary
using the momentum ETH daily strategy (copied from wf_testing_2/momentumETH_wf.ipynb).

Run from repo root:
    python topics/momentum/strategies/momentum_cpcv/test_cpcv.py
"""

import os
import sys
import math
import numpy as np
import pandas as pd

ROOT = os.path.expanduser('~/Desktop/epsilon/github/Epsilon-Quant-Research')
sys.path.insert(0, os.path.join(ROOT, 'infrastructure', 'data'))
sys.path.insert(0, os.path.join(ROOT, 'infrastructure', 'walkforward'))
sys.path.insert(0, os.path.join(ROOT, 'infrastructure', 'backtester'))

from binance_client import get_binance_client, get_data
from cpcv_engine import (
    generate_cpcv_splits,
    run_cpcv,
    cpcv_parameter_analysis,
    cpcv_summary,
)
from cpcv_visualizer import (
    plot_path_equity_curves,
    plot_path_distribution,
    plot_parameter_distributions,
    plot_parameter_correlation_matrix,
    plot_split_performance_heatmap,
    plot_tercile_comparison,
    plot_cpcv_results,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Strategy — exact copy from wf_testing_2/momentumETH_wf.ipynb
# ─────────────────────────────────────────────────────────────────────────────

PARAM_DEFS = {
    'ema_span':              ('int',   5,    40),
    'swing_caution':         ('int',   3,    14),
    'swing_stop':            ('int',   3,    10),
    'atr_caution':           ('int',   10,   30),
    'atr_stop':              ('int',   24,   30),
    'atr_size':              ('int',   3,    14),
    'adx_override':          ('int',   40,   80),
    'stop_atr_scale':        ('float', 0.5,  2.0),
    'risk_per_trade':        ('float', 0.005, 0.05),
    'max_leverage':          ('float', 1.0,  3.0),
    'stop_mult_pos_caution': ('float', 0.1,  0.9),
    'stop_mult_pos_normal':  ('float', 0.8,  2.0),
    'stop_mult_ent_both':    ('float', 0.5,  2.5),
    'stop_mult_ent_caution': ('float', 0.1,  0.9),
    'stop_mult_ent_normal':  ('float', 0.5,  1.5),
    'vol_ma_period':         ('int',   35,   40),
    'obv_ma_period':         ('int',   18,   24),
    'obv_lookback':          ('int',   24,   30),
}

FIXED_PARAMS = {
    'risk_per_trade':        0.045,
    'max_leverage':          2.6,
    'ema_span':              28,
    'stop_mult_pos_caution': 0.5,
    'obv_lookback':          28,
    'obv_ma_period':         20,
    'swing_caution':         5,
}


def my_strategy(df_slice: pd.DataFrame, params: dict):
    df = df_slice.copy()

    df['EMA']          = df['Close'].ewm(span=params['ema_span'], adjust=False).mean()
    df['Swing_Hi_Cau'] = df['High'].rolling(params['swing_caution']).max()
    df['Swing_Lo_Cau'] = df['Low'].rolling(params['swing_caution']).min()
    df['Swing_Hi_Stp'] = df['High'].rolling(params['swing_stop']).max()

    def atr(period):
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift(1)).abs()
        lc = (df['Low']  - df['Close'].shift(1)).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=period, adjust=False).mean()

    df['ATR_Cau'] = atr(params['atr_caution'])
    df['ATR_Stp'] = atr(params['atr_stop'])
    df['ATR_Sz']  = atr(params['atr_size'])

    up    = df['High'].diff()
    down  = -df['Low'].diff()
    pdm   = up.where((up > down) & (up > 0), 0.0)
    ndm   = down.where((down > up) & (down > 0), 0.0)
    atr14 = atr(14)
    pdi   = 100 * pdm.ewm(span=14, adjust=False).mean() / atr14
    ndi   = 100 * ndm.ewm(span=14, adjust=False).mean() / atr14
    dx    = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).fillna(0)
    df['ADX_14'] = dx.ewm(span=14, adjust=False).mean()

    df['Vol_MA']  = df['Volume'].rolling(params['vol_ma_period']).mean()
    direction     = df['Close'].diff().apply(lambda x: 1 if x > 0 else -1)
    df['OBV']     = (df['Volume'] * direction).cumsum()
    df['OBV_MA']  = df['OBV'].rolling(params['obv_ma_period']).mean()

    df['Caution_OBV']   = (
        (df['Close'] > df['Close'].shift(params['obv_lookback'])) &
        (df['OBV'] < df['OBV_MA'])
    )
    df['Caution_Long']  = (
        ((df['Swing_Hi_Cau'] - df['Low']) > 1.5 * df['ATR_Cau']) |
        df['Caution_OBV']
    )
    df['Caution_Short'] = (
        ((df['High'] - df['Swing_Lo_Cau']) > 1.5 * df['ATR_Cau']) |
        (df['Close'] > df['EMA'])
    )
    _valid = (
        df['Swing_Hi_Stp'].notna() & df['ATR_Stp'].notna() &
        df['ATR_Sz'].notna() & df['OBV_MA'].notna() & df['Vol_MA'].notna()
    )
    df['Entry_Long'] = (
        (df['Close'] > df['EMA']) &
        (~df['Caution_Long'] | (df['ADX_14'] > params['adx_override'])) &
        (df['Volume'] > df['Vol_MA']) &
        _valid
    )
    df['position_size_raw'] = (
        params['risk_per_trade'] / (df['ATR_Sz'] / df['Close'])
    ).clip(0.1, params['max_leverage'])

    n             = len(df)
    position      = [0]      * n
    position_size = [0.0]    * n
    stop_arr      = [np.nan] * n
    in_position   = 0
    stop_loss     = np.nan
    current_size  = 0.0

    for i in range(1, n):
        curr = df.iloc[i]

        # Check exit first, then allow same-bar re-entry
        if in_position == 1:
            if df.iloc[i - 1]['Close'] < stop_loss:
                in_position  = 0
                current_size = 0.0
                stop_loss    = np.nan
            else:
                sm        = (params['stop_mult_pos_caution']
                             if curr['Caution_Long'] else params['stop_mult_pos_normal'])
                stop_loss = max(
                    stop_loss,
                    curr['Swing_Hi_Stp'] - curr['ATR_Stp'] * sm * params['stop_atr_scale'],
                )

        if in_position == 0:
            if curr['Entry_Long']:
                in_position  = 1
                current_size = curr['position_size_raw']
                cl = curr['Caution_Long']
                cs = curr['Caution_Short']
                if cl and cs:
                    sm = params['stop_mult_ent_both']
                elif cl:
                    sm = params['stop_mult_ent_caution']
                else:
                    sm = params['stop_mult_ent_normal']
                stop_loss = (curr['Swing_Hi_Stp'] -
                             curr['ATR_Stp'] * sm * params['stop_atr_scale'])

        position[i]      = in_position
        position_size[i] = current_size
        stop_arr[i]      = stop_loss

    df['position']      = position
    df['position_size'] = position_size
    df['stop_loss']     = stop_arr

    indicator_cols = ['EMA', 'ATR_Cau', 'ADX_14', 'Swing_Hi_Cau', 'Vol_MA', 'OBV_MA']
    df['position']      = df['position'].fillna(0).astype(int)
    df['position_size'] = df['position_size'].fillna(0.0)
    df['stop_loss']     = df['stop_loss'].fillna(0.0)

    return df, indicator_cols


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

_PASS = '  PASS'
_FAIL = '  FAIL'

def _check(condition, label, detail=''):
    tag = _PASS if condition else _FAIL
    suffix = f'  ({detail})' if detail else ''
    print(f'{tag}  {label}{suffix}')
    return condition


# ─────────────────────────────────────────────────────────────────────────────
#  Test 1 — generate_cpcv_splits (no data needed)
# ─────────────────────────────────────────────────────────────────────────────

def test1_generate_splits():
    print('\n' + '═' * 60)
    print('TEST 1 — generate_cpcv_splits')
    print('═' * 60)

    N_BARS = 2150
    N, K, PURGE = 8, 2, 1

    meta   = generate_cpcv_splits(N_BARS, N, K, PURGE)
    groups = meta['groups']
    splits = meta['splits']
    paths  = meta['paths']

    ok = True

    # ── group count ───────────────────────────────────────────────────────────
    ok &= _check(len(groups) == N, f'len(groups) == {N}', f'got {len(groups)}')

    # ── groups tile 0..N_BARS without gaps or overlaps ────────────────────────
    prev_end = 0
    tiling_ok = True
    for i, (s, e) in enumerate(groups):
        if s != prev_end:
            tiling_ok = False
            print(f'       gap at group {i}: expected start {prev_end}, got {s}')
        prev_end = e
    tiling_ok = tiling_ok and (prev_end == N_BARS)
    ok &= _check(tiling_ok, f'groups tile 0..{N_BARS} without gaps', f'last end={prev_end}')

    # ── split count ───────────────────────────────────────────────────────────
    expected_splits = math.comb(N, K)
    ok &= _check(
        len(splits) == expected_splits,
        f'len(splits) == C({N},{K}) = {expected_splits}',
        f'got {len(splits)}',
    )

    # ── path count ────────────────────────────────────────────────────────────
    print(f'       paths generated: {len(paths)}')

    # ── each path covers all N groups exactly once ────────────────────────────
    bad_paths = []
    for p in paths:
        covered = sorted(g for g, _ in p['split_assignments'])
        if covered != list(range(N)):
            bad_paths.append(p['path_id'])
    ok &= _check(
        len(bad_paths) == 0,
        'every path covers groups 0..7 exactly once',
        f'{len(bad_paths)} bad paths' if bad_paths else '',
    )

    # ── example split inspection ──────────────────────────────────────────────
    sp = splits[0]
    tg = sp['test_group_indices']
    n_train  = len(sp['train_indices'])
    n_test   = sum(len(v) for v in sp['test_indices_by_group'].values())
    total    = n_train + n_test
    print(f'\n  Split 0: test_groups={tg}')
    print(f'    train_bars={n_train}  test_bars={n_test}  '
          f'sum={total}  (≈{N_BARS} after purge deduction)')
    ok &= _check(
        total <= N_BARS,
        'train + test <= n_bars (purge only subtracts from train)',
        f'{total} <= {N_BARS}',
    )
    ok &= _check(
        n_train > 0 and n_test > 0,
        'split 0 has non-empty train and test sets',
    )

    # ── example path inspection ───────────────────────────────────────────────
    p0 = paths[0]
    print(f'\n  Path 0 split_assignments: {p0["split_assignments"]}')
    groups_in_path = [g for g, _ in p0['split_assignments']]
    ok &= _check(
        sorted(groups_in_path) == list(range(N)),
        'path 0 covers groups 0..7 exactly once',
        f'groups={sorted(groups_in_path)}',
    )

    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  Test 2 — run_cpcv end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test2_run_cpcv(df):
    print('\n' + '═' * 60)
    print('TEST 2 — run_cpcv (n_trials=10)')
    print('═' * 60)

    N, K = 8, 2
    expected_splits = math.comb(N, K)   # 28
    free_params     = [p for p in PARAM_DEFS if p not in FIXED_PARAMS]

    results = run_cpcv(
        df           = df,
        strategy_fn  = my_strategy,
        param_defs   = PARAM_DEFS,
        fixed_params = FIXED_PARAMS,
        N            = N,
        k            = K,
        purge_bars   = 1,
        n_trials     = 10,
        burnin       = 100,
        cost         = 0.001,
        verbose      = True,
    )

    ok = True

    # ── split_results count ───────────────────────────────────────────────────
    ok &= _check(
        len(results['split_results']) == expected_splits,
        f'len(split_results) == {expected_splits}',
        f'got {len(results["split_results"])}',
    )

    # ── paths count ───────────────────────────────────────────────────────────
    n_paths = len(results['paths'])
    print(f'       paths: {n_paths}')
    ok &= _check(n_paths > 0, 'paths list is non-empty')

    # ── param_distributions shape ─────────────────────────────────────────────
    pd_shape = results['param_distributions'].shape
    ok &= _check(
        pd_shape == (expected_splits, len(free_params)),
        f'param_distributions.shape == ({expected_splits}, {len(free_params)})',
        f'got {pd_shape}',
    )
    print(f'\n  param_distributions.head():')
    print(results['param_distributions'].head().to_string(index=False))

    # ── param values within PARAM_DEFS ranges ────────────────────────────────
    range_ok = True
    for p in free_params:
        _, lo, hi = PARAM_DEFS[p]
        vals = results['param_distributions'][p].dropna()
        if not vals.empty and not (vals.between(lo - 1e-9, hi + 1e-9).all()):
            range_ok = False
            print(f'       OUT OF RANGE: {p}  range=[{lo},{hi}]  '
                  f'got min={vals.min():.4f} max={vals.max():.4f}')
    ok &= _check(range_ok, 'all optimised params within PARAM_DEFS ranges')

    # ── first valid path equity curve ─────────────────────────────────────────
    valid_path = next((p for p in results['paths'] if p['equity_curve'] is not None), None)
    if valid_path is not None:
        eq = valid_path['equity_curve']
        ok &= _check(
            isinstance(eq, pd.Series),
            'equity_curve is a pd.Series',
        )
        ok &= _check(
            pd.api.types.is_datetime64_any_dtype(eq.index),
            'equity_curve has a DatetimeIndex',
        )
        ok &= _check(
            eq.dtype == float and not eq.isna().all(),
            'equity_curve contains finite float values',
        )
        print(f'\n  Path {valid_path["path_id"]} equity_curve:')
        print(f'    head: {eq.head(3).values}')
        print(f'    tail: {eq.tail(3).values}')

        sharpe = valid_path['sharpe']
        ok &= _check(
            sharpe is not None and np.isfinite(sharpe),
            f'path sharpe is a finite number',
            f'got {sharpe}',
        )
    else:
        print('  WARN  no path produced a valid equity curve '
              '(all strategy_fn calls may have failed with n_trials=10)')

    # ── config recorded correctly ─────────────────────────────────────────────
    cfg = results['config']
    ok &= _check(cfg['N']        == N,     'config.N',        f'{cfg["N"]}')
    ok &= _check(cfg['k']        == K,     'config.k',        f'{cfg["k"]}')
    ok &= _check(cfg['n_trials'] == 10,    'config.n_trials', f'{cfg["n_trials"]}')
    ok &= _check(cfg['burnin']   == 100,   'config.burnin',   f'{cfg["burnin"]}')
    ok &= _check(cfg['cost']     == 0.001, 'config.cost',     f'{cfg["cost"]}')

    return ok, results


# ─────────────────────────────────────────────────────────────────────────────
#  Test 3 — cpcv_parameter_analysis
# ─────────────────────────────────────────────────────────────────────────────

def test3_parameter_analysis(results):
    print('\n' + '═' * 60)
    print('TEST 3 — cpcv_parameter_analysis')
    print('═' * 60)

    analysis = cpcv_parameter_analysis(results)
    ok       = True

    # ── distribution_stats ────────────────────────────────────────────────────
    dist = analysis['distribution_stats']
    free_params = [p for p in PARAM_DEFS if p not in FIXED_PARAMS]
    print('\n  distribution_stats:')
    print(dist.to_string())

    ok &= _check(
        set(dist.index) == set(free_params),
        'distribution_stats has one row per free param',
        f'index={list(dist.index)}',
    )
    nan_rows = dist.index[dist[['mean','std','cv','median','iqr']].isna().any(axis=1)].tolist()
    ok &= _check(
        len(nan_rows) == 0,
        'no NaN values in distribution_stats core columns',
        f'NaN in: {nan_rows}' if nan_rows else '',
    )

    # ── param_performance_corr ────────────────────────────────────────────────
    corr = analysis['param_performance_corr']
    print('\n  param_performance_corr:')
    print(corr.to_string())
    ok &= _check(
        set(corr.index) == set(free_params),
        'param_performance_corr has one row per free param',
    )

    # ── consensus_ranges ─────────────────────────────────────────────────────
    cr = analysis['consensus_ranges']
    print('\n  consensus_ranges:')
    print(cr.to_string())
    ok &= _check(
        set(cr.index) == set(free_params),
        'consensus_ranges has one row per free param',
    )
    missing_action = cr.index[cr['action'].isna() | (cr['action'] == '')].tolist()
    ok &= _check(
        len(missing_action) == 0,
        'every free param has an action in consensus_ranges',
        f'missing: {missing_action}' if missing_action else '',
    )

    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  Test 4 — cpcv_summary
# ─────────────────────────────────────────────────────────────────────────────

def test4_summary(results):
    print('\n' + '═' * 60)
    print('TEST 4 — cpcv_summary')
    print('═' * 60)

    try:
        cpcv_summary(results)
        ok = _check(True, 'cpcv_summary printed without exceptions')
    except Exception as e:
        ok = _check(False, 'cpcv_summary raised an exception', str(e))

    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  Test 5 — cpcv_parameter_analysis structure
# ─────────────────────────────────────────────────────────────────────────────

def test5_analysis_structure(results):
    print('\n' + '═' * 60)
    print('TEST 5 — cpcv_parameter_analysis structure')
    print('═' * 60)

    analysis = cpcv_parameter_analysis(results)
    ok       = True

    # ── keys present ──────────────────────────────────────────────────────────
    expected_keys = {
        'distribution_stats', 'param_performance_corr',
        'cross_param_corr', 'tercile_comparison', 'consensus_ranges',
    }
    actual_keys = set(analysis.keys())
    print(f'\n  analysis.keys(): {sorted(actual_keys)}')
    ok &= _check(
        actual_keys == expected_keys,
        'all expected keys present',
        f'missing={expected_keys - actual_keys}  extra={actual_keys - expected_keys}',
    )

    # ── distribution_stats ────────────────────────────────────────────────────
    dist = analysis['distribution_stats']
    free_params = [p for p in PARAM_DEFS if p not in FIXED_PARAMS]
    print(f'\n  distribution_stats.shape: {dist.shape}')
    ok &= _check(
        dist.shape == (len(free_params), 9),
        f'distribution_stats.shape == ({len(free_params)}, 9)',
        f'got {dist.shape}',
    )
    print('\n  distribution_stats.head():')
    print(dist.head().to_string())

    # ── param_performance_corr ────────────────────────────────────────────────
    ppc = analysis['param_performance_corr']
    print(f'\n  param_performance_corr.shape: {ppc.shape}')
    ok &= _check(
        ppc.shape == (len(free_params), 2),
        f'param_performance_corr.shape == ({len(free_params)}, 2)',
        f'got {ppc.shape}',
    )
    print('\n  param_performance_corr.head():')
    print(ppc.head().to_string())

    # ── cross_param_corr ──────────────────────────────────────────────────────
    cpc = analysis['cross_param_corr']
    print(f'\n  cross_param_corr.shape: {cpc.shape}')
    ok &= _check(
        cpc.shape == (len(free_params), len(free_params)),
        f'cross_param_corr.shape == ({len(free_params)}, {len(free_params)})',
        f'got {cpc.shape}',
    )

    # ── consensus_ranges — full table, every free param has an action ─────────
    cr = analysis['consensus_ranges']
    print('\n  consensus_ranges (full):')
    print(cr.to_string())
    missing_action = cr.index[cr['action'].isna() | (cr['action'] == '')].tolist()
    ok &= _check(
        set(cr.index) == set(free_params),
        'consensus_ranges has one row per free param',
        f'index={sorted(cr.index)}',
    )
    ok &= _check(
        len(missing_action) == 0,
        'every free param has an action in consensus_ranges',
        f'missing action: {missing_action}' if missing_action else '',
    )

    return ok, analysis


# ─────────────────────────────────────────────────────────────────────────────
#  Test 6 — each plot function runs without error
# ─────────────────────────────────────────────────────────────────────────────

def test6_plot_functions(results, analysis):
    print('\n' + '═' * 60)
    print('TEST 6 — visualiser functions (show=False)')
    print('═' * 60)

    plot_calls = [
        ('plot_path_equity_curves',
         lambda: plot_path_equity_curves(results, show=False)),
        ('plot_path_distribution',
         lambda: plot_path_distribution(results, show=False)),
        ('plot_parameter_distributions',
         lambda: plot_parameter_distributions(results, analysis=analysis, show=False)),
        ('plot_parameter_correlation_matrix',
         lambda: plot_parameter_correlation_matrix(analysis, show=False)),
        ('plot_split_performance_heatmap',
         lambda: plot_split_performance_heatmap(results, show=False)),
        ('plot_tercile_comparison',
         lambda: plot_tercile_comparison(results, analysis, show=False)),
        ('plot_cpcv_results',
         lambda: plot_cpcv_results(results, analysis=analysis, show=False)),
    ]

    n_pass = 0
    print()
    for name, fn in plot_calls:
        try:
            fn()
            print(f'{_PASS}  {name}')
            n_pass += 1
        except Exception as e:
            print(f'{_FAIL}  {name}  ({e})')

    print(f'\n  {n_pass} / {len(plot_calls)} plots passed')
    return n_pass == len(plot_calls)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_ok = True

    # ── Test 1 ────────────────────────────────────────────────────────────────
    ok1 = test1_generate_splits()
    all_ok &= ok1
    if not ok1:
        print('\nTest 1 failed — stopping early.')
        return

    # ── Fetch data ────────────────────────────────────────────────────────────
    print('\n' + '═' * 60)
    print('FETCHING DATA — ETHUSDT 1d 2150 bars')
    print('═' * 60)
    client = get_binance_client()
    df     = get_data(client, 'ETHUSDT', '1d', 2150)
    print(f'  {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} bars)')

    # ── Test 2 ────────────────────────────────────────────────────────────────
    ok2, results = test2_run_cpcv(df)
    all_ok &= ok2

    # ── Test 3 ────────────────────────────────────────────────────────────────
    ok3 = test3_parameter_analysis(results)
    all_ok &= ok3

    # ── Test 4 ────────────────────────────────────────────────────────────────
    ok4 = test4_summary(results)
    all_ok &= ok4

    # ── Test 5 ────────────────────────────────────────────────────────────────
    ok5, analysis = test5_analysis_structure(results)
    all_ok &= ok5

    # ── Test 6 ────────────────────────────────────────────────────────────────
    ok6 = test6_plot_functions(results, analysis)
    all_ok &= ok6

    # ── Final verdict ─────────────────────────────────────────────────────────
    print('\n' + '═' * 60)
    if all_ok:
        print('All CPCV engine + visualiser tests passed')
    else:
        print('One or more tests FAILED — see output above')
    print('═' * 60)


if __name__ == '__main__':
    main()
