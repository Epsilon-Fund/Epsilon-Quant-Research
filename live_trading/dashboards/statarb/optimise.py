"""
optimise.py — Walk-forward optimisation for one or all stat arb pairs.

Usage:
    python optimise.py                    -> runs all pairs in ASSET_CONFIG
    python optimise.py --pair FILSNX      -> runs FIL/SNX only
    python optimise.py --verify           -> print config table and exit

After running, the result is written to live_params.json.
Pairs in ACTIVE_ASSETS are processed; add a pair key to ACTIVE_ASSETS in
config.py to activate it in the dashboard after optimising.

Param sources — consensus values from walk-forward notebooks:
  FILSNX  -> topics/statistical-arbitrage/strategies/testing/Fil&SNX.ipynb
  ATOMARB -> topics/statistical-arbitrage/strategies/testing/ATOM&ARB.ipynb
  LINKTRX -> topics/statistical-arbitrage/strategies/testing/LINK&TRX.ipynb
  LTCAPT  -> topics/statistical-arbitrage/strategies/testing/LTC&APT.ipynb
"""

import argparse
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_LT_DIR        = os.path.abspath(os.path.join(_DASHBOARD_DIR, '..', '..'))
_ROOT          = os.path.dirname(_LT_DIR)
_INFRA         = os.path.join(_ROOT, 'infrastructure', 'data')
for _p in (_DASHBOARD_DIR, _LT_DIR, _INFRA):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.append(os.path.join(_ROOT, 'infrastructure', 'backtester'))
# ─────────────────────────────────────────────────────────────────────────────

from strategies import STRATEGY_REGISTRY

LIVE_PARAMS_PATH = os.path.join(_DASHBOARD_DIR, 'live_params.json')


# ══════════════════════════════════════════════════════════════════════════════
#  Pair config — one entry per pair.
#  param_defs and fixed_params sourced from each pair's walk-forward notebook.
#
#  To add a new pair:
#    1. Add an entry here (read its notebook for param_defs / fixed_params)
#    2. python optimise.py --pair <PAIR_KEY>
#    3. Add PAIR_KEY to ACTIVE_ASSETS in config.py
# ══════════════════════════════════════════════════════════════════════════════

ASSET_CONFIG = [
    # ── FILSNX — FILUSDT/SNXUSDT ─────────────────────────────────────────────
    # Source: Fil&SNX.ipynb  |  OOS Sharpe 1.51  |  WFE 0.66
    {
        'pair_key':  'FILSNX',
        'symbol_y':  'FILUSDT',
        'symbol_x':  'SNXUSDT',
        'strategy':  'stat_arb_spread',
        'lookback':  1500,
        'param_defs': {
            'lookback':    ('int',   75,  105),
            'z_lookback':  ('int',   50,   90),
            'entry':       ('float', 1.5,  3.0),
            'exit_z':      ('float', 0.5,  1.1),
            'stop_z':      ('float', 4.0,  5.0),
            'max_holding': ('int',    3,   14),
        },
        'fixed_params': {
            'entry':      1.8136,
            'stop_z':     4.647,
            'z_lookback': 85,
        },
    },

    # ── ATOMARB — ATOMUSDT/ARBUSDT ────────────────────────────────────────────
    # Source: ATOM&ARB.ipynb  |  OOS Sharpe 1.22  |  WFE 0.82
    {
        'pair_key':  'ATOMARB',
        'symbol_y':  'ATOMUSDT',
        'symbol_x':  'ARBUSDT',
        'strategy':  'stat_arb_spread',
        'lookback':  1500,
        'param_defs': {
            'lookback':    ('int',   80,  120),
            'z_lookback':  ('int',   70,  100),
            'entry':       ('float', 1.5,  3.0),
            'exit_z':      ('float', 0.5,  0.75),
            'stop_z':      ('float', 3.5,  5.0),
            'max_holding': ('int',   25,   40),
        },
        'fixed_params': {
            'z_lookback': 82,
            'entry':      1.7528,
            'stop_z':     4.4448,
            'lookback':   114,
            'exit_z':     0.6867,
        },
    },

    # ── LINKTRX — LINKUSDT/TRXUSDT ───────────────────────────────────────────
    # Source: LINK&TRX.ipynb  |  OOS Sharpe 1.21 (combined)  |  WFE 0.85
    {
        'pair_key':  'LINKTRX',
        'symbol_y':  'LINKUSDT',
        'symbol_x':  'TRXUSDT',
        'strategy':  'stat_arb_spread',
        'lookback':  1500,
        'param_defs': {
            'lookback':    ('int',   80,  130),
            'z_lookback':  ('int',   38,  120),
            'entry':       ('float', 1.5,  2.5),
            'exit_z':      ('float', 0.5,  1.2),
            'stop_z':      ('float', 3.5,  5.0),
            'max_holding': ('int',    3,   15),
        },
        'fixed_params': {
            'stop_z':      4.0385,
            'max_holding': 6,
            'lookback':    110,
            'entry':       1.9137,
        },
    },

    # ── LTCAPT — LTCUSDT/APTUSDT ─────────────────────────────────────────────
    # Source: LTC&APT.ipynb  |  OOS Sharpe 1.10  |  WFE 0.61
    {
        'pair_key':  'LTCAPT',
        'symbol_y':  'LTCUSDT',
        'symbol_x':  'APTUSDT',
        'strategy':  'stat_arb_spread',
        'lookback':  1500,
        'param_defs': {
            'lookback':    ('int',   50,  100),
            'z_lookback':  ('int',   38,   70),
            'entry':       ('float', 2.5,  2.8),
            'exit_z':      ('float', 1.0,  1.3),
            'stop_z':      ('float', 3.5,  5.0),
            'max_holding': ('int',    7,   15),
        },
        'fixed_params': {
            'stop_z':  4.2325,
            'lookback': 73,
            'entry':   2.6184,
            'exit_z':  1.099,
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  Merge helper
# ══════════════════════════════════════════════════════════════════════════════

def _merge_pair(df_y: pd.DataFrame, df_x: pd.DataFrame) -> pd.DataFrame:
    y = df_y[['Close']].rename(columns={'Close': 'Close_Y'})
    x = df_x[['Close']].rename(columns={'Close': 'Close_X'})
    return y.join(x, how='inner').dropna()


# ══════════════════════════════════════════════════════════════════════════════
#  Scoring, rejection, and backtest helpers
# ══════════════════════════════════════════════════════════════════════════════

def _score(metrics):
    """Normalised composite: Sharpe 50% | Calmar 30% | Return 20%."""
    SHARPE_MAX = 2.5
    CALMAR_MAX = 60.0
    RETURN_MAX = 15.0

    calmar = (metrics['total_return'] / abs(metrics['max_drawdown'])
              if metrics['max_drawdown'] != 0 else 0.0)

    s = np.clip(metrics['sharpe_ratio'] / SHARPE_MAX, 0, 1)
    c = np.clip(calmar                  / CALMAR_MAX, 0, 1)
    r = np.clip(metrics['total_return'] / RETURN_MAX, 0, 1)
    return 0.50 * s + 0.30 * c + 0.20 * r


def _reject(metrics):
    if metrics is None:               return True
    if metrics['num_trades']    < 6:  return True
    if metrics['win_rate']      < 0.4: return True
    if metrics['max_drawdown']  < -0.60: return True
    if metrics['profit_factor'] < 0.6: return True
    return False


def _run_backtest(strategy_df, cost, backtest_fn):
    try:
        return backtest_fn(data=strategy_df, cost=cost, show_plot=False,
                           save_html=None, show_trades=False, benchmark_data=None)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Core optimisation
# ══════════════════════════════════════════════════════════════════════════════

def optimise_pair(asset: dict, get_binance_client, get_data, backtest_fn,
                  optuna, WF_CONFIG, INDICATOR_WARMUP) -> dict:
    """Final-fit optimisation: one Optuna study over the full available history."""
    pair_key     = asset['pair_key']
    symbol_y     = asset['symbol_y']
    symbol_x     = asset['symbol_x']
    strategy     = asset['strategy']
    lookback_raw = asset['lookback']
    param_defs   = asset['param_defs']
    fixed_params = asset['fixed_params']
    cost         = WF_CONFIG['cost']
    n_trials     = WF_CONFIG['n_trials']

    if strategy not in STRATEGY_REGISTRY:
        print(f"WARNING: strategy '{strategy}' not in registry — skipping {pair_key}")
        return None

    strategy_fn = STRATEGY_REGISTRY[strategy]

    print(f"\n{'='*62}")
    print(f"Optimising {pair_key}  ({symbol_y}/{symbol_x})")
    print(f"{'='*62}")

    client  = get_binance_client()
    cutoff  = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=24)

    def _fetch_strip(sym):
        raw = get_data(client, sym, interval='1d', lookback=lookback_raw)
        last_naive = (raw.index[-1].tz_convert('UTC').tz_localize(None)
                      if raw.index.tz is not None else raw.index[-1])
        return raw.iloc[:-1] if last_naive > cutoff else raw

    df_y = _fetch_strip(symbol_y)
    df_x = _fetch_strip(symbol_x)

    df_raw = _merge_pair(df_y, df_x)
    print(f"  Merged: {df_raw.index[0].date()} -> {df_raw.index[-1].date()}  ({len(df_raw)} bars)")

    df = df_raw.iloc[INDICATOR_WARMUP:].copy()
    print(f"  After warmup ({INDICATOR_WARMUP} bars): {len(df)} bars  "
          f"({df.index[0].date()} -> {df.index[-1].date()})")

    free_params = [k for k in param_defs if k not in fixed_params]
    print(f"  Free ({len(free_params)}): {free_params}")
    print(f"  Fixed ({len(fixed_params)}): {list(fixed_params.keys())}")

    _error_reported = [False]

    def objective(trial):
        params = {}
        for name, (dtype, lo, hi) in param_defs.items():
            if name in fixed_params:
                params[name] = fixed_params[name]
            elif dtype == 'int':
                params[name] = trial.suggest_int(name, lo, hi)
            else:
                params[name] = trial.suggest_float(name, lo, hi)

        try:
            result = strategy_fn(df.copy(), params)
        except Exception as e:
            if not _error_reported[0]:
                import traceback
                print(f'\n[optimise] strategy error: {type(e).__name__}: {e}')
                traceback.print_exc()
                _error_reported[0] = True
            trial.set_user_attr('error', str(e))
            return -999.0

        if result is None:
            return -999.0

        strategy_df, _ = result
        m = _run_backtest(strategy_df, cost, backtest_fn)
        if _reject(m):
            return -999.0

        trial.set_user_attr('sharpe',        m['sharpe_ratio'])
        trial.set_user_attr('total_return',  m['total_return'])
        trial.set_user_attr('max_drawdown',  m['max_drawdown'])
        trial.set_user_attr('num_trades',    m['num_trades'])
        trial.set_user_attr('win_rate',      m['win_rate'])
        trial.set_user_attr('profit_factor', m['profit_factor'])
        return _score(m)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction  = 'maximize',
        study_name = f'statarb_{pair_key}',
        sampler    = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params  = {**study.best_params, **fixed_params}
    final_result = strategy_fn(df.copy(), best_params)
    if final_result is None:
        print(f"WARNING: final backtest failed for {pair_key}")
        return None

    final_df, _ = final_result
    m = _run_backtest(final_df, cost, backtest_fn)
    if m is None:
        print(f"WARNING: backtest returned None for {pair_key}")
        return None

    print(f"\n{pair_key} | trials: {n_trials} | "
          f"Sharpe: {m['sharpe_ratio']:.2f} | "
          f"Return: {m['total_return']*100:.1f}% | "
          f"MaxDD: {m['max_drawdown']*100:.1f}%")

    return {
        'symbol_y':         symbol_y,
        'symbol_x':         symbol_x,
        'strategy':         strategy,
        'optimised_on':     str(date.today()),
        'data_from':        str(df.index[0].date()),
        'data_to':          str(df.index[-1].date()),
        'is_sharpe':        round(m['sharpe_ratio'], 4),
        'is_return':        round(m['total_return'],  4),
        'is_maxdd':         round(m['max_drawdown'],  4),
        'n_trials':         n_trials,
        'fixed_param_keys': sorted(fixed_params.keys()),
        'params':           best_params,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _print_verification_table():
    print(f"\n{'='*70}")
    print("PAIR CONFIG VERIFICATION")
    print(f"{'='*70}")
    print(f"{'Pair key':<12} {'Y / X':<28} {'Params':>10} {'Fixed':>8}  Fixed values")
    print('-' * 70)
    for asset in ASSET_CONFIG:
        fp     = asset['fixed_params']
        sample = ', '.join(f"{k}={v}" for k, v in list(fp.items())[:3])
        if len(fp) > 3:
            sample += f' (+{len(fp)-3} more)'
        print(f"{asset['pair_key']:<12} {asset['symbol_y']}/{asset['symbol_x']:<28}"
              f"{len(asset['param_defs']):>10} {len(fp):>8}  {sample}")
    print(f"\nStrategy registry: {list(STRATEGY_REGISTRY.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward optimisation for stat arb pairs. Writes live_params.json."
    )
    parser.add_argument('--pair',   metavar='KEY',
                        help="Single pair to optimise, e.g. FILSNX. Omit to run all.")
    parser.add_argument('--verify', action='store_true',
                        help="Print config table and exit without optimising.")
    args = parser.parse_args()

    if args.verify:
        _print_verification_table()
        return

    import optuna
    sys.path.insert(0, os.path.join(_ROOT, 'infrastructure', 'backtester'))
    from binance_client import get_binance_client, get_data
    from engine import backtest as backtest_fn
    from config import WF_CONFIG, INDICATOR_WARMUP

    if args.pair:
        target = args.pair.upper()
        assets_to_run = [a for a in ASSET_CONFIG if a['pair_key'] == target]
        if not assets_to_run:
            known = [a['pair_key'] for a in ASSET_CONFIG]
            print(f"ERROR: '{target}' not in ASSET_CONFIG. Known: {known}")
            sys.exit(1)
    else:
        assets_to_run = ASSET_CONFIG

    with open(LIVE_PARAMS_PATH) as f:
        live_params = json.load(f)

    for asset in assets_to_run:
        pair_key = asset['pair_key']
        entry    = optimise_pair(asset, get_binance_client, get_data, backtest_fn,
                                 optuna, WF_CONFIG, INDICATOR_WARMUP)
        if entry is None:
            continue

        live_params[pair_key] = entry

        with open(LIVE_PARAMS_PATH, 'w') as f:
            json.dump(live_params, f, indent=2)

        print(f"✓ {pair_key} written to live_params.json")

    print("\nDone.")


if __name__ == '__main__':
    main()
