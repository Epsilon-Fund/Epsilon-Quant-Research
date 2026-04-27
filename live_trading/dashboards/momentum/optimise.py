"""
optimise.py — Walk-forward optimisation for one or all assets.

Usage:
    python optimise.py              -> runs all assets in ASSET_CONFIG
    python optimise.py --asset BTC  -> runs BTCUSDT only
    python optimise.py --asset ETH  -> runs ETHUSDT only

After running, add the coin to ACTIVE_ASSETS in config.py to activate it
in the dashboard.
"""

import argparse
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))          # dashboards/momentum/
_LT_DIR        = os.path.abspath(os.path.join(_DASHBOARD_DIR, '..', '..'))  # live_trading/
_ROOT          = os.path.dirname(_LT_DIR)                             # repo root
_INFRA         = os.path.join(_ROOT, 'infrastructure', 'data')
for _p in (_DASHBOARD_DIR, _LT_DIR, _INFRA):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.append(os.path.join(_ROOT, 'infrastructure', 'walkforward'))
# ─────────────────────────────────────────────────────────────────────────────

# Strategy functions live in strategies.py so dashboard.py can import them too
# without depending on optimise.py.
from strategies import STRATEGY_REGISTRY  # noqa: E402

# NOTE: binance_client and WF_CONFIG are imported lazily inside main()
# (after --verify check) so that `--verify` exits without touching them.

LIVE_PARAMS_PATH = os.path.join(_DASHBOARD_DIR, 'live_params.json')


# ══════════════════════════════════════════════════════════════════════════════
#  Asset config — single source of truth per coin.
#  param_defs and fixed_params sourced directly from each coin's notebook.
#
#  To add a new coin:
#    1. Add an entry here (read its notebook for param_defs / fixed_params)
#    2. python optimise.py --asset <SYMBOL>
#    3. Add it to ACTIVE_ASSETS in config.py
# ══════════════════════════════════════════════════════════════════════════════

ASSET_CONFIG = [
    # ── ETHUSDT  (momentum_swing) ─────────────────────────────────────────────
    # Source: momentumETH_wf.ipynb
    {
        "symbol":   "ETHUSDT",
        "strategy": "momentum_swing",
        "lookback": 2150,
        "param_defs": {
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
        },
        "fixed_params": {
            'risk_per_trade':        0.045,
            'max_leverage':          2.6,
            'ema_span':              28,
            'stop_mult_pos_caution': 0.5,
            'obv_lookback':          28,
            'obv_ma_period':         20,
            'swing_caution':         5,
        },
    },

    # ── SOLUSDT  (momentum_swing) ─────────────────────────────────────────────
    # Source: momentumSOL_wf.ipynb
    {
        "symbol":   "SOLUSDT",
        "strategy": "momentum_swing",
        "lookback": 2150,
        "param_defs": {
            'ema_span':              ('int',   23,   40),
            'swing_caution':         ('int',   3,    14),
            'swing_stop':            ('int',   3,    10),
            'atr_caution':           ('int',   26,   35),
            'atr_stop':              ('int',   10,   30),
            'atr_size':              ('int',   3,    14),
            'adx_override':          ('int',   40,   80),
            'stop_atr_scale':        ('float', 1,    1.3),
            'risk_per_trade':        ('float', 0.005, 0.05),
            'max_leverage':          ('float', 1.0,  3.0),
            'stop_mult_pos_caution': ('float', 0.1,  0.9),
            'stop_mult_pos_normal':  ('float', 1,    1.5),
            'stop_mult_ent_both':    ('float', 1,    2.5),
            'stop_mult_ent_caution': ('float', 0.1,  0.9),
            'stop_mult_ent_normal':  ('float', 1,    1.5),
            'vol_ma_period':         ('int',   19,   40),
            'obv_ma_period':         ('int',   29,   34),
            'obv_lookback':          ('int',   10,   30),
        },
        "fixed_params": {
            'risk_per_trade': 0.046,
            'max_leverage':   3,
            'stop_atr_scale': 1.261,
        },
    },

    # ── BNBUSDT  (momentum_swing) ─────────────────────────────────────────────
    # Source: momentumBNB_wf.ipynb
    {
        "symbol":   "BNBUSDT",
        "strategy": "momentum_swing",
        "lookback": 2150,
        "param_defs": {
            'ema_span':              ('int',   5,    40),
            'swing_caution':         ('int',   3,    14),
            'swing_stop':            ('int',   3,    10),
            'atr_caution':           ('int',   10,   30),
            'atr_stop':              ('int',   10,   30),
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
            'vol_ma_period':         ('int',   10,   40),
            'obv_ma_period':         ('int',   10,   40),
            'obv_lookback':          ('int',   10,   30),
        },
        "fixed_params": {
            'risk_per_trade': 0.0472,
            'max_leverage':   2.5607,
        },
    },

    # ── ADAUSDT  (momentum_swing) ─────────────────────────────────────────────
    # Source: momentumADA_wf.ipynb
    {
        "symbol":   "ADAUSDT",
        "strategy": "momentum_swing",
        "lookback": 2150,
        "param_defs": {
            'ema_span':              ('int',   16,   40),
            'swing_caution':         ('int',   3,    14),
            'swing_stop':            ('int',   3,    10),
            'atr_caution':           ('int',   10,   30),
            'atr_stop':              ('int',   10,   22),
            'atr_size':              ('int',   3,    14),
            'adx_override':          ('int',   40,   80),
            'stop_atr_scale':        ('float', 0.5,  2.0),
            'risk_per_trade':        ('float', 0.005, 0.05),
            'max_leverage':          ('float', 1.0,  3.0),
            'stop_mult_pos_caution': ('float', 0.4,  0.9),
            'stop_mult_pos_normal':  ('float', 0.8,  2.0),
            'stop_mult_ent_both':    ('float', 0.5,  2.5),
            'stop_mult_ent_caution': ('float', 0.1,  0.9),
            'stop_mult_ent_normal':  ('float', 0.5,  1.5),
            'vol_ma_period':         ('int',   10,   40),
            'obv_ma_period':         ('int',   10,   40),
            'obv_lookback':          ('int',   10,   30),
        },
        "fixed_params": {
            'risk_per_trade':        0.0457,
            'max_leverage':          2.5,
            'atr_caution':           22,
            'stop_atr_scale':        1.8709,
            'stop_mult_pos_caution': 0.6945,
            'ema_span':              32,
            'adx_override':          50,
        },
    },

    # ── XRPUSDT  (momentum_swing) ─────────────────────────────────────────────
    # Source: momentumXRP_wf.ipynb
    {
        "symbol":   "XRPUSDT",
        "strategy": "momentum_swing",
        "lookback": 2150,
        "param_defs": {
            'ema_span':              ('int',   5,    40),
            'swing_caution':         ('int',   3,    14),
            'swing_stop':            ('int',   3,    10),
            'atr_caution':           ('int',   10,   30),
            'atr_stop':              ('int',   10,   30),
            'atr_size':              ('int',   3,    14),
            'adx_override':          ('int',   40,   80),
            'stop_atr_scale':        ('float', 0.5,  2.0),
            'risk_per_trade':        ('float', 0.005, 0.05),
            'max_leverage':          ('float', 1.0,  3.0),
            'stop_mult_pos_caution': ('float', 0.1,  0.9),
            'stop_mult_pos_normal':  ('float', 1.5,  2.0),
            'stop_mult_ent_both':    ('float', 0.5,  2.5),
            'stop_mult_ent_caution': ('float', 0.1,  0.9),
            'stop_mult_ent_normal':  ('float', 0.5,  1.5),
            'vol_ma_period':         ('int',   21,   40),
            'obv_ma_period':         ('int',   10,   40),
            'obv_lookback':          ('int',   10,   30),
        },
        "fixed_params": {
            'risk_per_trade': 0.046,
            'max_leverage':   2.5,
            'atr_caution':    12,
            'swing_caution':  14,
            'adx_override':   44,
            'atr_size':       13,
        },
    },

    # ── AVAXUSDT  (momentum_no_vol) ───────────────────────────────────────────
    # Source: momentumAVAX_wf.ipynb  (vol_ma lines commented out)
    {
        "symbol":   "AVAXUSDT",
        "strategy": "momentum_no_vol",
        "lookback": 2150,
        "param_defs": {
            'ema_span':              ('int',   5,    40),
            'swing_caution':         ('int',   3,    14),
            'swing_stop':            ('int',   3,    10),
            'atr_caution':           ('int',   10,   30),
            'atr_stop':              ('int',   10,   30),
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
            'obv_ma_period':         ('int',   10,   40),
            'obv_lookback':          ('int',   10,   30),
        },
        "fixed_params": {
            'risk_per_trade':        0.048,
            'max_leverage':          3,
            'stop_atr_scale':        1.9546,
            'ema_span':              36,
            'stop_mult_pos_caution': 0.8571,
            'adx_override':          65,
            'swing_caution':         13,
            'swing_stop':            9,
        },
    },

    # ── BTCUSDT  (momentum_no_vol) ────────────────────────────────────────────
    # Source: momentumBTC_wf.ipynb
    {
        "symbol":   "BTCUSDT",
        "strategy": "momentum_no_vol",
        "lookback": 2150,
        "param_defs": {
            'ema_span':              ('int',   5,    40),
            'swing_caution':         ('int',   3,    14),
            'swing_stop':            ('int',   3,    10),
            'atr_caution':           ('int',   10,   30),
            'atr_stop':              ('int',   10,   30),
            'atr_size':              ('int',   3,    14),
            'adx_override':          ('int',   52,   65),
            'stop_atr_scale':        ('float', 0.5,  2.0),
            'risk_per_trade':        ('float', 0.005, 0.05),
            'max_leverage':          ('float', 1.0,  3.0),
            'stop_mult_pos_caution': ('float', 0.1,  0.6),
            'stop_mult_pos_normal':  ('float', 0.8,  2.0),
            'stop_mult_ent_both':    ('float', 1.0,  2.5),
            'stop_mult_ent_caution': ('float', 0.1,  0.9),
            'stop_mult_ent_normal':  ('float', 0.5,  1.5),
            'obv_ma_period':         ('int',   10,   40),
            'obv_lookback':          ('int',   10,   30),
        },
        "fixed_params": {
            'risk_per_trade':       0.0426,
            'max_leverage':         2.8325,
            'stop_atr_scale':       1,
            'stop_mult_pos_normal': 1,
            'stop_mult_ent_normal': 1,
        },
    },
]


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

    s = np.clip(metrics['sharpe_ratio']  / SHARPE_MAX, 0, 1)
    c = np.clip(calmar                   / CALMAR_MAX, 0, 1)
    r = np.clip(metrics['total_return']  / RETURN_MAX, 0, 1)

    return 0.50 * s + 0.30 * c + 0.20 * r


def _reject(metrics):
    """Returns True if this trial should be discarded."""
    if metrics is None:                  return True
    if metrics['num_trades']    < 7:     return True
    if metrics['win_rate']      < 0.35:  return True
    if metrics['max_drawdown']  < -0.80: return True
    if metrics['profit_factor'] < 0.8:   return True
    return False


def _run_backtest(strategy_df, cost, backtest_fn):
    try:
        return backtest_fn(
            data           = strategy_df,
            cost           = cost,
            show_plot      = False,
            save_html      = None,
            show_trades    = False,
            benchmark_data = None,
        )
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Core optimisation logic
# ══════════════════════════════════════════════════════════════════════════════

def optimise_asset(asset: dict, get_binance_client, get_data, backtest_fn,
                   optuna, WF_CONFIG, INDICATOR_WARMUP) -> dict:
    """
    Final-fit optimisation: one Optuna study over the full available history.
    """
    symbol       = asset["symbol"]
    strategy     = asset["strategy"]
    lookback     = asset["lookback"]
    param_defs   = asset["param_defs"]
    fixed_params = asset["fixed_params"]
    cost         = WF_CONFIG["cost"]
    n_trials     = WF_CONFIG["n_trials"]

    if strategy not in STRATEGY_REGISTRY:
        print(f"WARNING: strategy '{strategy}' not in registry — skipping {symbol}")
        return None

    strategy_fn = STRATEGY_REGISTRY[strategy]

    print(f"\n{'═'*60}")
    print(f"Optimising {symbol}  (strategy: {strategy})")
    print(f"{'═'*60}")

    client = get_binance_client()
    df_raw = get_data(client, symbol, interval='1d', lookback=lookback)

    last_raw = df_raw.index[-1]
    print(f"  [optimise] {symbol}  raw last bar: {last_raw}  (index.tz={df_raw.index.tz})  "
          f"len={len(df_raw)}")

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=24)
    last_utc_naive = (
        df_raw.index[-1].tz_convert('UTC').tz_localize(None)
        if df_raw.index.tz is not None
        else df_raw.index[-1]
    )
    before = len(df_raw)
    if last_utc_naive > cutoff:
        df_raw = df_raw.iloc[:-1]
    print(f"  [optimise] {symbol}  cutoff={cutoff}  last_utc_naive={last_utc_naive}  "
          f"len {before} -> {len(df_raw)}  {'(stripped)' if len(df_raw) < before else '(no strip)'}")

    df = df_raw.iloc[INDICATOR_WARMUP:].copy()

    print(f"Raw bars: {len(df_raw)}  ({df_raw.index[0].date()} -> {df_raw.index[-1].date()})")
    print(f"After warmup drop ({INDICATOR_WARMUP} bars): {len(df)} bars  "
          f"({df.index[0].date()} -> {df.index[-1].date()})")

    free_params = [k for k in param_defs if k not in fixed_params]
    print(f"Free params ({len(free_params)}): {free_params}")
    print(f"Fixed params ({len(fixed_params)}): {list(fixed_params.keys())}")

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
                print(f'\n[optimise] strategy_fn error (first occurrence): '
                      f'{type(e).__name__}: {e}')
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
        study_name = f'final_fit_{symbol}',
        sampler    = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_free   = study.best_params
    best_params = {**best_free, **fixed_params}

    final_result = strategy_fn(df.copy(), best_params)
    if final_result is None:
        print(f"WARNING: final backtest failed for {symbol}")
        return None

    final_strat_df, _ = final_result
    m = _run_backtest(final_strat_df, cost, backtest_fn)
    if m is None:
        print(f"WARNING: backtest returned None for {symbol}")
        return None

    is_sharpe = m['sharpe_ratio']
    is_return = m['total_return']
    is_maxdd  = m['max_drawdown']

    print(
        f"\n{symbol} | trials: {n_trials} | "
        f"Sharpe: {is_sharpe:.2f} | "
        f"Return: {is_return*100:.1f}% | "
        f"MaxDD: {is_maxdd*100:.1f}%"
    )

    return {
        "strategy":        strategy,
        "optimised_on":    str(date.today()),
        "data_from":       str(df.index[0].date()),
        "data_to":         str(df.index[-1].date()),
        "is_sharpe":       round(is_sharpe, 4),
        "is_return":       round(is_return, 4),
        "is_maxdd":        round(is_maxdd,  4),
        "n_trials":        n_trials,
        "fixed_param_keys": sorted(fixed_params.keys()),
        "params":          best_params,
    }


def _parse_symbol(raw: str) -> str:
    s = raw.upper()
    if not s.endswith("USDT"):
        s += "USDT"
    return s


def _print_verification_table():
    """Print a per-coin config summary and confirm nothing is accidentally shared."""
    COL = {
        'symbol':   9,
        'strategy': 17,
        'n_params': 13,
        'n_fixed':  12,
        'unique':   40,
    }
    header = (
        f"{'Symbol':<{COL['symbol']}} "
        f"{'Strategy':<{COL['strategy']}} "
        f"{'Param space':>{COL['n_params']}} "
        f"{'Fixed params':>{COL['n_fixed']}} "
        f"{'Fixed values (sample)'}"
    )
    sep = '-' * (sum(COL.values()) + len(COL))
    print(f"\n{'═'*len(sep)}")
    print("ASSET CONFIG VERIFICATION")
    print(f"{'═'*len(sep)}")
    print(header)
    print(sep)

    for asset in ASSET_CONFIG:
        fp = asset["fixed_params"]
        sample = ', '.join(f"{k}={v}" for k, v in list(fp.items())[:3])
        if len(fp) > 3:
            sample += f', (+{len(fp)-3} more)'
        print(
            f"{asset['symbol']:<{COL['symbol']}} "
            f"{asset['strategy']:<{COL['strategy']}} "
            f"{len(asset['param_defs']):>{COL['n_params']}} "
            f"{len(asset['fixed_params']):>{COL['n_fixed']}} "
            f"{sample}"
        )

    print(f"\nStrategy registry contains: {list(STRATEGY_REGISTRY.keys())}")
    print("No param_defs or fixed_params in registry — all config lives in ASSET_CONFIG.")


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward optimisation. Writes results to live_params.json."
    )
    parser.add_argument(
        "--asset",
        metavar="SYMBOL",
        help="Single asset to optimise, e.g. BTC or BTCUSDT. Omit to run all.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Print the config verification table and exit without optimising.",
    )
    args = parser.parse_args()

    if args.verify:
        _print_verification_table()
        return

    # ── Deferred imports — only reached when actually optimising ─────────────
    import optuna
    import sys as _sys
    _sys.path.insert(0, os.path.join(_ROOT, 'infrastructure', 'backtester'))
    from binance_client import get_binance_client, get_data
    from engine import backtest as backtest_fn
    from config import WF_CONFIG, INDICATOR_WARMUP
    # ─────────────────────────────────────────────────────────────────────────

    if args.asset:
        target     = _parse_symbol(args.asset)
        candidates = [a for a in ASSET_CONFIG if a["symbol"] == target]
        if not candidates:
            known = [a["symbol"] for a in ASSET_CONFIG]
            print(f"ERROR: '{target}' not in ASSET_CONFIG. Known: {known}")
            sys.exit(1)
        assets_to_run = candidates
    else:
        assets_to_run = ASSET_CONFIG

    with open(LIVE_PARAMS_PATH) as f:
        live_params = json.load(f)

    for asset in assets_to_run:
        symbol = asset["symbol"]
        entry  = optimise_asset(asset, get_binance_client, get_data, backtest_fn,
                               optuna, WF_CONFIG, INDICATOR_WARMUP)
        if entry is None:
            continue

        live_params[symbol] = entry

        with open(LIVE_PARAMS_PATH, "w") as f:
            json.dump(live_params, f, indent=2)

        print(f"✓ {symbol} written to live_params.json")

    print("\nDone. Run dashboard.py to verify data fetch.")


if __name__ == "__main__":
    main()
