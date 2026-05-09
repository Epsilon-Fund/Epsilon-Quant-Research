"""
optimise.py — Final-fit optimisation for BB Breakout assets.

Usage:
    python optimise.py              -> optimises all assets in ASSET_CONFIG
    python optimise.py --asset BTC  -> optimises BTCUSDT only
    python optimise.py --verify     -> prints config table and exits

Data source: live_trading/cache/hourly/ (updated automatically).
Run from the live_trading/ directory or anywhere — path setup handles it.

After running, add the coin to ACTIVE_ASSETS in config.py to activate it
in the dashboard.
"""

import argparse
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))           # dashboards/bbbreakout/
_LT_DIR        = os.path.abspath(os.path.join(_DASHBOARD_DIR, '..', '..'))  # live_trading/
_ROOT          = os.path.dirname(_LT_DIR)                              # repo root
_INFRA         = os.path.join(_ROOT, 'infrastructure', 'data')
for _p in (_DASHBOARD_DIR, _LT_DIR, _INFRA):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.append(os.path.join(_ROOT, 'infrastructure', 'backtester'))
# ─────────────────────────────────────────────────────────────────────────────

# ASSET_CONFIG and STRATEGY_REGISTRY both live in strategies.py so
# dashboard.py can import them without depending on optimise.py.
from strategies import STRATEGY_REGISTRY, ASSET_CONFIG  # noqa: E402

LIVE_PARAMS_PATH = os.path.join(_DASHBOARD_DIR, 'live_params.json')


# ══════════════════════════════════════════════════════════════════════════════
#  Scoring, rejection, and backtest helpers  (identical to momentum/optimise.py)
# ══════════════════════════════════════════════════════════════════════════════

def _score(metrics: dict) -> float:
    """
    Normalised composite: Sharpe 50% | Calmar 30% | Return 20%.

    Thresholds aligned with bb_cpcv per-asset notebooks (SCORE_FN) so the live
    final-fit Optuna run targets the same objective as the CPCV validation:
      SHARPE_MAX=3.5, CALMAR_MAX=16.0, RETURN_MAX=100.0
    """
    SHARPE_MAX = 3.5
    CALMAR_MAX = 16.0
    RETURN_MAX = 100.0

    calmar = (metrics['total_return'] / abs(metrics['max_drawdown'])
              if metrics['max_drawdown'] != 0 else 0.0)

    s = np.clip(metrics['sharpe_ratio']  / SHARPE_MAX, 0, 1)
    c = np.clip(calmar                   / CALMAR_MAX, 0, 1)
    r = np.clip(metrics['total_return']  / RETURN_MAX, 0, 1)

    return 0.50 * s + 0.30 * c + 0.20 * r


def _reject(metrics) -> bool:
    """
    Returns True if this trial should be discarded.

    Thresholds aligned with bb_cpcv per-asset notebooks (REJECT_FN).  The
    previous live thresholds (num_trades<15, win_rate<0.4, max_dd<-0.6) were
    stricter and rejected most leveraged BB-breakout trials whose IS metrics
    naturally land between the two ranges (e.g. 35-40% win-rate, -60% to
    -80% drawdown), forcing Optuna into worse local optima.
    """
    if metrics is None:                   return True
    if metrics['num_trades']    < 5:      return True
    if metrics['win_rate']      < 0.35:   return True
    if metrics['max_drawdown']  < -0.80:  return True
    if metrics['profit_factor'] < 0.5:    return True
    return False


def _run_backtest(strategy_df, cost: float, backtest_fn):
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

def optimise_asset(asset: dict, backtest_fn, optuna, WF_CONFIG, INDICATOR_WARMUP) -> dict:
    """
    Final-fit optimisation: one Optuna study over the full available history.

    Fetches hourly OHLCV from the local parquet cache (or Binance on cache miss).
    Strips the current incomplete 1H candle before optimising.
    """
    from shared.cache_manager import get_hourly_ohlcv

    symbol       = asset['symbol']
    strategy     = asset['strategy']
    lookback     = asset['lookback']    # days of hourly history to fetch
    param_defs   = asset['param_defs']
    fixed_params = asset['fixed_params']
    cost         = WF_CONFIG['cost']
    n_trials     = WF_CONFIG['n_trials']

    if strategy not in STRATEGY_REGISTRY:
        print(f"WARNING: strategy '{strategy}' not in registry — skipping {symbol}")
        return None

    strategy_fn = STRATEGY_REGISTRY[strategy]

    print(f"\n{'═'*60}")
    print(f"Optimising {symbol}  (strategy: {strategy})")
    print(f"{'═'*60}")

    # ── Fetch hourly data from cache ──────────────────────────────────────────
    end   = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=lookback + 10)
    df_raw = get_hourly_ohlcv(symbol, start, end)

    if df_raw.empty:
        print(f"ERROR: no hourly data for {symbol}. Run backfill_cache.py first.")
        return None

    # Strip incomplete bar: open timestamp < 1 h ago -> bar not yet closed
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=1)
    last_utc_naive = (
        df_raw.index[-1].tz_convert('UTC').tz_localize(None)
        if df_raw.index.tz is not None
        else df_raw.index[-1]
    )
    before = len(df_raw)
    if last_utc_naive > cutoff:
        df_raw = df_raw.iloc[:-1]

    print(f"  {symbol} raw bars: {before}  last: {df_raw.index[-1]}  "
          f"{'(stripped incomplete)' if len(df_raw) < before else ''}")

    # Pass full history — wf_engine does NOT strip burnin from the train window;
    # the strategy state-machine handles early NaN bars internally.
    df = df_raw.copy()

    print(f"  Using {len(df)} bars  "
          f"({df.index[0].date()} -> {df.index[-1].date()})")

    free_params = [k for k in param_defs if k not in fixed_params]
    print(f"  Free params  ({len(free_params)}): {free_params}")
    print(f"  Fixed params ({len(fixed_params)}): {list(fixed_params.keys())}")

    _error_reported = [False]

    def objective(trial):
        # Seed with all fixed_params first so keys absent from param_defs are
        # always present (e.g. risk_per_trade / max_leverage for BTC/ETH).
        params = dict(fixed_params)
        for name, (dtype, lo, hi) in param_defs.items():
            if name in fixed_params:
                params[name] = fixed_params[name]   # keep fixed value
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
        study_name = f'bb_final_fit_{symbol}',
        sampler    = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_free   = study.best_params
    best_params = {**best_free, **fixed_params}

    # Final evaluation on full history
    final_result = strategy_fn(df.copy(), best_params)
    if final_result is None:
        print(f"WARNING: final backtest failed for {symbol}")
        return None

    final_strat_df, _ = final_result
    m = _run_backtest(final_strat_df, cost, backtest_fn)
    if m is None:
        print(f"WARNING: backtest returned None for {symbol}")
        return None

    print(
        f"\n{symbol} | trials: {n_trials} | "
        f"Sharpe: {m['sharpe_ratio']:.2f} | "
        f"Return: {m['total_return']*100:.1f}% | "
        f"MaxDD: {m['max_drawdown']*100:.1f}%"
    )

    return {
        'strategy':         strategy,
        'optimised_on':     str(date.today()),
        'data_from':        str(df.index[0].date()),
        'data_to':          str(df.index[-1].date()),
        'is_sharpe':        round(m['sharpe_ratio'], 4),
        'is_return':        round(m['total_return'], 4),
        'is_maxdd':         round(m['max_drawdown'],  4),
        'n_trials':         n_trials,
        'fixed_param_keys': sorted(fixed_params.keys()),
        'params':           best_params,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _parse_symbol(raw: str) -> str:
    s = raw.upper()
    if not s.endswith('USDT'):
        s += 'USDT'
    return s


def _print_verification_table():
    """Print a per-coin config summary and confirm nothing is accidentally shared."""
    COL = {'symbol': 9, 'strategy': 17, 'n_params': 13, 'n_fixed': 12}
    header = (
        f"{'Symbol':<{COL['symbol']}} "
        f"{'Strategy':<{COL['strategy']}} "
        f"{'Param space':>{COL['n_params']}} "
        f"{'Fixed params':>{COL['n_fixed']}} "
        f"Fixed values (sample)"
    )
    sep = '-' * (sum(COL.values()) + len(COL) + 22)
    print(f"\n{'═'*len(sep)}")
    print("BB BREAKOUT ASSET CONFIG VERIFICATION")
    print(f"{'═'*len(sep)}")
    print(header)
    print(sep)

    for asset in ASSET_CONFIG:
        fp     = asset['fixed_params']
        sample = ', '.join(f"{k}={v}" for k, v in list(fp.items())[:3])
        if len(fp) > 3:
            sample += f'  (+{len(fp)-3} more)'
        print(
            f"{asset['symbol']:<{COL['symbol']}} "
            f"{asset['strategy']:<{COL['strategy']}} "
            f"{len(asset['param_defs']):>{COL['n_params']}} "
            f"{len(asset['fixed_params']):>{COL['n_fixed']}} "
            f"{sample}"
        )

    print(f"\nStrategy registry: {list(STRATEGY_REGISTRY.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description="BB Breakout final-fit optimisation. Writes results to live_params.json."
    )
    parser.add_argument(
        '--asset',
        metavar='SYMBOL',
        help="Single asset to optimise, e.g. BTC or BTCUSDT. Omit to run all.",
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help="Print the config verification table and exit without optimising.",
    )
    args = parser.parse_args()

    if args.verify:
        _print_verification_table()
        return

    # ── Deferred imports — only reached when actually optimising ─────────────
    import optuna
    _backtester_dir = os.path.join(_ROOT, 'infrastructure', 'backtester')
    if _backtester_dir not in sys.path:
        sys.path.insert(0, _backtester_dir)
    from engine import backtest as backtest_fn
    from config import WF_CONFIG, INDICATOR_WARMUP
    # ─────────────────────────────────────────────────────────────────────────

    if args.asset:
        target     = _parse_symbol(args.asset)
        candidates = [a for a in ASSET_CONFIG if a['symbol'] == target]
        if not candidates:
            known = [a['symbol'] for a in ASSET_CONFIG]
            print(f"ERROR: '{target}' not in ASSET_CONFIG. Known: {known}")
            sys.exit(1)
        assets_to_run = candidates
    else:
        assets_to_run = ASSET_CONFIG

    # Load existing live_params so other coins are not wiped
    if os.path.exists(LIVE_PARAMS_PATH):
        with open(LIVE_PARAMS_PATH) as f:
            raw = f.read().strip()
        live_params = json.loads(raw) if raw else {}
    else:
        live_params = {}

    for asset in assets_to_run:
        symbol = asset['symbol']
        entry  = optimise_asset(asset, backtest_fn, optuna, WF_CONFIG, INDICATOR_WARMUP)
        if entry is None:
            print(f"⚠  {symbol} skipped — optimisation returned None")
            continue

        live_params[symbol] = entry

        with open(LIVE_PARAMS_PATH, 'w') as f:
            json.dump(live_params, f, indent=2)

        print(f"✓  {symbol} written to live_params.json")

    print("\nDone. Add coins to ACTIVE_ASSETS in config.py to activate the dashboard.")


if __name__ == '__main__':
    main()
