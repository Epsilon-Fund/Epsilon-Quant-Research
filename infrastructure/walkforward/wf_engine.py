import os
import sys
import warnings
import numpy as np
import pandas as pd
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── resolve backtester path the same way engine.py resolves performance_metrics ──
_BACKTESTER_DIR = os.path.join(os.path.dirname(__file__), '..', 'backtester')
sys.path.insert(0, os.path.abspath(_BACKTESTER_DIR))
from engine import backtest


# ──────────────────────────────────────────────────────────────────────────────
#  Scoring & rejection defaults
# ──────────────────────────────────────────────────────────────────────────────

def _default_score(metrics):
    """
    Normalised composite: Sharpe 50% | Calmar 30% | Return 20%
    All components clipped to [0, 1] before weighting.
    Tune *_MAX caps to fit your strategy's realistic range.
    """
    SHARPE_MAX = 2.5
    CALMAR_MAX = 70.0
    RETURN_MAX = 15.0

    calmar = (metrics['total_return'] / abs(metrics['max_drawdown'])
              if metrics['max_drawdown'] != 0 else 0.0)

    s = np.clip(metrics['sharpe_ratio']   / SHARPE_MAX, 0, 1)
    c = np.clip(calmar                    / CALMAR_MAX, 0, 1)
    r = np.clip(metrics['total_return']   / RETURN_MAX, 0, 1)

    return 0.50 * s + 0.30 * c + 0.20 * r


def _default_reject(metrics):
    """Returns True if this trial should be discarded (score → -999)."""
    if metrics is None:              return True
    if metrics['num_trades']   < 10: return True
    if metrics['win_rate']     < 0.35: return True
    if metrics['max_drawdown'] < -0.80: return True
    if metrics['profit_factor'] < 0.8:  return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run_backtest(strategy_df, cost):
    try:
        return backtest(
            data           = strategy_df,
            cost           = cost,
            show_plot      = False,
            save_html      = None,
            show_trades    = False,
            benchmark_data = None,
        )
    except Exception:
        return None


def _calmar(metrics):
    if metrics is None or metrics['max_drawdown'] == 0:
        return 0.0
    return metrics['total_return'] / abs(metrics['max_drawdown'])


def _make_objective(df_train, strategy_fn, param_defs, fixed_params,
                    cost, score_fn, reject_fn):
    """Factory — returns an Optuna objective bound to one training slice."""

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
            result = strategy_fn(df_train.copy(), params)
        except Exception:
            return -999.0

        if result is None:
            return -999.0
            
        strategy_df, _ = result

        m = _run_backtest(strategy_df, cost)

        if reject_fn(m):
            return -999.0

        # store useful attrs for post-analysis
        trial.set_user_attr('sharpe',        m['sharpe_ratio'])
        trial.set_user_attr('calmar',        _calmar(m))
        trial.set_user_attr('total_return',  m['total_return'])
        trial.set_user_attr('max_drawdown',  m['max_drawdown'])
        trial.set_user_attr('num_trades',    m['num_trades'])
        trial.set_user_attr('win_rate',      m['win_rate'])
        trial.set_user_attr('profit_factor', m['profit_factor'])

        return score_fn(m)

    return objective


def _metrics_to_row(m, prefix):
    """Flatten a metrics dict into prefixed keys for the results dataframe."""
    if m is None:
        keys = ['return', 'sharpe', 'drawdown', 'calmar',
                'trades', 'winrate', 'profit_factor']
        return {f'{prefix}_{k}': None for k in keys}
    return {
        f'{prefix}_return':        m['total_return'],
        f'{prefix}_sharpe':        m['sharpe_ratio'],
        f'{prefix}_drawdown':      m['max_drawdown'],
        f'{prefix}_calmar':        _calmar(m),
        f'{prefix}_trades':        m['num_trades'],
        f'{prefix}_winrate':       m['win_rate'],
        f'{prefix}_profit_factor': m['profit_factor'],
    }


def _fmt(val, pct=False, dp=2):
    if val is None:
        return 'N/A'
    if pct:
        return f'{val*100:.{dp}f}%'
    return f'{val:.{dp}f}'


# ──────────────────────────────────────────────────────────────────────────────
#  Parameter stability
# ──────────────────────────────────────────────────────────────────────────────

def parameter_stability_table(all_best_params, param_defs, fixed_params):
    """
    Print stability stats for each parameter across folds.
    CV < 0.15 → stable candidate for fixing in future runs.

    Returns a tidy DataFrame.
    """
    int_params = {k for k, (t, _, _) in param_defs.items() if t == 'int'}
    rows = []

    print(f"\n{'─'*70}")
    print('PARAMETER STABILITY ACROSS FOLDS')
    print(f"{'─'*70}")
    print(f"{'Parameter':<30} {'Median':>10} {'Std':>10} {'CV':>8}  Note")
    print(f"{'─'*30} {'─'*10} {'─'*10} {'─'*8}  {'─'*20}")

    for name in param_defs:
        vals   = [p[name] for p in all_best_params]
        median = float(np.median(vals))
        std    = float(np.std(vals))
        cv     = std / abs(median) if median != 0 else 999.0

        note = ''
        if name in fixed_params:
            note = '[fixed]'
        elif cv < 0.15:
            note = '← stable'

        print(f'{name:<30} {median:>10.3f} {std:>10.3f} {cv:>8.3f}  {note}')
        rows.append({'param': name, 'median': median, 'std': std,
                     'cv': cv, 'fixed': name in fixed_params, 'stable': cv < 0.15})

    return pd.DataFrame(rows)


def consensus_params(all_best_params, param_defs):
    """Median parameter values across all folds — most robust for production."""
    int_params = {k for k, (t, _, _) in param_defs.items() if t == 'int'}
    result = {}
    for name in param_defs:
        vals = [p[name] for p in all_best_params]
        med  = np.median(vals)
        result[name] = int(round(med)) if name in int_params else round(float(med), 4)
    return result

def _build_stability_df(all_best_params, param_defs, fixed_params):
    int_params = {k for k, (t, _, _) in param_defs.items() if t == 'int'}
    rows = []
    for name in param_defs:
        vals   = [p[name] for p in all_best_params]
        median = float(np.median(vals))
        std    = float(np.std(vals))
        cv     = std / abs(median) if median != 0 else 999.0
        rows.append({'param': name, 'median': median, 'std': std,
                     'cv': cv, 'fixed': name in fixed_params, 'stable': cv < 0.15})
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
#  Main walk-forward function
# ──────────────────────────────────────────────────────────────────────────────

def walk_forward(
    df,
    strategy_fn,
    param_defs,
    fixed_params  = None,
    train_bars    = 730,
    test_bars     = 365,
    burnin_bars   = 60,
    n_trials      = 400,
    cost          = 0.001,
    score_fn      = None,
    reject_fn     = None,
    seed_base     = 42,
    save_csv      = None,
):
    """
    Rolling walk-forward optimisation with Optuna.

    Parameters
    ----------
    df           : full OHLCV DataFrame with DatetimeIndex
    strategy_fn  : fn(df_slice, params) → strategy_df
    param_defs   : {name: ('int'|'float', lo, hi)}
    fixed_params : {name: value}  anchored across all folds
    train_bars   : training window length in bars
    test_bars    : test window length in bars
    burnin_bars  : bars prepended to test slice for indicator warmup (trimmed before evaluation)
    n_trials     : Optuna trials per fold
    cost         : round-trip trading cost fraction
    score_fn     : fn(metrics) → float  (default: Sharpe/Calmar/Return composite)
    reject_fn    : fn(metrics) → bool   (default: min trade / drawdown filters)
    seed_base    : fold i uses seed_base + i
    save_csv     : optional path to save fold results CSV

    Returns
    -------
    dict
        fold_records      list[dict]      — per-fold metrics + params
        results_df        pd.DataFrame    — fold records as dataframe
        all_best_params   list[dict]      — best params per fold
        consensus_params  dict            — median params across folds
        stability_df      pd.DataFrame    — parameter stability table
        oos_combined_df   pd.DataFrame    — stitched OOS strategy dataframe
        oos_metrics       dict            — engine.backtest metrics on combined OOS
    """

    if fixed_params is None:
        fixed_params = {}
    if score_fn is None:
        score_fn = _default_score
    if reject_fn is None:
        reject_fn = _default_reject

    # ── build folds ────────────────────────────────────────────────────────────
    folds = []
    start = 0
    while start + train_bars + test_bars <= len(df):
        folds.append({
            'train':       df.iloc[start : start + train_bars].copy(),
            'test_burnin': df.iloc[start + train_bars - burnin_bars
                                   : start + train_bars + test_bars].copy(),
            'trim_at':     burnin_bars,
            'train_start': df.index[start],
            'train_end':   df.index[start + train_bars - 1],
            'test_start':  df.index[start + train_bars],
            'test_end':    df.index[start + train_bars + test_bars - 1],
        })
        start += test_bars

    print(f'Walk-forward: {len(folds)} fold(s)  '
          f'train={train_bars}  test={test_bars}  burnin={burnin_bars}  '
          f'trials={n_trials}')
    for i, f in enumerate(folds):
        print(f'  Fold {i+1}: train {f["train_start"].date()} → {f["train_end"].date()}  '
              f'| test {f["test_start"].date()} → {f["test_end"].date()}')

    free = [k for k in param_defs if k not in fixed_params]
    if fixed_params:
        print(f'\nFixed ({len(fixed_params)}): {list(fixed_params.keys())}')
        print(f'Free  ({len(free)}): {free}')

    # ── run folds ──────────────────────────────────────────────────────────────
    fold_records    = []
    all_best_params = []
    oos_slices      = []

    for i, fold in enumerate(folds):
        print(f"\n{'─'*60}")
        print(f'Fold {i+1}/{len(folds)}  '
              f'train: {fold["train_start"].date()} → {fold["train_end"].date()}  '
              f'test: {fold["test_start"].date()} → {fold["test_end"].date()}')

        # optimise on training window
        study = optuna.create_study(
            direction  = 'maximize',
            study_name = f'wf_fold_{i+1}',
            sampler    = optuna.samplers.TPESampler(seed=seed_base + i),
        )
        study.optimize(
            _make_objective(fold['train'], strategy_fn, param_defs,
                            fixed_params, cost, score_fn, reject_fn),
            n_trials          = n_trials,
            show_progress_bar = True,
        )

        best_params = {**fixed_params, **study.best_params}
        all_best_params.append(best_params)

        # IS performance
        try:
            train_strat, _ = strategy_fn(fold['train'].copy(), best_params)
            train_m     = _run_backtest(train_strat, cost)
        except Exception:
            train_m = None

        # OOS performance (burn-in prepended, then trimmed)
        test_m  = None
        oos_df  = None
        try:
            test_strat_full, indicator_cols  = strategy_fn(fold['test_burnin'].copy(), best_params)
            if test_strat_full is not None:
                oos_df = test_strat_full.iloc[fold['trim_at']:].copy()
                existing_cols = [c for c in indicator_cols if c in oos_df.columns]
                if existing_cols:
                   oos_df.dropna(subset=existing_cols, inplace=True)

                # find first real entry within OOS window and zero everything before it
                first_entry = oos_df[oos_df['position'].diff() != 0].index
                if len(first_entry) == 0:
                # no entries at all — zero entire slice
                    oos_df['position']      = 0
                    oos_df['position_size'] = 0.0
                    oos_df['stop_loss']     = 0.0
                else:
                # zero everything before first genuine OOS entry
                    before_entry = oos_df.index < first_entry[0]
                    oos_df.loc[before_entry, 'position']      = 0
                    oos_df.loc[before_entry, 'position_size'] = 0.0
                    oos_df.loc[before_entry, 'stop_loss']     = 0.0

                test_m = _run_backtest(oos_df, cost)
                if oos_df is not None and len(oos_df) > 0:
                    oos_slices.append(oos_df)
        except Exception:
            pass

        record = {
            'fold':         i + 1,
            'train_start':  str(fold['train_start'].date()),
            'train_end':    str(fold['train_end'].date()),
            'test_start':   str(fold['test_start'].date()),
            'test_end':     str(fold['test_end'].date()),
            'optuna_score': study.best_value,
            **_metrics_to_row(train_m, 'train'),
            **_metrics_to_row(test_m,  'test'),
            **{f'param_{k}': v for k, v in best_params.items()},
        }
        fold_records.append(record)

        print(f'\n  IS  → Sharpe: {_fmt(record["train_sharpe"])}  '
              f'Return: {_fmt(record["train_return"], pct=True)}  '
              f'DD: {_fmt(record["train_drawdown"], pct=True)}  '
              f'Calmar: {_fmt(record["train_calmar"])}  '
              f'Trades: {record["train_trades"]}')
        print(f'  OOS → Sharpe: {_fmt(record["test_sharpe"])}  '
              f'Return: {_fmt(record["test_return"], pct=True)}  '
              f'DD: {_fmt(record["test_drawdown"], pct=True)}  '
              f'Calmar: {_fmt(record["test_calmar"])}  '
              f'Trades: {record["test_trades"]}')
        print(f'\n  Best params: {best_params}')

    # ── summary ────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(fold_records)

    if save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        results_df.to_csv(save_csv, index=False)
        print(f'\n✓ Fold results saved → {save_csv}')

    print(f"\n{'═'*60}")
    print('WALK-FORWARD SUMMARY')
    print(f"{'═'*60}")

    valid = results_df[results_df['test_return'].notna() & results_df['train_return'].notna()]

    if len(valid) == 0:
        print('WARNING: no valid folds — loosen reject_fn filters or check strategy_fn output')
    else:
        print(f'\nOut-of-sample across {len(valid)} fold(s):')
        print(f'  Avg Sharpe:       {valid["test_sharpe"].mean():.2f}')
        print(f'  Avg Return:       {valid["test_return"].mean()*100:.1f}%')
        print(f'  Avg Max Drawdown: {valid["test_drawdown"].mean()*100:.1f}%')
        print(f'  Avg Calmar:       {valid["test_calmar"].mean():.2f}')
        print(f'  Avg Trades/fold:  {valid["test_trades"].mean():.0f}')
        print(f'  Folds profitable: {(valid["test_return"] > 0).sum()}/{len(valid)}')

        if valid['train_sharpe'].mean() != 0:
            deg = valid['test_sharpe'].mean() / valid['train_sharpe'].mean()
            label = 'good' if deg > 0.70 else ('acceptable' if deg > 0.50 else 'poor')
            print(f'  Sharpe OOS/IS:    {deg:.2f}  ({label})')

    # ── consensus params ───────────────────────────────────────────────────────
    cp = consensus_params(all_best_params, param_defs)

    # ── stability table ────────────────────────────────────────────────────────
    stability_df = _build_stability_df(all_best_params, param_defs, fixed_params)

    # ── combined OOS backtest ──────────────────────────────────────────────────
    oos_metrics  = None
    oos_combined = None

    if oos_slices:
        oos_combined = pd.concat(oos_slices)
        oos_combined = oos_combined[~oos_combined.index.duplicated(keep='first')].sort_index()

        print(f"\n{'─'*60}")
        print(f'COMBINED OOS: {oos_combined.index[0].date()} → '
              f'{oos_combined.index[-1].date()}  ({len(oos_combined)} bars)')

        oos_metrics = _run_backtest(oos_combined, cost)

        if oos_metrics:
            print(f'  Return:        {oos_metrics["total_return"]*100:.2f}%')
            print(f'  Sharpe:        {oos_metrics["sharpe_ratio"]:.2f}')
            print(f'  Max Drawdown:  {oos_metrics["max_drawdown"]*100:.2f}%')
            print(f'  Calmar:        {_calmar(oos_metrics):.2f}')
            print(f'  Profit Factor: {oos_metrics["profit_factor"]:.2f}')
            print(f'  Win Rate:      {oos_metrics["win_rate"]*100:.2f}%')
            print(f'  Num Trades:    {oos_metrics["num_trades"]}')

    return {
        'fold_records':     fold_records,
        'results_df':       results_df,
        'all_best_params':  all_best_params,
        'consensus_params': cp,
        'stability_df':     stability_df,
        'oos_combined_df':  oos_combined,
        'oos_metrics':      oos_metrics,
    }

# ──────────────────────────────────────────────────────────────────────────────
#  Plateau Analysis — 1-D sensitivity sweeps
# ──────────────────────────────────────────────────────────────────────────────

def plateau_analysis(
    df,
    strategy_fn,
    base_params,
    param_defs,
    fixed_params  = None,
    cost          = 0.001,
    score_fn      = None,
    reject_fn     = None,
    n_steps       = 20,
):
    """
    For each free parameter, sweep it across its full range while holding all
    others at base_params values.  Returns a dict of DataFrames (one per param)
    with columns [value, score, sharpe, calmar, return, drawdown, trades].

    Use the output to plot 1-D sensitivity curves:  broad flat curves → robust;
    narrow spikes → overfitting to a specific value.

    Parameters
    ----------
    df          : training-window DataFrame (or full dataset for a global view)
    strategy_fn : same fn(df, params) → (strategy_df, indicator_cols) used in WF
    base_params : dict of parameter values to hold fixed while sweeping each one
                  (typically consensus_params from a walk-forward run)
    param_defs  : same {name: ('int'|'float', lo, hi)} used in WF
    n_steps     : number of points to evaluate per parameter sweep
    """
    if fixed_params is None:
        fixed_params = {}
    if score_fn is None:
        score_fn = _default_score
    if reject_fn is None:
        reject_fn = _default_reject

    free_params = [k for k in param_defs if k not in fixed_params]
    sweep_results = {}

    for name in free_params:
        dtype, lo, hi = param_defs[name]

        if dtype == 'int':
            values = np.unique(np.linspace(lo, hi, n_steps).astype(int))
        else:
            values = np.linspace(lo, hi, n_steps)

        rows = []
        for val in values:
            trial_params = {**base_params, name: int(val) if dtype == 'int' else float(val)}

            try:
                result = strategy_fn(df.copy(), trial_params)
                if result is None:
                    rows.append({'value': val, 'score': None})
                    continue
                strategy_df, _ = result
                m = _run_backtest(strategy_df, cost)
            except Exception:
                rows.append({'value': val, 'score': None})
                continue

            if reject_fn(m):
                rows.append({'value': val, 'score': None})
                continue

            rows.append({
                'value':    val,
                'score':    score_fn(m),
                'sharpe':   m['sharpe_ratio'],
                'calmar':   _calmar(m),
                'return':   m['total_return'],
                'drawdown': m['max_drawdown'],
                'trades':   m['num_trades'],
            })

        sweep_results[name] = pd.DataFrame(rows)

    return sweep_results

def plateau_summary(sweep_results, base_params, stability_df=None, threshold=0.20):
    """
    Print a robustness verdict for each parameter based on its sweep.

    For each param, measures what fraction of the sweep range retains at least
    (1 - threshold) of the peak score.  >60% → robust plateau;  <30% → fragile.

    Parameters
    ----------
    sweep_results : output of plateau_analysis()
    base_params   : the parameter set used as the centre point
    stability_df  : optional output of _build_stability_df() — supplies fold CV per param
    threshold     : max acceptable fractional score drop (0.20 = 20% drop OK)
    """
    # ── build fold CV lookup if stability_df provided ─────────────────────────
    fold_cv = {}
    if stability_df is not None and 'param' in stability_df.columns:
        fold_cv = dict(zip(stability_df['param'], stability_df['cv']))

    # ── collect all rows first so we can sort before printing ─────────────────
    rows = []
    for name, sweep_df in sweep_results.items():
        valid = sweep_df.dropna(subset=['score'])
        base_val = base_params.get(name, '?')
        bv = f'{base_val:.4f}' if isinstance(base_val, float) else str(base_val)
        cv_fold = fold_cv.get(name, None)

        if len(valid) == 0:
            rows.append({
                'param':       name,
                'bv':          bv,
                'peak_score':  None,
                'plateau_pct': -1,       # sort last
                'cv_fold':     cv_fold,
                'verdict':     'no valid trials',
            })
            continue

        peak_score  = valid['score'].max()
        cutoff      = peak_score * (1 - threshold)
        above       = (valid['score'] >= cutoff).sum()
        plateau_pct = above / len(valid) * 100

        if   plateau_pct >= 60: verdict = 'Robust'
        elif plateau_pct >= 30: verdict = 'Moderate'
        else:                   verdict = 'FRAGILE'

        rows.append({
            'param':       name,
            'bv':          bv,
            'peak_score':  peak_score,
            'plateau_pct': plateau_pct,
            'cv_fold':     cv_fold,
            'verdict':     verdict,
        })

    # ── sort: no-data last, then descending plateau%, then descending peak score
    rows.sort(key=lambda r: (
        r['plateau_pct'] == -1,
        -r['plateau_pct'],
        -(r['peak_score'] if r['peak_score'] is not None else 0),
    ))

    # ── print ──────────────────────────────────────────────────────────────────
    W_PARAM   = 24
    W_BASE    = 10
    W_PEAK    = 10
    W_PLAT    = 10
    W_CV      = 8
    W_VERDICT = 24

    total = W_PARAM + W_BASE + W_PEAK + W_PLAT + W_CV + W_VERDICT + 5
    print(f"\n{'═' * total}")
    print('PLATEAU ANALYSIS — PARAMETER ROBUSTNESS')
    print(f"{'═' * total}")
    print(
        f'{"Parameter":<{W_PARAM}} '
        f'{"Consensus":>{W_BASE}} '
        f'{"Peak Score":>{W_PEAK}} '
        f'{"Plateau %":>{W_PLAT}} '
        f'{"Fold CV":>{W_CV}} '
        f'{"Verdict":<{W_VERDICT}}'
    )
    print(
        f'{"─" * W_PARAM} '
        f'{"─" * W_BASE} '
        f'{"─" * W_PEAK} '
        f'{"─" * W_PLAT} '
        f'{"─" * W_CV} '
        f'{"─" * W_VERDICT}'
    )

    verdicts = []
    for r in rows:
        peak_s  = f'{r["peak_score"]:>{W_PEAK}.3f}' if r['peak_score'] is not None else f'{"N/A":>{W_PEAK}}'
        plat_s  = f'{r["plateau_pct"]:>{W_PLAT}.1f}%' if r['plateau_pct'] >= 0    else f'{"N/A":>{W_PLAT}}'
        cv_s    = f'{r["cv_fold"]:>{W_CV}.3f}'         if r['cv_fold'] is not None  else f'{"N/A":>{W_CV}}'

        print(
            f'{r["param"]:<{W_PARAM}} '
            f'{r["bv"]:>{W_BASE}} '
            f'{peak_s} '
            f'{plat_s} '
            f'{cv_s} '
            f'{r["verdict"]:<{W_VERDICT}}'
        )
        verdicts.append({
            'param':       r['param'],
            'plateau_pct': r['plateau_pct'] if r['plateau_pct'] >= 0 else None,
            'cv_fold':     r['cv_fold'],
            'verdict':     r['verdict'],
        })

    return pd.DataFrame(verdicts)

# ──────────────────────────────────────────────────────────────────────────────
#  Neighbourhood Perturbation Test
# ──────────────────────────────────────────────────────────────────────────────

def perturbation_test(
    df,
    strategy_fn,
    base_params,
    param_defs,
    fixed_params  = None,
    cost          = 0.001,
    score_fn      = None,
    reject_fn     = None,
    pct_offsets   = (0.05, 0.10, 0.20),
    n_samples     = 50,
    seed          = 42,
):
    """
    Randomly perturb ALL free parameters simultaneously by up to ±pct of their
    range, and measure score degradation.  This tests whether the optimum lives
    on a broad hill or a narrow spike.

    Returns a DataFrame with columns [offset_pct, mean_score, median_score,
    std_score, min_score, pct_degradation_from_base].
    """
    if fixed_params is None:
        fixed_params = {}
    if score_fn is None:
        score_fn = _default_score
    if reject_fn is None:
        reject_fn = _default_reject

    rng = np.random.default_rng(seed)
    free = [k for k in param_defs if k not in fixed_params]

    # baseline score
    try:
        result = strategy_fn(df.copy(), base_params)
        strat_df, _ = result
        base_m = _run_backtest(strat_df, cost)
        base_score = score_fn(base_m)
    except Exception:
        print('ERROR: base_params failed to produce a valid backtest')
        return None

    records = []
    for pct in pct_offsets:
        scores = []
        for _ in range(n_samples):
            trial_params = dict(base_params)
            for name in free:
                dtype, lo, hi = param_defs[name]
                rng_width = (hi - lo) * pct
                noise     = rng.uniform(-rng_width, rng_width)
                new_val   = np.clip(base_params[name] + noise, lo, hi)
                trial_params[name] = int(round(new_val)) if dtype == 'int' else float(new_val)

            try:
                result = strategy_fn(df.copy(), trial_params)
                if result is None:
                    continue
                strat_df, _ = result
                m = _run_backtest(strat_df, cost)
                if reject_fn(m):
                    continue
                scores.append(score_fn(m))
            except Exception:
                continue

        if scores:
            mean_s = np.mean(scores)
            records.append({
                'offset_pct':   pct,
                'n_valid':      len(scores),
                'mean_score':   mean_s,
                'median_score': np.median(scores),
                'std_score':    np.std(scores),
                'min_score':    np.min(scores),
                'degradation':  (base_score - mean_s) / base_score if base_score else 0,
            })

    result_df = pd.DataFrame(records)

    print(f"\n{'═'*75}")
    print('PERTURBATION TEST — NEIGHBOURHOOD ROBUSTNESS')
    print(f"{'═'*75}")
    print(f'Base score: {base_score:.4f}')
    print(f'{"Offset":>8} {"N valid":>8} {"Mean":>8} {"Median":>8} '
          f'{"Std":>8} {"Min":>8} {"Degradation":>12}')
    print(f'{"─"*8} {"─"*8} {"─"*8} {"─"*8} {"─"*8} {"─"*8} {"─"*12}')

    for _, row in result_df.iterrows():
        print(f'{row["offset_pct"]*100:>7.0f}% {row["n_valid"]:>8.0f} '
              f'{row["mean_score"]:>8.4f} {row["median_score"]:>8.4f} '
              f'{row["std_score"]:>8.4f} {row["min_score"]:>8.4f} '
              f'{row["degradation"]*100:>10.1f}%')

    return result_df


# ──────────────────────────────────────────────────────────────────────────────
#  Transaction Cost Stress Test
# ──────────────────────────────────────────────────────────────────────────────

def cost_stress_test(
    oos_combined_df,
    cost_multipliers = (1.0, 1.5, 2.0, 3.0),
    base_cost        = 0.001,
):
    """
    Re-run the combined OOS backtest at escalating transaction costs.
    Fragile strategies degrade sharply; robust ones degrade gradually.

    Returns a DataFrame with one row per cost level.
    """
    records = []
    for mult in cost_multipliers:
        c = base_cost * mult
        m = _run_backtest(oos_combined_df, c)
        if m:
            records.append({
                'cost':          c,
                'cost_mult':     mult,
                'sharpe':        m['sharpe_ratio'],
                'total_return':  m['total_return'],
                'max_drawdown':  m['max_drawdown'],
                'calmar':        _calmar(m),
                'profit_factor': m['profit_factor'],
                'num_trades':    m['num_trades'],
            })

    result_df = pd.DataFrame(records)

    print(f"\n{'═'*75}")
    print('TRANSACTION COST STRESS TEST')
    print(f"{'═'*75}")
    print(f'{"Cost":>8} {"Mult":>6} {"Sharpe":>8} {"Return":>10} '
          f'{"MaxDD":>10} {"Calmar":>8} {"PF":>8}')
    print(f'{"─"*8} {"─"*6} {"─"*8} {"─"*10} {"─"*10} {"─"*8} {"─"*8}')

    for _, r in result_df.iterrows():
        print(f'{r["cost"]:>8.4f} {r["cost_mult"]:>5.1f}x '
              f'{r["sharpe"]:>8.2f} {r["total_return"]*100:>9.2f}% '
              f'{r["max_drawdown"]*100:>9.2f}% {r["calmar"]:>8.2f} '
              f'{r["profit_factor"]:>8.2f}')

    return result_df