import os
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── path setup ────────────────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_BACKTESTER = os.path.join(_THIS_DIR, '..', 'backtester')

sys.path.insert(0, os.path.abspath(_THIS_DIR))
sys.path.insert(0, os.path.abspath(_BACKTESTER))

from engine import backtest
from performance_metrics import build_realized_equity_curve
from wf_engine import (
    _default_score,
    _default_reject,
    _run_backtest,
    _calmar,
    _make_objective,
    _fmt,
)

try:
    from scipy.stats import pearsonr as _pearsonr, t as _t_dist
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    _t_dist = None


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _group_metrics(m):
    """Extract OOS metrics from a backtest result dict."""
    if m is None:
        return dict(sharpe=None, calmar=None, max_dd=None,
                    total_return=None, n_trades=None, win_rate=None,
                    profit_factor=None)
    return {
        'sharpe':        m['sharpe_ratio'],
        'calmar':        _calmar(m),
        'max_dd':        m['max_drawdown'],
        'total_return':  m['total_return'],
        'n_trades':      m['num_trades'],
        'win_rate':      m['win_rate'],
        'profit_factor': m.get('profit_factor'),
    }


def _compute_equity_curve(strategy_df, cost):
    """
    Compute an equity curve from a strategy DataFrame using realized-sizing.

    Entry notional = position_size × realized_equity (closed-trade equity only),
    matching the model in portfolio_metrics.build_realized_equity and engine.py.
    position_size is now respected (previously this function ignored it entirely).

    For single-asset strategies (Close available, no precomputed strategy_returns):
      uses build_realized_equity_curve with undirected Close.pct_change().
    For precomputed/pairs returns (strategy_returns column present):
      falls back to standard MTM compounding (returns are already directional).
    """
    df      = strategy_df.copy()
    eff_pos = df['position'].shift(1).fillna(0)
    eff_sz  = (df['position_size'].shift(1).fillna(0)
               if 'position_size' in df.columns
               else pd.Series(1.0, index=df.index))

    if 'strategy_returns' in df.columns:
        # Precomputed / pairs: direction already baked in — use standard compounding
        raw        = df['strategy_returns']
        pos_change = df['position'].diff().abs()
        net_ret    = eff_pos * raw - pos_change * cost
        return (1 + net_ret.fillna(0)).cumprod()

    elif 'Close' in df.columns:
        # Single-asset: use realized sizing with undirected bar returns
        return build_realized_equity_curve(
            position      = eff_pos,
            position_size = eff_sz,
            raw_returns   = df['Close'].pct_change().fillna(0),
            cost          = cost,
        )

    else:
        return pd.Series(1.0, index=df.index)


# ──────────────────────────────────────────────────────────────────────────────
#  Function 1: generate_cpcv_splits
# ──────────────────────────────────────────────────────────────────────────────

def generate_cpcv_splits(n_bars, N, k, purge_bars=1):
    """
    Partition n_bars into N contiguous groups and enumerate every C(N,k)
    CPCV split together with all distinct complete OOS paths.

    Parameters
    ----------
    n_bars     : total number of bars in the dataset
    N          : number of groups to partition into
    k          : number of groups held out per split (test set size)
    purge_bars : bars removed from the *training* side at each
                 train/test boundary to prevent leakage

    Returns
    -------
    dict
        "groups" : list of N (start_idx, end_idx) tuples  [end exclusive]
        "splits" : list of C(N,k) dicts, each with
                       split_id, test_group_indices, train_indices,
                       test_indices_by_group
        "paths"  : list of dicts defining complete OOS paths, each with
                       path_id, split_assignments [(group_idx, split_id), ...]
    """
    if k < 1 or N < k:
        raise ValueError(f'Require 1 <= k <= N; got N={N}, k={k}')
    if N % k != 0:
        raise ValueError(
            f'N must be divisible by k so paths are complete; got N={N}, k={k}')

    # ── N equal-sized groups ──────────────────────────────────────────────────
    base = n_bars // N
    groups = []
    for i in range(N):
        s = i * base
        e = s + base if i < N - 1 else n_bars
        groups.append((s, e))

    # ── C(N,k) splits ─────────────────────────────────────────────────────────
    splits = []
    for split_id, test_combo in enumerate(itertools.combinations(range(N), k)):
        test_set = set(test_combo)

        test_indices_by_group = {
            g: np.arange(groups[g][0], groups[g][1]) for g in test_combo
        }

        train_idx_list = []
        for g in range(N):
            if g in test_set:
                continue
            g_start, g_end = groups[g]
            eff_start = g_start
            eff_end   = g_end
            # boundary: test group immediately precedes this training group
            if (g - 1) in test_set:
                eff_start = min(g_start + purge_bars, g_end)
            # boundary: test group immediately follows this training group
            if (g + 1) in test_set:
                eff_end = max(eff_start, g_end - purge_bars)
            if eff_start < eff_end:
                train_idx_list.extend(range(eff_start, eff_end))

        splits.append({
            'split_id':              split_id,
            'test_group_indices':    test_combo,
            'train_indices':         np.array(train_idx_list, dtype=int),
            'test_indices_by_group': test_indices_by_group,
        })

    # ── all distinct complete paths ───────────────────────────────────────────
    # A path is a partition of all N groups into N/k disjoint subsets of size k,
    # where each subset corresponds to exactly one split's test set.
    # We enumerate using a greedy recursion that always assigns the smallest
    # unassigned group next, ensuring uniqueness.

    group_to_split_ids = {g: [] for g in range(N)}
    for sp in splits:
        for g in sp['test_group_indices']:
            group_to_split_ids[g].append(sp['split_id'])

    paths       = []
    path_id_ctr = [0]

    def _recurse(remaining, used_groups, current):
        if not remaining:
            paths.append({
                'path_id':           path_id_ctr[0],
                'split_assignments': sorted(current, key=lambda x: x[0]),
            })
            path_id_ctr[0] += 1
            return
        first = min(remaining)
        for sid in group_to_split_ids[first]:
            tg = set(splits[sid]['test_group_indices'])
            if tg & used_groups:
                continue
            for g in tg:
                current.append((g, sid))
            _recurse(remaining - tg, used_groups | tg, current)
            for _ in tg:
                current.pop()

    _recurse(set(range(N)), set(), [])

    return {'groups': groups, 'splits': splits, 'paths': paths}


# ──────────────────────────────────────────────────────────────────────────────
#  Function 2: run_cpcv
# ──────────────────────────────────────────────────────────────────────────────

def run_cpcv(
    df,
    strategy_fn,
    param_defs,
    fixed_params = None,
    N            = 8,
    k            = 2,
    purge_bars   = 1,
    n_trials     = 400,
    burnin       = 100,
    cost         = 0.001,
    score_fn     = None,
    reject_fn    = None,
    verbose      = True,
    n_jobs       = 1,
):
    """
    Combinatorial Purged Cross-Validation (CPCV).

    Partitions df into N groups, optimises on every C(N,k) training complement
    via Optuna TPE (same sampler as wf_engine), evaluates OOS on the k held-out
    groups with burnin warmup, then stitches all groups into complete equity paths.

    Parameters
    ----------
    df           : full OHLCV DataFrame with DatetimeIndex
    strategy_fn  : fn(df_slice, params) -> (strategy_df, indicator_cols)
    param_defs   : {name: ('int'|'float', lo, hi)}
    fixed_params : {name: value}  -- always override Optuna, same as walk_forward
    N            : number of groups (default 8)
    k            : groups held out per split (default 2)
    purge_bars   : bars purged from training side at every train/test boundary
    n_trials     : Optuna trials per split
    burnin       : bars prepended to each test group for indicator warmup;
                   these bars are excluded from OOS evaluation
    cost         : per-leg trading cost fraction (round-trip = 2 × cost)
    score_fn     : fn(metrics) -> float  (default: wf_engine composite)
    reject_fn    : fn(metrics) -> bool   (default: wf_engine min filters)
    verbose      : print one progress line per split
    n_jobs       : parallel Optuna trials per split (default 1 = serial).
                   Set to -1 to use all CPU cores, or a positive int.
                   Uses threads — strategy_fn must not mutate shared state.

    Returns
    -------
    dict
        split_results     : list of per-split dicts
        paths             : list of per-path dicts with equity_curve and metrics
        param_distributions : DataFrame (n_splits × n_free_params)
        group_boundaries  : list of (start_date, end_date) per group
        config            : dict of run parameters
        param_defs        : the param_defs dict (stored for analysis functions)
        fixed_params      : the fixed_params dict
        n_splits          : int
        n_paths           : int
    """
    if fixed_params is None:
        fixed_params = {}
    if score_fn is None:
        score_fn = _default_score
    if reject_fn is None:
        reject_fn = _default_reject

    free_params = [p for p in param_defs if p not in fixed_params]

    # ── generate structural metadata ──────────────────────────────────────────
    cpcv_meta = generate_cpcv_splits(len(df), N, k, purge_bars)
    groups    = cpcv_meta['groups']
    splits    = cpcv_meta['splits']
    path_meta = cpcv_meta['paths']
    n_splits  = len(splits)
    n_paths   = len(path_meta)

    if verbose:
        print(f'CPCV: N={N}  k={k}  splits={n_splits}  paths={n_paths}  '
              f'trials={n_trials}  burnin={burnin}  purge={purge_bars}')
        for i, (s, e) in enumerate(groups):
            print(f'  Group {i+1}: [{df.index[s].date()} → {df.index[e-1].date()}]'
                  f'  ({e - s} bars)')
        if fixed_params:
            print(f'\nFixed ({len(fixed_params)}): {list(fixed_params.keys())}')
            print(f'Free  ({len(free_params)}): {free_params}')
        print()

    # ── run each split ────────────────────────────────────────────────────────
    split_results = []
    # oos_dfs[split_id][group_idx] = trimmed OOS strategy DataFrame
    oos_dfs: dict[int, dict[int, pd.DataFrame | None]] = {}

    for split in splits:
        sid         = split['split_id']
        test_groups = split['test_group_indices']
        train_idx   = split['train_indices']

        df_train = df.iloc[train_idx].copy()

        # ── Optuna optimisation on training slice ──────────────────────────
        study = optuna.create_study(
            direction  = 'maximize',
            study_name = f'cpcv_split_{sid}',
            sampler    = optuna.samplers.TPESampler(seed=42 + sid),
        )
        study.optimize(
            _make_objective(df_train, strategy_fn, param_defs,
                            fixed_params, cost, score_fn, reject_fn),
            n_trials          = n_trials,
            n_jobs            = n_jobs,
            show_progress_bar = False,
        )

        best_params = {**fixed_params, **study.best_params}
        is_score    = study.best_value

        # ── IS Sharpe — one extra evaluation on the training slice ─────────
        # score_fn is a composite (Sharpe + Calmar + Return), so comparing it
        # directly against OOS Sharpe would be apples-to-oranges.  Run the
        # strategy once on df_train with best_params and extract the raw Sharpe.
        is_sharpe = float('nan')
        try:
            _is_result = strategy_fn(df_train, best_params)   # strategy copies internally
            if _is_result is not None:
                _is_strat_df, _ = _is_result
                _is_m = _run_backtest(_is_strat_df, cost)
                if _is_m is not None:
                    is_sharpe = float(_is_m['sharpe_ratio'])
        except Exception:
            pass

        # ── OOS evaluation per test group ──────────────────────────────────
        group_results = {}
        oos_dfs[sid]  = {}

        for g in test_groups:
            g_start, g_end = groups[g]

            # prepend burnin bars from immediately before the group
            burnin_start      = max(0, g_start - burnin)
            slice_with_burnin = df.iloc[burnin_start:g_end].copy()
            test_start_date   = df.index[g_start]

            oos_df    = None
            g_metrics = None
            try:
                result = strategy_fn(slice_with_burnin.copy(), best_params)
                if result is not None:
                    strat_df, indicator_cols = result

                    # clean NaNs on indicator columns (wf_engine convention)
                    existing_ind = [c for c in indicator_cols if c in strat_df.columns]
                    if existing_ind:
                        strat_df.dropna(subset=existing_ind, inplace=True)

                    # trim burnin: keep only genuine OOS bars
                    oos_df = strat_df.loc[strat_df.index >= test_start_date].copy()

                    if len(oos_df) > 0:
                        m = _run_backtest(oos_df, cost)
                        g_metrics = _group_metrics(m)
            except Exception:
                pass

            group_results[g]  = {'metrics': g_metrics, 'oos_strategy_df': oos_df}
            oos_dfs[sid][g]   = oos_df

        # ── verbose progress line ──────────────────────────────────────────
        if verbose:
            sharpe_strs = []
            for g in test_groups:
                gm = group_results[g]['metrics']
                sh = _fmt(gm['sharpe']) if (gm and gm['sharpe'] is not None) else 'N/A'
                sharpe_strs.append(sh)
            test_groups_1based = tuple(g + 1 for g in test_groups)
            print(f'Split {sid+1}/{n_splits} done | IS score: {_fmt(is_score)} | '
                  f'OOS groups {test_groups_1based} Sharpe: {", ".join(sharpe_strs)}')

        split_results.append({
            'split_id':           sid,
            'test_group_indices': test_groups,
            'best_params':        best_params,
            'is_score':           is_score,
            'is_sharpe':          is_sharpe,
            'group_results':      group_results,
        })

    # ── param_distributions ───────────────────────────────────────────────────
    param_dist_rows = [{p: sr['best_params'].get(p) for p in free_params}
                       for sr in split_results]
    param_distributions = pd.DataFrame(param_dist_rows)

    # ── stitch paths ──────────────────────────────────────────────────────────
    path_results = []

    for pm in path_meta:
        pid         = pm['path_id']
        assignments = pm['split_assignments']   # [(group_idx, split_id), ...]

        slices = []
        for g_idx, s_id in sorted(assignments, key=lambda x: x[0]):
            oos_df = oos_dfs.get(s_id, {}).get(g_idx)
            if oos_df is not None and len(oos_df) > 0:
                slices.append(oos_df)

        if not slices:
            path_results.append({
                'path_id':          pid,
                'split_assignments': assignments,
                'equity_curve':     None,
                'sharpe':           None,
                'calmar':           None,
                'max_dd':           None,
                'total_return':     None,
            })
            continue

        path_df = pd.concat(slices).sort_index()
        path_df = path_df[~path_df.index.duplicated(keep='first')]

        equity_curve        = _compute_equity_curve(path_df, cost)
        equity_curve.name   = f'path_{pid}'
        pm_metrics          = _run_backtest(path_df, cost)

        path_results.append({
            'path_id':          pid,
            'split_assignments': assignments,
            'equity_curve':     equity_curve,
            'sharpe':           pm_metrics['sharpe_ratio']  if pm_metrics else None,
            'calmar':           _calmar(pm_metrics)         if pm_metrics else None,
            'max_dd':           pm_metrics['max_drawdown']  if pm_metrics else None,
            'total_return':     pm_metrics['total_return']  if pm_metrics else None,
        })

    # ── group_boundaries ──────────────────────────────────────────────────────
    group_boundaries = [(df.index[s], df.index[e - 1]) for s, e in groups]

    # ── efficiency_stats ──────────────────────────────────────────────────────
    _eff_rows = []
    for sr in split_results:
        grp_sharpes = [
            gr['metrics']['sharpe']
            for gr in sr['group_results'].values()
            if gr['metrics'] and gr['metrics']['sharpe'] is not None
        ]
        oos_sh = float(np.mean(grp_sharpes)) if grp_sharpes else float('nan')
        is_sh  = sr.get('is_sharpe', float('nan'))   # IS Sharpe — same metric as OOS
        ratio  = (oos_sh / is_sh
                  if (is_sh != 0 and not (np.isnan(oos_sh) or np.isnan(is_sh)))
                  else float('nan'))
        _eff_rows.append({
            'split_id':         sr['split_id'],
            'is_score':         float(sr['is_score']),   # kept for reference only
            'is_sharpe':        is_sh,
            'oos_sharpe':       oos_sh,
            'efficiency_ratio': ratio,
            'test_groups':      sr['test_group_indices'],
        })

    _eff_df = pd.DataFrame(_eff_rows)
    _v_rat  = _eff_df['efficiency_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    _v_oos  = _eff_df['oos_sharpe'].replace([np.inf, -np.inf], np.nan).dropna()
    _v_is   = _eff_df['is_sharpe'].replace([np.inf, -np.inf], np.nan).dropna()

    efficiency_stats = {
        'per_split':                        _eff_df,
        'mean_efficiency':                  float(_v_rat.mean())          if len(_v_rat) else float('nan'),
        'std_efficiency':                   float(_v_rat.std(ddof=1))     if len(_v_rat) > 1 else 0.0,
        'median_efficiency':                float(_v_rat.median())        if len(_v_rat) else float('nan'),
        'pct_splits_positive_oos':          float((_v_oos > 0).mean())    if len(_v_oos) else float('nan'),
        'pct_splits_efficiency_above_half': float((_v_rat > 0.5).mean())  if len(_v_rat) else float('nan'),
    }

    results = {
        'split_results':      split_results,
        'paths':              path_results,
        'param_distributions': param_distributions,
        'group_boundaries':   group_boundaries,
        'efficiency_stats':   efficiency_stats,
        'config':             {
            'N': N, 'k': k, 'purge_bars': purge_bars,
            'n_trials': n_trials, 'burnin': burnin, 'cost': cost,
        },
        'param_defs':    param_defs,
        'fixed_params':  fixed_params,
        'n_splits':      n_splits,
        'n_paths':       n_paths,
    }

    if verbose:
        print(f'\nCPCV complete — {n_splits} splits optimised, '
              f'{n_paths} complete OOS paths stitched')

    # Convenience pickle (uncomment in notebook):
    # pd.to_pickle(results, 'eth_cpcv.pkl')

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Function 3: cpcv_parameter_analysis
# ──────────────────────────────────────────────────────────────────────────────

def cpcv_parameter_analysis(cpcv_results):
    """
    Analyse parameter behaviour across all CPCV splits.

    Parameters
    ----------
    cpcv_results : dict returned by run_cpcv

    Returns
    -------
    dict
        distribution_stats     : DataFrame (n_params × stats)
        param_performance_corr : DataFrame (n_params × {correlation, p_value})
        cross_param_corr       : DataFrame (n_params × n_params)
        tercile_comparison     : dict {param: {top/bottom tercile stats}}
        consensus_ranges       : DataFrame (n_params × range recommendations)
    """
    split_results = cpcv_results['split_results']
    param_dist    = cpcv_results['param_distributions']
    param_defs    = cpcv_results.get('param_defs', {})
    free_params   = list(param_dist.columns)

    # per-split OOS Sharpe = mean across its k test groups
    oos_sharpes = []
    for sr in split_results:
        sharpes = [
            gr['metrics']['sharpe']
            for gr in sr['group_results'].values()
            if gr['metrics'] and gr['metrics']['sharpe'] is not None
        ]
        oos_sharpes.append(np.mean(sharpes) if sharpes else np.nan)
    oos_sharpes = np.array(oos_sharpes, dtype=float)

    # ── distribution_stats ────────────────────────────────────────────────────
    dist_rows = []
    for p in free_params:
        vals = param_dist[p].dropna().values.astype(float)
        if len(vals) == 0:
            dist_rows.append({
                'param': p, 'mean': np.nan, 'std': np.nan, 'cv': np.nan,
                'min': np.nan, 'q25': np.nan, 'median': np.nan,
                'q75': np.nan, 'max': np.nan, 'iqr': np.nan,
            })
            continue
        med = float(np.median(vals))
        std = float(np.std(vals))
        cv  = std / abs(med) if med != 0 else 999.0
        q25 = float(np.percentile(vals, 25))
        q75 = float(np.percentile(vals, 75))
        dist_rows.append({
            'param':  p,
            'mean':   float(np.mean(vals)),
            'std':    std,
            'cv':     cv,
            'min':    float(np.min(vals)),
            'q25':    q25,
            'median': med,
            'q75':    q75,
            'max':    float(np.max(vals)),
            'iqr':    q75 - q25,
        })
    distribution_stats = pd.DataFrame(dist_rows).set_index('param')

    # ── param_performance_corr ────────────────────────────────────────────────
    corr_rows = []
    for p in free_params:
        vals  = param_dist[p].values.astype(float)
        valid = ~(np.isnan(vals) | np.isnan(oos_sharpes))
        x, y  = vals[valid], oos_sharpes[valid]
        if len(x) >= 3:
            if _HAS_SCIPY:
                r, pv = _pearsonr(x, y)
            else:
                r  = float(np.corrcoef(x, y)[0, 1])
                n  = len(x)
                ts = r * np.sqrt(n - 2) / np.sqrt(max(1 - r ** 2, 1e-10))
                # two-tailed p-value approximation via normal when scipy absent
                pv = float(2 * (1 - _normal_cdf(abs(ts))))
        else:
            r, pv = np.nan, np.nan
        corr_rows.append({'param': p, 'correlation': r, 'p_value': pv})
    param_performance_corr = pd.DataFrame(corr_rows).set_index('param')

    # ── cross_param_corr ──────────────────────────────────────────────────────
    if len(free_params) > 1:
        cross_param_corr = param_dist[free_params].corr()
    else:
        cross_param_corr = pd.DataFrame(
            [[1.0]], index=free_params, columns=free_params)

    # ── tercile_comparison ────────────────────────────────────────────────────
    valid_mask = ~np.isnan(oos_sharpes)
    n_valid    = int(valid_mask.sum())
    # Use valid-split count (not total splits) so each tercile is exactly 1/3
    # of the splits that actually produced a Sharpe — not inflated by failures.
    n_tercile  = max(1, n_valid // 3)

    tercile_comparison = {}
    if n_valid >= 3:
        valid_idx   = np.where(valid_mask)[0]
        sorted_pos  = np.argsort(oos_sharpes[valid_idx])
        top_idx     = valid_idx[sorted_pos[-n_tercile:]]
        bottom_idx  = valid_idx[sorted_pos[:n_tercile]]

        for p in free_params:
            vals     = param_dist[p].values.astype(float)
            top_v    = vals[top_idx]
            bot_v    = vals[bottom_idx]
            tm, ts   = float(np.mean(top_v)),    float(np.std(top_v))
            bm, bs   = float(np.mean(bot_v)),    float(np.std(bot_v))
            pooled   = np.sqrt((ts ** 2 + bs ** 2) / 2 + 1e-10)
            tercile_comparison[p] = {
                'top_tercile_mean':    tm,
                'top_tercile_std':     ts,
                'bottom_tercile_mean': bm,
                'bottom_tercile_std':  bs,
                'separation':          abs(tm - bm) / pooled,
            }

    # ── consensus_ranges ──────────────────────────────────────────────────────
    range_rows = []
    for p in free_params:
        if p not in distribution_stats.index:
            continue
        ds  = distribution_stats.loc[p]
        q25 = ds['q25']
        q75 = ds['q75']
        med = ds['median']
        cv  = ds['cv']

        if param_defs and p in param_defs:
            _, lo, hi = param_defs[p]
            curr_low, curr_high = lo, hi
        else:
            curr_low  = ds['min']
            curr_high = ds['max']

        if cv < 0.10:
            action = f'fix at {med:.4g}'
        elif cv < 0.25:
            action = 'narrow to IQR'
        else:
            action = 'keep current range'

        range_rows.append({
            'param':            p,
            'recommended_low':  q25,
            'recommended_high': q75,
            'current_low':      curr_low,
            'current_high':     curr_high,
            'action':           action,
        })
    consensus_ranges = pd.DataFrame(range_rows).set_index('param')

    return {
        'distribution_stats':     distribution_stats,
        'param_performance_corr': param_performance_corr,
        'cross_param_corr':       cross_param_corr,
        'tercile_comparison':     tercile_comparison,
        'consensus_ranges':       consensus_ranges,
    }


def _normal_cdf(x):
    """Approximate standard normal CDF — used only when scipy is absent."""
    return 0.5 * (1.0 + float(np.sign(x)) *
                  np.sqrt(1 - np.exp(-2 * x ** 2 / np.pi)))


# ──────────────────────────────────────────────────────────────────────────────
#  Function 5: cpcv_confidence_intervals
# ──────────────────────────────────────────────────────────────────────────────

def cpcv_confidence_intervals(cpcv_results, confidence=0.95):
    """
    Compute confidence intervals on mean path Sharpe, Calmar, and Max DD.

    Two methods:
      1. Naive t-interval: treats paths as independent (anticonservative).
      2. Effective-N adjusted CI: accounts for path overlap via an N×N binary
         overlap matrix. N_eff = N² / sum(overlap_matrix). Uses N_eff-1 degrees
         of freedom (conservative).

    Two paths overlap when they share at least one split in their
    split_assignments lists — i.e. they used the same Optuna-optimised params
    for at least one segment of their equity curve.

    Parameters
    ----------
    cpcv_results : dict returned by run_cpcv()
    confidence   : float, default 0.95

    Returns
    -------
    dict with keys 'sharpe', 'calmar', 'max_dd', 'confidence'.
    Each metric sub-dict contains:
        values, mean, std, n_paths, n_effective,
        naive_ci, adjusted_ci, conservative_lower_bound
    """
    paths = cpcv_results['paths']

    # only paths with a valid Sharpe — same filter for all three metrics because
    # Sharpe/Calmar/MaxDD all come from the same pm_metrics object per path
    valid = [p for p in paths if p['sharpe'] is not None]
    if len(valid) == 0:
        return {'sharpe': None, 'calmar': None, 'max_dd': None,
                'confidence': confidence}

    sharpes = np.array([p['sharpe'] for p in valid], dtype=float)
    calmars = np.array([p['calmar'] if p['calmar'] is not None else np.nan
                        for p in valid], dtype=float)
    maxdds  = np.array([p['max_dd'] if p['max_dd'] is not None else np.nan
                        for p in valid], dtype=float)

    N = len(valid)

    # ── weighted overlap matrix ───────────────────────────────────────────────
    # cell (i,j) = fraction of splits shared between paths i and j.
    #   sharing 0/4 splits → 0.00  (independent)
    #   sharing 1/4 splits → 0.25  (partially correlated)
    #   sharing 4/4 splits → 1.00  (same path, diagonal)
    #
    # Binary (any-shared → 1) overcounts: it treats sharing 1 of 4 splits the
    # same as sharing all 4, collapsing N_eff to ~2.3. Weighted overlap gives
    # N_eff = n_splits / splits_per_path = C(N,k) / (N/k), which for the
    # standard N=8, k=2 setup equals 28 / 4 = 7 — the true number of
    # independent Optuna runs divided by how many each path consumes.
    path_splits     = [frozenset(sid for _, sid in p['split_assignments'])
                       for p in valid]
    splits_per_path = max(len(ps) for ps in path_splits) if path_splits else 1
    overlap = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            overlap[i, j] = len(path_splits[i] & path_splits[j]) / splits_per_path

    n_eff = float(N ** 2) / float(overlap.sum()) if overlap.sum() > 0 else 1.0

    alpha = 1.0 - confidence

    # t critical values (or normal approximation when scipy absent)
    _fallback = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    _z = _fallback.get(round(confidence, 2), 1.960)

    if _HAS_SCIPY:
        t_naive = float(_t_dist.ppf(1 - alpha / 2, df=max(N - 1, 1)))
        t_eff   = float(_t_dist.ppf(1 - alpha / 2, df=max(n_eff - 1, 1)))
    else:
        t_naive = t_eff = _z

    def _ci_dict(vals):
        ok    = vals[~np.isnan(vals)]
        n     = len(ok)
        mean  = float(np.mean(ok))
        std   = float(np.std(ok, ddof=1)) if n > 1 else 0.0
        sem   = std / np.sqrt(n)         if n > 0 else 0.0
        sem_e = std / np.sqrt(n_eff)     if n_eff > 0 else 0.0
        naive_ci    = (mean - t_naive * sem,   mean + t_naive * sem)
        adjusted_ci = (mean - t_eff   * sem_e, mean + t_eff   * sem_e)
        return {
            'values':                  vals,
            'mean':                    mean,
            'std':                     float(np.std(ok)) if n > 1 else 0.0,
            'n_paths':                 n,
            'n_effective':             n_eff,
            'naive_ci':                naive_ci,
            'adjusted_ci':             adjusted_ci,
            'conservative_lower_bound': adjusted_ci[0],
        }

    return {
        'sharpe':     _ci_dict(sharpes),
        'calmar':     _ci_dict(calmars),
        'max_dd':     _ci_dict(maxdds),
        'confidence': confidence,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Function 6: cpcv_ci_summary
# ──────────────────────────────────────────────────────────────────────────────

def cpcv_ci_summary(ci_results):
    """
    Print a formatted confidence-interval table from cpcv_confidence_intervals().

    Shows naive (anticonservative) and overlap-adjusted (conservative) CIs for
    Sharpe, Calmar, and Max DD, followed by conservative floor annotations.
    """
    if ci_results is None:
        print('[cpcv_ci_summary] No CI results to display.')
        return

    conf    = ci_results.get('confidence', 0.95)
    pct     = int(round(conf * 100))
    sh      = ci_results.get('sharpe')
    ca      = ci_results.get('calmar')
    dd      = ci_results.get('max_dd')

    if sh is None:
        print('[cpcv_ci_summary] No valid paths in CI results.')
        return

    n_paths = sh['n_paths']
    n_eff   = sh['n_effective']

    W_TOT  = 72
    W_MET  = 7
    W_MET2 = 23
    W_VAL  = 8
    W_NOTE = 17

    def _fv(v, pct_scale=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'N/A'
        return f'{v*100:.2f}%' if pct_scale else f'{v:.3f}'

    print(f"\n{'═' * W_TOT}")
    print(f'CONFIDENCE INTERVALS  ({pct}%)')
    print(f"{'═' * W_TOT}")
    print(
        f'{"Metric":<{W_MET}}  {"Method":<{W_MET2}}  '
        f'{"Lower":>{W_VAL}}  {"Upper":>{W_VAL}}  Note'
    )
    print(
        f'{"─" * W_MET}  {"─" * W_MET2}  '
        f'{"─" * W_VAL}  {"─" * W_VAL}  {"─" * W_NOTE}'
    )

    rows = [
        ('Sharpe', sh, False),
        ('Calmar', ca, False),
        ('Max DD', dd, True),
    ]

    for label, data, as_pct in rows:
        if data is None:
            continue
        lo_n, hi_n = data['naive_ci']
        lo_a, hi_a = data['adjusted_ci']
        n_naive_str = f'Naive (N={n_paths})'
        n_adj_str   = f'Adjusted (N_eff={n_eff:.1f})'
        print(
            f'{label:<{W_MET}}  {n_naive_str:<{W_MET2}}  '
            f'{_fv(lo_n, as_pct):>{W_VAL}}  {_fv(hi_n, as_pct):>{W_VAL}}  '
            f'anticonservative'
        )
        print(
            f'{"":<{W_MET}}  {n_adj_str:<{W_MET2}}  '
            f'{_fv(lo_a, as_pct):>{W_VAL}}  {_fv(hi_a, as_pct):>{W_VAL}}  '
            f'conservative'
        )

    print(f"{'─' * W_TOT}")
    if sh:
        print(f"  Conservative Sharpe floor: {sh['conservative_lower_bound']:.3f}"
              f"  (adjusted lower bound)")
    if ca:
        print(f"  Conservative Calmar floor: {ca['conservative_lower_bound']:.3f}"
              f"  (adjusted lower bound)")


# ──────────────────────────────────────────────────────────────────────────────
#  Function 4: cpcv_summary
# ──────────────────────────────────────────────────────────────────────────────

def cpcv_summary(
    cpcv_results,
    show_group_legend = True,   # group date ranges
    show_distribution = True,   # path distribution percentile table
    show_paths        = False,  # full per-path detail (105 rows, opt-in)
    show_highlights   = True,   # top 5 / bottom 5 paths by Sharpe
    show_split_legend = True,   # split legend decoding gN→sM notation
    show_ci           = True,   # confidence intervals
    show_efficiency   = True,   # IS/OOS efficiency table
):
    """
    Print selected sections of the CPCV summary.

    Call with different flag combinations across two notebook cells to avoid
    one long scrollable block:

      # Cell 1 — group legend + distribution stats
      cpcv_summary(results, show_highlights=False, show_split_legend=False, show_ci=False)

      # Cell 2 — top/bottom highlights + split legend + CI
      cpcv_summary(results, show_group_legend=False, show_distribution=False)

    Parameters
    ----------
    cpcv_results      : dict returned by run_cpcv()
    show_group_legend : print group date-range table (default True)
    show_distribution : print path distribution percentile table (default True)
    show_paths        : print full per-path detail, one row per path (default False)
    show_highlights   : print top-5 / bottom-5 paths by Sharpe (default True)
    show_split_legend : print split legend decoding gN→sM notation (default True)
    show_ci           : compute and print confidence intervals (default True)
    show_efficiency   : print IS/OOS efficiency table (default True)
    """
    paths         = cpcv_results['paths']
    bounds        = cpcv_results['group_boundaries']
    config        = cpcv_results['config']
    N             = config['N']
    split_results = cpcv_results['split_results']
    sr_map        = {sr['split_id']: sr for sr in split_results}

    # ── collect valid path metrics (always computed; cheap) ───────────────────
    sharpes = [p['sharpe']       for p in paths if p['sharpe']       is not None]
    calmars = [p['calmar']       for p in paths if p['calmar']       is not None]
    maxdds  = [p['max_dd']       for p in paths if p['max_dd']       is not None]
    returns = [p['total_return'] for p in paths if p['total_return'] is not None]

    valid_paths = [p for p in paths if p['sharpe'] is not None]
    ranked      = sorted(valid_paths, key=lambda p: p['sharpe'], reverse=True)
    top5        = ranked[:5]
    bot5        = ranked[-5:][::-1]

    W_SHARPE = 8
    W_CALMAR = 8
    W_DD     = 9
    W_RET    = 9
    W_STAT   = 8
    W_COL    = 10
    W_RANK   = 6
    TOTAL0   = W_STAT + W_COL * 4 + 4
    TOTAL_H  = W_RANK + W_SHARPE + W_CALMAR + W_DD + W_RET + 4
    INDENT   = ' ' * (W_RANK + 1)

    def _pct_col(vals, q):
        return np.percentile(vals, q) if vals else float('nan')

    def _stat_row(label, sh, ca, dd, rt):
        sh_s = f'{sh:.2f}'      if np.isfinite(sh) else 'N/A'
        ca_s = f'{ca:.2f}'      if np.isfinite(ca) else 'N/A'
        dd_s = f'{dd*100:.2f}%' if np.isfinite(dd) else 'N/A'
        rt_s = f'{rt*100:.2f}%' if np.isfinite(rt) else 'N/A'
        return (
            f'{label:<{W_STAT}} '
            f'{sh_s:>{W_COL}} '
            f'{ca_s:>{W_COL}} '
            f'{dd_s:>{W_COL}} '
            f'{rt_s:>{W_COL}}'
        )

    def _highlight_rows(rank_label, pr):
        assignments  = sorted(pr['split_assignments'], key=lambda x: x[0])
        assign_str   = '  '.join(f'g{g+1}→s{sid+1}' for g, sid in assignments)
        metrics_line = (
            f'{rank_label:>{W_RANK}} '
            f'{_fmt(pr["sharpe"]):>{W_SHARPE}} '
            f'{_fmt(pr["calmar"]):>{W_CALMAR}} '
            f'{_fmt(pr["max_dd"],       pct=True):>{W_DD}} '
            f'{_fmt(pr["total_return"], pct=True):>{W_RET}}'
        )
        return metrics_line, f'{INDENT}{assign_str}'

    hdr = (
        f'{"Rank":>{W_RANK}} '
        f'{"Sharpe":>{W_SHARPE}} '
        f'{"Calmar":>{W_CALMAR}} '
        f'{"Max DD":>{W_DD}} '
        f'{"Return":>{W_RET}}'
    )
    sep = (
        f'{"─" * W_RANK} '
        f'{"─" * W_SHARPE} '
        f'{"─" * W_CALMAR} '
        f'{"─" * W_DD} '
        f'{"─" * W_RET}'
    )

    # ── Section 0: header + group legend ─────────────────────────────────────
    if show_group_legend:
        print(f"\n{'═' * TOTAL0}")
        print(f'CPCV SUMMARY  '
              f'(N={N}, k={config["k"]}, '
              f'{cpcv_results["n_splits"]} splits, '
              f'{len(paths)} paths)')
        print(f"{'═' * TOTAL0}")
        print('\n  Group date ranges:')
        for i, (start, end) in enumerate(bounds):
            print(f'    g{i+1}  {start.date()} → {end.date()}')

    # ── Section 1: path distribution percentile table ─────────────────────────
    if show_distribution:
        _nan = float('nan')
        print(f"\n{'═' * TOTAL0}")
        print(f'PATH DISTRIBUTION  ({len(paths)} paths)')
        print(f"{'═' * TOTAL0}")
        print(
            f'{"Metric":<{W_STAT}} '
            f'{"Sharpe":>{W_COL}} '
            f'{"Calmar":>{W_COL}} '
            f'{"Max DD":>{W_COL}} '
            f'{"Return":>{W_COL}}'
        )
        print(
            f'{"─" * W_STAT} '
            f'{"─" * W_COL} '
            f'{"─" * W_COL} '
            f'{"─" * W_COL} '
            f'{"─" * W_COL}'
        )
        _stats = [
            ('Mean',   np.mean(sharpes) if sharpes else _nan,
                       np.mean(calmars) if calmars else _nan,
                       np.mean(maxdds)  if maxdds  else _nan,
                       np.mean(returns) if returns else _nan),
            ('Std',    np.std(sharpes)  if sharpes else _nan,
                       np.std(calmars)  if calmars else _nan,
                       np.std(maxdds)   if maxdds  else _nan,
                       np.std(returns)  if returns else _nan),
            ('Min',    _pct_col(sharpes,  0), _pct_col(calmars,  0),
                       _pct_col(maxdds,   0), _pct_col(returns,  0)),
            ('Q25',    _pct_col(sharpes, 25), _pct_col(calmars, 25),
                       _pct_col(maxdds,  25), _pct_col(returns, 25)),
            ('Median', _pct_col(sharpes, 50), _pct_col(calmars, 50),
                       _pct_col(maxdds,  50), _pct_col(returns, 50)),
            ('Q75',    _pct_col(sharpes, 75), _pct_col(calmars, 75),
                       _pct_col(maxdds,  75), _pct_col(returns, 75)),
            ('Max',    _pct_col(sharpes,100), _pct_col(calmars,100),
                       _pct_col(maxdds, 100), _pct_col(returns,100)),
            ('IQR',    (_pct_col(sharpes,75) - _pct_col(sharpes,25)) if sharpes else _nan,
                       (_pct_col(calmars,75) - _pct_col(calmars,25)) if calmars else _nan,
                       (_pct_col(maxdds, 75) - _pct_col(maxdds, 25)) if maxdds  else _nan,
                       (_pct_col(returns,75) - _pct_col(returns,25)) if returns else _nan),
        ]
        for label, sh, ca, dd, rt in _stats:
            print(_stat_row(label, sh, ca, dd, rt))

    # ── Section 2: per-path detail (opt-in) ──────────────────────────────────
    if show_paths:
        W_PATH   = 6
        W_SPLITS = 22
        TOTAL_P  = W_PATH + W_SHARPE + W_CALMAR + W_DD + W_RET + W_SPLITS + 5
        n_valid  = len(valid_paths)
        print(f"\n{'═' * TOTAL_P}")
        print(f'PER-PATH DETAIL  ({n_valid} valid paths)')
        print(f"{'═' * TOTAL_P}")
        print(
            f'{"Path":>{W_PATH}} '
            f'{"Sharpe":>{W_SHARPE}} '
            f'{"Calmar":>{W_CALMAR}} '
            f'{"Max DD":>{W_DD}} '
            f'{"Return":>{W_RET}} '
            f'{"Splits used":<{W_SPLITS}}'
        )
        print(
            f'{"─" * W_PATH} '
            f'{"─" * W_SHARPE} '
            f'{"─" * W_CALMAR} '
            f'{"─" * W_DD} '
            f'{"─" * W_RET} '
            f'{"─" * W_SPLITS}'
        )
        for pr in paths:
            split_ids  = sorted({sid + 1 for _, sid in pr['split_assignments']})
            splits_str = str(split_ids)[:W_SPLITS]
            print(
                f'{pr["path_id"] + 1:>{W_PATH}} '
                f'{_fmt(pr["sharpe"]):>{W_SHARPE}} '
                f'{_fmt(pr["calmar"]):>{W_CALMAR}} '
                f'{_fmt(pr["max_dd"],   pct=True):>{W_DD}} '
                f'{_fmt(pr["total_return"], pct=True):>{W_RET}} '
                f'{splits_str:<{W_SPLITS}}'
            )
        if sharpes:
            print(f'{"─" * TOTAL_P}')
            print(
                f'{"Mean":>{W_PATH}} '
                f'{np.mean(sharpes):>{W_SHARPE}.2f} '
                f'{np.mean(calmars):>{W_CALMAR}.2f} '
                f'{np.mean(maxdds)*100:>{W_DD-1}.2f}% '
                f'{np.mean(returns)*100:>{W_RET-1}.2f}%'
            )
            print(
                f'{"Std":>{W_PATH}} '
                f'{np.std(sharpes):>{W_SHARPE}.2f} '
                f'{np.std(calmars):>{W_CALMAR}.2f} '
                f'{np.std(maxdds)*100:>{W_DD-1}.2f}% '
                f'{np.std(returns)*100:>{W_RET-1}.2f}%'
            )

    # ── Section 3: top 5 / bottom 5 ──────────────────────────────────────────
    if show_highlights and len(valid_paths) >= 2:
        print(f"\n{'═' * TOTAL_H}")
        print('TOP 5 PATHS BY SHARPE')
        print(f"{'═' * TOTAL_H}")
        print(hdr);  print(sep)
        for rank, pr in enumerate(top5, 1):
            m, a = _highlight_rows(f'#{rank}', pr)
            print(m);  print(a)

        print(f"\n{'═' * TOTAL_H}")
        print('BOTTOM 5 PATHS BY SHARPE')
        print(f"{'═' * TOTAL_H}")
        print(hdr);  print(sep)
        for rank, pr in enumerate(bot5, 1):
            m, a = _highlight_rows(f'#{rank}', pr)
            print(m);  print(a)

    # ── Section 4: split legend — decode sN notation ──────────────────────────
    if show_split_legend and len(valid_paths) >= 2:
        ref_splits = set()
        for pr in top5 + bot5:
            for _, sid in pr['split_assignments']:
                ref_splits.add(sid)

        all_groups = set(range(N))
        W_SID  = 4
        W_TEST = max(2 * config['k'] + config['k'] - 1 + 2, 8)
        W_DATE = 27

        print(f"\n{'─' * TOTAL_H}")
        print('Split legend  '
              '(gN→sM = group N\'s OOS data came from split M\'s evaluation)')
        print(f"  {'sN':<{W_SID+1}}  {'Tests':<{W_TEST}}  "
              f"{'Test period':<{W_DATE}}  Trained on")
        print(f"  {'─'*(W_SID+1)}  {'─'*W_TEST}  {'─'*W_DATE}  {'─'*32}")

        for sid in sorted(ref_splits):
            sr = sr_map.get(sid)
            if sr is None:
                continue
            tg  = sorted(sr['test_group_indices'])
            tr  = sorted(all_groups - set(tg))
            t_s = bounds[tg[0]][0].date()
            t_e = bounds[tg[-1]][1].date()
            test_str  = ','.join(f'g{g+1}' for g in tg)
            train_str = ','.join(f'g{g+1}' for g in tr)
            print(f"  s{sid+1:<{W_SID}}  {test_str:<{W_TEST}}  "
                  f"{str(t_s) + ' → ' + str(t_e):<{W_DATE}}  {train_str}")

    # ── Section 5: confidence intervals ──────────────────────────────────────
    if show_ci:
        ci = cpcv_confidence_intervals(cpcv_results)
        cpcv_ci_summary(ci)

    # ── Section 6: IS/OOS efficiency ─────────────────────────────────────────
    if show_efficiency:
        eff = cpcv_results.get('efficiency_stats')
        if eff is None:
            pass   # results from older runs that pre-date this feature
        else:
            eff_df   = eff['per_split']
            n_sp     = len(eff_df)
            mean_eff = eff['mean_efficiency']
            std_eff  = eff['std_efficiency']
            med_eff  = eff['median_efficiency']
            pct_pos  = eff['pct_splits_positive_oos']
            pct_half = eff['pct_splits_efficiency_above_half']

            # mean IS Sharpe across all splits (same metric as OOS Sharpe)
            v_is     = eff_df['is_sharpe'].replace([np.inf, -np.inf], np.nan).dropna()
            v_oos    = eff_df['oos_sharpe'].replace([np.inf, -np.inf], np.nan).dropna()
            mean_is  = float(v_is.mean())  if len(v_is)  else float('nan')
            mean_oos = float(v_oos.mean()) if len(v_oos) else float('nan')

            def _feff(v, pct=False):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return 'N/A'
                return f'{v * 100:.1f}%' if pct else f'{v:.3f}'

            W_EFF = 56
            print(f"\n{'═' * W_EFF}")
            print(f'IS/OOS EFFICIENCY  ({n_sp} splits)')
            print(f"{'─' * W_EFF}")
            print(f'  {"Mean IS Sharpe:":<28} {_feff(mean_is)}')
            print(f'  {"Mean OOS Sharpe:":<28} {_feff(mean_oos)}')
            print(f'  {"Mean efficiency:":<28} {_feff(mean_eff)}  (OOS / IS ratio)')
            print(f'  {"Std efficiency:":<28} {_feff(std_eff)}')
            print(f'  {"% splits OOS > 0:":<28} {_feff(pct_pos,  pct=True)}')
            print(f'  {"% splits eff > 0.5:":<28} {_feff(pct_half, pct=True)}')
            print(f"{'─' * W_EFF}")

            consistency = ('Consistent' if (not np.isnan(std_eff)  and std_eff  < 0.25)
                           else 'Volatile')
            generalise  = ('Generalising' if (not np.isnan(mean_eff) and mean_eff > 0.5)
                           else 'Overfitting')
            print(f'  Verdict: {consistency}  |  {generalise}')
            print(f"{'═' * W_EFF}")


# ──────────────────────────────────────────────────────────────────────────────
#  Function 7: cpcv_print_param_suggestions
# ──────────────────────────────────────────────────────────────────────────────

def cpcv_print_param_suggestions(cpcv_results, analysis):
    """
    Print the consensus_ranges table, then two focused copy-pasteable blocks:

      PARAM_DEFS (narrowed)  — only params whose action is 'narrow to IQR',
                               using Q25/Q75 as the new range.
      FIXED_PARAMS           — params whose action starts with 'fix at' (newly
                               converged) plus any params already in fixed_params.

    Params with 'keep current range' are omitted — their existing lines are fine.
    """
    param_defs   = cpcv_results.get('param_defs', {})
    fixed_params = cpcv_results.get('fixed_params', {})
    cr           = analysis['consensus_ranges']
    ds           = analysis['distribution_stats']

    print(cr.to_string())
    print()

    narrowed = {}   # param → (type_str, lo, hi)   — only 'narrow to IQR'
    to_fix   = {}   # param → (type_str, value)     — 'fix at X' from free params

    for p in cr.index:
        action = cr.loc[p, 'action']
        ptype  = param_defs[p][0] if p in param_defs else 'float'

        if action.startswith('fix at'):
            med = float(ds.loc[p, 'median']) if p in ds.index else 0.0
            val = int(round(med)) if ptype == 'int' else med
            to_fix[p] = (ptype, val)
        elif action == 'narrow to IQR':
            lo = float(cr.loc[p, 'recommended_low'])
            hi = float(cr.loc[p, 'recommended_high'])
            if ptype == 'int':
                lo, hi = int(round(lo)), int(round(hi))
            narrowed[p] = (ptype, lo, hi)

    def _vfmt(v, ptype):
        return str(int(round(v))) if ptype == 'int' else f'{float(v):.4g}'

    # ── narrowed PARAM_DEFS lines ─────────────────────────────────────────────
    if narrowed:
        max_k = max(len(p) for p in narrowed)
        pad   = max_k + 3
        print('# PARAM_DEFS — replace these lines (narrowed to IQR):')
        for p, (ptype, lo, hi) in narrowed.items():
            key_s  = f"'{p}':"
            type_s = f"'{ptype}',"
            print(f"    {key_s:<{pad}}  ({type_s:<9} {_vfmt(lo, ptype):>6},  {_vfmt(hi, ptype):>6}),")
    else:
        print('# PARAM_DEFS — no ranges narrowed.')

    # ── FIXED_PARAMS block (newly converged + already fixed) ─────────────────
    all_fixed = {}
    for p, v in fixed_params.items():
        ptype = param_defs[p][0] if p in param_defs else 'float'
        all_fixed[p] = (ptype, v)
    for p, tv in to_fix.items():
        all_fixed[p] = tv   # newly converged overrides if somehow duplicated

    if all_fixed:
        max_k = max(len(p) for p in all_fixed)
        pad   = max_k + 3
        print()
        print('# FIXED_PARAMS:')
        for p, (ptype, val) in all_fixed.items():
            key_s = f"'{p}':"
            print(f"    {key_s:<{pad}}  {_vfmt(val, ptype)},")
    else:
        print('# FIXED_PARAMS — none.')
