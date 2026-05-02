"""
CPCV Portfolio Engine
──────────────────────────────────────────────────────────────────────────────
Combines per-asset CPCV results into a portfolio-level return distribution
via random path sampling.  Companion to cpcv_engine.py.

Typical usage
─────────────
    from cpcv_portfolio import (
        load_asset_cpcv,
        sample_portfolio_paths,
        portfolio_confidence_intervals,
        portfolio_summary,
        per_asset_split_heatmaps,
    )

    assets  = load_asset_cpcv({'BTC': 'btcusdt_cpcv.pkl',
                                'ETH': 'ethusdt_cpcv.pkl',
                                'SOL': 'solusdt_cpcv.pkl'})
    weights = {'BTC': 0.40, 'ETH': 0.35, 'SOL': 0.25}
    paths   = sample_portfolio_paths(assets, weights, n_samples=2000)
    ci      = portfolio_confidence_intervals(paths, assets)
    portfolio_summary(paths, ci, weights, asset_results=assets)
    heatmaps = per_asset_split_heatmaps(paths, assets)
"""

import numpy as np
import pandas as pd

try:
    from scipy.stats import t as _t_dist
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    _t_dist    = None


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _infer_periods_per_year(index):
    """
    Infer the annualisation factor from a DatetimeIndex.
    Mirrors performance_metrics.infer_frequency() exactly so that Sharpe and
    Calmar formulas stay consistent with the backtester.
    """
    if len(index) < 2:
        return 365
    time_diffs  = index.to_series().diff().dropna()
    median_diff = time_diffs.median()
    hours       = median_diff.total_seconds() / 3600
    if hours <= 1:
        return 8760
    elif hours <= 4:
        return 2190
    elif hours <= 24:
        return 365
    elif hours <= 168:
        return 52
    return 12


def _compute_path_metrics(returns: pd.Series, periods_per_year: int):
    """
    Compute (total_return, sharpe, calmar, max_dd, equity_curve) from a net-return
    Series.

    Formulas replicate performance_metrics.calculate_all_metrics() exactly:
      equity         = (1 + returns.fillna(0)).cumprod()
      total_return   = equity[-1] - 1
      sharpe         = (mean / std) * sqrt(periods_per_year)
      max_dd         = min((equity - running_max) / running_max)
      calmar         = annualised_return / |max_dd|
                       where annualised_return = (1+total_return)^(1/n_years) - 1
    """
    r      = returns.fillna(0)
    equity = (1 + r).cumprod()
    n      = len(r)

    total_return = float(equity.iloc[-1] - 1)

    std    = float(r.std())
    sharpe = (float(r.mean() / std) * np.sqrt(periods_per_year)
              if (n >= 2 and std > 0) else 0.0)

    running_max = equity.cummax()
    max_dd      = float(((equity - running_max) / running_max).min())

    n_years = n / periods_per_year
    if n_years > 0 and max_dd != 0:
        ann_return = (1 + total_return) ** (1.0 / n_years) - 1
        calmar     = float(ann_return / abs(max_dd))
    else:
        calmar = 0.0

    return total_return, sharpe, calmar, max_dd, equity


def _fmt(val, pct=False, dp=2):
    """Format a float for printing; None or NaN → 'N/A'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'N/A'
    return f'{val * 100:.{dp}f}%' if pct else f'{val:.{dp}f}'


# ──────────────────────────────────────────────────────────────────────────────
#  Function 1: load_asset_cpcv
# ──────────────────────────────────────────────────────────────────────────────

def load_asset_cpcv(pkl_paths: dict) -> dict:
    """
    Load and cross-validate per-asset CPCV result dicts.

    Parameters
    ----------
    pkl_paths : dict
        {"ASSET": "path/to/asset_cpcv.pkl", ...}

    Validation (raises ValueError on mismatch)
    ------------------------------------------
    All assets must have the same:
      - N (number of groups, from config['N'])
      - n_splits
      - n_paths
      - group_boundaries (each boundary within 1-day calendar tolerance)

    Returns
    -------
    dict mapping asset name → full cpcv_results dict (as returned by run_cpcv()).
    """
    asset_results: dict = {}
    for asset, path in pkl_paths.items():
        asset_results[asset] = pd.read_pickle(path)

    # ── single-asset shortcut ─────────────────────────────────────────────────
    if len(asset_results) <= 1:
        for asset, results in asset_results.items():
            n_valid = sum(1 for p in results['paths'] if p['equity_curve'] is not None)
            print(
                f'Loaded 1 asset: {asset}  N={results["config"]["N"]}  '
                f'splits={results["n_splits"]}  paths={results["n_paths"]}  '
                f'valid_paths={n_valid}'
            )
        return asset_results

    # ── reference values from first asset ────────────────────────────────────
    ref_asset    = next(iter(asset_results))
    ref          = asset_results[ref_asset]
    ref_N        = ref['config']['N']
    ref_n_splits = ref['n_splits']
    ref_n_paths  = ref['n_paths']
    ref_bounds   = ref['group_boundaries']

    for asset, results in asset_results.items():
        if asset == ref_asset:
            continue

        N = results['config']['N']
        if N != ref_N:
            raise ValueError(
                f"Asset '{asset}' has N={N} groups but '{ref_asset}' has N={ref_N}."
            )

        n_splits = results['n_splits']
        if n_splits != ref_n_splits:
            raise ValueError(
                f"Asset '{asset}' has {n_splits} splits but "
                f"'{ref_asset}' has {ref_n_splits}."
            )

        n_paths = results['n_paths']
        if n_paths != ref_n_paths:
            raise ValueError(
                f"Asset '{asset}' has {n_paths} paths but "
                f"'{ref_asset}' has {ref_n_paths}."
            )

        # Calendar boundary check: assets with different data histories have
        # different group start/end dates — this is expected and handled by the
        # inner-join in sample_portfolio_paths.  We still flag large systematic
        # mismatches (> 1 day on every group boundary) which would indicate
        # that CPCV was run with a different N or k on the same dataset.
        bounds = results['group_boundaries']
        if len(bounds) != len(ref_bounds):
            raise ValueError(
                f"Asset '{asset}' has {len(bounds)} group boundaries but "
                f"'{ref_asset}' has {len(ref_bounds)}."
            )

        mismatches = []
        for i, ((s, e), (rs, re)) in enumerate(zip(bounds, ref_bounds)):
            s_diff = abs((pd.Timestamp(s) - pd.Timestamp(rs)).days)
            e_diff = abs((pd.Timestamp(e) - pd.Timestamp(re)).days)
            if s_diff > 1 or e_diff > 1:
                mismatches.append(i + 1)

        if len(mismatches) == len(bounds):
            # Every single group boundary mismatches — likely a structural
            # difference (different data window or incompatible N/k run).
            # Warn rather than hard-error so users can override if intentional.
            print(
                f"  [WARNING] '{asset}' group boundaries do not calendar-align "
                f"with '{ref_asset}' (all {len(bounds)} groups differ by > 1 day). "
                f"This is normal when assets have different data histories. "
                f"Paths will be aligned via inner-join on dates."
            )
        elif mismatches:
            print(
                f"  [WARNING] '{asset}' group(s) {mismatches} boundary dates "
                f"differ from '{ref_asset}' by > 1 day."
            )

    # ── summary ───────────────────────────────────────────────────────────────
    print(
        f'Loaded {len(asset_results)} assets  '
        f'(N={ref_N}  splits={ref_n_splits}  paths={ref_n_paths}):'
    )
    for asset, results in asset_results.items():
        n_valid = sum(1 for p in results['paths'] if p['equity_curve'] is not None)
        print(
            f'  {asset:<8}  valid_paths={n_valid}/{results["n_paths"]}'
        )

    return asset_results


# ──────────────────────────────────────────────────────────────────────────────
#  Function 2: sample_portfolio_paths
# ──────────────────────────────────────────────────────────────────────────────

def sample_portfolio_paths(
    asset_results : dict,
    weights       : dict,
    n_samples     : int = 2000,
    seed          : int = 42,
) -> list:
    """
    Build a portfolio return distribution by randomly combining per-asset CPCV paths.

    For each of n_samples draws:
      1. One valid path is independently sampled for each asset.
      2. Each path's equity_curve is converted to bar returns via pct_change().
      3. Returns are inner-joined on the common date index.
      4. Portfolio returns = weighted sum of per-asset returns.
      5. Portfolio metrics are computed using the same formulas as cpcv_engine.py
         (Sharpe, Calmar, MaxDD, TotalReturn from performance_metrics.py).

    Parameters
    ----------
    asset_results : dict returned by load_asset_cpcv()
    weights       : {"ASSET": weight, ...}  must sum to 1.0
    n_samples     : number of random portfolio samples  (default 2000)
    seed          : RNG seed  (default 42)

    Returns
    -------
    list of dicts, each with:
        "portfolio_returns"       : pd.Series  — bar-level net returns
        "equity_curve"            : pd.Series  — (1+returns).cumprod() × 100
        "sharpe"                  : float
        "calmar"                  : float
        "max_dd"                  : float
        "total_return"            : float
        "asset_path_indices"      : {asset: path_index}
        "asset_split_assignments" : {asset: split_assignments list from that path}
    """
    # ── validate ──────────────────────────────────────────────────────────────
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(
            f'weights must sum to 1.0; got {total_w:.8f}'
        )
    for asset in weights:
        if asset not in asset_results:
            raise ValueError(
                f"Asset '{asset}' is in weights but not in asset_results."
            )

    # ── pre-collect valid path indices per asset ──────────────────────────────
    valid_indices: dict = {}
    for asset in weights:
        pool = [
            i for i, p in enumerate(asset_results[asset]['paths'])
            if p['equity_curve'] is not None
        ]
        if not pool:
            raise ValueError(
                f"Asset '{asset}' has no paths with a valid equity_curve."
            )
        valid_indices[asset] = pool

    rng             = np.random.default_rng(seed)
    portfolio_paths = []
    n_skipped       = 0

    for _ in range(n_samples):
        # ── 1. draw one valid path index per asset ────────────────────────────
        asset_path_indices:      dict = {}
        asset_split_assignments: dict = {}
        returns_by_asset:        dict = {}

        for asset in weights:
            pool     = valid_indices[asset]
            path_idx = int(pool[rng.integers(0, len(pool))])
            path     = asset_results[asset]['paths'][path_idx]

            asset_path_indices[asset]      = path_idx
            asset_split_assignments[asset] = path['split_assignments']

            # equity_curve is cumulative (starts near 1.0); convert to bar returns.
            # First bar of pct_change is NaN → fill with 0 (no return on day 0).
            returns_by_asset[asset] = path['equity_curve'].pct_change().fillna(0)

        # ── 2. inner-join on common date index ────────────────────────────────
        common_idx = next(iter(returns_by_asset.values())).index
        for r in returns_by_asset.values():
            common_idx = common_idx.intersection(r.index)

        if len(common_idx) == 0:
            n_skipped += 1
            continue

        # ── 3. weighted portfolio returns ─────────────────────────────────────
        port_returns = pd.Series(0.0, index=common_idx)
        for asset, r in returns_by_asset.items():
            port_returns = port_returns + weights[asset] * r.reindex(common_idx).fillna(0)

        # ── 4. compute metrics (mirrors performance_metrics.py exactly) ───────
        ppy = _infer_periods_per_year(common_idx)
        total_return, sharpe, calmar, max_dd, equity = _compute_path_metrics(
            port_returns, ppy
        )

        portfolio_paths.append({
            'portfolio_returns':       port_returns,
            'equity_curve':            equity * 100,   # 1-based → 100-based
            'sharpe':                  sharpe,
            'calmar':                  calmar,
            'max_dd':                  max_dd,
            'total_return':            total_return,
            'asset_path_indices':      asset_path_indices,
            'asset_split_assignments': asset_split_assignments,
        })

    if n_skipped:
        print(
            f'[sample_portfolio_paths] {n_skipped} sample(s) skipped '
            f'(empty date intersection).  '
            f'{len(portfolio_paths)} portfolio paths collected.'
        )
    else:
        print(
            f'[sample_portfolio_paths] {len(portfolio_paths)} portfolio paths sampled.'
        )

    return portfolio_paths


# ──────────────────────────────────────────────────────────────────────────────
#  Function 3: portfolio_confidence_intervals
# ──────────────────────────────────────────────────────────────────────────────

def portfolio_confidence_intervals(
    portfolio_paths : list,
    asset_results   : dict,
    confidence      : float = 0.95,
) -> dict:
    """
    Estimate N_eff-adjusted confidence intervals on portfolio Sharpe, Calmar, MaxDD.

    Overlap definition
    ------------------
    For a pair of portfolio paths (i, j), the overlap is the average across all
    assets of the weighted per-asset split overlap:

        per_asset_overlap(asset) = |shared split_ids| / splits_per_path
        overlap(i, j)            = mean(per_asset_overlap) over all assets

    This matches cpcv_confidence_intervals() in cpcv_engine.py and
    _individual_floor() in coin_inclusion_analysis() exactly, so individual
    and portfolio Sharpe floors are directly comparable.

    N_eff estimation
    ----------------
    Computing the full N × N overlap matrix is O(N²).  Instead, 1 000 random
    pairs (i, j) are sampled (with replacement; diagonal i==j counts as overlap)
    and the mean overlap rate is used to estimate:

        sum(overlap_matrix) ≈ N² × mean_overlap_rate
        N_eff               = N² / sum(overlap_matrix)

    Returns a dict with the same structure as cpcv_confidence_intervals()
    in cpcv_engine.py:

        {
          "sharpe": {
              "values", "mean", "std", "n_paths", "n_effective",
              "naive_ci", "adjusted_ci", "conservative_lower_bound"
          },
          "calmar": <same>,
          "max_dd": <same>,
          "confidence": float
        }
    """
    valid = [
        p for p in portfolio_paths
        if p.get('sharpe') is not None and not np.isnan(p['sharpe'])
    ]

    if not valid:
        return {
            'sharpe': None, 'calmar': None, 'max_dd': None,
            'confidence': confidence,
        }

    N       = len(valid)
    sharpes = np.array([p['sharpe'] for p in valid], dtype=float)
    calmars = np.array(
        [p['calmar'] if p.get('calmar') is not None else np.nan for p in valid],
        dtype=float,
    )
    maxdds  = np.array(
        [p['max_dd'] if p.get('max_dd') is not None else np.nan for p in valid],
        dtype=float,
    )

    # ── precompute split-id frozensets per asset per sampled path ─────────────
    # path_split_sets[i][asset] = frozenset of split_ids used in that portfolio path
    path_split_sets = []
    for p in valid:
        entry: dict = {}
        for asset, sa in p['asset_split_assignments'].items():
            entry[asset] = frozenset(sid for _, sid in sa)
        path_split_sets.append(entry)

    # ── estimate N_eff from 1 000 random pair samples ─────────────────────────
    # Overlap definition: weighted average per-asset overlap fraction.
    # For each asset, overlap(i,j) = |shared split_ids| / splits_per_path.
    # The portfolio overlap is the mean of these per-asset values.
    #
    # This matches the formula in _individual_floor() / cpcv_confidence_intervals()
    # exactly, so individual and portfolio Sharpe floors are directly comparable.
    # The old binary-OR approach (any asset sharing any split = full overlap) gave
    # mean_overlap → 1 for multi-asset portfolios → N_eff ≈ 1 → uninformative CI.
    n_pairs = 1000
    rng     = np.random.default_rng(0)   # fixed seed for reproducible N_eff
    i_idx   = rng.integers(0, N, size=n_pairs)
    j_idx   = rng.integers(0, N, size=n_pairs)

    # Precompute splits_per_path per asset (max frozenset size across all portfolio paths)
    all_assets = list({a for ps in path_split_sets for a in ps})
    spp: dict = {}    # splits_per_path per asset
    for asset in all_assets:
        mx = max((len(ps.get(asset, frozenset())) for ps in path_split_sets), default=1)
        spp[asset] = max(mx, 1)

    overlap_sum = 0.0
    for k in range(n_pairs):
        i, j = int(i_idx[k]), int(j_idx[k])
        if i == j:
            overlap_sum += 1.0
            continue
        ps_i = path_split_sets[i]
        ps_j = path_split_sets[j]
        per_asset = []
        for asset in ps_i:
            if asset in ps_j:
                n_shared = len(ps_i[asset] & ps_j[asset])
                per_asset.append(n_shared / spp[asset])
        overlap_sum += (sum(per_asset) / len(per_asset)) if per_asset else 0.0

    mean_overlap = overlap_sum / n_pairs
    sum_overlap  = N ** 2 * max(mean_overlap, 1e-10)
    n_eff        = float(N ** 2) / sum_overlap

    # ── t critical values ─────────────────────────────────────────────────────
    alpha     = 1.0 - confidence
    _fallback = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    _z        = _fallback.get(round(confidence, 2), 1.960)

    if _HAS_SCIPY:
        t_naive = float(_t_dist.ppf(1 - alpha / 2, df=max(N - 1,     1)))
        t_eff   = float(_t_dist.ppf(1 - alpha / 2, df=max(n_eff - 1, 1)))
    else:
        t_naive = t_eff = _z

    def _ci_dict(vals: np.ndarray) -> dict:
        ok   = vals[~np.isnan(vals)]
        n    = len(ok)
        mean = float(np.mean(ok))
        std  = float(np.std(ok, ddof=1)) if n > 1 else 0.0
        sem  = std / np.sqrt(n)     if n > 0     else 0.0
        se_e = std / np.sqrt(n_eff) if n_eff > 0 else 0.0

        naive_ci    = (mean - t_naive * sem,  mean + t_naive * sem)
        adjusted_ci = (mean - t_eff   * se_e, mean + t_eff   * se_e)

        return {
            'values':                   vals,
            'mean':                     mean,
            'std':                      float(np.std(ok)) if n > 1 else 0.0,
            'n_paths':                  n,
            'n_effective':              n_eff,
            'naive_ci':                 naive_ci,
            'adjusted_ci':              adjusted_ci,
            'conservative_lower_bound': adjusted_ci[0],
        }

    return {
        'sharpe':     _ci_dict(sharpes),
        'calmar':     _ci_dict(calmars),
        'max_dd':     _ci_dict(maxdds),
        'confidence': confidence,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Function 4: portfolio_summary
# ──────────────────────────────────────────────────────────────────────────────

def portfolio_summary(
    portfolio_paths : list,
    ci_results      : dict,
    weights         : dict,
    asset_results   : dict = None,
):
    """
    Print three tables matching the style of cpcv_summary() / cpcv_ci_summary()
    in cpcv_engine.py.

    Table 1 — Portfolio path distribution
        Percentile rows (Mean / Std / Min / Q25 / Median / Q75 / Max / IQR)
        across Sharpe, Calmar, Max DD, Total Return.

    Table 2 — Confidence intervals (95% by default)
        Naive (N = n_samples) and overlap-adjusted (N_eff) CIs for Sharpe,
        Calmar, and Max DD, plus conservative floor annotations.

    Table 3 — Asset weights
        Asset | Weight | Median path Sharpe | Normalised contribution
        (contribution = weight × median_sharpe / Σ(w_j × median_sharpe_j))
        Median path Sharpe is computed from asset_results when provided;
        the column shows 'N/A' otherwise.

    Parameters
    ----------
    portfolio_paths : list returned by sample_portfolio_paths()
    ci_results      : dict returned by portfolio_confidence_intervals()
    weights         : {"ASSET": weight, ...}
    asset_results   : dict returned by load_asset_cpcv() — optional.
                      Enables the per-asset Median Sharpe column in Table 3.
    """
    valid = [
        p for p in portfolio_paths
        if p.get('sharpe') is not None and not np.isnan(p['sharpe'])
    ]

    sharpes = [p['sharpe']       for p in valid]
    calmars = [p['calmar']       for p in valid if p.get('calmar')       is not None]
    maxdds  = [p['max_dd']       for p in valid if p.get('max_dd')       is not None]
    returns = [p['total_return'] for p in valid if p.get('total_return') is not None]

    W_STAT = 8
    W_COL  = 10
    TOTAL0 = W_STAT + W_COL * 4 + 4
    _nan   = float('nan')

    def _pct_col(vals, q):
        return float(np.percentile(vals, q)) if vals else _nan

    def _stat_row(label, sh, ca, dd, rt):
        sh_s = f'{sh:.2f}'      if isinstance(sh, float) and np.isfinite(sh) else 'N/A'
        ca_s = f'{ca:.2f}'      if isinstance(ca, float) and np.isfinite(ca) else 'N/A'
        dd_s = f'{dd*100:.2f}%' if isinstance(dd, float) and np.isfinite(dd) else 'N/A'
        rt_s = f'{rt*100:.2f}%' if isinstance(rt, float) and np.isfinite(rt) else 'N/A'
        return (
            f'{label:<{W_STAT}} '
            f'{sh_s:>{W_COL}} '
            f'{ca_s:>{W_COL}} '
            f'{dd_s:>{W_COL}} '
            f'{rt_s:>{W_COL}}'
        )

    # ── Table 1: distribution ─────────────────────────────────────────────────
    print(f"\n{'═' * TOTAL0}")
    print(f'PORTFOLIO PATH DISTRIBUTION  ({len(valid)} sampled paths)')
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
        ('Mean',
         np.mean(sharpes) if sharpes else _nan,
         np.mean(calmars) if calmars else _nan,
         np.mean(maxdds)  if maxdds  else _nan,
         np.mean(returns) if returns else _nan),
        ('Std',
         np.std(sharpes)  if sharpes else _nan,
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
        ('IQR',
         (_pct_col(sharpes, 75) - _pct_col(sharpes, 25)) if sharpes else _nan,
         (_pct_col(calmars, 75) - _pct_col(calmars, 25)) if calmars else _nan,
         (_pct_col(maxdds,  75) - _pct_col(maxdds,  25)) if maxdds  else _nan,
         (_pct_col(returns, 75) - _pct_col(returns, 25)) if returns else _nan),
    ]
    for label, sh, ca, dd, rt in _stats:
        print(_stat_row(label, sh, ca, dd, rt))

    # ── Table 2: confidence intervals ─────────────────────────────────────────
    if ci_results is not None and ci_results.get('sharpe') is not None:
        conf    = ci_results.get('confidence', 0.95)
        pct     = int(round(conf * 100))
        sh_ci   = ci_results['sharpe']
        ca_ci   = ci_results.get('calmar')
        dd_ci   = ci_results.get('max_dd')
        n_paths = sh_ci['n_paths']
        n_eff   = sh_ci['n_effective']

        W_TOT  = 72
        W_MET  = 7
        W_MET2 = 23
        W_VAL  = 8
        W_NOTE = 17

        def _fv(v, pct_scale=False):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 'N/A'
            return f'{v * 100:.2f}%' if pct_scale else f'{v:.3f}'

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

        for label, data, as_pct in [
            ('Sharpe', sh_ci, False),
            ('Calmar', ca_ci, False),
            ('Max DD', dd_ci, True),
        ]:
            if data is None:
                continue
            lo_n, hi_n = data['naive_ci']
            lo_a, hi_a = data['adjusted_ci']
            print(
                f'{label:<{W_MET}}  '
                f'{f"Naive (N={n_paths})":<{W_MET2}}  '
                f'{_fv(lo_n, as_pct):>{W_VAL}}  '
                f'{_fv(hi_n, as_pct):>{W_VAL}}  '
                f'anticonservative'
            )
            print(
                f'{"":<{W_MET}}  '
                f'{f"Adjusted (N_eff={n_eff:.1f})":<{W_MET2}}  '
                f'{_fv(lo_a, as_pct):>{W_VAL}}  '
                f'{_fv(hi_a, as_pct):>{W_VAL}}  '
                f'conservative'
            )

        print(f"{'─' * W_TOT}")
        if sh_ci:
            print(
                f"  Conservative Sharpe floor: "
                f"{sh_ci['conservative_lower_bound']:.3f}"
                f"  (adjusted lower bound)"
            )
        if ca_ci:
            print(
                f"  Conservative Calmar floor: "
                f"{ca_ci['conservative_lower_bound']:.3f}"
                f"  (adjusted lower bound)"
            )

    # ── Table 3: asset weights ─────────────────────────────────────────────────
    # Per-asset median Sharpe across all of that asset's CPCV paths
    if asset_results is not None:
        ind_sharpes: dict = {}
        for asset in weights:
            if asset in asset_results:
                path_sh = [
                    p['sharpe']
                    for p in asset_results[asset]['paths']
                    if p.get('sharpe') is not None
                ]
                ind_sharpes[asset] = float(np.median(path_sh)) if path_sh else _nan
            else:
                ind_sharpes[asset] = _nan
    else:
        ind_sharpes = {asset: _nan for asset in weights}

    # Normalised contribution = w_i × med_sh_i / Σ_j(w_j × med_sh_j)
    raw_contrib  = {a: weights[a] * ind_sharpes[a] for a in weights}
    total_c      = sum(
        v for v in raw_contrib.values()
        if isinstance(v, float) and not np.isnan(v)
    )
    norm_contrib = {
        a: raw_contrib[a] / total_c
        if (total_c > 0 and not np.isnan(raw_contrib[a]))
        else _nan
        for a in weights
    }

    W_A   = max(max(len(a) for a in weights), 5) + 2
    W_W   = 8
    W_SH  = 15
    W_CON = 14
    TOTAL_W = W_A + W_W + W_SH + W_CON + 3

    print(f"\n{'═' * TOTAL_W}")
    print('ASSET WEIGHTS')
    print(f"{'═' * TOTAL_W}")
    print(
        f'{"Asset":<{W_A}} '
        f'{"Weight":>{W_W}} '
        f'{"Median Sharpe":>{W_SH}} '
        f'{"Contribution":>{W_CON}}'
    )
    print(
        f'{"─" * W_A} '
        f'{"─" * W_W} '
        f'{"─" * W_SH} '
        f'{"─" * W_CON}'
    )

    for asset in weights:
        sh_s  = (f'{ind_sharpes[asset]:.3f}'
                 if not np.isnan(ind_sharpes[asset]) else 'N/A')
        nc    = norm_contrib.get(asset, _nan)
        con_s = f'{nc * 100:.2f}%' if (isinstance(nc, float) and not np.isnan(nc)) else 'N/A'
        print(
            f'{asset:<{W_A}} '
            f'{weights[asset] * 100:>{W_W}.2f}% '
            f'{sh_s:>{W_SH}} '
            f'{con_s:>{W_CON}}'
        )

    total_nc = sum(
        v for v in norm_contrib.values()
        if isinstance(v, float) and not np.isnan(v)
    )
    total_nc_s = f'{total_nc * 100:.2f}%' if total_nc > 0 else 'N/A'

    print(
        f'{"─" * W_A} '
        f'{"─" * W_W} '
        f'{"─" * W_SH} '
        f'{"─" * W_CON}'
    )
    print(
        f'{"Total":<{W_A}} '
        f'{sum(weights.values()) * 100:>{W_W}.2f}% '
        f'{"":>{W_SH}} '
        f'{total_nc_s:>{W_CON}}'
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Function 5: per_asset_split_heatmaps
# ──────────────────────────────────────────────────────────────────────────────

def per_asset_split_heatmaps(
    portfolio_paths : list,
    asset_results   : dict,
) -> dict:
    """
    For each asset, summarise which group-pair splits drove strong or weak
    portfolio paths.

    For every unique (group_i, group_j) pair that appears as the test groups
    of one of the asset's splits, this function collects all sampled portfolio
    paths where that asset's chosen path used that split, then reports the
    mean portfolio Sharpe across those samples.

    The result feeds into the per-asset heatmap in cpcv_portfolio_visualizer —
    it mirrors the structure that plot_split_performance_heatmap() expects,
    but at portfolio rather than single-asset level.

    Parameters
    ----------
    portfolio_paths : list returned by sample_portfolio_paths()
    asset_results   : dict returned by load_asset_cpcv()

    Returns
    -------
    dict mapping asset name → pd.DataFrame with columns:
        group_i               : int
        group_j               : int
        mean_portfolio_sharpe : float  (NaN if no valid Sharpes in group)
        count                 : int    (number of sampled paths in this group)
    Sorted by (group_i, group_j).
    """
    result: dict = {}

    for asset, results in asset_results.items():
        split_results = results['split_results']

        # ── map each split → its canonical group pair ─────────────────────────
        # For k=2 each split has exactly two test groups — one unique pair.
        # For k>2 we take the first two groups as a representative pair so the
        # output still feeds cleanly into a 2-D heatmap.
        pair_to_split_ids: dict = {}
        for sr in split_results:
            sid  = sr['split_id']
            tg   = sr['test_group_indices']
            pair = tuple(sorted(tg[:2]))        # canonical (i < j)
            pair_to_split_ids.setdefault(pair, set()).add(sid)

        # ── cache per-portfolio-path which split_ids this asset used ──────────
        # asset_split_assignments[asset] = [(group_idx, split_id), ...]
        path_sids_for_asset = [
            frozenset(
                sid
                for _, sid in pp.get('asset_split_assignments', {}).get(asset, [])
            )
            for pp in portfolio_paths
        ]

        # ── for each group pair, collect portfolio Sharpes of matching paths ──
        rows = []
        for (gi, gj), split_ids_for_pair in pair_to_split_ids.items():
            sharpes_for_pair = []
            count            = 0

            for pp, path_sids in zip(portfolio_paths, path_sids_for_asset):
                if path_sids & split_ids_for_pair:
                    count += 1
                    sh = pp.get('sharpe')
                    if sh is not None and not np.isnan(sh):
                        sharpes_for_pair.append(sh)

            rows.append({
                'group_i':               gi,
                'group_j':               gj,
                'mean_portfolio_sharpe': (float(np.mean(sharpes_for_pair))
                                          if sharpes_for_pair else float('nan')),
                'count':                 count,
            })

        result[asset] = (
            pd.DataFrame(rows)
              .sort_values(['group_i', 'group_j'])
              .reset_index(drop=True)
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Function 6: diversification_benefit
# ──────────────────────────────────────────────────────────────────────────────

def diversification_benefit(
    portfolio_paths : list,
    asset_results   : dict,
    weights         : dict,
) -> pd.DataFrame:
    """
    For each sampled portfolio path, compute the diversification benefit as the
    difference between the realised portfolio Sharpe and the weighted-average
    of the individual asset path Sharpes used in that combination.

        diversification_benefit = portfolio_sharpe
                                  − Σ(weight_i × asset_sharpe_i)

    A positive value means the portfolio Sharpe exceeded the weighted blend of
    its components — genuine diversification. Negative means correlation or
    one asset's drag pulled the combined result below the blend.

    Parameters
    ----------
    portfolio_paths : list returned by sample_portfolio_paths()
    asset_results   : dict returned by load_asset_cpcv()
    weights         : {"ASSET": weight, ...}

    Returns
    -------
    pd.DataFrame — one row per valid sampled path — columns:
        path_id, portfolio_sharpe, weighted_avg_component_sharpe,
        diversification_benefit,
        {asset}_sharpe and {asset}_path_idx for each asset.

    Summary statistics are stored in df.attrs:
        mean_benefit, std_benefit, median_benefit, pct_paths_positive
    """
    assets = list(weights.keys())
    rows: list = []

    for path_id, pp in enumerate(portfolio_paths):
        port_sh = pp.get('sharpe')
        if port_sh is None or np.isnan(float(port_sh)):
            continue

        path_indices = pp.get('asset_path_indices', {})

        # per-asset Sharpe from the specific individual paths used
        asset_sh: dict = {}
        for asset in assets:
            idx = path_indices.get(asset)
            if idx is None:
                asset_sh[asset] = float('nan')
                continue
            sh = asset_results[asset]['paths'][idx].get('sharpe')
            asset_sh[asset] = float(sh) if sh is not None else float('nan')

        # weighted average component Sharpe — renormalise over non-nan assets
        valid = [a for a in assets if not np.isnan(asset_sh[a])]
        if not valid:
            continue
        w_sum     = sum(weights[a] for a in valid)
        wa_sharpe = (
            sum(weights[a] * asset_sh[a] for a in valid) / w_sum
            if w_sum > 0 else float('nan')
        )

        row = {
            'path_id':                       path_id,
            'portfolio_sharpe':              float(port_sh),
            'weighted_avg_component_sharpe': wa_sharpe,
            'diversification_benefit':       float(port_sh) - wa_sharpe,
        }
        for asset in assets:
            row[f'{asset}_sharpe']   = asset_sh[asset]
            row[f'{asset}_path_idx'] = path_indices.get(asset)
        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df):
        b = df['diversification_benefit'].dropna()
        df.attrs = {
            'mean_benefit':       float(b.mean()),
            'std_benefit':        float(b.std()),
            'median_benefit':     float(b.median()),
            'pct_paths_positive': float((b > 0).mean()),
        }
    else:
        df.attrs = {
            'mean_benefit': float('nan'), 'std_benefit': float('nan'),
            'median_benefit': float('nan'), 'pct_paths_positive': float('nan'),
        }

    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Function 7: asset_correlation_structure
# ──────────────────────────────────────────────────────────────────────────────

def asset_correlation_structure(
    portfolio_paths : list,
    asset_results   : dict,
    weights         : dict,
) -> dict:
    """
    For each sampled portfolio path, compute pairwise Pearson correlations
    between per-asset daily return series over the common date window.

    Individual asset equity_curves from cpcv_engine.py are cumulative products
    starting near 1.0.  They are converted to bar returns via .pct_change()
    before correlation is computed.  The aligned return DataFrame is .dropna()
    before .corr() so boundary NaNs do not inflate or deflate correlations.

    Parameters
    ----------
    portfolio_paths : list returned by sample_portfolio_paths()
    asset_results   : dict returned by load_asset_cpcv()
    weights         : {"ASSET": weight, ...}  — used only to determine asset list

    Returns
    -------
    dict with three keys:
      "pair_correlations" : pd.DataFrame — one row per (path, asset pair)
          columns: path_id, asset_a, asset_b, correlation
      "summary"           : pd.DataFrame — one row per unique asset pair
          columns: asset_a, asset_b, mean_corr, std_corr, min_corr,
                   q25_corr, median_corr, q75_corr, max_corr
      "high_corr_paths"   : list[int] — path_ids where any pair |corr| > 0.7
    """
    assets = list(weights.keys())
    pairs  = [(assets[i], assets[j])
              for i in range(len(assets))
              for j in range(i + 1, len(assets))]

    pair_rows:       list = []
    high_corr_paths: list = []

    for path_id, pp in enumerate(portfolio_paths):
        path_indices = pp.get('asset_path_indices', {})

        # build per-asset daily return series (1-based equity → pct_change)
        ret_dict: dict = {}
        for asset in assets:
            idx = path_indices.get(asset)
            if idx is None:
                continue
            ec = asset_results[asset]['paths'][idx].get('equity_curve')
            if ec is None:
                continue
            ret_dict[asset] = ec.pct_change()   # first bar will be NaN

        if len(ret_dict) < 2:
            continue

        # align on common date index; drop rows with any NaN (first bar + gaps)
        ret_df = pd.DataFrame(ret_dict).dropna()
        if len(ret_df) < 2:
            continue

        path_high = False
        for asset_a, asset_b in pairs:
            if asset_a not in ret_df.columns or asset_b not in ret_df.columns:
                continue
            corr = float(ret_df[asset_a].corr(ret_df[asset_b]))
            pair_rows.append({
                'path_id':     path_id,
                'asset_a':     asset_a,
                'asset_b':     asset_b,
                'correlation': corr,
            })
            if abs(corr) > 0.7:
                path_high = True

        if path_high:
            high_corr_paths.append(path_id)

    pair_df = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame(
        columns=['path_id', 'asset_a', 'asset_b', 'correlation']
    )

    # summary: per-pair distribution of correlations across all sampled paths
    summary_rows: list = []
    if len(pair_df):
        for (asset_a, asset_b), grp in pair_df.groupby(['asset_a', 'asset_b'], sort=False):
            c = grp['correlation'].dropna()
            summary_rows.append({
                'asset_a':    asset_a,
                'asset_b':    asset_b,
                'mean_corr':  float(c.mean()),
                'std_corr':   float(c.std()),
                'min_corr':   float(c.min()),
                'q25_corr':   float(c.quantile(0.25)),
                'median_corr': float(c.median()),
                'q75_corr':   float(c.quantile(0.75)),
                'max_corr':   float(c.max()),
            })

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame(
        columns=['asset_a', 'asset_b', 'mean_corr', 'std_corr',
                 'min_corr', 'q25_corr', 'median_corr', 'q75_corr', 'max_corr']
    )

    return {
        'pair_correlations': pair_df,
        'summary':           summary_df,
        'high_corr_paths':   high_corr_paths,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Function 8: worst_drawdown_decomposition
# ──────────────────────────────────────────────────────────────────────────────

def worst_drawdown_decomposition(
    portfolio_paths : list,
    asset_results   : dict,
    weights         : dict,
    worst_pct       : float = 0.10,
) -> dict:
    """
    Identify the worst (worst_pct) fraction of portfolio paths by max_dd, then
    decompose each drawdown into per-asset contributions.

    The drawdown period (peak → trough) is identified on the PORTFOLIO equity
    curve.  Each asset's contribution = weight × cumulative return over that
    period.  Contributions are then expressed as a fraction of the total
    portfolio drawdown magnitude so they are directly comparable across paths.

    Parameters
    ----------
    portfolio_paths : list returned by sample_portfolio_paths()
    asset_results   : dict returned by load_asset_cpcv()
    weights         : {"ASSET": weight, ...}
    worst_pct       : fraction of paths to decompose (default 0.10 = worst 10%)

    Returns
    -------
    dict with three keys:
      "worst_paths"    : pd.DataFrame — one row per decomposed path
          columns: path_id, portfolio_max_dd, dd_start, dd_end,
                   dd_duration_bars, {asset}_contribution, {asset}_pct_of_dd
      "summary"        : pd.DataFrame — one row per asset
          columns: asset, mean_pct_of_dd, std_pct_of_dd, max_pct_of_dd
      "primary_driver" : str — asset with highest mean absolute pct_of_dd
    """
    assets = list(weights.keys())
    _empty = pd.DataFrame()

    # collect valid paths with their original indices
    valid_indexed = [
        (idx, pp) for idx, pp in enumerate(portfolio_paths)
        if (pp.get('max_dd') is not None
            and not np.isnan(float(pp['max_dd']))
            and pp.get('equity_curve') is not None)
    ]
    if not valid_indexed:
        return {'worst_paths': _empty, 'summary': _empty, 'primary_driver': None}

    # sort by max_dd ascending (most negative first); take worst fraction
    valid_indexed.sort(key=lambda x: x[1]['max_dd'])
    n_worst = max(1, int(len(valid_indexed) * worst_pct))
    worst   = valid_indexed[:n_worst]

    rows: list = []
    for path_id, pp in worst:
        port_ec      = pp['equity_curve']          # 100-based cumulative
        port_dd      = float(pp['max_dd'])          # negative
        path_indices = pp.get('asset_path_indices', {})

        if len(port_ec) < 2:
            continue

        # ── locate the max-drawdown period on the portfolio equity curve ───────
        running_max  = port_ec.cummax()
        dd_series    = (port_ec - running_max) / running_max
        trough_loc   = int(dd_series.values.argmin())
        trough_date  = port_ec.index[trough_loc]

        # peak: the last date where equity == its running maximum before trough
        pre_trough   = port_ec.iloc[:trough_loc + 1]
        rm_pre       = pre_trough.cummax()
        at_peak_mask = (pre_trough == rm_pre)
        peak_date    = (at_peak_mask[at_peak_mask].index[-1]
                        if at_peak_mask.any() else port_ec.index[0])
        peak_loc     = port_ec.index.get_loc(peak_date)
        if isinstance(peak_loc, slice):
            peak_loc = peak_loc.start
        elif isinstance(peak_loc, np.ndarray):
            peak_loc = int(np.where(peak_loc)[0][0])
        dd_duration  = trough_loc - int(peak_loc)

        # date window that covers the drawdown (used to slice asset returns)
        dd_dates = port_ec.index[int(peak_loc) : trough_loc + 1]

        # ── per-asset return contribution over the drawdown window ─────────────
        asset_contribs:  dict = {}
        asset_pct_of_dd: dict = {}

        for asset in assets:
            idx = path_indices.get(asset)
            if idx is None:
                asset_contribs[asset]  = float('nan')
                asset_pct_of_dd[asset] = float('nan')
                continue

            ec_asset = asset_results[asset]['paths'][idx].get('equity_curve')
            if ec_asset is None:
                asset_contribs[asset]  = float('nan')
                asset_pct_of_dd[asset] = float('nan')
                continue

            # asset equity_curve is 1-based cumulative → convert to bar returns
            asset_rets = ec_asset.pct_change().fillna(0)
            # align to the portfolio DD window; missing dates → 0 return
            rets_window = asset_rets.reindex(dd_dates).fillna(0)
            cum_ret     = float((1 + rets_window).cumprod().iloc[-1] - 1) \
                          if len(rets_window) else 0.0
            asset_contribs[asset] = weights[asset] * cum_ret

        # express contributions as fraction of total DD magnitude
        abs_dd = abs(port_dd) if port_dd != 0 else float('nan')
        for asset in assets:
            c = asset_contribs[asset]
            asset_pct_of_dd[asset] = (
                c / abs_dd
                if (not np.isnan(abs_dd) and not np.isnan(c)) else float('nan')
            )

        row = {
            'path_id':          path_id,
            'portfolio_max_dd': port_dd,
            'dd_start':         peak_date,
            'dd_end':           trough_date,
            'dd_duration_bars': dd_duration,
        }
        for asset in assets:
            row[f'{asset}_contribution'] = asset_contribs[asset]
            row[f'{asset}_pct_of_dd']    = asset_pct_of_dd[asset]
        rows.append(row)

    worst_df = pd.DataFrame(rows)

    # ── summary: per-asset statistics across all decomposed paths ─────────────
    summary_rows: list = []
    for asset in assets:
        col  = f'{asset}_pct_of_dd'
        vals = worst_df[col].dropna() if col in worst_df.columns else pd.Series(dtype=float)
        summary_rows.append({
            'asset':          asset,
            'mean_pct_of_dd': float(vals.mean())        if len(vals) else float('nan'),
            'std_pct_of_dd':  float(vals.std())         if len(vals) > 1 else 0.0,
            'max_pct_of_dd':  float(vals.abs().max())   if len(vals) else float('nan'),
        })
    summary_df = pd.DataFrame(summary_rows)

    # primary driver: asset with largest mean absolute contribution
    primary_driver = None
    if len(summary_df):
        abs_means = summary_df['mean_pct_of_dd'].abs()
        if abs_means.notna().any():
            primary_driver = str(summary_df.loc[abs_means.idxmax(), 'asset'])

    return {
        'worst_paths':    worst_df,
        'summary':        summary_df,
        'primary_driver': primary_driver,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Function 9: optimal_weights
# ──────────────────────────────────────────────────────────────────────────────

def optimal_weights(
    asset_results    : dict,
    n_samples        : int   = 2000,
    seed             : int   = 42,
    n_weight_configs : int   = 500,
    min_weight       : float = 0.05,
    max_weight       : float = 0.60,
    objectives       : list  = None,
) -> dict:
    """
    Find optimal portfolio weights by evaluating many random weight
    configurations against the sampled CPCV path distribution.

    For each configuration, sample_portfolio_paths and
    portfolio_confidence_intervals are called to compute:
      mean_sharpe, median_sharpe, conservative_sharpe_floor (adj CI lower),
      mean_calmar, conservative_calmar_floor, mean_max_dd,
      sharpe_std, pct_paths_positive_sharpe.

    Weight vectors are generated by Dirichlet sampling clipped to
    [min_weight, max_weight] and renormalised; configs outside the feasible
    region are resampled.  The equal-weight vector is always included.

    **Computational cost** — n_weight_configs × n_samples path evaluations.
    With n_weight_configs=500 and n_samples=2000 that is ~1 million path
    evaluations.  Progress is printed every 50 configs.

    Parameters
    ----------
    asset_results    : dict returned by load_asset_cpcv()
    n_samples        : portfolio paths per weight configuration
    seed             : RNG seed — passed to both weight generation and
                       sample_portfolio_paths for reproducibility
    n_weight_configs : number of random weight vectors to evaluate
    min_weight       : minimum weight per asset (e.g. 0.05 = 5%)
    max_weight       : maximum weight per asset (e.g. 0.60 = 60%)
    objectives       : list of objective names to optimise; default:
                       ['max_conservative_sharpe', 'max_sharpe_calmar',
                        'min_drawdown', 'max_equal_weight_comparison']

    Returns
    -------
    dict with three keys:
      "all_configs"         : pd.DataFrame — one row per weight config with all metrics
      "optimal"             : dict mapping objective → dict with weights + metrics
      "equal_weight_result" : dict — metrics for the equal-weight baseline
    """
    import io, contextlib

    if objectives is None:
        objectives = [
            'max_conservative_sharpe',
            'max_sharpe_calmar',
            'min_drawdown',
            'max_equal_weight_comparison',
        ]

    assets   = list(asset_results.keys())
    n_assets = len(assets)
    rng      = np.random.default_rng(seed)

    # ── generate weight configs ───────────────────────────────────────────────
    def _sample_one():
        """Return one weight dict satisfying [min_weight, max_weight] per asset."""
        for _ in range(2000):
            w = rng.dirichlet(np.ones(n_assets))
            w = np.clip(w, min_weight, max_weight)
            w = w / w.sum()
            if np.all(w >= min_weight - 1e-9) and np.all(w <= max_weight + 1e-9):
                return dict(zip(assets, w.tolist()))
        # fallback: equal weight (always feasible when min_weight <= 1/n_assets)
        return {a: 1.0 / n_assets for a in assets}

    equal_w = {a: 1.0 / n_assets for a in assets}

    # deduplicate: collect configs and always include equal weight
    configs: list = [equal_w]
    seen_equal = True  # equal weight is always slot 0

    while len(configs) < n_weight_configs:
        w = _sample_one()
        configs.append(w)

    # ── evaluate each config ──────────────────────────────────────────────────
    rows:              list = []
    equal_weight_row:  dict = {}
    best_cons_sharpe        = float('-inf')

    _sink = io.StringIO()   # suppress per-call prints from sample_portfolio_paths

    for cfg_idx, w_cfg in enumerate(configs):
        with contextlib.redirect_stdout(_sink):
            p_paths = sample_portfolio_paths(
                asset_results, w_cfg,
                n_samples=n_samples, seed=seed,
            )
            ci = portfolio_confidence_intervals(p_paths, asset_results)

        sh_ci  = ci.get('sharpe') or {}
        ca_ci  = ci.get('calmar') or {}
        dd_ci  = ci.get('max_dd') or {}

        sharpes = [p['sharpe'] for p in p_paths if p.get('sharpe') is not None]
        n_pos   = sum(1 for s in sharpes if s > 0)

        mean_sh   = sh_ci.get('mean',   float('nan'))
        cons_sh   = sh_ci.get('conservative_lower_bound', float('nan'))
        mean_ca   = ca_ci.get('mean',   float('nan'))
        cons_ca   = ca_ci.get('conservative_lower_bound', float('nan'))
        mean_dd   = dd_ci.get('mean',   float('nan'))
        sh_std    = sh_ci.get('std',    float('nan'))
        med_sh    = float(np.median(sharpes)) if sharpes else float('nan')
        pct_pos   = n_pos / len(sharpes) if sharpes else float('nan')

        row = {
            'config_idx':                  cfg_idx,
            'mean_sharpe':                 mean_sh,
            'median_sharpe':               med_sh,
            'conservative_sharpe_floor':   cons_sh,
            'mean_calmar':                 mean_ca,
            'conservative_calmar_floor':   cons_ca,
            'mean_max_dd':                 mean_dd,
            'sharpe_std':                  sh_std,
            'pct_paths_positive_sharpe':   pct_pos,
        }
        for asset in assets:
            row[f'w_{asset}'] = w_cfg[asset]
        rows.append(row)

        if cfg_idx == 0:
            equal_weight_row = row.copy()
            equal_weight_row['weights'] = equal_w

        if not np.isnan(cons_sh):
            best_cons_sharpe = max(best_cons_sharpe, cons_sh)

        if (cfg_idx + 1) % 50 == 0:
            print(
                f'Config {cfg_idx + 1}/{len(configs)}'
                f' | best conservative Sharpe so far: '
                f'{best_cons_sharpe:.2f}'
            )

    all_df = pd.DataFrame(rows)

    # ── find optimal for each objective ──────────────────────────────────────
    def _row_to_result(r):
        w = {a: float(r[f'w_{a}']) for a in assets}
        return {
            'weights':                    w,
            'conservative_sharpe_floor':  float(r['conservative_sharpe_floor']),
            'mean_sharpe':                float(r['mean_sharpe']),
            'mean_calmar':                float(r['mean_calmar']),
            'mean_max_dd':                float(r['mean_max_dd']),
            'sharpe_std':                 float(r['sharpe_std']),
        }

    optimal: dict = {}
    for obj in objectives:
        if obj == 'max_conservative_sharpe':
            idx = all_df['conservative_sharpe_floor'].idxmax()
        elif obj == 'max_sharpe_calmar':
            all_df['_sc'] = all_df['mean_sharpe'] * all_df['mean_calmar']
            idx = all_df['_sc'].idxmax()
            all_df.drop(columns=['_sc'], inplace=True)
        elif obj == 'min_drawdown':
            # max_dd is negative; maximise (least negative) = smallest absolute DD
            idx = all_df['mean_max_dd'].idxmax()
        elif obj == 'max_equal_weight_comparison':
            idx = 0   # equal weight is always config 0
        else:
            continue
        optimal[obj] = _row_to_result(all_df.loc[idx])

    # ── print summary table ───────────────────────────────────────────────────
    W_OBJ  = 26
    W_W    = max(len(assets) * 9, 30)
    W_CS   = 18
    W_MS   = 13
    W_DD   = 10
    TOTAL  = W_OBJ + W_W + W_CS + W_MS + W_DD + 4

    def _w_str(w_dict):
        return ' '.join(f'{a}:{w_dict[a]:.2f}' for a in assets)

    obj_labels = {
        'max_conservative_sharpe':    'Max conservative Sharpe',
        'max_sharpe_calmar':          'Max Sharpe×Calmar',
        'min_drawdown':               'Min drawdown',
        'max_equal_weight_comparison': 'Equal weight',
    }

    print(f"\n{'═' * TOTAL}")
    print('OPTIMAL WEIGHTS SEARCH RESULTS')
    print(f"{'═' * TOTAL}")
    print(
        f'{"Objective":<{W_OBJ}} '
        f'{"Weights":<{W_W}} '
        f'{"Cons. Sharpe":>{W_CS}} '
        f'{"Mean Sharpe":>{W_MS}} '
        f'{"Mean DD":>{W_DD}}'
    )
    print(
        f'{"─" * W_OBJ} {"─" * W_W} '
        f'{"─" * W_CS} {"─" * W_MS} {"─" * W_DD}'
    )
    for obj, res in optimal.items():
        label  = obj_labels.get(obj, obj)
        w_s    = _w_str(res['weights'])
        cs_s   = _fmt(res['conservative_sharpe_floor'])
        ms_s   = _fmt(res['mean_sharpe'])
        dd_s   = _fmt(res['mean_max_dd'], pct=True)
        print(
            f'{label:<{W_OBJ}} '
            f'{w_s:<{W_W}} '
            f'{cs_s:>{W_CS}} '
            f'{ms_s:>{W_MS}} '
            f'{dd_s:>{W_DD}}'
        )

    equal_result = equal_weight_row.copy()
    equal_result['weights'] = equal_w

    return {
        'all_configs':         all_df,
        'optimal':             optimal,
        'equal_weight_result': equal_result,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Function 9: per_asset_yearly_breakdown
# ──────────────────────────────────────────────────────────────────────────────

def per_asset_yearly_breakdown(
    portfolio_paths : list,
    asset_results   : dict,
    weights         : dict,
) -> dict:
    """
    Calendar-year performance breakdown for each asset and the portfolio.

    For every sampled portfolio path:
      - Each asset's equity_curve (1-based, from asset_results) is converted
        to bar returns via pct_change, then sliced to each calendar year to
        compute year_return, year_sharpe, and year_max_dd.
      - Portfolio yearly stats are derived from portfolio_returns (already
        stored in each portfolio path dict).

    Parameters
    ----------
    portfolio_paths : list returned by sample_portfolio_paths()
    asset_results   : dict returned by load_asset_cpcv()
    weights         : {"ASSET": weight, ...} — determines asset list

    Returns
    -------
    dict with three keys:

    "asset_year_summary" : pd.DataFrame — one row per (asset, year)
        asset, year, mean_return, std_return, median_return,
        pct_paths_positive, mean_sharpe, mean_max_dd

    "portfolio_year_summary" : pd.DataFrame — one row per year
        year, mean_return, std_return, median_return,
        pct_paths_positive, mean_sharpe

    "asset_ranks" : pd.DataFrame — one row per (asset, year)
        asset, year, mean_return, rank  (rank 1 = best asset that year)
    """
    assets = list(weights.keys())

    def _yearly_stats(returns: pd.Series, ppy: int) -> dict:
        """Return {year: {year_return, year_sharpe, year_max_dd}}."""
        out: dict = {}
        for y in returns.index.year.unique():
            yr = returns[returns.index.year == y]
            if len(yr) < 2:
                continue
            equity_y = (1 + yr.fillna(0)).cumprod()
            tot_ret  = float(equity_y.iloc[-1] - 1)
            std_y    = float(yr.std())
            sharpe_y = float(yr.mean() / std_y * np.sqrt(ppy)) if std_y > 0 else 0.0
            run_max  = equity_y.cummax()
            max_dd_y = float(((equity_y - run_max) / run_max).min())
            out[int(y)] = {
                'year_return': tot_ret,
                'year_sharpe': sharpe_y,
                'year_max_dd': max_dd_y,
            }
        return out

    # ── collect raw per-path rows ─────────────────────────────────────────────
    # asset_rows: list of {asset, year, year_return, year_sharpe, year_max_dd}
    # port_rows:  list of {year, year_return, year_sharpe}
    asset_rows: list = []
    port_rows:  list = []

    for pp in portfolio_paths:
        port_returns       = pp.get('portfolio_returns')
        asset_path_indices = pp.get('asset_path_indices', {})

        # ── portfolio yearly stats ────────────────────────────────────────────
        if port_returns is not None and len(port_returns) > 1:
            ppy_port = _infer_periods_per_year(port_returns.index)
            for y, s in _yearly_stats(port_returns, ppy_port).items():
                port_rows.append({'year': y, **s})

        # ── per-asset yearly stats ────────────────────────────────────────────
        for asset in assets:
            path_idx = asset_path_indices.get(asset)
            if path_idx is None:
                continue
            ec = asset_results[asset]['paths'][path_idx].get('equity_curve')
            if ec is None or len(ec) < 2:
                continue
            # equity_curve is 1-based cumulative; pct_change gives bar returns
            asset_rets = ec.pct_change().fillna(0)
            ppy_asset  = _infer_periods_per_year(asset_rets.index)
            for y, s in _yearly_stats(asset_rets, ppy_asset).items():
                asset_rows.append({'asset': asset, 'year': y, **s})

    if not asset_rows:
        empty_a = pd.DataFrame(columns=['asset', 'year', 'mean_return',
                                        'std_return', 'median_return',
                                        'pct_paths_positive', 'mean_sharpe',
                                        'mean_max_dd'])
        empty_p = pd.DataFrame(columns=['year', 'mean_return', 'std_return',
                                        'median_return', 'pct_paths_positive',
                                        'mean_sharpe'])
        empty_r = pd.DataFrame(columns=['asset', 'year', 'mean_return', 'rank'])
        return {'asset_year_summary':     empty_a,
                'portfolio_year_summary': empty_p,
                'asset_ranks':            empty_r}

    asset_df = pd.DataFrame(asset_rows)
    port_df  = pd.DataFrame(port_rows) if port_rows else pd.DataFrame()

    # ── aggregate per (asset, year) ───────────────────────────────────────────
    def _agg_asset(grp):
        r = grp['year_return']
        s = grp['year_sharpe']
        d = grp['year_max_dd']
        return pd.Series({
            'mean_return':       float(r.mean()),
            'std_return':        float(r.std(ddof=1)) if len(r) > 1 else 0.0,
            'median_return':     float(r.median()),
            'pct_paths_positive': float((r > 0).mean()),
            'mean_sharpe':       float(s.mean()),
            'mean_max_dd':       float(d.mean()),
        })

    asset_summary = (asset_df
                     .groupby(['asset', 'year'], sort=True)
                     .apply(_agg_asset)
                     .reset_index())

    # ── aggregate per year (portfolio) ────────────────────────────────────────
    if len(port_df) > 0:
        def _agg_port(grp):
            r = grp['year_return']
            s = grp['year_sharpe']
            return pd.Series({
                'mean_return':        float(r.mean()),
                'std_return':         float(r.std(ddof=1)) if len(r) > 1 else 0.0,
                'median_return':      float(r.median()),
                'pct_paths_positive': float((r > 0).mean()),
                'mean_sharpe':        float(s.mean()),
            })
        port_summary = (port_df
                        .groupby('year', sort=True)
                        .apply(_agg_port)
                        .reset_index())
    else:
        port_summary = pd.DataFrame(columns=['year', 'mean_return', 'std_return',
                                             'median_return', 'pct_paths_positive',
                                             'mean_sharpe'])

    # ── asset ranks per year (rank 1 = highest mean_return) ──────────────────
    ranks_df = (asset_summary[['asset', 'year', 'mean_return']]
                .copy()
                .assign(rank=lambda df: df.groupby('year')['mean_return']
                        .rank(ascending=False, method='min')
                        .astype(int))
                .sort_values(['year', 'rank'])
                .reset_index(drop=True))

    return {
        'asset_year_summary':     asset_summary,
        'portfolio_year_summary': port_summary,
        'asset_ranks':            ranks_df,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helper — trade identification from oos_strategy_df
# ──────────────────────────────────────────────────────────────────────────────

def _extract_trades_from_df(oos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-trade records from an OOS strategy DataFrame.

    Works for both single-asset strategies (needs 'Close' column) and pairs /
    precomputed strategies (needs 'strategy_returns' column).  Identifies trade
    entry and exit via position transitions and attributes PnL via the equity
    curve built from bar-level net returns.

    Parameters
    ----------
    oos_df : pd.DataFrame — from group_results[g]['oos_strategy_df']
        Must contain 'position'; also 'Close' or 'strategy_returns'.

    Returns
    -------
    pd.DataFrame with columns:
        entry_date (Timestamp), exit_date (Timestamp), n_bars (int),
        pnl (float fractional), year (int)
    Returns empty DataFrame if no trades found or columns missing.
    """
    if oos_df is None or len(oos_df) < 2 or 'position' not in oos_df.columns:
        return pd.DataFrame()

    pos     = oos_df['position'].fillna(0)
    eff_pos = pos.shift(1).fillna(0)

    if 'strategy_returns' in oos_df.columns:
        bar_ret = eff_pos * oos_df['strategy_returns'].fillna(0)
    elif 'Close' in oos_df.columns:
        bar_ret = eff_pos * oos_df['Close'].pct_change().fillna(0)
    else:
        bar_ret = pd.Series(0.0, index=oos_df.index)

    equity = (1 + bar_ret).cumprod()

    prev_pos   = pos.shift(1).fillna(0)
    entry_mask = (prev_pos == 0) & (pos != 0)
    exit_mask  = (prev_pos != 0) & ((pos == 0) | (pos != prev_pos))

    entry_locs = list(np.where(entry_mask.values)[0])
    exit_locs  = list(np.where(exit_mask.values)[0])

    if not entry_locs or not exit_locs:
        return pd.DataFrame()

    trades = []
    for e_loc in entry_locs:
        candidates = [x for x in exit_locs if x > e_loc]
        if not candidates:
            break
        x_loc      = candidates[0]
        entry_date = pos.index[e_loc]
        exit_date  = pos.index[x_loc]
        n_bars     = int(x_loc - e_loc)
        eq_e       = float(equity.iloc[e_loc])
        eq_x       = float(equity.iloc[x_loc])
        pnl        = (eq_x / eq_e - 1.0) if (eq_e != 0 and not np.isnan(eq_e)) else 0.0
        trades.append({
            'entry_date': entry_date,
            'exit_date':  exit_date,
            'n_bars':     n_bars,
            'pnl':        pnl,
            'year':       int(entry_date.year),
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
#  Function 9: extract_portfolio_trade_stats
# ──────────────────────────────────────────────────────────────────────────────

def extract_portfolio_trade_stats(
    portfolio_paths : list,
    asset_results   : dict,
    weights         : dict,
) -> dict:
    """
    Extract trade-level and yearly statistics for every sampled portfolio path.

    Trade data is reconstructed from the per-asset, per-group OOS strategy
    DataFrames stored inside each asset's split_results.  For each portfolio
    path the function aggregates trades across all assets and all CPCV groups
    assigned to that path.

    Parameters
    ----------
    portfolio_paths : list returned by sample_portfolio_paths()
    asset_results   : dict returned by load_asset_cpcv()
    weights         : {"ASSET": weight, ...} — determines asset list

    Returns
    -------
    dict with three keys:

    "aggregate_stats" : pd.DataFrame — one row per path, columns:
        path_id, n_trades_total, n_trades_per_asset (dict),
        win_rate, avg_trade_return,
        avg_winning_trade_return, avg_losing_trade_return,
        avg_holding_period_bars, profit_factor,
        max_consecutive_wins, max_consecutive_losses,
        avg_trades_per_year

    "yearly_stats" : pd.DataFrame — one row per (path × year), columns:
        path_id, year, year_return, year_sharpe, year_max_dd, year_n_trades

    "summary" : dict mapping stat name → {"mean": float, "std": float}
    """
    assets = list(weights.keys())

    # ── pre-build {split_id: split_result} lookup per asset ──────────────────
    split_maps: dict = {
        asset: {sr['split_id']: sr
                for sr in asset_results[asset].get('split_results', [])}
        for asset in assets
        if asset in asset_results
    }

    # ── cache per-(asset, split_id, group_idx) trade DataFrames ──────────────
    # With N groups × 2 test groups per split each appears in ~(N-k) splits,
    # many portfolio paths share the same (asset, split_id, group_idx) triplet.
    # Caching avoids reconstructing the same group's trades thousands of times.
    _trade_cache: dict = {}   # key → pd.DataFrame or None

    def _group_trades(asset, split_id, group_idx):
        key = (asset, split_id, group_idx)
        if key in _trade_cache:
            return _trade_cache[key]
        sr = split_maps.get(asset, {}).get(split_id)
        if sr is None:
            _trade_cache[key] = None
            return None
        gr = sr['group_results'].get(group_idx)
        if gr is None:
            _trade_cache[key] = None
            return None
        tdf = _extract_trades_from_df(gr.get('oos_strategy_df'))
        _trade_cache[key] = tdf if len(tdf) > 0 else None
        return _trade_cache[key]

    # ── main loop ─────────────────────────────────────────────────────────────
    agg_rows: list = []
    yr_rows:  list = []

    for pp_idx, pp in enumerate(portfolio_paths):
        port_returns            = pp.get('portfolio_returns')
        asset_path_indices      = pp.get('asset_path_indices',      {})
        asset_split_assignments = pp.get('asset_split_assignments', {})

        # collect all per-trade records across assets/groups for this path
        all_pnl: list = []
        all_hp:  list = []
        year_n_trades: dict = {}
        n_trades_per_asset: dict = {}

        for asset in assets:
            sa = asset_split_assignments.get(asset, [])
            n_asset = 0
            for g_idx, sid in sa:
                tdf = _group_trades(asset, sid, g_idx)
                if tdf is None:
                    continue
                n_asset    += len(tdf)
                all_pnl.extend(tdf['pnl'].tolist())
                all_hp.extend(tdf['n_bars'].tolist())
                for y, cnt in tdf.groupby('year').size().items():
                    year_n_trades[int(y)] = year_n_trades.get(int(y), 0) + cnt
            n_trades_per_asset[asset] = n_asset

        total_n  = sum(n_trades_per_asset.values())
        wins     = [p for p in all_pnl if p > 0]
        losses   = [p for p in all_pnl if p < 0]

        win_rate        = len(wins) / len(all_pnl)       if all_pnl    else 0.0
        avg_trade_ret   = float(np.mean(all_pnl))        if all_pnl    else 0.0
        avg_win_ret     = float(np.mean(wins))           if wins        else 0.0
        avg_loss_ret    = float(np.mean(losses))         if losses      else 0.0
        avg_hp          = float(np.mean(all_hp))         if all_hp      else 0.0
        gross_w         = sum(wins)
        gross_l         = abs(sum(losses))
        pf_val          = (gross_w / gross_l if (wins and gross_l > 0)
                           else (float('inf') if wins else 0.0))

        # max consecutive wins / losses
        max_con_w = max_con_l = cur_w = cur_l = 0
        for p in all_pnl:
            if p > 0:
                cur_w += 1; cur_l = 0
            elif p < 0:
                cur_l += 1; cur_w = 0
            else:
                cur_w = cur_l = 0
            max_con_w = max(max_con_w, cur_w)
            max_con_l = max(max_con_l, cur_l)

        # avg_trades_per_year from portfolio_returns length
        n_years = 0.0
        if port_returns is not None and len(port_returns) > 1:
            ppy     = _infer_periods_per_year(port_returns.index)
            n_years = len(port_returns) / ppy
        avg_tpy = total_n / n_years if n_years > 0 else 0.0

        agg_rows.append({
            'path_id':                   pp_idx,
            'n_trades_total':            total_n,
            'n_trades_per_asset':        n_trades_per_asset,
            'win_rate':                  win_rate,
            'avg_trade_return':          avg_trade_ret,
            'avg_winning_trade_return':  avg_win_ret,
            'avg_losing_trade_return':   avg_loss_ret,
            'avg_holding_period_bars':   avg_hp,
            'profit_factor':             pf_val,
            'max_consecutive_wins':      max_con_w,
            'max_consecutive_losses':    max_con_l,
            'avg_trades_per_year':       avg_tpy,
        })

        # ── yearly stats from portfolio_returns ───────────────────────────────
        if port_returns is not None and len(port_returns) > 1:
            ppy = _infer_periods_per_year(port_returns.index)
            for y in sorted(port_returns.index.year.unique()):
                yr = port_returns[port_returns.index.year == y]
                if len(yr) < 2:
                    continue
                equity_y = (1 + yr.fillna(0)).cumprod()
                tot_ret  = float(equity_y.iloc[-1] - 1)
                std_y    = float(yr.std())
                sharpe_y = (float(yr.mean() / std_y * np.sqrt(ppy))
                            if std_y > 0 else 0.0)
                run_max  = equity_y.cummax()
                max_dd_y = float(((equity_y - run_max) / run_max).min())
                yr_rows.append({
                    'path_id':       pp_idx,
                    'year':          int(y),
                    'year_return':   tot_ret,
                    'year_sharpe':   sharpe_y,
                    'year_max_dd':   max_dd_y,
                    'year_n_trades': year_n_trades.get(int(y), 0),
                })

    agg_df = pd.DataFrame(agg_rows)
    yr_df  = (pd.DataFrame(yr_rows) if yr_rows else pd.DataFrame(
        columns=['path_id', 'year', 'year_return', 'year_sharpe',
                 'year_max_dd', 'year_n_trades']))

    # ── cross-path summary ────────────────────────────────────────────────────
    summary: dict = {}
    for col in ['n_trades_total', 'win_rate', 'avg_trade_return',
                'avg_holding_period_bars', 'profit_factor',
                'avg_trades_per_year', 'max_consecutive_wins',
                'max_consecutive_losses']:
        if col in agg_df.columns:
            vals = agg_df[col].replace([np.inf, -np.inf], np.nan).dropna()
            summary[col] = {
                'mean': float(vals.mean()) if len(vals) else float('nan'),
                'std':  float(vals.std())  if len(vals) > 1 else 0.0,
            }

    return {
        'aggregate_stats': agg_df,
        'yearly_stats':    yr_df,
        'summary':         summary,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Function 10: trade_stats_confidence_intervals
# ──────────────────────────────────────────────────────────────────────────────

def trade_stats_confidence_intervals(
    trade_stats     : dict,
    portfolio_paths : list,
    asset_results   : dict,
    confidence      : float = 0.95,
) -> dict:
    """
    Compute overlap-adjusted confidence intervals on aggregate and yearly stats.

    Uses the same N_eff estimation approach as portfolio_confidence_intervals():
    1 000 random path pairs are sampled; two paths overlap if they share any
    split_id for any asset.

    For yearly CIs the per-year N_eff is re-estimated using only the splits
    whose OOS group actually covers that calendar year (derived from the
    oos_strategy_df date ranges stored in split_results).

    Parameters
    ----------
    trade_stats     : dict returned by extract_portfolio_trade_stats()
    portfolio_paths : list returned by sample_portfolio_paths()
    asset_results   : dict returned by load_asset_cpcv()
    confidence      : float (default 0.95)

    Returns
    -------
    dict with keys:
        "aggregate_cis" : {stat_name: {mean, std, n_paths, n_effective,
                                        naive_ci, adjusted_ci,
                                        conservative_lower_bound}}
        "yearly_cis"    : {year (int): {"return": ci_dict, "sharpe": ci_dict}}
        "confidence"    : float
        "n_effective"   : float  (global N_eff)
    """
    agg_df = trade_stats['aggregate_stats']
    yr_df  = trade_stats['yearly_stats']

    valid = [
        p for p in portfolio_paths
        if p.get('sharpe') is not None and not np.isnan(p.get('sharpe', float('nan')))
    ]
    N = len(valid)
    if N < 2:
        return {'aggregate_cis': {}, 'yearly_cis': {},
                'confidence': confidence, 'n_effective': 1.0}

    # ── global N_eff ──────────────────────────────────────────────────────────
    path_split_sets = []
    for p in valid:
        entry: dict = {}
        for asset, sa in p['asset_split_assignments'].items():
            entry[asset] = frozenset(sid for _, sid in sa)
        path_split_sets.append(entry)

    rng     = np.random.default_rng(0)
    n_pairs = 1000
    i_idx   = rng.integers(0, N, size=n_pairs)
    j_idx   = rng.integers(0, N, size=n_pairs)
    n_ov    = sum(
        1 if (int(i_idx[k]) == int(j_idx[k])) else
        int(any(
            asset in path_split_sets[int(j_idx[k])] and
            (path_split_sets[int(i_idx[k])][asset] &
             path_split_sets[int(j_idx[k])][asset])
            for asset in path_split_sets[int(i_idx[k])]
        ))
        for k in range(n_pairs)
    )
    mean_ov      = n_ov / n_pairs
    n_eff_global = float(N ** 2) / (N ** 2 * max(mean_ov, 1e-10))

    # ── CI helper ─────────────────────────────────────────────────────────────
    alpha     = 1.0 - confidence
    _fallback = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    _z        = _fallback.get(round(confidence, 2), 1.960)

    def _t(df_val):
        if _HAS_SCIPY:
            return float(_t_dist.ppf(1 - alpha / 2, df=max(int(df_val), 1)))
        return _z

    def _ci_dict(vals_arr, n_eff):
        ok   = vals_arr[~np.isnan(vals_arr)]
        n    = len(ok)
        mean = float(np.mean(ok)) if n > 0 else float('nan')
        std  = float(np.std(ok, ddof=1)) if n > 1 else 0.0
        sem  = std / np.sqrt(n)     if n > 0     else 0.0
        se_e = std / np.sqrt(n_eff) if n_eff > 0 else 0.0
        tn   = _t(n - 1)
        te   = _t(n_eff - 1)
        lo_a = mean - te * se_e
        hi_a = mean + te * se_e
        return {
            'mean':                     mean,
            'std':                      std,
            'n_paths':                  n,
            'n_effective':              float(n_eff),
            'naive_ci':                 (mean - tn * sem, mean + tn * sem),
            'adjusted_ci':              (lo_a, hi_a),
            'conservative_lower_bound': lo_a,
        }

    # ── aggregate CIs ─────────────────────────────────────────────────────────
    agg_cis: dict = {}
    for col in ['win_rate', 'avg_trade_return', 'profit_factor',
                'avg_holding_period_bars', 'avg_trades_per_year']:
        if col in agg_df.columns:
            vals = agg_df[col].replace([np.inf, -np.inf], np.nan).values.astype(float)
            agg_cis[col] = _ci_dict(vals, n_eff_global)

    # ── yearly N_eff: use only splits whose group covers each calendar year ───
    # Build {(asset, split_id, group_idx): frozenset of years covered}
    group_years: dict = {}
    for asset, results in asset_results.items():
        for sr in results.get('split_results', []):
            sid = sr['split_id']
            for g_idx, gr in sr['group_results'].items():
                oos_df = gr.get('oos_strategy_df')
                if oos_df is not None and len(oos_df) > 0:
                    group_years[(asset, sid, g_idx)] = frozenset(
                        int(y) for y in oos_df.index.year.unique()
                    )

    # ── yearly CIs ────────────────────────────────────────────────────────────
    yearly_cis: dict = {}
    if len(yr_df) > 0:
        for year in sorted(yr_df['year'].unique()):
            yr_sub = yr_df[yr_df['year'] == year]
            if len(yr_sub) < 2:
                continue
            path_ids_yr = set(yr_sub['path_id'].tolist())

            # rebuild split-sets restricted to groups covering this year
            yr_split_sets = []
            for idx, p in enumerate(valid):
                if idx not in path_ids_yr:
                    continue
                entry: dict = {}
                for asset, sa in p['asset_split_assignments'].items():
                    yr_sids = frozenset(
                        sid for g_idx, sid in sa
                        if year in group_years.get((asset, sid, g_idx), frozenset())
                    )
                    if yr_sids:
                        entry[asset] = yr_sids
                yr_split_sets.append(entry)

            N_yr = len(yr_split_sets)
            if N_yr < 2:
                n_eff_yr = 1.0
            else:
                rng_yr  = np.random.default_rng(year)
                np_yr   = min(1000, N_yr * (N_yr - 1))
                ii      = rng_yr.integers(0, N_yr, size=np_yr)
                jj      = rng_yr.integers(0, N_yr, size=np_yr)
                ov_yr   = sum(
                    1 if (int(ii[k]) == int(jj[k])) else
                    int(any(
                        a in yr_split_sets[int(jj[k])] and
                        (yr_split_sets[int(ii[k])][a] & yr_split_sets[int(jj[k])][a])
                        for a in yr_split_sets[int(ii[k])]
                    ))
                    for k in range(np_yr)
                )
                mean_ov_yr = ov_yr / np_yr
                n_eff_yr   = float(N_yr ** 2) / (N_yr ** 2 * max(mean_ov_yr, 1e-10))

            ret_arr = yr_sub['year_return'].values.astype(float)
            sh_arr  = yr_sub['year_sharpe'].values.astype(float)
            yearly_cis[int(year)] = {
                'return': _ci_dict(ret_arr, n_eff_yr),
                'sharpe': _ci_dict(sh_arr,  n_eff_yr),
            }

    return {
        'aggregate_cis': agg_cis,
        'yearly_cis':    yearly_cis,
        'confidence':    confidence,
        'n_effective':   n_eff_global,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Function 11: trade_stats_summary
# ──────────────────────────────────────────────────────────────────────────────

def trade_stats_summary(trade_stats: dict, trade_cis: dict):
    """
    Print three tables:

    Table 1 — Aggregate trade statistics (win rate, avg return, profit
              factor, avg holding period, trades per year) with mean, std
              and overlap-adjusted 95% CI across all sampled paths.

    Table 2 — Yearly performance: mean return and Sharpe per calendar year
              with overlap-adjusted CIs.

    Table 3 — Year summary: best/worst year, positive-return years,
              conservative-CI-positive years.

    Parameters
    ----------
    trade_stats : dict returned by extract_portfolio_trade_stats()
    trade_cis   : dict returned by trade_stats_confidence_intervals()
    """
    agg_df     = trade_stats['aggregate_stats']
    yr_df      = trade_stats['yearly_stats']
    summary    = trade_stats['summary']
    agg_cis    = trade_cis.get('aggregate_cis', {})
    yearly_cis = trade_cis.get('yearly_cis',    {})
    n_eff      = trade_cis.get('n_effective',   float('nan'))
    conf       = trade_cis.get('confidence',    0.95)
    pct_conf   = int(round(conf * 100))
    _nan       = float('nan')

    def _fv(v, pct=False, dp=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'N/A'
        return f'{v * 100:.{dp}f}%' if pct else f'{v:.{dp}f}'

    def _ci_s(ci_dict, pct=False):
        if not ci_dict:
            return 'N/A'
        lo, hi = ci_dict.get('adjusted_ci', (float('nan'), float('nan')))
        return f'[{_fv(lo, pct)}, {_fv(hi, pct)}]'

    # ── Table 1 ───────────────────────────────────────────────────────────────
    W_S = 22; W_M = 10; W_D = 10; W_C = 26; W_N = 8
    TOT = W_S + W_M + W_D + W_C + W_N + 4
    print(f"\n{'═' * TOT}")
    print(f'AGGREGATE TRADE STATISTICS  ({len(agg_df)} paths, {pct_conf}% CI)')
    print(f"{'═' * TOT}")
    print(f'{"Stat":<{W_S}} {"Mean":>{W_M}} {"Std":>{W_D}} '
          f'{"Adj CI":>{W_C}} {"N_eff":>{W_N}}')
    print(f'{"─"*W_S} {"─"*W_M} {"─"*W_D} {"─"*W_C} {"─"*W_N}')

    neff_s = f'{n_eff:.1f}' if isinstance(n_eff, float) and not np.isnan(n_eff) else 'N/A'
    rows = [
        ('Win rate',            'win_rate',               True),
        ('Avg trade return',    'avg_trade_return',        True),
        ('Profit factor',       'profit_factor',           False),
        ('Avg holding (bars)',  'avg_holding_period_bars', False),
        ('Trades per year',     'avg_trades_per_year',     False),
    ]
    for label, key, as_pct in rows:
        s  = summary.get(key, {})
        ci = agg_cis.get(key)
        print(f'{label:<{W_S}} '
              f'{_fv(s.get("mean", _nan), as_pct):>{W_M}} '
              f'{_fv(s.get("std",  _nan), as_pct):>{W_D}} '
              f'{_ci_s(ci, as_pct):>{W_C}} '
              f'{neff_s:>{W_N}}')

    # ── Table 2 ───────────────────────────────────────────────────────────────
    if len(yr_df) > 0:
        years = sorted(yr_df['year'].unique())
        W_Y = 6; W_R = 10; W_RC = 24; W_SH = 10; W_SC = 22; W_NE = 8
        TOT2 = W_Y + W_R + W_RC + W_SH + W_SC + W_NE + 5
        print(f"\n{'═' * TOT2}")
        print(f'YEARLY PERFORMANCE  (mean across paths, {pct_conf}% adj CI)')
        print(f"{'═' * TOT2}")
        print(f'{"Year":<{W_Y}} {"Return":>{W_R}} {"Return CI":>{W_RC}} '
              f'{"Sharpe":>{W_SH}} {"Sharpe CI":>{W_SC}} {"N_eff":>{W_NE}}')
        print(f'{"─"*W_Y} {"─"*W_R} {"─"*W_RC} {"─"*W_SH} {"─"*W_SC} {"─"*W_NE}')

        for y in years:
            sub  = yr_df[yr_df['year'] == y]
            mr   = float(sub['year_return'].mean())
            ms   = float(sub['year_sharpe'].mean())
            yci  = yearly_cis.get(int(y), {})
            rci  = yci.get('return')
            sci  = yci.get('sharpe')
            ne_y = rci.get('n_effective', _nan) if rci else _nan
            ne_s = f'{ne_y:.1f}' if isinstance(ne_y, float) and not np.isnan(ne_y) else 'N/A'
            print(f'{str(y):<{W_Y}} '
                  f'{_fv(mr, pct=True):>{W_R}} '
                  f'{_ci_s(rci, pct=True):>{W_RC}} '
                  f'{_fv(ms):>{W_SH}} '
                  f'{_ci_s(sci):>{W_SC}} '
                  f'{ne_s:>{W_NE}}')

    # ── Table 3 ───────────────────────────────────────────────────────────────
    if len(yr_df) > 0:
        yr_means = yr_df.groupby('year')['year_return'].mean()
        best_y   = int(yr_means.idxmax())
        worst_y  = int(yr_means.idxmin())
        n_pos    = int((yr_means > 0).sum())
        n_tot    = len(yr_means)
        n_pos_ci = sum(
            1 for y, yci in yearly_cis.items()
            if yci.get('return', {}).get('conservative_lower_bound', float('nan')) > 0
        )
        n_ci_tot = len(yearly_cis)
        W_SUM = 56
        print(f"\n{'═' * W_SUM}")
        print('YEAR SUMMARY')
        print(f'{"─" * W_SUM}')
        print(f'  Best year (mean return):  {best_y} — {_fv(float(yr_means[best_y]), pct=True)}')
        print(f'  Worst year (mean return): {worst_y} — {_fv(float(yr_means[worst_y]), pct=True)}')
        print(f'  Positive mean return:     {n_pos}/{n_tot} years')
        if n_ci_tot > 0:
            print(f'  Conservative CI floor > 0: {n_pos_ci}/{n_ci_tot} years')
        print(f"{'═' * W_SUM}")


# ──────────────────────────────────────────────────────────────────────────────
#  Function 10: coin_inclusion_analysis
# ──────────────────────────────────────────────────────────────────────────────

def coin_inclusion_analysis(
    asset_results           : dict,
    base_weights            : dict,
    n_samples               : int   = 2000,
    seed                    : int   = 42,
    sharpe_floor_threshold  : float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate each asset's contribution to the portfolio to inform include/exclude
    decisions before running weight optimisation.

    For each asset:
      1. individual_floor — conservative Sharpe floor from the asset's own path
         distribution using the weighted overlap N_eff approach identical to
         cpcv_confidence_intervals() in cpcv_engine.py (full N×N weighted overlap
         matrix — feasible because individual asset path counts are small ~105).
      2. excluded_portfolio_floor — conservative Sharpe floor of a portfolio built
         from all OTHER assets at equal weight across that reduced set.
      3. included_portfolio_floor — conservative Sharpe floor of a portfolio built
         from ALL assets at the weights in base_weights (shared across all assets;
         computed once).
      4. marginal_contribution = included_portfolio_floor − excluded_portfolio_floor.

    Recommendation logic:
      "include"  — individual_floor >= sharpe_floor_threshold
                   AND marginal_contribution >= 0
      "exclude"  — individual_floor <  sharpe_floor_threshold
                   OR  marginal_contribution < -0.05
      "marginal" — otherwise (floor above threshold but small negative marginal
                   contribution between 0 and -0.05)

    Parameters
    ----------
    asset_results          : dict returned by load_asset_cpcv()
    base_weights           : {"ASSET": weight, ...} — weights for the full portfolio
                             used to compute included_portfolio_floor; must include
                             all assets in asset_results
    n_samples              : portfolio paths sampled per configuration (default 2000)
    seed                   : RNG seed (default 42)
    sharpe_floor_threshold : individual conservative floor below which asset is
                             flagged for exclusion (default 0.5)

    Returns
    -------
    pd.DataFrame with one row per asset and columns:
        asset, individual_floor, included_portfolio_floor,
        excluded_portfolio_floor, marginal_contribution, recommendation
    """
    import io, contextlib

    assets   = list(asset_results.keys())
    n_assets = len(assets)
    _sink    = io.StringIO()

    # ── step 1: individual conservative Sharpe floor per asset ────────────────
    # Mirrors cpcv_confidence_intervals() in cpcv_engine.py exactly:
    #   weighted overlap matrix → N_eff → t_eff × SEM_eff → lower bound.
    def _individual_floor(asset: str) -> float:
        paths_all = asset_results[asset]['paths']
        valid     = [p for p in paths_all if p.get('sharpe') is not None]
        if not valid:
            return float('nan')

        sharpes = np.array([p['sharpe'] for p in valid], dtype=float)
        N       = len(valid)

        # weighted overlap: cell (i,j) = shared splits / splits_per_path
        path_splits     = [frozenset(sid for _, sid in p['split_assignments'])
                           for p in valid]
        splits_per_path = max(len(ps) for ps in path_splits) if path_splits else 1

        overlap_sum = 0.0
        for i in range(N):
            for j in range(N):
                overlap_sum += len(path_splits[i] & path_splits[j]) / splits_per_path

        n_eff = float(N ** 2) / max(overlap_sum, 1e-10)

        alpha = 0.05   # 95% CI
        if _HAS_SCIPY:
            t_eff = float(_t_dist.ppf(1 - alpha / 2, df=max(n_eff - 1, 1)))
        else:
            t_eff = 1.960

        ok   = sharpes[~np.isnan(sharpes)]
        n    = len(ok)
        mean = float(np.mean(ok))
        std  = float(np.std(ok, ddof=1)) if n > 1 else 0.0
        se_e = std / np.sqrt(n_eff) if n_eff > 0 else 0.0
        return mean - t_eff * se_e

    print('Computing individual Sharpe floors...')
    individual_floors = {a: _individual_floor(a) for a in assets}

    # ── step 2: included_portfolio_floor — full portfolio at base_weights ──────
    # Computed once; shared for all assets' included_portfolio_floor column.
    print('Computing included portfolio floor (full portfolio)...')
    with contextlib.redirect_stdout(_sink):
        included_paths = sample_portfolio_paths(
            asset_results, base_weights,
            n_samples=n_samples, seed=seed,
        )
        included_ci = portfolio_confidence_intervals(included_paths, asset_results)
    included_floor = (included_ci.get('sharpe') or {}).get(
        'conservative_lower_bound', float('nan'))

    # ── step 3: excluded_portfolio_floor per asset ────────────────────────────
    # For each asset: build equal-weight portfolio over the remaining assets only.
    print(f'Computing excluded floors ({n_assets} portfolios)...')
    excluded_floors: dict = {}
    for i, asset in enumerate(assets):
        others = [a for a in assets if a != asset]
        if not others:
            excluded_floors[asset] = float('nan')
            print(f'  [{i + 1}/{n_assets}] {asset:<8}  skipped (no remaining assets)')
            continue

        eq_w        = {a: 1.0 / len(others) for a in others}
        sub_results = {a: asset_results[a] for a in others}

        with contextlib.redirect_stdout(_sink):
            ex_paths = sample_portfolio_paths(
                sub_results, eq_w,
                n_samples=n_samples, seed=seed,
            )
            ex_ci = portfolio_confidence_intervals(ex_paths, sub_results)

        excl_fl = (ex_ci.get('sharpe') or {}).get(
            'conservative_lower_bound', float('nan'))
        excluded_floors[asset] = excl_fl

        marg = included_floor - excl_fl if not (
            np.isnan(included_floor) or np.isnan(excl_fl)) else float('nan')
        print(
            f'  [{i + 1}/{n_assets}] {asset:<8}  '
            f'ind={individual_floors[asset]:.3f}  '
            f'excl_floor={excl_fl:.3f}  '
            f'marginal={marg:+.3f}'
        )

    # ── step 4: build result DataFrame and recommendation ─────────────────────
    rows = []
    for asset in assets:
        ind_fl  = individual_floors[asset]
        incl_fl = included_floor
        excl_fl = excluded_floors[asset]
        marg    = (incl_fl - excl_fl
                   if not (np.isnan(incl_fl) or np.isnan(excl_fl))
                   else float('nan'))

        ind_ok  = not np.isnan(ind_fl)
        marg_ok = not np.isnan(marg)

        if (ind_ok and ind_fl >= sharpe_floor_threshold
                and marg_ok and marg >= 0.0):
            rec = 'include'
        elif ((ind_ok and ind_fl < sharpe_floor_threshold)
              or (marg_ok and marg < -0.05)):
            rec = 'exclude'
        else:
            rec = 'marginal'

        rows.append({
            'asset':                    asset,
            'individual_floor':         ind_fl,
            'included_portfolio_floor': incl_fl,
            'excluded_portfolio_floor': excl_fl,
            'marginal_contribution':    marg,
            'recommendation':           rec,
        })

    df = pd.DataFrame(rows)

    # ── print summary table ───────────────────────────────────────────────────
    W_A   = max(8, max(len(a) for a in assets) + 1)
    W_FL  = 11
    W_REC = 12
    TOTAL = W_A + W_FL * 4 + W_REC + 5

    print(f"\n{'═' * TOTAL}")
    print('COIN INCLUSION ANALYSIS')
    print(f"{'═' * TOTAL}")
    print(
        f'{"Asset":<{W_A}} '
        f'{"Ind Floor":>{W_FL}} '
        f'{"Incl Floor":>{W_FL}} '
        f'{"Excl Floor":>{W_FL}} '
        f'{"Marginal":>{W_FL}} '
        f'{"Recommend":<{W_REC}}'
    )
    print(
        f'{"─" * W_A} '
        f'{"─" * W_FL} {"─" * W_FL} {"─" * W_FL} {"─" * W_FL} '
        f'{"─" * W_REC}'
    )
    for _, r in df.iterrows():
        ind_s  = _fmt(r['individual_floor'])
        incl_s = _fmt(r['included_portfolio_floor'])
        excl_s = _fmt(r['excluded_portfolio_floor'])
        marg_v = r['marginal_contribution']
        marg_s = (f'{marg_v:+.3f}' if not np.isnan(marg_v) else 'N/A')
        print(
            f'{r["asset"]:<{W_A}} '
            f'{ind_s:>{W_FL}} '
            f'{incl_s:>{W_FL}} '
            f'{excl_s:>{W_FL}} '
            f'{marg_s:>{W_FL}} '
            f'{r["recommendation"]:<{W_REC}}'
        )
    print(f'{"─" * TOTAL}')
    incl_count = int((df['recommendation'] == 'include').sum())
    excl_count = int((df['recommendation'] == 'exclude').sum())
    marg_count = int((df['recommendation'] == 'marginal').sum())
    print(
        f'  include: {incl_count}   '
        f'marginal: {marg_count}   '
        f'exclude: {excl_count}'
    )
    print(
        f'  Sharpe floor threshold: {sharpe_floor_threshold:.2f}  '
        f'(individual_floor < threshold OR marginal < -0.05 → exclude)'
    )
    print(f"{'═' * TOTAL}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Function 11: find_optimal_weights
# ──────────────────────────────────────────────────────────────────────────────

def find_optimal_weights(
    asset_results   : dict,
    included_assets : list,
    n_samples       : int   = 2000,
    seed            : int   = 42,
    n_configs       : int   = 500,
    min_weight      : float = 0.05,
    max_weight      : float = 0.60,
) -> dict:
    """
    Optimise portfolio weights over a user-specified included_assets subset.

    After reviewing coin_inclusion_analysis() output, pass the assets you want
    to keep.  Generates n_configs random weight vectors over those assets only
    (Dirichlet-sampled, clipped to [min_weight, max_weight], renormalised).
    The equal-weight vector is always included as config 0.

    Single objective: maximise conservative_sharpe_floor (the overlap-adjusted
    95% CI lower bound on mean Sharpe), using the same N_eff estimation as
    portfolio_confidence_intervals().

    Parameters
    ----------
    asset_results   : dict returned by load_asset_cpcv()
    included_assets : list of asset names to include (subset of asset_results)
    n_samples       : portfolio paths per weight configuration (default 2000)
    seed            : RNG seed (default 42)
    n_configs       : random weight vectors to evaluate (default 500)
    min_weight      : minimum weight per included asset (default 0.05)
    max_weight      : maximum weight per included asset (default 0.60)

    Returns
    -------
    dict:
        "all_configs"          : pd.DataFrame — n_configs rows with metrics +
                                 weight columns (w_{asset})
        "optimal_weights"      : dict {asset: weight} — config with highest
                                 conservative_sharpe_floor
        "optimal_floor"        : float — highest conservative Sharpe floor found
        "equal_weight_floor"   : float — conservative floor at equal weights
        "improvement_vs_equal" : float — optimal_floor − equal_weight_floor
    """
    import io, contextlib

    if not included_assets:
        raise ValueError('included_assets is empty.')
    missing = [a for a in included_assets if a not in asset_results]
    if missing:
        raise ValueError(f'Assets not in asset_results: {missing}')

    sub_results = {a: asset_results[a] for a in included_assets}
    n_assets    = len(included_assets)
    rng         = np.random.default_rng(seed)
    _sink       = io.StringIO()

    # ── feasibility guard ─────────────────────────────────────────────────────
    if min_weight * n_assets > 1.0 + 1e-9:
        raise ValueError(
            f'min_weight={min_weight} × n_assets={n_assets} = '
            f'{min_weight * n_assets:.3f} > 1.0 — infeasible constraint.'
        )

    # ── weight config generator ───────────────────────────────────────────────
    def _sample_one() -> dict:
        for _ in range(2000):
            w = rng.dirichlet(np.ones(n_assets))
            w = np.clip(w, min_weight, max_weight)
            w = w / w.sum()
            if np.all(w >= min_weight - 1e-9) and np.all(w <= max_weight + 1e-9):
                return dict(zip(included_assets, w.tolist()))
        # fallback: equal weight (always feasible when min_weight ≤ 1/n_assets)
        return {a: 1.0 / n_assets for a in included_assets}

    equal_w = {a: 1.0 / n_assets for a in included_assets}
    configs  = [equal_w]
    while len(configs) < n_configs:
        configs.append(_sample_one())

    # ── evaluate each config ──────────────────────────────────────────────────
    rows            : list  = []
    best_floor      : float = float('-inf')
    equal_floor_val : float = float('nan')

    for cfg_idx, w_cfg in enumerate(configs):
        with contextlib.redirect_stdout(_sink):
            p_paths = sample_portfolio_paths(
                sub_results, w_cfg,
                n_samples=n_samples, seed=seed,
            )
            ci = portfolio_confidence_intervals(p_paths, sub_results)

        sh_ci = ci.get('sharpe') or {}
        ca_ci = ci.get('calmar') or {}
        dd_ci = ci.get('max_dd') or {}

        sharpes = [p['sharpe'] for p in p_paths if p.get('sharpe') is not None]
        cons_sh = sh_ci.get('conservative_lower_bound', float('nan'))
        mean_sh = sh_ci.get('mean',   float('nan'))
        mean_ca = ca_ci.get('mean',   float('nan'))
        mean_dd = dd_ci.get('mean',   float('nan'))
        sh_std  = sh_ci.get('std',    float('nan'))
        med_sh  = float(np.median(sharpes)) if sharpes else float('nan')

        row = {
            'config_idx':                cfg_idx,
            'conservative_sharpe_floor': cons_sh,
            'mean_sharpe':               mean_sh,
            'median_sharpe':             med_sh,
            'mean_calmar':               mean_ca,
            'mean_max_dd':               mean_dd,
            'sharpe_std':                sh_std,
        }
        for a in included_assets:
            row[f'w_{a}'] = w_cfg[a]
        rows.append(row)

        if cfg_idx == 0:
            equal_floor_val = cons_sh

        if not np.isnan(cons_sh):
            best_floor = max(best_floor, cons_sh)

        if (cfg_idx + 1) % 50 == 0:
            print(
                f'Config {cfg_idx + 1}/{len(configs)} '
                f'| best conservative Sharpe floor so far: {best_floor:.3f}'
            )

    all_df = pd.DataFrame(rows)

    # ── identify optimal config (max conservative Sharpe floor) ───────────────
    opt_idx   = int(all_df['conservative_sharpe_floor'].idxmax())
    opt_row   = all_df.loc[opt_idx]
    opt_w     = {a: float(opt_row[f'w_{a}']) for a in included_assets}
    opt_floor = float(opt_row['conservative_sharpe_floor'])
    improv    = (opt_floor - equal_floor_val
                 if not np.isnan(equal_floor_val) else float('nan'))

    # ── print summary table ───────────────────────────────────────────────────
    def _w_str(w_dict: dict) -> str:
        return '  '.join(f'{a}:{w_dict[a]:.3f}' for a in included_assets)

    W_CFG = 22
    W_W   = max(n_assets * 11, 30)
    W_CS  = 20
    W_MS  = 13
    W_DD  = 12
    TOTAL = W_CFG + W_W + W_CS + W_MS + W_DD + 4

    eq_row = all_df.loc[0]

    print(f"\n{'═' * TOTAL}")
    print(
        f'FIND OPTIMAL WEIGHTS  '
        f'({n_configs} configs × {n_samples} paths | '
        f'{n_assets} assets | objective: max conservative Sharpe floor)'
    )
    print(f"{'═' * TOTAL}")
    print(
        f'{"Config":<{W_CFG}} '
        f'{"Weights":<{W_W}} '
        f'{"Cons. Sharpe Floor":>{W_CS}} '
        f'{"Mean Sharpe":>{W_MS}} '
        f'{"Mean DD":>{W_DD}}'
    )
    print(
        f'{"─" * W_CFG} {"─" * W_W} '
        f'{"─" * W_CS} {"─" * W_MS} {"─" * W_DD}'
    )

    # optimal row
    print(
        f'{"Optimal (max floor)":<{W_CFG}} '
        f'{_w_str(opt_w):<{W_W}} '
        f'{_fmt(opt_floor):>{W_CS}} '
        f'{_fmt(float(opt_row["mean_sharpe"])):>{W_MS}} '
        f'{_fmt(float(opt_row["mean_max_dd"]), pct=True):>{W_DD}}'
    )

    # equal weight row
    print(
        f'{"Equal weight":<{W_CFG}} '
        f'{_w_str(equal_w):<{W_W}} '
        f'{_fmt(equal_floor_val):>{W_CS}} '
        f'{_fmt(float(eq_row["mean_sharpe"])):>{W_MS}} '
        f'{_fmt(float(eq_row["mean_max_dd"]), pct=True):>{W_DD}}'
    )

    print(f'{"─" * TOTAL}')
    sign = '+' if not np.isnan(improv) and improv >= 0 else ''
    print(
        f'  Improvement vs equal weight: '
        f'{sign}{_fmt(improv)} Sharpe floor points'
    )
    print(f"{'═' * TOTAL}")

    return {
        'all_configs':          all_df,
        'optimal_weights':      opt_w,
        'optimal_floor':        opt_floor,
        'equal_weight_floor':   equal_floor_val,
        'improvement_vs_equal': improv,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Function 12: build_representative_oos_dfs
# ──────────────────────────────────────────────────────────────────────────────

def build_representative_oos_dfs(asset_results: dict, weights: dict) -> dict:
    """
    For each asset, select the CPCV path whose Sharpe is closest to the median
    across all valid paths ("representative path"), then stitch its per-group OOS
    strategy DataFrames into one continuous full-history DataFrame.

    The resulting dict feeds directly into ``plot_closed_trade_equity()`` from
    ``wf_visualizer.py``, giving the same closed-trade equity view that the
    portfolio master notebook produces for walk-forward OOS data.

    Selection rule — representative path
    -------------------------------------
    For each asset, all valid paths are ranked by Sharpe.  The path whose Sharpe
    is closest to the median is chosen.  This avoids both best-case and worst-case
    cherry-picking while producing a realistic single curve.

    Parameters
    ----------
    asset_results : dict returned by load_asset_cpcv()
    weights       : {"ASSET": weight} — only assets present in weights are processed

    Returns
    -------
    dict: {asset: oos_df}
        oos_df is a continuous OOS strategy DataFrame (position / Close /
        position_size columns, same schema as produced by the strategy_fn
        passed to run_cpcv).
    """
    rep_dfs: dict = {}

    for asset in weights:
        if asset not in asset_results:
            print(f'  [{asset}] not in asset_results — skipping')
            continue

        results = asset_results[asset]
        paths   = results['paths']

        valid = [p for p in paths
                 if p.get('sharpe') is not None
                 and p.get('equity_curve') is not None]
        if not valid:
            print(f'  [{asset}] no valid paths — skipping')
            continue

        # ── pick representative path (closest Sharpe to median) ───────────────
        median_sh = float(np.median([p['sharpe'] for p in valid]))
        rep_path  = min(valid, key=lambda p: abs(p['sharpe'] - median_sh))

        # ── build split_id → split_result lookup ──────────────────────────────
        split_map = {sr['split_id']: sr for sr in results['split_results']}

        # ── concatenate OOS DataFrames for each group in the representative path
        slices: list = []
        for g_idx, s_id in sorted(rep_path['split_assignments'], key=lambda x: x[0]):
            sr = split_map.get(s_id)
            if sr is None:
                continue
            oos_df = sr['group_results'].get(g_idx, {}).get('oos_strategy_df')
            if oos_df is not None and len(oos_df) > 0:
                slices.append(oos_df)

        if not slices:
            print(f'  [{asset}] OOS DataFrames missing from split_results — skipping')
            continue

        combined = pd.concat(slices).sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]

        rep_dfs[asset] = combined
        print(
            f'  {asset:<8}  rep Sharpe={rep_path["sharpe"]:.2f} '
            f'(median={median_sh:.2f})  '
            f'bars={len(combined)}  '
            f'[{combined.index[0].date()} → {combined.index[-1].date()}]'
        )

    return rep_dfs
