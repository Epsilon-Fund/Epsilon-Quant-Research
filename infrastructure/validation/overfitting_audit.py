"""
Overfitting-validation harness — Deflated Sharpe Ratio, PBO (CSCV), White's Reality Check.

Quantifies how much of a search-selected strategy's backtest performance is
selection bias. Built for the wf_engine/cpcv_engine pipeline (Optuna TPE,
n_trials per fold/split) but engine-agnostic: every function takes plain
returns arrays/matrices.

The three statistics
--------------------
1. **Deflated Sharpe Ratio** (Bailey & López de Prado 2014). Under the null of
   zero true edge, the best of N trials has an expected maximum Sharpe SR*
   determined by N and the cross-trial variance of Sharpe estimates. The DSR is
   the probability that the observed (selected) Sharpe exceeds SR* given the
   sample length and the non-normality (skew/kurtosis) of returns:

       SR*  = sqrt(V[SR_n]) * ((1-g)*z(1 - 1/N) + g*z(1 - 1/(N*e))),  g = Euler-Mascheroni
       DSR  = PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(T-1)
                              / sqrt(1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2) )

   All SR terms are per-bar (non-annualised); T = number of return observations.
   "Deflated Sharpe > 0" means SR_hat > SR* (equivalently DSR prob > 0.5).

2. **PBO via CSCV** (Bailey, Borwein, López de Prado, Zhu 2014). Takes the T x N
   matrix of candidate-config returns, partitions time into S blocks, and for
   every balanced split of blocks into IS/OOS halves asks: does the config
   ranked best IS fall in the bottom half OOS? PBO = fraction of splits where
   it does. PBO ~= 0.5 means IS ranking carries no OOS information.

3. **White's Reality Check** (White 2000). Bootstrap p-value for
   H0: max_n E[return_n] <= 0 — "after searching N configs, is the best one's
   mean return distinguishable from zero once selection is accounted for?"
   Uses the stationary bootstrap (Politis & Romano 1994); a studentised variant
   (closer to Hansen's SPA) is the default.

Conventions (match infrastructure/backtester/performance_metrics.py)
---------------------------------------------------------------------
- Sharpe = mean(returns)/std(returns, ddof=1) * sqrt(periods_per_year),
  no risk-free rate, flat bars included.
- Returns are per-bar net (post-cost) simple returns.
- Daily crypto bars annualise at 365.

Trial-data requirement
----------------------
PBO and the Reality Check need the full T x N candidate-return matrix. The
engines historically discarded per-trial Optuna data; `run_cpcv`/`walk_forward`
now accept `collect_trials=True` to persist per-trial params/scores, and
`build_trial_returns_matrix` re-runs saved configs (deterministic, vectorised
backtest) to rebuild the matrix. For legacy artifacts, replay the search with
the same design (sampler, seed, search space, objective) and state in the
findings note that the replay is same-design, not bit-identical.

Pre-registered gate (2026-06-10, momentum book audit; see also
brain/handoffs/2026-06-05_novelty_frontier_map.md Prompt 1)
-----------------------------------------------------------
    PASS requires ALL of:
      (a) deflated Sharpe > 0  (DSR prob > 0.5; >= 0.95 is the "strong" bar)
      (b) PBO < 0.5            (< 0.2 is the "good" bar)
      (c) post-haircut lower-CI annualised Sharpe > materiality bar
          (default 0.25 — set BEFORE looking at results)
    Reality-Check p < 0.05 is reported as supporting evidence, not gated.
    Anything else -> FLAG-FOR-REVIEW. Statistical survival != economic
    materiality (brain/CODEX.md "Realism calibration") — the verdict text must
    carry the assumption ledger.

Typical use
-----------
    from overfitting_audit import run_overfitting_audit
    verdict = run_overfitting_audit(
        trial_returns=matrix,             # T x N per-config net returns
        selected_oos_returns=oos_rets,    # the chosen config's OOS returns
        n_trials=400,                     # trials in one selection event
        periods_per_year=365,
        label='BTCUSDT momentum',
    )
    print(verdict.to_markdown())
"""

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329

# Pre-registered gate constants — change only with an explicit re-registration
# note in the findings doc that consumes them.
GATE_DSR_PROB = 0.5          # deflated Sharpe > 0
GATE_DSR_PROB_STRONG = 0.95  # reported, not gated
GATE_PBO_MAX = 0.5
GATE_PBO_GOOD = 0.2          # reported, not gated
GATE_HAIRCUT_LCI_MIN = 0.25  # annualised Sharpe materiality bar
GATE_RC_P = 0.05             # supporting evidence, not gated


# ──────────────────────────────────────────────────────────────────────────────
# Normal distribution (scalar; avoids a scipy dependency — root venv has none)
# ──────────────────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation to the inverse normal CDF (|rel err| < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ──────────────────────────────────────────────────────────────────────────────
# Sharpe helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_returns(returns) -> np.ndarray:
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if r.size < 3:
        raise ValueError(f"need >= 3 return observations, got {r.size}")
    return r


def per_bar_sharpe(returns) -> float:
    """mean/std (ddof=1) per-bar Sharpe; 0.0 for a flat series (std == 0)."""
    r = _clean_returns(returns)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def annualized_sharpe(returns, periods_per_year: float) -> float:
    return per_bar_sharpe(returns) * math.sqrt(periods_per_year)


def sharpe_moments(returns):
    """(per-bar SR, skew, non-excess kurtosis, T) of a return series."""
    r = _clean_returns(returns)
    sd = r.std(ddof=1)
    sr = float(r.mean() / sd) if sd > 0 else 0.0
    if sd > 0:
        z = (r - r.mean()) / sd
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4))  # non-excess: normal -> 3
    else:
        skew, kurt = 0.0, 3.0
    return sr, skew, kurt, r.size


def probabilistic_sharpe_ratio(returns, sr_benchmark: float) -> float:
    """
    PSR (Bailey & López de Prado 2012): P(true SR > sr_benchmark | sample).
    `sr_benchmark` is per-bar (non-annualised).
    """
    sr, skew, kurt, t = sharpe_moments(returns)
    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom_sq <= 0:
        # Pathological skew/kurtosis combination; PSR variance formula breaks.
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(denom_sq)
    return _norm_cdf(z)


def expected_max_sharpe_null(n_trials: float, var_trial_sr: float) -> float:
    """
    E[max per-bar SR] across n_trials independent zero-true-SR trials whose SR
    estimates have variance var_trial_sr (Bailey & López de Prado 2014, eq. for
    SR* via extreme-value theory).
    """
    if n_trials <= 1 or var_trial_sr <= 0:
        return 0.0
    n = float(n_trials)
    return math.sqrt(var_trial_sr) * (
        (1 - EULER_GAMMA) * _norm_ppf(1 - 1 / n)
        + EULER_GAMMA * _norm_ppf(1 - 1 / (n * math.e))
    )


def effective_n_trials(trial_returns) -> float:
    """
    Effective number of independent trials: N_eff = p + (1-p)*M, where p is the
    average pairwise correlation of trial return series and M the trial count
    (pre-registered in brain/handoffs/2026-06-05_novelty_deep_research.md § A.2;
    distinct from cpcv_engine's path-overlap N_eff — do not conflate).
    Flat (zero-variance) columns are excluded from the correlation average.
    """
    m = np.asarray(trial_returns, dtype=float)
    if m.ndim != 2:
        raise ValueError("trial_returns must be 2-D (T x N)")
    n_total = m.shape[1]
    live = m[:, m.std(axis=0, ddof=1) > 0]
    if live.shape[1] < 2:
        return float(n_total)
    corr = np.corrcoef(live, rowvar=False)
    iu = np.triu_indices_from(corr, k=1)
    pbar = float(np.nanmean(corr[iu]))
    pbar = min(max(pbar, 0.0), 1.0)
    return pbar + (1.0 - pbar) * n_total


def min_track_record_length(returns, sr_benchmark: float, confidence: float = 0.95) -> float:
    """
    MinTRL (Bailey & López de Prado 2012): observations needed for PSR(sr_benchmark)
    to clear `confidence`, given the sample's SR/skew/kurtosis. inf if SR <= benchmark.
    """
    sr, skew, kurt, _ = sharpe_moments(returns)
    if sr <= sr_benchmark:
        return float("inf")
    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom_sq <= 0:
        return float("nan")
    z = _norm_ppf(confidence)
    return 1.0 + denom_sq * (z / (sr - sr_benchmark)) ** 2


# ──────────────────────────────────────────────────────────────────────────────
# Stationary bootstrap (Politis & Romano 1994)
# ──────────────────────────────────────────────────────────────────────────────

def stationary_bootstrap_indices(t: int, mean_block_len: float, rng) -> np.ndarray:
    """One resample of indices [0, t): geometric block lengths, wrap-around."""
    p = 1.0 / max(mean_block_len, 1.0)
    starts = rng.integers(0, t, size=t)
    new_block = rng.random(t) < p
    new_block[0] = True
    idx = np.empty(t, dtype=np.int64)
    for i in range(t):
        idx[i] = starts[i] if new_block[i] else (idx[i - 1] + 1) % t
    return idx


def _default_block_len(t: int) -> float:
    # ~T^(1/3): standard rate-optimal order for the stationary bootstrap.
    return max(2.0, round(t ** (1.0 / 3.0)))


def stationary_bootstrap_sharpe_ci(
    returns, periods_per_year: float, n_boot: int = 2000,
    mean_block_len: float | None = None, confidence: float = 0.95, seed: int = 0,
):
    """
    Percentile CI for the annualised Sharpe of one return series.
    Returns (lower, upper, bootstrap_draws).
    """
    r = _clean_returns(returns)
    t = r.size
    block = mean_block_len or _default_block_len(t)
    rng = np.random.default_rng(seed)
    ann = math.sqrt(periods_per_year)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        rb = r[stationary_bootstrap_indices(t, block, rng)]
        sd = rb.std(ddof=1)
        draws[b] = (rb.mean() / sd) * ann if sd > 0 else 0.0
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(draws, [alpha, 1 - alpha])
    return float(lo), float(hi), draws


# ──────────────────────────────────────────────────────────────────────────────
# Deflated Sharpe Ratio
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DSRResult:
    sr_ann: float                 # observed annualised Sharpe of selected returns
    sr_star_ann: float            # expected-max-under-null Sharpe (the haircut), annualised
    deflated_sr_ann: float        # sr_ann - sr_star_ann
    dsr_prob: float               # PSR evaluated at SR*: P(true SR > SR*)
    n_trials: float               # raw trial count used
    n_eff: float                  # effective independent trials used for SR*
    var_trial_sr: float           # per-bar variance of trial Sharpes
    t_obs: int                    # observations in selected return series
    skew: float
    kurt: float                   # non-excess

    @property
    def passes(self) -> bool:
        return self.dsr_prob > GATE_DSR_PROB


def deflated_sharpe_ratio(
    selected_returns, n_trials: float, periods_per_year: float,
    trial_sharpes=None, var_trial_sr: float | None = None,
    n_eff: float | None = None,
) -> DSRResult:
    """
    DSR of a search-selected return series.

    Provide the cross-trial Sharpe dispersion either as `trial_sharpes`
    (per-bar SRs of every candidate, variance taken here) or directly as
    `var_trial_sr`. `n_eff` overrides `n_trials` in the SR* formula (pass
    effective_n_trials(matrix) when trials are correlated); both are recorded.
    """
    if var_trial_sr is None:
        if trial_sharpes is None:
            raise ValueError("need trial_sharpes or var_trial_sr for the SR* haircut")
        ts = np.asarray(trial_sharpes, dtype=float)
        ts = ts[np.isfinite(ts)]
        if ts.size < 2:
            raise ValueError("need >= 2 finite trial Sharpes")
        var_trial_sr = float(ts.var(ddof=1))
    n_used = float(n_eff if n_eff is not None else n_trials)
    sr_star = expected_max_sharpe_null(n_used, var_trial_sr)
    sr, skew, kurt, t = sharpe_moments(selected_returns)
    ann = math.sqrt(periods_per_year)
    return DSRResult(
        sr_ann=sr * ann,
        sr_star_ann=sr_star * ann,
        deflated_sr_ann=(sr - sr_star) * ann,
        dsr_prob=probabilistic_sharpe_ratio(selected_returns, sr_star),
        n_trials=float(n_trials),
        n_eff=n_used,
        var_trial_sr=var_trial_sr,
        t_obs=t,
        skew=skew,
        kurt=kurt,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PBO via CSCV
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PBOResult:
    pbo: float                    # fraction of CSCV splits where IS-best is bottom-half OOS
    n_combinations: int
    n_blocks: int
    n_trials: int
    logits: np.ndarray            # lambda per combination
    logit_quartiles: tuple        # (q25, q50, q75)
    p_oos_loss: float             # fraction of combinations where IS-best has OOS SR < 0
    degradation_slope: float      # OLS slope of (OOS SR of IS-best) on (IS SR of IS-best)
    sensitivity: dict             # {n_blocks: pbo} re-run at other block counts

    @property
    def passes(self) -> bool:
        return self.pbo < GATE_PBO_MAX


def _cscv_pbo_once(m: np.ndarray, n_blocks: int):
    """One CSCV pass. m is T x N, T already trimmed to a multiple of n_blocks."""
    t, n = m.shape
    rows = t // n_blocks
    blocks = m.reshape(n_blocks, rows, n)
    bsum = blocks.sum(axis=1)            # (S, N)
    bsq = (blocks ** 2).sum(axis=1)      # (S, N)
    tot_sum, tot_sq = bsum.sum(axis=0), bsq.sum(axis=0)

    combos = list(combinations(range(n_blocks), n_blocks // 2))
    n_half = rows * (n_blocks // 2)
    logits = np.empty(len(combos))
    oos_sr_best = np.empty(len(combos))
    is_sr_best = np.empty(len(combos))

    def _sr(s, q, cnt):
        mean = s / cnt
        var = np.maximum((q - cnt * mean ** 2) / (cnt - 1), 0.0)
        sd = np.sqrt(var)
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = np.where(sd > 0, mean / sd, 0.0)
        return sr

    chunk = 1024
    for c0 in range(0, len(combos), chunk):
        cset = combos[c0:c0 + chunk]
        sel = np.zeros((len(cset), n_blocks))
        for i, c in enumerate(cset):
            sel[i, list(c)] = 1.0
        sum_is = sel @ bsum
        sq_is = sel @ bsq
        sr_is = _sr(sum_is, sq_is, n_half)
        sr_oos = _sr(tot_sum - sum_is, tot_sq - sq_is, t - n_half)
        best = np.argmax(sr_is, axis=1)
        rows_idx = np.arange(len(cset))
        best_oos = sr_oos[rows_idx, best]
        # relative OOS rank of the IS-best config: omega in (0, 1)
        rank = (sr_oos < best_oos[:, None]).sum(axis=1)
        omega = (rank + 0.5) / n
        logits[c0:c0 + len(cset)] = np.log(omega / (1 - omega))
        oos_sr_best[c0:c0 + len(cset)] = best_oos
        is_sr_best[c0:c0 + len(cset)] = sr_is[rows_idx, best]

    pbo = float((logits <= 0).mean())
    if is_sr_best.std() > 0:
        slope = float(np.polyfit(is_sr_best, oos_sr_best, 1)[0])
    else:
        slope = float("nan")
    return pbo, logits, oos_sr_best, slope, len(combos)


def pbo_cscv(trial_returns, n_blocks: int = 16,
             sensitivity_blocks: tuple = (8, 12)) -> PBOResult:
    """
    Probability of Backtest Overfitting via Combinatorially Symmetric
    Cross-Validation on a T x N candidate-return matrix.

    The C(S, S/2) combinations are mutually dependent, so no formal CI is
    reported; `sensitivity` re-runs PBO at other block counts as a robustness
    range. Tail rows beyond a multiple of n_blocks are trimmed.
    """
    m = np.asarray(trial_returns, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("trial_returns must be T x N with N >= 2")
    if np.isnan(m).any():
        m = np.nan_to_num(m, nan=0.0)
    t = m.shape[0] - (m.shape[0] % n_blocks)
    if t < n_blocks * 2:
        raise ValueError(f"too few rows ({m.shape[0]}) for {n_blocks} blocks")
    pbo, logits, oos_best, slope, n_comb = _cscv_pbo_once(m[:t], n_blocks)
    sens = {}
    for s in sensitivity_blocks:
        ts = m.shape[0] - (m.shape[0] % s)
        sens[s], *_ = _cscv_pbo_once(m[:ts], s)
    q25, q50, q75 = np.quantile(logits, [0.25, 0.5, 0.75])
    return PBOResult(
        pbo=pbo, n_combinations=n_comb, n_blocks=n_blocks, n_trials=m.shape[1],
        logits=logits, logit_quartiles=(float(q25), float(q50), float(q75)),
        p_oos_loss=float((oos_best < 0).mean()), degradation_slope=slope,
        sensitivity=sens,
    )


# ──────────────────────────────────────────────────────────────────────────────
# White's Reality Check
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RealityCheckResult:
    p_value: float                # studentised if studentize=True
    p_value_raw: float            # non-studentised (classic White)
    stat_obs: float
    n_boot: int
    mean_block_len: float
    mc_se: float                  # Monte-Carlo SE of p_value

    @property
    def significant(self) -> bool:
        return self.p_value < GATE_RC_P


def whites_reality_check(
    trial_returns, n_boot: int = 2000, mean_block_len: float | None = None,
    studentize: bool = True, seed: int = 0,
) -> RealityCheckResult:
    """
    Bootstrap p-value for H0: max_n E[return_n] <= 0 over the N candidate
    configs (benchmark = zero / cash, matching the stack's no-risk-free Sharpe).
    Stationary bootstrap; bootstrap stats are centered at sample means (White
    2000). The studentised variant scales by per-config std (Hansen 2005's main
    correction for irrelevant noisy configs dominating the max).
    """
    m = np.asarray(trial_returns, dtype=float)
    if m.ndim != 2 or m.shape[1] < 1:
        raise ValueError("trial_returns must be T x N")
    if np.isnan(m).any():
        m = np.nan_to_num(m, nan=0.0)
    t, n = m.shape
    block = mean_block_len or _default_block_len(t)
    rng = np.random.default_rng(seed)
    mu = m.mean(axis=0)
    sd = m.std(axis=0, ddof=1)
    live = sd > 0
    sd_safe = np.where(live, sd, np.inf)
    rt = math.sqrt(t)

    stat_obs_stud = rt * float(np.max(mu / sd_safe)) if live.any() else 0.0
    stat_obs_raw = rt * float(np.max(mu))

    count_stud = 0
    count_raw = 0
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(t, block, rng)
        cnt = np.bincount(idx, minlength=t).astype(float)
        mu_b = (cnt @ m) / t
        diff = mu_b - mu
        if rt * float(np.max(diff / sd_safe)) >= stat_obs_stud:
            count_stud += 1
        if rt * float(np.max(diff)) >= stat_obs_raw:
            count_raw += 1
    p_stud = count_stud / n_boot
    p_raw = count_raw / n_boot
    p = p_stud if studentize else p_raw
    return RealityCheckResult(
        p_value=p, p_value_raw=p_raw,
        stat_obs=stat_obs_stud if studentize else stat_obs_raw,
        n_boot=n_boot, mean_block_len=float(block),
        mc_se=math.sqrt(max(p * (1 - p), 1e-12) / n_boot),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Trial-matrix construction (rebuild returns from saved trial params)
# ──────────────────────────────────────────────────────────────────────────────

def build_trial_returns_matrix(
    df: pd.DataFrame, strategy_fn, trial_params_list, cost: float,
    run_backtest=None, dedupe: bool = True, verbose: bool = False,
) -> pd.DataFrame:
    """
    T x N net-return matrix: one column per trial config, each run over `df`
    with the stack's backtest (1-bar execution lag, per-leg cost, realised
    sizing). `run_backtest(strategy_df, cost) -> metrics` defaults to
    infrastructure/backtester engine.backtest (imported flat, matching the
    notebooks' sys.path convention). Returns are equity.pct_change() — identical
    to how path Sharpes are computed. Configs that error or never trade become
    zero columns (kept: a search's duds are part of its candidate set).
    """
    if run_backtest is None:
        from engine import backtest as _bt  # noqa: flat import per repo convention
        run_backtest = lambda sdf, c: _bt(sdf, cost=c, show_plot=False)

    seen, configs = set(), []
    for p in trial_params_list:
        key = tuple(sorted(p.items())) if dedupe else len(configs)
        if key not in seen:
            seen.add(key)
            configs.append(dict(p))

    cols = {}
    for i, params in enumerate(configs):
        try:
            out = strategy_fn(df.copy(), params)
            sdf = out[0] if isinstance(out, tuple) else out
            metrics = run_backtest(sdf, cost)
            eq = metrics["equity_curve"]
            cols[i] = eq.pct_change().fillna(0.0)
        except Exception:
            cols[i] = pd.Series(0.0, index=df.index)
        if verbose and (i + 1) % 50 == 0:
            print(f"  trial returns: {i + 1}/{len(configs)}")
    mat = pd.DataFrame(cols).reindex(df.index).fillna(0.0)
    mat.columns = [f"trial_{i}" for i in mat.columns]
    return mat


# ──────────────────────────────────────────────────────────────────────────────
# Verdict object + orchestrator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OverfittingVerdict:
    label: str
    dsr: DSRResult | None
    pbo: PBOResult | None
    rc: RealityCheckResult | None
    sharpe_ci: tuple              # bootstrap CI of observed annualised OOS Sharpe
    haircut_sharpe_ci: tuple      # same CI shifted down by SR*_ann
    periods_per_year: float
    confidence: float
    materiality_min_sharpe: float
    notes: list = field(default_factory=list)

    @property
    def gates(self) -> dict:
        g = {}
        g["dsr_gt_0"] = (self.dsr is not None and self.dsr.passes)
        g["pbo_lt_0.5"] = (self.pbo is not None and self.pbo.passes)
        g["haircut_lci_material"] = (
            np.isfinite(self.haircut_sharpe_ci[0])
            and self.haircut_sharpe_ci[0] > self.materiality_min_sharpe
        )
        return g

    @property
    def verdict(self) -> str:
        if self.dsr is None or self.pbo is None:
            return "INSUFFICIENT-DATA"
        return "PASS" if all(self.gates.values()) else "FLAG-FOR-REVIEW"

    def to_markdown(self) -> str:
        """Markdown report fragment for pasting into a findings note."""
        g = self.gates
        tick = lambda b: "PASS" if b else "FAIL"
        lines = [
            f"### Overfitting audit — {self.label}",
            "",
            f"**Verdict: {self.verdict}** (gate: deflated Sharpe > 0 AND PBO < {GATE_PBO_MAX} "
            f"AND post-haircut lower-CI Sharpe > {self.materiality_min_sharpe})",
            "",
            "| Check | Value | Gate | Result |",
            "|---|---|---|---|",
        ]
        if self.dsr:
            d = self.dsr
            lines += [
                f"| Observed OOS Sharpe (ann.) | {d.sr_ann:.2f} | — | — |",
                f"| Selection haircut SR* (ann., N_eff={d.n_eff:.0f} of {d.n_trials:.0f} trials) "
                f"| {d.sr_star_ann:.2f} | — | — |",
                f"| Deflated Sharpe (ann.) | {d.deflated_sr_ann:.2f} | > 0 | {tick(g['dsr_gt_0'])} |",
                f"| DSR probability | {d.dsr_prob:.3f} | > {GATE_DSR_PROB} "
                f"(strong: ≥ {GATE_DSR_PROB_STRONG}) | {tick(g['dsr_gt_0'])} |",
            ]
        if self.pbo:
            p = self.pbo
            sens = ", ".join(f"S={k}: {v:.2f}" for k, v in p.sensitivity.items())
            lines += [
                f"| PBO (CSCV, S={p.n_blocks}, {p.n_combinations} combos, "
                f"N={p.n_trials}) | {p.pbo:.3f} | < {GATE_PBO_MAX} (good: < {GATE_PBO_GOOD}) "
                f"| {tick(g['pbo_lt_0.5'])} |",
                f"| PBO sensitivity | {sens} | — | — |",
                f"| P(IS-best loses OOS) | {p.p_oos_loss:.3f} | — | — |",
            ]
        if self.rc:
            r = self.rc
            lines.append(
                f"| Reality-Check p (studentised, B={r.n_boot}, block≈{r.mean_block_len:.0f}) "
                f"| {r.p_value:.4f} ± {2 * r.mc_se:.4f} | < {GATE_RC_P} (supporting) "
                f"| {'yes' if r.significant else 'no'} |"
            )
        lo, hi = self.sharpe_ci
        hlo, hhi = self.haircut_sharpe_ci
        lines += [
            f"| OOS Sharpe {int(self.confidence * 100)}% CI (stationary bootstrap) "
            f"| [{lo:.2f}, {hi:.2f}] | — | — |",
            f"| Post-haircut Sharpe CI | [{hlo:.2f}, {hhi:.2f}] "
            f"| lower > {self.materiality_min_sharpe} | {tick(g['haircut_lci_material'])} |",
        ]
        if self.notes:
            lines += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)

    def plot(self, figsize=(11, 4)):
        """
        Inline diagnostic figure for a notebook (matplotlib lazy-imported so the
        module core stays numpy/pandas-only). Returns the Figure.

        Left panel  — observed OOS Sharpe (with its bootstrap CI as an error bar)
                      vs the selection haircut SR* vs the deflated Sharpe, with
                      the materiality bar. Tells you at a glance how much of the
                      Sharpe is search luck.
        Right panel — the CSCV logit distribution: each value is one split's
                      IS-best config ranked OOS; mass LEFT of 0 = overfit splits,
                      so the shaded share left of the red line IS the PBO.
        """
        import matplotlib.pyplot as plt

        has_pbo = self.pbo is not None
        fig, axes = plt.subplots(1, 2 if has_pbo else 1, figsize=figsize)
        axes = np.atleast_1d(axes)

        ax = axes[0]
        if self.dsr is not None:
            d = self.dsr
            vals = [d.sr_ann, d.sr_star_ann, d.deflated_sr_ann]
            colors = ["#4878d0", "#d65f5f", "#6acc64"]
            ax.bar(["observed\nOOS", "selection\nhaircut SR*", "deflated"], vals, color=colors)
            lo, hi = self.sharpe_ci
            ax.errorbar([0], [d.sr_ann], yerr=[[d.sr_ann - lo], [hi - d.sr_ann]],
                        fmt="none", ecolor="black", capsize=4, lw=1)
            ax.axhline(self.materiality_min_sharpe, color="gray", ls="--", lw=0.8,
                       label=f"materiality ({self.materiality_min_sharpe})")
            ax.axhline(0, color="black", lw=0.6)
            ax.set_ylabel("Annualised Sharpe")
            ax.set_title(f"{self.label}: deflated Sharpe (DSR prob {d.dsr_prob:.3f})")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no DSR (need trial dispersion)", ha="center", va="center")
            ax.axis("off")

        if has_pbo:
            ax = axes[1]
            ax.hist(self.pbo.logits, bins=40, color="#9ecae1", edgecolor="white")
            ax.axvline(0, color="#d65f5f", ls="--", lw=1.2, label="overfit threshold")
            ax.set_xlabel("CSCV logit  (IS-best config's OOS rank)")
            ax.set_ylabel("splits")
            ax.set_title(f"PBO = {self.pbo.pbo:.3f}  (mass left of 0)")
            ax.legend(fontsize=8)

        fig.suptitle(f"Overfitting audit — {self.label}  [{self.verdict}]", y=1.02)
        fig.tight_layout()
        return fig


def run_overfitting_audit(
    selected_oos_returns, n_trials: float, periods_per_year: float,
    trial_returns=None, trial_sharpes=None, var_trial_sr: float | None = None,
    label: str = "", n_blocks: int = 16, n_boot: int = 2000,
    confidence: float = 0.95, materiality_min_sharpe: float = GATE_HAIRCUT_LCI_MIN,
    rc_studentize: bool = True, seed: int = 0, notes: list | None = None,
) -> OverfittingVerdict:
    """
    Full audit of one selected strategy.

    selected_oos_returns : bar-level net OOS returns of the chosen config.
    n_trials             : trials in the selection event that chose it.
    trial_returns        : optional T x N candidate-return matrix. Enables PBO,
                           Reality Check, N_eff, and (if trial_sharpes absent)
                           the cross-trial SR variance for DSR.
    trial_sharpes / var_trial_sr : alternative dispersion inputs for DSR when
                           no matrix is available.

    Returns an OverfittingVerdict; PBO/RC are None when no matrix was given and
    the verdict degrades to INSUFFICIENT-DATA.
    """
    notes = list(notes or [])
    pbo = rc = dsr = None
    n_eff = None

    if trial_returns is not None:
        mat = (trial_returns.values if isinstance(trial_returns, pd.DataFrame)
               else np.asarray(trial_returns, dtype=float))
        if trial_sharpes is None and var_trial_sr is None:
            sds = mat.std(axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                trial_sharpes = np.where(sds > 0, mat.mean(axis=0) / sds, 0.0)
        n_eff = effective_n_trials(mat)
        pbo = pbo_cscv(mat, n_blocks=n_blocks)
        rc = whites_reality_check(mat, n_boot=n_boot, studentize=rc_studentize, seed=seed)
    else:
        notes.append(
            "No trial-return matrix supplied: PBO and Reality Check unavailable; "
            "DSR uses raw n_trials (no correlation-based N_eff shrinkage)."
        )

    if trial_sharpes is not None or var_trial_sr is not None:
        dsr = deflated_sharpe_ratio(
            selected_oos_returns, n_trials, periods_per_year,
            trial_sharpes=trial_sharpes, var_trial_sr=var_trial_sr, n_eff=n_eff,
        )

    lo, hi, _ = stationary_bootstrap_sharpe_ci(
        selected_oos_returns, periods_per_year, n_boot=n_boot,
        confidence=confidence, seed=seed,
    )
    sr_star_ann = dsr.sr_star_ann if dsr else float("nan")
    return OverfittingVerdict(
        label=label, dsr=dsr, pbo=pbo, rc=rc,
        sharpe_ci=(lo, hi),
        haircut_sharpe_ci=(lo - sr_star_ann, hi - sr_star_ann),
        periods_per_year=periods_per_year, confidence=confidence,
        materiality_min_sharpe=materiality_min_sharpe, notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CPCV-artifact adapter
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_ohlcv_from_cpcv(results: dict) -> pd.DataFrame:
    """
    Rebuild the full OHLCV the CPCV run consumed by unioning every split's
    per-group OOS DataFrames (the 8 groups tile the whole sample). Avoids
    refetching from Binance, which would change the sample.
    """
    frames = []
    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    for sr in results["split_results"]:
        for g in sr["group_results"].values():
            df = g.get("oos_strategy_df")
            if df is not None:
                frames.append(df[[c for c in base_cols if c in df.columns]])
    if not frames:
        raise ValueError("no oos_strategy_df frames in artifact")
    full = pd.concat(frames).sort_index()
    return full[~full.index.duplicated(keep="first")]


def audit_cpcv_run(
    results: dict, df: pd.DataFrame | None = None, strategy_fn=None,
    score_fn=None, reject_fn=None, n_trials: int | None = None,
    seed: int = 42, label: str = "", run_backtest=None, verbose: bool = True,
    **audit_kwargs,
) -> OverfittingVerdict:
    """
    One-call audit for a run_cpcv results dict (the *_cpcv.pkl schema).

    Selected-OOS series: the median-Sharpe CPCV path's equity curve returns
    (a representative single OOS realisation, not the path mean — averaging
    paths would shrink variance and flatter the Sharpe).

    Trial matrix: if the artifact carries per-trial records (run_cpcv
    collect_trials=True), their configs are re-run over the full sample.
    Otherwise, when `strategy_fn` is given, the search is REPLAYED same-design
    (one TPE study, same n_trials/search space/objective, fixed seed) on the
    full sample and each explored config is run full-sample. State in the
    findings note which mode produced the matrix.
    """
    cfg = results["config"]
    n_trials = n_trials or cfg["n_trials"]
    if df is None:
        df = reconstruct_ohlcv_from_cpcv(results)
    ppy = _infer_periods_per_year(df.index)

    paths = [p for p in results["paths"] if p.get("sharpe") is not None]
    med = sorted(paths, key=lambda p: p["sharpe"])[len(paths) // 2]
    selected = med["equity_curve"].pct_change().dropna()

    trial_params = []
    for sr in results["split_results"]:
        for t in sr.get("trials", []) or []:
            trial_params.append({**results.get("fixed_params", {}), **t["params"]})

    matrix = None
    if trial_params:
        if verbose:
            print(f"building trial matrix from {len(trial_params)} persisted trials...")
        matrix = build_trial_returns_matrix(
            df, strategy_fn, trial_params, cfg["cost"],
            run_backtest=run_backtest, verbose=verbose,
        )
        mode_note = f"trial matrix from {len(trial_params)} persisted trial configs (collect_trials)"
    elif strategy_fn is not None:
        if verbose:
            print(f"replaying {n_trials}-trial TPE search (same-design, seed={seed})...")
        matrix = replay_search_trial_matrix(
            df, strategy_fn, results["param_defs"], results.get("fixed_params", {}),
            n_trials=n_trials, cost=cfg["cost"], score_fn=score_fn,
            reject_fn=reject_fn, seed=seed, run_backtest=run_backtest, verbose=verbose,
        )
        mode_note = (f"trial matrix from a same-design TPE replay ({n_trials} trials, "
                     f"seed={seed}) — not the original studies (engines discarded them)")
    else:
        mode_note = "no persisted trials and no strategy_fn: matrix unavailable"

    v = run_overfitting_audit(
        selected_oos_returns=selected, n_trials=n_trials, periods_per_year=ppy,
        trial_returns=matrix, label=label,
        notes=[mode_note,
               "selected-OOS series = median-Sharpe CPCV path equity returns"],
        **audit_kwargs,
    )
    return v


def replay_search_trial_matrix(
    df: pd.DataFrame, strategy_fn, param_defs: dict, fixed_params: dict,
    n_trials: int, cost: float, score_fn=None, reject_fn=None,
    seed: int = 42, run_backtest=None, verbose: bool = False,
    return_params: bool = False,
):
    """
    Run one fresh Optuna TPE study (same design as the engines: TPESampler with
    a fixed seed, same search space/objective/rejection) over the FULL sample,
    capturing every explored config's net-return series. This reproduces the
    *kind* of candidate set a 400-trial selection event sees; it is not a
    bit-identical replay of the original per-split studies.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if run_backtest is None:
        from engine import backtest as _bt  # noqa: flat import per repo convention
        run_backtest = lambda sdf, c: _bt(sdf, cost=c, show_plot=False)

    if score_fn is None:
        score_fn = _engine_default_score
    if reject_fn is None:
        reject_fn = _engine_default_reject

    captured = []
    captured_params = []

    def objective(trial):
        params = dict(fixed_params)
        for name, (ptype, lo, hi) in param_defs.items():
            if name in fixed_params:
                continue
            params[name] = (trial.suggest_int(name, int(lo), int(hi)) if ptype == "int"
                            else trial.suggest_float(name, float(lo), float(hi)))
        captured_params.append(dict(params))
        try:
            out = strategy_fn(df.copy(), params)
            sdf = out[0] if isinstance(out, tuple) else out
            metrics = run_backtest(sdf, cost)
        except Exception:
            captured.append(pd.Series(0.0, index=df.index))
            return -999.0
        rets = metrics["equity_curve"].pct_change().fillna(0.0)
        captured.append(rets)
        if reject_fn(metrics):
            return -999.0
        return score_fn(metrics)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    mat = pd.DataFrame({f"trial_{i}": s for i, s in enumerate(captured)})
    mat = mat.reindex(df.index).fillna(0.0)
    return (mat, captured_params) if return_params else mat


def _engine_default_score(metrics):
    # mirrors wf_engine._default_score (50/30/20, caps 2.5/60/15, raw calmar)
    calmar = (metrics["total_return"] / abs(metrics["max_drawdown"])
              if metrics["max_drawdown"] != 0 else 0.0)
    s = np.clip(metrics["sharpe_ratio"] / 2.5, 0, 1)
    c = np.clip(calmar / 60.0, 0, 1)
    r = np.clip(metrics["total_return"] / 15.0, 0, 1)
    return 0.50 * s + 0.30 * c + 0.20 * r


def _engine_default_reject(metrics):
    # mirrors wf_engine._default_reject
    if metrics is None:
        return True
    if metrics["num_trades"] < 7:
        return True
    if metrics["win_rate"] < 0.35:
        return True
    if metrics["max_drawdown"] < -0.80:
        return True
    if metrics["profit_factor"] < 0.8:
        return True
    return False


def _infer_periods_per_year(index) -> float:
    # mirrors infrastructure/backtester/performance_metrics.infer_frequency
    if len(index) < 2:
        return 365.0
    med = pd.Series(index).diff().median()
    hours = med.total_seconds() / 3600.0
    if hours <= 1:
        return 8760.0
    if hours <= 4:
        return 2190.0
    if hours <= 24:
        return 365.0
    if hours <= 168:
        return 52.0
    return 12.0


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic-null Monte Carlo
# ──────────────────────────────────────────────────────────────────────────────
# The empirical complement to the DSR: instead of an analytic expected-max
# formula, run the SAME candidate set on no-timing-signal synthetic data and
# ask how often it manufactures a max Sharpe as high as the real one. Captures
# fat tails, vol clustering, and cross-config dependence that DSR's
# normal/extreme-value assumptions approximate away.
#
# Null construction: stationary block bootstrap over bar-level RELATIVE bars
# (O/H/L/C expressed against the previous close, volume carried along), then
# re-chained into a price path. Short blocks destroy the long-range trend
# structure a momentum rule exploits while preserving drift, fat tails, and
# short-range vol clustering. `demean=True` additionally removes drift
# (geometric-mean close return forced to 1) — a harsher null that tests total
# edge vs cash rather than timing skill. For long-only trend strategies the
# drift-preserving null is the fair primary: it asks "does the timing rule add
# anything beyond luck + holding a drifting asset?"
#
# Pre-registered gate (2026-06-10): real statistic > 95th percentile of the
# null distribution. The statistic is max-of-N-configs Sharpe — the same
# selection event on both sides of the comparison.


def make_null_ohlcv(df, mean_block_len: float = 10.0, rng=None, demean: bool = False,
                    _indices=None):
    """
    One synthetic null OHLCV path from `df` (needs Open/High/Low/Close/Volume).

    Pass a dict {name: df} of index-aligned frames to resample them JOINTLY
    (same block indices -> cross-asset correlation preserved); returns a dict.
    `_indices` overrides the resampling indices (internal/testing).
    """
    if isinstance(df, dict):
        frames = list(df.values())
        idx0 = frames[0].index
        for f in frames[1:]:
            if not f.index.equals(idx0):
                raise ValueError("joint resampling requires identical indices — align first")
        if rng is None:
            rng = np.random.default_rng(0)
        t = len(idx0) - 1
        idx = stationary_bootstrap_indices(t, mean_block_len, rng)
        return {k: make_null_ohlcv(v, mean_block_len, demean=demean, _indices=idx)
                for k, v in df.items()}

    c = df["Close"].values
    prev_c = c[:-1]
    rel = {
        "Open": df["Open"].values[1:] / prev_c,
        "High": df["High"].values[1:] / prev_c,
        "Low": df["Low"].values[1:] / prev_c,
        "Close": c[1:] / prev_c,
    }
    vol = df["Volume"].values[1:]
    if demean:
        g = float(np.exp(np.mean(np.log(rel["Close"]))))
        rel = {k: v / g for k, v in rel.items()}

    t = len(prev_c)
    if _indices is None:
        if rng is None:
            rng = np.random.default_rng(0)
        _indices = stationary_bootstrap_indices(t, mean_block_len, rng)

    rc = rel["Close"][_indices]
    base = np.empty(t + 1)
    base[0] = c[0]
    np.multiply.accumulate(rc, out=base[1:])
    base[1:] *= c[0]
    prev_path = base[:-1]

    out = pd.DataFrame(index=df.index)
    out["Open"] = np.concatenate(([df["Open"].values[0]], rel["Open"][_indices] * prev_path))
    out["High"] = np.concatenate(([df["High"].values[0]], rel["High"][_indices] * prev_path))
    out["Low"] = np.concatenate(([df["Low"].values[0]], rel["Low"][_indices] * prev_path))
    out["Close"] = base
    out["Volume"] = np.concatenate(([df["Volume"].values[0]], vol[_indices]))
    return out


@dataclass
class NullMCResult:
    real_stat: float              # the real-data statistic (e.g. max-of-N-configs Sharpe)
    null_stats: np.ndarray        # the same statistic on each synthetic path
    percentile: float             # share of null paths the real stat beats
    q95: float                    # 95th percentile of the null distribution
    n_paths: int
    mean_block_len: float
    demeaned: bool

    @property
    def passes(self) -> bool:
        return self.real_stat > self.q95

    @property
    def mc_se(self) -> float:
        p = max(min(1.0 - self.percentile, 1.0), 1.0 / self.n_paths)
        return math.sqrt(p * (1 - p) / self.n_paths)

    def plot(self, figsize=(7, 4), label=""):
        """Histogram of the null statistic with the real value + null 95th pct.
        matplotlib lazy-imported. Returns the Figure."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(self.null_stats, bins=30, color="#9ecae1", edgecolor="white",
                label=f"null ({self.n_paths} paths)")
        ax.axvline(self.q95, color="#d65f5f", ls="--", label=f"null 95th pct = {self.q95:.2f}")
        ax.axvline(self.real_stat, color="#2a7d2a", lw=2, label=f"real = {self.real_stat:.2f}")
        verdict = "PASS" if self.passes else "FAIL"
        ax.set_xlabel("statistic (e.g. max-of-N-configs Sharpe)")
        ax.set_ylabel("null paths")
        ax.set_title(f"{label} synthetic-null MC — real pctile {self.percentile:.3f} ({verdict})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig


def summarize_null_mc(real_stat: float, null_stats, mean_block_len: float,
                      demeaned: bool) -> NullMCResult:
    ns = np.asarray(null_stats, dtype=float)
    ns = ns[np.isfinite(ns)]
    return NullMCResult(
        real_stat=float(real_stat), null_stats=ns,
        percentile=float((ns < real_stat).mean()),
        q95=float(np.quantile(ns, 0.95)),
        n_paths=ns.size, mean_block_len=float(mean_block_len), demeaned=demeaned,
    )


def synthetic_null_mc(df, eval_fn, n_paths: int = 200, mean_block_len: float = 10.0,
                      demean: bool = False, seed: int = 1000, real_stat: float | None = None,
                      verbose: bool = False) -> NullMCResult:
    """
    Serial reference driver: `eval_fn(df_synth) -> float` is the FULL pipeline
    statistic (e.g. "evaluate all N candidate configs, return the max Sharpe").
    `real_stat` defaults to eval_fn(df). Each path uses rng seed `seed + k` —
    reproducible and order-independent. For large runs, parallelise in a
    runner script with a top-level worker calling make_null_ohlcv directly
    (see topics/momentum/strategies/momentum_cpcv/run_synthetic_null_mc.py).
    """
    if real_stat is None:
        real_stat = float(eval_fn(df))
    stats = np.empty(n_paths)
    for k in range(n_paths):
        synth = make_null_ohlcv(df, mean_block_len,
                                rng=np.random.default_rng(seed + k), demean=demean)
        stats[k] = eval_fn(synth)
        if verbose and (k + 1) % 25 == 0:
            print(f"  null path {k + 1}/{n_paths}")
    return summarize_null_mc(real_stat, stats, mean_block_len, demean)
