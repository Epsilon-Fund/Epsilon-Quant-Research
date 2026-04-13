#!/usr/bin/env python3
"""
Pairs_Screening_v5_1.py
========================
Strategy-aligned pair screener for episodic mean-reversion.

V5.1 Changes from V5.0:
  - Fix 1: Multi-reference lookback scan (5 lookbacks instead of 1 fixed)
  - Fix 2: ADF computed on best-reference spread (not fixed lookback)
  - Fix 3: Relative beta smoothness (normalized to beta level)
  - Fix 4: Weight rebalance (RT 30, Beta 10, WRS 20, Regime 25, ADF 5)
  - Fix 5: Regime quality threshold (>= 2 RTs AND completion >= 25%)

Output:
  pairs_screen_v5_1_wf_input.csv  → top N candidates for WF template
  pairs_screen_v5_1_full.csv      → all scored pairs for diagnostics
"""

import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from scipy import stats

# ── Your infrastructure ──
sys.path.append(str(Path(__file__).resolve().parents[4] / "infrastructure" / "data"))
from binance_client import get_binance_client, get_data


warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# --- Output ---
OUTPUT_DIR = Path("output/screening")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Universe ---
# --- Universe (removed MATICUSDT) ---
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "UNIUSDT", "ATOMUSDT", "LTCUSDT", "ETCUSDT",
    "FILUSDT", "APTUSDT", "NEARUSDT", "ICPUSDT", "ARBUSDT",
    "OPUSDT", "IMXUSDT", "INJUSDT", "SNXUSDT", "TRXUSDT",
    "MKRUSDT", "AAVEUSDT", "RUNEUSDT", "FETUSDT", "TIAUSDT",
    "SUIUSDT", "SEIUSDT", "JUPUSDT", "POLUSDT",
]

# --- Data ---
TIMEFRAME = "1d"
N_BARS = 1500  # How much history to pull

# --- Multi-Reference Lookback Scan (Fix 1) ---
REF_LOOKBACKS = [100, 130, 155, 180, 210]
REF_ENTRY_Z = 1.5
REF_EXIT_Z = 1.0
REF_MAX_HOLD = 10

# --- Window Reversion Speed ---
WRS_WINDOW = 250
WRS_HALFLIFE_THRESHOLD = 10

# --- Regime Stability (Fix 5) ---
N_REGIMES = 4
REGIME_MIN_RTS = 2
REGIME_MIN_COMPLETION = 0.25

# --- Kurtosis Penalty ---
KURTOSIS_THRESHOLD = 3.0
KURTOSIS_MAX_PENALTY = 5.0

# --- Score Weights (Fix 4) ---
W_RT_FREQUENCY = 30
W_BETA_SMOOTH = 10
W_WINDOW_REVERSION = 20
W_REGIME_STABILITY = 25
W_ADF = 5
W_DATA_BONUS = 5
KURTOSIS_PENALTY_MAX = 5

# --- Beta Smoothness Relative Thresholds (Fix 3) ---
BETA_REL_BEST = 0.03
BETA_REL_WORST = 0.20

# --- RT Frequency Scoring Thresholds ---
RT_ANNUAL_BEST = 8.0
RT_ANNUAL_WORST = 0.5
RT_COMP_BEST = 0.70
RT_COMP_WORST = 0.15

# --- ADF Thresholds ---
ADF_BEST = 0.01
ADF_WORST = 0.20

# --- Output ---
TOP_N = 50
MIN_OVERLAP_DAYS = 365


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_prices(symbols, client, timeframe=TIMEFRAME, n_bars=N_BARS):
    """Pull close prices for all symbols via Binance client."""
    frames = {}
    for sym in symbols:
        try:
            df = get_data(client, sym, timeframe, n_bars)
            if df is None or df.empty:
                print(f"  [WARN] {sym}: no data returned, skipping")
                continue

            # normalise column names to lowercase
            # (get_data returns 'Close' capitalised)
            df.columns = [c.lower() for c in df.columns]

            if "close" not in df.columns:
                print(f"  [WARN] {sym}: columns {list(df.columns)}, skipping")
                continue

            frames[sym] = df["close"]
            print(f"  ✓ {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  [WARN] {sym}: {e}, skipping")
            continue

    if not frames:
        raise RuntimeError("No symbols loaded — check API connection")

    prices = pd.DataFrame(frames)
    prices = prices.sort_index()
    print(f"\n  Loaded {len(frames)}/{len(symbols)} symbols")
    print(f"  Date range: {prices.index[0]} → {prices.index[-1]}")
    return prices


# ═══════════════════════════════════════════════════════════════════
# CORE COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════

def compute_rolling_ols(log_y, log_x, lookback):
    """
    Rolling OLS: ln(Y) = alpha + beta * ln(X) + epsilon
    Vectorized using rolling sums for speed.
    """
    n = len(log_y)
    y = log_y.values
    x = log_x.values

    betas = np.full(n, np.nan)
    alphas = np.full(n, np.nan)
    spreads = np.full(n, np.nan)
    z_scores = np.full(n, np.nan)

    # precompute rolling sums
    cum_x = np.cumsum(x)
    cum_y = np.cumsum(y)
    cum_x2 = np.cumsum(x * x)
    cum_xy = np.cumsum(x * y)
    cum_y2 = np.cumsum(y * y)

    for i in range(lookback, n):
        j = i - lookback

        if j == 0:
            sx  = cum_x[i-1]
            sy  = cum_y[i-1]
            sx2 = cum_x2[i-1]
            sxy = cum_xy[i-1]
        else:
            sx  = cum_x[i-1]  - cum_x[j-1]
            sy  = cum_y[i-1]  - cum_y[j-1]
            sx2 = cum_x2[i-1] - cum_x2[j-1]
            sxy = cum_xy[i-1] - cum_xy[j-1]

        denom = lookback * sx2 - sx * sx
        if abs(denom) < 1e-14:
            continue

        beta_val  = (lookback * sxy - sx * sy) / denom
        alpha_val = (sy - beta_val * sx) / lookback

        betas[i]  = beta_val
        alphas[i] = alpha_val

        spread_val = y[i] - alpha_val - beta_val * x[i]
        spreads[i] = spread_val

        # compute spread std over the window
        # spread_k = y_k - alpha - beta * x_k
        # mean(spread) = (sy - alpha*lb - beta*sx)/lb = 0 by construction
        # but let's compute properly
        # sum_spread = sy_win - alpha*lb - beta*sx_win = 0
        # sum_spread2 = sum(y^2) - 2*alpha*sy - 2*beta*sxy + alpha^2*lb + 2*alpha*beta*sx + beta^2*sx2
        if j == 0:
            sy2_win = cum_y2[i-1]
        else:
            sy2_win = cum_y2[i-1] - cum_y2[j-1]

        sum_sp2 = (sy2_win
                    - 2.0 * alpha_val * sy
                    - 2.0 * beta_val * sxy
                    + alpha_val * alpha_val * lookback
                    + 2.0 * alpha_val * beta_val * sx
                    + beta_val * beta_val * sx2)

        var = sum_sp2 / lookback  # mean of spread is ~0
        # more precise: var = sum_sp2/lb - (sum_sp/lb)^2, but sum_sp≈0
        if var > 1e-20:
            sigma = np.sqrt(var)
            z_scores[i] = spread_val / sigma  # mean ≈ 0
        else:
            z_scores[i] = 0.0

    idx = log_y.index
    return (pd.Series(z_scores, index=idx),
            pd.Series(betas, index=idx),
            pd.Series(spreads, index=idx))


def calc_roundtrips(z_scores, entry_z, exit_z, max_hold):
    """
    Count completed round-trips from z-score series.
    Entry: |z| >= entry_z
    Exit:  |z| <= exit_z OR holding period >= max_hold
    """
    z = z_scores.dropna().values
    dates = z_scores.dropna().index

    if len(z) < 50:
        return 0, 0, 0.0, 0.0

    n_signals = 0
    n_roundtrips = 0
    in_trade = False
    hold_count = 0

    for i in range(len(z)):
        if not in_trade:
            if abs(z[i]) >= entry_z:
                in_trade = True
                hold_count = 0
                n_signals += 1
        else:
            hold_count += 1
            if abs(z[i]) <= exit_z:
                n_roundtrips += 1
                in_trade = False
            elif hold_count >= max_hold:
                in_trade = False

    completion_rate = n_roundtrips / n_signals if n_signals > 0 else 0.0

    n_days = (dates[-1] - dates[0]).days if len(dates) > 1 else 1
    n_years = max(n_days / 365.25, 0.1)
    annual_freq = n_roundtrips / n_years

    return n_roundtrips, n_signals, completion_rate, annual_freq


def calc_halflife(spread_series):
    """Ornstein-Uhlenbeck half-life from spread series."""
    spread = spread_series.dropna().values
    if len(spread) < 20:
        return np.inf

    y = np.diff(spread)
    x = spread[:-1]

    if np.std(x) < 1e-10:
        return np.inf

    try:
        phi = np.dot(x, y) / np.dot(x, x)
    except:
        return np.inf

    if phi >= 0:
        return np.inf

    halflife = -np.log(2) / phi

    if halflife <= 0 or halflife > 1000:
        return np.inf

    return halflife


# ═══════════════════════════════════════════════════════════════════
# METRIC COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════

def scan_best_reference(log_y, log_x):
    """
    V5.1 fix: Use RT density (rt_per_bar) but add a stability bonus
    for longer lookbacks that maintain high completion rates.
    This prevents short lookbacks from dominating purely on volume
    while surfacing long-lookback pairs that are genuinely stable.
    """
    best_combined = -1
    best_result = None
    all_results = []

    n = len(log_y)

    for lb in REF_LOOKBACKS:
        z_scores, betas, spreads = compute_rolling_ols(log_y, log_x, lb)

        n_rt, n_sig, rt_comp, rt_annual = calc_roundtrips(
            z_scores, REF_ENTRY_Z, REF_EXIT_Z, REF_MAX_HOLD
        )

        # ── Normalize: RT per tradeable bar ──
        tradeable_bars = n - lb
        if tradeable_bars > 0:
            rt_per_bar = n_rt / tradeable_bars
        else:
            rt_per_bar = 0.0

        # ── Stability bonus for longer lookbacks ──
        # Pairs that maintain high completion at longer lookbacks 
        # are more likely to be structurally mean-reverting
        # Bonus: 0% at lb=100, up to 15% at lb=210
        lb_stability_bonus = 1.0 + 0.15 * (lb - REF_LOOKBACKS[0]) / (REF_LOOKBACKS[-1] - REF_LOOKBACKS[0])

        # ── Minimum activity threshold ──
        # Need at least 4 RT/year to be interesting regardless of lookback
        if rt_annual < 4.0:
            combined = 0.0
        else:
            combined = rt_per_bar * (rt_comp ** 1.5) * lb_stability_bonus

        result = {
            'lookback': lb,
            'z_scores': z_scores,
            'betas': betas,
            'spreads': spreads,
            'n_rt': n_rt,
            'n_sig': n_sig,
            'rt_comp': rt_comp,
            'rt_annual': rt_annual,
            'rt_per_bar': rt_per_bar,
            'lb_stability_bonus': lb_stability_bonus,
            'combined': combined,
        }
        all_results.append(result)

        if combined > best_combined:
            best_combined = combined
            best_result = result

    if best_result is not None:
        best_result['all_lookback_results'] = [
            {k: v for k, v in r.items() if k not in ('z_scores', 'betas', 'spreads')}
            for r in all_results
        ]

    return best_result



def calc_beta_smoothness_relative(betas):
    """Fix 3: Relative beta smoothness — delta_beta_std / |beta_mean|."""
    betas_clean = betas.dropna()

    if len(betas_clean) < 50:
        return np.nan, np.nan, np.nan, np.nan

    delta_beta = betas_clean.diff().dropna()
    delta_beta_std = np.std(delta_beta)
    beta_mean = np.mean(betas_clean)
    beta_cv = np.std(betas_clean) / abs(beta_mean) if abs(beta_mean) > 1e-6 else np.nan

    if abs(beta_mean) > 1e-4:
        delta_beta_rel = delta_beta_std / abs(beta_mean)
    else:
        delta_beta_rel = np.nan

    return delta_beta_std, delta_beta_rel, beta_mean, beta_cv


def calc_window_reversion_speed(log_y, log_x, best_lookback, window=WRS_WINDOW):
    """Percentage of rolling windows where half-life < threshold."""
    n = len(log_y)
    if n < window + best_lookback:
        return 0.0, np.inf, 0

    halflife_list = []
    step = max(window // 4, 50)

    for start in range(0, n - window - best_lookback, step):
        end = start + window + best_lookback
        if end > n:
            break

        log_y_win = log_y.iloc[start:end]
        log_x_win = log_x.iloc[start:end]

        _, _, spreads_win = compute_rolling_ols(log_y_win, log_x_win, best_lookback)
        spread_valid = spreads_win.dropna()

        if len(spread_valid) > 30:
            hl = calc_halflife(spread_valid)
            if np.isfinite(hl):
                halflife_list.append(hl)

    if not halflife_list:
        return 0.0, np.inf, 0

    pct_fast = np.mean([1 if hl < WRS_HALFLIFE_THRESHOLD else 0 for hl in halflife_list])
    median_hl = np.median(halflife_list)

    return pct_fast, median_hl, len(halflife_list)


def calc_regime_stability(z_scores, n_regimes=N_REGIMES):
    """
    Fix 5: Regime quality scoring.
    Split data into n_regimes equal parts. Score regimes that have
    >= REGIME_MIN_RTS round-trips AND completion >= REGIME_MIN_COMPLETION.
    """
    z_clean = z_scores.dropna()

    if len(z_clean) < 100:
        return 0, n_regimes, [], 0.0

    regime_len = len(z_clean) // n_regimes
    regime_details = []
    quality_regimes = 0

    for r in range(n_regimes):
        start = r * regime_len
        end = start + regime_len if r < n_regimes - 1 else len(z_clean)
        z_regime = z_clean.iloc[start:end]

        n_rt, n_sig, rt_comp, rt_annual = calc_roundtrips(
            z_regime, REF_ENTRY_Z, REF_EXIT_Z, REF_MAX_HOLD
        )

        regime_details.append({
            'regime': r + 1,
            'n_rt': n_rt,
            'n_sig': n_sig,
            'completion': rt_comp,
            'annual_freq': rt_annual,
        })

        if n_rt >= REGIME_MIN_RTS and rt_comp >= REGIME_MIN_COMPLETION:
            quality_regimes += 1

    rt_counts = [d['n_rt'] for d in regime_details]
    if np.mean(rt_counts) > 0:
        consistency = 1.0 - min(np.std(rt_counts) / np.mean(rt_counts), 2.0) / 2.0
    else:
        consistency = 0.0

    return quality_regimes, n_regimes, regime_details, consistency


def calc_spread_kurtosis(spreads):
    """Excess kurtosis of the spread distribution."""
    s = spreads.dropna()
    if len(s) < 50:
        return 0.0
    return float(stats.kurtosis(s, fisher=True))


# ═══════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def linear_score(value, worst, best, max_points):
    """Linear interpolation between worst (0 pts) and best (max_points)."""
    if best == worst:
        return max_points if value >= best else 0.0

    ratio = (value - worst) / (best - worst)
    ratio = np.clip(ratio, 0.0, 1.0)
    return ratio * max_points


def score_pair(ref_result, log_y, log_x, n_overlap_days):
    """Compute composite score for a pair using all V5.1 metrics."""
    scores = {}
    metrics = {}

    best_lb = ref_result['lookback']
    z_scores = ref_result['z_scores']
    betas = ref_result['betas']
    spreads = ref_result['spreads']

    # ── Metric 1: Round-Trip Frequency & Completion (30 pts) ──
    rt_annual = ref_result['rt_annual']
    rt_comp = ref_result['rt_comp']
    n_rt = ref_result['n_rt']
    n_sig = ref_result['n_sig']

    score_freq = linear_score(rt_annual, RT_ANNUAL_WORST, RT_ANNUAL_BEST, 18)
    score_comp = linear_score(rt_comp, RT_COMP_WORST, RT_COMP_BEST, 12)
    scores['rt_frequency'] = score_freq + score_comp

    metrics['RT_Total'] = n_rt
    metrics['RT_Signals'] = n_sig
    metrics['RT_Completion'] = round(rt_comp, 4)
    metrics['RT_Annual_Freq'] = round(rt_annual, 3)
    metrics['Best_Ref_Lookback'] = best_lb
    metrics['Lookback_Scan'] = json.dumps(ref_result.get('all_lookback_results', []))

    # ── Metric 2: Beta Smoothness — Relative (10 pts) ──
    delta_beta_std, delta_beta_rel, beta_mean, beta_cv = calc_beta_smoothness_relative(betas)

    if np.isnan(delta_beta_rel):
        scores['beta_smoothness'] = 0.0
    else:
        scores['beta_smoothness'] = linear_score(
            delta_beta_rel, BETA_REL_WORST, BETA_REL_BEST, W_BETA_SMOOTH
        )

    metrics['Delta_Beta_Std'] = round(delta_beta_std, 6) if not np.isnan(delta_beta_std) else None
    metrics['Delta_Beta_Rel'] = round(delta_beta_rel, 6) if not np.isnan(delta_beta_rel) else None
    metrics['Beta_Mean'] = round(beta_mean, 6) if not np.isnan(beta_mean) else None
    metrics['Beta_CV'] = round(beta_cv, 4) if not np.isnan(beta_cv) else None

    # ── Metric 3: Window Reversion Speed (20 pts) ──
    wrs_pct, wrs_median_hl, wrs_n_windows = calc_window_reversion_speed(
        log_y, log_x, best_lb
    )

    score_pct_fast = linear_score(wrs_pct, 0.2, 0.8, 12)
    score_median_hl = linear_score(wrs_median_hl, WRS_HALFLIFE_THRESHOLD, 2.0, 8)
    scores['window_reversion'] = score_pct_fast + score_median_hl

    metrics['WRS_Pct_Fast'] = round(wrs_pct, 4)
    metrics['WRS_Median_HL'] = round(wrs_median_hl, 2) if np.isfinite(wrs_median_hl) else None
    metrics['WRS_N_Windows'] = wrs_n_windows

    # ── Metric 4: Regime Stability (25 pts) ──
    quality_regimes, total_regimes, regime_details, consistency = calc_regime_stability(z_scores)

    score_regime_count = linear_score(quality_regimes, 0, total_regimes, 15)
    score_consistency = consistency * 10
    scores['regime_stability'] = score_regime_count + score_consistency

    metrics['Regime_Quality'] = quality_regimes
    metrics['Regime_Total'] = total_regimes
    metrics['Regime_Consistency'] = round(consistency, 4)
    metrics['Regime_Detail'] = json.dumps(regime_details)

    # ── Metric 5: ADF on Best-Reference Spread (5 pts) ──
    spread_for_adf = spreads.dropna()
    if len(spread_for_adf) > 50:
        try:
            adf_pvalue = adfuller(spread_for_adf, maxlag=20, autolag='AIC')[1]
        except:
            adf_pvalue = 1.0
    else:
        adf_pvalue = 1.0

    scores['adf'] = linear_score(adf_pvalue, ADF_WORST, ADF_BEST, W_ADF)

    metrics['ADF_Pvalue'] = round(adf_pvalue, 6)

    # ── Kurtosis Penalty (up to -5 pts) ──
    excess_kurt = calc_spread_kurtosis(spreads)

    if excess_kurt > KURTOSIS_THRESHOLD:
        penalty = min(
            (excess_kurt - KURTOSIS_THRESHOLD) / KURTOSIS_THRESHOLD * KURTOSIS_PENALTY_MAX,
            KURTOSIS_PENALTY_MAX
        )
    else:
        penalty = 0.0

    scores['kurtosis_penalty'] = -penalty
    metrics['Spread_Kurtosis'] = round(excess_kurt, 4)

    # ── Data Bonus (5 pts) ──
    if n_overlap_days >= 900:
        scores['data_bonus'] = W_DATA_BONUS
    elif n_overlap_days >= 365:
        scores['data_bonus'] = W_DATA_BONUS * (n_overlap_days - 365) / (900 - 365)
    else:
        scores['data_bonus'] = 0.0

    metrics['Overlap_Days'] = n_overlap_days

    # ── Total Score ──
    total = sum(scores.values())
    total = max(total, 0.0)

    return round(total, 2), scores, metrics


# ═══════════════════════════════════════════════════════════════════
# MAIN SCREENING LOOP
# ═══════════════════════════════════════════════════════════════════

def run_screening():
    """Main screening pipeline."""

    print("=" * 70)
    print("  PAIRS SCREENER V5.1 — Strategy-Aligned with Multi-Reference Scan")
    print("=" * 70)

    # ── Connect & Load Data ──
    print("\n[1/4] Connecting to Binance & loading price data...")
    client = get_binance_client()
    prices = load_prices(SYMBOLS, client)
    available_symbols = prices.columns.tolist()
    print(f"  Available: {len(available_symbols)} symbols")

    # ── Generate Pairs ──
    all_pairs = list(combinations(available_symbols, 2))
    print(f"\n[2/4] Screening {len(all_pairs)} pairs...")

    # ── Screen Each Pair ──
    results = []
    t_start = time.time()

    for idx, (sym_y, sym_x) in enumerate(all_pairs):

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(all_pairs) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(all_pairs)}] {sym_y}/{sym_x} "
                  f"({rate:.1f} pairs/sec, ~{remaining:.0f}s remaining)")

        # Get aligned data
        mask = prices[[sym_y, sym_x]].dropna()
        n_overlap = len(mask)

        if n_overlap < MIN_OVERLAP_DAYS:
            continue

        n_overlap_days = (mask.index[-1] - mask.index[0]).days

        if n_overlap_days < MIN_OVERLAP_DAYS:
            continue

        log_y = np.log(mask[sym_y])
        log_x = np.log(mask[sym_x])

        # ── Multi-Reference Scan ──
        ref_result = scan_best_reference(log_y, log_x)

        if ref_result is None:
            continue

        # ── Score ──
        try:
            total_score, score_detail, metrics = score_pair(
                ref_result, log_y, log_x, n_overlap_days
            )
        except Exception as e:
            print(f"  [ERROR] {sym_y}/{sym_x}: {e}")
            continue

        # ── Store Result ──
        row = {
            'Pair': f"{sym_y}-{sym_x}",
            'Symbol_Y': sym_y,
            'Symbol_X': sym_x,
            'Score': total_score,
        }

        for k, v in score_detail.items():
            row[f'Score_{k}'] = round(v, 2)

        row.update(metrics)
        results.append(row)

    elapsed_total = time.time() - t_start
    print(f"\n  Screening complete: {len(results)} pairs scored in {elapsed_total:.1f}s")

    # ── Sort and Rank ──
    df = pd.DataFrame(results)
    df = df.sort_values('Score', ascending=False).reset_index(drop=True)
    df.insert(0, 'Rank', range(1, len(df) + 1))

    # ── Save Full Results ──
    full_path = OUTPUT_DIR / "pairs_screen_v5_1_full.csv"
    df.to_csv(full_path, index=False)
    print(f"\n[3/4] Full results saved: {full_path} ({len(df)} pairs)")

    # ── Save Top N for WF Template ──
    df_top = df.head(TOP_N).copy()
    wf_path = OUTPUT_DIR / "pairs_screen_v5_1_wf_input.csv"

    wf_cols = [
        'Rank', 'Pair', 'Symbol_Y', 'Symbol_X', 'Score',
        'Best_Ref_Lookback', 'RT_Total', 'RT_Signals', 'RT_Completion',
        'RT_Annual_Freq', 'Delta_Beta_Rel', 'Beta_Mean',
        'WRS_Pct_Fast', 'WRS_Median_HL', 'Regime_Quality',
        'Regime_Consistency', 'ADF_Pvalue', 'Spread_Kurtosis', 'Overlap_Days',
    ]
    wf_cols = [c for c in wf_cols if c in df_top.columns]
    df_top[wf_cols].to_csv(wf_path, index=False)
    print(f"[4/4] WF input saved: {wf_path} ({len(df_top)} pairs)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  TOP 15 PAIRS")
    print("=" * 70)

    summary_cols = [
        'Rank', 'Pair', 'Score', 'Best_Ref_Lookback',
        'RT_Annual_Freq', 'RT_Completion', 'Delta_Beta_Rel',
        'WRS_Pct_Fast', 'Regime_Quality', 'ADF_Pvalue',
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]

    with pd.option_context('display.max_columns', 20, 'display.width', 150):
        print(df[summary_cols].head(15).to_string(index=False))

    # ── Score Distribution ──
    print(f"\n  Score Distribution:")
    print(f"    Max:    {df['Score'].max():.1f}")
    print(f"    Top 10: {df['Score'].head(10).mean():.1f} (avg)")
    print(f"    Top 25: {df['Score'].head(25).mean():.1f} (avg)")
    print(f"    Top 50: {df['Score'].head(50).mean():.1f} (avg)")
    print(f"    Median: {df['Score'].median():.1f}")
    print(f"    Min:    {df['Score'].min():.1f}")

    # ── Check for TRX/LINK ──
    trx_link = df[
        ((df['Symbol_Y'] == 'TRXUSDT') & (df['Symbol_X'] == 'LINKUSDT')) |
        ((df['Symbol_Y'] == 'LINKUSDT') & (df['Symbol_X'] == 'TRXUSDT'))
    ]
    if not trx_link.empty:
        print(f"\n  ★ TRX/LINK: Rank {trx_link.iloc[0]['Rank']}, "
              f"Score {trx_link.iloc[0]['Score']}, "
              f"Best LB {trx_link.iloc[0].get('Best_Ref_Lookback', '?')}")

    return df


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df_results = run_screening()
