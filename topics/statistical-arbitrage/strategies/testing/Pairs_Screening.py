"""
Pairs Screener v3 — Expanded universe + stability filters
"""

import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from binance.client import Client
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────
API_KEY = ''
API_SECRET = ''
client = Client(API_KEY, API_SECRET)

INTERVAL = '1d'
LOOKBACK = '1500 days ago UTC'

EXCLUDE = {'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'MATICUSDT'}

SYMBOLS = [
    # L1s
    'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT',
    'NEARUSDT', 'FTMUSDT', 'ATOMUSDT', 'ALGOUSDT', 'APTUSDT',
    'SUIUSDT', 'SEIUSDT', 'TIAUSDT', 'INJUSDT', 'STXUSDT',
    'ICPUSDT',
    # L2s
    'ARBUSDT', 'OPUSDT',
    # DeFi
    'LINKUSDT', 'UNIUSDT', 'AAVEUSDT', 'MKRUSDT', 'SNXUSDT', 'CRVUSDT',
    'RUNEUSDT', 'PENDLEUSDT', 'JUPUSDT', 'ENAUSDT',
    # Infrastructure / AI / Compute
    'FETUSDT', 'RENDERUSDT', 'GRTUSDT', 'FILUSDT', 'WLDUSDT',
    # Gaming / Metaverse
    'IMXUSDT',
    # RWA / New
    'ONDOUSDT',
    # Other major
    'TRXUSDT', 'ADAUSDT', 'XRPUSDT', 'LTCUSDT', 'ETCUSDT',
]

SYMBOLS = [s for s in SYMBOLS if s not in EXCLUDE]

SECTORS = {
    'ETHUSDT': 'L1', 'BNBUSDT': 'L1', 'SOLUSDT': 'L1', 'AVAXUSDT': 'L1',
    'DOTUSDT': 'L1', 'NEARUSDT': 'L1', 'FTMUSDT': 'L1', 'ATOMUSDT': 'L1',
    'ALGOUSDT': 'L1', 'APTUSDT': 'L1', 'SUIUSDT': 'L1', 'SEIUSDT': 'L1',
    'TIAUSDT': 'L1', 'INJUSDT': 'L1', 'STXUSDT': 'L1', 'ICPUSDT': 'L1',
    'ARBUSDT': 'L2', 'OPUSDT': 'L2',
    'LINKUSDT': 'DeFi', 'UNIUSDT': 'DeFi', 'AAVEUSDT': 'DeFi',
    'MKRUSDT': 'DeFi', 'SNXUSDT': 'DeFi', 'CRVUSDT': 'DeFi',
    'RUNEUSDT': 'DeFi', 'PENDLEUSDT': 'DeFi', 'JUPUSDT': 'DeFi',
    'ENAUSDT': 'DeFi',
    'FETUSDT': 'Infra', 'RENDERUSDT': 'Infra', 'GRTUSDT': 'Infra',
    'FILUSDT': 'Infra', 'WLDUSDT': 'Infra',
    'IMXUSDT': 'Gaming',
    'ONDOUSDT': 'RWA',
    'TRXUSDT': 'Other', 'ADAUSDT': 'Other', 'XRPUSDT': 'Other',
    'LTCUSDT': 'Other', 'ETCUSDT': 'Other',
}

# ─── THRESHOLDS ──────────────────────────────────────────────────────
COINT_PVAL = 0.10
HALF_LIFE_MIN = 3
HALF_LIFE_MAX = 120
ROLLING_WINDOW = 250
ROLLING_COINT_THRESHOLD = 0.15
MIN_OVERLAP_BARS = 800

# NEW stability filters
BETA_CV_MAX = 0.30            # reject if rolling beta CV exceeds this
CORR_RANGE_MAX = 0.55         # reject if correlation swings more than this
ROLLING_BETA_WINDOW = 120     # window for rolling beta / correlation


# ─── DATA PULL ───────────────────────────────────────────────────────
def get_close(symbol):
    try:
        klines = client.get_historical_klines(symbol, INTERVAL, LOOKBACK)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_vol', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['close'] = df['close'].astype(float)
        return df['close'].rename(symbol)
    except Exception as e:
        print(f"  ✗ Failed to pull {symbol}: {e}")
        return None


def pull_all_data(symbols):
    print("Pulling data...")
    series = {}
    for sym in symbols:
        s = get_close(sym)
        if s is not None and len(s) > 400:
            series[sym] = s
            print(f"  ✓ {sym}: {len(s)} bars  ({s.index[0].date()} → {s.index[-1].date()})")
        else:
            print(f"  ✗ {sym}: insufficient data, skipping")
    print(f"\nPulled {len(series)} symbols")
    return series


def align_pair(series_dict, s1, s2):
    a = series_dict[s1]
    b = series_dict[s2]
    merged = pd.concat([a, b], axis=1).dropna()
    return merged[s1], merged[s2], len(merged)


# ─── HALF-LIFE ───────────────────────────────────────────────────────
def calc_half_life(spread):
    spread = spread.dropna().values
    if len(spread) < 50:
        return np.nan
    y = spread[1:]
    x = spread[:-1]
    A = np.column_stack([x, np.ones(len(x))])
    try:
        result = np.linalg.lstsq(A, y, rcond=None)
        phi = result[0][0]
        if phi >= 1.0 or phi <= 0.0:
            return np.nan
        hl = -np.log(2) / np.log(phi)
        return hl if hl > 0 else np.nan
    except:
        return np.nan


# ─── ROLLING COINTEGRATION ──────────────────────────────────────────
def rolling_coint_ratio(log_y, log_x, window=250):
    n = len(log_y)
    if n < window + 50:
        return 0.0
    coint_count = 0
    total = 0
    step = 20
    for end in range(window, n, step):
        start = end - window
        try:
            _, pval, _ = coint(log_y.iloc[start:end], log_x.iloc[start:end])
            if pval < 0.05:
                coint_count += 1
            total += 1
        except:
            pass
    return coint_count / total if total > 0 else 0.0


# ─── NEW: STABILITY METRICS ─────────────────────────────────────────
def calc_beta_stability(log_y, log_x, window=120):
    """
    Rolling OLS beta CV — lower = more stable relationship.
    Returns (beta_cv, beta_mean, beta_std).
    """
    n = len(log_y)
    if n < window + 50:
        return np.nan, np.nan, np.nan

    betas = []
    for end in range(window, n):
        start = end - window
        ly = log_y.iloc[start:end].values
        lx = log_x.iloc[start:end].values
        A = np.column_stack([lx, np.ones(len(lx))])
        try:
            coefs = np.linalg.lstsq(A, ly, rcond=None)[0]
            betas.append(coefs[0])
        except:
            pass

    if len(betas) < 50:
        return np.nan, np.nan, np.nan

    betas = np.array(betas)
    beta_mean = np.mean(betas)
    beta_std = np.std(betas)

    if abs(beta_mean) < 1e-6:
        return np.nan, beta_mean, beta_std

    beta_cv = beta_std / abs(beta_mean)
    return beta_cv, beta_mean, beta_std


def calc_corr_stability(log_y, log_x, window=120):
    """
    Rolling correlation range and std — lower = more stable.
    Returns (corr_range, corr_std, corr_mean).
    """
    ret_y = log_y.diff().dropna()
    ret_x = log_x.diff().dropna()

    merged = pd.concat([ret_y, ret_x], axis=1).dropna()
    if len(merged) < window + 50:
        return np.nan, np.nan, np.nan

    rolling_corr = merged.iloc[:, 0].rolling(window).corr(merged.iloc[:, 1]).dropna()

    if len(rolling_corr) < 50:
        return np.nan, np.nan, np.nan

    corr_range = rolling_corr.max() - rolling_corr.min()
    corr_std = rolling_corr.std()
    corr_mean = rolling_corr.mean()

    return corr_range, corr_std, corr_mean


# ─── MAIN SCREEN ─────────────────────────────────────────────────────
def screen_pairs(series_dict):
    symbols = list(series_dict.keys())
    all_pairs = list(combinations(symbols, 2))
    print(f"Testing {len(all_pairs)} pairs from {len(symbols)} symbols...\n")

    results = []

    for i, (s1, s2) in enumerate(all_pairs):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(all_pairs)}")

        # ── Pairwise alignment ──
        try:
            y_prices, x_prices, overlap = align_pair(series_dict, s1, s2)
        except:
            continue

        if overlap < MIN_OVERLAP_BARS:
            continue

        log_y = np.log(y_prices)
        log_x = np.log(x_prices)

        # ── Cointegration ──
        try:
            _, pval, _ = coint(log_y, log_x)
        except:
            continue

        # ── OLS spread ──
        A = np.column_stack([log_x.values, np.ones(len(log_x))])
        b = log_y.values
        try:
            coefs = np.linalg.lstsq(A, b, rcond=None)[0]
            beta = coefs[0]
            alpha = coefs[1]
        except:
            continue

        spread = log_y - (alpha + beta * log_x)

        # ── Half-life ──
        hl = calc_half_life(spread)

        # ── Rolling cointegration ──
        roll_ratio = rolling_coint_ratio(log_y, log_x, ROLLING_WINDOW)

        # ── ADF on spread ──
        try:
            _, adf_pval, _, _, _, _ = adfuller(spread.dropna(), maxlag=20)
        except:
            adf_pval = 1.0

        # ── NEW: Stability metrics ──
        beta_cv, beta_mean, beta_std = calc_beta_stability(log_y, log_x, ROLLING_BETA_WINDOW)
        corr_range, corr_std, corr_mean = calc_corr_stability(log_y, log_x, ROLLING_BETA_WINDOW)

        # ── Sector ──
        sec1 = SECTORS.get(s1, 'Unknown')
        sec2 = SECTORS.get(s2, 'Unknown')
        same_sector = sec1 == sec2

        # ── Pass/Fail ──
        pass_coint = pval < COINT_PVAL
        pass_hl = (not np.isnan(hl)) and HALF_LIFE_MIN <= hl <= HALF_LIFE_MAX
        pass_rolling = roll_ratio >= ROLLING_COINT_THRESHOLD
        pass_beta_cv = (not np.isnan(beta_cv)) and beta_cv <= BETA_CV_MAX
        pass_corr = (not np.isnan(corr_range)) and corr_range <= CORR_RANGE_MAX

        results.append({
            'Pair': f"{s1} / {s2}",
            'Symbol_Y': s1,
            'Symbol_X': s2,
            'Sector_Y': sec1,
            'Sector_X': sec2,
            'Same_Sector': same_sector,
            'Overlap_Bars': overlap,
            'Coint_pval': round(pval, 4),
            'Half_Life': round(hl, 1) if not np.isnan(hl) else np.nan,
            'Beta': round(beta, 4),
            'Beta_CV': round(beta_cv, 3) if not np.isnan(beta_cv) else np.nan,
            'Beta_Std': round(beta_std, 4) if not np.isnan(beta_std) else np.nan,
            'Corr_Range': round(corr_range, 3) if not np.isnan(corr_range) else np.nan,
            'Corr_Std': round(corr_std, 3) if not np.isnan(corr_std) else np.nan,
            'Corr_Mean': round(corr_mean, 3) if not np.isnan(corr_mean) else np.nan,
            'Rolling_Coint_%': round(roll_ratio * 100, 1),
            'ADF_pval': round(adf_pval, 4),
            'Pass_Coint': pass_coint,
            'Pass_HL': pass_hl,
            'Pass_Rolling': pass_rolling,
            'Pass_BetaCV': pass_beta_cv,
            'Pass_Corr': pass_corr,
        })

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("No pairs computed.")
        return df

    # ── DIAGNOSTICS ──
    print("\n" + "=" * 80)
    print("DIAGNOSTIC BREAKDOWN")
    print("=" * 80)
    print(f"Total pairs with ≥{MIN_OVERLAP_BARS} bars overlap: {len(df)}")
    print(f"Pass cointegration (p<{COINT_PVAL}):         {df['Pass_Coint'].sum()}")
    print(f"Pass half-life ({HALF_LIFE_MIN}-{HALF_LIFE_MAX}d):             {df['Pass_HL'].sum()}")
    print(f"Pass rolling coint (≥{int(ROLLING_COINT_THRESHOLD*100)}%):         {df['Pass_Rolling'].sum()}")
    print(f"Pass beta CV (≤{BETA_CV_MAX}):                {df['Pass_BetaCV'].sum()}")
    print(f"Pass corr range (≤{CORR_RANGE_MAX}):            {df['Pass_Corr'].sum()}")
    print(f"Pass original 3 (coint+HL+rolling):       {(df['Pass_Coint'] & df['Pass_HL'] & df['Pass_Rolling']).sum()}")
    print(f"Pass ALL 5 filters:                        {(df['Pass_Coint'] & df['Pass_HL'] & df['Pass_Rolling'] & df['Pass_BetaCV'] & df['Pass_Corr']).sum()}")

    # ── TOP 25 by coint p-value ──
    print("\n" + "-" * 80)
    print("TOP 25 BY COINTEGRATION P-VALUE")
    print("-" * 80)
    top = df.sort_values('Coint_pval').head(25)
    cols_top = ['Pair', 'Same_Sector', 'Overlap_Bars', 'Coint_pval', 'Half_Life',
                'Rolling_Coint_%', 'Beta_CV', 'Corr_Range', 'ADF_pval']
    print(top[cols_top].to_string(index=False))

    # ── PASS ALL 5 ──
    pass_all = df[
        df['Pass_Coint'] & df['Pass_HL'] & df['Pass_Rolling'] &
        df['Pass_BetaCV'] & df['Pass_Corr']
    ]
    cols_detail = ['Pair', 'Same_Sector', 'Overlap_Bars', 'Coint_pval', 'Half_Life',
                   'Rolling_Coint_%', 'Beta_CV', 'Corr_Range', 'Corr_Mean', 'Beta', 'ADF_pval']

    if len(pass_all) > 0:
        pass_all = pass_all.sort_values(
            ['Rolling_Coint_%', 'Coint_pval'],
            ascending=[False, True]
        )
        print("\n" + "=" * 80)
        print(f"🟢 PASS ALL 5 FILTERS ({len(pass_all)} pairs)")
        print("=" * 80)
        print(pass_all[cols_detail].to_string(index=False))
    else:
        print("\n🔴 NO PAIRS PASS ALL 5 FILTERS")

    # ── PASS ORIGINAL 3 BUT FAIL STABILITY ──
    pass_orig = df[df['Pass_Coint'] & df['Pass_HL'] & df['Pass_Rolling']]
    fail_stability = pass_orig[~(pass_orig['Pass_BetaCV'] & pass_orig['Pass_Corr'])]
    if len(fail_stability) > 0:
        fail_stability = fail_stability.sort_values('Coint_pval')
        print("\n" + "-" * 80)
        print(f"🟡 PASS ORIGINAL 3, FAIL STABILITY ({len(fail_stability)} pairs) — these would have been your old candidates")
        print("-" * 80)
        print(fail_stability[cols_detail].head(15).to_string(index=False))

    # ── PASS 4 OF 5 (near misses) ──
    df['Filters_Passed'] = (
        df['Pass_Coint'].astype(int) +
        df['Pass_HL'].astype(int) +
        df['Pass_Rolling'].astype(int) +
        df['Pass_BetaCV'].astype(int) +
        df['Pass_Corr'].astype(int)
    )
    near = df[df['Filters_Passed'] == 4].sort_values('Coint_pval')
    if len(near) > 0:
        print("\n" + "-" * 80)
        print(f"🟠 PASS 4 OF 5 FILTERS ({len(near)} pairs)")
        print("-" * 80)
        # Show which filter they fail
        for _, row in near.head(15).iterrows():
            failed = []
            if not row['Pass_Coint']: failed.append('Coint')
            if not row['Pass_HL']: failed.append('HL')
            if not row['Pass_Rolling']: failed.append('Rolling')
            if not row['Pass_BetaCV']: failed.append(f"BetaCV({row['Beta_CV']})")
            if not row['Pass_Corr']: failed.append(f"CorrRng({row['Corr_Range']})")
            fail_str = ', '.join(failed)
            print(f"  {row['Pair']:30s}  Coint={row['Coint_pval']:.4f}  HL={row['Half_Life']}  "
                  f"Roll={row['Rolling_Coint_%']}%  BetaCV={row['Beta_CV']}  CorrRng={row['Corr_Range']}  "
                  f"FAIL: {fail_str}")

    # ── SAVE ──
    df.to_csv('pairs_screen_v3_full.csv', index=False)
    if len(pass_all) > 0:
        pass_all.to_csv('pairs_screen_v3_passed.csv', index=False)
    print(f"\nSaved to pairs_screen_v3_full.csv")

    return df


# ─── RUN ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    series_dict = pull_all_data(SYMBOLS)
    results = screen_pairs(series_dict)
