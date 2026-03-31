# walk_forward.py

import pandas as pd
import numpy as np
import sys
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = r'/Users/justiniturregui/Desktop/epsilon/github/Epsilon-Quant-Research'
sys.path.append(ROOT + '/infrastructure/data')
sys.path.append(ROOT + '/infrastructure/backtester')

from binance_client import get_binance_client, get_data
from engine import backtest

# ── Data ───────────────────────────────────────────────────────────────────────
client = get_binance_client()
df     = get_data(client, 'BTCUSDT', '1d', 2150)
df     = df.reset_index()
print(f"Data loaded: {df['Time'].iloc[0].date()} → {df['Time'].iloc[-1].date()} | {len(df)} bars")

# ── Walk-forward settings ──────────────────────────────────────────────────────
TRAIN_BARS = 1050    # 3 years training per fold
TEST_BARS  = 275    #
N_TRIALS   = 400    # Optuna trials per fold

# ── Fixed parameters — anchored from cross-fold stability analysis ─────────────
# Run with FIXED_PARAMS = {} first to identify stable params via CV table
# then populate this dict with the consistent ones
FIXED_PARAMS = {
'ema_span': 21,
'adx_override': 63,
'max_leverage': 3,
'stop_mult_normal': 1,
'risk_per_trade': 0.46,
'atr_position': 13
}

# ── All parameter definitions — search ranges ──────────────────────────────────
ALL_PARAM_DEFS = {
    'ema_span':              ('int',   5,     40),#f
    'swing_caution':         ('int',   3,     14),
    'swing_stop':            ('int',   3,     10),
    'atr':    ('int',   10,    30),
    'atr_position':       ('int',   3,     14),
    'caution_threshold':     ('float', 0.8,   2.5),
    'adx_override':          ('int',   40,    80),#f
    'risk_per_trade':        ('float', 0.005, 0.05), #f
    'max_leverage':          ('float', 1.0,   3.0), #f
    'stop_mult_both':    ('float', 0.5,   2.5),
    'stop_mult_caution': ('float', 0.1,   0.9),
    'stop_mult_short':   ('float', 0.1,   1.5),
    'stop_mult_normal':  ('float', 0.8,   2.0),#f
}

INT_PARAMS = [k for k, (t, _, _) in ALL_PARAM_DEFS.items() if t == 'int']

# ── Indicator helpers ──────────────────────────────────────────────────────────
def calculate_atr(df, period):
    high_low   = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close  = np.abs(df['Low']  - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low  - close.shift())
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move   = high - high.shift()
    down_move = low.shift() - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr      = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm  = pd.Series(plus_dm,  index=close.index).ewm(alpha=1/period, adjust=False).mean()
    minus_dm = pd.Series(minus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean()

    plus_di  = 100 * (plus_dm  / atr)
    minus_di = 100 * (minus_dm / atr)

    dx  = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(period).mean()

# ── Strategy function ──────────────────────────────────────────────────────────
def run_strategy(df_slice, params, cost=0.001, test_start_idx=0):

    ema_span              = params['ema_span']
    swing_caution         = params['swing_caution']
    swing_stop            = params['swing_stop']
    atr    = params['atr']
    atr       = params['atr']
    atr_position       = params['atr_position']
    caution_threshold     = params['caution_threshold']
    adx_override          = params['adx_override']
    risk_per_trade        = params['risk_per_trade']
    max_leverage          = params['max_leverage']
    stop_mult_both    = params['stop_mult_both']
    stop_mult_caution = params['stop_mult_caution']
    stop_mult_short   = params['stop_mult_short']
    stop_mult_normal  = params['stop_mult_normal']

    s = df_slice.copy().reset_index(drop=True)
    if 'Time' in s.columns:
        s = s.set_index('Time')

    s['ema']          = s['Close'].ewm(span=ema_span, adjust=False).mean()
    s['swing_hi_cau'] = s['High'].rolling(window=swing_caution).max()
    s['swing_lo_cau'] = s['Low'].rolling(window=swing_caution).min()
    s['swing_hi_stp'] = s['High'].rolling(window=swing_stop).max()
    s['atr']      = calculate_atr(s, atr)
    s['atr_pos']       = calculate_atr(s, atr_position)
    s['adx']          = calculate_adx(s['High'], s['Low'], s['Close'], 14)

    s['caution_long'] = (
        (s['swing_hi_cau'] - s['Low']) > (caution_threshold * s['atr'])
    )
    s['caution_short'] = (
        ((s['High'] - s['swing_lo_cau']) > (caution_threshold * s['atr'])) |
        (s['Close'] > s['ema'])
    )
    s['entry_long'] = (
        (s['Close'] > s['ema']) &
        ((~s['caution_long']) | (s['adx'] > adx_override))
    )
    s['position_size'] = (
        risk_per_trade / (s['atr_pos'] / s['Close'])
    ).clip(0.1, max_leverage)

    s['position']      = 0
    s['position_size'] = s['position_size']
    s['Stop_Loss']     = np.nan
    s['Entry_Price']   = np.nan

    in_position  = 0
    stop_loss    = 0
    entry_price  = 0
    current_size = 0.0

    warmup = max(swing_caution, atr, atr_position, ema_span) + 5

    for i in range(warmup, len(s)):
        idx  = s.index[i]
        curr = s.iloc[i]
        prev = s.iloc[i - 1]

        if in_position == 1:
            if curr['caution_long']:
                sm = stop_mult_caution
            else:
                sm = stop_mult_normal
        else:
            if prev['entry_long']:
                if curr['caution_long'] and curr['caution_short']:
                    sm = stop_mult_both
                elif curr['caution_long']:
                    sm = stop_mult_caution
                elif curr['caution_short']:
                    sm = stop_mult_short
                else:
                    sm = stop_mult_normal
            else:
                sm = stop_mult_normal

        stop_distance = curr['atr'] * sm 

        if in_position == 1:
            if prev['Close'] < stop_loss:
                in_position  = 0
                current_size = 0.0
                s.loc[idx, ['position', 'position_size']] = [0, 0.0]
                continue
            stop_loss = max(stop_loss, curr['swing_hi_stp'] - stop_distance)
            s.loc[idx, ['position', 'position_size', 'Stop_Loss']] = [
                1, current_size, stop_loss
            ]
        else:
            if prev['entry_long']:
                stop_loss    = curr['swing_hi_stp'] - stop_distance
                entry_price  = curr['Close']
                in_position  = 1
                current_size = curr['position_size']
                s.loc[idx, ['position', 'position_size', 'Entry_Price', 'Stop_Loss']] = [
                    1, current_size, entry_price, stop_loss
                ]

    s.dropna(subset=['ema', 'atr', 'adx'], inplace=True)
    s['position']      = s['position'].fillna(0).astype(int)
    s['position_size'] = s['position_size'].fillna(0)
    s['Stop_Loss']     = s['Stop_Loss'].fillna(0)
    s['Entry_Price']   = s['Entry_Price'].fillna(0)
    
    # ── Trim burn-in bars — only score the true OOS period ─────────────────────
    if test_start_idx > 0:
        s = s.iloc[test_start_idx:].copy()
    
    try:
        results = backtest(
            data           = s,
            cost           = cost,
            show_plot      = False,
            save_html      = None,
            show_trades    = False,
            benchmark_data = None
        )
        return results
    except Exception:
        return None

# ── Result summariser ──────────────────────────────────────────────────────────
def summarise(result, label):
    if result is None:
        return {f'{label}_return':   None, f'{label}_sharpe':   None,
                f'{label}_drawdown': None, f'{label}_calmar':   None,
                f'{label}_trades':   None, f'{label}_winrate':  None}
    calmar = result['total_return'] / abs(result['max_drawdown']) \
             if result['max_drawdown'] != 0 else 0
    return {
        f'{label}_return':   result['total_return'],
        f'{label}_sharpe':   result['sharpe_ratio'],
        f'{label}_drawdown': result['max_drawdown'],
        f'{label}_calmar':   calmar,
        f'{label}_trades':   result['num_trades'],
        f'{label}_winrate':  result['win_rate'],
    }

def fmt(val, pct=False, decimals=2):
    if val is None:
        return 'N/A'
    return f"{val*100:.{decimals}f}%" if pct else f"{val:.{decimals}f}"

# ── Optuna objective factory ───────────────────────────────────────────────────
def make_objective(df_train, fixed_params):

    def objective(trial):

        # Build params — fixed ones bypass Optuna suggestion
        params = {}
        for name, (dtype, lo, hi) in ALL_PARAM_DEFS.items():
            if name in fixed_params:
                params[name] = fixed_params[name]
            elif dtype == 'int':
                params[name] = trial.suggest_int(name, lo, hi)
            else:
                params[name] = trial.suggest_float(name, lo, hi)

        result = run_strategy(df_train, params, cost=0.001)

        # Hard rejection filters — loosened to survive bear-market folds
        if result is None:                     return -999
        if result['num_trades']   < 20:        return -999
        if result['win_rate']     < 0.35:      return -999
        if result['max_drawdown'] < -0.80:     return -999
        if result['profit_factor'] < 0.8:      return -999

        calmar = result['total_return'] / abs(result['max_drawdown'])

        # Normalised composite score
        sharpe_norm = np.clip(result['sharpe_ratio'] / 2.5,  0, 1)
        calmar_norm = np.clip(calmar                 / 70.0, 0, 1)
        return_norm = np.clip(result['total_return'] / 15.0, 0, 1)

        score = 0.50 * sharpe_norm + 0.30 * calmar_norm + 0.20 * return_norm

        # Store raw metrics for later inspection
        trial.set_user_attr('sharpe',        result['sharpe_ratio'])
        trial.set_user_attr('calmar',        calmar)
        trial.set_user_attr('total_return',  result['total_return'])
        trial.set_user_attr('max_drawdown',  result['max_drawdown'])
        trial.set_user_attr('num_trades',    result['num_trades'])
        trial.set_user_attr('win_rate',      result['win_rate'])

        return score

    return objective

# ── Build folds ────────────────────────────────────────────────────────────────
BURNIN_BARS = 70   # enough bars to warm up the longest indicator

folds = []
start = 0
while start + TRAIN_BARS + TEST_BARS <= len(df):
    folds.append({
        'train':       df.iloc[start : start + TRAIN_BARS].copy(),
        # test slice includes trailing burn-in bars from end of training
        'test':        df.iloc[start + TRAIN_BARS - BURNIN_BARS
                               : start + TRAIN_BARS + TEST_BARS].copy(),
        'test_start_idx': BURNIN_BARS,   # ← where real test begins within the slice
        'train_start': df['Time'].iloc[start].date(),
        'train_end':   df['Time'].iloc[start + TRAIN_BARS - 1].date(),
        'test_start':  df['Time'].iloc[start + TRAIN_BARS].date(),
        'test_end':    df['Time'].iloc[start + TRAIN_BARS + TEST_BARS - 1].date(),
    })
    start += TEST_BARS

print(f"\n{len(folds)} folds defined:")
for i, f in enumerate(folds):
    print(f"  Fold {i+1}: train {f['train_start']} → {f['train_end']} | "
          f"test  {f['test_start']} → {f['test_end']}")

if FIXED_PARAMS:
    print(f"\nFixed parameters ({len(FIXED_PARAMS)}):")
    for k, v in FIXED_PARAMS.items():
        print(f"  {k}: {v}")
    print(f"Free parameters: {len(ALL_PARAM_DEFS) - len(FIXED_PARAMS)}")
else:
    print(f"\nAll {len(ALL_PARAM_DEFS)} parameters free")

# ── Run walk-forward ───────────────────────────────────────────────────────────
fold_records    = []
all_best_params = []

for fold_idx, fold in enumerate(folds):
    print(f"\n{'─'*60}")
    print(f"Fold {fold_idx + 1}/{len(folds)}")
    print(f"  Train: {fold['train_start']} → {fold['train_end']}")
    print(f"  Test:  {fold['test_start']}  → {fold['test_end']}")

    study = optuna.create_study(
        direction  = 'maximize',
        study_name = f'wf_fold_{fold_idx + 1}',
        sampler    = optuna.samplers.TPESampler(seed=42 + fold_idx)
    )
    study.optimize(
        make_objective(fold['train'], FIXED_PARAMS),
        n_trials          = N_TRIALS,
        show_progress_bar = True
    )

    best_params = study.best_params

    # Re-inject fixed params into best_params for evaluation and storage
    full_params = {**FIXED_PARAMS, **best_params}
    all_best_params.append(full_params)

    train_result = run_strategy(fold['train'], full_params, cost=0.001)
    test_result  = run_strategy(fold['test'],  full_params, cost=0.001,
                                test_start_idx=fold['test_start_idx'])

    record = {
        'fold':         fold_idx + 1,
        'train_start':  str(fold['train_start']),
        'train_end':    str(fold['train_end']),
        'test_start':   str(fold['test_start']),
        'test_end':     str(fold['test_end']),
        'optuna_score': study.best_value,
        **summarise(train_result, 'train'),
        **summarise(test_result,  'test'),
        **{f'param_{k}': v for k, v in full_params.items()}
    }
    fold_records.append(record)

    # Print fold summary
    print(f"\n  Train → Sharpe: {fmt(record['train_sharpe'])}  "
          f"Return: {fmt(record['train_return'], pct=True)}  "
          f"DD: {fmt(record['train_drawdown'], pct=True)}  "
          f"Calmar: {fmt(record['train_calmar'])}  "
          f"Trades: {record['train_trades']}")
    print(f"  Test  → Sharpe: {fmt(record['test_sharpe'])}  "
          f"Return: {fmt(record['test_return'], pct=True)}  "
          f"DD: {fmt(record['test_drawdown'], pct=True)}  "
          f"Calmar: {fmt(record['test_calmar'])}  "
          f"Trades: {record['test_trades']}")

    print(f"\n  Best params this fold:")
    print(f"  Params:  {full_params}")

# ── Aggregate OOS summary ──────────────────────────────────────────────────────
results_df = pd.DataFrame(fold_records)
print(f"\n{'═'*60}")
print("WALK-FORWARD SUMMARY")
print(f"{'═'*60}")

valid = results_df[
    results_df['test_return'].notna() &
    results_df['train_return'].notna()
]

if len(valid) == 0:
    print("WARNING: no valid folds — consider loosening rejection filters")
else:
    print(f"\nOut-of-sample across {len(valid)} folds:")
    print(f"  Avg Sharpe:            {valid['test_sharpe'].mean():.2f}")
    print(f"  Avg Return:            {valid['test_return'].mean()*100:.1f}%")
    print(f"  Avg Max Drawdown:      {valid['test_drawdown'].mean()*100:.1f}%")
    print(f"  Avg Calmar:            {valid['test_calmar'].mean():.2f}")
    print(f"  Avg Trades per fold:   {valid['test_trades'].mean():.0f}")
    print(f"  Folds profitable:      {(valid['test_return'] > 0).sum()}/{len(valid)}")

    print(f"\nDegradation (OOS / In-sample):")
    print(f"  Sharpe ratio:  {valid['test_sharpe'].mean() / valid['train_sharpe'].mean():.2f}  "
          f"(>0.50 acceptable, >0.70 good)")
    print(f"  Return:        {valid['test_return'].mean() / valid['train_return'].mean():.2f}")

# ── Parameter stability table ──────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("PARAMETER STABILITY ACROSS FOLDS")
print(f"{'─'*60}")
print(f"{'Parameter':<30} {'Median':>10} {'Std':>10} {'CV':>8}  {'Fixed?':>8}")
print(f"{'─'*30} {'─'*10} {'─'*10} {'─'*8}  {'─'*8}")

for k in ALL_PARAM_DEFS.keys():
    vals   = [p[k] for p in all_best_params]
    median = float(np.median(vals))
    std    = float(np.std(vals))
    cv     = std / abs(median) if median != 0 else 999
    fixed  = '✓' if k in FIXED_PARAMS else ''
    flag   = '  ← stable' if cv < 0.15 and k not in FIXED_PARAMS else ''
    print(f"{k:<30} {median:>10.3f} {std:>10.3f} {cv:>8.3f}  {fixed:>8}{flag}")

# ── Consensus parameters ───────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("CONSENSUS PARAMETERS (median across all folds):")
print(f"{'─'*60}")

consensus = {}
for k in ALL_PARAM_DEFS.keys():
    vals = [p[k] for p in all_best_params]
    consensus[k] = int(round(np.median(vals))) if k in INT_PARAMS \
                   else round(float(np.median(vals)), 4)

print("\nParams:  {", end="")
items = list(consensus.items())
for i, (k, v) in enumerate(items):
    comma = ", " if i < len(items) - 1 else ""
    print(f"'{k}': {v}{comma}", end="")
print("}")


# ── Cumulative OOS backtest — all folds stitched together ─────────────────────
print(f"\n{'═'*60}")
print("CUMULATIVE OOS BACKTEST")
print(f"{'═'*60}")

oos_slices = []

for fold_idx, fold in enumerate(folds):
    full_params = all_best_params[fold_idx]

    # Re-run test slice with burn-in to get the full strategy_df back
    # We need a version of run_strategy that returns the df not just metrics
    ema_span              = full_params['ema_span']
    swing_caution         = full_params['swing_caution']
    swing_stop            = full_params['swing_stop']
    atr_period            = full_params['atr']
    atr_position          = full_params['atr_position']
    caution_threshold     = full_params['caution_threshold']
    adx_override          = full_params['adx_override']
    risk_per_trade        = full_params['risk_per_trade']
    max_leverage          = full_params['max_leverage']
    stop_mult_both        = full_params['stop_mult_both']
    stop_mult_caution     = full_params['stop_mult_caution']
    stop_mult_short       = full_params['stop_mult_short']
    stop_mult_normal      = full_params['stop_mult_normal']

    s = fold['test'].copy().reset_index(drop=True)
    if 'Time' in s.columns:
        s = s.set_index('Time')

    s['ema']          = s['Close'].ewm(span=ema_span, adjust=False).mean()
    s['swing_hi_cau'] = s['High'].rolling(window=swing_caution).max()
    s['swing_lo_cau'] = s['Low'].rolling(window=swing_caution).min()
    s['swing_hi_stp'] = s['High'].rolling(window=swing_stop).max()
    s['atr']          = calculate_atr(s, atr_period)
    s['atr_pos']      = calculate_atr(s, atr_position)
    s['adx']          = calculate_adx(s['High'], s['Low'], s['Close'], 14)

    s['caution_long'] = (
        (s['swing_hi_cau'] - s['Low']) > (caution_threshold * s['atr'])
    )
    s['caution_short'] = (
        ((s['High'] - s['swing_lo_cau']) > (caution_threshold * s['atr'])) |
        (s['Close'] > s['ema'])
    )
    s['entry_long'] = (
        (s['Close'] > s['ema']) &
        ((~s['caution_long']) | (s['adx'] > adx_override))
    )
    s['position_size'] = (
        risk_per_trade / (s['atr_pos'] / s['Close'])
    ).clip(0.1, max_leverage)

    s['position']      = 0
    s['position_size'] = s['position_size']
    s['Stop_Loss']     = np.nan
    s['Entry_Price']   = np.nan

    in_position  = 0
    stop_loss    = 0
    entry_price  = 0
    current_size = 0.0

    warmup = max(swing_caution, atr_period, atr_position, ema_span) + 5

    for i in range(warmup, len(s)):
        idx  = s.index[i]
        curr = s.iloc[i]
        prev = s.iloc[i - 1]

        if in_position == 1:
            if curr['caution_long'] and curr['caution_short']:
                sm = stop_mult_both
            elif curr['caution_long']:
                sm = stop_mult_caution
            else:
                sm = stop_mult_normal
        else:
            if prev['entry_long']:
                if curr['caution_long'] and curr['caution_short']:
                    sm = stop_mult_both
                elif curr['caution_long']:
                    sm = stop_mult_caution
                elif curr['caution_short']:
                    sm = stop_mult_short
                else:
                    sm = stop_mult_normal
            else:
                sm = stop_mult_normal

        stop_distance = curr['atr'] * sm

        if in_position == 1:
            if prev['Close'] < stop_loss:
                in_position  = 0
                current_size = 0.0
                s.loc[idx, ['position', 'position_size']] = [0, 0.0]
                continue
            stop_loss = max(stop_loss, curr['swing_hi_stp'] - stop_distance)
            s.loc[idx, ['position', 'position_size', 'Stop_Loss']] = [
                1, current_size, stop_loss
            ]
        else:
            if prev['entry_long']:
                stop_loss    = curr['swing_hi_stp'] - stop_distance
                entry_price  = curr['Close']
                in_position  = 1
                current_size = curr['position_size']
                s.loc[idx, ['position', 'position_size', 'Entry_Price', 'Stop_Loss']] = [
                    1, current_size, entry_price, stop_loss
                ]

    s.dropna(subset=['ema', 'atr', 'adx'], inplace=True)
    s['position']      = s['position'].fillna(0).astype(int)
    s['position_size'] = s['position_size'].fillna(0)
    s['Stop_Loss']     = s['Stop_Loss'].fillna(0)
    s['Entry_Price']   = s['Entry_Price'].fillna(0)

    # Trim burn-in
    s = s.iloc[fold['test_start_idx']:].copy()

    oos_slices.append(s)
    print(f"  Fold {fold_idx + 1} OOS slice: {s.index[0].date()} → {s.index[-1].date()} "
          f"| {len(s)} bars")

# Concatenate all OOS slices
oos_combined = pd.concat(oos_slices)
oos_combined = oos_combined[~oos_combined.index.duplicated(keep='first')]
oos_combined = oos_combined.sort_index()

print(f"\nCombined OOS: {oos_combined.index[0].date()} → {oos_combined.index[-1].date()} "
      f"| {len(oos_combined)} bars")

# Run backtest on combined OOS
try:
    oos_results = backtest(
        data           = oos_combined,
        cost           = 0.001,
        show_plot      = True,
        save_html      = None,
        show_trades    = False,
        benchmark_data = None
    )

    calmar = oos_results['total_return'] / abs(oos_results['max_drawdown']) \
             if oos_results['max_drawdown'] != 0 else 0

    print(f"\n── Combined OOS Performance ──────────────────────────────")
    print(f"  Period:        {oos_combined.index[0].date()} → {oos_combined.index[-1].date()}")
    print(f"  Return:        {oos_results['total_return']*100:.2f}%")
    print(f"  Sharpe:        {oos_results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:  {oos_results['max_drawdown']*100:.2f}%")
    print(f"  Calmar:        {calmar:.2f}")
    print(f"  Profit Factor: {oos_results['profit_factor']:.2f}")
    print(f"  Win Rate:      {oos_results['win_rate']*100:.2f}%")
    print(f"  Num Trades:    {oos_results['num_trades']}")

except Exception as e:
    print(f"Combined OOS backtest failed: {e}")