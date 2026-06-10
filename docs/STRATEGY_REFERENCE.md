---
title: "Crypto-Momentum Research Stack — Strategy Reference"
created: 2026-06-05
status: active
owner: justin
project: crypto
para: resource
hubs:
  - STRATEGY_REFERENCE
tags:
  - crypto
  - research
---
# Crypto-Momentum Research Stack — Strategy Reference

_Compiled 2026-05-11. Re-stamp this header on any major refactor of the engine, metrics, or active strategy set._

This document is a formal reference for the crypto-momentum research stack: the walk-forward and CPCV engines, the strategies that feed them, the metric definitions they emit, and the portfolio module that aggregates per-asset results. Every claim is backed by a `path/to/file.py:LINE` reference into the actual source.

---

## A. Stack scope

### A.1 Codebase paths

| Component | Path |
|---|---|
| Walk-forward engine | `infrastructure/walkforward/wf_engine.py` |
| CPCV engine | `infrastructure/walkforward/cpcv_engine.py` |
| CPCV portfolio | `infrastructure/walkforward/cpcv_portfolio.py` |
| Cross-sectional strategy factory | `infrastructure/walkforward/xs_strategy.py` |
| CPCV visualiser | `infrastructure/walkforward/cpcv_visualizer.py` |
| Backtester engine | `infrastructure/backtester/engine.py` |
| Single-asset metrics | `infrastructure/backtester/performance_metrics.py` |
| Portfolio metrics + sweeps | `infrastructure/backtester/portfolio_metrics.py` |
| Live momentum strategies | `live_trading/dashboards/momentum/strategies.py` |
| BB breakout (research) | `topics/momentum/strategies/bb_breakout_wf/strategy_design/bb_breakout.py` |
| Live momentum config | `live_trading/dashboards/momentum/config.py` |

### A.2 Primary entry points

**Most up-to-date research (use these first):**

- `topics/momentum/strategies/momentum_cpcv/` — per-asset CPCV notebooks for the live momentum family (`ADA.ipynb`, `AVAX.ipynb`, `BNB.ipynb`, `BTC.ipynb`, `ETH.ipynb`, `SOL.ipynb`, `XRP.ipynb`) plus `cpcv_template.ipynb` and `portfolio_cpcv.ipynb`; see [[topics/momentum/strategies/momentum_cpcv/README|momentum CPCV README]].
- `topics/momentum/strategies/bb_cpcv/` — per-asset CPCV notebooks for the BB breakout strategy (`ADA`, `AVAX`, `BTC`, `ETH`, `LINK`, `MATIC`, `NEAR`) plus `cpcv_template.ipynb`, `portfolio_cpcv.ipynb`, and a `wf_params/` subfolder for upstream walk-forward output; see [[topics/momentum/strategies/bb_cpcv/README|BB CPCV README]].
- `topics/momentum/xs_momentum/` — **in-progress** cross-sectional momentum research (`xs_3_3_i_r2_d.ipynb`, `xs_3_2_i_r2.ipynb`, etc.) using `infrastructure/walkforward/xs_strategy.py::make_xs_strategy`. Not yet productionised.

**Older WF-only research (kept for reference, superseded by CPCV folders):**

- `topics/momentum/strategies/wf_testing/` — per-asset walk-forward notebooks (`momentumETH_wf.ipynb`, `momentumBTC_wf.ipynb`, ...). These are the source of `strategy_fn`, `PARAM_DEFS`, and `FIXED_PARAMS` pasted into the corresponding CPCV notebooks; current design notes live in [[topics/momentum/strategies/wf_testing_2/README|wf_testing_2 README]].
- `topics/momentum/strategies/bb_breakout_wf/` — BB breakout walk-forward research, source for `bb_cpcv/`; see [[topics/momentum/strategies/bb_breakout_wf/README|BB breakout WF README]].

**Production (live):**

- `live_trading/dashboards/momentum/streamlit_app.py` — Streamlit dashboard running on the active universe.
- `live_trading/dashboards/momentum/optimise.py` — walk-forward optimiser; writes `live_params.json` per asset.
- `live_trading/dashboards/momentum/strategies.py` — the two production callables: `momentum_swing` (ETH, SOL, BNB, ADA, XRP — volume-MA filter) and `momentum_no_vol` (AVAX, BTC — no volume filter), see `live_trading/dashboards/momentum/strategies.py:15` and `:108`.

**Related topic notes:**

- Momentum primers and topic hub: [[topics/momentum/README|momentum README]], [[topics/momentum/research/other-notes/W1|momentum week 1]], [[topics/momentum/research/other-notes/W2|momentum week 2]].
- Data/artifact map: [[docs/CRYPTO_DATA_MANIFEST|crypto data manifest]].
- ML and regime filters: [[topics/ml-prediction/notebooks/README|ML prediction README]], [[topics/regime-classifier/README|regime classifier README]].
- Stat-arb research: [[topics/statistical-arbitrage/README|stat-arb README]], [[topics/statistical-arbitrage/strategies/testing/README|pairs trading README]].
- Research topic stubs: [[topics/long-short/README|long-short README]], [[topics/memecoin-defi/README|memecoin DeFi README]].

**Active universe** (from `live_trading/dashboards/momentum/config.py:29-36`):

```python
ACTIVE_ASSETS = [
    "ADAUSDT",
    "AVAXUSDT",
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
]
```

### A.3 Dependencies

Live-trading subproject (`live_trading/requirements.txt:1-7`):

```
streamlit>=1.37.0
plotly
pandas
numpy
pyarrow
python-binance
optuna
```

No repo-root `requirements.txt` or `pyproject.toml`. The walk-forward / CPCV engines additionally import:

- `optuna` (TPE sampler — `wf_engine.py:6`, `cpcv_engine.py:7`)
- `scipy.stats` (Pearson r + t-distribution CIs — `cpcv_engine.py:31`, optional; falls back to a normal-CDF approximation when absent)

Python version is not pinned in the repo. The engines use modern syntax (`dict | None` unions in `portfolio_metrics.py:122-123`) so **Python ≥3.10** is required.

### A.4 Status snapshot (2026-05-11)

| Sub-stack | Status |
|---|---|
| `wf_engine.py` | Stable; used by both old WF notebooks and as the inner loop of CPCV |
| `cpcv_engine.py` | Stable; canonical entry point for new strategy research |
| `cpcv_portfolio.py` | Stable; bootstraps portfolio paths from per-asset CPCV pickles |
| `momentum_cpcv/` | Active research target; 7 assets covered |
| `bb_cpcv/` | Active research target; 7 assets covered |
| `xs_momentum/` | In progress — not yet finalised |
| Live momentum dashboard | Production; trades the 6 assets in `ACTIVE_ASSETS` |
| Live BB-breakout dashboard | Stub (`live_trading/dashboards/bbbreakout/`) — research not yet wired to live |
| Live stat-arb dashboard | Stub (`live_trading/dashboards/statarb/`) — not in scope for this doc |

---

## B. Engine reference: `wf_engine.py`

### B.1 Walk-forward loop structure

The fold builder is at `wf_engine.py:283-297`:

```python
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
```

**Defaults** (`wf_engine.py:230-240`): `train_bars=730`, `test_bars=365`, `burnin_bars=60`, step = `test_bars` (folds are non-overlapping on the test side; training windows slide forward by one test window).

**Embargo / purge:** there is **no explicit embargo** between train end and test start. The leak boundary is instead controlled by the `burnin_bars` prepended to the test slice (`wf_engine.py:289-290`), which are used to warm up indicators and then trimmed before evaluation (`wf_engine.py:359-361`):

```python
# trim burnin bars — keep only the real OOS window
test_start = fold['test_start']
oos_df = oos_df.loc[oos_df.index >= test_start].copy()
```

CPCV adds a separate `purge_bars` parameter for embargo on either side of each test group; see Section F.

### B.2 Optuna TPE configuration

Per-fold study creation at `wf_engine.py:322-333`:

```python
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
```

- **Sampler:** `optuna.samplers.TPESampler` with `seed = seed_base + i` (deterministic per fold; default `seed_base=42` — `wf_engine.py:238`).
- **TPE hyperparameters:** all defaults — no `n_startup_trials`, `n_ei_candidates`, `multivariate`, or `gamma` are overridden.
- **Pruner:** **none**. Trials run to completion (the strategy/backtest is fast enough that pruning has not been needed).
- **n_trials:** default 400 per fold (`wf_engine.py:234`); production momentum WF config in `live_trading/dashboards/momentum/config.py:5-11` uses the same 400.
- **Parallelism:** none in WF (single-threaded `study.optimize`). CPCV adds `n_jobs` (`cpcv_engine.py:227`, `:319`).

### B.3 Objective function — what is actually being maximised

The trial-level objective is built by `_make_objective` (`wf_engine.py:82-133`). After running the strategy and the backtest, it calls `score_fn(m)` (defaulting to `_default_score`). The verbatim default scoring at `wf_engine.py:21-38`:

```python
def _default_score(metrics):
    """
    Normalised composite: Sharpe 50% | Calmar 30% | Return 20%
    All components clipped to [0, 1] before weighting.
    Tune *_MAX caps to fit your strategy's realistic range.
    """
    SHARPE_MAX = 2.5
    CALMAR_MAX = 60.0
    RETURN_MAX = 15.0

    calmar = (metrics['total_return'] / abs(metrics['max_drawdown'])
              if metrics['max_drawdown'] != 0 else 0.0)

    s = np.clip(metrics['sharpe_ratio']   / SHARPE_MAX, 0, 1)
    c = np.clip(calmar                    / CALMAR_MAX, 0, 1)
    r = np.clip(metrics['total_return']   / RETURN_MAX, 0, 1)

    return 0.50 * s + 0.30 * c + 0.20 * r
```

**This is not Sharpe maximisation.** It is a hand-tuned weighted blend of three saturating components. Two subtle points:

1. The `calmar` term inside `_default_score` is computed inline as `total_return / |max_drawdown|` — **NOT annualised**. The `_calmar()` helper at `wf_engine.py:73-78` (which reads `metrics['calmar_ratio']` from `performance_metrics`) is annualised and is used only for **reporting** (`_metrics_to_row`, fold-summary prints, `oos_metrics`). Scoring and reporting calmars diverge by the annualisation factor; this is intentional saturation behaviour (caps tuned to the unannualised number), but worth knowing.
2. `metrics['total_return']` is cumulative return over the training window, not annualised. With `train_bars=730` (≈ 2y daily) and `RETURN_MAX=15.0`, the return component saturates at +1500% over the window.

Trials that fail validation are short-circuited to `-999.0`. Default reject (`wf_engine.py:41-48`):

```python
def _default_reject(metrics):
    """Returns True if this trial should be discarded (score → -999)."""
    if metrics is None:              return True
    if metrics['num_trades']   < 7: return True
    if metrics['win_rate']     < 0.35: return True
    if metrics['max_drawdown'] < -0.80: return True
    if metrics['profit_factor'] < 0.8:  return True
    return False
```

Both `score_fn` and `reject_fn` are user-overridable via kwargs to `walk_forward` (`wf_engine.py:236-237`).

### B.4 Result artifacts

#### B.4.1 `walk_forward()` return value (`wf_engine.py:453-461`)

```python
return {
    'fold_records':     fold_records,
    'results_df':       results_df,
    'all_best_params':  all_best_params,
    'consensus_params': cp,
    'stability_df':     stability_df,
    'oos_combined_df':  oos_combined,
    'oos_metrics':      oos_metrics,
}
```

#### B.4.2 `results_df` / `fold_records` columns

Built by `_metrics_to_row` (`wf_engine.py:136-150`) and the fold loop (`wf_engine.py:369-379`). Each fold row contains:

- `fold`, `train_start`, `train_end`, `test_start`, `test_end` — fold identification.
- `optuna_score` — best `_default_score` value reached in the training study.
- `train_return`, `train_sharpe`, `train_drawdown`, `train_calmar`, `train_trades`, `train_winrate`, `train_profit_factor` — IS metrics on the best params.
- `test_return`, `test_sharpe`, `test_drawdown`, `test_calmar`, `test_trades`, `test_winrate`, `test_profit_factor` — OOS metrics on the held-out window after burn-in trimming.
- `param_<name>` for every parameter — the per-fold best param values.

Optional CSV persistence is controlled by `save_csv` (`wf_engine.py:397-400`).

#### B.4.3 `oos_combined_df` and `*_oos.pkl`

`oos_combined_df` (`wf_engine.py:434-436`) is the stitched per-bar OOS strategy DataFrame across all folds, de-duplicated and sorted:

```python
oos_combined = pd.concat(oos_slices)
oos_combined = oos_combined[~oos_combined.index.duplicated(keep='first')].sort_index()
```

Per-asset notebooks then pickle either the stitched OOS dataframe or the full results dict. The convention documented in `topics/momentum/strategies/bb_cpcv/README.md:24-28` is `oos/{symbol.lower()}_{interval}_cpcv.pkl` (e.g. `btcusdt_1h_cpcv.pkl`). Pickle contents are whatever the strategy `strategy_fn` returns as the first element of its `(strategy_df, indicator_cols)` tuple — typically OHLCV plus `position`, `position_size`, `stop_loss`, plus any strategy-specific indicator columns.

#### B.4.4 `stability_df` and `consensus_params`

`consensus_params` (`wf_engine.py:200-208`): the elementwise median of each parameter across all folds, cast back to `int` for `int`-typed params. This is the production parameter set.

`stability_df` (`wf_engine.py:210-220`): one row per parameter, with `median`, `std`, `cv = std / |median|`, `fixed`, `stable` (`cv < 0.15`). Used to decide which params to freeze in the next iteration cycle.

#### B.4.5 `plateau_summary` output

`plateau_summary` (`wf_engine.py:549-662`) consumes the output of `plateau_analysis` (1-D parameter sensitivity sweeps, `wf_engine.py:467-547`) and returns a DataFrame with one row per parameter:

- `param` — parameter name.
- `plateau_pct` — fraction of the sweep range that retains ≥ `(1 - threshold)` of the peak score (default threshold `0.20`).
- `cv_fold` — per-fold CV from `stability_df`, if supplied (`wf_engine.py:566`).
- `verdict` — `Robust` (≥ 60% plateau), `Moderate` (30–60%), or `FRAGILE` (< 30%), per `wf_engine.py:592-594`.

Rows are sorted: invalid trials last, then descending `plateau_pct`, then descending `peak_score` (`wf_engine.py:606-610`).

#### B.4.6 Other persisted outputs

- `perturbation_test()` (`wf_engine.py:668-767`): random multi-param perturbation around `base_params`; returns a DataFrame with `offset_pct`, `n_valid`, `mean_score`, `median_score`, `std_score`, `min_score`, `degradation`.
- `cost_stress_test()` (`wf_engine.py:774-822`): re-runs the combined OOS backtest at escalating cost multipliers and returns Sharpe / Return / Calmar / Profit-factor at each.

---

## C. Strategy reference

The active strategies in the crypto-momentum stack. Old / out-of-scope strategies (J5/J6/J7 from earlier research generations, the Supertrend pullback and EMA crossover prototypes in `trade_pullback.ipynb`) are deliberately omitted — they have been superseded by the CPCV folders below.

### C.1 `momentum_swing` — live production, volume-MA filter

**File:** `live_trading/dashboards/momentum/strategies.py:15-105`. Universe: **ETH, SOL, BNB, ADA, XRP**.

#### Signal logic

Indicator block (`strategies.py:21-49`):

```python
df['EMA']          = df['Close'].ewm(span=params['ema_span'], adjust=False).mean()
df['Swing_Hi_Cau'] = df['High'].rolling(params['swing_caution']).max()
df['Swing_Lo_Cau'] = df['Low'].rolling(params['swing_caution']).min()
df['Swing_Hi_Stp'] = df['High'].rolling(params['swing_stop']).max()
# ... ATR_Cau / ATR_Stp / ATR_Sz via Wilder-style EMA of True Range ...
# ... ADX_14, OBV, OBV_MA, Vol_MA ...
```

Caution and entry conditions (`strategies.py:51-56`):

```python
df['Caution_OBV']   = (df['Close'] > df['Close'].shift(params['obv_lookback'])) & (df['OBV'] < df['OBV_MA'])
df['Caution_Long']  = ((df['Swing_Hi_Cau'] - df['Low']) > 1.5 * df['ATR_Cau']) | df['Caution_OBV']
df['Caution_Short'] = ((df['High'] - df['Swing_Lo_Cau']) > 1.5 * df['ATR_Cau']) | (df['Close'] > df['EMA'])
_valid = df['Swing_Hi_Stp'].notna() & df['ATR_Stp'].notna() & df['ATR_Sz'].notna() & df['OBV_MA'].notna() & df['Vol_MA'].notna()
df['Entry_Long']    = (df['Close'] > df['EMA']) & (~df['Caution_Long'] | (df['ADX_14'] > params['adx_override'])) & (df['Volume'] > df['Vol_MA']) & _valid
df['position_size_raw'] = (params['risk_per_trade'] / (df['ATR_Sz'] / df['Close'])).clip(0.1, params['max_leverage'])
```

The strategy is **long-only**. `Entry_Long` requires:

1. Trend filter: `Close > EMA`.
2. Caution gate: either `~Caution_Long` is true, **or** ADX > `adx_override` (high-trend bypass of the caution flag).
3. Volume filter: `Volume > Vol_MA`.
4. Indicator validity (no NaNs in any of `Swing_Hi_Stp`, `ATR_Stp`, `ATR_Sz`, `OBV_MA`, `Vol_MA`).

State machine for entry / exit / trailing stop (`strategies.py:58-93`):

```python
for i in range(1, n):
    prev = df.iloc[i - 1]
    curr = df.iloc[i]

    # 1. First, check if we need to exit an existing position
    if in_position == 1:
        if prev['Close'] < stop_loss:
            in_position  = 0
            current_size = 0.0
            stop_loss    = np.nan
        else:
            # If no exit, update the trailing stop as usual
            sm        = params['stop_mult_pos_caution'] if curr['Caution_Long'] else params['stop_mult_pos_normal']
            stop_loss = max(stop_loss, curr['Swing_Hi_Stp'] - curr['ATR_Stp'] * sm * params['stop_atr_scale'])

    # 2. Second, check if we should enter (even if we just exited above!)
    if in_position == 0:
        if curr['Entry_Long']:
            in_position  = 1
            current_size = curr['position_size_raw']
            cl = curr['Caution_Long']; cs = curr['Caution_Short']
            if cl and cs: sm = params['stop_mult_ent_both']
            elif cl:      sm = params['stop_mult_ent_caution']
            else:         sm = params['stop_mult_ent_normal']
            stop_loss = curr['Swing_Hi_Stp'] - curr['ATR_Stp'] * sm * params['stop_atr_scale']
    position[i]      = in_position
    position_size[i] = current_size
    stop_arr[i]      = stop_loss
```

Exit fires when the **previous** close falls below the trailing stop. Stops only ratchet up (`max(stop_loss, …)` — `strategies.py:79`).

#### Parameters

Parameters are read straight from the live params JSON; the strategy code does **not** carry a `PARAM_SPACE` of its own. The optimisation `PARAM_DEFS` live in `topics/momentum/strategies/wf_testing/momentum<ASSET>_wf.ipynb` and the corresponding `momentum_cpcv/<ASSET>.ipynb`. The names referenced by `momentum_swing` (read directly off `params['…']`):

| Param | Role |
|---|---|
| `ema_span` | EMA span on Close — primary trend filter |
| `swing_caution` | Rolling window for `Swing_Hi_Cau` / `Swing_Lo_Cau` (caution flag) |
| `swing_stop` | Rolling window for `Swing_Hi_Stp` (stop reference) |
| `atr_caution`, `atr_stop`, `atr_size` | ATR periods for caution flag, stop sizing, position sizing |
| `vol_ma_period` | Rolling-mean window for `Vol_MA` |
| `obv_ma_period`, `obv_lookback` | OBV smoothing window and lookback for `Caution_OBV` |
| `adx_override` | ADX value above which caution flag is bypassed |
| `stop_mult_ent_normal`, `stop_mult_ent_caution`, `stop_mult_ent_both` | Stop multipliers at entry (normal / caution / both-cautions) |
| `stop_mult_pos_caution`, `stop_mult_pos_normal` | Stop multipliers during position holding |
| `stop_atr_scale` | Global scale on the ATR component of the stop |
| `risk_per_trade` | Risk fraction; sized as `risk_per_trade / (ATR_Sz / Close)`, clipped to `[0.1, max_leverage]` |
| `max_leverage` | Upper clip on position size |

#### Regime filter

No top-down regime filter is applied inside `momentum_swing`. The closest equivalent is the `Caution_Long` flag, which combines a swing-range-vs-ATR test with the `Caution_OBV` divergence test (`strategies.py:51-52`).

#### Position sizing rule

`strategies.py:56` — risk-budget sizing:

```python
df['position_size_raw'] = (params['risk_per_trade'] / (df['ATR_Sz'] / df['Close'])).clip(0.1, params['max_leverage'])
```

Volatility-targeted: bigger ATR relative to price ⇒ smaller position. Floor at `0.1`, ceiling at `max_leverage`. Frozen at entry (`strategies.py:84` — `current_size = curr['position_size_raw']`) and not adjusted intra-trade.

#### Stop / target rule

- Stop set at entry from `Swing_Hi_Stp - ATR_Stp × stop_mult_ent_<regime> × stop_atr_scale` (`strategies.py:85-90`).
- Stop ratcheted up each bar while in position (`strategies.py:78-79`), never down.
- **No explicit profit target** — the trailing stop is the only exit.

#### Return value

`(df, indicator_cols)` where `indicator_cols = ['EMA', 'ATR_Cau', 'ADX_14', 'Swing_Hi_Cau', 'Vol_MA', 'OBV_MA']` (`strategies.py:100`). The engine uses `indicator_cols` to selectively drop rows with NaN warmup values (see Section G).

### C.2 `momentum_no_vol` — live production, no volume filter

**File:** `live_trading/dashboards/momentum/strategies.py:108-205`. Universe: **AVAX, BTC**.

Structurally identical to `momentum_swing` with three differences:

1. The `Vol_MA` indicator is **not computed**, and the `Entry_Long` condition drops the `(df['Volume'] > df['Vol_MA'])` clause (`strategies.py:150-154`):

   ```python
   df['Entry_Long'] = (
       (df['Close'] > df['EMA']) &
       (~df['Caution_Long'] | (df['ADX_14'] > params['adx_override'])) &
       _valid
   )
   ```
2. `_valid` (`strategies.py:146-149`) excludes the `Vol_MA.notna()` clause.
3. `indicator_cols` drops `Vol_MA` (`strategies.py:200`).

Parameter set, regime filter, sizing rule, and stop rule are **identical** to `momentum_swing`. The two functions are kept distinct for clarity; the registry maps coins to one or the other (`strategies.py:209-212`).

### C.3 `BBBreakout` — research only, 4H + 1H two-timeframe

**File:** `topics/momentum/strategies/bb_breakout_wf/strategy_design/bb_breakout.py:81-270`. Universe per CPCV research: BTC, ETH, ADA, SOL, AVAX, DOT, LINK, MATIC/POL, NEAR (`topics/momentum/strategies/bb_cpcv/README.md:11-22`). **Not yet wired to a live dashboard** (`live_trading/dashboards/bbbreakout/` is still a stub).

#### Signal logic — 4H setup

Vectorised 4H conditions (`bb_breakout.py:127-148`):

```python
h4_atr      = atr(h4, params["atr_period"])
h4_range    = candle_range(h4)
h4_bw       = bb_width(h4, params["bb_period"], params["bb_std"])
h4_slope    = ma_slope(h4, params["ma_period"])
h4_sma      = sma(h4, params["ma_period"])

h4_green = (h4["close"] > h4["open"])
h4_red   = (h4["close"] < h4["open"])

big = h4_range > params["breakout_atr_mult"] * h4_atr
two_big_green = big & big.shift(1) & h4_green & h4_green.shift(1)
two_big_red   = big & big.shift(1) & h4_red   & h4_red.shift(1)

bb_exp = h4_bw > h4_bw.shift(1)

h4_long  = two_big_green & bb_exp & (h4_slope > 0)
h4_short = two_big_red   & bb_exp & (h4_slope < 0)
```

Three conditions for a long setup on a 4H bar: (i) last two 4H candles both green and each with range > `breakout_atr_mult × ATR`; (ii) BB width expanding; (iii) SMA slope > 0. Short is the symmetric case.

#### Signal logic — 1H entry state machine

`bb_breakout.py:169-264` runs the entry state machine bar by bar. Key transitions:

- **Expiry E1** — `bars_since > max_1h_bars` ⇒ invalidate (`bb_breakout.py:210-212`).
- **Expiry E2** — current 1H candle range > `pullback_atr_mult × ATR(1H)` ⇒ invalidate (`bb_breakout.py:215-217`).
- **Expiry E3** — price has overshot the SMA by more than `pullback_bps` basis points on the wrong side (`bb_breakout.py:220-232`).
- **Entry P1 + P2** — price within `pullback_bps` of SMA on the correct side AND a matching reversal candlestick pattern fires (`bb_breakout.py:236-264`).

#### Parameters

Verbatim `PARAM_SPACE` (`bb_breakout.py:97-108`):

```python
PARAM_SPACE = {
    "bb_period":         ("int",   10, 40),
    "bb_std":            ("float", 1.5, 3.0),
    "atr_period":        ("int",   5,  20),
    "breakout_atr_mult": ("float", 1.2, 3.0),
    "ma_period":         ("int",   10, 30),
    "pullback_bps":      ("int",   5,  30),
    "max_1h_bars":       ("int",   12, 48),
    "pullback_atr_mult": ("float", 0.5, 2.0),
    "swing_lookback":    ("int",   5,  20),
    "pattern_wick":      ("float", 1.5, 3.0),
}
```

Meanings (`bb_breakout.py:56-66`):

| Param | Meaning |
|---|---|
| `bb_period` | Bollinger Band period (4H) |
| `bb_std` | BB σ multiplier (4H) |
| `atr_period` | ATR period, shared by 4H and 1H |
| `breakout_atr_mult` | Min candle range / ATR for the 4H setup |
| `ma_period` | SMA period, shared by 4H and 1H |
| `pullback_bps` | Pullback zone width AND overshoot tolerance, in basis points around SMA |
| `max_1h_bars` | Max 1H bars to wait for entry after 4H setup fires |
| `pullback_atr_mult` | Max 1H candle range during pullback phase |
| `swing_lookback` | ZigZag lookback for stop placement |
| `pattern_wick` | Pin-bar wick-to-body ratio minimum |

#### Regime filter

No external regime filter. The two embedded filters are: (i) `ma_slope > 0` / `< 0` requirement at setup time and (ii) `bb_width` expansion requirement.

#### Position sizing

Fixed unit size (`bb_breakout.py:253-254` — `position[i] = float(setup_direction)`). No volatility scaling at the strategy level; sizing is the caller's responsibility.

#### Stop / target rule

Stop placed at the last swing low (for longs) or swing high (for shorts), looked back over `swing_lookback` 1H bars (`bb_breakout.py:255-261`). Per the class docstring (`bb_breakout.py:87`), the **target is 1:1** against the stop distance — but the published file does not contain the exit-target code; that lives in whatever harness consumes the `position` / `stop_loss` output.

### C.4 Cross-sectional momentum — `make_xs_strategy` (in progress)

**File:** `infrastructure/walkforward/xs_strategy.py:275-721`. Research notebooks: `topics/momentum/xs_momentum/xs_3_*_i_r2*.ipynb`. **Status: in progress, not productionised.**

This is a strategy *factory*, not a single strategy: it captures a price/volume panel and returns a `strategy_fn(df_slice, params)` compatible with both `walk_forward()` and `run_cpcv()`.

#### Signal logic

Two selectable formulations (`xs_strategy.py:499-519`):

```python
if signal_kind == 'residual_sharpe':
    daily_btc   = win['BTC'].pct_change()
    beta        = daily_coin.rolling(J).cov(daily_btc).divide(
        daily_btc.rolling(J).var(), axis=0
    )
    daily_resid = daily_coin.subtract(beta.multiply(daily_btc, axis=0))
    mom_signal  = (
        daily_resid.rolling(J).mean() / daily_resid.rolling(J).std()
    ).shift(1)
elif signal_kind == 'rolling_sharpe':
    mom_signal = (
        daily_coin.rolling(J).mean() / daily_coin.rolling(J).std()
    ).shift(1)
```

- `residual_sharpe` (the `xs_3*` family) — strip BTC beta via rolling regression, then take the J-bar Sharpe of residuals. Warmup `2J + 1`.
- `rolling_sharpe` (the `xs_1*` family) — Sharpe of raw returns over J bars. Warmup `J + 1`.

The `.shift(1)` on the final line is the explicit 1-bar forward shift (see Section G).

#### Optional refinement layers (factory kwargs)

1. **Dynamic universe** via `universe_filter.get_universe()` (`xs_strategy.py:574-588`) — eligible coins must clear `min_avg_volume`, `min_age_days`, and `top_n` filters at the formation bar.
2. **Lee-Swaminathan two-stage volume refinement** (`xs_strategy.py:593-610`) — Stage 1 selects a pool of `pool_multiplier × n_pick` by composite rank; Stage 2 refines by highest 7d/30d volume change.
3. **Inverse-volatility intra-leg weighting** (`xs_strategy.py:628-639`) — weights coins in each leg by `1 / rolling_vol`, normalised.
4. **BTC regime tilt** — two modes:
   - `regime_mode='adx'` (Layer 1) via `compute_btc_regime_tilt_panel` (`xs_strategy.py:75-114`) — direction from EMA bull/bear, intensity from ADX scaled by `adx_scale`.
   - `regime_mode='breadth'` (Layer 2) via `compute_btc_regime_breadth_tilt_panel` (`xs_strategy.py:117-186`) — direction from BTC vs MA, intensity from universe breadth (fraction of coins on the same side of their own MA).
5. **Dispersion-confidence gate** (Layer 3, `xs_strategy.py:189-240`) — scales gross exposure by a multiplier in `[confidence_low, confidence_high]` based on rolling-quantile cross-sectional dispersion.

#### Parameters

Required Optuna parameters (`xs_strategy.py:342-346`):

| Param | Meaning |
|---|---|
| `J` (int) | Formation/lookback window for the Sharpe signal |
| `K` (int) | Holding/rebalance period |
| `pct_long` (float) | Fraction of eligible universe in the long leg |
| `pct_short` (float) | Fraction of eligible universe in the short leg |

Asymmetric `K_long` / `K_short` are also supported (`xs_strategy.py:481-490`).

Optional Optuna parameters (`xs_strategy.py:348-352`, `:443-475`):

- `regime_ema_period`, `regime_adx_period`, `regime_adx_scale` — ADX-mode tilt.
- `breadth_ma_period` (or legacy `breadth_ema_period`) — breadth-mode tilt.
- `vol_short_window`, `vol_long_window` — Lee-Swaminathan windows.
- `iv_vol_window` — inverse-vol window.
- `dispersion_window`, `confidence_low`, `confidence_high` — dispersion gate.

#### Regime filter

The tilt panel (Section above) is **the** regime filter — it skews long/short leg weights via `long_wo = 0.5 + tilt_lag` / `short_wo = 0.5 - tilt_lag` (`xs_strategy.py:660-661`). Tilt is lag-1 shifted before being combined with returns (`xs_strategy.py:657-658`).

#### Position sizing

The output frame carries fixed `position=1`, `position_size=1.0` (`xs_strategy.py:701-703`). Exposure comes from the `strategy_returns` series itself, which is the leg-weighted dollar-neutral spread return scaled by the dispersion confidence (`xs_strategy.py:666-667`):

```python
strategy_returns_arr = conf_lag * (long_wo  * long_filled
                                  - short_wo * short_filled)
```

#### Stop / target rule

None — pure systematic L/S rebalance.

#### Output

Returns `(result_df, ['strategy_returns'])` with per-bar columns: `strategy_returns`, `turnover`, `long_ret`, `short_ret`, `universe_size`, `long_turnover`, `short_turnover`, `n_long_held`, `n_short_held`, `long_weight`, `short_weight`, `regime_label`, `confidence` (`xs_strategy.py:696-717`).

---

## D. Metric reference

Every metric the engines report, with its verbatim formula and a trustworthiness rating.

### D.1 `total_return` — **STRONG**

`performance_metrics.py:30-33`:

```python
def calculate_total_return(equity_curve):
    total_return = equity_curve.iloc[-1] - 1
    return total_return
```

Cumulative return over the evaluation window. The equity curve is built by `build_equity_curve` (`performance_metrics.py:312-327`):

```python
if return_type == "log":
    equity_curve = np.exp(returns.cumsum())
else:
    equity_curve = (1 + returns).cumprod()
```

`net_returns` (the input) includes fees, since costs are baked into the position-change cost path inside `engine.backtest`. Flat periods (zero returns) are kept in the cumprod — they just do not move the curve. **Trust: STRONG.**

### D.2 `sharpe_ratio` — **MODERATE**

`performance_metrics.py:36-50`:

```python
def calculate_sharpe_ratio(returns, periods_per_year):
    returns = returns.dropna()

    if len(returns) < 2:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0:
        return 0.0

    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return sharpe
```

- Annualisation factor inferred from bar spacing by `infer_frequency` (`performance_metrics.py:6-25`): ≤1h → 8760, ≤4h → 2190, ≤24h → 365, ≤168h → 52, else 12.
- **No risk-free rate subtracted.** This is mean-to-σ × √N, not the textbook Sharpe.
- **Flat bars are included** in mean and σ — there is no `[returns != 0]` filter at the engine level. For low-turnover strategies (e.g. `momentum_swing`, which is flat ~half the time) this depresses both numerator and denominator and tends to *understate* the Sharpe an "in-market only" calculation would give.
- `pandas.Series.std()` defaults to `ddof=1` (sample stdev).

**Trust: MODERATE.** Annualisation is correct; the RF=0 simplification and flat-bar inclusion are well-known properties — interpret comparatively, not absolutely.

### D.3 `max_drawdown` — **STRONG**

`performance_metrics.py:53-59`:

```python
def calculate_max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    return max_dd
```

Returns a negative number (e.g. `-0.25` = 25% drawdown). Operates on the realised equity curve, so fees are included. **Trust: STRONG.**

### D.4 `calmar_ratio` — **STRONG (annualised)** / **MODERATE in `_default_score` (not annualised)**

The canonical metric (`performance_metrics.py:226-234`):

```python
def calculate_calmar_ratio(total_return, max_drawdown, periods_per_year, n_periods):
    if max_drawdown == 0:
        return 0.0
    # annualise the total return
    n_years = n_periods / periods_per_year
    if n_years == 0:
        return 0.0
    annualised_return = (1 + total_return) ** (1 / n_years) - 1
    return annualised_return / abs(max_drawdown)
```

This is correctly annualised (annualised CAGR over abs max DD). `metrics['calmar_ratio']` and the `_calmar()` helper in `wf_engine.py:73-78` both use this. **Trust: STRONG.**

The scoring-side calmar inside `_default_score` (`wf_engine.py:31-32`) is `total_return / |max_drawdown|` — **not annualised**. It exists to feed a hand-tuned saturation cap (`CALMAR_MAX=60.0`) and **should not be treated as a Calmar ratio** when comparing across windows of different lengths. **Trust: MODERATE — okay as a normalised score component, misleading if quoted.**

### D.5 `profit_factor` — **STRONG (mind the `inf` sentinel)**

`performance_metrics.py:211-223`:

```python
def calculate_profit_factor(trades_df):
    if len(trades_df) == 0:
        return 0.0

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0

    profit_factor = gross_profit / gross_loss
    return profit_factor
```

- Empty trade list ⇒ `0.0`.
- All-winners (gross_loss = 0) ⇒ `np.inf`. Be aware this can poison downstream means / quantile summaries; clip if aggregating.
- Otherwise: ratio of gross wins to gross losses across `trades_df`.

Trades are identified by `identify_trades` (`performance_metrics.py:117-173`) — entries and exits paired by transitions in `position`. **Trust: STRONG.**

### D.6 `win_rate` — **STRONG**

`performance_metrics.py:176-185`:

```python
def calculate_win_rate(trades_df):
    if len(trades_df) == 0:
        return 0.0

    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    total_trades = len(trades_df)

    win_rate = winning_trades / total_trades
    return win_rate
```

Win = trade `pnl > 0`. **Trades with exactly zero PnL are counted as losses** by this definition (`pnl > 0` is strict). **Trust: STRONG.**

### D.7 `num_trades` — **STRONG**

`performance_metrics.py:188-190`. Simply `len(trades_df)`. **Trust: STRONG.**

### D.8 `avg_win_loss_ratio` — **STRONG**

`performance_metrics.py:193-208`:

```python
def calculate_avg_win_loss_ratio(trades_df):
    if len(trades_df) == 0:
        return 0.0

    winning_trades = trades_df[trades_df['pnl'] > 0]['pnl']
    losing_trades = trades_df[trades_df['pnl'] < 0]['pnl']

    if len(winning_trades) == 0 or len(losing_trades) == 0:
        return 0.0

    avg_win = winning_trades.mean()
    avg_loss = abs(losing_trades.mean())

    ratio = avg_win / avg_loss
    return ratio
```

Returns `0.0` whenever either side is empty (i.e. for all-winner or all-loser samples — distinct from `profit_factor`'s `inf` behaviour). **Trust: STRONG.**

### D.9 `yearly_returns`, `yearly_sharpe`, `yearly_max_drawdown` — **MODERATE**

`performance_metrics.py:342-383`. Per-year groupby on `returns.index.year`. Note (`performance_metrics.py:355-361`):

```python
start_value = year_equity.iloc[0]
end_value = year_equity.iloc[-1]

if start_value != 0:
    yearly_return = (end_value - start_value) / start_value
else:
    yearly_return = 0.0
```

Yearly return is `(end - start) / start`, **not** `(end/start) - 1` — those are mathematically identical for non-zero starts, so this is fine. **Yearly Sharpe annualises using the same `periods_per_year` as the full series** — correct for cross-year comparability. **Yearly max-drawdown is computed within each year only**, so a drawdown that spans a year boundary is split. **Trust: MODERATE** — interpretation matters at year boundaries.

### D.10 What is included vs excluded

| Aspect | Behaviour | Source |
|---|---|---|
| Fees | Included; engine charges `cost × \|position.diff()\|` at every position change (cost is per-leg, round-trip = 2 × cost) | `wf_engine.py:55-70` |
| Slippage | **Not modelled** at the engine level; baked into `cost` if you set it that way |  |
| Flat (zero-position) periods | **Included** in Sharpe / total_return / drawdown calculations | `performance_metrics.py:38-50` |
| Indicator-warmup NaN rows | Dropped per-strategy via `dropna(subset=indicator_cols)` only — not blanket | `wf_engine.py:352-356` |
| Equity curve | Realized-sizing model for single-asset (entries notional'd against closed-trade equity only); MTM compounding for precomputed-returns paths | `performance_metrics.py:237-309`, `cpcv_engine.py:59-95` |

---

## E. Portfolio module

There are **two** portfolio surfaces in this stack — `infrastructure/backtester/portfolio_metrics.py` (used by both the live combined-portfolio notebook and the per-strategy WF notebooks) and `infrastructure/walkforward/cpcv_portfolio.py` (used by per-strategy CPCV portfolio notebooks).

### E.1 `portfolio_metrics.py` — bar-level aggregation

**File:** `infrastructure/backtester/portfolio_metrics.py`.

#### Sleeve return convention

Momentum sleeves are scored per-bar by `mom_bar_returns` (`portfolio_metrics.py:74-89`):

```python
def mom_bar_returns(df: pd.DataFrame, cost: float) -> pd.Series:
    pos  = df['position'].shift(1).fillna(0)
    size = (df['position_size'].shift(1).fillna(0)
            if 'position_size' in df.columns
            else pd.Series(1.0, index=df.index))
    ret  = df['Close'].pct_change().fillna(0)
    to   = df['position'].diff().abs().fillna(0)
    return ret * pos * size - cost * to
```

Note the **1-bar shift on `position` and `position_size`** — the position held during bar `t` is the position decided at bar `t-1`. Stat-arb sleeves expose `net_returns` directly (cost already baked in) and are passed through.

#### Weighting (3 levels)

`build_sleeve_weights` (`portfolio_metrics.py:178-196`) composes weights as `top-level bucket × within-bucket × sleeve normalisation`:

- **Bucket level** — `strategy_weights = {'statarb': w_sa, 'momentum': w_mom}` (free).
- **Stat-arb within bucket** — inverse-volatility via `sa_inverse_vol_weights` (`portfolio_metrics.py:155-175`), with `method='in_market'` (vol on bars where `position != 0`) or `'full'`.
- **Momentum within bucket** — `build_momentum_weights` (`portfolio_metrics.py:117-146`) — two-level: `strat_weights[tag]` (e.g. `{'bb': 0.4, 'wf2': 0.6}`) × `coin_weights[tag][coin]` (defaulting to equal within each tag).

#### Realised-sizing portfolio equity curve

`build_realized_equity` (`portfolio_metrics.py:203-282`) is the canonical multi-sleeve aggregator. Per-bar loop:

1. Earn bar returns on positions open at the start of the bar.
2. Exits realise PnL into `realized_equity` and reset `cum_mult`.
3. Entries size against **post-exits realized equity** (`entry_notional = weights[k] × realized_equity`). Unrealised gains from concurrently open positions do **not** inflate future entry notionals.
4. Reported equity each bar = `realized + Σ(entry_notional × (cum_mult - 1))`.

This is the same model used by `performance_metrics.build_realized_equity_curve` for the single-asset path; both files implement the algorithm explicitly (no shared inner function).

#### Sweeps

- `sweep_top_level` (`portfolio_metrics.py:289-336`) — scan the statarb/momentum split in `step%` increments, feeding the combined bar-return series through `engine.backtest` so Sharpe / Calmar / DD use the same code path as the equity chart.
- `sweep_momentum_strategy` (`portfolio_metrics.py:339-402`) — scan strategy-level weights inside the momentum bucket (e.g. `bb` vs `wf2`) with fixed coin weights.

#### Print helpers

`print_weight_audit`, `print_bucket_comparison`, `print_weight_sweep`, `print_momentum_sweep`, `print_per_coin_stats` — all in `portfolio_metrics.py:409-535`. They render the audit tables used by `topics/momentum/results/portfolio_*.ipynb` and `topics/epsilon_portfolio.ipynb`.

#### Assumption set

- Mixed-frequency sleeves (hourly + daily) are aligned by `pd.concat(..., axis=1).fillna(0)` (`portfolio_metrics.py:312-313`, `:383-386`). A daily-only sleeve contributes zero return on the intra-day bars where the hourly sleeve trades.
- Inverse-vol weights are computed on the **full history** in the pickle by default (`portfolio_metrics.py:156`); pass `window=N` to use only the last N bars. This is a backwards-looking, in-sample weighting choice — not point-in-time.
- Strategy-weights / coin-weights are static across the OOS window. There is no dynamic re-weighting.

### E.2 `cpcv_portfolio.py` — path-level bootstrap

**File:** `infrastructure/walkforward/cpcv_portfolio.py`.

Companion to `cpcv_engine.py`. Combines per-asset CPCV results (each a pickle of the dict returned by `run_cpcv`) into a portfolio-level **distribution** over randomly sampled paths.

Typical flow (`cpcv_portfolio.py:9-21`):

```python
assets  = load_asset_cpcv({'BTC': 'btcusdt_cpcv.pkl',
                            'ETH': 'ethusdt_cpcv.pkl',
                            'SOL': 'solusdt_cpcv.pkl'})
weights = {'BTC': 0.40, 'ETH': 0.35, 'SOL': 0.25}
paths   = sample_portfolio_paths(assets, weights, n_samples=2000)
ci      = portfolio_confidence_intervals(paths, assets)
portfolio_summary(paths, ci, weights, asset_results=assets)
heatmaps = per_asset_split_heatmaps(paths, assets)
```

Per-path metrics are computed by `_compute_path_metrics` (`cpcv_portfolio.py:62-94`), which replicates the formulas in `performance_metrics.calculate_all_metrics` line-for-line (equity = cumprod of net returns, Sharpe = mean/σ × √N, Calmar = annualised return / |max DD|, max DD = min of equity-vs-running-max ratio). The deliberate redundancy keeps the CPCV portfolio engine independent of an asset-side engine import.

`_infer_periods_per_year` (`cpcv_portfolio.py:42-60`) mirrors `performance_metrics.infer_frequency` exactly, so annualisation is consistent.

---

## F. CPCV implementation — `cpcv_engine.py`

**Implemented.** Canonical entry point: `run_cpcv()` at `cpcv_engine.py:213-506`.

### F.1 Block (group) construction

`generate_cpcv_splits` (`cpcv_engine.py:102-206`). The dataset is partitioned into `N` contiguous groups of size `n_bars // N` (last group absorbs the remainder, `cpcv_engine.py:131-137`):

```python
base = n_bars // N
groups = []
for i in range(N):
    s = i * base
    e = s + base if i < N - 1 else n_bars
    groups.append((s, e))
```

**Defaults** (`cpcv_engine.py:218-219`): `N=8`, `k=2`. `N % k == 0` is enforced so that paths are complete (`cpcv_engine.py:127-129`).

### F.2 Splits

`itertools.combinations(range(N), k)` enumerates every `C(N, k)` choice of `k` test groups (`cpcv_engine.py:141`). For the default `N=8, k=2` setup this yields **28 splits** — each Optuna-optimised independently. The remaining `N - k` groups form the training set.

### F.3 Embargo / purge

Purge is applied only at train/test **boundaries** within each split (`cpcv_engine.py:148-162`):

```python
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
```

Default `purge_bars = 1` (`cpcv_engine.py:220`). Bars are dropped **from the training side only**; the test groups stay intact. Test-side warmup is handled separately by `burnin` (`cpcv_engine.py:223`, default 100), which prepends pre-group bars to the OOS slice and trims them after indicator computation (`cpcv_engine.py:347-366`).

### F.4 Number of paths

`generate_cpcv_splits` enumerates every distinct way to assemble a complete OOS curve by picking one test-group assignment from one split for each group (recursion at `cpcv_engine.py:185-204`). For `N=8, k=2` this produces **105 complete paths**, each consuming `N/k = 4` splits.

### F.5 Per-split scoring

The Optuna sampler and objective are imported directly from `wf_engine` so scoring is identical (`cpcv_engine.py:21-28`, `:310-321`):

```python
from wf_engine import (
    _default_score,
    _default_reject,
    _run_backtest,
    _calmar,
    _make_objective,
    _fmt,
)
# ...
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
```

`n_trials` default 400 (`cpcv_engine.py:221`). `n_jobs` is exposed for thread-level parallelism per split (default 1, `cpcv_engine.py:227`).

### F.6 Aggregation across paths

`run_cpcv` stitches each path's `k` test-group OOS slices into a single DataFrame (`cpcv_engine.py:405-432`):

```python
for pm in path_meta:
    pid         = pm['path_id']
    assignments = pm['split_assignments']

    slices = []
    for g_idx, s_id in sorted(assignments, key=lambda x: x[0]):
        oos_df = oos_dfs.get(s_id, {}).get(g_idx)
        if oos_df is not None and len(oos_df) > 0:
            slices.append(oos_df)
    # ...
    path_df = pd.concat(slices).sort_index()
    path_df = path_df[~path_df.index.duplicated(keep='first')]

    equity_curve        = _compute_equity_curve(path_df, cost)
    equity_curve.name   = f'path_{pid}'
    pm_metrics          = _run_backtest(path_df, cost)
```

Per-path Sharpe / Calmar / Max DD / Total Return come from `engine.backtest` on the stitched path — same code as single-asset WF.

### F.7 Confidence intervals

`cpcv_confidence_intervals` (`cpcv_engine.py:686-787`) returns both:

- **Naive t-interval** using `N-1` degrees of freedom (anticonservative — paths share splits).
- **Effective-N adjusted CI** using a weighted-overlap matrix where cell `(i,j) = |splits_i ∩ splits_j| / splits_per_path`. Effective N is `N² / Σ overlap`. Uses `N_eff - 1` degrees of freedom; conservative. For the default `N=8, k=2` config, `N_eff = C(8,2) / (8/2) = 28 / 4 = 7` (`cpcv_engine.py:739-748`).

### F.8 Parameter analysis

`cpcv_parameter_analysis` (`cpcv_engine.py:513-673`) returns:

- `distribution_stats` — mean / median / IQR / CV per param across splits.
- `param_performance_corr` — Pearson r between each param's value and per-split mean OOS Sharpe (`cpcv_engine.py:576-594`).
- `cross_param_corr` — correlation matrix among parameters across splits.
- `tercile_comparison` — separation between top-tercile and bottom-tercile splits in each param (`cpcv_engine.py:603-630`).
- `consensus_ranges` — `action ∈ {'fix at X', 'narrow to IQR', 'keep current range'}` based on CV thresholds (`cpcv_engine.py:649-655`).

`cpcv_print_param_suggestions` (`cpcv_engine.py:1199-1268`) renders ready-to-paste `PARAM_DEFS` and `FIXED_PARAMS` blocks.

### F.9 IS/OOS efficiency tracking

`run_cpcv` evaluates IS Sharpe on the training slice with `best_params` and compares it to mean OOS Sharpe per split (`cpcv_engine.py:326-339`, `:448-481`). The efficiency ratio = `oos_sharpe / is_sharpe`. Summary thresholds (`cpcv_engine.py:1187-1191`):

- `std_efficiency < 0.25` → "Consistent" (else "Volatile").
- `mean_efficiency > 0.5` → "Generalising" (else "Overfitting").

---

## G. Conventions & invariants

### G.1 1-bar forward shift on signals & regime filters

The portfolio aggregator shifts position by 1 bar before earning returns (`portfolio_metrics.py:83-87`):

```python
pos  = df['position'].shift(1).fillna(0)
size = (df['position_size'].shift(1).fillna(0)
        if 'position_size' in df.columns
        else pd.Series(1.0, index=df.index))
ret  = df['Close'].pct_change().fillna(0)
```

The xs-momentum signal applies the shift explicitly inside the strategy (`xs_strategy.py:508-514`):

```python
mom_signal  = (
    daily_resid.rolling(J).mean() / daily_resid.rolling(J).std()
).shift(1)
```

The xs-momentum regime tilt and dispersion confidence are also lag-1 shifted before combining with returns (`xs_strategy.py:651-658`):

```python
def _lag1(a, fill):
    out     = np.empty_like(a)
    out[0]  = fill
    out[1:] = a[:-1]
    return out

tilt_lag = _lag1(tilt_arr, 0.0)
conf_lag = _lag1(conf_arr, 1.0)
```

**Rule:** any value derived from bar `t`'s observable data is used to size or direct trading **on bar `t+1`**.

### G.2 Never blanket `dropna`

The walk-forward and CPCV engines drop NaN rows only on the explicit `indicator_cols` list returned by the strategy. `wf_engine.py:352-356`:

```python
test_strat_full, indicator_cols  = strategy_fn(fold['test_burnin'].copy(), best_params)

if test_strat_full is not None:
    oos_df = test_strat_full.copy()
    existing_cols = [c for c in indicator_cols if c in oos_df.columns]
    if existing_cols:
       before_drop = len(oos_df)
       oos_df.dropna(subset=existing_cols, inplace=True)
```

Same pattern in CPCV at `cpcv_engine.py:360-363`. **Rule:** strategies return both a dataframe AND the list of indicator columns whose warmup NaNs are acceptable to drop. Engines never call `df.dropna()` without `subset=`.

### G.3 √(periods_per_year) annualisation

Single point of truth — `performance_metrics.py:49`:

```python
sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
```

`periods_per_year` is inferred from the median bar spacing (`performance_metrics.py:6-25`) and is **also** used unchanged by `calculate_calmar_ratio` (`performance_metrics.py:226-234`) and by the CPCV portfolio's `_infer_periods_per_year` (`cpcv_portfolio.py:42-60`). All three call sites use the same lookup table — there is no second annualisation convention anywhere in the stack.

### G.4 Calmar uses annualised return

`performance_metrics.py:226-234` (verbatim):

```python
def calculate_calmar_ratio(total_return, max_drawdown, periods_per_year, n_periods):
    if max_drawdown == 0:
        return 0.0
    # annualise the total return
    n_years = n_periods / periods_per_year
    if n_years == 0:
        return 0.0
    annualised_return = (1 + total_return) ** (1 / n_years) - 1
    return annualised_return / abs(max_drawdown)
```

The `_calmar()` helper in `wf_engine.py:73-78` reads `metrics['calmar_ratio']` directly from this function. **Caveat (re-stated from Section D.4):** `_default_score` recomputes a non-annualised calmar for *scoring saturation only* — do not confuse it with the reported metric.

### G.5 Optuna trial budget — ≥10–20 trials per free parameter

**Not enforced in code.** Both `walk_forward` (`wf_engine.py:234`) and `run_cpcv` (`cpcv_engine.py:221`) default to `n_trials = 400`, and `live_trading/dashboards/momentum/config.py:5-11` mirrors that 400. For strategies with 10 free params, this is ~40 trials per param — comfortably in the recommended band. The rule is a **discipline guideline**: when you add or unfreeze parameters, scale `n_trials` proportionally before re-running.

### G.6 Iteration cycles capped at 3–4

**Not enforced in code.** The CPCV → parameter analysis → narrow/fix → re-CPCV loop, formalised by `cpcv_parameter_analysis` (`cpcv_engine.py:513-673`) and `cpcv_print_param_suggestions` (`cpcv_engine.py:1199-1268`), is meant to be run at most 3–4 times per strategy/asset before declaring convergence. Beyond that, you are overfitting to the CPCV path distribution itself. This is a research-process invariant — track manually.

### G.7 Cost convention — `cost` is per-leg, round-trip = 2 × cost

Reinforced in three places:

- `wf_engine.py:55-59`:

  ```python
  def _run_backtest(strategy_df, cost):
      # `cost` is a per-leg fraction.  engine.backtest() applies it at every
      # position change (separately at entry and at exit), so the total
      # round-trip cost per trade = 2 × cost.
      # e.g. cost=0.001 → 0.1% per leg → 0.2% round-trip.
  ```
- `wf_engine.py:255-258` (in `walk_forward` docstring), `wf_engine.py:493-495` (in `plateau_analysis`), `wf_engine.py:691-693` (in `perturbation_test`), `wf_engine.py:787-790` (in `cost_stress_test`).
- `live_trading/dashboards/momentum/config.py:18-20`:

  ```python
  # Round-trip trading cost as a fraction of notional position size.
  # Entry cost + exit cost = size_usd * TRADING_COST_PCT * 2
  ```

The default everywhere is `0.001` (0.1% per leg, 0.2% round-trip).

### G.8 `position_size` shifted alongside `position`

Wherever `position` is shifted, `position_size` is shifted by the same amount. See `portfolio_metrics.py:83-87` (quoted in G.1) and `performance_metrics.py:262-263`:

```python
pos  = position.fillna(0).values.astype(float)
size = position_size.fillna(0).values.astype(float)
```

(Both `position` and `position_size` are already shift(1)-lagged by the caller per the function docstring at `performance_metrics.py:248-251`.) This guarantees that the per-bar return is `(prev position) × (prev size) × (this bar's price change)` — there is no scenario where size leaks from the future.

---

## H. Overfitting validation — `infrastructure/validation/overfitting_audit.py`

Every Optuna-searched strategy must clear this harness **before going live**. The CPCV/WF
stack measures how the *selected* config performs OOS, but says nothing about how much of
that performance is the selection itself — with `n_trials = 400` per fold/split (G.5), the
best-of-search Sharpe is upward-biased even on pure noise. This module quantifies that bias.

> **Plain-English explainer:** for what each statistic *means* and how to read a verdict
> (intuition, toy examples, the DSR×PBO interpretation matrix), see
> [[OVERFITTING_VALIDATION]]. This section is the terse API/gate reference.

### H.1 The three statistics

| Statistic | Question | Function |
|---|---|---|
| **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) | Does the observed OOS Sharpe exceed the expected *maximum* Sharpe of an N-trial search on zero-edge data? | `deflated_sharpe_ratio()` |
| **PBO via CSCV** (Bailey, Borwein, López de Prado, Zhu 2014) | Across all balanced IS/OOS partitions of time, how often does the IS-best config rank in the bottom half OOS? | `pbo_cscv()` |
| **White's Reality Check** (White 2000; studentised per Hansen 2005) | Bootstrap p-value that the best config's mean return beats zero *after* accounting for the search | `whites_reality_check()` |

All Sharpe conventions match `performance_metrics.py` (mean/std·√ppy, ddof=1, no risk-free,
flat bars included — D.2). CIs use the stationary bootstrap (Politis–Romano); the module is
numpy/pandas-only (no scipy in the root venv).

### H.2 Pre-registered gate

```
PASS requires ALL of:
  deflated Sharpe > 0          (DSR prob > 0.5; ≥ 0.95 is the "strong" bar)
  PBO < 0.5                    (< 0.2 is the "good" bar)
  post-haircut lower-CI Sharpe > 0.25   (materiality bar, set before results)
Reality-Check p < 0.05 is reported as supporting evidence, not gated.
Anything else ⇒ FLAG-FOR-REVIEW (no live deployment).
```

Statistical survival ≠ economic materiality — verdicts follow `brain/CODEX.md`
§ Realism calibration and ship an assumption ledger.

### H.3 Trial-data requirement and `collect_trials`

PBO and the Reality Check need the **T × N candidate-return matrix**, and the DSR haircut
needs the cross-trial Sharpe variance. Historically both engines discarded the Optuna study
after extracting `best_params` — per-trial data did not survive into `*_cpcv.pkl`. Two fixes:

- `run_cpcv(..., collect_trials=True)` stores per-trial `{number, value, params, user_attrs}`
  in each `split_results[i]['trials']`; `walk_forward(..., collect_trials=True)` returns the
  same per fold under `result['fold_trials']`. Cheap (scalars only) — **set it on every new
  search**. The CPCV template now does.
- `build_trial_returns_matrix()` re-runs saved trial configs through the standard backtest
  (deterministic) to rebuild the return matrix; `replay_search_trial_matrix()` replays a
  same-design TPE study for legacy artifacts that lack trial records (slower; the verdict
  notes which mode ran).

### H.4 Notebook / template integration

`cpcv_template.ipynb` ends with an **"Overfitting check (required before live)"** cell pair
calling `audit_cpcv_run(results, df, strategy_fn, score_fn, reject_fn, label=SYMBOL)` and
asserting the gate. The printed `verdict.to_markdown()` block must be pasted into the
strategy's findings note. The new-idea workflow hook lives in `brain/START_RESEARCH_IDEA.md`
step 6. Unit tests (synthetic overfit vs genuine-edge fixtures):
`./.venv/bin/python -m pytest infrastructure/validation/tests/ -q`.

### H.5 First application — live momentum book

The audit of the live 6-asset book (vs the 2.24 headline Sharpe) lives in
[[momentum_overfitting_audit_findings]], with the runner at
`topics/momentum/strategies/momentum_cpcv/run_overfitting_audit.py`. Its verdict
(`FLAG-FOR-REVIEW`: real edge, but PBO ≈ 0.8 on a flat plateau) is the canonical worked
example for [[OVERFITTING_VALIDATION]]'s interpretation matrix.

---

## Contributing notes

- Momentum, BB-breakout, XS-momentum, regime-filter, ML-prediction, and stat-arb research notes belong under the relevant `topics/<strategy>/` subtree or in `docs/` when they synthesize multiple strategy families.
- Use names that identify the strategy and validation layer, for example `momentum_<asset>_cpcv_findings.md`, `bb_<timeframe>_wf_findings.md`, `xs_<variant>_research.md`, or `regime_<filter>_findings.md`.
- Add `> Hub: [[STRATEGY_REFERENCE]]` near the top of every new crypto research note, and add a wikilink to that note in the most relevant section of this file. For duplicate basenames such as `README.md`, link by path with an alias.
- Immediately after the hub backlink, add `## Plain-English Summary` or `## Summary` with 2-5 bullets or one tight paragraph explaining what the note is about, why it was written, what data/experiment it covers, and the one-line takeaway/status before any results table or verdict.
