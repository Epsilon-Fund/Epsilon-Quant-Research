# Statistical Arbitrage — Pairs Trading Portfolio
> Hub: [[STRATEGY_REFERENCE]]


A daily-bar pairs trading strategy built on cointegrated cryptocurrency pairs, validated through walk-forward optimisation and combined into a 4-pair inverse-volatility-weighted portfolio.

---

## Table of Contents

1. [Core Idea](#1-core-idea)
2. [Signal Construction](#2-signal-construction)
3. [Strategy Parameters](#3-strategy-parameters)
4. [Position Logic](#4-position-logic)
5. [Structural Design](#5-structural-design)
6. [Walk-Forward Optimisation](#6-walk-forward-optimisation)
7. [Robustness Analysis](#7-robustness-analysis)
8. [Portfolio Pairs](#8-portfolio-pairs)
9. [Portfolio Construction](#9-portfolio-construction)
10. [OOS Results](#10-oos-results)
11. [File Structure](#11-file-structure)
12. [How to Run](#12-how-to-run)
13. [Infrastructure Dependencies](#13-infrastructure-dependencies)

---

## 1. Core Idea

Statistical arbitrage exploits temporary mispricings between two assets that share a long-run equilibrium relationship — known as **cointegration**. Unlike correlation (which measures whether two assets move in the same direction), cointegration means that even if both assets drift over time, a specific linear combination of their prices remains stationary and mean-reverting.

In practice: if Asset Y has historically traded at roughly `β` times Asset X, and today that ratio has blown out significantly, the strategy bets that the ratio will revert. It enters long on the underperformer and short on the outperformer simultaneously, making the trade market-neutral — the profit comes from the convergence, not from the overall market direction.

All signals are generated on **daily UTC close** bars. The strategy is designed to be largely uncorrelated to directional crypto moves since both legs are held at the same time.

---

## 2. Signal Construction

### Step 1 — Log Prices

Raw prices are converted to log prices before any calculations:

$$\log Y_t = \log(\text{Close}_{Y,t}), \qquad \log X_t = \log(\text{Close}_{X,t})$$

Log prices are used because they make returns additive, stabilise variance, and ensure the hedge ratio has a clean economic interpretation (percentage relationship rather than absolute price).

### Step 2 — Rolling OLS Hedge Ratio (β)

The hedge ratio `β` defines how much of Asset X is needed to hedge one unit of Asset Y. It is estimated using **Ordinary Least Squares (OLS)** regression of log_Y on log_X over a rolling `lookback` window:

$$\log Y_t = \alpha + \beta_t \cdot \log X_t + \varepsilon_t$$

OLS minimises the sum of squared residuals `ε` over the lookback window, producing the `β_t` that best explains log_Y using log_X at each point in time.

**Why rolling?** A static β estimated over the full history assumes the relationship between the two assets never changes — unrealistic for crypto over multi-year periods. A rolling β adapts to structural shifts (e.g. changes in market cap ratio, liquidity, or sector correlations) without completely forgetting the long-run relationship. The `lookback` window length controls how fast the hedge ratio adapts: shorter = more reactive, longer = more stable.

**Why OLS and not Kalman filter?** OLS is simpler, interpretable, and avoids overfitting from state-space model misspecification. The rolling window effectively acts as an equally-weighted Kalman filter with a hard memory cutoff.

### Step 3 — Spread

The spread is the residual of the hedged position:

$$s_t = \log Y_t - \beta_t \cdot \log X_t$$

When the spread is near its historical mean, the pair is in equilibrium. When it deviates significantly, the strategy expects mean-reversion.

Note: `β_t` is shifted forward by 1 bar before computing the spread — this ensures no look-ahead bias. The hedge ratio used today was estimated from data available at yesterday's close.

### Step 4 — Z-Score Normalisation

The spread is normalised into a z-score over a separate rolling `z_lookback` window:

$$\mu_t = \text{mean}(s_{t-n:t}), \qquad \sigma_t = \text{std}(s_{t-n:t}), \qquad n = z\_lookback$$

$$z_t = \frac{s_t - \mu_t}{\sigma_t}$$

The z-score expresses how many standard deviations the spread is from its recent mean. This is what the entry and exit thresholds are compared against. Using a separate `z_lookback` (shorter than `lookback`) allows the normalisation to adapt to recent volatility without destabilising the hedge ratio.

### Step 5 — Spread Return

The per-bar return of the strategy (before position sizing and costs) is:

$$r^{\text{spread}}_t = r^Y_t - \beta_t \cdot r^X_t$$

where `ret_Y` and `ret_X` are the log returns of each asset. This is the raw P&L of holding one unit long and `β` units short (or vice versa).

---

## 3. Strategy Parameters

There are 6 tunable parameters. Each controls a distinct aspect of the strategy and has a specific role in the signal construction and risk management.

### `lookback` (int)
**What it does:** Sets the rolling window (in bars) for the OLS regression that estimates the hedge ratio β.

**Effect:** Longer lookback = more stable hedge ratio that changes slowly; shorter lookback = hedge ratio adapts faster to recent price action. If too short, the hedge ratio becomes noisy and the spread is ill-defined. If too long, the strategy misses genuine regime changes in the pair's relationship.

**Typical range across pairs:** 75 – 160 bars.

---

### `z_lookback` (int)
**What it does:** Sets the rolling window (in bars) for computing the spread's mean and standard deviation used to normalise the z-score.

**Effect:** Shorter z_lookback = more reactive to recent spread behaviour, producing more frequent (but potentially noisier) signals. Longer z_lookback = more conservative normalisation based on a longer history. Decoupled from `lookback` intentionally so the two windows can be independently tuned.

**Typical range across pairs:** 50 – 90 bars.

---

### `entry_z` (float)
**What it does:** The z-score threshold at which the strategy enters a trade. The strategy goes long the spread when `z < −entry_z` and short the spread when `z > +entry_z`.

**Effect:** Higher entry threshold = fewer but higher-conviction trades (spread must deviate more before entry). Lower threshold = more frequent entries with lower signal quality. This is one of the most impactful parameters for trade frequency and win rate.

**Typical range across pairs:** 1.5 – 2.2.

---

### `exit_z` (float)
**What it does:** The z-score level at which the strategy exits a profitable trade. The position is closed when `|z| < exit_z`, indicating the spread has mean-reverted sufficiently.

**Effect:** Lower exit threshold = exits closer to the mean (larger completed reversion, fewer trades left open too long). Higher threshold = exits early (smaller profits per trade but faster capital recycling). Needs to be lower than `entry_z` to ensure trades have a valid exit target.

**Typical range across pairs:** 0.3 – 1.5.

---

### `stop_z` (float)
**What it does:** A hard stop-loss. The position is closed immediately if `|z|` exceeds this threshold — meaning the spread has moved even further away from the mean instead of reverting.

**Effect:** Protects against cointegration breakdown. If the pair's relationship permanently breaks down (e.g. one asset delists, fundamental divergence), the spread may trend instead of mean-revert. The stop-loss limits drawdown in these cases. This parameter was found to be highly stable across folds for all pairs (CV < 0.10) and is fixed for most pairs.

**Typical range across pairs:** 4.0 – 4.7.

---

### `max_holding` (int)
**What it does:** A time-based stop. The position is closed after this many bars regardless of the z-score, to avoid holding stale positions that never resolve.

**Effect:** Prevents capital being tied up in trades where mean-reversion is very slow or indefinitely delayed. Especially useful in low-liquidity regimes where spreads can drift for extended periods.

**Typical range across pairs:** 3 – 40 bars.

---

## 4. Position Logic

| Condition | Action |
|-----------|--------|
| `z_t > +entry_z` | Short spread: short Y, long X×β |
| `z_t < −entry_z` | Long spread: long Y, short X×β |
| `\|z_t\| < exit_z` | Exit (go flat) |
| `\|z_t\| > stop_z` | Hard stop-loss exit |
| Bars held ≥ `max_holding` | Time-based exit |

Position flip: if a long signal fires while the strategy is already short (or vice versa), the position reverses immediately rather than waiting for an explicit exit signal.

Positions are entered on the next bar's open (1-bar execution lag is enforced via `position.shift(1)` in the engine) to prevent look-ahead bias.

---

## 5. Structural Design

The strategy is built in layers, each with a distinct responsibility:

```
┌─────────────────────────────────────────────────────────┐
│                     DATA LAYER                          │
│  Raw OHLCV from Binance → daily close prices            │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│                   SIGNAL LAYER  (per notebook)          │
│  Rolling OLS → β_t → spread_t → z_t → position_t       │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│              WALK-FORWARD LAYER  (wf_engine.py)         │
│  Per-fold: Optuna optimises 6 params on TRAIN window    │
│  Best params applied frozen on TEST window (OOS)        │
│  Folds stitched into a continuous OOS equity curve      │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│               BACKTEST LAYER  (engine.py)               │
│  Applies position sizing, 1-bar lag, transaction costs  │
│  Computes net_returns per bar                           │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│              METRICS LAYER  (performance_metrics.py)    │
│  Sharpe, Calmar, Max Drawdown, Win Rate, Profit Factor  │
│  Trade identification, yearly breakdown                 │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│              PORTFOLIO LAYER  (portfolio.py)            │
│  4 pairs combined via inverse-vol weights               │
│  Correlation analysis, cost sensitivity sweep           │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Separate `lookback` and `z_lookback`:** Most implementations use a single window for both hedge ratio estimation and z-score normalisation. Decoupling them gives the optimiser more flexibility — a stable hedge ratio with a shorter normalisation window tends to produce cleaner signals.

**Rolling β with 1-bar lag:** The hedge ratio is always estimated from data available at the previous close. This eliminates look-ahead bias in the hedge ratio, which is a common subtle error in pairs trading implementations.

**Fixed vs Free parameters:** Parameters that are stable across market regimes (low CV across folds) are locked in `FIXED_PARAMS`. This reduces the degrees of freedom the optimiser searches over, which both speeds up optimisation and reduces overfitting risk. Parameters that genuinely vary across regimes (e.g. `entry_z` in trending vs mean-reverting markets) are kept free so the strategy can adapt.

**Burn-in period:** Each OOS fold prepends `burnin_bars` of training data to the test slice before trimming. This ensures all rolling indicators (spread mean, spread std, β) are fully warmed up at the start of each OOS window, preventing distorted signals at fold boundaries.

---

## 6. Walk-Forward Optimisation

### Fold Structure

```
Full history:
│← BURNIN →│←────────── TRAIN ──────────→│←── TEST ──→│  fold 1
            │←────────── TRAIN ──────────→│←── TEST ──→│  fold 2
                         │←────────── TRAIN ──────────→│←── TEST ──→│  fold 3
                                          ...
```

The window slides forward by `test_bars` at each step (non-overlapping OOS windows). The OOS results from all folds are stitched together into a single continuous equity curve — this is what's evaluated in `portfolio.py`.

### Optimiser

- **Framework:** Optuna with TPE (Tree-structured Parzen Estimator) sampler
- **Trials per fold:** 800
- **Direction:** maximise composite score
- **Seed:** `seed_base + fold_index` for reproducibility across folds

### Objective Function (Composite Score)

$$\text{score} = 0.50 \cdot \text{clip}\!\left(\frac{\text{Sharpe}}{2.5},\, 0, 1\right) + 0.30 \cdot \text{clip}\!\left(\frac{\text{Calmar}}{60},\, 0, 1\right) + 0.20 \cdot \text{clip}\!\left(\frac{\text{Return}}{15},\, 0, 1\right)$$

Sharpe is weighted most heavily as the primary quality measure. Calmar captures risk-adjusted return relative to drawdown. Return provides a small push towards profitable parameter sets when Sharpe is indeterminate (e.g. very few trades).

### Rejection Filters (per fold, per trial)

Trials that fail these checks are immediately assigned score = −999 and discarded:

| Filter | Threshold |
|--------|-----------|
| Minimum trades | ≥ 7 per fold |
| Win rate | ≥ 35% |
| Max drawdown | > −80% |
| Profit factor | > 0.8 |

Some notebooks apply custom tighter filters (e.g. FIL/SNX uses min_trades=8, win_rate≥50%, pf>0.6).

### Walk-Forward Efficiency (WFE)

$$\text{WFE} = \frac{\overline{\text{Sharpe}}_{\text{OOS}}}{\overline{\text{Sharpe}}_{\text{IS}}}$$

WFE measures how much of the in-sample edge transfers to out-of-sample. A WFE above 0.70 is considered good — it means the strategy is not overfitting significantly to the training data. A WFE below 0.50 is a warning sign.

---

## 7. Robustness Analysis

Beyond the walk-forward, each notebook runs three additional robustness checks using tools in `wf_engine.py`:

### Plateau Analysis
Sweeps each free parameter across its full range independently (holding all others fixed at consensus values) and measures what fraction of the range achieves at least 80% of the peak score. A wide flat plateau means the strategy is insensitive to the exact value — a desirable property. A narrow spike means the parameter is fragile and may not generalise.

**Verdict thresholds:**
- Plateau% ≥ 60% → Robust
- Plateau% 30–60% → Moderate
- Plateau% < 30% → FRAGILE

### Perturbation Test
Randomly perturbs all free parameters simultaneously by ±5%, ±10%, and ±20% of their ranges (50 samples per offset level). Measures mean score degradation. This tests whether the optimum is a broad hill or a narrow spike in the joint parameter space.

### Parameter Stability Table
Reports the median, standard deviation, and coefficient of variation (CV = std / median) of each parameter's best value across folds. Parameters with CV < 0.15 are considered stable candidates to fix in future runs.

---

## 8. Portfolio Pairs

### Walk-Forward Configuration

| Pair | Train Bars | Test Bars | Burn-in | Trials |
|------|-----------|-----------|---------|--------|
| FIL/SNX | 500 | 126 | 200 | 800 |
| LINK/TRX | 500 | 126 | 200 | 800 |
| LTC/APT | 500 | 126 | 200 | 800 |
| ATOM/ARB | 500 | 126 | 200 | 800 |

### Results Summary

| Pair | Symbols | OOS Window | Folds | Profitable | WFE | Sharpe | Return | Max DD |
|------|---------|------------|-------|------------|-----|--------|--------|--------|
| FIL/SNX | FILUSDT / SNXUSDT | Jul 2023 – Dec 2025 | 7 | 7/7 | 0.88 | 1.57 | 369% | −26.1% |
| LINK/TRX | LINKUSDT / TRXUSDT | Jul 2023 – Dec 2025 | 7 | 7/7 | 0.85 | 1.23 | 760% | −20.6% |
| LTC/APT | LTCUSDT / APTUSDT | Mar 2024 – Mar 2026 | 6 | 5/6 | 0.61 | 1.15 | 83% | −26.7% |
| ATOM/ARB | ATOMUSDT / ARBUSDT | Aug 2024 – Dec 2025 | 4 | 4/4 | 0.82 | 1.30 | 68% | −20.8% |

### Parameter Decisions per Pair

**FIL/SNX**
- Fixed: `entry_z=1.8136`, `stop_z=4.647`
- Free: `lookback` (75–105), `z_lookback` (50–90), `exit_z` (0.5–1.1), `max_holding` (3–14)
- Rationale: entry and stop are highly stable across all 7 folds; exit and timing parameters vary by regime

**LINK/TRX**
- Fixed: `stop_z=4.0385`, `max_holding=6`
- Free: `lookback`, `z_lookback`, `entry_z`, `exit_z`
- Rationale: stop and holding period are stable; entry/exit thresholds adapt to LINK/TRX's varying volatility regimes

**LTC/APT**
- Fixed: `stop_z=4.2325`
- Free: `lookback`, `z_lookback`, `entry_z`, `exit_z`, `max_holding`
- Rationale: only stop is fixed — entry and exit are kept free to allow the strategy to adapt across the pair's short history (Mar 2024 onwards)

**ATOM/ARB**
- Fixed: `z_lookback=82`, `stop_z=4.4448`, `entry_z=1.7528`
- Free: `lookback` (80–120), `exit_z` (0.5–1.5), `max_holding` (20–40)
- Rationale: three parameters stabilised quickly across 4 folds; exit timing kept free

---

## 9. Portfolio Construction

### Inverse Volatility Weighting

Pairs are weighted inversely to their volatility so that each pair contributes roughly equal risk to the portfolio. Volatility is measured using **in-market returns only** — days where the pair has an open position:

$$\sigma_k = \text{std}\!\left(r^{\text{net}}_k \;\middle|\; \text{position}_k \neq 0\right)$$

$$w_k = \frac{1/\sigma_k}{\displaystyle\sum_j 1/\sigma_j}$$

**Why in-market only?** On flat days, net_returns = 0 by definition. Including these zeros deflates the standard deviation estimate for low-frequency pairs (like LINK/TRX at 14% market time) relative to high-frequency pairs. Using only active trading days gives a true measure of the risk taken when capital is deployed.

### Current Weights (derived from OOS pkl data)

| Pair | In-Market Vol | In-Market Frequency | Weight |
|------|--------------|---------------------|--------|
| ATOM/ARB | 2.48% | 36% | 33.6% |
| FIL/SNX | 4.19% | 24% | 19.9% |
| LINK/TRX | 3.52% | 14% | 23.7% |
| LTC/APT | 3.64% | 12% | 22.9% |

Weights are computed dynamically at runtime from the pkl files — adding or updating a pair's pkl automatically recalibrates the whole portfolio without manual input.

### Transaction Costs

All OOS results include **10bps per leg** (0.001) applied at every position change. At 96 total trades across 4 pairs over 2.5 years, this is a very low-turnover strategy.

Cost sensitivity (portfolio level):

| Cost per leg | Return | Sharpe | Max DD | Calmar |
|-------------|--------|--------|--------|--------|
| 5bps | 188.0% | 2.07 | −10.1% | 4.83 |
| **10bps (baseline)** | **181.8%** | **2.03** | **−10.3%** | **4.58** |
| 20bps | 169.8% | 1.95 | −10.9% | 4.13 |
| 30bps | 158.3% | 1.87 | −11.4% | 3.73 |
| 50bps | 136.8% | 1.72 | −12.6% | 3.03 |

The strategy remains viable at 50bps — 5x the assumed cost — because of low turnover.

---

## 10. OOS Results

### Portfolio Metrics (inverse vol weighted, in-market vol method)

| Metric | Value |
|--------|-------|
| Total Return | 248% |
| Sharpe Ratio | 1.85 |
| Max Drawdown | −8.3% |
| Calmar Ratio | 7.16 |
| OOS Start | 2023-07-27 |
| OOS End | 2026-03-27 |
| Avg Capital Utilisation | ~22% |

### Yearly Breakdown

| Year | Return | Sharpe | Max DD |
|------|--------|--------|--------|
| 2023 | +8.7% | 1.54 | −5.0% |
| 2024 | +103.9% | 1.94 | −8.3% |
| 2025 | +58.6% | 2.70 | −7.7% |

### Diversification (common window: Aug 2024 – Dec 2025, 487 bars)

| Pair | In-Market Days | Frequency |
|------|---------------|-----------|
| ATOM/ARB | 172 / 487 | 35.3% |
| FIL/SNX | 128 / 487 | 26.3% |
| LINK/TRX | 59 / 487 | 12.1% |
| LTC/APT | 65 / 487 | 13.3% |

All 4 pairs in market on the same day: **0 times (0.0%)**
Any 2+ pairs in market on the same day: **94 times (19.3%)**

### Return Correlations (common window)

| Pair vs Pair | Correlation |
|-------------|-------------|
| ATOM/ARB vs FIL/SNX | 0.089 |
| ATOM/ARB vs LINK/TRX | 0.012 |
| ATOM/ARB vs LTC/APT | −0.012 |
| FIL/SNX vs LINK/TRX | 0.121 |
| FIL/SNX vs LTC/APT | 0.108 |
| LINK/TRX vs LTC/APT | 0.189 |

All correlations below 0.19. The pairs trade independently, providing genuine diversification rather than the same bet expressed four times.

---

## 11. File Structure

```
testing/
├── Fil&SNX.ipynb          # FIL/SNX walk-forward notebook
├── LINK&TRX.ipynb         # LINK/TRX walk-forward notebook
├── LTC&APT.ipynb          # LTC/APT walk-forward notebook
├── ATOM&ARB.ipynb         # ATOM/ARB walk-forward notebook
├── fil_snx_oos.pkl        # OOS data exported from Fil&SNX.ipynb
├── link_trx_oos.pkl       # OOS data exported from LINK&TRX.ipynb
├── ltc_apt_oos.pkl        # OOS data exported from LTC&APT.ipynb
├── atom_arb_oos.pkl       # OOS data exported from ATOM&ARB.ipynb
├── portfolio.py           # Portfolio aggregation, metrics, charts
├── Pairs_Screening.py     # Cointegration screener for new pair candidates
└── README.md              # This file
```

### Notebook Structure (each pair)

## Notebook Map

- [[topics/statistical-arbitrage/strategies/testing/ETH&BNB.ipynb|ETH / BNB]]
- [[topics/statistical-arbitrage/strategies/testing/AVAX&FIL.ipynb|AVAX / FIL]]
- [[topics/statistical-arbitrage/strategies/testing/ATOM&ARB.ipynb|ATOM / ARB]]
- [[topics/statistical-arbitrage/strategies/testing/Fil&SNX.ipynb|FIL / SNX]]
- [[topics/statistical-arbitrage/strategies/testing/LINK&TRX.ipynb|LINK / TRX]]
- [[topics/statistical-arbitrage/strategies/testing/LTC&APT.ipynb|LTC / APT]]
- [[topics/statistical-arbitrage/strategies/testing/testing playground copy 6.ipynb|testing playground copy 6]]

Each notebook follows the same cell structure:

| Cell | Content |
|------|---------|
| 1 | Imports and path setup |
| 2 | Data download (Binance OHLCV) |
| 3 | `PARAM_DEFS` — parameter search space |
| 4 | `FIXED_PARAMS` — locked parameters |
| 5 | `strategy_fn` — signal construction (rolling OLS, z-score, position logic) |
| 6 | Custom `score_fn` and `reject_fn` |
| 7 | Walk-forward execution (`wf_engine.walk_forward`) |
| 8 | Fold results table |
| 9 | Parameter stability table |
| 10 | Plateau analysis and charts |
| 11 | Perturbation test |
| 12 | OOS equity curve chart |
| 13+ | Ad-hoc analysis cells |
| Last | Save cell — exports OOS data to `<pair>_oos.pkl` |

### Pkl Schema

Each `*_oos.pkl` is a pandas DataFrame with DatetimeIndex (daily bars):

| Column | Type | Description |
|--------|------|-------------|
| `Close_Y` | float | Daily close price of asset Y |
| `Close_X` | float | Daily close price of asset X |
| `beta` | float | Rolling OLS hedge ratio β_t |
| `alpha` | float | Rolling OLS intercept α_t |
| `spread` | float | log_Y − β×log_X |
| `spread_mean` | float | Rolling mean of spread over z_lookback |
| `spread_std` | float | Rolling std of spread over z_lookback |
| `z` | float | Z-score of spread |
| `spread_return` | float | Raw per-bar spread return (pre-cost) |
| `position` | int | Signal: 1=long spread, −1=short spread, 0=flat |
| `position_size` | float | Fractional sizing (1.0 = full allocation) |
| `stop_loss` | bool/int | 1 if stop-loss was triggered on this bar |
| `strategy_returns` | float | Raw return × position (pre-cost) |
| `net_returns` | float | strategy_returns − 10bps × turnover |

---

## 12. How to Run

### Prerequisites

```bash
pip install pandas numpy optuna matplotlib
```

### Re-run a pair notebook

Open the notebook in Jupyter and run all cells. The final save cell writes `<pair>_oos.pkl` to this directory with costs applied at `COST_SAVE = 0.001`.

### Run the portfolio

```bash
python portfolio.py
```

Outputs:
- Equity curve chart (per-pair lines + portfolio line + drawdown panel)
- Portfolio metrics (return, Sharpe, drawdown, Calmar)
- Yearly breakdown (return, Sharpe, max DD per year)
- Per-pair summary (metrics for each pair individually)
- Vol method comparison table (full vs in-market weights)
- Correlation & overlap analysis
- Cost sensitivity sweep

### Config (top of `portfolio.py`)

```python
WEIGHT_SCHEME = 'inverse_vol'   # 'equal' | 'inverse_vol'
VOL_METHOD    = 'in_market'     # 'full'  | 'in_market'
VOL_WINDOW    = None            # None = full OOS history | int = last N bars
COST_SWEEP    = [0.0005, 0.001, 0.002, 0.003, 0.005]  # cost sensitivity levels
```

---

## 13. Infrastructure Dependencies

| Module | Path | Purpose |
|--------|------|---------|
| `engine.py` | `infrastructure/backtester/` | Applies position shift, sizing, costs; calls metrics |
| `performance_metrics.py` | `infrastructure/backtester/` | Sharpe, drawdown, trade identification, yearly stats |
| `wf_engine.py` | `infrastructure/walkforward/` | Walk-forward loop, Optuna integration, plateau/perturbation tools |
| `wf_visualizer.py` | `infrastructure/walkforward/` | Per-fold result charts and HTML output |
