# Statistical Arbitrage — Pairs Trading Portfolio

A daily-bar pairs trading strategy built on cointegrated crypto pairs, validated through walk-forward optimisation and combined into a 4-pair inverse-volatility-weighted portfolio.

---

## Strategy Overview

The strategy exploits mean-reversion of synthetic spreads between cointegrated cryptocurrency pairs. When two assets are cointegrated, their log-price spread reverts to a long-run mean. The strategy enters when the spread is statistically dislocated (high z-score) and exits when it reverts.

All signals are generated on **daily bars** (UTC close). The strategy operates on both legs simultaneously — long one asset, short the other — making it market-neutral in theory and largely uncorrelated to directional crypto moves.

---

## Signal Construction

### 1. Spread
For a pair (Y, X), the log-price spread is:

```
spread_t = log(Y_t) - β_t × log(X_t)
```

where `β_t` is a **rolling OLS hedge ratio** estimated over a `lookback` window. Using a rolling β allows the hedge ratio to adapt to structural changes in the relationship between the two assets over time.

### 2. Z-Score
The spread is normalised to a z-score over a separate `z_lookback` window:

```
z_t = (spread_t - mean(spread, z_lookback)) / std(spread, z_lookback)
```

### 3. Position Logic

| Condition | Position |
|-----------|----------|
| z > +entry_z | Short spread (short Y, long X×β) |
| z < −entry_z | Long spread (long Y, short X×β) |
| \|z\| < exit_z | Exit (flat) |
| \|z\| > stop_z | Stop-loss exit |
| Holding time > max_holding | Time-based exit |

Positions are held until one of the exit conditions fires. The `stop_z` parameter acts as a hard stop for spread blow-outs where cointegration breaks down.

---

## Walk-Forward Optimisation

Each pair notebook runs a **rolling walk-forward** to validate that parameters generalise out-of-sample. This prevents overfitting to a single historical period.

### Structure

```
[ BURNIN | ──── TRAIN ──── | TEST ] → fold 1
         [ ──── TRAIN ──── | TEST ] → fold 2
                  ...
```

- **Burn-in**: initial period discarded to allow indicators to warm up
- **Train**: Optuna optimises parameters (800 trials per fold) on this window
- **Test**: parameters are frozen and applied forward on unseen data

### Optimiser
- **Framework**: Optuna (TPE sampler)
- **Objective**: maximise OOS Sharpe ratio subject to constraints:
  - Minimum trades per fold
  - Win rate ≥ 50%
  - Max drawdown < −50%
  - Profit factor > 0.6

### Parameter Strategy

Each pair uses a mix of **fixed** and **free** parameters:

- **Fixed** (`FIXED_PARAMS`): parameters with low coefficient of variation (CV < 0.15) across folds that do not benefit from regime adaptation — locked to their optimal value to reduce degrees of freedom
- **Free**: parameters allowed to vary per fold so the strategy can adapt to different market regimes (e.g. tighter entry thresholds in trending markets)

---

## Portfolio Pairs

| Pair | Symbols | OOS Period | Folds | Profitable | WFE | OOS Sharpe | OOS Return |
|------|---------|------------|-------|------------|-----|------------|------------|
| FIL/SNX | FILUSDT / SNXUSDT | Jul 2023 – Dec 2025 | 7 | 7/7 | 0.88 | 1.57 | 369% |
| LINK/TRX | LINKUSDT / TRXUSDT | Jul 2023 – Dec 2025 | 7 | 7/7 | 0.85 | 1.23 | 760% |
| LTC/APT | LTCUSDT / APTUSDT | Mar 2024 – Mar 2026 | 6 | 5/6 | 0.61 | 1.15 | 83% |
| ATOM/ARB | ATOMUSDT / ARBUSDT | Aug 2024 – Dec 2025 | 4 | 4/4 | 0.82 | 1.30 | 68% |

**WFE** (Walk-Forward Efficiency) = OOS Sharpe / IS Sharpe. A ratio above 0.70 indicates robust generalisation.

### Fixed Parameters per Pair

| Pair | Fixed |
|------|-------|
| FIL/SNX | entry_z=1.81, stop_z=4.65 |
| LINK/TRX | stop_z=4.04, max_holding=6 |
| LTC/APT | stop_z=4.23 |
| ATOM/ARB | z_lookback=82, stop_z=4.44, entry_z=1.75 |

---

## Portfolio Construction

### Weighting
Pairs are weighted by **inverse volatility** using in-market returns only (days where the pair has an open position):

```
vol(k)    = std( net_returns[k]  where  position[k] ≠ 0 )
weight(k) = (1 / vol(k)) / Σ (1 / vol(j))
```

Using in-market returns avoids artificially deflating volatility estimates for pairs with low market frequency (flat days contribute zero return but dilute the standard deviation).

### Current Weights (OOS data)

| Pair | In-Market Vol | Weight |
|------|--------------|--------|
| ATOM/ARB | 2.48% | 33.6% |
| FIL/SNX | 4.19% | 19.9% |
| LINK/TRX | 3.52% | 23.7% |
| LTC/APT | 3.64% | 22.9% |

### Transaction Costs
All OOS results include **10bps per leg** (0.1%) applied at every position change. Cost sensitivity analysis shows the strategy remains viable up to 50bps per leg (Sharpe 1.72 at 50bps vs 2.03 at 10bps) due to low turnover (~96 trades across 4 pairs over 2.5 years).

---

## Portfolio OOS Results

| Metric | Value |
|--------|-------|
| Total Return | 248% |
| Sharpe Ratio | 1.85 |
| Max Drawdown | −8.3% |
| Calmar Ratio | 7.16 |
| OOS Period | Jul 2023 – Mar 2026 |

### Yearly Breakdown

| Year | Return | Sharpe | Max DD |
|------|--------|--------|--------|
| 2023 | +8.7% | 1.54 | −5.0% |
| 2024 | +103.9% | 1.94 | −8.3% |
| 2025 | +58.6% | 2.70 | −7.7% |

### Diversification (common window Aug 2024 – Dec 2025)

All pair return correlations are below 0.19. No single day had all 4 pairs in market simultaneously, and only 19.3% of days had any 2 pairs active at the same time. Average capital utilisation is ~22% — the strategy only deploys when spreads are genuinely dislocated.

| Pair vs Pair | Correlation |
|-------------|-------------|
| ATOM/ARB vs FIL/SNX | 0.089 |
| ATOM/ARB vs LINK/TRX | 0.012 |
| ATOM/ARB vs LTC/APT | −0.012 |
| FIL/SNX vs LINK/TRX | 0.121 |
| FIL/SNX vs LTC/APT | 0.108 |
| LINK/TRX vs LTC/APT | 0.189 |

---

## File Structure

```
testing/
├── Fil&SNX.ipynb          # FIL/SNX walk-forward notebook
├── LINK&TRX.ipynb         # LINK/TRX walk-forward notebook
├── LTC&APT.ipynb          # LTC/APT walk-forward notebook
├── ATOM&ARB.ipynb         # ATOM/ARB walk-forward notebook
├── fil_snx_oos.pkl        # OOS data saved from Fil&SNX.ipynb
├── link_trx_oos.pkl       # OOS data saved from LINK&TRX.ipynb
├── ltc_apt_oos.pkl        # OOS data saved from LTC&APT.ipynb
├── atom_arb_oos.pkl       # OOS data saved from ATOM&ARB.ipynb
├── portfolio.py           # Portfolio aggregation, metrics, charts
└── Pairs_Screening.py     # Cointegration screener for new pair candidates
```

### Pkl Schema
Each `*_oos.pkl` file is a pandas DataFrame with:

| Column | Description |
|--------|-------------|
| `position` | Signal: 1 = long spread, −1 = short spread, 0 = flat |
| `strategy_returns` | Raw per-bar return of the spread (pre-cost) |
| `net_returns` | strategy_returns after 10bps transaction cost |

---

## How to Run

### 1. Re-run a pair notebook
Open the notebook and run all cells. The final save cell writes `<pair>_oos.pkl` to this directory.

### 2. Run the portfolio
```bash
python portfolio.py
```

Outputs: equity curve chart, portfolio metrics, yearly breakdown, per-pair summary, correlation analysis, and cost sensitivity sweep.

### Config (top of portfolio.py)
```python
WEIGHT_SCHEME = 'inverse_vol'   # 'equal' | 'inverse_vol'
VOL_METHOD    = 'in_market'     # 'full' | 'in_market'
VOL_WINDOW    = None            # None = full history | int = last N bars
```

---

## Infrastructure Dependencies

| Module | Path | Purpose |
|--------|------|---------|
| `engine.py` | `infrastructure/backtester/` | Backtest runner, cost application |
| `performance_metrics.py` | `infrastructure/backtester/` | Sharpe, drawdown, trade stats |
| `wf_engine.py` | `infrastructure/walkforward/` | Walk-forward loop, Optuna integration |
| `wf_visualizer.py` | `infrastructure/walkforward/` | Fold-level result charts |
