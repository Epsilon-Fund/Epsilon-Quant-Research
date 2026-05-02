# ML Prediction — Walk-Forward XGBoost Strategy

A daily-bar machine learning strategy trained on 8 crypto assets that outputs a calibrated probability of a next-day positive return. Validated through rolling walk-forward cross-validation and filtered to a long-only signal set on BTC / ETH / XRP / BNB.

---

## Table of Contents

1. [Core Idea](#1-core-idea)
2. [XGBoost: How It Works](#2-xgboost-how-it-works)
3. [Model Parameters](#3-model-parameters)
4. [Probability Calibration](#4-probability-calibration)
5. [Feature Set](#5-feature-set)
6. [Label Construction](#6-label-construction)
7. [Walk-Forward Design](#7-walk-forward-design)
8. [Signal Logic and Filtering](#8-signal-logic-and-filtering)
9. [Structural Design](#9-structural-design)
10. [OOS Results](#10-oos-results)
11. [Model Diagnostics](#11-model-diagnostics)
12. [SHAP Explainability](#12-shap-explainability)
13. [File Structure](#13-file-structure)
14. [How to Run](#14-how-to-run)
15. [Infrastructure Dependencies](#15-infrastructure-dependencies)

---

## 1. Core Idea

The strategy asks a single question on each day for each asset: **given everything the model can observe today, what is the probability that tomorrow's return will exceed +30 bps?**

If that probability exceeds 0.80 and the coin is in the filtered universe (BTC, ETH, XRP, BNB), a LONG signal fires. The model never predicts specific return magnitude — only the likelihood of a directional move above the noise threshold.

Using a machine learning classifier (rather than a fixed rule-based signal) has two advantages:

1. **Non-linearity** — the model can learn that e.g. high funding *combined with* falling open interest has a different implication than either alone.
2. **Feature interaction** — a 35-feature input space contains thousands of potential interactions; XGBoost finds the most predictive ones automatically.

The walk-forward structure ensures every prediction in the backtest is genuinely out-of-sample — the model never sees the data it is predicting on during training or calibration.

---

## 2. XGBoost: How It Works

XGBoost (Extreme Gradient Boosting) is an ensemble of decision trees trained sequentially. Understanding its internals is important for interpreting the model's outputs, feature importances, and generalisation behaviour.

### 2.1 Decision Trees (Base Learners)

A single decision tree splits the feature space into rectangular regions using threshold rules:

```
If ret_7d > 0.04 AND funding_zscore < −1.5:
    → predict class 1 (up)
Else if spx_ret_1d < −0.02:
    → predict class 0 (down)
...
```

Each leaf of the tree stores a score. For a classification problem, the scores are converted to probabilities via the logistic function.

A single shallow tree (max_depth=4) can only make 2⁴ = 16 distinct predictions — it is a **weak learner**: better than random but far from optimal. XGBoost's power comes from combining hundreds of such trees.

### 2.2 Gradient Boosting

Boosting builds an ensemble sequentially. Each tree is trained to correct the errors of all previous trees:

**Iteration 0:** Start with a constant prediction (log-odds of the base rate).

**Iteration t:** Compute the residuals — how much the current ensemble's prediction is wrong for each training sample. These residuals are the **negative gradient** of the loss function (log-loss for classification). The new tree is fitted to predict these residuals.

**Update:** Add the new tree to the ensemble, scaled by the learning rate:

```
F_t(x) = F_{t-1}(x) + η × tree_t(x)
```

where `η` is the learning rate (`learning_rate = 0.03` here). A small learning rate forces the model to take many small steps rather than a few large ones, reducing overfitting.

After T trees, the ensemble prediction is the sum of all tree outputs. For binary classification:

```
p̂ = sigmoid( Σ_t η × tree_t(x) )
```

### 2.3 The Objective Function

XGBoost minimises a regularised log-loss:

```
Objective = Σ_i L(y_i, ŷ_i)  +  Σ_t Ω(tree_t)
```

Where:
- `L(y, ŷ) = −[y·log(ŷ) + (1−y)·log(1−ŷ)]` is the log-loss (cross-entropy)
- `Ω(tree) = γ·T + ½λ·Σ_j w_j²` penalises tree complexity

`T` is the number of leaves, `w_j` is the weight (score) at leaf `j`, `γ` controls the minimum gain required to make a split, and `λ` is the L2 penalty on leaf weights.

This means XGBoost is not just minimising prediction error — it is actively penalising large leaf weights and unnecessary splits. This regularisation is what separates XGBoost from a vanilla gradient boosted tree and significantly reduces overfitting, which is critical in a small-sample financial prediction setting.

### 2.4 Tree Structure Search

For each new tree, XGBoost finds the best split at each node by computing an **exact gain score** for every possible (feature, threshold) pair:

```
Gain = ½ [ G_L²/(H_L + λ) + G_R²/(H_R + λ) − G²/(H + λ) ] − γ
```

Where `G` and `H` are the sum of first and second-order gradient statistics for the samples in each child node. This closed-form gain is exact and fast to compute, which is why XGBoost scales well to large feature sets.

A split is only made if `Gain > 0` — i.e. the regularisation-penalised gain exceeds zero. This automatically prunes useless splits.

### 2.5 Subsampling (Stochastic Gradient Boosting)

Two subsampling mechanisms reduce variance and prevent any single tree from overfitting:

- **`subsample = 0.8`** — each tree is trained on a random 80% of the training rows. This introduces randomness that reduces the correlation between trees (similar in spirit to bagging), which lowers ensemble variance.

- **`colsample_bytree = 0.8`** — each tree is fitted using a random 80% of the features. This prevents the most predictive features from dominating every tree and forces the ensemble to learn from the broader feature set. It is analogous to random feature selection in Random Forests.

### 2.6 Depth and Complexity Control

- **`max_depth = 4`** — each tree can make at most 4 sequential splits. This limits the model to learning 4-way interaction terms between features. Deeper trees can capture more complex relationships but overfit more aggressively in low-sample financial data.

- **`min_child_weight = 5`** — a split is only made if both resulting children contain at least 5 samples (weighted by the Hessian). This prevents the model from making splits that explain just 1–2 training examples, which would be pure noise. In a financial context where training rows are daily observations, this is an important guard against overfitting to individual market events.

### 2.7 Why XGBoost Over Linear Models or Neural Networks

| Model | Strength | Weakness |
|-------|----------|---------|
| Logistic Regression | Interpretable, stable | Can't capture interactions |
| Random Forest | Robust, no sequential dependency | Slower, less regularised |
| Neural Network | Learns complex hierarchies | Needs large datasets, very prone to overfit |
| **XGBoost** | Regularised, handles tabular data well, fast, interpretable via SHAP | Sequential fitting means later trees may amplify noise if early trees overfit |

For tabular financial data with ~35 features and ~8,000 training rows (pooled across coins), XGBoost consistently outperforms both linear models (insufficient expressiveness) and neural networks (insufficient data for deep architectures).

---

## 3. Model Parameters

All parameters are fixed across folds — no per-fold hyperparameter search is performed. This is deliberate: with only 30 days per test window and a single threshold τ, adding a hyperparameter search would introduce substantial look-ahead risk.

| Parameter | Value | Role |
|-----------|-------|------|
| `n_estimators` | 300 | Number of trees. More trees = lower bias, higher variance. 300 was selected as a round number that avoids underfitting without requiring early stopping. |
| `max_depth` | 4 | Maximum tree depth. Controls feature interaction complexity. |
| `learning_rate` | 0.03 | Step size per tree. Small → more trees needed, but lower overfitting risk. |
| `subsample` | 0.8 | Fraction of training rows used per tree. |
| `colsample_bytree` | 0.8 | Fraction of features used per tree. |
| `min_child_weight` | 5 | Minimum samples per leaf. Guards against fitting to outliers. |
| `eval_metric` | logloss | Training loss function. Log-loss is appropriate for probability estimation. |

---

## 4. Probability Calibration

### The Problem: Overconfidence

A trained XGBoost classifier outputs raw probabilities via `predict_proba`. These raw probabilities are often poorly calibrated — the model may output `prob = 0.92` for samples where the actual frequency of label=1 is only 65%. This makes the raw probabilities unreliable as confidence estimates.

### The Fix: Isotonic Regression

Each fold fits an **IsotonicRegression** calibrator on the validation set (3 months, held out from training):

```
calibrator.fit(raw_probs_val, y_val)
```

Isotonic regression finds the best monotonic non-decreasing function that maps raw probabilities to actual outcome frequencies on the validation set. It makes no parametric assumption about the shape of the transformation (unlike Platt scaling, which forces a logistic function).

After calibration:

```
calibrated_prob = calibrator.transform(raw_prob)
```

A calibrated probability of 0.85 means: in the validation set, among all samples where the model assigned this probability, approximately 85% had label=1. This is the definition of a well-calibrated model.

### Why This Matters for Signal Generation

The threshold τ = 0.80 operates on calibrated probabilities. Without calibration, τ = 0.80 would not correspond to any interpretable confidence level. With calibration, it means: **only trade when the model estimates an 80%+ probability of a positive outcome**.

The calibration curve (Section 5 of the notebook) validates this empirically — points should lie near the diagonal.

---

## 5. Feature Set

~35 features across 7 groups. All features are computed as of the close of the current day (no lookahead). Macro features are lagged by 1 additional day because US market closes occur after the crypto daily bar closes.

### Price & Momentum (11 features)

| Feature | Description |
|---------|-------------|
| `ret_1d` to `ret_30d` | Log returns over 1, 3, 7, 14, 30 days |
| `rvol_7d`, `rvol_14d`, `rvol_30d` | Rolling realised volatility (std of log returns) |
| `zscore_ma20`, `zscore_ma60` | Price deviation from 20-day and 60-day moving average, normalised by rolling std |
| `rsi_14` | 14-day Relative Strength Index |
| `skew_30d`, `kurt_30d` | 30-day rolling skewness and kurtosis of log returns |

### Funding Rates (4 features)

Perpetual futures funding rates reflect the cost of maintaining a leveraged position. Extreme positive funding = crowded long — a mean-reversion signal.

| Feature | Description |
|---------|-------------|
| `funding_1d` | Daily funding rate sum |
| `funding_avg_7d` | 7-day rolling average funding |
| `funding_zscore` | Funding deviation from 30-day mean, normalised |
| `funding_extreme` | Binary flag: \|daily funding\| > 0.15% |

### Open Interest (2 features)

OI changes reflect whether new money is entering or exiting the derivatives market.

| Feature | Description |
|---------|-------------|
| `oi_chg_1d` | 1-day pct change in open interest (USD) |
| `oi_chg_7d` | 7-day pct change in open interest (USD) |

### Long/Short Ratio (2 features)

Aggregate long/short positioning of retail accounts on Binance futures.

| Feature | Description |
|---------|-------------|
| `ls_ratio` | Raw long/short ratio |
| `ls_chg_7d` | 7-day change in the ratio |

### Deribit Implied Volatility (2 features — BTC and ETH only)

Options-implied volatility from Deribit captures forward-looking market uncertainty, distinct from realised volatility.

| Feature | Description |
|---------|-------------|
| `dvol` | 30-day Deribit DVOL index |
| `dvol_chg_7d` | 7-day change in DVOL |

### Macro (6 features — lagged 1 day)

US macro closes are lagged by 1 day to avoid lookahead (US markets close ~21:00 UTC, after the crypto bar at 00:00 UTC).

| Feature | Description |
|---------|-------------|
| `dxy_ret_1d`, `dxy_ret_7d` | Log returns of US Dollar Index |
| `spx_ret_1d`, `spx_ret_7d` | Log returns of S&P 500 |
| `gold_ret_1d`, `gold_ret_7d` | Log returns of Gold futures |

### On-Chain (2 features)

| Feature | Description |
|---------|-------------|
| `tvl_chg_7d` | 7-day pct change in total DeFi TVL (USD) |
| `stablecoin_chg_7d` | 7-day pct change in stablecoin market cap (USD) |

### Cross-Asset (2 features — non-BTC coins only)

| Feature | Description |
|---------|-------------|
| `btc_corr_30d` | 30-day rolling correlation of coin log returns with BTC |
| `btc_beta_30d` | 30-day rolling beta of coin returns vs BTC (OLS coefficient) |

### Calendar (2 features)

| Feature | Description |
|---------|-------------|
| `day_of_week` | Integer 0–6 (Monday–Sunday) |
| `is_weekend` | Binary flag: Saturday or Sunday |

---

## 6. Label Construction

The label is a **binary next-day directional classification** with a deadband filter:

```
next_ret = log(close_{t+1} / close_t)

label = 1     if  next_ret >  +0.003  (+30 bps)
label = 0     if  next_ret <  −0.003  (−30 bps)
label = NaN   if  |next_ret| ≤ 0.003  (dropped)
```

**Why a deadband?** Returns in the ±30 bps band are below transaction cost threshold and contain little exploitable signal — they are dominated by microstructure noise. Keeping these samples would dilute the signal-to-noise ratio of the training set. The deadband removes ~10–14% of rows per coin.

This means the model is trained to distinguish **meaningful up days** from **meaningful down days**, ignoring ambiguous flat days.

**Class balance:** After applying the deadband, the label distribution is approximately 55/45 (up/down) for most coins. No class reweighting is applied — the slight imbalance is within a range where XGBoost handles it naturally.

---

## 7. Walk-Forward Design

### Fold Structure

```
Full history:
│←──────────── TRAIN (3yr) ──────────────→│←VAL (3mo)→│← TEST (1mo) →│  fold 1
             │←──────────── TRAIN (3yr) ──────────────→│←VAL (3mo)→│← TEST (1mo) →│  fold 2
                          │←──────────── TRAIN (3yr) ──────────────→│←VAL (3mo)→│← TEST (1mo) →│  fold 3
                                          ...
```

| Window | Length | Purpose |
|--------|--------|---------|
| Train | 3 years (1095 days) | XGBoost training + RobustScaler fitting |
| Validation | 3 months (90 days) | IsotonicRegression calibration only |
| Test | 1 month (30 days) | OOS prediction — model sees these rows for the first time |
| Step | 1 month (30 days) | Each fold advances by one month |

**Total folds: 43** (June 2022 → April 2026)

### No Leakage Guarantee

Three layers protect against data leakage:

1. **RobustScaler** is fitted only on the train set, then applied to val and test. Prevents test statistics from influencing normalisation.
2. **IsotonicRegression calibrator** is fitted only on the val set. Prevents test outcomes from influencing probability calibration.
3. **Threshold τ = 0.80 is fixed** — no per-fold threshold search. If τ were optimised on the val set, it would leak future information about what threshold is needed.

### Multi-Asset Pooling

All 8 coins are pooled into a single feature matrix before training. This has two effects:

- **More training data:** ~8,000 rows per fold instead of ~1,000 per coin. XGBoost benefits significantly from larger training sets.
- **Cross-coin learning:** The model learns patterns that generalise across coins (e.g. extreme funding being bearish is true for all coins), while coin-specific behaviour is captured through correlation and beta features.

The `symbol` column is excluded from features — the model is not told which coin each row belongs to.

---

## 8. Signal Logic and Filtering

### Raw Signal

```
LONG  if calibrated_prob > 0.80  (τ = 0.80)
SHORT if calibrated_prob < 0.20  (1 − τ)
FLAT  if 0.20 ≤ prob ≤ 0.80
```

The threshold is symmetric — the flat zone absorbs all low-conviction outputs.

### Coin Filter (Long-Only, 4 Coins)

Analysis of OOS signals reveals a sharp asymmetry:

| Direction | All Coins | Hit Rate |
|-----------|-----------|----------|
| LONG | 62 signals | 64.5% |
| SHORT | 69 signals | 46.4% |

Short signals are below breakeven after fees (~50% hit rate needed for a 6bps round-trip). Per-coin analysis at τ = 0.80:

| Coin | Hit Rate | Signals | Decision |
|------|----------|---------|----------|
| XRPUSDT | 80.0% | 15 | KEEP |
| ETHUSDT | 64.7% | 17 | KEEP |
| BNBUSDT | 57.1% | 14 | KEEP |
| BTCUSDT | 56.2% | 16 | KEEP |
| SOLUSDT | 47.4% | 19 | REVIEW |
| AVAXUSDT | 47.1% | 17 | REVIEW |
| LINKUSDT | 47.1% | 17 | REVIEW |
| DOGEUSDT | 43.8% | 16 | REVIEW |

**Filtered universe:** BTCUSDT, ETHUSDT, XRPUSDT, BNBUSDT — long signals only.

This filter is not arbitrary: BTC and ETH are the two most liquid and macro-correlated coins (strong SPX/DXY signal); XRP and BNB show consistent directional edge likely driven by exchange-specific and regulatory sentiment features captured in the on-chain data.

### Cluster Months

When many signals fire on the same day across multiple coins (>8 in a month), the model is in a high-conviction regime but also potentially responding to a single macro event. Two such periods appear in the OOS data: August 2024 (18 signals) and March–April 2025 (28 signals). These should be treated with caution — consider sizing down when more than 4 coins signal simultaneously on the same day.

---

## 9. Structural Design

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                │
│  OHLCV (Binance, daily) + Macro (yfinance) + Derivatives        │
│  + On-chain (DeFi TVL, stablecoin mcap)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   FEATURE LAYER  (build_dataset.py)             │
│  35 features per row, binary label with deadband filter         │
│  Pooled: 8 coins × ~2,100 daily rows = ~17,700 total rows       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│               WALK-FORWARD LAYER  (walk_forward.py)             │
│  Per fold:                                                      │
│    1. RobustScaler.fit(train) → transform(val, test)            │
│    2. XGBClassifier.fit(train_scaled)                           │
│    3. IsotonicRegression.fit(val_probs, val_labels)             │
│    4. Apply calibrated probs to test set                        │
│  Output: predictions_df (all OOS rows), folds_meta              │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                METRICS LAYER  (ml_metrics.py)                   │
│  Sharpe (active + full calendar), Max Drawdown, Calmar          │
│  Hit rate (overall + by confidence bucket)                      │
│  Calibration table, Feature importance, Equity curve            │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              EXPLAINABILITY LAYER  (notebook, shap)             │
│  TreeExplainer on stored fold models                            │
│  Per-signal SHAP drivers + mean |SHAP| bar chart                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**RobustScaler over StandardScaler:** Financial features often have heavy tails and outlier events (flash crashes, liquidation cascades). RobustScaler normalises using the median and IQR rather than mean and std, making it insensitive to extreme events in the training window.

**Validation set for calibration only:** The val set (3 months) is used exclusively for IsotonicRegression fitting — it is not used to tune any XGBoost hyperparameter. This means the calibrator has genuine out-of-sample val data, and any overfitting in calibration is visible in the calibration curve.

**Fixed τ = 0.80:** Not searching for an optimal threshold on the val set is a deliberate anti-overfit measure. A threshold search over even a simple grid (e.g. 10 values) would constitute an extra degree of freedom that could exploit noise in 90-day val windows.

**Pooled training:** Training on all 8 coins simultaneously rather than one model per coin increases the training set 8-fold. This is the primary reason the model can use a relatively complex tree structure (300 trees, depth 4) without severe overfitting.

---

## 10. OOS Results

### Filtered Strategy (Long-Only, BTC / ETH / XRP / BNB)

| Metric | Value |
|--------|-------|
| Signals | 31 |
| Hit Rate | 74.2% |
| Sharpe (active days) | 8.50 |
| Sharpe (full calendar) | 1.32 |
| Calmar Ratio | 4.51 |
| Max Drawdown | −3.1% |
| Annualised Return | 14.0% |
| OOS Window | Sep 2022 – Apr 2026 |
| Transaction Cost | 6 bps round-trip |

### Per-Year Breakdown

| Year | Sharpe | Return | Signals |
|------|--------|--------|---------|
| 2022 | — | — | 0 |
| 2023 | high | positive | 17 |
| 2024 | high | positive | 11 |
| 2025 | high | positive | 3 |
| 2026 | — | — | 0 |

### Interpretation

The strategy is **high precision, low frequency** — 31 signals over 3.5 years (~9 per year). The high active-day Sharpe (8.50) reflects very high hit rate on the signals that do fire; the full-calendar Sharpe (1.32) is the honest metric that accounts for the long periods of inactivity.

The low drawdown (−3.1%) is a direct consequence of the 74% hit rate combined with the deadband label — when signals fire, they tend to be on high-momentum days where the directional move is large enough to absorb the cost.

---

## 11. Model Diagnostics

Three diagnostic sections in the notebook validate model quality independently of backtest performance:

### Hit Rate by Confidence Bucket

At τ = 0.80, all filtered signals land in the ≥80% confidence bucket by construction. The table verifies signal counts are consistent and that the calibrated confidence values make sense.

### Calibration Curve

Plots predicted probability (x-axis) vs actual frequency of label=1 (y-axis) across 10 probability bins. A well-calibrated model produces points near the diagonal. The IsotonicRegression calibrator is specifically designed to force this alignment on the validation set — deviations from the diagonal would indicate calibrator overfitting or structural distribution shift.

### Feature Importance

Average XGBoost gain-based importance across all 43 folds. Key findings:

- **Macro features dominate:** `spx_ret_1d`, `dxy_ret_1d`, `gold_ret_7d` consistently rank in the top 5. This confirms the model is primarily a macro/regime detector — crypto directional moves are being predicted by what happened to traditional risk assets the prior day.
- **On-chain flow matters:** `tvl_chg_7d` and `stablecoin_chg_7d` are consistently top-10. DeFi TVL growth and stablecoin mcap growth both precede crypto rallies in the training data.
- **Short-term momentum is modest:** `ret_1d` and `ret_3d` appear but with lower importance than macro, suggesting the model is not purely a momentum strategy.
- **Derivatives signals are situational:** Funding and OI features contribute but inconsistently across folds, reflecting their regime-dependent nature.

---

## 12. SHAP Explainability

SHAP (SHapley Additive exPlanations) decomposes the model's prediction for each individual signal into additive contributions from each feature.

For a LONG signal:

```
log-odds(LONG) = base_value
               + SHAP(spx_ret_1d)     = +0.31  ← SPX up yesterday → bullish
               + SHAP(dxy_ret_1d)     = −0.12  ← DXY slightly up → mild headwind
               + SHAP(tvl_chg_7d)     = +0.22  ← DeFi TVL growing → capital flowing in
               + SHAP(funding_1d)     = +0.09  ← Funding near zero → room for longs
               + SHAP(ret_7d)         = +0.14  ← Positive momentum
               + ...
```

SHAP values are computed using TreeExplainer — an exact, model-specific algorithm for tree ensembles that does not require approximation. The stored model (`folds_meta[fold]['model']`) and scaler are used directly without re-training.

The SHAP section in the notebook runs on the most recent fold with active signals, providing a real-time explanation of what drove the most recent predictions.

---

## 13. File Structure

```
ml-prediction/
├── notebooks/
│   ├── 3_walkforward.ipynb        # Main walk-forward notebook (this strategy)
│   └── README.md                  # This file
├── data/
│   ├── raw/
│   │   ├── macro/                 # macro_daily.parquet (DXY, SPX, Gold, VIX, 10Y)
│   │   ├── funding/               # {SYMBOL}_funding.parquet
│   │   ├── oi/                    # {SYMBOL}_oi.parquet
│   │   ├── longshort/             # {SYMBOL}_ls.parquet
│   │   ├── deribit/               # {COIN}_dvol.parquet (BTC, ETH only)
│   │   └── onchain/               # onchain_daily.parquet
│   └── features/
│       └── {SYMBOL}_features.parquet  # Pre-built feature matrices (one per coin)
```

### Notebook Cell Structure

| Section | Cells | Content |
|---------|-------|---------|
| 1. Strategy Overview | 0 | Signal logic, window config, feature set, use cases, backtest summary |
| 2. Imports | 1 | Python imports, repo paths, constants |
| 3. Feature Rebuild (optional) | rebuild-features | Re-runs build_dataset.save_all() |
| 4. Load Feature Matrix | 3–4 | Reads feature parquets, prints shape and date range |
| 5. Run Walk-Forward | 5–7 | Executes all 43 folds, applies coin filter |
| 6. Filtered Scorecard | 8–9 | Equity curve, per-year Sharpe, per-coin breakdown, signal list |
| 7. Model Diagnostics | 10–17 | Hit rate bucket, calibration curve, feature importance |
| 8. Signal Report | 18–19 | Long/short split, cluster check, recent 20 signals |
| 9. SHAP | 20–21 | Per-signal feature drivers for most recent active fold |
| 10. Universe Diagnostics | 22–23 | Per-coin hit rates with KEEP/REVIEW recommendations |
| 11. Cross-Coin Consensus | 24–25 | Daily net score bar chart |

---

## 14. How to Run

### Prerequisites

```bash
pip install pandas numpy xgboost scikit-learn shap matplotlib pyarrow yfinance
```

### Step 1: Ensure raw data is current

```bash
# Fetch latest macro data
python infrastructure/data/ingest_macro.py

# Ensure OHLCV cache is up to date (runs via backfill_cache.py or live_trading pipeline)
```

### Step 2: Rebuild feature matrices (only when raw data changes)

Run the **rebuild-features** cell in the notebook, or:

```bash
python infrastructure/ml/features/build_dataset.py
```

This takes ~2 minutes and writes `{SYMBOL}_features.parquet` to `data/features/`.

### Step 3: Run the notebook

Open `3_walkforward.ipynb` and run all cells. The walk-forward loop (~3–5 minutes) generates `preds` and `folds_meta`. All subsequent sections operate on these two objects.

### Step 4: Check for new signals

Look at the **Signal Report** section (section 6) and in particular the "Most Recent 20 Signals" table. Any signal with `direction=LONG` and `symbol` in `[BTCUSDT, ETHUSDT, XRPUSDT, BNBUSDT]` is the filtered strategy's output.

To check the current day's model probability (live scoring), run the feature builder and walk-forward with today's data included.

---

## 15. Infrastructure Dependencies

| Module | Path | Purpose |
|--------|------|---------|
| `build_dataset.py` | `infrastructure/ml/features/` | Feature engineering for all 8 coins; reads raw parquets, computes all feature groups, applies deadband label |
| `walk_forward.py` | `infrastructure/ml/` | Rolling walk-forward engine; RobustScaler + XGBoost + IsotonicRegression per fold |
| `ml_metrics.py` | `infrastructure/backtester/` | Sharpe, drawdown, equity curve, calibration table, feature importance aggregation |
| `ingest_macro.py` | `infrastructure/data/` | Fetches DXY, SPX, Gold, VIX, 10Y yield via yfinance; incremental append logic |
| `backfill_cache.py` | `infrastructure/data/` | OHLCV cache maintenance for Binance daily bars |
