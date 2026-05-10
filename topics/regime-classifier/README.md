# BTC Regime Classifier

A two-stage machine learning pipeline that assigns a daily **market regime label** to BTC and predicts the next day's regime from observable features — enabling forward-looking regime awareness that can be used to gate or size positions in other trading strategies.

**Stage 1 — HMM Labelling:** A Gaussian Hidden Markov Model is fitted to 5+ years of BTC daily data to produce a statistically grounded set of regime labels. These are the ground truth.

**Stage 2 — XGBoost Prediction:** A walk-forward supervised classifier is trained to predict tomorrow's HMM regime from today's technical features. This is what gets used live — the HMM cannot be run in real time without lookahead bias.

**Stage 3 — Strategy Overlay:** The regime predictions are used as a filter or position-sizing signal in downstream trading strategies.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Stage 1 — HMM Regime Discovery](#2-stage-1--hmm-regime-discovery)
3. [Stage 2 — Feature Engineering](#3-stage-2--feature-engineering)
4. [Stage 2 — XGBoost Walk-Forward](#4-stage-2--xgboost-walk-forward)
5. [OOS Prediction Results](#5-oos-prediction-results)
6. [Regime Labelling Output](#6-regime-labelling-output)
7. [Forward Returns by Regime](#7-forward-returns-by-regime)
8. [Strategy Application — MoneyIn Long](#8-strategy-application--moneyin-long)
9. [Probability Thresholding](#9-probability-thresholding)
10. [Strategy Application — BB Breakout](#10-strategy-application--bb-breakout)
11. [When the Regime Filter Helps (and When It Doesn't)](#11-when-the-regime-filter-helps-and-when-it-doesnt)
12. [Keeping Predictions Up to Date](#12-keeping-predictions-up-to-date)
13. [File Structure](#13-file-structure)
14. [How to Run](#14-how-to-run)
15. [Output Schemas](#15-output-schemas)
16. [Infrastructure Dependencies](#16-infrastructure-dependencies)

---

## Strategy Testing Results — Quick Reference

Two notebooks test the regime filter on real strategies. Full results in sections 8–10.

| Notebook | Strategy | Verdict | Key result |
|----------|----------|---------|------------|
| `topics/momentum/results/moneyin_regime.ipynb` | MoneyIn Long (BTC trend, long-only) | **Filter helps** | Hybrid p>0.70: Return +423%, Sharpe 1.05, MaxDD −22% vs baseline 314% / 0.88 / −30% |
| `topics/momentum/results/portfolio_bb_regime.ipynb` | BB Breakout portfolio (6 coins) | **Filter hurts** | Baseline 754% Sharpe 2.24 → Bull+Chop 247% Sharpe 1.66 — filter removes the strategy's best vol-expansion trades |

**Recommended filter configuration for directional long strategies:** Hybrid probability threshold at 0.70 — allow positions when `p_regime_0 > 0.70` OR `pred_regime == 1` (Chop always allowed). Do not apply to volatility or mean-reversion strategies.

---

## 1. Motivation

Most trading strategies are implicitly regime-dependent: a trend-following strategy that works well in a sustained bull run will bleed in a choppy bear market, and a mean-reversion strategy designed for stable sideways markets will be destroyed in a vol explosion. Rather than fitting these regimes implicitly through parameter optimisation — which can mask regime risk — this project makes the regime an explicit, measurable variable.

The goal is not to build a standalone "trade the regime" system, but to answer a simpler question for every strategy in the fund: **should this strategy be trading today?**

**Why BTC specifically?** BTC is the dominant crypto asset. Its regime (bull, bear, risk-off) largely determines the direction and volatility environment for the entire alt market. A single BTC regime classifier is therefore applicable as a filter across most crypto strategies in the portfolio.

---

## 2. Stage 1 — HMM Regime Discovery

**Notebook:** `notebooks/1_hmm_regime_discovery.ipynb`

### What is a Hidden Markov Model?

A Hidden Markov Model assumes that at every point in time, the market is in one of K unobserved (hidden) states. Each state generates observable data according to a state-specific probability distribution. The model learns the state transition probabilities (how likely is the market to stay in, or transition out of, each regime) and the emission distributions (what does observed data look like in each regime) simultaneously from the data.

Formally, the HMM learns:
- **Transition matrix A:** where `A[i,j]` = probability of moving from regime i to regime j on any given day
- **Emission means μ and covariances Σ:** the multivariate Gaussian distribution of features in each regime
- **Initial probabilities π:** the prior probability of starting in each regime

### Model Configuration

| Setting | Value | Reason |
|---------|-------|--------|
| Model type | `GaussianHMM` (hmmlearn) | Continuous features → Gaussian emissions |
| Number of states | 4 | BIC-optimal (tested 2–6 states) |
| Covariance type | `full` | Each state has its own full covariance matrix |
| EM restarts | 5 | Avoids local optima; keeps best by log-likelihood |
| Random seed | 42 | Reproducibility |

**BIC model selection:** The number of states is not assumed — it is chosen by running the HMM for K=2,3,4,5,6 states and selecting the K that minimises the Bayesian Information Criterion. BIC penalises model complexity, preventing the model from simply splitting the data into ever-finer sub-regimes without predictive value. K=4 was optimal.

### Input Features (3)

The HMM is deliberately trained on a **minimal, orthogonal feature set** — the goal is to identify regimes, not overfit to noise. Three features that capture the three primary dimensions of a crypto market regime:

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `log_ret` | `log(Close_t / Close_{t-1})` | Direction — is the market trending up or down? |
| `rvol_5d` | Rolling 5-day std of log returns | Intensity — how volatile is the market? |
| `drawdown` | `(Close - max(Close, 20d)) / max(Close, 20d)` | Depth — how far are we from a recent peak? |

### Regime Labelling

After fitting, the HMM states are ordered by mean return descending so labels are consistent across refits and interpretable:

| Label | Name | Interpretation |
|-------|------|---------------|
| **0** | Bull | Positive returns, low-moderate vol, near highs |
| **1** | Recovery / Chop | Near-zero returns, moderate vol, below highs |
| **2** | Bear | Negative returns, elevated vol, sustained drawdown |
| **3** | Extreme Bear | Severe negative returns, high vol, deep drawdown |

### Viterbi Decoding vs Forward Filter

Two sets of regime labels are produced:

**`regime` (Viterbi):** The globally optimal state sequence given all data — computed with the Viterbi algorithm which uses the full forward and backward pass over the entire history. This is retrospectively optimal and uses future data, so it **cannot be used in live trading** without introducing lookahead bias. It is used as the training target for the XGBoost classifier.

**`regime_online` (Forward filter):** A causal, real-time estimate using only data up to and including today. At each step t, it computes the forward probabilities α_t (the probability of being in each state given all observations up to t), then takes the argmax:

```
regime_online_t = argmax_k P(state=k | observations_1:t)
```

This requires no future data and is safe to use in real-time. It is included as a feature in the XGBoost model (a lagged version of today's perceived regime).

### Regime Distribution (2020-08-03 to 2026-05-04, 2101 days)

| Regime | Name | Days | % |
|--------|------|------|---|
| 0 | Bull | 584 | 27.8% |
| 1 | Recovery/Chop | 586 | 27.9% |
| 2 | Bear | 475 | 22.6% |
| 3 | Extreme Bear | 456 | 21.7% |

The near-equal distribution reflects the BIC penalty at work — the model finds genuinely distinct regimes rather than a majority "normal" state with rare outliers.

---

## 3. Stage 2 — Feature Engineering

**Notebook:** `notebooks/2_regime_prediction.ipynb`

32 features are constructed from BTC OHLCV data. Every feature uses only data available at the current day's close — no lookahead. The target is **tomorrow's Viterbi regime** (shifted back by 1 day).

| Group | Features | Signal |
|-------|----------|--------|
| Lagged returns | `ret_1d`, `ret_2d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d` | Recent momentum direction |
| Rolling volatility | `rvol_5d`, `rvol_10d`, `rvol_20d`, `rvol_60d` | Volatility level |
| Vol ratio | `rvol_ratio_5_20` | Vol expansion vs contraction |
| Momentum | `mom_5d`, `mom_10d`, `mom_20d`, `mom_60d` | Cumulative return over window |
| Momentum spread | `mom_spread_5_20` | Short-term vs long-term momentum divergence |
| Drawdown | `dd_20d`, `dd_60d`, `dd_120d`, `dd_252d` | Distance from recent peak at 4 timescales — dominant feature family |
| Recovery | `recov_20d`, `recov_60d`, `recov_120d` | Distance from recent trough — near 0 at crash bottom, rising during recovery even when drawdown is still deep |
| Volume | `vol_chg_5d`, `vol_zscore` | Volume anomaly (regime shifts often coincide with vol spikes) |
| RSI | `rsi_14` | Overbought / oversold condition |
| Bollinger | `bb_width`, `bb_pos` | Volatility compression, price position within bands |
| ATR | `atr_14` | Normalised daily range |
| Regime features | `regime`, `regime_lag1`, `regime_duration` | Forward-filtered current and lagged regime, and how many days the current regime has been active |

The three regime features (`regime`, `regime_lag1`, `regime_duration`) use `regime_online` — the causal forward-filter estimate — ensuring no lookahead bias.

**Why include regime features as inputs?** Regimes are persistent. If today is classified as Bull day 12 (regime_duration=12), tomorrow is more likely to remain Bull than to flip to Extreme Bear. The XGBoost model can learn this persistence explicitly rather than having to re-infer it from price features alone.

**Why `dd_252d`?** `dd_60d` is the single most important feature (gain 0.17). It captures corrections well but cannot distinguish a mid-bull correction (deep short-term, shallow 1-year) from a macro bear (deep at all timescales). The 252-day drawdown adds this macro dimension.

**Why recovery features?** `dd_60d` stays deep for weeks after a crash bottoms while price is already recovering — the model cannot distinguish "still at the bottom" from "bouncing back." `recov_60d` = (close − 60d low) / 60d low answers exactly that: near 0 at the trough, rising quickly during recovery. This lets the model exit Bear predictions earlier at regime turns.

---

## 4. Stage 2 — XGBoost Walk-Forward

### What is XGBoost?

XGBoost (eXtreme Gradient Boosting) is an ensemble learning algorithm that builds a sequence of decision trees, where each new tree corrects the errors of all previous trees combined. It belongs to the family of **gradient boosting** methods.

**Decision trees:** A single decision tree splits the data by asking yes/no questions about features ("is `dd_60d` < −0.15?"), partitioning the feature space into rectangular regions. Each leaf of the tree assigns a class or score to all points that land in it. A single deep tree overfits; a single shallow tree underfits.

**Gradient boosting:** Rather than fitting one deep tree, gradient boosting fits many shallow trees in sequence. Each tree is trained on the **residuals** (errors) of the current ensemble — the difference between the predicted output so far and the truth. The key insight is that this residual-fitting process is equivalent to gradient descent in function space: at each step the algorithm fits a tree that points in the direction that most reduces the loss.

Formally, if the current ensemble prediction is $F_m(x)$, the next tree $h_{m+1}$ is fit to minimise:

$$\sum_i L(y_i,\ F_m(x_i) + h_{m+1}(x_i))$$

where $L$ is the loss function (here, multiclass log-loss). The learning rate $\eta$ scales each tree's contribution: $F_{m+1}(x) = F_m(x) + \eta \cdot h_{m+1}(x)$.

**Why XGBoost specifically?** XGBoost adds several improvements over vanilla gradient boosting:
- **Regularisation** (`reg_alpha`, `reg_lambda`): L1 and L2 penalties on leaf weights prevent individual trees from memorising noise
- **Column and row subsampling** (`colsample_bytree`, `subsample`): Each tree sees only a random subset of features and rows — like random forests, this reduces correlation between trees and improves generalisation
- **Second-order gradients**: Uses both the gradient and Hessian of the loss to fit each tree more accurately, enabling larger learning steps than first-order methods
- **Sparsity-aware splitting**: Handles missing values natively by learning a default direction for each split

**Why XGBoost for regime classification?** The regime label depends on combinations of features in a non-linear way — deep drawdown combined with high volatility signals Bear differently than deep drawdown combined with low volatility (which could be Extreme Bear with a quiet recovery starting). XGBoost's decision tree structure captures these interactions naturally, without requiring the user to specify them explicitly. Linear models or neural networks either miss these interactions or require extensive feature engineering to capture them.

### Why Walk-Forward?

A single train/test split is insufficient for a regime classifier: the model trained on 2020–2022 data has never seen a 2024–2025 bull market and would be predicting out-of-distribution. Walk-forward validation rolls the training window forward in time, producing genuinely out-of-sample (OOS) predictions at every point in the test history — the same way the model would actually be used in live trading.

### Fold Structure

```
Dates:  │←── train (expands) ──→│← val (90d) →│← test (90d) →│  fold 0
        │←──── train (expands) ──────→│← val →│←── test ─────→│  fold 1
        │←────── train ────────────────────→│← val →│← test ──→│  fold 2
                                                        ...
```

The training window expands with each fold (expanding window, not rolling). This means later folds have more training data — analogous to how a live system accumulates history over time.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `min_train_days` | 270 | Minimum rows before first fold starts |
| `val_days` | 90 | Optuna validates hyperparameters here |
| `test_days` | 90 | OOS scoring (never seen by optimiser) |
| `step_days` | 90 | Fold advances 90 days each step |
| `n_optuna_trials` | 100 | Trials per fold for hyperparameter search |
| Total folds | 16 | Covering 2022-04-06 to 2026-03-15 |

### Optuna Hyperparameter Search

Each fold runs **100 Optuna trials** using the **TPE sampler** (Tree-structured Parzen Estimator) with `multivariate=True`.

**What TPE does:** TPE builds a probabilistic model of which hyperparameter values have produced good results so far. It fits a kernel density estimator to the top-performing trials (the "good" region) and another to the rest (the "bad" region). New trial parameters are sampled from the ratio of these two distributions — concentrating search effort on promising regions rather than exploring uniformly like random search.

**Why multivariate=True?** Standard TPE models each hyperparameter independently, missing interactions. For example, `n_estimators` and `learning_rate` interact strongly: high learning rate needs fewer trees; low learning rate needs more. Multivariate TPE models the joint distribution of all parameters together, capturing these interactions.

**Hyperparameter search space:**

| Parameter | Range | Controls |
|-----------|-------|---------|
| `n_estimators` | 100 – 1000 | Ensemble size (number of trees) |
| `max_depth` | 3 – 7 | Maximum tree depth (complexity vs overfitting) |
| `learning_rate` | 0.001 – 0.15 (log) | Gradient step size — log scale because small differences matter near zero |
| `subsample` | 0.5 – 1.0 | Fraction of rows sampled per tree (row bagging) |
| `colsample_bytree` | 0.4 – 1.0 | Fraction of features sampled per tree (column bagging) |
| `min_child_weight` | 1 – 20 | Minimum sum of instance weights in a leaf — controls over-splitting |
| `gamma` | 0.0 – 2.0 | Minimum loss reduction required to make a split |
| `reg_alpha` | 0.0 – 3.0 | L1 regularisation on leaf weights |
| `reg_lambda` | 0.1 – 5.0 | L2 regularisation on leaf weights |

**Why wide ranges?** Ranges that are too narrow cause boundary-hitting — if Optuna consistently wants `learning_rate=0.005` but the lower bound is also 0.005, the model is constrained. The wider ranges let the optimiser converge to the true optimum rather than the allowed boundary.

### Per-Fold Procedure

For each fold:
1. Split data into `train`, `val`, `test` windows
2. Fit `RobustScaler` on training data only (val and test are transformed, not fitted)
3. Run 100 Optuna trials; each trial trains XGBoost on scaled train, evaluates validation accuracy
4. Retrain final model on `train + val` combined with best parameters (more data before scoring test)
5. Score test set: store per-class probabilities, hard prediction, and true label

**Why retrain on train+val?** The val set was used to select hyperparameters, so it is technically IS data for the hyperparameter selection step. Before scoring the truly OOS test set, the final model is given the maximum available data (train+val) to improve generalisation on the test window. This is the standard "walk-forward retrain" approach.

**Class weighting:** `compute_sample_weight('balanced')` is applied during training. Regime classes are not perfectly balanced, and without weighting the model would be biased toward predicting the majority class (Recovery/Chop). Balanced weighting ensures all four regimes are treated equally by the optimiser.

---

## 5. OOS Prediction Results

**16 folds, 1,440 test days, covering 2022-04-06 to 2026-03-15**

### Overall Accuracy

| Metric | Value |
|--------|-------|
| OOS accuracy | **0.871** |
| Naive baseline (always predict majority class) | 0.323 |
| Lift over baseline | **+0.548** |

The naive baseline always predicts regime 1 (Recovery/Chop, ~32% of days). An 87.1% accuracy vs a 32.3% naive baseline shows the model is capturing genuine predictive signal, not just memorising the class distribution.

### Per-Regime Accuracy

| Regime | Name | OOS Accuracy | Test Days |
|--------|------|-------------|-----------|
| 0 | Bull | **90.2%** | 386 |
| 1 | Recovery/Chop | **87.3%** | 465 |
| 2 | Bear | **85.2%** | 385 |
| 3 | Extreme Bear | **84.3%** | 204 |

All four regimes achieve above 84% accuracy. Extreme Bear at 84.3% is a significant improvement over earlier versions that included Fear & Greed features with pre-2018 neutral fills.

### Per-Fold Validation Accuracy

Validation accuracy ranged from **0.789** (fold 9, mid-2024 choppy period) to **0.978** (folds 1 and 14). Low val accuracy folds correspond to regime transition periods where the boundary between adjacent regimes is genuinely ambiguous — not model failure.

### Classification Report

```
               precision    recall  f1-score   support
         Bull       0.91      0.90      0.91       386
Recovery/Chop       0.84      0.87      0.86       465
         Bear       0.87      0.85      0.86       385
 Extreme Bear       0.85      0.84      0.85       204

     accuracy                           0.87      1440
```

Precision and recall are well-balanced across all four classes. Extreme Bear recall (0.84) is the lowest — the model occasionally calls Extreme Bear days as regular Bear, a conservative error that under-filters rather than over-filters positions.

---

## 6. Regime Labelling Output

**File:** `data/labels/btc_regimes.parquet`  
**Shape:** 2101 rows × 9 columns  
**Date range:** 2020-08-03 to 2026-05-04

| Column | Type | Description |
|--------|------|-------------|
| `regime` | int32 | Viterbi regime — retrospectively optimal, uses full history, lookahead unsafe |
| `regime_online` | int32 | Forward-filter regime — causal, no lookahead, safe for live use |
| `p_regime_0` | float64 | Forward-filter probability of being in Bull state at time t |
| `p_regime_1` | float64 | Forward-filter probability of Recovery/Chop state |
| `p_regime_2` | float64 | Forward-filter probability of Bear state |
| `p_regime_3` | float64 | Forward-filter probability of Extreme Bear state |
| `log_ret` | float64 | Log return (input feature stored for reference) |
| `rvol_5d` | float64 | 5-day rolling volatility (input feature) |
| `drawdown` | float64 | 20-day drawdown from high (input feature) |

**Important distinction:** `regime` and `regime_online` will often differ near regime transitions. `regime_online` is noisy at boundaries because it has only seen data up to today. `regime` is smoother because the Viterbi algorithm uses the full forward-backward pass. Neither is "wrong" — they answer different questions.

---

## 7. Forward Returns by Regime

A critical sanity check: do the predicted regime labels actually predict forward BTC returns? If the regime classifier predicted "Bull" on days that then had negative returns, the labels would be economically meaningless even if they achieved high accuracy.

Forward returns (annualised) in the OOS test period by predicted regime:

| Predicted Regime | Avg Annualised Forward Return | Days |
|-----------------|-------------------------------|------|
| 0 — Bull | **+67.9%** | 381 |
| 1 — Recovery/Chop | **−11.8%** | 481 |
| 2 — Bear | **+25.1%** | 376 |
| 3 — Extreme Bear | **−52.4%** | 202 |

Bull and Extreme Bear predictions are correctly positioned at opposite ends. Bear forward returns are positive (+25.1%) — a persistent characteristic of this model: fast V-shaped recoveries within bear markets are predicted as Bear even as price bounces sharply. Since the Long/Flat strategy goes flat (not short) on Bear days, this means missed gains rather than active losses. Extreme Bear is strongly negative (−52.4%), which is what matters most for capital protection.

**Direct regime-as-strategy backtest (OOS period 2022-04-07 to 2026-03-15):**

| Strategy | Ann Return | Sharpe | Max DD | Total Return |
|----------|-----------|--------|--------|-------------|
| Long / Flat / Short | 17.3% | 0.36 | −55.0% | 28.0% |
| **Long / Flat only** | **17.7%** | **0.69** | **−27.7%** | **70.7%** |
| Long / Flat / ExBear Short | 26.1% | 0.63 | −36.5% | 91.8% |
| Long / Half / Flat | 14.1% | 0.49 | −37.0% | 46.1% |
| BTC Buy & Hold | 14.2% | 0.26 | −68.8% | 0.6% |

Long/Flat only achieves the best risk-adjusted return (0.69 Sharpe, −27.7% MaxDD). Adding an Extreme Bear short leg (Long/Flat/ExBear Short) raises annualised return to 26.1% and total return to 91.8% at the cost of slightly lower Sharpe (0.63) and deeper drawdown (−36.5%) — a viable long-short configuration since Extreme Bear has a clean −52.4% average forward return.

**Why not short Bear?** The Bear regime (+25.1% forward return) is structurally unsuitable for shorting: Bear predictions cluster around crash bottoms and post-crash recovery periods where V-shaped reversals produce strong positive returns. Filtering by recovery level (recov_60d), consecutive prediction days, or probability threshold all made the Bear short leg worse — the positive forward return on Bear days is not a filter artefact but a structural property of where the model places Bear labels. Only Extreme Bear produces a reliable negative forward return.

Note: strategy application results in sections 8–10 use the full prediction history including 2021.

---

## 8. Strategy Application — MoneyIn Long

**Notebook:** `topics/momentum/results/moneyin_regime.ipynb`

### Strategy Description

A long-only directional trend strategy on BTC daily bars:
- **Entry:** Close > EMA(14) and no caution flag (no extended swing high)
- **Exit:** Close crosses below ATR(21) trailing stop anchored to recent swing high
- **Stop tightens** when caution flag is active (0.4× ATR) vs normal (1.0× ATR)
- Parameters are fixed — no walk-forward optimisation on the strategy itself

Baseline performance (2020-06-25 to 2026-05-04, no regime filter):
- Total Return: **313.6%** | Sharpe: **0.88** | MaxDD: **−30.0%** | Profit Factor: **1.73** | Win Rate: **52.0%** | Trades: 204

### How the Regime Filter Is Applied

The regime prediction for day T gates the position for day T. Days where the predicted regime is not in the allowed set have their position forced to zero. Days before the prediction coverage begins (before 2021-11-25) default to allowed — no signal is treated as no restriction.

```python
mask = regime_df['pred_regime'].isin(allow_regimes).astype(int)
mask = mask.reindex(df.index).fillna(1)   # default allow before predictions start
df['position'] = df['position'] * mask
```

### Hard Filter Results

| Variant | Allowed Regimes | Return | Sharpe | MaxDD | Calmar | PF | WR | Trades |
|---------|----------------|--------|--------|-------|--------|----|----|--------|
| Baseline | All | 313.6% | 0.88 | −30.0% | 10.45 | 1.73 | 52.0% | 204 |
| **Bull + Chop** | 0, 1 | **369.6%** | **0.98** | **−24.1%** | **15.31** | **1.93** | **54.1%** | **172** |
| Bull only | 0 | 222.9% | 0.83 | −22.1% | 10.06 | 1.86 | 52.2% | 138 |

**Bull + Chop** is the best hard-filter variant. It blocks 43.6% of the strategy's trading days (707 of 1,620 days with predictions) — specifically the Bear and Extreme Bear regime days where forward returns are negative.

**Bull-only** is too restrictive. It blocks 74.8% of days (1,212 of 1,620) and removes too many Chop days that have small but positive average forward returns (+1.4% annualised). The high Calmar ratio (good risk-adjusted return) but lower absolute return reflects a very conservative, rarely-trading variant.

### Yearly Breakdown

| Year | Baseline | Bull+Chop | Bull-only | BTC B&H |
|------|----------|-----------|-----------|---------|
| 2020 | +42.5% | +42.5% | +42.5% | +215.7% |
| 2021 | +41.7% | +48.5% | +48.5% | +57.6% |
| **2022** | **−17.4%** | **−8.8%** | **+3.9%** | **−65.3%** |
| 2023 | +56.6% | +55.3% | +31.8% | +154.5% |
| 2024 | +57.4% | +26.2% | +13.8% | +111.8% |
| 2025 | +3.6% | +14.3% | +3.0% | −7.3% |
| 2026 | −5.1% | +5.5% | −6.1% | −10.1% |

The filter's value is most visible in 2022 (the crypto bear market): baseline lost −17.4% while Bull+Chop limited the loss to −8.8% and Bull-only actually made +3.9%. In bull years (2023, 2024), the filter costs some return because it occasionally sidelines the strategy during valid Chop periods that turned profitable.

---

## 9. Probability Thresholding

Rather than using a hard binary gate (predicted label == Bull), the raw **Bull probability** `p_regime_0` can be used as a continuous confidence score. Only trade when the model is confident, not just when Bull edges out the other classes.

Two modes are tested across thresholds from 0.30 to 0.75:

| Mode | Logic |
|------|-------|
| **Pure** | Allow position only when `p_regime_0 > threshold` |
| **Hybrid** | Allow position when `p_regime_0 > threshold` OR `pred_regime == 1` (Chop always allowed) |

### Pure Threshold Results

| Threshold | Return | Sharpe | MaxDD | Calmar | PF | Blocked% |
|-----------|--------|--------|-------|--------|----|----------|
| 0.30 | 235.2% | 0.85 | −25.2% | 9.33 | 1.86 | 72.8% |
| 0.50 | 234.3% | 0.85 | −22.1% | 10.61 | 1.90 | 76.4% |
| 0.65 | 253.3% | 0.89 | −22.1% | 11.48 | 2.00 | 79.1% |
| **0.70** | **266.3%** | **0.91** | **−22.1%** | **12.07** | **2.04** | **80.7%** |
| 0.75 | 234.3% | 0.86 | −22.1% | 10.61 | 2.00 | 82.2% |

Pure thresholding at 0.70 beats the baseline on Sharpe (0.91 vs 0.88) and MaxDD, but undershoots Bull+Chop on return — because Chop days (regime 1, ~31% of days) have positive forward returns and Pure mode blocks them entirely.

### Hybrid Threshold Results

| Threshold | Return | Sharpe | MaxDD | Calmar | PF | Blocked% |
|-----------|--------|--------|-------|--------|----|----------|
| 0.30 | 369.6% | 0.98 | −24.1% | 15.31 | 1.93 | 44.2% |
| 0.50 | 389.5% | 1.01 | −22.1% | 17.65 | 1.94 | 45.6% |
| 0.60 | 408.0% | 1.03 | −22.1% | 18.48 | 1.99 | 47.8% |
| **0.70** | **422.7%** | **1.05** | **−22.1%** | **19.15** | **2.01** | **49.8%** |
| 0.75 | 374.1% | 1.00 | −22.1% | 16.95 | 1.97 | 51.3% |

### Best Configuration: Hybrid 0.70

| Metric | Baseline | Bull+Chop (hard) | Hybrid p>0.70 | Improvement vs Baseline |
|--------|----------|-----------------|---------------|------------------------|
| Return | 313.6% | 369.6% | **422.7%** | +109.1pp |
| Sharpe | 0.88 | 0.98 | **1.05** | +0.17 |
| MaxDD | −30.0% | −24.1% | **−22.1%** | +7.9pp |
| Calmar | 10.45 | 15.31 | **19.15** | +8.70 |
| Profit Factor | 1.73 | 1.93 | **2.01** | +0.28 |

**Why does the Hybrid at 0.70 beat the hard Bull+Chop filter?**

The hard filter treats all Bull-predicted days equally — a day where the model assigns 51% Bull probability gets the same treatment as a day with 95% Bull probability. The probability threshold distinguishes these: low-confidence Bull predictions (the model is wavering between Bull and Chop, often near regime transitions) are filtered out, removing the weakest entries.

Chop days are always allowed because their average forward return is small but consistently positive (+1.4% annualised) — screening them against a confidence threshold adds noise without value.

---

## 10. Strategy Application — BB Breakout

**Notebook:** `topics/momentum/results/portfolio_bb_regime.ipynb`

### Result

The regime filter **hurts** the BB Breakout strategy:

| Variant | Return | Sharpe | MaxDD |
|---------|--------|--------|-------|
| Baseline (no filter) | ~754% | 2.24 | ~−14% |
| Bull + Chop | 247% | 1.66 | ~−14% |
| Bull only | 247% | 2.05 | ~−14% |

### Why the Filter Hurts Here

BB Breakout is a **volatility expansion strategy**: it profits when price breaks out of a Bollinger Band compression, a signal that typically fires at the onset of large directional moves. The highest-volatility, most explosive breakouts occur precisely in Bear and Extreme Bear regimes — exactly the periods the filter would block.

Unlike MoneyIn Long (which is a sustained trend strategy that needs a stable directional environment), BB Breakout is **regime-agnostic**: the statistical edge of a breakout from a compressed band is similar whether the market is bullish or bearish. Applying a regime filter removes the strategy's best trades.

**The general rule:** The regime filter adds value to strategies whose edge *depends on directional momentum* (trend-following, breakout-with-follow-through). It is **neutral or harmful** for strategies whose edge comes from volatility dynamics, mean-reversion, or statistical arbitrage — these strategies can be profitable regardless of regime.

---

## 11. When the Regime Filter Helps (and When It Doesn't)

| Strategy Type | Filter Effect | Reason |
|---------------|--------------|--------|
| Long-only trend following | **Helps** | Avoids bear market periods where trend runs down |
| Long-biased momentum | **Helps** | Same as above — reduces false entries in negative drift regimes |
| Volatility breakout | **Hurts** | Best breakouts occur at Bear/Extreme Bear vol peaks |
| Mean reversion / stat arb | **Neutral** | Edge from relative pricing, not directional drift |
| Short-only or delta-hedged | **Neutral** | Filter is directionally biased toward Bull |

**Practical guidance:** Before applying the regime filter to any new strategy, check the strategy's per-regime performance. If the strategy makes money in Bear/Extreme Bear — do not apply the filter. If the strategy consistently loses in Bear/Extreme Bear — apply Hybrid 0.70 as a default starting point.

---

## 12. Keeping Predictions Up to Date

The HMM is fitted on the full BTC history available (re-fitted each run). The XGBoost walk-forward adds new folds as new data accumulates. To extend predictions to the current date, re-run both notebooks in order:

1. `notebooks/1_hmm_regime_discovery.ipynb` — re-fits HMM on updated BTC data, saves `btc_regimes.parquet`
2. `notebooks/2_regime_prediction.ipynb` — extends the walk-forward with any new complete 90-day test folds, saves `btc_regime_predictions.parquet`

**Important:** Must use the base `anaconda3` environment — `hmmlearn` is only available there, not in the `nexus` environment.

**Current prediction coverage:** 2022-04-06 to 2026-03-15 (16 folds, 1,440 OOS days)

---

## 13. File Structure

```
regime-classifier/
├── notebooks/
│   ├── 1_hmm_regime_discovery.ipynb      # Stage 1: HMM fitting, regime labelling
│   └── 2_regime_prediction.ipynb         # Stage 2: XGBoost walk-forward classifier
├── data/
│   ├── labels/
│   │   └── btc_regimes.parquet           # HMM output: Viterbi + online regime labels
│   └── predictions/
│       └── btc_regime_predictions.parquet  # XGBoost OOS predictions + probabilities
└── README.md                              # This file

Downstream usage:
topics/momentum/results/
├── moneyin_regime.ipynb                   # MoneyIn Long + regime filter analysis
└── portfolio_bb_regime.ipynb             # BB Breakout + regime filter analysis
```

### Notebook 1 Cell Structure

| Section | Content |
|---------|---------|
| 1. Load BTC Data | Load OHLCV from shared daily cache |
| 2. Feature Engineering | Compute log_ret, rvol_5d, drawdown; standardise |
| 3. BIC Model Selection | Fit HMM for K=2,3,4; plot BIC and log-likelihood; select optimal K |
| 4. Decode Regime Sequences | Viterbi decode → `regime`; relabel states by mean return |
| 5. Regime Characterisation | Per-regime stats: return, vol, Sharpe, drawdown; feature distribution plots |
| 6. Transition Matrix | Heatmap of A matrix; expected duration per regime |
| 7. Regime Timeline | BTC price (log scale) with regime shading + drawdown + vol panels |
| 8. Smoothed Probabilities | Stacked area chart of forward-backward probabilities |
| 9. Drawdown Validation | Verify Bear/Extreme Bear labels correlate with deep drawdowns |
| 10. Forward Filter + Save | Compute `regime_online` via forward pass; write `btc_regimes.parquet` |

### Notebook 2 Cell Structure

| Section | Content |
|---------|---------|
| 1. Load Data | HMM labels from notebook 1 + BTC OHLCV aligned to label index |
| 2. Feature Engineering | 32 price-derived features (incl. 4 drawdown + 3 recovery timescales); target = `next_regime` (Viterbi shifted -1 day) |
| 3. Walk-Forward Config | `WF_CONFIG`: min_train=270, val=90, test=90, step=90, trials=100 |
| 4. Optuna Objective + Fold Runner | `_run_fold`: TPE search (multivariate), retrain on train+val, score test |
| 5. Run Walk-Forward | Execute all 16 folds; per-fold val_acc and selected hyperparameters |
| 6. OOS Accuracy | 0.873 overall; per-regime breakdown; classification report |
| 7. Confusion Matrix | Raw counts + row-normalised recall matrix |
| 8. Feature Importance | Mean gain importance across folds (top 20 bar chart) |
| 9. Hyperparameter Evolution | Per-fold drift of all 9 hyperparameters |
| 10. SHAP Analysis | Beeswarm plots per regime class (last fold) |
| 11. SHAP Dependence | Dependence plots for top Bull and Bear drivers |
| 12. Regime Timeline | True (HMM) vs Predicted (XGBoost) colour strips with price |
| 13. Probability Time Series | Stacked probabilities + Bull probability alone vs price |
| 14. Save | Write `btc_regime_predictions.parquet` |
| 15. Direct Strategy Backtest | Long/Flat only, Long/Flat/ExBear Short, Long/Flat/Short, Long/Half/Flat vs Buy & Hold |

### Downstream Notebook Structure — `moneyin_regime.ipynb`

| Section | Content |
|---------|---------|
| Data | BTC daily OHLCV + regime predictions parquet |
| Strategy | MoneyIn Long: EMA(14) entry, ATR(21) trailing stop |
| Apply Regime Filter | Hard filter: `pred_regime.isin(allow_regimes)` |
| Backtest All Variants | Baseline, Bull+Chop, Bull-only — full metrics table |
| Equity Curves | Three variants vs BTC B&H with drawdown panel and regime strip |
| Yearly Breakdown | Per-year return for each variant |
| Probability Threshold Filter | Pure and Hybrid sweeps across 0.30–0.75; Sharpe/Return/MaxDD vs threshold plots |

### Downstream Notebook Structure — `portfolio_bb_regime.ipynb`

| Section | Content |
|---------|---------|
| Data | Load per-coin BB Breakout OOS pkl files; fetch hourly data for scenario testing |
| Scenarios | Realign entry/exit to specific UTC hours (0–23) or keep at daily close |
| Portfolio Performance | Equal-weight OOS backtest with per-coin stats + cost stress test |
| Regime Filter | Apply Bull+Chop and Bull-only filters to the combined portfolio |
| Closed-Trade Equity | Stepped equity curve on realised trade exits only |

---

## 14. How to Run

### Prerequisites

Base `anaconda3` environment (contains hmmlearn, xgboost, optuna, shap):
```
hmmlearn, xgboost, optuna, shap, pandas, numpy, matplotlib, sklearn, pyarrow
```

### Re-run the full pipeline from scratch

1. Open `notebooks/1_hmm_regime_discovery.ipynb` — run all cells. Saves `btc_regimes.parquet`.
2. Open `notebooks/2_regime_prediction.ipynb` — run all cells. This takes ~10–20 minutes (100 Optuna trials × 18 folds). Saves `btc_regime_predictions.parquet`.

### Extend predictions to today

Re-run both notebooks in order — notebook 1 re-fits the HMM on updated data, notebook 2 adds any new complete 90-day test folds.

### Apply the filter to a new strategy

```python
import pandas as pd

PREDS = r'topics/regime-classifier/data/predictions/btc_regime_predictions.parquet'
regime_df = pd.read_parquet(PREDS)
regime_df.index = pd.to_datetime(regime_df.index).tz_localize(None)

# Hard filter (Bull + Chop)
def apply_hard_filter(df, regime_df, allow_regimes=(0, 1)):
    mask = regime_df['pred_regime'].isin(allow_regimes).astype(int)
    mask = mask.reindex(df.index).fillna(1)  # allow before predictions start
    df = df.copy()
    df['position'] = df['position'] * mask
    return df

# Probability threshold filter (recommended: Hybrid 0.70)
def apply_prob_filter(df, regime_df, threshold=0.70, also_allow_chop=True):
    bull_conf = regime_df['p_regime_0'] > threshold
    if also_allow_chop:
        bull_conf = bull_conf | (regime_df['pred_regime'] == 1)
    mask = bull_conf.astype(int).reindex(df.index).fillna(1)
    df = df.copy()
    df['position'] = df['position'] * mask
    return df
```

---

## 15. Output Schemas

### `btc_regimes.parquet`

Shape: 2101 rows × 9 columns | 2020-08-03 to 2026-05-04

| Column | Type | Description |
|--------|------|-------------|
| `regime` | int32 | Viterbi regime label — 0=Bull, 1=Chop, 2=Bear, 3=Extreme Bear |
| `regime_online` | int32 | Forward-filter causal regime — safe for live use |
| `p_regime_0` | float64 | HMM forward-filter P(Bull) at time t |
| `p_regime_1` | float64 | HMM forward-filter P(Chop) at time t |
| `p_regime_2` | float64 | HMM forward-filter P(Bear) at time t |
| `p_regime_3` | float64 | HMM forward-filter P(Extreme Bear) at time t |
| `log_ret` | float64 | Daily log return (stored for reference) |
| `rvol_5d` | float64 | 5-day rolling volatility |
| `drawdown` | float64 | 20-day drawdown from rolling high |

### `btc_regime_predictions.parquet`

Shape: 1440 rows × 8 columns | 2022-04-06 to 2026-03-15

| Column | Type | Description |
|--------|------|-------------|
| `p_regime_0` | float64 | XGBoost probability: Bull for next day |
| `p_regime_1` | float64 | XGBoost probability: Chop for next day |
| `p_regime_2` | float64 | XGBoost probability: Bear for next day |
| `p_regime_3` | float64 | XGBoost probability: Extreme Bear for next day |
| `pred_regime` | int64 | Hard predicted class: argmax of probabilities |
| `true_regime` | float64 | Actual Viterbi regime (ground truth, for evaluation) |
| `fold` | int64 | Walk-forward fold index (0–15) |
| `val_acc` | float64 | Optuna best validation accuracy for this fold |

**Note:** `p_regime_0` in this file is the XGBoost-predicted probability that **tomorrow** is Bull. This is different from `p_regime_0` in `btc_regimes.parquet`, which is the HMM forward-filter probability that **today** is Bull. Use the predictions file for strategy gating.

---

## 16. Infrastructure Dependencies

| Module | Path | Used in |
|--------|------|---------|
| `binance_client.py` | `infrastructure/data/` | Fetching BTC OHLCV in notebooks and extension script |
| `engine.py` | `infrastructure/backtester/` | Backtesting strategy variants in `moneyin_regime.ipynb` |