# Infrastructure — Backtesting & Validation Engines

The shared machinery every crypto strategy in `topics/` is validated on. Strategy notebooks stay thin — they define a signal function and a parameter space, then call into these engines for backtesting, walk-forward optimisation, Combinatorial Purged Cross-Validation, and the overfitting audit. One implementation of the hard parts, reused everywhere, so a validation fix lands in every strategy at once.

## Modules

| Module | What it does |
|---|---|
| [`backtester/`](backtester/) | Vectorised event-loop backtester (`engine.py`): takes OHLCV + a position series, applies a 1-bar execution lag and per-leg costs, returns the full metric set (`performance_metrics.py`), portfolio aggregation helpers (`portfolio_metrics.py`), ML-specific metrics, and Plotly visualisers |
| [`walkforward/`](walkforward/) | `wf_engine.py` — rolling train/test optimisation with Optuna TPE search per fold; `cpcv_engine.py` — combinatorial purged cross-validation (N-choose-k splits, boundary purging, complete out-of-sample path stitching, overlap-aware confidence intervals); `cpcv_portfolio.py` + visualisers — multi-asset aggregation; `xs_strategy.py`, `ls_diagnostics.py` — cross-sectional and long/short tooling |
| [`validation/`](validation/) | `overfitting_audit.py` — the anti-self-deception layer: Deflated Sharpe Ratio (trial-count and effective-N aware), Probability of Backtest Overfitting via CSCV, White's Reality Check (studentised stationary bootstrap), stationary-bootstrap confidence intervals, minimum track-record length, synthetic-null Monte Carlo, and a pre-registered verdict object; covered by 38 unit tests |
| [`ml/`](ml/) | Rolling walk-forward engine for the ML prediction model (per-fold scaler → XGBoost → isotonic calibration → threshold selection on validation only) and feature-dataset builders |
| [`data/`](data/) | Ingestion clients: Binance (REST + bulk vision archives), Deribit, futures, macro, and on-chain series — feeding the parquet caches the notebooks and the live app read |

## How `topics/` notebooks consume this

Notebooks add the engine folders to `sys.path` and call the entry points directly — no package install, no hidden state:

```python
sys.path.insert(0, "<repo>/infrastructure/walkforward")
from cpcv_engine import run_cpcv

results = run_cpcv(
    df, strategy_fn, param_defs, fixed_params,
    N=8, k=2, n_trials=400, purge_bars=1,
    collect_trials=True,        # keep every explored config for the overfitting audit
)
```

`collect_trials=True` is structural: the audit in `validation/` needs the *full candidate matrix* a search chose from — not just the winner — to price the selection bias. The CPCV template notebook ends with a required gate cell that runs the audit and prints a pass/fail verdict; no strategy graduates to the [live app](../live_trading/README.md) without it.

## The validation philosophy

- **Leakage is treated as the default failure mode**: purging at every train/test boundary, indicator warm-up handled inside folds, execution lagged 1 bar, costs charged per leg.
- **One out-of-sample path is an anecdote.** CPCV produces 105 stitched OOS paths per asset (N = 8, k = 2), and confidence intervals use the overlap-adjusted effective path count.
- **Search effort is priced, not ignored.** Selecting the best of 400 Optuna trials inflates any backtest; the audit quantifies that inflation (DSR haircut), checks whether in-sample winners hold up out-of-sample (PBO), and verifies the family beats noise after accounting for the search (Reality Check, synthetic-null Monte Carlo).

Worked example of the whole stack end-to-end: the [momentum CPCV folder](../topics/momentum/strategies/momentum_cpcv/README.md) and its [audit findings](../topics/momentum/strategies/momentum_cpcv/momentum_overfitting_audit_findings.md). Plain-English explainer of the statistics: [docs/OVERFITTING_VALIDATION.md](../docs/OVERFITTING_VALIDATION.md).

Vault hub: docs/STRATEGY_REFERENCE.md
