# Momentum Research Hub

Crypto momentum research lives here: the live momentum family and its validation trail (walk-forward → CPCV → overfitting audit), Bollinger-breakout research on the same rails, and cross-sectional momentum experiments. The pattern is consistent across all of it: a strategy earns its way from a scratch notebook, through walk-forward optimisation, into Combinatorial Purged Cross-Validation, and only then into the [live trading app](../../live_trading/README.md).

## Where things live

| Folder | What's inside |
|---|---|
| [strategies/momentum_cpcv/](strategies/momentum_cpcv/README.md) | **The flagship validation gate** — per-asset CPCV notebooks for the live momentum family, the overfitting audit (DSR / PBO / Reality Check), and the synthetic-null Monte Carlo |
| [strategies/bb_cpcv/](strategies/bb_cpcv/README.md) | CPCV notebooks for the Bollinger-breakout family |
| [strategies/bb_breakout_wf/](strategies/bb_breakout_wf/README.md) | BB-breakout walk-forward and design notebooks |
| [strategies/wf_testing_2/](strategies/wf_testing_2/README.md) | Current-generation walk-forward notebooks for the trend follower (the step before CPCV) |
| [strategies/wf_testing/](strategies/wf_testing/README.md) | First-generation per-asset walk-forward notebooks |
| [strategies/xs_cpcv/](strategies/xs_cpcv/README.md) | Cross-sectional momentum CPCV notebooks |
| [strategies/testing/](strategies/testing/README.md) | Older strategy scratch notebooks (kept for the paper trail) |
| [xs_momentum/](xs_momentum/README.md) | Cross-sectional momentum research: universe construction, long/short notebooks, OOS artifacts |
| [outputs/](outputs/README.md) | Generated walk-forward HTML reports and fold-results CSVs |
| [results/](results/README.md) | Portfolio, CPCV, and regime-filter result notebooks |
| [research/](research/) | Side studies (Kronos forward-vol evaluation, cross-project hybrid scripts, weekly notes) |
| `wf_template.ipynb` | Reusable walk-forward notebook template — the entry point for any new asset or variant |

## The validation ladder

1. **Walk-forward** (`wf_template.ipynb`, `strategies/wf_testing_2/`) — rolling train/test optimisation with the engines in [infrastructure/](../../infrastructure/README.md); every fold's search is logged.
2. **CPCV** (`strategies/momentum_cpcv/`, `strategies/bb_cpcv/`) — 28 purged combinatorial splits stitched into 105 out-of-sample paths per asset, with overlap-aware confidence intervals.
3. **Overfitting gate** — Deflated Sharpe Ratio, PBO via CSCV, White's Reality Check, and a synthetic-null Monte Carlo, with a pre-registered pass bar. The first full application is written up in [momentum_overfitting_audit_findings.md](strategies/momentum_cpcv/momentum_overfitting_audit_findings.md).
4. **Live** — only after all three: the strategy gets a dashboard in [live_trading/](../../live_trading/README.md) and its journal starts.

## Related topics

- [ML prediction notebooks](../ml-prediction/notebooks/README.md) — XGBoost prediction experiments
- [Regime classifier](../regime-classifier/README.md) — BTC regime labelling used as a filter overlay

Vault hub: docs/STRATEGY_REFERENCE.md · data artifacts: docs/CRYPTO_DATA_MANIFEST.md
