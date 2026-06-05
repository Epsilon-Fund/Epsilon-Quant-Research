## CPCV Validation Notebooks
> Hub: [[STRATEGY_REFERENCE]]


Combinatorial Purged Cross-Validation (CPCV) for the momentum strategy family.
One notebook per asset. Run after walk-forward validation.

**To add a new asset:**
1. Copy `cpcv_template.ipynb` and rename to `{asset}_cpcv.ipynb`  
   (e.g. `btc_cpcv.ipynb`, `sol_cpcv.ipynb`).
2. Paste `strategy_fn`, `PARAM_DEFS`, and `FIXED_PARAMS` from the corresponding  
   walk-forward notebook in `../wf_testing_2/`.  
   These must match exactly — CPCV uses the same function signature.
3. Set `WF_SHARPE` to the combined OOS Sharpe from walk-forward (for comparison annotation).
4. Run all cells. Results are saved as `{symbol.lower()}_cpcv.pkl`.

## Notebook Map

| Notebook | Role |
|---|---|
| [[topics/momentum/strategies/momentum_cpcv/cpcv_template.ipynb|cpcv template]] | Template notebook — copy and rename per asset |
| [[topics/momentum/strategies/momentum_cpcv/ADA.ipynb|ADA CPCV]] | Per-asset CPCV notebook |
| [[topics/momentum/strategies/momentum_cpcv/AVAX.ipynb|AVAX CPCV]] | Per-asset CPCV notebook |
| [[topics/momentum/strategies/momentum_cpcv/BNB.ipynb|BNB CPCV]] | Per-asset CPCV notebook |
| [[topics/momentum/strategies/momentum_cpcv/BTC.ipynb|BTC CPCV]] | Per-asset CPCV notebook |
| [[topics/momentum/strategies/momentum_cpcv/ETH.ipynb|ETH CPCV]] | Per-asset CPCV notebook |
| [[topics/momentum/strategies/momentum_cpcv/SOL.ipynb|SOL CPCV]] | Per-asset CPCV notebook |
| [[topics/momentum/strategies/momentum_cpcv/XRP.ipynb|XRP CPCV]] | Per-asset CPCV notebook |
| [[topics/momentum/strategies/momentum_cpcv/portfolio_cpcv.ipynb|portfolio CPCV]] | Portfolio aggregation over per-asset CPCV results |

## Files

| File | Description |
|------|-------------|
| `cpcv_template.ipynb` | Template notebook — copy and rename per asset |
| `test_cpcv.py` | End-to-end validation script for the CPCV engine and visualiser |
| `*_cpcv.ipynb` | Per-asset CPCV notebooks (created from template) |
| `*_cpcv.pkl` | Pickled results dicts (created on first run) |
