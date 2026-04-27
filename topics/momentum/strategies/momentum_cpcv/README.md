## CPCV Validation Notebooks

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

**Files**

| File | Description |
|------|-------------|
| `cpcv_template.ipynb` | Template notebook — copy and rename per asset |
| `test_cpcv.py` | End-to-end validation script for the CPCV engine and visualiser |
| `*_cpcv.ipynb` | Per-asset CPCV notebooks (created from template) |
| `*_cpcv.pkl` | Pickled results dicts (created on first run) |
