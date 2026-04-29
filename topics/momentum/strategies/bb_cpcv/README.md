CPCV validation notebooks for the BB breakout strategy.

One CPCV notebook per asset/timeframe combination.
Copy `cpcv_template.ipynb` and rename to
`{asset}_{timeframe}_cpcv.ipynb` (e.g. `btc_1h_cpcv.ipynb`).
Paste `strategy_fn`, `PARAM_DEFS`, and `FIXED_PARAMS` from the
corresponding BB breakout walk-forward notebook.
Run per-asset notebooks before `portfolio_cpcv.ipynb`.

## Assets

| File | Symbol | Timeframe |
|------|--------|-----------|
| BTC.ipynb | BTCUSDT | 1h |
| ETH.ipynb | ETHUSDT | 1h |
| ADA.ipynb | ADAUSDT | 1h |
| SOL.ipynb | SOLUSDT | 1h |
| AVAX.ipynb | AVAXUSDT | 1h |
| DOT.ipynb | DOTUSDT | 1h |
| LINK.ipynb | LINKUSDT | 1h |
| MATIC.ipynb | POLUSDT | 1h |
| NEAR.ipynb | NEARUSDT | 1h |

## Pkl naming convention

Each per-asset notebook saves: `oos/{symbol.lower()}_{interval}_cpcv.pkl`

Example: BTCUSDT at 1h → `oos/btcusdt_1h_cpcv.pkl`

## Walk-forward reference folder

`../bb_breakout_wf/` — source of `strategy_fn`, `PARAM_DEFS`, and `FIXED_PARAMS`.
