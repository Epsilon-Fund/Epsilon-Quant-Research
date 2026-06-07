---
title: "Readme"
created: 2026-04-29
status: active
owner: justin
project: crypto
para: resource
hubs:
  - STRATEGY_REFERENCE
tags:
  - crypto
  - research
  - momentum
---
CPCV validation notebooks for the BB breakout strategy.

One CPCV notebook per asset/timeframe combination.
Copy `cpcv_template.ipynb` and rename to
`{asset}_{timeframe}_cpcv.ipynb` (e.g. `btc_1h_cpcv.ipynb`).
Paste `strategy_fn`, `PARAM_DEFS`, and `FIXED_PARAMS` from the
corresponding BB breakout walk-forward notebook.
Run per-asset notebooks before `portfolio_cpcv.ipynb`.

## Assets
> Hub: [[STRATEGY_REFERENCE]]


| File | Symbol | Timeframe |
|------|--------|-----------|
| [[topics/momentum/strategies/bb_cpcv/BTC.ipynb|BTC.ipynb]] | BTCUSDT | 1h |
| [[topics/momentum/strategies/bb_cpcv/ETH.ipynb|ETH.ipynb]] | ETHUSDT | 1h |
| [[topics/momentum/strategies/bb_cpcv/ADA.ipynb|ADA.ipynb]] | ADAUSDT | 1h |
| [[topics/momentum/strategies/bb_cpcv/AVAX.ipynb|AVAX.ipynb]] | AVAXUSDT | 1h |
| [[topics/momentum/strategies/bb_cpcv/LINK.ipynb|LINK.ipynb]] | LINKUSDT | 1h |
| [[topics/momentum/strategies/bb_cpcv/MATIC.ipynb|MATIC.ipynb]] | POLUSDT | 1h |
| [[topics/momentum/strategies/bb_cpcv/NEAR.ipynb|NEAR.ipynb]] | NEARUSDT | 1h |
| [[topics/momentum/strategies/bb_cpcv/cpcv_template.ipynb|cpcv_template.ipynb]] | template | 1h |
| [[topics/momentum/strategies/bb_cpcv/portfolio_cpcv.ipynb|portfolio_cpcv.ipynb]] | portfolio | 1h |

## Walk-Forward Parameter Notebooks

- [[topics/momentum/strategies/bb_cpcv/wf_params/ADA.ipynb|ADA WF params]]
- [[topics/momentum/strategies/bb_cpcv/wf_params/AVAX.ipynb|AVAX WF params]]
- [[topics/momentum/strategies/bb_cpcv/wf_params/BTC.ipynb|BTC WF params]]
- [[topics/momentum/strategies/bb_cpcv/wf_params/ETH.ipynb|ETH WF params]]
- [[topics/momentum/strategies/bb_cpcv/wf_params/NEAR.ipynb|NEAR WF params]]
- [[topics/momentum/strategies/bb_cpcv/wf_params/portfolio_cpcv.ipynb|portfolio WF params]]

## Pkl naming convention

Each per-asset notebook saves: `oos/{symbol.lower()}_{interval}_cpcv.pkl`

Example: BTCUSDT at 1h → `oos/btcusdt_1h_cpcv.pkl`

## Walk-forward reference folder

[[topics/momentum/strategies/bb_breakout_wf/README|BB breakout WF README]] — source of `strategy_fn`, `PARAM_DEFS`, and `FIXED_PARAMS`.
