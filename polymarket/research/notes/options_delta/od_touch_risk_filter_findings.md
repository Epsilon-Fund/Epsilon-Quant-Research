---
title: "OD Touch-Risk Filter Findings"
created: 2026-06-05
status: closed
owner: justin
project: polymarket
para: project
hubs:
  - strat_options_delta
  - COWORK
tags:
  - research
  - options-delta
---
# OD Touch-Risk Filter Findings

> Hub: [[strat_options_delta]] · [[COWORK]]
> Related: [[od_strategy_a_realism_reaudit_findings]] · [[block_k6_vol_findings]] · [[od_same_day_crypto_pricing_gate_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Summary

- Scope: OD Touch-Risk Filter Findings in the OD/options-delta area.
- Existing takeaway/status: Verdict: **LOG FEATURE ONLY**.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Headline

Verdict: **LOG FEATURE ONLY**.

The cheap offline test does **not** justify adding a Binance touch-risk skip gate to the longshot-harvest live loop. The held-out score has only weak separation: bad-touch AUC **0.616**, CI **[0.603, 0.628]**. Jump-driven touches, the exact events this idea most wanted to catch, are not cleanly forecastable either: AUC **0.634**, CI **[0.606, 0.659]**.

On the tiny PM rv-edge-scaled fill set, take-all stress EV is 4.19c [0.55c, 6.46c]. The best skip row is `skip_top_20pct` at 4.19c [0.82c, 6.66c], retaining 100.00% of weighted fills. This is a diagnostic over only 7 kept PM market clusters, not a powered gate.

The result is useful precisely because it is cheap: causal momentum, taker-flow imbalance, and Lee-Mykland jump features can be logged as descriptive live telemetry, but the offline evidence does not support a hard skip rule.

## Design

The unit for the powered separation test is one independent 4h Binance window. The script scans the **35,556** cached BTC/ETH/SOL 4h windows from the grown tail base and selects the first state in each window where the out-of-the-money side is at least `|z| >= 1` after the first 30 minutes. That yields **34,687** candidate windows. The side being scored is the side a longshot seller would not want to see touch: UP when spot is below strike, DOWN when spot is above strike.

The label `bad_touch` is whether the future Binance high/low crosses the strike before the 4h window ends, excluding the already-closed 5-minute bar used for the features. `terminal_bad` is whether the same side resolves in-the-money at the final close. `touch_jump_driven` is a coarse stress label: a bad-touch window with at least one future adverse Lee-Mykland jump before expiry. It does not identify the exact first-touch bar.

Features are causal and fixed before held-out evaluation:
- Distance/state: `abs_z`, time to expiry, EWMA sigma, and model OTM probability.
- Momentum: adverse-direction 15m/30m/60m/120m log-return sums.
- Flow proxy: adverse-direction taker-flow imbalance from Binance kline taker-buy volume over the same horizons. This is **not true L2 OFI** because Binance order-book depth is missing from the saved artifacts.
- Jumps: adverse and opposite Lee-Mykland jump counts plus max adverse return and realized vol.

Modeling is deliberately simple: a regularized logistic score trained before **2025-01-01**, with a one-week embargo and held-out evaluation from **2025-01-08** onward. Skip thresholds are train-score quantiles only; no threshold is tuned on held-out or PM fills.

## Separation

| sample | label | rows | positives | base rate | AUC | CI |
|---|---:|---:|---:|---:|---:|---:|
| heldout | bad_touch | 8,937 | 2,263 | 25.32% | 0.616 | [0.603, 0.628] |
| heldout | terminal_bad | 8,937 | 1,162 | 13.00% | 0.600 | [0.583, 0.616] |
| heldout | jump_driven_touch_vs_no_touch | 7,082 | 408 | 5.76% | 0.634 | [0.606, 0.659] |
| heldout_BTC | bad_touch | 2,981 | 750 | 25.16% | 0.614 | [0.591, 0.636] |
| heldout_ETH | bad_touch | 2,971 | 742 | 24.97% | 0.623 | [0.599, 0.646] |
| heldout_SOL | bad_touch | 2,985 | 771 | 25.83% | 0.614 | [0.592, 0.637] |

Read: the score is directionally above random for ordinary touches, but it is not strong enough for a trading gate. A hard skip feature should be catching toxic windows clearly; this score is in the "weak regime descriptor" range. The jump-driven row is the key honesty check: if jumps are the real left-tail, this offline score does not forecast them well.

![Held-out risk deciles](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_touch_risk_filter_decile_touch_rate.png)

Caption: held-out bad-touch and terminal-bad rates by score decile using train-set cutpoints. A deployable skip filter would show a steep monotone lift in the top deciles.

## Skip Sweep On Binance Tail Base

These rows apply train-score thresholds to train and held-out candidate windows. The table is about bad-event separation only; there is no PM price in the Binance tail base, so it is not a PnL table.

| sample | rule | retained_fraction | skipped_touch_rate | kept_touch_rate | skipped_terminal_bad_rate | kept_terminal_bad_rate | skipped_jump_driven_touch_rate | kept_jump_driven_touch_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heldout | skip_top_30pct | 63.97% | 32.92% | 21.04% | 16.99% | 10.76% | 6.49% | 3.48% |
| heldout | skip_top_20pct | 75.71% | 34.27% | 22.45% | 18.38% | 11.28% | 7.14% | 3.74% |
| heldout | skip_top_10pct | 86.87% | 34.95% | 23.87% | 19.69% | 11.99% | 7.33% | 4.15% |
| heldout | skip_top_5pct | 93.10% | 35.33% | 24.58% | 18.64% | 12.58% | 7.94% | 4.31% |

Read: skipping high-score windows lowers retained bad-touch rates somewhat, but the separation is too soft. A useful skip rule would remove a small toxic tail while preserving most good windows; here the retained-versus-skipped difference is not large enough to justify reducing an already-small live fill opportunity set.

## PM Fill Overlay

The PM overlay uses `od_strategy_a_tail_sizing_weighted_fills.parquet`, the same 7-market / 22-fill Strategy-A stress sample from [[od_strategy_a_realism_reaudit_findings]]. This is too small for a new OOS result, so it is only a sanity check: does the score obviously improve stress EV when applied to the actual harvest rows?

| policy | rule | fills_after | markets_after | retained_fraction_weighted | mean_stress_ev | stress_ev_ci_lo | stress_ev_ci_hi | mean_tail_ev | mean_realized_claim_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flat_1_contract | take_all | 22 | 7 | 100.00% | 1.74c | -0.36c | 5.44c | 4.14c | 18.36c |
| flat_1_contract | skip_top_30pct | 22 | 7 | 100.00% | 1.74c | -0.08c | 5.73c | 4.14c | 18.36c |
| flat_1_contract | skip_top_20pct | 22 | 7 | 100.00% | 1.74c | -0.72c | 5.44c | 4.14c | 18.36c |
| flat_1_contract | skip_top_10pct | 22 | 7 | 100.00% | 1.74c | -0.36c | 5.45c | 4.14c | 18.36c |
| flat_1_contract | skip_top_5pct | 22 | 7 | 100.00% | 1.74c | -0.37c | 5.44c | 4.14c | 18.36c |
| rv_edge_scaled | take_all | 22 | 7 | 100.00% | 4.19c | 0.55c | 6.46c | 5.79c | 14.07c |
| rv_edge_scaled | skip_top_30pct | 22 | 7 | 100.00% | 4.19c | 0.68c | 6.54c | 5.79c | 14.07c |
| rv_edge_scaled | skip_top_20pct | 22 | 7 | 100.00% | 4.19c | 0.82c | 6.66c | 5.79c | 14.07c |
| rv_edge_scaled | skip_top_10pct | 22 | 7 | 100.00% | 4.19c | 0.32c | 6.15c | 5.79c | 14.07c |
| rv_edge_scaled | skip_top_5pct | 22 | 7 | 100.00% | 4.19c | 0.68c | 6.35c | 5.79c | 14.07c |

![PM skip stress EV](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_touch_risk_filter_pm_skip_stress_ev.png)

Caption: rv-edge-scaled PM stress EV after train-threshold skip rules. Error bars are market-cluster bootstrap CIs over the tiny PM sample.

Read: no PM skip row earns promotion. The train-set thresholds do not drop any of the observed flat/fair Strategy-A fills, so the unchanged point estimate and small CI wiggle are bootstrap noise over the same kept rows. This is not enough to alter the live-loop sizing recommendation from [[od_strategy_a_realism_reaudit_findings]].

## Data Ledger

| bucket | read |
|---|---|
| available | 35,556 cached BTC/ETH/SOL 4h windows; 34,687 independent first-far candidate windows; Binance 5m klines with taker-buy volume from cached monthly zips. |
| missing | True Binance L2/order-book OFI is not in the saved cache. The flow term here is taker-flow imbalance from kline taker-buy volume, so it is a TFI/OFI proxy, not exchange order-book OFI. |
| lookahead | Features use bars whose close timestamp is at or before the candidate/fill timestamp. Future highs/lows are used only for labels. |

## Decision

**Drop the hard skip idea cheaply.** The offline score has weak touch separation, no compelling jump-driven separation, and no reliable PM stress-EV improvement. Do not add it as a required gate before quoting.

**Keep as live telemetry only:** log adverse momentum, taker-flow imbalance proxy, Lee-Mykland jump flags, distance-to-strike, and barrier acceleration in the live measurement loop. These are useful audit fields for post-fill adverse-selection analysis, but not a pre-trade skip rule yet.

Modeled assumptions:
- Binance 5m klines and taker-buy volume are sufficient for a cheap pre-test of momentum/flow/jump state.
- First `|z| >= 1` OTM state per 4h window is a fair independent proxy for a longshot-seller risk decision.
- Logistic weights and skip thresholds are trained before the held-out period.
- PM overlay uses existing stress EV, net of costs/rebate, with no mark-to-mid.

Live-only unknowns:
- True Binance L2/order-book OFI at quote time.
- Passive fill share and queue position conditional on risk state.
- Whether a score trained on historical Binance states maps to Polymarket maker fills when PM quotes are actually live.
- Whether jump-risk features matter only in the seconds before a touch, below the 5-minute historical resolution used here.

## Outputs

- Script: `scripts/od_touch_risk_filter.py`
- Candidate windows: `data/analysis/od_touch_risk_filter_candidate_windows.parquet`
- Scored PM fills: `data/analysis/od_touch_risk_filter_scored_pm_fills.parquet`
- Separation CSV: `data/analysis/csv_outputs/options_delta/od_touch_risk_filter_separation.csv`
- Decile CSV: `data/analysis/csv_outputs/options_delta/od_touch_risk_filter_deciles.csv`
- Skip sweep CSV: `data/analysis/csv_outputs/options_delta/od_touch_risk_filter_skip_sweep.csv`
- PM skip CSV: `data/analysis/csv_outputs/options_delta/od_touch_risk_filter_pm_skip.csv`
