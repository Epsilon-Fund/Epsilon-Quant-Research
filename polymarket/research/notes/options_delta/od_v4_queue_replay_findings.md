---
title: "OD v4 Exploratory Queue Replay: Can Passive Rich-Short Quotes Actually Fill?"
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
# OD v4 Exploratory Queue Replay: Can Passive Rich-Short Quotes Actually Fill?

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Gate note: [[od_v4_calibration_gate_findings]]
> MM benchmark: [[mm_deployable_cells_findings]] · [[block_k5_stress_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Summary

- Scope: OD v4 Exploratory Queue Replay: Can Passive Rich-Short Quotes Actually Fill? in the OD/options-delta area.
- Existing takeaway/status: Official gate context: Phase 0 failed: 23 fills / 8 markets, gross EV 16.92c, CI [-1.43c, 26.38c], realized ITM 39.13%.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Headline

Official gate context: Phase 0 failed: 23 fills / 8 markets, gross EV 16.92c, CI [-1.43c, 26.38c], realized ITM 39.13%.

User override: this queue replay was run anyway as an exploratory falsifier. It is **not** a reversal of the Phase 0 gate.

Exploratory Phase 1 verdict: **FAIL**.

Best exploratory OOS row after the incumbent haircut: 1.98c mean per market, CI [0.66c, 3.43c], 11 markets, 1.55 expected filled contracts.

Crucial read: the best deployable row uses `required_edge = 0c`. The 1c OD-richness variants are still positive in cents, but their `lift vs edge0` is negative. In other words, queue-aware execution found a tiny source/structure/capacity result, not evidence that the OD fair-value richness filter adds independent edge.

Plain-English read: queue realism does not create a new edge by itself. It changes which rich-short quotes plausibly fill, how much capacity survives the queue, and how much survives the incumbent-maker haircut. Because Phase 0 did not prove the primitive `price - realized ITM` edge, this note should be read as "what would execution look like if we pursued it anyway?"

## What This Replay Does

For every captured crypto-4h quote state in the K6/K3 panel, the replay asks whether the token at the ask is rich versus OD fair value. If it is rich enough, we try to passively **sell the rich token** at an ask quote that obeys the fair bound:

```text
quote price >= OD token fair + required edge
quote price > current best bid
```

Then the replay looks forward inside the captured LOB stream. A fill is counted only if later same-token BUY trade flow reaches our quote after consuming queue ahead:

```text
filled units = min(order size, max(0, future BUY flow at/above quote - queue ahead))
```

This is stricter than the v3 capacity proxy because it uses future trade flow to consume visible queue ahead. It is still a proxy: queue identity, hidden cancels, self queue rank, and true Polymarket matching-engine priority are not observable.

Sample construction: the base join produced 126,388 far/source quote states. The replay then uses one decision per market/token every 15 seconds, leaving 1,414 market-token-config combinations with at least one queue-adjusted fill. The cadence prevents the same future taker flow from being counted against many near-identical one-second quote states.

## Config Columns

`required_edge` is the minimum cents above OD fair demanded before selling the token. `improve_ticks` says how many 1c ticks we improve from the existing ask while staying passive and inside the fair bound. `wait_sec` is how long the quote rests before cancel/requote. `toxicity=basic` skips incomplete books, one-sided thin books, very wide spreads, and recent buy-flow / ask-depletion pressure that would be adverse for a seller. `dollar_delta_cap` clips expected fills when the market episode's running dollar-delta exposure gets too large.

`queue_adjusted` is the raw queue model. `queue_adjusted_after_top3_haircut` scales units and PnL by 5%, matching the K5 reality that top-3 maker wallets took about 95% of positive crypto-4h maker profit.

![Queue replay OOS summary](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_v4_queue_replay_summary.png)

## Best OOS Rows After Incumbent Haircut

Unit of observation: one market episode. PnL is resolution PnL, not mark-to-mid. CI is bootstrapped by market.

| config_id | markets | quote fills | filled units | edge | improve ticks | wait sec | toxicity | cap | mean net | median net | CI | win | mean ROC | ROC CI | daily PnL | lift vs edge0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edge_00c_imp_1t_wait_300s_tox_none_cap_50 | 11 | 37 | 1.55 | 0.00c | 1 | 300 | none | $50 | 1.98c | 0.63c | [0.66c, 3.43c] | 81.82% | 23.84% | [-8.00%, 53.12%] | $0.22 | 0.00c |
| edge_00c_imp_1t_wait_300s_tox_basic_cap_50 | 11 | 37 | 1.54 | 0.00c | 1 | 300 | basic | $50 | 1.92c | 0.63c | [0.66c, 3.35c] | 81.82% | 23.34% | [-8.27%, 51.81%] | $0.21 | 0.00c |
| edge_01c_imp_1t_wait_300s_tox_basic_cap_50 | 11 | 34 | 1.38 | 1.00c | 1 | 300 | basic | $50 | 1.54c | 0.63c | [0.51c, 2.74c] | 81.82% | 14.54% | [-22.41%, 51.41%] | $0.17 | -0.38c |
| edge_01c_imp_1t_wait_300s_tox_none_cap_50 | 11 | 34 | 1.36 | 1.00c | 1 | 300 | none | $50 | 1.53c | 0.63c | [0.45c, 2.73c] | 81.82% | 11.27% | [-23.87%, 44.03%] | $0.17 | -0.44c |
| edge_00c_imp_1t_wait_30s_tox_basic_cap_50 | 11 | 42 | 1.67 | 0.00c | 1 | 30 | basic | $50 | 2.89c | 0.40c | [0.33c, 6.42c] | 63.64% | 120.45% | [-23.75%, 361.52%] | $0.32 | 0.00c |
| edge_01c_imp_1t_wait_30s_tox_none_cap_50 | 11 | 36 | 1.44 | 1.00c | 1 | 30 | none | $50 | 2.58c | 0.40c | [0.33c, 5.88c] | 72.73% | 98.56% | [-37.31%, 326.75%] | $0.28 | -0.30c |
| edge_01c_imp_0t_wait_300s_tox_basic_cap_50 | 11 | 37 | 1.55 | 1.00c | 0 | 300 | basic | $50 | 1.10c | 0.54c | [0.33c, 2.00c] | 72.73% | -9.24% | [-45.13%, 23.53%] | $0.12 | -0.37c |
| edge_00c_imp_1t_wait_30s_tox_none_cap_50 | 11 | 40 | 1.61 | 0.00c | 1 | 30 | none | $50 | 2.88c | 0.40c | [0.31c, 6.33c] | 63.64% | 100.26% | [-24.28%, 301.59%] | $0.32 | 0.00c |
| edge_01c_imp_0t_wait_300s_tox_none_cap_50 | 11 | 37 | 1.51 | 1.00c | 0 | 300 | none | $50 | 1.07c | 0.46c | [0.30c, 1.94c] | 72.73% | -9.88% | [-45.58%, 22.18%] | $0.12 | -0.34c |
| edge_01c_imp_1t_wait_30s_tox_basic_cap_50 | 11 | 37 | 1.46 | 1.00c | 1 | 30 | basic | $50 | 2.60c | 0.40c | [0.29c, 5.98c] | 72.73% | 126.00% | [-35.48%, 403.29%] | $0.29 | -0.29c |
| edge_00c_imp_0t_wait_300s_tox_none_cap_50 | 11 | 40 | 1.70 | 0.00c | 0 | 300 | none | $50 | 1.41c | 0.46c | [0.24c, 2.76c] | 63.64% | -4.69% | [-43.17%, 32.17%] | $0.16 | 0.00c |
| edge_00c_imp_0t_wait_300s_tox_basic_cap_50 | 11 | 41 | 1.72 | 0.00c | 0 | 300 | basic | $50 | 1.47c | 0.54c | [0.24c, 2.82c] | 72.73% | -3.58% | [-42.65%, 33.72%] | $0.16 | 0.00c |

Read: this is the deployability view. A positive row here means the replay found plausible queue fills and then applied the non-incumbent capacity haircut. A lower CI above zero is encouraging only as an execution proxy; it cannot override Phase 0 calibration failure.

## Raw Queue-Adjusted OOS Rows Before Incumbent Haircut

| config_id | markets | quote fills | filled units | edge | improve ticks | wait sec | toxicity | cap | mean net | median net | CI | win | total capital |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edge_00c_imp_1t_wait_300s_tox_none_cap_50 | 11 | 37 | 31.00 | 0.00c | 1 | 300 | none | $50 | 39.50c | 12.55c | [12.83c, 69.25c] | 81.82% | $11.61 |
| edge_00c_imp_1t_wait_300s_tox_basic_cap_50 | 11 | 37 | 30.84 | 0.00c | 1 | 300 | basic | $50 | 38.49c | 12.55c | [12.49c, 67.36c] | 81.82% | $11.58 |
| edge_01c_imp_1t_wait_300s_tox_none_cap_50 | 11 | 34 | 27.14 | 1.00c | 1 | 300 | none | $50 | 30.65c | 12.55c | [9.70c, 54.56c] | 81.82% | $9.73 |
| edge_01c_imp_1t_wait_300s_tox_basic_cap_50 | 11 | 34 | 27.63 | 1.00c | 1 | 300 | basic | $50 | 30.84c | 12.55c | [9.52c, 54.35c] | 81.82% | $9.46 |
| edge_00c_imp_1t_wait_30s_tox_basic_cap_50 | 11 | 42 | 33.47 | 0.00c | 1 | 30 | basic | $50 | 57.88c | 7.92c | [7.29c, 128.23c] | 63.64% | $10.76 |
| edge_01c_imp_0t_wait_300s_tox_basic_cap_50 | 11 | 37 | 30.91 | 1.00c | 0 | 300 | basic | $50 | 21.92c | 10.83c | [6.84c, 40.20c] | 72.73% | $15.03 |
| edge_00c_imp_0t_wait_300s_tox_basic_cap_50 | 11 | 41 | 34.39 | 0.00c | 0 | 300 | basic | $50 | 29.34c | 10.83c | [6.38c, 57.04c] | 72.73% | $15.82 |
| edge_00c_imp_1t_wait_30s_tox_none_cap_50 | 11 | 40 | 32.11 | 0.00c | 1 | 30 | none | $50 | 57.53c | 7.92c | [6.37c, 126.12c] | 63.64% | $10.89 |

Read: this is what the same queue model says before applying the K5 top-maker haircut. The gap between this and the deployable table is the capacity/moat issue, not a pricing issue.

## Discovery Sample Sanity Check

| config_id | markets | quote fills | filled units | edge | improve ticks | wait sec | toxicity | cap | mean net | CI | win | lift vs edge0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edge_02c_imp_1t_wait_300s_tox_basic_cap_50 | 3 | 8 | 0.30 | 2.00c | 1 | 300 | basic | $50 | -0.90c | [-3.52c, 0.58c] | 66.67% | 0.07c |
| edge_02c_imp_1t_wait_300s_tox_none_cap_50 | 3 | 7 | 0.27 | 2.00c | 1 | 300 | none | $50 | -0.91c | [-3.52c, 0.58c] | 66.67% | 0.06c |
| edge_01c_imp_1t_wait_300s_tox_none_cap_50 | 3 | 7 | 0.27 | 1.00c | 1 | 300 | none | $50 | -0.94c | [-3.57c, 0.54c] | 66.67% | 0.03c |
| edge_01c_imp_1t_wait_300s_tox_basic_cap_50 | 3 | 8 | 0.29 | 1.00c | 1 | 300 | basic | $50 | -0.94c | [-3.57c, 0.54c] | 66.67% | 0.03c |
| edge_00c_imp_1t_wait_300s_tox_none_cap_50 | 3 | 7 | 0.27 | 0.00c | 1 | 300 | none | $50 | -0.96c | [-3.57c, 0.50c] | 66.67% | 0.00c |
| edge_00c_imp_1t_wait_300s_tox_basic_cap_50 | 3 | 8 | 0.29 | 0.00c | 1 | 300 | basic | $50 | -0.97c | [-3.57c, 0.50c] | 66.67% | 0.00c |
| edge_02c_imp_0t_wait_30s_tox_none_cap_50 | 3 | 8 | 0.30 | 2.00c | 0 | 30 | none | $50 | -0.90c | [-3.58c, 0.59c] | 66.67% | 0.05c |
| edge_02c_imp_0t_wait_30s_tox_basic_cap_50 | 3 | 8 | 0.30 | 2.00c | 0 | 30 | basic | $50 | -0.90c | [-3.58c, 0.59c] | 66.67% | 0.05c |

Discovery rows are included only to catch obvious shape breaks. The OOS rows decide whether the exploratory queue replay is worth reopening.

## Decision

This run was useful, but it remains subordinate to the calibration gate. The queue proxy does **not** revive OD as a standalone strategy: the best row is a 0c-edge structural quote, OD richness lowers the result versus that row, and the ROC lower bound is still negative after comparing against the MM structural benchmark. The next real unlock is still fair-value calibration: HAR-RV/Kronos or another causal forward-vol model has to make the primitive rich-short EV lower-CI positive on enough independent markets. Queue replay is worth refining only after that, or if we explicitly fold OD richness into the MM execution layer as a weak quote-selection feature.

## Outputs

- Summary CSV: `data/analysis/csv_outputs/options_delta/od_v4_queue_replay_summary.csv`
- Trade parquet: `data/analysis/od_v4_queue_replay_trades.parquet`
- Candidate parquet: `data/analysis/od_v4_queue_replay_candidates.parquet`
