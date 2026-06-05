---
tags: [dali, block-p1, rolling-rank, executable-cost, non-overlap, results]
---

# Block P1 Rolling-Rank Findings

> Table terms: [[polymarket_table_dictionary]]

## Headline

Yes on relative performance: continuous sizing beats decile gating in robust overall executable non-overlap cells with non-overlapping confidence intervals. Continuous mean exceeds gate mean in 100/100 matched robust cells. However, 0/200 robust overall executable cells have positive mean PnL, so this is not a standalone executable edge.

## Topline Cells

| cell | signal | W | H | mean/delta | CI or comparison | n cont/total | Sharpe or n gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| best continuous | tob_imbalance_level | 900 | 300 | -351.0 bps | [-379.0 bps, -315.9 bps] | 2,297 | -0.469 |
| best gate | tfi | 1800 | 300 | -607.6 bps | [-674.5 bps, -549.4 bps] | 1,491 | -0.436 |
| best cont-gate delta | weighted_mid_edge_bps | 1800 | 10 | 557.3 bps | cont -524.2 bps vs gate -1081.5 bps | 41,987 | 18,226 |

## Method

- Input: `data/analysis/block_a1_features.parquet` after complete-book and stale-book filtering (`book_staleness_seconds <= 5`), using `exchange_ts` for rank windows, non-overlap clocks, and exit alignment.
- Filtered panel: 21 markets / 42 assets, run IDs `a0, a0b`, from `2026-05-27 09:46:00.159000+00:00` to `2026-05-28 10:07:28+00:00`. This parquet currently contains A0/A0b rows; no A0c `run_id` appears in the input file.
- Signals: `ofi_l1` and `tfi` are direction-adjusted rolling sums over W; `tob_imbalance_level` is direction-adjusted current TOB imbalance; `weighted_mid_edge_bps` is direction-adjusted L1 weighted-mid edge.
- Rank transform: within each `(run_id, asset_id)`, compute a trailing W-second percentile rank with no future rows; rows need at least 30 rank-window observations.
- Mappings: continuous uses `position = 2 * rank - 1`; decile gate uses `+1` above 90th percentile, `-1` below 10th percentile, and no trade otherwise.
- PnL rows are identical across executable and mid reporting for a given mapping cell. Executable PnL crosses entry/exit touch and applies A1 `FEE_BY_CATEGORY`; mid PnL uses the direction-adjusted mid return.
- Non-overlap: one open position per market, selected by event time then rank extremeness, with hold fixed at H seconds.
- Exit quote control: future touch/mid must be observed at or before `exchange_ts + H` and no more than 5s stale.
- CIs: 200-sample 300s clock-block bootstrap for pooled rows; market-balanced rows bootstrap per-market means.

## Segment Scan

| segment | value | mapping | signal | W | H | mean | CI | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| family | daily_single_stock | continuous_rank | tfi | 1800 | 5 | -20.6 bps | [-36.5 bps, -5.0 bps] | 42 |
| family | daily_single_stock | continuous_rank | tfi | 900 | 5 | -85.4 bps | [-912.0 bps, -14.5 bps] | 42 |
| family | geopolitics_policy | continuous_rank | tob_imbalance_level | 30 | 5 | -110.1 bps | [-114.1 bps, -106.3 bps] | 15,870 |
| family | geopolitics_policy | continuous_rank | weighted_mid_edge_bps | 30 | 5 | -110.7 bps | [-115.4 bps, -106.0 bps] | 15,871 |
| spread_regime | q1_tight | continuous_rank | tfi | 30 | 5 | -111.8 bps | [-128.1 bps, -98.1 bps] | 2,225 |
| family | geopolitics_policy | continuous_rank | tob_imbalance_level | 30 | 10 | -112.4 bps | [-116.8 bps, -107.9 bps] | 8,813 |
| family | geopolitics_policy | continuous_rank | tob_imbalance_level | 300 | 30 | -112.8 bps | [-118.9 bps, -107.8 bps] | 5,933 |
| family | geopolitics_policy | continuous_rank | tob_imbalance_level | 60 | 5 | -113.0 bps | [-117.8 bps, -109.5 bps] | 19,046 |
| family | geopolitics_policy | continuous_rank | weighted_mid_edge_bps | 300 | 30 | -113.1 bps | [-118.9 bps, -107.5 bps] | 5,933 |
| family | geopolitics_policy | continuous_rank | weighted_mid_edge_bps | 30 | 10 | -113.2 bps | [-117.6 bps, -108.6 bps] | 8,813 |
| family | geopolitics_policy | continuous_rank | weighted_mid_edge_bps | 60 | 5 | -113.9 bps | [-117.9 bps, -110.3 bps] | 19,047 |
| family | geopolitics_policy | continuous_rank | tob_imbalance_level | 300 | 60 | -114.4 bps | [-120.5 bps, -108.2 bps] | 3,185 |
| family | geopolitics_policy | continuous_rank | tob_imbalance_level | 60 | 10 | -114.7 bps | [-119.0 bps, -110.8 bps] | 10,436 |
| family | geopolitics_policy | continuous_rank | tob_imbalance_level | 300 | 5 | -115.0 bps | [-121.0 bps, -109.2 bps] | 27,173 |

## Per-Market Best Rows

| mapping | market | slug | signal | W | H | mean | CI | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuous_rank | a0:1295976 | will-openai-announce-earbuds-or-headphones-. | tfi | 900 | 5 | -3.7 bps | [-6.4 bps, -1.0 bps] | 68 |
| continuous_rank | a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in-i. | tob_imbalance_level | 60 | 5 | -12.8 bps | [-14.3 bps, -11.4 bps] | 1,359 |
| continuous_rank | a0:573647 | will-gpt-6-be-released | ofi_l1 | 1800 | 10 | -18.2 bps | [-20.3 bps, -16.0 bps] | 40 |
| continuous_rank | a0:665531 | metamask-fdv-above-700m-one-day-after-launc. | tfi | 1800 | 5 | -20.6 bps | [-36.5 bps, -5.0 bps] | 42 |
| decile_gate | a0:1469737 | will-mojtaba-khamenei-be-head-of-state-in-i. | tob_imbalance_level | 60 | 60 | -21.3 bps | [-23.6 bps, -19.0 bps] | 128 |
| continuous_rank | a0:1633611 | will-china-invade-taiwan-by-december-31-2027 | tfi | 900 | 30 | -46.3 bps | [-69.3 bps, -31.6 bps] | 570 |
| continuous_rank | a0b:2176262 | strait-of-hormuz-traffic-returns-to-normal-. | tob_imbalance_level | 30 | 10 | -79.9 bps | [-84.8 bps, -75.7 bps] | 1,978 |
| continuous_rank | a0:665325 | us-iran-nuclear-deal-before-2027 | tob_imbalance_level | 30 | 5 | -101.6 bps | [-107.4 bps, -95.2 bps] | 6,517 |
| continuous_rank | a0:558934 | will-spain-win-the-2026-fifa-world-cup-963 | ofi_l1 | 30 | 30 | -111.2 bps | [-166.0 bps, -80.2 bps] | 46 |
| continuous_rank | a0:1090496 | nato-x-russia-military-clash-by-december-31. | tfi | 900 | 10 | -113.2 bps | [-154.8 bps, -82.3 bps] | 1,033 |
| continuous_rank | a0b:1971905 | strait-of-hormuz-traffic-returns-to-normal-. | tfi | 60 | 5 | -120.5 bps | [-126.8 bps, -114.4 bps] | 3,187 |
| continuous_rank | a0b:566136 | will-psg-win-the-202526-champions-league | tob_imbalance_level | 30 | 30 | -122.6 bps | [-177.2 bps, -70.1 bps] | 31 |

## Row-Count Heatmap Check

| signal | mapping | min n | median n | max n | sparse cells |
| --- | --- | --- | --- | --- | --- |
| ofi_l1 | continuous_rank | 1,300 | 17,695 | 86,133 | 0/25 |
| ofi_l1 | decile_gate | 1,192 | 8,957 | 37,262 | 0/25 |
| tfi | continuous_rank | 799 | 9,891 | 83,631 | 0/25 |
| tfi | decile_gate | 721 | 5,386 | 43,110 | 0/25 |
| tob_imbalance_level | continuous_rank | 1,283 | 16,992 | 85,960 | 0/25 |
| tob_imbalance_level | decile_gate | 1,134 | 8,756 | 34,397 | 0/25 |
| weighted_mid_edge_bps | continuous_rank | 1,283 | 16,991 | 85,960 | 0/25 |
| weighted_mid_edge_bps | decile_gate | 1,138 | 8,799 | 35,283 | 0/25 |

The detailed heatmap-ready counts live in `data/analysis/csv_outputs/dali/p1_rollingrank_row_count_heatmap.csv`. Sparse cells are retained in CSVs but are not used for the headline.

## Outputs

- `data/analysis/csv_outputs/dali/p1_rollingrank_surface.csv`: pooled, market-balanced, family/regime/time/clock segment surface.
- `data/analysis/csv_outputs/dali/p1_rollingrank_by_market.csv`: market-level surface for the same signal/window/horizon/mapping grid.
- `data/analysis/csv_outputs/dali/p1_rollingrank_row_count_heatmap.csv`: overall executable row-count heatmap by signal/window/horizon/mapping.

## Interpretation

Continuous rank sizing changes the exposure profile, but the P1 executable-cost test is still governed by spread crossing and short-horizon quote movement. Treat any positive sparse or segment-only cell as a diagnostic lead, not a deployable result, unless it survives the market-balanced rows and a fresh out-of-sample capture.
