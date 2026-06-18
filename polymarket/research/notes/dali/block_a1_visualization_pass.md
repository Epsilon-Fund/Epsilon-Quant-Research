---
tags: [dali, block-a1, viz, results]
---
> Hub: [[COWORK]]


## Summary

- Scope: Block A1 Visualization Pass in the Dali research lineage area.
- Existing takeaway/status: Rendering pass over existing A1/A1.1 outputs, not a new analysis run. It consolidates family hit-rate surfaces, market-horizon matrices, signal-cost waterfalls, and component comparisons for reviewer scanning.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
# Block A1 Visualization Pass

This is a rendering pass over existing A1/A1.1 outputs, not a new analysis run. It consolidates the live OFI read into charts a reviewer can scan quickly: family surfaces, market-horizon matrices, cost decomposition, and component comparison. The cumulative wall-clock bps timeline requested for this pass is skipped because the allowed CSV inputs do not contain timestamps or per-row returns; that chart requires `block_a1_features.parquet` or a precomputed time-series output.

## Family Hit-Rate Surfaces

These 3D charts stratify the A1.1 absolute-OFI decile surface by family bucket. Read them as shape diagnostics: high deciles should outperform low deciles if OFI magnitude is behaving coherently. Crypto combines `daily_crypto_up_down` and `crypto_4h_up_down`; sports combines game-line and neg-risk outright families.

![](../data/analysis/block_a1_viz/family_surface_crypto_hit_rate_3d.png)

![](../data/analysis/block_a1_viz/family_surface_sports_hit_rate_3d.png)

![](../data/analysis/block_a1_viz/family_surface_geopolitics_hit_rate_3d.png)

![](../data/analysis/block_a1_viz/family_surface_ai_hit_rate_3d.png)

## Market-Horizon Matrices

These use public A1 per-market metrics for the top markets by classifiable prints. Cell text is `n` for OFI top-decile rows and `ci` for hit-rate CI width in percentage points. Blank or `n/a` cells are usually suppressed public metrics, not missing raw data.

![697](../data/analysis/block_a1_viz/market_horizon_hit_rate_heatmap.png)

![](../data/analysis/block_a1_viz/market_horizon_directional_return_heatmap.png)

## Signal-Cost Waterfall

This decomposes the 5s top public alpha candidates into mid-return alpha, fee, half-spread, latency, and final `edge_after_cost_bps`. It is still a diagnostic overlay, not executable PnL, but it makes visually clear where the mid-alpha gets consumed.

![](../data/analysis/block_a1_viz/signal_cost_waterfall_top5_5s.png)

## Component Comparison

This compares top-decile hit rate across three A1.1 components: canonical OFI, TOB imbalance level, and top-5 depth pressure. Important: `tob_imbalance_level` is a standing-book imbalance feature, not an OFI flow measure.

![](../data/analysis/block_a1_viz/component_comparison_hit_rate.png)

## Files Rendered

- Family surfaces: `../data/analysis/block_a1_viz/family_surface_*_hit_rate_3d.png`
- Market matrices: `../data/analysis/block_a1_viz/market_horizon_*_heatmap.png`
- Cost waterfall: `../data/analysis/block_a1_viz/signal_cost_waterfall_top5_5s.png`
- Component comparison: `../data/analysis/block_a1_viz/component_comparison_hit_rate.png`

Use this note with `notes/dali/block_a1_results.md`, `notes/dali/block_a1_methodology_audit.md`, and `notes/dali/block_a11_plan_and_diagnostics.md` for cross-thread review.
