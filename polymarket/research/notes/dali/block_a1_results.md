---
tags: [dali, block-a1, results]
---
> Hub: [[COWORK]]


## Summary

- Scope: Block A1 Results in the Dali research lineage area.
- Existing takeaway/status: The material Block A1 win is sign convention: live `last_trade_price.side` is now established as token-side aggressor direction, which unblocks live OFI/TFI work. The cleanest depth-normalized OFI characterization is the pre-cost 5s top-decile aggregate, 5s/decile 10: 64.1% hit (CI [60.0%, 68.2%], n=156,638), directional return 94.9 bps. Crypto and sports carry the read; geopolitics is useful but thinner; AI/product...
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
> Table terms: [[polymarket_table_dictionary]]

# Block A1 Results

## Headline

The material Block A1 win is sign convention: live `last_trade_price.side` is now established as token-side aggressor direction, which unblocks live OFI/TFI work. The cleanest depth-normalized OFI characterization is the pre-cost 5s top-decile aggregate, 5s/decile 10: 64.1% hit (CI [60.0%, 68.2%], n=156,638), directional return 94.9 bps. Crypto and sports carry the read; geopolitics is useful but thinner; AI/product is a single thin live sample; equity index and single-stock activity are caveats only. This remains signal characterization, not a tradeable/not-tradeable decision.

## Sign Convention Resolution

- A0 + A0b inspected 8,158 `last_trade_price` events.
- 847 were classifiable from subsequent book transitions.
- `P(reported.side == BUY | inferred BUY)` was 99.9%; `P(reported.side == BUY | inferred SELL)` was 1.7%.
- A1 therefore treats live `last_trade_price.side` as token-side aggressor.

## Capture Audit Summary

- A0 totals matched the final audit: 4,326 `book`, 1,253,183 `price_change`, 50,630 `best_bid_ask`, 2,095 `last_trade_price`, 8,427 `new_market`.
- A0b totals matched the final audit: 12,212 `book`, 842,267 `price_change`, 58,660 `best_bid_ask`, 6,063 `last_trade_price`, 3,570 `new_market`, 20 `tick_size_change`, 3 `market_resolved`.
- Gap logs: A0 had 4 `disconnect_or_error` entries; A0b had 1. Flanking windows were kept.
- `market_resolved` events were honored by dropping post-resolution rows per asset.

## Depth-Normalized Aggregate

![](data/analysis/block_a1_plots/block_a1_decile_hit_rate_1s.png)
![](data/analysis/block_a1_plots/block_a1_decile_hit_rate_5s.png)
![](data/analysis/block_a1_plots/block_a1_decile_hit_rate_30s.png)
![](data/analysis/block_a1_plots/block_a1_decile_hit_rate_300s.png)

The pooled aggregate uses `OFI_scaled = OFI / mean_depth_at_touch` and bins on absolute signal magnitude rather than market identity. OFI levels are percentile labels inside each horizon, not fixed physical units: deciles 1-3 are low, 4-7 are middle, 8-9 are high, and 10 is extreme. The 5s top decile is the headline because it is the cleanest high-magnitude bin; the 30s top decile is weaker but still elevated (30s/decile 10: 57.0% hit (CI [51.4%, 63.8%], n=218,378), directional return 66.4 bps). The 300s structure is not a clean monotone OFI story: the smallest-magnitude bin is unusually strong (300s/decile 1: 70.6% hit (CI [62.5%, 80.6%], n=277,114), directional return 48.4 bps) while the top-magnitude bin is only 300s/decile 10: 51.8% hit (CI [40.2%, 61.2%], n=275,591), directional return 89.3 bps. Treat 300s as a QA caveat, not the headline.

## Horizon Surface

![](data/analysis/block_a1_plots/block_a1_ofi_surface_hit_rate_3d.png)
![](data/analysis/block_a1_plots/block_a1_ofi_surface_hit_rate_heatmap.png)
![](data/analysis/block_a1_plots/block_a1_ofi_surface_directional_return_heatmap.png)
![](data/analysis/block_a1_plots/block_a1_ofi_surface_family_share_heatmap.png)

The surface diagnostic recomputes the depth-normalized aggregate at horizons 1, 2, 3, 5, 10, 15, 30, 60, 120, and 300 seconds. The 3D plot is included because it is good for seeing the ridge shape at a glance. The heatmaps are more useful for reading exact cells because horizons are log-spaced, deciles are discrete buckets, and the 300s low-OFI behavior can otherwise look like a smooth surface even when it is a single bucket effect. Best surface cells by hit rate:

| h | decile | level | hit | dir ret | n | top family | share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | 1 | low | 70.6% | 48.4 bps | 277,114 | geopolitics_policy | 41.0% |
| 10 | 10 | extreme | 65.4% | 122.8 bps | 180,602 | daily_crypto_up_down | 51.8% |
| 5 | 10 | extreme | 64.1% | 94.9 bps | 156,638 | daily_crypto_up_down | 50.3% |
| 15 | 10 | extreme | 62.2% | 89.5 bps | 191,528 | daily_crypto_up_down | 55.1% |
| 3 | 10 | extreme | 61.3% | 72.7 bps | 144,748 | daily_crypto_up_down | 48.4% |
| 120 | 7 | middle | 59.3% | 59.3 bps | 258,829 | daily_crypto_up_down | 40.3% |
| 1 | 2 | low | 58.4% | 10.1 bps | 119,802 | daily_crypto_up_down | 47.8% |
| 120 | 9 | high | 58.2% | 134.5 bps | 259,085 | daily_crypto_up_down | 57.0% |
| 2 | 10 | extreme | 57.9% | 40.0 bps | 134,205 | crypto_4h_up_down | 49.3% |
| 60 | 9 | high | 57.3% | 132.1 bps | 241,536 | daily_crypto_up_down | 55.0% |
| 30 | 10 | extreme | 57.0% | 66.4 bps | 218,378 | daily_crypto_up_down | 51.5% |
| 300 | 7 | middle | 56.6% | 66.8 bps | 275,560 | daily_crypto_up_down | 40.4% |

## Per-Family Read

- Crypto: 4,067 trades across 5 markets; 5 primary-read markets and 5 reportable per-market 5s OFI rows; median reportable 5s hit 70.1%; mean post-cost overlay -970.3 bps.
- Sports: 2,941 trades across 4 markets; 4 primary-read markets and 0 reportable per-market 5s OFI rows; median reportable 5s hit n/a; mean post-cost overlay n/a.
- Geopolitics: 898 trades across 6 markets; 2 primary-read markets and 3 reportable per-market 5s OFI rows; median reportable 5s hit 58.5%; mean post-cost overlay -370.1 bps.
- Tech: 248 trades across 4 markets; 0 primary-read markets and 1 reportable per-market 5s OFI rows; median reportable 5s hit 58.5%; mean post-cost overlay -110.3 bps.

Per-market hit rates are diagnostic only; sparse or degenerate top-decile rows are suppressed in the public columns and kept only in raw audit columns. The CSV now has 41 reportable OFI rows and 43 suppressed OFI rows under the minimum classifiable/top-decile guard. AI/product cannot validate the Block B finding here: the live sample has only one primary-threshold-ish market and wide uncertainty. A2 should explicitly target AI/product markets rather than relying on incidental flow. Equity-index and single-stock markets are too dead in A0/A0b to report family statistics.

## Per-Market Breakdown

The table below is the 5s horizon because that is the cleanest aggregate A1 result. Blank `n/a` public hit or cost cells usually mean the row was suppressed by the reporting guard, not that raw data was missing.

| run | market | family | n class | label | 5s hit | top n | 5s dir ret | cost edge | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0b | btc-updown-4h-1779940800 | crypto_4h_up_down | 1,424 | primary_read | 70.1% | 8,363 | 108.3 bps | -788.7 bps | signal_present_pre_cost |
| a0b | bitcoin-up-or-down-on-may-28-2026 | daily_crypto_up_down | 1,096 | primary_read | 82.2% | 8,727 | 891.3 bps | -123.5 bps | signal_present_pre_cost |
| a0b | btc-updown-4h-1779926400 | crypto_4h_up_down | 990 | primary_read | 63.4% | 8,411 | 112.8 bps | -1139.5 bps | signal_present_pre_cost |
| a0b | btc-updown-4h-1779912000 | crypto_4h_up_down | 309 | primary_read | 67.1% | 1,989 | 533.5 bps | -993.7 bps | signal_present_pre_cost |
| a0b | ethereum-up-or-down-on-may-28-2026 | daily_crypto_up_down | 248 | primary_read | 70.8% | 14,005 | 268.9 bps | -1805.9 bps | signal_present_pre_cost |
| a0 | us-iran-nuclear-deal-before-2027 | geopolitics_policy | 206 | primary_read | 58.5% | 1,459 | 10.2 bps | -97.1 bps | signal_present_pre_cost |
| a0 | will-anthropic-have-the-best-ai-model-at-the-end-of-june-2 | ai_product | 153 | thin_wide_CI | 58.5% | 5,071 | 0.7 bps | -110.3 bps | signal_present_pre_cost |
| a0b | strait-of-hormuz-traffic-returns-to-normal-by-july-31 | geopolitics_policy | 131 | thin_wide_CI | 78.7% | 47 | 100.2 bps | 4.7 bps | signal_present_post_cost |
| a0 | nato-x-russia-military-clash-by-december-31-2026-244 | geopolitics_policy | 32 | thin_wide_CI | 29.1% | 399 | -79.6 bps | -1018.0 bps | absent |
| a0b | nba-okc-sas-2026-05-28 | sports_game_lines | 1,269 | primary_read | n/a | 2 | n/a | n/a | absent |
| a0 | will-spain-win-the-2026-fifa-world-cup-963 | sports_game_lines | 767 | primary_read | n/a | 2 | n/a | n/a | absent |
| a0 | will-france-win-the-2026-fifa-world-cup-924 | sports_game_lines | 705 | primary_read | n/a | 10 | n/a | n/a | absent |
| a0b | strait-of-hormuz-traffic-returns-to-normal-by-end-of-june | geopolitics_policy | 396 | primary_read | n/a | 54 | n/a | n/a | absent |
| a0b | will-psg-win-the-202526-champions-league | sports_neg_risk_outright | 200 | primary_read | n/a | 0 | n/a | n/a | absent |
| a0 | will-mojtaba-khamenei-be-head-of-state-in-iran-end-of-2026 | geopolitics_policy | 119 | thin_wide_CI | n/a | 20 | n/a | n/a | absent |
| a0 | will-google-have-the-best-ai-model-at-the-end-of-june-2026 | ai_product | 92 | thin_wide_CI | n/a | 16 | n/a | n/a | absent |
| a0 | will-china-invade-taiwan-by-december-31-2027 | geopolitics_policy | 14 | insufficient | n/a | 0 | n/a | n/a | data_thin |
| a0 | will-the-sp-500-have-the-best-performance-in-2026-545 | daily_equity_index | 3 | insufficient | n/a | 0 | n/a | n/a | data_thin |
| a0 | will-gpt-6-be-released | ai_product | 2 | insufficient | n/a | 0 | n/a | n/a | data_thin |
| a0 | will-openai-announce-earbuds-or-headphones-in-2026 | ai_product | 1 | insufficient | n/a | 2,117 | n/a | n/a | data_thin |
| a0 | metamask-fdv-above-700m-one-day-after-launch-696-977-652-2 | daily_single_stock | 1 | insufficient | n/a | 196 | n/a | n/a | data_thin |

High individual-market hit-rate rows need extra care. Public rows survived the guard; `raw only` rows are retained for audit but should not be cited as standalone evidence.

| run | h | market | n class | public hit | raw hit | top n | status | cost edge |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0b | 1 | strait-of-hormuz-traffic-returns-to-normal-by-end-of-june | 396 | 87.5% | 87.5% | 32 | yes | -41.5 bps |
| a0 | 1 | us-iran-nuclear-deal-before-2027 | 206 | 81.3% | 81.3% | 1,067 | yes | -62.9 bps |
| a0b | 1 | bitcoin-up-or-down-on-may-28-2026 | 1,096 | 71.6% | 71.6% | 5,170 | yes | -631.0 bps |
| a0b | 5 | bitcoin-up-or-down-on-may-28-2026 | 1,096 | 82.2% | 82.2% | 8,727 | yes | -123.5 bps |
| a0b | 5 | strait-of-hormuz-traffic-returns-to-normal-by-july-31 | 131 | 78.7% | 78.7% | 47 | yes | 4.7 bps |
| a0b | 5 | ethereum-up-or-down-on-may-28-2026 | 248 | 70.8% | 70.8% | 14,005 | yes | -1805.9 bps |
| a0b | 5 | btc-updown-4h-1779940800 | 1,424 | 70.1% | 70.1% | 8,363 | yes | -788.7 bps |
| a0b | 30 | strait-of-hormuz-traffic-returns-to-normal-by-end-of-june | 396 | 87.7% | 87.7% | 357 | yes | -42.7 bps |
| a0b | 30 | strait-of-hormuz-traffic-returns-to-normal-by-july-31 | 131 | 77.1% | 77.1% | 179 | yes | -41.7 bps |
| a0 | 30 | us-iran-nuclear-deal-before-2027 | 206 | 72.7% | 72.7% | 3,088 | yes | -61.4 bps |
| a0b | 300 | btc-updown-4h-1779912000 | 309 | 98.6% | 98.6% | 6,179 | yes | 2086.6 bps |
| a0b | 300 | strait-of-hormuz-traffic-returns-to-normal-by-end-of-june | 396 | 86.5% | 86.5% | 2,574 | yes | 21.1 bps |
| a0b | 300 | btc-updown-4h-1779940800 | 1,424 | 80.7% | 80.7% | 18,419 | yes | 623.3 bps |
| a0b | 300 | bitcoin-up-or-down-on-may-28-2026 | 1,096 | 79.2% | 79.2% | 37,088 | yes | 290.9 bps |
| a0 | 1 | will-mojtaba-khamenei-be-head-of-state-in-iran-end-of-2026 | 119 | n/a | 100.0% | 20 | raw only | n/a |
| a0 | 1 | metamask-fdv-above-700m-one-day-after-launch-696-977-652-2 | 1 | n/a | 98.7% | 149 | raw only | n/a |
| a0 | 1 | will-openai-announce-earbuds-or-headphones-in-2026 | 1 | n/a | 83.5% | 1,587 | raw only | n/a |
| a0 | 5 | will-france-win-the-2026-fifa-world-cup-924 | 705 | n/a | 100.0% | 10 | raw only | n/a |

Positive post-cost rows after the reporting guard are mostly longer-horizon crypto/geopolitics diagnostics, not deployment recommendations.

| run | h | market | family | hit | dir ret | cost edge | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a0b | 300 | btc-updown-4h-1779912000 | crypto_4h_up_down | 98.6% | 3613.8 bps | 2086.6 bps | signal_present_post_cost |
| a0b | 300 | btc-updown-4h-1779940800 | crypto_4h_up_down | 80.7% | 1520.4 bps | 623.3 bps | signal_present_post_cost |
| a0b | 300 | btc-updown-4h-1779926400 | crypto_4h_up_down | 69.0% | 1771.7 bps | 519.3 bps | signal_present_post_cost |
| a0b | 300 | bitcoin-up-or-down-on-may-28-2026 | daily_crypto_up_down | 79.2% | 1305.6 bps | 290.9 bps | signal_present_post_cost |
| a0b | 300 | strait-of-hormuz-traffic-returns-to-normal-by-end-of-june | geopolitics_policy | 86.5% | 167.0 bps | 21.1 bps | signal_present_post_cost |
| a0 | 30 | will-mojtaba-khamenei-be-head-of-state-in-iran-end-of-2026 | geopolitics_policy | 68.4% | 39.8 bps | 19.8 bps | signal_present_post_cost |
| a0b | 5 | strait-of-hormuz-traffic-returns-to-normal-by-july-31 | geopolitics_policy | 78.7% | 100.2 bps | 4.7 bps | signal_present_post_cost |

## Per-Market Scatter Diagnostics

These sampled scatter plots were generated during A1 and are now linked here rather than left unused. They show raw 30s OFI versus 30s forward return for the most active markets; use them to spot outliers, regime mixtures, and thin linear fits before trusting any individual hit rate.

![](data/analysis/block_a1_plots/block_a1_ofi_scatter_a0b_btc-updown-4h-1779940800.png)
![](data/analysis/block_a1_plots/block_a1_ofi_scatter_a0b_nba-okc-sas-2026-05-28.png)
![](data/analysis/block_a1_plots/block_a1_ofi_scatter_a0b_bitcoin-up-or-down-on-may-28-2026.png)
![](data/analysis/block_a1_plots/block_a1_ofi_scatter_a0b_btc-updown-4h-1779926400.png)
![](data/analysis/block_a1_plots/block_a1_ofi_scatter_a0_will-spain-win-the-2026-fifa-world-cup-963.png)
![](data/analysis/block_a1_plots/block_a1_ofi_scatter_a0_will-france-win-the-2026-fifa-world-cup-924.png)
![](data/analysis/block_a1_plots/block_a1_ofi_scatter_a0b_strait-of-hormuz-traffic-returns-to-normal-by-end-of-june.png)
![](data/analysis/block_a1_plots/block_a1_ofi_scatter_a0b_btc-updown-4h-1779912000.png)
![](data/analysis/block_a1_plots/block_a1_ofi_scatter_a0b_ethereum-up-or-down-on-may-28-2026.png)
![](data/analysis/block_a1_plots/block_a1_ofi_scatter_a0_us-iran-nuclear-deal-before-2027.png)
![](data/analysis/block_a1_plots/block_a1_ofi_scatter_a0b_will-psg-win-the-202526-champions-league.png)
![](data/analysis/block_a1_plots/block_a1_ofi_scatter_a0_will-anthropic-have-the-best-ai-model-at-the-end-of-june-2026.png)

## Maker Simulation Summary

![](data/analysis/block_a1_plots/block_a1_maker_fill_timeline.png)

The passive maker proxy found 7,042 touch-fill observations in the 5s horizon panel. Mean queue-blind maker net edge after rebate and 5s adverse selection was 245.6 bps in the raw proxy, but this is not a deployable edge estimate. The proxy follows the Domah audit pattern of using future prints as fill evidence, but differs because A1 uses live CLOB `last_trade_price` prints and current book touch, not historical wallet-address anchored copy fills. Queue position, cancel latency, and partial queue priority are not modeled.

## Cost Overlay Summary

Cost columns use the current Polymarket formula `fee = shares * fee_rate * p * (1 - p)` and normalize fee, half-spread, and latency slippage onto the same directional-return bps denominator. The previous A1 draft overstated latency slippage because pandas loaded capture timestamps at microsecond resolution while the latency offset was added as nanoseconds; this rerun converts timestamps to `datetime64[ns]` before applying the 100 ms WS latency assumption. Midas has request timeout settings but no explicit WS execution latency config. The cost overlay is a diagnostic column, not a deployment verdict.

## A2 Design Recommendations

- After reviewing this cost-QA rerun, run a 1-2 week VPS capture with the A1 wrapper/analyzer unchanged.
- Add a targeted AI/product panel with active markets selected by live trade-rate, not just quote activity.
- Keep crypto 4h up/down windows in the panel because they provide dense live trade prints and clean horizon structure.
- Keep sports game-line markets, but separate live/in-game from pre-game when possible.
- Keep geopolitics as fee-free robustness, but do not overweight it in the cross-family headline.
- Remove equity-index and single-stock names unless pre-screened live flow is materially higher.

## Caveats

- This is one 24h A0 panel plus one 12h A0b replacement panel; thin markets remain labeled, not excluded.
- `best_bid_ask` never mutates executable state; OFI is from `book` plus `price_change`.
- Live WS `last_trade_price` messages do not expose the historical CTF Exchange `taker = address(this)` artifact; all A1 math uses raw unfiltered live data.
- The PSG Champions League leg is negative-risk and should not be pooled blindly with clean binary markets.
- Bootstrap intervals use time-contiguous chunks, but the sample is still short.

Recommended next action for Justin: review the regenerated cost overlay and, if the corrected 5s signal/cost picture is acceptable, approve A2 as a 1-2 week VPS capture with a targeted AI/product sleeve and continued crypto 4h up/down coverage.
