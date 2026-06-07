# Block K3 v3h Hedged Dynamic-Basis Findings

> **Strat:** [[strat_options_delta]] (Options-Delta). Sibling: [[strat_market_making]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

Generated: 2026-06-06T23:08:35Z

## Headline

No strict-source hedged convergence bucket clears zero after costs. Best strict candidate is `far_absz_ge1|late_lt30m` at latency 1s with mean hedged PnL -1.93c CI [-0.0273, -0.0110].

This re-points K3 away from the dead 0-1s naked lead-lag race and tests a two-legged convergence trade: buy the cheap Polymarket leg when the **causal, demeaned dynamic logit gap** is outside an entry band, delta-hedge on Binance, and exit when the gap converges or before the near-expiry spike zone. Static basis larger than 10.00c is excluded at entry and forced flat because it is model/source error first, not alpha.

Costs are included as Polymarket taker fee on entry and exit, Binance hedge turnover at 6.0bp per notional traded, and funding at 1.0bp per 8h prorated by holding time. Source filter excludes direction-disagreement windows and windows with Binance settlement margin < 10.0bp.

## Best Strict-Source Regimes

| bucket | latency | entry | exit | trades | hedged_mean | hedged_CI | naked_mean | naked_CI | hedged_win | med_hold_s | top_mkt_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| far_absz_ge1|late_lt30m | 1 | 1.50 | 0.10 | 877 | -1.94c | [-0.0268, -0.0105] | -1.64c | [-0.0234, -0.0076] | 0.11% | 2.0 | 16.65% |
| far_absz_ge1|late_lt30m | 1 | 1.50 | 0.25 | 879 | -1.93c | [-0.0273, -0.0110] | -1.63c | [-0.0239, -0.0080] | 0.11% | 2.0 | 16.72% |
| far_absz_ge1|late_lt30m | 2 | 1.50 | 0.10 | 598 | -2.03c | [-0.0274, -0.0115] | -1.67c | [-0.0246, -0.0085] | 0.00% | 3.0 | 16.89% |
| far_absz_ge1|late_lt30m | 1 | 1.50 | 0.50 | 881 | -1.97c | [-0.0279, -0.0117] | -1.64c | [-0.0237, -0.0088] | 0.11% | 2.0 | 16.80% |
| far_absz_ge1|late_lt30m | 1 | 1.00 | 0.10 | 1478 | -1.97c | [-0.0283, -0.0116] | -1.45c | [-0.0212, -0.0086] | 0.07% | 2.0 | 12.25% |
| far_absz_ge1|late_lt30m | 2 | 1.50 | 0.25 | 600 | -2.05c | [-0.0287, -0.0102] | -1.69c | [-0.0254, -0.0081] | 0.00% | 3.0 | 17.00% |
| far_absz_ge1|late_lt30m | 1 | 1.00 | 0.25 | 1483 | -1.99c | [-0.0288, -0.0123] | -1.47c | [-0.0206, -0.0087] | 0.07% | 2.0 | 12.34% |
| far_absz_ge1|late_lt30m | 1 | 2.00 | 0.50 | 656 | -2.03c | [-0.0292, -0.0107] | -1.80c | [-0.0266, -0.0095] | 0.30% | 2.0 | 15.09% |
| far_absz_ge1|late_lt30m | 1 | 1.00 | 0.50 | 1490 | -2.03c | [-0.0296, -0.0122] | -1.48c | [-0.0207, -0.0083] | 0.07% | 2.0 | 12.35% |
| far_absz_ge1|late_lt30m | 1 | 2.00 | 0.10 | 654 | -2.05c | [-0.0298, -0.0109] | -1.81c | [-0.0267, -0.0092] | 0.31% | 2.0 | 14.98% |

## Latency Robustness

Same selected bucket and bands across action latencies. This is the robustness check for convergence versus a naked one-leg trade.

For the selected regime, hedged PnL changes from -1.93c at 1s to -2.23c at 10s (-0.30c drift), while naked changes from -1.63c to -1.47c (0.16c drift). The hedged version does not show the hoped-for slower latency decay.

| latency_s | trades | hedged_mean | hedged_CI | naked_mean | naked_CI | med_hold_s | p95_hold_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 879 | -1.93c | [-0.0273, -0.0110] | -1.63c | [-0.0239, -0.0080] | 2.0 | 2.0 |
| 2 | 600 | -2.05c | [-0.0287, -0.0102] | -1.69c | [-0.0254, -0.0081] | 3.0 | 3.0 |
| 5 | 317 | -2.09c | [-0.0307, -0.0124] | -1.58c | [-0.0230, -0.0084] | 6.0 | 6.0 |
| 10 | 186 | -2.23c | [-0.0332, -0.0133] | -1.47c | [-0.0206, -0.0086] | 11.0 | 27.0 |

## Source-Basis Filter

The strict source filter removes 10 / 24 windows: any Chainlink-vs-Binance direction disagreement or Binance settlement margin below 10.0bp. This is the hard risk filter; those windows are near-pin/source-basis cases and are not counted as clean alpha.

| sample | trades | markets | hedged_mean | hedged_CI | naked_mean | hedged_win |
| --- | --- | --- | --- | --- | --- | --- |
| all_windows | 59294 | 21 | -7.82c | [-0.1203, -0.0420] | -3.13c | 0.24% |
| strict_source | 45260 | 14 | -3.71c | [-0.0655, -0.0221] | -2.20c | 0.19% |

## Tail Concentration

Across strict-source trades, top bucket `far_absz_ge1|late_lt30m` contributes 93.34%. Top market `eth-updown-4h-1780056000` contributes 12.65%. Within the selected regime, top market `eth-updown-4h-1780056000` contributes 16.72%.

## Method

- Signal: `dynamic_logit_gap = (pm_logit - rv_physical_prob_logit) - causal_static_logit_gap`, where the static gap is an EWMA using only prior rows in the same market.
- RV physical-probability value: European digital `P=N(z)`, `z=ln(S/K)/(sigma*sqrt(tau))`, with Binance proxy spot, window-open strike, and causal EWMA vol. This is a physical forecast probability, not external option-IV fair.
- Trade direction: negative dynamic gap buys `UP`; positive dynamic gap buys `DOWN`.
- Hedge: 1 binary share is hedged every second with Binance notional from digital delta; `UP` uses short delta, `DOWN` uses long delta.
- Flatten: exit on convergence to the exit band, large static basis, or before `abs(z)<0.25` with tau <= 900s.

## Outputs

- Extended row panel: `data/analysis/csv_outputs/options_delta/k3v2_leadlag_causal_hedged_ext.csv`
- Trade ledger: `data/analysis/csv_outputs/options_delta/k3v3h_hedged_basis_trades.csv`
- Feature cache: `data/analysis/cache/k3v3h_panel_features.parquet`
- Base cache reused: `data/analysis/cache/k3v2_1s_panel_base.parquet`
- Repro script: `scripts/dali_block_k3v3h_hedged_basis.py`
