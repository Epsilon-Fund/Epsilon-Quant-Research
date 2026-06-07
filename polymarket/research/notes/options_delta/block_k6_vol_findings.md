# Block K6 Vol Gap Diagnostic

> **Strat:** [[strat_options_delta]] (Options-Delta). Sibling: [[strat_market_making]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Headline

No strict-source (|z|, tau) bucket clears zero after Polymarket fee plus banded Binance hedge turnover; the vol branch does not pass the falsifier on this IS panel.

Clean-source rows imply PM midpoint-implied vol minus causal EWMA of **3.7 vol pts** on average; the ex-post remaining captured-path diagnostic is **4.0 vol pts**. The sign is useful diagnostically, but the tradable test is the banded delta-hedged simulation below.

Best strict bucket/config by lower CI is `far_absz_ge1|late_lt30m` at latency 5s, entry gap 20.0 vol pts, band 10.00c: mean net -9.39c, CI [-0.1905, -0.0113].

## Inversion Method

For each row I invert the PM midpoint through the European digital model `P_up = N(log(S/K)/(sigma*sqrt(tau)))`, with `K` from the Binance window-open reference in the K3 panel. This `pm_mid_implied_vol_annualized` is a diagnostic representation of the PM price, not external option-IV fair. A positive finite implied vol exists only when the PM probability is on the same side of 50% as Binance moneyness. Rows that violate that are marked `no_positive_solution`; they are not forced into a bogus sigma.

Implied-vol validity:

| status | share |
| --- | --- |
| valid | 95.97% |
| no_positive_solution | 1.88% |
| implied_infinite_at_half | 1.85% |
| zero_moneyness_price_off_half | 0.23% |
| atm_underdetermined | 0.07% |

## Causal Vol Gap By Bucket

This table uses the clean source sample: strict Chainlink/Pyth-vs-Binance settlement filter, no large static 10c basis rows, and no toxic near-strike/near-expiry rows. `Remaining RV` and `full-window RV` columns are diagnostic/lookahead only.

| bucket | rows | valid IV | PM mid-IV | EWMA | PM-EWMA | PM-EWMA CI | remaining RV | PM-rem RV | full RV | PM-full RV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| far_absz_ge1|early_gt2h | 3743 | 100.00% | 40.6 vol pts | 42.3 vol pts | -1.7 vol pts | [-0.0513, 0.0852] | 40.9 vol pts | -0.3 vol pts | 37.2 vol pts | 3.5 vol pts |
| far_absz_ge1|late_lt30m | 23138 | 100.00% | 69.3 vol pts | 45.3 vol pts | 24.1 vol pts | [0.1385, 0.3418] | 38.5 vol pts | 30.2 vol pts | 47.2 vol pts | 22.1 vol pts |
| far_absz_ge1|mid_30m_2h | 33175 | 100.00% | 43.5 vol pts | 40.0 vol pts | 3.4 vol pts | [-0.0057, 0.0491] | 41.0 vol pts | 2.5 vol pts | 44.5 vol pts | -1.0 vol pts |
| mid_absz_0.25_1|early_gt2h | 45533 | 99.99% | 38.9 vol pts | 41.2 vol pts | -2.2 vol pts | [-0.0588, 0.0349] | 41.9 vol pts | -2.9 vol pts | 47.0 vol pts | -8.0 vol pts |
| mid_absz_0.25_1|late_lt30m | 1514 | 100.00% | 31.0 vol pts | 31.9 vol pts | -0.9 vol pts | [-0.0648, 0.0158] | 26.2 vol pts | 4.8 vol pts | 38.9 vol pts | -8.0 vol pts |
| mid_absz_0.25_1|mid_30m_2h | 29689 | 100.00% | 40.6 vol pts | 42.1 vol pts | -1.5 vol pts | [-0.0486, 0.0056] | 47.8 vol pts | -7.3 vol pts | 49.6 vol pts | -9.0 vol pts |
| near_absz_lt0.25|early_gt2h | 28539 | 89.35% | 49.3 vol pts | 44.2 vol pts | 5.0 vol pts | [0.0084, 0.6192] | 39.2 vol pts | 10.0 vol pts | 44.1 vol pts | 5.2 vol pts |
| near_absz_lt0.25|late_lt30m | 81 | 100.00% | 13.8 vol pts | 25.6 vol pts | -11.9 vol pts | [-0.1185, -0.1185] | 17.9 vol pts | -4.2 vol pts | 45.1 vol pts | -31.3 vol pts |
| near_absz_lt0.25|mid_30m_2h | 8855 | 88.80% | 39.4 vol pts | 41.1 vol pts | -1.8 vol pts | [-0.0214, 0.2740] | 40.9 vol pts | -1.5 vol pts | 45.7 vol pts | -6.3 vol pts |

## Gamma-Scalp Backtest

Rules: for each market/bucket/config, take the first eligible entry on the side implied by the causal PM-mid-IV-minus-EWMA gap; buy the OTM/positive-vega side when PM midpoint-implied IV is cheap, or the ITM/negative-vega proxy when PM midpoint-implied IV is rich. Hold to resolution, delta-hedge on Binance using the causal K3 digital delta, and rebalance only when target hedge notional moves by the band. Costs are the PM taker fee at entry plus Binance hedge turnover at 6.0bp. Entries exclude invalid PM-mid-IV, large static basis, and toxic near-expiry rows. Bucket entries are diagnostic and may overlap across buckets within the same market.

| bucket | lat | entry gap | band | trades | net | net CI | unhedged | PM fee | hedge cost | win | median hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| far_absz_ge1|late_lt30m | 5 | 20.0 vol pts | 10.00c | 12 | -9.39c | [-0.1905, -0.0113] | 0.72c | 0.05c | 9.56c | 50.00% | 1452 |
| far_absz_ge1|late_lt30m | 3 | 20.0 vol pts | 10.00c | 12 | -9.44c | [-0.1908, -0.0115] | 0.73c | 0.05c | 9.58c | 50.00% | 1454 |
| far_absz_ge1|late_lt30m | 1 | 20.0 vol pts | 10.00c | 12 | -9.51c | [-0.1917, -0.0123] | 0.72c | 0.05c | 9.62c | 50.00% | 1456 |
| far_absz_ge1|late_lt30m | 5 | 20.0 vol pts | 5.00c | 12 | -9.55c | [-0.1932, -0.0117] | 0.72c | 0.05c | 9.72c | 50.00% | 1452 |
| far_absz_ge1|late_lt30m | 3 | 20.0 vol pts | 5.00c | 12 | -9.60c | [-0.1935, -0.0120] | 0.73c | 0.05c | 9.74c | 50.00% | 1454 |
| far_absz_ge1|late_lt30m | 5 | 20.0 vol pts | 3.00c | 12 | -9.63c | [-0.1942, -0.0120] | 0.72c | 0.05c | 9.80c | 50.00% | 1452 |
| far_absz_ge1|late_lt30m | 1 | 20.0 vol pts | 5.00c | 12 | -9.67c | [-0.1945, -0.0128] | 0.72c | 0.05c | 9.78c | 50.00% | 1456 |
| far_absz_ge1|late_lt30m | 3 | 20.0 vol pts | 3.00c | 12 | -9.68c | [-0.1945, -0.0123] | 0.73c | 0.05c | 9.82c | 50.00% | 1454 |
| far_absz_ge1|late_lt30m | 5 | 20.0 vol pts | 1.00c | 12 | -9.70c | [-0.1950, -0.0123] | 0.72c | 0.05c | 9.87c | 50.00% | 1452 |
| far_absz_ge1|late_lt30m | 3 | 20.0 vol pts | 1.00c | 12 | -9.75c | [-0.1953, -0.0125] | 0.73c | 0.05c | 9.89c | 50.00% | 1454 |

## Caveats

- The causal comparison is PM midpoint-implied IV versus trailing/EWMA vol available at time `t`; it is not an external option-IV fair.
- The remaining-window and full-window realized vol columns are explicitly ex-post diagnostics.
- The full-window RV is pulled from the earlier K3 full-window pass when present; remaining RV is computed on the captured Binance path after each row, so it can understate unobserved pre/post-capture variance.
- The trade ledger uses Chainlink/Pyth settlement when available, and the strict source filter removes direction disagreements and small Binance settlement margins.

Outputs:

- `data/analysis/csv_outputs/options_delta/k6_vol_gap.csv`
- `data/analysis/k6_vol_gap_panel.parquet`
- `data/analysis/csv_outputs/options_delta/k6_gamma_scalp_trades.csv`
