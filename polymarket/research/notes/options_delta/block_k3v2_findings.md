# Block K3 v2 Lead-Lag Causal Findings

> **Strat:** [[strat_options_delta]] (Options-Delta). Sibling: [[strat_market_making]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

Generated: 2026-05-31T01:11:51Z

## Headline

1s lead-lag does not clear latency: cross-corr peaks at 0s and HY peaks at 1s. The causal basis screen is diagnostic only because the primary latency gate fails.

Pooled A0b + A0c crypto-roll in-sample has 24 in-window 4h contracts (BTC, ETH, SOL) and 296,745 1-second rows. Cross-correlation peaks at **+0s** (positive means Binance spot returns lead Polymarket logit-mid changes) with corr 0.071 and n=296,721. Window bootstrap CI for peak lead is [0, 0] seconds; corr-at-peak CI is [0.059, 0.101].

Hayashi-Yoshida on asynchronous spot-return intervals vs raw Polymarket logit-mid intervals peaks at **+1s** with HY corr 0.156; bootstrap peak-lead CI is [1, 1] seconds.

Message timestamp vs local receive delta is p50 -26ms / p95 14ms / p99 62ms; the small negative median indicates clock skew, not negative physical latency. This covers capture timestamps only, before decision, order routing, and fill. Against the pre-set 2-3s action-latency hurdle, the measured lead does not survive; with a cross-corr peak of 0s there is no positive latency budget.

## Model-Free Lead-Lag

The primary measurement uses Binance spot 1s log returns and Polymarket `UP` logit-mid 1s changes. It avoids option-model fair value entirely.

| direction | maxlag_s | n | F | p |
| --- | --- | --- | --- | --- |
| Binance spot -> Polymarket logit-mid | 10 | 296,481 | 378.31 | <1e-12 |
| Polymarket logit-mid -> Binance spot | 10 | 296,481 | 8.36 | <1e-12 |

The Granger test is pooled OLS at 1s with 10 one-second lags. The Binance-to-Polymarket direction is the one relevant for quote skew; the reverse direction is included as a sanity check for feedback/common-timestamp effects.

## Causal Fair Value And Executable Screen

The causal fair value is `N(d2)` with Binance spot, Binance window-open strike, and EWMA realized volatility known up to time t only (half-life 1800s). This removes the v1 full-window realized-vol lookahead. Because the model-free lead-lag fails the latency gate, this section is a diagnostic basis/source screen rather than a tradable convergence-alpha result. It still uses Binance as the hedge/reference venue, so static basis is not treated as alpha.

At current quotes, 37.84% of rows are positive after taker fee and 25.26% clear a 1c buffer. With a 3s capture->decide->order/fill latency simulation, 37.94% remain positive and 25.36% clear 1c. Max latency-adjusted edge is 25.77c.

Latency-adjusted positive-edge runs: count 3687, median 4.0s, p90 48.0s, p95 94.7s, max 6639.0s. For `>1c` runs: count 2614, median 4.0s, p90 43.0s, p95 97.0s, max 6639.0s.

Pooled causal median basis is 0.54c and p95 absolute causal basis is 14.57c. Large static basis should be treated as model/source error first; the quote-skew feed should use the demeaned dynamic logit gap. P95 absolute dynamic K2/K-PEG skew is 1.869 logit units, or 10.67c in probability space around the observed mid.

| asset | rows | markets | median_basis | p95_abs_basis | latency_gt_1c | max_latency_edge |
| --- | --- | --- | --- | --- | --- | --- |
| BTC | 124,547 | 10 | 1.02c | 15.09c | 39.25% | 25.77c |
| ETH | 86,098 | 7 | 0.78c | 11.81c | 24.39% | 21.78c |
| SOL | 86,100 | 7 | -0.03c | 17.60c | 6.24% | 20.03c |

Top latency-adjusted markets:

| asset | market | max_edge | gt_1c_rows | median_basis | p95_abs_basis |
| --- | --- | --- | --- | --- | --- |
| BTC | btc-updown-4h-1780056000 | 25.77c | 60.06% | 5.64c | 25.84c |
| ETH | eth-updown-4h-1780056000 | 21.78c | 37.87% | 2.70c | 14.08c |
| ETH | eth-updown-4h-1780041600 | 20.16c | 88.34% | -9.80c | 13.57c |
| SOL | sol-updown-4h-1780113600 | 20.03c | 3.67% | -9.44c | 20.77c |
| BTC | btc-updown-4h-1780113600 | 19.94c | 44.10% | 3.32c | 8.96c |
| BTC | btc-updown-4h-1780041600 | 19.30c | 92.31% | -13.98c | 19.34c |
| ETH | eth-updown-4h-1780113600 | 16.63c | 9.68% | 1.03c | 8.09c |
| BTC | btc-updown-4h-1779912000 | 12.79c | 34.53% | 0.46c | 6.95c |

## Chainlink-vs-Binance Source Basis

Polymarket resolves these markets from Chainlink Data Streams, while this hedge/fair screen uses Binance. Gamma market metadata confirms the Chainlink stream resolution source for the captured crypto up/down contracts. Public historical Chainlink stream ticks were not available through an unauthenticated endpoint in this run, so this pass separates source risk by resolved direction agreement and Binance settlement margin rather than counting it as alpha.

Resolved Chainlink direction disagreed with Binance open-to-close direction on 1/24 windows. Median absolute Binance settlement margin was 22.4bp; minimum was 0.2bp. Near-zero margins are the dangerous source-basis cases because a small Chainlink-vs-Binance gap can flip settlement.

| asset | market | binance_dir | chainlink_dir | disagree | abs_binance_margin_bp |
| --- | --- | --- | --- | --- | --- |
| BTC | btc-updown-4h-1779912000 | down | down | no | 82.5 |
| BTC | btc-updown-4h-1779926400 | down | down | no | 161.3 |
| BTC | btc-updown-4h-1779940800 | up | up | no | 32.7 |
| BTC | btc-updown-4h-1780041600 | down | down | no | 5.5 |
| BTC | btc-updown-4h-1780056000 | up | up | no | 43.5 |
| BTC | btc-updown-4h-1780070400 | down | down | no | 39.1 |
| BTC | btc-updown-4h-1780084800 | down | down | no | 8.1 |
| BTC | btc-updown-4h-1780099200 | up | up | no | 8.6 |
| BTC | btc-updown-4h-1780113600 | up | up | no | 3.3 |
| BTC | btc-updown-4h-1780128000 | up | up | no | 7.3 |
| ETH | eth-updown-4h-1780041600 | down | up | yes | 0.2 |
| ETH | eth-updown-4h-1780056000 | up | up | no | 119.0 |
| ETH | eth-updown-4h-1780070400 | down | down | no | 72.4 |
| ETH | eth-updown-4h-1780084800 | down | down | no | 12.0 |
| ETH | eth-updown-4h-1780099200 | up | up | no | 11.4 |
| ETH | eth-updown-4h-1780113600 | down | down | no | 2.5 |
| ETH | eth-updown-4h-1780128000 | up | up | no | 5.4 |
| SOL | sol-updown-4h-1780041600 | down | down | no | 57.3 |
| SOL | sol-updown-4h-1780056000 | up | up | no | 99.8 |
| SOL | sol-updown-4h-1780070400 | down | down | no | 64.4 |
| SOL | sol-updown-4h-1780084800 | down | down | no | 2.4 |
| SOL | sol-updown-4h-1780099200 | up | up | no | 76.5 |
| SOL | sol-updown-4h-1780113600 | down | down | no | 36.4 |
| SOL | sol-updown-4h-1780128000 | down | down | no | 4.9 |

## Output

- CSV panel: `data/analysis/csv_outputs/options_delta/k3v2_leadlag_causal.csv`
- Repro script: `scripts/dali_block_k3v2_leadlag_causal.py`
