---
tags: [dali, block-a16, binary-bet, results]
---
> Hub: [[COWORK]]


> Table terms: [[polymarket_table_dictionary]]

# Block A1.6 Binary-Bet Findings

## Headline

A1.6 enforces one open position per market and retests TOB imbalance as a binary-direction timing signal. 0 of 225 market-variant-horizon cells crossed zero with bootstrap CI lower bound above zero; 8 cells had positive mean PnL without clearing that robustness bar. Best cell: `a0b:2364426` / `signal_top_decile` / `fixed_900s` at 1823.7 bps on 8 non-overlapping trades, CI [-1529.2 bps, 4847.0 bps]. The A14f BTC-4h winner does not broadly replicate under non-overlap: 2 of 3 BTC-4h windows have any positive mean row.

## Universe

Binary market universe is selected from `block_a1_results.csv` with families `['crypto_4h_up_down', 'daily_crypto_up_down', 'sports_game_lines', 'sports_neg_risk_outright']` and `n_classifiable >= 30`. Fractional horizons are emitted only when `end_date` is present in the latest `data/markets/markets_*.parquet`. Fixed horizons use `end_date` as the resolution backstop when available, and otherwise use `market_resolved_at` from the feature table when the live capture observed resolution.

| market | slug | family | fractional? | end date |
| --- | --- | --- | --- | --- |
| a0b:2364426 | btc-updown-4h-1779912000 | crypto_4h_up_down | fixed only |  |
| a0b:2366225 | btc-updown-4h-1779926400 | crypto_4h_up_down | fixed only |  |
| a0b:2367777 | btc-updown-4h-1779940800 | crypto_4h_up_down | fixed only |  |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | daily_crypto_up_down | fixed only |  |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | daily_crypto_up_down | fixed only |  |
| a0:558934 | will-spain-win-the-2026-fifa-world-cup-963 | sports_game_lines | yes | 2026-07-20T00:00:00 |
| a0:558936 | will-france-win-the-2026-fifa-world-cup-924 | sports_game_lines | yes | 2026-07-20T00:00:00 |
| a0b:2327929 | nba-okc-sas-2026-05-28 | sports_game_lines | fixed only |  |
| a0b:566136 | will-psg-win-the-202526-champions-league | sports_neg_risk_outright | yes | 2026-05-31T00:00:00 |

## Per-Market Verdict

| market | slug | family | best variant | horizon | trades | mean | CI | win | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0:558934 | will-spain-win-the-2026-fifa-world-cup-963 | sports_game_lines | signal_regime_sustained_1s | frac_25pct | 4 | -106.4 bps | [n/a, n/a] | 0.0% | negative |
| a0:558936 | will-france-win-the-2026-fifa-world-cup-924 | sports_game_lines | signal_top_decile | frac_25pct | 3 | -458.2 bps | [n/a, n/a] | 0.0% | negative |
| a0b:2327929 | nba-okc-sas-2026-05-28 | sports_game_lines | signal_regime_sustained_1s | fixed_1800s | 7 | -543.3 bps | [-609.7 bps, -443.6 bps] | 0.0% | negative |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | daily_crypto_up_down | signal_regime_sustained_1s | fixed_1800s | 21 | -933.9 bps | [-1829.5 bps, 100.7 bps] | 33.3% | negative |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | daily_crypto_up_down | signal_regime_sustained_15s | fixed_900s | 28 | -153.8 bps | [-350.4 bps, 76.9 bps] | 21.4% | negative |
| a0b:2364426 | btc-updown-4h-1779912000 | crypto_4h_up_down | signal_top_decile | fixed_900s | 8 | 1823.7 bps | [-1529.2 bps, 4847.0 bps] | 62.5% | positive but not robust |
| a0b:2366225 | btc-updown-4h-1779926400 | crypto_4h_up_down | signal_regime_sustained_30s | fixed_1800s | 5 | 1388.9 bps | [-1211.2 bps, 3873.0 bps] | 40.0% | positive but not robust |
| a0b:2367777 | btc-updown-4h-1779940800 | crypto_4h_up_down | signal_top_decile | fixed_900s | 15 | -497.1 bps | [-2292.8 bps, 1940.7 bps] | 46.7% | negative |
| a0b:566136 | will-psg-win-the-202526-champions-league | sports_neg_risk_outright | signal_regime_sustained_5s | fixed_60s | 194 | -425.2 bps | [-425.2 bps, -425.2 bps] | 0.0% | negative |

## Cross-Market Pattern Read

The fixed-300s BTC-4h A14f winner is the key replication check. Under non-overlap, top-decile fixed-300s is negative on all three BTC 4h windows, including the original A14f winner. One other BTC 4h window has a positive longer-horizon regime row, but it does not clear the CI bar. This table compares top-decile fixed-300s against each market's best non-overlap row.

| market | slug | top-decile 300s | best variant | best horizon | trades | best mean | CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a0b:2364426 | btc-updown-4h-1779912000 | -1967.7 bps / 17 trades | signal_top_decile | fixed_900s | 8 | 1823.7 bps | [-1529.2 bps, 4847.0 bps] |
| a0b:2366225 | btc-updown-4h-1779926400 | -811.0 bps / 38 trades | signal_regime_sustained_30s | fixed_1800s | 5 | 1388.9 bps | [-1211.2 bps, 3873.0 bps] |
| a0b:2367777 | btc-updown-4h-1779940800 | -1811.1 bps / 34 trades | signal_top_decile | fixed_900s | 15 | -497.1 bps | [-2292.8 bps, 1940.7 bps] |

Family-level read:

| family | markets | robust+ | best market | variant | horizon | trades | mean | CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_4h_up_down | 3 | 0 | a0b:2364426 | signal_top_decile | fixed_900s | 8 | 1823.7 bps | [-1529.2 bps, 4847.0 bps] |
| daily_crypto_up_down | 2 | 0 | a0b:2362186 | signal_regime_sustained_15s | fixed_900s | 28 | -153.8 bps | [-350.4 bps, 76.9 bps] |
| sports_game_lines | 3 | 0 | a0:558934 | signal_regime_sustained_1s | frac_25pct | 4 | -106.4 bps | [n/a, n/a] |
| sports_neg_risk_outright | 1 | 0 | a0b:566136 | signal_regime_sustained_5s | fixed_60s | 194 | -425.2 bps | [-425.2 bps, -425.2 bps] |

Daily binaries and sports do not show a robust positive under non-overlap in this capture. The BTC-4h result remains a single-window clue rather than a repeated binary family effect.

## Regime-Filter Comparison

Lipton-style sustained imbalance regimes reduce noisy entry frequency, but in this capture they do not beat the top-decile entry in a robust way.

| variant | positive | robust+ | mean pnl | median pnl | trades | avg trades/hr |
| --- | --- | --- | --- | --- | --- | --- |
| signal_regime_sustained_1s | 3/45 | 0/45 | -561.5 bps | -518.9 bps | 3,273 | 5.78 |
| signal_top_decile | 2/45 | 0/45 | -615.5 bps | -437.8 bps | 2,108 | 3.96 |
| signal_regime_sustained_5s | 2/45 | 0/45 | -637.8 bps | -437.8 bps | 2,974 | 5.06 |
| signal_regime_sustained_30s | 1/45 | 0/45 | -695.0 bps | -437.8 bps | 2,256 | 3.31 |
| signal_regime_sustained_15s | 0/45 | 0/45 | -735.0 bps | -437.8 bps | 2,559 | 3.99 |

## Trades-Per-Hour

This table shows the highest deployment rates after non-overlap, not overlapping signal counts.

| market | variant | horizon | trades | trades/hr | mean hold | mean pnl |
| --- | --- | --- | --- | --- | --- | --- |
| a0b:2362186 | signal_regime_sustained_1s | fixed_60s | 350 | 29.17 | 60.0s | -285.6 bps |
| a0b:2362124 | signal_regime_sustained_1s | fixed_60s | 347 | 28.92 | 60.0s | -1632.2 bps |
| a0:558934 | signal_regime_sustained_1s | fixed_60s | 547 | 22.80 | 60.0s | -137.7 bps |
| a0:558934 | signal_regime_sustained_5s | fixed_60s | 545 | 22.71 | 60.0s | -139.3 bps |
| a0:558934 | signal_regime_sustained_15s | fixed_60s | 544 | 22.67 | 60.0s | -136.4 bps |
| a0b:2362186 | signal_regime_sustained_5s | fixed_60s | 272 | 22.67 | 60.0s | -349.4 bps |
| a0:558934 | signal_regime_sustained_30s | fixed_60s | 543 | 22.63 | 60.0s | -134.2 bps |
| a0b:2362124 | signal_regime_sustained_5s | fixed_60s | 271 | 22.58 | 60.0s | -1676.3 bps |
| a0b:2364426 | signal_regime_sustained_1s | fixed_60s | 61 | 22.51 | 60.0s | -1934.7 bps |
| a0b:2366225 | signal_regime_sustained_1s | fixed_60s | 135 | 20.12 | 60.0s | -973.4 bps |
| a0b:2362186 | signal_top_decile | fixed_60s | 241 | 20.09 | 60.0s | -242.8 bps |
| a0b:2362124 | signal_top_decile | fixed_60s | 236 | 19.67 | 60.0s | -1469.8 bps |
| a0b:2364426 | signal_regime_sustained_5s | fixed_60s | 52 | 19.19 | 60.0s | -1846.9 bps |
| a0b:2366225 | signal_top_decile | fixed_60s | 115 | 17.14 | 60.0s | -924.6 bps |
| a0:558936 | signal_regime_sustained_1s | fixed_60s | 396 | 16.50 | 60.0s | -561.7 bps |
| a0:558936 | signal_regime_sustained_5s | fixed_60s | 395 | 16.46 | 60.0s | -562.8 bps |
| a0b:566136 | signal_regime_sustained_1s | fixed_60s | 195 | 16.26 | 60.0s | -426.0 bps |
| a0:558936 | signal_regime_sustained_15s | fixed_60s | 390 | 16.25 | 60.0s | -564.0 bps |
| a0b:566136 | signal_regime_sustained_5s | fixed_60s | 194 | 16.17 | 60.0s | -425.2 bps |
| a0:558936 | signal_regime_sustained_30s | fixed_60s | 386 | 16.08 | 60.0s | -557.3 bps |

## Interpretation

Non-overlap is a much harsher and more realistic deployment constraint than A14's overlapping-position math. A positive overlapping cell can disappear if it repeatedly re-enters while one real position would still be open. Fractional horizons are limited by available `end_date` metadata; the A0b crypto/NBA markets are fixed-horizon only in this pass because they are absent from the current markets parquet, though resolved BTC 4h windows still use observed `market_resolved_at` as a fixed-horizon backstop.

Recommended next action for Justin: the binary-bet hypothesis is partial: the A14f 300s BTC-4h hint does not clear the robust CI bar under non-overlap, so A2 should capture more binary windows and pre-register non-overlap plus tight-spread capacity checks.
