---
title: Block K1 Maker-Economics Findings
created: 2026-06-05
status: candidate
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
  - strat_market_making
tags: [dali, block-k1, maker-economics, gate]
---

# Block K1 Maker-Economics Findings

> **Strat:** [[strat_market_making]] (Market-Making). Sibling: [[strat_options_delta]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Summary

K1 decomposes baseline maker economics by category using spread, rebate, adverse-selection, and inventory/resolution components. Crypto, Geopolitics, Sports, and Tech clear the generous baseline gate, so the note sends the branch onward to K2. It is a candidate screen, not a deployability proof.

## Headline

At least one category clears zero (Crypto, Geopolitics, Sports, Tech) under the K1 baseline maker decomposition. Under the gate rule, proceed to K2.

Canonical read: zero directional skew, 5s fill window, 30s adverse-selection horizon. Gate result: **PROCEED to K2**.

## Canonical Category Table

| segment | category | fills | fill rate | spread | rebate | adverse | inv/res | net | 95% CI | clears |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fee_enabled | Crypto | 2,963 | 0.03% | 667.5 bps | 66.9 bps | 103.9 bps | 274.8 bps | 355.7 bps | [284.2 bps, 434.3 bps] | yes |
| fee_enabled | Finance | 3 | 0.04% | 258.0 bps | 55.9 bps | -20.6 bps | 10.3 bps | 324.1 bps | [n/a, n/a] | no |
| fee_enabled | Sports | 4,532 | 0.21% | 89.7 bps | 45.4 bps | 0.9 bps | 2.5 bps | 131.7 bps | [126.5 bps, 138.9 bps] | yes |
| fee_enabled | Tech | 170 | 0.02% | 86.8 bps | 36.9 bps | 2.9 bps | 6.0 bps | 114.9 bps | [93.1 bps, 139.9 bps] | yes |
| fee_free | Geopolitics | 1,562 | 0.05% | 137.0 bps | 0.0 bps | 22.3 bps | 19.1 bps | 95.6 bps | [87.9 bps, 103.1 bps] | yes |

## Full Category Grid

| segment | category | fill W | horizon | fills | fill rate | net | 95% CI | clears |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fee_enabled | Crypto | 1 | 5 | 5,426 | 0.06% | 440.4 bps | [374.3 bps, 508.5 bps] | yes |
| fee_enabled | Crypto | 1 | 30 | 2,956 | 0.03% | 360.5 bps | [286.8 bps, 433.0 bps] | yes |
| fee_enabled | Crypto | 1 | 60 | 2,173 | 0.03% | 189.5 bps | [25.1 bps, 315.2 bps] | yes |
| fee_enabled | Crypto | 5 | 5 | 5,441 | 0.06% | 440.9 bps | [375.2 bps, 511.5 bps] | yes |
| fee_enabled | Crypto | 5 | 30 | 2,963 | 0.03% | 355.7 bps | [284.2 bps, 434.3 bps] | yes |
| fee_enabled | Crypto | 5 | 60 | 2,178 | 0.03% | 183.7 bps | [18.1 bps, 322.3 bps] | yes |
| fee_enabled | Crypto | 10 | 5 | 5,443 | 0.06% | 440.8 bps | [378.5 bps, 504.2 bps] | yes |
| fee_enabled | Crypto | 10 | 30 | 2,965 | 0.03% | 355.5 bps | [282.6 bps, 430.4 bps] | yes |
| fee_enabled | Crypto | 10 | 60 | 2,178 | 0.03% | 186.1 bps | [6.6 bps, 323.0 bps] | yes |
| fee_enabled | Finance | 1 | 5 | 3 | 0.04% | 316.4 bps | [n/a, n/a] | no |
| fee_enabled | Finance | 1 | 30 | 3 | 0.04% | 324.1 bps | [n/a, n/a] | no |
| fee_enabled | Finance | 1 | 60 | 3 | 0.04% | 319.0 bps | [n/a, n/a] | no |
| fee_enabled | Finance | 5 | 5 | 3 | 0.04% | 316.4 bps | [n/a, n/a] | no |
| fee_enabled | Finance | 5 | 30 | 3 | 0.04% | 324.1 bps | [n/a, n/a] | no |
| fee_enabled | Finance | 5 | 60 | 3 | 0.04% | 319.0 bps | [n/a, n/a] | no |
| fee_enabled | Finance | 10 | 5 | 3 | 0.04% | 316.4 bps | [n/a, n/a] | no |
| fee_enabled | Finance | 10 | 30 | 3 | 0.04% | 324.1 bps | [n/a, n/a] | no |
| fee_enabled | Finance | 10 | 60 | 3 | 0.04% | 319.0 bps | [n/a, n/a] | no |
| fee_enabled | Sports | 1 | 5 | 7,067 | 0.33% | 141.5 bps | [134.0 bps, 152.8 bps] | yes |
| fee_enabled | Sports | 1 | 30 | 4,509 | 0.21% | 131.7 bps | [126.4 bps, 138.1 bps] | yes |
| fee_enabled | Sports | 1 | 60 | 3,432 | 0.16% | 132.7 bps | [125.7 bps, 143.3 bps] | yes |
| fee_enabled | Sports | 5 | 5 | 7,105 | 0.33% | 141.4 bps | [133.9 bps, 151.5 bps] | yes |
| fee_enabled | Sports | 5 | 30 | 4,532 | 0.21% | 131.7 bps | [126.5 bps, 138.9 bps] | yes |
| fee_enabled | Sports | 5 | 60 | 3,446 | 0.16% | 132.5 bps | [125.7 bps, 142.3 bps] | yes |
| fee_enabled | Sports | 10 | 5 | 7,107 | 0.33% | 141.4 bps | [133.7 bps, 152.2 bps] | yes |
| fee_enabled | Sports | 10 | 30 | 4,534 | 0.21% | 131.7 bps | [126.2 bps, 137.9 bps] | yes |
| fee_enabled | Sports | 10 | 60 | 3,446 | 0.16% | 132.5 bps | [125.7 bps, 142.2 bps] | yes |
| fee_enabled | Tech | 1 | 5 | 181 | 0.02% | 121.6 bps | [98.5 bps, 144.5 bps] | yes |
| fee_enabled | Tech | 1 | 30 | 170 | 0.02% | 114.9 bps | [89.3 bps, 139.3 bps] | yes |
| fee_enabled | Tech | 1 | 60 | 159 | 0.01% | 112.7 bps | [88.3 bps, 137.5 bps] | yes |
| fee_enabled | Tech | 5 | 5 | 181 | 0.02% | 121.6 bps | [100.1 bps, 149.8 bps] | yes |
| fee_enabled | Tech | 5 | 30 | 170 | 0.02% | 114.9 bps | [93.1 bps, 139.9 bps] | yes |
| fee_enabled | Tech | 5 | 60 | 159 | 0.01% | 112.7 bps | [88.9 bps, 139.3 bps] | yes |
| fee_enabled | Tech | 10 | 5 | 181 | 0.02% | 121.6 bps | [100.1 bps, 145.1 bps] | yes |
| fee_enabled | Tech | 10 | 30 | 170 | 0.02% | 114.9 bps | [93.1 bps, 138.6 bps] | yes |
| fee_enabled | Tech | 10 | 60 | 159 | 0.01% | 112.7 bps | [90.0 bps, 135.8 bps] | yes |
| fee_free | Geopolitics | 1 | 5 | 2,086 | 0.07% | 112.5 bps | [106.0 bps, 119.5 bps] | yes |
| fee_free | Geopolitics | 1 | 30 | 1,560 | 0.05% | 96.0 bps | [87.3 bps, 104.3 bps] | yes |
| fee_free | Geopolitics | 1 | 60 | 1,275 | 0.04% | 79.0 bps | [68.5 bps, 87.7 bps] | yes |
| fee_free | Geopolitics | 5 | 5 | 2,089 | 0.07% | 112.2 bps | [106.5 bps, 119.4 bps] | yes |
| fee_free | Geopolitics | 5 | 30 | 1,562 | 0.05% | 95.6 bps | [87.9 bps, 103.1 bps] | yes |
| fee_free | Geopolitics | 5 | 60 | 1,277 | 0.04% | 79.1 bps | [70.8 bps, 87.6 bps] | yes |
| fee_free | Geopolitics | 10 | 5 | 2,089 | 0.07% | 112.2 bps | [105.9 bps, 118.9 bps] | yes |
| fee_free | Geopolitics | 10 | 30 | 1,562 | 0.05% | 95.6 bps | [87.6 bps, 103.6 bps] | yes |
| fee_free | Geopolitics | 10 | 60 | 1,277 | 0.04% | 79.1 bps | [69.3 bps, 88.6 bps] | yes |

## Best Market Rows

| market | slug | category | fill W | horizon | fills | fill rate | net | 95% CI | clears |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2384176 | sol-updown-4h-1780113600 | Crypto | 1 | 60 | 16 | 0.01% | 1592.0 bps | [334.7 bps, 3320.3 bps] | no |
| 2378289 | btc-updown-4h-1780056000 | Crypto | 1 | 30 | 135 | 0.02% | 1202.6 bps | [787.5 bps, 1611.9 bps] | yes |
| 2379669 | eth-updown-4h-1780070400 | Crypto | 1 | 60 | 61 | 0.02% | 1154.4 bps | [543.5 bps, 1924.1 bps] | yes |
| 2380737 | eth-updown-4h-1780084800 | Crypto | 1 | 30 | 41 | 0.02% | 998.1 bps | [516.2 bps, 1664.5 bps] | yes |
| 2382619 | eth-updown-4h-1780099200 | Crypto | 1 | 30 | 37 | 0.02% | 980.1 bps | [55.6 bps, 1934.8 bps] | yes |
| 2380718 | btc-updown-4h-1780084800 | Crypto | 1 | 5 | 538 | 0.22% | 927.4 bps | [651.0 bps, 1167.7 bps] | yes |
| 2377049 | eth-updown-4h-1780041600 | Crypto | 1 | 30 | 45 | 0.02% | 751.3 bps | [-331.9 bps, 2172.3 bps] | no |
| 2382607 | btc-updown-4h-1780099200 | Crypto | 1 | 60 | 84 | 0.04% | 707.8 bps | [218.3 bps, 1311.1 bps] | yes |
| 2362186 | ethereum-up-or-down-on-may-28-2026 | Crypto | 1 | 5 | 160 | 0.03% | 686.1 bps | [507.5 bps, 876.2 bps] | yes |
| 665531 | metamask-fdv-above-700m-one-day-after-laun | Finance | 1 | 30 | 1 | 0.02% | 654.0 bps | [n/a, n/a] | no |
| 2364426 | btc-updown-4h-1779912000 | Crypto | 1 | 60 | 65 | 0.05% | 651.6 bps | [210.3 bps, 1280.5 bps] | yes |
| 2379677 | btc-updown-4h-1780070400 | Crypto | 1 | 5 | 393 | 0.11% | 608.6 bps | [434.9 bps, 767.5 bps] | yes |
| 2384180 | eth-updown-4h-1780113600 | Crypto | 1 | 5 | 78 | 0.05% | 554.7 bps | [45.9 bps, 1138.1 bps] | yes |
| 1295976 | will-openai-announce-earbuds-or-headphones | Tech | 1 | 60 | 1 | 0.00% | 506.4 bps | [n/a, n/a] | no |
| 2382617 | sol-updown-4h-1780099200 | Crypto | 1 | 60 | 37 | 0.03% | 496.5 bps | [-415.3 bps, 1397.9 bps] | no |

## Method

- Data: `data/analysis/block_a1_features.parquet`, pooled runs `a0, a0b, a0c, a0c_roll` as one in-sample set. No holdout split.
- Exclusion: slugs containing `will-jd-vance` were removed because of the known quote-noise issue.
- Zero skew: no OFI/TFI/TOB directional signal is used. The simulated maker quotes both sides symmetrically at the current mid.
- Fill proxy: A1.4h-style passive evidence. A bid fill requires a real SELL print at or below the prior mid within W; an ask fill requires a real BUY print at or above the prior mid within W. W is reported for 1s, 5s, and 10s.
- Non-overlap: after an actual fill, the market is blocked until `fill_time + adverse_horizon`. Unfilled quote opportunities do not block later fills.
- Rebate: `feeRate * p * (1-p) * rebate_pct`, normalized by entry price. Rebate pct is 20% for Crypto, 25% for other fee-enabled categories, and 0% for Geopolitics.
- Decomposition: `net = spread_capture + rebate - adverse_selection - inv_resolution_charge`.
- Spread capture: half the quoted spread at the prior book state, normalized by entry mid. This keeps K1 as a generous full-priority maker gate.
- Adverse selection: signed mark-to-market loss from entry mid to future mid at 5/30/60s. Positive values are costs.
- Inventory/resolution charge: `0.5 * abs(future_mid - entry_mid) / entry_mid`, using the same future mid path; settlement-adjacent jumps are included when visible in A1. A1 does not carry final resolution prices, so no synthetic resolution payoff is added.
- CI: 500 bootstrap resamples of contiguous 300s fill-time blocks, blocked within market.

## Interpretation

Geopolitics is the fee-free negative control: it has no rebate cushion. Fee-enabled categories receive the official rebate cushion, but the gate only clears if the category mean is positive, the lower CI is above zero, and there are at least 30 non-overlap fills.

Important caveat: because fee-free Geopolitics also clears, the K1 pass is not a pure rebate-cushion proof. It is mostly a generous spread-capture/full-priority maker baseline. Treat the gate result as permission to proceed to K2 simulation and queue/capacity stress, not as a deployable market-making result.

CSV: `data/analysis/csv_outputs/market_making/k1_maker_economics.csv`.
