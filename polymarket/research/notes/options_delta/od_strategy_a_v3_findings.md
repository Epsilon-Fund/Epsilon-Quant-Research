---
title: "Sell Rich 4h Crypto UP/DOWN Digitals With OD Fair-Value Filters (Strategy A v3)"
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
# Sell Rich 4h Crypto UP/DOWN Digitals With OD Fair-Value Filters (Strategy A v3)

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Table terms: [[polymarket_table_dictionary]]

## Summary

- Scope: Sell Rich 4h Crypto UP/DOWN Digitals With OD Fair-Value Filters (Strategy A v3) in the OD/options-delta area.
- Existing takeaway/status: OD Strategy A v3 clears only after the OD/source filter; bare power still fails.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Headline

OD Strategy A v3 clears only after the OD/source filter; bare power still fails.

Phase-1 baseline: Bare lifecycle OOS `far_absz_ge1_all_tau`, global embargo: n=6 markets / 68 fills, mean 118.08c, CI [-17.14c, 323.46c].

Phase-2 OD filter verdict: Best OD filter row: `strict_rich_short_ge_005m` mean 105.20c, CI [16.19c, 258.69c], lower-CI lift 33.33c.

Phase-3 cap verdict: Best cap row: `cap_strict_rich_short_ge_010m`, cap $500, mean 105.20c, CI [16.19c, 258.69c]. Smallest positive finite cap: `cap_strict_rich_short_ge_010m`, cap $50, mean 46.29c, CI [14.17c, 94.34c].

Power did not improve under the official assumption: the K3/K6 + K-PEG overlap has only six globally non-overlapping OOS 4h slots. Pooling BTC/ETH/SOL gives many more fills and market episodes before embargo, but the pre-registered **global** embargo still permits only one overlapping 4h window at a time. The positive v3 result comes from the OD/source filter, not from more power. A per-asset diagnostic is shown below; it is not the official gate.

## Design

This is the OD strat, not a Block K lead-lag race. MM supplies the passive K-PEG lifecycle: eligible maker fills are aggregated into a market episode, inventory can be two-sided, Polymarket positions are carried to resolution, and the late near-50c spike zone remains excluded. OD adds the valuation layer: only accept fills when the token is rich versus Binance RV physical-probability fair, when PM midpoint-implied vol is above causal EWMA vol, when the Chainlink/Binance source-basis risk is acceptable, and when dollar-delta inventory stays inside a cap.

Decision gate: OOS `far_absz_ge1_all_tau`, global market-episode embargo, net-of-cost, lower 95% cluster-bootstrap CI > 0. Phase 1 reports the bare lifecycle baseline; Phase 2 tests whether OD filters lift that lower CI above zero. Costs include maker rebate on Polymarket fills. The hedge rows add Binance costs at `6.0bp` per hedge trade/settlement notional, but hedge is a footnote and cannot create the edge gate.

## Phase Map — No Hedge in Phases 1-3

The three v3 phases that matter for the OD decision are all **unhedged Polymarket lifecycle** tests:

- **Phase 1, power baseline:** replay the bare K-PEG lifecycle and ask whether more assets / wider longshot buckets create enough independent evidence.
- **Phase 2, OD entry filter:** before accepting a maker fill, ask whether the token is actually overpriced versus Binance RV physical-probability fair or backed by a strict source-basis filter.
- **Phase 3, risk caps:** keep the same unhedged lifecycle, but skip fills that would push episode dollar-delta exposure above a cap.

The Binance hedge is **Phase 4 only**. It is a variance/cost diagnostic after the unhedged decision has already been made. The v3 PASS/FAIL statement above is therefore about the OD valuation filter, not about hedge PnL.

## Power and Global Embargo

`Power` means how much independent evidence the backtest has. More independent market episodes usually tighten the confidence interval, so a real edge is less likely to be hidden by one huge winner or one bad loser. More fills are helpful, but they are not the same as more power if those fills all happen inside the same overlapping 4h risk window.

The 4h windows themselves are sequential within one asset: one BTC 4h market ends, then the next BTC 4h market opens. The `global market-episode embargo` is about **cross-asset same-time overlap**, not overlap within one asset's 4h schedule. After selecting one 4h market episode, the replay ignores every other market episode whose active time overlaps it, even if the other market is a different asset such as ETH or SOL. In plain English: assume the strategy can have only one 4h OD episode active at a time, so same-time BTC, ETH, and SOL windows compete for the same capital/risk slot.

That is why v3 found more fills but not much more official power. The data has 17 OOS far-|z| asset-market episodes before a global embargo, but only 6 global 4h time slots after the embargo because many BTC, ETH, and SOL episodes occur during the same 4h interval. The per-asset diagnostic shows what happens if BTC, ETH, and SOL capital are treated independently; it is useful, but it is not the pre-registered gate.

## Options Valuation Layer

The 4h UP/DOWN market is treated as a European cash-or-nothing digital option. It resolves UP if the close is above the window-open reference price. There is no barrier or path dependence in the payoff.

The causal fair price for the UP token is:

```text
z = ln(S / K) / (sigma * sqrt(tau))
P_up_fair = N(z)
P_down_fair = 1 - P_up_fair
```

Where:

- `S` is current Binance spot at the fill timestamp.
- `K` is the 4h window-open strike/reference price.
- `sigma` is causal EWMA realized volatility from the K6 panel, using only information available up to that timestamp.
- `tau` is time left to resolution in years.
- `N(z)` is the standard normal CDF.

For a specific Polymarket token, the RV physical-probability fair is `P_up_rv_fair` for an UP token and `1 - P_up_rv_fair` for a DOWN token. The OD richness test then asks:

```text
short/sell token edge = entry_price - token_rv_physical_prob_fair
long/buy token edge   = token_rv_physical_prob_fair - entry_price
```

The headline v3 filter is deliberately narrow: it keeps short/sell fills only when the token is rich versus RV physical-probability fair. That tests the OD thesis as a forecast-selection rule: sell longshot/vol tokens that are expensive versus the causal RV model and carry them to resolution.

## Practical OD Filter Example

Example fill: `btc-updown-4h-1780041600` sold/shorted the DOWN token at $0.231. The Binance RV physical-probability fair for that token was $0.035, so the token was rich by 19.67c. A `rich_short >= 1c` filter keeps this fill; if the edge were below `1c`, v3 would skip it even though the K-PEG maker lifecycle would have filled it.

The signed value-edge convention is:

```text
long token:  rv_physical_prob_fair - entry_price
short token: entry_price - rv_physical_prob_fair
```

The headline richness filter is stricter than generic value-edge: it keeps only short/sell fills where the token is overpriced versus RV physical-probability fair. That directly tests the OD thesis as a forecast-selection rule: sell the longshot/vol side that is expensive versus our causal RV model and carry.

## Table Glossary

- `filter`: the lifecycle/filter rule. `bare_lifecycle` is v2-style K-PEG with no OD valuation gate.
- `embargo`: `global` is the official one-position-at-a-time embargo; `per_asset` is a power diagnostic.
- `cap`: maximum absolute dollar-delta inventory allowed while accepting fills inside an episode.
- `markets`: selected non-overlapping market episodes.
- `fills`: K-PEG fills inside those selected episodes after filters and caps.
- `mean net`: mean dollars per market episode, displayed in cents.
- `CI`: 95% bootstrap confidence interval over market episodes.
- `CI lift`: lower-CI improvement versus the Phase-1 bare global gate row.
- `win`: share of selected market episodes with positive net PnL.
- `PnL std`: standard deviation of market-episode PnL.
- `two-sided`: share of episodes containing both UP and DOWN token fills.
- `hold min`: median fill-to-resolution hold time in minutes.
- Filter suffixes are compact: `005m` = $0.005 = 0.5c, `010m` = 1c, `050m` = 5c, and `05vp` = 5 annualized vol points. The CSV contains the thinner n<3 rows; the Markdown tables hide them unless no robust row exists.

## Bucket Glossary

Buckets describe the option state at the fill timestamp. They are based on moneyness and time left, not on the eventual winner.

Moneyness uses:

```text
abs_z = abs(ln(S / K) / (sigma * sqrt(tau)))
```

Where `S` is Binance spot, `K` is the window-open strike, `sigma` is causal EWMA vol, and `tau` is time left. In plain English, `abs_z` is "how many volatility-adjusted units away from the strike are we?"

- `near_absz_lt0.25`: very close to the strike. The option is jumpy; a small spot move can flip the market.
- `mid_absz_0.25_1`: moderately away from the strike. Delta is meaningful, but the outcome is not yet pinned.
- `far_absz_ge1`: at least one vol-adjusted unit from the strike. This is the longshot/pinned family that v2/v3 care about most.
- `longshot_absz_ge0.75`: widened longshot family. It includes `far_absz_ge1` plus somewhat less extreme longshots to test whether the edge has more power when the boundary is relaxed.
- `longshot_absz_ge0.50`: even wider longshot family. It adds more fills/episodes, but can dilute the pure far-|z| thesis.
- `longshot_short_price_le30`: short/sell fills where the token price is at or below 30c. This is a practical "sell the cheap-looking longshot premium" bucket, distinct from the model-based `abs_z` bucket.

Time buckets:

- `early_gt2h`: more than two hours left to resolution.
- `mid_30m_2h`: 30 minutes to two hours left.
- `late_lt30m`: less than 30 minutes left.

Intersection buckets combine both labels, e.g. `far_absz_ge1|late_lt30m` means the market is far from the strike and has less than 30 minutes left. `far_absz_ge1_all_tau` pools the far family across all time-left buckets.

How the phases use buckets:

- **Phase 1** reports the official far-|z| gate plus widened longshot buckets, so we can see whether power improves when the longshot definition is relaxed.
- **Phase 2** keeps the official `far_absz_ge1_all_tau` gate and changes the **entry filter** instead of the bucket.
- **Phase 3** also keeps the official far bucket and changes the **dollar-delta cap**, so cap effects are comparable to Phase 1/2.

## Phase 1 — Power Baseline

| bucket | embargo | markets | fills | mean net | CI | win | PnL std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| far_absz_ge1_all_tau | global | 6 | 68 | 118.08c | [-17.14c, 323.46c] | 66.67% | 252.28c |
| far_absz_ge1_all_tau | per_asset | 17 | 211 | 85.56c | [-1.00c, 209.70c] | 47.06% | 230.63c |
| longshot_absz_ge0.75_all_tau | global | 6 | 116 | 291.52c | [2.35c, 618.83c] | 66.67% | 439.12c |
| longshot_absz_ge0.75_all_tau | per_asset | 18 | 234 | 99.66c | [-0.87c, 244.34c] | 55.56% | 277.49c |
| longshot_absz_ge0.50_all_tau | global | 6 | 139 | 271.53c | [-91.03c, 680.83c] | 50.00% | 544.80c |
| longshot_absz_ge0.50_all_tau | per_asset | 18 | 266 | 100.80c | [-19.62c, 268.05c] | 55.56% | 326.75c |

Read: the official global far-|z| gate is the first row. The per-asset rows answer "what if BTC, ETH, and SOL capital were treated independently?" They are useful for power diagnosis, but they are not the pre-registered decision gate in this run.

## Phase 2 — OD Entry Filters

| filter | embargo | cap | markets | fills | mean net | CI | CI lift | win | PnL std | two-sided | hold min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_rich_short_ge_005m | global | none | 4 | 16 | 105.20c | [16.19c, 258.69c] | 33.33c | 100.00% | 156.57c | 25.00% | 48.5 |
| strict_rich_short_ge_010m | global | none | 4 | 16 | 105.20c | [16.19c, 258.69c] | 33.33c | 100.00% | 156.57c | 25.00% | 48.5 |
| strict_rich_short_ge_000m | global | none | 5 | 18 | 84.12c | [9.23c, 216.06c] | 26.37c | 80.00% | 143.55c | 20.00% | 42.3 |
| official_strict_source | global | none | 6 | 84 | 115.78c | [4.13c, 319.24c] | 21.27c | 66.67% | 247.10c | 100.00% | 5.5 |
| strict_rich_short_ge_020m | global | none | 4 | 6 | 20.24c | [3.81c, 40.72c] | 20.95c | 75.00% | 21.23c | 0.00% | 48.5 |
| strict_vol_premium_ge_20vp | global | none | 6 | 51 | 43.41c | [-1.07c, 132.06c] | 16.07c | 16.67% | 108.51c | 50.00% | 5.6 |
| strict_vol_premium_ge_10vp | global | none | 6 | 64 | 43.20c | [-1.45c, 131.99c] | 15.69c | 16.67% | 108.61c | 50.00% | 6.3 |
| strict_vol_premium_ge_00vp | global | none | 6 | 75 | 100.01c | [-4.03c, 300.18c] | 13.11c | 33.33% | 243.27c | 66.67% | 5.5 |
| strict_rich_000m_vol_00vp | global | none | 4 | 13 | 81.41c | [-4.48c, 237.23c] | 12.66c | 50.00% | 157.05c | 25.00% | 34.5 |
| strict_vol_premium_ge_05vp | global | none | 6 | 56 | 71.30c | [-4.94c, 220.02c] | 12.20c | 16.67% | 181.24c | 50.00% | 5.5 |
| strict_rich_010m_vol_00vp | global | none | 3 | 11 | 108.61c | [-8.76c, 316.38c] | 8.38c | 66.67% | 180.44c | 33.33% | 27.7 |
| strict_rich_020m_vol_00vp | global | none | 3 | 3 | 2.82c | [-8.76c, 18.21c] | 8.38c | 33.33% | 13.88c | 0.00% | 28.8 |
| value_edge_ge_000m | global | none | 6 | 26 | 76.61c | [-16.50c, 197.02c] | 0.64c | 66.67% | 153.37c | 33.33% | 41.8 |
| rich_short_ge_000m | global | none | 6 | 23 | 67.30c | [-17.79c, 181.39c] | -0.65c | 66.67% | 142.68c | 33.33% | 41.8 |
| value_edge_ge_005m | global | none | 5 | 24 | 91.97c | [-19.73c, 249.80c] | -2.59c | 80.00% | 166.24c | 40.00% | 42.3 |
| value_edge_ge_010m | global | none | 5 | 24 | 91.97c | [-19.73c, 249.80c] | -2.59c | 80.00% | 166.24c | 40.00% | 42.3 |

Read: these rows ask whether OD valuation adds independent keep on top of the maker lifecycle. `rich_short` means sell only when the PM token is rich versus Binance RV physical-probability fair. `vol_premium` means PM midpoint-implied vol is above causal EWMA vol. `strict` promotes the Chainlink/Binance source-basis filter from diagnostic to official candidate design ingredient.

Non-hedged read: the bare far-|z| lifecycle failed because its lower CI was -17.14c. The strict-source and strict-rich-short filters lift the lower CI above zero before any Binance hedge is applied. This says the RV valuation filter is doing selection work in this replay; it does not prove external option-IV mispricing.

## Phase 3 — Dollar-Delta Risk Caps

| filter | embargo | cap | markets | fills | mean net | CI | CI lift | win | PnL std | two-sided | hold min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cap_strict_rich_short_ge_010m | global | $500 | 4 | 16 | 105.20c | [16.19c, 258.69c] | 33.33c | 100.00% | 156.57c | 25.00% | 48.5 |
| cap_strict_rich_short_ge_010m | global | none | 4 | 16 | 105.20c | [16.19c, 258.69c] | 33.33c | 100.00% | 156.57c | 25.00% | 48.5 |
| cap_strict_rich_short_ge_010m | global | $250 | 4 | 12 | 96.33c | [16.19c, 232.09c] | 33.33c | 100.00% | 138.94c | 25.00% | 48.5 |
| cap_strict_rich_short_ge_010m | global | $100 | 4 | 10 | 66.11c | [16.19c, 141.42c] | 33.33c | 100.00% | 79.18c | 25.00% | 48.5 |
| cap_strict_rich_short_ge_010m | global | $50 | 3 | 7 | 46.29c | [14.17c, 94.34c] | 31.31c | 100.00% | 42.40c | 33.33% | 54.7 |
| cap_official_strict_source | global | none | 6 | 84 | 115.78c | [4.13c, 319.24c] | 21.27c | 66.67% | 247.10c | 100.00% | 5.5 |
| cap_official_strict_source | global | $500 | 6 | 80 | 112.05c | [4.13c, 308.05c] | 21.27c | 66.67% | 237.99c | 100.00% | 5.5 |
| cap_official_strict_source | global | $250 | 6 | 75 | 100.88c | [4.13c, 274.55c] | 21.27c | 66.67% | 210.73c | 100.00% | 5.5 |
| cap_official_strict_source | global | $100 | 6 | 72 | 54.39c | [4.13c, 135.07c] | 21.27c | 66.67% | 97.82c | 100.00% | 5.5 |
| cap_official_strict_source | global | $50 | 6 | 69 | 33.95c | [0.93c, 87.32c] | 18.07c | 50.00% | 65.49c | 83.33% | 5.5 |
| cap_bare_lifecycle | global | $25 | 6 | 66 | 14.22c | [-1.56c, 40.78c] | 15.58c | 33.33% | 31.83c | 66.67% | 5.5 |
| cap_official_strict_source | global | $25 | 6 | 66 | 14.22c | [-1.56c, 40.78c] | 15.58c | 33.33% | 31.83c | 66.67% | 5.5 |
| cap_strict_rich_010m_vol_00vp | global | $500 | 3 | 11 | 108.61c | [-8.76c, 316.38c] | 8.38c | 66.67% | 180.44c | 33.33% | 27.7 |
| cap_strict_rich_010m_vol_00vp | global | none | 3 | 11 | 108.61c | [-8.76c, 316.38c] | 8.38c | 66.67% | 180.44c | 33.33% | 27.7 |
| cap_strict_rich_010m_vol_00vp | global | $250 | 3 | 9 | 98.22c | [-8.76c, 285.21c] | 8.38c | 66.67% | 162.50c | 33.33% | 28.4 |
| cap_bare_lifecycle | global | none | 6 | 68 | 118.08c | [-17.14c, 323.46c] | 0.00c | 66.67% | 252.28c | 83.33% | 15.9 |

Read: caps are meant to shrink the fat-tail dispersion, not discover a new edge. A good cap should retain most of the mean while tightening the lower CI. If it improves CI only by deleting the large winner, that is not a better strategy; it is just less exposure.

Non-hedged read: the $50 strict-rich-short cap is the cleanest risk-control proof-of-concept in this table: it keeps only 3 markets / 7 fills, but the mean stays positive and the CI remains above zero while PnL standard deviation falls materially. The broader strict-source caps keep more markets but still rely on the same source filter that v3 promoted from diagnostic to design rule.

## Phase 4 — Hedge Footnote

| variant | filter | policy | h | markets | net | CI | hedge cost | prem retained | var reduced | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| episode_static | strict_rich_short_ge_010m | static_fraction | 1.00 | 4 | 94.42c | [18.53c, 225.01c] | 5.87c | 0.90 | 28.20% | 97.76 |
| episode_static | strict_rich_short_ge_010m | static_fraction | 0.75 | 4 | 97.12c | [17.95c, 233.43c] | 4.40c | 0.92 | 21.60% | 73.32 |
| episode_static | strict_rich_short_ge_010m | static_fraction | 0.50 | 4 | 99.81c | [17.36c, 241.85c] | 2.93c | 0.95 | 14.69% | 48.88 |
| episode_static | strict_rich_short_ge_010m | z_dependent | 1.00 | 4 | 100.02c | [17.24c, 242.56c] | 2.68c | 0.95 | 14.03% | 44.71 |
| episode_static | strict_rich_short_ge_010m | z_dependent | 0.75 | 4 | 101.31c | [16.97c, 246.59c] | 2.01c | 0.96 | 10.62% | 33.53 |
| episode_static | strict_rich_short_ge_010m | static_fraction | 0.25 | 4 | 102.51c | [16.77c, 250.27c] | 1.47c | 0.97 | 7.49% | 24.44 |
| episode_static | strict_rich_short_ge_010m | vol_dependent | 1.00 | 4 | 93.73c | [16.76c, 224.63c] | 3.56c | 0.89 | 27.10% | 59.37 |
| episode_static | strict_rich_short_ge_010m | z_dependent | 0.50 | 4 | 102.61c | [16.71c, 250.63c] | 1.34c | 0.98 | 7.15% | 22.36 |
| episode_static | strict_rich_short_ge_010m | vol_dependent | 0.75 | 4 | 96.55c | [16.61c, 233.15c] | 2.84c | 0.92 | 20.70% | 47.40 |
| episode_static | strict_rich_short_ge_010m | vol_dependent | 0.50 | 4 | 99.43c | [16.47c, 241.66c] | 1.90c | 0.95 | 14.07% | 31.60 |
| episode_static | strict_rich_short_ge_010m | z_dependent | 0.25 | 4 | 103.90c | [16.45c, 254.66c] | 0.67c | 0.99 | 3.61% | 11.18 |
| episode_static | strict_rich_short_ge_010m | vol_dependent | 0.25 | 4 | 102.31c | [16.33c, 250.18c] | 0.95c | 0.97 | 7.17% | 15.80 |
| episode_static | strict_rich_short_ge_010m | iv_rv_spread_dependent | 1.00 | 4 | 100.93c | [12.50c, 251.74c] | 2.58c | 0.96 | 4.04% | 43.02 |
| episode_static | strict_rich_short_ge_010m | iv_rv_spread_dependent | 0.75 | 4 | 101.50c | [12.42c, 253.48c] | 2.43c | 0.96 | 2.54% | 40.57 |
| episode_static | strict_rich_short_ge_010m | iv_rv_spread_dependent | 0.50 | 4 | 102.07c | [12.35c, 255.21c] | 2.29c | 0.97 | 1.02% | 38.13 |
| episode_static | strict_rich_short_ge_010m | iv_rv_spread_dependent | 0.25 | 4 | 102.64c | [12.27c, 256.95c] | 2.14c | 0.98 | -0.50% | 35.69 |

Portfolio-roll diagnostic rows:

| variant | filter | policy | h | markets | net | CI | hedge cost | prem retained | var reduced | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| portfolio_24h_roll | bare_lifecycle_per_asset_power_diag | vol_dependent | 0.25 | 17 | 85.48c | [-1.07c, 209.62c] | 0.35c | 1.00 | 0.00% | 5.75 |
| portfolio_24h_roll | bare_lifecycle_per_asset_power_diag | vol_dependent | 0.50 | 17 | 85.41c | [-1.15c, 209.55c] | 0.68c | 1.00 | 0.00% | 11.29 |
| portfolio_24h_roll | bare_lifecycle_per_asset_power_diag | vol_dependent | 0.75 | 17 | 85.34c | [-1.21c, 209.48c] | 0.92c | 1.00 | 0.00% | 15.27 |
| portfolio_24h_roll | bare_lifecycle_per_asset_power_diag | vol_dependent | 1.00 | 17 | 85.27c | [-1.29c, 209.41c] | 1.07c | 1.00 | 0.00% | 17.89 |
| portfolio_24h_roll | bare_lifecycle_per_asset_power_diag | iv_rv_spread_dependent | 0.25 | 17 | 84.42c | [-2.14c, 208.56c] | 0.56c | 0.99 | 0.00% | 9.31 |
| portfolio_24h_roll | bare_lifecycle_per_asset_power_diag | z_dependent | 0.25 | 17 | 84.38c | [-2.18c, 208.52c] | 1.53c | 0.99 | 0.00% | 25.52 |
| portfolio_24h_roll | bare_lifecycle_per_asset_power_diag | iv_rv_spread_dependent | 0.50 | 17 | 83.39c | [-3.17c, 207.53c] | 1.08c | 0.97 | 0.00% | 17.92 |
| portfolio_24h_roll | bare_lifecycle_per_asset_power_diag | z_dependent | 0.50 | 17 | 83.20c | [-3.36c, 207.34c] | 3.06c | 0.97 | 0.00% | 51.04 |

Read: `episode_static` is the v2-style one hedge per market episode. `portfolio_24h_roll` nets per-asset hedge changes across the day instead of paying a separate close/open at every 4h boundary. Here the portfolio-roll row is a cost/turnover diagnostic on the per-asset power universe; it is not part of the global gate. These rows are variance diagnostics only. The OD edge gate is unhedged.

## Decision

Pre-registered Phase-1 baseline row: Bare lifecycle OOS `far_absz_ge1_all_tau`, global embargo: n=6 markets / 68 fills, mean 118.08c, CI [-17.14c, 323.46c].

Decision: **PASS** for the OD-filtered v3 design; **FAIL** for the bare Phase-1 lifecycle baseline.

Interpretation: the baseline did not get more power under the official global embargo. The actual v3 improvement comes from OD/source filtering, especially strict-source plus rich-short filters. This is a positive OD-filter result, but still a small market-episode sample. Per-asset rows are encouraging only if we explicitly reopen the capital assumption and allow concurrent BTC/ETH/SOL episodes.

Outputs:

- `data/analysis/csv_outputs/options_delta/od_strategy_a_v3.csv`
- `data/analysis/od_strategy_a_v3_trades.parquet`
- `data/analysis/od_strategy_a_v3_fills.parquet`

Supersedes the next-step framing in [[od_strategy_a_v2_lifecycle_findings]]; v2 remains the accounting baseline.
