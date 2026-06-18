---
tags: [dali, block-kpeg, maker, peg-chase, optuna]
---

# Block K-PEG Dynamic Chase Findings

> **Strat:** [[strat_market_making]] (Market-Making). Sibling: [[strat_options_delta]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Headline

Optimized dynamic chasing is positive on the pooled IS ceiling: best 30s net is 759.4 bps, CI [384.2 bps, 1171.6 bps], fill rate 0.04%, and 409 fills. Best policy: peg offset `0` ticks, chase increment `2` ticks, micro-price cap `c=7` ticks, inventory scaling `0.304`, cadence `1s`.

One-share portfolio replay over the selected-policy sample: $4.7311 net on $269.9420 gross entry notional, or 175.3 bps, across 58.32 elapsed hours.

Falsifier check: categories with negative net at every curve setting: none.

## Plain-English Summary

This test asks a practical market-making question: if Midas keeps moving our passive quote to stay competitive, how far should it chase before the extra fills become toxic?

A market maker posts bids and asks instead of crossing the spread. The maker earns money when it buys slightly cheap or sells slightly rich, may receive a maker rebate, and loses money when the trade was informed and price moves against the maker after the fill. The trade-off is simple:

`net = spread captured + rebate - adverse selection - inventory/resolution risk`

In the main category tables, **net PnL is calculated per simulated fill first**, then averaged across all non-overlapping fills for the market or category. The reported `net` number is mean basis points per fill; the CI is the uncertainty around that mean. The `30s` horizon means we look 30 seconds after each fill to estimate adverse selection and inventory risk.

The portfolio table is different: it converts each fill's bps into dollars assuming **one share per fill**, then cumulatively sums those dollars over the actual replay span. This is still a ceiling replay, not a deployable account statement, because it ignores queue rank, partial fill sizing, cancellation latency, and capital constraints beyond the internal inventory cap.

The main result is not "chase as much as possible." For Crypto, the best setting joins the best quote and chases by 2 ticks, but stops about 7 ticks away from micro-price. In plain terms: be competitive, but leave a meaningful cushion before fair value.

## Concepts

- **Best bid / best ask:** the highest visible buy price and lowest visible sell price in the order book.
- **Tick:** the smallest price step. Here one tick is 1 cent.
- **Pegging:** keeping our quote tied to the best bid or best ask.
- **Join:** quote at the current best bid or ask. In this run, `peg_offset = 0`.
- **Improve:** quote one tick better than the current best quote to get priority.
- **Sit behind:** quote one tick worse than the best quote, safer but less likely to fill.
- **Chase increment:** how many ticks we move when the best bid rises, mirrored for asks.
- **Micro-price:** an order-book estimate of fair value using both price and size at the best bid/ask.
- **Chase cap `c`:** the stop line around micro-price. For bids, `c=7` means stop at `micro_price - 7 ticks`; `c=0` means chase up to micro-price; `c<0` means chase past estimated fair value.
- **Inventory scaling:** if we already hold inventory, reduce chase aggressiveness so we do not keep adding risk in the same direction.
- **Cadence:** how often the quote refreshes. The best result used a 1-second cadence.
- **Adverse selection:** the bad case where someone fills us because they know or react faster than we do, and price moves against us after the fill.
- **Sign flip:** the first tested point where the next extra chase step makes marginal PnL worse.

## Visuals

Selected policy by category. Crypto is the only robust category; Sports has too few fills.

![](data/analysis/kpeg_plots/kpeg_selected_category_net.png)

Crypto chase/cap heatmap. Each cell is mean net PnL in bps for that chase increment and micro-price cap. Circle = optimum; X = corrected first bad extra increment.

![](data/analysis/kpeg_plots/kpeg_crypto_chase_heatmap.png)

Crypto portfolio heatmap. Each cell is total net dollars over the replay using one-share fills, so it rewards both per-fill quality and fill count.

![](data/analysis/kpeg_plots/kpeg_crypto_portfolio_heatmap.png)

Where the category optimum, bad extra increment, and bad tighter cap occur relative to micro-price.

![](data/analysis/kpeg_plots/kpeg_optimum_vs_signflip.png)

Fill-rate versus net-PnL trade-off. Higher fill rate is not automatically better; toxic fills can reduce net.

![](data/analysis/kpeg_plots/kpeg_fillrate_vs_net.png)

Portfolio replay for the selected policy. This is cumulative one-share-per-fill net PnL through the captured sample.

![](data/analysis/kpeg_plots/kpeg_selected_portfolio_curve.png)

## Selected Policy by Category

| segment | category | fills | fill rate | spread | rebate | adverse | inv/res | net | 95% CI | clears |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fee_enabled | Crypto | 403 | 0.10% | 1306.9 bps | 45.0 bps | 196.9 bps | 462.3 bps | 692.7 bps | [352.8 bps, 1044.3 bps] | yes |
| fee_enabled | Sports | 6 | 0.00% | 5162.5 bps | 7.8 bps | -134.8 bps | 67.4 bps | 5237.7 bps | [425.8 bps, 10074.1 bps] | no |

Simple read: Crypto clears because it has enough fills and the confidence interval stays above zero. Sports looks profitable, but the fill count is too small to trust.

## Portfolio Replay

| scope | fills | span | elapsed | net $ | gross entry $ | return | fills/hr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Crypto | 403 | 2026-05-27T21:40 -> 2026-05-30T07:59 | 58.32h | $4.5915 | $267.2610 | 171.8 bps | 6.9 |
| Sports | 6 | 2026-05-30T01:17 -> 2026-05-30T02:39 | 1.37h | $0.1396 | $2.6810 | 520.7 bps | 4.4 |
| pooled | 409 | 2026-05-27T21:40 -> 2026-05-30T07:59 | 58.32h | $4.7311 | $269.9420 | 175.3 bps | 7.0 |

Simple read: the pooled selected policy made $4.7311 over 58.32 hours under the one-share-per-fill replay. That is the portfolio-style version of the same per-fill edge; it is useful for capacity intuition, while the bps table is cleaner for comparing trade quality.

## Marginal Curve Optima

| category | max setting | max net | fill rate | bad extra increment | extra location | bad tighter cap | cap location |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Crypto | inc=2, c=7, dist=6.8 | 692.7 bps | 0.10% | inc=3, c=7, -41.3 bps | before_microprice | c=6, -125.8 bps | before_microprice |
| Finance | inc=0, c=2, dist=2.9 | 690.3 bps | 0.07% | none | none | c=0, -355.3 bps | near_microprice |
| Geopolitics | inc=1, c=2, dist=2.4 | 132.9 bps | 0.01% | inc=2, c=2, -5.6 bps | before_microprice | c=1, -4.0 bps | near_microprice |
| Sports | inc=1, c=1, dist=2.3 | 1017.8 bps | 0.02% | inc=2, c=1, -16.7 bps | near_microprice | c=0, -865.2 bps | near_microprice |
| Tech | inc=0, c=0, dist=0.5 | 194.5 bps | 0.12% | none | none | c=-1, -6.9 bps | near_microprice |

Simple read: the best Crypto curve point is `inc=2, c=7`, which means the bot chases, but still stops well before micro-price. The corrected extra-increment sign flip is `inc=3, c=7`: at the same cap, one more tick of chase reduces mean net PnL. The separate tighter-cap flip asks what happens if we keep `inc=2` but move closer to micro-price; for Crypto, the first bad tighter cap is also before micro-price, so adverse selection binds before the fair-value boundary in this ceiling model.

## What We Tested

1. **Starting quote position:** sit behind, join, or improve the best quote.
2. **Chase increment:** move by 0 to 4 ticks when the best bid/ask moves.
3. **Micro-price cap:** stop before, at, or past estimated fair value.
4. **Inventory conditioning:** reduce chase as open inventory grows.
5. **Requote cadence:** refresh every 1, 2, 5, 10, or 30 seconds.

For each policy, the simulator replayed real A1 book/trade data. A modeled bid filled only when a real SELL trade crossed our modeled bid. A modeled ask filled only when a real BUY trade crossed our modeled ask. This makes both fill rate and adverse selection depend on how aggressively we chase.

## How Conclusions Were Reached

1. We pooled `a0`, `a0b`, `a0c`, and `a0c_roll` as one in-sample dataset and excluded `will-jd-vance`.
2. For each quote setting, we reconstructed fills using the A1.4h passive-fill proxy.
3. For every fill, we measured spread captured, maker rebate, price movement against us after 5/30/60 seconds, and an inventory/resolution charge.
4. Optuna searched 360 combinations of peg offset, chase increment, cap, inventory scaling, and cadence.
5. The best policy was selected on 30-second net PnL.
6. A curve sweep then measured how PnL changed as chase increment and micro-price cap changed.
7. Portfolio replay converted each selected fill into one-share dollars and cumulated those dollars over the actual sample span.

Bottom line: dynamic chasing can rescue the maker baseline in this optimistic replay, especially in Crypto, but only with a cap. The practical next question is whether this survives real queue position, latency, partial fills, and cancellation risk.

## Method

- Data: `data/analysis/block_a1_features.parquet`, pooled `a0, a0b, a0c, a0c_roll` as one in-sample set. No holdout split.
- Exclusion: slugs containing `will-jd-vance`.
- Tick size: fixed at 1 cent.
- Micro-price: `(ask * bid_size + bid * ask_size) / (bid_size + ask_size)`, falling back to mid when touch size is missing.
- Bid policy: `best_bid + peg_offset*tick + chase_increment*positive_bid_move_ticks*tick`, scaled by `max(0, 1 - inventory_scaling*abs(inventory))`, capped at `micro_price - c*tick`; ask side is mirrored.
- Cap sweep includes `c < 0`, allowing quotes past micro-price in the curve and optimizer.
- Cadence/staleness: quotes are refreshed on sampled book states at the optimized cadence and ignored after `2 * cadence`.
- Fill proxy: A1.4h extension. A SELL print fills our bid only if `trade_price <= modeled_bid`; a BUY print fills our ask only if `trade_price >= modeled_ask`. More aggressive chase settings therefore change both fill count and realized adverse selection.
- PnL: `realized_spread + rebate - adverse_selection - inventory/resolution_charge`; rebate follows K1 rules. Objective optimizes 30s net; 5/30/60s rows are included in the CSV.
- Portfolio PnL: one-share-per-fill replay, `net_usd = net_bps / 10000 * entry_price`, cumulatively summed by fill time. This is a capacity diagnostic, not a real account equity curve.
- Inventory: one-share fills with an internal ±5 open-lot cap per asset; chase size is reduced as open inventory grows.
- CI: 500 bootstrap resamples over contiguous 300s fill-time blocks within market.
- Optuna trials: 360.

## Interpretation

The marginal curve is the key read. The extra-increment flip holds cap fixed and asks whether one more chase tick helps. The tighter-cap flip holds chase increment fixed and asks whether moving closer to micro-price helps. If either flip appears at `c≈0`, the micro-price cap is behaving like the Stoikov fair-value boundary. If it appears at larger positive `c`, adverse selection binds before fair value. If it appears at negative `c`, the ceiling still tolerated some past-fair-value chasing before the next increment turned negative.

CSV: `data/analysis/csv_outputs/market_making/kpeg_chase_optimization.csv`.

Selected fills: `data/analysis/csv_outputs/market_making/kpeg_selected_fills.csv`.

Portfolio series: `data/analysis/csv_outputs/market_making/kpeg_portfolio_timeseries.csv`.

Candidate rows precomputed: 122,937 across 56 markets.
