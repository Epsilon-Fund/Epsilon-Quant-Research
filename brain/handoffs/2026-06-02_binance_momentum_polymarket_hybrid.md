---
title: "Handoff — Binance daily momentum overlaid with Polymarket BTC/ETH binaries"
tags: [handoff, cross-project, momentum, polymarket, options-delta, hybrid]
created: 2026-06-02
status: do not continue branch
purpose: >
  Evaluate whether the live Binance daily momentum book should be overlaid with Polymarket BTC/ETH daily
  or 4h binaries, as either a hedge sleeve or an alpha sleeve.
---

# Binance daily momentum plus Polymarket BTC/ETH binaries: hedge helps drawdowns only by paying away too much return, and the alpha sleeve does not clear even under generous proxy edges

> Links: [[COWORK]] · [[POLYMARKET_BRAIN]] · [[TODO]] · [[CODEX]] · [[STRATEGY_REFERENCE]] · [[strat_options_delta]] · [[od_strategy_a_v2_lifecycle_findings]] · [[block_k6_strategy_a_static_hedge_findings]] · [[od_same_day_crypto_pricing_gate_findings]] · [[od_cross_asset_gate0_universe_map_findings]] · [[od_rv_deribit_daily_capture_findings]] · [[polymarket_table_dictionary]]

Plain-English read: the standalone daily momentum book is already strong. A small ATM Polymarket DOWN-binary hedge can reduce drawdown and annualized volatility, but the return and Sharpe cost is too large. OTM tail hedges are worse. The alpha-sleeve framing, where we buy an UP binary when the momentum model says Polymarket is underpricing it, does not rescue the idea in the quick proxy replay. It increases volatility, worsens drawdown, and loses return even with assumed +3c/+6c gross quote edges. This is not worth building.

## Project separation rule

This is a cross-project research overlay only. `live_trading`/crypto and `polymarket` share no code, use separate virtual environments, and must never cross-import. The analysis script reads fixed crypto OOS pickle outputs and Polymarket CSV/parquet files as data inputs only. It does not import Polymarket project code into the crypto stack.

## What was measured from real data vs proxy

Real measured inputs:

- Momentum baseline returns, equity paths, positions, and trade stats came from saved daily `-m` OOS artifacts under `topics/momentum/strategies/momentum_cpcv/oos/`. No WF/CPCV engine rerun.
- BTC and ETH momentum positions came from saved per-asset OOS CPCV strategy frames.
- Current Polymarket daily BTC/ETH terminal quote depth came from `polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_current_quotes.csv` at 2026-06-02 16:48 UTC.
- Existing Polymarket 4h captured fill/LOB data exists around 2026-05-27 to 2026-05-30, but it is far too short to overlay the 2020-2026 momentum OOS.

Proxy/assumption inputs:

- There is no historical tradable Polymarket BTC/ETH daily quote path covering the momentum OOS range. The overlay therefore uses a transparent EWMA digital-option fair-value proxy.
- Hedge entry prices are model probability plus explicit spread, impact, and longshot-premium costs. ATM hedge cost is +1.25c over model fair. OTM -2% down hedge cost is +2.5c over model fair.
- Alpha sleeve scenarios assume gross Polymarket underpricing of +3c or +6c versus the model, then subtract 1.5c slippage/impact. Those edges are not observed historical PM edges.
- The overlay replay uses 201 deterministic CPCV portfolio paths, including a representative path for charts, while the standalone baseline uses all 2,000 saved portfolio paths. This is a quick scenario replay, not a full optimization.

## Step 0 baseline first

Primary baseline is the daily `-m` momentum book. The top-level `combined_cpcv_*` files include `-bb` legs, and `-bb` is not the correct strategy for this task. The primary portfolio baseline therefore comes from `topics/momentum/strategies/momentum_cpcv/oos/portfolio_cpcv_*.pkl`, plus BTC alone from `btcusdt_cpcv.pkl`.

Column meanings: CAGR, annualized volatility, Sharpe, and max drawdown are means across saved CPCV paths. Avg trade, median trade, and hit rate are trade-level summaries reconstructed from saved OOS strategy frames or the saved portfolio trade-stat pickles. Max drawdown is shown as a negative return; less negative is better.

| Standalone book | OOS date range | CPCV paths | CAGR | Ann. vol | Sharpe | Max drawdown | Avg trade | Median trade | Hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT daily momentum | 2020-05-23 to 2026-04-30 | 105 | 61.98% | 39.84% | 1.40 | -43.63% | 2.07% | -0.81% | 40.94% |
| Six-asset daily momentum portfolio | 2020-10-04 to 2026-04-28 | 2,000 | 77.05% | 27.16% | 2.24 | -26.20% | 6.55% | -0.59% | 47.05% |

Read: BTC alone is profitable but volatile and drawdown-heavy. The diversified daily momentum book is the baseline to beat: 77.05% CAGR, 2.24 Sharpe, and -26.20% max drawdown. Any Polymarket overlay has to improve risk-adjusted performance against this row, not just create a nice one-off convex payoff.

Sample and embargo: saved CPCV config is `N=8`, `k=2`, `28` purged splits, `105` single-asset paths, `purge_bars=1` daily bar, `burnin=100`, `cost=0.001`, and `n_trials=400`. The six-asset portfolio uses 2,000 sampled combinations of the daily per-asset CPCV paths. `topics/momentum/outputs/wf_fold_results.csv` and `wf_oos_full.html` were checked as older WF context, but the CPCV OOS artifacts above are the baseline used for this decision.

Yearly OOS mean returns:

| Year | BTC alone | Six-asset daily momentum |
|---:|---:|---:|
| 2020 | 201.71% | 33.72% |
| 2021 | 37.16% | 279.76% |
| 2022 | -32.39% | -19.98% |
| 2023 | 234.43% | 201.36% |
| 2024 | 73.30% | 74.28% |
| 2025 | 21.53% | 19.78% |
| 2026 | -8.30% | -5.62% |

Read: 2022 and early 2026 are the obvious stress periods. The right hedge would need to help there without damaging the large compounding years.

## Step 1 quick overlay design

The overlay is applied on top of fixed daily momentum portfolio returns. No optimization and no engine rebuild.

Hedge interpretation A:

- Momentum is long-only, so the hedge buys a DOWN binary when the BTC or ETH momentum leg is long.
- Size is payout notional as a percentage of that asset leg's momentum notional, not premium paid. Example: 10% size means a $1 payout binary notional equal to 10% of the BTC or ETH momentum leg notional.
- ATM strike uses the entry daily close as the binary threshold. OTM tail strike is 2% below the entry daily close.
- Entry price is EWMA fair probability plus conservative execution/friction costs.

Alpha interpretation B:

- The sleeve buys an ATM UP binary only when the momentum leg is long and the model probability is at least 53%.
- Since historical PM implied probabilities are missing across the OOS, the scenarios assume gross PM underpricing of +3c or +6c versus the model and then subtract 1.5c slippage/impact.
- These are "what if this edge existed" scenarios, not measured PM alpha.

## Step 2 results vs baseline

Column meanings: `Size` is binary payout notional as a percentage of the active BTC/ETH momentum leg notional. `Δ` columns are overlay minus baseline. For `Δ Max DD`, positive means the drawdown became less severe. `PM ROC` is annualized overlay P&L divided by average Polymarket collateral tied up in the replay; negative means the separate USDC collateral was paid to lose money.

| Scenario | Interpretation | Size | Strike | CAGR | Δ CAGR | Ann. vol | Δ vol | Sharpe | Δ Sharpe | Max DD | Δ Max DD | Avg PM collateral | PM ROC |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline daily momentum | baseline | 0% | none | 77.05% | 0.00% | 27.16% | 0.00% | 2.24 | 0.00 | -26.20% | 0.00% | 0.00% | n/a |
| A hedge ATM 5% | Hedge | 5% | ATM DOWN | 62.09% | -14.95% | 24.05% | -3.11% | 2.13 | -0.11 | -20.24% | 5.96% | 0.29% | -3393.66% |
| A hedge ATM 10% | Hedge | 10% | ATM DOWN | 47.23% | -29.82% | 23.53% | -3.62% | 1.76 | -0.48 | -19.02% | 7.18% | 0.58% | -3393.66% |
| A hedge ATM 20% | Hedge | 20% | ATM DOWN | 19.02% | -58.02% | 30.26% | 3.10% | 0.72 | -1.52 | -37.19% | -11.00% | 1.15% | -3393.66% |
| A hedge OTM -2% 5% | Hedge | 5% | OTM DOWN | 47.10% | -29.95% | 24.97% | -2.19% | 1.67 | -0.57 | -31.88% | -5.69% | 0.14% | -13484.53% |
| A hedge OTM -2% 10% | Hedge | 10% | OTM DOWN | 21.63% | -55.41% | 24.02% | -3.14% | 0.93 | -1.31 | -41.66% | -15.46% | 0.29% | -13484.53% |
| A hedge OTM -2% 20% | Hedge | 20% | OTM DOWN | -17.71% | -94.75% | 26.30% | -0.86% | -0.61 | -2.85 | -85.84% | -59.64% | 0.57% | -13484.53% |
| B alpha +3c gross edge 5% | Alpha sleeve | 5% | ATM UP | 58.34% | -18.70% | 31.01% | 3.86% | 1.63 | -0.60 | -32.32% | -6.12% | 0.24% | -4176.50% |
| B alpha +6c gross edge 5% | Alpha sleeve | 5% | ATM UP | 65.59% | -11.46% | 31.06% | 3.90% | 1.78 | -0.46 | -31.53% | -5.34% | 0.23% | -2463.67% |
| B alpha +6c gross edge 10% | Alpha sleeve | 10% | ATM UP | 53.92% | -23.13% | 35.89% | 8.73% | 1.38 | -0.86 | -36.95% | -10.76% | 0.46% | -2463.67% |

Read: the best hedge result is 5% or 10% ATM DOWN. It reduces max drawdown by 5.96 to 7.18 percentage points and lowers annual volatility by 3.11 to 3.62 points, but gives up 14.95 to 29.82 percentage points of CAGR and loses Sharpe. That is not a good trade-off for a book whose baseline Sharpe is already 2.24. OTM tail hedges are dominated: they cost return and worsen drawdown because they pay too rarely. The alpha sleeve is worse than baseline in every tested case, even under assumed quote edges.

Metric-by-metric comparison:

| Metric | Best value in this replay | Winner | Why it matters |
|---|---:|---|---|
| CAGR | 77.05% | Baseline daily momentum | Every overlay gave up return. The smallest loss was the B alpha +6c 5% scenario at 65.59%, still `-11.46` percentage points versus baseline. |
| Annualized volatility | 23.53% | A hedge ATM 10% | The ATM hedge reduces volatility by `-3.62` percentage points, but this comes with `-29.82` percentage points of CAGR and `-0.48` Sharpe. |
| Sharpe | 2.24 | Baseline daily momentum | No overlay improved risk-adjusted performance. The closest was A hedge ATM 5% at 2.13, still `-0.11` below baseline. |
| Max drawdown | -19.02% | A hedge ATM 10% | This is the one metric where the hedge helps: `+7.18` percentage points less severe than baseline. The improvement is not free; it costs too much CAGR and Sharpe. |
| Return on Polymarket collateral | n/a for baseline; all overlay rows negative | Baseline / do nothing | PM ROC is negative in every overlay scenario, from about `-2,464%` to `-13,485%` annualized in the proxy replay. The separate USDC collateral is not earning enough to justify itself. |
| Capacity realism | Baseline has Binance capacity; PM overlay constrained | Baseline daily momentum | Current BTC/ETH daily PM top-of-book depth is too small for a material overlay on the Binance book. If sized small enough to avoid moving PM, the hedge becomes immaterial. |

Read: the overlay only wins the narrow "make the worst drawdown shallower" metric, and even that win is paid for with a large deterioration in CAGR, Sharpe, and collateral efficiency. This is why the gate is closed despite the hedge having a visible protective effect.

![Representative equity and drawdown comparison](../../polymarket/research/data/analysis/plots/options_delta/binance_momentum_polymarket_hybrid_equity_drawdown.png)

Chart read: this representative CPCV path shows the same pattern visually. ATM hedge flattens some drawdowns but truncates compounding. The OTM hedge bleeds and can make drawdowns worse. The alpha sleeve trails baseline and adds path volatility.

![Overlay entry timing distribution](../../polymarket/research/data/analysis/plots/options_delta/binance_momentum_polymarket_hybrid_timing_distribution.png)

Chart read: most overlay entries happen early in existing momentum positions, with median position age around 5 to 6 daily bars. The alpha sleeve is not a rare late-stage convex expression; it mostly stacks extra binary risk into ordinary active trend periods.

## Worked BTC reversal example

This example is from representative CPCV path `21`, using the 10% ATM DOWN hedge proxy.

On 2020-05-30, BTC momentum was already long. BTC closed at `$9,697.72`; the next daily close was `$9,448.27`, a `-2.57%` reversal. The BTC momentum leg had position size `0.907` and the portfolio weight was `1/6`, so the BTC futures-style leg lost about `-0.39%` of total book equity for that one-day move.

The hedge bought an ATM DOWN binary at `51.25c`, with payout notional equal to `1.51%` of book equity and premium collateral of `0.775%` of book equity. Because BTC resolved below the entry threshold, the binary paid `$1`, producing `+0.737%` of book-equity P&L. For this one day, the hedge more than offset the BTC leg loss.

Read: this is exactly why the hedge is tempting. It works on a clean reversal day. The problem is portfolio-level repetition: buying this protection every active BTC/ETH momentum day pays too much premium and impact during the long compounding stretches.

## Step 3 risk and capacity reality check

Payoff mismatch: the momentum book has linear, uncapped exposure to trend continuation. A long binary hedge has capped upside and repeated premium bleed. A binary can patch a single reversal, but it cannot cheaply replicate the convexity needed across a multi-year trend book.

Basis and timing risk: the replay uses daily close-to-close proxy resolution. Real Polymarket daily BTC/ETH terminal markets resolve at a fixed timestamp from Binance 1m candles, not at the momentum backtest's generic daily close. The 4h products have even tighter timestamp dependence. This creates a live basis/timing mismatch between continuous Binance exposure and fixed PM resolution.

Capacity and liquidity: current BTC/ETH terminal quote depth is not remotely sized for a serious overlay. In the 2026-06-02 snapshot, terminal BTC median 24h volume was about `$76.9k` and terminal ETH median 24h volume about `$21.5k`. Median YES ask size was about `2,020` BTC contracts and `464` ETH contracts. A 5% overlay on a `$1M` book would want roughly `$7k-$10k` of payout notional per active BTC or ETH leg before accounting for position-size variation, already far above median top-of-book depth and material versus 24h ETH flow. This is likely the binding constraint. Any realistic overlay size is either immaterial to the Binance book or takes enough PM depth to change the price.

Capital efficiency: Polymarket collateral sits in USDC with no margin offset against Binance futures or spot exposure. The sim's average PM collateral looks small as a fraction of normalized book equity, but it scales linearly and has to be funded separately. The negative PM ROC numbers are a warning that the collateral is not just idle; it is actively paying for drag.

Prior OD evidence: [[strat_options_delta]] already says OD Strategy A v2 failed the primary OOS far-|z| gate under global time embargo, and the hedge overlay stayed gated. The hybrid framing does not change that. Interpretation A inherits the "hedge is not the edge" finding: hedging can reduce variance in some regimes, but it pays away return. Interpretation B inherits the daily/terminal binary pricing problem: without observed OOS-clean PM mispricing, the convex alpha sleeve is just extra binary risk.

## Step 4 verdict and gate

Gate outcome: **DO NOT CONTINUE THIS BRANCH.**

Reason: the baseline six-asset daily momentum book is strong enough that the overlay must clearly improve risk-adjusted performance. It does not. The best hedge variant improves max drawdown by about `6-7` percentage points but costs `15-30` percentage points of CAGR and loses Sharpe. The alpha sleeve loses against baseline even when granted assumed +3c/+6c gross quote edges. Capacity likely makes any live-sized hedge immaterial before it becomes useful.

Minimal OOS-clean test design if someone explicitly reopens this later: collect 60-100 independent BTC/ETH daily or settlement-aligned 4h Polymarket quote snapshots with top-of-book and depth, pre-register a model-vs-implied threshold, and evaluate only fixed-size paper orders net of fees, spread, impact, and timestamp basis. Do not build an execution system before that data exists.

Single next action: **stop this hybrid branch and do not create a build ticket.**

## Artifacts written

- Script: `topics/momentum/research/binance_momentum_polymarket_hybrid.py`
- Baseline CSV: `polymarket/research/data/analysis/csv_outputs/options_delta/binance_momentum_polymarket_hybrid_baseline.csv`
- Results CSV: `polymarket/research/data/analysis/csv_outputs/options_delta/binance_momentum_polymarket_hybrid_results.csv`
- Timing summary CSV: `polymarket/research/data/analysis/csv_outputs/options_delta/binance_momentum_polymarket_hybrid_timing.csv`
- PM depth metadata CSV: `polymarket/research/data/analysis/csv_outputs/options_delta/binance_momentum_polymarket_hybrid_metadata.csv`
- Charts: `polymarket/research/data/analysis/plots/options_delta/binance_momentum_polymarket_hybrid_equity_drawdown.png` and `polymarket/research/data/analysis/plots/options_delta/binance_momentum_polymarket_hybrid_timing_distribution.png`
