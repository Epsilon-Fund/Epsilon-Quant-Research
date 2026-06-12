---
title: "First-Mover Liquidity Scope in Newly Created Polymarket Markets"
tags: [market-making, first-mover, liquidity, novelty, block-k]
created: 2026-06-04
status: merits-live-stage-1-capture
---

# First-Mover Liquidity Scope in Newly Created Polymarket Markets

> Hub: [[strat_market_making]] · [[COWORK]]
> Context: [[2026-06-04_state_of_the_arc_and_novelty_frontier]] · [[block_k5_stress_findings]] · [[block_k5b_findings]]

## Plain-English Summary

- This note asks the cheap historical question behind the novelty thesis: after a Polymarket market is created, do dominant top-3 makers consolidate immediately, or is there early flow before they dominate?
- On owned historical fills, the cheap falsifier does **not** fire. In `$10k+` markets with first-week flow, only **12.9%** reached persistent 70% final-top3 maker share by day 7, and the lower-bound pre-consolidation non-top3 flow was about **$6.62B**, or **59.7%** of first-week maker dollars.
- This is **not** a historical edge proof. It says an early-flow / pre-dominance window exists in the fill tape. Whether it is quoteable, fillable, and not adversely selected is a live Stage-1 book/quote-capture question.

## Design

Data:
- Raw owned fill tape: `data/trades/*.parquet`.
- Market creation timestamps and metadata: `data/markets/markets_2026-05-06.parquet`.
- Universe start: markets created on or after **2025-08-01**, matching the K5-STRESS research era.
- Primary market scope: markets with at least **$10k** total observed maker flow after creation. This keeps the result deployability-relevant and avoids dust markets steering medians.

Definition:
- For each market, identify the **final top-3 maker wallets** by full observed maker USD in that market. Then, at each elapsed age bucket from market creation, measure how much cumulative maker flow those final top-3 wallets had captured using only fills up to that age.
- Primary consolidation gate: final top-3 maker share is at least **70%** at an age bucket, at least `$100` cumulative maker flow has arrived, and the share stays at least 70% through day 7.
- The conservative pre-consolidation volume lower bound is the cumulative maker flow at the previous age bucket. If a market does not consolidate by day 7, the lower bound is its full first-week maker flow.

Important caveat: final top-3 identity is an ex-post historical label used to measure incumbent arrival. It is **not** a live trading rule. The age-window flow and shares use no settlement/PnL lookahead, no mark-to-mid, and no outcome information, but a live system cannot know a market's final top-3 wallets at creation. Live Stage 1 should replace this descriptive label with book/flow telemetry and prior-known maker-watchlists.

Practical example: if a market has `$20k` of maker flow by 1h and the wallets that will eventually be that market's top 3 have only `$8k` of it, the 1h final-top3 share is 40%. If by 3d they reach 75% and stay above 70% through day 7, the conservative pre-consolidation flow lower bound is the previous bucket's cumulative flow, not the full 3d bucket. That avoids crediting the whole crossing bucket as safely uncontested.

## Coverage

The table below shows the primary `$10k+` market universe. `eligible total` is all observed maker flow after market creation; `first-week flow` is the subset used for the top3-window test. Markets with no first-week flow are not ignored strategically, but they are not useful evidence for a first-week first-mover liquidity window.

| category | eligible markets | eligible total | markets with first-week flow | first-week flow | no first-week flow |
| --- | ---: | ---: | ---: | ---: | ---: |
| other | 59,756 | $6.80B | 47,744 | $3.16B | 12,012 |
| daily_crypto | 79,140 | $4.20B | 79,140 | $4.15B | 0 |
| sports | 16,107 | $3.85B | 14,034 | $2.85B | 2,073 |
| geopolitics | 3,843 | $2.28B | 3,452 | $0.54B | 391 |
| economics | 228 | $0.29B | 213 | $0.005B | 15 |
| tech | 1,963 | $0.28B | 1,804 | $0.09B | 159 |
| finance | 4,060 | $0.28B | 4,029 | $0.13B | 31 |
| politics_negrisk | 1,395 | $0.28B | 1,268 | $0.011B | 127 |
| weather | 5,111 | $0.13B | 5,110 | $0.12B | 1 |
| crypto_4h | 740 | $0.017B | 740 | $0.017B | 0 |
| culture | 198 | $0.013B | 188 | $0.004B | 10 |

Read: daily crypto has essentially immediate first-week flow after creation; sports and other have large first-week flow but some delayed-start markets; geopolitics has large total flow but much less first-week flow because many high-volume markets start trading later.

## Main Result

Unit of observation: one `$10k+` market with at least some first-week maker flow. CIs are market-cluster bootstraps over markets. `consolidated by 7d` means persistent final-top3 share at or above 70% by day 7. `pre non-top3 flow` is the lower-bound maker flow before that persistent consolidation, excluding dollars already captured by the final top3.

| category | markets | first-week flow | consolidated by 7d | median consolidation time | pre non-top3 flow | pre non-top3 share | markets with `$1k+` pre-flow | top3 share 15m / 1h / 1d / 7d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| daily_crypto | 79,140 | $4.15B | 5.0% CI [4.9%, 5.2%] | 72h | $2.77B | 66.7% CI [66.4%, 67.0%] | 95.8% | 37% / 53% / 25% / 33% |
| sports | 14,034 | $2.85B | 26.3% CI [25.7%, 27.0%] | 24h | $1.67B | 58.7% CI [58.0%, 59.5%] | 73.3% | 80% / 73% / 59% / 40% |
| other | 47,744 | $3.16B | 20.2% CI [19.9%, 20.6%] | 72h | $1.63B | 51.5% CI [50.8%, 52.1%] | 63.0% | 38% / 55% / 49% / 46% |
| geopolitics | 3,452 | $0.54B | 10.3% CI [9.3%, 11.2%] | 24h | $0.36B | 66.9% CI [64.5%, 69.3%] | 70.9% | 43% / 41% / 45% / 32% |
| finance | 4,029 | $0.13B | 23.8% CI [22.5%, 25.2%] | 72h | $0.063B | 50.3% CI [45.8%, 54.5%] | 76.4% | 1% / 84% / 48% / 47% |
| weather | 5,110 | $0.12B | 22.2% CI [21.1%, 23.3%] | 72h | $0.053B | 42.6% CI [41.1%, 44.1%] | 80.0% | 4% / 24% / 44% / 54% |
| tech | 1,804 | $0.09B | 13.6% CI [12.1%, 15.2%] | 72h | $0.047B | 52.4% CI [50.0%, 55.2%] | 57.6% | 86% / 74% / 49% / 46% |
| politics_negrisk | 1,268 | $0.011B | 6.9% CI [5.4%, 8.6%] | 72h | $0.0069B | 64.1% CI [55.3%, 71.6%] | 38.2% | 0% / 22% / 35% / 34% |

Read: top3 consolidation is not near-instant. Under the primary 70% persistent-share definition, most categories have median pre-consolidation windows at the 7d measurement cap, because most markets do not persistently consolidate by day 7. The largest deployable early-flow pools are daily crypto, sports, and other; geopolitics is smaller in first-week dollars but has a cleanly large non-top3 share.

![Category ranking: first-mover scope](../../data/analysis/plots/market_making/mm_first_mover_liquidity_scope_category_ranking.png)

## Threshold Robustness

Across all `$10k+` markets with first-week flow:

| persistent final-top3 threshold | markets | first-week maker flow | consolidated by 7d | pre non-top3 share of first-week flow | markets with `$1k+` pre-flow |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 157,722 | $11.08B | 28.9% | 54.0% | 69.2% |
| 60% | 157,722 | $11.08B | 19.7% | 57.8% | 76.1% |
| 70% | 157,722 | $11.08B | 12.9% | 59.7% | 81.2% |
| 80% | 157,722 | $11.08B | 7.8% | 60.7% | 85.2% |

Read: the conclusion does not depend on choosing 70%. Even at a loose 50% persistent top3 threshold, most markets have not consolidated by day 7 and more than half of first-week maker flow is outside the final top3 before consolidation.

## Category Read

**Daily crypto:** largest early-flow pool and the cleanest statistical scope result, but it is the least attractive live-first target because the broader MM arc already found crypto adverse selection to be severe. Use it as a fast-market control, not the headline novelty sleeve.

**Sports:** strongest practical first-mover candidate from this batch. It has large first-week flow, fast resolution, objective outcomes, and a large pre-consolidation non-top3 dollar pool. The caveat is that final top3 share is high very early in some sports markets, so book capture must check whether the apparent window is actually quoteable or just volatile event flow.

**Other / residual misc:** large enough to matter, with a persistent window in aggregate, but it needs market taxonomy before capture. This should be split into the same recurring sleeves from [[mm_nonpolitics_target_screen_findings]] so we do not point a collector at an unstructured grab bag.

**Geopolitics / politics-like news:** the historical first-week flow is smaller than sports/daily crypto, but the non-top3 pre-consolidation share is large. This is plausible for a slow/retail-flow first-mover sleeve, but politics deployment and NegRisk handling remain lane-owned by the separate politics loop.

## Verdict

The novelty bet is **not dead cheaply**. The owned fill tape shows a measurable early-flow window before persistent final-top3 maker dominance, and the window is large enough in dollars to justify live Stage-1 capture.

But the result is only a **scope and capacity-assumption result**:
- Historical edge: not measured here. No PnL, no settlement, no mark-to-mid, no claim that passive quotes would profit.
- Capacity assumption: first-week pre-consolidation flow is large in several categories, especially daily crypto, sports, other, and geopolitics.
- Live deployability unknowns: spread/depth, queue position, visible book shape, fill probability, adverse selection around events, and whether our quotes would be hit before known maker wallets arrive.

## Recommended Next Measurement

Run a live Stage-1 new-market capture, not a trading bot:
- Watch newly created markets in sports, retagged other/residual sleeves, and geopolitics/politics-news-like markets. Add daily crypto only as a fast-market negative-control if capture cost is low.
- For each market, log first 7 days from creation: Gamma metadata, CLOB best bid/ask, spread, top depth, book depth near fair, trade arrivals, quoteability flags, and any observable maker/fill concentration after trades print.
- Pre-register the cheap pass/fail: if early markets have no tight/usable spread or depth, kill the branch; if they have repeated quoteable windows with retail flow before concentration, then proceed to a tiny passive quote/fill measurement loop.

Do not promote this to production edge. The correct next phrase is **live measurement loop**.

## Artifacts

- `data/analysis/csv_outputs/market_making/mm_first_mover_liquidity_scope_market_summary.csv`
- `data/analysis/csv_outputs/market_making/mm_first_mover_liquidity_scope_category_summary.csv`
- `data/analysis/csv_outputs/market_making/mm_first_mover_liquidity_scope_endpoint_summary.csv`
- `data/analysis/csv_outputs/market_making/mm_first_mover_liquidity_scope_threshold_sensitivity.csv`
- `data/analysis/csv_outputs/market_making/mm_first_mover_liquidity_scope_coverage_summary.csv`
- `data/analysis/csv_outputs/market_making/mm_first_mover_liquidity_scope_stage1_category_ranking.csv`
- `data/analysis/plots/market_making/mm_first_mover_liquidity_scope_category_ranking.png`
