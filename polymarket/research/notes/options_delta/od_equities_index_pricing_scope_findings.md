---
title: "OD Equities Index Pricing Scope Findings"
created: 2026-06-03
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
  - equities
  - index-updown
---
# OD Equities Index Pricing Scope Findings

> Hub: [[strat_options_delta]] · [[COWORK]]

## Summary

- Scope: OD Equities Index Pricing Scope Findings in the OD/options-delta area.
- Existing takeaway/status: SPX daily up/down clears the small-capital capacity and persistence scope, but the cheap pricing gate closes the branch.** This revises the old [[od_cross_asset_gate0_universe_map_findings]] deferral only on capacity: index up/down was thin under the prior `$50k/day` operation-scale bar, not under the `$10-$100` measurement scale now used for OD/MM live loops. The appended N(z) / realized-vol pricing gate below then...
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Verdict

**SPX daily up/down clears the small-capital capacity and persistence scope, but the cheap pricing gate closes the branch.** This revises the old [[od_cross_asset_gate0_universe_map_findings]] deferral only on capacity: index up/down was thin under the prior `$50k/day` operation-scale bar, not under the `$10-$100` measurement scale now used for OD/MM live loops. The appended N(z) / realized-vol pricing gate below then answers the pricing question: **CONFIRM-CLOSE / no Cboe-OPRA build**.

The options-surface path is now conditional only on future fresh evidence. The pre-registered cheap-first rule was: build Cboe/OPRA only if the realized-vol N(z) residual survives net of spread/fee with market-date CI. It did not.

Short version:
- **Proceed:** no further SPX pricing build from this sample.
- **Capacity:** SPX daily up/down remains the only equity template that clears small-capacity.
- **Pricing:** CONFIRM-CLOSE under the N(z) / realized-vol gate; do not build the listed-option surface.
- **Do not spend yet:** NDX, single-stock up/down, or close-above ladders.
- **Drop:** financial NegRisk baskets for this branch; they were intentionally excluded as thin one-offs.

## What Changed

The prior Gate-0 note used a `$50k/day` volume and `$25k/day` non-top3 headroom bar. That bar is sensible for an operation-scale system, but it is too harsh for a `$10-$100` live measurement loop. This pass changes only the capacity lens and adds persistence diagnostics. It does not claim pricing inefficiency.

The scrape was refreshed on **2026-06-03 10:21:34 UTC**. Sources:
- Gamma events/markets for current and recent-90d markets.
- CLOB order books for live best bid/ask and top-of-book depth.
- Local recent raw OrderFilled shards for equity-subset maker concentration where condition IDs were covered; otherwise the template uses an equity-family proxy from exact matched rows.

## Small-Capital Gate

Unit of observation in this table is a recurring Polymarket template, not an individual trade. `Run-rate volume/day` uses live 24h volume if a live market exists, otherwise recent-90d resolved volume divided by 90. `Non-top3 headroom/day` multiplies that run-rate by the equity-subset non-top3 maker-share proxy. `Persistence` distinguishes recurring daily templates from one-offs and spiky ladders.

| template | persistence | live 24h volume | recent 90d volume | non-top3 headroom/day | live spread | top3 share | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| SPX daily up/down | recurring daily, 38 recent dates | $24.4k | $8.54M | $12.7k | 1.0c | 48.0% | **clears small-capacity; pricing closed in append** |
| NDX daily up/down | intermittent daily, 17 recent dates | $0 | $250k | $0.9k | n/a | 67.6% | defer: no live template and headroom below small gate |
| single-stock up/down, recurring names | recurring daily | $0-$830/name | $443k-$633k/name | $2-$2.2k/name historical, but live mostly <$400/name | 3c-7c live | 48.7%-60.2% | defer: live flow/spread weak |
| equity close-above ladders | mostly one-off/intermittent | up to $16.1k | sparse by template | mostly <$2.2k | 2c-94c | often high | defer: not persistent daily up/down; many rows are sparse ladders |

Read: the only robust scope pass is **SPX daily up/down**. It has enough recurring history, live flow, non-top3 headroom, and a 1c live CLOB spread for `$10-$100` measurement tickets. NDX is clean but currently absent live; single-stock templates recur historically but live flow is tiny and spreads are wide; close-above ladders are not the clean recurring daily terminal digital requested here.

## Persistence Details

SPX daily up/down is not just one large day. In the recent window it had **38 settlement dates**, a median event gap of **1 day**, top-3 event-volume share of **12.9%**, and max/median event-volume ratio of **2.37x**. That is a recurring template with some event-size variation, not a single spike.

Single-stock markets also recur, but the live state is weak: on 2026-06-03 the active MSFT up/down market had about **$830** 24h volume at **3c** spread, TSLA about **$422** at **6c**, NVDA about **$271** at **5c**, GOOGL about **$58** at **7c**, and AAPL about **$4** at **6c**. That is not worth an options-surface build before SPX.

## External Pricing Path

For **SPX daily up/down**, the clean listed-option path would be:

1. Use Polymarket CLOB snapshots at candidate entry times. 2. Extract the PM threshold from the market: the SPX daily up/down market resolves against the official close, directionally relative to the relevant prior/open reference in the market rules. 3. Pull same-day **SPXW / XSP PM-settled listed options** around the PM threshold from Cboe DataShop/LiveVol or an equivalent OPRA vendor. 4. Compute the listed-option-implied digital fair value using a tight vertical call-spread slope around the threshold strike; use Black-Scholes `N(d2)` from interpolated 0DTE IV as a diagnostic cross-check, not the sole gate. 5. Compare to executable Polymarket bid/ask plus fees. Use first qualifying quote per market/date, carry to resolution, and report market/date-cluster CI.

Why this path would be clean enough if reopened: Cboe documents SPX/SPXW PM-settled weekly products and DataShop/LiveVol advertises live, delayed, and historical equity/options endpoints. Alpha Vantage documents US options APIs, including historical options since 2008, but historical options are a premium function; it is a fallback for single-name work, not the preferred SPX path. The path was **not built** in this pass because the cheaper realized-vol gate below did not survive.

## Decision

**Capacity verdict:** SPX daily up/down **clears small-capacity** at `$10-$100` scale.

**Pricing verdict:** **CONFIRM-CLOSE after the appended N(z) / realized-vol gate.** The listed-option residual was deliberately not built because the cheap executable residual did not clear.

Modeled assumptions:
- `$100` max ticket is the relevant scale for this reopen.
- `$1,000/day` non-top3 headroom is enough for a measurement loop because it is 10x the max ticket.
- Top-3 concentration from local raw recent trades is a capacity proxy, not a hard cap.

Live-only unknowns:
- Actual passive fill share and queue position in SPX daily up/down.
- Whether the 1c spread is reachable without becoming adverse-selection inventory.
- Whether OPRA/Cboe snapshots can be aligned tightly enough to PM quote timestamps if a future cheap residual justifies that spend.
- Whether a future fresh sample can clear the cheap N(z) gate before any options-surface work.

## Outputs

- Script: `scripts/od_equities_index_pricing_scope.py`
- Template summary: `data/analysis/csv_outputs/options_delta/od_equities_index_pricing_scope_template_summary.csv`
- Market detail: `data/analysis/csv_outputs/options_delta/od_equities_index_pricing_scope_market_detail.csv`
- Reference paths: `data/analysis/csv_outputs/options_delta/od_equities_index_pricing_scope_reference_paths.csv`

Public references checked:
- [Polymarket fetching markets](https://docs.polymarket.com/market-data/fetching-markets)
- [Polymarket order book](https://docs.polymarket.com/trading/orderbook)
- [Cboe DataShop documentation](https://datashop.cboe.com/documentation)
- [Cboe S&P 500 Weeklys specifications](https://www.cboe.com/tradable_products/sp_500/spx_weekly_options/specifications/)
- [Alpha Vantage API documentation](https://www.alphavantage.co/documentation/)

## 2026-06-03 N(z) / Realized-Vol Pricing Gate

Pricing verdict: **CONFIRM-CLOSE**. Do **not** build the Cboe/OPRA options surface for SPX daily up/down yet.

This append runs the cheap-first gate requested after the capacity pass. It compares actual executable SPX daily up/down fills to a causal S&P 500 `N(z)` digital fair value, then checks an empirical conditional probability table in the same spirit as [[od_conditional_prob_calibration_findings]]. The reference data is Yahoo Finance `^GSPC` daily closes plus completed 60-minute bars; no listed-option surface is used.

Headline numbers: **105,495** causal fill rows across **41** resolved SPX market dates survived the filters. The all-fill side-aware N(z) residual is **-0.49c**, market-date CI **[-0.56c, -0.41c]**. The first-per-date N(z) `edge > 0` row has model residual **3.18c**, CI **[1.88c, 5.43c]**, but realized net PnL **-2.41c**, CI **[-38.51c, 27.78c]**. That is not a clean executable residual worth escalating to OPRA.

### Design

Each row is an actual Polymarket fill for the SPX daily up/down market, not a midpoint. The market resolves `Up` if the official SPX close is above the most recent prior trading-day close, so the digital strike is the prior official close.

Causal reference rules:
- Exclude fills before the prior official close is known.
- Exclude fills at or after the current day's official 20:00 UTC close.
- For each remaining fill, use the latest completed Yahoo `^GSPC` 60-minute bar at or before the fill. Overnight/pre-open fills use the prior official close.
- Volatility is an EWMA of prior daily SPX close-to-close returns only.
- `N(z)` fair is `N(log(spot / prior_close) / (sigma * sqrt(time_to_close)))`.

Executable residual rules:
- If `maker_side = SELL`, the maker sold the token and the counterparty could buy it: residual is `fair - execution_price - fee`.
- If `maker_side = BUY`, the maker bought the token and the counterparty could sell it: residual is `execution_price - fair - fee`.
- The finance fee proxy is Polymarket's category formula `0.04 * p * (1-p)`.
- The gate uses market-date clustered CIs and first qualifying fill per date for threshold rows; no mark-to-mid and no options data.

### Residual Summary

| cell | fills | market_dates | weighted_notional_usd | mean_model_edge | edge_ci_lo | edge_ci_hi | mean_realized_net_pnl | pnl_ci_lo | pnl_ci_hi | mean_fee |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_causal_fills_nz_side_aware | 105,495 | 41 | $6,578,654 | -0.49c | -0.56c | -0.41c | -0.55c | -0.62c | -0.48c | 0.52c |
| all_causal_fills_empirical_side_aware | 105,495 | 41 | $6,578,654 | -0.49c | -0.56c | -0.42c | -0.55c | -0.62c | -0.48c | 0.52c |
| first_per_date_nz_edge_gt_0 | 41 | 41 | $1,083 | 3.18c | 1.88c | 5.43c | -2.41c | -38.51c | 27.78c | 0.99c |
| first_per_date_nz_edge_ge_1c | 41 | 41 | $1,233 | 3.16c | 2.16c | 4.86c | -8.30c | -41.03c | 23.57c | 0.99c |
| first_per_date_nz_edge_ge_2c | 41 | 41 | $1,252 | 3.53c | 2.65c | 5.47c | -14.11c | -39.53c | 19.76c | 0.99c |
| first_per_date_emp_edge_gt_0 | 41 | 41 | $799 | 3.65c | 2.29c | 5.24c | -7.97c | -29.19c | 16.52c | 0.99c |
| first_per_date_emp_edge_ge_1c | 41 | 41 | $753 | 4.35c | 3.13c | 5.72c | -11.51c | -30.28c | 14.51c | 0.99c |
| first_per_date_emp_edge_ge_2c | 41 | 41 | $769 | 4.42c | 3.22c | 5.74c | -13.18c | -31.99c | 11.13c | 0.99c |

Read: the all-fill N(z) residual is statistically below zero after spread/fee because actual taker-side fills are not systematically mispriced in the model's favor. Threshold rows can find positive model residuals by construction, but they do not clear realized net PnL with a market-date CI. The best-looking row by lower realized CI is `all_causal_fills_nz_side_aware`, and even that is **-0.55c**, CI **[-0.62c, -0.48c]**.

![SPX N(z) residual by date](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_equities_spx_nz_pricing_date_edges.png)

Caption: each bar is a resolved SPX market date. Blue is the side-aware `N(z)` residual after fee on actual fills; orange is realized net PnL for that same executable side. A pricing build would need a stable positive residual across dates, not a few outcome-driven bars.

### Empirical Calibration

| model | prob_bin | rows | mean_pred_up | observed_up | observed_minus_pred |
| --- | --- | --- | --- | --- | --- |
| N(z) | 40_60c | 139 | 50.23% | 58.99% | 8.77% |
| N(z) | 60_75c | 37 | 66.69% | 62.16% | -4.53% |
| N(z) | 10_25c | 32 | 16.37% | 25.00% | 8.63% |
| N(z) | 90_100c | 319 | 99.09% | 97.49% | -1.59% |
| N(z) | 75_90c | 36 | 83.25% | 83.33% | 0.08% |
| N(z) | 25_40c | 26 | 33.87% | 30.77% | -3.10% |
| N(z) | 0_10c | 244 | 0.91% | 7.79% | 6.87% |
| empirical | 40_60c | 131 | 53.62% | 58.02% | 4.39% |
| empirical | 60_75c | 62 | 68.84% | 64.52% | -4.33% |
| empirical | 0_10c | 218 | 4.91% | 5.05% | 0.14% |
| empirical | 90_100c | 291 | 96.80% | 98.28% | 1.48% |
| empirical | 25_40c | 42 | 31.98% | 28.57% | -3.41% |
| empirical | 75_90c | 52 | 86.63% | 86.54% | -0.09% |
| empirical | 10_25c | 37 | 21.41% | 29.73% | 8.32% |

Read: the realized-vol `N(z)` model is not perfect, but the empirical SPX calibration does not rescue the PM residual. This mirrors the crypto precedent in [[od_conditional_prob_calibration_findings]]: once a broad causal base-rate is used, the obvious standalone pricing gap is gone or not executable.

![SPX calibration](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_equities_spx_nz_pricing_calibration.png)

### Data Ledger

| bucket | detail |
|---|---|
| PM fills | Local raw trade shards cover SPX daily up/down fills through **2026-05-26 19:57:58 UTC**. The active June 3 and later-resolved May 27-June 2 markets remain in the scope table but are not in this fill tape. |
| Reference | Yahoo Finance chart API `^GSPC`, daily + 60-minute completed bars. This is a realized-vol proxy, not an option-implied surface. |
| Not built | Cboe/OPRA options surface. The pre-registered escalation condition was not met. |

Modeled assumptions:
- Yahoo 60-minute SPX bars are adequate for the cheap realized-vol pass.
- Finance taker fee is `4% * p * (1-p)`.
- Selling an over-fair token at an observed bid is an executable residual only for inventory/mint-capable flow; the note reports it but does not convert it into a standalone live strategy.

Live-only unknowns:
- Real quote queue/fill share for a passive SPX maker.
- Whether a tighter intraday SPX feed would materially change sub-hour fill states.
- Whether an OPRA surface would reveal tiny IV-vs-RV differences; this is not worth building until the cheap realized-vol residual survives.

### Decision

**CONFIRM-CLOSE / DEFER OPTIONS BUILD.** SPX cleared small-cap capacity, but the cheap pricing gate does not show a clean executable net residual. This establishes the stronger research state the reopen wanted: PM financial-binary pricing looks efficient across both crypto terminal digitals and the most-arbitraged equity-index up/down template under the cheap causal base-rate test.

Outputs:
- Script: `scripts/od_equities_spx_nz_pricing_gate.py`
- Scored fills: `data/analysis/od_equities_spx_nz_pricing_fills.parquet`
- Summary: `data/analysis/csv_outputs/options_delta/od_equities_spx_nz_pricing_summary.csv`
- Calibration: `data/analysis/csv_outputs/options_delta/od_equities_spx_nz_pricing_calibration.csv`
- Date summary: `data/analysis/csv_outputs/options_delta/od_equities_spx_nz_pricing_date_summary.csv`

## 2026-06-03 Implied-Vol N(z) Last Swing Audit

Strict verdict: **STILL-BLOCKED for the exact VIX + ES + best-ask gate**. This does **not** create a MERITS-BUILD signal and does **not** justify an options-chain or ML build. The broader SPX pricing thesis remains **CONFIRM-CLOSE** on the completed realized-vol/fill-level gate above, but the requested last swing cannot be reconstructed from available clean historical inputs.

What was requested: replace realized volatility with forward-looking **Cboe VIX** and replace cash spot/drift with **ES futures**, then compare to executable **Polymarket best ask** with market-date clustered CI. I did not substitute yfinance, midpoint marks, last-trade marks, or an options chain.

### Data Audit

| ingredient | required_for_strict_gate | available | coverage | detail |
| --- | --- | --- | --- | --- |
| Cboe VIX close | true | true | 42/42 (100.0%) | Direct Cboe VIX CSV covers every local SPX fill market-date. |
| Cboe VIX9D close | false | true | 42/42 (100.0%) | Direct Cboe VIX9D CSV is available as a short-horizon diagnostic, but the user-requested primary input is VIX. |
| CME ES front-month settlement | true | false | fill sample 1/42 (2.4%); scope 5/47 (10.6%) | CME public settlement endpoint only returns recent May 27-June 2 rows. The single fill-date overlap is May 27, whose local PM rows stop before that day's resolution close. |
| Historical PM SPX daily up/down best ask | true | false | 0 book lines / 0 best-bid-ask lines matched | Local live_clob files have no SPX daily up/down book snapshots. Existing SPX historical sample is actual fills only. |
| Local PM SPX daily up/down fills | false | true | 108,060 fill rows across 42 market-dates | Useful for the earlier no-mid fill-level N(z) close, but not a literal best-ask replay. |

Read: the Cboe vol input is not the blocker. Direct Cboe VIX/VIX9D files cover the local SPX sample. The blockers are the two executable/history legs: local PM data has **108,060** SPX daily up/down fill rows across **42** market-dates (2026-03-26 to 2026-05-27), but no historical SPX daily up/down CLOB best-ask snapshots; the local PM tape's latest row is **2026-05-26 19:46:03 UTC**. CME's public ES settlement endpoint returned **5** scope-date hits (2026-05-27, 2026-05-28, 2026-05-29, 2026-06-01, 2026-06-02). The only raw fill-date overlap is May 27, but those PM rows occur before the local tape ends on May 26, not near the May 27 resolution close required for the requested best-ask comparison.

### Why Not a Proxy

Using actual fills again would answer a different question: "did observed fills look rich/cheap versus a VIX-based model?" That is useful only as a sensitivity, and it would still need an ES-forward input for moneyness. Using the old Yahoo cash states would violate this task's clean-source instruction. Using PM last trade or fill VWAP as "best ask" would reintroduce the same non-executable quote problem the prompt explicitly ruled out.

### Missing To Run The Strict Gate

- Historical Polymarket CLOB snapshots for SPX daily up/down token best ask/best bid, with timestamps before the official SPX close.
- Historical ES front-month prices aligned to those PM quote timestamps, or at minimum CME settlement/last data for the same historical SPX market dates; the public CME endpoint checked here only exposed recent rows after the local PM fill tape ended.
- Official SPX settlement remains available through the resolved PM markets / prior close logic; that is not the blocker.

### Decision

**No ML, no options-chain build, no OPRA spend.** The rule-based last swing cannot be made decision-grade from the current cache without violating the no-mark-to-mid/no-yfinance discipline. If this is ever reopened, the next step is a live SPX daily up/down collector that logs PM best ask/bid plus Cboe VIX and ES front-month state at quote time; until then, the pricing branch stays closed by the realized-vol gate and blocked for the stricter VIX+ES replay.

Outputs:
- Script: `scripts/od_equities_spx_iv_nz_last_swing.py`
- Audit: `data/analysis/csv_outputs/options_delta/od_equities_spx_iv_nz_last_swing_data_audit.csv`
- VIX coverage: `data/analysis/csv_outputs/options_delta/od_equities_spx_iv_nz_last_swing_vix_coverage.csv`
- CME ES probe: `data/analysis/csv_outputs/options_delta/od_equities_spx_iv_nz_last_swing_cme_es_probe.csv`
- Local CLOB scan: `data/analysis/csv_outputs/options_delta/od_equities_spx_iv_nz_last_swing_clob_scan.csv`

References checked:
- [Cboe VIX historical data](https://www.cboe.com/tradable_products/vix/vix_historical_data)
- [Cboe direct VIX CSV](https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv)
- [CME E-mini S&P 500 settlements](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.settlements.html)

## 2026-06-03 SPX Opens Up/Down Current-Data Scope

Additive correction: the close-market analysis above did **not** include **S&P 500 (SPX) Opens Up or Down**. It is a separate Gamma series, `spx-open-daily-up-or-down`, with different timing and a cleaner overnight use case for ES/MES futures. This section scopes it with current Gamma/CLOB data plus local Dali candidate metadata. It does **not** claim a pricing edge.

Why a CLOB capture was not needed for this first pass: current Gamma plus the CLOB `/book` endpoint are enough to answer whether the product exists, whether it recurs, how much flow it has, and what the book looks like right now. The missing CLOB capture matters only for the next question: a historical or forward **executable pricing gate** with market-date CI, because that requires the best bid/ask actually available during the after-close-before-open window.

### Current-Data Results

| item | result |
|---|---:|
| Gamma series | `spx-open-daily-up-or-down` |
| Series recurrence | daily |
| Recent Gamma rows | 63 |
| Recent event dates | 54 |
| Median event gap | 1 day |
| Local Dali candidate rows | 31 |
| Local candidate fills | 62,131 |
| Local candidate volume | $6.27M |
| Median local event volume | $179k |
| Max local event volume | $502k |
| Weighted top-3 maker share | 40.7% |
| Weighted non-top3 share | 59.3% |
| Non-top3 headroom/day from current live 24h flow | $37.8k |

Read: **SPX opens clears the small-capacity scope at `$10-$100` scale.** This is not the same as the close-style SPX daily up/down product, and it should not have been omitted from the universe discussion. The local history is also not a one-day curiosity: it has recurring daily rows and large enough non-incumbent flow for a measurement loop assumption.

### Live Book Snapshot

As of **2026-06-03 16:15:43 UTC**, the active rows are split across two different phases:

| market | phase | current book |
|---|---|---|
| `spx-opens-up-or-down-on-june-3-2026` | post official open; outcome effectively known | post-open one-sided book around `0.001 / 0.999`; useful for volume/state only, not entry pricing |
| `spx-opens-up-or-down-on-june-4-2026` | pre-open candidate | Up `0.51 x 0.64`, Down `0.36 x 0.49`; about **13c spread**, best-depth only about **$69-$227** |

This is exactly where current data helps and where it stops. The June 4 market is visible before the official open, but at the snapshot time the June 3 official close was not yet known, so the final strike for "June 4 open above prior close" was not yet fixed. The clean ES/MES window starts after the prior official SPX close is known and runs until the next official open.

### What Is Missing

Local scan result: **3** SPX-open CLOB metadata sightings (`new_market`) and **0** replayable historical book lines / **0** price-change lines for these condition IDs. The local fills and Dali rows prove flow and recurrence, but they do not reconstruct the best ask/bid we could have hit before the open.

To run the strict pricing gate, collect:
- Polymarket SPX-open best bid/ask, depth, and timestamp from after the prior official close through the next official open.
- ES/MES front-month state at the same timestamps, plus VIX/VIX9D as optional vol diagnostics.
- Official WSJ/SPX open and prior close for settlement.

Do **not** substitute last trades, fill VWAP, post-open 0/1 books, or midpoint marks for the missing pre-open CLOB. Those would answer a different question and would reintroduce the exact execution bias this audit is trying to avoid.

### Decision

**MERITS-LIVE-COLLECTOR / NO PRICING VERDICT.** SPX opens is a real, recurring, small-capacity-clearing template and is plausibly the right equity-index product to test with ES/MES while the cash market is asleep. But the current cache cannot produce a market-date CI or net-of-cost executable residual, because the required pre-open CLOB book history is absent.

This does not reopen the close-style SPX daily up/down pricing verdict above: close-style SPX remains **CONFIRM-CLOSE / no OPRA build** under the realized-vol gate. Opens is a separate branch whose next cheap step is a live pre-open collector, not an options chain or ML build.

Outputs:
- Script: `scripts/od_equities_spx_open_updown_scope.py`
- Market detail: `data/analysis/csv_outputs/options_delta/od_equities_spx_open_updown_scope_market_detail.csv`
- Summary: `data/analysis/csv_outputs/options_delta/od_equities_spx_open_updown_scope_summary.csv`
- Local CLOB scan: `data/analysis/csv_outputs/options_delta/od_equities_spx_open_updown_scope_local_clob_scan.csv`
