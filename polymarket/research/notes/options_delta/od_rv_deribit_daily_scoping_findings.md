# OD-RV Scoping: PM Daily Crypto Digitals Versus Deribit 1-Day Options

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior notes: [[od_pricing_model_form_findings]] · [[od_v4_calibration_gate_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Headline

Phase 0 data verdict: **PARTIAL PASS: degraded Phase 1 ran, but no deployable verdict**.

This is a market-vs-market scoping run, not another attempt to out-price Polymarket with our own Gaussian or jump model. The trade idea is: compare the Polymarket daily crypto UP token to a Deribit option-market-implied digital on the same asset and date, then only care about the basis after two-venue costs and settlement mismatch.

The result is useful but not green-lighting. Free Deribit public data does expose historical option OHLC through `public/get_tradingview_chart_data` for the relevant expired BTC/ETH strikes, so a degraded single-window basis pass can be constructed. But the endpoints that would make this execution-grade - historical mark/IV and expired-instrument book summary - are not available for those expired instruments through the checked public endpoints. The PM daily LOB capture also covers only about 12 hours of the 24-hour market, not the full path into PM resolution.

Plain-English verdict: **start forward dual-capture before believing this RV trade.** The observed one-window basis is sometimes larger than the known cost floor, but the Deribit leg here is OHLC-close based, not bid/ask or mark-IV based, and the PM/Deribit settlement-time mismatch is still the main risk.

CI note: this scoping run has only one independent PM daily resolution for BTC and one for ETH. Hourly rows inside that day describe basis persistence, but they are not independent PnL samples, so this note deliberately avoids a fake confidence interval and specifies the forward sample needed for CI.

## Phase 0 Data Availability

The gate asked whether we can reconstruct the Deribit 1-day option chain for the captured PM daily window. The answer is a partial yes:

- Local artifacts contain the PM daily surface and old DVOL anchor, but no reusable Deribit per-instrument mark/IV history.
- Deribit's `get_mark_price_history` and `get_book_summary_by_instrument` checks fail for the expired target instruments.
- Deribit's `get_tradingview_chart_data` does return hourly OHLC for strikes bracketing the PM BTC/ETH strike, including volume/cost fields. This is enough for a *scoping* call-spread basis, not enough for executable bid/ask PnL.

| source | check / artifact | status | detail |
| --- | --- | --- | --- |
| local_artifact | data/analysis/csv_outputs/options_delta/od_pricing_model_form_deribit.csv | csv_report | CSV report, not reusable per-option history. |
| local_artifact | data/analysis/od_pricing_model_form_deribit.parquet | dvol_index | DVOL only; not a per-option surface. |
| deribit_api | BTC get_instruments expired=true | pass_empty_target_date | returned 48 expired option instruments; 0 near PM date |
| deribit_api | ETH get_instruments expired=true | pass_empty_target_date | returned 38 expired option instruments; 0 near PM date |
| deribit_api | BTC-28MAY26-74500-C get_book_summary_by_instrument | fail_expired_or_unavailable | {'code': -32602, 'data': {'reason': 'instrument is not open', 'param': 'instrument_name'}, 'message': 'Invalid params'} |
| deribit_api | BTC-28MAY26-74500-C get_mark_price_history | fail_expired_or_unavailable | {'code': -32602, 'data': {'reason': 'instrument is not active', 'param': 'instrument_name'}, 'message': 'Invalid params'} |
| deribit_api | BTC-28MAY26-74500-C tradingview_chart_data | pass_chart_ohlc | 25 hourly bars, total volume 44.60 |
| deribit_api | BTC-28MAY26-75000-C get_book_summary_by_instrument | fail_expired_or_unavailable | {'code': -32602, 'data': {'reason': 'instrument is not open', 'param': 'instrument_name'}, 'message': 'Invalid params'} |
| deribit_api | BTC-28MAY26-75000-C get_mark_price_history | fail_expired_or_unavailable | {'code': -32602, 'data': {'reason': 'instrument is not active', 'param': 'instrument_name'}, 'message': 'Invalid params'} |
| deribit_api | BTC-28MAY26-75000-C tradingview_chart_data | pass_chart_ohlc | 25 hourly bars, total volume 60.80 |
| deribit_api | BTC-28MAY26-75500-C get_book_summary_by_instrument | fail_expired_or_unavailable | {'code': -32602, 'data': {'reason': 'instrument is not open', 'param': 'instrument_name'}, 'message': 'Invalid params'} |
| deribit_api | BTC-28MAY26-75500-C get_mark_price_history | fail_expired_or_unavailable | {'code': -32602, 'data': {'reason': 'instrument is not active', 'param': 'instrument_name'}, 'message': 'Invalid params'} |
| deribit_api | BTC-28MAY26-75500-C tradingview_chart_data | pass_chart_ohlc | 25 hourly bars, total volume 8.80 |
| deribit_api | ETH-28MAY26-2050-C get_book_summary_by_instrument | fail_expired_or_unavailable | {'code': -32602, 'data': {'reason': 'instrument is not open', 'param': 'instrument_name'}, 'message': 'Invalid params'} |
| deribit_api | ETH-28MAY26-2050-C get_mark_price_history | fail_expired_or_unavailable | {'code': -32602, 'data': {'reason': 'instrument is not active', 'param': 'instrument_name'}, 'message': 'Invalid params'} |
| deribit_api | ETH-28MAY26-2050-C tradingview_chart_data | pass_chart_ohlc | 25 hourly bars, total volume 1037.00 |
| deribit_api | ETH-28MAY26-2075-C get_book_summary_by_instrument | fail_expired_or_unavailable | {'code': -32602, 'data': {'reason': 'instrument is not open', 'param': 'instrument_name'}, 'message': 'Invalid params'} |
| deribit_api | ETH-28MAY26-2075-C get_mark_price_history | fail_expired_or_unavailable | {'code': -32602, 'data': {'reason': 'instrument is not active', 'param': 'instrument_name'}, 'message': 'Invalid params'} |
| deribit_api | ETH-28MAY26-2075-C tradingview_chart_data | pass_chart_ohlc | 25 hourly bars, total volume 282.00 |
| deribit_api | ETH-28MAY26-2100-C get_book_summary_by_instrument | fail_expired_or_unavailable | {'code': -32602, 'data': {'reason': 'instrument is not open', 'param': 'instrument_name'}, 'message': 'Invalid params'} |
| deribit_api | ETH-28MAY26-2100-C get_mark_price_history | fail_expired_or_unavailable | {'code': -32602, 'data': {'reason': 'instrument is not active', 'param': 'instrument_name'}, 'message': 'Invalid params'} |
| deribit_api | ETH-28MAY26-2100-C tradingview_chart_data | pass_chart_ohlc | 25 hourly bars, total volume 574.00 |

Read: this is why the run proceeds to a degraded Phase 1 instead of stopping entirely. It is not a clean Phase 0 pass because the Deribit data is chart OHLC, not a historical option book/mark surface.

## Phase 1 Degraded Single-Window Basis

The captured PM daily markets are:

- BTC: `bitcoin-up-or-down-on-may-28-2026`, strike from Binance at PM window open = about `$75,326`, PM window `2026-05-27 16:00` to `2026-05-28 16:00` UTC.
- ETH: `ethereum-up-or-down-on-may-28-2026`, strike about `$2,071`, same PM window.

Practical replication example: for BTC, the PM strike sits between Deribit `BTC-28MAY26-75000-C` and `BTC-28MAY26-75500-C`. A cash digital is approximated as:

```text
Deribit digital ≈ spot * (call_price_75000 - call_price_75500) / 500
basis = PM UP mid - Deribit digital
```

The multiplication by spot converts Deribit option premium from coin units into a USD call value before taking the call-spread slope. This is still an approximation: the call spread is `$500` wide for BTC and `$25` wide for ETH, so it measures the average digital over the bracket, not an infinitesimal binary at the exact PM strike.

| asset | rows | PM coverage UTC | spread width bp | PM UP mid | Deribit digital | mean basis | median basis | p95 abs basis | basis > 0 | rep error | known cost floor | abs basis minus cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC | 7 | 05-27 21:19 -> 05-28 03:59 | 66.4 | 18.50c | 7.46c | 11.04c | 10.08c | 14.65c | 100.00% | 6.28c | 7.38c | 3.67c |
| ETH | 7 | 05-27 21:18 -> 05-28 03:59 | 120.7 | 9.21c | 5.48c | 3.73c | 4.89c | 7.60c | 85.71% | 2.51c | 3.14c | 2.13c |

Column read: `p95 abs basis` is the 95th percentile absolute PM-minus-Deribit gap during the overlapped hourly observations. `rep error` is a proxy from comparing the tight call spread with a wider neighboring spread. `known cost floor` includes PM taker fee at the PM mid, a small Deribit fee proxy, and the replication-error proxy; it does **not** include Deribit bid/ask spread because historical expired books were unavailable.

![PM vs Deribit daily basis](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_rv_deribit_daily_basis.png)

Read: this is visually worth forward-capturing, but not tradable from this artifact. The basis is not a powered result: it is one PM daily resolution, hourly Deribit chart observations, partial PM LOB coverage, and no Deribit bid/ask.

## Killer 1: Call-Spread Replication Error

The replicated digital is fragile near the strike because a binary payoff is a slope, and the slope estimate depends on the call-spread width. The BTC tight spread is `$500` wide, about `66bp` of strike. The ETH tight spread is `$25` wide, about `121bp` of strike. That is not tiny for a 1-day digital.

In this pass, `rep error` compares the tight bracket with a wider neighboring bracket. It is a proxy, not a complete error model. A real capture should store multiple strikes around the PM strike so the digital can be estimated from a local slope and a curvature/error band, not just one call spread.

## Killer 2: Settlement-Timing / Reference Mismatch

PM daily resolves at the PM window close. The Deribit date option behaves like an exchange-expiry instrument and the chart becomes floor-stale after the morning expiry region. That creates an approximately eight-hour mismatch versus the PM 16:00 UTC resolution.

| asset | strike | spot at Deribit proxy | spot at PM close | Deribit UP? | PM UP? | mismatch? | 8h drift bp | binary slip if mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC | 75326.01 | 73498.00 | 72928.73 | False | False | False | -77.8 | 0.00c |
| ETH | 2071.38 | 1995.56 | 1990.65 | False | False | False | -24.6 | 0.00c |

Read: there was no direction mismatch in this one BTC/ETH sample, but the 8-hour spot drift is large enough that this cannot be assumed away. If the Deribit proxy and PM close land on opposite sides of the strike, the "hedged" basis can still lose a full binary unit.

## Feasibility Verdict

The best one-window `|basis|-known-cost-floor` read is 3.67c. That is **not** an edge proof. It only says the market-vs-market RV idea is worth a forward data capture because the observed basis is not obviously dominated by the pieces we can measure. The unmeasured pieces - Deribit bid/ask, queue/leg execution, and settlement-reference mismatch - are exactly the pieces that decide whether this is real.

Do **not** trade this from the historical chart artifact. The right next step is a forward dual-capture that stores both venues' executable state at the same timestamps.

## Phase 2 Forward Dual-Capture Spec

Capture BTC and ETH only. SOL has no Deribit analogue.

Minimum cadence:

- Every 30 seconds during each PM daily crypto window; 5 seconds in the final hour and whenever PM UP is between 40c and 60c.
- PM: UP/DOWN top-of-book bid/ask/size, top-5 depth, trades, market slug, token IDs, outcome index, Chainlink/Pyth reference fields if obtainable, and final PM resolution timestamp/reference.
- Deribit: option book summaries or order books for the nearest expiry and at least five strikes around PM strike on each side; mark price, mark IV, bid/ask IV, underlying index, futures/perp index, and volume/open interest. Store raw instrument names because strike grids roll.
- Binance/Pyth/Chainlink: spot/index snapshots for settlement-reference diagnosis. This is required; otherwise the RV hedge may not actually cancel binary resolution risk.

Validation design:

- Treat one asset-day as one independent window. Hourly rows inside a day are basis observations, not independent PnL samples.
- Need at least about 60-100 independent asset-days before calling a 1-2c net basis positive with a useful 95% CI. BTC and ETH together produce at most two windows per calendar day, so this is roughly 30-50 calendar days of clean dual capture.
- Evaluate both directions: short PM / long Deribit-digital when PM is rich, and long PM / short Deribit-digital when PM is cheap. Include PM taker/maker route, Deribit bid/ask and fees, call-spread replication width, leg latency, and settlement mismatch PnL.
- Execution capital is two-venue: PM collateral plus Deribit option margin/premium. This gives up the PM-only simplicity, but it is the first version of OD that has a real liquid external fair-value instrument.

## Decision

OD-RV daily is **not closed** and **not validated**. It is a scoping partial-pass: the one captured PM daily window plus public Deribit chart OHLC can show a basis, but not an executable market-vs-market edge. Start forward dual-capture if we want to test it seriously; do not spend more time on historical DVOL/model-only proxies.

## Outputs

- Data gate CSV: `data/analysis/csv_outputs/options_delta/od_rv_deribit_daily_data_gate.csv`
- Basis CSV: `data/analysis/csv_outputs/options_delta/od_rv_deribit_daily_basis.csv`
- Summary CSV: `data/analysis/csv_outputs/options_delta/od_rv_deribit_daily_summary.csv`
- Settlement mismatch CSV: `data/analysis/csv_outputs/options_delta/od_rv_deribit_daily_settlement_mismatch.csv`
- Deribit chart parquet: `data/analysis/od_rv_deribit_daily_deribit_charts.parquet`
