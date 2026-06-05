# PM Financial-Binary Gate 0 Universe And Capacity Map

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Related: [[od_cross_asset_updown_scoping]] · [[od_pricing_model_form_findings]] · [[mm_deployable_cells_findings]]
> CSV: `data/analysis/csv_outputs/options_delta/od_cross_asset_gate0_universe_map.csv`

## Headline

Ranked by non-incumbent headroom times clean-reference availability, **only crypto-daily clears Gate 0 for a future external-fair-value test**. That is a universe/capacity pass, not permission to ignore settlement: the parallel OD-RV settlement check parks literal PM daily BTC/ETH vs Deribit daily because PM settles at 16:00 UTC and Deribit expires at 08:00 UTC. Index up/down and single-stock up/down are clean but sub-scale under the stated volume/headroom minimum. Close-above/price-band has a lot of volume, but the current volume is mostly crypto hit/barrier style threshold markets rather than clean terminal digitals, so it should not reopen a homemade pricing-model branch. True financial negative-risk baskets were sparse one-offs in this scrape.

Decision rule was pre-registered before the script ran: a family merits a future fair-value test only if daily run-rate volume is at least `$50,000`, non-top3 headroom is at least `$25,000` per day, and a clean external reference exists. The 90-day recent window is `2026-03-04` through `2026-06-02`. This note is only Gate 0; it does not claim any fair-value edge.

## Ranked Families

Unit of observation is a Polymarket binary market row. `run-rate volume` uses current live 24h volume when live markets exist; otherwise it uses recent 90-day resolved volume divided by 90. `non-top3 headroom` is run-rate volume multiplied by the K5-style non-top3 maker-share proxy. Exact concentration comes from the K5 stress wallet-market cache, joined through local Gamma condition IDs; newer markets that are not in the local metadata inherit the family median from matched cached markets.

| rank | family | decision | run-rate volume/day | non-top3 headroom/day | top3 share | median live spread | clean ref |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | crypto_daily | MERITS-FAIR-VALUE-TEST | $304,610 | $161,044 | 51.8% | 2.00c | yes |
| 2 | index_up_down | DO-NOT-PURSUE: thin | $27,884 | $18,021 | 52.7% | 1.00c | yes |
| 3 | single_stock_up_down | DO-NOT-PURSUE: thin | $6,994 | $3,567 | 57.3% | 7.00c | yes |
| 4 | close_above_price_band | DO-NOT-PURSUE: reference/capacity gate failed | $6,494,053 | $3,709,658 | 56.1% | 1.00c | no |
| 5 | neg_risk_baskets | DO-NOT-PURSUE: thin | $794 | $492 | 52.1% | 2.50c | no |

Read: the cheapest decisive filter does not say crypto-daily has edge. It says crypto-daily has enough volume, a plausible external reference, and enough non-incumbent fill flow to justify one later fair-value/capture gate. Everything else stops here unless the live universe changes.

## Sample Markets And Settlement Sources

The exact settlement timestamp used here is Gamma `endDate`/`closedTime`, stored per market in the detail CSV. The source string is the Gamma `resolutionSource` field when populated, otherwise the market description is the only source text.

| family | sample | state | volume field | settlement UTC | resolution source | concentration |
| --- | --- | --- | --- | --- | --- | --- |
| close_above_price_band | bitcoin-above-on-june-2-2026 | live | $338,049 | 2026-06-02T16:00:00 | description-only | family proxy from k5 stress cache |
| close_above_price_band | what-price-will-bitcoin-hit-in-june-2026 | live | $326,010 | 2026-07-01T04:00:00 | description-only | family proxy from k5 stress cache |
| close_above_price_band | bitcoin-above-on-june-2-2026 | live | $324,816 | 2026-06-02T16:00:00 | description-only | family proxy from k5 stress cache |
| crypto_daily | bitcoin-up-or-down-on-june-2-2026 | live | $244,567 | 2026-06-02T16:00:00 | https://www.binance.com/en/trade/BTC_USDT | family proxy from k5 stress cache |
| crypto_daily | ethereum-up-or-down-on-june-2-2026 | live | $52,837 | 2026-06-02T16:00:00 | https://www.binance.com/en/trade/ETH_USDT | family proxy from k5 stress cache |
| crypto_daily | solana-up-or-down-on-june-2-2026 | live | $6,588 | 2026-06-02T16:00:00 | https://www.binance.com/en/trade/SOL_USDT | family proxy from k5 stress cache |
| index_up_down | spx-up-or-down-on-june-2-2026 | live | $27,884 | 2026-06-02T20:00:00 | https://www.wsj.com/market-data/stocks | family proxy from k5 stress cache |
| index_up_down | spx-up-or-down-on-june-1-2026 | recent | $126,420 | 2026-06-02T00:15:40 | https://www.wsj.com/market-data/stocks | family proxy from k5 stress cache |
| index_up_down | spx-up-or-down-on-april-29-2026 | recent | $401,022 | 2026-04-30T00:13:28 | https://www.wsj.com/market-data/stocks | family proxy from k5 stress cache |
| neg_risk_baskets | spx-close-dec-2026 | live | $16,283 | 2026-12-31T21:00:00 | https://finance.yahoo.com/quote/%5EGSPC/history | k5 stress cache condition join |
| neg_risk_baskets | spx-close-dec-2026 | live | $3,145 | 2026-12-31T21:00:00 | https://finance.yahoo.com/quote/%5EGSPC/history | k5 stress cache condition join |
| neg_risk_baskets | spx-close-dec-2026 | live | $2,987 | 2026-12-31T21:00:00 | https://finance.yahoo.com/quote/%5EGSPC/history | k5 stress cache condition join |
| single_stock_up_down | meta-up-or-down-on-june-2-2026 | live | $2,094 | 2026-06-02T20:00:00 | https://pythdata.app/explore/Equity.US.META%2FUSD | family proxy from k5 stress cache |
| single_stock_up_down | nvda-up-or-down-on-june-2-2026 | live | $2,062 | 2026-06-02T20:00:00 | https://pythdata.app/explore/Equity.US.NVDA%2FUSD | family proxy from k5 stress cache |
| single_stock_up_down | googl-up-or-down-on-june-2-2026 | live | $1,618 | 2026-06-02T20:00:00 | https://pythdata.app/explore/Equity.US.GOOGL%2FUSD | family proxy from k5 stress cache |

## External Reference Read

| family | clean? | reference read | caveat |
| --- | --- | --- | --- |
| crypto_daily | yes | Binance settlement source on current PM daily templates; BTC/ETH have Deribit listed options for forward capture, SOL has no Deribit analogue. | Gate 0 clears volume/capacity, but the parallel OD-RV settlement check parks literal 16:00 UTC PM daily vs 08:00 UTC Deribit daily. Use settlement-aligned BTC/ETH windows or find a clean 16:00 external surface. |
| index_up_down | yes | SPX/NDX PM templates settle on official closes; SPX/NDX or ETF/futures options exist for external digital/call-spread checks. | Future test needs a declared data path for option/futures quotes; the PM settlement timestamp is clean. |
| single_stock_up_down | yes | Liquid single names have official closes and listed OCC equity options. | Corporate actions, halts, and official-close handling must be encoded; current PM flow is the main blocker. |
| close_above_price_band | no | The terminal close-above subset has listed-option analogues, but most volume comes from crypto hit/threshold baskets that are path-dependent barrier claims. | Do not let high crypto threshold volume reopen a homemade barrier model under Gate 0. |
| neg_risk_baskets | no | True financial neg-risk range baskets surfaced as one-offs, not recurring liquid templates. | Treat as PM-internal consistency research only if a liquid recurring financial basket appears. |

## Public Sources Checked

- [Polymarket market data overview](https://docs.polymarket.com/market-data/overview) for Gamma, CLOB, and Data API surface area.
- [Polymarket fees](https://docs.polymarket.com/trading/fees) for fee formula, category taker rates, and maker-rebate framing.
- [Deribit public instruments API](https://docs.deribit.com/api-reference/market-data/public-get_instruments) for BTC/ETH listed option availability and the live/expired instrument split.
- [Alpha Vantage documentation](https://www.alphavantage.co/documentation/) for equity daily data and historical options API availability/premium caveats.
- [Cboe SPX options specifications](https://www.cboe.com/tradable_products/sp_500/spx_options/specifications/) for listed SPX option surface availability.
- [OCC equity options product specifications](https://www.theocc.com/Clearance-and-Settlement/Clearing/Equity-Options-Product-Specifications) for listed single-stock option structure.
- [Yahoo Finance historical prices help](https://help.yahoo.com/kb/finance/historical-prices-sln2311.html) for settlement-source availability and licensing caveats on downloadable historical prices.

## Family Decisions

**crypto-daily:** **MERITS-FAIR-VALUE-TEST on Gate 0 only**, but the later test must obey settlement matching. BTC/ETH have a Deribit options analogue and clean Binance settlement source on current daily PM templates, yet [[od_rv_deribit_daily_capture_findings]] parks literal 16:00 UTC PM daily vs 08:00 UTC Deribit daily. The actionable route is settlement-aligned BTC/ETH capture or a clean 16:00 external surface; SOL should be excluded or treated as spot/futures-only until a real options surface exists. This is not permission to reopen crypto-4h absolute pricing.

**index up/down:** **DO-NOT-PURSUE for now: thin.** SPX/NDX settlement is clean and listed options/futures exist, but current live run-rate and non-top3 headroom both miss the minimum. Recheck if the daily SPX/NDX templates grow above the headroom threshold; no Gate 1 spend now.

**single-stock up/down:** **DO-NOT-PURSUE for now: thin.** The external reference story is clean for liquid names, but live flow and non-top3 headroom are below the minimum. Recheck only if the daily stock templates grow by an order of magnitude.

**close-above/price-band:** **DO-NOT-PURSUE as a family despite volume.** The high-volume rows are mostly crypto threshold/hit markets, which are path-dependent barrier claims. Terminal close-above subsets can be clean, but their current flow is not what drives the family totals. Do not use this as a backdoor into another handcrafted model-form test.

**neg-risk baskets:** **DO-NOT-PURSUE: thin and not clean enough.** The true financial neg-risk rows found were sparse one-offs. If a recurring, liquid financial range basket appears, it should first be an internal-consistency/merge-split accounting task, not OD fair value.

## Guardrails

- No pricing model, fair-value residual, or PnL test was run here.
- Maker concentration is a capacity proxy, not proof that a faster or better-priced entrant cannot steal share.
- Markets missing from the local Gamma metadata/K5 cache are not assigned exact maker concentration; those rows use family proxies.
- The close-above/price-band volume is not a clean terminal digital volume estimate because barrier/hit templates dominate the scrape.

## Outputs

- Family CSV: `data/analysis/csv_outputs/options_delta/od_cross_asset_gate0_universe_map.csv`
- Market detail CSV: `data/analysis/csv_outputs/options_delta/od_cross_asset_gate0_market_detail.csv`
- Reference checks CSV: `data/analysis/csv_outputs/options_delta/od_cross_asset_gate0_reference_checks.csv`
- Script: `scripts/od_cross_asset_gate0_universe_map.py`
