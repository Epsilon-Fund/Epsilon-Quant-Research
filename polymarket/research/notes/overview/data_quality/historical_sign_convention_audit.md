---
title: Historical Sign Convention Audit
created: 2026-05-27
status: closed
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
tags:
  - data-quality
  - sign-convention
  - tfi
  - audit
  - research
---

# Historical Sign Convention Audit

> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]

Generated: 2026-05-27

## Summary

This audit checks whether the historical `maker_side` field needs global inversion before TFI work. The source and empirical checks support the current `historical_to_aggressor()` mapping: `maker_side` is the passive maker token side, and token-side aggressor is the inverse. No normalization code change or downstream sign rerun was required by this audit.

## Executive Decision

`historical_to_aggressor()` is correct for the audited historical data. No
family showed evidence that the local `maker_side` field should be globally
inverted. The crypto slice has a small unconditional upward drift, so its
literal `P(up | maker_side=BUY)` is slightly above 0.5, but
`P(up | maker_side=SELL)` is much higher. That side contrast is the expected
pattern when `maker_side=BUY` is a passive bid hit by an aggressive seller and
`maker_side=SELL` is a passive ask lifted by an aggressive buyer.

No change was made to `lib/trade_sign_normalization.py`; downstream TFI
baseline analyses do not need a sign rerun for this audit.

## Documentation And Source Findings

- Repo docs identify the historical fill layer as a warproxxx seed plus
  Goldsky subgraph delta, consumed as `raw_trades`.
- The local Goldsky ingestion queries `orderFilledEvents` fields
  `makerAssetId`, `takerAssetId`, `makerAmountFilled`, and
  `takerAmountFilled`; it then sets `maker_side = BUY` exactly when
  `makerAssetId == '0'`.
- The seed builder uses the same rule. It treats `makerAssetId == '0'` as the
  maker paying USDC and receiving the outcome token, so `maker_side=BUY`.
- The direct Polygon decoder for newer logs also maps encoded V2 `side == 0`
  to `maker_asset_id='0'`, `taker_asset_id=token_id`, and then writes
  `maker_side=BUY`.
- I did not find a Goldsky page that defines a legacy subgraph column named
  `maker_side`; the local field is repo-derived. Goldsky's current
  Polymarket dataset docs describe `polymarket.order_filled` as a per-order
  fill dataset with `side` as order side and `order_type` as maker/taker.
  Their copy-trader guide is more explicit for V2: encoded side is the maker's
  side, and takers take the opposite side.
- Polymarket's on-chain order docs define V1 `makerAssetId`: if it is `0`, the
  order is a BUY giving pUSD/USDC for outcome tokens; `takerAssetId == 0` is a
  SELL receiving pUSD/USDC.

External references checked:

- [Goldsky: Indexing Polymarket](https://docs.goldsky.com/chains/polymarket)
- [Goldsky: Order Filled data source](https://app.goldsky.com/data-sources/dataset/order_filled)
- [Goldsky: Build a Polymarket copy-trader](https://docs.goldsky.com/compose/guides/build-a-polymarket-copy-trader)
- [Polymarket: Onchain Order Info](https://docs.polymarket.com/trading/orders/overview)

## Empirical Method

Sample source: the four fill slices used by the existing TFI magnitude analysis.
For each fill, the audit derived the outcome token as
`taker_asset_id` when `maker_asset_id='0'`, else `maker_asset_id`.

Historical true L2 book mid is not materialized in this repo. The empirical
audit therefore uses a same-token transaction-price proxy:

1. Build per `(market_id, outcome_token_id, timestamp)` VWAP from historical
   fills.
2. For each fill at time `t`, find the last same-token VWAP at or before
   `t - 30s`.
3. Find the first same-token VWAP at or after `t + 30s`.
4. Require same-token price data at or after `t + 60s`.
5. Require all lookup gaps to be no more than 300 seconds.
6. Compute `delta_price = price(t+30s proxy) - price(t-30s proxy)`.

Confidence intervals are row-level bootstrap intervals with
1,000 resamples and seed `20260527`. The z-score is
for `P(up | maker_side=BUY)` against the null `p=0.5`.

## Results

| family | eligible fills | P(up given BUY) 95% CI | P(up given SELL) 95% CI | SELL-BUY up contrast 95% CI | z(BUY vs 0.5) | interpretation | decision |
|---|---:|---:|---:|---:|---:|---|---|
| ai_product | 107,190 | 38.13% [37.77%, 38.47%] | 50.38% [49.92%, 50.87%] | 12.24% [11.61%, 12.89%] | -62.62 | A | historical_to_aggressor_correct |
| daily_crypto_up_down | 1,562,526 | 50.66% [50.57%, 50.74%] | 57.83% [57.59%, 58.07%] | 7.17% [6.92%, 7.43%] | 15.52 | mixed_A_drift | historical_to_aggressor_correct |
| daily_equity_index | 117,038 | 41.70% [41.37%, 42.03%] | 51.81% [51.29%, 52.34%] | 10.11% [9.50%, 10.71%] | -47.70 | A | historical_to_aggressor_correct |
| sports_game_lines | 178,046 | 39.27% [39.03%, 39.52%] | 44.68% [44.13%, 45.21%] | 5.41% [4.80%, 6.02%] | -81.46 | A | historical_to_aggressor_correct |

Down-move probabilities and mean deltas:

| family | P(down given BUY) 95% CI | P(down given SELL) 95% CI | P(flat given BUY) | P(flat given SELL) | mean delta after BUY | mean delta after SELL |
|---|---:|---:|---:|---:|---:|---:|
| ai_product | 49.30% [48.97%, 49.68%] | 37.87% [37.42%, 38.37%] | 12.57% | 11.75% | -0.0042 | 0.0031 |
| daily_crypto_up_down | 48.72% [48.63%, 48.80%] | 41.19% [40.95%, 41.44%] | 0.63% | 0.98% | 0.0077 | 0.0322 |
| daily_equity_index | 49.40% [49.06%, 49.72%] | 40.22% [39.73%, 40.78%] | 8.90% | 7.97% | -0.0042 | 0.0064 |
| sports_game_lines | 41.60% [41.34%, 41.85%] | 39.38% [38.87%, 39.92%] | 19.13% | 15.94% | -0.0016 | 0.0012 |

Interpretation keys:

- `A`: literal framework A; `P(up | maker_side=BUY)` is below 0.5.
- `mixed_A_drift`: `P(up | maker_side=BUY)` is above 0.5, but the SELL minus
  BUY up contrast is positive and significant, indicating family-level upward
  drift rather than inverted side semantics.
- `B`: inverted; BUY is significantly more upward than SELL.
- `C`: `P(up | maker_side=BUY)` is statistically indistinguishable from 0.5.
- `mixed`: significant but not cleanly classifiable.

## Data Windows

| family | source window | eligible audit window | markets | outcome tokens | avg pre lookup gap | avg post lookup gap |
|---|---|---|---:|---:|---:|---:|
| ai_product | 2025-08-07 22:06:24 -> 2026-04-23 23:59:34 | 2025-08-08 06:06:19 -> 2026-04-23 23:54:46 | 100 | 200 | 83.8s | 85.2s |
| daily_crypto_up_down | 2026-04-19 23:39:38 -> 2026-04-24 00:00:00 | 2026-04-21 03:59:40 -> 2026-04-23 23:59:00 | 250 | 500 | 2.2s | 2.2s |
| daily_equity_index | 2026-03-05 00:19:57 -> 2026-04-23 22:31:08 | 2026-03-10 19:50:57 -> 2026-04-23 20:04:42 | 100 | 200 | 57.7s | 58.4s |
| sports_game_lines | 2025-06-27 18:47:37 -> 2026-04-24 00:00:00 | 2025-09-30 21:29:49 -> 2026-04-23 23:58:58 | 100 | 200 | 40.3s | 39.5s |

## Per-Family Read

- `ai_product`: clear framework A. BUY-maker prints are followed by up moves
  only 38.13%
  of the time; SELL-maker prints are directionally higher.
- `daily_equity_index`: clear framework A, with a 10.1 percentage point
  SELL-minus-BUY up contrast.
- `sports_game_lines`: framework A by `P(up | BUY)` and by mean delta; flats
  are common, but the SELL-minus-BUY contrast is still positive.
- `daily_crypto_up_down`: mixed under the literal framework because crypto
  drifted upward in the analyzed window. The key diagnostic is that SELL-maker
  prints are 7.2 percentage points more likely to be followed by up moves than
  BUY-maker prints, which rejects the inverted-aggressor interpretation.

## Decision

Documentation and code provenance both say the local historical `maker_side`
field is the maker's token side. The empirical side contrast agrees across all
four audited families. The correct aggressor conversion remains:

- `maker_side=BUY` -> aggressor `SELL`
- `maker_side=SELL` -> aggressor `BUY`

No helper update or downstream TFI rerun is required.
