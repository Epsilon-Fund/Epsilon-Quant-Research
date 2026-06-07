---
title: "Dali Sign Convention Findings"
created: 2026-06-05
status: active
owner: justin
project: polymarket
para: project
hubs:
  - COWORK
tags:
  - research
  - dali
---
# Dali Sign Convention Findings
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


## Summary

- Scope: Dali Sign Convention Findings in the Dali research lineage area.
- Existing takeaway/status: A1 sign-convention note for live CLOB and historical fill semantics. It supports treating live `last_trade_price.side` as token-side aggressor direction, with caveats for YES/NO token interpretation.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
Generated: 2026-05-23

## Sources

- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T100728Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T110728Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T120728Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T130728Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T140728Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T150728Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T160729Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T170728Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T180729Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T190729Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T200730Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T210730Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T220730Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260527T230730Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260528T000730Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260528T010730Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260528T020730Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260528T030731Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260528T040731Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260528T050731Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260528T060731Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260528T070731Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260528T080731Z.jsonl`
- `data/live_clob/block_a0/block_a0_20260528_morning/block_a0_20260528_morning_20260528T090731Z.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/block_a0b_replacements_v2_20260527_20260527T211912Z.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/block_a0b_replacements_v2_20260527_20260527T221913Z.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/block_a0b_replacements_v2_20260527_20260527T231912Z.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/block_a0b_replacements_v2_20260527_20260528T001913Z.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/block_a0b_replacements_v2_20260527_20260528T011913Z.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/block_a0b_replacements_v2_20260527_20260528T021913Z.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/block_a0b_replacements_v2_20260527_20260528T031913Z.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/block_a0b_replacements_v2_20260527_20260528T041913Z.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/block_a0b_replacements_v2_20260527_20260528T051913Z.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/block_a0b_replacements_v2_20260527_20260528T061914Z.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/block_a0b_replacements_v2_20260527_20260528T071913Z.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/block_a0b_replacements_v2_20260527_20260528T081914Z.jsonl`

## Live CLOB Inference

- Total `last_trade_price` events inspected: 8158
- Classified from book transition: 847 (10.4%)
- Ambiguous: 816 (10.0%)
- Unclassifiable: 6495 (79.6%)
- Minimum classifiable trades required to establish normalization: 50

Sign normalization is established from this sample.

Conditional checks if classified trades exist:

- `P(reported.side == BUY | inferred aggressor == BUY)`: 99.9%
- `P(reported.side == BUY | inferred aggressor == SELL)`: 1.7%

`live_to_aggressor(..., semantics='aggressor')` may be used only if the classified-trade conditional probabilities support it.

## Historical Fill Semantics

Historical `maker_side` is the passive maker's token side. The aggressor is on
the opposite token side:

- `maker_side == BUY` means the maker bought tokens, so the taker/aggressor sold.
- `maker_side == SELL` means the maker sold tokens, so the taker/aggressor bought.

The helper `historical_to_aggressor()` implements this inversion. Current sanity
mapping:

- historical `BUY` -> `SELL`
- historical `SELL` -> `BUY`

## YES/NO Token Nuance

This audit is token-side, not market-direction-side. Buying YES and selling NO
can be economically similar for the underlying question, but they are different
CLOB token actions. Dali should keep token-side aggressor as the normalized
microstructure field, then map to market-direction only when outcome labels are
available and explicitly joined.

## Sample Rows

| received_at | reported_side | inferred_aggressor | trade_price | prior_bid | prior_ask | transaction_hash |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-05-27T10:13:29.218Z | SELL | UNCLASSIFIABLE | 0.828 | 0.828 | 0.829 | 0x1d71a8fb7cbcb6ebec057a1d74e0d6d3e0ee97c0513d707bb6ba6487484731d9 |
| 2026-05-27T10:13:42.578Z | BUY | UNCLASSIFIABLE | 0.172 | 0.171 | 0.172 | 0xbcc0a0de7d16cf2b4a76e721fbe1f59214821d73ed17d5680ac3ce8d61b25337 |
| 2026-05-27T10:14:11.478Z | BUY | BUY | 0.174 | 0.173 | 0.174 | 0x6253b17df7bce6559205dd62e325812e88c206f17c4ce45c73c73b7f7b4fe259 |
| 2026-05-27T10:16:02.004Z | SELL | UNCLASSIFIABLE | 0.697 | 0.696 | 0.697 | 0x4e784348d2fb5b9c1b657d91634cfe9b80ddcfd12b6ce0a6281289faec07fc45 |
| 2026-05-27T10:16:03.014Z | BUY | UNCLASSIFIABLE | 0.697 | 0.697 | 0.698 | 0xb0ea6fedecf6546b0566f67828eeaf903be3ab2e7c08d7961054888d4b604e43 |
| 2026-05-27T10:16:06.857Z | BUY | UNCLASSIFIABLE | 0.174 | 0.173 | 0.174 | 0x77b95ba7067f0789569ab4945178b27e62d887c88f6c359b66598f730dae2699 |
| 2026-05-27T10:16:35.941Z | BUY | BUY | 0.174 | 0.173 | 0.174 | 0xb4dffce1ac2656f0b4372520cc249fe14ee2944605bd5995c6628a883e4f97ff |
| 2026-05-27T10:16:35.938Z | BUY | UNCLASSIFIABLE | 0.172 | 0.171 | 0.172 | 0xec685c8d80d0f2249bf607fb4b56c8f3efca6d1b5dfdd9a78df9a8ffb083ff01 |
| 2026-05-27T10:19:19.755Z | BUY | BUY | 0.698 | 0.696 | 0.698 | 0xc3fa799d6896f06a6e2981b4d6c10f1b519dc2e45c71b9630a8cef8e61777fed |
| 2026-05-27T10:19:19.905Z | BUY | UNCLASSIFIABLE | 0.698 | 0.696 | 0.698 | 0xcbe00156dec112032ed9c0ea25a3b54250e166c0dc04f860ddd978fcbbd99862 |
| 2026-05-27T10:19:20.089Z | BUY | UNCLASSIFIABLE | 0.699 | 0.696 | 0.699 | 0x3dcad6f4140d4a80cdcba12ef621b003bf0f3c432481cd8b4f22899068248ee9 |
| 2026-05-27T10:19:38.249Z | BUY | UNCLASSIFIABLE | 0.172 | 0.171 | 0.172 | 0x95a0ee91db77a4b8664520082f3aaaa420f7dc467ba74dc928ba2b9e55018bfe |
| 2026-05-27T10:24:09.532Z | BUY | BUY | 0.786 | 0.783 | 0.786 | 0x27ca268d8606f1eb2d5551df8f0fce529d7d99393e36097035c330c3ceb6c051 |
| 2026-05-27T10:24:09.809Z | BUY | UNCLASSIFIABLE | 0.82 | 0.81 | 0.82 | 0x693ff3d27a3b1410e41fbeee32fdf3aea24b0fd4125f2b67a2449228222b4157 |
| 2026-05-27T10:24:50.694Z | BUY | UNCLASSIFIABLE | 0.172 | 0.171 | 0.172 | 0xd34663cfaa0ecc77f00e813871d3c46250a1ebc75ce5427efb5c9fe94ef97e9f |
| 2026-05-27T10:26:12.108Z | BUY | UNCLASSIFIABLE | 0.174 | 0.173 | 0.174 | 0x314b50d587d8b5ccd408b4050458f2e52ee1c95944d636f499141bd77538dd61 |
| 2026-05-27T10:27:58.702Z | SELL | UNCLASSIFIABLE | 0.173 | 0.173 | 0.174 | 0x92be0a7295755e17593571dc80880ed6ad7bfe9bfe84c98e8a4e49bdd20615f3 |
| 2026-05-27T10:29:32.426Z | BUY | UNCLASSIFIABLE | 0.829 | 0.828 | 0.829 | 0x41a8dcf37b80df176252737cb94cc2e7d51252d32244d8de5879cdd899b4f4b8 |
| 2026-05-27T10:31:35.680Z | BUY | UNCLASSIFIABLE | 0.787 | 0.783 | 0.786 | 0xa7e0360146595b8995d1dc618f5a62e6f0f8ba2ba6029263e4278b2f4d180ea9 |
| 2026-05-27T10:31:35.679Z | BUY | UNCLASSIFIABLE | 0.82 | 0.81 | 0.82 | 0xe0082b8bbbaf2357e82455271c57b18bb8423fb786c9ef67b9c40701fff9e432 |
