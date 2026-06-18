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
- Existing takeaway/status: Live `last_trade_price.side` is established as token-side aggressor direction, which unblocks live OFI/TFI interpretation. The note documents live CLOB inference, historical fill semantics, YES/NO token nuance, and sample rows.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
Generated: 2026-05-23

## A1 Live Capture Update

Generated: 2026-05-28

Sources:

- `data/live_clob/block_a0/block_a0_20260528_morning/*.jsonl`
- `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/*.jsonl`

Live CLOB inference over A0 + A0b:

- Total `last_trade_price` events inspected: 8,158
- Classified from book transition: 847 (10.4%)
- Ambiguous: 816 (10.0%)
- Unclassifiable: 6,495 (79.6%)
- `P(reported.side == BUY | inferred aggressor == BUY)`: 99.9%
- `P(reported.side == BUY | inferred aggressor == SELL)`: 1.7%

Conclusion: live `last_trade_price.side` is established as token-side aggressor
for Block A1. A1 uses raw unfiltered live WS data; the historical CTF Exchange
`taker = address(this)` decoding artifact does not appear in the live WS fields.

## Sources

- `data/live_clob/dali_clob_ai_product_20260523T160557Z.jsonl`
- `data/live_clob/dali_clob_ai_product_20260523T160956Z.jsonl`

## Live CLOB Inference

- Total `last_trade_price` events inspected: 1
- Classified from book transition: 0 (0.0%)
- Ambiguous: 0 (0.0%)
- Unclassifiable: 1 (100.0%)
- Minimum classifiable trades required to establish normalization: 50

Sign normalization is **not established** from this sample.

Conditional checks if classified trades exist:

- `P(reported.side == BUY | inferred aggressor == BUY)`: n/a
- `P(reported.side == BUY | inferred aggressor == SELL)`: n/a

`live_to_aggressor()` intentionally returns `UNKNOWN` by default until at least 50 classifiable live trades are available.

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
| 2026-05-23T16:12:38.855Z | BUY | UNCLASSIFIABLE | 0.715 | 0.714 | 0.715 | 0xc95ba7ebdb31b9008d4d09578ab6838096606ba0ddab21ae49a0779c45950e5d |
