---
title: Block E Lite Findings
created: 2026-05-27
status: archived
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
tags:
  - copytrade
  - block-e-lite
  - relayer
  - research
---

# Block E Lite Findings

> Hub: [[COWORK]]

> Table terms: [[polymarket_table_dictionary]]


## Summary

Block E Lite attributes the Block B operator-removal effect to hardcoded operator categories, maps those addresses to live RTDS identifier semantics, and snapshots competition in the current Block A target markets. It was later reinterpreted by [[relayer_dig_findings]] and [[copytrade_relayer_implications]], which showed the relayer category is really an exchange-internal leg artifact rather than a trader-flow population. Keep this note as the original diagnostic table and freshness snapshot.

Generated: 2026-05-27

Purpose: attribute the Block B operator-removal effect by hardcoded operator
category, document live identifier semantics for Block A, and snapshot
competition in the current Block A target markets.

## Step 0: Freshness Check

| Source | Family | Max Timestamp | Rows | Path |
| --- | --- | --- | --- | --- |
| cached_family_fills | crypto | 2026-04-24 00:00:00 UTC | 1641436 | data/analysis/dali_tfi_crypto_250_fills.parquet |
| cached_family_fills | equity_index | 2026-04-23 22:31:08 UTC | 164003 | data/analysis/dali_tfi_equity_index_100_fills.parquet |
| cached_family_fills | ai_product | 2026-04-23 23:59:34 UTC | 380287 | data/analysis/dali_tfi_ai_product_100_fills.parquet |
| cached_family_fills | sports | 2026-04-24 00:00:00 UTC | 291991 | data/analysis/dali_tfi_sports_100_fills.parquet |
| raw_trade_shards |  | 2026-05-26 19:57:58 UTC |  | data/trades/trades_delta_shard*.parquet + trades_seed.parquet |

Task 1 uses the cached Dali family fills and eval artifacts above. Task 3 uses
raw trade shards over `2026-04-26 19:57:58 UTC`
through `2026-05-26 19:57:58 UTC`.

Conditional follow-ons: **Skipped: raw shards max at 2026-05-26 19:57:58 UTC and cached family fills max at 2026-04-24 00:00:00 UTC, so the fresh-through-2026-05-27 gate is not met.**

## Section 1: Block B Operator Effect Attribution

Output CSV:
`data/analysis/csv_outputs/dali/dali_tfi_operator_category_attribution.csv`

Scope: 300s horizon, top-decile absolute TFI, inverse-maker-side convention,
`min_signal_usd=25`, `max_future_gap_seconds=300`, `exclude_last_seconds=600`.

| Family | Filter State | n_obs | Hit Rate | Mean Return (c) | Net After 1 Tick (c) |
| --- | --- | --- | --- | --- | --- |
| ai_product | all_fills | 1234 | 50.7% | 5.62 | 4.62 |
| ai_product | relayers_only_removed | 1134 | 53.8% | 5.85 | 4.85 |
| ai_product | mm_bots_only_removed | 1235 | 50.6% | 5.61 | 4.61 |
| ai_product | hft_only_removed | 1146 | 51.6% | 6.01 | 5.01 |
| ai_product | all_operators_removed | 1042 | 53.1% | 4.84 | 3.84 |
| crypto | all_fills | 1262 | 34.3% | -3.03 | -4.03 |
| crypto | relayers_only_removed | 861 | 52.6% | 7.00 | 6.00 |
| crypto | mm_bots_only_removed | 1233 | 34.6% | -2.69 | -3.69 |
| crypto | hft_only_removed | 1262 | 34.1% | -3.03 | -4.03 |
| crypto | all_operators_removed | 819 | 52.0% | 7.58 | 6.58 |
| equity_index | all_fills | 1376 | 47.1% | 5.14 | 4.14 |
| equity_index | relayers_only_removed | 1290 | 58.8% | 8.02 | 7.02 |
| equity_index | mm_bots_only_removed | 1376 | 47.2% | 5.14 | 4.14 |
| equity_index | hft_only_removed | 1376 | 47.2% | 5.14 | 4.14 |
| equity_index | all_operators_removed | 1290 | 58.8% | 8.02 | 7.02 |
| sports | all_fills | 2121 | 44.3% | 0.49 | -0.51 |
| sports | relayers_only_removed | 1339 | 44.7% | 2.32 | 1.32 |
| sports | mm_bots_only_removed | 2120 | 44.1% | 0.49 | -0.51 |
| sports | hft_only_removed | 2079 | 43.9% | 0.25 | -0.75 |
| sports | all_operators_removed | 1327 | 44.5% | 2.20 | 1.20 |

- The Block B operator effect on ai_product is driven primarily by relayers removal.
- The Block B operator effect on crypto is driven primarily by relayers removal.
- The Block B operator effect on equity_index is driven primarily by relayers removal.
- The Block B operator effect on sports is driven primarily by relayers removal.

## Section 2: Live Operator Detection Mapping

### RTDS proxyWallet Semantics

`polymarket/execution/watcher/leader_watcher.py` consumes RTDS
`topic=activity`, `type=trades` messages and compares
`payload.proxyWallet.lower()` directly against the configured leader address.
The local RTDS probe notes confirm that CLOB market WebSocket trade-like
messages do not include maker/taker/proxyWallet, while RTDS activity trades do
include `proxyWallet`, `conditionId`, `asset`, `side`, `size`, `price`,
`timestamp`, and `transactionHash`.

### Mapping Operator Addresses To RTDS Identifiers

The MM bot and HFT entries are trader/proxy wallet identities and can be matched
directly against `payload.proxyWallet.lower()`. The two relayer entries are
exchange/relayer addresses from the raw maker/taker fields, so they should not
be assumed to appear as RTDS `proxyWallet` values. Relayer-category filtering
therefore needs post-hoc raw-fill maker/taker checks, or a companion feed that
preserves fill-level maker/taker addresses; there is no one-to-one proxyWallet
mapping for those relayer addresses.

| Category | Address | Identifier Type | RTDS Handling | Seen Last 30d | Latest Seen | Distinct Markets |
| --- | --- | --- | --- | --- | --- | --- |
| relayer | 0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e | exchange/relayer address | post-hoc raw maker/taker | yes | 2026-04-28 11:00:40 UTC | 13688 |
| relayer | 0xc5d563a36ae78145c45a50134d48a1215220f80a | exchange/relayer address | post-hoc raw maker/taker | yes | 2026-04-28 11:00:40 UTC | 8537 |
| mm_bot | 0x04895657d3c2afebec8be4b6e60b9c56ad68ee4d | trader/proxy wallet | direct proxyWallet lookup | no |  | 0 |
| mm_bot | 0x297fbd45782af37d899015aebbc52437f3d55103 | trader/proxy wallet | direct proxyWallet lookup | no |  | 0 |
| mm_bot | 0x38e598961dd0456a7fb2e758bd433d3e59fb8a4a | trader/proxy wallet | direct proxyWallet lookup | yes | 2026-05-26 19:57:58 UTC | 31590 |
| mm_bot | 0x5f4d4927ea3ca72c9735f56778cfbb046c186be0 | trader/proxy wallet | direct proxyWallet lookup | no |  | 0 |
| mm_bot | 0xd44e29936409019f93993de8bd603ef6cb1bb15e | trader/proxy wallet | direct proxyWallet lookup | yes | 2026-05-26 19:57:57 UTC | 11214 |
| mm_bot | 0xdc669ba0adb45448020025f756070492d1070533 | trader/proxy wallet | direct proxyWallet lookup | yes | 2026-05-26 19:53:59 UTC | 2841 |
| mm_bot | 0xe9cbb1c9b3f7f411dd4fdf2ea7afa780c8b4d096 | trader/proxy wallet | direct proxyWallet lookup | yes | 2026-05-26 01:39:30 UTC | 46 |
| hft | 0x63d43bbb87f85af03b8f2f9e2fad7b54334fa2f1 | trader/proxy wallet | direct proxyWallet lookup | yes | 2026-05-24 09:18:28 UTC | 67 |
| hft | 0xe3726a1b9c6ba2f06585d1c9e01d00afaedaeb38 | trader/proxy wallet | direct proxyWallet lookup | yes | 2026-05-26 17:29:47 UTC | 6004 |
| hft | 0xe8dd7741ccb12350957ec71e9ee332e0d1e6ec86 | trader/proxy wallet | direct proxyWallet lookup | yes | 2026-05-26 19:57:36 UTC | 3712 |

### Block A Operator Filtering Procedure

Block A0's current CLOB market-channel capture contains market, asset, book,
price-change, best-bid-ask, and last-trade-price state, but not wallet
identity. MM bot and HFT category filtering can be applied live only with an
RTDS `activity/trades` companion stream that carries `proxyWallet`. The relayer
category, which is the category driving the Block B attribution here, cannot be
recovered from CLOB-only capture or RTDS `proxyWallet` alone; it needs post-hoc
raw-fill maker/taker checks or another companion feed that preserves maker/taker
addresses.

### Live Operator Discovery Procedure

For wallets new since 2026-04-24, keep Block A capture running as planned and
flag candidates post-hoc from raw fills in the same market/time windows:
high side-volume share, near-balanced maker/taker role, many counterparties,
and sub-second clustering. Treat those candidates as analysis labels until a
separate denylist-refresh decision is made.

## Section 3: Per-Market Competition Snapshot

Output CSV:
`data/analysis/csv_outputs/dali/block_a_market_competition_snapshot.csv`

| Market | Family | Class | Known Op Share | Relayer | MM Bot | HFT | Top Non-Op | Wallets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 950854 | daily_equity_index | DISTRIBUTED | 5.9% | 5.5% | 0.0% | 0.4% | 20.8% | 129 |
| 665531 | daily_single_stock | DISTRIBUTED | 5.3% | 5.3% | 0.0% | 0.0% | 22.1% | 86 |
| 631139 | ai_product | DISTRIBUTED | 3.2% | 0.3% | 0.0% | 2.9% | 25.1% | 613 |
| 631140 | ai_product | DISTRIBUTED | 2.1% | 1.3% | 0.0% | 0.8% | 27.0% | 1360 |
| 573647 | ai_product | DISTRIBUTED | 1.4% | 1.4% | 0.0% | 0.0% | 23.6% | 286 |
| 558934 | sports_game_lines | DISTRIBUTED | 0.9% | 0.9% | 0.0% | 0.0% | 23.1% | 6058 |
| 1469737 | geopolitics_policy | DISTRIBUTED | 0.6% | 0.6% | 0.0% | 0.0% | 25.8% | 904 |
| 1295976 | ai_product | DISTRIBUTED | 0.4% | 0.4% | 0.0% | 0.0% | 21.5% | 329 |
| 558936 | sports_game_lines | DISTRIBUTED | 0.3% | 0.3% | 0.0% | 0.0% | 22.0% | 7550 |
| 665325 | geopolitics_policy | DISTRIBUTED | 0.2% | 0.2% | 0.0% | 0.0% | 25.5% | 1035 |
| 1633611 | geopolitics_policy | DISTRIBUTED | 0.0% | 0.0% | 0.0% | 0.0% | 33.4% | 359 |
| 1090496 | geopolitics_policy | DISTRIBUTED | 0.0% | 0.0% | 0.0% | 0.0% | 26.9% | 352 |

Classification counts: {"DISTRIBUTED": 12}. Operator-removal effects should be most visible in no current Block A target markets; no markets look more like single-wallet concentration; 1090496, 1633611, 665325, 1469737, 573647, 631140, 631139, 1295976, 558936, 558934, 950854, 665531 are the better candidates for organic-flow interpretation.

## Step 2: Conditional Follow-Ons

- Re-run `scripts/validation/02_operator_detection.py` on extended data:
  **Skipped: raw shards max at 2026-05-26 19:57:58 UTC and cached family fills max at 2026-04-24 00:00:00 UTC, so the fresh-through-2026-05-27 gate is not met.**
- Extend Block B walk-forward with 2026-04-25 through 2026-05-27:
  **Skipped: raw shards max at 2026-05-26 19:57:58 UTC and cached family fills max at 2026-04-24 00:00:00 UTC, so the fresh-through-2026-05-27 gate is not met.**

No operator denylist entries were changed by this run.
