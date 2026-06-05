---
tags: [copytrade, relayer-implications, results]
---

# Copytrade Relayer Implications

> Table terms: [[polymarket_table_dictionary]]

## Headline

The proposed "invisible taker wallet" bias does **not** survive the recovery probe. The v1 exchange-internal population is real and large: **416,108,840 rows / $30.45B** where `taker` is one of the v1 exchange contracts. But a 50,000-row recovery sample found that `transaction.from` is a relay/submission layer, not the trader wallet: **2,051 distinct tx senders, 0 joined to `traders.parquet`**. The source-level check is decisive: Polymarket CTF Exchange `_matchOrders` emits the internal active leg as `maker = takerOrder.maker` and `taker = address(this)`, so the active wallet is already in the `maker` column. Result: no PnL/position rebuild is warranted for "missing takers"; the real caveat is role semantics, because v1 active taker-order legs are labeled as `maker` in raw events. Domah's smoke cell stays intact.

## Step 1 - Recovery Probe

Data gate passed. Required data artifacts were present:

- `data/trades/trades_seed.parquet`
- `data/trades/trades_delta_shard*.parquet` - 61 local delta shards
- `data/closed_positions.parquet`
- `data/traders.parquet`
- `data/cohorts/*.parquet` - 6 cohort parquets

Schema probe:

| file | tx identity fields present | absent |
|---|---|---|
| `data/trades/trades_seed.parquet` | `transaction_hash` | `block_number`, `tx_from`, `from`, `tx_origin`, `log_index` |
| `data/trades/trades_delta_shard1_2025-10-15_2025-11-13.parquet` | `transaction_hash` | `block_number`, `tx_from`, `from`, `tx_origin`, `log_index` |

The literal prompt path `data/trades/trades_delta_shard0.parquet` is not present in this workspace; I used the first present delta shard for the schema check.

Goldsky's `OrderFilledEvent` schema exposes `id`, `transactionHash`, `timestamp`, maker/taker asset fields, and amounts, but not `transaction.from`. I therefore used the repo's existing Polygon RPC pattern only for the 50k sample, not a full sweep.

Recovery sample result:

| metric | value |
|---|---:|
| v1 internal rows | 416,108,840 |
| v1 internal notional | $30,449,998,055.83 |
| sample rows | 50,000 |
| sample tx hashes recovered | 50,000 |
| distinct `tx_from_recovered` | 2,051 |
| recovered tx_from addresses joining `traders.parquet` | 0 |

Interpretation: `tx.from` is not usable as trader identity here. This is consistent with relayed CLOB settlement. The stronger source-level check is that the active order wallet is already emitted as `maker` on the internal leg. In the local data, **414,747,677 / 416,108,840** v1 internal rows have a `maker` address that joins the trader panel.

Artifacts:

- `data/analysis/copytrade_taker_recovery_sample.parquet`
- `data/analysis/copytrade_invisible_take_estimates.parquet`

## Step 2 - Cohort Bias Table

Because `tx.from` does not recover trader wallets, the patched cohort positions add no trader-attributed fills. The table below is therefore a diagnostic lower-bound table: patched metrics equal visible metrics up to floating-point noise.

| address | n_visible_fills | est_invisible_takes | est_invisible_volume_usd | visible_mkt_total_pnl | patched_mkt_total_pnl | mkt_pnl_delta_usd | visible_total_volume_usd | patched_total_volume_usd | volume_delta_pct | style_label | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee | 183,546 | 0 | 0.00 | 14,952,586.45 | 14,952,586.45 | 0.00 | 171,407,351.54 | 171,407,351.54 | -0.000000% | maker_heavy | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xee00ba338c59557141789b127927a55f5cc5cea1 | 113,935 | 0 | 0.00 | 4,658,451.29 | 4,658,451.29 | -0.00 | 149,162,152.08 | 149,162,152.08 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xd38b71f3e8ed1af71983e5c309eac3dfa9b35029 | 9,582 | 0 | 0.00 | 5,678,260.95 | 5,678,260.95 | 0.00 | 28,818,135.13 | 28,818,135.13 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x17db3fcd93ba12d38382a0cade24b200185c5f6d | 7,228 | 0 | 0.00 | 5,453,680.45 | 5,453,680.45 | -0.00 | 16,576,169.06 | 16,576,169.06 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x9d84ce0306f8551e02efef1680475fc0f1dc1344 | 604,311 | 0 | 0.00 | 4,007,338.41 | 4,007,338.41 | 0.00 | 170,931,888.49 | 170,931,888.49 | -0.000000% | maker_heavy | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x629bc4a1e53e1d475beb7ea3d388791e96dd995a | 53,428 | 0 | 0.00 | 1,795,456.64 | 1,795,456.64 | 0.00 | 22,605,509.66 | 22,605,509.66 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x559eeff0d61d8b95438df371db8f7ab7d50d7fe2 | 1,605 | 0 | 0.00 | -1,233.89 | -1,233.89 | -0.00 | 226,812.21 | 226,812.21 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x405809651dae4e0556e8f3740b1e16a7df2d35b5 | 24,245 | 0 | 0.00 | -247.09 | -247.09 | 0.00 | 8,442.95 | 8,442.95 | 0.000000% | maker_heavy | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xa06880a56f50bbb532bea20226131ed1ef689574 | 1,045 | 0 | 0.00 | -10,368.15 | -10,368.15 | 0.00 | 253,480.62 | 253,480.62 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x8c5b68b3814fffb5fadeea649e2400b28efba790 | 6,178 | 0 | 0.00 | 15,273.84 | 15,273.84 | -0.00 | 431,588.58 | 431,588.58 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xe10c1838ee3c214cd0fd8af9040bee5bd19ee7e2 | 479 | 0 | 0.00 | -2,974.07 | -2,974.07 | 0.00 | 32,338.92 | 32,338.92 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x879752bca887bc247ba883a52bc45e9c6be6bbbe | 6,837 | 0 | 0.00 | -8,012.99 | -8,012.99 | 0.00 | 508,907.31 | 508,907.31 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x5ad18c240201d1166e607acba4114ede37ff2ecc | 4,977 | 0 | 0.00 | -12,356.16 | -12,356.16 | -0.00 | 395,921.21 | 395,921.21 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xe18237fa7de9111636c18479ce81322c76edb624 | 4,422 | 0 | 0.00 | -7,543.78 | -7,543.78 | -0.00 | 543,236.57 | 543,236.57 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xfacff56a9445419af47c40d296db00f022db44ad | 9,419 | 0 | 0.00 | -561.28 | -561.28 | -0.00 | 190,399.88 | 190,399.88 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x40e3a0d39e180dd3c709a22f15d4da40bd044101 | 7,786 | 0 | 0.00 | -1,164.82 | -1,164.82 | 0.00 | 278,805.88 | 278,805.88 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x552c29093b072a68c59a078cd27e2541012ebe63 | 7,866 | 0 | 0.00 | -168.78 | -168.78 | -0.00 | 132,352.17 | 132,352.17 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x4a881b44dfade60d289ed2c7b9b68418ba907914 | 6,964 | 0 | 0.00 | 712.27 | 712.27 | -0.00 | 56,145.46 | 56,145.46 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xaebb94f2d2f10f18a21944a431b0489181ad4c67 | 504 | 0 | 0.00 | -1,408.53 | -1,408.53 | 0.00 | 8,067.65 | 8,067.65 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xf612d428b0dbb58f33d4e9197439f6d40d165f4f | 2,024 | 0 | 0.00 | -157.87 | -157.87 | 0.00 | 127,746.93 | 127,746.93 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xaab66edcd70fa11864483b496fe6a79d55ac923f | 2,581 | 0 | 0.00 | -115.46 | -115.46 | -0.00 | 19,770.35 | 19,770.35 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xad72f59af13c2229526ccf172433d6cabd2dd5bf | 3,268 | 0 | 0.00 | -2,099.15 | -2,099.15 | -0.00 | 72,079.05 | 72,079.05 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x27e451f0d932c57bf0c8bc483bdfe2300649ff39 | 1,026 | 0 | 0.00 | -2,265.86 | -2,265.86 | 0.00 | 95,063.33 | 95,063.33 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xa07481364592a84e3acaacf4ddc60b6d4afc65a0 | 633 | 0 | 0.00 | -2,704.81 | -2,704.81 | 0.00 | 184,527.45 | 184,527.45 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x4f46dadcd74c1e4a9134a690eadef21a8ca0b84b | 325 | 0 | 0.00 | -659.17 | -659.17 | 0.00 | 3,969.83 | 3,969.83 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x7e798cb38c4fb667d59e069c1315293e2d67ff8b | 916 | 0 | 0.00 | 285.71 | 285.71 | -0.00 | 74,930.71 | 74,930.71 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x2148aef58b3765a35f984fa70e71fedf8598d0a7 | 2,091 | 0 | 0.00 | -1,747.75 | -1,747.75 | -0.00 | 72,175.23 | 72,175.23 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x55a7310006103249f63ae18ff366fedc410e8fc5 | 991 | 0 | 0.00 | 44.98 | 44.98 | -0.00 | 6,216.72 | 6,216.72 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x058a2e4164089d7fdb29e1070a0668b4ba4bed5c | 466 | 0 | 0.00 | -748.93 | -748.93 | -0.00 | 12,175.12 | 12,175.12 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x9348fba201c097c012c1d9a51c1ba30f459d8cff | 1,156 | 0 | 0.00 | 198.38 | 198.38 | -0.00 | 2,529.13 | 2,529.13 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x4f205eb649e938fb59162f94191ebdba60bf2e89 | 869 | 0 | 0.00 | -258.91 | -258.91 | 0.00 | 3,788.59 | 3,788.59 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x5677fbe1e586fbe5cbba8b5ab447c27f8b850f1c | 466 | 0 | 0.00 | -279.08 | -279.08 | 0.00 | 2,046.04 | 2,046.04 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x71b6414cc8b5c9ba575a33f545823b8e1fefe44f | 741 | 0 | 0.00 | -37.95 | -37.95 | -0.00 | 2,117.75 | 2,117.75 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x2d3d34f601ccbde837b2347cd884c0aca7bf541d | 343 | 0 | 0.00 | 337.30 | 337.30 | 0.00 | 8,149.51 | 8,149.51 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x03dc083a54c3340568279873f6242ae0077e78cf | 423 | 0 | 0.00 | -287.44 | -287.44 | -0.00 | 3,785.83 | 3,785.83 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x429d4663308ec8dadb40948f25756dd8d5b1243c | 465 | 0 | 0.00 | 21.62 | 21.62 | -0.00 | 2,067.65 | 2,067.65 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xc0a9bf881d436d0689f5707174669169a0a5a815 | 1,081 | 0 | 0.00 | 7.24 | 7.24 | -0.00 | 8,491.02 | 8,491.02 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x787209402228c41bd6cc00fb705416611863b8d2 | 486 | 0 | 0.00 | -161.07 | -161.07 | 0.00 | 6,390.75 | 6,390.75 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xe93f3b637ebd1fe101476a8dca2c69bd98bc8239 | 961 | 0 | 0.00 | 48.75 | 48.75 | 0.00 | 4,170.15 | 4,170.15 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xba86312cec25807bcc6fc9b0a232639f1119a1d1 | 413 | 0 | 0.00 | 80.12 | 80.12 | 0.00 | 650.19 | 650.19 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x96942e2d8bcead417d836b4905818de316b1db65 | 847 | 0 | 0.00 | -55.15 | -55.15 | -0.00 | 3,968.00 | 3,968.00 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x69156c6ab2803c7dfb4bf1969d99b71a41616774 | 379 | 0 | 0.00 | -8.53 | -8.53 | -0.00 | 466.71 | 466.71 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xa79d69bde78405f64086454a4152c7143c3510cd | 398 | 0 | 0.00 | -118.10 | -118.10 | 0.00 | 1,994.74 | 1,994.74 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x94fcbc575909d6609261ea66f699e49f5672ac00 | 872 | 0 | 0.00 | -86.51 | -86.51 | 0.00 | 18,900.21 | 18,900.21 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xeceb8cc33bcc1cc2768e4a061d23926eead13204 | 736 | 0 | 0.00 | -29.87 | -29.87 | 0.00 | 18,086.98 | 18,086.98 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x100c5be2f79a429ad07671b61aec11016a335b90 | 340 | 0 | 0.00 | -13.48 | -13.48 | 0.00 | 2,199.23 | 2,199.23 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x57cc0b2042adf1deab135163294cde19b8cdd807 | 836 | 0 | 0.00 | -17.27 | -17.27 | 0.00 | 17,660.37 | 17,660.37 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x4e98d329f9c9cd205877ce97719682ce6e3d050f | 802 | 0 | 0.00 | -35.71 | -35.71 | -0.00 | 1,997.81 | 1,997.81 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xbe74f06dcff9e8a72709fd42ccfd49f22415199b | 706 | 0 | 0.00 | 4.51 | 4.51 | -0.00 | 1,927.21 | 1,927.21 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xf6bbc352acd779ebeb8f9f29aefebbb28ad16215 | 707 | 0 | 0.00 | -15.01 | -15.01 | 0.00 | 17,749.91 | 17,749.91 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xbb6b5ead3f5c7809bb4d55357d8c1f795e9fb0b9 | 2,317 | 0 | 0.00 | 11,806.38 | 11,806.38 | 0.00 | 286,744.41 | 286,744.41 | 0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0xee67f4f549180f564dd5910b1024b8c6729cef38 | 24,309 | 0 | 0.00 | 821,466.78 | 821,466.78 | 0.00 | 7,333,660.53 | 7,333,660.53 | 0.000000% | maker_heavy | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x2e73f6d6af7a7a7ceb28c1bce29d938128b3c4c2 | 5,442 | 0 | 0.00 | 1,141.65 | 1,141.65 | 0.00 | 7,764.67 | 7,764.67 | -0.000000% | maker_heavy | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x0f3a06e3e5a6c473aa5ce052934433c60571840b | 1,272 | 0 | 0.00 | -492.34 | -492.34 | 0.00 | 10,932.26 | 10,932.26 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x254841733145911941093127d86ec0383d72b742 | 3,279 | 0 | 0.00 | -587.85 | -587.85 | 0.00 | 13,428.09 | 13,428.09 | -0.000000% | mixed | no recovered rows in 50k sample; patched equals visible lower bound |
| 0x1b406adb231c2886e1ab599777dc03fab95dd85e | 77,246 | 0 | 0.00 | -4,734.94 | -4,734.94 | 0.00 | 1,171,387.47 | 1,171,387.47 | 0.000000% | maker_heavy | no recovered rows in 50k sample; patched equals visible lower bound |

Artifact:

- `data/analysis/copytrade_patched_positions_cohort.parquet`
- `data/analysis/csv_outputs/copytrade/copytrade_cohort_bias_table.csv`

## Step 3 - Systematicity

The systematicity test is intentionally boring after the source check:

| bucket | n traders | mean invisible_share | median invisible_share | 95% CI |
|---|---:|---:|---:|---|
| maker_heavy | 100,305 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| mixed | 2,410,486 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| taker_heavy | 61,874 | 0.0000 | 0.0000 | [0.0000, 0.0000] |

Spearman correlations versus cohort-selection columns and the requested columns are `NaN`, not because the effect is small but because `invisible_share` has zero variance after `tx.from` fails to recover trader addresses. See `data/analysis/csv_outputs/copytrade/copytrade_bias_systematicity.csv`.

Role-semantics caveat: if v1 exchange-internal maker rows are reclassified as "active taker order signer" for style only, some style ratios move. For the manual candidates: Domah remains maker-heavy (`7.89` visible maker:taker ratio, `5.67` after reclassifying internal rows), while `0x6a72...` would move from maker-heavy (`5.15`) to mixed (`3.46`). This affects style labels, not PnL attribution.

## Step 4 - v2 Verification

The exchange-internal event pattern is **ongoing**, not bounded to pre-2026-04-28 history.

| address | version | as_maker | as_taker | first_seen | last_seen |
|---|---:|---:|---:|---|---|
| `0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e` | v1 standard | 0 | 313,890,933 | 2022-11-21 19:50:09 | 2026-04-28 11:00:40 |
| `0xc5d563a36ae78145c45a50134d48a1215220f80a` | v1 neg-risk | 0 | 102,217,907 | 2023-12-22 03:18:40 | 2026-04-28 11:00:40 |
| `0xe111180000d2663c0091e4f400237545b87b996b` | v2 standard | 0 | 52,314,492 | 2026-04-27 00:16:14 | 2026-05-26 19:57:58 |
| `0xe2222d279d744050d28e00520010520000310f59` | v2 neg-risk | 0 | 10,930,374 | 2026-04-27 01:28:34 | 2026-05-26 19:57:58 |

In a 100,000-row post-cutover sample after `2026-04-28 11:00:40`, v2 exchange contracts appeared in the `taker` slot **35,885** times and in the `maker` slot **0** times. This is materially consistent with the v1 pattern.

## Step 5 - Domah Smoke Target

Domah's recorded lifetime `mkt_total_pnl` before patching was **$4,007,338.41**. After patching it remains **$4,007,338.41**. Domah had **24,010** v1 exchange-internal rows where he was the `maker` on the internal leg, totaling **$36.13M** notional; those rows were already visible to maker-filtered copytrade code.

Domah's style classification remains maker-heavy. Visible maker:taker ratio is **7.89**; reclassifying v1 internal maker rows as active-taker-order rows would still leave him at **5.67**, above the maker-heavy threshold of 4.

The current Domah cell remains intact:

- Macro family: still the strongest deployable family. Existing audit has `macro` `A_real_pnl = $274,311` and `C_real_pnl = $305,422`, ahead of the other positive families on deployable copy logic.
- Maker sub-cell: still survives. Role slice has maker-side `A_real_pnl = $95,602` and `C_real_pnl = $258,279`; taker-side is negative.
- 18-24 bucket: still holds up. Hour slice has `18-24` `A_real_pnl = $299,877` and `C_real_pnl = $350,872`, the strongest hour bucket.

Verdict: **cell intact**.

## Recommendation

Recommend **(a) document the corrected limitation, no rebuild**. The reason is not "bias is tiny"; the reason is that the alleged missing-wallet path is the wrong interpretation. The active wallet for `_matchOrders` is already emitted in `maker`, and `tx.from` is a relay/submission identity that does not join to the trader panel. Step 2 shows no PnL/volume delta for the inspection cohort, Step 3 has no systematic invisible-share signal to correlate, Step 4 says the exchange-internal event pattern is ongoing in v2, and Step 5 says Domah's deployable cell is unchanged.

Do **not** rebuild `closed_positions.parquet` or `traders.parquet` for hidden takers. Do consider a later lightweight style-only enhancement: add an `exchange_internal_leg` / `active_order_leg` flag so `style_maker_taker_ratio` is not read as pure passive-maker behavior in v1/v2 internal-leg rows.

## Denylist Diff Summary

Updated `data_infra/operator_denylist.py`:

- Renamed `PURE_RELAYERS` to version-aware `EXCHANGE_INTERNAL_LEG_V1`, `EXCHANGE_INTERNAL_LEG_V2`, and `EXCHANGE_INTERNAL_LEG`.
- Added v2 exchange contracts:
  - `0xe111180000d2663c0091e4f400237545b87b996b`
  - `0xe2222d279d744050d28e00520010520000310f59`
- Left `PURE_MM_BOTS`, `HFT`, and `is_operator_like()` unchanged.
- Updated `OPERATOR_ADDRESSES` to include all v1 and v2 exchange-internal-leg contracts.
- Updated `scripts/block_e_lite.py` to import `EXCHANGE_INTERNAL_LEG_V1` instead of `PURE_RELAYERS`.

## Caveats

- The sample used Polygon RPC for `eth_getTransactionByHash` because the local parquets and the Goldsky orderbook subgraph do not expose `tx_from`. This was a 50k sample only, not a full RPC sweep.
- `tx.from` is not trader identity for these fills. Treat `data/analysis/copytrade_invisible_take_estimates.parquet` as the output of the requested recovery path, not as a proof that relay sender identities are meaningful trader addresses.
- The patched cohort table is a lower-bound diagnostic and is intentionally equal to visible metrics because no `tx.from` sender joins `traders.parquet`.
- v2 verification is based on current local shards through `2026-05-26 19:57:58`; future direct-polygon shards should keep the v2 exchange constants in the denylist.
- Source reference: Polymarket CTF Exchange `Trading.sol` emits the internal active leg with `maker = takerOrder.maker` and `taker = address(this)` in `_matchOrders`; see [Polymarket ctf-exchange `Trading.sol`](https://raw.githubusercontent.com/Polymarket/ctf-exchange/main/src/exchange/mixins/Trading.sol), plus local notes `notes/copytrade/relayer_dig_findings.md` and `notes/copytrade/block_b_reinterpretation.md`.

Recommended next action for Justin: Proceed with the Domah macro / maker / 18-24 paper-trade or smoke plan without rebuilding copytrade PnL tables, and add an exchange-internal-leg role flag before making any style-only claims.
