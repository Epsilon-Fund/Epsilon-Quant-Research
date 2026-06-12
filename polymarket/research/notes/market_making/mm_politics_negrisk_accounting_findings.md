---
title: "MM Politics NegRisk Accounting Findings"
tags: [market-making, negrisk, k5-stress, accounting]
created: 2026-06-03
status: merits-live-measurement-loop
---

# MM Politics NegRisk Accounting Findings

> Hub: [[strat_market_making]] · [[COWORK]]

## Summary

- Scope: MM Politics NegRisk Accounting Findings in the MM/market-making area.
- Existing takeaway/status: MERITS-LIVE-MEASUREMENT-LOOP.** The `politics_negrisk` structured-non-top3 maker cell survives true NegRisk accounting. The original token-level K5-STRESS row was not a conversion artifact; after reconstructing merge/split/redeem inventory movements from Polygon receipts, the cell gets stronger on mean PnL and stays positive on market-cluster CI, non-rebate PnL, and median wallet.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Verdict

**MERITS-LIVE-MEASUREMENT-LOOP.** The `politics_negrisk` structured-non-top3 maker cell survives true NegRisk accounting. The original token-level K5-STRESS row was not a conversion artifact; after reconstructing merge/split/redeem inventory movements from Polygon receipts, the cell gets stronger on mean PnL and stays positive on market-cluster CI, non-rebate PnL, and median wallet.

This is not a build-now verdict. It is a live measurement-loop verdict: the remaining unknown is not historical accounting, it is live deployability -- real fill share, queue position, missed fills, cancels, capital/carry tolerance, and whether a non-incumbent can obtain the assumed capture without becoming the adverse-selection sink.

## Accounting Rebuild

Scope: the K5-STRESS `politics_negrisk` wallet-market cache, restricted to the structured maker playbook and non-top3 market-maker rows. The audit swept **284 structured wallets**.

Data path:
- Indexed Polymarket activity API `SPLIT,MERGE,REDEEM,CONVERSION` rows for the 284 wallets.
- Raw NegRisk activity indexed: **2,098,429 rows**.
- Relevant politics activity tx-wallet pairs: **125,937/125,937 covered** in receipt parquet.
- Polygon archive receipts decoded from CTF ERC1155 transfers and USDC ERC20 transfers.
- Final receipt parquet: **950,821 decoded rows**, **125,937 tx-wallet pairs**, **125,886 distinct tx hashes**, **0 missing receipt tx-wallet pairs**.
- No mark-to-mid: token deltas are valued at settlement where closed, otherwise latest trade/snapshot mark used only to net non-trade inventory movements into the existing realized K5 cache.

Relevant politics activity by API type:

| type | activity rows | tx-wallet pairs | wallets |
|---|---:|---:|---:|
| MERGE | 155,771 | 113,411 | 237 |
| REDEEM | 17,964 | 9,143 | 167 |
| SPLIT | 3,452 | 3,452 | 72 |
| CONVERSION | 0 | 0 | 0 |

Receipt/accounting audit:

| metric | value |
|---|---:|
| Relevant txs touching politics token adjustments | 124,473 |
| Receipt txs found | 125,886 / 125,886 |
| Receipt cash delta | $1,798,550,805 |
| Receipt abs token delta | 1,583,202,110 tokens |
| Token-value adjustment | -$333,339,100 |
| Cash allocation adjustment | +$367,951,600 |
| Net accounting adjustment | +$34,612,500 |

Interpretation: the conversion layer was material, but it did not manufacture the politics edge. It moved value across the basket in ways token-only trade PnL mis-attributed; once netted at wallet/position level, the structured maker edge remains and increases.

## Re-Judgement

All rows below are net of observed K5 costs/rebates; `net_without_rebate_bps` subtracts maker rebates as the non-rebate-dependence check. CI is market-cluster bootstrap.

| cut | wallets | markets | gross | net | bps | CI bps | median wallet bps | net ex rebate bps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original K5 token PnL, structured non-top3 | 284 | 7,913 | $385.7M | $45.4M | 1,177 | [515, 2,148] | 9.8 | 1,159 |
| Accounting-corrected, same frozen K5 membership | 284 | 7,913 | $385.7M | $71.6M | 1,857 | [1,160, 3,353] | 41.0 | 1,839 |
| Corrected-carry re-cut, structured non-top3 | 67 | 4,642 | $179.4M | $41.1M | 2,290 | [1,020, 3,621] | 14.5 | 2,276 |

The official decision row is the corrected-carry re-cut because the K5 gate includes carry-to-resolution and NegRisk conversions change the true carry denominator. It still clears the same de-biased K5-STRESS discipline: two-sided >=60%, carry >=50%, spike <=2%, non-top3, non-rebate-dependent, lookahead-free, net-of-cost, market-cluster CI positive.

## Capacity As Assumption

Non-top3 2026 maker flow for this politics-NegRisk capacity base is **$381.1M**, over **146 active days**, or **$2.61M/day**. This is a ladder of deployment assumptions, not a hard cap.

| assumption | headroom/day | EV/day at median wallet bps | EV/day at mean bps |
|---|---:|---:|---:|
| 0.25% of non-top3 2026 flow | $6,525 | $9 | $1,494 |
| 1.00% of non-top3 2026 flow | $26,102 | $38 | $5,978 |
| 5.00% of non-top3 2026 flow | $130,508 | $189 | $29,889 |

The mean-row EV is large because the historical cell has a very fat right tail. The median-wallet EV is much smaller, so live sizing should start as measurement, not conviction. Rule 3 framing: capacity is an assumption to test with live quote telemetry, not a proof from one historical capture proxy.

## Persistence And Monthly Translation

Follow-up on the capacity question: the 2026 non-top3 flow is **recurring, not only election-seasonal**. From 2026-01-01 through the last local trade shard on 2026-05-26, politics-NegRisk had non-top3 maker flow on **146/146 observed days**. Every observed 2026 day had at least **$250k** of non-top3 maker flow; **122/146** days had at least **$1M**.

Quarterly active-flow profile:

| quarter | active days | flow | median active-day flow | days >= $250k | days >= $1M |
|---|---:|---:|---:|---:|---:|
| 2024 Q1 | 90 | $25.0M | $0.20M | 34 | 2 |
| 2024 Q2 | 91 | $35.4M | $0.31M | 59 | 3 |
| 2024 Q3 | 92 | $290.1M | $2.59M | 92 | 92 |
| 2024 Q4 | 92 | $1.35B | $5.48M | 92 | 92 |
| 2025 Q1 | 90 | $96.7M | $0.55M | 84 | 26 |
| 2025 Q2 | 91 | $246.7M | $1.05M | 87 | 46 |
| 2025 Q3 | 92 | $50.7M | $0.48M | 89 | 6 |
| 2025 Q4 | 92 | $193.4M | $1.16M | 92 | 54 |
| 2026 Q1 | 90 | $276.3M | $2.68M | 90 | 80 |
| 2026 Q2-to-05-26 | 56 | $104.7M | $1.45M | 56 | 42 |

2026 flow mix by event bucket:

| event bucket | 2026 non-top3 flow | share | active days | markets |
|---|---:|---:|---:|---:|
| 2028 US presidential outrights | $138.8M | 36.4% | 146 | 191 |
| Trump personnel/policy | $91.6M | 24.0% | 139 | 87 |
| Non-US elections | $72.0M | 18.9% | 146 | 750 |
| Other politics | $59.8M | 15.7% | 146 | 3,807 |
| 2026 US races/midterms | $18.9M | 5.0% | 146 | 716 |
| 2024 US election cycle | $0.07M | 0.0% | 60 | 2 |

No-mark-to-mid edge persistence check: on the exact corrected-carry re-cut, settled-only close-year slices stay positive outside the 2024 election window. 2025 settled rows are **+1,402 bps**, CI **[387, 2,919]**, median wallet **398 bps**. 2026 settled rows are **+1,356 bps**, CI **[744, 2,507]**, median wallet **368 bps**. The 2024 US-election slice is mean-positive but not robust by CI (**+2,399 bps**, CI **[-69, 5,361]**, median **40 bps**), so the surviving thesis is not "only trade the general election spike." Settled event slices that clear include **Non-US elections** (**+3,673 bps**, CI **[1,671, 6,978]**), **Trump personnel/policy** (**+1,018 bps**, CI **[174, 1,182]**), and **Other politics** (**+1,046 bps**, CI **[216, 2,245]**). The 2028 US presidential outrights supply the largest 2026 flow bucket but have no settled-only PnL yet, so they remain a live/forward unknown rather than a historical edge proof.

Monthly deployable translation, weighted by the actual 2026 persistence profile:

| month | active days | non-top3 flow | median-bps EV/mo @0.25% / 1% / 5% capture | mean-bps EV/mo @0.25% / 1% / 5% capture |
|---|---:|---:|---:|---:|
| 2026-01 | 31 | $72.6M | $262 / $1,049 / $5,243 | $41.5k / $166.2k / $830.8k |
| 2026-02 | 28 | $92.8M | $335 / $1,342 / $6,708 | $53.2k / $212.6k / $1.06M |
| 2026-03 | 31 | $111.0M | $401 / $1,604 / $8,018 | $63.5k / $254.1k / $1.27M |
| 2026-04 | 30 | $75.9M | $274 / $1,098 / $5,488 | $43.5k / $173.9k / $869.6k |
| 2026-05-to-05-26 | 26 | $28.8M | $104 / $416 / $2,081 | $16.5k / $66.0k / $329.8k |
| **2026 YTD avg/month** | **146 total** | **$381.1M total** | **$275 / $1,102 / $5,508** | **$43.6k / $174.5k / $872.7k** |

Verdict refinement: **standing live-loop anchor, event-aware sizing**. The flow is daily enough to justify a standing measurement loop for queue/fill/cancel/adverse-selection telemetry. It is not a standing production-PnL promise: the median-wallet translation is only about **$0.4k-$1.6k/month** at 1% capture in the observed 2026 months, while the mean-row translation is a right-tail scenario that must be earned live. Sizing should expand around proven settled event buckets and shrink for forward-only/open buckets like 2028 outrights until live fills validate them.

## Live Unknowns

- Real fill share at our quotes versus top-maker queue depth.
- Queue position, cancels, missed fills, and post-fill adverse selection.
- Whether the politics NegRisk basket edge belongs to a few capital/carry specialists or is reproducible by a smaller non-incumbent.
- Operational handling of merge/split/redeem/composite inventory in the executor before any real sizing.
- Whether live activity/receipt reconciliation can run close enough to real time to keep basket PnL and exposure correct.

## Artifacts

- Script: `polymarket/research/scripts/mm_politics_negrisk_accounting.py`
- Raw activity: `data/analysis/mm_politics_negrisk_activity_raw.parquet`
- Receipt deltas: `data/analysis/mm_politics_negrisk_receipt_deltas.parquet`
- Corrected wallet-market cache: `data/analysis/mm_politics_negrisk_accounting_wallet_market.parquet`
- Exact corrected-carry re-cut: `data/analysis/mm_politics_negrisk_corrected_carry_recut_wallet_market.parquet`
- Summary CSV: `data/analysis/csv_outputs/market_making/mm_politics_negrisk_accounting_summary.csv`
- Capacity CSV: `data/analysis/csv_outputs/market_making/mm_politics_negrisk_capacity_ladder.csv`
- Event audit CSV: `data/analysis/csv_outputs/market_making/mm_politics_negrisk_event_audit.csv`
- Persistence CSVs: `data/analysis/csv_outputs/market_making/mm_politics_negrisk_persistence_*.csv`
- Persistence plot: `data/analysis/figures/mm_politics_negrisk_persistence_2026.png`
