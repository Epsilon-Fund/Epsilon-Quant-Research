---
title: "MM Equities Up/Down Structural Scope Findings"
tags: [market-making, equities, index-updown, k5-stress, structural-maker]
created: 2026-06-03
status: thin-data-live-collector-needed
---

# MM Equities Up/Down Structural Scope Findings

> Hub: [[strat_market_making]] · [[COWORK]]

## Summary

- Scope: MM Equities Up/Down Structural Scope Findings in the MM/market-making area.
- Existing takeaway/status: THIN-DATA.** Equity index up/down should not be reopened as an OD pricing branch; [[od_equities_index_pricing_scope_findings]] already closed that angle for SPX close-style markets. The untested question here was MM structural making: can a non-top3 maker earn spread plus carry-to-resolution, using the same K5-STRESS structured playbook that survived in [[mm_politics_negrisk_accounting_findings]]?
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Verdict

**THIN-DATA.** Equity index up/down should not be reopened as an OD pricing branch; [[od_equities_index_pricing_scope_findings]] already closed that angle for SPX close-style markets. The untested question here was MM structural making: can a non-top3 maker earn spread plus carry-to-resolution, using the same K5-STRESS structured playbook that survived in [[mm_politics_negrisk_accounting_findings]]?

The local raw fill tape has enough equity index activity to scope the question, but the **structured non-top3 settled maker sample is too small to judge offline**. The preferred close-style SPX/NDX cut has **34 target wallets, 54 target markets, 3,625 maker fills, and only $84.8k gross** after the two-sided/carry/spike/non-top3 filters. It is point-positive (**+246 bps**) and median-wallet-positive (**+269 bps**), but market-cluster CI crosses zero (**[-403, +838] bps**). That is not a live edge proof, and not a clean close either.

Decision: **do not force a powerless cut.** Recommend a live measurement collector for close-style SPX daily up/down first, with ES/MES post-fill drift logging. NDX, SPY/QQQ, and opens can be shadow-scoped, but they are thinner or wider and should not anchor the first MM sleeve.

## Design

This pass mirrors the politics structured-maker method, but without NegRisk conversion accounting because ordinary SPX/NDX up/down is a two-outcome non-NegRisk market.

Rules:
- Scope: equity index up/down markets discovered from the existing OD equity scope and Dali candidate tables. Preferred scope is close-style `spx-up-or-down-*` and `ndx-up-or-down-*`; SPY/QQQ and SPX opens are secondary diagnostics only.
- PnL source: `k5_stress_wallet_market_full.parquet`, which contains K5-STRESS wallet-market realized PnL, maker rebates, taker fees, two-sided/carry/spike fields, and market top-3 flags.
- No mark-to-mid: this pass uses only rows where `mark_source` contains `settlement`, so open/latest-trade-mark rows are excluded.
- Structured playbook: two-sided USD share >=60%, carry-token share >=50%, spike-zone USD share <=2%.
- Deployability cut: exclude each market's global top-3 maker wallets.
- Gate columns: net bps, market-cluster CI, median wallet bps, and net ex rebate bps, matching the politics note.

Power guardrail for a decision-grade offline cut: at least 50 target markets, 30 target wallets, and $250k structured non-top3 gross. The primary SPX/NDX cut barely clears market/wallet counts but fails dollar denominator badly.

## Coverage Check

Raw fills below exclude exchange-internal legs. K5 settled coverage is smaller than raw coverage because the K5 cache is maker-heavy wallet-market reconstruction, not every wallet in the raw tape.

| scope | discovered markets | raw fills | raw USD | raw markets | raw maker wallets | K5 settled markets | K5 settled maker fills | K5 settled maker USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SPX close up/down | 58 | 80,061 | $4.37M | 53 | 3,232 | 32 | 20,972 | $763.8k |
| NDX close up/down | 23 | 4,288 | $101.1k | 23 | 348 | 23 | 2,400 | $27.8k |
| SPX+NDX close | 81 | 84,349 | $4.47M | 76 | 3,580 | 55 | 23,372 | $791.7k |
| SPX opens diagnostic | 63 | 61,556 | $5.76M | 57 | 3,222 | 36 | 19,980 | $1.24M |
| SPY/QQQ diagnostic | 17 | 9,183 | $281.9k | 17 | 816 | 17 | 3,600 | $117.7k |

Read: the coverage gate should not stop at raw fills. SPX close has real flow, but once we ask the actual MM question -- structured, non-top3, settled maker PnL -- the usable denominator shrinks sharply.

## Structured Non-Top3 Gate

Unit of observation is a structured maker wallet-market row carried to settlement. CI resamples market clusters, so a single active day cannot masquerade as hundreds of independent observations.

| cut | target wallets | target markets | gross | net bps | CI bps | median wallet bps | net ex rebate bps | power |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SPX+NDX close, preferred | 34 | 54 | $84.8k | +246 | [-403, +838] | +269 | +206 | thin: gross <$250k |
| SPX close only | 33 | 32 | $81.5k | +311 | [-263, +1,045] | +352 | +272 | thin: markets <50, gross <$250k |
| NDX close only | 12 | 22 | $3.7k | -1,971 | [-4,052, +192] | -1,296 | -2,018 | thin: markets/wallets/gross |
| SPX opens diagnostic | 29 | 36 | $90.7k | +110 | [-240, +695] | +228 | +79 | thin and wider live book |
| SPY/QQQ diagnostic | 12 | 17 | $14.4k | +681 | [-281, +1,698] | +959 | +646 | too thin |
| QQQ diagnostic only | 11 | 8 | $4.3k | +723 | [+199, +1,830] | +1,094 | +682 | do not promote: only 8 markets |

Read: the primary close-style SPX/NDX row looks like an underpowered cousin of the politics result, not like a confirmed sleeve. It is positive after removing rebates, which is good, but the CI crosses zero and the gross denominator is only **$84.8k**. The QQQ diagnostic lower CI is positive, but at **8 markets / $4.3k gross** it is exactly the sort of tiny cut the realism rules tell us not to promote.

## Adverse-Selection Environment

The prompt's central live hypothesis is right: equities have a clean continuous reference in ES/MES, so post-fill drift can be measured better than in many politics markets and without the source-basis problems that complicated crypto. But the local cache cannot do that exact offline test.

Data audit:

| ingredient | available locally | usable now | detail |
|---|---|---|---|
| PM settled maker fills / K5 wallet-market PnL | yes | yes | Used here, settlement rows only. |
| PM historical CLOB books around equity-index fills | no | no | Local data has fills, not quote/cancel/queue history for SPX/NDX close-style up/down. |
| ES/MES intraday futures at fill timestamps | no | no | No local ES/MES intraday parquet/CSV. Prior OD audit only had a recent CME settlement probe, not fill-aligned futures states. |
| SPX cash / realized-vol proxy from OD pricing pass | yes | no for this question | Useful context, but not the requested continuous ES/MES adverse-selection reference. |

So the ES/MES adverse-selection comparison is a live-collector task, not an offline conclusion. The collector should log PM quote/fill/queue state and ES/MES mid at fill time plus short horizons such as 1m, 5m, 15m, 60m, and to official close. Then compare maker PnL and drift by two-sided/carry/spike status against the politics and crypto K5 analogs.

## Capacity As Assumption

No capacity EV ladder is reported because no equity up/down structured-maker cut clears the offline gate. The right Rule-3 framing is:

- Capacity is plausible at `$10-$100` measurement scale for SPX close-style up/down; the OD scope already found small-capacity flow and tight live spread.
- Edge is not proven offline; the structured-maker sample is too small after the non-top3/carry/two-sided filters.
- A future live loop should treat fill share, queue position, adverse selection, and non-incumbent capacity as measured quantities, not as assumptions promoted from this cache.

## Modeled Assumptions

- The K5-STRESS wallet-market cache is the correct no-mark-to-mid PnL source when restricted to settlement rows.
- The same structured playbook thresholds used for politics are appropriate for this first MM scope: two-sided >=60%, carry >=50%, spike <=2%.
- `$250k` structured non-top3 gross is a minimal offline power guardrail for a decision-grade cell; below that, a few events can dominate bps.
- Exchange-internal legs are excluded from raw coverage.

## Live-Only Unknowns

- Whether a non-incumbent can get filled passively near the SPX up/down top of book without becoming the adverse-selection sink.
- Queue position, cancels, missed fills, and true fill share versus incumbent top makers.
- ES/MES post-fill drift and whether it is dodgeable by quote pull/widen rules.
- Whether the positive SPX point estimate persists once measured prospectively with quote telemetry.
- Whether opens or SPY/QQQ have separate structural value despite being thinner/wider in this cache.

## Artifacts

- Script: `scripts/mm_equities_updown_structural_scope.py`
- Candidate markets: `data/analysis/csv_outputs/market_making/mm_equities_updown_structural_scope_markets.csv`
- Coverage: `data/analysis/csv_outputs/market_making/mm_equities_updown_structural_scope_coverage.csv`
- Structured gate: `data/analysis/csv_outputs/market_making/mm_equities_updown_structural_scope_gate.csv`
- Adverse-selection data audit: `data/analysis/csv_outputs/market_making/mm_equities_updown_structural_scope_adverse_selection_audit.csv`
