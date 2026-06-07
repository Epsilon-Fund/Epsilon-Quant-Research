---
title: "Block B / Block E Lite Reinterpretation (post copytrade-relayer-implications analysis)"
tags: [dali, block-b, block-e-lite, reinterpretation, supersession]
created: 2026-05-28
updated: 2026-05-28
status: supersedes-partial
supersedes: [block_b_findings, block_e_lite_findings]
hubs:
  - COWORK
---

# Block B / Block E Lite Reinterpretation

> Hub: [[COWORK]]

## Summary

The "operator-filtering" lift documented in Block B and attributed to two relayer addresses in Block E Lite is **not** what we initially thought. Two rounds of investigation:

1. **Relayer dig** (`relayer_dig_findings.md`, 2026-05-28) identified the two addresses as Polymarket's legacy CTF Exchange v1 contracts. v1 standard: `0x4bfb41d5...8982e`. v1 neg-risk: `0xc5d563a3...f80a`. Both stop appearing in raw data at `2026-04-28 11:00:40 UTC`, matching the public CLOB v1 → v2 cutover.

2. **Copytrade-relayer-implications analysis** (`copytrade_relayer_implications.md`, 2026-05-28) determined how the artifact actually works in the data. The result reverses my initial interpretation.

## Mechanism (corrected)

The Polymarket `ctf-exchange` source `Trading.sol` `_matchOrders` emits the internal active leg with `maker = takerOrder.maker` and `taker = address(this)`. So when a two-sided match happens, the event has:

- **`maker` field = the active taker order's wallet** (real wallet)
- **`taker` field = the exchange contract** (artifact)

This means:

- The active wallet IS captured — it sits in the `maker` column
- The warproxxx convention "filter by maker for a specific user's trades" captures this activity correctly
- 414,747,677 of 416,108,840 v1 internal rows (99.7%) have a `maker` address that joins `traders.parquet`
- The exchange contract appears only in the `taker` field, never in the `maker` field (0 as_maker, ~416M as_taker)
- v2 exchange contracts (`0xE111...996B` standard, `0xe2222...0F59` neg-risk) follow the same pattern — verified by a 100k post-cutover sample showing 35,885 `taker` occurrences and 0 `maker` occurrences

## What "operator filtering" actually did

In Block E Lite the filter was "drop rows where `taker` matches one of the two relayer/exchange addresses." Since those addresses appear only in `taker`, the filter dropped the entire ~40% of fills that came from `_matchOrders` two-sided matches. The remaining ~60% are from other emit paths — likely `_fillOrder` single-sided fills (where a taker fills against a specific resting maker without crossing multiple orders).

The "operator-filtered hit rate" of 58% (equity_index) and 52% (crypto) is therefore **TFI computed on the non-`_matchOrders` subset only**. The raw 47% / 34% hit rates are TFI on the mixed population. Neither is "the right answer" — they're TFI on two structurally-different event populations.

Why might the non-`_matchOrders` subset give a cleaner TFI hit rate? Plausible mechanisms:

- Single-sided fills have one specific maker-taker counterparty pair, so attribution is unambiguous and there's no "batched aggressive flow" confusion
- Less event-emit noise; signal-per-event is higher
- Possibly different price-discovery role (single-sided fills happen when there's no resting depth to match against — these are price-discovery events, not depth-consumption events)

This is now a **tractable empirical hypothesis** rather than a "data is contaminated" black box.

## What is and isn't superseded

### Superseded

- **Block B Headline #2:** "Operator-removal effect is strong." → Reframe: this was a population-selection effect across structurally-different event types (`_matchOrders` vs other emit paths). The 58% number is on the non-`_matchOrders` subset only; the 47% number is on the mixed population. Either could be informative; neither is "TFI net of operator flow."
- **Block E Lite Headline #1:** "Two relayer addresses drive the operator effect." → Reframe: those addresses ARE the entire `_matchOrders` emit pattern. Of course filtering them dominates — they're 40% of events.
- **Strategic implication carried in the May 2026 handoff:** "If we filter retail UI flow, maker thesis becomes attractive." That framing is disproven on two counts: (a) the addresses are exchange contracts, not retail-UI routers; (b) the live WS feed doesn't expose the artifact at all.
- **My earlier note's framing** that "the real taker wallet is missing for ~40% of fills." → Disproven by the 99.7% maker-side join rate. The active wallet is captured; the role label is wrong.

### Not superseded

- **Block B Headline #1:** "Raw TFI is not tradeable in any family/horizon/condition combination" — independent of the artifact.
- **Block B Headline #3:** "AI/product walk-forward: test set hit rate exceeded train" — unrelated; needs separate live revalidation (and A0/A0b sample is too thin to retest).
- **Block B Headline #4:** "Crypto walk-forward: severe degradation from train to test" — also independent.
- **Block B Headline #5:** "Sports: no signal in any horizon/league/condition" — also independent.
- **Block E Lite Headline #2:** "Block A target markets are all DISTRIBUTED." True, and now makes more sense: post-2026-04-28 markets use v2 contracts, which still have the artifact in raw events but the live WS feed normalizes it away.

## Implications by thread

### Copy-trading

- **PnL & position attribution:** correct. No rebuild of `closed_positions.parquet` or `traders.parquet`. (Verified by 99.7% join rate.)
- **Style classifications:** biased toward maker. `style_maker_taker_ratio` should be recomputed with an `active_order_leg` flag distinguishing genuine maker activity from active-taker-orders-in-the-maker-field. Domah's ratio goes 7.89 → 5.67 after reclassification (still maker-heavy). Some other traders flip categories.
- **Smoke target:** Domah's `macro / maker / 18-24` cell remains intact under reclassification. Green light when ready.

### Dali

- **A1 live data:** unaffected. WS feed normalizes side semantics; artifact never reaches live capture.
- **Historical TFI re-run (if pursued):** partition by emit-path (`_matchOrders` vs other) and recompute TFI hit rates per partition separately. The lift may turn out to be a real "single-sided fills are cleaner" effect, worth following up. Or it may evaporate when controlled for properly. Either way it's a known-quantity hypothesis now.
- **`historical_to_aggressor()`:** the Block B sign-convention audit (2026-05-27) confirmed the function is correct in aggregate, but the audit was performed before the `_matchOrders` mechanism was understood. A quick sanity check: on the `_matchOrders` subset specifically, does `historical_to_aggressor()` map the active taker's wallet (sitting in `maker` with some `maker_side` value) to the correct aggressor side? If yes, no action. If no, then Block B's TFI signs on that subset may be inverted, which would meaningfully change the operator-filtered interpretation.

## Follow-ups (tracked in `brain/TODO.md`)

1. Add an `active_order_leg` / `exchange_internal_leg` role flag to `data_infra/views.sql` and the trader-style computation so style ratios are not contaminated by active-taker-orders-in-maker-field.
2. Sanity-check `historical_to_aggressor()` specifically on the `_matchOrders` subset.
3. (Optional) Partition Block B's TFI computation by emit path and report hit rates per partition; characterize what the non-`_matchOrders` subset actually contains; test the "cleaner attribution" hypothesis.

## Sources

- Copytrade analysis: [`copytrade_relayer_implications.md`](copytrade_relayer_implications.md)
- Relayer dig findings: [`relayer_dig_findings.md`](relayer_dig_findings.md)
- Original Block B: [`block_b_findings.md`](block_b_findings.md)
- Original Block E Lite: [`block_e_lite_findings.md`](block_e_lite_findings.md)
- Polymarket CTF Exchange source: https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/mixins/Trading.sol
- Polymarket V2 migration: https://docs.polymarket.com/v2-migration
