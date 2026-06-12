---
title: "MM Structural Maker Directional Decomposition Findings"
tags: [market-making, k5-stress, directionality, structural-maker]
created: 2026-06-04
status: neutral-liquidity-gate-failed-for-nonpolitics
---

# MM Structural Maker Directional Decomposition Findings

> Hub: [[strat_market_making]] · [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

This note tests whether the K5-STRESS structured-maker edge survives after removing wallets whose lifetime behavior looks directional rather than neutral/offsetting. It joins structured non-top3 wallet-market rows to `traders_directionality.parquet` and reports the neutral `arb_like` subset separately from `two_sided_directional`, `mixed`, and other styles. The answer is strict: the clean neutral subset is empty for sports-like, equities up/down, and the read-only politics comparison row, and tiny/negative for residual misc. The non-politics sleeves therefore do not qualify as clean neutral market-making sleeves for a live candidate screen.

## Design

The unit of observation is a structured non-top3 wallet-market row from the K5-STRESS wallet-market cache. For the `other:misc_other` fallback, the row is first retagged into real sleeves, then the structured playbook is requalified at the retagged sleeve level: two-sided maker USD share at least 60%, carry-token share at least 50%, spike-zone USD share at most 2%, and not a global top-3 maker in that market.

The directionality join uses `traders_directionality.parquet` keyed by wallet address:

- `neutral_arb_like`: `primary_style == "arb_like"`, the strict neutral/offsetting subset.
- `two_sided_directional`: wallets that trade both sides but remain net directional in the directionality classifier.
- `mixed`: neither clean arb nor clean directional enough to classify.
- `pure_directional` and `unknown_directionality`: reported, but not counted as neutral.

The neutral gate is intentionally simple: the neutral subset must have positive market-bootstrap CI lower bound, positive median wallet bps, and positive net ex-rebate bps. The CI resamples market IDs, not individual rows. This is not an out-of-sample test; it is a contamination gate on an already historical K5-STRESS result.

## Gate Results

| sleeve | subset | wallets | markets | gross | net bps | CI bps | median wallet bps | ex-rebate bps | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sports_like | all structured | 504 | 31,748 | $314.5M | +236.9 | [+103.9, +365.7] | +44.1 | +210.1 | historical positive, not neutral |
| sports_like | neutral arb_like | 0 | 0 | $0 | n/a | n/a | n/a | n/a | FAILS_NEUTRAL_GATE |
| sports_like | two_sided_directional | 247 | 26,647 | $250.9M | +240.8 | [+79.1, +402.8] | +15.3 | +214.7 | directional carrier |
| residual_misc | all structured | 638 | 108,113 | $494.8M | +111.3 | [+40.9, +179.1] | +16.2 | +88.0 | historical positive, not neutral |
| residual_misc | neutral arb_like | 2 | 1,094 | $69.1k | -298.3 | [-996.2, +367.2] | -280.6 | -316.5 | FAILS_NEUTRAL_GATE |
| residual_misc | two_sided_directional | 293 | 97,768 | $396.5M | +113.7 | [+30.7, +192.9] | +14.6 | +91.5 | directional carrier |
| equities_updown_close_spx_ndx | all structured | 34 | 54 | $84.8k | +245.6 | [-398.0, +919.9] | +269.5 | +205.9 | thin and not neutral |
| equities_updown_close_spx_ndx | neutral arb_like | 0 | 0 | $0 | n/a | n/a | n/a | n/a | FAILS_NEUTRAL_GATE |
| equities_updown_close_spx_ndx | two_sided_directional | 19 | 51 | $57.2k | +154.6 | [-468.1, +835.3] | +256.8 | +112.7 | thin directional carrier |
| politics_negrisk_read_only | all structured | 67 | 4,642 | $179.4M | +2,290.2 | [+877.1, +3,745.0] | +14.5 | +2,276.2 | read-only comparison |
| politics_negrisk_read_only | neutral arb_like | 0 | 0 | $0 | n/a | n/a | n/a | n/a | read-only: no neutral row |
| politics_negrisk_read_only | two_sided_directional | 42 | 4,383 | $125.8M | +1,462.0 | [+362.3, +2,346.8] | +97.7 | +1,448.0 | read-only directional carrier |

## Read

Sports-like and residual misc remain historically positive after the retag, but their positive denominator is not a neutral-liquidity denominator. Sports-like has no `arb_like` wallets in the structured retagged row set; nearly 80% of gross sits in `two_sided_directional` wallets and another 20% in `mixed` wallets. Residual misc has only two `arb_like` wallets and that slice is negative.

Equities up/down was already thin in [[mm_equities_updown_structural_scope_findings]]. This decomposition makes it worse for a neutral-MM interpretation: the neutral subset is empty, and the positive point estimate lives in underpowered directional/mixed rows.

The politics comparison row is included only as a shared diagnostic for the separate politics lane. It does not change the politics verdict in [[mm_politics_negrisk_accounting_findings]]. It does show the same contamination pattern: the corrected-carry historical edge is not carried by wallets classified as neutral/offsetting.

## Decision

The clean neutral-liquidity gate fails for the non-politics sleeves requested here. The historical K5-STRESS structured-maker edge should be described as a real historical structured-maker pattern, but not promoted into a reproducible neutral market-making sleeve without a separate live measurement loop proving that the directional component is not required.

## Artifacts

- Directional decomposition CSV: `data/analysis/csv_outputs/market_making/mm_structural_maker_directional_decomposition.csv`
- Style mix CSV: `data/analysis/csv_outputs/market_making/mm_structural_maker_directionality_style_mix.csv`
- Retag summary CSV: `data/analysis/csv_outputs/market_making/mm_nonpolitics_misc_other_retag_gate.csv`
