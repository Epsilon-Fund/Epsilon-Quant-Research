---
title: "MM Non-Politics Target Screen Findings"
tags: [market-making, non-politics, misc-other, target-screen]
created: 2026-06-04
status: no-live-screen-candidates-after-directional-gate
---

# MM Non-Politics Target Screen Findings

> Hub: [[strat_market_making]] · [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

This note retags the `other:misc_other` fallback cell and applies the non-politics target-screen discipline requested after the directional decomposition. Retagging confirms that `misc_other` contained several historically positive structured non-top3 sleeves, especially sports-like and residual misc. But [[mm_structural_maker_directional_decomposition_findings]] gates the live screen: sports-like, residual misc, and equities up/down do not survive on the neutral/offsetting subset. Result: no non-politics sleeve graduates to a live Gamma/CLOB candidate list in this batch.

## Retag Design

The original deployable-cell script tagged `other:misc_other` as the fallback after obvious politics, macro, AI/tech, weather, crypto, and finance markers. This pass keeps the same original fallback definition, then splits it into:

- `sports_like`: recurring game/team/league language and common sports slug prefixes.
- `residual_misc`: fallback after the new retags.
- `politics_news_like`: politics/news language. This is out of this lane and should be read by the politics chat only.
- `culture_like`: Oscars, media, celebrity, music, social-platform, and entertainment language.
- `business_tech_like`: business, tech, AI, stock, crypto, macro-business language that was missed by the original fallback.

For each retagged sleeve, structured wallets are requalified at the sleeve level before reporting PnL. This prevents a wallet that was structured across the whole old `misc_other` bucket from being counted as structured in a tiny sub-sleeve where it was not actually two-sided/carrying.

## Retag Results

| sleeve | wallets | markets | gross | net PnL | net bps | CI bps | median wallet bps | ex-rebate bps | read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| residual_misc | 638 | 108,113 | $494.8M | $5.50M | +111.3 | [+41.7, +177.2] | +16.2 | +88.0 | historically positive, but fails neutral gate |
| sports_like | 504 | 31,748 | $314.5M | $7.45M | +236.9 | [+117.4, +374.2] | +44.1 | +210.1 | strongest non-politics historical row, but fails neutral gate |
| politics_news_like | 267 | 1,810 | $20.0M | $828.0k | +413.1 | [-292.8, +1,047.4] | +26.3 | +397.4 | out of lane; hand to politics only as context |
| culture_like | 220 | 2,005 | $12.7M | $702.3k | +554.7 | [+107.1, +1,148.9] | +93.4 | +545.5 | not part of requested live screen; needs neutral gate before any action |
| business_tech_like | 109 | 2,257 | $5.7M | $135.1k | +235.0 | [+74.1, +466.9] | +77.5 | +227.6 | not part of requested live screen; needs neutral gate before any action |

## Candidate Screen

The requested live Gamma/CLOB five-screen filter was conditional on a sleeve surviving the directional gate. None of the requested non-politics screen sleeves survived that gate.

| sleeve | historical edge | directional gate | live Gamma/CLOB screen | measurement-loop candidate |
|---|---|---|---|---|
| sports recurring / sports_like | positive: +236.9 bps, CI positive | failed: neutral subset empty | not run because precondition failed | no |
| residual_misc | positive: +111.3 bps, CI positive | failed: neutral subset tiny and negative | not run because precondition failed | no |
| equities up/down | point-positive but CI-crossing and thin | failed: neutral subset empty | not run because precondition failed | no |
| politics_news_like | out of lane | out of lane | out of lane | politics lane only |

This is an intentionally empty candidate list, not a data-fetch failure. The five-screen filter should not be applied to sleeves that already fail the neutral-liquidity gate, because doing so would convert a directional historical pattern into a live quote list.

## Decision

Do not start non-politics capture from this batch. The next valid non-politics measurement step would require either a new sleeve that first survives the neutral/offsetting decomposition, or an explicit strategic decision to test directional-maker behavior rather than neutral MM. Under the current prompt, sports-like, residual misc, and equities up/down are not measurement-loop candidates.

## Artifacts

- Retag summary CSV: `data/analysis/csv_outputs/market_making/mm_nonpolitics_misc_other_retag_gate.csv`
- Directional decomposition CSV: `data/analysis/csv_outputs/market_making/mm_structural_maker_directional_decomposition.csv`
- Original deployable-cell CSV: `data/analysis/csv_outputs/market_making/mm_deployable_cells.csv`
