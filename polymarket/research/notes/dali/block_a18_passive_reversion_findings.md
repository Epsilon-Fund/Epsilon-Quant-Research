---
tags: [dali, a18, reversion, passive-maker, microprice, results]
---

# Block A18 Passive Reversion-To-Microprice Findings

> Hub: [[COWORK]]

## Summary

- Scope: Block A18 Passive Reversion-To-Microprice Findings in the Dali research lineage area.
- Existing takeaway/status: No pooled route clears the pre-registered market-cluster lower-CI > 0 gate. Best pooled row by lower bound: `rolling_rank_sizing` / `passive_maker` / W=5 / H=30 with conditional EV `-1.232c` CI `[-1.631c, -0.924c]`, n `8,540`, executed fill `0.09%`.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Read Trail

- [[block_a1x_external_note_reconciliation]]
- [[block_a13_tob_imbalance_findings]]
- [[block_a15b_decoupled_findings]]
- [[block_a16_binary_bet_findings]]
- [[block_a14h_maker_non_overlap_findings]]

## Headline

No pooled route clears the pre-registered market-cluster lower-CI > 0 gate. Best pooled row by lower bound: `rolling_rank_sizing` / `passive_maker` / W=5 / H=30 with conditional EV `-1.232c` CI `[-1.631c, -0.924c]`, n `8,540`, executed fill `0.09%`.

Verdict: **CONFIRM-CLOSE: continuation is dead and explicit reversion also fails the market-cluster CI gate.**

This closes only the local Dali TOB signal under this explicit reversion framing. It does not touch off-book cross-market lead-lag or true-L2 ideas.

## Deciding Number

| mapping | route | fill | exit | markets | exec | exec fill | cond EV | cond CI | uncond EV | uncond CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_rank_sizing | passive maker | W=5 | H=30 | 55/75 | 8,540 | 0.09% | -1.232c | [-1.631c, -0.924c] | -0.001c | [-0.001c, -0.000c] |
| rolling_rank_sizing | passive maker | W=1 | H=30 | 55/75 | 8,373 | 0.09% | -1.201c | [-1.652c, -0.911c] | -0.001c | [-0.001c, -0.000c] |
| rolling_rank_sizing | passive maker | W=1 | H=60 | 55/75 | 6,762 | 0.07% | -1.240c | [-1.658c, -0.899c] | -0.001c | [-0.001c, -0.000c] |
| rolling_rank_sizing | passive maker | W=10 | H=30 | 55/75 | 8,657 | 0.09% | -1.242c | [-1.700c, -0.952c] | -0.001c | [-0.001c, -0.000c] |
| rolling_rank_sizing | passive maker | W=5 | H=60 | 55/75 | 6,889 | 0.07% | -1.260c | [-1.714c, -0.973c] | -0.001c | [-0.001c, -0.000c] |
| rolling_rank_sizing | passive maker | W=1 | H=10 | 55/75 | 10,716 | 0.11% | -1.245c | [-1.719c, -0.948c] | -0.001c | [-0.001c, -0.001c] |
| rolling_rank_sizing | passive maker | W=1 | H=5 | 55/75 | 11,698 | 0.12% | -1.277c | [-1.724c, -0.953c] | -0.001c | [-0.001c, -0.001c] |
| rolling_rank_sizing | passive maker | W=10 | H=60 | 55/75 | 6,950 | 0.07% | -1.284c | [-1.726c, -0.977c] | -0.001c | [-0.001c, -0.000c] |

Rows with conditional market-cluster CI lower > 0: `0`. Rows with unconditional-after-fill CI lower > 0: `0`. Worst executed one-contract loss in this replay: `-93.464c`.

## Test 1: Binary Decile Fade To Microprice

| mapping | route | fill | exit | markets | exec | exec fill | cond EV | cond CI | uncond EV | uncond CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary_decile | passive maker | W=1 | H=5 | 54/64 | 3,394 | 0.18% | -1.902c | [-2.454c, -1.463c] | -0.003c | [-0.005c, -0.002c] |
| binary_decile | passive maker | W=5 | H=5 | 54/64 | 3,725 | 0.19% | -1.955c | [-2.505c, -1.480c] | -0.004c | [-0.006c, -0.002c] |
| binary_decile | passive maker | W=10 | H=5 | 54/64 | 3,912 | 0.20% | -2.016c | [-2.552c, -1.575c] | -0.004c | [-0.006c, -0.003c] |
| binary_decile | taker | W=0 | H=5 | 64/64 | 65,619 | 3.39% | -5.662c | [-7.337c, -4.007c] | -0.192c | [-0.280c, -0.128c] |

## Test 2: Rolling-Rank Sizing

| mapping | route | fill | exit | markets | exec | exec fill | cond EV | cond CI | uncond EV | uncond CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_rank_sizing | passive maker | W=1 | H=5 | 55/75 | 11,698 | 0.12% | -1.277c | [-1.724c, -0.953c] | -0.001c | [-0.001c, -0.001c] |
| rolling_rank_sizing | passive maker | W=5 | H=5 | 55/75 | 12,045 | 0.12% | -1.322c | [-1.759c, -0.978c] | -0.001c | [-0.001c, -0.001c] |
| rolling_rank_sizing | passive maker | W=10 | H=5 | 55/75 | 12,238 | 0.13% | -1.354c | [-1.877c, -1.031c] | -0.001c | [-0.001c, -0.001c] |
| rolling_rank_sizing | taker | W=0 | H=5 | 74/75 | 214,391 | 2.21% | -3.664c | [-4.805c, -2.794c] | -0.043c | [-0.057c, -0.032c] |

The rolling-rank variant trades the same sign as a fade, but sizes each attempted position by `abs(rolling_rank)` on the 300s percentile scale. Conditional EV is reported per actual sized contract; unconditional EV is sized cents per signal opportunity after passive fill and non-overlap drag.

## Family Diagnostics

| family | mapping | route | cell | markets | exec | exec fill | cond EV | CI | uncond |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geopolitics_policy | rolling_rank_sizing | passive maker | W=10 H=30 | 8/8 | 1,336 | 0.08% | -0.273c | [-0.388c, -0.113c] | -0.000c |
| geopolitics_policy | rolling_rank_sizing | passive maker | W=10 H=5 | 8/8 | 1,616 | 0.10% | -0.288c | [-0.396c, -0.079c] | -0.000c |
| geopolitics_policy | rolling_rank_sizing | passive maker | W=10 H=10 | 8/8 | 1,540 | 0.09% | -0.284c | [-0.398c, -0.097c] | -0.000c |
| geopolitics_policy | rolling_rank_sizing | passive maker | W=5 H=30 | 8/8 | 1,319 | 0.08% | -0.276c | [-0.401c, -0.118c] | -0.000c |
| geopolitics_policy | rolling_rank_sizing | passive maker | W=1 H=10 | 8/8 | 1,498 | 0.09% | -0.283c | [-0.409c, -0.105c] | -0.000c |
| geopolitics_policy | rolling_rank_sizing | passive maker | W=5 H=5 | 8/8 | 1,595 | 0.10% | -0.290c | [-0.413c, -0.113c] | -0.000c |
| geopolitics_policy | rolling_rank_sizing | passive maker | W=1 H=5 | 8/8 | 1,566 | 0.10% | -0.286c | [-0.413c, -0.110c] | -0.000c |
| geopolitics_policy | rolling_rank_sizing | passive maker | W=5 H=10 | 8/8 | 1,519 | 0.09% | -0.287c | [-0.419c, -0.115c] | -0.000c |
| geopolitics_policy | rolling_rank_sizing | passive maker | W=1 H=30 | 8/8 | 1,302 | 0.08% | -0.275c | [-0.422c, -0.109c] | -0.000c |
| geopolitics_policy | rolling_rank_sizing | passive maker | W=5 H=60 | 8/8 | 1,111 | 0.07% | -0.315c | [-0.489c, -0.072c] | -0.000c |
| geopolitics_policy | rolling_rank_sizing | passive maker | W=10 H=60 | 8/8 | 1,119 | 0.07% | -0.319c | [-0.502c, -0.121c] | -0.000c |
| geopolitics_policy | rolling_rank_sizing | passive maker | W=1 H=60 | 8/8 | 1,095 | 0.07% | -0.311c | [-0.510c, -0.111c] | -0.000c |

Family and market rows are diagnostics, not kill-switches by themselves. The gate was pooled route/horizon with market-cluster CI.

## Best Single-Market Diagnostics

| market | slug | mapping | route | cell | exec | exec fill | cond EV | uncond |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0c:561229 | will-jd-vance-win-the-2028-us-preside. | rolling_rank_sizing | passive maker | W=1 H=5 | 92 | 0.01% | 0.000c | 0.000c |
| a0c:561229 | will-jd-vance-win-the-2028-us-preside. | rolling_rank_sizing | passive maker | W=5 H=5 | 92 | 0.01% | 0.000c | 0.000c |
| a0c:561229 | will-jd-vance-win-the-2028-us-preside. | rolling_rank_sizing | passive maker | W=10 H=5 | 92 | 0.01% | 0.000c | 0.000c |
| a0c:561229 | will-jd-vance-win-the-2028-us-preside. | rolling_rank_sizing | passive maker | W=1 H=10 | 84 | 0.01% | 0.000c | 0.000c |
| a0c:561229 | will-jd-vance-win-the-2028-us-preside. | rolling_rank_sizing | passive maker | W=5 H=10 | 84 | 0.01% | 0.000c | 0.000c |
| a0c:561229 | will-jd-vance-win-the-2028-us-preside. | rolling_rank_sizing | passive maker | W=10 H=10 | 84 | 0.01% | 0.000c | 0.000c |
| a0c:561229 | will-jd-vance-win-the-2028-us-preside. | rolling_rank_sizing | passive maker | W=1 H=30 | 77 | 0.01% | 0.000c | 0.000c |
| a0c:561229 | will-jd-vance-win-the-2028-us-preside. | rolling_rank_sizing | passive maker | W=5 H=30 | 77 | 0.01% | 0.000c | 0.000c |
| a0c:561229 | will-jd-vance-win-the-2028-us-preside. | rolling_rank_sizing | passive maker | W=10 H=30 | 77 | 0.01% | 0.000c | 0.000c |
| a0c:561229 | will-jd-vance-win-the-2028-us-preside. | rolling_rank_sizing | passive maker | W=1 H=60 | 73 | 0.01% | 0.000c | 0.000c |
| a0c:561229 | will-jd-vance-win-the-2028-us-preside. | rolling_rank_sizing | passive maker | W=5 H=60 | 73 | 0.01% | 0.000c | 0.000c |
| a0c:561229 | will-jd-vance-win-the-2028-us-preside. | rolling_rank_sizing | passive maker | W=10 H=60 | 73 | 0.01% | 0.000c | 0.000c |

Single-market positives do not reopen the signal because they do not have a market-cluster CI. They are useful only as live instrumentation hints if a pooled route had cleared.

## Method

- Input: `data/analysis/block_a1_features.parquet`, runs `a0`, `a0b`, `a0c`, and `a0c_roll`; no new capture.
- Quote states require complete books, finite bid/ask/microprice, positive spread, and `book_staleness_seconds <= 5`.
- Signal: `tob_imbalance_level = direction_factor * tob_imbalance`, plus a 300s rolling percentile transform mapped to `[-1,+1]`.
- Binary mapping: per-market signed top/bottom decile. Rolling mapping: all finite rolling-rank states, with position size `abs(rank)`.
- Direction: fade the signal. If continuation would buy the market-direction token, this sells it; if continuation would sell, this buys it.
- Target: entry-time weighted mid/microprice. Exit is first target touch or timeout. Exits are at bid for longs and ask for shorts.
- Taker route: immediate executable touch entry, taker fee on entry and exit.
- Passive route: post at the touch on the fade side. A long bid fills on a SELL print at or below bid; a short ask fills on a BUY print at or above ask, within W seconds. Entry receives the maker rebate; exit pays taker fee.
- Non-overlap: one open position per market per grid cell. Passive unfilled orders do not block; filled passive orders block from fill to exit.
- CI: market-cluster bootstrap over market clusters. Conditional EV resamples executed PnL per contract; unconditional EV resamples sized PnL divided by all signal opportunities in the sampled markets.

## Data

| run | rows | markets |
| --- | --- | --- |
| a0 | 1,399,032 | 12 |
| a0b | 1,698,775 | 9 |
| a0c | 4,254,020 | 17 |
| a0c_roll | 2,445,782 | 38 |

## Outputs

- `data/analysis/csv_outputs/dali/block_a18_passive_reversion_surface.csv`
- `data/analysis/csv_outputs/dali/block_a18_passive_reversion_market_clusters.csv`
- `data/analysis/csv_outputs/dali/block_a18_passive_reversion_executed.csv`

Elapsed: `208.1s`.
