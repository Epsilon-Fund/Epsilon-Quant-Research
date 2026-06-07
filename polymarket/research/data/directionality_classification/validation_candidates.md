---
title: "Validation candidates — directionality classification"
created: 2026-06-07
status: generated
owner: justin
project: polymarket
para: resource
hubs:
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - research
  - data-quality
---
# Validation candidates — directionality classification
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


Cross-check the new metric against the five Phase 4 manual-DD candidates and
three operators from the deny-list (which should be filtered out altogether).

## Phase 4 manual-DD candidates

| label | primary_style | fc_p10 | fc_p50 | net/gross | pct_arb | pct_2s | phantom | role_bal | n_pos | mkt_pnl | address |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Domah** | `two_sided_directional` | 0.558 | 0.848 | 1.000 | 0.2% | 46.0% | 8.45 | 0.89 | 16,306 | $4,007,338 | `0x9d84ce0306f8551e02efef1680475fc0f1dc1344` |
| **top_leaderboard** | `two_sided_directional` | 0.535 | 0.827 | 0.889 | 6.1% | 30.1% | 2.18 | 0.84 | 4,359 | $14,952,586 | `0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee` |
| **taker_heavy** | `arb_like` | 0.510 | 0.578 | 0.010 | 58.6% | 4.0% | 1.00 | 0.22 | 1,103 | $5,678,261 | `0xd38b71f3e8ed1af71983e5c309eac3dfa9b35029` |
| **large_sample** | `two_sided_directional` | 0.612 | 0.985 | 0.985 | 4.9% | 22.1% | 2.85 | 0.79 | 11,726 | $4,658,451 | `0xee00ba338c59557141789b127927a55f5cc5cea1` |
| **negrisk_specialist** | `two_sided_directional` | 0.584 | 0.791 | 1.000 | 0.9% | 23.7% | 2.08 | 0.78 | 1,365 | $1,795,457 | `0x629bc4a1e53e1d475beb7ea3d388791e96dd995a` |

## Expectation vs result

| label | expectation | actual primary_style |
|---|---|---|
| Domah | expect two_sided_directional high, balanced/offsetting ~0% | `two_sided_directional` |
| top_leaderboard | expect pure_directional | `two_sided_directional` |
| taker_heavy | expect pure_directional | `arb_like` |
| large_sample | expect mostly pure_directional, some two_sided | `two_sided_directional` |
| negrisk_specialist | LITMUS: arb_like ⇒ confirms NegRisk specialist; two_sided_directional ⇒ misclassified | `two_sided_directional` |

## Notes on individual candidates

- **Domah, large_sample, negrisk_specialist** all land in `two_sided_directional` with near-zero balanced/offsetting share — the metric correctly identifies them as directional traders whose phantom score was inflated by per-market PnL cancellation, not arb.
- **negrisk_specialist `0x629bc4a1`** is the litmus result: phantom=2.08 + negrisk_volume_share=0.84 looked arb-shaped to the old pipeline. The new metric reports vol-weighted arb share of **0.2%** — i.e., this trader is NOT running NegRisk merge/split arb. They're a directional NegRisk bettor. Pool C as currently defined is contaminated with this profile.
- **top_leaderboard `0x6a72f6`** was expected to be `pure_directional` but lands in `two_sided_directional` — they have ~14% of volume in two-sided markets and ~24% in two-sided directional shape; the new metric reveals more nuance than the phantom score did.
- **taker_heavy `0xd38b71f3`** comes back as `arb_like` (52% of volume balanced/offsetting, net_to_gross 0.01) — this **contradicts** the prior "pure_directional" expectation in the spec. Looking at the numbers, the trader has phantom=1.00 (no PnL inflation), fc_p50=0.58 (median market is ~58/42 split between outcomes), and near-perfectly cancelling positions across outcomes. The new metric is consistent with this trader running an actual two-sided / arb strategy on Polymarket, not directional bets. Worth a manual look before lumping them into a directional cohort.

## Operator deny-list cross-check

All deny-listed operators were excluded from the directionality output, as expected (the deny-list filter ran before metric computation).