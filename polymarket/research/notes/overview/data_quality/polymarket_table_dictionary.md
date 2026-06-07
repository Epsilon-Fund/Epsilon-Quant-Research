---
title: "Polymarket Table Dictionary"
created: 2026-06-05
status: active
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
# Polymarket Table Dictionary

> Hub: [[POLYMARKET_BRAIN]] / [[COWORK]]

## Summary

- Scope: Polymarket Table Dictionary in the Polymarket data-quality area.
- Existing takeaway/status: This is the shared definition note for compact Polymarket table columns, bucket labels, filters, and indicators. Use it to keep findings notes readable without repeating the same glossary everywhere.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Purpose

This is the shared definition note for compact Polymarket table columns, bucket labels, filters, and indicators. Use it to keep findings notes readable without repeating the same glossary everywhere.

Rule of thumb: a markdown note may keep CSV-native labels in a table, but the note must either explain them locally or link here. If a table has a shorthand term that is not in this dictionary, add it here or define it in the note before the table.

## Generic Result Columns

| term | plain-English meaning |
|---|---|
| `bucket` | A named subset of rows, markets, or episodes. The exact rule should be defined locally or in the bucket sections below. |
| `filter` | Entry-selection rule applied before computing the result. Example: keep only fills where a Polymarket token is rich versus Binance digital fair value. |
| `gate` | A pre-declared pass/fail test. Example: lower 95% CI must be above zero on the official OOS sample. |
| `markets` | Count of selected market episodes or markets, depending on the note. If an embargo is active, this usually means non-overlapping selected episodes. |
| `fills` | Count of individual K-PEG / maker / taker fills inside the selected markets or episodes. |
| `rows` / `n` | Count of table rows, events, fills, or observations. The note should say which unit is being counted. |
| `mean net` / `net` | Average net PnL after the costs included by that experiment. In OD notes this is usually dollars per market episode, displayed as cents. |
| `CI` / `net CI` | Bootstrap confidence interval, usually 95%. Positive mean with a negative lower CI is not a clean pass. |
| `CI lift` | Improvement in the lower confidence-bound versus the baseline row. A `+33c` lift means the downside bound rose by 33 cents. |
| `win` | Share of selected markets/episodes/trades with positive net PnL. |
| `PnL std` | Standard deviation of PnL across the unit being analyzed. Larger values mean more dispersion/fatter tails. |
| `bps` | Basis points. In maker/taker notes this is usually PnL or return scaled by notional. |
| `c` / `cents` | Dollar PnL shown in cents. `118.08c` means `$1.1808`, not 118%. |
| `top_mkt_share` | Share of the result contributed by the largest market. High values mean concentration risk. |
| `hold min` / `median hold min` | Median minutes from fill or episode entry to resolution/exit. |

## Embargo And Evidence Terms

| term | plain-English meaning |
|---|---|
| `global embargo` | After selecting one market episode, drop every other episode whose active time overlaps it, even if it is a different asset. This tests one shared capital/risk slot. |
| `per_asset` | Diagnostic where each asset gets its own embargo path, so BTC/ETH/SOL can all contribute overlapping 4h episodes. More power, but not the official one-slot gate unless explicitly reopened. |
| `OOS` / `oos_holdout` | Out-of-sample holdout used for the decision gate. |
| `IS` / `is_discovery` | In-sample discovery context. Useful for diagnosis, not the final gate. |
| `market episode` | One selected market's full inventory path from accepted fills through resolution. Multiple fills can live inside one episode. |
| `power` | Amount of independent evidence. More fills help only if they become more independent markets/episodes after embargo. |

## OD Strategy A Bucket Labels

These labels appear in [[od_strategy_a_v2_lifecycle_findings]], [[od_strategy_a_v3_findings]], and related OD notes.

Moneyness uses:

```text
abs_z = abs(ln(S / K) / (sigma * sqrt(tau)))
```

`S` is Binance spot, `K` is the 4h window-open strike/reference, `sigma` is causal volatility, and `tau` is time left. In plain English, `abs_z` says how far the current price is from the strike after scaling by volatility and time.

| label | plain-English meaning |
|---|---|
| `near_absz_lt0.25` | Very close to the strike. A small spot move can flip the UP/DOWN outcome. |
| `mid_absz_0.25_1` | Moderately away from the strike. Outcome is not pinned, but delta is meaningful. |
| `far_absz_ge1` | At least one volatility-adjusted unit from the strike. This is the far/longshot/pinned family. |
| `far_absz_ge1_all_tau` | Far from strike, pooled across all time-left buckets. This is the main OD Strategy A gate bucket. |
| `longshot_absz_ge0.75_all_tau` | Widened longshot family: includes `far_absz_ge1` plus somewhat less extreme longshots. |
| `longshot_absz_ge0.50_all_tau` | Even wider longshot family. More fills/episodes, but more dilution risk. |
| `early_gt2h` | More than two hours left to resolution. |
| `mid_30m_2h` | Thirty minutes to two hours left to resolution. |
| `late_lt30m` | Less than thirty minutes left to resolution. |
| `far_absz_ge1\|late_lt30m` | Intersection bucket: far from strike and less than thirty minutes left. |

Practical example: if BTC is far above the 4h strike with 20 minutes left, the DOWN token is a longshot. That fill would land in `far_absz_ge1|late_lt30m` and also count toward `far_absz_ge1_all_tau`.

## OD Strategy A Filters And Hedges

| label | plain-English meaning |
|---|---|
| `bare_lifecycle` | K-PEG maker lifecycle with no OD valuation filter. Accept eligible maker fills, carry inventory to resolution. |
| `official_strict_source` | Source-quality filter promoted from diagnostic to candidate design ingredient: settlement/source basis must pass strict Chainlink/Binance checks. |
| `rich_short_ge_010m` | Keep short/sell fills where Polymarket price is at least 1 cent rich versus Binance digital fair. |
| `strict_rich_short_ge_010m` | Same rich-short filter, plus strict source-quality gating. |
| `value_edge_ge_010m` | Keep long or short fills where signed model edge is at least 1 cent, regardless of direction. |
| `vol_premium_ge_10vp` | Polymarket implied vol is at least 10 annualized vol points above causal realized/EWMA vol. |
| `005m`, `010m`, `020m`, `050m` | Dollar thresholds in milli-dollar notation: 0.5c, 1c, 2c, and 5c. |
| `00vp`, `05vp`, `10vp`, `20vp` | Annualized volatility-point thresholds. |
| `cap_...` | Same filter with an added dollar-delta inventory cap. |
| `h` | Hedge ratio. `1.00` means full model hedge; `0.50` means half-size hedge. |
| `B` | Rebalance band in dollars/notional. `static` means set once and do not rebalance until settlement. |
| `episode_static` | Hedge separately inside each market episode. |
| `portfolio_24h_roll` | Net hedge changes across same-day/rolling windows instead of paying separate close/open costs every 4h. |
| `static_fraction` | Hedge policy using a fixed fraction of model delta. |
| `z_dependent`, `vol_dependent`, `iv_rv_spread_dependent` | Hedge policies that scale hedge size by moneyness, volatility, or implied-vs-realized vol spread. |
| `prem retained` | Hedged mean net divided by unhedged mean net. Values near 1 mean most premium survives the hedge. |
| `var reduced` | `1 - hedged variance / unhedged variance`. Negative values mean the hedge increased variance. |
| `turnover` / `rebal` | Hedge trading activity. More turnover usually means more cost drag. |

## OD Fair Source And Sizing Terms

| term | plain-English meaning |
|---|---|
| `rv_physical_prob_up` | Binance realized-volatility physical probability that the 4h window resolves UP, usually `N(z)` from causal EWMA sigma. This is not option-implied volatility and not Polymarket midpoint fair. |
| `token_rv_physical_prob_fair` | Token-side version of `rv_physical_prob_up`: UP token uses `rv_physical_prob_up`; DOWN token uses `1 - rv_physical_prob_up`. |
| `pm_mid_implied_vol_annualized` | Diagnostic annualized sigma obtained by inverting the Polymarket midpoint through the digital model. It represents the PM price, so it must not be treated as an external option fair. |
| `pm_mid_iv_minus_ewma` | PM-mid implied-vol diagnostic minus causal EWMA realized vol. Useful for describing PM richness, but still circular if used as fair. |
| `fair_prob_kind` | Source label for a fair-probability column. In OD Strategy A rows this should be `rv_physical_prob` unless a replacement-fair sensitivity explicitly names another fair source. |
| `rv_edge_scaled` | Strategy A sizing policy that scales size by edge versus RV physical-probability fair, capped at 3x. It is not an IV-edge sizing rule. |
| `replacement_edge_scaled` | Replacement-fair sensitivity sizing policy that scales size by edge versus the selected fair source, capped at 3x. |

## OFI, TOB, Decile, And A-Block Terms

| term | plain-English meaning |
|---|---|
| `OFI` | Order-flow imbalance: signed pressure from changes in bids/asks. |
| `MLOFI` | Multi-level OFI using depth beyond the best bid/ask. |
| `TOB` | Top of book: best bid/ask prices and sizes. |
| `tob_imbalance_level` | Standing-book imbalance `(best_bid_size - best_ask_size) / (best_bid_size + best_ask_size)`, sign-normalized where needed. |
| `TFI` | Trade-flow imbalance from historical aggressor fills. |
| `decile` | Equal-count bucket after sorting a signal by magnitude. Decile 10 is strongest magnitude, not necessarily most bullish. |
| `top_decile` | Largest absolute signal bucket. |
| `bottom_decile` | Smallest or opposite-tail bucket, depending on the note; check local design text. |
| `spread_bucket` | Quantile bucket by spread width. Usually narrower spread means cheaper execution. |
| `relative_depth_bucket` | Quantile bucket by available depth relative to typical market depth. |
| `time_to_resolution_bucket` | Bucket by time left before market resolution. |
| `dir ret` | Directional return in the signal's expected direction. |
| `hit` / `hit rate` | Share of observations where the predicted direction was correct. |
| `W`, `H`, `timeout` | Window, holding horizon, or timeout parameters. The local note should define the exact timing. |

## Maker, K-PEG, And Copytrade Terms

| term | plain-English meaning |
|---|---|
| `K-PEG` | Passive maker-entry lifecycle that pegs near a fair/anchor price and carries accepted inventory through a defined exit or resolution rule. |
| `maker USD` | Maker notional supplied by a wallet or group. |
| `top3 saturation bucket` | Markets grouped by how much maker USD is controlled by the top three makers. |
| `below-top3 PnL` | PnL for non-top-3 makers after excluding the dominant makers. |
| `adverse_30` | Adverse selection measured over a 30-second horizon. |
| `V0`, `V2` | Local strategy variants. The note using them must define the exact mechanics before the table. |
| `resolution_bucket` | Copytrade bucket by time to market resolution, often `2d`, `7d`, `30d`, or `60d`. |
| `invisible_share` | Share of a leader's activity that is not directly visible/copyable through the target execution surface. |

## Link Discipline

When writing a new Polymarket result note:

- Link this file near the top if the note contains markdown tables.
- Link specific prior notes with wikilinks, not plain text names.
- Define any table column that is not listed here.
- If a CSV emits a new compact label, either translate it in the markdown or add it to this dictionary in the same pass.
