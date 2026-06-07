---
title: "`0xd38b71f3` — strategy characterisation"
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
# `0xd38b71f3` — strategy characterisation
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


**Address:** `0xd38b71f3e8ed1af71983e5c309eac3dfa9b35029`
**Window:** 2025-08-23 → 2026-04-04 (~7.5 months)
**Headline:** $5.68M lifetime PnL, 556 markets, 1,103 closed positions, 9,582 fills.

This investigation was prompted by the directionality-classification metric
([build_traders_directionality.py](polymarket/research/scripts/build_traders_directionality.py))
flagging this trader as `arb_like` (vol-weighted balanced/offsetting share
= 58.6%, net/gross = 0.010) — overriding the Phase 4 prior of "directional
taker who picks off mispriced limits."

## TL;DR

**He is a synthetic directional bettor on sports event markets, using
CTF split-and-sell execution to construct long exposure on his chosen
side at a better effective price than direct purchase.** He is not an
arbitrageur and not a spread-capturer. The "balanced offsetting" signature
that fooled the directionality metric is an *artifact* of split-derived
positions: the on-chain split call (USDC → YES + NO basket) does not emit
an `OrderFilled` event, so our books record the subsequent sell of the
unwanted leg as a "sold without buying" position. In reality he is +X
on the side he wants and 0 on the other side, with $1·(1-p_sold) effective
cost basis on the kept side — a pure directional position.

**Fill-mirror copying this profile is structurally broken.** Splits are
invisible, so a follower bot has no signal that one of the trader's "sell"
fills was preceded by a synthetic split — it would just see a naked sell
of tokens it doesn't own. See "Copyability" at the bottom.

## 1. Per-market position structure — buy-one-sell-other dominates

547 of 556 markets are 2-outcome. Across the 1,103 closed positions, every
position falls into one of four buckets:

| bucket | n positions | sum bought ($) | sum sold ($) | sum PnL ($) |
|---|---:|---:|---:|---:|
| `bought_only` (long via direct buy) | 554 | 14,197,963 | 0 | 3,383,537 |
| `sold_no_buy` (short via split, no prior buy fill) | 545 | 0 | 14,374,282 | 2,088,634 |
| `bought_and_sold` (round-trip) | 2 | 90,416 | 14,571 | 106,136 |
| `net_short_position` (sold more than bought via fills) | 2 | 20,479 | 120,425 | 99,946 |

Pairing these by market: **545 of 556 markets (98%)** show the exact pattern
`bought_only` on one outcome + `sold_no_buy` on the other outcome. The
remaining ~2% are split/repositioning variants of the same.

This is the unambiguous signature of **split-and-sell directional construction**:
1. Off-chain or in the same transaction, split $X USDC → X YES + X NO.
2. Buy additional YES via `OrderFilled` (the side he wants).
3. Sell the X NO leg via `OrderFilled` at whatever bid is resting.
4. Hold the combined long-YES position to resolution.

Our books record (-X NO from the sell, no offsetting buy) and (+X YES from
the optional direct buy), giving the *appearance* of offsetting positions
even though the real exposure is +2X YES once you account for the split.

### Per-market fill concentration and net-to-gross (from the metric output)

| pct | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| fill_concentration | 0.51 | 0.52 | 0.56 | 0.66 | 0.79 |
| net_to_gross | 0.00 | 0.00 | 0.00 | 0.11 | 0.52 |

p50 net_to_gross = 0.002 means his recorded YES and NO positions are
perfectly offsetting in the median market — which is mechanically forced
by the buy-one/sell-other pattern.

## 2. Fill timing — both legs hit within the same block

Time between his first fill on one outcome and his first fill on the
opposite outcome, across all 547 two-outcome markets:

| pct | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| seconds between first-buy and first-sell on opposing outcomes | 0 | 0 | 0 | 0 | 0 |

**99.8% of markets have both opposing-side fills within 1 minute.** Most
are within the same block (sec_diff = 0). Spot-checking shows the two
fills typically share the same `transaction_hash` — i.e., one atomic
transaction emits both an `OrderFilled` on YES (sell) and one on NO (buy).
That is consistent with a `multicall` or bespoke router that bundles
`splitPosition` + `fillOrder` + `fillOrder` in a single tx.

Example (market 944360, "Spread: Ravens -3.5"):

```
2025-12-22 01:19:14  tx 0xe694d5de72  outcome=2  maker  +400,000 NO  @ $0.48
2025-12-22 01:19:14  tx 0xe694d5de72  outcome=1  taker  -400,000 YES @ $0.52
```

Same timestamp, same tx hash, equal-and-opposite token deltas: one
operation, two `OrderFilled` events, plus an invisible `splitPosition`
that provided the 400k YES being sold.

## 3. Maker / taker mix by entry vs exit phase

Fills classified by whether they grew (`entry`) or shrank (`exit`) the
position's absolute size at that outcome:

| leg_phase | role | n_fills | volume_usd |
|---|---|---:|---:|
| entry | maker | 2,109 | $13,153,570 |
| entry | taker | 7,462 | $15,650,906 |
| exit | maker | 8 | $7,641 |
| exit | taker | 3 | $6,017 |

**99.9% of fills are entry; only 11 fills total are exit.** He almost never
closes positions through trading — he holds to resolution. On entry: ~22%
of fills are maker (passive) and ~78% are taker (aggressive), consistent
with the published `style_role_balance = 0.22`. He uses both — maker on the
side he wants to *acquire cheaply* (post a bid and wait), taker on the side
he wants to *sell off after the split* (lift the resting bid).

## 4. Per-market PnL composition

Realised PnL decomposed into spread (round-trip) vs directional
(buy-and-hold to resolution) components, summed across all (market, outcome)
pairs:

| component | $ |
|---|---:|
| spread/round-trip PnL (matched buy-and-sell within a position) | $1,880 |
| directional PnL (price-move on net residual + resolution payout) | $5,676,381 |
| total realised PnL | $5,678,261 |

**$1.88k of $5.68M (0.03%) comes from spread capture.** Everything else is
directional outcome risk on the net positions he holds to resolution.

### Win/loss shape across markets

| | value |
|---|---:|
| total markets | 556 |
| winning markets (PnL > 0) | 277 (49.8%) |
| losing markets (PnL < 0) | 279 (50.2%) |
| avg PnL per winner | +$60,984 |
| avg PnL per loser | -$40,194 |
| avg-win / avg-loss ratio | 1.52 |
| per-market PnL p10 / p50 / p90 | -$62k / -$424 / +$88k |

He's right about which side wins ~50% of the time. His edge is not
prediction accuracy — it's **execution price**: when he's right, his
synthetic long was constructed at an effective cost basis well below the
nominal mid, so the payoff is larger than the comparable loss when he's
wrong. The 1.52 win/loss ratio is a structural feature of constructing
positions at p_y_market + p_n_market > $1 (i.e., when standing bids on
both sides sum to > $1, you can split USDC into a YES+NO pair and sell
the unwanted leg for more than its no-arb price, getting the kept leg
at a discount).

## 5. Round-trip economics

We have only **4 (market, outcome) pairs with meaningful round-trip volume**
(matched_tokens > 100, both sides bought and sold via `OrderFilled` —
$35k total round-trip volume, vs $28.8M total volume). For those four:

| pct | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| spread_capture (cents/$ on round-trip) | -3.3 | -0.8 | +4.3 | +9.3 | +11.7 |

Tiny sample; not a real spread-capture business. Round-trips are
incidental, not the strategy.

## 6. Market category breakdown

| family | n markets | total PnL | share of PnL | share of volume |
|---|---:|---:|---:|---:|
| sports | 545 | $5,478,226 | 96.5% | 98.0% |
| other | 11 | $200,035 | 3.5% | 2.0% |

**Pure sports trader.** Sports event markets are exactly where his
strategy works best:
- Lots of binary markets (one game = one moneyline + several
  spreads/totals) with thin midprice spreads (1-3 cents) but
  cross-side bids that frequently sum to > $1 during the rapid
  pre-game / in-game order flow.
- Fast resolution: hours to days, so capital recycles quickly.
- High liquidity at game time (where his strategy needs both-sided bids).

## 7. Comparison to known HFT operators (Cluster C deny-list)

Sub-second fill clustering — the share of his fills where the previous
fill (by him, anywhere in the data) occurred ≤ 1 second earlier:

| address | total fills | pct_sub_second |
|---|---:|---:|
| **`0xd38b71f3` (this trader)** | **9,582** | **80.0%** |
| `0xe8dd7741…` (HFT Cluster C) | 6.8M | 92.0% |
| `0xe3726a1b…` (HFT Cluster C) | 5.5M | 91.0% |
| `0x63d43bbb…` (HFT Cluster C) | 3.2M | 88.9% |

He is **operationally bot-driven** (80% sub-second), but a **distinct
operational profile from the HFT cluster**:

- Total volume: 9.6k fills vs HFT 3-7M fills — **2-3 orders of magnitude
  less throughput**. He's not running a continuous quoting bot.
- The HFT cluster has phantom ≈ 1.0 AND ~95-99% sub-second AND maker:taker
  ratio close to 1.0 — they're matched-arb flow. This trader has
  phantom = 1.00 (matches) and 80% sub-second (close) BUT maker:taker = 0.28
  (heavily taker, unlike HFT). He's an **event-triggered execution bot**,
  not a continuous matching engine: he fires when a sports market's
  cross-side bid stack supports the split-and-sell trade, then waits.

## 8. Synthesis

Best characterisation: **synthetic directional bettor on sports events,
using `splitPosition` + atomic dual `fillOrder` calls to construct
long-side exposure at effective prices below what direct purchase would
require.**

- **Not (a) market maker.** Maker fills are only 22% of his volume and are
  almost exclusively on the *opposite* side of his synthetic long (the
  leg he wants to dump after the split). True market makers quote both
  sides of a market continuously and round-trip through trading;
  he round-trips on 0 of 556 markets in any meaningful sense.
- **Not (c) cross-outcome arb.** $1.88k of $5.68M PnL (0.03%) comes from
  spread/round-trip capture. The rest is directional resolution PnL with
  win/loss split 50/50 and a 1.52 win/loss size ratio. An arb book would
  not have 50% losing markets at $40k average loss each.
- **(b) two-sided directional with aggressive hedging? No.** A two-sided
  directional trader (Domah-style) typically buys both sides at different
  times — his p50 time between opposing fills was non-zero hours.
  This trader's p50 is 0 seconds. His positions look offsetting only
  because the split is invisible.
- **Closest match: (d) something else — synthetic-construction directional
  bettor.** The strategy is one-sided exposure built via the cheapest
  execution route at each moment: when bid_YES + bid_NO > $1, split and
  sell the NO leg; when only one side has a fillable price, just buy.

## Copyability

**Fill-mirror copying is structurally broken for this profile** because:

1. **The split call is invisible.** A follower bot indexing `OrderFilled`
   events sees only "sold 400k YES at $0.52" and "bought 400k NO at $0.48".
   It does *not* see the `splitPosition` call in the same transaction that
   produced the 400k YES the trader sold. Without the split, the follower
   has no YES tokens to sell — mirroring his sell alone is impossible.
2. **The alpha lives in the basket, not the legs.** The "sell YES" leg by
   itself is a small loss in expectation (the YES will probably resolve to
   $0.5 ± something and he sold below market). The "buy NO" leg by itself
   is also marginal. Only the combination — split + sell-one-leg + hold-other
   — produces the trade's edge.
3. **Atomic execution against fleeting bids.** Both his fills are in the
   same transaction. By the time a follower sees the trade on-chain (one
   block later), the cross-side bids that made the trade profitable are
   gone — taken by him. A follower cannot replay the bids that no longer
   exist.
4. **The "net position" reading in our books is wrong.** A naive copy bot
   that tries to mirror net token deltas per market would see "-470k YES,
   +470k NO" and conclude "this trader has zero net exposure here" — when
   in fact he is +940k NO. Even net-position mirroring fails.

### Could *any* execution strategy replicate this profitably?

- **Direct re-implementation of the same strategy** (build your own
  split-and-sell bot, scan for bid_YES + bid_NO > $1 moments on sports
  markets, fire same atomic tx structure). This is mechanically possible
  but it is *competing with the trader*, not copying him. The bids he hits
  are the same bids you would hit, so there is real adverse selection
  between his bot and yours. Whoever has lower latency wins each
  opportunity; "copy" gives no information advantage.
- **Delayed net-position mirroring with split awareness.** A follower
  could compute "trader's true net exposure on market M after each tx"
  by detecting his split calls (i.e., index `CTF.splitPosition` events
  too, not just `OrderFilled`), then go long the same side via direct
  purchase. This *would* replicate his directional positions, but at
  worse prices because the follower buys after the trader's fills clear
  the bids. The follower also pays normal market spread instead of
  the trader's discount.
- **Selection mimicry, not execution mimicry.** Treat his trades as a
  signal for *which sports market and side* are mispriced, and execute via
  whatever route is locally cheapest. This degrades to a noisy "follow
  the smart money" play that depends on the trader's directional skill
  surviving across moments — and the data shows his win rate is 50/50,
  so any directional edge is small. The 1.52 win/loss ratio that drives
  his $5.68M PnL is a function of his *execution edge*, not his outcome
  prediction.

**Conclusion: this trader is not copyable via any fill-mirroring scheme.**
The only avenue with positive expectation would be to re-implement his
strategy independently, in which case he is a competitor, not a leader to
follow. This profile should be filed under "what's not copyable" in the
broader cohort design.

## Appendix — artifact index

All in `polymarket/research/data/analysis/leader_dthreed8b71_investigation/`:

| file | what's in it |
|---|---|
| `per_market.parquet` | one row per market: fill_concentration, net_to_gross, fills, PnL |
| `per_outcome.parquet` | one row per (market, outcome): buys, sells, residual, holding |
| `fills_labelled.parquet` | every fill with cum_tokens + entry/exit phase label |
| `per_market_first_opposite_timing.parquet` | sec_between first fills on opposing outcomes |
| `pnl_decomposition_per_outcome.parquet` | spread vs directional PnL components |
| `round_trip_economics.parquet` | matched buy-and-sell economics where applicable |
| `maker_taker_by_leg.parquet` | n_fills × volume cross-tab of role × entry/exit |
| `sub_second_clustering.parquet` | global pct_sub_second for the trader |
| `hft_operator_subsec.parquet` | same metric for the three HFT Cluster C operators |
| `family_breakdown.parquet` | per-family PnL and volume |
| `summary.json` | all the numbers used in this document |
| `scripts/build_investigation.py` | what built all the above |
