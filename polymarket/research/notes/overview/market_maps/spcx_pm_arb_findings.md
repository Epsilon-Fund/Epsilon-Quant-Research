---
title: "SPCX Polymarket — ladder↔bucket pricing-mismatch arb (Block S8): real lock, uninvestable size"
created: 2026-06-12
status: complete
owner: justin
project: polymarket
para: project
hubs:
  - spacex_ipo_market_map_handoff
  - POLYMARKET_BRAIN
tags:
  - spacex
  - polymarket
  - arbitrage
  - market-structure
  - findings
---

# SpaceX-IPO Polymarket — can you arb the ladder against the buckets? (Block S8)

Hub links: [[spacex_ipo_market_map_handoff]] · [[POLYMARKET_BRAIN]] · [[spcx_convergence_calc_findings]] · [[spacex_pdf_construction_audit]] · [[spcx_pm_pdf_monitor_findings]] · [[TODO]]

## Plain-English Summary

- **The only arb that matters here:** buy/sell the **threshold-ladder** markets ("cap above $X") against buy/sell the **bucket** markets ("cap between $X–$Y"), taker-only, to harvest the pricing mismatch the monitor keeps flagging (e.g. [2–2.5T] bucket printing +8 to +15pp "rich" vs the fitted ladder). This note measures whether that mismatch is a *riskless executable lock* and how much size it absorbs.
- **Answer: it is a real lock, but pocket-change-sized.** The one number that matters — walk the book buying best-ask, then best-ask+1, … until the marginal set stops being net-positive — was **$5–8 extractable at zero fee** across the morning's polls, on a rich side that is a single retail-sized bid (7–266 shares depending on the minute), not a deployable book. At Polymarket's *declared* 1000 bps taker fee the lock **vanishes entirely** (fees eat the first set). Fee is the single swing factor; see the one-knob output below.
- **Why "+14.6pp" becomes "+3 cents":** the bucket edges 1.5/2.5/3.5T have **no ladder strike to trade against** (the ladder steps 2.4 → 2.6, skipping 2.5). So the eye-popping pp-gap is measured against an *interpolated* curve value S(2.5T), not a price. The executable trade has to cover at 2.6T and cross real spreads; once you do, the 14.6pp collapses to a few cents on a thin bid.
- **The executable trade (when it's live):** SELL the rich bucket + BUY the wider ladder range that contains it, e.g. *sell [2–2.5T] bucket, buy >2T ladder, sell >2.6T ladder* → pay ~$1.98, locked payout $2.00, the [2.5–2.6T) sliver rides free. Net is positive only under the optimistic fee model and only for tens of dollars of notional.
- **Verdict: RV-with-tiny-size, not arb.** Track it as a live indicator (the dashboard wiring below); do not staff it with capital. A clean negative was a pre-registered acceptable outcome — nothing here changes the [[spcx_listing_day_gameplan]] plan.

## The market structure you're trading between (live Gamma/CLOB, verified)

| group | event slug | what each leg is | legs | structure |
|---|---|---|---|---|
| **Ladder** | `spacex-ipo-closing-market-cap-above` | "cap **above** $X?" — YES pays if close-cap > X (strictly) | 16 strikes $1.0T–$4.0T, **$0.2T steps** | independent binaries, non-exclusive |
| **Buckets** | `spacex-ipo-closing-market-cap` | "cap **between** $X–$Y?" — exactly one resolves YES | 7 cap brackets + a "No-IPO" leg | **NegRisk** mutually-exclusive group (`negRiskMarketID 0x2fcc…cc300`) |

Three facts that make the arb math safe:

1. **Co-resolution — no basis risk.** All 24 legs resolve off the same event: official closing price × outstanding shares on SpaceX's **first trading day**, primary-exchange listing page, same "no IPO by Dec 31 2027" fallback. The checker parses a resolution fingerprint from each market's own description; **all 24 share one key**. Combining a ladder leg with a bucket leg is genuinely riskless, not a cross-market bet.
2. **Boundary convention.** Ladder is *strictly above* (>$2T is NO at exactly $2.000T); buckets are lower-edge-inclusive ("exactly between brackets → higher bracket"). A close landing *exactly* on an edge breaks ladder↔bucket equivalence by one leg — but that's measure-zero: with 13,075,865,175 shares, a 2-decimal price can't hit $2.000T exactly. Every affected combo carries this as an explicit caveat rather than pricing it as zero.
3. **NO book = mirror of YES book.** On every leg, ask(NO) = 1 − bid(YES) to ~1e-17 (the matching engine crosses complements via mint/merge). So "sell the YES" is executed as a taker buy of NO — every leg below is a real taker order, no minting or market-making required.

**The structural catch (this is the whole story):** the bucket edges **1.5T, 2.5T, 3.5T are not ladder strikes**. So a bucket like [2–2.5T] cannot be replicated *exactly* from the ladder — there is no >$2.5T market. You can only bound it with the nearest wider/narrower ladder range. That gap between "what the monitor compares" (interpolated S(2.5T)) and "what you can actually trade" (cover at 2.6T) is where most of the flagged pp-divergence lives.

## The arb, two executable constructions

`scripts/spcx_pm_arb_check.py` enumerates every contiguous bucket run × both directions against the tightest ladder range that contains it ("containment covers"), plus the exact unions whose edges *do* land on strikes ([<1T], [1,2T), [2,3T), [≥3T)). Payoff floors are brute-force enumerated over every terminal state (each bracket edge, midpoints, below-min, above-max, No-IPO) under the resolution semantics above — floors are derived, never asserted. Each candidate is depth-walked level-by-level and netted under two taker fee schedules.

**Construction A — sell a rich bucket (the [2–2.5T] case the monitor keeps flagging):**

> SELL [2–2.5T] bucket  (= buy its NO @ ask)
> BUY  >2T ladder       (buy YES @ ask)
> SELL >2.6T ladder     (= buy NO @ ask)

Pay ~$1.98, **guaranteed payout $2.00** in every state, **$3.00 if the close lands in the free [2.5–2.6T) sliver**. This is the executable expression of "[2–2.5T] is rich vs the ladder." Because the cover is at 2.6T (not the non-existent 2.5T), you also own the 2.5–2.6T slice for free — which is *why* it's a strict lock, not an approximation.

**Construction B — buy a cheap bucket (the [2.5–3T] / tail case):**

> BUY  [2.5T→∞) buckets  (buy YES @ ask on each)
> SELL >2.6T ladder      (= buy NO @ ask)

Pay <$1.00 for a guaranteed $1.00 payout. Mirror logic — when the buckets above 2.5T are collectively cheap vs the ladder.

A and B are the two sides of the same mismatch and **share the >2.6T ladder leg**, so their sizes are *not additive* — you can run one or the other through that 2.6T book, not both at full size.

## What it's actually worth — one fee, one number

The output the dashboard shows (and the script prints): **walk the book one set at a time — buy best-ask, then best-ask+1, … — accumulating net profit, and stop at the first set where the marginal lock goes net-negative.** That cutoff is the "no more mispricing" point. One taker fee knob, stated on the tab:

> fee/share = `(fee_bps / 10000) · min(price, 1−price)` — Polymarket's documented CLOB taker formula. `fee_bps = 0` = what fills are observed to pay; `fee_bps = 1000` = the rate these markets *declare* (`taker_base_fee = 1000`). CLOB taker fills are relayer-gasless (no per-trade gas).

The single number across the morning's polls (the best executable ladder↔bucket lock each minute):

| poll (UTC) | best executable lock | extractable @ 0 bps | extractable @ 1000 bps |
|---|---|---|---|
| 10:17 | sell [2–2.5T] vs ladder cover | ~$0.3 over 21 sets | $0 |
| 10:19 | same | ~$1 over 58 sets | $0 |
| 10:48 | sell [<1.. 2–2.5T] run vs [0,2.6T) | ~$7 over 266 sets | $0 |
| 11:13 | sell [1–1.5..2–2.5T] run vs [>1T,>2.6T) | **$5.55 over 169 sets** ($669 notional) | $0 |

Read: at the fee Polymarket actually charges on fills (0), the walk extracts **single-digit dollars** before the book reprices the mispricing away; at the *declared* 1000 bps, the very first set is already net-negative, so **nothing is extractable.** The fee knob is the entire story — and either way it is pocket change on a retail-sized bid, never a deployable book. (Example 11:13 lock, taker-only: `SELL 1–1.5T @ 0.963 · SELL 1.5–2T @ 0.800 · SELL 2–2.5T @ 0.410 · BUY >1T @ 0.993 · SELL >2.6T @ 0.790` → pay $3.956, locked $4.00, the cap=1T-exactly boundary the only sub-floor state.)

## Why the monitor's pp-gap ≠ the executable cents

Reproducing the monitor's PCHIP gap table on the 09:43 UTC shard (mid basis):

| bucket | market (mid) | PCHIP ladder | flagged gap |
|---|---|---|---|
| **2–2.5T** | 0.550 | 0.464 | **+8.6pp** |
| **2.5–3T** | 0.163 | 0.206 | **−4.3pp** |

S(2.5T) = 0.281 is *interpolated* between the traded S(2.4T)=0.385 and S(2.6T)=0.200 — there is no $2.5T market. Because that interpolated knot enters the two flagged buckets with opposite signs, any fit error shows up as "[2–2.5] rich AND [2.5–3] cheap" simultaneously with **zero real mispricing**. The decomposition of the 8.6pp flag:

- **~±6pp = PCHIP interpolation artifact** (the non-traded S(2.5T) knot) — model vs market, untradeable by construction.
- **~4–5pp = mid-vs-executable spread** (the [2.5–3T] book was 10.3c bid / 17.6c ask — 7.3c wide; the mid overstates the bid you'd actually hit).
- **~2–3c/set = the real, directly-executable residue** — Construction A above, on a thin bid.

Read the monitor's bucket-gap panel as a **fit diagnostic at half-T edges, never as tradable spread.** The number to watch is the executable cents, not the pp.

## Other classes checked (all clean — not the ladder↔bucket mismatch, recorded for completeness)

- **NegRisk basket** (buy-all-YES across the 8 bucket outcomes): ask-sum $1.119–1.127 vs $1 needed → NO-ARB. Mint-and-sell recovers only $0.959–0.961 of the $1 mint → NO-ARB. The 0.1c No-IPO ask is ~8–10k shares ≈ $8–10 of end-of-life lottery inside the group, not an arb leg.
- **Ladder monotonicity** (YES(>a) + NO(>b) < $1 for a<b): **0/120 inversions**, nearest −0.7c. The ladder is internally consistent.
- **Exact ladder↔bucket unions** (edges on real strikes): all ≤ +0.4c gross, net-negative under both fee schedules; the [3,∞) union tops out at +0.4c × ~52 sets ≈ $0.21 gross.

## Reproduce / track

- Capture + analyze: `cd polymarket/research && PYTHONPATH=. uv run python scripts/spcx_pm_arb_check.py` (one poll) or `--watch 60 --polls N`. The single extractable-$ line prints at the bottom; set the fee with `--fee-bps 1000` (default 0). Offline replay from cache: `--from-parquet latest`.
- The one-knob output is the public helper `best_executable_arb(meta, snap, fee_bps)` → `{exists, best:{legs:[{action,market,price}], pay_per_set, payout_floor, net_usd, net_sets, ...}, verdict}` — this is what the dashboard imports.
- Full L2 books cached append-only → `data/analysis/spcx_convergence/pm_arb_books/books_<ts>.parquet` (+ `meta_<ts>.json`); results → `pm_arb/{summary,depth,legs}_<ts>.csv`.
- Tests: `tests/test_spcx_pm_arb_check.py` (26 green) — executable-only (mid-poisoning + source scan), co-resolution guard, depth-aware sizing, single-fee net-walk + fee-shrinks-the-walk, honest-null.

## Verdict

| construction | verdict |
|---|---|
| Sell rich bucket / run vs wider ladder range | **Real lock, uninvestable** — extractable ~$0.3–8 at 0 bps fee on a 7–266 share bid; **$0 at the declared 1000 bps** |
| Buy cheap buckets vs ladder | **Real lock, uninvestable** — same order of magnitude; shares the 2.6T ladder leg, so not additive with the sell side |
| Exact unions / monotonicity / basket | **NO-ARB** — internally consistent within spread |

**One line: the ladder and buckets are consistent at executable prices to within a few cents; the monitor's big pp-divergences are the interpolated-S(2.5T) knot plus wide-bucket spread, and the genuinely executable residue walks out to single-digit dollars at zero fee and to nothing at the declared fee. Track it, don't trade it.**

## Decision / next step

- Wire this as a **constant live indicator into the PM tab** of the `--serve` dashboard (prompt drafted for the dashboard chat): given live depth and the stated fee schedule, show the current best executable ladder↔bucket lock as `BUY x / SELL y,z`, taker-only, with net profit — so the pp-panel is read alongside its real cash value and we stop chasing the points.
- The cached shards make any future episode (esp. around the IPO cross and the 22:00 CET close, when books most likely dislocate) replayable offline with one command.
