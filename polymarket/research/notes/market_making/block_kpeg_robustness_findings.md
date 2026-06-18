---
title: Block K-PEG Robustness and Lookahead Audit
tags: [dali, block-kpeg, maker, robustness, lookahead, round-trip]
created: 2026-05-31
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
  - strat_market_making
status: cowork audit — pending independent Codex review (notes/market_making/block_kpeg_robustness_review.md)
relationship: Audits block_kpeg_findings.md (+759.6 bps pooled IS headline). Reconciles the K2-vs-K-PEG contradiction.
repro: scripts/dali_block_kpeg_robustness.py → data/analysis/{kpeg_robustness.csv, kpeg_robustness_phase.csv, kpeg_robustness_report.txt, kpeg_robustness_fills.parquet}
---

# Block K-PEG — Robustness / Lookahead Audit

> **Strat:** [[strat_market_making]] (Market-Making). Sibling: [[strat_options_delta]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Summary

This audit reconciles the positive K-PEG mark-to-mid headline with the negative K2 round-trip result. It finds the reproduction is faithful and not a lookahead bug, but the edge depends on marking to mid and not paying a feasible exit cost. The note is an audit artifact later reviewed in [[block_kpeg_robustness_review]].

## Headline

K-PEG's +759.6 bps pooled IS headline is **not lookahead and not an arithmetic bug — it reproduces exactly — but it is an artifact of marking the position to mid and never paying to exit.** Completing the round-trip as a taker exit (the same accounting K2 used) flips the result to **−742 bps pooled / −753 bps Crypto, CI [−1182, −364], win rate 26%.** K2 (which paid the exit) was right. The edge K-PEG sees is the project's known mean-reversion-to-micro-price captured by a deep passive bid — real as a descriptive pattern, untradeable once the exit cost is priced.

Frozen params audited: `peg_offset=0, chase_increment=2, chase_cap c=7, inventory_scaling=0.304, cadence=1s`.

## What is clean (credit where due)

- **Faithful reproduction.** The audit's V0 = **+759.4 pooled / +692.7 Crypto** matches `kpeg.simulate` and the note headline to the decimal (409 fills, 56 markets). The pipeline is faithful.
- **No lookahead.** Entry price uses only book state at-or-before the fill; `future_mid` enters strictly as a cost; the fill condition (`trade_price ≤ modeled_bid` / `≥ modeled_ask`) uses contemporaneous/past info only.
- **No calc bug.** Net reduces exactly to `token_side·(future_mid₃₀ − entry)/denom·1e4 + rebate − inv_charge₃₀`. The "realized spread vs adverse selection" split is cosmetic; net depends only on entry, future mid, rebate, and the synthetic inventory charge.
- The one place K-PEG is *conservative* — the synthetic `0.5·|Δmid|` inventory charge — actually drags V0 below the pure mark-to-mid ceiling (V1 +1155 → V0 +693 Crypto). So K-PEG is not winning on a generous risk model; it is winning purely on the missing exit.

## The decisive result — completing the round-trip

| version | pooled | Crypto | CI (Crypto) | win rate |
|---|---|---|---|---|
| V0 reproduce (mark-to-mid − synthetic risk charge) | +759.4 | +692.7 | [313.9, 1074.0] | 0.79 |
| V1 pure mark-to-mid ceiling (no risk charge, no exit) | +1215.9 | +1155.0 | [795.3, 1556.9] | 0.90 |
| **V2 round-trip (taker exit @60s, = K2 accounting)** | **−742.2** | **−753.2** | **[−1182.0, −363.8]** | **0.26** |

Mean **exit half-spread ≈ 1635 bps** — larger than the entire reversion captured. On books this wide, crossing the spread to exit costs more than the edge. This reconciles the K2 (negative, paid exit) vs K-PEG (positive, marked to mid) contradiction.

Descriptively, the edge is real: deep passive bid ~6.8 ticks below micro, fills on a transient dump, mid reverts ~5.5 ticks over 30s. That is the A15b/A1.7 mean-reversion-to-micro finding captured passively — and it dies on the exit.

## Crypto component means (per fill)

```
entry_price 0.663 | dist entry→micro 6.79 ticks | realized_spread 1306.9 bps
adverse_30 196.9 | inv_charge_30 462.3 | rebate 45.0 | exit_half_spread 1635.0 | exit_fee 213.3 bps
```

## Spread structure over the 4h window (corrects the "wide-at-open, narrows" intuition)

Unconditional crypto-4h book spread by time-since-window-open (cadence-1 quote states):

| phase | 0–15m | 15–30m | 30–60m | 60–120m | 120–240m |
|---|---|---|---|---|---|
| mean spread (bps) | 985 | 1131 | 985 | 1124 | **1965** |

Spread is roughly flat early and **widens into resolution**, not the reverse — consistent with digital delta/gamma spiking near the strike and liquidity thinning as the binary resolves. Brief enormous spreads do occur in the first seconds (one early fill saw ~12,400 bps) but are transient, not the average. **~92% of K-PEG's fills (341/370 phase-tagged) land in the 120–240m late window — exactly where spread is widest**, which is what makes mark-to-mid look flat and the round-trip worst. The policy is static across the window; it has no time-to-resolution awareness.

## Entry-price / gamma-zone buckets (Crypto)

| bucket | n | adverse_30 | V0 | V2 round-trip |
|---|---|---|---|---|
| near_50c (\|p−0.5\|<0.10) | 51 | 277 | +384 | −184 (least bad, thin) |
| mid (0.10–0.25) | 66 | 620 | +156 | −1848 (worst) |
| near_0/1 (0.25–0.50) | 286 | 85 | +872 | −602 |

Adverse selection is lowest near 0/1 (longshot stability), highest in the mid band. Every bucket is negative round-trip.

## Caveats on this audit (to be checked by Codex)

- V2 uses the **entry-time** half-spread as the exit cost (proxy). Spreads here are persistently wide, but Codex should recompute exit cost from the actual book at t+60s and confirm V2 is not artificially pessimistic.
- The placebo (shuffle `future_mid` within category) **inflated** rather than nulled, because entry price is correlated with each market's price level — a confounded null, not evidence of a bug. A clean null (per-market circular block-shift of `future_mid`) is the right replacement.
- OOS split is uninformative here: discovery (a0/a0b) has only 22 fills; holdout (a0c/a0c_roll) has 387. Holdout "passes" V0 (+791) and fails V2 (−790) — but the round-trip kills it regardless of split, so OOS is not load-bearing for the verdict.
- Fill model is generous (full priority, one-share, fills on any print through a deep quote). The −753 is therefore optimistic on fills.

## Implications / optimizations to discuss

1. **K-PEG optimized the wrong objective (mark-to-mid).** Any maker search must optimize net-of-round-trip-cost, or Optuna will keep finding deep-chase corners.
2. **The only escape from the exit-spread tax is to not cross it:** exit as a maker (reintroduces the A14h 0.2% fill-rate problem) or **hold to resolution and hedge direction externally on Binance (Track A).** K3 confirmed the 4h market has no anti-arb fee and Binance leads ~10s, so the hedge leg is feasible. This is the real argument for Track A — delta-hedging converts "pay ~1600 bps to exit" into "collect the binary settlement, neutralize direction."
3. **Time-to-resolution-aware quoting:** since spread widens and fills cluster late, quote only when the round-trip (maker-exit or hedged) is positive for that phase.

## Verdict

K-PEG is a clean, faithful sim with the **wrong PnL objective**. It is not evidence the maker thesis is alive; it is the mean-reversion finding re-expressed as a maker and flattered by a mark-to-mid exit. Trust K2: maker dead on this universe via a single Polymarket leg. The live question remains Track A (maker fill + external hedge / hold-to-resolution), which sidesteps the exit-spread tax that kills K-PEG.
