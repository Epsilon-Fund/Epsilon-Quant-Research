---
title: "Market-Making on Polymarket — Concepts, the Build-Up of Our Strat, and Where We Are"
created: 2026-06-15
status: active
owner: justin
project: polymarket
para: resource
audience: "anyone picking this up cold — assumes no quant-finance background, then goes deep"
hubs:
  - strat_market_making
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - market-making
  - concepts
  - synthesis
  - build-up
  - negrisk
  - plain-english
---

# Market-Making on Polymarket — Concepts, the Build-Up of Our Strat, and Where We Are

> Hubs: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · arc glossary [[block_k_plain_english_synthesis]]
> Table terms: [[polymarket_table_dictionary]] · methodology: [[CODEX]] § Realism calibration

## Plain-English Summary

- **What this note is.** One place that teaches market-making (MM) from first principles, then shows the **additive build** of the bot we are actually developing — each layer, what it adds, and the evidence that earned its place — then states **where we are** and **what the brain audit says to build next**. (A one-line glossary of what K1–K7 each tested is in §2.0; how K5 identified its "makers" is in §2.0b; the one-sided-fill realism bottleneck is §1.9.)
- **The one idea to hold onto.** MM is "earn the spread." On Polymarket the textbook one-liner quietly assumes you can always round-trip out passively; you usually can't. So our whole strat is a stack of devices — *slow-market selection, two-sided quoting, carry-to-resolution, spike-avoidance, non-incumbent cell selection, and the NegRisk redemption floor* — that exist to let you **keep** the spread instead of giving it back at the exit or losing it to adverse selection.
- **Where we are.** Building our *own* single-venue maker is **robustly closed** (tested ~10 ways). Real maker wallets *are* profitable (K5), but winner-take-most (top-3 take ~95%/market). The one live monetization candidate is **politics NegRisk neutral MM** (+2,290 bps historically) — run as a **measurement loop**, not yet switched on. We are currently only **capturing data** (observation-only), and that capture has lapsed since Jun 12.
- **Audit verdict (folded in below, Part 4).** The MM-bot thesis lives in our method's **novelty blind spot** — historical screening can't validate an edge nobody runs yet, so "no historical signal" is not a "no." That makes it a legitimate but *unproven* bet, and the single experiment that adjudicates it is **already designed**: the politics NegRisk Phase-2 loop. The cheapest high-value introspective wins are **never-run gates that need no new data** (the dali passive-fade framings) plus analyzing the **first-mover capture we already collected**.

---

# Part 1 — Market making from first principles (the full toolkit)

## 1.1 What market-making is, and the one equation

A prediction market sells shares that pay **$1 if an event happens, $0 if not**. A share at $0.66 means the market thinks ~66% chance — on a binary market, **the price *is* a probability**.

Two kinds of participants:

- **Takers** cross the spread to trade *now*. They pay a fee and accept a worse price for immediacy.
- **Makers** post resting orders ("I'll buy at $0.64, I'll sell at $0.68") and *wait* to be filled. They get the better price and, on Polymarket, a small **rebate**.

Our entire program is: **be the maker, not the taker.** The appeal is structural — Polymarket *pays* makers and *charges* takers, so in principle a maker is profitable before having any view on direction. Memorise this:

```
maker profit per trade = spread captured + rebate − adverse selection − inventory/resolution risk
```

The first two terms are what you earn; the last two are what kill you. **Adverse selection is the killer** (§1.4). Everything in Parts 2–3 is a fight over the right-hand side.

## 1.2 The order book — bid, ask, touch, tick, mid, micro-price, queue

- **Bid / ask (the "touch"):** highest price anyone will currently buy at (bid), lowest anyone will sell at (ask). The gap is the **spread**.
- **Tick:** smallest price step. On Polymarket it's **1 cent ($0.01)**.
- **Mid:** `(bid + ask) / 2`.
- **Micro-price:** a smarter fair value that leans toward the side with *less* resting size (that side is likelier to get hit next): `(ask·bid_size + bid·ask_size)/(bid_size + ask_size)`.
- **Queue (this is decisive and under-appreciated).** Quoting *at* the best bid/ask does **not** guarantee a fill. Orders at a price fill in time priority (FIFO). If you're behind a big resting order, you only get filled when the whole queue ahead of you does — which is disproportionately *when the price is about to move against everyone at that price*. So **queue position is itself a source of adverse selection**: the informed trader takes the front; the passive tail (you) gets the toxic remainder.

**Worked example.** Book is bid $0.64 / ask $0.66. You join the bid at $0.64 behind 5,000 shares. A seller dumps 5,200 shares: the 5,000 ahead of you fill first, you fill the last 200 — right as the mid slips to $0.63. You "won" a fill at $0.64 and are already down a cent. That's queue-driven adverse selection, and no amount of "I quoted at the touch" saves you from it.

## 1.3 Fees and rebates (Polymarket specifics — exact, and they drive category choice)

- **Maker fee = 0.** Makers are never charged.
- **Taker fee = `C · feeRate · p · (1−p)`.** Biggest at 50¢, zero near 0¢/100¢. `feeRate` is category-specific: **Crypto 0.07, Sports 0.03, Geopolitics 0** (no taker fee at all).
- **Maker rebate** is funded *out of* collected taker fees: **20% (Crypto), 25% (other fee-enabled), 0% for Geopolitics** (no taker fee collected → nothing to rebate).

**Why this matters:** "earn the rebate" is a real cushion in crypto, a smaller one elsewhere, and a **myth in geopolitics**. This is why fee-free geopolitics keeps showing up as *negative* in our maker tests — there's no rebate cushion to offset adverse selection.

## 1.4 Adverse selection — the killer, and how we measure it

- **Adverse selection:** the bad case where whoever fills your resting quote knows something (or is faster). You bid $0.64; someone sells to you at $0.64 *because* it's about to drop to $0.60. You "won" the trade and immediately lost. A maker **profits from uninformed flow and loses to informed flow** (Glosten-Milgrom, 1985) — the single most important idea in the whole program.
- **Toxic fill:** a fill that was adversely selected.
- **Markout:** how the mid moved in the seconds *after* your fill — how we *measure* adverse selection. A "60-second markout" of −336 bps means: 60s after your fills, the mid had moved 336 bps against you on average.

**The empirical backbone (K5-STRESS, deterministic full-population 60s markouts).** This single table is *why* the strat is built the way it is — adverse selection is wildly category-dependent:

| category             | 60s markout (adverse cost) | maker-favorable fills |
| -------------------- | -------------------------: | --------------------: |
| tech                 |                    −87 bps |                 39.3% |
| economics            |                   −203 bps |                 33.8% |
| geopolitics          |                   −230 bps |                 43.0% |
| sports               |                   −232 bps |                 40.6% |
| culture              |                   −312 bps |                 40.4% |
| **politics_negrisk** |               **−336 bps** |                 48.3% |
| other                |                   −403 bps |                 41.2% |
| finance              |                   −778 bps |                 45.8% |
| daily_crypto         |                 −1,051 bps |                 50.3% |
| weather              |                 −1,387 bps |                 35.3% |
| **crypto_4h**        |             **−1,886 bps** |                 39.7% |

**Read.** Fast crypto (4h up/down) is a slaughterhouse: −1,886 bps of adverse selection per fill. Politics NegRisk is **~5.6× milder** (−336 bps) and has the **highest maker-favorable rate** (48.3%). The slow, judgment-resolved markets (politics, sports, tech) are where a passive maker is least picked off — because there is no fast external venue (a Binance) leading the price and no second-by-second information arriving. **This table is the empirical case for "build the bot in slow markets, not crypto."** Source: [[block_k5_stress_findings]]. *(Caveat: this is the **full population** of maker fills; the **skilled** K5 cohort's realized markouts are maker-favorable — see §2.0b — so how toxic our own fills are depends on our fill selection, itself a live unknown per §1.9.)*

## 1.5 Inventory, the binary jump, and hold-to-resolution

- **Inventory risk:** while you hold a position, the price can move against you before you offload it.
- **Resolution / settlement:** when the market closes, every share snaps to exactly **$1 or $0**. Holding *into* resolution is a coin-flip-like jump, not a smooth move — unique to binary markets, and high-variance.
- **Hold-to-resolution:** deliberately *not* trading out; let the position settle at $1/$0. This **avoids paying to exit** but takes the full binary jump (unless the inventory is structurally balanced — §1.8).

## 1.6 Mark-to-mid vs realizable PnL — the distinction the whole program turns on

- **Mark-to-mid:** "value my open position at the current mid."
- **Realizable:** "what I'd actually get if I closed it" — which means crossing the spread.

On a 15–30¢-spread venue these are wildly different. **A pretty mark-to-mid profit can be a real loss the moment you try to bank it.** Our headline cautionary tale is K-PEG: the *same fills* were a smooth, broad-based **+$4.60** marked-to-mid and a smooth, broad-based **−$8.27** when actually exited. The signal was real; the realizability was not — on one venue.

**This is the answer to "isn't MM just making the spread?"** Yes — *when the round trip completes passively*. The spread is not free money; it's **compensation for inventory risk and adverse selection**. "Make the spread each time" assumes a timely offsetting passive fill that thin, slow books don't reliably provide. When the offsetting fill doesn't come, you either (a) cross the spread yourself and pay the exit tax (turning the captured spread into a loss), or (b) hold inventory. Carry-to-resolution and the balanced basket (below) are not a *different* strategy — they are **the inventory discipline that lets you keep the spread you nominally captured.**

## 1.7 The market-making theory we tried — Avellaneda-Stoikov

- **Avellaneda-Stoikov (A-S):** the standard math for optimal maker quoting. Its key feature: it needs **no view on direction** — it sets how *wide* to quote and how to *skew*, based purely on **inventory risk** and **volatility**. That's exactly "classic inventory-accumulation market making": quote both sides, and when you accumulate inventory, skew your quotes to shed it.
- **Inventory skew:** if you're long too much, shift quotes down (keener to sell, less keen to buy). The skew comes from inventory, *not* a price forecast.
- **Reservation price:** the inventory-adjusted center you quote around.
- **Logit-space adaptation:** plain A-S assumes prices wander freely; Polymarket prices are trapped in [0,1] and jump at resolution. The 2025 fix does the A-S math in log-odds space (compresses spreads near 0/1). **We tested exactly this (K2) — see Part 2.**

## 1.8 The options lens and NegRisk mechanics (two structural facts we exploit)

**A 4-hour crypto market *is* a digital option** (pays $1 if BTC is up at expiry). That gives us:
- **Delta** (sensitivity to the underlying) is a tall **spike right at the strike**, ~0 far away. **Gamma** (how fast delta changes) **blows up near the strike near expiry** (∝ 1/√τ). This "delta/gamma spike zone" — **near 50¢, late in the window** — is the danger area where hedging is brutally expensive and prices whip. Avoiding it is a core rule (§2, Layer 4).
- This lens is the seed of the **Options-Delta (OD)** sibling strat ([[strat_options_delta]]) — but see §2 Layer 7: OD is *not necessary* for the bot we're building.

**NegRisk (the structural gift of politics markets).** A NegRisk *event* is a set of **mutually-exclusive YES outcomes** where exactly one resolves to $1. Therefore the YES prices should sum to ~$1, and:
- A **complete YES set across all legs is redeemable for exactly $1**, regardless of which leg wins. You can **SPLIT** $1 into a full set, **MERGE** a full set back to $1, **REDEEM** after resolution, or **CONVERT** between forms.
- **Consequence for a maker:** a *balanced* inventory across the legs is **structurally near-neutral** — it has a built-in $1 floor. This is the device that lets you carry politics inventory to resolution *without* a directional bet and *without* an external hedge. Tracking these basket events (which don't show up as simple per-token fills) is exactly what `maker/negrisk_inventory.py` is for.

## 1.9 The realism bottleneck — "you'll get one-sided fills, so you must carry" is an assumption, not a measurement

A central claim above — that you often can't round-trip out passively, so you carry — is **partly an assumption, and it is the single biggest realism bottleneck.** Be precise about measured vs assumed:

- **What the sims assumed.** K2 and K-PEG are *single-leg* sims: enter one side, then **exit by crossing the spread**. That structurally bakes in an exit cost a *continuous two-sided* maker never pays — so their big losses partly measure a deliberately bad lifecycle, not the economics of real making. K5 makes exactly this point.
- **The one historical hint, and its limits.** The closest evidence that passive offset is hard is K-PEG *maker-exit*: only **~12% of exits filled passively** (you'd need ~40%). But that is crypto-4h, single-leg, on the *exit* leg — **not** a continuous two-sided book in a slow market. Suggestive, not decisive.
- **Skill inverts adverse selection.** The §1.4 table is the *full population* of maker fills (crypto-4h −1,886 bps). The **skilled K5 cohort's realized markouts are maker-favorable** (crypto-4h **+616 bps at 60s**; see §2.0b). Good makers don't passively eat toxic flow — they **select fills and skew inventory** to avoid it. Inventory control (the A-S skew) is precisely the tool for one-sided flow: when you accumulate, you quote to shed.
- **So carry is the residual, not the plan.** The right mental model: quote two-sided, *try* to offset passively, use inventory skew to manage the imbalance, and **carry only the residual you couldn't offset.** How large that residual is — your true passive two-sided fill rate, queue position, and missed-fill rate — is **unknown until we quote live.** That is exactly what the Phase-2 telemetry measures (`fill_share_this_market`, `top_maker_rank_at_fill`, missed fills).

**Implication for the bot decision:** the quantity that decides whether a politics neutral-MM bot works (the passive two-sided fill rate net of adverse selection) is *live-measurable and not yet measured*. That is an argument **for** running the small live loop — not for assuming the answer either way.

---

# Part 2 — The build-up: what each layer adds, and the evidence for it

This is the heart of the note: the current strat is **classic inventory MM plus a stack of additions**, each of which earned its place by a specific experiment. Read top to bottom as "what we add and why."

## 2.0 Block glossary — what each K-block tested (one line each)

The strat grew through a numbered sequence of experiments ("Block K"). So nothing is assumed, here is each one in a line — what it tested and what it found. Full arc: [[block_k_plain_english_synthesis]].

| block | what it tested | one-line result |
|---|---|---|
| **K1** ([[block_k1_maker_economics_findings]]) | baseline maker economics by category — is spread+rebate > adverse selection at all? | "passes" only by marking to mid (fee-free geopolitics also passes) → a permission slip, not an edge |
| **K2** ([[block_k2_quoting_findings]]) | classic optimized Avellaneda-Stoikov **inventory** maker (logit space) with a real taker exit | **−1,126 bps** — classic inventory MM is dead in crypto |
| **K2v2** ([[block_k2v2_findings]]) | add a **defensive** rule: pull/widen quotes when Binance moves against you | **−4,316 bps**; defense fired <0.1% → adverse selection is **structural, not a dodgeable latency race** |
| **K2v3** ([[block_k2v3_findings]]) | **anchor** quotes to Binance digital fair value instead of PM mid | 0/681 buckets clear; the anchor *raised* adverse selection (325 vs 145 bps) |
| **K-PEG** ([[block_kpeg_findings]]) | a heavily-optimized "chase" maker that follows the quote | +759 bps headline — but a **mark-to-mid artifact** |
| **K-PEG robustness / maker-exit** ([[block_kpeg_robustness_findings]]) | is K-PEG real? what if you exit passively instead of crossing? | no bug/lookahead, but a real exit flips it to **−753 bps**; passive exit fills only ~12% (need ~40%) |
| **K3** ([[block_k3_leadlag_findings]]) | is 4h crypto arb-able vs Binance? (lead-lag + basis) | no anti-arb fee, Binance leads ~10s, but post-fee basis thin/IS-only; K3v2: **no latency budget**; K3v3h hedged basis negative |
| **K4** ([[block_k4_arb_scan_findings]]) | risk-free intra-Polymarket arbitrage on our owned universe | essentially none capturable (1 interval, 0 at latency) |
| **K5** ([[block_k5_findings]]) | **model-free realized PnL of real maker wallets** (not a sim) | makers ARE profitable (crypto-4h **+171 bps**); hands us the winners' playbook; top-3 take ~95% |
| **K5b** ([[block_k5b_findings]]) | *why* top-3 makers dominate — speed vs capital vs structure | it's **capital/scale + structure, not speed** → build in Python, defer Rust |
| **K5-STRESS** ([[block_k5_stress_findings]]) | re-run K5 with survivorship / structure / non-top3 / rebate / NegRisk filters | 4 categories pass strict; the *typical* maker ≈ breakeven, only **structured non-top3** clears |
| **K6 (vol)** ([[block_k6_vol_findings]]) | is PM volatility overpriced, and can you gamma-scalp it? | vol IS overpriced (+24 pts far/late) but continuous-hedge **turnover** kills the scalp |
| **K6 Strategy A (static hedge)** ([[block_k6_strategy_a_static_hedge_findings]]) | replace continuous hedging with a one-time **static** hedge | fixes turnover (mean −9.39c → +1.07c) but the far/late gate fails on power (n=11) |
| **K7** ([[block_k7_findings]]) | cross-category **longshot-premium** harvest (are unlikely outcomes overpriced enough to trade?) | no category clears all gates — capacity-captured or CI-crossing |

(The sibling **OD** = "options-delta" blocks live in [[strat_options_delta]] — the digital-option pricing/vol/hedge lineage, not needed for the politics bot; see Layer 7.)

## 2.0b How K5 classified "winners" (so the playbook in Layers 2–4 isn't circular)

A fair worry: if we *picked* profitable wallets and then read off their behavior, "two-sided + carry" would be circular. K5 avoids that by defining the cohort **structurally as makers first, then measuring their realized PnL** ([[block_k5_findings]] § Method):

- **Who counts as a maker:** corrected **maker share ≥ 70%** with **≥ 1,000 passive fills** — a wallet that predominantly *posts* resting orders rather than crossing. Defined on behavior, not profit. **9,121** wallets qualified; the analysis cohort was **588** (top overall + top crypto-4h + top per-category makers).
- **Attribution care:** recomputed from raw `data/trades` with exchange-internal relayer legs (`EXCHANGE_INTERNAL_LEG_V1/_V2`) filtered on both maker and taker slots, so we count the **real wallet**, not exchange plumbing.
- **PnL is model-free:** realized from `data/closed_positions.parquet` (positions settle to actual $1/$0), rebate/fee-adjusted, with **market-block bootstrap CIs** so one huge market isn't counted as many.
- **Which markets:** measured across **all categories**, not just crypto. Net bps by category: politics_negrisk **+443** [116, 883] (pre-NegRisk-accounting), tech +184 (CI crosses 0), crypto_4h **+171** [34, 327] (clears), other +147 [81, 207], daily_crypto +83, sports +74 (crosses 0), **geopolitics −47** (fee-free, doesn't clear). The 64% two-sided / 78.8% carry / 0.8% spike behavior stats are **crypto-4h-specific**.
- **Honest caveats:** `closed_positions` is **closed-only**, so still-open *losing* inventory is excluded (biases PnL up); and the cohort skews toward **large** makers, so it describes skilled/capitalized behavior, not the median wallet (K5-STRESS then shows the *typical* maker ≈ breakeven).

**Read.** "Winners" really means "structurally-defined makers whose realized PnL we then measured" — so the playbook (two-sided, carry, spike-avoid) describes *how profitable makers actually behave*, not a curve-fit. But it is weighted to skilled/large wallets and closed positions, which is exactly why Layer 5 (capacity) and the live loop matter.

## Layer 0 — Classic single-venue A-S inventory maker → **robustly dead in crypto**

This is the textbook bot: quote both sides, skew on inventory, no directional view. We built it and optimized it. It died, four ways, all on **structural adverse selection**:

| test | what it added | result |
|---|---|---:|
| **K1** ([[block_k1_maker_economics_findings]]) | baseline maker economics by category | "passes" only by **marking to mid**; fee-free geopolitics also "passes" → not a real edge, just a permission slip |
| **K2** ([[block_k2_quoting_findings]]) | logit A-S, optimized, **real taker exit** | **−1,126 bps**, CI [−1,499, −748] |
| **K2v3** ([[block_k2v3_findings]]) | anchor quotes to Binance digital fair value | **0/681 buckets clear**; the anchor *raised* adverse selection (325 vs 145 bps) |
| **K2v2** ([[block_k2v2_findings]]) | defensive pull/widen when Binance moves | **−4,316 bps**; defense fired **<0.1%** of fills |

**The two lessons everything converged on:**
1. **Mark-to-mid is not money** (§1.6). The problem is always the **exit**, never the entry.
2. **Adverse selection here is structural, not a latency race.** K2v2's defensive trigger almost never fired — the toxic fills are *not* preceded by a visible Binance move. You can't out-react it; you can only **avoid being there**.

**Status: ROBUST-CLOSED. Do not rebuild a single-venue crypto neutral maker.** (Reopen-filter: died on large-magnitude structural adverse selection — stays shut.)

> **So why keep going?** Because both lessons point the same way: change *where* you quote and *how you exit*, not how you compute the quote. That's Layers 1–6.

## Layer 1 — **+ Slow-market selection** (quote where adverse selection is mild)

The single biggest lever. The §1.4 table shows adverse selection ranges from −1,886 bps (crypto_4h) to −230/−336 bps (geopolitics/politics). **Move the same passive quoting from fast crypto to slow, judgment-resolved markets and you cut the killer term ~5–6×.** This is *why the bot is politics/sports-first, not crypto-first*. Evidence: [[block_k5_stress_findings]].

## Layer 2 — **+ Two-sided continuous book** (offset passively, never cross to exit)

A real professional maker offsets a buy with a *later passive sell on the other side* — never crossing the spread to flatten. Our single-leg sims structurally baked in the exit tax a two-sided maker never pays. **K5's profitable makers are 64% two-sided** (crypto-4h). This is what makes "making the spread" actually realizable (§1.6) — *but how often you actually get the passive offset is the live unknown of §1.9.* Evidence: [[block_k5_findings]].

## Layer 3 — **+ Carry-to-resolution** (when the offsetting fill won't come, hold to $1/$0)

For the **residual** inventory you couldn't offset passively (how much is itself a live unknown — §1.9), the disciplined move is **not** to pay the exit tax — it's to **hold to resolution**. **K5 makers carry 78.8% of the time** (the complement is their rough intraday round-trip share). Cost: you take the binary jump on that inventory — which Layer 6 neutralizes structurally in NegRisk. Evidence: [[block_k5_findings]]; the carry-vs-exit arithmetic is the K-PEG +$4.60→−$8.27 result ([[block_kpeg_robustness_findings]]).

## Layer 4 — **+ Spike-zone avoidance** (don't quote near 50¢, late in the window)

The delta/gamma spike zone (§1.8) is where prices whip and adverse selection peaks. **K5's profitable makers put only 0.8% of activity there.** A simple, hard rule: don't be the passive maker at the coin-flip near expiry. Evidence: [[block_k5_findings]].

## Layer 5 — **+ Non-incumbent cell selection** (only quote where you can actually get fills)

**The capacity reality.** Top-3 wallets capture **~95%** of positive crypto-4h maker profit per market — the edge is **winner-take-most**. K5b shows the moat is **capital/scale + structure, not speed** (below-top3 field +29.4 bps, CI [−173, 260] crosses zero) → **build in Python/Midas, defer Rust**. So we only target **structured non-top3** cells with genuine headroom.

K5-STRESS formalizes the gate (18,724-wallet population; structured = two-sided ≥60%, carry ≥50%, spike ≤2%; then exclude each market's top-3). **4 categories pass strict:** crypto_4h, culture, other, sports. The deployable-cells pass then sizes it honestly: **8 sub-cells, but only ~$78/active day median** EV (≈$2.3k/30d), **~90% concentrated in one grab-bag cell (`other:misc_other`)**, crypto cells ≈ $0/day after capacity. Evidence: [[block_k5b_findings]], [[block_k5_stress_findings]], [[mm_deployable_cells_findings]].

**Read.** This is the sobering layer: even done right, *standalone* historical MM is small and capacity-bound. It is the reason the repo's standing view is "MM is an execution layer, not a standalone money-printer" — and the reason the politics path (Layer 6), with its far larger flow, matters.

## Layer 6 — **+ NegRisk redemption floor** (structural neutrality + the biggest flow)

Politics NegRisk is the standout cell once you account for it properly. After decoding **125,937/125,937** merge/split/redeem receipts, the corrected-carry structured-non-top3 edge is **+2,290 bps, CI [1,020, 3,621]** (median wallet 14.5 bps; +2,276 ex-rebate; settled-only holds outside 2024: 2025 +1,402, 2026 +1,356). Flow is huge and persistent: **$381.1M** 2026 non-top3 flow, active **146/146 days**. This is where the redemption floor (§1.8) lets you carry a balanced book to resolution without a hedge. Evidence: [[mm_politics_negrisk_accounting_findings]].

**The catch — capacity, again.** At 0.25%/1%/5% capture: **mean** EV/day ≈ $1.5k/$6.0k/$29.9k, but **median-wallet** EV/day is only **$9/$38/$189**. The big numbers are a right-tail scenario, and the whole +2,290 bps is a *historical* number for the wallets who did it — not yet proof *we* can get the fills (see Part 3/4).

## Layer 7 (optional) — **+ OD valuation overlay** — *not necessary for this bot*

OD is a **crypto digital-option** valuation layer (vol, basis, N(d₂)). Standalone OD is **closed** ([[od_strategy_a_realism_reaudit_findings]], [[od_methodology_realism_audit_findings]]); it survives only as a tiny **sizing/selection overlay** ("lean away from the side that looks rich"). For a **politics** NegRisk bot it is **irrelevant** — there is no Binance vol surface for "will X be confirmed." **Verdict: drop OD from the critical path of the politics bot; revisit only if you return to crypto 4h.**

## Layer (rejected) — Hedging — *only a crypto concept*

The static Binance hedge (K6 Strategy A) answers the binary jump on a *single crypto digital*. K6 showed *continuous* hedging is killed by turnover; the *static* far-family is a powerless near-miss ([[block_k6_strategy_a_static_hedge_findings]]). **In politics there is nothing to hedge externally** — you neutralize structurally via the balanced NegRisk basket (Layer 6). **Verdict: the politics bot does not hedge.**

## The build-up at a glance

| layer | what it adds | evidence | status for the bot |
|---|---|---|---|
| 0 | classic A-S inventory maker | K1/K2/K2v2/K2v3 | **dead in crypto (robust)** — the thing NOT to rebuild |
| 1 | slow-market selection | K5-STRESS markout table | **core** — cuts adverse selection ~5–6× |
| 2 | two-sided book | K5 (64%) | **core** — makes the spread realizable |
| 3 | carry-to-resolution | K5 (78.8%) | **core** — avoids the exit tax |
| 4 | spike-zone avoidance | K5 (0.8%) | **core** — a hard rule |
| 5 | non-incumbent cell selection | K5b, K5-STRESS, deployable cells | **core** — but caps standalone size (~$78/day) |
| 6 | NegRisk redemption floor | politics accounting (+2,290 bps) | **the live candidate** — structural neutrality + biggest flow |
| 7 | OD valuation overlay | OD re-audit | **optional / skip for politics** |
| — | external hedge | K6 static hedge | **crypto-only; skip for politics** |

---

# Part 3 — Where we are, and how we got here

**The arc.** "Block K" asked: can we make money providing liquidity on Polymarket instead of betting direction? Single-venue quoting is **robustly closed** (Layer 0). K5 proved real makers *do* profit, and handed us the winners' playbook (Layers 2–4). K5b/K5-STRESS/deployable-cells showed the profit is **capacity-bound and small standalone**. The consolidation view in the repo: **MM is the execution/lifecycle layer**; its highest-value live expression is the **politics NegRisk** cell.

**The methodological limit we hit (important).** Our method mines *historical wallet behavior*. It is excellent at **falsification** and at **finding copyable winners**, but **structurally blind to novelty**: an edge nobody exploits yet leaves *no* historical signal, so the method reads "no edge" — a false negative on exactly the opportunities worth most. See [[2026-06-04_state_of_the_arc_and_novelty_frontier]]. **This is the key frame for the bot decision in Part 4.**

**What's actually running / built right now:**
- **Live capture (observation-only, no orders):** 3 lanes — first-mover, broad diagnostics, slow crypto+finance (`scripts/mm_stage1_live_control.py`). ~64 GB over Jun 4–13. **Lapsed since Jun 12** (nothing capturing now). This feeds the **first-mover novelty branch + diagnostics**, *not* the politics loop.
- **Maker execution infra (built, measurement-grade, 256 tests green):** `polymarket/execution/maker/` — `maker_engine.py`, `negrisk_inventory.py`, `resolution_handler.py`, `event_calendar.py`, NegRisk-aware signing. Missing for production telemetry: true queue position, side-adjusted drift bps, volume-weighted fill share, book depth at quote/fill, quote age, cancel latency. See [[mm_maker_infra_audit_findings]].
- **Politics NegRisk Phase-2 loop:** designed, pre-registered, **not deployed** (no live orders anywhere yet). See [[mm_politics_negrisk_live_loop_design]].

**What's robustly closed (do not reopen):** single-venue crypto neutral MM (K2 family), K-PEG chase (mark-to-mid artifact), NegRisk **basket-consistency arb** (real but a ~4-second latency game, net-negative after fees on ~98% of episodes — [[mm_negrisk_consistency_scanner_findings]]), continuous-hedge gamma scalp, and PM terminal pricing (efficient net-of-cost across crypto and equities).

---

# Part 4 — Brain audit: what introspection says to build next

This builds on the Jun 4–5 novelty pass ([[2026-06-04_state_of_the_arc_and_novelty_frontier]], [[2026-06-05_novelty_frontier_map]]) rather than redoing it, and applies the [[CODEX]] § Realism-calibration **reopen filter** (don't reheat branches that died on robust grounds; prefer never-run cheap gates).

## 4.1 The decisive introspective insight

**The MM-bot thesis lives in our method's novelty blind spot.** The directional-decomposition finding says the *historical* structured-maker edge is **directional-carried, not neutral** — the clean neutral subset is empty/negative in sports, equities, and politics ([[mm_structural_maker_directional_decomposition_findings]]). Naively that reads "neutral MM has no edge." But neutral MM was only **directly tested (and killed) in crypto** (Layer 0); in **politics/sports it was only screened, never run**. So the absence of a neutral-maker signal there is a **false-negative candidate**, not a falsification. **Conclusion: building our own neutral-MM bot in slow markets is a legitimate *novelty bet* — but it has zero offline confirmation by construction, and can only be validated by a live loop.** Price it as speculative, size it tiny, and let the live loop adjudicate.

## 4.2 What changed since the Jun 4–5 map (introspection, not new ideas)

- The map's **#1 PM deterministic edge — NegRisk basket consistency — has since been CLOSED** (Jun 10 scanner). Remove it from the live menu.
- The maker infra is now **built and tested** — "going live is closer than the TODO wording implies."
- The capture has been **running but lapsed** — the first-mover lane has ~9 days of data nobody has analyzed yet.

→ The PM frontier has **narrowed** to: the live measurement loops + the never-run cheap gates. That's the whole actionable set.

## 4.3 Ranked moves that survive the reopen filter (cheapest / highest-introspection-value first)

1. **Run the politics NegRisk Phase-2 loop — the experiment that adjudicates the whole bot thesis.** It is already designed and pre-registered (≥30 settled markets or 90 days, ~$30 at 1 contract; gates: fill share >0% in ≥5 markets, post-fill 60s drift lower-CI > −500 bps, news-proximate adverse fills <50%, net-of-cost lower-CI >0, resolution drag <10%). This is the *direct* live test of classic neutral inventory MM in a slow market (Layers 1–6). **Nothing else should scale before this reads out.**
2. **Cheaply resolve a live contradiction the brain has with itself about the dali reversion signal.** The 73.7% TOB reversion was real, and the *continuation* framing was falsified. Here the brain disagrees with itself, which is exactly the kind of thing an introspective pass should catch: the high-level [[2026-06-04_state_of_the_arc_and_novelty_frontier]] says dali is "dead across all framings" and the passive route "fills ~0.1%," **but** `block_a1x_external_note_reconciliation.md` flags specific framings that were **scoped and never run** — (a) **continuous rolling-rank position sizing** (a genuinely different *exposure*, not a fill-rate question), (b) **ask-depletion/bid-support fade side**, (c) **fill-probability-gated passive / the A14e queue+latency model**. These need **no new data**. Be honest about EV: the measured ~0.1% passive fill rate suggests the maker-capture variants will be flow-starved like K-PEG's exit, so this is *not* a presumed edge. But it costs a few hours of compute and either closes the branch cleanly or surfaces the one framing (rolling-rank sizing) that isn't just a fill-rate question. Treat it as introspection, not a new bet.
3. **Analyze the first-mover capture we already collected.** ~9 days across 3 lanes is sitting unanalyzed. Measure live book shape, spread/depth, trade arrivals, and reconstruct adverse selection (per [[mm_clob_capture_semantics]] — public L2 is anonymous, so report clean-vs-ambiguous reconstruction rates). This is the forward-test the first-mover scope note asked for ([[mm_first_mover_liquidity_scope_findings]]) — and it costs only compute. **First: restart the lapsed capture so the series doesn't keep gapping.**
4. **Add the missing maker telemetry before scaling** (queue position, side-adjusted drift, fill share, book depth at quote/fill, quote age, cancel latency) — without these you can't tell whether a Phase-2 failure is capacity, speed, or bad quoting ([[mm_maker_infra_audit_findings]]).

## 4.4 What NOT to do (stay-closed per the reopen filter)

Single-venue crypto neutral MM (structural adverse selection, directly tested); K-PEG-style peg-chasing (mark-to-mid tax); NegRisk basket-consistency arb (latency game, net-negative after fees); continuous/banded gamma-scalp (turnover); any PM *terminal pricing/forecasting* strategy (efficient net-of-cost, robustly closed across crypto and equities). Reopening these is motivated reasoning.

## 4.5 The honest tension to settle with Alvaro

The repo's *standing* recommendation is **OD-primary + copy the directional winners (Strategy B)**. Your stated direction is the opposite — **our own MM bot, not copytrade.** Both are defensible, and the divergence is real, so name it explicitly. The evidence that would *justify the pivot* is concrete and already specified: **the politics NegRisk Phase-2 loop passing its pre-registered gates.** Until that reads out, "build the bot" and "copy the winners" are competing hypotheses, and the cheap, decisive move is the same either way — **run the measurement loop.** Build the big bot *after* the 30-market measurement says the fills and adverse selection are what we hope, not before.

---

## Cross-links

Hub: [[strat_market_making]]. Arc + glossary: [[block_k_plain_english_synthesis]]. State map: [[2026-06-04_state_of_the_arc_and_novelty_frontier]] · [[2026-06-05_novelty_frontier_map]]. Live loop: [[mm_politics_negrisk_live_loop_design]] · [[mm_politics_negrisk_accounting_findings]]. Evidence: [[block_k5_findings]] · [[block_k5b_findings]] · [[block_k5_stress_findings]] · [[mm_deployable_cells_findings]] · [[mm_negrisk_consistency_scanner_findings]] · [[mm_first_mover_liquidity_scope_findings]] · [[mm_maker_infra_audit_findings]]. Capture semantics: [[mm_clob_capture_semantics]]. Sibling/optional: [[strat_options_delta]].
