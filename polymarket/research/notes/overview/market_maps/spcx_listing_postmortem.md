---
title: "SpaceX (SPCX) IPO listing-day trade — realized outcome: long the $135 allocation, sold into the day-1 pop, +~€300 net, no hedge needed"
tags: [spacex, spcx, ipo, polymarket, hyperliquid, postmortem, realized, findings]
created: 2026-06-12
status: "CLOSED — trade executed and fully unwound 2026-06-12; position flat; net ~€300 profit realized"
---

# SPCX Listing-Day Trade — Realized Outcome (Postmortem)

> Hub: [[spacex_ipo_market_map_handoff]] · [[COWORK]] · [[POLYMARKET_BRAIN]]
> Companions: [[spcx_listing_day_gameplan]] (the pre-registered plan this is graded against) · [[spcx_convergence_calc_findings]] (the basis/hedge math) · [[spcx_pm_pdf_monitor_findings]] (the live dashboard, §10.x) · [[spcx_ipo_unwind_tape_findings]] (S2 unwind tape + S6 perp exit) · [[spcx_pm_arb_findings]] (S8 PM arb)

## Plain-English Summary

- **What happened.** SpaceX listed on 2026-06-12. The operator was allocated **11 shares at the $135 offer** (€116/share on Trade Republic, implied FX ~1.164), held a **pure long, unhedged**, and sold the whole position into day-1 strength: **6 shares at ~€140 ($163, a +21% take-profit) and the remaining 5 at $172**. The position is **flat**; realized profit is **~€300 net** (gross ≈ €303, net of TR's embedded FX spread + fees).
- **Why no hedge.** 11 shares (≈ €1,276 / $1,485) sat *inside* the pre-registered ~22-share comfort zone, so the gameplan's overflow-valve rule said **no perp hedge** — exactly what was done. The Hyperliquid short was never opened.
- **The thesis was right.** The whole SPCX arc said the same thing from three independent instruments: the stock would list well above $135. It did — it traded $163–172 in the operator's selling window vs the $135 entry (+21% to +27% in USD).
- **Status.** Trade closed, tooling shut down, poll log + static dashboard preserved for the record. This note is the postmortem the [[spcx_listing_day_gameplan]] § Post-day and [[spcx_pm_pdf_monitor_findings]] §6 anticipated. **All SPCX work is now complete and closed.**

---

## 1. The trade as executed

| leg | shares | price (USD) | price (EUR, TR) | vs €116 entry |
|---|---:|---:|---:|---:|
| allocation (buy) | 11 | $135.00 (offer) | €116.0 | — |
| sell tranche 1 | 6 | ~$163 | €140.0 | **+€24/sh** (+21%) |
| sell tranche 2 | 5 | $172 | ≈ €148.7 | **+€32.7/sh** (+28%) |
| **net** | **0 (flat)** | — | — | **≈ +€303 gross → ~€300 net** |

**Column meaning.** "EUR, TR" = the price as it appeared in the operator's Trade Republic account, which trades in EUR and embeds the USD↔EUR conversion in the spread (no itemised FX line — see [[spcx_trade_republic_execution_findings]]). The $135 offer booked as €116, implying an effective FX of ~1.164. "vs €116 entry" is the per-share gain in EUR.

**P&L arithmetic.** Tranche 1: 6 × (€140 − €116) = **€144**. Tranche 2: $172 ≈ €148.7 at the entry-implied FX, so 5 × (€148.7 − €116) = **€163.5**. Gross ≈ **€307**; the operator's stated **~€300 net** is gross minus TR's FX spread and any per-order fees — consistent, not overstated.

**A note on the second tranche's timing.** The 5 shares at $172 sold *above* where both the perp and the PM crowd closed the session (perp $168.4, PM-implied mean $168.7, P(close>$135) 98.6% on the dashboard's last poll). Selling into a local high rather than the close was good execution, not luck the model can claim — but it is consistent with the S2 finding that the cross/early-session captures ~89% of the day-1 high and that one should front-load into strength rather than hold for the close ([[spcx_ipo_unwind_tape_findings]]).

## 2. Graded against the pre-registered gameplan

The value of pre-registering ([[spcx_listing_day_gameplan]]) is that the result can be scored honestly against the rules written *before* the outcome was known. Every rule that fired, fired as written:

- **No pre-hedge (S1).** The rule was "do not pre-hedge; hedge only the overflow above the ~22-share comfort zone at Friday ~8:00 iff net basis > 0, and only pre-hedge early if the perp spiked ≥ ~$183." The fill (11 sh) was below comfort and the perp never spiked to the trigger → **no hedge** was the pre-registered call, and it was correct: hedging a position this size would have capped the +21–28% upside for no ruin-risk reduction that mattered at €1,276 notional. The leverage-is-the-ruin-mode caveat never bound because no leverage was used.
- **Long the allocation, sell into the pop (S2).** The unwind study said the day-1 high clusters early and to front-load selling into strength rather than carry to the close (5/6 mega-IPOs closed lower day-2). The operator sold the whole position intraday into $163–172 and **carried nothing overnight** — exactly the S2 prescription.
- **Offer priced $135, ceiling $162 void (D1).** The $135 offer was confirmed the night before; the "$162 upward-revision" contingency never triggered; basis held as computed.
- **PM/perp were directionally right, modestly rich intraday.** All day the crowd mean sat ~$168–180 and the perp ~$168–177 while P(close>$135) ran 91–99%. The stock traded $163–172 when the operator sold — i.e. the leveraged/crowd venues were a touch rich versus the intraday print, but the *direction* (lists well above $135) was unambiguous and correct. The convergence thesis paid.

**What did NOT get tested (honest ledger).** Because no hedge was opened, none of the perp-side live unknowns were exercised this trade: real passive/maker fill rate, the pair-close mechanics (S6 +46/+61-min gates), funding carry across conversion, or liquidation-buffer behaviour. The S8 PM executable-arb verdict (NO-ARB; the ladder↔bucket gaps are model/spread artifact, not cash) also stands untouched — no PM leg was traded. Those remain measured-on-analog-tape, not on this trade.

## 3. What the dashboard did, and what it was worth

The listing-day dashboard ([[spcx_pm_pdf_monitor_findings]]) ran through the day as the read-only advisory surface: EXECUTION tab for day-state + playbook, PM tab for the crowd distribution / divergence / executable-arb (all confirming NO PM edge), the S7 day-shape classifier and tail-sell screen for the Polymarket leg. It was never an order engine and placed no orders — execution was manual on Trade Republic, exactly as designed.

Its honest contribution to *this* trade was **confirmation and timing context**, not signal generation: it showed the crowd/perp holding well above $135 all day (supporting "hold the long, it'll list up"), and the tail-sell/PEAK screen + day-shape banner gave a structured read on selling into strength. The trade would have worked without it; the dashboard made the "sell into the pop, don't carry overnight" discipline legible in real time. That is the appropriate, modest claim — consistent with the thread's standing posture that SPCX merited *a minimal instrumented live test, not a trading-system build*.

## 4. Decision / next step

- **Trade: CLOSED, profitable (~€300 net), flat.** The SPCX convergence/allocation thesis is **realized and confirmed** on a real listing.
- **Tooling: shut down.** The listing-day monitor process is stopped; `playbook_state.json` persists the final closed state (fill 11 / sold 11 / hedged 0); the append-only parquet poll log (705 shards on listing day) and the static dashboard HTML are preserved for review — relaunch read-only (`--serve --backfill-days 7`) anytime to scrub the session.
- **Thread: complete.** All SPCX research blocks (S1–S8) and the listing-day dashboard stack (S5b–S5k + the day-of additions) are done; the live-only gates are now resolved by the trade itself. Nothing further is queued. Lessons that generalise — *don't hedge inside the comfort zone, front-load selling into day-1 strength, the prediction-market crowd is a directional confirmer not an intraday-precise oracle* — are recorded here for the next IPO-convergence setup.
