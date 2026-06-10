---
title: "SPCX Listing-Day Gameplan — Subscription → Pricing → Allocation → Unwind (Decision Tree + Research Blocks)"
tags: [spacex, spcx, ipo, hyperliquid, trade-republic, gameplan, execution, decision-tree, findings]
created: 2026-06-10
audience: "Justin (stock/perp side) + Cowork/Claude Code sessions building the Friday execution stack; companion to Alvaro's Polymarket-side playbook"
status: "live gameplan; all marks dated 2026-06-10 — refresh at every decision node; research blocks S1–S4 pending"
---

# SPCX Listing-Day Gameplan — Subscription → Pricing → Allocation → Unwind

> Hub: [[spacex_ipo_market_map_handoff]] · [[COWORK]] · [[POLYMARKET_BRAIN]]
> Companions: [[spcx_convergence_calc_findings]] (calculator + Cerebras tape) · [[spacex_ipo_coworker_addendum]] (TR/venue mechanics) · [[2026-06-09_spacex_ipo_convergence_trade]] (session snapshot) · Table terms: [[polymarket_table_dictionary]]
> Coworker input: `SpaceX_Execution_Playbook.docx` (Alvaro, 2026-06-08) — the Polymarket-side + directional-sell playbook this note reconciles with.

## Plain-English Summary

- **What this is.** The stock/perp-side gameplan for Justin's €10k Trade Republic SPCX subscription, from now (Wed Jun 10) through pricing night (Thu), allocation (Fri ~8:00), the Nasdaq IPO cross, and the day-1 unwind — structured as a decision tree with pre-registered rules, plus the research blocks (S1–S4) Claude Code must run before Friday to fill in the open parameters.
- **The reconciliation.** Two strategy frames coexist: the vault's **convergence lock** (long @offer + unlevered short HL perp, hold both to settlement — locked basis, direction-independent) and Alvaro's **sell-the-pop** (no perp, sell tranches into peak euphoria). They are not rivals — they apply to **different slices of the same allocation**. The binding constraint is Justin's free Hyperliquid margin (<€2k): at ≤1.5× leverage that hedges only ~15–22 shares. So the structure is a **two-sleeve plan**: hedge sleeve = min(fill, margin-supported shares) runs the convergence lock; residual sleeve = everything above that runs a hardened version of Alvaro's tranche plan.
- **What changed 2026-06-10 (web-sourced).** Book is now **~4× oversubscribed (~$250B vs $75B)** and institutional books close **today 4pm ET**; the `xyz:SPCX` perp has bled to **~$157 (+16% over offer, vs ~60% in May)** — the basis the calculator gated green is fading in real time. Pricing is **Jun 11**, first trade **Jun 12**.
- **Two corrections to Alvaro's playbook.** (1) **SPCX will not start trading at 15:30 CET.** Nasdaq IPO listings open via a separate **IPO cross** after a quote/display-only period; mega-IPOs historically print their first trade **hours** after the bell (Cerebras: first cash trade 12:59 ET = **18:59 CET**, 3.5h after the open). Plan the day around a 17:00–20:00 CET first print, not 15:30. (2) His "$135 fixed, confirmed in S-1/A" is too strong — the S-1/A is a red-herring; $135 is *expected*, the EU prospectus caps at **$162**, and the binding price lands in the **424B ~Thursday night**. Treat final price as a Thursday-night decision input, not a constant.
- **Cerebras access question — answered.** Cerebras was **not formally institutional-only**, but it had **no retail tranche and no broker retail program**; at ~20× oversubscribed, retail primary fill was negligible and retail's first access was the ~$350 open. SpaceX is structurally opposite (up-to-30% retail reservation + ~55.6M-share EU tranche via Trade Republic) — which is exactly why a basis-preserving long exists here and didn't for Cerebras. Cerebras is a **tape/path analog, not an access analog**.
- **Status.** Decision tree pre-registered below; open parameters route to Blocks S1 (hedge timing/size grid), S2 (unwind schedule from mega-IPO tape), S3 (IPO-cross data sources + levels), S4 (Trade Republic execution mechanics). Friday-morning live gate decides everything; nothing here is a commitment to trade.

---

## 1. State of play (2026-06-10, all marks dated)

| item | value | source / freshness |
|---|---|---|
| Offer price | $135 *expected*; EU prospectus max **$162**; binding price in 424B (~Jun 11 night) | EDGAR S-1/A + EU prospectus (see [[2026-06-09_spacex_ipo_convergence_trade]] §4) |
| Book | **~$250B orders vs $75B raise (~4× oversubscribed)**; was ~2× on Jun 8 | Reuters via Coindesk 2026-06-10; Bloomberg 06-08 |
| Institutional books close | **Wed Jun 10, 4pm ET (22:00 CET)** | Bloomberg 06-08 |
| Retail | up to 30% of offering reserved; ~55.6M-share EU tranche; TR pro-rata at offer price | EU prospectus; TR announcement |
| `xyz:SPCX` perp | **~$157**, premium ~**+16%** over $135 (was ~+60% May, ~+18% Jun 9) | Coindesk 2026-06-10 (intraday); refresh via calculator |
| Implied basis (if priced $135) | ~$22/share gross, **compressing ~daily** | perp mark − 135 (xyz is per-share, R=1) |
| Justin's position | €10k TR subscription bid ≈ **~80–90 shares requested** at $135 (FX-dependent) | user |
| Free HL margin | **<€2k** → hedgeable ~**14–15 shares at 1×**, ~**21–22 at 1.5×** (at $157 mark) | user constraint |
| Allocation known | Fri Jun 12 ~**8:00 CET** (TR notification) | TR / Alvaro playbook |
| Nasdaq bell / likely first SPCX print | 15:30 CET / **~17:00–20:00 CET** (IPO cross, see §4) | Nasdaq IPO cross docs; Cerebras precedent 18:59 CET |
| Settlement references | `vntl:SPACEX` cash-settles to first-day **close**; `xyz:SPCX` **converts in place** to an equity perp | trade[XYZ]/Ventuals docs (addendum) |

Column notes: "basis" = perp per-IPO-share-equiv − offer; "hedgeable shares" = margin × leverage / perp mark. Fill scenarios in §3.

**The strategic picture in one line:** the convergence basis still exists but is bleeding out (~60% → ~16% premium in four weeks); meanwhile a 4× book makes both a decent pop *and* a small pro-rata fill more likely. The plan must therefore work in the world where the locked basis on Friday morning is anywhere between ~$0 and ~$20/share.

---

## 2. Two frames, one allocation — the two-sleeve structure

**Frame A — Convergence lock** ([[spcx_convergence_calc_findings]]): long allocated shares at offer, short `xyz:SPCX` FDV-neutral (h=1 on the hedged shares), **unlevered or ≤1.5×**, hold both legs through settlement. Locked P&L = basis × hedged shares, direction-independent. Offline gate is green *at the 06-09 basis*; whether it survives to Friday is live-only.

**Frame B — Sell-the-pop** (Alvaro's playbook): no short; sell allocation in tranches into the post-open euphoria window using tape signals (VWAP, volume divergence, lower lows). Directional — profits iff the stock trades above offer when you sell; the Polymarket tail-selling on Alvaro's side pairs with it.

**Why both, sliced by margin:** with <€2k HL margin, Frame A physically cannot cover more than ~15–22 shares. Anything allocated beyond that is *necessarily* unhedged, i.e. Frame B by construction. The two-sleeve split is therefore not a compromise but the only structure consistent with the constraint:

| sleeve | size | strategy | unwind |
|---|---|---|---|
| **Hedge sleeve** | min(fill, margin-supported shares at ≤1.5×, *if* Friday basis clears the go/no-go) | Frame A: short xyz h=1 against these shares | **Simultaneous pair-close** (§5.1) — not a close-price bet |
| **Residual sleeve** | fill − hedge sleeve (possibly all of fill if basis is gone) | Frame B: hardened Alvaro tranche plan | Tranche sells into strength per §5.2 + S2 schedule |

Consistency note (forcing question discipline): the hedge sleeve makes money even if SPCX never moves; the residual sleeve is an explicit directional bet that the 80% P(close>$135) crowd is right. Holding both is coherent because the residual long was *given* to us at the offer price, not bought at the perp's premium.

---

## 3. Fill scenarios × hedge capacity

€10k ≈ 80–90 shares requested (exact count = €10k × EURUSD / final price — Block S1 takes FX live). Fill fractions are scenarios, not predictions; the 4× headline book is institutional and need not equal retail-tranche oversubscription.

| fill | shares (~) | cash deployed | hedgeable at 1× (~14 sh) | hedgeable at 1.5× (~21 sh) | residual sleeve |
|---:|---:|---:|---|---|---:|
| 10% | 8–9 | ~€1k | full | full | 0 |
| 25% | 20–22 | ~€2.5k | partial (14) | ~full (21) | 0–8 |
| 50% | 40–45 | ~€5k | 14 | 21 | 19–24 |
| 100% | 80–90 | ~€10k | 14 | 21 | 59–69 |

Read: at small fills the entire position can be locked; at large fills the plan is mostly Frame B whatever we decide. The expected case (high oversubscription → fill ≤ ~25%) is *fully hedgeable at ≤1.5×* — the constraint binds only in the high-fill branches, where the extra shares are house-money exposure anyway. **Open parameter → Block S1:** the exact hedge-size rule on the (final price × fill × Friday basis) grid, including whether 1.5× is justified vs 1× given the melt-up survival math.

---

## 4. How the IPO actually clears (and why Alvaro's clock is wrong)

Mechanics (Nasdaq IPO cross, see sources): the stock is in a **halt state** at the 9:30 ET bell. Nasdaq runs a **quote-only / display-only period** (~15 min minimum, extended in 5-min increments on volatility) during which members enter orders and Nasdaq disseminates the **Net Order Imbalance Indicator (NOII)** — paired shares, imbalance, and an **indicative clearing price** updated every second. The underwriter's stabilization desk (lead left: Goldman/Morgan Stanley) tells Nasdaq when to release; the **IPO cross** then executes as a single bulk print (the Nasdaq Official Opening Price), and continuous trading begins. For large IPOs this release routinely happens **2–4+ hours after the bell** — Cerebras first traded at 12:59 ET (18:59 CET); mega-IPOs historically print late morning–midday ET.

Implications, in order of importance:

1. **Re-anchor the whole afternoon:** allocation 8:00 CET → ~9–11 hours of perp-only signal → bell 15:30 CET (nothing tradable yet) → first print plausibly **17:00–20:00 CET** → close 22:00 CET. The "peak euphoria window" in Alvaro's playbook starts at the *cross*, not at 15:30. With a late cross, the window between first print and close compresses to 2–4 hours — tranche timing must be defined relative to the cross time, not wall-clock (Block S2).
2. **The indicative price is watchable.** During the display-only period the NOII indicative clearing price is the best pre-trade truth — better than the perp, because it aggregates real auction orders. **Open parameter → Block S3:** where retail can see NOII/indicative price free or cheap in the EU (Nasdaq tools, broker L2 feeds, financial-TV tickers), and the fallback if nowhere.
3. **Opening-print discipline survives:** do not sell into the first prints. The cross is a single clearing price; immediately after it, spread and volatility are at session max. Alvaro's "observe 10–15 min" rule is right — just starts at ~17:00–20:00 CET.
4. **The perp converges to the indicative, then the print.** Cerebras's perp pre-discovered the open ~2h ahead. From ~16:00 CET the xyz perp + NOII together give a high-quality forecast of the cross price — this is the input for last-minute residual-sleeve sizing decisions and Alvaro's Polymarket entries.

---

## 5. Unwind mechanics, per sleeve

### 5.1 Hedge sleeve — simultaneous pair-close, not a close bet

Because `xyz:SPCX` **converts in place** to a regular equity perp (it does not cash-settle at the close), the lock is realized by closing both legs at the same moment T after the perp tracks the listed stock: sell shares at S(T) on TR, buy back the perp at ~S(T) on HL → total = locked basis regardless of S(T). Rules:

- Wait until perp–spot tracking is confirmed (Cerebras: within ~$1–4 from day 1 post-open; verify live).
- Pick T for **liquidity, not price**: mid-session after the cross settles (e.g. 1–3h post-print), spreads tight on both venues. Avoid the first 30 min and the closing auction.
- Execute the TR sell first (slower venue, limit order), then immediately close the perp (fast venue). Leg risk is seconds-to-minutes of one-sided exposure; in a fast tape close the perp first only if short-side buffer is thin.
- If using `vntl:SPACEX` instead (thinner; avoid unless xyz is dislocated): it cash-settles to the **close**, so the share leg must be sold **at/near the close** (TR has no MOC order — limit order in the last minutes; slippage vs the official close is a real residual → Block S4 confirms TR's practical latency).
- Liquidation watch runs all day: `spcx_convergence_calc.py --watch` with `--live-entry/--live-short-notional/--live-margin --alert-buffer-pct` + `--parquet-log`. At 1×–1.5× the buffer survives a Cerebras-style +39% spike, but a >+50% freak print during the pre-cross hours is the tail to watch.

### 5.2 Residual sleeve — hardened Alvaro plan

Alvaro's 40/40/20 tranche structure, peak signals (volume declining at highs, bearish divergence, VWAP loss, first lower low, spread widening, perp divergence), and hard stops ($140 reassess / $125 sell-everything) are kept, with these hardenings:

- **Clock re-anchored to the cross** (per §4): Phase A = first 10–15 min post-print; Phase B/C windows defined in minutes-since-cross, not CET wall-clock. *S2 calibrated this (2026-06-10, [[spcx_ipo_unwind_tape_findings]]): on the only surviving intraday tape (Cerebras 1m) the volume-weighted peak was +1 min after the cross, anchored VWAP was lost +12 min and never reclaimed, and 52% of day-1 volume traded in the first 30 min; across 6 mega-IPOs the cross print averaged ~89% of the day-1 high. Read: schedule is second-order vs the PEAK signal set; front-load the tranches if anchored VWAP is lost early, and don't carry residual overnight (day-2 closed lower in 5/6).*
- **Limit-order-only** on TR (his rule, kept), 1–2 ticks inside the bid, re-pegged on a 5-min timer.
- **VWAP from the cross**, not from 15:30 (most charting tools anchor VWAP at 9:30 ET — an IPO needs anchored VWAP from the first print; Block S3 documents how to set this on TradingView).
- **One coordination trigger:** the residual sleeve's "PEAK" call and Alvaro's Polymarket tail-sell should fire on the same signal set — agree the exact definitions pre-session (the playbook's §5.1 list, made binary).
- Fade-risk asymmetry at 4× oversubscription: the bigger the book, the more pre-IPO holders/insiders are *not* sellers day 1 (lockups) but also the hotter the open — S2 should check whether mega-IPO fade size correlates with oversubscription before trusting the -13% fade estimate. *S2 answer: no relation on n=5 (ARM 10×→4% fade but CBRS 20×→19%; the ~SPCX-sized books sat mid-range) — keep −13% as an unconditional prior (sample median ~14%).*

### 5.3 Day-of screen map + indicator set (SUPERSEDED for feeds by [[spcx_listing_data_sources]] §1 — S3's verified runbook; indicator set below remains canonical)

S3 verdicts that overwrite this table's open cells: **no practical EU retail access to the official NOII** (TotalView-only; Webull = NL-only; IBKR ~$15/mo only if a funded account exists) → pre-registered proxy = xyz perp + CNBC/newswire "indicated to open" headlines; **TradingView free tier streams real-time Cboe BZX** (sufficient post-cross; Nasdaq-primary is 15-min delayed; $3/mo add-on needs paid plan/trial → take the 30-day trial Thursday, which also fixes the ~3-alert free cap); anchored VWAP = the free *drawing tool*, click-anchored on the first 1m candle; **Thursday-night final price: newswire beats EDGAR** (424B typically lands Friday pre-market; EDGAR RSS CIK 0001181412 form 424B as backstop).

**Rule zero: Trade Republic is the order-entry venue only, never a price feed or chart.** Its quotes (LS Exchange/Lang & Schwarz mirror) and UX are not built for this; every read happens elsewhere, only the sell tickets happen in TR.

| window (CET) | primary feed | what to read on it |
|---|---|---|
| 8:00–15:30 pre-open | Hyperliquid `xyz:SPCX` (app.hyperliquid.xyz/trade/SPCX, no account needed) + the `--watch` terminal | The only live SpaceX price. Level vs $135/final price; trend into the open; liq-buffer line if short is on |
| 8:00–15:30 | Polymarket SPCX markets (+ S5 dashboard once built) | PM-implied close distribution (mean/median/P75/P90), tail prices — Alvaro's entries + our pop prior |
| 15:30→cross (display-only) | NOII / indicative clearing price — source TBD by S3 (likely IBKR/Webull L2; fallback: perp + newswire headlines) | Indicative price vs perp vs PM mean: the best pre-trade truth about the cross |
| cross→22:00 | TradingView SPCX 1m + 5m (caveat: free tier is 15-min-delayed Nasdaq — S3 confirms cheapest real-time option; a delayed chart is useless on this day) | All indicators below |
| all day | HL perp vs listed price (after cross) | Perp leading down = leveraged money bailing first (early fade warning) |

Indicator set (residual sleeve, all post-cross, all real-time computable — this is Alvaro's §5 list made canonical):

1. **Anchored VWAP from the first print** (not session-anchored 15:30 VWAP) — the primary trend filter; price losing anchored VWAP and failing to reclaim = fade underway → accelerate tranches.
2. **5m volume at highs** — new price high on lower volume than the prior high (bearish divergence) = exhaustion → PEAK candidate.
3. **First lower low on 5m closes** — uptrend structurally broken.
4. **Spread width** — widening bid-ask (e.g. $0.10→$0.50) = market-maker uncertainty.
5. **Perp–spot divergence** — perp dropping while stock holds = sell.
6. **PM tail repricing** — >$2.4T/>$3T YES bids fading = crowd lowering the close → Alvaro signal, shared PEAK trigger.
Buy-side confirmations (hold, don't sell): dips bought within 1–2 min, green-candle volume > red-candle volume, new highs **on** volume spikes, tight spread.

PEAK = (2) + any one of (1)/(3)/(5)/(6), pre-agreed with Alvaro as the simultaneous stock-tranche + PM-tail-sell trigger.

### 5.4 Hard risk rules (both sleeves, pre-registered now)

- Perp leverage **≤1.5×**, ever ([[spcx_convergence_calc_findings]] survival math).
- No naked short beyond the pre-hedge rule that S1 outputs; if S1's rule isn't built/green by Thursday night, **no pre-hedge at all** — hedge only after allocation is known.
- If final price prints **>$135**, recompute basis before any hedge (at $162 the basis is roughly gone; auto no-trade for the hedge sleeve unless perp >$170s).
- If Friday-AM net basis < the S1 go/no-go threshold → hedge sleeve = 0, everything runs Frame B.
- Alvaro's $125 hard stop and €3k team max-loss stand unchanged on the residual sleeve.

---

## 6. Decision tree (the gameplan proper)

**D0 — Now → Thu morning (Jun 10–11).** No position. Snapshot the calculator when convenient (cached `latest.json`); continuous `--watch --parquet-log` is NOT required before Friday — S1's decay fit uses HL candle history. The logger becomes mandatory **Thursday evening onward** (pricing-night reaction, allocation morning, in-session liq alarm + funding/oracle/OI capture + purge insurance). Human actions (now fully specified by S3/S4): run the **8-question TR in-app support chat** ([[spcx_trade_republic_execution_findings]] §4 — Q1 flipping + Q2 allocation booking/pre-cross LS quoting are gating; paste answers back into that note). Thursday evening: take the **TradingView 30-day trial** (+$3 Nasdaq add-on, fixes the alert cap), set the S3 alert ladder, set the EDGAR 424B RSS backstop, and start both watchers (`spcx_convergence_calc.py --watch` + `spcx_pm_pdf_monitor.py --watch 45 --parquet-log --html`).

**D1 — Pricing night (Thu ~22:00 CET, 424B).** *S1 resolved this node (2026-06-10 run, perp $162.21/basis ~$27):* **do NOT pre-hedge.** Read the final price, re-run `--decision --offer <final>`, and do nothing — unless the perp has spiked through the frozen trigger **Z\* ≈ $36/sh over a $135 fill (perp ≥ ~$171)**, in which case the pre-hedge tranche opens at pessimistic-fill size only (never above the 10%-fill row of §3). Waiting dominates every pre-node otherwise: decay forfeited (~$1.8–4.3/sh over 2d) ≪ naked-melt-up premium (~$10.6/sh at P(fill≥10%)=0.80) + wait-option value. Assumptions CLI-tunable (`--p-fill`, `--meltup-dist`); *S2 recalibrated the melt-up distribution (2026-06-10, [[spcx_ipo_unwind_tape_findings]] §7): measured day-1 high-vs-offer across 6 mega-IPOs gives `--meltup-dist 0.184:1,0.300:1,0.466:1,0.532:1,0.700:1,1.088:1` (E=+54.5% vs the old +26%), which raises the frozen trigger to **Z\* ≈ $48/sh (perp ≥ ~$183)** — the no-pre-hedge verdict gets strictly stronger. Use this string in the Thursday-night `--decision` re-run.*

**D2 — Allocation (Fri ~8:00 CET).** Read exact share count. *S1 rule:* hedge sleeve = **min(fill, ~21.4 sh at 1.5× / ~14.3 sh at 1×) iff live net basis > 0** (auto-dead at a $162 print); open the short to h=1 on those shares. S1's Friday-8:00 projected mark is ~$158–160.4 (rate-only; live premium was below trend and rising — treat as indicative only). Decide residual-sleeve tranche sizes from the §3 row that materialized. Communicate the split to Alvaro (his scenario matrix keys off the same allocation number).

**D3 — Pre-open (Fri 8:30–15:30 CET).** Perp is the only signal; watch level vs $135 and vs the hedge entry. No action except liquidation-buffer monitoring. If the perp collapses to ~$135–140, the pop thesis is weakening: pre-agree with Alvaro what perp level (S1 outputs it) flips the residual sleeve from "sell into strength" to "sell early, risk-off."

**D4 — Display-only period → IPO cross (15:30 → ~17:00–20:00 CET).** Watch the indicative-price **proxy stack** (S3: xyz perp + CNBC/newswire "indicated to open" headlines + TR/LS pre-cross quote if it exists — HUMAN-CHECK #5; official NOII is not EU-retail-accessible) + the S5 monitor's crowd-vs-perp gap. No selling into the cross or first prints. Note cross time and price; start the S2-calibrated tranche clock.

**D5 — Post-cross session → close (cross → 22:00 CET).** Hedge sleeve: pair-close per §5.1 once tracking confirmed and books are deep. Residual sleeve: tranches per §5.2 with the peak-signal set; hard stops live. Last 30 min: no new decisions except Alvaro's Polymarket-resolution-driven needs (the close prints his settlement, not ours — our hedged sleeve should already be flat).

**Post-day.** Parquet log + fills → a `spcx_listing_postmortem` findings note; feed realized basis-at-each-node back into the calculator's assumptions ledger.

---

## 7. Research blocks for Claude Code (run before Friday)

All blocks are **independent** (parallelizable) except S1's dependence on the watch-log started at D0. Each lands as a findings note + (where applicable) code under `polymarket/research/`, per [[VAULT_MAP]] § Where to write things. Prompts live in chat per [[COWORK]] § prompt discipline — not committed.

**Block S1 — Hedge grid + pre-hedge timing rule (extends the queued calculator enhancement; highest priority).**
Extend `spcx_convergence_calc.py`: (a) fill-price axis $135→$162 × fill-fraction {10/25/50/100%} × margin constraint (configurable, default €2k ≈ live-FX USD) → hedgeable shares, locked $ and ROC at 1× and 1.5×, per cell; (b) basis-decay model fit on HL hourly/15m candles since mid-May (premium 60%→16%: fit decay rate + uncertainty) — candle history suffices, no live tape required; use any existing `--watch` parquet shards only as a supplementary cross-check; (c) the **pre-hedge timing EV**: for hedge-at-{now, D1, D2} × pessimistic-fill scenarios, EV = P(fill ≥ pre-hedge) × E[basis at that node] − P(shortfall) × E[naked-short loss | melt-up dist from Cerebras analogs], output as a **single pre-registered rule** ("hedge X shares at node Y iff basis ≥ Z"). Acceptance: unit tests extend the existing 11; rule reproduces the trivial corners (zero margin → never; basis 0 → never); a one-page decision table printed by a `--decision` flag. *Deliverable: extend [[spcx_convergence_calc_findings]] + script.* **DONE 2026-06-10** (27 tests green; rule: do NOT pre-hedge, hedge at D2 iff net basis > 0, ceiling ~21.4 sh at 1.5×; trigger raised to ~$183 by S2's measured melt-up dist).

**Block S2 — Unwind schedule from mega-IPO tape.**
Empirical study of listing-day microstructure for Cerebras (HL 15m perp + Yahoo 5m spot already cached) plus as many mega-IPO day-1 tapes as Yahoo/Stooq give at 1–5m (candidates: ARM 2023, Rivian 2021, Alibaba 2014, Facebook 2012, Reddit 2024, Cerebras 2026): minutes-from-first-print to volume-weighted intraday peak, fade onset/depth, % of day-1 volume by 30-min bucket, VWAP-loss timing, and whether a 40/40/20 tranche plan beats TWAP-from-cross and beats sell-at-cross, net of a $0.10–0.50 slippage budget. Lookahead-free: signals computable in real time only. Also: fade size vs oversubscription ratio across the sample (does a 4× book change the prior?). Acceptance: every chart annotated with axes/units/sample; tranche-rule comparison table with CIs across the IPO sample; an explicit "n is tiny, this is calibration not proof" header. *Deliverable: `spcx_ipo_unwind_tape_findings.md` (market_maps) + scripts.* **DONE 2026-06-10 → [[spcx_ipo_unwind_tape_findings]]** (CBRS turned out to have a real 1m tape — cached before Yahoo purges it; the other five degrade to labelled daily-OHLC proxies; policies statistically tied, PEAK signals primary; measured melt-up dist shipped to S1).

**Block S3 — IPO-cross visibility + data-source map (desk research, no code).**
Answer: where can an EU retail trader watch, free or cheaply, (a) the Nasdaq NOII / indicative clearing price during the display-only period, (b) real-time SPCX prints (TradingView free tier = 15-min delayed Nasdaq unless paid/broker feed — confirm; TR app feed latency — confirm), (c) L2 depth (Webull/IBKR/other free options that work in the EU). Document the exact TradingView setup: anchored-VWAP-from-first-print, 1m/5m layout, alert list. Output a one-page "screens" runbook replacing Alvaro's §6 with verified sources. Acceptance: every claim carries a checked URL + date; anything unverifiable marked HUMAN-CHECK. *Deliverable: `spcx_listing_data_sources.md` (market_maps).* **DONE 2026-06-10 → [[spcx_listing_data_sources]]** (no EU NOII access → perp+newswire proxy; TradingView free = real-time Cboe BZX, take the trial Thursday; newswire beats EDGAR for the Thursday price; 7 non-blocking HUMAN-CHECKs).

**Block S4 — Trade Republic execution mechanics (desk research + checklist for the human in-app check).**
Resolve Alvaro's research items 1–10 + 16–17 from published TR docs: order types for US stocks (market/limit/stop; trailing?), partial sells, fractional handling, pre-staging orders before US open, real-time vs delayed quotes, €1 fee semantics, FX conversion timing/spread, T+1 settlement, outage fallbacks (web interface? phone desk?), and any day-1 flipping restriction (none found in published terms as of 06-09 — re-verify). Where docs are silent, output the exact question list for the in-app support chat (human task, D0). Acceptance: table of answer / source / confidence / HUMAN-CHECK flag. *Deliverable: `spcx_trade_republic_execution_findings.md` (market_maps).* **DONE 2026-06-10 → [[spcx_trade_republic_execution_findings]]** (no published flipping rule; no MOC/trailing; EUR-on-LS structure; 8-question in-app chat list — Q1/Q2 gating, human runs it at D0).

**Block S5 — Live PM-PDF + cross-venue implied-price monitor (the one dashboard worth building).**
The convergence `--watch` web dashboard stays declined ([[spcx_convergence_calc_findings]] — infra-before-signal). This is a different object with a live decision attached: on the day, the PEAK call and Alvaro's tail-selling need the **PM-implied closing-cap distribution recomputed continuously** (when the stock rips, the >$2.4T/>$3T tails reprice in minutes — exactly when selling them is the trade), and nobody can refit a PCHIP survivor curve by hand mid-session. Alvaro's playbook lists this as his research items 14–15 and it does not exist yet. Scope (deliberately small): one read-only script, `scripts/spcx_pm_pdf_monitor.py` — poll PM CLOB (16-strike threshold surface + bucket markets, bid/ask not last) and HL `xyz:SPCX` every 30–60s → monotone PCHIP survivor fit → PDF stats (mean/median/mode/P25/P75/P90/P95 in cap *and* per-share on the 13.076B base), bucket-vs-continuous mispricing table (the addendum's $1.5–2.0T diagnostic, live), perp vs PM-mean gap, and after the cross a listed-price input → "what the crowd implies vs where it trades now". Terminal rich-table first; optional `--html` static auto-refresh snapshot (one file, no server) so it's glanceable on a phone/second screen. Append-only parquet log of every poll. **Not** Streamlit, no DB, no `live_trading/` imports. Acceptance: survivor monotonicity enforced + unit-tested; reproduces the addendum's 06-07 table from cached data within tolerance; degrades gracefully when a strike is one-sided/empty; resolves the addendum's open "−$3.3/share EV" sign discrepancy while implementing the EV integral. *Deliverable: script + `spcx_pm_pdf_monitor_findings.md` (market_maps).* **DONE 2026-06-10 → [[spcx_pm_pdf_monitor_findings]]** (live-verified; P(close>$135)=82.2%, mean $167; 1.5–2.0T bucket edge collapsed to +1.5pp; perp now −$3.6 below PM mean; −$3.3 EV line = payoff bug, dead — correct EV +$31.9).

**Friday gate (not a block — the live go/no-go).** Re-run the calculator on Friday-AM marks after allocation: if net basis ≥ S1 threshold → hedge sleeve on; else Frame B only. This gate cannot be pre-computed; it is the whole point of the *terminus = live* rule.

---

## 8. What is already answered (don't re-research)

- **Cerebras retail access:** institutional-skewed in practice, no retail tranche/program, ~20× oversubscribed, retail's first access was the ~$350 open; not formally retail-excluded. Confirmed by vault sources + fresh search (Investing.com, sahmcapital, TECHi, techstackipo). Consequence: the Cerebras convergence trade did not exist for retail; SPCX's is only possible because of the EU tranche.
- **Units/sizing:** xyz is per-share (R=1, basis = mark−135, 1:1 short); vntl is valuation-units (÷$1e9, ~76.5 contracts per 1,000 shares). [[spcx_convergence_calc_findings]] — settled, tested.
- **Leverage:** ≤1.5× or the Cerebras-style spike liquidates the short before it pays. Settled, tested, confirmed on real CBRS tape.
- **Settlement:** vntl → first-day close, cash; xyz → convert-in-place (pair-close unwind, §5.1). From venue docs.
- **$135 not yet binding; $162 EU ceiling; allocation pro-rata, can be ~zero; cash locked during window.** EDGAR + EU prospectus, see snapshot note.

## 9. Sources (fresh, 2026-06-10)

- [Coindesk — SPCX perp down 27% in 3 weeks; 4× oversubscribed; ~$250B book](https://www.coindesk.com/markets/2026/06/10/spacex-s-most-active-pre-ipo-market-has-fallen-27-in-three-weeks)
- [Bloomberg — books close Wednesday](https://www.bloomberg.com/news/articles/2026-06-08/spacex-ipo-is-said-to-be-well-oversubscribed-orders-close-wed) · [Yahoo — order books closing](https://finance.yahoo.com/markets/stocks/articles/spacex-ipo-oversubscribed-order-books-123545689.html) · [Seeking Alpha — 2× (Jun 8)](https://seekingalpha.com/news/4601213-spacex-ipo-over-two-times-oversubscribed)
- [CNBC — fixed $135 roadshow](https://www.cnbc.com/2026/06/03/spacex-ipo-stock-price-roadshow-musk.html) · [CNBC — retail allocation up in the air (Jun 9)](https://www.cnbc.com/2026/06/09/spacex-ipo-explained-stock-price-date.html) · [IPOScoop — $135/$75B](https://www.iposcoop.com/the-ipo-buzz-spacex-spcx-proposed-sets-135-share-ipo-price-to-raise-75-billion/)
- [Nasdaq IPO Cross FAQ (pdf)](https://www.nasdaqtrader.com/content/productsservices/trading/ipohalt/ipo_faq.pdf) · [IPO Cross fact sheet (pdf)](https://www.nasdaqtrader.com/content/productsservices/trading/IPOCross_fs.pdf) · [Nasdaq auction price discovery](https://www.nasdaq.com/articles/automation-and-information-produce-efficient-price-discovery-in-nasdaqs-auction-process)
- [TR IPO access (EQS)](https://www.eqs-news.com/news/corporate/starting-today-trade-republic-gives-european-retail-investors-direct-access-to-ipos/cb7d9263-f112-4539-a30e-2a87dcf950f7_en) · [TR order types](https://support.traderepublic.com/en-de/767-How-can-I-set-a-limit-order)
- Cerebras access: [Investing.com](https://www.investing.com/analysis/cerebras-48-billion-ipo-tests-the-markets-inference-bet-200680080) · [sahmcapital](https://www.sahmcapital.com/news/content/cerebras-ipo-the-market-is-already-mispricing-the-real-risk-2026-05-14) · [TECHi 20× oversubscribed](https://www.techi.com/cerebras-ipo-price-range-bump-150-160/) · [techstackipo trading day](https://www.techstackipo.com/ipo/cerebras/trading-day)

> Not investment advice — structural analysis for a personal-size position. Refresh every number at every decision node.
