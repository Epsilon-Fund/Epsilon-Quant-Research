---
title: "Mega-IPO listing-day unwind tape study — what six day-1 tapes say about selling SPCX on Friday (Block S2)"
tags: [spacex, spcx, ipo, unwind, microstructure, tape-study, cerebras, findings]
created: 2026-06-10
status: "done — calibration delivered; feeds the gameplan §5.2 residual sleeve + S1's --meltup-dist; § Block S6 added 2026-06-11 (perp exit mechanics: pair-close window + forced-flat backstop)"
---

# Mega-IPO listing-day unwind tape study — selling an IPO allocation on day 1 (Block S2)

> Hub: [[spacex_ipo_market_map_handoff]] · [[POLYMARKET_BRAIN]] · [[COWORK]]
> Companions: [[spcx_listing_day_gameplan]] (§5.2 residual sleeve, §7 Block S2 — the spec this answers) · [[spcx_convergence_calc_findings]] (Block S1 — consumes §7 below) · Table terms: [[polymarket_table_dictionary]]
> Code: `scripts/spcx_ipo_unwind_tape_study.py` (17 unit tests in `tests/test_spcx_ipo_unwind_tape_study.py`)

> ## ⚠ TINY n — CALIBRATION, NOT PROOF
> Six IPOs total, exactly **one** of which (Cerebras) still has a real intraday tape. Every comparison below is a prior-setter for Friday's SPCX unwind, not a validated strategy result. Bootstrap intervals are width illustrations on n=6, not inference.

## Plain-English Summary

- **What this is.** The empirical day-1 study behind the SPCX residual-sleeve unwind: when does a mega-IPO actually peak after its opening cross, how fast does it fade, where does the volume go, and which mechanical sell schedule — Alvaro's 40/40/20 tranches, TWAP-from-cross, or sell-everything at cross+15min — would have gotten the best average price.
- **Sample honesty up front.** Free intraday history is gone for everything older than ~30 days: ARM 2023, Rivian 2021, Alibaba 2014, Facebook 2012 and Reddit 2024 survive only as daily OHLC (Yahoo returns HTTP 422 for their intraday). Cerebras 2026-05-14 is inside Yahoo's 1-minute retention window, so it gives the one real minute-by-minute simulation — and the script cached that tape to CSV before it too gets purged.
- **Headline microstructure result (Cerebras).** The day's volume-weighted peak was **+1 minute after the cross**; the anchored VWAP was lost **+12 minutes** in and never reclaimed; **52% of all day-1 volume traded in the first 30 minutes**. There was no "peak euphoria later in the afternoon" — the cross *was* the euphoria.
- **Headline policy result.** Across all six IPOs the three policies land within ~1% of each other on average (88–89% of the day-1 high captured); which one wins is decided by whether the day faded (sell-early wins: CBRS, META) or rallied into the close (patience wins: ARM, RDDT) — a coin flip the schedule cannot predict. **Schedule choice is second-order; the gameplan's real-time PEAK signal set is what matters.** On the one real tape, sell-all-at-cross+15 beat both slower schedules by ~3–4.5% of the high.
- **The S1 hook delivered.** The measured day-1 high-vs-offer distribution across the six IPOs is `+18/30/47/53/70/109%`, E[move] = **+54.5%** — roughly double the old equal-weighted +13/26/39% Cerebras assumption. Fed into the S1 calculator it raises the pre-hedge trigger Z\* at a $135 fill from ~$36 to **~$48/share** (perp ≥ ~$183): the pre-registered **do-NOT-pre-hedge verdict gets strictly stronger**.

---

## 1. Design and data: what actually survives

Unit of observation: one IPO's first trading day. "Cross" = the opening auction print (Nasdaq IPO cross / NYSE open), the first moment the allocation is sellable. All intraday clocks are minutes-since-cross, matching the gameplan's re-anchored tranche windows.

| IPO | listing day | offer $ | tape available | source | status |
|---|---|---|---|---|---|
| Cerebras (CBRS) | 2026-05-14 | 185 | **real 1-minute** spot, cross→close (181 bars) + HL 15m perp (cached by the case study) | Yahoo 1m (inside retention) | full intraday simulation |
| ARM | 2023-09-14 | 51 | daily OHLCV only | Yahoo 1d (intraday purged, HTTP 422) | daily PROXY |
| Rivian (RIVN) | 2021-11-10 | 78 | daily OHLCV only | same | daily PROXY |
| Alibaba (BABA) | 2014-09-19 | 68 | daily OHLCV only | same | daily PROXY |
| Facebook (META) | 2012-05-18 | 38 | daily OHLCV only | same | daily PROXY |
| Reddit (RDDT) | 2024-03-21 | 34 | daily OHLCV only | same | daily PROXY |

Every fetched series is cached under `data/analysis/csv_outputs/market_maps/ipo_tapes/` — in particular `cbrs_spot_1m_listingday.csv`, which Yahoo will purge ~30 days after listing; the study reruns offline from the caches.

**Lookahead discipline.** All three policies are *fixed schedules known at the cross* — no execution price depends on any future bar (unit-tested by truncating the tape at each execution minute and checking the price is unchanged). Descriptive statistics (peak timing, fade depth) describe the day after the fact and are never used as trade signals.

---

## 2. Cerebras on the 1-minute tape — the only real intraday look

![Cerebras listing day, 1-minute tape: price, anchored VWAP, volume, and the three unwind policies](../../../data/analysis/plots/spcx_convergence/ipo_day1_path_cbrs_1m.png)

*Read: black = CBRS 1m closes from the 16:59Z cross to the 20:00Z cash close; purple dashed = VWAP anchored at the cross (the gameplan's primary trend filter); gray bars (right axis) = 1m volume; orange band = the 40/40/20 tranche windows; red dot = the sell-all-at-cross+15 execution ($337). The opening print region ($350 official open / $385 first 1m close) was the high of the day; price never traded above the cross bar again.*

Key measured numbers (definitions in parentheses):

| metric | value | definition |
|---|---|---|
| volume-weighted intraday peak | **+1 min** after cross (~$373) | max of the trailing 15-bar (15-min) rolling VWAP — where volume-weighted price, not a single print, topped out |
| fade onset | +2 min | first 1m close ≥5% below the *running* intraday high (running max uses past bars only) |
| anchored-VWAP loss | **+12 min**, never reclaimed | first close below the cross-anchored VWAP that stays below for ≥10 consecutive minutes |
| fade depth | **19.4%** (385 → 310.30) | (day high − close) / high |
| close vs offer | +68% | the fade still left the day deep green vs the $185 offer |

The fade-onset number needs a caveat: a −5% trigger fires inside the opening volatility burst (the tape whipped $385→$334→$370 in the first ten minutes), so "+2 min" overstates how fast the *structural* fade began. The anchored-VWAP loss at +12 min — never reclaimed for the remaining ~2.8 hours — is the honest marker: by the gameplan's own indicator set, the sell signal was on twelve minutes into the session.

![Cerebras day-1 volume by 30-minute bucket since the cross](../../../data/analysis/plots/spcx_convergence/ipo_day1_volume_buckets_cbrs.png)

*Read: x = 30-min buckets since the 16:59Z cross, y = % of total day-1 volume. **52% of all day-1 volume traded in the first 30 minutes**, 68% in the first hour; the remaining ~2 hours carried ~7–8% per bucket.*

Practical example of why this matters for SPCX: suppose the residual sleeve is 60 shares and the SPCX cross prints at 18:00 CET. If the tape is CBRS-shaped, by 19:00 CET two-thirds of the day's liquidity has already traded and the anchored VWAP has been lost for ~45 minutes — a tranche plan that still holds 60% of the sleeve at that point (as the mechanical 40/40/20 below does) is selling into the thin, fading two-thirds of the session. Front-load while the first-hour depth exists; TR limit orders 1–2 ticks inside the bid fill easily in a 52%-of-volume half hour and poorly after it.

---

## 3. The cross already captures most of the day — open/high/close vs offer

![Day-1 open, high and close vs the IPO offer across the six-IPO sample](../../../data/analysis/plots/spcx_convergence/ipo_day1_high_close_vs_offer.png)

*Read: for each IPO, the official opening-cross print (blue), day-1 high (red — these feed S1's melt-up distribution, §7) and day-1 close (gray), all as % above the offer price. Daily Yahoo OHLC; CBRS bars agree with the 1m tape.*

| IPO | open vs offer | high vs offer | close vs offer | open as % of day-1 high |
|---|---:|---:|---:|---:|
| Cerebras 2026 | +89% | +109% | +68% | 90.6% |
| ARM 2023 | +10% | +30% | +25% | 84.6% |
| Rivian 2021 | +37% | +53% | +29% | 89.4% |
| Alibaba 2014 | +36% | +47% | +38% | 93.0% |
| Facebook 2012 | +11% | +18% | +1% | 93.4% |
| Reddit 2024 | +38% | +70% | +48% | 81.3% |
| **mean** | | **+54.5%** | | **88.7%** |

The decision-relevant row is the last column: **the opening cross alone captured 81–93% (mean ~89%) of the day's eventual high** in every one of six mega-IPOs. The upside from holding past the cross for "the pop" averaged ~+13% of the cross price to the (unknowable in real time) top, against fade-to-close risk of similar size. This is the empirical backing for the gameplan's "do not romanticize the late-day peak" stance — most of the euphoria is in the cross price itself, which the NOII indicative will telegraph before the first print.

---

## 4. Policy comparison — 40/40/20 vs TWAP-from-cross vs sell-at-cross+15

The three policies, mechanically pre-registered (all clocks = minutes since cross; the cross bar is minute 0):

- **A — "Alvaro 40/40/20" (mechanical form):** observe minutes 0–14 (no selling — gameplan §5.2 Phase A); sell 40% as a TWAP over minutes 15–59, 40% over 60–179, 20% over 180–close. On CBRS's compressed 181-minute session the schedule's average execution lands around minute ~99 (on a full-length session it drifts later, ~110+). This is the *signal-free skeleton* of Alvaro's plan; his real plan accelerates on PEAK signals, which no historical mechanical version can honestly replicate.
- **B — TWAP-from-cross:** equal-weight average of every 1m close, cross to close. Average execution = mid-session (~minute 90 on CBRS) — slightly *earlier* than A's on a short session, which is why A and B are near-twins in the table below.
- **C — sell-all-at-cross+15:** one print, the 1m close 15 minutes after the cross.

**Daily-PROXY rows** (everything except CBRS) cannot be simulated and use OHLC stand-ins, labelled as such: C ≈ official open (the cross print itself; the actual cross+15 price is unknowable from daily data), B ≈ (O+H+L+C)/4, A ≈ 0.4·O + 0.4·(O+H+L+C)/4 + 0.2·C. These are crude — treat the PROXY rows as ±2–3%-of-high noise around the truth.

| IPO (tape kind) | A 40/40/20, % of high | B TWAP, % of high | C cross+15, % of high | best | day shape |
|---|---:|---:|---:|---|---|
| Cerebras 2026 (**1m sim**) | 82.9 ($319.35) | 84.1 ($323.89) | **87.5** ($336.78) | C | faded from the cross |
| ARM 2023 (PROXY) | 89.5 | **91.1** | 84.6 | B | rallied into close |
| Rivian 2021 (PROXY) | 87.9 | 88.3 | **89.4** | C | faded |
| Alibaba 2014 (PROXY) | 93.8 | **94.3** | 93.0 | B | flat-ish |
| Facebook 2012 (PROXY) | 90.7 | 90.7 | **93.4** | C | faded hard |
| Reddit 2024 (PROXY) | 84.6 | **86.6** | 81.3 | B | rallied intraday |
| **mean (boot95)** | 88.2 [85.3, 91.1] | 89.2 [86.5, 91.9] | 88.2 [84.8, 91.6] | — | — |

Column glossary: each policy cell = gross average sell price as % of the day-1 high (higher = better; 100% is unattainable). "boot95" = 10k-draw bootstrap over the six per-IPO values — a width illustration on n=6, per the header. Full dollar table incl. net-of-slippage columns: `data/analysis/csv_outputs/market_maps/ipo_unwind_policy_comparison.csv`.

![Policy comparison: average sell price as % of day-1 high, per IPO](../../../data/analysis/plots/spcx_convergence/ipo_unwind_policy_comparison.png)

*Read: grouped bars per IPO, one per policy. Note the pattern, not the levels: C (red) wins on fade days and loses on rally days; A and B are nearly indistinguishable everywhere.*

**Slippage.** The S2 spec's $0.10–0.50/share budget is flat per share, so it subtracts identically from every policy and **cannot change the ranking** — at SPCX-like prices ($135–200) even $0.50 is ~0.3% of price, an order of magnitude below the ~±4%-of-high spread between policies on a given day. Rankings would only move under per-order or spread-scaled costs (more orders = worse for A/B), which marginally favors C. Net columns at all three budgets are in the CSV.

**Read / interpretation.** Three honest conclusions, in order of confidence:

1. **The schedule is second-order.** Means are statistically and economically indistinguishable (88.2 / 89.2 / 88.2). What separates outcomes is the day's shape — fade vs rally — which is exactly what the gameplan's real-time PEAK signal set (anchored-VWAP loss, volume divergence, lower lows, perp divergence) exists to detect. S2's verdict: keep the signal set as the primary control, treat the 40/40/20 schedule as a default pace, not a commitment.
2. **On the only real tape, early selling won.** CBRS gave C +4.5%-of-high over A — and the anchored-VWAP-loss signal at +12 min would have told a signal-driven seller the same thing in real time. If Friday's tape looks CBRS-shaped (cross near the perp's pre-discovery level, immediate VWAP loss), accelerate; do not let the mechanical windows slow-walk the sleeve.
3. **The 15-minute observe rule is cheap.** C executes at cross+15 and still captured 87.5% on the worst (fade-from-open) tape in the sample — observing the first 15 minutes costs little even on a fading day, and on rally days it avoids selling the very bottom of the opening range. Alvaro's discipline survives contact with the data.

---

## 5. Fade depth vs oversubscription — no usable relation

![Fade depth vs order-book oversubscription multiple](../../../data/analysis/plots/spcx_convergence/ipo_fade_vs_oversubscription.png)

*Read: x = published order-book oversubscription multiple (sourcing confidence varies; see table), y = day-1 fade depth (high→close as % of high). Green dashed = SPCX's ~4× book (2026-06-10). RIVN excluded — its multiple was never published. n=5: a scatter to eyeball, NOT a regression.*

| IPO | oversub | source/confidence | fade depth |
|---|---:|---|---:|
| Reddit 2024 | ~4.5× | Globe and Mail 2024-03-18, "4–5×" | 12.7% |
| Facebook 2012 | ~5× | institutional demand ~5× (15–20× claims exist → 5 is the conservative pick) | 15.0% |
| ARM 2023 | ~10× | Bloomberg 2023-09-11 ("10×, could reach 15×") | 4.1% |
| Alibaba 2014 | ~17× | reports range 14–22×; midpoint, LOW confidence | 5.8% |
| Cerebras 2026 | ~20× | vault sources (case-study note) | 19.4% |
| Rivian 2021 | n/a | "oversubscribed", multiple never published | (15.7%) |

The gameplan asked whether a 4× book should change the fade prior. Answer: **no signal either way.** The two *hottest* books split to the extremes (ARM 10× faded least at 4%, CBRS 20× faded most at 19%), and the two ~SPCX-sized books (RDDT, META) sit mid-range at 13–15%. If anything the sample weakly *contradicts* "bigger book ⇒ bigger fade." Keep Alvaro's −13% fade estimate as an unconditional prior — it is the sample median (4.1/5.8/12.7/15.0/15.7/19.4 → median ≈ 14%) — and do not condition it on the book.

**Day-2 follow-through (bonus row, daily data):** five of six closed day 2 *below* the day-1 close (CBRS −10.1%, META −11.0%, RDDT −8.8%, ARM −4.5%, BABA −4.3%); only RIVN bucked it (+22.1%, into the 2021 EV mania). Holding unsold residual shares overnight has been negative-carry in this sample — finish the sleeve on day 1.

---

## 6. Mapping to SPCX Friday (the gameplan deltas)

What this study changes or confirms in [[spcx_listing_day_gameplan]] §5.2, in one place:

- **Confirmed: clock re-anchored to the cross.** All windows in minutes-since-cross; the CBRS session was only ~181 minutes long (late cross → 20:00Z close). SPCX with a 17:00–20:00 CET cross faces the same compressed session — wall-clock plans are meaningless.
- **Confirmed: observe 0–15 min, then act.** Cheap on fade days, valuable on rally days (§4.3).
- **New: front-load relative to the mechanical 40/40/20.** Liquidity halves after the first 30 minutes (§2). If the PEAK signal set fires early (especially anchored-VWAP loss not reclaimed ≥10 min — it fired at +12 min on CBRS and stayed on), compress the remaining tranches rather than riding the 60/180-minute windows.
- **New: most of the pop is in the cross price.** Mean cross = 89% of the day-1 high across six IPOs (§3). The NOII indicative price during the display-only window is therefore a forecast of ~the best price of the day, not a lowball — set tranche limit anchors off it.
- **Confirmed: fade prior ~13%, unconditional.** Sample median ≈ 14%; no oversubscription conditioning (§5).
- **New: don't carry residual overnight.** 5 of 6 day-2 closes were lower (§5).

---

## 7. The S1 hook — measured melt-up distribution for `--meltup-dist`

The day-1 high-vs-offer moves across the sample (red bars in §3's chart), equal-weighted, in the calculator's flag format:

```
--meltup-dist 0.184:1,0.300:1,0.466:1,0.532:1,0.700:1,1.088:1
```

(META +18.4%, ARM +30.0%, BABA +46.6%, RIVN +53.2%, RDDT +70.0%, CBRS +108.8% — also written to `ipo_day1_meltup_dist.csv`.) E[move] = **+54.5%**, vs +26.0% under the old equal-weighted +13/26/39% Cerebras-only assumption.

**Effect on the S1 rule (verified by re-running `--decision --offline` with the flag):** the pre-hedge trigger at a $135 fill rises from Z\* ≈ $36.40 to **Z\* ≈ $47.99/share** — i.e. pre-hedging before allocation would now require the perp to spike to ~**$183** (vs ~$171 under the old dist). Every pre-node cell stays "wait"; the Friday allocation gate (hedge iff live net basis > 0) is unchanged. The measured distribution makes the pre-registered no-pre-hedge verdict *strictly stronger* — the melt-up tail a naked short faces is fatter than the Cerebras-only guess assumed.

**Unit caveat (declared, matters):** the S1 calculator applies the move to the *live perp mark*, but these moves are measured *vs the offer*. With the perp already ~+20% over offer, feeding high-vs-offer moves double-counts the premium already paid — it implies a $250 day-1 high when the measured pattern implies ~$208. That bias is **conservative in the only direction that matters** (it inflates naked-short loss → inflates Z\* → blocks pre-hedging harder), so it is acceptable for the no-pre-hedge rule. For reference, the premium-adjusted distribution at the current ~1.20 mark/offer ratio — move' = (1+move)/1.20 − 1, negatives clipped to 0 — is `0:1,0.083:1,0.222:1,0.277:1,0.417:1,0.740:1`, E ≈ +29%: nearly the old assumption's mean but with the fat +74% tail that the old dist was missing. Use the raw measured string for gating (conservative); use the premium-adjusted one only if a *symmetric* EV estimate is ever needed.

---

## 8. Assumption ledger

**Modeled assumptions (declared in this note):**
- Daily-PROXY policy values for 5 of 6 IPOs (open ≈ cross+15, OHLC/4 ≈ TWAP) — ±2–3%-of-high stand-ins, not simulations.
- The mechanical 40/40/20 windows (15–60/60–180/180+) are one fixed reading of Alvaro's plan; his signal-driven accelerations are not (and cannot honestly be) backtested here.
- Flat $/share slippage — ranking-invariant by construction; per-order or depth-scaled costs would marginally penalize the multi-order policies.
- Oversubscription multiples are press-sourced with mixed confidence (BABA LOW, META conservative pick).
- Equal weights across six IPOs spanning 2012–2026 market regimes.

**Live-only unknowns this study cannot resolve:**
- SPCX's actual cross time and price, and whether Friday's tape is CBRS-shaped (fade) or ARM-shaped (rally) — the single biggest P&L driver, undiagnosable in advance.
- TR limit-order fill quality during the first post-cross hour (Block S4 / human in-app check).
- The NOII indicative's accuracy for SPCX specifically (Block S3 sources it).
- Whether the EU retail tranche changes day-1 seller composition vs these six (all of which had little/no retail primary allocation — SPCX is structurally novel here, as the gameplan notes).

---

## 9. Decision and next step

- **Gate outcome:** Block S2 delivered. The residual-sleeve plan in [[spcx_listing_day_gameplan]] §5.2 stands with the §6 deltas above (front-load on early VWAP loss; don't hold residual overnight; fade prior 13% unconditional).
- **S1 update:** re-run the Thursday-night `--decision` with the measured `--meltup-dist` string (§7); the frozen pre-hedge trigger becomes perp ≥ ~$183 over a $135 fill. No other S1 change.
- **Not pursued, deliberately:** paid intraday data (Polygon.io etc.) to recover ARM/RIVN/BABA/META/RDDT minute tapes. It would upgrade five PROXY rows to simulations, but with the policy spread already inside the day-shape noise the decision would not change — infra-before-signal says no.
- **Next:** Blocks S3 (data-source map) and S4 (TR mechanics) remain queued in [[TODO]] § SPCX; S5 (PM-PDF monitor) is the remaining build.

## Reproduction

```bash
cd polymarket/research
PYTHONPATH=. uv run python scripts/spcx_ipo_unwind_tape_study.py          # study (offline after first run; caches in ipo_tapes/)
PYTHONPATH=. uv run python -m pytest tests/test_spcx_ipo_unwind_tape_study.py -q   # 17 tests
PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --decision \
  --meltup-dist "0.184:1,0.300:1,0.466:1,0.532:1,0.700:1,1.088:1"          # S1 rule under the measured dist
```

Outputs: 5 PNGs in `data/analysis/plots/spcx_convergence/` (`ipo_*`), 4 CSVs in `data/analysis/csv_outputs/market_maps/` (`ipo_day1_microstructure.csv`, `ipo_unwind_policy_comparison.csv`, `ipo_day1_volume_buckets_cbrs.csv`, `ipo_day1_meltup_dist.csv`) + tape caches in `ipo_tapes/`.

Oversubscription sources: [Bloomberg — ARM 10×](https://www.bloomberg.com/news/articles/2023-09-11/arm-s-ipo-orders-are-already-oversubscribed-by-10-times) · [Globe and Mail — Reddit 4–5×](https://www.theglobeandmail.com/investing/investment-ideas/article-reddits-ipo-as-much-as-five-times-oversubscribed/) · [China Daily — Alibaba oversubscribed](https://www.chinadailyasia.com/business/2014-09/10/content_15164822.html) · [Wikipedia — Facebook IPO](https://en.wikipedia.org/wiki/Initial_public_offering_of_Facebook) · Rivian: multiple never published ([Bloomberg debut coverage](https://www.bloomberg.com/news/articles/2021-11-10/rivian-set-for-debut-after-year-s-blockbuster-11-9-billion-ipo)).

---

# Block S6 — Perp exit mechanics: the pair-close window and the forced-flat backstop (added 2026-06-11)

> Code: `scripts/spcx_perp_exit_mechanics.py` (5 unit tests in `tests/test_spcx_perp_exit_mechanics.py`). The **n=1 header from the top of this note applies in full** — the tape side is one IPO (Cerebras).

## What this block answers

The gameplan's hedge-sleeve exit ([[spcx_listing_day_gameplan]] §5.1) rested on three asserted-not-measured claims: (a) post-cross perp–spot tracking "within ~$1–4" (a daily-close figure, not the minute path) and the pair-close readiness thresholds (|gap|≤$2 for 15 min, ≥60 min post-cross); (b) a 21:00 CET forced dislocated close is cheap; (c) holding a short through the cash close into `xyz` conversion is dangerous. This block measures (a)–(b) on the Cerebras tape and resolves (c) from the trade[XYZ] docs + the tape.

**Data + one method note that matters:** HL `xyz:CBRS` 15m perp (the finest HL retains for the listing window) vs Yahoo CBRS spot (1m day-1, 5m for the +1/+2 sessions), funding from HL `fundingHistory` (hourly, fully retained across the conversion; cached to `ipo_tapes/cbrs_funding_hourly.csv`). Both venues stamp candles with their *open* time; every series is shifted to close-time before comparison — without that shift, a 15m perp close gets compared against a spot print up to 14 minutes fresher, which manufactures fake $20–30 "gaps" in the fast first hour (the first run of this script did exactly that; the fix is unit-tested). Each gap point still carries a ±(intra-15m drift) uncertainty — a few dollars in the first post-cross hour, cents later.

## The gap path

![Cerebras perp vs spot and the gap, listing day + 2 sessions](../../../data/analysis/plots/spcx_convergence/cbrs_perp_spot_gap_path.png)

*Read: top = perp (black) and spot (dashed, one color per session); bottom = gap = perp − spot at each 15m mark, with the gameplan's $2 (orange) and $1 (green) readiness bands. Dotted verticals = cash closes; the x-axis spans 05-14→05-18 so overnight/weekend stretches between sessions are empty. Full series: `cbrs_perp_spot_gap_path.csv`.*

Day-1 gap at the 15m marks, in order from the 16:59Z cross: **−8.3, −6.0, +4.1, +2.0, +0.8, +0.8, −0.1, −0.6, −0.1, +0.4, +0.1, +0.8, +1.7** (dollars). The answers to §(b) of the spec, with numbers:

| question | measured answer |
|---|---|
| first |gap| ≤ $2 sustained (2 consecutive 15m marks) | **+46 min** post-cross (17:45Z) |
| first |gap| ≤ $1 sustained | **+61 min** post-cross (18:00Z) |
| end-of-session dislocation into the cash close? | **No.** Worst final-hour \|gap\|: day 1 **$1.72**, day +1 $1.12, day +2 $0.57 |
| sign of the early dislocation | perp traded *below* spot (−$8.3, −$6.0) in the first 30 min — it overshot down while spot whipped; for a short this direction is benign (buy-back cheaper than the stock you're selling) |

The asserted "$1–4 from day 1" was the right order of magnitude but hid the structure: the first ~45 minutes are genuinely dislocated (±$4–8 at 15m granularity, worse intra-bar), and after ~an hour the gap never left ±$1.7 across three sessions.

## The buy-back cost ledger

Short entry = the perp 15m close of the bar containing the cross ($376.73 at 17:00Z — the first perp reference a cross-time short realistically has). Each exit row evaluates both legs at the same 15m mark; "pair-close drag" = the gap at exit, i.e. what the pair-close gives up vs perfect tracking. Full table: `cbrs_short_buyback_ledger.csv`.

| exit | perp | spot | gap = drag $/sh | cumulative funding $/sh (short) |
|---|---:|---:|---:|---:|
| cross+1h | 331.38 | 329.38 | +2.00 | 0.00 |
| cross+2h | 324.25 | 324.85 | −0.60 | 0.00 |
| cross+3h (= cash close, the backstop) | 312.02 | 310.30 | **+1.72** | +0.005 |
| next-day open+15m | 300.35 | 300.52 | −0.17 | −0.08 |
| next-day close | 279.07 | 280.19 | −1.12 | −0.02 |
| +2d close (Mon) | 296.68 | 296.53 | +0.15 | −0.43 |

Read: on the one real tape, **the 21:00-CET-style forced flat at the cash close cost ~$1.7/share of drag** (~0.6% of a $310 stock; on SPCX's ~$25 basis that's ~7% of the lock — annoying, not ruinous), and patience past the first hour bought almost nothing: every exit from +2h onward had |drag| ≤ $1.7. The mid-session window the gameplan prescribes (1–3h post-print) is exactly where drag first sits inside ±$1. Funding is noise at this horizon: a short held cross→close paid ~zero; held to +2d, −$0.43/share.

## Funding through the conversion

![CBRS hourly funding through the IPOP→equity conversion](../../../data/analysis/plots/spcx_convergence/cbrs_funding_through_conversion.png)

*Read: hourly funding rate in bps/h; green = longs pay shorts (short earns), red = short pays. Purple line = listing/conversion (05-14, same day per the spec index). Funding accrues 24/7 including the May 16–17 weekend when cash is closed.*

The conversion visibly flipped the funding regime, exactly as the docs imply: pre-conversion the IPOP multiplier (0.005) damps rates to ~zero (mean +0.009 bps/h despite a median |premium| of 49 bps); post-conversion the standard 0.5 multiplier kicks in and rates swing ±3–8 bps/h (mean −0.116 bps/h, median |premium| 8 bps — the oracle now tracks the listed price, so the premium itself collapsed). Worst single hour for a short: ~−8 bps/h ≈ $0.25/share/h on a $300 stock — material only if held for days, which the gameplan forbids anyway.

## Conversion + postponement mechanics (doc read)

From the trade[XYZ] docs (all fetched 2026-06-11):

- **Conversion trigger** — [IPOP overview](https://docs.trade.xyz/asset-directory/pre-ipo-perpetuals-ipops.md): "Conversion occurs after the underlying asset begins regular trading and external market data is sufficient to support standard oracle pricing, **typically the first regular session after listing**." The oracle's internal 30-min EWMA is replaced by external pricing; the funding multiplier moves 0.005 → 0.5. CBRS's [spec-index](https://docs.trade.xyz/consolidated-resources/specification-index-ipops.md) entry shows "Converted (on 5/14/2026)" — same day as listing; max leverage post-conversion 5×.
- **What is documented vs discretionary** — [risk disclaimer](https://docs.trade.xyz/legal-and-disclaimers/pre-ipo-perpetual-markets-risks-and-disclaimers.md): "The timing, methodology, reference price, and source of any post-listing reference price will be described in the applicable market specifications" — i.e. **operator-defined, not protocol-fixed**. And explicitly: "the transition from internal to external pricing may produce a **step change in mark price**, which can result in **immediate P&L changes for open positions and may trigger liquidations**." Also: "Settlement or conversion may occur at a price that differs materially from the market price at which you entered a position."
- **Open positions at conversion** — **nowhere documented**: no doc addresses margin-requirement or leverage-cap changes for positions held across conversion. That silence is itself the finding: position treatment at the transition is operator discretion.
- **Postponement** — spec index: "**Up to 60 days**" grace after the anticipated listing date before alternative settlement triggers; risk doc: "If the expected listing does not occur by the applicable outside date, if no reliable public reference price becomes available, or if extraordinary circumstances affect the referenced issuer or listing event, trading in the market **may be halted and settled or closed**." So a Friday non-listing does **not** create a same-evening settlement cliff — the IPOP keeps trading (with near-zero damped funding) for up to ~60 days.
- **Funding mechanics** (context) — [funding doc](https://docs.trade.xyz/perp-mechanics/funding.md): `F = 0.5·[P + clamp(r − P, ±0.0005)]`, hourly, positive → longs pay shorts, cap ±4%/h; [oracle doc](https://docs.trade.xyz/perp-mechanics/oracle-price.md): "When external markets are open, the externally-derived fair price is transmitted as the oracle price," with the 30-min EWMA as the no-external-input fallback (mark price samples the basis on a 150s constant).

What the CBRS tape shows at the transition moment itself: the perp's 17:00Z close was $376.73 against first cash prints of $350–385 — chaotic but *tracking the same chaos*, with no visible discontinuous mark jump at 15m granularity (the −$8/+$4 first-hour gaps are tape noise, not an oracle step). The docs' step-change warning did not materialize on this n=1, but it is the documented risk, and at ≤1.5× the S1 liquidation buffer (sized for a +39% spike) covers any step this tape exhibited many times over.

## Verdict (the four questions, in order)

1. **Pair-close readiness thresholds — CONFIRMED, with a sharpening.** |gap|≤$2-for-15-min first fired at **+46 min** post-cross; the ≥60-min rule lands almost exactly on the $1-sustained moment (**+61 min**). Keep both thresholds as written; treat the first ~45 minutes as a genuine no-pair-close zone (±$4–8 dislocations at 15m marks, sign favorable to shorts but unstable). The §5.1 "1–3h post-print" window is where measured drag sits inside ±$1.
2. **Cost of the 21:00 forced-flat backstop — measured ~$1.7/share** on the analog (no end-of-session dislocation; worst final-hour |gap| $1.72). The backstop is cheap insurance: ~0.6% of price, ~7% of an SPCX-sized basis. Keep it.
3. **Short through the close into conversion — tolerable on the n=1 tape, still don't choose it.** The tape says an accidental overnight hold was benign: the gap held inside ±$1.2 into the next open and funding cost pennies. But the docs make conversion-window position treatment operator-discretionary, explicitly warn of mark step-changes that "may trigger liquidations," and document nothing about margin/leverage treatment across the transition. With zero offsetting benefit (the pair earns nothing extra overnight), the rule stands: **flat by the close, by choice; if the backstop is somehow missed, the precedent says next-morning recovery is likely cheap rather than catastrophic — at ≤1.5× only.** (For SPCX note the conversion may be "the first regular session *after* listing" rather than listing day — CBRS converted same-day; either way the same-day pair-close avoids the question entirely.)
4. **Postponement rule — CORRECTED in urgency, confirmed in substance.** There is no same-evening settlement cliff: the IPOP survives up to ~60 days past the anticipated date with near-zero damped funding. So "no cross by 20:30 → close the perp **at market immediately**" overstates the urgency — a patient limit close within the hour is fine, and the real reasons to close stand (no same-day convergence event, weekend operator-discretion exposure, margin freed). The shares-only fallback is unchanged.

## Reproduction (S6)

```bash
cd polymarket/research
PYTHONPATH=. uv run python scripts/spcx_perp_exit_mechanics.py            # offline after first run
PYTHONPATH=. uv run python -m pytest tests/test_spcx_perp_exit_mechanics.py -q   # 5 tests
```

Outputs: `cbrs_perp_spot_gap_path.png`, `cbrs_funding_through_conversion.png` (plots/spcx_convergence/), `cbrs_perp_spot_gap_path.csv`, `cbrs_short_buyback_ledger.csv` (csv_outputs/market_maps/), plus caches `ipo_tapes/cbrs_spot_5m_postlisting.csv` and `ipo_tapes/cbrs_funding_hourly.csv`.

> Not investment advice — calibration for a personal-size position on tiny n.
