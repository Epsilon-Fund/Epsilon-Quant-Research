---
title: "SpaceX IPO crowd-PDF: the multi-peak shape is an interpolation artifact (methodology audit)"
tags: [spacex, ipo, polymarket, pdf, pchip, methodology, data-quality, audit]
created: 2026-06-08
status: "audit complete; recommends a method swap before the June 11/12 refresh"
audience: "Cowork/Codex/Justin stress-testing the colleague's SpaceX PDF construction"
---

# SpaceX IPO crowd-PDF: the multi-peak shape is an interpolation artifact (methodology audit)

> Hub: [[POLYMARKET_BRAIN]] · [[COWORK]]
> Companions: [[spacex_ipo_market_map_handoff]] · [[spacex_ipo_coworker_addendum]]
> Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

- This is a **stress test of a colleague's PDF-construction method**, not a strategy note. He sent three files (`spacex_pdf_analysis.py`, `ev_report.py`, `REPLICATION_GUIDE.md`) that turn Polymarket's "SpaceX IPO closing market cap above ___?" 16-strike ladder into a probability distribution of the day-1 closing share price. The shipped chart shows what looks like **2–3 humps**. Justin suspected the humps were unrealistic.
- **They are.** The multi-peak shape is a pure numerical **artifact of differentiating a cubic interpolant through coarse, kinky survivor points** — not a real multi-modal crowd belief. The raw, assumption-free probability mass (`S[strike_i] − S[strike_{i+1}]`) is a **single broad hump** peaking at 2.0–2.2T market cap, and the *independent* 7-bucket market is also single-humped. Verified four independent ways and adversarially stress-tested by a 4-agent workflow (V1/V3/V4 confirmed, V2's "it's really bimodal" steelman refuted).
- **Liquidity is not used at all, and the price basis matters.** The method interpolates *exactly* through every strike's **mid**-price with equal weight; no spread/depth weighting exists. We re-derive the distribution with a **1/spread²-weighted (liquidity-weighted) lognormal** fitted to the **best-ask** survivor — the taker convention, since to buy the "cap above K" YES you pay the ask. The repo has **no order-book capture**, so true depth-weighting isn't possible offline — but the fix doesn't need it. Best-ask vs mid is **minimal** (P(win) +0.4pp, mean +$0.6, ≈ half the spread); the liquidity-weighting choice moves things a touch more (~1.4pp) but neither changes the conclusion.
- **Central stats are robust; shape and tail stats are not.** P(close > $135) ≈ **80%**, mean ≈ **$168**, median ≈ **$164**, EV and Kelly are stable across every construction (PCHIP, lognormal, skew-normal, raw linear; mid or best-ask). But **mode ($161→$155), std ($41→$39), skewness (+0.87→+0.69), and especially excess kurtosis (+3.45→+0.78)** are materially distorted by the spurious peaks, and the **tail percentiles are the least trustworthy numbers in his report** (P1 $72→$96, P99 $308→$279). Full side-by-side in [§ Key metrics](#finding-3--liquidity-weighting-and-the-corrected-key-metrics-confirmed-high-confidence).
- **Bucket overlay (liquidity-weighted) is a clean consistency check:** the ladder-implied and bucket-market distributions agree within each bucket's own bid-ask spread band — liquidity-weighted RMS gap **1.66pp** — except the 1.5–2.0T bucket, which is **structural and already understood (not pursued here)**, reported only for completeness.
- **Recommended construction:** stop differentiating the PCHIP interpolant for the published PDF. Ship a **1/spread²-weighted lognormal** (single-hump by construction) with the **raw interval-mass histogram** beside it. A drop-in builder that reproduces the full metric set his report had is provided: [`spacex_pdf_builder_v2.py`](../../../scripts/spacex_pdf_builder_v2.py).

This note is a methodology/data-quality audit of the PDF construction only — no strategy content.

## Why this audit exists

The colleague's `REPLICATION_GUIDE.md` correctly frames the math as the prediction-market analog of Breeden–Litzenberger (1978): for a digital "cap above K" contract, `price = S(K) = P(cap > K)`, so the density is `f(K) = −dS/dK`. That part is right. The failure is in **how the continuous `S` is reconstructed from 16 discrete, noisy quotes and then differentiated.** This note isolates exactly where the shape goes wrong, whether it matters, and what to ship instead. It uses only data already in the vault (the [[spacex_ipo_market_map_handoff]] snapshot + the arrays embedded in the scripts) — **no live CLOB capture was needed**, consistent with Justin's expectation.

## The method under audit (and a worked example)

The colleague's pipeline:

1. For each of 16 strikes K ∈ {1.0T … 4.0T, 0.2T apart}, take the mid-price `mid = (YES_ask + (100 − NO_ask)) / 2` as `S(K)`.
2. Add boundary anchors `S(0)=1.0`, `S(4.5T)=0.005`, `S(5.0T)=0.001`, plus a `if s[i] >= s[i-1]: s[i] = s[i-1] − 0.0001` "monotonicity hack."
3. Fit `scipy.interpolate.PchipInterpolator` (Fritsch–Carlson monotone cubic) through all those points.
4. Differentiate: `pdf = −spline.derivative()`, then normalize and convert the cap axis to $/share via 13,091M shares.

**Worked example of the failure mode.** Look at the survivor curve between $2.4T and $2.8T. The crowd prices `S(2.4T)=29.5%`, `S(2.6T)=14.5%`, `S(2.8T)=10.5%`. So the slope is **steep** across [2.4, 2.6] (drops 15pp) and then **abruptly flat** across [2.6, 2.8] (drops only 4pp). PCHIP's "no-overshoot" rule sets the derivative *at the shared 2.6T knot* to a weighted harmonic mean of the two neighboring slopes — which gets pulled toward the *flat* side (the PDF at the 2.6T knot collapses to 0.32). But the area (probability mass) inside [2.4, 2.6] still has to equal the full 15pp. The only way a cubic can start high at 2.4, end low at 2.6, and still enclose 15pp is to **bulge upward in the middle** — manufacturing a spurious local peak at ≈$188/share. The same mechanism off the next kink and the noisy deep tail produces the other bumps.

## Finding 1 — the multimodality is an interpolation/differentiation artifact (CONFIRMED, high confidence)

The PDF was rebuilt four independent ways from the raw mids (no colleague code), counting local maxima in price space each time.

| Build variant | local maxima | notes |
|---|---:|---|
| (a) PCHIP + boundary anchors + 0.0001 hack (as shipped) | **6** | peaks at $132, $161, $188, $219, $232, **$298** |
| (b) PCHIP, **no** boundary anchors (16 strikes only) | **5** | $132, $161, $188, $219, $232 |
| (c) PCHIP, **no** 0.0001 hack | **6** | identical to (a) — the hack never fires |
| (d) plain CubicSpline (not-a-knot) | **5** in-support | also goes **negative** (min density −0.019) |
| Raw interval-mass histogram `S[i]−S[i+1]` (no interpolation) | **1** | single broad hump, peak 2.0–2.2T |

**Column meaning.** "local maxima" = number of distinct bumps in the price-space density tall enough to survive a 2%-of-peak prominence filter (stable across grid resolutions n=400…80,000). The histogram row is the assumption-minimal density: it makes *no* interpolation choice, just bins the probability mass the market directly quotes.

**Mechanism, verified exactly.** On [2.4, 2.6]T: secant slopes −0.80 then −0.75/T (steep), then −0.20/T (flat) at [2.6, 2.8]. The Fritsch–Carlson knot derivative `S'(2.6) = −0.3158` matches the weighted-harmonic-mean prediction to 1e-3, and the density *inside* (2.4, 2.6) bulges to **0.895 — above both endpoints (0.774 and 0.316) and above the secant (0.75)**: the textbook no-overshoot mid-interval bulge. All five core maxima sit **mid-interval, not at strikes** (cap offsets −0.066, −0.088, +0.063, +0.067, +0.041T from the nearest strike) — the overshoot signature.

**Read.** The humps are an artifact of *exact interpolation through a kinky survivor, then differentiation* — they are not crowd beliefs. Two honest refinements over the first-pass diagnosis: (1) the shipped chart actually has **6** maxima, not 5 — the extra one at ~$298 is created purely by the boundary anchors; (2) the artifact is **not unique to PCHIP** — a plain cubic spline shows the same bumps (and illegally goes negative), and even a near-interpolating monotone smoothing spline keeps them. The bumps reflect a genuine **data kink at the 2.4–2.8T transition**; only a *parametric unimodal* fit removes them by construction. The `0.0001` hack is inert and should simply be deleted (it implies it matters when it doesn't).

![[spacex_pdf_artifact_audit.png]]

*Figure. (A) The shipped PCHIP density with its spurious local maxima marked in red — the "3 humps" are really 5–6 bumps. (B) What the data actually says: the grey bars are the raw interval-mass histogram (one broad hump), and the red curve is the liquidity-weighted unimodal lognormal fit. (C) Cross-market overlay — threshold-ladder-implied bucket probabilities (blue) vs the standalone bucket market (orange); only $1.5–2.0T diverges materially (+7.2pp, in red). (D) Per-strike bid-ask spread ÷ mid: the deep-tail strikes are the *relatively* least reliable, though absolute quoted spreads are implausibly tight everywhere. Axes: x = day-1 closing $/share (A,B) or cap bucket (C) or strike (D); y = probability density (A,B), probability % (C), spread ratio (D). Data: Polymarket 2026-06-07 mids, 13.091B shares.*

## Finding 2 — a genuine bimodal belief is refuted (REFUTED, high confidence)

A "$135 weak-debut mode vs $185 strong-pop mode" story was steelmanned and tested. A real second mode requires probability mass to *increase again* somewhere as you move right. It doesn't:

- **Ladder masses** rise monotonically to the 2.0–2.2T peak (18.0pp) then fall monotonically — the only post-peak uptick is a +0.2pp wobble at 3.8–4.0T (1.1% of peak height), pure deep-tail quote noise. The supposed "$185 pop mode" maps to 2.42T, which sits in the **strictly decreasing** region.
- The **independent 7-bucket market** is also strictly single-humped (peak 2.0–2.5T, no dip-then-rise).

**Read.** Two structurally separate markets both force unimodality. There is no data support for multiple modes from any source.

## Finding 3 — liquidity weighting and the corrected key metrics (CONFIRMED, high confidence)

`PchipInterpolator(x, y)` takes only knot coordinates — **no weights, no spread, no depth.** Every strike is honored exactly with equal weight, including the thin deep-tail quotes. We re-derive the distribution two ways: (1) put liquidity back in by fitting a single-mode **lognormal survivor weighted by `1/spread²`** (tighter quotes count more), and (2) build it on the **best-ask** quotes rather than the mid — the **taker convention**, since to buy the "cap above K" YES contract you pay the ask. Fit: `μ ≈ 0.76, σ ≈ 0.23` in cap space (median cap ≈ 2.14T), weighted RMSE ≈ 1.1¢, unimodal by construction. Honest caveat on the liquidity proxy: by *quoted* half-spread every strike has SNR ≥ 4.3 (spreads are 0.1–1.0¢), so the weighting barely re-ranks them — those quoted spreads are **implausibly tight** and almost certainly a thin/stale snapshot, so they understate true uncertainty. True depth-weighting would need a live CLOB book capture the repo does not have. The $2.6T ($199/share) strike is the one persistent local-fit tension (the fit wants ~19¢ there vs the quoted 14.5¢) — worth a manual re-quote check at refresh.

The table below reproduces **every metric his report printed**, his PCHIP-on-mid construction vs the recommended liquidity-weighted lognormal on best-ask, at a **fixed 13.091B share count** so only the construction differs.

| metric | his PCHIP · mid | recommended lognormal · best-ask | what moved |
|---|---:|---:|---|
| P(close > $135) | 79.9% | 80.0% | stable |
| P(close < $135) | 20.1% | 20.0% | stable |
| Mean close | $166.9 | $168.1 | stable |
| Median close | $164.2 | $163.8 | stable |
| **Mode close** | **$161.3** | **$155.4** | distorted (mode sat on an artifact bump) |
| **Std dev** | **$40.7** | **$39.0** | distorted (bumps fatten it) |
| **Skewness** | **+0.870** | **+0.691** | distorted |
| **Excess kurtosis** | **+3.445** | **+0.784** | **badly distorted — "fat tails" are mostly artifact** |
| Avg gain if win | +$44.3 | +$45.2 | stable |
| Avg loss if loss | $17.5 | $15.6 | stable |
| Expected value | +$31.9 (+23.6%) | +$33.1 (+24.5%) | stable |
| Kelly fraction | 72% | 73% | stable |
| P1 | **$72** | **$96** | distorted (anchor-driven left tail) |
| P5 | $113 | $112 | stable |
| P10 | $124 | $122 | stable |
| P25 | $140 | $140 | stable |
| P50 (median) | $164 | $164 | stable |
| P75 | $187 | $191 | stable |
| P90 | $215 | $220 | ~stable |
| P95 | $235 | $239 | ~stable |
| P99 | **$308** | **$279** | distorted (artifact-inflated right tail) |

**Column meaning.** "his PCHIP · mid" = his exact pipeline (mid-prices, anchors + the inert monotonicity hack), density via `−dS/dx`. "recommended lognormal · best-ask" = the single-mode `1/spread²`-weighted lognormal fitted to the **best-ask** survivor. Both use 13.091B shares; the production builder uses the correct 13.076B, which scales every dollar figure by 0.9988 (≈ −$0.1–0.4, immaterial). "Mode" = peak of the density; "skewness/excess kurtosis" = standardized 3rd/4th moments (0 = normal); percentiles Px = the day-1 closing price below which x% of mass sits.

**Read.** The **central, decision-relevant numbers are construction-invariant** — P(win) ≈ 80%, mean ≈ $168, median ≈ $164, EV and Kelly all agree to within rounding. What his construction gets **wrong** is the *shape*: excess kurtosis is inflated ~4× (+3.45 vs +0.78), the mode is pulled onto a spurious bump, and the extreme percentiles (P1 "extreme crash" $72, P99 "extreme rip" $308) are artifact-driven — the unimodal fit puts them at $96 and $279. Anyone quoting his tail percentiles or "fat-tailed" language is quoting the artifact.

### Best-ask vs mid, and whether liquidity weighting matters

Switching the construction standard from mid to best-ask (the taker price) is **minimal**, because best-ask sits only ≈0.33pp above mid per strike (= half the average spread):

| construction (liquidity-weighted lognormal) | P(close > $135) | mean | median | std |
|---|---:|---:|---:|---:|
| mid | 79.7% | $167.5 | $163.2 | $38.7 |
| **best-ask (taker, recommended)** | **80.0%** | **$168.1** | **$163.8** | **$39.0** |
| Δ (ask − mid) | +0.4pp | +$0.6 | +$0.6 | +$0.3 |

Does the liquidity-weighting *assumption* still matter under best-ask? A little — and slightly more than the mid→ask switch: best-ask + `1/spread²` vs best-ask + equal-weight moves P(win) by ~1.4pp (80.0% → 81.5%) and std by ~$2.5. So liquidity weighting is the larger of the two small effects, but **neither changes the conclusion** (P(win) stays ~80%, mean ~$168, distribution stays single-humped). The skew-normal best-ask sensitivity agrees (P(win) 81.3%, mean $168.2), so right-skew is not load-bearing either.

## Finding 4 — liquidity-weighted bucket overlay (consistency check)

As a final construction check, overlay the threshold-ladder-implied bucket probabilities against the *standalone* 7-bucket "cap between x and y" market, weighting the comparison by each bucket's own bid-ask spread (its liquidity). The ladder side is model-free (`S(lo) − S(hi)` from the mids); the bucket side is the normalized mid with its spread shown as a tolerance band.

| cap bucket | ladder-implied | bucket market | bucket spread (±pp) | gap (bkt − ladder) |
|---|---:|---:|---:|---:|
| < 1.0T | 1.1% | 0.6% | 0.1 | −0.5pp |
| 1.0–1.5T | 5.0% | 3.4% | 0.9 | −1.6pp |
| 1.5–2.0T | 30.4% | 38.2% | 2.0 | +7.8pp |
| 2.0–2.5T | 41.5% | 41.7% | 1.0 | +0.2pp |
| 2.5–3.0T | 15.5% | 11.7% | 1.5 | −3.8pp |
| 3.0–3.5T | 4.3% | 3.3% | 1.4 | −1.0pp |
| 3.5T+ | 2.2% | 1.0% | 0.3 | −1.1pp |

Liquidity-weighted RMS gap (weights `1/spread²`): **1.66pp**.

**Column meaning.** "ladder-implied" = `S(lo) − S(hi)` from the threshold mids (no interpolation needed except a small piece at the 1.5T edge). "bucket market" = normalized mid of the standalone bucket contract (No-IPO dropped, renormalized to IPO-conditional mass). "bucket spread" = that bucket's bid-ask in pp, used both as a tolerance band and as the `1/spread²` weight. "gap" = how much richer (+) the bucket market prices that range than the ladder.

**Read.** The two independently-quoted markets are **broadly consistent** as constructions: the liquidity-weighted RMS gap is only **1.66pp**, and every bucket except 1.5–2.0T sits within a few pp of the ladder (the next-widest, 2.5–3.0T at −3.8pp, has a 1.5pp spread band). The **1.5–2.0T bucket is +7.8pp richer — that gap is structural and already separately understood; it is not pursued here and carries no edge.** It is listed only so the overlay is complete. Worth noting it also has the **widest spread in the book (2pp)**, i.e. the thinnest/least-reliable quote — consistent with a construction/quote effect rather than information.

## Mirror of the colleague's report (corrected construction)

For a like-for-like comparison with the 2-page report he sent, here is his front-page distribution chart and metric/percentile boxes rebuilt with the recommended construction (**best-ask, liquidity-weighted lognormal, 13.076B shares**). The shape is now a single hump and the tails are honest; the headline (P(close > $135) = 80%, mean $168) is unchanged. Regenerate with [`spacex_corrected_report.py`](../../../scripts/spacex_corrected_report.py).

![[spacex_corrected_report.png]]

*Figure. Corrected mirror of the colleague's `ev_report` front page. Top: the day-1 closing-price distribution (single mode, shaded profit/loss zones, IPO $135 green line / perp $166 orange / mean red / median purple, P(close > $135) = 80% box). Bottom-left: KEY TRADE METRICS. Bottom-right: DISTRIBUTION PERCENTILES with a corrected-vs-colleague column. Construction: best-ask liquidity-weighted lognormal, 13.076B shares.*

### Distribution percentiles — his labels, corrected numbers

His report tagged each percentile with a scenario label ("extreme crash" … "extreme rip"). Same labels, corrected construction:

| percentile | scenario label (his) | his report | corrected (best-ask) |
|---|---|---:|---:|
| P1 | extreme crash | $72 | $96 |
| P5 | bad day | $113 | $112 |
| P10 | weak open | $124 | $122 |
| P25 | below expectations | $140 | $140 |
| P50 | median / base case | $164 | $164 |
| P75 | good pop | $187 | $191 |
| P90 | strong pop | $215 | $220 |
| P95 | euphoric | $235 | $239 |
| P99 | extreme rip | $308 | $279 |

**Read.** The body (P5–P95) barely moves. The extremes tighten: his "extreme crash $72" and "extreme rip $308" were inflated by the PCHIP + boundary-anchor artifact; the honest construction puts the 1st/99th percentiles at **$96 / $279**.

## What the construction distorts (and what it doesn't)

The central, decision-relevant statistics are **robust** to the construction method — P(close > $135) ≈ 80%, mean ≈ $168, median ≈ $164, EV and Kelly all agree (see § Finding 3). The construction artifact corrupts the **shape and tails**, so the discipline is:

- Do **not** present the distribution as multi-modal or read "scenario modes" ($132 / $161 / $188) off the curve — those are interpolation bumps.
- Do **not** quote the **excess kurtosis (+3.4)** or call it "fat-tailed" — the unimodal value is +0.8; the kurtosis is ~4× inflated by the spurious peaks.
- Do **not** trust the extreme percentiles (**P1 $72, P99 $308**) — they are artifact-driven; the unimodal fit puts them at **$96 / $279**. The middle percentiles (P5–P95) are fine.
- Do **not** read the **mode** ($161) as meaningful — it is sitting on a bump; the true mode is ≈$155.

## Recommended construction + housekeeping

1. **Swap the construction.** For the published PDF, replace "fit PCHIP through 16 mids, then differentiate" with the **1/spread²-weighted lognormal on the best-ask survivor** (single-hump by construction, taker price basis) and show the **raw interval-mass histogram** beside it. Report the skew-normal as an agreeing right-skew sensitivity. Drop-in builder that emits the full metric set above: [`spacex_pdf_builder_v2.py`](../../../scripts/spacex_pdf_builder_v2.py); the his-style report figure regenerates from [`spacex_corrected_report.py`](../../../scripts/spacex_corrected_report.py) and the 4-panel diagnostic from [`spacex_pdf_audit_chart.py`](../../../scripts/spacex_pdf_audit_chart.py).
2. **Delete the inert `0.0001` monotonicity hack** — the survivor mids are already strictly monotone, so it never fires; keeping it implies it matters.
3. **Drop the boundary anchors** (or document them) — they add the spurious $298 tail bump and inflate the right-tail percentiles.
4. **Reconcile the `−$3.3/share` IPO EV line** in the PNG (already flagged in [[spacex_ipo_coworker_addendum]]) against the +$32/share mean-minus-entry arithmetic.
5. **Standardize on the exact S-1/A share count 13,075,865,175 (≈13.076B)**, not 13.091B, and store the convention beside every valuation (≈0.1% here, but it compounds across venues per [[spacex_ipo_market_map_handoff]]).
6. **Refresh the 16 strikes + 7 buckets June 11 evening / June 12 morning** before re-running; re-check the $2.6T strike, which is the one persistent local-fit tension.

## Column glossary / definitions

- **Survivor function `S(K)`** — P(day-1 closing market cap > K). Decreasing in K. Equals the fair value of a "cap above K" YES contract.
- **PDF `f(K) = −dS/dK`** — probability density of the closing cap; the colleague gets it by differentiating the fitted `S`.
- **PCHIP / Fritsch–Carlson** — monotone piecewise-cubic interpolant; preserves monotonicity of `S` but its derivative (the PDF) is only piecewise-quadratic and bulges to avoid overshoot, which is what manufactures the humps.
- **Interval-mass histogram** — `S[strike_i] − S[strike_{i+1}]`, the probability the market directly assigns to each strike gap; the assumption-free density, here unimodal.
- **mid / spread** — `mid = (YES_ask + (100 − NO_ask))/2`; `spread = YES_ask − YES_bid`. Used as the (only available) liquidity proxy.
- **bucket vs ladder** — "ladder" = the 16-strike "cap *above* K" threshold market; "bucket" = the separate "cap *between* x and y" market. See [[spacex_ipo_market_map_handoff]] for both snapshots.
