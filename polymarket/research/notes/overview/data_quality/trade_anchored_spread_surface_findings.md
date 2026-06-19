---
title: "Trade-Anchored Effective-Spread Surface (SPREAD-1)"
created: 2026-06-11
status: closed
owner: justin
project: polymarket
para: project
hubs:
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - research
  - data-quality
  - copytrade
  - spread
---
# A state-dependent spread-cost model from the public trade tape (Block SPREAD-1) — gate FAIL-with-diagnosis, level model survives for slow categories

> Hub: [[POLYMARKET_BRAIN]] · [[COWORK]]
> **Successor gate (2026-06-11): [[spread_surface_tradetime_regate_findings]] — SPREAD-1b ran the §7.3 trade-time re-gate on the frozen surface and PASSED all three pre-registered bars; the surface is now a validated trade-time cost prior (politics_negrisk excepted, time-averaged 2× caveat unchanged).**
> Table terms: [[polymarket_table_dictionary]] (SPREAD-1 terms added there in this pass)
> Code: `lib/spread_surface.py` (loader + `predict()` API) · `scripts/spread_surface_build.py` (all build/validation steps) · `tests/test_spread_surface.py` (16 tests)

## Plain-English Summary

- **What this is:** every spread number we use outside captured live_clob windows today is a flat assumption (the 2–3c constants in `data_infra/weather_analysis.py`) or a next-fill proxy ([[phase5_design]] §3.5). This block built a state-dependent estimate of the half-spread a taker pays, from data we have for every market and day: OrderFilled fills joined to the CLOB `/prices-history` midpoint, signed by the tape's own aggressor flag.
- **Premise check passed:** `/prices-history` `p` IS a book midpoint — median |p − replayed L1 mid| = 0.0c on all 9 tokens tested across 3 captures and 3 dates; deviation from last-trade is 10–100× larger.
- **Pre-registered validation gate: FAIL.** On 411 market-cells from 6 live captures, median absolute error = 1.000c (exactly at the ≤ 1c bar — passes, with zero margin) but Spearman rank correlation = 0.496 < 0.6.
- **Diagnosis (measured, not hand-waved):** the surface estimates **trade-time** execution cost, while the gate's target was the **time-averaged** quoted spread — and in fast crypto markets these are different quantities: trades execute when books are momentarily tight (measured on 40 crypto-roll tokens: 2.29c quoted half-spread at trade times vs 4.00c time-averaged). Add 1-min mid staleness (24.6% negative estimates overall, up to 55% in fast cells, vs ~0.4% in politics) and heavily tied binned predictions, and the rank bar is structurally out of reach for this design.
- **What survives:** for slow categories (geopolitics, sports, tech, residual other) the surface is a level-accurate trade-time cost prior — slow-pooled MedAE 0.75c; per category sports 0.0c, geopolitics 0.5c, other 0.9c, tech 1.0c (politics_negrisk got only 2 validation cells — no power) — corroborated by tape-only cross-checks (bid-ask bounce, Roll). Verdict: **usable as a level prior for slow categories; do NOT use for fast crypto or as a time-averaged quoted-spread predictor.** A trade-time-targeted re-validation would be a NEW pre-registered gate, not a reinterpretation of this one.

## 1. The idea and the one-line estimator

Under a rational-user model, a taker fill executes at the L1 touch. So the signed gap between a fill price and the prevailing midpoint *is* the half-spread paid:

```
half_spread_est = s_dir · (P_fill − mid_before_fill)
s_dir = +1 if the taker bought, −1 if the taker sold
```

The aggressor sign comes free from the tape: `maker_side` is the passive maker's token side, so the taker did the opposite (`s_dir = +1` iff `maker_side = 'SELL'`; see `lib/trade_sign_normalization.py`).

**Practical example.** A taker buys YES at 0.62 while the mid (last `/prices-history` bar strictly before the fill) is 0.60 → half_spread_est = +2c: the taker paid 2 cents over fair to cross. A taker sell at 0.58 against the same mid is also +2c (paid on the other side). A taker buy at 0.59 — *below* mid — gives −1c: under the rational-user model this "shouldn't happen" (you can't buy below mid at the touch), so negative estimates measure where the model's assumptions fail (stale mid, or fills that aren't simple touch-crossings). Negatives are kept, not filtered, and reported as a diagnostic.

**Dedup rule (one row per atomic fill).** Polymarket's `_matchOrders` path re-emits the whole aggressive order as an extra "internal leg" row (taker = one of the 4 exchange contracts), on which `maker_side` is the *aggressor's own side* — sign-inverted relative to normal rows ([[copytrade_attribution_repartition_findings]]). The estimator uses **only non-internal rows** (`taker` not in the 4 contracts): same-token internal legs are duplicates of their sibling maker legs, and cross-token (mint/merge) legs are mirrored by complementary-token siblings carrying the identical half-spread content (a sibling maker-buy of NO at 1−p vs NO-mid ≈ 1−YES-mid reproduces p − mid_YES exactly).

## 2. Pre-registered design and what happened at each step

| step | pre-registered | outcome |
|---|---|---|
| 1. Semantics check | confirm `/prices-history` `p` is a midpoint vs `lib/clob_book.py` mid() on 3+ captured markets; STOP if not | **PASS** — median abs deviation 0.0c on 9/9 tokens (3 captures: 2026-05-28, 05-29, 06-10); means 0.003–0.63c; vs last-trade 0.5–7c. Full-book-replay mid and `best_bid_ask`-event mid agree, so the cheap L1-event path is valid |
| 2. Estimator | dedup per atomic fill; join to ffilled mid strictly before fill | 99,473 usable per-fill estimates from a stratified sample (336 markets × 7 categories, 2026-03-01→05-26); only 44 fills had no prior mid bar and 10 had a bar older than 30 min |
| 3. Surface | median + IQR per price × TTR × activity × category cell; tick floor 0.5c; no ML | 460 full cells, 263 with n ≥ 50; 4-level fallback chain (drop TTR → drop activity → price-only) in `lib/spread_surface.py` |
| 4. Validation gate | MedAE ≤ 1c AND Spearman ≥ 0.6 on market-cells vs true live_clob L1 | **FAIL** — MedAE 1.000c (at the bar), Spearman 0.496 on 411 market-cells |
| 5. Diagnostics | negative-rate; size split; bounce + Roll cross-checks | all run — see §5; they localize the failure precisely |

Sampling detail (step 2): per category, markets stratified into fill-count quartiles (12 markets per quartile), up to 4 random fill-days per market, capped at 1,500 anchor fills per market-day; per (token, day), one cached `/prices-history` call at 1-min fidelity with a 1-hour pre-roll (1,126 calls, all non-empty, all cached under `data/analysis/spread_surface/mid_history/`). The trades tape ends 2026-05-26 (Goldsky's subgraph tail is currently *behind* our local tape, so it cannot be extended); the build window is March–May 2026.

Features per fill: price bucket from the joined mid (`p_lt_05` … `p_gt_95`, edges .05/.15/.35/.65/.85/.95); time-to-resolution bucket from the markets snapshot `end_date` (`ttr_lt_6h`/`ttr_6_24h`/`ttr_1_7d`/`ttr_7_30d`/`ttr_gt_30d`); trailing 60-min activity = distinct `transaction_hash` count on the market, quartiled per category; category from the Block K5 slug/question rules (`crypto_4h`, `daily_crypto`, `geopolitics`, `sports`, `politics_negrisk`, `tech`, `other`), applied as the same SQL CASE on both the tape side and the capture side.

## 3. The surface (what the tape says a taker pays)

![Median trade-anchored half-spread by price-level bucket and trailing activity quartile, pooled over categories and time-to-resolution](../../../data/analysis/plots/copytrade/spread_surface_v1_heatmap_price_activity.png)

**How to read it:** each cell is the median per-fill half-spread estimate (cents; price units are dollars 0–1, so 1c = 0.01) for fills landing in that price-level row and trailing-activity column; n is the fill count. Color: brighter = more expensive to cross. **Notice:** extreme-price books (`p_lt_05`, `p_gt_95`) are tick-tight everywhere (0.05–0.3c) — longshot/near-certain tokens cost almost nothing to cross at L1, consistent with penny books. Costs peak in the 5–35c price range (2–3.5c). At mid-to-high prices, busier markets are *cheaper* (Q4 0.5c vs Q1 1.5c at `p_65_85`) — more activity → tighter trade-time books; at `p_05_15` the gradient flips, which is contaminated by fast daily-crypto markets spending their final hours there (see §5).

Category × price overview (level-2 aggregates; full 4-dimension surface + IQRs in the CSVs):

| category | cheapest buckets | mid-range (.35–.65) | most expensive | read |
|---|---|---|---|---|
| politics_negrisk | 0.05c at both tails | 0.5c | 2.5c (.15–.35) | tight NegRisk books; negatives ~0–2% — the rational-user model fits politics almost perfectly |
| sports | 0.05c (>.95) | 0.5c | 1.5c | tight; negatives ~20% mid-range |
| geopolitics | 0.15–0.3c tails | 1.0c | 1.5c | well-behaved |
| tech | 0.15–0.2c tails | 1.5c | 1.5c | well-behaved, negatives 3–14% |
| other | 0.05–0.35c tails | 2.5c | 4.5c (.05–.15) | widest slow category |
| crypto_4h | 0.45–0.5c tails | 2.0c | 2.5c (.15–.35) | negatives 8–30% — borderline |
| daily_crypto | 0.45–0.85c tails | 1.5c | 5.5c (.15–.35) | **medians go negative at .65–.95 (−1.5c, −0.5c; 54–55% negative) — model failure cells, see §5** |

The `predict()` API never returns a negative number (0.5c tick floor, pre-registered), and the surface CSV carries `frac_negative` per cell so consumers can refuse contaminated cells.

## 4. Validation gate — FAIL, and exactly how it fails

Setup: on 6 capture runs (2026-05-27 → 06-10: block_a0 morning, a0b replacements-v2, a0c targeted, a0c 24h crypto roll, two mm_stage1 broad-live days), every market × 30-min window with ≥ 3 L1 events and a full trailing hour got: true half-spread = median (ask−bid)/2 over the window's `best_bid_ask` events; predicted = `SpreadSurface.predict(price, TTR, trailing rate, category)` with features computed at window start (trailing rate from distinct WS `transaction_hash`, same definition as the tape side). Windows aggregate to market-cells (median true and predicted per market × cell). 2,744 windows → 411 market-cells.

| slice | n cells | MedAE (gate ≤ 1c) | Spearman (gate ≥ 0.6) |
|---|---|---|---|
| **pooled (official gate)** | 411 | **1.000c — at the bar** | **0.496 — FAIL** |
| slow categories only | 217 | 0.75c | 0.457 |
| fast crypto only | 194 | 1.45c | 0.498 |

Per category: other 0.90c / 0.62 (passes alone), geopolitics 0.50c / 0.28, tech 1.00c / −0.25, sports 0.00c / 0.10, daily_crypto 0.47c / 0.44, crypto_4h 1.50c / 0.46, politics_negrisk 2.29c / n=2 (no power). Per capture, MedAE ranges 0.00c (a0b) to 1.50c (crypto roll).

![Predicted vs true half-spread per market-cell, colored by category, with the ±1c gate band](../../../data/analysis/plots/copytrade/spread_surface_v1_pred_vs_true.png)

**How to read it:** each point is one market-cell; x = true quoted half-spread (median over capture windows), y = surface prediction; the grey band is the ±1c MedAE gate. **Notice:** the mass of slow-category points (green/red/pink/purple) hugs the band at low spreads — levels are right. The blue `crypto_4h` cloud sits flat at y ≈ 1.5–3c while x runs 3–6c+: systematic under-prediction of the *time-averaged* quoted spread in fast markets. Predictions are also visibly quantized (52 distinct predicted values across 411 cells) — a binned surface cannot rank within a bin.

### Why it fails — three measured mechanisms

1. **Trade-time vs time-averaged spread (the big one).** The estimator can only see the book *when somebody trades*; the gate's target was the median quoted spread over whole 30-min windows. On the 24h crypto-roll capture (40 tokens), the quoted half-spread **as-of trade times** has median **2.29c**, vs **4.00c time-averaged** — trades systematically execute when books are ~2× tighter than average (makers pulse liquidity around fair-value moves; takers rationally cross when it's cheap). Our crypto_4h surface estimates (1.5–2.5c mid-range) match the trade-time quantity, not the time-averaged one. In slow markets with persistent quotes the two coincide — which is exactly where the gate nearly passes.
2. **1-min mid staleness contaminates fast cells.** 24.6% of all estimates are negative, but the rate is ~0.4–2% in politics_negrisk and 39–55% in daily_crypto mid/high-price cells: when the underlying moves within the minute, P_fill − stale_mid mixes the price move into the spread estimate (both signs, biasing medians toward 0 and below). The tape-only cross-checks in §5 — which condition on the mid NOT moving — stay positive (0.5c bounce, 0.4–0.95c Roll) exactly where the mid-anchored medians go negative. Size split corroborates: in daily_crypto, estimates *fall* with fill size (7.5c → −2c across quartiles) — large fills cluster at fast-move moments; no book-walking signature anywhere (slow categories are flat-to-decreasing in size).
3. **Ties cap the rank statistic.** A binned surface emits 52 distinct values over 411 cells, and the truth itself is massively tied (sports: P25 = P50 = P75 = 0.5c). Spearman ≥ 0.6 demands within-bin discrimination a rule-based binned table (pre-registered: no ML) structurally does not have once levels compress to a few ticks.

## 5. Pre-registered diagnostics (full tables in the CSVs)

- **(a) Negative rate:** 24.6% overall. By category: politics_negrisk 0.4–1.7%, tech 2.6–14%, sports 1.4–21%, geopolitics 3.8–19%, other 5–30%, crypto_4h 8–30%, daily_crypto 16–55%. The rational-user model's failure rate is a *fast-market phenomenon*, not a uniform defect.
- **(b) Size-quantile split:** no monotone increase of estimate with fill size in any slow category (sports 2.5c → 0.5c flat from q2; politics 0.05–0.5c flat) → **no book-walking signature** at the sizes the tape carries; the L1-touch assumption is not visibly violated by large fills. daily_crypto's strong *decrease* (7.5 → 3.5 → −0.5 → −2c) is the staleness mechanism above, not depth.
- **(c) Tape-only cross-checks per category × price bucket:** bid-ask bounce (consecutive opposite-sign fills within 60s and |Δmid| < 1 tick: half the price gap) and Roll (2·√−cov of successive fill-price changes, halved). Where the surface is clean (politics tails: surface 0.05c, bounce 0.05c; geopolitics mid: surface 1.0c, Roll 1.5c) they agree to within ~0.5c. Where the surface goes negative (daily_crypto .65–.95) bounce says 0.5c and Roll 0.4–0.8c — positive, plausible, and the right replacement level for contaminated cells.

## 6. Realism ledger (CODEX § realism calibration)

**Fair vs harsh knobs in the gate itself:** the ≤ 1c MedAE bar was fair and passed with zero margin (1.000c) — report it as "at the bar", not "comfortably under". The 0.6 Spearman bar was pre-registered by us and is honored as FAIL; but note it implicitly assumed the predicted and target quantities are the same thing, and §4 shows they are not in fast markets. The validation sample is also crypto-heavy (194/411 cells) because that's where captures exist — a composition choice, not a market fact.

**Modeled assumptions:**
- L1-touch execution (rational-user model) — diagnosed at 24.6% violation overall, concentrated in fast markets;
- 1-min midpoint fidelity, forward-filled, strictly-before join (30-min staleness cap; 54/99,527 anchors dropped);
- 0.5c tick floor on all predictions;
- capture-side trailing rate (WS tx hashes) ≡ tape-side trailing rate (distinct tx) — same definition, different transports;
- Block K5 slug taxonomy for categories on both sides.

**Live-only / explicitly NOT claimed by this surface:**
- depth at the touch and book-walking beyond L1 (size diagnostic found no signature, but the tape cannot prove depth);
- queue position and passive fill rates (this is a *taker cost* model only);
- adverse selection after the fill;
- quoted spread at arbitrary non-trade times in fast markets — §4 measured that this is a *different quantity* (~2× wider in crypto).

## 7. Decision and next step

**Gate outcome: FAIL-with-diagnosis (pre-registered Spearman bar missed: 0.496 < 0.6; MedAE bar met at exactly 1.000c).** The flat 2–3c assumption is NOT yet replaced as a validated general-purpose quoted-spread model.

What is nonetheless deliverable, with the failure mechanisms as guardrails:

1. **Slow-category level prior (usable now, with stated error):** for geopolitics, sports, tech, other — `SpreadSurface.predict()` gives trade-time half-spread levels validated to slow-pooled MedAE 0.75c (sports 0.0c, geopolitics 0.5c, other 0.9c, tech 1.0c) against true L1, corroborated by bounce/Roll. politics_negrisk levels are tape-plausible and bounce-corroborated but got only 2 validation cells — treat as unvalidated. This is already strictly better than a flat 2–3c constant for those categories (which the surface shows to be wrong by up to 40× at the tails: politics tails cost 0.05c, not 2–3c).
2. **Fast crypto: do not use the mid-anchored cells.** Cells with `frac_negative > 0.4` are contamination-flagged in the CSV; the bounce/Roll columns in `spread_surface_v1_diag_crosschecks.csv` are the honest level estimates there (~0.5–1c at trade times). The time-averaged quoted spread in these markets is ~2× the trade-time spread (4.0c vs 2.3c half) — any consumer that needs "what does the book quote on average" must NOT use this surface.
3. **If copytrade slippage modeling is the use case** (bot fills happen seconds-to-minutes after leader trades — much closer to trade-time than to time-averaged conditions), the right follow-up is a NEW pre-registered gate targeting **quoted half-spread as-of trade times** from the same captures. The §4 cross-check (2.29c trade-time quoted vs our 1.5–2.5c estimates) suggests it would be close, but that is a hypothesis to pre-register, not a result to claim. This would also be the natural place to fix the Spearman mechanics (rank over cells with non-tied targets, or pre-register a level-only bar).
4. **No ML, no further variants of this estimator on the same data** without the new target — iterating the estimator against the same time-averaged truth would be gate-shopping.

Artifacts: surface + meta + validation + diagnostics CSVs under `data/analysis/csv_outputs/copytrade/spread_surface_v1_*.csv` (8 files); per-fill estimates and caches under `data/analysis/spread_surface/` (documented in [[polymarket_data_manifest]]); charts under `data/analysis/plots/copytrade/`. Reproduce: `PYTHONPATH=. uv run python scripts/spread_surface_build.py semantics|fetch-mids|estimate|surface|validate|diagnose|charts` (all steps cached, reruns free).
