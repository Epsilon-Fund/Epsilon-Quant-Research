---
title: "Spread Surface Trade-Time Re-Gate (SPREAD-1b)"
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
# The spread surface re-gated against what copy fills actually pay — trade-time quoted half-spread (Block SPREAD-1b) — gate PASS on all three pre-registered bars

> Hub: [[POLYMARKET_BRAIN]] · [[COWORK]]
> Predecessor gate: [[trade_anchored_spread_surface_findings]] (SPREAD-1, FAIL-with-diagnosis — this block is its pre-registered §7.3 follow-up, a NEW gate, not a reinterpretation)
> Table terms: [[polymarket_table_dictionary]] (SPREAD-1 terms there apply; v1b additions added in this pass)
> Code: `scripts/spread_surface_build.py` subcommands `validate-tradetime` + `charts-1b` · `lib/spread_surface.py` (surface FROZEN; only a `cell_frac_negative` metadata passthrough added) · `tests/test_spread_surface.py` (22 tests, 6 new)

## Plain-English Summary

- **What this is:** SPREAD-1 built a state-dependent estimate of the half-spread a taker pays on Polymarket, and FAILED its validation gate — but the diagnosis showed the gate had scored it against the wrong quantity (the *time-averaged* quoted spread), while the surface estimates *trade-time* execution cost. Copytrade fills happen seconds-to-minutes after a leader's trade, i.e., at trade-time conditions — so this block re-gated the **frozen, unchanged** surface against the quoted half-spread prevailing **at trade times** on the same 6 live captures, under a new pre-registered gate.
- **Gate: PASS on all three bars.** On 397 market-cells: (a) pooled MedAE **0.75c** (bar ≤ 1c); (b) fast-crypto MedAE **0.80c** (bar ≤ 1.25c); (c) the surface beats the incumbent flat-3c assumption head-to-head on **71.7%** of non-tied cells (bar ≥ 60%). Spearman — the statistic that killed SPREAD-1 — was pre-registered here as a diagnostic only: 0.516 pooled, confirming ties/binning cap it, not levels.
- **The diagnosis from SPREAD-1 is confirmed by measurement:** in fast crypto the trade-time target (median 2.05c, crypto_4h cells) is ~⅔ of the time-averaged one (3.25c); in slow categories the two coincide (0.5c vs 0.5c) — exactly why SPREAD-1 nearly passed there and failed in crypto.
- **What this validates / does not validate:** the surface is now a validated **trade-time** taker-cost prior across categories including fast crypto (politics_negrisk still has only 2 validation cells — unvalidated). It remains **wrong by ~2× for anyone needing the time-averaged quoted spread** — that caveat carries forward unchanged.
- **Status:** SPREAD-1's surviving artifact is upgraded from "slow-categories-only level prior" to "validated trade-time cost prior, all categories except politics_negrisk (no power)". This unblocks the surface-fallback path that TODO § copytrade SPREAD-2 option (c) was gated on. Decision on re-issuing SPREAD-2 stays with Cowork.

## 1. Why a new gate, and what was frozen

SPREAD-1's estimator anchors each historical fill to the last `/prices-history` midpoint strictly before it — so by construction it sees the book *only when somebody trades*. Its validation gate, however, compared predictions to the median quoted half-spread over whole 30-minute capture windows. SPREAD-1 §4 measured these to be different quantities in fast markets (2.29c at trade times vs 4.00c time-averaged on the crypto-roll capture): makers pulse liquidity, and takers cross when books are momentarily tight.

For the intended consumer this mismatch matters in the surface's favor: a copy bot fills seconds-to-minutes after a leader's print, i.e., under trade-time conditions, not under book-average conditions. So the correct target for copytrade slippage is the **quoted half-spread as-of trade times** — and that is what this block scores, as a new pre-registered gate (SPREAD-1 §7.3 named it as the follow-up; gates are never reinterpreted after the fact).

**Frozen:** the surface CSV, its bins, the estimator, and `predict()` are bit-identical to SPREAD-1. No refit, no re-binning, no new estimator variants — iterating the estimator against partially-seen capture data would be gate-shopping. The only library change is a metadata passthrough (`cell_frac_negative` on the prediction object) so consumers can see the contamination flag SPREAD-1 already published per cell; predictions are unchanged (tested).

**Practical example.** At 12:00:00.021 UTC a trade prints on a BTC-4h token. The last `best_bid_ask` event strictly before it quoted 0.57 bid / 0.61 ask → the trade-time quoted half-spread is (0.61−0.57)/2 = **2c**. The frozen surface, fed the window's features (price level ~0.59, TTR 2h, busy market), predicts 2.5c → absolute error 0.5c. The incumbent flat-3c assumption errs by 1c on the same cell, so this cell counts as a head-to-head win for the surface in bar (c).

## 2. Pre-registered design

1. **Target:** on the same 6 capture runs (2026-05-27 → 06-10), for every market: at each trade print (`last_trade_price` event) inside a capture window, the prevailing quoted half-spread (ask−bid)/2 from the last `best_bid_ask` event strictly before the print. Window truth = median over the window's prints; market-cells aggregate windows exactly as SPREAD-1 §4 (median per market × category|price-bucket|TTR-bucket cell). Same window grid, same features at window start, same trailing-rate definition — only the truth changed.
2. **Bars (level-first, ties acknowledged):** (a) pooled MedAE ≤ 1c; (b) fast-crypto (crypto_4h + daily_crypto) MedAE ≤ 1.25c; (c) sign test — ≥ 60% of non-tied cells must have |pred − true| < |flat3c − true|; (d) Spearman reported as a diagnostic, NOT a bar.
3. **Contaminated-cell arm:** for predictions sourced from cells flagged `frac_negative > 0.4`, also score the bounce/Roll replacement levels from `spread_surface_v1_diag_crosschecks.csv`; report whether the hybrid beats surface-only.
4. **Realism ledger:** carried from SPREAD-1 §6, plus the trade-time-only scope caveat.

On the flat-3c comparator: the incumbent constants in `data_infra/weather_analysis.py` are **full-spread** numbers (`DEFAULT_SPREAD_CENTS = 2.0`, used as `bid = ask − spread`), while the pre-registration names "flat3c" against a half-spread truth. The official bar uses the literal pre-registered 3c; flat 1.5c and flat 1.0c half-spread variants are scored as sensitivity diagnostics in §5 so the comparator choice is inspectable.

**Sample:** 2,293 windows kept — the 2,744 valid SPREAD-1 windows minus 451 that contained no quoted trade print — carrying 57,501 prints (median 7 per window), aggregating to **397 market-cells** (SPREAD-1 had 411). Composition: crypto_4h 155, other 110, geopolitics 45, daily_crypto 32, tech 29, sports 24, politics_negrisk 2. Everything ran from SPREAD-1's caches; zero new API calls.

## 3. Gate result — PASS on all three bars

| bar | value | gate | verdict |
|---|---|---|---|
| (a) pooled MedAE | **0.750c** | ≤ 1c | **PASS** |
| (b) fast-crypto MedAE (n=187) | **0.800c** | ≤ 1.25c | **PASS** |
| (c) sign test vs flat-3c | **71.7%** (284/396 non-tied) | ≥ 60% | **PASS** |
| (d) Spearman (diagnostic) | 0.516 all cells; 0.347 on the 94 cells with non-tied targets | — | not a bar |

**How to read the table:** each market-cell is one captured market × (category, price-bucket, TTR-bucket) cell; MedAE is the median |predicted − true| over cells, in cents (price units are 0–1 dollars, so 1c = 0.01). The sign test counts only cells where the surface's and the flat-3c error differ ("non-tied"; 1 of 397 tied exactly). The Spearman stays low for the same structural reason SPREAD-1 diagnosed: a binned surface emits few distinct values and the truth itself is tick-tied (slow-category cells are mostly exactly 0.5c), so rank statistics saturate while levels are right.

Per category (MedAE / sign-test share vs flat3c / n): crypto_4h **1.00c / 54.2% / 155**, daily_crypto **0.43c / 71.9% / 32**, geopolitics **0.50c / 90.9% / 45**, other **1.00c / 80.0% / 110**, sports **0.00c / 91.7% / 24**, tech **1.00c / 86.2% / 29**, politics_negrisk **2.29c / n=2 — no power, unvalidated**.

Honest subset reads: the fast-crypto subset alone would NOT clear the 60% sign-test bar (57.2%; crypto_4h alone 54.2%) — flat-3c is genuinely competitive *as a level* in crypto mid-range where true trade-time spreads run 2–3c; the pre-registered bar was pooled and is reported as such. Per capture, MedAE ranges 0.00c (a0b replacements) to 1.36c (block_a0 morning, 20 cells); the crypto-roll capture is the only one whose sign test dips below 60% (49.1%).

![Predicted vs trade-time true quoted half-spread per market-cell, with the ±1c gate band and the flat-3c incumbent line](../../../data/analysis/plots/copytrade/spread_surface_v1b_pred_vs_true.png)

**How to read it:** each point is one market-cell; x = true trade-time quoted half-spread (median over the cell's windows), y = the frozen surface's prediction; grey band = the ±1c MedAE bar; dotted red line = the flat-3c incumbent (a horizontal predictor — every cell gets 3c). **Notice:** the mass sits inside the band at low spreads across all categories — this is the levels being right. The dispersion that remains is in two tails: a handful of over-predictions (y = 7.5–15c at x < 2c — sparse final-hour `other|p_15_35` and `daily_crypto|p_15_35` build cells whose tape medians are wide but whose captured books were tight) and under-predictions at x = 10–24c (thin-print final-hour crypto_4h cells with 2–8 prints, plus three long-TTR tech cells). Consumers should keep using the published `p25_cents`/`p75_cents` and `frac_negative` columns rather than treating any single cell median as exact.

![Trade-time vs time-averaged quoted half-spread on the same market-cells](../../../data/analysis/plots/copytrade/spread_surface_v1b_truth_compression.png)

**How to read it:** same 397 market-cells, x = the SPREAD-1 target (time-averaged quoted half-spread), y = the SPREAD-1b target (quoted half-spread as-of trade prints). **Notice:** slow categories (red/green/pink/brown) hug y = x — the two targets coincide where quotes persist; the blue crypto_4h cloud sits mostly below the line — books are systematically tighter at the moments trades happen (cell medians 2.05c vs 3.25c). This is the measured confirmation of SPREAD-1's central diagnosis, and the reason the same frozen surface fails one target and passes the other.

## 4. Measurement subtlety found and bounded — the "strictly before" quote is mostly the post-trade book

While validating, one mechanical fact surfaced: 99.1% of trade prints have a `best_bid_ask` event within 100ms *before* them (median gap 21ms) — Polymarket's websocket emits the trade-driven book update marginally before the trade print itself. So the literal pre-registered rule ("last quote strictly before the print") largely measures the book **just after** the trade's own liquidity consumption, not just before it.

Bounded by sensitivity: re-scoring every cell with the quote as-of (print − 1s) — genuinely pre-trade — changes the per-print truth in 21% of prints, mean +0.11c wider under the literal rule, and the gate **passes both ways** (t−1s variant: pooled MedAE 0.750c, fast 0.950c, sign test 77.1%). Direction of the bias is against the surface (wider truth, tight predictions), so the official PASS is the conservative reading. For the copytrade consumer the literal rule is arguably the *more* relevant one anyway: a copy fill lands after the leader's trade, on the post-impact book.

Two further robustness reads: restricting to windows with ≥ 3 prints (drops 36 cells) leaves the result unchanged (pooled 0.750c, fast 0.750c, sign test 71.9%); and no window had a negative (crossed-book) truth.

## 5. Flat-baseline sensitivity — where the surface actually earns its keep

| comparator | pooled MedAE | surface beats it head-to-head (pooled / fast / slow) |
|---|---|---|
| flat 3.0c (official bar) | 2.35c | **71.7%** / 57.2% / 84.7% |
| flat 1.5c (3c full-spread-consistent) | 1.00c | 57.7% / 66.5% / 50.0% |
| flat 1.0c | 0.50c | 59.1% / 74.9% / 44.9% |
| surface | 0.75c | — |

**Read:** the literal incumbent is beaten decisively. The harsher diagnostic comparators tell a sharper story: a flat 1c half-spread guess has a *lower pooled MedAE* than the surface (0.50c vs 0.75c) because the trade-time truth in slow categories is so compressed (median 0.5c) that "always guess 1c" is never off by much — yet the surface still wins the per-cell head-to-head against every flat level pooled, because flats lose badly exactly where cost models matter: fast crypto mid-range (true 2–3c, surface wins 74.9% vs flat-1c) and the tails (politics/sports near-certainty books cost 0.05c, not 1–3c). A flat constant and the surface are NOT interchangeable: the flat's small pooled MedAE hides large errors concentrated in the cells where trading actually happens. None of these sensitivity rows is a gate; they are reported so the comparator choice cannot flatter the result silently.

## 6. Contaminated-cell hybrid arm — bounce helps marginally, Roll hurts

Predictions sourced from contamination-flagged surface cells (`frac_negative > 0.4` — the fast-crypto cells where SPREAD-1's mid-staleness diagnosis fired) were alternatively replaced by the tape-only bounce / Roll levels from `spread_surface_v1_diag_crosschecks.csv` (tick floor applied; replacement missing → surface kept). 91 windows → **24 affected market-cells**.

| variant | MedAE on affected cells | beats surface-only head-to-head |
|---|---|---|
| surface-only | 0.50c | — |
| hybrid: bounce replacement | 0.50c | 10/16 non-tied (62.5%) |
| hybrid: Roll replacement | 0.93c | 3/16 (18.8%) |

**Read:** the bounce hybrid is a marginal head-to-head improvement with zero MedAE gain — optional, not required; the Roll hybrid is worse than leaving the surface alone — do not use. The practical guidance for consumers stands as in SPREAD-1: respect the `frac_negative` flag, and where a flagged cell must be priced, the bounce level is the honest substitute.

## 7. Realism ledger (CODEX § realism calibration)

**Fair vs harsh knobs:** all three bars were pre-registered before running and are honored as scored. The flat-3c comparator is the *literal* pre-registered incumbent but is generous relative to the full-spread-consistent 1.5c reading — which is why §5 scores the harsher variants too (the surface survives the head-to-head against all of them, but only the 3c row is the bar). The validation sample remains crypto-heavy (187/397 cells) because that is where captures exist — a composition choice, not a market fact. The sign-test tie rule is exact float equality (1/397 cells tied).

**Modeled assumptions:**
- trade-time conditions ≈ copy-fill conditions: copy fills land seconds-to-minutes after leader prints; this gate validates the quote as-of the print (and, per §4, largely the just-post-trade book). Drift between the leader's print and a copy fill placed minutes later is NOT modeled;
- the frozen SPREAD-1 surface and all its build assumptions (L1-touch model, 1-min mid staleness, 0.5c tick floor, K5 category taxonomy, binned cells);
- window features at window start; truth = median of per-print quotes within the window; market-cell = median over windows;
- flat comparators applied as constant half-spread predictions.

**Live-only / explicitly NOT claimed:**
- depth at the touch and book-walking for size — this is a marginal (per-contract) cost prior, not a fill simulator;
- queue position, passive fill rates, and post-fill adverse selection;
- politics_negrisk levels (2 validation cells — tape-plausible, bounce-corroborated, unvalidated);
- the **time-averaged** quoted spread: measured here again at ~1.5–2× the trade-time quantity in fast crypto (3.25c vs 2.05c cell medians; 4.00c vs 2.29c per-print in SPREAD-1 §4). Any consumer needing "what does the book quote on average" — e.g., a resting-quote MM model — must NOT use this surface;
- spread conditions at copy-fill latencies beyond ~1s after the leader print (the §4 sensitivity bounds only the first second).

## 8. Decision and next step

**Gate outcome: PASS (pooled MedAE 0.750c ≤ 1c; fast-crypto 0.800c ≤ 1.25c; 71.7% ≥ 60% vs flat-3c).** The frozen SPREAD-1 surface is now a **validated trade-time taker-cost prior** for all categories except politics_negrisk (no validation power), including the fast-crypto cells SPREAD-1 had quarantined for the time-averaged target. The flat 2–3c assumption is superseded *for trade-time cost consumers* — copytrade slippage modeling being the motivating one.

Scope guardrails that carry forward: (1) the time-averaged 2× caveat — this surface answers "what does crossing cost when trades happen", not "what does the book quote on average"; (2) `frac_negative`-flagged cells should be refused or bounce-substituted (§6); (3) per-cell IQRs are published — tail cells (final-hour, thin-print) have real dispersion (§3 chart read); (4) politics_negrisk stays unvalidated until a politics-heavy capture exists.

Concrete next action: this PASS satisfies the prerequisite that TODO § copytrade SPREAD-2 option (c) was gated on — SPREAD-2 (Phase 5 surface fallback + mid_at_trade + MTM equity) can now be re-issued as originally written, with the surface fallback no longer restricted to slow categories (politics_negrisk excepted). That re-issue decision is Cowork's; no SPREAD-2 code was touched in this block.

Artifacts: `data/analysis/csv_outputs/copytrade/spread_surface_v1b_{validation_windows, validation_market_cells, gate_summary, hybrid_comparison}.csv`; charts under `data/analysis/plots/copytrade/spread_surface_v1b_*.png`. Column glossary: window/cell CSVs reuse SPREAD-1 terms ([[polymarket_table_dictionary]]) plus `true_tt_half_c` (trade-time truth), `true_ta_half_c` (time-averaged truth, same window — diagnostic), `n_trades` (prints behind the truth), `med_quote_age_s` (median print-to-prior-quote gap), `pred_frac_negative` (contamination flag of the sourcing cell), `hyb_bounce_c`/`hyb_roll_c` (+ `_swapped` flags) for the hybrid arm. Reproduce: `PYTHONPATH=. uv run python scripts/spread_surface_build.py validate-tradetime && … charts-1b` (all inputs cached; no API calls).
