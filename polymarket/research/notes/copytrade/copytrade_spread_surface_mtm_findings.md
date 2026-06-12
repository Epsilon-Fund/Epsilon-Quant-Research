---
title: "Copy-Execution Phase 5: mid-as-data, category-gated surface fallback, and MTM equity (Block SPREAD-2)"
created: 2026-06-11
status: closed
owner: justin
project: polymarket
para: project
hubs:
  - COWORK
tags:
  - research
  - copytrade
  - spread
---
# Phase-5 copy execution upgraded: the trade-time spread surface replaces the flat-3c slippage fallback, plus a lookahead-free MTM equity curve (Block SPREAD-2)

> Hub: [[COWORK]]
> Builds on: [[trade_anchored_spread_surface_findings]] (SPREAD-1) · [[spread_surface_tradetime_regate_findings]] (SPREAD-1b PASS — the gate that licensed this) · [[phase5_design]] (the copy-backtest design this upgrades)
> Table terms: [[polymarket_table_dictionary]]
> Code: `lib/copy_slippage.py` (shared core + tests) · `scripts/phase5_spread_surface.py` (driver) · opt-in `--surface-fallback` in `scripts/domah_copy_audit.py`, `scripts/backtest/walkforward_stage15.py`, and `data_infra/weather_analysis.py`

## Plain-English Summary

- **What this is:** Phase 5 of copytrade models the slippage a copy fill pays as either the next observed trade in the market (`next_fill`) or, when no such trade exists in the window, a flat 3-cent guess (`fallback`). SPREAD-1b validated a state-dependent **trade-time** spread surface; this block wires it in as the fallback for the categories it cleared, makes the `/prices-history` mid a first-class column, and adds a lookahead-free mark-to-market (MTM) equity curve so cohorts are judged on more than resolution-only PnL.
- **Three upgrades, all delivered:** (1) `mid_at_trade` + `leader_vs_mid_cents` on leader anchors, decomposing next-fill cost into spread vs drift; (2) lookahead-free daily MTM equity per cohort; (3) the category-gated `surface_fallback` replacing flat-3c for the six validated K5 categories (politics_negrisk stays flat-3c — it was unvalidated at SPREAD-1b n=2).
- **Headline re-gate result:** swapping flat-3c → gated `surface_fallback` **flips no deployment verdict** — not one of the 7 leader audits changes its pure-taker profitability sign, and not one of the 72 stage15 cohort runs flips its PnL sign or crosses the Sharpe=1.0 gate. What it *does* do is replace the crude flat-3c on **42–100% of fallback rows** with a category-validated estimate, and reveal that flat-3c is **systematically wrong**: too cheap for HFT makers (high_conviction true ~3.8c, ultra_maker ~4.5c) and too expensive for patient/tight-book leaders (ee00ba ~1.4c, top_leaderboard ~2.0c).
- **Domah MTM (the "both options" add-on):** the same lookahead-free MTM run on the Domah copy-audit ledger, twice from identical fills — his own prices vs pure-taker copy prices. His book: **+$1.14M MTM, Sharpe 0.95**; the copy: **−$429k, Sharpe −0.36**, with a −$880k trough (2× the final loss). The ~$1.57M gap is pure execution price on identical flow; the copy bleeds from month 4 and never re-touches breakeven — not rescuable by timing or drawdown control.
- **Status / takeaway:** the surface fallback is a strictly better-calibrated, validated replacement for the flat-3c assumption that does not by itself rescue or break any cohort. It is now available as opt-in in all three evaluators; the executed re-run used the cheap cached-output path (no backtest re-run). Lookahead-free MTM is shipped as a first-class diagnostic, for stage15 cohorts and per-leader audit ledgers alike.

## 1. What changed vs the original SPREAD-2, and why now

SPREAD-1 built the trade-anchored spread surface but FAILED its first gate; SPREAD-1b re-gated the **frozen** surface against the trade-time quoted half-spread — the quantity a copy fill actually pays — and **PASSED** all three bars (pooled MedAE 0.75c, fast-crypto 0.80c, 71.7% head-to-head vs flat-3c). That PASS is the prerequisite this block was gated on. Parts 1–2 here depend only on the `/prices-history` mid (whose midpoint semantics SPREAD-1 confirmed at 0.0c vs replayed L1) and run unconditionally; Part 3 (the surface fallback) is gated per category by the SPREAD-1/1b result.

The surface is **FROZEN** — same CSV, bins, estimator, and `predict()` as SPREAD-1. This block adds no estimator variants; it only *consumes* the validated surface. The shared logic lives in one place, `lib/copy_slippage.py`, imported by the driver and by all three evaluator entry points.

**Practical example.** A leader posts (maker) a BUY of a BTC-4h token at 0.61 and gets filled; the copy bot detects it 20s later and finds **no** other trade in the 15–300s window (a quiet moment). Under the old model it pays a flat 3c → copy fill 0.64. Under SPREAD-2, the token is K5 category `crypto_4h` (validated), the leader was a maker so the copy must cross a **full** spread (2× the surface's ~2.5c half-spread = 5c, floored/capped to [0.5c, 8c]) → copy fill 0.66. The fill is tagged `slippage_model = surface_fallback` so the audit can separate validated-surface fills from the residual flat-3c ones.

## 2. Pre-registered design (evaluation upgrade only — the cohort signal is untouched)

1. **`mid_at_trade` (unconditional).** For each leader anchor, the forward-filled `/prices-history` mid strictly before the fill (reusing the SPREAD-1 cached fetch + cache keying), plus `leader_vs_mid_cents` (signed, adverse-positive). Used to decompose next-fill slippage into a **spread** component (the surface's predicted half-spread) and a **drift** component (how far the copy landed beyond mid + half-spread); a fill beyond mid + half-spread is flagged drift-dominated.
2. **MTM equity curve (unconditional).** Lookahead-free daily MTM per cohort: each open position is marked to its forward-filled token mid (`t ≤ grid`), and to resolution at `end_date`; MTM Sharpe + max drawdown are reported alongside the resolution-only numbers. No future quote ever enters a past grid point (bisect, `ts ≤ t`) — the `spcx_backfill_history.py` pattern.
3. **Category-gated surface fallback.** `slippage_source = 'surface_fallback'` for the six K5 categories that cleared a SPREAD-1b bar — `crypto_4h, daily_crypto, geopolitics, sports, tech, other`. **politics_negrisk is excluded** (SPREAD-1b n=2, no power) and keeps flat-3c, as does any non-validated category. **Full spread (2× half) when the leader fill was maker-side** (the copy must cross to take that side), half otherwise. Floor/cap [0.5c, 8c]. Contamination-flagged cells (`frac_negative > 0.4`) use the SPREAD-1b **bounce** replacement level (Roll lost the hybrid arm). Flat-3c is kept everywhere as the comparison arm.
4. **Pre-registered re-run.** Existing audited leaders (Domah + 6 completed audits) and the 72 cached stage15 cohort runs under {flat-3c, gated surface_fallback}: does the deployment verdict flip, and does the >40%-flat-fallback population shrink? Plus the drift-vs-spread split on next-fill slippage (Domah).
5. **Realism ledger.** Depth/capacity/fill-share remain live-only; mid fidelity is 1-min (the SPREAD-1 trade-time staleness caveat carries verbatim); the surface is a trade-time cost model, never a time-averaged quote model.

**Quiet-by-construction activity bucket.** A fallback row means no other trade printed in the 15–300s window — i.e., a *quiet* moment — so the surface is queried at trailing trade-rate 0 → activity quartile `act_q1`, the principled bucket for a quiet market. The prediction then depends only on (category, price-bucket, TTR-bucket, maker-side), a few hundred distinct keys, so the re-pricing is memoized (the largest leader audit is 5.1M fills).

**Execution scope (honest about what was run).** The surface fallback only changes *fallback* rows; `next_fill` rows are untouched. So the re-run needs no backtest re-execution — it recomputes the fallback price on the cached leader fragments and cached stage15 audit logs. The capability is wired as opt-in (`--surface-fallback`) into all three evaluators for future live use, but the executed numbers below come from the cached-output path (no 200M-row WF SQL re-scan).

## 3. Re-gate result — no verdict flips; flat-3c is replaced on 42–100% of fallback rows

### 3.1 Leader audits (Domah + 6) — pure-taker (Branch B) under {flat-3c, surface_fallback}

| leader | fallback share | fallback shrink (→validated surface) | mean surface fallback (c) | Branch-B PnL Δ (surface − flat3c) | verdict flip |
|---|---|---|---|---|---|
| domah | 50.9% | 77.0% | 3.42 | +$27.3k | no |
| high_conviction | 26.9% | 100% | 3.82 | −$2.01M | no |
| ultra_maker | 3.8% | 100% | 4.49 | −$67.1k | no |
| top_leaderboard | 5.2% | 99.1% | 1.96 | +$121k | no |
| negrisk_directional_1 | 53.2% | 41.6% | 4.04 | −$28.0k | no |
| negrisk_directional_2 | 35.8% | 68.8% | 3.22 | +$182k | no |
| ee00ba | 56.5% | 98.9% | 1.39 | +$1.31M | no |

**Column meaning.** *fallback share* = rows with no observed next-fill (the only rows the surface touches). *fallback shrink* = of those, the share now served by the validated surface rather than flat-3c (the un-shrunk remainder is politics_negrisk — correctly held on flat-3c; it is largest for the two NegRisk-directional leaders, which is the expected sign). *mean surface fallback (c)* = the average gated surface slippage applied to those rows. *Branch-B PnL Δ* = change in pure-taker copy PnL (every fill crosses) from swapping the fallback regime; *verdict flip* = whether the pure-taker profitability sign changes.

**Read.** No leader's pure-taker profitability sign flips. The economically interesting finding is in the *direction* of the re-pricing: the surface says flat-3c **under-charges the HFT makers** (high_conviction 3.82c, ultra_maker 4.49c → PnL drops, most starkly high_conviction −$2.0M because 1.38M of its fills are repriced wider) and **over-charges patient/tight-book leaders** (ee00ba 1.39c, top_leaderboard 1.96c → PnL improves +$1.3M / +$121k). Flat-3c and the surface are not interchangeable: the flat constant hides large, category-systematic mis-pricing exactly on the fallback rows where the model is the only information available.

### 3.2 Stage15 cohorts (72 runs) — §8 gate verdict under {flat-3c, surface_fallback}

| cohort | runs | total PnL flat3c | total PnL surface | Δ% | mean fallback share | mean shrink | PnL-sign flips | Sharpe-cross-1.0 flips |
|---|---|---|---|---|---|---|---|---|
| BC_directional_negrisk | 24 | −$159.8k | −$164.7k | −3.0% | 23.3% | 81.0% | 0 | 0 |
| B_high_pf_with_size | 24 | −$132.6k | −$137.6k | −3.7% | 24.2% | 94.4% | 0 | 0 |
| E_patient_accumulators | 24 | −$245.9k | −$248.6k | −1.1% | 28.7% | 88.9% | 0 | 0 |

**Read.** All three cohorts are net-negative under both regimes — they fail the §8 deployment criteria regardless of the slippage model (consistent with the standing copytrade finding that no cohort clears the gate yet). The surface fallback makes them marginally *more* negative (−1.1% to −3.7%) — expected, since stage15 leaders are makers by construction (signals are pulled on `maker IN cohort`), so the copy crosses the full spread, and for these categories the full spread averages slightly above 3c. Crucially, **0 of 72 runs flip their PnL sign and 0 cross the Sharpe=1.0 line** (the same 12 runs sit ≥1.0 under both regimes). 20/72 runs trip the design's >40%-fallback quiet-market flag; the surface shrinks the *flat*-3c portion of that by 81–94%. *Scope note:* this checks the load-bearing §8 criteria (PnL sign, Sharpe threshold) per run; per-run monthly Sharpe can still swing materially on small-n runs, but no run crosses the deployment threshold and every cohort total stays net-negative, so the deployment verdict (no cohort deployable) is unchanged.

**Bounce hybrid never fired (0 swaps anywhere).** No fallback row — leader or cohort — resolved to a contamination-flagged cell (`frac_negative > 0.4`); those cells are daily-crypto mid/high-price, which these leaders' quiet-moment fallback rows don't populate. The hybrid arm is wired and tested but is inert on this data.

## 4. Mark-to-market equity per cohort (lookahead-free)

Each stage15 cohort's copy ledger is marked daily: open positions to their forward-filled token mid (`t ≤ grid`, the same `/prices-history` mid the surface uses, chunked ≤10-day at hourly fidelity), resolved positions to `position_resolution`. 669 held tokens fetched (0 transient failures). Drawdown is reported as % of gross capital deployed (the MTM curve is cumulative PnL, so a peak-relative fraction is degenerate).

| cohort | positions | deployed capital | final MTM equity | MTM Sharpe (daily, ann.) | MTM max drawdown (USD) | … (% of deployed) |
|---|---|---|---|---|---|---|
| BC_directional_negrisk | 1,062 | $1.37M | −$159.8k | −1.46 | $183.5k | 13.4% |
| B_high_pf_with_size | 3,367 | $3.00M | −$132.6k | −0.78 | $165.7k | 5.5% |
| E_patient_accumulators | 3,591 | $3.60M | −$245.9k | −1.20 | $249.9k | 6.9% |

![Lookahead-free MTM equity per cohort (daily grid; open positions marked to forward-filled token mid, resolved positions to outcome)](../../data/analysis/plots/copytrade/spread_surface_phase5_mtm_equity.png)

**How to read it.** Each line is a cohort's cumulative MTM equity (realized + unrealized) on a daily grid; the dotted line is breakeven. Endpoints equal the resolution-only PnL (a clean cross-check: −$159.8k / −$132.6k / −$245.9k match the §3.2 flat-3c totals to rounding). **Notice:** the MTM path carries information resolution-only PnL hides. `BC_directional_negrisk` sat **MTM-positive for ~14 months** (2024-09 → 2025-11) before a sharp markdown into a −$160k resolution — a book that looked like it was working for over a year and wasn't; `E_patient_accumulators` takes its worst drawdown ($250k, 6.9% of deployed) in a late-2025/early-2026 cliff. All three cohorts have **negative MTM Sharpe** (−0.78 to −1.46), so they are not merely net-losers at resolution but carry negative risk-adjusted equity paths throughout — a stricter read than the resolution-only numbers alone. This is the deliverable's point: the MTM curve is a more honest cohort diagnostic than terminal PnL, and it confirms (does not rescue) the standing "no cohort deployable" verdict.

## 4b. Domah-audit MTM: the leader's own book vs the pure-taker copy, as equity paths

The cohort MTM above judges the stage15 ledgers; this section runs the same lookahead-free daily MTM on the **Domah copy-audit ledger** (170,005 fills, $40.5M gross volume) — twice, from the same fills: once at **Domah's own fill prices** (his actual book) and once at the **pure-taker copy price** (Branch B: next-fill, flat-3c fallback — the same branch as §3's re-gate). Marks come from one `interval=max&fidelity=1440` `/prices-history` call per held token (the only request form the endpoint serves coarse fidelity in; explicit-span requests are capped at ~15 days regardless of fidelity). Open positions mark to the forward-filled daily mid; resolved positions realize `resolution_price` at `end_date`.

| ledger | final MTM equity | realized | MTM Sharpe (daily, ann.) | max drawdown (USD) | …(% of gross volume) |
|---|---|---|---|---|---|
| Domah's own book | **+$1.14M** | +$868k | **0.95** | $500k | 1.2% |
| pure-taker copy (Branch B) | **−$429k** | −$393k | **−0.36** | $886k | 2.2% |

![Domah copy-audit ledger: lookahead-free MTM equity, own book vs pure-taker copy](../../data/analysis/plots/copytrade/spread_surface_phase5_domah_mtm_equity.png)

**How to read it.** Both lines are cumulative MTM equity (realized + unrealized) over the same 170k fills on a daily grid — the ONLY difference between them is the entry price (his fills vs taker-copy fills). The flat tail after mid-2026 is the book sitting unchanged once the trade tape ends (2026-05-26) with marks frozen at each token's last quote, until late-2026 market resolutions realize; the small terminal step is those resolutions. **Notice three things.** (1) The vertical gap between the curves (~$1.57M at the end) is the pure execution-price tax on identical flow — it grows monotonically because every fill pays it. (2) Domah's own book is no smooth ride either: flat-to-negative for the first ~9 months (trough ≈ −$165k), then one regime (late-2025 → early-2026) delivers essentially all of the +$1.1M — his edge is episodic, which matters for anyone hoping to copy a "steady" maker. (3) The copy curve's trough (−$880k, 2025-12) is **2× its final loss** — a copier with finite risk tolerance would almost surely have stopped out near the bottom, making the realistic copy outcome worse than the terminal −$429k. The MTM path confirms §3's verdict with a stricter lens: the copy is not a near-miss that better timing or drawdown control could rescue; it bleeds from month 4 onward and never reaches breakeven again.

**Reconciliation + approximation ledger.** §3's Branch-B flat-3c total (−$632k) marks unresolved positions at their last fragment price; the MTM final (−$429k = −$393k realized − $36k open-marked) marks them at the forward-filled daily mid instead — same fills, different unresolved-marking convention, both reported. Mid coverage at time of writing: 4,912 of 5,890 held tokens (83.8%) have full-life daily mids; the remaining 978 (mostly dead/quiet books the endpoint returns nothing for) mark at their fill-weighted average entry price — a neutral approximation that biases both curves identically and cannot affect the own-vs-copy gap. The realized components (+$868k / −$393k) are mid-independent and exact. 22.6% of fills sit in positions unresolved as of the data tail; they stay open and marked throughout.

## 5. Domah next-fill slippage: spread vs drift

For every Domah `next_fill` anchor with a `mid_at_trade`, the realized copy cost vs mid is split into a **spread** component (the surface's predicted half-spread) and a **drift** component (`copy_vs_mid − predicted_half_spread` — how far the fill landed beyond mid + half-spread); a fill beyond mid + half is flagged drift-dominated. Sample: **16,016 next-fill anchors with mid** (from the partial mid cache — see coverage note below; the per-category means are saturated at this n).

| K5 category | n | mean spread (c) | mean drift (c) | drift-dominated share | mean leader_vs_mid (c) |
|---|---|---|---|---|---|
| geopolitics | 4,950 | 1.65 | −0.79 | 21.3% | −0.30 |
| other | 5,676 | 2.21 | −1.36 | 13.2% | −0.33 |
| politics_negrisk | 4,860 | 0.98 | −0.46 | 27.2% | −0.62 |
| sports | 160 | 0.75 | −0.25 | 21.9% | −0.82 |
| tech | 370 | 8.99 | −8.05 | 17.6% | +0.12 |
| **overall** | **16,016** | **1.80** | **−1.05** | **20.1%** | **−0.41** |

![Domah next-fill copy cost decomposed into spread (predicted half-spread) and drift (beyond mid + half) by K5 category](../../data/analysis/plots/copytrade/spread_surface_phase5_domah_drift_spread.png)

**How to read it.** Blue = the surface's predicted half-spread (the modeled "spread" cost of crossing at fair); orange = the residual drift (positive = the market ran away from us between the leader's print and the copy fill; negative = the fill landed *inside* the predicted band, i.e. better than fair-mid-plus-spread). `leader_vs_mid` is where the leader's *own* fill sat relative to mid in his direction (negative = at/inside mid).

**Read — copy slippage is spread-dominated, not drift-dominated.** Mean drift is *negative* in every category (overall −1.05c), so on average Domah's copy fills land **inside** the surface's predicted half-spread band — the market does not systematically run away between the leader's print and the copy fill. Only **20%** of fills are drift-dominated (beyond mid + half); the other 80% are explained by the modeled spread, which is exactly the quantity the surface predicts — direct support for using the surface as the copy-cost model rather than a flat constant. `leader_vs_mid ≈ −0.4c` (negative across the board) confirms Domah fills at or inside mid in his own direction — consistent with his maker-heavy style (he posts and is hit, rather than crossing). `tech` is a small-n outlier (n=370): the surface predicts a wide ~9c spread there while realized cost is far less, producing a large negative drift — treat as noise, not signal, at this n.

**Coverage note (honest about the sample).** This diagnostic uses the partial Domah mid cache (~5k of 27,413 token-days fetched at the time of writing; the full fetch continues in the background and is idempotent/cached). 16,016 anchors with ~5k each in the three large categories is already saturated for the per-category means; re-running `regate --with-domah-drift` after the fetch completes refreshes this section at full coverage for free (no re-fetch). The drift split is **descriptive** and uses the surface's predicted half-spread for *all* categories (including politics_negrisk) — this is distinct from the gated surface_fallback *re-pricing* of §3, which excludes politics_negrisk.

## 6. Realism ledger (CODEX § realism calibration)

**Fair vs harsh knobs.** The gated category set and the bounce-over-Roll choice are the SPREAD-1b PASS results, carried verbatim — not re-tuned here. politics_negrisk is held on flat-3c (honest: it was never validated). The maker→full-spread doubling is a modeled assumption about how a copy bot replicates a maker entry by taking; it is the conservative direction (more cost), and it is why the stage15 cohorts (all-maker signals) move slightly negative.

**Modeled assumptions:** copy fill ≈ trade-time conditions (SPREAD-1b's validated regime); fallback rows are quiet → `act_q1`; the leader price-level is used as the surface's price input on fallback rows (coarse buckets make this robust); the frozen SPREAD-1 surface and all its build assumptions (L1-touch, 1-min mid fidelity, 0.5c tick floor, K5 taxonomy); MTM marks use 1-min/coarse `/prices-history` mid forward-filled.

**Live-only / explicitly NOT claimed:** depth at the touch and capacity (this is a per-contract cost model, not a fill simulator); queue position and passive fill rates; post-fill adverse selection; the **time-averaged** quoted spread (the surface is a trade-time cost model — never use it for "what does the book quote on average", per the SPREAD-1b 2× caveat); fill probability of the copy order itself.

## 7. Decision and next step

The validated trade-time surface is now the calibrated, category-gated replacement for the flat-3c slippage fallback across the Phase-5 copy evaluators. It does not by itself flip any deployment verdict — every leader's pure-taker sign and every stage15 cohort's gate verdict are unchanged — but it removes a systematically biased constant (flat-3c under-charges HFT makers and over-charges tight-book leaders) on the 42–100% of fallback rows where it is the only available estimate, and it ships a lookahead-free MTM equity curve as a first-class cohort diagnostic.

Concrete status: capability wired as opt-in (`--surface-fallback`) in all three evaluators; executed re-run via the cached-output path; politics_negrisk correctly excluded. No change to any cohort signal or sizing rule. The §4b Domah MTM closes the strongest remaining hope for the pure-taker copy: the loss is not a terminal-PnL artifact but a persistent negative equity path (trough 2× the final loss), while the same flow at the leader's own prices earns Sharpe ~0.95 — the copytrade problem is, measurably, an execution-price problem, which is exactly the quantity the SPREAD-1b surface now models.

Artifacts: `data/analysis/csv_outputs/copytrade/spread_surface_phase5_{leader_regate, stage15_regate, stage15_cohort_rollup, mtm_summary, mtm_equity_curves, domah_mtm_summary, domah_mtm_curves, domah_drift_spread, domah_drift_by_cat}.csv`; charts under `data/analysis/plots/copytrade/spread_surface_phase5_*.png`. Reproduce: `PYTHONPATH=. uv run python scripts/phase5_spread_surface.py {fetch-mids,fetch-mtm,fetch-mtm-leader,regate --with-domah-drift,mtm,mtm-leader,charts}` (all fetches cache-on-200-only and resume idempotently).
