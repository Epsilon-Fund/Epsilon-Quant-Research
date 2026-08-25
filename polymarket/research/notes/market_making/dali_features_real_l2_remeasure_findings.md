---
title: "Dali features re-measured on real captured L2 — workflow hardening + class-B closure re-tests through the institutional fill engine"
created: 2026-07-21
status: active
owner: justin
project: polymarket
para: project
hubs:
  - strat_market_making
  - POLYMARKET_BRAIN
tags:
  - polymarket
  - market-making
  - dali
  - data-quality
  - audit
---

> **HISTORICAL EVIDENCE (2026-08-25).** Detailed findings behind the active market-making project, kept for verification. Read the de-jargoned canon surface first — [[strat_market_making]] + [[mm_model]] — and treat every number here as preliminary per the hub reliability ledger.


# Dali features re-measured on real captured L2 — workflow hardening + class-B closure re-tests

> Hub: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · Decision this executes: [[pm_dali_workflow_revision_decision]] · Why: [[pm_prealvaro_pipeline_trust_audit_findings]] · Data: [[mm_vps_capture_setup]] · [[polymarket_data_manifest]] · Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

- **What this is.** The Tier-2 workflow hardening and Tier-3 re-measurement from [[pm_dali_workflow_revision_decision]], executed on the real 24/7 VPS L2 capture. The capture covers **politics NegRisk and esports** — *different* markets than dali's crypto/geopolitics — so every result here is fresh out-of-sample evidence about these markets, not a replication of dali's.
- **Part A shipped and committed:** one owned metric module that reproduces BOTH of the audit's disputed Finding-1 hit-rate figures from a single code path; a REQUIRED, thresholded, lead-lag-aware capture-quality gate; and a book-measured cost helper that retires the estimated spread surface wherever the captured book covers a timestamp.
- **Part B headline results:** (1) the corrected dali descriptive TOB claim **replicates and generalizes** to the new markets (esports ~71% conditional hit rate on 209k out-of-sample episodes, politics ~68%); (2) under a doubly-disjoint clean split, the dali-descended features **do carry conditional signal** — settling A17's condemned sub-claim in the *opposite* direction of its original framing (OFI on politics is the exception: dead); (3) the taker cost floor on these markets is far lower than dali's crypto floor (fees are zero, spreads 0.1–1c) **but still sits above the measured 5s signal** — no taker reopen; (4) the dali passive/maker framings (A14c/A14h/A18/P2 flavors), re-run through the coded MM fill engine under the full Optimistic/Prob/RiskAverse queue bracket, come out **negative in all 54 grid cells on BOTH universes — 0 cells CI-positive, 53/54 with the market-cluster CI entirely below zero** — so the assumption-grade closures are **upgraded to robust, confirmed on real data**. No closure flips positive.
- **The one reopen:** narrow and precise — the *feature-signal reading* attached to A17. "These features carry no conditional signal" is refuted on real L2 under the clean split. What reopens is the signal **as an MM-gate input candidate** (the Tier-3 outcome-1 path: feed TOB/microprice state to the NSQ quoter's skew/toxicity gate), not any taker or standalone passive edge.
- **Standing caveat on every passive number here:** no queue model is calibrated against our own fills (Join-2's stated first-order job). A positive under the bracket would be a *reopen-warranting candidate*, not an edge; the negatives reported here are the bracket's *best case* too, since the bracket turned out to barely bite (§ B3 read).

## Assumption ledger (modeled vs live-only)

**Modeled assumptions (everything in Part B3):**
- Queue-position attribution: the Optimistic / Prob(f=0.5) / RiskAverse bracket, uncalibrated (`calibrate(live_fills)` is a Phase-2 stub).
- Round-trip latency: constant 200ms (engine default; the Join-2 harness measured ~160ms on one config — 200ms is mildly conservative).
- Exits: dali's `forced_taker` convention, priced from the reconstructed book at t+H (≤5s staleness) — not a simulated exit queue. Maker-exit variants (A14c's `exit_symmetric_maker`) were NOT re-run; taker exits are the well-defined executable convention and were part of every dali grid.
- Gap handling: heartbeat inference (30s universe-wide silence) stands in for the absent `capture_gaps` sidecars (§ B0b).
- One token per binary market (the NO book mirrors the YES book), 1-contract orders, per-asset |signal| q90 thresholds (the dali blocks' own descriptive conditioning).

**Live-only unknowns (cannot be resolved offline; Join-2 owns them):**
- Our real fill rate / queue share; adverse selection conditional on OUR quote resting (the book reacts to us); edge persistence vs other participants' latency.

## Evidence classes (the trust audit's convention)

**A** = computed here, reproduction script committed · **B** = mechanism argument · **C** = documentary (code/config/data field) · **D** = reproduces a pre-existing published number. Every number below is tagged.

---

## Part A — workflow hardening (Tier-2 rules made code; all committed)

### A1. One owned metric definition — and the Finding-1 reproduction gate

`lib/microstructure_metrics.py` defines hit rate (explicit `zero_move` parameter: `"miss"` default = executable-relevance reading; `"exclude"` = the conditional reading), directional-return bps (sign-weighted), and non-overlap episode counting ONCE. The committed test `tests/test_microstructure_metrics.py` proves the single `hit_rate` code path reproduces BOTH Finding-1 figures on the exact Retest-C universe (a0c_roll crypto-4h markets, ≥300 last-trade events, discovery threshold recomputed):

| figure | official / audit | this module | class |
|---|---|---|---|
| zero-as-miss hit rate (Retest C's "36.0%") | 36.0% / 36.1% | **36.14%** (n=3,143 episodes) | D |
| conditional hit rate (audit's recompute) | 63.0% | **63.29%** (n=1,795 nonzero-move episodes) | D |
| discovery threshold abs_q90 | 0.937422 | **0.937422** (exact) | D |

**Discrepancy reported honestly:** the retest surface's pooled `mean_directional_return_bps = +58.1` turns out to be a **raw** episode-mean (no sign weighting) — a third metric variant mislabeled "directional". This reconstruction yields **+61.2 raw** (~3 bps off the official CSV; episode-set drift at reconstruction boundaries is the presumed source — bounded, since the hit-rate anchors reproduce to <0.2pp) and **+82.3 sign-weighted**. Irony worth recording: under the CONSISTENT sign-weighted definition, the dali holdout directional return (+82.3) did not degrade versus discovery (+72.9) at all — the audit's "+73 → +58 bps" comparison was itself a residual raw-vs-sign-weighted mismatch, one more instance of Finding 1's disease. None of this changes any closure.

### A2. Required capture-quality gate (thresholded, fail-closed)

`mm_eval/capture_gate.py` ports the alvaro reconstruction-audit checks + `mm_engine.book.BookTracker` into a library gate every analysis must pass before consuming a slice (Tier-2 rule 2). Checks: **lead-lag-aware BBA checksum** — each `best_bid_ask` checkpoint classified `clean` / `clean_lagged` (mismatch at arrival that resolves once the same-millisecond `book`/`price_change` updates apply — the intra-ms ordering artifact from [[mm_reconstruction_audit_findings]]) / `stale`-with-reason / `mismatch`; ≤5s staleness gating; sidecar + heartbeat-inferred gap handling; trade-in-spread coherence; exchange-ts-primary ordering. Explicit thresholds: **PASS** = ≥95% lead-lag-aware fresh-clean AND ≥95% trade-in-spread AND ≤20% stale AND ≥1,000 checkpoints; **MARGINAL** ≥85% fresh-clean; else **FAIL**; `require_pass()` raises (`CaptureQualityError`) and both the feature builder and the retest engine call it fail-closed.

### A3. Book-measured costs

`lib/book_costs.py::BookCostIndex` — as-of lookup over the gated L1-state shards: last fresh measured L1 at/≤ ts, ≤5s staleness, stale states never served. The retired SPREAD-1/1b/2 surface survives only behind `source="surface_fallback"` for uncovered timestamps; every quote carries its `source` so measured and estimated costs can never silently mix (Tier-2 rule 3 made code).

---

## Part B — measurements on the real L2 (pre-registered design)

### B0. Data + design

- **Archive:** `r2:epsilon-polymarket-data/parquet/`, 33 days (2026-06-19 → 07-21), 35.7 GiB, universes **politics_negrisk + esports only** (see B0b).
- **Pre-registered 14-day window:** 2026-06-19..25 (train week) + 2026-07-15..21 (test week) — a 5-week span with a ~3-week embargo between halves. Chosen for tractability after the full-33-day replay proved memory-infeasible on this machine (two swap-thrash incidents; workers re-sized to RAM per the CODEX rule).
- **Splits:** analysis A fits thresholds on train days, evaluates on test days. Analysis B additionally hash-splits MARKETS into disjoint halves — evaluation is on **test-markets × test-days** (doubly disjoint; the Task-5.1 fix). Episodes are non-overlapping 5s via the shared metric module. CIs: percentile cluster bootstrap over markets (the A18 convention), 1,000 draws, seeded.
- Scripts (all committed): `scripts/mm_real_l2_gate_scan.py` (gate + L1-state recording in one replay), `scripts/mm_real_l2_features.py` (features, fail-closed behind the gate), `scripts/mm_dali_retest_engine.py` (engine re-tests), `scripts/mm_real_l2_remeasure_analysis.py` (analyses A–D). Outputs: `data/analysis/csv_outputs/market_making/real_l2_remeasure/*.csv`.

### B0b. Data-quality findings surfaced by the gate work

1. **(class C) The `culture` and `crypto-as-control` universes were never captured to Parquet.** [[mm_vps_capture_setup]] describes four target universes; the archive holds exactly two, on every one of the 33 days. Consequence: there is **no crypto control** on the new capture — nothing here can re-measure dali's own market class directly. The capture-setup note now carries this correction; the VPS discovery config should be checked.
2. **(class C) No `capture_gaps` sidecars and no `metadata/` prefix exist in R2** for the rolling capture (the setup note promises both). Since raw JSONL — which carries `capture_gaps.jsonl` — is expired the moment Parquet is confirmed, **gap ground truth is being permanently deleted**. The gate falls back to labelled heartbeat inference (30s universe-wide silence). Upstream fix needed: sync `capture_gaps.jsonl` (or a `capture_gaps.parquet`) into the Parquet layout before raw expiry.
3. **(class A) The replay-integrity claim re-ran and HOLDS on the new markets.** Over 17.3M BBA checkpoints across 28 slices: **weighted lead-lag-aware fresh-clean = 98.0%** (per-slice 93.4–99.8%); verdicts **22 PASS / 6 MARGINAL / 0 FAIL**. The raw-ordering clean % is 18–62% — the same intra-ms artifact JOIN-1 diagnosed, now handled by the gate's `clean_lagged` class instead of a manual correction. This is the dali-era "99.29% clean" claim independently rebuilt (different code path, different markets, lead-lag-aware classification): the book reconstructs faithfully here too.
4. **(class A) All six MARGINALs are July esports days**, failing on trade-in-spread (91.3–93.9% vs the 95% floor; 07-17 also dipped to 93.4% clean with 12.8% stale). Signature: very heavy in-play weeks (1.2–1.8M checkpoints/day, 3.6–4.4M L1 changes/day vs June's ~2–2.8M). Read: under in-play load the book lags the tape slightly more often; politics never dropped below PASS. Downstream analyses accept MARGINAL (labelled); nothing rests on the marginal slices alone.

### B1. Descriptive TOB replication on the new markets (class A)

Non-overlapping 5s episodes at extreme |TOB imbalance|, TEST week only, through the shared metric module — the same corrected claim the audit distilled from A13 ("conditional on the mid moving within 5s, it moves toward the imbalance"):

| universe | threshold | episodes | markets | conditional hit | zero-as-miss hit | zero-move share | directional bps [95% CI] |
|---|---|---|---|---|---|---|---|
| esports | dali 0.937422 | 333,791 | 1,123 | **71.5%** | 44.8% | 37% | **+136.5** [+124.7, +149.5] |
| politics_negrisk | dali 0.937422 | 124,732 | 253 | **68.4%** | 16.3% | 76% | **+26.1** [+17.9, +36.6] |
| politics_negrisk | per-asset q90 (train-fitted) | 105,851 | 150 | 57.8% | 13.1% | 77% | +7.5 [+3.6, +14.8] |

*Column notes: `conditional hit` = zero-move episodes excluded (the A13 reading); `zero-as-miss` = the Retest-C reading; `directional bps` = sign-weighted mean 5s mid move with market-cluster CI. The esports per-asset-threshold row is structurally absent: esports markets live hours (match markets), so train-day assets don't exist on test days — per-asset threshold transfer is impossible there and the fixed dali threshold is the honest out-of-sample read.*

**Read:** the corrected descriptive claim **generalizes** — on esports it is *stronger* than on dali's crypto holdout (71.5% vs 63%, +137 vs +58–82 bps), on politics it is present but small in bps and drowned by a 76% zero-move share (deep, slow books). This further supports the audit's Finding-1 correction: the signal was never dead; the old 73.7%→36.0% "collapse" was a metric artifact.

### B2. Taker cost floor per category — the class-A new-market measurement

Fee ground truth (class C): **`fee_rate_bps == 0` on ALL 3.29M trades in the window** (947,945 politics + 2,339,356 esports). These markets are fee-free AND rebate-free: the executable floor is the spread alone, and no rebate can flatter a maker result.

Spread/depth **at trade times** (executability-weighted; as-of join of each trade to the last fresh measured L1 ≤5s before it; 1.25M trades costed):

| universe | price bucket | trades | median spread | p90 spread | median spread (bps of mid) | median touch depth |
|---|---|---|---|---|---|---|
| politics_negrisk | 20–80c | 78,455 | 1.0c | 2.0c | 180 bps | 479 |
| politics_negrisk | p>95c | 182,770 | 0.1c | 0.2c | **10 bps** | 535,000 |
| politics_negrisk | ALL | 387,774 | 0.1c | 1.0c | 34 bps | 10,151 |
| esports | 20–80c | 655,520 | 1.0c | 5.0c | 290 bps | 1,056 |
| esports | ALL | 860,829 | 1.0c | 5.0c | 299 bps | 1,004 |

**Read (per-closure consequence):** the floor here is an order of magnitude LOWER than dali's crypto-4h floor (which carried 0.07 taker fees and wider effective spreads — margins −500 to −2,150 bps). This is a genuine new-market finding. **But it does not reopen the taker thesis:** the measured 5s directional signal (+26 bps politics / +137 bps esports, B1) still sits below the mid-bucket round-trip floors (180 / 290 bps). The politics extreme-price books (10 bps floor) are the only cheap venue, and the signal there is proportionally tiny (mid moves are rare — 76% zero-move — and those books barely move by construction). Class-A closures (A14x/A16, Block I) **stay closed**; the new floor numbers are recorded for future gate design.

### B3. Class-B re-tests through the institutional fill engine (class A; UNVALIDATED-FILL-MODEL caveat on every row)

The A14c/A14h (post-at-mid continuation), A18 (join-the-heavy-touch fade-to-microprice) and P2-flavor (OFI-triggered touch join) framings, run as engine strategies through `mm_engine`'s latency-gated `FillSimulator` under all three coded queue models, on every 14-day-window slice; 1-contract orders, W=5s fill windows, per-asset q90 signal thresholds, blocked to non-overlap at max(H)=60s; book-measured forced-taker exits at H∈{5,30,60}s; fee-free per the captured schedule.

**Pooled 14-day result: 0 of 54 (universe × framing × queue-model × H) cells have a CI lower bound above zero; every cell's mean is negative, and in 53/54 the CI upper bound is also below zero** (the exception: esports touch_fade H=60 optimistic, CI hi +0.02c). Mean PnL cents/contract [market-cluster 95% CI] — the optimistic column IS the bracket's best case; Prob/RiskAverse differ only in the 3rd decimal or a handful of episodes:

| universe | framing | H | episodes | markets | optimistic leg | bracket spread |
|---|---|---|---|---|---|---|
| esports | mid_continuation (A14c/h) | 5s | 707 | 143 | **−1.76 [−2.01, −1.51]** | ±0.00 |
| esports | mid_continuation | 60s | 608 | 127 | −1.37 [−1.92, −0.88] | ±0.00 |
| esports | touch_fade (A18) | 5s | 286 | 83 | **−2.20 [−2.76, −1.70]** | −2.20→−2.26 |
| esports | touch_fade | 60s | 250 | 76 | −1.09 [−2.08, +0.02] | −1.09→−1.19 |
| esports | touch_fade_ofi (P2 flavor) | 5s | 120 | 52 | −2.20 [−3.22, −1.24] | −2.20→−2.25 |
| politics_negrisk | mid_continuation | 5s | 80 | 22 | −0.89 [−1.73, −0.46] | ±0.00 |
| politics_negrisk | touch_fade | 5s | 20 | 8 | −0.78 [−2.15, −0.18] | ±0.00 |
| politics_negrisk | touch_fade_ofi | 5s | 217 | 11 | −0.11 [−0.56, −0.04] | −0.108→−0.109 |

*(Full 54-row table: `data/analysis/csv_outputs/market_making/real_l2_remeasure/classb_retest_queue_bracket.csv`.)*

Politics nuance: TOB-triggered passive fills barely exist there (typical day: ~18k signals posted → 2 entry fills; touch_fade 0 on most June days — the books are too deep and trades too sparse for these framings to even *execute*), but the July week's OFI-triggered fills (a politics market cluster with real L1 turnover) delivered 217 episodes — enough for a CI, and it is still below zero. The closest-to-zero cell in the whole grid (politics touch_fade_ofi, −0.11c) is "statistically negative, economically ~zero" — nothing that resembles a reopen.

**Why the bracket barely bites (important methodological read):** fill counts are identical or near-identical across Optimistic/Prob/RiskAverse. At W=5s, passive fills are **sweep-driven** — the level ahead is consumed by the very trade that fills us (post-fill queue_ahead = 0 by construction), and cancel-attribution (the entire disagreement between the queue models) has almost no time to act. Consequence A: the queue-model uncertainty band, dali's stated reason these closures were assumption-grade, is empirically ~zero *for this framing class* — the negatives are not an artifact of the pessimistic end of the bracket. Consequence B: the adverse-selection mechanism is now visible directly — a maker joining the heavy touch on the signal side gets filled precisely when a sweep trades THROUGH his side, i.e., exactly when the signal's prediction is being violated. Win rates on filled episodes are 15–34%. The conditional drift (B1) is real, and the passive maker structurally cannot harvest it, because the fills anti-select against it.

### B4. A17-flavor feature-signal reading under a doubly-disjoint clean split (class A)

Thresholds fitted on train-markets × train-days; evaluation on the disjoint test-markets × test-days episodes:

| universe | feature | episodes | markets | conditional hit | directional bps [95% CI] |
|---|---|---|---|---|---|
| esports | tob_imbalance | 147,640 | 545 | **73.4%** | **+153.8** [+132.4, +174.4] |
| esports | micro_dev/spread | 147,091 | 543 | **73.3%** | **+152.2** [+131.1, +174.5] |
| esports | ofi_5s | 63,033 | 465 | 66.8% | +104.0 [+89.0, +120.0] |
| politics_negrisk | tob_imbalance | 37,456 | 126 | **73.2%** | +32.3 [+18.6, +48.8] |
| politics_negrisk | micro_dev/spread | 37,147 | 125 | **72.9%** | +31.1 [+17.9, +48.6] |
| politics_negrisk | ofi_5s | 97,958 | 158 | 50.2% | **−0.9 [−7.4, +4.4] — no signal** |

**Read:** A17's condemned calibration-table reading is now settled on real data, in the direction the audit suspected: **the features DO carry conditional signal** under a split that cannot leak (disjoint markets AND a 3-week-embargoed disjoint period). The exception is genuinely informative: L1 OFI carries nothing on politics (deep books where L1 flow is noise) while TOB state and microprice deviation are strong everywhere. The A17 *executable-taker deployment kill* is untouched — it stands on the cost floor (B2), here as on dali's markets.

---

## Per-closure verdicts

| closure | class (revision note) | verdict here | basis |
|---|---|---|---|
| Taker-continuation family (A14/A14b/d/f/g, A16) | A (documentary floor) | **stays closed** — new-market floor measured, signal still below it | B1 + B2 |
| Block I lead-lag | A | **stays closed** (not re-tested; no contradicting new-market evidence) | — |
| A14c / A14h maker-at-mid | B → | **confirmed-closed-on-real-data (upgraded to robust)** — negative, CI<0, entire queue bracket, latency-gated, fee-free, book-measured exits | B3 |
| A18 passive reversion-to-microprice | B → | **confirmed-closed-on-real-data (upgraded to robust)** — same; and the anti-selection mechanism is now directly observed | B3 |
| P2 passive fade (OFI flavor) | B → | **confirmed-closed-on-real-data** — CI below zero on BOTH universes (politics −0.11c [−0.56, −0.04], n=217; esports −2.2c); no tail cell reappeared | B3 |
| A17 feature-signal reading ("features carry no conditional signal") | B → | **REOPENED-AND-RESOLVED in the corrected direction: the features DO carry conditional signal** (except OFI-politics). Executable-taker kill unchanged. | B4 |
| Maker monetizability of the descriptive signal | open-by-default | **partially tested, negative for the passive-harvest mechanism** (touch_fade IS the maker-harvest framing — anti-selected, −2.1c). The **quoter-gate-input** variant (TOB/micro_dev feeding the NSQ skew/toxicity gate) remains **UNTESTED — now the best-motivated next experiment** given B4. | B3 + B4 |

## Banner / ledger / TODO edits made with this note

1. [[block_a17_lightgbm_findings]] banner: appended the clean-split resolution (features carry conditional signal on real L2; deployment kill unchanged).
2. [[block_a18_passive_reversion_findings]], [[block_a14c_maker_at_mid_findings]], [[block_a14h_maker_non_overlap_findings]], [[block_p2_reversion_findings]] banners: appended "confirmed-closed-on-real-data (2026-07-21 re-test through the coded queue bracket)".
3. [[pm_dali_workflow_revision_decision]]: Tier-3 marked EXECUTED with pointer here.
4. [[mm_vps_capture_setup]]: correction re missing universes + gap-sidecar deletion.
5. [[TODO]] § MM/dali: this re-measurement recorded; new task for the capture-pipeline fixes (universes + gaps).
6. **Recommended for Cowork** (their file): [[COWORK]] § dali line currently says "no reopens" — should read "no *edge* reopens; A17's feature-signal sub-claim resolved positive (features carry conditional signal) per [[dali_features_real_l2_remeasure_findings]]".

## Limits of this note

- The 14-day window is a pre-registered subset of the 33-day archive (compute-bound); the remaining 19 days are gate-scannable with the committed scripts at any time.
- No crypto-as-control universe exists in the capture (B0b) — dali's own market class is untested by this pass.
- Politics B3 TOB-framing cells are thin by the market's own structure (fills don't happen); the OFI framing reached power (217 episodes) and is also CI-negative. The near-non-executability of TOB-triggered passive entries on politics is itself the politics verdict for this framing class.
- Every B3 number inherits the unvalidated-fill-model caveat; Join-2 calibration is upstream of promoting ANY passive number, positive or negative, to a deployment decision.
- The maker-exit variants of A14c/h were not re-run (taker-exit convention only).

## Decision and next step

- **No trading-edge branch reopens.** The class-B passive/maker closures are now robust closures, confirmed on real out-of-sample L2 through the institutional fill engine.
- **One reading reopens and immediately resolves:** the dali-descended features (TOB state, microprice deviation; OFI on fast markets only) carry real conditional signal out-of-sample under a clean split. Their only plausible monetization channel — after taker (B2) and passive-harvest (B3) both fail on costs/anti-selection — is as **gate inputs to the NSQ inventory quoter** (skew/pull decisions, not entries). That experiment (Tier-3 outcome 1) is scoped, not run: it competes with Join-2 on the roadmap and its evaluation belongs in `mm_eval/cpcv.py` at the full institutional bar. Consult Cowork.
- **Capture-pipeline fixes** (universes + gap sidecars, B0b) should be scheduled — they are cheap and protect every future capture-based study.
