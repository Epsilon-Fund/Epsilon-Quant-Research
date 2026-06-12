---
title: "Dali falsification archive + done-log (through 2026-05-29)"
created: 2026-06-10
status: archived
owner: justin
project: infra
para: archive
hubs:
  - COWORK
  - CODEX
  - TODO
tags:
  - obsidian
  - brain
  - dali
  - archive
---
# Dali falsification archive + done-log

> Moved out of [[TODO]] on 2026-06-10 to keep the master task list focused on open items (TODO had grown to ~72 KB / 433 lines, against its own "keep done pruned" rule). Nothing here is live work — it is the completed 2026-05-28 dali A1.x falsification record, the post-A14h/A1.7 strategic-state snapshot, and the recent-done log. The durable substance lives in the linked `*_findings` notes; the live dali backlog (A0c, A2, Blocks C–J, research gaps) stays in [[TODO]] § dali. See also [[2026-06-10_brain_audit]].

---

## Dali A1.x falsification record + strategic-state snapshot (2026-05-28)

### Block A1: Live OFI/Maker sniff test ✅ (2026-05-28)
A0/A0b replayed into `data/analysis/block_a1_features.parquet`; cost-QA diagnostics written to `notes/dali/block_a1_results.md`.

- [x] Live sign convention resolved on A0 + A0b
- [x] Capture audits confirmed clean
- [x] Batch replay wrapper added
- [x] Depth-normalized OFI diagnostics written
- [x] Maker proxy and corrected cost overlay written
- [x] A1.1 segment/L2-proxy diagnostics written to `notes/dali/block_a11_plan_and_diagnostics.md`
- [x] **A1.2 MLOFI** (2026-05-28). Per-level OFI for L1..L10 with integrated / depth-weighted / exp-decay variants. **Result: L1 CKS wins at 1s/5s/30s by 15-25pp; only 300s depth-weighted shows +1.27pp delta. Keep L1 as A2 baseline; optional MLOFI sidecar logging.** See `notes/dali/block_a12_mlofi_findings.md`.
- [x] **A1.3 TOB imbalance level deep-dive** (2026-05-28). **Headline: `tob_imbalance_level` hits 73.7% at 5s top decile (CI [67.6%, 77.7%]), beats L1 OFI's 64.1%.** Persistent, market-dependent flip times (sub-second crypto vs minutes geopolitics). OFI adds no incremental info above TOB. Conditional TFI lights up to 62.6% in high-TOB regimes at 5s. **Promote `tob_imbalance_level` to a primary A2 candidate.** See `notes/dali/block_a13_tob_imbalance_findings.md`.
- [x] **A1.4 executable taker QA** (2026-05-28). **DECISIVE NEGATIVE: 0/12 market-horizon cells positive after entering at ask + exiting at bid.** Mean PnL -1300 to -2150 bps. The mid-mid alpha is real as descriptive pattern but is NOT tradeable as a taker on this universe at 5s/10s horizons. Mean gap vs A1's mid-return overlay: -560.8 bps (overlay was over-optimistic by that much). See `notes/dali/block_a14_executable_taker_findings.md`.
- [x] **A1.4b refined-exit taker on TOB signal** (2026-05-28). 0/36 cells positive. Best config: `cfg_signal_reversal` (gains +325 bps mean vs A1.4) but still -1060 bps average. Take-profit configs UNDERPERFORM fixed-5s (right-tail clipping artifact). Confirms refined exit alone is necessary but not sufficient. See `notes/dali/block_a14b_refined_exit_findings.md`.
- [x] **A1.4c maker-at-mid simulation** (2026-05-28). **1/16 markets "maker thesis lives": btc-updown-4h-1779912000 with W=10s, H=30s, exit_symmetric_maker → +554.9 bps mean PnL at 9.0% fill rate (5s adverse selection 248 bps).** 5 markets fills-too-rare, 10 markets adverse-selection-wipes-rebate. **Decisive next test is queue+latency model.** See `notes/dali/block_a14c_maker_at_mid_findings.md`.
- [x] **A1.4d tight-spread-conditional taker entry** (2026-05-28). 0/198 cells positive at 5s/30s; 1 positive cell at 300s on same BTC-4h-1779912000 market (S=500 bps, +843 bps mean) but CI [-799, 1770] crosses zero. Spread filter alone doesn't rescue taker. Universe selection is necessary but not sufficient. See `notes/dali/block_a14d_tight_spread_findings.md`.
- [x] **A1.5 TOB extensions** (2026-05-28). Multi-level imbalance: L1 wins at 1s/5s/30s; `exp_decay_alpha_0p5` wins at 300s by +4.92pp. Micro-price-as-target: 97.3% hit at 1s — but contaminated by autocorrelation (signal and target share imbalance term). Micro-price-change-as-signal: underperforms TOB by ~20pp. Keep L1 TOB primary; exp-decay at 300s sidecar. See `notes/dali/block_a15_tob_extensions_findings.md`.
- [x] **A1.4f combined refined-exit + tight-spread** (2026-05-28). 5/660 positive cells, ALL on BTC-4h-1779912000 at 300s fixed-horizon, CI [-611, 1741] (crosses zero). Combination didn't surface new winners — just confirmed the same single-market result. See `notes/dali/block_a14f_combined_findings.md`.
- [x] **A1.4g exit-family exploration up to 300s** (2026-05-28). 0/165 cells crossed zero. **BTC-4h-1779912000's best smart exit (asymmetric TP3000/SL300) is -492 bps mean.** Smart exits KILL the BTC-4h fixed-horizon winner. Reveals that the BTC-4h "edge" only exists at 300s fixed hold, not with intelligent exits. See `notes/dali/block_a14g_exit_family_findings.md`.
- [x] **A1.5b decoupled micro-price target** (2026-05-28). OFI vs micro-price target: +5.45pp at 1s, +0.09pp at 5s, -2.14pp at 30s, -1.04pp at 300s. TFI degrades vs micro-target everywhere. **Diagnosis: original OFI/TFI/TOB signal is mostly mean-reversion to micro-price (fair value), not multi-second alpha.** See `notes/dali/block_a15b_decoupled_findings.md`.
- [x] **A1.6 binary-bet hypothesis under non-overlap** (2026-05-28). **0/225 cells cleared CI robustness bar.** Critically, the A14f BTC-4h winner does NOT replicate: `btc-updown-4h-1779912000` at fixed_300s top_decile goes from +844 bps overlap (A14f) to -1968 bps non-overlap (A1.6) — same market, same signal, same horizon. **A14f was an overlap-math artifact.** Regime-filter (Lipton-style) didn't help. See `notes/dali/block_a16_binary_bet_findings.md`.
- [x] **A1.4h maker-at-mid non-overlap retest** (2026-05-28). **Maker thesis dead.** Same BTC-4h cell that was +554.9 bps in A14c overlap math is -451.3 bps non-overlap, fill rate collapses 9.0% → 0.2%. 15 of 16 markets verdict "fills too rare even in best case." 0/192 robust positive cells. The A14c maker positive was the same overlap-math artifact as A14f's taker positive. See `notes/dali/block_a14h_maker_non_overlap_findings.md`.
- [x] **A1.7 LightGBM Tier 2 minimal pass** (2026-05-28). **No Tier 2 edge found.** 0/10 markets with positive ML mean PnL after non-overlap backtest with executable cost. **Diagnostic: probability calibration breaks at high confidence.** Model is well-calibrated through P=0.65-0.70 then systematically underperforms at P≥0.70 (gap up to -16pp). Consistent with A15b's mean-reversion diagnosis — extreme OFI events mean-revert most strongly at 5s. **This forecloses Tier 3 (DeepLOB) too** because the signal-to-PnL gap is structural, not architecture-shaped. See `notes/dali/block_a17_lightgbm_findings.md`.

### Strategic state (post-A14h + A1.7 — original direct local microstructure branch falsified)

**The original direct local microstructure continuation branch is falsified across all three tiers and both execution modes.** Dali itself is not globally closed; it is the broader Polymarket research lineage and redesign trail that later fed Block K/MM/OD.

| tier / mode | result |
|---|---|
| Tier 1 taker (8 angles) | Dead under non-overlap math. A14, A14b, A14d, A14f, A14g, A1.6 all converge. |
| Tier 1 maker | Dead under non-overlap math. A1.4h confirmed +554 bps A14c result was overlap artifact (collapsed to -451 bps + fill rate 9.0% → 0.2%). |
| Tier 2 (LightGBM) | No edge. 0/10 markets positive. Calibration breaks at high confidence (-16pp gap at P≥0.70). |
| Tier 3 (LSTM/DeepLOB) | Foreclosed by A1.7 calibration diagnosis. Signal-to-PnL gap is structural mean-reversion to fair value, not model-architecture-shaped. |

**Diagnostic synthesis across the negative results:**
- A15b: signal mostly predicts mean-reversion to micro-price (fair value)
- A1.6: overlap math systematically over-states deployable PnL; non-overlap kills positives
- A1.4h: maker fill rate is flow-capped (Polymarket has slow takers despite deep books), not competition-capped
- A1.7 calibration: high-confidence ML predictions UNDERPERFORM because extreme OFI is the most mean-reverting

**The TOB/OFI signal is REAL but it's structural mean-reversion to fair value on Polymarket's wide-spread universe. That's not a model problem; it's a market-structure problem.** No additional capture (A2), execution refinement (A14e queue+latency), or model complexity (Tier 3 DeepLOB) will change this conclusion.

**Salvageable lessons from dali (worth preserving):**
- Live sign convention infrastructure
- Capture + replay + analysis pipeline
- TOB imbalance as a STATE variable (not for direct trading) — could feed copytrade leader screens, fair-value calculations
- Polymarket-specific microstructure facts: exchange-internal-leg artifact, depth ≠ flow, calibration-breaks-at-extremes
- Methodology lessons: non-overlap math by default; Briola caveat is real and observed; overlap math is treacherous

**Pivot to copytrade primary. Three candidate directions:**

1. **Copytrade scaling + smoke deployment (RECOMMENDED).** Most shovel-ready. Infrastructure mostly built; smoke target identified; first $10 fill is operationally hours of work. Leverages 9 months of prior work. 2. **Block I — Cross-platform arb (Polymarket vs Kalshi vs options).** Different cost structure means signals that died on Polymarket might survive on Kalshi. Different domain class. 3. **Block J — Resolution-criteria LLM signal scanning.** LLM reads news + resolution criteria. Heaviest setup but potentially high-value if it works.

**A0c capture runs as scheduled** for data archival, but no further analysis budget on dali microstructure.

---

## done-log (recent, through 2026-05-29) — archived from TODO

## done (recent)

- [x] **External strategy-note triage** (2026-05-29, v2). Uploaded a 24-strategy OFI/TOB/L2 mid-frequency research note (external origin). Archived as `notes/overview/foundations/external_ofi_tob_l2_midfreq_strategy_research.md`; triaged in `notes/dali/block_a1x_external_note_reconciliation.md`. **Corrected framing (v2 supersedes a too-aggressive v1):** A1.4–A1.7 closed the *directional-continuation* use of the local signal (taker + maker-at-mid), NOT the signal — the 73.7% TOB hit (A1.3) is real. Three genuinely-untested items remain: **(a) continuous rolling-rank sizing vs decile gating** (no new data), **(b) explicit mean-reversion-to-microprice framing incl. a passive/maker reversion route** (no new data — the diagnosis is a strategy instruction we never followed; key open cell = can a passive route capture the reversion at a fill rate that survives non-overlap?), **(c) true-L2 features** (#5/#6/#7/#15, needs A2 capture). Plus **#21 cross-market lead-lag** (off-book, Block I). Do-first = (a)+(b) on existing A0/A0b/A0c replay; do NOT re-run continuation taker/maker-at-mid. Copytrade stays primary by default but (a)/(b) are a few hours of replay on data we own.
- [x] **A1.4 executable taker QA** (2026-05-28). 0/12 cells positive; mid-mid alpha is real but wiped by spread crossing. Strategic implication: maker thesis pivot or tighter-spread universe before A2. See `polymarket/research/notes/dali/block_a14_executable_taker_findings.md`.
- [x] **A1.3 TOB imbalance level deep-dive** (2026-05-28). `tob_imbalance_level` is the strongest A1.x signal: 73.7% hit at 5s top decile. Beats L1 OFI; OFI adds nothing incremental. Persistence is market-specific. Promote to A2 primary candidate. See `polymarket/research/notes/dali/block_a13_tob_imbalance_findings.md`.
- [x] **A1.2 MLOFI** (2026-05-28). L1 wins at <300s. Don't redesign A2 around per-level OFI. See `polymarket/research/notes/dali/block_a12_mlofi_findings.md`.
- [x] Copytrade-relayer implications (2026-05-28). Active wallet is in `maker`, not missing. No PnL/position rebuild. Domah smoke cell intact. Style labels biased toward maker. `operator_denylist.py` updated v1/v2-aware. See [`polymarket/research/notes/copytrade/copytrade_relayer_implications.md`](../polymarket/research/notes/copytrade/copytrade_relayer_implications.md).
- [x] Relayer-identity dig (2026-05-28). The two "relayer" addresses are Polymarket's legacy CTF Exchange v1 contracts (standard + neg-risk). See [`polymarket/research/notes/copytrade/relayer_dig_findings.md`](../polymarket/research/notes/copytrade/relayer_dig_findings.md) and [`polymarket/research/notes/copytrade/block_b_reinterpretation.md`](../polymarket/research/notes/copytrade/block_b_reinterpretation.md).
- [x] Block A0 (24h) + A0b (12h) live capture complete (2026-05-27/28). A0: 2,095 trades / 12 markets. A0b: 6,063 trades / 9 markets. Crypto 4h up/down + NBA in-game = volume-rich families. AI/product family still too thin to validate Block B finding.
- [x] Block B: historical TFI deep-dive complete (2026-05-27) — **reinterpretation pending; see dali / Reinterpretation below**
- [x] Historical sign convention audit complete (2026-05-27)
- [x] Block A0 smoke capture + 12-market shortlist locked (2026-05-27)
- [x] Dali infrastructure: market screen, TFI baseline, live CLOB capture, OFI replay, backtest engine
- [x] Obsidian vault consolidated into repo root (2026-05-27)
