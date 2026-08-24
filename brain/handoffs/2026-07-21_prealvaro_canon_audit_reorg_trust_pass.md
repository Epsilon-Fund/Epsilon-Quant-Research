---
title: "Handoff — pre-Alvaro canon audit, pipeline trust audit, PM reorg, code pass"
created: 2026-07-21
status: closed
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
  - TODO
tags:
  - handoff
  - polymarket
  - audit
  - reorg
---

# Handoff 2026-07-21 — pre-Alvaro canon audit → pipeline trust audit → reorg → code pass

> Hub: [[COWORK]] · [[TODO]] · Deliverables: [[pm_prealvaro_canon_audit_findings]] · [[pm_prealvaro_pipeline_trust_audit_findings]] · [[dali_capture_status_rollup]]

## What ran (one session, four phases, branch `justin`)

1. **Phase 1 — canon classification** (commit `audit(polymarket): pre-Alvaro canon audit`): all 62 pre-Alvaro notes audited note+code, verdicts 16 CANON / 17 CLOSED-ROBUST / 29 HISTORICAL / 0 DEMOTED; banner + normalized frontmatter on every note; ledger shipped; eightdelta pointers on 8 block-flow notes; hub/glossary/TODO links.
2. **Phase 1b — pipeline trust audit** (operator-directed redefinition: cross-reference vs institutional pipeline, verify assumptions, snowball analysis, correct-or-condemn — recorded as a standing memory). Six adversarial sub-audits with fresh computations. Deliverable: [[pm_prealvaro_pipeline_trust_audit_findings]]. Commit `audit(polymarket): Phase 1b pipeline trust audit`.
3. **Phase 2 — reorg** (operator-approved map with tweaks; commit `reorg(polymarket): one PM root`): `midas/` → `polymarket/midas/`; `topics/prediction-markets/` + `sports-arb/` + `midas_audit.md` → `polymarket/archive/`; dali folder name kept (glossary rewritten to the true chronology: Midas → dali lineage → Block K split → MM/OD); canvases deleted; notebooks/data cwd-artifact deduped; `research/data/README.md` manifest pointer added.
4. **Phase 3 — code pass** (this commit): pysdk gateway pair resolved as **complementary, both stay** (child process + parent wrapper — documented in execution README); CLOB three-stack map + `_kernel/README.md` freeze doc; 9 HISTORICAL capture snapshots + 2 duplicate audits archived to `notes/dali/archive/` behind [[dali_capture_status_rollup]]; **tests green: execution 359, research 539**.

## Decisions made (and by whom)

- **Operator:** approved the migration map (keep `dali` name; everything PM under `polymarket/`; midas → `polymarket/midas/`); redefined the audit standard (classification insufficient — correct or condemn).
- **Audit verdicts (main findings, all evidence-backed):**
  - The **"73.7% → 36.0% OOS collapse" is a metric-mismatch artifact** — under consistent units the descriptive TOB signal replicated (~63% conditional OOS); closure stands on Retest A/B + [[block_a18_passive_reversion_findings]] execution grounds. Banners on A13/A0c-holdout/A15/A17 + external-OFI caveat corrected.
  - The **2026-06-10 sign fix never reached the data** + a **new, larger defect proven**: aggressor double-count + phantom complementary-token positions in the both-role explode → `closed_positions`/`traders`/`directionality` **condemned pending bundle-aware regeneration**; esports latency screen condemned; Join-2 market-screen directionality stays a preference, never a gate, until rebuilt. Midas bot verified clean.
  - **A17 regime confound confirmed** (Task-5.1 shape, timestamp-proven); deployment kill survives on the cost floor; calibration table condemned as evidence.
  - Kill-robustness: replay 99.29% checksum-clean (validated post-hoc, a check the dali era never ran); every pre-era fill proxy optimistic (kills die under their own best case; A18's −1.232c is an upper bound); fees capture-verified (no kill manufactured; **reopen trigger recorded**: if crypto up/down fees ever go to 0, rerun the fee-dominated A14 daily-crypto cells).
  - **No branch reopens.** Most kills strengthen.

## Blockers / open items (owner: Justin unless noted)

- **Regeneration thread** (chips spawned in-session): bundle-aware dedup in `build_closed_positions.py` → regenerate the table chain → quantify double-count magnitude → rebuild the esports screen. Until then, `traders.parquet` PnL *levels* for taker-heavy wallets and all `primary_style` composition claims are untrusted.
- Fossil flagged, untouched: `polymarket/midas/weather_tail_analysis/` contains a nested stale copy of a research script pointing at the retired `polymarket-copy/` layout (dead before and after the move); candidate for deletion in a future midas cleanup.
- The never-encoded capture-quality gate and the weather passive execution-mix live measurement remain the only legitimately open pre-Alvaro loose ends (both cheap, neither urgent; the weather loop should reuse Join-2 infra if ever run).
- Fill-model validation against real fills remains the Join-2 live loop's first-order job (now doubly motivated — it also caps every pre-era passive number).

## Where the evidence lives

Verdicts + per-note reasons: [[pm_prealvaro_canon_audit_findings]] (with a 2026-07-21 supersession banner). Corrections + condemned artifacts + action list: [[pm_prealvaro_pipeline_trust_audit_findings]]. Capture layer: [[dali_capture_status_rollup]]. Hygiene + graph audit clean at the 2026-07-16 baseline after every phase; alvaro-branch collision surface = only COWORK/TODO hunks (smart-merge per [[MERGE_PROTOCOL]]).
