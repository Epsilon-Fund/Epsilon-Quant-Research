---
title: "Dali L2 capture status — one-page rollup of the A0/A0b/A0c/crypto-roll captures"
created: 2026-07-21
status: active
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
tags:
  - polymarket
  - dali
  - capture
---

> **PARKED (2026-08-25).** This research lineage is closed/parked. Any concept from it that the active market-making project needs is explained inline in [[strat_market_making]] / [[mm_model]] — do not build on this note. (An older audit banner below may say CANON/CLOSED-ROBUST/HISTORICAL; that classified the note as *history*, not as active work.)


# Dali L2 capture status — one-page rollup (A0 / A0b / A0c / crypto-roll)

> Hub: [[COWORK]] · Canon context: [[pm_prealvaro_canon_audit_findings]] · Integrity re-audit: [[pm_prealvaro_pipeline_trust_audit_findings]]

## Summary

- One page replacing the ~9 per-checkpoint status snapshots of the four dali live L2 captures (May 2026). The intermediate checkpoints and the two non-additive "A1 capture audit" duplicates now live in `notes/dali/archive/`; the four terminal data-quality records remain canonical and in place.
- Takeaway: all four captures completed and are usable; A0/A0b were clean, A0c was noisier (one 126s gap, 17 reconnects), and the crypto-roll capture had imperfect roll-window discovery (77 misses). A 2026-07-21 post-hoc checksum validation rated the replayed streams 87–99% integrity-clean per run.

## The four captures — terminal records (canonical, unchanged)

| Capture | Terminal record (read this) | Span / events | Data-quality notes |
|---|---|---|---|
| A0 (12-market shortlist, 24h) | [[block_a0_capture_status_final]] | 25 shards, ~1.31M events | clean: max inter-shard gap 0.085s, 5 connects |
| A0b (replacements v2, ~12h) | [[block_a0b_capture_status_final]] | 12 shards, ~981k events | one 0.707s gap; 3 markets resolved mid-capture |
| A0c (targeted holdout, 24h) | [[block_a0c_capture_status_final]] | 25 shards, ~2.6M events | **noisy**: one 126s intra-run gap, 17 connects — but the holdout retest ran on a0c_roll and gates staleness, so this noise does not contaminate the anchor closure (see trust audit) |
| A0c crypto-roll (rolling BTC/ETH/SOL 4h, 24h) | [[block_a0c_crypto_roll_status_final]] | 24 chunks, ~1.38M events | roll-discovery imperfect: 77 misses → roll-window coverage < 100%; lowest replay-integrity score of the four (87.1% checksum-clean vs 95–99%) |

Post-hoc integrity (computed 2026-07-21, a check the dali era never ran): price-change self-checksum over 9.11M checkpoints = **99.29% exact L1 agreement** pooled (a0 98.8%, a0b 96.4%, a0c 95.5%, a0c_roll 87.1%). Details and the known replay caveats (book lags exchange truth by an event burst; hourly-shard state resets) in [[pm_prealvaro_pipeline_trust_audit_findings]] § Kill-robustness.

## What was archived and why

`notes/dali/archive/` now holds the intermediate checkpoints (`*_smoke`, `*_quick`, `*_latest` for each capture), the auto-generated wrap-up (`block_a0c_auto_final_summary`), and the two "A1 capture audit" notes (`block_a1_capture_audit_a0`, `block_a1_capture_audit_a0b`) — the latter because they are byte-identical re-saves of the terminal status notes with no independent pass/fail gate (canon-audit finding). Every archived note carries its verdict banner; nothing was deleted.

## Read / decision

The capture layer is certified usable with the caveats above; no further work on it is planned. If a capture-quality **gate** is ever wanted (max-gap / min-events thresholds before trusting a capture), it still has to be written — the "A1 gate" never encoded one.
