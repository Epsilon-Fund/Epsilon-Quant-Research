---
title: "Dali workflow revision — what to close, what to fix, what to re-measure on real L2"
created: 2026-07-21
status: active
owner: justin
project: polymarket
para: resource
hubs:
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - polymarket
  - dali
  - audit
  - decision
---

# Dali workflow revision — what to close, what to fix, what to re-measure on real L2

> Hub: [[POLYMARKET_BRAIN]] · [[COWORK]] · Built on: [[pm_prealvaro_canon_audit_findings]] (classification) + [[pm_prealvaro_pipeline_trust_audit_findings]] (corrections/condemnations, with its § Methodology evidence-class ledger) · Data: [[mm_vps_capture_setup]], [[polymarket_data_manifest]]

## Plain-English Summary

- The canon + pipeline audits (2026-07-21) flagged a lot: a metric-mismatch headline, a never-propagated sign fix, a condemned position pipeline, an A17 regime confound, and a stack of hypothesis-grade "kills strengthen" claims. This note answers the natural follow-up — **"so should the whole dali workflow be revised?"** — by separating the *strategy program* (stays closed) from the *workflow/machinery* (revise) and from *one scoped empirical question* worth re-running on data we didn't have then.
- **Decision in one line:** fix five recurring workflow defects; then re-measure on real captured L2 with **liberty to reopen any closure that rested on an assumption rather than institutionally-backed evidence** — reopening is warranted, not motivated reasoning, when the closure's grounds were estimated/assumed (unvalidated fill model, estimated spread, class-A-provisional check) OR when the new data tests markets the closure never saw.
- **Correction to an earlier draft of this note (operator, 2026-07-21):** a blanket "no dali branch reopens" was wrong. The reopen filter ([[CODEX]] § Realism 5) protects branches that died on **robust** grounds — but the pipeline audit's own evidence-class ledger shows several dali closures died on **assumption** grounds. Those are reopen-eligible. Nothing here is treated as fact unless it has institutionally-backed methodology **on real data** — not assumed, not estimated.
- **Load-bearing factual correction (operator):** the new VPS L2 capture is on **different markets** (politics NegRisk, esports, culture, crypto-as-control) than the dali captures (A0/A0b/A0c crypto/geopolitics mix). So this is **not a replication** — it is fresh out-of-sample evidence, and any dali closure proven only on dali's markets is *untested* on these.
- **The load-bearing new fact (operator, 2026-07-21):** we now have **real, continuously-captured L2**. Spread, depth, and book state at any timestamp should be **read from the captured book, not estimated**. Estimating spread (the copytrade SPREAD-1/1b/2 surface; any reconstructed-mid proxy) is bad practice with unreliable results and is retired as a costing method wherever the real book covers the timestamp.
- Ordering: the copytrade-side **regeneration** (separately chipped — bundle-aware dedup) is where conclusions could actually flip, so it outranks this. The workflow fixes are mostly merge-and-codify off the `alvaro` branch. The re-measurement is a Cowork-scoped roadmap decision because it competes with Join-2, and **Join-2's fill-model calibration is upstream of trusting any passive number this would produce.**

## Tier 1 — sort each closure by the EVIDENCE CLASS it rests on (not "all stay closed")

The reopen filter only protects closures that died on robust, institutionally-backed grounds. Split the dali closures by what actually killed them:

**A — Robust / documentary → stays closed absent contradicting new-market evidence:**
- **Taker-continuation family** (A14/A14b/A14d/A14f/A14g, A16): killed by a **spread cost floor** with margins (−500 to −2,150 bps) an order of magnitude above any reconstruction error, on top of *documentary* (class-C) fee ground truth. Do not re-run to "check if taking works" — a cost floor is a cost floor. **BUT** the floor was measured on dali's markets; whether politics-NegRisk/esports books have the same floor is a *new-market* question Tier 3 legitimately measures (not a reopen of the taker thesis, a fresh measurement of a different book).
- **Block I spot-return lead-lag**: structural spread/fee headwind, every partition CI far below zero; documentary fee input.

**B — Assumption-dependent → REOPEN-ELIGIBLE if Tier 3 warrants:**
- **A14c / A14h / A18 / P2 passive & maker framings**: these are negative **under an unvalidated fill model** that the pipeline audit could only *argue* (class B) is optimistic. "Negative under an assumed fill model" is not a robust closure. A **validated** fill model on real L2 (Join-2's first-order job) can legitimately flip these — reopening is warranted, not motivated reasoning.
- **A17's calibration/feature-signal claim**: condemned as evidence (regime confound); the "features carry no conditional signal" reading is reopen-eligible under a clean split. (The *executable-taker* deployment kill stays in class A above.)
- **The descriptive signal's monetizability**: proven un-monetizable *for a taker paying the spread*; **never tested for a maker earning it**. Not a closure at all for the maker mechanism — open by default.

**Rule:** a Tier-3 result may move a branch from B to "reopened" with evidence, or confirm a B closure on real data (upgrading it to robust). It may not reopen an A branch without new-market evidence that directly contradicts the documentary floor.

## Tier 2 — REVISE the workflow (the audit exposed five recurring process defects)

These are not dali-specific; unfixed, they recur in every future capture-based study. Most are merge-and-codify off `alvaro`'s already-built machinery.

1. **One owned metric definition.** The 73.7-vs-36 disaster was two scripts defining "hit rate" differently (zero-move rows excluded vs counted-as-miss; overlapping vs non-overlap) with no owner. Fix: a single shared metrics module (hit rate, directional return, non-overlap episode count) imported by *both* discovery and any retest. No metric redefined at a call site.
2. **Mandatory capture-quality gate before analysis.** The old "A1 gate" encoded no thresholds and was a byte-identical re-save. Fix: adopt `alvaro`'s `mm_reconstruction_audit.py` + `mm_engine/book.py` checks (BBA-checksum clean/stale/mismatch classification, `capture_gaps.jsonl` interleaved as `note_gap`, ≤5s staleness gating in *every* consumer, trade-in-spread coherence, exchange-ts-primary ordering) as a **required pre-analysis gate** with pass thresholds. Nothing gets a "feature panel" until it passes.
3. **Read the book; do not estimate spread/depth.** With real L2 (below), quoted spread and depth are **measured** from the captured book at the timestamp, subject to the staleness gate. The estimated spread surface (SPREAD-1/1b/2, [[trade_anchored_spread_surface_findings]], [[spread_surface_tradetime_regate_findings]]) is retired as a costing method wherever the real book covers the fill — kept only as a labelled fallback for pre-capture history. (This is the operator's 2026-07-21 point, made a standing rule.)
4. **A data-defect fix is not done until derived artifacts are regenerated or condemned in place.** The 2026-06-10 sign fix shipped as a view-flag no builder consumed. Fix: fixes carry a regeneration step or an explicit in-place condemnation (as now done for `traders.parquet` et al.).
5. **Findings ship with committed reproduction scripts.** Class-A numbers (including this audit's own — see the chip) are provisional until a runnable, reviewable script reproduces them and asserts its anchor.

## Tier 3 — RE-MEASURE once, on real L2, as an MM-gate input (never a standalone edge)

Two things changed since dali ran, and both are the *specific* conditions the reopen filter says justify a never-run cheap gate rather than reheating a closure:

- **The binding constraint is gone.** Dali had ~3 days of capture and (correctly) declared it powerless for OOS. The VPS now captures **24/7 to R2** ([[mm_vps_capture_setup]]) — months of independent, less-concurrent data support the whole-market CPCV splits the MM era uses.
- **Real L2, so no estimated costs.** Spread/depth read from the captured book at fill time (Tier-2 rule 3), not the retired surface.

**Scope:** re-measure the dali-descended features — TOB imbalance, OFI, microprice-reversion — on the real L2, using the institutional harness (one metric definition, capture-quality gate, whole-market nested CPCV, **book-measured** spread/depth). Two legitimate outcomes, both valuable:
1. **MM-gate input** — the feature improves the NSQ toxicity/skew gate's OOS decision (subject to DSR/PBO/CI), or it doesn't. Null is a real result.
2. **Reopen** — the measurement contradicts a **class-B** closure (e.g., a maker framing that was only ever negative under an assumed fill model turns positive under book-measured costs on real markets, at the full institutional bar). Reopening the relevant section is then *correct*, and the finding note must say so plainly and re-open the branch in the ledger.

**Guardrail (not a cap):** an *edge claim* — "this makes money" — requires the full institutional bar on real data: whole-market CPCV, DSR/PBO/CI, book-measured costs, and a **validated** fill model (Join-2). Until the fill model is validated, a positive is "reopen-warranting candidate," not "edge." That is a standard-of-evidence bar, not a prohibition on reopening.

**Cowork-scoped roadmap decision, not an automatic run:** it competes with Join-2 for attention, and Join-2's fill calibration is upstream of trusting any passive number. Consult this note with Cowork before scheduling.

## Practical example (why Tier 3 ≠ reopening)

Dali asked "can I *take* on the TOB signal?" → no (cost floor, Tier 1, closed). Tier 3 asks a different, never-run question: "the TOB imbalance state is a real ~63%-conditional predictor of inside-spread drift — does feeding it to the NSQ *quoter's* skew/toxicity gate improve the gate's OOS decision on months of real L2?" A quoter earning the spread has the opposite cost structure to a taker paying it, so a signal that's worthless to a taker can still sharpen a maker's gate. Same signal, different mechanism, different cost sign — which is exactly the reopen filter's "never-run cheap gate" case, not a resurrection.

## Evidence status carried in from the audits

- Tier 1 closures rest on: documentary fee ground truth (class C, strong); replay integrity 99.3% (class A, provisional — reproduction chipped); fill-model direction (class B, hypothesis). The *closures* are robust; the *"a fortiori strengthen"* embellishments are hypotheses.
- Tier 3's premise (the ~63% OOS descriptive replication) is **class-A provisional** — if the chipped reproduction script disagrees, Tier 3's justification weakens and this note must be revised.

## Decision and next step

- **Closures:** re-sorted by evidence class, not frozen. Class-A/documentary closures (taker cost floor, Block I) stay closed absent contradicting new-market data; class-B/assumption closures (passive & maker framings, A17 feature-signal reading, maker monetizability) are **reopen-eligible** and Tier 3 is allowed to reopen them with institutional evidence on real data.
- **Workflow:** adopt Tier 2 (1–5). Items 1–3 are the implementation prompt paired with this note; item 4 is now house rule; item 5 is the reproduction chip.
- **Re-measurement:** Tier 3 is **proposed, not scheduled** — Cowork decides against the Join-2 roadmap. Do not run it before the copytrade regeneration and the reproduction-script chip land. Reopening a class-B branch is an accepted, expected outcome, not a failure.
- **Superseding trigger:** if the reproduction script contradicts the class-A numbers this note leans on, revise here first.
