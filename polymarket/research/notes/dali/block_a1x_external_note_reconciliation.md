---
title: "Reconciliation — External Strategy Library vs A1.4–A1.7 (what's tested vs what's still open)"
tags: [dali, reconciliation, external-research, strategy-triage]
created: 2026-05-29
revised: 2026-05-29 (v2 — corrected a v1 overreach: signal-is-real ≠ framing-failed)
inputs:
  - external_ofi_tob_l2_midfreq_strategy_research.md (the uploaded note being triaged)
  - block_a13_tob_imbalance_findings.md (73.7% hit rate — the real signal)
  - block_a14_executable_taker_findings.md ... block_a17_lightgbm_findings.md
  - block_a15b_decoupled_findings.md (mean-reversion-to-micro-price diagnosis)
  - block_a16_binary_bet_findings.md (overlap-vs-non-overlap kill)
status: analysis
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
---

# Reconciliation — External Strategy Library vs A1.4–A1.7

> Hub: [[COWORK]]

> Table terms: [[polymarket_table_dictionary]]


## Summary

This reconciliation corrects the first pass that over-closed the uploaded external OFI/TOB/L2 strategy library. It distinguishes tested-dead directional-continuation uses from genuinely untested framings and data-dependent ideas. The key correction is that A1.x falsified a continuation framing, not the existence of the TOB/OFI signal itself.

## Correction to v1

The first version of this note foreclosed ~18/24 strategies. That was an overreach. It made a logical
error worth naming so we don't repeat it:

> **"The simple directional framing A1.4 tested failed" was treated as "the signal is untradeable."**
> Those are not the same claim.

A1.x established two things that are *both true and narrower than v1 implied*:

1. **The signal is real.** A1.3: `tob_imbalance_level` hits **73.7% directional at 5s top decile**
   (CI [67.6, 77.7]). That is a genuinely high hit rate and it is not in dispute.
2. **One specific *way of using* it failed.** Every A1.4–A1.7 backtest used the signal as a **directional
   continuation entry** — go the way the signal points, capture a price move that has to exceed the spread
   you cross (taker) or survive a fill (maker-at-mid). That framing is dead. A1.5b/A1.7 explain *why*: the
   move the signal predicts is mostly **reversion to micro-price (fair value)**, and that target sits
   *inside* the spread, so a continuation taker pays more than the move is worth.

The honest gap: **we falsified the framing, not the signal.** Several things in the external note are
genuinely *different framings* of the same high-hit signal, and a few need data we don't have yet. Those
were wrongly buried in v1. Reclassified below.

## Three buckets (replacing the v1 "foreclosed" column)

- **TESTED-DEAD** — the specific framing was actually run in A1.4–A1.7 and came back negative. Don't re-run.
- **UNTESTED-FRAMING** — a real signal used a way we have *not* backtested. Genuinely open. Each still
  has to clear the same bar (executable cost + non-overlap), and I name the headwind so this isn't false hope.
- **UNTESTED-NEEDS-DATA** — requires true multi-level L2 / event-level capture we don't have. A1.2 used an
  L2 *proxy*, not true per-level book mutations (the external note flags this too), so "A1.2 killed MLOFI"
  was overstated.

## Verdict table (v2)

| # | Strategy | Bucket | Note |
|---|---|---|---|
| 1 | Rolling-rank L2 imbalance **sizing** | **UNTESTED-FRAMING** | A1.x gated on the *top decile* (binary in/out). Continuous percentile→[-1,+1] **position sizing** is a different exposure — it doesn't concentrate on the extreme, which A1.7 showed is the *most* mean-reverting. Genuinely novel here. Bar: must still clear executable cost; test continuous sizing vs decile gating directly. |
| 2 | Extreme normalized OFI momentum | **TESTED-DEAD** (as directional taker) | = A1.4 (0/12) + A1.4d (0/198). The *momentum/continuation* read is dead. |
| 3 | TOB filter + OFI trigger | **TESTED-DEAD** (as directional taker) | A1.4b refined-exit on TOB = 0/36; A1.6 non-overlap kill. Continuation framing dead. |
| 4 | Weighted-mid / microprice | **UNTESTED-FRAMING** (reversion anchor) | We tested it as *drift/momentum*. The diagnosis says micro-price is the **reversion target**. Trading the signal as "fade extreme → exit at micro-price" is the framing the data actually points to and was **not** cleanly tested. Headwind: target is inside the spread, so the per-trade edge is small — needs high hit rate (we have it) + cheap execution. |
| 5 | L1–L2 divergence | **UNTESTED-NEEDS-DATA** | Needs true L2. As a *regime/fade* signal (L1 says one thing, L5/L10 contradict) it's a different use than anything A1.x ran. Not foreclosed — uncaptured. |
| 6 | MLOFI linear score | **UNTESTED-NEEDS-DATA / partial** | A1.2 tested multi-level OFI on an L2 *proxy* for *directional continuation* (L1 won, deeper added little). True per-level OFI as a **reversion-strength estimator** is untested. |
| 7 | MLOFI PCA / integrated | **UNTESTED-NEEDS-DATA** | Same as #6; PC2 (touch-vs-deep divergence) is the interesting untested piece. |
| 8 | EWMA OFI | **UNTESTED-FRAMING (marginal)** | Tested only as continuation timing. As reversion timing, untested. Low expected delta. |
| 9 | Ask-depletion / bid-support decomp | **UNTESTED-FRAMING** | The *fade* side (fast replenishment ⇒ move fails) is what the diagnosis predicts and was never run as a primary strategy. Needs event-level data for the clean version. |
| 10 | OFI + TFI confirmation | **TESTED-DEAD** (as directional gate) | A1.3 + A1.4. Continuation framing dead. |
| 11 | Flow-toxicity / maker-trap filter | **UNTESTED-FRAMING** | Relevant *only* if a maker/reversion-via-liquidity framing is pursued. Real headwind: A1.4h showed maker fill rate is flow-capped (9.0%→0.2% non-overlap). |
| 12 | Fill-probability gated passive | **UNTESTED (= deferred A14e)** | This is the queue+latency model. It's the natural execution layer for a *reversion-as-maker* strategy. Deferred, not dead. |
| 13 | Spread/depth regime-gated OFI | **TESTED-DEAD** (as directional taker) | A1.4d/f. Universe selection necessary, not sufficient — *for the continuation framing*. |
| 14 | Market-selection similarity scan | **METHODOLOGY (live)** | = our A0c ranking. Useful for any surviving framing. |
| 15 | L2 liquidity-vacuum / breakout | **UNTESTED-NEEDS-DATA** | Needs true L2 ladder depth. Untested. |
| 16 | Replenishment / resilience | **UNTESTED-FRAMING** | The diagnosis (extreme pressure reverts) *is* a replenishment claim; trading it explicitly (fade fast-replenish, ride slow-replenish) is untested. Needs event-level data. |
| 17 | Large-trade impact / reversal | **UNTESTED-FRAMING** | Reversal side is diagnosis-consistent and untested as a primary. |
| 18 | Queue-imbalance logistic | **UNTESTED-FRAMING** | A1.7 showed the *continuation* read miscalibrates at P≥0.70. A logistic predicting **reversion** from the same imbalance is a different (untested) target. |
| 19 | Clustered OFI (ClusterLOB) | **UNTESTED-NEEDS-DATA** | Needs event-level participant data. Heavy. |
| 20 | Order-event decomposition | **UNTESTED-NEEDS-DATA** | Needs add/cancel/trade event stream. |
| 21 | **Cross-market reference lead-lag** | **GENUINELY NEW (off-book)** | Binance/OKX BTC OFI → Polymarket crypto. Edge is "Polymarket lags the underlying," not local book→book. Mean-reversion diagnosis does **not** apply. Maps to Block I. Still the single strongest fresh bet. |
| 22 | Funding/OI regime + OFI | **PARTIAL-NEW** | External regime half is fresh; OFI-trigger half is the dead continuation read. Salvage the regime half into #21. |
| 23 | Uncertainty-weighted ML | **TESTED-DEAD (Tier 2, continuation)** | A1.7. A reversion-target ML is untested but low priority until a rule-based reversion edge exists (Briola discipline). |
| 24 | RL execution overlay | **UNTESTED (needs an alpha)** | Only useful once a validated reversion/lead-lag alpha exists. Then it's an execution tool, not alpha discovery. |

## What A1.x actually closed vs left open

**Closed (do not re-run):** the *directional-continuation* use of the local OFI/TOB/TFI signal, in both
taker (A1.4/b/d/f/g, A1.6) and maker-at-mid (A1.4c/A1.4h) execution. That program is genuinely dead and
the non-overlap math is the reason the apparent winners evaporated.

**Left open (the corrected list):**

1. **Continuous rolling-rank sizing (#1)** instead of decile gating. Different exposure to the
   reversion-heavy extreme. Cheapest novel test — runs on the data we already have.
2. **Explicit mean-reversion framing (#4 + #9/#16/#17 fade side).** The diagnosis is a *strategy
   instruction we never followed*: the signal predicts reversion to micro-price, so the trade is fade-to-
   fair-value, not ride-the-move. We only ever backtested ride-the-move (plus one `signal_reversal` exit in
   A1.4b, which is not the same as a primary reversion strategy). **This is the highest-value untested item
   that needs no new data.** Honest headwind: the reversion target sits inside the spread, so per-trade edge
   is thin — it lives or dies on (a) the high hit rate we already have and (b) execution that doesn't pay
   the full spread, i.e. a maker/passive route — which collides with the A1.4h fill-rate problem. So the
   real open question is narrow and concrete: **can a passive/maker route capture the reversion at a fill
   rate that survives non-overlap?** That specific cell was not tested.
3. **True-L2 features (#5,#6,#7,#15,#19,#20).** A1.2 used an L2 proxy for a continuation target. True
   per-level OFI, L1-L2 divergence, and liquidity-vacuum signals are uncaptured, not falsified. Gated on
   A2 capture spend — only worth it if (1) or (2) shows something first.
4. **Cross-market lead-lag (#21)** — off-book, unaffected by any of the above, maps to Block I.

## Recommended action (revised)

- **Cheap, no-new-data, do-first:** on the existing A0/A0b/A0c replay, run two things A1.x never did —
  (a) continuous rolling-rank sizing vs decile gating, and (b) an **explicit reversion strategy** (fade
  top/bottom-decile signal, target micro-price) under both taker and *passive/maker* execution with
  non-overlap math and CI bars. This directly answers "what do we make of the high hit rate." If both clear
  zero with CI, *that* closes the local signal honestly. If the passive-reversion cell clears, the thesis
  reopens.
- **Medium:** scope **#21 cross-market lead-lag** (timestamp-alignment feasibility first).
- **Do NOT** re-run any directional-continuation taker/maker-at-mid backtest — that's the part A1.x
  genuinely closed.
- **Sequencing vs copytrade:** copytrade stays primary by default, but items (a)/(b) above are a few hours
  of replay on data we own, so they're cheap to settle before committing more A2 capture budget.
