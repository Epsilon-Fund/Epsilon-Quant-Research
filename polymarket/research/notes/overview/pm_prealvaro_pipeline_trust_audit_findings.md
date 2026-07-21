---
title: "Pre-Alvaro pipeline trust audit — snowball analysis, corrected results, and condemned artifacts"
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
  - audit
  - pipeline
  - dali
  - copytrade
---

# Pre-Alvaro pipeline trust audit — snowball analysis, corrected results, condemned artifacts

> Hub: [[POLYMARKET_BRAIN]] · [[COWORK]] · Companion to (and correcting parts of) [[pm_prealvaro_canon_audit_findings]] · Law: [[CODEX]] § Realism calibration · Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

- This is **Phase 1b** of the pre-Alvaro audit, run to the operator's standard: not "classify the notes" but **verify the assumptions, cross-reference against the institutional pipeline (mm_eval/CPCV, queue models, capture-integrity checks, Task-5.1's regime-confound lesson), trace error snowballs through the pipelined chains (a→b→c), and correct or condemn results**. Six adversarial audits ran, several with fresh computations over the actual parquet artifacts. **Before citing anything here, read § Methodology and epistemic status — computed numbers are provisional (no reproduction scripts committed yet), and every mechanism argument is marked as hypothesis, not fact.**
- **Three headline findings.** (1) The lineage's most-quoted "collapse" — TOB 73.7% → 36.0% OOS — is a **false comparison between two different metrics**; under consistent units the descriptive signal *replicated* (~70–75% → ~63% conditional; +73 → +58 bps directional). The branch closure still stands, but on execution grounds, not the stated reason. (2) The 2026-06-10 sign-inversion fix **never reached the data**: every derived table on disk predates it, builders would reproduce the bias today, and a **new, larger defect** was proven — the both-role position explode double-counts every aggressor's matched orders and fabricates phantom complementary-token positions. (3) Task 5.1's regime confound **recurred exactly once** pre-era (A17's per-market chrono split — timestamp-proven: test slices end 3–27 min before expiry) — the deployment kill survives on the cost floor, but A17's calibration table is condemned as evidence.
- **No branch reopens.** Every kill survives adversarial stress on replay integrity, fill-model bias, fee regime, and split design — most *strengthen* (the fill proxies were optimistic, so the losses are understated). The corrections change stated reasons, scopes, and headline numbers — and condemn several **data artifacts** (not notes) pending regeneration.
- **Live-path actions:** the esports latency screen is materially wrong at its core mechanism (rebuild before any use); the active Join-2 market screen consumes a stale contaminated directionality table (fine as a preference, must not become a gate); regeneration needs bundle-aware dedup, not just the `active_order_leg` flag.

## Methodology and epistemic status of THIS audit — read before citing anything below

Every claim in this note belongs to one of four evidence classes. **Only class D is settled fact; classes A–C are provisional or hypothesis.** The operator's rule applies to this audit as much as to the work it audits: an unverified stress test proves nothing.

| Class | Meaning | Status |
|---|---|---|
| **A — COMPUTED, uncommitted** | A fresh computation was run in-session over on-disk artifacts (feature parquet, retest CSVs, trades shards) with stated inputs/thresholds. **No reproduction script was committed** — the numbers are as reliable as one unreviewed run. | **PROVISIONAL until a committed script reproduces them.** |
| **B — REASONED / HYPOTHESIS** | A mechanism argument (e.g., "queue-filtering selects more adverse fills, so the real EV is worse"). No measurement exists or can exist pre-live-calibration. | **Claude hypothesis, NOT fact.** Marked `[HYPOTHESIS]` in the text. |
| **C — DOCUMENTARY** | Direct quotation of code lines, config files, file mtimes, or git history. Verifiable by opening the referenced file. | Reliable as far as the file goes; interpretation may still be class B. |
| **D — REPRODUCED / PRE-EXISTING** | A number that reproduces an already-published artifact (e.g., the recomputation returning the official 36.0%) or that was already on the record. | Settled. |

**How each stress test was actually conducted (and its own weaknesses):**

- **Finding 1 (metric mismatch):** read both hit-rate constructions in source (class C), then recomputed both constructions on both samples from `block_a1_features.parquet` at the discovery threshold 0.937422 (class A). Internal validity check: the recomputation reproduces the official 36.0% on the exact retest universe (class D anchor) — that is the main reason to believe the rest of the table. Weaknesses: single unreviewed run; episode-counting method (same-sign ≤5s-gap runs) is one reasonable definition among several; per-family numbers not cross-checked by a second implementation.
- **Finding 2 (double-count):** builder code paths quoted (class C); the twice-represented-aggressor mechanism verified by scanning ONE shard (1.22M sibling rows, class A) — the *mechanism* is strong on that shard, but the **magnitude on final PnL levels is unmeasured** and the shard may not represent all eras (v1 vs v2 exchange). "CONTAMINATED-MATERIAL" for `traders.parquet` levels is therefore mechanism-certain, magnitude-unknown.
- **Finding 3 (A17 regime confound):** split timestamps computed from the feature parquet (class A); cross-market overlap likewise. The "kill survives on the cost floor" conclusion is class B reasoning over class D numbers (the note's own published deltas and means).
- **Replay integrity:** checksum comparison designed and run in-session (class A, 9.11M checkpoints). The 99.29% figure and the lead-lag explanation (76.8% match at +3 events) are one run's output; the alignment heuristic itself is a design choice that a re-implementation should confirm.
- **Fill-model bias:** almost entirely **class B**. The bias *directions* are mechanism arguments; the A18 "real EV plausibly −1.5 to −2.5c" range is an explicit `[HYPOTHESIS]` with no measurement behind it. The only class-C/A content is the quoted fill logic and the executed-fills spread stats.
- **Fee audit:** captured fee configs quoted (class C — the strongest ground truth in this audit); fee-share arithmetic over existing CSVs (class A); the A14 "would flip at fee=0" counterfactual is approximate (±100–200 bps, stated) and moot given the class-C fee reality.
- **Split audit (Block I / P3′ / phase5):** split code quoted (class C); "confound impossible by construction" is class B reasoning over that code.

**Every chain-trust verdict below means exactly this and no more:** "survived the checks described above, at the evidence class stated." TRUSTED ≠ certified true; it means no check this audit ran falsified it, and the identified biases point in the kill-preserving direction *per class-B reasoning*.

**Required follow-up for institutional standing (open):** commit the reproduction scripts for the class-A computations (metric-mismatch table, checksum harness, A17 split timestamps, sibling-row scan) so they can be re-run and reviewed; until then treat every class-A number as provisional.

## The dependency graph audited

```
dali chain:      captures (A0/A0b/A0c/A0c_roll) → ClobBook replay → block_a1_features.parquet
                 → sign convention → signal blocks (A1/A13/A15) → execution kills (A14x/A16/A17/A18/P/I)
copytrade chain: fills parquet → maker/taker semantics + operator denylist → closed_positions
                 → traders/directionality/copyability/cohorts → leader audits/profiles → copy verdicts
                 → (side-branch) esports latency screen; (live) MM sleeve gates + Join-2 market screen
```

An error at any left node invalidates everything to its right. Each node below got an adversarial pass; per-chain verdicts use TRUSTED / TRUSTED-WITH-CORRECTION / UNTRUSTED.

## Finding 1 — the "73.7% → 36.0%" collapse is a metric-mismatch artifact (corrected result)

**What was wrong.** [[block_a13_tob_imbalance_findings]] computed its 73.7% hit rate on overlapping event rows **excluding zero-move windows from the denominator** — of the cited n=299,864, only 59,778 rows (20%) ever entered the hit calculation. [[block_a0c_holdout_retest_findings]] (Retest C) computed its 36.0% on **non-overlapping events counting zero-moves as misses**. The two numbers are different quantities; comparing them and declaring "artifact_confirmed" was a false comparison. The retest's own surface CSV contained the tell — pooled `mean_directional_return_bps = +58.1` OOS vs +72.9 IS — and the note never surfaced it.

**Recomputed (fresh runs over `block_a1_features.parquet`, discovery threshold 0.937422):**

| sample | retest metric (non-overlap, zero=miss) | conditional metric (zero excluded) |
|---|---|---|
| discovery a0/a0b crypto_4h | 26.6% | 70.0% |
| discovery a0/a0b daily crypto | 20.2% | 74.4% |
| holdout a0c_roll (exact retest universe) | 36.1% (reproduces the official 36.0%) | 63.0% (≈1,800 independent episodes) |

Under the retest's own metric, discovery scores 20–27% — the holdout **beat** in-sample. Under the consistent conditional metric, the signal degraded ~7–10pp OOS — real, but nothing like a collapse.

**Corrected statement of the TOB finding (replaces the 73.7% headline):** at extreme top-of-book imbalance (|level| ≥ ~0.94), *conditional on the mid moving at all within 5s*, it moves toward the imbalance ~70–75% in discovery and ~63% out-of-sample; mean 5s directional mid-move +58–73 bps; ~5–6k independent in-sample episodes (not 299,864 — one persistent-state episode spans ~40 rows); 43–73% of windows have no move at all; the move is largely inside-spread reversion to microprice ([[block_a15b_decoupled_findings]]) and is not monetizable after spread, fees, and adverse selection.

**What survives and why the branch stays closed:** the closure never actually depended on Retest C. Retest A (taker execution, −2,341 bps pooled), Retest B (deep-book fade artifact-confirmed), and [[block_a18_passive_reversion_findings]] (4 captures, market-cluster CI, −1.232c [−1.631, −0.924]) kill the *executable* claims robustly. Capture-noise was also ruled out as an alternative explanation: Retest C ran on a0c_roll (not the noisy main a0c), gates staleness ≤5s on entry and exit, honors snapshot resyncs, and only 0.23% of its events fall within 120s of a connectivity gap.

**Chain verdicts:** anchor closure **TRUSTED-WITH-CORRECTION** (kill right, stated reason wrong); A13 **UNTRUSTED as headlined / TRUSTED as the corrected claim above** — note the irony that the descriptive signal is *more* alive than the old framing suggested, and still worthless to trade.

## Finding 2 — the sign fix never reached the data, and the position pipeline has a second, larger defect

**Background:** [[copytrade_attribution_repartition_findings]] (2026-06-10) proved `historical_to_aggressor()` was sign-inverted on the ~35–41% of fills that are `_matchOrders` internal legs, and shipped the `active_order_leg` flag in `sql/views.sql` (commit `e788e34`).

**2a — the fix is opt-in and nothing opted in.** Zero of the derived-table builders (`build_traders_table.py`, `build_closed_positions.py`, `build_traders_directionality.py`, `build_copyability_metrics.py`, `build_cohorts.py`) consume the flag; every derived parquet on disk predates 2026-06-10 (`closed_positions` May 9 → `traders` May 12 → `directionality`/`cohorts` May 16 → `copyability` May 18 → leader audits Jun 5). Re-running the builders today reproduces the contamination. (By contrast the dali/MM PnL caches — K5/K5-STRESS, spread surface, equities scope — filtered `EXCHANGE_INTERNAL_LEG` at build time and are clean.)

**2b — new defect (proven on-shard, previously unregistered):** the aggressor of every `_matchOrders` bundle appears **twice** — its wallet sits in `taker` on every sibling row *and* in `maker` on the internal leg (verified: 1,221,859 of 1,221,859 sibling rows on `trades_delta_shard3`). `build_closed_positions.py` explodes both roles with no dedup, so:

- **same-token bundles:** the aggressor's position size, cash flow, and PnL on aggressive orders are counted ≈ 2× (win rates and bps-metrics roughly survive; PnL *levels* and volumes do not);
- **cross-token (mint/merge) bundles** (49–76% of internal legs): the sibling rows land in the complementary outcome token, fabricating a phantom offsetting position — a pure directional aggressor **looks balanced/arb-like**, which is precisely what `traders_directionality`'s style features measure. The `phantom_position_score` failure mode, re-imported through the back door.

This **falsifies the scope claim** in the 06-10 note ("PnL/position attribution is unaffected by this flag — style framing only"): true of the label swap, false of the position pipeline. **Correct remediation is bundle-aware dedup in `build_closed_positions.py` (credit the aggressor from the internal leg only), then regenerate in order: closed_positions → traders → directionality → copyability → cohorts.** The `active_order_leg` flag alone is insufficient.

**2c — where the contamination does and does not matter:**

| Consumer | Verdict | Why |
|---|---|---|
| Copy/cohort verdicts (0/72 pass, carriers NOT-COPYABLE) | CONTAMINATED-BUT-IMMATERIAL | kills were execution-cost-driven; contamination could only have hidden a winner, not manufactured a kill |
| Domah smoke cell | CORRECTED at conclusion level | re-validated post-fix inside the 06-10 note (survives, sharpens) |
| MM non-politics neutral-gate kill | IMMATERIAL — stands **a fortiori** | PnL side used the internal-leg-clean K5-STRESS cache; the label side's bias pushes wallets *toward* arb_like, and even so sports_like found 0/504 arb_like |
| Composition claims ("80% of gross sits in two_sided wallets") | UNRELIABLE both directions | style shares rest on biased labels |
| `traders.parquet` PnL levels / volume ranks (taker-heavy wallets) | **CONDEMNED pending regeneration** | ≈2× double-count on aggressive flow |
| **Esports latency trader screen** ([[esports_latency_arb_market_map]] era artifacts) | **CONDEMNED — rebuild before any use** | its core fingerprint splits taker buys from passive maker fills, and aggressive `_matchOrders` snipes are mislabelled *passive* — it under-detects exactly what it hunts |
| **Join-2 market screen, screen 4** (`mm_join2_market_screen.py`, live thread) | Flagged | reads the May-16 directionality parquet as a *preference*; harmless as rank-jitter, **must not be promoted to a kill criterion before regeneration** |
| Midas bot | CLEAN | consumes none of these tables; `leader_rankings.parquet` was never built |

## Finding 3 — Task 5.1's regime confound recurred once (A17), verdict survives, one sub-claim condemned

A17's per-market chronological split put train = calm mid-window and test = near-expiry endgame for the dominant btc-4h markets — **timestamp-proven** (test slices end 27, 24, and 3 minutes before expiry), plus cross-market temporal leakage (one market's train overlaps another's test on the same underlying). Same shape Task 5.1 caught in the MM era. Consequences:

- The **deployment kill stands on the cost floor**: the ML *beat* both rule baselines by +1,805–6,922 bps on the big markets and still landed at −418 to −1,397 bps against a touch+fee floor of −139 to −8,319 bps. No regime-matched re-test moves that across zero.
- The **calibration table is condemned as evidence**: "well-calibrated to P≈0.70 then breaks" is exactly what a regime shift does to a probability model — it is not evidence the features carry no conditional signal.
- **Scope correction:** cite A17 as "no *executable* ML edge near the cost floor," never as "these features carry no signal." The settling re-test, if ever wanted: whole-capture split (train a0 markets → test a0b markets), per the Task-5.1 fix.

The other three pre-era split designs are clean — notably [[block_i_leadlag_feasibility_findings]] (whole-market split) and [[block_p3prime_oos_findings]] (cross-capture holdout with an anti-leak tripwire in code) **embodied the Task-5.1 fix before it was named**; the copytrade phase5 walk-forward is confound-immune by construction (one documented pro-edge leak — Cohort E's lifetime style join — verdict survives a fortiori).

## Kill-robustness results (replay, fills, fees)

**Replay layer — SOUND-WITH-CAVEATS.** The dali `ClobBook` mechanics are correct (snapshot=reset, deltas=absolute level replacement, BBA never mutates) — the current-era `BookTracker` literally wraps the dali class. A checksum validation the dali era never ran was run now: **9.11M delta checkpoints, 99.29% exact L1 agreement** (a0 98.8% … a0c_roll 87.1%; the institutional gate calls ≥95% a pass). Real distortions, all noise/lag not direction: the maintained book lags exchange truth by an event burst (BBA messages lead their deltas); hourly-shard state resets discard 13.3% of rows and restart rolling windows; zero integrity instrumentation existed (`capture_gaps.jsonl` was explicitly excluded from replay). Kills survive — their failure margins were cents against a ~0.1–2c reconstruction error. **Carried asterisk:** "no signal at ≤5s in fast markets" claims (especially a0c_roll) have residual risk that book lag smeared a real ms-scale effect — a known-unknown, not reopen-evidence.

**Fill models `[HYPOTHESIS-CLASS SECTION — mechanism reasoning, not measurement]` — every pre-era proxy sits at or beyond the Optimistic end of the institutional bracket** (that placement is documentary: the fill logic is quoted code), so *per the mechanism argument* live calibration should only push measured EVs *down*: every kill dies under its own best case. Specifics: A14c's overlap multiplicity alone was worth ~1,000 bps (that one IS measured — by A14h itself); A14h's exit proxy flatters PnL one-directionally `[HYPOTHESIS: missed maker exits are adversely conditioned — argued, not measured]`; **A18's −1.232c is an optimistic upper bound per the same argument** — a real queue filters fills toward large through-volume (= continuation = maximally adverse for a fade), so real conditional EV is `[HYPOTHESIS]` plausibly −1.5 to −2.5c at far fewer fills than the simulated 0.09%. None of these directions is validated against real fills; they carry the same unvalidated-fill-model caveat as everything else. The weather sticky/track-down bracket anticipated the institutional standard; the queue model narrows the prior heavily toward sticky (best sticky cell −427% ROI), with the live measurement loop still the correct disposition — run it, if ever, as an execution-mix measurement on the Join-2 infrastructure, never a profit test. Standing caveat inherited by *every* passive result in the repo, both eras: **no fill model has ever been validated against real fills** — that validation is the Join-2 live loop's stated first-order job.

**Fees — no kill was manufactured by fees.** The dali `FEE_BY_CATEGORY` table was **capture-verified**: the generated capture configs carry live per-market fee fields showing crypto up/down genuinely at 0.07 taker / 0.20 rebate in May 2026 and geopolitics fee-free; the MM era's fee=0 is a politics/esports-specific fact, not a correction. One material nuance: the A14 **daily-crypto cells are fee-dominated** (≈1,221 bps round-trip fees inside a −997/−1,017 bps loss) and would flip positive in a fee-free world — a world contradicted by the captured schedule. **Conditional reopen trigger (recorded here deliberately): if Polymarket ever zero-fees daily/4h crypto up/down, the A14 daily-crypto cells merit a rerun.** Everything else: rebates credited where real (kills conservative), A18 negative in every fee world, Block I fee share 16% of a −21.6c margin, copytrade evaluators model no fees (approval-flattering omission only).

## Consolidated chain-trust table

| Chain / node | Verdict | Correction issued |
|---|---|---|
| Captures → replay → features | TRUSTED-WITH-CAVEATS | book-lag + shard-reset asterisk on ≤5s fast-market claims; integrity checks now run post-hoc (99.3% clean) |
| Sign convention (live side semantics, [[sign_convention_findings_a1]]) | TRUSTED | consistent with MM-era labeling; unchanged |
| Historical position pipeline (closed_positions → traders → directionality → …) | **UNTRUSTED pending regeneration** | double-count + phantom-position defect; bundle-aware dedup required; regeneration order specified |
| A13 descriptive signal | TRUSTED as corrected claim | 73.7% headline condemned; corrected statement above |
| A14 taker kills | TRUSTED | fee-dominance nuance on daily-crypto cells + conditional reopen trigger |
| A14c/A14h maker kills | TRUSTED | fills optimistic → losses understated |
| A16 / P1 / P2 / P3′ kills | TRUSTED | P2 positives doubly condemned (CI mislabel + fill optimism) |
| A17 ML closure | TRUSTED-WITH-CORRECTION | scope narrowed to executable edge; calibration table condemned as evidence |
| A18 closure | TRUSTED | −1.232c is an upper bound; real EV worse |
| A0c holdout anchor | TRUSTED-WITH-CORRECTION | Retest-C framing condemned (metric mismatch); closure carried by Retests A/B + A18 |
| Block I closure | TRUSTED | OOS split ceremonial; all-partition structural kill |
| Copytrade correction chain (relayer_dig → repartition → SPREAD-2) | TRUSTED | 06-10 note's "PnL unaffected" scope sentence corrected by Finding 2b |
| Copy/cohort/carrier kills | TRUSTED | contamination direction strengthens them |
| Esports latency screen | **CONDEMNED (artifact)** | rebuild on regenerated tables before any use |
| MM non-politics neutral-gate kill | TRUSTED (a fortiori) | composition shares unreliable both directions |
| Live Join-2 screen 4 | flagged | preference OK, gate NO, until regeneration |
| Midas bot inputs | CLEAN | no contaminated consumer |

## Action list (concrete, ordered)

1. **Bundle-aware dedup** in `build_closed_positions.py` (credit aggressor from the internal leg only), then regenerate `closed_positions → traders → directionality → copyability → cohorts`. Only then re-read any composition/style claim.
2. **Rebuild the esports latency trader screen** on regenerated tables before anyone acts on it.
3. **Join-2 market screen:** keep screen 4 (directionality preference) demoted to preference; do not promote to a kill criterion pre-regeneration.
4. Banner/ledger corrections (done in this pass — see below).
5. Fill-model validation stays the Join-2 live loop's first-order job (already standing law in the MM thread; now doubly motivated).
6. Recorded reopen trigger: crypto up/down fee schedule → 0 ⇒ rerun A14 daily-crypto cells.
7. Optional (bounded, not urgent): quantify the double-count magnitude on `traders.parquet` PnL levels for the top-20 taker-heavy wallets after dedup.

## Corrections applied to Phase-1 outputs

The Phase-1 ledger and banners propagated the false 73.7%→36% comparison ([[pm_prealvaro_canon_audit_findings]] was written before this pass). Corrected in place: the A13, A0c-holdout, A15, A17, and attribution-repartition banners; the external-OFI caveat; the ledger's affected rows and examples; and the COWORK § dali line. Each correction points here.

## Limits of this audit

- Corrections were computed where artifacts were on disk (feature parquet, retest surface CSVs, executed-fills CSVs, trades shards); the double-count *mechanism* is proven on one shard (1.22M rows) — the *magnitude* on final PnL levels awaits the dedup regeneration (action 7).
- No fill model in either era is validated against reality; every passive-EV number in the repo is a modeled bound until the Join-2 measurement runs.
- The audit trusted the captured fee configs as ground truth for the May-2026 fee regime; if those `fd` fields were themselves wrong, the A14 fee analysis inherits it (no evidence suggests so).

## Decision and next step

Every pre-Alvaro branch **stays closed**; no reopens. The corpus's negatives survived the checks this audit ran (see § Methodology for what those checks were and their evidence classes) — the "a fortiori" strengthening claims are mechanism-based hypotheses, not measurements. What changed is the *record*: two headline numbers corrected (73.7% and the 36% framing), one scope claim falsified ("PnL unaffected"), one sub-claim condemned (A17 calibration), three data artifacts condemned pending regeneration, and two live-path flags installed. Next: Phase 3 (code efficiency pass), then the regeneration work (action 1–2) as its own thread when the operator schedules it.
