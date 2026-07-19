---
title: "Pre-Alvaro Polymarket canon audit — verdict ledger for the dali/copytrade/foundations lineage"
created: 2026-07-19
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
  - canon
  - dali
  - copytrade
---

# Pre-Alvaro Polymarket canon audit — verdict ledger

> Hub: [[POLYMARKET_BRAIN]] · [[COWORK]] · Law applied: [[CODEX]] § Realism calibration + § Anti-patterns · Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

- This note is the **verdict ledger** for a full audit (2026-07-19) of every Polymarket research note written *before* the current MM Join-2 era: 44 dali notes, 12 copytrade notes, and 6 foundation/synthesis notes (62 total), plus the stray `topics/prediction-markets/` pipeline. Each note's conclusions were checked against the house realism-calibration law, and for headline claims the backing script was code-read (no re-runs).
- **Headline: the lineage is better than feared.** Every falsified branch was killed on robust grounds (large-magnitude structural negatives, non-overlap math, genuine OOS holdouts) — none by split noise or theatrical strictness. **Zero notes were demoted**: no pre-Alvaro note rests a positive conclusion on a methodology gap, because the positives were consistently disclaimed in-note and later discharged by proper kills.
- The real risk is **quotability, not conclusions**: several in-sample numbers (the 73.7% TOB hit rate, 64.1% OFI hit, TFI +7.74c, A14c +554.9 bps, Domah's +$4.0M) float in old notes without their kill-context and could be re-cited as live. This pass adds status banners to every audited note so a cold reader can't mistake a superseded snapshot for a live result.
- Verdict counts: **16 CANON · 17 CLOSED-ROBUST · 29 HISTORICAL · 0 DEMOTED.**
- Reorg input (Phase 2): `topics/prediction-markets/` is a vendor-API dead end with zero code coupling → archive; ~12 capture-status snapshots are HISTORICAL and collapsible; the two "A1 capture audit" notes are byte-identical duplicates of the final status notes.

## What was audited, and how

**Scope (pre-Alvaro = everything before the MM Join-2 / Alvaro-era market-making thread):** `polymarket/research/notes/dali/` (44 notes), `polymarket/research/notes/copytrade/` (12), `notes/overview/foundations/` (3: [[dali_literature_synthesis]], [[dali_factor_construction]], [[external_ofi_tob_l2_midfreq_strategy_research]]), the pre-Block-K-split synthesis notes ([[block_k_maker_options_research]], [[block_k_plain_english_synthesis]], [[codex_audit_phase1_results]]), and the stray `topics/prediction-markets/` collector pipeline. Current-era MM/OD/news-agent work was explicitly out of scope and none of it is demoted here.

**Method:** each note was read in full by an audit subagent; for every headline claim the backing script in `polymarket/research/scripts/` was opened and code-read (timestamp filtering, overlap handling, sign conventions, fee/fill assumptions). No heavy computation was re-run — this is a code-reading audit, not a replication. Final verdicts were assigned by the main agent; three subagent drafts were overridden (documented in § Overrides below).

**Verdict definitions** (with a practical example each):

| Verdict | Meaning | Example |
|---|---|---|
| `CANON` | Methodology sound, conclusion still load-bearing — safe to cite as-is | [[sign_convention_findings_a1]]: live trade `side` = token-side aggressor (99.9%/1.7% conditionals, n=847 classified) — every OFI/TFI sign downstream rests on it |
| `CLOSED-ROBUST` | Branch correctly falsified on robust grounds; **stays closed** (reopening is motivated reasoning per [[CODEX]] § Realism 5) | [[block_a14h_maker_non_overlap_findings]]: the A14c maker "+554.9 bps winner" collapses to −451.3 bps under one-position-per-market non-overlap |
| `DEMOTED` | Conclusion unreliable — a named methodology gap does the work | (none issued — see § The demotion that didn't happen) |
| `HISTORICAL` | Status snapshot / superseded artifact, no live conclusion | [[block_a0_capture_status_quick]]: mid-run checkpoint of a capture whose terminal note exists |

Every audited note now carries a **status banner** directly under its frontmatter (verdict, one-line reason, link back to this ledger), so the verdict travels with the note.

## Verdict counts

| Cluster | CANON | CLOSED-ROBUST | HISTORICAL | DEMOTED | Total |
|---|---|---|---|---|---|
| dali — capture/status/plan | 4 | 0 | 11 | 0 | 15 |
| dali — A1/A11–A14g findings | 4 | 6 | 3 | 0 | 13 |
| dali — A14h–A18/P/I/TFI/sign | 1 | 9 | 6 | 0 | 16 |
| copytrade | 4 | 2 | 6 | 0 | 12 |
| foundations + Block-K synthesis | 3 | 0 | 3 | 0 | 6 |
| **Total** | **16** | **17** | **29** | **0** | **62** |

Read: the center of gravity is HISTORICAL (snapshots, plans, superseded bridges) and CLOSED-ROBUST (sound kills). The CANON set is small and specific: capture data-quality records, the sign convention, the A1 methodology audit, the A13 descriptive signal (with caveats), the strategic triage note, the four rigorous copytrade notes, and the two Block-K origin docs.

## dali — capture/status/plan (15 notes)

These are data-plumbing artifacts (capture health snapshots, a runbook, two capture audits, one plan). No strategy claims, so the statistical realism bars mostly don't apply.

| Note | Verdict | One-line reason | Superseded by |
|---|---|---|---|
| [[block_a0_capture_status_smoke]] | HISTORICAL | ~22s liveness smoke, separate run dir | [[block_a0_capture_status_quick]] |
| [[block_a0_capture_status_quick]] | HISTORICAL | mid-run checkpoint (11 shards) | [[block_a0_capture_status_latest]] |
| [[block_a0_capture_status_latest]] | HISTORICAL | later checkpoint (24 shards) | [[block_a0_capture_status_final]] |
| [[block_a0_capture_status_final]] | CANON | terminal A0 data-quality record (25 shards, max gap 0.085s) — gates all A1 work | — |
| [[block_a0b_capture_status_latest]] | HISTORICAL | mid-run checkpoint | [[block_a0b_capture_status_final]] |
| [[block_a0b_capture_status_final]] | CANON | terminal A0b record | — |
| [[block_a0c_capture_status_latest]] | HISTORICAL | single ~98s placeholder shard | [[block_a0c_capture_status_final]] |
| [[block_a0c_capture_status_final]] | CANON (caveat) | terminal A0c record, but noisy: one 126s intra-run gap + 17 reconnects vs A0's 5 — users of A0c-based results should know | — |
| [[block_a0c_auto_final_summary]] | HISTORICAL | pure rollup/pointer, self-labeled non-verdict | — |
| [[block_a0c_crypto_roll_status_latest]] | HISTORICAL | single ~91s chunk | [[block_a0c_crypto_roll_status_final]] |
| [[block_a0c_crypto_roll_status_final]] | CANON | only record of roll-discovery health (77 discovery misses → roll-window coverage is not 100%) | — |
| [[block_a0_runbook]] | HISTORICAL | archived procedural runbook | — |
| [[block_a1_capture_audit_a0]] | HISTORICAL (duplicate) | byte-identical to [[block_a0_capture_status_final]] except one sentence — no independent gate applied | duplicate of A0 final |
| [[block_a1_capture_audit_a0b]] | HISTORICAL (duplicate) | byte-identical to [[block_a0b_capture_status_final]] | duplicate of A0b final |
| [[dali_live_l2_capture_plan]] | HISTORICAL (live caveat) | plan executed by A0/A0b/A0c; the anonymous-L2 caveat (public PM CLOB L2 carries no wallet/order-owner identity) is still live for any attribution work | — |

**Read:** the capture pipeline was healthy and honestly reported. Two findings matter beyond hygiene: (1) the "A1 gate" audits never encoded a pass/fail threshold — they are re-saves of the status notes, so if a capture-quality gate is ever wanted it still has to be written; (2) the audit script's gap math is correct (verified), but total reconnect downtime is only type-counted, never summed — inter-shard boundary gaps are the only quantified continuity measure.

## dali — A1 / A11–A14g findings (13 notes)

Data = one ~24h capture (A0) + one ~12h replacement (A0b), replayed lookahead-free (post-resolution rows dropped, forward values last-at-or-before — both verified in code). CIs are 200-draw block bootstraps over 300s blocks. **Two cluster-wide caveats:** observations are overlapping event rows (non-overlap math only arrives at A14h/A16), and decile thresholds are computed over the full capture (mild in-sample threshold lookahead). Both are harmless for negative verdicts and disqualifying for standalone positives — which is exactly how the notes ended up using them.

| Note | Verdict | One-line reason |
|---|---|---|
| [[block_a1_methodology_audit]] | CANON | accurate implementation audit that itself names the family's realism gaps; line-refs verified against the analyzer |
| [[block_a1_results]] | CANON (caveat) | sign convention established + honest pre-cost OFI surface; **the 64.1% hit rate is overlap-inflated IS description, never edge evidence** |
| [[block_a1_visualization_pass]] | HISTORICAL | rendering pass, no new numbers |
| [[block_a1x_external_note_reconciliation]] | CANON | the strategic triage that installed "falsified the continuation *framing*, not the signal" and spawned A18/Block I |
| [[block_a11_plan_and_diagnostics]] | HISTORICAL | diagnostic sidecar superseded by A13 + the A2 plan |
| [[block_a12_mlofi_findings]] | CLOSED-ROBUST | multi-level OFI loses to L1 symmetrically on identical rows; exact L1 reconciliation QA |
| [[block_a13_tob_imbalance_findings]] | CANON (caveat) | the cluster's real descriptive signal — but **73.7% is IS-only (replicated at 36.0% OOS, [[block_a0c_holdout_retest_findings]]), n≈300k is state-variable-inflated, and the move sits inside the spread ([[block_a15b_decoupled_findings]]); cite only together with the kills** |
| [[block_a14_executable_taker_findings]] | CLOSED-ROBUST | 0/12 cells positive under assumptions *generous* to the strategy |
| [[block_a14b_refined_exit_findings]] | CLOSED-ROBUST | 0/36; exit engineering cannot beat spread cost |
| [[block_a14c_maker_at_mid_findings]] | HISTORICAL | its positive cell was a queue-blind, fill-double-counting artifact — the clue that triggered A14h, where it died |
| [[block_a14d_tight_spread_findings]] | CLOSED-ROBUST | 6/198 cells in one market with CI through zero, correctly not claimed |
| [[block_a14f_combined_findings]] | CLOSED-ROBUST | 0/660 cells with CI-low > 0; stacking creates no winner |
| [[block_a14g_exit_family_findings]] | CLOSED-ROBUST | 0/165; final nail for taker-continuation |

**Read:** the falsification discipline here is genuinely institutional: kills are conservative (optimistic fill models, CIs honored, negatives never dressed up). The audit's practical output for this cluster is the caveat banners on A1-results and A13 so their in-sample positives can't escape their kill-context.

## dali — A14h–A18 / P / I / TFI / sign-convention (16 notes)

| Note | Verdict | One-line reason |
|---|---|---|
| [[block_a14h_maker_non_overlap_findings]] | CLOSED-ROBUST | non-overlap correctly killed the A14c artifact (+554.9 → −451.3 bps; fill rate 9.0% → 0.2%) |
| [[block_a14i_pyramiding_findings]] | CLOSED-ROBUST | K-cap sweep explains the artifact mechanism (positive only at K=∞ ≈ 153 concurrent positions) |
| [[block_a15_tob_extensions_findings]] | HISTORICAL | characterization only; its 73.7% anchor later failed OOS; the 300s "winner" is composition-confounded |
| [[block_a15b_decoupled_findings]] | HISTORICAL | the diagnostic (move = reversion to microprice, not continuation) that legitimately seeded A18 |
| [[block_a16_binary_bet_findings]] | CLOSED-ROBUST | 0/225 robust; refuses wide-CI positives |
| [[block_a17_lightgbm_findings]] | CLOSED-ROBUST | negative despite two pro-edge leakage channels; note: its "walk-forward" label overstates a single chronological split; ML-after-failed-baseline was run as a deliberate Tier-2 closure |
| [[block_a18_passive_reversion_findings]] | CLOSED-ROBUST | the strongest closure in the cluster: pre-registered pooled market-cluster CI gate, multi-capture (75 markets), conditional AND unconditional EV |
| [[block_i_leadlag_feasibility_findings]] | CLOSED-ROBUST | structural spread/fee kill with cluster CIs + assumption ledger; perp-OFI/sub-second explicitly left out of scope |
| [[block_p1_rollingrank_findings]] | CLOSED-ROBUST | 0/200 executable cells; the relative-sizing insight survives as a design input |
| [[block_p2_reversion_findings]] | HISTORICAL | fragile tail positive, correctly hedged in-note, falsified OOS by P3′/A0c; **if ever re-cited: its "bootstrap" CI is actually a Gaussian interval on n≈11 with no multiplicity control** |
| [[block_p3prime_oos_findings]] | CLOSED-ROBUST | pre-registered OOS bar on a genuine later holdout capture, discovery thresholds frozen |
| [[block_a0c_holdout_retest_findings]] | CLOSED-ROBUST | **anchor closure of the whole local-signal lineage**: 73.7% → 36.0% OOS (CI [32.8, 39.2]); discovery/holdout separation audited |
| [[dali_market_universe_screen]] | HISTORICAL | planning snapshot, superseded by the captures and then the MM thread |
| [[dali_tfi_baseline_results]] | HISTORICAL (hard caveat) | pre-A1 exploratory harness: CI-free, overlapping, unvalidated sign proxy — self-flagged in-note; **its +7.74c table must never be cited as a result** |
| [[sign_convention_findings]] | HISTORICAL (superseded) | superseded by [[sign_convention_findings_a1]]; carries a stale "not established" section beneath the update that establishes it — read the _a1 note instead |
| [[sign_convention_findings_a1]] | CANON | **the one durable live output of the dali cluster**: live `side` = token-side aggressor (99.9%/1.7%, n=847) — underwrites every downstream OFI/TFI sign and the MM thread's aggressor labeling |

**Read:** no dali branch was killed by theatrical strictness — every closure is a large-magnitude structural negative (adverse selection, spread/fee headwind, replication collapse), so per [[CODEX]] § Realism 5 all CLOSED-ROBUST entries stay closed. The one reopen-shaped candidate (the passive/reversion framing of the falsified continuation signal) was already run and closed as A18.

## copytrade (12 notes)

| Note | Verdict | One-line reason |
|---|---|---|
| [[block_b_findings]] | HISTORICAL | its operator-filter headline was an emit-path artifact (signs inverted on ~35–41% of fills); the surviving "no tradable TFI" claim is carried by the repartition note |
| [[block_b_reinterpretation]] | HISTORICAL | correct bridge note; every follow-up executed and absorbed by [[copytrade_attribution_repartition_findings]] |
| [[block_e_audit]] | HISTORICAL | coverage inventory, no live conclusion |
| [[block_e_lite_findings]] | HISTORICAL | self-archived; "relayers drive the lift" became circular once the addresses were identified as exchange contracts |
| [[copytrade_attribution_repartition_findings]] | CANON | **the house-standard note**: exact reconciliation anchors, Wilson + bootstrap CIs, paired composition control, explicit realism ledger, correctly declines a powerless OOS split |
| [[copytrade_relayer_implications]] | CANON | twice-falsified "invisible taker wallet" hypothesis; prevented an expensive wrong rebuild |
| [[copytrade_spread_surface_mtm_findings]] | CANON | SPREAD-2 terminal verdict — execution price is the copy killer (Domah +$1.14M at own prices vs −$429k as taker copy, identical fills) |
| [[copytrade_structural_directional_carriers_findings]] | CLOSED-ROBUST | taker-copy of carrier direction fails net-of-cost in all three sleeves with cluster-bootstrap CIs; minor gap: the replay script was never committed (reproducibility, not correctness) |
| [[phase5_design]] | HISTORICAL | sound design spec, since executed; its flat-3c fallback was superseded by SPREAD-2 with no verdict flips |
| [[profile_domah]] | HISTORICAL | honest whale dossier, but winner-screened by construction and its $4.0M is at-his-own-prices — the copy-relevant conclusions live in SPREAD-2 |
| [[relayer_dig_findings]] | CANON | decisive identification: the two biggest "trader" addresses are Polymarket's CTF Exchange v1 contracts — the pivot of the whole Block-B chain |
| [[weather_ftc_state]] | CLOSED-ROBUST | taker path killed by *observed* 5–7c slippage; the passive branch was correctly dispositioned as a live measurement loop (never run); in-note CI-free positives were retracted in-note |

**Read:** the 2026-06 chain (relayer_dig → relayer_implications → attribution_repartition → SPREAD-2) is the most rigorous work in the pre-Alvaro corpus. The weak era is 2026-05 (CI-free attribution tables, CI-free weather edges) — all of it since superseded or retracted. One systemic gap was mooted rather than resolved: per-leader audits are in-sample cell-mining on winner-screened wallets; SPREAD-2's negative made the question moot for taker-copy, but any future *maker*-copy revival must confront it.

## Foundations + Block-K synthesis (6 notes)

| Note | Verdict | One-line reason | Stale parts (now bannered in-place) |
|---|---|---|---|
| [[dali_literature_synthesis]] | CANON (with banners) | literature core (CKS/A-S/Glosten-Milgrom/Briola, PM-specific theory) still the reference | §5 status + §6 roadmap = executed history; §4.1 fee table stale; §4.3 priors are un-sourced and CI-free — label "prior, unverified" |
| [[dali_factor_construction]] | HISTORICAL | pre-data brainstorm whose strategy set was bypassed by the actual lineage | its §6 "Honest Assessment" Sharpe table is invented (no data, no CI) — must never be quoted as a result |
| [[external_ofi_tob_l2_midfreq_strategy_research]] | HISTORICAL | correctly self-archived strategy library | its caveat cited the falsified 73.7% as "the signal itself is real" — fixed to "IS-only, 36.0% OOS" in this pass |
| [[block_k_maker_options_research]] | CANON (with banners) | the theory framing + decision-spine (`net = rebate + spread − adverse selection − inventory risk`) still governs MM/OD | the 4 recommended validation tests are executed (K1–K4; the "$40M arb worth scoping" is resolved-negative by K4); rebate-cushion sections predate the fee=0/rebate=0 era |
| [[block_k_plain_english_synthesis]] | CANON (with banners) | glossary + K-block history + the two lessons are the shared DNA of ~25 downstream notes — whole-note demotion would be wrong | §0 pre-§8 TL;DR superseded; §2 fee/rebate bullets stale; §6 "Strategy A never tested with a static hedge" — since tested and gated negative/blocked; §8's +171 bps superseded by K5-STRESS numbers |
| [[codex_audit_phase1_results]] | HISTORICAL | closed execution log, correctly archived, scrupulously avoids CI-free positives | none misleading |

**Read:** nothing the ~25 downstream MM/OD citers typically pull from the Block-K docs (glossary terms, decision spine, history, lessons) is stale — the staleness is confined to forward-looking recommendation sections, so per-section banners (applied in this pass) are the right fix, not demotion. The single biggest staleness driver across both docs is the **fee-regime shift**: the Block-K-era 20/25% rebate-cushion story vs the current MM era modeling fee=0/rebate=0 on target categories.

## The demotion that didn't happen — and what replaced it

This audit was commissioned on the suspicion that pre-Alvaro conclusions were below institutional standard. The suspicion was **wrong about the negatives and right about the raw positives**:

- Every falsified branch died on robust grounds. Non-overlap math, genuine holdout captures, cluster CIs, and fees did the killing — all four "reopen filter" mechanisms from [[CODEX]] § Realism 5 (powerless OOS, borrowed baseline, single arbitrary cell, capacity-proxy cap) came back clean.
- Every headline positive was either disclaimed in its own note ("characterization, not tradeability"), retracted in-note (weather), or discharged by a proper downstream kill (A14c → A14h; P2 → P3′; 73.7% → A0c holdout).
- Therefore no note *rests* an unreliable conclusion on a gap → 0 DEMOTED. What the corpus needed instead was **context-locking**: banners that staple each in-sample positive to its kill, so numbers like 73.7%, 64.1%, +7.74c, +554.9 bps, and Domah's +$4.0M cannot be quoted naked. That is what this pass installed.

**Practical example of the risk this fixes:** a future prompt asks "did we ever find a TOB signal on Polymarket?" A cold gbrain search hits [[block_a13_tob_imbalance_findings]] and returns "73.7% hit rate, CI [67.6, 77.7], n≈300k" — three CI-passing-looking facts that are all IS-only, overlap-inflated, and inside-spread. The banner now forces the 36.0% OOS replication into the same field of view.

## Overrides (main-agent verdicts that differ from auditor drafts)

1. **block_a1_capture_audit_a0 / _a0b:** draft CANON-flagged → final **HISTORICAL (duplicate)**. The data is canonical, but it lives in the terminal status notes; keeping duplicate "audits" that add no gate invites citing an audit that never audited.
2. **sign_convention_findings:** draft DEMOTED → final **HISTORICAL (superseded)**. The underlying finding is sound and fully carried by [[sign_convention_findings_a1]]; the problem is a stale contradictory section, which the banner now flags. DEMOTED would wrongly cast doubt on a correct convention.
3. **dali_literature_synthesis:** draft DEMOTED → final **CANON (with banners)**. Same logic the auditor itself applied to the Block-K docs: the load-bearing content (literature) is sound; the stale sections are status snapshots, now bannered.

## Cross-cutting hygiene findings (feed Phase 2/3)

- ~11 capture-status snapshots + 2 duplicate "audit" notes are pure HISTORICAL clutter → collapse candidates (Phase 3 rollup per capture).
- Frontmatter is inconsistent across dali (full YAML vs tags-only vs bare `> Hub:`) — normalized to include `status:` where notes were touched in this pass.
- [[sign_convention_findings]] and [[sign_convention_findings_a1]] share an H1 title (navigation hazard).
- `topics/prediction-markets/` (surveyed separately): vendor-API (Falcon) pipeline, zero import coupling repo-wide, no data on disk, cannot run (no key), every capability superseded by on-chain successors in `polymarket/research/` — **archive** in Phase 2, nothing worth porting.
- The copytrade structural-carriers replay script was never committed — if that branch is ever revisited, the replay must be reconstructed from `domah_copy_audit.py`.

## External context: eightdelta

Several pre-Alvaro notes touch block-flow / large-trader effects. A separate project, **eightdelta** (outside this repo, at `~/Desktop/is this the bottom/eightdelta`, read-only), studies block-trade identification and underlying-impact with more institutional rigor (Deribit block trades; SEBI/Jane Street case study in `research/part1-sebi-janestreet/part1_foundations.md`; expiry-effects in `research/part3-expiry-analysis/part3_foundations.md`; data-QA discipline in `data/PROBE_FINDINGS.md` / `data/QA_FINDINGS.md`). Pointers were added in this pass to the block-flow-relevant notes ([[relayer_dig_findings]], [[copytrade_attribution_repartition_findings]], [[copytrade_structural_directional_carriers_findings]], [[profile_domah]], [[block_b_reinterpretation]], [[block_a1x_external_note_reconciliation]], [[dali_factor_construction]], [[external_ofi_tob_l2_midfreq_strategy_research]]). The two most reusable epsilon-side results for that context: exchange contracts masquerade as whales (the relayer-dig trap), and correctly-signed `_matchOrders` sweep flow *reverts* (CIs below 50% in 3 of 4 families).

## Limits of this audit (assumption ledger)

- **Modeled/checked:** note-vs-script consistency for headline claims; timestamp/overlap/sign/fee handling by code-reading; internal cross-note reconciliation (anchors like 7.89→5.67, 73.7%→36.0% verified consistent across notes).
- **Not done:** no computation re-runs, no data-file replays, no verification that committed CSVs match what scripts would produce today. A claim could in principle survive code-reading and still fail replication — nothing surfaced suggests this, but the audit's confidence is code-level, not output-level.
- Each note was read by one audit subagent (five in total, batched by cluster); the main agent cross-checked verdicts against sibling batches and the house law but did not independently re-read all 62 notes.

## Decision and next step

- **Gate outcome:** pre-Alvaro corpus certified — 16 CANON notes safe to cite, 17 robust closures that stay closed, 29 historical artifacts now bannered, 0 demotions.
- **Do not** re-litigate any CLOSED-ROBUST branch on the basis of this audit; the only legitimately open pre-Alvaro loose ends are (a) the weather passive-execution live measurement loop (designed, never run) and (b) the never-encoded capture-quality gate — both cheap, neither urgent.
- **Next:** Phase 2 repo reorganisation (migration map → operator sign-off → execute), using this ledger's HISTORICAL/duplicate/archive findings as input.
