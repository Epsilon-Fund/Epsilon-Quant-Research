---
title: "Brain audit — research-content + infra (2026-06-10)"
created: 2026-06-10
status: active
owner: justin
project: infra
para: area
hubs:
  - COWORK
  - CODEX
  - TODO
  - VAULT_MAP
tags:
  - obsidian
  - brain
  - infra
  - audit
---
# Brain audit — 2026-06-10

> Cowork-orchestrated audit run by two strong-model (Fable) subagents reading the vault, with every load-bearing claim re-verified against the files before action. Two passes: (1) research-content (mistakes / contradictions / stale claims / weak-grounds closures, bounded by the [[CODEX]] § Realism calibration reopen filter); (2) infra + cheap high-leverage brain edits. The cheap, verified fixes were applied the same day (see § Applied). The judgment-heavy research items and the deferred structural refactors are logged below for adjudication.
>
> **Model provenance:** both audit subagents were dispatched with the model override set to Fable, not the orchestrator's own model. The exact internal build string is not surfaced by the agent tooling; what is verified is the output — every checkable claim below was confirmed against the files.

## Applied 2026-06-10 (committed to the brain)

Factual fixes:
- `SMOKE_REAL` path corrected from the nonexistent `midas/scripts/SMOKE_REAL.md` to the real `polymarket/execution/scripts/SMOKE_REAL.md`, and the false "doesn't exist yet" claim removed — in [[TODO]] § copytrade exec, [[COWORK]], and [[2026-06-05_polymarket_research_onboarding]]. (On the critical path of the #1 live priority.)
- [[TODO]] frontmatter `status: closed` → `status: active` (it is the live master list; the wrong value would mislead status-filtering hygiene scans).
- [[OPERATING_RHYTHMS]]: stale `brain/GENERATED_INDEX.md` path corrected to `brain/generated/GENERATED_INDEX.md` (2 refs).

OD/MM primacy contradiction resolved (law files now match the findings):
- [[COWORK]] § MM bullet and [[TODO]] § MM "Conclusion" no longer say "consolidate to OD-primary / OD supplies the actual edge." Corrected to the current verdict: OD valuation did **not** independently clear and survives only as cautious sizing/selection; the durable layer is the source-clean passive MM execution/lifecycle anchored on the politics NegRisk live loop. Source: [[strat_options_delta]] (2026-06-02 synthesis, "this supersedes the older 'OD = the actual edge' reframing") and [[od_strategy_a_realism_reaudit_findings]].

gbrain retrieval wired into the bootstrap (it was referenced in only one file no agent reads, so it would have gone unused):
- [[CODEX]] § On startup checklist, [[VAULT_MAP]] § Deeper inspection, and [[COWORK]] § Cowork prompt discipline now point agents at the gbrain MCP tools (`search` / `traverse_graph` / `get_backlinks`) for "find prior work" before grep/folder scans. Retrieval only; synthesis stays in-agent. See [[gbrain_retrieval_layer]].

TODO prune (conservative):
- The 2026-05-28 dali A1.x falsification checklist + post-A14h/A1.7 strategic-state snapshot + the recent done-log moved to [[2026-05-28_dali_falsification_archive]], leaving pointers. [[TODO]] dropped from 433 lines / ~72.6 KB to 367 / ~61 KB. The open dali backlog (A0c, A2, Blocks C–J, research gaps) and all active-strategy sections were deliberately left intact.

## Research-content findings (verified; not yet acted on)

Headline: the findings notes are in good shape — the vault already ran its own reopen-filter pass ([[od_strategy_a_realism_reaudit_findings]]) on 2026-06-02, which caught most four-mechanism kills. So few legitimate reopens; the drift was in the orientation layer (above).

Reopen candidates that survive the filter (cheap to re-adjudicate):
- **Kronos/HAR forward-vol bake-off is a logically unfalsifiable deadlock.** [[block_k6_kronos_vol_bakeoff_findings]] is gated until the static-hedge far/late cell clears (n=11, CI [-1.04c, +3.99c] — [[block_k6_strategy_a_static_hedge_findings]]), which [[od_strategy_a_realism_reaudit_findings]] itself later called an arbitrary hard gate; meanwhile [[od_v4_calibration_gate_findings]] reopens only with a HAR/Kronos-corrected fair. Neither can fire. Cheap fix: add HAR-RV (and Kronos if assets suffice) as one more replacement fair in the existing [[od_replacement_fair_sensitivity_findings]] harness on the 23-fill panel — hours of repricing, no new data. (Honest caveat: EWMA-RV physical-prob fairs already fail per [[od_methodology_realism_audit_findings]]; expected lift is modest. The value is removing an unfalsifiable gate, not an expected win.)
- **Same-day Arm T touch-fade standalone close** rests on the 5% capacity assumption ([[od_same_day_crypto_pricing_gate_findings]]); at the historically observed 22.68% non-top3 ceiling the EV is 0.79c CI [0.52c, 1.23c] — clears. Already specced to be resolved by the live loop's fill-share logs ([[TODO]] § OD); just don't let the "CLOSE" label suppress that conditional. Real risk is the tail (zero observed adverse touches, ~96c loss-given-touch), not the capacity number.

Never-run cheap gates worth scheduling (highest information-per-dollar):
- **DSR/PBO/Monte-Carlo overfitting audit of the live crypto momentum book.** The 2.24 Sharpe is the max of a 400-trial-per-fold Optuna search ([[STRATEGY_REFERENCE]] § B.2), is traded live, and gates trust in every crypto verdict. Data owned, pure compute. Prompt drafted in [[2026-06-05_novelty_frontier_map]] (Prompt 1), never run.
- **NegRisk-basket consistency scanner + persistence** — the strongest deterministic PM edge, independently corroborated twice ([[mm_politics_negrisk_accounting_findings]] $34.6M gap; the $40M academic study in [[2026-06-05_novelty_deep_research]]). [[2026-06-05_novelty_frontier_map]] Prompt 5, never run.
- **Weather FTC live-look diagnostic** ([[TODO]] § copytrade) — 5 min/market manual observation discriminates execution models whose backtests span -427% to +3,048% ROI on the same cells ([[weather_ftc_state]]).
- **Block B on event-source-classified data** and the **`active_order_leg` flag in views.sql** ([[TODO]] § dali Reinterpretation) — both cheap, both decide whether copytrade style/cohort claims rest on artifacts.
- The other three [[2026-06-05_novelty_frontier_map]] prompts (robustness/overfitting, vol-managed overlay, funding-carry, Kalshi macro-vol throttle) have no findings notes.

Robust closures — DO NOT reopen (filter applied both ways): dali local microstructure all framings (non-overlap math, CI-decisive passive-reversion negative, broken ML calibration); K-PEG +759 bps (mark-to-mid artifact, realizable -753); single-venue neutral crypto MM (structural adverse selection -1,126 to -4,316 bps); PM financial-binary pricing efficiency (PM ≈ empirical across 5+ fairs); intra-PM arb ≈ 0; Binance-momentum × PM-binary hybrid (overlays dominated on the same CPCV paths). Anchors in [[POLYMARKET_BRAIN]] § Falsification And Redesign Anchors.

## Staleness for a human to adjudicate (not auto-edited)

- [[TODO]] § dali still carries 2026-05-29-dated A0c capture tasks as unchecked (`[ ] build shortlist / launch / audit / extended analysis`) although the capture ran and was consumed ([[block_a0c_capture_status_final]], used in [[block_a18_passive_reversion_findings]], [[block_p3prime_oos_findings]]). Left as-is — checking these off is a research-state call. Same for the duplicate "Launch A0c 24h capture" entry.
- [[strat_market_making]] hub § Open tasks still lists "decompose why top-3 dominate" (done in [[block_k5b_findings]]) and the SPX close-style collector (decided do-not-run in [[TODO]]). Hub open-task list lags its own body.
- Frontmatter `status: closed` also present on [[strat_options_delta]] while it is listed as an active thread — likely a 2026-06-07 backfill artifact ([[2026-06-07_brain_hygiene_cleanup]]).
- [[block_kpeg_robustness_findings]] frontmatter says "pending independent Codex review"; the review exists and is closed ([[block_kpeg_robustness_review]]).
- [[mm_deployable_cells_findings]] presents the $78/active-day run-rate with no in-note pointer to its Rule-3 demotion to "modeled assumption, not hard cap" in [[od_strategy_a_realism_reaudit_findings]].
- [[2026-06-03_politics_negrisk_phase1_review]]'s three "fix before Phase-2 smoke" items were verified landed in code (data-api fill polling, `amounts` resolution fix, `maker/cli.py`) but no note formally closes the review's fix list.

## Deferred structural edits (recommended, not yet done — higher breakage risk on 2-day-old files)

- **Bootstrap de-duplication.** The identical Agent Bootstrap block appears verbatim in 8 files (CODEX, COWORK, VAULT_MAP, ONBOARDING, START_RESEARCH_IDEA, codex_lane, cowork_lane, TODO). Pick VAULT_MAP as canonical, replace the rest with a pointer. Drift bomb; only in sync because it is new.
- **Triplicated "Where to write things" tables** (CODEX / VAULT_MAP / COWORK). Keep VAULT_MAP's (most complete), reduce the others to a pointer + role-specific deltas.
- **Make the law files timeless.** Delete the per-thread "Active threads (as of 2026-06-01)" / "Strategic state (2026-05-28)" prose from CODEX and COWORK and point to [[TODO]] + [[VAULT_MAP]] § Active research branches — removes the structural cause of recurring drift (three places claiming to describe current state). Note COWORK's 05-28 snapshot has no SPCX row though SPCX is TODO's top time-sensitive thread.
- **Add a gbrain re-index step to the weekly hygiene cadence** ([[OPERATING_RHYTHMS]] / the `brain-hygiene-weekly` task) so retrieval doesn't silently rot now that the bootstrap points agents at it.
- **Shorten the implementation-prompt preamble**: it forces a full read of COWORK (orchestration law) on implementation agents whose own law says "follow CODEX, not COWORK," and omits VAULT_MAP. Keep one canonical preamble in COWORK; CODEX points to it.

Do NOT bother (tempting but low value): renaming CODEX.md (breaks `[[CODEX]]` backlinks), reformatting COWORK's big Foundations link-dump (gbrain graph replaces it), deduping per-folder CLAUDE.md invariants (load-bearing for agents launched in-subfolder), an @-include refactor (infra-before-signal), cleaning the root `Untitled*.canvas/.base` and `.claude/worktrees/` leftovers (cosmetic, already ignored).
