---
title: "Cowork Orientation"
tags: [cowork, orientation, index]
created: 2026-05-27
purpose: Fast, token-cheap orientation for any Cowork session opening this repo
---

# Cowork Orientation

This file is the shared Cowork law: repo invariants, prompt discipline, and where things live. It is timeless and person-agnostic: it carries no dated thread status. Cowork is the **orchestration** agent. If you're Cowork and you're reading this, run Agent Bootstrap first so your local personal overlay is loaded, then obey this shared law. Every Cowork-authored prompt for an implementation agent (Codex *or* Claude Code) must explicitly redirect it to run its bootstrap and read `brain/CODEX.md` first.

## Agent Bootstrap (do this before anything else)

Run the Agent Bootstrap — canonical copy in [[VAULT_MAP]] § Agent Bootstrap.

## Active threads

Thread status is authoritative in [[TODO]] and [[VAULT_MAP]] § Active research branches; this file carries no dated status. **NOTE:** the former single "Block K" thread is **split into two strats** so prompts can target one without disambiguating — **MM** (market-making) and **OD** (options-delta). "Block K" remains the historical name for their joint research arc.

| thread | location | hub | source-of-truth task list |
|---|---|---|---|
| **copytrade** | `polymarket/research/` + `polymarket/execution/` (Midas) | [[COWORK]] § copytrade cluster below | `brain/TODO.md` § copytrade |
| **MM — Market-Making** | `polymarket/research/` | [[strat_market_making]] | `brain/TODO.md` § MM |
| **OD — Options-Delta** | `polymarket/research/` | [[strat_options_delta]] | `brain/TODO.md` § OD |
| **dali / Polymarket research lineage** | `polymarket/research/` (shared infra) | [[COWORK]] § dali cluster below | `brain/TODO.md` § dali |

`brain/TODO.md` is the authoritative live task list. Read it before suggesting next actions.

## Canonical strategy docs

These live in strategy subfolders under `polymarket/research/notes/`. Load on demand, not by default.

Links below are Obsidian wikilinks by note basename so the graph connects and any chat can navigate the
clusters. Note basenames are unique across the vault. The Obsidian-level map is [[POLYMARKET_BRAIN]];
the on-disk notes folder map is [[INDEX]].

**Block K is now split into two strats.** Shared origin: [[block_k_plain_english_synthesis]] (START HERE —
glossary + full arc) and [[block_k_maker_options_research]] (foundation). Target a strat by its hub:

**MM — Market-Making (active).** Hub: [[strat_market_making]].
- Economics & quoting: [[block_k1_maker_economics_findings]], [[block_k2_quoting_findings]], [[block_k2v2_findings]], [[block_k2v3_findings]]
- Maker-fill entry (bridge to OD Strategy A): [[block_kpeg_findings]], [[block_kpeg_robustness_findings]], [[block_kpeg_robustness_review]]
- Real-maker reality check: [[block_k5_findings]], [[block_k5_stress_findings]], [[block_k5b_findings]]
- Deployability / current standing: [[mm_deployable_cells_findings]] — de-biased gate passes 4 categories, but standalone is **sub-scale** (~$78/active day, ~90% in one grab-bag cell; crypto cells ≈ $0). Moat is scale/structure, **not speed** (defer Rust). **Recommendation (updated 2026-06-04): the durable layer is the MM execution/lifecycle (source-clean passive entry + carry-to-resolution), anchored on the politics NegRisk live loop; OD valuation did NOT independently clear and survives only as cautious sizing/selection. This supersedes the earlier "consolidate to OD-primary" framing — see [[strat_options_delta]] and [[od_strategy_a_realism_reaudit_findings]].**

**OD — Options-Delta (active).** Hub: [[strat_options_delta]].
- Methodology / realism guardrails: [[od_methodology_realism_audit_findings]]
- Vol mispricing & harvest: [[block_k6_vol_findings]], [[block_k6_strategy_a_static_hedge_findings]], [[od_strategy_a_v2_lifecycle_findings]], [[block_k6_kronos_vol_bakeoff_findings]], [[block_k7_findings]]
- Lead-lag & basis: [[block_k3_leadlag_findings]], [[block_k3v2_findings]], [[block_k3v3h_findings]], [[block_k3v3h2_findings]]
- Arb: [[block_k4_arb_scan_findings]]
- Directional falsifier: [[block_optd_ceiling_findings]]
- Cross-project hybrid with Binance daily momentum: [[2026-06-02_binance_momentum_polymarket_hybrid]] — CLOSED. Polymarket BTC/ETH binary overlay improved drawdown only by paying away too much CAGR/Sharpe; alpha sleeve did not beat [[STRATEGY_REFERENCE]] daily momentum baseline even under generous proxy edge assumptions.

**dali — research lineage / redesign trail.** Direct local microstructure-continuation branches were falsified in specific retests, but dali itself is not closed; it is the broader Polymarket research line that kept redesigning into Block K, MM, OD, and future Justin-directed Polymarket work. Key falsification/redesign anchors: [[block_p3prime_oos_findings]], [[block_a0c_holdout_retest_findings]].
- Signal-as-input framing: [[block_a1x_external_note_reconciliation]]
- Capture + replay status: [[block_a0_capture_status_smoke]], [[block_a0_capture_status_quick]], [[block_a0_capture_status_latest]], [[block_a0_capture_status_final]], [[block_a0b_capture_status_latest]], [[block_a0b_capture_status_final]], [[block_a0c_capture_status_latest]], [[block_a0c_capture_status_final]], [[block_a0c_crypto_roll_status_latest]], [[block_a0c_crypto_roll_status_final]], [[block_a0c_auto_final_summary]]
- A1 diagnostics + execution tests: [[block_a1_results]], [[block_a1_capture_audit_a0]], [[block_a1_capture_audit_a0b]], [[block_a1_methodology_audit]], [[block_a1_visualization_pass]], [[block_a11_plan_and_diagnostics]], [[block_a12_mlofi_findings]], [[block_a14_executable_taker_findings]], [[block_a14b_refined_exit_findings]], [[block_a14d_tight_spread_findings]], [[block_a14f_combined_findings]], [[block_a14g_exit_family_findings]], [[block_a14i_pyramiding_findings]], [[block_a15_tob_extensions_findings]], [[block_a16_binary_bet_findings]]
- Plot-gallery attachment index: [[polymarket_plot_gallery_index]]
- Maker headwinds: [[block_a14c_maker_at_mid_findings]], [[block_a14h_maker_non_overlap_findings]]
- Reversion / rolling-rank: [[block_p1_rollingrank_findings]], [[block_p2_reversion_findings]], [[block_a15b_decoupled_findings]]
- Key signal & ML: [[block_a13_tob_imbalance_findings]], [[block_a17_lightgbm_findings]] (full series: block_a14*/a15*/a16/a17)
- Dali setup + external context: [[dali_live_l2_capture_plan]], [[dali_tfi_baseline_results]], [[external_ofi_tob_l2_midfreq_strategy_research]], [[sign_convention_findings]], [[sign_convention_findings_a1]]

**copytrade (primary thread).** [[copytrade_relayer_implications]], [[copytrade_attribution_repartition_findings]], [[relayer_dig_findings]], [[block_b_reinterpretation]], [[block_e_lite_findings]], [[block_e_audit]], [[profile_domah]], [[phase5_design]], [[weather_ftc_state]], [[topics/prediction-markets/README|prediction-markets pipeline]].
- Execution + probes: [[midas/README|Midas README]], [[polymarket/execution/README|Polymarket execution README]], [[polymarket/execution/PLAN|execution plan]], [[polymarket/execution/CLAUDE|execution rules]], [[polymarket/execution/scripts/SMOKE_REAL|SMOKE_REAL]], [[polymarket/execution/tests/probes/WS_PROBE_FINDINGS|WS probe findings]], [[polymarket/execution/tests/probes/NEGRISK_FINDINGS|NegRisk findings]], [[archive/midas_audit|Midas audit]].
- Copy-execution audit artifacts: [[polymarket/research/data/analysis/cross_leader_synthesis_v2|cross-leader synthesis v2]], [[polymarket/research/data/analysis/domah_audit_report|Domah audit]], [[polymarket/research/data/analysis/domah_followups/family_heuristic_validation|family heuristic validation]], [[polymarket/research/data/analysis/domah_followups/leader_ee00ba_audit_report|leader ee00ba audit]], [[polymarket/research/data/analysis/domah_followups/politics_deep_dive|Domah politics deep dive]], [[polymarket/research/data/analysis/leader_dthreed8b71_investigation/dthreed8b71_strategy_profile|dthreed8b71 profile]], [[polymarket/research/data/analysis/leader_high_conviction/leader_high_conviction_audit_report|high-conviction leader audit]], [[polymarket/research/data/analysis/leader_negrisk_directional_1/leader_negrisk_directional_1_audit_report|NegRisk directional 1 audit]], [[polymarket/research/data/analysis/leader_negrisk_directional_2/leader_negrisk_directional_2_audit_report|NegRisk directional 2 audit]], [[polymarket/research/data/analysis/leader_top_leaderboard/leader_top_leaderboard_audit_report|top-leaderboard audit]], [[polymarket/research/data/analysis/leader_ultra_maker/leader_ultra_maker_audit_report|ultra-maker audit]].
- Copyability + directionality diagnostics: [[polymarket/research/data/copyability_candidates/copyability_metric_distributions|copyability metric distributions]], [[polymarket/research/data/directionality_classification/contamination_crosstabs|directionality contamination crosstabs]], [[polymarket/research/data/directionality_classification/directionality_metric_distributions|directionality metric distributions]], [[polymarket/research/data/directionality_classification/validation_candidates|directionality validation candidates]].

**Foundations / methodology.** [[polymarket/research/README|research README]], [[polymarket/research/CLAUDE|research rules]], [[polymarket/research/RESEARCH_FINDINGS|research findings]], [[dali_literature_synthesis]], [[dali_factor_construction]], [[dali_market_universe_screen]], [[block_a0_runbook]], [[historical_sign_convention_audit]], [[api_reconciliation_v1]], [[goldsky_incremental_freshness]], [[validation_report]], [[polymarket_csv_output_audit]], [[polymarket_data_manifest]], [[storage_consolidation_audit_2026_06_05]], [[polymarket_table_dictionary]], [[polymarket_plot_gallery_index]], [[esports_latency_arb_market_map]], [[polymarket/research/notes/overview/market_maps/esports_latency_trader_screen|esports latency trader screen]], [[polymarket/research/notebooks/README|Polymarket notebook index]], [[task5_trigger_conditions]], [[glossary]], [[docs/CRYPTO_DATA_MANIFEST|crypto data manifest]], [[newsletters/README|newsletter template index]], [[meetings/README|meeting template index]]. Latest cross-thread snapshot: [[2026-06-04_state_of_the_arc_and_novelty_frontier]] (**the single high-level map** — everything closed and how, what survives, the methodological limit that historical wallet-screening is *structurally blind to novelty*, and the live+novelty frontier; read this first to orient). Prior frontier doc: [[2026-06-03_politics_negrisk_live_loop]] (the OD/MM/dali arc converged: PM-binary pricing is efficient across crypto+equities, OD folds into MM, and the proven standing execution edge is the politics NegRisk structured non-top3 maker (2,290 bps, accounting-confirmed). Full live-MEASUREMENT-loop design; central test is whether politics adverse selection is dodgeable in a way crypto's wasn't — now best read as a *novel neutral-MM* test, since the historical winners are directional, not neutral). Phase-1 review: [[2026-06-03_politics_negrisk_phase1_review]]. Repo/CV context: [[2026-06-03_cv_handoff]]. Graph maintenance: [[2026-06-01_brain_audit]] → [[2026-06-04_obsidian_orphan_link_pass]]. Prior snapshots: [[2026-06-02_binance_momentum_polymarket_hybrid]] (CLOSED cross-project hybrid; daily momentum baseline beats PM binary hedge/alpha overlays on Sharpe and CAGR) and [[2026-06-02_od_negrisk_firstpassage_kickoff]] (the OD first-passage/barrier sub-branch — now closed efficient and folded into the convergence above). Prior: [[2026-05-31_kronos_hermes_eval]] (Kronos = gated forward-vol; Hermes = copytrade ops) → [[2026-05-30_maker_options_delta_pivot]] (older: [[2026-05-27_cowork_transition]]). Strategy-A static-hedge diagnostic: [[block_k6_strategy_a_static_hedge_findings]]; superseding lifecycle gate: [[od_strategy_a_v2_lifecycle_findings]]. Kronos stays gated off unless the embargo/capital assumption is explicitly reopened and an OOS design clears.

SpaceX IPO cross-market handoff: [[spacex_ipo_market_map_handoff]] maps PM, Hyperliquid, TradingView, proxy funds, and the agent build plan for the SpaceX overvaluation/EV thread. Companion coworker angle: [[spacex_ipo_coworker_addendum]] covers the new DOCX/PNG material on Class A vs Class B, `SPCX` vs `SPACEX`, Trade Republic, TradingView, and PCHIP distribution diagnostics.

> Older handoff used legacy web-chat draft names that are not committed here: `ML_for_Polymarket_Deep_Research`, `Polymarket_Execution_Research`, `Polymarket_Factor_Construction_and_Models`, and `Literature_Synthesis_and_Strategy_Layout`. The closest committed equivalents are [[dali_literature_synthesis]], [[dali_factor_construction]], and [[block_k_maker_options_research]]. If a future thread asks for one of the missing drafts, surface that fact rather than fabricating content.

### New notes in this cluster

- Every strategy note must be independently readable from a cold start. Start with the actual trade/research idea in plain English, then put internal labels like K5, KPEG, A14, Kronos, or Hermes in parentheses. Do not use code names as the only headline.
- Every new Markdown findings/results note must open with a concise `## Plain-English Summary` or `## Summary` immediately after hub/table-term backlinks. In 2-5 bullets or one tight paragraph, state what the note is about, why it was written, what data/experiment it covers, and the one-line takeaway/status before any results table or verdict.
- Do not manually hard-wrap prose mid-sentence or mid-list-item. Keep a bold label and its explanation on one logical line, or split the idea into separate sentences/paragraphs.
- If a note contains markdown tables, link [[polymarket_table_dictionary]] near the top and define or link every compact CSV column, bucket label, filter name, and indicator.
- If a table or diagnostic would be easier to understand visually, ask Codex to generate a Python chart and embed it in the note with a caption explaining axes, units, sample, and the plain-English read.
- MM notes belong in `polymarket/research/notes/market_making/` as `block_k<code>_findings.md` or `mm_<topic>_findings.md`; add `> Hub: [[strat_market_making]] · [[COWORK]]` near the top and add the note to the MM cluster above.
- OD notes belong in `polymarket/research/notes/options_delta/` as `block_k<code>_findings.md` or `od_<topic>_findings.md`; add `> Hub: [[strat_options_delta]] · [[COWORK]]` near the top and add the note to the OD cluster above.
- copytrade notes belong in `polymarket/research/notes/copytrade/` as `copytrade_<topic>_findings.md`, `profile_<leader>.md`, or `<leader>_audit_findings.md`; add `> Hub: [[COWORK]]` near the top and add the note to the copytrade cluster above.
- dali notes belong in `polymarket/research/notes/dali/` with `block_a<code>_findings.md`, `block_p<code>_findings.md`, or `dali_<topic>.md`; add `> Hub: [[COWORK]]` near the top and link them under `dali — research lineage / redesign trail`.
- Cross-branch explainers and plain-English summaries belong in `polymarket/research/notes/overview/synthesis/`; academic/deep research belongs in `polymarket/research/notes/overview/foundations/`; data-quality/methodology notes belong in `polymarket/research/notes/overview/data_quality/`; market maps/screens belong in `polymarket/research/notes/overview/market_maps/`. Link broad notes from [[POLYMARKET_BRAIN]]. Generated CSV result/report tables belong under `polymarket/research/data/analysis/csv_outputs/<cluster>/`, following [[polymarket_csv_output_audit]].

### Cowork prompt discipline

**This section is the single canonical implementation-prompt preamble** — [[CODEX]] points here; do not maintain a second copy elsewhere.

Every Cowork-authored implementation prompt must start with a context preamble that tells the agent to run the Agent Bootstrap first, then read `brain/CODEX.md`, `brain/TODO.md`, `brain/COWORK.md` **§ Active threads only**, `brain/POLYMARKET_BRAIN.md`, then the relevant strategy hub. The shared law still includes `brain/CODEX.md`; do not skip it because it is the implementation-agent README for the repo.

For data-heavy prompts, add one short line after the read-order preamble instead of listing raw folders manually: "For data artifacts, use [[polymarket_data_manifest]], [[polymarket_csv_output_audit]], [[polymarket_plot_gallery_index]], [[storage_consolidation_audit_2026_06_05]], and/or [[docs/CRYPTO_DATA_MANIFEST|crypto data manifest]] as applicable; do not relink raw shards one by one." This keeps prompts short while pointing Codex at the durable map.

For "find prior work on X" subtasks, instruct the agent to use the local **gbrain MCP tools** (semantic `search` + `traverse_graph`/`get_backlinks`) instead of reading hubs end-to-end — it indexes this vault and resolves `[[basename]]` links as a graph. Retrieval only; synthesis stays in-agent. See [[gbrain_retrieval_layer]].

Required preamble template:

```markdown
Before doing anything else, read:
1. Run the Agent Bootstrap (canonical copy in brain/VAULT_MAP.md § Agent Bootstrap): seed/read `local_agents/codex.md` from `brain/agents/templates/codex.local.template.md` if missing.
2. `brain/CODEX.md`
3. `brain/TODO.md`
4. `brain/COWORK.md` — § Active threads only
5. `brain/POLYMARKET_BRAIN.md`
6. The relevant strategy hub for this task
```

Optional data-artifact line:

```markdown
For data artifacts, use the relevant data/artifact manifests; do not relink raw shards one by one.
```

Prompt files should not be committed to the repo; prompts live in chat and outputs live as linked findings/results notes.

## Repo conventions (don't violate)

- `polymarket/` and the crypto-momentum work (`topics/`, root) are **independent projects**. Separate `pyproject.toml`, separate venv. Never cross-import.
- `polymarket/research/` uses **uv**, **DuckDB over Parquet**, no Postgres, no DB server.
- All metrics must be **lookahead-free** (filter by timestamp before aggregating).
- Parquet shards are **append-only**. Never edit in place.
- Addresses: lowercase, `0x`-prefixed. Source data is already canonical — never re-case.
- Run scripts with `PYTHONPATH=. uv run python …` from inside `polymarket/research/`.
- `pip install` is never to be run directly; use uv, or `--break-system-packages` only if explicitly necessary outside a venv.

## Cowork vs Codex split (current intent)

- **Cowork (this tool)**: strategic discussion, prompt drafting for Codex, interpretation of Codex outputs, updating living docs in `brain/` and `polymarket/research/notes/`.
- **Codex**: implementation, running long analyses, producing CSVs / findings docs / scripts.

If something Cowork is being asked to do can be settled with code, repo inspection, shell commands, notebooks, data queries, tests, file edits, or a Codex run, Cowork should not conduct the work directly. Its maximum useful contribution is to frame the question, define acceptance criteria, pre-register the test, and provide a copyable Codex prompt.

Cowork-specific features are separate from this boundary. Use Cowork for strategic discussion, interpretation, memory/brain updates, prompt design, and features only Cowork itself can perform; do not treat those as permission to do code-like research inside Cowork.

## Anti-patterns (carried over from handoff)

- Building infra before validating signal. Roadmap discipline is "fastest path to first dollar."
- Optimizing on insufficient data — respect Task 5 triggers.
- Adding ML when rule-based hasn't shown edge (the Briola caveat).
- Trusting promising results without confidence intervals.
- Confusing **forecasting accuracy** with **net-of-cost trading profit**.

## Decision rules to enforce

- Strategy candidates: ask the forcing question — "If price didn't move between entry and resolution, would I still make money?" → splits trade-the-price vs hold-to-resolution.
- Promising signal? → require cost-adjusted edge, not just R² or hit rate.
- New ML idea? → only after the rule-based baseline shows edge.

## The research loop (and its terminus)

The repeatable routine this repo has converged on:

1. Cowork reads the brain, forms an opinion, and drafts a **pre-registered** Codex gate prompt.
2. Codex runs it offline, writes a `*_findings.md`, updates the hub + `brain/TODO.md`.
3. Cowork interprets through [[CODEX]] § Realism calibration (statistical vs economic, borrowed baselines, power-as-assumption, capacity-as-assumption, assumption-vs-live ledger) and decides: CLOSE / enhance / reopen.
4. Repeat with an enhancement or a realism pass.

**Terminus = live.** When the only remaining unknowns are live-only — passive fill rate, queue position, adverse selection, real non-incumbent capacity, an unsampled loss tail — the branch has hit the **offline ceiling**. Do not keep re-running offline gates on questions only live data can answer. Either graduate to a minimal live **MEASUREMENT loop** (1-contract, instrumented, hard risk caps — not a trading system) or stop. The 2026-06-02 OD longshot-harvest re-audit is the worked anchor: positive per-contract edge offline, but fill/queue/capacity/tail are all live-only → live measurement loop, not a build.

When asked "what else can we check," apply the [[CODEX]] § Realism calibration **reopen filter**: prefer cheap **never-run** gates (e.g. a passive/reversion framing of a real-but-mis-framed signal) over reheating closures that died on robust grounds.

## Where to write things

The canonical table is [[VAULT_MAP]] § Where to write things — use it. Orchestration-role deltas only:

- **Codex prompts → paste inline in chat as a single copyable markdown code block (```` ```markdown … ``` ````), not saved as repo files.** Justin keeps them in chat history; the repo gets only the *output* of running the prompt (a `*_results.md` or `*_findings.md` in `polymarket/research/notes/`).
- Task list updates → edit `brain/TODO.md` directly; keep "done (recent)" pruned.

## Strategic state

Thread status is authoritative in [[TODO]] and [[VAULT_MAP]] § Active research branches; this file carries no dated status. Dated strategic snapshots live in `brain/handoffs/` (latest cross-thread map: [[2026-06-04_state_of_the_arc_and_novelty_frontier]]; the 2026-05-28 dali-falsification snapshot formerly inlined here is preserved in [[2026-06-10_relay_retirement_branch_model]] and [[TODO]] § dali).
