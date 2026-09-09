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

**Surface skills for the task (Sherpa).** As part of the bootstrap — and again whenever the task shifts — run the skill router on a one-line description of the work and load whatever it surfaces:

```bash
python3 tools/sherpa.py "<the task in a sentence>"
```

It returns the top-N installed skills, each with a one-line "use when", ranked by keyword + local-semantic match (offline; keyword-only if the local embedder is down). Cowork is the main surface for this — it's the reliable, agent-agnostic auto-surfacing mechanism and complements Claude Code's native description-triggering. Wrapper skill: `find-skills`. See [[VAULT_MAP]] § Agent Bootstrap step 6 and [[SKILL_MAP]] § Sherpa. (Surfacing a skill is orchestration support, not code-like research — it stays within § Cowork vs Codex split.)

## Active threads

> Rewritten 2026-08-25 for the market-making handoff. **One active thread.** `brain/TODO.md` is the authoritative live task list; [[TODO_ARCHIVE]] holds the pre-handoff history.

| thread | status | canon surface | tasks |
|---|---|---|---|
| **Market-making (Polymarket)** | **ACTIVE — the only live research thread** | [[strat_market_making]] → [[mm_model]] | [[TODO]] § Market-Making |
| copy-trading | deprioritised (paused pre-first-live-trade; shares live execution infra with MM) | folder banners | [[TODO_ARCHIVE]] |
| news-agent / calibration observatory | deprioritised (shipped; Justin-owned items) | [[strat_news_agent_showcase]] | [[TODO_ARCHIVE]] |
| valuation overlay (OD) · microstructure lineage (dali) · earlier MM eras (K) | **PARKED** — historical record, do not build on | PARKED banners on every note | [[TODO_ARCHIVE]] |

## Canonical strategy docs

> Cleaned 2026-08-25. The active project's canon surface is **[[strat_market_making]] (hub: both lanes, honest framing, reliability ledger) → [[mm_model]] (fundamentals vs additions)** — de-jargoned and self-contained. Detailed findings notes carry HISTORICAL EVIDENCE banners and back the canon with numbers; treat those numbers as preliminary per the hub's reliability ledger.

Operational references: [[mm_vps_capture_setup]] (live capture → R2), [[mm_engine_build_log]] (engine history/component map), [[MM_JOIN2_RUNBOOK]] (live machinery runbook), [[polymarket_data_manifest]] · [[polymarket_table_dictionary]] · [[METRICS_REFERENCE]].

Parked/deprioritised clusters (copy-trading, valuation overlay, microstructure lineage, earlier MM eras, the SpaceX IPO one-off) are indexed from [[POLYMARKET_BRAIN]] and marked by banners at note level. Do not navigate into them for active work; where a concept from them matters, it is explained inline in the canon surface.

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

**Claude Code `/goal` cap (4000 chars).** A `/goal` prompt handed to Claude Code is hard-capped at **4000 characters**. When the load-bearing context exceeds that, do NOT truncate the design out of the prompt — instead commit a **reference doc** (the full PRD/context, unlimited length) to `brain/handoffs/<date>_<topic>_prd_reference.md` and emit a ≤4000-char `/goal` that lists it as read-first item 2 (after the Bootstrap) and points to it for all detail. The `/goal` itself still lives in chat (not committed); the reference doc is committed like any findings/context note. Char-count the `/goal` before handing it over (`wc -m`). The `prd-scaffold` skill (`library/skills/prd-scaffold/`) is the tool for co-authoring the PRD and emitting the `/goal`. Worked instance: [[2026-07-07_mm_task5_prd_reference]] + its emitted `/goal`.

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

### Delegation discipline (efficient-fable, orchestration variant)

For token-heavy READING — vault scans, repo audits, multi-source gathering — Cowork applies the orchestration variant of the `efficient-fable` pattern ([[SKILL_MAP]] § Runtime efficiency skills): spawn parallel read-only subagents to gather, and keep judgment and synthesis local. Cowork subagents never edit files or run analyses — anything implementation-shaped still becomes a pre-registered Codex prompt per § Cowork vs Codex split above.

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
