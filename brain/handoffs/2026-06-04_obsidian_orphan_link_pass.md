---
title: "Obsidian Orphan Link Pass - 2026-06-04"
created: 2026-06-05
status: archived
owner: justin
project: infra
para: archive
hubs:
  - COWORK
  - CODEX
tags:
  - obsidian
  - brain
  - handoff
---
# Obsidian Orphan Link Pass - 2026-06-04

> Hub: [[COWORK]] · [[POLYMARKET_BRAIN]] · [[STRATEGY_REFERENCE]]

## Scope

This pass connected source Markdown notes and Jupyter notebooks through Obsidian wikilinks without rewriting notebook JSON. Hidden agent/worktree folders were excluded from the working graph because they duplicate scratch state rather than durable research memory.

## What Changed

- Added notebook index hubs for [[topics/README|research topics]], [[topics/momentum/README|momentum]], [[polymarket/research/notebooks/README|Polymarket research notebooks]], [[topics/regime-classifier/notebooks/README|regime classifier notebooks]], [[topics/prediction-markets/notebooks/README|prediction-market notebooks]], and the smaller testing folders.
- Linked crypto notebooks back to [[STRATEGY_REFERENCE]] through local README hubs.
- Added crypto artifact indexes for [[topics/momentum/outputs/README|momentum output reports]] and [[topics/momentum/strategies/README|momentum strategy artifacts]].
- Linked Polymarket notebooks back to [[POLYMARKET_BRAIN]], [[COWORK]], and the relevant strategy notes.
- Added [[polymarket_plot_gallery_index]] so generated plot-gallery PNGs stay reachable through wikilinks before any cleanup.
- Added [[polymarket_data_manifest]] for Polymarket Parquet/CSV/JSONL/DuckDB artifacts and [[docs/CRYPTO_DATA_MANIFEST|crypto data manifest]] for crypto/live-trading Parquet/pickle/CSV/JSONL artifacts.
- Added [[newsletters/README|newsletter template index]] and [[meetings/README|meeting template index]] for non-research PDF templates.
- Connected fresh handoffs, execution docs, and the maker module to their Polymarket hubs.
- Repointed the unresolved `block_j` wikilink to [[dali_literature_synthesis|Block J]] context.

## Final Audit

Excluding hidden/runtime folders (`.git`, `.obsidian`, `.claude`, `.agents`, `.vscode`, virtualenvs, caches, and `.tmp`), and resolving CSV links without treating every generated CSV as an Obsidian attachment node:

- Source graph scanned: **469 files** = 186 Markdown plus 283 non-Markdown attachment nodes.
- Non-Markdown attachment nodes: 141 notebooks, 132 PNGs, 6 HTML reports, 4 PDFs.
- Broken wikilinks: **0**.
- Orphan Markdown nodes: **0**.
- Orphan non-Markdown nodes: **0**.

## Operational Effect

- Future Cowork/Codex sessions can start from hubs and manifests instead of re-scanning notebooks, plots, Parquet shards, CSV piles, and runtime artifacts from scratch.
- Data-heavy prompts can stay short: point Codex at the relevant manifest and the strategy hub, then specify the actual research question.
- The graph now separates durable knowledge from generated artifacts: notebooks and human-readable reports are linked, Parquet/JSONL shards are family-documented, CSV findings support is directly indexed, zip/raw payloads are ignored, and bytecode caches are disposable.
- Robustness improves because new notes have hub backlinks, plain-English summaries, table-term links, and manifest destinations; fewer future notes should become orphans or undocumented result dumps.

## Cleanup Done

- Deleted 10 empty root `Untitled*.canvas` scratch files after explicit approval.
- Deleted six empty root `Untitled*.base` scratch files after approval.
- Deleted `output/Untitled.png`, the lone unlinked scratch image.
- Deleted `2026-06-04.md`, a zero-byte root daily-note shell.
- Deleted the zero-byte accidental absolute-path sidecar under `Users/`, then removed the empty `Users/` directory tree.

## Deletion Candidates

No deletion candidates remain from the audited Obsidian attachment set. The newsletter and meeting PDFs are non-empty templates and are now indexed rather than deleted.
