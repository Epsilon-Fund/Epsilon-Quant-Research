---
title: "Polymarket Notes Index"
created: 2026-06-05
status: generated
owner: justin
project: polymarket
para: project
hubs:
  - COWORK
tags:
  - research
  - polymarket
---
# Polymarket Notes Index

> Hub: [[POLYMARKET_BRAIN]] · [[COWORK]]

This folder is intentionally split by strategy branch. Keep note basenames unique so Obsidian wikilinks resolve cleanly after files move between folders.

## Folders

- `overview/synthesis/` — cross-branch synthesis, plain-English explainers, and high-level maps.
- `overview/foundations/` — academic/deep research and external research libraries.
- `overview/data_quality/` — validation, reconciliation, freshness, trigger, data manifests, and methodology notes.
- `overview/market_maps/` — market screens and maps.
- `market_making/` — MM / maker notes: K1, K2, K5, K-PEG, real-maker dominance, and maker-entry mechanics.
- `options_delta/` — OD notes: K3, K4, K6, K7, basis, vol, hedging, longshot premium, and static-hedge Strategy A.
- `copytrade/` — copytrade notes: Domah, relayer work, Block B/E, Phase 5, weather FTC, and execution-facing research artifacts.
- `dali/` — dali / Polymarket research-lineage notes: A0/A1/A14/A15/A16/A17/P blocks, capture status, falsified direct branches, and redesign cues that fed Block K/MM/OD.

## Future Notes

Add a `> Hub:` backlink near the top of each new note and link it from the relevant hub in [[COWORK]] and/or [[POLYMARKET_BRAIN]]. Each strategy note must open with what the strategy actually is in plain English; code names like K5, KPEG, A14, Kronos, or Hermes can be secondary labels, not the only headline. Immediately after hub/table-term backlinks, add `## Plain-English Summary` or `## Summary` with 2-5 bullets or one tight paragraph explaining what the note is about, why it was written, what data/experiment it covers, and the one-line takeaway/status before any results table or verdict. If a note is broader than a single branch, put it in the appropriate `overview/` subfolder and cross-link the affected strategy hubs.

Generated CSV result/report tables belong under `polymarket/research/data/analysis/csv_outputs/<cluster>/`, not directly under `data/analysis/`. See [[polymarket_csv_output_audit]] for the current layout and cleanup rule, and [[polymarket_data_manifest]] for the family-level map of Parquet/CSV/JSONL/DuckDB artifacts.

Current broad market maps include [[esports_latency_arb_market_map]], [[polymarket/research/notes/overview/market_maps/esports_latency_trader_screen|esports latency trader screen]], [[spacex_ipo_market_map_handoff]], and its companion [[spacex_ipo_coworker_addendum]].

If a note contains markdown tables, link [[polymarket_table_dictionary]] near the top and define or link every compact CSV column, bucket label, filter name, and indicator. Generated plot galleries that need attachment-level graph links belong in [[polymarket_plot_gallery_index]].

If a table, bucket comparison, distribution, or time series is easier to inspect visually, generate the chart with Python, store new figures under `polymarket/research/data/analysis/plots/<cluster>/`, embed the image in the note, and explain the axes/units/sample beside it.
