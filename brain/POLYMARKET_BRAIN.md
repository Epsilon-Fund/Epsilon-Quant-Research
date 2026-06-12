---
title: "Polymarket Brain Map"
created: 2026-06-05
status: closed
owner: justin
project: infra
para: area
hubs:
  - COWORK
  - CODEX
tags:
  - obsidian
  - brain
  - infra
---
# Polymarket Brain Map

> Start here when you need the Obsidian-level map of the Polymarket project. For execution details, read the linked notes themselves.

## Why Wikilinks Matter

Wikilinks keep the research memory navigable. A finding is only useful later if a future Codex or Cowork session can discover its hub, sibling notes, and closure status without re-reading the whole repo. Obsidian then gives us a graph view of clusters, orphans, and broken links, which is exactly how we avoid repeating dead branches or losing the one note that explains why a strategy was closed.

## Core Hubs

- [[COWORK]] — strategic orientation, active clusters, and prompt discipline.
- [[CODEX]] — implementation orientation and required startup context.
- [[TODO]] — authoritative active task list.
- [[glossary]] — cross-project terms.

## Overarching Polymarket Docs

- [[polymarket/research/README|research README]] — copytrade/data infrastructure overview and reproduction guide.
- [[polymarket/research/CLAUDE|research rules]] — Polymarket research implementation conventions.
- [[polymarket/research/notebooks/README|research notebook index]] — bridge from Markdown notes to notebooks.
- [[polymarket/execution/README|execution README]] — Polymarket execution bot entry point.
- [[block_k_plain_english_synthesis]] — plain-English explanation of the Block K arc and the MM/OD split.
- [[block_k_maker_options_research]] — foundation research for maker/options-delta work.
- [[strat_market_making]] — MM hub.
- [[strat_options_delta]] — OD hub.
- [[od_methodology_realism_audit_findings]] — OD-specific realism audit: RV-model fair is physical-probability fair, not option-IV fair; PM implied vol is diagnostic only.
- [[polymarket/research/RESEARCH_FINDINGS|RESEARCH_FINDINGS]] — copytrade/data-side overview findings.
- [[METRICS_REFERENCE]] — authoritative metric formulas and caveats.
- [[dali_literature_synthesis]] and [[dali_factor_construction]] — academic/deep-research foundation for the dali / Polymarket research lineage.
- [[external_ofi_tob_l2_midfreq_strategy_research]] — imported external OFI/TOB/L2 research library and triage source.
- [[polymarket_csv_output_audit]] — CSV output layout and convention for generated result/report tables.
- [[polymarket_data_manifest]] — family-level manifest for Parquet, CSV, JSONL, DuckDB, and raw Polymarket data artifacts.
- [[polymarket_table_dictionary]] — shared definitions for compact table columns, bucket labels, filters, and indicators.
- [[mm_clob_capture_semantics]] — public PM CLOB capture semantics: anonymous L2, trade-print alignment, timestamp handling, and reconstruction limits.
- [[trade_anchored_spread_surface_findings]] — SPREAD-1: trade-anchored half-spread surface from /prices-history mid + OrderFilled; `/prices-history` confirmed a true book midpoint; pre-registered validation gate FAIL (Spearman 0.496 < 0.6, MedAE exactly 1.000c) with a measured diagnosis — trade-time vs time-averaged spread divergence in fast crypto (2.29c vs 4.00c) + 1-min mid staleness; slow-category level prior survives (MedAE ≤ 0.9c).
- [[spread_surface_tradetime_regate_findings]] — SPREAD-1b: the pre-registered successor gate — same frozen surface scored against the quoted half-spread as-of trade times (the quantity copy fills pay). **PASS on all three bars** (pooled MedAE 0.75c, fast-crypto 0.80c, 71.7% head-to-head vs flat-3c); the surface is a validated trade-time taker-cost prior for all categories except politics_negrisk (n=2); time-averaged consumers still must not use it.
- [[polymarket_plot_gallery_index]] — wikilinked index of generated Polymarket plot-gallery attachments.
- [[spacex_ipo_market_map_handoff]] — SpaceX IPO cross-market map for PM, Hyperliquid, TradingView, proxy funds, and agent handoff.
- [[spacex_ipo_coworker_addendum]] — companion note from the coworker DOCX/PNG covering Class A vs Class B, `xyz:SPCX` vs `vntl:SPACEX`, Trade Republic, TradingView, and the PCHIP distribution.
- [[spacex_pdf_construction_audit]] — methodology stress-test of the coworker PCHIP scripts: the multi-peak shape is an interpolation/differentiation artifact (not crowd belief); central stats (P(close>$135)≈80%, mean≈$167) are method-invariant, but shape/tail stats (mode, excess kurtosis +3.4→+0.8, P1/P99) are distorted; liquidity is unused. Reproduces the full original metric set under a liquidity-weighted lognormal and ships a drop-in replacement builder. Strategy framing intentionally out of scope.

## Strategy Folders

Folder index: [[INDEX]].

- `polymarket/research/notes/market_making/` — MM notes: K1/K2/K5/K-PEG, real-maker playbook, maker dominance.
- `polymarket/research/notes/options_delta/` — OD notes: K3/K4/K6/K7, basis, vol, static hedge, longshot premium.
- `polymarket/research/notes/copytrade/` — copytrade notes: relayer work, Domah profile, Block B/E, Phase 5, weather FTC.
- `polymarket/research/notes/dali/` — dali lineage notes: A0/A1/A14/A15/A16/A17/P blocks, capture state, falsified branches, and redesign cues that fed Block K/MM/OD.
- `polymarket/research/notes/overview/synthesis/` — cross-branch synthesis, plain-English explainers, and high-level maps.
- `polymarket/research/notes/overview/foundations/` — academic/deep research and external research libraries.
- `polymarket/research/notes/overview/data_quality/` — validation, reconciliation, freshness, trigger, and methodology notes.
- `polymarket/research/notes/overview/market_maps/` — market screens and maps.

## Falsification And Redesign Anchors

- dali is not globally closed; its original direct local microstructure continuation branch was falsified/redesigned via [[block_p3prime_oos_findings]], [[block_a0c_holdout_retest_findings]], [[block_a14h_maker_non_overlap_findings]], and [[block_a17_lightgbm_findings]].
- Single-venue Polymarket market-making is closed; surviving maker value is in [[block_k5_findings]], [[block_k5b_findings]], and the copy/learn route.
- Continuous/banded options-delta gamma scalp is closed; static-hedge Strategy A moved through [[block_k6_strategy_a_static_hedge_findings]] into [[od_strategy_a_v2_lifecycle_findings]], where the primary OOS lifecycle gate failed and the hedge overlay stayed gated.
- OD pricing-method caveat is now explicit in [[od_methodology_realism_audit_findings]]: old `fair` language often meant causal realized-vol physical probability, not external option-implied fair.
- Cross-project Binance daily momentum plus Polymarket BTC/ETH binary overlay is closed in [[2026-06-02_binance_momentum_polymarket_hybrid]]: the [[STRATEGY_REFERENCE]] daily momentum baseline keeps the best CAGR and Sharpe, while the PM hedge only improves drawdown by paying away too much return and the alpha sleeve fails even under proxy quote-edge assumptions.
- Latest politics NegRisk live-loop handoffs: [[2026-06-03_politics_negrisk_live_loop]] and [[2026-06-03_politics_negrisk_phase1_review]].
- Latest graph cleanup context: [[2026-06-04_obsidian_orphan_link_pass]].

## Prompt Rule

Every Cowork-authored Codex prompt must begin by telling Codex to read [[CODEX]] first, then [[TODO]], [[COWORK]], this map, and the relevant strategy hub. [[CODEX]] is the implementation-agent README; this rule keeps Codex from running on stale or partial context.
