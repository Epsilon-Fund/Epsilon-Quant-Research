---
title: "Strategic Context — May 2026 Handoff to Cowork"
tags: [handoff, strategy, status, context]
created: 2026-05-27
status: handoff-snapshot
purpose: Capture strategic context at the point of switching from web-chat AI to Cowork integrated workflow
---

# Strategic Context — Handoff Snapshot

## What This Document Is

A snapshot of the strategic thinking, open questions, and outstanding work as of 2026-05-27, captured at the point of switching primary AI tooling from web-chat to Cowork. The repo state itself is authoritative for what has been done; this document captures *why* and *what's next* in ways that may not be fully encoded in other notes.

Other notes to read in conjunction:

- Legacy external draft: `ML_for_Polymarket_Deep_Research` — top-level research synthesis
- Legacy external draft: `Polymarket_Execution_Research` — execution architecture
- Legacy external draft: `Polymarket_Factor_Construction_and_Models` — strategy design with critical re-examination
- Legacy external draft: `Literature_Synthesis_and_Strategy_Layout` — academic foundations and block roadmap
- [[codex_audit_phase1_results]] — Phase 1 + Block B results
- [[block_b_findings]] — TFI deep-dive
- [[block_e_lite_findings]] — operator attribution
- [[historical_sign_convention_audit]] — sign convention resolution

> **Cowork note (added on intake):** of the wikilinks above, the four that resolve cleanly inside this repo are `codex_audit_phase1_results`, `block_b_findings`, `block_e_lite_findings`, `historical_sign_convention_audit` — all under `polymarket/research/notes/`. The "ML_for_Polymarket_Deep_Research" / "Polymarket_Execution_Research" / "Polymarket_Factor_Construction_and_Models" / "Literature_Synthesis_and_Strategy_Layout" names appear to be web-chat drafts; closest in-repo equivalents are `polymarket/research/notes/overview/foundations/dali_factor_construction.md` and `polymarket/research/notes/overview/foundations/dali_literature_synthesis.md`. See `brain/COWORK.md` for the resolved map.

## Core Strategic Framing

### Trade-the-Price vs Hold-to-Resolution

The project is primarily **trade-the-price** (short-horizon price prediction) rather than hold-to-resolution (terminal probability estimation). This choice was deliberate based on:

- Better feedback loops (hours, not months)
- Higher capital velocity
- Alignment with institutional microstructure research (CKS et al.)
- Match with existing Midas infrastructure

Hold-to-resolution remains a parallel track for niche strategies (resolution criteria edges, cross-platform arb) but is not the current priority.

### The Forcing Question

For any candidate strategy, the discriminating question is: **"If the market price didn't move between entry and resolution, would I still make money?"**

- Yes → hold-to-resolution. Edge is in terminal probability estimation.
- No → trade-the-price. Edge is in information arrival speed.

### Maker vs Taker — Structurally Asymmetric

Polymarket's fee structure (March 2026) creates strong asymmetry:

- **Taker fees**: 0% (geopolitics) to 1.8% (crypto), peak at 50¢ price
- **Maker fees**: 0%, with 20-25% rebate on counterparty's taker fee (50% in finance category)

This means:

- Taker strategies are cost-constrained; need strong directional edge
- Maker strategies have positive baseline economics before any directional edge (you earn rebates just for providing liquidity)
- Maker thesis aligns with existing Midas architecture
- Maker strategies open more markets (don't need fee-free geopolitics only)

The **Avellaneda-Stoikov framework** is the canonical reference for maker strategy design. This has not yet been implemented for Dali.

### Signal Hierarchy

From the academic literature (Cont-Kukanov-Stoikov 2014, Kolm-Turiel-Westray 2023, Sirignano-Cont 2019):

- **OFI (Order Flow Imbalance)** is the canonical signal primitive
- **TFI (Trade Flow Imbalance)** is a degraded subset, computable from fills
- OFI achieves R² ~65% on equity data; TFI ~32%
- OFI captures information from placements and cancellations that fills miss

For Dali specifically:

- Historical TFI analysis is complete (Block B), result was ambiguous-mixed
- Live OFI capture starting Block A0 (2026-05-28)
- The OFI hypothesis has not yet been tested on Polymarket data

### The Briola Caveat (Critical)

Briola, Bartolucci & Aste (2024) showed that even state-of-the-art LOB models achieving 70%+ directional accuracy produce **zero or negative net returns** after realistic transaction costs and execution modeling.

**Forecasting accuracy ≠ trading profit.** This caveat must be enforced on every result that looks promising. Block A1 analysis should include cost-adjusted edge computation, not just statistical predictive power.

## Current Status by Block

### Complete

- **Phase 1 infrastructure**: TFI baseline, live capture, replay parser, maintained book state, OFI computation (with unit tests), executable-price backtest engine
- **Historical sign convention audit**: `historical_to_aggressor()` confirmed correct (Goldsky-derived field semantics verified empirically across 4 families)
- **Block B (TFI deep-dive)**: OUTCOME 3 — mixed results requiring live validation. Strongest finding: operator-removal effect (equity_index 47% → 58% hit rate, crypto 34% → 52%)
- **Block E Lite (operator attribution)**: The Block B operator effect is driven almost entirely by 2 relayer addresses, not MM bots or HFT. Major simplification of the operator-filtering picture.

### In Progress / Imminent

- **Block A0 (24h dual-thesis capture)**: VPS setup required. Runbook available at `notes/dali/block_a0_runbook.md`. Recommended region: AWS us-east-1 or DigitalOcean NYC (low latency to Polymarket infrastructure)
- **Live sign convention audit**: Will auto-resolve once Block A0 produces 50+ classifiable trades

### Outstanding (Not Yet Drafted)

- **Block A1 (analysis prompt)**: To be drafted after Block A0 returns data. Must include:
  - Taker thesis analysis (CKS-style R² and hit rate)
  - Maker thesis analysis (counterfactual fill quality, adverse selection)
  - Operator filtering as a comparison dimension (but expect modest effects given Block A target markets are DISTRIBUTED with low relayer share)
  - Cost-adjusted edge per market per category
  - Walk-forward validation on extended data window

### Deferred (Trigger-Conditional)

- **Block D (backtest engine extensions)**: When Block A1 shows tradeable signal in at least one strategy class
- **Block F (Optuna parameter search)**: Task 5 triggers — 3+ families × 24h × 200+ trades × resolved sign convention
- **Block E proper**: Largely subsumed by existing wallet/copy-trading infrastructure per audit
- **Block G — Live deployment infrastructure**: When at least one strategy is validated
- **Block H — ML models (LightGBM, NN)**: After rule-based baseline shows clear edge. Not before. The Briola caveat applies.
- **Block I — Cross-platform arbitrage**: Future independent track
- **Block J — Resolution criteria scanning**: Async background, low priority

## Key Findings From This Phase

### Block B (TFI Deep-Dive) Headlines

1. Raw TFI is not tradeable in any family/horizon/condition combination
2. Operator-removal effect is strong — but Block E Lite shows it's really driven by just 2 relayer addresses, not the broader 12-address denylist
3. AI/product walk-forward: test set hit rate (54.78%) exceeded train (48.68%) — unusual and slightly bullish if it replicates
4. Crypto walk-forward: severe degradation from train (36%) to test (25%) — strategy likely dead in current regime
5. Sports: no signal in any horizon, league, or condition tested

### Block E Lite Headlines

1. **Two relayer addresses drive the operator effect.** Removing MM bots or HFT wallets has near-zero impact. Filtering the 2 relayers gets ~95% of the lift.
2. **Block A target markets are all DISTRIBUTED.** Known operator share ranges 0.0%-5.9%. The operator-filtering trick may not materially replicate on these specific markets.
3. **Live operator filtering is constrained.** Relayer addresses don't appear in RTDS proxyWallet events. Filtering requires post-hoc join to raw fills, not real-time.
4. **The relayers are infrastructure, not traders.** Most likely Polymarket UI router or major wallet/aggregator routes. Their effect on TFI signal is probably "drowning out signal with uninformed volume," not "smart money trading against TFI."

## Open Strategic Questions

These deserve investigation when time/data allows:

1. **Who are the two relayers?** Block-explorer investigation. If they're the Polymarket UI router, "filter relayers" essentially means "exclude retail UI flow." Different strategic implication than "exclude smart money."
2. **Does the AI/product walk-forward test result replicate?** With fresh data (when ingestion catches up), extend the test window and see if the unusual pattern holds.
3. **Can the maker thesis pencil out on geopolitics markets?** Zero fees + 0% relayer share + retail-distributed flow + reasonable liquidity. This is theoretically the cleanest environment for maker strategy testing.
4. **Is there genuine OFI predictive power on Polymarket?** The decisive test, Block A1.
5. **How does Polymarket's bounded [0,1] price space affect Avellaneda-Stoikov style market making?** The paper assumes unbounded Gaussian dynamics. Adaptation needed if pursuing maker strategy seriously.

## Tools and Methodology Notes

### Codex vs Cowork Roles

Going forward:

- **Codex**: implementation, analysis execution, structured outputs. Writes scripts, runs computations, produces CSVs and findings docs.
- **Cowork**: interpretation, planning, prompt drafting, strategic discussion. Reads Codex outputs, suggests next steps, updates living docs.

### Anti-Patterns to Avoid

- **Building infrastructure before validating signal.** The roadmap was reframed from "validate everything first" to "fastest path to first dollar" earlier in the project. Maintain that discipline.
- **Optimizing on insufficient data.** Task 5 triggers exist for a reason. Don't run parameter searches before they're met.
- **Adding ML when rule-based hasn't shown edge.** The Briola caveat is the operative warning.
- **Trusting positive results without confidence intervals.** Block B showed several promising-looking buckets that didn't survive statistical scrutiny.

### Data Discipline

- Always report actual data window, not assumed wall-clock window
- Always note when analyses use stale cached data vs fresh
- Sign convention: historical = `historical_to_aggressor()` confirmed correct; live = still UNKNOWN until 50+ classifiable trades in Block A0
- Operator filtering: use `OPERATOR_ADDRESSES` (12 hardcoded) as baseline; consider relayer-only subset given Block E Lite finding

## Immediate Next Decisions

In rough priority order:

1. **Set up VPS for Block A0 capture** (US-East region, ~$10/month)
2. **Run Block A0 capture** starting 2026-05-28 morning
3. **Draft Block A1 analysis prompt** while capture runs
4. **Investigate the two relayer addresses** when time permits (1-hour task)
5. **Refresh cached family fills** to enable conditional follow-ons that were skipped in Block E Lite

## Notes on the Cowork Transition

This document was created at the point of transitioning primary AI tooling from web-chat to Cowork. The expectation is that Cowork's direct repo access will improve workflow significantly for this phase of work. The other-chat ideation continues to be useful for brainstorming and discussions that shouldn't be in repo context.

For Cowork orientation, point it at the notes listed at the top of this document. The **block status** and **immediate next decisions** sections of this doc are the highest-leverage briefing material.
