---
title: "News-Agent Fair-Value + Calibration Showcase — thread hub"
created: 2026-07-04
status: gated — v0 STOP (early-stop fired 2026-07-05); v0b redesign proposed, awaiting sign-off
owner: justin
project: polymarket
para: project
hubs:
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - news-agent
  - showcase
  - hub
---
# News-Agent Fair-Value + Calibration Showcase (hub)

> Hub backlinks: [[POLYMARKET_BRAIN]] · [[COWORK]] · [[TODO]]

## Plain-English Summary

- **The idea:** a public, non-monetized showcase of Epsilon's own news-informed fair-value probability on high-liquidity politics prediction markets, shown next to the Polymarket mid with a confidence band — with the **calibration/Brier track record over time as the hook**. A showcase of research craft, not a trading strategy.
- **Lineage:** this is the never-run Block J / LLM-forecaster frontier ([[TODO]] § future blocks). It does **not** reopen the closed financial-binary pricing results ([[od_same_day_crypto_pricing_gate_findings]], [[od_equities_index_pricing_scope_findings]]) — those cover crypto/equities binaries vs liquid underlyings, not news-driven politics fair value.
- **Discipline:** no infra before signal — the v0 gate ([[newsagent_v0_gate_preregistration]]) decides whether the view is worth displaying before any dashboard exists. Forecast ledger is append-only and anti-post-hoc (vendored `superforecasting` skill, `SF_BOOK=polymarket`), scored read-only by the first-party `calibrate` skill.

## Current state (2026-07-05)

**v0 gate: STOP — the pre-registered early-stop fired.** On the falsifier trio the news-agent's Brier was 0.547 vs the mid's 0.354 (diff +0.194 > +0.15 bar) with 40% news-tracking (< 50% bar): the status-quo-anchored single-sample forecaster under-reacts to probability-shifting news even when it is in-packet. **No dashboard was built** (no-infra-before-signal held). The pipeline, isolation protocol, and canaries all worked; a redesigned v0b gate (ensemble + evidence-weighting prompt + multi-source packets) is proposed in the findings note and **awaits Justin's sign-off** — it was deliberately not run to avoid gate-shopping.

## Notes in this cluster

- [[newsagent_v0_gate_preregistration]] — locked metric + cheapest falsifier for the v0 gate (incl. Amendment 1: packet source).
- [[newsagent_v0_gate_findings]] — v0 results: **STOP verdict**, mechanism diagnosis, v0b redesign proposal, STRETCH/BACKLOG.
- [[newsagent_repo_data_radar_findings]] — external radar (repos, news APIs, bias datasets) + source-weighting proposal for sign-off.

## Where things live

- Code: `polymarket/research/scripts/newsagent_*.py` (v0) and `polymarket/research/newsagent/` (v1 package, if gated in).
- Raw pulls (append-only): `polymarket/research/data/newsagent/`.
- Result CSVs: `polymarket/research/data/analysis/csv_outputs/news_agent/`.
- Plots: `polymarket/research/data/analysis/plots/news_agent/`.
- Forecast ledger (git-ignored runtime, append-only): `polymarket/research/data/superforecast/`.
