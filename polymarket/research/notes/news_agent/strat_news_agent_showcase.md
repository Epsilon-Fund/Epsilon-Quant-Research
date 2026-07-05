---
title: "News-Agent Fair-Value + Calibration Showcase — thread hub"
created: 2026-07-04
status: shipped as Calibration Observatory — fair-value framing closed by v0+v0b; live measurement loop running since 2026-07-05
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

## Current state (2026-07-05, evening — Observatory v2)

**Rebuilt as Epsilon's INDEPENDENT fair value, judged against resolved outcomes; the mid is discovery + context only.** The v0/v0b closures stand untouched (the "our % beats the mid" claim stays CLOSED and displayed). The daily number now comes from a **hybrid two-stage model**: Stage A — a cheap LLM extracts structured features per article (cached per (market, article), prompt-versioned); Stage B — a transparent decayed log-odds model (onboarding prior from the five-perspective ensemble + decisive-signal evidence weighting) whose single evidence weight is **fitted on resolved outcomes** (v0 archive: α=1.4, 53 pairs / 9 markets — a starting point, refit on every settlement). Divergence flag pre-registered: |FV−mid| ≥ 15pp AND band half ≤ 12pp AND ≥ 5 relevant articles/72h — "where our model most disagrees", never an edge claim; the internal edge question (Q-DIV-EDGE) is pre-registered and NOT run. Dashboard restyled to the epsilon-webs1te design language (read-only borrow) with FV+band-vs-mid time series (14-day labeled reconstruction), FV-construction breakdown, divergence layer, reliability panel, interim scoring (ForecastBench convention). Full build record: [[newsagent_observatory_v2_findings]]. Ledger sf-2026-001…005 updated append-only; `calibrate` wired (0 settled yet; first resolution 2026-07-17).

**Open items for Justin:** (1) source-weighting Scheme A sign-off (unchanged); (2) `GUARDIAN_API_KEY` (registered) + `ANTHROPIC_API_KEY` (daily cron; out-of-band paths verified meanwhile); (3) GDELT-via-BigQuery credential (`GOOGLE_APPLICATION_CREDENTIALS` + `uv add google-cloud-bigquery`) for the stretch historical calibration — scaffold ships in `newsagent/gdelt_bq.py`; (4) hand `data/newsagent/showcase/` to the website colleague.

## Notes in this cluster

- [[newsagent_v0_gate_preregistration]] — locked metric + cheapest falsifier for the v0 gate (incl. Amendment 1: packet source).
- [[newsagent_v0_gate_findings]] — v0 results: **STOP verdict**, mechanism diagnosis, v0b redesign proposal, STRETCH/BACKLOG.
- [[newsagent_repo_data_radar_findings]] — external radar (repos, news APIs, bias datasets) + source-weighting proposal for sign-off.
- [[newsagent_observatory_v2_findings]] — the v2 rebuild: hybrid Stage-A/Stage-B FV model, outcome calibration (α fit), divergence flag rule + pre-registered Q-DIV-EDGE, redesigned dashboard, cost ledger, Justin items.

## Where things live

- Code: `polymarket/research/scripts/newsagent_*.py` (v0) and `polymarket/research/newsagent/` (v1 package, if gated in).
- Raw pulls (append-only): `polymarket/research/data/newsagent/`.
- Result CSVs: `polymarket/research/data/analysis/csv_outputs/news_agent/`.
- Plots: `polymarket/research/data/analysis/plots/news_agent/`.
- Forecast ledger (git-ignored runtime, append-only): `polymarket/research/data/superforecast/`.
