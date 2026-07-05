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

## Current state (2026-07-05)

**Both pre-registered gates failed; the honest showcase shipped anyway — with the failure as its content.** v0 (single-sample forecaster) and v0b (5-perspective ensemble, Amendment 2) both fired the same early-stop on the falsifier trio (Brier diff +0.19 / +0.17 vs the mid, tracking 40%): headline-packet LLM forecasting recovers materially less information than a liquid politics mid on shock transitions. **The fair-value framing is therefore CLOSED** (no v0c). What shipped instead is the **Epsilon Calibration Observatory** — a public measurement loop where the news-agent's % + band is displayed next to the PM mid *as a scored experiment*: daily snapshots to the append-only superforecasting ledger (`SF_BOOK=polymarket`, entries sf-2026-001…005 live since 2026-07-05), Brier-scored by `calibrate` on settlement, retrospective gate scoreboard shown prominently (market currently winning — that is the content, not a bug). Package: `polymarket/research/newsagent/` (`run_daily.py` stages: fetch → prompts → forecast → publish); dashboard artifacts regenerate to `data/newsagent/showcase/` (self-contained HTML + JSON, numbers-only default + analytical toggle, IP-scrubbed, Guardian/Wikipedia attribution).

**Open items for Justin:** (1) source-weighting Scheme A sign-off (Wikipedia RSP tiers + Iffy blocklist — radar note); (2) a registered Guardian dev key (`GUARDIAN_API_KEY`; demo key in use) and `ANTHROPIC_API_KEY` for the cron-run forecast stage (today's forecasts were produced out-of-band via the documented `--forecasts-file` path); (3) hand the showcase JSON/HTML to the website colleague when ready.

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
