---
title: "News-Agent Showcase — Repo/Data Radar Findings (repos, news access, bias datasets)"
created: 2026-07-04
status: assessment complete — adoption gated on v0/v1; source-weighting scheme awaits owner sign-off
owner: justin
project: polymarket
para: project
hubs:
  - strat_news_agent_showcase
  - POLYMARKET_BRAIN
tags:
  - external-research
  - repo-audit
  - news-agent
---
# News-Agent Showcase — Repo/Data Radar Findings

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]]
> Criterion + verdict scale per [[2026-06-28_external_repo_audit]]: what is liftable NOW for this thread, licence + integration cost stated, verdict capped at **Adopt / Borrow-pattern / Reference / Skip**; AGPL (or unlicensed) → reimplement, never vendor.

## Plain-English Summary

- **What this is:** the external radar for the news-agent fair-value showcase — LLM-forecaster repos, FinceptTerminal, news-access APIs (GDELT/NewsAPI/RSS/Guardian), and public source-bias datasets — run by three parallel research subagents on 2026-07-04.
- **One-line takeaway:** the pipeline design comes from the Halawi et al. 2024 paper (repo unlicensed → reimplement); the only Adopt-grade code dependency is **Metaculus/forecasting-tools (MIT)**; news access is **GDELT (keyless, 2017+ history) + RSS + Guardian Open Platform**; the licence-clean source-weighting proposal is **Wikipedia perennial-sources tiers (CC BY-SA) + Iffy Index blocklist (CC BY)** — AllSides is NC-grey, Ad Fontes/MBFC data are paid/contract-gated, and unofficial MBFC scrapes are ruled out.
- **Two design-critical empirical facts:** (a) GDELT's `enddatetime` bound silently leaks ~24h of future articles — client-side `seendate` filtering is mandatory for lookahead-free reconstruction (now enforced in our v0 scripts); (b) ForecastBench finds un-anchored LLMs score somewhat worse than market crowds (~0.113 vs 0.093 superforecaster Brier), and market-anchored ("freeze value") LLMs are better but **circular for a fair-value-vs-mid showcase** — our forecaster therefore never sees the mid, and expectations for the public Brier hook are set accordingly.

## Priority stack

| Rank | Item | Licence | Verdict | What we take |
|---|---|---|---|---|
| 1 | **Metaculus/forecasting-tools** | MIT | **Adopt** (v1/production path) | research→N-samples→aggregate→extract→clamp harness + spend caps; the only dependency worth taking |
| 2 | **Halawi et al. 2024** (`dannyallover/llm_forecasting`) | repo UNLICENSED; paper open | **Reference — reimplement** | the canonical pipeline: retrieval → relevance-rate → summarize → scratchpad → trimmed-mean ensemble |
| 3 | **GDELT DOC 2.0 (+GKG/BigQuery for v2 tone)** | open data, cite+link | **Adopt** | keyless retrieval, 2017+ headline history, V2Tone via BigQuery free tier for v2 calibration |
| 4 | **RSS set (BBC/Sky/Guardian/Politico/The Hill) + Guardian Open Platform** | public feeds; Guardian dev key free non-commercial | **Adopt** | live probed 200s; Guardian = UK anchor w/ full text for internal scoring only (display headline+link only) |
| 5 | **ForecastBench** | MIT code / CC BY-SA data | **Borrow-pattern** | interim scoring of unresolved markets vs prior-day price; superforecaster prompts; log-odds aggregation |
| 6 | **metac-bot-template** | no licence file | **Borrow-pattern** | prompt scaffold (status-quo weighting, scenarios, `Probability: ZZ%`, clamp [0.01,0.99]) |
| 7 | **Wikipedia RSP tiers + Iffy Index** | CC BY-SA 4.0 / CC BY 4.0 | **Adopt (proposed, needs sign-off)** | reliability tiers → weights + zero-weight blocklist; MediaWiki API parse, weekly refresh |
| 8 | **FinceptTerminal** | AGPL-3.0 + commercial dual | **Borrow-pattern (design only, never read code)** | "Fincept-style" news panel = RSS config → local store w/ FTS → ticker/feed/detail panes |
| 9 | **NewsData.io** | free tier permits commercial display, 12h delay | **Reference (fallback feed)** | only NewsAPI-family member licence-viable for a public page |
| 10 | AllSides | CC BY-NC 4.0 (primary page unverified) | Reference | bias axis only via hand-curated table + written NC OK — Scheme C option |
| 11 | Schoenegger & Park 2024 | no code; OSF data unconfirmed | Reference | multi-model ensemble justification |
| 12 | MediaCloud / CC-NEWS | free acad. quota / open WARCs | Reference | v2 coverage-volume cross-check / full-text-at-scale only if needed |
| 13 | NewsAPI.org, GNews, Mediastack | free tiers prohibit production/public use | **Skip** | — |
| 14 | Ad Fontes data, MBFC official API | paid/contract-gated | **Skip** (Borrow the two-axis weighting *pattern*) | — |
| 15 | Unofficial MBFC scrapes/mirrors, Baly corpus, NewsGuard, Polymarket/agents (archived), AutoCast, adj.news | unlicensed / restricted / dead | **Skip — do not use** | — |

## Prompting/aggregation patterns adopted into the engine design

1. **Scratchpad decomposition** with explicit base-rate step (Halawi; reused by ForecastBench).
2. **Status-quo weighting** + scenario-each-way + rigid output format + clamp [1%, 99%] (metac-bot-template).
3. **N-sample trimmed mean**; the sample dispersion is the honest **confidence band** — no audited repo ships a band; it must be derived from ensemble spread (Halawi; forecasting-tools N=5).
4. **Relevance-rate-then-summarize** on title + first 250 words — the summaries double as the dashboard's evidence panel (Halawi).
5. **No market anchoring** in the headline number (circularity; ForecastBench freeze-value warning). If ever shown, an anchored variant is a labeled diagnostic.
6. **Interim scoring vs prior-day market price** for unresolved markets so the track record has content from day one (ForecastBench).
7. **Don't bolt on recalibration prematurely** — Halawi found the ensemble naturally calibrated; verify with our `calibrate` skill first.

## Source-weighting proposal (for Justin's sign-off — not hard-adopted)

| # | Scheme | Licence risk | Read |
|---|---|---|---|
| **A (recommended)** | Wikipedia perennial-sources reliability tiers → weights, + Iffy Index zero-weight blocklist, + first-party tone | **Low** (CC BY-SA + CC BY, attribution footer) | cleanest legally; US+UK first-class; maintained; publicly defensible |
| B | Flat curated whitelist (wires + broadsheets + public broadcasters), no external dataset | None | zero dependency; our judgment is the product; used as the **v0/v1 default until sign-off** |
| C | AllSides bias axis (hand-curated top-50, CC BY-NC + attribution + written OK) × RSP reliability tiers | Medium→Low with written OK | closest open approximation of the Ad Fontes two-axis design |

**v0/v1 default pending sign-off: Scheme B** (neutral curated list), because the gate and the first ledger entries must not depend on an unapproved external dataset.

## Operational gotchas (live-probed)

- GDELT: 1 req/5s hard throttle with sticky 429s; ~24h end-bound slop (client-side `seendate` filter mandatory); title text is tokenised (display-ugly); wire-syndication duplicates need title-hash dedupe; DOC API history starts 2017-01-01.
- Reuters/AP have no public RSS (2026); Google News RSS is personal-use-only — not for the public page.
- Guardian dev key: 500 calls/day, 1/s, non-commercial, full text for internal scoring only.

## Decision

Adopt the GDELT+RSS+Guardian retrieval stack and the Halawi-pattern engine (reimplemented first-party for v0/v1; forecasting-tools as the production adoption path); default to weighting Scheme B pending sign-off on Scheme A; never vendor FinceptTerminal/unlicensed/AGPL code — pattern-borrow only. Full subagent reports summarized here were produced 2026-07-04; licences marked UNCONFIRMED above should be re-verified before any public launch.

## Delta 2026-07-05 — Observatory v2 rebuild additions

Verdicts per the same [[2026-06-28_external_repo_audit]] lens (liftable-now, licence + integration cost, Adopt/Borrow-pattern/Reference/Skip):

| Item | Licence | Verdict | What we take / cost |
|---|---|---|---|
| **GDELT via Google BigQuery** (`gdelt-bq.gdeltv2.gkg`) | open data (cite+link); client lib `google-cloud-bigquery` Apache-2.0 | **Adopt (scaffolded)** | bypasses the DOC-API datacenter-IP 429 block entirely; free tier 1 TB/mo covers daily-tone queries easily. Cost: one GCP service account (Justin) + one dependency. Scaffold shipped in `newsagent/gdelt_bq.py` with graceful missing-credential messaging + the residential-IP DOC fallback (client-side `seendate` filter enforced). |
| **epsilon-webs1te local clone** | internal (colleague-owned) | **Style reference only (read-only)** | design tokens extracted (dark `#0a0a0a`, `#F0EFE9` text, single `#C8FF00` accent, Instrument Serif / IBM Plex Sans / Geist Mono, 24px-radius panels, uppercase micro-labels) and reproduced with local font stacks in the self-contained dashboard. Nothing shipped into or restructured in that repo. |
| **Calibration libs** (scikit-learn calibration, `calibration` pkgs) | BSD/MIT | **Skip (first-party wins)** | the vendored `calibrate` skill already provides Brier + Murphy decomposition, reliability diagrams, ECE/MCE on our own ledger format; an external lib would add a dependency without capability. Isotonic/Platt recalibration stays deferred per the "don't bolt on recalibration prematurely" pattern (§ above). |
| **forecasting-tools (MIT)** | MIT | **Adopt-later (unchanged)** | still the production path if the onboarding-prior stage ever needs true N-sample ensembles; v2's daily loop no longer makes daily LLM forecasts, so the urgency dropped. |
| **FinceptTerminal** | AGPL-3.0 | **Borrow-pattern (design only, unchanged)** | the v2 dashboard's evidence feed (domain-tagged headline list, ticker-ish density) is the borrowed *pattern*; no AGPL code read or vendored. |
