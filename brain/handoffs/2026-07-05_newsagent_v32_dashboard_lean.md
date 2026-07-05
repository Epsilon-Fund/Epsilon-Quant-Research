---
title: "Handoff — Observatory v3.2: two-pane dashboard (feed + surfaced donuts), AllSides-seeded lean axis (Ground News ToS-blocked), movers + evidence badges, deployed-site design"
created: 2026-07-05
status: complete
owner: justin
project: polymarket
para: project
hubs:
  - COWORK
  - strat_news_agent_showcase
tags:
  - handoff
  - news-agent
---
# Handoff 2026-07-05 — Observatory v3.2 (third attended session today)

> Hub: [[COWORK]] · [[strat_news_agent_showcase]] · Build record: [[newsagent_observatory_v32_findings]] · Base: v3.1 commit `fb63de8`.

## What ran

- **Dashboard rebuilt at the right level** (the v3.1 "2-col" had landed inside cards): page = sticky left news/evidence feed pane (aggregated, deduped, lean-chipped, own scroll) + right markets pane; **all 24 donut gauges surfaced** in a full-width overview grid (zero clicks — the "donuts buried in dropdowns" complaint is gone); detail cards keep the collapse/picker machinery; restyled to the **deployed** epsilon site (charcoal/cream/terracotta, 01. sections, pill buttons; inspected epsilon-site-changes.vercel.app live and reproduced with local fonts; old black+lime removed test-enforced).
- **Lean axis:** Ground News **not viable** — no API, and its ToS prohibits automated *and* manual monitoring/copying (radar delta updated with the Skip verdict). Shipped the plan's named fallback: curated **AllSides-seeded** table (`newsagent/sourcelean.py`; Guardian −2, BBC 0, Politico −1, The Hill 0, Bloomberg −1, Fox +2, Daily Mail +1, Sky honestly unrated), DECLARED extremity multipliers 1.0/0.9/0.75 composed with Scheme-A reliability in `sourceweights.annotate()` (reliability ≠ lean; blocklist precedence test-pinned). Explainer + feed show the lean mix and our-own-packet L/C/R coverage.
- **Refit under composed weights** (existing `--fit` path, 355 pairs / 49 markets): **α 2.1 → 2.7** (mechanical ~1/0.75 Guardian compensation; pooled Brier 0.2214, unchanged-to-better). **band_mult stays 0.5** — the coverage selector was tie-breaking into 0.25/undercoverage on 3-bucket granularity; implemented the v3.1 note's "narrowest knob AT nominal" reading (6-line change, flagged for Justin's veto/bless).
- **Extensions:** big-movers strip (published-snapshots-only ΔFV; correctly empty until a second published day) + evidence-quality badge (thin/moderate/strong; "strong" ≡ the divergence flag's confidence leg).
- **Re-publish:** backfill-compute re-evolved the 5 legacy series; 19 day-one states reset (backup `fv_state.pre-v32.json`) and recomputed; 24 markets re-ledgered (sf-2026-001…024, same-day pre-settlement). Headline rows stable (Becerra −77.9pp, United Russia +33.5pp, Fed −35.8pp); still no flag.

## Verification

76 tests green (11 new). IP-scrub clean (the only "cdn"/"secret" hits are an outbound JPM-PDF link and a BBC headline). Inline JS `node --check` pass. Headless-Chrome renders verified top-to-bottom at 1500px; the apparent 420px overflow was diagnosed as **Chrome's ~500px minimum-window crop artifact**, not a page bug (scrollWidth probe = innerWidth at 500px); CSS hardened defensively; a real sub-500px device check is the one residual.

## Needs Justin (delta over v3.1's list)

1. **Lean sign-off** — table + multipliers (running live, refit-covered; veto = one-line revert + refit).
2. **band_mult selector call** — narrowest-≥-nominal (keeps 0.5) vs closest-to-nominal (0.25, undercovers).
3. **AllSides written-NC-OK** email (non-blocking; non-monetised + attributed meanwhile).
4. Unchanged: Scheme-A uncovered weights, `GUARDIAN_API_KEY`, `GEMINI_API_KEY`, automation keys, July settlements (first: US–Iran 07-17).

## Where things landed

Code `polymarket/research/newsagent/` (+ `sourcelean.py`); fit script selector fix in `scripts/newsagent_hist_backfill.py`; params `fv_params.json` (α 2.7); tests 76 green; page + JSON regenerated under `data/newsagent/showcase/` (git-ignored); notes: [[newsagent_observatory_v32_findings]] + radar delta + hub § Current state + [[TODO]] § News-Agent updated. STRETCH/BACKLOG (news-driven-only universe, rates-via-options track, lean-diversity band term, cross-spectrum chip, movers alerting, web-launch automation) marked in the findings note — none started.
