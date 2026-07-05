---
title: "News-Agent Fair-Value Showcase — v0 Gate Pre-Registration"
created: 2026-07-04
status: pre-registered — locked before any gate metric was computed
owner: justin
project: polymarket
para: project
hubs:
  - strat_news_agent_showcase
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - research
  - news-agent
  - showcase
  - pre-registration
---
# News-Agent Fair-Value Showcase — v0 Gate Pre-Registration

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

- **What this is:** the pre-registered gate design for the "Epsilon News-Agent Fair-Value + Calibration Showcase" — a public view showing Epsilon's own news-informed fair-value probability next to the Polymarket mid on high-liquidity politics markets, with a Brier/calibration track record as the hook. It is a **showcase, not a trading strategy**.
- **Why it exists:** per the no-infra-before-signal law, the v0 gate decides whether the news-informed % is worth displaying at all *before* any dashboard is built. This note locks the metric, sample, and cheapest falsifier **before** any gate number is computed.
- **What it covers:** a bounded recent window (2026-06-08 → 2026-07-02) of resolved high-liquidity politics/geopolitics markets, forecast retrospectively but lookahead-free (news ≤ t only, all events post-LLM-training-cutoff), scored against the PM mid.
- **One-line status:** pre-registration only — no results in this note; results land in [[newsagent_v0_gate_findings]].

## What this is NOT (scope fence)

- **Not a reopen of closed pricing.** The "PM pricing is efficient" closures cover **financial binaries vs their liquid underlying** — same-day crypto touch/terminal ladders ([[od_same_day_crypto_pricing_gate_findings]]) and SPX daily up/down ([[od_equities_index_pricing_scope_findings]]). Neither covers news-driven politics fair value from a news-agent; that is the never-run Block J / LLM-forecaster frontier ([[TODO]] § future blocks).
- **Not "beating the mid."** Success is a credible, well-calibrated, legible public view. The gate below explicitly does *not* require our Brier to be lower than the market's.
- **Not monetized, and IP-scrubbed before anything public.** No proprietary thresholds, wallet data, or alpha surfaces appear in the showcase.

## The v0 question

On a bounded recent window, does a news-informed fair-value % per politics market **(a) stay credible** (not embarrassingly miscalibrated vs the mid), **(b) diverge** from the mid enough to be a view rather than a mirror, and **(c) track** the same reality the mid reacts to, legibly? If not, stop before the dashboard.

## Sample (locked)

**Window:** snapshot dates 2026-06-08 → 2026-06-29, every 3 days: {06-08, 06-11, 06-14, 06-17, 06-20, 06-23, 06-26, 06-29}, snapshot time 12:00 UTC. All events are after the forecaster LLM's training cutoff (Jan 2026), so outcomes cannot be known from model weights — only from the supplied news.

**Resolved-market universe rule:** Gamma politics-tagged (`tag_id=2`) markets with `closed=true`, `endDate` ∈ [2026-06-05, 2026-07-03], ranked by `volumeNum`, then filtered:

1. Binary market with unambiguous resolution text.
2. `volumeNum` ≥ $3M (high-liquidity bar).
3. Information content: CLOB mid ∈ [5c, 95c] on ≥ 3 snapshot dates while open (drops dead-at-1c markets like `will-china-invade-taiwan-by-june-30-2026`).
4. Event-family dedupe: at most 2 markets per family (e.g. the US–Iran June cluster is one macro event); target ≥ 5 distinct families; include ≥ 1 UK-family market if any passes filters 1–3 (candidate: the `starmer-out` family).
5. Target n = 10 resolved markets (fewer if filters bind; report the actual universe table in the findings note).

**Live display-read markets (unscored):** up to 4 current high-liquidity politics markets (e.g. July Fed decision, Putin-out-before-2027, Brazil presidential election) get forecasts on 2 recent dates purely to preview the gap display; they carry no gate weight.

**Practical example.** `us-x-iran-permanent-peace-deal-by-june-15-2026` ($177M volume) is one row-family. At snapshot 2026-06-11 12:00 UTC the pipeline hands the forecaster the market question + resolution criteria + up to 12 GDELT headlines from 06-08→06-11, and gets back `p = 62%, band [48, 74]`. The PM mid at the same timestamp (say 71c) is recorded separately. That (market, date) pair contributes one Brier observation for us and one for the mid after resolution (YES → our Brier contribution (0.62−1)² = 0.144, mid's (0.71−1)² = 0.084), one |gap| observation (9pp), and one tracking observation vs the 06-14 snapshot.

## Forecast protocol (locked)

- **Inputs per (market, snapshot):** question, Gamma resolution description, "today is `<snapshot date>`", and a news packet: GDELT DOC 2.0 `artlist`, market-specific query terms (recorded verbatim in the outputs CSV), window [t−72h, t] (widened to 7d if empty), English, `sort=hybridrel`, deduped, max 12 items of (title, seendate, domain). Throttled ≥ 6s/request, cached append-only under `data/newsagent/v0/`.
- **The forecaster never sees:** the PM mid/price history (independence of the view), the outcome, anything after t, or any web/tool access (no-tools instruction; packets are the only evidence).
- **Output contract:** JSON `{p_pct, band_lo, band_hi, drivers[]}` where band is an 80% credible interval on fair p and drivers cite packet items.
- **Model:** one Claude Sonnet call per (market, snapshot), superforecasting-style instructions (outside view/base rate first, then news adjustment). Model class is an assumption logged below; the live v1 ledger uses the strongest available model.
- **Contamination controls:** post-cutoff events only; no tools; 3 empty-packet canary calls (expect wide-band base-rate output, not the true outcome); the orchestrating agent (which has seen resolutions while building the universe) writes prompts mechanically from templates and never injects its own probability.

## Pre-registered metrics and gate

All computed on identical (market, snapshot) pairs; family-clustered bootstrap (1,000 draws, resample event families) for the CI on the Brier difference.

| Metric | Definition | Bar |
|---|---|---|
| **M1 credibility** | pooled Brier(ours) − Brier(mid), same pairs | ≤ +0.05 |
| **M2a divergence** | median \|p_ours − mid\| | ≥ 2pp |
| **M2b sanity** | p90 \|p_ours − mid\| ≤ 35pp AND pooled Brier(ours) | ≤ 0.25 |
| **M2c tracking** | among consecutive-snapshot pairs where the mid moved ≥ 8pp, share where our % moved the same direction | ≥ 65% |
| **M3 band (descriptive, non-gating)** | band width distribution; last-snapshot band vs realized outcome direction | report only |

**PASS = M1 ∧ M2a ∧ M2b ∧ M2c** → build v1 (dashboard + live ledger). **Any other combination = STOP**: write the findings note saying the view is not worth displaying, and do not build the dashboard.

**Cheapest falsifier (early stop):** run the 3 highest-volume distinct-family markets first (planned: the June-15 US–Iran deal, the June Fed no-change, `starmer-out-by-june-30`). If on those three Brier(ours) > Brier(mid) + 0.15 **and** M2c < 50%, abort the remaining calls and declare STOP — wrong *and* illegible is unrecoverable by adding sample.

**Why these bars.** M1 at +0.05: the mid of a $10M+ market is a formidable aggregator; a public calibration hook survives being slightly worse than the mid, not materially worse. M2a at 2pp: below that the "view" is a re-render of the mid and there is nothing to show. M2b: a view that strays >35pp at p90 or is worse than a coin is noise, not analysis. M2c at 65%: when the market visibly repriced on news, a news-agent that usually moved the other way cannot be publicly narrated.

## Assumption ledger (realism calibration, [[CODEX]] § Realism calibration)

**Modeled assumptions:** GDELT headline packets ≈ the news a live agent would have seen at t (title-level only in v0; v1 adds RSS/fuller text); Sonnet-class forecaster ≈ lower bound on the shipped model; 12:00 UTC mid from CLOB `/prices-history` (fidelity=60, chunked ≤ 10d per the known ~15d span cap) as "the market view"; every-3-days cadence ≈ daily cadence viability.

**Live-only unknowns (what only the running ledger resolves):** calibration on a *forward* sample scored by `calibrate` on the append-only superforecasting ledger; band honesty over time; source-weighting effects (v0 is unweighted); persistence across news regimes beyond one geopolitically-correlated month.

**Power honesty (Rule 1):** ~10 markets across ~5–6 families in one 3.5-week window is a **viability screen, not a validated result**. The window is dominated by one macro cluster (US–Iran); family clustering is mandatory in the CI and the per-family table must be shown. A PASS here reads "MERITS-SHOWCASE-BUILD with a live measurement ledger," never "edge." The durable public claim is earned by the live ledger only.

## Amendment 1 — packet source (2026-07-05, BEFORE any scored forecast was produced)

**What changed:** the news-packet source moves from GDELT DOC 2.0 to the union of (a) **Guardian Open Platform** date-bounded search (`from-date`/`to-date`, headline+timestamp+section only) and (b) **Wikipedia Current Events portal daily pages** (one page per calendar day, keyword-filtered bullets). Packet size, window ([t−72h, t], widen to 7d if <3 items), dedupe, and the max-12 rule are unchanged. GDELT remains the intended v1 primary once reachable.

**Why:** GDELT entered a persistent aggressive-throttle state (HTTP 429 on single spaced requests for >1.5h, reproduced from two unrelated IPs — this machine and the Hetzner VPS). No scored forecast existed at amendment time (only the 3 empty-packet canaries, which contain no news), so this is a data-source substitution before data collection, not a post-hoc change.

**Lookahead note:** both replacement sources are time-stamped by construction (Guardian `webPublicationDate`; Wikipedia page-per-day), which is *stricter* than GDELT's sloppy end-bound (~24h leak, per the radar's live probes). Coverage caveat logged in the assumption ledger: Guardian is a single outlet (UK-slanted) and the WP portal is curated/coarse; v0 packets are therefore thinner than the intended v1 multi-source stream — a conservative bias for the gate (a pass on thin packets understates the shipped pipeline).

## Decision rule restated

- PASS → v1: news-agent pipeline + append-only ledger (SF_BOOK=polymarket) + `calibrate` scoring + site-ready local dashboard.
- STOP → findings note documents the failure legibly; no dashboard; propose (at most) a redesigned gate for sign-off.

Results: [[newsagent_v0_gate_findings]] (written after this note was locked).
