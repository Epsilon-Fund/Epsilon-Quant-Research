---
title: "Observatory v3 — multi-source news (RSS + newsletters), Scheme-A bias weighting, 24-market universe, gauge/grid charts (attended build)"
created: 2026-07-05
status: shipped — attended/manual loop over 24 markets; Scheme-A live and the uncovered-source proposal APPROVED + ACTIVATED 2026-08-24 (α refit 2.7 → 2.85); website parked
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
  - fair-value
---
# Observatory v3 — broader sources, bias-weighted evidence, a 24-market universe, and overview charts

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]] · Table terms: [[polymarket_table_dictionary]]
> Builds ADDITIVELY on [[newsagent_observatory_v2_findings]] (hybrid Stage-A/Stage-B model, GDELT layer). The closed "our % beats the mid" gates ([[newsagent_v0_gate_findings]]) stay closed and displayed; the thesis is unchanged — our independent FV, judged against resolved outcomes; the mid is discovery + context.

## Plain-English Summary

- **What this note is:** the build record of Observatory v3 (2026-07-05, late evening): the news layer went multi-source (free RSS set + a read-only newsletter-inbox module), article influence is now scaled by **Scheme-A source-reliability weights** (Wikipedia perennial-sources tiers + Iffy Index blocklist, both licence-clean), the live universe expanded **5 → 24 markets**, and the dashboard gained a per-market **gauge** plus an **overview grid with sparkline trends** (gap-ranked, flags first).
- **Everything ran attended** — no cron, no Anthropic API key; Stage-A extraction (899 articles, one wave) and the 19 new onboarding priors were produced in-session by batched cheap-model subagents through the documented out-of-band files. The paid-key + n8n web-launch automation is scaffolded as documentation only (`newsagent/AUTOMATION.md`) and deliberately NOT wired.
- **One-line status:** 24 markets scored and ledgered append-only (sf-2026-001…024); no divergence flag fired on day one — the two large disagreements (Becerra −77.9pp, United Russia +33.5pp) carry wide day-one bands, so the confidence leg withheld them, exactly as designed; v2 behavior is preserved wherever v3 sources/weights are absent.

## Sources (v3)

| Source | Role | Display rule | Status |
|---|---|---|---|
| Guardian Open Platform | full-text packets (trail + lede + last para) | headline + link only | unchanged (demo key) |
| **RSS set: BBC, Sky, Politico, The Hill** (7 feeds) | keyword-filtered headlines + descriptions | headline + link only | **live — 177 items on first fetch** |
| Wikipedia Current Events | curated daily bullets | text shown (CC BY-SA attribution) | unchanged |
| **Newsletters (ING daily + weekly macro)** | analysis-grade full text for Stage A | **NEVER displayed** (private/licensed; generic label only) | module live, **awaiting Justin's read-only credential** |
| GDELT GKG via BigQuery | attention burst + tone (Stage-B) | analytical chip | unchanged (v2.1) |

Mechanics: all RSS feeds are fetched ONCE per run (day-cached) and filtered per market by time window + keywords; packets dedupe cross-source by **normalized-title hash**; packet slots cap at newsletters ≤2, Guardian ≤6, RSS ≤4, Wikipedia fills the rest (12 total). Reuters/AP have no public RSS and Google News RSS is personal-use-only — both skipped (radar). Politico's `politics-news.xml` validated but was empty at probe time; its `congress.xml` carries 30 items — both wired.

**Newsletter ingestion** (`newsagent/email_ingest.py`): read-only IMAP (`BODY.PEEK`, mailbox opened readonly — no message is ever marked/moved), sender-filtered to the ING list, one article-shaped item per email with `display=False` enforced end-to-end (the test suite asserts no `newsletter:` string reaches the public HTML). Gmail-API OAuth is scaffolded behind the same interface. **Needs Justin:** an app-password credential at `secrets/email_imap.json` (`{"host": "imap.gmail.com", "user": ..., "app_password": ...}`) — gitignored like the GDELT key; the pipeline runs without it (newsletters simply absent).

## Scheme A — source-reliability weighting (live; uncovered-source weights PROPOSED)

**Mechanism** (`newsagent/sourceweights.py`): the Wikipedia perennial-sources table encodes each outlet's community-consensus status as a row class (`s-gr`/`s-nc`/`s-gu`/`s-d`/`s-b`); we parse it via the MediaWiki API (439 entries), join the **Iffy Index** blocklist (2,040 domains, CC BY), and map status → weight with a DECLARED table: generally-reliable 1.0, no-consensus 0.7, generally-unreliable 0.3, deprecated/blacklisted/Iffy-listed 0.0. The weight scales an article's influence in **Stage-B aggregation only** (contribution and band dispersion — a blocklisted source can neither be the decisive signal nor move the band); Stage-A extraction is unweighted. Cache + attended weekly refresh (`sourceweights.refresh(force=True)`); a stale or failed refresh serves the previous cache and warns — the daily loop can't be broken by a weights outage. Attribution for both datasets is in the page footer.

**Verified live:** all five wired outlets resolve to generally-reliable (weight 1.0) from real RSP rows — including the Sky News UK vs Sky News Australia split — and the parser demonstrably differentiates (Fox News → s-nc/0.7, Daily Mail → s-d/0.0, Iffy-listed test domain → 0.0). So today Scheme A changes no number; it becomes load-bearing exactly when sources broaden — which is the point of wiring it before they do.

**Proposal for sign-off (uncovered sources — NOT hard-adopted; all run at the neutral 1.0 = previous flat Scheme B until Justin approves):**

> **APPROVED AS PROPOSED — Justin, 2026-08-24. LIVE from that date.** The three rows below are now the declared weights in `sourceweights.py` (`UNCOVERED_SOURCE_W` / `WP_CE_W` / `UNKNOWN_W`), joined by the v3.1 additions (Bloomberg 0.9, bank research desks 0.9). Activating them changed real Stage-B inputs, so α was refit through `scripts/newsagent_hist_backfill.py --fit` on the same 355-pair / 49-market resolved sample: **α 2.7 → 2.85**, pooled Brier 0.2214 → 0.2215 (i.e. the reweighting bought principle, not fit), band_mult unchanged at 0.5. The neutral-1.0 baseline was re-run first and reproduced α = 2.7 exactly, which is what makes the +0.15 attributable to this decision and nothing else. See [[newsagent_observatory_v33_findings]].

| Source | Proposed weight | Rationale |
|---|---|---|
| ING research newsletters | 0.9 | institutional research desk; analysis-grade but a single house view — a notch below the reliable-wire tier |
| Wikipedia Current Events bullets | 0.8 | curated tertiary digest of agency reporting; not a primary outlet, but editorially filtered |
| Unknown/uncovered news domains | 0.5 | conservative default once Scheme A is signed off (today: 1.0 pre-sign-off neutral so nothing silently changes) |

## Stage-A v3 (clarity) + band rigor

- **Extraction prompt v3** adds a per-article **clarity** field (how unambiguous the item's signal is on the question — independent of direction and strength). Versioned cache key → full re-extraction: **899 unique (market, article) records** in one 8-batch attended wave (archive 364 + legacy live/backfill re-extraction + the 24-market fetch).
- **Band v3** (magnitudes DECLARED, structure per the build notes): `half = floor + band_mult·(18·dispersion + 8·(1−clarity) − 1.5·min(n_rel,6))`, capped 35pp — where **dispersion is now the reliability-WEIGHTED cross-source disagreement** (Scheme-A weights: reliable sources disagreeing widens, concurring narrows) and low weighted clarity widens. `band_mult = 1.0` is the single coverage-calibration knob.
- **Band-coverage calibration** (`fvmodel.band_coverage()`): read-only over the sf ledger; for binary outcomes the checkable statement is bucketed — realized YES-frequency vs the average band per FV bucket. Currently `collecting — 0 settled / 24 active`; the pre-registered procedure is: at ≥20 settled forecasts, rescale `band_mult` once so bucket coverage ≈ nominal, documented in this note's successor. **Deliberately not built into the `calibrate` engine**: that engine is now the public `lemma-calibrate` package (two project shims); upstreaming interval coverage there is flagged as a skills-lifecycle follow-up, not smuggled into a research commit.
- **α refit on the v3 features** (same v0-archive outcomes, 53 pairs / 9 markets): **α = 1.8 at the declared γ = 1.0**, pooled Brier **0.4267** — better than the v2-feature fit (0.4518) at every γ, and the γ-sensitivity curve is now nearly flat (0.428 → 0.423 across γ∈[0,2]), i.e. the v2.1 monotone-in-γ artifact was partly extraction noise. γ stays declared; the fit still reads as a starting point, refit on every settlement.

## Universe — 24 markets (discovery + curation)

Discovery: `scripts/newsagent_v3_universe.py` (Gamma, server-side `liquidity_num_min` + politics tag + paginated sweep; binary, informative mid 5–95c, ends ≤2027, sports/joke markets screened). 45 candidates → **24 curated** (5 v2 + 19 new), ≤2 per event family, retrieval keys hand-tuned per market. Mix: 9 slow/structural (Fed ×2, House, Senate, California governor + wealth-tax, Lula, Bardella, United Russia, Nobel) and 15 shock (Iran cluster ×4, Hormuz ×2, Ukraine/Russia ×3, Putin, Trump-out, Taiwan, Netanyahu, Cuba, US-Iran meeting). **Still no informative UK binary live** (the Starmer family resolved; nothing above the liquidity floor) — noted in config; revisit each slate refresh. New markets onboard **today-only** (five-perspective prior from today's packet; no 14-day reconstruction — their series build forward, shown as "new" in the grid).

**Day-one readout highlights** (full table in `showcase.json`; ledger sf-2026-001…024): the two biggest disagreements are **Becerra −77.9pp** (our prior 14% vs mid 91.9% — the market clearly carries California-race information our packet didn't; band [1, 35] → flag correctly withheld) and **United Russia +33.5pp** (our 90% vs mid 56.5% — arguably the mirror case where OUR structural prior is the informed one; also unflagged on day-one band width). **No divergence flag fired** — day-one bands are wide by construction. These two rows are exactly what the Observatory exists to score: if the market is right, our Brier pays for it in public; the interesting question is symmetric.

**Practical example (how a v3 row got its number):** United Russia's five-perspective prior came out 90% (dominant-party structural base rate). Its packet had 12 items (Guardian + RSS + WP, deduped); Stage A found no item asserting a qualifying development against ER dominance (relevance high, phases none/speculative), so S_t ≈ 0, the evidence state stayed ~0, and FV = prior = 90% with band [69, 99] (slow floor + dispersion default at low n_rel). The card's breakdown table shows precisely this: prior 90, carry 0, no article steps.

## Charts (no redundancy, site design language)

- **Overview grid** (new, top of page): all 24 markets — FV / mid / gap / band / compact **sparkline** of the FV path — ordered flags-first then |gap| (the divergence layer's ranking). The sparkline is a preview; the full FV+band-vs-mid time series stays in the market card (overview → detail, not duplication). Day-one markets show "new".
- **Gauge per market** (new, in each card): 0–100 arc with the band segment, FV needle (accent), mid tick (grey) — the % as a visual next to the numerals.
- Unchanged detail views: full time series, FV-construction breakdown, GDELT attention chip, divergence bar layer, reliability panel, honest gate scoreboard.
- Verified in-browser: 24 cards/gauges/rows, grid scrolls in its own container, no horizontal page scroll on mobile, IP-scrub assertions green (no local paths, no scripts, no newsletter content).

## Attended run — what actually executed (2026-07-05 late)

1. `--stage fetch`: 24 markets × (Guardian + RSS + WP) packets; GDELT refreshed to 24 name-sets (33 total cached series); newsletters gracefully skipped (no credential — exact instruction printed).
2. Extraction: 899 pending → 8 Haiku-subagent batches → **899/899 ingested** (per-entry JSON repair needed on 0 batches this round).
3. Onboarding: 19 new five-perspective priors (2 Sonnet subagents, isolation rules; the 5 legacy priors explicitly protected from overwrite).
4. `--fit`: α refit on v3 features (above); `backfill-compute` re-evolved the 5 legacy series under v3 features/weights; `--stage publish`: 24 markets → ledger (append-only; 19 new sf entries) → dashboard.
5. Note on re-extraction effects: v3 semantics shifted some legacy numbers (e.g. Fed-July FV 83.9 → 53.7 as re-judged evidence fell under the slow-market drip threshold) — visible in the ledger history by design.

Cost: ~899 cheap-model extractions + 19 mid-model priors, all in-session; steady-state daily cost is unchanged (only new articles extract; ~50–150/day across 24 markets).

## Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions:** RSP status ≈ source reliability (community consensus as proxy; declared weight map); newsletter and WP weights pending sign-off run at 1.0 (pre-sign-off neutral — nothing changed silently); auto-drafted-then-hand-tuned retrieval keys ≈ adequate market coverage (thin packets on Nobel/Lula day one say coverage varies); day-one priors for 19 markets are single-shot five-perspective ensembles (their quality is exactly what the forward ledger measures); α/γ/band knobs as declared in v2/v2.1 notes.

**Live-only unknowns:** forward calibration over the 24-market book (first resolutions: US–Iran meeting 07-17, Fed July 07-29, Hormuz 07-31, MOU 07-31); band coverage (procedure pre-registered above); divergence-flag fire rate at 24 markets; whether the Becerra/United-Russia style disagreements resolve for the market or for us — scored either way; Q-DIV-EDGE (unchanged, NOT run).

**Power honesty:** the α refit is still 53 pairs / 9 markets / one month; the 24-market forward book is the real sample, and it needs weeks-to-months of resolutions before any claim beyond "we publish and score."

## What Justin needs to do

1. **Email credential** for newsletter ingestion: read-only app password → `secrets/email_imap.json` (module docstring has the exact shape). Until then the loop runs without newsletters.
2. ~~**Scheme-A sign-off**, now including the uncovered-source proposal table above (ING 0.9 / WP-CE 0.8 / unknown 0.5).~~ **DONE 2026-08-24 — APPROVED and live** (α refit 2.7 → 2.85).
3. Unchanged: `ANTHROPIC_API_KEY` + `GUARDIAN_API_KEY` + cron-env `GOOGLE_APPLICATION_CREDENTIALS` for the future unattended switch (see `newsagent/AUTOMATION.md` — scaffold only); website handoff stays parked per this round's mandate.
4. ~~**Settlements ahead:** four markets resolve in July — run `sf settle` + slate refresh + `--fit` per the runbook when they do.~~ **DONE 2026-08-24** (overdue by 3-5 weeks): all four settled and scored, slate refreshed, α refit. First public Brier **0.0794** on n=4. See [[newsagent_observatory_v33_findings]].

## STRETCH / BACKLOG (marked; nothing blocks the shipped v3 slice)

- **Tone → direction calibration** once resolved outcomes accrue (GDELT tone + Stage-A tone are stored and displayed, unwired).
- **Event-driven cadence** — the GDELT burst signal + Stage-A cache make intraday re-scoring nearly free.
- **Web-launch automation** — the documented paid-key + n8n switch (`newsagent/AUTOMATION.md`); flip only with Justin's keys and sign-off.
- **Interval coverage upstreamed into `lemma-calibrate`** — skills-lifecycle candidate (RC backlog), not a research-commit smuggle.
- **Gmail-API OAuth path** for the inbox (IMAP app-password is the v3 path).
- **UK re-entry** — first informative UK binary above the liquidity floor joins the slate.

## Outputs

- Package additions: `newsagent/email_ingest.py`, `newsagent/sourceweights.py`, `newsagent/AUTOMATION.md`; upgraded `feeds.py` (RSS set, title-hash dedupe, packet slots), `features.py` (PROMPT_VERSION 3, clarity), `fvmodel.py` (weighted band v3, `band_coverage`), `dashboard.py` (gauge, overview grid, sparklines), `config.py` (24-market slate), `run_daily.py` (source wiring).
- Discovery: `scripts/newsagent_v3_universe.py` (repeatable slate refresh).
- Params: `fv_params.json` — α 1.8 (fitted, v3 features), γ 1.0 + band_mult 1.0 + λ/floors/caps (declared).
- Tests: `tests/test_newsagent_fv.py` — **42 green** (v3 adds: RSS dedupe/window filter, newsletter parse + privacy + graceful-absence, Scheme-A weight mapping, weighted-dispersion/clarity/band_mult behavior, coverage-collecting, gauge/sparkline/grid render + ordering).
- Live artifacts (git-ignored): `data/newsagent/live/` (rss_cache, sourceweights.json, gdelt_daily.json, 24-market day dirs, priors, fv_series), showcase JSON+HTML.
- Ledger: sf-2026-001…024 (`SF_BOOK=polymarket`, append-only).
