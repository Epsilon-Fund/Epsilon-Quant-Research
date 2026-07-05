---
title: "Observatory v2 — Epsilon's independent news-driven fair value, judged against outcomes (hybrid Stage-A/Stage-B rebuild)"
created: 2026-07-05
status: shipped — live daily loop rebuilt on the hybrid FV model; forward ledger running; internal edge question pre-registered, NOT run
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
# Observatory v2 — an independent news-driven fair value on politics/macro questions, judged against actual outcomes

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]] · Table terms: [[polymarket_table_dictionary]]
> Prior gates (CLOSED, unchanged): [[newsagent_v0_gate_findings]] · pre-registration: [[newsagent_v0_gate_preregistration]] · Radar: [[newsagent_repo_data_radar_findings]]

## Plain-English Summary

- **What this note is:** the build record of the Observatory rebuild (2026-07-05). The public page now presents **Epsilon's own fair-value probability** per liquid politics/macro question, produced by a two-stage hybrid model, and judged over time against **actual resolved outcomes**. Polymarket is only the discovery layer (which liquid questions matter) and display context — **never the benchmark**. This does NOT reopen the closed "our % beats the mid" claim; both v0/v0b gate failures stay displayed on the page.
- **The mechanism:** Stage A — a cheap LLM (Haiku-class) extracts structured features per news article (relevance, stance, event phase, strength, tone, novelty), one extraction per (market, article), cached forever. Stage B — a transparent log-odds model turns features into the fair value: a one-time onboarding prior (five-perspective ensemble, never shown the mid) plus a decayed evidence state, with the **single evidence weight α fitted on resolved outcomes** from the v0 gate archive (9 resolved markets, 53 lookahead-free pairs).
- **What shipped today:** the full daily loop ran end-to-end — packets (Guardian full-text + Wikipedia Current Events), out-of-band Haiku extraction (374 articles), Stage-B FV + bands + divergence flags for the 5-market live slate, append-only ledger snapshots (sf-2026-001…005 updated), a 14-day labeled reconstructed FV time-series, and the redesigned site-styled dashboard (charts, FV-construction breakdown, divergence layer, reliability panel).
- **One-line status:** live and honest — no divergence flag fired today (the one large FV-vs-mid gap, Fed −35.8pp, was correctly withheld by the confidence leg); the internal "is there edge in flagged divergences?" question is **pre-registered below and not run**.

## What changed vs the shipped v1 Observatory (design)

The v1 Observatory (2026-07-05 morning) published the five-perspective LLM ensemble's number directly, daily. v2 splits language from calibration, exactly per the reframed PRD ([[COWORK]]-side plan doc § Project 2 — build-note add-ons):

| Piece | v1 (this morning) | v2 (this note) |
|---|---|---|
| Daily number | 5-perspective LLM forecast per market per day | **Stage-B model**: prior + decayed evidence update (no daily LLM forecast) |
| LLM's role | forecasting | **extraction only** (Stage A) + one-time onboarding prior |
| News input | titles only | **body text**: trailText + lede + last paragraph of relevance-ranked top-k (internal use only; the page shows headline + link) |
| Calibration | none (methodological only) | **α fitted on resolved outcomes** (v0 archive); refit as the forward ledger settles |
| Daily LLM cost | 5 forecasts/day (Sonnet-class) | ~0 forecasts; only *new* articles get one cheap extraction each (cached) |
| Divergence | gap shown | **flag rule** |FV−mid| ≥ 15pp AND band half ≤ 12pp AND ≥ 5 relevant articles/72h |
| Time series | none | FV + band vs mid per market, 14-day labeled reconstruction so it is populated day one |

**Practical example (how one article moves the number).** On 2026-06-20 the Starmer-out market's packet contained *"Former Greater Manchester mayor Andy Burnham is elected to the House of Commons"*. Stage A emits `{relevance 0.9, stance toward_yes, event_phase in_progress, strength 0.8, novelty ~1}` → contribution c = 0.9·0.8·0.8·1.0·(0.5+0.5·1.0) ≈ +0.65. With decisive-signal weighting it counts fully; Stage B adds α·c ≈ +0.9 logit-units to the evidence state, moving an 8% FV to ~17% — and the dashboard's FV-construction table shows that article with its exact +pp effect. No LLM was asked "what's the probability"; the LLM only read the article.

## Stage-B calibration on resolved outcomes (the honest numbers)

**Calibration set:** the v0 gate archive — 56 archived lookahead-free packets across **9 resolved June-2026 politics markets** (10 selected minus one with no archived packets), 5 YES / 4 NO, ~6 event families, giving **53 (market, snapshot) pairs** while markets were open. Onboarding priors for the archive markets were produced now by the same five-perspective protocol from each market's *first* snapshot packet only (isolation rules as in the gates; a fresh **empty-packet canary on the Iran-deal question answered 3.7%** — base rate, not the true YES outcome → no hindsight contamination in the prior worker).

**Fit:** grid search on the single α minimizing pooled Brier against outcomes.

| quantity | value | read |
|---|---|---|
| α (evidence weight) | **1.4** | one confirmed decisive article (c ≈ 0.7) moves ~1 logit-unit |
| pooled Brier at α* | **0.4587** | vs **0.4628** prior-only — evidence helps, modestly |
| pooled Brier of the mid (context) | 0.2777 | context only; never a fitting target or gate |
| n | 53 pairs / 9 markets / ~6 families | one geopolitically extreme month |

![Stage-B alpha fit](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/news_agent/newsagent_stageb_alpha_fit.png)

Caption: pooled Brier (y) versus the evidence weight α (x) on the 53 archive pairs; dashed green = fitted α, dotted grey = prior-only baseline. The curve is shallow: evidence improves on the prior but cannot rescue the irreducible pairs (below).

**Read — why the pooled Brier is high and why that is fine.** Roughly half the pairs are *pre-shock days on markets that later resolved YES* (e.g. the Iran deal at snapshots 06-08/06-11, when the market itself sat at ~5c). A news-only model **cannot** know the future there — no evidence existed yet; those pairs are irreducible without inside information, which is precisely the Observatory's stated stance ("shock misses are expected and scored anyway"). Per-family: the model improves on the prior for Colombia (0.35→0.32), Starmer (0.93→0.87), Vučić (0.92→0.90); it is roughly flat on the Iran cluster. The fit is a **starting point refit on every settlement**, not a validated result — n is small, one macro regime, and the same month contains the two shock transitions that killed the v0/v0b display claim.

**Column glossary for the output CSVs** (`newsagent_stageb_pairs.csv` / `_family.csv` / `_fit.csv` under `data/analysis/csv_outputs/news_agent/`): `p0_pct` = onboarding prior (five-perspective ensemble, %); `A` = cumulative decayed evidence state at the snapshot; `fv` = Stage-B fair value (fraction); `y` = resolved outcome; `brier_fv`/`brier_prior`/`brier_mid_context` = per-pair squared errors of the model / the prior alone / the Polymarket mid (context); `mtype` = market type (slow/shock); `n_missing_feats` = packet articles without cached features (0 in the final run).

## Model form — every knob declared, one knob fitted

FV_t = sigmoid( logit(p0) + α·A_t ), with A_t = λ^Δdays·A_{t−1} + S_t.

- **Fitted on outcomes (1 parameter):** α = 1.4.
- **Declared (not fit — n far too small):** per-type decay λ (slow 0.9/day, shock 0.6/day); band floors (slow ±8pp, shock ±12pp, cap ±35pp = the v0 gate's sanity bar); decisive-signal weighting (strongest new signal each way counts fully, everything else corroborates at 0.25 — chosen after plain sums let a pile of weak conflict headlines drown the one decisive "deal could be signed by Sunday" item in the archive's Iran packets); slow-market guards (daily evidence below |S|=0.5 is drip-filtered; accumulated evidence capped at 1.5 logit-units/α). The guards exist because without them a steady drip of mildly-positive coverage compounded the Dems-House FV to an absurd 98.9% during the build — the worked example is recorded in `fvmodel.py`.
- **Extraction semantics versioned:** the Stage-A prompt is v2; v1 judged stance as "does the subject welcome this?" and inverted e.g. *challenger rises, leader vows to stay* on leader-exit questions (the market repriced 19→54c on the Burnham news that v1 had marked toward_no). The cache key includes the prompt version, so semantics changes can never silently mix. This was a definitional bug fixed on inspection, not outcome-tuning; both extraction rounds are preserved in the day logs.
- **In-sample honesty:** the model *form* (weighting scheme, guards) was selected while looking at the calibration set (its failures, not its outcomes-vs-mid score). There is no pre-registered claim here to protect — the forward ledger is the test, and the page says so.

## Today's live run (2026-07-05, all five markets, end-to-end)

| market | type | prior p0 | FV [band] | mid | gap | flag |
|---|---|---|---|---|---|---|
| Putin out before 2027 | shock | 3.0% | **3.0%** [1, 24] | 9.5% | −6.5pp | no |
| Hormuz traffic normal by Jul 31 | shock | 8.0% | **8.2%** [1, 21.9] | 15.5% | −7.3pp | no |
| Fed no-change (July) | slow | 53.7% | **53.7%** [45.7, 61.7] | 89.5% | −35.8pp | **no** (evidence-thin) |
| US–Iran meeting by Jul 17 | shock | 27.3% | **27.5%** [8, 47] | 38.0% | −10.5pp | no |
| Dems control House 2026 | slow | 54.3% | **74.6%** [66.6, 82.6] | 83.5% | −8.9pp | no |

Read: the two slow markets behave as designed — Dems-House moved off its prior only on decisive days (capped), and the Fed's big −35.8pp disagreement did **not** flag because the packet had <5 relevant articles (the confidence leg withholding exactly the kind of thin-evidence disagreement we should not amplify). The gap itself is still displayed — it is our honest independent view; the market prices Fed information our news feed does not carry. Ledger: sf-2026-001…005 updated append-only; the pre-correction 98.9% Dems snapshot from the first publish attempt remains in the ledger history (append-only means the correction is visible, which is the point).

**Interim scoring (populated day one):** 66 forecast-days across the 14-day reconstruction + today, mean interim Brier **0.0445** (each day's FV scored against the *next day's* mid — the ForecastBench convention, labeled interim, replaced by outcome scoring at settlement). The reconstructed segment is drawn dashed, labeled, and **never enters the ledger**; its onboarding priors were produced against the earliest reconstructed packet (asof 2026-06-21) under the same isolation rules.

## Divergence flag — the pre-registered rule (X, Y stated)

**Public flag fires only when |FV − mid| ≥ 15pp AND band half-width ≤ 12pp AND ≥ 5 relevant articles in the last 72h.** Rationale: 15pp ≈ 2× the v0 median |gap| (7.5pp) so only tail disagreements flag; the confidence leg (narrow band + evidence floor) means a flag can never fire on thin or internally-contradictory evidence; shock-type markets (floor 12pp) flag only at maximum confidence by construction. The flag is presented as "**where our model most disagrees**" — informational, never an edge claim.

### Pre-registered internal research question (NOT run — registered before any flag has fired)

> **Q-DIV-EDGE:** Over the forward ledger, among snapshots where the public divergence flag fired under the exact shipped rule above, is there positive expected value in taking the FV side at the mid? **Metric:** mean per-contract PnL versus the mid at flag time, family-clustered bootstrap CI. **Sample floor:** ≥ 30 resolved flagged snapshots across ≥ 10 distinct markets. **Pass bar:** CI lower bound > 0 net of a taker-spread haircut. **Discipline:** until that gate is run and passes, no edge is asserted anywhere, public or internal; this registration exists to prevent post-hoc cherry-picking of flags.

## Cost ledger (the cheap path held)

- Stage A: **738 unique (market, article) extractions** this build (364 archive + 374 live/backfill), batched ~60–130 per cheap-model call; daily steady-state is only *new* articles (~10–40/day across 5 markets) ≈ **pennies/day** at Haiku API prices.
- LLM forecasts: **16 one-time** five-perspective calls (9 archive priors + 5 live priors + 2 canaries) — priors are set once per market, not daily.
- Stage B and the dashboard are LLM-free (deterministic Python).
- This build ran the LLM work **out-of-band via Claude Code subagents** (documented `--features-file`/`--priors-file` paths) because no `ANTHROPIC_API_KEY` is set; the API path is implemented and unit-shaped but **unverified against the live API** — flagged below.

## Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions:** Guardian+WP packets ≈ the live news view (single-outlet slant acknowledged; Scheme B flat source weights pending Scheme A sign-off); Haiku-class extraction ≈ shipped extractor; the v0 archive month ≈ a usable calibration regime (it is shock-heavy — α may be biased up); declared λ/floors/guards are judgment values; the 14-day reconstruction uses priors produced today against old packets (isolation-instructed, canary-checked, but the workers are post-hoc by construction — hence display-only, never scored).

**Live-only unknowns (what the forward ledger resolves):** forward calibration of FV and bands (Brier/reliability via `calibrate` on settlement); whether the divergence flag fires at a sane rate; α stability across news regimes; Q-DIV-EDGE.

**Power honesty:** 53 pairs / 9 markets / 1 month is a fit, not a validation. The dashboard's reliability panel says "in-sample calibration set" explicitly until forward resolutions accumulate.

## What Justin needs to do (unchanged items consolidated)

1. **`ANTHROPIC_API_KEY`** for unattended daily runs (extraction + onboarding of new markets). Until then: out-of-band flow per `run_daily.py` docstring.
2. **`GUARDIAN_API_KEY`** registered dev key (demo key in use; 500 calls/day is enough but unregistered).
3. **GDELT/BigQuery (stretch):** GCP project → BigQuery API → service account (BigQuery User + Job User) → JSON key → `GOOGLE_APPLICATION_CREDENTIALS`; then `uv add google-cloud-bigquery`. Module `newsagent/gdelt_bq.py` is scaffolded, fails with exact instructions, and includes the residential-IP DOC-API fallback with the mandatory client-side `seendate` filter.
4. **Scheme A source-weighting sign-off** (unchanged; table in [[newsagent_repo_data_radar_findings]]; flat Scheme B running).
5. **Website handoff:** `data/newsagent/showcase/` (showcase.json + index.html) to the site colleague; the page is style-matched to epsilon-webs1te (read-only borrow — nothing shipped into his repo).

## Daily runbook

```bash
cd polymarket/research
PYTHONPATH=. uv run python -m newsagent.run_daily --stage all      # with API keys
# out-of-band (no keys): --stage fetch → --stage extract (writes pending) →
#   agents produce features → --stage extract --features-file done.json → --stage publish
```

Settlement: `sf settle` per resolved market (SF_BOOK=polymarket), refresh the slate in `config.py`, re-run `scripts/newsagent_stageb_calibration.py --fit` including the new resolved pairs, then `calibrate` renders the public track record.

## STRETCH / BACKLOG (marked; nothing here blocks the shipped loop)

- **Historical GKG calibration at scale** (BigQuery V2Tone per query/day as a Stage-B feature; the client scaffold ships in `gdelt_bq.py`) — needs the GCP credential.
- **Event-driven cadence** — news-burst triggers re-running Stage B intraday (Stage A cache makes this nearly free).
- **Local extraction model** — zero out API cost for Stage A once volumes justify it.
- **Q-DIV-EDGE gate** — as pre-registered above, only after the sample floor.
- **Scheme A weights** in `SOURCE_W_DEFAULT`'s place once signed off; per-source reliability tiers plug into `contribution()`.
- **Slate refresh discipline** — UK market re-entry when an informative UK binary lists (slate rule already documents it).

## Outputs

- Package (rebuilt): `polymarket/research/newsagent/` — `features.py` (Stage A, versioned cache), `fvmodel.py` (Stage B + fit), `feeds.py` (full-text + relevance rank + mid history), `gdelt_bq.py` (scaffold), `run_daily.py` (stages incl. backfill), `dashboard.py` (site-styled page), `fv_params.json` (α=1.4, declared knobs, committed).
- Calibration: `scripts/newsagent_stageb_calibration.py`; CSVs `newsagent_stageb_{pairs,family,fit}.csv`; plot `newsagent_stageb_alpha_fit.png`.
- Tests: `tests/test_newsagent_fv.py` — **22 green** (cache/versioning, weighting, state math incl. same-day idempotency, band monotonicity, flag rule, breakdown-sums-to-FV, dashboard render + IP-scrub assertions).
- Live artifacts (git-ignored, regenerable): `data/newsagent/live/` (day dirs, feature cache, priors, fv_state, fv_series, backfill), `data/newsagent/showcase/{showcase.json,index.html}`.
- Ledger: `SF_BOOK=polymarket` sf-2026-001…005 (append-only; today's correction visible by design).
