---
title: "Observatory v3.1 — Gmail newsletters live, macro-research PDFs, tractability tags, source-bias explainer, dashboard UX, and a 355-pair historical refit (attended build)"
created: 2026-07-05
status: shipped — attended/manual; Gmail read-only token minted; alpha/band_mult refit on 49 resolved markets; website parked
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
  - calibration
---
# Observatory v3.1 — richer sources, honest tractability labels, and a much larger calibration sample

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]] · Table terms: [[polymarket_table_dictionary]]
> Builds ADDITIVELY on [[newsagent_observatory_v3_findings]] (commit `282b5a2`). The closed "our % beats the mid" gates ([[newsagent_v0_gate_findings]]) stay closed and displayed; the thesis is unchanged — our independent fair value judged against resolved outcomes; the Polymarket mid is discovery + display context only.

## Plain-English Summary

- **What this note is:** the build record of Observatory v3.1 (2026-07-05, one attended session): the newsletter inbox went live through **Gmail-API read-only OAuth** (consent minted; 10 ING + 6 Bloomberg items reachable), five **public macro-research PDFs** (JPM ×3, GS, BofA) joined the sources, Stage-A extraction gained a **Gemini 2.5 Flash provider flag**, every market is tagged **news-driven vs data-driven** (with an honest "our news-FV is structurally blind here" note on the data-driven ones), each card explains **how its number formed by source lean × reliability tier**, and the dashboard was rebuilt around a default-visible subset with 2-column cards.
- **The headline number:** the Stage-B calibration sample grew from **53 pairs / 9 markets (one extreme month)** to **355 pairs / 49 markets (Feb–Jul 2026, 19 event families)** via a lookahead-free historical backfill of resolved Polymarket politics/macro markets. Refit: **α = 2.1** (evidence beats prior-only, pooled Brier 0.2216 vs 0.2317) and the pre-registered band-coverage rescale set **band_mult = 0.5** (bands were roughly twice as wide as outcomes warranted).
- **A real privacy bug was found and fixed before it fired:** v3's public evidence feed had no `display=False` filter — the moment the Gmail token armed newsletters, private newsletter titles would have leaked onto the public page. Now filtered end-to-end (evidence, drivers, FV-breakdown), with regression tests.
- **One-line status:** 24 markets re-published under the refit params (ledger sf-2026-001…024 updated pre-settlement, append-only); still no divergence flag on day one; 65 tests green; everything ran attended; website stays parked.

## What shipped (v3.1 slice)

| Item | Status | Where |
|---|---|---|
| Gmail-API OAuth newsletter path (read-only) + Bloomberg sender | **LIVE** — token minted 2026-07-05 (attended browser consent) | `newsagent/email_ingest.py` |
| Public macro-research PDFs (JPM Recap/Guide/Brief, GS Monitor, BofA CMO) | **LIVE** — 5/5 fetched on first probe | `newsagent/pdf_ingest.py` |
| Gemini 2.5 Flash extraction behind a provider flag + spot-check tool | built; **needs `GEMINI_API_KEY`** | `newsagent/features.py`, `scripts/newsagent_provider_spotcheck.py` |
| Market tractability tags (news vs data-driven) | LIVE — 4 markets tagged data-driven | `newsagent/config.py` |
| Ratings explainer (source lean × reliability) | LIVE on every card | `newsagent/fvmodel.py` `source_bias_breakdown` |
| Dashboard UX rework (2-col cards, subset, picker, collapse, flags-first) | LIVE | `newsagent/dashboard.py` |
| Historical backfill + α/band_mult refit | **DONE** — 355 pairs / 49 markets | `scripts/newsagent_hist_backfill.py` |
| Newsletter privacy fix (`display=False` enforced at the page boundary) | FIXED + regression-tested | `dashboard.py`, `run_daily.py` |

## Sources delta (over the v3 table)

| Source | Role | Display rule | Status |
|---|---|---|---|
| **Gmail inbox via OAuth** (`gmail.readonly`) | ING THINK + ING research + **Bloomberg** newsletters → analysis-grade Stage-A text | **NEVER displayed** (generic label only; count shown as "+N private analysis items") | live — token at `secrets/gmail_token.json`, auto-refresh; IMAP path kept as fallback |
| **JPM Weekly Market Recap / Guide to the Markets (UK) / Weekly Brief** | weekly macro research PDFs (static URLs) | headline + source + link only; body internal | live |
| **GS Weekly Market Monitor** | date-templated (most recent Friday, `MMDDYY`) | headline + source + link only | live (one-week-back retry, then silent degrade) |
| **BofA Capital Markets Outlook** | date-templated (most recent Monday, `MM-DD-YYYY`) | headline + source + link only | live (same degrade) |

Mechanics: PDFs get a **disclaimer-strip** (declared line-pattern list — boilerplate/legal lines never reach Stage A), are macro-tagged, carry an internal-only `scan_text` so a Fed mention on page 3 still keyword-matches a rates market, and cap at ≤2 packet slots (newsletters ≤2, PDFs ≤2, Guardian ≤6, RSS ≤4, WP fills; 12 total). Like RSS/newsletters they are live-only sources and are excluded from historical reconstruction by construction. A sender-filter bug was also fixed in passing: the v3 `ing.com` substring never matched ING THINK's real sending domain `economics.ingthink.com` — the explicit filter now does.

**Uncovered-source weight proposal — additions for sign-off** (all run at neutral 1.0 until Justin approves, alongside the v3 table in [[newsagent_observatory_v3_findings]]):

> **APPROVED AS PROPOSED — Justin, 2026-08-24. LIVE from that date**, together with the v3 rows (ING 0.9 / WP-CE 0.8 / unknown 0.5). Bloomberg newsletters and the JPM/GS/BofA research PDFs now carry 0.9 in `sourceweights.UNCOVERED_SOURCE_W`, keyed on their actual packet domains (`newsletter:Bloomberg`, `am.jpmorgan.com`, `jpmorganfunds.com`, `am.gs.com`, `ml.com`). α refit on activation: **2.7 → 2.85**. See [[newsagent_observatory_v33_findings]].

| Source | Proposed weight | Rationale |
|---|---|---|
| Bloomberg newsletters | 0.9 | major wire-adjacent outlet; newsletter form is analysis, a notch below the wire tier |
| Bank research desks (JPM/GS/BofA PDFs) | 0.9 | institutional research; single-house view, same tier as ING |

## Newsletter privacy — the leak that never fired

v3 marked newsletter items `display=False` and tested the *parser*, but `build_showcase` put **all** packet articles into the public evidence feed, drivers took raw breakdown titles, and the analytical FV-breakdown table rendered raw domains/titles. No credential existed, so nothing ever leaked — but this session's Gmail consent would have armed it the same day. Fix: the page boundary now filters `display=False` items from the evidence feed (showing only "+N private analysis items"), genericizes newsletter rows in drivers and the breakdown ("ING THINK (newsletter) — private analysis item"), and a regression test pushes a fake newsletter through packet+breakdown and asserts neither its title nor any `newsletter:` tag reaches the HTML. The live check on today's real page (an ING item sits in the Bardella packet): clean.

## Market tractability tags (Justin's call: keep all 24)

Four markets are tagged **data-driven** — their dominant information channel is not news text, so our news-FV is structurally blind there. Their cards carry a visible note and the ◆ marker in the grid; they stay scored in public (the honest-measurement point):

| Market | Why data-driven |
|---|---|
| Fed July no-change · Fed September no-change | rate odds price off Fed-funds futures/options-implied probabilities |
| Becerra CA governor | primary races move on private/campaign polling the packet never sees |
| CA billionaire wealth tax | ballot measures ride issue polling, not coverage |

**Practical example:** the Fed-July card now says "◆ not news-tractable — … the market carries information our packet cannot see" directly above a −35.8pp gap. A reader no longer has to wonder whether we think we know better than the mid on rates — we say explicitly that this gap is expected blindness, and the Brier will score it anyway. **Backlog (deliberately not done now):** restrict the news-FV universe to news-driven markets; add a rates-via-options/futures econ track as a separate labeled method.

## Ratings explainer — how the number formed

Each card renders `source_bias_breakdown`: relevant directional articles in the 72h window grouped by stance, then by Scheme-A reliability tier (higher-reliability w≥0.9 / mid 0.5–0.9 / more-biased <0.5); blocklisted items are counted but by construction carry zero weight. Example sentence from a test row: *"2 items leaned YES (higher-reliability: theguardian.com; more-biased/unreliable: someblog.example); 1 leaned NO (higher-reliability: bbc.co.uk); 1 relevant but neutral; 1 blocklisted-source item carried zero weight."* This is the "explanation of ratings" — a reader sees the bias mix behind the FV, not just the number.

## Dashboard UX (single long scroll replaced)

- **Default view:** 8 cards (flagged first, then top |gap|); the rest hidden. Controls: *default view / show all / flags only* + a **choose-markets picker** (checkboxes, localStorage-persisted). Grid rows link to their card and reveal it if hidden.
- **Collapsible cards:** header row (question + FV/mid/gap + flag) always visible; body toggles. Flagged cards start expanded, everything else collapsed — 24 markets now fit in a couple of screens.
- **2-column card body:** charts left (gauge + numerals + FV/band-vs-mid series), news right (ratings explainer, drivers, evidence feed) — the requested news-feed-beside-charts layout. Stacks on mobile.
- Self-contained as before (inline JS, no CDN; `node --check` on the emitted script passes; IP-scrub greps clean).

## Historical PM calibration — the big backfill

**Why:** the v3 α rode on 53 pairs / 9 markets from one geopolitically extreme month — honest but thin, and `band_mult` had never seen data.

**Design (pre-registered in the script docstring before the fit ran):**

1. **Universe:** Gamma sweep of closed+resolved politics/macro binaries — UMA-resolved, unambiguous 0/1 outcome, volume ≥ $200k, resolution ≥ **2026-02-15** (past the extraction/prior models' knowledge cutoffs), ≥14-day life, sports/meme/price-binary noise filters, **≤4 markets per coarse event family** (correlated same-event outcomes inflate n without adding information — the small-K cluster lesson). Result: **40 markets, 19 families, 12/40 YES** (Iran ×4, Fed ×4, Hungary ×4, Ukraine/Russia ×4, Israel ×4, Trump-admin ×4, plus Cuba, Epstein, Comey, Starmer, Taiwan-impeachment, Peru, Tamil Nadu, QatarEnergy, Bab-el-Mandeb, X-ban, aliens, anti-cartel, Omar singletons).
2. **Snapshots:** every 4 days over the last 30 days before each market's end (≤8/market) → **302 reconstructed packets**, Guardian+WP only (timestamped by construction; RSS/newsletters/PDFs are live-only and excluded). **Lookahead audit: 2,040 articles, 0 violations.** Retrieval keys were auto-drafted from question text (declared: no hand-tuning at this breadth; Stage-A relevance absorbs noisy packets, and 11 empty packets simply degrade to prior-only pairs).
3. **Stage A:** 1,872 unique (market, article) features via **16 token-cheap Haiku subagent batches** (117 each, ≤4 parallel; all validated + cached).
4. **Priors:** 40 five-perspective onboarding priors from each market's FIRST snapshot packet only (lookahead-free), via 4 Sonnet subagents, each with an **outcome-knowledge canary**. Result: **0 "known"** (nothing excluded), 2 "suspected" (both Hungary front-runner markets). One market (a Hormuz-blockade variant) produced no usable first packet → no prior → skipped; 39/40 entered the fit.
5. **Attention series:** one chunked GDELT-GKG BigQuery scan (2025-07→2026-06, ~37 GB ≈ 4% of the monthly free tier; the 25 GB per-query guard forced ≤70-day chunks).
6. **Mids (context only, never a fit target):** CLOB `/prices-history` months-old data is only served via `interval=max` + explicit `fidelity` — the startTs/endTs chunk path silently returns nothing, and closed markets additionally need `closed=true` on the Gamma slug lookup. Both probed live; 966 mid-days landed. (This extends the known prices-history gotchas.)

**Fit results (α grid on archive + backfill pooled; γ stays declared at 1.0):**

| Sample | n pairs | Brier(FV) | Brier(prior-only) | Brier(mid, context) |
|---|---|---|---|---|
| v0 archive (Jun 2026) | 53 | 0.427 | 0.463 | 0.278 |
| backfill (Feb–Jul 2026) | 302 | 0.186 | 0.191 | 0.126 |
| **pooled** | **355** | **0.2216** | **0.2317** | — |

Column read: one row per (market, snapshot-date) pair; `Brier(FV)` scores the evolved fair value at that snapshot against the market's eventual 0/1 outcome; `prior-only` freezes the onboarding prior (α=0); `mid` is the same-day market mid scored the same way — shown for context only (the mid is never a benchmark we fit to or claim to beat; it being better is expected and consistent with the closed gates).

- **α = 2.1** (was 1.8). Evidence still helps — modestly — at 6.7× the sample. Sensitivity: dropping the 2 canary-"suspected" Hungary markets moves α to 2.0 with essentially the same Brier → the fit is not riding on possibly-contaminated priors.
- **band_mult = 0.5** (was the declared 1.0). Pre-registered coverage rescale, executed on the backfill pairs (declared substitution: (market, snapshot) pairs rather than ledger forecasts — the forward ledger takes over as live markets settle). Bucketed coverage jumps 0.33 → 1.00 between band_mult 0.25 and 0.5; 0.5 is the narrowest knob at nominal. Plain English: **v3 bands were about twice as wide as resolved outcomes warranted.**
- **Honest caveat on the coverage detail:** realized YES-frequency sits at the *upper edge* of every covered bucket (e.g. bucket 0–20%: freq 20.1 vs band [1.2, 21.2]) — the model runs systematically a touch too NO-leaning on this sample (a status-quo-anchored engine under-reaching on YES resolutions). Coverage passes, but at the edge; worth rechecking at the first forward rescale.
- **Power honesty:** pairs remain clustered by event family (≤4 markets/family, but snapshots within a market are serially dependent and families share shocks). The pooled Brier is a point read, not a clustered CI; 12/40 YES base rate. The backfill Brier (0.186) is much lower than the archive's (0.427) mostly because the backfill sample includes many confident-NO markets — not because the model got better.

**Practical example (one backfill market end-to-end):** "Epstein suicide note released by May 8?" — the prior subagent, reading only the first packet (April), set a low prior; mid-window packets contained two items reporting the note *unsealed* (Stage A: `completed`, strength 0.95, toward_yes); the evidence state jumped, FV rose across the remaining snapshots. The market resolved NO (the specific criteria weren't met by the deadline) — the pair rows record exactly that trajectory and its Brier cost, which is the point: transparent, replayable calibration rows rather than a single opaque fit.

## Gemini provider flag (zero-cost extraction path — not yet trusted)

`--provider gemini` / `NEWSAGENT_EXTRACT_PROVIDER=gemini` routes Stage-A extraction through Gemini 2.5 Flash (free tier) with the identical prompt/validation. **Guardrail:** α was fit on Haiku-era extraction, so the provider must not flip silently — `scripts/newsagent_provider_spotcheck.py` re-extracts a deterministic sample of already-cached articles via Gemini into a comparison CSV (never overwriting the cache) and reports stance/phase agreement + mean |Δ| per numeric field. **Blocked on `GEMINI_API_KEY`** (not on this machine); until the spot-check passes, Anthropic/out-of-band remain the extraction paths.

## Attended run — what actually executed (2026-07-05, one session)

1. Gmail consent flow (browser step completed by Justin) → `secrets/gmail_token.json` (0600, refresh automatic); live probe: 10 ING THINK + 6 Bloomberg messages, read-only.
2. Daily loop: fetch (177 RSS, 4 newsletters in-window, 5/5 PDFs, GDELT refreshed) → 15 new extractions via 2 Haiku subagent batches → publish (24 markets, sf-2026-001…024 updated).
3. Backfill: discover (40 markets) → fetch (302 packets, ~55 min on the Guardian demo key) → 16 Haiku extraction batches + 4 Sonnet prior batches (≤4 parallel; `stay-within-limits` checked before the wave) → GDELT scan → mids → `--fit`.
4. Re-publish of the 24 live markets under the refit params (same-day republish is idempotent; ledger entries updated pre-settlement through the sf CLI).
5. Verification: 65 tests green; IP-scrub greps on the real page clean; inline-JS `node --check` pass; lookahead audit 0/2,040.

Cost: ~1,900 cheap-model extractions + 40 mid-model priors + 2 small daily batches, all in-session subagents; steady-state daily cost unchanged.

## Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions:** auto-drafted retrieval keys ≈ adequate historical coverage (declared; empty/noisy packets degrade to prior-only pairs, and 11/302 packets were empty); backfill (market, snapshot) pairs stand in for ledger forecasts in the first band_mult rescale (pre-registered knob, declared substitution); the family cap (≤4) bounds but does not eliminate event-cluster correlation; priors from present-day models answering as-of past dates are honest given the canary protocol (0 known, 2 suspected, sensitivity-checked); Scheme-A weight proposals for Bloomberg/bank-desk sources await sign-off (neutral 1.0 meanwhile); disclaimer-strip patterns are declared, not learned.

**Live-only unknowns** (the first four are now KNOWN — settled 2026-08-24, pooled Brier 0.0794 on n=4, [[newsagent_observatory_v33_findings]]): forward calibration of the 24-market book under the new α/band_mult (first resolutions: US–Iran meeting 07-17, Fed July 07-29, Hormuz + Iran-MOU 07-31); whether narrower bands change the divergence-flag fire rate (none fired at re-publish); Gemini feature agreement (blocked on key); newsletter/PDF marginal value to Stage-B (they now enter packets daily — their weight sign-off pending); the upper-edge coverage pattern (recheck at the first forward rescale).

## What Justin needs to do

1. ~~**Scheme-A sign-off** — one table now: v3's ING 0.9 / WP-CE 0.8 / unknown 0.5 **plus** Bloomberg 0.9 and bank research desks 0.9 (all running neutral 1.0 until then).~~ **DONE 2026-08-24 — APPROVED as proposed and activated** (α 2.7 → 2.85).
2. **`GUARDIAN_API_KEY`** — registered but not exported on this machine; the backfill ran on the demo key (worked, but it's the fragile path). ~~Add to the shell env or `secrets/`-based loading.~~ **The loading is now built (2026-08-24):** `config.load_env()` reads the git-ignored `polymarket/research/.env` at import, and that file carries a `GUARDIAN_API_KEY=` line **awaiting its value** — paste it there and the demo-key fallback stops being used. Still the one open credential item on this list.
3. **`GEMINI_API_KEY`** (free tier) — unblocks the provider spot-check; do not flip the provider before it passes.
4. ~~**July settlements** unchanged: `sf settle` + slate refresh + refit per the runbook.~~ **DONE 2026-08-24** — all four settled/scored, slate refreshed (4 retired, 4 added, Putin re-slug handled), α refit via `scripts/newsagent_hist_backfill.py --fit`. See [[newsagent_observatory_v33_findings]].
5. Website stays parked; `newsagent/AUTOMATION.md` updated for the new credential/provider reality.

## STRETCH / BACKLOG (marked; nothing blocks the shipped v3.1 slice)

- **News-driven-only universe restriction** + a **rates-via-options/futures econ track** (separate, labeled method for the Fed-type markets) — the tagged blindness makes both natural next steps.
- **Tone → direction calibration** once resolved outcomes accrue (tone stored+displayed, still unwired).
- **Event-driven cadence** (GDELT burst + caches make intraday re-scoring nearly free).
- **Web-launch automation** (documented switch in `newsagent/AUTOMATION.md`; needs keys + sign-off).
- **Forward band_mult recheck** at ≥20 settled ledger forecasts (the pre-registered ledger-based rescale; this round's pair-based rescale is the declared bootstrap).
- **UK re-entry** when an informative UK binary clears the liquidity floor.

## Outputs

- Package: `newsagent/email_ingest.py` (Gmail OAuth consent/refresh/fetch, Bloomberg + ingthink senders), `newsagent/pdf_ingest.py` (new), `features.py` (provider flag, `pick_provider`, `_gemini_rows`), `config.py` (`DATA_DRIVEN`, `tract()`), `fvmodel.py` (`source_bias_breakdown`, `band_components` refactor), `dashboard.py` (UX rework + privacy boundary), `run_daily.py` (PDF wiring, `--provider`, public-safe drivers), `AUTOMATION.md` (updated).
- Scripts: `scripts/newsagent_hist_backfill.py` (new: discover/fetch/ingest/pull-gdelt/fit), `scripts/newsagent_provider_spotcheck.py` (new).
- Params: `fv_params.json` — **α 2.1 (fitted, 355 pairs/49 markets) + band_mult 0.5 (coverage-rescaled)**; γ 1.0 + λ/floors/s_min/shift-clips declared, unchanged.
- CSVs: `newsagent_hist_pairs.csv` (355 rows: slug/family/src/date/p0/A/fv/y/mid + Briers + band components), `newsagent_hist_fit.csv` (α curve), `newsagent_hist_bandmult.csv` (coverage curve) under `data/analysis/csv_outputs/news_agent/`.
- Plot: `data/analysis/plots/news_agent/newsagent_hist_backfill_fit.png` (α curve + coverage curve, fit markers).
- Tests: `tests/test_newsagent_fv.py` — **65 green** (v3.1 adds: Gmail token preference/refresh-keep/consent pointer, Bloomberg+ingthink senders, PDF date templates/week-back/disclaimer-strip/degrade/cache/packet slot, provider selection/match-rows/no-key errors, tract tags, bias-breakdown grouping/newsletter labels, newsletter privacy end-to-end, UX render + collapse/expand + picker, `band_components`).
- Data (git-ignored): `data/newsagent/hist/` (universe, 302 packets, mids, priors incl. canaries), enlarged `feature_cache`, `pdf_cache`, `newsletter_cache`; dashboard artifacts regenerated.
- Ledger: sf-2026-001…024 updated (append-only, pre-settlement updates through the sf CLI).
