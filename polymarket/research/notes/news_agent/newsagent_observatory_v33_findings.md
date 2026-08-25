---
title: "Observatory v3.3 — the first public track record: four July markets settled and scored, every open sign-off recorded, Scheme-A weights activated (α 2.7 → 2.85), slate refreshed"
created: 2026-08-24
status: shipped — 4 settlements scored (first public Brier 0.0794), all v3/v3.1/v3.2 + both scoping sign-offs recorded, slate refreshed to 24 live markets, dashboard regenerated with a public track-record panel; NOT deployed (Justin pushes to Vercel manually)
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
  - calibration
  - settlement
---
# Observatory v3.3 — the day the scoreboard stopped being empty

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]] · [[CODEX]] · [[TODO]] · Table terms: [[polymarket_table_dictionary]]
> Follows [[newsagent_observatory_v32_findings]] (the shipped v3.2 page + lean axis) and closes the open sign-off blocks in [[newsagent_data_channel_scoping]] and [[newsagent_agentreach_scoping]].

## Plain-English Summary

- **What this round was.** The Calibration Observatory publishes our own probability on 24 liquid politics/macro questions and promises to be scored in public when they resolve. Four of them resolved in July and had been sitting **unsettled for 3–5 weeks**. This round pays that debt: the four are verified against Polymarket's resolution, settled in the append-only forecast ledger, and rendered as the **first public track record** the project has ever had. Everything else in the round follows from that: refresh the slate, refit the model, regenerate the page.
- **The number.** **4 settled, mean Brier 0.0794** (log-loss 0.3008; Spiegelhalter Z −0.86, p = 0.39 — i.e. n=4 cannot reject or confirm calibration, and the page says so). Three of the four were quiet NO markets we called well; the fourth — the July Fed decision, which resolved **YES** — is our worst row at Brier 0.214, and it is exactly the market our own tags call structurally blind.
- **The honesty tax on that number.** The observatory runs **attended, not daily**. The scored probabilities are the last ones we published *before* each market resolved, which makes them **12–26 days stale**. The page states this above the table, in the same size type as the Brier: staleness is part of the score, not an excuse subtracted from it.
- **Every open sign-off is now recorded.** Justin decided all twelve rows of the data-channel table, all three agent-reach blocks, and the three carried v3.2 items in one pass on 2026-08-24. The consequential one is **Scheme-A's uncovered-source weights going live** (ING 0.9 / Wikipedia-Current-Events 0.8 / Bloomberg 0.9 / bank research desks 0.9 / genuinely unknown 0.5, replacing a neutral 1.0). That changes what Stage B sees, so α was refit: **2.7 → 2.85**, with the pre-change baseline re-run first to prove the move is attributable to this decision and nothing else.
- **Slate.** Four resolved markets retired, four added (Hormuz-Oct, the Clarity Act, Le Pen, Flávio Bolsonaro), back to 24. A fifth surprise: Polymarket had silently **re-slugged** the Putin market, which would have forked a second ledger entry for the same question — handled with a `gamma_slug` indirection so our key never moves.
- **One-line status:** settled, refit, republished, tested (578 green) — **not deployed**; Justin pushes to Vercel.

---

## 1 · Settlements — the overdue four

### 1a · Verification before settlement

Every settlement was verified against Gamma before anything was written to the ledger. Closed markets do not come back from a plain slug lookup — the `closed=true` flag is required (the gotcha logged in [[newsagent_observatory_v31_findings]]) — and two of the slate's slugs had been **renamed by Polymarket** and returned zero rows even with the flag; those were recovered through `/public-search`.

Column meanings: *outcomePrices* is Gamma's settled `["Yes","No"]` price pair, so `["1","0"]` = YES and `["0","1"]` = NO; *closedTime* is when Polymarket closed the market (which can trail the question's own deadline — Hormuz-Jul closed on 08-04 for a 07-31 question); *uma* is the UMA oracle's resolution status.

| Market (our slug) | sf id | outcomes | outcomePrices | closedTime | uma | Outcome used |
|---|---|---|---|---|---|---|
| `us-x-iran-diplomatic-meeting-by-july-17-2026-…704` | sf-2026-004 | ["Yes","No"] | `["0","1"]` | 2026-07-18 | resolved | **NO** (0) |
| `will-there-be-no-change-in-fed-interest-rates-after-the-july-2026-meeting` | sf-2026-003 | ["Yes","No"] | `["1","0"]` | 2026-07-29 | resolved | **YES** (1) |
| `strait-of-hormuz-traffic-returns-to-normal-by-july-31` | sf-2026-002 | ["Yes","No"] | `["0","1"]` | 2026-08-04 | resolved | **NO** (0) |
| `will-iran-announce-withdrawal-from-mou-negotiations-by-july-31-…846` | sf-2026-009 | ["Yes","No"] | `["0","1"]` | 2026-08-01 | resolved | **NO** (0) |

**Did anything else resolve?** All 24 slate markets were re-checked the same way. Nothing else has closed: the other 20 are open, and the two "missing" slugs were renames, not resolutions (§ 4b). So the settled set is exactly these four.

### 1b · The scores

`sf settle <id> --outcome 0|1` per the vendored superforecasting skill ([[SKILL_MAP]] provenance; `SF_BOOK=polymarket`). The state machine scores at settlement and then refuses edits — that refusal is the whole point of publishing a track record at all.

*Unit of observation: one resolved market.* **our published p** is the probability standing in the ledger when the market resolved — not a number computed today. **Brier** = (p − outcome)²; lower is better; 0.25 is what you get by always saying 50%. **mid (context)** is the Polymarket mid at our last snapshot, shown for context only — the "our number beats the mid" claim is closed ([[newsagent_v0_gate_findings]]) and is not being re-litigated with n=4. **snapshot age** is the gap between our last published update and the resolution date.

| Market | resolved | our published p | outcome | Brier | mid (context) | snapshot age | method |
|---|---|---|---|---|---|---|---|
| US × Iran diplomatic meeting by July 17 | 2026-07-17 | 24.8% | NO | **0.062** | 38.5% | 12d | news |
| No change in Fed rates after the July meeting | 2026-07-29 | 53.7% | **YES** | **0.214** | 89.5% | 24d | news |
| Hormuz traffic returns to normal by July 31 | 2026-07-31 | 18.4% | NO | **0.034** | 16.5% | 26d | news |
| Iran announces MOU withdrawal by July 31 | 2026-07-31 | 8.9% | NO | **0.008** | 6.5% | 26d | news |
| **Pooled** | — | mean 26.5% | base rate 25% | **0.0794** | — | ≤26d | news |

**Full scorecard from the `calibrate` skill** (read-only over the same ledger; `PYTHONPATH=. uv run python -m lib.calibration.cli --book polymarket score`):

| metric | value | how to read it |
|---|---|---|
| Brier | **0.0794** | mean squared error of the four forecasts |
| Murphy: reliability | 0.0794 | miscalibration component (lower better) |
| Murphy: resolution | 0.1875 | discrimination — how far forecasts moved off the base rate |
| Murphy: uncertainty | 0.1875 | irreducible, set by the 25% base rate of this sample |
| log-loss | 0.3008 | punishes confident-and-wrong harder than Brier |
| ECE / MCE | 0.246 / 0.463 | calibration error across bins — **meaningless at n=4**, one market per bin |
| Spiegelhalter Z | −0.86 (p = 0.39) | \|Z\| > 1.96 would reject calibration-in-the-large; it does not |
| calibration-in-the-large | mean forecast 26.5% vs base rate 25% | +1.5pp of over-forecasting; noise at this n |

**Read.** The pooled 0.0794 looks excellent and mostly is not information: three of four were low-probability questions that resolved NO, which is the easy direction. The single informative row is the Fed market — a coin-flip forecast (53.7%) on a question our own model is tagged blind for, resolving YES. The Murphy split says the same thing without the story: resolution (0.1875) exactly equals uncertainty (0.1875), so **all** the apparent skill in this sample is the base rate, not discrimination. Four settlements is a start, not evidence; the page displays it with the n and the bands rather than a headline.

### 1c · The staleness caveat, stated where it costs us

The four scored numbers were published on **2026-07-05** and the markets resolved between 07-17 and 07-31 — the observatory did not run in between. A daily loop would have had 12–26 more days of evidence on each. Two consequences, both recorded on the public page:

1. **The score is what it is.** We do not re-run the model as of the resolution date and score *that*; the ledger is forward-only and settled entries reject edits. Scoring a retro-computed number would be exactly the post-hoc move the whole design exists to prevent.
2. **Cadence is now a first-class caveat.** The v3.2 "big movers" strip called its numbers *day-over-day*; with a 50-day gap between published snapshots that was simply wrong, and the wording is fixed (§ 6). The same fix labels every mover row with its own from/to dates and day gap.

---

## 2 · Sign-offs recorded (2026-08-24)

All decisions were Justin's, given in chat on 2026-08-24, and are written into the notes that carry the tables — not summarized here. Pointers:

| Block | Where the decisions now live | Net effect |
|---|---|---|
| Data-channel, 12 rows | [[newsagent_data_channel_scoping]] § 7 (+ § 7a "what changed") | OpenBB approved **for when the build starts**; AGPL boundary acknowledged; FRED key registered + loaded; BLS skipped; TradingEconomics **declined**; tag split applied; Option C + display-only + per-method tracks approved; **§ 4f DC-1…8 LOCKED** as the standing pre-registration (DC-5 = 1.0) |
| agent-reach, 3 blocks | [[newsagent_agentreach_scoping]] § Sign-off | Weights approved as proposed; official primary docs may show headline + source + link (bodies internal); reach slot ≤2 displacing Guardian 6→4 approved; **burner Twitter = NO**; skill reinstalled in English and recorded in `skills-lock.json` as a vendor-owned never-patch exception, kept machine-wide |
| Carried v3.2 items | [[newsagent_observatory_v3_findings]], [[newsagent_observatory_v31_findings]], [[newsagent_observatory_v32_findings]] | Scheme-A uncovered weights **APPROVED → activated** (§ 3); AllSides lean table + extremity multipliers approved **as shipped** (no code change); band_mult selector call approved — **keep 0.5**, recheck at the next coverage rescale as pre-registered |

Two of those deserve emphasis because they change behaviour rather than paperwork: the **Scheme-A activation** (§ 3) and the **per-method ledger labelling** shipped ahead of the data channel it exists for (DC-8 — a method label added *after* a method switch is worthless, so it goes in first).

---

## 3 · Scheme-A activation and the α refit

### 3a · What changed

`sourceweights.py` previously returned a neutral **1.0** for every source outside the Wikipedia perennial-sources (RSP) news-outlet scope, deliberately, so that nothing moved before sign-off. Those sources now carry declared weights:

| Source type | Packet domain(s) | Weight (was 1.0) |
|---|---|---|
| ING research newsletters | `newsletter:ING THINK`, `newsletter:ING research` | **0.9** |
| Bloomberg newsletters | `newsletter:Bloomberg`, `bloomberg.com` | **0.9** |
| Bank research desks | `am.jpmorgan.com`, `jpmorganfunds.com`, `am.gs.com`, `ml.com` | **0.9** |
| Wikipedia Current Events | `en.wikipedia.org (Current events)` | **0.8** |
| Genuinely unknown domains | anything else with no RSP tier | **0.5** |

Precedence is unchanged and test-enforced: the **Iffy blocklist wins over everything** (a blocklisted domain cannot be resurrected by a declared row), then the RSP tier, then these rows, then the 0.5 default. The weights scale Stage-B contribution and band dispersion only — never Stage-A extraction.

### 3b · The refit, and why the α move is attributable

Activating weights changes the evidence scale, so α (the single fitted parameter) must be refit or it is silently mis-scaled. Both fits ran on the **same** 355-pair / 49-market resolved sample (v0 archive + the v3.1 historical backfill), same declared γ = 1.0, same code path (`scripts/newsagent_hist_backfill.py --fit`). The only difference between the two rows is the weight table:

| Run | α* | pooled Brier at α* | prior-only Brier | band_mult* (coverage vs nominal 0.8) |
|---|---|---|---|---|
| Neutral 1.0 (pre-sign-off baseline, re-run today) | **2.70** | 0.2214 | 0.2317 | 0.5 (1.0) |
| **Approved Scheme-A weights (live)** | **2.85** | 0.2215 | 0.2317 | 0.5 (1.0) |

**Read.** The baseline re-run reproduces the v3.2 α of **exactly 2.7**, which is what licenses the attribution: the entire **+0.15** move is the uncovered-source weighting and nothing else — no data change, no code drift, no re-extraction. Mechanically it is compensation, not learning: down-weighting Wikipedia digests to 0.8 and unknown sources to 0.5 shrinks the daily evidence score `A_t`, so the fitted α rises to keep the same total shift. The pooled Brier is **unchanged to four decimals** (0.2214 → 0.2215). That is the honest summary: this decision bought a defensible, declared reliability scale, not accuracy. It will start to matter when the packet actually broadens — which is precisely when agent-reach's official-source rows (§ 2) would arrive.

`band_mult` stays **0.5** under the new weights (bucket coverage 1.0 vs nominal 0.8 on the 302 backfill pairs) — the same value the approved selector reading gives, so Justin's "keep 0.5" call costs nothing today. The pre-registered forward rescale still waits for ≥20 settled forecasts; we are at 4.

### 3c · What the refit does *not* include

The four freshly settled markets are **not** in the fit sample. `--fit` runs over the v0 archive plus the historical backfill universe; adding today's resolutions would mean reconstructing their pre-close packets and Stage-A features through `--discover`/`--fetch`/`--ingest` first. That is a legitimate next round (n grows from 49 markets to 53), but it is not what "refit on settlement" did today, and pretending otherwise would overstate the sample. The settled four live in the **forward ledger**, which is the track record — a different and stricter object than the fit sample.

---

## 4 · Slate refresh

### 4a · Retired and added

Retired (resolved, settled, recorded in `config.RETIRED_MARKETS` with their sf ids): the four from § 1. Added by hand from `scripts/newsagent_v3_universe.py` under the standing slate rule — binary, open, informative mid (5–95c), liquidity ≥ $100k, ≤ 2 per event family:

| New market | mid at add | liquidity | family (count after) | mtype | tag |
|---|---|---|---|---|---|
| Hormuz traffic normal by **Oct 31** | 15.5% | $210k | hormuz (2/2, with the Dec market) | shock | news |
| **Clarity Act** (H.R.3633) signed into law in 2026 | 16.5% | $524k | us-legislation (1/2, **new family**) | slow | news |
| **Marine Le Pen** wins the 2027 French election | 31.7% | $167k | france (2/2, with Bardella) | slow | news |
| **Flávio Bolsonaro** wins the 2026 Brazilian election | 34.3% | $221k | brazil (2/2, with Lula) | slow | news |

Rationale for the picks, briefly: Hormuz-Oct is the direct successor of the resolved Hormuz-Jul and keeps a near-dated question in that family; the Clarity Act is the slate's **first non-election, non-geopolitics question** and sits squarely in the Politico/The Hill coverage our RSS set actually reads; Le Pen and Bolsonaro pair with candidates already on the slate for the *same* election, which is correlated but not a 1−p mirror (both races have other candidates), and both carry informative mids.

Two drift flags for the next refresh, not acted on because only *resolved* markets get retired: **Trump Nobel has drifted to 2.1%**, below the 5c informative floor the slate rule uses at selection time, and **Becerra sits at 95.2%**, at the top edge. Neither is dishonest to keep — they are live forecasts — but both are now uninformative by our own selection criteria.

### 4b · The Putin re-slug — a fork that nearly happened

`putin-out-before-2027` returned **zero rows** from Gamma, with and without `closed=true`. It had not resolved: Polymarket had re-slugged it to `putin-out-before-2027-346` (the same happened to the Iran-MOU market, which had genuinely resolved). Had the slate simply been updated to the new slug, every piece of state keyed by the old one — ledger entry **sf-2026-001**, the stored prior, `fv_state`/`fv_series`, the GDELT series, and the Stage-A feature cache (`sha1(slug|title)`) — would have been orphaned, and the next publish would have created a **second ledger entry for the same question**. That is a silent integrity failure in an append-only book.

Fix: a `gamma_slug` field in the market config. Our key never moves; only the Gamma lookup follows the rename (`feeds.market_state(slug, gamma_slug)`), and a lookup that returns nothing now raises a message that names the cause and the fix instead of an `IndexError`. Tests: `test_market_state_follows_a_reslug_without_moving_our_key`, `test_market_state_raises_a_useful_error_when_gamma_has_nothing`.

---

## 5 · The 2026-08-24 publish

A full attended run: fetch → onboard (4 new priors) → Stage-A extraction (199 items) → publish → ledger → dashboard. Ledger entries **sf-2026-025…028** were created for the four new markets; the other 20 were updated in place. Round facts:

| Channel | Result |
|---|---|
| GDELT attention (BigQuery) | refreshed 2026-08-04 → 08-24 for all 24 markets; the credential now loads from `.env` (§ 6) |
| RSS (BBC/Sky/Politico/The Hill) | 193 items across the feed set |
| Guardian | live (demo key — `GUARDIAN_API_KEY` still awaiting its value) |
| Macro PDFs | 5/5 reports |
| **Newsletters** | **DOWN** — the Gmail OAuth refresh token has been revoked/expired since it last ran on 2026-07-05 (Google expires refresh tokens for unverified/testing apps). Needs a re-consent; see § 6 for the bug this exposed |

### 5a · Extraction provenance — a caveat that must not be buried

There is no `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` on this machine, so Stage A ran through the documented **out-of-band** path: the pipeline emitted 199 pending extractions and the attended session produced the features under the unchanged v3 prompt semantics (`PROMPT_VERSION 3`). Previous rounds used **Haiku** subagents, and **α is fitted on Haiku-era features**. Today's snapshot therefore mixes a different extractor with a Haiku-era α — a provider change in everything but name.

It is recorded rather than hidden: every one of the 199 cache records carries `_meta.source = "oob:attended-2026-08-24 (claude-opus-5 in-session; alpha era = haiku-4-5 …)"`, and the four new priors carry the same attribution in `priors.json`. The v3.1 rule — *never flip the provider silently, spot-check first* — is unchanged and this is exactly the comparison `scripts/newsagent_provider_spotcheck.py` exists to make. Treat today's FVs as provider-mixed until that spot-check runs.

### 5b · What the page says today

Three divergence flags fired (the rule is unchanged: |FV − mid| ≥ 15pp **and** band half ≤ 12pp **and** ≥ 5 relevant articles in 72h):

| Market | FV | mid | gap | why it cleared the confidence leg |
|---|---|---|---|---|
| No change in Fed rates after September | 44.0% | 66.5% | −22.5pp | 6 relevant articles, band ±8.1pp |
| Democrats control the Senate | 31.7% | 49.5% | −17.8pp | 5 relevant, band ±8.3pp |
| Russia × Ukraine ceasefire by Dec 31 | 5.6% | 21.5% | −15.9pp | 6 relevant, band ±12.0pp |

And the largest gap on the whole page — **Becerra, FV 14.0% vs mid 95.2%, −81.2pp** — correctly did **not** flag: zero relevant articles in 72h, band ±14.5pp. That is the confidence leg doing its job on a market our tags now call **poll-driven**: we are blind, the model says something extreme, and the design refuses to dress it up as a disagreement worth showing.

**One behaviour worth flagging for review, recorded as it happened.** Marine Le Pen's card moved from its onboarding prior of 25.0% to **59.9%** on a *single* article — a Guardian opinion column reporting that her candidacy ban is lifted and calling her a bigger threat than Bardella. 59.9% is not a coincidence: it is the slow-market shift cap (±1.5 logits from the prior) binding exactly, the first time it has bound on a live single-item day. Two readings, both honest: the guard worked (the raw move was larger), and a 35-point move on one column is more than an opinion piece should buy. The input is *not* being revised after seeing the output — that would be the post-hoc edit this whole system is built to refuse — so the number is in the ledger as published, and the open question for the next round is a **prompt-semantics** one: for a "wins the election" question, should "a barrier to candidacy was removed" be `event_phase: speculative` rather than `in_progress`? That is a rule change, decided before the next extraction, not a retro-fit to this card.

---

## 6 · Code changes, and three real bugs found

Everything below is in `polymarket/research/`; tests are in `tests/test_newsagent_fv.py` (95 newsagent tests; **578 green across the whole research suite**).

| Area | Change |
|---|---|
| `config.py` | tag split `DATA_DRIVEN` / `POLL_DRIVEN` (+ `NOT_NEWS_TRACTABLE`, `tract()` → `news\|data\|poll`, `tract_note()`); refreshed Becerra copy; `load_env()` + `ENV_KEYS`; `DATA_CHANNEL_MARKETS` (empty by design); slate refresh + `RETIRED_MARKETS` + `gamma_slug` |
| `sourceweights.py` | uncovered-source weights activated (§ 3a); resolution order documented; blocklist precedence made explicit |
| `ledger.py` | `method_for` / `record_method` / `method_of` (DC-8 labelling), `settled_records()` (read-only), `_last_forecast_dates()` |
| `dashboard.py` | new **public track-record panel** (§ 06) with per-method scores, the n=0 transition line, and the staleness note; `◆ poll-driven` chip + copy; movers wording fix; `_extended_scorecard()` (calibrate, optional) |
| `feeds.py` | `market_state(slug, gamma_slug)` + a useful error when Gamma returns nothing |
| `email_ingest.py` | dead-credential degradation (below) |
| `run_daily.py` | passes `gamma_slug` and the run date through; uses `tract_note()` |

**Bug 1 — a dead credential could stop the daily run.** The revoked Gmail token raised straight out of `--stage fetch`, killing the whole run before any market was fetched. Every other optional channel degrades (GDELT prints `skipped`, PDFs degrade weekly); newsletters did not. Now they do, with the reason printed and the re-consent command in the message. Test: `test_dead_newsletter_credential_degrades_instead_of_breaking_the_run`.

**Bug 2 — "snapshot age" was negative.** The first cut of the track record read `updated_at` from the ledger's active file, but `sf settle` rewrites that field to the settlement instant — so a 26-day-stale forecast looked *fresher than the resolution*. Fixed by reading the last probability-bearing event from the append-only `events.jsonl`, which is the only field settlement cannot touch. This one matters beyond cosmetics: the staleness caveat is a public honesty claim, and it was silently wrong in the direction that flattered us.

**Bug 3 — `showcase.json` stopped being JSON.** `calibrate`'s scorecard includes a pandas reliability table; passing it straight into the page payload broke serialization. The extended scorecard is now reduced to scalars, and the headline Brier is computed from the ledger by dashboard code, so **the page never depends on `lemma-calibrate` being installed** — calibrate stays the scorer of record for the note, not a runtime dependency of the daily loop.

**Also fixed:** the `.env` parser treats `KEY=   # not pasted yet` as **unset** (an earlier cut armed the comment text as a Guardian key — it would have sent garbage to the API), and relative `*_CREDENTIALS` paths resolve against the `.env` file so a run from another cwd still finds the GDELT service account.

---

## 7 · Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions.** Gamma's `outcomePrices` is the resolution of record (UMA `resolved` on all four). The last published snapshot is the forecast of record, staleness included. Today's Stage-A features come from a different extractor than the one α was fitted on (§ 5a) — declared, not measured. The four new onboarding priors are single-shot five-perspective ensembles from the same attended session; their quality is exactly what the forward ledger will measure. The uncovered-source weights are declared judgments, not fits. Family counts in § 4a use coarse hand buckets.

**Live-only unknowns.** Whether the four settled markets are representative of anything (n=4, three of them easy NOs). Whether a daily cadence would have improved or hurt these four — untestable now, and the next resolutions are the only way to find out. Whether the attended-extractor snapshot differs materially from Haiku features (blocked on the spot-check / a key). Whether the slow-market shift cap is the right magnitude when a single high-clarity item lands (§ 5b). Whether the Gmail re-consent holds longer than 7 days without app verification.

**Power honesty.** n = 4 settled forecasts. Every aggregate in § 1b is a point read on that sample: ECE/MCE are uninterpretable at one market per bin, Spiegelhalter Z does not reject calibration and could not at this n, and the Murphy split shows resolution exactly equal to uncertainty — i.e. **no measured discrimination yet**. The page presents the number with the n attached and no claim beyond it. The α refit sample is unchanged at 355 pairs / 49 markets, and the +0.15 move is a mechanical rescale, not evidence of improvement.

---

## 8 · Outputs

- **Ledger** (git-ignored, append-only): `data/superforecast/forecasts/` — sf-2026-002/003/004/009 **SETTLED + SCORED**; sf-2026-025…028 created; `ledger_methods.json` added (per-snapshot method labels).
- **Params:** `newsagent/fv_params.json` — **α 2.85** (was 2.7), band_mult 0.5, γ 1.0, λ/floors/clips declared, sample 355 pairs / 49 markets.
- **Page** (git-ignored, regenerated): `data/newsagent/showcase/{index.html,showcase.json}` — 24 markets, new § 06 public track record, verified in headless Chrome.
- **Fit artifacts:** `data/analysis/csv_outputs/news_agent/newsagent_hist_{pairs,fit,bandmult}.csv`, `data/analysis/plots/news_agent/newsagent_hist_backfill_fit.png`.
- **Day state:** `data/newsagent/live/2026-08-24/` (24 market+packet files, 199-item extraction work list), 199 new feature-cache records, 4 new priors.
- **Repo:** `skills-lock.json` gains the `agent-reach` never-patch entry; `.env.example` documents the credential names.
- **Notes updated:** both scoping notes (decisions recorded), v3/v3.1/v3.2 findings (carried sign-offs answered), the hub, and `brain/TODO.md`.
- **Not touched:** any deployment. The Vercel push is Justin's.

## 9 · Decision and next step

**Decision: the observatory now has a public track record and it is honest about being four markets deep.** Nothing in this round claims skill; the Murphy decomposition says explicitly that the sample's apparent quality is its base rate. The settlement debt is paid, the sign-offs are recorded where the tables live, and the model's one fitted parameter moved for a stated, attributable reason.

**Next, in rough priority order:**

1. **Run the loop more than monthly.** The single biggest defect in this round's score is a 12–26 day snapshot age. Even a weekly attended run would halve it. (Unattended still needs `ANTHROPIC_API_KEY` + cron-env credentials per `newsagent/AUTOMATION.md`.)
2. **Re-consent Gmail** (`PYTHONPATH=. uv run python -m newsagent.email_ingest --consent`) and paste `GUARDIAN_API_KEY` into `.env` — the loading is now built, the values are the only gap.
3. **Provider spot-check** before leaning on any future non-Haiku extraction (`scripts/newsagent_provider_spotcheck.py`, blocked on `GEMINI_API_KEY`).
4. **Data-channel step 3** — the retrospective dry run of the declared structural method on the resolved **July Fed** market (our worst row, 53.7% on a YES). If a vintage-stamped reaction function cannot beat that, the channel is not worth building. § 4f is locked and waiting.
5. **Extraction-semantics question** from § 5b (barrier-removed → `speculative` vs `in_progress` on "wins the election" questions), decided **before** the next extraction wave.
6. **Fold the four settled markets into the fit sample** on the next refit (via `--discover`/`--fetch`/`--ingest`), which is the honest way to make "refit on every settlement" literally true.
