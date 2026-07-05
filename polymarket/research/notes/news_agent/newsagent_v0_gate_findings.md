---
title: "News-Agent Fair-Value Showcase — v0 Gate Findings: early-stop STOP, no dashboard"
created: 2026-07-05
status: closed — pre-registered early-stop fired; v1 dashboard gated out; redesigned gate proposed for sign-off
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
  - gate-result
---
# News-Agent Fair-Value Showcase — v0 Gate Findings

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]]
> Pre-registration (locked before any computation): [[newsagent_v0_gate_preregistration]] · Radar: [[newsagent_repo_data_radar_findings]] · Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

- **What this note is:** the result of the pre-registered v0 gate for the public news-agent fair-value showcase — the test of whether an LLM news-agent's independent fair-value % on high-liquidity politics markets is credible and legible enough to display next to the Polymarket mid.
- **The verdict: STOP (early-stop fired).** On the three pre-registered falsifier markets (the June-15 US–Iran peace deal, Starmer-out-by-June-30, Cepeda-wins-Colombia — 15 forecast/mid pairs), the news-agent's Brier was **0.547 vs the mid's 0.354** (diff **+0.194**, early-stop bar +0.15) and it moved in the same direction as big market repricings only **2/5 = 40%** of the time (bar ≥50%). Wrong *and* illegible — per the locked rule, the remaining 41 forecast calls were aborted and **the v1 dashboard was not built.**
- **The mechanism is specific, not vague:** the single-sample, status-quo-weighted forecaster behaves as a **status-quo machine** (3–12% on everything). It wins on quiet NO-markets (it beat the mid on all 6 Cepeda pairs) and gets destroyed on YES-transitions **even when the decisive evidence was in its packet** (it held 3% on the Iran deal on 06-14 with *"Trump says Iran peace deal could be signed by Sunday"* in evidence; the deal was signed that weekend).
- **What survived:** the entire pipeline (universe selection, lookahead-free packets, isolated forecasters, contamination canaries, scoring) works end-to-end and is reusable; the canaries were clean (no outcome leakage); and the failure diagnosis directly parameterizes a redesigned v0b gate — **proposed below for Justin's sign-off, not run** (rerunning a redesigned gate without sign-off would be gate-shopping).

## Verdict

**STOP.** The pre-registered cheapest falsifier fired: `Brier(ours) − Brier(mid) = +0.194 > +0.15` **and** `tracking 40% < 50%` on the falsifier trio. Under the locked decision rule this aborts the run and blocks the dashboard. The showcase concept is not killed — this specific forecaster design is. See § Redesigned gate proposal.

## What was run (design recap)

Design and metrics were locked in [[newsagent_v0_gate_preregistration]] before any computation; two deviations are documented there and here:

1. **Amendment 1 (packet source):** GDELT DOC 2.0 entered a persistent HTTP-429 state (reproduced from two unrelated IPs, >1.5h — including the Hetzner VPS), so packets switched to **Guardian Open Platform + Wikipedia Current Events daily pages** before any scored forecast existed. Both are time-stamped by construction; the Wikipedia page for the snapshot day itself is excluded (it aggregates the full day → would leak past the 12:00 UTC snapshot).
2. **Falsifier trio composition:** the pre-registration planned Iran-deal + Fed-June + Starmer; the Fed markets failed the pre-registered information filter (mid pinned >95c all month), so the trio became the top-3-by-volume across distinct families in the actual universe: **Iran-deal ($177M), Starmer ($7.9M), Cepeda ($7.3M)**.

Forecaster protocol as locked: one isolated Claude-Sonnet call per (market, snapshot); inputs = question + resolution criteria + "today is `<date>`" + ≤12 timestamped headlines (≤ snapshot instant); **no mid, no outcome, no tools** beyond one Read of its own prompt file; output = p% + 80% band + drivers. All events post-LLM-training-cutoff (June 2026).

**Contamination canaries (3, empty packet): CLEAN.** Vučić 5% [2,12], Starmer 5% [2,10], Iran-deal 2% [0,6] — all base-rate/status-quo answers on markets where two of three truly resolved YES. The isolation design works; packets were the only information channel. (`data/newsagent/v0/canary_results.json`)

## The falsifier table

Unit of observation: one (market, snapshot) pair. `ours` = news-agent fair value (fraction), `[band]` = its self-reported 80% credible interval, `mid` = Polymarket mid at the same 12:00 UTC instant (CLOB `/prices-history`, fidelity=60), `y` = resolved outcome, `Bo`/`Bm` = per-pair Brier for ours/mid (lower is better).

| family | date | y | ours [band] | mid | gap | Bo | Bm |
|---|---|---|---|---|---|---|---|
| iran_us_deal | 06-08 | 1 | 0.03 [0.01,0.08] | 0.055 | −0.03 | 0.941 | 0.893 |
| iran_us_deal | 06-11 | 1 | 0.03 [0.01,0.08] | 0.051 | −0.02 | 0.941 | 0.901 |
| iran_us_deal | 06-14 | 1 | 0.03 [0.01,0.10] | 0.229 | −0.20 | 0.941 | 0.594 |
| iran_us_deal | 06-17 | 1 | 0.03 [0.01,0.08] | 0.999 | −0.97 | 0.941 | 0.000 |
| starmer_uk | 06-08 | 1 | 0.06 [0.02,0.14] | 0.115 | −0.06 | 0.884 | 0.783 |
| starmer_uk | 06-11 | 1 | 0.06 [0.03,0.12] | 0.135 | −0.08 | 0.884 | 0.748 |
| starmer_uk | 06-14 | 1 | 0.05 [0.02,0.12] | 0.195 | −0.15 | 0.902 | 0.648 |
| starmer_uk | 06-17 | 1 | 0.05 [0.02,0.10] | 0.335 | −0.29 | 0.902 | 0.442 |
| starmer_uk | 06-20 | 1 | 0.08 [0.03,0.18] | 0.545 | −0.47 | 0.846 | 0.207 |
| colombia_election | 06-08 | 0 | 0.04 [0.01,0.10] | 0.165 | −0.13 | 0.002 | 0.027 |
| colombia_election | 06-11 | 0 | 0.03 [0.01,0.08] | 0.145 | −0.12 | 0.001 | 0.021 |
| colombia_election | 06-14 | 0 | 0.03 [0.01,0.08] | 0.105 | −0.08 | 0.001 | 0.011 |
| colombia_election | 06-17 | 0 | 0.12 [0.05,0.22] | 0.115 | +0.01 | 0.014 | 0.013 |
| colombia_election | 06-20 | 0 | 0.08 [0.03,0.18] | 0.115 | −0.04 | 0.006 | 0.013 |
| colombia_election | 06-23 | 0 | 0.03 [0.01,0.08] | 0.006 | +0.02 | 0.001 | 0.000 |

**Read:** the two YES-transition markets (Iran deal, Starmer) contribute all of the loss — our p never leaves single digits while the mid walks from ~5–13c to 55–100c. The NO-market (Cepeda) is the mirror image: we beat the mid on 6/6 pairs because the mid carried a persistent ~10–16c longshot premium against an outcome that never happened. A status-quo machine wins exactly and only where nothing happens.

## Pre-registered metrics (falsifier subset)

| Metric | Bar | Result | Pass |
|---|---|---|---|
| M1 credibility: Brier(ours) − Brier(mid) | ≤ +0.05 | **+0.194** (family-bootstrap 95% CI [−0.01, +0.34]) | **FAIL** |
| M2a divergence: median \|gap\| | ≥ 2pp | 7.5pp | PASS |
| M2b sanity: p90 \|gap\| ≤ 35pp AND Brier(ours) ≤ 0.25 | both | p90 28.5pp ✓, Brier **0.547** ✗ | **FAIL** |
| M2c tracking: direction agreement on mid moves ≥8pp | ≥ 65% | **40%** (2/5) | **FAIL** |
| M3 band (descriptive) | report | median width 9pp; last-snapshot band consistent with outcome in 1/3 markets | too narrow |
| **Early-stop rule** | diff > +0.15 AND tracking < 50% | fired | **STOP** |

CI note: the family-clustered bootstrap CI on the Brier difference ([−0.01, +0.34]) technically straddles zero at n=3 families — the early-stop was a point-estimate rule and would fire regardless; the CI's width is a sample-size statement, folded into § Realism read.

![Falsifier time series](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/news_agent/newsagent_v0_timeseries.png)

Caption: per market — PM mid (grey), news-agent p with 80% band (green), resolved outcome (red dashed). Axes are probability vs snapshot date (June 2026). The read: green stays pinned near zero while grey walks to the outcome on the two YES markets; the band almost never contains where the market ends up.

![Gap distribution](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/news_agent/newsagent_v0_gap_hist.png)

Caption: distribution of (our % − mid) in percentage points across the 15 pairs. Almost entirely negative — the agent sits below the mid nearly everywhere, i.e. a systematic under-reaction bias, not symmetric disagreement.

## Mechanism diagnosis (why it failed — specific and fixable)

1. **Status-quo anchoring dominates evidence.** The prompt's "the world changes slowly; weight the status quo" instruction (borrowed from the Metaculus template) plus Sonnet's conservatism produced near-identical 3–8% outputs regardless of packet content. The 06-14 Iran packet contained *"Trump says Iran peace deal could be signed by Sunday, with strait of Hormuz to open"* — the forecaster cited hardliner protests and held 3% while the market tripled to 23c. This is a prompt-design failure, demonstrably: the **same protocol did override the status quo when evidence was conclusive** (Cepeda 06-23: read the runoff-result headlines, reasoned "status-quo weighting overridden by convergent reporting"). It reacts to *settled facts*, not to *probability-shifting news* — precisely the thing a fair-value view must do.
2. **Single-sample point forecasts, no ensemble.** The radar's SOTA findings (Halawi et al.; ForecastBench) prescribe N-sample trimmed-mean ensembles with dispersion as the band; the locked v0 protocol used one call per pair. The self-reported bands (median 9pp wide) were far too narrow — a live shock (Iran deal, Starmer, Vučić) blows through them.
3. **Packet thinness (Amendment 1) is a real but secondary factor.** Single-outlet Guardian + curated WP bullets diluted with off-topic items (football, Brexit charts). Secondary because the decisive headlines *were present* in the failing pairs — retrieval succeeded; weighing failed.
4. **Context worth stating:** ForecastBench's published result — un-anchored LLMs score materially worse than market crowds (~0.113 vs 0.093 Brier on their bench, and that's for frontier ensembles) — makes this outcome unsurprising in hindsight. A shock-heavy month (the market itself sat at 5c three days before the Iran deal signed) maximized the penalty for under-reaction.

## Realism calibration read ([[CODEX]] § Realism calibration)

- **Fair knobs:** the mid as the comparison baseline (that's the showcase's own framing — the % is displayed next to the mid); lookahead-free packets; post-cutoff events; the +0.05 M1 tolerance (generous — it does not require beating the market).
- **Harsh-but-declared knobs:** one geopolitically extreme month (two regime-shock YES-transitions in three falsifier markets); Sonnet-class single samples standing in for the shipped ensemble; thinned packets under Amendment 1. None of these were imposed post-hoc — all were locked or documented before scoring — but they mean the STOP reads as **"this v0 design fails" not "no news-agent can be displayed."**
- **Power honesty:** 15 pairs / 3 families / one macro window. The early-stop was designed as a cheap kill for exactly this configuration; the CI straddling zero at n=3 families is expected and does not soften the rule-based verdict.
- **Assumption ledger:** modeled — Guardian+WP packets ≈ live news view; Sonnet ≈ shipped model floor; 12:00 UTC mids; every-3-days cadence. Live-only unknowns (moot until a redesign passes) — forward-sample calibration on the append-only ledger; band honesty over time; source-weighting effects.

## Decision and next step

- **v1 dashboard, daily ledger snapshots, and website integration: NOT BUILT** (gated out by this result). No code beyond the v0 gate pipeline was written.
- **The concept is not closed.** This is a first falsification of a specific cheap design with a mechanism-level diagnosis, on the never-run Block-J frontier — not a robust closure of the showcase idea. Per the reopen discipline, a redesign is legitimate *only* as a new pre-registered gate.

### Redesigned gate proposal (v0b — REQUIRES JUSTIN'S SIGN-OFF, not run)

One more cheap offline gate, changing exactly the diagnosed failure points, re-using the existing pipeline and the same 10-market universe (plus, if desired, a second month for regime diversity):

1. **Forecaster:** N=5 samples per pair, trimmed-mean aggregate, band = sample dispersion floored at ±8pp (Halawi/ForecastBench pattern, radar-verified MIT source for the harness if we adopt `forecasting-tools`).
2. **Prompt:** drop the blanket status-quo clause; replace with explicit evidence-weighting ("if credible reporting asserts an imminent qualifying event, the forecast must move materially; cite the driver") + base-rate step retained.
3. **Packets:** multi-source (GDELT when recovered + Guardian + WP + BBC/Sky/Politico RSS), with the Halawi relevance-rate-then-summarize step to kill off-topic dilution.
4. **Same M1/M2 bars, same early-stop, same falsifier trio.** If v0b also fails M1/M2 → close the showcase-as-fair-value framing entirely and (optionally) pivot the public page to something our results *do* support (e.g. "market-implied odds explained, with news context" — display journalism, no fair-value claim).
5. **Cost:** ~300 Sonnet samples ≈ small; zero new infra.

## STRETCH / BACKLOG (recorded for the concept, all gated behind a passing gate)

- v2 historical news→PM-move calibration via GDELT GKG/BigQuery V2Tone (token-cheap subagents; "this language historically moved politics mids X pp").
- Event-driven/intraday cadence; per-market news-burst triggers.
- ForecastBench-style interim scoring (unresolved forecasts scored vs prior-day mid) so a public track record has content from day one.
- Fincept-style news panel (RSS config → local FTS → ticker/feed/detail — design-only borrow, AGPL code never read).
- Source weighting Scheme A (Wikipedia RSP tiers + Iffy blocklist, CC-clean) — proposal table in [[newsagent_repo_data_radar_findings]], awaiting sign-off.
- Website integration into Epsilon-Fund/epsilon-webs1te (colleague owns UX): numbers-only default + analytical toggle, IP-scrub checklist.

## Outputs

- Pairs CSV: `data/analysis/csv_outputs/news_agent/newsagent_v0_pairs.csv` (falsifier subset — the early-stop aborted the other 41 calls)
- Metrics CSV: `data/analysis/csv_outputs/news_agent/newsagent_v0_metrics.csv` · family table: `newsagent_v0_family_table.csv`
- Universe CSV: `data/analysis/csv_outputs/news_agent/newsagent_v0_universe.csv` (60 candidates → 10 selected, 6 families)
- Plots: `data/analysis/plots/news_agent/newsagent_v0_timeseries.png`, `newsagent_v0_gap_hist.png`
- Raw (append-only): `data/newsagent/v0/` — candidates, price cache, 56 news packets (guardian+wp), 37 rendered prompts, 15 forecasts, canary results
- Scripts: `scripts/newsagent_v0_universe.py`, `newsagent_v0_news.py` (GDELT, kept for v1), `newsagent_v0_news2.py` (Amendment-1 fetcher), `newsagent_v0_prompts.py`, `newsagent_v0_score.py`
