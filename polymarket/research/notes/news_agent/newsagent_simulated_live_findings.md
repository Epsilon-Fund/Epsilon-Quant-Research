---
title: "Simulated-live reconstruction — replaying the model over 44 resolved markets, and what it says about run cadence (it is not the binding constraint)"
created: 2026-08-25
status: complete — 44 resolved markets, 326 reconstructed snapshots, pre-registered readout. Pooled Brier at resolution 0.1895 (log-loss 0.6359) against a 29.5% base rate; calibration REJECTED (Spiegelhalter Z +3.40, p 0.001) — the model leans systematically too far toward NO. The Brier-vs-staleness curve is FLAT, and the mechanism is that the reconstructed FV moves a mean of 3.35pp off its prior. Four settled July markets folded into the fit sample (355→379 pairs, 49→53 markets); alpha 2.85 → 2.75 (noise: the surface is flat 2.40–3.00), band_mult 0.5 → 0.75 (a degradation — no knob reaches nominal coverage any more). Dashboard v3.4 shipped, 699 tests green, NOT deployed.
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
  - reconstruction
---
# Simulated-live reconstruction — how accurate would this actually have been?

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]] · [[CODEX]] · [[TODO]] · Table terms: [[polymarket_table_dictionary]]
> Machinery reused from [[newsagent_observatory_v31_findings]] (the 302-packet lookahead-free backfill). Follows [[newsagent_observatory_v33_findings]] (the first four settlements) and sits beside [[newsagent_data_channel_v3_findings]] (the data channel, GO) and [[newsagent_agentreach_scoping]] (the reach extension). The closed gate in [[newsagent_v0_gate_findings]] stays closed and stays displayed.

## Plain-English Summary

- **What this is.** The Observatory has published real forecasts on 24 questions but only **four** have resolved, so the public accuracy story was a promise rather than a number. This pass builds the missing number a different way: take **44 markets that have already resolved**, replay what the current model *would have published* on each of them day by day, and score it. That replay is called the **reconstruction** throughout, it is kept permanently separate from the forward ledger, and it is never added to it.
- **The headline.** Pooled **Brier 0.1895**, log-loss **0.6359**, on 44 markets with a **29.5%** base rate. That looks respectable and mostly is not skill: the Murphy split puts **resolution at 0.0355** against **uncertainty 0.2082**, so almost all of it is the base rate.
- **The number that matters more.** The reconstruction forecasts **15.2%** on average against a **29.5%** realised YES rate, and **Spiegelhalter Z = +3.40 (p = 0.001) rejects calibration.** The model is systematically too NO-leaning — not a little, measurably. This confirms the "upper-edge coverage" worry flagged in v3.1 and turns it from a caveat into a finding.
- **The cadence question, answered — and the answer is not the one we expected.** Scoring the number the model would have been showing 1, 5, 9 … 29 days before each resolution gives a **flat** curve (balanced set: 0.157 at 1 day, 0.157 at 29 days, wandering 0.136–0.163 with no trend). Running the loop more often would **not** have improved this score. The mechanism is visible rather than mysterious: at resolution the reconstructed FV sits a mean of **3.35pp** from its onboarding prior and only **34%** of markets moved as much as 1pp, so prior-only scores **0.1920** against the model's **0.1895**. **In this reconstruction the binding constraint is not cadence; it is how much the evidence can move the number at all.**
- **Where does it work?** Not where we would like. Per tract: **news-driven n=34 Brier 0.2228**, poll-driven n=6 **0.0779**, data-driven n=4 **0.0741**. The method scores *worse* on the markets it is built for than on the ones our own tags call structurally blind — though base rates differ sharply across the three groups, so much of that gap is composition.
- **The honest refit.** The four settled July markets are folded into the fit sample (355 pairs/49 markets → **379 pairs/53 markets**). **α 2.85 → 2.75**, which is **noise, not learning** — the α surface is flat to four decimals from 2.40 to 3.00. **band_mult 0.5 → 0.75**, and that one is a real degradation: **no knob on the grid reaches nominal band coverage any more**.
- **One-line status:** reconstruction built and scored, dashboard v3.4 shipped with the staleness curve as its centrepiece, two real bugs found and fixed in the live data channel, **700 tests green**, **not deployed** — Justin pushes to Vercel.

---

## What this pass is and is not

| | |
|---|---|
| **Is** | A lookahead-free replay of the current model over 44 already-resolved markets, scored against a readout pre-registered before the first number existed. |
| **Is not** | The track record. That is the forward superforecasting ledger (`SF_BOOK=polymarket`), which is at **n=4** and is the only genuinely out-of-sample thing on the page. |
| **Is not** | Out-of-sample. **α was fitted on these same markets.** The reconstruction Brier is an *upper bound* on what the same model would score on markets it had not seen. This is stated on the public page, not just here. |
| **Is not** | A claim about the market mid. The [[newsagent_v0_gate_findings]] closure stands and is restated wherever divergence flags appear. |
| **Method scope** | **News-only.** The data channel (`news+data`, live on the September Fed market) has zero settled forecasts and no reconstruction history, so it does not appear in this track at all. |

**Practical example, one market end to end.** *"Strait of Hormuz traffic returns to normal by July 31?"* is one of the four settled markets folded in this round. Its onboarding prior of record — the five-perspective ensemble the live pipeline produced on 2026-07-05, before anything resolved — was **8.0%**. The reconstruction then replays seven snapshots (07-06 … 07-30) through the same Stage-A → Stage-B path the live loop uses: each snapshot's packet is rebuilt from Guardian and Wikipedia Current Events *as they stood on that date*, its articles are scored for relevance/stance/phase/strength, and the evidence state decays and accumulates. Through July the packets are dominated by blockade announcements, tanker strikes and an IRGC claim to have stopped four vessels — all `toward_no` — so the FV never lifts off a low prior. The market resolved **NO**. That row contributes a small Brier, and it is exactly the kind of row that makes the pooled number look better than the model is: an easy question the model was already right about before reading anything.

---

## 1 · Pre-registration — locked 2026-08-25 before any number existed

The full text lives in the docstring of `scripts/newsagent_simlive.py` (PR-1 … PR-10) and was written before the first reconstruction row was computed. In brief:

| Item | Locked |
|---|---|
| **PR-1 Sample** | Every market in the backfill universe with a lookahead-free onboarding prior whose canary is not "known": the 40 from v3.1 plus the 4 settled July markets. |
| **PR-2 Trajectory** | The v3.1 grid, **unchanged**: 4-day snapshots over the last 30 days, ≤8/market. Between snapshots the published number is the last one published — the same step-function semantics the forward ledger has. |
| **PR-3 Headline** | Pooled Brier + log-loss of the last number standing before resolution, one row per market. |
| **PR-4 Staleness** | Brier at each grid horizon, **raw** (all markets) and **balanced** (only markets with all 8 snapshots, so the shape cannot be an artifact of which markets drop out). Plus a **derived** cadence table. |
| **PR-5/6** | Murphy decomposition and reliability curve on 5 declared bins of width 20pp; thin bins marked, never dropped. |
| **PR-7 Splits** | Per family, and per tract with a mechanical assignment rule declared in advance (`config.tract()` where known; otherwise family `fed` → data, an election/ballot pattern → poll, else news). |
| **PR-8 Divergence** | Description only. No hit rate, no edge claim, closure restated. |
| **PR-9/10** | The evidence-poorer caveat, and permanent separation from the forward ledger. |
| **Verdict rule** | **None.** This is a measurement, not a gate. |

### 1a · Two declared deviations

Both are recorded here rather than in a commit message, because both change what the numbers mean.

**Deviation 1 — the family cap does not govern the four settled markets.** `FAMILY_CAP = 4` is a *discovery* rule: it stops a correlated cluster inflating n while adding no information. The four settled markets cannot arrive through discovery at all (iran and fed were already saturated), but they are our own forward slate's settlements and "refit on every settlement" is a standing commitment. They enter through a new `--add-settled` stage instead. The cost is real and is reported: the sample now carries **iran ×7** and **fed ×5**, so those two families are over-weighted relative to the declared cap.

**Deviation 2 — their prior is the ledger prior of record, not a fresh backfill prior.** The backfill protocol asks a model to set an onboarding prior from the first reconstructed packet and answer an outcome-knowledge canary. **The only forecaster available in this attended session had already read the v3.3 settlement table**, so its canary would honestly answer "known" and all four markets would be excluded. Instead the reconstruction uses the **five-perspective ensemble the live pipeline produced on 2026-07-05**, before any of them resolved. That is lookahead-free *by construction* rather than by self-report, which is strictly stronger than a canary. Two consequences, applied mechanically:

1. Their reconstruction **starts at 2026-07-05**, the day that prior actually existed, so no snapshot is anchored on a prior from the future. The clip is enforced in `snapshot_dates`, not by hand.
2. That prior was formed from a **live** packet (RSS, newsletters and PDFs included), so for these four markets the `p0` is richer than the Guardian+WP-only world the rest of the reconstruction lives in.

There is also a correlation to name: because those four markets share their prior with the forward ledger entries, the reconstruction and forward numbers are **not independent** for them. One more reason the two tracks are never summed.

---

## 2 · The sample

**44 resolved markets, 326 reconstructed snapshots, 2,239 article-features, 0 missing from cache.** Resolutions run 2026-02-15 → 2026-07-31; base rate 13/44 YES (29.5%).

Column meanings for everything below: *unit of observation* is one resolved market unless a table says otherwise. **Brier** = (p − outcome)², lower is better, 0.25 is what you get by always saying 50%. **base rate** is the share of that group's markets that resolved YES. **mean forecast** is the average probability the reconstruction was showing — printed beside every Brier precisely so a low score that is really just a low base rate cannot pass for skill.

---

## 3 · Results

### 3a · At resolution

| metric | value | how to read it |
|---|---|---|
| Brier | **0.1895** | mean squared error of the last number standing before resolution |
| log-loss | **0.6359** | punishes confident-and-wrong harder than Brier |
| base rate | 29.5% | 13 of 44 resolved YES |
| mean forecast | **15.2%** | the model's average probability — **half the base rate** |
| Murphy: reliability | 0.0240 | miscalibration (lower better) |
| Murphy: resolution | **0.0355** | discrimination — how far forecasts moved off the base rate |
| Murphy: uncertainty | 0.2082 | irreducible, set by the base rate |
| Spiegelhalter Z | **+3.399 (p = 0.001)** | \|Z\| > 1.96 **rejects** calibration — and it does |
| prior-only Brier | 0.1920 | the onboarding prior alone, α = 0 |
| mean \|FV − prior\| | **3.35pp** | how far evidence moved the number, on average |
| share moved ≥ 1pp | **34%** | how often it moved at all |

**Read.** Two things, and the second is the important one. First, the pooled Brier is respectable but is mostly the base rate: resolution (0.0355) is small against uncertainty (0.2082), so the model has little measured discrimination. Second, **calibration is rejected**. At n=44 that is a real test with real power, and it says the model is systematically too NO-leaning: it forecasts 15.2% where 29.5% happens. v3.1 saw the shape of this ("realized YES-frequency sits at the *upper edge* of every covered bucket") and called it a caveat to recheck; on 44 resolved markets it is no longer a caveat.

### 3b · Reliability at n = 44

*Unit of observation: one resolved market, binned by the reconstruction's final probability. Bins are the declared 5 × 20pp bins.*

| bin | n | mean forecast | observed YES | note |
|---|---|---|---|---|
| 0–20% | 33 | 6.2% | **21.2%** | the mass of the sample, and the miss is here |
| 20–40% | 6 | 31.5% | 33.3% | close |
| 40–60% | 4 | 51.6% | 75.0% | under-forecasts |
| 60–80% | 1 | 69.2% | 100.0% | thin |
| 80–100% | 0 | — | — | the model never gets this confident |

**Read.** Every populated bin sits **above** the diagonal — realised YES exceeds the forecast everywhere. That is one coherent failure, not scatter: 33 of 44 markets land in the bottom bin at a mean 6.2% and resolve YES 21.2% of the time. The model is good at saying "probably not" and bad at saying how *un*likely "probably not" is.

### 3c · Brier vs days before resolution — the centrepiece

![Reconstruction — Brier vs staleness, and reliability at n=44](../../data/analysis/plots/news_agent/newsagent_simlive.png)

**Chart read.** Left panel: pooled Brier against snapshot age, old on the left and fresh on the right, so "improving as we get closer" would be a line sloping **down** to the right. Terracotta is the **balanced** set (the 30 markets carrying all eight snapshots — the honest read, because its sample does not change along the x-axis); dashed grey is the raw set, where n falls with horizon as short-lived markets drop out. Right panel: the reliability curve of § 3b, dot size = markets in the bin, dashed diagonal = perfect calibration.

| days before resolution | n (raw) | Brier (raw) | n (balanced) | **Brier (balanced)** |
|---|---|---|---|---|
| 29 | 30 | 0.1568 | 30 | 0.1568 |
| 25 | 39 | 0.1586 | 30 | 0.1371 |
| 21 | 40 | 0.1863 | 30 | 0.1523 |
| 17 | 41 | 0.1528 | 30 | 0.1385 |
| 13 | 44 | 0.1885 | 30 | 0.1630 |
| 9 | 44 | 0.1855 | 30 | 0.1606 |
| 5 | 44 | 0.1760 | 30 | 0.1359 |
| **1** | 44 | 0.1895 | 30 | **0.1570** |

**Read, and this is the answer to the question Justin actually asked.** The curve is **flat**. A number published one day before resolution scores 0.1570; the same model's number 29 days out scores 0.1568. There is no trend — the wobble between 0.136 and 0.163 is noise on 30 markets. **In this reconstruction, a 26-day-stale snapshot was not meaningfully worse than a fresh one.**

That is not a licence to never run the loop, and the § 3a diagnostic says why: the reconstructed FV barely moves. Mean |FV − prior| is 3.35pp and only 34% of markets ever move 1pp, so a fresher snapshot is *nearly the same number* as a stale one — there is little for staleness to cost. The honest phrasing is: **cadence is not the binding constraint here; the evidence channel's ability to move the number is.** And that conclusion is bounded by § 5 — the reconstruction runs in a deliberately evidence-poorer world than the live loop.

**Derived — expected Brier by publish cadence.** *Derived, not measured, and labelled derived on the page too:* a number published under an N-day cadence has an age uniform on [0, N), so this averages the balanced curve over the horizons inside that window.

| run the loop | expected Brier |
|---|---|
| every 4 days | 0.1570 |
| every 8 days | 0.1464 |
| every 12 days | 0.1512 |
| every 16 days | 0.1541 |
| every 20 days | 0.1510 |
| every 28 days | 0.1492 |

**Read.** The column does not decrease as cadence tightens. Weekly-ish (8 days, 0.1464) is nominally the best cell and monthly (28 days, 0.1492) is nominally better than every-4-days (0.1570) — which is not a finding that monthly is *better*, it is the flat curve plus sampling noise showing up as a ragged ordering. The defensible statement is the negative one: **on this evidence there is no measurable accuracy return to running the loop more often.**

### 3d · Per tract — where does this actually work?

*Unit of observation: one resolved market. Tract assignment is the rule declared in PR-7(b), applied mechanically.*

| market type | n | Brier | log-loss | base rate | mean forecast | |
|---|---|---|---|---|---|---|
| **news-driven** | 34 | **0.2228** | 0.7501 | 32.4% | 13.6% | |
| poll-driven | 6 | 0.0779 | 0.2345 | 16.7% | 18.0% | |
| data-driven | 4 | 0.0741 | 0.2670 | 25.0% | 24.6% | thin |

**Read, and it is not the flattering answer.** The reconstruction scores **worst on the news-driven markets the whole method exists for**, and best on the ones our own tractability tags call structurally blind. Before reading that as "the model works where it cannot see", note the base rates: news-driven markets resolved YES 32.4% of the time against 16.7% for poll-driven, and a systematically NO-leaning model is flattered by a low base rate. Much of this gap is composition rather than skill — which is exactly why the base-rate and mean-forecast columns sit beside every Brier, on the page as well as here. What survives that caveat is narrower and still uncomfortable: **there is no evidence in this sample that the news channel does better where news is the channel.**

### 3e · Per family

Families with n ≥ 4 (the rest are singletons and are in the CSV):

| family | n | Brier | base rate | mean forecast |
|---|---|---|---|---|
| iran | 7 | **0.3960** | 42.9% | 10.2% |
| fed | 5 | 0.0782 | 40.0% | 33.5% |
| ukraine_russia | 4 | 0.3518 | 50.0% | 14.5% |
| israel | 4 | 0.0099 | 0.0% | 8.0% |
| hungary | 4 | 0.1162 | 25.0% | 25.2% |
| other:will | 4 | 0.1897 | 25.0% | 5.3% |
| trump_admin | 4 | 0.0222 | 0.0% | 9.0% |

**Read.** The damage is concentrated: **iran (0.396)** and **ukraine_russia (0.352)** are the two families where things actually happened — base rates 42.9% and 50% — and the model was showing 10.2% and 14.5%. The families it scores beautifully on (israel 0.0099, trump_admin 0.0222) are families where **nothing happened at all** (base rate 0%), and a NO-leaning model cannot lose there. This is the same story as § 3b told by event type: the model is priced for a quiet world and the sample's YES resolutions came from the two loudest families in it.

### 3f · Divergence flags — description only

Of **227** reconstructed snapshots that had a market mid to compare against, **4** cleared the divergence rule (|FV − mid| ≥ 15pp **and** band half ≤ 12pp **and** ≥ 5 relevant articles/72h), across **4** markets. **2** of those markets resolved YES; **3** of the 4 flagged snapshots had our number below the mid.

> **The v0 fair-value-vs-mid gate is CLOSED** — two pre-registered designs failed it ([[newsagent_v0_gate_findings]]) — **and is not reopened here.** The paragraph above describes where the model most disagreed with the market and how those questions resolved. It is not a hit rate, not an edge claim, and no mid-relative score is computed from it. This closure is restated on the public page in the same panel.

For completeness and *not* as a comparison: on the 27 markets that carry a mid at their final snapshot, the mid's Brier is 0.1273 and ours on that same subset is 0.1129. That figure is recorded here for the same reason every prior note records it — so a reader can see the market's own number — and is **deliberately kept off the public page's reconstruction panel**.

---

## 4 · The honest refit

The four settled markets are now in the fit sample, which is what v3.3 § 9 item 6 committed to.

| Run | sample | α* | pooled Brier at α* | prior-only | band_mult* |
|---|---|---|---|---|---|
| v3.3 (before) | 355 pairs / 49 markets | 2.85 | 0.2215 | 0.2317 | 0.5 (coverage 1.0) |
| **this pass** | **379 pairs / 53 markets** | **2.75** | **0.2104** | 0.2216 | **0.75 (coverage 0.6)** |

**Read the α move as noise, not learning.** The α surface is flat to four decimals from **2.40 to 3.00** (0.2104–0.2106), and the previous 2.85 scores **0.2104** on the new sample too — identical to the argmin. Nothing was learned about α; the sample grew 7% and the grid's argmin wandered 0.1. Reported because the procedure is pre-registered and its output stands, not because it means anything.

**The band_mult move is a real degradation and should not be read as a rescale.** The pre-registered selector takes the narrowest knob whose bucketed band coverage reaches the nominal 0.8, and falls back to closest-to-nominal only if none does. On this sample **none does**: coverage tops out at **0.6** (3 of 5 scoreable buckets) and stays there from 0.75 all the way to 3.0. So the selector took its fallback branch. Bands are not merely wider now — **at no width do they cover two of the five buckets.** Justin approved "keep 0.5, recheck at the next coverage rescale"; this is that recheck, and it moved.

Per-source split of the fit sample, reported so the settled-forward rows cannot hide inside the pooled number:

| sample | n pairs | Brier(FV) | Brier(prior-only) | Brier(mid, context) |
|---|---|---|---|---|
| v0 archive (Jun 2026) | 53 | 0.427 | 0.463 | 0.278 |
| historical backfill | 302 | 0.186 | 0.191 | 0.126 |
| **settled forward (the 4)** | 24 | **0.046** | 0.073 | 0.025 |

**Read.** The settled-forward rows score far better than either other block — and that is **not** evidence the model improved. Three of the four are quiet NO markets, and their prior is the richer live-packet prior of § 1a rather than a Guardian+WP reconstruction prior. They are not comparable to the backfill rows and are not offered as such.

**Two things were deliberately NOT refit.** α was **not** refit on reach composition — no reach item has ever entered a fitted packet ([[newsagent_agentreach_scoping]] § 9), so refitting on a composition change that never happened would be a false attribution. And α was **not** refit for the data channel — under **DC-4** `p_struct` enters through `p0`, not through `A_t`, so there is nothing to refit.

---

## 5 · The caveat that bounds every number above

**Reconstruction is an evidence-poorer world than the live loop.** It uses only channels with a timestamped archive: **Guardian** (date-bounded search), **Wikipedia Current Events** (whole past days) and the **GDELT-GKG** day-partitioned attention series. **RSS feeds, newsletters, macro-research PDFs and agent-reach official documents have no timestamped archive and are live-only** — they are absent from every reconstructed packet by construction. The live page reads all of them.

Which way that biases the score is **not** claimed in either direction, and it matters most for the cadence conclusion: a live daily loop ingests roughly three times as many articles per unit time as the 4-day reconstruction grid does (the packet's Guardian slot is capped at 6 items per fetch, so fetch frequency is itself an evidence-volume lever). The flat staleness curve is therefore evidence about **this** evidence diet, not proof that a richer daily loop would also be flat.

---

## 6 · Dashboard v3.4

Deployed epsilon-site design kept exactly (test-enforced palette); the v3.3 track-record panel and the reach display boundary are untouched.

| Item | Where |
|---|---|
| **§ 07 Simulated-live reconstruction** — pooled scores, the reliability curve, and the **Brier-vs-staleness curve as the centrepiece**, in a visually distinct dashed panel with a standing "reconstruction" chip and a first sentence that says what it is *not* | `dashboard._simlive_html`, `_svg_staleness` |
| **Per-tract split displayed** — news vs data vs poll with base rate and mean forecast beside every Brier | same panel |
| **In-sample warning** — α was fitted on these markets; the Brier is an upper bound | same panel |
| **Reach reality, in one line** — the page channel reaches **two** domains (federalreserve.gov, state.gov/releases); **ukmto.org stays navigation-only** for lack of a URL date; the dateline path is **T+1** | feed pane footer |
| **September Fed card** — `p_struct` anchor + evolved FV + mid + MPT-absence-in-words, the n=0 transition line, the double-count guard note, **the § 5 regime split with the holding row marked WORSE than doing nothing**, and an explicit line that today's FV *equals* the structural anchor because no news cleared the threshold | `_dc_regime_html`, `_tract_html` |
| Existing panels intact | donut grid (01), movers with per-row date gaps (02), divergence layer (03), market detail (04), retrospective gates (05), forward track record (06), reliability (08), method (09) |

Forward ledger stays the headline; the reconstruction is context, renders below it, and § 06 is referenced from § 07 as the only genuinely out-of-sample numbers on the page.

**Verification.** **700 tests green** across the research suite (17 new, covering the loader's rejection of anything not labelled `reconstruction`, the never-merge separation, absent-reconstruction degradation, the tract split, the staleness finding and its mechanism, the in-sample warning, the v0-closure restatement, the evidence-poorer caveat, palette enforcement inside the new section, the regime split including its absent-file degradation, MPT-dark not breaking the panel, and the reach note). Inline JS passes `node --check`; the page is fully static and self-contained (0 external refs); IP-scrub greps clean; verified rendered in headless Chrome (26/26 element checks on the real DOM). **A pre-existing time-dependent failure was found and diagnosed in passing.** `tests/test_spcx_pm_pdf_monitor.py::test_bucket_gap_sparkline_from_synthetic_two_poll_log` failed during the first runs of this session and passes now. It stamps two synthetic polls 30 minutes apart; inside the 30 minutes after UTC midnight those straddle a UTC calendar day, while `DashboardState.backfill_from_parquet(days=1)` reads today's partition only, so it loads 1 shard and asserts 2. The suite going green again as the clock moved past 00:30Z is the confirmation. **SPCX is a closed thread and this is not a newsagent defect**, so it is flagged rather than fixed; the repair belongs in that test (pin both timestamps to a fixed instant), not in the monitor. It is worth noting only because it is the same failure class as § 7 — a wall-clock window nobody had run in.

---

## 7 · Three bugs found in the live data channel, all fixed

Found by running the daily loop on a date that happened to fall inside a window nothing had exercised before.

**Bug 1 — FRED's clock is US Central, so the data channel is unrefreshable for hours every day.** `realtime_end = today` (UTC) is rejected with HTTP 400 for the whole window between UTC midnight and Central midnight, because FRED's own date is still yesterday. The snapshot died outright. Fixed by parsing the server's date out of its own error message and clamping — no timezone assumption is hardcoded, and clamping the *end* of a realtime window can only ever remove a later vintage, never add one.

**Bug 2 — the clamp poisoned the on-disk vintage cache.** With the HTTP call clamped but the cache filename still claiming the later date, every subsequent run read rows whose realtime spans all ended a day early, `as_of(t)` matched nothing, and the channel reported "inputs incomplete" **forever, with no error to look at**. Fixed by resolving the effective realtime end *before* the cache key is built, in `vintage_series`, so one decision keys both the request and the file. The poisoned files were deleted.

**Bug 3 — a failed refresh silently overwrote a good snapshot with an empty one.** The read side degrades gracefully on a *stale* snapshot (7-day bound, reason printed) but has no defence against a *fresh but empty* one: the live Fed card would have lost its structural anchor entirely rather than ageing. The snapshot script now refuses to overwrite `p_struct_latest.json` when a run produced no market, and says so.

All three are the same class of defect the reconstruction is about — **a pipeline that fails quietly in a window nobody ran it in** — and all three would have degraded the September 16 forecast, which is the first real test of the data channel.

---

## 8 · Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions.** The 4-day reconstruction grid is the v3.1 grid, reused unchanged, and it fixes the publish cadence the staleness curve is read *at* — the curve measures snapshot **age** at a fixed cadence, not the extra evidence a daily-fetching model would accumulate (§ 5). Stage-A features for the four folded-in markets were produced **out-of-band by an attended session model, not by the Haiku extractor α was fitted on** — the same provider-mix caveat v3.3 recorded, stamped into every one of the 194 cache records as `oob:simlive-2026-08-25`. The tract assignment rule for historical-only markets is declared and mechanical but is a judgment about which channel dominates a question. The family cap is breached by declaration for the four settled markets (§ 1a). Their prior is the ledger prior of record, formed from a richer live packet than the rest of the sample sees. Retrieval keys throughout the backfill are auto-drafted, never hand-tuned.

**Live-only unknowns.** Whether the flat staleness curve survives a richer evidence diet — the one thing this pass cannot test, and the single most important open question it raises. Whether the rejected calibration is fixable by a declared recentering or is telling us the prior ensemble is the problem. Whether the news-vs-poll/data tract ordering holds at larger n or is composition. Whether band coverage failing to reach nominal at any width is a band-model problem or the same NO-lean showing up in a second statistic. Whether the September 16 Fed settlement lands where § 5's holding-regime weakness predicts.

**Power honesty.** **n = 44 resolved markets and 326 snapshots, but the snapshots within a market are serially dependent and the markets cluster by event family (iran ×7, fed ×5, ukraine_russia ×4).** No confidence interval is computed and none should be quoted. Every aggregate here is a point read. The reconstruction is **in-sample for α**. The single result that carries real statistical weight is the calibration rejection (Z = +3.40), because it is a test with power at this n and it is driven by 33 markets in one bin rather than by a handful of outliers. The forward ledger remains at **n = 4**.

---

## 9 · Decision and next step

**Decision: the accuracy story on the page is now real, and it is less flattering than the promise was.** The model is respectable at ranking quiet questions, measurably mis-calibrated toward NO, has little measured discrimination, and — on this evidence diet — gains nothing from being run more often. All four statements are now on the public page with the numbers attached.

**The cadence answer, stated plainly for the decision Justin was not committing to:** there is no measurable accuracy return to a tighter run cadence in this reconstruction, so cadence should be chosen for **freshness of the published page and the ability to settle markets promptly**, not for score. The 12–26 day staleness that v3.3 called "the single biggest defect in this round's score" is, on 44 markets, **not** the defect — the evidence channel's inability to move the number is.

**Next, in order:**

1. **Attack the NO-lean, not the cadence.** The calibration rejection is the first result in this project with real statistical power behind it. The obvious candidates — a declared recentering of the prior ensemble, or revisiting the slow-market drip filter and shift caps that keep 66% of markets within 1pp of their prior — should be **pre-registered before anything is computed**, exactly as this pass was.
2. **Do not re-tune α or band_mult toward these numbers.** Both moved for procedural reasons on a flat surface; treating either as a signal would be fitting to noise.
3. **Decide what to do about band coverage.** No knob reaches nominal any more. That is a band-model question, and it is now blocking the honest use of the band on the public page.
4. **2026-09-16 remains the first real test of the data channel**, in the regime § 5 says it is weakest, with three freshly-fixed bugs standing between it and a silent failure. Settle it, score it, never merge it with the news track.
5. **Re-run the reconstruction with a richer diet if the archive ever allows it.** The single caveat that bounds this whole pass is that RSS/newsletters/PDFs/reach have no timestamped archive. If any of them ever gains one, the cadence question deserves re-asking.
6. **Deployment is Justin's.** Nothing here was pushed.

---

## 10 · Outputs

- **Scripts:** `scripts/newsagent_simlive.py` (new — pre-registration in the docstring, `--reconstruct` / `--score` / chart), `scripts/newsagent_hist_backfill.py` (new `--add-settled` stage, `--oob-source` provenance, `--gdelt-from/--gdelt-to` top-up, onboard-date clipping in `snapshot_dates`).
- **Data-channel fixes:** `scripts/newsagent_datachannel_dryrun.py` (`fred()` clock-skew clamp, `fred_today()`, cache-key resolution in `vintage_series`), `scripts/newsagent_datachannel_snapshot.py` (effective as-of date, refusal to overwrite a good snapshot with an empty one).
- **Package:** `newsagent/dashboard.py` — `_simlive`, `_simlive_bins`, `_svg_staleness`, `_simlive_html`, `_dc_regime_html`, `REACH_STATUS_NOTE`, reach note in the feed pane, `p0_used_pct` on the card, `.recon`/`.reconchip`/`.reconlead`/`.callout`/`.rowhi` styles, sections renumbered 07/08/09.
- **Params:** `newsagent/fv_params.json` — **α 2.75** (was 2.85), **band_mult 0.75** (was 0.5), 379 pairs / 53 markets; γ 1.0, λ/floors/s_min/shift-clips declared, unchanged. The notes field records the flat-surface reading and the coverage failure.
- **CSVs:** `newsagent_simlive_trajectory.csv` (326 rows: slug/family/tract/date/days_to_end/p0/A/fv/band/n_rel/mid/gap/flag/y), `newsagent_simlive_staleness.csv`, `newsagent_simlive_splits.csv`, `newsagent_simlive_reliability.csv`, plus refreshed `newsagent_hist_{pairs,fit,bandmult}.csv` — all under `data/analysis/csv_outputs/news_agent/`.
- **Plot:** `data/analysis/plots/news_agent/newsagent_simlive.png` (staleness curve + reliability), refreshed `newsagent_hist_backfill_fit.png`.
- **Run record** (git-ignored): `data/newsagent/simlive/simlive_results.json` — the full pre-registered readout, labelled `reconstruction`.
- **Data** (git-ignored): universe grows to 44 markets, 30 new reconstructed packets, 194 new feature-cache records stamped `oob:simlive-2026-08-25`, GDELT series topped up 2026-06-20 → 2026-08-02, 4 ledger priors of record written into `hist_priors.json` with the deviation note attached.
- **Ledger:** a real 2026-08-25 forward publish — sf-2026-001…028 updated (append-only), 47 new live extractions, data channel anchored at `p_struct` 50.7%.
- **Page** (git-ignored, regenerated): `data/newsagent/showcase/{index.html,showcase.json}` — 24 markets, new § 07.
- **Tests:** `tests/test_newsagent_fv.py` **+17** (112 in that file); repo-wide **700 green**.
- **Not touched:** any deployment.
