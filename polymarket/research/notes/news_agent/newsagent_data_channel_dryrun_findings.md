---
title: "Data-channel dry run — scoring the resolved July-2026 Fed market with vintage-only structural inputs: the declared reaction function FAILS its pre-registered bars (NO-GO)"
created: 2026-08-24
status: complete — NO-GO. p_struct scored Brier 0.4223 against the news-FV's 0.2144 on the same days (outcome YES); directional sanity and stability both passed. Nothing built, OpenBB NOT installed, no model/dashboard/ledger touched.
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
  - data-channel
  - falsifier
---
# Data-channel dry run — can official statistics score a Fed market the news packet cannot see?

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]] · [[CODEX]] · Parent: [[newsagent_data_channel_scoping]] (§ 4f DC-1…DC-8, LOCKED 2026-08-24) · Table terms: [[polymarket_table_dictionary]]
> Follows [[newsagent_observatory_v33_findings]] (the settlement that made this test possible). **Nothing was built.** `fvmodel.py`, the dashboard, the ledger and `fv_params.json` are untouched; OpenBB is still **not** installed.

## Plain-English Summary

- **The test.** The scoping pass designed a second evidence channel for the two Fed markets: instead of news, build the probability from official statistics (`p_struct`). Before building anything, the cheapest possible falsifier was pre-registered: run the **declared** method backwards over the **resolved** July-2026 Fed market, feeding it only what was publicly known on each day, and see whether it beats the number the Observatory actually published.
- **The answer is no, and it is not close.** Over the 15 days where both numbers exist, `p_struct` scores **Brier 0.4223** against the news-FV's **0.2144** — the market resolved **YES (no change)** and the structural method drifted *away* from it, from 44% down to 34%. It also loses to prior-only (0.2144) and to its own base-rate-only benchmark (0.2236). **Verdict: NO-GO.**
- **The failure is specific and diagnosable, not vague.** The June SEP moved the Fed's own end-2026 median from 3.4% to 3.8% while the target midpoint sat at 3.625% — a genuine hawkish signal that the declared rule detected correctly. The rule then **spread that pressure evenly over the remaining meetings**, which pushed down the probability of holding at the *next* meeting. The market did the opposite and was right: the Atlanta Fed tracker (display-only) shows the hike being priced into the **September–December** window (P(hold) → 0.14) while the window containing **July** stayed at 0.82–0.89. The signal was real; the **allocation across meetings** was wrong.
- **Two of three criteria passed.** Directional sanity: **0 violations** — every mapped release moved `p_struct` the right way. Stability: max single-day move **0.174 logits** against a 1.5-logit reference — the re-anchoring is gentle, not violent. The method is well-behaved. It is simply worse than doing nothing.
- **The honest headline.** On this market the *data term actively hurt*: a pre-registered sensitivity sweep shows the Brier improving monotonically as the macro adjustment is turned off (b1 → 0 gives 0.2236). Whatever the channel is worth, this reaction function is worth less than its own base rate.
- **One-line status:** falsifier run as specified, NO-GO recorded, amendment proposed for a future pass, **nothing built and no knob changed after seeing a result**.

## What this pass is and is not

| | |
|---|---|
| **Is** | One retrospective trajectory (57 days) on one resolved market, computed from the § 4f declared method with strictly point-in-time inputs, scored against criteria locked before the first number existed. |
| **Is not** | A calibration claim, a fit, or a build. n = 1 market — no CI is claimable and none is claimed ([[CODEX]] § Anti-patterns). Nothing was fitted on this market; no constant was changed after seeing an output. |
| **New files** | `scripts/newsagent_datachannel_dryrun.py`, its CSV/plot outputs, and this note. |
| **Untouched** | `newsagent/fvmodel.py`, `config.py`, `dashboard.py`, `fv_params.json`, the sf ledger, every published page, and the research venv (no OpenBB install — row 1 of the sign-off authorises it only "when the build starts"). |

---

## 1 · The pre-registration (locked before any number existed)

Locked to the session scratchpad as `PREREG_datachannel_dryrun.md` **before** a single `p_struct` value, Brier or comparison was computed. It declares the window, the inputs, the full reaction function with every constant, the three criteria and the verdict rule. Reproduced in substance:

> **Window.** Daily t over **2026-06-03 … 2026-07-29** (57 days). Outcome **YES = 1**, verified on Gamma (UMA `resolved`), settled as `sf-2026-003`.
>
> **Criteria.** (a) Accuracy on the intersection with the news-FV series: **a1** Brier(`p_struct`) ≤ Brier(news-FV), **a2** ≤ Brier(prior-only), **a3** vs base-rate-only *reported, no bar* — "if `p_struct` ≈ base-rate-only, the win came from the reference class, NOT from the objective data, and the note must say exactly that". (b) **Directional sanity:** on every mapped release day `p_struct` must not move against the release's implication (0.5pp dead-band); PASS = zero violations. (c) **Stability:** measure, don't enforce, the largest single-day |Δ logit| against the live model's ±1.5-logit slow-market shift cap; a jump > 1.5 is a FLAG, not an automatic fail.
>
> **Verdict rule.** GO iff a1 AND a2 AND b pass AND c raises no flag. NO-GO otherwise, and the channel stays unbuilt.
>
> **Anti-post-hoc.** No constant may be changed after seeing any output; a discovered mis-specification is reported **as specified** and the fix is proposed for a *next* pass. n = 1 — no CI, no calibration claim.

The pre-registration also recorded what had been done *before* the lock — source-availability probes only (does the Cleveland JSON carry daily vintages, does ALFRED honour realtime windows, what does the MPT file contain) — so that nobody has to wonder whether the design was tuned against a peek at the answer.

---

## 2 · Inputs, and the proof they are point-in-time (DC-7)

Every input at date *t* is what a person could have downloaded on *t*:

| Object | Source | Point-in-time mechanism |
|---|---|---|
| Target range midpoint | FRED `DFEDTARU` / `DFEDTARL` | never revised; ALFRED anyway |
| Core PCE y/y, π(t) | FRED `PCEPILFE` **+** Cleveland Fed `nowcast_month.json` (**Core PCE Inflation**) | published index levels as of *t* via ALFRED, chained forward through the unpublished months with the nowcast **vintage dated ≤ t** |
| Unemployment, u(t) | FRED `UNRATE` | ALFRED realtime = *t* (a later revision is invisible) |
| SEP medians | FRED `FEDTARMD`, `JCXFEMD`, `UNRATEMD` | ALFRED realtime = *t* |
| Meeting calendar | federalreserve.gov 2026 FOMC calendar | published years in advance (Jan 27-28, Mar 17-18\*, Apr 28-29, Jun 16-17\*, **Jul 28-29**, Sep 15-16\*, Oct 27-28, Dec 8-9\*; \* = SEP meeting) |
| **Never an input** (DC-6) | Atlanta Fed MPT, dealer surveys, the Polymarket mid, the news-FV | fetched separately, display only |

**The vintage mechanism is verified, not assumed.** `FEDTARMD` for 2026 reads **3.4** as known on 2026-06-10 and **3.8** as known on 2026-06-18 — the June SEP is correctly invisible before it was published. The same query pattern hides the annual revision that lifted the May core-PCE index from 130.082 to 130.094 at the end of July.

**Gotcha worth logging: FRED's ingestion date is not the release date.** The Cleveland Fed flagged the June core-PCE **Actual** on **2026-07-27**, but `PCEPILFE` did not carry June in ALFRED until **2026-07-30** — one day *after* the FOMC decision. A pipeline keyed to FRED therefore still had June as a nowcast on decision day; a pipeline keyed to the Cleveland "Actual" flag would have had the print. Neither is wrong, but they are not the same clock, and a live channel must pick one and say which.

---

## 3 · The declared reaction function (DC-3), and a worked day

Every constant below was fixed in the pre-registration. None is fitted; the Taylor coefficients are **borrowed** from the macro literature and labelled as such ([[CODEX]] realism rule 2).

```
Step 1 — desired rate, as a Taylor-style adjustment of the Fed's OWN projected path
    r_desired(t) = FEDTARMD(t) + 0.5·(π(t) − JCXFEMD(t)) − 0.5·(u(t) − UNRATEMD(t))
Step 2 — per-meeting pressure, in 25bp "clicks"
    gap(t) = r_desired(t) − target_midpoint(t)
    m(t)   = (gap(t) / meetings_remaining_in_2026(t)) / 0.25
Step 3 — map to P(no change) at THIS meeting
    logit(P_hold) = b0 + b1·|m(t)|,   b1 = −3.0 (declared),
    b0 = logit(base rate of no-change FOMC decisions, 1994–2025, ZLB stretches excluded)
DC-2 — integrate the unpublished print, don't point-estimate it
    π ~ Normal(center = π(t), σ = 0.063pp·√k), k = unpublished months chained
    p_struct(t) = E_π[P_hold(π)],  20,000 antithetic draws, seed 0
```

**The base rate came out lower than intuition expects: 0.5272** — 87 target-change dates across 184 meetings in the 23 non-ZLB years. That is a direct consequence of DC-3's own instruction to exclude ZLB periods: those are exactly the years when the Fed held at every meeting, so removing them removes most of the "holds". It is reported here as computed, and its consequences are dissected in § 6.

**Worked day — 2026-07-29, decision day.** Published core PCE ran to May (index 130.082); the Cleveland vintages for June (+0.187% m/m) and July (+0.270% m/m) chained it forward to a y/y of **3.359%** against the June SEP's core-PCE median of **3.3%**. Unemployment stood at **4.2%** against the SEP's **4.3%**. So `r_desired = 3.8 + 0.5·(3.359 − 3.3) − 0.5·(4.2 − 4.3) = 3.880%` against a midpoint of **3.625%** — a gap of **+0.255pp**, spread over the 4 meetings left in 2026 = 0.064pp per meeting = **0.255 clicks**. Then `logit(P) = 0.1088 − 3.0×0.255 = −0.656` → **p_struct = 0.343**. The market resolved **YES** that evening.

---

## 4 · What the trajectory looks like

![Data-channel dry run — July 2026 Fed market](../../data/analysis/plots/news_agent/newsagent_datachannel_dryrun.png)

**Chart read.** *Top panel:* terracotta is `p_struct`; the dotted grey line is its own base-rate-only benchmark (52.7%); blue is the news-FV the Observatory actually published (flat — see below); the dashed dark line is the Polymarket mid (context); the two green lines are the Atlanta Fed market-implied lower bounds, **display only** (§ 7). The market resolved at the top of the chart (YES = 100%). *Bottom panel:* the mechanical drivers — the desired-rate `gap` in pp and the per-meeting pressure `m` in clicks. Vertical guides mark the June SEP, the May core-PCE print, the June jobs report and the decision.

Unit of observation: one day. `p_struct` is the pre-registered method's probability of **no change at the July 2026 meeting**; all series are on the same axis.

| Date | p_struct | π core PCE y/y | u | SEP median | midpoint | gap (pp) | meetings left | m (clicks) |
|---|---|---|---|---|---|---|---|---|
| 2026-06-03 | 0.440 | 3.34 | 4.3 | 3.4 | 3.625 | +0.146 | 5 | +0.117 |
| 2026-06-17 (June SEP) | 0.422 | 3.30 | 4.3 | **3.8** | 3.625 | +0.177 | 5 | +0.142 |
| 2026-06-25 (May core PCE) | 0.393 | 3.31 | 4.3 | 3.8 | 3.625 | +0.182 | 4 | +0.182 |
| 2026-07-08 | 0.307 | 3.47 | 4.2 | 3.8 | 3.625 | +0.308 | 4 | +0.308 |
| 2026-07-29 (decision) | 0.343 | 3.36 | 4.2 | 3.8 | 3.625 | +0.255 | 4 | +0.255 |

**Read.** The method behaves exactly as designed and moves in the direction the design intends: the June SEP's 40bp upward shift, a firm inflation print and a tighter labour market all raise the desired-rate gap, and a bigger gap lowers the probability of holding. Nothing is broken. It is simply pointed at the wrong object — the pressure it measures is pressure over the *rest of the year*, and it charges all of it to the very next meeting.

**Why the news-FV line is flat.** The Observatory's published number for this market never moved off its onboarding prior of 53.7% across the whole reconstructed window: no day cleared the slow-market decisive-evidence threshold. That is the "structurally blind" tag being literally true, and it means **a2 (vs prior-only) and a1 (vs news-FV) are the same test on this market**.

---

## 5 · Verdict per criterion

Intersection = the 15 days (2026-06-21 … 2026-07-05) where a published news-FV exists. Outcome YES = 1, so lower Brier is better and a number *below* 0.25 beats a coin flip.

| Criterion | Value | Comparator | Result |
|---|---|---|---|
| **a1** Brier(p_struct) vs news-FV | **0.4223** | 0.2144 | **FAIL** |
| **a2** Brier(p_struct) vs prior-only | **0.4223** | 0.2144 | **FAIL** |
| **a3** Brier(p_struct) vs base-rate-only *(no bar)* | 0.4223 | 0.2236 | worse than its own base rate |
| — *context only:* Polymarket mid | — | 0.0399 | (not a bar; the beat-the-mid claim is closed) |
| — p_struct over its own 57-day window | 0.3974 | — | reported, never substituted |
| **b** Directional violations | **0** of 2 directional release days | 0 | **PASS** |
| **c** Max single-day \|Δ logit\| | **0.174** (2026-06-25) | 1.5 reference | **no flag** (end-to-end 0.412) |

**Directional detail (b).** June SEP on 06-17: median 3.4 → 3.8 (upward) → Δp_struct **−0.029** (correct: more pressure, less likely to hold). May core PCE on 06-25: π rose → Δp_struct **−0.041** (correct). The June jobs report on 07-02 moved p_struct **−0.033**; unemployment carried no pre-registered direction, so it is reported and not scored.

**Verdict rule as locked: GO requires a1 AND a2 AND b AND no c flag. a1 and a2 fail → NO-GO.**

---

## 6 · Why it failed — three mechanisms, separated

The failure is not "the data is useless". It is three specific design errors, in decreasing order of importance.

**1 · The per-meeting allocation is wrong (the big one).** The rule converts a desired-rate gap into per-meeting pressure by dividing evenly by the meetings remaining, then treats that quotient as if it were the chance *this* meeting moves. But a +0.26pp gap spread over 4 meetings is a market that expects **one hike sometime this year**, not a quarter of a hike at each meeting — and the FOMC does not deliver quarter-clicks. The correct object is a *distribution over which meeting* moves, informed by the calendar (SEP meetings carry more moves than non-SEP ones; July 2026 was **not** an SEP meeting). The declared rule has no such structure, so it charges the nearest meeting with pressure that belongs to September or December. The market-implied data in § 7 shows exactly that misallocation, from the outside.

**2 · The base rate is the base rate of the wrong reference class.** DC-3 instructs "excluding ZLB periods", which removes 2009–2015 and 2020–2021 — precisely the stretches where the Fed held at every meeting. The surviving sample is nothing but active hiking and cutting cycles, giving a "no change" rate of **0.5272** where the honest all-years figure is **0.6484** (90 change dates / 256 meetings). A per-meeting hold probability of 53% is not a defensible prior for an FOMC meeting in any regime.

**3 · The macro adjustment carried negative information here.** A pre-registered sensitivity sweep on the declared slope: the Brier improves **monotonically** as the macro term is switched off.

| b1 | mean p_struct | Brier (intersection) |
|---|---|---|
| −3.0 *(declared)* | 0.350 | 0.4229 |
| −2.0 | 0.407 | 0.3521 |
| −1.0 | 0.467 | 0.2847 |
| −0.5 | 0.497 | 0.2532 |
| **0.0** *(base rate only)* | **0.527** | **0.2236** |

**Post-hoc diagnostics, labelled as such and NOT part of the verdict.** Swapping only `b0` for the all-years base rate (0.6484) lifts the mean probability to 0.471 and the Brier to **0.2811** — better, still worse than prior-only's 0.2144. So the NO-GO does **not** hinge on the ZLB choice: fixing the base rate alone does not rescue the method. Mechanism 1 is the binding defect. These diagnostics change nothing about the verdict; they exist to tell Justin whether the design is salvageable, and the answer is "only with a structural change, not a constant tweak".

---

## 7 · What the market-implied benchmark showed (DISPLAY ONLY — DC-6)

The Atlanta Fed **Market Probability Tracker** never entered `p_struct` and never will under § 4f. It is reported here because it *diagnoses* the failure from the outside.

**The caveat, stated wherever this is plotted:** MPT probabilities are computed over **3-month SOFR windows**, not over single meetings. `1 − P(cut) − P(hike)` for the window containing a meeting is therefore a **lower bound** on P(no change at that meeting alone) — any change anywhere in the window counts against it. `Prob: cut` / `Prob: hike` are in **percent**, not fractions.

| Window | What it covers | Over 2026-06-03 … 07-29 |
|---|---|---|
| ref **2026-06-17** (Jun 17 → Sep 16) — *contains the July meeting* | July decision | p_hold lower bound **0.82 → 0.89**, quoted only until **2026-06-11** |
| ref **2026-09-16** (Sep 16 → Dec 16) — *the next window* | Sept/Oct/Dec decisions | p_hold lower bound **0.50 → 0.23**, collapsing at the June SEP (0.328 on 06-17 → 0.227 on 06-18), bottoming at **0.144** in late July |

**Read — this is the whole diagnosis in two lines.** The same June SEP that our rule read as "less likely to hold in July" was read by the rates market as "**a hike is coming, in the September-to-December window**": the July-containing window stayed at 82–89% hold while the next window fell to 14–23%. Our method detected the pressure correctly and then allocated it to the wrong meeting. That is a fixable structural error, and it is the specific thing an amended rule must get right.

**Gotcha logged: the MPT stops quoting a window once that window opens.** The Jun-17 window was quoted for only 9 dates in June and vanishes from 06-12 onward. So for any given meeting the MPT is available **only before its window starts** — for the live September market that means the Sep-16 window is quotable now and will go dark from roughly early September. A display panel that assumes a continuous MPT series will show a hole exactly when the meeting gets interesting. The scoping note's § 4c description ("windows land on FOMC/IMM dates incl. 2026-09-16") is correct but does not say this; it does now, by reference to this note.

---

## 8 · Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions.** The Taylor coefficients (±0.5) are **borrowed**, not derived here. b1 = −3.0 is a declared judgement made from stated reasoning before any output. σ = 0.063pp is the § 3 core-PCE surprise RMSE measured on 12 releases, treated as the one-month y/y uncertainty and scaled √k for k chained months. The SEP's Q4/Q4 projections are compared against a 12-month y/y — an approximation. The 2026 FOMC calendar is treated as vintage-safe (published years in advance). Intermeeting moves are counted in the base-rate numerator, a small declared overcount. The base rate uses 8 scheduled meetings per year throughout.

**Live-only unknowns.** Whether an amended per-meeting allocation would beat prior-only on *other* meetings — this is n=1 and the September market is the next test bed, prospectively rather than retrospectively. Whether the FRED-vs-Cleveland ingestion-clock gap matters on a meeting where the print lands *inside* the window (here it did not: June core PCE hit FRED the day after the decision). Whether the double-count guard (§ 4d of the scoping note) fires cleanly on real packets — never exercised, since nothing was built.

**Power honesty.** **n = 1 market, one meeting, 57 days.** Every number here is a point read on a single question; no confidence interval is computed and none should be quoted. A NO-GO on one market is enough to *stop a build* (that is what a cheap falsifier is for) but would never be enough to declare a method calibrated had it passed — the pre-registration said so before the run.

---

## 9 · Decision and next step

**Decision: NO-GO. The data-evidence channel is not built, and OpenBB is not installed.** The declared DC-3 reaction function, fed genuinely point-in-time inputs, produced a probability that moved *away* from the outcome and lost to the number it was meant to replace, to that number's own prior, and to its own base rate. Row 1 of the sign-off authorises the install "when the build starts"; the build does not start.

What survives the test, and is worth keeping: the **input plumbing is sound and provably lookahead-free** (ALFRED realtime + daily Cleveland vintages + the never-revised target series), the **directional behaviour is correct**, the **re-anchoring is gentle** (0.17 logits max daily, against the 1.5 cap), and the **failure has a named mechanism** rather than a shrug.

**Proposed amendment for a future pass — to be locked in [[newsagent_data_channel_scoping]] § 4f BEFORE any re-run, per the amendment discipline:**

1. **DC-3a — allocate across meetings instead of dividing by them.** Model `P(move at meeting i)` from the remaining-gap in clicks with a declared allocation over the remaining calendar (e.g. hazard weights that favour SEP meetings), then `p_struct = 1 − P(move at THIS meeting)`. The object to predict is a single meeting; the rule must say which meeting absorbs the pressure.
2. **DC-3b — fix the reference class.** Use the all-years per-meeting hold rate (0.6484 here) or, better, a regime-conditioned rate; the ZLB exclusion as written removes the wrong years for a *per-meeting* base rate.
3. **Re-test on a different resolved meeting first** (an SEP meeting and a non-SEP meeting), still retrospectively, still pre-registered — n=1 was enough to kill, it is not enough to bless.
4. **Only then** consider the live September market, and only as a **display-only** second number until it has settled forecasts of its own (DC-8 labelling is already shipped and waiting: `ledger.method_for()` returns `news+data` the moment a market joins `config.DATA_CHANNEL_MARKETS`, which stays empty).

**What Justin does not need to do:** nothing is waiting on him. No key, no install, no permission. The next move is a research decision about whether the amended rule is worth one more retrospective pass, and the two California markets remain unmappable at any tier regardless.

## 10 · Outputs

- **Script:** `scripts/newsagent_datachannel_dryrun.py` — reproduces every number here (`--chart` for the figure). Uses only the FRED key from the git-ignored `.env`; the keyless `fredgraph.csv` fallback stays available if the key is ever absent.
- **Pre-registration:** `PREREG_datachannel_dryrun.md`, locked to the session scratchpad before the first computation, then copied beside its predecessor at `data/newsagent/datachannel/` (git-ignored but session-durable, the same place `PREREG_consensus_accuracy.md` lives); reproduced in substance in § 1.
- **CSVs** (git-ignored): `data/analysis/csv_outputs/news_agent/newsagent_datachannel_dryrun_trajectory.csv` (57 rows: p_struct, its point estimate, news-FV, mid, both MPT lower bounds, base-rate-only, and every input) and `…_criteria.csv` (the scored bars).
- **Plot:** `data/analysis/plots/news_agent/newsagent_datachannel_dryrun.png`.
- **Cache** (git-ignored): `data/newsagent/datachannel/` — ALFRED vintage pulls, the Cleveland nowcast JSON, `mpt_histdata.xlsx`, and `dryrun_pstruct.json` (the full run record incl. scores).
- **Not touched:** `fvmodel.py`, `config.py`, `dashboard.py`, `fv_params.json`, the sf ledger, every published page, and the research venv.
