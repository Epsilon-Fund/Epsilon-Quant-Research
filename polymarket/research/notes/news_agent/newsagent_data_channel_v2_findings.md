---
title: "Data-channel v2 — the amended reaction function is far more accurate and still fails, because the pre-registered directional criterion was written for a hiking regime (NO-GO)"
created: 2026-08-24
status: complete — NO-GO under the locked verdict rule. Accuracy criteria A and B both PASS (pooled Brier 0.1273 vs base-rate-only 0.2349; July 0.1673 vs the 0.2144 bar and vs v1's 0.4223), directional criterion C FAILS with 6 violations — all six on cutting-pressure meetings, none on hiking ones. Nothing built, OpenBB NOT installed, config/model/ledger/dashboard untouched.
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
# Data-channel v2 — fixing the reaction function, and being caught by our own criterion

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]] · [[CODEX]] · Parent: [[newsagent_data_channel_scoping]] § 4f **Amendment 1** (locked 2026-08-24, before any number here existed) · Predecessor: [[newsagent_data_channel_dryrun_findings]] (the v1 NO-GO) · Table terms: [[polymarket_table_dictionary]]
> **Nothing was built.** `fvmodel.py`, `config.py`, `dashboard.py`, `fv_params.json`, the sf ledger and every published page are untouched; OpenBB is still **not** installed.

## Plain-English Summary

- **What this is.** The v1 dry run killed a designed second evidence channel for the Fed markets: build a probability from official statistics instead of news. It named two specific defects. This pass locked amendments for both into the standing pre-registration **before computing anything**, then re-ran on **eight** resolved FOMC decisions instead of one.
- **The amendments worked, and the improvement is large.** Pooled over 8 meetings the amended rule scores **Brier 0.1273** against base-rate-only's **0.2349** — the bar v1 failed by the widest margin, now cleared comfortably. On the July-2026 market that killed v1, scored on the identical 15-day window, it goes from **0.4223 → 0.1673**, beating the 0.2144 bar that is both the news-FV and its own prior.
- **And it still fails, on the third criterion.** Directional sanity — which v1 *passed* with zero violations — now shows **6 violations**, so under the locked verdict rule (GO iff A **and** B **and** C) the answer is **NO-GO** and nothing is built.
- **The failure is in our criterion, not in the rule — which does not rescue it.** All six violations land on meetings where the Fed was under **cutting** pressure; **zero** land on the three hiking-pressure meetings. Criterion C's sign map ("SEP median up ⇒ holding less likely") is only correct when the desired rate is *above* the target. Under cutting pressure the sign flips, and the criterion flags correct behaviour as a violation. v1 passed C only because it was tested on a single hiking market. **The criterion was mis-specified and I carried it forward without re-deriving it for a cutting regime — that is a pre-registration defect, and the verdict stands anyway, because the whole point of locking a rule is that you do not rewrite it after seeing the number.**
- **The uncomfortable finding underneath the win.** Decomposing the two amendments: **DC-3b alone — just fixing the base rate's reference class — scores 0.1200, better than the two amendments combined (0.1273).** The DC-3a allocation machinery, which the v1 note called the binding defect, contributes essentially nothing, and its SEP hazard weighting is mildly *negative* (W_SEP = 1.0 scores 0.1245 against the declared 2.0's 0.1273). v1's diagnosis of which defect was binding was wrong.
- **One-line status:** amendments locked before the run, run as specified on 8 meetings, **NO-GO** recorded, nothing built, and the v3 proposal is a two-line criterion fix plus deleting most of DC-3a.

## What this pass is and is not

| | |
|---|---|
| **Is** | Eight retrospective daily trajectories (45 days each) on eight resolved FOMC decisions, computed from the § 4f **Amendment 1** rule with strictly point-in-time inputs, scored against criteria locked before the first number existed. |
| **Is not** | A calibration claim, a fit, or a build. n = 8 meetings — **no confidence interval is computed and none is quoted**. No constant was changed after seeing an output; the sensitivity sweeps below are *reported*, never selected from. |
| **New files** | `scripts/newsagent_datachannel_v2.py`, three CSVs, one plot, and this note. |
| **Untouched** | `newsagent/fvmodel.py`, `config.py`, `dashboard.py`, `fv_params.json`, the sf ledger, every published page, and the research venv (**no OpenBB install** — § 7 row 1 authorises it only "when the build starts", and the build does not start). |

---

## 1 · What was locked, and when

[[newsagent_data_channel_scoping]] § 4f now carries **Amendment 1**, written with every constant fixed before a single v2 number existed. In brief — the full text with rationale lives there, not here:

- **DC-3a — allocate the gap across the remaining calendar; do not divide by it.** v1 took the desired-rate gap, divided it by the number of meetings left, and charged the quotient to the next meeting. v2 treats the gap as a *total magnitude* `G` in 25bp clicks, allocates it over the remaining meetings of the SEP projection year with declared hazard weights (`W_SEP = 2.0` for a projection meeting, `W_NON = 1.0` otherwise), and maps only this meeting's share to a probability: `logit(P_move) = logit(h_base) + B1A·μ₁`, `B1A = 1.5`.
- **DC-3b — one declared reference class: the all-years per-meeting hold rate.** v1's "excluding ZLB periods" removed 2009–2015 and 2020–2021 — exactly the stretches where the Fed held at every meeting — and produced a 52.7% per-meeting hold rate. v2 uses **all** scheduled meetings 1994–2025: **0.6484 hold** (90 change dates / 256 meetings), recomputed in code rather than hardcoded. No regime conditioning; that alternative is declared and rejected in the amendment so it cannot be reached for later.
- **Test set, criteria and verdict rule** were fixed in the same edit: eight meetings (4 SEP / 4 non-SEP, 3 moves / 5 holds), criteria A/B/C, **GO iff all three**.
- **A disclosure is recorded in the amendment** and is repeated here because it would otherwise be invisible: while deriving DC-3a the July-2026 arithmetic was worked through by hand to check the mechanics, which made it visible that the amended rule beats the old one on that one meeting *before* the constants were locked. No other constant value was evaluated, and the constants were chosen from stated reasoning. **Criterion B is knowingly compromised by that peek**; the weight of this test rests on **criterion A** and on the seven meetings whose numbers did not exist in any form when the lock was written.

## 2 · Inputs — unchanged, and still provably point-in-time

Every input at date *t* is what a person could have downloaded on *t*, using the plumbing the v1 pass proved lookahead-free ([[newsagent_data_channel_dryrun_findings]] § 2): ALFRED realtime vintages for the revised series, daily Cleveland Fed nowcast vintages for the unpublished months, and the never-revised `DFEDTARU`/`DFEDTARL` for the target range. The v2 script imports those functions rather than reimplementing them.

**Why scoring on the decision day is safe, verified rather than assumed.** `DFEDTARU` records a change at its **effective** date, which is the day *after* the decision. So a vintage read as of the decision day cannot see the outcome. Spot-checked against the FRED change dates: the 2025-09-17 decision appears as an 2025-09-18 change, 2025-10-29 as 2025-10-30, 2025-12-10 as 2025-12-11.

**Carried forward from v1, still true:** FRED's ingestion clock is not the release clock (June-2026 core PCE was flagged Actual by Cleveland on 07-27 but did not reach ALFRED until 07-30, one day *after* the July decision). A live channel must pick one clock and say which. This pass uses FRED/ALFRED for published prints and Cleveland vintages for the unpublished tail, which is the conservative combination.

## 3 · The test set, and a worked meeting

Eight resolved decisions, fixed before any computation. **Outcome is coded in the market's own framing: 1 = no change, 0 = change**, so `p_struct` is directly a probability of the YES side.

**Worked meeting — 2026-07-29, the one that killed v1.** Inputs are the ones already recorded in the v1 note: `r_desired = 3.880%` against a midpoint of `3.625%`, so `gap = +0.255pp` and `G = 1.02` clicks. The remaining 2026 meetings from that date are Jul 29 (non-SEP, w = 1), Sep 16 (SEP, w = 2), Oct 28 (non-SEP, w = 1), Dec 9 (SEP, w = 2), so `Σw = 6` and `share₁ = 1/6 = 0.167`. Therefore `μ₁ = 1.02 × 0.167 = 0.170`, and `logit(P_move) = −0.6118 + 1.5 × 0.170 = −0.357` → `P_move = 0.412` → **`p_struct = 0.588`**. v1 produced **0.343** on identical inputs. The market resolved **YES (no change)**.

The whole difference is where the pressure goes. v1 charged a quarter of the year's expected hike to July; v2 gives July one-sixth of it and puts the rest on September and December — which is where the rates market put it (v1 note § 7, display-only).

## 4 · Per-meeting results

*Unit of observation: one FOMC decision.* **G** is total remaining pressure in 25bp clicks (`|r_desired − midpoint| / 0.25`). **share₁** is the fraction of that pressure the hazard weights allocate to *this* meeting. **μ₁ = G × share₁** is the expected clicks at this meeting, the only quantity that enters the probability. **p_struct** is P(no change at this meeting) on the decision day. **Brier** = (p_struct − outcome)²; lower is better; base-rate-only always says 0.6484. **v1 p** is the original DC-3 rule re-scored on identical inputs, reported so the amendment's contribution is separable. **viol** is directional-sanity violations over that meeting's 45-day trajectory.

| Meeting | SEP | Outcome | G (clicks) | share₁ | μ₁ | **p_struct** | **Brier** | v1 p | v1 Brier | base Brier | viol |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-09-17 | ✓ | **MOVE** (cut) | 2.92 | 0.400 | 1.170 | 0.242 | **0.0588** | 0.057 | 0.0033 | 0.4204 | 1 |
| 2025-10-29 | — | **MOVE** (cut) | 2.33 | 0.333 | 0.777 | 0.365 | **0.1334** | 0.034 | 0.0011 | 0.4204 | 2 |
| 2025-12-10 | ✓ | **MOVE** (cut) | 1.03 | 1.000 | 1.030 | 0.287 | **0.0822** | 0.058 | 0.0033 | 0.4204 | 1 |
| 2026-01-28 | — | hold | 0.39 | 0.083 | 0.032 | 0.637 | **0.1316** | 0.491 | 0.2594 | 0.1236 | 1 |
| 2026-03-18 | ✓ | hold | 0.35 | 0.182 | 0.064 | 0.626 | **0.1400** | 0.489 | 0.2611 | 0.1236 | 1 |
| 2026-04-29 | — | hold | 0.26 | 0.111 | 0.029 | 0.638 | **0.1311** | 0.493 | 0.2572 | 0.1236 | 0 |
| 2026-06-17 | ✓ | hold | 0.71 | 0.250 | 0.177 | 0.586 | **0.1716** | 0.422 | 0.3343 | 0.1236 | 0 |
| 2026-07-29 | — | hold | 1.02 | 0.167 | 0.170 | 0.588 | **0.1694** | 0.343 | 0.4323 | 0.1236 | 0 |
| **Pooled** | 4/8 | 3 moves | — | — | — | mean 0.496 | **0.1273** | mean 0.298 | 0.1940 | **0.2349** | **6** |

**Read.** The amended rule is doing the thing it was designed to do. On the three meetings that moved it sits at 24–37%, well below the base rate, and on the five holds it sits at 59–64%, near it. v1's apparent brilliance on the moves (Brier 0.001–0.003) is not skill: it says "move" almost everywhere — mean probability of holding **0.298** across a sample that held 5 times out of 8 — so it nails the three cuts and pays for it heavily on every hold. v2's mean is 0.496 against a realised hold rate of 0.625, i.e. still under-forecasting holds, but not grotesquely.

![Data-channel v2 — eight resolved FOMC decisions](../../data/analysis/plots/news_agent/newsagent_datachannel_v2.png)

**Chart read.** Eight panels, one per decision, all on the same 0–1 axis. Terracotta is `p_struct` (v2); the dotted grey line is the v1 rule on identical inputs; the dashed line is base-rate-only at 0.6484; the star at the right edge is the realised outcome (1 = no change, at the top; 0 = a move, at the bottom). The top row is the 2025 cutting sequence — both rules sit below the base rate, correctly, and v2 is the more moderate of the two. The bottom row is the 2026 hold sequence — v2 tracks just under the base rate while v1 sits 15–25 points lower and is wrong by that much on every panel. The gap between the terracotta and dotted lines *is* the amendment.

## 5 · Verdict per criterion

| Criterion | Value | Comparator | Result |
|---|---|---|---|
| **A** Pooled mean Brier over 8 meetings | **0.1273** | base-rate-only **0.2349** | **PASS** |
| **B** July-2026 market, dry run's own 15-day window (2026-06-21 … 07-05) | **0.1673** | bar **0.2144** (news-FV = prior-only) | **PASS** *(peek-compromised — see § 1)* |
| **C** Directional violations, 0.5pp dead-band | **6** | 0 | **FAIL** |
| — *reported, not a bar:* v1 rule pooled on the same 8 meetings | 0.1940 | — | |
| — *reported, not a bar:* v1 rule on the July window | 0.4223 | — | reproduces the v1 note exactly |

**Verdict rule as locked: GO requires A AND B AND C. C fails → NO-GO.**

### 5a · Why C failed, and why that does not rescue the pass

Every violation, with the sign of the desired-rate gap on that meeting:

| Meeting | gap at decision | Regime | Violations |
|---|---|---|---|
| 2025-09-17 | **−0.731pp** | cutting pressure | 1 |
| 2025-10-29 | **−0.583pp** | cutting pressure | 2 |
| 2025-12-10 | **−0.258pp** | cutting pressure | 1 |
| 2026-01-28 | **−0.097pp** | cutting pressure | 1 |
| 2026-03-18 | **−0.088pp** | cutting pressure | 1 |
| 2026-04-29 | +0.066pp | hiking pressure | **0** |
| 2026-06-17 | +0.177pp | hiking pressure | **0** |
| 2026-07-29 | +0.255pp | hiking pressure | **0** |

**Six of six violations on negative-gap meetings; zero of three on positive-gap meetings.** That is not a coincidence and it is not the model misbehaving.

Criterion C says: *an SEP median moving up, or inflation rising, must not move `p_struct` up.* That is correct only when the desired rate sits **above** the target — the Fed is under pressure to hike, and anything hawkish makes holding less likely. When the desired rate sits **below** the target, the Fed is under pressure to cut, and the *same* hawkish news **shrinks** `|gap|`, reduces the pressure to act, and correctly makes holding **more** likely. The rule tracks the magnitude of pressure; the criterion tracks the direction of the news. Those agree in a hiking regime and disagree in a cutting one.

Worked violation: on **2025-09-17** the SEP median moved **down** (more cuts projected). Criterion C reads "median down ⇒ holding more likely" and expects `p_struct` to rise. The rule reads "more cuts projected, further below the target ⇒ more pressure to act now" and drops `p_struct` by 0.154. The rule is right. The criterion is written for the other half of the cycle.

**v1 passed C on two directional days in a single hiking-pressure market.** That is the entire evidential basis for the criterion having ever looked sound, and this pass is the first time it met a cutting cycle.

**Why the NO-GO stands anyway.** The criterion was locked before the run. Discovering after the fact that a locked criterion was mis-specified and then declaring victory under a rewritten one is precisely the post-hoc move the whole pre-registration discipline exists to prevent — it is the same class of error as the v0 gate's Amendment discipline, and [[CODEX]] § Realism calibration cuts both ways. The mis-specification is mine: I carried the dry run's directional map into Amendment 1 without re-deriving it for a sample that, by my own test-set design, deliberately included a cutting cycle. **The fix is a two-line change and it belongs in a v3 lock, not in this verdict.**

## 6 · The uncomfortable decomposition — v1 named the wrong defect

Two amendments shipped together, so the honest question is what each one bought. Pooled Brier at decision day over the same 8 meetings, everything else held identical:

| Variant | Base rate | Slope | Allocation | Pooled Brier | mean p |
|---|---|---|---|---|---|
| v1 as run in the dry run | ZLB-excluded 0.5272 | 3.0 | none (divide by N) | 0.1940 | 0.298 |
| **+ DC-3b only** | **all-years 0.6484** | 3.0 | none (divide by N) | **0.1200** | 0.386 |
| + DC-3a only | ZLB-excluded 0.5272 | 1.5 | allocate, W_SEP 2.0 | 0.1784 | 0.384 |
| **DECLARED v2** (both) | all-years 0.6484 | 1.5 | allocate, W_SEP 2.0 | **0.1273** | 0.496 |
| v2 with no SEP preference | all-years 0.6484 | 1.5 | allocate, W_SEP 1.0 | 0.1245 | 0.485 |

**Read, and it is a correction to the previous note.** [[newsagent_data_channel_dryrun_findings]] § 6 ranked the defects as "1 · per-meeting allocation (the big one)" and "2 · the base rate", and its post-hoc diagnostic concluded that fixing the base rate alone "does not rescue the method". On this eight-meeting sample that ranking is **backwards**: fixing only the reference class takes 0.1940 → **0.1200**, which is better than both amendments together, while fixing only the allocation takes it to 0.1784. The v1 diagnostic reached the opposite conclusion because it was computed on one market over a 15-day window; pooled across a sample containing three actual moves, the base rate dominates.

**And the SEP hazard weighting — the headline idea of DC-3a — is mildly counterproductive.** W_SEP = 1.0 (no preference at all) scores 0.1245 against the declared 2.0's 0.1273, and it degrades monotonically as the preference strengthens. The concentration-at-projection-meetings intuition is not visible in this sample.

### 6a · Sensitivity — reported, never selected from

The verdict is on the declared `B1A = 1.5` / `W_SEP = 2.0` and on nothing else. These are here so a knife-edge dependence would be visible:

| B1A | Pooled Brier | mean p | | W_SEP | Pooled Brier | mean p |
|---|---|---|---|---|---|---|
| 0.5 | 0.1872 | 0.597 | | 1.0 | 0.1245 | 0.485 |
| 1.0 | 0.1501 | 0.544 | | 1.5 | 0.1252 | 0.491 |
| **1.5** *(declared)* | **0.1273** | 0.496 | | **2.0** *(declared)* | **0.1273** | 0.496 |
| 2.0 | 0.1165 | 0.456 | | 3.0 | 0.1317 | 0.504 |
| 3.0 | 0.1158 | 0.400 | | | | |

**Read.** The B1A surface improves monotonically to about 2.0 and then flattens — the declared 1.5 is **not** at the optimum, which is the reassuring direction for a constant that was supposed to be chosen from reasoning rather than from an outcome. The W_SEP surface is essentially flat and slightly favours turning the idea off. Neither sweep is a licence to move a constant; both are recorded so the next lock can be argued from evidence.

## 7 · Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions.** The Taylor coefficients (±0.5) remain **borrowed** from the macro literature, not derived here — labelled as such since v1. `B1A = 1.5` and `W_SEP = 2.0` are declared judgments with stated reasoning; `σ = 0.063pp·√k` is the v1 core-PCE surprise RMSE, unchanged. The SEP's Q4/Q4 projections are compared against a 12-month y/y, an approximation. Intermeeting moves are counted in the base-rate numerator, a small declared overcount, and the base rate assumes 8 scheduled meetings per year throughout. The horizon is the SEP projection year, so a December meeting has N = 1 and receives the entire remaining gap — defensible but never stress-tested beyond the single 2025-12-10 case in this sample.

**Live-only unknowns.** Whether the amended rule holds up on meetings *outside* 2025–2026 — this sample is one cutting sequence followed by one holding sequence, and contains **no hikes at all**, so the hiking half of the cycle is tested only through the sign of the gap and never through a realised hike. Whether the criterion-C fix proposed below actually passes when it is locked and run. Whether the FRED-vs-Cleveland ingestion-clock gap matters on a meeting where a print lands *inside* the window. Whether the § 4d double-count guard fires cleanly on real packets — still never exercised, because nothing has been built.

**Power honesty.** **n = 8 meetings, 3 of them moves, all within a 14-month window of a single easing-then-holding cycle.** Every number here is a point read; **no confidence interval is computed and none should be quoted**. Eight meetings is enough to kill a method and is not enough to bless one — the pre-registration said so before the run, and it says so now that the accuracy numbers came out well. The apparent quality of the pooled Brier also owes something to the sample's composition: 5 of 8 outcomes were holds and the base rate is 0.6484, so a method that simply leans toward holding scores decently.

## 8 · Decision and next step

**Decision: NO-GO. The data-evidence channel is not built, OpenBB is not installed, and no market joins `config.DATA_CHANNEL_MARKETS`.** Under the rule locked before the run, criterion C fails and that is the end of it.

**But this is a different NO-GO from the last one, and the difference matters.** v1 failed on accuracy: its probability moved *away* from the outcome and lost to doing nothing. v2 passes both accuracy bars, clearly and on a sample eight times larger, and fails only a well-behavedness check whose sign map turns out to have been written for one half of the interest-rate cycle. That is a **defect in the test**, and it is now diagnosed rather than suspected.

**Proposed v3 — three changes, to be locked in [[newsagent_data_channel_scoping]] § 4f BEFORE any re-run:**

1. **Fix criterion C's sign map.** The implied direction must be taken on the **pressure magnitude** `|gap|`, not on the raw direction of the news: a release that increases `|r_desired − midpoint|` must not raise `p_struct`, whichever side of the target the desired rate is on. Two lines in the checker, and it makes the criterion regime-neutral instead of silently hiking-only.
2. **Cut most of DC-3a.** On the evidence of § 6, keep the allocation *framing* (it is the conceptually right object — a single meeting, not a per-meeting quotient) but set **`W_SEP = 1.0`**, i.e. allocate evenly across the remaining calendar, and say plainly that the SEP-concentration hypothesis was tested and not supported. Do **not** re-tune `B1A` toward the 2.0 the sweep prefers; that would be exactly the fitting the pre-registration forbids. Declare it once, from reasoning, and leave it.
3. **Extend the sample backwards before believing anything.** This set contains no hikes. Add the 2022–2023 hiking sequence and a stretch of the 2015–2018 normalisation, both of which have ALFRED coverage, and re-run. If the amended rule survives a hiking cycle on a fixed, pre-registered criterion, that is the point at which a build is worth discussing — and only then as **display-only** under DC-8's per-method labelling, which is already shipped and waiting.

**What Justin needs to decide:** whether a v3 pass is worth the time. My recommendation is **yes, and it is cheap** — the plumbing, the test harness and the eight-meeting trajectory set all exist and re-run in a few minutes, the criterion fix is two lines, and the accuracy result is strong enough that leaving it at "NO-GO on a mis-specified check" would be the wrong place to stop. But it is a research-priority call, not a blocker: nothing is waiting on a key, an install or a permission, and the two California markets remain unmappable at any tier regardless.

## 9 · Outputs

- **Script:** `scripts/newsagent_datachannel_v2.py` — reproduces every number here (`--chart` for the figure). Imports the v1 vintage plumbing unchanged; uses only the FRED key from the git-ignored `.env`.
- **Amendment:** [[newsagent_data_channel_scoping]] § 4f **Amendment 1**, locked 2026-08-24 before any computation, incl. the test set, the criteria, the verdict rule and the peek disclosure.
- **CSVs** (git-ignored): `data/analysis/csv_outputs/news_agent/newsagent_datachannel_v2_meetings.csv` (8 rows, the scored table above), `…_trajectory.csv` (every daily row across all 8 windows), `…_sensitivity.csv`.
- **Plot:** `data/analysis/plots/news_agent/newsagent_datachannel_v2.png`.
- **Run record** (git-ignored): `data/newsagent/datachannel/v2_results.json` — criteria, verdict, constants, per-meeting results incl. every violation with its date and driver.
- **Not touched:** `fvmodel.py`, `config.py`, `dashboard.py`, `fv_params.json`, the sf ledger, every published page, and the research venv.
