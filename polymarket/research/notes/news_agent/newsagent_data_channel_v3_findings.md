---
title: "Data-channel v3 — the amended rule passes on 40 resolved FOMC meetings across three rate regimes (GO), and the channel is BUILT and live on the September Fed market"
created: 2026-08-24
status: complete — GO. Pooled Brier 0.1617 vs base-rate-only 0.2795 over 40 meetings; July market 0.1926 vs its 0.2144 bar; regime-neutral criterion C shows 0 violations; 0 meetings dropped by the leakage guard. BUILT the same day: OpenBB installed, offline ingest boundary, p_struct wired as the re-anchorable prior on the September Fed market (now in DATA_CHANNEL_MARKETS, method news+data, n=0 track), § 4d double-count guard live. 658 tests green. No alpha refit; no calibration claim.
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
# Data-channel v3 — the rule survives a hiking cycle, and the channel goes live

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]] · [[CODEX]] · Parent: [[newsagent_data_channel_scoping]] § 4f (**Amendment 2**, locked 2026-08-24 before any number here existed) · Predecessors: [[newsagent_data_channel_dryrun_findings]] (v1 NO-GO on accuracy), [[newsagent_data_channel_v2_findings]] (v2 NO-GO on a mis-specified criterion) · Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

- **What this is.** Third attempt at a second evidence channel for the Fed markets: build the probability from official statistics instead of news. v1 failed on accuracy. v2 passed both accuracy bars and failed a directional check whose sign map turned out to be written for one half of the interest-rate cycle. v3 fixes that check, drops the one hypothesis v2 tested and could not support, and re-runs on **40 resolved FOMC decisions spanning normalisation, hiking, cutting and holding**.
- **It passes, on all three criteria.** Pooled **Brier 0.1617 against base-rate-only's 0.2795** over 40 meetings (criterion A, the bar that carries the weight); the July-2026 market scores **0.1926** against its **0.2144** bar; the regime-neutral criterion C shows **zero violations** across all 40 trajectories. Zero meetings were dropped by the new leakage guard. **Verdict: GO.**
- **So the channel is built.** OpenBB installed (sign-off row 1, "at build start"), an offline ingest boundary that `newsagent/*` may never import from, `p_struct` wired as the **re-anchorable prior** on the live **September Fed market** under Option C with DC-5's declared blend weight of 1.0, the § 4d double-count guard live, the market added to `config.DATA_CHANNEL_MARKETS`, and its card moved from "structurally blind" to **"data-channel scored"** with its own calibration track starting at **n=0**. **658 tests green.**
- **The caveat that matters most, stated before the good news is enjoyed:** split by regime, the method is **worse than doing nothing in the holding regime** (0.1528 against base-rate-only's 0.1236) — and the live September market is *in* that regime. It wins big in hiking (0.1360 vs 0.3277) and cutting (0.0772 vs 0.4205), roughly ties in normalisation (0.2060 vs 0.2535), and **loses where it is now deployed.**
- **A second honesty item.** v2 dropped the SEP-concentration weighting because its 8-meeting sweep said `W_SEP = 1.0` was best. On 40 meetings the sweep says the **opposite** (2.0 best). The effect is small and its sign is unstable, which means it was never well-supported in either direction — and the locked 1.0 stands, because that is what "locked" means.
- **One-line status:** GO on a pre-registered 40-meeting test, channel built and live on one market, **no calibration claim** — n=0 settled forecasts under the new method, and the first will be 2026-09-16.

## What this pass is and is not

| | |
|---|---|
| **Is** | 40 retrospective daily trajectories (45 days each) on 40 resolved FOMC decisions, from the § 4f Amendment-2 rule on strictly point-in-time inputs, scored against criteria locked before the first number existed — followed by a build authorised by that result. |
| **Is not** | A calibration claim. n = 40 meetings, **no confidence interval is computed and none is quoted**. The live market has **zero** settled forecasts under the new method. |
| **New files** | `scripts/newsagent_datachannel_v3.py`, `scripts/newsagent_datachannel_snapshot.py`, `newsagent/datachannel.py`, `tests/test_newsagent_datachannel.py`, three CSVs, one plot, this note. |
| **Now touched** (first time in this thread) | `newsagent/config.py` (`DATA_CHANNEL_MARKETS`), `newsagent/run_daily.py` (Option C wiring), `newsagent/dashboard.py` (card copy), the research venv (OpenBB). `fvmodel.py` and `fv_params.json` remain **untouched** — DC-4 holds, α is not refit. |

---

## 1 · What was locked, and when

[[newsagent_data_channel_scoping]] § 4f carries **Amendment 2**, written with every constant fixed before a single v3 number existed. Recorded there and repeated here because it is the whole basis for trusting the result:

- **V3-1 · Criterion C becomes regime-neutral.** The implied direction is taken on the pressure **magnitude** `|r_desired − midpoint|`, not on the raw direction of the news. A release that increases `|gap|` must not raise `p_struct`, on either side of the target.
- **V3-2 · `W_SEP = 1.0`** — allocate evenly across the remaining calendar; the SEP-concentration hypothesis was tested on 8 meetings and not supported. The allocation *framing* is kept.
- **V3-3 · `B1A` stays 1.5** — deliberately **not** moved toward the 2.0–3.0 the v2 sweep preferred.
- **V3-4 · The sample: 40 meetings, listed in full** — 2017–18 normalisation, 2022–23 hiking, the three 2025 cuts, the five resolved 2026 holds. 21 moves / 19 holds; 18 hikes, 3 cuts; 20 SEP / 20 non-SEP.
- **V3-5 · A hard per-meeting leakage guard**, and 2015–16 excluded.
- **Verdict rule unchanged: GO iff A and B and C.** Criterion B stays **peek-compromised** from Amendment 1's disclosure; the weight rests on A.

**What was done before the lock, recorded so nobody has to wonder:** data-availability probes only — confirming ALFRED serves historical SEP vintages, that the Cleveland nowcast starts 2013-07, and reading the FRED target-change dates in order to compose a balanced sample and check it for leakage. No reaction-function output of any kind existed.

### 1a · A coding deviation from the lock, and how it was handled

The first execution of the v3 script scored **45** meetings, not the declared 40: it included the whole 2025 calendar where Amendment 2's table carries only the three 2025 cutting meetings. **That is a coding deviation from the lock, not a change to it.** The lock is the spec, so the script was corrected and re-run on the declared 40. Both results are recorded here so the sample cannot be suspected of having moved after a number was seen:

| Sample | Pooled Brier | base-rate-only | Verdict |
|---|---|---|---|
| 45 meetings (the erroneous first run) | 0.1690 | 0.2622 | GO |
| **40 meetings (the locked sample)** | **0.1617** | **0.2795** | **GO** |

Both pass, and the direction of the difference is against the method's favour in the erroneous run (the five extra 2026-style holds are meetings where `p_struct` scores worse than the base rate). Criteria B and C are identical in both.

## 2 · Inputs — unchanged, still provably point-in-time

The plumbing is the dry run's, unchanged and re-proved: ALFRED realtime vintages for revised series, daily Cleveland Fed nowcast vintages for the unpublished tail, and the never-revised `DFEDTARU`/`DFEDTARL` for the target range (DC-7). The v3 script imports those functions rather than reimplementing them, so a drift between the retro-test and the live channel is impossible.

**The leakage guard (V3-5) earned its place.** v2 verified on three meetings that `DFEDTARU` records a change at its *effective* date, the day **after** the decision. Extending the sample showed that assumption is **not universal**: the December-2015 liftoff is stamped **on its own decision day**, which would have leaked the outcome into its own score. 2015–16 are therefore excluded, and every remaining meeting carries a runtime assertion that the decision-day vintage still shows the pre-decision level. **0 of 40 meetings were dropped** — the guard found nothing left to catch, which is the outcome you want from a guard you added because you found something.

One operational note: v3 needs ~280 distinct (series, realtime-window) ALFRED pulls where v2 needed 56, and FRED answers **HTTP 429** well before that. The pulls are cached on disk per window, so a bounded exponential backoff makes forward progress; no parallelism and no second key.

## 3 · Per-meeting results

*Unit of observation: one FOMC decision.* **G** is total remaining pressure in 25bp clicks. **μ₁ = G × share₁** is the pressure allocated to *this* meeting, the only quantity entering the probability. **p_struct** is P(no change) on the decision day. **Brier** = (p_struct − outcome)²; base-rate-only always says 0.6484. **v1** is the original DC-3 rule on identical inputs. **viol** is regime-neutral criterion-C violations over the 45-day trajectory.

| Meeting | SEP | Regime | Outcome | G | μ₁ | **p_struct** | **Brier** | base | v1 Brier | viol |
|---|---|---|---|---|---|---|---|---|---|---|
| 2017-02-01 | — | normalisation | hold | 1.32 | 0.165 | 0.554 | 0.1990 | 0.1236 | 0.2649 | 0 |
| 2017-03-15 | ✓ | normalisation | **MOVE** | 2.46 | 0.351 | 0.521 | 0.2718 | 0.4204 | 0.3306 | 0 |
| 2017-05-03 | — | normalisation | hold | 1.20 | 0.199 | 0.578 | 0.1784 | 0.1236 | 0.2551 | 0 |
| 2017-06-14 | ✓ | normalisation | **MOVE** | 1.59 | 0.319 | 0.533 | 0.2846 | 0.4204 | 0.3182 | 0 |
| 2017-07-26 | — | normalisation | hold | 0.23 | 0.057 | 0.627 | 0.1391 | 0.1236 | 0.2312 | 0 |
| 2017-09-20 | ✓ | normalisation | hold | 0.71 | 0.236 | 0.564 | 0.1900 | 0.1236 | 0.2469 | 0 |
| 2017-11-01 | — | normalisation | hold | 1.08 | 0.539 | 0.451 | 0.3013 | 0.1236 | 0.2921 | 0 |
| 2017-12-13 | ✓ | normalisation | **MOVE** | 1.08 | 1.080 | 0.271 | 0.0732 | 0.4204 | 0.1637 | 0 |
| 2018-01-31 | — | normalisation | hold | 1.43 | 0.179 | 0.585 | 0.1721 | 0.1236 | 0.2611 | 0 |
| 2018-03-21 | ✓ | normalisation | **MOVE** | 2.07 | 0.296 | 0.542 | 0.2937 | 0.4204 | 0.3375 | 0 |
| 2018-05-02 | — | normalisation | hold | 1.37 | 0.229 | 0.567 | 0.1876 | 0.1236 | 0.2669 | 0 |
| 2018-06-13 | ✓ | normalisation | **MOVE** | 2.50 | 0.500 | 0.466 | 0.2170 | 0.4204 | 0.2860 | 0 |
| 2018-08-01 | — | normalisation | hold | 1.35 | 0.337 | 0.526 | 0.2242 | 0.1236 | 0.2807 | 0 |
| 2018-09-26 | ✓ | normalisation | **MOVE** | 1.67 | 0.557 | 0.445 | 0.1976 | 0.4204 | 0.2599 | 0 |
| 2018-11-08 | — | normalisation | hold | 0.96 | 0.480 | 0.473 | 0.2776 | 0.1236 | 0.2884 | 0 |
| 2018-12-19 | ✓ | normalisation | **MOVE** | 0.99 | 0.986 | 0.299 | 0.0894 | 0.4204 | 0.1726 | 0 |
| 2022-01-26 | — | hiking | hold | 6.83 | 0.854 | 0.339 | 0.4373 | 0.1236 | 0.4886 | 0 |
| 2022-03-16 | ✓ | hiking | **MOVE** | 9.36 | 1.337 | 0.199 | 0.0396 | 0.4204 | 0.0083 | 0 |
| 2022-05-04 | — | hiking | **MOVE** | 7.29 | 1.215 | 0.230 | 0.0528 | 0.4204 | 0.0175 | 0 |
| 2022-06-15 | ✓ | hiking | **MOVE** | 11.09 | 2.218 | 0.062 | 0.0039 | 0.4204 | 0.0004 | 0 |
| 2022-07-27 | — | hiking | **MOVE** | 8.22 | 2.054 | 0.078 | 0.0061 | 0.4204 | 0.0009 | 0 |
| 2022-09-21 | ✓ | hiking | **MOVE** | 8.95 | 2.983 | 0.021 | 0.0004 | 0.4204 | 0.0004 | 0 |
| 2022-11-02 | — | hiking | **MOVE** | 6.82 | 3.408 | 0.020 | 0.0004 | 0.4204 | 0.0004 | 0 |
| 2022-12-14 | ✓ | hiking | **MOVE** | 1.60 | 1.600 | 0.146 | 0.0214 | 0.4204 | 0.0625 | 0 |
| 2023-02-01 | — | hiking | **MOVE** | 6.67 | 0.834 | 0.346 | 0.1194 | 0.4204 | 0.0879 | 0 |
| 2023-03-22 | ✓ | hiking | **MOVE** | 6.00 | 0.858 | 0.338 | 0.1139 | 0.4204 | 0.0961 | 0 |
| 2023-05-03 | — | hiking | **MOVE** | 5.04 | 0.840 | 0.344 | 0.1181 | 0.4204 | 0.1246 | 0 |
| 2023-06-14 | ✓ | hiking | hold | 3.78 | 0.757 | 0.372 | 0.3941 | 0.1236 | 0.4572 | 0 |
| 2023-07-26 | — | hiking | **MOVE** | 4.03 | 1.006 | 0.290 | 0.0839 | 0.4204 | 0.1024 | 0 |
| 2023-09-20 | ✓ | hiking | hold | 1.10 | 0.367 | 0.515 | 0.2348 | 0.1236 | 0.3364 | 0 |
| 2023-11-01 | — | hiking | hold | 0.79 | 0.394 | 0.505 | 0.2449 | 0.1236 | 0.3080 | 0 |
| 2023-12-13 | ✓ | hiking | hold | 0.55 | 0.549 | 0.448 | 0.3045 | 0.1236 | 0.2704 | 0 |
| 2025-09-17 | ✓ | cutting | **MOVE** | 2.92 | 0.975 | 0.300 | 0.0899 | 0.4204 | 0.0033 | 0 |
| 2025-10-29 | — | cutting | **MOVE** | 2.33 | 1.166 | 0.244 | 0.0594 | 0.4204 | 0.0011 | 0 |
| 2025-12-10 | ✓ | cutting | **MOVE** | 1.03 | 1.030 | 0.287 | 0.0822 | 0.4204 | 0.0033 | 0 |
| 2026-01-28 | — | holding | hold | 0.39 | 0.048 | 0.632 | 0.1357 | 0.1236 | 0.2594 | 0 |
| 2026-03-18 | ✓ | holding | hold | 0.35 | 0.051 | 0.631 | 0.1363 | 0.1236 | 0.2611 | 0 |
| 2026-04-29 | — | holding | hold | 0.26 | 0.044 | 0.633 | 0.1350 | 0.1236 | 0.2572 | 0 |
| 2026-06-17 | ✓ | holding | hold | 0.71 | 0.142 | 0.599 | 0.1612 | 0.1236 | 0.3343 | 0 |
| 2026-07-29 | — | holding | hold | 1.02 | 0.255 | 0.557 | 0.1960 | 0.1236 | 0.4323 | 0 |
| **Pooled (40)** | 20 | — | 21 moves | — | — | mean 0.403 | **0.1617** | **0.2795** | 0.2347 | **0** |

![Data-channel v3 — 40 resolved FOMC decisions](../../data/analysis/plots/news_agent/newsagent_datachannel_v3.png)

**Chart read.** *Top:* `p_struct` on each decision day, coloured by regime; **circles held, crosses moved**, and the dashed line is base-rate-only at 0.6484. A well-behaved method puts circles high and crosses low, and it broadly does — the 2022 crosses collapse to 0.02–0.23 as the hiking pressure builds, and the 2026 circles sit just under the base rate. The visible failures are the circles that sit low: 2022-01-26 (held while the model saw 6.8 clicks of pressure) and 2023-06-14 (the "skip" meeting). *Bottom:* per-meeting Brier against base-rate-only. The grey bars are flat by construction; the terracotta bars beat them almost everywhere a move happened and lose modestly on the quiet holds at the right-hand end.

## 4 · Verdict per criterion

| Criterion | Value | Comparator | Result |
|---|---|---|---|
| **A** Pooled mean Brier, 40 meetings | **0.1617** | base-rate-only **0.2795** | **PASS** |
| **B** July-2026 market, the dry run's own 15-day window | **0.1926** | bar **0.2144** | **PASS** *(peek-compromised)* |
| **C** Regime-neutral directional violations | **0** | 0 | **PASS** |
| — leakage guard (V3-5) | 0 dropped of 40 | — | clean |
| — *reported:* v1 rule, same 40 meetings | 0.2347 | — | |

**Verdict rule as locked: GO requires A AND B AND C. All three pass → GO.**

**Criterion C, and why its passing means something now.** v2's version produced 6 violations, all on cutting-pressure meetings, because it read the raw direction of the news rather than the direction of the *pressure*. The v3 version is not a weaker test — it is the test v2 meant to run, applied on both sides of the cycle, and it is now exercised against **18 hikes, 3 cuts and 19 holds** rather than 2 directional days on one hiking market. Zero violations across 40 trajectories is the first evidence the rule is well-behaved that is worth anything.

## 5 · By regime — including where the method loses

This split is **reported, never a bar**, and it exists so a method that works in one regime and not another cannot hide inside a pooled average.

| Regime | n | **p_struct** | base-rate-only | v1 rule | violations |
|---|---|---|---|---|---|
| hiking (2022–23) | 16 | **0.1360** | 0.3277 | 0.2134 | 0 |
| cutting (2025) | 3 | **0.0772** | 0.4205 | 0.0026 | 0 |
| normalisation (2017–18) | 16 | **0.2060** | 0.2535 | 0.2763 | 0 |
| **holding (2026)** | 5 | **0.1528** | **0.1236** | 0.3089 | 0 |

**Read, and this is the most important paragraph in the note.** The method earns its keep exactly where there is pressure to read: in the hiking cycle it more than halves the base rate's error, and in the cutting sequence it does better still. In the slow normalisation of 2017–18 it is a modest improvement. **In the quiet holding regime it is worse than doing nothing** — 0.1528 against 0.1236 — because when `|gap|` is small the model still shades below the base rate while the answer keeps being "hold". And the market this channel has just been wired to, the **September 2026 Fed decision, sits in that holding regime**: today's `p_struct` is 50.7% against a base rate of 64.8%.

That is not a reason to withhold the build — the pre-registered bars were met on the pooled set, which is what they were set on — but it is the reason no calibration claim is made here, and it is the specific thing the n=0 forward track will test first.

**The cutting column also flatters v1 and should not be read as v1 doing well.** v1's Brier of 0.0026 there comes from saying "move" almost everywhere (mean p 0.298 in v2's measurement); all three cutting meetings moved, so it collects three near-perfect scores and pays for that posture on every hold — which is exactly what its 0.4323 on 2026-07-29 is.

## 6 · Sensitivity — reported, never selected from, and one finding that reverses v2

The verdict is on the declared `B1A = 1.5` / `W_SEP = 1.0` and nothing else.

| B1A | Pooled Brier | mean p | | W_SEP | Pooled Brier | mean p |
|---|---|---|---|---|---|---|
| 0.5 | 0.2093 | 0.554 | | **1.0** *(declared)* | **0.1617** | 0.403 |
| 1.0 | 0.1739 | 0.470 | | 1.5 | 0.1578 | 0.412 |
| **1.5** *(declared)* | **0.1617** | 0.403 | | 2.0 | **0.1571** | 0.419 |
| 2.0 | 0.1628 | 0.351 | | 3.0 | 0.1587 | 0.431 |
| 3.0 | 0.1833 | 0.276 | | | | |

**B1A.** On 8 meetings the sweep preferred 2.0–3.0 and the declared 1.5 sat away from the optimum; on 40 it is **at** the optimum (1.5 → 0.1617, 2.0 → 0.1628). The declared value was chosen from stated reasoning before either sweep existed, and the larger sample moved toward it rather than away. That is reassuring but should not be over-read: the surface is flat between 1.0 and 2.0.

**W_SEP, and this reverses a v2 conclusion.** v2 dropped the SEP-concentration hypothesis because its 8-meeting sweep put `W_SEP = 1.0` ahead of 2.0 (0.1245 vs 0.1273). On 40 meetings the ordering **flips**: 2.0 is best (0.1571) and the locked 1.0 is worst of the four (0.1617). The honest conclusion is not "we should have kept it" — it is that **the effect is small and its sign is unstable across samples, i.e. it was never supported in either direction**, and v2's stated reason for dropping it ("tested and not supported") was right for the wrong reason. `W_SEP` stays at the locked 1.0, because a constant is not re-tuned after seeing an outcome; the reversal is recorded so the next lock can argue from both samples rather than one.

## 7 · The build — what shipped the same day

Criterion A, B and C all passed, which is what § 7 row 1 authorises. Everything below was built after the verdict, not before it.

### 7a · The AGPL boundary, and a finding about OpenBB

The lean set is installed (`openbb-core`, `-economy`, `-fred`, `-bls`, `-nasdaq`, `-federal-reserve`; ~30 packages, no version changes to existing pins). The boundary Justin acknowledged is enforced in code, not by convention: **`newsagent/*` must never import OpenBB**, and `tests/test_newsagent_datachannel.py::test_newsagent_never_imports_openbb` walks the AST of every module in the package to assert it.

**Measured finding: OpenBB cannot serve this channel's inputs.** `obb.economy.fred_series` exposes `symbol / start_date / end_date / limit / provider` and **no realtime (vintage) parameters at all**. DC-7 is categorical — "current FRED values for revised series must **not** be used to reconstruct an as-of state; ALFRED vintages or nothing" — so every vintage-critical series is pulled by direct ALFRED call, exactly as the dry run and v2/v3 did and proved lookahead-free. **OpenBB is installed and is not on the critical path.** It buys nothing the channel needs and adds an AGPL dependency surface of ~30 packages. **Recommendation for Justin: consider `uv remove`-ing it** — the authorisation was to install it when the build started, and the build started and did not need it. It is left installed pending that call rather than removed unilaterally.

### 7b · The pieces

| Piece | What it does |
|---|---|
| `scripts/newsagent_datachannel_snapshot.py` | The **offline ingest boundary**. Computes `p_struct` under the locked rule from vintage-only inputs and writes `data/newsagent/datachannel/p_struct_latest.json`. Imports every constant from the v3 module so retro-test and live channel cannot drift apart. |
| `newsagent/datachannel.py` | The **read side**. Loads the JSON, hands `p_struct` to Stage B, and implements the § 4d guard. Computes nothing, fetches nothing, imports no OpenBB. |
| `newsagent/run_daily.py` | **Option C wiring.** On a data-channel market `p0` is replaced by `p_struct` (DC-5 blend weight **1.0**, declared); α, λ, the shift caps and the band are untouched (**DC-4**). |
| `newsagent/config.py` | `DATA_CHANNEL_MARKETS` now contains the September Fed market — it joined **on the day its number really changed method**, never in advance. |
| `newsagent/dashboard.py` | Card copy: **"◆ data-channel scored"**, the mandatory **n=0** transition line, the market-implied absence rendered in words, and a line naming any double-counted articles. |

### 7c · The § 4d double-count guard

The news packet already contains articles *about* the data. Once the print moves `p_struct`, the same information would move `A_t` again. The guard is a **declared rule over fields we already have** — no new LLM call: an article on a data-channel market that Stage A tagged `economic` **and** whose visible text names one of the mapped releases contributes **0** to the day's evidence score.

Both legs are required, and the tests pin why: the event-type leg alone would swallow a Fed governor's speech (genuinely new information the data channel never saw), and the keyword leg alone would catch a campaign story that mentions inflation in passing. Excluded items are **still fetched, still displayed and still counted in the evidence list** — labelled, not hidden, because suppressing them from the page would misrepresent what the model read. **On the first live run it fired once**, on a Guardian story reporting the latest inflation print.

### 7d · The MPT hole, handled rather than discovered later

The Atlanta Fed Market Probability Tracker is display-only under DC-6 and **stops quoting a 3-month window once that window opens** (dry-run § 7), so for the September meeting it is dark exactly when the meeting gets interesting. The snapshot writes a **structured absence** rather than omitting the field, and the card renders it in words: *"the Atlanta Fed Market Probability Tracker stops quoting a 3-month window once that window opens, so no market-implied context is available for this meeting."* Test-enforced, including the case where the block is missing entirely.

### 7e · What the page says today

| | Before | After |
|---|---|---|
| Fed-September FV | 44.0% | **50.7%** |
| Anchor | onboarding prior | **`p_struct` = 50.7%** |
| Method label | `news` | **`news+data`** (switch dated in the ledger history) |
| Card | "◆ data-driven — … a data-evidence channel … is NOT yet built" | **"◆ data-channel scored"** + n=0 line + MPT absence |
| Divergence flag | YES (−22.5pp) | YES (−15.8pp vs mid 66.5%) |

**The published number equals the structural anchor exactly**, because no day cleared the slow-market decisive-evidence threshold and the drip filter zeroed the day's evidence (`carry_pp` 0.0, `drip_filtered` true, 0 contributing articles). That is the same mechanism that pinned the news-FV to its prior throughout the v1 dry run — on this market, on most days, the anchor *is* the answer. It makes the choice of anchor consequential, which is the point of Option C, and it also means the news half of "news+data" has yet to do any work here.

## 8 · Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions.** The Taylor coefficients (±0.5) remain **borrowed** from the macro literature, not derived here. `B1A = 1.5` and `W_SEP = 1.0` are declared judgments; the σ = 0.063pp core-PCE surprise RMSE is v1's measurement, unchanged. SEP Q4/Q4 projections are compared against a 12-month y/y — an approximation. The horizon is the SEP projection year, so a December meeting receives the entire remaining gap. The regime labels in § 5 are **hand-assigned for reporting** and are never an input. The double-count keyword list is declared and will need maintenance as coverage language changes.

**Live-only unknowns.** Whether the method's holding-regime weakness (§ 5) bites on the very market it is now deployed on — this is the first thing the forward track will show, and it is the reason for the n=0 line. Whether the FRED-vs-Cleveland ingestion-clock gap matters on a meeting where a print lands *inside* the final week. Whether the § 4d guard's keyword list catches the right articles at scale — it fired once, correctly, on one day. Whether a 7-day snapshot-staleness bound is right: too tight and the channel degrades needlessly, too loose and it anchors on a stale macro state.

**Power honesty.** **n = 40 meetings across three regimes; the live method has n = 0 settled forecasts.** No confidence interval is computed and none should be quoted. The pooled Brier owes something to sample composition — 21 of 40 outcomes were moves, and a method that shades below the base rate scores well on a sample that moves more than half the time. The July result (criterion B) is **peek-compromised** and is not independent evidence. α was **not** refit and stays 2.85: the data channel enters through `p0`, not through `A_t`, so DC-4 says there is nothing to refit — and the fit sample contains no data-channel days regardless.

## 9 · Decision and next step

**Decision: GO, and the channel is built and live on one market.** Three passes: v1 failed on accuracy, v2 passed accuracy and failed a criterion that turned out to be regime-specific, v3 fixed the criterion, dropped the unsupported hypothesis, extended the sample eightfold across three rate regimes, and passed all three bars with zero directional violations and zero leakage drops. The channel that resulted is deliberately narrow: **one market**, anchored on a declared rule, labelled `news+data`, with its own calibration track starting at zero.

**Next, in order:**

1. **2026-09-16 is the first real test** — the first settled forecast under the new method, on a market in the regime where § 5 says the method is weakest. Settle it, score it, and let the per-method track say what it says. Do not merge it with the news track, ever.
2. **Re-run the snapshot before every publish.** `p_struct` is only re-anchored when a fresh snapshot exists; the code degrades to the onboarding prior after 7 days and prints why, but a degraded run publishes a number that is not what the card claims to be. Worth a line in the runbook.
3. **Decide on OpenBB (§ 7a).** It is installed, authorised, and unused — it cannot serve vintages, which is the only thing this channel needs from a data layer.
4. **Do not extend the channel to a second market on this evidence.** The two California markets remain unmappable at any tier, and the only other Fed market has resolved. If a new Fed market joins the slate, it joins the channel only after the September forecast has settled and been scored.
5. **Do not claim calibration.** n=0 under the new method. The historical Brier on the page belongs to the news-only method and is attributed to it.

## 10 · Outputs

- **Scripts:** `scripts/newsagent_datachannel_v3.py` (the retro-test; `--chart`), `scripts/newsagent_datachannel_snapshot.py` (the offline ingest boundary).
- **Module:** `newsagent/datachannel.py` — read side, Option C anchor, § 4d guard, MPT absence.
- **Amendment:** [[newsagent_data_channel_scoping]] § 4f **Amendment 2**, locked before any computation, with the sample, criteria, verdict rule and the carried peek disclosure.
- **Tests:** `tests/test_newsagent_datachannel.py` — 22 tests (AGPL boundary by AST, double-count guard both legs, snapshot degradation, DC-8 labelling, card copy incl. the n=0 line and the MPT absence). Repo-wide **658 green**.
- **CSVs** (git-ignored): `newsagent_datachannel_v3_{meetings,trajectory,sensitivity}.csv`.
- **Plot:** `data/analysis/plots/news_agent/newsagent_datachannel_v3.png`.
- **Run records** (git-ignored): `data/newsagent/datachannel/v3_results.json`, `p_struct_latest.json`, `p_struct_2026-08-24.json`.
- **Not touched:** `newsagent/fvmodel.py`, `newsagent/fv_params.json` (α stays 2.85), the historical fit sample, and any deployment — the Vercel push is Justin's.
