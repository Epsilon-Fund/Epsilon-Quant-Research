---
title: "EXPLORATORY, IN-SAMPLE: if we had traded our disagreements with the mid, what would have happened? (retrospective, 44-market reconstruction)"
created: 2026-08-26
status: exploratory — hypothesis-generating only. NOT a gate, NOT a result, NOT publishable. The v0 beat-the-mid claim stays CLOSED; the Q-DIV-EDGE sample floor is NOT met. Headline under the declared rule (T=15pp, first qualifying day, 1.0pp haircut + category fees): 19 positions across 19 markets, $5.41 deployed, total PnL **-$0.41, ROI -7.7%**, win rate 26.3%. Deleting the single best market takes it to -$1.35; deleting the single worst takes it to +$0.32 — one market of nineteen flips the sign. The confidence-gated subset (the full shipped divergence rule) is 4 positions, ROI -56.5%. Verdict: no signal, and no *evidence* of no signal either; the measurement has no power. Do not pursue until the forward Q-DIV-EDGE sample exists.
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
  - exploratory
  - divergence
  - in-sample
---
# EXPLORATORY, IN-SAMPLE — retrospective PnL of trading our own disagreements with the mid

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]] · [[CODEX]] · [[TODO]] · Table terms: [[polymarket_table_dictionary]]
> Input: the trajectory CSV from [[newsagent_simulated_live_findings]]. Pre-registered question: [[newsagent_observatory_v2_findings]] § Q-DIV-EDGE. Closed claim that this note does **not** reopen: [[newsagent_v0_gate_findings]].

> [!warning] Status label — it travels with every number below
> **EXPLORATORY · IN-SAMPLE · HYPOTHESIS-GENERATING.** This pass may **not** be published on the showcase page, may **not** be quoted as an edge, and may **not** be summarised without this label. It is a measurement made to decide whether a *real* study is worth designing later. It is not that study.

## Plain-English Summary

- **What this is.** Justin asked a natural question: the news agent sometimes disagrees sharply with the Polymarket mid — if we had actually traded those disagreements, would we have made money? This note answers that question *retrospectively*, on the 44 already-resolved markets from the simulated-live reconstruction, under a rule declared in writing before any PnL number existed.
- **What this is not, and this is load-bearing.** The claim "our number beats the mid" was tested and **failed** a pre-registered gate in [[newsagent_v0_gate_findings]]. That closure **stands and is not reopened here.** The pre-registered forward version of this question, **Q-DIV-EDGE**, needs ≥30 resolved flagged snapshots across ≥10 distinct markets on the *forward* ledger before it may be run. **That floor is not met** — the forward ledger is at n=4 settlements. This is the smaller, honest, retrospective stand-in Justin asked for, and it carries all the weaknesses that implies.
- **The headline.** Under the declared rule — take our side at the mid whenever `|FV − mid| ≥ 15pp`, first qualifying day per market, one contract, hold to resolution, 1.0pp spread haircut plus the category fee — the result is **19 positions across 19 markets, $5.41 of capital deployed, total PnL −$0.41, ROI −7.7%, win rate 26.3% (5 of 19)**.
- **The number that actually matters is not the headline.** **Removing the single best market turns −$0.41 into −$1.35. Removing the single worst turns it into +$0.32.** One market out of nineteen decides the sign. At this sample size the headline is not a measurement of anything; it is a coin landing.
- **The confidence-gated subset is worse and even thinner.** Applying the *full* shipped divergence rule (gap ≥ 15pp **and** band half ≤ 12pp **and** ≥5 relevant articles in 72h) leaves **4 positions**, total **−$1.30 on $2.30 deployed, ROI −56.5%**. The gate that was supposed to select our *best* disagreements selected our worst ones here. On four positions that is also meaningless.
- **Fed, called out as asked.** 3 Fed positions, **all three lost**, −$0.35 on $0.35 deployed (ROI −100%). Two of the three had **zero relevant articles** — they were the model disagreeing with the market from ignorance, not from evidence.
- **The verdict.** No edge is visible, and — equally important — **no absence of edge is demonstrated either.** The measurement has no power. **Do not pursue this as a strategy, and do not pursue it as a study until the forward Q-DIV-EDGE sample exists.** Full reasoning in § 8.

---

## 1 · Why this note exists, and what it is allowed to say

Three separate things are easy to confuse, so they are separated here once and referred back to throughout.

| | The claim | Status | This note's relationship to it |
|---|---|---|---|
| **The v0 gate** | "The news agent's fair value is more accurate than the Polymarket mid." | **CLOSED — FAILED.** Brier 0.547 vs the mid's 0.354; early-stop fired ([[newsagent_v0_gate_findings]]). | **Not reopened.** Nothing here is a mid-relative accuracy claim. |
| **Q-DIV-EDGE** | "Among *forward* snapshots where the shipped divergence flag fired, is taking our side at the mid +EV?" | **Pre-registered, NOT RUN.** Sample floor ≥30 resolved flagged snapshots across ≥10 markets; forward ledger is at n=4 settlements. | **Not this.** This note is retrospective and in-sample; Q-DIV-EDGE is forward and out-of-sample. They must never be conflated or pooled. |
| **This note** | "On the 44 already-resolved reconstruction markets, what would the declared trade rule have returned?" | **EXPLORATORY.** No gate, no bar, no verdict-by-threshold. | Its only legitimate output is a *hypothesis* and a recommendation about whether to build the real study later. |

The distinction between the second and third row is the whole point. Q-DIV-EDGE is designed so that flags are identified *before* outcomes are known and the sample is large enough to survive family clustering. This pass has neither property. It is run anyway because a cheap, clearly-labelled look is more useful than a guess — provided it is labelled, which is what the warning banner above is for.

---

## 2 · Pre-registration — locked before any number existed

Written to the scratchpad and locked before the first PnL row was computed, reproduced verbatim here. The one correction made during implementation is recorded honestly in § 2a.

**P-1 · Input.** `data/analysis/csv_outputs/news_agent/newsagent_simlive_trajectory.csv` — the 326 reconstructed snapshot rows over 44 resolved markets from [[newsagent_simulated_live_findings]]. No other input; the model is not re-run.

**P-2 · Rule.** On each reconstructed snapshot day that carries a mid and where `|FV − mid| ≥ T`:

- `FV > mid` → **buy YES** at `mid`
- `FV < mid` → **buy NO** at `1 − mid`

One contract per position; hold to resolution; payoff 1 or 0. **T = 15pp, DECLARED** — this is the threshold the shipped divergence flag already uses, not a value chosen by looking at returns.

**P-3 · Entry convention.** **PRIMARY: the first qualifying day only**, at most one position per market. **SECONDARY (declared at the same moment, reported as a companion, never as the headline): every qualifying day.** Both were fixed before any PnL was computed.

**P-4 · Costs, declared.**

- *(a) Fees.* The canonical repo schedule, `FEE_BY_CATEGORY` in `scripts/dali_block_a1_analyze.py`, applied unchanged: `fee_per_contract = fee_rate(category) × p × (1 − p)`, where `p` is the entry price. `fee_rate` runs **Geopolitics 0.00 · Sports 0.03 · Politics/Finance/Tech 0.04 · Economics/Culture/Weather/Other 0.05 · Crypto 0.07**. At `p = 0.5` that spans **0% (geopolitics) to 1.75% ≈ 1.8% (crypto)** of notional — the "0% geopolitics through 1.8% crypto" ledger. Category is assigned by the canonical mechanical `family_category(family)` mapping, applied unchanged. Fee is charged **at entry only**, because resolution is a settlement rather than a taker trade.
- *(b) Spread haircut.* **DECLARED = 1.0pp (0.01)**, added to the entry price on whichever side we buy, entry clipped to [0.01, 0.99]. Rationale: the trajectory CSV carries only the mid, so trading at the mid is optimistic; a taker crossing half a 2pp spread pays about 1pp. One value, declared, not tuned. **The zero-haircut and zero-cost results are reported beside the headline** so the sensitivity is visible.

**P-5 · Readout.** Total PnL and ROI; a **mandatory per-market PnL table**; best- and worst-market share of the total; win rate; position count; the same for the **confidence-gated subset** (gap ≥ 15pp AND band half ≤ 12pp AND `n_rel` ≥ 5); the family breakout with Fed called out and singletons marked; every **prior-only** position (`n_rel = 0`) flagged, with PnL reported with and without them.

**P-6 · Anti-post-hoc.** The declared T = 15pp is the headline, full stop. A sensitivity table over the declared grid `T ∈ {5, 10, 15, 20, 25, 30}pp` is reported **as context only, never as the headline**, and is not used to reselect T. **No market may be dropped after its result is seen.** No confidence interval is computed or quoted. There is no pass/fail bar.

**P-7 · Declared caveats.** α was fitted on these markets; the mid is reconstructed, not executable; family clustering breaks independence; a handful of positions can produce any headline at this n; prior-only positions are disagreements from ignorance. All are expanded in § 7.

### 2a · One declared correction, recorded rather than hidden

The first implementation sized positions at **$1 of capital each** (which buys `1/entry` contracts) rather than at **one contract each**. The pre-registration's own words — "payoff 1 or 0" — specify one contract, so the code was corrected to that before any result was written down. Because a number had already been produced under the other convention, **both are reported** (§ 4c) rather than the corrected one silently replacing it. That matters here more than it usually would: **the two conventions disagree on the sign**, and disclosing both is the only way the reader can see that the sign was a sizing artifact rather than a finding. Nothing was selected on the basis of which looked better.

### 2b · A worked example, so the rule is inspectable

Take `us-x-iran-ceasefire-by-april-7` on **2026-03-25**, 13 days before resolution. The reconstruction was showing **FV 5.5%**; the mid was **27.5%**. The gap is 22.0pp, which clears T = 15pp, and this is the market's first qualifying day, so it enters. FV < mid, so the rule buys **NO**: the raw price is `1 − 0.275 = 0.725`, the declared 1.0pp haircut makes the entry **0.735**, and the `iran` family maps to **Geopolitics**, whose fee rate is **0.00**, so this position pays no fee at all. The market resolved **YES** — there was a ceasefire. Our contract paid 0, and the position lost its full entry: **−$0.735**. That single row is the worst market in the sample and, on its own, is 26.6% of all the losses in it.

---

## 3 · The sample

Of the **326** reconstructed snapshots across **44** resolved markets, **227 carry a mid** and are therefore tradeable at all — the remaining 99 have no price to trade against and are excluded by construction, not by choice. Applying T = 15pp leaves **56 qualifying snapshots across 19 markets**; the primary first-day-only convention takes **19 positions, one per market**.

So the honest framing of the sample is: **19 independent-ish bets, drawn from 19 of the 44 markets, clustered into 13 event families of which 9 are singletons.** Every number in § 4 rests on those 19 rows, and the per-market table in § 4b is there precisely so no reader has to take the aggregate on trust.

---

## 4 · Results

### 4a · Headline — the declared rule

*Unit of observation: one position = one market's first qualifying snapshot. `stake` is capital actually deployed (entry price + fee per contract), so ROI is measured on capital at risk rather than on a notional.*

| metric | value |
|---|---|
| positions | **19** (one per market, 19 distinct markets) |
| capital deployed | **$5.41** |
| fees paid | $0.09 |
| **total PnL** | **−$0.41** |
| **ROI on capital deployed** | **−7.7%** |
| win rate | **26.3%** (5 of 19) |
| sides taken | 8 YES, 11 NO |
| gross gains / gross losses | +$2.35 / −$2.76 |
| best single market | `qatarenergy…` **+$0.94** — 39.9% of all gains |
| worst single market | `us-x-iran-ceasefire-by-april-7` **−$0.74** — 26.6% of all losses |
| **total with the best market removed** | **−$1.35** |
| **total with the worst market removed** | **+$0.32** |

**Read.** The headline is mildly negative, and the last two rows say why you should not care that it is mildly negative. The total is −$0.41 while a single market swings it by $0.94 in one direction and $0.74 in the other — **the sign of this result is decided by which one market you delete.** The conventional way to report "best market share of total PnL" breaks down when the total is near zero (it is −226% and +177% respectively, which is arithmetic, not information), so the shares are given against **gross** gains and losses instead, and the delete-one-market rows are given directly. That is the finding of this section: not the −7.7%, but the fact that −7.7% is not a measurement.

### 4b · Per-market PnL — mandatory table

*One row per position, sorted by PnL. `gap` = |FV − mid| in percentage points at entry. `band` = the model's half-band width in pp. `n_rel` = relevant articles in the 72h window; **`n_rel = 0` means the model disagreed with the market having read nothing relevant** and is flagged **PRIOR-ONLY**. `entry` = price paid after the 1.0pp haircut. `y` = resolved outcome (1 = YES). `cat` = fee category from the canonical mapping.*

| market | family | cat | date | d-to-end | side | FV% | mid% | gap | band | n_rel | entry | y | W/L | PnL | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qatarenergy-announces/resumes-lng-production-in-qatar | other:qatarenergy | Other | 2026-04-05 | 25 | YES | 23.1 | 5.0 | 18.1 | 12.0 | 2 | 0.060 | 1 | **W** | **+0.937** | |
| will-trump-talk-to-xi-jinping-in-march | trump_admin | Other | 2026-03-02 | 29 | NO | 29.3 | 57.0 | 27.7 | 21.8 | 0 | 0.440 | 0 | **W** | **+0.548** | **PRIOR-ONLY** |
| ukraine-strikes-another-tanker-in-black-sea-by-march | ukraine_russia | Tech | 2026-03-26 | 5 | YES | 84.7 | 55.5 | 29.2 | 19.2 | 1 | 0.565 | 1 | **W** | **+0.425** | |
| us-x-iran-permanent-peace-deal-by-may-31 | iran | Geopolitics | 2026-05-06 | 25 | NO | 6.7 | 23.5 | 16.8 | 12.7 | 5 | 0.775 | 0 | **W** | **+0.225** | |
| israel-closes-its-airspace-by-may-31 | israel | Other | 2026-05-10 | 21 | NO | 5.3 | 23.0 | 17.7 | 21.8 | 0 | 0.780 | 0 | **W** | **+0.211** | **PRIOR-ONLY** |
| us-forces-seize-another-oil-tanker-by-april-30 | other:forces | Other | 2026-04-25 | 5 | NO | 17.0 | 100.0 | 83.0 | 12.4 | 2 | 0.010 | 1 | L | −0.011 | |
| us-x-cuba-diplomatic-meeting-by-may-31 | other:cuba | Other | 2026-06-01 | 29 | NO | 10.9 | 100.0 | 89.0 | 18.6 | 2 | 0.010 | 1 | L | −0.011 | |
| netanyahu-out-by-april-30 | israel | Other | 2026-04-29 | 1 | YES | 18.1 | 0.1 | 17.9 | 18.4 | 4 | 0.011 | 0 | L | −0.012 | |
| will-there-be-no-change-in-fed-interest-rates-after-… (Apr) | **fed** | Other | 2026-04-08 | 21 | NO | 56.0 | 98.5 | 42.5 | 17.8 | 0 | 0.025 | 1 | L | −0.027 | **PRIOR-ONLY** |
| epstein-suicide-note-released-by-may-8 | other:epstein | Other | 2026-05-02 | 29 | YES | 22.9 | 1.8 | 21.2 | 18.4 | 3 | 0.028 | 0 | L | −0.029 | |
| russia-x-ukraine-ceasefire-by-february-28 | ukraine_russia | Tech | 2026-02-03 | 25 | YES | 33.8 | 4.0 | 29.8 | 16.5 | 7 | 0.051 | 0 | L | −0.052 | |
| will-the-fed-decrease-interest-rates-by-25-bps-after-… | **fed** | Other | 2026-02-17 | 29 | YES | 30.7 | 6.5 | 24.2 | 10.2 | 2 | 0.075 | 0 | L | −0.078 | |
| us-forces-enter-iran-by-april-30 | iran | Geopolitics | 2026-04-05 | 25 | NO | 20.2 | 91.5 | 71.3 | 19.2 | 1 | 0.095 | 1 | L | −0.095 | |
| starmer-out-by-may-31 | other:starmer | Other | 2026-05-18 | 13 | YES | 33.3 | 14.6 | 18.7 | 14.5 | 2 | 0.156 | 0 | L | −0.162 | |
| bab-el-mandeb-strait-effectively-closed-by-april-30 | other:mandeb | Other | 2026-04-01 | 29 | YES | 39.5 | 17.5 | 22.0 | 12.0 | 5 | 0.185 | 0 | L | −0.193 | |
| will-there-be-no-change-in-fed-interest-rates-after-… (Jul) | **fed** | Other | 2026-07-08 | 21 | NO | 53.7 | 77.5 | 23.8 | 17.8 | 0 | 0.235 | 1 | L | −0.244 | **PRIOR-ONLY** |
| will-the-next-prime-minister-of-hungary-be-peter-magyar | hungary | Other | 2026-03-14 | 29 | NO | 40.7 | 63.5 | 22.8 | 21.8 | 0 | 0.375 | 1 | L | −0.387 | **PRIOR-ONLY** |
| will-a-us-anti-cartel-operation-outside-of-the-united-… | other:will | Other | 2026-04-01 | 29 | NO | 13.0 | 29.5 | 16.5 | 19.2 | 1 | 0.715 | 1 | L | −0.725 | |
| us-x-iran-ceasefire-by-april-7 | iran | Geopolitics | 2026-03-25 | 13 | NO | 5.5 | 27.5 | 21.9 | 12.0 | 6 | 0.735 | 1 | L | **−0.735** | |
| **TOTAL** | | | | | | | | | | | **$5.41 deployed** | | **5W/14L** | **−0.414** | |

![Per-market PnL, exploratory divergence rule](../../data/analysis/plots/news_agent/newsagent_divergence_pnl.png)

**Chart read.** One horizontal bar per position, sorted worst at the bottom to best at the top, in dollars of PnL on a one-contract position. Green = a profitable position, red = a losing one, **brown = a prior-only position (zero relevant articles)**. The thing to notice is not the sum but the shape: **five bars carry essentially all the magnitude in each direction, and the other fourteen are near-zero slivers.** Those slivers are the positions entered at 1–5c, where the rule bought a near-certain-loser for a penny and lost the penny. Aggregating 19 rows of which 14 are rounding error is exactly the situation where a total tells you nothing.

**Read of the table.** Two structural facts are visible here that the aggregate hides.

First, **the win/loss split is not where the money is.** The rule won only 5 of 19, but all five wins were entered between 6c and 78c, where a win is worth something, while eight of the fourteen losses were entered below 10c and cost almost nothing. That is why a 26.3% win rate produces only a −7.7% ROI rather than a catastrophe — and it is also the mechanism that makes the sizing convention (§ 4c) flip the sign.

Second, **the very large gaps are not the good positions.** The three largest gaps in the sample — 89.0pp, 83.0pp, 71.3pp — all lost. Those are the rows where the mid was at 91–100c and the model was still saying 11–20%, i.e. the market had already priced a resolved-in-practice event and the reconstruction had not noticed. A large gap is at least as likely to be the model being stale as the model being right, which is a point against gap magnitude as a selection signal.

### 4c · The three declared cost variants, plus the sizing companion

*All four rows are the same 19 positions; only the costing or the sizing changes.*

| variant | capital deployed | total PnL | ROI | note |
|---|---|---|---|---|
| **declared (1.0pp haircut + fees)** | $5.41 | **−$0.41** | **−7.7%** | **the headline** |
| zero haircut, fees only | $5.25 | −$0.25 | −4.7% | declared sensitivity |
| zero cost (no haircut, no fees) | $5.16 | −$0.16 | −3.2% | the fully optimistic bound |
| *companion: $1 notional per position* | $19.60 | *+$3.68* | *+18.8%* | *§ 2a — different sizing, opposite sign* |

**Read.** The first three rows say that **costs are not what makes this negative** — the fully cost-free version is still −3.2%. Total fees across all 19 positions are **$0.09**, because most of the sample maps to fee-free `Geopolitics` or to small entries where `p(1−p)` is tiny. The interesting row is the fourth: sizing every position at $1 of capital instead of one contract turns −7.7% into **+18.8%**. That is not a better result, it is a **leverage artifact** — $1 at a 1c entry buys 100 contracts, so the notional convention massively overweights exactly the extreme-price positions that the per-contract convention correctly treats as pennies. Reported here because a reader who reruns this with different sizing will get a different sign, and they should learn that from the note rather than by surprise.

### 4d · Every-qualifying-day — the declared secondary

| arm | positions | markets | deployed | PnL | ROI | win rate |
|---|---|---|---|---|---|---|
| first qualifying day (primary) | 19 | 19 | $5.41 | −$0.41 | −7.7% | 26.3% |
| every qualifying day (secondary) | 56 | 19 | $12.05 | +$1.95 | **+16.2%** | 25.0% |

**Read, and read it as a warning rather than as good news.** The secondary convention is positive where the primary is negative — but the two arms trade **the same 19 markets**, so this is not 56 independent bets. It is the same 19 opinions re-entered up to 8 times each, which concentrates the sample onto whichever markets happened to flag repeatedly. Crucially, **the sign of the every-day arm reverses once prior-only positions are excluded**: +$1.95 becomes **−$1.49 (ROI −23.0%)**, because 19 of the 56 positions are prior-only and they contribute +$3.44 of the +$1.95. In other words the entire positive result of the secondary arm comes from repeatedly re-entering markets the model had read nothing about. That is § 6's point arriving early.

### 4e · The confidence-gated subset — the full shipped divergence rule

The shipped flag is not gap alone. It is **gap ≥ 15pp AND band half ≤ 12pp AND ≥5 relevant articles in 72h** — the two extra conditions exist precisely to filter out low-confidence and low-evidence disagreements. Applying it:

| market | date | side | FV% | mid% | band | n_rel | entry | y | PnL |
|---|---|---|---|---|---|---|---|---|---|
| us-x-iran-permanent-peace-deal-by-may-31 | 2026-05-26 | NO | 1.3 | 27.5 | 12.0 | 8 | 0.735 | 0 | **+0.265** |
| bab-el-mandeb-strait-effectively-closed-by-april-30 | 2026-04-01 | YES | 39.5 | 17.5 | 12.0 | 5 | 0.185 | 0 | −0.193 |
| qatarenergy-announces/resumes-lng-production-in-qatar | 2026-04-09 | NO | 18.1 | 38.5 | 12.0 | 5 | 0.625 | 1 | −0.637 |
| us-x-iran-ceasefire-by-april-7 | 2026-03-25 | NO | 5.5 | 27.5 | 12.0 | 6 | 0.735 | 1 | −0.735 |
| **TOTAL** | | | | | | | **$2.30 deployed** | **1W/3L** | **−$1.30** |

**ROI −56.5%.** Zero prior-only positions, by construction — the `n_rel ≥ 5` condition excludes them.

One row deserves a note because it looks like a contradiction and is not. **`qatarenergy` appears in both tables on opposite sides.** In the ungated table its first qualifying day is 2026-04-05, where FV 23.1% > mid 5.0% and the rule buys YES at 6c — the sample's best position. Under the confidence gate that day is excluded (`n_rel = 2`, below the ≥5 bar), so the gated entry is the *next* day that clears all three conditions, 2026-04-09, by which point the mid has repriced to 38.5% and the FV has fallen to 18.1% — so the gate buys **NO at 62.5c** on a market that resolved YES. Same market, same model, four days apart, +$0.94 versus −$0.64. That is not a bug in either rule; it is the clearest single illustration in this note of how much a small entry-timing choice moves the result at this sample size.

**Read.** This is the uncomfortable one, and it should be reported as uncomfortable rather than explained away. The confidence gate is the mechanism we shipped to identify our *best* disagreements, and on this sample it selected a strictly worse subset than the ungated rule: **−56.5% against −7.7%**. Two of its four positions are `iran` — the same family the reconstruction already scores worst on (family Brier 0.396, per [[newsagent_simulated_live_findings]] § 3e) — so this is plausibly the known iran weakness showing up in a second statistic rather than an independent finding about the gate. **But four positions cannot distinguish those two stories, and this note does not claim to.** What it does establish is that there is no evidence here that the confidence gate improves position selection, which is worth knowing before anyone builds on the assumption that it does.

Note also that the gated rule finds only these 4 flagged snapshots against the **≥30 across ≥10 markets** that Q-DIV-EDGE requires. Even pooling retrospective and forward flags — which the discipline forbids — would not reach the floor. The floor is not a formality being waived here; it is roughly 8× away.

---

## 5 · Family breakout, with Fed called out

*One row per event family, on the 19 headline positions. **Singleton** = the family contributes exactly one market, so its row is a single bet and its ROI is not an estimate of anything.*

| family | positions | deployed | PnL | ROI | win rate | singleton? | prior-only positions |
|---|---|---|---|---|---|---|---|
| other:qatarenergy | 1 | $0.06 | **+0.937** | +1492% | 1.00 | **yes** | 0 |
| trump_admin | 1 | $0.45 | +0.548 | +121% | 1.00 | **yes** | **1 of 1** |
| ukraine_russia | 2 | $0.63 | +0.373 | +59% | 0.50 | no | 0 |
| israel | 2 | $0.80 | +0.199 | +25% | 0.50 | no | 1 of 2 |
| other:forces | 1 | $0.01 | −0.011 | −100% | 0.00 | **yes** | 0 |
| other:cuba | 1 | $0.01 | −0.011 | −100% | 0.00 | **yes** | 0 |
| other:epstein | 1 | $0.03 | −0.029 | −100% | 0.00 | **yes** | 0 |
| other:starmer | 1 | $0.16 | −0.162 | −100% | 0.00 | **yes** | 0 |
| other:mandeb | 1 | $0.19 | −0.193 | −100% | 0.00 | **yes** | 0 |
| **fed** | **3** | **$0.35** | **−0.349** | **−100%** | **0.00** | no | **2 of 3** |
| hungary | 1 | $0.39 | −0.387 | −100% | 0.00 | **yes** | 1 of 1 |
| iran | 3 | $1.61 | −0.605 | −38% | 0.33 | no | 0 |
| other:will | 1 | $0.73 | −0.725 | −100% | 0.00 | **yes** | 0 |

**Read — and note that 9 of 13 families are singletons.** A "+1492% ROI" on `other:qatarenergy` is one bet that paid, entered at 6c; a "−100%" on five other singletons is one bet that did not. None of those rows are estimates. Only three families contribute more than one position, and they are the three the reconstruction is already over-weighted in.

**Fed, since it was asked for specifically.** Three positions, **all three lost**, ROI −100% on $0.35 deployed. The mechanism is legible in the per-market table and is not flattering: two of the three Fed positions had **`n_rel = 0`** — no relevant articles at all — and in both the model was disagreeing with a market that was 77.5c and 98.5c confident of "no change", from a position of having read nothing about it. The market was right both times. The third Fed position (a 25bp-cut market, `n_rel = 2`) also lost. This is worth sitting with, because `fed` is the family the reconstruction scores *best* on by Brier (0.078, § 3e of [[newsagent_simulated_live_findings]]) — **being well-calibrated on a family and being able to trade against its mid are completely different things**, and this sample shows them coming apart. Three positions cannot establish that, but it is precisely the hypothesis this pass exists to generate.

**iran**, the other multi-market family, is 1W/2L for −$0.61 — the largest family loss in absolute dollars, driven by the two ceasefire/deal markets where the reconstruction stayed at 5–6% while the events actually happened. This is the *same* failure mode the original v0 gate identified in [[newsagent_v0_gate_findings]]: a status-quo-leaning model gets destroyed on YES-transitions. That the failure reappears here, three months and one model rebuild later, is arguably the most durable observation in this note.

---

## 6 · Prior-only positions — disagreeing from ignorance

A position is **prior-only** when its snapshot had **`n_rel = 0`**: zero relevant articles in the 72h window. The FV in that case is the onboarding prior, essentially untouched by evidence. When such a snapshot produces a 20–40pp gap against the mid, the model is not disagreeing with the market *about the news* — **it is disagreeing because it has not read any**, and the market has. That is a coin flip wearing the costume of a signal, and it is the single distinction that separates this from a real study.

| arm | positions | deployed | PnL | ROI |
|---|---|---|---|---|
| headline, all positions | 19 | $5.41 | −$0.41 | −7.7% |
| headline, **prior-only removed** | 14 | $3.52 | **−$0.52** | **−14.7%** |
| *(the 5 prior-only positions alone)* | 5 | $1.90 | +$0.10 | +5.4% |
| every-qualifying-day, all | 56 | $12.05 | +$1.95 | +16.2% |
| every-qualifying-day, **prior-only removed** | 37 | $6.49 | **−$1.49** | **−23.0%** |

**Read, and this is the most decision-relevant table in the note.** In the headline arm the effect is modest: removing the 5 prior-only positions moves −7.7% to −14.7%, i.e. the evidence-backed subset is *worse*, not better. In the every-qualifying-day arm the effect is total: **the entire +16.2% is prior-only positions, and the evidence-backed remainder is −23.0%.** Under both conventions, then, the positions where the model had actually read something did **worse** than the positions where it had read nothing. If the news channel were adding value to divergence selection, that ordering would run the other way. On 19 and 37 positions this is not proof of anything — but it is the opposite of the pattern a working signal would produce, and it is a specific, testable prediction for the forward study to overturn.

---

## 7 · Caveats — every one of these is load-bearing

1. **α was fitted on these very markets.** The FV series being traded here comes from a model whose single fitted parameter was calibrated on this sample ([[newsagent_simulated_live_findings]] § 4). Any PnL computed on it is an **upper bound** on what the same rule would return on markets the model had not seen. A negative in-sample result is therefore worse than it looks, not better.
2. **The mid is reconstructed and is not an executable price.** It comes from the reconstruction's price history, not from a book we could have hit. There is no depth, no queue, no partial fill, and no evidence that the size implied here was available at that price. The 1.0pp haircut is a gesture at this, not a solution to it.
3. **Markets cluster by family and positions are not independent.** The reconstruction carries iran ×7 and fed ×5 by declared deviation ([[newsagent_simulated_live_findings]] § 1a), and 9 of the 13 families here are singletons. **No confidence interval is computed and none may be quoted.** Every number in this note is a point read on a clustered sample.
4. **At n = 19, a handful of positions produce any headline you like.** Deleting the best market gives −$1.35; deleting the worst gives +$0.32; changing the position-sizing convention gives +$3.68. Three different defensible choices, three different signs. **This is why the per-market table in § 4b is mandatory and why the total is not the finding.**
5. **Prior-only positions are disagreement from ignorance** (§ 6), and under both entry conventions they outperformed the evidence-backed positions — which is the reverse of what a working signal implies.
6. **The evidence diet is poorer than live.** The reconstruction sees only Guardian, Wikipedia Current Events and GDELT-GKG; RSS, newsletters, macro PDFs and reach documents are live-only ([[newsagent_simulated_live_findings]] § 5). Whether a richer diet would generate better or merely *more* disagreements is untested in either direction.
7. **The reconstruction is systematically NO-leaning** (Spiegelhalter Z = +3.40, calibration rejected, § 3a of the same note). A NO-leaning model generates NO-side divergence positions preferentially — 11 of 19 here — so this sample's composition is partly a symptom of a known miscalibration rather than a neutral draw of disagreements.
8. **This is one snapshot grid.** Entries are taken on the reconstruction's 4-day grid; a different grid would produce different first-qualifying days and therefore different entry prices, on markets whose mids move fast.

---

## 8 · Verdict — is the real study worth running?

**On the trading question: no edge is visible, and no absence of edge is demonstrated.** Both halves matter. The declared rule returned −7.7%; the confidence-gated version returned −56.5% on four positions; the evidence-backed subset was worse than the prior-only subset. Nothing here supports trading our divergences. But nothing here **refutes** it either, because at 19 positions across 13 families — 9 of them singletons — with a sign that flips on deleting one market or changing the sizing convention, **this measurement has no power to distinguish a real edge from its absence.** Reporting it as "we tested it and it doesn't work" would be as dishonest as reporting the +18.8% variant as an edge. The correct summary is: **we looked, cheaply, and learned that we cannot tell.**

**On whether to run the real study when the sample exists: yes, but with two design changes this pass earned.**

1. **Q-DIV-EDGE should keep its floor and its forward-only discipline.** ≥30 resolved flagged snapshots across ≥10 markets, identified before outcomes are known, family-clustered bootstrap CI, lower bound > 0 net of haircut. This pass is not evidence for weakening any of that; if anything it is evidence for the floor, since it demonstrates concretely what a below-floor sample produces.
2. **Add a pre-registered `n_rel > 0` condition, or measure with and without it as a declared primary split.** § 6 is the one genuinely new thing here: the current shipped flag already requires `n_rel ≥ 5`, but the *ungated* divergence display does not, and the every-day arm's whole apparent profit came from zero-evidence disagreements. Whatever else Q-DIV-EDGE measures, it should report the evidence-backed and ignorance-driven positions separately, because pooling them is how a null gets dressed up as a signal.
3. **Pre-register the position-sizing convention explicitly.** § 2a and § 4c show that per-contract and per-dollar sizing disagree on the sign of this sample. Q-DIV-EDGE's registration currently says "mean per-contract PnL", which is the right convention and should be stated as a locked choice rather than an incidental phrasing.

**On what must not happen next.** The [[newsagent_v0_gate_findings]] closure is untouched by this note and stays closed. Nothing here goes on the showcase page, into the ledger, or into any summary of what the news agent can do. There is no dashboard change and no ledger write in this pass. If this note is ever cited, it must be cited as what its own title says it is: **exploratory, in-sample, and hypothesis-generating.**

**Concrete next action:** none, on this branch. Q-DIV-EDGE stays in the backlog behind its sample floor; the forward ledger needs roughly 8× its current flagged-and-resolved count before the real study is runnable. The news-agent thread remains **deprioritised** per [[TODO]], and this pass does not change that.

---

## 9 · Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions.** The 1.0pp spread haircut is a declared judgment, not a measured spread — no book data enters this pass. The fee schedule is the canonical `FEE_BY_CATEGORY` table applied through the canonical `family_category` mapping, both unchanged, but that mapping was written for the dali microstructure universe and sends most news-agent families to `Other` (5%) and the iran family to fee-free `Geopolitics`; the mapping is mechanical and declared, not tuned for this sample. Fees are charged at entry only. Entry prices are clipped to [0.01, 0.99]. The entry grid is the reconstruction's 4-day snapshot grid, inherited unchanged.

**Live-only unknowns.** Whether any of these prices were executable in any size — the single largest gap between this note and a tradeable claim. Whether the divergence flag fires at a usable rate forward. Whether the § 6 ordering (evidence-backed worse than prior-only) survives a real sample or is noise. Whether the confidence gate's poor showing here is the known iran weakness or a defect in the gate.

**Power honesty.** **n = 19 positions, 19 markets, 13 families, 9 of them singletons; the gated subset is n = 4.** No confidence interval is computed and none may be quoted. The sign of the headline is not robust to deleting one market or to the position-sizing convention. This is a point read on a clustered, in-sample, below-floor sample, and it is labelled exploratory everywhere it appears for that reason.

---

## 10 · Outputs

- **Script:** `scripts/newsagent_divergence_pnl.py` — pre-registration P-1…P-7 in the module docstring, the declared rule, all four cost/sizing variants, the gated subset, the family breakout, the declared sensitivity grid, and the chart. One invocation reproduces every number in this note: `PYTHONPATH=. uv run python scripts/newsagent_divergence_pnl.py`.
- **CSV:** `data/analysis/csv_outputs/news_agent/newsagent_divergence_pnl_positions.csv` — the 19 headline positions, one row each.
- **JSON:** `data/analysis/csv_outputs/news_agent/newsagent_divergence_pnl_summary.json` — all arms, families, positions and the sensitivity grid.
- **Plot:** `data/analysis/plots/news_agent/newsagent_divergence_pnl.png` — per-market PnL bars, prior-only positions coloured separately.
- **Pre-registration:** locked to the session scratchpad before any computation, reproduced verbatim in § 2 with the single implementation correction disclosed in § 2a.
- **Not touched:** the dashboard, the showcase page, the forward ledger, `fv_params.json`, and the [[newsagent_v0_gate_findings]] closure.

---

## Appendix A · Declared sensitivity grid — context only, never the headline

> **This table is not a result.** It exists to show that the declared T = 15pp was not selected by looking at returns, and to make the instability visible. **Reading the best cell out of it and quoting it would be exactly the post-hoc selection P-6 forbids.**

| T | entry rule | positions | markets | deployed | PnL | ROI | win rate |
|---|---|---|---|---|---|---|---|
| 5pp | first | 28 | 28 | $7.76 | +$0.24 | +3.1% | 28.6% |
| 5pp | every | 105 | 28 | $25.72 | +$7.28 | +28.3% | 31.4% |
| 10pp | first | 24 | 24 | $7.13 | +$1.87 | +26.2% | 37.5% |
| 10pp | every | 78 | 24 | $20.94 | +$5.06 | +24.2% | 33.3% |
| **15pp** | **first (declared headline)** | **19** | **19** | **$5.41** | **−$0.41** | **−7.7%** | **26.3%** |
| 15pp | every | 56 | 19 | $12.05 | +$1.95 | +16.2% | 25.0% |
| 20pp | first | 17 | 17 | $4.29 | −$1.29 | −30.0% | 17.6% |
| 20pp | every | 47 | 17 | $8.67 | +$1.33 | +15.4% | 21.3% |
| 25pp | first | 14 | 14 | $2.64 | +$0.36 | +13.6% | 21.4% |
| 25pp | every | 34 | 14 | $4.72 | +$3.29 | +69.7% | 23.5% |
| 30pp | first | 11 | 11 | $1.26 | +$0.74 | +58.7% | 18.2% |
| 30pp | every | 23 | 11 | $2.12 | +$2.89 | +136.4% | 21.7% |

**Read.** The declared cell is the *only negative first-day cell in the grid*, which is precisely the kind of coincidence that would tempt a post-hoc analyst to move the threshold — and precisely why the threshold was declared in advance. The column has no monotone structure: ROI goes +3.1 → +26.2 → **−7.7** → −30.0 → +13.6 → +58.7 as T tightens, on samples shrinking from 28 positions to 11. The apparently spectacular T=30 cells are 11 and 23 positions on markets priced at 1–10c, where one lottery ticket paying off produces a triple-digit percentage. **The honest reading of this grid is that it is noise across the board**, and that its noisiness is a stronger argument for the Q-DIV-EDGE sample floor than any single cell in it.
