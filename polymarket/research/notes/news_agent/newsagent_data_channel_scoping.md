---
title: "Data-evidence channel scoping — recompiling official statistics into Stage-B evidence for the Observatory's four structurally-blind markets (OpenBB free tier, $0)"
created: 2026-08-24
status: scoping complete + SIGNED OFF 2026-08-24 (all 12 rows decided — see § 7); the pre-registered dry run then FAILED (NO-GO, [[newsagent_data_channel_dryrun_findings]]) — the channel stays unbuilt and OpenBB stays uninstalled. Install verified but still NOT installed; free-consensus check FAILED its pre-registered coverage bar; TradingEconomics DECLINED; § 4f locked as the standing pre-registration; the only rows APPLIED so far are the tag split (row 7) and the .env key loading (rows 3-4). The channel itself remains unbuilt.
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
  - data-channel
  - scoping
---
# Data-evidence channel scoping — can official statistics score the markets our news packet cannot see?

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]] · Table terms: [[polymarket_table_dictionary]]
> Builds on [[newsagent_observatory_v31_findings]] (tractability tags) and [[newsagent_observatory_v32_findings]] (current model + params). This is a **SCOPING pass, not a build**: no change to `fvmodel.py`, `config.py`, the dashboard, the ledger, or any published page. The closed "our % beats the mid" gates ([[newsagent_v0_gate_findings]]) stay closed.

## Plain-English Summary

- **The problem.** The Calibration Observatory scores 24 Polymarket questions with a news-driven fair value. Four of them are tagged **data-driven** in `newsagent/config.py` → `DATA_DRIVEN`, and their public cards admit "not news-tractable — structurally blind": two Fed rate-decision markets, the Becerra California-governor market, and the California billionaire-wealth-tax market. This note asks whether a **second evidence channel built from official statistics** (via OpenBB used strictly as a library, free tier, $0) could give those four an honest number.
- **Install verdict: PASSES, cleanly — but nothing was installed.** The lean set (`openbb-core` + `economy` + `fred` + `bls` + `nasdaq`) resolves against the full research stack on Python 3.14 with **26 added packages and zero version changes to any existing pin** — pandas stays 3.0.5, numpy stays 2.5.2. A real install into a throwaway venv imports and makes live calls. Adding the free, keyless `openbb-federal-reserve` (FOMC documents + projections) costs 4 more packages and still changes no pin. Everything OpenBB ships is **AGPL-3.0-only** — fine for our offline use, with one boundary rule below.
- **The channel only covers two of the four markets.** The two Fed markets map onto real official statistics. **Becerra and the CA wealth tax do not** — those move on private/campaign and issue polling, which is not an official statistic and is not in OpenBB at any price. The v3.1 tag lumps two different kinds of blindness together; the honest fix is to split the tag into `data-driven` and `poll-driven`.
- **The pre-registered consensus check FAILED — and the failure is coverage, not accuracy.** Where the free Nasdaq calendar prints a consensus it is good: it beats "last month's print repeats" on 6/6 series, its `actual` matches the never-revised official CPI index **17/17 exactly**, and its CPI error (RMSE 0.13pp) is on par with the Cleveland Fed nowcast (0.16pp). But it simply **omits half the releases** — 6 of 12 CPI releases and 8 of 12 employment reports are absent from the feed entirely — against a pre-registered 0.80 coverage bar.
- **The recommendation costs nothing.** Do **not** buy TradingEconomics. Anchor the channel on the **Cleveland Fed nowcast** (free, no key, every month, with daily point-in-time vintages back to 2013) rather than on consensus, and keep the free consensus as a cross-check when it appears. The two free sources the channel actually needs — Cleveland Fed nowcasts and the Atlanta Fed **Market Probability Tracker** (daily fed-path probabilities since 2023) — are both complete and both carry vintage history, which is what a lookahead-free refit needs.
- **One-line status (updated 2026-08-24):** design and pre-registration written and now **SIGNED OFF** — all twelve rows of [§ 7](#7--sign-off-table-for-justin--answered-2026-08-24) are decided, § 4f is LOCKED as the standing pre-registration, TradingEconomics is DECLINED, and the tag split + credential loading + per-method ledger labels are **applied and test-enforced**. The channel itself is still **unbuilt** and OpenBB is still **not installed** (approved only for when the build starts); no `p_struct`, no fitted parameter, no Fed card copy change.

## What this pass is and is not

| | |
|---|---|
| **Is** | An install/dependency check, a market→source mapping, one pre-registered measurement of free consensus quality, and a written design + pre-registration for how objective data would become Stage-B evidence. |
| **Is not** | Any change to `fvmodel.py`, `newsagent/config.py`, the dashboard, `fv_params.json`, the sf ledger, or the published page. No parameter was fitted. OpenBB was **not** installed into the research venv. |
| **New files** | `scripts/newsagent_data_channel_scoping.py` (reproduces every number here), its CSV/plot outputs under the git-ignored `data/` tree, and this note. |

---

## 1 · Install verdict — lean OpenBB set, Python 3.14, $0

**Method.** Two independent checks, neither of which mutated the research venv:

1. **Resolution against the real stack.** `uv pip compile` on the research `pyproject.toml` dependency list (15 runtime + 4 dev) with and without the OpenBB packages, at `--python-version 3.14`, then a diff of the two lockfiles.
2. **A real install + live calls.** A throwaway `uv venv -p 3.14`, `uv pip install` of the lean set, then imports and actual network calls.

**Result — resolution is clean.**

| Check | Result |
|---|---|
| Python floor | every OpenBB package declares `requires-python <4,>=3.10` → **3.14 is in range** (the repo's `requires-python >=3.14` was the main a-priori risk; it is a non-issue) |
| Packages added by the lean set | **26** (`openbb-core` 1.6.13, `openbb-economy` 1.6.2, `openbb-fred` 1.6.2, `openbb-bls` 1.3.1, `openbb-nasdaq` 1.6.3 + transitive: fastapi, starlette, uvicorn, pyjwt/uuid7/cachebox/deepdiff/ruff/nasdaq-data-link/…) |
| Version changes to existing pins | **NONE** — pandas 3.0.5, numpy 2.5.2, aiohttp 3.14.3, httpx 0.28.1, requests 2.34.2 identical with and without OpenBB |
| Packages dropped | none |
| With `openbb-federal-reserve` added | 30 added, still **zero** version changes to existing pins |
| Real install (throwaway py3.14 venv) | succeeded; `openbb_core` imports in ~2.4 s; the static `openbb` namespace builds on first import; `obb.economy.calendar(provider="nasdaq")` returned 164 rows and `obb.economy.fomc_documents(year=2026)` returned 29 documents |

**Column read:** "packages added" counts wheels that appear in the resolved lockfile *only* when OpenBB is included; "version changes" counts packages already in the research stack whose resolved version moves because of OpenBB. The second row is the one that matters — a zero there means the lean set is additive and cannot silently re-pin the existing stack.

**Do NOT install `openbb[all]`.** The meta-package pulls ~40 providers including `openbb-tradingeconomics` (paid), charting (plotly stack) and econometrics (statsmodels/arch). Nothing here needs them.

### Keys — exactly what is needed, and what is not

| Env var | Needed for | Free? | Verified |
|---|---|---|---|
| `OPENBB_FRED_API_KEY` | every `openbb-fred` call | free signup at fredaccount.stlouisfed.org | **required** — the provider raises `Missing credential 'fred_api_key'` without it |
| `OPENBB_BLS_API_KEY` | every `openbb-bls` call | free signup (BLS v2 registration) | **required by OpenBB**, though the BLS API itself answers **without** a key (verified live on v1 *and* v2); registering raises the quota from 25 to 500 queries/day |
| `OPENBB_NASDAQ_API_KEY` | Nasdaq Data Link **datasets** | free tier exists | **NOT required** for the economic calendar — every calendar call in this note ran keyless |
| *(none)* | `openbb-federal-reserve` (FOMC documents, EFFR, target range) | — | provider declares `credentials=[]`; verified keyless |

Per repo convention these belong in a git-ignored `.env` / `secrets/` load, alongside the existing `GUARDIAN_API_KEY` / `GEMINI_API_KEY` items already waiting in [[newsagent_observatory_v32_findings]]. **Flagged for Justin — nothing was registered on his behalf.**

There is also a **keyless fallback for FRED**: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>` returns the full history with no key (this note's realized-value cross-checks all ran through it). It is undocumented-but-stable; treat the API key as the supported path and this as the degrade path.

### License — AGPL-3.0-only, and the one boundary rule

Every `openbb-*` package is **AGPL-3.0-only**. Reasoning for our use, stated plainly so Justin can overrule it:

- We **use** OpenBB as an unmodified library, offline, inside a research repo. We do not distribute it, and the AGPL's §13 network clause is triggered by *users interacting with the program remotely over a network* — the Observatory publishes a **static HTML page generated locally**; no reader ever interacts with OpenBB.
- The residual risk is not the dashboard, it is **linking**: if the `newsagent` package itself imported `openbb` *and* that package later became a network service, the combined work could be argued into AGPL copyleft.
- **Boundary rule (recommended, cheap):** keep OpenBB in an **offline ingest script** that writes a plain JSON/Parquet snapshot; `newsagent/*` reads the snapshot and never imports `openbb`. This also keeps the live daily loop independent of a 26-package dependency and matches how GDELT/BigQuery is already isolated in `gdelt_bq.py`. **Flagged for Justin's awareness — not a lawyer's opinion.**

---

## 2 · Market → data mapping

The four markets carrying `DATA_DRIVEN` in `newsagent/config.py`, with live status pulled from Gamma on 2026-08-24:

| Market (slug) | Status today | Dominant information channel | Objective-source mapping | Verdict |
|---|---|---|---|---|
| `…fed-interest-rates-after-the-july-2026-meeting` | **RESOLVED YES** 2026-07-29 ($40.5M volume) — the ledger settlement is still pending per [[TODO]] | policy decision | full mapping below | **mappable** (now a scoring case, not a forecasting one) |
| `…fed-interest-rates-after-the-september-2026-meeting-615` | **LIVE**, mid **67.5%** — meeting **Sept 15–16** (an SEP meeting) | policy decision | full mapping below | **mappable** |
| `will-xavier-becerra-win-the-california-governor-election-in-2026` | **LIVE**, mid **95.2%** | general-election polling | none — see below | **not mappable** |
| `billionaire-one-time-wealth-tax-passes-in-california-election-2026` | **LIVE**, mid **28.5%** | ballot-measure issue polling | none — see below | **not mappable** |

### 2a · The Fed markets — the full source table

Every row was fetched live during this pass unless marked. "Vintage history?" is the column that decides whether a source can be used in a lookahead-free historical fit, which is the repo's binding invariant (`brain/CODEX.md` § Key invariants).

| Object | Concrete source | Access | Vintage history? | Role in the channel |
|---|---|---|---|---|
| Current policy state | FRED `DFEDTARU` / `DFEDTARL` (**3.75 / 3.50** on 2026-08-22); EFFR 3.63% via `openbb-federal-reserve` | FRED key (or keyless CSV); federal_reserve keyless | n/a (not revised) | defines what "no change" means for the resolution |
| Headline CPI | FRED `CPIAUCSL` (SA) / `CPIAUCNS` (NSA) = BLS `CUSR0000SA0` / `CUUR0000SA0` | FRED or BLS key | NSA never revised; SA revised annually | inflation state |
| Core CPI | FRED `CPILFESL` / `CPILFENS` = BLS `CUSR0000SA0L1E` / `CUUR0000SA0L1E` | same | same | inflation state |
| PCE / core PCE | FRED `PCEPI` / `PCEPILFE` | FRED | revised | the Fed's target measure |
| Labour | FRED `PAYEMS`, `UNRATE` = BLS `CES0000000001`, `LNS14000000` | FRED or BLS | **heavily revised** | dual-mandate side |
| **CPI / core-CPI / PCE / core-PCE nowcast** | **Cleveland Fed inflation nowcasting public JSON** (`nowcast_month.json`) | **keyless, free** | **YES — daily vintages, 158 target months (2013-07 → 2026-08), plus the realized actual per month** | **recommended anchor for the pre-release print distribution** |
| GDP nowcast | FRED `GDPNOW` (Atlanta Fed GDPNow; also appears as an "Atlanta Fed GDPNow" calendar event) | FRED | **NO** — FRED carries the latest estimate per quarter; point-in-time needs ALFRED vintages | secondary macro state |
| **Fed-path probabilities (market-implied)** | **Atlanta Fed Market Probability Tracker** `mpt_histdata.xlsx` — 297,277 rows, daily **2023-03-29 → 2026-08-20**, from CME 3-month SOFR options; fields `Prob: cut`, `Prob: hike`, rate mean/mode/percentiles per reference window (windows land on FOMC/IMM dates incl. **2026-09-16**) | free download, no key | **YES — daily** | **display-only benchmark, never an FV input** (see § 4c) |
| Fed-path probabilities (survey) | NY Fed **Survey of Primary Dealers** results (PDF/XLSX per meeting, back to 2011) | free | yes, per meeting | optional second benchmark; survey-based, published ~2 weeks pre-meeting |
| FOMC projections + documents | `obb.economy.fomc_documents(year=…, provider="federal_reserve")` — 29 docs for 2026, `doc_type` ∈ {monetary_policy, minutes, projections, beige_book, press_conference}; **projections** published 2026-03-18 and 2026-06-17, next at the **Sept 15–16** meeting | keyless | yes (dated docs) | the Fed's own forward guidance; Sept is an SEP meeting |
| SEP medians as data | FRED `FEDTARMD` (median projected rate), `FEDTARRH` / `FEDTARRL` (central tendency), `FEDTARMDLR` (**3.1** longer-run, as of 2026-06-17) | FRED | quarterly, dated | the Fed's own path vs the current range |
| Release calendar + consensus | Nasdaq economic calendar via `openbb-nasdaq` | **keyless** | no (see § 3 caveat) | scheduling + a consensus cross-check — **with the coverage hole measured in § 3** |
| Release *schedule* fallback | FRED releases-dates endpoint (FRED key). BLS's own schedule pages return **HTTP 403** to automated fetches | FRED key | n/a | needed because the free calendar omits releases (§ 3) |

**Practical example — what the channel would actually see between now and the September FOMC.** Pulling the free calendar forward from 2026-08-25 to the Sept 15–16 meeting returns July core PCE (Aug 27, consensus +0.2% m/m, 3.3% y/y) and August PPI (Sept 11) — **and neither the August CPI nor the August employment report**, the two releases that most move a rate decision. They are missing from the *calendar*, not from the world: the prints themselves will land in FRED/BLS on schedule, and the Cleveland Fed nowcast for August CPI updates *daily* in the meantime. This is the concrete reason § 4 anchors on the nowcast and treats the calendar as a scheduling convenience rather than a dependency.

### 2b · The two California markets — the tag is doing double duty

`DATA_DRIVEN` currently means "the dominant channel is not news text". That covers two genuinely different situations:

- **Fed ×2 — official statistics exist**, are free, complete, vintage-stamped, and public. A data channel is buildable.
- **Becerra (mid 95.2%) and the CA wealth tax (mid 28.5%) — the dominant channel is polling**, which is *not* an official statistic. What California publishes officially — the Secretary of State's Report of Registration, the certified ballot-measure list, the Legislative Analyst's Office fiscal analysis — is objective but far too slow and far too weakly linked to the outcome to move a probability. OpenBB has nothing for either at any tier, and no free-tier provider does.

**APPLIED 2026-08-24** (was: proposed): the tag is now **split** in `newsagent/config.py` into `DATA_DRIVEN` (official statistics → the Fed market) and `POLL_DRIVEN` (private/issue polling → Becerra, CA wealth tax), with `tract()` returning `news | data | poll` and the dashboard rendering a `◆ data-driven` or `◆ poll-driven` chip with type-specific card copy. The July Fed market left the slate the same day (resolved YES 2026-07-29, settled), so `DATA_DRIVEN` currently holds the **September** Fed market alone. One further wrinkle, **also fixed on 2026-08-24**: v3.1's stated reason for the Becerra tag — "state-primary races move on private/campaign polling" — was stale. The June primary has passed and the market sits at 95.2%; the blindness is real but its *cause* is now **general-election** polling, and the card note now says so (test-enforced: `test_becerra_note_is_refreshed_to_general_election`).

---

## 3 · Consensus accuracy check (pre-registered, then run)

### 3a · Pre-registration — locked before any comparison was computed

Locked 2026-08-24 to `scratchpad/PREREG_consensus_accuracy.md` **before** a single consensus-vs-actual number existed; reproduced verbatim in substance:

> **Question.** Is the FREE economic-calendar consensus (Nasdaq via `openbb-nasdaq`, no API key) accurate and complete enough to anchor the CENTER of a predictive distribution over an upcoming macro print, so that a data-evidence channel can be built without paying TradingEconomics?
>
> **Sample.** Six US releases mapped to the Fed markets — S1 CPI MoM (`CPIAUCSL`), S2 Core CPI MoM (`CPILFESL`), S3 Nonfarm Payrolls (`PAYEMS`), S4 Unemployment Rate (`UNRATE`), S5 Core PCE MoM (`PCEPILFE`), S6 Retail Sales MoM (`RSAFS`). PRIMARY window = the 12 most recent releases per series in [2025-08-24, 2026-08-24]; a pre-registered robustness EXTENSION repeats everything on [2024-08-24, 2026-08-24] (≤24 releases), reported separately and never substituted. Official realized values from the keyless FRED CSV; nowcast = the Cleveland Fed vintage published **strictly before** the release date.
>
> **Metrics.** **M1 Coverage** = share of *expected* releases carrying a consensus, **bar ≥ 0.80**. **M2 Integrity** = share where the calendar's `actual` matches the FRED official value within tolerance (0.05pp percent-type, 20k payrolls), **bar ≥ 0.90**. **M3 Bias** = mean surprise (actual − consensus), **bar |mean| ≤ 0.25 × RMSE**. **M4 Dispersion** = RMSE of the surprise in native units — the number the design consumes as the σ of the print distribution; no bar. **M5 Skill vs naive** = RMSE(consensus) vs RMSE(previous print as the forecast), **bar: consensus strictly better on ≥ 4/6 series AND on CPI and Core CPI**. **M6 Skill vs nowcast** (S1, S2 only) = RMSE(consensus) vs RMSE(pre-release Cleveland nowcast); no bar — it decides which object anchors the center.
>
> **Verdict rule.** PASS iff M1 ≥ 0.80 AND M2 ≥ 0.90 AND M3 holds on ≥ 4/6 AND M5 holds. CONDITIONAL PASS if M1/M2 pass but the nowcast beats consensus on CPI/Core CPI → the nowcast anchors, consensus becomes the cross-check. FAIL → escalate to Justin as a paid-provider (TradingEconomics) cost decision.
>
> **Anti-post-hoc clauses.** No series or date is dropped after seeing results except for a documented availability reason, and every drop is reported with its count. Event-name → series matching is fixed from the calendar's own fields BEFORE metrics run. Un-computable metrics are reported N/A, never replaced. The retrospectively retrieved consensus is ASSUMED to be the pre-release value — not vintage-verifiable, recorded as a modeled assumption.

### 3b · Results

Unit of observation: one **release** (a series × its reference month). `n_expected` counts reference months in the window for which FRED carries an official print — i.e. months a release genuinely happened, so the late-2025 CPI/jobs suspension is not counted against the feed. `n_found` counts releases present in the free calendar at all. All errors are in the print's native units (pp, except payrolls in thousands).

**Primary window (12 months to 2026-08-24):**

| Series | n_expected | n_found | M1 coverage | M2 integrity | M3 bias | M4 RMSE consensus | M5 RMSE naive | M6 RMSE nowcast | M3 pass | M5 pass |
|---|---|---|---|---|---|---|---|---|---|---|
| CPI MoM | 12 | 6 | **0.50** | 0.83 | −0.033 | **0.129** | 0.449 | 0.159 | no | **yes** |
| Core CPI MoM | 12 | 6 | **0.50** | 1.00 | −0.033 | **0.100** | 0.168 | 0.124 | no | **yes** |
| Nonfarm Payrolls | 12 | 4 | **0.25** | 0.00 | +24.3k | 62.5k | 134.5k | — | yes | **yes** |
| Unemployment Rate | 12 | 4 | **0.33** | 0.50 | 0.000 | 0.100 | 0.132 | — | yes | **yes** |
| Core PCE MoM | 12 | 5 | **0.42** | 0.80 | −0.040 | **0.063** | 0.110 | — | no | **yes** |
| Retail Sales MoM | 12 | 10 | **0.83** | 0.50 | +0.060 | 0.253 | 0.679 | — | yes | **yes** |

**Robustness extension (24 months), same metrics:** coverage 0.63 / 0.63 / 0.16 / 0.21 / 0.38 / 0.72; CPI RMSE 0.107 vs nowcast 0.124; Core CPI 0.103 vs nowcast **0.097** (the one cell where the nowcast wins); M5 holds on 5/6 (Unemployment Rate ties its naive benchmark).

**Two supplementary reads (labelled supplementary — not part of the pre-registered verdict):**

- **Never-revised integrity: 17/17 exact, max |diff| 0.005.** Comparing the feed's `actual` for the *NSA CPI index level* — a number BLS never revises — against FRED `CPIAUCNS` matches on every release. So M2's low scores are **revisions, not feed errors**: payrolls (0.00) and retail sales (0.50) score badly because their first prints are genuinely revised afterwards, and SA CPI moves with annual seasonal-factor revisions.
- **Where the coverage goes:** the shortfall is entirely *absence*, not silence. In the primary window CPI is absent 6/12 times, payrolls 8/12, core PCE 7/12, retail sales 2/12 — and when a release **is** in the feed it carries a consensus in 34 of 35 cases.

![Coverage vs accuracy for the free consensus](../../data/analysis/plots/news_agent/newsagent_datachannel_consensus_check.png)

**Chart read.** *Panel A* — grey is every release that actually happened (FRED-confirmed); terracotta is what the free feed carried with a consensus; the dashed line is the pre-registered 0.80 bar. Only retail sales gets near it. *Panel B* — RMSE of three candidate anchors for the same releases: the consensus (terracotta) beats the naive "last print repeats" benchmark (grey) everywhere, and sits within ~20% of the Cleveland Fed nowcast (dark) on CPI and core CPI. The two panels together are the whole finding: **accuracy is fine, presence is not.**

### 3c · Verdict

**FAIL against the pre-registered rule** — M1 misses the 0.80 bar on 6/6 series in the primary window (0.25–0.83), M2 misses the 0.90 bar on 5/6 (core CPI alone passes, at 1.00), and M3 holds on only 3/6. M5 passes cleanly (consensus beats naive 6/6), and M6 shows consensus and nowcast are near-equivalent in accuracy.

The pre-registered FAIL action is "escalate to Justin as a paid-provider cost decision". **My recommendation is to decline that escalation**, and this is explicitly a *post-hoc* redesign, offered as such rather than a moved goalpost:

1. What failed is the **consensus feed's completeness**, not the objective-data path. FRED/BLS/Cleveland/Atlanta-Fed coverage of the underlying statistics is 100%.
2. The design in § 4 does not actually need a consensus. It needs a **center for the pre-release distribution**, and the Cleveland Fed nowcast supplies one that is (a) statistically comparable to consensus (0.159 vs 0.129 on CPI; 0.124 vs 0.100 on core; the nowcast *wins* core CPI over 24 months), (b) published **every month**, and (c) **vintage-stamped daily**, which consensus is not — a decisive advantage for a lookahead-free refit.
3. TradingEconomics would buy back the ~50% missing calendar rows and nothing else. That is not worth money **before** the channel has demonstrated it can score a single market.

**Power honesty.** The primary window gives n=6 scored CPI pairs and n=4–10 elsewhere; an RMSE on n=6 is a point read with wide error bars, and no confidence interval is claimed here ([[CODEX]] § Anti-patterns: no CI-free "positive" results — accordingly the *only* claim being made from these numbers is the coverage failure, which is a count, not an estimate). The 24-month extension roughly doubles n and moves nothing qualitatively.

---

## 4 · Evidence-transform design (design only — nothing implemented)

### 4a · What Stage B is today, in one line

`FV_t = sigmoid( logit(p0) + α·A_t )` with `A_t = λ^Δdays·A_{t-1} + S_t`, where `p0` is the onboarding prior (five-perspective ensemble, set once), `S_t` is the day's news-evidence score from Stage-A article features × source weights, and **α = 2.7** is the single fitted parameter (355 pairs / 49 markets, [[newsagent_observatory_v32_findings]]). λ, floors, γ, `s_min`, `shift_clip`, `band_mult` and the band-Q constants are **declared**, not fitted.

### 4b · Three ways objective data could enter, and why one wins

| Option | Mechanism | Why not / why yes |
|---|---|---|
| **A — data as a synthetic article** | Turn each release/nowcast into a contribution `c_data` that enters `S_t` like an article | Cheapest to code and reuses the breakdown display, but it forces a *structural probability* onto a heuristic evidence scale, and it silently re-calibrates **α** — a knob fitted purely on news evidence would start absorbing a second, differently-scaled channel. Rejected. |
| **B — a separate structural FV replaces the news FV** | Data-driven markets get their own model and their own displayed number | Honest and clean, but it forks the model, the band, and the calibration track for 2 of 24 markets, and throws away the news evidence entirely (a Fed market *does* respond to Powell headlines). Rejected as the primary. |
| **C — structural probability as a re-anchorable prior (RECOMMENDED)** | `p0` for a data-driven market becomes `p_struct(t)`, recomputed when new objective data lands; news evidence keeps flowing through the unchanged `α·A_t` path | Uses the existing machinery exactly as built — the model already permits re-anchoring the prior "with a documented trigger", and this makes the trigger explicit and mechanical. **α, λ, band, ledger and dashboard code paths are untouched.** The cost is a double-count guard (§ 4d) and per-market method labelling (§ 4e). |

**Practical example (Fed September, as of 2026-08-24).** Onboarding prior aside, the channel would compute `p_struct` from: current range 3.50–3.75%, core CPI 2.5% y/y and headline 3.4% y/y, the Cleveland nowcast for August CPI updating daily, unemployment 4.2%, and the June SEP median path — then re-anchor on Aug-27 core PCE, on the August CPI print in early September, and on the SEP released at the meeting itself. Powell/Fed-speaker headlines continue to move `A_t` exactly as they do today. The card would show `p_struct`, the evolved FV, the PM mid (67.5%), and — clearly separated — the Atlanta Fed market-implied `P(hold) ≈ 42%` as a benchmark.

### 4c · The independence hazard — state it before building anything

The most informative single object for a Fed market is the **market-implied** probability (MPT / fed-funds-futures). Wiring it into `p_struct` would make the Observatory's "independent fair value" a **restatement of another market's price** — and since the Polymarket Fed mid is itself largely futures-driven, our FV would converge on the mid *by construction*, producing a good Brier for a reason that is not forecasting skill. That is precisely the claim the v0/v0b gates closed ([[newsagent_v0_gate_findings]]).

**Design rule (recommended, binding if adopted): market-implied probabilities are DISPLAY-ONLY.** They render next to the mid as a labelled benchmark, exactly as the PM mid does today, and never enter `p_struct`, `A_t`, or any fit. `p_struct` is built only from **published statistics, nowcasts, and the Fed's own projections** — objects that are not prediction-market prices.

The mapping is also not exact and the note should say so on the page: MPT's Sept-16 reference window is a **3-month SOFR window**, so its `P(hold)` (42.2% on 2026-08-20) is a **lower bound** on "no change at the September meeting" — a hold in September followed by an October or December hike still lands outside the range. That is most of the distance between 42% and the market's 67.5%, and a reader deserves to be told that rather than left to infer a 25-point disagreement.

### 4d · The double-count guard

The news packet already contains articles *about* the data ("CPI came in at 0.1%"). Once the print itself moves `p_struct`, the same information would move `A_t` again. Recommended: a **declared** Stage-A exclusion — articles whose extracted content is a *report of a release already ingested by the data channel* contribute 0 to `S_t` for that market, implemented as a rule over the existing feature fields (not a new LLM call), with the excluded items still displayed in the evidence feed and labelled "already counted in the data channel". Declared, not fitted; test-enforced.

### 4e · Public honesty during the transition

- **Card note.** For the two Fed markets, "◆ not news-tractable — structurally blind" becomes **"◆ data-channel scored — this number is built from official statistics (CPI/PCE/labour + Cleveland Fed nowcast + FOMC projections); market-implied odds shown for context, never used as an input."** For Becerra and the CA wealth tax, the existing structurally-blind note **stays exactly as it is** — nothing about this pass improves them.
- **Never merge the track records.** The sf ledger must record a `method` per snapshot (`news` vs `news+data`), the calibration page must show the two tracks separately with the switch date marked, and until the data channel has resolved forecasts of its own the card must say **"new method — no settled track record yet (n=0)"** while the historical news-only Brier stays visible and attributed to the old method.
- **The July Fed market is the first honest test bed.** It resolved YES on 2026-07-29 and is still awaiting `sf settle`. Scoring the *proposed* structural method retrospectively on it — with vintage-stamped inputs only — is a legitimate, cheap dry run precisely because both the Cleveland nowcast and the MPT carry point-in-time history.

### 4f · Pre-registration for the data channel — LOCKED 2026-08-24

> **DRY-RUN RESULT (2026-08-24): the declared DC-3 reaction function FAILED its pre-registered bars on the resolved July-2026 Fed market — NO-GO.** `p_struct` scored Brier 0.4223 against the news-FV's 0.2144 (outcome YES); directional sanity (0 violations) and stability (0.174 max daily logit move) both passed. The binding defect is per-meeting allocation, not the inputs: the rule divides the desired-rate gap by the remaining meetings and charges the quotient to the *next* one, while the rates market priced the same June-SEP shift into the September–December window. Three amendments are proposed in [[newsagent_data_channel_dryrun_findings]] § 9 and **must be locked here before any re-run**. DC-1…DC-8 below stand unchanged in the meantime.
>
> **STATUS: LOCKED.** Justin approved rows 8-11 of § 7 on **2026-08-24**, which makes DC-1 … DC-8 below the **standing pre-registration** for the data-evidence channel. From that date they are no longer a proposal: any build of this channel is bound by them, DC-5 carries its declared value (blend weight **1.0**, full replacement), DC-3's reaction function ships in **declared** form first, and changing any of it requires a dated amendment recorded in this section BEFORE the change (the [[newsagent_v0_gate_findings]] Amendment-1/2 precedent), never after seeing a result.

Written unfitted, so no knob can be chosen after seeing an outcome. Nothing below has been run.

> **DC-1 · Object.** For a data-driven market, `p_struct(t) = P(resolution criterion | objective information available at t)`, computed from a declared transform, recomputed only when a mapped source publishes.
> **DC-2 · Fed-market transform.** A predictive distribution over each mapped print (center = the Cleveland Fed nowcast vintage as of t where one exists, else the previous print; σ = the trailing surprise RMSE measured in § 3, refreshed on a declared schedule), mapped to `P(no change)` by a **reaction-function** step.
> **DC-3 · Reaction function — status and fitting procedure.** DECLARED at first (an explicit, inspectable rule over the inflation gap, the labour gap and the SEP median), never silently fitted. If it is ever fitted: labels = realized FOMC decisions from `DFEDTARU` changes; features = macro state as of the day before each meeting, vintage-stamped only; sample = FOMC meetings excluding ZLB periods; split = time-ordered with the final 20% held out; metric = Brier; **bar = beats the base rate of "no change" out-of-sample**; on failure the channel is display-only and does not replace any FV.
> **DC-4 · α is untouched.** The data channel enters through `p0`, never through `A_t`. Should that ever change, α must be refit through the existing path (`scripts/newsagent_hist_backfill.py --fit`) and the refit reported — no silent α changes ([[newsagent_observatory_v31_findings]] discipline).
> **DC-5 · Blend weight.** If `p_struct` is blended with the onboarding prior rather than replacing it, the weight is **DECLARED at 1.0** (full replacement) and listed for sign-off, exactly as γ was in v2.1 — n is far too small to fit it.
> **DC-6 · Market-implied inputs are display-only** (§ 4c), including the Atlanta Fed MPT and the NY Fed dealer survey.
> **DC-7 · Lookahead.** Only vintage-stamped inputs may enter any historical fit: Cleveland Fed daily nowcast vintages, MPT daily history, and never-revised index levels. Current FRED values for revised series (`PAYEMS`, SA CPI, retail sales) must **not** be used to reconstruct an as-of state; ALFRED vintages or nothing.
> **DC-8 · Method labelling.** Every ledger entry records its method; calibration tracks never merge across methods (§ 4e).

---

#### AMENDMENT 1 — DC-3a and DC-3b, LOCKED 2026-08-24 (written before any v2 number existed)

> **V2 RESULT (2026-08-24, same day): NO-GO under the verdict rule locked below — but on the third criterion, not on accuracy.** Both accuracy bars **PASS** and by a wide margin: pooled Brier over the 8 meetings **0.1273** against base-rate-only's **0.2349** (criterion A, the bar v1 failed worst), and the July-2026 market scores **0.1673** against the 0.2144 bar on the identical 15-day window where v1 scored 0.4223 (criterion B — peek-compromised, see the disclosure). **Criterion C fails with 6 violations, and all six sit on cutting-pressure meetings while none sit on the three hiking-pressure ones**: C's sign map ("SEP median up ⇒ holding less likely") is only correct when the desired rate is *above* target, and v1 passed it solely because it was tested on one hiking market. **The rule is right and the criterion is regime-specific — and the NO-GO stands anyway**, because a locked criterion is not rewritten after seeing the number. Two further findings: **DC-3b alone (the reference-class fix) scores 0.1200, better than both amendments combined**, so v1's ranking of the defects was backwards; and the `W_SEP` SEP-concentration idea is mildly counterproductive (W_SEP 1.0 → 0.1245). **Nothing built, OpenBB still NOT installed.** Full write-up, decomposition and the three-change v3 proposal: [[newsagent_data_channel_v2_findings]].

> **Status: LOCKED.** These two rows amend DC-3 and bind any re-run of the data channel from this date. They were written, with every constant fixed, **before** a single v2 `p_struct`, Brier or comparison was computed — the [[newsagent_v0_gate_findings]] Amendment-1/2 precedent. DC-1, DC-2, DC-4 … DC-8 stand unchanged. The defect being repaired is the one the dry run named: [[newsagent_data_channel_dryrun_findings]] § 6 mechanism 1 (per-meeting allocation) and mechanism 2 (reference class).

**DC-3a · Allocate the gap across the remaining calendar; do not divide by it.**

The object to predict is a *single* meeting, so the rule must say which meeting absorbs the pressure. Steps 1 and 2 of DC-3 are unchanged up to the gap; everything after it is replaced.

```
Step 1 (UNCHANGED from DC-3) — desired rate from the Fed's own projected path
    r_desired(t) = FEDTARMD_y(t) + 0.5·(π(t) − JCXFEMD_y(t)) − 0.5·(u(t) − UNRATEMD_y(t))
    where y = the SEP projection year that the horizon below runs to.

Step 2a — total remaining pressure, in 25bp clicks (a MAGNITUDE, not a per-meeting rate)
    G(t) = |r_desired(t) − target_midpoint(t)| / 0.25

Step 3a — allocate G over the remaining meetings with DECLARED hazard weights
    Horizon: the scheduled FOMC meetings from t through the end of SEP year y,
             indexed i = 1..N with i = 1 the NEXT meeting (the one being scored).
    w_i   = W_SEP  if meeting i carries a Summary of Economic Projections
          = W_NON  otherwise
    W_SEP = 2.0,  W_NON = 1.0                     (DECLARED, not fitted)
    share_i = w_i / Σ_j w_j
    μ_i     = G(t) · share_i                       (expected clicks at meeting i)

Step 4a — map the allocated pressure to this meeting's probability
    logit(P_move at meeting 1) = logit(h_base) + B1A · μ_1
    B1A = 1.5                                      (DECLARED, not fitted)
    p_struct(t) = 1 − P_move,  clipped to [0.02, 0.98]

DC-2 is UNCHANGED: π is integrated, not point-estimated —
    π ~ Normal(center = Cleveland nowcast chain as of t, σ = 0.063pp·√k),
    p_struct(t) = E_π[ 1 − P_move(π) ],  20,000 antithetic draws, seed 0.
```

*Why W_SEP = 2.0.* The FOMC concentrates policy changes at meetings that publish a new projection path: an SEP meeting is where the committee re-states where it thinks rates are going, and a change is the natural way to start delivering it. A 2:1 ratio is a **declared judgment**, in the same class as γ = 1.0 and the lean multipliers — n is far too small to fit it, and it is deliberately coarse. It is also **weaker than it was pre-2019**, when only SEP meetings carried a press conference; every meeting has one now, so 2:1 is the conservative end of the plausible range rather than the middle.

*Why B1A = 1.5.* On the log-odds scale, one **full click** of pressure allocated to a single meeting should mean "a move here is now more likely than not, but not near-certain". From the DC-3b base hazard of 0.3516 (logit −0.6118), μ₁ = 1.0 gives logit 0.888 → **P(move) = 0.708**; μ₁ = 2.0 gives 0.916. That is the intended shape, and it is the reasoning the constant was chosen from — not a value selected against an outcome.

*Sign convention.* B1A is **positive** and acts on P(**move**); DC-3's B1 was negative and acted on P(hold). Same direction of travel, opposite framing — stated because the sign flip is otherwise an easy misreading.

*Worked example — the July 2026 meeting, decision day 2026-07-29.* Inputs are the ones already recorded in [[newsagent_data_channel_dryrun_findings]] § 3: `r_desired = 3.880%`, midpoint `3.625%`, so `gap = +0.255pp` and `G = 1.02` clicks. Remaining 2026 meetings from that date: **Jul 29 (non-SEP, w=1)**, Sep 16 (SEP, w=2), Oct 28 (non-SEP, w=1), Dec 9 (SEP, w=2) → Σw = 6, `share₁ = 1/6 = 0.1667`, `μ₁ = 1.02 × 0.1667 = 0.170`. Then `logit(P_move) = −0.6118 + 1.5 × 0.170 = −0.357` → `P_move = 0.412` → **p_struct = 0.588**. The old rule produced 0.343 on the same inputs. The difference is entirely allocation: the same +0.26pp of pressure now sits mostly on September and December, which is where the rates market put it (§ 7 of the dry run, display-only).

**DC-3b · Reference class — ONE choice, declared: the all-years per-meeting hold rate.**

`h_base = 1 − (all-years per-meeting no-change rate)`, computed over **every scheduled FOMC meeting 1994–2025 with no ZLB exclusion** (the dry run measured this at **0.6484** hold, i.e. `h_base = 0.3516`; the v2 run recomputes it rather than hardcoding it). The ZLB exclusion in the original DC-3 removed 2009–2015 and 2020–2021 — precisely the stretches where the Fed held at every meeting — which is the wrong reference class for a *per-meeting* question. **No regime conditioning is used.** A regime-conditioned rate would require choosing and dating a regime classifier, which is a fitting decision the sample cannot support; the alternative is declared here and rejected, so it cannot be reached for later.

**Pre-registered test design for the v2 re-run**

> **Test set — 8 resolved FOMC decisions, fixed before any computation.** Chosen for balance, not for outcome: 4 SEP / 4 non-SEP, 3 moves / 5 holds. The three moves are the only scheduled changes in the window; every 2026 meeting to date was a hold.
>
> | # | Decision date | SEP? | Outcome | Note |
> |---|---|---|---|---|
> | 1 | 2025-09-17 | SEP | **MOVE** (cut, 4.50 → 4.25) | |
> | 2 | 2025-10-29 | non-SEP | **MOVE** (cut, 4.25 → 4.00) | |
> | 3 | 2025-12-10 | SEP | **MOVE** (cut, 4.00 → 3.75) | last meeting of its SEP year: N = 1, share = 1 |
> | 4 | 2026-01-28 | non-SEP | hold | |
> | 5 | 2026-03-18 | SEP | hold | |
> | 6 | 2026-04-29 | non-SEP | hold | |
> | 7 | 2026-06-17 | SEP | hold | |
> | 8 | 2026-07-29 | non-SEP | hold | **the market that killed v1** (sf-2026-003) |
>
> **Scoring.** Outcome is coded in the market's own framing: **1 = no change, 0 = change**, so `p_struct` is directly a probability of the YES side. Per-meeting score = Brier at `p_struct` on the **decision day itself**, computed from ALFRED/Cleveland vintages as of that day. This is safe and is verified rather than assumed: `DFEDTARU` records a change at its *effective* date, the day **after** the decision, so the decision-day vintage cannot see the outcome. Daily trajectories run over **decision date − 45 days … decision date** for the directional check.
>
> **Criteria, locked before running.**
> * **A · Pooled accuracy.** Mean Brier of `p_struct` over all 8 meetings **strictly below** mean Brier of base-rate-only (the same `1 − h_base` on every meeting). This is the bar the dry run failed by the widest margin.
> * **B · The July market specifically.** Brier of `p_struct` over the **same 15-day intersection the dry run used** (2026-06-21 … 2026-07-05) **strictly below 0.2144**, which is both the news-FV and the prior-only figure on that market (they coincide because the published FV never left its prior). Scored on the identical window so it is directly comparable to the recorded 0.4223.
> * **C · Directional sanity.** Zero violations across all 8 trajectories, 0.5pp dead-band: an SEP median moving up, or π rising, must not move `p_struct` up. Unemployment carries **no** pre-registered direction and is reported, not scored — unchanged from the dry run.
>
> **Verdict rule. GO iff A AND B AND C. NO-GO otherwise, and the channel stays unbuilt.** A GO authorises exactly what § 7 row 1 authorises and nothing more.
>
> **Reported, never a bar** (so a win cannot be quietly reattributed): per-meeting Brier for all 8; the same for base-rate-only; a sensitivity sweep over `B1A ∈ {0.5, 1.0, 1.5, 2.0, 3.0}` and `W_SEP ∈ {1.0, 1.5, 2.0, 3.0}`; and the old DC-3 rule re-scored on the same 8 meetings, so the amendment's contribution is separable from the reference-class change.
>
> **Anti-post-hoc.** No constant may be changed after seeing any output. n = 8 meetings — **no confidence interval is claimable and none will be quoted**; 8 is enough to kill a method, not to bless one.
>
> **DISCLOSURE (integrity, recorded because it would otherwise be invisible).** While deriving DC-3a, the July-2026 arithmetic above was worked through by hand to check the mechanics produce a probability in range and move in the intended direction. That calculation yields `p_struct = 0.588` on a market that resolved YES, i.e. it was **visible that the amended rule beats the old one on meeting #8 before the constants were locked**. The constants were chosen from the reasoning stated above, not tuned against that number, and no other value of B1A or W_SEP was evaluated. But criterion **B is knowingly compromised by this peek and must be read as such**: the weight of the test rests on **criterion A**, the pooled 8-meeting bar, and on the seven meetings whose numbers had not been computed in any form when this section was locked. If A fails and B passes, that is a fail.

---

---

#### AMENDMENT 2 — regime-neutral criterion C, W_SEP = 1.0, extended sample. LOCKED 2026-08-24 (written before any v3 number existed)

> **V3 RESULT (2026-08-24, same day): GO — all three criteria pass, and the channel is BUILT.** Pooled **Brier 0.1617 vs base-rate-only 0.2795** over the locked 40 meetings (criterion A); July-2026 **0.1926** vs its 0.2144 bar (criterion B, peek-compromised); the regime-neutral criterion C shows **0 violations across all 40 trajectories**, where v2's regime-specific version produced 6; the V3-5 leakage guard dropped **0 of 40**. Shipped the same day under sign-off row 1: OpenBB installed, the offline ingest boundary, `p_struct` wired as the re-anchorable prior on the **September Fed market** (Option C, DC-5 blend weight 1.0, DC-4 honoured — α untouched at 2.85), the § 4d double-count guard live (fired once on day one), the market added to `config.DATA_CHANNEL_MARKETS`, and its card moved to **"data-channel scored"** with the n=0 transition line. **658 tests green.** Two honest caveats recorded in [[newsagent_data_channel_v3_findings]]: the method is **worse than base-rate-only in the holding regime (0.1528 vs 0.1236) — the regime the live market is in** — and the `W_SEP` sweep **reverses** v2's finding on the larger sample, so the SEP-concentration effect was never supported in either direction. Also measured: **OpenBB cannot serve this channel** (`fred_series` has no realtime/vintage parameters), so it is installed, authorised and not on the critical path.

> **Status: LOCKED.** Written with every constant fixed **before** a single v3 `p_struct`, Brier, violation count or comparison was computed. Amendment 1 stands except where explicitly replaced below. What was done *before* this lock, and recorded so nobody has to wonder: **data-availability probes only** — confirming that ALFRED serves historical SEP vintages (`FEDTARMD` reads 3.4 for end-2022 as known on 2022-07-27, 1.4 for end-2017 as known on 2017-06-14), that the Cleveland nowcast vintages start 2013-07, and reading the FRED target-change dates in order to *compose a balanced sample and check it for leakage*. No reaction-function output of any kind was produced.

**V3-1 · Criterion C becomes regime-neutral.** The implied direction is taken on the **pressure MAGNITUDE**, not on the raw direction of the news:

```
On a release day (the SEP median changed, or a new core-PCE month was published):
    d_pressure = |r_desired(t) − midpoint(t)| − |r_desired(t−1) − midpoint(t−1)|
    VIOLATION iff  (d_pressure > 0 and Δp_struct > +DEAD_BAND)
                or (d_pressure < 0 and Δp_struct < −DEAD_BAND)
    DEAD_BAND = 0.005 (0.5pp, unchanged);  |d_pressure| must exceed 1e-4 pp to score.
```

**This fixes a mis-specification, and the mis-specification is mine.** v2's criterion C read "an SEP median moving up, or inflation rising, must not move `p_struct` up". That is correct **only when the desired rate sits above the target** — under cutting pressure the same hawkish news *shrinks* the gap, reduces the pressure to act, and correctly makes holding **more** likely. v1 passed the criterion on two directional days in a **single hiking-pressure market**, which is the entire evidential basis it ever had; v2 was the first time it met a cutting cycle and it produced **6 violations, all six on negative-gap meetings and none on the three positive-gap ones** ([[newsagent_data_channel_v2_findings]] § 5a). The rule was right and the test was regime-specific. Taking the direction on `|gap|` makes the check say what it always meant to say, on both sides of the cycle. Unemployment still carries no pre-registered direction and is reported, not scored.

**V3-2 · `W_SEP = 1.0` — allocate evenly across the remaining calendar.** The allocation *framing* of DC-3a is kept, because a single meeting is the right object to predict and a per-meeting quotient is not. The **SEP-concentration hypothesis is dropped**: v2's § 6 sweep measured it as flat-to-mildly-counterproductive (`W_SEP` 1.0 → pooled Brier 0.1245, 1.5 → 0.1252, 2.0 → 0.1273, 3.0 → 0.1317), i.e. the preference degrades monotonically as it strengthens. It was declared, tested on 8 meetings and **not supported**, and that is recorded here rather than quietly dropped.

**V3-3 · `B1A` stays 1.5.** The v2 sweep prefers 2.0–3.0 (pooled 0.1165 / 0.1158 against 1.5's 0.1273). **It is deliberately NOT moved.** Re-tuning a constant toward a value chosen by looking at an outcome is precisely the fitting this pre-registration exists to forbid, and the fact that the declared value sits *away* from the sweep's optimum is evidence the reasoning-first derivation was honest. It stays at 1.5 with the § 4f Amendment-1 rationale unchanged.

**V3-4 · Extended sample — 40 meetings, declared in full before running.** v2's sample was one easing-then-holding cycle containing **no hikes at all**. This adds a full hiking cycle and a normalisation cycle, both with verified ALFRED and Cleveland coverage:

| Block | Decision dates | Moves |
|---|---|---|
| **2017 normalisation** | Feb 1, **Mar 15\***, May 3, **Jun 14\***, Jul 26, Sep 20\*, Nov 1, **Dec 13\*** | 3 hikes |
| **2018 normalisation** | Jan 31, **Mar 21\***, May 2, **Jun 13\***, Aug 1, **Sep 26\***, Nov 8, **Dec 19\*** | 4 hikes |
| **2022 hiking** | Jan 26, **Mar 16\***, **May 4**, **Jun 15\***, **Jul 27**, **Sep 21\***, **Nov 2**, **Dec 14\*** | 7 hikes |
| **2023 hiking→hold** | **Feb 1**, **Mar 22\***, **May 3**, Jun 14\*, **Jul 26**, Sep 20\*, Nov 1, Dec 13\* | 4 hikes |
| **2025 cutting** (from v2) | **Sep 17\***, **Oct 29**, **Dec 10\*** | 3 cuts |
| **2026 holding** (from v2) | Jan 28, Mar 18\*, Apr 29, Jun 17\*, **Jul 29** | 0 |

`*` = SEP meeting; **bold** = the Fed moved. **40 meetings: 21 moves / 19 holds, 18 of the moves hikes and 3 cuts, 20 SEP / 20 non-SEP.** 2015 and 2016 are **excluded deliberately** — see V3-5.

**V3-5 · A hard per-meeting leakage guard, and why 2015–2016 are out.** v2 verified that `DFEDTARU` records a change at its *effective* date, the day **after** the decision, so a decision-day vintage cannot see the outcome. Checking that assumption across the extended sample found it is **not universal**: the December-2015 liftoff is stamped **on the decision day itself** (`2015-12-16`), which would leak the outcome into its own score. Therefore:

* 2015 and 2016 are excluded from the sample, and
* every remaining meeting carries a **runtime assertion**: the target midpoint read as of the decision day must still equal the **pre-decision** level. Any meeting that fails is **dropped from the sample and recorded by name** — never silently scored.

**Criteria and verdict rule — unchanged from Amendment 1.**
* **A · Pooled accuracy.** Mean Brier over the sample **strictly below** base-rate-only. This is the bar that carries the weight.
* **B · The July-2026 market**, same 15-day intersection, **strictly below 0.2144**.
* **C · Directional sanity**, now regime-neutral per V3-1: **zero violations**.
* **GO iff A AND B AND C.** NO-GO otherwise, and the channel stays unbuilt.

> **Criterion B remains PEEK-COMPROMISED**, carried forward from Amendment 1's disclosure: the July-2026 arithmetic was worked through by hand while deriving DC-3a, so it was visible that the amended rule beats the old one on that one meeting before the constants were locked. **The weight of this test rests on criterion A**, now over 40 meetings across three distinct rate regimes, and on criterion C, whose fix is a strict widening of what counts as correct behaviour rather than a loosening of the bar.

**Reported, never a bar:** per-meeting Brier for all 40; base-rate-only and the v1 rule on the same set; the pooled result split by regime (hiking / cutting / holding) so a method that only works in one is visible; sensitivity over `B1A` and `W_SEP`; and the count of meetings dropped by the V3-5 leakage guard.

**Anti-post-hoc.** No constant may be changed after seeing any output. n = 40 meetings — larger than v2's 8 and still **no confidence interval is claimable and none will be quoted**.

---

## 5 · Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions.** The retrospectively-retrieved Nasdaq consensus is assumed to be the pre-release value (not vintage-verifiable — the live channel must snapshot consensus daily to be clean going forward). `n_expected` uses "FRED carries a print for that reference month" as the definition of "a release happened". MoM-vs-YoY row assignment uses each row's already-published `previous` value matched against FRED's prior-month official figure, so the current release's actual/consensus never influenced the matching; 5 rows were dropped as unresolvable and are listed in `newsagent_datachannel_dropped_rows.csv`. Surprise RMSE measured over 12 months is taken as a stand-in for the σ of a forward print distribution. The AGPL reading in § 1 is engineering judgment, not legal advice.

**Live-only unknowns.** Whether a declared reaction function can produce a `P(no change)` that beats the base rate out-of-sample (DC-3 — unknown until run). Whether the double-count guard (§ 4d) fires cleanly on real packets. Whether re-anchoring `p0` mid-life destabilises the evolved FV for slow markets whose `shift_clip` guard was tuned around a *static* prior. Whether the Nasdaq calendar's coverage holes are stable or drift. Whether readers interpret a two-method page as honest or as hedging.

**Power honesty.** n = 4–10 releases per series in the primary window. The coverage finding is a **count** and robust; every RMSE here is a point read without a CI and must not be quoted as a calibrated σ without re-measurement on a longer window.

---

## 6 · Gotchas logged

- **`openbb-nasdaq`'s calendar fetcher fails two ways on long ranges.** It raises `Nasdaq Error -> No record found` for an entire range when a no-event date resolves first in its async gather (a race, not an absence), and its ~30-way concurrency trips `api.nasdaq.com` rate-limiting (HTTP 403). Under a naive bisecting retry both degrade into **silently empty dates** — which in this study would have masqueraded as missing releases and inflated the coverage failure. The bulk pull therefore goes direct to the same public endpoint, sequentially and paced, recording ok/empty/failed per date (final pull: **433 ok / 88 genuinely empty / 0 failed**). Use the provider for interactive/short ranges; use a paced direct pull for backfills.
- **The Cleveland Fed inflation nowcast is not on FRED.** The scoping brief assumed it was retrievable there (as GDPNow is, via `GDPNOW`); no FRED series for it could be found. The real free path is the Cleveland Fed's own public JSON (`nowcast_month.json`) — which is *better* than a FRED series would be, because it carries daily vintages and the realized actual per target month.
- **`GDPNOW` on FRED is latest-vintage only** — one revised value per quarter. Fine live, lookahead-unsafe for backfills without ALFRED.
- **BLS's own API answers without a key** (v1 and v2, verified) even though `openbb-bls` demands one; the key buys quota, not access.
- **BLS schedule pages return HTTP 403** to automated fetches; use FRED's releases-dates endpoint for release scheduling.
- **The Atlanta Fed MPT's `Prob: cut` / `Prob: hike` are in percent, not fractions** (0.82 and 56.98 → `P(hold)` = 42.2%, not a negative number). Its `LICENSE` sheet is an embedded image and cannot be read by openpyxl — read the terms on the site.

---

## 7 · Sign-off table for Justin — ANSWERED 2026-08-24

Every row was a decision, a registration, or an acknowledgement. **Justin decided all twelve rows on 2026-08-24** (in chat); the last column records each outcome verbatim in substance, and § 7a below says what was actually *done* on the back of them. Rows that were merely approved-in-principle are marked as such: approval is not a build.

| # | Item | What it costs | Recommendation | Blocking? | **Decision — Justin, 2026-08-24** |
|---|---|---|---|---|---|
| 1 | **Install the lean OpenBB set into the research venv** (`openbb-core`, `-economy`, `-fred`, `-bls`, `-nasdaq`, + keyless `-federal-reserve`) — 30 packages, **zero** changes to existing pins | $0, disk only | **Approve when the channel is greenlit, not before** — the scoping ran in a throwaway venv and the research venv is untouched | blocks any build | **APPROVED — install when the build starts, not before.** The research venv is still untouched today. |
| 2 | **AGPL-3.0-only acknowledgement** + the boundary rule: OpenBB stays in an offline ingest script; `newsagent/*` never imports it | $0 | Acknowledge the reasoning in § 1 or overrule it | blocks any build | **ACKNOWLEDGED.** OpenBB stays in an offline ingest script writing JSON/Parquet; `newsagent/*` never imports it — the same isolation `gdelt_bq.py` already has. |
| 3 | **Register + export `OPENBB_FRED_API_KEY`** (free, fredaccount.stlouisfed.org) | $0 | Do it — also unlocks release-date scheduling, which the free calendar cannot supply | blocks the FRED path | **APPROVED — key registered by Justin and loaded.** It lives in the git-ignored `polymarket/research/.env`, read by `config.load_env()` at import; verified live against FRED `DFEDTARU` (3.75 on 2026-08-24). Value never printed or logged. |
| 4 | **Register + export `OPENBB_BLS_API_KEY`** (free) | $0 | Optional-but-recommended: OpenBB demands it, and it lifts BLS from 25 to 500 queries/day | blocks the BLS path only | **APPROVED in principle, SKIPPED for now** (Justin): the BLS API answers keyless, so the key only lifts quota 25 → 500/day. The `.env` carries a commented placeholder and `config.ENV_KEYS` already lists the name. |
| 5 | `OPENBB_NASDAQ_API_KEY` | $0 | **Not needed** — the calendar is keyless. Register only if Nasdaq Data Link datasets are ever wanted | no | **NOT NEEDED** — confirmed. |
| 6 | **TradingEconomics (paid consensus)** | **$$** | **Decline for now.** The free consensus failed on coverage, not accuracy, and the recommended design does not need it (§ 3c) | no | **DECLINED.** The failure was coverage, not accuracy, and the recommended design does not need a consensus feed. |
| 7 | **Split the `DATA_DRIVEN` tag** into `data-driven` (Fed ×2) and `poll-driven` (Becerra, CA wealth tax); refresh the stale "state-primary polling" wording on the Becerra card | $0 | Approve — it is an honesty fix regardless of whether the channel is built | no | **APPROVED and APPLIED the same day** — see § 2b. |
| 8 | **Design decision: Option C** (structural probability as a re-anchorable prior) over A (data-as-article) or B (separate FV) | $0 | Approve to make § 4f the standing pre-registration | blocks the build | **APPROVED** — § 4f is now the standing pre-registration. |
| 9 | **Design decision: market-implied probabilities are DISPLAY-ONLY** (Atlanta Fed MPT, dealer survey) — they never enter `p_struct`, `A_t`, or any fit | $0 | Approve — this is what keeps "independent fair value" true (§ 4c) | blocks the build | **APPROVED** — Atlanta Fed MPT and the dealer survey may render as labelled benchmarks and never enter `p_struct`, `A_t` or any fit. |
| 10 | **Display decision:** the two Fed cards move from "structurally blind" to "data-channel scored", with separate per-method calibration tracks and an explicit "no settled track record yet (n=0)" during the transition | $0 | Approve alongside #8 | blocks publishing | **APPROVED** — with per-method ledger labels and the explicit "new method — no settled track record yet (n=0)" transition line. Both shipped 2026-08-24: `ledger.method_for/record_method` label every snapshot, and the public track-record panel reports per method and never merges (DC-8). |
| 11 | **Declared knobs awaiting a value:** DC-5 blend weight (proposed **1.0**, full replacement) and the DC-3 reaction function's declared form | $0 | Bless the proposal or set your own; both are DECLARED, refit-covered, and reversible | blocks the build | **APPROVED AS PROPOSED** — DC-5 blend weight = **1.0** (full replacement); DC-3 ships in **declared** form (an inspectable rule over the inflation gap, the labour gap and the SEP median) and may only be fitted under its own pre-registered procedure. |
| 12 | Carried from [[newsagent_observatory_v32_findings]], unchanged by this pass: Scheme-A + lean sign-offs, `GUARDIAN_API_KEY`, `GEMINI_API_KEY`, and the **July Fed settlement** (`sf settle` — the market resolved YES on 2026-07-29) | $0 | Unchanged | — | **ALL ANSWERED 2026-08-24.** Scheme-A uncovered weights APPROVED and LIVE (ING 0.9 / WP-CE 0.8 / unknown 0.5 / Bloomberg 0.9 / bank desks 0.9); lean table + extremity multipliers APPROVED as shipped; band_mult selector call APPROVED (keep 0.5); `GUARDIAN_API_KEY` now loads from `.env` when its value is pasted (demo-key fallback meanwhile); `GEMINI_API_KEY` still open; the four July settlements are **settled and scored** — see [[newsagent_observatory_v33_findings]]. |

### 7a · What the decisions changed on 2026-08-24 (and what they did not)

**Applied the same day — code, and test-enforced:**

- **Row 7 — tag split.** `newsagent/config.py` now carries `DATA_DRIVEN` (official statistics exist: the September Fed market) and `POLL_DRIVEN` (private/issue polling: Becerra, CA wealth tax), with `NOT_NEWS_TRACTABLE` as the merged view and `tract()` returning `news | data | poll`. The dashboard renders `◆ data-driven` / `◆ poll-driven` chips with type-specific card copy, and the Becerra note now names **general-election** polling instead of the stale primary wording. Tests: `test_tract_tagging`, `test_becerra_note_is_refreshed_to_general_election`, `test_html_poll_driven_card_says_polling_not_data`.
- **Rows 3-4 — credential loading.** The pipeline did **not** read `polymarket/research/.env` before today: every module read `os.environ` directly, so a key existed only if the calling shell happened to export it (which is exactly how the v3.1 backfill ended up running on the Guardian demo key). `config.load_env()` now loads that git-ignored file at import for `GUARDIAN_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, `OPENBB_FRED_API_KEY` and `OPENBB_BLS_API_KEY`. Rules it enforces: the **real environment always wins** over the file; a value that is only an inline comment (`KEY=   # not pasted yet`) is treated as **unset** rather than armed as a garbage credential; a relative `*_CREDENTIALS` path is resolved against the `.env`'s own directory so a run from another cwd still finds it; and the function returns **key NAMES only** — no value is ever printed, logged or written to a note. Justin's FRED key is loaded and was verified live against FRED `DFEDTARU`; the BLS key is skipped by his instruction (the BLS API answers keyless); `GUARDIAN_API_KEY` has a placeholder line awaiting its value, with the demo-key fallback unchanged until then. Tests: `test_load_env_sets_only_missing_keys_and_skips_comments`, `test_load_env_resolves_credential_paths_against_the_env_file`, `test_load_env_absent_file_is_not_an_error`.
- **Row 10 — per-method honesty, shipped ahead of the channel.** `ledger.method_for()` / `record_method()` label every ledger snapshot with the method that produced it (`news` today; `news+data` only when a market actually joins `config.DATA_CHANNEL_MARKETS`, which is deliberately **empty**), and the new public track-record panel reports Brier **per method**, never merged, printing "new method — no settled track record yet (n=0)" for a live method with nothing settled. That is DC-8 in force before the channel exists, which is the only order that works: retro-fitting method labels after a switch is exactly the thing that would make the two tracks unrecoverable.
- **Row 8/11 — § 4f is LOCKED** as the standing pre-registration (banner added there), with DC-5 = 1.0 and DC-3 in declared form.

**Approved but deliberately NOT done:**

- **Row 1 — the OpenBB install has not happened.** Justin's approval is explicitly "when the build starts, not before", so the research venv is still OpenBB-free and the scoping numbers still come from a throwaway venv. Nothing in `newsagent/*` imports OpenBB, and nothing should until the channel is greenlit.
- **The channel itself is still unbuilt.** No `p_struct`, no ingest script, no re-anchoring, no Fed card copy change. The two Fed cards' honest "structurally blind" note stays exactly as it was — one of them (July) has in fact left the slate entirely, having resolved YES and been settled.

**Consequence worth stating plainly:** the § 9 sequence below is unchanged in substance — steps (1) and (2) are now done, and the next real work is step (3), the retrospective dry run of the declared structural method on the resolved July Fed market. That market resolved **YES** (no change) on 2026-07-29, and our news-only forecast had it at **53.7%** (Brier 0.214 — our worst of the four settlements). If a declared reaction function built from vintage-stamped inputs cannot beat 53.7% on that meeting, the channel does not deserve to be built.

---

## 8 · Outputs

- **Script:** `scripts/newsagent_data_channel_scoping.py` — reproduces every number in § 1 and § 3 plus the figure (`--chart`). It runs against a throwaway OpenBB env (`uv run --with openbb-core --with openbb-economy --with openbb-nasdaq …`) so the research venv stays clean; the chart step needs the research venv's matplotlib.
- **CSVs** (git-ignored, `data/analysis/csv_outputs/news_agent/`): `newsagent_datachannel_consensus_pairs.csv` (67 release rows: series, release date, reference month, matching rule, consensus, calendar actual, previous, FRED official, pre-release nowcast), `…_consensus_summary.csv` (M1–M6 per series × sample), `…_coverage_split.csv` (expected / found / with-consensus / absent), `…_integrity_nsa.csv` (the 17 never-revised checks), `…_dropped_rows.csv` (the 5 documented drops).
- **Plot:** `data/analysis/plots/news_agent/newsagent_datachannel_consensus_check.png`.
- **Cache** (git-ignored, `data/newsagent/datachannel/`): `calendar_raw.csv` (20,115 rows, 2024-08-24 → 2026-08-24), `calendar_fetch_status.csv` (per-date ok/empty/failed), `fred_*.csv`.
- **Pre-registration:** locked to the session scratchpad before the pull; reproduced in § 3a.
- **Not touched:** `fvmodel.py`, `config.py`, `dashboard.py`, `fv_params.json`, the sf ledger, and every published page — as scoped.

## 9 · Decision and next step

**Decision: the data-evidence channel is worth building for the Fed markets, at $0, but only under the § 4f pre-registration and the display-only rule for market-implied inputs. SIGNED OFF by Justin 2026-08-24 (§ 7).** It is not buildable for Becerra or the CA wealth tax at any tier, and those two should be re-tagged rather than promised a fix.

**Next step (in order):** ~~(1) Justin signs off rows 1–2 and 8–11 of § 7~~ **done 2026-08-24**; ~~(2) lock § 4f as the standing pre-registration~~ **done 2026-08-24**; ~~(3) dry-run the declared structural method retrospectively on the resolved July Fed market~~ **DONE 2026-08-24 → NO-GO**: `p_struct` drifted from 44% to 34% on a market that resolved YES, scoring **Brier 0.4223 vs the news-FV's 0.2144** and losing to its own base-rate-only benchmark (0.2236) — the dry run was embarrassing, in exactly the way a cheap falsifier is supposed to be ([[newsagent_data_channel_dryrun_findings]]); **(4) NOT DONE and not authorised** — the install, the ingest boundary and the Fed card copy all wait behind a *passing* dry run. Row 1's approval says "when the build starts"; the build has not started. **Do not** buy a consensus feed at any point in that sequence.

**What the failure did NOT kill.** The input plumbing is sound and provably lookahead-free (ALFRED realtime vintages + daily Cleveland nowcast vintages + never-revised target-range history), the method moves in the right direction on every mapped release, and the re-anchoring is gentle enough for the live slow-market guard. The defect is one design step — how pressure is allocated across the remaining meetings — and the amendment queue for it is in [[newsagent_data_channel_dryrun_findings]] § 9.

One scope note for whoever picks up step (3): the slate refresh of 2026-08-24 retired the July Fed market (resolved) and did **not** add a replacement Fed market, so `DATA_DRIVEN` now holds the **September** meeting alone. The dry run still uses July — a resolved meeting is exactly what a retrospective falsifier needs — but the live channel, if built, would first serve September (meeting 2026-09-15/16, an SEP meeting).
