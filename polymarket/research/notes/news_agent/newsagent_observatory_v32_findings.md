---
title: "Observatory v3.2 — page-level 2-column dashboard (feed pane + surfaced donuts), political-lean axis (Ground News not viable → AllSides-seeded), big movers + evidence-quality badges, deployed-site design language (attended build)"
created: 2026-07-05
status: shipped — attended/manual; lean table + declared multipliers awaiting Justin sign-off (running live, refit-covered); website parked
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
  - dashboard
---
# Observatory v3.2 — the dashboard becomes a two-pane page, and bias gets a lean axis

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[COWORK]] · Table terms: [[polymarket_table_dictionary]]
> Builds ADDITIVELY on [[newsagent_observatory_v31_findings]] (commit `fb63de8`). The closed "our % beats the mid" gates ([[newsagent_v0_gate_findings]]) stay closed and displayed; the thesis is unchanged — our independent fair value judged against resolved outcomes; the Polymarket mid is discovery + display context only.

## Plain-English Summary

- **What this note is:** the build record of Observatory v3.2 (2026-07-05, one attended session): the dashboard was rebuilt as a **page-level two-column layout** (left = a sticky aggregated news/evidence feed with its own scroll; right = markets), every market's **donut gauge moved into a full-width overview grid** visible without expanding anything, and the whole page was restyled to the **deployed epsilon site's new design language** (warm charcoal/cream/terracotta, numbered sections, pill buttons — inspected live and reproduced with local fonts).
- **Bias gained a second axis.** Justin's pick was Ground News for political lean; the access check found **no public API and a ToS that expressly prohibits both automated and manual monitoring/copying** — so the plan's named fallback shipped instead: a **curated lean table seeded from verified AllSides ratings** (the axis Ground News itself averages over). Lean extremity now scales Stage-B contribution (composed with the existing reliability weight; **reliability ≠ lean**, RSP + Iffy keep the blocklist role), and the per-card ratings explainer shows the lean mix behind every number plus our own packet's left/center/right coverage distribution.
- **α refit under the composed weights** on the same 355-pair / 49-market resolved sample: **α = 2.7** (from 2.1 — almost exactly the mechanical 1/0.75 compensation for Guardian's new lean multiplier), pooled Brier a hair better (0.2214 vs 0.2216); **band_mult stays 0.5** after a granularity fix to the coverage selector (documented below — a judgment call flagged for Justin).
- **Two cheap extensions shipped:** a **big-movers strip** (largest day-over-day FV changes from published snapshots only — reconstructed history never counts; correctly empty until two published days exist) and an **evidence-quality badge** per market (thin / moderate / strong from n_relevant + band width, so readers discount thin-evidence markets — the "strong" bar is by construction the divergence flag's confidence leg).
- **One-line status:** 24 markets re-published under the refit params (ledger sf-2026-001…024 updated pre-settlement, append-only); still no divergence flag; **76 tests green**; IP-scrub clean; rendered page verified in headless Chrome at desktop + mobile widths; website stays parked (built local, site-ready).

## Ground News — the access verdict (why the fallback shipped)

The v3.2 mandate: use Ground News for the lean axis "ToS-aware", with a curated-table fallback if access isn't viable. It is not viable:

1. **No official API.** No developer program, feed, or documented API exists (searched 2026-07-05; their public surfaces are the app, site, extension and help center).
2. **ToS prohibition.** ground.news Terms & Conditions (read 2026-07-05) prohibit using "any robot, spider, or other automatic device, process, or means to access the Site for any purpose, including monitoring or copying any of the material on the Site" **and** "any manual process to monitor or copy any of the material" not expressly authorized. That forecloses both a structured fetch *and* systematically transcribing their ratings by hand. Non-monetised use doesn't cure a contract prohibition.
3. **The fallback is equivalent for our purposes.** Ground News's own per-source ratings are averages of AllSides + Ad Fontes + MBFC. AllSides (the radar's "easier-access alt") publishes its ratings openly under CC BY-NC 4.0; the Observatory is non-monetised and attributes in the footer. The radar's outstanding "written NC OK" nicety remains a Justin item (below).

Per-story L/C/R coverage bars: taking Ground News's per-story distribution would hit the same ToS wall, so the explainer computes **our own packet's** L/C/R distribution from the lean table — labeled on the page as exactly that ("what we read", never a third party's per-story figure). This is arguably the more honest object anyway: it describes the evidence that actually formed our number.

## The lean layer (`newsagent/sourcelean.py`)

**Table (verified against AllSides published ratings, 2026-07-05; scale −2 Left … +2 Right; None = unrated):**

| Domain | Lean | AllSides rating (note) |
|---|---|---|
| theguardian.com | −2 | Left (moved from Lean Left, Nov 2024 editorial review) |
| bbc.co.uk | 0 | Center |
| politico.com | −1 | Lean Left |
| thehill.com | 0 | Center |
| bloomberg.com | −1 | Lean Left |
| foxnews.com | +2 | Right (since 2021) |
| dailymail.co.uk | +1 | Lean Right (moved from Right, Sep 2025 editorial review) |
| news.sky.com | unrated | no published AllSides rating found — honestly absent |
| newsletters / bank-research PDFs / Wikipedia CE | unrated | out of scope, same scope rule as the RSP reliability tiers |

**Routing (the "same Stage-B source weighting" ask):** `sourceweights.annotate()` now attaches three fields per article — `source_w_rel` (reliability-only; the blocklist lives here), `source_lean`, and the effective `source_w = source_w_rel × LEAN_EXTREMITY_MULT[|lean|]` that contribution and band consume. **DECLARED multipliers: center/unrated 1.0 · |lean| 1 → 0.9 · |lean| 2 → 0.75** (n far too small to fit; values listed for sign-off below). Properties enforced by test: extremity is symmetric (left and right discount identically — direction never flips or directs evidence), lean can never resurrect a blocklisted source, unrated means no adjustment.

**Practical example:** a Guardian article with a decisive YES development used to enter Stage B at weight 1.0 (RSP generally-reliable). It now enters at 1.0 × 0.75 = 0.75 — and because most calibration evidence is Guardian, the α grid-fit rose ~1/0.75 to compensate, leaving FVs nearly unchanged where packets are Guardian-dominated and *relatively* up-weighting BBC/The Hill (center) where sources mix. That relative reweighting — not a level shift — is the substantive change.

**Explainer + feed:** each card's "how the number formed" now appends a lean mix per stance (e.g. "2 items leaned YES … [lean mix: 1 more-biased, 1 centrist]"), tags each source with a lean chip in the tier table, and adds the packet coverage line (left/center/right/unrated counts). Feed items in the left pane carry the same chips (L/LL/C/LR/R).

## Dashboard v3.2 (the UX overhaul)

The v3.1 audit (build notes) found the earlier "2-column" ask had landed at the wrong level — `.cards` was a single stacked column, the evidence feed (`ul.ev`) was cramped inside each card, and the gauges lived inside default-collapsed cards ("donuts buried in dropdowns"). v3.2 fixes all three at the right level:

1. **Page shell = 2 columns.** Left pane: the aggregated news/evidence feed — every public item the extractor read across all 24 markets, deduped cross-market by (title, domain), newest first, each with domain, lean chip, date, link, and chips linking to the market(s) it informed. `position:sticky` with its own scroll. Right pane: everything else. Stacks below ~820px (feed drops under the markets).
2. **Donuts surfaced.** Section 01 is a full-width donut grid: all 24 markets' rings (FV arc; faint band segment; grey mid tick; FV numeral in the center; terracotta ring when flagged) with fv/mid/gap numerals, ⚑/◆ badges and the evidence-quality chip — all visible with **zero clicks**. Deep detail (full time series, FV construction, GDELT chip) stays in the collapsible section-04 cards; the per-card evidence list moved out to the left pane (cards link to it instead — no duplication).
3. **Design language = the deployed site** (https://epsilon-site-changes.vercel.app, inspected 2026-07-05 via fetched HTML/CSS): bg `#242423`, surfaces `#49413c`/`#34322d`, cream text-as-accent `#d1cab7`, muted `#b8b7ad`, hairline borders at text-color ~10%, **terracotta `#cc5c44`** as the only pop color (flags, thin-evidence, negative moves), 8–10px radii, 50px pill buttons, numbered "01." section heads (muted number + light heading), mono uppercase chips, IBM Plex Sans light headings / Geist Mono numerals — all as local font stacks, nothing fetched. The old black + acid-lime theme is fully gone (test-enforced: `#C8FF00`/`#0a0a0a` absent). Note: the site loads Instrument Serif but its visible pages set headings in IBM Plex Sans light — the dashboard follows the *rendered* look.
4. **Extensions.** § 02 Big movers: day-over-day ΔFV per market from **published live snapshots only** (mirrors the append-only ledger; the reconstructed backfill segment is excluded by construction), top-6 by |Δ| with ▲/▼ (▼ terracotta), linking to the card. Currently — correctly — showing the "movers appear once two published daily snapshots exist" note, since 2026-07-05 is the only published date; the mechanism is test-proven on synthetic series. Evidence-quality badge: `thin` (terracotta) / `moderate` / `strong` from n_relevant + band half-width; **strong ≡ the divergence flag's confidence leg** (≥5 relevant/72h AND half ≤ 12pp) so the badge and the flag rule can never disagree.

Verified in headless Chrome: 2-pane layout with sticky feed; 24/24 donuts visible without expanding; sections 03–06 all render in the new language. Mobile honesty note: the first 420px capture *looked* like horizontal overflow — investigation (a scrollWidth probe injected into the page) showed it was a **capture artifact, not a page bug**: current Chrome headless enforces a ~500px minimum window width and crops the PNG to the requested 420px, slicing the right donut column. At the real 500px viewport `scrollWidth == innerWidth` (no page overflow, two clean grid tracks). Defensive CSS was applied anyway (global `svg{max-width:100%}`, `minmax(0,1fr)` mobile column, a scroll container for the wide divergence SVG, donut-cell minimum 172→150px) and the grid arithmetically fits real phone widths (at 420px: two 159px tracks ≥ the 150px minimum); a true <500px device check is listed as a residual item below.

## α refit + the band_mult granularity call (flag for Justin)

Refit via the existing pre-registered path (`scripts/newsagent_hist_backfill.py --fit`, which re-annotates cached Stage-A features — so the composed weights flow through automatically):

| Param | v3.1 | v3.2 | Read |
|---|---|---|---|
| α (evidence weight) | 2.1 | **2.7** | mechanical compensation: dominant-source Guardian now ×0.75, and 2.1/0.75 ≈ 2.8; pooled Brier 0.2214 vs 0.2216 (unchanged-to-slightly-better); evidence still beats prior-only (0.2317) |
| band_mult | 0.5 | **0.5 (unchanged)** | see below |

**The judgment call:** the coverage selector ("rescale band_mult so bucket coverage ≈ nominal 0.8") ran on ~3 scoreable FV buckets, so coverage is quantized to thirds — under the new weights the curve came out 0.25 → 0.667, 0.5 → 1.0, and the closest-to-0.8 rule tie-broke INTO undercoverage (0.25) on pure granularity. The v3.1 findings phrased the intent as "the **narrowest knob at nominal**"; I implemented that reading explicitly (smallest band_mult with coverage ≥ nominal, closest-fallback otherwise), which keeps 0.5 — i.e. public bands keep their v3.1 calibrated width instead of silently halving as a side effect of a UX/bias release. The selector change is 6 lines in `band_mult_coverage()`, commented in place. **Justin: veto or bless** — the forward-ledger rescale at ≥20 settled forecasts (already pre-registered) will have real granularity and supersedes this either way.

Re-publish sequence mirrored v3: `backfill-compute` re-evolved the 5 legacy markets' display series under the composed weights; the 19 day-one markets' evidence states were reset (state file backed up first) so today's publish recomputed them cleanly; then `--stage publish` updated all 24 ledger entries (same-day pre-settlement update through the sf CLI, established idempotent). Headline rows are stable (Becerra −77.9pp, United Russia +33.5pp, Fed-July −35.8pp; still no flag — day-one bands stay wide).

## Assumption ledger ([[CODEX]] § Realism calibration)

**Modeled assumptions:** AllSides categorical ratings ≈ the lean axis (they're one of Ground News's three inputs; categories are coarse and periodically revised — table dated, rating-move notes kept in the module); the DECLARED extremity multipliers (1.0/0.9/0.75) are values, not fits — sign-off pending, α-refit-covered; lean applies to outlet domains only (newsletters/PDFs/WP-CE unrated — same scope rule as reliability); the movers strip reads the published fv_series as the ledger mirror (one point per published day, written by the same publish that writes sf) rather than re-parsing sf files; badge thresholds reuse the declared divergence-confidence bars.

**Live-only unknowns:** whether the lean-composed weighting changes forward calibration (first resolutions: US–Iran meeting 07-17, Fed July 07-29, Hormuz + MOU 07-31 — refit on each per the runbook); band coverage at the forward rescale (supersedes the granularity call above); movers-strip behavior once a real multi-day published history exists; whether lean chips + coverage lines change how readers interpret the explainer (a display question, but the point of the page); a true sub-500px-device render (headless Chrome can't produce one — see the mobile honesty note; analytic fit checked, real-phone check pending).

**Power honesty:** the α refit is the same 355 pairs / 49 markets as v3.1 — the lean change re-weights within it, adds no new information; the pooled Brier delta (−0.0002) is noise, not evidence the lean axis "works". What it buys today is transparency (the explainer) and robustness for when sources broaden beyond the current reliable-tier set — same argument as Scheme A in v3.

## What Justin needs to do

1. **Lean sign-off (new):** the table + declared multipliers above (center/unrated 1.0 · |1| 0.9 · |2| 0.75). Running live (α-refit-covered) — veto reverts to lean_mult ≡ 1.0 with a one-line change + refit.
2. **band_mult selector call (new):** bless the "narrowest knob ≥ nominal" reading (keeps 0.5) or revert to closest-to-nominal (0.25, undercovering on 3-bucket granularity).
3. **AllSides "written NC OK" (radar nicety, now live-relevant):** a short permission email for the NC-licensed ratings use, per the radar's Scheme-C note. Non-blocking (non-monetised + attributed meanwhile).
4. **Unchanged from v3/v3.1:** Scheme-A uncovered-source weight sign-off (ING 0.9 / WP-CE 0.8 / unknown 0.5 / Bloomberg 0.9 / bank desks 0.9 — all still neutral 1.0); `GUARDIAN_API_KEY` export; `GEMINI_API_KEY` for the provider spot-check; July settlements → `sf settle` + slate refresh + `--fit`.
5. **Website:** still parked. The page is built local and site-ready (the colleague lifts `showcase.json`, or the HTML as-is — it now matches the deployed design).

## STRETCH / BACKLOG (marked; nothing blocks the shipped v3.2 slice)

- **News-driven-only universe restriction** + **rates-via-options/futures econ track** (carried from v3.1 — the tractability tags make both natural).
- **Lean-diversity band term** (one-sided-spectrum evidence widens the band): deliberately NOT added now — band coverage was just recalibrated; bundle with the forward-ledger rescale.
- **Cross-spectrum corroboration display** ("confirmed across the spectrum" chip when L and R sources agree directionally) — display-only, cheap.
- **Tone → direction calibration**, **event-driven cadence**, **web-launch automation** (`newsagent/AUTOMATION.md`), **UK re-entry**, **interval coverage into lemma-calibrate** — all carried, unchanged.
- **Movers alerting** (flag |Δ| ≥ threshold in the strip) once a real published history accrues.

## Outputs

- Package: `newsagent/sourcelean.py` (new — verdict, table, mults, labels), `sourceweights.py` (annotate composes rel × lean; per-axis fields), `fvmodel.py` (`source_bias_breakdown` lean mix/coverage/leans; `evidence_quality`), `run_daily.py` (badge wiring), `dashboard.py` (rewritten: 2-pane shell, feed pane builder, donut grid, movers, badges, deployed-site tokens).
- Script: `scripts/newsagent_hist_backfill.py` — band_mult selector = narrowest-at-nominal (6-line change, commented).
- Params: `fv_params.json` — **α 2.7** (refit, composed weights, 355 pairs/49 markets), band_mult 0.5, γ 1.0 + λ/floors/clips declared, unchanged.
- Tests: `tests/test_newsagent_fv.py` — **76 green** (v3.2 adds 11: lean table/mults/labels, annotate composition incl. blocklist-precedence, explainer lean mix + coverage, badge tiers, movers live-only + day-one note, feed dedupe/private-count, 2-pane + donut-grid render, token swap, evidence-moved-out-of-cards, scrub-with-lean end-to-end).
- Artifacts (git-ignored, regenerated): `data/newsagent/showcase/{index.html,showcase.json}` (~250KB self-contained page; JSON now carries `feed`, `movers`, `evidence_quality`, per-item `lean`); fit CSVs + plot refreshed under `data/analysis/{csv_outputs,plots}/news_agent/`; `fv_state.pre-v32.json` backup.
- Ledger: sf-2026-001…024 updated (append-only, same-day pre-settlement via the sf CLI).
