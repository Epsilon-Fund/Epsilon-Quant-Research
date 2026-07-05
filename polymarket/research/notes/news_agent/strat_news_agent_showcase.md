---
title: "News-Agent Fair-Value + Calibration Showcase — thread hub"
created: 2026-07-04
status: shipped as Calibration Observatory — fair-value framing closed by v0+v0b; live measurement loop running since 2026-07-05
owner: justin
project: polymarket
para: project
hubs:
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - news-agent
  - showcase
  - hub
---
# News-Agent Fair-Value + Calibration Showcase (hub)

> Hub backlinks: [[POLYMARKET_BRAIN]] · [[COWORK]] · [[TODO]]

## Plain-English Summary

- **The idea:** a public, non-monetized showcase of Epsilon's own news-informed fair-value probability on high-liquidity politics prediction markets, shown next to the Polymarket mid with a confidence band — with the **calibration/Brier track record over time as the hook**. A showcase of research craft, not a trading strategy.
- **Lineage:** this is the never-run Block J / LLM-forecaster frontier ([[TODO]] § future blocks). It does **not** reopen the closed financial-binary pricing results ([[od_same_day_crypto_pricing_gate_findings]], [[od_equities_index_pricing_scope_findings]]) — those cover crypto/equities binaries vs liquid underlyings, not news-driven politics fair value.
- **Discipline:** no infra before signal — the v0 gate ([[newsagent_v0_gate_preregistration]]) decides whether the view is worth displaying before any dashboard exists. Forecast ledger is append-only and anti-post-hoc (vendored `superforecasting` skill, `SF_BOOK=polymarket`), scored read-only by the first-party `calibrate` skill.

## Current state (2026-07-05, evening — Observatory v2)

**Rebuilt as Epsilon's INDEPENDENT fair value, judged against resolved outcomes; the mid is discovery + context only.** The v0/v0b closures stand untouched (the "our % beats the mid" claim stays CLOSED and displayed). The daily number now comes from a **hybrid two-stage model**: Stage A — a cheap LLM extracts structured features per article (cached per (market, article), prompt-versioned); Stage B — a transparent decayed log-odds model (onboarding prior from the five-perspective ensemble + decisive-signal evidence weighting) whose single evidence weight is **fitted on resolved outcomes** (v0 archive: α=1.4, 53 pairs / 9 markets — a starting point, refit on every settlement). Divergence flag pre-registered: |FV−mid| ≥ 15pp AND band half ≤ 12pp AND ≥ 5 relevant articles/72h — "where our model most disagrees", never an edge claim; the internal edge question (Q-DIV-EDGE) is pre-registered and NOT run. Dashboard restyled to the epsilon-webs1te design language (read-only borrow) with FV+band-vs-mid time series (14-day labeled reconstruction), FV-construction breakdown, divergence layer, reliability panel, interim scoring (ForecastBench convention). Full build record: [[newsagent_observatory_v2_findings]]. Ledger sf-2026-001…005 updated append-only; `calibrate` wired (0 settled yet; first resolution 2026-07-17).

**v2.1 same-day addendum:** the GDELT/BigQuery attention layer is LIVE (creds delivered) — per-market daily GKG volume+tone, burst amplification of Stage-A evidence (γ=1.0 declared after the in-sample grid showed monotone burst-saturation overfit; slow-market guard restated as an α-invariant ±1.5-logit shift cap). Tone displayed, not wired. See [[newsagent_observatory_v2_findings]] § GDELT attention layer.

**v3 (2026-07-05 late, attended build):** sources went multi-outlet — free RSS set (BBC/Sky/Politico/The Hill, title-hash deduped, headline+link display) + a read-only newsletter-inbox module (ING; `display=False` enforced; awaiting Justin's IMAP credential); **Scheme-A source weighting live** (RSP tiers 439 entries + Iffy blocklist 2,040 domains → Stage-B weights; all five wired outlets verify generally-reliable; uncovered-source weights PROPOSED for sign-off: ING 0.9 / WP-CE 0.8 / unknown 0.5 — neutral 1.0 until approved); universe **5 → 24 markets** (ledger sf-2026-001…024); Stage-A v3 adds clarity → band v3 = reliability-weighted dispersion + clarity widening + single band_mult knob with a pre-registered coverage-rescale procedure (`fvmodel.band_coverage`, 0 settled yet); α refit **1.8** on v3 features (γ-sensitivity now flat); dashboard gains per-market gauge + gap-ranked overview grid with sparklines. All attended/manual; web-launch automation is scaffold-only (`newsagent/AUTOMATION.md`); website parked. See [[newsagent_observatory_v3_findings]].

**v3.1 (2026-07-05, second attended session):** the un-run n8n follow-up + new UX/universe/calibration items landed. **Gmail-API read-only OAuth is LIVE** (consent minted → `secrets/gmail_token.json`; ING THINK + ING + **Bloomberg** newsletters feed Stage A; a latent v3 privacy leak — no `display=False` filter at the page boundary — was fixed and regression-tested before the token armed it). **Five public macro-research PDFs** added (JPM Recap/Guide/Brief static, GS Monitor Friday-templated, BofA CMO Monday-templated; disclaimer-strip; headline+link display; silent weekly degrade). **Gemini 2.5 Flash extraction** behind a provider flag + a cache spot-check script (blocked on `GEMINI_API_KEY`; α is Haiku-era — never flip silently). All 24 markets tagged **news-driven vs data-driven** (Fed ×2, Becerra, CA wealth-tax = data-driven, cards say "not news-tractable — structurally blind"; still scored). Per-card **source-bias ratings explainer** (lean × Scheme-A reliability tier). Dashboard rebuilt: default-visible subset + picker + collapsible **2-column cards** (charts | news feed). **Historical backfill executed:** 40 resolved politics/macro markets (post-cutoff resolutions, family-capped, canary-checked — 0 known / 2 suspected, sensitivity-clean), 302 lookahead-free packets (0/2,040 violations), 1,872 features via 16 Haiku subagent batches, 40 Sonnet priors → **α refit 2.1 on 355 pairs / 49 markets** (was 53/9) and the pre-registered coverage rescale set **band_mult 0.5** (v3 bands ~2× too wide; realized freq rides the upper band edge — model slightly NO-leaning, recheck at first forward rescale). 65 tests green. See [[newsagent_observatory_v31_findings]].

**Open items for Justin:** (1) Scheme-A sign-off — one table: ING 0.9 / WP-CE 0.8 / unknown 0.5 + **Bloomberg 0.9 / bank research desks 0.9** (all neutral 1.0 until then); (2) export `GUARDIAN_API_KEY` (registered; demo key worked for the backfill but is the fragile path); (3) `GEMINI_API_KEY` (free tier) to unblock the provider spot-check; (4) `ANTHROPIC_API_KEY` + cron-env `GOOGLE_APPLICATION_CREDENTIALS` for the future unattended switch; (5) website handoff stays parked; (6) July settlements (`sf settle`): US–Iran meeting 07-17, Fed 07-29, Hormuz + MOU 07-31 — refit via `scripts/newsagent_hist_backfill.py --fit`.

## Notes in this cluster

- [[newsagent_v0_gate_preregistration]] — locked metric + cheapest falsifier for the v0 gate (incl. Amendment 1: packet source).
- [[newsagent_v0_gate_findings]] — v0 results: **STOP verdict**, mechanism diagnosis, v0b redesign proposal, STRETCH/BACKLOG.
- [[newsagent_repo_data_radar_findings]] — external radar (repos, news APIs, bias datasets) + source-weighting proposal for sign-off.
- [[newsagent_observatory_v2_findings]] — the v2 rebuild: hybrid Stage-A/Stage-B FV model, outcome calibration (α fit), divergence flag rule + pre-registered Q-DIV-EDGE, redesigned dashboard, cost ledger, GDELT attention layer (v2.1).
- [[newsagent_observatory_v3_findings]] — the v3 build: RSS + newsletter sources, Scheme-A bias weighting (+ sign-off proposal), 24-market universe, clarity/band rigor + coverage procedure, gauge/overview-grid charts, attended run record.
- [[newsagent_observatory_v31_findings]] — the v3.1 build: Gmail-OAuth newsletters live (+ privacy fix), macro-research PDFs, Gemini provider flag, tractability tags, source-bias explainer, dashboard UX rework, and the 355-pair historical backfill refit (α 2.1, band_mult 0.5).

## Where things live

- Code: `polymarket/research/scripts/newsagent_*.py` (v0) and `polymarket/research/newsagent/` (v1 package, if gated in).
- Raw pulls (append-only): `polymarket/research/data/newsagent/`.
- Result CSVs: `polymarket/research/data/analysis/csv_outputs/news_agent/`.
- Plots: `polymarket/research/data/analysis/plots/news_agent/`.
- Forecast ledger (git-ignored runtime, append-only): `polymarket/research/data/superforecast/`.
