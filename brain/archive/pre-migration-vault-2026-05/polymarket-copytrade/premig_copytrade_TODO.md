---
title: "Polymarket copytrade — todos (May 2026)"
created: 2026-05-18
archived: 2026-09-09
status: archived — pre-migration vault (May 2026); historical, not current
owner: justin
project: copytrade
para: archive
hubs:
  - pre-migration-vault-2026-05
tags:
  - archive
  - pre-migration-vault
  - copytrade
---

> **ARCHIVED — PRE-MIGRATION VAULT (May 2026).** Historical record, imported 2026-09-09 from the standalone Cowork vault that predates this repo becoming the brain. Index: [[pre-migration-vault-2026-05]].
> **DEPRIORITISED (2026-08-25).** The copy-trading thread is not the active research thread (that is [[strat_market_making]]). It is not archived either — its execution/signing infrastructure is live and shared with the market-making machinery. Pick this thread back up only with Justin.

# polymarket copytrade — todos

> **Framework note (2026-05-18).** Per-leader copy-execution audit framework replaces the cohort-pool framing as of the 2026-05-16 pm-data pivot. Many items below from earlier ingests are now superseded — consolidated to `## later — deferred (cohort framework on ice)`. The "now" sections reflect the current paradigm. Authoritative source: `inbox/_archive/2026-05/2026-05-16_pm-data.md` and `research/notes/methodology-lessons.md` 2026-05-16 entries.

## now — strategic decision

- [ ] **Smoke target re-selection.** Original `04-cohort-pools.md` recommendation was `0x6a72f61820b2…` on a passes-all-6-exec-readiness-filters basis. The audit framework's empirical evidence (lifetime-PnL ↔ copyability Spearman ρ = −0.21) suggests this was the wrong selection heuristic. Re-rank candidates by recency × hold-to-resolution × audit-deployable-cell-count, *excluding* `split_position_signature > 60%` (drops 10 of top-50, including `0x6a72f61820b2`'s neighbours on the leaderboard). Likely new candidates: Domah's `macro / maker / 18-24` cell (currently the highest n_deployable_cells = 3 audited leader) or any leader from the next 5 audits that lands ≥3 cells.
- [ ] **Decide on paper-trade-first vs $10-smoke-first.** New option from per-leader audit: live paper-trading on Domah's `macro / maker / 18-24` cell with logging only, no capital. Lets you measure real queue priority and fill mix without smoke-runbook overhead. Recommendation: paper-trade Domah's deployable cell while audit framework matures to ≥12 leaders.

## now — open

> 2026-05-19 scheduled-ingest stamp: items from `inbox/2026-05-15_pm-data.md` (sections 4+5) and `inbox/2026-05-16_pm-data.md` (sections 4+5) were hand-ingested into the appropriate "now —" subsections prior to the nightly task firing. Source files archived to `inbox/_archive/2026-05/` on 2026-05-19 with no new TODO duplicates added.

- [ ] **Small-live deployment: weather FTC TP under WS-passive at (p_in=0.50, p_out=0.90)** for 2-4 weeks. Measure actual fill mix vs optimistic-strict bracket (+0.87% to −2.5% ROI on canonical 0.60/0.90 — the deployable range is determined by queue priority on the exit ask). Focus cities: Seoul, Shanghai, Tokyo, Wellington, London. Capital small enough that 2.5% drawdown is an acceptable learning cost. Backtested expectation at (0.50, 0.90): ~6 fills/day, +6% ROI per filled trade optimistic. (from `inbox/2026-05-15_pm-data.md` v2)
- [ ] (housekeeping) Extract the slippage diagnostic + fill-scenario tooling pattern from `data_infra/weather_analysis.py` into a shared `data_infra/slippage_proxies.py`. Include both the next-fill (taker) helpers AND the WS-passive helpers (`passive_pnl_from_audit`, `eval_pair_passive`, `grid_passive`). Phase 5 cohort backtests can use the same self-discrediting diagnostic + execution-model split. (from `inbox/2026-05-15_pm-data.md` v2)
- [ ] (optional, lower priority now) `fallback_cents` sensitivity sweep ∈ {1, 2, 3, 5, 8} — only informative for the *taker* execution path. Less critical now that the deployable signal lives under passive. (from `inbox/2026-05-15_pm-data.md`)

## now — exec (path to first real-money fill)

1. [ ] **PLAN.md sync + snapshot commit + tag.** Marks engineering-complete state. ~10 min.
2. [ ] **Slack message to colleague** about the kernel encoding bug we worked around (so he knows before midas's executor goes live).
3. [ ] **Get Polymarket credentials into `.env`.** Private key from UI → `derive_api_keys.py` → fill `.env`. Read-only auth check (fetch open orders, expect empty list) before any submission.
4. [ ] **Pre-flight on smoke target.** Profile `0x6a72f61820b2…`'s current open positions via Gamma + recent fills via RTDS pre-subscription. Confirm no NegRisk markets in the active window.
5. [ ] **First real-money smoke** per `scripts/SMOKE_REAL.md` against `0x6a72f61820b2…`. VPN on, `MAX_REAL_ORDERS=1`, `REQUIRE_OPERATOR_CONFIRM=true`, `SIZING_USD=10`. Cross-check on Polymarket UI.

## now — data / research (per-leader audit framework)

- [ ] **Audit 5 more leaders** using `scripts/domah_copy_audit.py`. Sort top-50 by `active_days_last_90d` desc + `hold_to_resolution_share` desc, exclude `split_position_signature > 60%`. ~90 sec per leader. Goal: reach 12 audits before re-checking cross-leader intersection. (from `inbox/2026-05-16_pm-data.md`)
- [ ] **Re-check cross-leader intersection at 12 audits.** Current state: only 2 narrow shared cells across 7 audits — too thin for signal vs noise discrimination. (from `inbox/2026-05-16_pm-data.md`)
- [ ] **If `other / taker / afternoons` cell stays deployable for ≥3 leaders**, design a cohort-level signal there. Currently 2 leaders. (from `inbox/2026-05-16_pm-data.md`)
- [ ] **Mirror new repo MDs into cowork** when the user has a moment: any new files in `polymarket-copy/notes/` or `polymarket-copy/docs/` (specifically anything about `traders_directionality.parquet`, `split_position_signature`, or the audit framework). `cp` into `polymarket-copytrade/research/notes/`.
- [ ] **Weather strategy WS-passive live-look diagnostic** (carried from 2026-05-16 journal). 5-minute manual observation per active weather market: track best_bid/best_ask after a cross, classify as track-down-feasible / wide-spread-sticky / real-crash. 3-4 markets needed to characterise regime dominance.
- [ ] (lower-priority housekeeping) Extract slippage diagnostic + fill-scenario tooling from `data_infra/weather_analysis.py` into shared `data_infra/slippage_proxies.py`. (from `inbox/2026-05-15_pm-data.md` v2)
- [ ] (lower-priority) `fallback_cents` sensitivity sweep ∈ {1, 2, 3, 5, 8} — taker-execution-path only. (from `inbox/2026-05-15_pm-data.md`)

## now — data / metric infrastructure

- [ ] **Document `traders_directionality.parquet`** when the user mirrors the MDs in. New canonical arb-detection / directionality source replacing `phantom_position_score < 2.0` filter. Will likely warrant its own section in `notes/formulas.md`.
- [ ] **Document `split_position_signature`** trader-level metric. Catches `splitPosition`-based directional construction (the `0xd38b71f3` failure mode). 10 of top-50 traders by lifetime PnL fail this filter.
- [ ] **30d-rolling metric versions** still useful for ADR 0001 interface, but lower priority now that the schema needs a v0.5 redesign around per-leader audit output. `active_days_last_90d` and `hold_to_resolution_share` are the new primary predictors.


## next — exec hardening (post-smoke)

- [ ] **resolution-handler path** in exec — load-bearing for any leader from Pool E (patient_accumulators) or anyone with high held-to-resolution rate. The 96.4%-overall finding means the bot's "leader sells X% → bot sells X%" mirror-exit rarely fires; the bot needs to recognise market resolution and close the journal entry without an `OrderFilled` event. Acceptable to skip for the smoke; required before scaling.
- [ ] tear down safety harness (`MAX_REAL_ORDERS`, operator-confirm) once stable.
- [ ] provision VPS in non-blocked region (US East / Frankfurt / Tokyo candidates) for unattended operation.
- [ ] add `POLYMARKET_LEADER_RANKINGS_PATH` env var + read-on-refresh logic (gates multi-leader).

## later — deferred (cohort framework on ice, per 2026-05-16 pivot)

The following items were authored under the cohort-based copy-trading paradigm. They're preserved here as historical record and as potentially-useful work IF the per-leader audit framework matures into a cohort signal (≥3 leaders sharing a deployable cell). Don't action without re-validating the underlying thesis.

- [ ] ~~Stage 1.5 backtests outputs analysis~~ — Stage 1.5 is on ice; results were untrustworthy under cohort framing
- [ ] ~~Edge-interpretation notebook~~ — designed for Stage 1.5 outputs that won't be re-run as-is
- [ ] ~~Cohort-profile pre-analysis~~ — superseded by per-leader audit framework
- [ ] ~~`cohorts/exec_smoke_candidates.parquet` 6-filter shortlist~~ — `phantom_position_score < 2.0` filter is partially obsolete; replaced by fill-concentration metrics in `traders_directionality.parquet`
- [ ] ~~6 four-pool qualifiers as derived cohort~~ — cohort-robust under old framing but empirically doesn't predict copyability
- [ ] **(still useful)** pull `notes/validation_report.md`, `notes/api_reconciliation_v1.md`, `notes/profile_domah.md` into cowork from repo
- [ ] ~~Walk-forward driver for cohort-selection robustness~~ — paradigm changed; future driver should be per-leader audit, not cohort-level WF
- [ ] ~~CPCV harness for cohort-selection robustness~~ — same
- [ ] ~~Hypothesis battery on A vs B out-of-sample edge, 3-pool predictive vs 1-pool, etc.~~ — these were cohort-shaped hypotheses; the audit framework's empirical answer is "weak cross-leader signal until ≥12 audits"

## next — Phase 5 (backtesting) — replaced by per-leader audit work above

## later

**v2 deferred**
- [ ] index NegRisk merge/split events from Polygon (true position-level PnL — closes phantom-score gap).
- [ ] mark-to-market on open positions (needs CLOB price history).
- [ ] true running peak position size (replace `peak_fill_abs_token` proxy).
- [ ] markets parquet snapshot pinning per analysis.
- [ ] composite NegRisk handling on exec side (multi-outcome position keying, rebalance detection) — unlocks Pool C members and `0x629bc4a1e53e…` as exec targets.
- [ ] multi-leader cohort orchestration (RTDS multi-subscribe, per-leader weights, cohort risk semantics).
- [ ] `style_median_fill_size_usd` via sampled pass.

## done
- [x] cowork scaffolding (2026-05-09)
- [x] data Phase 1 — data infrastructure (`goldsky.py`, `gamma.py`, `views.py`, `operator_denylist.py`, `sql/views.sql`)
- [x] data Phase 2 — `closed_positions.parquet` (270M rows, $0 self-consistency)
- [x] data Phase 3 — `traders.parquet` (2.58M addresses, full metric suite)
- [x] data Phase 4 — six cohort pools materialised at `data/cohorts/*.parquet` (3,304 / 556 / 113 / 2,152 / 225 / 7,703); 14,053 union rows; 947 in 3+ pools; 6 in 4+; 0 in 5+
- [x] `profile_trader()` exists at `data_infra/trader_profile.py` (per data-side README)
- [x] cohort cross-pool diagnostics — overlap matrix in `04-cohort-pools.md`
- [x] 5 manual-inspection candidates surfaced in RESEARCH_FINDINGS
- [x] exec engineering — 7 modules + vendored kernel, 214 unit tests, fake-venue end-to-end pass
- [x] data state dump → `research/02-data.md` (2026-05-09)
- [x] exec state dump → `research/01-execution.md` (2026-05-09)
- [x] end-to-end design → `research/03-system-design.md` (2026-05-09)
- [x] interface ADR v0.1 → `decisions/0001-leader-rankings-schema.md` (2026-05-09)
- [x] **formulas reference** — `polymarket-copy/docs/METRICS_REFERENCE.md` written; mirrored to `research/notes/formulas.md` (2026-05-10)
- [x] interface ADR v0.2 — concrete source-column mapping with trustworthiness ratings (2026-05-10)
- [x] **research findings + data README mirrored** to cowork notes (2026-05-10)
- [x] **`04-cohort-pools.md` written** — six pool spec, archetype commentary, exec-readiness filter, candidate shortlist with exec colour, smoke target proposed (2026-05-10)
- [x] **point-in-time bankroll computation** — `bankroll_timeseries.parquet` (429M rows, 12GB) + new `rolling_bankroll_usd_30d` column on `traders.parquet`. Old `est_bankroll_usd_30d_max_approx` renamed `est_bankroll_lifetime_peak_deprecated`. Phase 3 bankroll bug surfaced + fixed (placeholder-date filter inconsistency inflated whale peaks 5-7×). Sanity checks pass. **Unblocks multi-leader integration from data side.** (2026-05-11)
- [x] **Phase 5 design locked** in `phase5_design.md`: walk-forward (expanding IS, monthly refresh), 3 cohorts (B / B∩C / E), 4 resolution buckets (2d/7d/30d/60d), 2 sizing rules (fixed-% + leader-proportional). Next-fill slippage model (15s-5min from different trader, $3 fallback). (2026-05-11)
- [x] **Stage 1 ran** (72 backtests). Surfaced 2024-cohort emptiness + slippage-drowning issues; led to Stage 1.5 spec (percentile + floor + Top-K=10). (2026-05-11)
- [x] **ADR 0001 v0.3** — bankroll graduates SUSPECT → STRONG; phantom baseline calibration corrected; multi-leader gating closed from data side. (2026-05-11)
- [x] **Weather FTC TP analysis executed** — 5 bookkeeping bugs fixed in `ftc_tp_sizing.py` (incl. 4-orders-of-magnitude compounding bug). Slippage tooling shipped (`lookup_next_fills_batch`, fill scenarios, self-discrediting diagnostic). Initial verdict under taker execution: shelve. (2026-05-15)
- [x] **Weather FTC TP analysis — WS-passive revision** — `passive_pnl_from_audit`, `eval_pair_passive`, `grid_passive` helpers + `scripts/passive_analysis.py`. Revises the verdict: under WS-passive limit-posting via CLOB WebSocket, **`(p_in=0.50, p_out=0.90)` is deployable** (~6 fills/day, +6% ROI optimistic exit; -2.5% strict exit on canonical 0.60/0.90 — true performance bounded by queue priority). Two new methodology lessons captured (execution-model-as-first-class; queue-priority-as-dominant-deployability-question). (2026-05-15)
