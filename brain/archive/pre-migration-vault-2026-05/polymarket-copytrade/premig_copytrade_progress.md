---
title: "Polymarket copytrade — progress log (May 2026)"
created: 2026-05-15
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

# polymarket copytrade — progress log

Append-only weekly rollup. Newest at top.

---

## week of 2026-05-11

- **2026-05-11 — Phase 4 → Phase 5 transition shipped (multi-session catch-up).** Bankroll point-in-time computation deployed (`bankroll_timeseries.parquet`, 429M rows / 12GB; new canonical `rolling_bankroll_usd_30d`). `phase5_design.md` locked: walk-forward, 3 cohorts × 4 resolution buckets × 2 sizing rules. Stage 1 (72 backtests) surfaced threshold + slippage issues → Stage 1.5 spec finalised (percentile + floor + Top-K=10, fixed-% sizing dropped to 1%). Slippage model = next-fill-within-15s-5min/$3 fallback. Conviction filter dropped. ADR 0001 v0.3 patched (bankroll gap closed). 4 methodology lessons logged.
- **2026-05-15 — Weather FTC TP analysis: shelve (taker) → deployable (passive).** Five bookkeeping bugs fixed in `ftc_tp_sizing.py` (compounding-convention mismatch was 4 orders of magnitude wrong: 2,272,346% → 628.6%). Slippage tooling shipped (`lookup_next_fills_batch`, fill scenarios, self-discrediting diagnostic). Taker verdict: shelve. WS-passive revision: `(p_in=0.50, p_out=0.90)` deployable at +6% ROI per fill, ~6 fills/day; queue priority determines true performance. 10 methodology lessons captured. (scheduled ingest from `inbox/2026-05-15_pm-data.md`)
- **2026-05-16 — Framework pivot: cohort backtests → per-trader copy-execution audit.** Phase 5 cohort framework on ice (wrong unit-of-analysis). `scripts/domah_copy_audit.py` ships (~90s per leader). 7 leaders audited; cross-leader signal at this granularity is thin (2 shared cells). `phantom_position_score` retired as primary arb filter — Pool C was 91% contaminated. New `split_position_signature` metric catches architectural uncopyability (10 of top-50 by PnL). Recency + hold-to-resolution are new primary copyability predictors (lifetime PnL ρ = −0.21). `04-cohort-pools.md` and ADR 0001 v0.2 superseded by per-leader framing. 6 methodology lessons captured. (scheduled ingest from `inbox/2026-05-16_pm-data.md`, backfill)

---

## week of 2026-05-04

- Set up cowork structure for polymarket subproject (README, TODO, research/, decisions/).
- Captured **data-track state dump** in `research/02-data.md`: Phase 3 complete (2.58M-trader parquet built from 1.064B raw fills; $0 self-consistency on 270M closed positions). Ready for Phase 4 cohort pool materialisation.
- Captured **exec-track state dump** in `research/01-execution.md`: engineering complete, 214 tests passing, fake-venue end-to-end verified (real RTDS feed → classifier → risk → fake venue → fills, math correct). Three operational tasks (PLAN.md sync, creds, smoke) stand between today and first real-money fill. UK is geo-blocked for order submission; needs VPN/VPS.
- Drafted **end-to-end system design** in `research/03-system-design.md` stitching both tracks. Identified four integration gaps: bankroll mismatch (point-in-time vs lifetime), time-window mismatch (30d vs lifetime), NegRisk gap (both sides), multi-leader orchestration.
- Wrote **first ADR** `decisions/0001-leader-rankings-schema.md` — interface contract for `leader_rankings.parquet`.
- Cascaded updates: README status table now reflects all three tracks; TODO restructured around exec smoke path + data Phase 4 + integration debt; glossary + stack inventory now cover the exec stack (RTDS, kernel, py-clob-client, journal-replay, etc).
- **2026-05-10 — formulas reference shipped.** Claude Code wrote `polymarket-copy/docs/METRICS_REFERENCE.md` (1019 lines, 74 columns, 4 self-consistency invariants, 13 known limitations); mirrored to `research/notes/formulas.md`. Cohort pools are already materialised at `data/cohorts/*.parquet` — 14,053 union rows, 947 addresses qualify for 3+ pools, max overlap = 4. ADR 0001 v0.2 patched: source-column mapping is now concrete with trustworthiness ratings; 6 of 14 schema columns are STRONG passthroughs from `traders.parquet`, 5 are 30d-rolling gaps, 2 are SUSPECT-context-dependent (`estimated_bankroll_usd`, Sharpe-derived `rank_score`). Major finding: 96.4% of closed positions held to resolution — implications for exec's mirror-exit logic flagged in TODO.
- **2026-05-10 — RESEARCH_FINDINGS + data README mirrored.** `research/notes/research-findings.md` and `research/notes/data-readme.md`. New context I didn't have before: per-pool counts (A=3,304 / B=556 / C=113 / D=2,152 / E=225 / F=7,703), full overlap matrix, 5 named candidates with full per-candidate dossiers, and the fact that `profile_trader()` already exists at `data_infra/trader_profile.py` (was on my TODO as "implement"; mistake on my part).
- **2026-05-10 — `04-cohort-pools.md` shipped.** Per-pool archetype commentary, exec-readiness commentary, exec-readiness filter for v1 smoke (6 conditions: `negrisk_volume_share < 0.3`, `phantom_position_score < 2.0`, `style_role_balance > 0.5`, `active_days > 180`, multi-pool ≥ 2, NOT in Pool C), and per-candidate exec colour. Recommended smoke target: **`0x6a72f61820b2…`** — passes all 6 filters, top-of-leaderboard ($14.95M PnL), low NegRisk (0.05), maker-conviction (role_balance 0.84), 4,359 positions over 302 days. Three of the five candidates from RESEARCH_FINDINGS are deferred for v1: `0xd38b71f3` (taker-heavy → needs `current_book` mode), `0x17db3fcd93ba` (small sample, defer to backtest), `0x629bc4a1e53e` (84% NegRisk, blocked on exec NegRisk handling).

## historical (pre-cowork) — data track

*Backfilled from 2026-05-09 state dump.*

- **Phase 1 — data infrastructure**: `data_infra/{goldsky,gamma,views,operator_denylist}.py`, `sql/views.sql` (canonical views: `raw_trades`, `trader_actions`, `trader_actions_orphan`, `traders_raw`, `traders_filtered`).
- **Phase 2 — closed positions**: 270M-row parquet, one row per `(address, market_id, outcome_index)`. Self-consistency check: total realised PnL = $0.00 across all closed positions.
- **Phase 3 — trader aggregation**: 2.58M-row `traders.parquet` with activity, position-level + market-level PnL, style profile, phantom-position score, operator-like flag, lifetime-peak bankroll proxy.
- **Data ingest**: 1.064B raw fills total — warproxxx CSV seed (Dec 2022 → Oct 2025, ~151M rows) + Goldsky GraphQL delta (Oct 2025 → Apr 23 2026, ~913M rows). Schema-uniform across shards.

## historical (pre-cowork) — execution track

*Backfilled from 2026-05-09 state dump.*

- **Architecture**: 7 modules (`watcher`, `signal`, `risk`, `mirror`, `journal`, `config`, `cli`) + vendored kernel (`_kernel/`) from `midas/executor/`. Single Python process, threading-based, no asyncio.
- **State model**: no database. Position state, dedup, in-flight orders all rebuilt by replaying the journal (today + yesterday) at startup. Journal is the database.
- **Risk pipeline**: 7 independent breakers as pure functions; first veto wins; only kill-switch halts.
- **HTTP client substitute**: replaces a real wire-encoding bug in the vendored kernel that would have caused 100% rejection on real venue. 30 unit tests covering decoding/transport/round-trip.
- **Verification**: RTDS connection (32 fills, no drops), watcher-journal cross-checked against Polymarket Data API, classifier all-branches, all 7 risk breakers, fake-venue 2-fill end-to-end with correct sizing math.
- **PoC scope**: single hardcoded leader (`POLYMARKET_LEADER_ADDRESS`), fixed-USD sizing, mirror exits only, `leader_fill` pricing default, IOC-on-the-wire (FOK requested).
