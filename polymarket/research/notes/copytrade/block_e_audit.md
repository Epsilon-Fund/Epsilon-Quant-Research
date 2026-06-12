---
title: "Block E Audit: Existing Wallet / Competition Analysis Coverage"
created: 2026-06-05
status: generated
owner: justin
project: polymarket
para: project
hubs:
  - COWORK
tags:
  - research
  - copytrade
---
# Block E Audit: Existing Wallet / Competition Analysis Coverage
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


## Summary

- Scope: Block E Audit: Existing Wallet / Competition Analysis Coverage in the copytrade area.
- Existing takeaway/status: Inventory audit of existing wallet, operator-denylist, competition, and copy-execution coverage. It records what exists before deeper Block E work and flags that several relevant copyability/wallet-analysis artifacts were untracked or gitignored at the time of the audit.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
**Audit date:** 2026-05-27  
**Scope:** inventory only. I did not run new wallet analysis or modify the operator denylist.  
**Important git caveat:** several of the most relevant copyability and wallet-analysis files are currently untracked or gitignored in this worktree, so they have no last commit date. For those, I report the local file mtime or generated-artifact timestamp.

## Section 1: What Exists

### Operator Detection And Filtering

| path | what it does | inputs | outputs | recency |
|---|---|---|---|---|
| `polymarket/research/data_infra/operator_denylist.py` | Defines explicit operator/MM/HFT address sets and the reusable `OPERATOR_ADDRESSES` frozenset. Also contains an `is_operator_like(...)` heuristic helper. | Hardcoded addresses sourced from the validation report. | 12 categorized addresses: 2 relayers, 7 pure MM bots, 3 HFT. | tracked, last commit 2026-05-18 (`57172b0`). |
| `polymarket/research/notes/overview/data_quality/validation_report.md` | Documents the original operator-detection scan and the rationale for clusters A/B/C. | `data/trades/trades_delta_shard*.parquet`, `trades_seed.parquet`, `data/markets/markets_2026-05-06.parquet`. | Markdown report generated 2026-05-08; includes candidate denylist and criteria. | tracked, last commit 2026-05-18 (`57172b0`). |
| `polymarket/research/scripts/validation/02_operator_detection.py` | Scans top 50 addresses by maker+taker fills, computes maker/taker counts, distinct counterparties, markets, volume, sub-second clustering, and an operator scorecard. | Raw trade parquet views from `scripts/validation/_common.py`. | stdout tables used in `validation_report.md`. | tracked, last commit 2026-05-18 (`57172b0`). |
| `polymarket/research/scripts/build_traders_table.py` | Builds `data/traders.parquet`, including `is_operator_like` from hardcoded denylist plus heuristics. | `data/closed_positions.parquet`, `joined_fills`, `OPERATOR_ADDRESSES`. | `data/traders.parquet`; docs say 2,576,698 rows, 2,572,665 after `NOT is_operator_like` (4,033 flagged). | tracked, last commit 2026-05-18 (`57172b0`); file currently modified. |
| `polymarket/research/sql/views.sql` | Defines `traders_filtered AS SELECT * FROM traders_raw WHERE NOT is_operator_like`. | `data/traders.parquet`. | Standard filtered trader view used by cohort selection. | tracked, last commit 2026-05-18 (`57172b0`). |
| `polymarket/research/docs/METRICS_REFERENCE.md` | Documents how `is_operator_like` works and says `traders_filtered` drops 4,033 rows. | Derived parquet schema and source references. | Metrics reference. | untracked; local mtime 2026-05-18. |

### Copy-Trading Research And Wallet Profiling

| path | what it does | inputs | outputs | recency |
|---|---|---|---|---|
| `polymarket/research/README.md` | High-level copy-trading pipeline overview: data infra, closed positions, trader panel, cohort exploration, backtesting backlog. | Repo artifacts and prior phase outputs. | Orientation doc; reports 12 operator-shaped addresses, 2.576M traders, 9,788 cohort qualifiers. | tracked, last commit 2026-05-18 (`57172b0`); file currently modified. |
| `polymarket/research/RESEARCH_FINDINGS.md` | Consolidated findings across phases 1-4: data shape, operator clusters, trader metric distributions, cohort pools, manual DD candidates. | Validation report, `traders.parquet`, cohorts, diagnostics. | Research log with top candidate wallet profiles. | untracked; local mtime 2026-05-18. |
| `polymarket/research/scripts/build_traders_table.py` | Per-wallet activity, PnL, style profile, drawdown, bankroll approximation, operator flag. | `closed_positions.parquet`, `joined_fills`. | `data/traders.parquet`. | tracked, last commit 2026-05-18 (`57172b0`); file modified. |
| `polymarket/research/scripts/build_traders_directionality.py` | Rule-based per-wallet behavior classifier replacing `phantom_position_score` as arb-vs-directional discriminator. Produces `primary_style` = `pure_directional`, `two_sided_directional`, `arb_like`, `mixed`, `insufficient_data`. | `data/closed_positions.parquet`, `OPERATOR_ADDRESSES`. | `data/directionality_classification/traders_directionality.parquet`; population docs. | untracked; local mtime 2026-05-18. Generated artifact mtime 2026-05-16. |
| `polymarket/research/data/directionality_classification/directionality_metric_distributions.md` | Documents directionality metric distributions and thresholds. | `traders_directionality.parquet`. | Markdown distribution report. | gitignored data artifact; mtime 2026-05-16. |
| `polymarket/research/data/directionality_classification/contamination_crosstabs.md` | Shows old Pool C was heavily contaminated: 91.2% of old NegRisk specialist pool looked directional, not arb. | `traders_directionality.parquet`, old Pool C recomputation. | Markdown crosstabs. | gitignored data artifact; mtime 2026-05-16. |
| `polymarket/research/scripts/build_copyability_metrics.py` | Builds per-wallet sidecar for candidate prefiltering. Adds fragmentation, hold-to-resolution, split-position signature, family concentration, recent activity, audit status, deployable-cell count. | `closed_positions.parquet`, `traders.parquet`, `traders_directionality.parquet`, raw fills, markets, audit reports. | `data/copyability_candidates/traders_copyability_metrics.parquet`. | untracked; local mtime 2026-05-18. Generated artifact mtime 2026-05-18. |
| `polymarket/research/data/copyability_candidates/copyability_metric_distributions.md` | Documents copyability sidecar schema and distributions over 572,205 non-operator traders with >50 closed positions. | `traders_copyability_metrics.parquet`. | Markdown report, including top-50 PnL table and known trader checks. | gitignored data artifact; mtime 2026-05-18. |
| `polymarket/research/scripts/build_cohorts.py` | Materializes six cohort pools from `traders_filtered`, now with directionality sidecar join for Pool C. | `traders_filtered`, `closed_positions.parquet`, `traders_directionality.parquet`. | `data/cohorts/{pool}.parquet`. | untracked; local mtime 2026-05-18. |
| `polymarket/research/notebooks/cohort_exploration.ipynb` | Interactive cross-pool diagnostics: overlap matrix, style scatter, top-20 union, metric correlations, profile examples. | `data/cohorts/*.parquet`, `traders.parquet`, `profile_trader`. | Notebook analysis and charts. | untracked; local mtime 2026-05-18. |
| `polymarket/research/data_infra/trader_profile.py` | Generates per-address research dossier: headline metrics, style, capital footprint, monthly PnL, market mix, holding distribution, activity cadence, cohort positioning. | `traders.parquet`, `closed_positions.parquet`, raw fills, markets, cohorts. | Dict or markdown profile, e.g. `notes/copytrade/profile_domah.md`. | untracked; local mtime 2026-05-18. |

### Copy-Execution Audits

| path | what it does | inputs | outputs | recency |
|---|---|---|---|---|
| `polymarket/research/scripts/domah_copy_audit.py` | Fill-mirroring audit for a leader under multiple execution branches: role-mirrored, pure taker, pure maker, optimistic/realistic maker fills. Can run on any leader. | Raw trades, markets, `closed_positions.parquet`. | `data/analysis/*_audit_{fragments,positions,family,slices,diagnostics,report}.parquet/md`. | untracked; local mtime 2026-05-18. |
| `polymarket/research/scripts/run_leader_audits_batch_v2.py` | Runs the Domah audit pipeline on five additional leaders. | Leader address list plus augmented family keywords. | Per-leader audit directories under `data/analysis/leader_*`. | untracked; local mtime 2026-05-18. |
| `polymarket/research/scripts/cross_leader_analysis.py` | Compares Domah and `0xee00ba...` and writes cross-leader deployable-cell sections. | Existing audit parquet outputs. | Appended/derived cross-leader report artifacts. | untracked; local mtime 2026-05-18. |
| `polymarket/research/scripts/cross_leader_synthesis_v2.py` | Consolidates 7 audited leaders into leader x family x role x hour cells; computes deployable intersections; updates copyability deployable counts. | Audit artifacts for Domah, `ee00ba`, and 5 batch leaders. | `data/analysis/cross_leader_synthesis_v2.md`, `*_cells.parquet`, `*_anchors.parquet`. | untracked; local mtime 2026-05-18; report mtime 2026-05-18. |
| `polymarket/research/data/analysis/leader_dthreed8b71_investigation/dthreed8b71_strategy_profile.md` | Deep strategy characterization of `0xd38b71f3...`: identifies split-and-sell synthetic directional sports strategy and marks fill-mirror copying structurally broken. | Trader fills, `closed_positions.parquet`, markets, HFT operator comparators. | Markdown strategy profile plus per-market/per-outcome/fill artifacts. | gitignored data artifact; mtime 2026-05-16. |
| `polymarket/research/scripts/domah_family_validation.py` | Validates and extends market-family heuristic used in Domah audits. | Existing Domah audit fragments/positions/family tables. | `data/analysis/domah_followups/family_heuristic_validation.md`. | untracked; local mtime 2026-05-18. |
| `polymarket/research/scripts/domah_politics_deep_dive.py` | Slices Domah politics copyability by lifecycle, hour, position size, market concentration, fill timing, and spotlight markets. | Domah audit fragments/positions. | `data/analysis/domah_followups/politics_deep_dive.md`. | untracked; local mtime 2026-05-18. |

### Wallet-Level Analysis In Other Contexts

| path | what it does | inputs | outputs | recency |
|---|---|---|---|---|
| `polymarket/research/notes/overview/market_maps/esports_latency_trader_screen.md` | Screens esports wallets for possible latency-arb fingerprints using a price-snap proxy. Produces address-level and cell-level rankings. | Raw fills, markets snapshot, `closed_positions.parquet`, `traders.parquet`. | Markdown report plus `data/analysis/esports_latency_traders/*.parquet`. | untracked note mtime 2026-05-20; artifacts mtime around 2026-05-20. |
| `polymarket/research/notebooks/esports_latency_trader_screen.ipynb` | Visual companion for the esports wallet screen. | Cached parquet screen artifacts. | Notebook visuals: aggregate score leaderboard, stale-entry subset, PnL/volume, concentration. | untracked; mtime 2026-05-20. |
| `polymarket/research/scripts/dali_market_universe_screen.py` | Historical market-universe screen for Dali price-path work. Builds market-level fill aggregate and candidate screens. | Raw fills and Gamma market metadata. | `data/analysis/dali_*market_screen.csv/parquet`. | untracked; local mtime 2026-05-23. |
| `polymarket/research/notes/dali/dali_market_universe_screen.md` | Documents market-universe findings and a one-off existing competition section for forward candidates. | Dali screen artifacts including `dali_forward_candidate_competition_audit.csv`. | Markdown note; recommends rejecting markets with low post-operator retention or top non-operator actor >40% USD share. | untracked; mtime 2026-05-23. |
| `polymarket/research/data/analysis/csv_outputs/dali/dali_forward_candidate_competition_audit.csv` | Candidate-market competition audit referenced by the Dali note. | Forward candidate fills, operator denylist. | CSV with retention/concentration-style diagnostics. | gitignored data artifact; mtime 2026-05-23. |
| `polymarket/research/scripts/dali_tfi_deep_dive.py` and `polymarket/research/notes/copytrade/block_b_findings.md` | Block B TFI deep dive. Includes operator-filter comparison that revealed large hit-rate improvements after removing known operators. | Cached family fills / market-second bars; `OPERATOR_ADDRESSES`. | `data/analysis/csv_outputs/dali/dali_tfi_operator_filter_comparison.csv`, `notes/copytrade/block_b_findings.md`. | script untracked mtime 2026-05-18; note untracked mtime 2026-05-27. |

### Polymarket API / Leaderboard Usage

| path | what it does | inputs | outputs | recency |
|---|---|---|---|---|
| `polymarket/research/scripts/api_reconciliation.py` | Compares local `traders.parquet` metrics for 10 selected addresses against public Data API `/value`, `/positions`, `/activity`. | `traders.parquet`, Polymarket Data API. | `notes/overview/data_quality/api_reconciliation_v1.md`. | untracked; local mtime 2026-05-18. |
| `polymarket/research/notes/overview/data_quality/api_reconciliation_v1.md` | Documents API limitations: `/positions` is open-only, `/value` is current portfolio value, lifetime UI PnL not exposed. | API reconciliation output. | Markdown report. | untracked; mtime 2026-05-18. |
| `polymarket/research/scripts/validation/07_sample_trader_sanity.py` | Attempts profile endpoints on `data-api`, `gamma-api`, and `lb-api` for Domah; profile endpoint did not resolve. | Public endpoints and local parquet. | stdout used in validation report. | tracked, last commit 2026-05-18 (`57172b0`). |

I did not find a successful pull of the public Polymarket leaderboard itself. The repo uses local lifetime `mkt_total_pnl` / cohort rankings as a richer internal leaderboard, and "top_leaderboard" labels refer to those local rankings rather than an imported leaderboard API table.

### Execution-Side Copy-Trading Code

| path | what it does | inputs | outputs | recency |
|---|---|---|---|---|
| `polymarket/execution/PLAN.md` | Defines the live copy-trading PoC: mirror one hardcoded leader via RTDS, fixed USD sizing, risk gates, journal-backed state. | Design decisions and probes. | Rolling implementation plan. | tracked, last commit 2026-05-18 (`37492d5`); file modified. |
| `polymarket/execution/watcher/leader_watcher.py` | Subscribes to RTDS `activity/trades`, filters by `proxyWallet`, emits leader fill events. | RTDS WebSocket payloads, `POLYMARKET_LEADER_ADDRESS`. | `LeaderFillObserved` journal/queue events. | tracked, last commit 2026-05-18 (`37492d5`). |
| `polymarket/execution/signal/classifier.py` | Classifies leader fills as entry/exit based on leader and bot position ledgers rebuilt from journal. | `LeaderFillObserved`, journal history. | `MirrorSignalEmitted`, dropped-fill events. | tracked, last commit 2026-05-18 (`37492d5`). |
| `polymarket/execution/mirror/mirror_engine.py` | Applies risk and submits mirror orders; tracks bot positions and daily PnL from journal. | Mirror signals, venue adapter, risk config. | Orders, `FillRecorded`, risk halts. | tracked, last commit 2026-05-18 (`37492d5`); file modified. |

This code is production copy-trading infrastructure, but it does not do wallet clustering or competition analysis. It consumes a single configured leader and leaves leader quality to research.

## Section 2: Operator Denylist Specifically

### How operator addresses were identified

The methodology is documented in `notes/overview/data_quality/validation_report.md` and generated by `scripts/validation/02_operator_detection.py`.

The original scan:

- Took the top 50 addresses by combined maker + taker fill count.
- Computed maker count, taker count, maker:taker ratio, distinct counterparties, distinct markets, total USD volume, and sub-second clustering.
- Identified three visible clusters:
  - Cluster A, pure relayers: 0 maker fills, all taker fills, around 2M counterparties, billions of dollars of flow.
  - Cluster B, pure MM bots: extreme maker:taker ratios, often >50 and sometimes >1,000.
  - Cluster C, HFT/arb flow: maker:taker ratio near 1.0, very high sub-second clustering, high fill counts.

The validation report recommended applying the filter at Phase 3/trader panel rather than stripping Layer A raw data.

### Current count and categorization

There are two "counts" in the repo:

- **12 explicitly categorized hardcoded addresses** in `operator_denylist.py`:
  - `PURE_RELAYERS`: 2
  - `PURE_MM_BOTS`: 7
  - `HFT`: 3
- **4,033 `is_operator_like` rows** reported in `METRICS_REFERENCE.md` / `README.md` after the `build_traders_table.py` heuristic is applied to `traders.parquet`.

Only the 12 hardcoded addresses are behavior-category labeled. The additional heuristic-flagged rows are not categorized by behavior type in a persisted sidecar or report.

### Criteria defining inclusion

The documented criteria are:

- hardcoded address in `OPERATOR_ADDRESSES`;
- extreme maker:taker ratio (`>50` or `<0.02`);
- distinct counterparties >500,000;
- sub-second clustering >95% and total fills >1,000,000.

One code-level nuance: the helper in `operator_denylist.py` says "extreme ratio" generally, and the validation report also mentions pure maker/pure taker shapes. The final SQL in `build_traders_table.py` explicitly handles ratios only when both sides needed for that ratio exist, then relies on hardcoded addresses / counterparty fan-out / sub-second heuristics for the rest. Future pure-maker addresses with zero taker fills may not be dynamically captured by the ratio branch unless they hit another heuristic or are added to the hardcoded list.

### Documentation coverage

Documentation exists and is decent:

- `validation_report.md` explains the original scan and clusters.
- `operator_denylist.py` summarizes category meanings.
- `METRICS_REFERENCE.md` documents the downstream `is_operator_like` disjunction and `traders_filtered`.
- `README.md` summarizes the denylist in project status.

Documentation gaps:

- No refresh cadence or "last checked through data date" attached to `operator_denylist.py` itself.
- No methodology for categorizing the 4,021 heuristic-only `is_operator_like` rows.
- No operator taxonomy output table with fields like `address`, `operator_type`, `trigger_reason`, `evidence_timestamp`, `data_through`.

### Obvious gaps

- The hardcoded list is seeded from a 2026-05-08 validation report over fills ending around 2026-04-24. It may miss operators that emerged after that tail.
- Block B's operator-filter result used `OPERATOR_ADDRESSES`, not necessarily the full 4,033-row heuristic flag, so the strongest TFI improvement may still be attributable to the 12 known large actors rather than the broader heuristic population.
- No recent live/Block A capture has yet validated whether RTDS `proxyWallet` identities map cleanly enough to apply the same filtering online.
- Operator categorization is not integrated into TFI outputs; the Block B result says "operator removed" but not "relayers removed vs MM bots removed vs HFT removed."

## Section 3: Gaps Relative To Block E Original Scope

| component | classification | rationale |
|---|---|---|
| A. Wallet clustering by behavioral fingerprint | **PARTIALLY DONE** | Rule-based behavioral classification exists via `build_traders_directionality.py` (`primary_style`), `build_copyability_metrics.py`, cohort pools, esports latency wallet screen, and one deep strategy profile. Missing: unsupervised clustering, wallet-linkage/ownership clustering, cluster stability, and a single canonical wallet archetype table. |
| B. Per-market competitive concentration analysis | **PARTIALLY DONE** | Dali has a one-off forward-candidate competition audit and note-level guards about operator retention and top-actor concentration. Esports screen has cell-level concentration. Missing: a general per-market concentration dataset across all markets / Block A target universes, time-varying concentration, and category-wide competition scores. |
| C. Profitability profile per archetype | **PARTIALLY DONE** | Trader panel, cohorts, directionality classes, copyability metrics, and leader audit reports provide profitability by wallet/cohort/selected archetype. Missing: systematic profitability summaries for each behavior archetype after resolving the `arb_like` contamination issue and split-construction exceptions. |
| D. Target market filtering combining competition + structural criteria | **PARTIALLY DONE** | Dali note proposes structural + competition guards; Phase 5 design includes market volume/liquidity/category filters; Block B identifies family-level TFI candidates. Missing: a reusable target-market filter artifact that combines operator retention, top-actor concentration, market structure, and Block A feasibility. |
| E. Specific bot identification with strategy characterization | **PARTIALLY DONE** | Explicit operator denylist, HFT cluster, esports latency candidates, and `0xd38b71f3` strategy profile exist. Missing: systematic bot catalogue, strategy labels for heuristic-only operators, and direct characterization of the actors whose removal improved Block B TFI. |

Nothing in the original Block E scope is fully complete. The repo has much more foundation than expected, especially per-wallet profitability/copyability, but the exact "competition and operator taxonomy" layer remains incomplete.

## Section 4: Recommended Block E Scope (Revised)

### Add or extend

1. **Operator taxonomy audit.** Deliverable: a table/document that separates the 12 hardcoded operators from the broader `is_operator_like` population, with trigger reason, category, data-through date, and whether each category affects Block B TFI.

2. **Block B operator-effect attribution.** Deliverable: compare TFI results across at least these conceptual states: all fills, hardcoded relayers removed, hardcoded MM/HFT removed, full `is_operator_like` removed where possible. This should explain who the "operators" are in the Block B result.

3. **Canonical wallet archetype inventory.** Deliverable: consolidate existing `primary_style`, copyability metrics, cohort membership, audit status, and known exception labels (`flagged_uncopyable`) into a readable research inventory. This is not a new clustering build; it is mainly unifying what already exists.

4. **Generalize competition audit for Block A candidate universes.** Deliverable: reuse the Dali competition-audit idea across the specific markets/families Block A will capture, reporting operator-retained fill share, top non-operator actor share, distinct active wallets, and whether the market is dominated by known excluded actors.

5. **Bot/strategy characterization backlog.** Deliverable: prioritized list of addresses that need manual strategy characterization, starting with operators or wallets most connected to Block B signal changes.

### Drop or avoid duplicating

- Do not rebuild basic trader PnL, style metrics, or cohort pools from scratch. `traders.parquet`, `traders_directionality.parquet`, and `traders_copyability_metrics.parquet` already cover that foundation.
- Do not redo the Domah/cross-leader copy audit framework unless a new leader is specifically selected for audit.
- Do not spend Block E on public leaderboard ingestion unless the goal is UI/API reconciliation. Local leaderboard/ranking by `mkt_total_pnl`, cohort membership, and copyability metrics is richer than the public API appears to expose.
- Do not treat `phantom_position_score` as the primary arb archetype filter; existing directionality work already found it is contaminated.

### Run now or wait for Block A data?

Run a narrowed Block E **now** for operator taxonomy and historical competition coverage. The Block B operator-filter result is strong enough that waiting for Block A would leave the live capture design under-specified.

Do not wait for Block A to start operator taxonomy. Do reserve one follow-up pass after Block A data arrives, because live L2/RTDS may reveal active operators or queue behavior that historical fill-only data cannot classify.

## Section 5: Block A Integration Question

Recommended Block A handling:

1. **Use the existing 12-address `OPERATOR_ADDRESSES` denylist as the baseline filtered view.** It is documented, conservative, and already produced a large Block B improvement.

2. **Do not modify the denylist before Block A starts.** Instead, capture raw/unfiltered data and produce filtered views downstream. The validation report's original recommendation still looks right: keep raw Layer A identity-neutral so denylist changes do not force rebuilds.

3. **Run Block A analysis both with and without operator filtering.** This should be a standard comparison, not a one-off. Given Block B moved equity_index from 47% to 58% and crypto from 34% to 52%, every Block A signal readout should expose `all_fills` vs `operator_removed`.

4. **Add a third comparison when feasible: full `is_operator_like` removed.** The hardcoded 12 and the broader 4,033 `is_operator_like` population are different things. If address-level data is available in Block A, compare both.

5. **Do not wait for Block E results before using operator filtering.** Use the known denylist now, keep raw data, and let Block E refine taxonomy and categories in parallel.

Bottom line: Block A should proceed with dual views (`all_fills`, `operator_removed`) immediately. Block E should explain and improve the operator taxonomy, not become a prerequisite for basic Block A measurement.
