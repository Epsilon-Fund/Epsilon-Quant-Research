# Research Findings
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


Consolidated research log across phases 1–4. What we've **learned** about
the data and the trader population. For repo orientation and "how to
reproduce", see [`README.md`](README.md). For per-column formulas, see
[`docs/METRICS_REFERENCE.md`](docs/METRICS_REFERENCE.md).

All numbers below come from the existing artifacts cited inline.

---

## Data shape (Phases 1–2 inspection)

Phase 1 inspection in [`notes/overview/data_quality/validation_report.md`](notes/overview/data_quality/validation_report.md). Phase 2 self-consistency in `scripts/build_closed_positions.py:249-256`.

- **1 064 500 317 fills** in `raw_trades` (913.4 M Goldsky delta + 151.1 M warproxxx seed). Coverage: 2022-11-21 19:50:09 → 2026-04-24 00:00:00 UTC.
- **2 619 353 distinct addresses** appear at least once as maker or taker. **2 576 698** show up in `closed_positions` (i.e. participated in at least one resolved-market fill).
- **797 229 markets** have at least one closed-position row; 1 036 708 markets total in the Gamma snapshot.
- **22 % of fills are on NegRisk markets**, 78 % on regular CTF Exchange. The split shapes everything downstream — see "phantom score" below.
- **96.0 % of fills are on closed (resolved) markets** at snapshot time — gives Phase 2 a near-complete population to work with.
- **0.379 % of fills are orphan** (`market_id IS NULL`, i.e. the outcome token doesn't appear in the Gamma snapshot). Validation found these aren't aging — recent month (2026-04) is 0.05 %, legacy months spike up to 3.4 %. Routed to `trader_actions_orphan` and excluded from position math.
- **905 self-trades** (`maker = taker`), all from 2022-11 → 2023-04. Early-protocol noise. Filtered at view level.
- **12 operator-shaped addresses** identified and deny-listed:
  - 2 **pure relayers** (Cluster A): `0x4bfb41…` (305 M fills, $17.34 B notional, 1.99 M counterparties, **0 maker fills**) and `0xc5d563…` (100 M fills, $12.60 B). These are matcher/relayer bots.
  - 7 **pure MM bots** (Cluster B): extreme maker:taker ratios (45× to 870 000×). `0x297fbd…`, `0x048956…`, `0xdc669b…`, etc. Liquidity-providing bots.
  - 3 **HFT cluster** (Cluster C): ratio ≈ 1.0 with 95–99 % sub-second clustering. `0xe8dd77…`, `0x63d43b…`, `0xe3726a…`. Matched buy/sell flow at machine pace.
- **Sign asymmetry** (Phase 1 check 5, in validation report): 4:1 BUY-by-maker vs SELL-by-maker fills, median per-token ratio 8.3:1. **This is structural, not corruption** — most positions are bought and held to redemption (not closed via on-book sells). Drives the redemption-synthesis design in Phase 2.
- **Self-consistency confirmed** (build_closed_positions.py:249-256): `Σ realised_pnl = $0.00` across all **269 974 929 closed positions**, both NegRisk and regular subtotals also $0. Every winner's redemption matched a loser's lost stake on the resolved outcome.
- **96.4 % of positions held to resolution** (260.4 M of 270.0 M). Confirms the redemption-synthesis is load-bearing — most positions never close via on-book sells.

### Trader-activity distribution

From [`notes/overview/data_quality/validation_report.md`](notes/overview/data_quality/validation_report.md) §2 and the Phase 3 sanity check in `scripts/build_traders_table.py:574-587`:

| metric | value |
|---|---:|
| total addresses with ≥ 1 closed position | **2 576 698** |
| p10 n_closed_positions | 2 |
| p25 | 6 |
| **p50 (median)** | **16** |
| p75 | 45 |
| p90 | 114 |
| p95 | 256 |
| p99 | 1 644 |
| max | 1 123 635 (operator) |

Implication: cohort thresholds at 100+ closed positions filter to top ~11 %, 500+ to top ~3.5 %, 1 000+ to top ~2 % of the active trader population.

---

## Trader-metric distribution (Phase 3)

Numbers below from `scripts/sanity_phase3.py` output and `scripts/build_traders_table.py` final summary.

### Sharpe distribution — annualisation artifacts at the tail

Across `traders_filtered` with `n_closed_positions > 50`:

| | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| `pos_sharpe` | -0.81 | 2.58 | 4.38 | **14.90** | **1.66 × 10¹⁵** |
| `mkt_sharpe` | -0.69 | — | 3.09 | **11.17** | 107.20 |

p99 above 5 = artifact territory. Naive `sqrt(N / years_active)` annualisation blows up when `active_days` is small (a 30-day fluke gets multiplied by `sqrt(12)`), `n_closed_positions` is small, or `pos_std_pnl` is near zero (NegRisk arb traders with matched-pair PnLs). The `1.66 × 10¹⁵` outlier is a `pos_std_pnl ≈ 0` numerical artifact.

Top-Sharpe outliers split into two buckets:

1. **Tiny-PnL short-active-window flukes** — active_days = 0–30, n_pos ≈ 60, mkt_pnl < $400. Lucky runs.
2. **NegRisk arb traders with low-variance matched-pair PnLs** — `0x20d2309cd9` (60 k positions, $298 k, phantom 12.6), `0x56bad0e7a0` (48 k positions, $200 k, phantom 35.2). Real activity, but Sharpe inflated by low per-trade variance.

**Decision: Sharpe is DIAGNOSTIC, not a primary ranker.** Cohort filters require simultaneous guards (`n_pos > 200 AND active_days > 90 AND mkt_std_pnl > 1.0`) to exclude blowups; even with guards, prefer `mkt_profit_factor` and `mkt_dollar_win_rate` for ranking.

### Phantom position score — the NegRisk arb detector

`phantom_position_score = volume-weighted avg over (address, market) of [Σ |per-outcome PnL| / |Σ per-outcome PnL|]`. `1.0 = clean directional, > 1.0 = NegRisk arb-shaped`.

- Domah's score is **8.45**, matching his known NegRisk arb behaviour. He holds both YES and NO of large markets — at position level he looks like `winner + equal-magnitude loser`; at market level the pair cancels.
- Across all qualifying cohort-pool members, score is roughly bimodal — clean directional traders hover at 1.0–1.5; arb traders cluster at 2–10+.
- **Practical reading**: scores ≥ 2 imply per-position metrics are unreliable; rank such traders by `mkt_*` family only.

### Style profile populations

From `traders.parquet` (50 cols documented in METRICS_REFERENCE):

- **`style_role_balance`** (1.0 = pure maker, 0.0 = pure taker): Polymarket is broadly maker-favouring at the trader level. Most cohort candidates have role_balance > 0.7. The notable exception in the Phase 4 top-20 union is `0xd38b71f3` with role_balance = **0.22** — taker-heavy and still very profitable, an unusual profile.
- **Maker-heavy population**: most "patient accumulator" cohort members, most NegRisk specialists, most high-Sharpe directional traders.
- **Taker-heavy population**: smaller, includes some HFT cluster operators (deny-listed) and a handful of legitimate active-flow traders worth manual inspection.
- **`style_pct_sub_second`**: heavy concentration > 95 % flags HFT/operator-shaped behaviour. The deny-list heuristic combines this with `n_fills_total > 1 000 000` — the AND is what keeps single-fast-traders in the cohort while excluding bots.
- **`style_avg_holding_hours`**: extremely long-tailed. Median around 24 hours for fast traders, 1 700+ hours (~71 days) for patient accumulators (domah is in this bucket).

### Bankroll distribution (lifetime peak deployed capital)

`est_bankroll_usd_30d_max_approx` per `traders.parquet`:

| | non-null | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| value | 2 573 723 / 2 576 698 (99.9 %) | **$319** | $5 990 | $56 966 | $1 606 840 923 |

The max ($1.6 B) is an operator address (`0x4bfb41…`); for non-operators the max is in the $50–100 M range. Domah's lifetime peak is $59.07 M.

**Caveat re-stated**: this is the **maximum ever** deployed at entry value, not point-in-time. Useful as a descriptive scale indicator, **not** for forward-looking sizing decisions in any backtest. Phase 5 needs a rolling 30-day-prior version that respects walk-forward discipline.

---

## Cohort exploration (Phase 4)

Six stratified pools materialised at `data/cohorts/*.parquet`. Defensive guards baked into every Sharpe-touching filter (n_pos > 200, active_days > 90, mkt_std_pnl > 1.0). Definitions verbatim in `scripts/build_cohorts.py:33-97` and tabulated in METRICS_REFERENCE §3.

### Per-pool counts and headline medians

| pool | n | med n_pos | med days | med PF | med PnL | med phantom |
|---|---:|---:|---:|---:|---:|---:|
| **A** `high_sharpe_directional` | 3 304 | 833 | 155 | 1.56 | $4 115 | 1.49 |
| **B** `high_profit_factor_with_size` | 556 | 357 | 332 | 3.01 | $109 921 | 1.46 |
| **C** `negrisk_specialists` | **113** | 892 | 434 | 1.94 | $127 458 | 1.65 |
| **D** `sports_directional_fast` | 2 152 | 1 251 | 131 | 1.24 | $1 779 | 1.43 |
| **E** `patient_accumulators` | 225 | 1 365 | 552 | 1.63 | $222 025 | 1.76 |
| **F** `high_kelly_edge` | **7 703** | 377 | 250 | 1.51 | $2 132 | 1.58 |

Union: **9 788 unique addresses** in 1+ pools, 3 278 in 2+, **947 in 3+**, 40 in 4+, **0 in 5+ or 6**.

### Cross-pool overlap matrix

Counts of addresses qualifying for both pools (i, j):

```
            A_sharpe   B_pf  C_negr  D_sport  E_patient  F_kelly
A_sharpe       3,304    202       0    1,501         58    1,911
B_pf             202    556      55       23         76      246
C_negr             0     55     113        0         29       97
D_sport        1,501     23       0    2,152          0      947
E_patient         58     76      29        0        225      147
F_kelly        1,911    246      97      947        147    7,703
```

### Key insights from the diagnostics

1. **Pools span genuinely independent dimensions.** No address qualifies for 5/6 pools by design — the criteria are orthogonal. 3-pool overlap is the natural ceiling for breadth (947 addresses).
2. **A ∩ C = 0 by construction**: A excludes `negrisk_share > 0.5`, C requires `> 0.7`. The two never intersect.
3. **Profit factor and Sharpe are less redundant than expected.** A ∩ B (high-Sharpe AND high-profit-factor) is only **202** out of 3 304 in A and 556 in B. Many high-PF traders have low Sharpe because their wins are concentrated in a few big bets — drags per-position std up, drags Sharpe down.
4. **F (high_kelly_edge) is too permissive.** 7 703 qualifiers — way more than any other pool. Kelly > 0.05 with DWR > 0.55 is a low bar in this data. The marginal F-only entrants are unlikely to be cohort-worthy on their own; pair with another pool.
5. **C ∩ B = 55 — the cleanest "skill-not-just-arb-flow" candidates.** NegRisk specialists who *also* clear the absolute-PnL + profit-factor bar. About half of all NegRisk specialists.
6. **Domah qualifies for E + F** (patient_accumulators ∪ high_kelly_edge): two-pool candidate. 100th percentile of PnL within both pools, but only 45–56th percentile of profit factor — high volume, modest per-position edge. Matches what we know about him.

### Guard sanity (Phase 4 invariant D.3)

Across the 14 053 pool rows (union with duplicates), the count of rows violating any guard (`mkt_sharpe > 100`, `mkt_kelly_fraction > 1`, `pos_std_pnl < $0.01`, `mkt_std_pnl < $0.01`) is **0**. Guards held.

---

## Top candidates surfaced (Phase 4)

Five addresses from the cross-pool top-20 union flagged for manual due diligence. Output of `scripts/cohort_diagnostics.py`. Headline metrics are from `data/traders.parquet`.

### 1. `0xd38b71f3e8ed1af71983e5c309eac3dfa9b35029`

**Why interesting**: $5.68 M PnL, **3 pools** (`high_sharpe_directional`, `sports_directional_fast`, `high_kelly_edge`), `phantom_position_score = 1.00` (no NegRisk arb), `negrisk_volume_share = 0.01` (non-NegRisk), and **`style_role_balance = 0.22`** — taker-heavy. Highly atypical: most edge in this dataset lives on the maker side. A $5 M-PnL taker-dominant trader is either smart at picking off mispriced resting limits or has retail-flow access we don't see. The most genuinely surprising profile in the leaderboard.

| metric | value |
|---|---:|
| n_closed_positions | 1 103 |
| active_days | 224 |
| mkt_total_pnl | $5 678 261 |
| mkt_profit_factor | 1.51 |
| mkt_dollar_win_rate | 0.601 |
| mkt_sharpe (DIAGNOSTIC) | 3.66 |
| phantom_position_score | 1.00 |
| negrisk_volume_share | 0.01 |
| style_role_balance | 0.22 |
| est_bankroll_usd_30d_max_approx (descriptive) | $1 072 321 |

**Manual due diligence**: Are they running an obvious strategy (e.g., crossing every wide spread > X bps)? Which markets dominate their PnL — a single hot category, or broad? Is the taker-heavy stance consistent across time, or did it flip recently?

### 2. `0x6a72f61820b2…` — top of leaderboard

**Why interesting**: **$14.95 M lifetime PnL**, 4 359 closed positions, 2 pools (`high_sharpe_directional`, `high_kelly_edge`), `phantom_position_score = 2.18` (low, not arb), `negrisk_volume_share = 0.05` (non-NegRisk). Highest single PnL in the union. A non-arb directional trader at the top of the leaderboard — the canonical "ranking-target" archetype.

| metric | value |
|---|---:|
| n_closed_positions | 4 359 |
| active_days | 302 |
| mkt_total_pnl | $14 952 586 |
| mkt_profit_factor | 1.25 |
| mkt_dollar_win_rate | 0.555 |
| mkt_sharpe | 2.58 |
| phantom_position_score | 2.18 |
| style_role_balance | 0.84 |
| est_bankroll_usd_30d_max_approx | $29 769 205 |

**Manual due diligence**: What's the win-distribution shape — a few mega-winners, or many small? Time-distributed PnL or front/back-loaded? Are the markets they win on consistent (sport-specific, election-specific)?

### 3. `0x17db3fcd93ba…` — small-sample extremes

**Why interesting**: **$5.45 M from only 235 positions**, profit factor **6.54**, dollar win rate **0.867**. Just clears the n_pos > 200 + days > 90 sample-size guards (n_pos = 235, active_days = 99). Either elite skill or a long lucky run on a few high-conviction bets. Manual inspection of their market mix would resolve which.

| metric | value |
|---|---:|
| n_closed_positions | 235 |
| active_days | 99 |
| mkt_total_pnl | $5 453 680 |
| mkt_profit_factor | 6.54 |
| mkt_dollar_win_rate | 0.867 |
| mkt_sharpe | 4.96 |
| phantom_position_score | 1.36 |
| style_role_balance | 0.59 |

**Manual due diligence**: Are the wins concentrated on a single high-stakes market (e.g., one election outcome)? Profile their position-by-position PnL chart — is it a smooth ladder or one big spike?

### 4. `0xee00ba338c59…` — large-sample stable

**Why interesting**: $4.66 M PnL, **11 726 closed positions**, **active_days = 618** (~1.7 yr), 3 pools (`patient_accumulators`, `high_kelly_edge`, `high_sharpe_directional`). The strongest large-sample multi-pool profile in the top 10 — likely a serious systematic strategy. Inspect for cohort baseline.

| metric | value |
|---|---:|
| n_closed_positions | 11 726 |
| active_days | 618 |
| mkt_total_pnl | $4 658 451 |
| mkt_profit_factor | 1.25 |
| mkt_dollar_win_rate | 0.556 |
| mkt_sharpe | 2.02 |
| phantom_position_score | 2.85 |
| negrisk_volume_share | 0.23 |
| style_role_balance | 0.79 |
| est_bankroll_usd_30d_max_approx | $36 200 469 |

**Manual due diligence**: Is the strategy stable across the 1.7 years (Sharpe consistency by year), or front-loaded? Phantom = 2.85 implies some NegRisk-arb activity — what fraction of their PnL is from NegRisk markets?

### 5. `0x629bc4a1e53e…` — only 4-pool qualifier

**Why interesting**: One of only **6 four-pool qualifiers** in the entire 9 788-trader cohort union. $1.80 M PnL, profit factor 2.74, **active_days = 1 096 (3 yr)**, phantom 2.08, `negrisk_volume_share = 0.84`. NegRisk specialist with persistent edge AND broad pool coverage — the most multi-dimensionally robust candidate in the dataset.

| metric | value |
|---|---:|
| pools | `high_profit_factor_with_size`, `patient_accumulators`, `high_kelly_edge`, `negrisk_specialists` |
| n_closed_positions | 1 365 |
| active_days | 1 096 |
| mkt_total_pnl | $1 795 457 |
| mkt_profit_factor | 2.74 |
| mkt_dollar_win_rate | 0.733 |
| mkt_sharpe | 2.08 |
| phantom_position_score | 2.08 |
| negrisk_volume_share | 0.84 |
| style_role_balance | 0.78 |

**Manual due diligence**: Sustained 3-year edge is the holy grail — but is it real or is the trader running a strategy that benefits from a structural quirk that may not last? What fraction of PnL came in each year?

---

## Open questions / hypotheses to test

Phase 5 backtesting will pressure-test these:

1. **Does cohort A (high Sharpe) outperform cohort B (high PF) out-of-sample?** They overlap by only 202 / 3 304 — very different selection signals. Out-of-sample edge is the test.
2. **Are NegRisk specialists profitable on non-NegRisk markets too?** The notebook has a `broad_edge` vs `negrisk_only` split for Pool C members; need to backtest both subsets independently.
3. **What's the trader half-life?** Do top candidates from 2024 still rank top in 2026? Walk-forward would surface this directly.
4. **How much edge survives slippage / latency / capacity?** Copy-trading isn't free — order routing is delayed vs the leader, sizing is constrained by bankroll, and crowded copy-trade flow moves prices. None of this is in our current numbers.
5. **Is 3-pool membership predictive vs 1-pool?** Multi-pool members are robust on multiple metrics simultaneously, but they may also be in-sample-fit. Walk-forward on "qualified for ≥ k pools at time T-1" vs forward returns at T is the test.
6. **Does taker-heavy `0xd38b71f3` represent a generalisable strategy** or one-off retail-flow access? Need to check whether other taker-heavy traders show similar patterns.

---

## Known unknowns (limitations of current research)

These are constraints that no Phase 1–4 work can lift; they would require either external data or new infrastructure.

1. **Cannot validate exact PnL against Polymarket UI.** The public API exposes only currently-open mark-to-market and partial-realized within open positions. Lifetime PnL is on the UI profile pages, not scriptable. Cross-check against the UI is order-of-magnitude only; treat differences <20 % as internally normal. See [`notes/overview/data_quality/api_reconciliation_v1.md`](notes/overview/data_quality/api_reconciliation_v1.md).
2. **Merge/split blindspot affects ~the ~20 % of traders with `phantom_position_score > 2`.** Their per-position PnL is inflated/deflated by the unobserved mint amount. `mkt_*` aggregates correctly, but per-position diagnostics are unreliable. Closing this requires indexing CTF `Splits`/`Merges` events outside `OrderFilled`.
3. **No backtest yet.** All current rankings are in-sample. Phase 4 cohort selection might be over-fit to the historical distribution — we don't know without walk-forward.
4. **Bankroll metric leaks future capacity into past sizing decisions.** `est_bankroll_usd_30d_max_approx` is lifetime peak. Anyone using it to size a 2024-era backtest position would be using post-2024 information. Phase 5 needs the rolling-prior version.
5. **No FPMM legacy trades** (pre-2022 AMM). CTF Exchange only. Drops a small slice of older market activity.
6. **Open positions excluded entirely from PnL.** A trader with 80 % of their book still in unresolved markets is invisible to our metrics. We don't have mark-to-market machinery.
7. **Goldsky lag ~9 days, plus time since last sync.** As of the snapshot stamp, the data tail is ~16 days behind real-time. Any walk-forward backtest must use cutoffs ≥ 9 days behind run date.

---

## Decisions taken so far (with reasoning)

1. **Goldsky over direct Polygon RPC indexing.** Cheaper (no node-running infra), sufficient resolution for v1 (CTF Exchange `OrderFilled` events). Trade-off: ~9-day indexer lag, no access to mint/merge/redeem events. Acceptable for offline ranking; may revisit for live execution.
2. **Materialised `closed_positions.parquet`, kept `trader_actions` as a SQL view.** The position table is queried by every downstream phase and is too expensive to recompute (270 M rows after redemption synthesis). The view is only ever scanned once per build. Storage trade: 27 GB on disk, ~12 min to build, but every downstream query now runs in seconds. (Source: `scripts/build_closed_positions.py`.)
3. **Computed both `pos_*` and `mkt_*` metric families.** NegRisk arb traders inflate position-level metrics by holding both YES and NO of the same market. Market-level aggregation cancels the pair. Carrying both lets cohort selection choose the right unit per question. (Source: `scripts/build_traders_table.py:99-262` and METRICS_REFERENCE §C "Position-level vs market-level".)
4. **Operator deny-list as a hybrid.** Compute all metrics for all addresses (identity-neutral substrate), then filter via `traders_filtered = traders_raw WHERE NOT is_operator_like`. The flag combines a hand-curated 12-address list with five operator-shape heuristics. Trade-off: raw `traders.parquet` includes operators (useful for analysing them, e.g. our $61 M relayer); the view-based filter keeps cohort selection clean. (Source: `data_infra/operator_denylist.py` + `scripts/build_traders_table.py:501-516`.)
5. **Deferred merge/split indexing.** The full fix (indexing CTF `Splits`/`Merges` and propagating the implied mint legs through position math) is large. Instead, `phantom_position_score` flags affected traders; `mkt_*` family is robust without the missing legs. v1 acceptable; v2 work item.
6. **Phase 4 is exploration, not strategy.** Six pools, no composite ranking, no allocation logic. Phase 5 will pressure-test selection rules under walk-forward + CPCV. Phase 4 is identifying *what to test*, not *what to allocate*.
7. **Defensive guards baked into every Sharpe-touching cohort filter.** Without them, the p99 of `pos_sharpe` is 14.90 and the max is 1.66 × 10¹⁵. With simultaneous `n_pos > 200 AND active_days > 90 AND mkt_std_pnl > 1.0`, the artifact tail is fully excluded (Phase 4 invariant D.3 holds across all 14 053 pool rows). Lesson learned: never let an annualised Sharpe rank a cohort without sample-size and variance floors.
8. **`peak_fill_abs_token` instead of true running peak.** The `peak_position_size` formula in the original spec required a 1.4 B-row sort and OOM'd at 300 GB temp during the build. v1 ships a lower-bound proxy; renamed for honesty. (Source: `scripts/build_closed_positions.py:60-67`.)
9. **`approx_count_distinct` for counterparty counts.** Exact `COUNT(DISTINCT)` over 1 B fills × 2 sides was too heavy. HyperLogLog has ~2 % relative error — far below the threshold gap for operator detection (`> 500 000`).
10. **Persistent DuckDB file (`_traders_build.duckdb`) for the Phase 3 build.** Each metric stage is a `CREATE TABLE IF NOT EXISTS`, so re-runs after a crash skip completed stages. Saved ~10 minutes of redundant compute when the build OOM'd mid-flight on early attempts.
