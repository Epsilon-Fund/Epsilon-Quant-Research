---
title: "Phase 5 — Walk-Forward Cohort Backtesting"
created: 2026-06-05
status: candidate
owner: justin
project: polymarket
para: project
hubs:
  - COWORK
tags:
  - research
  - copytrade
---
# Phase 5 — Walk-Forward Cohort Backtesting
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


## Summary

- Scope: Phase 5 — Walk-Forward Cohort Backtesting in the copytrade area.
- Existing takeaway/status: Locked design specification for Phase 5 walk-forward cohort backtesting. It defines walk-forward mechanics, cohort definitions, per-trade filters, sizing, fill/slippage modeling, and latency documentation; implementation was pending when written.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
**Status:** Design specification, locked. Implementation pending.
**Prerequisite:** Phases 1-4 complete. `bankroll_timeseries.parquet` materialised. All cohort pools in `data/cohorts/*.parquet`.

---

## 1. Objective

Validate cohort-based copy-trading strategies on historical Polymarket data using walk-forward backtesting. Specifically:

1. Determine whether candidate cohort definitions produce out-of-sample edge after accounting for realistic execution slippage.
2. Identify the operational rhythm (resolution-time bucket) at which each cohort is most effective.
3. Quantify cohort robustness across slippage assumptions — produce breakeven slippage estimates per cohort.
4. Produce data that lets us decide which cohort (if any) to deploy on the execution side.

## 2. What Phase 5 is NOT

- Not a live trading system.
- Not a recommendation for which specific traders to follow (cohort selection at runtime is automatic from the rule, not hand-picked).
- Not CPCV (deferred to v2 — pure walk-forward in v1).
- Not capacity modelling (assumes infinite liquidity at touch + slippage; v2 will add depth-aware caps).
- Not crowding modelling (assumes our fills don't move prices; v2 if ever).
- Not mark-to-market on open positions during the test window (only resolved positions count toward PnL).

## 3. Locked Methodological Decisions

### 3.1 Walk-Forward Mechanics

Monthly rolling refresh:

```
For each refresh date T (2024-01-01, 2024-02-01, ..., 2026-05-01):
  IS WINDOW: all data with resolution_ts < T (expanding)
  OOS WINDOW: [T, T+1 month)

  Step 1: Compute trader metrics from IS data only
          (no leakage of post-T resolutions)
  Step 2: Apply cohort filter rule to those metrics
          (yields list of qualified addresses at time T)
  Step 3: For OOS window:
          a. Find every trade those qualified addresses made
          b. Apply per-trade filters (resolution-time bucket,
             market criteria, maker-only filter)
          c. For each surviving trade, simulate copy with chosen
             sizing rule + slippage model
          d. Record realised PnL when the copied position resolves
  Step 4: Append OOS PnL records to running ledger
```

The output PnL curve is the **stitched OOS sequence** across all refresh dates. Sharpe, drawdown, and other metrics are computed over this stitched curve — never over IS data.

**Why expanding IS over rolling IS:** trader edge typically requires years of evidence to establish; rolling a 12-month IS window means losing the longer-track-record signal. Expanding from data start to T captures everything available at decision time. Trade-off: weights older performance equally with recent. Defensible for v1; consider rolling in v2 if results suggest "trader half-life" matters.

**Survivorship handling:** a trader who stops trading after some date naturally disappears from later cohort selections (their metrics stale or their volume drops below thresholds). Their pre-cessation PnL counts; their absence from later periods reflects reality.

### 3.2 Cohort Definitions

Three cohorts, all share the same defensive guards: `n_closed_positions > 200`, `active_days > 90`, `mkt_std_pnl > 1.0`, and `NOT is_operator_like`.

**Cohort 1 — `B_high_pf_with_size`**

Phase 4 Pool B as-is.

```
mkt_profit_factor > 2.0
mkt_total_pnl > $50,000
n_closed_positions > 100  (note: lower than the 200 default; B-pool
                           original threshold)
active_days > 90
```

**Cohort 2 — `BC_directional_negrisk`**

Pool B AND Pool C intersection, additionally filtered for non-arb behaviour.

```
mkt_profit_factor > 2.0
mkt_total_pnl > $50,000
negrisk_volume_share > 0.7
phantom_position_score < 3.0  (excludes arb-driven PnL)
n_closed_positions > 200
active_days > 90
```

**Cohort 3 — `E_patient_accumulators`**

Phase 4 Pool E as-is.

```
style_role_balance > 0.7  (already maker-heavy by definition)
style_avg_holding_hours > 168  (one week+)
mkt_total_pnl > $100,000
n_closed_positions > 100
active_days > 180
```

Cohort definitions are evaluated **at each refresh date** against the trader metrics computed from IS data. Cohort membership changes month to month as traders qualify or de-qualify.

### 3.3 Per-Trade Filters

Applied to each OOS trade *before* the simulator considers copying it:

**Resolution-time bucket** — four parallel runs per cohort:

- 2d: `(market.end_date - trade.timestamp) < 2 days`
- 7d: `< 7 days`
- 30d: `< 30 days`
- 60d: `< 60 days`

Each bucket is a separate backtest. Reveals which rhythm each cohort fits. (Tatv: "resolution time is the most expressive single knob.")

**Maker-only filter (default):** copy only fills where the leader was the maker. Excludes taker fills at trade level, regardless of trader identity. Rationale: taker fills are latency-sensitive (leader was racing into a moving orderbook); 30+ seconds of detect-to-fill lag makes these uncopyable cleanly. Maker fills are by definition patient (leader posted, waited to be hit).

Sensitivity flag: a `--include_takers` mode runs the same backtest without the filter. Useful to quantify the value of taker fills.

**Market criteria:**

- `market.volume_at_trade > $10,000`
- `market.liquidity_at_trade > $5,000`
- `market.category NOT IN category_denylist`

The `category_denylist` is empirical — populated from Phase 4 diagnostics when categories with consistent cohort-level losses are identified. Initially empty; updated as Stage 1 results surface category-specific leaks.

### 3.4 Sizing Rules

Two sizing rules backtested in parallel:

**Rule A — Fixed-% of bankroll**

```
position_size_usd = strategy_capital × 0.02  (2% per signal)
```

Caps applied:

- Max 5 concurrent positions per leader (prevents concentration if leader fires many signals fast)
- Max 20% bankroll exposure to single leader
- Max 30% bankroll exposure to single market category

**Rule B — Leader-proportional**

```
leader_fraction = leader_trade_usd / leader_bankroll_30d_prior(T)
position_size_usd = min(
    leader_fraction × strategy_capital,
    strategy_capital × 0.05  (cap at 5% of own bankroll)
)
```

Uses `bankroll_timeseries.parquet` for point-in-time leader bankroll. Same caps as Rule A.

Backtest output annotates each fill with the sizing rule used so analysis can compare side-by-side.

### 3.5 Fill Model and Slippage

Leader fills at price P at time T on market M, outcome O. The copy
fill price is derived from the next qualifying fill in the same market
and outcome, by a different trader, within a configurable time window.

**Methodology:**

For each leader fill, look up the next OrderFilled event satisfying:
- Same `market_id`
- Same `outcome_token_id`
- Same direction (buy if leader bought, sell if leader sold)
- Different maker AND different taker (neither side is the leader)
- `timestamp > leader.timestamp + min_seconds`
- `timestamp < leader.timestamp + max_seconds`

That fill's price becomes the assumed copy fill price. The implicit
slippage = copy_price − leader_price (or the reverse for sells).

If no qualifying fill is found in the window, fallback applies:
`copy_price = leader_price + fallback_slippage_cents/100` (sign-adjusted).

**Rationale for this model:**

This matches our execution policy: the bot posts a limit at the
leader's price on detection; if the market moves before the order
fills, the bot pushes the bid up (or down for sells) until filled.
The next other-trader fill at the same direction captures where this
push would have landed.

Slippage can be negative when the market moves in our favour between
the leader's fill and the next fill. This is realistic and is not
filtered out.

**Stage 1 default settings:**
- `min_seconds = 15`
- `max_seconds = 300` (5 minutes)
- `fallback_slippage_cents = 3.0`

**Stage 2 sensitivity grid:**
- (min, max) window: `[(15, 60), (15, 300), (30, 120), (30, 600), (60, 300)]`
- `fallback_slippage_cents ∈ {1, 2, 3, 5, 8}`
- 25 combinations applied to top 5 Stage 1 combos

**Stage 3 breakeven analysis:**
- Top 2-3 Stage 2 combos
- Fine-grid search on `fallback_slippage_cents` to find PnL = 0
- Output: "Cohort X breaks even at fallback Y¢"

**Data implications:**

The audit log captures: `leader_price`, `copy_price`, `slippage_cents`,
`slippage_source` (`'next_fill'` or `'fallback'`). Analysis can split
results by `slippage_source` to see how much PnL came from
data-derived vs fallback-derived simulations.

If a high fraction of trades (say > 40%) hit the fallback, the cohort
is heavily on quiet markets where the slippage model is least
reliable. Flag in summary; consider tightening market liquidity
criteria.

**Note on taker fills:** the previous design had a separate
`taker_penalty` constant. With next-fill slippage, this is implicit —
markets where the leader was a taker (raced into thin liquidity) will
naturally show larger gaps to the next-fill, capturing the higher
slippage organically. No separate `taker_penalty` parameter needed.

#### 3.5.1 Performance Note

Computing next-fill slippage per leader trade is potentially expensive
on a ~1B-row trades parquet. The implementation should use DuckDB's
ASOF JOIN to attach the next-fill in a single SQL pass per refresh
date, rather than a row-by-row lookup. Example pattern:

```sql
WITH leader_fills AS (
  SELECT * FROM raw_trades
  WHERE (maker IN qualified_cohort_addresses OR taker IN qualified_cohort_addresses)
    AND timestamp BETWEEN refresh_date AND refresh_date + INTERVAL 1 MONTH
)
SELECT
  lf.*,
  COALESCE(nf.price, lf.price + fallback_cents/100.0) AS copy_price,
  CASE WHEN nf.price IS NOT NULL THEN 'next_fill' ELSE 'fallback' END AS slippage_source
FROM leader_fills lf
ASOF LEFT JOIN raw_trades nf
  ON lf.market_id = nf.market_id
  AND lf.outcome_token_id = nf.outcome_token_id
  AND lf.direction = nf.direction
  AND lf.maker != nf.maker
  AND lf.taker != nf.taker
  AND nf.timestamp >= lf.timestamp + INTERVAL min_seconds SECOND
  AND nf.timestamp <= lf.timestamp + INTERVAL max_seconds SECOND
```

### 3.6 Execution Latency Documentation

The backtest is agnostic to which detection method the execution side uses. Slippage parameters should be tuned to match:

| Detection method | Expected detect-to-fill lag | Maps to fallback_slippage_cents |
|---|---|---|
| Goldsky subgraph polling | 5-15 minutes | High (`fallback ≥ 5`) |
| Polymarket CLOB WebSocket | 5-30 seconds | Medium (`fallback 2-3`) |
| Polygon event subscription | 1-5 seconds | Low (`fallback 1-2`) |
| Same-second mempool intercept | sub-second | None (`fallback 0`) |

Phase 5 results should be read with the execution side's chosen detection method in mind. When deploying a cohort, the realistic `fallback_slippage_cents` to consult is the one matching live latency.

### 3.7 Cohort Recompute Cadence

Monthly. Locked.

Rationale: weekly is too frequent (trader metrics don't shift meaningfully week-to-week, and cohort churn introduces noise). Quarterly is too slow (a trader who blew up in March stays in cohort through June). Monthly balances stability and responsiveness; matches Tatv's reported cadence.

### 3.8 Test Windows

Three independent windows:

- 2024-01 → 2024-12 (full year)
- 2025-01 → 2025-12 (full year)
- 2026-01 → 2026-04 (partial, data-tail constrained)

A cohort that works in all three is robust. A cohort that works in only one is a coincidence. Cross-window comparison is the actual test.

## 4. Capital Utilisation as First-Class Output

Every backtest run reports:

- **Deployment ratio:** % of OOS window time that capital was deployed vs idle
- **Signal frequency:** qualifying signals per week
- **Position concentration:** max % of bankroll in single position over the window
- **Trader concentration:** number of distinct leaders contributing PnL
- **Category breakdown:** PnL contribution per market category

Low deployment (< 30%) → cohort is too tight, edge per trade may be real but capital wastes. High deployment (> 95%) with poor PnL → cohort is too loose, edge diluted by noise. Sweet spot is empirical.

Tatv: "If you are not using your capital, you are not running the strategy you think you are running."

## 5. Output Schema

### 5.1 Per-Backtest Audit Log

For each backtest run, write `data/backtests/{run_id}.parquet`:

```
run_id              VARCHAR    -- includes cohort, bucket, sizing, slippage params
refresh_date        DATE       -- which monthly refresh selected this leader
leader_address      VARCHAR
market_id           VARCHAR
condition_id        VARCHAR
outcome_index       INT
neg_risk            BOOLEAN
category            VARCHAR    -- from Gamma if available
trade_timestamp     TIMESTAMP  -- leader's original fill time
resolution_date     DATE       -- market.end_date
days_to_resolution  INT        -- at time of trade
resolution_bucket   VARCHAR    -- '2d', '7d', '30d', '60d'
leader_maker_side   VARCHAR    -- 'maker' or 'taker'
leader_trade_usd    DOUBLE     -- their notional
leader_price        DOUBLE
copy_price          DOUBLE     -- after slippage
copy_size_usd       DOUBLE     -- after sizing rule + caps
copy_token_amount   DOUBLE
position_resolution DOUBLE     -- 0 or 1, eventual outcome
copy_pnl_usd        DOUBLE     -- realised PnL on this copy
sizing_rule         VARCHAR    -- 'fixed_pct' or 'leader_proportional'
slippage_cents      DOUBLE
slippage_source     VARCHAR    -- 'next_fill' or 'fallback'
```

### 5.2 Per-Run Summary

For each backtest run, also write `data/backtests/{run_id}_summary.json`:

- Total PnL, total volume, signal count
- Sharpe, Sortino, max drawdown (over monthly PnL aggregates)
- Win rate, profit factor (per-position)
- Deployment ratio, signal frequency
- Category breakdown
- Top 10 contributing leaders, bottom 10
- Top 10 contributing markets, bottom 10

### 5.3 Cross-Run Comparison Notebook

A notebook at `notebooks/phase5_analysis.ipynb` reads all audit logs and produces:

- Heatmap: cohort × resolution_bucket × sizing_rule, coloured by Sharpe
- Slippage sensitivity surface per top combination
- Breakeven slippage table
- Category attribution per cohort
- Maker-only vs include-takers comparison

## 6. Methodological Risks and Mitigations

| Risk | Severity | Mitigation in v1 |
|------|----------|------------------|
| Lookahead via metrics | Critical | Compute IS metrics fresh per refresh date, filtering by `resolution_ts < T` |
| Lookahead via bankroll | Critical | Use `bankroll_timeseries.parquet` keyed on T |
| Lookahead via resolution price | Critical | Copy PnL uses resolution_price known at market.end_date, never the snapshot's latest value |
| Survivorship bias | Moderate | Cohort re-qualifies monthly; dropouts naturally absent from later windows |
| Capacity (infinite liquidity assumption) | Moderate | Documented limitation; v2 will add depth caps |
| Crowding (own fills move prices) | Low (solo bot) | Documented as out-of-scope |
| Merge/split blindspot (Pool C) | Low (Cohort 2 filters phantom_score < 3) | Phantom score filter mostly excludes arb traders; remaining residual documented |
| Category leakage (e.g. League of Legends) | Moderate | Empirical category denylist populated from Stage 1 diagnostics |
| Goldsky data tail lag | Low | All backtests end by 2026-04; data tail is 2026-04-23 |

## 7. Implementation Phases

### Stage 1: Main Backtest Matrix

- 3 cohorts × 4 resolution buckets × 2 sizing rules × 3 windows
- All at default slippage settings (`min_seconds=15`, `max_seconds=300`, `fallback_slippage_cents=3.0`)
- 72 backtest runs
- **Deliverable:** comparison heatmap, identification of top 5 cohort + bucket + sizing combinations

### Stage 2: Slippage Sensitivity on Top Performers

- Top 5 from Stage 1
- 25-combo grid each over (`min_seconds`, `max_seconds`) windows × `fallback_slippage_cents`
- 125 runs total (some skipped if breakeven obvious)
- **Deliverable:** slippage-robustness profile per top combination, including breakdown by `slippage_source` (`next_fill` vs `fallback`)

### Stage 3: Breakeven Analysis

- Top 2-3 combinations from Stage 2
- Fine-grid search on `fallback_slippage_cents` to find PnL = 0
- 20-30 runs
- **Deliverable:** "Cohort X breaks even at fallback Y¢" decision-useful numbers

### Stage 4: Final Report

- Synthesis document at `notes/copytrade/phase5_results.md`
- Recommendation: which cohort (if any) to deploy on execution side, at what slippage assumption, with what sizing rule, at what resolution-time rhythm

## 8. Success Criteria

A cohort passes Phase 5 if **at default slippage settings** (`min_seconds=15`, `max_seconds=300`, `fallback=3¢`):

- Annualised return > 20% on at least 2 of 3 test windows
- Sharpe > 1.0 on the stitched OOS curve
- Max drawdown < 30%
- Capital utilisation > 40%
- Sample size: > 100 OOS signals across the test window

A cohort is *robust* if at +2¢ on fallback (`fallback=5¢`), it still meets the criteria above with a slightly relaxed Sharpe (> 0.7).

Failure to meet criteria does not mean the cohort definition is wrong — it may mean the operational reality (slippage) is too expensive. Breakeven slippage from Stage 3 quantifies the gap.

## 9. Open Questions Deferred to Phase 5 v2

- CPCV: combinatorial purged cross-validation over many train/test splits
- Capacity/depth modelling
- Rolling IS window (vs expanding)
- Composite cohort scoring (weighted sum of metrics) vs hard thresholds
- Cohort overrides / hands-on hybrid mode (Tatv's Setup C)
- Per-trader leader-proportional sizing tuned to their specific bankroll volatility
- Crowding effects from concurrent copy-traders

---

**End of design.**
