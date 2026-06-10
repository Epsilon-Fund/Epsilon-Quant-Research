---
title: "Metrics reference — derived datasets"
created: 2026-06-05
status: closed
owner: justin
project: polymarket
para: resource
hubs:
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - research
  - polymarket
---
# Metrics reference — derived datasets

> Table terms: [[polymarket_table_dictionary]]

Every column in our derived parquets, with formula, source, edge cases, and
trustworthiness rating. Source code paths are relative to `polymarket/research/`.
Re-stamp Section A whenever data is refreshed.

---

## A. Data scope

| | |
|---|---|
| **Snapshot date** (this stamp) | 2026-05-10 |
| **Markets snapshot file** | `data/markets/markets_2026-05-06.parquet` |
| **Raw fill count (`raw_trades`)** | 1 064 500 317 |
| **Earliest fill timestamp** | 2022-11-21 19:50:09 UTC (warproxxx seed start) |
| **Latest fill timestamp** | 2026-04-24 00:00:00 UTC (Goldsky tail; subgraph indexer lags ~9 days behind real-time) |
| **`closed_positions.parquet`** | 269 974 929 rows, 2 576 698 addresses, 797 229 markets |
| **`traders.parquet`** | 2 576 698 rows (all addresses), 2 572 665 after `NOT is_operator_like` |
| **`bankroll_timeseries.parquet`** | 429 250 741 rows, 2 509 096 addresses, dates 2022-11-21 → 2028-01-02 (tail extended by far-future placeholder end_dates in Gamma) |
| **Cohort parquets** (`data/cohorts/*.parquet`) | 6 files: `high_sharpe_directional`, `high_profit_factor_with_size`, `negrisk_specialists`, `sports_directional_fast`, `patient_accumulators`, `high_kelly_edge` |

Sources: `raw_trades` view at [`sql/views.sql:28-31`](../sql/views.sql); the closed-position build at [`scripts/build_closed_positions.py:29-117`](../scripts/build_closed_positions.py); the traders build at [`scripts/build_traders_table.py:78-537`](../scripts/build_traders_table.py).

---

## B. SQL view chain

The chain is loaded by `data_infra.views.load_views(con)` against any DuckDB connection. Definitions live in [`sql/views.sql`](../sql/views.sql); placeholders `{TRADES_GLOB}`, `{SEED_PATH}`, `{MARKETS_PATH}`, `{TRADERS_PATH}` are substituted by the loader.

### B.1 `raw_trades` — file: [`sql/views.sql:28-31`](../sql/views.sql)

```sql
CREATE OR REPLACE VIEW raw_trades AS
SELECT * FROM read_parquet('{TRADES_GLOB}')
UNION ALL BY NAME
SELECT * FROM read_parquet('{SEED_PATH}');
```

**What it adds:** unifies all the per-shard delta parquets and the warproxxx seed parquet into one logical fill table. Schema is the canonical 13-column trade record (`timestamp`, `market_id`, `condition_id`, `neg_risk`, `maker`, `taker`, `maker_asset_id`, `taker_asset_id`, `usd_amount`, `token_amount`, `price`, `maker_side`, `transaction_hash`).

**What it filters:** nothing. Includes orphan fills (NULL `market_id`) and self-trades (`maker = taker`).

### B.2 `markets_tokens` (TABLE, not view) — file: [`sql/views.sql:38-52`](../sql/views.sql)

```sql
CREATE OR REPLACE TABLE markets_tokens AS
SELECT
    CAST(m.id AS VARCHAR) AS market_id,
    m.condition_id,
    m.neg_risk,
    m.closed,
    TRY_CAST(m.end_date AS TIMESTAMP) AS end_date,
    m.outcome_prices,
    m.outcomes,
    m.clob_token_ids,
    r.i AS outcome_index,
    m.clob_token_ids[r.i] AS outcome_token_id
FROM read_parquet('{MARKETS_PATH}') m,
     range(1, len(m.clob_token_ids) + 1) AS r(i)
WHERE len(m.clob_token_ids) > 0;
```

**What it adds:** unnests Gamma's `clob_token_ids` array into one row per `(market_id, outcome_index, outcome_token_id)`. Materialised as a TABLE (~2 M rows) so the JOIN against the 1 B-row `raw_trades` builds a hot hash table once.

**What it filters:** markets with no outcome tokens (`len(clob_token_ids) = 0`) — typically zero-row test markets.

### B.3 `joined_fills` — file: [`sql/views.sql:62-100`](../sql/views.sql)

```sql
CREATE OR REPLACE VIEW joined_fills AS
SELECT
    pre.timestamp, pre.market_id, pre.condition_id, pre.neg_risk,
    pre.maker, pre.taker, pre.maker_asset_id,
    pre.token_amount, pre.usd_amount, pre.price, pre.transaction_hash,
    pre.outcome_token_id, mt.outcome_index
FROM (
    SELECT
        rt.timestamp, rt.market_id, rt.condition_id, rt.neg_risk,
        rt.maker, rt.taker, rt.maker_asset_id,
        rt.token_amount, rt.usd_amount, rt.price, rt.transaction_hash,
        CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id
             ELSE rt.maker_asset_id END AS outcome_token_id
    FROM raw_trades rt
    WHERE rt.maker IS NOT NULL
      AND rt.taker IS NOT NULL
      AND rt.maker <> rt.taker
      AND rt.market_id IS NOT NULL
) pre
JOIN markets_tokens mt
    ON mt.market_id = pre.market_id
   AND mt.outcome_token_id = pre.outcome_token_id;
```

**What it adds vs `raw_trades`:** (1) `outcome_token_id` derived (the non-zero asset_id of the fill), (2) `outcome_index` joined from `markets_tokens` (1 or 2 for binary markets), (3) one row per fill (no explode yet), (4) `exchange_internal_match` (BOOLEAN, added 2026-06-10) — TRUE iff the fill's `taker` is one of the 4 CTF-Exchange exchange-internal-leg contracts (v1 `0x4bfb41…`, `0xc5d563…`; v2 `0xe11118…`, `0xe2222d…`). Such a row is the `_matchOrders` internal active leg of a two-sided match, where the wallet in the `maker` column was actually the ACTIVE (aggressor) order signer. The address list is hardcoded in the view; the source of truth is [`data_infra/operator_denylist.py`](../data_infra/operator_denylist.py) `EXCHANGE_INTERNAL_LEG` — keep in sync. See [[copytrade_attribution_repartition_findings]].

**What it filters:** rows with NULL maker, NULL taker, self-trades (`maker = taker`), orphan fills (NULL `market_id`), and fills whose `outcome_token_id` doesn't match any market in the snapshot. The CASE-as-join-key is materialised in the inner subquery so the hash join sees a plain column.

### B.4 `trader_actions` — file: [`sql/views.sql:106-140`](../sql/views.sql)

```sql
CREATE OR REPLACE VIEW trader_actions AS
SELECT
    timestamp, 'maker' AS role, maker AS address,
    market_id, condition_id, neg_risk, outcome_token_id, outcome_index,
    CASE WHEN maker_asset_id = '0' THEN  token_amount ELSE -token_amount END AS token_delta,
    CASE WHEN maker_asset_id = '0' THEN -usd_amount   ELSE  usd_amount   END AS usd_delta,
    token_amount, usd_amount, price, transaction_hash
FROM joined_fills
UNION ALL
SELECT
    timestamp, 'taker' AS role, taker AS address,
    market_id, condition_id, neg_risk, outcome_token_id, outcome_index,
    CASE WHEN maker_asset_id = '0' THEN -token_amount ELSE  token_amount END AS token_delta,
    CASE WHEN maker_asset_id = '0' THEN  usd_amount   ELSE -usd_amount   END AS usd_delta,
    token_amount, usd_amount, price, transaction_hash
FROM joined_fills;
```

**What it adds vs `joined_fills`:** the **explode** — every fill becomes 2 rows, one with `address = maker` and one with `address = taker`. Sign convention is from each address's POV: `token_delta > 0` ⇒ received outcome tokens, `usd_delta > 0` ⇒ received USDC. **Across the two rows of any single fill, sums are zero by construction** (sign-symmetry invariant, proven by `scripts/smoke_test_views.py`).

**`active_order_leg` (BOOLEAN, added 2026-06-10):** TRUE ⇔ this row's address signed the ACTIVE (aggressor) order of the fill. Derived from `joined_fills.exchange_internal_match` (see B.3): on a maker-role row it is TRUE iff the fill's taker is an exchange-internal-leg contract (`_matchOrders` bundle — the wallet labelled `maker` was the aggressor) and FALSE for a genuine passive maker; on a taker-role row it is TRUE for normal fills (the taker is the aggressor by definition) and FALSE only when the row's address IS the exchange contract itself — an artifact leg, not a trader, which should be excluded from per-trader style metrics. **PnL/position attribution is unaffected by this flag — it is style framing only.** Style ratios computed without it overstate maker-ness (e.g., Domah's maker:taker ratio 7.89 → 5.67 after reclassification). See [[copytrade_attribution_repartition_findings]].

**What it filters:** inherits from `joined_fills` (no NULL/self-trade/orphan fills, outcome_index resolved).

**Caveat:** referencing this view multiple times in a single query (e.g., a SEMI JOIN with `IN (subquery)`) triggers DuckDB CTE auto-materialisation of the 2 B-row exploded result — in our smoke tests that consumed 178 GB of temp before crashing. **Always reference once per query**, or pre-filter at `joined_fills` level.

### B.5 `trader_actions_orphan` — file: [`sql/views.sql:158-185`](../sql/views.sql)

Same explode shape as `trader_actions` but filtered to **orphan fills** (`market_id IS NULL`). No `outcome_index`. Audit-only — never feeds position math.

### B.6 `traders_raw` — file: [`sql/views.sql:149-150`](../sql/views.sql)

```sql
CREATE OR REPLACE VIEW traders_raw AS
SELECT * FROM read_parquet('{TRADERS_PATH}');
```

**What it adds:** registers `data/traders.parquet` as a SQL-queryable view. Loaded conditionally — if the parquet doesn't exist yet (first build), the loader silently skips this and the next view.

### B.7 `traders_filtered` — file: [`sql/views.sql:152-153`](../sql/views.sql)

```sql
CREATE OR REPLACE VIEW traders_filtered AS
SELECT * FROM traders_raw WHERE NOT is_operator_like;
```

**What it adds vs `traders_raw`:** drops 4 033 rows where `is_operator_like = TRUE` (12 hardcoded operator/MM-bot addresses + heuristic flags from the final-join SQL at `scripts/build_traders_table.py:501-516`). 2 572 665 rows remain.

**What it filters:** see Section C — "What addresses are excluded by `traders_filtered`".

---

## C. Glossary of derived concepts

### What is a "closed position" in this dataset

One row in `closed_positions.parquet`, keyed by `(address, market_id, outcome_index)`. A position exists for an address when they participated in **any fill** (as maker or taker) on a market that is `closed = TRUE` in the Gamma snapshot, on a specific outcome of that market. Open markets are excluded entirely. A trader who held both YES and NO of a single market produces **two** rows (one per outcome), even if the trader sees them as a single bet.

### What is the redemption synthesis

When a market resolves, the winning outcome's tokens are redeemable at $1 and the losing outcome's tokens at $0. **This redemption is an on-chain event but not an `OrderFilled` event** — the trader interacts with the CTF contract directly, and our `raw_trades` doesn't see it. To close the books on positions held to resolution, the build at [`scripts/build_closed_positions.py:103-110`](../scripts/build_closed_positions.py) synthesises a redemption:

```sql
redemption_value = final_token_position * resolution_price
realised_pnl     = realised_cash_flow + redemption_value
```

Where `resolution_price = CAST(outcome_prices[outcome_index] AS DOUBLE)` from the markets snapshot (≈ 1.0 for the winning outcome, ≈ 0.0 for the losing one, with tiny floating-point fuzz like 0.99999998… in practice).

For a position fully exited via trades (`final_token_position = 0`), redemption is $0 and `realised_pnl = realised_cash_flow`. For a position held to resolution (`final_token_position ≠ 0`), the redemption term carries the entire payout.

### How is `outcome_index` derived

For each fill, the `outcome_token_id` is the non-zero side's asset id (`'0'` is the USDC token; the outcome token's id is a uint256 string). At [`sql/views.sql:90`](../sql/views.sql):

```sql
CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id
     ELSE rt.maker_asset_id END AS outcome_token_id
```

Then `outcome_index` is the **1-based position** of that token in `markets.clob_token_ids`, materialised in the `markets_tokens` table at [`sql/views.sql:48-49`](../sql/views.sql):

```sql
range(1, len(m.clob_token_ids) + 1) AS r(i)
m.clob_token_ids[r.i] AS outcome_token_id
```

For binary markets (the vast majority), `outcome_index ∈ {1, 2}`. The validation report (Phase 0) confirmed 35 / 35 sampled closed markets reconciled cleanly: the trade-implied resolved outcome (last-50-trade median price closest to 1.0) matched the markets parquet's `outcome_prices` winner.

### What does `maker_side` mean

`maker_side` is a column on `raw_trades`/`joined_fills` with values `'BUY'` or `'SELL'`, **from the maker's perspective**:
- `maker_side = 'BUY'` ⇔ `maker_asset_id = '0'` (maker paid USDC, received outcome tokens)
- `maker_side = 'SELL'` ⇔ `maker_asset_id ≠ '0'` (maker delivered outcome tokens, received USDC)

The taker is always on the opposite side of the same fill. Don't read `maker_side` as the trader's intent — read it as the role of the maker in this specific fill.

### What addresses are excluded by `traders_filtered`

`is_operator_like = TRUE` is set in [`scripts/build_traders_table.py:501-516`](../scripts/build_traders_table.py) by the disjunction:

1. **Hardcoded deny-list** — `address IN OPERATOR_ADDRESSES` from [`data_infra/operator_denylist.py`](../data_infra/operator_denylist.py): 2 pure relayers (`0x4bfb41…`, `0xc5d563…`), 7 pure MM bots (extreme maker:taker ratios), 3 HFT cluster addresses.
2. **Maker:taker ratio > 50** — pure liquidity-providing bots.
3. **Maker:taker ratio < 0.02** — pure crossing/relayer addresses.
4. **`n_distinct_counterparties > 500 000`** — fan-out implies aggregator/matcher.
5. **`style_pct_sub_second > 95.0` AND `n_fills_total > 1 000 000`** — high-volume HFT.

These five conditions are computed from columns already in `traders.parquet`. The filter drops 4 033 rows out of 2 576 698.

### Definition of "maker" vs "taker" in our schema

Per fill, the `maker` and `taker` columns are passed through verbatim from the on-chain `OrderFilled` event emitted by the CTF Exchange contract. The contract designates the address whose signed limit order was resting as the maker, and the address whose action crossed it as the taker. **This is a per-fill role, not a trader type.** The same address is maker on some fills and taker on others — ~98.5 % of addresses in our data appear as both.

### Position-level vs market-level metrics — the collapse

`closed_positions.parquet` has one row per `(address, market_id, outcome_index)` — separate rows for YES and NO of a single market. `pos_*` metrics aggregate directly across these rows. `mkt_*` metrics collapse first to `(address, market_id)` by **summing `realised_pnl` across outcomes within the market**, then aggregate across markets.

**SQL implementation** at [`scripts/build_traders_table.py:185-262`](../scripts/build_traders_table.py):

```sql
WITH per_market AS (
    SELECT address, market_id,
           sum(realised_pnl) AS market_pnl,        -- ← this is the collapse
           any_value(resolution_ts) AS market_resolution_ts,
           any_value(neg_risk) AS neg_risk,
           sum(gross_usd_volume) AS market_volume
    FROM closed_positions
    GROUP BY address, market_id
),
agg AS (
    SELECT address,
           count(*) AS mkt_n_markets_traded,
           sum(market_pnl) AS mkt_total_pnl,
           ...
    FROM per_market GROUP BY address
)
```

**Why `mkt_*` is NegRisk-robust where `pos_*` is not:** Polymarket NegRisk markets allow traders to mint USDC into a YES+NO pair, sell one side and keep the other (split), or merge a YES+NO pair back into USDC. **These mints/merges are not `OrderFilled` events** — our trades data sees the trader's positions on each outcome but not the mint/merge tying them together. At the position level, a NegRisk arb trader looks like a winner on one outcome and an equal-sized loser on the other; their `pos_total_pnl = sum of both` is correct in aggregate but their per-position PnLs are inflated/deflated by the mint amount they can't see. **Collapsing to market level cancels these phantom legs against each other.** The diagnostic for this is `phantom_position_score` (Section 2 below): it equals 1.0 for clean directional traders and grows large for split/merge arb traders.

---

## D. Self-consistency invariants

Each invariant is automatically computed by a sanity script. The ones below are the ones load-bearing for the metrics in this doc.

### D.1 Σ realised_pnl across all closed positions ≈ $0

**Source:** [`scripts/build_closed_positions.py:249-256`](../scripts/build_closed_positions.py)

```sql
SELECT round(sum(realised_pnl), 0) AS total_realised,
       round(sum(CASE WHEN neg_risk THEN realised_pnl ELSE 0 END), 0) AS negrisk_pnl,
       round(sum(CASE WHEN NOT neg_risk THEN realised_pnl ELSE 0 END), 0) AS regular_pnl,
       count(*) AS total_positions
FROM closed_positions
```

**Last result:** total_realised = $0, negrisk_pnl = $0, regular_pnl = $0 (across 269 974 929 positions).

**What it covers:** every winner's redemption payout was matched by a loser's lost stake on the resolved outcome. Confirms the redemption-synthesis math is internally consistent.

**What it doesn't cover:**
- **Open positions** — filtered out (`mt.closed = TRUE`). A trader's lifetime book includes open positions whose final PnL is unknown.
- **Operator/MM fees** — extracted via the operator addresses' nominal "PnL" (e.g., `0x4bfb41…` shows $61 M on our books from accumulated matching flow, not real edge).
- **Orphan markets** — fills with NULL `market_id` are excluded from `joined_fills`. ~0.4 % of fills.
- **NegRisk mints/merges** — the trader's own bookkeeping (split USDC into YES+NO, merge them back) is not an `OrderFilled` event. Our `realised_pnl` over a NegRisk arb position is correct in `mkt_total_pnl` but inflated in `pos_total_pnl`.

### D.2 `mkt_total_pnl` and `pos_total_pnl` agree to the cent

**Source:** [`scripts/sanity_phase3.py:24-33`](../scripts/sanity_phase3.py)

```sql
SELECT
    count(*) FILTER (WHERE abs(mkt_total_pnl - pos_total_pnl) > 1.0
                      AND n_closed_positions > 10) AS n_drift_above_1usd,
    count(*) FILTER (WHERE abs(mkt_total_pnl - pos_total_pnl) > 0.01
                      AND n_closed_positions > 10) AS n_drift_above_1cent,
    round(max(abs(mkt_total_pnl - pos_total_pnl)), 4) AS max_abs_drift,
    count(*) AS total_rows
FROM traders
```

**Last result:** n_drift_above_1usd = 0, n_drift_above_1cent = 0, max_abs_drift = $0.0000 across 2 576 698 rows.

**What it covers:** confirms the position→market collapse is just a re-grouping of `realised_pnl`, not a different number. Both `pos_total_pnl` and `mkt_total_pnl` sum the same column.

**What it doesn't:** the drift check only looks at `total` aggregates. **Other** `pos_*` vs `mkt_*` metrics legitimately differ (win_rate, sharpe, profit_factor) — that's the whole point of the dual representation.

### D.3 Cohort guards held — no artifact-shaped row in any pool

**Source:** [`scripts/build_cohorts.py:165-172`](../scripts/build_cohorts.py)

```sql
SELECT
    sum(CASE WHEN mkt_sharpe > 100 THEN 1 ELSE 0 END) AS sharpe_above_100,
    sum(CASE WHEN mkt_kelly_fraction > 1 THEN 1 ELSE 0 END) AS kelly_above_1,
    sum(CASE WHEN pos_std_pnl < 0.01 THEN 1 ELSE 0 END) AS pos_std_below_1cent,
    sum(CASE WHEN mkt_std_pnl < 0.01 THEN 1 ELSE 0 END) AS mkt_std_below_1cent
FROM _all_pool_rows
```

**Last result:** all four counts = 0 across the 14 053 pool rows (union with duplicates).

**What it covers:** the `mkt_std_pnl > 1.0`, `n_closed_positions > 200`, `active_days > 90` guards in pool definitions actually hold for every qualifying address.

**What it doesn't:** doesn't validate the underlying metrics — only that the artifact tail was excluded.

### D.4 External reconciliation (informational, not pass/fail)

**Source:** [`scripts/api_reconciliation.py:1-200`](../scripts/api_reconciliation.py)

For 10 hand-picked traders (3 directional, 3 NegRisk-heavy, 2 operators, 2 random middle), fetches `data-api.polymarket.com` `/value`, `/positions` (paginated), `/activity`. Output: [`notes/overview/data_quality/api_reconciliation_v1.md`](../notes/overview/data_quality/api_reconciliation_v1.md).

**What it covers:** order-of-magnitude sanity that our lifetime closed-position PnL plus current portfolio value lands in a plausible total range.

**What it doesn't:** the public API doesn't expose lifetime P&L (only currently-open mark-to-market and partial-realized within open positions). **Strict reconciliation requires the Polymarket UI**, which is not in scope. **Treat differences <20% as internally normal.**

---

## 1. `data/closed_positions.parquet` columns

One row per `(address, market_id, outcome_index)` on resolved markets. Built by [`scripts/build_closed_positions.py:29-117`](../scripts/build_closed_positions.py).

### 1.1 Identity

#### `address`
- **Type:** VARCHAR
- **Definition:** lowercased Ethereum address of the trader.
- **Source:** [`build_closed_positions.py:39`](../scripts/build_closed_positions.py): `CASE s.role WHEN 'maker' THEN jf.maker ELSE jf.taker END AS address`
- **Inputs:** `joined_fills.maker`, `joined_fills.taker`, `(VALUES ('maker'), ('taker'))` cross-join sides table
- **Edge cases:** rows where `maker = taker` are pre-filtered at `joined_fills` level. NULL maker/taker pre-filtered. Lowercase guaranteed by upstream parquets.
- **Trustworthiness:** STRONG.

#### `market_id`
- **Type:** VARCHAR
- **Definition:** Polymarket market id, stringified.
- **Source:** [`build_closed_positions.py:40`](../scripts/build_closed_positions.py): passthrough from `joined_fills.market_id`
- **Inputs:** `raw_trades.market_id`
- **Edge cases:** orphan fills (NULL `market_id`) are excluded by `joined_fills`. ~0.4 % of raw fills.
- **Trustworthiness:** STRONG.

#### `condition_id`
- **Type:** VARCHAR
- **Definition:** CTF `condition_id` (the on-chain market identifier; market_id is the Gamma id).
- **Source:** [`build_closed_positions.py:40`](../scripts/build_closed_positions.py): passthrough
- **Trustworthiness:** STRONG.

#### `neg_risk`
- **Type:** BOOLEAN
- **Definition:** `TRUE` iff the market is on the NegRisk CTF Exchange variant. NegRisk markets allow split/merge against linked markets in a category.
- **Source:** [`build_closed_positions.py:40`](../scripts/build_closed_positions.py): passthrough
- **Trustworthiness:** STRONG, but read in conjunction with `phantom_position_score` for arb-aware analysis.

#### `outcome_token_id`
- **Type:** VARCHAR
- **Definition:** uint256 outcome-token id (the non-zero asset side of every fill on this position).
- **Source:** [`build_closed_positions.py:41`](../scripts/build_closed_positions.py)
- **Inputs:** derived in `joined_fills` (Section B.3)
- **Trustworthiness:** STRONG.

#### `outcome_index`
- **Type:** INTEGER
- **Definition:** 1-based position of this outcome's token within `markets.clob_token_ids`. For binary markets, ∈ {1, 2}.
- **Source:** [`build_closed_positions.py:41`](../scripts/build_closed_positions.py): from `markets_tokens.outcome_index`
- **Edge cases:** fills whose `outcome_token_id` doesn't appear in any market's `clob_token_ids` are excluded by `joined_fills` (rare token-drift cases). 33 multi-outcome markets in Gamma have indices 3-7; the vast majority are binary.
- **Trustworthiness:** STRONG. Validation report sampled 35 markets and 35 / 35 reconciled.

### 1.2 Activity per position

#### `n_fills`
- **Type:** BIGINT
- **Definition:** count of `trader_actions` rows for this address+market+outcome — equals the count of fills the address participated in (on either maker or taker side).
- **Source:** [`build_closed_positions.py:76`](../scripts/build_closed_positions.py): `COUNT(*) AS n_fills`
- **Inputs:** the `actions` CTE (the inline maker/taker explode of `joined_fills` for closed markets)
- **Edge cases:** self-trades pre-filtered, so a trader is never on both sides of a single fill. Multiple atomic-bucket fills (same `transaction_hash` + same trader + same outcome) are counted individually — there's no same-bucket collapse in this build (deferred to v2).
- **Trustworthiness:** STRONG.

#### `first_fill_ts`
- **Type:** TIMESTAMP
- **Definition:** timestamp of the trader's earliest fill on this position.
- **Source:** [`build_closed_positions.py:77`](../scripts/build_closed_positions.py): `MIN(timestamp) AS first_fill_ts`
- **Trustworthiness:** STRONG.

#### `last_fill_ts`
- **Type:** TIMESTAMP
- **Definition:** timestamp of the trader's latest fill on this position.
- **Source:** [`build_closed_positions.py:78`](../scripts/build_closed_positions.py): `MAX(timestamp) AS last_fill_ts`
- **Trustworthiness:** STRONG.

#### `resolution_ts`
- **Type:** TIMESTAMP
- **Definition:** market end-date from Gamma, when the market resolved (= when redemption became live).
- **Source:** [`build_closed_positions.py:95`](../scripts/build_closed_positions.py): `mt.end_date AS resolution_ts`
- **Inputs:** `markets_tokens.end_date = TRY_CAST(m.end_date AS TIMESTAMP)` from [`sql/views.sql:44`](../sql/views.sql)
- **Edge cases:** Gamma's `end_date` is VARCHAR but parses cleanly via `TRY_CAST` for closed markets in our snapshot (0 NULLs, 0 unparseable). Some markets have placeholder pre-Polymarket-era end_dates (2011, 2018) — surfaces as `holding_duration_hours < 0` (~22 M rows, 8.2 % of positions).
- **Trustworthiness:** MODERATE. The 8 % with negative duration suggests Gamma's `end_date` is sometimes set to a placeholder before the market actually trades.

#### `holding_duration_hours`
- **Type:** DOUBLE
- **Definition:** hours from first fill to resolution.
- **Source:** [`build_closed_positions.py:96-98`](../scripts/build_closed_positions.py): `epoch(mt.end_date - agg.first_fill_ts) / 3600.0`
- **Edge cases:** `NULL` when `resolution_ts IS NULL`. **Negative when resolution_ts < first_fill_ts** (placeholder Gamma `end_date` issue). Filter `holding_duration_hours >= 0` for any duration aggregation.
- **Trustworthiness:** MODERATE. ~8 % of rows are negative or non-physical.

### 1.3 Volume

#### `gross_token_volume`
- **Type:** DOUBLE
- **Definition:** sum of unsigned `token_amount` across all fills on this position (counts both buys and sells).
- **Source:** [`build_closed_positions.py:74`](../scripts/build_closed_positions.py): `SUM(token_amount) AS gross_token_volume`
- **Note:** this **double-counts** within the maker/taker explode — for a trader's single fill, both their +X and the counterparty's −X carry `token_amount = X`. Across one trader's positions, it represents that trader's total face-value token activity (buy + sell sides summed).
- **Trustworthiness:** STRONG (mechanically correct), MODERATE for "volume" intuition (it's gross face value, not net economic exposure).

#### `gross_usd_volume`
- **Type:** DOUBLE
- **Definition:** sum of unsigned `usd_amount` across all fills on this position.
- **Source:** [`build_closed_positions.py:75`](../scripts/build_closed_positions.py): `SUM(usd_amount) AS gross_usd_volume`
- **Same caveat as `gross_token_volume`.**
- **Trustworthiness:** STRONG.

#### `total_bought_usd`
- **Type:** DOUBLE
- **Definition:** sum of `usd_amount` across fills where this address received outcome tokens (`token_delta > 0`).
- **Source:** [`build_closed_positions.py:79`](../scripts/build_closed_positions.py): `SUM(CASE WHEN token_delta > 0 THEN usd_amount ELSE 0 END)`
- **Trustworthiness:** STRONG.

#### `total_sold_usd`
- **Type:** DOUBLE
- **Definition:** sum of `usd_amount` across fills where this address gave up outcome tokens (`token_delta < 0`).
- **Source:** [`build_closed_positions.py:80`](../scripts/build_closed_positions.py): `SUM(CASE WHEN token_delta < 0 THEN usd_amount ELSE 0 END)`
- **Trustworthiness:** STRONG.

### 1.4 PnL

#### `final_token_position`
- **Type:** DOUBLE
- **Definition:** trader's residual outcome-token holding when the market resolved, in token units.
- **Source:** [`build_closed_positions.py:72`](../scripts/build_closed_positions.py): `SUM(token_delta) AS final_token_position`
- **Edge cases:** can be negative (net short via splits we don't see); typically positive. `> 1e-6` threshold defines `is_held_to_resolution`.
- **Caveat:** for NegRisk arb traders who minted YES+NO pairs and never merged them back, our number is correct **for the YES-side fills minus YES-side trades** but is missing the offsetting NO-side mint. Their per-outcome `final_token_position` may understate true holdings.
- **Trustworthiness:** STRONG for non-NegRisk traders; MODERATE for NegRisk arbs.

#### `realised_cash_flow`
- **Type:** DOUBLE
- **Definition:** net USDC received minus paid by this trader on this position from on-book fills only.
- **Source:** [`build_closed_positions.py:73`](../scripts/build_closed_positions.py): `SUM(usd_delta) AS realised_cash_flow`
- **Trustworthiness:** STRONG.

#### `resolution_price`
- **Type:** DOUBLE
- **Definition:** the market's resolution price for this outcome — typically very close to 0.0 or 1.0. Pulled from `markets.outcome_prices[outcome_index]`.
- **Source:** [`build_closed_positions.py:105`](../scripts/build_closed_positions.py): `CAST(mt.outcome_prices[agg.outcome_index] AS DOUBLE) AS resolution_price`
- **Edge cases:** `outcome_prices` is `LIST<VARCHAR>` in the markets parquet; the cast handles the conversion. Floating-point fuzz: many rows have values like 0.99999983… instead of exactly 1.0 — operationally identical at redemption.
- **Trustworthiness:** STRONG.

#### `redemption_value`
- **Type:** DOUBLE
- **Definition:** cash equivalent of the residual position at the resolution price.
- **Source:** [`build_closed_positions.py:106-107`](../scripts/build_closed_positions.py): `final_token_position * resolution_price`
- **Trustworthiness:** STRONG for non-NegRisk; MODERATE for NegRisk-arb residuals (see `final_token_position`).

#### `realised_pnl`
- **Type:** DOUBLE
- **Definition:** lifetime realised PnL on this position = on-book cash flow + redemption.
- **Source:** [`build_closed_positions.py:108-109`](../scripts/build_closed_positions.py): `realised_cash_flow + final_token_position * resolution_price`
- **Edge cases:** sign convention follows trader's POV — positive = profit. **Sums to $0 across all 270 M positions** (invariant D.1).
- **Caveat:** for a NegRisk arb trader, this is **inflated on the winning outcome and equally deflated on the losing outcome** — they net out at the market level (`mkt_total_pnl`), not the position level. Use `mkt_total_pnl` if you want to rank arb traders.
- **Trustworthiness:** STRONG at trader-aggregate level, MODERATE per-position for NegRisk arbs.

### 1.5 Diagnostic

#### `peak_fill_abs_token`
- **Type:** DOUBLE
- **Definition:** the largest absolute `token_delta` of any single fill on this position. v1 proxy for "did this position ever get large".
- **Source:** [`build_closed_positions.py:81`](../scripts/build_closed_positions.py): `MAX(abs(token_delta)) AS peak_fill_abs_token`
- **Caveat:** **NOT a true peak position size.** True running-cumulative-max would have required sorting the 1.4 B-row collapsed table — OOM'd at 300 GB temp during the build. This is a **lower bound** on the running peak (always ≤ true peak). Phase 4+ can refine.
- **Trustworthiness:** SUSPECT for "peak position size" interpretation. Trustworthy as `MAX(|fill size|)`.

#### `is_held_to_resolution`
- **Type:** BOOLEAN
- **Definition:** `TRUE` iff `abs(final_token_position) > 1e-6`, i.e. the trader did not fully exit via trades.
- **Source:** [`build_closed_positions.py:111`](../scripts/build_closed_positions.py): `abs(agg.final_token_position) > 1e-6 AS is_held_to_resolution`
- **Last summary:** 260 381 355 / 269 974 929 = **96.4 %** of positions held to resolution. Confirms the validation-report finding that most positions don't close via trades.
- **Trustworthiness:** STRONG.

---

## 2. `data/traders.parquet` columns

One row per address with at least one closed-position activity. Built by [`scripts/build_traders_table.py:78-537`](../scripts/build_traders_table.py). Grouped here by metric family.

### 2.1 Activity

#### `address`
- **Type:** VARCHAR. **Source:** [`build_traders_table.py:103`](../scripts/build_traders_table.py). **Trustworthiness:** STRONG.

#### `n_closed_positions`
- **Type:** BIGINT
- **Definition:** count of distinct `(market_id, outcome_index)` pairs the trader has a row for in `closed_positions`. Equals `n_distinct_outcomes`.
- **Formula:** [`build_traders_table.py:104`](../scripts/build_traders_table.py): `count(*) AS n_closed_positions`
- **Inputs:** `closed_positions`
- **Trustworthiness:** STRONG.

#### `n_distinct_markets`
- **Type:** BIGINT
- **Definition:** distinct `market_id` count for this address (≤ `n_closed_positions` since both YES and NO of one market count as 2 positions but 1 market).
- **Formula:** [`build_traders_table.py:105`](../scripts/build_traders_table.py): `count(DISTINCT market_id) AS n_distinct_markets`
- **Trustworthiness:** STRONG.

#### `n_distinct_outcomes`
- **Type:** BIGINT
- **Definition:** alias for `n_closed_positions` (one row per outcome). Carried separately for downstream readability.
- **Formula:** [`build_traders_table.py:135`](../scripts/build_traders_table.py): `n_closed_positions AS n_distinct_outcomes`
- **Trustworthiness:** STRONG.

#### `n_fills_total`
- **Type:** BIGINT
- **Definition:** sum of `n_fills` across all the trader's closed positions. Equals the number of times this address was a side of a fill on closed markets.
- **Formula:** [`build_traders_table.py:106`](../scripts/build_traders_table.py): `sum(n_fills) AS n_fills_total`
- **Trustworthiness:** STRONG.

#### `total_volume_usd`
- **Type:** DOUBLE
- **Definition:** sum of `(total_bought_usd + total_sold_usd)` across the trader's positions — gross face-value USD activity.
- **Formula:** [`build_traders_table.py:107`](../scripts/build_traders_table.py): `sum(total_bought_usd + total_sold_usd) AS total_volume_usd`
- **Trustworthiness:** STRONG.

#### `first_activity_ts`
- **Type:** TIMESTAMP
- **Definition:** earliest `first_fill_ts` across the trader's positions.
- **Formula:** [`build_traders_table.py:108`](../scripts/build_traders_table.py): `min(first_fill_ts)`
- **Trustworthiness:** STRONG.

#### `last_activity_ts`
- **Type:** TIMESTAMP
- **Definition:** latest `last_fill_ts` across the trader's positions.
- **Formula:** [`build_traders_table.py:109`](../scripts/build_traders_table.py): `max(last_fill_ts)`
- **Trustworthiness:** STRONG.

#### `active_days`
- **Type:** BIGINT
- **Definition:** integer days between first and last activity.
- **Formula:** [`build_traders_table.py:140`](../scripts/build_traders_table.py): `DATEDIFF('day', first_activity_ts, last_activity_ts)`
- **Edge cases:** can be 0 for single-day traders. Sharpe annualisation uses `GREATEST(active_days, 30)` to floor against this.
- **Trustworthiness:** STRONG.

### 2.2 Position-level PnL (`pos_*` family)

These metrics aggregate over `closed_positions` rows — one row per `(address, market_id, outcome_index)`. **Inflated for NegRisk arb traders** (see Section C glossary). Use `mkt_*` versions as the NegRisk-robust alternative.

#### `pos_total_pnl`
- **Type:** DOUBLE
- **Definition:** sum of `realised_pnl` across all the trader's positions.
- **Formula:** [`build_traders_table.py:110`](../scripts/build_traders_table.py): `sum(realised_pnl) AS pos_total_pnl`
- **Note:** `pos_total_pnl == mkt_total_pnl` to the cent (invariant D.2). The two diverge in higher-moment metrics, not in totals.
- **Trustworthiness:** STRONG.

#### `pos_winners`
- **Type:** BIGINT
- **Definition:** count of positions with `realised_pnl > 0`.
- **Formula:** [`build_traders_table.py:111`](../scripts/build_traders_table.py): `sum(CASE WHEN realised_pnl > 0 THEN 1 ELSE 0 END)`
- **Trustworthiness:** STRONG mechanically, MODERATE as a "wins" indicator for NegRisk arbs (each arb trade produces one winner + one loser at position level).

#### `pos_losers`
- **Type:** BIGINT
- **Formula:** [`build_traders_table.py:112`](../scripts/build_traders_table.py): `sum(CASE WHEN realised_pnl < 0 THEN 1 ELSE 0 END)`
- **Trustworthiness:** as `pos_winners`.

#### `pos_win_rate`
- **Type:** DOUBLE
- **Definition:** `pos_winners / (pos_winners + pos_losers)`. Excludes flat positions (`realised_pnl = 0`) from the denominator.
- **Formula:** [`build_traders_table.py:144-146`](../scripts/build_traders_table.py)
- **Edge cases:** NULL when winners + losers = 0.
- **Trustworthiness:** MODERATE. NegRisk arb traders have artificially symmetric win rates (~0.5) because each arb pair produces one of each.

#### `pos_dollar_win_rate`
- **Type:** DOUBLE
- **Definition:** `Σwinning_pnl / (Σwinning_pnl + Σ|losing_pnl|)`. Dollar-weighted version of `pos_win_rate`.
- **Formula:** [`build_traders_table.py:147-149`](../scripts/build_traders_table.py)
- **Trustworthiness:** STRONG. More informative than `pos_win_rate` because it weights by stake.

#### `pos_avg_win_usd`
- **Type:** DOUBLE
- **Definition:** mean PnL across winning positions (positive number).
- **Formula:** [`build_traders_table.py:150-151`](../scripts/build_traders_table.py): `pos_sum_win / pos_winners`
- **Edge cases:** NULL when no winners.
- **Trustworthiness:** STRONG.

#### `pos_avg_loss_usd`
- **Type:** DOUBLE
- **Definition:** mean magnitude of loss across losing positions (positive number).
- **Formula:** [`build_traders_table.py:152-153`](../scripts/build_traders_table.py): `pos_sum_loss_abs / pos_losers`
- **Trustworthiness:** STRONG.

#### `pos_profit_factor`
- **Type:** DOUBLE
- **Definition:** `Σwinning_pnl / Σ|losing_pnl|`, **capped at 100**.
- **Formula:** [`build_traders_table.py:154-156`](../scripts/build_traders_table.py): `LEAST(pos_sum_win / pos_sum_loss_abs, 100.0)`
- **Edge cases:** NULL when no losses.
- **Trustworthiness:** STRONG. Capping protects against degenerate near-zero denominators.

#### `pos_sharpe`
- **Type:** DOUBLE
- **Definition:** **naive annualised Sharpe** over per-position PnL.
- **Formula:** [`build_traders_table.py:157-162`](../scripts/build_traders_table.py)

  ```sql
  pos_mean_pnl / pos_std_pnl
   * sqrt(n_closed_positions::DOUBLE
          / (GREATEST(DATEDIFF('day', first_activity_ts,
                                last_activity_ts), 30) / 365.25))
  ```

  where `pos_mean_pnl = avg(realised_pnl)` and `pos_std_pnl = stddev_pop(realised_pnl)` ([`build_traders_table.py:117-118`](../scripts/build_traders_table.py)).
- **Edge cases:** NULL when `pos_std_pnl ≤ 0`. `GREATEST(active_days, 30)` floor prevents division blowup for single-day traders. Despite this, the floor doesn't fully prevent annualisation artifacts: p99 across `traders_filtered` (n_pos > 50) was **14.90** and max was **1.66×10¹⁵** in Phase 3 sanity.
- **Trustworthiness:** **SUSPECT**. **Use as DIAGNOSTIC, not a primary ranker.** Trustworthy only when `n_closed_positions > 200 AND active_days > 90 AND pos_std_pnl > 1.0` (the cohort guards). Even with guards, prefer `pos_profit_factor` and `pos_dollar_win_rate` for ranking.

#### `pos_sortino`
- **Type:** DOUBLE
- **Definition:** Sharpe variant using downside-only standard deviation; same naive annualisation.
- **Formula:** [`build_traders_table.py:163-168`](../scripts/build_traders_table.py): `pos_mean_pnl / pos_downside_std * sqrt(N/years)` where `pos_downside_std = sqrt(avg(CASE WHEN realised_pnl < 0 THEN realised_pnl² ELSE 0 END))`.
- **Trustworthiness:** **SUSPECT**, same caveats as `pos_sharpe`.

#### `pos_kelly_fraction`
- **Type:** DOUBLE
- **Definition:** Kelly criterion fraction = `p − (1−p)·(avg_loss / avg_win)` where `p = pos_win_rate`.
- **Formula:** [`build_traders_table.py:170-175`](../scripts/build_traders_table.py)
- **Edge cases:** NULL when no winners or no losers. Can be negative (negative-edge bettor).
- **Caveat:** assumes win/loss distributions are stationary and Kelly-applicable. For prediction markets with bimodal payoffs, this is a rough indicator at best. Cohort filter caps at 0.5 (`> 0.5` implies degenerate sample).
- **Trustworthiness:** MODERATE. Useful as a relative ranker; don't read as a literal sizing recommendation.

### 2.3 Market-level PnL (`mkt_*` family) — NegRisk-robust

These metrics first collapse `closed_positions` to one row per `(address, market_id)` by summing `realised_pnl` across outcomes, then aggregate. NegRisk arb pairs cancel at this stage. Sources at [`build_traders_table.py:183-262`](../scripts/build_traders_table.py).

#### `mkt_n_markets_traded`
- **Type:** BIGINT
- **Formula:** `count(*)` over the per-market grouping. **Trustworthiness:** STRONG.

#### `mkt_total_pnl`
- **Type:** DOUBLE
- **Definition:** `sum(market_pnl) where market_pnl = sum(realised_pnl) per (address, market_id)`. Equals `pos_total_pnl` to the cent (invariant D.2).
- **Trustworthiness:** STRONG. **Primary cohort metric.**

#### `mkt_winners` / `mkt_losers` / `mkt_win_rate` / `mkt_dollar_win_rate`
- **Type:** BIGINT / BIGINT / DOUBLE / DOUBLE
- **Definitions:** identical structure to `pos_*` versions but operating on `market_pnl` (per-market summed PnL) instead of per-position.
- **Formulas:** [`build_traders_table.py:202-232`](../scripts/build_traders_table.py)
- **Trustworthiness:** STRONG. NegRisk-robust by construction.

#### `mkt_avg_win_usd` / `mkt_avg_loss_usd`
- **Same structure** as `pos_*`. **Trustworthiness:** STRONG.

#### `mkt_profit_factor`
- **Type:** DOUBLE. Capped at 100.
- **Formula:** [`build_traders_table.py:237-239`](../scripts/build_traders_table.py): `LEAST(mkt_sum_win / mkt_sum_loss_abs, 100.0)`
- **Trustworthiness:** STRONG. **Primary cohort ranker.**

#### `mkt_sharpe`
- **Type:** DOUBLE
- **Definition:** naive annualised Sharpe over per-market PnL.
- **Formula:** [`build_traders_table.py:240-245`](../scripts/build_traders_table.py): `mkt_mean_pnl / mkt_std_pnl * sqrt(mkt_n_markets / years_resolved)`. Time normalisation uses `first_res_ts` to `last_res_ts` (resolution timestamps), not first/last activity.
- **Trustworthiness:** **SUSPECT** for the same reasons as `pos_sharpe`. p99 = 11.17, max = 107.20 in Phase 3 sanity. Use as DIAGNOSTIC only, with cohort guards.

#### `mkt_sortino`
- **Same caveats** as `mkt_sharpe`. **Trustworthiness:** SUSPECT.

#### `mkt_kelly_fraction`
- **Same structure** as `pos_kelly_fraction`. Cohort cap at 0.5. **Trustworthiness:** MODERATE.

#### `negrisk_volume_share`
- **Type:** DOUBLE
- **Definition:** fraction of the trader's `gross_usd_volume` that occurred on NegRisk markets, in [0, 1].
- **Formula:** [`build_traders_table.py:258-260`](../scripts/build_traders_table.py): `negrisk_volume / total_mkt_volume`
- **Trustworthiness:** STRONG.

#### `mkt_max_drawdown_usd`
- **Type:** DOUBLE
- **Definition:** maximum peak-to-trough drop in cumulative market-level PnL, ordered by `market_resolution_ts`.
- **Formula:** [`build_traders_table.py:322-352`](../scripts/build_traders_table.py)

  ```sql
  WITH per_market AS (
      SELECT address, market_id, sum(realised_pnl) AS market_pnl,
             any_value(resolution_ts) AS market_resolution_ts
      FROM closed_positions GROUP BY address, market_id
  ),
  ordered AS (
      SELECT address, market_resolution_ts, market_pnl,
             SUM(market_pnl) OVER (PARTITION BY address ORDER BY market_resolution_ts
                                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_pnl
      FROM per_market
  ),
  with_peak AS (
      SELECT address, cum_pnl,
             MAX(cum_pnl) OVER (PARTITION BY address ORDER BY market_resolution_ts
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_peak
      FROM ordered
  )
  SELECT address, MAX(running_peak - cum_pnl) AS mkt_max_drawdown_usd
  FROM with_peak GROUP BY address
  ```

- **Edge cases:** drawdown can exceed `abs(mkt_total_pnl)` legitimately (a "comeback" trader who hit a deep mid-period drawdown then recovered). Phase 3 sanity (c) found 196 270 / 2 576 698 (7.6 %) such cases — physically valid.
- **Trustworthiness:** STRONG.

### 2.4 Phantom score

#### `phantom_position_score`
- **Type:** DOUBLE
- **Definition:** volume-weighted average of `Σ|per-outcome PnL| / |Σ per-outcome PnL|` over the trader's markets. **1.0 = clean directional, ≫ 1.0 = NegRisk arb / split-merge behaviour.**
- **Formula:** [`build_traders_table.py:269-295`](../scripts/build_traders_table.py)

  ```sql
  WITH per_market AS (
      SELECT address, market_id,
             sum(realised_pnl) AS market_pnl,
             sum(abs(realised_pnl)) AS market_abs_pnl,
             sum(gross_usd_volume) AS market_volume
      FROM closed_positions GROUP BY address, market_id
  ),
  scored AS (
      SELECT *,
             CASE WHEN abs(market_pnl) > 0.01
                  THEN LEAST(market_abs_pnl / abs(market_pnl), 1000.0)
                  ELSE NULL END AS phantom_ratio
      FROM per_market
  )
  SELECT address,
      sum(... phantom_ratio * market_volume ...) / sum(... market_volume ...) AS phantom_position_score
  FROM scored GROUP BY address
  ```

- **Edge cases:** per-market ratio capped at 1000 to bound near-zero-`market_pnl` blow-ups. Markets with `|market_pnl| ≤ $0.01` are excluded from the average (the ratio is undefined and not informative). Trader-level NULL when none of their markets contribute.
- **Trustworthiness:** STRONG as a relative diagnostic. Domah's score is 8.45 (matches his known NegRisk arb behaviour). Operator addresses can have very high phantoms because they're on every fill.

### 2.5 Style profile

#### `style_maker_fill_count` / `style_taker_fill_count`
- **Type:** BIGINT
- **Definition:** count of fills the trader was on the maker / taker side, across **all of `joined_fills` (closed + open markets)**.
- **Formula:** [`build_traders_table.py:362-374`](../scripts/build_traders_table.py): `sum(CASE WHEN s.role = 'maker'/'taker' THEN 1 ELSE 0 END)` over `joined_fills CROSS JOIN (VALUES ('maker'),('taker')) AS s(role)`.
- **Edge cases:** the CROSS JOIN means each fill is counted once per side. Pre-filtered by `joined_fills` (no NULL/self/orphan).
- **Trustworthiness:** STRONG.

#### `style_maker_taker_ratio`
- **Type:** DOUBLE
- **Definition:** maker fills / taker fills, **capped at 1000.0**. Returns 1000.0 if taker count = 0 (pure maker).
- **Formula:** [`build_traders_table.py:488-491`](../scripts/build_traders_table.py): `LEAST(style_maker_fill_count / style_taker_fill_count, 1000.0)`
- **Caveat (2026-06-10):** the maker count includes `_matchOrders` exchange-internal active legs, where the address in the `maker` column was actually the AGGRESSOR (see `active_order_leg`, B.4). The ratio therefore overstates maker-ness; for style claims, reclassify internal-leg maker fills to the taker side (Domah 7.89 → 5.67; one audited leader flips maker_heavy → mixed). PnL columns are unaffected. See [[copytrade_attribution_repartition_findings]].
- **Trustworthiness:** STRONG mechanically; MODERATE as a passive-maker style indicator (see caveat).

#### `style_role_balance`
- **Type:** DOUBLE
- **Definition:** maker fills / total fills, in [0, 1]. **1.0 = pure maker, 0.0 = pure taker.** More interpretable than raw ratio.
- **Formula:** [`build_traders_table.py:492-495`](../scripts/build_traders_table.py)
- **Caveat (2026-06-10):** same maker-count bias as `style_maker_taker_ratio` — the numerator includes `_matchOrders` exchange-internal active legs (aggressive orders labelled maker; see `active_order_leg`, B.4), so it overstates maker share. Note the `patient_accumulators` cohort screen gates on `style_role_balance > 0.7`. See [[copytrade_attribution_repartition_findings]].
- **Trustworthiness:** STRONG mechanically; MODERATE as a passive-maker style indicator (see caveat).

#### `style_avg_fill_size_usd` / `style_max_fill_size_usd`
- **Type:** DOUBLE
- **Definition:** mean / max `usd_amount` per fill across all of the trader's `joined_fills` rows.
- **Formula:** [`build_traders_table.py:368-369`](../scripts/build_traders_table.py): `avg(jf.usd_amount)`, `max(jf.usd_amount)`.
- **Trustworthiness:** STRONG.

#### `style_median_fill_size_usd`
- **Type:** DOUBLE
- **Definition:** **always NULL in v1.**
- **Formula:** [`build_traders_table.py:370`](../scripts/build_traders_table.py): `CAST(NULL AS DOUBLE) AS style_median_fill_size_usd`
- **Caveat:** the original implementation used `approx_quantile(usd_amount, 0.5)` over the CROSS JOIN'd 2 B-row stream; the process exited silently at the start of step 6a (no traceback, no spill). Best guess: `approx_quantile`'s per-group t-digest state interacted poorly with the 2.6 M-group hash table. Deferred — can be added as a separate sampled pass if needed.
- **Trustworthiness:** N/A (always NULL).

#### `style_buy_sell_symmetry`
- **Type:** DOUBLE
- **Definition:** per-market `|buy_usd − sell_usd| / (buy_usd + sell_usd)`, averaged over the trader's markets. **0 = perfectly symmetric (MM-shaped), 1 = fully directional.**
- **Formula:** [`build_traders_table.py:381-402`](../scripts/build_traders_table.py)

  ```sql
  WITH per_market AS (
      SELECT
          CASE s.role WHEN 'maker' THEN jf.maker ELSE jf.taker END AS address,
          jf.market_id,
          sum(CASE WHEN (s.role = 'maker' AND jf.maker_asset_id = '0')
                    OR (s.role = 'taker' AND jf.maker_asset_id <> '0')
                   THEN jf.usd_amount ELSE 0 END) AS buy_usd,
          sum(CASE WHEN (s.role = 'maker' AND jf.maker_asset_id <> '0')
                    OR (s.role = 'taker' AND jf.maker_asset_id = '0')
                   THEN jf.usd_amount ELSE 0 END) AS sell_usd
      FROM joined_fills jf
      CROSS JOIN (VALUES ('maker'), ('taker')) AS s(role)
      GROUP BY 1, 2
  )
  SELECT address,
      avg(CASE WHEN buy_usd + sell_usd > 0
               THEN abs(buy_usd - sell_usd) / (buy_usd + sell_usd)
               ELSE NULL END) AS style_buy_sell_symmetry
  FROM per_market GROUP BY address
  ```

- **Edge cases:** markets where both sides are 0 contribute NULL (excluded from the average).
- **Trustworthiness:** STRONG.

#### `style_pct_sub_second`
- **Type:** DOUBLE
- **Definition:** percentage of the trader's fills that share a 1-second-truncated timestamp bucket with another fill from the same address. Approximation of "% fills within 1s of another fill from same address".
- **Formula:** [`build_traders_table.py:407-423`](../scripts/build_traders_table.py)

  ```sql
  WITH per_addr_sec AS (
      SELECT
          CASE s.role WHEN 'maker' THEN jf.maker ELSE jf.taker END AS address,
          date_trunc('second', jf.timestamp) AS sec,
          count(*) AS n_in_sec
      FROM joined_fills jf
      CROSS JOIN (VALUES ('maker'), ('taker')) AS s(role)
      GROUP BY 1, 2
  )
  SELECT address,
      100.0 * sum(CASE WHEN n_in_sec > 1 THEN n_in_sec ELSE 0 END)::DOUBLE
             / nullif(sum(n_in_sec), 0) AS style_pct_sub_second
  FROM per_addr_sec GROUP BY address
  ```

- **Caveat:** the bucketing approach is slightly conservative — two fills 0.999 s and 1.001 s apart in different second-buckets aren't counted as "sub-second" even though their gap is < 1 s. A true window function would be more precise but ~10× more expensive. This approximation is sufficient for operator detection (the use case).
- **Trustworthiness:** STRONG as a relative diagnostic, MODERATE for absolute claims about "exact within-1s rate".

#### `style_avg_holding_hours` / `style_median_holding_hours`
- **Type:** DOUBLE
- **Definition:** mean / median `holding_duration_hours` across the trader's positions, **filtered to `holding_duration_hours >= 0`** (excludes the 8 % of positions with placeholder pre-fill `end_date`).
- **Formula:** [`build_traders_table.py:122-127`](../scripts/build_traders_table.py)
- **Trustworthiness:** STRONG. Median is robust to the long-tail of held-to-resolution positions.

### 2.6 Counterparty graph

#### `n_distinct_counterparties`
- **Type:** BIGINT
- **Definition:** **HyperLogLog approximation** (`approx_count_distinct`) of distinct addresses on the other side of this trader's fills.
- **Formula:** [`build_traders_table.py:425-437`](../scripts/build_traders_table.py)

  ```sql
  SELECT
      CASE s.role WHEN 'maker' THEN jf.maker ELSE jf.taker END AS address,
      approx_count_distinct(
          CASE s.role WHEN 'maker' THEN jf.taker ELSE jf.maker END
      ) AS n_distinct_counterparties
  FROM joined_fills jf
  CROSS JOIN (VALUES ('maker'), ('taker')) AS s(role)
  GROUP BY 1
  ```

- **Caveat:** **HLL has typical ~2 % relative error**. Exact `COUNT(DISTINCT)` over 1 B fills × 2 sides was too heavy. For operator-detection thresholds (e.g., `> 500 000` triggers `is_operator_like`), the error is far below the threshold gap.
- **Trustworthiness:** MODERATE for exact value, STRONG for threshold comparison.

### 2.7 Drawdown

#### `pos_max_drawdown_usd`
- **Type:** DOUBLE
- **Definition:** maximum peak-to-trough drop in cumulative PnL across the trader's per-position PnL series, ordered by `resolution_ts`.
- **Formula:** [`build_traders_table.py:298-320`](../scripts/build_traders_table.py)

  ```sql
  WITH ordered AS (
      SELECT address, resolution_ts, realised_pnl,
             SUM(realised_pnl) OVER (PARTITION BY address ORDER BY resolution_ts
                                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_pnl
      FROM closed_positions
  ),
  with_peak AS (
      SELECT address, cum_pnl,
             MAX(cum_pnl) OVER (PARTITION BY address ORDER BY resolution_ts
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_peak
      FROM ordered
  )
  SELECT address, MAX(running_peak - cum_pnl) AS pos_max_drawdown_usd
  FROM with_peak GROUP BY address
  ```

- **Edge cases:** see `mkt_max_drawdown_usd`. 7.6 % of traders have `pos_max_drawdown_usd > abs(pos_total_pnl)` — physically valid recovery profile.
- **Trustworthiness:** STRONG.

### 2.8 Capital footprint

#### `rolling_bankroll_usd_30d` — **canonical, backtest-safe**
- **Type:** DOUBLE
- **Definition:** per-trader summary stat = `MAX over lifetime of bankroll_30d_prior(T)`, where `bankroll_30d_prior(T) = MAX over [T-30 days, T] of concurrent deployed capital`. Sourced from `data/bankroll_timeseries.parquet` (see §4 below for the canonical per-day series).
- **Formula:** [`build_bankroll_timeseries.py:170-196`](../scripts/build_bankroll_timeseries.py)

  ```sql
  CREATE OR REPLACE TABLE per_trader_bankroll AS
  SELECT address, MAX(bankroll_30d_prior) AS rolling_bankroll_usd_30d
  FROM read_parquet('data/bankroll_timeseries.parquet')
  GROUP BY address
  ```

- **Edge cases:** NULL for traders with no positions where `total_bought_usd > 0 AND resolution_ts >= first_fill_ts` (97.38 % coverage; missing 64 627 traders are pure-sell-only, e.g. minted-and-sold from splits).
- **For point-in-time queries** (e.g., "what was domah's bankroll on 2025-06-01?"), read from `data/bankroll_timeseries.parquet` directly. `traders.parquet` carries only the per-trader max.
- **Trustworthiness:** STRONG. Future-leakage check in `scripts/sanity_bankroll.py` passed 20/20 random samples (each `bankroll_30d_prior(T) ≤ cumulative-bought-by-T`, no time travel).

#### `est_bankroll_lifetime_peak_deprecated` — **deprecated, do not use for new work**
- **Type:** DOUBLE
- **Definition:** the original lifetime-peak-of-deployed-capital from the Phase 3 build. Renamed from `est_bankroll_usd_30d_max_approx` when `rolling_bankroll_usd_30d` was added.
- **Formula:** [`build_traders_table.py:451-476`](../scripts/build_traders_table.py) (unchanged — value passed through from the prior parquet).
- **Why deprecated:** the original computation had an **inconsistent filter**: it added `+total_bought_usd` events for every position with `total_bought_usd > 0`, but only subtracted `-total_bought_usd` events for positions with `resolution_ts >= first_fill_ts`. Positions with placeholder pre-fill end_dates (~8 % of positions) contributed `+amount` events without compensating `-amount`, monotonically inflating deployed_capital and overestimating the lifetime peak. Large-sample traders accumulated many such phantom legs.
- **Impact magnitude:** for the 5 Phase-4 top candidates, the new `rolling_bankroll_usd_30d` is:
  - `0x6a72f618…`: **18.27 %** of old (largest correction; high-volume large-sample trader)
  - `0xd38b71f3…`: 118.45 % of old (small upward correction — minor same-timestamp ordering effect)
  - `0x17db3fcd…`: 100.00 % (essentially unchanged)
  - `0xee00ba33…`: 19.50 % of old (another 80 % correction; 11 k-position large-sample trader)
  - `0x629bc4a1…`: 97.80 % (essentially unchanged)
  - `0x9d84ce…` (domah): 98.53 % of old ($58.2 M vs $59.1 M)
- **Why kept around:** downstream code in earlier sessions may reference the old name. Retained for backwards compatibility; do not use in new analysis.
- **Trustworthiness:** **SUSPECT** for large-sample traders due to the inconsistent-filter bug. Use `rolling_bankroll_usd_30d` instead.

### 2.9 Flags

#### `is_operator_like`
- **Type:** BOOLEAN
- **Definition:** TRUE iff the address matches our hardcoded operator deny-list OR triggers any of the operator-shape heuristics.
- **Formula:** [`build_traders_table.py:501-516`](../scripts/build_traders_table.py)

  ```sql
  (
      mp.address IN ({operators_sql})
      OR (style_taker_fill_count > 0 AND maker/taker > 50.0)
      OR (style_maker_fill_count > 0 AND style_taker_fill_count > 0 AND maker/taker < 0.02)
      OR coalesce(n_distinct_counterparties, 0) > 500000
      OR (coalesce(style_pct_sub_second, 0.0) > 95.0 AND n_fills_total > 1000000)
  ) AS is_operator_like
  ```

  Where the deny-list is the 12 addresses in [`data_infra/operator_denylist.py`](../data_infra/operator_denylist.py): 2 pure relayers, 7 pure MM bots, 3 HFT cluster.
- **Edge cases:** flags both deny-listed and heuristic-matched addresses. 4 033 of 2 576 698 = 0.16 % flagged.
- **Trustworthiness:** STRONG. The deny-list is hand-curated; the heuristic is operator-shaped by construction. False positives at the extreme tails of legitimate HFT/MM-style traders are possible — Phase 3 manual inspection of the cohort top-20 found no obvious miss.

---

## 4. `data/bankroll_timeseries.parquet` columns

Per-trader per-day rolling 30-day-prior bankroll. **The canonical source for point-in-time bankroll queries.** `traders.parquet.rolling_bankroll_usd_30d` is just the per-trader max of this series.

Built by [`scripts/build_bankroll_timeseries.py`](../scripts/build_bankroll_timeseries.py). 429 250 741 rows, 2 509 096 distinct addresses, dates 2022-11-21 → 2028-01-02 (tail extended beyond the data tail by far-future placeholder end_dates in the Gamma snapshot — filter `WHERE date <= '2026-04-24'` for analyses anchored to the actual data window).

Sorted by `(date, address)` for date-filtered point lookups.

#### `address`
- **Type:** VARCHAR
- **Definition:** lowercased 0x-prefixed address. Same convention as everywhere else.
- **Source:** [`build_bankroll_timeseries.py:75`](../scripts/build_bankroll_timeseries.py): passthrough from `closed_positions.address`.
- **Trustworthiness:** STRONG.

#### `date`
- **Type:** DATE
- **Definition:** the day the bankroll is evaluated at. One row per active day per trader.
- **Source:** [`build_bankroll_timeseries.py:131-143`](../scripts/build_bankroll_timeseries.py): generated via `generate_series(first_day, last_day, INTERVAL '1 day')` over the trader's first event date through their last event date. ASOF LEFT JOIN forward-fills deployed values from the cumulative-event table.
- **Edge cases:** rows where `bankroll_30d_prior = 0` are dropped before write (skip-if-no-activity rule from spec). So `date` rows densely cover every day the trader had non-zero deployed capital in the prior 30 days.
- **Trustworthiness:** STRONG.

#### `bankroll_30d_prior`
- **Type:** DOUBLE
- **Definition:** maximum, over the 30-day window `[date − 30 days, date]`, of the trader's concurrent deployed capital (sum of `total_bought_usd` across positions open at any point in the window).
- **Source:** [`build_bankroll_timeseries.py:155-167`](../scripts/build_bankroll_timeseries.py)

  ```sql
  -- Stage 1: spans
  CREATE TABLE spans AS
  SELECT address,
         date_trunc('day', first_fill_ts)::DATE AS start_date,
         date_trunc('day', resolution_ts)::DATE AS end_date,
         total_bought_usd AS amount
  FROM read_parquet('data/closed_positions.parquet')
  WHERE total_bought_usd > 0
    AND resolution_ts IS NOT NULL
    AND resolution_ts >= first_fill_ts;

  -- Stage 2: events  (+amount on start, -amount on end+1)
  -- Stage 3: cumsum  → deployed_at_each_event_date
  -- Stage 4: ASOF JOIN per-trader day series → daily_deployed
  -- Stage 5: rolling MAX window:
  MAX(deployed) OVER (
      PARTITION BY address ORDER BY date
      RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
  ) AS bankroll_30d_prior
  ```

- **Edge cases:**
  - Positions with `total_bought_usd ≤ 0` excluded (pure-sell positions from splits — they don't represent capital deployment in our framing).
  - Positions with `resolution_ts < first_fill_ts` excluded (placeholder pre-fill Gamma end_dates).
  - Rows with `bankroll_30d_prior = 0` dropped before write (per spec).
  - Span granularity is **DAY**, not timestamp — same-day open + close on a position contributes `+amount` to that day's deployed (deployed = amount on day X, drops to 0 on X+1).
- **Backtest-safe by construction:** `bankroll_30d_prior(T)` uses only positions with `start_date ≤ T` (positions that had already opened by time T). The rolling-MAX window is strictly retrospective. The future-leakage sanity check in [`scripts/sanity_bankroll.py`](../scripts/sanity_bankroll.py) verifies `bankroll_30d_prior(T) ≤ Σ total_bought_usd for positions with first_fill_ts ≤ T` for 20 random samples — passed cleanly.
- **Trustworthiness:** STRONG for backtest sizing. The DAY granularity (vs TIMESTAMP) means intra-day open-and-close spikes that cancel within the same day are missed; for prediction-market trading at human timescales this is fine, and it actually matches the resolution at which a copy-trader could act on a leader's signal anyway.

---

## 3. `data/cohorts/*.parquet` columns

Each cohort parquet (`high_sharpe_directional`, `high_profit_factor_with_size`, `negrisk_specialists`, `sports_directional_fast`, `patient_accumulators`, `high_kelly_edge`) is a `WHERE`-filtered slice of `traders_aug` — which is `traders_filtered` + two extra computed columns. All `traders.parquet` columns above pass through verbatim. The two **new** columns are:

#### `pos_std_pnl`
- **Type:** DOUBLE
- **Definition:** population standard deviation of per-position `realised_pnl` for the trader.
- **Source:** [`scripts/build_cohorts.py:206-209`](../scripts/build_cohorts.py)

  ```sql
  CREATE OR REPLACE TABLE m_std_aux AS
  WITH pos_lvl AS (
      SELECT address, stddev_pop(realised_pnl) AS pos_std_pnl
      FROM read_parquet('data/closed_positions.parquet')
      GROUP BY address
  )
  ```

- **Inputs:** `closed_positions.realised_pnl`
- **Why it lives only in cohort parquets and not `traders.parquet`:** historical accident — Phase 3 didn't expose this column, and adding it to `traders.parquet` requires a full re-build of `m_pos`. Phase 4 instead computes it on the fly from `closed_positions` once and joins into `traders_filtered` to form `traders_aug`, with a runtime cost of ~44 s.
- **Edge cases:** can be 0 for traders with all-equal PnLs (extremely rare; would block Sharpe). Cohort guard `pos_std_pnl > 1.0` excludes these.
- **Trustworthiness:** STRONG.

#### `mkt_std_pnl`
- **Type:** DOUBLE
- **Definition:** population standard deviation of per-market PnL (after the position→market collapse).
- **Source:** [`scripts/build_cohorts.py:211-218`](../scripts/build_cohorts.py)

  ```sql
  mkt_per_market AS (
      SELECT address, market_id, sum(realised_pnl) AS market_pnl
      FROM read_parquet('data/closed_positions.parquet')
      GROUP BY address, market_id
  ),
  mkt_lvl AS (
      SELECT address, stddev_pop(market_pnl) AS mkt_std_pnl
      FROM mkt_per_market GROUP BY address
  )
  ```

- **Trustworthiness:** STRONG. Used as the artifact-blowup guard for Sharpe-based cohorts (`mkt_std_pnl > 1.0`).

### Cohort-specific WHERE clauses

Documented at [`scripts/build_cohorts.py:33-97`](../scripts/build_cohorts.py). Reproduced for traceability:

| Pool | Filter |
|---|---|
| `high_sharpe_directional` | `mkt_sharpe > 1.5 AND mkt_sharpe < 10 AND mkt_std_pnl > 1.0 AND n_closed_positions > 200 AND active_days > 90 AND negrisk_volume_share < 0.5 AND NOT is_operator_like` |
| `high_profit_factor_with_size` | `mkt_profit_factor > 2.0 AND mkt_profit_factor < 100 AND mkt_total_pnl > 50000 AND n_closed_positions > 100 AND active_days > 90 AND NOT is_operator_like` |
| `negrisk_specialists` | `negrisk_volume_share > 0.7 AND mkt_total_pnl > 50000 AND n_closed_positions > 200 AND active_days > 90 AND mkt_profit_factor > 1.3 AND NOT is_operator_like` |
| `sports_directional_fast` | `negrisk_volume_share < 0.3 AND mkt_sharpe > 1.0 AND mkt_sharpe < 10 AND mkt_std_pnl > 1.0 AND style_avg_holding_hours < 48 AND n_closed_positions > 200 AND active_days > 90 AND NOT is_operator_like` |
| `patient_accumulators` | `style_role_balance > 0.7 AND style_avg_holding_hours > 168 AND mkt_total_pnl > 100000 AND n_closed_positions > 100 AND active_days > 180 AND NOT is_operator_like` |
| `high_kelly_edge` | `mkt_kelly_fraction > 0.05 AND mkt_kelly_fraction < 0.5 AND n_closed_positions > 200 AND active_days > 90 AND mkt_dollar_win_rate > 0.55 AND NOT is_operator_like` |

---

## Known limitations (consolidated)

Read-once list, in order of how likely they are to bite you.

1. **`pos_sharpe` and `mkt_sharpe` annualisation is poison at the tail.** Naive `sqrt(N / years_active)` blows up when `active_days` is small (a 30-day fluke gets multiplied by `sqrt(12)`), when `n_closed_positions` is small, or when `pos_std_pnl` is near zero (NegRisk arb traders with matched-pair PnLs that net cleanly). p99 of `pos_sharpe` across `traders_filtered` (n_pos > 50) is **14.90** and max is **1.66×10¹⁵**. **Treat Sharpe as DIAGNOSTIC, never primary ranker.** All Sharpe-based cohort filters require `n_closed_positions > 200 AND active_days > 90 AND mkt_std_pnl > 1.0` simultaneously.

2. **NegRisk merge/split events are not modelled.** A trader who mints USDC into a YES+NO pair, sells one side, and merges back is invisible to `OrderFilled`. Their per-position `realised_pnl` is correct in aggregate (`pos_total_pnl == mkt_total_pnl`) but **inflated on the winning outcome and equally deflated on the losing outcome**. Use `mkt_*` family for ranking arb-heavy traders. Use `phantom_position_score` to detect them: 1.0 = clean, ≫ 1.0 = arb-shaped.

3. **`est_bankroll_lifetime_peak_deprecated` is DEPRECATED.** Use `rolling_bankroll_usd_30d` (from `traders.parquet`) for per-trader bankroll summary, or `data/bankroll_timeseries.parquet` for point-in-time queries. The deprecated column is the lifetime-peak-of-deployed-capital from the Phase 3 build; it had an inconsistent filter that inflated values for traders with placeholder-end-date positions (some by 80%+). The replacement column is backtest-safe by construction. The deprecated column is kept for backwards compatibility with older code; do not use it in new analysis.

4. **`peak_fill_abs_token` is a lower bound on true peak position size.** It's the largest single-fill `|token_delta|`, not the running cumulative max. The true peak would have required sorting a 1.4 B-row collapsed table and OOM'd at 300 GB temp.

5. **`style_median_fill_size_usd` ships as NULL in v1.** `approx_quantile` over the CROSS JOIN'd 2 B-row stream silently exited the build. Add as a separate sampled pass if needed.

6. **`n_distinct_counterparties` is HLL-approximate (~2 % error).** Fine for threshold comparisons (e.g., `> 500 000` for operator detection); don't read as exact.

7. **`holding_duration_hours < 0` for ~8 % of positions.** Caused by Gamma's `end_date` being set to a placeholder before the market trades. Filter `holding_duration_hours >= 0` for any duration aggregation. Also flows into `style_avg_holding_hours` (already filtered there) but **not** into `pos_max_drawdown_usd` (where `resolution_ts` is the sort key — the affected positions sort to the start, biasing drawdown low for the affected traders).

8. **External reconciliation is order-of-magnitude only.** Polymarket's public API doesn't expose lifetime PnL — only currently-open mark-to-market and partial-realized within open positions. The Polymarket UI shows lifetime P&L on profile pages but isn't scriptable. **Treat differences <20% as internally normal.**

9. **`raw_trades` lags real-time by ~9 days.** Goldsky's subgraph indexer trails. For walk-forward backtests, set the test cutoff to ≥ 9 days behind the test run date.

10. **Restart loss in `raw_trades`.** Each kill of the historical sync drops up to ~500 k rows from each actively-writing shard's in-flight parquet (the file footer wasn't flushed; cursor had advanced past those rows). Visible as small gaps in the resulting parquet, ~0.5–1 % of total. Affects per-day fill counts but not per-trader aggregates materially.

11. **905 self-trades dropped at view level.** All from 2022-11 → 2023-04 (early-protocol noise). `joined_fills` filters `maker <> taker`. Negligible; documented for completeness.

12. **Orphan fills (~0.4 % of `raw_trades`) excluded from `joined_fills`.** Routed to `trader_actions_orphan` for audit but never enter position math. Their `realised_pnl` is unknowable (no market_id, no resolution).

13. **The cohort overlap maxes at 4 pools.** No address qualifies for 5 or 6 of the 6 pools — by design (the pools span deliberately-orthogonal dimensions). The 947 addresses qualifying for 3+ pools are the most cohort-robust candidates.
