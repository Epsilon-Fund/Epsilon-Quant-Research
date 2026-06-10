-- Phase 2 / Layer-B view definitions.
-- Loaded into a DuckDB connection by data_infra.views.load_views(con).
-- Placeholders {TRADES_GLOB}, {SEED_PATH}, {MARKETS_PATH} are substituted
-- at load time via str.format().
--
-- Conventions:
--   - All addresses lowercase (already so in source parquets).
--   - Sign convention in trader_actions is from the address's POV:
--       token_delta > 0 → received outcome tokens
--       token_delta < 0 → gave up outcome tokens
--       usd_delta > 0   → received USDC
--       usd_delta < 0   → paid USDC
--   - outcome_index is the 1-based position in markets.clob_token_ids.
--   - Filters at view level (trader_actions only):
--       maker != taker        (drops 905 self-trades, 2022-11 → 2023-04)
--       market_id IS NOT NULL (orphans go to trader_actions_orphan instead)
--       outcome_index resolved (drops fills whose token_id isn't in the snapshot)
--   - active_order_leg (BOOLEAN, trader_actions) — "this row's address signed
--     the ACTIVE (aggressor) order of the fill". CTF Exchange `_matchOrders`
--     emits the internal active leg of a two-sided match with
--     maker = takerOrder.maker and taker = address(this), so when the fill's
--     `taker` is one of the 4 exchange-internal-leg contracts the wallet in the
--     `maker` column was actually the aggressor. Semantics per role:
--       maker-role row: TRUE iff the fill's taker is an exchange-internal-leg
--         contract (active order signer labelled as maker); FALSE = genuine
--         passive maker.
--       taker-role row: TRUE for normal fills (taker is the aggressor by
--         definition); FALSE iff the row's address IS the exchange contract
--         itself — an artifact leg, not a trader; exclude those rows from
--         per-trader style metrics.
--     PnL/position attribution is unaffected by this flag — style framing only.
--     joined_fills carries the fill-level precursor `exchange_internal_match`
--     (TRUE iff taker is an exchange-internal-leg contract).
--     The 4 addresses are hardcoded below; SOURCE OF TRUTH is
--     data_infra/operator_denylist.py (EXCHANGE_INTERNAL_LEG) — keep in sync.
--
-- Performance design:
--   The view performs the JOIN to markets_tokens ONCE on raw_trades, then
--   UNIONs the maker- and taker-perspective rows. This keeps the JOIN's hash
--   side small (~2 M markets) and avoids materialising the 2 B-row exploded
--   intermediate before joining. Downstream scoped queries (filter by address,
--   filter by closed markets, etc.) push down through the union safely.


-- (a) raw_trades — direct view over all parquet, no filtering.
CREATE OR REPLACE VIEW raw_trades AS
SELECT * FROM read_parquet('{TRADES_GLOB}')
UNION ALL BY NAME
SELECT * FROM read_parquet('{SEED_PATH}');


-- markets_tokens — one row per (market_id, outcome_index, outcome_token_id).
-- Materialised as a TABLE (not a view) so the JOIN in joined_fills uses
-- a hot hash table instead of recomputing the unnest every query.
-- Built once at load_views() time; ~2 M rows.
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


-- joined_fills — internal helper: raw_trades joined to markets_tokens once,
-- with outcome_token_id and outcome_index pre-computed. Each row is one
-- on-chain fill. Trader_actions explodes this 1:2 (maker + taker).
--
-- The CASE expression that picks the outcome token is materialised in a
-- pre-projection subquery so the JOIN key is a plain column reference (DuckDB
-- can hash-join on it cleanly without re-evaluating the case per probe row).
CREATE OR REPLACE VIEW joined_fills AS
SELECT
    pre.timestamp,
    pre.market_id,
    pre.condition_id,
    pre.neg_risk,
    pre.maker,
    pre.taker,
    pre.maker_asset_id,
    pre.token_amount,
    pre.usd_amount,
    pre.price,
    pre.transaction_hash,
    pre.outcome_token_id,
    mt.outcome_index,
    -- TRUE iff this fill row is the `_matchOrders` exchange-internal active leg
    -- (the wallet in `maker` was the aggressor / active order signer).
    -- Hardcoded; keep in sync with data_infra/operator_denylist.py
    -- EXCHANGE_INTERNAL_LEG (source of truth).
    pre.taker IN (
        '0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e',  -- CTF Exchange v1 standard
        '0xc5d563a36ae78145c45a50134d48a1215220f80a',  -- CTF Exchange v1 neg-risk
        '0xe111180000d2663c0091e4f400237545b87b996b',  -- CTF Exchange v2 standard (post 2026-04-28)
        '0xe2222d279d744050d28e00520010520000310f59'   -- CTF Exchange v2 neg-risk (post 2026-04-28)
    ) AS exchange_internal_match
FROM (
    SELECT
        rt.timestamp,
        rt.market_id,
        rt.condition_id,
        rt.neg_risk,
        rt.maker,
        rt.taker,
        rt.maker_asset_id,
        rt.token_amount,
        rt.usd_amount,
        rt.price,
        rt.transaction_hash,
        CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
            AS outcome_token_id
    FROM raw_trades rt
    WHERE rt.maker IS NOT NULL
      AND rt.taker IS NOT NULL
      AND rt.maker <> rt.taker
      AND rt.market_id IS NOT NULL
) pre
JOIN markets_tokens mt
    ON mt.market_id = pre.market_id
   AND mt.outcome_token_id = pre.outcome_token_id;


-- (b) trader_actions — exploded per-address fills.
-- Each fill becomes 2 rows: one for the maker, one for the taker, with
-- correctly-signed token_delta and usd_delta from each address's POV.
CREATE OR REPLACE VIEW trader_actions AS
SELECT
    timestamp,
    'maker' AS role,
    maker AS address,
    market_id,
    condition_id,
    neg_risk,
    outcome_token_id,
    outcome_index,
    -- maker BOUGHT iff maker_asset_id = '0' (paid USDC, received outcome tokens)
    CASE WHEN maker_asset_id = '0' THEN  token_amount ELSE -token_amount END AS token_delta,
    CASE WHEN maker_asset_id = '0' THEN -usd_amount   ELSE  usd_amount   END AS usd_delta,
    token_amount,
    usd_amount,
    price,
    transaction_hash,
    -- maker-role row was the active order signer iff this fill is the
    -- _matchOrders exchange-internal leg (see header).
    exchange_internal_match AS active_order_leg
FROM joined_fills
UNION ALL
SELECT
    timestamp,
    'taker' AS role,
    taker AS address,
    market_id,
    condition_id,
    neg_risk,
    outcome_token_id,
    outcome_index,
    CASE WHEN maker_asset_id = '0' THEN -token_amount ELSE  token_amount END AS token_delta,
    CASE WHEN maker_asset_id = '0' THEN  usd_amount   ELSE -usd_amount   END AS usd_delta,
    token_amount,
    usd_amount,
    price,
    transaction_hash,
    -- a normal taker-role row is the aggressor by definition; FALSE only when
    -- the address itself is the exchange contract (artifact leg — exclude from
    -- per-trader style metrics; see header).
    NOT exchange_internal_match AS active_order_leg
FROM joined_fills;


-- ============ TRADERS (Phase 3) ============
-- The two views below are loaded *conditionally* by data_infra/views.py:
-- they require data/traders.parquet to exist on disk. If you're running
-- before build_traders_table.py has produced it, the loader skips this
-- block.

CREATE OR REPLACE VIEW traders_raw AS
SELECT * FROM read_parquet('{TRADERS_PATH}');

CREATE OR REPLACE VIEW traders_filtered AS
SELECT * FROM traders_raw WHERE NOT is_operator_like;

-- ============ ORPHAN TRADER ACTIONS ============
-- (c) trader_actions_orphan — same explode shape, but for fills where
-- market_id IS NULL (no Gamma match). No outcome_index. Audit only.
CREATE OR REPLACE VIEW trader_actions_orphan AS
WITH base AS (
    SELECT
        timestamp,
        maker, taker,
        maker_asset_id,
        token_amount, usd_amount, price, transaction_hash,
        CASE WHEN maker_asset_id = '0' THEN taker_asset_id ELSE maker_asset_id END
            AS outcome_token_id
    FROM raw_trades
    WHERE maker IS NOT NULL
      AND taker IS NOT NULL
      AND maker <> taker
      AND market_id IS NULL
)
SELECT
    timestamp, 'maker' AS role, maker AS address, outcome_token_id,
    CASE WHEN maker_asset_id = '0' THEN  token_amount ELSE -token_amount END AS token_delta,
    CASE WHEN maker_asset_id = '0' THEN -usd_amount   ELSE  usd_amount   END AS usd_delta,
    token_amount, usd_amount, price, transaction_hash
FROM base
UNION ALL
SELECT
    timestamp, 'taker' AS role, taker AS address, outcome_token_id,
    CASE WHEN maker_asset_id = '0' THEN -token_amount ELSE  token_amount END AS token_delta,
    CASE WHEN maker_asset_id = '0' THEN  usd_amount   ELSE -usd_amount   END AS usd_delta,
    token_amount, usd_amount, price, transaction_hash
FROM base;
