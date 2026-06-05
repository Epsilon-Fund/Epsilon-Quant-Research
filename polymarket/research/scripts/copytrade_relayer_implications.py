"""Quantify copy-trading bias from v1 exchange-internal taker legs.

This is intentionally a magnitude-estimation script, not a rebuild of
closed_positions.parquet or traders.parquet.

Run from polymarket/research/:
    PYTHONPATH=. uv run python scripts/copytrade_relayer_implications.py
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import httpx
import numpy as np
import pandas as pd

from data_infra.views import latest_markets_path, load_views


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

TRADERS = DATA / "traders.parquet"
CLOSED_POSITIONS = DATA / "closed_positions.parquet"
DIRECTIONALITY = DATA / "directionality_classification" / "traders_directionality.parquet"

DB_PATH = ANALYSIS / "_copytrade_relayer_implications.duckdb"
SUMMARY_PATH = ANALYSIS / "copytrade_relayer_implications_summary.json"
SAMPLE_PATH = ANALYSIS / "copytrade_taker_recovery_sample.parquet"
TX_CACHE_PATH = ANALYSIS / "copytrade_tx_from_cache.parquet"
ESTIMATES_PATH = ANALYSIS / "copytrade_invisible_take_estimates.parquet"
PATCHED_POSITIONS_PATH = ANALYSIS / "copytrade_patched_positions_cohort.parquet"
COHORT_BIAS_PATH = ANALYSIS / "csv_outputs" / "copytrade" / "copytrade_cohort_bias_table.csv"
SYSTEMATICITY_PATH = ANALYSIS / "csv_outputs" / "copytrade" / "copytrade_bias_systematicity.csv"

DEFAULT_RPC_URL = "https://polygon.drpc.org"
SAMPLE_TARGET = 50_000
BOOTSTRAP_REPS = 200
RANDOM_SEED = 1729

V1_EXCHANGE = tuple(
    sorted(
        {
            "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",
            "0xc5d563a36ae78145c45a50134d48a1215220f80a",
        }
    )
)
V2_EXCHANGE = tuple(
    sorted(
        {
            "0xe111180000d2663c0091e4f400237545b87b996b",
            "0xe2222d279d744050d28e00520010520000310f59",
        }
    )
)

DOMAH = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344"
MANUAL_CANDIDATES = [
    "0xd38b71f3e8ed1af71983e5c309eac3dfa9b35029",
    "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee",
    "0x17db3fcd93ba12d38382a0cade24b200185c5f6d",
    "0xee00ba338c59557141789b127927a55f5cc5cea1",
    "0x629bc4a1e53e1d475beb7ea3d388791e96dd995a",
]

COHORT_SELECTION_COLUMNS = [
    "mkt_sharpe",
    "mkt_std_pnl",
    "n_closed_positions",
    "active_days",
    "negrisk_volume_share",
    "mkt_profit_factor",
    "mkt_total_pnl",
    "pct_markets_balanced_and_offsetting_vw",
    "style_avg_holding_hours",
    "style_role_balance",
    "mkt_kelly_fraction",
    "mkt_dollar_win_rate",
]
REQUESTED_CORRELATION_COLUMNS = [
    "pos_sharpe",
    "mkt_total_pnl",
    "pos_win_rate",
    "n_closed_positions",
    "style_maker_taker_ratio",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sql_list(values: tuple[str, ...] | list[str]) -> str:
    return ", ".join(f"'{v.lower()}'" for v in values)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA threads=8")
    con.execute("SET preserve_insertion_order=false")
    load_views(con)
    con.execute(
        f"CREATE OR REPLACE VIEW closed_positions AS "
        f"SELECT * FROM read_parquet('{CLOSED_POSITIONS}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW traders AS SELECT * FROM read_parquet('{TRADERS}')"
    )
    return con


def schema_gate(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    seed_schema = con.sql(
        "DESCRIBE SELECT * FROM read_parquet('data/trades/trades_seed.parquet') LIMIT 0"
    ).fetchdf()
    delta_files = sorted((DATA / "trades").glob("trades_delta_shard*.parquet"))
    if not delta_files:
        raise SystemExit("missing data/trades/trades_delta_shard*.parquet")
    delta_schema = con.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{delta_files[0]}') LIMIT 0"
    ).fetchdf()

    identity_candidates = {
        "transaction_hash",
        "block_number",
        "tx_from",
        "from",
        "tx_origin",
        "log_index",
    }
    seed_cols = set(seed_schema["column_name"].str.lower())
    delta_cols = set(delta_schema["column_name"].str.lower())
    identity = {
        "seed": sorted(seed_cols & identity_candidates),
        "delta": sorted(delta_cols & identity_candidates),
        "delta_file_used": str(delta_files[0].relative_to(ROOT)),
        "literal_shard0_present": (DATA / "trades" / "trades_delta_shard0.parquet").exists(),
    }
    log(f"schema identity fields: {identity}")
    return identity


def build_population_tables(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    v1 = sql_list(V1_EXCHANGE)
    v2 = sql_list(V2_EXCHANGE)

    log("building v1 month counts")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE copytrade_v1_month_counts AS
        SELECT
            date_trunc('month', timestamp) AS month,
            count(*)::DOUBLE AS month_n,
            sum(usd_amount)::DOUBLE AS month_usd,
            min(timestamp) AS first_ts,
            max(timestamp) AS last_ts
        FROM raw_trades
        WHERE taker IN ({v1})
        GROUP BY 1
        ORDER BY 1
        """
    )

    totals = con.sql(
        """
        SELECT
            sum(month_n)::DOUBLE AS n_v1_internal,
            sum(month_usd)::DOUBLE AS usd_v1_internal,
            min(first_ts) AS first_ts,
            max(last_ts) AS last_ts,
            count(*) AS n_months
        FROM copytrade_v1_month_counts
        """
    ).fetchdf().iloc[0].to_dict()

    log("building deterministic stratified 50k v1 recovery sample")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE copytrade_v1_month_quota AS
        WITH total AS (
            SELECT sum(month_n) AS total_n FROM copytrade_v1_month_counts
        ),
        raw_quota AS (
            SELECT
                month,
                month_n,
                month_usd,
                greatest(1, round(month_n / total_n * {SAMPLE_TARGET}))::BIGINT AS quota
            FROM copytrade_v1_month_counts, total
        ),
        adjusted AS (
            SELECT *,
                sum(quota) OVER () AS quota_sum,
                row_number() OVER (ORDER BY month_n DESC) AS size_rank
            FROM raw_quota
        )
        SELECT
            month,
            month_n,
            month_usd,
            greatest(
                1,
                quota
                + CASE
                    WHEN size_rank = 1 THEN {SAMPLE_TARGET} - quota_sum
                    ELSE 0
                  END
            )::BIGINT AS quota
        FROM adjusted
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE copytrade_recovery_sample_base AS
        WITH candidates AS (
            SELECT
                rt.transaction_hash,
                CAST(NULL AS BIGINT) AS log_index,
                rt.timestamp AS ts,
                rt.maker,
                rt.taker AS taker_recorded,
                rt.market_id,
                rt.condition_id,
                rt.neg_risk,
                rt.maker_asset_id,
                rt.taker_asset_id,
                CASE WHEN rt.maker_asset_id = '0'
                     THEN rt.taker_asset_id ELSE rt.maker_asset_id END AS outcome_token_id,
                rt.token_amount,
                rt.usd_amount,
                rt.price,
                date_trunc('month', rt.timestamp) AS month,
                mq.month_n,
                mq.month_usd,
                mq.quota,
                hash(
                    concat_ws(
                        '|',
                        rt.transaction_hash,
                        coalesce(rt.maker, ''),
                        coalesce(rt.market_id, ''),
                        coalesce(rt.maker_asset_id, ''),
                        coalesce(rt.taker_asset_id, ''),
                        CAST(rt.usd_amount AS VARCHAR),
                        CAST(rt.token_amount AS VARCHAR),
                        CAST(rt.timestamp AS VARCHAR)
                    )
                ) AS sample_hash
            FROM raw_trades rt
            JOIN copytrade_v1_month_quota mq
              ON mq.month = date_trunc('month', rt.timestamp)
            WHERE rt.taker IN ({v1})
              AND rt.transaction_hash IS NOT NULL
              AND rt.market_id IS NOT NULL
        ),
        oversampled AS (
            SELECT *,
                row_number() OVER (PARTITION BY month ORDER BY sample_hash) AS rn
            FROM candidates
            WHERE sample_hash::DOUBLE / 18446744073709551616.0
                  < least(1.0, 4.0 * quota::DOUBLE / month_n)
        )
        SELECT
            transaction_hash,
            log_index,
            ts,
            maker,
            taker_recorded,
            CAST(NULL AS VARCHAR) AS tx_from_recovered,
            CAST(NULL AS BIGINT) AS block_number,
            market_id,
            condition_id,
            neg_risk,
            maker_asset_id,
            taker_asset_id,
            outcome_token_id,
            token_amount,
            usd_amount,
            price,
            month,
            month_n,
            month_usd,
            quota,
            CAST(NULL AS DOUBLE) AS sample_weight,
            sample_hash
        FROM oversampled
        WHERE rn <= quota
        """
    )
    sample_n = con.sql("SELECT count(*) FROM copytrade_recovery_sample_base").fetchone()[0]
    log(f"sample rows before tx recovery: {sample_n:,}")

    log("building maker-side lower-bound counts")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE copytrade_v1_internal_by_maker AS
        SELECT
            maker AS address,
            count(*)::DOUBLE AS n_v1_internal_taker_legs_paired_with_addr_as_maker,
            sum(usd_amount)::DOUBLE AS v1_internal_taker_legs_paired_with_addr_as_maker_usd
        FROM raw_trades
        WHERE taker IN ({v1})
          AND maker IS NOT NULL
        GROUP BY maker
        """
    )

    log("verifying v1/v2 exchange-address emit pattern")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE copytrade_exchange_emit_summary AS
        SELECT
            label,
            address,
            sum(as_maker)::DOUBLE AS as_maker,
            sum(as_taker)::DOUBLE AS as_taker,
            sum(maker_usd)::DOUBLE AS maker_usd,
            sum(taker_usd)::DOUBLE AS taker_usd,
            min(first_seen) AS first_seen,
            max(last_seen) AS last_seen
        FROM (
            SELECT 'v1' AS label, maker AS address, count(*) AS as_maker, 0 AS as_taker,
                   sum(usd_amount) AS maker_usd, 0::DOUBLE AS taker_usd,
                   min(timestamp) AS first_seen, max(timestamp) AS last_seen
            FROM raw_trades WHERE maker IN ({v1}) GROUP BY maker
            UNION ALL
            SELECT 'v1' AS label, taker AS address, 0 AS as_maker, count(*) AS as_taker,
                   0::DOUBLE AS maker_usd, sum(usd_amount) AS taker_usd,
                   min(timestamp) AS first_seen, max(timestamp) AS last_seen
            FROM raw_trades WHERE taker IN ({v1}) GROUP BY taker
            UNION ALL
            SELECT 'v2' AS label, maker AS address, count(*) AS as_maker, 0 AS as_taker,
                   sum(usd_amount) AS maker_usd, 0::DOUBLE AS taker_usd,
                   min(timestamp) AS first_seen, max(timestamp) AS last_seen
            FROM raw_trades WHERE maker IN ({v2}) GROUP BY maker
            UNION ALL
            SELECT 'v2' AS label, taker AS address, 0 AS as_maker, count(*) AS as_taker,
                   0::DOUBLE AS maker_usd, sum(usd_amount) AS taker_usd,
                   min(timestamp) AS first_seen, max(timestamp) AS last_seen
            FROM raw_trades WHERE taker IN ({v2}) GROUP BY taker
        )
        GROUP BY label, address
        ORDER BY label, address
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE copytrade_v2_post_cutover_sample AS
        SELECT
            count(*)::DOUBLE AS sample_rows,
            sum(CASE WHEN maker IN ({v2}) THEN 1 ELSE 0 END)::DOUBLE AS v2_as_maker_rows,
            sum(CASE WHEN taker IN ({v2}) THEN 1 ELSE 0 END)::DOUBLE AS v2_as_taker_rows,
            sum(CASE WHEN maker IN ({v2}) THEN usd_amount ELSE 0 END)::DOUBLE AS v2_as_maker_usd,
            sum(CASE WHEN taker IN ({v2}) THEN usd_amount ELSE 0 END)::DOUBLE AS v2_as_taker_usd,
            min(timestamp) AS first_ts,
            max(timestamp) AS last_ts
        FROM (
            SELECT *
            FROM (
                SELECT *
                FROM raw_trades
                WHERE timestamp > TIMESTAMP '2026-04-28 11:00:40'
            ) filtered_post_cutover
            USING SAMPLE reservoir(100000 ROWS)
        )
        """
    )

    emit_summary = con.sql("SELECT * FROM copytrade_exchange_emit_summary").fetchdf()
    v2_sample = con.sql("SELECT * FROM copytrade_v2_post_cutover_sample").fetchdf().iloc[0].to_dict()
    return {
        "v1_internal_totals": totals,
        "sample_rows_before_recovery": int(sample_n),
        "exchange_emit_summary": emit_summary.to_dict(orient="records"),
        "v2_post_cutover_sample": v2_sample,
    }


def _rpc_batch(
    url: str,
    hashes: list[str],
    *,
    timeout_s: float,
    max_attempts: int,
) -> list[dict[str, Any]]:
    payload = [
        {"jsonrpc": "2.0", "id": i, "method": "eth_getTransactionByHash", "params": [h]}
        for i, h in enumerate(hashes)
    ]
    backoff = 1.0
    with httpx.Client(timeout=timeout_s) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict):
                    data = [data]
                by_id = {int(row.get("id", -1)): row for row in data}
                out: list[dict[str, Any]] = []
                for i, tx_hash in enumerate(hashes):
                    row = by_id.get(i, {})
                    result = row.get("result") or {}
                    out.append(
                        {
                            "transaction_hash": tx_hash,
                            "tx_from_recovered": (result.get("from") or "").lower() or None,
                            "tx_to": (result.get("to") or "").lower() or None,
                            "block_number": (
                                int(result["blockNumber"], 16)
                                if result.get("blockNumber")
                                else None
                            ),
                        }
                    )
                return out
            except Exception:
                if attempt == max_attempts:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 20.0)
    raise RuntimeError("unreachable")


def recover_tx_from(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    base = con.sql(
        "SELECT DISTINCT transaction_hash FROM copytrade_recovery_sample_base"
    ).fetchdf()
    hashes = sorted(base["transaction_hash"].dropna().unique().tolist())

    if TX_CACHE_PATH.exists():
        cached = con.sql(f"SELECT * FROM read_parquet('{TX_CACHE_PATH}')").fetchdf()
        cached = cached[cached["tx_from_recovered"].notna()].copy()
    else:
        cached = pd.DataFrame(
            columns=["transaction_hash", "tx_from_recovered", "tx_to", "block_number"]
        )
    done = set(cached["transaction_hash"].dropna().tolist())
    todo = [h for h in hashes if h not in done]
    log(f"recovering tx.from for {len(todo):,} uncached / {len(hashes):,} sample transactions")

    if todo:
        rpc_url = os.environ.get("POLYGON_RPC_URL", DEFAULT_RPC_URL)
        batch_size = int(os.environ.get("COPYTRADE_RPC_BATCH_SIZE", "100"))
        workers = int(os.environ.get("COPYTRADE_RPC_WORKERS", "1"))
        sleep_s = float(os.environ.get("COPYTRADE_RPC_SLEEP_S", "0.0"))
        batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
        recovered: list[dict[str, Any]] = []
        t0 = time.time()

        def save_cache(rows: list[dict[str, Any]]) -> None:
            nonlocal cached
            if not rows:
                return
            cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
            cached = cached.drop_duplicates("transaction_hash", keep="last")
            con.register("_tx_cache_df", cached)
            con.execute(
                f"COPY _tx_cache_df TO '{TX_CACHE_PATH}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )

        if workers <= 1:
            pending: list[dict[str, Any]] = []
            for i, batch in enumerate(batches, 1):
                try:
                    pending.extend(
                        _rpc_batch(
                            rpc_url,
                            batch,
                            timeout_s=30.0,
                            max_attempts=3,
                        )
                    )
                except Exception as exc:
                    log(f"  warning: tx.from batch {i} failed after retries: {exc}")
                if sleep_s > 0:
                    time.sleep(sleep_s)
                if i % 10 == 0 or i == len(batches):
                    recovered.extend(pending)
                    save_cache(pending)
                    pending = []
                    log(
                        f"  tx.from batches {i:,}/{len(batches):,}; "
                        f"rows={len(recovered):,}; elapsed={time.time() - t0:,.0f}s"
                    )
        else:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = [
                    executor.submit(
                        _rpc_batch,
                        rpc_url,
                        batch,
                        timeout_s=30.0,
                        max_attempts=3,
                    )
                    for batch in batches
                ]
                pending = []
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        pending.extend(future.result())
                    except Exception as exc:
                        log(f"  warning: tx.from batch failed after retries: {exc}")
                    if i % 10 == 0 or i == len(futures):
                        recovered.extend(pending)
                        save_cache(pending)
                        pending = []
                        log(
                            f"  tx.from batches {i:,}/{len(futures):,}; "
                            f"rows={len(recovered):,}; elapsed={time.time() - t0:,.0f}s"
                        )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE copytrade_tx_cache AS
        SELECT * FROM read_parquet('{TX_CACHE_PATH}')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE copytrade_recovery_sample AS
        WITH recovered AS (
            SELECT
                b.* EXCLUDE (tx_from_recovered, block_number, sample_weight),
                c.tx_from_recovered,
                c.block_number,
                b.month_n / count(*) OVER (PARTITION BY b.month) AS sample_weight
            FROM copytrade_recovery_sample_base b
            LEFT JOIN copytrade_tx_cache c USING (transaction_hash)
        )
        SELECT * FROM recovered
        """
    )
    con.execute(
        f"COPY copytrade_recovery_sample TO '{SAMPLE_PATH}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    stats = con.sql(
        """
        SELECT
            count(*) AS rows,
            count(DISTINCT transaction_hash) AS n_transactions,
            count(*) FILTER (WHERE tx_from_recovered IS NOT NULL) AS recovered_rows,
            count(DISTINCT tx_from_recovered) FILTER (WHERE tx_from_recovered IS NOT NULL)
                AS distinct_tx_from,
            min(ts) AS first_ts,
            max(ts) AS last_ts
        FROM copytrade_recovery_sample
        """
    ).fetchdf().iloc[0].to_dict()
    log(f"recovery sample written: {stats}")
    return stats


def bootstrap_invisible_share_ci(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    log("bootstrapping invisible-share CIs on (market_id, day) blocks")
    sample = con.sql(
        """
        SELECT
            tx_from_recovered AS address,
            market_id,
            CAST(date_trunc('day', ts) AS DATE) AS day,
            usd_amount,
            sample_weight
        FROM copytrade_recovery_sample
        WHERE tx_from_recovered IS NOT NULL
        """
    ).fetchdf()
    if sample.empty:
        return pd.DataFrame(columns=["address", "invisible_share_ci_low", "invisible_share_ci_high"])

    visible = con.sql(
        f"""
        SELECT address, total_volume_usd AS visible_volume
        FROM read_parquet('{TRADERS}')
        WHERE address IN (SELECT DISTINCT tx_from_recovered FROM copytrade_recovery_sample)
        """
    ).fetchdf()
    visible_map = visible.set_index("address")["visible_volume"].to_dict()

    sample["weighted_usd"] = sample["usd_amount"] * sample["sample_weight"]
    observed = sorted(sample["address"].dropna().unique().tolist())
    addr_index = {addr: i for i, addr in enumerate(observed)}
    visible_arr = np.array([float(visible_map.get(addr, 0.0) or 0.0) for addr in observed])

    sample["block"] = sample["market_id"].astype(str) + "|" + sample["day"].astype(str)
    block_ids = sample["block"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    boot = np.zeros((BOOTSTRAP_REPS, len(observed)), dtype=np.float64)

    for b in range(BOOTSTRAP_REPS):
        drawn = rng.choice(block_ids, size=len(block_ids), replace=True)
        mult = pd.Series(drawn).value_counts().rename("mult")
        rows = sample.join(mult, on="block", how="inner")
        grouped = rows.groupby("address", sort=False)["weighted_usd"].apply(
            lambda s: float(np.sum(s.to_numpy() * rows.loc[s.index, "mult"].to_numpy()))
        )
        idx = np.fromiter((addr_index[a] for a in grouped.index), dtype=np.int64)
        vals = grouped.to_numpy(dtype=np.float64)
        shares = np.zeros(len(observed), dtype=np.float64)
        shares[idx] = vals / np.maximum(visible_arr[idx] + vals, 1e-12)
        boot[b] = shares
        if (b + 1) % 50 == 0:
            log(f"  bootstrap reps {b + 1}/{BOOTSTRAP_REPS}")

    ci = pd.DataFrame(
        {
            "address": observed,
            "invisible_share_ci_low": np.quantile(boot, 0.025, axis=0),
            "invisible_share_ci_high": np.quantile(boot, 0.975, axis=0),
        }
    )
    return ci


def build_estimates(con: duckdb.DuckDBPyConnection, v1_total: float) -> dict[str, Any]:
    log("building per-trader invisible-take estimates")
    con.execute(
        """
        CREATE OR REPLACE TABLE copytrade_sample_estimates AS
        SELECT
            tx_from_recovered AS address,
            sum(sample_weight)::DOUBLE AS est_invisible_take_count,
            sum(usd_amount * sample_weight)::DOUBLE AS est_invisible_take_volume_usd,
            count(*) AS sample_rows
        FROM copytrade_recovery_sample
        WHERE tx_from_recovered IS NOT NULL
        GROUP BY tx_from_recovered
        """
    )
    ci = bootstrap_invisible_share_ci(con)
    con.register("_invisible_ci", ci)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE copytrade_invisible_take_estimates AS
        SELECT
            t.address,
            t.is_operator_like,
            t.style_maker_fill_count::DOUBLE AS n_maker_fills_total,
            CASE WHEN t.address IN ({sql_list(V1_EXCHANGE)})
                 THEN 0::DOUBLE
                 ELSE coalesce(t.style_taker_fill_count, 0)::DOUBLE
            END AS n_taker_fills_total,
            coalesce(m.n_v1_internal_taker_legs_paired_with_addr_as_maker, 0)::DOUBLE
                AS n_v1_internal_taker_legs_paired_with_addr_as_maker,
            {float(v1_total)}::DOUBLE AS n_v1_internal_taker_legs_total,
            coalesce(se.est_invisible_take_count, 0)::DOUBLE AS est_invisible_take_count,
            coalesce(se.est_invisible_take_volume_usd, 0)::DOUBLE
                AS est_invisible_take_volume_usd,
            t.total_volume_usd::DOUBLE AS visible_volume_usd,
            CASE WHEN coalesce(t.total_volume_usd, 0) + coalesce(se.est_invisible_take_volume_usd, 0) > 0
                 THEN coalesce(se.est_invisible_take_volume_usd, 0)
                      / (coalesce(t.total_volume_usd, 0) + coalesce(se.est_invisible_take_volume_usd, 0))
                 ELSE 0 END AS invisible_share,
            ci.invisible_share_ci_low,
            ci.invisible_share_ci_high,
            coalesce(se.sample_rows, 0) AS recovery_sample_rows,
            t.n_closed_positions,
            t.active_days,
            t.mkt_total_pnl,
            t.pos_total_pnl,
            t.pos_sharpe,
            t.mkt_win_rate,
            t.pos_win_rate,
            t.style_maker_taker_ratio,
            t.style_role_balance
        FROM traders t
        LEFT JOIN copytrade_sample_estimates se USING (address)
        LEFT JOIN copytrade_v1_internal_by_maker m USING (address)
        LEFT JOIN _invisible_ci ci USING (address)
        """
    )
    con.execute(
        f"COPY copytrade_invisible_take_estimates TO '{ESTIMATES_PATH}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    stats = con.sql(
        """
        SELECT
            count(*) AS rows,
            count(*) FILTER (WHERE est_invisible_take_count > 0) AS addresses_with_sample_estimate,
            sum(est_invisible_take_count) AS est_total_count,
            sum(est_invisible_take_volume_usd) AS est_total_usd,
            avg(invisible_share) FILTER (WHERE NOT is_operator_like) AS mean_share_filtered,
            median(invisible_share) FILTER (WHERE NOT is_operator_like) AS median_share_filtered
        FROM copytrade_invisible_take_estimates
        """
    ).fetchdf().iloc[0].to_dict()
    log(f"estimates written: {stats}")
    return stats


def build_inspection_cohort(con: duckdb.DuckDBPyConnection) -> list[str]:
    fixed = [a.lower() for a in MANUAL_CANDIDATES + [DOMAH]]
    fixed_sql = sql_list(fixed)
    random_rows = con.sql(
        f"""
        SELECT address
        FROM traders
        WHERE NOT is_operator_like
          AND n_closed_positions > 200
          AND active_days > 90
          AND address NOT IN ({fixed_sql})
        ORDER BY hash(address)
        LIMIT 50
        """
    ).fetchdf()["address"].tolist()
    cohort = fixed + [a.lower() for a in random_rows]
    con.register("_inspection_cohort", pd.DataFrame({"address": cohort}))
    log(f"inspection cohort size: {len(cohort)}")
    return cohort


def build_patched_positions_and_bias(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    log("building patched mini-closed_positions for inspection cohort")
    build_inspection_cohort(con)
    markets_path = latest_markets_path()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE copytrade_cohort_extra_actions AS
        SELECT
            s.tx_from_recovered AS address,
            s.market_id,
            s.condition_id,
            s.neg_risk,
            s.outcome_token_id,
            mt.outcome_index,
            s.ts,
            CASE WHEN s.maker_asset_id = '0'
                 THEN -s.token_amount * s.sample_weight
                 ELSE  s.token_amount * s.sample_weight END AS token_delta,
            CASE WHEN s.maker_asset_id = '0'
                 THEN  s.usd_amount * s.sample_weight
                 ELSE -s.usd_amount * s.sample_weight END AS usd_delta,
            s.token_amount * s.sample_weight AS token_amount_weighted,
            s.usd_amount * s.sample_weight AS usd_amount_weighted,
            s.sample_weight,
            s.transaction_hash
        FROM copytrade_recovery_sample s
        JOIN _inspection_cohort c ON c.address = s.tx_from_recovered
        JOIN markets_tokens mt
          ON mt.market_id = s.market_id
         AND mt.outcome_token_id = s.outcome_token_id
        WHERE s.tx_from_recovered IS NOT NULL
          AND mt.closed = TRUE
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE copytrade_extra_positions AS
        SELECT
            a.address,
            a.market_id,
            any_value(a.condition_id) AS condition_id,
            any_value(a.neg_risk) AS neg_risk,
            any_value(a.outcome_token_id) AS outcome_token_id,
            a.outcome_index,
            sum(a.sample_weight)::DOUBLE AS n_fills,
            min(a.ts) AS first_fill_ts,
            max(a.ts) AS last_fill_ts,
            sum(abs(a.token_delta)) AS gross_token_volume,
            sum(a.usd_amount_weighted) AS gross_usd_volume,
            sum(CASE WHEN a.token_delta > 0 THEN a.usd_amount_weighted ELSE 0 END)
                AS total_bought_usd,
            sum(CASE WHEN a.token_delta < 0 THEN a.usd_amount_weighted ELSE 0 END)
                AS total_sold_usd,
            sum(a.token_delta) AS final_token_position,
            sum(a.usd_delta) AS realised_cash_flow,
            max(abs(a.token_delta)) AS peak_fill_abs_token
        FROM copytrade_cohort_extra_actions a
        GROUP BY a.address, a.market_id, a.outcome_index
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE copytrade_patched_positions_stage AS
        SELECT
            address,
            market_id,
            condition_id,
            neg_risk,
            outcome_token_id,
            outcome_index,
            n_fills::DOUBLE AS n_fills,
            first_fill_ts,
            last_fill_ts,
            resolution_ts,
            gross_token_volume,
            gross_usd_volume,
            total_bought_usd,
            total_sold_usd,
            final_token_position,
            realised_cash_flow,
            peak_fill_abs_token
        FROM closed_positions
        WHERE address IN (SELECT address FROM _inspection_cohort)
        UNION ALL
        SELECT
            ep.address,
            ep.market_id,
            ep.condition_id,
            ep.neg_risk,
            ep.outcome_token_id,
            ep.outcome_index,
            ep.n_fills,
            ep.first_fill_ts,
            ep.last_fill_ts,
            mt.end_date AS resolution_ts,
            ep.gross_token_volume,
            ep.gross_usd_volume,
            ep.total_bought_usd,
            ep.total_sold_usd,
            ep.final_token_position,
            ep.realised_cash_flow,
            ep.peak_fill_abs_token
        FROM copytrade_extra_positions ep
        JOIN markets_tokens mt
          ON mt.market_id = ep.market_id AND mt.outcome_index = ep.outcome_index
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE copytrade_patched_positions_cohort AS
        WITH agg AS (
            SELECT
                address,
                market_id,
                any_value(condition_id) AS condition_id,
                bool_or(neg_risk) AS neg_risk,
                any_value(outcome_token_id) AS outcome_token_id,
                outcome_index,
                sum(n_fills) AS n_fills,
                min(first_fill_ts) AS first_fill_ts,
                max(last_fill_ts) AS last_fill_ts,
                any_value(resolution_ts) AS resolution_ts,
                sum(gross_token_volume) AS gross_token_volume,
                sum(gross_usd_volume) AS gross_usd_volume,
                sum(total_bought_usd) AS total_bought_usd,
                sum(total_sold_usd) AS total_sold_usd,
                sum(final_token_position) AS final_token_position,
                sum(realised_cash_flow) AS realised_cash_flow,
                max(peak_fill_abs_token) AS peak_fill_abs_token
            FROM copytrade_patched_positions_stage
            GROUP BY address, market_id, outcome_index
        )
        SELECT
            agg.address,
            agg.market_id,
            agg.condition_id,
            agg.neg_risk,
            agg.outcome_token_id,
            agg.outcome_index,
            agg.n_fills,
            agg.first_fill_ts,
            agg.last_fill_ts,
            agg.resolution_ts,
            CASE WHEN agg.resolution_ts IS NOT NULL
                 THEN epoch(agg.resolution_ts - agg.first_fill_ts) / 3600.0
                 ELSE NULL END AS holding_duration_hours,
            agg.gross_token_volume,
            agg.gross_usd_volume,
            agg.total_bought_usd,
            agg.total_sold_usd,
            agg.final_token_position,
            agg.realised_cash_flow,
            CAST(mt.outcome_prices[agg.outcome_index] AS DOUBLE) AS resolution_price,
            agg.final_token_position * CAST(mt.outcome_prices[agg.outcome_index] AS DOUBLE)
                AS redemption_value,
            agg.realised_cash_flow
              + agg.final_token_position * CAST(mt.outcome_prices[agg.outcome_index] AS DOUBLE)
                AS realised_pnl,
            agg.peak_fill_abs_token,
            abs(agg.final_token_position) > 1e-6 AS is_held_to_resolution
        FROM agg
        JOIN markets_tokens mt USING (market_id, outcome_index)
        """
    )
    con.execute(
        f"COPY copytrade_patched_positions_cohort TO '{PATCHED_POSITIONS_PATH}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    log("building cohort bias table")
    con.execute(
        """
        CREATE OR REPLACE TABLE copytrade_patched_metrics AS
        WITH pos AS (
            SELECT
                address,
                sum(realised_pnl) AS patched_pos_total_pnl,
                sum(CASE WHEN realised_pnl > 0 THEN 1 ELSE 0 END) AS pos_winners,
                sum(CASE WHEN realised_pnl < 0 THEN 1 ELSE 0 END) AS pos_losers,
                sum(total_bought_usd + total_sold_usd) AS patched_total_volume_usd
            FROM copytrade_patched_positions_cohort
            GROUP BY address
        ),
        per_market AS (
            SELECT address, market_id, sum(realised_pnl) AS market_pnl
            FROM copytrade_patched_positions_cohort
            GROUP BY address, market_id
        ),
        mkt AS (
            SELECT
                address,
                sum(market_pnl) AS patched_mkt_total_pnl,
                sum(CASE WHEN market_pnl > 0 THEN 1 ELSE 0 END) AS mkt_winners,
                sum(CASE WHEN market_pnl < 0 THEN 1 ELSE 0 END) AS mkt_losers
            FROM per_market
            GROUP BY address
        )
        SELECT
            p.address,
            m.patched_mkt_total_pnl,
            p.patched_pos_total_pnl,
            CASE WHEN m.mkt_winners + m.mkt_losers > 0
                 THEN m.mkt_winners / (m.mkt_winners + m.mkt_losers)
                 ELSE NULL END AS patched_mkt_win_rate,
            CASE WHEN p.pos_winners + p.pos_losers > 0
                 THEN p.pos_winners / (p.pos_winners + p.pos_losers)
                 ELSE NULL END AS patched_pos_win_rate,
            p.patched_total_volume_usd
        FROM pos p
        JOIN mkt m USING (address)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE copytrade_cohort_bias_table AS
        SELECT
            c.address,
            t.n_fills_total AS n_visible_fills,
            e.est_invisible_take_count AS est_invisible_takes,
            e.est_invisible_take_volume_usd AS est_invisible_volume_usd,
            t.mkt_total_pnl AS visible_mkt_total_pnl,
            pm.patched_mkt_total_pnl,
            pm.patched_mkt_total_pnl - t.mkt_total_pnl AS mkt_pnl_delta_usd,
            CASE WHEN abs(t.mkt_total_pnl) > 1e-9
                 THEN (pm.patched_mkt_total_pnl - t.mkt_total_pnl) / abs(t.mkt_total_pnl)
                 ELSE NULL END AS mkt_pnl_delta_pct,
            t.pos_total_pnl AS visible_pos_total_pnl,
            pm.patched_pos_total_pnl,
            pm.patched_pos_total_pnl - t.pos_total_pnl AS pos_pnl_delta_usd,
            CASE WHEN abs(t.pos_total_pnl) > 1e-9
                 THEN (pm.patched_pos_total_pnl - t.pos_total_pnl) / abs(t.pos_total_pnl)
                 ELSE NULL END AS pos_pnl_delta_pct,
            t.mkt_win_rate AS visible_mkt_win_rate,
            pm.patched_mkt_win_rate,
            100.0 * (pm.patched_mkt_win_rate - t.mkt_win_rate) AS win_rate_delta_pp,
            t.total_volume_usd AS visible_total_volume_usd,
            pm.patched_total_volume_usd,
            CASE WHEN t.total_volume_usd > 0
                 THEN (pm.patched_total_volume_usd - t.total_volume_usd) / t.total_volume_usd
                 ELSE NULL END AS volume_delta_pct,
            CASE
              WHEN t.style_maker_fill_count / nullif(t.style_taker_fill_count + e.est_invisible_take_count, 0) > 4
                THEN 'maker_heavy'
              WHEN t.style_maker_fill_count / nullif(t.style_taker_fill_count + e.est_invisible_take_count, 0) < 0.25
                THEN 'taker_heavy'
              ELSE 'mixed'
            END AS style_label,
            CASE
              WHEN e.recovery_sample_rows = 0 THEN 'no recovered rows in 50k sample; patched equals visible lower bound'
              WHEN e.invisible_share_ci_high IS NOT NULL
                   AND e.invisible_share_ci_high - e.invisible_share_ci_low > 0.25
                THEN 'wide sample CI'
              ELSE 'sample-weighted patched estimate'
            END AS notes
        FROM _inspection_cohort c
        JOIN traders t USING (address)
        JOIN copytrade_invisible_take_estimates e USING (address)
        JOIN copytrade_patched_metrics pm USING (address)
        ORDER BY
            CASE WHEN c.address IN ('{DOMAH}', {sql_list(MANUAL_CANDIDATES)}) THEN 0 ELSE 1 END,
            abs(pm.patched_mkt_total_pnl - t.mkt_total_pnl) DESC
        """
    )
    con.execute(
        f"COPY copytrade_cohort_bias_table TO '{COHORT_BIAS_PATH}' "
        "(HEADER, DELIMITER ',')"
    )
    stats = con.sql(
        """
        SELECT
            count(*) AS rows,
            max(abs(mkt_pnl_delta_pct)) AS max_abs_mkt_pnl_delta_pct,
            max(abs(volume_delta_pct)) AS max_abs_volume_delta_pct,
            avg(abs(volume_delta_pct)) AS mean_abs_volume_delta_pct
        FROM copytrade_cohort_bias_table
        """
    ).fetchdf().iloc[0].to_dict()
    log(f"cohort artifacts written: {stats}")
    return stats


def style_bucket_expr() -> str:
    return """
        CASE
          WHEN t.style_maker_taker_ratio > 4 THEN 'maker_heavy'
          WHEN t.style_maker_taker_ratio < 0.25 THEN 'taker_heavy'
          ELSE 'mixed'
        END
    """


def _bootstrap_stat(values: np.ndarray, stat: str, reps: int = 500) -> tuple[float, float]:
    if len(values) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(RANDOM_SEED + (1 if stat == "mean" else 2))
    vals = values.astype(float)
    if len(vals) > 200_000:
        vals = rng.choice(vals, size=200_000, replace=False)
    n = len(vals)
    out = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        if stat == "mean":
            out[i] = float(np.mean(vals[idx]))
        else:
            out[i] = float(np.median(vals[idx]))
    return (float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)))


def _spearman_ci(rho: float, n: int) -> tuple[float, float]:
    if not np.isfinite(rho) or n <= 3 or abs(rho) >= 1:
        return (math.nan, math.nan)
    z = np.arctanh(rho)
    se = 1.0 / math.sqrt(n - 3)
    return (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))


def build_systematicity(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    log("building mkt/pos std sidecar for systematicity correlations")
    con.execute(
        """
        CREATE OR REPLACE TABLE copytrade_m_std_aux AS
        WITH pos_lvl AS (
            SELECT address, stddev_pop(realised_pnl) AS pos_std_pnl
            FROM closed_positions
            GROUP BY address
        ),
        mkt_per_market AS (
            SELECT address, market_id, sum(realised_pnl) AS market_pnl
            FROM closed_positions
            GROUP BY address, market_id
        ),
        mkt_lvl AS (
            SELECT address, stddev_pop(market_pnl) AS mkt_std_pnl
            FROM mkt_per_market
            GROUP BY address
        )
        SELECT pl.address, pl.pos_std_pnl, ml.mkt_std_pnl
        FROM pos_lvl pl
        LEFT JOIN mkt_lvl ml USING (address)
        """
    )
    log("loading slim systematicity frame into pandas")
    all_cols = list(dict.fromkeys(COHORT_SELECTION_COLUMNS + REQUESTED_CORRELATION_COLUMNS))
    select_cols = ", ".join(f"base.{c}" if c not in {"mkt_std_pnl", "pct_markets_balanced_and_offsetting_vw"} else c for c in all_cols)
    df = con.sql(
        f"""
        WITH base AS (
            SELECT
                e.address,
                e.invisible_share,
                e.is_operator_like,
                {style_bucket_expr()} AS style_bucket,
                t.mkt_sharpe,
                s.mkt_std_pnl,
                t.n_closed_positions,
                t.active_days,
                t.negrisk_volume_share,
                t.mkt_profit_factor,
                t.mkt_total_pnl,
                d.pct_markets_balanced_and_offsetting_vw,
                t.style_avg_holding_hours,
                t.style_role_balance,
                t.mkt_kelly_fraction,
                t.mkt_dollar_win_rate,
                t.pos_sharpe,
                t.pos_win_rate,
                t.style_maker_taker_ratio
            FROM copytrade_invisible_take_estimates e
            JOIN traders t USING (address)
            LEFT JOIN copytrade_m_std_aux s USING (address)
            LEFT JOIN read_parquet('{DIRECTIONALITY}') d USING (address)
            WHERE NOT e.is_operator_like
        )
        SELECT address, invisible_share, style_bucket, {select_cols}
        FROM base
        """
    ).fetchdf()

    rows: list[dict[str, Any]] = []
    for bucket, grp in df.groupby("style_bucket", dropna=False):
        vals = grp["invisible_share"].to_numpy(dtype=float)
        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))
        mean_ci = _bootstrap_stat(vals, "mean", reps=300)
        median_ci = _bootstrap_stat(vals, "median", reps=300)
        rows.append(
            {
                "section": "style_bucket",
                "metric": "invisible_share_mean",
                "bucket_or_column": bucket,
                "n": len(vals),
                "value": mean_val,
                "ci_low": mean_ci[0],
                "ci_high": mean_ci[1],
            }
        )
        rows.append(
            {
                "section": "style_bucket",
                "metric": "invisible_share_median",
                "bucket_or_column": bucket,
                "n": len(vals),
                "value": median_val,
                "ci_low": median_ci[0],
                "ci_high": median_ci[1],
            }
        )

    inv_rank = df["invisible_share"].rank(method="average")
    for col in all_cols:
        tmp = pd.DataFrame({"x": inv_rank, "y": df[col]})
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
        n = len(tmp)
        if n < 4 or tmp["y"].nunique(dropna=True) < 2:
            rho = math.nan
            ci_low = math.nan
            ci_high = math.nan
        else:
            y_rank = tmp["y"].rank(method="average")
            rho = float(tmp["x"].corr(y_rank))
            ci_low, ci_high = _spearman_ci(rho, n)
        rows.append(
            {
                "section": (
                    "requested_correlation"
                    if col in REQUESTED_CORRELATION_COLUMNS
                    else "cohort_selection_correlation"
                ),
                "metric": "spearman_rho",
                "bucket_or_column": col,
                "n": n,
                "value": rho,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(SYSTEMATICITY_PATH, index=False)
    log(f"systematicity written: {SYSTEMATICITY_PATH}")
    headline = out[
        (out["section"].isin(["style_bucket", "requested_correlation"]))
    ].to_dict(orient="records")
    return {"rows": len(out), "headline": headline}


def domah_implications(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    domah_bias = con.sql(
        f"SELECT * FROM copytrade_cohort_bias_table WHERE address = '{DOMAH}'"
    ).fetchdf()
    domah_est = con.sql(
        f"""
        SELECT *
        FROM copytrade_invisible_take_estimates
        WHERE address = '{DOMAH}'
        """
    ).fetchdf()
    audit_family_path = ANALYSIS / "domah_audit_family.parquet"
    audit_role_path = ANALYSIS / "domah_audit_slice_role.parquet"
    audit_hour_path = ANALYSIS / "domah_audit_slice_hour.parquet"
    family = (
        con.sql(f"SELECT * FROM read_parquet('{audit_family_path}')").fetchdf()
        if audit_family_path.exists()
        else pd.DataFrame()
    )
    role = (
        con.sql(f"SELECT * FROM read_parquet('{audit_role_path}')").fetchdf()
        if audit_role_path.exists()
        else pd.DataFrame()
    )
    hour = (
        con.sql(f"SELECT * FROM read_parquet('{audit_hour_path}')").fetchdf()
        if audit_hour_path.exists()
        else pd.DataFrame()
    )

    invisible_by_hour = con.sql(
        f"""
        SELECT
            CASE
                WHEN hour(ts) < 6 THEN '00-06'
                WHEN hour(ts) < 12 THEN '06-12'
                WHEN hour(ts) < 18 THEN '12-18'
                ELSE '18-24'
            END AS hour_bucket,
            count(*) AS sample_rows,
            sum(sample_weight) AS est_count,
            sum(usd_amount * sample_weight) AS est_usd
        FROM copytrade_recovery_sample
        WHERE tx_from_recovered = '{DOMAH}'
        GROUP BY 1
        ORDER BY est_usd DESC
        """
    ).fetchdf()

    return {
        "bias": domah_bias.to_dict(orient="records"),
        "estimate": domah_est.to_dict(orient="records"),
        "audit_family": family.to_dict(orient="records"),
        "audit_role": role.to_dict(orient="records"),
        "audit_hour": hour.to_dict(orient="records"),
        "invisible_by_hour": invisible_by_hour.to_dict(orient="records"),
    }


def json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, pd.Timestamp)):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=UTC)
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    if pd.isna(obj):
        return None
    return str(obj)


def main() -> int:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    t0 = time.time()
    con = connect()
    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC),
        "artifacts": {
            "sample": str(SAMPLE_PATH.relative_to(ROOT)),
            "estimates": str(ESTIMATES_PATH.relative_to(ROOT)),
            "patched_positions": str(PATCHED_POSITIONS_PATH.relative_to(ROOT)),
            "cohort_bias": str(COHORT_BIAS_PATH.relative_to(ROOT)),
            "systematicity": str(SYSTEMATICITY_PATH.relative_to(ROOT)),
        },
    }
    summary["schema_gate"] = schema_gate(con)
    population = build_population_tables(con)
    summary.update(population)
    summary["recovery_sample"] = recover_tx_from(con)
    summary["estimates"] = build_estimates(
        con,
        v1_total=float(population["v1_internal_totals"]["n_v1_internal"]),
    )
    summary["cohort_bias"] = build_patched_positions_and_bias(con)
    summary["systematicity"] = build_systematicity(con)
    summary["domah"] = domah_implications(con)
    summary["elapsed_seconds"] = time.time() - t0

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default)
    )
    log(f"summary written: {SUMMARY_PATH}")
    log(f"done in {time.time() - t0:,.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
