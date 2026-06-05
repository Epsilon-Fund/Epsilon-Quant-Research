"""Politics NegRisk accounting audit for the K5-STRESS maker cell.

This script reconstructs non-trade NegRisk inventory movements for the
structured/non-top3 politics maker cohort. The local fill shards do not contain
split, merge, redeem, or convert operations; this script discovers those
operations from Polymarket's public activity feed, verifies/decodes the relevant
transactions from Polygon receipts, and then adjusts the cached K5 wallet-market
PnL.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/mm_politics_negrisk_accounting.py --refresh-activity --refresh-receipts

The script does not mutate source parquet shards. It writes analysis artifacts
under data/analysis and CSV summaries under data/analysis/csv_outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_infra.operator_denylist import EXCHANGE_INTERNAL_LEG
from scripts.sync_polygon_order_fills import DEFAULT_RPC_URL, rpc_call


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
CSV_DIR = ANALYSIS / "csv_outputs" / "market_making"
TRADES_GLOB = str(DATA / "trades" / "*.parquet")
CLOSED_POSITIONS = DATA / "closed_positions.parquet"
MARKETS = sorted((DATA / "markets").glob("markets_*.parquet"))[-1]
TOKEN_CACHE = DATA / "markets" / "gamma_token_lookup_cache.jsonl"
WALLET_MARKET = ANALYSIS / "k5_stress_wallet_market_full.parquet"

ACTIVITY_RAW = ANALYSIS / "mm_politics_negrisk_activity_raw.parquet"
ACTIVITY_CHECKPOINT = ANALYSIS / "mm_politics_negrisk_activity_checkpoint.jsonl"
RECEIPT_DELTAS = ANALYSIS / "mm_politics_negrisk_receipt_deltas.parquet"
RECEIPT_CHECKPOINT = ANALYSIS / "mm_politics_negrisk_receipt_checkpoint.jsonl"
ACCOUNTING_PARQUET = ANALYSIS / "mm_politics_negrisk_accounting_wallet_market.parquet"
RECUT_PARQUET = ANALYSIS / "mm_politics_negrisk_corrected_carry_recut_wallet_market.parquet"
SUMMARY_CSV = CSV_DIR / "mm_politics_negrisk_accounting_summary.csv"
CAPACITY_CSV = CSV_DIR / "mm_politics_negrisk_capacity_ladder.csv"
EVENT_AUDIT_CSV = CSV_DIR / "mm_politics_negrisk_event_audit.csv"

API_BASE = "https://data-api.polymarket.com"
USER_AGENT = "epsilon-mm-politics-negrisk-accounting/0.1"
ACTIVITY_START_TS = int(datetime(2023, 12, 1, tzinfo=UTC).timestamp())
ACTIVITY_END_TS = int(datetime(2026, 6, 3, tzinfo=UTC).timestamp())

RNG_SEED = 20260602
BOOTSTRAP_SAMPLES = 500
STRUCT_TWO_SIDED_MIN = 0.60
STRUCT_CARRY_MIN = 0.50
STRUCT_SPIKE_MAX = 0.02

CTF = "0x4d97dcd97ec945f40cf65f87097ace5ea0476045"
USDC = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"
ZERO_ADDR = "0x0000000000000000000000000000000000000000"

ERC20_TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
ERC1155_TRANSFER_SINGLE_TOPIC = (
    "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
)
ERC1155_TRANSFER_BATCH_TOPIC = (
    "0x4a39dc06d4c0dbc64b70af90fd698a233a518aa5d07e595d983b8c0526c8f7fb"
)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA preserve_insertion_order=false")
    temp_dir = ANALYSIS / ".duckdb_tmp_mm_politics_negrisk"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    return con


def sql_list(values: list[str] | set[str]) -> str:
    vals = sorted({str(v).lower().replace("'", "''") for v in values if str(v)})
    return ",".join(f"'{v}'" for v in vals) if vals else "''"


def politics_classifier_expr() -> str:
    """Mirror the K5-STRESS politics_negrisk category predicate."""
    return """
        neg_risk AND (
            slug_l LIKE '%election%' OR slug_l LIKE '%president%' OR slug_l LIKE '%senate%'
            OR slug_l LIKE '%congress%' OR slug_l LIKE '%trump%' OR slug_l LIKE '%biden%'
            OR slug_l LIKE '%democrat%' OR slug_l LIKE '%republican%'
            OR question_l LIKE '%election%' OR question_l LIKE '%president%' OR question_l LIKE '%senate%'
            OR question_l LIKE '%congress%' OR question_l LIKE '%trump%' OR question_l LIKE '%biden%'
        )
    """


def setup_base_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE politics_wm AS
        SELECT *
        FROM read_parquet('{WALLET_MARKET}')
        WHERE category = 'politics_negrisk'
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE wallet_category AS
        SELECT
            address,
            category,
            count(DISTINCT market_id) AS markets,
            sum(gross_usd_volume) AS gross_usd_volume,
            sum(net_pnl_usd) AS net_pnl_usd,
            sum(maker_rebate_usd) AS maker_rebate_usd,
            sum(taker_fee_usd) AS taker_fee_usd,
            sum(maker_fills) AS maker_fills,
            sum(maker_usd) AS maker_usd,
            sum(gross_token_volume) AS gross_token_volume,
            sum(abs_final_token_position) AS abs_final_token_position,
            sum(spike_zone_usd) AS spike_zone_usd,
            sum(CASE WHEN distinct_outcomes_made >= 2 THEN maker_usd ELSE 0 END) AS two_sided_maker_usd
        FROM politics_wm
        GROUP BY 1, 2
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE structured_wallets AS
        SELECT
            *,
            two_sided_maker_usd / nullif(maker_usd, 0) AS two_sided_usd_share,
            abs_final_token_position / nullif(gross_token_volume, 0) AS carry_token_share,
            spike_zone_usd / nullif(maker_usd, 0) AS spike_zone_usd_share,
            (
                two_sided_maker_usd / nullif(maker_usd, 0) >= {STRUCT_TWO_SIDED_MIN}
                AND abs_final_token_position / nullif(gross_token_volume, 0) >= {STRUCT_CARRY_MIN}
                AND coalesce(spike_zone_usd / nullif(maker_usd, 0), 0) <= {STRUCT_SPIKE_MAX}
                AND maker_fills > 0
            ) AS structured_playbook
        FROM wallet_category
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE target_wm AS
        SELECT wm.*
        FROM politics_wm wm
        JOIN structured_wallets sw USING (address, category)
        WHERE sw.structured_playbook
          AND NOT wm.is_global_top3_market_maker
        """
    )


def build_token_map(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE token_map_base AS
        SELECT DISTINCT
            CAST(cp.market_id AS VARCHAR) AS market_id,
            CAST(cp.condition_id AS VARCHAR) AS condition_id,
            CAST(cp.outcome_token_id AS VARCHAR) AS outcome_token_id,
            cp.outcome_index
        FROM read_parquet('{CLOSED_POSITIONS}') cp
        JOIN (SELECT DISTINCT market_id FROM politics_wm) m ON CAST(cp.market_id AS VARCHAR) = m.market_id

        UNION

        SELECT DISTINCT
            CAST(rt.market_id AS VARCHAR) AS market_id,
            CAST(rt.condition_id AS VARCHAR) AS condition_id,
            CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END AS outcome_token_id,
            NULL::BIGINT AS outcome_index
        FROM read_parquet('{TRADES_GLOB}') rt
        JOIN (SELECT DISTINCT market_id FROM politics_wm) m ON CAST(rt.market_id AS VARCHAR) = m.market_id
        WHERE rt.market_id IS NOT NULL
          AND rt.maker IS NOT NULL
          AND rt.taker IS NOT NULL
          AND rt.maker <> rt.taker
          AND lower(rt.maker) NOT IN ({internals})
          AND lower(rt.taker) NOT IN ({internals})

        UNION

        SELECT DISTINCT
            CAST(m.id AS VARCHAR) AS market_id,
            CAST(m.condition_id AS VARCHAR) AS condition_id,
            CAST(m.clob_token_ids[r.i] AS VARCHAR) AS outcome_token_id,
            r.i - 1 AS outcome_index
        FROM read_parquet('{MARKETS}') m,
             range(1, len(m.clob_token_ids) + 1) AS r(i)
        JOIN (SELECT DISTINCT market_id FROM politics_wm) pm ON CAST(m.id AS VARCHAR) = pm.market_id
        WHERE len(m.clob_token_ids) > 0
        """
    )
    # Add token-cache entries only for known politics condition IDs. These rows
    # sometimes cover markets absent from the latest compact markets snapshot.
    if TOKEN_CACHE.exists():
        rows: list[dict[str, Any]] = []
        condition_ids = {
            r[0]
            for r in con.execute("SELECT DISTINCT lower(condition_id) FROM token_map_base").fetchall()
            if r[0]
        }
        with TOKEN_CACHE.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = str(row.get("condition_id", "")).lower()
                if cid in condition_ids:
                    rows.append(
                        {
                            "market_id": str(row.get("market_id")),
                            "condition_id": str(row.get("condition_id")),
                            "outcome_token_id": str(row.get("token_id")),
                            "outcome_index": None,
                        }
                    )
        if rows:
            cache_df = pd.DataFrame(rows)
            con.register("token_cache_rows", cache_df)
            con.execute(
                """
                INSERT INTO token_map_base
                SELECT DISTINCT market_id, condition_id, outcome_token_id, outcome_index
                FROM token_cache_rows
                """
            )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE token_map AS
        SELECT
            any_value(market_id) AS market_id,
            any_value(condition_id) AS condition_id,
            outcome_token_id,
            min(outcome_index) AS outcome_index
        FROM token_map_base
        WHERE outcome_token_id IS NOT NULL
        GROUP BY outcome_token_id
        """
    )
    return con.execute("SELECT * FROM token_map").df()


def build_mark_prices(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE closed_marks AS
        SELECT
            CAST(market_id AS VARCHAR) AS market_id,
            CAST(outcome_token_id AS VARCHAR) AS outcome_token_id,
            max(resolution_ts) AS mark_ts,
            any_value(resolution_price) AS resolution_price
        FROM read_parquet('{CLOSED_POSITIONS}')
        WHERE CAST(market_id AS VARCHAR) IN (SELECT DISTINCT market_id FROM politics_wm)
        GROUP BY 1, 2
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE latest_trade_marks AS
        WITH t AS (
            SELECT
                CAST(rt.market_id AS VARCHAR) AS market_id,
                CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END AS outcome_token_id,
                rt.price,
                rt.timestamp,
                row_number() OVER (
                    PARTITION BY CAST(rt.market_id AS VARCHAR),
                    CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END
                    ORDER BY rt.timestamp DESC
                ) AS rn
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN (SELECT DISTINCT market_id FROM politics_wm) m ON CAST(rt.market_id AS VARCHAR) = m.market_id
            WHERE rt.market_id IS NOT NULL
        )
        SELECT market_id, outcome_token_id, price AS latest_price, timestamp AS latest_ts
        FROM t
        WHERE rn = 1
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE market_snapshot_marks AS
        SELECT
            CAST(m.id AS VARCHAR) AS market_id,
            CAST(m.clob_token_ids[r.i] AS VARCHAR) AS outcome_token_id,
            TRY_CAST(m.outcome_prices[r.i] AS DOUBLE) AS snapshot_price
        FROM read_parquet('{MARKETS}') m,
             range(1, len(m.clob_token_ids) + 1) AS r(i)
        WHERE CAST(m.id AS VARCHAR) IN (SELECT DISTINCT market_id FROM politics_wm)
          AND len(m.clob_token_ids) > 0
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE token_marks AS
        SELECT
            tm.market_id,
            tm.condition_id,
            tm.outcome_token_id,
            coalesce(cm.resolution_price, ltm.latest_price, msm.snapshot_price) AS mark_price,
            CASE
                WHEN cm.resolution_price IS NOT NULL THEN 'settlement'
                WHEN ltm.latest_price IS NOT NULL THEN 'last_trade_proxy'
                WHEN msm.snapshot_price IS NOT NULL THEN 'market_snapshot'
                ELSE 'missing'
            END AS mark_source
        FROM token_map tm
        LEFT JOIN closed_marks cm USING (market_id, outcome_token_id)
        LEFT JOIN latest_trade_marks ltm USING (market_id, outcome_token_id)
        LEFT JOIN market_snapshot_marks msm USING (market_id, outcome_token_id)
        """
    )
    return con.execute("SELECT * FROM token_marks").df()


def fetch_activity(refresh: bool) -> pd.DataFrame:
    if ACTIVITY_RAW.exists() and not refresh:
        return pq.read_table(ACTIVITY_RAW).to_pandas()

    con = connect()
    setup_base_tables(con)
    wallets = [r[0] for r in con.execute("SELECT DISTINCT address FROM structured_wallets WHERE structured_playbook").fetchall()]
    rows: list[dict[str, Any]] = []
    seen_wallets: set[str] = set()
    if ACTIVITY_CHECKPOINT.exists():
        with ACTIVITY_CHECKPOINT.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(row)
        seen_wallets = {str(r.get("address", "")).lower() for r in rows if r.get("address")}
        if rows:
            print(
                f"[activity] loaded checkpoint rows={len(rows):,} wallets={len(seen_wallets):,}",
                flush=True,
            )

    def get_with_retry(client: httpx.Client, url: str) -> httpx.Response | None:
        for attempt in range(1, 6):
            try:
                return client.get(url)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == 5:
                    print(f"[activity] timeout after retries url={url}", flush=True)
                    return None
                time.sleep(min(2.0 * attempt, 8.0))
        return None

    def fetch_wallet_window(
        client: httpx.Client,
        wallet: str,
        start_ts: int | None,
        end_ts: int | None,
        *,
        depth: int = 0,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            url = (
                f"{API_BASE}/activity?user={wallet}"
                "&type=SPLIT,MERGE,REDEEM,CONVERSION"
                f"&limit=500&offset={offset}"
            )
            if start_ts is not None and end_ts is not None:
                url += f"&start={start_ts}&end={end_ts}"
            response = get_with_retry(client, url)
            if response is None and depth == 0:
                return fetch_wallet_window(
                    client, wallet, ACTIVITY_START_TS, ACTIVITY_END_TS, depth=depth + 1
                )
            if (
                response is None
                and start_ts is not None
                and end_ts is not None
                and end_ts - start_ts > 60
                and depth < 24
            ):
                mid = (start_ts + end_ts) // 2
                return (
                    fetch_wallet_window(client, wallet, start_ts, mid, depth=depth + 1)
                    + fetch_wallet_window(client, wallet, mid, end_ts, depth=depth + 1)
                )
            if response is None:
                print(
                    f"[activity] WARNING: dropping timed-out leaf wallet={wallet} "
                    f"start={start_ts} end={end_ts} offset={offset}",
                    flush=True,
                )
                break
            if response.status_code == 400 and depth == 0:
                return fetch_wallet_window(
                    client, wallet, ACTIVITY_START_TS, ACTIVITY_END_TS, depth=depth + 1
                )
            if (
                response.status_code == 400
                and start_ts is not None
                and end_ts is not None
                and end_ts - start_ts > 60
                and depth < 24
            ):
                # Dense wallets can exceed the endpoint's effective offset
                # depth. Split by time so we still get a complete history.
                mid = (start_ts + end_ts) // 2
                return (
                    fetch_wallet_window(client, wallet, start_ts, mid, depth=depth + 1)
                    + fetch_wallet_window(client, wallet, mid, end_ts, depth=depth + 1)
                )
            if response.status_code == 400 and depth >= 24:
                print(
                    f"[activity] WARNING: dropping over-dense leaf wallet={wallet} "
                    f"start={start_ts} end={end_ts} offset={offset}",
                    flush=True,
                )
                break
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list) or not page:
                break
            for row in page:
                row["address"] = wallet.lower()
                rows.append(row)
            if len(page) < 500:
                break
            offset += 500
            time.sleep(0.02)
        return rows

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20, follow_redirects=True) as client:
        for i, wallet in enumerate(wallets, start=1):
            if wallet.lower() in seen_wallets:
                if i % 25 == 0:
                    print(f"[activity] wallet {i}/{len(wallets)} already checkpointed", flush=True)
                continue
            before = len(rows)
            wallet_rows = fetch_wallet_window(client, wallet, None, None)
            rows.extend(wallet_rows)
            with ACTIVITY_CHECKPOINT.open("a") as f:
                for row in wallet_rows:
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
            got = len(rows) - before
            if got >= 500 or i % 10 == 0:
                print(
                    f"[activity] wallet {i}/{len(wallets)} {wallet} rows={got:,} total={len(rows):,}",
                    flush=True,
                )
            elif i % 25 == 0:
                print(f"[activity] {i}/{len(wallets)} wallets, {len(rows):,} rows", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "address",
                "proxyWallet",
                "timestamp",
                "conditionId",
                "type",
                "size",
                "usdcSize",
                "transactionHash",
                "slug",
                "eventSlug",
                "title",
            ]
        )
    df.columns = [str(c) for c in df.columns]
    if not df.empty:
        df = df.drop_duplicates(subset=["address", "transactionHash", "type", "conditionId", "timestamp", "size"])
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ACTIVITY_RAW)
    return df


def topic_to_address(topic: str) -> str:
    return "0x" + topic.lower().removeprefix("0x")[-40:]


def words(data: str) -> list[str]:
    raw = data.lower().removeprefix("0x")
    if not raw:
        return []
    return [raw[i : i + 64] for i in range(0, len(raw), 64)]


def uint_from_word(word: str) -> int:
    return int(word, 16)


def read_uint_array(data_words: list[str], offset_bytes: int) -> list[int]:
    start = offset_bytes // 32
    if start >= len(data_words):
        return []
    n = uint_from_word(data_words[start])
    out: list[int] = []
    for i in range(n):
        idx = start + 1 + i
        if idx >= len(data_words):
            break
        out.append(uint_from_word(data_words[idx]))
    return out


def parse_receipt_for_wallet(receipt: dict[str, Any], wallet: str) -> dict[str, Any]:
    wallet = wallet.lower()
    token_deltas: dict[str, float] = defaultdict(float)
    cash_delta = 0.0
    ctf_transfer_logs = 0
    usdc_transfer_logs = 0

    for log in receipt.get("logs", []) or []:
        addr = str(log.get("address", "")).lower()
        topics = [str(t).lower() for t in log.get("topics", [])]
        if not topics:
            continue
        topic0 = topics[0]
        if addr == CTF and topic0 == ERC1155_TRANSFER_SINGLE_TOPIC and len(topics) >= 4:
            from_addr = topic_to_address(topics[2])
            to_addr = topic_to_address(topics[3])
            dw = words(str(log.get("data", "")))
            if len(dw) < 2:
                continue
            token_id = str(uint_from_word(dw[0]))
            amount = uint_from_word(dw[1]) / 1_000_000.0
            if from_addr == wallet:
                token_deltas[token_id] -= amount
            if to_addr == wallet:
                token_deltas[token_id] += amount
            if from_addr == wallet or to_addr == wallet:
                ctf_transfer_logs += 1
        elif addr == CTF and topic0 == ERC1155_TRANSFER_BATCH_TOPIC and len(topics) >= 4:
            from_addr = topic_to_address(topics[2])
            to_addr = topic_to_address(topics[3])
            if from_addr != wallet and to_addr != wallet:
                continue
            dw = words(str(log.get("data", "")))
            if len(dw) < 2:
                continue
            ids = read_uint_array(dw, uint_from_word(dw[0]))
            vals = read_uint_array(dw, uint_from_word(dw[1]))
            for token_id, raw_val in zip(ids, vals):
                amount = raw_val / 1_000_000.0
                if from_addr == wallet:
                    token_deltas[str(token_id)] -= amount
                if to_addr == wallet:
                    token_deltas[str(token_id)] += amount
            ctf_transfer_logs += 1
        elif addr == USDC and topic0 == ERC20_TRANSFER_TOPIC and len(topics) >= 3:
            from_addr = topic_to_address(topics[1])
            to_addr = topic_to_address(topics[2])
            if from_addr != wallet and to_addr != wallet:
                continue
            dw = words(str(log.get("data", "")))
            if not dw:
                continue
            amount = uint_from_word(dw[0]) / 1_000_000.0
            if from_addr == wallet:
                cash_delta -= amount
            if to_addr == wallet:
                cash_delta += amount
            usdc_transfer_logs += 1

    return {
        "cash_delta_usd": cash_delta,
        "token_deltas": dict(token_deltas),
        "ctf_transfer_logs": ctf_transfer_logs,
        "usdc_transfer_logs": usdc_transfer_logs,
    }


def receipt_rpc_call(client: httpx.Client, rpc_urls: list[str], tx_hash: str) -> Any:
    backoff = 2.0
    for attempt in range(1, 13):
        saw_retryable_error = False
        for rpc_url in rpc_urls:
            try:
                receipt = rpc_call(
                    client,
                    rpc_url,
                    "eth_getTransactionReceipt",
                    [tx_hash],
                    timeout_s=30,
                    max_attempts=3,
                )
                if receipt is not None:
                    return receipt
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status != 429 or attempt == 12:
                    raise
                retry_after = exc.response.headers.get("Retry-After") if exc.response is not None else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                time.sleep(min(delay, 90.0))
                saw_retryable_error = True
            except (json.JSONDecodeError, httpx.TimeoutException, httpx.TransportError):
                if attempt == 12:
                    raise
                saw_retryable_error = True
                continue
        if not saw_retryable_error:
            return None
        time.sleep(min(backoff, 60.0))
        backoff = min(backoff * 1.5, 60.0)
    raise RuntimeError(f"receipt retries exhausted for {tx_hash}")


def fetch_receipts(activity: pd.DataFrame, refresh: bool, rpc_url: str, workers: int) -> pd.DataFrame:
    if RECEIPT_DELTAS.exists() and not refresh:
        return pq.read_table(RECEIPT_DELTAS).to_pandas()
    rpc_urls = [u.strip() for u in rpc_url.split(",") if u.strip()]
    if not rpc_urls:
        rpc_urls = [DEFAULT_RPC_URL]

    con = connect()
    setup_base_tables(con)
    build_token_map(con)
    politics_conditions = {
        r[0].lower()
        for r in con.execute("SELECT DISTINCT condition_id FROM token_map WHERE condition_id IS NOT NULL").fetchall()
    }
    activity = activity.copy()
    activity["condition_id_l"] = activity.get("conditionId", "").astype(str).str.lower()
    relevant = activity[
        activity["condition_id_l"].isin(politics_conditions)
        & activity["transactionHash"].notna()
        & activity["address"].notna()
    ].copy()
    keys = (
        relevant[["transactionHash", "address"]]
        .drop_duplicates()
        .rename(columns={"transactionHash": "tx_hash"})
        .to_dict("records")
    )
    print(f"[receipts] relevant API tx-wallet pairs: {len(keys):,}", flush=True)

    out_rows: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str]] = set()
    if RECEIPT_CHECKPOINT.exists():
        with RECEIPT_CHECKPOINT.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                tx_hash = str(row.get("tx_hash", "")).lower()
                wallet = str(row.get("address", "")).lower()
                if tx_hash and wallet and row.get("receipt_found"):
                    done_keys.add((tx_hash, wallet))
                    out_rows.append(row)
        print(
            f"[receipts] loaded checkpoint rows={len(out_rows):,} tx-wallets={len(done_keys):,}",
            flush=True,
        )
    keys = [
        row
        for row in keys
        if (str(row["tx_hash"]).lower(), str(row["address"]).lower()) not in done_keys
    ]
    if done_keys:
        print(f"[receipts] remaining tx-wallet pairs: {len(keys):,}", flush=True)
    for idx, row in enumerate(keys):
        row["_rpc_url"] = rpc_urls[idx % len(rpc_urls)]
    if len(rpc_urls) > 1:
        print(f"[receipts] rotating across {len(rpc_urls)} RPC endpoints", flush=True)
    client = httpx.Client(timeout=30, follow_redirects=True)

    def one(row: dict[str, str]) -> list[dict[str, Any]]:
        tx_hash = str(row["tx_hash"]).lower()
        wallet = str(row["address"]).lower()
        primary = str(row.get("_rpc_url") or rpc_urls[0])
        ordered_rpc_urls = [primary] + [u for u in rpc_urls if u != primary]
        receipt = receipt_rpc_call(client, ordered_rpc_urls, tx_hash)
        if receipt is None:
            return [
                {
                    "tx_hash": tx_hash,
                    "address": wallet,
                    "receipt_found": False,
                    "cash_delta_usd": 0.0,
                    "outcome_token_id": None,
                    "token_delta": 0.0,
                    "ctf_transfer_logs": 0,
                    "usdc_transfer_logs": 0,
                }
            ]
        parsed = parse_receipt_for_wallet(receipt, wallet)
        base = {
            "tx_hash": tx_hash,
            "address": wallet,
            "receipt_found": True,
            "block_number": int(str(receipt.get("blockNumber", "0x0")), 16),
            "cash_delta_usd": float(parsed["cash_delta_usd"]),
            "ctf_transfer_logs": int(parsed["ctf_transfer_logs"]),
            "usdc_transfer_logs": int(parsed["usdc_transfer_logs"]),
        }
        token_deltas = parsed["token_deltas"]
        if not token_deltas:
            return [{**base, "outcome_token_id": None, "token_delta": 0.0}]
        nonzero_token_rows = [
            {**base, "outcome_token_id": token_id, "token_delta": float(delta)}
            for token_id, delta in token_deltas.items()
            if abs(delta) > 1e-12
        ]
        if not nonzero_token_rows:
            return [{**base, "outcome_token_id": None, "token_delta": 0.0}]
        return nonzero_token_rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, row) for row in keys]
        checkpoint_file = RECEIPT_CHECKPOINT.open("a")
        for i, fut in enumerate(as_completed(futures), start=1):
            rows = fut.result()
            out_rows.extend(rows)
            for row in rows:
                checkpoint_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            checkpoint_file.flush()
            if i % 500 == 0:
                print(f"[receipts] {i:,}/{len(keys):,}", flush=True)
        checkpoint_file.close()
    client.close()

    df = pd.DataFrame(out_rows)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), RECEIPT_DELTAS)
    return df


def ratio_ci(piece: pd.DataFrame, net_col: str = "net_pnl_usd_corrected") -> tuple[float, float]:
    if piece.empty:
        return math.nan, math.nan
    blocks = (
        piece.groupby("market_id", as_index=False)
        .agg(net=(net_col, "sum"), gross=("gross_usd_volume", "sum"))
        .query("gross > 0")
    )
    if blocks.empty:
        return math.nan, math.nan
    vals = blocks[["net", "gross"]].to_numpy(float)
    rng = np.random.default_rng(RNG_SEED)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(vals), len(vals))
        den = vals[idx, 1].sum()
        estimates.append(vals[idx, 0].sum() / den * 10_000 if den > 0 else math.nan)
    lo, hi = np.nanquantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_piece(piece: pd.DataFrame, label: str, net_col: str) -> dict[str, Any]:
    gross = float(piece["gross_usd_volume"].sum())
    net = float(piece[net_col].sum())
    ci_lo, ci_hi = ratio_ci(piece, net_col)
    wallets = (
        piece.groupby("address", as_index=False)
        .agg(net=(net_col, "sum"), gross=("gross_usd_volume", "sum"))
        .query("gross > 0")
    )
    wallet_bps = wallets["net"] / wallets["gross"] * 10_000 if not wallets.empty else pd.Series(dtype=float)
    return {
        "label": label,
        "wallets": int(piece["address"].nunique()),
        "markets": int(piece["market_id"].nunique()),
        "rows": int(len(piece)),
        "gross_usd_volume": gross,
        "net_pnl_usd": net,
        "net_bps": net / gross * 10_000 if gross > 0 else math.nan,
        "ci_lo_bps": ci_lo,
        "ci_hi_bps": ci_hi,
        "median_wallet_bps": float(wallet_bps.median()) if len(wallet_bps) else math.nan,
        "q25_wallet_bps": float(wallet_bps.quantile(0.25)) if len(wallet_bps) else math.nan,
        "q75_wallet_bps": float(wallet_bps.quantile(0.75)) if len(wallet_bps) else math.nan,
        "net_without_rebate_bps": float((piece[net_col].sum() - piece["maker_rebate_usd"].sum()) / gross * 10_000)
        if gross > 0
        else math.nan,
    }


def build_accounting(activity: pd.DataFrame, receipt_deltas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = connect()
    setup_base_tables(con)
    token_map = build_token_map(con)
    token_marks = build_mark_prices(con)
    con.register("token_marks_input", token_marks)
    con.execute("CREATE OR REPLACE TEMP TABLE token_marks_df AS SELECT * FROM token_marks_input")

    activity = activity.copy()
    activity["tx_hash"] = activity["transactionHash"].astype(str).str.lower()
    activity["address"] = activity["address"].astype(str).str.lower()
    tx_meta = (
        activity.groupby(["tx_hash", "address"], as_index=False)
        .agg(
            api_types=("type", lambda x: ",".join(sorted(set(map(str, x))))),
            api_condition_ids=("conditionId", lambda x: ",".join(sorted(set(str(v).lower() for v in x if pd.notna(v))))),
            api_event_slugs=("eventSlug", lambda x: ",".join(sorted(set(str(v) for v in x if pd.notna(v))))),
            api_min_timestamp=("timestamp", "min"),
            api_rows=("type", "size"),
        )
    )

    receipt_deltas = receipt_deltas.copy()
    receipt_deltas["tx_hash"] = receipt_deltas["tx_hash"].astype(str).str.lower()
    receipt_deltas["address"] = receipt_deltas["address"].astype(str).str.lower()
    receipt_deltas["outcome_token_id"] = receipt_deltas["outcome_token_id"].astype("string")
    con.register("receipt_deltas_df", receipt_deltas)
    con.execute("CREATE OR REPLACE TEMP TABLE receipt_deltas AS SELECT * FROM receipt_deltas_df")
    con.register("tx_meta_df", tx_meta)
    con.execute("CREATE OR REPLACE TEMP TABLE tx_meta AS SELECT * FROM tx_meta_df")

    # Token value adjustments are exact at the token/market level.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE token_adjustments AS
        SELECT
            rd.tx_hash,
            rd.address,
            tm.market_id,
            tm.condition_id,
            rd.outcome_token_id,
            sum(rd.token_delta) AS token_delta,
            any_value(tm.mark_price) AS mark_price,
            any_value(tm.mark_source) AS mark_source,
            sum(rd.token_delta * coalesce(tm.mark_price, 0.0)) AS token_value_delta_usd
        FROM receipt_deltas rd
        JOIN token_marks_df tm ON rd.outcome_token_id = tm.outcome_token_id
        WHERE rd.outcome_token_id IS NOT NULL
        GROUP BY 1, 2, 3, 4, 5
        """
    )
    # Allocate transaction cash only to txs that actually touched politics
    # tokens. For multi-market conversions, allocate by absolute token flow so
    # category totals are preserved without pretending the cash belongs to one
    # binary condition.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE tx_cash AS
        SELECT
            tx_hash,
            address,
            max(cash_delta_usd) AS cash_delta_usd,
            max(ctf_transfer_logs) AS ctf_transfer_logs,
            max(usdc_transfer_logs) AS usdc_transfer_logs,
            bool_or(receipt_found) AS receipt_found
        FROM receipt_deltas
        GROUP BY 1, 2
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE cash_weights AS
        WITH by_market AS (
            SELECT
                tx_hash,
                address,
                market_id,
                sum(abs(token_delta)) AS abs_token_delta
            FROM token_adjustments
            GROUP BY 1, 2, 3
        ),
        denom AS (
            SELECT tx_hash, address, sum(abs_token_delta) AS denom
            FROM by_market
            GROUP BY 1, 2
        )
        SELECT
            bm.tx_hash,
            bm.address,
            bm.market_id,
            bm.abs_token_delta / nullif(d.denom, 0) AS cash_weight
        FROM by_market bm
        JOIN denom d USING (tx_hash, address)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE cash_adjustments AS
        SELECT
            cw.tx_hash,
            cw.address,
            cw.market_id,
            tc.cash_delta_usd * cw.cash_weight AS cash_delta_alloc_usd
        FROM cash_weights cw
        JOIN tx_cash tc USING (tx_hash, address)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE adjustments_by_market AS
        WITH token_mkt AS (
            SELECT tx_hash, address, market_id, sum(token_value_delta_usd) AS token_value_delta_usd
            FROM token_adjustments
            GROUP BY 1, 2, 3
        ),
        all_keys AS (
            SELECT tx_hash, address, market_id FROM token_mkt
            UNION
            SELECT tx_hash, address, market_id FROM cash_adjustments
        )
        SELECT
            k.tx_hash,
            k.address,
            k.market_id,
            coalesce(t.token_value_delta_usd, 0.0) AS token_value_delta_usd,
            coalesce(c.cash_delta_alloc_usd, 0.0) AS cash_delta_alloc_usd,
            coalesce(t.token_value_delta_usd, 0.0) + coalesce(c.cash_delta_alloc_usd, 0.0) AS accounting_adjustment_usd
        FROM all_keys k
        LEFT JOIN token_mkt t USING (tx_hash, address, market_id)
        LEFT JOIN cash_adjustments c USING (tx_hash, address, market_id)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE market_adjustment AS
        SELECT
            address,
            market_id,
            sum(token_value_delta_usd) AS token_value_delta_usd,
            sum(cash_delta_alloc_usd) AS cash_delta_alloc_usd,
            sum(accounting_adjustment_usd) AS accounting_adjustment_usd,
            count(DISTINCT tx_hash) AS nontrade_txs
        FROM adjustments_by_market
        GROUP BY 1, 2
        """
    )

    # Correct final-token carry using fill token inventory plus on-chain
    # non-trade token deltas.
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE fill_token_final AS
        WITH actions AS (
            SELECT
                lower(rt.maker) AS address,
                CAST(rt.market_id AS VARCHAR) AS market_id,
                CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END AS outcome_token_id,
                CASE WHEN rt.maker_asset_id = '0' THEN rt.token_amount ELSE -rt.token_amount END AS token_delta
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN (SELECT DISTINCT address FROM structured_wallets WHERE structured_playbook) sw ON lower(rt.maker) = sw.address
            JOIN (SELECT DISTINCT market_id FROM politics_wm) pm ON CAST(rt.market_id AS VARCHAR) = pm.market_id
            WHERE rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND lower(rt.maker) NOT IN ({internals})
              AND lower(rt.taker) NOT IN ({internals})
            UNION ALL
            SELECT
                lower(rt.taker) AS address,
                CAST(rt.market_id AS VARCHAR) AS market_id,
                CASE WHEN rt.maker_asset_id = '0' THEN rt.taker_asset_id ELSE rt.maker_asset_id END AS outcome_token_id,
                CASE WHEN rt.maker_asset_id = '0' THEN -rt.token_amount ELSE rt.token_amount END AS token_delta
            FROM read_parquet('{TRADES_GLOB}') rt
            JOIN (SELECT DISTINCT address FROM structured_wallets WHERE structured_playbook) sw ON lower(rt.taker) = sw.address
            JOIN (SELECT DISTINCT market_id FROM politics_wm) pm ON CAST(rt.market_id AS VARCHAR) = pm.market_id
            WHERE rt.maker IS NOT NULL
              AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
              AND lower(rt.maker) NOT IN ({internals})
              AND lower(rt.taker) NOT IN ({internals})
        )
        SELECT address, market_id, outcome_token_id, sum(token_delta) AS fill_final_token_position
        FROM actions
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE nontrade_token_final AS
        SELECT
            ta.address,
            ta.market_id,
            ta.outcome_token_id,
            sum(ta.token_delta) AS nontrade_token_delta
        FROM token_adjustments ta
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE corrected_abs_final AS
        WITH keys AS (
            SELECT address, market_id, outcome_token_id FROM fill_token_final
            UNION
            SELECT address, market_id, outcome_token_id FROM nontrade_token_final
        ),
        token_final AS (
            SELECT
                k.address,
                k.market_id,
                k.outcome_token_id,
                coalesce(f.fill_final_token_position, 0.0) AS fill_final_token_position,
                coalesce(n.nontrade_token_delta, 0.0) AS nontrade_token_delta,
                coalesce(f.fill_final_token_position, 0.0) + coalesce(n.nontrade_token_delta, 0.0)
                    AS corrected_final_token_position
            FROM keys k
            LEFT JOIN fill_token_final f USING (address, market_id, outcome_token_id)
            LEFT JOIN nontrade_token_final n USING (address, market_id, outcome_token_id)
        )
        SELECT
            address,
            market_id,
            sum(abs(corrected_final_token_position)) AS corrected_abs_final_token_position,
            sum(abs(fill_final_token_position)) AS raw_rebuilt_abs_final_token_position,
            sum(abs(nontrade_token_delta)) AS abs_nontrade_token_delta
        FROM token_final
        GROUP BY 1, 2
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE corrected_wm AS
        SELECT
            wm.*,
            coalesce(ma.token_value_delta_usd, 0.0) AS token_value_delta_usd,
            coalesce(ma.cash_delta_alloc_usd, 0.0) AS cash_delta_alloc_usd,
            coalesce(ma.accounting_adjustment_usd, 0.0) AS accounting_adjustment_usd,
            coalesce(ma.nontrade_txs, 0) AS nontrade_txs,
            wm.base_pnl_usd + coalesce(ma.accounting_adjustment_usd, 0.0) AS base_pnl_usd_corrected,
            wm.net_pnl_usd + coalesce(ma.accounting_adjustment_usd, 0.0) AS net_pnl_usd_corrected,
            coalesce(caf.corrected_abs_final_token_position, wm.abs_final_token_position)
                AS corrected_abs_final_token_position,
            coalesce(caf.raw_rebuilt_abs_final_token_position, wm.abs_final_token_position)
                AS raw_rebuilt_abs_final_token_position,
            coalesce(caf.abs_nontrade_token_delta, 0.0) AS abs_nontrade_token_delta
        FROM target_wm wm
        LEFT JOIN market_adjustment ma USING (address, market_id)
        LEFT JOIN corrected_abs_final caf USING (address, market_id)
        """
    )
    corrected_wm = con.execute("SELECT * FROM corrected_wm").df()
    pq.write_table(pa.Table.from_pandas(corrected_wm, preserve_index=False), ACCOUNTING_PARQUET)

    # Re-cut the structured playbook with corrected carry.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE corrected_wallet_category AS
        SELECT
            address,
            category,
            count(DISTINCT market_id) AS markets,
            sum(gross_usd_volume) AS gross_usd_volume,
            sum(net_pnl_usd_corrected) AS net_pnl_usd_corrected,
            sum(maker_rebate_usd) AS maker_rebate_usd,
            sum(taker_fee_usd) AS taker_fee_usd,
            sum(maker_fills) AS maker_fills,
            sum(maker_usd) AS maker_usd,
            sum(gross_token_volume) AS gross_token_volume,
            sum(corrected_abs_final_token_position) AS corrected_abs_final_token_position,
            sum(spike_zone_usd) AS spike_zone_usd,
            sum(CASE WHEN distinct_outcomes_made >= 2 THEN maker_usd ELSE 0 END) AS two_sided_maker_usd
        FROM (
            SELECT
                wm.*,
                wm.net_pnl_usd + coalesce(ma.accounting_adjustment_usd, 0.0)
                    AS net_pnl_usd_corrected,
                coalesce(caf.corrected_abs_final_token_position, wm.abs_final_token_position)
                    AS corrected_abs_final_token_position
            FROM politics_wm wm
            JOIN structured_wallets sw USING (address, category)
            LEFT JOIN market_adjustment ma USING (address, market_id)
            LEFT JOIN corrected_abs_final caf USING (address, market_id)
        )
        GROUP BY 1, 2
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE corrected_structured_wallets AS
        SELECT
            *,
            two_sided_maker_usd / nullif(maker_usd, 0) AS two_sided_usd_share,
            corrected_abs_final_token_position / nullif(gross_token_volume, 0) AS corrected_carry_token_share,
            spike_zone_usd / nullif(maker_usd, 0) AS spike_zone_usd_share,
            (
                two_sided_maker_usd / nullif(maker_usd, 0) >= {STRUCT_TWO_SIDED_MIN}
                AND corrected_abs_final_token_position / nullif(gross_token_volume, 0) >= {STRUCT_CARRY_MIN}
                AND coalesce(spike_zone_usd / nullif(maker_usd, 0), 0) <= {STRUCT_SPIKE_MAX}
                AND maker_fills > 0
            ) AS corrected_structured_playbook
        FROM corrected_wallet_category
        """
    )
    corrected_structured = con.execute("SELECT * FROM corrected_structured_wallets").df()
    recut = corrected_wm.merge(
        corrected_structured[["address", "category", "corrected_structured_playbook"]],
        on=["address", "category"],
        how="left",
    )
    recut = recut[recut["corrected_structured_playbook"].fillna(False)].copy()
    pq.write_table(pa.Table.from_pandas(recut, preserve_index=False), RECUT_PARQUET)

    summaries = [
        summarize_piece(corrected_wm, "original_k5_structured_non_top3_token_pnl", "net_pnl_usd"),
        summarize_piece(corrected_wm, "original_k5_structured_non_top3_accounting_corrected", "net_pnl_usd_corrected"),
        summarize_piece(recut, "corrected_carry_recut_structured_non_top3", "net_pnl_usd_corrected"),
    ]
    summary = pd.DataFrame(summaries)

    event_audit = con.execute(
        """
        SELECT
            count(DISTINCT tx_hash) AS relevant_txs,
            sum(accounting_adjustment_usd) AS accounting_adjustment_usd,
            sum(token_value_delta_usd) AS token_value_delta_usd,
            sum(cash_delta_alloc_usd) AS cash_delta_alloc_usd
        FROM adjustments_by_market
        """
    ).df()
    receipt_audit = con.execute(
        """
        SELECT
            count(DISTINCT tx_hash) AS receipt_txs,
            count(DISTINCT CASE WHEN receipt_found THEN tx_hash END) AS receipt_found_txs,
            sum(cash_delta_usd) AS receipt_cash_delta_usd,
            sum(abs(token_delta)) AS receipt_abs_token_delta
        FROM receipt_deltas
        """
    ).df()
    event_audit = pd.concat([event_audit, receipt_audit], axis=1)

    return corrected_wm, summary, event_audit


def build_capacity_ladder(corrected_summary: pd.DataFrame) -> pd.DataFrame:
    con = connect()
    setup_base_tables(con)
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE top3 AS
        SELECT DISTINCT market_id, address
        FROM politics_wm
        WHERE global_market_maker_rank <= 3
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE raw_non_top3_flow AS
        SELECT
            sum(rt.usd_amount) AS non_top3_maker_usd_2026,
            count(DISTINCT CAST(rt.timestamp AS DATE)) AS active_days_2026
        FROM read_parquet('{TRADES_GLOB}') rt
        JOIN (SELECT DISTINCT market_id FROM politics_wm) pm ON CAST(rt.market_id AS VARCHAR) = pm.market_id
        LEFT JOIN top3 t ON CAST(rt.market_id AS VARCHAR) = t.market_id AND lower(rt.maker) = t.address
        WHERE rt.timestamp >= TIMESTAMP '2026-01-01'
          AND rt.timestamp < TIMESTAMP '2027-01-01'
          AND rt.maker IS NOT NULL
          AND rt.taker IS NOT NULL
          AND rt.maker <> rt.taker
          AND lower(rt.maker) NOT IN ({internals})
          AND lower(rt.taker) NOT IN ({internals})
          AND t.address IS NULL
        """
    )
    flow = con.execute("SELECT * FROM raw_non_top3_flow").df().iloc[0]
    non_top3 = float(flow["non_top3_maker_usd_2026"])
    active_days = int(flow["active_days_2026"])
    per_day = non_top3 / active_days if active_days > 0 else math.nan
    row = corrected_summary[corrected_summary["label"].eq("corrected_carry_recut_structured_non_top3")].iloc[0]
    mean_bps = float(row["net_bps"])
    median_bps = float(row["median_wallet_bps"])
    out = []
    for label, capture in [
        ("fixed_0.25%_non_top3_capture_2026_active_day", 0.0025),
        ("fixed_1.00%_non_top3_capture_2026_active_day", 0.01),
        ("fixed_5.00%_non_top3_capture_2026_active_day", 0.05),
    ]:
        headroom_day = per_day * capture if np.isfinite(per_day) else math.nan
        out.append(
            {
                "scenario": label,
                "capture_rate_of_non_top3": capture,
                "non_top3_maker_usd_2026": non_top3,
                "active_days_2026": active_days,
                "non_top3_maker_usd_per_active_day_2026": per_day,
                "headroom_day_usd": headroom_day,
                "ev_day_median_wallet_bps_usd": headroom_day * median_bps / 10_000
                if np.isfinite(headroom_day) and np.isfinite(median_bps)
                else math.nan,
                "ev_day_mean_bps_usd": headroom_day * mean_bps / 10_000
                if np.isfinite(headroom_day) and np.isfinite(mean_bps)
                else math.nan,
                "mean_bps_used": mean_bps,
                "median_wallet_bps_used": median_bps,
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-activity", action="store_true")
    parser.add_argument("--refresh-receipts", action="store_true")
    parser.add_argument("--rpc-url", default=os.environ.get("POLYGON_RPC_URL", DEFAULT_RPC_URL))
    parser.add_argument("--receipt-workers", type=int, default=8)
    args = parser.parse_args()

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    activity = fetch_activity(args.refresh_activity)
    receipt_deltas = fetch_receipts(activity, args.refresh_receipts, args.rpc_url, args.receipt_workers)
    corrected_wm, summary, event_audit = build_accounting(activity, receipt_deltas)
    capacity = build_capacity_ladder(summary)

    summary.to_csv(SUMMARY_CSV, index=False)
    event_audit.to_csv(EVENT_AUDIT_CSV, index=False)
    capacity.to_csv(CAPACITY_CSV, index=False)

    print("[summary]")
    print(summary.to_string(index=False))
    print("[event audit]")
    print(event_audit.to_string(index=False))
    print("[capacity]")
    print(capacity.to_string(index=False))
    print(f"wrote {ACCOUNTING_PARQUET}")
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {CAPACITY_CSV}")
    print(f"wrote {EVENT_AUDIT_CSV}")


if __name__ == "__main__":
    main()
