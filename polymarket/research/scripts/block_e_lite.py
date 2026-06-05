"""Block E Lite operator attribution and Block A competition snapshot.

Run from ``polymarket/research``:

    PYTHONPATH=. uv run python scripts/block_e_lite.py
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml

from data_infra.operator_denylist import (
    EXCHANGE_INTERNAL_LEG_V1,
    HFT,
    OPERATOR_ADDRESSES,
    PURE_MM_BOTS,
)
from scripts.dali_tfi_deep_dive import (
    DEFAULT_INPUTS,
    FamilyInput,
    assign_magnitude_bucket,
    filter_eval,
    metric_rows,
    read_candidates,
    read_eval,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
TRADES_GLOB = ROOT / "data" / "trades" / "trades_delta_shard*.parquet"
TRADES_SEED = ROOT / "data" / "trades" / "trades_seed.parquet"
BLOCK_A_CONFIG = ROOT / "configs" / "block_a0_capture.generated.yaml"

OUT_ATTRIBUTION = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_operator_category_attribution.csv"
OUT_COMPETITION = ANALYSIS / "csv_outputs" / "dali" / "block_a_market_competition_snapshot.csv"
OUT_NOTE = NOTES / "block_e_lite_findings.md"

HORIZON_SECONDS = 300
MIN_SIGNAL_USD = 25.0
MAX_FUTURE_GAP_SECONDS = 300
EXCLUDE_LAST_SECONDS = 600
MIN_BUCKET_OBS = 100
BOOTSTRAP_SAMPLES = 300
COMPETITION_LOOKBACK_DAYS = 30
FOLLOW_ON_TARGET_DATE = pd.Timestamp("2026-05-27", tz="UTC")

EXCHANGE_INTERNAL_LEG_SET = {a.lower() for a in EXCHANGE_INTERNAL_LEG_V1}
MM_BOT_SET = {a.lower() for a in PURE_MM_BOTS}
HFT_SET = {a.lower() for a in HFT}
OPERATOR_SET = {a.lower() for a in OPERATOR_ADDRESSES}

OPERATOR_CATEGORY_SETS = {
    "exchange_internal_leg": EXCHANGE_INTERNAL_LEG_SET,
    "mm_bot": MM_BOT_SET,
    "hft": HFT_SET,
}

FILTER_STATES: list[tuple[str, set[str]]] = [
    ("exchange_internal_legs_only_removed", EXCHANGE_INTERNAL_LEG_SET),
    ("mm_bots_only_removed", MM_BOT_SET),
    ("hft_only_removed", HFT_SET),
    ("all_operators_removed", OPERATOR_SET),
]

ATTRIBUTION_KEEP_COLUMNS = [
    "analysis_component",
    "family",
    "family_label",
    "operator_filter_state",
    "removed_category",
    "horizon_seconds",
    "magnitude_bucket",
    "sign_convention",
    "n_obs",
    "hit_rate_pct",
    "mean_return_cents",
    "net_edge_after_1tick_cents",
    "cached_fill_min_ts",
    "cached_fill_max_ts",
]


def fmt_ts(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt_pct(value: Any, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}%"


def fmt_num(value: Any, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    headers = [label for _, label in columns]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(key, "")) for key, _ in columns]
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def connect_trades() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(
        f"""
        CREATE VIEW raw_trades AS
            SELECT * FROM read_parquet('{TRADES_GLOB}')
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{TRADES_SEED}')
        """
    )
    return con


def collect_freshness(
    con: duckdb.DuckDBPyConnection, inputs: list[FamilyInput]
) -> tuple[pd.DataFrame, dict[str, pd.Timestamp]]:
    rows: list[dict[str, Any]] = []
    family_windows: dict[str, pd.Timestamp] = {}
    for inp in inputs:
        if not inp.fills_path.exists():
            continue
        df = con.execute(
            """
            SELECT
                min(timestamp) AS min_ts,
                max(timestamp) AS max_ts,
                count(*) AS n_rows
            FROM read_parquet(?)
            """,
            [str(inp.fills_path)],
        ).fetchdf()
        row = df.iloc[0].to_dict()
        max_ts = pd.Timestamp(row["max_ts"]).tz_localize("UTC")
        min_ts = pd.Timestamp(row["min_ts"]).tz_localize("UTC")
        family_windows[inp.family] = max_ts
        rows.append(
            {
                "source": "cached_family_fills",
                "family": inp.family,
                "family_label": inp.label,
                "path": str(inp.fills_path.relative_to(ROOT)),
                "n_rows": int(row["n_rows"]),
                "min_ts": min_ts,
                "max_ts": max_ts,
            }
        )

    raw = con.sql(
        """
        SELECT
            min(timestamp) AS min_ts,
            max(timestamp) AS max_ts
        FROM raw_trades
        """
    ).fetchdf()
    raw_min = pd.Timestamp(raw.loc[0, "min_ts"]).tz_localize("UTC")
    raw_max = pd.Timestamp(raw.loc[0, "max_ts"]).tz_localize("UTC")
    rows.append(
        {
            "source": "raw_trade_shards",
            "family": "",
            "family_label": "",
            "path": "data/trades/trades_delta_shard*.parquet + trades_seed.parquet",
            "n_rows": "",
            "min_ts": raw_min,
            "max_ts": raw_max,
        }
    )
    return pd.DataFrame(rows), {"raw_min": raw_min, "raw_max": raw_max}


def read_family_fills(inp: FamilyInput) -> pd.DataFrame:
    cols = [
        "timestamp",
        "market_id",
        "maker",
        "taker",
        "maker_side",
        "price",
        "usd_amount",
        "transaction_hash",
    ]
    fills = pd.read_parquet(inp.fills_path, columns=cols)
    fills["market_id"] = fills["market_id"].astype(str)
    fills["maker"] = fills["maker"].str.lower()
    fills["taker"] = fills["taker"].str.lower()
    fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True, errors="coerce")
    fills["price"] = pd.to_numeric(fills["price"], errors="coerce")
    fills["usd_amount"] = pd.to_numeric(fills["usd_amount"], errors="coerce")
    fills = fills[
        fills["timestamp"].notna()
        & fills["price"].notna()
        & fills["usd_amount"].gt(0)
    ].copy()
    fills["signed_piece"] = np.where(
        fills["maker_side"].eq("BUY"),
        fills["usd_amount"],
        -fills["usd_amount"],
    )
    fills["price_x_usd"] = fills["price"] * fills["usd_amount"]
    return fills


def build_eval_from_fills(
    inp: FamilyInput,
    fills: pd.DataFrame,
    *,
    exclude_addresses: set[str],
    horizons: Iterable[int],
) -> pd.DataFrame:
    working = fills
    if exclude_addresses:
        working = working[
            ~working["maker"].isin(exclude_addresses)
            & ~working["taker"].isin(exclude_addresses)
        ]
    if working.empty:
        return pd.DataFrame()

    bars = (
        working.groupby(["market_id", "timestamp"], dropna=False)
        .agg(
            n_fills=("price", "size"),
            n_txs=("transaction_hash", "nunique"),
            gross_usd=("usd_amount", "sum"),
            signed_maker_usd=("signed_piece", "sum"),
            price_x_usd=("price_x_usd", "sum"),
        )
        .reset_index()
        .rename(columns={"timestamp": "second_ts"})
    )
    bars["vwap_price"] = bars["price_x_usd"] / bars["gross_usd"]
    candidates = read_candidates(inp)[["market_id", "candidate_end_ts"]]
    bars = bars.merge(candidates, on="market_id", how="left")
    bars = bars.rename(columns={"candidate_end_ts": "end_ts"})
    bars = bars[bars["vwap_price"].between(0.01, 0.99)].sort_values(
        ["market_id", "second_ts"]
    )
    if bars.empty:
        return pd.DataFrame()

    eval_parts: list[pd.DataFrame] = []
    for horizon in horizons:
        for market_id, g in bars.groupby("market_id", sort=False):
            base = g.copy()
            base["target_ts"] = base["second_ts"] + pd.to_timedelta(
                horizon, unit="s"
            )
            future = g[["second_ts", "vwap_price"]].rename(
                columns={"second_ts": "future_ts", "vwap_price": "future_vwap_price"}
            )
            merged = pd.merge_asof(
                base.sort_values("target_ts"),
                future.sort_values("future_ts"),
                left_on="target_ts",
                right_on="future_ts",
                direction="forward",
            )
            merged["market_id"] = market_id
            merged["horizon_seconds"] = horizon
            merged["future_gap_seconds"] = (
                merged["future_ts"] - merged["target_ts"]
            ).dt.total_seconds()
            merged["future_price_change"] = (
                merged["future_vwap_price"] - merged["vwap_price"]
            )
            eval_parts.append(merged)

    if not eval_parts:
        return pd.DataFrame()
    out = pd.concat(eval_parts, ignore_index=True)
    out["family"] = inp.family
    out["family_label"] = inp.label
    out["abs_signal_usd"] = out["signed_maker_usd"].abs()
    out["seconds_to_end"] = (out["end_ts"] - out["second_ts"]).dt.total_seconds()
    meta = read_candidates(inp)
    return out.merge(meta, on="market_id", how="left")


def score_operator_attribution(attribution: pd.DataFrame) -> dict[str, str]:
    primary: dict[str, str] = {}
    if attribution.empty:
        return primary
    for label, g in attribution.groupby("family_label", dropna=False):
        base = g[g["operator_filter_state"].eq("all_fills")]
        if base.empty:
            primary[str(label)] = "not identifiable from the generated rows"
            continue
        base_hit = float(base.iloc[0]["hit_rate_pct"])
        candidates = g[g["operator_filter_state"].isin([s for s, _ in FILTER_STATES])]
        candidates = candidates[
            candidates["operator_filter_state"].ne("all_operators_removed")
        ].copy()
        if candidates.empty:
            primary[str(label)] = "not identifiable from the generated rows"
            continue
        candidates["hit_rate_delta_pct"] = candidates["hit_rate_pct"] - base_hit
        best = candidates.sort_values("hit_rate_delta_pct", ascending=False).iloc[0]
        if pd.isna(best["hit_rate_delta_pct"]) or best["hit_rate_delta_pct"] <= 0:
            primary[str(label)] = "no single positive category removal"
        else:
            primary[str(label)] = str(best["removed_category"])
    return primary


def build_operator_category_attribution(
    inputs: list[FamilyInput], freshness: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    evals = {inp.family: read_eval(inp) for inp in inputs}
    family_fresh = {
        row["family"]: row
        for row in freshness[freshness["source"].eq("cached_family_fills")].to_dict(
            "records"
        )
    }

    for inp in inputs:
        family_rows: list[dict[str, Any]] = []
        base_sub = filter_eval(
            evals[inp.family],
            exclude_last_seconds=EXCLUDE_LAST_SECONDS,
            min_signal_usd=MIN_SIGNAL_USD,
            max_future_gap_seconds=MAX_FUTURE_GAP_SECONDS,
            horizons=[HORIZON_SECONDS],
        )
        base_sub["operator_filter_state"] = "all_fills"
        base_sub["removed_category"] = "none"
        base_sub = assign_magnitude_bucket(base_sub, ["family", "horizon_seconds"])
        family_rows.extend(
            metric_rows(
                base_sub,
                group_cols=[
                    "family",
                    "family_label",
                    "operator_filter_state",
                    "removed_category",
                    "horizon_seconds",
                    "magnitude_bucket",
                ],
                component="operator_category_attribution",
                min_obs=MIN_BUCKET_OBS,
                bootstrap_samples=BOOTSTRAP_SAMPLES,
            )
        )

        fills = read_family_fills(inp)
        for state, addresses in FILTER_STATES:
            filtered_eval = build_eval_from_fills(
                inp,
                fills,
                exclude_addresses=addresses,
                horizons=[HORIZON_SECONDS],
            )
            filtered_sub = filter_eval(
                filtered_eval,
                exclude_last_seconds=EXCLUDE_LAST_SECONDS,
                min_signal_usd=MIN_SIGNAL_USD,
                max_future_gap_seconds=MAX_FUTURE_GAP_SECONDS,
                horizons=[HORIZON_SECONDS],
            )
            filtered_sub["operator_filter_state"] = state
            filtered_sub["removed_category"] = state.replace("_only_removed", "")
            filtered_sub["removed_category"] = filtered_sub[
                "removed_category"
            ].replace("all_operators_removed", "all_operators")
            filtered_sub = assign_magnitude_bucket(
                filtered_sub, ["family", "horizon_seconds"]
            )
            family_rows.extend(
                metric_rows(
                    filtered_sub,
                    group_cols=[
                        "family",
                        "family_label",
                        "operator_filter_state",
                        "removed_category",
                        "horizon_seconds",
                        "magnitude_bucket",
                    ],
                    component="operator_category_attribution",
                    min_obs=MIN_BUCKET_OBS,
                    bootstrap_samples=BOOTSTRAP_SAMPLES,
                )
            )

        out = pd.DataFrame(family_rows)
        if not out.empty:
            out = out[
                out["horizon_seconds"].eq(HORIZON_SECONDS)
                & out["magnitude_bucket"].eq("top_decile")
                & out["sign_convention"].eq("inverse_maker_side")
            ].copy()
            fresh_row = family_fresh.get(inp.family, {})
            out["cached_fill_min_ts"] = fresh_row.get("min_ts", pd.NaT)
            out["cached_fill_max_ts"] = fresh_row.get("max_ts", pd.NaT)
            rows.extend(out.to_dict("records"))

    attribution = pd.DataFrame(rows)
    if attribution.empty:
        return attribution
    state_order = {
        "all_fills": 0,
        "relayers_only_removed": 1,
        "mm_bots_only_removed": 2,
        "hft_only_removed": 3,
        "all_operators_removed": 4,
    }
    attribution["state_order"] = attribution["operator_filter_state"].map(state_order)
    attribution = attribution.sort_values(["family_label", "state_order"]).drop(
        columns=["state_order"]
    )
    return attribution[[c for c in ATTRIBUTION_KEEP_COLUMNS if c in attribution.columns]]


def load_block_a_targets() -> pd.DataFrame:
    data = yaml.safe_load(BLOCK_A_CONFIG.read_text())
    rows: list[dict[str, Any]] = []
    for market in data.get("markets", []):
        rows.append(
            {
                "market_id": str(market.get("id")),
                "condition_id": market.get("condition_id"),
                "question": market.get("question"),
                "slug": market.get("slug"),
                "family": market.get("family"),
                "category": market.get("category"),
                "end_date": market.get("end_date"),
                "config_volume": market.get("volume"),
                "config_liquidity": market.get("liquidity"),
                "config_mid": market.get("mid"),
                "config_spread": market.get("spread"),
            }
        )
    return pd.DataFrame(rows)


def operator_category(addr: str) -> str:
    addr = (addr or "").lower()
    if addr in RELAYER_SET:
        return "relayer"
    if addr in MM_BOT_SET:
        return "mm_bot"
    if addr in HFT_SET:
        return "hft"
    return "non_operator"


def top_wallets_repr(wallets: pd.DataFrame, total_side_volume: float) -> str:
    parts: list[str] = []
    for i, row in enumerate(wallets.head(5).itertuples(index=False), start=1):
        share = 100.0 * float(row.wallet_volume_usd) / total_side_volume
        parts.append(
            f"{i}:{row.addr}:{row.operator_category}:"
            f"${float(row.wallet_volume_usd):.0f}:{share:.1f}%"
        )
    return "; ".join(parts)


def build_competition_snapshot(
    con: duckdb.DuckDBPyConnection,
    targets: pd.DataFrame,
    raw_max: pd.Timestamp,
) -> pd.DataFrame:
    window_end = raw_max
    window_start = raw_max - pd.Timedelta(days=COMPETITION_LOOKBACK_DAYS)
    con.register("block_a_targets", targets[["market_id"]])
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE target_fills AS
        SELECT
            CAST(t.market_id AS VARCHAR) AS market_id,
            t.timestamp,
            lower(t.maker) AS maker,
            lower(t.taker) AS taker,
            CAST(t.usd_amount AS DOUBLE) AS usd_amount
        FROM raw_trades t
        JOIN block_a_targets b
            ON CAST(t.market_id AS VARCHAR) = b.market_id
        WHERE
            t.timestamp >= ?
            AND t.timestamp <= ?
            AND t.usd_amount > 0
        """,
        [window_start.to_pydatetime(), window_end.to_pydatetime()],
    )
    fills_summary = con.sql(
        """
        SELECT
            market_id,
            count(*) AS n_fills,
            sum(usd_amount) AS total_fill_volume_usd,
            min(timestamp) AS observed_min_ts,
            max(timestamp) AS observed_max_ts
        FROM target_fills
        GROUP BY market_id
        """
    ).fetchdf()
    wallet_volume = con.sql(
        """
        WITH sides AS (
            SELECT market_id, maker AS addr, usd_amount
            FROM target_fills
            WHERE maker IS NOT NULL
            UNION ALL
            SELECT market_id, taker AS addr, usd_amount
            FROM target_fills
            WHERE taker IS NOT NULL
        )
        SELECT
            market_id,
            addr,
            count(*) AS side_fill_count,
            sum(usd_amount) AS wallet_volume_usd
        FROM sides
        GROUP BY market_id, addr
        """
    ).fetchdf()
    if not wallet_volume.empty:
        wallet_volume["addr"] = wallet_volume["addr"].str.lower()
        wallet_volume["operator_category"] = wallet_volume["addr"].map(
            operator_category
        )

    out_rows: list[dict[str, Any]] = []
    fill_by_market = {
        str(row["market_id"]): row for row in fills_summary.to_dict("records")
    }
    for target in targets.to_dict("records"):
        market_id = str(target["market_id"])
        g = wallet_volume[wallet_volume["market_id"].eq(market_id)].copy()
        total_side = float(g["wallet_volume_usd"].sum()) if not g.empty else 0.0
        fill_row = fill_by_market.get(market_id, {})
        if total_side <= 0:
            out_rows.append(
                {
                    **target,
                    "window_start": window_start,
                    "window_end": window_end,
                    "observed_min_ts": pd.NaT,
                    "observed_max_ts": pd.NaT,
                    "n_fills": 0,
                    "total_fill_volume_usd": 0.0,
                    "total_wallet_side_volume_usd": 0.0,
                    "distinct_active_wallets": 0,
                    "known_operator_volume_share_pct": 0.0,
                    "relayer_volume_share_pct": 0.0,
                    "mm_bot_volume_share_pct": 0.0,
                    "hft_volume_share_pct": 0.0,
                    "top_non_operator_address": "",
                    "top_non_operator_volume_share_pct": 0.0,
                    "top_5_wallets": "",
                    "competition_class": "NO_RECENT_FILLS",
                }
            )
            continue

        category_volume = (
            g.groupby("operator_category", dropna=False)["wallet_volume_usd"]
            .sum()
            .to_dict()
        )
        op_volume = sum(
            float(category_volume.get(cat, 0.0))
            for cat in ("relayer", "mm_bot", "hft")
        )
        non_ops = g[~g["addr"].isin(OPERATOR_SET)].sort_values(
            "wallet_volume_usd", ascending=False
        )
        top_non = non_ops.iloc[0] if not non_ops.empty else None
        top_non_share = (
            100.0 * float(top_non["wallet_volume_usd"]) / total_side
            if top_non is not None
            else 0.0
        )
        known_op_share = 100.0 * op_volume / total_side
        if known_op_share > 40.0:
            competition_class = "OPERATOR_DOMINATED"
        elif top_non_share > 40.0:
            competition_class = "CONCENTRATED"
        else:
            competition_class = "DISTRIBUTED"
        top_wallets = g.sort_values("wallet_volume_usd", ascending=False).copy()
        out_rows.append(
            {
                **target,
                "window_start": window_start,
                "window_end": window_end,
                "observed_min_ts": fill_row.get("observed_min_ts", pd.NaT),
                "observed_max_ts": fill_row.get("observed_max_ts", pd.NaT),
                "n_fills": int(fill_row.get("n_fills", 0)),
                "total_fill_volume_usd": float(
                    fill_row.get("total_fill_volume_usd", 0.0)
                ),
                "total_wallet_side_volume_usd": total_side,
                "distinct_active_wallets": int(g["addr"].nunique()),
                "known_operator_volume_share_pct": known_op_share,
                "relayer_volume_share_pct": 100.0
                * float(category_volume.get("relayer", 0.0))
                / total_side,
                "mm_bot_volume_share_pct": 100.0
                * float(category_volume.get("mm_bot", 0.0))
                / total_side,
                "hft_volume_share_pct": 100.0
                * float(category_volume.get("hft", 0.0))
                / total_side,
                "top_non_operator_address": ""
                if top_non is None
                else str(top_non["addr"]),
                "top_non_operator_volume_share_pct": top_non_share,
                "top_5_wallets": top_wallets_repr(top_wallets, total_side),
                "competition_class": competition_class,
            }
        )
    return pd.DataFrame(out_rows)


def collect_recent_operator_activity(
    con: duckdb.DuckDBPyConnection,
    raw_max: pd.Timestamp,
) -> pd.DataFrame:
    window_start = raw_max - pd.Timedelta(days=COMPETITION_LOOKBACK_DAYS)
    operator_df = pd.DataFrame({"addr": sorted(OPERATOR_SET)})
    con.register("operator_addrs", operator_df)
    return con.execute(
        """
        WITH sides AS (
            SELECT lower(maker) AS addr, timestamp, market_id, usd_amount
            FROM raw_trades
            WHERE timestamp >= ? AND lower(maker) IN (SELECT addr FROM operator_addrs)
            UNION ALL
            SELECT lower(taker) AS addr, timestamp, market_id, usd_amount
            FROM raw_trades
            WHERE timestamp >= ? AND lower(taker) IN (SELECT addr FROM operator_addrs)
        )
        SELECT
            addr,
            count(*) AS side_fill_count,
            count(DISTINCT market_id) AS distinct_markets,
            sum(usd_amount) AS side_volume_usd,
            max(timestamp) AS latest_seen_ts
        FROM sides
        GROUP BY addr
        """,
        [window_start.to_pydatetime(), window_start.to_pydatetime()],
    ).fetchdf()


def note_freshness(freshness: pd.DataFrame) -> str:
    rows: list[dict[str, Any]] = []
    for row in freshness.to_dict("records"):
        rows.append(
            {
                "source": row["source"],
                "family": row["family_label"] or row["family"],
                "max_ts": fmt_ts(row["max_ts"]),
                "n_rows": row["n_rows"],
                "path": row["path"],
            }
        )
    return md_table(
        rows,
        [
            ("source", "Source"),
            ("family", "Family"),
            ("max_ts", "Max Timestamp"),
            ("n_rows", "Rows"),
            ("path", "Path"),
        ],
    )


def note_attribution(
    attribution: pd.DataFrame, primary: dict[str, str]
) -> tuple[str, str]:
    rows: list[dict[str, Any]] = []
    for row in attribution.to_dict("records"):
        rows.append(
            {
                "family": row["family_label"],
                "state": row["operator_filter_state"],
                "n_obs": row["n_obs"],
                "hit": fmt_pct(row["hit_rate_pct"]),
                "mean": fmt_num(row["mean_return_cents"]),
                "net1": fmt_num(row["net_edge_after_1tick_cents"]),
            }
        )
    table = md_table(
        rows,
        [
            ("family", "Family"),
            ("state", "Filter State"),
            ("n_obs", "n_obs"),
            ("hit", "Hit Rate"),
            ("mean", "Mean Return (c)"),
            ("net1", "Net After 1 Tick (c)"),
        ],
    )
    sentences = []
    for family in sorted(primary):
        sentences.append(
            f"- The Block B operator effect on {family} is driven primarily by "
            f"{primary[family]} removal."
        )
    return table, "\n".join(sentences)


def note_operator_mapping(recent_activity: pd.DataFrame) -> str:
    activity_by_addr = {
        row["addr"]: row for row in recent_activity.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for category, addrs in [
        ("relayer", RELAYER_SET),
        ("mm_bot", MM_BOT_SET),
        ("hft", HFT_SET),
    ]:
        for addr in sorted(addrs):
            row = activity_by_addr.get(addr, {})
            if category == "relayer":
                id_type = "exchange/relayer address"
                rtds_handling = "post-hoc raw maker/taker"
            else:
                id_type = "trader/proxy wallet"
                rtds_handling = "direct proxyWallet lookup"
            rows.append(
                {
                    "category": category,
                    "address": addr,
                    "id_type": id_type,
                    "rtds_handling": rtds_handling,
                    "seen": "yes" if row else "no",
                    "latest": fmt_ts(row.get("latest_seen_ts")),
                    "markets": int(row.get("distinct_markets", 0) or 0),
                }
            )
    table = md_table(
        rows,
        [
            ("category", "Category"),
            ("address", "Address"),
            ("id_type", "Identifier Type"),
            ("rtds_handling", "RTDS Handling"),
            ("seen", "Seen Last 30d"),
            ("latest", "Latest Seen"),
            ("markets", "Distinct Markets"),
        ],
    )
    return table


def note_competition(competition: pd.DataFrame) -> tuple[str, str]:
    rows: list[dict[str, Any]] = []
    for row in competition.sort_values(
        ["competition_class", "known_operator_volume_share_pct"],
        ascending=[True, False],
    ).to_dict("records"):
        rows.append(
            {
                "market": str(row["market_id"]),
                "family": row["family"],
                "class": row["competition_class"],
                "op": fmt_pct(row["known_operator_volume_share_pct"]),
                "rel": fmt_pct(row["relayer_volume_share_pct"]),
                "mm": fmt_pct(row["mm_bot_volume_share_pct"]),
                "hft": fmt_pct(row["hft_volume_share_pct"]),
                "top_non": fmt_pct(row["top_non_operator_volume_share_pct"]),
                "wallets": row["distinct_active_wallets"],
            }
        )
    table = md_table(
        rows,
        [
            ("market", "Market"),
            ("family", "Family"),
            ("class", "Class"),
            ("op", "Known Op Share"),
            ("rel", "Relayer"),
            ("mm", "MM Bot"),
            ("hft", "HFT"),
            ("top_non", "Top Non-Op"),
            ("wallets", "Wallets"),
        ],
    )
    counts = competition["competition_class"].value_counts().to_dict()
    op_markets = competition[
        competition["competition_class"].eq("OPERATOR_DOMINATED")
    ]["market_id"].astype(str).tolist()
    concentrated = competition[
        competition["competition_class"].eq("CONCENTRATED")
    ]["market_id"].astype(str).tolist()
    distributed = competition[
        competition["competition_class"].eq("DISTRIBUTED")
    ]["market_id"].astype(str).tolist()
    interpretation = (
        f"Classification counts: {json.dumps(counts, sort_keys=True)}. "
        f"Operator-removal effects should be most visible in "
        f"{', '.join(op_markets) if op_markets else 'no current Block A target markets'}; "
        f"{', '.join(concentrated) if concentrated else 'no markets'} look more like "
        "single-wallet concentration; "
        f"{', '.join(distributed) if distributed else 'no markets'} are the better "
        "candidates for organic-flow interpretation."
    )
    return table, interpretation


def write_note(
    *,
    freshness: pd.DataFrame,
    attribution: pd.DataFrame,
    primary: dict[str, str],
    competition: pd.DataFrame,
    recent_operator_activity: pd.DataFrame,
    raw_window: dict[str, pd.Timestamp],
) -> None:
    attribution_table, attribution_sentences = note_attribution(attribution, primary)
    competition_table, competition_interpretation = note_competition(competition)
    raw_max = raw_window["raw_max"]
    cached_max = freshness[
        freshness["source"].eq("cached_family_fills")
    ]["max_ts"].max()
    follow_on_ready = bool(
        raw_max >= FOLLOW_ON_TARGET_DATE and cached_max >= FOLLOW_ON_TARGET_DATE
    )
    if follow_on_ready:
        follow_on_status = (
            "Eligible: raw shards and cached family fills both reach "
            f"{fmt_ts(FOLLOW_ON_TARGET_DATE)}."
        )
    else:
        follow_on_status = (
            "Skipped: raw shards max at "
            f"{fmt_ts(raw_max)} and cached family fills max at {fmt_ts(cached_max)}, "
            "so the fresh-through-2026-05-27 gate is not met."
        )

    note = f"""# Block E Lite Findings

Generated: 2026-05-27

Purpose: attribute the Block B operator-removal effect by hardcoded operator
category, document live identifier semantics for Block A, and snapshot
competition in the current Block A target markets.

## Step 0: Freshness Check

{note_freshness(freshness)}

Task 1 uses the cached Dali family fills and eval artifacts above. Task 3 uses
raw trade shards over `{fmt_ts(raw_max - pd.Timedelta(days=COMPETITION_LOOKBACK_DAYS))}`
through `{fmt_ts(raw_max)}`.

Conditional follow-ons: **{follow_on_status}**

## Section 1: Block B Operator Effect Attribution

Output CSV:
`data/analysis/csv_outputs/dali/dali_tfi_operator_category_attribution.csv`

Scope: 300s horizon, top-decile absolute TFI, inverse-maker-side convention,
`min_signal_usd=25`, `max_future_gap_seconds=300`, `exclude_last_seconds=600`.

{attribution_table}

{attribution_sentences}

## Section 2: Live Operator Detection Mapping

### RTDS proxyWallet Semantics

`polymarket/execution/watcher/leader_watcher.py` consumes RTDS
`topic=activity`, `type=trades` messages and compares
`payload.proxyWallet.lower()` directly against the configured leader address.
The local RTDS probe notes confirm that CLOB market WebSocket trade-like
messages do not include maker/taker/proxyWallet, while RTDS activity trades do
include `proxyWallet`, `conditionId`, `asset`, `side`, `size`, `price`,
`timestamp`, and `transactionHash`.

### Mapping Operator Addresses To RTDS Identifiers

The MM bot and HFT entries are trader/proxy wallet identities and can be matched
directly against `payload.proxyWallet.lower()`. The two relayer entries are
exchange/relayer addresses from the raw maker/taker fields, so they should not
be assumed to appear as RTDS `proxyWallet` values. Relayer-category filtering
therefore needs post-hoc raw-fill maker/taker checks, or a companion feed that
preserves fill-level maker/taker addresses; there is no one-to-one proxyWallet
mapping for those relayer addresses.

{note_operator_mapping(recent_operator_activity)}

### Block A Operator Filtering Procedure

Block A0's current CLOB market-channel capture contains market, asset, book,
price-change, best-bid-ask, and last-trade-price state, but not wallet
identity. MM bot and HFT category filtering can be applied live only with an
RTDS `activity/trades` companion stream that carries `proxyWallet`. The relayer
category, which is the category driving the Block B attribution here, cannot be
recovered from CLOB-only capture or RTDS `proxyWallet` alone; it needs post-hoc
raw-fill maker/taker checks or another companion feed that preserves maker/taker
addresses.

### Live Operator Discovery Procedure

For wallets new since 2026-04-24, keep Block A capture running as planned and
flag candidates post-hoc from raw fills in the same market/time windows:
high side-volume share, near-balanced maker/taker role, many counterparties,
and sub-second clustering. Treat those candidates as analysis labels until a
separate denylist-refresh decision is made.

## Section 3: Per-Market Competition Snapshot

Output CSV:
`data/analysis/csv_outputs/dali/block_a_market_competition_snapshot.csv`

{competition_table}

{competition_interpretation}

## Step 2: Conditional Follow-Ons

- Re-run `scripts/validation/02_operator_detection.py` on extended data:
  **{follow_on_status}**
- Extend Block B walk-forward with 2026-04-25 through 2026-05-27:
  **{follow_on_status}**

No operator denylist entries were changed by this run.
"""
    OUT_NOTE.write_text(note, encoding="utf-8")


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    inputs = [inp for inp in DEFAULT_INPUTS if inp.eval_path.exists()]
    if not inputs:
        raise SystemExit("No Dali eval inputs found.")

    con = connect_trades()
    freshness, raw_window = collect_freshness(con, inputs)

    attribution = build_operator_category_attribution(inputs, freshness)
    attribution.to_csv(OUT_ATTRIBUTION, index=False)
    primary = score_operator_attribution(attribution)

    targets = load_block_a_targets()
    competition = build_competition_snapshot(con, targets, raw_window["raw_max"])
    competition.to_csv(OUT_COMPETITION, index=False)

    recent_operator_activity = collect_recent_operator_activity(
        con, raw_window["raw_max"]
    )
    write_note(
        freshness=freshness,
        attribution=attribution,
        primary=primary,
        competition=competition,
        recent_operator_activity=recent_operator_activity,
        raw_window=raw_window,
    )

    print(f"wrote {OUT_ATTRIBUTION.relative_to(ROOT)}")
    print(f"wrote {OUT_COMPETITION.relative_to(ROOT)}")
    print(f"wrote {OUT_NOTE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
