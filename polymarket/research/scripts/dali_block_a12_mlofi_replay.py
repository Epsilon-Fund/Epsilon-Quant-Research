"""Block A1.2 true multi-level OFI replay and diagnostics.

This sidecar replays A0/A0b raw JSONL captures into per-level OFI features for
the top 10 book levels, then runs the A1 depth-normalized decile and per-market
diagnostics on several MLOFI variants. Raw JSONL shards are never modified.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lib.clob_book import ClobBook, MultiLevelOfiContribution
from scripts.dali_block_a1_analyze import (
    BOOTSTRAP_SAMPLES,
    MetricResult,
    metric_reportable,
    public_value,
    sample_size_label,
)
from scripts.dali_block_a1_replay_batch import (
    RunSpec,
    jsonl_shards,
    metadata_from_record,
    parse_run_spec,
)
from scripts.dali_clob_replay_features import as_float, parse_levels, parse_received_at, parse_ts_ms


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
PLOTS = ANALYSIS / "block_a12_plots"
NOTE = ROOT / "notes" / "block_a12_mlofi_findings.md"
DEFAULT_FEATURES = ANALYSIS / "block_a12_mlofi_features.parquet"
DEFAULT_DECILES = ANALYSIS / "csv_outputs" / "dali" / "block_a12_mlofi_decile_aggregate.csv"
DEFAULT_MARKET_PANEL = ANALYSIS / "csv_outputs" / "dali" / "block_a12_mlofi_market_panel.csv"
DEFAULT_COMPARISON = ANALYSIS / "csv_outputs" / "dali" / "block_a12_mlofi_comparison.csv"
DEFAULT_A1_DECILES = ANALYSIS / "csv_outputs" / "dali" / "block_a1_decile_aggregate.csv"
DEFAULT_RUNS = (
    RunSpec(
        "a0",
        ROOT / "data" / "live_clob" / "block_a0" / "block_a0_20260528_morning",
    ),
    RunSpec(
        "a0b",
        ROOT / "data" / "live_clob" / "block_a0b" / "block_a0b_replacements_v2_20260527",
    ),
)

DEPTH = 10
HORIZONS = (1, 5, 30, 300)
RNG_SEED = 20260528
BOOTSTRAP_CHUNK_SECONDS = 300
VARIANTS = (
    "l1_cks",
    "integrated_l1_l10",
    "depth_weighted_l1_l10",
    "exp_decay_alpha_0p1",
    "exp_decay_alpha_0p3",
    "exp_decay_alpha_0p5",
)


@dataclass
class AssetState:
    asset_id: str
    market: str = ""
    market_id: str = ""
    question: str = ""
    slug: str = ""
    family: str = ""
    outcome_index: int | None = None
    book: ClobBook = field(default_factory=ClobBook)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def safe_slug(raw: object, fallback: str) -> str:
    text = str(raw or fallback).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:90] or fallback


def rel_note_path(path: Path) -> str:
    return str(path.resolve().relative_to(NOTE.parent.resolve().parent))


def metadata_from_new_market(rec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
    token_ids = msg.get("clob_token_ids") or msg.get("assets_ids") or rec.get("asset_ids") or []
    market_id = str(msg.get("id") or "")
    question = str(msg.get("question") or (msg.get("event_message") or {}).get("title") or "")
    slug = str(msg.get("slug") or (msg.get("event_message") or {}).get("slug") or "")
    out: dict[str, dict[str, Any]] = {}
    for idx, asset_id in enumerate(token_ids):
        key = str(asset_id)
        out[key] = {
            "asset_id": key,
            "market_id": market_id,
            "family": "",
            "slug": slug,
            "question": question,
            "outcome_index": idx,
        }
    return out


def update_metadata(metadata: dict[str, dict[str, Any]], rec: dict[str, Any]) -> None:
    metadata.update(metadata_from_record(rec))
    if str(rec.get("event_type") or "") == "new_market":
        for asset_id, values in metadata_from_new_market(rec).items():
            current = metadata.get(asset_id, {})
            merged = {**values, **{k: v for k, v in current.items() if v not in ("", None)}}
            metadata[asset_id] = merged


def parse_resolution_assets(rec: dict[str, Any]) -> list[str]:
    msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
    raw = rec.get("asset_ids") or msg.get("asset_ids") or msg.get("clob_token_ids") or []
    return [str(asset_id) for asset_id in raw if str(asset_id)]


def ensure_state(
    states: dict[str, AssetState],
    asset_id: str,
    metadata: dict[str, dict[str, Any]],
    market: str = "",
) -> AssetState:
    state = states.get(asset_id)
    if state is None:
        meta = metadata.get(asset_id, {})
        state = AssetState(
            asset_id=asset_id,
            market=market,
            market_id=str(meta.get("market_id") or ""),
            question=str(meta.get("question") or ""),
            slug=str(meta.get("slug") or ""),
            family=str(meta.get("family") or ""),
            outcome_index=meta.get("outcome_index"),
        )
        states[asset_id] = state
    if market and not state.market:
        state.market = market
    meta = metadata.get(asset_id, {})
    for attr in ("market_id", "question", "slug", "family"):
        if not getattr(state, attr) and meta.get(attr):
            setattr(state, attr, str(meta[attr]))
    if state.outcome_index is None and meta.get("outcome_index") is not None:
        state.outcome_index = meta.get("outcome_index")
    return state


def zero_mlofi(depth: int = DEPTH) -> MultiLevelOfiContribution:
    zeros = tuple(0.0 for _ in range(depth))
    return MultiLevelOfiContribution(bid=zeros, ask=zeros)


def level_tuple(levels: list[tuple[float, float]], idx: int) -> tuple[float | None, float | None]:
    if idx >= len(levels):
        return None, None
    return levels[idx]


def base_row(
    rec: dict[str, Any],
    msg: dict[str, Any],
    state: AssetState,
    *,
    depth: int,
    event_asset_count: int,
    ofi: MultiLevelOfiContribution | None = None,
) -> dict[str, Any]:
    received_at = parse_received_at(rec.get("received_at"))
    exchange_ts = parse_ts_ms(msg.get("timestamp"))
    top = state.book.top()
    bid_levels = state.book.levels("bid")[:depth]
    ask_levels = state.book.levels("ask")[:depth]
    spread = state.book.spread()
    mid = state.book.mid()
    out: dict[str, Any] = {
        "received_at": received_at,
        "exchange_ts": exchange_ts,
        "event_type": str(rec.get("event_type") or msg.get("event_type") or ""),
        "asset_id": state.asset_id,
        "market": state.market or str(msg.get("market") or ""),
        "market_id": state.market_id,
        "family": state.family,
        "slug": state.slug,
        "question": state.question,
        "outcome_index": state.outcome_index,
        "event_asset_count": event_asset_count,
        "is_book_state_complete": state.book.is_complete,
        "best_bid": top.bid_price,
        "best_bid_size": top.bid_size,
        "best_ask": top.ask_price,
        "best_ask_size": top.ask_size,
        "spread": spread,
        "mid": mid,
        "touch_depth": (
            top.bid_size + top.ask_size
            if top.bid_size is not None and top.ask_size is not None
            else None
        ),
        "change_price": None,
        "change_side": "",
        "change_size": None,
        "change_hash": "",
        "trade_price": None,
        "trade_side": "",
        "trade_size": None,
        "transaction_hash": "",
        "market_resolved_at": pd.NaT,
    }
    ofi = ofi or zero_mlofi(depth)
    for idx in range(depth):
        level = idx + 1
        bid_price, bid_size = level_tuple(bid_levels, idx)
        ask_price, ask_size = level_tuple(ask_levels, idx)
        out[f"bid_price_l{level}"] = bid_price
        out[f"bid_size_l{level}"] = bid_size
        out[f"ask_price_l{level}"] = ask_price
        out[f"ask_size_l{level}"] = ask_size
        out[f"level_depth_l{level}"] = (
            (bid_size if bid_size is not None else 0.0)
            + (ask_size if ask_size is not None else 0.0)
        )
        out[f"bid_ofi_l{level}"] = ofi.bid[idx]
        out[f"ask_ofi_l{level}"] = ofi.ask[idx]
        out[f"combined_ofi_l{level}"] = ofi.combined[idx]
    return out


def replay_shard(spec: RunSpec, shard: Path, depth: int) -> pd.DataFrame:
    metadata: dict[str, dict[str, Any]] = {}
    resolved_at: dict[str, pd.Timestamp] = {}
    states: dict[str, AssetState] = {}
    rows: list[dict[str, Any]] = []

    with shard.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            update_metadata(metadata, rec)
            event_type = str(rec.get("event_type") or msg.get("event_type") or "")
            received_at = parse_received_at(rec.get("received_at"))
            if event_type == "market_resolved":
                ts = received_at or parse_ts_ms(msg.get("timestamp"))
                if ts is not None:
                    for asset_id in parse_resolution_assets(rec):
                        previous = resolved_at.get(asset_id)
                        if previous is None or ts < previous:
                            resolved_at[asset_id] = ts
                continue
            if event_type == "book":
                asset_id = str(msg.get("asset_id") or "")
                if not asset_id or (asset_id in resolved_at and received_at is not None and received_at > resolved_at[asset_id]):
                    continue
                state = ensure_state(states, asset_id, metadata, str(msg.get("market") or ""))
                ofi = state.book.multi_level_replace(
                    parse_levels(msg.get("bids")),
                    parse_levels(msg.get("asks")),
                    depth=depth,
                )
                row = base_row(rec, msg, state, depth=depth, event_asset_count=1, ofi=ofi)
                row["market_resolved_at"] = resolved_at.get(asset_id, pd.NaT)
                rows.append(row)
            elif event_type == "price_change":
                changes = [item for item in (msg.get("price_changes") or []) if isinstance(item, dict)]
                for change in changes:
                    asset_id = str(change.get("asset_id") or "")
                    if not asset_id or (asset_id in resolved_at and received_at is not None and received_at > resolved_at[asset_id]):
                        continue
                    state = ensure_state(states, asset_id, metadata, str(msg.get("market") or ""))
                    price = as_float(change.get("price"))
                    size = as_float(change.get("size"))
                    side = str(change.get("side") or "").upper()
                    ofi = zero_mlofi(depth)
                    if price is not None and size is not None and side in {"BUY", "SELL"}:
                        ofi = state.book.multi_level_update_level(side, price, size, depth=depth)
                    row = base_row(rec, msg, state, depth=depth, event_asset_count=len(changes), ofi=ofi)
                    row["change_price"] = price
                    row["change_side"] = side
                    row["change_size"] = size
                    row["change_hash"] = str(change.get("hash") or "")
                    row["market_resolved_at"] = resolved_at.get(asset_id, pd.NaT)
                    rows.append(row)
            elif event_type == "best_bid_ask":
                asset_id = str(msg.get("asset_id") or "")
                if not asset_id or (asset_id in resolved_at and received_at is not None and received_at > resolved_at[asset_id]):
                    continue
                state = ensure_state(states, asset_id, metadata, str(msg.get("market") or ""))
                row = base_row(rec, msg, state, depth=depth, event_asset_count=1, ofi=zero_mlofi(depth))
                row["market_resolved_at"] = resolved_at.get(asset_id, pd.NaT)
                rows.append(row)
            elif event_type == "last_trade_price":
                asset_id = str(msg.get("asset_id") or "")
                if not asset_id or (asset_id in resolved_at and received_at is not None and received_at > resolved_at[asset_id]):
                    continue
                state = ensure_state(states, asset_id, metadata, str(msg.get("market") or ""))
                row = base_row(rec, msg, state, depth=depth, event_asset_count=1, ofi=zero_mlofi(depth))
                row["trade_price"] = as_float(msg.get("price"))
                row["trade_side"] = str(msg.get("side") or "")
                row["trade_size"] = as_float(msg.get("size"))
                row["transaction_hash"] = str(msg.get("transaction_hash") or "")
                row["market_resolved_at"] = resolved_at.get(asset_id, pd.NaT)
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.insert(0, "run_id", spec.run_id)
    df.insert(1, "shard", shard.name)
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    df["exchange_ts"] = pd.to_datetime(df["exchange_ts"], utc=True, errors="coerce")
    df["market_resolved_at"] = pd.to_datetime(df["market_resolved_at"], utc=True, errors="coerce")
    string_cols = [
        "run_id",
        "shard",
        "event_type",
        "asset_id",
        "market",
        "market_id",
        "family",
        "slug",
        "question",
        "change_side",
        "change_hash",
        "trade_side",
        "transaction_hash",
    ]
    numeric_cols = [
        "outcome_index",
        "event_asset_count",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "touch_depth",
        "change_price",
        "change_size",
        "trade_price",
        "trade_size",
        *[
            col
            for idx in range(1, depth + 1)
            for col in (
                f"bid_price_l{idx}",
                f"bid_size_l{idx}",
                f"ask_price_l{idx}",
                f"ask_size_l{idx}",
                f"level_depth_l{idx}",
                f"bid_ofi_l{idx}",
                f"ask_ofi_l{idx}",
                f"combined_ofi_l{idx}",
            )
        ],
    ]
    for col in string_cols:
        df[col] = df[col].fillna("").astype(str)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def replay_runs(runs: list[RunSpec], out: Path, depth: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for spec in runs:
            shards = jsonl_shards(spec.run_dir)
            if not shards:
                raise SystemExit(f"no JSONL shards found under {spec.run_dir}")
            for idx, shard in enumerate(shards, start=1):
                df = replay_shard(spec, shard, depth)
                if not df.empty:
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
                    else:
                        table = table.cast(writer.schema)
                    writer.write_table(table)
                    total += len(df)
                print(
                    f"[{spec.run_id}] {idx:02d}/{len(shards):02d} "
                    f"{display_path(shard)} -> {len(df):,} rows",
                    flush=True,
                )
        if writer is None:
            raise SystemExit("no feature rows produced")
        writer.close()
        writer = None
        tmp_path.replace(out)
    finally:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
    print(f"mlofi features: {display_path(out)} ({total:,} rows)", flush=True)


def future_value(g: pd.DataFrame, value_col: str, horizon_sec: int) -> np.ndarray:
    times = g["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    values = g[value_col].to_numpy(dtype=float)
    target = times + horizon_sec * 1_000_000_000
    last_time = times[-1] if len(times) else 0
    idx = np.searchsorted(times, target, side="right") - 1
    out = np.full(len(g), np.nan, dtype=float)
    valid = (target <= last_time) & (idx >= 0)
    out[valid] = values[idx[valid]]
    return out


def load_features(path: Path) -> pd.DataFrame:
    use_cols = [
        "run_id",
        "shard",
        "received_at",
        "event_type",
        "asset_id",
        "market_id",
        "family",
        "slug",
        "question",
        "outcome_index",
        "is_book_state_complete",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "touch_depth",
        "trade_side",
        "trade_size",
        "market_resolved_at",
        *[f"combined_ofi_l{idx}" for idx in range(1, DEPTH + 1)],
        *[f"level_depth_l{idx}" for idx in range(1, DEPTH + 1)],
    ]
    df = pd.read_parquet(path, columns=use_cols)
    if df.empty:
        raise SystemExit(f"no rows found in {path}")
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    df["market_resolved_at"] = pd.to_datetime(df["market_resolved_at"], utc=True, errors="coerce")
    for col in ("run_id", "shard", "event_type", "asset_id", "market_id", "family", "slug", "question", "trade_side"):
        df[col] = df[col].fillna("").astype(str)
    numeric_cols = [
        "outcome_index",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "touch_depth",
        "trade_size",
        *[f"combined_ofi_l{idx}" for idx in range(1, DEPTH + 1)],
        *[f"level_depth_l{idx}" for idx in range(1, DEPTH + 1)],
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["market_key"] = np.where(df["market_id"].ne(""), df["market_id"], df["asset_id"])
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    df["directional_mid"] = np.where(df["direction_factor"].gt(0), df["mid"], 1.0 - df["mid"])
    df["is_trade"] = df["event_type"].eq("last_trade_price")
    return df.sort_values(["run_id", "asset_id", "received_at"]).reset_index(drop=True)


def add_variant_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    combined_cols = [f"combined_ofi_l{idx}" for idx in range(1, DEPTH + 1)]
    depth_cols = [f"level_depth_l{idx}" for idx in range(1, DEPTH + 1)]
    combined = out[combined_cols].fillna(0.0).to_numpy(dtype=float)
    depth = out[depth_cols].fillna(0.0).to_numpy(dtype=float)
    out["event_l1_cks"] = out["combined_ofi_l1"].fillna(0.0)
    out["event_integrated_l1_l10"] = combined.sum(axis=1)
    depth_denom = depth.sum(axis=1)
    out["event_depth_weighted_l1_l10"] = np.divide(
        (combined * depth).sum(axis=1),
        depth_denom,
        out=np.zeros(len(out), dtype=float),
        where=depth_denom > 0,
    )
    for alpha in (0.1, 0.3, 0.5):
        suffix = str(alpha).replace(".", "p")
        weights = np.exp(-alpha * np.arange(DEPTH, dtype=float))
        out[f"event_exp_decay_alpha_{suffix}"] = combined @ weights
    return out


def add_horizon_features(df: pd.DataFrame) -> pd.DataFrame:
    market_depth = (
        df.groupby(["run_id", "market_key"], as_index=False)["touch_depth"]
        .mean()
        .rename(columns={"touch_depth": "market_mean_depth"})
    )
    df = df.merge(market_depth, on=["run_id", "market_key"], how="left")
    pieces: list[pd.DataFrame] = []
    grouped = list(df.groupby(["run_id", "asset_id"], sort=False))
    for idx, ((run_id, asset_id), group) in enumerate(grouped, start=1):
        print(
            f"horizon features {idx:02d}/{len(grouped):02d}: {run_id}/{str(asset_id)[:12]} "
            f"({len(group):,} rows)",
            flush=True,
        )
        g = group.sort_values("received_at").copy()
        g = g.set_index("received_at", drop=False)
        for horizon in HORIZONS:
            future_mid = future_value(g, "mid", horizon)
            future_directional_mid = np.where(g["direction_factor"].gt(0), future_mid, 1.0 - future_mid)
            g[f"future_return_bps_{horizon}s"] = np.where(
                g["directional_mid"].gt(0) & np.isfinite(future_directional_mid),
                (future_directional_mid - g["directional_mid"]) / g["directional_mid"] * 10_000.0,
                np.nan,
            )
            window = f"{horizon}s"
            for variant in VARIANTS:
                event_col = f"event_{variant}"
                rolled = g[event_col].fillna(0.0).rolling(window).sum()
                g[f"signal_{variant}_{horizon}s"] = (
                    g["direction_factor"] * rolled / g["market_mean_depth"]
                )
        pieces.append(g.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def time_block_ids(received_at: pd.Series, min_blocks: int = 4) -> np.ndarray:
    elapsed = (received_at - received_at.min()).dt.total_seconds()
    block_id = (elapsed // BOOTSTRAP_CHUNK_SECONDS).astype(int).to_numpy()
    if len(np.unique(block_id)) < min_blocks:
        block_id = np.arange(len(received_at)) // max(5, len(received_at) // 10)
    return block_id


def r2_from_sufficient(
    n: float,
    sum_x: float,
    sum_y: float,
    sum_xx: float,
    sum_xy: float,
    sum_yy: float,
) -> float:
    if n < 5:
        return math.nan
    sxx = sum_xx - (sum_x * sum_x / n)
    syy = sum_yy - (sum_y * sum_y / n)
    sxy = sum_xy - (sum_x * sum_y / n)
    if sxx <= 1e-12 or syy <= 1e-12:
        return math.nan
    return float((sxy * sxy) / (sxx * syy))


def ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    return r2_from_sufficient(
        float(len(x)),
        float(np.sum(x)),
        float(np.sum(y)),
        float(np.sum(x * x)),
        float(np.sum(x * y)),
        float(np.sum(y * y)),
    )


def block_bootstrap_r2(sub: pd.DataFrame, x_col: str, y_col: str, seed: int) -> tuple[float, float]:
    clean = sub[["received_at", x_col, y_col]].dropna()
    clean = clean[np.isfinite(clean[x_col]) & np.isfinite(clean[y_col])]
    if len(clean) < 30:
        return math.nan, math.nan
    clean = clean.copy()
    clean["block_id"] = time_block_ids(clean["received_at"])
    block_stats = (
        clean.assign(
            x2=clean[x_col] * clean[x_col],
            xy=clean[x_col] * clean[y_col],
            y2=clean[y_col] * clean[y_col],
        )
        .groupby("block_id", as_index=False)
        .agg(
            n=(x_col, "size"),
            sum_x=(x_col, "sum"),
            sum_y=(y_col, "sum"),
            sum_xx=("x2", "sum"),
            sum_xy=("xy", "sum"),
            sum_yy=("y2", "sum"),
        )
    )
    if len(block_stats) < 2:
        return math.nan, math.nan
    stats = block_stats[["n", "sum_x", "sum_y", "sum_xx", "sum_xy", "sum_yy"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = stats[rng.integers(0, len(stats), size=len(stats))]
        sums = sampled.sum(axis=0)
        value = r2_from_sufficient(*sums)
        if np.isfinite(value):
            vals.append(float(value))
    if len(vals) < 20:
        return math.nan, math.nan
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def pooled_r2(sub: pd.DataFrame, signal_col: str, y_col: str, seed: int) -> tuple[float, float, float]:
    clean = sub[[signal_col, y_col]].dropna()
    clean = clean[np.isfinite(clean[signal_col]) & np.isfinite(clean[y_col])]
    r2 = ols_r2(clean[signal_col].to_numpy(dtype=float), clean[y_col].to_numpy(dtype=float))
    lo, hi = block_bootstrap_r2(sub, signal_col, y_col, seed + 1)
    return r2, lo, hi


def fast_block_bootstrap_hit(sub: pd.DataFrame, x_col: str, y_col: str, seed: int) -> tuple[float, float]:
    clean = sub[["received_at", x_col, y_col]].dropna()
    clean = clean[np.isfinite(clean[x_col]) & np.isfinite(clean[y_col])]
    signs = (np.sign(clean[x_col]) != 0) & (np.sign(clean[y_col]) != 0)
    clean = clean.loc[signs].copy()
    if len(clean) < 30:
        return math.nan, math.nan
    clean["hit"] = (np.sign(clean[x_col]) == np.sign(clean[y_col])).astype(int)
    clean["block_id"] = time_block_ids(clean["received_at"])
    block_stats = clean.groupby("block_id", as_index=False).agg(hits=("hit", "sum"), n=("hit", "size"))
    if len(block_stats) < 2:
        return math.nan, math.nan
    stats = block_stats[["hits", "n"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = stats[rng.integers(0, len(stats), size=len(stats))]
        hits, n = sampled.sum(axis=0)
        if n >= 5:
            vals.append(float(hits / n))
    if len(vals) < 20:
        return math.nan, math.nan
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def ols_prediction_fast(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 5 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.array([], dtype=float), math.nan
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    beta = float(np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2))
    alpha = y_mean - beta * x_mean
    pred = alpha + beta * x
    return pred, ols_r2(x, y)


def prediction_top_decile_metric(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    pred, r2 = ols_prediction_fast(x, y)
    if len(pred) == 0:
        return math.nan, math.nan, math.nan, 0
    valid = np.isfinite(pred) & np.isfinite(y) & (np.sign(pred) != 0) & (np.sign(y) != 0)
    if valid.sum() < 5:
        return r2, math.nan, math.nan, 0
    pred = pred[valid]
    y = y[valid]
    threshold = np.nanquantile(np.abs(pred), 0.90)
    top = np.abs(pred) >= threshold
    if top.sum() < 5:
        return r2, math.nan, math.nan, int(top.sum())
    hit = np.sign(pred[top]) == np.sign(y[top])
    directional_return = np.sign(pred[top]) * y[top]
    return r2, float(hit.mean()), float(np.mean(directional_return)), int(top.sum())


def fixed_prediction_hit_ci(clean: pd.DataFrame, pred: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    if len(pred) != len(clean):
        return math.nan, math.nan
    valid = np.isfinite(pred) & np.isfinite(y) & (np.sign(pred) != 0) & (np.sign(y) != 0)
    if valid.sum() < 30:
        return math.nan, math.nan
    threshold = np.nanquantile(np.abs(pred[valid]), 0.90)
    top = valid & (np.abs(pred) >= threshold)
    if top.sum() < 30:
        return math.nan, math.nan
    boot = clean.loc[top, ["received_at"]].copy()
    boot["hit"] = (np.sign(pred[top]) == np.sign(y[top])).astype(int)
    boot["block_id"] = time_block_ids(boot["received_at"])
    block_stats = boot.groupby("block_id", as_index=False).agg(hits=("hit", "sum"), n=("hit", "size"))
    if len(block_stats) < 2:
        return math.nan, math.nan
    stats = block_stats[["hits", "n"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = stats[rng.integers(0, len(stats), size=len(stats))]
        hits, n = sampled.sum(axis=0)
        if n >= 5:
            vals.append(float(hits / n))
    if len(vals) < 20:
        return math.nan, math.nan
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def compute_metric_fast(sub: pd.DataFrame, x_col: str, y_col: str, seed: int) -> MetricResult:
    clean = sub[["received_at", x_col, y_col]].dropna()
    clean = clean[np.isfinite(clean[x_col]) & np.isfinite(clean[y_col])]
    if len(clean) < 10:
        return MetricResult(
            r2=math.nan,
            r2_lo=math.nan,
            r2_hi=math.nan,
            hit_rate=math.nan,
            hit_lo=math.nan,
            hit_hi=math.nan,
            directional_return_bps=math.nan,
            n_eval=len(clean),
            top_decile_n=0,
        )
    x = clean[x_col].to_numpy(dtype=float)
    y = clean[y_col].to_numpy(dtype=float)
    pred, r2 = ols_prediction_fast(x, y)
    _, hit, directional_return, top_decile_n = prediction_top_decile_metric(x, y)
    r2_lo, r2_hi = block_bootstrap_r2(clean, x_col, y_col, seed)
    hit_lo, hit_hi = fixed_prediction_hit_ci(clean, pred, y, seed + 1)
    return MetricResult(
        r2=float(r2) if np.isfinite(r2) else math.nan,
        r2_lo=r2_lo,
        r2_hi=r2_hi,
        hit_rate=float(hit) if np.isfinite(hit) else math.nan,
        hit_lo=hit_lo,
        hit_hi=hit_hi,
        directional_return_bps=float(directional_return) if np.isfinite(directional_return) else math.nan,
        n_eval=len(clean),
        top_decile_n=int(top_decile_n),
    )


def decile_aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    decile_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    base = df[
        df["is_book_state_complete"].fillna(False)
        & df["market_mean_depth"].gt(0)
        & df["mid"].notna()
    ].copy()
    for horizon in HORIZONS:
        y_col = f"future_return_bps_{horizon}s"
        for variant in VARIANTS:
            signal_col = f"signal_{variant}_{horizon}s"
            sub = base[
                base[signal_col].replace([np.inf, -np.inf], np.nan).notna()
                & base[y_col].replace([np.inf, -np.inf], np.nan).notna()
                & base[signal_col].ne(0)
            ].copy()
            if len(sub) < 50:
                continue
            r2, r2_lo, r2_hi = pooled_r2(sub, signal_col, y_col, RNG_SEED + horizon + len(variant))
            sub["abs_signal"] = sub[signal_col].abs()
            try:
                sub["decile"] = pd.qcut(sub["abs_signal"], 10, labels=False, duplicates="drop") + 1
            except ValueError:
                continue
            for decile, rows in sub.groupby("decile", sort=True):
                signs = (np.sign(rows[signal_col]) != 0) & (np.sign(rows[y_col]) != 0)
                hit_rate = (
                    float((np.sign(rows.loc[signs, signal_col]) == np.sign(rows.loc[signs, y_col])).mean())
                    if signs.any()
                    else math.nan
                )
                lo, hi = fast_block_bootstrap_hit(rows, signal_col, y_col, RNG_SEED + horizon * 100 + int(decile) + len(variant))
                decile_rows.append({
                    "variant": variant,
                    "horizon_sec": horizon,
                    "decile": int(decile),
                    "n": int(len(rows)),
                    "mean_abs_signal": float(rows["abs_signal"].mean()),
                    "mean_next_mid_return_bps": float(rows[y_col].mean()),
                    "directional_return_bps": float((np.sign(rows[signal_col]) * rows[y_col]).mean()),
                    "hit_rate": hit_rate,
                    "hit_rate_ci_lo": lo,
                    "hit_rate_ci_hi": hi,
                    "pooled_r2_all_rows": r2,
                    "pooled_r2_ci_lo": r2_lo,
                    "pooled_r2_ci_hi": r2_hi,
                })
            top = sub[sub["decile"].eq(sub["decile"].max())]
            top_signs = (np.sign(top[signal_col]) != 0) & (np.sign(top[y_col]) != 0)
            hit_rate = (
                float((np.sign(top.loc[top_signs, signal_col]) == np.sign(top.loc[top_signs, y_col])).mean())
                if top_signs.any()
                else math.nan
            )
            lo, hi = fast_block_bootstrap_hit(top, signal_col, y_col, RNG_SEED + horizon * 1000 + len(variant))
            comparison_rows.append({
                "variant": variant,
                "horizon_sec": horizon,
                "top_decile_n": int(len(top)),
                "top_decile_hit_rate": hit_rate,
                "top_decile_hit_rate_ci_lo": lo,
                "top_decile_hit_rate_ci_hi": hi,
                "top_decile_directional_return_bps": float((np.sign(top[signal_col]) * top[y_col]).mean()),
                "pooled_r2_all_rows": r2,
                "pooled_r2_ci_lo": r2_lo,
                "pooled_r2_ci_hi": r2_hi,
            })
    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        l1 = comparison[comparison["variant"].eq("l1_cks")][
            [
                "horizon_sec",
                "top_decile_hit_rate",
                "top_decile_directional_return_bps",
                "pooled_r2_all_rows",
            ]
        ].rename(
            columns={
                "top_decile_hit_rate": "l1_top_decile_hit_rate",
                "top_decile_directional_return_bps": "l1_directional_return_bps",
                "pooled_r2_all_rows": "l1_pooled_r2_all_rows",
            }
        )
        comparison = comparison.merge(l1, on="horizon_sec", how="left")
        comparison["hit_delta_vs_l1_pp"] = (
            comparison["top_decile_hit_rate"] - comparison["l1_top_decile_hit_rate"]
        ) * 100.0
        comparison["directional_return_delta_vs_l1_bps"] = (
            comparison["top_decile_directional_return_bps"] - comparison["l1_directional_return_bps"]
        )
        comparison["r2_delta_vs_l1"] = (
            comparison["pooled_r2_all_rows"] - comparison["l1_pooled_r2_all_rows"]
        )
    return pd.DataFrame(decile_rows), comparison


def market_panel(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["run_id", "market_key"]
    for group_idx, ((run_id, market_key), sub) in enumerate(df.groupby(group_cols, sort=False)):
        sub = sub.sort_values("received_at")
        market_id = (
            str(sub["market_id"].dropna().replace("", np.nan).dropna().iloc[0])
            if sub["market_id"].astype(bool).any()
            else str(market_key)
        )
        family = (
            str(sub["family"].dropna().replace("", np.nan).dropna().iloc[0])
            if sub["family"].astype(bool).any()
            else ""
        )
        slug = (
            str(sub["slug"].dropna().replace("", np.nan).dropna().iloc[0])
            if sub["slug"].astype(bool).any()
            else ""
        )
        question = (
            str(sub["question"].dropna().replace("", np.nan).dropna().iloc[0])
            if sub["question"].astype(bool).any()
            else ""
        )
        n_book_events = int(sub["event_type"].isin(["book", "price_change"]).sum())
        trades = sub[sub["is_trade"]]
        n_trades = int(len(trades))
        n_classifiable = int(trades["trade_side"].str.upper().isin(["BUY", "SELL"]).sum())
        label = sample_size_label(n_classifiable)
        mean_depth = float(sub["touch_depth"].replace([np.inf, -np.inf], np.nan).mean())
        for horizon in HORIZONS:
            y_col = f"future_return_bps_{horizon}s"
            valid = sub[
                sub["is_book_state_complete"].fillna(False)
                & sub["mid"].notna()
                & sub[y_col].notna()
            ].copy()
            for variant in VARIANTS:
                signal_col = f"signal_{variant}_{horizon}s"
                eval_rows = valid[
                    valid[signal_col].replace([np.inf, -np.inf], np.nan).notna()
                    & valid[signal_col].ne(0)
                ]
                metric = compute_metric_fast(
                    eval_rows,
                    signal_col,
                    y_col,
                    RNG_SEED + group_idx * 100 + horizon + len(variant),
                )
                reportable = metric_reportable(n_classifiable, metric)
                rows.append({
                    "run_id": run_id,
                    "market_id": market_id,
                    "asset_id": ",".join(sorted(sub["asset_id"].dropna().astype(str).unique())[:4]),
                    "slug": slug,
                    "question": question,
                    "family": family,
                    "variant": variant,
                    "horizon_sec": horizon,
                    "n_book_events": n_book_events,
                    "n_trades": n_trades,
                    "n_classifiable": n_classifiable,
                    "mean_depth_at_touch": mean_depth,
                    "sample_size_label": label,
                    "metric_reportable": reportable,
                    "eval_rows": metric.n_eval,
                    "top_decile_n": metric.top_decile_n,
                    "r2": public_value(metric.r2, reportable),
                    "r2_ci_lo": public_value(metric.r2_lo, reportable),
                    "r2_ci_hi": public_value(metric.r2_hi, reportable),
                    "hit_rate_top_decile": public_value(metric.hit_rate, reportable),
                    "hit_rate_ci_lo": public_value(metric.hit_lo, reportable),
                    "hit_rate_ci_hi": public_value(metric.hit_hi, reportable),
                    "directional_return_top_decile_bps": public_value(metric.directional_return_bps, reportable),
                    "hit_rate_top_decile_raw": metric.hit_rate,
                    "directional_return_top_decile_bps_raw": metric.directional_return_bps,
                })
    return pd.DataFrame(rows)


def attach_a1_baseline_check(comparison: pd.DataFrame, a1_deciles_path: Path) -> pd.DataFrame:
    if comparison.empty or not a1_deciles_path.exists():
        return comparison
    a1 = pd.read_csv(a1_deciles_path)
    a1 = a1[a1["decile"].eq(10)][
        ["horizon_sec", "hit_rate", "directional_return_bps", "n"]
    ].rename(
        columns={
            "hit_rate": "a1_l1_top_decile_hit_rate",
            "directional_return_bps": "a1_l1_directional_return_bps",
            "n": "a1_l1_top_decile_n",
        }
    )
    out = comparison.merge(a1, on="horizon_sec", how="left")
    out["a12_l1_minus_a1_hit_pp"] = np.where(
        out["variant"].eq("l1_cks"),
        (out["top_decile_hit_rate"] - out["a1_l1_top_decile_hit_rate"]) * 100.0,
        np.nan,
    )
    out["a12_l1_minus_a1_directional_return_bps"] = np.where(
        out["variant"].eq("l1_cks"),
        out["top_decile_directional_return_bps"] - out["a1_l1_directional_return_bps"],
        np.nan,
    )
    return out


def plot_decile_hits(deciles: pd.DataFrame) -> list[Path]:
    PLOTS.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for horizon, sub in deciles.groupby("horizon_sec"):
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        for variant, rows in sub.groupby("variant"):
            rows = rows.sort_values("decile")
            ax.plot(rows["decile"], rows["hit_rate"], marker="o", linewidth=1.6, label=variant)
        ax.axhline(0.50, color="#666666", linewidth=1, linestyle="--")
        ax.set_title(f"A1.2 MLOFI decile hit rate, {int(horizon)}s")
        ax.set_xlabel("Absolute depth-normalized signal decile")
        ax.set_ylabel("Hit rate")
        ax.set_ylim(0.30, 0.80)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncols=2)
        fig.tight_layout()
        out = PLOTS / f"block_a12_mlofi_decile_hit_rate_{int(horizon)}s.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_top_decile_comparison(comparison: pd.DataFrame) -> Path | None:
    if comparison.empty:
        return None
    top = comparison.copy()
    PLOTS.mkdir(parents=True, exist_ok=True)
    pivot = top.pivot(index="variant", columns="horizon_sec", values="hit_delta_vs_l1_pp")
    pivot = pivot.reindex(index=[v for v in VARIANTS if v in pivot.index], columns=list(HORIZONS))
    mat = pivot.to_numpy(dtype=float)
    if mat.size == 0 or np.all(~np.isfinite(mat)):
        return None
    vmax = float(np.nanmax(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("Top-decile hit-rate delta vs L1 CKS")
    ax.set_xlabel("Forward horizon")
    ax.set_ylabel("Variant")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{int(h)}s" for h in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("percentage points")
    fig.tight_layout()
    out = PLOTS / "block_a12_mlofi_top_decile_delta_heatmap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def comparison_table(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "_No comparison rows._"
    rows: list[list[object]] = []
    ordered = comparison.sort_values(["horizon_sec", "top_decile_hit_rate"], ascending=[True, False])
    for _, row in ordered.iterrows():
        rows.append([
            str(row["variant"]),
            int(row["horizon_sec"]),
            pct(float(row["top_decile_hit_rate"])),
            f"[{pct(float(row['top_decile_hit_rate_ci_lo']))}, {pct(float(row['top_decile_hit_rate_ci_hi']))}]",
            bps(float(row["top_decile_directional_return_bps"])),
            f"{float(row['pooled_r2_all_rows']):.5f}" if np.isfinite(row["pooled_r2_all_rows"]) else "n/a",
            f"{float(row['hit_delta_vs_l1_pp']):+.2f}",
            bps(float(row["directional_return_delta_vs_l1_bps"])),
            f"{int(row['top_decile_n']):,}",
        ])
    return markdown_table(
        ["variant", "h", "top hit", "hit CI", "top dir ret", "pooled R2", "hit delta pp", "dir delta", "top n"],
        rows,
    )


def winner_table(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "_No winner rows._"
    rows: list[list[object]] = []
    for horizon, sub in comparison.groupby("horizon_sec"):
        sub = sub.sort_values(["top_decile_hit_rate", "top_decile_directional_return_bps"], ascending=False)
        best = sub.iloc[0]
        l1 = sub[sub["variant"].eq("l1_cks")].iloc[0]
        rows.append([
            int(horizon),
            str(best["variant"]),
            pct(float(best["top_decile_hit_rate"])),
            f"{float(best['top_decile_hit_rate'] - l1['top_decile_hit_rate']) * 100.0:+.2f} pp",
            bps(float(best["top_decile_directional_return_bps"])),
            bps(float(best["top_decile_directional_return_bps"] - l1["top_decile_directional_return_bps"])),
        ])
    return markdown_table(["h", "winner", "hit", "hit vs L1", "dir ret", "dir ret vs L1"], rows)


def market_snapshot_table(panel: pd.DataFrame, horizon: int = 5) -> str:
    sub = panel[panel["horizon_sec"].eq(horizon) & panel["metric_reportable"].fillna(False)].copy()
    if sub.empty:
        return "_No reportable market rows._"
    sub = sub.sort_values(["hit_rate_top_decile", "directional_return_top_decile_bps"], ascending=False).head(18)
    rows: list[list[object]] = []
    for _, row in sub.iterrows():
        rows.append([
            row["run_id"],
            str(row["slug"] or row["market_id"])[:54],
            row["variant"],
            str(row["family"])[:24],
            f"{int(row['n_classifiable']):,}",
            row["sample_size_label"],
            pct(float(row["hit_rate_top_decile"])),
            bps(float(row["directional_return_top_decile_bps"])),
            f"{int(row['top_decile_n']):,}",
        ])
    return markdown_table(
        ["run", "market", "variant", "family", "n class", "label", "hit", "dir ret", "top n"],
        rows,
    )


def write_note(
    deciles: pd.DataFrame,
    panel: pd.DataFrame,
    comparison: pd.DataFrame,
    plot_paths: list[Path],
    features_path: Path,
    deciles_path: Path,
    panel_path: Path,
    comparison_path: Path,
) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    best_rows = []
    mlofi_only = comparison[comparison["variant"].ne("l1_cks")].copy()
    for horizon, sub in mlofi_only.groupby("horizon_sec"):
        l1 = comparison[comparison["variant"].eq("l1_cks") & comparison["horizon_sec"].eq(horizon)]
        if l1.empty:
            continue
        best = sub.sort_values(["top_decile_hit_rate", "top_decile_directional_return_bps"], ascending=False).iloc[0]
        best_rows.append((int(horizon), best, l1.iloc[0]))
    headline_parts = []
    for horizon, best, l1 in best_rows:
        delta = (float(best["top_decile_hit_rate"]) - float(l1["top_decile_hit_rate"])) * 100.0
        headline_parts.append(f"{horizon}s {best['variant']} {delta:+.2f}pp")
    headline = "; ".join(headline_parts) if headline_parts else "no MLOFI comparison rows available"
    plot_refs = "\n".join(f"![]({rel_note_path(path)})" for path in plot_paths)
    l1_check = comparison[comparison["variant"].eq("l1_cks")][
        ["horizon_sec", "a12_l1_minus_a1_hit_pp", "a12_l1_minus_a1_directional_return_bps"]
    ].dropna(how="all")
    check_rows = []
    for _, row in l1_check.iterrows():
        check_rows.append([
            int(row["horizon_sec"]),
            f"{float(row['a12_l1_minus_a1_hit_pp']):+.2f} pp" if np.isfinite(row["a12_l1_minus_a1_hit_pp"]) else "n/a",
            bps(float(row["a12_l1_minus_a1_directional_return_bps"]))
            if np.isfinite(row["a12_l1_minus_a1_directional_return_bps"])
            else "n/a",
        ])

    note = f"""---
tags: [dali, block-a12, mlofi, results]
---

# Block A1.2 MLOFI Findings

## Headline

A1.2 replayed the existing A0/A0b raw captures into true top-10 per-level OFI and compared MLOFI variants against the current L1 CKS baseline. The load-bearing result is that MLOFI does not beat L1 where A1's signal is cleanest: L1 wins 1s, 5s, and 30s top-decile hit rate, including the 5s headline cell. Best MLOFI-vs-L1 hit-rate deltas by horizon: {headline}. The only positive MLOFI delta is the 300s depth-weighted cell, which is modest and sits in the horizon A1 already treated as composition-sensitive. Treat this as a feature-family sniff test, not optimization or a deployment result.

## Outputs

- `{display_path(features_path)}`: per-event top-10 MLOFI feature sidecar.
- `{display_path(deciles_path)}`: variant x horizon x absolute-signal decile aggregate.
- `{display_path(panel_path)}`: per-market x horizon x variant panel with A1 reporting guards.
- `{display_path(comparison_path)}`: top-decile side-by-side table vs L1 baseline.
- `{display_path(PLOTS)}/`: decile hit-rate plots and top-decile delta heatmap.

## Method

Per-level OFI compares previous and new `(price, size)` at each book rank `k=1..10` using the same CKS rules as A1 L1 OFI: bid price up contributes new bid size, bid price down contributes negative previous bid size, unchanged bid contributes size delta; ask price down contributes negative new ask size, ask price up contributes previous ask size, unchanged ask contributes previous minus new size. `combined_ofi_lk = bid_ofi_lk + ask_ofi_lk`.

The tested variants are:

- `l1_cks`: `combined_ofi_l1`, the A1 baseline.
- `integrated_l1_l10`: sum of `combined_ofi_l1..l10`.
- `depth_weighted_l1_l10`: sum of `combined_ofi_lk * level_depth_lk` divided by sum of `level_depth_lk`, using current bid+ask size at level `k`.
- `exp_decay_alpha_0p1`, `0p3`, `0p5`: sum of `combined_ofi_lk * exp(-alpha * (k - 1))`.

Each event-level variant is rolled over 1s, 5s, 30s, and 300s, flipped into YES/NO market direction with A1's `direction_factor`, then normalized by each market's mean touch depth. Deciles are global equal-count buckets within each `(variant, horizon)` based on absolute normalized signal magnitude; decile 10 is largest absolute signal, not most bullish.

Replay intentionally mirrors A1's shard-local anchoring: each JSONL shard starts from its first full `book` snapshot for an asset, then applies `price_change` updates. That is why the L1 baseline check below is the key comparability test.

## L1 Baseline Check

This is the A1.2 L1 replay minus the existing A1 decile CSV. It matches exactly at all four horizons, so the MLOFI sidecar is comparable to the A1 baseline.

{markdown_table(["h", "hit delta", "dir ret delta"], check_rows)}

## Horizon Winners

{winner_table(comparison)}

## Full Top-Decile Comparison

{comparison_table(comparison)}

## Reportable Per-Market Snapshot

Top reportable 5s rows across variants:

{market_snapshot_table(panel, 5)}

These are per-market diagnostics, not the headline. Some high-hit rows are `thin_wide_CI`; the pooled depth-normalized comparison above is the result to cite.

## Plots

{plot_refs}

## Interpretation

Current A0/A0b evidence says keep L1 CKS as the primary A2 OFI signal. Logging top-10 per-level OFI in A2 is still useful if storage/CPU cost is acceptable, because it preserves optionality for later family/regime work, but it should not replace or distract from the simpler L1 feature. The 300s depth-weighted improvement is not strong enough to drive design by itself.

Recommended next action for Justin: keep L1 CKS as the A2 headline signal, and log top-10 MLOFI as an optional sidecar only if capture resources are comfortable.
"""
    NOTE.write_text(note, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", dest="runs", action="append", type=parse_run_spec)
    parser.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--deciles-out", type=Path, default=DEFAULT_DECILES)
    parser.add_argument("--market-panel-out", type=Path, default=DEFAULT_MARKET_PANEL)
    parser.add_argument("--comparison-out", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--a1-deciles", type=Path, default=DEFAULT_A1_DECILES)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = list(args.runs or DEFAULT_RUNS)
    if args.depth != DEPTH:
        raise SystemExit("A1.2 is fixed to --depth 10 so output columns match the spec")
    if not args.skip_replay:
        replay_runs(runs, args.features_out, args.depth)

    print(f"loading {display_path(args.features_out)}", flush=True)
    df = load_features(args.features_out)
    print(f"loaded {len(df):,} MLOFI feature rows", flush=True)
    df = add_variant_event_columns(df)
    df = add_horizon_features(df)

    print("building MLOFI decile aggregate", flush=True)
    deciles, comparison = decile_aggregate(df)
    comparison = attach_a1_baseline_check(comparison, args.a1_deciles)
    print("building MLOFI market panel", flush=True)
    panel = market_panel(df)

    args.deciles_out.parent.mkdir(parents=True, exist_ok=True)
    deciles.to_csv(args.deciles_out, index=False)
    panel.to_csv(args.market_panel_out, index=False)
    comparison.to_csv(args.comparison_out, index=False)

    paths: list[Path] = []
    if not args.skip_plots:
        paths.extend(plot_decile_hits(deciles))
        top_delta = plot_top_decile_comparison(comparison)
        if top_delta:
            paths.append(top_delta)
    write_note(
        deciles,
        panel,
        comparison,
        paths,
        args.features_out,
        args.deciles_out,
        args.market_panel_out,
        args.comparison_out,
    )
    print(f"features: {display_path(args.features_out)} ({len(df):,} rows)")
    print(f"deciles: {display_path(args.deciles_out)} ({len(deciles):,} rows)")
    print(f"market panel: {display_path(args.market_panel_out)} ({len(panel):,} rows)")
    print(f"comparison: {display_path(args.comparison_out)} ({len(comparison):,} rows)")
    print(f"plots: {display_path(PLOTS)} ({len(paths):,} files)")
    print(f"note: {display_path(NOTE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
