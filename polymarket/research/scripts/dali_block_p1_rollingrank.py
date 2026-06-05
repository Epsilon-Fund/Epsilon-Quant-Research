"""P1 rolling-rank sizing versus decile gating.

This sidecar reads the A1 replay feature parquet and compares two mappings of
the same trailing percentile-rank signal state:

* continuous position: ``2 * rank - 1``
* top/bottom-decile gate: ``+1`` above 90th pct, ``-1`` below 10th pct

The analysis is research-only. It does not touch live-trading code, raw
captures, or prior A1 artifacts.
"""
from __future__ import annotations

import math
import re
import zlib
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from dali_block_a1_analyze import FEE_BY_CATEGORY, family_category


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
FEATURES = ANALYSIS / "block_a1_features.parquet"
SURFACE_OUT = ANALYSIS / "csv_outputs" / "dali" / "p1_rollingrank_surface.csv"
BY_MARKET_OUT = ANALYSIS / "csv_outputs" / "dali" / "p1_rollingrank_by_market.csv"
HEATMAP_OUT = ANALYSIS / "csv_outputs" / "dali" / "p1_rollingrank_row_count_heatmap.csv"
NOTE = NOTES / "block_p1_rollingrank_findings.md"

SIGNALS = ("ofi_l1", "tob_imbalance_level", "tfi", "weighted_mid_edge_bps")
WINDOWS = (30, 60, 300, 900, 1800)
HORIZONS = (5, 10, 30, 60, 300)
MAPPINGS = ("continuous_rank", "decile_gate")
PNL_COLS = {"executable": "pnl_executable_bps", "mid": "pnl_mid_bps"}

BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260530
MAX_STALE_BOOK_SECONDS = 5.0
MAX_FUTURE_QUOTE_AGE_SECONDS = 5.0
MIN_RANK_COUNT = 30
MIN_REPORT_TRADES = 30

SEGMENTS = {
    "overall": None,
    "family": "family",
    "spread_regime": "spread_regime",
    "depth_regime": "depth_regime",
    "time_to_resolution": "time_to_resolution",
    "clock_utc_6h": "clock_utc_6h",
}


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def stable_seed_offset(*parts: object, modulo: int = 100_000) -> int:
    text = "|".join(str(part) for part in parts)
    return int(zlib.crc32(text.encode("utf-8")) % modulo)


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def compact_number(value: object) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{int(value):,}"


def safe_text(value: object, max_len: int = 56) -> str:
    text = str(value if value is not None else "").replace("|", "/")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "."


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def qbucket(series: pd.Series, labels: list[str]) -> pd.Series:
    out = pd.Series("unknown", index=series.index, dtype="object")
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return out
    try:
        bucketed = pd.qcut(valid.rank(method="first"), len(labels), labels=labels)
    except ValueError:
        return out
    out.loc[bucketed.index] = bucketed.astype(str)
    return out


def fee_amount(category: np.ndarray, price: np.ndarray) -> np.ndarray:
    rates = np.array(
        [FEE_BY_CATEGORY.get(str(cat), FEE_BY_CATEGORY["Other"])["fee_rate"] for cat in category],
        dtype=float,
    )
    p = np.clip(price.astype(float), 0.0, 1.0)
    return rates * p * (1.0 - p)


def load_features(path: Path) -> pd.DataFrame:
    cols = [
        "run_id",
        "received_at",
        "exchange_ts",
        "event_type",
        "asset_id",
        "market_id",
        "family",
        "slug",
        "question",
        "outcome_index",
        "is_book_state_complete",
        "book_staleness_seconds",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "tob_imbalance",
        "ofi_combined_event",
        "trade_side",
        "last_trade_side",
        "trade_size",
        "market_resolved_at",
    ]
    con = duckdb.connect()
    select_cols = ", ".join(cols)
    df = con.execute(f"SELECT {select_cols} FROM read_parquet('{path}')").df()
    con.close()
    if df.empty:
        raise SystemExit(f"no rows found in {display_path(path)}")

    df["source_row"] = np.arange(len(df), dtype=np.int64)
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    df["exchange_ts"] = pd.to_datetime(df["exchange_ts"], utc=True, errors="coerce")
    df["event_ts"] = df["exchange_ts"]
    df["market_resolved_at"] = pd.to_datetime(df["market_resolved_at"], utc=True, errors="coerce")
    for col in ("run_id", "event_type", "asset_id", "market_id", "family", "slug", "question"):
        df[col] = df[col].fillna("").astype(str)
    numeric_cols = [
        "outcome_index",
        "book_staleness_seconds",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "tob_imbalance",
        "ofi_combined_event",
        "trade_size",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ok = (
        df["event_ts"].notna()
        & df["is_book_state_complete"].fillna(False).astype(bool)
        & df["book_staleness_seconds"].le(MAX_STALE_BOOK_SECONDS)
        & df["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & df["mid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_bid"].ge(0.0)
        & df["best_ask"].le(1.0)
        & df["best_ask"].ge(df["best_bid"])
        & df["mid"].gt(0.0)
        & df["mid"].lt(1.0)
    )
    df = df.loc[ok].copy()
    if df.empty:
        raise SystemExit("no rows survive complete/fresh touch filter")

    df["market_key"] = df["run_id"] + ":" + df["market_id"]
    df["asset_key"] = df["run_id"] + ":" + df["asset_id"]
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    df["directional_mid"] = np.where(df["direction_factor"].gt(0), df["mid"], 1.0 - df["mid"])
    df["touch_depth"] = df[["best_bid_size", "best_ask_size"]].sum(axis=1, min_count=2)
    df["spread_bps"] = np.where(
        df["mid"].gt(0) & df["spread"].notna(),
        df["spread"] / df["mid"] * 10_000.0,
        np.nan,
    )

    side = (
        df["trade_side"]
        .fillna(df["last_trade_side"])
        .fillna("")
        .astype(str)
        .str.upper()
        .map({"BUY": 1.0, "SELL": -1.0})
        .fillna(0.0)
    )
    df["is_trade"] = df["event_type"].eq("last_trade_price")
    df["signed_trade_size_live"] = df["is_trade"].astype(float) * side * df["trade_size"].fillna(0.0)
    df["ofi_combined_event"] = df["ofi_combined_event"].fillna(0.0)

    depth_denom = df["best_bid_size"] + df["best_ask_size"]
    weighted_mid = np.divide(
        df["best_ask"] * df["best_bid_size"] + df["best_bid"] * df["best_ask_size"],
        depth_denom,
        out=np.full(len(df), np.nan, dtype=float),
        where=depth_denom.to_numpy(dtype=float) > 0,
    )
    df["weighted_mid_edge_bps"] = df["direction_factor"] * (
        (weighted_mid - df["mid"]) / df["mid"] * 10_000.0
    )
    df["tob_imbalance_level"] = df["direction_factor"] * df["tob_imbalance"]

    df["category"] = df["family"].map(family_category)
    market_depth = (
        df.groupby("market_key", as_index=False)["touch_depth"]
        .mean()
        .rename(columns={"touch_depth": "market_mean_depth"})
    )
    df = df.merge(market_depth, on="market_key", how="left", sort=False)
    df["relative_depth"] = df["touch_depth"] / df["market_mean_depth"]
    df["spread_regime"] = qbucket(
        df["spread_bps"],
        ["q1_tight", "q2", "q3", "q4_wide"],
    )
    df["depth_regime"] = qbucket(
        df["relative_depth"],
        ["q1_shallow", "q2", "q3", "q4_deep"],
    )

    ttr = (df["market_resolved_at"] - df["event_ts"]).dt.total_seconds()
    df["time_to_resolution"] = "unresolved"
    resolved = ttr.notna()
    df.loc[resolved & ttr.le(300), "time_to_resolution"] = "resolved_le_5m"
    df.loc[resolved & ttr.gt(300) & ttr.le(1800), "time_to_resolution"] = "resolved_5_30m"
    df.loc[resolved & ttr.gt(1800) & ttr.le(7200), "time_to_resolution"] = "resolved_30_120m"
    df.loc[resolved & ttr.gt(7200), "time_to_resolution"] = "resolved_gt_120m"
    hour = df["event_ts"].dt.hour
    df["clock_utc_6h"] = pd.cut(
        hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["utc_00_06", "utc_06_12", "utc_12_18", "utc_18_24"],
    ).astype(str)

    df["event_time_ns"] = df["event_ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
    df["resolution_time_ns"] = df["market_resolved_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    df.loc[df["market_resolved_at"].isna(), "resolution_time_ns"] = np.iinfo(np.int64).max

    df = df.sort_values(
        ["run_id", "asset_id", "event_ts", "received_at", "source_row"],
        kind="mergesort",
    ).reset_index(drop=True)
    df["market_code"] = df["market_key"].astype("category").cat.codes.astype(np.int32)
    return df


def add_future_quotes(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    grouped = list(df.groupby("asset_key", sort=False))
    for idx, (_asset, group) in enumerate(grouped, start=1):
        if idx % 10 == 0 or idx == len(grouped):
            print(f"future quotes {idx:02d}/{len(grouped):02d}", flush=True)
        g = group.sort_values(["event_time_ns", "received_at", "source_row"], kind="mergesort").copy()
        times = g["event_time_ns"].to_numpy(dtype=np.int64)
        bid = g["best_bid"].to_numpy(dtype=float)
        ask = g["best_ask"].to_numpy(dtype=float)
        mid = g["mid"].to_numpy(dtype=float)
        for horizon in HORIZONS:
            target = times + horizon * 1_000_000_000
            idxs = np.searchsorted(times, target, side="right") - 1
            valid = (idxs >= 0) & (target <= times[-1])
            age_sec = np.full(len(g), np.nan, dtype=float)
            future_bid = np.full(len(g), np.nan, dtype=float)
            future_ask = np.full(len(g), np.nan, dtype=float)
            future_mid = np.full(len(g), np.nan, dtype=float)
            if valid.any():
                age_sec[valid] = (target[valid] - times[idxs[valid]]) / 1_000_000_000.0
                quote_ok = valid & (age_sec <= MAX_FUTURE_QUOTE_AGE_SECONDS)
                future_bid[quote_ok] = bid[idxs[quote_ok]]
                future_ask[quote_ok] = ask[idxs[quote_ok]]
                future_mid[quote_ok] = mid[idxs[quote_ok]]
            g[f"future_bid_{horizon}s"] = future_bid
            g[f"future_ask_{horizon}s"] = future_ask
            g[f"future_mid_{horizon}s"] = future_mid
            g[f"future_quote_age_{horizon}s"] = age_sec
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["run_id", "asset_id", "event_time_ns", "received_at", "source_row"],
        kind="mergesort",
    ).reset_index(drop=True)


def rolling_percentile_position(g: pd.DataFrame, signal: pd.Series, window_sec: int) -> tuple[np.ndarray, np.ndarray]:
    indexed = pd.Series(signal.to_numpy(dtype=float), index=g["event_ts"])
    roller = indexed.rolling(f"{window_sec}s")
    rank = roller.rank(method="average")
    count = roller.count()
    pct_rank = (rank - 1.0) / (count - 1.0)
    pct_rank = pct_rank.where(count.gt(1), 0.5)
    pct_rank = pct_rank.where(count.gt(0), np.nan)
    position = (2.0 * pct_rank - 1.0).clip(-1.0, 1.0)
    return position.to_numpy(dtype=float), count.to_numpy(dtype=float)


def signal_series(g: pd.DataFrame, signal_name: str, window_sec: int) -> pd.Series:
    if signal_name == "ofi_l1":
        base = g["direction_factor"].to_numpy(dtype=float) * g["ofi_combined_event"].fillna(0.0).to_numpy(dtype=float)
        return pd.Series(base, index=g["event_ts"]).rolling(f"{window_sec}s").sum().reset_index(drop=True)
    if signal_name == "tfi":
        base = g["direction_factor"].to_numpy(dtype=float) * g["signed_trade_size_live"].fillna(0.0).to_numpy(dtype=float)
        return pd.Series(base, index=g["event_ts"]).rolling(f"{window_sec}s").sum().reset_index(drop=True)
    if signal_name == "tob_imbalance_level":
        return g["tob_imbalance_level"].astype(float).reset_index(drop=True)
    if signal_name == "weighted_mid_edge_bps":
        return g["weighted_mid_edge_bps"].astype(float).reset_index(drop=True)
    raise ValueError(f"unknown signal {signal_name}")


def add_rank_for_signal(df: pd.DataFrame, signal_name: str, window_sec: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    grouped = list(df.groupby("asset_key", sort=False))
    for idx, (_asset, group) in enumerate(grouped, start=1):
        if idx % 10 == 0 or idx == len(grouped):
            print(f"rank {signal_name} {window_sec}s {idx:02d}/{len(grouped):02d}", flush=True)
        g = group.sort_values(["event_time_ns", "received_at", "source_row"], kind="mergesort").copy()
        sig = signal_series(g, signal_name, window_sec)
        position, rank_count = rolling_percentile_position(g, sig, window_sec)
        g["rank_position"] = position
        g["rank_count"] = rank_count
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["run_id", "asset_id", "event_time_ns", "received_at", "source_row"],
        kind="mergesort",
    ).reset_index(drop=True)


def apply_non_overlap(events: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    ordered = events.sort_values(
        ["market_code", "event_time_ns", "nonoverlap_score", "rank_count", "source_row"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    keep_positions: list[int] = []
    for _market, sub in ordered.groupby("market_code", sort=False):
        open_until = -1
        event_times = sub["event_time_ns"].to_numpy(dtype=np.int64)
        source_pos = np.arange(len(sub), dtype=np.int64)
        for local_pos, entry_ns in zip(source_pos, event_times, strict=False):
            exit_ns = int(entry_ns + horizon * 1_000_000_000)
            if entry_ns <= open_until:
                continue
            keep_positions.append(int(sub.index[local_pos]))
            open_until = exit_ns
    if not keep_positions:
        return ordered.iloc[0:0].copy()
    out = events.loc[keep_positions].sort_values(
        ["market_code", "event_time_ns", "source_row"],
        kind="mergesort",
    )
    return out.reset_index(drop=True)


def executable_and_mid_pnl(base: pd.DataFrame, position: np.ndarray, horizon: int) -> pd.DataFrame:
    out = base.copy()
    out["position"] = position.astype(float)
    abs_pos = np.abs(position.astype(float))
    side = np.sign(position.astype(float))
    token_side = side * out["direction_factor"].to_numpy(dtype=float)

    future_bid = out[f"future_bid_{horizon}s"].to_numpy(dtype=float)
    future_ask = out[f"future_ask_{horizon}s"].to_numpy(dtype=float)
    future_mid = out[f"future_mid_{horizon}s"].to_numpy(dtype=float)
    bid = out["best_bid"].to_numpy(dtype=float)
    ask = out["best_ask"].to_numpy(dtype=float)
    mid = out["mid"].to_numpy(dtype=float)
    direction_factor = out["direction_factor"].to_numpy(dtype=float)
    directional_mid = out["directional_mid"].to_numpy(dtype=float)
    future_directional_mid = np.where(direction_factor > 0, future_mid, 1.0 - future_mid)
    market_return = np.where(
        (directional_mid > 0.0) & np.isfinite(future_directional_mid),
        (future_directional_mid - directional_mid) / directional_mid * 10_000.0,
        np.nan,
    )
    out["pnl_mid_bps"] = position * market_return

    entry = np.where(token_side > 0, ask, bid)
    exit_px = np.where(token_side > 0, future_bid, future_ask)
    gross = np.where(token_side > 0, exit_px - entry, entry - exit_px)
    exec_ok = (
        np.isfinite(token_side)
        & (token_side != 0)
        & np.isfinite(entry)
        & np.isfinite(exit_px)
        & (entry > 0)
        & (exit_px >= 0)
        & np.isfinite(mid)
    )
    pnl_exec = np.full(len(out), np.nan, dtype=float)
    if exec_ok.any():
        category = out.loc[exec_ok, "category"].astype(str).to_numpy()
        entry_ok = entry[exec_ok]
        exit_ok = exit_px[exec_ok]
        fees = fee_amount(category, entry_ok) + fee_amount(category, exit_ok)
        unit_pnl = (gross[exec_ok] - fees) / entry_ok * 10_000.0
        pnl_exec[exec_ok] = abs_pos[exec_ok] * unit_pnl
    out["pnl_executable_bps"] = pnl_exec
    out["entry_price"] = entry
    out["exit_price"] = exit_px
    return out


def valid_horizon_mask(df: pd.DataFrame, horizon: int) -> pd.Series:
    target_ns = df["event_time_ns"].to_numpy(dtype=np.int64) + horizon * 1_000_000_000
    resolution_ns = df["resolution_time_ns"].to_numpy(dtype=np.int64)
    return (
        df[f"future_bid_{horizon}s"].replace([np.inf, -np.inf], np.nan).notna()
        & df[f"future_ask_{horizon}s"].replace([np.inf, -np.inf], np.nan).notna()
        & df[f"future_mid_{horizon}s"].replace([np.inf, -np.inf], np.nan).notna()
        & (target_ns <= resolution_ns)
    )


def block_stats(values: np.ndarray, block_labels: np.ndarray) -> np.ndarray:
    labels, inverse = np.unique(block_labels, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    if len(labels) == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack([sums, counts]).astype(float)


def bootstrap_from_blocks(values: np.ndarray, block_labels: np.ndarray, seed: int) -> tuple[float, float]:
    finite = np.isfinite(values) & np.isfinite(block_labels)
    values = values[finite]
    block_labels = block_labels[finite]
    if len(values) < 5:
        return math.nan, math.nan
    stats = block_stats(values, block_labels)
    if len(stats) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(stats), size=(BOOTSTRAP_SAMPLES, len(stats)))
    sampled = stats[sample_idx]
    sums = sampled[:, :, 0].sum(axis=1)
    counts = sampled[:, :, 1].sum(axis=1)
    means = sums / counts
    if len(means) < 20:
        return math.nan, math.nan
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_market_means(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[sample_idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_values(values: np.ndarray) -> dict[str, float]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return {
            "mean_pnl_bps": math.nan,
            "median_pnl_bps": math.nan,
            "win_rate": math.nan,
            "sharpe_per_trade": math.nan,
            "t_stat": math.nan,
        }
    std = float(np.std(clean, ddof=1)) if len(clean) > 1 else math.nan
    mean = float(np.mean(clean))
    return {
        "mean_pnl_bps": mean,
        "median_pnl_bps": float(np.median(clean)),
        "win_rate": float(np.mean(clean > 0.0)),
        "sharpe_per_trade": mean / std if np.isfinite(std) and std > 0 else math.nan,
        "t_stat": mean / (std / math.sqrt(len(clean))) if np.isfinite(std) and std > 0 else math.nan,
    }


def summary_row(
    *,
    frame: pd.DataFrame,
    pre_count: int,
    pnl_kind: str,
    signal: str,
    window: int,
    horizon: int,
    mapping: str,
    segment_type: str,
    segment_value: str,
    aggregation: str,
    seed_parts: Iterable[object],
) -> dict[str, object]:
    pnl_col = PNL_COLS[pnl_kind]
    values = frame[pnl_col].to_numpy(dtype=float) if not frame.empty else np.array([], dtype=float)
    clean_values = values[np.isfinite(values)]
    stats = summarize_values(values)
    if aggregation == "market_balanced":
        market_means = (
            frame.groupby("market_key", sort=False)[pnl_col].mean().replace([np.inf, -np.inf], np.nan)
            if not frame.empty
            else pd.Series(dtype=float)
        )
        market_values = market_means.dropna().to_numpy(dtype=float)
        mean = float(np.mean(market_values)) if len(market_values) else math.nan
        ci_lo, ci_hi = bootstrap_market_means(
            market_values,
            RNG_SEED + stable_seed_offset(*seed_parts, "market_balanced", pnl_kind),
        )
        std = float(np.std(market_values, ddof=1)) if len(market_values) > 1 else math.nan
        stats["mean_pnl_bps"] = mean
        stats["median_pnl_bps"] = float(np.median(market_values)) if len(market_values) else math.nan
        stats["win_rate"] = float(np.mean(market_values > 0.0)) if len(market_values) else math.nan
        stats["sharpe_per_trade"] = mean / std if np.isfinite(std) and std > 0 else math.nan
        stats["t_stat"] = mean / (std / math.sqrt(len(market_values))) if np.isfinite(std) and std > 0 else math.nan
        n_markets = int(len(market_values))
    else:
        ci_lo, ci_hi = bootstrap_from_blocks(
            values,
            frame["clock_block_id"].to_numpy(dtype=float) if not frame.empty else np.array([], dtype=float),
            RNG_SEED + stable_seed_offset(*seed_parts, "pooled", pnl_kind),
        )
        n_markets = int(frame["market_key"].nunique()) if not frame.empty else 0
    return {
        "signal": signal,
        "window_sec": window,
        "horizon_sec": horizon,
        "mapping": mapping,
        "pnl_kind": pnl_kind,
        "aggregation": aggregation,
        "segment_type": segment_type,
        "segment_value": segment_value,
        "n_base_rows": int(pre_count),
        "n_candidate_rows": int(pre_count),
        "n_trades": int(len(clean_values)),
        "n_markets": n_markets,
        "mean_pnl_bps": stats["mean_pnl_bps"],
        "ci_lo_bps": ci_lo,
        "ci_hi_bps": ci_hi,
        "median_pnl_bps": stats["median_pnl_bps"],
        "win_rate": stats["win_rate"],
        "sharpe_per_trade": stats["sharpe_per_trade"],
        "t_stat": stats["t_stat"],
        "sparse_flag": bool(len(clean_values) < MIN_REPORT_TRADES),
    }


def summarize_surface(
    trades: pd.DataFrame,
    pre_trades: pd.DataFrame,
    *,
    signal: str,
    window: int,
    horizon: int,
    mapping: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pre_counts: dict[tuple[str, str], int] = {("overall", "all"): int(len(pre_trades))}
    trade_groups: dict[tuple[str, str], pd.DataFrame] = {("overall", "all"): trades}
    for segment_type, segment_col in SEGMENTS.items():
        if segment_col is None:
            continue
        pre_values = pre_trades[segment_col].fillna("unknown").astype(str)
        for value, count in pre_values.value_counts(sort=False).items():
            pre_counts[(segment_type, str(value))] = int(count)
        if not trades.empty:
            trade_values = trades[segment_col].fillna("unknown").astype(str)
            for value, idx in trade_values.groupby(trade_values, sort=False).groups.items():
                trade_groups[(segment_type, str(value))] = trades.loc[idx]

    for pnl_kind in PNL_COLS:
        for segment_type, _segment_col in SEGMENTS.items():
            segment_values = sorted(
                value for seg_type, value in pre_counts if seg_type == segment_type
            )
            for segment_value in segment_values:
                pre_count = pre_counts[(segment_type, segment_value)]
                trade_sub = trade_groups.get((segment_type, segment_value), trades.iloc[0:0])
                base_kwargs = {
                    "pnl_kind": pnl_kind,
                    "signal": signal,
                    "window": window,
                    "horizon": horizon,
                    "mapping": mapping,
                    "segment_type": segment_type,
                    "segment_value": str(segment_value),
                    "seed_parts": (signal, window, horizon, mapping, segment_type, segment_value),
                }
                rows.append(
                    summary_row(
                        frame=trade_sub,
                        pre_count=pre_count,
                        aggregation="pooled_non_overlap",
                        **base_kwargs,
                    )
                )
                if segment_type == "overall":
                    rows.append(
                        summary_row(
                            frame=trade_sub,
                            pre_count=pre_count,
                            aggregation="market_balanced",
                            **base_kwargs,
                        )
                    )
    return rows


def summarize_by_market(
    trades: pd.DataFrame,
    pre_trades: pd.DataFrame,
    *,
    signal: str,
    window: int,
    horizon: int,
    mapping: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if pre_trades.empty:
        return rows

    def first_value(series: pd.Series) -> object:
        return series.iloc[0] if len(series) else ""

    def mode_value(series: pd.Series) -> str:
        mode = series.mode(dropna=True)
        return str(mode.iloc[0]) if not mode.empty else "unknown"

    pre_meta = (
        pre_trades.groupby("market_key", sort=False)
        .agg(
            pre_count=("market_key", "size"),
            run_id=("run_id", first_value),
            market_id=("market_id", first_value),
            slug=("slug", first_value),
            family=("family", first_value),
            spread_regime_mode=("spread_regime", mode_value),
            depth_regime_mode=("depth_regime", mode_value),
            time_to_resolution_mode=("time_to_resolution", mode_value),
        )
        .reset_index()
    )
    trade_groups = {market: group for market, group in trades.groupby("market_key", sort=False)}
    for meta in pre_meta.itertuples(index=False):
        market = str(meta.market_key)
        trade_sub = trade_groups.get(market, trades.iloc[0:0])
        for pnl_kind in PNL_COLS:
            row = summary_row(
                frame=trade_sub,
                pre_count=int(meta.pre_count),
                pnl_kind=pnl_kind,
                signal=signal,
                window=window,
                horizon=horizon,
                mapping=mapping,
                segment_type="market",
                segment_value=market,
                aggregation="pooled_non_overlap",
                seed_parts=(signal, window, horizon, mapping, market),
            )
            row.update(
                {
                    "market": market,
                    "run_id": str(meta.run_id),
                    "market_id": str(meta.market_id),
                    "slug": str(meta.slug),
                    "family": str(meta.family),
                    "spread_regime_mode": str(meta.spread_regime_mode),
                    "depth_regime_mode": str(meta.depth_regime_mode),
                    "time_to_resolution_mode": str(meta.time_to_resolution_mode),
                }
            )
            rows.append(row)
    return rows


def evaluate_mapping(
    ranked: pd.DataFrame,
    *,
    signal: str,
    window: int,
    horizon: int,
    mapping: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_ok = (
        ranked["rank_count"].ge(MIN_RANK_COUNT)
        & ranked["rank_position"].replace([np.inf, -np.inf], np.nan).notna()
    )
    horizon_ok = valid_horizon_mask(ranked, horizon)
    base = ranked.loc[rank_ok & horizon_ok].copy()
    if base.empty:
        return base, base
    if mapping == "continuous_rank":
        position = base["rank_position"].to_numpy(dtype=float)
        trade_mask = np.abs(position) > 1e-12
        score = np.abs(position)
    elif mapping == "decile_gate":
        rank_position = base["rank_position"].to_numpy(dtype=float)
        rank = (rank_position + 1.0) / 2.0
        position = np.where(rank >= 0.90, 1.0, np.where(rank <= 0.10, -1.0, 0.0))
        trade_mask = position != 0.0
        score = np.abs(rank_position)
    else:
        raise ValueError(f"unknown mapping {mapping}")

    pre = base.loc[trade_mask].copy()
    if pre.empty:
        return pre, pre
    pre_position = position[trade_mask]
    pre["nonoverlap_score"] = score[trade_mask]
    pre = executable_and_mid_pnl(pre, pre_position, horizon)
    finite = (
        pre["pnl_executable_bps"].replace([np.inf, -np.inf], np.nan).notna()
        & pre["pnl_mid_bps"].replace([np.inf, -np.inf], np.nan).notna()
    )
    pre = pre.loc[finite].copy()
    block_ns = int(BOOTSTRAP_CHUNK_SECONDS * 1_000_000_000)
    pre["clock_block_id"] = (
        pre["market_code"].to_numpy(dtype=np.int64) * 1_000_000_000
        + (pre["event_time_ns"].to_numpy(dtype=np.int64) // block_ns)
    )
    trades = apply_non_overlap(pre, horizon)
    return pre, trades


def row_count_heatmap(surface: pd.DataFrame) -> pd.DataFrame:
    overall = surface[
        surface["aggregation"].eq("pooled_non_overlap")
        & surface["segment_type"].eq("overall")
        & surface["segment_value"].eq("all")
        & surface["pnl_kind"].eq("executable")
    ].copy()
    return overall[
        [
            "signal",
            "window_sec",
            "horizon_sec",
            "mapping",
            "n_base_rows",
            "n_candidate_rows",
            "n_trades",
            "n_markets",
            "sparse_flag",
        ]
    ].sort_values(["signal", "mapping", "window_sec", "horizon_sec"])


def write_note(surface: pd.DataFrame, by_market: pd.DataFrame, heatmap: pd.DataFrame, df: pd.DataFrame) -> None:
    overall = surface[
        surface["aggregation"].eq("pooled_non_overlap")
        & surface["segment_type"].eq("overall")
        & surface["segment_value"].eq("all")
        & surface["pnl_kind"].eq("executable")
        & ~surface["sparse_flag"]
    ].copy()
    robust = overall[overall["n_trades"].ge(MIN_REPORT_TRADES)].copy()
    cont = robust[robust["mapping"].eq("continuous_rank")]
    gate = robust[robust["mapping"].eq("decile_gate")]

    joined = cont.merge(
        gate,
        on=["signal", "window_sec", "horizon_sec", "pnl_kind", "aggregation", "segment_type", "segment_value"],
        suffixes=("_continuous", "_gate"),
    )
    joined["delta_cont_minus_gate_bps"] = joined["mean_pnl_bps_continuous"] - joined["mean_pnl_bps_gate"]
    best_cont = cont.sort_values(["mean_pnl_bps", "n_trades"], ascending=False).head(1)
    best_gate = gate.sort_values(["mean_pnl_bps", "n_trades"], ascending=False).head(1)
    best_delta = joined.sort_values(["delta_cont_minus_gate_bps", "n_trades_continuous"], ascending=False).head(1)

    if joined.empty:
        headline = "No: there were not enough matched robust cells to compare continuous sizing against decile gating."
    else:
        positive_delta = int(joined["delta_cont_minus_gate_bps"].gt(0).sum())
        total_delta = int(len(joined))
        positive_overall = int(robust["mean_pnl_bps"].gt(0).sum())
        robust_positive_ci = joined[
            joined["delta_cont_minus_gate_bps"].gt(0)
            & joined["ci_lo_bps_continuous"].gt(joined["ci_hi_bps_gate"])
        ]
        if len(robust_positive_ci):
            headline = (
                "Yes on relative performance: continuous sizing beats decile gating in robust "
                "overall executable non-overlap cells with non-overlapping confidence intervals."
            )
        else:
            headline = (
                "No: continuous sizing does not beat decile gating after executable cost and "
                "non-overlap by a CI-clean standard."
            )
        headline += (
            f" Continuous mean exceeds gate mean in {positive_delta}/{total_delta} matched robust cells. "
            f"However, {positive_overall}/{len(robust)} robust overall executable cells have positive mean PnL, "
            "so this is not a standalone executable edge."
        )

    def row_desc(row: pd.Series, suffix: str = "") -> list[str]:
        return [
            str(row[f"signal{suffix}"] if f"signal{suffix}" in row else row["signal"]),
            str(int(row[f"window_sec{suffix}"] if f"window_sec{suffix}" in row else row["window_sec"])),
            str(int(row[f"horizon_sec{suffix}"] if f"horizon_sec{suffix}" in row else row["horizon_sec"])),
            bps(float(row[f"mean_pnl_bps{suffix}"] if f"mean_pnl_bps{suffix}" in row else row["mean_pnl_bps"])),
            f"[{bps(float(row[f'ci_lo_bps{suffix}'] if f'ci_lo_bps{suffix}' in row else row['ci_lo_bps']))}, "
            f"{bps(float(row[f'ci_hi_bps{suffix}'] if f'ci_hi_bps{suffix}' in row else row['ci_hi_bps']))}]",
            compact_number(row[f"n_trades{suffix}"] if f"n_trades{suffix}" in row else row["n_trades"]),
            f"{float(row[f'sharpe_per_trade{suffix}'] if f'sharpe_per_trade{suffix}' in row else row['sharpe_per_trade']):.3f}"
            if np.isfinite(float(row[f"sharpe_per_trade{suffix}"] if f"sharpe_per_trade{suffix}" in row else row["sharpe_per_trade"]))
            else "n/a",
        ]

    top_rows: list[list[str]] = []
    if not best_cont.empty:
        top_rows.append(["best continuous", *row_desc(best_cont.iloc[0])])
    if not best_gate.empty:
        top_rows.append(["best gate", *row_desc(best_gate.iloc[0])])
    if not best_delta.empty:
        r = best_delta.iloc[0]
        top_rows.append(
            [
                "best cont-gate delta",
                str(r["signal"]),
                str(int(r["window_sec"])),
                str(int(r["horizon_sec"])),
                bps(float(r["delta_cont_minus_gate_bps"])),
                f"cont {bps(float(r['mean_pnl_bps_continuous']))} vs gate {bps(float(r['mean_pnl_bps_gate']))}",
                compact_number(r["n_trades_continuous"]),
                compact_number(r["n_trades_gate"]),
            ]
        )

    market_best = by_market[
        by_market["aggregation"].eq("pooled_non_overlap")
        & by_market["pnl_kind"].eq("executable")
        & by_market["n_trades"].ge(MIN_REPORT_TRADES)
    ].copy()
    market_rows: list[list[str]] = []
    if not market_best.empty:
        market_best = (
            market_best.sort_values(["mean_pnl_bps", "n_trades"], ascending=False)
            .groupby(["mapping", "market"], as_index=False)
            .head(1)
            .sort_values(["mean_pnl_bps", "n_trades"], ascending=False)
            .head(12)
        )
        for row in market_best.itertuples(index=False):
            market_rows.append(
                [
                    str(row.mapping),
                    str(row.market),
                    safe_text(row.slug, 44),
                    str(row.signal),
                    str(int(row.window_sec)),
                    str(int(row.horizon_sec)),
                    bps(float(row.mean_pnl_bps)),
                    f"[{bps(float(row.ci_lo_bps))}, {bps(float(row.ci_hi_bps))}]",
                    compact_number(row.n_trades),
                ]
            )

    segment_rows: list[list[str]] = []
    seg = surface[
        surface["aggregation"].eq("pooled_non_overlap")
        & surface["pnl_kind"].eq("executable")
        & surface["segment_type"].isin(["family", "spread_regime", "depth_regime", "time_to_resolution", "clock_utc_6h"])
        & surface["n_trades"].ge(MIN_REPORT_TRADES)
    ].copy()
    if not seg.empty:
        seg = seg.sort_values(["mean_pnl_bps", "n_trades"], ascending=False).head(14)
        for row in seg.itertuples(index=False):
            segment_rows.append(
                [
                    str(row.segment_type),
                    str(row.segment_value),
                    str(row.mapping),
                    str(row.signal),
                    str(int(row.window_sec)),
                    str(int(row.horizon_sec)),
                    bps(float(row.mean_pnl_bps)),
                    f"[{bps(float(row.ci_lo_bps))}, {bps(float(row.ci_hi_bps))}]",
                    compact_number(row.n_trades),
                ]
            )

    heat_rows: list[list[str]] = []
    heat_summary = (
        heatmap.groupby(["signal", "mapping"], as_index=False)
        .agg(
            min_trades=("n_trades", "min"),
            median_trades=("n_trades", "median"),
            max_trades=("n_trades", "max"),
            sparse_cells=("sparse_flag", "sum"),
            cells=("sparse_flag", "size"),
        )
        .sort_values(["signal", "mapping"])
    )
    for row in heat_summary.itertuples(index=False):
        heat_rows.append(
            [
                str(row.signal),
                str(row.mapping),
                compact_number(row.min_trades),
                compact_number(row.median_trades),
                compact_number(row.max_trades),
                f"{int(row.sparse_cells)}/{int(row.cells)}",
            ]
        )

    run_ids = ", ".join(sorted(df["run_id"].unique()))
    market_count = int(df["market_key"].nunique())
    asset_count = int(df["asset_key"].nunique())
    min_ts = df["event_ts"].min()
    max_ts = df["event_ts"].max()

    note = f"""---
tags: [dali, block-p1, rolling-rank, executable-cost, non-overlap, results]
---

# Block P1 Rolling-Rank Findings

## Headline

{headline}

## Topline Cells

{markdown_table(["cell", "signal", "W", "H", "mean/delta", "CI or comparison", "n cont/total", "Sharpe or n gate"], top_rows)}

## Method

- Input: `{display_path(FEATURES)}` after complete-book and stale-book filtering (`book_staleness_seconds <= {MAX_STALE_BOOK_SECONDS:g}`), using `exchange_ts` for rank windows, non-overlap clocks, and exit alignment.
- Filtered panel: {market_count:,} markets / {asset_count:,} assets, run IDs `{run_ids}`, from `{min_ts}` to `{max_ts}`. This parquet currently contains A0/A0b rows; no A0c `run_id` appears in the input file.
- Signals: `ofi_l1` and `tfi` are direction-adjusted rolling sums over W; `tob_imbalance_level` is direction-adjusted current TOB imbalance; `weighted_mid_edge_bps` is direction-adjusted L1 weighted-mid edge.
- Rank transform: within each `(run_id, asset_id)`, compute a trailing W-second percentile rank with no future rows; rows need at least {MIN_RANK_COUNT} rank-window observations.
- Mappings: continuous uses `position = 2 * rank - 1`; decile gate uses `+1` above 90th percentile, `-1` below 10th percentile, and no trade otherwise.
- PnL rows are identical across executable and mid reporting for a given mapping cell. Executable PnL crosses entry/exit touch and applies A1 `FEE_BY_CATEGORY`; mid PnL uses the direction-adjusted mid return.
- Non-overlap: one open position per market, selected by event time then rank extremeness, with hold fixed at H seconds.
- Exit quote control: future touch/mid must be observed at or before `exchange_ts + H` and no more than {MAX_FUTURE_QUOTE_AGE_SECONDS:g}s stale.
- CIs: {BOOTSTRAP_SAMPLES}-sample 300s clock-block bootstrap for pooled rows; market-balanced rows bootstrap per-market means.

## Segment Scan

{markdown_table(["segment", "value", "mapping", "signal", "W", "H", "mean", "CI", "n"], segment_rows)}

## Per-Market Best Rows

{markdown_table(["mapping", "market", "slug", "signal", "W", "H", "mean", "CI", "n"], market_rows)}

## Row-Count Heatmap Check

{markdown_table(["signal", "mapping", "min n", "median n", "max n", "sparse cells"], heat_rows)}

The detailed heatmap-ready counts live in `{display_path(HEATMAP_OUT)}`. Sparse cells are retained in CSVs but are not used for the headline.

## Outputs

- `{display_path(SURFACE_OUT)}`: pooled, market-balanced, family/regime/time/clock segment surface.
- `{display_path(BY_MARKET_OUT)}`: market-level surface for the same signal/window/horizon/mapping grid.
- `{display_path(HEATMAP_OUT)}`: overall executable row-count heatmap by signal/window/horizon/mapping.

## Interpretation

Continuous rank sizing changes the exposure profile, but the P1 executable-cost test is still governed by spread crossing and short-horizon quote movement. Treat any positive sparse or segment-only cell as a diagnostic lead, not a deployable result, unless it survives the market-balanced rows and a fresh out-of-sample capture.
"""
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    print("load features", flush=True)
    df = load_features(FEATURES)
    print(f"filtered rows: {len(df):,}", flush=True)
    print("add future quotes", flush=True)
    df = add_future_quotes(df)

    surface_rows: list[dict[str, object]] = []
    market_rows: list[dict[str, object]] = []
    total_cells = len(SIGNALS) * len(WINDOWS)
    cell_idx = 0
    for signal in SIGNALS:
        for window in WINDOWS:
            cell_idx += 1
            print(f"signal/window {cell_idx:02d}/{total_cells:02d}: {signal} W={window}s", flush=True)
            ranked = add_rank_for_signal(df, signal, window)
            for horizon in HORIZONS:
                for mapping in MAPPINGS:
                    pre, trades = evaluate_mapping(
                        ranked,
                        signal=signal,
                        window=window,
                        horizon=horizon,
                        mapping=mapping,
                    )
                    print(
                        f"  {signal} W={window}s H={horizon}s {mapping}: "
                        f"pre={len(pre):,} nonoverlap={len(trades):,}",
                        flush=True,
                    )
                    if pre.empty:
                        continue
                    surface_rows.extend(
                        summarize_surface(
                            trades,
                            pre,
                            signal=signal,
                            window=window,
                            horizon=horizon,
                            mapping=mapping,
                        )
                    )
                    market_rows.extend(
                        summarize_by_market(
                            trades,
                            pre,
                            signal=signal,
                            window=window,
                            horizon=horizon,
                            mapping=mapping,
                        )
                    )

    surface = pd.DataFrame(surface_rows)
    by_market = pd.DataFrame(market_rows)
    heatmap = row_count_heatmap(surface)
    SURFACE_OUT.parent.mkdir(parents=True, exist_ok=True)
    surface.to_csv(SURFACE_OUT, index=False)
    by_market.to_csv(BY_MARKET_OUT, index=False)
    heatmap.to_csv(HEATMAP_OUT, index=False)
    write_note(surface, by_market, heatmap, df)
    print(f"wrote {display_path(SURFACE_OUT)} ({len(surface):,} rows)", flush=True)
    print(f"wrote {display_path(BY_MARKET_OUT)} ({len(by_market):,} rows)", flush=True)
    print(f"wrote {display_path(HEATMAP_OUT)} ({len(heatmap):,} rows)", flush=True)
    print(f"wrote {display_path(NOTE)}", flush=True)


if __name__ == "__main__":
    main()
