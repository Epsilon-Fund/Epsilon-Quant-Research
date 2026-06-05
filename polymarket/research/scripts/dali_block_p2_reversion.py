"""P2 explicit mean-reversion-to-microprice execution test.

This sidecar tests the framing left open by A1.x: fade extreme local book
signals and exit toward the current weighted mid/microprice, instead of
riding the signal as directional continuation.

No raw captures or canonical A1 artifacts are modified.
"""
from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from dali_block_a1_analyze import FEE_BY_CATEGORY, family_category
from dali_block_a14c_maker_at_mid import REBATE_PCT_BY_CATEGORY


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
FEATURES = ANALYSIS / "block_a1_features.parquet"
A1_RESULTS = ANALYSIS / "csv_outputs" / "dali" / "block_a1_results.csv"
SURFACE_OUT = ANALYSIS / "csv_outputs" / "dali" / "p2_reversion_surface.csv"
FRONTIER_OUT = ANALYSIS / "csv_outputs" / "dali" / "p2_reversion_passive_fillfrontier.csv"
NOTE = NOTES / "block_p2_reversion_findings.md"

TARGET_TYPES = ("micro_price", "half_to_micro_price")
TIMEOUT_SECONDS = (5, 10, 30, 60)
PASSIVE_FILL_WINDOWS = (1, 5, 10)
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260530
MIN_CI_N = 5
ROBUST_N = 30
STOP_SPREAD_MULTIPLIER = 1.0


@dataclass(frozen=True)
class SignalSpec:
    name: str
    horizon_sec: int
    source: str


@dataclass(frozen=True)
class AssetState:
    times: np.ndarray
    bids: np.ndarray
    asks: np.ndarray


SIGNALS = (
    SignalSpec("tob_imbalance_level", 0, "tob"),
    SignalSpec("ofi_5s", 5, "ofi"),
)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_text(value: object, max_len: int = 52) -> str:
    text = str(value if value is not None else "").replace("|", "/")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "."


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def stable_seed_offset(*parts: object, modulo: int = 100_000) -> int:
    text = "|".join(str(part) for part in parts)
    return int(zlib.crc32(text.encode("utf-8")) % modulo)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def qbucket(series: pd.Series, labels: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return out
    try:
        bucketed = pd.qcut(valid.rank(method="first"), len(labels), labels=labels)
    except ValueError:
        return out
    out.loc[bucketed.index] = bucketed.astype(str)
    return out


def fee_amount(category: str, price: float | np.ndarray) -> float | np.ndarray:
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = np.clip(price, 0.0, 1.0)
    return params["fee_rate"] * p * (1.0 - p)


def maker_rebate_amount(category: str, price: float) -> float:
    pct_rebate = REBATE_PCT_BY_CATEGORY.get(category, REBATE_PCT_BY_CATEGORY["Other"])
    return float(fee_amount(category, price) * pct_rebate)


def load_market_labels() -> pd.DataFrame:
    if not A1_RESULTS.exists():
        return pd.DataFrame(
            columns=["run_id", "market_id", "sample_size_label", "a1_mean_depth_at_touch"]
        )
    results = pd.read_csv(A1_RESULTS, dtype={"run_id": str, "market_id": str})
    label_priority = {"primary_read": 0, "thin_wide_CI": 1}
    results["label_priority"] = results["sample_size_label"].map(label_priority).fillna(9)
    labels = (
        results.sort_values(["run_id", "market_id", "label_priority", "horizon_sec"])
        .drop_duplicates(["run_id", "market_id"])
        .rename(columns={"mean_depth_at_touch": "a1_mean_depth_at_touch"})
    )
    keep = ["run_id", "market_id", "sample_size_label", "a1_mean_depth_at_touch"]
    return labels[[col for col in keep if col in labels.columns]]


def load_features(path: Path = FEATURES) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing input features: {display_path(path)}")
    cols = [
        "run_id",
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
        "tob_imbalance",
        "ofi_combined_event",
        "trade_price",
        "trade_side",
        "last_trade_side",
        "trade_size",
    ]
    con = duckdb.connect()
    select_cols = ", ".join(cols)
    df = con.execute(f"SELECT {select_cols} FROM read_parquet('{path}')").df()
    con.close()
    if df.empty:
        raise SystemExit(f"no rows found in {display_path(path)}")

    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    for col in ("run_id", "event_type", "asset_id", "market_id", "family", "slug", "question"):
        df[col] = df[col].fillna("").astype(str)
    numeric = [
        "outcome_index",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "tob_imbalance",
        "ofi_combined_event",
        "trade_price",
        "trade_size",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_book_state_complete"] = df["is_book_state_complete"].fillna(False).astype(bool)
    df["market"] = df["run_id"] + ":" + df["market_id"]
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    df["touch_depth"] = df[["best_bid_size", "best_ask_size"]].sum(axis=1, min_count=2)
    df["spread_bps"] = np.where(
        df["mid"].gt(0) & df["spread"].notna(),
        df["spread"] / df["mid"] * 10_000.0,
        np.nan,
    )
    size_sum = df["best_bid_size"] + df["best_ask_size"]
    df["weighted_mid"] = np.where(
        size_sum.gt(0)
        & df["best_bid"].notna()
        & df["best_ask"].notna()
        & df["best_bid_size"].notna()
        & df["best_ask_size"].notna(),
        (df["best_ask"] * df["best_bid_size"] + df["best_bid"] * df["best_ask_size"]) / size_sum,
        df["mid"] + 0.5 * df["spread"] * df["tob_imbalance"],
    )
    df["weighted_mid"] = df["weighted_mid"].clip(lower=0.0, upper=1.0)
    df["micro_edge_bps"] = np.where(
        df["mid"].gt(0) & df["weighted_mid"].notna(),
        (df["weighted_mid"] - df["mid"]) / df["mid"] * 10_000.0,
        np.nan,
    )
    df["trade_side_norm"] = (
        df["trade_side"]
        .fillna(df["last_trade_side"])
        .fillna("")
        .astype(str)
        .str.upper()
    )

    market_depth = (
        df.groupby(["run_id", "market_id"], as_index=False)["touch_depth"]
        .mean()
        .rename(columns={"touch_depth": "market_mean_depth"})
    )
    df = df.merge(market_depth, on=["run_id", "market_id"], how="left")
    df["relative_depth"] = df["touch_depth"] / df["market_mean_depth"]
    spread_labels = ["spread_d01_tight"] + [f"spread_d{i:02d}" for i in range(2, 10)] + [
        "spread_d10_wide"
    ]
    depth_labels = ["depth_d01_shallow"] + [f"depth_d{i:02d}" for i in range(2, 10)] + [
        "depth_d10_deep"
    ]
    df["spread_decile"] = qbucket(df["spread_bps"], spread_labels).fillna("spread_unknown")
    df["depth_decile"] = qbucket(df["relative_depth"], depth_labels).fillna("depth_unknown")

    labels = load_market_labels()
    if not labels.empty:
        df = df.merge(labels, on=["run_id", "market_id"], how="left")
    else:
        df["sample_size_label"] = ""
        df["a1_mean_depth_at_touch"] = np.nan
    df["sample_size_label"] = df["sample_size_label"].fillna("unlabeled")
    return df.sort_values(["run_id", "market_id", "asset_id", "received_at"]).reset_index(drop=True)


def add_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    groups = list(df.groupby(["run_id", "market_id", "asset_id"], sort=False))
    for idx, ((run_id, market_id, asset_id), group) in enumerate(groups, start=1):
        if idx % 20 == 0:
            print(f"signal features {idx:,}/{len(groups):,}", flush=True)
        g = group.sort_values("received_at").copy()
        g["tob_imbalance_ffill"] = g["tob_imbalance"].ffill()
        g["signal_tob_imbalance_level"] = g["direction_factor"] * g["tob_imbalance_ffill"]
        g["ofi_combined_event"] = g["ofi_combined_event"].fillna(0.0)
        g = g.set_index("received_at", drop=False)
        mean_depth = float(g["market_mean_depth"].replace([np.inf, -np.inf], np.nan).mean())
        if not np.isfinite(mean_depth) or mean_depth <= 0:
            mean_depth = 1.0
        g["signal_ofi_5s"] = (
            g["direction_factor"].to_numpy(dtype=float)
            * g["ofi_combined_event"].rolling("5s").sum().to_numpy(dtype=float)
            / mean_depth
        )
        pieces.append(g.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def build_signal_events(df: pd.DataFrame) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    quote_ok = (
        df["is_book_state_complete"]
        & df["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & df["mid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["weighted_mid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["spread"].replace([np.inf, -np.inf], np.nan).gt(0)
    )
    base_cols = [
        "run_id",
        "market_id",
        "market",
        "received_at",
        "asset_id",
        "family",
        "slug",
        "question",
        "outcome_index",
        "sample_size_label",
        "direction_factor",
        "best_bid",
        "best_ask",
        "spread",
        "spread_bps",
        "mid",
        "weighted_mid",
        "micro_edge_bps",
        "touch_depth",
        "relative_depth",
        "spread_decile",
        "depth_decile",
    ]
    for spec in SIGNALS:
        sig_col = f"signal_{spec.name}"
        valid = quote_ok & df[sig_col].replace([np.inf, -np.inf], np.nan).notna() & df[sig_col].ne(0.0)
        pieces: list[pd.DataFrame] = []
        for _, group in df.loc[valid, base_cols + [sig_col]].groupby(["run_id", "market_id"], sort=False):
            if len(group) < 20:
                continue
            q10 = float(group[sig_col].quantile(0.10))
            q90 = float(group[sig_col].quantile(0.90))
            tails = group[group[sig_col].le(q10) | group[sig_col].ge(q90)].copy()
            if tails.empty:
                continue
            tails["signal_variant"] = spec.name
            tails["signal_horizon_sec"] = spec.horizon_sec
            tails["signal_value"] = tails[sig_col].astype(float)
            tails["abs_signal"] = tails["signal_value"].abs()
            tails["signal_tail"] = np.where(tails["signal_value"].ge(q90), "top_decile", "bottom_decile")
            tails["signal_decile_threshold_low"] = q10
            tails["signal_decile_threshold_high"] = q90
            # Continuation token side is sign(market_signal) * direction_factor.
            # P2 tests the fade, so it takes the opposite token exposure.
            tails["token_side"] = -np.sign(tails["signal_value"].to_numpy(dtype=float)) * tails[
                "direction_factor"
            ].to_numpy(dtype=float)
            tails = tails[tails["token_side"].isin([-1.0, 1.0])].copy()
            pieces.append(tails.drop(columns=[sig_col]))
        if pieces:
            out.append(pd.concat(pieces, ignore_index=True))
    if not out:
        raise SystemExit("no top/bottom-decile signal events found")
    events = pd.concat(out, ignore_index=True)
    events["event_time_ns"] = events["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    events["event_id"] = np.arange(len(events), dtype=np.int64)
    return events.sort_values(["signal_variant", "market", "event_time_ns", "asset_id"]).reset_index(drop=True)


def first_qualifying_trade(
    start_ns: int,
    end_ns: int,
    quote_price: float,
    desired_trade_side: str,
    price_relation: str,
    trade_times: np.ndarray,
    trade_prices: np.ndarray,
    trade_sides: np.ndarray,
) -> int | None:
    left = int(np.searchsorted(trade_times, start_ns, side="left"))
    right = int(np.searchsorted(trade_times, end_ns, side="right"))
    if left >= right:
        return None
    for idx in range(left, right):
        if trade_sides[idx] != desired_trade_side:
            continue
        px = trade_prices[idx]
        if not np.isfinite(px):
            continue
        if price_relation == "le" and px <= quote_price + 1e-12:
            return int(trade_times[idx])
        if price_relation == "ge" and px >= quote_price - 1e-12:
            return int(trade_times[idx])
    return None


def passive_entry_candidates(events: pd.DataFrame, market: pd.DataFrame, fill_window_sec: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for asset_id, rows in events.groupby("asset_id", sort=False):
        asset_rows = market[market["asset_id"].eq(asset_id)].sort_values("received_at")
        trades = asset_rows[
            asset_rows["event_type"].eq("last_trade_price")
            & asset_rows["trade_side_norm"].isin(["BUY", "SELL"])
            & asset_rows["trade_price"].notna()
        ].copy()
        trade_times = trades["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        trade_prices = trades["trade_price"].to_numpy(dtype=float)
        trade_sides = trades["trade_side_norm"].to_numpy(dtype=object)
        piece = rows.copy()
        fill_time = np.full(len(piece), np.nan, dtype=float)
        entry_price = np.where(piece["token_side"].to_numpy(dtype=float) > 0, piece["best_bid"], piece["best_ask"])
        if len(trade_times):
            for pos, row in enumerate(piece.itertuples(index=False)):
                start = int(row.event_time_ns)
                end = start + fill_window_sec * 1_000_000_000
                if row.token_side > 0:
                    found = first_qualifying_trade(
                        start,
                        end,
                        float(row.best_bid),
                        "SELL",
                        "le",
                        trade_times,
                        trade_prices,
                        trade_sides,
                    )
                else:
                    found = first_qualifying_trade(
                        start,
                        end,
                        float(row.best_ask),
                        "BUY",
                        "ge",
                        trade_times,
                        trade_prices,
                        trade_sides,
                    )
                if found is not None:
                    fill_time[pos] = float(found)
        piece["entry_filled"] = np.isfinite(fill_time)
        piece["entry_time_ns"] = fill_time
        piece["entry_price"] = entry_price
        piece["entry_liquidity"] = "maker"
        parts.append(piece)
    return pd.concat(parts, ignore_index=True) if parts else events.iloc[0:0].copy()


def taker_entry_candidates(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["entry_filled"] = True
    out["entry_time_ns"] = out["event_time_ns"].astype(float)
    out["entry_price"] = np.where(out["token_side"].to_numpy(dtype=float) > 0, out["best_ask"], out["best_bid"])
    out["entry_liquidity"] = "taker"
    return out


def target_price(row: pd.Series, target_type: str) -> float:
    micro = float(row["weighted_mid"])
    entry = float(row["entry_price"])
    return target_price_value(entry, micro, target_type)


def target_price_value(entry: float, micro: float, target_type: str) -> float:
    if target_type == "micro_price":
        return micro
    if target_type == "half_to_micro_price":
        return entry + 0.5 * (micro - entry)
    raise ValueError(f"unknown target_type={target_type}")


def build_asset_state(asset_rows: pd.DataFrame) -> AssetState | None:
    state = asset_rows[
        asset_rows["is_book_state_complete"]
        & asset_rows["best_bid"].notna()
        & asset_rows["best_ask"].notna()
        & asset_rows["mid"].notna()
    ].sort_values("received_at")
    if state.empty:
        return None
    return AssetState(
        times=state["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64"),
        bids=state["best_bid"].to_numpy(dtype=float),
        asks=state["best_ask"].to_numpy(dtype=float),
    )


def simulate_exit_one(
    row: pd.Series,
    state: AssetState,
    target_type: str,
    timeout_sec: int,
    category: str,
    execution_mode: str,
) -> dict[str, object] | None:
    times = state.times
    bids = state.bids
    asks = state.asks
    entry_ns = int(row["entry_time_ns"])
    timeout_ns = entry_ns + timeout_sec * 1_000_000_000
    left = int(np.searchsorted(times, entry_ns, side="right"))
    right = int(np.searchsorted(times, timeout_ns, side="right") - 1)
    if right < left or left >= len(times):
        return None
    right = min(right, len(times) - 1)

    side = float(row["token_side"])
    entry = float(row["entry_price"])
    tgt = float(target_price(row, target_type))
    spread = float(row["spread"])
    stop = entry - side * STOP_SPREAD_MULTIPLIER * spread
    stop = float(np.clip(stop, 0.0, 1.0))

    if side > 0:
        touch = bids[left : right + 1]
        target_hits = np.flatnonzero(touch >= tgt - 1e-12)
        stop_hits = np.flatnonzero(touch <= stop + 1e-12)
    else:
        touch = asks[left : right + 1]
        target_hits = np.flatnonzero(touch <= tgt + 1e-12)
        stop_hits = np.flatnonzero(touch >= stop - 1e-12)

    target_idx = left + int(target_hits[0]) if len(target_hits) else None
    stop_idx = left + int(stop_hits[0]) if len(stop_hits) else None
    if target_idx is not None and (stop_idx is None or target_idx <= stop_idx):
        exit_idx = target_idx
        reason = "target_reached"
    elif stop_idx is not None:
        exit_idx = stop_idx
        reason = "adverse_stop"
    else:
        exit_idx = right
        reason = "timeout"

    exit_price = float(bids[exit_idx] if side > 0 else asks[exit_idx])
    exit_ns = int(times[exit_idx]) if reason != "timeout" else timeout_ns
    if exit_ns <= entry_ns or not np.isfinite(exit_price) or not np.isfinite(entry) or entry <= 0:
        return None

    gross = side * (exit_price - entry)
    entry_fee = float(fee_amount(category, entry)) if execution_mode == "T" else 0.0
    entry_rebate = maker_rebate_amount(category, entry) if execution_mode == "P" else 0.0
    exit_fee = float(fee_amount(category, exit_price))
    pnl = (gross - entry_fee - exit_fee + entry_rebate) / np.clip(entry, 0.01, 0.99) * 10_000.0
    return {
        "entry_time_ns_int": entry_ns,
        "exit_time_ns": float(exit_ns),
        "exit_time_ns_int": exit_ns,
        "entry_price": entry,
        "exit_price": exit_price,
        "target_price": tgt,
        "stop_price": stop,
        "target_edge_bps": side * (tgt - entry) / np.clip(entry, 0.01, 0.99) * 10_000.0,
        "exit_reason": reason,
        "hold_seconds": (exit_ns - entry_ns) / 1_000_000_000.0,
        "gross_bps": gross / np.clip(entry, 0.01, 0.99) * 10_000.0,
        "entry_fee_bps": entry_fee / np.clip(entry, 0.01, 0.99) * 10_000.0,
        "entry_rebate_bps": entry_rebate / np.clip(entry, 0.01, 0.99) * 10_000.0,
        "exit_fee_bps": exit_fee / np.clip(entry, 0.01, 0.99) * 10_000.0,
        "pnl_bps": float(pnl),
    }


def simulate_exit_values(
    *,
    state: AssetState,
    entry_ns: int,
    side: float,
    entry: float,
    micro: float,
    spread: float,
    target_type: str,
    timeout_sec: int,
    category: str,
    execution_mode: str,
) -> dict[str, object] | None:
    times = state.times
    bids = state.bids
    asks = state.asks
    timeout_ns = entry_ns + timeout_sec * 1_000_000_000
    left = int(np.searchsorted(times, entry_ns, side="right"))
    right = int(np.searchsorted(times, timeout_ns, side="right") - 1)
    if right < left or left >= len(times):
        return None
    right = min(right, len(times) - 1)

    tgt = float(target_price_value(entry, micro, target_type))
    stop = float(np.clip(entry - side * STOP_SPREAD_MULTIPLIER * spread, 0.0, 1.0))
    if side > 0:
        touch = bids[left : right + 1]
        target_hits = np.flatnonzero(touch >= tgt - 1e-12)
        stop_hits = np.flatnonzero(touch <= stop + 1e-12)
    else:
        touch = asks[left : right + 1]
        target_hits = np.flatnonzero(touch <= tgt + 1e-12)
        stop_hits = np.flatnonzero(touch >= stop - 1e-12)

    target_idx = left + int(target_hits[0]) if len(target_hits) else None
    stop_idx = left + int(stop_hits[0]) if len(stop_hits) else None
    if target_idx is not None and (stop_idx is None or target_idx <= stop_idx):
        exit_idx = target_idx
        reason = "target_reached"
    elif stop_idx is not None:
        exit_idx = stop_idx
        reason = "adverse_stop"
    else:
        exit_idx = right
        reason = "timeout"

    exit_price = float(bids[exit_idx] if side > 0 else asks[exit_idx])
    exit_ns = int(times[exit_idx]) if reason != "timeout" else timeout_ns
    if exit_ns <= entry_ns or not np.isfinite(exit_price) or not np.isfinite(entry) or entry <= 0:
        return None

    denom = float(np.clip(entry, 0.01, 0.99))
    gross = side * (exit_price - entry)
    entry_fee = float(fee_amount(category, entry)) if execution_mode == "T" else 0.0
    entry_rebate = maker_rebate_amount(category, entry) if execution_mode == "P" else 0.0
    exit_fee = float(fee_amount(category, exit_price))
    pnl = (gross - entry_fee - exit_fee + entry_rebate) / denom * 10_000.0
    return {
        "entry_time_ns_int": entry_ns,
        "exit_time_ns": float(exit_ns),
        "exit_time_ns_int": exit_ns,
        "entry_price": entry,
        "exit_price": exit_price,
        "target_price": tgt,
        "stop_price": stop,
        "target_edge_bps": side * (tgt - entry) / denom * 10_000.0,
        "exit_reason": reason,
        "hold_seconds": (exit_ns - entry_ns) / 1_000_000_000.0,
        "gross_bps": gross / denom * 10_000.0,
        "entry_fee_bps": entry_fee / denom * 10_000.0,
        "entry_rebate_bps": entry_rebate / denom * 10_000.0,
        "exit_fee_bps": exit_fee / denom * 10_000.0,
        "pnl_bps": float(pnl),
    }


def execute_non_overlap(
    candidates: pd.DataFrame,
    market: pd.DataFrame,
    target_type_name: str,
    timeout_sec: int,
    execution_mode: str,
) -> pd.DataFrame:
    filled = candidates[
        candidates["entry_filled"].fillna(False)
        & candidates["entry_time_ns"].replace([np.inf, -np.inf], np.nan).notna()
        & candidates["entry_price"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    if filled.empty:
        return filled.iloc[0:0].copy()
    category = family_category(str(market["family"].replace("", np.nan).dropna().iloc[0]))
    asset_state_by_id = {}
    for asset_id, rows in market.groupby("asset_id", sort=False):
        state = build_asset_state(rows)
        if state is not None:
            asset_state_by_id[asset_id] = state
    if execution_mode == "T":
        filled = filled.sort_values(["event_time_ns", "abs_signal"], ascending=[True, False]).reset_index(drop=True)
    else:
        filled["fill_time_ns_int"] = filled["entry_time_ns"].astype("int64")
        filled = filled.sort_values(
            ["fill_time_ns_int", "event_time_ns", "abs_signal"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    open_until_ns = -1
    order_times = (
        filled["event_time_ns"].to_numpy(dtype=np.int64)
        if execution_mode == "T"
        else filled["fill_time_ns_int"].to_numpy(dtype=np.int64)
    )
    signal_times = filled["event_time_ns"].to_numpy(dtype=np.int64)
    entry_times = filled["entry_time_ns"].to_numpy(dtype=np.float64).astype(np.int64)
    asset_ids = filled["asset_id"].astype(str).to_numpy()
    token_sides = filled["token_side"].to_numpy(dtype=float)
    entry_prices = filled["entry_price"].to_numpy(dtype=float)
    micros = filled["weighted_mid"].to_numpy(dtype=float)
    spreads = filled["spread"].to_numpy(dtype=float)
    keep_idx: list[int] = []
    exit_rows: list[dict[str, object]] = []
    idx = 0
    while idx < len(filled):
        signal_ns = int(signal_times[idx])
        entry_ns = int(entry_times[idx])
        if signal_ns <= open_until_ns or entry_ns <= open_until_ns:
            idx += 1
            continue
        state = asset_state_by_id.get(asset_ids[idx])
        if state is None:
            idx += 1
            continue
        exit_info = simulate_exit_values(
            state=state,
            entry_ns=entry_ns,
            side=float(token_sides[idx]),
            entry=float(entry_prices[idx]),
            micro=float(micros[idx]),
            spread=float(spreads[idx]),
            target_type=target_type_name,
            timeout_sec=timeout_sec,
            category=category,
            execution_mode=execution_mode,
        )
        if exit_info is None:
            idx += 1
            continue
        exit_ns = int(exit_info["exit_time_ns_int"])
        if exit_ns <= entry_ns:
            idx += 1
            continue
        keep_idx.append(idx)
        exit_info["category"] = category
        exit_rows.append(exit_info)
        open_until_ns = exit_ns
        idx = int(np.searchsorted(order_times, open_until_ns + 1, side="left"))
    if not keep_idx:
        return filled.iloc[0:0].copy()
    out = filled.iloc[keep_idx].copy().reset_index(drop=True)
    exit_df = pd.DataFrame(exit_rows)
    for col in exit_df.columns:
        out[col] = exit_df[col].to_numpy()
    return out


def augment_tails(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return pd.concat([df, df.assign(signal_tail="both_tails")], ignore_index=True)


def segment_count_rows(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    src = augment_tails(df)
    configs = [
        ("all", []),
        ("family", ["family"]),
        ("market", ["market", "family", "slug"]),
        ("spread_decile", ["spread_decile"]),
        ("depth_decile", ["depth_decile"]),
        ("family_spread_decile", ["family", "spread_decile"]),
        ("market_spread_decile", ["market", "family", "slug", "spread_decile"]),
        ("market_depth_decile", ["market", "family", "slug", "depth_decile"]),
    ]
    parts: list[pd.DataFrame] = []
    for segment_type, cols in configs:
        group_cols = ["signal_variant", "signal_horizon_sec", "signal_tail", *cols]
        grouped = src.groupby(group_cols, dropna=False).size().reset_index(name=value_name)
        grouped["segment_type"] = segment_type
        grouped["market"] = grouped["market"] if "market" in grouped else "ALL"
        grouped["slug"] = grouped["slug"] if "slug" in grouped else "ALL"
        grouped["family"] = grouped["family"] if "family" in grouped else "ALL"
        if "spread_decile" in grouped:
            grouped["regime_type"] = "spread_decile"
            grouped["regime_value"] = grouped["spread_decile"]
        elif "depth_decile" in grouped:
            grouped["regime_type"] = "depth_decile"
            grouped["regime_value"] = grouped["depth_decile"]
        else:
            grouped["regime_type"] = "all"
            grouped["regime_value"] = "all"
        parts.append(
            grouped[
                [
                    "signal_variant",
                    "signal_horizon_sec",
                    "signal_tail",
                    "segment_type",
                    "market",
                    "slug",
                    "family",
                    "regime_type",
                    "regime_value",
                    value_name,
                ]
            ]
        )
    return pd.concat(parts, ignore_index=True)


def bootstrap_mean_ci(rows: pd.DataFrame, seed: int) -> tuple[float, float]:
    del seed
    clean = rows[["entry_time_ns_int", "pnl_bps"]].dropna().copy()
    clean = clean[np.isfinite(clean["pnl_bps"])]
    if len(clean) < MIN_CI_N:
        return math.nan, math.nan
    mean = float(clean["pnl_bps"].mean())
    elapsed = (clean["entry_time_ns_int"] - clean["entry_time_ns_int"].min()) / 1_000_000_000.0
    block_id = (elapsed // BOOTSTRAP_CHUNK_SECONDS).astype(int).to_numpy()
    block_means = pd.Series(clean["pnl_bps"].to_numpy(dtype=float)).groupby(block_id).mean()
    if len(block_means) >= 2:
        se = float(block_means.std(ddof=1) / math.sqrt(len(block_means)))
    elif len(clean) >= MIN_CI_N:
        se = float(clean["pnl_bps"].std(ddof=1) / math.sqrt(len(clean)))
    else:
        return math.nan, math.nan
    if not np.isfinite(se):
        return math.nan, math.nan
    return mean - 1.96 * se, mean + 1.96 * se


def segment_stat_rows(df: pd.DataFrame, seed_base: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    src = augment_tails(df)
    configs = [
        ("all", []),
        ("family", ["family"]),
        ("market", ["market", "family", "slug"]),
        ("spread_decile", ["spread_decile"]),
        ("depth_decile", ["depth_decile"]),
        ("family_spread_decile", ["family", "spread_decile"]),
        ("market_spread_decile", ["market", "family", "slug", "spread_decile"]),
        ("market_depth_decile", ["market", "family", "slug", "depth_decile"]),
    ]
    rows: list[dict[str, object]] = []
    for segment_type, cols in configs:
        group_cols = ["signal_variant", "signal_horizon_sec", "signal_tail", *cols]
        for key, sub in src.groupby(group_cols, dropna=False, sort=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            key_map = dict(zip(group_cols, key_tuple, strict=True))
            ci_lo, ci_hi = bootstrap_mean_ci(
                sub,
                seed_base + stable_seed_offset(segment_type, *key_tuple, modulo=50_000),
            )
            counts = sub["exit_reason"].value_counts().to_dict()
            row = {
                "signal_variant": key_map["signal_variant"],
                "signal_horizon_sec": int(key_map["signal_horizon_sec"]),
                "signal_tail": key_map["signal_tail"],
                "segment_type": segment_type,
                "market": key_map.get("market", "ALL"),
                "slug": key_map.get("slug", "ALL"),
                "family": key_map.get("family", "ALL"),
                "regime_type": "all",
                "regime_value": "all",
                "n_executed": int(len(sub)),
                "mean_pnl_bps": float(sub["pnl_bps"].mean()),
                "median_pnl_bps": float(sub["pnl_bps"].median()),
                "win_rate": float(sub["pnl_bps"].gt(0).mean()),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "mean_target_edge_bps": float(sub["target_edge_bps"].mean()),
                "mean_gross_bps": float(sub["gross_bps"].mean()),
                "mean_entry_fee_bps": float(sub["entry_fee_bps"].mean()),
                "mean_entry_rebate_bps": float(sub["entry_rebate_bps"].mean()),
                "mean_exit_fee_bps": float(sub["exit_fee_bps"].mean()),
                "mean_hold_seconds": float(sub["hold_seconds"].mean()),
                "target_reached_rate": float(sub["exit_reason"].eq("target_reached").mean()),
                "adverse_stop_rate": float(sub["exit_reason"].eq("adverse_stop").mean()),
                "timeout_rate": float(sub["exit_reason"].eq("timeout").mean()),
                "n_target_reached": int(counts.get("target_reached", 0)),
                "n_adverse_stop": int(counts.get("adverse_stop", 0)),
                "n_timeout": int(counts.get("timeout", 0)),
            }
            if "spread_decile" in key_map:
                row["regime_type"] = "spread_decile"
                row["regime_value"] = key_map["spread_decile"]
            elif "depth_decile" in key_map:
                row["regime_type"] = "depth_decile"
                row["regime_value"] = key_map["depth_decile"]
            rows.append(row)
    return pd.DataFrame(rows)


SEGMENT_KEYS = [
    "signal_variant",
    "signal_horizon_sec",
    "signal_tail",
    "segment_type",
    "market",
    "slug",
    "family",
    "regime_type",
    "regime_value",
]


def summarize_combo(
    signals: pd.DataFrame,
    entry_candidates: pd.DataFrame,
    executed: pd.DataFrame,
    *,
    execution_mode: str,
    fill_window_sec: int,
    target_type_name: str,
    timeout_sec: int,
    seed_base: int,
) -> pd.DataFrame:
    denominators = segment_count_rows(signals, "n_signal_events")
    filled_counts = segment_count_rows(entry_candidates[entry_candidates["entry_filled"]], "n_entry_filled_raw")
    stats = segment_stat_rows(executed, seed_base)
    out = denominators.merge(filled_counts, on=SEGMENT_KEYS, how="left")
    out = out.merge(stats, on=SEGMENT_KEYS, how="left")
    out["n_entry_filled_raw"] = out["n_entry_filled_raw"].fillna(0).astype(int)
    out["n_executed"] = out["n_executed"].fillna(0).astype(int)
    out["raw_entry_fill_rate"] = out["n_entry_filled_raw"] / out["n_signal_events"].replace(0, np.nan)
    out["executed_fill_rate"] = out["n_executed"] / out["n_signal_events"].replace(0, np.nan)
    out["nonoverlap_keep_rate_after_fill"] = out["n_executed"] / out["n_entry_filled_raw"].replace(0, np.nan)
    out["execution_mode"] = execution_mode
    out["fill_window_sec"] = fill_window_sec
    out["target_type"] = target_type_name
    out["timeout_sec"] = timeout_sec
    out["survives_ci_gt_0"] = out["ci_lo"].gt(0) & out["n_executed"].ge(MIN_CI_N)
    out["survives_robust30_ci_gt_0"] = out["ci_lo"].gt(0) & out["n_executed"].ge(ROBUST_N)
    front = [
        "execution_mode",
        "fill_window_sec",
        "target_type",
        "timeout_sec",
        *SEGMENT_KEYS,
        "n_signal_events",
        "n_entry_filled_raw",
        "n_executed",
        "raw_entry_fill_rate",
        "executed_fill_rate",
        "nonoverlap_keep_rate_after_fill",
        "mean_pnl_bps",
        "median_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "survives_ci_gt_0",
        "survives_robust30_ci_gt_0",
    ]
    rest = [col for col in out.columns if col not in front]
    return out[front + rest]


def run_grid(df: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    surface_parts: list[pd.DataFrame] = []
    market_groups = {market: rows.copy() for market, rows in df.groupby("market", sort=False)}
    signal_denoms = events.groupby("signal_variant", sort=False)
    total_work = len(SIGNALS) * (len(TARGET_TYPES) * len(TIMEOUT_SECONDS) * (1 + len(PASSIVE_FILL_WINDOWS)))
    work_idx = 0

    for signal_variant, signal_events in signal_denoms:
        signal_events = signal_events.copy()
        print(f"signal {signal_variant}: {len(signal_events):,} top/bottom-decile events", flush=True)

        taker_entries = taker_entry_candidates(signal_events)
        for target in TARGET_TYPES:
            for timeout in TIMEOUT_SECONDS:
                work_idx += 1
                print(
                    f"P2 grid {work_idx}/{total_work}: {signal_variant} T target={target} timeout={timeout}s",
                    flush=True,
                )
                executed_parts: list[pd.DataFrame] = []
                for market, rows in taker_entries.groupby("market", sort=False):
                    executed = execute_non_overlap(
                        rows,
                        market_groups[market],
                        target,
                        timeout,
                        "T",
                    )
                    if not executed.empty:
                        executed_parts.append(executed)
                executed_all = pd.concat(executed_parts, ignore_index=True) if executed_parts else taker_entries.iloc[0:0]
                surface_parts.append(
                    summarize_combo(
                        signal_events,
                        taker_entries,
                        executed_all,
                        execution_mode="T",
                        fill_window_sec=0,
                        target_type_name=target,
                        timeout_sec=timeout,
                        seed_base=RNG_SEED + stable_seed_offset(signal_variant, "T", target, timeout),
                    )
                )

        passive_entries_by_window: dict[int, pd.DataFrame] = {}
        for fill_window in PASSIVE_FILL_WINDOWS:
            fill_parts: list[pd.DataFrame] = []
            for market, rows in signal_events.groupby("market", sort=False):
                filled = passive_entry_candidates(rows, market_groups[market], fill_window)
                fill_parts.append(filled)
            passive_entries_by_window[fill_window] = pd.concat(fill_parts, ignore_index=True)

        for fill_window, passive_entries in passive_entries_by_window.items():
            for target in TARGET_TYPES:
                for timeout in TIMEOUT_SECONDS:
                    work_idx += 1
                    print(
                        f"P2 grid {work_idx}/{total_work}: {signal_variant} P W={fill_window}s "
                        f"target={target} timeout={timeout}s",
                        flush=True,
                    )
                    executed_parts = []
                    for market, rows in passive_entries.groupby("market", sort=False):
                        executed = execute_non_overlap(
                            rows,
                            market_groups[market],
                            target,
                            timeout,
                            "P",
                        )
                        if not executed.empty:
                            executed_parts.append(executed)
                    executed_all = (
                        pd.concat(executed_parts, ignore_index=True)
                        if executed_parts
                        else passive_entries.iloc[0:0]
                    )
                    surface_parts.append(
                        summarize_combo(
                            signal_events,
                            passive_entries,
                            executed_all,
                            execution_mode="P",
                            fill_window_sec=fill_window,
                            target_type_name=target,
                            timeout_sec=timeout,
                            seed_base=RNG_SEED
                            + stable_seed_offset(signal_variant, "P", fill_window, target, timeout),
                        )
                    )

    surface = pd.concat(surface_parts, ignore_index=True) if surface_parts else pd.DataFrame()
    passive_frontier = surface[
        surface["execution_mode"].eq("P")
        & surface["segment_type"].isin(
            [
                "all",
                "family",
                "market",
                "spread_decile",
                "depth_decile",
                "family_spread_decile",
                "market_spread_decile",
                "market_depth_decile",
            ]
        )
    ].copy()
    passive_frontier = passive_frontier.sort_values(
        ["survives_ci_gt_0", "ci_lo", "mean_pnl_bps", "executed_fill_rate"],
        ascending=[False, False, False, False],
    )
    return surface, passive_frontier


def top_rows_table(df: pd.DataFrame, limit: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    sub = df.copy()
    sub = sub[sub["n_executed"].gt(0)]
    if sub.empty:
        return "_No executed rows._"
    sub = sub.sort_values(["ci_lo", "mean_pnl_bps", "executed_fill_rate"], ascending=False).head(limit)
    rows = []
    for row in sub.itertuples(index=False):
        rows.append(
            [
                row.signal_variant,
                row.segment_type,
                safe_text(row.market, 18),
                safe_text(row.slug, 34),
                row.regime_value,
                row.signal_tail,
                f"W={int(row.fill_window_sec)}",
                str(row.target_type).replace("_", " "),
                f"{int(row.timeout_sec)}s",
                f"{int(row.n_signal_events):,}",
                f"{int(row.n_executed):,}",
                pct(float(row.executed_fill_rate)),
                bps(float(row.mean_pnl_bps)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
            ]
        )
    return markdown_table(
        [
            "signal",
            "segment",
            "market",
            "slug",
            "regime",
            "tail",
            "fill",
            "target",
            "timeout",
            "signals",
            "exec",
            "fill rate",
            "mean",
            "CI",
        ],
        rows,
    )


def tight_spread_table(frontier: pd.DataFrame) -> str:
    tight = frontier[
        frontier["regime_type"].eq("spread_decile")
        & frontier["regime_value"].eq("spread_d01_tight")
        & frontier["signal_tail"].eq("both_tails")
        & frontier["n_executed"].gt(0)
    ].copy()
    return top_rows_table(tight, limit=10)


def write_note(surface: pd.DataFrame, frontier: pd.DataFrame, events: pd.DataFrame) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    passive_market_regime = frontier[
        frontier["segment_type"].isin(["market_spread_decile", "market_depth_decile"])
        & frontier["n_executed"].ge(MIN_CI_N)
    ].copy()
    passive_market_regime_both = passive_market_regime[
        passive_market_regime["signal_tail"].eq("both_tails")
    ].copy()
    passive_survivors = passive_market_regime[passive_market_regime["ci_lo"].gt(0)].copy()
    passive_survivors_both = passive_market_regime_both[passive_market_regime_both["ci_lo"].gt(0)].copy()
    passive_robust = passive_market_regime[
        passive_market_regime["ci_lo"].gt(0) & passive_market_regime["n_executed"].ge(ROBUST_N)
    ].copy()
    passive_robust_both = passive_market_regime_both[
        passive_market_regime_both["ci_lo"].gt(0) & passive_market_regime_both["n_executed"].ge(ROBUST_N)
    ].copy()
    all_passive_survivors = frontier[
        frontier["n_executed"].ge(MIN_CI_N)
        & frontier["ci_lo"].gt(0)
    ].copy()
    taker = surface[
        surface["execution_mode"].eq("T")
        & surface["n_executed"].ge(MIN_CI_N)
    ].copy()
    taker_survivors = taker[taker["ci_lo"].gt(0)].copy()
    best_passive = (
        frontier[frontier["n_executed"].gt(0)]
        .sort_values(["ci_lo", "mean_pnl_bps", "executed_fill_rate"], ascending=False)
        .head(1)
    )
    best_text = "No passive executions were generated."
    if not best_passive.empty:
        best = best_passive.iloc[0]
        best_text = (
            f"Best passive row by CI lower bound is `{safe_text(best['signal_variant'])}` / "
            f"`{safe_text(best['segment_type'])}` / `{safe_text(best['market'])}` / "
            f"`{safe_text(best['regime_value'])}`, W={int(best['fill_window_sec'])}s, "
            f"target `{best['target_type']}`, timeout={int(best['timeout_sec'])}s, "
            f"{int(best['n_executed']):,} executions, fill rate "
            f"{pct(float(best['executed_fill_rate']))}, mean {bps(float(best['mean_pnl_bps']))}, "
            f"CI [{bps(float(best['ci_lo']))}, {bps(float(best['ci_hi']))}]."
        )

    if passive_survivors.empty:
        headline = (
            "No: no passive market-regime-timeout cell clears CI lower > 0 under non-overlap. "
            "This closes the local reversion framing on the current A1 data."
        )
    else:
        headline = (
            f"Yes, but only narrowly: {len(passive_survivors):,} tail-aware passive "
            f"market-regime-timeout rows clear CI lower > 0 under non-overlap, with "
            f"{len(passive_robust):,} still clearing at n >= {ROBUST_N}. The combined `both_tails` "
            f"frontier has {len(passive_survivors_both):,} CI-positive rows, so this reopens the "
            f"local signal only as a fragile tail-specific anomaly rather than a clean reusable edge."
        )

    signal_counts = (
        events.groupby("signal_variant").size().reset_index(name="events").sort_values("signal_variant")
    )
    signal_rows = [[row.signal_variant, f"{int(row.events):,}"] for row in signal_counts.itertuples(index=False)]

    note = f"""---
tags: [dali, p2, micro-price, reversion, results]
---

# P2 Reversion-To-Microprice Findings

## Headline

{headline}

{best_text}

## Open Question Answer

The narrow test was: does **any passive fade-to-microprice cell** survive non-overlap, fill-rate drag, and net execution costs? Counting `market_spread_decile` and `market_depth_decile` rows with at least {MIN_CI_N} non-overlap executions, the answer is:

- Passive tail-aware survivor rows with CI lower > 0: {len(passive_survivors):,}
- Passive tail-aware robust survivor rows with CI lower > 0 and n >= {ROBUST_N}: {len(passive_robust):,}
- Passive combined-`both_tails` survivor rows with CI lower > 0: {len(passive_survivors_both):,}
- Passive combined-`both_tails` robust survivor rows with CI lower > 0 and n >= {ROBUST_N}: {len(passive_robust_both):,}
- Any-segment passive survivor rows, for context only: {len(all_passive_survivors):,}
- Taker fade survivor rows, for execution-control context: {len(taker_survivors):,}

## Passive Frontier

{top_rows_table(passive_market_regime, limit=14)}

## Tight-Spread Passive Rows

{tight_spread_table(frontier)}

## Taker Fade Control

{top_rows_table(taker, limit=10)}

## Signal Event Counts

{markdown_table(["signal", "top/bottom-decile events"], signal_rows)}

## Method

- Input: `{display_path(FEATURES)}`.
- Signals: `tob_imbalance_level = direction_factor * tob_imbalance` and a 5s OFI sidecar, `ofi_5s = direction_factor * rolling_sum(ofi_combined_event, 5s) / mean_depth`.
- Trigger: per-market signed top and bottom deciles. The CSV reports `top_decile`, `bottom_decile`, and `both_tails`.
- Trade direction: fade the signal. If continuation would buy a token, P2 sells it; if continuation would sell, P2 buys it.
- Target: current weighted mid/microprice, `weighted_mid = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)`, or half-way from entry price to that weighted mid.
- Mode T: immediate taker fade at the opposite executable touch, with taker fee on entry and exit.
- Mode P: post at the touch on the fade side. A long bid fills on a SELL print at or below the bid; a short ask fills on a BUY print at or above the ask, within W in {PASSIVE_FILL_WINDOWS}. Entry gets the maker rebate; exit pays taker fee at bid/ask.
- Exit: first of target reached, one-spread adverse stop, or timeout in {TIMEOUT_SECONDS}. Exits are marked to bid for long and ask for short.
- Non-overlap: one open position per market per grid cell. Taker signals block from entry to exit. Passive signals block only after an actual fill; candidates are considered in fill-time order and are skipped if either signal time or fill time falls inside an open interval.
- CI: normal interval over contiguous 300s block means of non-overlap executed PnL.
- Segments: all, family, market, spread decile, depth decile, family x spread decile, market x spread decile, and market x depth decile.

## Outputs

- `{display_path(SURFACE_OUT)}`
- `{display_path(FRONTIER_OUT)}`
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> int:
    print(f"loading {display_path(FEATURES)}", flush=True)
    df = load_features(FEATURES)
    print(f"loaded {len(df):,} rows across {df['market'].nunique():,} markets", flush=True)
    print("adding signal columns", flush=True)
    df = add_signal_columns(df)
    print("building top/bottom-decile fade events", flush=True)
    events = build_signal_events(df)
    print(f"built {len(events):,} signal events", flush=True)
    surface, frontier = run_grid(df, events)
    SURFACE_OUT.parent.mkdir(parents=True, exist_ok=True)
    surface.to_csv(SURFACE_OUT, index=False)
    frontier.to_csv(FRONTIER_OUT, index=False)
    write_note(surface, frontier, events)
    print(f"surface: {display_path(SURFACE_OUT)} ({len(surface):,} rows)")
    print(f"frontier: {display_path(FRONTIER_OUT)} ({len(frontier):,} rows)")
    print(f"note: {display_path(NOTE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
