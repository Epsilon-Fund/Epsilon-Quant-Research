"""OPT-D pooled in-sample directional ceiling search.

This is deliberately an optimizer/falsifier, not an OOS estimate. It pools
A0/A0b/A0c/A0c-roll and asks how high the strategy-family ceiling gets after
allowing signal, direction, gates, execution, and exits to move together.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
import warnings
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np
import pandas as pd
try:
    import optuna
except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment guard
    raise SystemExit("Optuna is required. Run with: uv run --with optuna python scripts/dali_block_optd_ceiling_search.py") from exc

from dali_block_a1_analyze import FEE_BY_CATEGORY, family_category
from dali_block_a1_replay_batch import display_path


ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
FEATURES = ANALYSIS / "block_a1_features.parquet"
OUT_CSV = ANALYSIS / "csv_outputs" / "options_delta" / "optd_ceiling_search.csv"
NOTE = NOTES / "block_optd_ceiling_findings.md"

RUN_IDS = ("a0", "a0b", "a0c", "a0c_roll")
BOOTSTRAP_CHUNK_SECONDS = 300
BOOTSTRAP_SAMPLES = 200
STALE_BOOK_MAX_SECONDS = 5.0
FOUR_HOURS_SECONDS = 4 * 60 * 60
RNG_SEED = 20260530
MIN_OBJECTIVE_N = 30
MAX_TRIAL_CANDIDATES = 250_000

BASE_SIGNALS = ("ofi_l1", "ofi_5s", "tob_imbalance_level", "weighted_mid_edge_bps")
SIGNALS = BASE_SIGNALS + tuple(f"rrank_{s}" for s in BASE_SIGNALS)
FIXED_HORIZONS = (5, 10, 30, 60, 300, 900)
PASSIVE_WINDOWS = (1, 2, 5, 10, 30)
FAMILY_GATES = ("all", "Crypto", "Geopolitics", "Sports", "Other")
SPREAD_GATES = ("all", "tight_q20", "tight_q50", "mid_q20_q80", "wide_q50")
DEPTH_GATES = ("all", "deep_q50", "deep_q80", "shallow_q50", "shallow_q20")


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def safe_text(value: object, max_len: int = 58) -> str:
    text = str(value if value is not None else "").replace("|", "/").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "."


def stable_seed(*parts: object) -> int:
    return RNG_SEED + int(zlib.crc32("|".join(map(str, parts)).encode("utf-8")) % 100_000)


def fee_amount(category: str, price: float) -> float:
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = float(np.clip(price, 0.0, 1.0))
    return float(params["fee_rate"] * p * (1.0 - p))


def maker_rebate_amount(category: str, price: float) -> float:
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    return fee_amount(category, price) * float(params.get("maker_rebate_pct", 0.0))


def crypto_window_end(slug: str) -> int | None:
    match = re.match(r"^(btc|eth|sol)-updown-4h-(\d+)$", str(slug))
    if not match:
        return None
    return int(match.group(2)) + FOUR_HOURS_SECONDS


def load_features() -> pd.DataFrame:
    cols = [
        "run_id",
        "received_at",
        "exchange_ts",
        "event_type",
        "asset_id",
        "market_id",
        "family",
        "slug",
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
        "trade_price",
        "trade_side",
        "last_trade_side",
        "trade_size",
    ]
    placeholders = ", ".join(["?"] * len(RUN_IDS))
    con = duckdb.connect()
    query = f"""
        SELECT {", ".join(cols)}
        FROM read_parquet(?)
        WHERE run_id IN ({placeholders})
          AND (coalesce(is_book_state_complete, false) OR event_type = 'last_trade_price')
    """
    df = con.execute(query, [str(FEATURES), *RUN_IDS]).df()
    con.close()
    if df.empty:
        raise SystemExit(f"no rows loaded from {display_path(FEATURES)}")
    print(f"loaded pooled feature rows: {len(df):,}", flush=True)
    return prepare_features(df)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["received_at"] = pd.to_datetime(out["received_at"], utc=True)
    out["exchange_ts"] = pd.to_datetime(out["exchange_ts"], utc=True, errors="coerce")
    out["event_ts"] = out["exchange_ts"].where(out["exchange_ts"].notna(), out["received_at"])
    for col in ("run_id", "event_type", "asset_id", "market_id", "family", "slug", "trade_side", "last_trade_side"):
        out[col] = out[col].fillna("").astype(str)
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
        "trade_price",
        "trade_size",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["is_book_state_complete"] = out["is_book_state_complete"].fillna(False).astype(bool)
    out["market"] = out["run_id"] + ":" + out["market_id"]
    out["category"] = out["family"].map(family_category)
    out["direction_factor"] = np.where(out["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    out["trade_side_norm"] = (
        out["trade_side"]
        .where(out["trade_side"].ne(""), out["last_trade_side"])
        .fillna("")
        .astype(str)
        .str.upper()
    )
    out["touch_depth"] = out[["best_bid_size", "best_ask_size"]].sum(axis=1, min_count=2)
    depth = (
        out.groupby(["run_id", "market_id"], as_index=False)["touch_depth"]
        .mean()
        .rename(columns={"touch_depth": "market_mean_depth"})
    )
    out = out.merge(depth, on=["run_id", "market_id"], how="left")
    out["relative_depth"] = out["touch_depth"] / out["market_mean_depth"].replace(0.0, np.nan)
    out["spread_bps"] = np.where(out["mid"].gt(0), out["spread"] / out["mid"] * 10_000.0, np.nan)
    size_sum = out["best_bid_size"] + out["best_ask_size"]
    out["weighted_mid"] = np.where(
        size_sum.gt(0),
        (out["best_ask"] * out["best_bid_size"] + out["best_bid"] * out["best_ask_size"]) / size_sum,
        np.nan,
    )
    out["weighted_mid_edge_bps"] = np.where(
        out["mid"].gt(0) & out["weighted_mid"].notna(),
        out["direction_factor"] * (out["weighted_mid"] - out["mid"]) / np.clip(out["mid"], 0.01, 0.99) * 10_000.0,
        np.nan,
    )
    return out.sort_values(["run_id", "market_id", "asset_id", "event_ts"]).reset_index(drop=True)


def valid_quote_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["is_book_state_complete"]
        & df["event_ts"].notna()
        & df["book_staleness_seconds"].le(STALE_BOOK_MAX_SECONDS)
        & df["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & df["mid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_bid"].ge(0.0)
        & df["best_ask"].le(1.0)
        & df["best_ask"].ge(df["best_bid"])
    )


def add_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    groups = list(df.groupby(["run_id", "market_id", "asset_id"], sort=False))
    for idx, (_, group) in enumerate(groups, start=1):
        if idx == 1 or idx % 50 == 0:
            print(f"signal/rolling-rank features {idx}/{len(groups)}", flush=True)
        g = group.sort_values("event_ts").copy()
        g["tob_imbalance"] = g["tob_imbalance"].ffill()
        g["tob_imbalance_level"] = g["direction_factor"] * g["tob_imbalance"]
        g["ofi_l1"] = g["direction_factor"] * g["ofi_combined_event"].fillna(0.0) / g[
            "market_mean_depth"
        ].replace(0.0, np.nan)
        g = g.set_index("event_ts", drop=False)
        ofi_5s = g["ofi_combined_event"].fillna(0.0).rolling("5s").sum()
        g["ofi_5s"] = g["direction_factor"] * ofi_5s / g["market_mean_depth"].replace(0.0, np.nan)
        for signal in BASE_SIGNALS:
            ranked = g[signal].rolling("300s", min_periods=10).rank(pct=True)
            g[f"rrank_{signal}"] = 2.0 * (ranked - 0.5)
        pieces.append(g.reset_index(drop=True))
    out = pd.concat(pieces, ignore_index=True)
    for signal in SIGNALS:
        out[signal] = pd.to_numeric(out[signal], errors="coerce").astype("float32")
    return out.sort_values(["run_id", "market_id", "asset_id", "event_ts"]).reset_index(drop=True)


@dataclass
class AssetState:
    times: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    mid: np.ndarray
    weighted_mid: np.ndarray
    signals: dict[str, np.ndarray]


@dataclass
class TradeState:
    times: np.ndarray
    price: np.ndarray
    side: np.ndarray


@dataclass
class SearchData:
    quotes: pd.DataFrame
    quote_arrays: dict[str, np.ndarray]
    asset_states: list[AssetState | None]
    trade_states: list[TradeState | None]
    category_by_market: np.ndarray
    category_name_by_code: dict[int, str]
    market_name_by_code: dict[int, str]
    slug_by_market_code: dict[int, str]
    spread_thresholds: dict[str, float]
    depth_thresholds: dict[str, float]
    signal_abs_thresholds: dict[str, dict[float, float]]
    run_counts: pd.DataFrame


def build_search_data(df: pd.DataFrame) -> SearchData:
    quote_df = df[valid_quote_mask(df)].copy()
    if quote_df.empty:
        raise SystemExit("no valid quote rows")
    quote_df["event_ns"] = quote_df["event_ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
    quote_df["window_end_ns"] = quote_df["slug"].map(crypto_window_end).fillna(0).astype("int64") * 1_000_000_000
    quote_df["is_crypto_4h"] = quote_df["window_end_ns"].gt(0)

    quote_df["asset_code"], asset_values = pd.factorize(quote_df["asset_id"].astype(str), sort=True)
    quote_df["market_code"], market_values = pd.factorize(quote_df["market"].astype(str), sort=True)
    quote_df["category_code"], category_values = pd.factorize(quote_df["category"].astype(str), sort=True)
    quote_df["family_code"], _ = pd.factorize(quote_df["family"].astype(str), sort=True)

    spread_clean = quote_df["spread_bps"].replace([np.inf, -np.inf], np.nan).dropna()
    depth_clean = quote_df["relative_depth"].replace([np.inf, -np.inf], np.nan).dropna()
    spread_thresholds = {
        "q20": float(spread_clean.quantile(0.20)),
        "q50": float(spread_clean.quantile(0.50)),
        "q80": float(spread_clean.quantile(0.80)),
    }
    depth_thresholds = {
        "q20": float(depth_clean.quantile(0.20)),
        "q50": float(depth_clean.quantile(0.50)),
        "q80": float(depth_clean.quantile(0.80)),
    }
    signal_abs_thresholds: dict[str, dict[float, float]] = {}
    for signal in SIGNALS:
        values = quote_df[signal].replace([np.inf, -np.inf], np.nan).dropna().abs()
        signal_abs_thresholds[signal] = {q: float(values.quantile(q)) for q in (0.80, 0.85, 0.90, 0.95, 0.98)}

    quote_arrays = {
        "event_ns": quote_df["event_ns"].to_numpy(dtype=np.int64),
        "asset_code": quote_df["asset_code"].to_numpy(dtype=np.int32),
        "market_code": quote_df["market_code"].to_numpy(dtype=np.int32),
        "category_code": quote_df["category_code"].to_numpy(dtype=np.int16),
        "best_bid": quote_df["best_bid"].to_numpy(dtype=np.float64),
        "best_ask": quote_df["best_ask"].to_numpy(dtype=np.float64),
        "mid": quote_df["mid"].to_numpy(dtype=np.float64),
        "spread": quote_df["spread"].fillna(0.0).to_numpy(dtype=np.float64),
        "weighted_mid": quote_df["weighted_mid"].to_numpy(dtype=np.float64),
        "spread_bps": quote_df["spread_bps"].to_numpy(dtype=np.float64),
        "relative_depth": quote_df["relative_depth"].to_numpy(dtype=np.float64),
        "direction_factor": quote_df["direction_factor"].to_numpy(dtype=np.float64),
        "window_end_ns": quote_df["window_end_ns"].to_numpy(dtype=np.int64),
        "is_crypto_4h": quote_df["is_crypto_4h"].to_numpy(dtype=bool),
    }
    for signal in SIGNALS:
        quote_arrays[signal] = quote_df[signal].to_numpy(dtype=np.float64)

    asset_count = int(quote_df["asset_code"].max()) + 1
    asset_states: list[AssetState | None] = [None] * asset_count
    for asset_code, group in quote_df.groupby("asset_code", sort=False):
        g = group.sort_values("event_ns")
        asset_states[int(asset_code)] = AssetState(
            times=g["event_ns"].to_numpy(dtype=np.int64),
            bid=g["best_bid"].to_numpy(dtype=np.float64),
            ask=g["best_ask"].to_numpy(dtype=np.float64),
            mid=g["mid"].to_numpy(dtype=np.float64),
            weighted_mid=g["weighted_mid"].to_numpy(dtype=np.float64),
            signals={signal: g[signal].to_numpy(dtype=np.float64) for signal in SIGNALS},
        )

    asset_code_by_id = dict(zip(asset_values.astype(str), range(len(asset_values)), strict=False))
    trade_states: list[TradeState | None] = [None] * asset_count
    trades = df[
        df["event_type"].eq("last_trade_price")
        & df["event_ts"].notna()
        & df["trade_price"].replace([np.inf, -np.inf], np.nan).notna()
        & df["trade_side_norm"].isin(["BUY", "SELL"])
    ].copy()
    if not trades.empty:
        trades["asset_code"] = trades["asset_id"].astype(str).map(asset_code_by_id)
        trades = trades[trades["asset_code"].notna()].copy()
        trades["event_ns"] = trades["event_ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
        trades["side_code"] = trades["trade_side_norm"].map({"BUY": 1, "SELL": -1}).astype("int8")
        for asset_code, group in trades.groupby("asset_code", sort=False):
            g = group.sort_values("event_ns")
            trade_states[int(asset_code)] = TradeState(
                times=g["event_ns"].to_numpy(dtype=np.int64),
                price=g["trade_price"].to_numpy(dtype=np.float64),
                side=g["side_code"].to_numpy(dtype=np.int8),
            )

    category_by_market = (
        quote_df.groupby("market_code")["category_code"].first().sort_index().to_numpy(dtype=np.int16)
    )
    market_name_by_code = dict(enumerate(market_values.astype(str)))
    category_name_by_code = dict(enumerate(category_values.astype(str)))
    slug_by_market_code = (
        quote_df.groupby("market_code")["slug"].first().astype(str).to_dict()
    )
    run_counts = (
        df.groupby("run_id", as_index=False)
        .agg(n_rows=("event_type", "size"), n_markets=("market_id", "nunique"))
        .sort_values("run_id")
    )
    print(
        f"valid quote rows: {len(quote_df):,}; assets={asset_count:,}; markets={len(market_values):,}; trades={len(trades):,}",
        flush=True,
    )
    return SearchData(
        quotes=quote_df,
        quote_arrays=quote_arrays,
        asset_states=asset_states,
        trade_states=trade_states,
        category_by_market=category_by_market,
        category_name_by_code=category_name_by_code,
        market_name_by_code=market_name_by_code,
        slug_by_market_code=slug_by_market_code,
        spread_thresholds=spread_thresholds,
        depth_thresholds=depth_thresholds,
        signal_abs_thresholds=signal_abs_thresholds,
        run_counts=run_counts,
    )


def signal_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.isfinite(values) & (np.abs(values) >= threshold) & (np.sign(values) != 0)


def apply_family_mask(data: SearchData, family_gate: str) -> np.ndarray:
    if family_gate == "all":
        return np.ones(len(data.quotes), dtype=bool)
    category_codes = data.quote_arrays["category_code"]
    allowed = [code for code, name in data.category_name_by_code.items() if name == family_gate]
    if not allowed:
        return np.zeros(len(data.quotes), dtype=bool)
    return category_codes == int(allowed[0])


def apply_spread_mask(data: SearchData, spread_gate: str) -> np.ndarray:
    spread = data.quote_arrays["spread_bps"]
    if spread_gate == "all":
        return np.ones(len(spread), dtype=bool)
    q20, q50, q80 = data.spread_thresholds["q20"], data.spread_thresholds["q50"], data.spread_thresholds["q80"]
    if spread_gate == "tight_q20":
        return spread <= q20
    if spread_gate == "tight_q50":
        return spread <= q50
    if spread_gate == "mid_q20_q80":
        return (spread >= q20) & (spread <= q80)
    if spread_gate == "wide_q50":
        return spread >= q50
    raise ValueError(spread_gate)


def apply_depth_mask(data: SearchData, depth_gate: str) -> np.ndarray:
    depth = data.quote_arrays["relative_depth"]
    if depth_gate == "all":
        return np.ones(len(depth), dtype=bool)
    q20, q50, q80 = data.depth_thresholds["q20"], data.depth_thresholds["q50"], data.depth_thresholds["q80"]
    if depth_gate == "deep_q50":
        return depth >= q50
    if depth_gate == "deep_q80":
        return depth >= q80
    if depth_gate == "shallow_q50":
        return depth <= q50
    if depth_gate == "shallow_q20":
        return depth <= q20
    raise ValueError(depth_gate)


def first_passive_fill(
    trades: TradeState | None,
    *,
    start_ns: int,
    end_ns: int,
    side: float,
    quote_price: float,
) -> int | None:
    if trades is None or len(trades.times) == 0:
        return None
    left = int(np.searchsorted(trades.times, start_ns, side="left"))
    right = int(np.searchsorted(trades.times, end_ns, side="right"))
    if left >= right:
        return None
    if side > 0:
        desired_side = -1
        for pos in range(left, right):
            if trades.side[pos] == desired_side and trades.price[pos] <= quote_price + 1e-12:
                return int(trades.times[pos])
    else:
        desired_side = 1
        for pos in range(left, right):
            if trades.side[pos] == desired_side and trades.price[pos] >= quote_price - 1e-12:
                return int(trades.times[pos])
    return None


def fixed_exit_idx(state: AssetState, target_ns: int) -> int | None:
    idx = int(np.searchsorted(state.times, target_ns, side="right") - 1)
    if idx < 0:
        return None
    return idx


def touch_exit_price(state: AssetState, side: float, idx: int) -> float:
    return float(state.bid[idx] if side > 0 else state.ask[idx])


def exit_for_trade(
    state: AssetState,
    *,
    signal_name: str,
    entry_ns: int,
    entry_signal_sign: float,
    side: float,
    entry_price: float,
    entry_spread: float,
    entry_micro: float,
    target_ns: int,
    exit_style: str,
    tp_mult: float,
    sl_mult: float,
) -> tuple[int, float, str] | None:
    left = int(np.searchsorted(state.times, entry_ns, side="right"))
    right = int(np.searchsorted(state.times, target_ns, side="right") - 1)
    if left >= len(state.times) or right < left:
        return None
    right = min(right, len(state.times) - 1)
    if exit_style == "fixed":
        idx = fixed_exit_idx(state, target_ns)
        if idx is None or idx <= left - 1:
            return None
        return int(state.times[idx]), touch_exit_price(state, side, idx), "fixed"

    if exit_style == "signal_reversal":
        sig = state.signals[signal_name][left : right + 1]
        rev = np.flatnonzero(np.isfinite(sig) & (np.sign(sig) != 0) & (np.sign(sig) != entry_signal_sign))
        idx = left + int(rev[0]) if len(rev) else right
        return int(state.times[idx]), touch_exit_price(state, side, idx), "signal_reversal" if len(rev) else "fixed_timeout"

    if exit_style == "target_micro_price":
        target = float(entry_micro)
        if not np.isfinite(target):
            idx = right
            return int(state.times[idx]), touch_exit_price(state, side, idx), "fixed_timeout"
        touch = state.bid[left : right + 1] if side > 0 else state.ask[left : right + 1]
        hit = np.flatnonzero(touch >= target - 1e-12) if side > 0 else np.flatnonzero(touch <= target + 1e-12)
        idx = left + int(hit[0]) if len(hit) else right
        return int(state.times[idx]), touch_exit_price(state, side, idx), "target_micro" if len(hit) else "fixed_timeout"

    if exit_style == "tp_sl":
        spread = max(float(entry_spread), 1e-4)
        target = float(np.clip(entry_price + side * tp_mult * spread, 0.0, 1.0))
        stop = float(np.clip(entry_price - side * sl_mult * spread, 0.0, 1.0))
        touch = state.bid[left : right + 1] if side > 0 else state.ask[left : right + 1]
        tp = np.flatnonzero(touch >= target - 1e-12) if side > 0 else np.flatnonzero(touch <= target + 1e-12)
        sl = np.flatnonzero(touch <= stop + 1e-12) if side > 0 else np.flatnonzero(touch >= stop - 1e-12)
        tp_idx = left + int(tp[0]) if len(tp) else None
        sl_idx = left + int(sl[0]) if len(sl) else None
        if tp_idx is not None and (sl_idx is None or tp_idx <= sl_idx):
            idx = tp_idx
            reason = "take_profit"
        elif sl_idx is not None:
            idx = sl_idx
            reason = "stop_loss"
        else:
            idx = right
            reason = "fixed_timeout"
        return int(state.times[idx]), touch_exit_price(state, side, idx), reason

    raise ValueError(exit_style)


def block_bootstrap_ci(values: np.ndarray, times_ns: np.ndarray, *, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    times_ns = np.asarray(times_ns, dtype=np.int64)
    mask = np.isfinite(values)
    values = values[mask]
    times_ns = times_ns[mask]
    if len(values) < 2:
        return math.nan, math.nan
    block_id = ((times_ns - int(times_ns.min())) // int(BOOTSTRAP_CHUNK_SECONDS * 1_000_000_000)).astype(int)
    blocks = [np.flatnonzero(block_id == bid) for bid in np.unique(block_id)]
    if len(blocks) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))])
        samples.append(float(np.mean(values[idx])))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate_config(data: SearchData, params: dict[str, Any], *, trial_number: int) -> dict[str, Any]:
    signal_name = str(params["signal"])
    signal_values = data.quote_arrays[signal_name]
    abs_threshold = data.signal_abs_thresholds[signal_name][float(params["entry_abs_quantile"])]
    mask = (
        signal_mask(signal_values, abs_threshold)
        & apply_family_mask(data, str(params["family_gate"]))
        & apply_spread_mask(data, str(params["spread_gate"]))
        & apply_depth_mask(data, str(params["depth_gate"]))
    )
    if params["horizon_type"] == "boundary_4h":
        mask &= data.quote_arrays["is_crypto_4h"]
    cand_idx = np.flatnonzero(mask)
    n_signal_events = int(len(cand_idx))
    if n_signal_events == 0:
        return empty_result(params, n_signal_events)

    if n_signal_events > MAX_TRIAL_CANDIDATES:
        strengths = np.abs(signal_values[cand_idx])
        keep = np.argpartition(strengths, -MAX_TRIAL_CANDIDATES)[-MAX_TRIAL_CANDIDATES:]
        cand_idx = cand_idx[keep]
    market_codes = data.quote_arrays["market_code"][cand_idx]
    times = data.quote_arrays["event_ns"][cand_idx]
    order = np.lexsort((times, market_codes))
    cand_idx = cand_idx[order]
    market_codes = market_codes[order]

    pnl_values: list[float] = []
    pnl_weighted: list[float] = []
    entry_times: list[int] = []
    hold_seconds: list[float] = []
    filled_raw = 0
    exit_reasons: Counter[str] = Counter()
    market_exec: Counter[int] = Counter()
    direction_mult = 1.0 if params["direction"] == "momentum" else -1.0
    execution = str(params["execution"])
    pass_w_ns = int(params.get("passive_window_sec", 0)) * 1_000_000_000

    unique_markets, starts = np.unique(market_codes, return_index=True)
    starts = list(starts) + [len(cand_idx)]
    for mpos, market_code in enumerate(unique_markets):
        group = cand_idx[starts[mpos] : starts[mpos + 1]]
        open_until = -1
        group_times = data.quote_arrays["event_ns"][group]
        local_idx = 0
        while local_idx < len(group):
            qidx = int(group[local_idx])
            event_ns = int(data.quote_arrays["event_ns"][qidx])
            if event_ns <= open_until:
                local_idx = int(np.searchsorted(group_times, open_until + 1, side="left"))
                continue
            sig_value = float(signal_values[qidx])
            sig_sign = float(np.sign(sig_value))
            if sig_sign == 0 or not np.isfinite(sig_sign):
                local_idx += 1
                continue
            direction_factor = float(data.quote_arrays["direction_factor"][qidx])
            side = direction_mult * sig_sign * direction_factor
            if side not in (-1.0, 1.0):
                local_idx += 1
                continue
            asset_code = int(data.quote_arrays["asset_code"][qidx])
            state = data.asset_states[asset_code]
            if state is None:
                local_idx += 1
                continue
            if execution == "taker":
                entry_ns = event_ns
                entry_px = float(data.quote_arrays["best_ask"][qidx] if side > 0 else data.quote_arrays["best_bid"][qidx])
                entry_fee = fee_amount(
                    data.category_name_by_code[int(data.quote_arrays["category_code"][qidx])],
                    entry_px,
                )
                entry_rebate = 0.0
            else:
                entry_quote = float(data.quote_arrays["best_bid"][qidx] if side > 0 else data.quote_arrays["best_ask"][qidx])
                fill_ns = first_passive_fill(
                    data.trade_states[asset_code],
                    start_ns=event_ns,
                    end_ns=event_ns + pass_w_ns,
                    side=side,
                    quote_price=entry_quote,
                )
                if fill_ns is None:
                    local_idx += 1
                    continue
                filled_raw += 1
                entry_ns = int(fill_ns)
                entry_px = entry_quote
                category = data.category_name_by_code[int(data.quote_arrays["category_code"][qidx])]
                entry_fee = 0.0
                entry_rebate = maker_rebate_amount(category, entry_px)
            if entry_ns <= open_until or not np.isfinite(entry_px) or entry_px <= 0:
                local_idx += 1
                continue
            if params["horizon_type"] == "boundary_4h":
                target_ns = int(data.quote_arrays["window_end_ns"][qidx])
                if target_ns <= entry_ns:
                    local_idx += 1
                    continue
            else:
                target_ns = entry_ns + int(params["horizon_sec"]) * 1_000_000_000
            exit_info = exit_for_trade(
                state,
                signal_name=signal_name,
                entry_ns=entry_ns,
                entry_signal_sign=sig_sign,
                side=side,
                entry_price=entry_px,
                entry_spread=float(data.quote_arrays["spread"][qidx]),
                entry_micro=float(data.quote_arrays["weighted_mid"][qidx]),
                target_ns=target_ns,
                exit_style=str(params["exit_style"]),
                tp_mult=float(params.get("tp_mult", 1.0)),
                sl_mult=float(params.get("sl_mult", 1.0)),
            )
            if exit_info is None:
                local_idx += 1
                continue
            exit_ns, exit_px, exit_reason = exit_info
            if exit_ns <= entry_ns or not np.isfinite(exit_px):
                local_idx += 1
                continue
            category = data.category_name_by_code[int(data.quote_arrays["category_code"][qidx])]
            gross = side * (exit_px - entry_px)
            exit_fee = fee_amount(category, exit_px)
            pnl = (gross - entry_fee - exit_fee + entry_rebate) / float(np.clip(entry_px, 0.01, 0.99)) * 10_000.0
            if not np.isfinite(pnl):
                local_idx += 1
                continue
            if params["mapping"] == "continuous_rank_sizing":
                strength = min(1.0, abs(sig_value) / max(abs_threshold, 1e-12))
            else:
                strength = 1.0
            pnl_values.append(float(pnl))
            pnl_weighted.append(float(pnl) * float(strength))
            entry_times.append(entry_ns)
            hold_seconds.append((exit_ns - entry_ns) / 1_000_000_000.0)
            exit_reasons[exit_reason] += 1
            market_exec[int(market_code)] += 1
            open_until = exit_ns
            local_idx = int(np.searchsorted(group_times, open_until + 1, side="left"))

    n_executed = len(pnl_values)
    if execution == "taker":
        filled_raw = n_signal_events
    if n_executed == 0:
        return empty_result(params, n_signal_events, n_entry_filled_raw=filled_raw)
    pnl_arr = np.asarray(pnl_values, dtype=float)
    objective_arr = np.asarray(pnl_weighted if params["mapping"] == "continuous_rank_sizing" else pnl_values, dtype=float)
    time_arr = np.asarray(entry_times, dtype=np.int64)
    ci_lo, ci_hi = block_bootstrap_ci(objective_arr, time_arr, seed=stable_seed("optd", trial_number))
    mean = float(np.mean(objective_arr))
    std = float(np.std(objective_arr, ddof=1)) if len(objective_arr) > 1 else math.nan
    sharpe = float(mean / std * math.sqrt(len(objective_arr))) if std and np.isfinite(std) and std > 0 else math.nan
    return {
        **params,
        "n_signal_events": n_signal_events,
        "n_entry_filled_raw": int(filled_raw),
        "n_executed": int(n_executed),
        "raw_entry_fill_rate": filled_raw / n_signal_events if n_signal_events else math.nan,
        "executed_fill_rate": n_executed / n_signal_events if n_signal_events else math.nan,
        "mean_pnl_bps": mean,
        "median_pnl_bps": float(np.median(objective_arr)),
        "unweighted_mean_pnl_bps": float(np.mean(pnl_arr)),
        "win_rate": float(np.mean(pnl_arr > 0)),
        "sharpe_like": sharpe,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "mean_hold_seconds": float(np.mean(hold_seconds)),
        "n_markets_executed": int(len(market_exec)),
        "candidate_cap_applied": bool(n_signal_events > MAX_TRIAL_CANDIDATES),
        "top_exit_reason": exit_reasons.most_common(1)[0][0] if exit_reasons else "",
        "take_profit_rate": exit_reasons["take_profit"] / n_executed,
        "stop_loss_rate": exit_reasons["stop_loss"] / n_executed,
        "target_micro_rate": exit_reasons["target_micro"] / n_executed,
        "signal_reversal_rate": exit_reasons["signal_reversal"] / n_executed,
    }


def empty_result(params: dict[str, Any], n_signal_events: int, n_entry_filled_raw: int = 0) -> dict[str, Any]:
    return {
        **params,
        "n_signal_events": int(n_signal_events),
        "n_entry_filled_raw": int(n_entry_filled_raw),
        "n_executed": 0,
        "raw_entry_fill_rate": math.nan if n_signal_events == 0 else n_entry_filled_raw / n_signal_events,
        "executed_fill_rate": 0.0 if n_signal_events else math.nan,
        "mean_pnl_bps": math.nan,
        "median_pnl_bps": math.nan,
        "unweighted_mean_pnl_bps": math.nan,
        "win_rate": math.nan,
        "sharpe_like": math.nan,
        "ci_lo": math.nan,
        "ci_hi": math.nan,
        "mean_hold_seconds": math.nan,
        "n_markets_executed": 0,
        "candidate_cap_applied": bool(n_signal_events > MAX_TRIAL_CANDIDATES),
        "top_exit_reason": "",
        "take_profit_rate": math.nan,
        "stop_loss_rate": math.nan,
        "target_micro_rate": math.nan,
        "signal_reversal_rate": math.nan,
    }


def sample_params(trial: optuna.Trial) -> dict[str, Any]:
    signal = trial.suggest_categorical("signal", SIGNALS)
    mapping = trial.suggest_categorical("mapping", ["decile_gate", "continuous_rank_sizing"])
    direction = trial.suggest_categorical("direction", ["momentum", "fade"])
    execution = trial.suggest_categorical("execution", ["taker", "passive_maker"])
    horizon_type = trial.suggest_categorical("horizon_type", ["fixed", "boundary_4h"])
    exit_style = trial.suggest_categorical("exit_style", ["fixed", "tp_sl", "signal_reversal", "target_micro_price"])
    params: dict[str, Any] = {
        "signal": signal,
        "mapping": mapping,
        "direction": direction,
        "execution": execution,
        "horizon_type": horizon_type,
        "exit_style": exit_style,
        "entry_abs_quantile": trial.suggest_categorical("entry_abs_quantile", [0.80, 0.85, 0.90, 0.95, 0.98]),
        "family_gate": trial.suggest_categorical("family_gate", FAMILY_GATES),
        "spread_gate": trial.suggest_categorical("spread_gate", SPREAD_GATES),
        "depth_gate": trial.suggest_categorical("depth_gate", DEPTH_GATES),
        "horizon_sec": trial.suggest_categorical("horizon_sec", FIXED_HORIZONS),
        "passive_window_sec": trial.suggest_categorical("passive_window_sec", PASSIVE_WINDOWS),
        "tp_mult": trial.suggest_categorical("tp_mult", [0.5, 1.0, 2.0, 4.0]),
        "sl_mult": trial.suggest_categorical("sl_mult", [0.5, 1.0, 2.0, 4.0]),
    }
    if execution == "taker":
        params["passive_window_sec"] = 0
    if exit_style != "tp_sl":
        params["tp_mult"] = 1.0
        params["sl_mult"] = 1.0
    return params


def run_optuna(data: SearchData, n_trials: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    sampler = optuna.samplers.TPESampler(seed=RNG_SEED, multivariate=True, group=True, n_startup_trials=40)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        started = time.time()
        result = evaluate_config(data, params, trial_number=trial.number)
        result["trial"] = trial.number
        result["elapsed_sec"] = time.time() - started
        if result["n_executed"] < MIN_OBJECTIVE_N or not np.isfinite(result["mean_pnl_bps"]):
            objective_value = -100_000.0 + float(result["n_executed"])
        else:
            objective_value = float(result["mean_pnl_bps"])
            if np.isfinite(result.get("sharpe_like", math.nan)):
                objective_value += 0.01 * float(result["sharpe_like"])
        result["objective_value"] = objective_value
        rows.append(result)
        if (trial.number + 1) % 25 == 0:
            best = max(rows, key=lambda r: r.get("objective_value", -1e18))
            print(
                f"trial {trial.number + 1}/{n_trials}: best objective={best['objective_value']:.2f}, "
                f"mean={best.get('mean_pnl_bps', math.nan):.2f}, n={best.get('n_executed', 0)}",
                flush=True,
            )
        return objective_value

    study.optimize(objective, n_trials=n_trials, gc_after_trial=False, show_progress_bar=False)
    out = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    return out


def best_rows(results: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    eligible = results[results["n_executed"].ge(MIN_OBJECTIVE_N) & results["mean_pnl_bps"].replace([np.inf, -np.inf], np.nan).notna()]
    if eligible.empty:
        eligible = results[results["mean_pnl_bps"].replace([np.inf, -np.inf], np.nan).notna()]
    if eligible.empty:
        raise SystemExit("no executable optimizer trials produced finite PnL")
    best_mean = eligible.sort_values(["mean_pnl_bps", "sharpe_like"], ascending=False).iloc[0]
    best_lcb = eligible.sort_values(["ci_lo", "mean_pnl_bps"], ascending=False).iloc[0]
    return best_mean, best_lcb


def verdict_for_best(best: pd.Series) -> str:
    ci_hi = float(best.get("ci_hi", math.nan))
    ci_lo = float(best.get("ci_lo", math.nan))
    if np.isfinite(ci_hi) and ci_hi < 0:
        return "directional signal CLOSED: optimized ceiling CI upper < 0"
    if np.isfinite(ci_lo) and ci_lo > 0:
        return "POTENTIAL: optimized in-sample config has CI lower > 0; validate when more data lands"
    return "INCONCLUSIVE/POTENTIAL lead only: optimized IS ceiling does not close below zero"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def params_summary(row: pd.Series) -> str:
    return (
        f"{row.signal}, {row.mapping}, {row.direction}, {row.execution}, "
        f"{row.horizon_type}/{int(row.horizon_sec)}s, {row.exit_style}, "
        f"family={row.family_gate}, spread={row.spread_gate}, depth={row.depth_gate}, q={row.entry_abs_quantile}"
    )


def write_note(results: pd.DataFrame, data: SearchData, n_trials: int) -> None:
    best_mean, best_lcb = best_rows(results)
    verdict = verdict_for_best(best_mean)
    eligible = results[results["n_executed"].ge(MIN_OBJECTIVE_N)]
    positive_mean_count = int((eligible["mean_pnl_bps"] > 0).sum())
    ci_positive_count = int((eligible["ci_lo"] > 0).sum())
    ci_negative_count = int((eligible["ci_hi"] < 0).sum())
    top = (
        eligible
        .sort_values(["mean_pnl_bps", "ci_lo"], ascending=False)
        .head(12)
    )
    top_rows = [
        [
            int(row.trial),
            bps(float(row.mean_pnl_bps)),
            f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
            int(row.n_executed),
            pct(float(row.executed_fill_rate)),
            f"{float(row.sharpe_like):.2f}" if np.isfinite(float(row.sharpe_like)) else "n/a",
            safe_text(params_summary(row), 120),
        ]
        for row in top.itertuples(index=False)
    ]
    run_rows = [
        [row.run_id, f"{int(row.n_rows):,}", int(row.n_markets)]
        for row in data.run_counts.itertuples(index=False)
    ]
    cap_rate = float(results["candidate_cap_applied"].fillna(False).mean()) if len(results) else 0.0
    lines = [
        "---",
        "tags: [dali, optd, optuna, ceiling, in-sample, results]",
        "---",
        "",
        "# OPT-D Directional Ceiling Search Findings",
        "",
        "## Headline",
        "",
        f"Best in-sample net-of-cost config: trial `{int(best_mean.trial)}` with mean `{bps(float(best_mean.mean_pnl_bps))}`, CI `[{bps(float(best_mean.ci_lo))}, {bps(float(best_mean.ci_hi))}]`, n `{int(best_mean.n_executed)}`, fill `{pct(float(best_mean.executed_fill_rate))}`.",
        "",
        f"Verdict: **{verdict}**.",
        "",
        f"Among trials with n >= {MIN_OBJECTIVE_N}: `{positive_mean_count}` had positive mean, `{ci_positive_count}` had CI lower > 0, and `{ci_negative_count}` had CI upper < 0.",
        "",
        "An in-sample positive is a lead, not a result. Only an optimized ceiling with CI upper < 0 would close the directional family rigorously.",
        "",
        "## Best Config",
        "",
        f"- Params: `{params_summary(best_mean)}`",
        f"- Raw entry fill rate: `{pct(float(best_mean.raw_entry_fill_rate))}`",
        f"- Unweighted mean PnL: `{bps(float(best_mean.unweighted_mean_pnl_bps))}`",
        f"- Win rate: `{pct(float(best_mean.win_rate))}`",
        f"- Mean hold: `{float(best_mean.mean_hold_seconds):.1f}s`",
        f"- Markets executed: `{int(best_mean.n_markets_executed)}`",
        f"- Top exit reason: `{best_mean.top_exit_reason}`",
        "",
        "## Best By CI Lower",
        "",
        f"Trial `{int(best_lcb.trial)}`: `{bps(float(best_lcb.mean_pnl_bps))}` CI `[{bps(float(best_lcb.ci_lo))}, {bps(float(best_lcb.ci_hi))}]`, n `{int(best_lcb.n_executed)}`, params `{params_summary(best_lcb)}`.",
        "",
        "## Top Trials",
        "",
        markdown_table(["trial", "mean", "CI", "n", "fill", "sharpe", "params"], top_rows),
        "",
        "## Data",
        "",
        markdown_table(["run", "rows loaded", "markets"], run_rows),
        "",
        f"- Valid quote rows searched: `{len(data.quotes):,}`.",
        f"- Optuna trials: `{n_trials}` with TPE sampler.",
        f"- Candidate cap applied in `{pct(cap_rate)}` of trials; capped trials keep the strongest `{MAX_TRIAL_CANDIDATES:,}` signal rows before non-overlap.",
        "",
        "## Method",
        "",
        "- Pooled `a0`, `a0b`, `a0c`, and `a0c_roll` as one in-sample set; no holdout and no train/test split.",
        "- Signal set: OFI L1, OFI 5s, TOB imbalance level, weighted-mid edge, and 300s rolling-rank variants of each.",
        "- Search dimensions include mapping, direction, fixed/boundary horizons, exits, spread/depth/family gates, taker versus passive maker execution, and passive fill window.",
        "- Event ordering uses exchange timestamps when present, falling back to receive timestamps.",
        "- Quote states require complete books and `book_staleness_seconds <= 5`.",
        "- PnL is net of `FEE_BY_CATEGORY`; passive maker entries use the A1.4h/P2 trade-through fill proxy and maker rebate.",
        "- Non-overlap is enforced per market. CI bars use 300s clock-block bootstrap over executed trade PnL.",
        "",
        "## Outputs",
        "",
        f"- `{display_path(OUT_CSV)}`",
        f"- `{display_path(NOTE)}`",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=350)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    df = add_signal_features(load_features())
    data = build_search_data(df)
    results = run_optuna(data, n_trials=args.trials)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results, data, args.trials)
    best_mean, _ = best_rows(results)
    print(
        f"wrote {display_path(OUT_CSV)} rows={len(results):,}; best mean={best_mean.mean_pnl_bps:.2f} bps "
        f"CI=[{best_mean.ci_lo:.2f}, {best_mean.ci_hi:.2f}] n={int(best_mean.n_executed)}",
        flush=True,
    )
    print(f"wrote {display_path(NOTE)}; elapsed={time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
