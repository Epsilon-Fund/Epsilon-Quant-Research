"""Block A18 passive reversion-to-microprice gate.

This is the narrow "never-run framing" left open by A1.x: use the real
top-of-book imbalance signal as a fade-to-microprice strategy, not as
directional continuation.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np
import pandas as pd

from dali_block_a1_analyze import FEE_BY_CATEGORY, family_category


ANALYSIS = ROOT / "data" / "analysis"
FEATURES = ANALYSIS / "block_a1_features.parquet"
OUT_DIR = ANALYSIS / "csv_outputs" / "dali"
SURFACE_OUT = OUT_DIR / "block_a18_passive_reversion_surface.csv"
MARKET_OUT = OUT_DIR / "block_a18_passive_reversion_market_clusters.csv"
EXEC_OUT = OUT_DIR / "block_a18_passive_reversion_executed.csv"
NOTE = ROOT / "notes" / "dali" / "block_a18_passive_reversion_findings.md"
TODO = REPO / "brain" / "TODO.md"

RUN_IDS = ("a0", "a0b", "a0c", "a0c_roll")
HORIZONS = (5, 10, 30, 60)
PASSIVE_WINDOWS = (1, 5, 10)
BOOTSTRAP_SAMPLES = 500
RNG_SEED = 20260602
MIN_CI_MARKETS = 3
STALE_BOOK_MAX_SECONDS = 5.0


@dataclass
class AssetState:
    times: np.ndarray
    bid: np.ndarray
    ask: np.ndarray


@dataclass
class TradeState:
    times: np.ndarray
    price: np.ndarray
    side: np.ndarray


@dataclass
class ReplayData:
    quotes: pd.DataFrame
    q: dict[str, np.ndarray]
    asset_states: list[AssetState | None]
    trade_states: list[TradeState | None]
    category_by_code: dict[int, str]
    market_by_code: dict[int, str]
    family_by_market: dict[int, str]
    slug_by_market: dict[int, str]
    run_counts: pd.DataFrame


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_text(value: object, max_len: int = 70) -> str:
    text = str(value if value is not None else "").replace("|", "/")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "."


def bps(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.1f} bps"


def cents(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.3f}c"


def pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.2f}%"


def stable_seed(*parts: object) -> int:
    return RNG_SEED + int(zlib.crc32("|".join(map(str, parts)).encode("utf-8")) % 100_000)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def fee_amount(category: str, price: float) -> float:
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = float(np.clip(price, 0.0, 1.0))
    return float(params["fee_rate"] * p * (1.0 - p))


def maker_rebate_amount(category: str, price: float) -> float:
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    return fee_amount(category, price) * float(params.get("maker_rebate_pct", 0.0))


def parse_ints(raw: str | None, default: Iterable[int]) -> tuple[int, ...]:
    if not raw:
        return tuple(default)
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def load_features(path: Path = FEATURES) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing features parquet: {display_path(path)}")
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
        "trade_price",
        "trade_side",
        "last_trade_side",
    ]
    placeholders = ",".join(["?"] * len(RUN_IDS))
    query = f"""
        SELECT {", ".join(cols)}
        FROM read_parquet(?)
        WHERE run_id IN ({placeholders})
          AND (coalesce(is_book_state_complete, false) OR event_type = 'last_trade_price')
    """
    con = duckdb.connect()
    df = con.execute(query, [str(path), *RUN_IDS]).df()
    con.close()
    if df.empty:
        raise SystemExit(f"no feature rows loaded from {display_path(path)}")
    print(f"loaded rows: {len(df):,}", flush=True)
    return prepare_features(df)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["received_at"] = pd.to_datetime(out["received_at"], utc=True)
    out["exchange_ts"] = pd.to_datetime(out["exchange_ts"], utc=True, errors="coerce")
    out["event_ts"] = out["exchange_ts"].where(out["exchange_ts"].notna(), out["received_at"])
    for col in ("run_id", "event_type", "asset_id", "market_id", "family", "slug", "trade_side", "last_trade_side"):
        out[col] = out[col].fillna("").astype(str)
    for col in (
        "outcome_index",
        "book_staleness_seconds",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "tob_imbalance",
        "trade_price",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["is_book_state_complete"] = out["is_book_state_complete"].fillna(False).astype(bool)
    out["market"] = out["run_id"] + ":" + out["market_id"]
    out["asset_key"] = out["run_id"] + ":" + out["market_id"] + ":" + out["asset_id"]
    out["category"] = out["family"].map(family_category)
    out["direction_factor"] = np.where(out["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    out["trade_side_norm"] = (
        out["trade_side"]
        .where(out["trade_side"].ne(""), out["last_trade_side"])
        .fillna("")
        .astype(str)
        .str.upper()
    )
    size_sum = out["best_bid_size"] + out["best_ask_size"]
    out["weighted_mid"] = np.where(
        size_sum.gt(0)
        & out["best_bid"].notna()
        & out["best_ask"].notna()
        & out["best_bid_size"].notna()
        & out["best_ask_size"].notna(),
        (out["best_ask"] * out["best_bid_size"] + out["best_bid"] * out["best_ask_size"]) / size_sum,
        out["mid"] + 0.5 * out["spread"] * out["tob_imbalance"],
    )
    out["weighted_mid"] = out["weighted_mid"].clip(lower=0.0, upper=1.0)
    out["spread_bps"] = np.where(out["mid"].gt(0), out["spread"] / out["mid"] * 10_000.0, np.nan)
    return out.sort_values(["run_id", "market_id", "asset_id", "event_ts"]).reset_index(drop=True)


def valid_quote_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["is_book_state_complete"]
        & df["event_ts"].notna()
        & df["book_staleness_seconds"].le(STALE_BOOK_MAX_SECONDS)
        & df["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & df["mid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["weighted_mid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_bid"].ge(0.0)
        & df["best_ask"].le(1.0)
        & df["best_ask"].ge(df["best_bid"])
        & df["spread"].gt(0.0)
    )


def add_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    groups = list(df.groupby(["run_id", "market_id", "asset_id"], sort=False))
    for idx, (_, group) in enumerate(groups, start=1):
        if idx == 1 or idx % 50 == 0:
            print(f"signal features {idx}/{len(groups)}", flush=True)
        g = group.sort_values("event_ts").copy()
        g["tob_imbalance"] = g["tob_imbalance"].ffill()
        g["tob_imbalance_level"] = g["direction_factor"] * g["tob_imbalance"]
        g = g.set_index("event_ts", drop=False)
        ranked = g["tob_imbalance_level"].rolling("300s", min_periods=10).rank(pct=True)
        g["rrank_tob_imbalance_level"] = 2.0 * (ranked - 0.5)
        pieces.append(g.reset_index(drop=True))
    out = pd.concat(pieces, ignore_index=True)
    for col in ("tob_imbalance_level", "rrank_tob_imbalance_level"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out.sort_values(["run_id", "market_id", "asset_id", "event_ts"]).reset_index(drop=True)


def build_replay_data(df: pd.DataFrame) -> ReplayData:
    quote_df = df[valid_quote_mask(df)].copy()
    if quote_df.empty:
        raise SystemExit("no valid quote rows")
    quote_df["event_ns"] = quote_df["event_ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
    quote_df["asset_code"], asset_values = pd.factorize(quote_df["asset_key"].astype(str), sort=True)
    quote_df["market_code"], market_values = pd.factorize(quote_df["market"].astype(str), sort=True)
    quote_df["category_code"], category_values = pd.factorize(quote_df["category"].astype(str), sort=True)

    q = {
        "event_ns": quote_df["event_ns"].to_numpy(dtype=np.int64),
        "asset_code": quote_df["asset_code"].to_numpy(dtype=np.int32),
        "market_code": quote_df["market_code"].to_numpy(dtype=np.int32),
        "category_code": quote_df["category_code"].to_numpy(dtype=np.int16),
        "direction_factor": quote_df["direction_factor"].to_numpy(dtype=np.float64),
        "best_bid": quote_df["best_bid"].to_numpy(dtype=np.float64),
        "best_ask": quote_df["best_ask"].to_numpy(dtype=np.float64),
        "spread": quote_df["spread"].to_numpy(dtype=np.float64),
        "weighted_mid": quote_df["weighted_mid"].to_numpy(dtype=np.float64),
        "tob": quote_df["tob_imbalance_level"].to_numpy(dtype=np.float64),
        "rrank_tob": quote_df["rrank_tob_imbalance_level"].to_numpy(dtype=np.float64),
    }

    asset_count = int(quote_df["asset_code"].max()) + 1
    asset_states: list[AssetState | None] = [None] * asset_count
    for asset_code, group in quote_df.groupby("asset_code", sort=False):
        g = group.sort_values("event_ns")
        asset_states[int(asset_code)] = AssetState(
            times=g["event_ns"].to_numpy(dtype=np.int64),
            bid=g["best_bid"].to_numpy(dtype=np.float64),
            ask=g["best_ask"].to_numpy(dtype=np.float64),
        )

    asset_code_by_key = dict(zip(asset_values.astype(str), range(len(asset_values)), strict=False))
    trades = df[
        df["event_type"].eq("last_trade_price")
        & df["event_ts"].notna()
        & df["trade_price"].replace([np.inf, -np.inf], np.nan).notna()
        & df["trade_side_norm"].isin(["BUY", "SELL"])
    ].copy()
    trade_states: list[TradeState | None] = [None] * asset_count
    if not trades.empty:
        trades["asset_code"] = trades["asset_key"].astype(str).map(asset_code_by_key)
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

    market_by_code = dict(enumerate(market_values.astype(str)))
    category_by_code = dict(enumerate(category_values.astype(str)))
    family_by_market = quote_df.groupby("market_code")["family"].first().astype(str).to_dict()
    slug_by_market = quote_df.groupby("market_code")["slug"].first().astype(str).to_dict()
    run_counts = (
        df.groupby("run_id", as_index=False)
        .agg(n_rows=("event_type", "size"), n_markets=("market_id", "nunique"))
        .sort_values("run_id")
    )
    print(
        f"valid quotes: {len(quote_df):,}; assets={asset_count:,}; "
        f"markets={len(market_values):,}; trades={len(trades):,}",
        flush=True,
    )
    return ReplayData(
        quotes=quote_df,
        q=q,
        asset_states=asset_states,
        trade_states=trade_states,
        category_by_code=category_by_code,
        market_by_code=market_by_code,
        family_by_market=family_by_market,
        slug_by_market=slug_by_market,
        run_counts=run_counts,
    )


def decile_candidate_mask(data: ReplayData) -> tuple[np.ndarray, np.ndarray]:
    values = data.q["tob"]
    markets = data.q["market_code"]
    valid = np.isfinite(values) & (values != 0.0)
    q10_by_market: dict[int, float] = {}
    q90_by_market: dict[int, float] = {}
    for market_code in np.unique(markets[valid]):
        vals = values[valid & (markets == market_code)]
        if len(vals) < 20:
            continue
        q10_by_market[int(market_code)] = float(np.quantile(vals, 0.10))
        q90_by_market[int(market_code)] = float(np.quantile(vals, 0.90))
    mask = np.zeros(len(values), dtype=bool)
    tails = np.zeros(len(values), dtype=np.int8)
    for market_code, q10 in q10_by_market.items():
        q90 = q90_by_market[market_code]
        m = valid & (markets == market_code)
        bottom = m & (values <= q10)
        top = m & (values >= q90)
        mask |= bottom | top
        tails[bottom] = -1
        tails[top] = 1
    return mask, tails


def rolling_rank_candidate_mask(data: ReplayData) -> tuple[np.ndarray, np.ndarray]:
    values = data.q["rrank_tob"]
    mask = np.isfinite(values) & (values != 0.0)
    tails = np.zeros(len(values), dtype=np.int8)
    tails[mask & (values > 0.0)] = 1
    tails[mask & (values < 0.0)] = -1
    return mask, tails


def build_candidates(data: ReplayData, mapping: str) -> pd.DataFrame:
    if mapping == "binary_decile":
        mask, tails = decile_candidate_mask(data)
        signal = data.q["tob"]
        strength = np.ones(mask.sum(), dtype=np.float64)
    elif mapping == "rolling_rank_sizing":
        mask, tails = rolling_rank_candidate_mask(data)
        signal = data.q["rrank_tob"]
        strength = np.abs(signal[mask]).astype(np.float64)
    else:
        raise ValueError(mapping)

    idx = np.flatnonzero(mask)
    sig = signal[idx].astype(np.float64)
    direction = data.q["direction_factor"][idx]
    side = -np.sign(sig) * direction
    keep = np.isfinite(sig) & np.isfinite(side) & np.isin(side, [-1.0, 1.0])
    idx = idx[keep]
    sig = sig[keep]
    side = side[keep]
    strength = strength[keep]
    tail_vals = tails[idx]
    market = data.q["market_code"][idx]
    event_ns = data.q["event_ns"][idx]
    order = np.lexsort((-np.abs(sig), event_ns, market))
    idx = idx[order]
    sig = sig[order]
    side = side[order]
    strength = strength[order]
    tail_vals = tail_vals[order]

    out = pd.DataFrame(
        {
            "qidx": idx.astype(np.int64),
            "mapping": mapping,
            "market_code": data.q["market_code"][idx].astype(np.int32),
            "asset_code": data.q["asset_code"][idx].astype(np.int32),
            "category_code": data.q["category_code"][idx].astype(np.int16),
            "event_ns": data.q["event_ns"][idx].astype(np.int64),
            "signal_value": sig.astype(np.float64),
            "abs_signal": np.abs(sig).astype(np.float64),
            "tail_code": tail_vals.astype(np.int8),
            "side": side.astype(np.float64),
            "strength": strength.astype(np.float64),
            "best_bid": data.q["best_bid"][idx].astype(np.float64),
            "best_ask": data.q["best_ask"][idx].astype(np.float64),
            "spread": data.q["spread"][idx].astype(np.float64),
            "weighted_mid": data.q["weighted_mid"][idx].astype(np.float64),
        }
    )
    print(f"{mapping}: {len(out):,} candidate signal rows", flush=True)
    return out


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
        for pos in range(left, right):
            if trades.side[pos] == -1 and trades.price[pos] <= quote_price + 1e-12:
                return int(trades.times[pos])
    else:
        for pos in range(left, right):
            if trades.side[pos] == 1 and trades.price[pos] >= quote_price - 1e-12:
                return int(trades.times[pos])
    return None


def add_taker_entries(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    side = out["side"].to_numpy(dtype=float)
    out["entry_time_ns"] = out["event_ns"].to_numpy(dtype=np.int64)
    out["entry_price"] = np.where(side > 0, out["best_ask"], out["best_bid"])
    out["entry_filled"] = True
    return out


def add_passive_entries(data: ReplayData, candidates: pd.DataFrame, fill_window_sec: int) -> pd.DataFrame:
    out = candidates.copy()
    side = out["side"].to_numpy(dtype=float)
    out["entry_price"] = np.where(side > 0, out["best_bid"], out["best_ask"])
    fill_time = np.full(len(out), -1, dtype=np.int64)
    total_assets = out["asset_code"].nunique()
    for apos, (asset_code, idx) in enumerate(out.groupby("asset_code", sort=False).groups.items(), start=1):
        if apos == 1 or apos % 50 == 0:
            print(f"  passive fills W={fill_window_sec}s asset {apos}/{total_assets}", flush=True)
        trades = data.trade_states[int(asset_code)]
        if trades is None:
            continue
        locs = np.fromiter(idx, dtype=np.int64)
        starts = out.loc[locs, "event_ns"].to_numpy(dtype=np.int64)
        sides = out.loc[locs, "side"].to_numpy(dtype=float)
        prices = out.loc[locs, "entry_price"].to_numpy(dtype=float)
        for j, loc in enumerate(locs):
            found = first_passive_fill(
                trades,
                start_ns=int(starts[j]),
                end_ns=int(starts[j] + fill_window_sec * 1_000_000_000),
                side=float(sides[j]),
                quote_price=float(prices[j]),
            )
            if found is not None:
                fill_time[int(loc)] = found
    out["entry_time_ns"] = fill_time
    out["entry_filled"] = out["entry_time_ns"].ge(0)
    return out


def simulate_exit(
    state: AssetState,
    *,
    entry_ns: int,
    side: float,
    entry_price: float,
    target_price: float,
    timeout_sec: int,
) -> tuple[int, float, str] | None:
    timeout_ns = entry_ns + timeout_sec * 1_000_000_000
    left = int(np.searchsorted(state.times, entry_ns, side="right"))
    right = int(np.searchsorted(state.times, timeout_ns, side="right") - 1)
    if left >= len(state.times) or right < left:
        return None
    right = min(right, len(state.times) - 1)
    touch = state.bid[left : right + 1] if side > 0 else state.ask[left : right + 1]
    hits = np.flatnonzero(touch >= target_price - 1e-12) if side > 0 else np.flatnonzero(touch <= target_price + 1e-12)
    if len(hits):
        idx = left + int(hits[0])
        reason = "target_micro"
    else:
        idx = right
        reason = "timeout"
    exit_ns = int(state.times[idx]) if reason == "target_micro" else timeout_ns
    exit_price = float(state.bid[idx] if side > 0 else state.ask[idx])
    if exit_ns <= entry_ns or not np.isfinite(exit_price) or not np.isfinite(entry_price) or entry_price <= 0:
        return None
    return exit_ns, exit_price, reason


def execute_non_overlap(
    data: ReplayData,
    entries: pd.DataFrame,
    *,
    execution: str,
    timeout_sec: int,
) -> pd.DataFrame:
    filled = entries[
        entries["entry_filled"].fillna(False)
        & entries["entry_time_ns"].replace([np.inf, -np.inf], np.nan).notna()
        & entries["entry_price"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    if filled.empty:
        return filled.iloc[0:0].copy()
    if execution == "taker":
        filled = filled.sort_values(["market_code", "event_ns", "abs_signal"], ascending=[True, True, False])
    else:
        filled = filled.sort_values(["market_code", "entry_time_ns", "event_ns", "abs_signal"], ascending=[True, True, True, False])

    rows: list[dict[str, object]] = []
    for market_code, group in filled.groupby("market_code", sort=False):
        open_until = -1
        order_times = group["event_ns"].to_numpy(dtype=np.int64) if execution == "taker" else group["entry_time_ns"].to_numpy(dtype=np.int64)
        local = group.reset_index(drop=True)
        idx = 0
        while idx < len(local):
            row = local.iloc[idx]
            signal_ns = int(row["event_ns"])
            entry_ns = int(row["entry_time_ns"])
            if signal_ns <= open_until or entry_ns <= open_until:
                idx += 1
                continue
            state = data.asset_states[int(row["asset_code"])]
            if state is None:
                idx += 1
                continue
            exit_info = simulate_exit(
                state,
                entry_ns=entry_ns,
                side=float(row["side"]),
                entry_price=float(row["entry_price"]),
                target_price=float(row["weighted_mid"]),
                timeout_sec=timeout_sec,
            )
            if exit_info is None:
                idx += 1
                continue
            exit_ns, exit_price, exit_reason = exit_info
            category = data.category_by_code[int(row["category_code"])]
            side = float(row["side"])
            entry_price = float(row["entry_price"])
            gross = side * (exit_price - entry_price)
            entry_fee = fee_amount(category, entry_price) if execution == "taker" else 0.0
            entry_rebate = maker_rebate_amount(category, entry_price) if execution == "passive_maker" else 0.0
            exit_fee = fee_amount(category, exit_price)
            pnl_amount = gross - entry_fee - exit_fee + entry_rebate
            denom = float(np.clip(entry_price, 0.01, 0.99))
            out = row.to_dict()
            out.update(
                {
                    "market": data.market_by_code[int(market_code)],
                    "family": data.family_by_market[int(market_code)],
                    "slug": data.slug_by_market[int(market_code)],
                    "execution": execution,
                    "timeout_sec": int(timeout_sec),
                    "exit_time_ns": int(exit_ns),
                    "exit_price": float(exit_price),
                    "exit_reason": exit_reason,
                    "hold_seconds": (exit_ns - entry_ns) / 1_000_000_000.0,
                    "gross_cents": gross * 100.0,
                    "entry_fee_cents": entry_fee * 100.0,
                    "entry_rebate_cents": entry_rebate * 100.0,
                    "exit_fee_cents": exit_fee * 100.0,
                    "pnl_cents": pnl_amount * 100.0,
                    "pnl_bps": pnl_amount / denom * 10_000.0,
                    "sized_pnl_cents": pnl_amount * 100.0 * float(row["strength"]),
                    "sized_notional_contracts": float(row["strength"]),
                    "target_edge_cents": side * (float(row["weighted_mid"]) - entry_price) * 100.0,
                }
            )
            rows.append(out)
            open_until = int(exit_ns)
            idx = int(np.searchsorted(order_times, open_until + 1, side="left"))
    return pd.DataFrame(rows)


def cluster_bootstrap_ratio(
    numer_by_market: pd.Series,
    denom_by_market: pd.Series,
    *,
    seed: int,
) -> tuple[float, float, float]:
    joined = pd.concat([numer_by_market.rename("numer"), denom_by_market.rename("denom")], axis=1).fillna(0.0)
    joined = joined[joined["denom"].gt(0)]
    if joined.empty:
        return math.nan, math.nan, math.nan
    point = float(joined["numer"].sum() / joined["denom"].sum())
    if len(joined) < MIN_CI_MARKETS:
        return point, math.nan, math.nan
    arr = joined[["numer", "denom"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        take = rng.integers(0, len(arr), size=len(arr))
        denom = float(arr[take, 1].sum())
        if denom > 0:
            samples.append(float(arr[take, 0].sum() / denom))
    if not samples:
        return point, math.nan, math.nan
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return point, float(lo), float(hi)


def summarize_config(
    *,
    data: ReplayData,
    candidates: pd.DataFrame,
    entries: pd.DataFrame,
    executed: pd.DataFrame,
    mapping: str,
    execution: str,
    fill_window_sec: int,
    timeout_sec: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_counts = (
        candidates.groupby("market_code", as_index=False)
        .agg(n_signal_events=("qidx", "size"), family=("market_code", lambda s: data.family_by_market[int(s.iloc[0])]))
    )
    filled_counts = (
        entries[entries["entry_filled"]]
        .groupby("market_code", as_index=False)
        .agg(n_entry_filled_raw=("qidx", "size"))
    )
    market = signal_counts.merge(filled_counts, on="market_code", how="left")
    market["n_entry_filled_raw"] = market["n_entry_filled_raw"].fillna(0).astype(int)
    if executed.empty:
        exec_stats = pd.DataFrame(columns=["market_code"])
    else:
        exec_stats = (
            executed.groupby("market_code", as_index=False)
            .agg(
                n_executed=("qidx", "size"),
                mean_pnl_cents=("pnl_cents", "mean"),
                mean_sized_pnl_cents=("sized_pnl_cents", "mean"),
                sum_sized_pnl_cents=("sized_pnl_cents", "sum"),
                sum_size_contracts=("sized_notional_contracts", "sum"),
                mean_pnl_bps=("pnl_bps", "mean"),
                median_pnl_cents=("pnl_cents", "median"),
                win_rate=("pnl_cents", lambda s: float((s > 0).mean())),
                mean_hold_seconds=("hold_seconds", "mean"),
                target_micro_rate=("exit_reason", lambda s: float((s == "target_micro").mean())),
                timeout_rate=("exit_reason", lambda s: float((s == "timeout").mean())),
                mean_entry_price=("entry_price", "mean"),
                worst_pnl_cents=("pnl_cents", "min"),
            )
        )
    market = market.merge(exec_stats, on="market_code", how="left")
    market["n_executed"] = market["n_executed"].fillna(0).astype(int)
    market["raw_entry_fill_rate"] = market["n_entry_filled_raw"] / market["n_signal_events"].replace(0, np.nan)
    market["executed_fill_rate"] = market["n_executed"] / market["n_signal_events"].replace(0, np.nan)
    market["nonoverlap_keep_rate_after_fill"] = market["n_executed"] / market["n_entry_filled_raw"].replace(0, np.nan)
    market["uncond_sized_cents_per_signal"] = market["sum_sized_pnl_cents"].fillna(0.0) / market["n_signal_events"].replace(0, np.nan)
    market["market"] = market["market_code"].map(data.market_by_code)
    market["slug"] = market["market_code"].map(data.slug_by_market)
    market["mapping"] = mapping
    market["execution"] = execution
    market["fill_window_sec"] = fill_window_sec
    market["timeout_sec"] = timeout_sec

    rows: list[dict[str, object]] = []
    segments = [("all", "ALL", market)]
    for family, sub in market.groupby("family", sort=True):
        segments.append(("family", str(family), sub))

    exec_by_market = executed.groupby("market_code") if not executed.empty else None
    for segment_type, segment_value, sub_market in segments:
        market_codes = set(sub_market["market_code"].astype(int).tolist())
        sub_exec = (
            executed[executed["market_code"].astype(int).isin(market_codes)].copy()
            if not executed.empty
            else executed
        )
        denom_events = sub_market.set_index("market_code")["n_signal_events"].astype(float)
        filled_raw = int(sub_market["n_entry_filled_raw"].sum())
        n_signal = int(sub_market["n_signal_events"].sum())
        n_exec = int(sub_market["n_executed"].sum())

        if sub_exec.empty:
            cond_mean = cond_lo = cond_hi = math.nan
            cond_bps = cond_bps_lo = cond_bps_hi = math.nan
            actual_contract_mean = actual_contract_lo = actual_contract_hi = math.nan
            uncond_mean = 0.0
            uncond_lo = math.nan
            uncond_hi = math.nan
            win_rate = mean_hold = target_rate = timeout_rate = median = worst = math.nan
        else:
            numer = sub_exec.groupby("market_code")["pnl_cents"].sum()
            denom = sub_exec.groupby("market_code").size().astype(float)
            cond_mean, cond_lo, cond_hi = cluster_bootstrap_ratio(
                numer, denom, seed=stable_seed(mapping, execution, fill_window_sec, timeout_sec, segment_type, segment_value, "cond_cents")
            )
            numer_bps = sub_exec.groupby("market_code")["pnl_bps"].sum()
            cond_bps, cond_bps_lo, cond_bps_hi = cluster_bootstrap_ratio(
                numer_bps, denom, seed=stable_seed(mapping, execution, fill_window_sec, timeout_sec, segment_type, segment_value, "cond_bps")
            )
            size_denom = sub_exec.groupby("market_code")["sized_notional_contracts"].sum()
            size_numer = sub_exec.groupby("market_code")["sized_pnl_cents"].sum()
            actual_contract_mean, actual_contract_lo, actual_contract_hi = cluster_bootstrap_ratio(
                size_numer, size_denom, seed=stable_seed(mapping, execution, fill_window_sec, timeout_sec, segment_type, segment_value, "actual_contract")
            )
            uncond_mean, uncond_lo, uncond_hi = cluster_bootstrap_ratio(
                size_numer,
                denom_events,
                seed=stable_seed(mapping, execution, fill_window_sec, timeout_sec, segment_type, segment_value, "uncond"),
            )
            win_rate = float(sub_exec["pnl_cents"].gt(0).mean())
            mean_hold = float(sub_exec["hold_seconds"].mean())
            target_rate = float(sub_exec["exit_reason"].eq("target_micro").mean())
            timeout_rate = float(sub_exec["exit_reason"].eq("timeout").mean())
            median = float(sub_exec["pnl_cents"].median())
            worst = float(sub_exec["pnl_cents"].min())

        rows.append(
            {
                "mapping": mapping,
                "execution": execution,
                "fill_window_sec": fill_window_sec,
                "timeout_sec": timeout_sec,
                "segment_type": segment_type,
                "segment_value": segment_value,
                "n_markets_signal": int(sub_market["market_code"].nunique()),
                "n_markets_executed": int(sub_market.loc[sub_market["n_executed"].gt(0), "market_code"].nunique()),
                "n_signal_events": n_signal,
                "n_entry_filled_raw": filled_raw,
                "n_executed": n_exec,
                "raw_entry_fill_rate": filled_raw / n_signal if n_signal else math.nan,
                "executed_fill_rate": n_exec / n_signal if n_signal else math.nan,
                "nonoverlap_keep_rate_after_fill": n_exec / filled_raw if filled_raw else math.nan,
                "mean_pnl_cents": cond_mean,
                "ci_lo_cents": cond_lo,
                "ci_hi_cents": cond_hi,
                "mean_pnl_bps": cond_bps,
                "ci_lo_bps": cond_bps_lo,
                "ci_hi_bps": cond_bps_hi,
                "mean_actual_contract_cents": actual_contract_mean,
                "actual_contract_ci_lo_cents": actual_contract_lo,
                "actual_contract_ci_hi_cents": actual_contract_hi,
                "uncond_sized_cents_per_signal": uncond_mean,
                "uncond_ci_lo_cents": uncond_lo,
                "uncond_ci_hi_cents": uncond_hi,
                "median_pnl_cents": median,
                "worst_pnl_cents": worst,
                "win_rate": win_rate,
                "mean_hold_seconds": mean_hold,
                "target_micro_rate": target_rate,
                "timeout_rate": timeout_rate,
                "survives_cluster_ci_gt_0": bool(np.isfinite(cond_lo) and cond_lo > 0 and n_exec > 0),
                "survives_uncond_cluster_ci_gt_0": bool(np.isfinite(uncond_lo) and uncond_lo > 0 and n_exec > 0),
            }
        )
    return pd.DataFrame(rows), market


def run_replay(data: ReplayData, mappings: tuple[str, ...], horizons: tuple[int, ...], windows: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    surface_parts: list[pd.DataFrame] = []
    market_parts: list[pd.DataFrame] = []
    exec_parts: list[pd.DataFrame] = []

    for mapping in mappings:
        candidates = build_candidates(data, mapping)
        if candidates.empty:
            continue
        taker_entries = add_taker_entries(candidates)
        for horizon in horizons:
            print(f"{mapping} taker H={horizon}s", flush=True)
            executed = execute_non_overlap(data, taker_entries, execution="taker", timeout_sec=horizon)
            surface, market = summarize_config(
                data=data,
                candidates=candidates,
                entries=taker_entries,
                executed=executed,
                mapping=mapping,
                execution="taker",
                fill_window_sec=0,
                timeout_sec=horizon,
            )
            surface_parts.append(surface)
            market_parts.append(market)
            if not executed.empty:
                exec_parts.append(executed)

        for window in windows:
            print(f"{mapping} passive fill scan W={window}s", flush=True)
            passive_entries = add_passive_entries(data, candidates, window)
            for horizon in horizons:
                print(f"{mapping} passive W={window}s H={horizon}s", flush=True)
                executed = execute_non_overlap(data, passive_entries, execution="passive_maker", timeout_sec=horizon)
                surface, market = summarize_config(
                    data=data,
                    candidates=candidates,
                    entries=passive_entries,
                    executed=executed,
                    mapping=mapping,
                    execution="passive_maker",
                    fill_window_sec=window,
                    timeout_sec=horizon,
                )
                surface_parts.append(surface)
                market_parts.append(market)
                if not executed.empty:
                    exec_parts.append(executed)

    surface_all = pd.concat(surface_parts, ignore_index=True) if surface_parts else pd.DataFrame()
    market_all = pd.concat(market_parts, ignore_index=True) if market_parts else pd.DataFrame()
    exec_all = pd.concat(exec_parts, ignore_index=True) if exec_parts else pd.DataFrame()
    return surface_all, market_all, exec_all


def best_rows(surface: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = surface[surface["segment_type"].eq("all")].copy()
    primary = pooled[pooled["timeout_sec"].eq(5)].copy()
    best = pooled.sort_values(["ci_lo_cents", "mean_pnl_cents", "executed_fill_rate"], ascending=False)
    return primary, best


def verdict(surface: pd.DataFrame) -> str:
    pooled = surface[surface["segment_type"].eq("all")].copy()
    if pooled.empty:
        return "CONFIRM-CLOSE: no executable reversion rows."
    passive = pooled[pooled["execution"].eq("passive_maker")]
    taker = pooled[pooled["execution"].eq("taker")]
    passive_clear = bool(passive["survives_cluster_ci_gt_0"].any())
    taker_clear = bool(taker["survives_cluster_ci_gt_0"].any())
    if passive_clear:
        return "REOPEN-PASSIVE-REVERSION: pooled passive row has market-cluster lower-CI > 0."
    if taker_clear:
        return "TACTICAL-TAKER-ONLY-ANOMALY: taker cleared but passive did not; this conflicts with the spread-headwind thesis."
    return "CONFIRM-CLOSE: continuation is dead and explicit reversion also fails the market-cluster CI gate."


def table_from_rows(rows_df: pd.DataFrame, limit: int = 12) -> str:
    if rows_df.empty:
        return "_No rows._"
    sub = rows_df.head(limit)
    rows = []
    for row in sub.itertuples(index=False):
        rows.append(
            [
                row.mapping,
                row.execution.replace("_", " "),
                f"W={int(row.fill_window_sec)}",
                f"H={int(row.timeout_sec)}",
                f"{int(row.n_markets_executed)}/{int(row.n_markets_signal)}",
                f"{int(row.n_executed):,}",
                pct(float(row.executed_fill_rate)),
                cents(float(row.mean_pnl_cents)),
                f"[{cents(float(row.ci_lo_cents))}, {cents(float(row.ci_hi_cents))}]",
                cents(float(row.uncond_sized_cents_per_signal)),
                f"[{cents(float(row.uncond_ci_lo_cents))}, {cents(float(row.uncond_ci_hi_cents))}]",
            ]
        )
    return markdown_table(
        ["mapping", "route", "fill", "exit", "markets", "exec", "exec fill", "cond EV", "cond CI", "uncond EV", "uncond CI"],
        rows,
    )


def family_table(surface: pd.DataFrame, limit: int = 12) -> str:
    fam = surface[surface["segment_type"].eq("family")].copy()
    if fam.empty:
        return "_No rows._"
    fam = fam.sort_values(["ci_lo_cents", "mean_pnl_cents"], ascending=False).head(limit)
    rows = []
    for row in fam.itertuples(index=False):
        rows.append(
            [
                safe_text(row.segment_value, 24),
                row.mapping,
                row.execution.replace("_", " "),
                f"W={int(row.fill_window_sec)} H={int(row.timeout_sec)}",
                f"{int(row.n_markets_executed)}/{int(row.n_markets_signal)}",
                f"{int(row.n_executed):,}",
                pct(float(row.executed_fill_rate)),
                cents(float(row.mean_pnl_cents)),
                f"[{cents(float(row.ci_lo_cents))}, {cents(float(row.ci_hi_cents))}]",
                cents(float(row.uncond_sized_cents_per_signal)),
            ]
        )
    return markdown_table(["family", "mapping", "route", "cell", "markets", "exec", "exec fill", "cond EV", "CI", "uncond"], rows)


def market_diag_table(market: pd.DataFrame, limit: int = 12) -> str:
    if market.empty:
        return "_No rows._"
    sub = market[market["n_executed"].gt(0)].copy()
    if sub.empty:
        return "_No executed market rows._"
    sub = sub.sort_values(["mean_pnl_cents", "n_executed"], ascending=False).head(limit)
    rows = []
    for row in sub.itertuples(index=False):
        rows.append(
            [
                safe_text(row.market, 14),
                safe_text(row.slug, 38),
                row.mapping,
                row.execution.replace("_", " "),
                f"W={int(row.fill_window_sec)} H={int(row.timeout_sec)}",
                f"{int(row.n_executed):,}",
                pct(float(row.executed_fill_rate)),
                cents(float(row.mean_pnl_cents)),
                cents(float(row.uncond_sized_cents_per_signal)),
            ]
        )
    return markdown_table(["market", "slug", "mapping", "route", "cell", "exec", "exec fill", "cond EV", "uncond"], rows)


def write_note(surface: pd.DataFrame, market: pd.DataFrame, exec_rows: pd.DataFrame, elapsed: float) -> None:
    primary, ranked = best_rows(surface)
    v = verdict(surface)
    pooled = surface[surface["segment_type"].eq("all")].copy()
    best = ranked.iloc[0] if not ranked.empty else None
    decile_primary = primary[primary["mapping"].eq("binary_decile")].sort_values(["execution", "fill_window_sec"])
    rrank_primary = primary[primary["mapping"].eq("rolling_rank_sizing")].sort_values(["execution", "fill_window_sec"])
    positive_rows = int(pooled["survives_cluster_ci_gt_0"].sum()) if not pooled.empty else 0
    uncond_positive_rows = int(pooled["survives_uncond_cluster_ci_gt_0"].sum()) if not pooled.empty else 0
    worst = float(exec_rows["pnl_cents"].min()) if not exec_rows.empty else math.nan
    data_rows = [
        [row.run_id, f"{int(row.n_rows):,}", int(row.n_markets)]
        for row in surface.attrs.get("run_counts", pd.DataFrame()).itertuples(index=False)
    ]

    headline = (
        "No pooled route clears the pre-registered market-cluster lower-CI > 0 gate."
        if positive_rows == 0
        else f"{positive_rows} pooled row(s) clear the market-cluster lower-CI > 0 gate."
    )
    if best is not None:
        deciding = (
            f"Best pooled row by lower bound: `{best.mapping}` / `{best.execution}` / "
            f"W={int(best.fill_window_sec)} / H={int(best.timeout_sec)} with conditional EV "
            f"`{cents(float(best.mean_pnl_cents))}` CI `[{cents(float(best.ci_lo_cents))}, "
            f"{cents(float(best.ci_hi_cents))}]`, n `{int(best.n_executed):,}`, executed fill "
            f"`{pct(float(best.executed_fill_rate))}`."
        )
    else:
        deciding = "No executable row was produced."

    lines = [
        "---",
        "tags: [dali, a18, reversion, passive-maker, microprice, results]",
        "---",
        "",
        "# Block A18 Passive Reversion-To-Microprice Findings",
        "",
        "> Hub: [[COWORK]]",
        "",
        "## Read Trail",
        "",
        "- [[block_a1x_external_note_reconciliation]]",
        "- [[block_a13_tob_imbalance_findings]]",
        "- [[block_a15b_decoupled_findings]]",
        "- [[block_a16_binary_bet_findings]]",
        "- [[block_a14h_maker_non_overlap_findings]]",
        "",
        "## Headline",
        "",
        f"{headline} {deciding}",
        "",
        f"Verdict: **{v}**",
        "",
        "This closes only the local Dali TOB signal under this explicit reversion framing. It does not touch off-book cross-market lead-lag or true-L2 ideas.",
        "",
        "## Deciding Number",
        "",
        table_from_rows(ranked.head(8)),
        "",
        f"Rows with conditional market-cluster CI lower > 0: `{positive_rows}`. Rows with unconditional-after-fill CI lower > 0: `{uncond_positive_rows}`. Worst executed one-contract loss in this replay: `{cents(worst)}`.",
        "",
        "## Test 1: Binary Decile Fade To Microprice",
        "",
        table_from_rows(decile_primary),
        "",
        "## Test 2: Rolling-Rank Sizing",
        "",
        table_from_rows(rrank_primary),
        "",
        "The rolling-rank variant trades the same sign as a fade, but sizes each attempted position by `abs(rolling_rank)` on the 300s percentile scale. Conditional EV is reported per actual sized contract; unconditional EV is sized cents per signal opportunity after passive fill and non-overlap drag.",
        "",
        "## Family Diagnostics",
        "",
        family_table(surface),
        "",
        "Family and market rows are diagnostics, not kill-switches by themselves. The gate was pooled route/horizon with market-cluster CI.",
        "",
        "## Best Single-Market Diagnostics",
        "",
        market_diag_table(market),
        "",
        "Single-market positives do not reopen the signal because they do not have a market-cluster CI. They are useful only as live instrumentation hints if a pooled route had cleared.",
        "",
        "## Method",
        "",
        "- Input: `data/analysis/block_a1_features.parquet`, runs `a0`, `a0b`, `a0c`, and `a0c_roll`; no new capture.",
        f"- Quote states require complete books, finite bid/ask/microprice, positive spread, and `book_staleness_seconds <= {STALE_BOOK_MAX_SECONDS:g}`.",
        "- Signal: `tob_imbalance_level = direction_factor * tob_imbalance`, plus a 300s rolling percentile transform mapped to `[-1,+1]`.",
        "- Binary mapping: per-market signed top/bottom decile. Rolling mapping: all finite rolling-rank states, with position size `abs(rank)`.",
        "- Direction: fade the signal. If continuation would buy the market-direction token, this sells it; if continuation would sell, this buys it.",
        "- Target: entry-time weighted mid/microprice. Exit is first target touch or timeout. Exits are at bid for longs and ask for shorts.",
        "- Taker route: immediate executable touch entry, taker fee on entry and exit.",
        "- Passive route: post at the touch on the fade side. A long bid fills on a SELL print at or below bid; a short ask fills on a BUY print at or above ask, within W seconds. Entry receives the maker rebate; exit pays taker fee.",
        "- Non-overlap: one open position per market per grid cell. Passive unfilled orders do not block; filled passive orders block from fill to exit.",
        "- CI: market-cluster bootstrap over market clusters. Conditional EV resamples executed PnL per contract; unconditional EV resamples sized PnL divided by all signal opportunities in the sampled markets.",
        "",
        "## Data",
        "",
        markdown_table(["run", "rows", "markets"], data_rows),
        "",
        "## Outputs",
        "",
        f"- `{display_path(SURFACE_OUT)}`",
        f"- `{display_path(MARKET_OUT)}`",
        f"- `{display_path(EXEC_OUT)}`",
        "",
        f"Elapsed: `{elapsed:.1f}s`.",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_todo(surface: pd.DataFrame) -> None:
    if not TODO.exists():
        return
    text = TODO.read_text(encoding="utf-8")
    marker = "## dali"
    if marker not in text:
        return
    pooled = surface[surface["segment_type"].eq("all")].copy()
    positive_rows = int(pooled["survives_cluster_ci_gt_0"].sum()) if not pooled.empty else 0
    best = pooled.sort_values(["ci_lo_cents", "mean_pnl_cents"], ascending=False).iloc[0] if not pooled.empty else None
    if best is None:
        line = "- A18 passive reversion gate: no executable rows; see [[block_a18_passive_reversion_findings]]."
    else:
        line = (
            "- A18 passive reversion-to-microprice gate: "
            f"{positive_rows} pooled market-cluster CI-positive rows; best `{best.mapping}`/`{best.execution}` "
            f"W={int(best.fill_window_sec)} H={int(best.timeout_sec)} = {cents(float(best.mean_pnl_cents))} "
            f"CI [{cents(float(best.ci_lo_cents))}, {cents(float(best.ci_hi_cents))}], "
            f"exec fill {pct(float(best.executed_fill_rate))}. Verdict in [[block_a18_passive_reversion_findings]]."
        )
    if "A18 passive reversion-to-microprice gate:" in text:
        text = re.sub(r"- A18 passive reversion-to-microprice gate:.*", line, text)
    else:
        insert = text.index(marker) + len(marker)
        text = text[:insert] + "\n" + line + text[insert:]
    TODO.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mappings", default="binary_decile,rolling_rank_sizing")
    parser.add_argument("--horizons", default=",".join(str(x) for x in HORIZONS))
    parser.add_argument("--passive-windows", default=",".join(str(x) for x in PASSIVE_WINDOWS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mappings = tuple(part.strip() for part in args.mappings.split(",") if part.strip())
    horizons = parse_ints(args.horizons, HORIZONS)
    windows = parse_ints(args.passive_windows, PASSIVE_WINDOWS)
    started = time.time()
    df = add_signal_features(load_features())
    data = build_replay_data(df)
    surface, market, executed = run_replay(data, mappings, horizons, windows)
    surface.attrs["run_counts"] = data.run_counts
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    surface.to_csv(SURFACE_OUT, index=False)
    market.to_csv(MARKET_OUT, index=False)
    executed.to_csv(EXEC_OUT, index=False)
    elapsed = time.time() - started
    write_note(surface, market, executed, elapsed)
    update_todo(surface)
    print(f"wrote {display_path(SURFACE_OUT)} rows={len(surface):,}", flush=True)
    print(f"wrote {display_path(MARKET_OUT)} rows={len(market):,}", flush=True)
    print(f"wrote {display_path(EXEC_OUT)} rows={len(executed):,}", flush=True)
    print(f"wrote {display_path(NOTE)}", flush=True)
    print(f"elapsed={elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
