"""Replay Dali CLOB market-channel JSONL into feature rows.

Input comes from ``dali_live_clob_capture.py``. The output is one row per
asset touched by each market-channel message, with maintained-book top of book,
depth, trade side, receive-lag telemetry, and Cont-Kukanov-Stoikov OFI.

Important convention: ``best_bid_ask`` events have no size information. They
are stored as telemetry fields only and never mutate the maintained executable
book used for OFI or simulated fills.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lib.clob_book import ClobBook, OfiContribution


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_DIR = ROOT / "data" / "live_clob"
DEFAULT_OUT_DIR = ROOT / "data" / "analysis"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


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
    telemetry_best_bid: float | None = None
    telemetry_best_ask: float | None = None
    last_trade_price: float | None = None
    last_trade_side: str = ""
    last_trade_size: float | None = None
    last_depth_update_received_at: pd.Timestamp | None = None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_ts_ms(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric > 10_000_000_000:
        return pd.to_datetime(numeric, unit="ms", utc=True)
    return pd.to_datetime(numeric, unit="s", utc=True)


def parse_received_at(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.to_datetime(value, utc=True)


def parse_levels(raw: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        price = as_float(item.get("price"))
        size = as_float(item.get("size"))
        if price is None or size is None:
            continue
        out.append((price, size))
    return out


def level_stats(
    levels: list[tuple[float, float]],
    side: str,
    top_n: int,
) -> dict[str, float | int | None]:
    top = levels[:top_n]
    if not top:
        return {
            f"{side}_levels": 0,
            f"{side}_top{top_n}_shares": None,
            f"{side}_top{top_n}_notional": None,
        }
    shares = sum(size for _, size in top)
    notional = sum(price * size for price, size in top)
    return {
        f"{side}_levels": len(levels),
        f"{side}_top{top_n}_shares": shares,
        f"{side}_top{top_n}_notional": notional,
    }


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    manifest = path.with_suffix(".manifest.json")
    if not manifest.exists():
        return {}
    raw = json.loads(manifest.read_text())
    return {str(token["token_id"]): token for token in raw.get("tokens", [])}


def ensure_state(
    states: dict[str, AssetState],
    asset_id: str,
    meta: dict[str, dict[str, Any]],
    market: str = "",
) -> AssetState:
    state = states.get(asset_id)
    if state is None:
        token_meta = meta.get(asset_id, {})
        state = AssetState(
            asset_id=asset_id,
            market=market,
            market_id=str(token_meta.get("market_id") or ""),
            question=str(token_meta.get("question") or ""),
            slug=str(token_meta.get("slug") or ""),
            family=str(token_meta.get("family") or ""),
            outcome_index=token_meta.get("outcome_index"),
        )
        states[asset_id] = state
    if market and not state.market:
        state.market = market
    return state


def book_staleness_seconds(
    state: AssetState,
    received_at: pd.Timestamp | None,
) -> float | None:
    if state.last_depth_update_received_at is None or received_at is None:
        return None
    return (received_at - state.last_depth_update_received_at).total_seconds()


def base_row(
    rec: dict[str, Any],
    msg: dict[str, Any],
    state: AssetState,
    top_n: int,
    event_asset_count: int,
    ofi: OfiContribution | None = None,
) -> dict[str, Any]:
    received_at = parse_received_at(rec.get("received_at"))
    exchange_ts = parse_ts_ms(msg.get("timestamp"))
    lag_ms = None
    if received_at is not None and exchange_ts is not None:
        lag_ms = (received_at - exchange_ts).total_seconds() * 1000

    top = state.book.top()
    best_bid = top.bid_price
    best_ask = top.ask_price
    spread = state.book.spread()
    mid = state.book.mid()
    bid_levels = state.book.levels("bid")
    ask_levels = state.book.levels("ask")
    bid_stats = level_stats(bid_levels, "bid", top_n)
    ask_stats = level_stats(ask_levels, "ask", top_n)
    bid_shares = bid_stats[f"bid_top{top_n}_shares"]
    ask_shares = ask_stats[f"ask_top{top_n}_shares"]
    book_imbalance = None
    if bid_shares is not None and ask_shares is not None and bid_shares + ask_shares:
        book_imbalance = (bid_shares - ask_shares) / (bid_shares + ask_shares)

    tob_imbalance = None
    if top.bid_size is not None and top.ask_size is not None and top.bid_size + top.ask_size:
        tob_imbalance = (top.bid_size - top.ask_size) / (top.bid_size + top.ask_size)

    event_type = str(rec.get("event_type") or msg.get("event_type") or "")
    ofi = ofi or OfiContribution()
    live_latency = None
    if event_type in {"price_change", "best_bid_ask", "last_trade_price"} and state.book.is_complete:
        live_latency = lag_ms

    row: dict[str, Any] = {
        "received_at": received_at,
        "exchange_ts": exchange_ts,
        "receive_lag_ms": lag_ms,
        "latency_book_event_ms": lag_ms if event_type == "book" else None,
        "latency_live_update_ms": live_latency,
        "event_type": event_type,
        "asset_id": state.asset_id,
        "market": state.market or msg.get("market") or "",
        "market_id": state.market_id,
        "family": state.family,
        "slug": state.slug,
        "question": state.question,
        "outcome_index": state.outcome_index,
        "event_asset_count": event_asset_count,
        "is_book_state_complete": state.book.is_complete,
        "book_staleness_seconds": book_staleness_seconds(state, received_at),
        "best_bid": best_bid,
        "best_bid_size": top.bid_size,
        "best_ask": best_ask,
        "best_ask_size": top.ask_size,
        "spread": spread,
        "mid": mid,
        "telemetry_best_bid": state.telemetry_best_bid,
        "telemetry_best_ask": state.telemetry_best_ask,
        "telemetry_spread": (
            state.telemetry_best_ask - state.telemetry_best_bid
            if state.telemetry_best_bid is not None and state.telemetry_best_ask is not None
            else None
        ),
        "last_trade_price": state.last_trade_price,
        "last_trade_side": state.last_trade_side,
        "last_trade_size": state.last_trade_size,
        "book_imbalance_top_n": book_imbalance,
        "tob_imbalance": tob_imbalance,
        "ofi_bid_event": ofi.bid,
        "ofi_ask_event": ofi.ask,
        "ofi_combined_event": ofi.combined,
        **bid_stats,
        **ask_stats,
    }
    return row


def mark_depth_update(state: AssetState, rec: dict[str, Any]) -> None:
    state.last_depth_update_received_at = parse_received_at(rec.get("received_at"))


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["asset_id", "received_at"]).reset_index(drop=True)
    df["is_trade"] = df["event_type"].eq("last_trade_price").astype(int)
    side_sign = df["last_trade_side"].map({"BUY": 1.0, "SELL": -1.0}).fillna(0.0)
    df["signed_trade_size"] = (
        df["is_trade"] * side_sign * df["last_trade_size"].fillna(0.0)
    )
    df["mid_change"] = df.groupby("asset_id")["mid"].diff()

    pieces = []
    for _, group in df.groupby("asset_id", sort=False):
        group = group.set_index("received_at", drop=False).sort_index()
        for window in ("5s", "15s", "60s", "300s"):
            suffix = window[:-1]
            group[f"trade_count_{suffix}s"] = group["is_trade"].rolling(window).sum().to_numpy()
            group[f"signed_trade_size_{suffix}s"] = (
                group["signed_trade_size"].rolling(window).sum().to_numpy()
            )
            group[f"mid_change_{suffix}s"] = group["mid_change"].rolling(window).sum().to_numpy()
            group[f"ofi_bid_{suffix}s"] = group["ofi_bid_event"].rolling(window).sum().to_numpy()
            group[f"ofi_ask_{suffix}s"] = group["ofi_ask_event"].rolling(window).sum().to_numpy()
            group[f"ofi_combined_{suffix}s"] = (
                group["ofi_combined_event"].rolling(window).sum().to_numpy()
            )

        roll = group["ofi_combined_5s"].rolling("60s")
        mean = roll.mean()
        std = roll.std().replace(0, pd.NA)
        group["ofi_zscore_5s_60s"] = ((group["ofi_combined_5s"] - mean) / std).to_numpy()
        group["ofi_typical_abs_60s"] = (
            group["ofi_combined_5s"].abs().rolling("60s").mean().to_numpy()
        )
        pieces.append(group.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def replay(path: Path, top_n: int) -> pd.DataFrame:
    meta = load_manifest(path)
    states: dict[str, AssetState] = {}
    rows: list[dict[str, Any]] = []

    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            event_type = str(rec.get("event_type") or msg.get("event_type") or "")

            if event_type == "book":
                asset_id = str(msg.get("asset_id") or "")
                if not asset_id:
                    continue
                state = ensure_state(states, asset_id, meta, str(msg.get("market") or ""))
                ofi = state.book.replace(
                    parse_levels(msg.get("bids")),
                    parse_levels(msg.get("asks")),
                )
                mark_depth_update(state, rec)
                last = as_float(msg.get("last_trade_price"))
                if last is not None:
                    state.last_trade_price = last
                rows.append(base_row(rec, msg, state, top_n, 1, ofi))

            elif event_type == "best_bid_ask":
                asset_id = str(msg.get("asset_id") or "")
                if not asset_id:
                    continue
                state = ensure_state(states, asset_id, meta, str(msg.get("market") or ""))
                state.telemetry_best_bid = as_float(msg.get("best_bid"))
                state.telemetry_best_ask = as_float(msg.get("best_ask"))
                rows.append(base_row(rec, msg, state, top_n, 1, OfiContribution()))

            elif event_type == "last_trade_price":
                asset_id = str(msg.get("asset_id") or "")
                if not asset_id:
                    continue
                state = ensure_state(states, asset_id, meta, str(msg.get("market") or ""))
                state.last_trade_price = as_float(msg.get("price"))
                state.last_trade_side = str(msg.get("side") or "")
                state.last_trade_size = as_float(msg.get("size"))
                row = base_row(rec, msg, state, top_n, 1, OfiContribution())
                row["trade_price"] = as_float(msg.get("price"))
                row["trade_side"] = str(msg.get("side") or "")
                row["trade_size"] = as_float(msg.get("size"))
                row["transaction_hash"] = str(msg.get("transaction_hash") or "")
                rows.append(row)

            elif event_type == "price_change":
                changes = msg.get("price_changes") or []
                changes = [item for item in changes if isinstance(item, dict)]
                for change in changes:
                    asset_id = str(change.get("asset_id") or "")
                    if not asset_id:
                        continue
                    state = ensure_state(states, asset_id, meta, str(msg.get("market") or ""))
                    state.telemetry_best_bid = as_float(change.get("best_bid"))
                    state.telemetry_best_ask = as_float(change.get("best_ask"))
                    price = as_float(change.get("price"))
                    size = as_float(change.get("size"))
                    side = str(change.get("side") or "").upper()
                    ofi = OfiContribution()
                    if price is not None and size is not None and side in {"BUY", "SELL"}:
                        ofi = state.book.update_level(side, price, size)
                        mark_depth_update(state, rec)
                    row = base_row(rec, msg, state, top_n, len(changes), ofi)
                    row["change_price"] = price
                    row["change_side"] = side
                    row["change_size"] = size
                    row["change_hash"] = str(change.get("hash") or "")
                    rows.append(row)

    return add_rolling_features(pd.DataFrame(rows))


def default_out(path: Path, out_dir: Path) -> Path:
    stem = path.stem.replace("dali_clob_", "dali_clob_features_")
    return out_dir / f"{stem}.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="?", type=Path)
    parser.add_argument("--latest", action="store_true", help="use latest data/live_clob/*.jsonl")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.latest:
        candidates = sorted(DEFAULT_IN_DIR.glob("*.jsonl"))
        if not candidates:
            raise SystemExit(f"no JSONL captures found in {DEFAULT_IN_DIR}")
        path = candidates[-1]
    elif args.jsonl:
        path = args.jsonl
    else:
        raise SystemExit("pass a JSONL path or --latest")

    out = args.out or default_out(path, args.out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = replay(path, args.top_n)
    df.to_parquet(out, index=False)
    print(f"input: {display_path(path)}")
    print(f"output: {display_path(out)}")
    print(f"rows: {len(df):,}")
    if len(df):
        print(f"assets: {df['asset_id'].nunique():,}")
        print(f"event types: {df['event_type'].value_counts().to_dict()}")
        lag = df["receive_lag_ms"].dropna()
        if len(lag):
            print(
                "receive lag ms: "
                f"median={lag.median():.1f}, p90={lag.quantile(0.90):.1f}, "
                f"p99={lag.quantile(0.99):.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
