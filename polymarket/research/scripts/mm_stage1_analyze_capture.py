"""Analyze MM Stage-1 first-mover CLOB capture runs.

Read-only post-capture analyzer. It consumes the JSONL envelopes written by
``scripts/dali_block_a0_capture.py`` and reports market/category quotability
plus a first pass at toxicity:

- spread and top/depth distribution from book / price_change / best_bid_ask
- trade-arrival rate from last_trade_price events when present
- post-trade mid drift in the trade direction using captured PM mids only

This intentionally does not use settlement, mark-to-mid PnL, or external
reference data. It is a Stage-1 measurement script, not a backtest.
"""
from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "data" / "live_clob" / "mm_stage1_first_mover"
DEFAULT_OUT_DIR = ROOT / "data" / "analysis" / "csv_outputs" / "market_making"


@dataclass(frozen=True)
class TokenMeta:
    token_id: str
    market_id: str
    slug: str
    question: str
    family: str
    created_at: str


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def event_ts(rec: dict[str, Any], msg: dict[str, Any]) -> pd.Timestamp:
    raw = msg.get("timestamp")
    if raw not in (None, ""):
        try:
            val = int(float(raw))
            unit = "ms" if val > 10_000_000_000 else "s"
            return pd.to_datetime(val, unit=unit, utc=True)
        except (TypeError, ValueError, OverflowError):
            pass
    return pd.to_datetime(rec.get("received_at"), utc=True)


def best_book_side(rows: list[dict[str, Any]], *, is_bid: bool) -> tuple[float | None, float | None]:
    parsed: list[tuple[float, float]] = []
    for row in rows or []:
        price = parse_float(row.get("price"))
        size = parse_float(row.get("size"))
        if price is None or size is None:
            continue
        parsed.append((price, size))
    if not parsed:
        return None, None
    return (max if is_bid else min)(parsed, key=lambda x: x[0])


def depth_near(rows: list[dict[str, Any]], best: float | None, *, is_bid: bool, ticks: float = 0.02) -> float:
    if best is None:
        return 0.0
    total = 0.0
    for row in rows or []:
        price = parse_float(row.get("price"))
        size = parse_float(row.get("size"))
        if price is None or size is None:
            continue
        if is_bid and price >= best - ticks:
            total += size
        if not is_bid and price <= best + ticks:
            total += size
    return total


def load_token_meta(run_dir: Path) -> dict[str, TokenMeta]:
    cfg_path = run_dir / "capture_config.yaml"
    config = yaml.safe_load(cfg_path.read_text()) or {}
    out: dict[str, TokenMeta] = {}
    for market in config.get("markets") or []:
        for token_id in market.get("clob_token_ids") or []:
            out[str(token_id)] = TokenMeta(
                token_id=str(token_id),
                market_id=str(market.get("id") or ""),
                slug=str(market.get("slug") or ""),
                question=str(market.get("question") or ""),
                family=str(market.get("family") or market.get("category") or ""),
                created_at=str(market.get("created_at") or ""),
            )
    return out


def iter_records(run_dir: Path):
    for path in sorted(run_dir.glob("*.jsonl")):
        if path.name == "capture_gaps.jsonl":
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                yield json.loads(line)


def parse_capture(run_dir: Path, token_meta: dict[str, TokenMeta]) -> tuple[pd.DataFrame, pd.DataFrame]:
    quote_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for rec in iter_records(run_dir):
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        et = str(rec.get("event_type") or msg.get("event_type") or "")
        ts = event_ts(rec, msg)

        if et == "book":
            token_id = str(msg.get("asset_id") or "")
            bid, bid_size = best_book_side(msg.get("bids") or [], is_bid=True)
            ask, ask_size = best_book_side(msg.get("asks") or [], is_bid=False)
            if token_id and bid is not None and ask is not None:
                meta = token_meta.get(token_id)
                quote_rows.append(
                    {
                        "ts": ts,
                        "event_type": et,
                        "token_id": token_id,
                        "market_id": meta.market_id if meta else str(msg.get("market") or ""),
                        "slug": meta.slug if meta else "",
                        "family": meta.family if meta else "",
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2.0,
                        "spread": ask - bid,
                        "bid_size": bid_size,
                        "ask_size": ask_size,
                        "bid_depth_2c": depth_near(msg.get("bids") or [], bid, is_bid=True),
                        "ask_depth_2c": depth_near(msg.get("asks") or [], ask, is_bid=False),
                    }
                )
        elif et == "price_change":
            for change in msg.get("price_changes") or []:
                if not isinstance(change, dict):
                    continue
                token_id = str(change.get("asset_id") or "")
                bid = parse_float(change.get("best_bid"))
                ask = parse_float(change.get("best_ask"))
                if token_id and bid is not None and ask is not None:
                    meta = token_meta.get(token_id)
                    quote_rows.append(
                        {
                            "ts": ts,
                            "event_type": et,
                            "token_id": token_id,
                            "market_id": meta.market_id if meta else str(msg.get("market") or ""),
                            "slug": meta.slug if meta else "",
                            "family": meta.family if meta else "",
                            "bid": bid,
                            "ask": ask,
                            "mid": (bid + ask) / 2.0,
                            "spread": ask - bid,
                            "bid_size": None,
                            "ask_size": None,
                            "bid_depth_2c": None,
                            "ask_depth_2c": None,
                        }
                    )
        elif et == "best_bid_ask":
            token_id = str(msg.get("asset_id") or "")
            bid = parse_float(msg.get("best_bid"))
            ask = parse_float(msg.get("best_ask"))
            if token_id and bid is not None and ask is not None:
                meta = token_meta.get(token_id)
                quote_rows.append(
                    {
                        "ts": ts,
                        "event_type": et,
                        "token_id": token_id,
                        "market_id": meta.market_id if meta else str(msg.get("market") or ""),
                        "slug": meta.slug if meta else "",
                        "family": meta.family if meta else "",
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2.0,
                        "spread": ask - bid,
                        "bid_size": None,
                        "ask_size": None,
                        "bid_depth_2c": None,
                        "ask_depth_2c": None,
                    }
                )
        elif et == "last_trade_price":
            token_id = str(msg.get("asset_id") or msg.get("assetId") or "")
            price = parse_float(msg.get("price") or msg.get("last_trade_price"))
            side_raw = str(msg.get("side") or msg.get("trade_side") or "").upper()
            direction = 1 if side_raw == "BUY" else -1 if side_raw == "SELL" else None
            if token_id and price is not None:
                meta = token_meta.get(token_id)
                trade_rows.append(
                    {
                        "ts": ts,
                        "token_id": token_id,
                        "market_id": meta.market_id if meta else str(msg.get("market") or ""),
                        "slug": meta.slug if meta else "",
                        "family": meta.family if meta else "",
                        "price": price,
                        "side": side_raw,
                        "direction": direction,
                    }
                )

    return pd.DataFrame(quote_rows), pd.DataFrame(trade_rows)


def attach_trade_drift(quotes: pd.DataFrame, trades: pd.DataFrame, horizons_s: list[int]) -> pd.DataFrame:
    if quotes.empty or trades.empty:
        return trades

    quote_by_token = {
        token: frame.sort_values("ts").reset_index(drop=True)
        for token, frame in quotes.dropna(subset=["mid"]).groupby("token_id")
    }

    rows: list[dict[str, Any]] = []
    for trade in trades.to_dict("records"):
        q = quote_by_token.get(trade["token_id"])
        row = dict(trade)
        if q is None or q.empty:
            rows.append(row)
            continue
        ts_values = q["ts"].tolist()
        prior_idx = bisect_left(ts_values, trade["ts"]) - 1
        prior_mid = q.iloc[prior_idx]["mid"] if prior_idx >= 0 else None
        direction = trade.get("direction")
        if direction is None and prior_mid is not None:
            direction = 1 if trade["price"] >= prior_mid else -1
            row["direction"] = direction
        row["prior_mid"] = prior_mid
        for horizon in horizons_s:
            target = trade["ts"] + pd.Timedelta(seconds=horizon)
            idx = bisect_left(ts_values, target)
            future_mid = q.iloc[idx]["mid"] if idx < len(q) else None
            row[f"mid_after_{horizon}s"] = future_mid
            if prior_mid is not None and future_mid is not None and direction is not None:
                row[f"trade_direction_mid_drift_cents_{horizon}s"] = 100.0 * direction * (future_mid - prior_mid)
            else:
                row[f"trade_direction_mid_drift_cents_{horizon}s"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(quotes: pd.DataFrame, trades: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame()

    span_hours = (
        (quotes["ts"].max() - quotes["ts"].min()).total_seconds() / 3600.0
        if len(quotes) > 1
        else 0.0
    )
    trade_counts = (
        trades.groupby("market_id").size().rename("trade_events").to_frame()
        if not trades.empty
        else pd.DataFrame(columns=["trade_events"])
    )

    grouped = quotes.groupby(["family", "market_id", "slug"], dropna=False)
    rows = []
    for keys, g in grouped:
        family, market_id, slug = keys
        row = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "family": family,
            "market_id": market_id,
            "slug": slug,
            "quote_events": len(g),
            "book_events": int(g["event_type"].eq("book").sum()),
            "capture_span_hours": span_hours,
            "median_spread": float(g["spread"].median()),
            "p25_spread": float(g["spread"].quantile(0.25)),
            "p75_spread": float(g["spread"].quantile(0.75)),
            "median_top_bid_size": float(g["bid_size"].dropna().median()) if g["bid_size"].notna().any() else None,
            "median_top_ask_size": float(g["ask_size"].dropna().median()) if g["ask_size"].notna().any() else None,
            "median_bid_depth_2c": float(g["bid_depth_2c"].dropna().median()) if g["bid_depth_2c"].notna().any() else None,
            "median_ask_depth_2c": float(g["ask_depth_2c"].dropna().median()) if g["ask_depth_2c"].notna().any() else None,
            "trade_events": int(trade_counts.loc[market_id, "trade_events"]) if market_id in trade_counts.index else 0,
        }
        row["trade_events_per_hour"] = row["trade_events"] / span_hours if span_hours > 0 else None
        for horizon in (60, 300):
            col = f"trade_direction_mid_drift_cents_{horizon}s"
            if not trades.empty and col in trades:
                sub = trades[trades["market_id"].eq(market_id)][col].dropna()
                row[f"mean_trade_direction_mid_drift_cents_{horizon}s"] = float(sub.mean()) if len(sub) else None
                row[f"n_drift_obs_{horizon}s"] = int(len(sub))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["family", "quote_events"], ascending=[True, False])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", default="mm_stage1_capture")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    token_meta = load_token_meta(run_dir)
    quotes, trades = parse_capture(run_dir, token_meta)
    trades = attach_trade_drift(quotes, trades, [60, 300])
    summary = summarize(quotes, trades, run_dir)

    summary_path = args.out_dir / f"{args.out_prefix}_market_summary.csv"
    quote_path = args.out_dir / f"{args.out_prefix}_quote_events.csv"
    trade_path = args.out_dir / f"{args.out_prefix}_trade_drift.csv"
    summary.to_csv(summary_path, index=False)
    quotes.to_csv(quote_path, index=False)
    trades.to_csv(trade_path, index=False)

    print(f"quotes: {len(quotes):,} -> {quote_path.relative_to(ROOT)}")
    print(f"trades: {len(trades):,} -> {trade_path.relative_to(ROOT)}")
    print(f"summary rows: {len(summary):,} -> {summary_path.relative_to(ROOT)}")
    if not summary.empty:
        print(summary[["family", "slug", "quote_events", "median_spread", "trade_events_per_hour"]].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
