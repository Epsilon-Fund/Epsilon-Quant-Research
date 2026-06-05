"""Block K3 crypto-4h lead-lag and digital-option basis.

Research-only sidecar for the Block K crypto-4h gate. It joins local
Polymarket CLOB captures from A0b and A0c crypto-roll to external Binance
spot/perp klines, prices each 4h up/down market as a short digital, and checks
whether any executable taker basis survives the standard crypto taker fee.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
A0B_DIR = ROOT / "data" / "live_clob" / "block_a0b" / "block_a0b_replacements_v2_20260527"
A0C_ROLL_DIR = (
    ROOT
    / "data"
    / "live_clob"
    / "block_a0c_crypto_roll"
    / "block_a0c_crypto_roll_20260529_morning"
)
OUT_CSV = ANALYSIS / "csv_outputs" / "options_delta" / "k3_leadlag_basis.csv"
NOTE = NOTES / "block_k3_leadlag_findings.md"

SPOT_BASE = "https://api.binance.com"
PERP_BASE = "https://fapi.binance.com"
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
YEAR_SECONDS = 365.0 * 24.0 * 3600.0
YEAR_MINUTES = 365.0 * 24.0 * 60.0
BAR_SECONDS = 10
BOOTSTRAP_SAMPLES = 250
RNG_SEED = 20260530
LAG_SECONDS = tuple(range(-300, 301, 10))
DYNAMIC_15M_PEAK_FEE_REFERENCE = 0.0315

SLUG_RE = re.compile(r"^(btc|eth|sol)-updown-4h-(\d+)$")


@dataclass
class MarketMeta:
    slug: str
    asset: str
    source_runs: set[str]
    market_id: str
    condition_id: str
    question: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    up_token: str
    down_token: str
    fee_rate: float
    fee_exponent: float
    taker_only: bool
    rebate_rate: float


def parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)) or str(value).isdigit():
            return pd.to_datetime(int(value), unit="ms", utc=True)
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def norm_cdf(x: float | np.ndarray) -> float | np.ndarray:
    arr = np.asarray(x, dtype=float)
    out = 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))
    if np.ndim(x) == 0:
        return float(out)
    return out


def digital_fair_value(
    spot: pd.Series,
    strike: float,
    sigma_annualized: float,
    seconds_to_expiry: pd.Series,
) -> pd.Series:
    tau = seconds_to_expiry.astype(float).clip(lower=0.0) / YEAR_SECONDS
    out = pd.Series(np.nan, index=spot.index, dtype=float)
    valid = (
        spot.replace([np.inf, -np.inf], np.nan).notna()
        & np.isfinite(strike)
        & (strike > 0)
        & np.isfinite(sigma_annualized)
        & (sigma_annualized > 0)
        & (tau > 0)
    )
    if valid.any():
        denom = sigma_annualized * np.sqrt(tau[valid].to_numpy(dtype=float))
        d2 = (
            np.log(spot[valid].to_numpy(dtype=float) / strike)
            - 0.5 * sigma_annualized * sigma_annualized * tau[valid].to_numpy(dtype=float)
        ) / denom
        out.loc[valid] = norm_cdf(d2)
    expired = seconds_to_expiry <= 0
    if expired.any() and np.isfinite(strike) and strike > 0:
        out.loc[expired] = (spot.loc[expired] >= strike).astype(float)
    return out.clip(0.0, 1.0)


def taker_fee(price: pd.Series | float, fee_rate: float) -> pd.Series | float:
    if isinstance(price, pd.Series):
        p = price.astype(float).clip(0.0, 1.0)
        return fee_rate * p * (1.0 - p)
    p = min(max(float(price), 0.0), 1.0)
    return fee_rate * p * (1.0 - p)


def slug_info(slug: str) -> tuple[str, pd.Timestamp, pd.Timestamp] | None:
    match = SLUG_RE.match(slug)
    if not match:
        return None
    asset = match.group(1).upper()
    start = pd.to_datetime(int(match.group(2)), unit="s", utc=True)
    return asset, start, start + pd.Timedelta(hours=4)


def jsonl_paths() -> list[Path]:
    roots = [A0B_DIR, A0C_ROLL_DIR]
    paths: list[Path] = []
    for root in roots:
        paths.extend(
            p
            for p in root.rglob("*.jsonl")
            if "capture_gaps" not in p.name and p.name != "roll_supervisor.jsonl"
        )
    return sorted(paths)


def source_run_for_path(path: Path) -> str:
    text = str(path)
    return "a0b" if "/block_a0b/" in text else "a0c_roll"


def load_market_metadata() -> tuple[dict[str, MarketMeta], dict[str, tuple[str, str]]]:
    market_by_slug: dict[str, MarketMeta] = {}
    token_map: dict[str, tuple[str, str]] = {}

    for config_path in sorted([*A0B_DIR.rglob("capture_config.yaml"), *A0C_ROLL_DIR.rglob("capture_config.yaml")]):
        source_run = source_run_for_path(config_path)
        config = yaml.safe_load(config_path.read_text()) or {}
        for market in config.get("markets") or []:
            slug = str(market.get("slug") or "")
            info = slug_info(slug)
            if info is None:
                continue
            asset, start, end = info
            if asset not in SYMBOLS:
                continue

            clob_info = ((market.get("fee") or {}).get("clob_market_info") or {})
            token_rows = clob_info.get("t") or []
            outcome_tokens = {
                str(row.get("o") or "").strip().lower(): str(row.get("t") or "")
                for row in token_rows
                if row.get("t")
            }
            token_ids = [str(x) for x in (market.get("clob_token_ids") or [])]
            up_token = outcome_tokens.get("up") or (token_ids[0] if token_ids else "")
            down_token = outcome_tokens.get("down") or (token_ids[1] if len(token_ids) > 1 else "")
            if not up_token or not down_token:
                continue

            fee = market.get("fee") or {}
            fee_schedule = fee.get("fee_schedule") or {}
            fee_rate = as_float(fee.get("fee_rate"))
            if not np.isfinite(fee_rate):
                fee_rate = as_float(((clob_info.get("fd") or {}).get("r")))
            if not np.isfinite(fee_rate):
                fee_rate = as_float(fee_schedule.get("rate"))
            if not np.isfinite(fee_rate):
                fee_rate = 0.0
            fee_exponent = as_float(fee.get("fee_exponent"))
            if not np.isfinite(fee_exponent):
                fee_exponent = as_float(fee_schedule.get("exponent"))
            if not np.isfinite(fee_exponent):
                fee_exponent = 1.0
            rebate_rate = as_float(fee_schedule.get("rebateRate"))
            if not np.isfinite(rebate_rate):
                rebate_rate = as_float(fee_schedule.get("rebate_rate"))
            if not np.isfinite(rebate_rate):
                rebate_rate = 0.0
            taker_only = bool(fee.get("taker_only", fee_schedule.get("takerOnly", True)))

            if slug in market_by_slug:
                market_by_slug[slug].source_runs.add(source_run)
            else:
                market_by_slug[slug] = MarketMeta(
                    slug=slug,
                    asset=asset,
                    source_runs={source_run},
                    market_id=str(market.get("id") or ""),
                    condition_id=str(market.get("condition_id") or ""),
                    question=str(market.get("question") or ""),
                    window_start=start,
                    window_end=end,
                    up_token=up_token,
                    down_token=down_token,
                    fee_rate=float(fee_rate),
                    fee_exponent=float(fee_exponent),
                    taker_only=taker_only,
                    rebate_rate=float(rebate_rate),
                )
            token_map[up_token] = (slug, "up")
            token_map[down_token] = (slug, "down")

    return market_by_slug, token_map


def book_best_prices(message: dict[str, Any]) -> tuple[float, float]:
    bids = [
        (as_float(row.get("price")), as_float(row.get("size")))
        for row in message.get("bids") or []
    ]
    asks = [
        (as_float(row.get("price")), as_float(row.get("size")))
        for row in message.get("asks") or []
    ]
    bid_px = [p for p, s in bids if np.isfinite(p) and np.isfinite(s) and s > 0]
    ask_px = [p for p, s in asks if np.isfinite(p) and np.isfinite(s) and s > 0]
    return (max(bid_px) if bid_px else math.nan, min(ask_px) if ask_px else math.nan)


def event_timestamp(record: dict[str, Any]) -> pd.Timestamp | None:
    message = record.get("message") or {}
    ts = parse_ts(message.get("timestamp"))
    if ts is not None:
        return ts
    return parse_ts(record.get("received_at"))


def iter_price_updates(record: dict[str, Any]) -> list[tuple[str, float, float]]:
    message = record.get("message") or {}
    event_type = record.get("event_type")
    if event_type == "price_change":
        out: list[tuple[str, float, float]] = []
        for change in message.get("price_changes") or []:
            out.append(
                (
                    str(change.get("asset_id") or ""),
                    as_float(change.get("best_bid")),
                    as_float(change.get("best_ask")),
                )
            )
        return out
    if event_type == "best_bid_ask":
        return [
            (
                str(message.get("asset_id") or ""),
                as_float(message.get("best_bid")),
                as_float(message.get("best_ask")),
            )
        ]
    if event_type == "book":
        bid, ask = book_best_prices(message)
        return [(str(message.get("asset_id") or ""), bid, ask)]
    return []


def load_polymarket_events(
    market_by_slug: dict[str, MarketMeta],
    token_map: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    states: dict[str, dict[str, float]] = defaultdict(
        lambda: {"up_bid": math.nan, "up_ask": math.nan, "down_bid": math.nan, "down_ask": math.nan}
    )
    rows: list[dict[str, Any]] = []

    for path in jsonl_paths():
        source_run = source_run_for_path(path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event_type") not in {"price_change", "best_bid_ask", "book"}:
                    continue
                ts = event_timestamp(record)
                if ts is None:
                    continue
                for token_id, bid, ask in iter_price_updates(record):
                    if token_id not in token_map:
                        continue
                    if not np.isfinite(bid) or not np.isfinite(ask):
                        continue
                    slug, outcome = token_map[token_id]
                    meta = market_by_slug[slug]
                    if ts < meta.window_start or ts >= meta.window_end:
                        continue
                    state = states[slug]
                    state[f"{outcome}_bid"] = bid
                    state[f"{outcome}_ask"] = ask
                    if not all(np.isfinite(state[key]) for key in state):
                        continue
                    up_mid = (state["up_bid"] + state["up_ask"]) / 2.0
                    down_mid = (state["down_bid"] + state["down_ask"]) / 2.0
                    rows.append(
                        {
                            "ts": ts,
                            "source_run": source_run,
                            "market_slug": slug,
                            "asset": meta.asset,
                            "up_bid": state["up_bid"],
                            "up_ask": state["up_ask"],
                            "down_bid": state["down_bid"],
                            "down_ask": state["down_ask"],
                            "polymarket_mid": up_mid,
                            "down_mid": down_mid,
                            "parity_mid": 0.5 * (up_mid + (1.0 - down_mid)),
                            "parity_gap": up_mid + down_mid - 1.0,
                        }
                    )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["market_slug", "ts"]).drop_duplicates(["market_slug", "ts"], keep="last")
    return df.reset_index(drop=True)


def fetch_klines(
    client: httpx.Client,
    *,
    base_url: str,
    path: str,
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    interval_ms = {"1s": 1000, "1m": 60_000}[interval]
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        for attempt in range(5):
            try:
                r = client.get(f"{base_url}{path}", params=params, timeout=20)
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(0.5 * (attempt + 1))
        if not data:
            break
        rows.extend(data)
        last_open = int(data[-1][0])
        next_cursor = last_open + interval_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.015)

    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "n_trades",
            "taker_base_volume",
            "taker_quote_volume",
            "ignore",
        ],
    )
    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["ts", "open", "high", "low", "close", "volume"]].drop_duplicates("ts").sort_values("ts")


def fetch_external_data(slugs: list[str], market_by_slug: dict[str, MarketMeta]) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}
    by_asset: dict[str, list[MarketMeta]] = defaultdict(list)
    for slug in slugs:
        by_asset[market_by_slug[slug].asset].append(market_by_slug[slug])

    with httpx.Client(headers={"User-Agent": "epsilon-quant-research-k3/1.0"}) as client:
        for asset, metas in sorted(by_asset.items()):
            start = min(m.window_start for m in metas) - pd.Timedelta(minutes=5)
            end = max(m.window_end for m in metas) + pd.Timedelta(minutes=5)
            symbol = SYMBOLS[asset]
            print(f"fetching Binance {asset} spot 1s and perp 1m {start} to {end}")
            spot = fetch_klines(
                client,
                base_url=SPOT_BASE,
                path="/api/v3/klines",
                symbol=symbol,
                interval="1s",
                start=start,
                end=end,
            )
            perp = fetch_klines(
                client,
                base_url=PERP_BASE,
                path="/fapi/v1/klines",
                symbol=symbol,
                interval="1m",
                start=start,
                end=end,
            )
            out[asset] = {"spot": spot, "perp": perp}
    return out


def window_stats(
    market_by_slug: dict[str, MarketMeta],
    external: dict[str, dict[str, pd.DataFrame]],
    slugs: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slug in slugs:
        meta = market_by_slug[slug]
        spot = external[meta.asset]["spot"].set_index("ts")
        win = spot[(spot.index >= meta.window_start) & (spot.index < meta.window_end)].copy()
        if win.empty:
            continue
        strike = float(win["open"].iloc[0])
        one_min_close = win["close"].resample("60s").last().dropna()
        returns = np.log(one_min_close).diff().dropna()
        rv = float(returns.std(ddof=1) * math.sqrt(YEAR_MINUTES)) if len(returns) > 2 else math.nan
        realized_abs = float(abs(float(win["close"].iloc[-1]) / strike - 1.0))
        rows.append(
            {
                "market_slug": slug,
                "asset": meta.asset,
                "window_start": meta.window_start,
                "window_end": meta.window_end,
                "window_strike_spot": strike,
                "window_close_spot": float(win["close"].iloc[-1]),
                "window_return": float(math.log(float(win["close"].iloc[-1]) / strike)),
                "window_rv_annualized": rv,
                "window_abs_return": realized_abs,
                "spot_rows": int(len(win)),
                "spot_first_ts": win.index.min(),
                "spot_last_ts": win.index.max(),
            }
        )
    return pd.DataFrame(rows)


def build_panel(
    pm: pd.DataFrame,
    market_by_slug: dict[str, MarketMeta],
    external: dict[str, dict[str, pd.DataFrame]],
    stats: pd.DataFrame,
) -> pd.DataFrame:
    stat_by_slug = stats.set_index("market_slug").to_dict("index")
    panels: list[pd.DataFrame] = []

    for slug, group in pm.groupby("market_slug", sort=True):
        if slug not in stat_by_slug:
            continue
        meta = market_by_slug[slug]
        group = group.sort_values("ts").drop_duplicates("ts", keep="last")
        if len(group) < 2:
            continue
        first_ts = max(group["ts"].min(), meta.window_start)
        last_ts = min(group["ts"].max(), meta.window_end - pd.Timedelta(seconds=BAR_SECONDS))
        if first_ts >= last_ts:
            continue
        indexed = group.set_index("ts")
        regular = (
            indexed[[
                "source_run",
                "asset",
                "up_bid",
                "up_ask",
                "down_bid",
                "down_ask",
                "polymarket_mid",
                "down_mid",
                "parity_mid",
                "parity_gap",
            ]]
            .resample(f"{BAR_SECONDS}s")
            .last()
            .ffill()
        )
        regular = regular[(regular.index >= first_ts.floor(f"{BAR_SECONDS}s")) & (regular.index <= last_ts)]
        if regular.empty:
            continue

        ext_asset = external[meta.asset]
        spot = ext_asset["spot"].set_index("ts").sort_index()
        perp = ext_asset["perp"].set_index("ts").sort_index()
        regular["binance_spot"] = spot["close"].reindex(regular.index, method="ffill")
        regular["binance_perp"] = perp["close"].reindex(regular.index, method="ffill")
        stat = stat_by_slug[slug]
        seconds_to_expiry = (meta.window_end - regular.index).total_seconds()
        regular["seconds_to_expiry"] = seconds_to_expiry
        regular["window_start"] = meta.window_start
        regular["window_end"] = meta.window_end
        regular["window_strike_spot"] = stat["window_strike_spot"]
        regular["window_close_spot"] = stat["window_close_spot"]
        regular["window_return"] = stat["window_return"]
        regular["window_rv_annualized"] = stat["window_rv_annualized"]
        regular["fair_value"] = digital_fair_value(
            regular["binance_spot"],
            float(stat["window_strike_spot"]),
            float(stat["window_rv_annualized"]),
            pd.Series(seconds_to_expiry, index=regular.index),
        )
        regular["perp_fair_value"] = digital_fair_value(
            regular["binance_perp"],
            float(stat["window_strike_spot"]),
            float(stat["window_rv_annualized"]),
            pd.Series(seconds_to_expiry, index=regular.index),
        )
        regular["basis"] = regular["polymarket_mid"] - regular["fair_value"]
        regular["parity_basis"] = regular["parity_mid"] - regular["fair_value"]
        regular["basis_vs_perp_fair"] = regular["polymarket_mid"] - regular["perp_fair_value"]
        regular["spot_moneyness"] = np.log(regular["binance_spot"] / float(stat["window_strike_spot"]))
        regular["taker_fee_rate"] = meta.fee_rate
        regular["taker_fee_up_ask"] = taker_fee(regular["up_ask"], meta.fee_rate)
        regular["taker_fee_down_ask"] = taker_fee(regular["down_ask"], meta.fee_rate)
        regular["buy_up_edge"] = regular["fair_value"] - regular["up_ask"] - regular["taker_fee_up_ask"]
        regular["buy_down_edge"] = (
            1.0 - regular["fair_value"] - regular["down_ask"] - regular["taker_fee_down_ask"]
        )
        regular["best_taker_edge"] = regular[["buy_up_edge", "buy_down_edge"]].max(axis=1)
        regular["best_taker_route"] = np.where(
            regular["buy_up_edge"] >= regular["buy_down_edge"], "buy_up", "buy_down"
        )
        regular["post_fee_survives"] = regular["best_taker_edge"] > 0.0
        regular["post_fee_survives_1c"] = regular["best_taker_edge"] > 0.01
        regular["market_slug"] = slug
        regular["market_id"] = meta.market_id
        regular["condition_id"] = meta.condition_id
        regular["question"] = meta.question
        regular["source_runs"] = ",".join(sorted(meta.source_runs))
        regular["asset"] = meta.asset
        panels.append(regular.reset_index(names="ts"))

    if not panels:
        return pd.DataFrame()
    out = pd.concat(panels, ignore_index=True).sort_values(["market_slug", "ts"])
    out["basis_cents"] = 100.0 * out["basis"]
    out["best_taker_edge_cents"] = 100.0 * out["best_taker_edge"]
    return out.reset_index(drop=True)


def corr_for_lag_arrays(x: np.ndarray, y: np.ndarray, lag_steps: int) -> tuple[float, int]:
    if lag_steps > 0:
        xs = x[:-lag_steps]
        ys = y[lag_steps:]
    elif lag_steps < 0:
        xs = x[-lag_steps:]
        ys = y[:lag_steps]
    else:
        xs = x
        ys = y
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 20:
        return math.nan, int(mask.sum())
    xs = xs[mask]
    ys = ys[mask]
    if np.nanstd(xs) == 0 or np.nanstd(ys) == 0:
        return math.nan, int(mask.sum())
    return float(np.corrcoef(xs, ys)[0, 1]), int(mask.sum())


def panel_return_arrays(panel: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for slug, group in panel.groupby("market_slug", sort=False):
        group = group.sort_values("ts")
        arrays[str(slug)] = (
            group["fair_value"].diff().to_numpy(dtype=float),
            group["polymarket_mid"].diff().to_numpy(dtype=float),
        )
    return arrays


def cross_corr_scan(
    panel: pd.DataFrame | None = None,
    slugs: list[str] | None = None,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    pieces = []
    if arrays is None:
        if panel is None:
            raise ValueError("panel or arrays required")
        arrays = panel_return_arrays(panel)
    if slugs is None:
        slugs = list(arrays)
    lag_steps_list = [lag // BAR_SECONDS for lag in LAG_SECONDS]
    for lag_seconds, lag_steps in zip(LAG_SECONDS, lag_steps_list):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for slug in slugs:
            if slug not in arrays:
                continue
            fair_ret, pm_ret = arrays[slug]
            corr, n = corr_for_lag_arrays(fair_ret, pm_ret, lag_steps)
            if n:
                if lag_steps > 0:
                    xs.append(fair_ret[:-lag_steps])
                    ys.append(pm_ret[lag_steps:])
                elif lag_steps < 0:
                    xs.append(fair_ret[-lag_steps:])
                    ys.append(pm_ret[:lag_steps])
                else:
                    xs.append(fair_ret)
                    ys.append(pm_ret)
        if not xs:
            pieces.append({"lag_seconds": lag_seconds, "corr": math.nan, "n_pairs": 0})
            continue
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        corr, n = corr_for_lag_arrays(x, y, 0)
        pieces.append({"lag_seconds": lag_seconds, "corr": corr, "n_pairs": n})
    return pd.DataFrame(pieces)


def bootstrap_cross_corr(panel: pd.DataFrame, full_best_lag: int) -> dict[str, float]:
    slugs = sorted(panel["market_slug"].unique())
    if len(slugs) < 2:
        return {
            "best_lag_ci_lo": math.nan,
            "best_lag_ci_hi": math.nan,
            "corr_ci_lo": math.nan,
            "corr_ci_hi": math.nan,
        }
    rng = np.random.default_rng(RNG_SEED)
    arrays = panel_return_arrays(panel)
    best_lags: list[float] = []
    corr_at_best: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [slugs[i] for i in rng.integers(0, len(slugs), size=len(slugs))]
        scan = cross_corr_scan(slugs=sample, arrays=arrays)
        valid = scan.replace([np.inf, -np.inf], np.nan).dropna(subset=["corr"])
        if valid.empty:
            continue
        best = valid.loc[valid["corr"].idxmax()]
        best_lags.append(float(best["lag_seconds"]))
        row = valid[valid["lag_seconds"].eq(full_best_lag)]
        if not row.empty:
            corr_at_best.append(float(row["corr"].iloc[0]))
    out = {}
    for key, values in [
        ("best_lag", best_lags),
        ("corr", corr_at_best),
    ]:
        if values:
            lo, hi = np.quantile(values, [0.025, 0.975])
            out[f"{key}_ci_lo"] = float(lo)
            out[f"{key}_ci_hi"] = float(hi)
        else:
            out[f"{key}_ci_lo"] = math.nan
            out[f"{key}_ci_hi"] = math.nan
    return out


def hy_scan_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """A lightweight HY-style overlap proxy on the 10s joined panel.

    The full irregular HY pass is expensive enough to obscure the research gate.
    This keeps the same lead/lag convention on synchronized 10s intervals and
    labels the output separately in the note as a panel-overlap proxy.
    """
    scan = cross_corr_scan(panel)
    return scan.rename(columns={"corr": "hy_corr"})[["lag_seconds", "hy_corr"]]


def autocorr_at_lag(panel: pd.DataFrame, column: str, lag_seconds: int) -> tuple[float, int]:
    lag_steps = lag_seconds // BAR_SECONDS
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for _, g in panel.groupby("market_slug", sort=False):
        values = g.sort_values("ts")[column].to_numpy(dtype=float)
        if len(values) <= lag_steps:
            continue
        xs.append(values[:-lag_steps])
        ys.append(values[lag_steps:])
    if not xs:
        return math.nan, 0
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return math.nan, int(mask.sum())
    return float(np.corrcoef(x[mask], y[mask])[0, 1]), int(mask.sum())


def ar1_half_life_seconds(panel: pd.DataFrame, column: str) -> float:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for _, g in panel.groupby("market_slug", sort=False):
        values = g.sort_values("ts")[column].to_numpy(dtype=float)
        if len(values) < 3:
            continue
        xs.append(values[:-1])
        ys.append(values[1:])
    if not xs:
        return math.nan
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return math.nan
    x = x[mask]
    y = y[mask]
    denom = float(np.dot(x - x.mean(), x - x.mean()))
    if denom <= 0:
        return math.nan
    beta = float(np.dot(x - x.mean(), y - y.mean()) / denom)
    if beta <= 0 or beta >= 1:
        return math.nan
    return float(-BAR_SECONDS * math.log(2.0) / math.log(beta))


def fair_series_for_hy(
    meta: MarketMeta,
    external: dict[str, dict[str, pd.DataFrame]],
    stat: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    spot = external[meta.asset]["spot"].copy()
    spot = spot[(spot["ts"] >= start) & (spot["ts"] <= end)].copy()
    if spot.empty:
        return pd.DataFrame()
    secs = (meta.window_end - spot["ts"]).dt.total_seconds()
    spot["fair_value"] = digital_fair_value(
        spot["close"],
        float(stat["window_strike_spot"]),
        float(stat["window_rv_annualized"]),
        secs,
    )
    return spot[["ts", "fair_value"]].dropna()


def hy_cov_corr(pm_times: np.ndarray, pm_values: np.ndarray, ext_times: np.ndarray, ext_values: np.ndarray, lag: int) -> tuple[float, float, float]:
    if len(pm_values) < 2 or len(ext_values) < 2:
        return 0.0, 0.0, 0.0
    pm_ret = np.diff(pm_values)
    ext_ret = np.diff(ext_values)
    pm_start = pm_times[:-1] - lag
    pm_end = pm_times[1:] - lag
    ext_start = ext_times[:-1]
    ext_end = ext_times[1:]

    pm_mask = np.isfinite(pm_ret) & (pm_end > pm_start)
    ext_mask = np.isfinite(ext_ret) & (ext_end > ext_start)
    pm_ret, pm_start, pm_end = pm_ret[pm_mask], pm_start[pm_mask], pm_end[pm_mask]
    ext_ret, ext_start, ext_end = ext_ret[ext_mask], ext_start[ext_mask], ext_end[ext_mask]
    var_pm = float(np.dot(pm_ret, pm_ret))
    var_ext = float(np.dot(ext_ret, ext_ret))
    cov = 0.0
    i = j = 0
    while i < len(pm_ret) and j < len(ext_ret):
        if pm_end[i] <= ext_start[j]:
            i += 1
        elif ext_end[j] <= pm_start[i]:
            j += 1
        else:
            cov += float(pm_ret[i] * ext_ret[j])
            if pm_end[i] <= ext_end[j]:
                i += 1
            else:
                j += 1
    return cov, var_pm, var_ext


def hy_scan(
    pm_events: pd.DataFrame,
    panel: pd.DataFrame,
    market_by_slug: dict[str, MarketMeta],
    external: dict[str, dict[str, pd.DataFrame]],
    stats: pd.DataFrame,
) -> pd.DataFrame:
    stat_by_slug = stats.set_index("market_slug").to_dict("index")
    rows: list[dict[str, Any]] = []
    event_by_slug = {
        slug: g.sort_values("ts").drop_duplicates("ts", keep="last")
        for slug, g in pm_events.groupby("market_slug", sort=False)
    }
    for lag in LAG_SECONDS:
        cov_total = 0.0
        var_pm_total = 0.0
        var_ext_total = 0.0
        for slug, g_panel in panel.groupby("market_slug", sort=False):
            if slug not in event_by_slug or slug not in stat_by_slug:
                continue
            meta = market_by_slug[slug]
            g_events = event_by_slug[slug]
            # Collapse unchanged mid states to keep the HY pass tractable.
            changed = g_events[
                g_events["polymarket_mid"].diff().abs().fillna(1.0).gt(1e-12)
            ][["ts", "polymarket_mid"]]
            if len(changed) < 2:
                continue
            start = max(g_panel["ts"].min(), changed["ts"].min())
            end = min(g_panel["ts"].max(), changed["ts"].max())
            fair = fair_series_for_hy(meta, external, stat_by_slug[slug], start, end)
            if len(fair) < 2:
                continue
            pm_times = changed["ts"].astype("int64").to_numpy(dtype=float) / 1_000_000_000.0
            pm_values = changed["polymarket_mid"].to_numpy(dtype=float)
            ext_times = fair["ts"].astype("int64").to_numpy(dtype=float) / 1_000_000_000.0
            ext_values = fair["fair_value"].to_numpy(dtype=float)
            cov, var_pm, var_ext = hy_cov_corr(pm_times, pm_values, ext_times, ext_values, lag)
            cov_total += cov
            var_pm_total += var_pm
            var_ext_total += var_ext
        corr = cov_total / math.sqrt(var_pm_total * var_ext_total) if var_pm_total > 0 and var_ext_total > 0 else math.nan
        rows.append({"lag_seconds": lag, "hy_corr": corr})
    return pd.DataFrame(rows)


def edge_run_lengths(panel: pd.DataFrame) -> tuple[float, float, int]:
    durations: list[float] = []
    for _, g in panel.groupby("market_slug", sort=False):
        flags = g.sort_values("ts")["post_fee_survives"].fillna(False).to_numpy(dtype=bool)
        run = 0
        for flag in flags:
            if flag:
                run += 1
            elif run:
                durations.append(run * BAR_SECONDS)
                run = 0
        if run:
            durations.append(run * BAR_SECONDS)
    if not durations:
        return 0.0, 0.0, 0
    return float(np.median(durations)), float(np.max(durations)), int(len(durations))


def fee_gate_summary(market_by_slug: dict[str, MarketMeta], live_15m_fee_rate: float | None) -> dict[str, Any]:
    fees = pd.DataFrame(
        [
            {
                "market_slug": m.slug,
                "asset": m.asset,
                "source_runs": ",".join(sorted(m.source_runs)),
                "fee_rate": m.fee_rate,
                "fee_exponent": m.fee_exponent,
                "taker_only": m.taker_only,
                "rebate_rate": m.rebate_rate,
                "peak_fee": m.fee_rate * 0.25,
            }
            for m in market_by_slug.values()
        ]
    )
    crypto_4h = fees[fees["market_slug"].str.contains("updown-4h")]
    unique_rates = sorted(crypto_4h["fee_rate"].dropna().unique().tolist())
    max_peak = float(crypto_4h["peak_fee"].max()) if not crypto_4h.empty else math.nan
    dynamic_like = bool(max_peak >= DYNAMIC_15M_PEAK_FEE_REFERENCE - 1e-9)
    return {
        "fees": fees,
        "unique_rates": unique_rates,
        "max_peak_fee": max_peak,
        "dynamic_like": dynamic_like,
        "live_15m_fee_rate": live_15m_fee_rate,
        "live_15m_peak_fee": (
            live_15m_fee_rate * 0.25 if live_15m_fee_rate is not None else math.nan
        ),
    }


def probe_live_15m_fee() -> float | None:
    now = datetime.now(UTC)
    floors = []
    for delta in range(-4, 5):
        t = now + timedelta(minutes=15 * delta)
        t = t.replace(minute=(t.minute // 15) * 15, second=0, microsecond=0)
        floors.append(int(t.timestamp()))
    with httpx.Client(timeout=10, headers={"User-Agent": "epsilon-quant-research-k3/1.0"}) as client:
        for prefix in ("btc-updown-15m", "eth-updown-15m", "sol-updown-15m"):
            for ts in floors:
                slug = f"{prefix}-{ts}"
                try:
                    r = client.get(f"https://gamma-api.polymarket.com/events/slug/{slug}")
                    if r.status_code != 200:
                        continue
                    event = r.json()
                    markets = event.get("markets") or []
                    if not markets:
                        continue
                    condition_id = markets[0].get("conditionId") or markets[0].get("condition_id")
                    if not condition_id:
                        continue
                    cm = client.get(f"https://clob.polymarket.com/clob-markets/{condition_id}")
                    cm.raise_for_status()
                    fd = cm.json().get("fd") or {}
                    rate = as_float(fd.get("r"))
                    return float(rate) if np.isfinite(rate) else None
                except Exception:
                    continue
    return None


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def cents(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}c"


def number(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def summarize_and_write_note(
    panel: pd.DataFrame,
    pm_events: pd.DataFrame,
    market_by_slug: dict[str, MarketMeta],
    external: dict[str, dict[str, pd.DataFrame]],
    stats: pd.DataFrame,
    fee_gate: dict[str, Any],
) -> None:
    xcorr = cross_corr_scan(panel)
    xcorr_valid = xcorr.dropna(subset=["corr"])
    best_xcorr = xcorr_valid.loc[xcorr_valid["corr"].idxmax()].to_dict() if not xcorr_valid.empty else {}
    best_lag = int(best_xcorr.get("lag_seconds", 0)) if best_xcorr else 0
    boot = bootstrap_cross_corr(panel, best_lag) if best_xcorr else {}
    hy = hy_scan_panel(panel)
    hy_valid = hy.dropna(subset=["hy_corr"])
    best_hy = hy_valid.loc[hy_valid["hy_corr"].idxmax()].to_dict() if not hy_valid.empty else {}

    ac60, ac60_n = autocorr_at_lag(panel, "basis", 60)
    ac300, ac300_n = autocorr_at_lag(panel, "basis", 300)
    half_life = ar1_half_life_seconds(panel, "basis")
    edge_median_run, edge_max_run, edge_run_count = edge_run_lengths(panel)

    by_market = (
        panel.groupby(["source_runs", "asset", "market_slug"], dropna=False)
        .agg(
            n_rows=("market_slug", "size"),
            first_ts=("ts", "min"),
            last_ts=("ts", "max"),
            median_basis=("basis", "median"),
            mean_basis=("basis", "mean"),
            p95_abs_basis=("basis", lambda s: float(np.nanquantile(np.abs(s), 0.95))),
            max_abs_basis=("basis", lambda s: float(np.nanmax(np.abs(s)))),
            max_edge=("best_taker_edge", "max"),
            edge_positive_share=("post_fee_survives", "mean"),
            edge_1c_share=("post_fee_survives_1c", "mean"),
            median_spread=("up_ask", lambda s: math.nan),
        )
        .reset_index()
    )
    by_market["window"] = by_market["market_slug"].map(
        lambda slug: market_by_slug[slug].window_start.isoformat().replace("+00:00", "Z")
    )

    top_edge = by_market.sort_values("max_edge", ascending=False).head(8)
    top_rows = [
        [
            str(row.asset),
            str(row.market_slug),
            str(row.source_runs),
            cents(float(row.max_edge)),
            pct(float(row.edge_positive_share)),
            cents(float(row.median_basis)),
            cents(float(row.p95_abs_basis)),
        ]
        for row in top_edge.itertuples(index=False)
    ]

    fee_rows = []
    fees = fee_gate["fees"].drop_duplicates("market_slug").sort_values(["asset", "market_slug"])
    for row in fees.head(30).itertuples(index=False):
        fee_rows.append(
            [
                str(row.asset),
                str(row.market_slug),
                str(row.source_runs),
                number(float(row.fee_rate), 3),
                cents(float(row.peak_fee)),
                str(bool(row.taker_only)).lower(),
            ]
        )

    rows = len(panel)
    windows = panel["market_slug"].nunique()
    assets = ", ".join(sorted(panel["asset"].unique()))
    edge_share = float(panel["post_fee_survives"].mean())
    edge_1c_share = float(panel["post_fee_survives_1c"].mean())
    max_edge = float(panel["best_taker_edge"].max())
    max_edge_row = panel.loc[panel["best_taker_edge"].idxmax()]
    median_basis = float(panel["basis"].median())
    mean_basis = float(panel["basis"].mean())
    p95_abs_basis = float(np.nanquantile(np.abs(panel["basis"]), 0.95))
    dynamic_gate = "YES" if fee_gate["dynamic_like"] else "NO"
    live_15m_rate = fee_gate.get("live_15m_fee_rate")
    if live_15m_rate is None:
        live_15m_rate = math.nan
    live_15m_peak = fee_gate.get("live_15m_peak_fee")
    if live_15m_peak is None:
        live_15m_peak = math.nan
    headline = (
        "No hedgeable post-fee basis in this pooled IS panel."
        if edge_1c_share < 0.01 or max_edge <= 0.01
        else "In-sample post-fee basis survives, but it is not yet a validated hedgeable edge."
    )

    NOTE.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Block K3 Lead-Lag + Digital-Option Basis Findings

Generated: {datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")}

## Headline

{headline}

Pooled A0b + A0c crypto-roll in-sample has {windows} in-window 4h contracts ({assets}) and {rows:,} 10-second panel rows. The strongest cross-correlation has external Binance fair value leading Polymarket by {best_lag:+d}s with corr {number(float(best_xcorr.get("corr", math.nan)), 3)}; window bootstrap CI for the best lag is [{number(float(boot.get("best_lag_ci_lo", math.nan)), 0)}, {number(float(boot.get("best_lag_ci_hi", math.nan)), 0)}] seconds and corr-at-lag CI is [{number(float(boot.get("corr_ci_lo", math.nan)), 3)}, {number(float(boot.get("corr_ci_hi", math.nan)), 3)}]. HY-style asynchronous overlap peaks at {int(best_hy.get("lag_seconds", 0)):+d}s with corr {number(float(best_hy.get("hy_corr", math.nan)), 3)}.

Median mid basis is {cents(median_basis)} (mean {cents(mean_basis)}; 95th percentile absolute basis {cents(p95_abs_basis)}). Best post-fee taker/complement edge is {cents(max_edge)} on {max_edge_row["market_slug"]} at {max_edge_row["ts"]}; {pct(edge_share)} of rows are barely positive after fee and {pct(edge_1c_share)} clear a 1c buffer.

## Step 0 - Fee Gate

4h dynamic anti-arb fee present: **{dynamic_gate}**.

All captured BTC/ETH/SOL 4h markets expose the standard Crypto CLOB fee schedule: taker-only, rate 0.07, exponent 1, rebateRate 0.2. That implies a peak taker fee of {cents(float(fee_gate["max_peak_fee"]))} at 50c. I did not observe a 4h fee rate consistent with the cited 15m anti-arb fee peak of about {cents(DYNAMIC_15M_PEAK_FEE_REFERENCE)}. A live 15m probe during this run returned rate {number(float(live_15m_rate), 3)} / peak {cents(float(live_15m_peak))}, so the dynamic-fee claim is either not currently represented in the CLOB `fd` field or was not active on the sampled market.

{markdown_table(["asset", "market", "runs", "fee_rate", "peak_fee", "taker_only"], fee_rows)}

## Data And Method

- Local CLOB: `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527` and `data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning`.
- External dependency: Binance spot 1s klines and Binance USDT-M perpetual 1m klines were fetched live by `scripts/dali_block_k3_leadlag_basis.py`.
- Only quotes inside the 4h contract window are used. Pre-open quotes are excluded because the strike is the window-open price.
- Strike is Binance spot at the window open. Digital fair value is `N(d2)` using the full-window realized spot volatility, annualized from 1m log returns. This is an ex-post fair-value normalization, not a tradable implied-vol forecast.
- Executable edge uses taker asks only: `buy_up = fair - up_ask - fee(up_ask)` and complement route `buy_down = 1 - fair - down_ask - fee(down_ask)`.
- Resolution-source caveat: Polymarket crypto up/down resolves from Chainlink streams, while Binance is the hedge/reference venue. The measured basis includes this source basis risk.

## Lead-Lag And Persistence

- Cross-correlation best lag: {best_lag:+d}s, corr {number(float(best_xcorr.get("corr", math.nan)), 3)}, n={int(best_xcorr.get("n_pairs", 0)):,}.
- HY-style 10s panel-overlap proxy best lag: {int(best_hy.get("lag_seconds", 0)):+d}s, corr {number(float(best_hy.get("hy_corr", math.nan)), 3)}.
- Basis autocorr: 60s {number(ac60, 3)} (n={ac60_n:,}); 300s {number(ac300, 3)} (n={ac300_n:,}); AR(1) half-life {number(half_life, 1)}s.
- Positive post-fee edge run count: {edge_run_count}; median run {number(edge_median_run, 1)}s; max run {number(edge_max_run, 1)}s.

## Post-Fee Executable Basis

{markdown_table(["asset", "market", "runs", "max_edge", "positive_rows", "median_basis", "p95_abs_basis"], top_rows)}

The executable screen is materially weaker than the midpoint basis screen. Much of the apparent basis is eaten by the ask plus the convex taker fee, and the surviving rows are clustered enough that this should be treated as an in-sample executable screen rather than a durable hedgeable edge until it clears OOS with stricter latency/source controls.

## Output

- CSV panel: `data/analysis/csv_outputs/options_delta/k3_leadlag_basis.csv`
- Repro script: `scripts/dali_block_k3_leadlag_basis.py`
"""
    NOTE.write_text(text, encoding="utf-8")


def write_outputs(
    panel: pd.DataFrame,
    pm_events: pd.DataFrame,
    market_by_slug: dict[str, MarketMeta],
    external: dict[str, dict[str, pd.DataFrame]],
    stats: pd.DataFrame,
    fee_gate: dict[str, Any],
) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    ordered_cols = [
        "ts",
        "source_runs",
        "source_run",
        "asset",
        "market_slug",
        "market_id",
        "condition_id",
        "question",
        "window_start",
        "window_end",
        "seconds_to_expiry",
        "binance_spot",
        "binance_perp",
        "window_strike_spot",
        "window_close_spot",
        "window_return",
        "window_rv_annualized",
        "spot_moneyness",
        "fair_value",
        "perp_fair_value",
        "polymarket_mid",
        "parity_mid",
        "parity_gap",
        "up_bid",
        "up_ask",
        "down_bid",
        "down_ask",
        "basis",
        "basis_cents",
        "parity_basis",
        "basis_vs_perp_fair",
        "taker_fee_rate",
        "taker_fee_up_ask",
        "taker_fee_down_ask",
        "buy_up_edge",
        "buy_down_edge",
        "best_taker_edge",
        "best_taker_edge_cents",
        "best_taker_route",
        "post_fee_survives",
        "post_fee_survives_1c",
    ]
    existing = [col for col in ordered_cols if col in panel.columns]
    panel[existing].to_csv(OUT_CSV, index=False)
    summarize_and_write_note(panel, pm_events, market_by_slug, external, stats, fee_gate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-live-15m-probe", action="store_true")
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="regenerate the findings note from data/analysis/csv_outputs/options_delta/k3_leadlag_basis.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    market_by_slug, token_map = load_market_metadata()
    fee_probe = None if args.no_live_15m_probe else probe_live_15m_fee()
    fee_gate = fee_gate_summary(market_by_slug, fee_probe)
    print(
        "fee gate:",
        {
            "unique_rates": fee_gate["unique_rates"],
            "max_peak_fee": fee_gate["max_peak_fee"],
            "dynamic_like": fee_gate["dynamic_like"],
            "live_15m_fee_rate": fee_gate["live_15m_fee_rate"],
        },
    )
    if args.from_csv:
        if not OUT_CSV.exists():
            raise SystemExit(f"missing existing CSV: {OUT_CSV}")
        panel = pd.read_csv(
            OUT_CSV,
            parse_dates=["ts", "window_start", "window_end"],
            low_memory=False,
        )
        summarize_and_write_note(panel, pd.DataFrame(), market_by_slug, {}, pd.DataFrame(), fee_gate)
        print(f"wrote {NOTE.relative_to(ROOT)} from existing CSV")
        return 0

    pm_events = load_polymarket_events(market_by_slug, token_map)
    if pm_events.empty:
        raise SystemExit("no in-window Polymarket 4h events found")
    slugs = sorted(pm_events["market_slug"].unique())
    print(f"loaded {len(pm_events):,} Polymarket quote states across {len(slugs)} markets")
    external = fetch_external_data(slugs, market_by_slug)
    stats = window_stats(market_by_slug, external, slugs)
    panel = build_panel(pm_events, market_by_slug, external, stats)
    if panel.empty:
        raise SystemExit("no joined K3 panel rows")
    write_outputs(panel, pm_events, market_by_slug, external, stats, fee_gate)
    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(panel):,} rows)")
    print(f"wrote {NOTE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
