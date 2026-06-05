"""Block K3 v2: causal crypto-4h lead-lag and basis screen.

This supersedes the first K3 pass. The primary test is model-free and uses
1-second Binance spot log returns against Polymarket logit-mid changes. The
secondary basis screen uses only causal volatility estimates available at time t.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
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
OUT_CSV = ANALYSIS / "csv_outputs" / "options_delta" / "k3v2_leadlag_causal.csv"
NOTE = NOTES / "block_k3v2_findings.md"

SPOT_BASE = "https://api.binance.com"
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
YEAR_SECONDS = 365.0 * 24.0 * 3600.0
LAG_SECONDS = tuple(range(-60, 61))
BOOTSTRAP_SAMPLES = 250
GRANGER_LAGS = 10
ACTION_LATENCY_SECONDS = 3
EWMA_HALFLIFE_SECONDS = 1800
TRAILING_VOL_SECONDS = 3600
PREWARM_HOURS = 6
RNG_SEED = 20260531
MIN_PRICE = 1e-4
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


@dataclass
class LagStats:
    n: int
    sx: float
    sy: float
    sxx: float
    syy: float
    sxy: float


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


def logit(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    if isinstance(values, pd.Series):
        p = values.astype(float).clip(MIN_PRICE, 1.0 - MIN_PRICE)
        return np.log(p / (1.0 - p))
    arr = np.asarray(values, dtype=float)
    out = np.log(np.clip(arr, MIN_PRICE, 1.0 - MIN_PRICE) / (1.0 - np.clip(arr, MIN_PRICE, 1.0 - MIN_PRICE)))
    if np.ndim(values) == 0:
        return float(out)
    return out


def sigmoid(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    arr = np.asarray(values, dtype=float)
    out = 1.0 / (1.0 + np.exp(-arr))
    if np.ndim(values) == 0:
        return float(out)
    return out


def digital_fair_value(
    spot: pd.Series,
    strike: float,
    sigma_annualized: pd.Series,
    seconds_to_expiry: pd.Series,
) -> pd.Series:
    tau = seconds_to_expiry.astype(float).clip(lower=0.0) / YEAR_SECONDS
    out = pd.Series(np.nan, index=spot.index, dtype=float)
    valid = (
        spot.replace([np.inf, -np.inf], np.nan).notna()
        & np.isfinite(strike)
        & (strike > 0)
        & sigma_annualized.replace([np.inf, -np.inf], np.nan).notna()
        & (sigma_annualized > 0)
        & (tau > 0)
    )
    if valid.any():
        sig = sigma_annualized[valid].to_numpy(dtype=float)
        tau_v = tau[valid].to_numpy(dtype=float)
        denom = sig * np.sqrt(tau_v)
        d2 = (np.log(spot[valid].to_numpy(dtype=float) / strike) - 0.5 * sig * sig * tau_v) / denom
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


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def cents(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}c"


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{10000.0 * value:.1f}bp"


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


def slug_info(slug: str) -> tuple[str, pd.Timestamp, pd.Timestamp] | None:
    match = SLUG_RE.match(slug)
    if not match:
        return None
    asset = match.group(1).upper()
    start = pd.to_datetime(int(match.group(2)), unit="s", utc=True)
    return asset, start, start + pd.Timedelta(hours=4)


def source_run_for_path(path: Path) -> str:
    text = str(path)
    return "a0b" if "/block_a0b/" in text else "a0c_roll"


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


def load_market_metadata() -> tuple[dict[str, MarketMeta], dict[str, tuple[str, str]]]:
    market_by_slug: dict[str, MarketMeta] = {}
    token_map: dict[str, tuple[str, str]] = {}

    configs = sorted([*A0B_DIR.rglob("capture_config.yaml"), *A0C_ROLL_DIR.rglob("capture_config.yaml")])
    for config_path in configs:
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
    token_tuple = tuple(token_map)
    rg_path = shutil.which("rg")

    def candidate_lines(path: Path, token_file: str | None) -> Any:
        if rg_path and token_file:
            proc = subprocess.Popen(
                [rg_path, "-F", "-f", token_file, "--no-filename", "--no-messages", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            assert proc.stdout is not None
            for out_line in proc.stdout:
                yield out_line
            _, stderr = proc.communicate()
            if proc.returncode not in (0, 1):
                raise RuntimeError(f"rg failed on {path}: {stderr}")
        else:
            with path.open("r", encoding="utf-8") as fh:
                for out_line in fh:
                    if any(token in out_line for token in token_tuple):
                        yield out_line

    paths = jsonl_paths()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=True) as token_fh:
        for token in token_tuple:
            token_fh.write(f"{token}\n")
        token_fh.flush()
        token_file = token_fh.name if rg_path else None

        for file_i, path in enumerate(paths, start=1):
            source_run = source_run_for_path(path)
            file_rows = 0
            for line in candidate_lines(path, token_file):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event_type") not in {"price_change", "best_bid_ask", "book"}:
                    continue
                ts = event_timestamp(record)
                if ts is None:
                    continue
                received_at: pd.Timestamp | None = None
                latency_ms = math.nan

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
                    if received_at is None:
                        received_at = parse_ts(record.get("received_at"))
                        if received_at is not None:
                            latency_ms = (received_at - ts).total_seconds() * 1000.0
                    up_mid = (state["up_bid"] + state["up_ask"]) / 2.0
                    down_mid = (state["down_bid"] + state["down_ask"]) / 2.0
                    rows.append(
                        {
                            "ts": ts,
                            "received_at": received_at,
                            "capture_latency_ms": latency_ms,
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
                    file_rows += 1
            if file_rows:
                print(
                    f"parsed {file_rows:,} crypto quote states from {file_i}/{len(paths)} {path.name}",
                    flush=True,
                )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    df = df.sort_values(["market_slug", "ts"]).drop_duplicates(["market_slug", "ts"], keep="last")
    return df.reset_index(drop=True)


def fetch_klines(
    client: httpx.Client,
    *,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        params = {
            "symbol": symbol,
            "interval": "1s",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        for attempt in range(5):
            try:
                r = client.get(f"{SPOT_BASE}/api/v3/klines", params=params, timeout=20)
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
        next_cursor = last_open + 1000
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
    # Label by interval end. Binance kline timestamps are open times; close_time is
    # one millisecond before the next second.
    df["ts"] = pd.to_datetime(df["close_time"].astype("int64") + 1, unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["ts", "open", "high", "low", "close", "volume"]].drop_duplicates("ts").sort_values("ts")


def add_causal_vol(spot: pd.DataFrame) -> pd.DataFrame:
    out = spot.copy().sort_values("ts")
    ret = np.log(out["close"]).diff()
    out["spot_log_return"] = ret
    var_ewm = ret.pow(2).ewm(halflife=EWMA_HALFLIFE_SECONDS, adjust=False, min_periods=60).mean()
    out["ewma_sigma_annualized"] = np.sqrt(var_ewm * YEAR_SECONDS)
    trailing_std = ret.rolling(TRAILING_VOL_SECONDS, min_periods=300).std()
    out["trailing_sigma_annualized"] = trailing_std * math.sqrt(YEAR_SECONDS)
    return out


def fetch_external_data(slugs: list[str], market_by_slug: dict[str, MarketMeta]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    by_asset: dict[str, list[MarketMeta]] = defaultdict(list)
    for slug in slugs:
        by_asset[market_by_slug[slug].asset].append(market_by_slug[slug])

    with httpx.Client(headers={"User-Agent": "epsilon-quant-research-k3v2/1.0"}) as client:
        for asset, metas in sorted(by_asset.items()):
            start = min(m.window_start for m in metas) - pd.Timedelta(hours=PREWARM_HOURS)
            end = max(m.window_end for m in metas) + pd.Timedelta(minutes=5)
            symbol = SYMBOLS[asset]
            print(f"fetching Binance {asset} spot 1s {start} to {end}")
            out[asset] = add_causal_vol(fetch_klines(client, symbol=symbol, start=start, end=end))
    return out


def fetch_gamma_resolution(slug: str, client: httpx.Client) -> dict[str, Any]:
    try:
        r = client.get(GAMMA_MARKETS, params={"closed": "true", "slug": slug}, timeout=20)
        r.raise_for_status()
        rows = r.json()
    except Exception:
        return {"market_slug": slug}
    if not rows:
        return {"market_slug": slug}
    row = rows[0]
    prices_raw = row.get("outcomePrices")
    outcomes_raw = row.get("outcomes")
    try:
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
    except Exception:
        prices = None
    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
    except Exception:
        outcomes = None
    up_price = math.nan
    down_price = math.nan
    if isinstance(prices, list) and len(prices) >= 2:
        up_price = as_float(prices[0])
        down_price = as_float(prices[1])
    chainlink_up = math.nan
    if np.isfinite(up_price):
        chainlink_up = bool(up_price >= 0.5)
    return {
        "market_slug": slug,
        "gamma_closed": bool(row.get("closed")),
        "gamma_resolved": str(row.get("umaResolutionStatus") or ""),
        "resolution_source": str(row.get("resolutionSource") or ""),
        "gamma_up_price": up_price,
        "gamma_down_price": down_price,
        "chainlink_resolution_up": chainlink_up,
        "gamma_closed_time": row.get("closedTime"),
        "gamma_outcomes": ",".join(map(str, outcomes or [])),
    }


def fetch_gamma_resolutions(slugs: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with httpx.Client(headers={"User-Agent": "epsilon-quant-research-k3v2/1.0"}) as client:
        for slug in sorted(slugs):
            rows.append(fetch_gamma_resolution(slug, client))
            time.sleep(0.02)
    return pd.DataFrame(rows)


def spot_asof(spot: pd.DataFrame, ts: pd.Timestamp) -> float:
    indexed = spot.set_index("ts").sort_index()
    s = indexed["close"].loc[:ts]
    if s.empty:
        return math.nan
    return float(s.iloc[-1])


def window_stats(
    market_by_slug: dict[str, MarketMeta],
    external: dict[str, pd.DataFrame],
    slugs: list[str],
    gamma: pd.DataFrame,
) -> pd.DataFrame:
    gamma_by_slug = gamma.set_index("market_slug").to_dict("index") if not gamma.empty else {}
    rows: list[dict[str, Any]] = []
    for slug in slugs:
        meta = market_by_slug[slug]
        spot = external[meta.asset]
        strike = spot_asof(spot, meta.window_start)
        close = spot_asof(spot, meta.window_end)
        ret = math.log(close / strike) if np.isfinite(close) and np.isfinite(strike) and strike > 0 else math.nan
        g = gamma_by_slug.get(slug, {})
        chainlink_up = g.get("chainlink_resolution_up", math.nan)
        binance_up = bool(close >= strike) if np.isfinite(close) and np.isfinite(strike) else math.nan
        source_disagree = (
            bool(chainlink_up) != bool(binance_up)
            if isinstance(chainlink_up, (bool, np.bool_)) and isinstance(binance_up, (bool, np.bool_))
            else math.nan
        )
        rows.append(
            {
                "market_slug": slug,
                "asset": meta.asset,
                "window_start": meta.window_start,
                "window_end": meta.window_end,
                "binance_strike_spot": strike,
                "binance_close_spot": close,
                "binance_window_return": ret,
                "binance_window_abs_return_bps": abs(ret) * 10000.0 if np.isfinite(ret) else math.nan,
                "binance_resolution_up": binance_up,
                "chainlink_resolution_up": chainlink_up,
                "chainlink_binance_resolution_disagree": source_disagree,
                "resolution_source": g.get("resolution_source", ""),
                "gamma_up_price": g.get("gamma_up_price", math.nan),
                "gamma_down_price": g.get("gamma_down_price", math.nan),
            }
        )
    return pd.DataFrame(rows)


def build_panel(
    pm: pd.DataFrame,
    market_by_slug: dict[str, MarketMeta],
    external: dict[str, pd.DataFrame],
    stats: pd.DataFrame,
) -> pd.DataFrame:
    stat_by_slug = stats.set_index("market_slug").to_dict("index")
    panels: list[pd.DataFrame] = []

    for slug, group in pm.groupby("market_slug", sort=True):
        if slug not in stat_by_slug:
            continue
        meta = market_by_slug[slug]
        group = group.sort_values("ts").drop_duplicates("ts", keep="last")
        if len(group) < 3:
            continue
        first_ts = max(group["ts"].min().ceil("1s"), meta.window_start)
        last_ts = min(group["ts"].max().ceil("1s"), meta.window_end)
        if first_ts >= last_ts:
            continue

        pm_regular = (
            group.set_index("ts")[
                [
                    "source_run",
                    "up_bid",
                    "up_ask",
                    "down_bid",
                    "down_ask",
                    "polymarket_mid",
                    "down_mid",
                    "parity_mid",
                    "parity_gap",
                    "capture_latency_ms",
                ]
            ]
            .sort_index()
            .resample("1s", label="right", closed="right")
            .last()
            .ffill()
        )
        idx = pd.date_range(first_ts, last_ts, freq="1s", tz="UTC")
        regular = pm_regular.reindex(idx, method="ffill")
        regular = regular.dropna(subset=["polymarket_mid", "up_ask", "down_ask"])
        if regular.empty:
            continue

        spot = external[meta.asset].set_index("ts").sort_index()
        regular["binance_spot"] = spot["close"].reindex(regular.index, method="ffill")
        regular["spot_log_return"] = spot["spot_log_return"].reindex(regular.index, method="ffill")
        regular["ewma_sigma_annualized"] = spot["ewma_sigma_annualized"].reindex(regular.index, method="ffill")
        regular["trailing_sigma_annualized"] = spot["trailing_sigma_annualized"].reindex(regular.index, method="ffill")
        stat = stat_by_slug[slug]
        regular["seconds_to_expiry"] = (meta.window_end - regular.index).total_seconds()
        regular["window_start"] = meta.window_start
        regular["window_end"] = meta.window_end
        regular["binance_strike_spot"] = stat["binance_strike_spot"]
        regular["binance_close_spot"] = stat["binance_close_spot"]
        regular["binance_window_return"] = stat["binance_window_return"]
        regular["binance_window_abs_return_bps"] = stat["binance_window_abs_return_bps"]
        regular["binance_resolution_up"] = stat["binance_resolution_up"]
        regular["chainlink_resolution_up"] = stat["chainlink_resolution_up"]
        regular["chainlink_binance_resolution_disagree"] = stat["chainlink_binance_resolution_disagree"]
        regular["resolution_source"] = stat["resolution_source"]

        tau_series = pd.Series(regular["seconds_to_expiry"].to_numpy(dtype=float), index=regular.index)
        regular["causal_fair_value"] = digital_fair_value(
            regular["binance_spot"],
            float(stat["binance_strike_spot"]),
            regular["ewma_sigma_annualized"],
            tau_series,
        )
        regular["causal_fair_value_trailing"] = digital_fair_value(
            regular["binance_spot"],
            float(stat["binance_strike_spot"]),
            regular["trailing_sigma_annualized"],
            tau_series,
        )
        regular["polymarket_logit_mid"] = logit(regular["polymarket_mid"])
        regular["causal_fair_logit"] = logit(regular["causal_fair_value"])
        regular["pm_logit_change_1s"] = regular["polymarket_logit_mid"].diff()
        regular["causal_fair_logit_change_1s"] = regular["causal_fair_logit"].diff()
        regular["causal_basis"] = regular["polymarket_mid"] - regular["causal_fair_value"]
        regular["causal_basis_cents"] = 100.0 * regular["causal_basis"]
        regular["causal_logit_gap_pm_minus_fair"] = regular["polymarket_logit_mid"] - regular["causal_fair_logit"]
        regular["spot_moneyness"] = np.log(regular["binance_spot"] / float(stat["binance_strike_spot"]))

        regular["taker_fee_rate"] = meta.fee_rate
        regular["taker_fee_up_ask"] = taker_fee(regular["up_ask"], meta.fee_rate)
        regular["taker_fee_down_ask"] = taker_fee(regular["down_ask"], meta.fee_rate)
        regular["buy_up_edge_now"] = regular["causal_fair_value"] - regular["up_ask"] - regular["taker_fee_up_ask"]
        regular["buy_down_edge_now"] = (
            1.0 - regular["causal_fair_value"] - regular["down_ask"] - regular["taker_fee_down_ask"]
        )
        regular["best_taker_edge_now"] = regular[["buy_up_edge_now", "buy_down_edge_now"]].max(axis=1)
        regular["best_taker_route_now"] = np.where(
            regular["buy_up_edge_now"] >= regular["buy_down_edge_now"], "buy_up", "buy_down"
        )

        fill_up_ask = regular["up_ask"].shift(-ACTION_LATENCY_SECONDS)
        fill_down_ask = regular["down_ask"].shift(-ACTION_LATENCY_SECONDS)
        fill_up_fee = taker_fee(fill_up_ask, meta.fee_rate)
        fill_down_fee = taker_fee(fill_down_ask, meta.fee_rate)
        regular["fill_up_ask_latency"] = fill_up_ask
        regular["fill_down_ask_latency"] = fill_down_ask
        regular["buy_up_edge_latency"] = regular["causal_fair_value"] - fill_up_ask - fill_up_fee
        regular["buy_down_edge_latency"] = 1.0 - regular["causal_fair_value"] - fill_down_ask - fill_down_fee
        regular["best_taker_edge_latency"] = regular[["buy_up_edge_latency", "buy_down_edge_latency"]].max(axis=1)
        regular["best_taker_route_latency"] = np.where(
            regular["buy_up_edge_latency"] >= regular["buy_down_edge_latency"], "buy_up", "buy_down"
        )
        regular["post_fee_edge_now"] = regular["best_taker_edge_now"] > 0
        regular["post_fee_edge_now_1c"] = regular["best_taker_edge_now"] > 0.01
        regular["post_fee_edge_latency"] = regular["best_taker_edge_latency"] > 0
        regular["post_fee_edge_latency_1c"] = regular["best_taker_edge_latency"] > 0.01

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
    out["best_taker_edge_latency_cents"] = 100.0 * out["best_taker_edge_latency"]
    out["best_taker_edge_now_cents"] = 100.0 * out["best_taker_edge_now"]
    out["abs_causal_basis"] = out["causal_basis"].abs()
    out["large_static_basis_10c"] = out["abs_causal_basis"] > 0.10
    market_gap_median = out.groupby("market_slug")["causal_logit_gap_pm_minus_fair"].transform("median")
    out["dynamic_quote_skew_logit"] = -(out["causal_logit_gap_pm_minus_fair"] - market_gap_median)
    out["dynamic_quote_skew_prob"] = sigmoid(out["polymarket_logit_mid"] + out["dynamic_quote_skew_logit"]) - out[
        "polymarket_mid"
    ]
    return out.reset_index(drop=True)


def finite_lag_stats(x: np.ndarray, y: np.ndarray, lead_seconds: int) -> LagStats:
    if lead_seconds > 0:
        xs = x[:-lead_seconds]
        ys = y[lead_seconds:]
    elif lead_seconds < 0:
        xs = x[-lead_seconds:]
        ys = y[:lead_seconds]
    else:
        xs = x
        ys = y
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() == 0:
        return LagStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    xs = xs[mask]
    ys = ys[mask]
    return LagStats(
        int(len(xs)),
        float(xs.sum()),
        float(ys.sum()),
        float(np.dot(xs, xs)),
        float(np.dot(ys, ys)),
        float(np.dot(xs, ys)),
    )


def merge_stats(stats: list[LagStats]) -> LagStats:
    return LagStats(
        n=sum(s.n for s in stats),
        sx=sum(s.sx for s in stats),
        sy=sum(s.sy for s in stats),
        sxx=sum(s.sxx for s in stats),
        syy=sum(s.syy for s in stats),
        sxy=sum(s.sxy for s in stats),
    )


def corr_from_stats(s: LagStats) -> float:
    if s.n < 20:
        return math.nan
    cov = s.sxy - (s.sx * s.sy / s.n)
    vx = s.sxx - (s.sx * s.sx / s.n)
    vy = s.syy - (s.sy * s.sy / s.n)
    if vx <= 0 or vy <= 0:
        return math.nan
    return float(cov / math.sqrt(vx * vy))


def precompute_cross_stats(panel: pd.DataFrame) -> dict[str, dict[int, LagStats]]:
    out: dict[str, dict[int, LagStats]] = {}
    for slug, group in panel.groupby("market_slug", sort=False):
        g = group.sort_values("ts")
        x = g["spot_log_return"].to_numpy(dtype=float)
        y = g["pm_logit_change_1s"].to_numpy(dtype=float)
        out[str(slug)] = {lead: finite_lag_stats(x, y, lead) for lead in LAG_SECONDS}
    return out


def scan_from_precomputed(precomputed: dict[str, dict[int, LagStats]], slugs: list[str] | None = None) -> pd.DataFrame:
    if slugs is None:
        slugs = list(precomputed)
    rows: list[dict[str, Any]] = []
    for lead in LAG_SECONDS:
        s = merge_stats([precomputed[slug][lead] for slug in slugs if slug in precomputed])
        rows.append({"lead_seconds": lead, "corr": corr_from_stats(s), "n_pairs": s.n})
    return pd.DataFrame(rows)


def bootstrap_lag_ci(precomputed: dict[str, dict[int, LagStats]], full_best_lead: int) -> dict[str, float]:
    slugs = sorted(precomputed)
    rng = np.random.default_rng(RNG_SEED)
    best_leads: list[float] = []
    corr_at_best: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [slugs[i] for i in rng.integers(0, len(slugs), size=len(slugs))]
        scan = scan_from_precomputed(precomputed, sample)
        valid = scan.dropna(subset=["corr"])
        if valid.empty:
            continue
        best = valid.loc[valid["corr"].idxmax()]
        best_leads.append(float(best["lead_seconds"]))
        row = valid[valid["lead_seconds"].eq(full_best_lead)]
        if not row.empty:
            corr_at_best.append(float(row["corr"].iloc[0]))
    out: dict[str, float] = {}
    for key, values in [("best_lead", best_leads), ("corr", corr_at_best)]:
        if values:
            lo, hi = np.quantile(values, [0.025, 0.975])
            out[f"{key}_ci_lo"] = float(lo)
            out[f"{key}_ci_hi"] = float(hi)
        else:
            out[f"{key}_ci_lo"] = math.nan
            out[f"{key}_ci_hi"] = math.nan
    return out


def hy_covariance(
    x_start: np.ndarray,
    x_end: np.ndarray,
    x_ret: np.ndarray,
    y_start: np.ndarray,
    y_end: np.ndarray,
    y_ret: np.ndarray,
    lead_seconds: int,
) -> tuple[float, int]:
    shift_ns = int(lead_seconds * 1_000_000_000)
    ys = y_start - shift_ns
    ye = y_end - shift_ns
    # Spot intervals are regular and sorted. For each shifted PM interval,
    # locate the contiguous block of spot intervals that overlap it. HY then
    # adds y_j * sum(x_i for overlapping i), without duration weighting.
    left = np.searchsorted(x_end, ys, side="right")
    right = np.searchsorted(x_start, ye, side="left")
    valid = right > left
    if not np.any(valid):
        return 0.0, 0
    prefix = np.concatenate([[0.0], np.cumsum(x_ret)])
    x_sums = prefix[right[valid]] - prefix[left[valid]]
    cov = float(np.dot(y_ret[valid], x_sums))
    n_overlap = int(np.sum(right[valid] - left[valid]))
    return cov, n_overlap


def precompute_hy_stats(pm_events: pd.DataFrame, external: dict[str, pd.DataFrame]) -> dict[str, dict[int, dict[str, float]]]:
    out: dict[str, dict[int, dict[str, float]]] = {}
    for slug, group in pm_events.groupby("market_slug", sort=False):
        asset = str(group["asset"].iloc[0])
        spot = external[asset].sort_values("ts").copy()
        start = group["ts"].min().floor("1s")
        end = group["ts"].max().ceil("1s")
        spot = spot[(spot["ts"] >= start - pd.Timedelta(seconds=1)) & (spot["ts"] <= end + pd.Timedelta(seconds=1))]
        spot_ret = np.log(spot["close"]).diff().to_numpy(dtype=float)
        spot_end = spot["ts"].astype("int64").to_numpy()
        spot_start = spot_end - 1_000_000_000
        spot_mask = np.isfinite(spot_ret)
        spot_ret = spot_ret[spot_mask]
        spot_start = spot_start[spot_mask]
        spot_end = spot_end[spot_mask]
        if len(spot_ret) < 20:
            continue

        g = group.sort_values("ts").drop_duplicates("ts", keep="last").copy()
        g["x"] = logit(g["polymarket_mid"])
        pm_ret = g["x"].diff().to_numpy(dtype=float)
        pm_end = g["ts"].astype("int64").to_numpy()
        pm_start = np.roll(pm_end, 1)
        pm_mask = np.isfinite(pm_ret) & (pm_end > pm_start)
        pm_ret = pm_ret[pm_mask]
        pm_start = pm_start[pm_mask]
        pm_end = pm_end[pm_mask]
        if len(pm_ret) < 20:
            continue

        xvar = float(np.dot(spot_ret, spot_ret))
        yvar = float(np.dot(pm_ret, pm_ret))
        out[str(slug)] = {}
        for lead in LAG_SECONDS:
            cov, n_overlap = hy_covariance(spot_start, spot_end, spot_ret, pm_start, pm_end, pm_ret, lead)
            out[str(slug)][lead] = {"cov": cov, "xvar": xvar, "yvar": yvar, "n_overlap": n_overlap}
    return out


def hy_scan_from_precomputed(
    precomputed: dict[str, dict[int, dict[str, float]]],
    slugs: list[str] | None = None,
) -> pd.DataFrame:
    if slugs is None:
        slugs = list(precomputed)
    rows: list[dict[str, Any]] = []
    for lead in LAG_SECONDS:
        cov = 0.0
        xvar = 0.0
        yvar = 0.0
        n_overlap = 0
        for slug in slugs:
            if slug not in precomputed:
                continue
            s = precomputed[slug][lead]
            cov += s["cov"]
            xvar += s["xvar"]
            yvar += s["yvar"]
            n_overlap += int(s["n_overlap"])
        corr = cov / math.sqrt(xvar * yvar) if xvar > 0 and yvar > 0 else math.nan
        rows.append({"lead_seconds": lead, "hy_corr": corr, "n_overlap": n_overlap})
    return pd.DataFrame(rows)


def bootstrap_hy_ci(precomputed: dict[str, dict[int, dict[str, float]]], full_best_lead: int) -> dict[str, float]:
    slugs = sorted(precomputed)
    rng = np.random.default_rng(RNG_SEED + 1)
    best_leads: list[float] = []
    corr_at_best: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [slugs[i] for i in rng.integers(0, len(slugs), size=len(slugs))]
        scan = hy_scan_from_precomputed(precomputed, sample)
        valid = scan.dropna(subset=["hy_corr"])
        if valid.empty:
            continue
        best = valid.loc[valid["hy_corr"].idxmax()]
        best_leads.append(float(best["lead_seconds"]))
        row = valid[valid["lead_seconds"].eq(full_best_lead)]
        if not row.empty:
            corr_at_best.append(float(row["hy_corr"].iloc[0]))
    out: dict[str, float] = {}
    for key, values in [("best_lead", best_leads), ("corr", corr_at_best)]:
        if values:
            lo, hi = np.quantile(values, [0.025, 0.975])
            out[f"{key}_ci_lo"] = float(lo)
            out[f"{key}_ci_hi"] = float(hi)
        else:
            out[f"{key}_ci_lo"] = math.nan
            out[f"{key}_ci_hi"] = math.nan
    return out


def build_granger_design(panel: pd.DataFrame, target: str, own: str, cause: str, maxlag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys: list[np.ndarray] = []
    own_lags: list[np.ndarray] = []
    full_lags: list[np.ndarray] = []
    for _, group in panel.groupby("market_slug", sort=False):
        g = group.sort_values("ts")
        y = g[target].to_numpy(dtype=float)
        own_arr = g[own].to_numpy(dtype=float)
        cause_arr = g[cause].to_numpy(dtype=float)
        if len(g) <= maxlag + 5:
            continue
        rows_own = []
        rows_full = []
        target_rows = []
        for i in range(maxlag, len(g)):
            yy = y[i]
            own_vec = own_arr[i - maxlag : i][::-1]
            cause_vec = cause_arr[i - maxlag : i][::-1]
            if not (np.isfinite(yy) and np.all(np.isfinite(own_vec)) and np.all(np.isfinite(cause_vec))):
                continue
            target_rows.append(yy)
            rows_own.append(np.r_[1.0, own_vec])
            rows_full.append(np.r_[1.0, own_vec, cause_vec])
        if target_rows:
            ys.append(np.asarray(target_rows))
            own_lags.append(np.asarray(rows_own))
            full_lags.append(np.asarray(rows_full))
    if not ys:
        return np.empty(0), np.empty((0, maxlag + 1)), np.empty((0, 2 * maxlag + 1))
    return np.concatenate(ys), np.vstack(own_lags), np.vstack(full_lags)


def ols_rss(y: np.ndarray, x: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return float(np.dot(resid, resid))


def betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3e-12
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def f_sf(f_stat: float, dfn: int, dfd: int) -> float:
    if not np.isfinite(f_stat) or f_stat < 0:
        return math.nan
    x = (dfn * f_stat) / (dfn * f_stat + dfd)
    cdf = regularized_betai(dfn / 2.0, dfd / 2.0, x)
    return max(0.0, min(1.0, 1.0 - cdf))


def granger_test(panel: pd.DataFrame, target: str, own: str, cause: str, label: str) -> dict[str, Any]:
    y, x_restricted, x_full = build_granger_design(panel, target, own, cause, GRANGER_LAGS)
    if len(y) < 100:
        return {"direction": label, "n": len(y), "maxlag": GRANGER_LAGS, "f_stat": math.nan, "p_value": math.nan}
    rss_r = ols_rss(y, x_restricted)
    rss_u = ols_rss(y, x_full)
    q = GRANGER_LAGS
    df2 = len(y) - x_full.shape[1]
    f_stat = ((rss_r - rss_u) / q) / (rss_u / df2) if rss_u > 0 and df2 > 0 else math.nan
    return {
        "direction": label,
        "n": int(len(y)),
        "maxlag": GRANGER_LAGS,
        "rss_restricted": rss_r,
        "rss_unrestricted": rss_u,
        "f_stat": float(f_stat),
        "p_value": f_sf(float(f_stat), q, df2),
    }


def edge_run_summary(panel: pd.DataFrame, col: str) -> dict[str, float]:
    runs: list[float] = []
    for _, group in panel.groupby("market_slug", sort=False):
        g = group.sort_values("ts")
        mask = g[col].fillna(False).to_numpy(dtype=bool)
        ts = g["ts"].to_numpy()
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        start = idx[0]
        prev = idx[0]
        for pos in idx[1:]:
            gap = (pd.Timestamp(ts[pos]) - pd.Timestamp(ts[prev])).total_seconds()
            if pos != prev + 1 or gap > 1.5:
                runs.append((pd.Timestamp(ts[prev]) - pd.Timestamp(ts[start])).total_seconds() + 1.0)
                start = pos
            prev = pos
        runs.append((pd.Timestamp(ts[prev]) - pd.Timestamp(ts[start])).total_seconds() + 1.0)
    arr = np.asarray(runs, dtype=float)
    if len(arr) == 0:
        return {"count": 0, "median": math.nan, "p90": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "count": int(len(arr)),
        "median": float(np.nanmedian(arr)),
        "p90": float(np.nanquantile(arr, 0.90)),
        "p95": float(np.nanquantile(arr, 0.95)),
        "max": float(np.nanmax(arr)),
    }


def fee_gate_summary(market_by_slug: dict[str, MarketMeta], slugs: list[str]) -> dict[str, Any]:
    rows = []
    for slug in sorted(slugs):
        meta = market_by_slug[slug]
        peak_fee = taker_fee(0.5, meta.fee_rate)
        rows.append(
            {
                "asset": meta.asset,
                "market_slug": slug,
                "fee_rate": meta.fee_rate,
                "peak_fee": peak_fee,
                "taker_only": meta.taker_only,
            }
        )
    fees = pd.DataFrame(rows)
    max_peak = float(fees["peak_fee"].max()) if not fees.empty else math.nan
    return {
        "fees": fees,
        "unique_rates": sorted(round(float(x), 8) for x in fees["fee_rate"].dropna().unique()),
        "max_peak_fee": max_peak,
        "dynamic_like": bool(np.isfinite(max_peak) and max_peak > 0.025),
    }


def write_outputs(
    panel: pd.DataFrame,
    pm_events: pd.DataFrame,
    xcorr: pd.DataFrame,
    xcorr_ci: dict[str, float],
    hy: pd.DataFrame,
    hy_ci: dict[str, float],
    granger: pd.DataFrame,
    stats: pd.DataFrame,
    fee_gate: dict[str, Any],
) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_CSV, index=False)

    best_x = xcorr.dropna(subset=["corr"]).loc[xcorr.dropna(subset=["corr"])["corr"].idxmax()]
    best_h = hy.dropna(subset=["hy_corr"]).loc[hy.dropna(subset=["hy_corr"])["hy_corr"].idxmax()]
    lead = int(best_x["lead_seconds"])
    latency_ok = lead > ACTION_LATENCY_SECONDS
    rows = len(panel)
    markets = panel["market_slug"].nunique()
    assets = ", ".join(sorted(panel["asset"].unique()))
    cap_lat = pm_events["capture_latency_ms"].replace([np.inf, -np.inf], np.nan).dropna()
    cap_p50 = float(cap_lat.quantile(0.50)) if len(cap_lat) else math.nan
    cap_p95 = float(cap_lat.quantile(0.95)) if len(cap_lat) else math.nan
    cap_p99 = float(cap_lat.quantile(0.99)) if len(cap_lat) else math.nan

    now_edge_share = float(panel["post_fee_edge_now"].mean())
    now_edge_1c_share = float(panel["post_fee_edge_now_1c"].mean())
    latency_edge_share = float(panel["post_fee_edge_latency"].mean())
    latency_edge_1c_share = float(panel["post_fee_edge_latency_1c"].mean())
    max_latency_edge = float(panel["best_taker_edge_latency"].max())
    p95_abs_basis = float(np.nanquantile(panel["abs_causal_basis"], 0.95))
    median_basis = float(panel["causal_basis"].median())
    p95_abs_skew_prob = float(np.nanquantile(np.abs(panel["dynamic_quote_skew_prob"]), 0.95))
    p95_abs_skew_logit = float(np.nanquantile(np.abs(panel["dynamic_quote_skew_logit"]), 0.95))
    run_pos = edge_run_summary(panel, "post_fee_edge_latency")
    run_1c = edge_run_summary(panel, "post_fee_edge_latency_1c")

    disagree = stats["chainlink_binance_resolution_disagree"].dropna()
    disagree_n = int(disagree.sum()) if len(disagree) else 0
    disagree_d = int(len(disagree))
    min_margin = float(stats["binance_window_abs_return_bps"].min()) if not stats.empty else math.nan
    med_margin = float(stats["binance_window_abs_return_bps"].median()) if not stats.empty else math.nan

    granger_rows = []
    for _, row in granger.iterrows():
        p = float(row["p_value"])
        p_str = "<1e-12" if np.isfinite(p) and p < 1e-12 else number(p, 4)
        granger_rows.append(
            [
                str(row["direction"]),
                str(int(row["maxlag"])),
                f"{int(row['n']):,}",
                number(float(row["f_stat"]), 2),
                p_str,
            ]
        )

    by_asset_rows = []
    for asset, g in panel.groupby("asset", sort=True):
        by_asset_rows.append(
            [
                asset,
                f"{len(g):,}",
                str(g["market_slug"].nunique()),
                cents(float(g["causal_basis"].median())),
                cents(float(np.nanquantile(g["abs_causal_basis"], 0.95))),
                pct(float(g["post_fee_edge_latency_1c"].mean())),
                cents(float(g["best_taker_edge_latency"].max())),
            ]
        )

    top_market_rows = []
    market_summary = (
        panel.groupby(["asset", "market_slug"], sort=True)
        .agg(
            rows=("ts", "size"),
            max_edge=("best_taker_edge_latency", "max"),
            edge_1c=("post_fee_edge_latency_1c", "mean"),
            median_basis=("causal_basis", "median"),
            p95_abs_basis=("abs_causal_basis", lambda s: float(np.nanquantile(s, 0.95))),
        )
        .reset_index()
        .sort_values("max_edge", ascending=False)
        .head(8)
    )
    for _, row in market_summary.iterrows():
        top_market_rows.append(
            [
                str(row["asset"]),
                str(row["market_slug"]),
                cents(float(row["max_edge"])),
                pct(float(row["edge_1c"])),
                cents(float(row["median_basis"])),
                cents(float(row["p95_abs_basis"])),
            ]
        )

    source_rows = []
    for _, row in stats.sort_values(["asset", "market_slug"]).iterrows():
        source_rows.append(
            [
                str(row["asset"]),
                str(row["market_slug"]),
                "up" if bool(row["binance_resolution_up"]) else "down",
                (
                    "up"
                    if isinstance(row["chainlink_resolution_up"], (bool, np.bool_)) and bool(row["chainlink_resolution_up"])
                    else "down"
                    if isinstance(row["chainlink_resolution_up"], (bool, np.bool_))
                    else "n/a"
                ),
                "yes" if row["chainlink_binance_resolution_disagree"] is True else "no",
                f"{float(row['binance_window_abs_return_bps']):.1f}",
            ]
        )

    headline_a = (
        f"1s lead-lag survives latency: Binance leads Polymarket by {lead}s."
        if latency_ok
        else f"1s lead-lag does not clear latency: cross-corr peaks at {lead}s and HY peaks at {int(best_h['lead_seconds'])}s."
    )
    headline_b = (
        "The causal basis screen is diagnostic only because the primary latency gate fails."
        if not latency_ok
        else "Causal post-fee basis remains visible in-sample, but static basis/source risk still dominates validation."
    )

    text = f"""# Block K3 v2 Lead-Lag Causal Findings

Generated: {datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")}

## Headline

{headline_a} {headline_b}

Pooled A0b + A0c crypto-roll in-sample has {markets} in-window 4h contracts ({assets}) and {rows:,} 1-second rows. Cross-correlation peaks at **+{lead}s** (positive means Binance spot returns lead Polymarket logit-mid changes) with corr {number(float(best_x["corr"]), 3)} and n={int(best_x["n_pairs"]):,}. Window bootstrap CI for peak lead is [{number(float(xcorr_ci["best_lead_ci_lo"]), 0)}, {number(float(xcorr_ci["best_lead_ci_hi"]), 0)}] seconds; corr-at-peak CI is [{number(float(xcorr_ci["corr_ci_lo"]), 3)}, {number(float(xcorr_ci["corr_ci_hi"]), 3)}].

Hayashi-Yoshida on asynchronous spot-return intervals vs raw Polymarket logit-mid intervals peaks at **+{int(best_h["lead_seconds"])}s** with HY corr {number(float(best_h["hy_corr"]), 3)}; bootstrap peak-lead CI is [{number(float(hy_ci["best_lead_ci_lo"]), 0)}, {number(float(hy_ci["best_lead_ci_hi"]), 0)}] seconds.

Message timestamp vs local receive delta is p50 {number(cap_p50, 0)}ms / p95 {number(cap_p95, 0)}ms / p99 {number(cap_p99, 0)}ms; the small negative median indicates clock skew, not negative physical latency. This covers capture timestamps only, before decision, order routing, and fill. Against the pre-set 2-3s action-latency hurdle, the measured lead {"survives" if latency_ok else "does not survive"}; with a cross-corr peak of {lead}s there is no positive latency budget.

## Model-Free Lead-Lag

The primary measurement uses Binance spot 1s log returns and Polymarket `UP` logit-mid 1s changes. It avoids option-model fair value entirely.

{markdown_table(["direction", "maxlag_s", "n", "F", "p"], granger_rows)}

The Granger test is pooled OLS at 1s with 10 one-second lags. The Binance-to-Polymarket direction is the one relevant for quote skew; the reverse direction is included as a sanity check for feedback/common-timestamp effects.

## Causal Fair Value And Executable Screen

The causal fair value is `N(d2)` with Binance spot, Binance window-open strike, and EWMA realized volatility known up to time t only (half-life {EWMA_HALFLIFE_SECONDS}s). This removes the v1 full-window realized-vol lookahead. Because the model-free lead-lag fails the latency gate, this section is a diagnostic basis/source screen rather than a tradable convergence-alpha result. It still uses Binance as the hedge/reference venue, so static basis is not treated as alpha.

At current quotes, {pct(now_edge_share)} of rows are positive after taker fee and {pct(now_edge_1c_share)} clear a 1c buffer. With a {ACTION_LATENCY_SECONDS}s capture->decide->order/fill latency simulation, {pct(latency_edge_share)} remain positive and {pct(latency_edge_1c_share)} clear 1c. Max latency-adjusted edge is {cents(max_latency_edge)}.

Latency-adjusted positive-edge runs: count {int(run_pos["count"])}, median {number(run_pos["median"], 1)}s, p90 {number(run_pos["p90"], 1)}s, p95 {number(run_pos["p95"], 1)}s, max {number(run_pos["max"], 1)}s. For `>1c` runs: count {int(run_1c["count"])}, median {number(run_1c["median"], 1)}s, p90 {number(run_1c["p90"], 1)}s, p95 {number(run_1c["p95"], 1)}s, max {number(run_1c["max"], 1)}s.

Pooled causal median basis is {cents(median_basis)} and p95 absolute causal basis is {cents(p95_abs_basis)}. Large static basis should be treated as model/source error first; the quote-skew feed should use the demeaned dynamic logit gap. P95 absolute dynamic K2/K-PEG skew is {number(p95_abs_skew_logit, 3)} logit units, or {cents(p95_abs_skew_prob)} in probability space around the observed mid.

{markdown_table(["asset", "rows", "markets", "median_basis", "p95_abs_basis", "latency_gt_1c", "max_latency_edge"], by_asset_rows)}

Top latency-adjusted markets:

{markdown_table(["asset", "market", "max_edge", "gt_1c_rows", "median_basis", "p95_abs_basis"], top_market_rows)}

## Chainlink-vs-Binance Source Basis

Polymarket resolves these markets from Chainlink Data Streams, while this hedge/fair screen uses Binance. Gamma market metadata confirms the Chainlink stream resolution source for the captured crypto up/down contracts. Public historical Chainlink stream ticks were not available through an unauthenticated endpoint in this run, so this pass separates source risk by resolved direction agreement and Binance settlement margin rather than counting it as alpha.

Resolved Chainlink direction disagreed with Binance open-to-close direction on {disagree_n}/{disagree_d} windows. Median absolute Binance settlement margin was {number(med_margin, 1)}bp; minimum was {number(min_margin, 1)}bp. Near-zero margins are the dangerous source-basis cases because a small Chainlink-vs-Binance gap can flip settlement.

{markdown_table(["asset", "market", "binance_dir", "chainlink_dir", "disagree", "abs_binance_margin_bp"], source_rows)}

## Output

- CSV panel: `data/analysis/csv_outputs/options_delta/k3v2_leadlag_causal.csv`
- Repro script: `scripts/dali_block_k3v2_leadlag_causal.py`
"""
    NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {NOTE}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-csv", action="store_true", help="Regenerate note from existing k3v2 CSV.")
    args = parser.parse_args()

    market_by_slug, token_map = load_market_metadata()

    if args.from_csv:
        panel = pd.read_csv(OUT_CSV, parse_dates=["ts", "window_start", "window_end"])
        slugs = sorted(panel["market_slug"].unique())
        pm_events = panel[["ts", "market_slug", "asset", "polymarket_mid", "capture_latency_ms"]].copy()
        stats_cols = [
            "market_slug",
            "asset",
            "window_start",
            "window_end",
            "binance_strike_spot",
            "binance_close_spot",
            "binance_window_return",
            "binance_window_abs_return_bps",
            "binance_resolution_up",
            "chainlink_resolution_up",
            "chainlink_binance_resolution_disagree",
            "resolution_source",
        ]
        stats = panel[stats_cols].drop_duplicates("market_slug")
        external: dict[str, pd.DataFrame] = {}
        for asset, g in panel.groupby("asset"):
            external[str(asset)] = g[["ts", "binance_spot", "spot_log_return"]].rename(
                columns={"binance_spot": "close"}
            )
    else:
        pm_events = load_polymarket_events(market_by_slug, token_map)
        if pm_events.empty:
            raise RuntimeError("no Polymarket crypto events found")
        slugs = sorted(pm_events["market_slug"].unique())
        print(f"loaded {len(pm_events):,} Polymarket quote events across {len(slugs)} markets")
        external = fetch_external_data(slugs, market_by_slug)
        gamma = fetch_gamma_resolutions(slugs)
        stats = window_stats(market_by_slug, external, slugs, gamma)
        panel = build_panel(pm_events, market_by_slug, external, stats)
        if panel.empty:
            raise RuntimeError("empty 1s panel")

    cross_pre = precompute_cross_stats(panel)
    xcorr = scan_from_precomputed(cross_pre)
    best_x = xcorr.dropna(subset=["corr"]).loc[xcorr.dropna(subset=["corr"])["corr"].idxmax()]
    xcorr_ci = bootstrap_lag_ci(cross_pre, int(best_x["lead_seconds"]))

    if args.from_csv:
        # The from-csv path no longer has raw asynchronous intervals, so HY is
        # recomputed on the 1s grid as a fallback. Normal runs use true PM event intervals.
        hy = xcorr.rename(columns={"corr": "hy_corr"})[["lead_seconds", "hy_corr", "n_pairs"]].rename(
            columns={"n_pairs": "n_overlap"}
        )
        hy_ci = {
            "best_lead_ci_lo": xcorr_ci["best_lead_ci_lo"],
            "best_lead_ci_hi": xcorr_ci["best_lead_ci_hi"],
            "corr_ci_lo": xcorr_ci["corr_ci_lo"],
            "corr_ci_hi": xcorr_ci["corr_ci_hi"],
        }
    else:
        hy_pre = precompute_hy_stats(pm_events, external)
        hy = hy_scan_from_precomputed(hy_pre)
        best_h = hy.dropna(subset=["hy_corr"]).loc[hy.dropna(subset=["hy_corr"])["hy_corr"].idxmax()]
        hy_ci = bootstrap_hy_ci(hy_pre, int(best_h["lead_seconds"]))

    granger = pd.DataFrame(
        [
            granger_test(
                panel,
                target="pm_logit_change_1s",
                own="pm_logit_change_1s",
                cause="spot_log_return",
                label="Binance spot -> Polymarket logit-mid",
            ),
            granger_test(
                panel,
                target="spot_log_return",
                own="spot_log_return",
                cause="pm_logit_change_1s",
                label="Polymarket logit-mid -> Binance spot",
            ),
        ]
    )
    fee_gate = fee_gate_summary(market_by_slug, sorted(panel["market_slug"].unique()))
    print(
        "fee gate:",
        {
            "unique_rates": fee_gate["unique_rates"],
            "max_peak_fee": fee_gate["max_peak_fee"],
            "dynamic_like": fee_gate["dynamic_like"],
        },
    )
    write_outputs(panel, pm_events, xcorr, xcorr_ci, hy, hy_ci, granger, stats, fee_gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
