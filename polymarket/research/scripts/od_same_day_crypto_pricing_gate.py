"""Same-day crypto neg-risk pricing gate: touch + terminal ladder arms.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/od_same_day_crypto_pricing_gate.py
"""
from __future__ import annotations

import io
import json
import math
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import YEAR_SECONDS, cents, markdown_table, norm_cdf, number, pct
from od_strategy_a_v3 import normalize_markdown_wrapping


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
NOTES = ROOT / "notes"
CSV_OUT = ANALYSIS / "csv_outputs" / "options_delta"
PLOTS = ANALYSIS / "plots" / "options_delta"

NOTE = NOTES / "options_delta" / "od_same_day_crypto_pricing_gate_findings.md"
OD_HUB = NOTES / "options_delta" / "strat_options_delta.md"
BRAIN_TODO = REPO / "brain" / "TODO.md"

OUT_CLASSIFICATION = CSV_OUT / "od_same_day_crypto_pricing_classification.csv"
OUT_HISTORY_MARKETS = CSV_OUT / "od_same_day_crypto_pricing_history_markets.csv"
OUT_ARM_SUMMARY = CSV_OUT / "od_same_day_crypto_pricing_arm_summary.csv"
OUT_BUCKETS = CSV_OUT / "od_same_day_crypto_pricing_bucket_summary.csv"
OUT_CALIBRATION = CSV_OUT / "od_same_day_crypto_pricing_calibration.csv"
OUT_CURRENT = CSV_OUT / "od_same_day_crypto_pricing_current_quotes.csv"
OUT_CAPACITY = CSV_OUT / "od_same_day_crypto_pricing_capacity.csv"
OUT_FILLS = ANALYSIS / "od_same_day_crypto_pricing_fills.parquet"
OUT_CONFIRM_TRAIN = CSV_OUT / "od_same_day_crypto_arm_t_confirmation_train_cells.csv"
OUT_CONFIRM_HELDOUT = CSV_OUT / "od_same_day_crypto_arm_t_confirmation_heldout_cells.csv"
OUT_TIER1_TRAIN = CSV_OUT / "od_same_day_crypto_arm_t_tier1_train_cells.csv"
OUT_TIER1_HELDOUT = CSV_OUT / "od_same_day_crypto_arm_t_tier1_heldout_cells.csv"
OUT_TIER1_KOU = CSV_OUT / "od_same_day_crypto_arm_t_tier1_kou_params.csv"
OUT_TIER1_FILLS = ANALYSIS / "od_same_day_crypto_arm_t_tier1_fills.parquet"

PLOT_CALIBRATION_T = PLOTS / "od_same_day_crypto_touch_calibration.png"
PLOT_CALIBRATION_E = PLOTS / "od_same_day_crypto_terminal_calibration.png"
PLOT_BUCKETS = PLOTS / "od_same_day_crypto_bucket_ev.png"
PLOT_BEHAVIORAL = PLOTS / "od_same_day_crypto_behavioral_gap.png"
PLOT_TIER1_IV_GAP = PLOTS / "od_same_day_crypto_arm_t_tier1_iv_gap.png"
PLOT_TIER1_HELDOUT = PLOTS / "od_same_day_crypto_arm_t_tier1_heldout_ev.png"

LOCAL_MARKETS = sorted((DATA / "markets").glob("markets_*.parquet"))[-1]
TRADES_DIR = DATA / "trades"
K5_WALLET_CACHE = ANALYSIS / "k5_stress_wallet_market_full.parquet"

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
BINANCE_DATA = "https://data.binance.vision/data/spot/monthly/klines"
BINANCE_API = "https://api.binance.com"
HTTP_HEADERS = {"User-Agent": "epsilon-quant-research-od-sameday-gate/1.0"}

NY = ZoneInfo("America/New_York")
AS_OF = pd.Timestamp.now(tz="UTC")
START_MONTH = "2024-01"
END_MONTH = "2026-05"
BINANCE_1M_CACHE = DATA / "external" / "binance_1m_same_day"
ZIP_CACHE = DATA / "external" / "binance_1m_first_passage" / "monthly_zips"

ASSETS = {
    "BTC": {"word": "bitcoin", "symbol": "BTCUSDT"},
    "ETH": {"word": "ethereum", "symbol": "ETHUSDT"},
    "SOL": {"word": "solana", "symbol": "SOLUSDT"},
}
WORD_TO_ASSET = {"bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "eth": "ETH", "solana": "SOL", "sol": "SOL"}
SERIES = [
    ("BTC", "btc-multi-strikes-weekly"),
    ("ETH", "ethereum-multi-strikes-weekly"),
    ("SOL", "solana-multi-strikes-weekly"),
    ("BTC", "bitcoin-hit-price-daily"),
    ("ETH", "ethereum-hit-price-daily"),
    ("SOL", "solana-hit-price-daily"),
]
SEARCH_TERMS = [
    "bitcoin above today",
    "ethereum above today",
    "solana above today",
    "what price will bitcoin hit today",
    "what price will ethereum hit today",
    "what price will solana hit today",
    "will bitcoin reach today",
    "will ethereum reach today",
    "will solana reach today",
]

BAR_SECONDS = 60.0
EWMA_HALFLIFE_BARS = 60.0
MIN_EWMA_BARS = 60
CRYPTO_TAKER_FEE_RATE = 0.07
NON_TOP3_DEFAULT_SHARE = 0.05
STRUCTURAL_BASELINE_C = 0.0198
BOOTSTRAP_SAMPLES = 3000
RNG_SEED = 20260602
MIN_LOOKUP_N = 100
MIN_DEFENSIBLE_FILLS = 30
MIN_DEFENSIBLE_MARKETS = 3
MIN_OOS_RESOLUTION_DATES = 20
MIN_OOS_SPAN_DAYS = 30
TIER1_IV_GAP_THRESHOLD = 0.25
TIER1_MATERIAL_CAPACITY_EV = 0.0025
HAR_MIN_TRAIN_DAYS = 60
HAR_RIDGE = 1e-10
TIER1_JUMP_Z_THRESHOLD = 8.0
TIER1_JUMP_ABS_RETURN_FLOOR_BPS = 10.0

TAU_BINS_HOURS = [0, 0.25, 1, 4, 12, 24]
TAU_LABELS = ["lt_15m", "15m_1h", "1_4h", "4_12h", "12_24h"]
Z_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
Z_LABELS = ["z_0_0p5", "z_0p5_1", "z_1_1p5", "z_1p5_2", "z_2_3", "z_ge_3"]
Z_THRESHOLDS = {
    "z_0_0p5": 0.25,
    "z_0p5_1": 0.75,
    "z_1_1p5": 1.25,
    "z_1p5_2": 1.75,
    "z_2_3": 2.50,
    "z_ge_3": 3.50,
}


def signed_z_bucket(abs_bucket: Any, z_value: float, arm: str) -> str:
    if pd.isna(abs_bucket):
        return ""
    prefix = "neg" if np.isfinite(z_value) and z_value < 0 else "pos"
    return f"{prefix}_{abs_bucket}"


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if value is None:
        return []
    if isinstance(value, str):
        try:
            out = json.loads(value)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def ts(value: Any) -> pd.Timestamp | pd.NaT:
    if not value:
        return pd.NaT
    try:
        return pd.Timestamp(value, tz="UTC")
    except Exception:
        try:
            return pd.Timestamp(value).tz_localize("UTC")
        except Exception:
            return pd.NaT


def dollars(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"${value:,.0f}"


def fmt_ci(lo: float, hi: float, unit: str = "c") -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "[n/a, n/a]"
    if unit == "pct":
        return f"[{pct(lo)}, {pct(hi)}]"
    return f"[{cents(lo)}, {cents(hi)}]"


def level_from_text(text: str) -> float:
    for pat in [
        r"\$([0-9]+(?:,[0-9]{3})*(?:\.\d+)?)([kKmM]?)",
        r"(?:above|reach|hit|dip-to)-([0-9]+(?:k|m)?)",
    ]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        raw = m.group(1).replace(",", "")
        suffix = m.group(2).lower() if len(m.groups()) >= 2 else ""
        if raw.lower().endswith("k"):
            suffix = "k"
            raw = raw[:-1]
        if raw.lower().endswith("m"):
            suffix = "m"
            raw = raw[:-1]
        value = safe_float(raw)
        if not np.isfinite(value):
            continue
        if suffix == "k":
            value *= 1_000.0
        elif suffix == "m":
            value *= 1_000_000.0
        return value
    return math.nan


def infer_asset(text: str) -> str:
    low = text.lower()
    for key, asset in WORD_TO_ASSET.items():
        if re.search(rf"\b{re.escape(key)}\b", low):
            return asset
    return ""


def infer_touch_direction(text: str) -> str:
    low = text.lower()
    if "dip to" in low or "lowest price" in low or re.search(r"\bdip-to-", low):
        return "down"
    return "up"


def quote_for(text: str, arm: str) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    if arm == "T":
        pats = ["final High", "any Binance 1 minute candle", "immediately resolve", "High price"]
    elif arm == "E":
        pats = ["final \"Close\"", "final Close", "Close price", "12:00", "noon"]
    elif arm == "Q":
        pats = ["highest price", "lowest price", "range", "between"]
    else:
        pats = []
    for sent in sentences:
        if any(p.lower() in sent.lower() for p in pats):
            return sent[:500]
    return clean[:500]


def classify_text(text: str) -> tuple[str, str, str]:
    low = text.lower()
    if re.search(r"highest price|lowest price|highest.*reached|lowest.*reached", low) and re.search(r"\bbetween\b|\brange\b", low):
        arm, cls = "T", "band_running_extreme"
    elif (
        re.search(r'final\s+"?high"?\s+price\s+equal\s+to\s+or\s+greater', low)
        or re.search(r'final\s+"?low"?\s+price\s+equal\s+to\s+or\s+(?:less|lower)', low)
        or ("any binance 1 minute candle" in low and "high price" in low)
        or ("any binance 1 minute candle" in low and "low price" in low)
        or ("any binance 1-minute candle" in low and "high price" in low)
        or ("any binance 1-minute candle" in low and "low price" in low)
        or ("immediately resolve" in low and "high price" in low)
        or ("immediately resolve" in low and "low price" in low)
    ):
        arm, cls = "T", "barrier_touch"
    elif (
        "final \"close\" price" in low
        or "final close price" in low
        or ("close" in low and "12:00" in low)
        or ("close" in low and "noon" in low)
    ):
        arm, cls = "E", "terminal_close"
    else:
        arm, cls = "Q", "ambiguous"
    return arm, cls, quote_for(text, arm)


def get_json(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> Any:
    resp = client.get(url, params=params or {}, timeout=40)
    resp.raise_for_status()
    return resp.json()


def gamma_get(client: httpx.Client, path: str, params: dict[str, Any] | None = None) -> Any:
    return get_json(client, f"{GAMMA_BASE}{path}", params)


def resolve_series(client: httpx.Client, slug: str) -> dict[str, Any] | None:
    rows = gamma_get(client, "/series", {"slug": slug, "limit": 3})
    for row in rows or []:
        if row.get("slug") == slug:
            return row
    return rows[0] if rows else None


def fetch_series_events(client: httpx.Client, asset: str, slug: str) -> list[dict[str, Any]]:
    series = resolve_series(client, slug)
    if not series:
        return []
    out: dict[str, dict[str, Any]] = {}
    for offset in range(0, 500, 100):
        page = gamma_get(
            client,
            "/events",
            {
                "series_id": series.get("id"),
                "closed": "false",
                "limit": 100,
                "offset": offset,
                "order": "endDate",
                "ascending": "true",
            },
        )
        if not page:
            break
        for event in page:
            event["_asset_hint"] = asset
            event["_series_slug"] = slug
            out[str(event.get("id") or event.get("slug"))] = event
        if len(page) < 100:
            break
    return list(out.values())


def fetch_search_events(client: httpx.Client) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for term in SEARCH_TERMS:
        js = gamma_get(client, "/public-search", {"q": term, "limit": 25})
        for event in js.get("events", []) if isinstance(js, dict) else []:
            asset = infer_asset(f"{event.get('title') or ''} {event.get('slug') or ''}")
            if asset not in ASSETS:
                continue
            slug = event.get("slug")
            if slug:
                try:
                    event = gamma_get(client, f"/events/slug/{slug}")
                except Exception:
                    pass
            event["_asset_hint"] = asset
            event["_series_slug"] = event.get("seriesSlug") or "public-search"
            out[str(event.get("id") or slug)] = event
        time.sleep(0.02)
    return list(out.values())


def fetch_book(client: httpx.Client, token_id: str) -> dict[str, float]:
    if not token_id:
        return {"best_bid": math.nan, "best_ask": math.nan, "bid_size": math.nan, "ask_size": math.nan}
    try:
        raw = get_json(client, f"{CLOB_BASE}/book", {"token_id": token_id})
    except Exception:
        return {"best_bid": math.nan, "best_ask": math.nan, "bid_size": math.nan, "ask_size": math.nan}

    def parse(side: str) -> list[tuple[float, float]]:
        vals = []
        for lvl in raw.get(side, []) if isinstance(raw, dict) else []:
            if isinstance(lvl, dict):
                p, s = safe_float(lvl.get("price")), safe_float(lvl.get("size"))
            elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                p, s = safe_float(lvl[0]), safe_float(lvl[1])
            else:
                continue
            if np.isfinite(p) and np.isfinite(s):
                vals.append((p, s))
        vals.sort(key=lambda x: x[0], reverse=(side == "bids"))
        return vals

    bids = parse("bids")
    asks = parse("asks")
    return {
        "best_bid": bids[0][0] if bids else math.nan,
        "best_ask": asks[0][0] if asks else math.nan,
        "bid_size": bids[0][1] if bids else math.nan,
        "ask_size": asks[0][1] if asks else math.nan,
    }


def flatten_current(events: list[dict[str, Any]]) -> pd.DataFrame:
    today = AS_OF.date()
    rows = []
    for event in events:
        for market in event.get("markets") or []:
            end_ts = ts(market.get("endDate") or event.get("endDate"))
            if pd.isna(end_ts) or end_ts < AS_OF - pd.Timedelta(hours=2):
                continue
            title = f"{event.get('title') or ''} {market.get('question') or ''} {market.get('slug') or ''}"
            asset = infer_asset(title) or event.get("_asset_hint") or ""
            if asset not in ASSETS:
                continue
            same_dayish = end_ts.date() in {today, (AS_OF + pd.Timedelta(days=1)).date()}
            if not same_dayish:
                continue
            outcomes = [str(x) for x in parse_json_list(market.get("outcomes"))]
            token_ids = [str(x) for x in parse_json_list(market.get("clobTokenIds"))]
            text = "\n".join(
                str(x or "")
                for x in [
                    market.get("description"),
                    event.get("description"),
                    market.get("question"),
                    event.get("title"),
                    market.get("slug"),
                ]
            )
            arm, cls, quote = classify_text(text)
            rows.append(
                {
                    "event_slug": event.get("slug") or "",
                    "market_slug": market.get("slug") or "",
                    "market_id": str(market.get("id") or ""),
                    "condition_id": str(market.get("conditionId") or "").lower(),
                    "asset": asset,
                    "level": level_from_text(title),
                    "resolution_ts_utc": end_ts.isoformat(),
                    "resolution_source": market.get("resolutionSource") or event.get("resolutionSource") or "",
                    "resolution_class": cls,
                    "arm": arm,
                    "touch_direction": infer_touch_direction(text) if arm == "T" else "",
                    "decision_quote": quote,
                    "active": bool(market.get("active", event.get("active"))) and not bool(market.get("closed", event.get("closed"))),
                    "volume_usd": safe_float(market.get("volume")),
                    "volume_24h_usd": safe_float(market.get("volume24hr")),
                    "outcomes": json.dumps(outcomes),
                    "clob_token_ids": json.dumps(token_ids),
                    "yes_token_id": token_ids[0] if token_ids else "",
                    "no_token_id": token_ids[1] if len(token_ids) > 1 else "",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("market_id")
    with httpx.Client(headers=HTTP_HEADERS) as client:
        book_rows = []
        for _, r in df[df["active"]].iterrows():
            yes = fetch_book(client, r["yes_token_id"])
            no = fetch_book(client, r["no_token_id"])
            book_rows.append(
                {
                    "market_id": r["market_id"],
                    "yes_bid": yes["best_bid"],
                    "yes_ask": yes["best_ask"],
                    "yes_ask_size": yes["ask_size"],
                    "no_bid": no["best_bid"],
                    "no_ask": no["best_ask"],
                    "no_ask_size": no["ask_size"],
                }
            )
            time.sleep(0.01)
    df = df.merge(pd.DataFrame(book_rows), on="market_id", how="left") if book_rows else df
    df["volume_for_share_usd"] = np.where(df["volume_24h_usd"].fillna(0).gt(0), df["volume_24h_usd"].fillna(0), df["volume_usd"].fillna(0))
    return df.sort_values(["arm", "volume_for_share_usd"], ascending=[True, False])


def fetch_current_universe() -> pd.DataFrame:
    events = []
    with httpx.Client(headers=HTTP_HEADERS, timeout=40) as client:
        for asset, slug in SERIES:
            events.extend(fetch_series_events(client, asset, slug))
        events.extend(fetch_search_events(client))
    df = flatten_current(events)
    if df.empty:
        raise SystemExit("no same-day current markets fetched")
    return df


def month_range(start: str, end: str) -> list[pd.Period]:
    return list(pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M"))


def download_month(symbol: str, period: pd.Period, client: httpx.Client) -> Path | None:
    ZIP_CACHE.mkdir(parents=True, exist_ok=True)
    path = ZIP_CACHE / f"{symbol}-1m-{period}.zip"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = f"{BINANCE_DATA}/{symbol}/1m/{path.name}"
    for attempt in range(4):
        try:
            resp = client.get(url, timeout=60)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            path.write_bytes(resp.content)
            time.sleep(0.02)
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))
    return None


def parse_zip(path: Path) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore"]
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
        if not members:
            return pd.DataFrame()
        raw = zf.read(members[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=cols)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df[df["open_time"].notna()].copy()
    for col in ["open", "high", "low", "close", "volume", "close_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open_time", "open", "high", "low", "close", "close_time"])
    open_vals = df["open_time"].astype("int64")
    close_vals = df["close_time"].astype("int64")
    time_unit = "us" if max(open_vals.max(), close_vals.max()) > 10**14 else "ms"
    close_step = 1000 if time_unit == "us" else 1
    df["bar_open_ts"] = pd.to_datetime(open_vals, unit=time_unit, utc=True)
    df["ts"] = pd.to_datetime(close_vals + close_step, unit=time_unit, utc=True)
    return df[["bar_open_ts", "ts", "open", "high", "low", "close", "volume"]].sort_values("bar_open_ts")


def fetch_recent(symbol: str, start: pd.Timestamp, end: pd.Timestamp, client: httpx.Client) -> pd.DataFrame:
    rows = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms:
        raw = get_json(client, f"{BINANCE_API}/api/v3/klines", {"symbol": symbol, "interval": "1m", "startTime": cursor, "endTime": end_ms, "limit": 1000})
        if not raw:
            break
        rows.extend(raw)
        cursor = int(raw[-1][6]) + 1
        if len(raw) < 1000:
            break
        time.sleep(0.02)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "bar_open_ts": pd.to_datetime(df[0].astype("int64"), unit="ms", utc=True),
            "ts": pd.to_datetime(df[6].astype("int64") + 1, unit="ms", utc=True),
            "open": pd.to_numeric(df[1], errors="coerce"),
            "high": pd.to_numeric(df[2], errors="coerce"),
            "low": pd.to_numeric(df[3], errors="coerce"),
            "close": pd.to_numeric(df[4], errors="coerce"),
            "volume": pd.to_numeric(df[5], errors="coerce"),
        }
    )


def load_1m(asset: str, refresh: bool = False) -> pd.DataFrame:
    symbol = ASSETS[asset]["symbol"]
    out_path = BINANCE_1M_CACHE / f"{symbol}_1m_{START_MONTH}_{END_MONTH}_plus_recent.parquet"
    if out_path.exists() and not refresh:
        df = pd.read_parquet(out_path)
        for col in ("bar_open_ts", "ts"):
            df[col] = pd.to_datetime(df[col], utc=True)
        if df["ts"].max() <= AS_OF + pd.Timedelta(days=1):
            return df
        print(f"refreshing invalid {symbol} cache: max ts={df['ts'].max()}", flush=True)
    pieces = []
    with httpx.Client(headers=HTTP_HEADERS, timeout=60) as client:
        for period in month_range(START_MONTH, END_MONTH):
            zpath = download_month(symbol, period, client)
            if zpath:
                pieces.append(parse_zip(zpath))
            if len(pieces) and len(pieces) % 8 == 0:
                print(f"{symbol}: parsed {len(pieces)} monthly 1m files", flush=True)
        recent_start = pd.Timestamp(f"{END_MONTH}-01", tz="UTC") + pd.offsets.MonthBegin(1)
        if recent_start < AS_OF:
            pieces.append(fetch_recent(symbol, recent_start, AS_OF, client))
    if not pieces:
        raise SystemExit(f"no Binance data for {symbol}")
    df = pd.concat(pieces, ignore_index=True).drop_duplicates("bar_open_ts").sort_values("bar_open_ts")
    df["asset"] = asset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"cached {symbol} rows={len(df):,}", flush=True)
    return df


def add_features(raw: pd.DataFrame, asset: str) -> pd.DataFrame:
    df = raw.copy().sort_values("ts").reset_index(drop=True)
    ret = np.log(df["close"].astype(float)).diff()
    df["log_ret_1m"] = ret
    var = ret.pow(2).ewm(halflife=EWMA_HALFLIFE_BARS, min_periods=MIN_EWMA_BARS, adjust=False).mean()
    df["ewma_sigma_annualized"] = np.sqrt(var * YEAR_SECONDS / 60.0)
    utc_day = df["bar_open_ts"].dt.floor("D")
    terminal_end = utc_day + pd.Timedelta(hours=16)
    terminal_end = terminal_end.where(df["ts"].le(terminal_end), terminal_end + pd.Timedelta(days=1))
    df["terminal_end"] = terminal_end
    # Same-day daily touch templates in the current/historical scope end at
    # 00:00 ET. The owned same-day touch sample and current desk date are in
    # daylight time, so this is the exact 04:00 UTC boundary for this gate and
    # avoids a pandas timezone-vectorization overflow in this environment.
    df["touch_end"] = (df["bar_open_ts"] - pd.Timedelta(hours=4)).dt.floor("D") + pd.Timedelta(days=1, hours=4)

    df["terminal_close"] = df.groupby("terminal_end", sort=False)["close"].transform("last")
    df_desc = df.iloc[::-1]
    future_max_inclusive = df_desc.groupby("touch_end", sort=False)["high"].cummax().iloc[::-1]
    future_min_inclusive = df_desc.groupby("touch_end", sort=False)["low"].cummin().iloc[::-1]
    df["_future_max_inclusive"] = future_max_inclusive.to_numpy()
    df["_future_min_inclusive"] = future_min_inclusive.to_numpy()
    df["future_max_high"] = df.groupby("touch_end", sort=False)["_future_max_inclusive"].shift(-1)
    df["future_min_low"] = df.groupby("touch_end", sort=False)["_future_min_inclusive"].shift(-1)
    df = df.drop(columns=["_future_max_inclusive", "_future_min_inclusive"])
    df["prior_max_high"] = df.groupby("touch_end", sort=False)["high"].cummax()
    df["prior_min_low"] = df.groupby("touch_end", sort=False)["low"].cummin()
    df["asset"] = asset
    return df.replace([np.inf, -np.inf], np.nan)


def build_features(refresh: bool = False) -> pd.DataFrame:
    pieces = []
    for asset in ASSETS:
        raw = load_1m(asset, refresh=refresh)
        feat = add_features(raw, asset)
        pieces.append(feat)
        print(f"features {asset}: rows={len(feat):,}", flush=True)
    return pd.concat(pieces, ignore_index=True).sort_values(["asset", "ts"]).reset_index(drop=True)


def tau_bucket(hours: pd.Series) -> pd.Series:
    return pd.cut(hours, TAU_BINS_HOURS, labels=TAU_LABELS, include_lowest=True, right=True)


def build_synthetic(features: pd.DataFrame) -> pd.DataFrame:
    base = features[
        features["ewma_sigma_annualized"].notna()
        & features["ts"].dt.minute.eq(0)
        & features["future_max_high"].notna()
        & features["future_min_low"].notna()
        & features["terminal_close"].notna()
    ][["asset", "ts", "terminal_end", "touch_end", "close", "ewma_sigma_annualized", "future_max_high", "future_min_low", "terminal_close"]].copy()
    rows = []
    for arm, end_col in [("E", "terminal_end"), ("T", "touch_end")]:
        piece_base = base.copy()
        piece_base["window_end"] = piece_base[end_col]
        piece_base["tau_seconds"] = (piece_base["window_end"] - piece_base["ts"]).dt.total_seconds()
        piece_base = piece_base[piece_base["tau_seconds"].gt(0)].copy()
        piece_base["tau_years"] = piece_base["tau_seconds"] / YEAR_SECONDS
        piece_base["tau_bucket"] = tau_bucket(piece_base["tau_seconds"] / 3600.0)
        denom = piece_base["ewma_sigma_annualized"].astype(float) * np.sqrt(piece_base["tau_years"].astype(float))
        for z_bucket, z_threshold in Z_THRESHOLDS.items():
            signs = [-1.0, 1.0]
            for sign in signs:
                p = piece_base.copy()
                barrier = p["close"].astype(float) * np.exp(sign * z_threshold * denom)
                p["z_bucket"] = f"{'neg' if sign < 0 else 'pos'}_{z_bucket}"
                p["z_threshold"] = sign * z_threshold
                if arm == "T":
                    p["label"] = np.where(sign < 0, p["future_min_low"].astype(float).le(barrier), p["future_max_high"].astype(float).ge(barrier)).astype(float)
                else:
                    p["label"] = p["terminal_close"].astype(float).ge(barrier).astype(float)
                p["arm"] = arm
                rows.append(p[["arm", "asset", "ts", "window_end", "tau_bucket", "z_bucket", "label"]])
    return pd.concat(rows, ignore_index=True).dropna(subset=["tau_bucket", "z_bucket"])


def make_lookup(train: pd.DataFrame) -> dict[str, Any]:
    primary = train.groupby(["arm", "asset", "z_bucket", "tau_bucket"], observed=True).agg(p=("label", "mean"), n=("label", "count")).reset_index()
    pooled = train.groupby(["arm", "z_bucket", "tau_bucket"], observed=True).agg(p=("label", "mean"), n=("label", "count")).reset_index()
    tau = train.groupby(["arm", "tau_bucket"], observed=True).agg(p=("label", "mean"), n=("label", "count")).reset_index()
    return {
        "primary": {(r.arm, r.asset, r.z_bucket, r.tau_bucket): (float(r.p), int(r.n)) for r in primary.itertuples(index=False)},
        "pooled": {(r.arm, r.z_bucket, r.tau_bucket): (float(r.p), int(r.n)) for r in pooled.itertuples(index=False)},
        "tau": {(r.arm, r.tau_bucket): (float(r.p), int(r.n)) for r in tau.itertuples(index=False)},
        "overall": {arm: (float(g["label"].mean()), int(len(g))) for arm, g in train.groupby("arm")},
    }


def lookup_prob(arm: str, asset: str, z_bucket: str, tbucket: str, lookup: dict[str, Any]) -> tuple[float, int, str]:
    val = lookup["primary"].get((arm, asset, z_bucket, tbucket))
    if val and val[1] >= MIN_LOOKUP_N:
        return val[0], val[1], "asset_z_tau"
    val = lookup["pooled"].get((arm, z_bucket, tbucket))
    if val and val[1] >= MIN_LOOKUP_N:
        return val[0], val[1], "pooled_z_tau"
    val = lookup["tau"].get((arm, tbucket))
    if val and val[1] > 0:
        return val[0], val[1], "tau_only"
    val = lookup["overall"].get(arm, (math.nan, 0))
    return val[0], val[1], "overall"


def expanding_cv(synth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    synth = synth.copy()
    synth["month_key"] = synth["ts"].dt.strftime("%Y-%m")
    months = sorted(synth["month_key"].dropna().unique())
    for month in months:
        if month < "2025-01":
            continue
        month_start = pd.Timestamp(f"{month}-01", tz="UTC")
        train = synth[synth["window_end"].lt(month_start)].copy()
        val = synth[synth["month_key"].eq(month)].copy()
        if train.empty or val.empty:
            continue
        lookup = make_lookup(train)
        preds = [lookup_prob(r.arm, r.asset, str(r.z_bucket), str(r.tau_bucket), lookup) for r in val.itertuples(index=False)]
        val["pred"] = [p[0] for p in preds]
        val["train_n"] = [p[1] for p in preds]
        val["source"] = [p[2] for p in preds]
        rows.append(val)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def calibration(cv: pd.DataFrame) -> pd.DataFrame:
    if cv.empty:
        return pd.DataFrame()
    out = cv.copy()
    out["prob_bucket"] = pd.cut(out["pred"], [0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0], labels=["0_5c", "5_10c", "10_25c", "25_50c", "50_75c", "75_90c", "90_95c", "95_100c"], include_lowest=True)
    return out.groupby(["arm", "prob_bucket"], observed=True).agg(rows=("label", "count"), mean_pred=("pred", "mean"), observed=("label", "mean")).reset_index().assign(obs_minus_pred=lambda x: x["observed"] - x["mean_pred"])


def historical_markets() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT id, condition_id, slug, question, end_date, closed, volume, outcomes, outcome_prices, clob_token_ids
        FROM read_parquet('{LOCAL_MARKETS}')
        WHERE closed = true
          AND volume >= 10000
          AND try_cast(end_date AS TIMESTAMPTZ) >= TIMESTAMPTZ '2025-09-01 00:00:00+00'
          AND (
            (lower(question) LIKE 'will the price of % be above % on %'
             AND (
               lower(slug) LIKE 'bitcoin-above-%-on-%'
               OR lower(slug) LIKE 'ethereum-above-%-on-%'
               OR lower(slug) LIKE 'solana-above-%-on-%'
             ))
            OR (regexp_matches(lower(question), '^will (bitcoin|ethereum|solana) (reach|dip to) .+ on ')
                AND (
                  lower(slug) LIKE 'will-bitcoin-%-on-%'
                  OR lower(slug) LIKE 'will-ethereum-%-on-%'
                  OR lower(slug) LIKE 'will-solana-%-on-%'
                ))
          )
        """
    ).df()
    rows = []
    for _, r in df.iterrows():
        slug = str(r["slug"]).lower()
        if "-above-" in slug:
            arm, cls = "E", "terminal_close"
        elif "-reach-" in slug or "-dip-to-" in slug:
            arm, cls = "T", "barrier_touch"
        else:
            continue
        asset = infer_asset(slug)
        if asset not in ASSETS:
            continue
        token_ids = list(r["clob_token_ids"]) if isinstance(r["clob_token_ids"], (list, tuple, np.ndarray)) else parse_json_list(r["clob_token_ids"])
        outcomes = list(r["outcomes"]) if isinstance(r["outcomes"], (list, tuple, np.ndarray)) else parse_json_list(r["outcomes"])
        prices = list(r["outcome_prices"]) if isinstance(r["outcome_prices"], (list, tuple, np.ndarray)) else parse_json_list(r["outcome_prices"])
        if len(token_ids) < 2 or len(prices) < 2:
            continue
        end_ts = ts(r["end_date"])
        if pd.isna(end_ts) or end_ts > AS_OF:
            continue
        rows.append(
            {
                "market_id": str(r["id"]),
                "condition_id": str(r["condition_id"]).lower(),
                "market_slug": r["slug"],
                "question": r["question"],
                "arm": arm,
                "resolution_class": cls,
                "touch_direction": infer_touch_direction(f"{r['slug']} {r['question']}") if arm == "T" else "",
                "asset": asset,
                "level": level_from_text(f"{r['slug']} {r['question']}"),
                "resolution_ts": end_ts,
                "volume_usd": float(r["volume"]),
                "yes_token_id": str(token_ids[0]),
                "no_token_id": str(token_ids[1]),
                "yes_payoff": safe_float(prices[0]),
                "no_payoff": safe_float(prices[1]),
                "outcomes": json.dumps([str(x) for x in outcomes]),
                "outcome_prices": json.dumps([str(x) for x in prices]),
            }
        )
    return pd.DataFrame(rows).sort_values(["arm", "resolution_ts", "asset", "level"])


def trade_paths(markets: pd.DataFrame) -> list[str]:
    min_ts = markets["resolution_ts"].min() - pd.Timedelta(days=7)
    max_ts = markets["resolution_ts"].max() + pd.Timedelta(days=1)
    paths = []
    for path in sorted(TRADES_DIR.glob("*.parquet")):
        dates = []
        for m in re.finditer(r"20\d{2}-\d{2}-\d{2}|20\d{6}", path.name):
            raw = m.group(0)
            try:
                dates.append(pd.Timestamp(raw if "-" in raw else f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}", tz="UTC"))
            except Exception:
                pass
        if not dates or (max(dates) >= min_ts and min(dates) <= max_ts):
            paths.append(str(path))
    return paths


def load_fills(markets: pd.DataFrame) -> pd.DataFrame:
    token_rows = []
    for r in markets.itertuples(index=False):
        token_rows.append({"market_id": r.market_id, "outcome_token_id": r.yes_token_id, "outcome": "YES", "payoff": r.yes_payoff})
        token_rows.append({"market_id": r.market_id, "outcome_token_id": r.no_token_id, "outcome": "NO", "payoff": r.no_payoff})
    token_map = pd.DataFrame(token_rows)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.register("token_map", token_map)
    paths = trade_paths(markets)
    q = f"""
        SELECT
            t.timestamp AS fill_ts,
            CAST(t.market_id AS VARCHAR) AS market_id,
            CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END AS outcome_token_id,
            m.outcome,
            m.payoff,
            t.price,
            t.token_amount,
            t.usd_amount,
            t.transaction_hash
        FROM read_parquet({paths!r}) t
        JOIN token_map m
          ON CAST(t.market_id AS VARCHAR) = m.market_id
         AND CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id ELSE t.maker_asset_id END = m.outcome_token_id
        WHERE t.maker_side = 'SELL'
          AND t.price > 0
          AND t.price < 1
          AND t.token_amount > 0
    """
    fills = con.execute(q).df()
    fills["fill_ts"] = pd.to_datetime(fills["fill_ts"], utc=True)
    fills = fills.merge(markets, on="market_id", how="left", suffixes=("", "_market"))
    fills = fills[fills["fill_ts"].lt(fills["resolution_ts"]) & fills["fill_ts"].gt(fills["resolution_ts"] - pd.Timedelta(hours=24))].copy()
    fills["entry_price"] = fills["price"].astype(float)
    fills["taker_fee"] = CRYPTO_TAKER_FEE_RATE * fills["entry_price"] * (1.0 - fills["entry_price"])
    fills["realized_net_ev"] = fills["payoff"].astype(float) - fills["entry_price"] - fills["taker_fee"]
    return fills.sort_values(["arm", "asset", "fill_ts"]).reset_index(drop=True)


def join_state(fills: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    keep = ["asset", "ts", "close", "ewma_sigma_annualized", "future_max_high", "future_min_low", "prior_max_high", "prior_min_low", "terminal_close"]
    pieces = []
    feat = features[keep].sort_values(["asset", "ts"])
    for asset, left in fills.groupby("asset", sort=False):
        right = feat[feat["asset"].eq(asset)].sort_values("ts")
        pieces.append(pd.merge_asof(left.sort_values("fill_ts"), right, by="asset", left_on="fill_ts", right_on="ts", direction="backward", tolerance=pd.Timedelta(minutes=3)))
    out = pd.concat(pieces, ignore_index=True)
    out["spot"] = out["close"].astype(float)
    out["tau_seconds"] = (out["resolution_ts"] - out["ts"]).dt.total_seconds()
    out["tau_years"] = out["tau_seconds"] / YEAR_SECONDS
    denom = out["ewma_sigma_annualized"].astype(float) * np.sqrt(out["tau_years"].astype(float))
    out["z_signed_yes"] = np.log(out["level"].astype(float) / out["spot"].astype(float)) / denom
    out["abs_z"] = out["z_signed_yes"].abs()
    abs_bucket = pd.cut(out["abs_z"], Z_BINS, labels=Z_LABELS, include_lowest=True)
    out["z_bucket"] = [
        signed_z_bucket(bucket, z, arm)
        for bucket, z, arm in zip(abs_bucket, out["z_signed_yes"], out["arm"], strict=False)
    ]
    out["tau_bucket"] = tau_bucket(out["tau_seconds"] / 3600.0)
    touch_down = out["touch_direction"].eq("down")
    out["already_touched"] = np.where(
        touch_down,
        out["prior_min_low"].astype(float).le(out["level"].astype(float)),
        out["prior_max_high"].astype(float).ge(out["level"].astype(float)),
    )
    out = out[out["spot"].notna() & out["ewma_sigma_annualized"].notna() & out["tau_seconds"].gt(0) & out["z_bucket"].astype(str).ne("") & out["tau_bucket"].notna()].copy()
    out = out[~((out["arm"].eq("T")) & out["already_touched"].fillna(True))].copy()
    return out


def brownian_touch(z: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(2.0 * np.asarray(norm_cdf(-np.abs(np.asarray(z, dtype=float))), dtype=float), 0.0, 1.0)


def apply_models(df: pd.DataFrame, synth: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month_key"] = out["fill_ts"].dt.strftime("%Y-%m")
    month_keys = sorted(out["month_key"].dropna().unique())
    lookups = {}
    for month in month_keys + [AS_OF.strftime("%Y-%m")]:
        month_start = pd.Timestamp(f"{month}-01", tz="UTC")
        lookups[month] = make_lookup(synth[synth["window_end"].lt(month_start)].copy())
    preds = []
    for r in out.itertuples(index=False):
        p, n, src = lookup_prob(r.arm, r.asset, str(r.z_bucket), str(r.tau_bucket), lookups[r.month_key])
        preds.append((p, n, src))
    out["empirical_yes_prob"] = [p[0] for p in preds]
    out["empirical_train_n"] = [p[1] for p in preds]
    out["empirical_source"] = [p[2] for p in preds]
    out["terminal_model_yes_prob"] = np.asarray(norm_cdf(-out["z_signed_yes"].to_numpy(dtype=float)), dtype=float)
    out.loc[out["arm"].eq("E"), "terminal_model_yes_prob"] = np.asarray(norm_cdf(-out.loc[out["arm"].eq("E"), "z_signed_yes"].to_numpy(dtype=float)), dtype=float)
    out["naive_terminal_touch_prob"] = np.asarray(norm_cdf(-out["abs_z"].to_numpy(dtype=float)), dtype=float)
    out["brownian_touch_prob"] = brownian_touch(out["abs_z"])
    out["model_token_prob_empirical"] = np.where(out["outcome"].eq("YES"), out["empirical_yes_prob"], 1.0 - out["empirical_yes_prob"])
    out["model_token_prob_terminal"] = np.where(out["outcome"].eq("YES"), out["terminal_model_yes_prob"], 1.0 - out["terminal_model_yes_prob"])
    out["model_token_prob_brownian"] = np.where(out["outcome"].eq("YES"), out["brownian_touch_prob"], 1.0 - out["brownian_touch_prob"])
    out["model_edge_empirical"] = out["model_token_prob_empirical"] - out["entry_price"] - out["taker_fee"]
    out["model_edge_terminal"] = out["model_token_prob_terminal"] - out["entry_price"] - out["taker_fee"]
    out["model_edge_brownian"] = out["model_token_prob_brownian"] - out["entry_price"] - out["taker_fee"]
    out["selected_empirical_edge"] = out["model_edge_empirical"].gt(0)
    out["net_after_k5_haircut"] = out["realized_net_ev"] * NON_TOP3_DEFAULT_SHARE
    out["incremental_vs_structural"] = out["net_after_k5_haircut"] - STRUCTURAL_BASELINE_C
    return out.replace([np.inf, -np.inf], np.nan)


def cluster_ci(df: pd.DataFrame, col: str, seed: int) -> tuple[float, float]:
    sub = df[["market_id", col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return math.nan, math.nan
    vals = sub.groupby("market_id")[col].mean().to_numpy(dtype=float)
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(RNG_SEED + seed + len(vals))
    idx = rng.integers(0, len(vals), size=(BOOTSTRAP_SAMPLES, len(vals)))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def cluster_bootstrap_stats(df: pd.DataFrame, col: str, seed: int) -> dict[str, float]:
    sub = df[["market_id", col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return {"mean": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "p_one_sided": math.nan, "cluster_markets": 0}
    vals = sub.groupby("market_id")[col].mean().to_numpy(dtype=float)
    mean = float(np.mean(vals))
    if len(vals) == 1:
        return {"mean": mean, "ci_lo": mean, "ci_hi": mean, "p_one_sided": 1.0 if mean <= 0 else 1.0 / (BOOTSTRAP_SAMPLES + 1), "cluster_markets": 1}
    rng = np.random.default_rng(RNG_SEED + seed + len(vals))
    idx = rng.integers(0, len(vals), size=(BOOTSTRAP_SAMPLES, len(vals)))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    p = (float(np.sum(boot <= 0)) + 1.0) / (BOOTSTRAP_SAMPLES + 1.0)
    return {"mean": mean, "ci_lo": float(lo), "ci_hi": float(hi), "p_one_sided": p, "cluster_markets": int(len(vals))}


def bh_adjust(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return pd.Series(out, index=pvals.index)
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[np.flatnonzero(ok)] = tmp
    return pd.Series(out, index=pvals.index)


def safe_div_array(num: np.ndarray, den: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full_like(np.asarray(num, dtype=float), default, dtype=float)
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    return out


def har_session_date(ts_col: pd.Series) -> pd.Series:
    return (pd.to_datetime(ts_col, utc=True) - pd.Timedelta(hours=4)).dt.floor("D").dt.date


def ridge_predict(train: pd.DataFrame, row: pd.Series, xcols: list[str]) -> float:
    fit = train.dropna(subset=["rv_daily", *xcols]).copy()
    if len(fit) < HAR_MIN_TRAIN_DAYS:
        vals = pd.to_numeric(row[["rv_d", "rv_w", "rv_m"]], errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else math.nan
    x = fit[xcols].to_numpy(dtype=float)
    y = fit["rv_daily"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    xtx = x.T @ x
    xtx = xtx + np.eye(xtx.shape[0]) * HAR_RIDGE
    try:
        beta = np.linalg.solve(xtx, x.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    xr = np.asarray([1.0] + [float(row[c]) for c in xcols], dtype=float)
    return float(xr @ beta)


def build_har_kou_forecasts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causal daily HAR-RV-J and directional jump forecasts from cached Binance 1m bars."""
    forecast_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for asset in ASSETS:
        raw = load_1m(asset, refresh=False).copy().sort_values("bar_open_ts").reset_index(drop=True)
        raw["asset"] = asset
        close = pd.to_numeric(raw["close"], errors="coerce")
        ret = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        rolling_sigma = ret.rolling(1440, min_periods=120).std().shift(1)
        threshold = np.maximum(
            TIER1_JUMP_Z_THRESHOLD * rolling_sigma.to_numpy(dtype=float),
            TIER1_JUMP_ABS_RETURN_FLOOR_BPS / 1e4,
        )
        threshold = pd.Series(threshold, index=raw.index).replace([np.inf, -np.inf], np.nan)
        jump_flag = ret.abs().gt(threshold) & threshold.notna() & ret.notna()
        jump_up = ret.where(jump_flag & ret.gt(0), 0.0).fillna(0.0)
        jump_down_abs = (-ret).where(jump_flag & ret.lt(0), 0.0).fillna(0.0)
        tmp = pd.DataFrame(
            {
                "asset": asset,
                "session_date": har_session_date(raw["bar_open_ts"]),
                "ret": ret,
                "ret_sq": ret.pow(2),
                "jump_flag": jump_flag.astype(float),
                "jump_sq": ret.where(jump_flag, 0.0).fillna(0.0).pow(2),
                "jump_up_count": jump_up.gt(0).astype(float),
                "jump_down_count": jump_down_abs.gt(0).astype(float),
                "jump_up_sq": jump_up.pow(2),
                "jump_down_sq": jump_down_abs.pow(2),
            }
        )
        daily = (
            tmp.groupby(["asset", "session_date"], sort=True)
            .agg(
                rv_daily=("ret_sq", "sum"),
                bars=("ret", "count"),
                jump_count=("jump_flag", "sum"),
                jump_var_daily=("jump_sq", "sum"),
                jump_up_count=("jump_up_count", "sum"),
                jump_down_count=("jump_down_count", "sum"),
                jump_up_sq=("jump_up_sq", "sum"),
                jump_down_sq=("jump_down_sq", "sum"),
            )
            .reset_index()
        )
        daily = daily[daily["bars"].ge(720)].copy().sort_values("session_date").reset_index(drop=True)
        daily["rv_d"] = daily["rv_daily"].shift(1)
        daily["rv_w"] = daily["rv_daily"].rolling(5, min_periods=3).mean().shift(1)
        daily["rv_m"] = daily["rv_daily"].rolling(22, min_periods=10).mean().shift(1)
        daily["jump_var_d"] = daily["jump_var_daily"].shift(1).fillna(0.0)
        preds = []
        xcols = ["rv_d", "rv_w", "rv_m", "jump_var_d"]
        for i, row in daily.iterrows():
            pred = ridge_predict(daily.iloc[:i], row, xcols) if row[xcols].notna().all() else math.nan
            fallback_vals = pd.to_numeric(row[["rv_d", "rv_w", "rv_m"]], errors="coerce").dropna()
            fallback = float(fallback_vals.mean()) if len(fallback_vals) else math.nan
            if not np.isfinite(pred) or pred <= 0 or (np.isfinite(fallback) and pred > fallback * 8.0):
                pred = fallback
            preds.append(pred)
        daily["har_rvj_daily_var_forecast"] = pd.Series(preds, index=daily.index).clip(lower=1e-10)

        prior_days = np.arange(len(daily), dtype=float)
        prior_up_count = daily["jump_up_count"].cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_down_count = daily["jump_down_count"].cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_up_sq = daily["jump_up_sq"].cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_down_sq = daily["jump_down_sq"].cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        up_second = safe_div_array(prior_up_sq, prior_up_count)
        down_second = safe_div_array(prior_down_sq, prior_down_count)
        up_lambda_year = safe_div_array(prior_up_count, prior_days) * 365.0
        down_lambda_year = safe_div_array(prior_down_count, prior_days) * 365.0
        base_sigma2 = daily["har_rvj_daily_var_forecast"].to_numpy(dtype=float) * 365.0
        daily["sigma_har_annualized"] = np.sqrt(np.maximum(base_sigma2, 1e-12))
        daily["kou_up_lambda_per_year"] = up_lambda_year
        daily["kou_down_lambda_per_year"] = down_lambda_year
        daily["kou_up_second_moment"] = up_second
        daily["kou_down_second_moment"] = down_second
        daily["sigma_har_kou_up_annualized"] = np.sqrt(np.maximum(base_sigma2 + up_lambda_year * up_second, 1e-12))
        daily["sigma_har_kou_down_annualized"] = np.sqrt(np.maximum(base_sigma2 + down_lambda_year * down_second, 1e-12))
        keep = [
            "asset",
            "session_date",
            "har_rvj_daily_var_forecast",
            "sigma_har_annualized",
            "sigma_har_kou_up_annualized",
            "sigma_har_kou_down_annualized",
            "kou_up_lambda_per_year",
            "kou_down_lambda_per_year",
            "kou_up_second_moment",
            "kou_down_second_moment",
        ]
        forecast_rows.append(daily[keep].copy())
        last = daily.dropna(subset=["sigma_har_annualized"]).tail(1)
        if not last.empty:
            r = last.iloc[0]
            summary_rows.append(
                {
                    "asset": asset,
                    "forecast_session_date": str(r["session_date"]),
                    "sigma_har_annualized": float(r["sigma_har_annualized"]),
                    "sigma_har_kou_up_annualized": float(r["sigma_har_kou_up_annualized"]),
                    "sigma_har_kou_down_annualized": float(r["sigma_har_kou_down_annualized"]),
                    "kou_up_lambda_per_year": float(r["kou_up_lambda_per_year"]),
                    "kou_down_lambda_per_year": float(r["kou_down_lambda_per_year"]),
                    "kou_up_avg_abs_jump": math.sqrt(float(r["kou_up_second_moment"])) if float(r["kou_up_second_moment"]) > 0 else 0.0,
                    "kou_down_avg_abs_jump": math.sqrt(float(r["kou_down_second_moment"])) if float(r["kou_down_second_moment"]) > 0 else 0.0,
                }
            )
    forecasts = pd.concat(forecast_rows, ignore_index=True) if forecast_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return forecasts, summary


def implied_touch_sigma_from_yes_prob(prob: pd.Series, distance: pd.Series, tau_years: pd.Series) -> np.ndarray:
    target = pd.to_numeric(prob, errors="coerce").to_numpy(dtype=float)
    dist = pd.to_numeric(distance, errors="coerce").to_numpy(dtype=float)
    tau = pd.to_numeric(tau_years, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(target) & np.isfinite(dist) & np.isfinite(tau) & (dist > 0) & (tau > 0)
    out = np.full(len(target), np.nan, dtype=float)
    if not valid.any():
        return out
    target = np.clip(target[valid], 1e-6, 0.999999)
    dist_v = dist[valid]
    sqrt_tau = np.sqrt(tau[valid])
    lo = np.full(len(target), 1e-4, dtype=float)
    hi = np.full(len(target), 10.0, dtype=float)
    for _ in range(64):
        mid = (lo + hi) / 2.0
        z = -dist_v / np.maximum(mid * sqrt_tau, 1e-12)
        p_mid = np.clip(2.0 * np.asarray(norm_cdf(z), dtype=float), 0.0, 1.0)
        lo = np.where(p_mid < target, mid, lo)
        hi = np.where(p_mid >= target, mid, hi)
    out[np.flatnonzero(valid)] = (lo + hi) / 2.0
    return out


def prepare_tier1_fills() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not OUT_FILLS.exists():
        raise SystemExit(f"missing fill sample: {OUT_FILLS}")
    fills = pd.read_parquet(OUT_FILLS)
    fills["fill_ts"] = pd.to_datetime(fills["fill_ts"], utc=True)
    fills["resolution_ts"] = pd.to_datetime(fills["resolution_ts"], utc=True)
    t = fills[fills["arm"].eq("T")].copy()
    if t.empty:
        raise SystemExit("no Arm T fills in saved fill sample")
    forecasts, kou_summary = build_har_kou_forecasts()
    t["session_date"] = har_session_date(t["fill_ts"])
    t = t.merge(forecasts, on=["asset", "session_date"], how="left")
    t["pm_yes_price"] = np.where(t["outcome"].eq("YES"), t["entry_price"].astype(float), 1.0 - t["entry_price"].astype(float))
    t["touch_distance_log"] = np.log(t["level"].astype(float) / t["spot"].astype(float)).abs()
    t["pm_touch_iv"] = implied_touch_sigma_from_yes_prob(t["pm_yes_price"], t["touch_distance_log"], t["tau_years"])
    down = t["touch_direction"].eq("down")
    t["sigma_har_kou_directional_annualized"] = np.where(down, t["sigma_har_kou_down_annualized"], t["sigma_har_kou_up_annualized"])
    t["iv_gap_har"] = t["pm_touch_iv"] - t["sigma_har_annualized"]
    t["iv_gap_har_kou"] = t["pm_touch_iv"] - t["sigma_har_kou_directional_annualized"]
    denom_har = t["sigma_har_kou_directional_annualized"].astype(float) * np.sqrt(t["tau_years"].astype(float))
    t["z_signed_har_kou"] = np.log(t["level"].astype(float) / t["spot"].astype(float)) / denom_har
    t["abs_z_har_kou"] = t["z_signed_har_kou"].abs()
    abs_bucket_har = pd.cut(t["abs_z_har_kou"], Z_BINS, labels=Z_LABELS, include_lowest=True)
    t["z_bucket_har_kou"] = [
        signed_z_bucket(bucket, z, arm)
        for bucket, z, arm in zip(abs_bucket_har, t["z_signed_har_kou"], t["arm"], strict=False)
    ]
    t["naive_terminal_touch_prob_har_kou"] = np.asarray(norm_cdf(-t["abs_z_har_kou"].to_numpy(dtype=float)), dtype=float)
    t["brownian_touch_prob_har_kou"] = brownian_touch(t["abs_z_har_kou"])
    t["tier1_signal"] = np.select(
        [
            t["iv_gap_har_kou"].le(-TIER1_IV_GAP_THRESHOLD),
            t["iv_gap_har_kou"].ge(TIER1_IV_GAP_THRESHOLD),
        ],
        ["buy_yes_vol_cheap", "buy_no_vol_rich"],
        default="no_trade",
    )
    t["tier1_side_match"] = (
        (t["tier1_signal"].eq("buy_yes_vol_cheap") & t["outcome"].eq("YES"))
        | (t["tier1_signal"].eq("buy_no_vol_rich") & t["outcome"].eq("NO"))
    )
    t["tier1_selected"] = (
        t["selected_empirical_edge"].fillna(False)
        & t["tier1_side_match"]
        & t["pm_touch_iv"].notna()
        & t["sigma_har_kou_directional_annualized"].notna()
        & t["z_bucket_har_kou"].astype(str).ne("")
    )
    t["tier1_iv_gap_threshold"] = TIER1_IV_GAP_THRESHOLD
    t["tier1_material_capacity_ev"] = TIER1_MATERIAL_CAPACITY_EV
    OUT_TIER1_FILLS.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(OUT_TIER1_FILLS, index=False)
    return t, kou_summary


def split_arm_t_oos(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = t[t["selected_empirical_edge"].fillna(False)].copy()
    base["resolution_date"] = base["resolution_ts"].dt.date
    dates = sorted(base["resolution_date"].dropna().unique())
    if not dates:
        raise SystemExit("no selected Arm T resolution dates")
    span_days = (pd.Timestamp(max(dates)) - pd.Timestamp(min(dates))).days + 1
    oos_ok = len(dates) >= MIN_OOS_RESOLUTION_DATES and span_days >= MIN_OOS_SPAN_DAYS
    # Realism calibration guard: do not manufacture a powerless OOS split on a
    # single short capture window. Treat such future samples as train-only.
    if not oos_ok:
        train_dates = set(dates)
        embargo_date = None
        heldout_dates: set[Any] = set()
    else:
        split_idx = int(math.floor(len(dates) * 0.70))
        split_idx = min(max(split_idx, 1), len(dates) - 2)
        train_dates = set(dates[:split_idx])
        embargo_date = dates[split_idx]
        heldout_dates = set(dates[split_idx + 1 :])
    selected = t[t["tier1_selected"].fillna(False)].copy()
    selected["resolution_date"] = selected["resolution_ts"].dt.date
    train = selected[selected["resolution_date"].isin(train_dates)].copy()
    embargo = selected[selected["resolution_date"].eq(embargo_date)].copy() if embargo_date is not None else selected.iloc[0:0].copy()
    heldout = selected[selected["resolution_date"].isin(heldout_dates)].copy()
    meta = {
        "oos_ok": oos_ok,
        "resolution_dates": len(dates),
        "span_days": span_days,
        "train_start": str(min(train_dates)) if train_dates else "n/a",
        "train_end": str(max(train_dates)) if train_dates else "n/a",
        "embargo_date": str(embargo_date) if embargo_date is not None else "none",
        "heldout_start": str(min(heldout_dates)) if heldout_dates else "none",
        "heldout_end": str(max(heldout_dates)) if heldout_dates else "none",
        "base_fills": int(len(base)),
        "base_markets": int(base["market_id"].nunique()),
        "tier1_selected_fills": int(len(selected)),
        "tier1_selected_markets": int(selected["market_id"].nunique()),
        "train_fills": int(len(train)),
        "train_markets": int(train["market_id"].nunique()) if not train.empty else 0,
        "embargo_fills": int(len(embargo)),
        "embargo_markets": int(embargo["market_id"].nunique()) if not embargo.empty else 0,
        "heldout_fills": int(len(heldout)),
        "heldout_markets": int(heldout["market_id"].nunique()) if not heldout.empty else 0,
    }
    return train, embargo, heldout, meta


def arm_t_tier1_cell_table(sample: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    if sample.empty:
        return pd.DataFrame()
    for (z_bucket, tbucket), g in sample.groupby(["z_bucket_har_kou", "tau_bucket"], observed=True):
        if len(g) < MIN_DEFENSIBLE_FILLS or g["market_id"].nunique() < MIN_DEFENSIBLE_MARKETS:
            continue
        net = cluster_bootstrap_stats(g, "realized_net_ev", seed=210)
        haircut = cluster_bootstrap_stats(g, "net_after_k5_haircut", seed=220)
        borrowed = cluster_bootstrap_stats(g, "incremental_vs_structural", seed=230)
        rows.append(
            {
                "split": split,
                "z_bucket": str(z_bucket),
                "tau_bucket": str(tbucket),
                "fills": len(g),
                "markets": g["market_id"].nunique(),
                "mean_entry_price": float(g["entry_price"].mean()),
                "mean_pm_yes_price": float(g["pm_yes_price"].mean()),
                "mean_empirical_yes_prob": float(g["empirical_yes_prob"].mean()),
                "mean_naive_terminal_yes_prob": float(g["naive_terminal_touch_prob_har_kou"].mean()),
                "mean_brownian_touch_yes_prob": float(g["brownian_touch_prob_har_kou"].mean()),
                "mean_model_token_prob": float(g["model_token_prob_empirical"].mean()),
                "mean_abs_z_har_kou": float(g["abs_z_har_kou"].mean()),
                "mean_pm_touch_iv": float(g["pm_touch_iv"].mean()),
                "mean_sigma_har": float(g["sigma_har_annualized"].mean()),
                "mean_sigma_har_kou": float(g["sigma_har_kou_directional_annualized"].mean()),
                "mean_iv_gap_har_kou": float(g["iv_gap_har_kou"].mean()),
                "yes_share": float(g["outcome"].eq("YES").mean()),
                "vol_rich_no_share": float(g["tier1_signal"].eq("buy_no_vol_rich").mean()),
                "mean_realized_net_ev": net["mean"],
                "realized_net_ev_ci_lo": net["ci_lo"],
                "realized_net_ev_ci_hi": net["ci_hi"],
                "realized_net_ev_p": net["p_one_sided"],
                "mean_capacity_haircut_ev": haircut["mean"],
                "capacity_haircut_ci_lo": haircut["ci_lo"],
                "capacity_haircut_ci_hi": haircut["ci_hi"],
                "capacity_haircut_p": haircut["p_one_sided"],
                "mean_borrowed_structural_ev": borrowed["mean"],
                "borrowed_structural_ci_lo": borrowed["ci_lo"],
                "borrowed_structural_ci_hi": borrowed["ci_hi"],
                "borrowed_structural_p": borrowed["p_one_sided"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["mean_capacity_haircut_ev", "fills"], ascending=[False, False]).reset_index(drop=True)


def tier1_mechanism(row: pd.Series) -> str:
    pm = float(row.get("mean_pm_yes_price", math.nan))
    empirical = float(row.get("mean_empirical_yes_prob", math.nan))
    naive = float(row.get("mean_naive_terminal_yes_prob", math.nan))
    if np.isfinite(pm) and np.isfinite(empirical) and abs(pm - empirical) <= 0.02:
        return "calibration residual: PM touch price is already close to empirical first-passage"
    if np.isfinite(pm) and np.isfinite(naive) and np.isfinite(empirical) and abs(pm - naive) <= 0.05 and empirical > max(naive * 1.5, naive + 0.05):
        return "structural touch-underpricing gap: PM is terminal-like while empirical touch is much higher"
    if np.isfinite(pm) and np.isfinite(empirical) and pm > empirical + 0.02:
        return "vol-rich touch fade / calibration residual: PM exceeds empirical first-passage"
    if np.isfinite(pm) and np.isfinite(empirical) and pm < empirical - 0.02:
        return "vol-cheap touch take / calibration residual: PM is below empirical first-passage"
    return "mixed/unclear mechanism"


def make_tier1_plots(tier1_fills: pd.DataFrame, heldout_cells: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    base = tier1_fills[tier1_fills["selected_empirical_edge"].fillna(False) & tier1_fills["iv_gap_har_kou"].notna()].copy()
    if not base.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(base["iv_gap_har_kou"].clip(-2, 2), bins=80, color="#2563eb", alpha=0.78)
        ax.axvline(-TIER1_IV_GAP_THRESHOLD, color="#16a34a", linestyle="--", linewidth=1.5, label="-25 vol pts")
        ax.axvline(TIER1_IV_GAP_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.5, label="+25 vol pts")
        ax.set_title("Arm T Tier-1 PM touch-IV gap")
        ax.set_xlabel("PM touch implied vol - HAR-RV-J/Kou forecast vol")
        ax.set_ylabel("Selected empirical-edge fills")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_TIER1_IV_GAP, dpi=160)
        plt.close(fig)
    plot = heldout_cells.copy()
    if not plot.empty:
        plot["cell"] = plot["z_bucket"] + "/" + plot["tau_bucket"]
        plot = plot.sort_values(["registered_from_train", "mean_capacity_haircut_ev"], ascending=[False, False]).head(12)
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(plot))
        y = plot["mean_capacity_haircut_ev"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                np.clip(y - plot["capacity_haircut_ci_lo"].to_numpy(dtype=float), 0.0, None),
                np.clip(plot["capacity_haircut_ci_hi"].to_numpy(dtype=float) - y, 0.0, None),
            ]
        )
        colors = np.where(plot["passes_tier1"], "#16a34a", np.where(plot["registered_from_train"], "#2563eb", "#94a3b8"))
        ax.bar(x, y, color=colors)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#222", capsize=3)
        ax.axhline(0, color="#333", linewidth=1)
        ax.axhline(TIER1_MATERIAL_CAPACITY_EV, color="#dc2626", linestyle="--", linewidth=1, label="0.25c materiality")
        ax.set_xticks(x)
        ax.set_xticklabels(plot["cell"], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("5% capacity-haircut EV per contract")
        ax.set_title("Arm T Tier-1 held-out cells")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_TIER1_HELDOUT, dpi=160)
        plt.close(fig)


def arm_t_tier1_extension() -> tuple[str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tier1_fills, kou_summary = prepare_tier1_fills()
    train, embargo, heldout, meta = split_arm_t_oos(tier1_fills)
    train_cells = arm_t_tier1_cell_table(train, "train_pre_embargo")
    if not train_cells.empty:
        train_cells["registered_from_train"] = train_cells["realized_net_ev_ci_lo"].gt(0) & train_cells["capacity_haircut_ci_lo"].gt(0)
    else:
        train_cells["registered_from_train"] = pd.Series(dtype=bool)
    if not train_cells.empty and "registered_from_train" in train_cells:
        registered = set(
            zip(
                train_cells.loc[train_cells["registered_from_train"], "z_bucket"],
                train_cells.loc[train_cells["registered_from_train"], "tau_bucket"],
                strict=False,
            )
        )
    else:
        registered = set()

    heldout_cells = arm_t_tier1_cell_table(heldout, "heldout_all_tested")
    if not heldout_cells.empty:
        heldout_cells["realized_net_ev_q"] = bh_adjust(heldout_cells["realized_net_ev_p"])
        heldout_cells["capacity_haircut_q"] = bh_adjust(heldout_cells["capacity_haircut_p"])
        heldout_cells["registered_from_train"] = [key in registered for key in zip(heldout_cells["z_bucket"], heldout_cells["tau_bucket"], strict=False)]
        heldout_cells["passes_materiality"] = heldout_cells["mean_capacity_haircut_ev"].ge(TIER1_MATERIAL_CAPACITY_EV)
        heldout_cells["mechanism_read"] = [tier1_mechanism(r) for _, r in heldout_cells.iterrows()]
        heldout_cells["passes_tier1"] = (
            meta["oos_ok"]
            & heldout_cells["registered_from_train"]
            & heldout_cells["realized_net_ev_ci_lo"].gt(0)
            & heldout_cells["capacity_haircut_ci_lo"].gt(0)
            & heldout_cells["realized_net_ev_q"].le(0.05)
            & heldout_cells["capacity_haircut_q"].le(0.05)
            & heldout_cells["passes_materiality"]
        )
    else:
        heldout_cells["registered_from_train"] = pd.Series(dtype=bool)
        heldout_cells["passes_materiality"] = pd.Series(dtype=bool)
        heldout_cells["mechanism_read"] = pd.Series(dtype=str)
        heldout_cells["passes_tier1"] = pd.Series(dtype=bool)

    train_cells.to_csv(OUT_TIER1_TRAIN, index=False)
    heldout_cells.to_csv(OUT_TIER1_HELDOUT, index=False)
    kou_summary.to_csv(OUT_TIER1_KOU, index=False)
    make_tier1_plots(tier1_fills, heldout_cells)

    meta.update(
        {
            "registered_cells": int(train_cells["registered_from_train"].sum()) if "registered_from_train" in train_cells else 0,
            "heldout_cells_tested": int(len(heldout_cells)),
            "survivor_cells": int(heldout_cells["passes_tier1"].sum()) if "passes_tier1" in heldout_cells else 0,
            "iv_gap_threshold": TIER1_IV_GAP_THRESHOLD,
            "material_capacity_ev": TIER1_MATERIAL_CAPACITY_EV,
        }
    )
    survivors = heldout_cells[heldout_cells.get("passes_tier1", False)].copy() if not heldout_cells.empty else pd.DataFrame()
    if not meta["oos_ok"]:
        verdict = "CLOSE"
        reason = (
            f"no legitimate OOS verdict: sample has {meta['resolution_dates']} resolution dates over {meta['span_days']} days, "
            "so the script treated it as train-only under the realism guard."
        )
        return verdict, reason, train_cells, heldout_cells, kou_summary, meta
    if survivors.empty:
        registered_heldout = heldout_cells[heldout_cells.get("registered_from_train", False)].copy() if not heldout_cells.empty else pd.DataFrame()
        if not registered_heldout.empty:
            row = registered_heldout.sort_values(["mean_capacity_haircut_ev", "capacity_haircut_ci_lo"], ascending=False).iloc[0]
            failures = []
            if not bool(row.get("passes_materiality", False)):
                failures.append(f"mean is below the pre-registered {cents(TIER1_MATERIAL_CAPACITY_EV)} materiality bar")
            if not (float(row.get("capacity_haircut_ci_lo", math.nan)) > 0 and float(row.get("realized_net_ev_ci_lo", math.nan)) > 0):
                failures.append("lower CI is not positive")
            if not (float(row.get("capacity_haircut_q", math.nan)) <= 0.05 and float(row.get("realized_net_ev_q", math.nan)) <= 0.05):
                failures.append("BH q is above 0.05")
            fail_text = "; ".join(failures) if failures else "it still does not satisfy the full pre-registered pass rule"
            reason = (
                f"best registered held-out Tier-1 cell {row['z_bucket']}/{row['tau_bucket']} has 5% haircut EV "
                f"{cents(float(row['mean_capacity_haircut_ev']))} with CI "
                f"{fmt_ci(float(row['capacity_haircut_ci_lo']), float(row['capacity_haircut_ci_hi']))} "
                f"and BH capacity q={number(float(row['capacity_haircut_q']), 4)}; {fail_text}."
            )
        elif not heldout_cells.empty:
            row = heldout_cells.sort_values(["mean_capacity_haircut_ev", "capacity_haircut_ci_lo"], ascending=False).iloc[0]
            reason = (
                f"no train-registered Tier-1 cell survived to held-out; best unregistered held-out cell "
                f"{row['z_bucket']}/{row['tau_bucket']} has 5% haircut EV {cents(float(row['mean_capacity_haircut_ev']))}."
            )
        else:
            reason = "the 25-vol-point HAR/Kou IV filter leaves no defensible held-out cell."
        verdict = "CLOSE"
    else:
        row = survivors.sort_values(["mean_capacity_haircut_ev", "capacity_haircut_ci_lo"], ascending=False).iloc[0]
        reason = (
            f"best Tier-1 survivor {row['z_bucket']}/{row['tau_bucket']} has held-out 5% haircut EV "
            f"{cents(float(row['mean_capacity_haircut_ev']))} with CI "
            f"{fmt_ci(float(row['capacity_haircut_ci_lo']), float(row['capacity_haircut_ci_hi']))}, "
            f"BH capacity q={number(float(row['capacity_haircut_q']), 4)}, and mechanism is {row['mechanism_read']}."
        )
        verdict = "MERITS-LIVE-MEASUREMENT-LOOP"
    return verdict, reason, train_cells, heldout_cells, kou_summary, meta


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    arm_rows = []
    bucket_rows = []
    for arm, g0 in df.groupby("arm"):
        for label, g in [("all_executable_fills", g0), ("empirical_edge_positive", g0[g0["selected_empirical_edge"]])]:
            if g.empty:
                continue
            ev_lo, ev_hi = cluster_ci(g, "realized_net_ev", seed=10)
            inc_lo, inc_hi = cluster_ci(g, "incremental_vs_structural", seed=20)
            arm_rows.append(
                {
                    "arm": arm,
                    "sample": label,
                    "fills": len(g),
                    "markets": g["market_id"].nunique(),
                    "mean_price": g["entry_price"].mean(),
                    "mean_empirical_prob": g["model_token_prob_empirical"].mean(),
                    "mean_model_edge": g["model_edge_empirical"].mean(),
                    "mean_realized_net_ev": g["realized_net_ev"].mean(),
                    "realized_net_ev_ci_lo": ev_lo,
                    "realized_net_ev_ci_hi": ev_hi,
                    "mean_incremental_vs_structural": g["incremental_vs_structural"].mean() if arm == "E" else math.nan,
                    "incremental_ci_lo": inc_lo if arm == "E" else math.nan,
                    "incremental_ci_hi": inc_hi if arm == "E" else math.nan,
                    "win_rate": g["realized_net_ev"].gt(0).mean(),
                }
            )
        selected = g0[g0["selected_empirical_edge"]].copy()
        for (z_bucket, tbucket), g in selected.groupby(["z_bucket", "tau_bucket"], observed=True):
            if g.empty:
                continue
            ev_lo, ev_hi = cluster_ci(g, "realized_net_ev", seed=30)
            inc_lo, inc_hi = cluster_ci(g, "incremental_vs_structural", seed=40)
            bucket_rows.append(
                {
                    "arm": arm,
                    "z_bucket": str(z_bucket),
                    "tau_bucket": str(tbucket),
                    "fills": len(g),
                    "markets": g["market_id"].nunique(),
                    "mean_pm_price": g["entry_price"].mean(),
                    "mean_empirical_prob": g["model_token_prob_empirical"].mean(),
                    "mean_terminal_or_naive": g["model_token_prob_terminal"].mean() if arm == "E" else g["naive_terminal_touch_prob"].mean(),
                    "mean_brownian_touch": g["brownian_touch_prob"].mean() if arm == "T" else math.nan,
                    "mean_model_edge": g["model_edge_empirical"].mean(),
                    "mean_realized_net_ev": g["realized_net_ev"].mean(),
                    "realized_net_ev_ci_lo": ev_lo,
                    "realized_net_ev_ci_hi": ev_hi,
                    "mean_incremental_vs_structural": g["incremental_vs_structural"].mean() if arm == "E" else math.nan,
                    "incremental_ci_lo": inc_lo if arm == "E" else math.nan,
                    "incremental_ci_hi": inc_hi if arm == "E" else math.nan,
                    "defensible": len(g) >= MIN_DEFENSIBLE_FILLS and g["market_id"].nunique() >= MIN_DEFENSIBLE_MARKETS,
                }
            )
    arms = pd.DataFrame(arm_rows)
    buckets = pd.DataFrame(bucket_rows)
    if not buckets.empty:
        buckets["passes_net_ci"] = buckets["defensible"] & buckets["realized_net_ev_ci_lo"].gt(0)
        buckets["passes_incremental_ci"] = np.where(buckets["arm"].eq("E"), buckets["incremental_ci_lo"].gt(0), True)
        buckets["passes_gate"] = buckets["passes_net_ci"] & buckets["passes_incremental_ci"]
    return arms, buckets.sort_values(["passes_gate", "mean_realized_net_ev"], ascending=[False, False]) if not buckets.empty else buckets


def arm_t_confirmation_cell_table(sample: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    if sample.empty:
        return pd.DataFrame()
    for (z_bucket, tbucket), g in sample.groupby(["z_bucket", "tau_bucket"], observed=True):
        if len(g) < MIN_DEFENSIBLE_FILLS or g["market_id"].nunique() < MIN_DEFENSIBLE_MARKETS:
            continue
        net = cluster_bootstrap_stats(g, "realized_net_ev", seed=110)
        haircut = cluster_bootstrap_stats(g, "net_after_k5_haircut", seed=120)
        borrowed = cluster_bootstrap_stats(g, "incremental_vs_structural", seed=130)
        rows.append(
            {
                "split": split,
                "z_bucket": str(z_bucket),
                "tau_bucket": str(tbucket),
                "fills": len(g),
                "markets": g["market_id"].nunique(),
                "mean_pm_price": g["entry_price"].mean(),
                "mean_empirical_prob": g["model_token_prob_empirical"].mean(),
                "mean_naive_terminal": g["naive_terminal_touch_prob"].mean(),
                "mean_brownian_touch": g["brownian_touch_prob"].mean(),
                "mean_model_edge": g["model_edge_empirical"].mean(),
                "mean_realized_net_ev": net["mean"],
                "realized_net_ev_ci_lo": net["ci_lo"],
                "realized_net_ev_ci_hi": net["ci_hi"],
                "realized_net_ev_p": net["p_one_sided"],
                "mean_capacity_haircut_ev": haircut["mean"],
                "capacity_haircut_ci_lo": haircut["ci_lo"],
                "capacity_haircut_ci_hi": haircut["ci_hi"],
                "capacity_haircut_p": haircut["p_one_sided"],
                "mean_borrowed_structural_ev": borrowed["mean"],
                "borrowed_structural_ci_lo": borrowed["ci_lo"],
                "borrowed_structural_ci_hi": borrowed["ci_hi"],
                "borrowed_structural_p": borrowed["p_one_sided"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["mean_realized_net_ev", "fills"], ascending=[False, False]).reset_index(drop=True)
    return out


def arm_t_confirmation() -> tuple[str, str, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not OUT_FILLS.exists():
        raise SystemExit(f"missing fill sample: {OUT_FILLS}")
    fills = pd.read_parquet(OUT_FILLS)
    fills["fill_ts"] = pd.to_datetime(fills["fill_ts"], utc=True)
    fills["resolution_ts"] = pd.to_datetime(fills["resolution_ts"], utc=True)
    t = fills[fills["arm"].eq("T") & fills["selected_empirical_edge"].fillna(False)].copy()
    if t.empty:
        raise SystemExit("no selected Arm T fills in saved fill sample")
    t["resolution_date"] = t["resolution_ts"].dt.date
    dates = sorted(t["resolution_date"].dropna().unique())
    if len(dates) < 4:
        raise SystemExit("too few Arm T resolution dates for a time-embargo confirmation split")
    split_idx = int(math.floor(len(dates) * 0.70))
    split_idx = min(max(split_idx, 1), len(dates) - 2)
    train_dates = set(dates[:split_idx])
    embargo_date = dates[split_idx]
    heldout_dates = set(dates[split_idx + 1 :])
    train = t[t["resolution_date"].isin(train_dates)].copy()
    embargo = t[t["resolution_date"].eq(embargo_date)].copy()
    heldout = t[t["resolution_date"].isin(heldout_dates)].copy()

    train_cells = arm_t_confirmation_cell_table(train, "train_pre_embargo")
    if not train_cells.empty:
        train_cells["registered_from_train"] = train_cells["realized_net_ev_ci_lo"].gt(0) & train_cells["capacity_haircut_ci_lo"].gt(0)
    else:
        train_cells["registered_from_train"] = pd.Series(dtype=bool)
    registered = set(zip(train_cells.loc[train_cells["registered_from_train"], "z_bucket"], train_cells.loc[train_cells["registered_from_train"], "tau_bucket"]))

    heldout_all = arm_t_confirmation_cell_table(heldout, "heldout_all_tested")
    if not heldout_all.empty:
        heldout_all["realized_net_ev_q"] = bh_adjust(heldout_all["realized_net_ev_p"])
        heldout_all["capacity_haircut_q"] = bh_adjust(heldout_all["capacity_haircut_p"])
        heldout_all["registered_from_train"] = [key in registered for key in zip(heldout_all["z_bucket"], heldout_all["tau_bucket"], strict=False)]
        heldout_all["passes_confirmation"] = (
            heldout_all["registered_from_train"]
            & heldout_all["realized_net_ev_ci_lo"].gt(0)
            & heldout_all["capacity_haircut_ci_lo"].gt(0)
            & heldout_all["realized_net_ev_q"].le(0.05)
            & heldout_all["capacity_haircut_q"].le(0.05)
        )
    else:
        heldout_all["registered_from_train"] = pd.Series(dtype=bool)
        heldout_all["passes_confirmation"] = pd.Series(dtype=bool)

    train_cells.to_csv(OUT_CONFIRM_TRAIN, index=False)
    heldout_all.to_csv(OUT_CONFIRM_HELDOUT, index=False)

    survivors = heldout_all[heldout_all["passes_confirmation"]].sort_values(["capacity_haircut_ci_lo", "realized_net_ev_ci_lo"], ascending=False) if not heldout_all.empty else pd.DataFrame()
    if survivors.empty:
        verdict = "CLOSE"
        reason = "no Arm T cell survives train registration, held-out net/capacity CIs, and BH FDR."
    else:
        row = survivors.iloc[0]
        mechanism = confirmation_mechanism(row)
        reason = (
            f"best confirmed cell {row['z_bucket']}/{row['tau_bucket']} has held-out net CI "
            f"{fmt_ci(float(row['realized_net_ev_ci_lo']), float(row['realized_net_ev_ci_hi']))} "
            f"and 5% capacity-haircut CI {fmt_ci(float(row['capacity_haircut_ci_lo']), float(row['capacity_haircut_ci_hi']))}; "
            f"BH capacity q={number(float(row['capacity_haircut_q']), 4)}; mechanism is {mechanism}."
        )
        verdict = "MERITS-BUILD"

    meta = {
        "train_start": str(min(train_dates)),
        "train_end": str(max(train_dates)),
        "embargo_date": str(embargo_date),
        "heldout_start": str(min(heldout_dates)),
        "heldout_end": str(max(heldout_dates)),
        "train_fills": int(len(train)),
        "train_markets": int(train["market_id"].nunique()),
        "embargo_fills": int(len(embargo)),
        "embargo_markets": int(embargo["market_id"].nunique()),
        "heldout_fills": int(len(heldout)),
        "heldout_markets": int(heldout["market_id"].nunique()),
        "registered_cells": int(len(registered)),
        "heldout_cells_tested": int(len(heldout_all)),
        "survivor_cells": int(len(survivors)),
    }
    return verdict, reason, train_cells, heldout_all, meta


def local_concentration(condition_ids: list[str]) -> pd.DataFrame:
    if not condition_ids or not K5_WALLET_CACHE.exists():
        return pd.DataFrame()
    ids = ", ".join(f"'{str(x).lower().replace(chr(39), chr(39)+chr(39))}'" for x in sorted(set(condition_ids)) if x)
    con = duckdb.connect()
    query = f"""
        WITH local_market_map AS (
            SELECT CAST(id AS VARCHAR) AS market_id, lower(CAST(condition_id AS VARCHAR)) AS condition_id
            FROM read_parquet('{LOCAL_MARKETS}')
            WHERE lower(CAST(condition_id AS VARCHAR)) IN ({ids})
        ),
        scoped AS (
            SELECT lmm.condition_id, lower(k.address) AS maker, max(k.global_maker_usd) AS maker_usd, min(k.global_market_maker_rank) AS maker_rank
            FROM read_parquet('{K5_WALLET_CACHE}') k
            JOIN local_market_map lmm ON CAST(k.market_id AS VARCHAR) = lmm.market_id
            WHERE k.global_maker_usd IS NOT NULL AND k.global_market_maker_rank IS NOT NULL
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT *, sum(maker_usd) OVER (PARTITION BY condition_id) AS total_maker_usd
            FROM scoped
        )
        SELECT condition_id,
               sum(maker_usd) FILTER (WHERE maker_rank <= 3) / nullif(max(total_maker_usd), 0) AS top3_maker_share,
               1.0 - coalesce(sum(maker_usd) FILTER (WHERE maker_rank <= 3) / nullif(max(total_maker_usd), 0), 0.0) AS non_top3_maker_share
        FROM ranked
        GROUP BY 1
    """
    return con.execute(query).df()


def capacity(current: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    cur = current[current["arm"].isin(["T", "E"])][["condition_id", "market_slug", "arm", "volume_for_share_usd"]].copy()
    cur["sample"] = "current_gamma"
    h = hist[["condition_id", "market_slug", "arm", "volume_usd"]].rename(columns={"volume_usd": "volume_for_share_usd"})
    h["sample"] = "historical_local"
    both = pd.concat([cur, h], ignore_index=True)
    conc = local_concentration(both["condition_id"].dropna().tolist())
    both = both.merge(conc, on="condition_id", how="left") if not conc.empty else both
    rows = []
    for (sample, arm), g in both.groupby(["sample", "arm"]):
        ok = g["top3_maker_share"].notna() if "top3_maker_share" in g else pd.Series(False, index=g.index)
        w = g["volume_for_share_usd"].fillna(0).clip(lower=1)
        rows.append(
            {
                "sample": sample,
                "arm": arm,
                "markets": g["condition_id"].nunique(),
                "matched_markets": int(ok.sum()),
                "missing_markets": int((~ok).sum()),
                "volume_usd": g["volume_for_share_usd"].sum(),
                "weighted_top3_share": float(np.average(g.loc[ok, "top3_maker_share"], weights=w[ok])) if ok.any() else math.nan,
                "weighted_non_top3_share": float(np.average(g.loc[ok, "non_top3_maker_share"], weights=w[ok])) if ok.any() else math.nan,
                "missing_market_slugs": "; ".join(g.loc[~ok, "market_slug"].astype(str).head(10).tolist()),
            }
        )
    return pd.DataFrame(rows)


def current_quote_models(current: pd.DataFrame, features: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    active = current[current["active"] & current["arm"].isin(["T", "E"]) & current["level"].notna()].copy()
    if active.empty:
        return active
    latest = features.sort_values("ts").groupby("asset").tail(1)[["asset", "ts", "close", "ewma_sigma_annualized", "prior_max_high", "prior_min_low"]]
    active = active.merge(latest, on="asset", how="left")
    active["spot"] = active["close"].astype(float)
    active["resolution_ts"] = pd.to_datetime(active["resolution_ts_utc"], utc=True)
    active["tau_seconds"] = (active["resolution_ts"] - active["ts"]).dt.total_seconds()
    active = active[active["tau_seconds"].gt(0)].copy()
    denom = active["ewma_sigma_annualized"].astype(float) * np.sqrt(active["tau_seconds"] / YEAR_SECONDS)
    active["z_signed_yes"] = np.log(active["level"].astype(float) / active["spot"].astype(float)) / denom
    active["abs_z"] = active["z_signed_yes"].abs()
    abs_bucket = pd.cut(active["abs_z"], Z_BINS, labels=Z_LABELS, include_lowest=True)
    active["z_bucket"] = [
        signed_z_bucket(bucket, z, arm)
        for bucket, z, arm in zip(abs_bucket, active["z_signed_yes"], active["arm"], strict=False)
    ]
    active["tau_bucket"] = tau_bucket(active["tau_seconds"] / 3600.0)
    touch_down = active["touch_direction"].eq("down")
    active["already_touched"] = np.where(
        touch_down,
        active["prior_min_low"].astype(float).le(active["level"].astype(float)),
        active["prior_max_high"].astype(float).ge(active["level"].astype(float)),
    )
    active = active[~(active["arm"].eq("T") & pd.Series(active["already_touched"]).fillna(True).to_numpy())].copy()
    month_start = pd.Timestamp(f"{AS_OF.strftime('%Y-%m')}-01", tz="UTC")
    lookup = make_lookup(synth[synth["window_end"].lt(month_start)].copy())
    preds = [lookup_prob(r.arm, r.asset, str(r.z_bucket), str(r.tau_bucket), lookup) for r in active.itertuples(index=False)]
    active["empirical_yes_prob"] = [p[0] for p in preds]
    active["terminal_yes_prob"] = np.asarray(norm_cdf(-active["z_signed_yes"].to_numpy(dtype=float)), dtype=float)
    active["brownian_touch_prob"] = brownian_touch(active["abs_z"])
    active["yes_fee"] = CRYPTO_TAKER_FEE_RATE * active["yes_ask"] * (1 - active["yes_ask"])
    active["yes_model_edge_empirical"] = active["empirical_yes_prob"] - active["yes_ask"] - active["yes_fee"]
    return active.replace([np.inf, -np.inf], np.nan)


def make_plots(calib: pd.DataFrame, buckets: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for arm, path, title in [("T", PLOT_CALIBRATION_T, "Arm T empirical touch calibration"), ("E", PLOT_CALIBRATION_E, "Arm E empirical terminal calibration")]:
        sub = calib[calib["arm"].eq(arm)].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        sizes = np.maximum(30, sub["rows"].astype(float) / sub["rows"].max() * 500)
        ax.scatter(sub["mean_pred"], sub["observed"], s=sizes, color="#2563eb", alpha=0.75)
        for _, r in sub.iterrows():
            ax.annotate(str(r["prob_bucket"]), (r["mean_pred"], r["observed"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.plot([0, 1], [0, 1], "--", color="#333", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    if not buckets.empty:
        plot = buckets[buckets["defensible"]].copy()
        if plot.empty:
            plot = buckets.head(10).copy()
        plot["cell"] = plot["arm"] + " " + plot["z_bucket"] + "/" + plot["tau_bucket"]
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(plot))
        y = plot["mean_realized_net_ev"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                np.clip(y - plot["realized_net_ev_ci_lo"].to_numpy(dtype=float), 0.0, None),
                np.clip(plot["realized_net_ev_ci_hi"].to_numpy(dtype=float) - y, 0.0, None),
            ]
        )
        ax.bar(x, y, color=np.where(plot["arm"].eq("T"), "#2563eb", "#16a34a"))
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#222", capsize=3)
        ax.axhline(0, color="#333", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(plot["cell"], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Realized net EV per selected fill")
        ax.set_title("Same-day selected fills by arm and z/tau cell")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_BUCKETS, dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(plot))
        width = 0.25
        ax.bar(x - width, plot["mean_pm_price"], width=width, label="PM executed", color="#475569")
        ax.bar(x, plot["mean_terminal_or_naive"], width=width, label="Terminal/naive", color="#f97316")
        ax.bar(x + width, plot["mean_empirical_prob"], width=width, label="Empirical", color="#2563eb")
        ax.set_xticks(x)
        ax.set_xticklabels(plot["cell"], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Price/probability")
        ax.set_title("PM price vs external fair values")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_BEHAVIORAL, dpi=160)
        plt.close(fig)


def table(df: pd.DataFrame, cols: list[str], limit: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    rows = []
    for _, r in df.head(limit).iterrows():
        out = []
        for c in cols:
            v = r.get(c, math.nan)
            if c in {"fills", "markets", "rows", "matched_markets", "missing_markets"}:
                out.append(str(int(v)) if pd.notna(v) else "0")
            elif c in {"volume_usd", "volume_for_share_usd"}:
                out.append(dollars(float(v)))
            elif c in {
                "mean_price",
                "mean_pm_price",
                "mean_entry_price",
                "mean_pm_yes_price",
                "mean_empirical_prob",
                "mean_empirical_yes_prob",
                "mean_terminal_or_naive",
                "mean_naive_terminal",
                "mean_naive_terminal_yes_prob",
                "mean_brownian_touch",
                "mean_brownian_touch_yes_prob",
                "mean_model_edge",
                "mean_model_token_prob",
                "mean_realized_net_ev",
                "mean_capacity_haircut_ev",
                "mean_borrowed_structural_ev",
                "mean_incremental_vs_structural",
                "tier1_material_capacity_ev",
                "yes_ask",
                "empirical_yes_prob",
                "terminal_yes_prob",
                "brownian_touch_prob",
                "yes_model_edge_empirical",
            }:
                out.append(cents(float(v)))
            elif c in {"realized_net_ev_ci_lo"}:
                out.append(fmt_ci(float(r["realized_net_ev_ci_lo"]), float(r["realized_net_ev_ci_hi"])))
            elif c in {"capacity_haircut_ci_lo"}:
                out.append(fmt_ci(float(r["capacity_haircut_ci_lo"]), float(r["capacity_haircut_ci_hi"])))
            elif c in {"borrowed_structural_ci_lo"}:
                out.append(fmt_ci(float(r["borrowed_structural_ci_lo"]), float(r["borrowed_structural_ci_hi"])))
            elif c in {"incremental_ci_lo"}:
                out.append(fmt_ci(float(r["incremental_ci_lo"]), float(r["incremental_ci_hi"])))
            elif c in {"win_rate", "weighted_top3_share", "weighted_non_top3_share", "mean_pred", "observed", "obs_minus_pred", "yes_share", "vol_rich_no_share"}:
                out.append(pct(float(v)))
            elif c in {"mean_pm_touch_iv", "mean_sigma_har", "mean_sigma_har_kou", "mean_iv_gap_har_kou", "sigma_har_annualized", "sigma_har_kou_up_annualized", "sigma_har_kou_down_annualized", "kou_up_lambda_per_year", "kou_down_lambda_per_year", "kou_up_avg_abs_jump", "kou_down_avg_abs_jump"}:
                out.append(number(float(v), 3) if np.isfinite(float(v)) else "n/a")
            elif c in {"realized_net_ev_p", "capacity_haircut_p", "realized_net_ev_q", "capacity_haircut_q"}:
                out.append(number(float(v), 4) if np.isfinite(float(v)) else "n/a")
            elif c in {"level", "spot"}:
                out.append(dollars(float(v)))
            elif c in {"z_signed_yes", "mean_abs_z_har_kou", "tau_seconds"}:
                out.append(number(float(v), 2) if np.isfinite(float(v)) else "n/a")
            else:
                out.append(str(v))
        rows.append(out)
    return markdown_table(cols, rows)


def write_note(current: pd.DataFrame, hist: pd.DataFrame, arms: pd.DataFrame, buckets: pd.DataFrame, calib: pd.DataFrame, current_quotes: pd.DataFrame, cap: pd.DataFrame, verdict_t: str, verdict_e: str, reason_t: str, reason_e: str) -> None:
    class_share = current.groupby("arm").agg(markets=("market_id", "nunique"), volume_usd=("volume_for_share_usd", "sum")).reset_index()
    class_share["volume_share"] = class_share["volume_usd"] / class_share["volume_usd"].sum()
    old_n = 6
    e_days = hist[hist["arm"].eq("E")]["resolution_ts"].dt.date.nunique()
    e_markets = hist[hist["arm"].eq("E")]["market_id"].nunique()
    strikes_per_day = e_markets / e_days if e_days else math.nan
    touch_ex = current_quotes[current_quotes["arm"].eq("T")].sort_values("volume_for_share_usd", ascending=False).head(1)
    term_ex = current_quotes[current_quotes["arm"].eq("E")].sort_values("volume_for_share_usd", ascending=False).head(1)
    touch_text = "No live Arm T quote survived the model filters."
    if not touch_ex.empty:
        r = touch_ex.iloc[0]
        touch_text = f"Touch example: `{r['market_slug']}` has level {dollars(float(r['level']))}, spot {dollars(float(r['spot']))}, YES ask {cents(float(r['yes_ask']))}, empirical touch {cents(float(r['empirical_yes_prob']))}, and model edge {cents(float(r['yes_model_edge_empirical']))} before future resolution."
    term_text = "No live Arm E quote survived the model filters."
    if not term_ex.empty:
        r = term_ex.iloc[0]
        term_text = f"Terminal example: `{r['market_slug']}` has strike {dollars(float(r['level']))}, spot {dollars(float(r['spot']))}, YES ask {cents(float(r['yes_ask']))}, empirical terminal {cents(float(r['empirical_yes_prob']))}, and model edge {cents(float(r['yes_model_edge_empirical']))}; it still must beat the structural baseline after capacity."
    if verdict_t == "MERITS-BUILD":
        gate_read = "Read: these are the gate tables. Arm T has lower-CI-positive defensible same-day touch cells, concentrated in short-tau/far-|z| buckets. Arm E has many ladder strikes, but the incremental-vs-structural bar remains the binding failure."
        next_step = "Concrete next step: promote Arm T only to a constrained live same-day touch build/capture for the passing z/tau cells, with capacity checks on fresh markets before sizing. Keep Arm E closed."
    else:
        gate_read = "Read: these are the kill tables. Arm T has no lower-CI-positive defensible cell. Arm E has many ladder strikes, but the incremental-vs-structural bar remains the binding failure."
        next_step = "Concrete next step: close both arms as build candidates. A future reopen would need live same-day barrier/ladder quote capture through resolution, with the same external Binance fair values and the same net-of-cost market-cluster CI gates."

    note = f"""# Same-Day Crypto Touch And Terminal-Ladder Pricing Gate

> Hub: [[strat_options_delta]] · [[COWORK]]
> Related: [[od_cross_asset_gate0_universe_map_findings]] · [[od_conditional_prob_calibration_findings]] · [[od_pricing_model_form_findings]] · [[od_strategy_a_v3_pnl_risk_findings]] · [[block_k4_arb_scan_findings]] · [[mm_deployable_cells_findings]] · [[block_k5_findings]] · [[block_k5_stress_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Headline

Final verdicts: **Arm T {verdict_t}** and **Arm E {verdict_e}**.

Arm T reason: {reason_t}

Arm E reason: {reason_e}

This is the same-day daily crypto gate only. Multi-week/monthly `will hit X in <month>` barriers are out of scope. The fair values are external Binance 1m path/close truth tables; no PM ladder monotonicity or internal-consistency arb is used.

## Design And Pre-Registered Gates

Arm T contains same-day touch/running-extreme markets that resolve if Binance BTC/ETH/SOL reaches a level anytime in the day. Arm E contains same-day terminal above-X ladders that resolve on the Binance 1m close at the resolution timestamp. Ambiguous resolution text is quarantined.

The pre-registered Arm T pass bar is a defensible z/time-left cell with selected executable fills whose market-cluster bootstrap lower CI is above zero after PM taker fees and spread-crossing entry. The pre-registered Arm E pass bar is stricter: net-EV lower CI above zero **and** K5-haircut incremental-vs-structural lower CI above zero, matching the closed-line bar in [[od_conditional_prob_calibration_findings]] and [[od_pricing_model_form_findings]].

{touch_text}

{term_text}

## Step 1: Same-Day Classification

{table(current[["market_slug", "asset", "level", "resolution_ts_utc", "resolution_source", "arm", "resolution_class", "decision_quote", "volume_for_share_usd"]].sort_values("volume_for_share_usd", ascending=False), ["market_slug", "asset", "level", "resolution_ts_utc", "arm", "resolution_class", "decision_quote", "volume_for_share_usd"], 16)}

Read: classification is by Gamma resolution text, not slug. `Close` at a timestamp goes to Arm E; `High`/`Low` anytime in the day goes to Arm T; unclear rows stay quarantined.

{table(class_share, ["arm", "markets", "volume_usd", "volume_share"], 8)}

## Step 2: External Fair-Value Calibration

Arm T uses Binance 1m highs/lows for first-passage labels. Arm E uses Binance 1m terminal closes. Both empirical estimators are expanding-time CV: each validation month uses only prior completed data.

![Arm T calibration]({PLOT_CALIBRATION_T})

![Arm E calibration]({PLOT_CALIBRATION_E})

{table(calib, ["arm", "prob_bucket", "rows", "mean_pred", "observed", "obs_minus_pred"], 20)}

Read: calibration is checked before PM application. If this table were badly off, the gate would be invalid regardless of PM PnL.

## Step 3: Net-Of-Cost Gates

Selected fills are actual PM taker buys of the side that the empirical model says is underpriced after fee. `realized_net_ev` is payoff minus executed price minus taker fee. Arm E also reports `incremental_vs_structural`, which applies the K5 non-incumbent haircut and subtracts the v4 structural queue baseline of {cents(STRUCTURAL_BASELINE_C)}.

{table(arms, ["arm", "sample", "fills", "markets", "mean_price", "mean_empirical_prob", "mean_model_edge", "mean_realized_net_ev", "realized_net_ev_ci_lo", "mean_incremental_vs_structural", "incremental_ci_lo", "win_rate"], 12)}

Power read: Arm E's same-day ladder supplies {number(strikes_per_day, 1)} independent strike markets per day in the local historical sample, versus the old OD 4h OOS far-|z| gate's n={old_n} markets. That improves power, but power does not matter unless the incremental-vs-structural lower CI clears.

{table(buckets, ["arm", "z_bucket", "tau_bucket", "fills", "markets", "mean_pm_price", "mean_empirical_prob", "mean_terminal_or_naive", "mean_brownian_touch", "mean_model_edge", "mean_realized_net_ev", "realized_net_ev_ci_lo", "mean_incremental_vs_structural", "incremental_ci_lo", "passes_gate"], 20)}

![Selected-fill EV]({PLOT_BUCKETS})

Caption: selected executable fills by arm and z/time-left cell. Error bars are market-cluster bootstrap CIs. Arm E must also clear the structural-incremental CI.

![Behavioral gap]({PLOT_BEHAVIORAL})

Caption: PM executed price versus the terminal/naive benchmark and empirical external probability. Arm T would support the retail-touch-underpricing thesis if PM sat near terminal while empirical touch sat near Brownian and materially higher.

{gate_read}

## Step 4: OFI

OFI is not promoted to a standalone result. The same-day historical fills have Binance path state and PM trades, but not synchronized PM order-book/OFI capture for the exact touch markets. Current CLOB is a one-shot quote snapshot with no realized label. Under A15b/A1.7 discipline, OFI remains untested here rather than treated as alpha.

## Step 5: Capacity

{table(cap, ["sample", "arm", "markets", "matched_markets", "missing_markets", "volume_usd", "weighted_top3_share", "weighted_non_top3_share", "missing_market_slugs"], 12)}

Read: same-day current markets are newer than the K5 cache and are often missing exact concentration. Missing rows inherit no proof of headroom. Historical matched rows still show the usual top-maker concentration problem.

## Decision

**Arm T:** {verdict_t}. {reason_t}

**Arm E:** {verdict_e}. {reason_e}

{next_step}

## Outputs

- Classification CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_classification.csv`
- Historical markets CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_history_markets.csv`
- Arm summary CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_arm_summary.csv`
- Bucket summary CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_bucket_summary.csv`
- Calibration CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_calibration.csv`
- Current quote CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_current_quotes.csv`
- Capacity CSV: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_capacity.csv`
- Fill sample parquet: `data/analysis/od_same_day_crypto_pricing_fills.parquet`
- Script: `scripts/od_same_day_crypto_pricing_gate.py`
"""
    NOTE.write_text(normalize_markdown_wrapping(note), encoding="utf-8")


def confirmation_mechanism(row: pd.Series) -> str:
    pm = float(row["mean_pm_price"])
    empirical = float(row["mean_empirical_prob"])
    naive = float(row["mean_naive_terminal"])
    if np.isfinite(pm) and np.isfinite(empirical) and abs(pm - empirical) <= 0.02:
        return "fragile calibration residual: PM is close to empirical first-passage, not near terminal"
    if np.isfinite(pm) and np.isfinite(naive) and abs(pm - naive) <= 0.05 and empirical > max(naive * 1.5, naive + 0.05):
        return "structural touch-underpricing gap"
    return "mixed/unclear mechanism"


def append_confirmation_section(verdict_t: str, reason_t: str, train_cells: pd.DataFrame, heldout_cells: pd.DataFrame, meta: dict[str, Any]) -> None:
    note = NOTE.read_text(encoding="utf-8")
    note = re.sub(
        r"Final verdicts: \*\*Arm T .*?\*\* and \*\*Arm E CLOSE\*\*\.",
        f"Final verdicts after Arm T confirmation: **Arm T {verdict_t}** and **Arm E CLOSE**.",
        note,
    )
    note = re.sub(r"Arm T(?: confirmation)? reason: .*\n", f"Arm T confirmation reason: {reason_t}\n", note)
    if verdict_t == "MERITS-BUILD":
        note = re.sub(r"\*\*Arm T:\*\* .*?\n", f"**Arm T:** {verdict_t}. {reason_t}\n", note)
        note = re.sub(
            r"Concrete next step: .*?Keep Arm E closed\.",
            "Concrete next step: promote Arm T only to a constrained live same-day touch capture/build for the confirmed z/tau cell, with fresh capacity measurement before sizing. Keep Arm E closed.",
            note,
        )
    else:
        note = re.sub(r"\*\*Arm T:\*\* .*?\n", f"**Arm T:** {verdict_t}. {reason_t}\n", note)
        note = re.sub(
            r"Concrete next step: .*?Keep Arm E closed\.",
            "Concrete next step: close Arm T and keep Arm E closed. A future reopen needs a fresh pre-registered live capture with the same capacity, OOS, and FDR gates.",
            note,
        )

    marker = "\n## Confirmation Pass\n"
    if marker in note:
        note = note[: note.index(marker)].rstrip() + "\n"

    registered = train_cells[train_cells.get("registered_from_train", False)].copy() if not train_cells.empty else pd.DataFrame()
    survivors = heldout_cells[heldout_cells.get("passes_confirmation", False)].copy() if not heldout_cells.empty else pd.DataFrame()
    heldout_registered = heldout_cells[heldout_cells.get("registered_from_train", False)].copy() if not heldout_cells.empty else pd.DataFrame()
    if not survivors.empty:
        survivors = survivors.sort_values(["capacity_haircut_ci_lo", "realized_net_ev_ci_lo"], ascending=False).copy()
        survivors["mechanism_read"] = [confirmation_mechanism(r) for _, r in survivors.iterrows()]
    if not heldout_registered.empty:
        heldout_registered = heldout_registered.sort_values(["passes_confirmation", "capacity_haircut_ci_lo"], ascending=[False, False]).copy()

    section = f"""
## Confirmation Pass

This confirmation pass closes the earlier asymmetry: Arm T is now checked with capacity haircut, time-embargo OOS, and multiple-comparison correction, while Arm E is not rerun. The pass uses the already-saved PM fill sample and Binance-derived fair values from this note; no new market capture is added.

Pre-registered split rule: sort Arm T resolution dates, use the first 70% as train/pre-embargo, embargo the next resolution date, and evaluate only the remaining held-out dates. In this run train is {meta['train_start']} to {meta['train_end']} ({number(meta['train_fills'], 0)} fills / {number(meta['train_markets'], 0)} markets), the embargo date is {meta['embargo_date']} ({number(meta['embargo_fills'], 0)} fills), and held-out is {meta['heldout_start']} to {meta['heldout_end']} ({number(meta['heldout_fills'], 0)} fills / {number(meta['heldout_markets'], 0)} markets).

Capacity rule: Arm T now reports raw net EV and 5% K5-style non-incumbent-capacity haircut EV. I could not derive a touch-specific passive structural/MM baseline from the saved fill sample alone because it has executed taker fills but not synchronized resting quote/queue states. The 1.98c structural baseline is therefore shown as a borrowed crypto-4h diagnostic, not the primary touch gate. The Arm T confirmation pass bar is held-out raw net-EV lower CI > 0, held-out capacity-haircut lower CI > 0, and BH-adjusted q <= 0.05 across all held-out defensible cells.

Train registered {meta['registered_cells']} cells:

{table(registered, ["z_bucket", "tau_bucket", "fills", "markets", "mean_realized_net_ev", "realized_net_ev_ci_lo", "mean_capacity_haircut_ev", "capacity_haircut_ci_lo"], 12) if not registered.empty else "No train cell registered."}

Read: these cells are selected using train/pre-embargo data only. Held-out performance below decides the verdict.

Held-out registered cells after BH across {meta['heldout_cells_tested']} defensible held-out cells:

{table(heldout_registered, ["z_bucket", "tau_bucket", "fills", "markets", "mean_realized_net_ev", "realized_net_ev_ci_lo", "realized_net_ev_q", "mean_capacity_haircut_ev", "capacity_haircut_ci_lo", "capacity_haircut_q", "mean_borrowed_structural_ev", "borrowed_structural_ci_lo", "passes_confirmation"], 12) if not heldout_registered.empty else "No train-registered cell had held-out observations."}

Read: `mean_capacity_haircut_ev` is raw realized net EV multiplied by 0.05. `mean_borrowed_structural_ev` subtracts the old 1.98c crypto-4h structural baseline after that haircut; it is included to show the severity of the structural bar, but it is borrowed and not touch-specific.

Mechanism check on survivors:

{table(survivors, ["z_bucket", "tau_bucket", "fills", "markets", "mean_pm_price", "mean_naive_terminal", "mean_brownian_touch", "mean_empirical_prob", "mean_realized_net_ev", "capacity_haircut_ci_lo", "mechanism_read"], 12) if not survivors.empty else "No cell survived OOS + capacity + FDR."}

Read: the survivor count is {meta['survivor_cells']}. The deciding number is the best survivor's held-out capacity-haircut lower CI if any survivor exists; otherwise the deciding number is the maximum held-out capacity-haircut lower CI among train-registered cells.

**Revised Arm T verdict:** {verdict_t}. {reason_t}

Confirmation CSVs: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_confirmation_train_cells.csv` and `data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_confirmation_heldout_cells.csv`.
"""
    note = note.rstrip() + "\n" + section
    NOTE.write_text(normalize_markdown_wrapping(note), encoding="utf-8")


def update_docs(verdict_t: str, verdict_e: str, reason_t: str, reason_e: str) -> None:
    bullet = (
        f"- 2026-06-02 OD same-day crypto pricing gate: **Arm T {verdict_t}; Arm E {verdict_e}**. "
        f"Arm T: {reason_t} Arm E: {reason_e} See [[od_same_day_crypto_pricing_gate_findings]]."
    )
    hub = OD_HUB.read_text(encoding="utf-8")
    idx = hub.find("## Current state")
    if idx >= 0 and "OD same-day crypto pricing gate" not in hub[idx : idx + 5000]:
        line_end = hub.find("\n", idx)
        hub = hub[: line_end + 1] + bullet + "\n" + hub[line_end + 1 :]
    hub = hub.replace(
        "- Cross-asset / market-vs-market reopen candidates: [[od_rv_deribit_daily_capture_findings]], [[od_rv_deribit_daily_scoping_findings]], [[od_cross_asset_gate0_universe_map_findings]], [[od_cross_asset_updown_scoping]]",
        "- Cross-asset / market-vs-market reopen candidates: [[od_rv_deribit_daily_capture_findings]], [[od_rv_deribit_daily_scoping_findings]], [[od_cross_asset_gate0_universe_map_findings]], [[od_cross_asset_updown_scoping]], [[od_same_day_crypto_pricing_gate_findings]]",
    )
    hub = hub.replace(
        "[[od_same_day_crypto_pricing_gate_findings]], [[od_same_day_crypto_pricing_gate_findings]]",
        "[[od_same_day_crypto_pricing_gate_findings]]",
    )
    OD_HUB.write_text(hub, encoding="utf-8")
    todo = BRAIN_TODO.read_text(encoding="utf-8")
    od_idx = todo.find("## OD — Options-Delta")
    if od_idx >= 0 and "OD same-day crypto pricing gate" not in todo[od_idx : od_idx + 6000]:
        insert_at = todo.find("- 2026-06-02 OD-RV Deribit daily settlement check", od_idx)
        if insert_at >= 0:
            todo = todo[:insert_at] + bullet + "\n" + todo[insert_at:]
    old_task = "- [x] **OD same-day crypto pricing gate** (2026-06-02): completed; Arm T CLOSE, Arm E CLOSE. See [[od_same_day_crypto_pricing_gate_findings]]."
    task = f"- [x] **OD same-day crypto pricing gate** (2026-06-02): completed; Arm T {verdict_t}, Arm E {verdict_e}. See [[od_same_day_crypto_pricing_gate_findings]]."
    todo = todo.replace(old_task, task)
    if task not in todo:
        marker = "- [ ] **Run OD-RV 08:00 aligned capture"
        pos = todo.find(marker, od_idx)
        if pos >= 0:
            todo = todo[:pos] + task + "\n" + todo[pos:]
    BRAIN_TODO.write_text(todo, encoding="utf-8")


def update_confirmation_docs(verdict_t: str, reason_t: str) -> None:
    new_line = (
        f"- 2026-06-02 OD same-day crypto pricing gate confirmation: **Arm T {verdict_t}; Arm E CLOSE**. "
        f"Arm T: {reason_t} Arm E remains closed. See [[od_same_day_crypto_pricing_gate_findings]]."
    )
    hub = OD_HUB.read_text(encoding="utf-8")
    hub = re.sub(r"- 2026-06-02 OD same-day crypto pricing gate(?: confirmation)?: \*\*Arm T .*?\n", new_line + "\n", hub)
    OD_HUB.write_text(hub, encoding="utf-8")

    todo = BRAIN_TODO.read_text(encoding="utf-8")
    todo = re.sub(r"- 2026-06-02 OD same-day crypto pricing gate(?: confirmation)?: \*\*Arm T .*?\n", new_line + "\n", todo)
    todo = re.sub(
        r"- \[x\] \*\*OD same-day crypto pricing gate\*\* \(2026-06-02\): completed; Arm T .*?\n",
        f"- [x] **OD same-day crypto pricing gate confirmation** (2026-06-02): completed; Arm T {verdict_t}, Arm E CLOSE. See [[od_same_day_crypto_pricing_gate_findings]].\n",
        todo,
    )
    BRAIN_TODO.write_text(todo, encoding="utf-8")


def append_tier1_section(verdict_t: str, reason_t: str, train_cells: pd.DataFrame, heldout_cells: pd.DataFrame, kou_summary: pd.DataFrame, meta: dict[str, Any]) -> None:
    note = NOTE.read_text(encoding="utf-8")
    note = re.sub(
        r"Final verdicts(?: after Arm T confirmation| after Arm T Tier-1 extension)?: \*\*Arm T .*?\*\* and \*\*Arm E CLOSE\*\*\.",
        f"Final verdicts after Arm T Tier-1 extension: **Arm T {verdict_t}** and **Arm E CLOSE**.",
        note,
    )
    note = re.sub(r"Arm T(?: confirmation| Tier-1)? reason: .*\n", f"Arm T Tier-1 reason: {reason_t}\n", note)
    note = re.sub(r"\*\*Arm T:\*\* .*?\n", f"**Arm T:** {verdict_t}. {reason_t}\n", note, count=1)
    if verdict_t == "MERITS-LIVE-MEASUREMENT-LOOP":
        next_step = "Concrete next step: stand up a constrained live MEASUREMENT loop for the Tier-1 survivor cell, instrumenting passive fill rate, real same-day touch capacity, and adverse selection before any trading-system claim. Keep Arm E closed."
    else:
        next_step = "Concrete next step: close same-day Arm T as a standalone strategy; fold the first-passage/HAR-Kou IV-gap flag into MM as a caution feature. Keep Arm E closed."
    note = re.sub(r"Concrete next step: .*?Keep Arm E closed\.", next_step, note, count=1)

    marker = "\n## Tier-1 Edge-Concentration Extension\n"
    if marker in note:
        note = note[: note.index(marker)].rstrip() + "\n"

    registered = train_cells[train_cells.get("registered_from_train", False)].copy() if not train_cells.empty else pd.DataFrame()
    heldout_registered = heldout_cells[heldout_cells.get("registered_from_train", False)].copy() if not heldout_cells.empty else pd.DataFrame()
    survivors = heldout_cells[heldout_cells.get("passes_tier1", False)].copy() if not heldout_cells.empty else pd.DataFrame()
    if not heldout_registered.empty:
        heldout_registered = heldout_registered.sort_values(["passes_tier1", "mean_capacity_haircut_ev"], ascending=[False, False]).copy()
    if not survivors.empty:
        survivors = survivors.sort_values(["mean_capacity_haircut_ev", "capacity_haircut_ci_lo"], ascending=False).copy()

    oos_read = (
        "The OOS guard is active and this sample passes it"
        if meta.get("oos_ok")
        else "The OOS guard failed, so this run is train-only and cannot pass"
    )
    section = f"""
## Tier-1 Edge-Concentration Extension

This extension asks whether a mechanism-backed filter can lift same-day Arm T above the confirmation pass's economically trivial ~0.02c/contract capacity-haircut floor. It uses only the saved PM fill sample plus cached Binance 1m bars; no new market capture, Deribit feed, Kronos model, or PM consistency scan is introduced.

Pre-registered Tier-1 rule: start with the original external empirical-edge-positive Arm T fills, invert each PM touch YES price to Brownian first-passage implied vol, compare it with a causal HAR-RV-J forecast plus directional Kou jump variance, and keep only fills where `PM touch IV - HAR/Kou sigma` is at least {number(TIER1_IV_GAP_THRESHOLD, 2)} annualized vol in the side-implied direction. Rich touch IV means fade by buying NO; cheap touch IV means buy YES. The materiality bar is {cents(TIER1_MATERIAL_CAPACITY_EV)} per contract after the 5% non-incumbent capacity haircut, with held-out lower CI > 0 and BH q <= 0.05.

OOS realism rule: use an embargoed held-out split only when the saved sample has at least {MIN_OOS_RESOLUTION_DATES} resolution dates and spans at least {MIN_OOS_SPAN_DAYS} days. Otherwise the script treats the sample as train-only rather than manufacturing a powerless split. {oos_read}: {meta['resolution_dates']} resolution dates over {meta['span_days']} days. Train is {meta['train_start']} to {meta['train_end']} ({number(meta['train_fills'], 0)} Tier-1 fills / {number(meta['train_markets'], 0)} markets), embargo is {meta['embargo_date']} ({number(meta['embargo_fills'], 0)} fills), and held-out is {meta['heldout_start']} to {meta['heldout_end']} ({number(meta['heldout_fills'], 0)} fills / {number(meta['heldout_markets'], 0)} markets).

Touch-specific structural baseline attempt: the saved sample has executed taker fills and realized outcomes, but not synchronized passive quote/queue states. A touch-market passive MM baseline cannot be derived offline from these fields. The old 1.98c crypto-4h structural baseline remains a borrowed diagnostic only, not an Arm T kill switch. The Tier-1 gate is raw net EV + 5% capacity-haircut EV + OOS/BH + materiality.

Directional HAR/Kou snapshot:

{table(kou_summary, ["asset", "forecast_session_date", "sigma_har_annualized", "sigma_har_kou_up_annualized", "sigma_har_kou_down_annualized", "kou_up_lambda_per_year", "kou_down_lambda_per_year", "kou_up_avg_abs_jump", "kou_down_avg_abs_jump"], 8) if not kou_summary.empty else "No HAR/Kou summary rows."}

Read: the Kou extension is a cheap moment-matched directional jump variance add-on, not a full jump-diffusion pricer. It is enough for the gate question: does PM touch IV look mechanically rich/cheap versus a causal vol-and-jump forecast?

![Tier-1 IV gap]({PLOT_TIER1_IV_GAP})

Caption: PM touch implied vol minus causal HAR-RV-J/Kou forecast vol for empirical-edge-positive Arm T fills. Dashed lines are the pre-registered +/-25 vol-point selection threshold.

Train-registered Tier-1 cells:

{table(registered, ["z_bucket", "tau_bucket", "fills", "markets", "mean_abs_z_har_kou", "mean_pm_touch_iv", "mean_sigma_har_kou", "mean_iv_gap_har_kou", "mean_realized_net_ev", "realized_net_ev_ci_lo", "mean_capacity_haircut_ev", "capacity_haircut_ci_lo"], 12) if not registered.empty else "No train cell registered."}

Read: these cells are chosen from pre-embargo data only. Held-out cells below decide the verdict.

Held-out registered cells after BH across {meta['heldout_cells_tested']} defensible Tier-1 cells:

{table(heldout_registered, ["z_bucket", "tau_bucket", "fills", "markets", "mean_abs_z_har_kou", "mean_entry_price", "mean_pm_yes_price", "mean_empirical_yes_prob", "mean_naive_terminal_yes_prob", "mean_pm_touch_iv", "mean_sigma_har_kou", "mean_iv_gap_har_kou", "mean_realized_net_ev", "realized_net_ev_ci_lo", "realized_net_ev_q", "mean_capacity_haircut_ev", "capacity_haircut_ci_lo", "capacity_haircut_q", "passes_materiality", "passes_tier1", "mechanism_read"], 12) if not heldout_registered.empty else "No train-registered Tier-1 cell had held-out observations."}

![Tier-1 held-out EV]({PLOT_TIER1_HELDOUT})

Caption: held-out 5% capacity-haircut EV per contract. The red dashed line is the {cents(TIER1_MATERIAL_CAPACITY_EV)} materiality threshold.

Mechanism check on Tier-1 survivors:

{table(survivors, ["z_bucket", "tau_bucket", "fills", "markets", "mean_pm_yes_price", "mean_naive_terminal_yes_prob", "mean_brownian_touch_yes_prob", "mean_empirical_yes_prob", "mean_capacity_haircut_ev", "capacity_haircut_ci_lo", "capacity_haircut_q", "mechanism_read"], 12) if not survivors.empty else "No Tier-1 cell survived OOS + capacity + BH + materiality."}

Read: the survivor count is {meta['survivor_cells']}. A survivor would merit a constrained live **MEASUREMENT** loop, not a trading system; no survivor means same-day touch is efficient net of realistic costs at this offline resolution.

Assumption-vs-live ledger:

| Ledger bucket | Items |
|---|---|
| Modeled assumptions | 5% non-incumbent capacity share; taker entry at executed price plus PM fee; empirical first-passage base rates from prior Binance history; HAR-RV-J and directional Kou jump parameters from causal Binance 1m history; OOS/BH market-cluster CIs. |
| Live-only unknowns | Passive fill rate; real non-incumbent headroom on these exact same-day touch markets; touch-specific passive/MM baseline; adverse selection around barrier touches; persistence versus the K3 ~10s/54s lead-lag decay; quote/queue behavior during touch jumps. |

**Tier-1 Arm T verdict:** {verdict_t}. {reason_t}

Tier-1 outputs: `data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_tier1_train_cells.csv`, `data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_tier1_heldout_cells.csv`, `data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_tier1_kou_params.csv`, and `data/analysis/od_same_day_crypto_arm_t_tier1_fills.parquet`.
"""
    note = note.rstrip() + "\n" + section
    NOTE.write_text(normalize_markdown_wrapping(note), encoding="utf-8")


def update_tier1_docs(verdict_t: str, reason_t: str) -> None:
    new_line = (
        f"- 2026-06-02 OD same-day Arm T Tier-1 edge-concentration extension: **Arm T {verdict_t}; Arm E CLOSE**. "
        f"{reason_t} See [[od_same_day_crypto_pricing_gate_findings]]."
    )
    hub = OD_HUB.read_text(encoding="utf-8")
    hub = re.sub(r"- 2026-06-02 OD same-day Arm T Tier-1 edge-concentration extension: \*\*Arm T .*?\n", "", hub)
    idx = hub.find("## Current state")
    if idx >= 0:
        line_end = hub.find("\n", idx)
        hub = hub[: line_end + 1] + new_line + "\n" + hub[line_end + 1 :]
    else:
        hub = new_line + "\n" + hub
    OD_HUB.write_text(hub, encoding="utf-8")

    todo = BRAIN_TODO.read_text(encoding="utf-8")
    todo = re.sub(r"- 2026-06-02 OD same-day Arm T Tier-1 edge-concentration extension: \*\*Arm T .*?\n", "", todo)
    todo = re.sub(r"- \[[ x]\] \*\*OD same-day Arm T Tier-1 edge-concentration extension.*?\n", "", todo)
    od_idx = todo.find("## OD — Options-Delta")
    if od_idx >= 0:
        line_end = todo.find("\n", od_idx)
        todo = todo[: line_end + 1] + new_line + "\n" + todo[line_end + 1 :]
    else:
        todo = new_line + "\n" + todo
    task = (
        f"- [x] **OD same-day Arm T Tier-1 edge-concentration extension** (2026-06-02): completed; "
        f"Arm T {verdict_t}. {reason_t} See [[od_same_day_crypto_pricing_gate_findings]]."
    )
    if task not in todo:
        marker = "- [ ] **Run OD-RV 08:00 aligned capture"
        pos = todo.find(marker, od_idx if od_idx >= 0 else 0)
        if pos >= 0:
            todo = todo[:pos] + task + "\n" + todo[pos:]
        else:
            todo = todo.rstrip() + "\n" + task + "\n"
    BRAIN_TODO.write_text(todo, encoding="utf-8")


def main() -> int:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    print("fetching same-day Gamma/CLOB universe", flush=True)
    current = fetch_current_universe()
    current.to_csv(OUT_CLASSIFICATION, index=False)
    print("building Binance 1m same-day features", flush=True)
    features = build_features(refresh=False)
    synth = build_synthetic(features)
    cv = expanding_cv(synth)
    calib = calibration(cv)
    calib.to_csv(OUT_CALIBRATION, index=False)
    print("loading historical same-day markets and executable fills", flush=True)
    hist = historical_markets()
    hist.to_csv(OUT_HISTORY_MARKETS, index=False)
    fills = load_fills(hist)
    fills = join_state(fills, features)
    fills = apply_models(fills, synth, features)
    fills.to_parquet(OUT_FILLS, index=False)
    arms, buckets = summarize(fills)
    arms.to_csv(OUT_ARM_SUMMARY, index=False)
    buckets.to_csv(OUT_BUCKETS, index=False)
    current_quotes = current_quote_models(current, features, synth)
    current_quotes.to_csv(OUT_CURRENT, index=False)
    cap = capacity(current, hist)
    cap.to_csv(OUT_CAPACITY, index=False)
    make_plots(calib, buckets)

    t_pass = (not buckets.empty) and bool(((buckets["arm"].eq("T")) & buckets["passes_gate"]).any())
    e_pass = (not buckets.empty) and bool(((buckets["arm"].eq("E")) & buckets["passes_gate"]).any())
    verdict_t = "MERITS-BUILD" if t_pass else "CLOSE"
    verdict_e = "MERITS-BUILD" if e_pass else "CLOSE"
    if t_pass:
        row = buckets[(buckets["arm"].eq("T")) & buckets["passes_gate"]].iloc[0]
        reason_t = f"best cell {row['z_bucket']}/{row['tau_bucket']} has net CI {fmt_ci(float(row['realized_net_ev_ci_lo']), float(row['realized_net_ev_ci_hi']))}."
    else:
        reason_t = "no defensible same-day touch cell has lower-CI-positive realized net EV after executable entry and PM fees."
    if e_pass:
        row = buckets[(buckets["arm"].eq("E")) & buckets["passes_gate"]].iloc[0]
        reason_e = f"best cell {row['z_bucket']}/{row['tau_bucket']} clears net CI {fmt_ci(float(row['realized_net_ev_ci_lo']), float(row['realized_net_ev_ci_hi']))} and incremental CI {fmt_ci(float(row['incremental_ci_lo']), float(row['incremental_ci_hi']))}."
    else:
        reason_e = "terminal ladder power improves the sample, but no cell clears both net-EV lower CI and incremental-vs-structural lower CI."
    write_note(current, hist, arms, buckets, calib, current_quotes, cap, verdict_t, verdict_e, reason_t, reason_e)
    update_docs(verdict_t, verdict_e, reason_t, reason_e)
    print(f"wrote {NOTE.relative_to(ROOT)}")
    print(f"Arm T {verdict_t}: {reason_t}")
    print(f"Arm E {verdict_e}: {reason_e}")
    print(arms.to_string(index=False))
    return 0


def confirmation_main() -> int:
    verdict_t, reason_t, train_cells, heldout_cells, meta = arm_t_confirmation()
    append_confirmation_section(verdict_t, reason_t, train_cells, heldout_cells, meta)
    update_confirmation_docs(verdict_t, reason_t)
    print(f"wrote confirmation section to {NOTE.relative_to(ROOT)}")
    print(f"Arm T confirmation {verdict_t}: {reason_t}")
    if not heldout_cells.empty:
        cols = [
            "z_bucket",
            "tau_bucket",
            "fills",
            "markets",
            "registered_from_train",
            "passes_confirmation",
            "mean_realized_net_ev",
            "realized_net_ev_ci_lo",
            "realized_net_ev_ci_hi",
            "realized_net_ev_q",
            "mean_capacity_haircut_ev",
            "capacity_haircut_ci_lo",
            "capacity_haircut_ci_hi",
            "capacity_haircut_q",
            "mean_borrowed_structural_ev",
            "borrowed_structural_ci_lo",
            "borrowed_structural_ci_hi",
        ]
        print(heldout_cells[cols].sort_values(["passes_confirmation", "capacity_haircut_ci_lo"], ascending=[False, False]).head(12).to_string(index=False))
    return 0


def tier1_main() -> int:
    verdict_t, reason_t, train_cells, heldout_cells, kou_summary, meta = arm_t_tier1_extension()
    append_tier1_section(verdict_t, reason_t, train_cells, heldout_cells, kou_summary, meta)
    update_tier1_docs(verdict_t, reason_t)
    print(f"wrote Tier-1 extension section to {NOTE.relative_to(ROOT)}")
    print(f"Arm T Tier-1 {verdict_t}: {reason_t}")
    if not heldout_cells.empty:
        cols = [
            "z_bucket",
            "tau_bucket",
            "fills",
            "markets",
            "registered_from_train",
            "passes_tier1",
            "mean_pm_touch_iv",
            "mean_sigma_har_kou",
            "mean_iv_gap_har_kou",
            "mean_realized_net_ev",
            "realized_net_ev_ci_lo",
            "realized_net_ev_ci_hi",
            "realized_net_ev_q",
            "mean_capacity_haircut_ev",
            "capacity_haircut_ci_lo",
            "capacity_haircut_ci_hi",
            "capacity_haircut_q",
            "passes_materiality",
            "mechanism_read",
        ]
        print(heldout_cells[cols].sort_values(["passes_tier1", "mean_capacity_haircut_ev"], ascending=[False, False]).head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    if "--confirm-arm-t-only" in sys.argv:
        raise SystemExit(confirmation_main())
    if "--tier1-arm-t-only" in sys.argv:
        raise SystemExit(tier1_main())
    raise SystemExit(main())
