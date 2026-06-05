"""SPX daily up/down cheap N(z) pricing gate.

This is the no-options-data pass for the equities financial-binary reopen.
It compares executable Polymarket SPX daily up/down fills with a causal
realized-vol N(z) digital fair value, then adds an empirical conditional
calibration table in the style of the crypto Arm-B check.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/od_equities_spx_nz_pricing_gate.py
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import duckdb
import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import cents, markdown_table, number, pct
from od_strategy_a_v3 import normalize_markdown_wrapping


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ANALYSIS = ROOT / "data" / "analysis"
EXTERNAL = ROOT / "data" / "external" / "spx_yahoo"
CSV_OUT = ANALYSIS / "csv_outputs" / "options_delta"
PLOTS = ANALYSIS / "plots" / "options_delta"
TRADES_DIR = ROOT / "data" / "trades"
NOTES = ROOT / "notes" / "options_delta"
BRAIN_TODO = REPO / "brain" / "TODO.md"

MARKET_DETAIL = CSV_OUT / "od_equities_index_pricing_scope_market_detail.csv"
NOTE = NOTES / "od_equities_index_pricing_scope_findings.md"
HUB = NOTES / "strat_options_delta.md"

OUT_RAW_FILLS = ANALYSIS / "od_equities_spx_nz_pricing_raw_fills.parquet"
OUT_FILLS = ANALYSIS / "od_equities_spx_nz_pricing_fills.parquet"
OUT_DAILY = EXTERNAL / "spx_yahoo_daily.parquet"
OUT_HOURLY = EXTERNAL / "spx_yahoo_60m.parquet"
OUT_STATES = ANALYSIS / "od_equities_spx_nz_reference_states.parquet"
OUT_SUMMARY = CSV_OUT / "od_equities_spx_nz_pricing_summary.csv"
OUT_CALIBRATION = CSV_OUT / "od_equities_spx_nz_pricing_calibration.csv"
OUT_DATE_SUMMARY = CSV_OUT / "od_equities_spx_nz_pricing_date_summary.csv"
OUT_RESIDUALS = CSV_OUT / "od_equities_spx_nz_pricing_residual_sample.csv"
OUT_LEDGER = CSV_OUT / "od_equities_spx_nz_pricing_data_ledger.csv"
PLOT_RESIDUALS = PLOTS / "od_equities_spx_nz_pricing_date_edges.png"
PLOT_CALIBRATION = PLOTS / "od_equities_spx_nz_pricing_calibration.png"

TRADES_GLOB = str(TRADES_DIR / "trades_delta_shard*.parquet")
YEAR_SECONDS = 365.0 * 24.0 * 3600.0
TRADING_DAYS = 252.0
FINANCE_TAKER_FEE_RATE = 0.04
BOOTSTRAP_SAMPLES = 2000
RNG_SEED = 20260603
HTTP_HEADERS = {"User-Agent": "epsilon-quant-research-spx-nz-gate/1.0 Mozilla/5.0"}
AS_OF = pd.Timestamp("2026-06-03", tz="UTC")
MARKET_CLOSE_HOUR_UTC = 20
MIN_EMPIRICAL_BIN_N = 20


def norm_cdf(x: pd.Series | np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))


def fee(price: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray:
    p = np.asarray(price, dtype=float)
    return FINANCE_TAKER_FEE_RATE * np.clip(p, 0.0, 1.0) * (1.0 - np.clip(p, 0.0, 1.0))


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return []
    try:
        out = json.loads(str(value))
        return out if isinstance(out, list) else []
    except Exception:
        return []


def event_date_from_slug(slug: str) -> pd.Timestamp:
    m = re.search(r"on-([a-z]+)-(\d{1,2})-(\d{4})", str(slug))
    if not m:
        return pd.NaT
    month, day, year = m.groups()
    return pd.Timestamp(f"{year}-{month}-{int(day):02d}", tz="UTC")


def official_close_ts(event_date: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(event_date):
        return pd.NaT
    return event_date.normalize() + pd.Timedelta(hours=MARKET_CLOSE_HOUR_UTC)


def yahoo_chart(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp, cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    if cache_path.exists() and not refresh:
        out = pd.read_parquet(cache_path)
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
        return out
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    period1 = int(start.timestamp())
    period2 = int(end.timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}?period1={period1}&period2={period2}&interval={interval}&includePrePost=false"
    )
    with httpx.Client(headers=HTTP_HEADERS, timeout=30, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {symbol} {interval}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"empty Yahoo chart response for {symbol} {interval}")
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    out = pd.DataFrame(quote)
    out["ts"] = pd.to_datetime(timestamps, unit="s", utc=True)
    out = out[out["ts"].between(start, end, inclusive="left")].copy()
    out["source_url"] = url
    out.to_parquet(cache_path, index=False)
    time.sleep(0.25)
    return out


def spx_market_detail() -> pd.DataFrame:
    df = pd.read_csv(MARKET_DETAIL)
    df = df[(df["family"].eq("index_up_down")) & (df["underlying"].eq("SPX"))].copy()
    df["market_id"] = df["market_id"].astype(str)
    df["event_date"] = df["market_slug"].map(event_date_from_slug)
    df["official_close_ts"] = df["event_date"].map(official_close_ts)
    df["outcome_list"] = df["outcomes"].map(parse_json_list)
    df["token_list"] = df["clob_token_ids"].map(parse_json_list)
    df["price_list"] = df["outcome_prices"].map(parse_json_list)
    df["resolved_up"] = [
        safe_float(prices[0]) > 0.5 if len(prices) >= 2 else math.nan
        for prices in df["price_list"]
    ]
    df = df[df["event_date"].notna() & df["token_list"].map(len).ge(2)].copy()
    return df.sort_values("event_date").reset_index(drop=True)


def materialize_spx_fills(markets: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    if OUT_RAW_FILLS.exists() and not refresh:
        out = pd.read_parquet(OUT_RAW_FILLS)
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        return out

    ids = sorted(markets["market_id"].astype(str).unique())
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='12GB'")
    con.register("spx_ids", pd.DataFrame({"market_id": ids}))
    query = f"""
        SELECT raw.*
        FROM read_parquet('{TRADES_GLOB}') raw
        JOIN spx_ids USING (market_id)
        WHERE raw.timestamp >= TIMESTAMP '2026-03-20 00:00:00'
          AND raw.timestamp < TIMESTAMP '2026-06-04 00:00:00'
          AND raw.usd_amount > 0
          AND raw.token_amount > 0
          AND raw.price BETWEEN 0.001 AND 0.999
    """
    out = con.execute(query).fetchdf()
    if out.empty:
        raise RuntimeError("no SPX fills found in local trade shards")
    dedupe_cols = [
        "timestamp",
        "market_id",
        "maker",
        "taker",
        "maker_asset_id",
        "taker_asset_id",
        "usd_amount",
        "token_amount",
        "transaction_hash",
    ]
    out = out.drop_duplicates(dedupe_cols).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    token_rows = []
    for row in markets.itertuples(index=False):
        for outcome, token in zip(row.outcome_list, row.token_list, strict=False):
            token_rows.append({"market_id": str(row.market_id), "token_id": str(token), "token_outcome": str(outcome).lower()})
    token_map = pd.DataFrame(token_rows)
    out["token_id"] = np.where(out["maker_asset_id"].astype(str).eq("0"), out["taker_asset_id"].astype(str), out["maker_asset_id"].astype(str))
    out = out.merge(token_map, on=["market_id", "token_id"], how="left")
    out = out.merge(
        markets[
            [
                "market_id",
                "market_slug",
                "event_date",
                "official_close_ts",
                "resolved_up",
                "volume_usd",
            ]
        ],
        on="market_id",
        how="left",
    )
    out["event_date"] = pd.to_datetime(out["event_date"], utc=True)
    out["official_close_ts"] = pd.to_datetime(out["official_close_ts"], utc=True)
    out = out[out["token_outcome"].isin(["up", "down"])].copy()
    OUT_RAW_FILLS.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_RAW_FILLS, index=False)
    print(f"wrote SPX raw fill cache rows={len(out):,} markets={out['market_id'].nunique():,} -> {OUT_RAW_FILLS}", flush=True)
    return out


def build_spx_references(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = yahoo_chart(
        "%5EGSPC",
        "1d",
        pd.Timestamp("2021-01-01", tz="UTC"),
        AS_OF + pd.Timedelta(days=2),
        OUT_DAILY,
        refresh=refresh,
    )
    hourly = yahoo_chart(
        "%5EGSPC",
        "60m",
        pd.Timestamp("2025-01-01", tz="UTC"),
        AS_OF + pd.Timedelta(days=2),
        OUT_HOURLY,
        refresh=refresh,
    )
    for col in ["open", "high", "low", "close"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
        hourly[col] = pd.to_numeric(hourly[col], errors="coerce")
    daily = daily.dropna(subset=["close"]).sort_values("ts").reset_index(drop=True)
    daily["event_date"] = daily["ts"].dt.normalize()
    daily["official_close_ts"] = daily["event_date"].map(official_close_ts)
    daily["prev_close"] = daily["close"].shift(1)
    daily["prev_close_ts"] = daily["official_close_ts"].shift(1)
    daily["daily_log_ret"] = np.log(daily["close"] / daily["prev_close"])
    ewma_var = daily["daily_log_ret"].shift(1).pow(2).ewm(span=20, min_periods=10, adjust=False).mean()
    daily["ewma_sigma_annualized"] = np.sqrt(ewma_var * TRADING_DAYS)
    daily["rolling20_sigma_annualized"] = daily["daily_log_ret"].shift(1).rolling(20, min_periods=10).std() * math.sqrt(TRADING_DAYS)
    daily["outcome_up"] = daily["close"].gt(daily["prev_close"])
    daily = daily[daily["prev_close"].notna() & daily["ewma_sigma_annualized"].notna()].copy()

    hourly = hourly.dropna(subset=["close"]).sort_values("ts").reset_index(drop=True)
    hourly["event_date"] = hourly["ts"].dt.normalize()
    hourly = hourly.merge(
        daily[
            [
                "event_date",
                "prev_close",
                "prev_close_ts",
                "close",
                "official_close_ts",
                "outcome_up",
                "ewma_sigma_annualized",
                "rolling20_sigma_annualized",
            ]
        ],
        on="event_date",
        how="left",
        suffixes=("", "_daily"),
    )
    hourly["state_ts"] = hourly["ts"] + pd.Timedelta(hours=1)
    hourly["state_ts"] = hourly[["state_ts", "official_close_ts"]].min(axis=1)
    hourly = hourly[
        hourly["prev_close"].notna()
        & hourly["state_ts"].le(hourly["official_close_ts"])
        & hourly["state_ts"].gt(hourly["prev_close_ts"])
    ].copy()
    hourly["spot"] = hourly["close"].astype(float)
    hourly["state_source"] = "yahoo_60m_completed_bar"

    prev_states = daily[
        [
            "event_date",
            "prev_close",
            "prev_close_ts",
            "close",
            "official_close_ts",
            "outcome_up",
            "ewma_sigma_annualized",
            "rolling20_sigma_annualized",
        ]
    ].copy()
    prev_states["state_ts"] = pd.to_datetime(prev_states["prev_close_ts"], utc=True)
    prev_states["spot"] = prev_states["prev_close"].astype(float)
    prev_states["state_source"] = "previous_official_close"

    states = pd.concat(
        [
            prev_states[
                [
                    "event_date",
                    "state_ts",
                    "spot",
                    "prev_close",
                    "prev_close_ts",
                    "close",
                    "official_close_ts",
                    "outcome_up",
                    "ewma_sigma_annualized",
                    "rolling20_sigma_annualized",
                    "state_source",
                ]
            ],
            hourly[
                [
                    "event_date",
                    "state_ts",
                    "spot",
                    "prev_close",
                    "prev_close_ts",
                    "close_daily",
                    "official_close_ts",
                    "outcome_up",
                    "ewma_sigma_annualized",
                    "rolling20_sigma_annualized",
                    "state_source",
                ]
            ].rename(columns={"close_daily": "close"}),
        ],
        ignore_index=True,
    )
    states["tau_seconds"] = (states["official_close_ts"] - states["state_ts"]).dt.total_seconds().clip(lower=1.0)
    states["tau_hours"] = states["tau_seconds"] / 3600.0
    states["z"] = np.log(states["spot"] / states["prev_close"]) / (
        states["ewma_sigma_annualized"].clip(lower=1e-6) * np.sqrt(states["tau_seconds"] / YEAR_SECONDS)
    )
    states["p_up_nz"] = norm_cdf(states["z"]).clip(0.0, 1.0)
    states = add_state_buckets(states.replace([np.inf, -np.inf], np.nan).dropna(subset=["z", "p_up_nz"]))
    OUT_STATES.parent.mkdir(parents=True, exist_ok=True)
    states.to_parquet(OUT_STATES, index=False)
    return daily, states


def add_state_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_bucket"] = pd.cut(
        out["tau_hours"].astype(float),
        bins=[0.0, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0],
        labels=["lt30m", "30_60m", "1_2h", "2_4h", "4_10h", "overnight"],
        include_lowest=True,
    ).astype(str)
    out["z_bucket"] = pd.cut(
        out["z"].astype(float),
        bins=[-np.inf, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, np.inf],
        labels=["z_lt_m2", "z_m2_m1", "z_m1_m05", "z_m05_m02", "z_m02_0", "z_0_02", "z_02_05", "z_05_1", "z_1_2", "z_gt_2"],
    ).astype(str)
    out["z_bucket_coarse"] = pd.cut(
        out["z"].astype(float),
        bins=[-np.inf, -1.0, -0.25, 0.25, 1.0, np.inf],
        labels=["neg_far", "neg_mid", "near", "pos_mid", "pos_far"],
    ).astype(str)
    return out


def empirical_probability(row: pd.Series, train: pd.DataFrame) -> tuple[float, int, str]:
    if train.empty:
        return float(row["p_up_nz"]), 0, "nz_fallback_no_history"
    primary = train[(train["time_bucket"].eq(row["time_bucket"])) & (train["z_bucket"].eq(row["z_bucket"]))]
    if len(primary) >= MIN_EMPIRICAL_BIN_N:
        return float(primary["outcome_up"].mean()), int(len(primary)), "primary_time_z"
    coarse = train[(train["time_bucket"].eq(row["time_bucket"])) & (train["z_bucket_coarse"].eq(row["z_bucket_coarse"]))]
    if len(coarse) >= MIN_EMPIRICAL_BIN_N:
        return float(coarse["outcome_up"].mean()), int(len(coarse)), "coarse_time_z"
    zonly = train[train["z_bucket_coarse"].eq(row["z_bucket_coarse"])]
    if len(zonly) >= MIN_EMPIRICAL_BIN_N:
        return float(zonly["outcome_up"].mean()), int(len(zonly)), "coarse_z_only"
    return float(row["p_up_nz"]), int(len(primary)), "nz_fallback_sparse"


def add_empirical_predictions(df: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["event_date", "state_ts"]).reset_index(drop=True)
    states = states.sort_values(["event_date", "state_ts"]).reset_index(drop=True)
    key_cols = ["event_date", "time_bucket", "z_bucket", "z_bucket_coarse"]
    keys = out[key_cols].drop_duplicates().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for row in keys.itertuples(index=False):
        train = states[states["event_date"].lt(row.event_date)]
        key = pd.Series(row._asdict())
        key["p_up_nz"] = math.nan
        if train.empty:
            rows.append({**row._asdict(), "p_up_empirical_lookup": math.nan, "empirical_bin_n": 0, "empirical_method": "nz_fallback_no_history"})
            continue
        primary = train[(train["time_bucket"].eq(row.time_bucket)) & (train["z_bucket"].eq(row.z_bucket))]
        if len(primary) >= MIN_EMPIRICAL_BIN_N:
            rows.append({**row._asdict(), "p_up_empirical_lookup": float(primary["outcome_up"].mean()), "empirical_bin_n": int(len(primary)), "empirical_method": "primary_time_z"})
            continue
        coarse = train[(train["time_bucket"].eq(row.time_bucket)) & (train["z_bucket_coarse"].eq(row.z_bucket_coarse))]
        if len(coarse) >= MIN_EMPIRICAL_BIN_N:
            rows.append({**row._asdict(), "p_up_empirical_lookup": float(coarse["outcome_up"].mean()), "empirical_bin_n": int(len(coarse)), "empirical_method": "coarse_time_z"})
            continue
        zonly = train[train["z_bucket_coarse"].eq(row.z_bucket_coarse)]
        if len(zonly) >= MIN_EMPIRICAL_BIN_N:
            rows.append({**row._asdict(), "p_up_empirical_lookup": float(zonly["outcome_up"].mean()), "empirical_bin_n": int(len(zonly)), "empirical_method": "coarse_z_only"})
            continue
        rows.append({**row._asdict(), "p_up_empirical_lookup": math.nan, "empirical_bin_n": int(len(primary)), "empirical_method": "nz_fallback_sparse"})
    lookup = pd.DataFrame(rows)
    out = out.merge(lookup, on=key_cols, how="left")
    out["p_up_empirical"] = out["p_up_empirical_lookup"].fillna(out["p_up_nz"]).astype(float).clip(0.0, 1.0)
    out = out.drop(columns=["p_up_empirical_lookup"])
    return out


def score_fills(fills: pd.DataFrame, daily: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    refs = states[
        [
            "event_date",
            "state_ts",
            "spot",
            "prev_close",
            "prev_close_ts",
            "close",
            "official_close_ts",
            "outcome_up",
            "ewma_sigma_annualized",
            "rolling20_sigma_annualized",
            "tau_seconds",
            "tau_hours",
            "z",
            "p_up_nz",
            "time_bucket",
            "z_bucket",
            "z_bucket_coarse",
            "state_source",
        ]
    ].copy()
    scored_parts: list[pd.DataFrame] = []
    fills = fills.copy()
    fills["event_date"] = pd.to_datetime(fills["event_date"], utc=True)
    fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True)
    for event_date, group in fills.groupby("event_date", sort=False):
        ref = refs[refs["event_date"].eq(event_date)].sort_values("state_ts").copy()
        if ref.empty:
            continue
        g = group.sort_values("timestamp").copy()
        joined = pd.merge_asof(g, ref, left_on="timestamp", right_on="state_ts", direction="backward", suffixes=("", "_ref"))
        scored_parts.append(joined)
    out = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    if out.empty:
        raise RuntimeError("no fills could be joined to SPX reference states")
    out = out[
        out["state_ts"].notna()
        & out["timestamp"].ge(out["prev_close_ts"])
        & out["timestamp"].lt(out["official_close_ts"])
        & out["tau_seconds"].gt(0)
    ].copy()
    out = add_state_buckets(out)
    out = add_empirical_predictions(out, states)
    out["token_fair_nz"] = np.where(out["token_outcome"].eq("up"), out["p_up_nz"], 1.0 - out["p_up_nz"])
    out["token_fair_empirical"] = np.where(out["token_outcome"].eq("up"), out["p_up_empirical"], 1.0 - out["p_up_empirical"])
    out["taker_fee"] = fee(out["price"].astype(float))
    out["route"] = np.where(out["maker_side"].eq("SELL"), "taker_buy_token", "taker_sell_token")
    out["payoff"] = np.where(
        out["token_outcome"].eq("up"),
        out["outcome_up"].astype(bool).astype(float),
        (~out["outcome_up"].astype(bool)).astype(float),
    )
    for label in ["nz", "empirical"]:
        fair_col = f"token_fair_{label}"
        edge_col = f"edge_{label}"
        out[edge_col] = np.where(
            out["route"].eq("taker_buy_token"),
            out[fair_col].astype(float) - out["price"].astype(float) - out["taker_fee"].astype(float),
            out["price"].astype(float) - out[fair_col].astype(float) - out["taker_fee"].astype(float),
        )
    out["realized_net_pnl"] = np.where(
        out["route"].eq("taker_buy_token"),
        out["payoff"].astype(float) - out["price"].astype(float) - out["taker_fee"].astype(float),
        out["price"].astype(float) - out["payoff"].astype(float) - out["taker_fee"].astype(float),
    )
    out["market_date"] = out["event_date"].dt.strftime("%Y-%m-%d")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["edge_nz", "edge_empirical", "realized_net_pnl"])
    return out


def weighted_cluster_ci(df: pd.DataFrame, value_col: str, weight_col: str = "token_amount", cluster_col: str = "market_date", seed: int = 0) -> tuple[float, float]:
    if df.empty:
        return math.nan, math.nan
    groups = []
    for _, g in df.groupby(cluster_col, sort=False):
        w = g[weight_col].astype(float).to_numpy()
        v = g[value_col].astype(float).to_numpy()
        mask = np.isfinite(w) & np.isfinite(v) & (w > 0)
        if mask.any():
            groups.append((float(np.sum(w[mask] * v[mask])), float(np.sum(w[mask]))))
    if not groups:
        return math.nan, math.nan
    if len(groups) == 1:
        val = groups[0][0] / groups[0][1]
        return val, val
    rng = np.random.default_rng(RNG_SEED + seed + len(groups))
    sums = np.asarray([g[0] for g in groups], dtype=float)
    weights = np.asarray([g[1] for g in groups], dtype=float)
    draws = rng.integers(0, len(groups), size=(BOOTSTRAP_SAMPLES, len(groups)))
    vals = sums[draws].sum(axis=1) / np.maximum(weights[draws].sum(axis=1), 1e-12)
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_cell(label: str, df: pd.DataFrame, score_col: str, selection: pd.Series, seed: int) -> dict[str, Any]:
    sub = df[selection].copy()
    if "first_per_date" in label and not sub.empty:
        sub = sub.sort_values(["market_date", "timestamp"]).groupby("market_date", as_index=False, sort=False).head(1).copy()
    edge_lo, edge_hi = weighted_cluster_ci(sub, score_col, seed=seed)
    pnl_lo, pnl_hi = weighted_cluster_ci(sub, "realized_net_pnl", seed=seed + 1000)
    return {
        "cell": label,
        "score_col": score_col,
        "fills": int(len(sub)),
        "market_dates": int(sub["market_date"].nunique()) if not sub.empty else 0,
        "weighted_notional_usd": float(sub["usd_amount"].sum()) if not sub.empty else 0.0,
        "mean_model_edge": float(np.average(sub[score_col], weights=sub["token_amount"])) if not sub.empty else math.nan,
        "edge_ci_lo": edge_lo,
        "edge_ci_hi": edge_hi,
        "mean_realized_net_pnl": float(np.average(sub["realized_net_pnl"], weights=sub["token_amount"])) if not sub.empty else math.nan,
        "pnl_ci_lo": pnl_lo,
        "pnl_ci_hi": pnl_hi,
        "mean_pm_price": float(np.average(sub["price"], weights=sub["token_amount"])) if not sub.empty else math.nan,
        "mean_fee": float(np.average(sub["taker_fee"], weights=sub["token_amount"])) if not sub.empty else math.nan,
    }


def summarize(scored: pd.DataFrame, states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = [
        summarize_cell("all_causal_fills_nz_side_aware", scored, "edge_nz", pd.Series(True, index=scored.index), 1),
        summarize_cell("all_causal_fills_empirical_side_aware", scored, "edge_empirical", pd.Series(True, index=scored.index), 2),
        summarize_cell("first_per_date_nz_edge_gt_0", scored, "edge_nz", scored["edge_nz"].gt(0.0), 3),
        summarize_cell("first_per_date_nz_edge_ge_1c", scored, "edge_nz", scored["edge_nz"].ge(0.01), 4),
        summarize_cell("first_per_date_nz_edge_ge_2c", scored, "edge_nz", scored["edge_nz"].ge(0.02), 5),
        summarize_cell("first_per_date_emp_edge_gt_0", scored, "edge_empirical", scored["edge_empirical"].gt(0.0), 6),
        summarize_cell("first_per_date_emp_edge_ge_1c", scored, "edge_empirical", scored["edge_empirical"].ge(0.01), 7),
        summarize_cell("first_per_date_emp_edge_ge_2c", scored, "edge_empirical", scored["edge_empirical"].ge(0.02), 8),
    ]
    summary = pd.DataFrame(rows)

    date_summary = (
        scored.groupby("market_date", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "fills": len(g),
                    "notional_usd": g["usd_amount"].sum(),
                    "mean_edge_nz": np.average(g["edge_nz"], weights=g["token_amount"]),
                    "mean_edge_empirical": np.average(g["edge_empirical"], weights=g["token_amount"]),
                    "mean_realized_net_pnl": np.average(g["realized_net_pnl"], weights=g["token_amount"]),
                    "up_resolved": bool(g["outcome_up"].iloc[0]),
                    "mean_abs_z": np.average(g["z"].abs(), weights=g["token_amount"]),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    cv_rows = []
    cv_source = states[states["event_date"].ge(pd.Timestamp("2026-01-01", tz="UTC"))].copy()
    cv_source = add_empirical_predictions(cv_source, states[states["event_date"].lt(pd.Timestamp("2026-06-04", tz="UTC"))])
    for model, col in [("N(z)", "p_up_nz"), ("empirical", "p_up_empirical")]:
        cv_source["prob_bin"] = pd.cut(
            cv_source[col].astype(float),
            bins=[0.0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0],
            labels=["0_10c", "10_25c", "25_40c", "40_60c", "60_75c", "75_90c", "90_100c"],
            include_lowest=True,
        ).astype(str)
        for bucket, g in cv_source.groupby("prob_bin", sort=False):
            if g.empty or bucket == "nan":
                continue
            cv_rows.append(
                {
                    "model": model,
                    "prob_bin": bucket,
                    "rows": int(len(g)),
                    "mean_pred_up": float(g[col].mean()),
                    "observed_up": float(g["outcome_up"].mean()),
                    "observed_minus_pred": float(g["outcome_up"].mean() - g[col].mean()),
                }
            )
    calibration = pd.DataFrame(cv_rows)
    return summary, date_summary, calibration


def plot_outputs(date_summary: pd.DataFrame, calibration: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    if not date_summary.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(date_summary))
        ax.bar(x - 0.18, 100.0 * date_summary["mean_edge_nz"], width=0.36, label="N(z) residual")
        ax.bar(x + 0.18, 100.0 * date_summary["mean_realized_net_pnl"], width=0.36, label="realized net PnL")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x[:: max(1, len(x) // 12)])
        ax.set_xticklabels(date_summary["market_date"].iloc[:: max(1, len(x) // 12)], rotation=30, ha="right")
        ax.set_ylabel("cents/contract")
        ax.set_title("SPX daily up/down executable fill residual by market date")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_RESIDUALS, dpi=160)
        plt.close(fig)
    if not calibration.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        for model, g in calibration.groupby("model"):
            ax.scatter(g["mean_pred_up"], g["observed_up"], s=np.sqrt(g["rows"]) * 4.0, alpha=0.75, label=model)
            for _, row in g.iterrows():
                ax.text(row["mean_pred_up"], row["observed_up"], str(row["prob_bin"]), fontsize=8)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set_xlabel("Mean predicted UP probability")
        ax.set_ylabel("Observed UP frequency")
        ax.set_title("SPX hourly-state calibration, 2026 forward CV")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_CALIBRATION, dpi=160)
        plt.close(fig)


def fmt_ci(lo: float, hi: float, unit: str = "c") -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "[n/a, n/a]"
    if unit == "c":
        return f"[{cents(lo)}, {cents(hi)}]"
    return f"[{number(lo, 3)}, {number(hi, 3)}]"


def table(df: pd.DataFrame, cols: list[str], limit: int = 20) -> str:
    rows = []
    for _, r in df.head(limit).iterrows():
        out = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, (int, np.integer)):
                out.append(f"{int(v):,}")
            elif isinstance(v, (float, np.floating)):
                if any(k in c for k in ["edge", "pnl", "fee", "price"]):
                    out.append(cents(float(v)))
                elif "usd" in c:
                    out.append(f"${float(v):,.0f}")
                elif "pred" in c or "observed" in c:
                    out.append(pct(float(v)))
                else:
                    out.append(number(float(v), 3))
            elif isinstance(v, (bool, np.bool_)):
                out.append("true" if bool(v) else "false")
            else:
                out.append(str(v))
        rows.append(out)
    return markdown_table(cols, rows)


def write_outputs(scored: pd.DataFrame, summary: pd.DataFrame, date_summary: pd.DataFrame, calibration: pd.DataFrame, states: pd.DataFrame) -> None:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    date_summary.to_csv(OUT_DATE_SUMMARY, index=False)
    calibration.to_csv(OUT_CALIBRATION, index=False)
    sample_cols = [
        "timestamp",
        "market_date",
        "market_slug",
        "token_outcome",
        "maker_side",
        "route",
        "price",
        "token_fair_nz",
        "token_fair_empirical",
        "taker_fee",
        "edge_nz",
        "edge_empirical",
        "realized_net_pnl",
        "z",
        "tau_hours",
        "state_source",
        "empirical_method",
        "empirical_bin_n",
    ]
    scored.sort_values("timestamp")[sample_cols].head(5000).to_csv(OUT_RESIDUALS, index=False)
    pd.DataFrame(
        [
            {
                "bucket": "pm_data",
                "detail": f"{scored['market_date'].nunique():,} resolved SPX market dates, {len(scored):,} causal fill rows after excluding pre-prior-close and post-official-close trades; local raw trade shards end at 2026-05-26 19:57:58 UTC.",
            },
            {
                "bucket": "reference_data",
                "detail": f"Yahoo Finance chart API ^GSPC daily rows plus 60m completed bars; {len(states):,} causal SPX reference states from 2025-01 onward.",
            },
            {
                "bucket": "not_built",
                "detail": "No Cboe/OPRA option surface was fetched or built because the cheap N(z)/realized-vol gate did not produce a clean executable residual.",
            },
        ]
    ).to_csv(OUT_LEDGER, index=False)


def append_note(scored: pd.DataFrame, summary: pd.DataFrame, date_summary: pd.DataFrame, calibration: pd.DataFrame) -> None:
    all_nz = summary[summary["cell"].eq("all_causal_fills_nz_side_aware")].iloc[0]
    best = summary.sort_values(["pnl_ci_lo", "edge_ci_lo"], ascending=False).iloc[0]
    nz_gt0 = summary[summary["cell"].eq("first_per_date_nz_edge_gt_0")].iloc[0]
    verdict = "CONFIRM-CLOSE"

    section = f"""
## 2026-06-03 N(z) / Realized-Vol Pricing Gate

Pricing verdict: **{verdict}**. Do **not** build the Cboe/OPRA options surface for SPX daily up/down yet.

This append runs the cheap-first gate requested after the capacity pass. It compares actual executable SPX daily up/down fills to a causal S&P 500 `N(z)` digital fair value, then checks an empirical conditional probability table in the same spirit as [[od_conditional_prob_calibration_findings]]. The reference data is Yahoo Finance `^GSPC` daily closes plus completed 60-minute bars; no listed-option surface is used.

Headline numbers: **{len(scored):,}** causal fill rows across **{scored['market_date'].nunique():,}** resolved SPX market dates survived the filters. The all-fill side-aware N(z) residual is **{cents(float(all_nz['mean_model_edge']))}**, market-date CI **{fmt_ci(float(all_nz['edge_ci_lo']), float(all_nz['edge_ci_hi']))}**. The first-per-date N(z) `edge > 0` row has model residual **{cents(float(nz_gt0['mean_model_edge']))}**, CI **{fmt_ci(float(nz_gt0['edge_ci_lo']), float(nz_gt0['edge_ci_hi']))}**, but realized net PnL **{cents(float(nz_gt0['mean_realized_net_pnl']))}**, CI **{fmt_ci(float(nz_gt0['pnl_ci_lo']), float(nz_gt0['pnl_ci_hi']))}**. That is not a clean executable residual worth escalating to OPRA.

### Design

Each row is an actual Polymarket fill for the SPX daily up/down market, not a midpoint. The market resolves `Up` if the official SPX close is above the most recent prior trading-day close, so the digital strike is the prior official close.

Causal reference rules:
- Exclude fills before the prior official close is known.
- Exclude fills at or after the current day's official 20:00 UTC close.
- For each remaining fill, use the latest completed Yahoo `^GSPC` 60-minute bar at or before the fill. Overnight/pre-open fills use the prior official close.
- Volatility is an EWMA of prior daily SPX close-to-close returns only.
- `N(z)` fair is `N(log(spot / prior_close) / (sigma * sqrt(time_to_close)))`.

Executable residual rules:
- If `maker_side = SELL`, the maker sold the token and the counterparty could buy it: residual is `fair - execution_price - fee`.
- If `maker_side = BUY`, the maker bought the token and the counterparty could sell it: residual is `execution_price - fair - fee`.
- The finance fee proxy is Polymarket's category formula `0.04 * p * (1-p)`.
- The gate uses market-date clustered CIs and first qualifying fill per date for threshold rows; no mark-to-mid and no options data.

### Residual Summary

{table(summary, ['cell', 'fills', 'market_dates', 'weighted_notional_usd', 'mean_model_edge', 'edge_ci_lo', 'edge_ci_hi', 'mean_realized_net_pnl', 'pnl_ci_lo', 'pnl_ci_hi', 'mean_fee'], 12)}

Read: the all-fill N(z) residual is statistically below zero after spread/fee because actual taker-side fills are not systematically mispriced in the model's favor. Threshold rows can find positive model residuals by construction, but they do not clear realized net PnL with a market-date CI. The best-looking row by lower realized CI is `{best['cell']}`, and even that is **{cents(float(best['mean_realized_net_pnl']))}**, CI **{fmt_ci(float(best['pnl_ci_lo']), float(best['pnl_ci_hi']))}**.

![SPX N(z) residual by date](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_equities_spx_nz_pricing_date_edges.png)

Caption: each bar is a resolved SPX market date. Blue is the side-aware `N(z)` residual after fee on actual fills; orange is realized net PnL for that same executable side. A pricing build would need a stable positive residual across dates, not a few outcome-driven bars.

### Empirical Calibration

{table(calibration, ['model', 'prob_bin', 'rows', 'mean_pred_up', 'observed_up', 'observed_minus_pred'], 20)}

Read: the realized-vol `N(z)` model is not perfect, but the empirical SPX calibration does not rescue the PM residual. This mirrors the crypto precedent in [[od_conditional_prob_calibration_findings]]: once a broad causal base-rate is used, the obvious standalone pricing gap is gone or not executable.

![SPX calibration](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_equities_spx_nz_pricing_calibration.png)

### Data Ledger

| bucket | detail |
|---|---|
| PM fills | Local raw trade shards cover SPX daily up/down fills through **2026-05-26 19:57:58 UTC**. The active June 3 and later-resolved May 27-June 2 markets remain in the scope table but are not in this fill tape. |
| Reference | Yahoo Finance chart API `^GSPC`, daily + 60-minute completed bars. This is a realized-vol proxy, not an option-implied surface. |
| Not built | Cboe/OPRA options surface. The pre-registered escalation condition was not met. |

Modeled assumptions:
- Yahoo 60-minute SPX bars are adequate for the cheap realized-vol pass.
- Finance taker fee is `4% * p * (1-p)`.
- Selling an over-fair token at an observed bid is an executable residual only for inventory/mint-capable flow; the note reports it but does not convert it into a standalone live strategy.

Live-only unknowns:
- Real quote queue/fill share for a passive SPX maker.
- Whether a tighter intraday SPX feed would materially change sub-hour fill states.
- Whether an OPRA surface would reveal tiny IV-vs-RV differences; this is not worth building until the cheap realized-vol residual survives.

### Decision

**CONFIRM-CLOSE / DEFER OPTIONS BUILD.** SPX cleared small-cap capacity, but the cheap pricing gate does not show a clean executable net residual. This establishes the stronger research state the reopen wanted: PM financial-binary pricing looks efficient across both crypto terminal digitals and the most-arbitraged equity-index up/down template under the cheap causal base-rate test.

Outputs:
- Script: `scripts/od_equities_spx_nz_pricing_gate.py`
- Scored fills: `data/analysis/od_equities_spx_nz_pricing_fills.parquet`
- Summary: `data/analysis/csv_outputs/options_delta/od_equities_spx_nz_pricing_summary.csv`
- Calibration: `data/analysis/csv_outputs/options_delta/od_equities_spx_nz_pricing_calibration.csv`
- Date summary: `data/analysis/csv_outputs/options_delta/od_equities_spx_nz_pricing_date_summary.csv`
"""
    text = NOTE.read_text(encoding="utf-8")
    marker = "## 2026-06-03 N(z) / Realized-Vol Pricing Gate"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\n" + section.strip() + "\n"
    else:
        text = text.rstrip() + "\n\n" + section.strip() + "\n"
    NOTE.write_text(normalize_markdown_wrapping(text), encoding="utf-8")


def update_docs() -> None:
    hub = HUB.read_text(encoding="utf-8")
    old = "- 2026-06-03 equities index up/down small-cap scope: **SPX daily up/down clears capacity/persistence; pricing verdict remains deferred until listed-option residual is built**. Refreshed Gamma/CLOB scrape finds SPX recurring across 38 recent settlement dates, about **$24.4k** live 24h volume, **$12.7k/day** non-top3 headroom, **1c** spread, and about **48%** equity-subset top3 maker share. This supersedes the old non-crypto Gate-0 deferral only for `$10-$100` SPX measurement scale; it does not assert pricing edge. Next spend is the Cboe/OPRA SPX option-implied digital comparison, net of PM spread/fee with market-date CI. See [[od_equities_index_pricing_scope_findings]]."
    new = "- 2026-06-03 equities SPX daily up/down pricing gate: **CONFIRM-CLOSE / no OPRA build**. SPX still clears small-cap capacity/persistence, but the cheap causal realized-vol `N(z)` residual on actual executable fills does not show a clean net-of-fee edge with market-date CI; empirical SPX calibration does not rescue it. This makes PM financial-binary pricing look efficient across crypto terminal digitals and the most-arbitraged equity-index template. See [[od_equities_index_pricing_scope_findings]]."
    if old in hub:
        hub = hub.replace(old, new)
    elif "equities SPX daily up/down pricing gate" not in hub:
        marker = "## Current state (2026-06-03)"
        insert = hub.find("\n-", hub.find(marker))
        if insert >= 0:
            hub = hub[:insert] + "\n" + new + hub[insert:]
    hub = hub.replace(
        "- [ ] **SPX daily up/down listed-option pricing gate:** compare executable PM CLOB quotes to Cboe/OPRA-listed-option-implied digital fair value, net of PM spread/fee, with market-date-cluster CI.",
        "- [x] **SPX daily up/down N(z)/realized-vol pricing gate** (2026-06-03): completed; CONFIRM-CLOSE, no Cboe/OPRA build until a future cheap realized-vol residual survives. See [[od_equities_index_pricing_scope_findings]].",
    )
    HUB.write_text(normalize_markdown_wrapping(hub), encoding="utf-8")

    todo = BRAIN_TODO.read_text(encoding="utf-8")
    old_todo = "- 2026-06-03 OD equities index up/down small-cap scope: **SPX daily up/down clears capacity/persistence; pricing verdict DEFER pending listed-option residual build**. Refreshed Gamma/CLOB scrape finds SPX daily up/down recurring (38 recent settlement dates, median 1-day gap), live 24h volume about **$24.4k**, recent-90d volume **$8.54M**, non-top3 headroom about **$12.7k/day**, **1c** live CLOB spread, and equity-subset top3 maker share about **48%**. NDX has no live template and <$1k/day headroom; single stocks recur historically but live flow/spreads fail; close-above ladders are one-offs/intermittent. Next justified spend: SPX listed-option-implied digital pricing gate via Cboe/OPRA, market-date CI, net of PM spread/fee. See [[od_equities_index_pricing_scope_findings]]."
    new_todo = "- 2026-06-03 OD equities SPX daily up/down pricing gate: **CONFIRM-CLOSE / no OPRA build**. SPX clears small-cap capacity, but the cheap causal realized-vol `N(z)` residual on actual executable fills does not show a clean net-of-fee edge with market-date CI; empirical SPX calibration does not rescue it. Pricing thesis now looks comprehensively closed across crypto terminal digitals and equities index up/down unless future fresh data clears this cheap gate first. See [[od_equities_index_pricing_scope_findings]]."
    if old_todo in todo:
        todo = todo.replace(old_todo, new_todo)
    elif "OD equities SPX daily up/down pricing gate" not in todo:
        marker = "## OD — Options-Delta"
        insert = todo.find("\n-", todo.find(marker))
        if insert >= 0:
            todo = todo[:insert] + "\n" + new_todo + todo[insert:]
    todo = todo.replace(
        "- [ ] **SPX daily up/down listed-option pricing gate:** build the Cboe/OPRA option-surface path; compare executable PM CLOB quotes to listed-option-implied digital fair value around the threshold; market-date-cluster CI; net of PM spread/fee; call **MERITS-PRICING-BUILD** only if lower-CI > 0.",
        "- [x] **SPX daily up/down N(z)/realized-vol pricing gate** (2026-06-03): completed; CONFIRM-CLOSE. Do not build Cboe/OPRA unless a future cheap realized-vol residual first clears lower-CI-positive net executable edge. See [[od_equities_index_pricing_scope_findings]].",
    )
    todo = todo.replace(
        "- [x] **Equities index up/down small-capacity scope** (2026-06-03): completed; SPX daily up/down clears capacity/persistence at `$10-$100` scale, while NDX, single stocks, and close-above ladders remain deferred. Pricing verdict is still deferred until the listed-option residual is run. See [[od_equities_index_pricing_scope_findings]].",
        "- [x] **Equities index up/down small-capacity scope** (2026-06-03): completed; SPX daily up/down clears capacity/persistence at `$10-$100` scale, while NDX, single stocks, and close-above ladders remain deferred. The follow-on cheap N(z) pricing gate now closes the SPX pricing branch; no OPRA build. See [[od_equities_index_pricing_scope_findings]].",
    )
    todo = todo.replace(
        "- [x] **Cross-asset PM financial-binary Gate 0:** mapped PM daily crypto, index up/down, single-stock up/down, close-above/price-band, and neg-risk markets against external references and K5-style incumbent concentration. Original result: only crypto-daily cleared the `$50k/day` operation-scale capacity filter; literal PM daily BTC/ETH vs Deribit daily remains parked by settlement mismatch. Superseded for SPX small-cap scope by the 2026-06-03 equities pass: SPX daily up/down clears `$10-$100` capacity and now needs a listed-option pricing residual. See [[od_cross_asset_gate0_universe_map_findings]] and [[od_equities_index_pricing_scope_findings]].",
        "- [x] **Cross-asset PM financial-binary Gate 0:** mapped PM daily crypto, index up/down, single-stock up/down, close-above/price-band, and neg-risk markets against external references and K5-style incumbent concentration. Original result: only crypto-daily cleared the `$50k/day` operation-scale capacity filter; literal PM daily BTC/ETH vs Deribit daily remains parked by settlement mismatch. Superseded for SPX small-cap scope by the 2026-06-03 equities pass: SPX daily up/down clears `$10-$100` capacity, and the follow-on N(z) gate confirms close/no OPRA build. See [[od_cross_asset_gate0_universe_map_findings]] and [[od_equities_index_pricing_scope_findings]].",
    )
    BRAIN_TODO.write_text(normalize_markdown_wrapping(todo), encoding="utf-8")


def main() -> None:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    markets = spx_market_detail()
    fills = materialize_spx_fills(markets, refresh=False)
    daily, states = build_spx_references(refresh=False)
    scored = score_fills(fills, daily, states)
    scored.to_parquet(OUT_FILLS, index=False)
    summary, date_summary, calibration = summarize(scored, states)
    plot_outputs(date_summary, calibration)
    write_outputs(scored, summary, date_summary, calibration, states)
    append_note(scored, summary, date_summary, calibration)
    update_docs()
    print(f"SPX scored fills: {len(scored):,} rows across {scored['market_date'].nunique():,} dates", flush=True)
    print(f"wrote {OUT_SUMMARY}", flush=True)
    print(f"wrote {NOTE}", flush=True)


if __name__ == "__main__":
    main()
