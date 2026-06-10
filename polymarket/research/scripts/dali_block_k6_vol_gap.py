"""Block K6: Polymarket implied-vs-realized vol diagnostic.

This script reuses the K3 v3-H 1s feature cache and separates causal
comparisons from ex-post realized-vol diagnostics.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import YEAR_SECONDS, cents, markdown_table, number, pct
from dali_block_k3v3h_hedged_basis import (
    BINANCE_HEDGE_COST_BPS,
    BOOTSTRAP_SAMPLES,
    FEATURE_CACHE,
    RNG_SEED,
    load_features,
    normal_pdf,
    taker_fee,
    timestamp_ns,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"

K3V1_CSV = ANALYSIS / "csv_outputs" / "options_delta" / "k3_leadlag_basis.csv"
OUT_CSV = ANALYSIS / "csv_outputs" / "options_delta" / "k6_vol_gap.csv"
OUT_PANEL = ANALYSIS / "k6_vol_gap_panel.parquet"
OUT_TRADES = ANALYSIS / "csv_outputs" / "options_delta" / "k6_gamma_scalp_trades.csv"
NOTE = NOTES / "block_k6_vol_findings.md"

MIN_PRICE = 1e-5
MIN_TAU_SECONDS = 60.0
ENTRY_LATENCIES = (1, 3, 5)
ENTRY_VOL_GAPS = (0.05, 0.10, 0.20)
REHEDGE_BANDS = (0.01, 0.03, 0.05, 0.10)
SOURCE_FILTERS = ("all", "strict")
ROBUST_MIN_TRADES = 5
NORM = NormalDist()


@dataclass
class GammaTrade:
    source_filter: str
    latency_s: int
    entry_vol_gap: float
    rehedge_band: float
    asset: str
    market_slug: str
    state_bucket: str
    moneyness_bucket: str
    time_bucket: str
    route: str
    vol_side: str
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    hold_seconds: float
    seconds_to_expiry_entry: float
    entry_abs_z: float
    entry_log_moneyness: float
    entry_pm_iv: float
    entry_ewma_sigma: float
    entry_trailing_sigma: float
    entry_remaining_rv: float
    entry_full_window_rv: float
    entry_iv_minus_ewma: float
    pm_entry_price: float
    pm_entry_fee: float
    payoff: float
    digital_pnl: float
    hedge_pnl: float
    hedge_turnover_notional: float
    hedge_cost: float
    hedge_rebalances: int
    net_pnl: float
    unhedged_pnl: float
    source_ok: bool
    source_disagree: bool
    settlement_margin_bp: float
    large_static_entry: bool
    exit_capture_gap_seconds: float


def volpct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f} vol pts"


def ci_text(lo: float, hi: float, digits: int = 4) -> str:
    return f"[{number(lo, digits)}, {number(hi, digits)}]"


def norm_ppf_array(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, MIN_PRICE, 1.0 - MIN_PRICE)
    return np.fromiter((NORM.inv_cdf(float(x)) for x in clipped), dtype=float, count=len(clipped))


def bootstrap_market_mean(data: pd.DataFrame, col: str, seed_offset: int = 0) -> tuple[float, float]:
    finite = data[np.isfinite(data[col].astype(float))]
    if finite.empty:
        return math.nan, math.nan
    means = finite.groupby("market_slug", sort=False)[col].mean().to_numpy(dtype=float)
    means = means[np.isfinite(means)]
    if len(means) == 0:
        return math.nan, math.nan
    if len(means) == 1:
        return float(means[0]), float(means[0])
    rng = np.random.default_rng(RNG_SEED + seed_offset + len(finite))
    draws = rng.integers(0, len(means), size=(BOOTSTRAP_SAMPLES, len(means)))
    vals = means[draws].mean(axis=1)
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def load_full_window_rv() -> pd.Series:
    if not K3V1_CSV.exists():
        return pd.Series(dtype=float, name="k3v1_full_window_rv")
    usecols = ["market_slug", "window_rv_annualized"]
    full = pd.read_csv(K3V1_CSV, usecols=usecols)
    rv = full.groupby("market_slug", sort=False)["window_rv_annualized"].first()
    rv.name = "k3v1_full_window_rv"
    return rv


def add_implied_vol(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["market_slug", "ts"]).reset_index(drop=True).copy()
    spot = df["binance_spot"].astype(float).to_numpy()
    strike = df["binance_strike_spot"].astype(float).to_numpy()
    tau_years = (df["seconds_to_expiry"].astype(float).clip(lower=1.0) / YEAR_SECONDS).to_numpy()
    pm = df["polymarket_mid"].astype(float).to_numpy()

    log_m = np.log(spot / strike)
    q = norm_ppf_array(pm)
    sqrt_tau = np.sqrt(tau_years)
    abs_m = np.abs(log_m)
    abs_q = np.abs(q)
    sign_consistent = np.sign(log_m) * np.sign(q) > 0
    base_valid = (
        np.isfinite(spot)
        & np.isfinite(strike)
        & (spot > 0)
        & (strike > 0)
        & np.isfinite(tau_years)
        & (tau_years > 0)
        & np.isfinite(pm)
        & (pm > 0)
        & (pm < 1)
    )
    identifiable = base_valid & (abs_m > 1e-10) & (abs_q > 1e-8)
    valid = identifiable & sign_consistent

    sigma_pm = np.full(len(df), np.nan, dtype=float)
    sigma_pm[valid] = abs_m[valid] / (abs_q[valid] * sqrt_tau[valid])
    sigma_pm[(sigma_pm <= 0) | ~np.isfinite(sigma_pm)] = np.nan

    status = np.full(len(df), "valid", dtype=object)
    status[~base_valid] = "bad_input"
    status[base_valid & (abs_m <= 1e-10) & (abs_q <= 1e-8)] = "atm_underdetermined"
    status[base_valid & (abs_m <= 1e-10) & (abs_q > 1e-8)] = "zero_moneyness_price_off_half"
    status[base_valid & (abs_m > 1e-10) & (abs_q <= 1e-8)] = "implied_infinite_at_half"
    status[identifiable & ~sign_consistent] = "no_positive_solution"
    status[valid] = "valid"

    delta_pm = np.full(len(df), np.nan, dtype=float)
    delta_pm[valid] = normal_pdf(q[valid]) / (spot[valid] * sigma_pm[valid] * sqrt_tau[valid])

    df["log_spot_moneyness"] = log_m
    df["pm_norm_quantile"] = q
    df["pm_mid_implied_vol_annualized"] = sigma_pm
    df["pm_mid_iv_status"] = status
    df["pm_mid_iv_valid"] = status == "valid"
    df["pm_mid_implied_delta_up"] = delta_pm
    df["pm_iv_annualized"] = df["pm_mid_implied_vol_annualized"]  # Legacy alias: PM midpoint inverted to vol.
    df["pm_iv_status"] = df["pm_mid_iv_status"]
    df["pm_iv_valid"] = df["pm_mid_iv_valid"]
    df["pm_delta_up_implied"] = df["pm_mid_implied_delta_up"]
    df["pm_mid_iv_minus_ewma"] = df["pm_mid_implied_vol_annualized"] - df["ewma_sigma_annualized"]
    df["pm_mid_iv_minus_trailing"] = df["pm_mid_implied_vol_annualized"] - df["trailing_sigma_annualized"]
    df["iv_minus_ewma"] = df["pm_mid_iv_minus_ewma"]  # Legacy alias.
    df["iv_minus_trailing"] = df["pm_mid_iv_minus_trailing"]
    return df


def add_realized_vol(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["market_slug", "ts"]).reset_index(drop=True).copy()
    rem = np.full(len(df), np.nan, dtype=float)
    captured_full = np.full(len(df), np.nan, dtype=float)
    exit_gap = np.full(len(df), np.nan, dtype=float)

    for _, idx in df.groupby("market_slug", sort=False).groups.items():
        loc = np.asarray(idx, dtype=int)
        g = df.iloc[loc]
        spot = g["binance_spot"].astype(float).to_numpy()
        ts_ns = timestamp_ns(g["ts"])
        if len(g) < 3 or not np.all(np.isfinite(spot)) or np.any(spot <= 0):
            continue
        log_spot = np.log(spot)
        ret = np.diff(log_spot)
        sq = ret * ret
        duration = max(float((ts_ns[-1] - ts_ns[0]) / 1_000_000_000.0), 1.0)
        full_sigma = math.sqrt(float(np.nansum(sq)) / duration * YEAR_SECONDS)
        captured_full[loc] = full_sigma

        future_sq = np.zeros(len(g), dtype=float)
        future_sq[:-1] = np.cumsum(sq[::-1])[::-1]
        remaining_seconds = np.maximum((ts_ns[-1] - ts_ns) / 1_000_000_000.0, 1.0)
        rem_sigma = np.sqrt(future_sq / remaining_seconds * YEAR_SECONDS)
        rem_sigma[-1] = np.nan
        rem[loc] = rem_sigma

        window_end = pd.to_datetime(g["window_end"], utc=True)
        last_ts = pd.to_datetime(g["ts"].iloc[-1], utc=True)
        gap_seconds = float((window_end.iloc[-1] - last_ts).total_seconds())
        exit_gap[loc] = gap_seconds

    df["remaining_captured_rv_annualized"] = rem
    df["captured_path_rv_annualized"] = captured_full
    df["exit_capture_gap_seconds"] = exit_gap

    full_rv = load_full_window_rv()
    if not full_rv.empty:
        df = df.merge(full_rv, left_on="market_slug", right_index=True, how="left")
        df["full_window_rv_annualized"] = df["k3v1_full_window_rv"].fillna(df["captured_path_rv_annualized"])
        df = df.drop(columns=["k3v1_full_window_rv"])
    else:
        df["full_window_rv_annualized"] = df["captured_path_rv_annualized"]

    df["pm_mid_iv_minus_remaining_captured_rv"] = df["pm_mid_implied_vol_annualized"] - df["remaining_captured_rv_annualized"]
    df["pm_mid_iv_minus_full_window_rv"] = df["pm_mid_implied_vol_annualized"] - df["full_window_rv_annualized"]
    df["iv_minus_remaining_captured_rv"] = df["pm_mid_iv_minus_remaining_captured_rv"]  # Legacy alias.
    df["iv_minus_full_window_rv"] = df["pm_mid_iv_minus_full_window_rv"]
    return df


def build_panel(refresh_features: bool = False, refresh_panel: bool = False) -> pd.DataFrame:
    if OUT_PANEL.exists() and not refresh_panel and not refresh_features:
        print(f"loading vol panel cache {OUT_PANEL}", flush=True)
        return pd.read_parquet(OUT_PANEL)
    base = load_features(refresh_features=refresh_features)
    print("inverting digital prices to implied vol", flush=True)
    panel = add_implied_vol(base)
    print("adding ex-post remaining/full-window realized vol diagnostics", flush=True)
    panel = add_realized_vol(panel)
    panel.to_parquet(OUT_PANEL, index=False)
    print(f"wrote {OUT_PANEL}", flush=True)
    return panel


def finite_mean(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else math.nan


def summarize_vol_gaps(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    samples = {
        "all_rows": pd.Series(True, index=panel.index),
        "clean_source_no_static": panel["source_ok_strict"].fillna(False).astype(bool)
        & ~panel["large_static_basis_10c"].fillna(False).astype(bool)
        & ~panel["toxic_near_expiry"].fillna(False).astype(bool),
    }
    cols_for_ci = [
        "iv_minus_ewma",
        "iv_minus_trailing",
        "iv_minus_remaining_captured_rv",
        "iv_minus_full_window_rv",
    ]
    for sample_name, mask in samples.items():
        sample = panel[mask].copy()
        for state_bucket, g in sample.groupby("state_bucket", sort=True):
            valid = g[g["pm_mid_iv_valid"].fillna(False)]
            status_counts = g["pm_mid_iv_status"].value_counts(normalize=True)
            row: dict[str, Any] = {
                "row_type": "vol_gap_bucket",
                "sample": sample_name,
                "state_bucket": state_bucket,
                "moneyness_bucket": str(g["moneyness_bucket"].iloc[0]),
                "time_bucket": str(g["time_bucket"].iloc[0]),
                "n_rows": int(len(g)),
                "n_markets": int(g["market_slug"].nunique()),
                "n_iv_valid": int(len(valid)),
                "iv_valid_share": float(len(valid) / len(g)) if len(g) else math.nan,
                "no_positive_solution_share": float(status_counts.get("no_positive_solution", 0.0)),
                "bad_or_underdetermined_share": float(1.0 - status_counts.get("valid", 0.0)),
                "mean_pm_mid_implied_vol": finite_mean(valid["pm_mid_implied_vol_annualized"]),
                "median_pm_mid_implied_vol": float(valid["pm_mid_implied_vol_annualized"].median()) if len(valid) else math.nan,
                "mean_pm_iv": finite_mean(valid["pm_iv_annualized"]),
                "median_pm_iv": float(valid["pm_iv_annualized"].median()) if len(valid) else math.nan,
                "mean_ewma_sigma": finite_mean(valid["ewma_sigma_annualized"]),
                "mean_trailing_sigma": finite_mean(valid["trailing_sigma_annualized"]),
                "mean_remaining_captured_rv": finite_mean(valid["remaining_captured_rv_annualized"]),
                "mean_full_window_rv": finite_mean(valid["full_window_rv_annualized"]),
                "mean_abs_iv_minus_ewma": finite_mean(valid["iv_minus_ewma"].abs()),
                "p95_abs_iv_minus_ewma": float(valid["iv_minus_ewma"].abs().quantile(0.95)) if len(valid) else math.nan,
                "mean_iv_minus_ewma": finite_mean(valid["iv_minus_ewma"]),
                "mean_iv_minus_trailing": finite_mean(valid["iv_minus_trailing"]),
                "mean_iv_minus_remaining_captured_rv": finite_mean(valid["iv_minus_remaining_captured_rv"]),
                "mean_iv_minus_full_window_rv": finite_mean(valid["iv_minus_full_window_rv"]),
            }
            for offset, col in enumerate(cols_for_ci):
                lo, hi = bootstrap_market_mean(valid, col, seed_offset=offset * 1000 + len(state_bucket))
                row[f"{col}_ci_lo"] = lo
                row[f"{col}_ci_hi"] = hi
            rows.append(row)
    return pd.DataFrame(rows)


def resolution_up_value(g: pd.DataFrame) -> bool:
    chain = g["chainlink_resolution_up"].dropna()
    if not chain.empty:
        return bool(chain.iloc[0])
    return bool(g["binance_resolution_up"].dropna().iloc[0])


def market_arrays(g: pd.DataFrame) -> dict[str, Any]:
    g = g.sort_values("ts").reset_index(drop=True)
    resolution_up = resolution_up_value(g)
    return {
        "ts": pd.to_datetime(g["ts"], utc=True).to_numpy(),
        "ts_ns": timestamp_ns(g["ts"]),
        "spot": g["binance_spot"].to_numpy(dtype=float),
        "delta": g["digital_delta"].to_numpy(dtype=float),
        "pm_iv_valid": g["pm_mid_iv_valid"].to_numpy(dtype=bool),
        "iv_minus_ewma": g["iv_minus_ewma"].to_numpy(dtype=float),
        "pm_iv": g["pm_mid_implied_vol_annualized"].to_numpy(dtype=float),
        "ewma": g["ewma_sigma_annualized"].to_numpy(dtype=float),
        "trailing": g["trailing_sigma_annualized"].to_numpy(dtype=float),
        "remaining_rv": g["remaining_captured_rv_annualized"].to_numpy(dtype=float),
        "full_rv": g["full_window_rv_annualized"].to_numpy(dtype=float),
        "log_m": g["log_spot_moneyness"].to_numpy(dtype=float),
        "abs_z": g["abs_z"].to_numpy(dtype=float),
        "seconds_to_expiry": g["seconds_to_expiry"].to_numpy(dtype=float),
        "up_ask": g["up_ask"].to_numpy(dtype=float),
        "down_ask": g["down_ask"].to_numpy(dtype=float),
        "fee_rate": g["taker_fee_rate"].fillna(0.07).to_numpy(dtype=float),
        "large_static": g["large_static_basis_10c"].fillna(False).to_numpy(dtype=bool),
        "toxic": g["toxic_near_expiry"].fillna(False).to_numpy(dtype=bool),
        "state_bucket": g["state_bucket"].astype(str).to_numpy(),
        "moneyness_bucket": g["moneyness_bucket"].astype(str).to_numpy(),
        "time_bucket": g["time_bucket"].astype(str).to_numpy(),
        "source_ok": bool(g["source_ok_strict"].iloc[0]),
        "source_disagree": bool(g["chainlink_binance_resolution_disagree"].iloc[0]),
        "settlement_margin_bp": float(g["binance_window_abs_return_bps"].iloc[0]),
        "asset": str(g["asset"].iloc[0]),
        "market_slug": str(g["market_slug"].iloc[0]),
        "resolution_up": resolution_up,
        "window_end": pd.to_datetime(g["window_end"].iloc[-1], utc=True),
        "exit_capture_gap_seconds": float(g["exit_capture_gap_seconds"].iloc[0]),
    }


def trade_route(log_m: float, iv_gap: float) -> tuple[str, str] | None:
    if not np.isfinite(log_m) or not np.isfinite(iv_gap) or abs(log_m) <= 1e-10:
        return None
    if iv_gap < 0:
        return ("buy_up", "long_vol") if log_m < 0 else ("buy_down", "long_vol")
    return ("buy_up", "short_vol_proxy") if log_m > 0 else ("buy_down", "short_vol_proxy")


def banded_hedge_path(
    arr: dict[str, Any],
    route: str,
    entry_pos: int,
    exit_pos: int,
    band_notional: float,
) -> tuple[float, float, float, int]:
    if exit_pos <= entry_pos:
        return 0.0, 0.0, 0.0, 0
    spot = arr["spot"][entry_pos : exit_pos + 1]
    delta = arr["delta"][entry_pos : exit_pos + 1]
    finite_delta = np.where(np.isfinite(delta), delta, 0.0)
    target = -finite_delta if route == "buy_up" else finite_delta

    current = float(target[0])
    turnover = abs(current) * float(spot[0])
    hedge_pnl = 0.0
    rebalances = 0
    for i in range(1, len(spot)):
        if not np.isfinite(spot[i]) or not np.isfinite(spot[i - 1]):
            continue
        hedge_pnl += current * float(spot[i] - spot[i - 1])
        target_i = float(target[i])
        rebalance_notional = abs(target_i - current) * float(spot[i])
        if np.isfinite(rebalance_notional) and rebalance_notional >= band_notional:
            turnover += rebalance_notional
            current = target_i
            rebalances += 1
    if np.isfinite(spot[-1]):
        turnover += abs(current) * float(spot[-1])
    hedge_cost = turnover * BINANCE_HEDGE_COST_BPS / 10000.0
    return float(hedge_pnl), float(turnover), float(hedge_cost), int(rebalances)


def build_trade(
    arr: dict[str, Any],
    *,
    source_filter: str,
    latency_s: int,
    entry_vol_gap: float,
    rehedge_band: float,
) -> list[GammaTrade]:
    if source_filter == "strict" and not arr["source_ok"]:
        return []
    n = len(arr["spot"])
    if n <= latency_s + 3:
        return []
    exit_pos = n - 1
    trades: list[GammaTrade] = []
    seen_buckets: set[str] = set()
    for signal_pos in range(0, n - latency_s - 2):
        bucket = str(arr["state_bucket"][signal_pos])
        if bucket in seen_buckets:
            continue
        if (
            not arr["pm_iv_valid"][signal_pos]
            or bool(arr["toxic"][signal_pos])
            or bool(arr["large_static"][signal_pos])
            or float(arr["seconds_to_expiry"][signal_pos]) <= MIN_TAU_SECONDS
        ):
            continue
        iv_gap = float(arr["iv_minus_ewma"][signal_pos])
        if not np.isfinite(iv_gap) or abs(iv_gap) < entry_vol_gap:
            continue
        route_side = trade_route(float(arr["log_m"][signal_pos]), iv_gap)
        if route_side is None:
            continue
        route, vol_side = route_side
        entry_pos = signal_pos + latency_s
        if (
            entry_pos >= exit_pos
            or not arr["pm_iv_valid"][entry_pos]
            or bool(arr["toxic"][entry_pos])
            or bool(arr["large_static"][entry_pos])
            or float(arr["seconds_to_expiry"][entry_pos]) <= MIN_TAU_SECONDS
        ):
            continue
        if route == "buy_up":
            entry_price = float(arr["up_ask"][entry_pos])
            payoff = 1.0 if arr["resolution_up"] else 0.0
        else:
            entry_price = float(arr["down_ask"][entry_pos])
            payoff = 0.0 if arr["resolution_up"] else 1.0
        if not np.isfinite(entry_price):
            continue
        fee_rate = float(arr["fee_rate"][entry_pos]) if np.isfinite(float(arr["fee_rate"][entry_pos])) else 0.07
        pm_fee = taker_fee(entry_price, fee_rate)
        digital_pnl = payoff - entry_price - pm_fee
        hedge_pnl, turnover, hedge_cost, rebalances = banded_hedge_path(
            arr, route, entry_pos, exit_pos, rehedge_band
        )
        net_pnl = digital_pnl + hedge_pnl - hedge_cost
        hold_seconds = float((arr["ts_ns"][exit_pos] - arr["ts_ns"][entry_pos]) / 1_000_000_000.0)
        trades.append(
            GammaTrade(
            source_filter=source_filter,
            latency_s=latency_s,
            entry_vol_gap=entry_vol_gap,
            rehedge_band=rehedge_band,
            asset=arr["asset"],
            market_slug=arr["market_slug"],
            state_bucket=str(arr["state_bucket"][entry_pos]),
            moneyness_bucket=str(arr["moneyness_bucket"][entry_pos]),
            time_bucket=str(arr["time_bucket"][entry_pos]),
            route=route,
            vol_side=vol_side,
            signal_ts=pd.Timestamp(arr["ts"][signal_pos]),
            entry_ts=pd.Timestamp(arr["ts"][entry_pos]),
            exit_ts=pd.Timestamp(arr["ts"][exit_pos]),
            hold_seconds=hold_seconds,
            seconds_to_expiry_entry=float(arr["seconds_to_expiry"][entry_pos]),
            entry_abs_z=float(arr["abs_z"][entry_pos]),
            entry_log_moneyness=float(arr["log_m"][entry_pos]),
            entry_pm_iv=float(arr["pm_iv"][entry_pos]),
            entry_ewma_sigma=float(arr["ewma"][entry_pos]),
            entry_trailing_sigma=float(arr["trailing"][entry_pos]),
            entry_remaining_rv=float(arr["remaining_rv"][entry_pos]),
            entry_full_window_rv=float(arr["full_rv"][entry_pos]),
            entry_iv_minus_ewma=float(arr["iv_minus_ewma"][entry_pos]),
            pm_entry_price=entry_price,
            pm_entry_fee=pm_fee,
            payoff=payoff,
            digital_pnl=digital_pnl,
            hedge_pnl=hedge_pnl,
            hedge_turnover_notional=turnover,
            hedge_cost=hedge_cost,
            hedge_rebalances=rebalances,
            net_pnl=net_pnl,
            unhedged_pnl=digital_pnl,
            source_ok=bool(arr["source_ok"]),
            source_disagree=bool(arr["source_disagree"]),
            settlement_margin_bp=float(arr["settlement_margin_bp"]),
            large_static_entry=bool(arr["large_static"][entry_pos]),
            exit_capture_gap_seconds=float(arr["exit_capture_gap_seconds"]),
            )
        )
        seen_buckets.add(bucket)
    return trades


def simulate_gamma(panel: pd.DataFrame, refresh_trades: bool = False) -> pd.DataFrame:
    if OUT_TRADES.exists() and not refresh_trades:
        print(f"loading gamma trade cache {OUT_TRADES}", flush=True)
        return pd.read_csv(OUT_TRADES, parse_dates=["signal_ts", "entry_ts", "exit_ts"])
    groups = [market_arrays(g) for _, g in panel.groupby("market_slug", sort=False)]
    combos = [
        (source_filter, latency, entry_gap, band)
        for source_filter in SOURCE_FILTERS
        for latency in ENTRY_LATENCIES
        for entry_gap in ENTRY_VOL_GAPS
        for band in REHEDGE_BANDS
    ]
    trades: list[GammaTrade] = []
    for i, (source_filter, latency, entry_gap, band) in enumerate(combos, start=1):
        print(
            f"gamma sim {i}/{len(combos)} source={source_filter} latency={latency}s "
            f"entry_gap={entry_gap:.2f} band={band:.2f}",
            flush=True,
        )
        for arr in groups:
            new_trades = build_trade(
                arr,
                source_filter=source_filter,
                latency_s=latency,
                entry_vol_gap=entry_gap,
                rehedge_band=band,
            )
            trades.extend(new_trades)
    out = pd.DataFrame([t.__dict__ for t in trades])
    out.to_csv(OUT_TRADES, index=False)
    print(f"wrote {OUT_TRADES}", flush=True)
    return out


def summarize_gamma(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_specs = [
        ("gamma_config", ["source_filter", "latency_s", "entry_vol_gap", "rehedge_band"]),
        ("gamma_bucket", ["source_filter", "latency_s", "entry_vol_gap", "rehedge_band", "state_bucket"]),
    ]
    for row_type, keys in group_specs:
        for key, g in trades.groupby(keys, sort=True):
            if len(g) < ROBUST_MIN_TRADES:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            row = {k: v for k, v in zip(keys, key, strict=True)}
            row.update(
                {
                    "row_type": row_type,
                    "sample": str(row.get("source_filter", "")),
                    "n_trades": int(len(g)),
                    "n_markets": int(g["market_slug"].nunique()),
                    "mean_net_pnl": float(g["net_pnl"].mean()),
                    "mean_unhedged_pnl": float(g["unhedged_pnl"].mean()),
                    "mean_digital_pnl": float(g["digital_pnl"].mean()),
                    "mean_hedge_pnl": float(g["hedge_pnl"].mean()),
                    "mean_hedge_cost": float(g["hedge_cost"].mean()),
                    "mean_pm_fee": float(g["pm_entry_fee"].mean()),
                    "mean_turnover_notional": float(g["hedge_turnover_notional"].mean()),
                    "mean_rebalances": float(g["hedge_rebalances"].mean()),
                    "win_rate_net": float((g["net_pnl"] > 0).mean()),
                    "median_hold_seconds": float(g["hold_seconds"].median()),
                    "p95_hold_seconds": float(g["hold_seconds"].quantile(0.95)),
                    "long_vol_share": float((g["vol_side"] == "long_vol").mean()),
                    "tail_market_share": float(g["market_slug"].value_counts(normalize=True).iloc[0]),
                }
            )
            net_lo, net_hi = bootstrap_market_mean(g, "net_pnl", seed_offset=20_000 + len(keys))
            un_lo, un_hi = bootstrap_market_mean(g, "unhedged_pnl", seed_offset=30_000 + len(keys))
            row["net_pnl_ci_lo"] = net_lo
            row["net_pnl_ci_hi"] = net_hi
            row["unhedged_pnl_ci_lo"] = un_lo
            row["unhedged_pnl_ci_hi"] = un_hi
            rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(panel: pd.DataFrame, trades: pd.DataFrame, summary: pd.DataFrame) -> None:
    vol_summary = summarize_vol_gaps(panel)
    gamma_summary = summarize_gamma(trades)
    combined = pd.concat([vol_summary, gamma_summary, summary], ignore_index=True, sort=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}", flush=True)


def top_clean_bucket_rows(vol_summary: pd.DataFrame) -> list[list[str]]:
    clean = vol_summary[
        vol_summary["row_type"].eq("vol_gap_bucket") & vol_summary["sample"].eq("clean_source_no_static")
    ].copy()
    clean = clean.sort_values(["state_bucket"])
    rows: list[list[str]] = []
    for _, row in clean.iterrows():
        rows.append(
            [
                str(row["state_bucket"]),
                str(int(row["n_rows"])),
                pct(float(row["iv_valid_share"])),
                volpct(float(row["mean_pm_iv"])),
                volpct(float(row["mean_ewma_sigma"])),
                volpct(float(row["mean_iv_minus_ewma"])),
                ci_text(float(row["iv_minus_ewma_ci_lo"]), float(row["iv_minus_ewma_ci_hi"])),
                volpct(float(row["mean_remaining_captured_rv"])),
                volpct(float(row["mean_iv_minus_remaining_captured_rv"])),
                volpct(float(row["mean_full_window_rv"])),
                volpct(float(row["mean_iv_minus_full_window_rv"])),
            ]
        )
    return rows


def gamma_rows(gamma_summary: pd.DataFrame) -> list[list[str]]:
    if gamma_summary.empty:
        return []
    strict = gamma_summary[
        gamma_summary["row_type"].eq("gamma_bucket") & gamma_summary["source_filter"].eq("strict")
    ].copy()
    strict = strict.sort_values(["net_pnl_ci_lo", "mean_net_pnl"], ascending=[False, False]).head(10)
    rows: list[list[str]] = []
    for _, row in strict.iterrows():
        rows.append(
            [
                str(row["state_bucket"]),
                str(int(row["latency_s"])),
                volpct(float(row["entry_vol_gap"])),
                cents(float(row["rehedge_band"])),
                str(int(row["n_trades"])),
                cents(float(row["mean_net_pnl"])),
                ci_text(float(row["net_pnl_ci_lo"]), float(row["net_pnl_ci_hi"])),
                cents(float(row["mean_unhedged_pnl"])),
                cents(float(row["mean_pm_fee"])),
                cents(float(row["mean_hedge_cost"])),
                pct(float(row["win_rate_net"])),
                number(float(row["median_hold_seconds"]), 0),
            ]
        )
    return rows


def write_note(panel: pd.DataFrame, trades: pd.DataFrame, vol_summary: pd.DataFrame, gamma_summary: pd.DataFrame) -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    valid_share = float(panel["pm_mid_iv_valid"].mean())
    no_solution_share = float((panel["pm_mid_iv_status"] == "no_positive_solution").mean())
    clean_mask = (
        panel["source_ok_strict"].fillna(False).astype(bool)
        & ~panel["large_static_basis_10c"].fillna(False).astype(bool)
        & ~panel["toxic_near_expiry"].fillna(False).astype(bool)
        & panel["pm_mid_iv_valid"].fillna(False).astype(bool)
    )
    clean = panel[clean_mask]
    clean_iv_gap = float(clean["iv_minus_ewma"].mean()) if len(clean) else math.nan
    clean_expost_gap = float(clean["iv_minus_remaining_captured_rv"].mean()) if len(clean) else math.nan

    positive = pd.DataFrame()
    if not gamma_summary.empty:
        positive = gamma_summary[
            gamma_summary["row_type"].eq("gamma_bucket")
            & gamma_summary["source_filter"].eq("strict")
            & (gamma_summary["n_trades"] >= ROBUST_MIN_TRADES)
            & (gamma_summary["net_pnl_ci_lo"] > 0)
        ].copy()
    best = pd.DataFrame()
    if not gamma_summary.empty:
        best = gamma_summary[
            gamma_summary["row_type"].eq("gamma_bucket") & gamma_summary["source_filter"].eq("strict")
        ].sort_values(["net_pnl_ci_lo", "mean_net_pnl"], ascending=[False, False]).head(1)

    if not positive.empty:
        headline = (
            "A strict-source gamma/vol bucket clears zero after costs, but treat it as provisional "
            "because this is a small-window diagnostic."
        )
    else:
        headline = (
            "No strict-source (|z|, tau) bucket clears zero after Polymarket fee plus banded Binance "
            "hedge turnover; the vol branch does not pass the falsifier on this IS panel."
        )

    if not best.empty:
        b = best.iloc[0]
        best_text = (
            f"Best strict bucket/config by lower CI is `{b['state_bucket']}` at latency "
            f"{int(b['latency_s'])}s, entry gap {volpct(float(b['entry_vol_gap']))}, "
            f"band {cents(float(b['rehedge_band']))}: mean net {cents(float(b['mean_net_pnl']))}, "
            f"CI {ci_text(float(b['net_pnl_ci_lo']), float(b['net_pnl_ci_hi']))}."
        )
    else:
        best_text = "No gamma trades survived the strict diagnostic filters."

    invalid_table = (
        panel["pm_mid_iv_status"].value_counts(normalize=True).rename("share").reset_index().rename(columns={"index": "status"})
    )
    invalid_rows = [[str(r["pm_mid_iv_status"]), pct(float(r["share"]))] for _, r in invalid_table.head(8).iterrows()]

    note = f"""# Block K6 Vol Gap Diagnostic

## Headline

{headline}

Clean-source rows imply PM midpoint-implied vol minus causal EWMA of **{volpct(clean_iv_gap)}** on average; the ex-post remaining captured-path diagnostic is **{volpct(clean_expost_gap)}**. The sign is useful diagnostically, but the tradable test is the banded delta-hedged simulation below.

{best_text}

## Inversion Method

For each row I invert the PM midpoint through the European digital model `P_up = N(log(S/K)/(sigma*sqrt(tau)))`, with `K` from the Binance window-open reference in the K3 panel. This `pm_mid_implied_vol_annualized` is a diagnostic representation of the PM price, not external option-IV fair. A positive finite implied vol exists only when the PM probability is on the same side of 50% as Binance moneyness. Rows that violate that are marked `no_positive_solution`; they are not forced into a bogus sigma.

Implied-vol validity:

{markdown_table(["status", "share"], invalid_rows)}

## Causal Vol Gap By Bucket

This table uses the clean source sample: strict Chainlink/Pyth-vs-Binance settlement filter, no large static 10c basis rows, and no toxic near-strike/near-expiry rows. `Remaining RV` and `full-window RV` columns are diagnostic/lookahead only.

{markdown_table(
    [
        "bucket",
        "rows",
        "valid IV",
        "PM mid-IV",
        "EWMA",
        "PM-EWMA",
        "PM-EWMA CI",
        "remaining RV",
        "PM-rem RV",
        "full RV",
        "PM-full RV",
    ],
    top_clean_bucket_rows(vol_summary),
)}

## Gamma-Scalp Backtest

Rules: for each market/bucket/config, take the first eligible entry on the side implied by the causal PM-mid-IV-minus-EWMA gap; buy the OTM/positive-vega side when PM midpoint-implied IV is cheap, or the ITM/negative-vega proxy when PM midpoint-implied IV is rich. Hold to resolution, delta-hedge on Binance using the causal K3 digital delta, and rebalance only when target hedge notional moves by the band. Costs are the PM taker fee at entry plus Binance hedge turnover at {BINANCE_HEDGE_COST_BPS:.1f}bp. Entries exclude invalid PM-mid-IV, large static basis, and toxic near-expiry rows. Bucket entries are diagnostic and may overlap across buckets within the same market.

{markdown_table(
    [
        "bucket",
        "lat",
        "entry gap",
        "band",
        "trades",
        "net",
        "net CI",
        "unhedged",
        "PM fee",
        "hedge cost",
        "win",
        "median hold",
    ],
    gamma_rows(gamma_summary),
)}

## Caveats

- The causal comparison is PM midpoint-implied IV versus trailing/EWMA vol available at time `t`; it is not an external option-IV fair.
- The remaining-window and full-window realized vol columns are explicitly ex-post diagnostics.
- The full-window RV is pulled from the earlier K3 full-window pass when present; remaining RV is computed on the captured Binance path after each row, so it can understate unobserved pre/post-capture variance.
- The trade ledger uses Chainlink/Pyth settlement when available, and the strict source filter removes direction disagreements and small Binance settlement margins.

Outputs:

- `{OUT_CSV.relative_to(ROOT)}`
- `{OUT_PANEL.relative_to(ROOT)}`
- `{OUT_TRADES.relative_to(ROOT)}`
"""
    NOTE.write_text(note, encoding="utf-8")
    print(f"wrote {NOTE}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-features", action="store_true", help="Rebuild the K3 v3-H feature cache.")
    parser.add_argument("--refresh-panel", action="store_true", help="Rebuild implied-vol/realized-vol panel cache.")
    parser.add_argument("--refresh-trades", action="store_true", help="Rerun gamma scalp simulation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    if not FEATURE_CACHE.exists() and not args.refresh_features:
        print(f"{FEATURE_CACHE} missing; rebuilding from K3 v2 base", flush=True)
    panel = build_panel(refresh_features=args.refresh_features, refresh_panel=args.refresh_panel)
    trades = simulate_gamma(panel, refresh_trades=args.refresh_trades or args.refresh_panel)
    vol_summary = summarize_vol_gaps(panel)
    gamma_summary = summarize_gamma(trades)
    combined = pd.concat([vol_summary, gamma_summary], ignore_index=True, sort=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}", flush=True)
    write_note(panel, trades, vol_summary, gamma_summary)


if __name__ == "__main__":
    main()
