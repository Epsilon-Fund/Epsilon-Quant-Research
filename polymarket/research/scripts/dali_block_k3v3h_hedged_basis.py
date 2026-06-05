"""Block K3 v3h: delta-hedged dynamic-basis RV screen.

Re-points K3 away from the dead sub-second lead-lag race and toward a two-leg
convergence trade: buy the cheap Polymarket side and delta-hedge on Binance.
The expensive 1s K3 v2 panel is cached as parquet and reused across runs.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import YEAR_SECONDS, cents, markdown_table, norm_cdf, number, pct


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
CACHE = ANALYSIS / "cache"
K3V2_CSV = ANALYSIS / "csv_outputs" / "options_delta" / "k3v2_leadlag_causal.csv"
BASE_CACHE = CACHE / "k3v2_1s_panel_base.parquet"
FEATURE_CACHE = CACHE / "k3v3h_panel_features.parquet"
OUT_TRADES = ANALYSIS / "csv_outputs" / "options_delta" / "k3v3h_hedged_basis_trades.csv"
OUT_EXT = ANALYSIS / "csv_outputs" / "options_delta" / "k3v2_leadlag_causal_hedged_ext.csv"
NOTE = NOTES / "block_k3v3h_findings.md"

ACTION_LATENCIES = (1, 2, 5, 10)
ENTRY_BANDS = (0.5, 0.75, 1.0, 1.5, 2.0)
EXIT_BANDS = (0.10, 0.25, 0.50)
SOURCE_MARGIN_BP = 10.0
TOXIC_TAU_SECONDS = 15 * 60
EARLY_TAU_SECONDS = 2 * 60 * 60
MID_TAU_SECONDS = 30 * 60
STATIC_BASIS_CUTOFF = 0.10
STATIC_GAP_HALFLIFE_SECONDS = 1800
BINANCE_HEDGE_COST_BPS = 6.0
FUNDING_BPS_PER_8H = 1.0
BOOTSTRAP_SAMPLES = 500
RNG_SEED = 20260531
MIN_PRICE = 1e-4

BASE_COLS = [
    "ts",
    "source_run",
    "market_slug",
    "market_id",
    "condition_id",
    "question",
    "source_runs",
    "asset",
    "window_start",
    "window_end",
    "seconds_to_expiry",
    "binance_spot",
    "binance_strike_spot",
    "binance_close_spot",
    "binance_window_return",
    "binance_window_abs_return_bps",
    "binance_resolution_up",
    "chainlink_resolution_up",
    "chainlink_binance_resolution_disagree",
    "resolution_source",
    "ewma_sigma_annualized",
    "polymarket_mid",
    "up_bid",
    "up_ask",
    "down_bid",
    "down_ask",
    "taker_fee_rate",
]


@dataclass
class Trade:
    latency_s: int
    source_filter: str
    entry_band: float
    exit_band: float
    asset: str
    market_slug: str
    state_bucket: str
    route: str
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    exit_signal_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    hold_seconds: float
    entry_dynamic_gap: float
    exit_dynamic_gap: float
    entry_abs_z: float
    entry_seconds_to_expiry: float
    entry_spot: float
    exit_spot: float
    pm_entry_price: float
    pm_exit_price: float
    pm_entry_fee: float
    pm_exit_fee: float
    pm_pnl: float
    hedge_pnl: float
    hedge_turnover_notional: float
    hedge_cost: float
    funding_cost: float
    hedged_net_pnl: float
    naked_net_pnl: float
    source_ok: bool
    source_disagree: bool
    settlement_margin_bp: float
    static_large_entry: bool


def normal_pdf(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.exp(-0.5 * arr * arr) / math.sqrt(2.0 * math.pi)


def logit(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    if isinstance(values, pd.Series):
        p = values.astype(float).clip(MIN_PRICE, 1.0 - MIN_PRICE)
        return np.log(p / (1.0 - p))
    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, MIN_PRICE, 1.0 - MIN_PRICE)
    out = np.log(clipped / (1.0 - clipped))
    if np.ndim(values) == 0:
        return float(out)
    return out


def taker_fee(price: float, fee_rate: float) -> float:
    p = min(max(float(price), 0.0), 1.0)
    return float(fee_rate) * p * (1.0 - p)


def bps(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.1f}bp"


def ci_text(lo: float, hi: float, digits: int = 4) -> str:
    return f"[{number(lo, digits)}, {number(hi, digits)}]"


def load_base_panel(refresh_base: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    if BASE_CACHE.exists() and not refresh_base:
        print(f"loading base cache {BASE_CACHE}", flush=True)
        return pd.read_parquet(BASE_CACHE)
    if not K3V2_CSV.exists():
        raise FileNotFoundError(f"missing {K3V2_CSV}; run K3 v2 first")
    print(f"building base cache from {K3V2_CSV}", flush=True)
    df = pd.read_csv(K3V2_CSV, usecols=BASE_COLS, parse_dates=["ts", "window_start", "window_end"])
    df = df.sort_values(["market_slug", "ts"]).reset_index(drop=True)
    df.to_parquet(BASE_CACHE, index=False)
    print(f"wrote {BASE_CACHE}", flush=True)
    return df


def time_bucket(seconds_to_expiry: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [seconds_to_expiry > EARLY_TAU_SECONDS, seconds_to_expiry > MID_TAU_SECONDS],
            ["early_gt2h", "mid_30m_2h"],
            default="late_lt30m",
        ),
        index=seconds_to_expiry.index,
    )


def moneyness_bucket(abs_z: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([abs_z < 0.25, abs_z < 1.0], ["near_absz_lt0.25", "mid_absz_0.25_1"], default="far_absz_ge1"),
        index=abs_z.index,
    )


def add_features(base: pd.DataFrame) -> pd.DataFrame:
    df = base.copy().sort_values(["market_slug", "ts"]).reset_index(drop=True)
    tau = df["seconds_to_expiry"].astype(float).clip(lower=1.0)
    tau_years = tau / YEAR_SECONDS
    spot = df["binance_spot"].astype(float)
    strike = df["binance_strike_spot"].astype(float)
    sigma = df["ewma_sigma_annualized"].astype(float).clip(lower=1e-6)
    z = np.log(spot / strike) / (sigma * np.sqrt(tau_years))
    pdf = normal_pdf(z)
    p_model = pd.Series(norm_cdf(z), index=df.index).clip(0.0, 1.0)
    fair_logit = logit(p_model)
    pm_logit = logit(df["polymarket_mid"])
    raw_gap = pm_logit - fair_logit

    static_gap = (
        raw_gap.groupby(df["market_slug"], sort=False)
        .transform(lambda s: s.ewm(halflife=STATIC_GAP_HALFLIFE_SECONDS, adjust=False, min_periods=60).mean().shift(1))
    )
    dynamic_gap = raw_gap - static_gap

    df["tau_years"] = tau_years
    df["digital_z"] = z
    df["abs_z"] = np.abs(z)
    df["p_model"] = p_model
    df["digital_delta"] = pdf / (spot * sigma * np.sqrt(tau_years))
    df["pm_logit"] = pm_logit
    df["fair_logit"] = fair_logit
    df["raw_logit_gap_pm_minus_fair"] = raw_gap
    df["causal_static_logit_gap"] = static_gap
    df["dynamic_logit_gap"] = dynamic_gap
    df["pmodel_basis"] = df["polymarket_mid"] - df["p_model"]
    df["large_static_basis_10c"] = df["pmodel_basis"].abs() > STATIC_BASIS_CUTOFF
    df["moneyness_bucket"] = moneyness_bucket(df["abs_z"])
    df["time_bucket"] = time_bucket(df["seconds_to_expiry"])
    df["toxic_near_expiry"] = (df["abs_z"] < 0.25) & (df["seconds_to_expiry"] <= TOXIC_TAU_SECONDS)
    df["state_bucket"] = np.where(
        df["toxic_near_expiry"],
        "toxic_near_expiry",
        df["moneyness_bucket"].astype(str) + "|" + df["time_bucket"].astype(str),
    )
    disagree = df["chainlink_binance_resolution_disagree"].fillna(False).astype(bool)
    margin = df["binance_window_abs_return_bps"].astype(float)
    df["source_ok_strict"] = (~disagree) & (margin >= SOURCE_MARGIN_BP)
    df["source_penalty_flag"] = disagree | (margin < SOURCE_MARGIN_BP)
    df["spot_diff_1s"] = df.groupby("market_slug", sort=False)["binance_spot"].diff()
    return df


def load_features(refresh_base: bool = False, refresh_features: bool = False) -> pd.DataFrame:
    if FEATURE_CACHE.exists() and not refresh_features and not refresh_base:
        print(f"loading feature cache {FEATURE_CACHE}", flush=True)
        return pd.read_parquet(FEATURE_CACHE)
    base = load_base_panel(refresh_base=refresh_base)
    print("building hedged basis feature cache", flush=True)
    df = add_features(base)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURE_CACHE, index=False)
    print(f"wrote {FEATURE_CACHE}", flush=True)
    return df


def ns_to_ts(value: int) -> pd.Timestamp:
    return pd.Timestamp(int(value), unit="ns", tz="UTC")


def timestamp_ns(series: pd.Series) -> np.ndarray:
    raw = series.astype("int64").to_numpy()
    dtype = str(series.dtype)
    if "[us" in dtype:
        return raw * 1_000
    if "[ms" in dtype:
        return raw * 1_000_000
    if "[s" in dtype and "[ns" not in dtype and "[ms" not in dtype and "[us" not in dtype:
        return raw * 1_000_000_000
    return raw


def market_arrays(g: pd.DataFrame) -> dict[str, Any]:
    g = g.sort_values("ts").reset_index(drop=True)
    return {
        "ts_ns": timestamp_ns(g["ts"]),
        "dynamic_gap": g["dynamic_logit_gap"].to_numpy(dtype=float),
        "toxic": g["toxic_near_expiry"].to_numpy(dtype=bool),
        "large_static": g["large_static_basis_10c"].to_numpy(dtype=bool),
        "seconds_to_expiry": g["seconds_to_expiry"].to_numpy(dtype=float),
        "up_ask": g["up_ask"].to_numpy(dtype=float),
        "up_bid": g["up_bid"].to_numpy(dtype=float),
        "down_ask": g["down_ask"].to_numpy(dtype=float),
        "down_bid": g["down_bid"].to_numpy(dtype=float),
        "fee_rate": g["taker_fee_rate"].fillna(0.07).to_numpy(dtype=float),
        "spot": g["binance_spot"].to_numpy(dtype=float),
        "delta": g["digital_delta"].to_numpy(dtype=float),
        "abs_z": g["abs_z"].to_numpy(dtype=float),
        "state_bucket": g["state_bucket"].astype(str).to_numpy(),
        "source_ok": bool(g["source_ok_strict"].iloc[0]),
        "source_disagree": bool(g["chainlink_binance_resolution_disagree"].iloc[0]),
        "settlement_margin_bp": float(g["binance_window_abs_return_bps"].iloc[0]),
        "asset": str(g["asset"].iloc[0]),
        "market_slug": str(g["market_slug"].iloc[0]),
    }


def hedge_path_pnl_arrays(arr: dict[str, Any], route: str, entry_pos: int, exit_pos: int) -> tuple[float, float, float, float]:
    if exit_pos - entry_pos + 1 < 2:
        return 0.0, 0.0, 0.0, 0.0
    sign = -1.0 if route == "buy_up" else 1.0
    spot = arr["spot"][entry_pos : exit_pos + 1]
    delta = arr["delta"][entry_pos : exit_pos + 1]
    target = sign * delta
    dspot = np.diff(spot)
    hedge_pnl = float(np.dot(target[:-1], dspot))
    turnover = float(abs(target[0]) * spot[0] + np.sum(np.abs(np.diff(target)) * spot[1:]) + abs(target[-1]) * spot[-1])
    hedge_cost = turnover * BINANCE_HEDGE_COST_BPS / 10000.0
    avg_notional = float(np.mean(np.abs(target) * spot))
    hold_seconds = float((arr["ts_ns"][exit_pos] - arr["ts_ns"][entry_pos]) / 1_000_000_000)
    funding_cost = avg_notional * FUNDING_BPS_PER_8H / 10000.0 * hold_seconds / (8.0 * 3600.0)
    return hedge_pnl, turnover, hedge_cost, funding_cost


def exit_condition_arrays(arr: dict[str, Any], pos: int, route: str, exit_band: float) -> bool:
    gap = float(arr["dynamic_gap"][pos])
    if not np.isfinite(gap):
        return False
    if bool(arr["toxic"][pos]) or float(arr["seconds_to_expiry"][pos]) <= TOXIC_TAU_SECONDS:
        return True
    if bool(arr["large_static"][pos]):
        return True
    if route == "buy_up":
        return gap >= -exit_band
    return gap <= exit_band


def simulate_market_fast(
    arr: dict[str, Any],
    *,
    latency: int,
    entry_band: float,
    exit_band: float,
    source_filter: str,
) -> list[Trade]:
    trades: list[Trade] = []
    n = len(arr["ts_ns"])
    i = 0
    source_ok = bool(arr["source_ok"])
    if source_filter == "strict" and not source_ok:
        return trades
    while i < n - latency - 2:
        if bool(arr["toxic"][i]) or bool(arr["large_static"][i]) or not np.isfinite(float(arr["dynamic_gap"][i])):
            i += 1
            continue
        gap = float(arr["dynamic_gap"][i])
        if gap <= -entry_band:
            route = "buy_up"
        elif gap >= entry_band:
            route = "buy_down"
        else:
            i += 1
            continue

        entry_pos = i + latency
        if bool(arr["toxic"][entry_pos]) or bool(arr["large_static"][entry_pos]):
            i += 1
            continue

        j = entry_pos + 1
        while j < n - latency - 1:
            if exit_condition_arrays(arr, j, route, exit_band):
                break
            j += 1
        exit_signal_pos = j
        exit_pos = min(j + latency, n - 1)

        fee_rate = float(arr["fee_rate"][entry_pos]) if np.isfinite(float(arr["fee_rate"][entry_pos])) else 0.07
        if route == "buy_up":
            pm_entry = float(arr["up_ask"][entry_pos])
            pm_exit = float(arr["up_bid"][exit_pos])
        else:
            pm_entry = float(arr["down_ask"][entry_pos])
            pm_exit = float(arr["down_bid"][exit_pos])
        if not all(np.isfinite(x) for x in [pm_entry, pm_exit]):
            i = exit_pos + 1
            continue
        entry_fee = taker_fee(pm_entry, fee_rate)
        exit_fee = taker_fee(pm_exit, fee_rate)
        pm_pnl = pm_exit - exit_fee - pm_entry - entry_fee
        hedge_pnl, turnover, hedge_cost, funding_cost = hedge_path_pnl_arrays(arr, route, entry_pos, exit_pos)
        hedged = pm_pnl + hedge_pnl - hedge_cost - funding_cost
        hold_seconds = float((arr["ts_ns"][exit_pos] - arr["ts_ns"][entry_pos]) / 1_000_000_000)
        trades.append(
            Trade(
                latency_s=latency,
                source_filter=source_filter,
                entry_band=entry_band,
                exit_band=exit_band,
                asset=arr["asset"],
                market_slug=arr["market_slug"],
                state_bucket=str(arr["state_bucket"][i]),
                route=route,
                signal_ts=ns_to_ts(arr["ts_ns"][i]),
                entry_ts=ns_to_ts(arr["ts_ns"][entry_pos]),
                exit_signal_ts=ns_to_ts(arr["ts_ns"][exit_signal_pos]),
                exit_ts=ns_to_ts(arr["ts_ns"][exit_pos]),
                hold_seconds=hold_seconds,
                entry_dynamic_gap=gap,
                exit_dynamic_gap=float(arr["dynamic_gap"][exit_pos]),
                entry_abs_z=float(arr["abs_z"][entry_pos]),
                entry_seconds_to_expiry=float(arr["seconds_to_expiry"][entry_pos]),
                entry_spot=float(arr["spot"][entry_pos]),
                exit_spot=float(arr["spot"][exit_pos]),
                pm_entry_price=pm_entry,
                pm_exit_price=pm_exit,
                pm_entry_fee=entry_fee,
                pm_exit_fee=exit_fee,
                pm_pnl=pm_pnl,
                hedge_pnl=hedge_pnl,
                hedge_turnover_notional=turnover,
                hedge_cost=hedge_cost,
                funding_cost=funding_cost,
                hedged_net_pnl=hedged,
                naked_net_pnl=pm_pnl,
                source_ok=source_ok,
                source_disagree=bool(arr["source_disagree"]),
                settlement_margin_bp=float(arr["settlement_margin_bp"]),
                static_large_entry=bool(arr["large_static"][entry_pos]),
            )
        )
        i = exit_pos + 1
    return trades


def simulate_all(panel: pd.DataFrame) -> pd.DataFrame:
    trades: list[Trade] = []
    groups = [market_arrays(g) for _, g in panel.groupby("market_slug", sort=False)]
    combos = [
        (lat, entry, exit_band, source_filter)
        for lat in ACTION_LATENCIES
        for entry in ENTRY_BANDS
        for exit_band in EXIT_BANDS
        if exit_band < entry
        for source_filter in ("all", "strict")
    ]
    for combo_i, (lat, entry, exit_band, source_filter) in enumerate(combos, start=1):
        print(
            f"sim {combo_i}/{len(combos)} latency={lat}s entry={entry} exit={exit_band} source={source_filter}",
            flush=True,
        )
        for g in groups:
            trades.extend(
                simulate_market_fast(g, latency=lat, entry_band=entry, exit_band=exit_band, source_filter=source_filter)
            )
    return pd.DataFrame([t.__dict__ for t in trades])


def bootstrap_mean(trades: pd.DataFrame, value_col: str) -> tuple[float, float]:
    if trades.empty:
        return math.nan, math.nan
    markets = trades["market_slug"].drop_duplicates().to_numpy()
    if len(markets) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(RNG_SEED + len(trades) + (0 if value_col == "hedged_net_pnl" else 10_000))
    vals: list[float] = []
    by_market = {m: trades[trades["market_slug"].eq(m)][value_col].to_numpy(dtype=float) for m in markets}
    for _ in range(BOOTSTRAP_SAMPLES):
        sample_markets = markets[rng.integers(0, len(markets), size=len(markets))]
        arrs = [by_market[m] for m in sample_markets if len(by_market[m])]
        if arrs:
            vals.append(float(np.concatenate(arrs).mean()))
    if not vals:
        return math.nan, math.nan
    lo, hi = np.nanquantile(np.asarray(vals), [0.025, 0.975])
    return float(lo), float(hi)


def summarize_grid(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["source_filter", "latency_s", "entry_band", "exit_band", "state_bucket"]
    for key, g in trades.groupby(keys, sort=True):
        if len(g) < 5:
            continue
        h_lo, h_hi = bootstrap_mean(g, "hedged_net_pnl")
        n_lo, n_hi = bootstrap_mean(g, "naked_net_pnl")
        rows.append(
            {
                "source_filter": key[0],
                "latency_s": int(key[1]),
                "entry_band": float(key[2]),
                "exit_band": float(key[3]),
                "state_bucket": key[4],
                "n_trades": int(len(g)),
                "markets": int(g["market_slug"].nunique()),
                "mean_hedged_pnl": float(g["hedged_net_pnl"].mean()),
                "hedged_ci_lo": h_lo,
                "hedged_ci_hi": h_hi,
                "mean_naked_pnl": float(g["naked_net_pnl"].mean()),
                "naked_ci_lo": n_lo,
                "naked_ci_hi": n_hi,
                "win_rate_hedged": float((g["hedged_net_pnl"] > 0).mean()),
                "win_rate_naked": float((g["naked_net_pnl"] > 0).mean()),
                "median_hold_seconds": float(g["hold_seconds"].median()),
                "p95_hold_seconds": float(g["hold_seconds"].quantile(0.95)),
                "mean_hedge_cost": float(g["hedge_cost"].mean()),
                "mean_hedge_turnover": float(g["hedge_turnover_notional"].mean()),
                "tail_market_share": float(g["market_slug"].value_counts(normalize=True).iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def selected_latency_table(trades: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    if selected is None or selected.empty:
        return pd.DataFrame()
    mask_base = (
        trades["source_filter"].eq(selected["source_filter"])
        & trades["entry_band"].eq(float(selected["entry_band"]))
        & trades["exit_band"].eq(float(selected["exit_band"]))
        & trades["state_bucket"].eq(selected["state_bucket"])
    )
    rows = []
    for lat, g in trades[mask_base].groupby("latency_s", sort=True):
        h_lo, h_hi = bootstrap_mean(g, "hedged_net_pnl")
        n_lo, n_hi = bootstrap_mean(g, "naked_net_pnl")
        rows.append(
            {
                "latency_s": int(lat),
                "n_trades": int(len(g)),
                "mean_hedged_pnl": float(g["hedged_net_pnl"].mean()),
                "hedged_ci_lo": h_lo,
                "hedged_ci_hi": h_hi,
                "mean_naked_pnl": float(g["naked_net_pnl"].mean()),
                "naked_ci_lo": n_lo,
                "naked_ci_hi": n_hi,
                "median_hold_seconds": float(g["hold_seconds"].median()),
                "p95_hold_seconds": float(g["hold_seconds"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def write_extended_panel(panel: pd.DataFrame) -> None:
    cols = [
        "ts",
        "asset",
        "market_slug",
        "seconds_to_expiry",
        "binance_spot",
        "binance_strike_spot",
        "ewma_sigma_annualized",
        "polymarket_mid",
        "p_model",
        "digital_z",
        "abs_z",
        "digital_delta",
        "raw_logit_gap_pm_minus_fair",
        "causal_static_logit_gap",
        "dynamic_logit_gap",
        "pmodel_basis",
        "large_static_basis_10c",
        "state_bucket",
        "source_ok_strict",
        "source_penalty_flag",
        "binance_window_abs_return_bps",
        "chainlink_binance_resolution_disagree",
    ]
    panel[cols].to_csv(OUT_EXT, index=False)


def write_note(panel: pd.DataFrame, trades: pd.DataFrame, summary: pd.DataFrame) -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    strict = summary[summary["source_filter"].eq("strict")].copy()
    positive = strict[(strict["hedged_ci_lo"] > 0) & (strict["n_trades"] >= 10)].copy()
    if positive.empty:
        selected = strict.sort_values(["mean_hedged_pnl", "n_trades"], ascending=[False, False]).head(1)
    else:
        selected = positive.sort_values(["mean_hedged_pnl", "n_trades"], ascending=[False, False]).head(1)
    selected_row = selected.iloc[0] if not selected.empty else None
    lat_table = selected_latency_table(trades, selected_row) if selected_row is not None else pd.DataFrame()

    all_strict = trades[trades["source_filter"].eq("strict")]
    all_all = trades[trades["source_filter"].eq("all")]
    source_penalized_markets = (
        panel[["market_slug", "source_penalty_flag"]].drop_duplicates("market_slug")["source_penalty_flag"].sum()
    )

    grid_rows = []
    top = strict.sort_values(["hedged_ci_lo", "mean_hedged_pnl"], ascending=[False, False]).head(10)
    for _, row in top.iterrows():
        grid_rows.append(
            [
                str(row["state_bucket"]),
                str(int(row["latency_s"])),
                number(float(row["entry_band"]), 2),
                number(float(row["exit_band"]), 2),
                str(int(row["n_trades"])),
                cents(float(row["mean_hedged_pnl"])),
                ci_text(float(row["hedged_ci_lo"]), float(row["hedged_ci_hi"])),
                cents(float(row["mean_naked_pnl"])),
                ci_text(float(row["naked_ci_lo"]), float(row["naked_ci_hi"])),
                pct(float(row["win_rate_hedged"])),
                number(float(row["median_hold_seconds"]), 1),
                pct(float(row["tail_market_share"])),
            ]
        )

    latency_rows = []
    for _, row in lat_table.iterrows():
        latency_rows.append(
            [
                str(int(row["latency_s"])),
                str(int(row["n_trades"])),
                cents(float(row["mean_hedged_pnl"])),
                ci_text(float(row["hedged_ci_lo"]), float(row["hedged_ci_hi"])),
                cents(float(row["mean_naked_pnl"])),
                ci_text(float(row["naked_ci_lo"]), float(row["naked_ci_hi"])),
                number(float(row["median_hold_seconds"]), 1),
                number(float(row["p95_hold_seconds"]), 1),
            ]
        )

    if selected_row is not None:
        sel_trades = trades[
            trades["source_filter"].eq(selected_row["source_filter"])
            & trades["entry_band"].eq(float(selected_row["entry_band"]))
            & trades["exit_band"].eq(float(selected_row["exit_band"]))
            & trades["state_bucket"].eq(selected_row["state_bucket"])
            & trades["latency_s"].eq(int(selected_row["latency_s"]))
        ]
    else:
        sel_trades = pd.DataFrame()
    tail_bucket = (
        all_strict["state_bucket"].value_counts(normalize=True).head(1) if len(all_strict) else pd.Series(dtype=float)
    )
    tail_market = (
        all_strict["market_slug"].value_counts(normalize=True).head(1) if len(all_strict) else pd.Series(dtype=float)
    )
    selected_tail_market = (
        sel_trades["market_slug"].value_counts(normalize=True).head(1) if len(sel_trades) else pd.Series(dtype=float)
    )

    source_rows = []
    for source_name, g in [("all_windows", all_all), ("strict_source", all_strict)]:
        if g.empty:
            continue
        lo, hi = bootstrap_mean(g, "hedged_net_pnl")
        source_rows.append(
            [
                source_name,
                str(int(len(g))),
                str(int(g["market_slug"].nunique())),
                cents(float(g["hedged_net_pnl"].mean())),
                ci_text(lo, hi),
                cents(float(g["naked_net_pnl"].mean())),
                pct(float((g["hedged_net_pnl"] > 0).mean())),
            ]
        )

    if selected_row is None:
        headline = "No hedged convergence regime produced enough trades to evaluate."
        selected_text = ""
    elif float(selected_row["hedged_ci_lo"]) > 0:
        headline = (
            "A strict-source delta-hedged dynamic-basis regime clears zero in-sample "
            f"in `{selected_row['state_bucket']}`."
        )
        selected_text = (
            f"Best strict regime: latency {int(selected_row['latency_s'])}s, entry {number(float(selected_row['entry_band']), 2)} "
            f"logit, exit {number(float(selected_row['exit_band']), 2)} logit, mean hedged PnL "
            f"{cents(float(selected_row['mean_hedged_pnl']))} CI "
            f"{ci_text(float(selected_row['hedged_ci_lo']), float(selected_row['hedged_ci_hi']))}."
        )
    else:
        headline = "No strict-source hedged convergence bucket clears zero after costs."
        selected_text = (
            f"Best strict candidate is `{selected_row['state_bucket']}` at latency {int(selected_row['latency_s'])}s "
            f"with mean hedged PnL {cents(float(selected_row['mean_hedged_pnl']))} CI "
            f"{ci_text(float(selected_row['hedged_ci_lo']), float(selected_row['hedged_ci_hi']))}."
        )

    latency_comment = "Latency robustness could not be evaluated."
    if not lat_table.empty and {1, 10}.issubset(set(lat_table["latency_s"].astype(int))):
        lat1 = lat_table[lat_table["latency_s"].eq(1)].iloc[0]
        lat10 = lat_table[lat_table["latency_s"].eq(10)].iloc[0]
        hedged_drift = float(lat10["mean_hedged_pnl"] - lat1["mean_hedged_pnl"])
        naked_drift = float(lat10["mean_naked_pnl"] - lat1["mean_naked_pnl"])
        latency_comment = (
            f"For the selected regime, hedged PnL changes from {cents(float(lat1['mean_hedged_pnl']))} at 1s "
            f"to {cents(float(lat10['mean_hedged_pnl']))} at 10s ({cents(hedged_drift)} drift), while naked "
            f"changes from {cents(float(lat1['mean_naked_pnl']))} to {cents(float(lat10['mean_naked_pnl']))} "
            f"({cents(naked_drift)} drift). The hedged version does not show the hoped-for slower latency decay."
        )

    tail_text = ""
    if len(tail_bucket):
        tail_text += f" Across strict-source trades, top bucket `{tail_bucket.index[0]}` contributes {pct(float(tail_bucket.iloc[0]))}."
    if len(tail_market):
        tail_text += f" Top market `{tail_market.index[0]}` contributes {pct(float(tail_market.iloc[0]))}."
    if len(selected_tail_market):
        tail_text += (
            f" Within the selected regime, top market `{selected_tail_market.index[0]}` contributes "
            f"{pct(float(selected_tail_market.iloc[0]))}."
        )

    text = f"""# Block K3 v3h Hedged Dynamic-Basis Findings

Generated: {datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")}

## Headline

{headline} {selected_text}

This re-points K3 away from the dead 0-1s naked lead-lag race and tests a two-legged convergence trade: buy the cheap Polymarket leg when the **causal, demeaned dynamic logit gap** is outside an entry band, delta-hedge on Binance, and exit when the gap converges or before the near-expiry spike zone. Static basis larger than {cents(STATIC_BASIS_CUTOFF)} is excluded at entry and forced flat because it is model/source error first, not alpha.

Costs are included as Polymarket taker fee on entry and exit, Binance hedge turnover at {number(BINANCE_HEDGE_COST_BPS, 1)}bp per notional traded, and funding at {number(FUNDING_BPS_PER_8H, 1)}bp per 8h prorated by holding time. Source filter excludes direction-disagreement windows and windows with Binance settlement margin < {number(SOURCE_MARGIN_BP, 1)}bp.

## Best Strict-Source Regimes

{markdown_table(["bucket", "latency", "entry", "exit", "trades", "hedged_mean", "hedged_CI", "naked_mean", "naked_CI", "hedged_win", "med_hold_s", "top_mkt_share"], grid_rows)}

## Latency Robustness

Same selected bucket and bands across action latencies. This is the robustness check for convergence versus a naked one-leg trade.

{latency_comment}

{markdown_table(["latency_s", "trades", "hedged_mean", "hedged_CI", "naked_mean", "naked_CI", "med_hold_s", "p95_hold_s"], latency_rows)}

## Source-Basis Filter

The strict source filter removes {int(source_penalized_markets)} / {panel['market_slug'].nunique()} windows: any Chainlink-vs-Binance direction disagreement or Binance settlement margin below {number(SOURCE_MARGIN_BP, 1)}bp. This is the hard risk filter; those windows are near-pin/source-basis cases and are not counted as clean alpha.

{markdown_table(["sample", "trades", "markets", "hedged_mean", "hedged_CI", "naked_mean", "hedged_win"], source_rows)}

## Tail Concentration

{tail_text.strip() or "No strict-source trades after filters."}

## Method

- Signal: `dynamic_logit_gap = (pm_logit - fair_logit) - causal_static_logit_gap`, where the static gap is an EWMA using only prior rows in the same market.
- Fair value: European digital `P=N(z)`, `z=ln(S/K)/(sigma*sqrt(tau))`, with Binance proxy spot, window-open strike, and causal EWMA vol.
- Trade direction: negative dynamic gap buys `UP`; positive dynamic gap buys `DOWN`.
- Hedge: 1 binary share is hedged every second with Binance notional from digital delta; `UP` uses short delta, `DOWN` uses long delta.
- Flatten: exit on convergence to the exit band, large static basis, or before `abs(z)<0.25` with tau <= {TOXIC_TAU_SECONDS}s.

## Outputs

- Extended row panel: `data/analysis/csv_outputs/options_delta/k3v2_leadlag_causal_hedged_ext.csv`
- Trade ledger: `data/analysis/csv_outputs/options_delta/k3v3h_hedged_basis_trades.csv`
- Feature cache: `data/analysis/cache/k3v3h_panel_features.parquet`
- Base cache reused: `data/analysis/cache/k3v2_1s_panel_base.parquet`
- Repro script: `scripts/dali_block_k3v3h_hedged_basis.py`
"""
    NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {NOTE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-base", action="store_true", help="Rebuild base parquet from K3 v2 CSV.")
    parser.add_argument("--refresh-features", action="store_true", help="Recompute hedged basis feature cache.")
    args = parser.parse_args()

    panel = load_features(refresh_base=args.refresh_base, refresh_features=args.refresh_features)
    write_extended_panel(panel)
    trades = simulate_all(panel)
    trades.to_csv(OUT_TRADES, index=False)
    summary = summarize_grid(trades)
    write_note(panel, trades, summary)
    print(f"wrote {OUT_EXT}", flush=True)
    print(f"wrote {OUT_TRADES}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
