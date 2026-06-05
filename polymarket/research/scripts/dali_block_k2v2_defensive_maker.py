"""Block K2 v2: Binance-anchored defensive maker box-check.

Research-only final maker falsifier. Quotes are anchored to Binance-implied
European digital fair value, then a leading-venue defense pulls or widens a
resting side when Binance fair moves away before the Polymarket taker print.
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from dali_block_k2v3_digital_maker import (
    ANALYSIS,
    BOOTSTRAP_SAMPLES,
    DEFAULT_SIGMA,
    FILL_WINDOW_SEC,
    GAMMA_CACHE,
    HEDGE_FEE_RATE,
    NOTES,
    ROOT,
    TICK,
    bootstrap_mean_ci,
    bps,
    gamma_settlement_maps,
    load_feature_tables,
    load_gamma_cache,
    maker_rebate_bps,
    markdown_table,
    norm_cdf,
    norm_pdf,
    pct,
    state_at_or_before,
    taker_fee_bps,
)
from dali_block_k3v2_leadlag_causal import add_causal_vol, fetch_klines
from dali_block_k3v3h_hedged_basis import FEATURE_CACHE, load_features, timestamp_ns


OUT_CSV = ANALYSIS / "csv_outputs" / "market_making" / "k2v2_defensive_maker.csv"
OUT_FILLS = ANALYSIS / "k2v2_defensive_maker_fills.parquet"
NOTE = NOTES / "block_k2v2_findings.md"
DAILY_BINANCE_CACHE = ANALYSIS / "cache" / "k2v2_daily_binance_1s.parquet"
DAILY_MODEL_CACHE = ANALYSIS / "cache" / "k2v2_daily_model_surface.parquet"

YEAR_SECONDS = 365.0 * 24.0 * 3600.0
SPOT_BASE = "https://api.binance.com"
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
DEFAULT_DAILY_SIGMA = {"BTC": 0.60, "ETH": 0.80, "SOL": 1.20}
PREWARM_HOURS = 6
TAKER_HOLD_SEC = 60
RESOLUTION_BUFFER_SEC = 60
ROBUST_MIN_FILLS = 30
RNG_SEED = 20260531
SPIKE_ABS_Z = 0.25
SPIKE_TAU_MIN = 30.0

BASE_SPREAD_BPS = (100.0, 250.0, 500.0, 750.0)
REACTION_LATENCIES = (1.0, 2.0, 5.0, 10.0)
TOX_BANDS = (0.00005, 0.00010, 0.00025, 0.00050, 0.00100)
DEFENSE_MODES = ("pull", "widen")
DELTA_MULTS = (1.0, 2.0)


@dataclass(frozen=True)
class DefensiveConfig:
    name: str
    base_spread_bps: float
    reaction_latency_sec: float
    tox_band_prob: float
    defense_mode: str
    delta_mult: float
    bucket_widen: float = 0.5
    inventory_cap: int = 1


def cents(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}c"


def asset_from_daily_meta(meta: dict[str, Any]) -> str:
    text = f"{meta.get('resolutionSource') or ''} {meta.get('question') or ''} {meta.get('description') or ''}".lower()
    if "btc" in text or "bitcoin" in text:
        return "BTC"
    if "eth" in text or "ethereum" in text:
        return "ETH"
    if "sol" in text or "solana" in text:
        return "SOL"
    return ""


def parse_time(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def ns_to_ts(value: int) -> pd.Timestamp:
    return pd.Timestamp(int(value), unit="ns", tz="UTC")


def ts_ns(series: pd.Series) -> np.ndarray:
    return timestamp_ns(pd.to_datetime(series, utc=True))


def z_bucket(abs_z: float) -> str:
    if not np.isfinite(abs_z):
        return "nan"
    if abs_z < 0.25:
        return "near_absz_lt0.25"
    if abs_z < 1.0:
        return "mid_absz_0.25_1"
    return "far_absz_ge1"


def tau_bucket(tau_sec: float) -> str:
    if not np.isfinite(tau_sec):
        return "nan"
    if tau_sec <= 30 * 60:
        return "late_lt30m"
    if tau_sec <= 2 * 60 * 60:
        return "mid_30m_2h"
    return "early_gt2h"


def family_slug_maps(markets: pd.DataFrame, gamma_cache: dict[str, Any]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    family_by_slug = dict(zip(markets["slug"].astype(str), markets["family"].astype(str), strict=False))
    daily: dict[str, dict[str, Any]] = {}
    for row in markets[markets["family"].eq("daily_crypto_up_down")].itertuples(index=False):
        meta = gamma_cache.get(str(row.market_id), {})
        start = parse_time(meta.get("eventStartTime"))
        end = parse_time(meta.get("endDate"))
        asset = asset_from_daily_meta(meta)
        if start is None or end is None or not asset:
            continue
        daily[str(row.slug)] = {
            "market_id": str(row.market_id),
            "slug": str(row.slug),
            "asset": asset,
            "window_start": start,
            "window_end": end,
            "question": str(row.question),
            "resolution_source": str(meta.get("resolutionSource") or ""),
        }
    return family_by_slug, daily


def build_4h_model_surface(family_by_slug: dict[str, str]) -> pd.DataFrame:
    panel = load_features()
    rows = []
    for slug, g in panel.groupby("market_slug", sort=False):
        if family_by_slug.get(str(slug)) != "crypto_4h_up_down":
            continue
        g = g.sort_values("ts").copy()
        tau = g["seconds_to_expiry"].astype(float).clip(lower=1.0)
        tau_years = tau / YEAR_SECONDS
        spot = g["binance_spot"].astype(float)
        strike = g["binance_strike_spot"].astype(float)
        sigma = g["ewma_sigma_annualized"].astype(float).clip(lower=1e-6)
        log_m = np.log(spot / strike)
        z = g["digital_z"].astype(float)
        theta = norm_pdf(z.to_numpy(dtype=float)) * log_m.to_numpy(dtype=float) / np.maximum(
            2.0 * sigma.to_numpy(dtype=float) * np.power(tau_years.to_numpy(dtype=float), 1.5) * YEAR_SECONDS,
            1e-12,
        )
        out = pd.DataFrame(
            {
                "ts": pd.to_datetime(g["ts"], utc=True),
                "ts_ns": timestamp_ns(pd.to_datetime(g["ts"], utc=True)),
                "slug": str(slug),
                "family": "crypto_4h_up_down",
                "asset": g["asset"].astype(str).to_numpy(),
                "window_start": pd.to_datetime(g["window_start"], utc=True),
                "window_end": pd.to_datetime(g["window_end"], utc=True),
                "window_start_ns": timestamp_ns(pd.to_datetime(g["window_start"], utc=True)),
                "window_end_ns": timestamp_ns(pd.to_datetime(g["window_end"], utc=True)),
                "window_strike_spot": g["binance_strike_spot"].astype(float).to_numpy(),
                "window_close_spot": g["binance_close_spot"].astype(float).to_numpy(),
                "seconds_to_expiry": tau.to_numpy(dtype=float),
                "binance_spot": spot.to_numpy(dtype=float),
                "sigma_causal": sigma.to_numpy(dtype=float),
                "z": z.to_numpy(dtype=float),
                "abs_z": g["abs_z"].astype(float).to_numpy(),
                "fair_up_causal": g["p_model"].astype(float).to_numpy(),
                "delta_up": g["digital_delta"].astype(float).to_numpy(),
                "theta_up_per_sec": theta,
                "source_ok_strict": g["source_ok_strict"].fillna(False).astype(bool).to_numpy(),
                "binance_window_abs_return_bps": g["binance_window_abs_return_bps"].astype(float).to_numpy(),
                "source_direction_mismatch": g["chainlink_binance_resolution_disagree"].fillna(False).astype(bool).to_numpy(),
            }
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fetch_daily_binance(daily_meta: dict[str, dict[str, Any]], refresh: bool = False) -> pd.DataFrame:
    DAILY_BINANCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if DAILY_BINANCE_CACHE.exists() and not refresh:
        return pd.read_parquet(DAILY_BINANCE_CACHE)
    if not daily_meta:
        return pd.DataFrame()
    rows = []
    by_asset: dict[str, list[dict[str, Any]]] = {}
    for meta in daily_meta.values():
        by_asset.setdefault(meta["asset"], []).append(meta)
    with httpx.Client(headers={"User-Agent": "epsilon-quant-research-k2v2/1.0"}) as client:
        for asset, metas in sorted(by_asset.items()):
            start = min(m["window_start"] for m in metas) - pd.Timedelta(hours=PREWARM_HOURS)
            end = max(m["window_end"] for m in metas) + pd.Timedelta(minutes=5)
            print(f"fetching daily Binance {asset} 1s {start} to {end}", flush=True)
            spot = add_causal_vol(fetch_klines(client, symbol=SYMBOLS[asset], start=start, end=end))
            if spot.empty:
                continue
            spot["asset"] = asset
            rows.append(spot)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out.to_parquet(DAILY_BINANCE_CACHE, index=False)
    return out


def spot_asof(spot: pd.DataFrame, ts: pd.Timestamp) -> float:
    s = spot.set_index("ts").sort_index()["close"].loc[:ts]
    return float(s.iloc[-1]) if len(s) else math.nan


def build_daily_model_surface(daily_meta: dict[str, dict[str, Any]], refresh_binance: bool = False, refresh_model: bool = False) -> pd.DataFrame:
    DAILY_MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if DAILY_MODEL_CACHE.exists() and not refresh_model and not refresh_binance:
        return pd.read_parquet(DAILY_MODEL_CACHE)
    if not daily_meta:
        return pd.DataFrame()
    spot_all = fetch_daily_binance(daily_meta, refresh=refresh_binance)
    rows = []
    for slug, meta in daily_meta.items():
        spot = spot_all[spot_all["asset"].eq(meta["asset"])].copy().sort_values("ts")
        if spot.empty:
            continue
        start = meta["window_start"]
        end = meta["window_end"]
        strike = spot_asof(spot, start)
        close = spot_asof(spot, end)
        g = spot[(spot["ts"] >= start) & (spot["ts"] <= end)].copy()
        if g.empty or not np.isfinite(strike) or strike <= 0:
            continue
        sigma = g["ewma_sigma_annualized"].astype(float).ffill().fillna(DEFAULT_DAILY_SIGMA.get(meta["asset"], 0.80)).clip(0.05, 3.0)
        tau = (end - g["ts"]).dt.total_seconds().clip(lower=1.0)
        tau_years = tau / YEAR_SECONDS
        log_m = np.log(g["close"].astype(float) / strike)
        z = log_m / np.maximum(sigma * np.sqrt(tau_years), 1e-12)
        fair = norm_cdf(z.to_numpy(dtype=float))
        delta = norm_pdf(z.to_numpy(dtype=float)) / np.maximum(
            g["close"].astype(float).to_numpy() * sigma.to_numpy(dtype=float) * np.sqrt(tau_years.to_numpy(dtype=float)),
            1e-12,
        )
        theta = norm_pdf(z.to_numpy(dtype=float)) * log_m.to_numpy(dtype=float) / np.maximum(
            2.0 * sigma.to_numpy(dtype=float) * np.power(tau_years.to_numpy(dtype=float), 1.5) * YEAR_SECONDS,
            1e-12,
        )
        ret = math.log(close / strike) if np.isfinite(close) and close > 0 else math.nan
        out = pd.DataFrame(
            {
                "ts": g["ts"],
                "ts_ns": timestamp_ns(g["ts"]),
                "slug": slug,
                "family": "daily_crypto_up_down",
                "asset": meta["asset"],
                "window_start": start,
                "window_end": end,
                "window_start_ns": int(start.value),
                "window_end_ns": int(end.value),
                "window_strike_spot": strike,
                "window_close_spot": close,
                "seconds_to_expiry": tau.to_numpy(dtype=float),
                "binance_spot": g["close"].astype(float).to_numpy(),
                "sigma_causal": sigma.to_numpy(dtype=float),
                "z": z.to_numpy(dtype=float),
                "abs_z": np.abs(z.to_numpy(dtype=float)),
                "fair_up_causal": np.clip(fair, 0.0, 1.0),
                "delta_up": delta,
                "theta_up_per_sec": theta,
                "source_ok_strict": True,
                "binance_window_abs_return_bps": abs(ret) * 10000.0 if np.isfinite(ret) else math.nan,
                "source_direction_mismatch": False,
            }
        )
        rows.append(out)
    model = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not model.empty:
        model.to_parquet(DAILY_MODEL_CACHE, index=False)
    return model


def attach_model(states: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    parts = []
    model_cols = [
        "ts_ns",
        "asset",
        "window_start_ns",
        "window_end_ns",
        "window_strike_spot",
        "window_close_spot",
        "seconds_to_expiry",
        "binance_spot",
        "sigma_causal",
        "z",
        "abs_z",
        "fair_up_causal",
        "delta_up",
        "theta_up_per_sec",
        "source_ok_strict",
        "binance_window_abs_return_bps",
        "source_direction_mismatch",
    ]
    for slug, g in states.groupby("slug", sort=False):
        m = model[model["slug"].eq(str(slug))][model_cols].sort_values("ts_ns")
        if m.empty:
            continue
        piece = pd.merge_asof(g.sort_values("t_ns"), m, left_on="t_ns", right_on="ts_ns", direction="backward")
        piece = piece[piece["fair_up_causal"].notna()].copy()
        if piece.empty:
            continue
        is_up = piece["outcome_index"].fillna(0).astype(int).eq(0)
        piece["token_fair"] = np.where(is_up, piece["fair_up_causal"], 1.0 - piece["fair_up_causal"])
        piece["token_delta"] = np.where(is_up, piece["delta_up"], -piece["delta_up"])
        piece["token_theta_per_sec"] = np.where(is_up, piece["theta_up_per_sec"], -piece["theta_up_per_sec"])
        piece["tau_sec"] = piece["seconds_to_expiry"].astype(float).clip(lower=0.0)
        piece["tau_min"] = piece["tau_sec"] / 60.0
        piece["local_spread_bps"] = (piece["best_ask"] - piece["best_bid"]) / piece["mid"].clip(0.01, 0.99) * 10_000.0
        piece["z_bucket"] = piece["abs_z"].map(z_bucket)
        piece["tau_bucket"] = piece["tau_sec"].map(tau_bucket)
        piece["state_bucket"] = piece["z_bucket"].astype(str) + "|" + piece["tau_bucket"].astype(str)
        piece["spike_zone"] = piece["abs_z"].lt(SPIKE_ABS_Z) & piece["tau_min"].le(SPIKE_TAU_MIN)
        parts.append(piece)
    joined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if joined.empty:
        return joined
    joined["dt_next_sec"] = (
        joined.sort_values(["asset_id", "t_ns"])
        .groupby("asset_id")["t_ns"]
        .shift(-1)
        .sub(joined["t_ns"])
        .div(1e9)
        .clip(lower=0.0, upper=60.0)
        .fillna(0.0)
    )
    return joined.sort_values(["asset_id", "t_ns"]).reset_index(drop=True)


def build_state_dict(states: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    numeric = [
        "t_ns",
        "best_bid",
        "best_ask",
        "mid",
        "token_fair",
        "token_delta",
        "token_theta_per_sec",
        "binance_spot",
        "sigma_causal",
        "z",
        "abs_z",
        "tau_sec",
        "local_spread_bps",
    ]
    for aid, g in states.sort_values("t_ns").groupby("asset_id", sort=False):
        d = {col: g[col].to_numpy(dtype=np.int64 if col == "t_ns" else float) for col in numeric}
        for col in ("market_id", "slug", "family", "z_bucket", "tau_bucket", "state_bucket"):
            d[col] = g[col].astype(str).to_numpy(dtype=object)
        d["outcome_index"] = g["outcome_index"].to_numpy(dtype=int)
        d["window_end_ns"] = g["window_end_ns"].to_numpy(dtype=np.int64)
        d["window_start_ns"] = g["window_start_ns"].to_numpy(dtype=np.int64)
        d["window_strike_spot"] = g["window_strike_spot"].to_numpy(dtype=float)
        d["window_close_spot"] = g["window_close_spot"].to_numpy(dtype=float)
        d["spike_zone"] = g["spike_zone"].to_numpy(dtype=bool)
        out[str(aid)] = d
    return out


def build_trade_dict(trades: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for aid, g in trades.sort_values("t_ns").groupby("asset_id", sort=False):
        out[str(aid)] = {
            "t_ns": g["t_ns"].to_numpy(dtype=np.int64),
            "price": g["trade_price"].to_numpy(dtype=float),
            "side": g["trade_side"].astype(str).to_numpy(dtype=object),
        }
    return out


def build_candidate_pool(states: pd.DataFrame, trades: pd.DataFrame, settlement_by_token: dict[str, float]) -> pd.DataFrame:
    state_dict = build_state_dict(states)
    records = []
    for aid, tg in trades.groupby("asset_id", sort=False):
        st = state_dict.get(str(aid))
        if st is None:
            continue
        tg = tg.sort_values("t_ns").copy()
        trade_times = tg["t_ns"].to_numpy(dtype=np.int64)
        idx = np.searchsorted(st["t_ns"], trade_times, side="right") - 1
        valid = idx >= 0
        idx_safe = np.clip(idx, 0, len(st["t_ns"]) - 1)
        age = (trade_times - st["t_ns"][idx_safe]) / 1e9
        valid &= (age >= 0) & (age <= FILL_WINDOW_SEC)
        pos = np.flatnonzero(valid)
        if len(pos) == 0:
            continue
        qi = idx[pos]
        piece = tg.iloc[pos].copy()
        piece["quote_time_ns"] = st["t_ns"][qi]
        piece["quote_age_sec"] = age[pos]
        for col in (
            "best_bid",
            "best_ask",
            "mid",
            "token_fair",
            "token_delta",
            "token_theta_per_sec",
            "binance_spot",
            "sigma_causal",
            "z",
            "abs_z",
            "tau_sec",
            "local_spread_bps",
            "window_strike_spot",
            "window_close_spot",
        ):
            piece[col] = st[col][qi]
        for col in ("window_start_ns", "window_end_ns"):
            piece[col] = st[col][qi]
        for col in ("z_bucket", "tau_bucket", "state_bucket"):
            piece[col] = st[col][qi]
        piece["spike_zone"] = st["spike_zone"][qi]
        piece["settlement"] = settlement_by_token.get(str(aid), math.nan)
        records.append(piece)
    if not records:
        return pd.DataFrame()
    out = pd.concat(records, ignore_index=True).sort_values(["market_id", "asset_id", "t_ns"]).reset_index(drop=True)
    out["fill_time_ns"] = out["t_ns"].astype(np.int64)
    return out


def lookup(st: dict[str, np.ndarray], target_ns: int, col: str) -> float:
    return float(state_at_or_before(st["t_ns"], st[col], np.asarray(target_ns, dtype=np.int64)))


def lookup_bool(st: dict[str, np.ndarray], target_ns: int, col: str) -> bool:
    idx = np.searchsorted(st["t_ns"], np.asarray(target_ns, dtype=np.int64), side="right") - 1
    return bool(idx >= 0 and idx < len(st[col]) and st[col][idx])


def quote_half(row: Any, cfg: DefensiveConfig) -> tuple[float, float]:
    anchor = float(np.clip(row.token_fair, 0.001, 0.999))
    base_half = cfg.base_spread_bps / 10_000.0 * max(anchor, 0.01)
    e_abs_spot = (
        float(row.binance_spot)
        * float(row.sigma_causal)
        * math.sqrt(max(cfg.reaction_latency_sec, 0.001) / YEAR_SECONDS)
        * math.sqrt(2.0 / math.pi)
    )
    delta_half = cfg.delta_mult * abs(float(row.token_delta)) * e_abs_spot
    bucket_risk = 0.0
    if str(row.z_bucket).startswith("near_"):
        bucket_risk += 1.0
    if str(row.tau_bucket).startswith("late_"):
        bucket_risk += 1.0
    half = max(base_half, delta_half, TICK / 2.0) * (1.0 + cfg.bucket_widen * bucket_risk)
    return float(half), float(delta_half)


def defended_quote(row: Any, cfg: DefensiveConfig, token_side: int, state_dict: dict[str, dict[str, np.ndarray]]) -> tuple[float, float, float, float, bool, bool, float]:
    if bool(row.spike_zone):
        return math.nan, math.nan, math.nan, math.nan, True, False, 0.0
    anchor = float(np.clip(row.token_fair, 0.001, 0.999))
    half, delta_half = quote_half(row, cfg)
    bid = float(np.clip(anchor - half, 0.001, 0.999))
    ask = float(np.clip(anchor + half, 0.001, 0.999))

    st = state_dict.get(str(row.asset_id))
    if st is None or cfg.defense_mode == "none" or not np.isfinite(cfg.tox_band_prob):
        return bid, ask, half, delta_half, False, False, 0.0
    decision_ns = int(row.fill_time_ns) - int(round(cfg.reaction_latency_sec * 1_000_000_000))
    if decision_ns <= int(row.quote_time_ns):
        return bid, ask, half, delta_half, False, False, 0.0
    if lookup_bool(st, decision_ns, "spike_zone"):
        return math.nan, math.nan, half, delta_half, True, True, math.nan
    defense_fair = lookup(st, decision_ns, "token_fair")
    if not np.isfinite(defense_fair):
        return bid, ask, half, delta_half, False, False, 0.0
    adverse_move = (anchor - defense_fair) if token_side > 0 else (defense_fair - anchor)
    toxic = bool(adverse_move > cfg.tox_band_prob)
    if not toxic:
        return bid, ask, half, delta_half, False, False, adverse_move
    if cfg.defense_mode == "pull":
        return math.nan, math.nan, half, delta_half, True, True, adverse_move
    if token_side > 0:
        bid = min(bid, float(np.clip(defense_fair - half, 0.001, 0.999)))
    else:
        ask = max(ask, float(np.clip(defense_fair + half, 0.001, 0.999)))
    return bid, ask, half, delta_half, False, True, adverse_move


def exit_taker(row: Any, token_side: int, entry_price: float, state_dict: dict[str, dict[str, np.ndarray]]) -> dict[str, Any] | None:
    st = state_dict.get(str(row.asset_id))
    if st is None:
        return None
    latest_exit = int(row.window_end_ns) - RESOLUTION_BUFFER_SEC * 1_000_000_000
    target = min(int(row.fill_time_ns) + TAKER_HOLD_SEC * 1_000_000_000, latest_exit)
    if target <= int(row.fill_time_ns):
        return None
    bid = lookup(st, target, "best_bid")
    ask = lookup(st, target, "best_ask")
    if not np.isfinite(bid) or not np.isfinite(ask):
        return None
    exit_price = bid if token_side > 0 else ask
    fee = taker_fee_bps(exit_price, entry_price)
    gross = token_side * (exit_price - entry_price) / max(entry_price, 0.01) * 10_000.0
    return {"exit_time_ns": target, "exit_price": exit_price, "exit_fee_bps": fee, "gross_exit_bps": gross}


def build_fill(row: Any, cfg: DefensiveConfig, token_side: int, entry_price: float, bid: float, ask: float, half: float, delta_half: float, exit_info: dict[str, Any], state_dict: dict[str, dict[str, np.ndarray]], defense_triggered: bool, adverse_move_prob: float) -> dict[str, Any]:
    st = state_dict[str(row.asset_id)]
    denom = max(entry_price, 0.01)
    exit_ns = int(exit_info["exit_time_ns"])
    future_fair = lookup(st, exit_ns, "token_fair")
    future_spot = lookup(st, exit_ns, "binance_spot")
    adverse = -token_side * (future_fair - float(row.token_fair)) / denom * 10_000.0 if np.isfinite(future_fair) else math.nan
    delta_attr = (
        -token_side * float(row.token_delta) * (future_spot - float(row.binance_spot)) / denom * 10_000.0
        if np.isfinite(future_spot)
        else math.nan
    )
    theta_attr = (
        -token_side * float(row.token_theta_per_sec) * ((exit_ns - int(row.fill_time_ns)) / 1e9) / denom * 10_000.0
        if np.isfinite(float(row.token_theta_per_sec))
        else math.nan
    )
    settlement = float(row.settlement) if np.isfinite(float(row.settlement)) else math.nan
    resolution = -token_side * (settlement - float(row.token_fair)) / denom * 10_000.0 if np.isfinite(settlement) else math.nan
    spread_capture = token_side * (float(row.token_fair) - entry_price) / denom * 10_000.0
    rebate = maker_rebate_bps(entry_price)
    net = float(exit_info["gross_exit_bps"]) + rebate - float(exit_info["exit_fee_bps"])
    return {
        "row_type": "fill",
        "config_name": cfg.name,
        "base_spread_bps": cfg.base_spread_bps,
        "reaction_latency_sec": cfg.reaction_latency_sec,
        "tox_band_prob": cfg.tox_band_prob,
        "defense_mode": cfg.defense_mode,
        "delta_mult": cfg.delta_mult,
        "bucket_widen": cfg.bucket_widen,
        "market_id": str(row.market_id),
        "slug": str(row.slug),
        "asset_id": str(row.asset_id),
        "run_id": str(row.run_id),
        "family": str(row.family),
        "fill_time_ns": int(row.fill_time_ns),
        "exit_time_ns": exit_ns,
        "quote_time_ns": int(row.quote_time_ns),
        "fill_ts": ns_to_ts(int(row.fill_time_ns)),
        "exit_ts": ns_to_ts(exit_ns),
        "quote_age_sec": float(row.quote_age_sec),
        "maker_side": "BUY" if token_side > 0 else "SELL",
        "token_side": token_side,
        "trade_side": str(row.trade_side),
        "trade_price": float(row.trade_price),
        "entry_price": entry_price,
        "quote_bid": bid,
        "quote_ask": ask,
        "half_spread_prob": half,
        "delta_half_prob": delta_half,
        "defense_triggered": defense_triggered,
        "adverse_fair_move_prob": adverse_move_prob,
        "local_mid": float(row.mid),
        "local_spread_bps": float(row.local_spread_bps),
        "digital_fair": float(row.token_fair),
        "binance_spot": float(row.binance_spot),
        "sigma_causal": float(row.sigma_causal),
        "z": float(row.z),
        "abs_z": float(row.abs_z),
        "token_delta": float(row.token_delta),
        "tau_min": float(row.tau_sec) / 60.0,
        "z_bucket": str(row.z_bucket),
        "tau_bucket": str(row.tau_bucket),
        "state_bucket": str(row.state_bucket),
        "settlement": settlement,
        "exit_price": float(exit_info["exit_price"]),
        "spread_capture_bps": spread_capture,
        "rebate_bps": rebate,
        "exit_fee_bps": float(exit_info["exit_fee_bps"]),
        "gross_exit_bps": float(exit_info["gross_exit_bps"]),
        "adverse_selection_bps": adverse,
        "delta_spot_attr_bps": delta_attr,
        "inventory_theta_bps": theta_attr,
        "resolution_risk_bps": resolution,
        "net_pnl_bps": net,
    }


def simulate(candidates: pd.DataFrame, cfg: DefensiveConfig, state_dict: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for _, group in candidates.sort_values(["market_id", "asset_id", "fill_time_ns"]).groupby(["market_id", "asset_id"], sort=False):
        open_until = -1
        for row in group.itertuples(index=False):
            now = int(row.fill_time_ns)
            if now < open_until:
                continue
            side = str(row.trade_side)
            token_side = 1 if side == "SELL" else -1
            bid, ask, half, delta_half, disabled, defense_triggered, adverse_move = defended_quote(row, cfg, token_side, state_dict)
            if disabled or not np.isfinite(bid) or not np.isfinite(ask):
                continue
            trade_price = float(row.trade_price)
            entry_price = math.nan
            if token_side > 0 and trade_price <= bid + 1e-12:
                entry_price = bid
            elif token_side < 0 and trade_price >= ask - 1e-12:
                entry_price = ask
            if not np.isfinite(entry_price):
                continue
            exit_info = exit_taker(row, token_side, entry_price, state_dict)
            if exit_info is None:
                continue
            open_until = int(exit_info["exit_time_ns"])
            rows.append(
                build_fill(
                    row,
                    cfg,
                    token_side,
                    entry_price,
                    bid,
                    ask,
                    half,
                    delta_half,
                    exit_info,
                    state_dict,
                    defense_triggered,
                    adverse_move,
                )
            )
    return pd.DataFrame(rows)


def config_grid() -> list[DefensiveConfig]:
    configs: list[DefensiveConfig] = []
    for base in BASE_SPREAD_BPS:
        for latency in REACTION_LATENCIES:
            for delta_mult in DELTA_MULTS:
                configs.append(
                    DefensiveConfig(
                        name=f"base{int(base)}_L{int(latency)}_m{delta_mult:g}_no_defense",
                        base_spread_bps=base,
                        reaction_latency_sec=latency,
                        tox_band_prob=math.inf,
                        defense_mode="none",
                        delta_mult=delta_mult,
                    )
                )
                for tox in TOX_BANDS:
                    for mode in DEFENSE_MODES:
                        configs.append(
                            DefensiveConfig(
                                name=f"base{int(base)}_L{int(latency)}_m{delta_mult:g}_{mode}_tox{int(tox*10000)}bp",
                                base_spread_bps=base,
                                reaction_latency_sec=latency,
                                tox_band_prob=tox,
                                defense_mode=mode,
                                delta_mult=delta_mult,
                            )
                        )
    return configs


def objective(row: pd.Series) -> float:
    n = int(row["n_fills"])
    mean = float(row["mean_net_pnl_bps"])
    std = float(row["std_net_pnl_bps"])
    if n < ROBUST_MIN_FILLS or not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        return -1e9 + n
    return mean / std * math.sqrt(n)


def summarize(fills: pd.DataFrame, row_type: str, group_cols: list[str] | None = None) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame()
    group_cols = group_cols or []
    rows = []
    grouped = [((), fills)] if not group_cols else list(fills.groupby(group_cols, dropna=False, sort=True))
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        lo, hi = bootstrap_mean_ci(g, "net_pnl_bps", RNG_SEED + len(g))
        rec: dict[str, Any] = {
            "row_type": row_type,
            "n_fills": int(len(g)),
            "n_markets": int(g["market_id"].nunique()),
            "n_assets": int(g["asset_id"].nunique()),
            "mean_net_pnl_bps": float(g["net_pnl_bps"].mean()),
            "median_net_pnl_bps": float(g["net_pnl_bps"].median()),
            "std_net_pnl_bps": float(g["net_pnl_bps"].std(ddof=1)) if len(g) > 1 else math.nan,
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_lower_positive": bool(np.isfinite(lo) and lo > 0),
            "clears_zero": bool(len(g) >= ROBUST_MIN_FILLS and np.isfinite(lo) and lo > 0),
            "win_rate": float(g["net_pnl_bps"].gt(0).mean()),
            "defense_trigger_rate": float(g["defense_triggered"].mean()),
            "mean_spread_capture_bps": float(g["spread_capture_bps"].mean()),
            "mean_rebate_bps": float(g["rebate_bps"].mean()),
            "mean_exit_fee_bps": float(g["exit_fee_bps"].mean()),
            "mean_gross_exit_bps": float(g["gross_exit_bps"].mean()),
            "mean_adverse_selection_bps": float(g["adverse_selection_bps"].mean()),
            "mean_delta_spot_attr_bps": float(g["delta_spot_attr_bps"].mean()),
            "mean_inventory_theta_bps": float(g["inventory_theta_bps"].mean()),
            "mean_resolution_risk_bps": float(g["resolution_risk_bps"].mean()),
            "mean_local_spread_bps": float(g["local_spread_bps"].mean()),
            "mean_abs_z": float(g["abs_z"].mean()),
            "mean_tau_min": float(g["tau_min"].mean()),
            "top_market_share": float(g["market_id"].value_counts(normalize=True).iloc[0]),
        }
        for col in ("config_name", "base_spread_bps", "reaction_latency_sec", "tox_band_prob", "defense_mode", "delta_mult"):
            if col in g:
                rec[col] = g[col].iloc[0]
        for col, key in zip(group_cols, keys, strict=False):
            rec[col] = str(key)
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["objective"] = out.apply(objective, axis=1)
    return out


def source_rows(model: pd.DataFrame) -> pd.DataFrame:
    if model.empty:
        return pd.DataFrame()
    latest = model.sort_values("ts_ns").groupby("slug", sort=False).tail(1)
    rows = []
    for r in latest.itertuples(index=False):
        rows.append(
            {
                "row_type": "source_basis",
                "slug": str(r.slug),
                "family": str(r.family),
                "source_ok_strict": bool(r.source_ok_strict),
                "source_direction_mismatch": bool(r.source_direction_mismatch),
                "binance_window_abs_return_bps": float(r.binance_window_abs_return_bps),
            }
        )
    return pd.DataFrame(rows)


def write_note(results: pd.DataFrame, fills: pd.DataFrame, states: pd.DataFrame, candidates: pd.DataFrame, daily_meta: dict[str, dict[str, Any]]) -> None:
    config_rows = results[results["row_type"].eq("config_search")].copy()
    bucket_rows = results[results["row_type"].eq("bucket")].copy()
    best_config = config_rows[config_rows["n_fills"] >= ROBUST_MIN_FILLS].sort_values(
        ["ci_lo", "mean_net_pnl_bps"], ascending=[False, False]
    ).head(1)
    robust = bucket_rows[bucket_rows["clears_zero"].fillna(False)].copy()
    if not robust.empty:
        best_bucket = robust.sort_values(["ci_lo", "mean_net_pnl_bps"], ascending=[False, False]).iloc[0]
        headline = (
            "At least one defensive maker bucket clears zero after costs in-sample, but treat it as exploratory "
            "unless it is the pre-specified moderate bucket."
        )
    else:
        best_bucket_df = bucket_rows[bucket_rows["n_fills"] >= 5].sort_values(["ci_lo", "mean_net_pnl_bps"], ascending=[False, False]).head(1)
        best_bucket = best_bucket_df.iloc[0] if not best_bucket_df.empty else None
        headline = "No Binance-anchored defensive maker bucket clears zero after costs; the maker thesis remains closed."

    selected_config_name = ""
    if not best_config.empty:
        bc = best_config.iloc[0]
        selected_config_name = str(bc.config_name)
        best_config_text = (
            f"Best full-sample config: `{bc.config_name}`, n={int(bc.n_fills)}, mean {bps(float(bc.mean_net_pnl_bps))}, "
            f"CI [{bps(float(bc.ci_lo))}, {bps(float(bc.ci_hi))}], "
            f"defense trigger {pct(float(bc.defense_trigger_rate))}."
        )
    else:
        best_config_text = "No config reached the minimum fill count."

    if best_bucket is not None:
        best_bucket_text = (
            f"Best bucket: `{best_bucket.get('z_bucket', '')}|{best_bucket.get('tau_bucket', '')}` under "
            f"`{best_bucket.config_name}`, n={int(best_bucket.n_fills)}, mean {bps(float(best_bucket.mean_net_pnl_bps))}, "
            f"CI [{bps(float(best_bucket.ci_lo))}, {bps(float(best_bucket.ci_hi))}]."
        )
    else:
        best_bucket_text = "No evaluable bucket rows."

    top_rows = []
    for _, r in config_rows.sort_values(["ci_lo", "mean_net_pnl_bps"], ascending=[False, False]).head(10).iterrows():
        top_rows.append(
            [
                str(r.config_name),
                str(int(r.n_fills)),
                str(int(r.n_markets)),
                bps(float(r.mean_net_pnl_bps)),
                f"[{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}]",
                pct(float(r.win_rate)),
                pct(float(r.defense_trigger_rate)),
                bps(float(r.mean_spread_capture_bps)),
                bps(float(r.mean_rebate_bps)),
                bps(float(r.mean_exit_fee_bps)),
            ]
        )

    bucket_table = []
    for _, r in bucket_rows.sort_values(["ci_lo", "mean_net_pnl_bps"], ascending=[False, False]).head(12).iterrows():
        bucket_table.append(
            [
                str(r.config_name),
                str(r.get("z_bucket", "")),
                str(r.get("tau_bucket", "")),
                str(int(r.n_fills)),
                bps(float(r.mean_net_pnl_bps)),
                f"[{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}]",
                "yes" if bool(r.clears_zero) else "no",
            ]
        )

    family_rows = []
    family_source = fills[fills["config_name"].eq(selected_config_name)].copy() if selected_config_name else fills
    for fam, g in family_source.groupby("family", sort=True):
        lo, hi = bootstrap_mean_ci(g, "net_pnl_bps", RNG_SEED + len(g) + 77)
        family_rows.append(
            [
                str(fam),
                str(int(len(g))),
                str(int(g["market_id"].nunique())),
                bps(float(g["net_pnl_bps"].mean())),
                f"[{bps(lo)}, {bps(hi)}]",
                bps(float(g["rebate_bps"].mean())),
                bps(float(g["exit_fee_bps"].mean())),
            ]
        )

    guard = results[results["row_type"].eq("guardrail")]
    guard_text = ""
    if not guard.empty:
        gr = guard.iloc[0]
        guard_text = (
            f"Bucket tests: {int(gr.n_bucket_tests)}; raw CI-positive buckets: {int(gr.n_raw_ci_positive_buckets)}; "
            f"robust clears with n>={ROBUST_MIN_FILLS}: {int(gr.n_robust_clears)}."
        )
    max_defense_rate = float(config_rows["defense_trigger_rate"].max()) if not config_rows.empty else math.nan
    total_defense_triggers = int(fills["defense_triggered"].sum()) if not fills.empty else 0
    max_adverse_move = float(fills["adverse_fair_move_prob"].max()) if not fills.empty else math.nan
    total_defense_rate = float(fills["defense_triggered"].mean()) if not fills.empty else math.nan
    total_defense_rate_text = "<0.1%" if np.isfinite(total_defense_rate) and 0 < total_defense_rate < 0.001 else pct(total_defense_rate)
    defense_text = (
        f"Defense opportunity was scarce: {total_defense_triggers:,} fills triggered the pull/widen rule across the whole grid "
        f"({total_defense_rate_text} of simulated fills), "
        f"the largest per-config trigger rate was {pct(max_defense_rate)}, and the largest observed pre-fill adverse fair move "
        f"was {cents(max_adverse_move)}. Config labels round the smallest toxicity band to `tox0bp`; use the numeric "
        f"`tox_band_prob` column in the CSV for exact thresholds."
    )

    text = f"""# Block K2 v2 Defensive Maker Findings

Generated: {datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")}

## Headline

{headline}

{best_config_text}

{best_bucket_text}

{guard_text}

{defense_text}

## Best Configs

{markdown_table(["config", "fills", "mkts", "mean", "95% CI", "win", "defense", "spread", "rebate", "exit_fee"], top_rows)}

## Best Buckets

{markdown_table(["config", "|z|", "tau", "fills", "mean", "95% CI", "clears"], bucket_table)}

## Family Split

Selected best config only.

{markdown_table(["family", "fills", "markets", "mean", "95% CI", "rebate", "exit_fee"], family_rows)}

## Method

- Universe: A1 feature panel, pooled `a0b+a0c_roll`, families `crypto_4h_up_down` and `daily_crypto_up_down`; no holdout.
- Reservation price: Binance-implied European digital fair `N(z)`, not Polymarket mid.
- 4h surface: reused `{FEATURE_CACHE.relative_to(ROOT)}`.
- Daily surface: rebuilt from Binance 1s spot candles using Gamma `eventStartTime/endDate`; cache `{DAILY_MODEL_CACHE.relative_to(ROOT)}`.
- Defensive rule: if Binance token fair at `fill_time - reaction_latency` moves away from the resting side by more than `tox_band`, either pull the quote or widen that side before the Polymarket taker print.
- Fill proxy: A1.4h style, one-share passive fill only when a real taker print crosses the modeled quote within {FILL_WINDOW_SEC}s of the quote state.
- Exit: taker after {TAKER_HOLD_SEC}s or before `window_end - {RESOLUTION_BUFFER_SEC}s`; entry maker rebate and exit taker fee use the K1 crypto fee/rebate table.
- Hard flatten: no quotes in `abs_z < {SPIKE_ABS_Z}` and `tau <= {SPIKE_TAU_MIN:.0f}m`.
- CI: {BOOTSTRAP_SAMPLES} bootstrap samples over fill-time market blocks.

## Diagnostics

- Quote states with model: {len(states):,}
- Candidate taker prints with fresh prior quote: {len(candidates):,}
- Simulated fills across all configs: {len(fills):,}
- Daily markets modeled: {len(daily_meta)}

## Outputs

- Summary CSV: `data/analysis/csv_outputs/market_making/k2v2_defensive_maker.csv`
- Fill ledger: `data/analysis/k2v2_defensive_maker_fills.parquet`
- Repro script: `scripts/dali_block_k2v2_defensive_maker.py`
"""
    NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {NOTE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-binance", action="store_true")
    parser.add_argument("--refresh-daily-model", action="store_true")
    args = parser.parse_args()

    states_raw, trades, markets = load_feature_tables()
    gamma_cache = load_gamma_cache(markets["market_id"].astype(str).tolist())
    settlement_by_token, _ = gamma_settlement_maps(gamma_cache)
    family_by_slug, daily_meta = family_slug_maps(markets, gamma_cache)
    model4h = build_4h_model_surface(family_by_slug)
    model_daily = build_daily_model_surface(
        daily_meta,
        refresh_binance=args.refresh_binance,
        refresh_model=args.refresh_daily_model,
    )
    model = pd.concat([model4h, model_daily], ignore_index=True, sort=False)
    print(f"model rows 4h={len(model4h):,} daily={len(model_daily):,}", flush=True)
    states = attach_model(states_raw, model)
    print(f"states with model {len(states):,}", flush=True)
    state_dict = build_state_dict(states)
    candidates = build_candidate_pool(states, trades, settlement_by_token)
    print(f"candidate prints {len(candidates):,}", flush=True)

    config_summaries = []
    all_fills = []
    for i, cfg in enumerate(config_grid(), start=1):
        print(f"sim {i}/{len(config_grid())} {cfg.name}", flush=True)
        fills = simulate(candidates, cfg, state_dict)
        if fills.empty:
            continue
        all_fills.append(fills)
        config_summaries.append(summarize(fills, "config_search"))

    fills_all = pd.concat(all_fills, ignore_index=True, sort=False) if all_fills else pd.DataFrame()
    config_summary = pd.concat(config_summaries, ignore_index=True, sort=False) if config_summaries else pd.DataFrame()

    bucket_summary = summarize(fills_all, "bucket", ["config_name", "z_bucket", "tau_bucket"]) if not fills_all.empty else pd.DataFrame()
    guard = pd.DataFrame(
        [
            {
                "row_type": "guardrail",
                "n_bucket_tests": int(len(bucket_summary)),
                "n_raw_ci_positive_buckets": int(bucket_summary["ci_lower_positive"].fillna(False).sum()) if not bucket_summary.empty else 0,
                "n_robust_clears": int(bucket_summary["clears_zero"].fillna(False).sum()) if not bucket_summary.empty else 0,
                "robust_min_fills": ROBUST_MIN_FILLS,
            }
        ]
    )
    pieces = [config_summary, bucket_summary, guard, source_rows(model)]
    results = pd.concat([p for p in pieces if p is not None and not p.empty], ignore_index=True, sort=False)
    results.to_csv(OUT_CSV, index=False)
    if not fills_all.empty:
        fills_all.to_parquet(OUT_FILLS, index=False)
    write_note(results, fills_all, states, candidates, daily_meta)
    print(f"wrote {OUT_CSV}", flush=True)
    if not fills_all.empty:
        print(f"wrote {OUT_FILLS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
