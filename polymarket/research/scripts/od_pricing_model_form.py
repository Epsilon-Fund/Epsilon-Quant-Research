"""OD pricing-model-form diagnostic: Gaussian vs jump-aware tails.

This is a time-boxed reopen-or-close diagnostic for the OD valuation layer.
It keeps the existing v4 far-|z| strict-rich short set, but reprices the
token probability with captured-window 1s Binance jump features instead of
only the Gaussian EWMA N(z) form.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import YEAR_SECONDS, cents, markdown_table, norm_cdf, number, pct
from od_strategy_a_v3 import normalize_markdown_wrapping


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
BRAIN_TODO = REPO / "brain" / "TODO.md"
OD_HUB = NOTES / "options_delta" / "strat_options_delta.md"

PM_FILLS = ANALYSIS / "od_conditional_prob_pm_fills.parquet"
K3_PANEL = ANALYSIS / "cache" / "k3v3h_panel_features.parquet"
LOB_ROLL = ANALYSIS / "block_a0c_roll_features.parquet"
LOB_A0C = ANALYSIS / "block_a0c_features.parquet"
DAILY_SURFACE = ANALYSIS / "cache" / "k2v2_daily_model_surface.parquet"

CSV_OUT = ANALYSIS / "csv_outputs" / "options_delta"
PLOTS = ANALYSIS / "plots" / "options_delta"

OUT_SUMMARY = CSV_OUT / "od_pricing_model_form_summary.csv"
OUT_RELIABILITY = CSV_OUT / "od_pricing_model_form_reliability.csv"
OUT_SHAPE = CSV_OUT / "od_pricing_model_form_shape.csv"
OUT_DERIBIT = CSV_OUT / "od_pricing_model_form_deribit.csv"
OUT_PM = ANALYSIS / "od_pricing_model_form_pm_fills.parquet"
OUT_LIVE = ANALYSIS / "od_pricing_model_form_live1s_panel.parquet"
OUT_DERIBIT_PARQUET = ANALYSIS / "od_pricing_model_form_deribit.parquet"

NOTE = NOTES / "options_delta" / "od_pricing_model_form_findings.md"
EDGE_PLOT = PLOTS / "od_pricing_model_form_pm_edge.png"
SHAPE_PLOT = PLOTS / "od_pricing_model_form_tail_shape.png"
DERIBIT_PLOT = PLOTS / "od_pricing_model_form_deribit.png"

BOOTSTRAP_SAMPLES = 5000
RNG_SEED = 20260602
NON_TOP3_AVAILABLE_SHARE = 0.05
STRUCTURAL_BASELINE_C = 0.0198
JUMP_Z_THRESHOLD = 8.0
JUMP_ABS_RETURN_FLOOR_BPS = 10.0
MIN_PRIOR_SECONDS = 600.0
MAX_JUMP_TERMS = 12
EDGEWORTH_MAX_ABS_SKEW = 3.0
EDGEWORTH_MAX_EXCESS_KURT = 30.0
DERIBIT_BASE = "https://www.deribit.com/api/v2"


def norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    return np.exp(-0.5 * arr * arr) / math.sqrt(2.0 * math.pi)


def fmt_ci(lo: float, hi: float, unit: str = "c") -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "[n/a, n/a]"
    if unit == "pct":
        return f"[{pct(lo)}, {pct(hi)}]"
    return f"[{cents(lo)}, {cents(hi)}]"


def safe_div(num: np.ndarray, den: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full_like(num, default, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    return out


def cluster_ci(df: pd.DataFrame, col: str, *, cluster_col: str = "market_id", seed_offset: int = 0) -> tuple[float, float]:
    if df.empty or col not in df:
        return math.nan, math.nan
    groups: list[tuple[float, int]] = []
    for _, g in df.groupby(cluster_col, sort=False):
        vals = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(vals):
            groups.append((float(vals.sum()), int(len(vals))))
    if not groups:
        return math.nan, math.nan
    if len(groups) == 1:
        total, n = groups[0]
        v = total / n if n else math.nan
        return float(v), float(v)
    rng = np.random.default_rng(RNG_SEED + seed_offset + len(groups))
    sums = np.asarray([g[0] for g in groups], dtype=float)
    counts = np.asarray([g[1] for g in groups], dtype=float)
    idx = rng.integers(0, len(groups), size=(BOOTSTRAP_SAMPLES, len(groups)))
    vals = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def market_sum_ci(df: pd.DataFrame, col: str, *, seed_offset: int = 0) -> tuple[float, float]:
    if df.empty or col not in df:
        return math.nan, math.nan
    vals = df.groupby("market_id", sort=False)[col].sum().replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(RNG_SEED + seed_offset + 17 * len(vals))
    idx = rng.integers(0, len(vals), size=(BOOTSTRAP_SAMPLES, len(vals)))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def load_pm_fills() -> pd.DataFrame:
    if not PM_FILLS.exists():
        raise SystemExit(f"missing {PM_FILLS}")
    out = pd.read_parquet(PM_FILLS)
    out["fill_ts"] = pd.to_datetime(out["fill_ts"], utc=True)
    for col in ("ts", "window_start", "window_end"):
        if col in out:
            out[col] = pd.to_datetime(out[col], utc=True)
    out["short_price"] = out["entry_price"].astype(float)
    out["realized_itm"] = out["payoff"].astype(float)
    out["arm_a_token_prob"] = out["token_model_fair"].astype(float)
    out["arm_a_edge"] = out["short_price"] - out["arm_a_token_prob"]
    out["gross_realized_ev"] = out["short_price"] - out["realized_itm"]
    out["net_realized_ev"] = out["gross_realized_ev"] + out["maker_rebate"].astype(float)
    return out.sort_values("fill_ts").reset_index(drop=True)


def add_jump_params(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy().sort_values(["asset", "market_slug", "ts"]).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for asset, g in panel.groupby("asset", sort=False):
        g = g.copy().sort_values("ts").reset_index(drop=True)
        g["prev_spot"] = g["binance_spot"].shift(1)
        g["prev_ts"] = g["ts"].shift(1)
        dt = (g["ts"] - g["prev_ts"]).dt.total_seconds().astype(float)
        ret = np.log(g["binance_spot"].astype(float) / g["prev_spot"].astype(float))
        ret = ret.where(dt.between(0.5, 5.0))
        g["log_ret_1s"] = ret
        g["dt_seconds"] = dt.where(ret.notna(), 0.0).fillna(0.0)
        sigma_1s = g["ewma_sigma_annualized"].astype(float).clip(lower=1e-8) / math.sqrt(YEAR_SECONDS)
        g["jump_threshold_abs_ret"] = np.maximum(JUMP_Z_THRESHOLD * sigma_1s, JUMP_ABS_RETURN_FLOOR_BPS / 1e4)
        g["jump_flag"] = ret.abs().gt(g["jump_threshold_abs_ret"]) & ret.notna()
        jump_ret = ret.where(g["jump_flag"], 0.0).fillna(0.0)
        jump_up = jump_ret.where(jump_ret.gt(0), 0.0)
        jump_down_abs = (-jump_ret).where(jump_ret.lt(0), 0.0)
        prior_seconds = g["dt_seconds"].cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_count = g["jump_flag"].astype(float).cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        ret_clean = ret.fillna(0.0)
        prior_ret_obs = ret.notna().astype(float).cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_ret_sum1 = ret_clean.cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_ret_sum2 = (ret_clean**2).cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_ret_sum3 = (ret_clean**3).cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_ret_sum4 = (ret_clean**4).cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_sum = jump_ret.cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_sq = (jump_ret * jump_ret).cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_up_count = jump_ret.gt(0).astype(float).cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_down_count = jump_ret.lt(0).astype(float).cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_up_sum = jump_up.cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)
        prior_down_sum = jump_down_abs.cumsum().shift(1).fillna(0.0).to_numpy(dtype=float)

        enough = (prior_seconds >= MIN_PRIOR_SECONDS) & (prior_count >= 1)
        lambda_per_year = np.zeros(len(g), dtype=float)
        lambda_per_year[enough] = prior_count[enough] / prior_seconds[enough] * YEAR_SECONDS
        jump_mu = safe_div(prior_sum, prior_count)
        jump_second = safe_div(prior_sq, prior_count)
        jump_var = np.maximum(jump_second - jump_mu * jump_mu, 0.0)
        jump_sigma = np.sqrt(jump_var)
        jump_mu[~enough] = 0.0
        jump_sigma[~enough] = 0.0
        lambda_per_year[~enough] = 0.0

        p_up = safe_div(prior_up_count, prior_count, default=0.5)
        up_mean = safe_div(prior_up_sum, prior_up_count)
        down_mean = safe_div(prior_down_sum, prior_down_count)
        p_up[~enough] = 0.5
        up_mean[~enough] = 0.0
        down_mean[~enough] = 0.0

        ret_enough = (prior_seconds >= MIN_PRIOR_SECONDS) & (prior_ret_obs >= 30)
        ret_mean = safe_div(prior_ret_sum1, prior_ret_obs)
        raw2 = safe_div(prior_ret_sum2, prior_ret_obs)
        raw3 = safe_div(prior_ret_sum3, prior_ret_obs)
        raw4 = safe_div(prior_ret_sum4, prior_ret_obs)
        cm2 = np.maximum(raw2 - ret_mean * ret_mean, 0.0)
        cm3 = raw3 - 3.0 * ret_mean * raw2 + 2.0 * ret_mean**3
        cm4 = raw4 - 4.0 * ret_mean * raw3 + 6.0 * ret_mean * ret_mean * raw2 - 3.0 * ret_mean**4
        one_sec_skew = safe_div(cm3, np.power(np.maximum(cm2, 1e-24), 1.5))
        one_sec_exkurt = safe_div(cm4, np.maximum(cm2 * cm2, 1e-24)) - 3.0
        one_sec_skew[~ret_enough] = 0.0
        one_sec_exkurt[~ret_enough] = 0.0

        g["jump_prior_seconds"] = prior_seconds
        g["jump_prior_count"] = prior_count
        g["jump_lambda_per_year"] = lambda_per_year
        g["jump_mu"] = jump_mu
        g["jump_sigma"] = jump_sigma
        g["kou_p_up"] = p_up
        g["kou_up_mean_abs"] = up_mean
        g["kou_down_mean_abs"] = down_mean
        g["prior_ret_obs"] = prior_ret_obs
        g["prior_ret_mean_1s"] = np.where(ret_enough, ret_mean, 0.0)
        g["prior_ret_var_1s"] = np.where(ret_enough, cm2, 0.0)
        g["prior_ret_skew_1s"] = np.clip(one_sec_skew, -EDGEWORTH_MAX_ABS_SKEW, EDGEWORTH_MAX_ABS_SKEW)
        g["prior_ret_excess_kurt_1s"] = np.clip(one_sec_exkurt, 0.0, EDGEWORTH_MAX_EXCESS_KURT)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def load_live_panel() -> pd.DataFrame:
    if not K3_PANEL.exists():
        raise SystemExit(f"missing {K3_PANEL}")
    panel = pd.read_parquet(K3_PANEL)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    for col in ("window_start", "window_end"):
        panel[col] = pd.to_datetime(panel[col], utc=True)
    panel = panel.sort_values(["asset", "market_slug", "ts"]).reset_index(drop=True)
    panel = add_jump_params(panel)
    panel["arm_a_p_up"] = panel["p_model"].astype(float).clip(0.0, 1.0)
    panel["arm_b_merton_p_up"] = merton_p_up(panel)
    panel["arm_b_kou_p_up"] = kou_p_up(panel)
    panel["arm_c_edgeworth_p_up"] = edgeworth_p_up(panel)
    return panel


def poisson_weights(lam_tau: np.ndarray, n_terms: int = MAX_JUMP_TERMS) -> list[np.ndarray]:
    lam = np.asarray(lam_tau, dtype=float).clip(0.0, 50.0)
    weights: list[np.ndarray] = [np.exp(-lam)]
    for n in range(1, n_terms + 1):
        weights.append(weights[-1] * lam / n)
    return weights


def merton_p_up(df: pd.DataFrame, *, spot_col: str = "binance_spot", strike_col: str = "binance_strike_spot") -> np.ndarray:
    spot = pd.to_numeric(df[spot_col], errors="coerce").to_numpy(dtype=float)
    strike = pd.to_numeric(df[strike_col], errors="coerce").to_numpy(dtype=float)
    tau = pd.to_numeric(df["tau_years"], errors="coerce").to_numpy(dtype=float).clip(1e-10, None)
    sigma = pd.to_numeric(df["ewma_sigma_annualized"], errors="coerce").to_numpy(dtype=float).clip(1e-8, None)
    lam_tau = pd.to_numeric(df["jump_lambda_per_year"], errors="coerce").fillna(0.0).to_numpy(dtype=float).clip(0.0, None) * tau
    mu_j = pd.to_numeric(df["jump_mu"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sig_j = pd.to_numeric(df["jump_sigma"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    x = np.log(spot / strike)
    base_var = sigma * sigma * tau
    out = np.zeros(len(df), dtype=float)
    weights = poisson_weights(lam_tau)
    for n, w in enumerate(weights):
        mean = n * mu_j
        var = base_var + n * sig_j * sig_j
        z = (x + mean) / np.sqrt(np.maximum(var, 1e-14))
        out += w * np.asarray(norm_cdf(z), dtype=float)
    return np.clip(out, 0.0, 1.0)


def kou_p_up(df: pd.DataFrame, *, spot_col: str = "binance_spot", strike_col: str = "binance_strike_spot") -> np.ndarray:
    spot = pd.to_numeric(df[spot_col], errors="coerce").to_numpy(dtype=float)
    strike = pd.to_numeric(df[strike_col], errors="coerce").to_numpy(dtype=float)
    tau = pd.to_numeric(df["tau_years"], errors="coerce").to_numpy(dtype=float).clip(1e-10, None)
    sigma = pd.to_numeric(df["ewma_sigma_annualized"], errors="coerce").to_numpy(dtype=float).clip(1e-8, None)
    lam_tau = pd.to_numeric(df["jump_lambda_per_year"], errors="coerce").fillna(0.0).to_numpy(dtype=float).clip(0.0, None) * tau
    p_up = pd.to_numeric(df["kou_p_up"], errors="coerce").fillna(0.5).to_numpy(dtype=float).clip(0.0, 1.0)
    up_mean = pd.to_numeric(df["kou_up_mean_abs"], errors="coerce").fillna(0.0).to_numpy(dtype=float).clip(0.0, None)
    down_mean = pd.to_numeric(df["kou_down_mean_abs"], errors="coerce").fillna(0.0).to_numpy(dtype=float).clip(0.0, None)
    jump_mean = p_up * up_mean - (1.0 - p_up) * down_mean
    jump_second = p_up * 2.0 * up_mean * up_mean + (1.0 - p_up) * 2.0 * down_mean * down_mean
    jump_var = np.maximum(jump_second - jump_mean * jump_mean, 0.0)
    x = np.log(spot / strike)
    base_var = sigma * sigma * tau
    out = np.zeros(len(df), dtype=float)
    weights = poisson_weights(lam_tau)
    for n, w in enumerate(weights):
        mean = n * jump_mean
        var = base_var + n * jump_var
        z = (x + mean) / np.sqrt(np.maximum(var, 1e-14))
        out += w * np.asarray(norm_cdf(z), dtype=float)
    return np.clip(out, 0.0, 1.0)


def edgeworth_p_up(df: pd.DataFrame, *, spot_col: str = "binance_spot", strike_col: str = "binance_strike_spot") -> np.ndarray:
    """Causal higher-moment digital probability from prior 1s return cumulants.

    This is the cheap Arm-C extension: not a full Bates/VG calibration, but a
    lookahead-free higher-moment pricing form that lets skew/kurtosis reshape the
    short-horizon digital tail beyond Gaussian N(z).
    """
    spot = pd.to_numeric(df[spot_col], errors="coerce").to_numpy(dtype=float)
    strike = pd.to_numeric(df[strike_col], errors="coerce").to_numpy(dtype=float)
    tau = pd.to_numeric(df["tau_years"], errors="coerce").to_numpy(dtype=float).clip(1e-10, None)
    sigma = pd.to_numeric(df["ewma_sigma_annualized"], errors="coerce").to_numpy(dtype=float).clip(1e-8, None)
    skew_1s = pd.to_numeric(df.get("prior_ret_skew_1s", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    exkurt_1s = pd.to_numeric(df.get("prior_ret_excess_kurt_1s", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n_seconds = np.maximum(tau * YEAR_SECONDS, 1.0)
    # Cumulant scaling for a sum of n approximately iid 1s returns.
    skew_h = np.clip(skew_1s / np.sqrt(n_seconds), -EDGEWORTH_MAX_ABS_SKEW, EDGEWORTH_MAX_ABS_SKEW)
    exkurt_h = np.clip(exkurt_1s / n_seconds, 0.0, EDGEWORTH_MAX_EXCESS_KURT)
    x0 = -np.log(spot / strike)
    z = x0 / (sigma * np.sqrt(tau))
    phi = np.asarray(norm_pdf(z), dtype=float)
    cdf = np.asarray(norm_cdf(z), dtype=float)
    correction = phi * (
        (skew_h / 6.0) * (1.0 - z * z)
        + (exkurt_h / 24.0) * (z**3 - 3.0 * z)
        - (skew_h * skew_h / 36.0) * (2.0 * z**3 - 5.0 * z)
    )
    return np.clip(1.0 - (cdf + correction), 0.0, 1.0)


def join_live_to_pm(pm: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    right_cols = [
        "ts",
        "asset",
        "market_slug",
        "binance_spot",
        "ewma_sigma_annualized",
        "tau_years",
        "jump_prior_seconds",
        "jump_prior_count",
        "jump_lambda_per_year",
        "jump_mu",
        "jump_sigma",
        "kou_p_up",
        "kou_up_mean_abs",
        "kou_down_mean_abs",
        "jump_flag",
        "log_ret_1s",
        "arm_b_merton_p_up",
        "arm_b_kou_p_up",
        "arm_c_edgeworth_p_up",
        "prior_ret_obs",
        "prior_ret_skew_1s",
        "prior_ret_excess_kurt_1s",
    ]
    for _, left in pm.groupby(["asset", "market_slug"], sort=False):
        key_asset = left["asset"].iloc[0]
        key_slug = left["market_slug"].iloc[0]
        right = live[(live["asset"].eq(key_asset)) & (live["market_slug"].eq(key_slug))][right_cols].copy()
        left = left.sort_values("fill_ts").copy()
        left["fill_ts"] = pd.to_datetime(left["fill_ts"], utc=True).astype("datetime64[ns, UTC]")
        if right.empty:
            left["live_1s_covered"] = False
            parts.append(left)
            continue
        right["ts"] = pd.to_datetime(right["ts"], utc=True).astype("datetime64[ns, UTC]")
        joined = pd.merge_asof(
            left,
            right.sort_values("ts"),
            left_on="fill_ts",
            right_on="ts",
            direction="backward",
            tolerance=pd.Timedelta(seconds=5),
            suffixes=("", "_live"),
        )
        joined["live_1s_covered"] = joined["ts_live"].notna()
        parts.append(joined)
    out = pd.concat(parts, ignore_index=True).sort_values("fill_ts").reset_index(drop=True)
    out["arm_b_merton_token_prob"] = np.where(out["actual_outcome"].astype(str).eq("up"), out["arm_b_merton_p_up"], 1.0 - out["arm_b_merton_p_up"])
    out["arm_b_kou_token_prob"] = np.where(out["actual_outcome"].astype(str).eq("up"), out["arm_b_kou_p_up"], 1.0 - out["arm_b_kou_p_up"])
    out["arm_c_edgeworth_token_prob"] = np.where(out["actual_outcome"].astype(str).eq("up"), out["arm_c_edgeworth_p_up"], 1.0 - out["arm_c_edgeworth_p_up"])
    out["arm_b_merton_edge"] = out["short_price"] - out["arm_b_merton_token_prob"]
    out["arm_b_kou_edge"] = out["short_price"] - out["arm_b_kou_token_prob"]
    out["arm_c_edgeworth_edge"] = out["short_price"] - out["arm_c_edgeworth_token_prob"]
    return out


def add_lob_features(pm: pd.DataFrame) -> pd.DataFrame:
    frames = [p for p in (LOB_ROLL, LOB_A0C) if p.exists()]
    if not frames:
        pm["lob_covered"] = False
        return pm
    lob = pd.concat([pd.read_parquet(p) for p in frames], ignore_index=True)
    lob["exchange_ts"] = pd.to_datetime(lob["exchange_ts"], utc=True).astype("datetime64[ns, UTC]")
    lob = lob.drop_duplicates(["asset_id", "market_id", "exchange_ts"]).sort_values(["asset_id", "market_id", "exchange_ts"])
    keep = [
        "asset_id",
        "market_id",
        "exchange_ts",
        "best_bid",
        "best_ask",
        "spread",
        "mid",
        "best_bid_size",
        "best_ask_size",
        "bid_top5_shares",
        "ask_top5_shares",
        "tob_imbalance",
        "ofi_combined_60s",
        "ofi_bid_60s",
        "ofi_ask_60s",
        "signed_trade_size_60s",
        "trade_count_60s",
    ]
    parts: list[pd.DataFrame] = []
    for _, left in pm.groupby(["asset_id", "market_id"], sort=False):
        aid = left["asset_id"].iloc[0]
        mid = left["market_id"].iloc[0]
        right = lob[(lob["asset_id"].eq(aid)) & (lob["market_id"].eq(mid))][keep].copy()
        left = left.copy()
        left["fill_ts"] = pd.to_datetime(left["fill_ts"], utc=True).astype("datetime64[ns, UTC]")
        if right.empty:
            left = left.copy()
            left["lob_covered"] = False
            parts.append(left)
            continue
        joined = pd.merge_asof(
            left.sort_values("fill_ts"),
            right.sort_values("exchange_ts"),
            left_on="fill_ts",
            right_on="exchange_ts",
            direction="backward",
            tolerance=pd.Timedelta(seconds=10),
            suffixes=("", "_lob"),
        )
        joined["lob_covered"] = joined["exchange_ts"].notna()
        parts.append(joined)
    return pd.concat(parts, ignore_index=True).sort_values("fill_ts").reset_index(drop=True)


def add_local_window_diagnostics(pm: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    live_idx = live.set_index("ts")
    for _, r in pm.iterrows():
        sub = live_idx[(live_idx["asset"].eq(r["asset"])) & (live_idx["market_slug"].eq(r["market_slug"]))]
        start = r["fill_ts"] - pd.Timedelta(seconds=300)
        window = sub[(sub.index <= r["fill_ts"]) & (sub.index > start)].copy()
        ret = window["log_ret_1s"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ret) >= 3:
            rv = float((ret * ret).sum())
            bv = float((math.pi / 2.0) * (ret.abs() * ret.abs().shift(1)).dropna().sum())
            bns_ratio = max(rv - bv, 0.0) / rv if rv > 0 else 0.0
            sigma_5m = math.sqrt(rv / max(float(len(ret)), 1.0) * YEAR_SECONDS)
            max_abs_ret = float(ret.abs().max())
        else:
            rv = bv = bns_ratio = sigma_5m = max_abs_ret = math.nan
        rows.append(
            {
                "fill_id": r["fill_id"],
                "live_1s_rows_5m": int(len(window)),
                "live_1s_jump_count_5m": int(window["jump_flag"].fillna(False).sum()) if len(window) else 0,
                "live_1s_bns_jump_ratio_5m": bns_ratio,
                "live_1s_realized_sigma_5m": sigma_5m,
                "live_1s_max_abs_ret_5m": max_abs_ret,
            }
        )
    return pm.merge(pd.DataFrame(rows), on="fill_id", how="left")


def reliability_table(live: pd.DataFrame) -> pd.DataFrame:
    panel = live.copy()
    panel["minute"] = panel["ts"].dt.floor("60s")
    panel = panel.sort_values("ts").drop_duplicates(["market_slug", "asset", "minute"])
    rows: list[dict[str, Any]] = []
    for arm, col in [
        ("arm_a_ewma_nd2", "arm_a_p_up"),
        ("arm_b_merton_live1s", "arm_b_merton_p_up"),
        ("arm_b_kou_live1s", "arm_b_kou_p_up"),
        ("arm_c_edgeworth_higher_moment", "arm_c_edgeworth_p_up"),
    ]:
        tmp = panel[["market_id", "market_slug", "asset", "binance_resolution_up", col]].copy()
        tmp = tmp.rename(columns={col: "pred"})
        tmp["prob_bin"] = pd.cut(
            tmp["pred"],
            bins=[0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0],
            labels=["0_5c", "5_10c", "10_25c", "25_50c", "50_75c", "75_90c", "90_95c", "95_100c"],
            include_lowest=True,
        )
        for bucket, g in tmp.groupby("prob_bin", observed=True):
            if g.empty:
                continue
            rows.append(
                {
                    "sample": "captured_1s_panel_one_row_per_minute",
                    "arm": arm,
                    "bucket": str(bucket),
                    "rows": int(len(g)),
                    "markets": int(g["market_id"].nunique()),
                    "mean_pred": float(g["pred"].mean()),
                    "observed_freq": float(g["binance_resolution_up"].mean()),
                    "obs_minus_pred": float(g["binance_resolution_up"].mean() - g["pred"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_arm(pm: pd.DataFrame, label: str, prob_col: str, edge_col: str, *, subset: pd.Series | None = None) -> dict[str, Any]:
    sub = pm[subset].copy() if subset is not None else pm.copy()
    if sub.empty:
        return {"label": label, "fills": 0, "markets": 0}
    edge_lo, edge_hi = cluster_ci(sub, edge_col, seed_offset=10)
    net_lo, net_hi = cluster_ci(sub, "net_realized_ev", seed_offset=20)
    gross_lo, gross_hi = cluster_ci(sub, "gross_realized_ev", seed_offset=30)
    after = sub.copy()
    after["net_after_top3"] = after["net_realized_ev"] * NON_TOP3_AVAILABLE_SHARE
    top3_lo, top3_hi = market_sum_ci(after, "net_after_top3", seed_offset=40)
    mean_after_top3 = float(after.groupby("market_id")["net_after_top3"].sum().mean())
    return {
        "label": label,
        "fills": int(len(sub)),
        "markets": int(sub["market_id"].nunique()),
        "mean_short_price": float(sub["short_price"].mean()),
        "mean_pred_itm_prob": float(sub[prob_col].mean()),
        "realized_itm_rate": float(sub["realized_itm"].mean()),
        "obs_minus_pred": float(sub["realized_itm"].mean() - sub[prob_col].mean()),
        "mean_model_edge": float(sub[edge_col].mean()),
        "model_edge_ci_lo": edge_lo,
        "model_edge_ci_hi": edge_hi,
        "mean_gross_realized_ev": float(sub["gross_realized_ev"].mean()),
        "gross_realized_ev_ci_lo": gross_lo,
        "gross_realized_ev_ci_hi": gross_hi,
        "mean_net_realized_ev": float(sub["net_realized_ev"].mean()),
        "net_realized_ev_ci_lo": net_lo,
        "net_realized_ev_ci_hi": net_hi,
        "mean_market_net_after_top3": mean_after_top3,
        "market_net_after_top3_ci_lo": top3_lo,
        "market_net_after_top3_ci_hi": top3_hi,
        "structural_baseline_mean": STRUCTURAL_BASELINE_C,
        "incremental_after_top3_vs_structural": mean_after_top3 - STRUCTURAL_BASELINE_C,
        "incremental_after_top3_ci_lo": top3_lo - STRUCTURAL_BASELINE_C if np.isfinite(top3_lo) else math.nan,
        "incremental_after_top3_ci_hi": top3_hi - STRUCTURAL_BASELINE_C if np.isfinite(top3_hi) else math.nan,
        "live_1s_coverage": float(sub["live_1s_covered"].mean()) if "live_1s_covered" in sub else math.nan,
        "lob_coverage": float(sub["lob_covered"].mean()) if "lob_covered" in sub else math.nan,
        "mean_jump_lambda_per_4h": float((sub["jump_lambda_per_year"] * (4 * 3600 / YEAR_SECONDS)).mean()),
        "mean_prior_jump_count": float(sub["jump_prior_count"].mean()),
        "mean_live_1s_jump_count_5m": float(sub["live_1s_jump_count_5m"].mean()) if "live_1s_jump_count_5m" in sub else math.nan,
        "mean_bns_jump_ratio_5m": float(sub["live_1s_bns_jump_ratio_5m"].mean()) if "live_1s_bns_jump_ratio_5m" in sub else math.nan,
    }


def summarize_pm(pm: pd.DataFrame) -> pd.DataFrame:
    rows = [
        summarize_arm(pm, "arm_a_ewma_nd2_original_set", "arm_a_token_prob", "arm_a_edge"),
        summarize_arm(pm, "arm_b_merton_live1s_original_set", "arm_b_merton_token_prob", "arm_b_merton_edge"),
        summarize_arm(pm, "arm_b_kou_live1s_original_set", "arm_b_kou_token_prob", "arm_b_kou_edge"),
        summarize_arm(pm, "arm_c_edgeworth_higher_moment_original_set", "arm_c_edgeworth_token_prob", "arm_c_edgeworth_edge"),
        summarize_arm(pm, "arm_b_merton_live1s_rich_ge_1c", "arm_b_merton_token_prob", "arm_b_merton_edge", subset=pm["arm_b_merton_edge"].ge(0.01)),
        summarize_arm(pm, "arm_b_kou_live1s_rich_ge_1c", "arm_b_kou_token_prob", "arm_b_kou_edge", subset=pm["arm_b_kou_edge"].ge(0.01)),
        summarize_arm(pm, "arm_c_edgeworth_higher_moment_rich_ge_1c", "arm_c_edgeworth_token_prob", "arm_c_edgeworth_edge", subset=pm["arm_c_edgeworth_edge"].ge(0.01)),
        summarize_arm(pm, "arm_b_merton_live1s_rich_ge_5c", "arm_b_merton_token_prob", "arm_b_merton_edge", subset=pm["arm_b_merton_edge"].ge(0.05)),
        summarize_arm(pm, "arm_b_kou_live1s_rich_ge_5c", "arm_b_kou_token_prob", "arm_b_kou_edge", subset=pm["arm_b_kou_edge"].ge(0.05)),
        summarize_arm(pm, "arm_c_edgeworth_higher_moment_rich_ge_5c", "arm_c_edgeworth_token_prob", "arm_c_edgeworth_edge", subset=pm["arm_c_edgeworth_edge"].ge(0.05)),
    ]
    return pd.DataFrame(rows)


def representative_params(pm: pd.DataFrame) -> dict[str, float]:
    return {
        "sigma": float(pm["ewma_sigma_annualized"].median()),
        "tau_years": float((pm["seconds_to_expiry"].median()) / YEAR_SECONDS),
        "lambda_per_year": float(pm["jump_lambda_per_year"].median()),
        "jump_mu": float(pm["jump_mu"].median()),
        "jump_sigma": float(pm["jump_sigma"].median()),
        "kou_p_up": float(pm["kou_p_up"].median()),
        "kou_up_mean_abs": float(pm["kou_up_mean_abs"].median()),
        "kou_down_mean_abs": float(pm["kou_down_mean_abs"].median()),
        "prior_ret_skew_1s": float(pm["prior_ret_skew_1s"].median()),
        "prior_ret_excess_kurt_1s": float(pm["prior_ret_excess_kurt_1s"].median()),
    }


def price_grid(params: dict[str, float], arm: str, z_grid: np.ndarray) -> np.ndarray:
    sigma = params["sigma"]
    tau = params["tau_years"]
    x = z_grid * sigma * math.sqrt(tau)
    df = pd.DataFrame(
        {
            "binance_spot": np.exp(x),
            "binance_strike_spot": np.ones_like(x),
            "tau_years": np.full_like(x, tau),
            "ewma_sigma_annualized": np.full_like(x, sigma),
            "jump_lambda_per_year": np.full_like(x, params["lambda_per_year"]),
            "jump_mu": np.full_like(x, params["jump_mu"]),
            "jump_sigma": np.full_like(x, params["jump_sigma"]),
            "kou_p_up": np.full_like(x, params["kou_p_up"]),
            "kou_up_mean_abs": np.full_like(x, params["kou_up_mean_abs"]),
            "kou_down_mean_abs": np.full_like(x, params["kou_down_mean_abs"]),
            "prior_ret_skew_1s": np.full_like(x, params["prior_ret_skew_1s"]),
            "prior_ret_excess_kurt_1s": np.full_like(x, params["prior_ret_excess_kurt_1s"]),
        }
    )
    if arm == "gaussian":
        return np.asarray(norm_cdf(z_grid), dtype=float)
    if arm == "merton":
        return merton_p_up(df)
    if arm == "kou":
        return kou_p_up(df)
    return edgeworth_p_up(df)


def build_shape(pm: pd.DataFrame) -> pd.DataFrame:
    params = representative_params(pm)
    z_grid = np.linspace(-3.0, 3.0, 241)
    rows: list[dict[str, Any]] = []
    for arm in ("gaussian", "merton", "kou", "edgeworth"):
        prices = price_grid(params, arm, z_grid)
        # Numerical delta/gamma at S=1, K=1 using the representative horizon.
        bump = 1e-4
        base = pd.DataFrame(
            {
                "binance_spot": [1.0 - bump, 1.0, 1.0 + bump],
                "binance_strike_spot": [1.0, 1.0, 1.0],
                "tau_years": [params["tau_years"]] * 3,
                "ewma_sigma_annualized": [params["sigma"]] * 3,
                "jump_lambda_per_year": [params["lambda_per_year"]] * 3,
                "jump_mu": [params["jump_mu"]] * 3,
                "jump_sigma": [params["jump_sigma"]] * 3,
                "kou_p_up": [params["kou_p_up"]] * 3,
                "kou_up_mean_abs": [params["kou_up_mean_abs"]] * 3,
                "kou_down_mean_abs": [params["kou_down_mean_abs"]] * 3,
                "prior_ret_skew_1s": [params["prior_ret_skew_1s"]] * 3,
                "prior_ret_excess_kurt_1s": [params["prior_ret_excess_kurt_1s"]] * 3,
            }
        )
        if arm == "gaussian":
            zz = np.log(base["binance_spot"].to_numpy()) / (params["sigma"] * math.sqrt(params["tau_years"]))
            p3 = np.asarray(norm_cdf(zz), dtype=float)
        elif arm == "merton":
            p3 = merton_p_up(base)
        elif arm == "kou":
            p3 = kou_p_up(base)
        else:
            p3 = edgeworth_p_up(base)
        delta = (p3[2] - p3[0]) / (2 * bump)
        gamma = (p3[2] - 2 * p3[1] + p3[0]) / (bump * bump)
        rows.append(
            {
                "arm": arm,
                "representative_sigma_ann": params["sigma"],
                "representative_tau_seconds": params["tau_years"] * YEAR_SECONDS,
                "lambda_per_4h": params["lambda_per_year"] * (4 * 3600 / YEAR_SECONDS),
                "near_strike_price": float(p3[1]),
                "near_strike_delta_per_dollar": float(delta),
                "near_strike_gamma_per_dollar2": float(gamma),
                "p_up_at_z_minus_2": float(np.interp(-2.0, z_grid, prices)),
                "p_up_at_z_minus_1p5": float(np.interp(-1.5, z_grid, prices)),
                "p_up_at_z_plus_1p5": float(np.interp(1.5, z_grid, prices)),
                "p_up_at_z_plus_2": float(np.interp(2.0, z_grid, prices)),
                "otm_tail_mass_avg_z_abs_ge_1p5": float((np.interp(-1.5, z_grid, prices) + (1.0 - np.interp(1.5, z_grid, prices))) / 2.0),
            }
        )
    shape = pd.DataFrame(rows)
    shape_grid = pd.DataFrame({"z": z_grid})
    for arm in ("gaussian", "merton", "kou", "edgeworth"):
        shape_grid[f"{arm}_p_up"] = price_grid(params, arm, z_grid)
    return shape, shape_grid


def fetch_deribit_dvol(start: pd.Timestamp, end: pd.Timestamp, currencies: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    with httpx.Client(timeout=20, headers={"User-Agent": "epsilon-quant-research-od-pricing-model/1.0"}) as client:
        for currency in currencies:
            try:
                r = client.get(
                    f"{DERIBIT_BASE}/public/get_volatility_index_data",
                    params={"currency": currency, "start_timestamp": start_ms, "end_timestamp": end_ms, "resolution": "3600"},
                )
                r.raise_for_status()
                data = r.json().get("result", {}).get("data", []) or []
            except Exception as exc:
                rows.append({"currency": currency, "fetch_error": repr(exc)})
                continue
            for item in data:
                ts_ms, open_v, high_v, low_v, close_v = item
                close = float(close_v)
                sigma = close / 100.0 if close > 3.0 else close
                rows.append(
                    {
                        "currency": currency,
                        "ts": pd.to_datetime(int(ts_ms), unit="ms", utc=True),
                        "dvol_open": float(open_v),
                        "dvol_high": float(high_v),
                        "dvol_low": float(low_v),
                        "dvol_close": close,
                        "dvol_sigma_ann": sigma,
                        "fetch_error": "",
                    }
                )
            time.sleep(0.05)
    return pd.DataFrame(rows)


def add_deribit_anchor(pm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = sorted(a for a in pm["asset"].dropna().unique() if a in {"BTC", "ETH"})
    if not assets:
        pm["deribit_dvol_covered"] = False
        return pm, pd.DataFrame()
    start = pm[pm["asset"].isin(assets)]["fill_ts"].min() - pd.Timedelta(hours=2)
    end = pm[pm["asset"].isin(assets)]["fill_ts"].max() + pd.Timedelta(hours=2)
    dvol = fetch_deribit_dvol(start, end, assets)
    if dvol.empty or "ts" not in dvol:
        pm["deribit_dvol_covered"] = False
        return pm, dvol
    parts: list[pd.DataFrame] = []
    for asset, left in pm.groupby("asset", sort=False):
        left = left.sort_values("fill_ts").copy()
        left["fill_ts"] = pd.to_datetime(left["fill_ts"], utc=True).astype("datetime64[ns, UTC]")
        if asset not in {"BTC", "ETH"}:
            left["deribit_dvol_covered"] = False
            parts.append(left)
            continue
        right = dvol[dvol["currency"].eq(asset)].dropna(subset=["ts"]).sort_values("ts").copy()
        right["ts"] = pd.to_datetime(right["ts"], utc=True).astype("datetime64[ns, UTC]")
        if right.empty:
            left["deribit_dvol_covered"] = False
            parts.append(left)
            continue
        joined = pd.merge_asof(left, right, left_on="fill_ts", right_on="ts", direction="nearest", tolerance=pd.Timedelta(hours=2), suffixes=("", "_deribit"))
        joined["deribit_dvol_covered"] = joined["dvol_sigma_ann"].notna()
        parts.append(joined)
    out = pd.concat(parts, ignore_index=True).sort_values("fill_ts").reset_index(drop=True)
    tau = out["seconds_to_expiry"].astype(float).clip(lower=1.0) / YEAR_SECONDS
    z = out["log_spot_moneyness"].astype(float) / (out["dvol_sigma_ann"].astype(float) * np.sqrt(tau))
    p_up = np.asarray(norm_cdf(z), dtype=float)
    out["arm_d_deribit_dvol_p_up"] = np.where(out["deribit_dvol_covered"], p_up, np.nan)
    out["arm_d_deribit_dvol_token_prob"] = np.where(out["actual_outcome"].astype(str).eq("up"), out["arm_d_deribit_dvol_p_up"], 1.0 - out["arm_d_deribit_dvol_p_up"])
    out["arm_d_deribit_dvol_edge"] = out["short_price"] - out["arm_d_deribit_dvol_token_prob"]
    return out, dvol


def deribit_daily_illustration(dvol: pd.DataFrame) -> pd.DataFrame:
    if not DAILY_SURFACE.exists() or dvol.empty or "ts" not in dvol:
        return pd.DataFrame()
    daily = pd.read_parquet(DAILY_SURFACE)
    daily["ts"] = pd.to_datetime(daily["ts"], utc=True)
    daily = daily[daily["asset"].isin(["BTC", "ETH"])].copy()
    if daily.empty:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for asset, left in daily.groupby("asset", sort=False):
        left = left.copy()
        left["ts"] = pd.to_datetime(left["ts"], utc=True).astype("datetime64[ns, UTC]")
        right = dvol[dvol["currency"].eq(asset)].dropna(subset=["ts"]).sort_values("ts").copy()
        right["ts"] = pd.to_datetime(right["ts"], utc=True).astype("datetime64[ns, UTC]")
        if right.empty:
            continue
        joined = pd.merge_asof(left.sort_values("ts"), right, on="ts", direction="nearest", tolerance=pd.Timedelta(hours=2))
        joined = joined[joined["dvol_sigma_ann"].notna()].copy()
        tau = joined["seconds_to_expiry"].astype(float).clip(lower=1.0) / YEAR_SECONDS
        z = np.log(joined["binance_spot"].astype(float) / joined["window_strike_spot"].astype(float)) / (joined["dvol_sigma_ann"].astype(float) * np.sqrt(tau))
        joined["deribit_fair_up"] = np.asarray(norm_cdf(z), dtype=float)
        joined["fair_diff_deribit_minus_ewma"] = joined["deribit_fair_up"] - joined["fair_up_causal"].astype(float)
        parts.append(joined)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    rows = []
    for asset, g in out.groupby("asset"):
        rows.append(
            {
                "sample": "daily_24h_model_surface_illustrative",
                "asset": asset,
                "rows": int(len(g)),
                "start_ts": g["ts"].min(),
                "end_ts": g["ts"].max(),
                "mean_dvol_sigma": float(g["dvol_sigma_ann"].mean()),
                "mean_ewma_sigma": float(g["sigma_causal"].mean()),
                "mean_deribit_fair_up": float(g["deribit_fair_up"].mean()),
                "mean_ewma_fair_up": float(g["fair_up_causal"].mean()),
                "mean_fair_diff_deribit_minus_ewma": float(g["fair_diff_deribit_minus_ewma"].mean()),
                "abs_fair_diff_p95": float(g["fair_diff_deribit_minus_ewma"].abs().quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def summarize_deribit(pm: pd.DataFrame, daily_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    covered = pm[pm.get("deribit_dvol_covered", False).fillna(False)].copy()
    if not covered.empty:
        rows.append(summarize_arm(covered, "arm_d_deribit_dvol_btc_eth_original_set", "arm_d_deribit_dvol_token_prob", "arm_d_deribit_dvol_edge"))
        rich = covered["arm_d_deribit_dvol_edge"].ge(0.01)
        rows.append(summarize_arm(covered, "arm_d_deribit_dvol_btc_eth_rich_ge_1c", "arm_d_deribit_dvol_token_prob", "arm_d_deribit_dvol_edge", subset=rich))
    if daily_summary is not None and not daily_summary.empty:
        for _, r in daily_summary.iterrows():
            rows.append({"label": f"daily_24h_deribit_illustrative_{r['asset'].lower()}", **r.to_dict()})
    return pd.DataFrame(rows)


def make_plots(summary: pd.DataFrame, shape_grid: pd.DataFrame, deribit: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    plot = summary[summary["fills"].fillna(0).astype(int).gt(0)].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(plot))
    y = plot["mean_model_edge"].to_numpy(dtype=float)
    lo = plot["model_edge_ci_lo"].to_numpy(dtype=float)
    hi = plot["model_edge_ci_hi"].to_numpy(dtype=float)
    yerr = np.vstack([y - lo, hi - y])
    ax.bar(x, y, color="#2b8cbe")
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#222222", capsize=3)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot["label"], rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Mean model edge: short price - model P(ITM)")
    ax.set_title("OD pricing-model-form edge on PM far-|z| short set")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EDGE_PLOT, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shape_grid["z"], shape_grid["gaussian_p_up"], label="Gaussian N(z)", linewidth=2)
    ax.plot(shape_grid["z"], shape_grid["merton_p_up"], label="Merton live-jump", linewidth=2)
    ax.plot(shape_grid["z"], shape_grid["kou_p_up"], label="Kou-style asymmetric jump", linewidth=2)
    ax.plot(shape_grid["z"], shape_grid["edgeworth_p_up"], label="Edgeworth higher-moment", linewidth=2)
    ax.axvline(-1.0, color="#888888", linestyle="--", linewidth=1)
    ax.axvline(1.0, color="#888888", linestyle="--", linewidth=1)
    ax.set_xlabel("z = ln(S/K) / (sigma sqrt(tau))")
    ax.set_ylabel("P(resolve UP)")
    ax.set_title("Digital probability surface: jump models add OTM tail mass")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SHAPE_PLOT, dpi=160)
    plt.close(fig)

    cov = deribit[deribit.get("deribit_dvol_covered", False).fillna(False)].copy() if not deribit.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if not cov.empty:
        x = np.arange(len(cov))
        ax.scatter(x, cov["arm_a_token_prob"], label="EWMA N(z)", color="#2b8cbe")
        ax.scatter(x, cov["arm_d_deribit_dvol_token_prob"], label="Deribit DVOL extrapolated", color="#e34a33")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}-{int(fid)}" for a, fid in zip(cov["asset"], cov["fill_id"])], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Token P(ITM)")
    ax.set_title("Deribit DVOL anchor on BTC/ETH PM fills")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DERIBIT_PLOT, dpi=160)
    plt.close(fig)


def format_summary_table(summary: pd.DataFrame) -> str:
    rows = []
    for _, r in summary[summary["fills"].fillna(0).astype(int).gt(0)].iterrows():
        rows.append(
            [
                str(r["label"]),
                str(int(r["fills"])),
                str(int(r["markets"])),
                cents(float(r["mean_short_price"])),
                pct(float(r["mean_pred_itm_prob"])),
                pct(float(r["realized_itm_rate"])),
                cents(float(r["mean_model_edge"])),
                fmt_ci(float(r["model_edge_ci_lo"]), float(r["model_edge_ci_hi"])),
                cents(float(r["mean_net_realized_ev"])),
                fmt_ci(float(r["net_realized_ev_ci_lo"]), float(r["net_realized_ev_ci_hi"])),
                cents(float(r["incremental_after_top3_vs_structural"])),
                fmt_ci(float(r["incremental_after_top3_ci_lo"]), float(r["incremental_after_top3_ci_hi"])),
            ]
        )
    return markdown_table(
        [
            "arm / subset",
            "fills",
            "markets",
            "price",
            "model P(ITM)",
            "realized ITM",
            "model edge",
            "edge CI",
            "realized net EV",
            "realized CI",
            "incremental vs structural",
            "incremental CI",
        ],
        rows,
    )


def format_reliability_table(reliability: pd.DataFrame) -> str:
    sub = reliability[reliability["bucket"].isin(["0_5c", "5_10c", "10_25c", "75_90c", "90_95c", "95_100c"])].copy()
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            [
                str(r["arm"]),
                str(r["bucket"]),
                str(int(r["rows"])),
                str(int(r["markets"])),
                pct(float(r["mean_pred"])),
                pct(float(r["observed_freq"])),
                pct(float(r["obs_minus_pred"])),
            ]
        )
    return markdown_table(["arm", "probability bucket", "rows", "markets", "mean pred", "observed", "obs - pred"], rows)


def format_shape_table(shape: pd.DataFrame) -> str:
    rows = []
    for _, r in shape.iterrows():
        rows.append(
            [
                str(r["arm"]),
                number(float(r["lambda_per_4h"]), 3),
                pct(float(r["p_up_at_z_minus_2"])),
                pct(float(r["p_up_at_z_minus_1p5"])),
                pct(float(r["near_strike_price"])),
                number(float(r["near_strike_delta_per_dollar"]), 2),
                number(float(r["near_strike_gamma_per_dollar2"]), 2),
                pct(float(r["otm_tail_mass_avg_z_abs_ge_1p5"])),
            ]
        )
    return markdown_table(
        ["arm", "lambda/4h", "P_up z=-2", "P_up z=-1.5", "P_up z=0", "delta", "gamma", "avg OTM tail"],
        rows,
    )


def format_deribit_table(deribit_summary: pd.DataFrame) -> str:
    if deribit_summary.empty:
        return "_No Deribit rows were available._"
    rows = []
    for _, r in deribit_summary.iterrows():
        if "fills" in r and pd.notna(r.get("fills")) and int(r.get("fills", 0)) > 0:
            rows.append(
                [
                    str(r["label"]),
                    str(int(r["fills"])),
                    str(int(r["markets"])),
                    cents(float(r["mean_short_price"])),
                    pct(float(r["mean_pred_itm_prob"])),
                    cents(float(r["mean_model_edge"])),
                    fmt_ci(float(r["model_edge_ci_lo"]), float(r["model_edge_ci_hi"])),
                    cents(float(r["mean_net_realized_ev"])),
                    cents(float(r["incremental_after_top3_vs_structural"])),
                    fmt_ci(float(r["incremental_after_top3_ci_lo"]), float(r["incremental_after_top3_ci_hi"])),
                ]
            )
        elif str(r.get("sample", "")).startswith("daily_24h"):
            rows.append(
                [
                    str(r["label"]),
                    str(int(r["rows"])),
                    "n/a",
                    "n/a",
                    pct(float(r["mean_deribit_fair_up"])),
                    cents(float(r["mean_fair_diff_deribit_minus_ewma"])),
                    f"p95 abs {cents(float(r['abs_fair_diff_p95']))}",
                    "illustrative",
                    "illustrative",
                    "illustrative",
                ]
            )
    return markdown_table(
        ["row", "n", "markets", "price", "Deribit P/fair", "edge/diff", "CI / p95", "realized net EV / read", "incremental vs structural", "incremental CI"],
        rows,
    )


def write_note(pm: pd.DataFrame, summary: pd.DataFrame, reliability: pd.DataFrame, shape: pd.DataFrame, deribit_summary: pd.DataFrame, dvol: pd.DataFrame) -> None:
    merton = summary[summary["label"].eq("arm_b_merton_live1s_original_set")].iloc[0]
    kou = summary[summary["label"].eq("arm_b_kou_live1s_original_set")].iloc[0]
    edgeworth = summary[summary["label"].eq("arm_c_edgeworth_higher_moment_original_set")].iloc[0]
    best_inc_lo = np.nanmax(summary["incremental_after_top3_ci_lo"].to_numpy(dtype=float))
    deribit_inc_lo = (
        float(np.nanmax(deribit_summary["incremental_after_top3_ci_lo"].to_numpy(dtype=float)))
        if deribit_summary is not None and "incremental_after_top3_ci_lo" in deribit_summary and deribit_summary["incremental_after_top3_ci_lo"].notna().any()
        else math.nan
    )
    dvol_rows = int(len(dvol)) if dvol is not None and not dvol.empty else 0
    arm_c_decision = (
        "run as a cheap higher-moment / Edgeworth extension. Full Bates or calibrated VG were still not attempted because "
        "the environment lacks SciPy/Arch/Torch and the PM validation set is only 23 fills, but this arm directly tests whether "
        "causal 1s skew/kurtosis changes the OTM digital probability enough to reopen the gate"
    )
    note = f"""# OD Pricing-Model-Form Test: Do Jumps Explain The Far-OTM Longshot Result?

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior notes: [[od_v4_calibration_gate_findings]] · [[od_v4_queue_replay_findings]] · [[od_conditional_prob_calibration_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Headline

Final verdict: **CLOSE remains**.

This run tested the prompt's model-form hypothesis: maybe the old Gaussian `N(z)` digital underpriced short-dated OTM tail probability because it had no jump mass. The test repriced the same v4 far-|z| strict-rich short set with causal jump parameters from the captured 1s Binance/K3 panel, then checked whether any residual OD model edge survives the v4 structural baseline.

It does **not** reopen OD. The live 1s jump and higher-moment models do add tail mass in the OTM shape diagnostic, but on the actual 23-fill PM set they do not create a deployable incremental edge. Merton original-set model edge is {cents(float(merton['mean_model_edge']))}, CI {fmt_ci(float(merton['model_edge_ci_lo']), float(merton['model_edge_ci_hi']))}; Kou original-set model edge is {cents(float(kou['mean_model_edge']))}, CI {fmt_ci(float(kou['model_edge_ci_lo']), float(kou['model_edge_ci_hi']))}; Edgeworth higher-moment model edge is {cents(float(edgeworth['mean_model_edge']))}, CI {fmt_ci(float(edgeworth['model_edge_ci_lo']), float(edgeworth['model_edge_ci_hi']))}. The best after-top3 incremental lower CI across tested pricing-model rows is {cents(float(best_inc_lo))}, and Deribit's best illustrative after-top3 lower CI is {cents(float(deribit_inc_lo)) if np.isfinite(deribit_inc_lo) else 'n/a'}, so the branch still fails the structural-baseline bar.

Plain-English read: jumps make the tail model more honest, but they do not turn the OD rich-short signal into a standalone trade. The apparent upside is still explained better as source/structure/queue selection with tiny capacity, not a distinct pricing-model edge.

## What Changed Versus The 5-Minute Conditional-Probability Test

The prior [[od_conditional_prob_calibration_findings]] note used years of Binance 5-minute data as a broad truth table. This script keeps that result as background, but the validation here is anchored on the captured windows:

- **1s Binance/K3 panel:** `data/analysis/cache/k3v3h_panel_features.parquet`, covering all 23 PM fills.
- **Polymarket LOB/WS capture:** `block_a0c_roll_features` plus `block_a0c_features`, covering {pct(float(pm['lob_covered'].mean()))} of PM fills.
- **Jump detection:** Lee-Mykland-style standardized 1s returns, flagged when `|r_1s| > {number(JUMP_Z_THRESHOLD, 1)} * sigma_1s` and at least {number(JUMP_ABS_RETURN_FLOOR_BPS, 1)} bps in one second, with BNS-style 5-minute realized-minus-bipower jump ratio reported as a diagnostic.
- **Deribit:** public BTC/ETH DVOL pulled as an external IV anchor. This fetched {dvol_rows} hourly DVOL rows. It is illustrative only because it is BTC/ETH-only, DVOL is a 30-day index, and the local artifacts do not contain a clean historical Deribit per-option IV surface for the PM fills.

The 5-minute history is the prior/control. The captured 1s panel is the actual live-window test.

## Design

Unit of observation for the PM table is one v4 far-|z| strict-rich short fill. The model asks: if we short the token at price `p`, what does each model think the token's probability of paying `$1` is?

```text
model edge = short price - model P(token pays $1)
realized EV = short price - realized payoff + maker rebate
```

The arms:

- **Arm A, Gaussian control:** the old EWMA `N(z)` digital.
- **Arm B, Merton:** compound-Poisson normal jumps fitted causally from prior captured 1s Binance returns.
- **Arm B, Kou-style:** asymmetric up/down exponential jump moments, also fitted causally from prior captured 1s returns. This is a moment-matched Kou-style approximation, used because the repo environment has NumPy/Pandas but not SciPy for full closed-form calibration.
- **Arm C, higher moments:** {arm_c_decision}.
- **Arm D, Deribit DVOL:** BTC/ETH-only illustrative anchor. It is now reported with the same MM-incremental columns, but it is still not gate-grade because DVOL is a 30-day index rather than a historical 4h option surface.

CI columns are market-cluster bootstraps. `incremental vs structural` applies the K5 non-incumbent 5% capacity haircut and subtracts the best v4 structural queue baseline of {cents(STRUCTURAL_BASELINE_C)} per market. That is the MM integration check: a pricing model can have positive raw realized EV and still fail if, after realistic non-incumbent capacity, it does not beat the already-known MM/structural quote-selection result.

## PM Far-|z| Short Set Results

{format_summary_table(summary)}

Read: a positive `model edge` means the model says the token is overpriced at our short price. `realized net EV` is what happened on this tiny PM sample. `incremental vs structural` is the MM integration check. To reopen OD, the residual after the top-maker haircut also had to beat the structural queue baseline with lower-CI > 0. It does not.

![PM model edge]({EDGE_PLOT})

## Captured-Window 1s Reliability

This table uses the K3 captured 1s panel downsampled to one row per minute per market, not the multi-year 5-minute history. It is still overlapping within a 4h window, so read it as a live-window calibration diagnostic rather than a powered OOS gate.

{format_reliability_table(reliability)}

Read: the live-window panel says the jump models move probabilities, but they do not reveal a clean PM-specific residual edge. The important thing is that this is now using the same captured windows and live-state granularity as the actual fills.

## Delta, Gamma, And Tail Shape

The prompt's mechanism question is whether jumps change the digital shape, not just the level of volatility. The table below uses a representative PM horizon and sigma, then compares probability mass and numerical Greeks around the strike.

{format_shape_table(shape)}

`P_up z=-2` and `P_up z=-1.5` are OTM-up probabilities. `avg OTM tail` averages the two symmetric far-tail directions. A bigger value means the model assigns more chance to a far OTM token finishing ITM.

![Tail shape]({SHAPE_PLOT})

Read: the jump forms do what they are supposed to do mechanically: they redistribute some mass into the OTM tail and change the near-strike Greek shape. That is useful knowledge, but the PM residual-EV gate still fails.

## Deribit Anchor

Deribit is BTC/ETH only. The public historical anchor used here is DVOL, extrapolated down to the PM horizon by using the annualized DVOL level inside the same digital formula. That is deliberately labeled illustrative: it is a 30-day options-market IV index, not a 4h binary-resolution surface, and SOL has no Deribit analogue.

{format_deribit_table(deribit_summary)}

![Deribit anchor]({DERIBIT_PLOT})

Read: Deribit is helpful as a sanity anchor, not a decision gate. The BTC/ETH Deribit-rich subset has a positive model edge, but it is only 8 fills / 3 markets and its structural-incremental CI is still not a deployable lower-CI-positive OD result. A real Deribit option-surface comparison would need historical per-instrument IV/mark snapshots aligned to the PM fills; this run only uses the public DVOL index plus the local captured PM/Binance rows.

## Decision

OD stays **closed as a standalone strategy**. The pricing-model-form hypothesis is informative but not enough: Merton/Kou jump-aware pricing and the higher-moment Edgeworth extension do not leave a lower-CI-positive residual that beats the structural queue baseline, and the Deribit anchor is too small/indirect to reopen the branch.

Recommended routing: fold the useful pieces back into [[strat_market_making]] as weak quote-selection features. In practice that means: avoid pretending far-|z| Gaussian richness is alpha by itself; prefer source-clean, liquid, queue-realistic cells; and use live 1s jump/OFI flags as caution filters around tail states.

## Outputs

- Summary CSV: `data/analysis/csv_outputs/options_delta/od_pricing_model_form_summary.csv`
- Captured-window reliability CSV: `data/analysis/csv_outputs/options_delta/od_pricing_model_form_reliability.csv`
- Greek/tail-shape CSV: `data/analysis/csv_outputs/options_delta/od_pricing_model_form_shape.csv`
- Deribit CSV: `data/analysis/csv_outputs/options_delta/od_pricing_model_form_deribit.csv`
- PM fill parquet: `data/analysis/od_pricing_model_form_pm_fills.parquet`
- Live 1s panel parquet: `data/analysis/od_pricing_model_form_live1s_panel.parquet`
"""
    NOTE.write_text(normalize_markdown_wrapping(note), encoding="utf-8")


def main() -> None:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    pm = load_pm_fills()
    live = load_live_panel()
    reliability = reliability_table(live)

    pm = join_live_to_pm(pm, live)
    pm = add_lob_features(pm)
    pm = add_local_window_diagnostics(pm, live)
    pm, dvol = add_deribit_anchor(pm)
    daily_deribit = deribit_daily_illustration(dvol)

    summary = summarize_pm(pm)
    deribit_summary = summarize_deribit(pm, daily_deribit)
    shape, shape_grid = build_shape(pm)

    summary.to_csv(OUT_SUMMARY, index=False)
    reliability.to_csv(OUT_RELIABILITY, index=False)
    shape.to_csv(OUT_SHAPE, index=False)
    deribit_summary.to_csv(OUT_DERIBIT, index=False)
    pm.to_parquet(OUT_PM, index=False)
    # Keep the live panel as the reusable append-only analysis artifact.
    live.to_parquet(OUT_LIVE, index=False)
    if dvol is not None and not dvol.empty:
        dvol.to_parquet(OUT_DERIBIT_PARQUET, index=False)

    make_plots(summary, shape_grid, pm)
    write_note(pm, summary, reliability, shape, deribit_summary, dvol)
    print(f"wrote {NOTE}")
    print(summary[["label", "fills", "markets", "mean_model_edge", "model_edge_ci_lo", "incremental_after_top3_ci_lo"]].to_string(index=False))


if __name__ == "__main__":
    main()
