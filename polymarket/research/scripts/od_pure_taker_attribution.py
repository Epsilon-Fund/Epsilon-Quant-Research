"""Pure-taker OD attribution on captured crypto-4h LOB.

This removes the K-PEG/passive maker lifecycle from Strategy A and asks whether
the v3 OD digital fair value alone beats the executable best ask plus taker fee.
It uses the already-captured K6/K3 one-second panel; no new capture or web fetch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
PANEL_PATH = ANALYSIS / "k6_vol_gap_panel.parquet"
OUT_DIR = ANALYSIS / "csv_outputs" / "options_delta"
OUT_SUMMARY = OUT_DIR / "od_pure_taker_attribution_summary.csv"
OUT_SPREAD = OUT_DIR / "od_pure_taker_attribution_spread_regime.csv"
OUT_TRADES = ANALYSIS / "od_pure_taker_attribution_trades.parquet"

YEAR_SECONDS = 365.0 * 24.0 * 3600.0
BOOTSTRAP_SAMPLES = 2000
RNG_SEED = 20260603


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    description: str
    min_edge: float
    far_only: bool = False
    late_only: bool = False
    longshot_price_max: float | None = None


CELL_SPECS = [
    CellSpec("strict_all_edge_gt_0", "Strict source, any z/tau, edge > 0", 0.0),
    CellSpec("strict_all_edge_ge_1c", "Strict source, any z/tau, edge >= 1c", 0.01),
    CellSpec("strict_all_edge_ge_2c", "Strict source, any z/tau, edge >= 2c", 0.02),
    CellSpec("strict_far_edge_gt_0", "Strict source, far |z| >= 1, edge > 0", 0.0, far_only=True),
    CellSpec("strict_far_edge_ge_1c", "Strict source, far |z| >= 1, edge >= 1c", 0.01, far_only=True),
    CellSpec("strict_far_edge_ge_2c", "Strict source, far |z| >= 1, edge >= 2c", 0.02, far_only=True),
    CellSpec(
        "strict_far_late_edge_gt_0",
        "Strict source, far |z| >= 1, late <30m, edge > 0",
        0.0,
        far_only=True,
        late_only=True,
    ),
    CellSpec(
        "strict_far_late_edge_ge_1c",
        "Strict source, far |z| >= 1, late <30m, edge >= 1c",
        0.01,
        far_only=True,
        late_only=True,
    ),
    CellSpec(
        "strict_longshot_ask_le_30c_edge_gt_0",
        "Strict source, selected ask <= 30c, edge > 0",
        0.0,
        longshot_price_max=0.30,
    ),
    CellSpec(
        "strict_longshot_ask_le_30c_edge_ge_1c",
        "Strict source, selected ask <= 30c, edge >= 1c",
        0.01,
        longshot_price_max=0.30,
    ),
]


def norm_cdf(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))


def taker_fee(price: pd.Series, fee_rate: pd.Series) -> pd.Series:
    p = price.astype(float).clip(0.0, 1.0)
    r = fee_rate.astype(float).fillna(0.07)
    return r * p * (1.0 - p)


def recompute_v3_p_model(df: pd.DataFrame) -> pd.Series:
    spot = df["binance_spot"].astype(float)
    strike = df["binance_strike_spot"].astype(float)
    sigma = df["ewma_sigma_annualized"].astype(float).clip(lower=1e-6)
    tau = df["seconds_to_expiry"].astype(float).clip(lower=1.0) / YEAR_SECONDS
    z = np.log(spot / strike) / (sigma * np.sqrt(tau))
    out = pd.Series(norm_cdf(z), index=df.index).clip(0.0, 1.0)
    expired = df["seconds_to_expiry"].astype(float) <= 0
    if expired.any():
        out.loc[expired] = (spot.loc[expired] >= strike.loc[expired]).astype(float)
    return out


def bootstrap_ci(values: np.ndarray, seed_offset: int = 0) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(RNG_SEED + seed_offset + len(vals))
    draws = rng.integers(0, len(vals), size=(BOOTSTRAP_SAMPLES, len(vals)))
    means = vals[draws].mean(axis=1)
    lo, hi = np.nanquantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def base_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"missing {PANEL_PATH}")
    usecols = [
        "ts",
        "source_run",
        "up_bid",
        "up_ask",
        "down_bid",
        "down_ask",
        "polymarket_mid",
        "binance_spot",
        "ewma_sigma_annualized",
        "seconds_to_expiry",
        "window_start",
        "window_end",
        "binance_strike_spot",
        "binance_close_spot",
        "binance_resolution_up",
        "chainlink_resolution_up",
        "chainlink_binance_resolution_disagree",
        "taker_fee_rate",
        "market_slug",
        "market_id",
        "asset",
        "p_model",
        "abs_z",
        "moneyness_bucket",
        "time_bucket",
        "state_bucket",
        "source_ok_strict",
        "source_penalty_flag",
        "toxic_near_expiry",
    ]
    df = pd.read_parquet(PANEL_PATH, columns=usecols)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for col in [
        "up_bid",
        "up_ask",
        "down_bid",
        "down_ask",
        "p_model",
        "abs_z",
        "binance_spot",
        "binance_strike_spot",
        "ewma_sigma_annualized",
        "seconds_to_expiry",
        "taker_fee_rate",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    finite_quotes = (
        df["up_bid"].between(0.0, 1.0)
        & df["up_ask"].between(0.0, 1.0)
        & df["down_bid"].between(0.0, 1.0)
        & df["down_ask"].between(0.0, 1.0)
        & (df["up_bid"] <= df["up_ask"])
        & (df["down_bid"] <= df["down_ask"])
    )
    clean = (
        finite_quotes
        & df["source_ok_strict"].fillna(False).astype(bool)
        & ~df["toxic_near_expiry"].fillna(False).astype(bool)
        & df["p_model"].notna()
        & df["binance_spot"].notna()
        & df["binance_strike_spot"].notna()
        & df["ewma_sigma_annualized"].notna()
        & df["seconds_to_expiry"].gt(0)
    )
    df = df.loc[clean].copy()
    df["p_model_stored"] = df["p_model"].astype(float).clip(0.0, 1.0)
    df["p_model_contemp"] = recompute_v3_p_model(df)
    return df.sort_values(["market_slug", "ts"]).reset_index(drop=True)


def add_edges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["up_fee"] = taker_fee(out["up_ask"], out["taker_fee_rate"])
    out["down_fee"] = taker_fee(out["down_ask"], out["taker_fee_rate"])

    for label, p_col in [("stored", "p_model_stored"), ("contemp", "p_model_contemp")]:
        p_up = out[p_col].astype(float).clip(0.0, 1.0)
        p_down = 1.0 - p_up
        out[f"buy_up_edge_{label}"] = p_up - out["up_ask"] - out["up_fee"]
        out[f"buy_down_edge_{label}"] = p_down - out["down_ask"] - out["down_fee"]
        up_better = out[f"buy_up_edge_{label}"] >= out[f"buy_down_edge_{label}"]
        out[f"best_route_{label}"] = np.where(up_better, "buy_up", "buy_down")
        out[f"best_edge_{label}"] = np.where(
            up_better, out[f"buy_up_edge_{label}"], out[f"buy_down_edge_{label}"]
        )

    # Evaluate the stored-model selected route on the contemporaneous model.
    stored_up = out["best_route_stored"].eq("buy_up")
    out["stored_route_edge_contemp"] = np.where(
        stored_up, out["buy_up_edge_contemp"], out["buy_down_edge_contemp"]
    )
    out["selected_ask"] = np.where(stored_up, out["up_ask"], out["down_ask"])
    out["selected_bid"] = np.where(stored_up, out["up_bid"], out["down_bid"])
    out["selected_fee"] = np.where(stored_up, out["up_fee"], out["down_fee"])
    out["selected_mid"] = 0.5 * (out["selected_bid"] + out["selected_ask"])
    out["selected_spread"] = out["selected_ask"] - out["selected_bid"]
    out["selected_rel_spread"] = out["selected_spread"] / out["selected_ask"].replace(0.0, np.nan)
    out["selected_cross_cost"] = out["selected_ask"] - out["selected_mid"] + out["selected_fee"]

    fair_selected = np.where(stored_up, out["p_model_contemp"], 1.0 - out["p_model_contemp"])
    out["selected_mid_edge_contemp"] = fair_selected - out["selected_mid"]
    out["taker_cost_share_of_mid_edge"] = (
        out["selected_cross_cost"] / out["selected_mid_edge_contemp"].replace(0.0, np.nan)
    )

    resolution = out["chainlink_resolution_up"]
    resolution = resolution.where(resolution.notna(), out["binance_resolution_up"])
    out["resolution_up_used"] = resolution.astype(bool)
    out["payoff"] = np.where(
        out["best_route_stored"].eq("buy_up"),
        out["resolution_up_used"].astype(float),
        (~out["resolution_up_used"]).astype(float),
    )
    out["pnl_contemp"] = out["payoff"] - out["selected_ask"] - out["selected_fee"]
    out["pnl_stored"] = out["payoff"] - out["selected_ask"] - out["selected_fee"]
    out["p_model_abs_diff"] = (out["p_model_stored"] - out["p_model_contemp"]).abs()
    out["edge_abs_diff"] = (out["best_edge_stored"] - out["stored_route_edge_contemp"]).abs()
    return out


def mask_for_cell(df: pd.DataFrame, spec: CellSpec) -> pd.Series:
    mask = df["best_edge_stored"].gt(spec.min_edge) & df["stored_route_edge_contemp"].gt(spec.min_edge)
    if spec.far_only:
        mask &= df["abs_z"].astype(float).ge(1.0)
    if spec.late_only:
        mask &= df["time_bucket"].astype(str).eq("late_lt30m")
    if spec.longshot_price_max is not None:
        mask &= df["selected_ask"].astype(float).le(spec.longshot_price_max)
    return mask


def first_per_market(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    sub = df.loc[mask].sort_values(["market_slug", "ts"]).copy()
    if sub.empty:
        return sub
    return sub.groupby("market_slug", sort=False).head(1).reset_index(drop=True)


def summarize_cell(df: pd.DataFrame, spec: CellSpec, seed_offset: int) -> dict[str, object]:
    stale_mask = df["best_edge_stored"].gt(spec.min_edge)
    if spec.far_only:
        stale_mask &= df["abs_z"].astype(float).ge(1.0)
    if spec.late_only:
        stale_mask &= df["time_bucket"].astype(str).eq("late_lt30m")
    if spec.longshot_price_max is not None:
        stale_mask &= df["selected_ask"].astype(float).le(spec.longshot_price_max)
    contemp_mask = mask_for_cell(df, spec)
    selected = first_per_market(df, contemp_mask)
    stale_only = int((stale_mask & ~contemp_mask).sum())
    market_pnl = selected.groupby("market_slug", sort=False)["pnl_contemp"].sum().to_numpy(dtype=float)
    lo, hi = bootstrap_ci(market_pnl, seed_offset=seed_offset)
    mean = float(np.nanmean(market_pnl)) if len(market_pnl) else math.nan
    median = float(np.nanmedian(market_pnl)) if len(market_pnl) else math.nan
    win = float(np.mean(market_pnl > 0)) if len(market_pnl) else math.nan
    return {
        "cell_id": spec.cell_id,
        "description": spec.description,
        "min_edge_c": 100.0 * spec.min_edge,
        "far_only": spec.far_only,
        "late_only": spec.late_only,
        "longshot_price_max_c": None if spec.longshot_price_max is None else 100.0 * spec.longshot_price_max,
        "candidate_rows_stored": int(stale_mask.sum()),
        "candidate_rows_contemp": int(contemp_mask.sum()),
        "stale_only_rejected_rows": stale_only,
        "selected_markets": int(selected["market_slug"].nunique()),
        "selected_trades": int(len(selected)),
        "mean_pnl_c": 100.0 * mean if np.isfinite(mean) else math.nan,
        "median_pnl_c": 100.0 * median if np.isfinite(median) else math.nan,
        "ci_lo_c": 100.0 * lo if np.isfinite(lo) else math.nan,
        "ci_hi_c": 100.0 * hi if np.isfinite(hi) else math.nan,
        "win_rate": win,
        "mean_edge_c": 100.0 * float(selected["stored_route_edge_contemp"].mean()) if len(selected) else math.nan,
        "median_edge_c": 100.0 * float(selected["stored_route_edge_contemp"].median()) if len(selected) else math.nan,
        "mean_selected_ask_c": 100.0 * float(selected["selected_ask"].mean()) if len(selected) else math.nan,
        "median_abs_spread_c": 100.0 * float(selected["selected_spread"].median()) if len(selected) else math.nan,
        "median_rel_spread_pct": 100.0 * float(selected["selected_rel_spread"].median()) if len(selected) else math.nan,
        "median_cost_share": float(
            selected["taker_cost_share_of_mid_edge"].replace([np.inf, -np.inf], np.nan).median()
        )
        if len(selected)
        else math.nan,
        "mean_p_model_abs_diff_c": 100.0 * float(selected["p_model_abs_diff"].mean()) if len(selected) else math.nan,
        "max_p_model_abs_diff_c": 100.0 * float(selected["p_model_abs_diff"].max()) if len(selected) else math.nan,
    }


def selected_trade_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for spec in CELL_SPECS:
        sub = first_per_market(df, mask_for_cell(df, spec)).copy()
        if sub.empty:
            continue
        sub.insert(0, "cell_id", spec.cell_id)
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    cols = [
        "cell_id",
        "market_slug",
        "market_id",
        "asset",
        "ts",
        "window_start",
        "window_end",
        "best_route_stored",
        "selected_ask",
        "selected_fee",
        "selected_spread",
        "selected_rel_spread",
        "p_model_stored",
        "p_model_contemp",
        "stored_route_edge_contemp",
        "selected_mid_edge_contemp",
        "selected_cross_cost",
        "taker_cost_share_of_mid_edge",
        "abs_z",
        "moneyness_bucket",
        "time_bucket",
        "state_bucket",
        "payoff",
        "pnl_contemp",
        "source_run",
    ]
    return pd.concat(rows, ignore_index=True)[cols]


def spread_regime(df: pd.DataFrame) -> pd.DataFrame:
    groups = ["moneyness_bucket", "time_bucket", "state_bucket"]
    rows: list[dict[str, object]] = []
    for keys, g in df.groupby(groups, dropna=False, sort=True):
        candidate = g[g["stored_route_edge_contemp"].gt(0.0)]
        selected = first_per_market(g, g["stored_route_edge_contemp"].gt(0.0))
        cost_share = candidate["taker_cost_share_of_mid_edge"].replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "moneyness_bucket": keys[0],
                "time_bucket": keys[1],
                "state_bucket": keys[2],
                "rows": int(len(g)),
                "markets": int(g["market_slug"].nunique()),
                "candidate_rows_edge_gt_0": int(len(candidate)),
                "selected_markets_first_edge_gt_0": int(selected["market_slug"].nunique()),
                "median_abs_spread_c": 100.0 * float(g["selected_spread"].median()),
                "p90_abs_spread_c": 100.0 * float(g["selected_spread"].quantile(0.90)),
                "median_relative_spread_pct": 100.0 * float(g["selected_rel_spread"].median()),
                "p90_relative_spread_pct": 100.0 * float(g["selected_rel_spread"].quantile(0.90)),
                "median_taker_fee_c": 100.0 * float(g["selected_fee"].median()),
                "median_cross_cost_c": 100.0 * float(g["selected_cross_cost"].median()),
                "median_mid_edge_c_candidates": 100.0 * float(candidate["selected_mid_edge_contemp"].median())
                if len(candidate)
                else math.nan,
                "median_net_edge_c_candidates": 100.0 * float(candidate["stored_route_edge_contemp"].median())
                if len(candidate)
                else math.nan,
                "median_cost_share_candidates": float(cost_share.median()) if len(candidate) else math.nan,
                "max_net_edge_c": 100.0 * float(g["stored_route_edge_contemp"].max()),
                "mean_selected_ask_c_candidates": 100.0 * float(candidate["selected_ask"].mean())
                if len(candidate)
                else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["moneyness_bucket", "time_bucket"])


def write_outputs(summary: pd.DataFrame, spread: pd.DataFrame, trades: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    spread.to_csv(OUT_SPREAD, index=False)
    trades.to_parquet(OUT_TRADES, index=False)
    print(f"wrote {OUT_SUMMARY}")
    print(f"wrote {OUT_SPREAD}")
    print(f"wrote {OUT_TRADES}")


def main() -> None:
    panel = add_edges(base_panel())
    summary = pd.DataFrame([summarize_cell(panel, spec, i) for i, spec in enumerate(CELL_SPECS)])
    spread = spread_regime(panel)
    trades = selected_trade_table(panel)
    write_outputs(summary, spread, trades)

    primary = summary[summary["cell_id"].eq("strict_far_edge_gt_0")]
    if not primary.empty:
        row = primary.iloc[0]
        print(
            "primary strict_far_edge_gt_0: "
            f"markets={int(row['selected_markets'])} "
            f"mean={row['mean_pnl_c']:.2f}c "
            f"CI=[{row['ci_lo_c']:.2f}c,{row['ci_hi_c']:.2f}c] "
            f"stale_rejects={int(row['stale_only_rejected_rows'])}"
        )


if __name__ == "__main__":
    main()
