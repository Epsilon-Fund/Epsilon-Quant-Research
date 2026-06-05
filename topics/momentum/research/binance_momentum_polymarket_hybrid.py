#!/usr/bin/env python3
"""
Quick hybrid overlay check for daily Binance momentum plus Polymarket BTC/ETH binaries.

This script intentionally does not re-run WF/CPCV. It reads saved daily momentum
OOS artifacts and treats Polymarket files as data inputs only. It imports no
Polymarket project code.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
MOM_DIR = ROOT / "topics/momentum/strategies/momentum_cpcv/oos"
PM_CSV_DIR = ROOT / "polymarket/research/data/analysis/csv_outputs/options_delta"
PLOT_DIR = ROOT / "polymarket/research/data/analysis/plots/options_delta"
OUT_CSV = PM_CSV_DIR / "binance_momentum_polymarket_hybrid_results.csv"
BASELINE_CSV = PM_CSV_DIR / "binance_momentum_polymarket_hybrid_baseline.csv"
TIMING_CSV = PM_CSV_DIR / "binance_momentum_polymarket_hybrid_timing.csv"


ASSETS = ["BTC", "ETH"]
OVERLAY_MAX_PATHS = 200
PORTFOLIO_WEIGHTS = {
    "ADA": 1 / 6,
    "AVAX": 1 / 6,
    "BTC": 1 / 6,
    "ETH": 1 / 6,
    "SOL": 1 / 6,
    "XRP": 1 / 6,
}


def norm_cdf(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Normal CDF without scipy."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def returns_from_equity(eq: pd.Series) -> pd.Series:
    return eq.astype(float).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def max_drawdown(r: pd.Series) -> float:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def metrics_from_returns(r: pd.Series) -> dict[str, float]:
    r = r.dropna().astype(float)
    ppy = 365.0
    total = float((1.0 + r).prod() - 1.0)
    years = len(r) / ppy
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and total > -1 else np.nan
    vol = float(r.std(ddof=1) * math.sqrt(ppy))
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(ppy)) if r.std(ddof=1) > 0 else np.nan
    return {
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_dd": max_drawdown(r),
        "total_return": total,
        "n_days": len(r),
    }


def summarize_paths(paths: list[dict], return_key: str = "portfolio_returns") -> pd.DataFrame:
    rows = []
    for path_id, p in enumerate(paths):
        if return_key in p:
            r = p[return_key].copy()
        else:
            r = returns_from_equity(p["equity_curve"])
        row = metrics_from_returns(r)
        row["path_id"] = path_id
        row["start"] = r.index.min()
        row["end"] = r.index.max()
        rows.append(row)
    return pd.DataFrame(rows)


def split_map(asset_result: dict) -> dict[int, dict]:
    return {sr["split_id"]: sr for sr in asset_result.get("split_results", [])}


def build_asset_path_df(asset_result: dict, assignments: list[tuple[int, int]]) -> pd.DataFrame:
    smap = split_map(asset_result)
    pieces = []
    for group_idx, split_id in assignments:
        sr = smap.get(split_id)
        if sr is None:
            continue
        gr = sr["group_results"].get(group_idx)
        if gr is None:
            continue
        df = gr.get("oos_strategy_df")
        if df is not None and len(df):
            pieces.append(df.copy())
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def add_binary_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    log_close = np.log(out["Close"].astype(float))
    log_ret = log_close.diff()
    sigma = log_ret.ewm(span=60, min_periods=20).std().fillna(log_ret.expanding().std())
    sigma = sigma.clip(lower=0.005, upper=0.20)
    drift20 = log_ret.rolling(20, min_periods=5).mean().fillna(0.0).clip(-0.02, 0.02)
    out["next_close"] = out["Close"].shift(-1)
    out["sigma_daily"] = sigma
    out["drift20"] = drift20
    out["position_age"] = active_position_age(out["position"].fillna(0).astype(float))
    return out


def active_position_age(pos: pd.Series) -> pd.Series:
    age = []
    current = 0
    prev_active = False
    for val in pos.fillna(0).values:
        active = val != 0
        if active:
            current = current + 1 if prev_active else 1
        else:
            current = 0
        age.append(current)
        prev_active = active
    return pd.Series(age, index=pos.index)


@dataclass(frozen=True)
class Scenario:
    name: str
    interpretation: str
    size_pct: float
    strike: str
    kind: str
    entry_cost: float = 0.0
    gross_edge: float = 0.0
    slippage: float = 0.0
    min_model_prob: float = 0.0


SCENARIOS = [
    Scenario("A_hedge_ATM_5pct", "A hedge", 0.05, "ATM", "hedge", entry_cost=0.0125),
    Scenario("A_hedge_ATM_10pct", "A hedge", 0.10, "ATM", "hedge", entry_cost=0.0125),
    Scenario("A_hedge_ATM_20pct", "A hedge", 0.20, "ATM", "hedge", entry_cost=0.0125),
    Scenario("A_hedge_OTM2pct_5pct", "A hedge", 0.05, "OTM -2% down", "hedge", entry_cost=0.0250),
    Scenario("A_hedge_OTM2pct_10pct", "A hedge", 0.10, "OTM -2% down", "hedge", entry_cost=0.0250),
    Scenario("A_hedge_OTM2pct_20pct", "A hedge", 0.20, "OTM -2% down", "hedge", entry_cost=0.0250),
    Scenario("B_alpha_3c_edge_5pct", "B alpha sleeve", 0.05, "ATM up", "alpha", gross_edge=0.030, slippage=0.015, min_model_prob=0.53),
    Scenario("B_alpha_6c_edge_5pct", "B alpha sleeve", 0.05, "ATM up", "alpha", gross_edge=0.060, slippage=0.015, min_model_prob=0.53),
    Scenario("B_alpha_6c_edge_10pct", "B alpha sleeve", 0.10, "ATM up", "alpha", gross_edge=0.060, slippage=0.015, min_model_prob=0.53),
]


def scenario_asset_overlay(
    asset_df: pd.DataFrame,
    scenario: Scenario,
    asset_weight: float,
    collect_trades: bool = False,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, int]:
    if asset_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), 0

    df = asset_df.copy() if "next_close" in asset_df.columns else add_binary_model_columns(asset_df)
    active = (df["position"].fillna(0.0) > 0) & df["next_close"].notna()
    pos_size = df.get("position_size", pd.Series(1.0, index=df.index)).fillna(0.0).clip(lower=0.0)
    payout_notional = scenario.size_pct * asset_weight * pos_size

    if scenario.kind == "hedge":
        if scenario.strike == "ATM":
            strike = df["Close"].astype(float)
        else:
            strike = df["Close"].astype(float) * 0.98
        z_down = np.log(strike / df["Close"].astype(float)) / df["sigma_daily"]
        p_model = pd.Series(norm_cdf(z_down), index=df.index).clip(0.01, 0.99)
        entry_price = (p_model + scenario.entry_cost).clip(0.01, 0.99)
        payoff = (df["next_close"].astype(float) < strike).astype(float)
        enter = active
        side = "buy_down"
    else:
        strike = df["Close"].astype(float)
        z_up = (df["drift20"] - np.log(strike / df["Close"].astype(float))) / df["sigma_daily"]
        p_model = pd.Series(norm_cdf(z_up), index=df.index).clip(0.01, 0.99)
        entry_price = (p_model - scenario.gross_edge + scenario.slippage).clip(0.01, 0.99)
        payoff = (df["next_close"].astype(float) > strike).astype(float)
        enter = active & (p_model >= scenario.min_model_prob) & ((p_model - entry_price) >= 0.01)
        side = "buy_up"

    contracts = payout_notional.where(enter, 0.0)
    pnl = contracts * (payoff - entry_price)
    collateral = contracts * entry_price

    # Entry at t, resolution/PnL at t+1. Collateral is active over the holding day.
    pnl_resolved = pnl.shift(1).fillna(0.0)
    cap_active = collateral.fillna(0.0)
    n_entries = int(enter.sum())

    trades = pd.DataFrame()
    if collect_trades and n_entries:
        next_ts = df.index.to_series().shift(-1)
        mask_idx = enter[enter].index
        trades = pd.DataFrame(
            {
                "asset": None,
                "entry_ts": mask_idx,
                "resolution_ts": next_ts.loc[mask_idx].values,
                "scenario": scenario.name,
                "side": side,
                "strike_type": scenario.strike,
                "close": df.loc[mask_idx, "Close"].astype(float).values,
                "next_close": df.loc[mask_idx, "next_close"].astype(float).values,
                "position_size": pos_size.loc[mask_idx].astype(float).values,
                "position_age": df.loc[mask_idx, "position_age"].astype(float).values,
                "p_model": p_model.loc[mask_idx].astype(float).values,
                "entry_price": entry_price.loc[mask_idx].astype(float).values,
                "payoff": payoff.loc[mask_idx].astype(float).values,
                "contracts_per_book": contracts.loc[mask_idx].astype(float).values,
                "collateral_per_book": collateral.loc[mask_idx].astype(float).values,
                "pnl_per_book": pnl.loc[mask_idx].astype(float).values,
            }
        )

    return pnl_resolved, cap_active, trades, n_entries


def overlay_path(
    base_returns: pd.Series,
    asset_path_dfs: dict[str, pd.DataFrame],
    scenario: Scenario,
    collect_trades: bool = False,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, int]:
    overlay = pd.Series(0.0, index=base_returns.index)
    collateral = pd.Series(0.0, index=base_returns.index)
    all_trades = []
    n_entries_total = 0
    for asset in ASSETS:
        asset_df = asset_path_dfs.get(asset, pd.DataFrame())
        pnl, cap, trades, n_entries = scenario_asset_overlay(asset_df, scenario, PORTFOLIO_WEIGHTS[asset], collect_trades=collect_trades)
        n_entries_total += n_entries
        overlay = overlay.add(pnl.reindex(overlay.index).fillna(0.0), fill_value=0.0)
        collateral = collateral.add(cap.reindex(collateral.index).fillna(0.0), fill_value=0.0)
        if len(trades):
            trades["asset"] = asset
            all_trades.append(trades)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return overlay, collateral, trades_df, n_entries_total


def prebuild_asset_path_cache(portfolio_paths: list[dict], asset_results: dict[str, dict]) -> list[dict[str, pd.DataFrame]]:
    cache = []
    for p in portfolio_paths:
        entry = {}
        for asset in ASSETS:
            assignments = p["asset_split_assignments"].get(asset, [])
            df = build_asset_path_df(asset_results[asset], assignments)
            entry[asset] = add_binary_model_columns(df) if len(df) else df
        cache.append(entry)
    return cache


def load_current_pm_depth_summary() -> dict[str, float]:
    path = PM_CSV_DIR / "od_same_day_crypto_pricing_current_quotes.csv"
    if not path.exists():
        return {}
    q = pd.read_csv(path)
    q = q[(q["asset"].isin(["BTC", "ETH"])) & (q["resolution_class"] == "terminal_close")].copy()
    if q.empty:
        return {}
    q["yes_spread"] = q["yes_ask"] - q["yes_bid"]
    q["no_spread"] = q["no_ask"] - q["no_bid"]
    return {
        "snapshot_ts": str(q["ts"].dropna().iloc[0]) if q["ts"].notna().any() else "",
        "btc_terminal_median_24h_volume": float(q.loc[q["asset"] == "BTC", "volume_24h_usd"].median()),
        "eth_terminal_median_24h_volume": float(q.loc[q["asset"] == "ETH", "volume_24h_usd"].median()),
        "btc_terminal_median_yes_ask_size": float(q.loc[q["asset"] == "BTC", "yes_ask_size"].median()),
        "eth_terminal_median_yes_ask_size": float(q.loc[q["asset"] == "ETH", "yes_ask_size"].median()),
        "btc_terminal_median_spread": float(q.loc[q["asset"] == "BTC", "yes_spread"].median()),
        "eth_terminal_median_spread": float(q.loc[q["asset"] == "ETH", "yes_spread"].median()),
    }


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    PM_CSV_DIR.mkdir(parents=True, exist_ok=True)

    portfolio_paths = load_pickle(MOM_DIR / "portfolio_cpcv_paths.pkl")
    portfolio_trade = load_pickle(MOM_DIR / "portfolio_cpcv_trade_stats.pkl")
    portfolio_yearly = load_pickle(MOM_DIR / "portfolio_cpcv_yearly.pkl")
    btc_result = load_pickle(MOM_DIR / "btcusdt_cpcv.pkl")
    asset_results = {asset: load_pickle(MOM_DIR / f"{asset.lower()}usdt_cpcv.pkl") for asset in ASSETS}

    base_metrics_paths = summarize_paths(portfolio_paths)
    base_mean = base_metrics_paths[["cagr", "ann_vol", "sharpe", "max_dd", "total_return"]].mean()
    representative_path_id = int((base_metrics_paths["sharpe"] - base_mean["sharpe"]).abs().idxmin())
    sampled_ids = np.linspace(0, len(portfolio_paths) - 1, min(OVERLAY_MAX_PATHS, len(portfolio_paths)), dtype=int).tolist()
    path_ids = sorted(set(sampled_ids + [representative_path_id]))
    overlay_paths = [portfolio_paths[i] for i in path_ids]
    asset_path_cache = prebuild_asset_path_cache(overlay_paths, asset_results)

    btc_metrics_paths = summarize_paths(btc_result["paths"], return_key="not_present")
    btc_mean = btc_metrics_paths[["cagr", "ann_vol", "sharpe", "max_dd", "total_return"]].mean()

    baseline_rows = [
        {
            "series": "BTCUSDT daily momentum",
            "paths": len(btc_result["paths"]),
            "start": str(btc_metrics_paths["start"].min().date()),
            "end": str(btc_metrics_paths["end"].max().date()),
            **btc_mean.to_dict(),
        },
        {
            "series": "six_asset_daily_momentum",
            "paths": len(portfolio_paths),
            "start": str(base_metrics_paths["start"].min().date()),
            "end": str(base_metrics_paths["end"].max().date()),
            "avg_trade_return": portfolio_trade["summary"]["avg_trade_return"]["mean"],
            "hit_rate": portfolio_trade["summary"]["win_rate"]["mean"],
            **base_mean.to_dict(),
        },
    ]
    pd.DataFrame(baseline_rows).to_csv(BASELINE_CSV, index=False)

    results = []
    timing_frames = []
    scenario_equity_curves = {}
    scenario_dd_curves = {}

    for scenario in SCENARIOS:
        path_rows = []
        rep_overlay = None
        rep_combined = None
        rep_collateral = None
        collect_trades = scenario.name in {"A_hedge_ATM_10pct", "B_alpha_6c_edge_5pct"}
        for local_idx, path_id in enumerate(path_ids):
            p = overlay_paths[local_idx]
            base_ret = p["portfolio_returns"].copy()
            overlay_ret, collateral, trades, n_entries = overlay_path(
                base_ret,
                asset_path_cache[local_idx],
                scenario,
                collect_trades=collect_trades,
            )
            combined_ret = base_ret.add(overlay_ret.reindex(base_ret.index).fillna(0.0), fill_value=0.0)
            m = metrics_from_returns(combined_ret)
            overlay_pnl_total = float((1.0 + combined_ret).prod() - (1.0 + base_ret).prod())
            avg_collateral = float(collateral.mean())
            annual_overlay_pnl = float(overlay_ret.mean() * 365.0)
            overlay_roc = annual_overlay_pnl / avg_collateral if avg_collateral > 0 else np.nan
            m.update(
                {
                    "path_id": path_id,
                    "overlay_total_pnl_per_book": overlay_pnl_total,
                    "avg_pm_collateral_per_book": avg_collateral,
                    "overlay_roc_ann": overlay_roc,
                    "n_overlay_entries": int(n_entries),
                }
            )
            path_rows.append(m)
            if len(trades):
                trades["path_id"] = path_id
                timing_frames.append(trades)
            if path_id == representative_path_id:
                rep_overlay = overlay_ret
                rep_combined = combined_ret
                rep_collateral = collateral
        path_df = pd.DataFrame(path_rows)
        row = path_df[["cagr", "ann_vol", "sharpe", "max_dd", "overlay_roc_ann", "avg_pm_collateral_per_book", "n_overlay_entries"]].mean().to_dict()
        row.update(
            {
                "scenario": scenario.name,
                "interpretation": scenario.interpretation,
                "overlay_paths": len(path_ids),
                "size_pct_of_asset_notional": scenario.size_pct,
                "strike": scenario.strike,
                "delta_cagr": row["cagr"] - base_mean["cagr"],
                "delta_ann_vol": row["ann_vol"] - base_mean["ann_vol"],
                "delta_sharpe": row["sharpe"] - base_mean["sharpe"],
                "delta_max_dd": row["max_dd"] - base_mean["max_dd"],
            }
        )
        results.append(row)

        if rep_combined is not None:
            scenario_equity_curves[scenario.name] = (1.0 + rep_combined.fillna(0.0)).cumprod()
            scenario_dd_curves[scenario.name] = scenario_equity_curves[scenario.name] / scenario_equity_curves[scenario.name].cummax() - 1.0

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_CSV, index=False)

    timing_df = pd.concat(timing_frames, ignore_index=True) if timing_frames else pd.DataFrame()
    if timing_df.empty:
        pd.DataFrame().to_csv(TIMING_CSV, index=False)
    else:
        timing_summary = (
            timing_df.groupby(["scenario", "asset"])
            .agg(
                n_entries=("pnl_per_book", "size"),
                mean_pnl_per_book=("pnl_per_book", "mean"),
                mean_contracts_per_book=("contracts_per_book", "mean"),
                mean_collateral_per_book=("collateral_per_book", "mean"),
                payoff_rate=("payoff", "mean"),
                median_position_age=("position_age", "median"),
                mean_entry_price=("entry_price", "mean"),
                mean_model_prob=("p_model", "mean"),
            )
            .reset_index()
        )
        timing_summary.to_csv(TIMING_CSV, index=False)

    rep_base_ret = portfolio_paths[representative_path_id]["portfolio_returns"].copy()
    rep_base_eq = (1.0 + rep_base_ret.fillna(0.0)).cumprod()
    rep_base_dd = rep_base_eq / rep_base_eq.cummax() - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(rep_base_eq.index, rep_base_eq.values, label="Baseline daily momentum", linewidth=2.0, color="#1f2937")
    for name in ["A_hedge_ATM_10pct", "A_hedge_OTM2pct_10pct", "B_alpha_6c_edge_5pct"]:
        if name in scenario_equity_curves:
            axes[0].plot(scenario_equity_curves[name].index, scenario_equity_curves[name].values, label=name, linewidth=1.2)
    axes[0].set_title("Representative CPCV Path: Baseline vs Hybrid Overlay Equity")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(rep_base_dd.index, rep_base_dd.values, label="Baseline daily momentum", linewidth=2.0, color="#1f2937")
    for name in ["A_hedge_ATM_10pct", "A_hedge_OTM2pct_10pct", "B_alpha_6c_edge_5pct"]:
        if name in scenario_dd_curves:
            axes[1].plot(scenario_dd_curves[name].index, scenario_dd_curves[name].values, label=name, linewidth=1.2)
    axes[1].set_title("Representative CPCV Path: Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    eq_path = PLOT_DIR / "binance_momentum_polymarket_hybrid_equity_drawdown.png"
    fig.savefig(eq_path, dpi=160)
    plt.close(fig)

    if not timing_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for scenario_name, grp in timing_df[timing_df["scenario"].isin(["A_hedge_ATM_10pct", "B_alpha_6c_edge_5pct"])].groupby("scenario"):
            ax.hist(grp["position_age"], bins=np.arange(1, 46, 2), alpha=0.55, label=scenario_name)
        ax.set_title("Overlay Entry Timing Within Existing Momentum Positions")
        ax.set_xlabel("Momentum position age at binary entry (daily bars)")
        ax.set_ylabel("Entry count across sampled CPCV paths")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        timing_path = PLOT_DIR / "binance_momentum_polymarket_hybrid_timing_distribution.png"
        fig.savefig(timing_path, dpi=160)
        plt.close(fig)

    depth = load_current_pm_depth_summary()
    meta_path = PM_CSV_DIR / "binance_momentum_polymarket_hybrid_metadata.csv"
    pd.DataFrame([depth]).to_csv(meta_path, index=False)

    print(f"baseline_csv={BASELINE_CSV}")
    print(f"results_csv={OUT_CSV}")
    print(f"timing_csv={TIMING_CSV}")
    print(f"equity_plot={eq_path}")
    print(f"representative_path_id={representative_path_id}")
    if not timing_df.empty:
        print(f"timing_plot={timing_path}")
    print(results_df.sort_values(["interpretation", "size_pct_of_asset_notional"]).to_string(index=False))


if __name__ == "__main__":
    main()
