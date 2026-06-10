"""OD Strategy A v3 PnL/risk deep-dive.

Polymarket-only follow-up to v3:
- per-asset concurrent capital is the primary model;
- global one-slot embargo is a conservative sensitivity;
- no Binance hedge sweeps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import cents, number, pct
from od_strategy_a_v2_lifecycle import (
    A1_FEATURES,
    ANALYSIS,
    BRAIN_TODO,
    BOOTSTRAP_SAMPLES,
    NOTES,
    OD_HUB,
    RNG_SEED,
    markdown_table,
    parse_fill_ids,
)
from od_strategy_a_v3 import (
    FilterSpec,
    GATE_BUCKET,
    GATE_SPLIT,
    NOTE as V3_NOTE,
    build_episode_set,
    bucket_mask,
    filter_mask,
    load_v3_fills,
    normalize_markdown_wrapping,
    resolve_token_rv_physical_prob_fair,
)


OUT_CSV = ANALYSIS / "csv_outputs" / "options_delta" / "od_strategy_a_v3_pnl_risk.csv"
OUT_EPISODES = ANALYSIS / "od_strategy_a_v3_pnl_risk_episodes.parquet"
OUT_DISTRIBUTIONS = ANALYSIS / "od_strategy_a_v3_pnl_risk_distributions.parquet"
OUT_CAPTURE = ANALYSIS / "od_strategy_a_v3_pnl_risk_capture_curve.parquet"
NOTE = NOTES / "options_delta" / "od_strategy_a_v3_pnl_risk_findings.md"

PRIMARY_SPEC = FilterSpec(
    "phase2_od_filter",
    "strict_rich_short_ge_010m",
    source_policy="strict",
    richness_threshold=0.01,
)
STRICT_SOURCE_SPEC = FilterSpec("phase2_od_filter", "official_strict_source", source_policy="strict")
BARE_SPEC = FilterSpec("phase1_power", "bare_lifecycle")
RICH_NO_SOURCE_SPEC = FilterSpec("phase2_od_filter", "rich_short_ge_010m", richness_threshold=0.01)

PRIMARY_EMBARGO = "per_asset_time_nonoverlap"
GLOBAL_EMBARGO = "global_time_nonoverlap"
NON_TOP3_AVAILABLE_SHARE = 0.05
MM_CRYPTO_STRUCTURED_MEDIAN_BPS = 2.4
MM_CRYPTO_STRUCTURED_NON_TOP3_BPS = 189.0
MM_CRYPTO_STRUCTURED_NON_TOP3_CI_LO_BPS = 21.8
MM_CRYPTO_DEPLOYABLE_EV_DAY_USD = 0.0


@dataclass(frozen=True)
class EpisodeArtifacts:
    fills: pd.DataFrame
    episodes: pd.DataFrame
    summary: pd.DataFrame
    sizing: pd.DataFrame
    capture: pd.DataFrame
    tail: pd.DataFrame
    incremental: pd.DataFrame


def ci_text(lo: float, hi: float, *, unit: str = "c") -> str:
    if unit == "pct":
        return f"[{pct(lo)}, {pct(hi)}]"
    if unit == "usd":
        return f"[${lo:,.2f}, ${hi:,.2f}]"
    if unit == "bps":
        return f"[{lo:,.1f} bps, {hi:,.1f} bps]"
    return f"[{cents(lo)}, {cents(hi)}]"


def fmt_usd(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"${value:,.2f}"


def bootstrap_mean_ci(vals: np.ndarray, seed_offset: int = 0) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.sort(vals)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(RNG_SEED + seed_offset + len(vals))
    draws = rng.integers(0, len(vals), size=(BOOTSTRAP_SAMPLES, len(vals)))
    boot = vals[draws].mean(axis=1)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def complement(side: str) -> str:
    return "down" if str(side).lower() == "up" else "up"


def add_claim_fields(fills: pd.DataFrame) -> pd.DataFrame:
    out = fills.copy()
    token_side = out["actual_outcome"].astype(str).str.lower()
    is_long = out["token_position"].astype(float).gt(0)
    token_rv_fair = resolve_token_rv_physical_prob_fair(out, context="od_strategy_a_v3_pnl_risk.add_claim_fields")
    out["claim_side"] = np.where(is_long, token_side, token_side.map(complement))
    out["claim_unit_cost"] = np.where(is_long, out["entry_price"].astype(float), 1.0 - out["entry_price"].astype(float))
    out["claim_rv_physical_prob_fair"] = np.where(is_long, token_rv_fair.astype(float), 1.0 - token_rv_fair.astype(float))
    out["claim_model_fair"] = out["claim_rv_physical_prob_fair"]  # Legacy alias.
    out["claim_fair_kind"] = "rv_physical_prob"
    out["edge_vs_rv_physical_prob"] = out["claim_rv_physical_prob_fair"] - out["claim_unit_cost"]
    out["claim_edge"] = out["edge_vs_rv_physical_prob"]  # Legacy alias.
    out["claim_payoff"] = np.where(is_long, out["payoff"].astype(float), 1.0 - out["payoff"].astype(float))
    out["claim_unit_pnl"] = out["claim_payoff"] - out["claim_unit_cost"] + out["maker_rebate"].astype(float)
    out["claim_unit_delta_dollars"] = out["signed_delta_exposure"].astype(float) * out["binance_spot"].astype(float)
    return out


def selected_fill_mask(fills: pd.DataFrame, spec: FilterSpec = PRIMARY_SPEC) -> pd.Series:
    return filter_mask(fills, spec) & fills["entry_split"].eq(GATE_SPLIT) & bucket_mask(fills, GATE_BUCKET)


def as_episode_fills(ep: pd.Series, fills_by_id: pd.DataFrame) -> pd.DataFrame:
    ids = parse_fill_ids(str(ep["fill_ids"]))
    return fills_by_id.loc[ids].sort_values("fill_ts_key").copy()


def risk_after_two_sided_pairing(up_qty: float, up_cost: float, down_qty: float, down_cost: float) -> float:
    if up_qty <= 0 and down_qty <= 0:
        return 0.0
    pair_qty = min(up_qty, down_qty)
    up_avg = up_cost / up_qty if up_qty > 1e-12 else 0.0
    down_avg = down_cost / down_qty if down_qty > 1e-12 else 0.0
    paired_cost = pair_qty * (up_avg + down_avg)
    unmatched_cost = max(0.0, up_qty - pair_qty) * up_avg + max(0.0, down_qty - pair_qty) * down_avg
    overpaid_pair_risk = max(0.0, paired_cost - pair_qty)
    return float(unmatched_cost + overpaid_pair_risk)


def episode_from_weighted_fills(g: pd.DataFrame, *, policy: str) -> dict[str, Any]:
    g = g.sort_values("fill_ts_key").copy()
    up_qty = down_qty = 0.0
    up_cost = down_cost = 0.0
    running_delta = 0.0
    peak_complement = 0.0
    peak_two_sided_risk = 0.0
    peak_abs_delta = 0.0
    total_cost = 0.0
    total_rebate = 0.0
    settlement = 0.0

    for _, row in g.iterrows():
        size = float(row.get("size", 1.0))
        if size <= 0:
            continue
        cost = float(row["claim_unit_cost"]) * size
        total_cost += cost
        total_rebate += float(row["maker_rebate"]) * size
        settlement += float(row["claim_payoff"]) * size
        if row["claim_side"] == "up":
            up_qty += size
            up_cost += cost
        else:
            down_qty += size
            down_cost += cost
        running_delta += float(row["claim_unit_delta_dollars"]) * size
        peak_complement = max(peak_complement, up_cost + down_cost)
        peak_two_sided_risk = max(peak_two_sided_risk, risk_after_two_sided_pairing(up_qty, up_cost, down_qty, down_cost))
        peak_abs_delta = max(peak_abs_delta, abs(running_delta))

    net_pnl = settlement - total_cost + total_rebate
    first = g.iloc[0]
    end_ts = pd.Timestamp(first["window_end"])
    capital = peak_complement if peak_complement > 1e-12 else math.nan
    net_risk_capital = peak_two_sided_risk if peak_two_sided_risk > 1e-12 else math.nan
    return {
        "policy": policy,
        "market_id": str(first["market_id"]),
        "market_slug": str(first["market_slug"]),
        "asset": str(first["asset"]),
        "window_start": pd.Timestamp(first["window_start"]),
        "window_end": end_ts,
        "start_ts": pd.Timestamp(g["fill_ts"].min()),
        "end_ts": end_ts,
        "n_fills": int(len(g)),
        "weighted_fills": float(g["size"].sum()),
        "net_pnl": float(net_pnl),
        "settlement_value": float(settlement),
        "claim_cost": float(total_cost),
        "maker_rebate": float(total_rebate),
        "peak_complement_capital": float(peak_complement),
        "peak_two_sided_risk_capital": float(peak_two_sided_risk),
        "peak_abs_dollar_delta": float(peak_abs_delta),
        "capital_at_risk": float(capital),
        "roc": float(net_pnl / capital) if np.isfinite(capital) and capital > 1e-12 else math.nan,
        "roc_two_sided_risk": float(net_pnl / net_risk_capital) if np.isfinite(net_risk_capital) and net_risk_capital > 1e-12 else math.nan,
        "mean_edge_vs_rv_physical_prob": float(np.average(g["edge_vs_rv_physical_prob"], weights=g["size"])) if g["size"].sum() > 0 else math.nan,
        "mean_claim_edge": float(np.average(g["claim_edge"], weights=g["size"])) if g["size"].sum() > 0 else math.nan,
        "mean_entry_price": float(np.average(g["entry_price"], weights=g["size"])) if g["size"].sum() > 0 else math.nan,
        "median_abs_z": float(g["abs_z"].median()),
        "median_seconds_to_expiry": float(g["seconds_to_expiry"].median()),
        "hold_seconds": float((end_ts - pd.Timestamp(g["fill_ts"].median())).total_seconds()),
        "two_sided_claims": bool(g["claim_side"].nunique() >= 2),
        "any_shorted_token_itm": bool((g["token_position"].astype(float).lt(0) & g["payoff"].astype(float).eq(1.0)).any()),
    }


def build_weighted_episodes(fills: pd.DataFrame, *, policy: str, selected_mask: pd.Series) -> pd.DataFrame:
    sub = fills[selected_mask].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["size"] = 1.0
    if policy == "flat_1_contract":
        pass
    elif policy in {"rv_edge_scaled", "fair_value_scaled"}:
        sub["size"] = np.clip(sub["edge_vs_rv_physical_prob"].astype(float) / 0.05, 0.25, 3.0)
    elif policy == "fractional_kelly_25pct":
        q = sub["claim_rv_physical_prob_fair"].astype(float).clip(1e-6, 1 - 1e-6)
        c = sub["claim_unit_cost"].astype(float).clip(1e-6, 1 - 1e-6)
        full_kelly_fraction = ((q - c) / (1.0 - c)).clip(lower=0.0)
        sub["size"] = np.clip(0.25 * full_kelly_fraction / c, 0.0, 3.0)
    elif policy == "dollar_delta_cap_50":
        rows: list[pd.DataFrame] = []
        for _, g in sub.groupby("market_id", sort=False):
            running = 0.0
            sizes: list[float] = []
            for _, row in g.sort_values("fill_ts_key").iterrows():
                unit_delta = abs(float(row["claim_unit_delta_dollars"]))
                remaining = max(0.0, 50.0 - running)
                size = 1.0 if unit_delta <= 1e-12 else min(1.0, remaining / unit_delta)
                sizes.append(size)
                running += unit_delta * size
            gg = g.sort_values("fill_ts_key").copy()
            gg["size"] = sizes
            rows.append(gg[gg["size"].gt(1e-9)])
        sub = pd.concat(rows, ignore_index=True) if rows else sub.iloc[0:0].copy()
    else:
        raise ValueError(f"unknown sizing policy {policy}")

    rows = [episode_from_weighted_fills(g, policy=policy) for _, g in sub.groupby("market_id", sort=False) if not g.empty]
    return pd.DataFrame(rows)


def active_days(eps: pd.DataFrame) -> float:
    if eps.empty:
        return math.nan
    start = pd.to_datetime(eps["window_start"]).min()
    end = pd.to_datetime(eps["window_end"]).max()
    seconds = max((end - start).total_seconds(), 1.0)
    return seconds / 86400.0


def peak_concurrent_capital(eps: pd.DataFrame, capital_col: str = "capital_at_risk") -> float:
    events: list[dict[str, Any]] = []
    for _, row in eps.iterrows():
        cap = float(row.get(capital_col, math.nan))
        if not np.isfinite(cap):
            continue
        events.append({"ts": pd.Timestamp(row["start_ts"]), "delta": cap})
        events.append({"ts": pd.Timestamp(row["window_end"]), "delta": -cap})
    if not events:
        return math.nan
    ev = pd.DataFrame(events).sort_values(["ts", "delta"], ascending=[True, False])
    running = ev["delta"].cumsum()
    return float(running.max())


def summarize_episode_distribution(eps: pd.DataFrame, *, label: str, benchmark_bps: float | None = None) -> dict[str, Any]:
    if eps.empty:
        return {"label": label}
    pnl = eps["net_pnl"].to_numpy(dtype=float)
    roc = eps["roc"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    mean_lo, mean_hi = bootstrap_mean_ci(pnl, seed_offset=10)
    roc_lo, roc_hi = bootstrap_mean_ci(roc, seed_offset=20)
    days = active_days(eps)
    cadence = len(eps) / days if np.isfinite(days) and days > 0 else math.nan
    daily_vals = pnl * cadence if np.isfinite(cadence) else np.array([])
    day_lo, day_hi = bootstrap_mean_ci(daily_vals, seed_offset=30) if len(daily_vals) else (math.nan, math.nan)
    peak_cap = peak_concurrent_capital(eps)
    daily_mean = float(np.mean(pnl) * cadence) if np.isfinite(cadence) else math.nan
    daily_return = daily_mean / peak_cap if np.isfinite(peak_cap) and peak_cap > 1e-12 else math.nan
    lower_vs_benchmark = math.nan
    if benchmark_bps is not None and np.isfinite(roc_lo):
        lower_vs_benchmark = roc_lo * 10_000.0 - benchmark_bps
    return {
        "label": label,
        "n_markets": int(eps["market_id"].nunique()),
        "n_fills": int(eps["n_fills"].sum()),
        "active_days": float(days),
        "episodes_per_day": float(cadence),
        "mean_net_pnl": float(np.mean(pnl)),
        "median_net_pnl": float(np.median(pnl)),
        "net_ci_lo": mean_lo,
        "net_ci_hi": mean_hi,
        "win_rate": float(np.mean(pnl > 0)),
        "pnl_std": float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0,
        "mean_capital_at_risk": float(eps["capital_at_risk"].mean()),
        "median_capital_at_risk": float(eps["capital_at_risk"].median()),
        "peak_concurrent_capital": peak_cap,
        "mean_peak_delta": float(eps["peak_abs_dollar_delta"].mean()),
        "mean_roc": float(np.nanmean(roc)) if len(roc) else math.nan,
        "median_roc": float(np.nanmedian(roc)) if len(roc) else math.nan,
        "roc_ci_lo": roc_lo,
        "roc_ci_hi": roc_hi,
        "mean_roc_bps": float(np.nanmean(roc) * 10_000.0) if len(roc) else math.nan,
        "roc_ci_lo_bps": float(roc_lo * 10_000.0) if np.isfinite(roc_lo) else math.nan,
        "daily_pnl": daily_mean,
        "daily_pnl_ci_lo": day_lo,
        "daily_pnl_ci_hi": day_hi,
        "daily_return_on_peak_capital": daily_return,
        "lower_roc_bps_minus_benchmark": lower_vs_benchmark,
        "mean_hold_minutes": float(eps["hold_seconds"].mean() / 60.0),
        "median_hold_minutes": float(eps["hold_seconds"].median() / 60.0),
        "short_itm_episode_share": float(eps["any_shorted_token_itm"].mean()),
    }


def summarize_sizing(sizing_eps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, g in sizing_eps.groupby("policy", sort=True):
        row = summarize_episode_distribution(g, label=policy, benchmark_bps=None)
        pnl = g["net_pnl"].sort_values().to_numpy(dtype=float)
        tail_n = max(1, int(math.ceil(0.05 * len(pnl)))) if len(pnl) else 0
        row.update(
            {
                "policy": policy,
                "cvar_5": float(np.mean(pnl[:tail_n])) if tail_n else math.nan,
                "worst_episode": float(np.min(pnl)) if len(pnl) else math.nan,
                "mean_weighted_fills": float(g["weighted_fills"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def max_drawdown(vals: list[float]) -> float:
    if not vals:
        return math.nan
    equity = np.cumsum(np.asarray(vals, dtype=float))
    peaks = np.maximum.accumulate(np.r_[0.0, equity])[:-1]
    dd = peaks - equity
    return float(max(0.0, np.max(dd))) if len(dd) else 0.0


def build_tail_table(primary_eps: pd.DataFrame, fills: pd.DataFrame, primary_mask: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pnl = primary_eps.sort_values("start_ts")["net_pnl"].to_numpy(dtype=float)
    tail_n = max(1, int(math.ceil(0.05 * len(pnl)))) if len(pnl) else 0
    rows.append(
        {
            "row_type": "observed_primary",
            "n_markets": len(primary_eps),
            "mean_net_pnl": float(np.mean(pnl)) if len(pnl) else math.nan,
            "median_net_pnl": float(np.median(pnl)) if len(pnl) else math.nan,
            "worst_episode": float(np.min(pnl)) if len(pnl) else math.nan,
            "cvar_5": float(np.mean(np.sort(pnl)[:tail_n])) if tail_n else math.nan,
            "max_drawdown": max_drawdown(list(pnl)),
            "short_itm_episode_share": float(primary_eps["any_shorted_token_itm"].mean()) if len(primary_eps) else math.nan,
        }
    )

    full_mask = (
        fills["eligible"].fillna(False).astype(bool)
        & fills["strict_source_eligible"].fillna(False).astype(bool)
        & fills["abs_z"].astype(float).ge(1.0)
        & fills["token_position"].astype(float).lt(0)
        & fills["rich_short_edge"].astype(float).ge(0.01)
    )
    full_short = fills[full_mask].copy()
    itm_rate = float(full_short["payoff"].astype(float).eq(1.0).mean()) if len(full_short) else math.nan

    selected = fills[primary_mask].copy()
    rng = np.random.default_rng(RNG_SEED + 404)
    sims = []
    sim_drawdowns = []
    if len(selected) and np.isfinite(itm_rate):
        by_market = {m: g.copy() for m, g in selected.groupby("market_id", sort=False)}
        market_ids = list(by_market)
        for _ in range(5000):
            ep_pnl = []
            for market_id in market_ids:
                g = by_market[market_id]
                token_wins = rng.random(len(g)) < itm_rate
                unit_pnl = np.where(
                    token_wins,
                    g["entry_price"].to_numpy(dtype=float) - 1.0 + g["maker_rebate"].to_numpy(dtype=float),
                    g["entry_price"].to_numpy(dtype=float) + g["maker_rebate"].to_numpy(dtype=float),
                )
                ep_pnl.append(float(unit_pnl.sum()))
            sims.append(float(np.mean(ep_pnl)))
            sim_drawdowns.append(max_drawdown(ep_pnl))
    rows.append(
        {
            "row_type": "stress_empirical_far_short_itm",
            "n_markets": len(primary_eps),
            "full_panel_short_fills": int(len(full_short)),
            "empirical_short_itm_rate": itm_rate,
            "mean_net_pnl": float(np.mean(sims)) if sims else math.nan,
            "median_net_pnl": float(np.median(sims)) if sims else math.nan,
            "net_ci_lo": float(np.quantile(sims, 0.025)) if sims else math.nan,
            "net_ci_hi": float(np.quantile(sims, 0.975)) if sims else math.nan,
            "max_drawdown": float(np.mean(sim_drawdowns)) if sim_drawdowns else math.nan,
            "stress_trials": len(sims),
        }
    )

    if len(primary_eps):
        slot = primary_eps.copy()
        slot["slot"] = pd.to_datetime(slot["window_start"]).astype("int64")
        slot_pnl = slot.groupby("slot")["net_pnl"].sum().sort_index().to_numpy(dtype=float)
        slot_tail_n = max(1, int(math.ceil(0.05 * len(slot_pnl)))) if len(slot_pnl) else 0
        rows.append(
            {
                "row_type": "observed_concurrent_slot_portfolio",
                "n_markets": len(primary_eps),
                "n_slots": int(len(slot_pnl)),
                "mean_net_pnl": float(np.mean(slot_pnl)) if len(slot_pnl) else math.nan,
                "median_net_pnl": float(np.median(slot_pnl)) if len(slot_pnl) else math.nan,
                "worst_episode": float(np.min(slot_pnl)) if len(slot_pnl) else math.nan,
                "cvar_5": float(np.mean(np.sort(slot_pnl)[:slot_tail_n])) if slot_tail_n else math.nan,
                "max_drawdown": max_drawdown(list(slot_pnl)),
                "pnl_std": float(np.std(slot_pnl, ddof=1)) if len(slot_pnl) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def enrich_depth(fills: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "fill_id",
        "asset_id",
        "market_id",
        "fill_ts",
        "entry_price",
        "token_position",
        "half_spread",
        "claim_unit_pnl",
        "edge_vs_rv_physical_prob",
        "claim_edge",
        "claim_unit_cost",
    ]
    small = fills[cols].copy()
    small["asset_id"] = small["asset_id"].astype(str)
    small["market_id"] = small["market_id"].astype(str)
    con = duckdb.connect()
    con.register("fills", small)
    depth = con.execute(
        f"""
        SELECT
            f.fill_id,
            b.received_at AS book_ts,
            b.best_bid,
            b.best_bid_size,
            b.best_ask,
            b.best_ask_size,
            b.bid_top5_shares,
            b.ask_top5_shares,
            abs(b.signed_trade_size_300s) AS observed_flow_300s,
            b.trade_count_300s
        FROM fills f
        ASOF LEFT JOIN read_parquet('{A1_FEATURES.as_posix()}') b
        ON f.asset_id = b.asset_id AND f.fill_ts >= b.received_at
        ORDER BY f.fill_id
        """
    ).fetchdf()
    out = small.merge(depth, on="fill_id", how="left")
    is_sell = out["token_position"].astype(float).lt(0)
    out["touch_depth_shares"] = np.where(is_sell, out["best_ask_size"], out["best_bid_size"])
    out["top5_depth_shares"] = np.where(is_sell, out["ask_top5_shares"], out["bid_top5_shares"])
    out["observed_flow_300s"] = out["observed_flow_300s"].fillna(0.0)
    out["capture_flow_shares"] = np.maximum(1.0, out["observed_flow_300s"].astype(float))
    out["touch_depth_shares"] = out["touch_depth_shares"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.0)
    out["top5_depth_shares"] = out["top5_depth_shares"].replace([np.inf, -np.inf], np.nan).fillna(out["touch_depth_shares"]).clip(lower=0.0)
    return out


def build_capture_curve(selected_fills: pd.DataFrame) -> pd.DataFrame:
    if selected_fills.empty:
        return pd.DataFrame()
    depth = enrich_depth(selected_fills)
    rows: list[dict[str, Any]] = []
    targets = [1, 2, 5, 10, 25, 50, 100]
    for target in targets:
        gross_units = np.minimum.reduce(
            [
                np.full(len(depth), float(target)),
                depth["top5_depth_shares"].to_numpy(dtype=float),
                depth["capture_flow_shares"].to_numpy(dtype=float),
            ]
        )
        touch = depth["touch_depth_shares"].to_numpy(dtype=float)
        half_spread = depth["half_spread"].astype(float).fillna(0.0).to_numpy()
        slippage = np.where(gross_units > 1e-12, np.maximum(0.0, gross_units - touch) / gross_units * half_spread, 0.0)
        expected_unit = depth["edge_vs_rv_physical_prob"].to_numpy(dtype=float) + selected_fills.set_index("fill_id").loc[depth["fill_id"], "maker_rebate"].to_numpy(dtype=float) - slippage
        realized_unit = depth["claim_unit_pnl"].to_numpy(dtype=float) - slippage
        realistic_units = gross_units * NON_TOP3_AVAILABLE_SHARE
        rows.append(
            {
                "target_contracts_per_fill": target,
                "n_fills": int(len(depth)),
                "gross_units": float(np.nansum(gross_units)),
                "realistic_units_after_top3_haircut": float(np.nansum(realistic_units)),
                "gross_expected_ev": float(np.nansum(expected_unit * gross_units)),
                "gross_realized_pnl": float(np.nansum(realized_unit * gross_units)),
                "realistic_expected_ev": float(np.nansum(expected_unit * realistic_units)),
                "realistic_realized_pnl": float(np.nansum(realized_unit * realistic_units)),
                "realistic_capital": float(np.nansum(depth["claim_unit_cost"].to_numpy(dtype=float) * realistic_units)),
                "mean_touch_depth": float(np.nanmean(touch)),
                "mean_top5_depth": float(np.nanmean(depth["top5_depth_shares"].to_numpy(dtype=float))),
                "mean_observed_flow_300s": float(np.nanmean(depth["capture_flow_shares"].to_numpy(dtype=float))),
                "mean_slippage_penalty": float(np.nanmean(slippage)),
            }
        )
    out = pd.DataFrame(rows)
    days = active_days(
        selected_fills.groupby("market_id", as_index=False)
        .agg(window_start=("window_start", "first"), window_end=("window_end", "first"), start_ts=("fill_ts", "min"), net_pnl=("claim_unit_pnl", "sum"))
    )
    out["realistic_expected_ev_per_day"] = out["realistic_expected_ev"] / days if np.isfinite(days) and days > 0 else math.nan
    out["realistic_realized_pnl_per_day"] = out["realistic_realized_pnl"] / days if np.isfinite(days) and days > 0 else math.nan
    return out


def build_incremental_table(fills: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    configs = [
        ("bare_lifecycle_per_asset", BARE_SPEC, PRIMARY_EMBARGO),
        ("official_strict_source_per_asset", STRICT_SOURCE_SPEC, PRIMARY_EMBARGO),
        ("rich_short_no_source_per_asset", RICH_NO_SOURCE_SPEC, PRIMARY_EMBARGO),
        ("strict_rich_short_per_asset", PRIMARY_SPEC, PRIMARY_EMBARGO),
        ("strict_rich_short_global_sensitivity", PRIMARY_SPEC, GLOBAL_EMBARGO),
    ]
    fills_by_id = fills.set_index("fill_id", drop=False)
    weighted_by_label: dict[str, pd.DataFrame] = {}
    for label, spec, embargo in configs:
        eps = build_episode_set(fills, spec=spec, bucket=GATE_BUCKET, split=GATE_SPLIT, embargo_mode=embargo)
        if eps.empty:
            continue
        # Recompute through the claim/capital lens so all rows share the same accounting.
        ep_rows = []
        for _, ep in eps.iterrows():
            g = as_episode_fills(ep, fills_by_id)
            g["size"] = 1.0
            ep_rows.append(episode_from_weighted_fills(g, policy=label))
        weps = pd.DataFrame(ep_rows)
        weighted_by_label[label] = weps
        row = summarize_episode_distribution(weps, label=label, benchmark_bps=MM_CRYPTO_STRUCTURED_MEDIAN_BPS)
        row.update({"row_type": "strategy_variant", "variant": label})
        rows.append(row)

    primary = weighted_by_label.get("strict_rich_short_per_asset")
    strict = weighted_by_label.get("official_strict_source_per_asset")
    if primary is not None and strict is not None and not primary.empty and not strict.empty:
        primary_markets = set(primary["market_id"])
        strict_same = strict[strict["market_id"].isin(primary_markets)].copy()
        merged = primary[["market_id", "net_pnl", "roc"]].merge(
            strict_same[["market_id", "net_pnl", "roc"]],
            on="market_id",
            how="inner",
            suffixes=("_primary", "_strict_source_same_market"),
        )
        if not merged.empty:
            diff = merged["net_pnl_primary"] - merged["net_pnl_strict_source_same_market"]
            lo, hi = bootstrap_mean_ci(diff.to_numpy(dtype=float), seed_offset=77)
            rows.append(
                {
                    "row_type": "incremental_same_markets",
                    "variant": "strict_rich_short_minus_strict_source_same_markets",
                    "n_markets": int(len(merged)),
                    "mean_net_pnl": float(diff.mean()),
                    "median_net_pnl": float(diff.median()),
                    "net_ci_lo": lo,
                    "net_ci_hi": hi,
                    "incremental_positive": bool(lo > 0),
                }
            )
    return pd.DataFrame(rows)


def update_hub_and_brain(primary_row: dict[str, Any], tail_row: pd.Series | None, incremental_row: pd.Series | None) -> None:
    gate = "FAIL"
    if tail_row is not None and incremental_row is not None:
        tail_pass = bool(float(tail_row.get("net_ci_lo", math.nan)) > 0) if np.isfinite(float(tail_row.get("net_ci_lo", math.nan))) else False
        incr_pass = bool(float(incremental_row.get("net_ci_lo", math.nan)) > 0) if np.isfinite(float(incremental_row.get("net_ci_lo", math.nan))) else False
        gate = "PASS" if tail_pass and incr_pass else "FAIL"

    hub = OD_HUB.read_text()
    if not hub.lstrip().startswith("---"):
        yaml_idx = hub.find("\n---\n")
        if yaml_idx >= 0:
            hub = hub[yaml_idx + 1 :]
    bullet = (
        f"- 2026-06-01 OD v3 PnL/risk deep-dive: **{gate}** on the strict gate. The official capital model is now per-asset concurrent, Polymarket-only. Primary OOS strict-rich/source-filtered far-|z| row: "
        f"n={int(primary_row.get('n_markets', 0))} markets, mean {cents(float(primary_row.get('mean_net_pnl', math.nan)))}, median {cents(float(primary_row.get('median_net_pnl', math.nan)))}, "
        f"ROC mean {pct(float(primary_row.get('mean_roc', math.nan)))}, daily run-rate {fmt_usd(float(primary_row.get('daily_pnl', math.nan)))} on peak concurrent capital {fmt_usd(float(primary_row.get('peak_concurrent_capital', math.nan)))}. "
        "The tail and incremental-over-MM tests are now the gating objections; hedge is demoted to historical footnote. See [[od_strategy_a_v3_pnl_risk_findings]]."
    )
    marker_idx = hub.find("## Current state")
    if marker_idx >= 0:
        next_idx = hub.find("\n## ", marker_idx + 1)
        if next_idx < 0:
            next_idx = len(hub)
        section = hub[marker_idx:next_idx]
        lines = [ln for ln in section.splitlines() if "OD v3 PnL/risk deep-dive" not in ln]
        heading = lines[0]
        rest = "\n".join(lines[1:]).strip()
        new_section = f"{heading}\n\n{bullet}"
        if rest:
            new_section += "\n" + rest
        hub = hub[:marker_idx] + new_section.rstrip() + "\n" + hub[next_idx:]
    else:
        hub = hub.rstrip() + "\n\n## Current state (2026-06-01)\n\n" + bullet + "\n"
    OD_HUB.write_text(hub)

    todo = BRAIN_TODO.read_text()
    od_marker = "## OD"
    line = (
        f"- 2026-06-01: OD v3 PnL/risk deep-dive wrote `notes/options_delta/od_strategy_a_v3_pnl_risk_findings.md`. "
        f"Verdict {gate}: per-asset concurrent capital is official; tail stress and incremental-over-MM decide whether OD remains separate from MM.\n"
    )
    todo = "\n".join(ln for ln in todo.splitlines() if "OD v3 PnL/risk deep-dive wrote" not in ln) + "\n"
    od_idx = todo.find(od_marker)
    if od_idx >= 0:
        line_end = todo.find("\n", od_idx)
        if line_end < 0:
            line_end = len(todo)
        suffix = todo[line_end + 1 :]
        if not suffix.startswith("\n"):
            suffix = "\n" + suffix
        todo = todo[: line_end + 1] + line + suffix
    else:
        todo = todo.rstrip() + "\n\n## OD\n" + line
    BRAIN_TODO.write_text(todo)


def write_note(artifacts: EpisodeArtifacts) -> None:
    summary = artifacts.summary.copy()
    primary = summary[summary["label"].eq("strict_rich_short_per_asset")]
    primary_row = primary.iloc[0].to_dict() if not primary.empty else {}
    global_row = summary[summary["label"].eq("strict_rich_short_global_sensitivity")]
    strict_row = summary[summary["label"].eq("official_strict_source_per_asset")]
    tail = artifacts.tail.copy()
    stress = tail[tail["row_type"].eq("stress_empirical_far_short_itm")]
    observed_tail = tail[tail["row_type"].eq("observed_primary")]
    incr = artifacts.incremental[artifacts.incremental["row_type"].eq("incremental_same_markets")]

    tail_pass = bool((not stress.empty) and float(stress.iloc[0].get("net_ci_lo", math.nan)) > 0)
    incr_pass = bool((not incr.empty) and float(incr.iloc[0].get("net_ci_lo", math.nan)) > 0)
    gate_pass = tail_pass and incr_pass

    capital_rows = []
    for label in [
        "strict_rich_short_per_asset",
        "strict_rich_short_global_sensitivity",
        "official_strict_source_per_asset",
        "bare_lifecycle_per_asset",
    ]:
        sub = summary[summary["label"].eq(label)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        capital_rows.append(
            [
                label,
                str(int(r["n_markets"])),
                str(int(r["n_fills"])),
                cents(float(r["mean_net_pnl"])),
                cents(float(r["median_net_pnl"])),
                ci_text(float(r["net_ci_lo"]), float(r["net_ci_hi"])),
                pct(float(r["mean_roc"])),
                pct(float(r["median_roc"])),
                ci_text(float(r["roc_ci_lo"]), float(r["roc_ci_hi"]), unit="pct"),
                fmt_usd(float(r["daily_pnl"])),
                fmt_usd(float(r["peak_concurrent_capital"])),
            ]
        )

    capture_rows = []
    cap = artifacts.capture.copy()
    if not cap.empty:
        for _, r in cap.iterrows():
            capture_rows.append(
                [
                    str(int(r["target_contracts_per_fill"])),
                    number(float(r["gross_units"]), 1),
                    number(float(r["realistic_units_after_top3_haircut"]), 2),
                    fmt_usd(float(r["realistic_capital"])),
                    fmt_usd(float(r["realistic_expected_ev"])),
                    fmt_usd(float(r["realistic_realized_pnl"])),
                    fmt_usd(float(r["realistic_expected_ev_per_day"])),
                ]
            )

    sizing_rows = []
    for _, r in artifacts.sizing.sort_values(["net_ci_lo", "mean_net_pnl"], ascending=[False, False]).iterrows():
        sizing_rows.append(
            [
                str(r["policy"]),
                str(int(r["n_markets"])),
                cents(float(r["mean_net_pnl"])),
                cents(float(r["median_net_pnl"])),
                ci_text(float(r["net_ci_lo"]), float(r["net_ci_hi"])),
                pct(float(r["mean_roc"])),
                cents(float(r["pnl_std"])),
                cents(float(r["cvar_5"])),
                cents(float(r["worst_episode"])),
                number(float(r["mean_weighted_fills"]), 2),
            ]
        )

    tail_rows = []
    for _, r in tail.iterrows():
        tail_rows.append(
            [
                str(r["row_type"]),
                str(int(r.get("n_markets", 0))) if np.isfinite(float(r.get("n_markets", 0))) else "-",
                cents(float(r.get("mean_net_pnl", math.nan))),
                cents(float(r.get("median_net_pnl", math.nan))),
                cents(float(r.get("worst_episode", math.nan))),
                cents(float(r.get("cvar_5", math.nan))),
                cents(float(r.get("max_drawdown", math.nan))),
                pct(float(r.get("empirical_short_itm_rate", math.nan))) if "empirical_short_itm_rate" in r else "-",
                ci_text(float(r.get("net_ci_lo", math.nan)), float(r.get("net_ci_hi", math.nan))) if np.isfinite(float(r.get("net_ci_lo", math.nan))) else "-",
            ]
        )

    incremental_rows = []
    for _, r in artifacts.incremental.iterrows():
        incremental_rows.append(
            [
                str(r.get("variant", r.get("label", ""))),
                str(int(r.get("n_markets", 0))) if np.isfinite(float(r.get("n_markets", 0))) else "-",
                cents(float(r.get("mean_net_pnl", math.nan))),
                cents(float(r.get("median_net_pnl", math.nan))),
                ci_text(float(r.get("net_ci_lo", math.nan)), float(r.get("net_ci_hi", math.nan))) if np.isfinite(float(r.get("net_ci_lo", math.nan))) else "-",
                pct(float(r.get("mean_roc", math.nan))) if np.isfinite(float(r.get("mean_roc", math.nan))) else "-",
                number(float(r.get("roc_ci_lo_bps", math.nan)), 1) if np.isfinite(float(r.get("roc_ci_lo_bps", math.nan))) else "-",
                number(float(r.get("lower_roc_bps_minus_benchmark", math.nan)), 1) if np.isfinite(float(r.get("lower_roc_bps_minus_benchmark", math.nan))) else "-",
            ]
        )

    headline = "PASS" if gate_pass else "FAIL"
    primary_text = (
        f"Primary per-asset concurrent row: n={int(primary_row.get('n_markets', 0))} markets / {int(primary_row.get('n_fills', 0))} fills, "
        f"mean {cents(float(primary_row.get('mean_net_pnl', math.nan)))}, median {cents(float(primary_row.get('median_net_pnl', math.nan)))}, "
        f"mean ROC {pct(float(primary_row.get('mean_roc', math.nan)))}, daily run-rate {fmt_usd(float(primary_row.get('daily_pnl', math.nan)))} "
        f"on peak concurrent capital {fmt_usd(float(primary_row.get('peak_concurrent_capital', math.nan)))}."
    )
    stress_text = "Tail stress was unavailable."
    if not stress.empty:
        sr = stress.iloc[0]
        stress_text = (
            f"Full-panel far-|z| strict-rich shorts had empirical ITM rate {pct(float(sr['empirical_short_itm_rate']))}; "
            f"the simulated priced-in tail mean is {cents(float(sr['mean_net_pnl']))}, CI {ci_text(float(sr['net_ci_lo']), float(sr['net_ci_hi']))}."
        )
    incr_text = "Incremental-over-MM comparison was unavailable."
    if not incr.empty:
        ir = incr.iloc[0]
        incr_text = (
            f"Same-market strict-rich minus strict-source incremental mean is {cents(float(ir['mean_net_pnl']))}, "
            f"CI {ci_text(float(ir['net_ci_lo']), float(ir['net_ci_hi']))}."
        )

    note = f"""# OD Strategy A v3 PnL/Risk Deep-Dive

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior OD note: [[od_strategy_a_v3_findings]]
> MM benchmark notes: [[mm_deployable_cells_findings]] · [[block_k5_stress_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Headline

OD Strategy A v3 **{headline}** under the new strict gate: survive the priced-in left tail **and** beat the same-market MM/structural baseline with lower-CI > 0.

{primary_text}

{stress_text}

{incr_text}

My read: the per-asset capital framing makes the raw OD row look economically interesting, but the strict pass/fail is still blocked by sample size and by the incremental-over-MM test. The OD richness filter is useful selection, but in this replay it does not yet prove an independent edge beyond the structural/source-filtered lifecycle that MM already highlighted.

## What Changed From v3

This note changes the capital model, not the trade idea. v3 kept the old global one-slot embargo as the official gate, which meant same-time BTC, ETH, and SOL markets competed for one risk slot. This deep-dive adopts **per-asset concurrent** as official: BTC, ETH, and SOL 4h markets can be held at the same time, because they are separate Polymarket markets and separate risk sleeves. The global one-slot row remains below as a conservative sensitivity.

This is Polymarket-only. There is no Binance hedge sweep here. The hedge was already demoted by v3 because it cut little variance in the far-|z| gold-mine bucket and consumed premium. Risk control here is capital sizing, market capture, and diversification across assets.

## Why `n=4` Does Not Mean Four Crypto Windows Worked

`n` is the number of selected **market episodes** after the row's filters, not the number of available crypto 4h windows and not the number of winners. A market episode is one resolved Polymarket market such as one BTC, ETH, or SOL 4h UP/DOWN contract, with all accepted fills inside it aggregated into one PnL observation.

The `n=4` row is the old **global sensitivity**: OOS + far-|z| + strict Chainlink/Binance source filter + rich-short edge >= 1c + one global active episode at a time. That global one-slot rule discards same-time ETH/SOL/BTC opportunities after one of them is selected. Under the new official per-asset concurrent capital model, the same OD filter has `n=7`, because BTC, ETH, and SOL can be active at the same time. The broader bare per-asset far-|z| row has `n=17`, because it does not require strict source plus rich-short richness.

So `n=4` means: four non-overlapping global-time episodes survived a narrow filter. It does **not** mean only four crypto 4h windows existed, nor that the other windows "failed"; most were outside OOS, had no eligible K-PEG fill, were not far-|z| at entry, failed the source filter, lacked a rich-short edge, or were removed by the conservative global embargo.

## Capital Definition

Every fill is converted into a synthetic long claim. A long UP is a long UP claim costing `price`; a short UP is a long DOWN claim costing `1 - price`; a short DOWN is a long UP claim costing `1 - price`. That is the cleanest way to understand shorting binary tokens: selling a token is economically the same as buying its complement.

`capital_at_risk` in the return tables is the conservative **peak complement capital tied up** inside the episode. The parquet also stores `peak_two_sided_risk_capital`, which nets paired UP/DOWN complete sets, and `peak_abs_dollar_delta`, which is the directional OD exposure. ROC uses the conservative complement-capital denominator.

## Task 1 — Capital And Return

| row | markets | fills | mean net | median net | CI | mean ROC | median ROC | ROC CI | $/day | peak concurrent capital |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{markdown_table([], capital_rows).splitlines()[2] if False else chr(10).join('| ' + ' | '.join(row) + ' |' for row in capital_rows)}

Read: `strict_rich_short_per_asset` is the official OD row. The global sensitivity shows the old one-slot assumption. `official_strict_source_per_asset` is the structural/source-filtered lifecycle without the OD richness cut.

The positive cents/episode and negative mean ROC can coexist because the ROC denominator is tiny for near-99c shorts. A loss of 6c on 6c complement capital is roughly -99% ROC, while a large winning episode contributes more cents but a less extreme percentage. That is why the table reports both mean and median.

## Task 2 — Capture / Capacity Proxy

Capacity is a proxy, not a full queue replay. I as-of joined each rich-short fill to `block_a1_features.parquet`, used same-side touch/top-5 depth plus 300s observed trade flow, then applied the K5 incumbent reality that top-3 wallets capture roughly 95% of each market. The `realistic` columns therefore assume only 5% of the observed opportunity is actually accessible to us without winning the incumbent queue.

| target contracts/fill | gross units | realistic units | realistic capital | realistic expected EV | realistic realized PnL | realistic expected EV/day |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join('| ' + ' | '.join(row) + ' |' for row in capture_rows)}

Read: the gross book can look deep, but the incumbent haircut collapses deployable dollars. This is the OD version of the MM capacity lesson: crypto 4h may show high bps, yet practical non-incumbent dollars are small unless we solve queue/capture.

## Task 3 — Sizing Policies

All sizing policies use the same OOS far-|z| strict-rich/source-filtered fill set.

| policy | markets | mean net | median net | CI | mean ROC | PnL std | CVaR 5% | worst | weighted fills |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join('| ' + ' | '.join(row) + ' |' for row in sizing_rows)}

Sizing definitions: `flat_1_contract` buys one synthetic complement per accepted fill. `rv_edge_scaled` sizes roughly proportional to edge versus RV physical-probability fair, capped at 3x. `dollar_delta_cap_50` clips fills once running episode dollar-delta reaches $50. `fractional_kelly_25pct` uses a quarter-Kelly binary-contract proxy from RV physical-probability fair, also capped at 3x.

## Task 4 — Left Tail

| row | markets | mean | median | worst | CVaR 5% | max drawdown | empirical ITM | stress CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join('| ' + ' | '.join(row) + ' |' for row in tail_rows)}

The bad event is simple: we short an apparently overpriced longshot token and that token resolves ITM. The observed OOS set is too small to trust by itself, so the stress row prices disasters with the full-panel empirical ITM rate for far-|z| strict-rich shorts. Diversification helps only if the bad longshots are not all the same macro event; the slot portfolio row is the first check of same-time BTC/ETH/SOL aggregation.

## Task 5 — Incremental Over MM

MM benchmark used here: K5-STRESS crypto_4h structured-non-top3 median `2.4 bps`; aggregate structured-non-top3 `189 bps` with CI lower `21.8 bps`; deployable crypto 4h cells in `mm_deployable_cells_findings` round to about `$0/day` after incumbent capacity.

| variant | markets | mean net | median net | CI | mean ROC | lower ROC bps | lower bps minus MM median |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join('| ' + ' | '.join(row) + ' |' for row in incremental_rows)}

The key line is `strict_rich_short_minus_strict_source_same_markets`. If it is not lower-CI positive, OD has not proven that RV-richness adds EV beyond simply selecting the same source-filtered/structural markets. In that case this should fold back into MM as a quote-selection feature rather than stand alone as a separate OD edge.

## Gate Verdict

Pre-registered pass condition: per-asset concurrent, OOS, net-of-cost, far-|z| family, priced-in left tail survives with lower-CI > 0, and incremental-over-MM lower-CI > 0. This run does **{'pass' if gate_pass else 'not pass'}** that strict gate.

CSV: `data/analysis/csv_outputs/options_delta/od_strategy_a_v3_pnl_risk.csv`

Episode parquet: `data/analysis/od_strategy_a_v3_pnl_risk_episodes.parquet`

Distribution parquet: `data/analysis/od_strategy_a_v3_pnl_risk_distributions.parquet`
"""
    NOTE.write_text(normalize_markdown_wrapping(note))
    update_hub_and_brain(primary_row, stress.iloc[0] if not stress.empty else None, incr.iloc[0] if not incr.empty else None)


def run() -> EpisodeArtifacts:
    fills = add_claim_fields(load_v3_fills(refresh=False))
    primary_mask = selected_fill_mask(fills, PRIMARY_SPEC)
    selected = fills[primary_mask].copy()

    sizing_eps = pd.concat(
        [
            build_weighted_episodes(fills, policy="flat_1_contract", selected_mask=primary_mask),
            build_weighted_episodes(fills, policy="rv_edge_scaled", selected_mask=primary_mask),
            build_weighted_episodes(fills, policy="dollar_delta_cap_50", selected_mask=primary_mask),
            build_weighted_episodes(fills, policy="fractional_kelly_25pct", selected_mask=primary_mask),
        ],
        ignore_index=True,
        sort=False,
    )
    primary_eps = sizing_eps[sizing_eps["policy"].eq("flat_1_contract")].copy()
    capture = build_capture_curve(selected)
    tail = build_tail_table(primary_eps, fills, primary_mask)
    incremental = build_incremental_table(fills)
    summary = incremental[incremental["row_type"].eq("strategy_variant")].copy()
    sizing = summarize_sizing(sizing_eps)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    episodes_out = sizing_eps.copy()
    episodes_out.to_parquet(OUT_EPISODES, index=False)
    pd.concat(
        [
            sizing.assign(table="sizing"),
            tail.assign(table="tail"),
            incremental.assign(table="incremental"),
        ],
        ignore_index=True,
        sort=False,
    ).to_parquet(OUT_DISTRIBUTIONS, index=False)
    capture.to_parquet(OUT_CAPTURE, index=False)

    artifacts = EpisodeArtifacts(
        fills=fills,
        episodes=episodes_out,
        summary=summary,
        sizing=sizing,
        capture=capture,
        tail=tail,
        incremental=incremental,
    )
    write_note(artifacts)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_EPISODES}")
    print(f"wrote {OUT_DISTRIBUTIONS}")
    print(f"wrote {OUT_CAPTURE}")
    print(f"wrote {NOTE}")
    return artifacts


if __name__ == "__main__":
    run()
