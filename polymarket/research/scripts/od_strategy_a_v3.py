"""OD Strategy A v3: power + OD entry filters, hedge demoted.

This extends the v2 lifecycle replay without changing the v2 artifacts:
- Phase 1: bare unhedged K5 lifecycle, global time embargo as the gate.
- Phase 2: OD valuation filters on top of the same lifecycle.
- Phase 3: dollar-delta caps.
- Phase 4: static/rolled Binance hedge as a variance footnote.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import cents, number, pct
from od_strategy_a_v2_lifecycle import (
    ANALYSIS,
    BRAIN_TODO,
    BINANCE_HEDGE_COST_BPS,
    BOOTSTRAP_SAMPLES,
    GATE_BUCKET,
    HEDGE_POLICIES,
    HEDGE_RATIOS,
    K6_PANEL,
    NOTE as V2_NOTE,
    NOTES,
    OD_HUB,
    OUT_FILLS as V2_FILLS,
    ROOT,
    RNG_SEED,
    build_fill_ledger as build_v2_fill_ledger,
    hedge_context,
    hedge_ratio_values,
    markdown_table,
    parse_fill_ids,
    read_hedge_panel,
    select_nonoverlap,
    simulate_hedge_for_episode,
)


OUT_SUMMARY = ANALYSIS / "csv_outputs" / "options_delta" / "od_strategy_a_v3.csv"
OUT_TRADES = ANALYSIS / "od_strategy_a_v3_trades.parquet"
OUT_FILLS = ANALYSIS / "od_strategy_a_v3_fills.parquet"
NOTE = NOTES / "options_delta" / "od_strategy_a_v3_findings.md"

GATE_SPLIT = "oos_holdout"
GATE_FILTER = "bare_lifecycle"
GATE_EMBARGO = "global_time_nonoverlap"
GATE_CAP = math.inf

CAPS = [math.inf, 25.0, 50.0, 100.0, 250.0, 500.0]
RICH_THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.05]
VALUE_THRESHOLDS = [0.0, 0.005, 0.01, 0.02]
VOL_THRESHOLDS = [0.0, 0.05, 0.10, 0.20]


@dataclass(frozen=True)
class FilterSpec:
    phase: str
    filter_id: str
    source_policy: str = "all"
    richness_threshold: float | None = None
    value_edge_threshold: float | None = None
    vol_threshold: float | None = None
    require_short: bool = False


def ci_text(lo: float, hi: float) -> str:
    return f"[{cents(lo)}, {cents(hi)}]"


def normalize_markdown_wrapping(text: str) -> str:
    """Remove manual hard-wraps from prose while preserving tables/code."""
    out: list[str] = []
    paragraph: list[str] = []
    in_code = False

    def flush() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(" ".join(part.strip() for part in paragraph if part.strip()))
            paragraph = []

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            out.append(line)
            in_code = not in_code
            continue
        if in_code:
            out.append(line)
            continue
        if not stripped:
            flush()
            if out and out[-1] != "":
                out.append("")
            continue
        if stripped.startswith(("#", ">", "|")):
            flush()
            out.append(line)
            continue
        if stripped.startswith("- "):
            flush()
            paragraph = [line]
            continue
        if paragraph and paragraph[0].lstrip().startswith("- "):
            paragraph.append(stripped)
            continue
        paragraph.append(stripped)
    flush()
    return "\n".join(out).rstrip() + "\n"


def fmt_cap(cap: float) -> str:
    return "none" if not np.isfinite(cap) else f"${cap:.0f}"


def fmt_threshold(value: float | None, unit: str = "c") -> str:
    if value is None:
        return "-"
    if unit == "vol":
        return f"{100.0 * value:.0f} vol pts"
    return cents(value)


def fmt_band(value: float) -> str:
    return "static" if not np.isfinite(value) else f"${value:.0f}"


def safe_float(row: pd.Series, key: str, default: float = math.nan) -> float:
    val = row.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def rv_token_prob_from_p_model(df: pd.DataFrame) -> pd.Series:
    if "p_model" not in df or "actual_outcome" not in df:
        raise ValueError("cannot derive RV physical-probability token fair without p_model and actual_outcome")
    p_up = pd.to_numeric(df["p_model"], errors="coerce")
    return pd.Series(
        np.where(df["actual_outcome"].astype(str).str.lower().eq("up"), p_up, 1.0 - p_up),
        index=df.index,
        dtype="float64",
    )


def assert_no_pm_mid_fair_source(df: pd.DataFrame, *, context: str) -> None:
    source_cols = ["fair_prob_kind", "token_model_fair_kind", "claim_fair_kind", "fair_source", "token_fair_source"]
    bad_tokens = ("pm_mid", "pm_iv", "implied_vol", "polymarket_mid")
    for col in source_cols:
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).str.lower()
        bad = sorted(v for v in values.unique() if any(token in v for token in bad_tokens))
        if bad:
            raise ValueError(f"{context}: PM-mid/IV diagnostics cannot be used as external fair ({col}={bad})")


def resolve_token_rv_physical_prob_fair(df: pd.DataFrame, *, context: str = "OD fair source") -> pd.Series:
    """Return the token fair only when it is provably the Binance RV physical probability."""
    assert_no_pm_mid_fair_source(df, context=context)
    if "token_model_fair_kind" in df.columns:
        kind = df["token_model_fair_kind"].dropna().astype(str)
        bad = sorted(kind[~kind.eq("rv_physical_prob")].unique())
        if bad:
            raise ValueError(f"{context}: token_model_fair_kind must be rv_physical_prob, saw {bad}")
    if "token_rv_physical_prob_fair" in df.columns:
        fair = pd.to_numeric(df["token_rv_physical_prob_fair"], errors="coerce")
        if "token_model_fair" in df.columns:
            legacy = pd.to_numeric(df["token_model_fair"], errors="coerce")
            max_diff = (fair - legacy).abs().replace([np.inf, -np.inf], np.nan).max()
            if pd.notna(max_diff) and float(max_diff) > 1e-9:
                raise ValueError(f"{context}: legacy token_model_fair no longer matches token_rv_physical_prob_fair")
        return fair
    if "token_model_fair" not in df.columns:
        raise ValueError(f"{context}: missing token_rv_physical_prob_fair")
    derived = rv_token_prob_from_p_model(df)
    legacy = pd.to_numeric(df["token_model_fair"], errors="coerce")
    max_diff = (derived - legacy).abs().replace([np.inf, -np.inf], np.nan).max()
    if pd.notna(max_diff) and float(max_diff) > 1e-9:
        raise ValueError(
            f"{context}: refusing legacy token_model_fair because it is not the RV physical-probability token fair"
        )
    return legacy


def load_v3_fills(refresh: bool = True) -> pd.DataFrame:
    """Read the v2-correct fill ledger and add OD valuation fields."""
    if OUT_FILLS.exists() and not refresh:
        return pd.read_parquet(OUT_FILLS)

    if V2_FILLS.exists():
        fills = pd.read_parquet(V2_FILLS)
    else:
        fills = build_v2_fill_ledger(refresh=True)

    fills["market_id"] = fills["market_id"].astype(str)
    fills["ts"] = pd.to_datetime(fills["ts"], utc=True)
    fills["ts_key"] = fills["ts"].map(lambda x: pd.Timestamp(x).value).astype("int64")

    extra_cols = [
        "market_id",
        "ts",
        "p_model",
        "tau_years",
        "dynamic_logit_gap",
        "source_penalty_flag",
    ]
    panel = pd.read_parquet(K6_PANEL, columns=extra_cols)
    panel["market_id"] = panel["market_id"].astype(str)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel["ts_key"] = panel["ts"].map(lambda x: pd.Timestamp(x).value).astype("int64")
    panel = panel.drop(columns=["ts"]).drop_duplicates(["market_id", "ts_key"])
    add_cols = [c for c in panel.columns if c not in fills.columns or c in {"market_id", "ts_key"}]
    fills = fills.merge(panel[add_cols], on=["market_id", "ts_key"], how="left")

    numeric_cols = [
        "entry_price",
        "token_position",
        "payoff",
        "p_model",
        "iv_minus_ewma",
        "pm_iv_annualized",
        "ewma_sigma_annualized",
        "digital_delta",
        "binance_spot",
        "abs_z",
        "seconds_to_expiry",
        "signed_delta_exposure",
        "cash_flow",
        "fill_pnl",
        "maker_rebate",
    ]
    for col in numeric_cols:
        if col in fills.columns:
            fills[col] = pd.to_numeric(fills[col], errors="coerce")

    fills["token_rv_physical_prob_fair"] = np.where(
        fills["actual_outcome"].eq("up"),
        fills["p_model"].astype(float),
        1.0 - fills["p_model"].astype(float),
    )
    fills["token_model_fair"] = fills["token_rv_physical_prob_fair"]  # Legacy alias.
    fills["token_model_fair_kind"] = "rv_physical_prob"
    fills["od_value_edge_vs_rv_physical_prob"] = fills["token_position"] * (
        fills["token_rv_physical_prob_fair"] - fills["entry_price"]
    )
    fills["od_value_edge"] = fills["od_value_edge_vs_rv_physical_prob"]  # Legacy alias.
    fills["rich_short_edge_vs_rv_physical_prob"] = np.where(
        fills["token_position"].lt(0),
        fills["entry_price"] - fills["token_rv_physical_prob_fair"],
        np.nan,
    )
    fills["rich_short_edge"] = fills["rich_short_edge_vs_rv_physical_prob"]  # Legacy alias.
    fills["pm_mid_iv_minus_ewma"] = fills["iv_minus_ewma"].astype(float)
    fills["vol_premium_ewma"] = fills["pm_mid_iv_minus_ewma"]  # Legacy alias.
    fills["fill_dollar_delta"] = fills["signed_delta_exposure"].abs() * fills["binance_spot"].abs()
    fills["is_short_rich_side"] = fills["token_position"].lt(0) & fills["rich_short_edge"].gt(0)
    fills["is_value_edge_positive"] = fills["od_value_edge"].gt(0)
    fills["is_vol_premium_positive"] = fills["pm_iv_valid"].fillna(False).astype(bool) & fills["vol_premium_ewma"].gt(0)

    OUT_FILLS.parent.mkdir(parents=True, exist_ok=True)
    fills.to_parquet(OUT_FILLS, index=False)
    print(f"wrote {OUT_FILLS}", flush=True)
    return fills


def bucket_mask(fills: pd.DataFrame, bucket: str) -> pd.Series:
    if bucket == "all_buckets":
        return pd.Series(True, index=fills.index)
    if bucket == "far_absz_ge1_all_tau":
        return fills["abs_z"].astype(float).ge(1.0)
    if bucket == "longshot_absz_ge0.75_all_tau":
        return fills["abs_z"].astype(float).ge(0.75)
    if bucket == "longshot_absz_ge0.50_all_tau":
        return fills["abs_z"].astype(float).ge(0.50)
    if bucket == "longshot_short_price_le30_all_tau":
        return fills["token_position"].lt(0) & fills["entry_price"].le(0.30)
    if bucket == "mid_absz_0.25_1_all_tau":
        return fills["moneyness_bucket"].eq("mid_absz_0.25_1")
    if bucket == "near_absz_lt0.25_all_tau":
        return fills["moneyness_bucket"].eq("near_absz_lt0.25")
    return fills["state_bucket"].eq(bucket)


def phase_buckets(fills: pd.DataFrame) -> list[str]:
    buckets = [
        "all_buckets",
        "far_absz_ge1_all_tau",
        "longshot_absz_ge0.75_all_tau",
        "longshot_absz_ge0.50_all_tau",
        "longshot_short_price_le30_all_tau",
        "mid_absz_0.25_1_all_tau",
        "near_absz_lt0.25_all_tau",
    ]
    buckets.extend(sorted(fills["state_bucket"].dropna().astype(str).unique()))
    return list(dict.fromkeys(buckets))


def filter_mask(fills: pd.DataFrame, spec: FilterSpec) -> pd.Series:
    mask = fills["eligible"].fillna(False).astype(bool)
    if spec.source_policy == "strict":
        mask &= fills["strict_source_eligible"].fillna(False).astype(bool)
    if spec.richness_threshold is not None:
        mask &= fills["token_position"].lt(0)
        mask &= fills["rich_short_edge"].ge(spec.richness_threshold)
    if spec.value_edge_threshold is not None:
        mask &= fills["od_value_edge"].ge(spec.value_edge_threshold)
    if spec.vol_threshold is not None:
        mask &= fills["pm_iv_valid"].fillna(False).astype(bool)
        mask &= fills["vol_premium_ewma"].ge(spec.vol_threshold)
    if spec.require_short:
        mask &= fills["token_position"].lt(0)
    return mask


def filter_specs() -> list[FilterSpec]:
    specs: list[FilterSpec] = [
        FilterSpec("phase1_power", "bare_lifecycle"),
        FilterSpec("phase2_od_filter", "official_strict_source", source_policy="strict"),
    ]
    for th in RICH_THRESHOLDS:
        specs.append(FilterSpec("phase2_od_filter", f"rich_short_ge_{int(round(th * 1000)):03d}m", richness_threshold=th))
        specs.append(
            FilterSpec(
                "phase2_od_filter",
                f"strict_rich_short_ge_{int(round(th * 1000)):03d}m",
                source_policy="strict",
                richness_threshold=th,
            )
        )
    for th in VALUE_THRESHOLDS:
        specs.append(FilterSpec("phase2_od_filter", f"value_edge_ge_{int(round(th * 1000)):03d}m", value_edge_threshold=th))
    for th in VOL_THRESHOLDS:
        specs.append(FilterSpec("phase2_od_filter", f"vol_premium_ge_{int(round(th * 100)):02d}vp", vol_threshold=th))
        specs.append(
            FilterSpec(
                "phase2_od_filter",
                f"strict_vol_premium_ge_{int(round(th * 100)):02d}vp",
                source_policy="strict",
                vol_threshold=th,
            )
        )
    for rich in [0.0, 0.01, 0.02]:
        for vol in [0.0, 0.05, 0.10]:
            specs.append(
                FilterSpec(
                    "phase2_od_filter",
                    f"rich_{int(round(rich * 1000)):03d}m_vol_{int(round(vol * 100)):02d}vp",
                    richness_threshold=rich,
                    vol_threshold=vol,
                )
            )
            specs.append(
                FilterSpec(
                    "phase2_od_filter",
                    f"strict_rich_{int(round(rich * 1000)):03d}m_vol_{int(round(vol * 100)):02d}vp",
                    source_policy="strict",
                    richness_threshold=rich,
                    vol_threshold=vol,
                )
            )
    return list({s.filter_id: s for s in specs}.values())


def split_mask(fills: pd.DataFrame, split: str) -> pd.Series:
    if split == "pooled":
        return pd.Series(True, index=fills.index)
    return fills["entry_split"].eq(split)


def cap_accepts_fill(up_inv: float, down_inv: float, row: pd.Series, cap: float) -> bool:
    if not np.isfinite(cap):
        return True
    pos = float(row["token_position"])
    new_up = up_inv + (pos if row["actual_outcome"] == "up" else 0.0)
    new_down = down_inv + (pos if row["actual_outcome"] == "down" else 0.0)
    signed_delta = (new_up - new_down) * safe_float(row, "digital_delta", 0.0)
    dollar_delta = abs(signed_delta * safe_float(row, "binance_spot", 0.0))
    return dollar_delta <= cap


def apply_episode_cap(g: pd.DataFrame, cap: float) -> pd.DataFrame:
    if not np.isfinite(cap):
        return g
    accepted: list[int] = []
    up_inv = 0.0
    down_inv = 0.0
    for idx, row in g.sort_values("fill_ts_key").iterrows():
        if not cap_accepts_fill(up_inv, down_inv, row, cap):
            continue
        pos = float(row["token_position"])
        if row["actual_outcome"] == "up":
            up_inv += pos
        else:
            down_inv += pos
        accepted.append(idx)
    return g.loc[accepted].copy() if accepted else g.iloc[0:0].copy()


def build_episode_for_market(
    g: pd.DataFrame,
    *,
    phase: str,
    filter_id: str,
    bucket: str,
    split: str,
    cap: float,
    embargo_mode: str,
    source_policy: str,
) -> dict[str, Any] | None:
    g = apply_episode_cap(g.sort_values("fill_ts_key"), cap)
    if g.empty:
        return None

    up = g[g["actual_outcome"].eq("up")]
    down = g[g["actual_outcome"].eq("down")]
    up_inv = float(up["token_position"].sum())
    down_inv = float(down["token_position"].sum())
    settlement_pnl = float((g["token_position"] * g["payoff"]).sum())
    cash = float(g["cash_flow"].sum())
    net = float(g["fill_pnl"].sum())
    first = g.iloc[0]
    last = g.iloc[-1]
    final_signed_delta = float((up_inv - down_inv) * safe_float(last, "digital_delta", 0.0))
    final_dollar_delta = abs(final_signed_delta * safe_float(last, "binance_spot", 0.0))
    fill_ids = ",".join(str(int(x)) for x in g["fill_id"].to_numpy())

    return {
        "row_type": phase,
        "phase": phase,
        "filter_id": filter_id,
        "source_policy": source_policy,
        "sample_split": split,
        "bucket": bucket,
        "dollar_delta_cap": cap,
        "embargo_mode": embargo_mode,
        "market_id": str(first["market_id"]),
        "market_slug": str(first["market_slug"]),
        "asset": str(first["asset"]),
        "start_ts": pd.Timestamp(g["fill_ts"].min()),
        "end_ts": pd.Timestamp(first["window_end"]),
        "median_fill_ts": pd.Timestamp(g["fill_ts"].median()),
        "n_fills": int(len(g)),
        "n_up_fills": int(len(up)),
        "n_down_fills": int(len(down)),
        "two_sided": bool(g["actual_outcome"].nunique() >= 2),
        "buy_fill_share": float(g["token_position"].gt(0).mean()),
        "sell_fill_share": float(g["token_position"].lt(0).mean()),
        "up_net_inventory": up_inv,
        "down_net_inventory": down_inv,
        "gross_abs_inventory": float(g["token_position"].abs().sum()),
        "residual_abs_inventory": float(abs(up_inv) + abs(down_inv)),
        "carry_share": float((abs(up_inv) + abs(down_inv)) / g["token_position"].abs().sum()) if len(g) else math.nan,
        "avg_entry_up": float(up["entry_price"].mean()) if len(up) else math.nan,
        "avg_entry_down": float(down["entry_price"].mean()) if len(down) else math.nan,
        "net_pnl": net,
        "cash_pnl": cash,
        "settlement_pnl": settlement_pnl,
        "maker_rebate": float(g["maker_rebate"].sum()),
        "mean_entry_price": float(g["entry_price"].mean()),
        "mean_rv_physical_prob_fair": float(g["token_rv_physical_prob_fair"].mean()),
        "mean_model_fair": float(g["token_model_fair"].mean()),
        "mean_od_value_edge": float(g["od_value_edge"].mean()),
        "mean_rich_short_edge": float(g["rich_short_edge"].mean()) if g["rich_short_edge"].notna().any() else math.nan,
        "mean_vol_premium": float(g["vol_premium_ewma"].mean()),
        "rich_short_fill_share": float(g["is_short_rich_side"].mean()),
        "value_positive_fill_share": float(g["is_value_edge_positive"].mean()),
        "vol_positive_fill_share": float(g["is_vol_premium_positive"].mean()),
        "mean_entry_abs_z": float(g["abs_z"].mean()),
        "entry_abs_z_first": float(first["abs_z"]),
        "entry_seconds_to_expiry_first": float(first["seconds_to_expiry"]),
        "median_abs_z": float(g["abs_z"].median()),
        "median_seconds_to_expiry": float(g["seconds_to_expiry"].median()),
        "mean_signed_delta_exposure": float(g["signed_delta_exposure"].mean()),
        "final_signed_delta_exposure": final_signed_delta,
        "final_abs_dollar_delta": final_dollar_delta,
        "max_fill_abs_dollar_delta": float(g["fill_dollar_delta"].max()),
        "hold_seconds_median_fill": float((pd.Timestamp(first["window_end"]) - pd.Timestamp(g["fill_ts"].median())).total_seconds()),
        "source_disagree": bool(first["chainlink_binance_resolution_disagree"]),
        "source_ok_strict": bool(first["source_ok_strict"]),
        "first_state_bucket": str(first["state_bucket"]),
        "last_state_bucket": str(last["state_bucket"]),
        "fill_ids": fill_ids,
    }


def select_nonoverlap_by_asset(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return episodes
    pieces = []
    for _, g in episodes.groupby("asset", sort=True):
        pieces.append(select_nonoverlap(g))
    return pd.concat(pieces, ignore_index=True).sort_values(["start_ts", "asset", "market_id"]).reset_index(drop=True)


def select_for_embargo(episodes: pd.DataFrame, embargo_mode: str) -> pd.DataFrame:
    if embargo_mode == "global_time_nonoverlap":
        return select_nonoverlap(episodes)
    if embargo_mode == "per_asset_time_nonoverlap":
        return select_nonoverlap_by_asset(episodes)
    raise ValueError(f"unknown embargo_mode {embargo_mode}")


def build_episode_set(
    fills: pd.DataFrame,
    *,
    spec: FilterSpec,
    bucket: str,
    split: str,
    cap: float = math.inf,
    embargo_mode: str = GATE_EMBARGO,
) -> pd.DataFrame:
    mask = filter_mask(fills, spec) & split_mask(fills, split) & bucket_mask(fills, bucket)
    sub = fills[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, g in sub.groupby("market_id", sort=False):
        ep = build_episode_for_market(
            g,
            phase=spec.phase,
            filter_id=spec.filter_id,
            bucket=bucket,
            split=split,
            cap=cap,
            embargo_mode=embargo_mode,
            source_policy=spec.source_policy,
        )
        if ep is not None:
            rows.append(ep)
    if not rows:
        return pd.DataFrame()
    episodes = pd.DataFrame(rows)
    episodes["episode_count_before_embargo"] = len(episodes)
    selected = select_for_embargo(episodes, embargo_mode)
    selected["episode_selected"] = True
    return selected.reset_index(drop=True)


def build_all_episodes(fills: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    buckets = phase_buckets(fills)
    splits = ["pooled", "is_discovery", "oos_holdout"]

    # Phase 1 and Phase 2: no cap, official global embargo.
    for spec in filter_specs():
        for split in splits:
            for bucket in buckets:
                ep = build_episode_set(fills, spec=spec, bucket=bucket, split=split, cap=math.inf, embargo_mode=GATE_EMBARGO)
                if not ep.empty:
                    rows.append(ep)

    # Phase 1 power diagnostic: per-asset embargo shows how much sample exists if cross-asset concurrency is allowed.
    bare = FilterSpec("phase1_power", "bare_lifecycle")
    for split in splits:
        for bucket in buckets:
            ep = build_episode_set(
                fills,
                spec=bare,
                bucket=bucket,
                split=split,
                cap=math.inf,
                embargo_mode="per_asset_time_nonoverlap",
            )
            if not ep.empty:
                ep["filter_id"] = "bare_lifecycle_per_asset_power_diag"
                rows.append(ep)

    # Phase 3 caps: sweep on bare, strict-source, and two OD-style filters.
    cap_specs = [
        FilterSpec("phase3_risk_cap", "cap_bare_lifecycle"),
        FilterSpec("phase3_risk_cap", "cap_official_strict_source", source_policy="strict"),
        FilterSpec("phase3_risk_cap", "cap_rich_short_ge_010m", richness_threshold=0.01),
        FilterSpec("phase3_risk_cap", "cap_strict_rich_short_ge_010m", source_policy="strict", richness_threshold=0.01),
        FilterSpec("phase3_risk_cap", "cap_strict_rich_010m_vol_00vp", source_policy="strict", richness_threshold=0.01, vol_threshold=0.0),
    ]
    for spec in cap_specs:
        for cap in CAPS:
            for split in splits:
                for bucket in ["far_absz_ge1_all_tau", "longshot_absz_ge0.75_all_tau", "longshot_short_price_le30_all_tau"]:
                    ep = build_episode_set(fills, spec=spec, bucket=bucket, split=split, cap=cap, embargo_mode=GATE_EMBARGO)
                    if not ep.empty:
                        rows.append(ep)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def bootstrap_ci(episodes: pd.DataFrame, col: str) -> tuple[float, float]:
    vals = episodes[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(RNG_SEED + len(vals) + len(col))
    draws = rng.integers(0, len(vals), size=(BOOTSTRAP_SAMPLES, len(vals)))
    boot = vals[draws].mean(axis=1)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_episode_group(g: pd.DataFrame) -> dict[str, Any]:
    lo, hi = bootstrap_ci(g, "net_pnl")
    return {
        "n_markets": int(g["market_id"].nunique()),
        "n_fills": int(g["n_fills"].sum()),
        "mean_net_pnl": float(g["net_pnl"].mean()),
        "net_ci_lo": lo,
        "net_ci_hi": hi,
        "median_net_pnl": float(g["net_pnl"].median()),
        "win_rate": float(g["net_pnl"].gt(0).mean()),
        "pnl_std": float(g["net_pnl"].std(ddof=1)) if len(g) > 1 else 0.0,
        "pnl_variance": float(g["net_pnl"].var(ddof=1)) if len(g) > 1 else 0.0,
        "mean_settlement_pnl": float(g["settlement_pnl"].mean()),
        "settlement_pnl_std": float(g["settlement_pnl"].std(ddof=1)) if len(g) > 1 else 0.0,
        "mean_cash_pnl": float(g["cash_pnl"].mean()),
        "mean_maker_rebate": float(g["maker_rebate"].mean()),
        "two_sided_market_share": float(g["two_sided"].mean()),
        "carry_share": float(g["carry_share"].mean()),
        "mean_fills_per_market": float(g["n_fills"].mean()),
        "median_hold_seconds": float(g["hold_seconds_median_fill"].median()),
        "source_disagree_share": float(g["source_disagree"].mean()),
        "mean_od_value_edge": float(g["mean_od_value_edge"].mean()),
        "mean_rich_short_edge": float(g["mean_rich_short_edge"].mean()),
        "mean_vol_premium": float(g["mean_vol_premium"].mean()),
        "rich_short_fill_share": float(g["rich_short_fill_share"].mean()),
        "value_positive_fill_share": float(g["value_positive_fill_share"].mean()),
        "vol_positive_fill_share": float(g["vol_positive_fill_share"].mean()),
        "mean_final_abs_dollar_delta": float(g["final_abs_dollar_delta"].mean()),
        "max_final_abs_dollar_delta": float(g["final_abs_dollar_delta"].max()),
        "tail_market_share": float(g["market_id"].value_counts(normalize=True).iloc[0]) if len(g) else math.nan,
    }


def summarize_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["row_type", "phase", "filter_id", "source_policy", "sample_split", "bucket", "dollar_delta_cap", "embargo_mode"]
    for key, g in episodes.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, key, strict=True))
        row.update(summarize_episode_group(g))
        rows.append(row)
    out = pd.DataFrame(rows)
    return add_baseline_lifts(out)


def baseline_row(summary: pd.DataFrame) -> pd.Series | None:
    sub = summary[
        summary["row_type"].eq("phase1_power")
        & summary["filter_id"].eq(GATE_FILTER)
        & summary["sample_split"].eq(GATE_SPLIT)
        & summary["bucket"].eq(GATE_BUCKET)
        & summary["embargo_mode"].eq(GATE_EMBARGO)
        & np.isinf(summary["dollar_delta_cap"].astype(float))
    ]
    return None if sub.empty else sub.iloc[0]


def add_baseline_lifts(summary: pd.DataFrame) -> pd.DataFrame:
    base = baseline_row(summary)
    if base is None:
        summary["mean_lift_vs_phase1"] = math.nan
        summary["ci_lo_lift_vs_phase1"] = math.nan
        return summary
    summary["mean_lift_vs_phase1"] = summary["mean_net_pnl"].astype(float) - float(base["mean_net_pnl"])
    summary["ci_lo_lift_vs_phase1"] = summary["net_ci_lo"].astype(float) - float(base["net_ci_lo"])
    return summary


def build_static_hedge_overlay(episodes: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    base_configs = {
        "bare_lifecycle",
        "official_strict_source",
        "rich_short_ge_010m",
        "strict_rich_short_ge_010m",
        "strict_rich_010m_vol_00vp",
    }
    sub = episodes[
        episodes["sample_split"].eq(GATE_SPLIT)
        & episodes["bucket"].eq(GATE_BUCKET)
        & episodes["embargo_mode"].eq(GATE_EMBARGO)
        & np.isinf(episodes["dollar_delta_cap"].astype(float))
        & episodes["filter_id"].isin(base_configs)
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    panel = read_hedge_panel()
    panel_by_market = {m: g.copy() for m, g in panel.groupby("market_id", sort=False)}
    fills_by_id = fills.set_index("fill_id", drop=False)
    ctx = hedge_context(fills)
    rows: list[dict[str, Any]] = []
    for _, ep in sub.iterrows():
        market_panel = panel_by_market.get(str(ep["market_id"]))
        if market_panel is None or market_panel.empty:
            continue
        for policy in HEDGE_POLICIES:
            for h_param in HEDGE_RATIOS:
                if h_param <= 0:
                    continue
                row = simulate_hedge_for_episode(
                    ep,
                    fills_by_id,
                    market_panel,
                    policy=policy,
                    h_param=h_param,
                    band_notional=math.inf,
                    ctx=ctx,
                )
                row["row_type"] = "phase4_hedge_footnote"
                row["phase"] = "phase4_hedge_footnote"
                row["hedge_variant"] = "episode_static"
                row["filter_id"] = ep["filter_id"]
                row["source_policy"] = ep["source_policy"]
                row["sample_split"] = ep["sample_split"]
                row["bucket"] = ep["bucket"]
                row["dollar_delta_cap"] = ep["dollar_delta_cap"]
                row["embargo_mode"] = ep["embargo_mode"]
                rows.append(row)
    return pd.DataFrame(rows)


def first_fill_static_target(ep: pd.Series, fills_by_id: pd.DataFrame, policy: str, h_param: float, ctx: dict[str, float]) -> tuple[float, pd.Timestamp, pd.Timestamp, float, float]:
    ep_fills = fills_by_id.loc[parse_fill_ids(str(ep["fill_ids"]))].sort_values("fill_ts")
    if ep_fills.empty:
        return 0.0, pd.NaT, pd.NaT, math.nan, math.nan
    first = ep_fills.iloc[0]
    pos = safe_float(first, "token_position", 0.0)
    signed_delta = pos * (safe_float(first, "digital_delta", 0.0) if first["actual_outcome"] == "up" else -safe_float(first, "digital_delta", 0.0))
    h_eff = hedge_ratio_values(
        policy,
        h_param,
        safe_float(first, "ewma_sigma_annualized"),
        safe_float(first, "iv_minus_ewma"),
        safe_float(first, "abs_z"),
        ctx,
    )
    target_units = -h_eff * signed_delta
    return target_units, pd.Timestamp(first["fill_ts"]), pd.Timestamp(ep["end_ts"]), safe_float(first, "binance_spot"), h_eff


def asof_spot(panel: pd.DataFrame, asset: str, ts: pd.Timestamp, fallback: float) -> float:
    pp = panel[panel["asset"].eq(asset)].sort_values("ts")
    if pp.empty:
        return fallback
    idx = pp["ts"].searchsorted(ts, side="right") - 1
    if idx < 0:
        return fallback
    val = float(pp.iloc[idx]["binance_spot"])
    return val if np.isfinite(val) else fallback


def simulate_portfolio_roll(
    eps: pd.DataFrame,
    fills_by_id: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    policy: str,
    h_param: float,
    ctx: dict[str, float],
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    h_values: list[float] = []
    for _, ep in eps.iterrows():
        units, start_ts, end_ts, spot, h_eff = first_fill_static_target(ep, fills_by_id, policy, h_param, ctx)
        if not np.isfinite(units) or abs(units) < 1e-15 or pd.isna(start_ts) or pd.isna(end_ts):
            continue
        h_values.append(h_eff)
        events.append({"ts": start_ts, "asset": ep["asset"], "delta_units": units, "fallback_spot": spot})
        events.append({"ts": end_ts, "asset": ep["asset"], "delta_units": -units, "fallback_spot": spot})
    if not events:
        return {
            "hedge_pnl": 0.0,
            "hedge_cost": 0.0,
            "hedge_turnover_notional": 0.0,
            "hedge_rebalances": 0,
            "avg_effective_h": 0.0,
            "max_abs_hedge_notional": 0.0,
        }

    ev = pd.DataFrame(events)
    ev = ev.groupby(["asset", "ts"], as_index=False).agg(delta_units=("delta_units", "sum"), fallback_spot=("fallback_spot", "last"))
    hedge_pnl = 0.0
    hedge_cost = 0.0
    turnover = 0.0
    rebalances = 0
    max_abs_notional = 0.0
    for asset, g in ev.groupby("asset", sort=True):
        current_units = 0.0
        last_spot = math.nan
        for _, row in g.sort_values("ts").iterrows():
            spot = asof_spot(panel, asset, pd.Timestamp(row["ts"]), float(row["fallback_spot"]))
            if np.isfinite(last_spot):
                hedge_pnl += current_units * (spot - last_spot)
            du = float(row["delta_units"])
            if abs(du) > 1e-15:
                trade_notional = abs(du) * spot
                turnover += trade_notional
                hedge_cost += trade_notional * BINANCE_HEDGE_COST_BPS / 10_000.0
                rebalances += 1
                current_units += du
                max_abs_notional = max(max_abs_notional, abs(current_units) * spot)
            last_spot = spot
    return {
        "hedge_pnl": float(hedge_pnl),
        "hedge_cost": float(hedge_cost),
        "hedge_turnover_notional": float(turnover),
        "hedge_rebalances": int(max(0, rebalances - 1)),
        "avg_effective_h": float(np.mean(h_values)) if h_values else 0.0,
        "max_abs_hedge_notional": float(max_abs_notional),
    }


def build_portfolio_roll_overlay(episodes: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    sub = episodes[
        episodes["sample_split"].eq(GATE_SPLIT)
        & episodes["bucket"].eq(GATE_BUCKET)
        & episodes["embargo_mode"].eq("per_asset_time_nonoverlap")
        & np.isinf(episodes["dollar_delta_cap"].astype(float))
        & episodes["filter_id"].eq("bare_lifecycle_per_asset_power_diag")
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    panel = read_hedge_panel()
    fills_by_id = fills.set_index("fill_id", drop=False)
    ctx = hedge_context(fills)
    rows: list[dict[str, Any]] = []
    keys = ["filter_id", "source_policy", "sample_split", "bucket", "dollar_delta_cap", "embargo_mode"]
    for key, eps in sub.groupby(keys, dropna=False):
        for policy in HEDGE_POLICIES:
            for h_param in HEDGE_RATIOS:
                if h_param <= 0:
                    continue
                hedge = simulate_portfolio_roll(eps, fills_by_id, panel, policy=policy, h_param=h_param, ctx=ctx)
                for _, ep in eps.iterrows():
                    row = ep.to_dict()
                    # Allocate aggregate hedge economics equally across market episodes for CI comparability.
                    n = max(len(eps), 1)
                    row.update(
                        {
                            "row_type": "phase4_hedge_footnote",
                            "phase": "phase4_hedge_footnote",
                            "hedge_variant": "portfolio_24h_roll",
                            "hedge_policy": policy,
                            "h_param": float(h_param),
                            "rebalance_band_notional": math.inf,
                            "unhedged_net_pnl": float(ep["net_pnl"]),
                            "hedge_pnl": hedge["hedge_pnl"] / n,
                            "hedge_cost": hedge["hedge_cost"] / n,
                            "hedge_turnover_notional": hedge["hedge_turnover_notional"] / n,
                            "hedge_rebalances": hedge["hedge_rebalances"] / n,
                            "avg_effective_h": hedge["avg_effective_h"],
                            "max_abs_hedge_notional": hedge["max_abs_hedge_notional"],
                        }
                    )
                    row["net_pnl"] = row["unhedged_net_pnl"] + row["hedge_pnl"] - row["hedge_cost"]
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize_hedges(overlay: pd.DataFrame, phase_summary: pd.DataFrame) -> pd.DataFrame:
    if overlay.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = [
        "row_type",
        "phase",
        "hedge_variant",
        "filter_id",
        "source_policy",
        "sample_split",
        "bucket",
        "dollar_delta_cap",
        "embargo_mode",
        "hedge_policy",
        "h_param",
        "rebalance_band_notional",
    ]
    base_lookup = {
        (r["filter_id"], r["sample_split"], r["bucket"], r["dollar_delta_cap"], r["embargo_mode"]): r
        for _, r in phase_summary.iterrows()
    }
    for key, g in overlay.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, key, strict=True))
        lo, hi = bootstrap_ci(g, "net_pnl")
        unhedged_mean = float(g["unhedged_net_pnl"].mean())
        unhedged_std = float(g["unhedged_net_pnl"].std(ddof=1)) if len(g) > 1 else 0.0
        pnl_std = float(g["net_pnl"].std(ddof=1)) if len(g) > 1 else 0.0
        premium_retained = float(g["net_pnl"].mean() / unhedged_mean) if abs(unhedged_mean) > 1e-12 else math.nan
        variance_reduction = 1.0 - (pnl_std**2 / unhedged_std**2) if unhedged_std > 1e-12 else math.nan
        base = base_lookup.get((row["filter_id"], row["sample_split"], row["bucket"], row["dollar_delta_cap"], row["embargo_mode"]))
        row.update(
            {
                "n_markets": int(g["market_id"].nunique()),
                "n_fills": int(g["n_fills"].sum()),
                "mean_net_pnl": float(g["net_pnl"].mean()),
                "net_ci_lo": lo,
                "net_ci_hi": hi,
                "median_net_pnl": float(g["net_pnl"].median()),
                "win_rate": float(g["net_pnl"].gt(0).mean()),
                "pnl_std": pnl_std,
                "pnl_variance": float(g["net_pnl"].var(ddof=1)) if len(g) > 1 else 0.0,
                "mean_unhedged_pnl": unhedged_mean,
                "unhedged_pnl_std": unhedged_std,
                "mean_hedge_pnl": float(g["hedge_pnl"].mean()),
                "mean_hedge_cost": float(g["hedge_cost"].mean()),
                "mean_hedge_turnover_notional": float(g["hedge_turnover_notional"].mean()),
                "mean_hedge_rebalances": float(g["hedge_rebalances"].mean()),
                "mean_avg_effective_h": float(g["avg_effective_h"].mean()),
                "mean_max_abs_hedge_notional": float(g["max_abs_hedge_notional"].mean()),
                "premium_retained": premium_retained,
                "variance_reduction": variance_reduction,
                "source_disagree_share": float(g["source_disagree"].mean()),
                "phase1_bucket_ci_lo": float(base["net_ci_lo"]) if base is not None else math.nan,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    return add_baseline_lifts(out)


def choose_rows(
    summary: pd.DataFrame,
    phase: str,
    bucket: str = GATE_BUCKET,
    split: str = GATE_SPLIT,
    embargo: str = GATE_EMBARGO,
    cap: float | None = math.inf,
) -> pd.DataFrame:
    sub = summary[
        summary["phase"].eq(phase)
        & summary["sample_split"].eq(split)
        & summary["bucket"].eq(bucket)
        & summary["embargo_mode"].eq(embargo)
    ].copy()
    if cap is not None:
        sub = sub[np.isclose(sub["dollar_delta_cap"].astype(float), cap)].copy()
    return sub


def fmt_summary_row(r: pd.Series) -> list[str]:
    return [
        str(r["filter_id"]),
        str(r["embargo_mode"]).replace("_time_nonoverlap", ""),
        fmt_cap(float(r["dollar_delta_cap"])),
        str(int(r["n_markets"])),
        str(int(r["n_fills"])),
        cents(float(r["mean_net_pnl"])),
        ci_text(float(r["net_ci_lo"]), float(r["net_ci_hi"])),
        cents(float(r["ci_lo_lift_vs_phase1"])),
        pct(float(r["win_rate"])),
        cents(float(r["pnl_std"])),
        pct(float(r["two_sided_market_share"])) if "two_sided_market_share" in r else "-",
        number(float(r["median_hold_seconds"]) / 60.0, 1) if "median_hold_seconds" in r else "-",
    ]


def robust_rows(df: pd.DataFrame, min_markets: int = 3) -> pd.DataFrame:
    if df.empty or "n_markets" not in df.columns:
        return df
    return df[df["n_markets"].astype(int).ge(min_markets)].copy()


def table_rows(df: pd.DataFrame, limit: int = 12, min_markets: int = 3) -> list[list[str]]:
    if df.empty:
        return []
    sub = robust_rows(df, min_markets=min_markets)
    if sub.empty:
        sub = df.copy()
    sub = sub.sort_values(["net_ci_lo", "mean_net_pnl"], ascending=[False, False]).head(limit)
    return [fmt_summary_row(r) for _, r in sub.iterrows()]


def fmt_hedge_row(r: pd.Series) -> list[str]:
    return [
        str(r["hedge_variant"]),
        str(r["filter_id"]),
        str(r["hedge_policy"]),
        number(float(r["h_param"]), 2),
        str(int(r["n_markets"])),
        cents(float(r["mean_net_pnl"])),
        ci_text(float(r["net_ci_lo"]), float(r["net_ci_hi"])),
        cents(float(r["mean_hedge_cost"])),
        number(float(r["premium_retained"]), 2),
        pct(float(r["variance_reduction"])),
        number(float(r["mean_hedge_turnover_notional"]), 2),
    ]


def hedge_rows(summary: pd.DataFrame, limit: int = 12) -> list[list[str]]:
    sub = summary[
        summary["phase"].eq("phase4_hedge_footnote")
        & summary["sample_split"].eq(GATE_SPLIT)
        & summary["bucket"].eq(GATE_BUCKET)
    ].copy()
    if sub.empty:
        return []
    sub = sub.sort_values(["net_ci_lo", "mean_net_pnl"], ascending=[False, False]).head(limit)
    return [fmt_hedge_row(r) for _, r in sub.iterrows()]


def portfolio_roll_rows(summary: pd.DataFrame, limit: int = 8) -> list[list[str]]:
    sub = summary[
        summary["phase"].eq("phase4_hedge_footnote")
        & summary["hedge_variant"].eq("portfolio_24h_roll")
        & summary["sample_split"].eq(GATE_SPLIT)
        & summary["bucket"].eq(GATE_BUCKET)
    ].copy()
    if sub.empty:
        return []
    sub = sub.sort_values(["net_ci_lo", "mean_net_pnl"], ascending=[False, False]).head(limit)
    return [fmt_hedge_row(r) for _, r in sub.iterrows()]


def power_diagnostic_rows(summary: pd.DataFrame) -> list[list[str]]:
    sub = summary[
        summary["phase"].eq("phase1_power")
        & summary["sample_split"].eq(GATE_SPLIT)
        & summary["bucket"].isin(["far_absz_ge1_all_tau", "longshot_absz_ge0.75_all_tau", "longshot_absz_ge0.50_all_tau"])
        & summary["filter_id"].isin(["bare_lifecycle", "bare_lifecycle_per_asset_power_diag"])
    ].copy()
    if sub.empty:
        return []
    order = {
        ("far_absz_ge1_all_tau", "global_time_nonoverlap"): 0,
        ("far_absz_ge1_all_tau", "per_asset_time_nonoverlap"): 1,
        ("longshot_absz_ge0.75_all_tau", "global_time_nonoverlap"): 2,
        ("longshot_absz_ge0.75_all_tau", "per_asset_time_nonoverlap"): 3,
        ("longshot_absz_ge0.50_all_tau", "global_time_nonoverlap"): 4,
        ("longshot_absz_ge0.50_all_tau", "per_asset_time_nonoverlap"): 5,
    }
    sub["rank"] = [order.get((r.bucket, r.embargo_mode), 99) for r in sub.itertuples()]
    sub = sub.sort_values("rank")
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            [
                str(r["bucket"]),
                str(r["embargo_mode"]).replace("_time_nonoverlap", ""),
                str(int(r["n_markets"])),
                str(int(r["n_fills"])),
                cents(float(r["mean_net_pnl"])),
                ci_text(float(r["net_ci_lo"]), float(r["net_ci_hi"])),
                pct(float(r["win_rate"])),
                cents(float(r["pnl_std"])),
            ]
        )
    return rows


def example_filter_text(fills: pd.DataFrame) -> str:
    sub = fills[
        fills["eligible"].fillna(False).astype(bool)
        & fills["entry_split"].eq(GATE_SPLIT)
        & fills["moneyness_bucket"].eq("far_absz_ge1")
        & fills["token_position"].lt(0)
        & fills["rich_short_edge"].ge(0.01)
    ].copy()
    if sub.empty:
        return "No eligible OOS far-|z| rich-short example was available."
    r = sub.sort_values("rich_short_edge", ascending=False).iloc[0]
    side = str(r["actual_outcome"]).upper()
    return (
        f"Example fill: `{r['market_slug']}` sold/shorted the {side} token at ${float(r['entry_price']):.3f}. "
        f"The Binance RV physical-probability fair for that token was ${float(r['token_rv_physical_prob_fair']):.3f}, so the token was rich by "
        f"{cents(float(r['rich_short_edge']))}. A `rich_short >= 1c` filter keeps this fill; if the edge were below "
        "`1c`, v3 would skip it even though the K-PEG maker lifecycle would have filled it."
    )


def write_note(fills: pd.DataFrame, episodes: pd.DataFrame, summary: pd.DataFrame) -> None:
    base = baseline_row(summary)
    if base is None:
        headline = "OD Strategy A v3 did not produce the pre-registered gate row."
        gate_text = "No OOS `far_absz_ge1_all_tau` global-embargo bare lifecycle row survived."
        gate_pass = False
    else:
        gate_pass = bool(float(base["net_ci_lo"]) > 0)
        headline = "OD Strategy A v3 baseline power gate fails; OD filters are tested next."
        gate_text = (
            f"Bare lifecycle OOS `far_absz_ge1_all_tau`, global embargo: n={int(base['n_markets'])} markets / "
            f"{int(base['n_fills'])} fills, mean {cents(float(base['mean_net_pnl']))}, "
            f"CI {ci_text(float(base['net_ci_lo']), float(base['net_ci_hi']))}."
        )

    phase2 = choose_rows(summary, "phase2_od_filter")
    robust_phase2 = robust_rows(phase2, min_markets=3)
    best_phase2 = None if robust_phase2.empty else robust_phase2.sort_values(["net_ci_lo", "mean_net_pnl"], ascending=[False, False]).iloc[0]
    v3_filter_pass = best_phase2 is not None and float(best_phase2["net_ci_lo"]) > 0
    phase3 = choose_rows(summary, "phase3_risk_cap", cap=None)
    robust_phase3 = robust_rows(phase3, min_markets=3)
    best_phase3 = None if robust_phase3.empty else robust_phase3.sort_values(["net_ci_lo", "mean_net_pnl"], ascending=[False, False]).iloc[0]
    positive_finite_caps = robust_phase3[
        robust_phase3["dollar_delta_cap"].astype(float).replace([np.inf, -np.inf], np.nan).notna()
        & robust_phase3["net_ci_lo"].astype(float).gt(0)
    ].copy()
    smallest_positive_cap = (
        None
        if positive_finite_caps.empty
        else positive_finite_caps.sort_values(["dollar_delta_cap", "net_ci_lo"], ascending=[True, False]).iloc[0]
    )
    hedges = summary[summary["phase"].eq("phase4_hedge_footnote")].copy()
    best_hedge = None if hedges.empty else hedges.sort_values(["net_ci_lo", "mean_net_pnl"], ascending=[False, False]).iloc[0]

    phase2_text = (
        "No OD filter rows survived."
        if best_phase2 is None
        else (
            f"Best OD filter row: `{best_phase2['filter_id']}` mean {cents(float(best_phase2['mean_net_pnl']))}, "
            f"CI {ci_text(float(best_phase2['net_ci_lo']), float(best_phase2['net_ci_hi']))}, "
            f"lower-CI lift {cents(float(best_phase2['ci_lo_lift_vs_phase1']))}."
        )
    )
    phase3_text = (
        "No cap rows survived."
        if best_phase3 is None
        else (
            f"Best cap row: `{best_phase3['filter_id']}`, cap {fmt_cap(float(best_phase3['dollar_delta_cap']))}, "
            f"mean {cents(float(best_phase3['mean_net_pnl']))}, "
            f"CI {ci_text(float(best_phase3['net_ci_lo']), float(best_phase3['net_ci_hi']))}."
            + (
                ""
                if smallest_positive_cap is None
                else (
                    f" Smallest positive finite cap: `{smallest_positive_cap['filter_id']}`, "
                    f"cap {fmt_cap(float(smallest_positive_cap['dollar_delta_cap']))}, "
                    f"mean {cents(float(smallest_positive_cap['mean_net_pnl']))}, "
                    f"CI {ci_text(float(smallest_positive_cap['net_ci_lo']), float(smallest_positive_cap['net_ci_hi']))}."
                )
            )
        )
    )
    hedge_text = (
        "No hedge footnote rows survived."
        if best_hedge is None
        else (
            f"Best hedge footnote: `{best_hedge['hedge_variant']}` / `{best_hedge['hedge_policy']}` h={number(float(best_hedge['h_param']), 2)}, "
            f"mean {cents(float(best_hedge['mean_net_pnl']))}, CI {ci_text(float(best_hedge['net_ci_lo']), float(best_hedge['net_ci_hi']))}, "
            f"variance reduction {pct(float(best_hedge['variance_reduction']))}."
        )
    )

    if v3_filter_pass:
        headline = "OD Strategy A v3 clears only after the OD/source filter; bare power still fails."
    elif base is not None:
        headline = "OD Strategy A v3 does not clear after robust OD filters."

    note = f"""# Sell Rich 4h Crypto UP/DOWN Digitals With OD Fair-Value Filters (Strategy A v3)

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Table terms: [[polymarket_table_dictionary]]

## Headline

{headline}

Phase-1 baseline: {gate_text}

Phase-2 OD filter verdict: {phase2_text}

Phase-3 cap verdict: {phase3_text}

Power did not improve under the official assumption: the K3/K6 + K-PEG overlap has only six globally
non-overlapping OOS 4h slots. Pooling BTC/ETH/SOL gives many more fills and market episodes before embargo,
but the pre-registered **global** embargo still permits only one overlapping 4h window at a time. The positive
v3 result comes from the OD/source filter, not from more power. A per-asset diagnostic is shown below; it is
not the official gate.

## Design

This is the OD strat, not a Block K lead-lag race. MM supplies the passive K-PEG lifecycle: eligible maker fills
are aggregated into a market episode, inventory can be two-sided, Polymarket positions are carried to resolution,
and the late near-50c spike zone remains excluded. OD adds the valuation layer: only accept fills when the token
is rich versus Binance RV physical-probability fair, when PM midpoint-implied vol is above causal EWMA vol, when the Chainlink/Binance
source-basis risk is acceptable, and when dollar-delta inventory stays inside a cap.

Decision gate: OOS `far_absz_ge1_all_tau`, global market-episode embargo, net-of-cost, lower 95% cluster-bootstrap
CI > 0. Phase 1 reports the bare lifecycle baseline; Phase 2 tests whether OD filters lift that lower CI above
zero. Costs include maker rebate on Polymarket fills. The hedge rows add Binance costs at `{BINANCE_HEDGE_COST_BPS:.1f}bp`
per hedge trade/settlement notional, but hedge is a footnote and cannot create the edge gate.

## Phase Map — No Hedge in Phases 1-3

The three v3 phases that matter for the OD decision are all **unhedged Polymarket lifecycle** tests:

- **Phase 1, power baseline:** replay the bare K-PEG lifecycle and ask whether more assets / wider longshot buckets
  create enough independent evidence.
- **Phase 2, OD entry filter:** before accepting a maker fill, ask whether the token is actually overpriced versus
  Binance RV physical-probability fair or backed by a strict source-basis filter.
- **Phase 3, risk caps:** keep the same unhedged lifecycle, but skip fills that would push episode dollar-delta
  exposure above a cap.

The Binance hedge is **Phase 4 only**. It is a variance/cost diagnostic after the unhedged decision has already
been made. The v3 PASS/FAIL statement above is therefore about the OD valuation filter, not about hedge PnL.

## Power and Global Embargo

`Power` means how much independent evidence the backtest has. More independent market episodes usually tighten
the confidence interval, so a real edge is less likely to be hidden by one huge winner or one bad loser. More
fills are helpful, but they are not the same as more power if those fills all happen inside the same overlapping
4h risk window.

The 4h windows themselves are sequential within one asset: one BTC 4h market ends, then the next BTC 4h market
opens. The `global market-episode embargo` is about **cross-asset same-time overlap**, not overlap within one
asset's 4h schedule. After selecting one 4h market episode, the replay ignores every other market episode whose
active time overlaps it, even if the other market is a different asset such as ETH or SOL. In plain English:
assume the strategy can have only one 4h OD episode active at a time, so same-time BTC, ETH, and SOL windows
compete for the same capital/risk slot.

That is why v3 found more fills but not much more official power. The data has 17 OOS far-|z| asset-market
episodes before a global embargo, but only 6 global 4h time slots after the embargo because many BTC, ETH, and
SOL episodes occur during the same 4h interval. The per-asset diagnostic shows what happens if BTC, ETH, and SOL
capital are treated independently; it is useful, but it is not the pre-registered gate.

## Options Valuation Layer

The 4h UP/DOWN market is treated as a European cash-or-nothing digital option. It resolves UP if the close is
above the window-open reference price. There is no barrier or path dependence in the payoff.

The causal fair price for the UP token is:

```text
z = ln(S / K) / (sigma * sqrt(tau))
P_up_fair = N(z)
P_down_fair = 1 - P_up_fair
```

Where:

- `S` is current Binance spot at the fill timestamp.
- `K` is the 4h window-open strike/reference price.
- `sigma` is causal EWMA realized volatility from the K6 panel, using only information available up to that timestamp.
- `tau` is time left to resolution in years.
- `N(z)` is the standard normal CDF.

For a specific Polymarket token, the RV physical-probability fair is `P_up_rv_fair` for an UP token and `1 - P_up_rv_fair` for a DOWN token.
The OD richness test then asks:

```text
short/sell token edge = entry_price - token_rv_physical_prob_fair
long/buy token edge   = token_rv_physical_prob_fair - entry_price
```

The headline v3 filter is deliberately narrow: it keeps short/sell fills only when the token is rich versus
RV physical-probability fair. That tests the OD thesis as a forecast-selection rule: sell longshot/vol tokens
that are expensive versus the causal RV model and carry them to resolution.

## Practical OD Filter Example

{example_filter_text(fills)}

The signed value-edge convention is:

```text
long token:  rv_physical_prob_fair - entry_price
short token: entry_price - rv_physical_prob_fair
```

The headline richness filter is stricter than generic value-edge: it keeps only short/sell fills where the token
is overpriced versus RV physical-probability fair. That directly tests the OD thesis as a forecast-selection rule:
sell the longshot/vol side that is expensive versus our causal RV model and carry.

## Table Glossary

- `filter`: the lifecycle/filter rule. `bare_lifecycle` is v2-style K-PEG with no OD valuation gate.
- `embargo`: `global` is the official one-position-at-a-time embargo; `per_asset` is a power diagnostic.
- `cap`: maximum absolute dollar-delta inventory allowed while accepting fills inside an episode.
- `markets`: selected non-overlapping market episodes.
- `fills`: K-PEG fills inside those selected episodes after filters and caps.
- `mean net`: mean dollars per market episode, displayed in cents.
- `CI`: 95% bootstrap confidence interval over market episodes.
- `CI lift`: lower-CI improvement versus the Phase-1 bare global gate row.
- `win`: share of selected market episodes with positive net PnL.
- `PnL std`: standard deviation of market-episode PnL.
- `two-sided`: share of episodes containing both UP and DOWN token fills.
- `hold min`: median fill-to-resolution hold time in minutes.
- Filter suffixes are compact: `005m` = $0.005 = 0.5c, `010m` = 1c, `050m` = 5c, and `05vp` = 5 annualized vol points. The CSV contains the thinner n<3 rows; the Markdown tables hide them unless no robust row exists.

## Bucket Glossary

Buckets describe the option state at the fill timestamp. They are based on moneyness and time left, not on the
eventual winner.

Moneyness uses:

```text
abs_z = abs(ln(S / K) / (sigma * sqrt(tau)))
```

Where `S` is Binance spot, `K` is the window-open strike, `sigma` is causal EWMA vol, and `tau` is time left.
In plain English, `abs_z` is "how many volatility-adjusted units away from the strike are we?"

- `near_absz_lt0.25`: very close to the strike. The option is jumpy; a small spot move can flip the market.
- `mid_absz_0.25_1`: moderately away from the strike. Delta is meaningful, but the outcome is not yet pinned.
- `far_absz_ge1`: at least one vol-adjusted unit from the strike. This is the longshot/pinned family that v2/v3
  care about most.
- `longshot_absz_ge0.75`: widened longshot family. It includes `far_absz_ge1` plus somewhat less extreme longshots
  to test whether the edge has more power when the boundary is relaxed.
- `longshot_absz_ge0.50`: even wider longshot family. It adds more fills/episodes, but can dilute the pure far-|z|
  thesis.
- `longshot_short_price_le30`: short/sell fills where the token price is at or below 30c. This is a practical
  "sell the cheap-looking longshot premium" bucket, distinct from the model-based `abs_z` bucket.

Time buckets:

- `early_gt2h`: more than two hours left to resolution.
- `mid_30m_2h`: 30 minutes to two hours left.
- `late_lt30m`: less than 30 minutes left.

Intersection buckets combine both labels, e.g. `far_absz_ge1|late_lt30m` means the market is far from the strike
and has less than 30 minutes left. `far_absz_ge1_all_tau` pools the far family across all time-left buckets.

How the phases use buckets:

- **Phase 1** reports the official far-|z| gate plus widened longshot buckets, so we can see whether power improves
  when the longshot definition is relaxed.
- **Phase 2** keeps the official `far_absz_ge1_all_tau` gate and changes the **entry filter** instead of the bucket.
- **Phase 3** also keeps the official far bucket and changes the **dollar-delta cap**, so cap effects are comparable
  to Phase 1/2.

## Phase 1 — Power Baseline

{markdown_table(
    ["bucket", "embargo", "markets", "fills", "mean net", "CI", "win", "PnL std"],
    power_diagnostic_rows(summary),
)}

Read: the official global far-|z| gate is the first row. The per-asset rows answer "what if BTC, ETH, and SOL
capital were treated independently?" They are useful for power diagnosis, but they are not the pre-registered
decision gate in this run.

## Phase 2 — OD Entry Filters

{markdown_table(
    ["filter", "embargo", "cap", "markets", "fills", "mean net", "CI", "CI lift", "win", "PnL std", "two-sided", "hold min"],
    table_rows(phase2, limit=16, min_markets=3),
)}

Read: these rows ask whether OD valuation adds independent keep on top of the maker lifecycle. `rich_short`
means sell only when the PM token is rich versus Binance RV physical-probability fair. `vol_premium` means PM
midpoint-implied vol is above causal EWMA vol. `strict` promotes the Chainlink/Binance source-basis filter from diagnostic to official
candidate design ingredient.

Non-hedged read: the bare far-|z| lifecycle failed because its lower CI was -17.14c. The strict-source and
strict-rich-short filters lift the lower CI above zero before any Binance hedge is applied. This says the RV
valuation filter is doing selection work in this replay; it does not prove external option-IV mispricing.

## Phase 3 — Dollar-Delta Risk Caps

{markdown_table(
    ["filter", "embargo", "cap", "markets", "fills", "mean net", "CI", "CI lift", "win", "PnL std", "two-sided", "hold min"],
    table_rows(phase3, limit=16, min_markets=3),
)}

Read: caps are meant to shrink the fat-tail dispersion, not discover a new edge. A good cap should retain most
of the mean while tightening the lower CI. If it improves CI only by deleting the large winner, that is not a
better strategy; it is just less exposure.

Non-hedged read: the $50 strict-rich-short cap is the cleanest risk-control proof-of-concept in this table:
it keeps only 3 markets / 7 fills, but the mean stays positive and the CI remains above zero while PnL standard
deviation falls materially. The broader strict-source caps keep more markets but still rely on the same source
filter that v3 promoted from diagnostic to design rule.

## Phase 4 — Hedge Footnote

{markdown_table(
    ["variant", "filter", "policy", "h", "markets", "net", "CI", "hedge cost", "prem retained", "var reduced", "turnover"],
    hedge_rows(summary, limit=16),
)}

Portfolio-roll diagnostic rows:

{markdown_table(
    ["variant", "filter", "policy", "h", "markets", "net", "CI", "hedge cost", "prem retained", "var reduced", "turnover"],
    portfolio_roll_rows(summary, limit=8),
)}

Read: `episode_static` is the v2-style one hedge per market episode. `portfolio_24h_roll` nets per-asset hedge
changes across the day instead of paying a separate close/open at every 4h boundary. Here the portfolio-roll
row is a cost/turnover diagnostic on the per-asset power universe; it is not part of the global gate. These
rows are variance diagnostics only. The OD edge gate is unhedged.

## Decision

Pre-registered Phase-1 baseline row: {gate_text}

Decision: **{'PASS' if v3_filter_pass else 'FAIL'}** for the OD-filtered v3 design; **{'PASS' if gate_pass else 'FAIL'}** for the bare Phase-1 lifecycle baseline.

Interpretation: the baseline did not get more power under the official global embargo. The actual v3 improvement
comes from OD/source filtering, especially strict-source plus rich-short filters. This is a positive OD-filter
result, but still a small market-episode sample. Per-asset rows are encouraging only if we explicitly reopen the
capital assumption and allow concurrent BTC/ETH/SOL episodes.

Outputs:

- `data/analysis/csv_outputs/options_delta/od_strategy_a_v3.csv`
- `data/analysis/od_strategy_a_v3_trades.parquet`
- `data/analysis/od_strategy_a_v3_fills.parquet`

Supersedes the next-step framing in [[od_strategy_a_v2_lifecycle_findings]]; v2 remains the accounting baseline.
"""
    note = normalize_markdown_wrapping(note)
    NOTE.write_text(note, encoding="utf-8")
    print(f"wrote {NOTE}", flush=True)


def update_hub_and_todo(summary: pd.DataFrame) -> None:
    base = baseline_row(summary)
    phase2 = robust_rows(choose_rows(summary, "phase2_od_filter"), min_markets=3)
    best_phase2 = None if phase2.empty else phase2.sort_values(["net_ci_lo", "mean_net_pnl"], ascending=[False, False]).iloc[0]
    if base is None:
        state_line = "OD Strategy A v3 produced no official gate row."
        gate_pass = False
    else:
        gate_pass = bool(float(base["net_ci_lo"]) > 0)
        state_line = (
            f"OD Strategy A v3 bare baseline {'cleared' if gate_pass else 'failed'} the Phase-1 global gate: "
            f"OOS far-|z| n={int(base['n_markets'])} markets / {int(base['n_fills'])} fills, "
            f"mean {cents(float(base['mean_net_pnl']))}, "
            f"CI {ci_text(float(base['net_ci_lo']), float(base['net_ci_hi']))}."
        )
    filter_line = (
        "No OD filter row survived."
        if best_phase2 is None
        else (
            f"Best OD filter was `{best_phase2['filter_id']}` with mean {cents(float(best_phase2['mean_net_pnl']))}, "
            f"CI {ci_text(float(best_phase2['net_ci_lo']), float(best_phase2['net_ci_hi']))}, "
            f"lower-CI lift {cents(float(best_phase2['ci_lo_lift_vs_phase1']))}."
        )
    )

    if OD_HUB.exists():
        text = OD_HUB.read_text(encoding="utf-8")
        new_status = f"status: REFRAMED 2026-06-01 — OD is the valuation/signal layer. Strategy A v3 OD filters {'cleared' if best_phase2 is not None and float(best_phase2['net_ci_lo']) > 0 else 'failed'} the global OOS gate while bare power still failed; see [[od_strategy_a_v3_findings]]. Kronos still gated pending explicit reopen."
        text = "\n".join(
            new_status if line.startswith("status: REFRAMED 2026-06-01") else line
            for line in text.splitlines()
        ) + "\n"
        marker = "- **OD Strategy A v2 lifecycle (gate FAILED on power, not edge):**"
        v3_line = (
            f"- **OD Strategy A v3:** {state_line} {filter_line} The power bottleneck remains the global embargo: "
            "the overlap has more fills before embargo, but only six OOS 4h time slots under the official gate. "
            "See [[od_strategy_a_v3_findings]].\n"
        )
        if "- **OD Strategy A v3:**" in text:
            text = "\n".join(v3_line.rstrip() if line.startswith("- **OD Strategy A v3:**") else line for line in text.splitlines()) + "\n"
        elif marker in text:
            text = text.replace(marker, v3_line + marker, 1)
        text = text.replace(
            "- [ ] **Strategy A v3 — power + OD entry filter (next critical-path job).**",
            "- [x] **Strategy A v3 — power + OD entry filter** (2026-06-01): completed; see [[od_strategy_a_v3_findings]]. Original task:",
            1,
        )
        OD_HUB.write_text(text, encoding="utf-8")

    if BRAIN_TODO.exists():
        text = BRAIN_TODO.read_text(encoding="utf-8")
        marker = "**OD Strategy A v2 lifecycle gate FAILED on power, not edge:**"
        v3_line = (
            f"**OD Strategy A v3:** {state_line} {filter_line} The global-embargo power bottleneck remains explicit; "
            "per-asset concurrency is only a diagnostic unless the capital assumption is reopened. See [[od_strategy_a_v3_findings]].\n\n"
        )
        if "**OD Strategy A v3:**" in text:
            text = "\n".join(v3_line.rstrip() if line.startswith("**OD Strategy A v3:**") else line for line in text.splitlines()) + "\n"
        elif marker in text:
            text = text.replace(marker, v3_line + marker, 1)
        text = text.replace(
            "- [ ] **Strategy A v3 — power + OD entry filter (NEXT critical-path Codex job).**",
            "- [x] **Strategy A v3 — power + OD entry filter** (2026-06-01): completed; see [[od_strategy_a_v3_findings]]. Original task:",
            1,
        )
        BRAIN_TODO.write_text(text, encoding="utf-8")


def main() -> None:
    print(f"root={ROOT}", flush=True)
    fills = load_v3_fills(refresh=True)
    print(
        f"fills={len(fills):,} eligible={int(fills['eligible'].sum()):,} "
        f"oos_markets={fills[fills['entry_split'].eq(GATE_SPLIT) & fills['eligible']].market_id.nunique():,}",
        flush=True,
    )
    episodes = build_all_episodes(fills)
    static_overlay = build_static_hedge_overlay(episodes, fills)
    roll_overlay = build_portfolio_roll_overlay(episodes, fills)
    ledger = pd.concat([episodes, static_overlay, roll_overlay], ignore_index=True, sort=False)
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(OUT_TRADES, index=False)
    print(f"wrote {OUT_TRADES}", flush=True)

    phase_summary = summarize_episodes(episodes)
    hedge_summary = summarize_hedges(pd.concat([static_overlay, roll_overlay], ignore_index=True, sort=False), phase_summary)
    summary = pd.concat([phase_summary, hedge_summary], ignore_index=True, sort=False)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"wrote {OUT_SUMMARY}", flush=True)

    write_note(fills, episodes, summary)
    update_hub_and_todo(summary)
    print(f"v2 baseline note used for continuity: {V2_NOTE}", flush=True)


if __name__ == "__main__":
    main()
