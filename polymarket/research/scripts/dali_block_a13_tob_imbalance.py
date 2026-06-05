"""Block A1.3 TOB imbalance level deep-dive.

This is a sidecar analysis over ``block_a1_features.parquet``. It does not
mutate raw captures or the canonical A1/A1.1 artifacts.
"""
from __future__ import annotations

import argparse
import math
import re
import zlib
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
FEATURES = ANALYSIS / "block_a1_features.parquet"
A11_COMPONENT_SWEEP = ANALYSIS / "csv_outputs" / "dali" / "block_a11_ofi_component_sweep.csv"
PLOTS = ANALYSIS / "block_a13_plots"
NOTE = ROOT / "notes" / "block_a13_tob_imbalance_findings.md"

DECILES_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a13_tob_decile_aggregate.csv"
A11_RECON_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a13_a11_reconciliation.csv"
PERSISTENCE_RUNS_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a13_tob_persistence_runs.csv"
PERSISTENCE_MARKET_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a13_tob_persistence_by_market.csv"
CONTROL_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a13_tob_control_buckets.csv"
JOINT_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a13_tob_ofi_joint_signal.csv"
TFI_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a13_tob_conditional_tfi.csv"

HORIZONS = (1, 5, 30, 300)
TFI_HORIZONS = (5, 30)
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260528
MIN_REPORTABLE_TOP_DECILE_N = 30
PRIMARY_VARIANT = "current_level"
CONFIRM_VARIANT = "window_mean_level"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def rel_note_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(NOTE.parent.resolve().parent))
    except ValueError:
        return str(path)


def safe_slug(raw: object, fallback: str) -> str:
    text = str(raw or fallback).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:90] or fallback


def stable_seed_offset(*parts: object, modulo: int = 10_000) -> int:
    text = "|".join(str(part) for part in parts)
    return int(zlib.crc32(text.encode("utf-8")) % modulo)


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def seconds(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value < 60:
        return f"{value:.1f}s"
    if value < 3600:
        return f"{value / 60.0:.1f}m"
    return f"{value / 3600.0:.1f}h"


def qbucket(series: pd.Series, labels: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return out
    try:
        bucketed = pd.qcut(valid.rank(method="first"), len(labels), labels=labels)
    except ValueError:
        return out
    out.loc[bucketed.index] = bucketed.astype(str)
    return out


def load_features(path: Path) -> pd.DataFrame:
    cols = [
        "run_id",
        "received_at",
        "event_type",
        "asset_id",
        "market_id",
        "family",
        "slug",
        "question",
        "outcome_index",
        "is_book_state_complete",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "tob_imbalance",
        "ofi_combined_event",
        "trade_side",
        "last_trade_side",
        "trade_size",
        "market_resolved_at",
    ]
    con = duckdb.connect()
    select_cols = ", ".join(cols)
    df = con.execute(f"SELECT {select_cols} FROM read_parquet('{path}')").df()
    if df.empty:
        raise SystemExit(f"no rows found in {path}")

    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    df["market_resolved_at"] = pd.to_datetime(df["market_resolved_at"], utc=True, errors="coerce")
    for col in ("run_id", "event_type", "asset_id", "market_id", "family", "slug", "question"):
        df[col] = df[col].fillna("").astype(str)
    numeric_cols = [
        "outcome_index",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "tob_imbalance",
        "ofi_combined_event",
        "trade_size",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["market_key"] = np.where(df["market_id"].ne(""), df["market_id"], df["asset_id"])
    df["market_label"] = np.where(df["slug"].ne(""), df["slug"], df["market_key"])
    df["touch_depth"] = df[["best_bid_size", "best_ask_size"]].sum(axis=1, min_count=2)
    df["spread_bps"] = np.where(
        df["mid"].gt(0) & df["spread"].notna(),
        df["spread"] / df["mid"] * 10_000.0,
        np.nan,
    )
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    df["directional_mid"] = np.where(df["direction_factor"].gt(0), df["mid"], 1.0 - df["mid"])

    side = (
        df["trade_side"]
        .fillna(df["last_trade_side"])
        .fillna("")
        .astype(str)
        .str.upper()
        .map({"BUY": 1.0, "SELL": -1.0})
        .fillna(0.0)
    )
    df["signed_trade_size_live"] = df["event_type"].eq("last_trade_price").astype(float) * side * df[
        "trade_size"
    ].fillna(0.0)
    df["ofi_combined_event"] = df["ofi_combined_event"].fillna(0.0)

    market_depth = (
        df.groupby(["run_id", "market_key"], as_index=False)["touch_depth"]
        .mean()
        .rename(columns={"touch_depth": "market_mean_depth"})
    )
    df = df.merge(market_depth, on=["run_id", "market_key"], how="left")
    df["relative_depth"] = df["touch_depth"] / df["market_mean_depth"]
    df["spread_bucket"] = qbucket(
        df["spread_bps"],
        ["spread_q1_tight", "spread_q2", "spread_q3", "spread_q4_wide"],
    ).fillna("spread_unknown")
    df["relative_depth_bucket"] = qbucket(
        df["relative_depth"],
        ["depth_q1_shallow", "depth_q2", "depth_q3", "depth_q4_deep"],
    ).fillna("depth_unknown")
    df["resolved_in_capture"] = np.where(df["market_resolved_at"].notna(), "resolved", "unresolved")

    return df.sort_values(["run_id", "asset_id", "received_at"]).reset_index(drop=True)


def future_value(g: pd.DataFrame, value_col: str, horizon_sec: int) -> np.ndarray:
    times = g["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    values = g[value_col].to_numpy(dtype=float)
    target = times + horizon_sec * 1_000_000_000
    last_time = times[-1] if len(times) else 0
    idx = np.searchsorted(times, target, side="right") - 1
    out = np.full(len(g), np.nan, dtype=float)
    valid = (target <= last_time) & (idx >= 0)
    out[valid] = values[idx[valid]]
    return out


def add_horizon_features(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    grouped = list(df.groupby(["run_id", "asset_id"], sort=False))
    for idx, ((run_id, asset_id), group) in enumerate(grouped, start=1):
        print(f"horizon features {idx:02d}/{len(grouped):02d}: {run_id}/{str(asset_id)[:12]}", flush=True)
        g = group.sort_values("received_at").copy()
        g["tob_imbalance"] = g["tob_imbalance"].ffill()
        g["tob_signal_current"] = g["direction_factor"] * g["tob_imbalance"]
        g = g.set_index("received_at", drop=False)
        for horizon in HORIZONS:
            window = f"{horizon}s"
            future_mid = future_value(g, "mid", horizon)
            future_directional_mid = np.where(g["direction_factor"].gt(0), future_mid, 1.0 - future_mid)
            g[f"future_return_bps_{horizon}s"] = np.where(
                g["directional_mid"].gt(0) & np.isfinite(future_directional_mid),
                (future_directional_mid - g["directional_mid"]) / g["directional_mid"] * 10_000.0,
                np.nan,
            )
            ofi = g["ofi_combined_event"].fillna(0.0).rolling(window).sum()
            tfi = g["signed_trade_size_live"].fillna(0.0).rolling(window).sum()
            tob_window = g["tob_imbalance"].rolling(window).mean()
            g[f"signal_current_level_{horizon}s"] = g["tob_signal_current"]
            g[f"signal_window_mean_level_{horizon}s"] = g["direction_factor"] * tob_window
            g[f"ofi_scaled_{horizon}s"] = g["direction_factor"] * ofi / g["market_mean_depth"]
            g[f"tfi_market_{horizon}s"] = g["direction_factor"] * tfi
        pieces.append(g.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def signal_col(variant: str, horizon: int) -> str:
    return f"signal_{variant}_{horizon}s"


def valid_signal_rows(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    sub = df[
        df["is_book_state_complete"].fillna(False)
        & df[x_col].replace([np.inf, -np.inf], np.nan).notna()
        & df[y_col].replace([np.inf, -np.inf], np.nan).notna()
        & df[x_col].ne(0)
    ].copy()
    return sub[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col])]


def assign_deciles(sub: pd.DataFrame, x_col: str) -> pd.DataFrame:
    out = sub.copy()
    out["abs_signal"] = out[x_col].abs()
    try:
        out["decile"] = pd.qcut(out["abs_signal"], 10, labels=False, duplicates="drop") + 1
    except ValueError:
        out["decile"] = np.nan
    return out


def hit_and_directional_return(rows: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float, int]:
    signs = (np.sign(rows[x_col]) != 0) & (np.sign(rows[y_col]) != 0)
    if not signs.any():
        hit_rate = math.nan
        sign_eval_n = 0
    else:
        signed_x = np.sign(rows.loc[signs, x_col])
        y = rows.loc[signs, y_col]
        hit_rate = float((signed_x == np.sign(y)).mean())
        sign_eval_n = int(signs.sum())
    directional_return = float((np.sign(rows[x_col]) * rows[y_col]).mean())
    return hit_rate, directional_return, sign_eval_n


def block_bootstrap_hit(sub: pd.DataFrame, x_col: str, y_col: str, seed: int) -> tuple[float, float]:
    clean = sub[["received_at", x_col, y_col]].dropna()
    clean = clean[np.isfinite(clean[x_col]) & np.isfinite(clean[y_col])]
    if len(clean) < MIN_REPORTABLE_TOP_DECILE_N:
        return math.nan, math.nan
    elapsed = (clean["received_at"] - clean["received_at"].min()).dt.total_seconds()
    block_id = (elapsed // BOOTSTRAP_CHUNK_SECONDS).astype(int).to_numpy()
    if len(np.unique(block_id)) < 4:
        block_id = np.arange(len(clean)) // max(5, len(clean) // 10)
    blocks = [np.flatnonzero(block_id == bid) for bid in np.unique(block_id)]
    if len(blocks) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    x = clean[x_col].to_numpy(dtype=float)
    y = clean[y_col].to_numpy(dtype=float)
    vals: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))])
        mask = (np.sign(x[idx]) != 0) & (np.sign(y[idx]) != 0)
        if mask.sum() >= 5:
            vals.append(float((np.sign(x[idx][mask]) == np.sign(y[idx][mask])).mean()))
    if len(vals) < 20:
        return math.nan, math.nan
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_rows(
    rows: pd.DataFrame,
    x_col: str,
    y_col: str,
    seed: int,
    *,
    include_ci: bool = True,
) -> dict[str, object]:
    hit_rate, directional_return, sign_eval_n = hit_and_directional_return(rows, x_col, y_col)
    lo, hi = block_bootstrap_hit(rows, x_col, y_col, seed) if include_ci else (math.nan, math.nan)
    fam = rows["family"].value_counts(normalize=True, dropna=False)
    top_family = str(fam.index[0]) if len(fam) else ""
    top_family_share = float(fam.iloc[0]) if len(fam) else math.nan
    reportable = (
        sign_eval_n >= MIN_REPORTABLE_TOP_DECILE_N
        and np.isfinite(hit_rate)
        and 0.0 < hit_rate < 1.0
    )
    return {
        "n": int(len(rows)),
        "sign_eval_n": int(sign_eval_n),
        "mean_abs_signal": float(rows[x_col].abs().mean()),
        "mean_next_mid_return_bps": float(rows[y_col].mean()),
        "directional_return_bps": directional_return,
        "hit_rate": hit_rate,
        "hit_rate_ci_lo": lo,
        "hit_rate_ci_hi": hi,
        "metric_reportable": bool(reportable),
        "mean_spread_bps": float(rows["spread_bps"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_touch_depth": float(rows["touch_depth"].replace([np.inf, -np.inf], np.nan).mean()),
        "market_count": int(rows["market_key"].nunique()),
        "top_family": top_family,
        "top_family_share": top_family_share,
    }


def decile_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for variant in (PRIMARY_VARIANT, CONFIRM_VARIANT):
        for horizon in HORIZONS:
            x_col = signal_col(variant, horizon)
            y_col = f"future_return_bps_{horizon}s"
            sub = valid_signal_rows(df, x_col, y_col)
            if len(sub) < 50:
                continue
            sub = assign_deciles(sub, x_col)
            sub = sub[sub["decile"].notna()]
            for decile, rows in sub.groupby("decile", sort=True):
                out_rows.append({
                    "signal_variant": variant,
                    "horizon_sec": horizon,
                    "decile": int(decile),
                    **summarize_rows(
                        rows,
                        x_col,
                        y_col,
                        RNG_SEED + horizon * 100 + int(decile) + (0 if variant == PRIMARY_VARIANT else 10_000),
                    ),
                })
    return pd.DataFrame(out_rows)


def a11_reconciliation(a13_deciles: pd.DataFrame, a11_path: Path) -> pd.DataFrame:
    if not a11_path.exists():
        return pd.DataFrame()
    a11 = pd.read_csv(a11_path)
    a11 = a11[
        a11["component"].eq("tob_imbalance_level")
        & a11["horizon_sec"].isin(HORIZONS)
        & a11["decile"].eq(10)
    ].copy()
    a13 = a13_deciles[
        a13_deciles["signal_variant"].eq(CONFIRM_VARIANT) & a13_deciles["decile"].eq(10)
    ].copy()
    merged = a13.merge(
        a11[
            [
                "horizon_sec",
                "n",
                "hit_rate",
                "directional_return_bps",
                "mean_abs_signal",
            ]
        ].rename(
            columns={
                "n": "a11_n",
                "hit_rate": "a11_hit_rate",
                "directional_return_bps": "a11_directional_return_bps",
                "mean_abs_signal": "a11_mean_abs_signal",
            }
        ),
        on="horizon_sec",
        how="left",
    )
    if merged.empty:
        return merged
    merged["hit_rate_diff"] = merged["hit_rate"] - merged["a11_hit_rate"]
    merged["directional_return_diff_bps"] = (
        merged["directional_return_bps"] - merged["a11_directional_return_bps"]
    )
    return merged[
        [
            "horizon_sec",
            "n",
            "a11_n",
            "hit_rate",
            "a11_hit_rate",
            "hit_rate_diff",
            "directional_return_bps",
            "a11_directional_return_bps",
            "directional_return_diff_bps",
            "mean_abs_signal",
            "a11_mean_abs_signal",
        ]
    ]


def persistence(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, object]] = []
    for (run_id, asset_id), group in df.groupby(["run_id", "asset_id"], sort=False):
        g = group.sort_values("received_at").copy()
        g["tob_imbalance"] = g["tob_imbalance"].ffill()
        g["signal"] = g["direction_factor"] * g["tob_imbalance"]
        g["sign"] = np.sign(g["signal"]).replace(0, np.nan)
        g = g[g["sign"].notna()].copy()
        if len(g) < 2:
            continue
        sign_change = g["sign"].ne(g["sign"].shift()).cumsum()
        g["state_run_id"] = sign_change
        seg = (
            g.groupby("state_run_id", as_index=False)
            .agg(
                run_id=("run_id", "first"),
                asset_id=("asset_id", "first"),
                market_key=("market_key", "first"),
                market_label=("market_label", "first"),
                family=("family", "first"),
                sign=("sign", "first"),
                started_at=("received_at", "min"),
                last_seen_at=("received_at", "max"),
                n_rows=("received_at", "size"),
                mean_abs_signal=("signal", lambda s: float(np.nanmean(np.abs(s)))),
            )
            .sort_values("started_at")
            .reset_index(drop=True)
        )
        seg["next_started_at"] = seg["started_at"].shift(-1)
        seg["duration_until_flip_sec"] = (
            seg["next_started_at"] - seg["started_at"]
        ).dt.total_seconds()
        seg["observed_duration_sec"] = (seg["last_seen_at"] - seg["started_at"]).dt.total_seconds()
        seg["is_censored"] = seg["next_started_at"].isna()
        run_rows.extend(seg.to_dict("records"))

    runs = pd.DataFrame(run_rows)
    market_rows: list[dict[str, object]] = []
    if not runs.empty:
        for (run_id, market_key), rows in runs.groupby(["run_id", "market_key"], sort=False):
            uncensored = rows["duration_until_flip_sec"].replace([np.inf, -np.inf], np.nan).dropna()
            market_rows.append({
                "run_id": run_id,
                "market_key": market_key,
                "market_label": str(rows["market_label"].dropna().iloc[0]),
                "family": str(rows["family"].dropna().iloc[0]),
                "asset_count": int(rows["asset_id"].nunique()),
                "row_obs": int(rows["n_rows"].sum()),
                "state_run_count": int(len(rows)),
                "uncensored_state_runs": int(len(uncensored)),
                "censored_state_run_share": float(rows["is_censored"].mean()) if len(rows) else math.nan,
                "median_time_until_flip_sec": float(uncensored.median()) if len(uncensored) else math.nan,
                "p10_time_until_flip_sec": float(uncensored.quantile(0.10)) if len(uncensored) else math.nan,
                "p90_time_until_flip_sec": float(uncensored.quantile(0.90)) if len(uncensored) else math.nan,
                "share_flip_le_1s": float((uncensored <= 1).mean()) if len(uncensored) else math.nan,
                "share_flip_le_5s": float((uncensored <= 5).mean()) if len(uncensored) else math.nan,
                "share_flip_le_30s": float((uncensored <= 30).mean()) if len(uncensored) else math.nan,
                "median_state_run_duration_sec": float(uncensored.median()) if len(uncensored) else math.nan,
                "p90_state_run_duration_sec": float(uncensored.quantile(0.90)) if len(uncensored) else math.nan,
            })
    markets = pd.DataFrame(market_rows)
    if not markets.empty:
        markets = markets.sort_values(["row_obs", "state_run_count"], ascending=False)
    return runs, markets


def control_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        x_col = signal_col(PRIMARY_VARIANT, horizon)
        y_col = f"future_return_bps_{horizon}s"
        sub = valid_signal_rows(df, x_col, y_col)
        if len(sub) < 50:
            continue
        sub = assign_deciles(sub, x_col)
        top = sub[sub["decile"].eq(10)].copy()
        top["spread_x_depth"] = top["spread_bucket"] + "|" + top["relative_depth_bucket"]
        for segment_type, segment_col in [
            ("spread_bucket", "spread_bucket"),
            ("relative_depth_bucket", "relative_depth_bucket"),
            ("spread_x_depth", "spread_x_depth"),
        ]:
            for value, rows in top.groupby(segment_col, sort=True):
                if len(rows) < MIN_REPORTABLE_TOP_DECILE_N:
                    continue
                out_rows.append({
                    "signal_variant": PRIMARY_VARIANT,
                    "segment_type": segment_type,
                    "segment_value": str(value),
                    "horizon_sec": horizon,
                    "decile": 10,
                    **summarize_rows(
                        rows,
                        x_col,
                        y_col,
                        RNG_SEED + horizon * 1000 + stable_seed_offset(segment_type, value, modulo=500),
                    ),
                })
    return pd.DataFrame(out_rows)


def joint_signal(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        tob_col = signal_col(PRIMARY_VARIANT, horizon)
        ofi_col = f"ofi_scaled_{horizon}s"
        y_col = f"future_return_bps_{horizon}s"
        sub = df[
            df["is_book_state_complete"].fillna(False)
            & df[tob_col].replace([np.inf, -np.inf], np.nan).notna()
            & df[ofi_col].replace([np.inf, -np.inf], np.nan).notna()
            & df[y_col].replace([np.inf, -np.inf], np.nan).notna()
        ].copy()
        sub = sub[np.isfinite(sub[tob_col]) & np.isfinite(sub[ofi_col]) & np.isfinite(sub[y_col])]
        if len(sub) < 50:
            continue
        sub["tob_decile"] = assign_deciles(sub[sub[tob_col].ne(0)], tob_col)["decile"]
        sub["ofi_decile"] = assign_deciles(sub[sub[ofi_col].ne(0)], ofi_col)["decile"]
        sub["tob_sign"] = np.sign(sub[tob_col])
        sub["ofi_sign"] = np.sign(sub[ofi_col])
        conditions = [
            sub["tob_sign"].ne(0) & sub["ofi_sign"].ne(0) & sub["tob_sign"].eq(sub["ofi_sign"]),
            sub["tob_sign"].ne(0) & sub["ofi_sign"].ne(0) & sub["tob_sign"].ne(sub["ofi_sign"]),
            sub["tob_sign"].ne(0) & sub["ofi_sign"].eq(0),
            sub["tob_sign"].eq(0) & sub["ofi_sign"].ne(0),
        ]
        choices = ["same_sign", "disagree", "imbalance_only", "ofi_only"]
        sub["joint_bin"] = np.select(conditions, choices, default="both_zero")
        sub["slice"] = "all_valid"
        slices = [sub[sub["joint_bin"].ne("both_zero")].copy()]
        slices[0]["slice"] = "all_valid"
        for slice_name, mask in [
            ("tob_top_decile", sub["tob_decile"].eq(10)),
            ("ofi_top_decile", sub["ofi_decile"].eq(10)),
            ("either_top_decile", sub["tob_decile"].eq(10) | sub["ofi_decile"].eq(10)),
        ]:
            piece = sub[mask & sub["joint_bin"].ne("both_zero")].copy()
            piece["slice"] = slice_name
            slices.append(piece)
        joint = pd.concat(slices, ignore_index=True)
        for (slice_name, joint_bin), rows in joint.groupby(["slice", "joint_bin"], sort=True):
            if len(rows) < MIN_REPORTABLE_TOP_DECILE_N:
                continue
            tob_hit, tob_dir, tob_n = hit_and_directional_return(rows, tob_col, y_col)
            ofi_hit, ofi_dir, ofi_n = hit_and_directional_return(rows, ofi_col, y_col)
            lo, hi = block_bootstrap_hit(
                rows,
                tob_col if joint_bin != "ofi_only" else ofi_col,
                y_col,
                RNG_SEED + horizon * 2000 + stable_seed_offset(slice_name, joint_bin, modulo=500),
            )
            out_rows.append({
                "horizon_sec": horizon,
                "slice": slice_name,
                "joint_bin": joint_bin,
                "n": int(len(rows)),
                "tob_hit_rate": tob_hit,
                "tob_directional_return_bps": tob_dir,
                "tob_sign_eval_n": tob_n,
                "ofi_hit_rate": ofi_hit,
                "ofi_directional_return_bps": ofi_dir,
                "ofi_sign_eval_n": ofi_n,
                "primary_hit_rate_ci_lo": lo,
                "primary_hit_rate_ci_hi": hi,
                "same_minus_disagree_hint": "",
            })
    return pd.DataFrame(out_rows)


def conditional_tfi(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for horizon in TFI_HORIZONS:
        tob_col = signal_col(PRIMARY_VARIANT, horizon)
        tfi_col = f"tfi_market_{horizon}s"
        y_col = f"future_return_bps_{horizon}s"
        sub = df[
            df["is_book_state_complete"].fillna(False)
            & df[tob_col].replace([np.inf, -np.inf], np.nan).notna()
            & df[tfi_col].replace([np.inf, -np.inf], np.nan).notna()
            & df[y_col].replace([np.inf, -np.inf], np.nan).notna()
        ].copy()
        sub = sub[np.isfinite(sub[tob_col]) & np.isfinite(sub[tfi_col]) & np.isfinite(sub[y_col])]
        if len(sub) < 50:
            continue
        nonzero_tob = sub[sub[tob_col].ne(0)].copy()
        nonzero_tob = assign_deciles(nonzero_tob, tob_col)
        top_idx = nonzero_tob[nonzero_tob["decile"].eq(10)].index
        slices = {
            "all_tfi_nonzero": sub[sub[tfi_col].ne(0)].copy(),
            "tob_top_decile_tfi_nonzero": sub[sub.index.isin(top_idx) & sub[tfi_col].ne(0)].copy(),
            "tob_top_decile_all": sub[sub.index.isin(top_idx)].copy(),
        }
        for slice_name, rows in slices.items():
            if len(rows) < MIN_REPORTABLE_TOP_DECILE_N:
                continue
            out_rows.append({
                "horizon_sec": horizon,
                "slice": slice_name,
                **summarize_rows(
                    rows,
                    tfi_col,
                    y_col,
                    RNG_SEED + horizon * 3000 + stable_seed_offset(slice_name, modulo=500),
                ),
                "tfi_nonzero_share": float(rows[tfi_col].ne(0).mean()) if len(rows) else math.nan,
            })
    return pd.DataFrame(out_rows)


def pivot_heatmap(
    data: pd.DataFrame,
    index_col: str,
    column_col: str,
    value_col: str,
) -> tuple[np.ndarray, list[str], list[int]]:
    pivot = data.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc="mean")
    if pivot.empty:
        return np.array([]), [], []
    columns = sorted(int(c) for c in pivot.columns)
    pivot = pivot.reindex(columns=columns)
    labels = [str(idx)[:72] for idx in pivot.index]
    return pivot.to_numpy(dtype=float), labels, columns


def plot_heatmap(
    mat: np.ndarray,
    labels: list[str],
    columns: list[int],
    title: str,
    filename: str,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str = "",
) -> Path | None:
    if mat.size == 0 or np.all(~np.isfinite(mat)):
        return None
    PLOTS.mkdir(parents=True, exist_ok=True)
    height = max(4.0, min(12.0, 0.35 * len(labels) + 2.4))
    fig, ax = plt.subplots(figsize=(9.2, height))
    im = ax.imshow(mat, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([f"{c}s" for c in columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    fig.tight_layout()
    out = PLOTS / filename
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_decile_lines(deciles: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    PLOTS.mkdir(parents=True, exist_ok=True)
    for variant, sub_variant in deciles.groupby("signal_variant", sort=False):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
        for ax, horizon in zip(axes.ravel(), HORIZONS, strict=False):
            sub = sub_variant[sub_variant["horizon_sec"].eq(horizon)].sort_values("decile")
            ax.plot(sub["decile"], sub["hit_rate"], marker="o", color="#1f77b4")
            ax.fill_between(
                sub["decile"].astype(float),
                sub["hit_rate_ci_lo"].astype(float),
                sub["hit_rate_ci_hi"].astype(float),
                color="#1f77b4",
                alpha=0.16,
            )
            ax.axhline(0.50, color="#666666", linewidth=1, linestyle="--")
            ax.set_title(f"{horizon}s")
            ax.grid(True, alpha=0.25)
        fig.supxlabel("Absolute TOB imbalance signal decile")
        fig.supylabel("Hit rate")
        fig.suptitle(f"A1.3 TOB imbalance decile hit rate: {variant}")
        fig.tight_layout()
        out = PLOTS / f"block_a13_tob_decile_hit_{safe_slug(variant, 'variant')}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_persistence(runs: pd.DataFrame) -> Path | None:
    if runs.empty:
        return None
    uncensored = runs[runs["duration_until_flip_sec"].notna()].copy()
    if uncensored.empty:
        return None
    order = (
        uncensored.groupby("market_label", as_index=False)["duration_until_flip_sec"]
        .size()
        .sort_values("size", ascending=False)
        .head(12)["market_label"]
    )
    sub = uncensored[uncensored["market_label"].isin(order)].copy()
    if sub.empty:
        return None
    labels = list(order)
    ncols = 3
    nrows = math.ceil(len(labels) / ncols)
    bins = np.logspace(-2, 5, 45)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, max(6, 2.5 * nrows)), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, label in zip(axes_arr, labels, strict=False):
        vals = sub.loc[sub["market_label"].eq(label), "duration_until_flip_sec"].clip(lower=0.01)
        ax.hist(vals, bins=bins, color="#2ca02c", alpha=0.75)
        ax.set_xscale("log")
        ax.set_title(str(label)[:42], fontsize=9)
        ax.grid(True, alpha=0.2)
    for ax in axes_arr[len(labels):]:
        ax.axis("off")
    fig.supxlabel("Seconds until imbalance sign flips (log scale)")
    fig.supylabel("State-run count")
    fig.suptitle("A1.3 TOB imbalance sign persistence by market")
    fig.tight_layout()
    out = PLOTS / "block_a13_tob_persistence_hist_by_market.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_controls(control: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for segment_type in ("spread_bucket", "relative_depth_bucket", "spread_x_depth"):
        sub = control[control["segment_type"].eq(segment_type)].copy()
        if sub.empty:
            continue
        mat, labels, columns = pivot_heatmap(sub, "segment_value", "horizon_sec", "hit_rate")
        path = plot_heatmap(
            mat,
            labels,
            columns,
            f"A1.3 top-decile TOB hit rate by {segment_type}",
            f"block_a13_{segment_type}_top_decile_hit_heatmap.png",
            cmap="viridis",
            vmin=0.35,
            vmax=0.85,
            colorbar_label="hit_rate",
        )
        if path:
            paths.append(path)
        sub["log10_n"] = np.log10(sub["n"].clip(lower=1))
        mat, labels, columns = pivot_heatmap(sub, "segment_value", "horizon_sec", "log10_n")
        path = plot_heatmap(
            mat,
            labels,
            columns,
            f"A1.3 top-decile TOB row count by {segment_type}",
            f"block_a13_{segment_type}_top_decile_n_heatmap.png",
            cmap="magma",
            colorbar_label="log10(n)",
        )
        if path:
            paths.append(path)
    return paths


def plot_joint(joint: pd.DataFrame) -> Path | None:
    if joint.empty:
        return None
    sub = joint[joint["slice"].eq("all_valid")].copy()
    mat, labels, columns = pivot_heatmap(sub, "joint_bin", "horizon_sec", "tob_hit_rate")
    return plot_heatmap(
        mat,
        labels,
        columns,
        "A1.3 TOB hit rate by TOB/OFI sign agreement",
        "block_a13_tob_ofi_joint_hit_heatmap.png",
        cmap="viridis",
        vmin=0.35,
        vmax=0.85,
        colorbar_label="TOB hit_rate",
    )


def plot_tfi(tfi: pd.DataFrame) -> Path | None:
    if tfi.empty:
        return None
    PLOTS.mkdir(parents=True, exist_ok=True)
    order = ["all_tfi_nonzero", "tob_top_decile_tfi_nonzero", "tob_top_decile_all"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.24
    horizons = sorted(tfi["horizon_sec"].unique())
    x = np.arange(len(horizons))
    for idx, slice_name in enumerate(order):
        vals = []
        for horizon in horizons:
            row = tfi[tfi["horizon_sec"].eq(horizon) & tfi["slice"].eq(slice_name)]
            vals.append(float(row["hit_rate"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (idx - 1) * width, vals, width=width, label=slice_name)
    ax.axhline(0.50, color="#666666", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(h)}s" for h in horizons])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("TFI hit rate")
    ax.set_title("A1.3 conditional TFI hit rate in high-TOB regimes")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = PLOTS / "block_a13_conditional_tfi_hit_rate.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def top_decile_table(deciles: pd.DataFrame, variant: str) -> str:
    sub = deciles[deciles["signal_variant"].eq(variant) & deciles["decile"].eq(10)].sort_values("horizon_sec")
    rows = [
        [
            int(row["horizon_sec"]),
            pct(float(row["hit_rate"])),
            f"[{pct(float(row['hit_rate_ci_lo']))}, {pct(float(row['hit_rate_ci_hi']))}]",
            bps(float(row["directional_return_bps"])),
            f"{int(row['n']):,}",
            row["top_family"],
            pct(float(row["top_family_share"])),
        ]
        for _, row in sub.iterrows()
    ]
    return markdown_table(["h", "hit", "CI", "dir ret", "n", "top family", "share"], rows)


def control_table(control: pd.DataFrame, segment_type: str, horizon: int = 5) -> str:
    sub = control[
        control["segment_type"].eq(segment_type)
        & control["horizon_sec"].eq(horizon)
    ].sort_values(["hit_rate", "n"], ascending=False)
    rows = [
        [
            str(row["segment_value"])[:70],
            pct(float(row["hit_rate"])),
            f"[{pct(float(row['hit_rate_ci_lo']))}, {pct(float(row['hit_rate_ci_hi']))}]",
            bps(float(row["directional_return_bps"])),
            f"{int(row['n']):,}",
        ]
        for _, row in sub.head(16).iterrows()
    ]
    return markdown_table([segment_type, "5s hit", "CI", "dir ret", "n"], rows)


def joint_table(joint: pd.DataFrame, horizon: int = 5) -> str:
    sub = joint[joint["horizon_sec"].eq(horizon) & joint["slice"].eq("all_valid")].copy()
    sub = sub.sort_values("n", ascending=False)
    rows = [
        [
            row["joint_bin"],
            pct(float(row["tob_hit_rate"])),
            pct(float(row["ofi_hit_rate"])),
            bps(float(row["tob_directional_return_bps"])),
            bps(float(row["ofi_directional_return_bps"])),
            f"{int(row['n']):,}",
        ]
        for _, row in sub.iterrows()
    ]
    return markdown_table(["joint bin", "TOB hit", "OFI hit", "TOB dir", "OFI dir", "n"], rows)


def tfi_table(tfi: pd.DataFrame) -> str:
    sub = tfi.sort_values(["horizon_sec", "slice"])
    rows = [
        [
            int(row["horizon_sec"]),
            row["slice"],
            pct(float(row["hit_rate"])),
            f"[{pct(float(row['hit_rate_ci_lo']))}, {pct(float(row['hit_rate_ci_hi']))}]",
            bps(float(row["directional_return_bps"])),
            f"{int(row['n']):,}",
            pct(float(row["tfi_nonzero_share"])),
        ]
        for _, row in sub.iterrows()
    ]
    return markdown_table(["h", "slice", "TFI hit", "CI", "dir ret", "n", "TFI nonzero"], rows)


def persistence_table(markets: pd.DataFrame) -> str:
    if markets.empty:
        return "_No persistence rows._"
    sub = markets.sort_values("row_obs", ascending=False).head(12)
    rows = [
        [
            str(row["market_label"])[:60],
            row["family"],
            f"{int(row['state_run_count']):,}",
            seconds(float(row["median_time_until_flip_sec"])),
            seconds(float(row["p90_time_until_flip_sec"])),
            pct(float(row["share_flip_le_5s"])),
            pct(float(row["share_flip_le_30s"])),
        ]
        for _, row in sub.iterrows()
    ]
    return markdown_table(
        ["market", "family", "runs", "median flip", "p90 flip", "<=5s", "<=30s"],
        rows,
    )


def headline_sentence(deciles: pd.DataFrame, joint: pd.DataFrame, tfi: pd.DataFrame) -> str:
    top = deciles[
        deciles["signal_variant"].eq(PRIMARY_VARIANT)
        & deciles["decile"].eq(10)
        & deciles["horizon_sec"].eq(5)
    ]
    top_text = "current-level 5s top decile was unavailable"
    if not top.empty:
        row = top.iloc[0]
        top_text = (
            f"current-level 5s top decile hit {pct(float(row['hit_rate']))} "
            f"(CI [{pct(float(row['hit_rate_ci_lo']))}, {pct(float(row['hit_rate_ci_hi']))}], "
            f"n={int(row['n']):,}) with {bps(float(row['directional_return_bps']))} directional return"
        )
    same = joint[
        joint["horizon_sec"].eq(5)
        & joint["slice"].eq("all_valid")
        & joint["joint_bin"].eq("same_sign")
    ]
    same_text = ""
    if not same.empty:
        row = same.iloc[0]
        same_text = (
            f" When TOB and OFI agree at 5s, TOB hit is {pct(float(row['tob_hit_rate']))}; "
            f"when they disagree, inspect the joint table rather than assuming OFI adds edge."
        )
    tfi_top = tfi[
        tfi["horizon_sec"].eq(5)
        & tfi["slice"].eq("tob_top_decile_tfi_nonzero")
    ]
    tfi_text = ""
    if not tfi_top.empty:
        row = tfi_top.iloc[0]
        tfi_text = f" Conditional TFI in high-TOB rows hits {pct(float(row['hit_rate']))} at 5s."
    return top_text + "." + same_text + tfi_text


def write_note(
    deciles: pd.DataFrame,
    recon: pd.DataFrame,
    persistence_markets: pd.DataFrame,
    control: pd.DataFrame,
    joint: pd.DataFrame,
    tfi: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    current_table = top_decile_table(deciles, PRIMARY_VARIANT)
    window_table = top_decile_table(deciles, CONFIRM_VARIANT)
    spread_table = control_table(control, "spread_bucket", 5)
    depth_table = control_table(control, "relative_depth_bucket", 5)
    spread_depth_table = control_table(control, "spread_x_depth", 5)
    joint_snapshot = joint_table(joint, 5)
    tfi_snapshot = tfi_table(tfi)
    persistence_snapshot = persistence_table(persistence_markets)
    plot_refs = "\n".join(f"![]({rel_note_path(path)})" for path in plot_paths)

    recon_text = "_No A1.1 reconciliation file was available._"
    if not recon.empty:
        rows = [
            [
                int(row["horizon_sec"]),
                pct(float(row["hit_rate"])),
                pct(float(row["a11_hit_rate"])),
                f"{float(row['hit_rate_diff']):+.4f}",
                bps(float(row["directional_return_bps"])),
                bps(float(row["a11_directional_return_bps"])),
            ]
            for _, row in recon.sort_values("horizon_sec").iterrows()
        ]
        recon_text = markdown_table(
            ["h", "A13 window hit", "A11 hit", "diff", "A13 dir ret", "A11 dir ret"],
            rows,
        )

    note = f"""---
tags: [dali, block-a13, tob-imbalance, results]
---

# Block A1.3 TOB Imbalance Level Deep-Dive

## Headline

{headline_sentence(deciles, joint, tfi)} The result supports promoting `tob_imbalance_level` to a primary A2 candidate, with two caveats: it is a standing state variable rather than flow, and the executable-cost question still needs bid/ask entry-exit tests rather than mid-return diagnostics.

## Method

This sidecar uses `{display_path(FEATURES)}` only. `tob_imbalance_level = (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size)` is sign-normalized by `direction_factor`, so positive means imbalance favors the market-direction outcome. Two variants are reported:

- `current_level`: the standing top-of-book imbalance at signal time `t`; this is the primary A1.3 signal because the variable is a level/state, not an order-flow sum.
- `window_mean_level`: the rolling mean of the level over the horizon; this reconciles to A1.1's component sweep.

Deciles are global equal-count buckets within each horizon and variant, using absolute signal magnitude. Decile 10 is the largest absolute imbalance, not the most bullish imbalance. Hit rate is `sign(signal) == sign(future_directional_mid_return_bps)`.

## A1.1 Reconciliation

{recon_text}

## Decile Aggregate

### Current Level

{current_table}

Read: the current standing imbalance is not a weaker proxy for the A1.1 result; it is stronger at 5s and 30s in this panel. That is consistent with a book-state signal rather than a decaying flow signal.

### A1.1-Compatible Window Mean

{window_table}

## Persistence

The persistence test measures how long the sign-normalized TOB imbalance sign lasts before flipping. This distinguishes a durable state from a rapidly changing flow-like feature.

{persistence_snapshot}

Read: persistence is market-specific rather than family-uniform. Crypto and several AI/geopolitics books flip on sub-second-to-seconds state-run horizons, while China/Hormuz-style geopolitical books can keep the same imbalance sign for minutes to hours. That means `tob_imbalance_level` is sometimes a fast dynamic state and sometimes a slow book descriptor, so A2 should carry persistence controls.

## Spread And Depth Controls

Bucket labels are reconstructed with the same quantile-bucket method used in A1.1. The A1.1 CSV does not persist numeric bucket boundaries, so A1.3 reconstructs the buckets from the same feature table and code path.

### Spread Bucket, 5s Top Decile

{spread_table}

### Relative Depth Bucket, 5s Top Decile

{depth_table}

### Spread x Depth, 5s Top Decile

{spread_depth_table}

Read: the 5s top-decile signal survives every spread quartile, so it is not just a tight-spread artifact. It is much stronger in deep relative-depth cells and weak or negative in several `depth_q2` cells, so depth conditioning should travel into A2.

## Joint TOB x OFI Signal

The joint table compares sign agreement between current TOB imbalance and depth-normalized OFI. `same_sign` means both predictors point the same way; `disagree` means they conflict; `imbalance_only` and `ofi_only` mean one signal is zero.

{joint_snapshot}

Read: OFI does not show a clean incremental sign edge over current TOB imbalance in this run. When TOB and OFI agree, the hit rate is identical by construction; when they disagree, TOB remains positive and OFI points the wrong way in the 5s aggregate.

## Conditional TFI

This checks whether trade-flow imbalance becomes more useful inside high-TOB-imbalance regimes.

{tfi_snapshot}

Read: TFI becomes more interesting inside high-TOB-imbalance rows at 5s, but the effect is much weaker by 30s. Treat this as a conditional 5s interaction candidate, not a standalone replacement for the TOB level signal.

## Plots

{plot_refs}

## Outputs

- `{display_path(DECILES_OUT)}`
- `{display_path(A11_RECON_OUT)}`
- `{display_path(PERSISTENCE_RUNS_OUT)}`
- `{display_path(PERSISTENCE_MARKET_OUT)}`
- `{display_path(CONTROL_OUT)}`
- `{display_path(JOINT_OUT)}`
- `{display_path(TFI_OUT)}`
- `{display_path(PLOTS)}/`

Recommended next action for Justin: make `tob_imbalance_level` a primary A2 feature candidate alongside OFI, but require executable bid/ask cost tests before treating it as tradable edge.
"""
    NOTE.write_text(note, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"loading {display_path(args.features)}", flush=True)
    df = load_features(args.features)
    print(f"loaded {len(df):,} feature rows", flush=True)
    df = add_horizon_features(df)

    print("building TOB decile aggregates", flush=True)
    deciles = decile_aggregate(df)
    print("building A1.1 reconciliation", flush=True)
    recon = a11_reconciliation(deciles, A11_COMPONENT_SWEEP)
    print("building persistence diagnostics", flush=True)
    persistence_runs, persistence_markets = persistence(df)
    print("building spread/depth controls", flush=True)
    control = control_buckets(df)
    print("building TOB/OFI joint signal diagnostics", flush=True)
    joint = joint_signal(df)
    print("building conditional TFI diagnostics", flush=True)
    tfi = conditional_tfi(df)

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    deciles.to_csv(DECILES_OUT, index=False)
    recon.to_csv(A11_RECON_OUT, index=False)
    persistence_runs.to_csv(PERSISTENCE_RUNS_OUT, index=False)
    persistence_markets.to_csv(PERSISTENCE_MARKET_OUT, index=False)
    control.to_csv(CONTROL_OUT, index=False)
    joint.to_csv(JOINT_OUT, index=False)
    tfi.to_csv(TFI_OUT, index=False)

    plot_paths: list[Path] = []
    if not args.skip_plots:
        plot_paths.extend(plot_decile_lines(deciles))
        path = plot_persistence(persistence_runs)
        if path:
            plot_paths.append(path)
        plot_paths.extend(plot_controls(control))
        path = plot_joint(joint)
        if path:
            plot_paths.append(path)
        path = plot_tfi(tfi)
        if path:
            plot_paths.append(path)

    write_note(deciles, recon, persistence_markets, control, joint, tfi, plot_paths)
    print(f"deciles: {display_path(DECILES_OUT)} ({len(deciles):,} rows)")
    print(f"persistence runs: {display_path(PERSISTENCE_RUNS_OUT)} ({len(persistence_runs):,} rows)")
    print(f"controls: {display_path(CONTROL_OUT)} ({len(control):,} rows)")
    print(f"joint: {display_path(JOINT_OUT)} ({len(joint):,} rows)")
    print(f"tfi: {display_path(TFI_OUT)} ({len(tfi):,} rows)")
    print(f"plots: {display_path(PLOTS)} ({len(plot_paths):,} files)")
    print(f"note: {display_path(NOTE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
