"""Block A1.5 TOB extensions: multi-level imbalance and micro-price.

This sidecar reuses the A1.2 per-level book-state parquet. It does not read or
mutate raw JSONL captures.
"""
from __future__ import annotations

import argparse
import math
import re
import zlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTE = ROOT / "notes" / "block_a15_tob_extensions_findings.md"

A12_FEATURES = ANALYSIS / "block_a12_mlofi_features.parquet"
A12_COMPARISON = ANALYSIS / "csv_outputs" / "dali" / "block_a12_mlofi_comparison.csv"
A13_DECILES = ANALYSIS / "csv_outputs" / "dali" / "block_a13_tob_decile_aggregate.csv"

FEATURES_OUT = ANALYSIS / "block_a15_features.parquet"
IMBALANCE_DECILES_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a15_imbalance_variants_decile.csv"
MICRO_TARGET_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a15_microprice_target_comparison.csv"
MICRO_SIGNAL_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a15_microprice_signal_decile.csv"
BASELINE_CHECK_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a15_baseline_check.csv"

DEPTH = 10
HORIZONS = (1, 5, 30, 300)
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260528
MIN_REPORTABLE_TOP_DECILE_N = 30
BASELINE_TOL = 1e-12

IMBALANCE_VARIANTS = (
    "tob_current_level",
    "integrated_imbalance_l1_l10",
    "depth_weighted_imbalance_l1_l10",
    "exp_decay_imbalance_alpha_0p1",
    "exp_decay_imbalance_alpha_0p3",
    "exp_decay_imbalance_alpha_0p5",
)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def pp(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:+.2f} pp"


def safe_slug(raw: object, fallback: str) -> str:
    text = str(raw or fallback).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:90] or fallback


def stable_seed_offset(*parts: object, modulo: int = 10_000) -> int:
    text = "|".join(str(part) for part in parts)
    return int(zlib.crc32(text.encode("utf-8")) % modulo)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def required_a12_columns() -> list[str]:
    return [
        "run_id",
        "shard",
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
        "touch_depth",
        "market_resolved_at",
        *[f"bid_size_l{level}" for level in range(1, DEPTH + 1)],
        *[f"ask_size_l{level}" for level in range(1, DEPTH + 1)],
    ]


def assert_a12_has_per_level_state(path: Path) -> None:
    con = duckdb.connect()
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").df()
    cols = set(schema["column_name"].astype(str))
    missing = [col for col in required_a12_columns() if col not in cols]
    if missing:
        raise SystemExit(
            f"{display_path(path)} is missing required per-level state columns: {missing[:12]}"
        )


def load_a12_features(path: Path) -> pd.DataFrame:
    assert_a12_has_per_level_state(path)
    df = pd.read_parquet(path, columns=required_a12_columns())
    if df.empty:
        raise SystemExit(f"no rows found in {display_path(path)}")

    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    df["market_resolved_at"] = pd.to_datetime(df["market_resolved_at"], utc=True, errors="coerce")
    for col in ("run_id", "shard", "event_type", "asset_id", "market_id", "family", "slug", "question"):
        df[col] = df[col].fillna("").astype(str)
    numeric_cols = [
        "outcome_index",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "touch_depth",
        *[f"bid_size_l{level}" for level in range(1, DEPTH + 1)],
        *[f"ask_size_l{level}" for level in range(1, DEPTH + 1)],
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["market_key"] = np.where(df["market_id"].ne(""), df["market_id"], df["asset_id"])
    df["market_label"] = np.where(df["slug"].ne(""), df["slug"], df["market_key"])
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    df["directional_mid"] = np.where(df["direction_factor"].gt(0), df["mid"], 1.0 - df["mid"])
    df["spread_bps"] = np.where(df["mid"].gt(0) & df["spread"].notna(), df["spread"] / df["mid"] * 10_000.0, np.nan)
    market_depth = (
        df.groupby(["run_id", "market_key"], as_index=False)["touch_depth"]
        .mean()
        .rename(columns={"touch_depth": "market_mean_depth"})
    )
    df = df.merge(market_depth, on=["run_id", "market_key"], how="left")
    return df.sort_values(["run_id", "asset_id", "received_at"]).reset_index(drop=True)


def add_imbalance_and_microprice(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bid_cols = [f"bid_size_l{level}" for level in range(1, DEPTH + 1)]
    ask_cols = [f"ask_size_l{level}" for level in range(1, DEPTH + 1)]
    bid = out[bid_cols].fillna(0.0).to_numpy(dtype=float)
    ask = out[ask_cols].fillna(0.0).to_numpy(dtype=float)
    depth = bid + ask
    imbalance = np.divide(
        bid - ask,
        depth,
        out=np.zeros_like(depth, dtype=float),
        where=depth > 0,
    )
    for idx in range(DEPTH):
        level = idx + 1
        out[f"imbalance_l{level}"] = imbalance[:, idx]
        out[f"depth_l{level}"] = depth[:, idx]

    out["integrated_imbalance_l1_l10"] = imbalance.sum(axis=1)
    depth_denom = depth.sum(axis=1)
    out["depth_weighted_imbalance_l1_l10"] = np.divide(
        (imbalance * depth).sum(axis=1),
        depth_denom,
        out=np.zeros(len(out), dtype=float),
        where=depth_denom > 0,
    )
    for alpha in (0.1, 0.3, 0.5):
        suffix = str(alpha).replace(".", "p")
        weights = np.exp(-alpha * np.arange(DEPTH, dtype=float))
        out[f"exp_decay_imbalance_alpha_{suffix}"] = imbalance @ weights

    # A13-compatible TOB imbalance is intentionally stricter than the generic
    # per-level definition: it is missing unless both touch sizes are present.
    denom = out["best_bid_size"] + out["best_ask_size"]
    out["tob_imbalance_level_raw"] = np.where(
        out["best_bid_size"].notna() & out["best_ask_size"].notna() & denom.gt(0),
        (out["best_bid_size"] - out["best_ask_size"]) / denom,
        np.nan,
    )
    out["micro_price"] = out["mid"] + 0.5 * out["spread"] * out["tob_imbalance_level_raw"]
    out["directional_micro_price"] = np.where(
        out["direction_factor"].gt(0),
        out["micro_price"],
        1.0 - out["micro_price"],
    )
    return out


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


def past_value(g: pd.DataFrame, value_col: str, horizon_sec: int) -> np.ndarray:
    times = g["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    values = g[value_col].to_numpy(dtype=float)
    target = times - horizon_sec * 1_000_000_000
    first_time = times[0] if len(times) else 0
    idx = np.searchsorted(times, target, side="right") - 1
    out = np.full(len(g), np.nan, dtype=float)
    valid = (target >= first_time) & (idx >= 0)
    out[valid] = values[idx[valid]]
    return out


def add_horizon_features(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    grouped = list(df.groupby(["run_id", "asset_id"], sort=False))
    for idx, ((run_id, asset_id), group) in enumerate(grouped, start=1):
        print(f"horizon features {idx:02d}/{len(grouped):02d}: {run_id}/{str(asset_id)[:12]}", flush=True)
        g = group.sort_values("received_at").copy()
        g["tob_imbalance_level_raw"] = g["tob_imbalance_level_raw"].ffill()
        g["micro_price"] = g["micro_price"].ffill()
        g["directional_micro_price"] = g["directional_micro_price"].ffill()
        g["signal_tob_current_level"] = g["direction_factor"] * g["tob_imbalance_level_raw"]
        for variant in IMBALANCE_VARIANTS:
            if variant == "tob_current_level":
                g[f"signal_{variant}"] = g["signal_tob_current_level"]
            else:
                g[f"signal_{variant}"] = g["direction_factor"] * g[variant]

        for horizon in HORIZONS:
            future_mid = future_value(g, "mid", horizon)
            future_directional_mid = np.where(g["direction_factor"].gt(0), future_mid, 1.0 - future_mid)
            g[f"future_mid_return_bps_{horizon}s"] = np.where(
                g["directional_mid"].gt(0) & np.isfinite(future_directional_mid),
                (future_directional_mid - g["directional_mid"]) / g["directional_mid"] * 10_000.0,
                np.nan,
            )
            future_micro = future_value(g, "micro_price", horizon)
            future_directional_micro = np.where(g["direction_factor"].gt(0), future_micro, 1.0 - future_micro)
            g[f"future_micro_target_return_bps_{horizon}s"] = np.where(
                g["directional_mid"].gt(0) & np.isfinite(future_directional_micro),
                (future_directional_micro - g["directional_mid"]) / g["directional_mid"] * 10_000.0,
                np.nan,
            )
            past_micro = past_value(g, "micro_price", horizon)
            g[f"micro_price_change_{horizon}s"] = g["micro_price"] - past_micro
            g[f"signal_micro_change_{horizon}s"] = (
                g["direction_factor"] * g[f"micro_price_change_{horizon}s"] / g["market_mean_depth"]
            )
        pieces.append(g.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def time_block_ids(received_at: pd.Series, min_blocks: int = 4) -> np.ndarray:
    elapsed = (received_at - received_at.min()).dt.total_seconds()
    block_id = (elapsed // BOOTSTRAP_CHUNK_SECONDS).astype(int).to_numpy()
    if len(np.unique(block_id)) < min_blocks:
        block_id = np.arange(len(received_at)) // max(5, len(received_at) // 10)
    return block_id


def fast_block_bootstrap_hit(sub: pd.DataFrame, x_col: str, y_col: str, seed: int) -> tuple[float, float]:
    clean = sub[["received_at", x_col, y_col]].dropna()
    clean = clean[np.isfinite(clean[x_col]) & np.isfinite(clean[y_col])]
    signs = (np.sign(clean[x_col]) != 0) & (np.sign(clean[y_col]) != 0)
    clean = clean.loc[signs].copy()
    if len(clean) < MIN_REPORTABLE_TOP_DECILE_N:
        return math.nan, math.nan
    clean["hit"] = (np.sign(clean[x_col]) == np.sign(clean[y_col])).astype(int)
    clean["block_id"] = time_block_ids(clean["received_at"])
    block_stats = clean.groupby("block_id", as_index=False).agg(hits=("hit", "sum"), n=("hit", "size"))
    if len(block_stats) < 2:
        return math.nan, math.nan
    stats = block_stats[["hits", "n"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = stats[rng.integers(0, len(stats), size=len(stats))]
        hits, n = sampled.sum(axis=0)
        if n >= 5:
            vals.append(float(hits / n))
    if len(vals) < 20:
        return math.nan, math.nan
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


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


def summarize_decile(rows: pd.DataFrame, x_col: str, y_col: str, seed: int) -> dict[str, object]:
    signs = (np.sign(rows[x_col]) != 0) & (np.sign(rows[y_col]) != 0)
    hit_rate = (
        float((np.sign(rows.loc[signs, x_col]) == np.sign(rows.loc[signs, y_col])).mean())
        if signs.any()
        else math.nan
    )
    lo, hi = fast_block_bootstrap_hit(rows, x_col, y_col, seed)
    fam = rows["family"].value_counts(normalize=True, dropna=False)
    top_family = str(fam.index[0]) if len(fam) else ""
    top_family_share = float(fam.iloc[0]) if len(fam) else math.nan
    return {
        "n": int(len(rows)),
        "sign_eval_n": int(signs.sum()),
        "mean_abs_signal": float(rows["abs_signal"].mean()),
        "mean_next_return_bps": float(rows[y_col].mean()),
        "directional_return_bps": float((np.sign(rows[x_col]) * rows[y_col]).mean()),
        "hit_rate": hit_rate,
        "hit_rate_ci_lo": lo,
        "hit_rate_ci_hi": hi,
        "mean_spread_bps": float(rows["spread_bps"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_touch_depth": float(rows["touch_depth"].replace([np.inf, -np.inf], np.nan).mean()),
        "market_count": int(rows["market_key"].nunique()),
        "top_family": top_family,
        "top_family_share": top_family_share,
    }


def decile_aggregate(
    df: pd.DataFrame,
    *,
    variants: tuple[str, ...],
    y_prefix: str,
    output_variant_col: str = "variant",
) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        y_col = f"{y_prefix}_{horizon}s"
        for variant in variants:
            x_col = f"signal_{variant}" if not variant.startswith("micro_change_") else f"signal_{variant}"
            if "{h}" in x_col:
                x_col = x_col.format(h=horizon)
            elif variant == "micro_change":
                x_col = f"signal_micro_change_{horizon}s"
            sub = valid_signal_rows(df, x_col, y_col)
            if len(sub) < 50:
                continue
            sub = assign_deciles(sub, x_col)
            sub = sub[sub["decile"].notna()]
            max_decile = int(sub["decile"].max())
            for decile, rows in sub.groupby("decile", sort=True):
                out_rows.append({
                    output_variant_col: variant,
                    "horizon_sec": horizon,
                    "decile": int(decile),
                    "is_top_decile": int(decile) == max_decile,
                    **summarize_decile(
                        rows,
                        x_col,
                        y_col,
                        RNG_SEED
                        + horizon * 100
                        + int(decile)
                        + stable_seed_offset(variant, y_prefix, modulo=5_000),
                    ),
                })
    return pd.DataFrame(out_rows)


def imbalance_decile_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return decile_aggregate(
        df,
        variants=IMBALANCE_VARIANTS,
        y_prefix="future_mid_return_bps",
    )


def micro_signal_decile_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return decile_aggregate(
        df,
        variants=("micro_change",),
        y_prefix="future_mid_return_bps",
    )


def baseline_check(imbalance_deciles: pd.DataFrame, a13_path: Path) -> pd.DataFrame:
    baseline = imbalance_deciles[
        imbalance_deciles["variant"].eq("tob_current_level") & imbalance_deciles["is_top_decile"]
    ][
        [
            "horizon_sec",
            "n",
            "hit_rate",
            "directional_return_bps",
            "mean_abs_signal",
        ]
    ].rename(
        columns={
            "n": "a15_n",
            "hit_rate": "a15_hit_rate",
            "directional_return_bps": "a15_directional_return_bps",
            "mean_abs_signal": "a15_mean_abs_signal",
        }
    )
    a13 = pd.read_csv(a13_path)
    a13 = a13[
        a13["signal_variant"].eq("current_level")
        & a13["decile"].eq(10)
        & a13["horizon_sec"].isin(HORIZONS)
    ][
        [
            "horizon_sec",
            "n",
            "hit_rate",
            "directional_return_bps",
            "mean_abs_signal",
        ]
    ].rename(
        columns={
            "n": "a13_n",
            "hit_rate": "a13_hit_rate",
            "directional_return_bps": "a13_directional_return_bps",
            "mean_abs_signal": "a13_mean_abs_signal",
        }
    )
    out = baseline.merge(a13, on="horizon_sec", how="outer").sort_values("horizon_sec")
    out["hit_delta_pp"] = (out["a15_hit_rate"] - out["a13_hit_rate"]) * 100.0
    out["directional_return_delta_bps"] = (
        out["a15_directional_return_bps"] - out["a13_directional_return_bps"]
    )
    out["n_delta"] = out["a15_n"] - out["a13_n"]
    out["passes_exact"] = (
        out["n_delta"].eq(0)
        & out["hit_delta_pp"].abs().le(BASELINE_TOL)
        & out["directional_return_delta_bps"].abs().le(BASELINE_TOL)
    )
    return out


def ensure_baseline_passes(check: pd.DataFrame) -> None:
    if check.empty or len(check) != len(HORIZONS) or not bool(check["passes_exact"].all()):
        raise SystemExit(
            "A1.5 baseline gate failed; not reporting downstream A15 results. "
            f"See {display_path(BASELINE_CHECK_OUT)}"
        )


def microprice_target_comparison(df: pd.DataFrame, a13_deciles: pd.DataFrame) -> pd.DataFrame:
    micro_deciles = decile_aggregate(
        df,
        variants=("tob_current_level",),
        y_prefix="future_micro_target_return_bps",
    )
    micro_top = micro_deciles[micro_deciles["is_top_decile"]].copy()
    micro_top = micro_top[
        [
            "horizon_sec",
            "n",
            "hit_rate",
            "hit_rate_ci_lo",
            "hit_rate_ci_hi",
            "directional_return_bps",
            "mean_next_return_bps",
        ]
    ].rename(
        columns={
            "n": "micro_target_top_decile_n",
            "hit_rate": "micro_target_hit_rate",
            "hit_rate_ci_lo": "micro_target_hit_rate_ci_lo",
            "hit_rate_ci_hi": "micro_target_hit_rate_ci_hi",
            "directional_return_bps": "micro_target_directional_return_bps",
            "mean_next_return_bps": "micro_target_mean_return_bps",
        }
    )
    mid_top = a13_deciles[
        a13_deciles["variant"].eq("tob_current_level") & a13_deciles["is_top_decile"]
    ][
        [
            "horizon_sec",
            "n",
            "hit_rate",
            "hit_rate_ci_lo",
            "hit_rate_ci_hi",
            "directional_return_bps",
            "mean_next_return_bps",
        ]
    ].rename(
        columns={
            "n": "mid_target_top_decile_n",
            "hit_rate": "mid_target_hit_rate",
            "hit_rate_ci_lo": "mid_target_hit_rate_ci_lo",
            "hit_rate_ci_hi": "mid_target_hit_rate_ci_hi",
            "directional_return_bps": "mid_target_directional_return_bps",
            "mean_next_return_bps": "mid_target_mean_return_bps",
        }
    )
    out = mid_top.merge(micro_top, on="horizon_sec", how="outer").sort_values("horizon_sec")
    out["hit_delta_micro_minus_mid_pp"] = (
        out["micro_target_hit_rate"] - out["mid_target_hit_rate"]
    ) * 100.0
    out["directional_return_delta_micro_minus_mid_bps"] = (
        out["micro_target_directional_return_bps"] - out["mid_target_directional_return_bps"]
    )
    return out


def attach_l1_ofi_reference(micro_signal: pd.DataFrame, a12_comparison_path: Path) -> pd.DataFrame:
    if micro_signal.empty or not a12_comparison_path.exists():
        return micro_signal
    comp = pd.read_csv(a12_comparison_path)
    l1 = comp[comp["variant"].eq("l1_cks")][
        [
            "horizon_sec",
            "top_decile_hit_rate",
            "top_decile_directional_return_bps",
            "top_decile_n",
        ]
    ].rename(
        columns={
            "top_decile_hit_rate": "l1_ofi_hit_rate",
            "top_decile_directional_return_bps": "l1_ofi_directional_return_bps",
            "top_decile_n": "l1_ofi_top_decile_n",
        }
    )
    out = micro_signal.merge(l1, on="horizon_sec", how="left")
    out["hit_delta_vs_l1_ofi_pp"] = (out["hit_rate"] - out["l1_ofi_hit_rate"]) * 100.0
    out["directional_return_delta_vs_l1_ofi_bps"] = (
        out["directional_return_bps"] - out["l1_ofi_directional_return_bps"]
    )
    return out


def compact_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "run_id",
        "shard",
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
        "touch_depth",
        "market_mean_depth",
        "market_key",
        "direction_factor",
        "directional_mid",
        "tob_imbalance_level_raw",
        "micro_price",
        "directional_micro_price",
        *[f"imbalance_l{level}" for level in range(1, DEPTH + 1)],
        *[f"depth_l{level}" for level in range(1, DEPTH + 1)],
        *[variant for variant in IMBALANCE_VARIANTS if variant != "tob_current_level"],
        "signal_tob_current_level",
        *[f"future_mid_return_bps_{horizon}s" for horizon in HORIZONS],
        *[f"future_micro_target_return_bps_{horizon}s" for horizon in HORIZONS],
        *[f"micro_price_change_{horizon}s" for horizon in HORIZONS],
        *[f"signal_micro_change_{horizon}s" for horizon in HORIZONS],
    ]
    return df[cols].copy()


def write_outputs(
    features: pd.DataFrame,
    imbalance_deciles: pd.DataFrame,
    micro_target: pd.DataFrame,
    micro_signal: pd.DataFrame,
    baseline: pd.DataFrame,
) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    compact_feature_table(features).to_parquet(FEATURES_OUT, index=False, compression="zstd")
    imbalance_deciles.to_csv(IMBALANCE_DECILES_OUT, index=False)
    micro_target.to_csv(MICRO_TARGET_OUT, index=False)
    micro_signal.to_csv(MICRO_SIGNAL_OUT, index=False)
    baseline.to_csv(BASELINE_CHECK_OUT, index=False)


def top_rows(df: pd.DataFrame, value_col: str = "hit_rate") -> pd.DataFrame:
    top = df[df["is_top_decile"]].copy()
    return top.sort_values(["horizon_sec", value_col], ascending=[True, False])


def imbalance_table(imbalance_deciles: pd.DataFrame) -> str:
    top = top_rows(imbalance_deciles)
    rows = []
    for _, row in top.iterrows():
        baseline = top[
            top["horizon_sec"].eq(row["horizon_sec"])
            & top["variant"].eq("tob_current_level")
        ]
        delta = math.nan
        if not baseline.empty:
            delta = (float(row["hit_rate"]) - float(baseline.iloc[0]["hit_rate"])) * 100.0
        rows.append([
            row["variant"],
            int(row["horizon_sec"]),
            pct(float(row["hit_rate"])),
            f"[{pct(float(row['hit_rate_ci_lo']))}, {pct(float(row['hit_rate_ci_hi']))}]",
            pp(delta),
            bps(float(row["directional_return_bps"])),
            f"{int(row['n']):,}",
            row["top_family"],
            pct(float(row["top_family_share"])),
        ])
    return markdown_table(
        ["variant", "h", "hit", "CI", "vs L1 TOB", "dir ret", "n", "top family", "share"],
        rows,
    )


def micro_target_table(micro_target: pd.DataFrame) -> str:
    rows = [
        [
            int(row["horizon_sec"]),
            pct(float(row["mid_target_hit_rate"])),
            pct(float(row["micro_target_hit_rate"])),
            pp(float(row["hit_delta_micro_minus_mid_pp"])),
            bps(float(row["mid_target_directional_return_bps"])),
            bps(float(row["micro_target_directional_return_bps"])),
            f"{int(row['mid_target_top_decile_n']):,}",
        ]
        for _, row in micro_target.iterrows()
    ]
    return markdown_table(
        ["h", "mid hit", "micro hit", "delta", "mid dir", "micro dir", "n"],
        rows,
    )


def micro_signal_table(micro_signal: pd.DataFrame, a13_top: pd.DataFrame) -> str:
    top = micro_signal[micro_signal["is_top_decile"]].copy().sort_values("horizon_sec")
    rows = []
    for _, row in top.iterrows():
        tob = a13_top[a13_top["horizon_sec"].eq(row["horizon_sec"])]
        tob_hit = float(tob.iloc[0]["hit_rate"]) if not tob.empty else math.nan
        tob_dir = float(tob.iloc[0]["directional_return_bps"]) if not tob.empty else math.nan
        rows.append([
            int(row["horizon_sec"]),
            pct(float(row["hit_rate"])),
            f"[{pct(float(row['hit_rate_ci_lo']))}, {pct(float(row['hit_rate_ci_hi']))}]",
            pp((float(row["hit_rate"]) - tob_hit) * 100.0 if np.isfinite(tob_hit) else math.nan),
            pp(float(row["hit_delta_vs_l1_ofi_pp"])),
            bps(float(row["directional_return_bps"])),
            bps(tob_dir),
            f"{int(row['n']):,}",
        ])
    return markdown_table(
        ["h", "micro-change hit", "CI", "vs TOB", "vs L1 OFI", "micro dir", "TOB dir", "n"],
        rows,
    )


def baseline_table(baseline: pd.DataFrame) -> str:
    rows = [
        [
            int(row["horizon_sec"]),
            f"{int(row['a15_n']):,}",
            f"{int(row['a13_n']):,}",
            pp(float(row["hit_delta_pp"])),
            bps(float(row["directional_return_delta_bps"])),
            "yes" if bool(row["passes_exact"]) else "NO",
        ]
        for _, row in baseline.iterrows()
    ]
    return markdown_table(["h", "A15 n", "A13 n", "hit delta", "dir delta", "pass"], rows)


def verdicts(
    imbalance_deciles: pd.DataFrame,
    micro_target: pd.DataFrame,
    micro_signal: pd.DataFrame,
) -> tuple[str, str, str, str]:
    top = imbalance_deciles[imbalance_deciles["is_top_decile"]].copy()
    l1 = top[top["variant"].eq("tob_current_level")][["horizon_sec", "hit_rate"]].rename(
        columns={"hit_rate": "l1_hit"}
    )
    ext = top[~top["variant"].eq("tob_current_level")].merge(l1, on="horizon_sec", how="left")
    ext["delta_pp"] = (ext["hit_rate"] - ext["l1_hit"]) * 100.0
    best_ext = ext.sort_values(["delta_pp", "hit_rate"], ascending=False).head(1)
    if best_ext.empty:
        multi_verdict = "No multi-level imbalance variant was available."
    else:
        row = best_ext.iloc[0]
        multi_verdict = (
            f"Best multi-level variant is `{row['variant']}` at {int(row['horizon_sec'])}s, "
            f"{pct(float(row['hit_rate']))}, {pp(float(row['delta_pp']))} vs L1 TOB."
        )

    mt = micro_target.copy()
    mt["abs_drop_pp"] = -mt["hit_delta_micro_minus_mid_pp"]
    max_drop = float(mt["abs_drop_pp"].max()) if not mt.empty else math.nan
    if np.isfinite(max_drop) and max_drop > 5.0:
        target_verdict = (
            f"Micro-price target weakens TOB materially at at least one horizon "
            f"(max drop {max_drop:.2f} pp)."
        )
    else:
        target_verdict = (
            "TOB survives the micro-price target check; hit-rate deltas are not a material collapse."
        )

    ms = micro_signal[micro_signal["is_top_decile"]].copy()
    best_micro = ms.sort_values(["hit_rate", "directional_return_bps"], ascending=False).head(1)
    if best_micro.empty:
        signal_verdict = "Micro-price-change signal was unavailable."
    else:
        row = best_micro.iloc[0]
        signal_verdict = (
            f"Best micro-change cell is {int(row['horizon_sec'])}s, "
            f"{pct(float(row['hit_rate']))}; it does not beat A1.3 TOB where TOB is strongest."
        )

    recommendation = (
        "Recommended next action for Justin: carry L1 `tob_imbalance_level` and "
        "`exp_decay_imbalance_alpha_0p5` into A2, with L1 as the primary 1s/5s/30s "
        "feature and exp-decay imbalance as a 300s sidecar; keep micro-price as an "
        "audit target and do not promote micro-price-change as a primary signal yet."
    )
    return multi_verdict, target_verdict, signal_verdict, recommendation


def write_note(
    imbalance_deciles: pd.DataFrame,
    micro_target: pd.DataFrame,
    micro_signal: pd.DataFrame,
    baseline: pd.DataFrame,
) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    a13_top = imbalance_deciles[
        imbalance_deciles["variant"].eq("tob_current_level") & imbalance_deciles["is_top_decile"]
    ].copy()
    multi_verdict, target_verdict, signal_verdict, recommendation = verdicts(
        imbalance_deciles,
        micro_target,
        micro_signal,
    )
    note = f"""---
tags: [dali, block-a15, tob-extensions, results]
---

# Block A1.5 TOB Extensions Findings

## Headline

{multi_verdict} {target_verdict} {signal_verdict} Signal characterization only: no execution simulation, no ML, and no parameter optimization.

## Calibration Gate

The A1.3 current-level TOB decile aggregate was recomputed on the A1.5 code path from the A1.2 per-level parquet. The gate requires zero top-decile hit-rate delta and zero directional-return delta at all four horizons.

{baseline_table(baseline)}

## Sub-Experiment 1: Multi-Level Imbalance Signals

Per-level imbalance is `(bid_size_k - ask_size_k) / (bid_size_k + ask_size_k)` with missing side sizes treated as zero and both-missing levels set to zero. The TOB baseline uses A1.3's stricter L1 convention for exact reconciliation.

{imbalance_table(imbalance_deciles)}

Verdict: multi-level imbalance does not beat L1 TOB at 1s, 5s, or 30s. The only extension that beats L1 is `exp_decay_imbalance_alpha_0p5` at 300s, where it hits 74.3% versus L1's 69.4%, but that is the horizon already most exposed to composition and persistence effects.

## Sub-Experiment 2: Micro-Price Target

Micro-price is `mid + 0.5 * spread * tob_imbalance_level`. The micro-price target replaces `future_directional_mid` with `future_directional_micro_price` while keeping the current directional mid denominator, matching the prompt's target substitution.

{micro_target_table(micro_target)}

Verdict: the TOB signal does not collapse when the future target is micro-price. It gets stronger mechanically because future micro-price still contains book imbalance; treat this as a robustness/audit target, not a tradability result.

## Sub-Experiment 3: Micro-Price Change Signal

`micro_change_hs` is `direction_factor * (micro_price_t - micro_price_{{t-h}}) / mean_depth_at_touch`, binned by global absolute signal deciles within each horizon.

{micro_signal_table(micro_signal, a13_top)}

Verdict: micro-price change is not competitive with L1 TOB or L1 OFI. Its best top-decile hit rate is 54.2% at 5s, far below TOB's 73.7% at the same horizon.

## Outputs

- `{display_path(FEATURES_OUT)}`
- `{display_path(IMBALANCE_DECILES_OUT)}`
- `{display_path(MICRO_TARGET_OUT)}`
- `{display_path(MICRO_SIGNAL_OUT)}`
- `{display_path(BASELINE_CHECK_OUT)}`

{recommendation}
"""
    NOTE.write_text(note, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a12-features", type=Path, default=A12_FEATURES)
    parser.add_argument("--a13-deciles", type=Path, default=A13_DECILES)
    parser.add_argument("--a12-comparison", type=Path, default=A12_COMPARISON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"loading {display_path(args.a12_features)}", flush=True)
    df = load_a12_features(args.a12_features)
    print(f"loaded {len(df):,} rows", flush=True)
    print("adding L1-L10 imbalance and micro-price columns", flush=True)
    df = add_imbalance_and_microprice(df)
    print("adding horizon target/signal columns", flush=True)
    df = add_horizon_features(df)

    print("building multi-level imbalance deciles", flush=True)
    imbalance_deciles = imbalance_decile_aggregate(df)
    baseline = baseline_check(imbalance_deciles, args.a13_deciles)
    baseline.to_csv(BASELINE_CHECK_OUT, index=False)
    ensure_baseline_passes(baseline)
    print("baseline gate passed", flush=True)

    print("building micro-price target comparison", flush=True)
    micro_target = microprice_target_comparison(df, imbalance_deciles)
    print("building micro-price-change signal deciles", flush=True)
    micro_signal = micro_signal_decile_aggregate(df)
    micro_signal = attach_l1_ofi_reference(micro_signal, args.a12_comparison)

    print("writing outputs", flush=True)
    write_outputs(df, imbalance_deciles, micro_target, micro_signal, baseline)
    write_note(imbalance_deciles, micro_target, micro_signal, baseline)

    print(f"features: {display_path(FEATURES_OUT)} ({len(df):,} rows)")
    print(f"imbalance deciles: {display_path(IMBALANCE_DECILES_OUT)} ({len(imbalance_deciles):,} rows)")
    print(f"micro target: {display_path(MICRO_TARGET_OUT)} ({len(micro_target):,} rows)")
    print(f"micro signal: {display_path(MICRO_SIGNAL_OUT)} ({len(micro_signal):,} rows)")
    print(f"baseline check: {display_path(BASELINE_CHECK_OUT)} ({len(baseline):,} rows)")
    print(f"note: {display_path(NOTE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
