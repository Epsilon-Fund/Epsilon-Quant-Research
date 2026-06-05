"""Block A1.1 segment heatmaps and L2 proxy diagnostics.

This is a sidecar analysis over the A1 feature parquet. It intentionally does
not mutate raw JSONL captures or the canonical Block A1 result files.

The current A1 feature table contains true maintained top-of-book OFI plus
top-5 book-state snapshots. That lets A1.1 test L2 *proxy* questions cheaply:
spread/depth regimes, top-5 imbalance, and top-5 depth-pressure changes. A
true multi-level OFI replay, with one OFI contribution per book level, remains
an A2/A1.1-deep replay task.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
FEATURES = ANALYSIS / "block_a1_features.parquet"
SEGMENT_SURFACE = ANALYSIS / "csv_outputs" / "dali" / "block_a11_segment_surface.csv"
COMPONENT_SWEEP = ANALYSIS / "csv_outputs" / "dali" / "block_a11_ofi_component_sweep.csv"
PLOTS = ANALYSIS / "block_a11_plots"
NOTE = ROOT / "notes" / "block_a11_plan_and_diagnostics.md"

HORIZONS = (1, 5, 10, 30, 300)
RNG_SEED = 20260528
MIN_SEGMENT_ROWS = 100

COMPONENTS = [
    "ofi_combined_mean_depth",
    "ofi_bid_mean_depth",
    "ofi_ask_mean_depth",
    "ofi_combined_instant_depth",
    "top5_depth_pressure_mean_depth",
    "tob_imbalance_level",
    "top5_imbalance_level",
]

SEGMENT_TYPES = [
    "family",
    "market",
    "spread_bucket",
    "relative_depth_bucket",
    "run_id",
    "resolved_in_capture",
    "time_to_resolution_bucket",
]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_slug(raw: object, fallback: str) -> str:
    text = str(raw or fallback).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:90] or fallback


def pct(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{100.0 * x:.1f}%"


def bps(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.1f} bps"


def rel_note_path(path: Path) -> str:
    return str(path.resolve().relative_to(NOTE.parent.resolve().parent))


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
        "book_imbalance_top_n",
        "tob_imbalance",
        "ofi_bid_event",
        "ofi_ask_event",
        "ofi_combined_event",
        "bid_top5_shares",
        "ask_top5_shares",
        "bid_top5_notional",
        "ask_top5_notional",
        "market_resolved_at",
    ]
    select_cols = ", ".join(cols)
    con = duckdb.connect()
    df = con.execute(f"SELECT {select_cols} FROM read_parquet('{path}')").df()
    if df.empty:
        raise SystemExit(f"no feature rows found in {path}")
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    df["market_resolved_at"] = pd.to_datetime(df["market_resolved_at"], utc=True, errors="coerce")
    for col in cols:
        if col in {"run_id", "event_type", "asset_id", "market_id", "family", "slug", "question"}:
            df[col] = df[col].fillna("").astype(str)
    numeric_cols = [
        "outcome_index",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "book_imbalance_top_n",
        "tob_imbalance",
        "ofi_bid_event",
        "ofi_ask_event",
        "ofi_combined_event",
        "bid_top5_shares",
        "ask_top5_shares",
        "bid_top5_notional",
        "ask_top5_notional",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["market_key"] = np.where(df["market_id"].ne(""), df["market_id"], df["asset_id"])
    df["market_label"] = np.where(df["slug"].ne(""), df["slug"], df["market_key"])
    df["touch_depth"] = df[["best_bid_size", "best_ask_size"]].sum(axis=1, min_count=2)
    df["top5_depth"] = df[["bid_top5_shares", "ask_top5_shares"]].sum(axis=1, min_count=2)
    df["spread_bps"] = np.where(
        df["mid"].gt(0) & df["spread"].notna(),
        df["spread"] / df["mid"] * 10_000.0,
        np.nan,
    )
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    df["directional_mid"] = np.where(df["direction_factor"].gt(0), df["mid"], 1.0 - df["mid"])
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
    time_to_res = (df["market_resolved_at"] - df["received_at"]).dt.total_seconds()
    df["time_to_resolution_bucket"] = "unresolved"
    df.loc[time_to_res.notna() & time_to_res.le(60), "time_to_resolution_bucket"] = "resolved_le_1m"
    df.loc[time_to_res.gt(60) & time_to_res.le(300), "time_to_resolution_bucket"] = "resolved_1_5m"
    df.loc[time_to_res.gt(300) & time_to_res.le(1800), "time_to_resolution_bucket"] = "resolved_5_30m"
    df.loc[time_to_res.gt(1800) & time_to_res.le(7200), "time_to_resolution_bucket"] = "resolved_30m_2h"
    df.loc[time_to_res.gt(7200), "time_to_resolution_bucket"] = "resolved_gt_2h"
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
        print(
            f"horizon features {idx:02d}/{len(grouped):02d}: {run_id}/{str(asset_id)[:12]} "
            f"({len(group):,} rows)",
            flush=True,
        )
        g = group.sort_values("received_at").copy()
        for col in (
            "bid_top5_shares",
            "ask_top5_shares",
            "bid_top5_notional",
            "ask_top5_notional",
            "tob_imbalance",
            "book_imbalance_top_n",
        ):
            g[col] = g[col].ffill()
        g["bid_top5_delta_event"] = g["bid_top5_shares"].diff().fillna(0.0)
        g["ask_top5_delta_event"] = g["ask_top5_shares"].diff().fillna(0.0)
        g["top5_depth_pressure_event"] = (
            g["bid_top5_delta_event"].fillna(0.0) - g["ask_top5_delta_event"].fillna(0.0)
        )
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
            ofi_bid = g["ofi_bid_event"].fillna(0.0).rolling(window).sum()
            ofi_ask = g["ofi_ask_event"].fillna(0.0).rolling(window).sum()
            ofi_combined = g["ofi_combined_event"].fillna(0.0).rolling(window).sum()
            top5_pressure = g["top5_depth_pressure_event"].fillna(0.0).rolling(window).sum()
            tob_imb = g["tob_imbalance"].rolling(window).mean()
            top5_imb = g["book_imbalance_top_n"].rolling(window).mean()
            g[f"signal_ofi_combined_mean_depth_{horizon}s"] = (
                g["direction_factor"] * ofi_combined / g["market_mean_depth"]
            )
            g[f"signal_ofi_bid_mean_depth_{horizon}s"] = (
                g["direction_factor"] * ofi_bid / g["market_mean_depth"]
            )
            g[f"signal_ofi_ask_mean_depth_{horizon}s"] = (
                g["direction_factor"] * ofi_ask / g["market_mean_depth"]
            )
            g[f"signal_ofi_combined_instant_depth_{horizon}s"] = (
                g["direction_factor"] * ofi_combined / g["touch_depth"]
            )
            g[f"signal_top5_depth_pressure_mean_depth_{horizon}s"] = (
                g["direction_factor"] * top5_pressure / g["market_mean_depth"]
            )
            g[f"signal_tob_imbalance_level_{horizon}s"] = g["direction_factor"] * tob_imb
            g[f"signal_top5_imbalance_level_{horizon}s"] = g["direction_factor"] * top5_imb
        pieces.append(g.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def assign_deciles(sub: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    out = sub.copy()
    out["abs_signal"] = out[signal_col].abs()
    try:
        out["decile"] = pd.qcut(out["abs_signal"], 10, labels=False, duplicates="drop") + 1
    except ValueError:
        out["decile"] = np.nan
    return out


def summarize_rows(rows: pd.DataFrame, signal_col: str, y_col: str) -> dict[str, object]:
    signs = (np.sign(rows[signal_col]) != 0) & (np.sign(rows[y_col]) != 0)
    hit_rate = (
        float((np.sign(rows.loc[signs, signal_col]) == np.sign(rows.loc[signs, y_col])).mean())
        if signs.any()
        else math.nan
    )
    top_family = ""
    top_family_share = math.nan
    fam = rows["family"].value_counts(normalize=True, dropna=False)
    if len(fam):
        top_family = str(fam.index[0])
        top_family_share = float(fam.iloc[0])
    return {
        "n": int(len(rows)),
        "mean_abs_signal": float(rows[signal_col].abs().mean()),
        "mean_next_mid_return_bps": float(rows[y_col].mean()),
        "directional_return_bps": float((np.sign(rows[signal_col]) * rows[y_col]).mean()),
        "hit_rate": hit_rate,
        "mean_spread_bps": float(rows["spread_bps"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_touch_depth": float(rows["touch_depth"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_relative_depth": float(rows["relative_depth"].replace([np.inf, -np.inf], np.nan).mean()),
        "market_count": int(rows["market_key"].nunique()),
        "top_family": top_family,
        "top_family_share": top_family_share,
    }


def component_sweep(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        y_col = f"future_return_bps_{horizon}s"
        for component in COMPONENTS:
            signal_col = f"signal_{component}_{horizon}s"
            sub = df[
                df["is_book_state_complete"].fillna(False)
                & df[signal_col].replace([np.inf, -np.inf], np.nan).notna()
                & df[y_col].replace([np.inf, -np.inf], np.nan).notna()
                & df[signal_col].ne(0)
            ].copy()
            if len(sub) < 500:
                continue
            sub = assign_deciles(sub, signal_col)
            sub = sub[sub["decile"].notna()]
            for decile, rows in sub.groupby("decile", sort=True):
                summary = summarize_rows(rows, signal_col, y_col)
                out_rows.append({
                    "component": component,
                    "horizon_sec": horizon,
                    "decile": int(decile),
                    **summary,
                })
    return pd.DataFrame(out_rows)


def segment_surface(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        signal_col = f"signal_ofi_combined_mean_depth_{horizon}s"
        y_col = f"future_return_bps_{horizon}s"
        sub = df[
            df["is_book_state_complete"].fillna(False)
            & df[signal_col].replace([np.inf, -np.inf], np.nan).notna()
            & df[y_col].replace([np.inf, -np.inf], np.nan).notna()
            & df[signal_col].ne(0)
        ].copy()
        if len(sub) < 500:
            continue
        sub = assign_deciles(sub, signal_col)
        sub = sub[sub["decile"].notna()]
        for segment_type in SEGMENT_TYPES:
            segment_col = "market_label" if segment_type == "market" else segment_type
            for (segment_value, decile), rows in sub.groupby([segment_col, "decile"], sort=True):
                if len(rows) < MIN_SEGMENT_ROWS:
                    continue
                summary = summarize_rows(rows, signal_col, y_col)
                out_rows.append({
                    "segment_type": segment_type,
                    "segment_value": str(segment_value),
                    "horizon_sec": horizon,
                    "decile": int(decile),
                    "ofi_level": ofi_level(int(decile)),
                    **summary,
                })
    return pd.DataFrame(out_rows)


def ofi_level(decile: int) -> str:
    if decile <= 3:
        return "low"
    if decile <= 7:
        return "middle"
    if decile <= 9:
        return "high"
    return "extreme"


def pivot_for_heatmap(
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
    labels = [str(idx)[:54] for idx in pivot.index]
    return pivot.to_numpy(dtype=float), labels, columns


def plot_heatmap(
    mat: np.ndarray,
    y_labels: list[str],
    horizons: list[int],
    title: str,
    filename: str,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str = "",
) -> Path | None:
    if mat.size == 0 or np.all(~np.isfinite(mat)):
        return None
    PLOTS.mkdir(parents=True, exist_ok=True)
    height = max(4.2, min(13.0, 0.34 * len(y_labels) + 2.4))
    fig, ax = plt.subplots(figsize=(9.5, height))
    im = ax.imshow(mat, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Forward horizon")
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels([f"{h}s" for h in horizons], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    fig.tight_layout()
    out = PLOTS / filename
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_segment_heatmaps(seg: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    top = seg[seg["decile"].eq(10)].copy()
    for segment_type in SEGMENT_TYPES:
        sub = top[top["segment_type"].eq(segment_type)].copy()
        if sub.empty:
            continue
        if segment_type == "market":
            order = (
                sub.groupby("segment_value", as_index=False)["n"].sum()
                .sort_values("n", ascending=False)
                .head(18)["segment_value"]
            )
            sub = sub[sub["segment_value"].isin(order)]
        mat, labels, horizons = pivot_for_heatmap(sub, "segment_value", "horizon_sec", "hit_rate")
        path = plot_heatmap(
            mat,
            labels,
            horizons,
            f"A1.1 top-decile OFI hit rate by {segment_type}",
            f"block_a11_{segment_type}_top_decile_hit_heatmap.png",
            cmap="viridis",
            vmin=0.35,
            vmax=0.75,
            colorbar_label="hit_rate",
        )
        if path:
            paths.append(path)
        n_sub = sub.copy()
        n_sub["log10_n"] = np.log10(n_sub["n"].clip(lower=1))
        mat, labels, horizons = pivot_for_heatmap(n_sub, "segment_value", "horizon_sec", "log10_n")
        path = plot_heatmap(
            mat,
            labels,
            horizons,
            f"A1.1 top-decile row count by {segment_type}",
            f"block_a11_{segment_type}_top_decile_n_heatmap.png",
            cmap="magma",
            colorbar_label="log10(n)",
        )
        if path:
            paths.append(path)
    return paths


def plot_component_heatmaps(comp: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    top = comp[comp["decile"].eq(10)].copy()
    if top.empty:
        return paths
    mat, labels, horizons = pivot_for_heatmap(top, "component", "horizon_sec", "hit_rate")
    path = plot_heatmap(
        mat,
        labels,
        horizons,
        "A1.1 component sweep: top-decile hit rate",
        "block_a11_component_top_decile_hit_heatmap.png",
        cmap="viridis",
        vmin=0.35,
        vmax=0.75,
        colorbar_label="hit_rate",
    )
    if path:
        paths.append(path)
    mat, labels, horizons = pivot_for_heatmap(top, "component", "horizon_sec", "directional_return_bps")
    finite = mat[np.isfinite(mat)]
    vmax = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else None
    path = plot_heatmap(
        mat,
        labels,
        horizons,
        "A1.1 component sweep: top-decile directional return",
        "block_a11_component_top_decile_directional_return_heatmap.png",
        cmap="coolwarm",
        vmin=-vmax if vmax else None,
        vmax=vmax,
        colorbar_label="bps",
    )
    if path:
        paths.append(path)
    return paths


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def top_component_table(comp: pd.DataFrame) -> str:
    top = comp[comp["decile"].eq(10)].copy()
    if top.empty:
        return "_No component rows._"
    top = top.sort_values(["hit_rate", "directional_return_bps"], ascending=False).head(18)
    rows = [
        [
            row["component"],
            int(row["horizon_sec"]),
            pct(float(row["hit_rate"])),
            bps(float(row["directional_return_bps"])),
            f"{int(row['n']):,}",
            row["top_family"],
            pct(float(row["top_family_share"])),
        ]
        for _, row in top.iterrows()
    ]
    return markdown_table(
        ["component", "h", "hit", "dir ret", "n", "top family", "share"],
        rows,
    )


def segment_snapshot_table(seg: pd.DataFrame, segment_type: str, horizon: int = 5) -> str:
    sub = seg[
        seg["segment_type"].eq(segment_type)
        & seg["horizon_sec"].eq(horizon)
        & seg["decile"].eq(10)
    ].copy()
    if sub.empty:
        return "_No segment rows._"
    sub = sub.sort_values(["hit_rate", "n"], ascending=False).head(16)
    rows = [
        [
            str(row["segment_value"])[:64],
            pct(float(row["hit_rate"])),
            bps(float(row["directional_return_bps"])),
            f"{int(row['n']):,}",
            f"{float(row['mean_spread_bps']):.1f}",
            f"{float(row['mean_relative_depth']):.2f}",
        ]
        for _, row in sub.iterrows()
    ]
    return markdown_table(
        [segment_type, "5s hit", "5s dir ret", "n", "spread bps", "rel depth"],
        rows,
    )


def write_note(seg: pd.DataFrame, comp: pd.DataFrame, plot_paths: list[Path]) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    family_hit = segment_snapshot_table(seg, "family", 5)
    market_hit = segment_snapshot_table(seg, "market", 5)
    spread_hit = segment_snapshot_table(seg, "spread_bucket", 5)
    depth_hit = segment_snapshot_table(seg, "relative_depth_bucket", 5)
    run_hit = segment_snapshot_table(seg, "run_id", 5)
    resolved_hit = segment_snapshot_table(seg, "resolved_in_capture", 5)
    time_to_res_hit = segment_snapshot_table(seg, "time_to_resolution_bucket", 5)
    component_hit = top_component_table(comp)
    heatmap_refs = "\n".join(f"![]({rel_note_path(path)})" for path in plot_paths)
    note = f"""---
tags: [dali, block-a1, a1-1, diagnostics]
---

# Block A1.1 Segment And L2 Proxy Diagnostics

## Read

I agree with the A1.1 direction: run segmentation, side-aware cost QA, and OFI component checks before approving A2. The L2 work should start in A1.1 because A0/A0b already contain maintained top-5 depth snapshots. This sidecar uses those snapshots to test whether the A1 signal is a touch-only effect, a broader top-5 depth/imbalance effect, or a spread/depth/regime artifact. It does not replace true multi-level OFI replay; that remains part of A2 or a deeper A1.1 follow-up.

## Outputs

- `{display_path(SEGMENT_SURFACE)}`: segment x horizon x absolute-OFI decile diagnostics.
- `{display_path(COMPONENT_SWEEP)}`: combined, bid-only, ask-only, instant-depth, top-5 pressure, L1 imbalance, and top-5 imbalance sweep.
- `{display_path(PLOTS)}/`: heatmaps by family, market, spread bucket, relative depth bucket, run, resolution status, time-to-resolution, and component.

## What L2 Means Here

Current A1 canonical OFI is L1/top-of-book OFI computed from an L2-maintained executable book. A1.1 adds L2 proxies available in `block_a1_features.parquet`: top-5 bid/ask shares, top-5 notional, and top-5 book imbalance. The `top5_depth_pressure_mean_depth` component is a rolling change in top-5 bid depth minus rolling change in top-5 ask depth, normalized by each market's mean touch depth. Because the A1 parquet does not persist every per-level previous/new pair, this is not full MLOFI.

## Component Sweep

{component_hit}

## Segment Snapshots

### Family

{family_hit}

### Market

{market_hit}

### Spread Bucket

{spread_hit}

### Relative Depth Bucket

{depth_hit}

### Run

{run_hit}

### Resolved In Capture

{resolved_hit}

### Time To Resolution

{time_to_res_hit}

## Heatmaps

{heatmap_refs}

## A2 Plan

1. Capture design: run a 1-2 week VPS capture with active-market reselection, retaining raw `book`, `price_change`, `last_trade_price`, lifecycle, and telemetry messages unchanged. Keep crypto 4h up/down, separate pre-game from in-game sports, include fee-free geopolitics, and add a targeted AI/product sleeve selected by live trade-rate.
2. L2 feature design: replay raw JSONL into a new A2 feature table with per-level OFI columns `bid_ofi_l1..l10`, `ask_ofi_l1..l10`, `combined_ofi_l1..l10`, plus depth-weighted, exponentially weighted, and integrated/PCA-style OFI variants. Keep the current L1 CKS OFI as the baseline.
3. Cost design: replace the old mid-cost overlay with executable scenarios: enter at ask and exit at bid, enter at ask and mark exit at mid, inventory-reduction/sell-at-bid diagnostics, and paired YES/NO complement routes for clean binary markets.
4. Segment design: preserve the A1.1 segment surface schema so A2 can answer which families, spread/depth regimes, and resolution windows actually carry signal after executable costs.
5. Gate: approve capture budget only after A1.1 says whether crypto 5s/10s alpha survives executable entry/exit, whether 300s low-OFI is composition leakage, whether ask-only/top-5 components improve cost-adjusted signal, and which families deserve the panel slots.

Recommended next action for Justin: review the A1.1 segment/component heatmaps, then choose whether to run the deeper raw MLOFI replay before provisioning A2.
"""
    NOTE.write_text(note, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--segment-out", type=Path, default=SEGMENT_SURFACE)
    parser.add_argument("--component-out", type=Path, default=COMPONENT_SWEEP)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"loading {display_path(args.features)}", flush=True)
    df = load_features(args.features)
    print(f"loaded {len(df):,} feature rows", flush=True)
    df = add_horizon_features(df)
    print("building segment surface", flush=True)
    seg = segment_surface(df)
    print("building component sweep", flush=True)
    comp = component_sweep(df)
    args.segment_out.parent.mkdir(parents=True, exist_ok=True)
    seg.to_csv(args.segment_out, index=False)
    comp.to_csv(args.component_out, index=False)
    paths: list[Path] = []
    if not args.skip_plots:
        paths.extend(plot_segment_heatmaps(seg))
        paths.extend(plot_component_heatmaps(comp))
    write_note(seg, comp, paths)
    print(f"features: {display_path(args.features)}")
    print(f"segment surface: {display_path(args.segment_out)} ({len(seg):,} rows)")
    print(f"component sweep: {display_path(args.component_out)} ({len(comp):,} rows)")
    print(f"plots: {display_path(PLOTS)} ({len(paths):,} files)")
    print(f"note: {display_path(NOTE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
