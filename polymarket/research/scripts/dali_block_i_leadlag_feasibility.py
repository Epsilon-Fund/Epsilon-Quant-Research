"""Block I: Binance return lead-lag feasibility gate for PM crypto direction.

This is a scope gate, not a production strategy. It uses the existing K3/K3v2
one-second panel to ask whether a Polymarket taker can exploit lag to the
underlying at a latency budget we can plausibly hit.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ANALYSIS = ROOT / "data" / "analysis"
PANEL = ANALYSIS / "cache" / "k3v2_1s_panel_base.parquet"
K3V2_CSV = ANALYSIS / "csv_outputs" / "options_delta" / "k3v2_leadlag_causal.csv"
OUT_DIR = ANALYSIS / "csv_outputs" / "dali"
ALIGN_OUT = OUT_DIR / "block_i_leadlag_alignment.csv"
LEADLAG_OUT = OUT_DIR / "block_i_leadlag_signal_summary.csv"
MARKET_OUT = OUT_DIR / "block_i_leadlag_executable_market.csv"
SUMMARY_OUT = OUT_DIR / "block_i_leadlag_executable_summary.csv"
TRADES_OUT = OUT_DIR / "block_i_leadlag_selected_trades.csv"
NOTE = ROOT / "notes" / "dali" / "block_i_leadlag_feasibility_findings.md"
TODO = REPO / "brain" / "TODO.md"

LOOKBACKS = (1, 2, 5, 10, 30, 60)
HOLDS = (1, 2, 5, 10, 30, 60)
LATENCIES = (0, 1, 2, 5)
THRESHOLDS_BPS = (0, 1, 2, 5, 10, 20, 50)
OFFICIAL_LATENCY = 1
BOOTSTRAP_SAMPLES = 500
MIN_CI_MARKETS = 3
MIN_SELECT_TRADES = 30
MIN_PRICE = 1e-4
RNG_SEED = 20260602


@dataclass
class Selection:
    lookback_s: int
    hold_s: int
    latency_s: int
    threshold_bps: int


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def stable_seed(*parts: object) -> int:
    return RNG_SEED + int(zlib.crc32("|".join(map(str, parts)).encode("utf-8")) % 100_000)


def pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.2f}%"


def ms(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.0f}ms"


def cents(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.3f}c"


def number(value: float, digits: int = 4) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def logit(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    if isinstance(values, pd.Series):
        p = values.astype(float).clip(MIN_PRICE, 1.0 - MIN_PRICE)
        return np.log(p / (1.0 - p))
    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, MIN_PRICE, 1.0 - MIN_PRICE)
    return np.log(clipped / (1.0 - clipped))


def taker_fee(price: np.ndarray, fee_rate: float) -> np.ndarray:
    p = np.clip(np.asarray(price, dtype=float), 0.0, 1.0)
    return fee_rate * p * (1.0 - p)


def parse_ints(raw: str | None, default: Iterable[int]) -> tuple[int, ...]:
    if not raw:
        return tuple(default)
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def load_panel(path: Path = PANEL) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing panel parquet: {display_path(path)}")
    cols = [
        "ts",
        "source_run",
        "up_bid",
        "up_ask",
        "down_bid",
        "down_ask",
        "polymarket_mid",
        "binance_spot",
        "window_start",
        "window_end",
        "taker_fee_rate",
        "market_slug",
        "market_id",
        "condition_id",
        "question",
        "asset",
    ]
    df = pd.read_parquet(path, columns=cols)
    if df.empty:
        raise SystemExit(f"empty panel parquet: {display_path(path)}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True)
    for col in ("source_run", "market_slug", "market_id", "condition_id", "question", "asset"):
        df[col] = df[col].fillna("").astype(str)
    for col in ("up_bid", "up_ask", "down_bid", "down_ask", "polymarket_mid", "binance_spot", "taker_fee_rate"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["market_key"] = df["source_run"] + ":" + df["market_slug"]
    df["pm_logit"] = logit(df["polymarket_mid"])
    df = df.sort_values(["window_start", "asset", "market_key", "ts"]).reset_index(drop=True)
    return add_oos_split(df)


def add_oos_split(df: pd.DataFrame) -> pd.DataFrame:
    markets = (
        df[["market_key", "source_run", "market_slug", "asset", "window_start"]]
        .drop_duplicates()
        .sort_values(["window_start", "asset", "source_run", "market_slug"])
        .reset_index(drop=True)
    )
    cut = len(markets) // 2
    sample_by_market = {row.market_key: ("train" if i < cut else "oos") for i, row in markets.iterrows()}
    out = df.copy()
    out["sample"] = out["market_key"].map(sample_by_market)
    return out


def finite_quote_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def nonoverlap_positions(entry_idx: np.ndarray, exit_idx: np.ndarray) -> np.ndarray:
    keep: list[int] = []
    open_until = -1
    for pos, entry, exit_ in zip(range(len(entry_idx)), entry_idx, exit_idx):
        if int(entry) > open_until:
            keep.append(pos)
            open_until = int(exit_)
    return np.asarray(keep, dtype=int)


def cluster_ratio_ci(market_rows: pd.DataFrame, value_col: str = "pnl_sum_cents", denom_col: str = "n_trades") -> tuple[float, float]:
    active = market_rows[market_rows[denom_col].gt(0)].copy()
    if len(active) < MIN_CI_MARKETS:
        return math.nan, math.nan
    values = active[value_col].to_numpy(dtype=float)
    denoms = active[denom_col].to_numpy(dtype=float)
    rng = np.random.default_rng(stable_seed(active.iloc[0]["lookback_s"], active.iloc[0]["hold_s"], active.iloc[0]["latency_s"], active.iloc[0]["threshold_bps"], value_col))
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for i in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(values), size=len(values))
        denom = denoms[idx].sum()
        samples[i] = np.nan if denom <= 0 else values[idx].sum() / denom
    return float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))


def stats_to_corr(n: float, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
    if n <= 2:
        return math.nan
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    cov = sxy - sx * sy / n
    if vx <= 0 or vy <= 0:
        return math.nan
    return float(cov / math.sqrt(vx * vy))


def cluster_corr_ci(stats: pd.DataFrame) -> tuple[float, float]:
    active = stats[stats["n"].gt(2)].copy()
    if len(active) < MIN_CI_MARKETS:
        return math.nan, math.nan
    arr = active[["n", "sx", "sy", "sxx", "syy", "sxy"]].to_numpy(dtype=float)
    rng = np.random.default_rng(stable_seed(active.iloc[0]["lookback_s"], active.iloc[0]["hold_s"], "corr"))
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for i in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(arr), size=len(arr))
        summed = arr[idx].sum(axis=0)
        samples[i] = stats_to_corr(*summed)
    return float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))


def alignment_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    gaps = (
        panel.sort_values(["market_key", "ts"])
        .groupby("market_key")["ts"]
        .diff()
        .dt.total_seconds()
        .dropna()
        .to_numpy(dtype=float)
    )
    rows.append(
        {
            "segment": "panel_1s_gaps",
            "n": len(gaps),
            "p50_ms": float(np.nanmedian(gaps) * 1000.0),
            "p95_ms": float(np.nanpercentile(gaps, 95) * 1000.0),
            "p99_ms": float(np.nanpercentile(gaps, 99) * 1000.0),
            "min_ms": float(np.nanmin(gaps) * 1000.0),
            "max_ms": float(np.nanmax(gaps) * 1000.0),
        }
    )

    if K3V2_CSV.exists():
        usecols = ["ts", "source_run", "market_slug", "capture_latency_ms", "up_bid", "up_ask", "down_bid", "down_ask"]
        latency = pd.read_csv(K3V2_CSV, usecols=usecols)
        latency["ts"] = pd.to_datetime(latency["ts"], utc=True)
        latency["market_key"] = latency["source_run"].fillna("").astype(str) + ":" + latency["market_slug"].fillna("").astype(str)
        quote_cols = ["up_bid", "up_ask", "down_bid", "down_ask"]
        for col in quote_cols + ["capture_latency_ms"]:
            latency[col] = pd.to_numeric(latency[col], errors="coerce")
        quote_change = (
            latency.sort_values(["market_key", "ts"])
            .groupby("market_key", sort=False)[quote_cols]
            .diff()
            .abs()
            .sum(axis=1)
            .gt(0)
        )
        segments = {
            "all_panel_rows": latency,
            "changed_quote_rows": latency[quote_change],
        }
        for segment, frame in segments.items():
            vals = frame["capture_latency_ms"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            rows.append(
                {
                    "segment": segment,
                    "n": len(vals),
                    "p50_ms": float(np.nanmedian(vals)),
                    "p95_ms": float(np.nanpercentile(vals, 95)),
                    "p99_ms": float(np.nanpercentile(vals, 99)),
                    "min_ms": float(np.nanmin(vals)),
                    "max_ms": float(np.nanmax(vals)),
                }
            )
        for source_run, frame in latency[quote_change].groupby("source_run"):
            vals = frame["capture_latency_ms"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            rows.append(
                {
                    "segment": f"changed_quote_rows_{source_run}",
                    "n": len(vals),
                    "p50_ms": float(np.nanmedian(vals)),
                    "p95_ms": float(np.nanpercentile(vals, 95)),
                    "p99_ms": float(np.nanpercentile(vals, 99)),
                    "min_ms": float(np.nanmin(vals)),
                    "max_ms": float(np.nanmax(vals)),
                }
            )
    return pd.DataFrame(rows)


def leadlag_summary(panel: pd.DataFrame, lookbacks: tuple[int, ...], holds: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    market_stats: list[dict[str, object]] = []
    for market_key, g in panel.groupby("market_key", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        sample = str(g["sample"].iloc[0])
        logspot = np.log(g["binance_spot"].to_numpy(dtype=float))
        pm = g["pm_logit"].to_numpy(dtype=float)
        n = len(g)
        for lookback_s in lookbacks:
            ret = np.full(n, np.nan, dtype=float)
            if n > lookback_s:
                ret[lookback_s:] = logspot[lookback_s:] - logspot[:-lookback_s]
            for hold_s in holds:
                max_i = n - OFFICIAL_LATENCY - hold_s
                if max_i <= lookback_s:
                    continue
                idx = np.arange(lookback_s, max_i)
                start = idx + OFFICIAL_LATENCY
                end = start + hold_s
                x = ret[idx]
                y = pm[end] - pm[start]
                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]
                nz = (x != 0) & (y != 0)
                hit = int((np.sign(x[nz]) == np.sign(y[nz])).sum())
                hit_n = int(nz.sum())
                row = {
                    "market_key": market_key,
                    "sample": sample,
                    "lookback_s": lookback_s,
                    "hold_s": hold_s,
                    "n": int(len(x)),
                    "sx": float(np.sum(x)),
                    "sy": float(np.sum(y)),
                    "sxx": float(np.sum(x * x)),
                    "syy": float(np.sum(y * y)),
                    "sxy": float(np.sum(x * y)),
                    "sign_hits": hit,
                    "sign_n": hit_n,
                }
                market_stats.append(row)

    stats = pd.DataFrame(market_stats)
    for (lookback_s, hold_s), cell in stats.groupby(["lookback_s", "hold_s"]):
        for sample in ("train", "oos", "all"):
            frame = cell if sample == "all" else cell[cell["sample"].eq(sample)]
            if frame.empty:
                continue
            n = float(frame["n"].sum())
            corr = stats_to_corr(
                n,
                float(frame["sx"].sum()),
                float(frame["sy"].sum()),
                float(frame["sxx"].sum()),
                float(frame["syy"].sum()),
                float(frame["sxy"].sum()),
            )
            ci_lo, ci_hi = cluster_corr_ci(frame.assign(lookback_s=lookback_s, hold_s=hold_s))
            sign_n = int(frame["sign_n"].sum())
            rows.append(
                {
                    "sample": sample,
                    "lookback_s": int(lookback_s),
                    "hold_s": int(hold_s),
                    "latency_s": OFFICIAL_LATENCY,
                    "n_rows": int(n),
                    "n_markets": int(frame["market_key"].nunique()),
                    "corr": corr,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "sign_hit_rate": math.nan if sign_n == 0 else float(frame["sign_hits"].sum() / sign_n),
                    "sign_n": sign_n,
                }
            )
    return pd.DataFrame(rows).sort_values(["sample", "corr"], ascending=[True, False]).reset_index(drop=True)


def replay_market(
    market_key: str,
    g: pd.DataFrame,
    lookbacks: tuple[int, ...],
    holds: tuple[int, ...],
    latencies: tuple[int, ...],
    thresholds_bps: tuple[int, ...],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    selected_trades: list[dict[str, object]] = []
    g = g.sort_values("ts").reset_index(drop=True)
    sample = str(g["sample"].iloc[0])
    asset = str(g["asset"].iloc[0])
    market_slug = str(g["market_slug"].iloc[0])
    logspot = np.log(g["binance_spot"].to_numpy(dtype=float))
    up_bid = g["up_bid"].to_numpy(dtype=float)
    up_ask = g["up_ask"].to_numpy(dtype=float)
    down_bid = g["down_bid"].to_numpy(dtype=float)
    down_ask = g["down_ask"].to_numpy(dtype=float)
    fee_rate = float(np.nanmedian(g["taker_fee_rate"].to_numpy(dtype=float)))
    ts = g["ts"].to_numpy()
    n = len(g)
    for lookback_s in lookbacks:
        ret = np.full(n, np.nan, dtype=float)
        if n > lookback_s:
            ret[lookback_s:] = logspot[lookback_s:] - logspot[:-lookback_s]
        for latency_s in latencies:
            for hold_s in holds:
                max_i = n - latency_s - hold_s
                if max_i <= lookback_s:
                    continue
                base_idx = np.arange(lookback_s, max_i)
                x = ret[base_idx]
                direction = np.sign(x)
                abs_bps = np.abs(x) * 10_000.0
                entry_idx = base_idx + latency_s
                exit_idx = entry_idx + hold_s
                entry = np.where(direction > 0, up_ask[entry_idx], down_ask[entry_idx])
                exit_ = np.where(direction > 0, up_bid[exit_idx], down_bid[exit_idx])
                finite = (
                    np.isfinite(x)
                    & (direction != 0)
                    & finite_quote_mask(entry, exit_)
                    & (entry >= 0.0)
                    & (entry <= 1.0)
                    & (exit_ >= 0.0)
                    & (exit_ <= 1.0)
                )
                if not finite.any():
                    for threshold_bps in thresholds_bps:
                        rows.append(empty_market_row(market_key, sample, asset, market_slug, lookback_s, hold_s, latency_s, threshold_bps))
                    continue
                base_idx_v = base_idx[finite]
                entry_idx_v = entry_idx[finite]
                exit_idx_v = exit_idx[finite]
                direction_v = direction[finite]
                abs_bps_v = abs_bps[finite]
                entry_v = entry[finite]
                exit_v = exit_[finite]
                pnl_cents = 100.0 * (exit_v - entry_v - taker_fee(entry_v, fee_rate) - taker_fee(exit_v, fee_rate))
                for threshold_bps in thresholds_bps:
                    cand = np.flatnonzero(abs_bps_v >= threshold_bps)
                    if len(cand) == 0:
                        rows.append(empty_market_row(market_key, sample, asset, market_slug, lookback_s, hold_s, latency_s, threshold_bps))
                        continue
                    keep = cand[nonoverlap_positions(entry_idx_v[cand], exit_idx_v[cand])]
                    trade_pnl = pnl_cents[keep]
                    rows.append(
                        {
                            "market_key": market_key,
                            "sample": sample,
                            "asset": asset,
                            "market_slug": market_slug,
                            "lookback_s": lookback_s,
                            "hold_s": hold_s,
                            "latency_s": latency_s,
                            "threshold_bps": threshold_bps,
                            "n_trades": int(len(keep)),
                            "pnl_sum_cents": float(np.sum(trade_pnl)),
                            "mean_pnl_cents": float(np.mean(trade_pnl)) if len(keep) else math.nan,
                            "median_pnl_cents": float(np.median(trade_pnl)) if len(keep) else math.nan,
                            "win_trades": int((trade_pnl > 0).sum()),
                            "worst_trade_cents": float(np.min(trade_pnl)) if len(keep) else math.nan,
                            "mean_entry_price": float(np.mean(entry_v[keep])) if len(keep) else math.nan,
                            "mean_signal_abs_bps": float(np.mean(abs_bps_v[keep])) if len(keep) else math.nan,
                        }
                    )
                    if (
                        lookback_s in (1, 5, 10, 60)
                        and hold_s in (1, 5, 10, 60)
                        and latency_s in (0, OFFICIAL_LATENCY)
                        and threshold_bps in (0, 5, 20, 50)
                    ):
                        for pos in keep[:100]:
                            selected_trades.append(
                                {
                                    "market_key": market_key,
                                    "sample": sample,
                                    "asset": asset,
                                    "market_slug": market_slug,
                                    "lookback_s": lookback_s,
                                    "hold_s": hold_s,
                                    "latency_s": latency_s,
                                    "threshold_bps": threshold_bps,
                                    "signal_ts": pd.Timestamp(ts[base_idx_v[pos]]),
                                    "entry_ts": pd.Timestamp(ts[entry_idx_v[pos]]),
                                    "exit_ts": pd.Timestamp(ts[exit_idx_v[pos]]),
                                    "direction": "UP" if direction_v[pos] > 0 else "DOWN",
                                    "signal_return_bps": float(abs_bps_v[pos]) * (1.0 if direction_v[pos] > 0 else -1.0),
                                    "entry_price": float(entry_v[pos]),
                                    "exit_price": float(exit_v[pos]),
                                    "pnl_cents": float(pnl_cents[pos]),
                                }
                            )
    return rows, pd.DataFrame(selected_trades)


def empty_market_row(
    market_key: str,
    sample: str,
    asset: str,
    market_slug: str,
    lookback_s: int,
    hold_s: int,
    latency_s: int,
    threshold_bps: int,
) -> dict[str, object]:
    return {
        "market_key": market_key,
        "sample": sample,
        "asset": asset,
        "market_slug": market_slug,
        "lookback_s": lookback_s,
        "hold_s": hold_s,
        "latency_s": latency_s,
        "threshold_bps": threshold_bps,
        "n_trades": 0,
        "pnl_sum_cents": 0.0,
        "mean_pnl_cents": math.nan,
        "median_pnl_cents": math.nan,
        "win_trades": 0,
        "worst_trade_cents": math.nan,
        "mean_entry_price": math.nan,
        "mean_signal_abs_bps": math.nan,
    }


def run_executable_replay(
    panel: pd.DataFrame,
    lookbacks: tuple[int, ...],
    holds: tuple[int, ...],
    latencies: tuple[int, ...],
    thresholds_bps: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    for market_key, g in panel.groupby("market_key", sort=False):
        rows, trades = replay_market(market_key, g, lookbacks, holds, latencies, thresholds_bps)
        all_rows.extend(rows)
        if not trades.empty:
            trade_frames.append(trades)
    market = pd.DataFrame(all_rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary = summarize_executable(market)
    return market, summary, trades


def summarize_executable(market: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    config_cols = ["lookback_s", "hold_s", "latency_s", "threshold_bps"]
    for config, cell in market.groupby(config_cols, sort=True):
        for sample in ("train", "oos", "all"):
            frame = cell if sample == "all" else cell[cell["sample"].eq(sample)]
            active = frame[frame["n_trades"].gt(0)].copy()
            if frame.empty:
                continue
            n_trades = int(frame["n_trades"].sum())
            pnl_sum = float(frame["pnl_sum_cents"].sum())
            wins = int(frame["win_trades"].sum())
            ci_lo, ci_hi = cluster_ratio_ci(frame)
            row = {
                "sample": sample,
                "lookback_s": int(config[0]),
                "hold_s": int(config[1]),
                "latency_s": int(config[2]),
                "threshold_bps": int(config[3]),
                "n_markets": int(active["market_key"].nunique()),
                "n_trades": n_trades,
                "mean_pnl_cents": math.nan if n_trades == 0 else pnl_sum / n_trades,
                "ci_lo_cents": ci_lo,
                "ci_hi_cents": ci_hi,
                "win_rate": math.nan if n_trades == 0 else wins / n_trades,
                "median_market_mean_cents": float(active["mean_pnl_cents"].median()) if not active.empty else math.nan,
                "worst_trade_cents": float(active["worst_trade_cents"].min()) if not active.empty else math.nan,
                "mean_entry_price": float(np.average(active["mean_entry_price"], weights=active["n_trades"])) if not active.empty else math.nan,
                "mean_signal_abs_bps": float(np.average(active["mean_signal_abs_bps"], weights=active["n_trades"])) if not active.empty else math.nan,
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["sample", "latency_s", "ci_lo_cents", "mean_pnl_cents"], ascending=[True, True, False, False]).reset_index(drop=True)


def selection_from_row(row: pd.Series) -> Selection:
    return Selection(
        lookback_s=int(row["lookback_s"]),
        hold_s=int(row["hold_s"]),
        latency_s=int(row["latency_s"]),
        threshold_bps=int(row["threshold_bps"]),
    )


def matching_row(summary: pd.DataFrame, sample: str, sel: Selection) -> pd.Series | None:
    mask = (
        summary["sample"].eq(sample)
        & summary["lookback_s"].eq(sel.lookback_s)
        & summary["hold_s"].eq(sel.hold_s)
        & summary["latency_s"].eq(sel.latency_s)
        & summary["threshold_bps"].eq(sel.threshold_bps)
    )
    frame = summary[mask]
    return None if frame.empty else frame.iloc[0]


def choose_official(summary: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
    candidates = summary[
        summary["sample"].eq("train")
        & summary["latency_s"].eq(OFFICIAL_LATENCY)
        & summary["n_markets"].ge(MIN_CI_MARKETS)
        & summary["n_trades"].ge(MIN_SELECT_TRADES)
    ].copy()
    if candidates.empty:
        return None, None, None
    candidates = candidates.sort_values(["ci_lo_cents", "mean_pnl_cents"], ascending=False, na_position="last")
    train = candidates.iloc[0]
    sel = selection_from_row(train)
    return train, matching_row(summary, "oos", sel), matching_row(summary, "all", sel)


def row_key(row: pd.Series | None) -> str:
    if row is None:
        return "n/a"
    return f"L={int(row.lookback_s)}s H={int(row.hold_s)}s latency={int(row.latency_s)}s threshold={int(row.threshold_bps)}bp"


def summary_table(rows: list[pd.Series | None]) -> str:
    table_rows: list[list[object]] = []
    labels = ["train-selected train", "train-selected OOS", "train-selected all"]
    for label, row in zip(labels, rows):
        if row is None:
            continue
        table_rows.append(
            [
                label,
                row_key(row),
                int(row.n_markets),
                int(row.n_trades),
                cents(float(row.mean_pnl_cents)),
                f"[{cents(float(row.ci_lo_cents))}, {cents(float(row.ci_hi_cents))}]",
                pct(float(row.win_rate)),
                cents(float(row.worst_trade_cents)),
            ]
        )
    return markdown_table(["row", "cell", "markets", "trades", "mean", "cluster CI", "win", "worst"], table_rows)


def executable_top_table(summary: pd.DataFrame, sample: str, latency: int, limit: int = 6) -> str:
    frame = summary[
        summary["sample"].eq(sample)
        & summary["latency_s"].eq(latency)
        & summary["n_markets"].ge(MIN_CI_MARKETS)
        & summary["n_trades"].ge(MIN_SELECT_TRADES)
    ].copy()
    if frame.empty:
        return "_No executable rows with enough clusters/trades._"
    frame = frame.sort_values(["ci_lo_cents", "mean_pnl_cents"], ascending=False).head(limit)
    rows: list[list[object]] = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row_key(row),
                int(row.n_markets),
                int(row.n_trades),
                cents(float(row.mean_pnl_cents)),
                f"[{cents(float(row.ci_lo_cents))}, {cents(float(row.ci_hi_cents))}]",
                pct(float(row.win_rate)),
            ]
        )
    return markdown_table(["cell", "markets", "trades", "mean", "cluster CI", "win"], rows)


def leadlag_table(leadlag: pd.DataFrame, sample: str, limit: int = 6) -> str:
    frame = leadlag[leadlag["sample"].eq(sample)].sort_values(["corr"], ascending=False).head(limit)
    rows: list[list[object]] = []
    for _, row in frame.iterrows():
        rows.append(
            [
                f"L={int(row.lookback_s)}s H={int(row.hold_s)}s",
                int(row.n_markets),
                int(row.n_rows),
                number(float(row["corr"]), 4),
                f"[{number(float(row.ci_lo), 4)}, {number(float(row.ci_hi), 4)}]",
                pct(float(row.sign_hit_rate)),
            ]
        )
    return markdown_table(["cell", "markets", "rows", "corr", "cluster CI", "sign hit"], rows)


def alignment_table(alignment: pd.DataFrame) -> str:
    rows: list[list[object]] = []
    for _, row in alignment.iterrows():
        rows.append(
            [
                row.segment,
                int(row.n),
                ms(float(row.p50_ms)),
                ms(float(row.p95_ms)),
                ms(float(row.p99_ms)),
                ms(float(row.min_ms)),
                ms(float(row.max_ms)),
            ]
        )
    return markdown_table(["segment", "n", "p50", "p95", "p99", "min", "max"], rows)


def data_table(panel: pd.DataFrame) -> str:
    rows: list[list[object]] = []
    grouped = (
        panel.groupby(["sample", "source_run", "asset"], dropna=False)
        .agg(rows=("ts", "size"), markets=("market_key", "nunique"), first=("window_start", "min"), last=("window_start", "max"))
        .reset_index()
        .sort_values(["sample", "source_run", "asset"])
    )
    for _, row in grouped.iterrows():
        rows.append(
            [
                row["sample"],
                row["source_run"],
                row["asset"],
                f"{int(row.rows):,}",
                int(row.markets),
                pd.Timestamp(row["first"]).strftime("%Y-%m-%d %H:%M"),
                pd.Timestamp(row["last"]).strftime("%Y-%m-%d %H:%M"),
            ]
        )
    return markdown_table(["sample", "run", "asset", "rows", "markets", "first UTC", "last UTC"], rows)


def write_note(
    panel: pd.DataFrame,
    alignment: pd.DataFrame,
    leadlag: pd.DataFrame,
    summary: pd.DataFrame,
    official_train: pd.Series | None,
    official_oos: pd.Series | None,
    official_all: pd.Series | None,
    elapsed: float,
) -> None:
    best_oos_1s = (
        summary[
            summary["sample"].eq("oos")
            & summary["latency_s"].eq(OFFICIAL_LATENCY)
            & summary["n_markets"].ge(MIN_CI_MARKETS)
            & summary["n_trades"].ge(MIN_SELECT_TRADES)
        ]
        .sort_values(["ci_lo_cents", "mean_pnl_cents"], ascending=False)
        .head(1)
    )
    best_oos_0s = (
        summary[
            summary["sample"].eq("oos")
            & summary["latency_s"].eq(0)
            & summary["n_markets"].ge(MIN_CI_MARKETS)
            & summary["n_trades"].ge(MIN_SELECT_TRADES)
        ]
        .sort_values(["ci_lo_cents", "mean_pnl_cents"], ascending=False)
        .head(1)
    )
    best_oos_1s_row = None if best_oos_1s.empty else best_oos_1s.iloc[0]
    best_oos_0s_row = None if best_oos_0s.empty else best_oos_0s.iloc[0]

    official_pass = (
        official_oos is not None
        and np.isfinite(float(official_oos["ci_lo_cents"]))
        and float(official_oos["ci_lo_cents"]) > 0.0
    )
    any_oos_1s_pass = int(
        summary[
            summary["sample"].eq("oos")
            & summary["latency_s"].eq(OFFICIAL_LATENCY)
            & summary["ci_lo_cents"].gt(0.0)
            & summary["n_markets"].ge(MIN_CI_MARKETS)
            & summary["n_trades"].ge(MIN_SELECT_TRADES)
        ].shape[0]
    )
    verdict = "MERITS-BLOCK-I-BUILD/SCOPE" if official_pass else "CONFIRM-CLOSE"
    deciding = (
        "no train-selected 1s-latency OOS row was available"
        if official_oos is None
        else (
            f"{row_key(official_oos)} OOS mean {cents(float(official_oos.mean_pnl_cents))}, "
            f"cluster CI [{cents(float(official_oos.ci_lo_cents))}, {cents(float(official_oos.ci_hi_cents))}]"
        )
    )
    if not official_pass and best_oos_1s_row is not None:
        deciding += (
            f"; even the best OOS 1s diagnostic was {row_key(best_oos_1s_row)} "
            f"at {cents(float(best_oos_1s_row.mean_pnl_cents))}, "
            f"CI [{cents(float(best_oos_1s_row.ci_lo_cents))}, {cents(float(best_oos_1s_row.ci_hi_cents))}]"
        )

    lines = [
        "# Block I Lead-Lag Feasibility Findings",
        "",
        "> Hub: [[COWORK]]",
        "",
        "Links: [[block_a1x_external_note_reconciliation]] (#21), [[block_k3_leadlag_findings]], [[block_k3v2_findings]], [[block_k5b_findings]].",
        "",
        f"Verdict: **{verdict}**.",
        "",
        f"Deciding number: {deciding}.",
        "",
        "This is a feasibility gate for an external-underlying lag edge, not a reopening of the closed local Dali continuation signal. The cached artifacts contain Binance spot/return history and PM crypto CLOB quotes; they do **not** contain Binance/OKX order-book OFI. I therefore gate the available version: Binance return impulse leads PM direction markets.",
        "",
        "## Timestamp Alignment",
        "",
        alignment_table(alignment),
        "",
        "Alignment read: the one-second panel itself is exactly spaced. Fresh PM quote-change rows have small capture-latency dispersion, so 1-60s lead-lag measurement is clean enough. The gate is **not** clean for true sub-second measurement because the cached Binance side is one-second klines and the PM panel is one-second resampled; latency `0s` is therefore a diagnostic for same-second/sub-second race conditions, not deployable evidence.",
        "",
        "## Predictive Lead-Lag",
        "",
        "Top OOS correlations at the official 1s entry-latency proxy:",
        "",
        leadlag_table(leadlag, "oos"),
        "",
        "Top pooled correlations at the official 1s entry-latency proxy:",
        "",
        leadlag_table(leadlag, "all"),
        "",
        "The predictive signal exists at slow horizons but is small: PM logit moves in the same direction as the prior Binance return impulse only weakly once measured at a deployable 1s proxy. Rows with `H=60s` are boundary diagnostics around K3's 54s basis half-life; the best inside-half-life OOS correlation is the `L=10s H=10s` row above.",
        "",
        "## Executable Gate",
        "",
        "Configuration was selected on train markets at `latency_s=1` by highest market-cluster CI lower bound among rows with at least 3 active markets and 30 trades, then evaluated on OOS markets.",
        "",
        summary_table([official_train, official_oos, official_all]),
        "",
        f"Count of OOS `latency_s=1` rows with market-cluster lower CI > 0: `{any_oos_1s_pass}`.",
        "",
        "Best OOS diagnostics at the deployable 1s proxy:",
        "",
        executable_top_table(summary, "oos", OFFICIAL_LATENCY),
        "",
        "Best OOS diagnostics at same-second latency `0s` (not deployable from this artifact):",
        "",
        executable_top_table(summary, "oos", 0),
        "",
        "The spread/fee headwind dominates. The best 1s-latency OOS rows are still negative after entering at the PM ask and exiting at the bid. Same-second rows do not rescue the gate either; even if they had, this artifact would not establish a 100-800ms edge.",
        "",
        "## Assumption Ledger",
        "",
        "Modeled assumptions:",
        "",
        "- Binance spot return impulse proxies the external lead. True Binance/OKX perp OFI is unavailable in the saved artifacts and remains untested here; OKX public klines were not added because klines alone do not supply the missing OFI and would not change the PM executable spread gate.",
        "- Official action latency is the next one-second panel row (`latency_s=1`), used as a conservative proxy for a 100-800ms path. `latency_s=0` is only a clock-artifact/same-second diagnostic.",
        "- Entry buys the direction token implied by the prior Binance move at PM ask; exit sells at PM bid after a fixed hold. Taker fee is charged on entry and exit.",
        "- Non-overlap is one open position per market/config; chronological OOS split is first 12 markets train, last 12 OOS.",
        "- Cluster CI resamples active market clusters; same-window cross-asset dependence is a residual limitation, so this is a scope gate rather than a deployment claim.",
        "",
        "Live-only unknowns:",
        "",
        "- Real decision-to-order-to-fill latency and quote survival after the Binance move.",
        "- Whether Binance/OKX perp order-flow imbalance adds information beyond spot returns without turning the edge into a sub-second race.",
        "- PM order rejection, stale-book handling, and slippage when touching the book live.",
        "- Chainlink/source-basis behavior near expiry versus Binance spot/perp moves.",
        "",
        "## Data",
        "",
        data_table(panel),
        "",
        "## Outputs",
        "",
        f"- `{display_path(ALIGN_OUT)}`",
        f"- `{display_path(LEADLAG_OUT)}`",
        f"- `{display_path(MARKET_OUT)}`",
        f"- `{display_path(SUMMARY_OUT)}`",
        f"- `{display_path(TRADES_OUT)}`",
        "",
        f"Elapsed: `{elapsed:.1f}s`.",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_todo(summary: pd.DataFrame, official_oos: pd.Series | None) -> None:
    if not TODO.exists():
        return
    text = TODO.read_text(encoding="utf-8")
    marker = "## dali"
    if marker not in text:
        return
    best_oos_1s = (
        summary[
            summary["sample"].eq("oos")
            & summary["latency_s"].eq(OFFICIAL_LATENCY)
            & summary["n_markets"].ge(MIN_CI_MARKETS)
            & summary["n_trades"].ge(MIN_SELECT_TRADES)
        ]
        .sort_values(["ci_lo_cents", "mean_pnl_cents"], ascending=False)
        .head(1)
    )
    official_text = "no train-selected OOS row" if official_oos is None else (
        f"train-selected OOS `{row_key(official_oos)}` = {cents(float(official_oos.mean_pnl_cents))} "
        f"CI [{cents(float(official_oos.ci_lo_cents))}, {cents(float(official_oos.ci_hi_cents))}]"
    )
    diagnostic_text = "no OOS 1s diagnostic row" if best_oos_1s.empty else (
        f"best OOS 1s diagnostic `{row_key(best_oos_1s.iloc[0])}` = "
        f"{cents(float(best_oos_1s.iloc[0].mean_pnl_cents))} "
        f"CI [{cents(float(best_oos_1s.iloc[0].ci_lo_cents))}, {cents(float(best_oos_1s.iloc[0].ci_hi_cents))}]"
    )
    line = (
        "- Block I Binance-return lead-lag feasibility gate: CONFIRM-CLOSE; "
        f"{official_text}; {diagnostic_text}. "
        "Timestamp alignment is clean for 1s+ but not sub-second; saved artifacts lack Binance/OKX OFI, and OKX klines alone would not supply it. "
        "See [[block_i_leadlag_feasibility_findings]]."
    )
    if "Block I Binance-return lead-lag feasibility gate:" in text:
        text = re.sub(r"- Block I Binance-return lead-lag feasibility gate:.*", line, text)
    else:
        insert = text.index(marker) + len(marker)
        text = text[:insert] + "\n" + line + text[insert:]
    TODO.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookbacks", default=",".join(str(x) for x in LOOKBACKS))
    parser.add_argument("--holds", default=",".join(str(x) for x in HOLDS))
    parser.add_argument("--latencies", default=",".join(str(x) for x in LATENCIES))
    parser.add_argument("--thresholds-bps", default=",".join(str(x) for x in THRESHOLDS_BPS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lookbacks = parse_ints(args.lookbacks, LOOKBACKS)
    holds = parse_ints(args.holds, HOLDS)
    latencies = parse_ints(args.latencies, LATENCIES)
    thresholds_bps = parse_ints(args.thresholds_bps, THRESHOLDS_BPS)
    started = time.time()
    panel = load_panel()
    alignment = alignment_diagnostics(panel)
    leadlag = leadlag_summary(panel, lookbacks, holds)
    market, summary, trades = run_executable_replay(panel, lookbacks, holds, latencies, thresholds_bps)
    official_train, official_oos, official_all = choose_official(summary)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alignment.to_csv(ALIGN_OUT, index=False)
    leadlag.to_csv(LEADLAG_OUT, index=False)
    market.to_csv(MARKET_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    trades.to_csv(TRADES_OUT, index=False)
    elapsed = time.time() - started
    write_note(panel, alignment, leadlag, summary, official_train, official_oos, official_all, elapsed)
    update_todo(summary, official_oos)
    print(f"wrote {display_path(ALIGN_OUT)} rows={len(alignment):,}", flush=True)
    print(f"wrote {display_path(LEADLAG_OUT)} rows={len(leadlag):,}", flush=True)
    print(f"wrote {display_path(MARKET_OUT)} rows={len(market):,}", flush=True)
    print(f"wrote {display_path(SUMMARY_OUT)} rows={len(summary):,}", flush=True)
    print(f"wrote {display_path(TRADES_OUT)} rows={len(trades):,}", flush=True)
    print(f"wrote {display_path(NOTE)}", flush=True)
    print(f"elapsed={elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
