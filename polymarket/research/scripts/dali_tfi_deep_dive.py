"""Block B deep-dive for Dali historical fill-only TFI.

The script reuses the bounded Dali eval/fills outputs already under
``data/analysis``. It deliberately avoids repeated scans of the full historical
trade shards.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/dali_tfi_deep_dive.py
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nbformat as nbf
import numpy as np
import pandas as pd

from data_infra.operator_denylist import OPERATOR_ADDRESSES


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTEBOOKS = ROOT / "notebooks"
FIGS = NOTEBOOKS / "figs"
NOTES = ROOT / "notes"

OUT_SUMMARY = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_deep_dive_summary.csv"
OUT_PER_MARKET = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_per_market_diagnostics.csv"
OUT_PER_MARKET_TOP = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_per_market_top_decile.csv"
OUT_PER_MARKET_HET = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_per_market_heterogeneity_summary.csv"
OUT_WALK_FORWARD = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_walk_forward.csv"
OUT_VOLUME = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_volume_interaction.csv"
OUT_SPORTS = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_sports_explicit.csv"
OUT_SPORTS_LEAGUE = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_sports_league_breakdown.csv"
OUT_OPERATOR = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_operator_filter_comparison.csv"
OUT_NOTE = NOTES / "block_b_findings.md"

BUCKETS = [
    "bottom_decile",
    "bottom_quartile_ex_decile",
    "middle_two_quartiles",
    "top_quartile_ex_decile",
    "top_decile",
]
VOLUME_BUCKETS = [
    "bottom_quartile",
    "middle_two_quartiles",
    "top_quartile",
    "top_decile",
]
EXCLUSION_WINDOWS = [0, 60, 120, 300, 600, 1200, 1800, 3600, 7200]
REVISED_EXCLUSION_WINDOWS = [300, 600, 1800, 3600, 7200]
DEFAULT_HORIZONS = [30, 120, 300]
COST_TICKS = [1, 2, 5, 10]
SIGN_CONVENTIONS = {
    "maker_side": 1.0,
    "inverse_maker_side": -1.0,
}


@dataclass(frozen=True)
class FamilyInput:
    family: str
    label: str
    eval_path: Path
    fills_path: Path
    seconds_path: Path
    candidates_path: Path


DEFAULT_INPUTS = [
    FamilyInput(
        family="daily_crypto_up_down",
        label="crypto",
        eval_path=ANALYSIS / "dali_tfi_crypto_250_eval.parquet",
        fills_path=ANALYSIS / "dali_tfi_crypto_250_fills.parquet",
        seconds_path=ANALYSIS / "dali_tfi_crypto_250_seconds.parquet",
        candidates_path=ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_crypto_250_candidates.csv",
    ),
    FamilyInput(
        family="daily_equity_index",
        label="equity_index",
        eval_path=ANALYSIS / "dali_tfi_equity_index_100_eval.parquet",
        fills_path=ANALYSIS / "dali_tfi_equity_index_100_fills.parquet",
        seconds_path=ANALYSIS / "dali_tfi_equity_index_100_seconds.parquet",
        candidates_path=ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_equity_index_100_candidates.csv",
    ),
    FamilyInput(
        family="ai_product",
        label="ai_product",
        eval_path=ANALYSIS / "dali_tfi_ai_product_100_eval.parquet",
        fills_path=ANALYSIS / "dali_tfi_ai_product_100_fills.parquet",
        seconds_path=ANALYSIS / "dali_tfi_ai_product_100_seconds.parquet",
        candidates_path=ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_ai_product_100_candidates.csv",
    ),
    FamilyInput(
        family="sports_game_lines",
        label="sports",
        eval_path=ANALYSIS / "dali_tfi_sports_100_eval.parquet",
        fills_path=ANALYSIS / "dali_tfi_sports_100_fills.parquet",
        seconds_path=ANALYSIS / "dali_tfi_sports_100_seconds.parquet",
        candidates_path=ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_sports_100_candidates.csv",
    ),
]


def stable_seed(*parts: object) -> int:
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0xFFFF_FFFF


def safe_name(raw: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    return out or "family"


def read_candidates(inp: FamilyInput) -> pd.DataFrame:
    cols = ["market_id", "question", "slug", "end_ts", "n_fills", "usd_volume"]
    df = pd.read_csv(inp.candidates_path, usecols=lambda c: c in cols)
    df["market_id"] = df["market_id"].astype(str)
    df["candidate_end_ts"] = pd.to_datetime(df.pop("end_ts"), utc=True, errors="coerce")
    return df


def read_eval(inp: FamilyInput) -> pd.DataFrame:
    df = pd.read_parquet(inp.eval_path)
    df["family"] = inp.family
    df["family_label"] = inp.label
    df["market_id"] = df["market_id"].astype(str)
    for col in ("second_ts", "target_ts", "end_ts", "future_ts"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df["future_gap_seconds"] = pd.to_numeric(df["future_gap_seconds"], errors="coerce")
    df["signed_maker_usd"] = pd.to_numeric(df["signed_maker_usd"], errors="coerce")
    df["future_price_change"] = pd.to_numeric(df["future_price_change"], errors="coerce")
    df["vwap_price"] = pd.to_numeric(df["vwap_price"], errors="coerce")
    df["abs_signal_usd"] = df["signed_maker_usd"].abs()
    df["seconds_to_end"] = (df["end_ts"] - df["second_ts"]).dt.total_seconds()
    candidates = read_candidates(inp)
    return df.merge(candidates, on="market_id", how="left")


def filter_eval(
    df: pd.DataFrame,
    *,
    exclude_last_seconds: int,
    min_signal_usd: float,
    max_future_gap_seconds: int,
    horizons: Iterable[int],
) -> pd.DataFrame:
    horizons = set(horizons)
    mask = (
        df["horizon_seconds"].isin(horizons)
        & df["future_vwap_price"].notna()
        & df["future_price_change"].notna()
        & df["future_gap_seconds"].le(max_future_gap_seconds)
        & df["signed_maker_usd"].notna()
        & df["signed_maker_usd"].ne(0)
        & df["abs_signal_usd"].ge(min_signal_usd)
        & df["vwap_price"].between(0.01, 0.99)
    )
    has_end = df["end_ts"].notna()
    mask &= (~has_end) | df["future_ts"].le(df["end_ts"])
    mask &= (~has_end) | df["seconds_to_end"].ge(exclude_last_seconds)
    out = df.loc[mask].copy()
    out["exclude_last_seconds"] = int(exclude_last_seconds)
    return out


def assign_magnitude_bucket(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["magnitude_bucket"] = pd.NA
    out["bucket_q10_signal_usd"] = np.nan
    out["bucket_q25_signal_usd"] = np.nan
    out["bucket_q75_signal_usd"] = np.nan
    out["bucket_q90_signal_usd"] = np.nan
    if out.empty:
        return out

    for _, idx in out.groupby(group_cols, dropna=False, sort=False).groups.items():
        vals = out.loc[idx, "abs_signal_usd"].to_numpy(dtype=float)
        if len(vals) == 0 or np.isnan(vals).all():
            continue
        q10, q25, q75, q90 = np.nanquantile(vals, [0.10, 0.25, 0.75, 0.90])
        bucket = np.select(
            [
                vals <= q10,
                vals <= q25,
                vals < q75,
                vals < q90,
                vals >= q90,
            ],
            BUCKETS,
            default="middle_two_quartiles",
        )
        out.loc[idx, "magnitude_bucket"] = bucket
        out.loc[idx, "bucket_q10_signal_usd"] = q10
        out.loc[idx, "bucket_q25_signal_usd"] = q25
        out.loc[idx, "bucket_q75_signal_usd"] = q75
        out.loc[idx, "bucket_q90_signal_usd"] = q90
    return out[out["magnitude_bucket"].notna()].copy()


def bootstrap_ci(
    hit: np.ndarray,
    signed_move_cents: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    if n_boot <= 0 or len(hit) < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    hit = hit.astype(float, copy=False)
    signed_move_cents = signed_move_cents.astype(float, copy=False)
    n = len(hit)
    hit_samples: list[np.ndarray] = []
    mean_samples: list[np.ndarray] = []
    chunk = 50
    for _ in range(0, n_boot, chunk):
        take = min(chunk, n_boot - sum(len(x) for x in hit_samples))
        idx = rng.integers(0, n, size=(take, n), dtype=np.int64)
        hit_samples.append(hit[idx].mean(axis=1))
        mean_samples.append(signed_move_cents[idx].mean(axis=1))
    hit_dist = np.concatenate(hit_samples)
    mean_dist = np.concatenate(mean_samples)
    hit_low, hit_high = np.nanpercentile(hit_dist, [2.5, 97.5])
    mean_low, mean_high = np.nanpercentile(mean_dist, [2.5, 97.5])
    return (100.0 * hit_low, 100.0 * hit_high, mean_low, mean_high)


def metric_rows(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    component: str,
    min_obs: int,
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if df.empty:
        return rows
    grouped = df.groupby(group_cols, dropna=False, sort=True)
    for key, g in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key_tuple, strict=True))
        sign = np.sign(g["signed_maker_usd"].to_numpy(dtype=float))
        future = g["future_price_change"].to_numpy(dtype=float)
        abs_signal = g["abs_signal_usd"].to_numpy(dtype=float)
        gross_usd = g.get("gross_usd", pd.Series(np.nan, index=g.index)).to_numpy(dtype=float)
        vwap_price = g.get("vwap_price", pd.Series(np.nan, index=g.index)).to_numpy(dtype=float)
        future_gap = g.get("future_gap_seconds", pd.Series(np.nan, index=g.index)).to_numpy(dtype=float)
        for convention, mult in SIGN_CONVENTIONS.items():
            signed_move_cents = sign * mult * future * 100.0
            signed_move_cents = signed_move_cents[np.isfinite(signed_move_cents)]
            if len(signed_move_cents) < min_obs:
                continue
            hit = signed_move_cents > 0
            std = float(np.nanstd(signed_move_cents, ddof=1))
            hit_low, hit_high, mean_low, mean_high = bootstrap_ci(
                hit,
                signed_move_cents,
                n_boot=bootstrap_samples,
                seed=stable_seed(component, convention, *key_tuple),
            )
            row: dict[str, object] = {
                "analysis_component": component,
                **base,
                "sign_convention": convention,
                "n_obs": int(len(signed_move_cents)),
                "hit_rate_pct": 100.0 * float(np.nanmean(hit)),
                "hit_rate_ci_low_pct": hit_low,
                "hit_rate_ci_high_pct": hit_high,
                "mean_return_cents": float(np.nanmean(signed_move_cents)),
                "mean_return_ci_low_cents": mean_low,
                "mean_return_ci_high_cents": mean_high,
                "median_return_cents": float(np.nanmedian(signed_move_cents)),
                "p10_signed_move_cents": float(np.nanpercentile(signed_move_cents, 10)),
                "p90_signed_move_cents": float(np.nanpercentile(signed_move_cents, 90)),
                "return_sharpe_like": (
                    float(np.nanmean(signed_move_cents) / std)
                    if std > 0 and math.isfinite(std)
                    else np.nan
                ),
                "avg_abs_signal_usd": float(np.nanmean(abs_signal)),
                "avg_second_gross_usd": float(np.nanmean(gross_usd)),
                "avg_vwap_price": float(np.nanmean(vwap_price)),
                "avg_future_gap_seconds": float(np.nanmean(future_gap)),
            }
            for tick in COST_TICKS:
                row[f"net_edge_after_{tick}tick_cents"] = row["mean_return_cents"] - tick
            rows.append(row)
    return rows


def add_market_metadata(rows: pd.DataFrame, market_meta: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "market_id" not in rows.columns:
        return rows
    keep = ["market_id", "question", "slug", "n_fills", "usd_volume"]
    return rows.merge(market_meta[keep].drop_duplicates("market_id"), on="market_id", how="left")


def build_exclusion_sweep(
    evals: dict[str, pd.DataFrame],
    *,
    min_signal_usd: float,
    max_future_gap_seconds: int,
    horizons: list[int],
    min_obs: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    all_rows: list[dict[str, object]] = []
    for family, df in evals.items():
        family_rows: list[dict[str, object]] = []
        for exclude in EXCLUSION_WINDOWS:
            sub = filter_eval(
                df,
                exclude_last_seconds=exclude,
                min_signal_usd=min_signal_usd,
                max_future_gap_seconds=max_future_gap_seconds,
                horizons=horizons,
            )
            sub = assign_magnitude_bucket(sub, ["family", "horizon_seconds"])
            rows = metric_rows(
                sub,
                group_cols=[
                    "family",
                    "family_label",
                    "exclude_last_seconds",
                    "horizon_seconds",
                    "magnitude_bucket",
                ],
                component="resolution_exclusion_sweep",
                min_obs=min_obs,
                bootstrap_samples=bootstrap_samples,
            )
            family_rows.extend(rows)
        out = pd.DataFrame(family_rows)
        if not out.empty:
            label = str(out["family_label"].dropna().iloc[0])
            out.to_csv(ANALYSIS / f"tfi_exclusion_sweep_{label}.csv", index=False)
            all_rows.extend(family_rows)
    return pd.DataFrame(all_rows)


def build_per_market(
    evals: dict[str, pd.DataFrame],
    market_meta: dict[str, pd.DataFrame],
    *,
    exclude_last_seconds: int,
    min_signal_usd: float,
    max_future_gap_seconds: int,
    horizons: list[int],
    min_obs: int,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diag_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for family, df in evals.items():
        sub = filter_eval(
            df,
            exclude_last_seconds=exclude_last_seconds,
            min_signal_usd=min_signal_usd,
            max_future_gap_seconds=max_future_gap_seconds,
            horizons=horizons,
        )
        diag_rows.extend(
            metric_rows(
                sub,
                group_cols=["family", "family_label", "market_id", "horizon_seconds"],
                component="per_market_diagnostics",
                min_obs=min_obs,
                bootstrap_samples=bootstrap_samples,
            )
        )
        bucketed = assign_magnitude_bucket(sub, ["family", "market_id", "horizon_seconds"])
        top = bucketed[bucketed["magnitude_bucket"].eq("top_decile")]
        top_rows.extend(
            metric_rows(
                top,
                group_cols=["family", "family_label", "market_id", "horizon_seconds"],
                component="per_market_top_decile",
                min_obs=min_obs,
                bootstrap_samples=bootstrap_samples,
            )
        )

    diag = pd.DataFrame(diag_rows)
    top = pd.DataFrame(top_rows)
    if not diag.empty:
        diag = pd.concat(
            [
                add_market_metadata(
                    diag[diag["family"].eq(family)].copy(),
                    market_meta[family],
                )
                for family in sorted(diag["family"].dropna().unique())
            ],
            ignore_index=True,
        )
    if not top.empty:
        top = pd.concat(
            [
                add_market_metadata(
                    top[top["family"].eq(family)].copy(),
                    market_meta[family],
                )
                for family in sorted(top["family"].dropna().unique())
            ],
            ignore_index=True,
        )

    if top.empty:
        return diag, top, pd.DataFrame()
    het = (
        top.assign(
            strong_hit=lambda x: x["hit_rate_pct"].gt(55.0),
            ci_excludes_half=lambda x: x["hit_rate_ci_low_pct"].gt(50.0),
            positive_ev=lambda x: x["mean_return_cents"].gt(0),
            negative_ev=lambda x: x["mean_return_cents"].lt(0),
            positive_after_1tick=lambda x: x["net_edge_after_1tick_cents"].gt(0),
        )
        .groupby(["family", "family_label", "horizon_seconds", "sign_convention"], dropna=False)
        .agg(
            n_markets=("market_id", "nunique"),
            frac_markets_hit_gt_55=("strong_hit", "mean"),
            frac_markets_hit_ci_low_gt_50=("ci_excludes_half", "mean"),
            frac_markets_positive_ev=("positive_ev", "mean"),
            frac_markets_negative_ev=("negative_ev", "mean"),
            frac_markets_positive_after_1tick=("positive_after_1tick", "mean"),
            median_market_hit_rate_pct=("hit_rate_pct", "median"),
            p90_market_hit_rate_pct=("hit_rate_pct", lambda s: s.quantile(0.90)),
            top_market_hit_rate_pct=("hit_rate_pct", "max"),
        )
        .reset_index()
    )
    return diag, top, het


def chronological_split(sub: pd.DataFrame) -> pd.Series:
    split = pd.Series("test", index=sub.index, dtype=object)
    for family, g in sub.groupby("family", sort=False):
        times = np.array(sorted(g["second_ts"].dropna().unique()))
        if len(times) < 3:
            split.loc[g.index] = "train"
            continue
        q60 = times[int(0.60 * (len(times) - 1))]
        q80 = times[int(0.80 * (len(times) - 1))]
        fam_idx = g.index
        split.loc[fam_idx[g["second_ts"].le(q60)]] = "train"
        split.loc[fam_idx[g["second_ts"].gt(q60) & g["second_ts"].le(q80)]] = "validation"
        split.loc[fam_idx[g["second_ts"].gt(q80)]] = "test"
    return split


def train_threshold_buckets(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub.copy()
    out["magnitude_bucket"] = pd.NA
    for (family, horizon), g in out.groupby(["family", "horizon_seconds"], dropna=False, sort=False):
        train_vals = g.loc[g["wf_split"].eq("train"), "abs_signal_usd"].to_numpy(dtype=float)
        if len(train_vals) < 10:
            vals = g["abs_signal_usd"].to_numpy(dtype=float)
        else:
            vals = train_vals
        q10, q25, q75, q90 = np.nanquantile(vals, [0.10, 0.25, 0.75, 0.90])
        idx = g.index
        all_vals = out.loc[idx, "abs_signal_usd"].to_numpy(dtype=float)
        out.loc[idx, "magnitude_bucket"] = np.select(
            [
                all_vals <= q10,
                all_vals <= q25,
                all_vals < q75,
                all_vals < q90,
                all_vals >= q90,
            ],
            BUCKETS,
            default="middle_two_quartiles",
        )
    return out[out["magnitude_bucket"].notna()].copy()


def build_walk_forward(
    evals: dict[str, pd.DataFrame],
    *,
    exclude_last_seconds: int,
    min_signal_usd: float,
    max_future_gap_seconds: int,
    horizons: list[int],
    min_obs: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, df in evals.items():
        sub = filter_eval(
            df,
            exclude_last_seconds=exclude_last_seconds,
            min_signal_usd=min_signal_usd,
            max_future_gap_seconds=max_future_gap_seconds,
            horizons=horizons,
        )
        sub["wf_split"] = chronological_split(sub)
        sub = train_threshold_buckets(sub)
        rows.extend(
            metric_rows(
                sub,
                group_cols=[
                    "family",
                    "family_label",
                    "wf_split",
                    "horizon_seconds",
                    "magnitude_bucket",
                ],
                component="walk_forward_chronological_split",
                min_obs=min_obs,
                bootstrap_samples=bootstrap_samples,
            )
        )
    return pd.DataFrame(rows)


def read_seconds(inp: FamilyInput) -> pd.DataFrame:
    df = pd.read_parquet(inp.seconds_path, columns=["market_id", "second_ts", "n_fills"])
    df["market_id"] = df["market_id"].astype(str)
    df["second_ts"] = pd.to_datetime(df["second_ts"], utc=True, errors="coerce")
    df["n_fills"] = pd.to_numeric(df["n_fills"], errors="coerce").fillna(0)
    return df


def rolling_trade_count(seconds: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    parts = []
    for market_id, g in seconds.sort_values(["market_id", "second_ts"]).groupby("market_id", sort=False):
        s = (
            g.set_index("second_ts")["n_fills"]
            .sort_index()
            .rolling(f"{int(window_seconds)}s", min_periods=1)
            .sum()
        )
        parts.append(
            pd.DataFrame(
                {
                    "market_id": market_id,
                    "second_ts": s.index,
                    "rolling_trade_count": s.to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def expand_volume_buckets(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.groupby(["family", "horizon_seconds"], dropna=False, sort=False):
        vol = g["rolling_trade_count"].to_numpy(dtype=float)
        if len(vol) == 0 or np.isnan(vol).all():
            continue
        q25, q75, q90 = np.nanquantile(vol, [0.25, 0.75, 0.90])
        masks = {
            "bottom_quartile": vol <= q25,
            "middle_two_quartiles": (vol > q25) & (vol < q75),
            "top_quartile": vol >= q75,
            "top_decile": vol >= q90,
        }
        for bucket, mask in masks.items():
            if not mask.any():
                continue
            piece = g.loc[mask].copy()
            piece["volume_bucket"] = bucket
            piece["volume_q25_trade_count"] = q25
            piece["volume_q75_trade_count"] = q75
            piece["volume_q90_trade_count"] = q90
            rows.append(piece)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_volume_interaction(
    inputs: list[FamilyInput],
    evals: dict[str, pd.DataFrame],
    *,
    exclude_last_seconds: int,
    min_signal_usd: float,
    max_future_gap_seconds: int,
    horizons: list[int],
    volume_window_seconds: int,
    min_obs: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_family = {inp.family: inp for inp in inputs}
    for family, df in evals.items():
        inp = by_family[family]
        sub = filter_eval(
            df,
            exclude_last_seconds=exclude_last_seconds,
            min_signal_usd=min_signal_usd,
            max_future_gap_seconds=max_future_gap_seconds,
            horizons=horizons,
        )
        volume = rolling_trade_count(read_seconds(inp), volume_window_seconds)
        sub = sub.merge(volume, on=["market_id", "second_ts"], how="left")
        sub = sub[sub["rolling_trade_count"].notna()].copy()
        sub = expand_volume_buckets(sub)
        sub = assign_magnitude_bucket(sub, ["family", "horizon_seconds", "volume_bucket"])
        rows.extend(
            metric_rows(
                sub,
                group_cols=[
                    "family",
                    "family_label",
                    "horizon_seconds",
                    "volume_bucket",
                    "magnitude_bucket",
                ],
                component="tfi_volume_interaction",
                min_obs=min_obs,
                bootstrap_samples=bootstrap_samples,
            )
        )
    return pd.DataFrame(rows)


def infer_sports_league(question: object, slug: object) -> str:
    text = f"{question or ''} {slug or ''}".lower()
    patterns = [
        ("nba", r"\bnba\b|lakers|cavaliers|raptors|rockets|warriors|knicks|celtics|timberwolves|thunder|pacers|nuggets"),
        ("nhl", r"\bnhl\b|stanley|hockey|oilers|panthers|maple-leafs|leafs|jets|capitals|hurricanes|stars"),
        ("mlb", r"\bmlb\b|baseball|yankees|mets|dodgers|red-sox|cubs|padres|braves|phillies"),
        ("soccer", r"\bfifa\b|world-cup|champions-league|premier-league|laliga|serie-a|mls|soccer|football"),
        ("nfl", r"\bnfl\b|super-bowl|chiefs|eagles|ravens|bills|cowboys"),
        ("tennis", r"tennis|wimbledon|french-open|us-open|australian-open|atp|wta"),
        ("mma_boxing", r"\bufc\b|mma|boxing|fight"),
    ]
    for label, pattern in patterns:
        if re.search(pattern, text):
            return label
    return "other_sports"


def build_sports_explicit(
    sports_eval: pd.DataFrame,
    *,
    exclude_last_seconds: int,
    min_signal_usd: float,
    max_future_gap_seconds: int,
    horizons: list[int],
    min_obs: int,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = filter_eval(
        sports_eval,
        exclude_last_seconds=exclude_last_seconds,
        min_signal_usd=min_signal_usd,
        max_future_gap_seconds=max_future_gap_seconds,
        horizons=horizons,
    )
    sub["sports_league"] = [
        infer_sports_league(q, s) for q, s in zip(sub["question"], sub["slug"], strict=False)
    ]
    sub["sports_game_phase"] = "unknown_start_time"
    bucketed = assign_magnitude_bucket(sub, ["family", "horizon_seconds"])
    explicit = pd.DataFrame(
        metric_rows(
            bucketed,
            group_cols=[
                "family",
                "family_label",
                "horizon_seconds",
                "sports_game_phase",
                "magnitude_bucket",
            ],
            component="sports_explicit",
            min_obs=min_obs,
            bootstrap_samples=bootstrap_samples,
        )
    )
    league_bucketed = assign_magnitude_bucket(
        sub,
        ["family", "sports_league", "horizon_seconds"],
    )
    league = pd.DataFrame(
        metric_rows(
            league_bucketed,
            group_cols=[
                "family",
                "family_label",
                "sports_league",
                "horizon_seconds",
                "magnitude_bucket",
            ],
            component="sports_league_breakdown",
            min_obs=min_obs,
            bootstrap_samples=bootstrap_samples,
        )
    )
    return explicit, league


def build_eval_from_filtered_fills(inp: FamilyInput) -> pd.DataFrame:
    cols = [
        "timestamp",
        "market_id",
        "maker",
        "taker",
        "maker_side",
        "price",
        "usd_amount",
        "transaction_hash",
    ]
    fills = pd.read_parquet(inp.fills_path, columns=cols)
    fills["market_id"] = fills["market_id"].astype(str)
    fills["maker"] = fills["maker"].str.lower()
    fills["taker"] = fills["taker"].str.lower()
    fills = fills[
        ~fills["maker"].isin(OPERATOR_ADDRESSES)
        & ~fills["taker"].isin(OPERATOR_ADDRESSES)
        & fills["price"].notna()
        & fills["usd_amount"].gt(0)
    ].copy()
    fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True, errors="coerce")
    fills["signed_piece"] = np.where(
        fills["maker_side"].eq("BUY"),
        fills["usd_amount"],
        -fills["usd_amount"],
    )
    fills["price_x_usd"] = fills["price"] * fills["usd_amount"]
    bars = (
        fills.groupby(["market_id", "timestamp"], dropna=False)
        .agg(
            n_fills=("price", "size"),
            n_txs=("transaction_hash", "nunique"),
            gross_usd=("usd_amount", "sum"),
            signed_maker_usd=("signed_piece", "sum"),
            price_x_usd=("price_x_usd", "sum"),
        )
        .reset_index()
        .rename(columns={"timestamp": "second_ts"})
    )
    bars["vwap_price"] = bars["price_x_usd"] / bars["gross_usd"]
    candidates = read_candidates(inp)[["market_id", "candidate_end_ts"]]
    bars = bars.merge(candidates, on="market_id", how="left")
    bars = bars.rename(columns={"candidate_end_ts": "end_ts"})
    bars = bars[bars["vwap_price"].between(0.01, 0.99)].sort_values(["market_id", "second_ts"])

    eval_parts = []
    for horizon in DEFAULT_HORIZONS:
        for market_id, g in bars.groupby("market_id", sort=False):
            base = g.copy()
            base["target_ts"] = base["second_ts"] + pd.to_timedelta(horizon, unit="s")
            future = g[["second_ts", "vwap_price"]].rename(
                columns={"second_ts": "future_ts", "vwap_price": "future_vwap_price"}
            )
            merged = pd.merge_asof(
                base.sort_values("target_ts"),
                future.sort_values("future_ts"),
                left_on="target_ts",
                right_on="future_ts",
                direction="forward",
            )
            merged["market_id"] = market_id
            merged["horizon_seconds"] = horizon
            merged["future_gap_seconds"] = (
                merged["future_ts"] - merged["target_ts"]
            ).dt.total_seconds()
            merged["future_price_change"] = (
                merged["future_vwap_price"] - merged["vwap_price"]
            )
            eval_parts.append(merged)
    if not eval_parts:
        return pd.DataFrame()
    out = pd.concat(eval_parts, ignore_index=True)
    out["family"] = inp.family
    out["family_label"] = inp.label
    out["abs_signal_usd"] = out["signed_maker_usd"].abs()
    out["seconds_to_end"] = (out["end_ts"] - out["second_ts"]).dt.total_seconds()
    meta = read_candidates(inp)
    return out.merge(meta, on="market_id", how="left")


def build_operator_comparison(
    inputs: list[FamilyInput],
    evals: dict[str, pd.DataFrame],
    *,
    exclude_last_seconds: int,
    min_signal_usd: float,
    max_future_gap_seconds: int,
    horizons: list[int],
    min_obs: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for inp in inputs:
        all_sub = filter_eval(
            evals[inp.family],
            exclude_last_seconds=exclude_last_seconds,
            min_signal_usd=min_signal_usd,
            max_future_gap_seconds=max_future_gap_seconds,
            horizons=horizons,
        )
        all_sub["operator_filter_state"] = "all_fills"
        all_sub = assign_magnitude_bucket(all_sub, ["family", "horizon_seconds"])
        rows.extend(
            metric_rows(
                all_sub,
                group_cols=[
                    "family",
                    "family_label",
                    "operator_filter_state",
                    "horizon_seconds",
                    "magnitude_bucket",
                ],
                component="operator_filter_comparison",
                min_obs=min_obs,
                bootstrap_samples=bootstrap_samples,
            )
        )

        filtered_eval = build_eval_from_filtered_fills(inp)
        filtered_sub = filter_eval(
            filtered_eval,
            exclude_last_seconds=exclude_last_seconds,
            min_signal_usd=min_signal_usd,
            max_future_gap_seconds=max_future_gap_seconds,
            horizons=horizons,
        )
        filtered_sub["operator_filter_state"] = "operator_removed"
        filtered_sub = assign_magnitude_bucket(filtered_sub, ["family", "horizon_seconds"])
        rows.extend(
            metric_rows(
                filtered_sub,
                group_cols=[
                    "family",
                    "family_label",
                    "operator_filter_state",
                    "horizon_seconds",
                    "magnitude_bucket",
                ],
                component="operator_filter_comparison",
                min_obs=min_obs,
                bootstrap_samples=bootstrap_samples,
            )
        )
    return pd.DataFrame(rows)


def ordered_bucket_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "magnitude_bucket" not in df.columns:
        return df
    out = df.copy()
    out["magnitude_bucket"] = pd.Categorical(out["magnitude_bucket"], BUCKETS, ordered=True)
    return out.sort_values([c for c in ["family", "horizon_seconds", "sign_convention", "magnitude_bucket"] if c in out.columns])


def monotone_top_decile_signals(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for key, g in df.groupby(key_cols, dropna=False):
        ordered = (
            g.set_index("magnitude_bucket")
            .reindex(BUCKETS)
            .dropna(subset=["hit_rate_pct"])
        )
        if "top_decile" not in ordered.index or len(ordered) < 4:
            continue
        hits = ordered["hit_rate_pct"].to_numpy(dtype=float)
        top = ordered.loc["top_decile"]
        is_monotone = bool(np.all(np.diff(hits) >= -0.25))
        row = dict(zip(key_cols, key if isinstance(key, tuple) else (key,), strict=True))
        row.update(
            {
                "monotone_hit_by_magnitude": is_monotone,
                "top_decile_n_obs": int(top["n_obs"]),
                "top_decile_hit_rate_pct": float(top["hit_rate_pct"]),
                "top_decile_hit_ci_low_pct": float(top["hit_rate_ci_low_pct"]),
                "top_decile_mean_return_cents": float(top["mean_return_cents"]),
                "top_decile_net_after_1tick_cents": float(top["net_edge_after_1tick_cents"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_resolution_sweep(exclusion: pd.DataFrame) -> list[Path]:
    if exclusion.empty:
        return []
    import matplotlib.pyplot as plt

    subset = exclusion[
        exclusion["magnitude_bucket"].eq("top_decile")
        & exclusion["horizon_seconds"].eq(300)
    ].copy()
    if subset.empty:
        return []
    out_path = FIGS / "block_b_resolution_sweep_top_decile_300s.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for (family_label, convention), g in subset.groupby(["family_label", "sign_convention"]):
        g = g.sort_values("exclude_last_seconds")
        linestyle = "-" if convention == "inverse_maker_side" else "--"
        label = f"{family_label} | {convention}"
        axes[0].plot(g["exclude_last_seconds"], g["hit_rate_pct"], marker="o", linestyle=linestyle, label=label)
        axes[1].plot(g["exclude_last_seconds"], g["mean_return_cents"], marker="o", linestyle=linestyle, label=label)
    axes[0].axhline(50, color="#777777", linewidth=0.8, linestyle=":")
    axes[1].axhline(0, color="#777777", linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("Top-decile hit rate (%)")
    axes[1].set_ylabel("Top-decile mean EV (cents)")
    axes[1].set_xlabel("Excluded seconds before market end")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return [out_path]


def plot_per_market(top: pd.DataFrame) -> list[Path]:
    if top.empty:
        return []
    import matplotlib.pyplot as plt

    subset = top[top["horizon_seconds"].eq(300)].copy()
    if subset.empty:
        return []
    families = list(subset["family_label"].dropna().drop_duplicates())
    out_path = FIGS / "block_b_per_market_top_decile_hit_distribution_300s.png"
    fig, axes = plt.subplots(2, len(families), figsize=(4 * len(families), 7), sharex=True, sharey=True)
    if len(families) == 1:
        axes = np.array(axes).reshape(2, 1)
    for col, family in enumerate(families):
        for row, convention in enumerate(SIGN_CONVENTIONS):
            ax = axes[row, col]
            vals = subset[
                subset["family_label"].eq(family)
                & subset["sign_convention"].eq(convention)
            ]["hit_rate_pct"]
            ax.hist(vals, bins=np.arange(0, 105, 5), color="#457b9d", alpha=0.8)
            ax.axvline(50, color="#666666", linestyle=":", linewidth=1)
            ax.axvline(55, color="#b05a28", linestyle="--", linewidth=1)
            ax.set_title(f"{family}\n{convention}", fontsize=9)
            ax.grid(axis="y", alpha=0.20)
    fig.supxlabel("Per-market top-decile hit rate (%)")
    fig.supylabel("Market count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return [out_path]


def plot_sports(sports_league: pd.DataFrame) -> list[Path]:
    if sports_league.empty:
        return []
    import matplotlib.pyplot as plt

    subset = sports_league[
        sports_league["magnitude_bucket"].eq("top_decile")
        & sports_league["sign_convention"].eq("inverse_maker_side")
    ].copy()
    if subset.empty:
        return []
    out_path = FIGS / "block_b_sports_league_top_decile_inverse.png"
    pivot = subset.pivot_table(
        index="sports_league",
        columns="horizon_seconds",
        values="hit_rate_pct",
        aggfunc="first",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(50, color="#666666", linestyle=":", linewidth=1)
    ax.axhline(55, color="#b05a28", linestyle="--", linewidth=1)
    ax.set_ylabel("Top-decile hit rate (%)")
    ax.set_xlabel("")
    ax.set_title("Sports league breakdown | inverse maker side")
    ax.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return [out_path]


def plot_volume(volume: pd.DataFrame) -> list[Path]:
    if volume.empty:
        return []
    import matplotlib.pyplot as plt

    subset = volume[
        volume["magnitude_bucket"].eq("top_decile")
        & volume["horizon_seconds"].eq(300)
        & volume["sign_convention"].eq("inverse_maker_side")
    ].copy()
    if subset.empty:
        return []
    subset["volume_bucket"] = pd.Categorical(subset["volume_bucket"], VOLUME_BUCKETS, ordered=True)
    out_path = FIGS / "block_b_volume_top_decile_300s_inverse.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for family, g in subset.groupby("family_label"):
        g = g.sort_values("volume_bucket")
        axes[0].plot(g["volume_bucket"].astype(str), g["hit_rate_pct"], marker="o", label=family)
        axes[1].plot(g["volume_bucket"].astype(str), g["mean_return_cents"], marker="o", label=family)
    axes[0].axhline(50, color="#666666", linestyle=":", linewidth=1)
    axes[0].axhline(55, color="#b05a28", linestyle="--", linewidth=1)
    axes[1].axhline(0, color="#666666", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Top-decile hit rate (%)")
    axes[1].set_ylabel("Top-decile mean EV (cents)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.20)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return [out_path]


def csv_block(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "No rows.\n"
    have = [c for c in cols if c in df.columns]
    return df[have].head(max_rows).round(4).to_csv(index=False)


def notebook_image(path: Path) -> str:
    return f"![{path.stem}]({path.relative_to(NOTEBOOKS)})"


def write_notebook(path: Path, title: str, sections: list[tuple[str, str]], figures: list[Path]) -> None:
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\nGenerated by `scripts/dali_tfi_deep_dive.py`.")]
    for heading, body in sections:
        cells.append(nbf.v4.new_markdown_cell(f"## {heading}\n\n{body}"))
    if figures:
        cells.append(nbf.v4.new_markdown_cell("## Figures\n\n" + "\n\n".join(notebook_image(p) for p in figures)))
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nbf.writes(nb), encoding="utf-8")


def build_notebooks(
    exclusion: pd.DataFrame,
    per_market_top: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    sports: pd.DataFrame,
    sports_league: pd.DataFrame,
    volume: pd.DataFrame,
    figures: dict[str, list[Path]],
) -> None:
    top_cols = [
        "family_label",
        "exclude_last_seconds",
        "horizon_seconds",
        "sign_convention",
        "magnitude_bucket",
        "n_obs",
        "hit_rate_pct",
        "hit_rate_ci_low_pct",
        "mean_return_cents",
        "net_edge_after_1tick_cents",
    ]
    write_notebook(
        NOTEBOOKS / "block_b_resolution_sweep.ipynb",
        "Block B Resolution Sweep",
        [
            (
                "Top-Decile Rows",
                "```csv\n"
                + csv_block(
                    exclusion[exclusion["magnitude_bucket"].eq("top_decile")].sort_values(
                        ["family_label", "exclude_last_seconds", "horizon_seconds", "sign_convention"]
                    ),
                    top_cols,
                    max_rows=80,
                )
                + "```",
            )
        ],
        figures.get("resolution", []),
    )
    market_cols = [
        "family_label",
        "horizon_seconds",
        "sign_convention",
        "n_markets",
        "frac_markets_hit_gt_55",
        "frac_markets_positive_ev",
        "median_market_hit_rate_pct",
        "top_market_hit_rate_pct",
    ]
    strong_cols = [
        "family_label",
        "market_id",
        "horizon_seconds",
        "sign_convention",
        "n_obs",
        "hit_rate_pct",
        "hit_rate_ci_low_pct",
        "mean_return_cents",
        "question",
    ]
    write_notebook(
        NOTEBOOKS / "block_b_per_market_heterogeneity.ipynb",
        "Block B Per-Market Heterogeneity",
        [
            ("Family Distribution Summary", "```csv\n" + csv_block(heterogeneity, market_cols, max_rows=80) + "```"),
            (
                "Markets With Top-Decile Hit Rate Above 55%",
                "```csv\n"
                + csv_block(
                    per_market_top[per_market_top["hit_rate_pct"].gt(55)].sort_values(
                        "hit_rate_pct", ascending=False
                    ),
                    strong_cols,
                    max_rows=80,
                )
                + "```",
            ),
        ],
        figures.get("per_market", []),
    )
    sports_cols = [
        "horizon_seconds",
        "sports_game_phase",
        "sign_convention",
        "magnitude_bucket",
        "n_obs",
        "hit_rate_pct",
        "hit_rate_ci_low_pct",
        "mean_return_cents",
    ]
    league_cols = [
        "sports_league",
        "horizon_seconds",
        "sign_convention",
        "magnitude_bucket",
        "n_obs",
        "hit_rate_pct",
        "mean_return_cents",
    ]
    write_notebook(
        NOTEBOOKS / "block_b_sports_analysis.ipynb",
        "Block B Sports Analysis",
        [
            (
                "Hit-Rate By Magnitude",
                "Game start time is not available in the local market metadata, so pre-game vs in-game segmentation is left as a TODO rather than inferred from resolution time.\n\n"
                "```csv\n"
                + csv_block(sports.sort_values(["horizon_seconds", "sign_convention", "magnitude_bucket"]), sports_cols, max_rows=120)
                + "```",
            ),
            ("League Breakdown", "```csv\n" + csv_block(sports_league, league_cols, max_rows=120) + "```"),
        ],
        figures.get("sports", []),
    )
    vol_cols = [
        "family_label",
        "horizon_seconds",
        "volume_bucket",
        "sign_convention",
        "magnitude_bucket",
        "n_obs",
        "hit_rate_pct",
        "hit_rate_ci_low_pct",
        "mean_return_cents",
        "net_edge_after_1tick_cents",
    ]
    write_notebook(
        NOTEBOOKS / "block_b_volume_interaction.ipynb",
        "Block B Volume Interaction",
        [
            (
                "Volume-Regime Hit-Rate By Magnitude",
                "Rolling volume is a trailing 300-second trade-count window computed from cached market-second bars.\n\n"
                "```csv\n"
                + csv_block(volume.sort_values(["family_label", "horizon_seconds", "volume_bucket", "sign_convention", "magnitude_bucket"]), vol_cols, max_rows=160)
                + "```",
            )
        ],
        figures.get("volume", []),
    )


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{100 * x:.1f}%"


def synthesize_findings(
    exclusion: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    volume: pd.DataFrame,
    operator_cmp: pd.DataFrame,
    sports_league: pd.DataFrame,
    walk_forward: pd.DataFrame,
    *,
    exclude_last_seconds: int,
    volume_window_seconds: int,
) -> str:
    decision_sign = "inverse_maker_side"
    ex_top = monotone_top_decile_signals(
        exclusion,
        ["family", "family_label", "exclude_last_seconds", "horizon_seconds", "sign_convention"],
    )
    vol_top = monotone_top_decile_signals(
        volume,
        ["family", "family_label", "horizon_seconds", "volume_bucket", "sign_convention"],
    )
    op_top = monotone_top_decile_signals(
        operator_cmp,
        ["family", "family_label", "operator_filter_state", "horizon_seconds", "sign_convention"],
    )
    strict_ex = pd.DataFrame()
    if not ex_top.empty:
        strict_ex = ex_top[
            ex_top["sign_convention"].eq(decision_sign)
            & ex_top["exclude_last_seconds"].isin(REVISED_EXCLUSION_WINDOWS)
            & ex_top["monotone_hit_by_magnitude"]
            & ex_top["top_decile_hit_rate_pct"].gt(55.0)
            & ex_top["top_decile_hit_ci_low_pct"].gt(50.0)
            & ex_top["top_decile_net_after_1tick_cents"].gt(0)
        ].copy()
    strict_vol = pd.DataFrame()
    loose_volume = pd.DataFrame()
    if not vol_top.empty:
        strict_vol = vol_top[
            vol_top["sign_convention"].eq(decision_sign)
            & vol_top["monotone_hit_by_magnitude"]
            & vol_top["top_decile_hit_rate_pct"].gt(55.0)
            & vol_top["top_decile_hit_ci_low_pct"].gt(50.0)
            & vol_top["top_decile_net_after_1tick_cents"].gt(0)
        ].copy()
        loose_volume = vol_top[
            vol_top["sign_convention"].eq(decision_sign)
            & vol_top["top_decile_hit_rate_pct"].gt(55.0)
            & vol_top["top_decile_hit_ci_low_pct"].gt(50.0)
            & vol_top["top_decile_net_after_1tick_cents"].gt(0)
        ].copy()
    strict_operator = pd.DataFrame()
    if not op_top.empty:
        strict_operator = op_top[
            op_top["sign_convention"].eq(decision_sign)
            & op_top["operator_filter_state"].eq("operator_removed")
            & op_top["monotone_hit_by_magnitude"]
            & op_top["top_decile_hit_rate_pct"].gt(55.0)
            & op_top["top_decile_hit_ci_low_pct"].gt(50.0)
            & op_top["top_decile_net_after_1tick_cents"].gt(0)
        ].copy()

    mixed_market = pd.DataFrame()
    if not heterogeneity.empty:
        mixed_market = heterogeneity[
            heterogeneity["sign_convention"].eq(decision_sign)
            & heterogeneity["frac_markets_hit_gt_55"].gt(0.20)
        ].copy()

    if not strict_ex.empty or not strict_vol.empty:
        outcome = "OUTCOME 1 - Salvageable Fill Signal"
        verdict = (
            "At least one unfiltered historical-aggressor condition passes the "
            "monotone magnitude, top-decile hit-rate, and rough one-tick net-EV "
            "screen. Treat it as a Block A live-capture target, not as a "
            "tradable rule."
        )
    elif not strict_operator.empty or not mixed_market.empty or not loose_volume.empty:
        outcome = "OUTCOME 3 - Mixed Results Requiring Live Validation"
        verdict = (
            "No unfiltered historical-aggressor exclusion or volume condition "
            "passes the full Outcome 1 screen. There are promising subsets, "
            "especially after operator removal and in a few high-volume/market "
            "cells, but the evidence is not coherent enough across families, "
            "time, costs, and conditioning buckets to validate fill-only TFI by "
            "itself."
        )
    else:
        outcome = "OUTCOME 2 - Fill Data Exhausted"
        verdict = (
            "The structured variants remain flat, tail-driven, or too noisy after "
            "minimum-bucket and confidence-interval filters. Fill-only TFI is not "
            "directly validated by this pass."
        )

    best_ex = (
        ex_top[ex_top["sign_convention"].eq(decision_sign)]
        .sort_values("top_decile_hit_rate_pct", ascending=False)
        .head(12)
        if not ex_top.empty
        else pd.DataFrame()
    )
    best_vol = (
        vol_top[vol_top["sign_convention"].eq(decision_sign)]
        .sort_values("top_decile_hit_rate_pct", ascending=False)
        .head(12)
        if not vol_top.empty
        else pd.DataFrame()
    )
    het_view = (
        heterogeneity.sort_values("frac_markets_hit_gt_55", ascending=False).head(12)
        if not heterogeneity.empty
        else pd.DataFrame()
    )
    op_view = (
        operator_cmp[
            operator_cmp["magnitude_bucket"].eq("top_decile")
            & operator_cmp["horizon_seconds"].eq(300)
        ]
        .sort_values(["family_label", "sign_convention", "operator_filter_state"])
        .head(40)
        if not operator_cmp.empty
        else pd.DataFrame()
    )
    sports_view = (
        sports_league[
            sports_league["magnitude_bucket"].eq("top_decile")
            & sports_league["sign_convention"].eq("inverse_maker_side")
        ]
        .sort_values("hit_rate_pct", ascending=False)
        .head(20)
        if not sports_league.empty
        else pd.DataFrame()
    )
    wf_view = (
        walk_forward[
            walk_forward["magnitude_bucket"].eq("top_decile")
            & walk_forward["sign_convention"].eq("inverse_maker_side")
        ]
        .sort_values(["family_label", "horizon_seconds", "wf_split"])
        .head(60)
        if not walk_forward.empty
        else pd.DataFrame()
    )

    lines = [
        "# Block B Findings - Historical Fill-Only TFI Deep Dive",
        "",
        "Generated by `scripts/dali_tfi_deep_dive.py`.",
        "",
        f"## Decision Matrix Outcome: {outcome}",
        "",
        verdict,
        "",
        "Important framing:",
        "",
        "- Historical fills contain no L2 spread/depth, so cost columns are conservative tick proxies, not tradability evidence.",
        "- Historical `maker_side` is passive maker token side; `inverse_maker_side` is the confirmed historical token-side aggressor convention. Both are reported, but the decision outcome is based on `inverse_maker_side`.",
        "- Buckets with fewer than 100 observations are omitted from reported tables.",
        "- Hit-rate and mean-EV intervals are bootstrap 95% CIs.",
        "- Passive-maker-side positive rows, especially crypto, are diagnostic artifacts to inspect in the CSVs/notebooks, not enough by themselves to classify fill-only TFI as salvageable.",
        "",
        "## Resolution Contamination Sweep",
        "",
        f"Exclusion windows used: {EXCLUSION_WINDOWS}. The revised prompt windows ({REVISED_EXCLUSION_WINDOWS}) are included, and the handoff's shorter windows are kept for continuity.",
        "",
        "Top-decile historical-aggressor (`inverse_maker_side`) candidates with the highest hit rates:",
        "",
        "```csv",
        csv_block(
            best_ex,
            [
                "family_label",
                "exclude_last_seconds",
                "horizon_seconds",
                "sign_convention",
                "monotone_hit_by_magnitude",
                "top_decile_n_obs",
                "top_decile_hit_rate_pct",
                "top_decile_hit_ci_low_pct",
                "top_decile_mean_return_cents",
                "top_decile_net_after_1tick_cents",
            ],
            max_rows=12,
        ).rstrip(),
        "```",
        "",
        "## Per-Market Heterogeneity",
        "",
        f"Default conditioning exclusion: {exclude_last_seconds} seconds before market end.",
        "",
        "```csv",
        csv_block(
            het_view,
            [
                "family_label",
                "horizon_seconds",
                "sign_convention",
                "n_markets",
                "frac_markets_hit_gt_55",
                "frac_markets_hit_ci_low_gt_50",
                "frac_markets_positive_ev",
                "frac_markets_positive_after_1tick",
                "median_market_hit_rate_pct",
                "top_market_hit_rate_pct",
            ],
            max_rows=12,
        ).rstrip(),
        "```",
        "",
        "## Walk-Forward Split",
        "",
        "Each family is split chronologically into 60% train, 20% validation, and 20% test. Magnitude thresholds for this table are fitted on the train slice and then applied forward.",
        "",
        "```csv",
        csv_block(
            wf_view,
            [
                "family_label",
                "wf_split",
                "horizon_seconds",
                "n_obs",
                "hit_rate_pct",
                "hit_rate_ci_low_pct",
                "mean_return_cents",
                "net_edge_after_1tick_cents",
            ],
            max_rows=60,
        ).rstrip(),
        "```",
        "",
        "## Volume Interaction",
        "",
        f"Rolling volume is trailing trade count over {volume_window_seconds} seconds.",
        "",
        "```csv",
        csv_block(
            best_vol,
            [
                "family_label",
                "horizon_seconds",
                "volume_bucket",
                "sign_convention",
                "monotone_hit_by_magnitude",
                "top_decile_n_obs",
                "top_decile_hit_rate_pct",
                "top_decile_hit_ci_low_pct",
                "top_decile_mean_return_cents",
                "top_decile_net_after_1tick_cents",
            ],
            max_rows=12,
        ).rstrip(),
        "```",
        "",
        "## Sports Explicit Analysis",
        "",
        "Local market metadata has no game-start field, so pre-game vs in-game segmentation is documented as TODO rather than inferred from `end_date`. League labels are regex-derived from question/slug.",
        "",
        "```csv",
        csv_block(
            sports_view,
            [
                "sports_league",
                "horizon_seconds",
                "n_obs",
                "hit_rate_pct",
                "hit_rate_ci_low_pct",
                "mean_return_cents",
                "net_edge_after_1tick_cents",
            ],
            max_rows=20,
        ).rstrip(),
        "```",
        "",
        "## Operator Filter",
        "",
        "The existing `data_infra.operator_denylist.OPERATOR_ADDRESSES` helper was applied by rebuilding market-second bars from cached family fills with denylisted maker/taker addresses removed.",
        "",
        "Operator-filtered rows can identify live-capture targets, but this is not treated as direct tradability evidence because the live market WebSocket may not expose the same address-level filter.",
        "",
        "```csv",
        csv_block(
            op_view,
            [
                "family_label",
                "operator_filter_state",
                "horizon_seconds",
                "sign_convention",
                "n_obs",
                "hit_rate_pct",
                "mean_return_cents",
                "net_edge_after_1tick_cents",
            ],
            max_rows=40,
        ).rstrip(),
        "```",
        "",
        "## Bottom Line",
        "",
        "The output should be used to prioritize Block A live OFI capture targets only where patterns are coherent across related buckets. No result here establishes live tradability because historical fills lack L2 spread/depth and queue context.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-signal-usd", type=float, default=25.0)
    parser.add_argument("--max-future-gap-seconds", type=int, default=300)
    parser.add_argument("--default-exclude-last-seconds", type=int, default=600)
    parser.add_argument("--horizons", default="30,120,300")
    parser.add_argument("--min-bucket-obs", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--volume-window-seconds", type=int, default=300)
    parser.add_argument("--skip-operator-filter", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    inputs = [inp for inp in DEFAULT_INPUTS if inp.eval_path.exists()]
    missing = [inp.eval_path for inp in DEFAULT_INPUTS if not inp.eval_path.exists()]
    if missing:
        print(f"missing eval inputs: {[p.name for p in missing]}")
    if not inputs:
        raise SystemExit("no Dali eval inputs found")

    evals = {inp.family: read_eval(inp) for inp in inputs}
    market_meta = {inp.family: read_candidates(inp) for inp in inputs}

    exclusion = build_exclusion_sweep(
        evals,
        min_signal_usd=args.min_signal_usd,
        max_future_gap_seconds=args.max_future_gap_seconds,
        horizons=horizons,
        min_obs=args.min_bucket_obs,
        bootstrap_samples=args.bootstrap_samples,
    )
    per_market, per_market_top, heterogeneity = build_per_market(
        evals,
        market_meta,
        exclude_last_seconds=args.default_exclude_last_seconds,
        min_signal_usd=args.min_signal_usd,
        max_future_gap_seconds=args.max_future_gap_seconds,
        horizons=horizons,
        min_obs=args.min_bucket_obs,
        bootstrap_samples=args.bootstrap_samples,
    )
    walk_forward = build_walk_forward(
        evals,
        exclude_last_seconds=args.default_exclude_last_seconds,
        min_signal_usd=args.min_signal_usd,
        max_future_gap_seconds=args.max_future_gap_seconds,
        horizons=horizons,
        min_obs=args.min_bucket_obs,
        bootstrap_samples=args.bootstrap_samples,
    )
    volume = build_volume_interaction(
        inputs,
        evals,
        exclude_last_seconds=args.default_exclude_last_seconds,
        min_signal_usd=args.min_signal_usd,
        max_future_gap_seconds=args.max_future_gap_seconds,
        horizons=horizons,
        volume_window_seconds=args.volume_window_seconds,
        min_obs=args.min_bucket_obs,
        bootstrap_samples=args.bootstrap_samples,
    )
    sports, sports_league = build_sports_explicit(
        evals["sports_game_lines"],
        exclude_last_seconds=args.default_exclude_last_seconds,
        min_signal_usd=args.min_signal_usd,
        max_future_gap_seconds=args.max_future_gap_seconds,
        horizons=horizons,
        min_obs=args.min_bucket_obs,
        bootstrap_samples=args.bootstrap_samples,
    )
    operator_cmp = pd.DataFrame()
    if not args.skip_operator_filter:
        operator_cmp = build_operator_comparison(
            inputs,
            evals,
            exclude_last_seconds=args.default_exclude_last_seconds,
            min_signal_usd=args.min_signal_usd,
            max_future_gap_seconds=args.max_future_gap_seconds,
            horizons=horizons,
            min_obs=args.min_bucket_obs,
            bootstrap_samples=args.bootstrap_samples,
        )

    for df, path in [
        (per_market, OUT_PER_MARKET),
        (per_market_top, OUT_PER_MARKET_TOP),
        (heterogeneity, OUT_PER_MARKET_HET),
        (walk_forward, OUT_WALK_FORWARD),
        (volume, OUT_VOLUME),
        (sports, OUT_SPORTS),
        (sports_league, OUT_SPORTS_LEAGUE),
        (operator_cmp, OUT_OPERATOR),
    ]:
        df.to_csv(path, index=False)

    consolidated = pd.concat(
        [
            exclusion,
            per_market,
            per_market_top,
            heterogeneity,
            walk_forward,
            volume,
            sports,
            sports_league,
            operator_cmp,
        ],
        ignore_index=True,
        sort=False,
    )
    consolidated = ordered_bucket_frame(consolidated)
    consolidated.to_csv(OUT_SUMMARY, index=False)

    figures = {
        "resolution": plot_resolution_sweep(exclusion),
        "per_market": plot_per_market(per_market_top),
        "sports": plot_sports(sports_league),
        "volume": plot_volume(volume),
    }
    build_notebooks(
        exclusion=exclusion,
        per_market_top=per_market_top,
        heterogeneity=heterogeneity,
        sports=sports,
        sports_league=sports_league,
        volume=volume,
        figures=figures,
    )

    note = synthesize_findings(
        exclusion=exclusion,
        heterogeneity=heterogeneity,
        volume=volume,
        operator_cmp=operator_cmp,
        sports_league=sports_league,
        walk_forward=walk_forward,
        exclude_last_seconds=args.default_exclude_last_seconds,
        volume_window_seconds=args.volume_window_seconds,
    )
    OUT_NOTE.write_text(note, encoding="utf-8")

    print(f"families: {[inp.family for inp in inputs]}")
    print(f"exclusion rows: {len(exclusion):,}")
    print(f"per-market rows: {len(per_market):,}")
    print(f"per-market top-decile rows: {len(per_market_top):,}")
    print(f"walk-forward rows: {len(walk_forward):,}")
    print(f"volume rows: {len(volume):,}")
    print(f"sports rows: {len(sports):,}; sports league rows: {len(sports_league):,}")
    print(f"operator rows: {len(operator_cmp):,}")
    print(f"summary: {OUT_SUMMARY.relative_to(ROOT)}")
    print(f"note: {OUT_NOTE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
