"""Block A1.5b decoupled micro-price target test.

Signals are OFI and TFI. Micro-price is only used as a future target.
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
FEATURES = ANALYSIS / "block_a1_features.parquet"
A1_DECILES = ANALYSIS / "csv_outputs" / "dali" / "block_a1_decile_aggregate.csv"
RESULTS_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a15b_decoupled_results.csv"
BASELINE_CHECK_OUT = ANALYSIS / "csv_outputs" / "dali" / "block_a15b_baseline_check.csv"
NOTE = ROOT / "notes" / "block_a15b_decoupled_findings.md"

HORIZONS = (1, 5, 30, 300)
SIGNALS = ("ofi", "tfi")
TARGETS = ("mid_target", "micro_target")
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260528
MIN_BOOTSTRAP_N = 30
BASELINE_TOL = 1e-12


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
    ]
    con = duckdb.connect()
    select_cols = ", ".join(cols)
    df = con.execute(f"SELECT {select_cols} FROM read_parquet('{path}')").df()
    if df.empty:
        raise SystemExit(f"no rows found in {display_path(path)}")

    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
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
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    df["directional_mid"] = np.where(df["direction_factor"].gt(0), df["mid"], 1.0 - df["mid"])
    df["touch_depth"] = df[["best_bid_size", "best_ask_size"]].sum(axis=1, min_count=2)
    df["spread_bps"] = np.where(
        df["mid"].gt(0) & df["spread"].notna(),
        df["spread"] / df["mid"] * 10_000.0,
        np.nan,
    )
    side_sign = (
        df["trade_side"]
        .fillna(df["last_trade_side"])
        .fillna("")
        .astype(str)
        .str.upper()
        .map({"BUY": 1.0, "SELL": -1.0})
        .fillna(0.0)
    )
    df["is_trade"] = df["event_type"].eq("last_trade_price")
    df["signed_trade_size_live"] = df["is_trade"].astype(float) * side_sign * df["trade_size"].fillna(0.0)
    df["ofi_combined_event"] = df["ofi_combined_event"].fillna(0.0)
    df["micro_price"] = df["mid"] + 0.5 * df["spread"] * df["tob_imbalance"]

    market_depth = (
        df.groupby(["run_id", "market_key"], as_index=False)["touch_depth"]
        .mean()
        .rename(columns={"touch_depth": "market_mean_depth"})
    )
    df = df.merge(market_depth, on=["run_id", "market_key"], how="left")
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
        g = g.set_index("received_at", drop=False)
        for horizon in HORIZONS:
            window = f"{horizon}s"
            ofi = g["ofi_combined_event"].fillna(0.0).rolling(window).sum()
            tfi = g["signed_trade_size_live"].fillna(0.0).rolling(window).sum()
            g[f"signal_ofi_{horizon}s"] = g["direction_factor"] * ofi / g["market_mean_depth"]
            g[f"signal_tfi_{horizon}s"] = g["direction_factor"] * tfi / g["market_mean_depth"]

            future_mid = future_value(g, "mid", horizon)
            future_directional_mid = np.where(g["direction_factor"].gt(0), future_mid, 1.0 - future_mid)
            g[f"return_mid_target_{horizon}s"] = np.where(
                g["directional_mid"].gt(0) & np.isfinite(future_directional_mid),
                (future_directional_mid - g["directional_mid"]) / g["directional_mid"] * 10_000.0,
                np.nan,
            )

            future_micro = future_value(g, "micro_price", horizon)
            future_directional_micro = np.where(g["direction_factor"].gt(0), future_micro, 1.0 - future_micro)
            g[f"return_micro_target_{horizon}s"] = np.where(
                g["directional_mid"].gt(0) & np.isfinite(future_directional_micro),
                (future_directional_micro - g["directional_mid"]) / g["directional_mid"] * 10_000.0,
                np.nan,
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
    if len(clean) < MIN_BOOTSTRAP_N:
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
        & df["market_mean_depth"].gt(0)
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


def decile_results(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for signal in SIGNALS:
        for target in TARGETS:
            for horizon in HORIZONS:
                x_col = f"signal_{signal}_{horizon}s"
                y_col = f"return_{target}_{horizon}s"
                sub = valid_signal_rows(df, x_col, y_col)
                if len(sub) < 50:
                    continue
                sub = assign_deciles(sub, x_col)
                sub = sub[sub["decile"].notna()]
                max_decile = int(sub["decile"].max())
                for decile, rows in sub.groupby("decile", sort=True):
                    out_rows.append({
                        "signal": signal,
                        "target": target,
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
                            + stable_seed_offset(signal, target, modulo=5_000),
                        ),
                    })
    return pd.DataFrame(out_rows)


def baseline_check(results: pd.DataFrame, a1_deciles_path: Path) -> pd.DataFrame:
    ours = results[results["signal"].eq("ofi") & results["target"].eq("mid_target")][
        ["horizon_sec", "decile", "n", "mean_abs_signal", "mean_next_return_bps", "directional_return_bps", "hit_rate"]
    ].rename(
        columns={
            "n": "a15b_n",
            "mean_abs_signal": "a15b_mean_abs_signal",
            "mean_next_return_bps": "a15b_mean_next_mid_return_bps",
            "directional_return_bps": "a15b_directional_return_bps",
            "hit_rate": "a15b_hit_rate",
        }
    )
    a1 = pd.read_csv(a1_deciles_path)[
        ["horizon_sec", "decile", "n", "mean_abs_ofi_scaled", "mean_next_mid_return_bps", "directional_return_bps", "hit_rate"]
    ].rename(
        columns={
            "n": "a1_n",
            "mean_abs_ofi_scaled": "a1_mean_abs_signal",
            "mean_next_mid_return_bps": "a1_mean_next_mid_return_bps",
            "directional_return_bps": "a1_directional_return_bps",
            "hit_rate": "a1_hit_rate",
        }
    )
    out = ours.merge(a1, on=["horizon_sec", "decile"], how="outer").sort_values(["horizon_sec", "decile"])
    out["n_delta"] = out["a15b_n"] - out["a1_n"]
    out["hit_delta_pp"] = (out["a15b_hit_rate"] - out["a1_hit_rate"]) * 100.0
    out["directional_return_delta_bps"] = (
        out["a15b_directional_return_bps"] - out["a1_directional_return_bps"]
    )
    out["mean_abs_signal_delta"] = out["a15b_mean_abs_signal"] - out["a1_mean_abs_signal"]
    out["passes_exact"] = (
        out["n_delta"].eq(0)
        & out["hit_delta_pp"].abs().le(BASELINE_TOL)
        & out["directional_return_delta_bps"].abs().le(BASELINE_TOL)
        & out["mean_abs_signal_delta"].abs().le(BASELINE_TOL)
    )
    return out


def ensure_baseline_passes(check: pd.DataFrame) -> None:
    if check.empty or len(check) != len(HORIZONS) * 10 or not bool(check["passes_exact"].all()):
        raise SystemExit(
            "A1.5b calibration gate failed; not reporting downstream results. "
            f"See {display_path(BASELINE_CHECK_OUT)}"
        )


def top_decile_comparison(results: pd.DataFrame) -> pd.DataFrame:
    top = results[results["is_top_decile"]].copy()
    wide = top.pivot_table(
        index=["signal", "horizon_sec"],
        columns="target",
        values=["n", "hit_rate", "hit_rate_ci_lo", "hit_rate_ci_hi", "directional_return_bps", "mean_next_return_bps"],
        aggfunc="first",
    )
    wide.columns = [f"{target}_{metric}" for metric, target in wide.columns]
    wide = wide.reset_index()
    wide["hit_delta_micro_minus_mid_pp"] = (
        wide["micro_target_hit_rate"] - wide["mid_target_hit_rate"]
    ) * 100.0
    wide["directional_return_delta_micro_minus_mid_bps"] = (
        wide["micro_target_directional_return_bps"] - wide["mid_target_directional_return_bps"]
    )
    return wide.sort_values(["signal", "horizon_sec"])


def summary_table(summary: pd.DataFrame, signal: str) -> str:
    sub = summary[summary["signal"].eq(signal)].sort_values("horizon_sec")
    rows = [
        [
            int(row["horizon_sec"]),
            pct(float(row["mid_target_hit_rate"])),
            pct(float(row["micro_target_hit_rate"])),
            pp(float(row["hit_delta_micro_minus_mid_pp"])),
            bps(float(row["mid_target_directional_return_bps"])),
            bps(float(row["micro_target_directional_return_bps"])),
            bps(float(row["directional_return_delta_micro_minus_mid_bps"])),
            f"{int(row['mid_target_n']):,}",
            f"{int(row['micro_target_n']):,}",
        ]
        for _, row in sub.iterrows()
    ]
    return markdown_table(
        ["h", "mid hit", "micro hit", "delta", "mid dir", "micro dir", "dir delta", "mid n", "micro n"],
        rows,
    )


def calibration_table(check: pd.DataFrame) -> str:
    top = check[check["decile"].eq(10)].sort_values("horizon_sec")
    rows = [
        [
            int(row["horizon_sec"]),
            f"{int(row['a15b_n']):,}",
            f"{int(row['a1_n']):,}",
            pp(float(row["hit_delta_pp"])),
            bps(float(row["directional_return_delta_bps"])),
            "yes" if bool(row["passes_exact"]) else "NO",
        ]
        for _, row in top.iterrows()
    ]
    return markdown_table(["h", "A15b n", "A1 n", "hit delta", "dir delta", "pass"], rows)


def verdict(summary: pd.DataFrame) -> tuple[str, str]:
    ofi = summary[summary["signal"].eq("ofi")].copy()
    tfi = summary[summary["signal"].eq("tfi")].copy()
    ofi_1 = ofi[ofi["horizon_sec"].eq(1)]
    ofi_5 = ofi[ofi["horizon_sec"].eq(5)]
    if ofi_5.empty:
        headline = "OFI 5s comparison unavailable."
    else:
        row = ofi_5.iloc[0]
        one_sec = ""
        if not ofi_1.empty:
            row_1 = ofi_1.iloc[0]
            one_sec = (
                f" The one clear micro-target hit-rate improvement is at 1s "
                f"({pct(float(row_1['micro_target_hit_rate']))} vs "
                f"{pct(float(row_1['mid_target_hit_rate']))}, "
                f"{pp(float(row_1['hit_delta_micro_minus_mid_pp']))})."
            )
        headline = (
            f"OFI's 5s top-decile hit rate is essentially unchanged when the target is future "
            f"micro-price instead of future mid: "
            f"{pct(float(row['micro_target_hit_rate']))} vs {pct(float(row['mid_target_hit_rate']))}, "
            f"{pp(float(row['hit_delta_micro_minus_mid_pp']))}.{one_sec}"
        )

    ofi_positive = bool((ofi["hit_delta_micro_minus_mid_pp"] > 0).all()) if not ofi.empty else False
    tfi_positive = bool((tfi["hit_delta_micro_minus_mid_pp"] > 0).all()) if not tfi.empty else False
    if ofi_positive:
        interpretation = (
            "The decoupled test argues against the OFI result being only mid-price mean reversion: "
            "OFI also points toward the future imbalance-adjusted fair value."
        )
    else:
        interpretation = (
            "The decoupled test does not fully clear the mean-reversion concern for OFI across horizons: "
            "hit-rate improves at 1s, is flat at 5s, and weakens at 30s/300s, even though directional "
            "return bps are larger against the micro-price target."
        )
    if tfi_positive:
        interpretation += " TFI also improves against micro-price, but remains secondary to OFI unless its absolute hit rates are stronger in later A2 data."
    else:
        interpretation += " TFI does not show the same clean micro-target improvement pattern."
    recommendation = (
        "Recommended next action for Justin: carry OFI-vs-micro-price-target as an A2 audit column "
        "alongside L1 TOB, and treat TFI as a conditional sidecar rather than a primary signal."
    )
    return headline + " " + interpretation, recommendation


def write_note(results: pd.DataFrame, check: pd.DataFrame) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    summary = top_decile_comparison(results)
    head, recommendation = verdict(summary)
    note = f"""---
tags: [dali, block-a15b, micro-price, results]
---

# Block A1.5b Decoupled Micro-Price Target Findings

## Headline

{head} This removes the A1.5 contamination path because TOB imbalance is target-only here; entries are driven by OFI or TFI.

## Calibration Gate

OFI with `mid_target` recomputes A1's depth-normalized decile aggregate from `block_a1_features.parquet`. The gate checks every decile row; the table below shows the top-decile rows.

{calibration_table(check)}

## OFI Signal

{summary_table(summary, "ofi")}

## TFI Signal

{summary_table(summary, "tfi")}

## Method

- `signal_ofi_h = direction_factor * rolling_sum(ofi_combined_event, h) / mean_depth_at_touch`.
- `signal_tfi_h = direction_factor * rolling_sum(signed_live_trade_size, h) / mean_depth_at_touch`.
- `mid_target` uses A1's future directional mid return.
- `micro_target` uses `future_micro_price = future_mid + 0.5 * future_spread * future_tob_imbalance`, direction-adjusted, with current directional mid as the denominator.
- Deciles are global equal-count buckets within each `(signal, target, horizon)` and are based on absolute signal magnitude.

## Outputs

- `{display_path(RESULTS_OUT)}`
- `{display_path(BASELINE_CHECK_OUT)}`

{recommendation}
"""
    NOTE.write_text(note, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--a1-deciles", type=Path, default=A1_DECILES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"loading {display_path(args.features)}", flush=True)
    df = load_features(args.features)
    print(f"loaded {len(df):,} rows", flush=True)
    print("adding signal and target horizons", flush=True)
    df = add_horizon_features(df)

    print("building decoupled decile results", flush=True)
    results = decile_results(df)
    check = baseline_check(results, args.a1_deciles)
    BASELINE_CHECK_OUT.parent.mkdir(parents=True, exist_ok=True)
    check.to_csv(BASELINE_CHECK_OUT, index=False)
    ensure_baseline_passes(check)
    print("baseline gate passed", flush=True)

    results.to_csv(RESULTS_OUT, index=False)
    write_note(results, check)
    print(f"results: {display_path(RESULTS_OUT)} ({len(results):,} rows)")
    print(f"baseline check: {display_path(BASELINE_CHECK_OUT)} ({len(check):,} rows)")
    print(f"note: {display_path(NOTE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
