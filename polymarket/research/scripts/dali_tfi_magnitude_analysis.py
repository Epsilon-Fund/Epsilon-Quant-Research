"""Build TFI hit-rate-by-magnitude notebook and figures."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTEBOOKS = ROOT / "notebooks"
FIGS = NOTEBOOKS / "figs"
OUT_NOTEBOOK = NOTEBOOKS / "tfi_magnitude_analysis.ipynb"
OUT_SUMMARY = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_magnitude_summary.csv"

BUCKETS = [
    "bottom_decile",
    "bottom_quartile_ex_decile",
    "middle_two_quartiles",
    "top_quartile_ex_decile",
    "top_decile",
]


@dataclass(frozen=True)
class FamilyInput:
    family: str
    eval_path: Path
    exclude_last_seconds: int = 600


DEFAULT_INPUTS = [
    FamilyInput("daily_crypto_up_down", ANALYSIS / "dali_tfi_crypto_250_exlast600_eval.parquet"),
    FamilyInput("daily_equity_index", ANALYSIS / "dali_tfi_equity_index_100_eval.parquet"),
    FamilyInput("ai_product", ANALYSIS / "dali_tfi_ai_product_100_eval.parquet"),
    FamilyInput("sports_game_lines", ANALYSIS / "dali_tfi_sports_100_eval.parquet"),
]


def bucket_magnitude(values: pd.Series) -> pd.Series:
    q10, q25, q75, q90 = values.quantile([0.10, 0.25, 0.75, 0.90])
    return pd.cut(
        values,
        bins=[-np.inf, q10, q25, q75, q90, np.inf],
        labels=BUCKETS,
        include_lowest=True,
        duplicates="drop",
    ).astype(str)


def load_eval(inp: FamilyInput, min_signal_usd: float, max_future_gap_seconds: int) -> pd.DataFrame:
    df = pd.read_parquet(inp.eval_path)
    for col in ("second_ts", "future_ts", "end_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    mask = (
        df["future_vwap_price"].notna()
        & df["future_gap_seconds"].le(max_future_gap_seconds)
        & df["signed_maker_usd"].ne(0)
        & df["signed_maker_usd"].abs().ge(min_signal_usd)
    )
    if "end_ts" in df.columns:
        mask &= df["end_ts"].isna() | df["future_ts"].le(df["end_ts"])
        seconds_to_end = (df["end_ts"] - df["second_ts"]).dt.total_seconds()
        mask &= df["end_ts"].isna() | seconds_to_end.ge(inp.exclude_last_seconds)
    out = df.loc[mask].copy()
    out["family"] = inp.family
    out["abs_signal_usd"] = out["signed_maker_usd"].abs()
    return out


def summarize_family(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, hdf in df.groupby("horizon_seconds"):
        if len(hdf) < 20:
            continue
        hdf = hdf.copy()
        hdf["magnitude_bucket"] = bucket_magnitude(hdf["abs_signal_usd"])
        for convention, mult in (("maker_side", 1.0), ("inverse_maker_side", -1.0)):
            signal_direction = np.sign(hdf["signed_maker_usd"]) * mult
            signed_return = signal_direction * hdf["future_price_change"] * 100.0
            tmp = hdf.assign(
                sign_convention=convention,
                signed_return_cents=signed_return,
                hit=signed_return.gt(0),
            )
            for bucket in BUCKETS:
                bdf = tmp[tmp["magnitude_bucket"].eq(bucket)]
                if bdf.empty:
                    continue
                std = bdf["signed_return_cents"].std(ddof=1)
                rows.append(
                    {
                        "family": bdf["family"].iloc[0],
                        "horizon_seconds": horizon,
                        "sign_convention": convention,
                        "magnitude_bucket": bucket,
                        "n_obs": len(bdf),
                        "hit_rate_pct": 100.0 * bdf["hit"].mean(),
                        "mean_return_cents": bdf["signed_return_cents"].mean(),
                        "median_return_cents": bdf["signed_return_cents"].median(),
                        "return_sharpe_like": (
                            bdf["signed_return_cents"].mean() / std
                            if std and np.isfinite(std)
                            else np.nan
                        ),
                        "avg_abs_signal_usd": bdf["abs_signal_usd"].mean(),
                    }
                )
    return pd.DataFrame(rows)


def interpretation(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "No usable rows."
    ordered = rows.set_index("magnitude_bucket").reindex(BUCKETS)
    first = ordered.dropna(subset=["hit_rate_pct", "mean_return_cents"]).head(1)
    last = ordered.dropna(subset=["hit_rate_pct", "mean_return_cents"]).tail(1)
    if first.empty or last.empty:
        return "Too few populated magnitude buckets."
    hit_delta = float(last["hit_rate_pct"].iloc[0] - first["hit_rate_pct"].iloc[0])
    ev_delta = float(last["mean_return_cents"].iloc[0] - first["mean_return_cents"].iloc[0])
    if hit_delta > 5:
        return f"Hit rate rises with magnitude (+{hit_delta:.1f}pp): informative-flow pattern."
    if abs(hit_delta) <= 5 and ev_delta > 0.5:
        return f"Hit rate is roughly flat, EV rises (+{ev_delta:.2f}c): tail/asymmetry pattern."
    if abs(hit_delta) <= 5 and abs(ev_delta) <= 0.5:
        return "Hit rate and EV are roughly flat: weak or artifact-prone pattern."
    if hit_delta < -5:
        return f"Hit rate is inverted ({hit_delta:.1f}pp): wrong sign/noise warning."
    return f"Mixed pattern: hit delta {hit_delta:.1f}pp, EV delta {ev_delta:.2f}c."


def plot_family_horizon(summary: pd.DataFrame, family: str, horizon: int, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    subset = summary[
        summary["family"].eq(family) & summary["horizon_seconds"].eq(horizon)
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, convention in zip(axes, ("maker_side", "inverse_maker_side"), strict=False):
        rows = (
            subset[subset["sign_convention"].eq(convention)]
            .set_index("magnitude_bucket")
            .reindex(BUCKETS)
            .reset_index()
        )
        x = np.arange(len(BUCKETS))
        ax2 = ax.twinx()
        ax.plot(x, rows["hit_rate_pct"], marker="o", color="#1f77b4", label="hit rate")
        ax2.bar(x, rows["mean_return_cents"], alpha=0.30, color="#ff7f0e", label="EV cents")
        ax.axhline(50, color="#888888", linewidth=0.8, linestyle="--")
        ax2.axhline(0, color="#aa5500", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Hit rate (%)")
        ax2.set_ylabel("Mean signed return (cents)")
        ax.set_title(f"{family} | {horizon}s | {convention}")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(np.arange(len(BUCKETS)))
    axes[-1].set_xticklabels(BUCKETS, rotation=25, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_notebook(summary: pd.DataFrame, figures: list[Path]) -> None:
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# TFI Magnitude Analysis\n\n"
            "Generated by `scripts/dali_tfi_magnitude_analysis.py`.\n\n"
            "Sign convention is not established from live CLOB yet, so this "
            "notebook reports both `maker_side` and `inverse_maker_side`."
        )
    ]
    if summary.empty:
        cells.append(nbf.v4.new_markdown_cell("No summary rows were generated."))
    else:
        for (family, horizon, convention), rows in summary.groupby(
            ["family", "horizon_seconds", "sign_convention"],
            sort=True,
        ):
            rows = rows.set_index("magnitude_bucket").reindex(BUCKETS).reset_index()
            cells.append(
                nbf.v4.new_markdown_cell(
                    f"## {family} | {horizon}s | {convention}\n\n"
                    f"{interpretation(rows)}\n\n"
                    + rows[
                        [
                            "magnitude_bucket",
                            "n_obs",
                            "hit_rate_pct",
                            "mean_return_cents",
                            "median_return_cents",
                            "return_sharpe_like",
                        ]
                    ].round(4).to_csv(index=False)
                )
            )
        cells.append(nbf.v4.new_markdown_cell("## Figures"))
        for fig in figures:
            cells.append(nbf.v4.new_markdown_cell(f"![{fig.stem}]({fig.relative_to(NOTEBOOKS)})"))
    nb["cells"] = cells
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    OUT_NOTEBOOK.write_text(nbf.writes(nb), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-signal-usd", type=float, default=25.0)
    parser.add_argument("--max-future-gap-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = []
    used = []
    for inp in DEFAULT_INPUTS:
        if inp.eval_path.exists():
            df = load_eval(inp, args.min_signal_usd, args.max_future_gap_seconds)
            summaries.append(summarize_family(df))
            used.append(inp)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)

    figures = []
    if not summary.empty:
        for family in summary["family"].drop_duplicates():
            for horizon in sorted(summary.loc[summary["family"].eq(family), "horizon_seconds"].unique()):
                out = FIGS / f"tfi_magnitude_{family}_{int(horizon)}s.png"
                plot_family_horizon(summary, family, int(horizon), out)
                figures.append(out)
    build_notebook(summary, figures)
    print(f"inputs: {[inp.eval_path.name for inp in used]}")
    print(f"summary rows: {len(summary):,} -> {OUT_SUMMARY.relative_to(ROOT)}")
    print(f"figures: {len(figures):,} -> {FIGS.relative_to(ROOT)}")
    print(f"notebook: {OUT_NOTEBOOK.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
