"""Task-4 MM validation run — drive the engine over politics_negrisk + esports VPS Parquet,
under the optimistic↔pessimistic queue bracket, and emit the standardized decision report.

This is the runner for the :mod:`mm_eval` layer. It does NOT touch the engine: it discovers
quotable markets, materializes per-token replay dirs, runs the ``SymmetricQuoter`` (the A/B
baseline arm) under each queue model, attaches markout / adverse-selection / scorecard /
breakeven, runs the CPCV-style temporal-stability check, records the WIRED-BUT-DORMANT
overfitting status, and writes CSVs + plots + a markdown/JSON summary.

Usage (from polymarket/research/):
    PYTHONPATH=. uv run python scripts/mm_validation_run.py [--top-k 12] [--horizon 30]
        [--n-boot 2000] [--n-blocks 8] [--cache <dir>] [--quick]

Honesty rails (brain/CODEX.md): bracketed under both queue models; CIs not point estimates;
no profitability claim; captured fee=0 (rebate=0) is primary, the category rebate is a labeled
sensitivity; the ~12-day sample (R2 clone, 2026-06-19→06-30) supports temporal stability across
daily blocks, but a strict overfitting OOS split still only bites once a parameter is FIT
(Join 2 / Task 5), so the DSR/CPCV apparatus stays dormant on the 0-parameter quoter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mm_eval import markets as mk
from mm_eval.ab import BASELINE_ARM, run_arm
from mm_eval import report as rpt
from mm_eval.overfitting_hook import dormant_status
from mm_eval.runner import OPTIMISTIC, PESSIMISTIC

RESEARCH = Path(__file__).resolve().parents[1]
L2_ROOT = RESEARCH / "l2_data"
CSV_OUT = RESEARCH / "data/analysis/csv_outputs/market_making"
PLOT_OUT = RESEARCH / "data/analysis/plots/market_making"
SCRATCH = Path("/private/tmp/claude-501/-Users-justiniturregui-Desktop-github-epsilon-quant-research/"
               "0f7d0ead-3619-42e4-bdff-f93cd8f86dfb/scratchpad")
DEFAULT_CACHE = SCRATCH / "mm_eval_cache"
UNIVERSES = ("politics_negrisk", "esports")


def plot_markout_curves(mo_df: pd.DataFrame, path: Path) -> None:
    """Universe-mean markout-to-fill curve vs horizon, per queue model (the adverse-selection shape)."""
    fig, axes = plt.subplots(1, len(UNIVERSES), figsize=(11, 4), sharey=False)
    for ax, u in zip(np.atleast_1d(axes), UNIVERSES):
        sub = mo_df[mo_df["universe"] == u]
        for q, c in [(OPTIMISTIC, "#4878d0"), ("Prob(0.5)", "#888"), (PESSIMISTIC, "#d65f5f")]:
            g = sub[sub["queue"] == q].groupby("horizon_s")["markout_to_fill_cents"].mean()
            if len(g):
                ax.plot(g.index, g.values, "o-", color=c, label=q)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(u); ax.set_xlabel("horizon (s)"); ax.set_ylabel("mean markout-to-fill (¢/contract)")
        ax.legend(fontsize=8)
    fig.suptitle("Per-contract markout to mid vs horizon (universe mean, queue-bracketed)", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_breakeven_scatter(vd_df: pd.DataFrame, path: Path) -> None:
    """Half-spread (the breakeven A* with no rebate) vs measured pessimistic adverse selection.

    Points below the y=x line: adverse selection exceeds the spread → DEAD. Above: VIABLE-ish.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = {"politics_negrisk": "#4878d0", "esports": "#d65f5f"}
    for u in UNIVERSES:
        sub = vd_df[vd_df["universe"] == u]
        ax.scatter(sub["half_spread_cents"], sub["adverse_pess_cents"], s=40, alpha=0.75,
                   color=colors[u], label=u, edgecolor="white", linewidth=0.5)
    lim = max(vd_df["half_spread_cents"].max(), vd_df["adverse_pess_cents"].max(), 1) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="breakeven (A* = half-spread)")
    ax.set_xlabel("half-spread ¢ (breakeven adverse selection A*, no rebate)")
    ax.set_ylabel("measured adverse selection ¢ (pessimistic queue)")
    ax.set_title("Spread vs adverse selection — below the line survives, above is dead")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=30, help="primary markout horizon (s)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-blocks", type=int, default=8, help="temporal-stability time blocks")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--l2-root", type=Path, default=L2_ROOT,
                    help="root holding {date}/{universe}/ parquet (default: repo l2_data; "
                         "use the full R2 clone for the ~12-day sample)")
    ap.add_argument("--quick", action="store_true", help="top_k=4, n_boot=500 (smoke)")
    args = ap.parse_args()
    if args.quick:
        args.top_k, args.n_boot = 4, 500
    l2_root = args.l2_root

    CSV_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    summary = {"horizon_s": args.horizon, "top_k": args.top_k, "n_boot": args.n_boot,
               "n_blocks": args.n_blocks, "universes": {}}
    frames = {k: [] for k in ("scorecard", "markout", "breakeven", "verdict", "stability")}

    for universe in UNIVERSES:
        print(f"\n=== {universe} ===", flush=True)
        span = mk.capture_span(l2_root, universe, con=con)
        days = span["days"]
        print(f"  capture span: {span['hours']:.1f} h ({days:.2f} days)", flush=True)

        specs = mk.select_markets(l2_root, universe, top_k=args.top_k, con=con)
        print(f"  selected {len(specs)} quotable markets", flush=True)
        mk.build_compact(l2_root, universe, specs, args.cache, con=con)
        specs_and_dirs = [(s, mk.materialize_token(s, args.cache, con=con)) for s in specs]

        arm = run_arm(BASELINE_ARM, specs_and_dirs, days=days,
                      primary_horizon_s=args.horizon, n_boot=args.n_boot, n_blocks=args.n_blocks)

        frames["scorecard"].append(rpt.scorecard_df(arm))
        frames["markout"].append(rpt.markout_df(arm))
        frames["breakeven"].append(rpt.breakeven_df(arm))
        vd = rpt.verdict_df(arm); frames["verdict"].append(vd)
        frames["stability"].append(rpt.stability_df(arm))

        summary["universes"][universe] = {
            "span_hours": span["hours"], "days": days, "n_markets": len(specs),
            "median_half_spread_cents": float(vd["half_spread_cents"].median()),
            "median_fill_rate_optimistic": float(vd["fill_rate_optimistic"].median()),
            "median_fill_rate_pessimistic": float(vd["fill_rate_pessimistic"].median()),
            "median_net_edge_opt_cents": float(vd["net_edge_opt_cents"].median()),
            "median_net_edge_pess_cents": float(vd["net_edge_pess_cents"].median()),
            "median_adverse_pess_cents": float(vd["adverse_pess_cents"].median()),
            "verdict_counts_no_rebate": vd["verdict_no_rebate"].value_counts().to_dict(),
            "verdict_counts_representative": vd["verdict_representative"].value_counts().to_dict(),
        }
        print(f"  verdicts (no-rebate): {summary['universes'][universe]['verdict_counts_no_rebate']}",
              flush=True)
        print(f"  median net edge ¢ Opt/RA: {vd['net_edge_opt_cents'].median():+.3f} / "
              f"{vd['net_edge_pess_cents'].median():+.3f}", flush=True)
        print(rpt.render_verdict_markdown(arm, universe), flush=True)

    con.close()

    out = {}
    for key, fr in frames.items():
        df = pd.concat(fr, ignore_index=True)
        df.to_csv(CSV_OUT / f"mm_validation_{key}.csv", index=False)
        out[key] = df

    # plots
    plot_markout_curves(out["markout"], PLOT_OUT / "mm_validation_markout_curves.png")
    plot_breakeven_scatter(out["verdict"], PLOT_OUT / "mm_validation_breakeven_scatter.png")

    # dormant overfitting demonstration on pooled per-token ND-PnL (optimistic, ex-rebate)
    pooled = out["scorecard"][out["scorecard"]["queue"] == OPTIMISTIC]["nd_pnl_ex_rebate"].dropna().to_numpy()
    dorm = dormant_status(pooled if pooled.size >= 3 else None)
    summary["overfitting_dormant"] = {
        "wired": dorm.wired, "n_trials": dorm.n_trials, "sr_star_haircut": dorm.sr_star_haircut,
        "pbo_available": dorm.pbo_available, "oos_split_available": dorm.oos_split_available,
        "markdown": dorm.to_markdown(),
    }

    (SCRATCH / "mm_validation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nCSVs -> {CSV_OUT}\nPlots -> {PLOT_OUT}")
    print(dorm.to_markdown())


if __name__ == "__main__":
    main()
