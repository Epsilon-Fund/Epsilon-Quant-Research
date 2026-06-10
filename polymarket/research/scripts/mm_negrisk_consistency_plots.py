"""Figures for the MM NegRisk basket-consistency scanner findings note.

Reads the analyzer CSVs plus the raw poller run dir and writes PNGs to
data/analysis/plots/market_making/.

Run from polymarket/research after mm_negrisk_consistency_analyze.py:
  PYTHONPATH=. uv run python scripts/mm_negrisk_consistency_plots.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_DIR = Path("data/analysis/csv_outputs/market_making")
PLOT_DIR = Path("data/analysis/plots/market_making")


def latest_complete_snapshot(run_dir: Path) -> pd.DataFrame:
    """Last cycle's per-event sums."""
    rows = []
    with open(run_dir / "cycles.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["cycle"], r["event_id"], r["ask_sum"], r["bid_sum"]))
    df = pd.DataFrame(rows, columns=["cycle", "event_id", "ask_sum", "bid_sum"])
    last = df["cycle"].max()
    return df[df["cycle"] == last]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    snap = latest_complete_snapshot(args.run_dir)
    ep = pd.read_csv(CSV_DIR / "mm_negrisk_consistency_episodes.csv")

    # 1. Consistency surface: histogram of basket sums at the last cycle.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    asks = snap["ask_sum"].dropna()
    asks = asks[(asks > 0.85) & (asks < 1.6)]
    axes[0].hist(asks, bins=120, color="#3a6ea5")
    axes[0].axvline(1.0, color="red", lw=1, ls="--")
    axes[0].set_title(f"Sum of best asks per complete NegRisk basket (n={len(asks)})")
    axes[0].set_xlabel("sum of best asks ($ to buy every leg)")
    axes[0].set_ylabel("events")
    bids = snap["bid_sum"].dropna()
    bids = bids[(bids > 0.4) & (bids < 1.2)]
    axes[1].hist(bids, bins=120, color="#3a875a")
    axes[1].axvline(1.0, color="red", lw=1, ls="--")
    axes[1].set_title(f"Sum of best bids per basket, conservative (n={len(bids)})")
    axes[1].set_xlabel("sum of best bids ($ proceeds to sell every leg)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mm_negrisk_consistency_sum_histograms.png", dpi=150)
    plt.close(fig)

    if not len(ep):
        print("no episodes; only histograms written")
        return

    # 2. Persistence: episode minimum duration, with right-censoring split.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    closed = ep[~ep["open_at_run_end"]]
    open_ep = ep[ep["open_at_run_end"]]
    bins = np.arange(0, max(ep["min_duration_s"].max(), 600) + 240, 120)
    ax.hist([closed["min_duration_s"], open_ep["min_duration_s"]], bins=bins, stacked=True,
            label=[f"closed within run (n={len(closed)})", f"still open at run end / censored (n={len(open_ep)})"],
            color=["#3a6ea5", "#c4a35a"])
    ax.set_xlabel("violation episode duration, seconds (poll cadence ~120s; lower bound)")
    ax.set_ylabel("episodes")
    ax.set_title("How long do NegRisk basket violations survive?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mm_negrisk_consistency_persistence.png", dpi=150)
    plt.close(fig)

    # 3. Edge vs executable size at top-of-book.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"buy_all": "#3a6ea5", "sell_all": "#3a875a"}
    for d, sub in ep.groupby("direction"):
        sz = sub["min_basket_shares_median"].clip(lower=0.05)
        ax.scatter(sz, sub["edge_mean_c"], s=22, alpha=0.7, label=d, color=colors.get(d))
    ax.set_xscale("log")
    ax.set_xlabel("baskets assemblable at top-of-book (min shares across legs, log scale)")
    ax.set_ylabel("gross violation edge (cents per basket)")
    ax.set_title("Violation size vs executable depth — is anything both big and deep?")
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mm_negrisk_consistency_edge_vs_depth.png", dpi=150)
    plt.close(fig)

    # 4. Gross vs net-of-fee edge by category.
    agg = ep.groupby(["direction", "category"]).agg(
        gross=("edge_mean_c", "mean"), net_repo=("net_edge_repo_c", "mean"), net_harsh=("net_edge_harsh_c", "mean"),
        n=("event_id", "size"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(agg))
    w = 0.27
    ax.bar(x - w, agg["gross"], w, label="gross edge", color="#3a6ea5")
    ax.bar(x, agg["net_repo"], w, label="net, repo fee schedule (5% x p(1-p))", color="#c4a35a")
    ax.bar(x + w, agg["net_harsh"], w, label="net, Gamma-declared (10% x min(p,1-p))", color="#a54a3a")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.direction[:4]}:{r.category}\nn={r.n}" for r in agg.itertuples()], fontsize=8)
    ax.set_ylabel("edge, cents per basket")
    ax.set_title("Gross vs net-of-fee violation edge by direction and event family")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mm_negrisk_consistency_gross_vs_net.png", dpi=150)
    plt.close(fig)

    print("plots written to", PLOT_DIR)


if __name__ == "__main__":
    main()
