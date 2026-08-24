"""Render the Task-5 ladder chart from the ladder CSV (post-hoc; no engine work).

One panel per category: each ladder config's OOS pooled per-contract costed net under the
three queue models (the bracket), with the IS value as a hollow marker — so selection-vs-
report gaps and bracket width are visible at a glance.

Usage (from polymarket/research/):
    PYTHONPATH=. uv run python scripts/mm_task5_plot_ladder.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESEARCH = Path(__file__).resolve().parents[1]
CSV_OUT = RESEARCH / "data/analysis/csv_outputs/market_making"
PLOT_OUT = RESEARCH / "data/analysis/plots/market_making"


def main() -> None:
    df = pd.read_csv(CSV_OUT / "mm_task5_ladder_table.csv")
    universes = list(df.universe.unique())
    fig, axes = plt.subplots(1, len(universes), figsize=(7.2 * len(universes), 5.2))
    axes = np.atleast_1d(axes)
    for ax, u in zip(axes, universes):
        sub = df[df.universe == u].reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        ax.scatter(sub["OOS_pooled_c_Optimistic"], y, marker="^", color="#4878d0",
                   label="OOS Optimistic", zorder=3)
        ax.scatter(sub["OOS_pooled_c_Prob"], y, marker="o", color="#888",
                   label="OOS Prob(0.5)", zorder=3)
        ax.scatter(sub["OOS_pooled_c_RiskAverse"], y, marker="v", color="#d65f5f",
                   label="OOS RiskAverse", zorder=3)
        for yi, (o, r) in zip(y, zip(sub["OOS_pooled_c_Optimistic"],
                                     sub["OOS_pooled_c_RiskAverse"])):
            ax.plot([o, r], [yi, yi], color="#bbb", lw=1.2, zorder=2)
        ax.scatter(sub["IS_pooled_c"], y, marker="s", facecolors="none",
                   edgecolors="#333", label="IS (selection window)", zorder=3)
        ax.axvline(0, color="black", lw=0.8)
        labels = [f"{c}\n[{k}]" for c, k in zip(sub["config"], sub["keep"].str.split(" ").str[0])]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("pooled costed net ¢/contract")
        ax.set_title(u)
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("Task-5 ladder — IS selection vs OOS queue bracket (costed ¢/contract; KEPT/DROP per config)",
                 y=1.02)
    fig.tight_layout()
    out = PLOT_OUT / "mm_task5_ladder_oos.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
