#!/usr/bin/env python3
"""Build the README research map image.

The image mirrors the "Documented research" README tables. It is intentionally
not a size, priority, or note-count chart.
"""

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUT = Path(__file__).with_name("readme_research_map.png")

POLYMARKET = [
    ("Copy-trading", "Wallet/cohort skill, copied-entry realism, and PnL persistence."),
    ("Market-making", "Passive quoting, spread capture, queue position, capacity, adverse selection."),
    ("Options / fair value", "Binary prices versus external references, volatility, settlement, fair value."),
    ("Order-flow microstructure", "CLOB/L2 state, OFI/TFI, lead-lag, sign conventions, tradability."),
    ("Execution", "Midas paper/live stack, maker engine, risk controls, journals, runbooks."),
]

CRYPTO = [
    ("Live momentum", "Multi-asset Binance trend system, dashboards, and production parameters."),
    ("Breakout research", "Volatility-expansion strategies with walk-forward and CPCV validation."),
    ("Statistical arbitrage", "Pairs and relative-value research, notebooks, and strategy stubs."),
    ("Regime and ML filters", "BTC regimes, XGBoost filters, overlays for gating and sizing."),
    ("Validation infrastructure", "Walk-forward, CPCV, portfolio aggregation, and shared metrics."),
    ("Exploratory sleeves", "Cross-sectional momentum, long/short, memecoin/DeFi, and early ideas."),
]


PALETTE = {
    "ink": "#1f2933",
    "muted": "#637083",
    "line": "#d8dee8",
    "panel": "#ffffff",
    "bg": "#f8fafc",
    "poly": "#2563eb",
    "poly_soft": "#dbeafe",
    "crypto": "#0f766e",
    "crypto_soft": "#ccfbf1",
    "gold": "#d97706",
    "rose": "#e11d48",
    "violet": "#7c3aed",
}


def add_wrapped_text(ax, x, y, text, width, **kwargs):
    ax.text(x, y, "\n".join(wrap(text, width=width)), **kwargs)


def draw_panel(ax, x, y, w, h, title, subtitle, items, accent, soft, footer):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.1,
            edgecolor=PALETTE["line"],
            facecolor=PALETTE["panel"],
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.018, y + h - 0.116),
            w - 0.036,
            0.082,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=0,
            facecolor=soft,
        )
    )
    ax.add_patch(Rectangle((x + 0.035, y + h - 0.105), 0.012, 0.06, color=accent, linewidth=0))
    ax.text(x + 0.058, y + h - 0.067, title, fontsize=17, fontweight="bold", color=PALETTE["ink"], va="center")
    ax.text(x + w - 0.04, y + h - 0.067, subtitle, fontsize=9.5, color=PALETTE["muted"], ha="right", va="center")

    start_y = y + h - 0.165
    step = (h - 0.235) / len(items)
    row_h = min(0.094, step - 0.012)
    marker_colors = [accent, PALETTE["gold"], PALETTE["violet"], PALETTE["rose"], "#16a34a", "#0891b2"]

    for idx, (name, desc) in enumerate(items):
        row_y = start_y - idx * step
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.035, row_y - row_h + 0.02),
                w - 0.07,
                row_h,
                boxstyle="round,pad=0.004,rounding_size=0.010",
                linewidth=0.7,
                edgecolor="#eef2f7",
                facecolor="#fbfdff",
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.055, row_y - 0.033),
                0.022,
                0.022,
                boxstyle="round,pad=0.003,rounding_size=0.004",
                linewidth=0,
                facecolor=marker_colors[idx % len(marker_colors)],
            )
        )
        ax.text(x + 0.09, row_y - 0.008, name, fontsize=12.2, fontweight="bold", color=PALETTE["ink"], va="center")
        add_wrapped_text(
            ax,
            x + 0.09,
            row_y - 0.041,
            desc,
            58,
            fontsize=8.9,
            color=PALETTE["muted"],
            va="top",
            linespacing=1.18,
        )

    ax.plot([x + 0.035, x + w - 0.035], [y + 0.066, y + 0.066], color="#eef2f7", linewidth=1.0)
    ax.text(x + 0.04, y + 0.035, footer, fontsize=9.3, color=PALETTE["muted"], va="center")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "savefig.dpi": 190,
        }
    )
    fig = plt.figure(figsize=(13.2, 8.2), facecolor=PALETTE["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.93, "Documented research map", fontsize=24, fontweight="bold", color=PALETTE["ink"])
    ax.text(
        0.05,
        0.885,
        "A visual companion to the README tables above. This is not a size or priority chart.",
        fontsize=12.2,
        color=PALETTE["muted"],
    )

    ax.add_patch(Rectangle((0.05, 0.855), 0.42, 0.004, color=PALETTE["poly"], linewidth=0))
    ax.add_patch(Rectangle((0.53, 0.855), 0.42, 0.004, color=PALETTE["crypto"], linewidth=0))

    draw_panel(
        ax,
        0.05,
        0.055,
        0.42,
        0.79,
        "Polymarket",
        "prediction-market branch",
        POLYMARKET,
        PALETTE["poly"],
        PALETTE["poly_soft"],
        "Main roots: polymarket/research/, polymarket/execution/, midas/",
    )
    draw_panel(
        ax,
        0.53,
        0.055,
        0.42,
        0.79,
        "Crypto",
        "systematic Binance branch",
        CRYPTO,
        PALETTE["crypto"],
        PALETTE["crypto_soft"],
        "Main roots: live_trading/, topics/, infrastructure/, docs/",
    )

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.12, facecolor=PALETTE["bg"])
    plt.close(fig)


if __name__ == "__main__":
    main()
