"""Render the SPCX listing-day (Fri 2026-06-12) decision-tree flowchart PNG.

Vertical timeline flowchart with 6 node boxes (D1..CLOSE-OUT) connected by
arrows, a two-way hedge branch between D2 and D3, and a full-width red
kill-switch banner at the bottom. Styled to match the dark SPCX dashboard
palette. Output: scripts/assets/spcx_decision_tree.png (~2200 px wide).

Run from polymarket/research/:
    PYTHONPATH=. uv run python scripts/make_spcx_decision_tree_chart.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---------------------------------------------------------------- palette ---
BG = "#0b0e12"
BOX_FILL = "#12161d"
BOX_EDGE = "#2d3845"
BODY = "#dde5ec"
WATCH = "#8b9aa8"
ARROW = "#5d6b78"
GREEN = "#34d399"
RED = "#f87171"

MONO = "Menlo"
TITLE_FS = 15.0
BODY_FS = 12.5
BRANCH_FS = 12.0
BANNER_FS = 16.0

PAD = 0.012  # FancyBboxPatch round pad (axes/data units)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "spcx_decision_tree.png")

# ------------------------------------------------------------------ nodes ---
# Each node: accent color, title, list of (text, color) body lines, height.
NODES = [
    dict(
        key="D1", accent="#a78bfa", height=0.082,
        title="D1 · THU ~22:00 — PRICING NIGHT",
        lines=[
            ("Final price prints (424B)", BODY),
            ("NO pre-hedge — exception: perp ≥ ~$183", BODY),
        ],
    ),
    dict(
        key="D2", accent="#38bdf8", height=0.142,
        title="D2 · FRI ~8:00 — ALLOCATION",
        lines=[
            ("Fill known — comfort zone: ~22 sh held long-only, never hedged", BODY),
            ("overflow above comfort AND net basis > 0 →", BODY),
            ("hedge the OVERFLOW only (cap ~21 sh @1.5× / ~14 @1×) at 8:00", BODY),
            ("expected case (fill ≤ 22 sh): NO hedge — margin stays free", BODY),
            ("tell Alvaro the split", BODY),
        ],
    ),
    dict(
        key="D3", accent="#38bdf8", height=0.082,
        title="D3 · 8:30–15:30 — PRE-OPEN",
        lines=[
            ("No action", BODY),
            ("watch: perp ≤ ~$140 = risk-off", WATCH),
        ],
    ),
    dict(
        key="D4", accent="#f5a623", height=0.082,
        title="D4 · 15:30 → CROSS — DISPLAY-ONLY",
        lines=[
            ("NEVER sell into the cross", BODY),
            ("log the first print", BODY),
        ],
    ),
    dict(
        key="D5", accent="#34d399", height=0.146,
        title="D5 · CROSS → 22:00 — POST-CROSS",
        lines=[
            ("observe 0–15 min", BODY),
            ("tranches: 40% by min 60 · 80% by 180 · 100% by close", BODY),
            ("front-load if AVWAP lost", BODY),
            ("pair-close hedge when |perp−spot| ≤ $2 for 15 min AND +60 min", BODY),
            ("FORCED-FLAT: perp must be 0 by 21:00 (no cross by 20:30 → close it)", BODY),
        ],
    ),
    dict(
        key="D6", accent="#f87171", height=0.066,
        title="CLOSE-OUT · 21:30–22:00",
        lines=[
            ("no new decisions · end state: perp 0, shares 0, nothing overnight", BODY),
        ],
    ),
]

BANNER_TEXT = "ANY TIME: SPOT ≤ $125 → SELL EVERYTHING"

# Vertical gaps below each node (the D2→D3 gap carries the branch visual).
GAP_DEFAULT = 0.034
GAP_BRANCH = 0.072
GAP_BANNER = 0.040

BOX_X0, BOX_X1 = 0.06, 0.94  # node box horizontal extent (axes coords)


def draw_node(ax, node, y_top):
    """Draw one node box with left accent stripe, title and body lines.

    Returns the y of the box bottom (excluding the round pad).
    """
    h = node["height"]
    y0 = y_top - h
    box = FancyBboxPatch(
        (BOX_X0, y0), BOX_X1 - BOX_X0, h,
        boxstyle=f"round,pad={PAD}",
        facecolor=BOX_FILL, edgecolor=BOX_EDGE, linewidth=1.5, zorder=2,
    )
    ax.add_patch(box)
    # Left accent stripe (thick colored left edge).
    ax.add_patch(Rectangle(
        (BOX_X0 - PAD + 0.0015, y0 - PAD + 0.004), 0.0065, h + 2 * PAD - 0.008,
        facecolor=node["accent"], edgecolor="none", zorder=3,
    ))
    tx = BOX_X0 + 0.022
    ax.text(tx, y_top - 0.014, node["title"],
            ha="left", va="top", fontsize=TITLE_FS, fontweight="bold",
            family=MONO, color=node["accent"], zorder=4)
    line_h = 0.0225
    y = y_top - 0.014 - 0.034
    for text, color in node["lines"]:
        ax.text(tx, y, text, ha="left", va="top", fontsize=BODY_FS,
                family=MONO, color=color, zorder=4)
        y -= line_h
    return y0


def draw_arrow(ax, y_from, y_to, x_from=0.5, x_to=0.5, color=ARROW, rad=0.0):
    ax.annotate(
        "", xy=(x_to, y_to), xytext=(x_from, y_from),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.8, mutation_scale=20,
            shrinkA=0, shrinkB=0,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=1,
    )


def main():
    fig = plt.figure(figsize=(11, 14), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    y_top = 0.985
    bottoms = []
    tops = []
    for i, node in enumerate(NODES):
        tops.append(y_top)
        y_bot = draw_node(ax, node, y_top)
        bottoms.append(y_bot)
        gap = GAP_BRANCH if node["key"] == "D2" else GAP_DEFAULT
        y_top = y_bot - gap - 2 * PAD

    # --- plain vertical arrows (all gaps except the D2→D3 branch gap) -------
    for i in range(len(NODES) - 1):
        if NODES[i]["key"] == "D2":
            continue
        draw_arrow(ax, bottoms[i] - PAD, tops[i + 1] + PAD)

    # --- D2 → D3 two-way branch: diverge from D2, re-join at D3 ------------
    d2_bot = bottoms[1] - PAD
    d3_top = tops[2] + PAD
    mid_y = (d2_bot + d3_top) / 2.0
    # left path: basis > 0 → HEDGE ON (green)
    draw_arrow(ax, d2_bot, d3_top, x_from=0.40, x_to=0.47,
               color=GREEN, rad=0.25)
    # right path: basis ≤ 0 → NO HEDGE (red)
    draw_arrow(ax, d2_bot, d3_top, x_from=0.60, x_to=0.53,
               color=RED, rad=-0.25)
    ax.text(0.355, mid_y, "overflow + basis > 0 → HEDGE OVERFLOW",
            ha="right", va="center", fontsize=BRANCH_FS, fontweight="bold",
            family=MONO, color=GREEN, zorder=4)
    ax.text(0.645, mid_y, "fill ≤ comfort or basis ≤ 0 → NO HEDGE",
            ha="left", va="center", fontsize=BRANCH_FS, fontweight="bold",
            family=MONO, color=RED, zorder=4)

    # --- red kill-switch banner ---------------------------------------------
    banner_h = 0.052
    banner_top = bottoms[-1] - GAP_BANNER - 2 * PAD
    banner_y0 = banner_top - banner_h
    ax.add_patch(FancyBboxPatch(
        (BOX_X0, banner_y0), BOX_X1 - BOX_X0, banner_h,
        boxstyle=f"round,pad={PAD}",
        facecolor="#2f1313", edgecolor=RED, linewidth=2.0, zorder=2,
    ))
    ax.text(0.5, banner_y0 + banner_h / 2.0, BANNER_TEXT,
            ha="center", va="center", fontsize=BANNER_FS, fontweight="bold",
            family=MONO, color=RED, zorder=4)

    # Fit the y-range to the actual layout so nothing clips at the axes edge.
    ax.set_ylim(banner_y0 - PAD - 0.012, 1.0)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=BG,
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    # --- acceptance report ---------------------------------------------------
    size_kb = os.path.getsize(OUT_PATH) / 1024.0
    print(f"wrote: {OUT_PATH}")
    print(f"size:  {size_kb:.1f} KB")
    try:
        from PIL import Image
        with Image.open(OUT_PATH) as im:
            print(f"pixels: {im.size[0]} x {im.size[1]}")
    except ImportError:
        print("pixels: PIL not available — file size reported above only")
    assert size_kb > 100, "PNG smaller than 100 KB — check rendering"


if __name__ == "__main__":
    main()
