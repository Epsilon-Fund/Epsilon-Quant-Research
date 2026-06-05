"""Build notebooks/weather_chop_by_city.ipynb.

Simple, fast notebook: chop rate per (city × month × barrier) over the
highest-temperature daily markets, plus 2-3 clean charts so the user can
eyeball which city has been the choppiest recently.

Definition: chop = `crossed_and_crashed` from
`data/analysis/weather_tail_per_instance.parquet` (the price crossed up
through the barrier and then resolved to zero — i.e., a head-fake / reversal).
Chop rate within a (city, month, barrier) cell = sum(crashed) / sum(crossed).

Run: python3 polymarket/research/scripts/build_weather_chop_by_city_notebook.py
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "weather_chop_by_city.ipynb"


def md(src: str): return nbf.v4.new_markdown_cell(src.strip("\n"))
def code(src: str):
    c = nbf.v4.new_code_cell(src.strip("\n"))
    c["execution_count"] = None
    c["outputs"] = []
    return c


cells: list = []

cells.append(md("""
# Weather chop-rate by city × month × threshold

**Chop** = `crossed_and_crashed` in
`data/analysis/weather_tail_per_instance.parquet`: price crossed up through the
barrier (entry signal fired) and the market then resolved to zero. I.e., the
tail-event signal head-faked.

**Chop rate** in a (city, month, barrier) cell = `Σcrashed / Σcrossed`.
Only cells with ≥10 crosses are kept (avoid 0/1 noise).

Restricted to `highest-temperature-in-<city>` markets — that's where the
volume is. Cities are kept if they have ≥500 resolved instances overall.
"""))

cells.append(code("""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
# Notebook may be opened from polymarket/research/ or anywhere; auto-locate.
while not (ROOT / "data" / "analysis" / "weather_tail_per_instance.parquet").exists():
    if ROOT.parent == ROOT:
        raise SystemExit("can't find data/analysis/weather_tail_per_instance.parquet")
    ROOT = ROOT.parent
PARQUET = ROOT / "data" / "analysis" / "weather_tail_per_instance.parquet"

df = pd.read_parquet(PARQUET, columns=[
    "slug_family", "end_ts", "barrier_price", "crossed", "crossed_and_crashed",
])
# Restrict to highest-temp cities (lowest-temp has ~1% of the volume).
df = df[df["slug_family"].str.startswith("highest-temperature-in-")].copy()
df["city"] = df["slug_family"].str.replace("highest-temperature-in-", "", regex=False)

df["end_ts"] = pd.to_datetime(df["end_ts"])
df["month"] = df["end_ts"].dt.to_period("M").astype(str)
print(f"rows: {len(df):,}")
print(f"date range: {df['end_ts'].min().date()} → {df['end_ts'].max().date()}")
print(f"distinct cities: {df['city'].nunique()}")


def heatmap(df_pivot, title, xlabel, ylabel, cbar_label="chop rate",
            cmap="magma_r", annot_fmt=".2f", figsize=(10, 6)):
    \"\"\"Minimal seaborn-free heatmap helper. df_pivot is a pandas DataFrame
    where rows are y-axis labels, columns are x-axis labels.\"\"\"
    fig, ax = plt.subplots(figsize=figsize)
    arr = df_pivot.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap=cmap)
    ax.set_xticks(range(df_pivot.shape[1]))
    ax.set_xticklabels(df_pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(df_pivot.shape[0]))
    ax.set_yticklabels(df_pivot.index)
    if annot_fmt:
        vmax = np.nanmax(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if not np.isnan(v):
                    col = "white" if v > (vmax * 0.55) else "black"
                    ax.text(j, i, format(v, annot_fmt), ha="center", va="center",
                            color=col, fontsize=8)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig, ax
"""))

cells.append(md("""
## Group: chop rate per (city × month × barrier)
"""))

cells.append(code("""
g = (df.groupby(["city", "month", "barrier_price"], as_index=False)
       .agg(n_total=("crossed", "size"),
            n_crossed=("crossed", "sum"),
            n_crashed=("crossed_and_crashed", "sum")))
g["chop_rate"] = g["n_crashed"] / g["n_crossed"].where(g["n_crossed"] > 0)
# Min-cross floor to suppress 0/1 noise.
g = g[g["n_crossed"] >= 10].copy()

# Keep cities with enough overall mass (≥500 resolved instances).
city_n = df.groupby("city").size().sort_values(ascending=False)
big_cities = city_n[city_n >= 500].index.tolist()
print(f"cities kept (≥500 instances): {len(big_cities)}")
g = g[g["city"].isin(big_cities)]
print(g.head())
"""))

cells.append(md("""
## Chart 1 — city × month heatmap at barrier 0.85

Top-15 cities by overall (last-6-months) chop rate. Cells coloured by chop
rate; blank where fewer than 10 crosses in that month.
"""))

cells.append(code("""
BARRIER = 0.85
sub = g[g["barrier_price"] == BARRIER].copy()
# Last 6 months in the data:
last6 = sorted(sub["month"].unique())[-6:]
sub = sub[sub["month"].isin(last6)]

# rank cities by overall chop rate in the window
rank = (sub.groupby("city").apply(
    lambda d: d["n_crashed"].sum() / max(d["n_crossed"].sum(), 1)
).sort_values(ascending=False))
top_cities = rank.head(15).index.tolist()
print(f"Top-15 cities by overall chop rate (last 6 months, barrier {BARRIER}):")
print(rank.head(15).round(3).to_string())

pivot = (sub[sub["city"].isin(top_cities)]
         .pivot_table(index="city", columns="month", values="chop_rate")
         .reindex(top_cities).reindex(columns=last6))
heatmap(pivot,
        title=f"Chop rate by city × month (barrier {BARRIER}, last 6 months)",
        xlabel="end-month",
        ylabel="city (top-15 by overall chop rate)")
plt.show()
"""))

cells.append(md("""
## Chart 2 — city × barrier heatmap, last 3 months pooled

Pools chop counts across the 3 most-recent months, then computes chop rate
per (city, barrier). Top-15 cities by pooled chop rate.
"""))

cells.append(code("""
last3 = sorted(g["month"].unique())[-3:]
sub3 = g[g["month"].isin(last3)].copy()

pool = (sub3.groupby(["city", "barrier_price"])
            .agg(n_crossed=("n_crossed", "sum"),
                 n_crashed=("n_crashed", "sum"))
            .reset_index())
pool = pool[pool["n_crossed"] >= 20]   # firmer min-cross for the pooled view
pool["chop_rate"] = pool["n_crashed"] / pool["n_crossed"]

rank3 = (pool.groupby("city").apply(
    lambda d: d["n_crashed"].sum() / d["n_crossed"].sum()
).sort_values(ascending=False))
top15 = rank3.head(15).index.tolist()

pivot2 = (pool[pool["city"].isin(top15)]
          .pivot_table(index="city", columns="barrier_price", values="chop_rate")
          .reindex(top15))
heatmap(pivot2,
        title=f"Chop rate by city × barrier (months: {', '.join(last3)}, pooled)",
        xlabel="barrier price",
        ylabel="city (top-15 by pooled chop rate)",
        figsize=(9, 6))
plt.show()
"""))

cells.append(md("""
## Chart 3 — monthly chop-rate trend, ALL cities at barrier 0.85

One line per city, ordered/coloured by overall chop rate (choppiest = dark red).
With ~50 cities this gets visually busy — use it to spot trend direction, not
individual cities; for picking out a single city use the heatmap above.
"""))

cells.append(code("""
BARRIER = 0.85
sub = g[g["barrier_price"] == BARRIER].copy()
rank_all = (sub.groupby("city").apply(
    lambda d: d["n_crashed"].sum() / max(d["n_crossed"].sum(), 1)
).sort_values(ascending=False))

trend = (sub.pivot_table(index="month", columns="city", values="chop_rate")
            .reindex(columns=rank_all.index).sort_index())

# Colour each city by its rank — choppiest dark, calmest light.
cmap = plt.get_cmap("magma_r")
colors = {c: cmap(0.1 + 0.8 * i / max(len(rank_all) - 1, 1))
          for i, c in enumerate(rank_all.index)}

fig, ax = plt.subplots(figsize=(13, 7))
for c in rank_all.index:
    ax.plot(trend.index, trend[c], marker="o", markersize=3, linewidth=1.0,
            alpha=0.7, color=colors[c], label=c)
ax.set_title(f"Monthly chop rate by city (barrier {BARRIER}, all {len(rank_all)} cities)")
ax.set_xlabel("end-month")
ax.set_ylabel("chop rate")
ax.grid(True, alpha=0.3)
# Multi-column legend so it fits.
ax.legend(title="city (ranked)", bbox_to_anchor=(1.02, 1), loc="upper left",
          fontsize=7, ncol=2)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
## Chart 4 — chop rate vs threshold

X-axis: barrier price (the available barrier prices in the dataset are 0.50,
0.60, 0.70, 0.80, 0.85, 0.90, 0.95 — close to "0.6 → 0.95 in 0.05 steps"
modulo the dataset's grid).  Y-axis: chop rate.  One line per top-6 city plus
a thick "ALL cities" pooled line so you can see the shape vs the noise.

Pooled across the last 3 months.
"""))

cells.append(code("""
last3 = sorted(g["month"].unique())[-3:]
sub_t = g[g["month"].isin(last3)].copy()

# Pool over months, keep all cities, all barriers in the requested range.
pool_t = (sub_t.groupby(["city", "barrier_price"])
                .agg(n_crossed=("n_crossed", "sum"),
                     n_crashed=("n_crashed", "sum"))
                .reset_index())
pool_t = pool_t[pool_t["barrier_price"] >= 0.60]
pool_t = pool_t[pool_t["n_crossed"] >= 20]
pool_t["chop_rate"] = pool_t["n_crashed"] / pool_t["n_crossed"]

# All cities ranked by overall pooled chop rate.
rank_t = (pool_t.groupby("city")
                .apply(lambda d: d["n_crashed"].sum() / d["n_crossed"].sum())
                .sort_values(ascending=False))

curve = (pool_t.pivot_table(index="barrier_price", columns="city", values="chop_rate")
                .reindex(columns=rank_t.index).sort_index())

# Overall pooled line across all kept cities.
overall = (pool_t.groupby("barrier_price")
                  .apply(lambda d: d["n_crashed"].sum() / d["n_crossed"].sum()))

cmap = plt.get_cmap("magma_r")
colors = {c: cmap(0.1 + 0.8 * i / max(len(rank_t) - 1, 1))
          for i, c in enumerate(rank_t.index)}

fig, ax = plt.subplots(figsize=(13, 7))
for c in rank_t.index:
    ax.plot(curve.index, curve[c], marker="o", markersize=3, linewidth=1.0,
            alpha=0.7, color=colors[c], label=c)
ax.plot(overall.index, overall.values, marker="s", linewidth=3.0, color="black",
        label="ALL (pooled)")
ax.set_title(f"Chop rate vs threshold — all {len(rank_t)} cities + pooled, months {', '.join(last3)}")
ax.set_xlabel("barrier price")
ax.set_ylabel("chop rate")
ax.grid(True, alpha=0.3)
ax.legend(title="city (ranked)", bbox_to_anchor=(1.02, 1), loc="upper left",
          fontsize=7, ncol=2)
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
## All cities ranked — last 3 months pooled (all barriers)
"""))

cells.append(code("""
out = (rank3.rename("chop_rate").to_frame()
       .merge(pool.groupby("city")["n_crossed"].sum().rename("n_crossed_pooled"),
              left_index=True, right_index=True)
       .merge(pool.groupby("city")["n_crashed"].sum().rename("n_crashed_pooled"),
              left_index=True, right_index=True)
       .round(3))
out
"""))


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB_PATH)
    print(f"wrote {NB_PATH} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
