"""Task-5.1 findings charts (PRD §10) — split diagnosis, surface, IS→OOS, toxicity trace, inventory.

Five charts for the findings note + the human audit:

1. **Split diagnosis** — every market's observed lifecycle as a timeline bar, colored by
   CPCV fold, with the OLD Task-5 calendar cut (2026-06-24) overlaid: shows at a glance
   why "one vertical line through concurrent lifecycles" made OOS = the endgame, and how
   whole-market folds keep each market's full τ arc together.
2. **(cohort × τ-regime) heatmap** — the kept rung's costed ¢/contract per cell
   (politics: midlife/approach/endgame; esports: pre/in-play) — where edge lives vs dies.
3. **Inner→outer scatter colored by cap** — per config: training-groups pooled vs
   held-out-groups pooled (mean over CPCV splits). The Task-5 tell ("drop is uniform and
   scales with cap") re-tested under the fixed split.
4. **Toxicity trace** — Lens 1 (sweep/VPIN) + Lens 2 (AS z) through the sample's biggest
   spike episode, with the graduated-response firing points (LOTECH-style).
5. **Inventory path** — symmetric baseline vs NeutralSpikeQuoter position through the
   same spike (one-sided stacking vs neutral avoidance).

Usage (from polymarket/research/):
    PYTHONPATH=. uv run python scripts/mm_task5_1_charts.py [--charts 1,2,3,4,5]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from mm_engine import (BACKTEST, ConstantLatency, FeeModel, NeutralSpikeQuoter,
                       RiskAverseQueue, SymmetricQuoter, Telemetry, run_engine)
from mm_engine.feeds.replay_parquet import replay_parquet
from mm_engine.telemetry import JsonlSink
from mm_eval import cpcv
from mm_eval.metrics import CENTS
from mm_eval.tape import TradeTape, tape_feed

RESEARCH = Path(__file__).resolve().parents[1]
CSV_OUT = RESEARCH / "data/analysis/csv_outputs/market_making"
PLOT_OUT = RESEARCH / "data/analysis/plots/market_making"
SELECTION_JSON = RESEARCH / "data/markets/mm_task5_1_selection.json"
SCRATCH = Path("/private/tmp/claude-501/-Users-justiniturregui-Desktop-github-epsilon-quant-research/"
               "b6ea1a3f-cca4-465b-b11c-5ab18e4a749c/scratchpad")
CACHE = SCRATCH / "mm_task5_1_cache"
UNIVERSES = ("politics_negrisk", "esports")
OLD_SPLIT_MS = int(datetime(2026, 6, 24, tzinfo=timezone.utc).timestamp() * 1000)


def _dt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


# ── 1. split diagnosis ─────────────────────────────────────────────────────────

def chart_split_diagnosis(groups: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    cmap = plt.get_cmap("tab10")
    for ax, u in zip(axes, UNIVERSES):
        g = groups[groups.universe == u].sort_values("t_first").reset_index(drop=True)
        for i, r in g.iterrows():
            ax.barh(i, (_dt(r.t_last) - _dt(r.t_first)).total_seconds() / 86400,
                    left=mdates.date2num(_dt(r.t_first)), height=0.62,
                    color=cmap(int(r.fold) % 10), alpha=0.85)
            ax.barh(i, (r.lead_in_b - r.lead_in_a) / 86400e3,
                    left=mdates.date2num(_dt(r.lead_in_a)), height=0.62,
                    color="black", alpha=0.35)
            ax.text(mdates.date2num(_dt(r.t_last)) + 0.1, i,
                    f"fold {int(r.fold)} · {r.cohort_aggr[:4]}", fontsize=6.5, va="center")
        ax.axvline(mdates.date2num(_dt(OLD_SPLIT_MS)), color="red", ls="--", lw=1.6)
        ax.text(mdates.date2num(_dt(OLD_SPLIT_MS)), len(g) - 0.3, " old Task-5 cut\n (06-24)",
                color="red", fontsize=8, va="top")
        ax.set_title(f"{u}: {len(g)} whole-market groups")
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.suptitle("Split diagnosis: whole-market CPCV folds (colors) vs the old calendar cut (red) — "
                 "each bar is one event group's observed lifecycle; black head = leakage-safe lead-in")
    fig.tight_layout()
    fig.savefig(PLOT_OUT / "mm_task5_1_split_diagnosis.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── 2. cohort × τ-regime heatmap (kept rung) ───────────────────────────────────

def chart_surface() -> None:
    f = CSV_OUT / "mm_task5_1_surface.parquet"
    if not f.exists():
        print("surface.parquet missing — run the ladder first; skipping chart 2")
        return
    s = pd.read_parquet(f)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, u in zip(axes, UNIVERSES):
        sub = s[(s.universe == u) & (s.regime != "full")]
        if sub.empty:
            ax.set_title(f"{u} (no data)")
            continue
        agg = (sub.groupby(["cohort_aggr", "regime"])
               .agg(usd=("modal_usd", "sum"), qty=("modal_qty", "sum"),
                    n=("group_id", "nunique")).reset_index())
        agg["c"] = agg.usd / agg.qty.replace(0, np.nan) * CENTS
        order = [f"tau_{n}" for n, *_ in cpcv.TAU_REGIMES[u]][::-1]
        piv = agg.pivot_table(index="cohort_aggr", columns="regime", values="c") \
                 .reindex(columns=[c for c in order if c in agg.regime.unique()])
        im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", vmin=-3, vmax=3, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([c.replace("tau_", "") for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iloc[i, j]
                cell = agg[(agg.cohort_aggr == piv.index[i]) & (agg.regime == piv.columns[j])]
                n = int(cell.n.iloc[0]) if len(cell) else 0
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}¢\n{n} grp", ha="center", va="center", fontsize=8)
        ax.set_title(u)
        fig.colorbar(im, ax=ax, shrink=0.8, label="costed ¢/contract")
    fig.suptitle("Kept-rung performance surface: cohort (aggressiveness) × τ-regime "
                 "(pessimistic queue, modal config, pooled over groups)")
    fig.tight_layout()
    fig.savefig(PLOT_OUT / "mm_task5_1_surface_heatmap.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── 3. inner→outer scatter by cap ──────────────────────────────────────────────

def chart_inner_outer() -> None:
    f = SCRATCH / "mm_task5_1_all_rows.json"
    g = CSV_OUT / "mm_task5_1_groups.parquet"
    sp = CSV_OUT / "mm_task5_1_splits.parquet"
    if not (f.exists() and sp.exists()):
        print("ladder outputs missing — skipping chart 3")
        return
    df = pd.read_json(f)
    groups = pd.read_parquet(g)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, u in zip(axes, UNIVERSES):
        gsub = groups[groups.universe == u].sort_values(["fold", "order_idx"])
        n_folds = int(gsub["n_folds"].iloc[0])
        fold_sizes = [int((gsub["fold"] == k).sum()) for k in range(n_folds)]
        splits = cpcv.generate_group_cpcv_splits(len(gsub), n_folds, 2,
                                                 fold_sizes=fold_sizes)
        o2g = {i: gid for i, gid in enumerate(gsub["group_id"].tolist())}
        pess = df[(df.universe == u) & (df.queue == "RiskAverse")
                  & (df.kind.isin(["nsq", "symmetric"]))]
        usd = pess.pivot_table(index="group_id", columns="config", values="full_usd",
                               aggfunc="sum")
        qty = pess.pivot_table(index="group_id", columns="config", values="full_qty",
                               aggfunc="sum")
        pts = []
        for cfg in usd.columns:
            cap = np.inf
            if "cap=" in cfg:
                cap = float(cfg.split("cap=")[1].split(",")[0].rstrip("]"))
            tr_vals, te_vals = [], []
            for s in splits["splits"]:
                tr = [o2g[i] for i in s["train_groups"] if o2g[i] in usd.index]
                te = [o2g[i] for fl in s["test_groups_by_fold"].values() for i in fl
                      if o2g[i] in usd.index]
                if not tr or not te:
                    continue
                qt, qe = qty.loc[tr, cfg].sum(), qty.loc[te, cfg].sum()
                if qt > 0 and qe > 0:
                    tr_vals.append(usd.loc[tr, cfg].sum() / qt * CENTS)
                    te_vals.append(usd.loc[te, cfg].sum() / qe * CENTS)
            if tr_vals:
                pts.append((float(np.mean(tr_vals)), float(np.mean(te_vals)), cap))
        if not pts:
            continue
        P = pd.DataFrame(pts, columns=["inner", "outer", "cap"])
        caps = sorted(P.cap.unique())
        colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(caps)))
        for cap, col in zip(caps, colors):
            s = P[P.cap == cap]
            ax.scatter(s.inner, s.outer, color=col, s=42, alpha=0.85,
                       label=f"cap={cap:g}" if np.isfinite(cap) else "baseline (no cap)")
        lim = max(abs(P.inner).max(), abs(P.outer).max()) * 1.15
        ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.7, ls=":")
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("training-groups pooled ¢/ct (inner)")
        ax.set_ylabel("held-out-groups pooled ¢/ct (outer)")
        ax.set_title(u)
        ax.legend(fontsize=7)
    fig.suptitle("Inner→outer transfer per config, colored by inventory cap — the Task-5 "
                 "'drop scales with cap' tell re-tested under the whole-market split")
    fig.tight_layout()
    fig.savefig(PLOT_OUT / "mm_task5_1_inner_outer_by_cap.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── 4+5. spike episode: toxicity trace + inventory paths ──────────────────────

def find_spike_token(sel: dict) -> tuple[str, str, tuple[int, int]]:
    """The token with the largest 30-min |mid move| (politics preferred), and its window."""
    con = duckdb.connect()
    best = None
    for u in UNIVERSES:
        for t in sel["universes"][u]["tokens"]:
            tdir = CACHE / u / t["token_id"]
            try:
                row = con.execute(
                    "WITH m AS (SELECT timestamp_ms//1800000 AS b, "
                    " avg((best_bid+best_ask)/2.0) AS mid FROM read_parquet(?) "
                    " WHERE best_bid IS NOT NULL AND best_ask IS NOT NULL GROUP BY 1), "
                    "d AS (SELECT b, mid - lag(mid) OVER (ORDER BY b) AS dm FROM m) "
                    "SELECT b, abs(dm) FROM d WHERE dm IS NOT NULL "
                    "ORDER BY abs(dm) DESC LIMIT 1",
                    [str(tdir / "bba_x.parquet")]).fetchone()
            except Exception:
                continue
            if row is None:
                continue
            b, move = row
            score = move * (1.5 if u == "politics_negrisk" else 1.0)
            if best is None or score > best[0]:
                a = int(b) * 1800000 - 3 * 3600000
                z = int(b) * 1800000 + 3 * 3600000
                best = (score, u, t["token_id"], (a, z))
    con.close()
    return best[1], best[2], best[3]


def chart_spike(sel: dict) -> None:
    u, tok, (a, b) = find_spike_token(sel)
    print(f"spike episode: {u} {tok[:14]}… window {_dt(a):%m-%d %H:%M} → {_dt(b):%H:%M} UTC")
    tdir = CACHE / u / tok
    spec_hs = next(t["half_spread"] for t in sel["universes"][u]["tokens"]
                   if t["token_id"] == tok)
    events = list(replay_parquet(tdir, gaps=[]))

    runs = {}
    for name, strat, extra in (
            ("baseline", SymmetricQuoter, {}),
            ("nsq", NeutralSpikeQuoter,
             {"skew_k": 2e-5, "inv_cap": 200.0, "nsq_lens1": True, "nsq_lens2": True})):
        tape = TradeTape()
        params = {"half_spread": spec_hs, "size": 100.0, "tick": 0.001,
                  "trade_tape": tape, **extra}
        tele = Telemetry(fills=JsonlSink(keep=True), orders=JsonlSink(keep=False),
                         quotes=JsonlSink(keep=False))
        strategy = strat()
        # instrument the NSQ decision state through the spike window
        trace = []
        if name == "nsq":
            orig_quote = strategy.quote

            def quote_traced(book, inventory, params, _s=strategy, _o=orig_quote, _tr=trace):
                out = _o(book, inventory, params)
                if a <= book.ts_exchange <= b:
                    d, ex = (_s._vpin.flag_and_side(_s._cfg) if _s._cfg and _s._cfg["nsq_lens1"]
                             else (False, None))
                    _tr.append((book.ts_exchange, _s._vpin.reading(),
                                _s._asz.last_z, d, ex,
                                _s._asz.flag(book.ts_exchange, _s._cfg) if _s._cfg else False,
                                inventory))
                return out
            strategy.quote = quote_traced
        r = run_engine(tape_feed(iter(events), tape), strategy=strategy,
                       queue_model=RiskAverseQueue(), latency_model=ConstantLatency(0.0),
                       mode=BACKTEST, params=params, fee_model=FeeModel(), telemetry=tele)
        runs[name] = {"fills": r.fills, "trace": trace}

    con = duckdb.connect()
    mids = con.execute(
        "SELECT timestamp_ms, (best_bid+best_ask)/2.0 AS mid FROM read_parquet(?) "
        "WHERE timestamp_ms BETWEEN ? AND ? AND best_bid IS NOT NULL AND best_ask IS NOT NULL "
        "ORDER BY timestamp_ms", [str(tdir / "bba_x.parquet"), a, b]).df()
    con.close()

    tr = pd.DataFrame(runs["nsq"]["trace"],
                      columns=["ts", "vpin", "as_z", "dir_flag", "exposed", "as_flag", "inv"])
    t_axis = [_dt(x) for x in tr.ts]

    # chart 4 — toxicity trace
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
    axes[0].plot([_dt(x) for x in mids.timestamp_ms], mids["mid"], lw=0.9, color="black")
    axes[0].set_ylabel("mid")
    axes[0].set_title(f"{u} {tok[:14]}… — spike episode")
    axes[1].plot(t_axis, tr.vpin, lw=0.9, color="#4878d0", label="VPIN (Lens 1)")
    d_on = tr[tr.dir_flag]
    axes[1].scatter([_dt(x) for x in d_on.ts], d_on.vpin, s=8, color="#d65f5f",
                    label="directional_flag", zorder=3)
    axes[1].set_ylabel("VPIN")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[2].plot(t_axis, tr.as_z, lw=0.9, color="#6acc65", label="AS z (Lens 2)")
    axes[2].axhline(-2, color="red", lw=0.8, ls="--", label="z = −2")
    a_on = tr[tr.as_flag]
    axes[2].scatter([_dt(x) for x in a_on.ts], a_on.as_z, s=8, color="#d65f5f",
                    label="as_flag", zorder=3)
    axes[2].set_ylabel("AS z-score")
    axes[2].legend(fontsize=8, loc="lower left")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.suptitle("Toxicity trace through the spike: Lens 1 (volume-clock VPIN + sweep weighting) "
                 "and Lens 2 (post-fill drift z vs calm baseline), with firing points")
    fig.tight_layout()
    fig.savefig(PLOT_OUT / "mm_task5_1_toxicity_trace.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # chart 5 — inventory paths
    fig, ax1 = plt.subplots(figsize=(12.5, 4.6))
    for name, color in (("baseline", "#d65f5f"), ("nsq", "#4878d0")):
        fl = [f for f in runs[name]["fills"] if a <= f["ts_exchange"] <= b]
        if not fl:
            continue
        ax1.step([_dt(f["ts_exchange"]) for f in fl],
                 [f["position_after"] for f in fl], where="post", lw=1.2,
                 color=color, label=f"{name} inventory")
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_ylabel("net position (contracts)")
    ax1.legend(loc="upper left", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot([_dt(x) for x in mids.timestamp_ms], mids["mid"], lw=0.8, color="grey", alpha=0.7)
    ax2.set_ylabel("mid (grey)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.suptitle("Inventory path into the spike: symmetric baseline (one-sided stacking) vs "
                 "NeutralSpikeQuoter (suspend exposed side, carry balanced)")
    fig.tight_layout()
    fig.savefig(PLOT_OUT / "mm_task5_1_inventory_path.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", default="1,2,3,4,5")
    args = ap.parse_args()
    which = set(args.charts.split(","))
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads(SELECTION_JSON.read_text())
    groups = pd.read_parquet(CSV_OUT / "mm_task5_1_groups.parquet")
    if "1" in which:
        chart_split_diagnosis(groups)
        print("chart 1 done")
    if "2" in which:
        chart_surface()
        print("chart 2 done")
    if "3" in which:
        chart_inner_outer()
        print("chart 3 done")
    if "4" in which or "5" in which:
        chart_spike(sel)
        print("charts 4+5 done")
    print(f"plots -> {PLOT_OUT}")


if __name__ == "__main__":
    main()
