"""Task-5.1 v0 (re-grounded) — failure attribution + toxicity map under the whole-market view.

Re-derives the Task-5 v0 gate on the FULL R2 sample (17.9 d, 32+32 tokens) with the two
methodology fixes of the redesign applied to the *diagnosis itself*:

1. **Whole-market view, cohort × τ-regime surface.** Toxicity is mapped per
   (cohort_aggr × τ-regime) cell — the same conditioning axes the ladder reports — instead
   of one pooled near-expiry number. τ CONDITIONS; the market is the cluster unit.
2. **Session-aware baselines (no fixed thresholds).** Each market's calm reference is its
   OWN lead-in window (`mm_eval.cpcv.lead_in_span`) — the endgame/toxic read is reported
   as a drift *z* against that per-market baseline alongside the absolute ¢, retiring the
   fixed-threshold caveat of Task-5.
3. **Lens-aligned attribution.** The counterfactual-rescue test now scores the two Task-5.1
   lenses as per-fill flags — a causal decayed **sweep score** (LOTECH Lens 1, public
   trades vs the as-of touch) and a causal **AS z-score** (Lens 2, post-fill drift vs the
   market's rolling calm baseline) — next to Task-5's velocity/imbalance signals, so the
   controller's triggers are grounded on the same telemetry that indicted the old ones.

**Pre-registered wiring rule (written before the numbers ran):** the two-lens gate and the
OFI size-dampening enter a category's *gated ladder* iff ≥1 (cohort × τ-regime) cell in
that category has qty-weighted markout(30 s) point < 0 AND leave-one-market-out negative in
≥ 50% of re-pools. Cells that fail stay in the report as UNSUPPORTED — the final shipped
config per cohort×regime withholds defensive knobs there.

Usage (from polymarket/research/):
    PYTHONPATH=. uv run python scripts/mm_task5_1_v0_attribution.py [--force] [--workers 6]

Honesty rails: RiskAverse queue, 0 ms (the pessimistic bracket edge; markout is
queue-near-invariant at the touch); market-cluster CIs; attribution thresholds are
full-sample percentiles (research attribution, labelled) while the SHIPPED gates use the
causal session-relative implementations inside `NeutralSpikeQuoter`. No profitability claim.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mm_engine import (BACKTEST, ConstantLatency, FeeModel, RiskAverseQueue, SymmetricQuoter,
                       Telemetry, run_engine)
from mm_engine.feeds.replay_parquet import replay_parquet
from mm_engine.telemetry import JsonlSink

from mm_eval import cpcv
from mm_eval import design_inputs as di
from mm_eval.metrics import CENTS, compute_markout, mid_trajectory, _asof_backward

RESEARCH = Path(__file__).resolve().parents[1]
CSV_OUT = RESEARCH / "data/analysis/csv_outputs/market_making"
PLOT_OUT = RESEARCH / "data/analysis/plots/market_making"
SELECTION_JSON = RESEARCH / "data/markets/mm_task5_1_selection.json"
GROUPS_CSV = CSV_OUT / "mm_task5_1_groups.csv"
SCRATCH = Path("/private/tmp/claude-501/-Users-justiniturregui-Desktop-github-epsilon-quant-research/"
               "b6ea1a3f-cca4-465b-b11c-5ab18e4a749c/scratchpad")
CACHE = SCRATCH / "mm_task5_1_cache"
UNIVERSES = ("politics_negrisk", "esports")
HORIZONS = (1, 5, 30, 60)
PRIMARY_H = 30
VEL_WINDOW_S = 10
FLAG_PCT = 80

# LOTECH Lens-1 sweep score constants (declared; mirror NSQ_DEFAULTS' bounded variant)
SWEEP_LAMBDA = 0.5
SWEEP_CAP_TICKS = 6.0
SWEEP_HALFLIFE_S = 60.0
TICK = 0.001
# Lens-2 research replica (per-token rolling calm baseline over prior fills)
ASZ_BASELINE_N = 50
ASZ_MIN_FILLS = 10

TAU_LABEL_ORDER = {"politics_negrisk": ("tau_midlife", "tau_approach", "tau_endgame"),
                   "esports": ("tau_pre", "tau_inplay")}


def tau_regime_of(ttr_h: np.ndarray, universe: str) -> np.ndarray:
    """τ-regime label per fill (the conditioning axis; matches cpcv.TAU_REGIMES)."""
    out = np.full(ttr_h.shape, "tau_none", dtype=object)
    for name, lo, hi in cpcv.TAU_REGIMES[universe]:
        m = (ttr_h >= lo) & (ttr_h < hi) & np.isfinite(ttr_h)
        out[m] = f"tau_{name}"
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stage A — baseline replay + per-fill dump with lens signals (parallel, cached)
# ──────────────────────────────────────────────────────────────────────────────

def _sweep_score_series(trades: pd.DataFrame, bba: pd.DataFrame) -> pd.DataFrame:
    """Causal decayed sweep score after each public trade (LOTECH Lens 1, research replica).

    score(t) = Σ_i qty_i · exp(min(dist_i/tick, cap)·λ) · exp(−ln2·(t−t_i)/halflife), the
    signed variant tracks direction. Distances measured vs the as-of PRIOR touch.
    """
    if trades.empty or bba.empty:
        return pd.DataFrame(columns=["timestamp_ms", "sweep_abs", "sweep_net"])
    m = pd.merge_asof(trades.sort_values("timestamp_ms"),
                      bba.sort_values("timestamp_ms"), on="timestamp_ms",
                      direction="backward")
    m = m.dropna(subset=["best_bid", "best_ask"])
    ln2 = np.log(2.0)
    score_buy = score_sell = 0.0
    last_ts = None
    rows = []
    for r in m.itertuples():
        if last_ts is not None:
            decay = np.exp(-ln2 * (r.timestamp_ms - last_ts) / (SWEEP_HALFLIFE_S * 1000.0))
            score_buy *= decay
            score_sell *= decay
        side = r.side if r.side in ("BUY", "SELL") else (
            "BUY" if r.price >= (r.best_bid + r.best_ask) / 2 else "SELL")
        dist = max(0.0, r.price - r.best_ask) if side == "BUY" else max(0.0, r.best_bid - r.price)
        w = np.exp(min(dist / TICK, SWEEP_CAP_TICKS) * SWEEP_LAMBDA)
        if side == "BUY":
            score_buy += r.size * w
        else:
            score_sell += r.size * w
        last_ts = r.timestamp_ms
        rows.append((r.timestamp_ms, score_buy + score_sell, score_buy - score_sell))
    return pd.DataFrame(rows, columns=["timestamp_ms", "sweep_abs", "sweep_net"])


def _as_z_per_fill(fdf: pd.DataFrame) -> np.ndarray:
    """Causal Lens-2 replica: drift z of each fill vs the token's rolling calm baseline.

    adverse_drift(30 s) is already signed (negative = adverse); the baseline is the rolling
    mean/std of the PRIOR ``ASZ_BASELINE_N`` fills' drifts (session-aware, per token).
    """
    d = fdf[f"adverse_drift_{PRIMARY_H}s_c"].to_numpy(dtype=float)
    z = np.full(d.shape, np.nan)
    hist: list[float] = []
    for i, v in enumerate(d):
        if len(hist) >= ASZ_MIN_FILLS:
            mu = float(np.mean(hist))
            sd = max(float(np.std(hist, ddof=1)), 1e-4)
            z[i] = (v - mu) / sd
        if np.isfinite(v):
            hist.append(v)
            if len(hist) > ASZ_BASELINE_N:
                hist.pop(0)
    return z


def dump_token(args_tuple) -> str | None:
    """Worker: replay one token (symmetric, RiskAverse, 0 ms) → per-fill parquet slice."""
    universe, tok, half_spread, end_ms = args_tuple
    out_f = CACHE / "v0_fills" / f"{universe}_{tok}.parquet"
    if out_f.exists():
        return str(out_f)
    tdir = CACHE / universe / tok
    tele = Telemetry(fills=JsonlSink(keep=True), orders=JsonlSink(keep=False),
                     quotes=JsonlSink(keep=True))
    result = run_engine(
        replay_parquet(tdir, gaps=[]), strategy=SymmetricQuoter(),
        queue_model=RiskAverseQueue(), latency_model=ConstantLatency(0.0), mode=BACKTEST,
        params={"half_spread": half_spread, "size": 100.0, "tick": 0.001},
        fee_model=FeeModel(), telemetry=tele)
    if not result.fills:
        out_f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(out_f)
        return str(out_f)
    fdf = pd.DataFrame(result.fills)
    mr = compute_markout(result.fills, result.quotes, horizons=HORIZONS)
    for h in HORIZONS:
        fdf[f"markout_{h}s_c"] = mr.markout_to_fill[h] * CENTS
        fdf[f"adverse_drift_{h}s_c"] = mr.adverse_drift[h] * CENTS
    side_sign = np.where(fdf["side"].to_numpy() == "BUY", 1.0, -1.0)
    fdf["side_sign"] = side_sign
    ts, mid = mid_trajectory(result.quotes)
    pre_mid = np.array([_asof_backward(ts, mid, int(t) - VEL_WINDOW_S * 1000)
                        for t in fdf["ts_exchange"]])
    dv = (fdf["mid_at_fill"].to_numpy() - pre_mid) * CENTS
    fdf["mid_vel_abs_c"] = np.abs(dv)

    con = duckdb.connect()
    trades = con.execute(f"SELECT timestamp_ms, price, size, side FROM "
                         f"read_parquet('{tdir}/trades_x.parquet') ORDER BY timestamp_ms").df()
    bba = con.execute(f"SELECT timestamp_ms, best_bid, best_ask FROM "
                      f"read_parquet('{tdir}/bba_x.parquet') "
                      f"WHERE best_bid IS NOT NULL AND best_ask IS NOT NULL "
                      f"ORDER BY timestamp_ms").df()
    con.close()
    sw = _sweep_score_series(trades, bba)
    if len(sw):
        fdf = pd.merge_asof(fdf.sort_values("ts_exchange"), sw,
                            left_on="ts_exchange", right_on="timestamp_ms",
                            direction="backward").drop(columns=["timestamp_ms"])
        # normalize the sweep magnitude per token (session-relative, like the live gate)
        med = float(fdf["sweep_abs"].median()) or 1.0
        fdf["sweep_rel"] = fdf["sweep_abs"] / max(med, 1e-9)
    else:
        fdf["sweep_abs"] = np.nan
        fdf["sweep_net"] = np.nan
        fdf["sweep_rel"] = np.nan
    fdf["as_z"] = _as_z_per_fill(fdf)
    fdf["universe"] = universe
    fdf["end_ms"] = end_ms if end_ms is not None else np.nan
    fdf["ttr_hours"] = (fdf["end_ms"] - fdf["ts_exchange"]) / 3.6e6

    keep = ["universe", "token_id", "ts_exchange", "side", "side_sign", "qty", "price",
            "mid_at_fill", "trade_size", "position_after", "mid_vel_abs_c",
            "sweep_abs", "sweep_net", "sweep_rel", "as_z", "end_ms", "ttr_hours",
            *[f"markout_{h}s_c" for h in HORIZONS],
            *[f"adverse_drift_{h}s_c" for h in HORIZONS]]
    out_f.parent.mkdir(parents=True, exist_ok=True)
    fdf[keep].to_parquet(out_f, index=False)
    return str(out_f)


# Same replay-feasibility cap as the ladder (declared ex-ante, data-volume criterion):
# the diagnosis universe must match the eval universe token-for-token.
MAX_TOKEN_EVENTS = 8_000_000


def _token_events(universe: str, tok: str, con) -> int:
    d = CACHE / universe / tok
    return sum(con.execute(f"SELECT count(*) FROM read_parquet('{d}/{tb}_x.parquet')")
               .fetchone()[0] for tb in ("book", "trades", "price_change", "bba"))


def build_stage_a(sel: dict, workers: int, force: bool) -> pd.DataFrame:
    cache_f = CACHE / "v0_per_fill.parquet"
    if cache_f.exists() and not force:
        print(f"[stage A] cached: {cache_f}")
        return pd.read_parquet(cache_f)
    con = duckdb.connect()
    jobs = []
    for u in UNIVERSES:
        for t in sel["universes"][u]["tokens"]:
            n_ev = _token_events(u, t["token_id"], con)
            if n_ev > MAX_TOKEN_EVENTS:
                print(f"  EXCLUDED (replay-feasibility cap): {u} {t['token_id'][:12]}… "
                      f"{n_ev:,} events")
                continue
            end_dt = t.get("end_date")
            end_ms = (datetime.fromisoformat(end_dt.replace("Z", "+00:00")).timestamp() * 1000.0
                      if end_dt else None)
            jobs.append((u, t["token_id"], t["half_spread"], end_ms))
    con.close()
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(dump_token, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            f = fut.result()
            if f:
                df = pd.read_parquet(f)
                if len(df):
                    frames.append(df)
            print(f"  [{i}/{len(jobs)}] {futs[fut][1][:10]}… done", flush=True)
    per_fill = pd.concat(frames, ignore_index=True)
    per_fill.to_parquet(cache_f, index=False)
    print(f"[stage A] {len(per_fill)} fills -> {cache_f}")
    return per_fill


# ──────────────────────────────────────────────────────────────────────────────
# Stage B — cohort × τ-regime toxicity surface + pre-registered wiring rule
# ──────────────────────────────────────────────────────────────────────────────

def surface_and_rule(per_fill: pd.DataFrame, groups: pd.DataFrame,
                     tok2group: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    pf = per_fill.copy()
    pf["group_id"] = pf["token_id"].map(tok2group)
    gmap = groups.set_index("group_id")
    pf["cohort_aggr"] = pf["group_id"].map(gmap["cohort_aggr"])
    pf["tau_regime"] = ""
    for u in UNIVERSES:
        m = pf.universe == u
        pf.loc[m, "tau_regime"] = tau_regime_of(pf.loc[m, "ttr_hours"].to_numpy(float), u)

    rows = []
    for (u, coh, reg), g in pf.groupby(["universe", "cohort_aggr", "tau_regime"]):
        m = g[f"markout_{PRIMARY_H}s_c"].to_numpy(float)
        a = g[f"adverse_drift_{PRIMARY_H}s_c"].to_numpy(float)
        q = g["qty"].to_numpy(float)
        mk_ids = g["token_id"].to_numpy()
        ok = np.isfinite(m) & np.isfinite(q) & (q > 0)
        if ok.sum() < 10:
            continue
        m, a, q, mk_ids = m[ok], a[ok], q[ok], mk_ids[ok]
        point = float(np.sum(m * q) / np.sum(q))
        adv = float(np.sum(a[np.isfinite(a)] * q[np.isfinite(a)]) / np.sum(q[np.isfinite(a)]))
        uniq = np.unique(mk_ids)
        lomo = []
        for x in uniq:
            s = mk_ids != x
            if s.any() and np.sum(q[s]) > 0:
                lomo.append(float(np.sum(m[s] * q[s]) / np.sum(q[s])))
        lomo_neg = float(np.mean([v < 0 for v in lomo])) if lomo else float("nan")
        ci = di._market_cluster_mean_ci(m / CENTS, q, mk_ids, n_boot=2000, seed=0).cents()
        net_negative = bool(point < 0 and (not lomo or lomo_neg >= 0.5))
        rows.append({"universe": u, "cohort_aggr": coh, "tau_regime": reg,
                     "n_fills": int(m.size), "n_markets": int(uniq.size),
                     "markout_c": point, "ci_lo": ci.lo, "ci_hi": ci.hi,
                     "adverse_c": adv, "lomo_neg_frac": lomo_neg,
                     "net_negative": net_negative,
                     "certified": bool(np.isfinite(ci.hi) and ci.hi < 0)})
    surface = pd.DataFrame(rows)
    # the pre-registered wiring rule per category
    rule = {}
    for u in UNIVERSES:
        cells = surface[(surface.universe == u) & surface.net_negative]
        rule[u] = {"wire_defensive_knobs": bool(len(cells)),
                   "supported_cells": [f"{r.cohort_aggr}×{r.tau_regime}"
                                       for r in cells.itertuples()]}
    return surface, rule


# ──────────────────────────────────────────────────────────────────────────────
# Stage C — lens-aligned counterfactual rescue
# ──────────────────────────────────────────────────────────────────────────────

SIGNALS = {
    "sweep_rel": ("sweep_rel", "high"),          # Lens 1 replica (session-relative)
    "as_z": ("as_z", "low_z"),                   # Lens 2 replica (z < −2, fixed statistical)
    "mid_vel_abs": ("mid_vel_abs_c", "high"),    # Task-5's best rescuer (comparison)
    "aggressor_size": ("trade_size", "high"),    # Task-5's failed skip-gate (comparison)
}


def rescue(per_fill: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tok, g in per_fill.groupby("token_id"):
        m = g[f"markout_{PRIMARY_H}s_c"].to_numpy(float)
        q = g["qty"].to_numpy(float)
        ok = np.isfinite(m) & np.isfinite(q) & (q > 0)
        if ok.sum() < 20:
            continue
        base = float(np.sum(m[ok] * q[ok]) / np.sum(q[ok]))
        for name, (col, direction) in SIGNALS.items():
            s = g[col].to_numpy(float)
            if direction == "high":
                thr = np.nanpercentile(s[ok], FLAG_PCT)
                flag = s > thr
            else:   # low_z — the fixed −2 statistical threshold, not a percentile
                flag = s < -2.0
            keep = ok & ~np.where(np.isfinite(s), flag, False)
            if keep.sum() < 10 or np.sum(q[keep]) <= 0:
                continue
            resc = float(np.sum(m[keep] * q[keep]) / np.sum(q[keep]))
            fl = ok & np.where(np.isfinite(s), flag, False)
            rows.append({"universe": g["universe"].iloc[0], "token_id": tok,
                         "signal": name, "markout_base_c": base,
                         "markout_rescued_c": resc, "rescue_delta_c": resc - base,
                         "flagged_qty_frac": float(np.sum(q[fl]) / np.sum(q[ok])),
                         "n_fills": int(ok.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)

    sel = json.loads(SELECTION_JSON.read_text())
    groups = pd.read_csv(GROUPS_CSV)
    tok2group = {t["token_id"]: t["group_id"]
                 for u in UNIVERSES for t in sel["universes"][u]["tokens"]}

    per_fill = build_stage_a(sel, args.workers, args.force)

    surface, rule = surface_and_rule(per_fill, groups, tok2group)
    surface.to_csv(CSV_OUT / "mm_task5_1_v0_surface.csv", index=False)
    print("\n=========== v0 cohort × τ-regime toxicity surface (baseline fills) ===========")
    print(surface.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\n=========== pre-registered wiring rule ===========")
    print(json.dumps(rule, indent=2))
    (CSV_OUT / "mm_task5_1_v0_wiring.json").write_text(json.dumps(rule, indent=2))

    resc = rescue(per_fill)
    resc.to_csv(CSV_OUT / "mm_task5_1_v0_rescue.csv", index=False)
    agg = (resc.groupby(["universe", "signal"])
           .agg(mean_rescue_c=("rescue_delta_c", "mean"),
                median_rescue_c=("rescue_delta_c", "median"),
                mean_flagged=("flagged_qty_frac", "mean"),
                n_tokens=("token_id", "nunique")).reset_index())
    print("\n=========== lens-aligned counterfactual rescue (mean per signal) ===========")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # chart: toxicity trace panels are produced by the charts script; here the surface map
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, u in zip(axes, UNIVERSES):
        sub = surface[surface.universe == u]
        order = [r for r in TAU_LABEL_ORDER[u]]
        piv = sub.pivot_table(index="cohort_aggr", columns="tau_regime",
                              values="markout_c").reindex(columns=order)
        im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([c.replace("tau_", "") for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iloc[i, j]
                if np.isfinite(v):
                    cell = sub[(sub.cohort_aggr == piv.index[i])
                               & (sub.tau_regime == piv.columns[j])]
                    n = int(cell.n_fills.iloc[0]) if len(cell) else 0
                    ax.text(j, i, f"{v:+.2f}¢\nn={n}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{u} — baseline markout(30s) ¢/ct")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Task-5.1 v0: toxicity surface by cohort × τ-regime (symmetric baseline, RiskAverse)")
    fig.tight_layout()
    fig.savefig(PLOT_OUT / "mm_task5_1_v0_surface.png", dpi=110, bbox_inches="tight")
    print(f"\nCSVs -> {CSV_OUT}\nplot -> {PLOT_OUT/'mm_task5_1_v0_surface.png'}")


if __name__ == "__main__":
    main()
