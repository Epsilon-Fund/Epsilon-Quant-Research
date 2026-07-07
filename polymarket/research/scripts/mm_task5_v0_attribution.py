"""Task-5 v0 gate — Task-4 failure attribution + the near-expiry regime, BOTH universes.

Data-only (no strategy code). This is the folded Step-0 that gates the Task-5 build:

1. **Failure attribution.** For each Task-4 token (12 politics + 12 esports, RiskAverse
   pessimistic queue, 0-ms — identical to the Task-4 setup) we re-dump the per-fill
   telemetry and ask *what observable, strategy-actionable signal flags the toxic fills*:
   pre-fill mid velocity (10 s), book imbalance at fill, aggressor trade size, depth
   evaporation, and time-to-resolution. The attribution is a **counterfactual rescue**: per
   token, recompute the qty-weighted markout(30 s) excluding fills flagged by each signal
   (at that token's 80th-percentile threshold) — which signal, used as a skip-gate, recovers
   the most edge, and would any DEAD token have been VIABLE-shaped without the flagged fills?

2. **Near-expiry regime, PRE-REGISTERED READ (written before the numbers were run).**
   TTR = Gamma ``end_date`` − fill ts. Per universe we bucket markout/adverse by TTR with
   market-cluster CIs. **Decision rule (pre-registered):** decisions 2 (τ-flatten) & 4
   (toxicity gate) of the Task-5 PRD are *wired* iff, in that universe, the qty-weighted net
   markout(30 s) point estimate in the near-expiry bucket (TTR < 6 h) is **negative** AND
   stays negative under leave-one-market-out in ≥ 50% of re-pools. CI status is reported
   alongside (certified iff the market-cluster CI upper bound < 0; else directional). If the
   rule fails for a universe, decisions 2 & 4 are reported as UNSUPPORTED there — not
   silently shipped.

Usage (from polymarket/research/):
    PYTHONPATH=. uv run python scripts/mm_task5_v0_attribution.py [--force] [--quick]

Honesty rails: pessimistic queue (bracket's honest lower bound; Task 4 showed markout is
near queue-invariant at the touch); market-cluster bootstrap CIs (fills are autocorrelated —
the market is the independent unit); attribution thresholds here are FULL-SAMPLE percentiles
(research attribution, labeled as such) — any *shipped* gate in v1 uses causal, rolling
thresholds instead. No profitability claim.
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
import matplotlib.pyplot as plt

from mm_engine import (BACKTEST, ConstantLatency, FeeModel, RiskAverseQueue, SymmetricQuoter,
                       Telemetry, run_engine)
from mm_engine.feeds.replay_parquet import replay_parquet
from mm_engine.telemetry import JsonlSink

from mm_eval import design_inputs as di
from mm_eval import markets as mk
from mm_eval.metrics import CENTS, compute_markout, mid_trajectory, _asof_backward

RESEARCH = Path(__file__).resolve().parents[1]
L2_ROOT = Path.home() / "epsilon_l2_full"
CSV_OUT = RESEARCH / "data/analysis/csv_outputs/market_making"
PLOT_OUT = RESEARCH / "data/analysis/plots/market_making"
META_JSON = RESEARCH / "data/markets/mm_task5_market_meta.json"
SCRATCH = Path("/private/tmp/claude-501/-Users-justiniturregui-Desktop-github-epsilon-quant-research/"
               "bdab8b15-7f54-4662-bb7c-a5684c173098/scratchpad")
CACHE = SCRATCH / "mm_task5_cache"
UNIVERSES = ("politics_negrisk", "esports")
HORIZONS = (1, 5, 30, 60)
PRIMARY_H = 30
VEL_WINDOW_S = 10          # pre-fill mid-velocity lookback (strategy-observable)
DEPTH_WINDOW_S = 300       # trailing depth median for the evaporation ratio
FLAG_PCT = 80              # attribution threshold: token's 80th percentile of the signal

# Near-expiry TTR bucket edges (hours). Politics uses the design-inputs buckets; esports
# matches live hours not months, so its buckets are finer near zero.
TTR_EDGES = {
    "politics_negrisk": (0.0, 6.0, 24.0, 72.0, 168.0, 720.0, np.inf),
    "esports": (0.0, 0.5, 2.0, 6.0, 24.0, 72.0, np.inf),
}
TTR_LABELS = {
    "politics_negrisk": ("<6h", "6-24h", "1-3d", "3-7d", "7-30d", ">30d"),
    "esports": ("<30m", "30m-2h", "2-6h", "6-24h", "1-3d", ">3d"),
}
NEAR_EXPIRY_BUCKETS = {"politics_negrisk": ("<6h",), "esports": ("<30m", "30m-2h", "2-6h")}


def load_meta() -> dict:
    return json.loads(META_JSON.read_text())


def end_ms(rec: dict) -> float:
    ed = rec.get("endDate")
    if not ed:
        return float("nan")
    return datetime.fromisoformat(ed.replace("Z", "+00:00")).timestamp() * 1000.0


# ──────────────────────────────────────────────────────────────────────────────
# Stage A — replay + per-fill dump (RiskAverse, 0-ms; identical setup to Task 4)
# ──────────────────────────────────────────────────────────────────────────────

def replay_token_fills(token_dir: Path, spec: mk.MarketSpec) -> pd.DataFrame:
    tele = Telemetry(fills=JsonlSink(keep=True), orders=JsonlSink(keep=False),
                     quotes=JsonlSink(keep=True))
    params = {"half_spread": spec.half_spread, "size": 100.0, "tick": 0.001}
    result = run_engine(
        replay_parquet(token_dir, gaps=[]),
        strategy=SymmetricQuoter(), queue_model=RiskAverseQueue(),
        latency_model=ConstantLatency(0.0), mode=BACKTEST, params=params,
        fee_model=FeeModel(), telemetry=tele,
    )
    if not result.fills:
        return pd.DataFrame()
    fdf = pd.DataFrame(result.fills)
    mr = compute_markout(result.fills, result.quotes, horizons=HORIZONS)
    for h in HORIZONS:
        fdf[f"markout_{h}s_c"] = mr.markout_to_fill[h] * CENTS
        fdf[f"adverse_drift_{h}s_c"] = mr.adverse_drift[h] * CENTS
    side_sign = np.where(fdf["side"].to_numpy() == "BUY", 1.0, -1.0)
    fdf["side_sign"] = side_sign

    # pre-fill mid velocity over VEL_WINDOW_S (strategy-observable: needs only the book
    # stream a live quoter maintains). signed_vel > 0 = the mid was already moving TOWARD
    # the side that then filled us (we were run over — continuation toxicity).
    ts, mid = mid_trajectory(result.quotes)
    pre_mid = np.array([_asof_backward(ts, mid, int(t) - VEL_WINDOW_S * 1000)
                        for t in fdf["ts_exchange"]])
    dv = (fdf["mid_at_fill"].to_numpy() - pre_mid) * CENTS
    fdf["mid_vel_abs_c"] = np.abs(dv)
    fdf["mid_vel_signed_c"] = -side_sign * dv   # >0: mid falling into our bid / rising into our ask
    fdf["universe"] = spec.universe
    fdf["market"] = spec.market
    keep = ["universe", "token_id", "market", "ts_exchange", "side", "side_sign", "qty",
            "price", "mid_at_fill", "trade_size", "position_after",
            "mid_vel_abs_c", "mid_vel_signed_c",
            *[f"markout_{h}s_c" for h in HORIZONS], *[f"adverse_drift_{h}s_c" for h in HORIZONS]]
    return fdf[keep]


def attach_book_signals(fdf: pd.DataFrame, token_dir: Path,
                        con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """As-of attach imbalance/depth at fill + the trailing-median depth-evaporation ratio."""
    bookf = token_dir / "book_x.parquet"
    book_df = con.execute(
        f"SELECT timestamp_ms, bids, asks FROM read_parquet('{bookf}') ORDER BY timestamp_ms"
    ).df() if bookf.exists() else pd.DataFrame(columns=["timestamp_ms", "bids", "asks"])
    top = di.book_top_series(book_df)
    fdf = di.asof_attach_book(fdf, top)
    if top.empty or fdf.empty:
        fdf["depth_evap"] = np.nan
        return fdf
    # trailing median depth over DEPTH_WINDOW_S before each fill → evaporation ratio
    tts = top["ts"].to_numpy(dtype=np.int64)
    td = top["tob_depth"].to_numpy(dtype=float)
    ratios = np.full(len(fdf), np.nan)
    fts = fdf["ts_exchange"].to_numpy(dtype=np.int64)
    lo = np.searchsorted(tts, fts - DEPTH_WINDOW_S * 1000, side="left")
    hi = np.searchsorted(tts, fts, side="right")
    for i in range(len(fdf)):
        window = td[lo[i]:hi[i]]
        if window.size >= 3:
            med = float(np.median(window))
            cur = fdf["depth_at_fill"].iloc[i]
            if med > 0 and np.isfinite(cur):
                ratios[i] = cur / med
    fdf["depth_evap"] = ratios    # <1 = depth below trailing median (evaporating)
    return fdf


def build_stage_a(force: bool, quick: bool) -> pd.DataFrame:
    cache_f = CACHE / "task5_per_fill.parquet"
    if cache_f.exists() and not force:
        print(f"[stage A] cached per-fill dump: {cache_f}", flush=True)
        return pd.read_parquet(cache_f)
    CACHE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    top_k = 4 if quick else 12
    frames = []
    for universe in UNIVERSES:
        print(f"\n[stage A] === {universe} ===", flush=True)
        specs = mk.select_markets(L2_ROOT, universe, top_k=top_k, con=con)
        mk.build_compact(L2_ROOT, universe, specs, CACHE, con=con)
        for i, spec in enumerate(specs, 1):
            tdir = mk.materialize_token(spec, CACHE, con=con)
            fdf = replay_token_fills(tdir, spec)
            if not fdf.empty:
                fdf = attach_book_signals(fdf, tdir, con)
                fdf["half_spread_c"] = spec.half_spread * CENTS
                frames.append(fdf)
            print(f"  [{i}/{len(specs)}] {spec.token_id[:10]}… fills={len(fdf)}", flush=True)
    con.close()
    per_fill = pd.concat(frames, ignore_index=True)
    per_fill.to_parquet(cache_f, index=False)
    print(f"[stage A] cached {len(per_fill)} fills -> {cache_f}", flush=True)
    return per_fill


# ──────────────────────────────────────────────────────────────────────────────
# Stage B1 — near-expiry regime + THE PRE-REGISTERED CHECK
# ──────────────────────────────────────────────────────────────────────────────

def add_ttr(per_fill: pd.DataFrame, meta: dict) -> pd.DataFrame:
    end_map = {cid: end_ms(rec) for cid, rec in meta.items()}
    lab_map = {cid: str(rec.get("question", cid))[:44] for cid, rec in meta.items()}
    out = per_fill.copy()
    out["end_ms"] = out["market"].map(end_map)
    out["ttr_hours"] = (out["end_ms"] - out["ts_exchange"]) / 3.6e6
    out["mkt_label"] = out["market"].map(lab_map)
    return out


def ttr_regime(per_fill: pd.DataFrame, universe: str) -> pd.DataFrame:
    sub = per_fill[per_fill.universe == universe]
    idx = di.ttr_bucketize(sub["ttr_hours"].to_numpy(), TTR_EDGES[universe])
    be = di.bucketed_edge(sub[f"markout_{PRIMARY_H}s_c"].to_numpy(),
                          sub[f"adverse_drift_{PRIMARY_H}s_c"].to_numpy(),
                          sub["qty"].to_numpy(), idx, TTR_LABELS[universe],
                          market_ids=sub["market"].to_numpy(), n_boot=2000, seed=0)
    return pd.DataFrame([{
        "universe": universe, "bucket": b.label, "n_fills": b.n_fills, "n_markets": b.n_markets,
        "markout_30s_c": b.markout_cents.point, "markout_lo": b.markout_cents.lo,
        "markout_hi": b.markout_cents.hi, "adverse_30s_c": b.adverse_cents.point,
        "adverse_lo": b.adverse_cents.lo, "adverse_hi": b.adverse_cents.hi,
        "adverse_rate": b.adverse_rate,
    } for b in be])


def preregistered_check(per_fill: pd.DataFrame, universe: str) -> dict:
    """The v0 gate: near-expiry (TTR < 6 h) markout point < 0 AND LOMO-negative ≥ 50%."""
    sub = per_fill[(per_fill.universe == universe) & (per_fill["ttr_hours"] < 6.0)
                   & np.isfinite(per_fill["ttr_hours"])]
    m = sub[f"markout_{PRIMARY_H}s_c"].to_numpy()
    q = sub["qty"].to_numpy()
    mkts = sub["market"].to_numpy()
    ok = np.isfinite(m) & np.isfinite(q) & (q > 0)
    m, q, mkts = m[ok], q[ok], mkts[ok]
    if m.size == 0:
        return {"universe": universe, "n_fills": 0, "wire_decisions_2_4": False,
                "reason": "no near-expiry fills observed"}
    point = float(np.sum(m * q) / np.sum(q))
    uniq = np.unique(mkts)
    lomo = []
    for u in uniq:
        sel = mkts != u
        if sel.any() and np.sum(q[sel]) > 0:
            lomo.append(float(np.sum(m[sel] * q[sel]) / np.sum(q[sel])))
    lomo_neg_frac = float(np.mean([x < 0 for x in lomo])) if lomo else float("nan")
    ci = di._market_cluster_mean_ci(m / CENTS, q, mkts, n_boot=2000, seed=0).cents()
    wire = bool(point < 0 and (not lomo or lomo_neg_frac >= 0.5))
    certified = bool(np.isfinite(ci.hi) and ci.hi < 0)
    return {"universe": universe, "n_fills": int(m.size), "n_markets": int(uniq.size),
            "markout_point_c": point, "ci_lo": ci.lo, "ci_hi": ci.hi,
            "lomo_negative_frac": lomo_neg_frac, "wire_decisions_2_4": wire,
            "ci_status": "certified-negative" if certified else "directional",
            "reason": f"point {point:+.2f}c, LOMO-neg {lomo_neg_frac:.0%}, CI [{ci.lo:+.2f}, {ci.hi:+.2f}]c"}


# ──────────────────────────────────────────────────────────────────────────────
# Stage B2 — failure attribution: counterfactual rescue per signal
# ──────────────────────────────────────────────────────────────────────────────

SIGNALS = {
    "mid_vel_abs": ("mid_vel_abs_c", "high"),          # fast tape
    "mid_vel_signed": ("mid_vel_signed_c", "high"),    # being run over (continuation)
    "aggressor_size": ("trade_size", "high"),          # large taker
    "abs_imbalance": ("abs_imbalance_at_fill", "high"),# lopsided book
    "depth_evap": ("depth_evap", "low"),               # depth below trailing median
    "near_expiry": ("ttr_hours", "low_6h"),            # TTR < 6 h (fixed, not percentile)
}


def counterfactual_rescue(per_fill: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    """Per token × signal: markout(30s) with flagged fills removed vs baseline."""
    pf = per_fill.copy()
    pf["abs_imbalance_at_fill"] = pf["imbalance_at_fill"].abs()
    rows = []
    for tok, g in pf.groupby("token_id"):
        m = g[f"markout_{PRIMARY_H}s_c"].to_numpy()
        q = g["qty"].to_numpy()
        ok = np.isfinite(m) & np.isfinite(q) & (q > 0)
        if ok.sum() < 20:
            continue
        base = float(np.sum(m[ok] * q[ok]) / np.sum(q[ok]))
        vrow = verdicts[verdicts.token_id == tok]
        verdict = vrow["verdict_no_rebate"].iloc[0] if len(vrow) else "?"
        half_spread = float(g["half_spread_c"].iloc[0])
        for name, (col, direction) in SIGNALS.items():
            s = g[col].to_numpy()
            if direction == "high":
                thr = np.nanpercentile(s[ok], FLAG_PCT)
                flag = s > thr
            elif direction == "low":
                thr = np.nanpercentile(s[ok], 100 - FLAG_PCT)
                flag = s < thr
            else:  # low_6h — fixed near-expiry rule
                flag = s < 6.0
            keep = ok & ~np.where(np.isfinite(s), flag, False)
            if keep.sum() < 10 or np.sum(q[keep]) <= 0:
                continue
            resc = float(np.sum(m[keep] * q[keep]) / np.sum(q[keep]))
            flagged_frac = float(np.sum(q[ok & np.where(np.isfinite(s), flag, False)])
                                 / np.sum(q[ok]))
            rows.append({"universe": g["universe"].iloc[0], "token_id": tok,
                         "verdict": verdict, "half_spread_c": half_spread, "signal": name,
                         "markout_base_c": base, "markout_rescued_c": resc,
                         "rescue_delta_c": resc - base, "flagged_qty_frac": flagged_frac,
                         "n_fills": int(ok.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)

    meta = load_meta()
    per_fill = build_stage_a(args.force, args.quick)
    per_fill = add_ttr(per_fill, meta)
    verdicts = pd.read_csv(CSV_OUT / "mm_validation_verdict.csv", dtype={"token_id": str})

    # B1 — TTR regime + pre-registered check
    regimes = pd.concat([ttr_regime(per_fill, u) for u in UNIVERSES], ignore_index=True)
    regimes.to_csv(CSV_OUT / "mm_task5_v0_ttr_regime.csv", index=False)
    checks = [preregistered_check(per_fill, u) for u in UNIVERSES]
    pd.DataFrame(checks).to_csv(CSV_OUT / "mm_task5_v0_preregistered_check.csv", index=False)

    # B2 — counterfactual rescue attribution
    rescue = counterfactual_rescue(per_fill, verdicts)
    rescue.to_csv(CSV_OUT / "mm_task5_v0_rescue.csv", index=False)

    # summary tables
    print("\n================ v0 — TTR REGIME (markout & adverse by bucket) ================")
    print(regimes.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n================ v0 — PRE-REGISTERED NEAR-EXPIRY CHECK ================")
    for c in checks:
        print(json.dumps(c, indent=2, default=str))
    print("\n================ v0 — COUNTERFACTUAL RESCUE (mean per signal × verdict) ==========")
    agg = (rescue.groupby(["universe", "verdict", "signal"])
           .agg(mean_rescue_c=("rescue_delta_c", "mean"),
                median_rescue_c=("rescue_delta_c", "median"),
                mean_flagged_frac=("flagged_qty_frac", "mean"),
                n_tokens=("token_id", "nunique")).reset_index())
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    dead = rescue[rescue.verdict == "DEAD"]
    if len(dead):
        print("\n-- DEAD tokens: does any single skip-gate turn the markout positive? --")
        piv = dead.pivot_table(index=["universe", "token_id"], columns="signal",
                               values="markout_rescued_c")
        piv["base"] = dead.groupby(["universe", "token_id"])["markout_base_c"].first()
        print(piv.to_string(float_format=lambda x: f"{x:+.3f}"))

    # plot: TTR regime per universe
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, u in zip(axes, UNIVERSES):
        df = regimes[(regimes.universe == u) & (regimes.n_fills > 0)]
        x = np.arange(len(df))
        ax.errorbar(x - 0.07, df["markout_30s_c"],
                    yerr=[df["markout_30s_c"] - df["markout_lo"],
                          df["markout_hi"] - df["markout_30s_c"]],
                    fmt="o-", color="#4878d0", capsize=3, label="net markout(30s) ¢")
        ax.errorbar(x + 0.07, df["adverse_30s_c"],
                    yerr=[df["adverse_30s_c"] - df["adverse_lo"],
                          df["adverse_hi"] - df["adverse_30s_c"]],
                    fmt="s--", color="#d65f5f", capsize=3, label="adverse selection ¢")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r.bucket}\nn={r.n_fills}\n({r.n_markets} mkt)"
                            for r in df.itertuples()], fontsize=8)
        ax.set_title(u); ax.set_ylabel("¢/contract")
        ax.legend(fontsize=8)
    fig.suptitle("Task-5 v0: markout & adverse selection vs time-to-resolution (RiskAverse, market-cluster CIs)")
    fig.tight_layout()
    fig.savefig(PLOT_OUT / "mm_task5_v0_ttr_regime.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "n_fills": int(len(per_fill)), "preregistered": checks}
    (SCRATCH / "mm_task5_v0_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nCSVs -> {CSV_OUT}\nPlot -> {PLOT_OUT/'mm_task5_v0_ttr_regime.png'}")


if __name__ == "__main__":
    main()
