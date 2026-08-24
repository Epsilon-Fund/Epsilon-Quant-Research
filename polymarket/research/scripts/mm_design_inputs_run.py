"""Task-5 design-input analysis — the *market screen* + the *time-to-resolution regime*.

Runs on the SAME ~11-day VPS Parquet (politics_negrisk + esports, 2026-06-19→06-30) and the SAME
top-12 quotable tokens as Task 4 ([[mm_symmetric_quoter_validation_findings]]), and consumes the
Task-4 per-token VIABLE/DEAD verdict. It drives ``mm_engine`` (SymmetricQuoter, RiskAverse
pessimistic queue, 0-ms latency) exactly as the Task-4 runner does — **no engine change** — dumps
the per-fill telemetry, and hands it to :mod:`mm_eval.design_inputs` for two analyses:

1. **Failure attribution → the market screen.** Per-token observable features (half-spread,
   top-of-book depth, book imbalance, price extremity, trade intensity, mid volatility) compared
   VIABLE-vs-DEAD (rank-AUC + median-diff CI, bootstrapped over tokens), AND a per-fill markout
   regression on those features with a **token-clustered bootstrap**.
2. **NegRisk time-to-resolution regime.** Per politics fill: time-to-resolution from the Gamma
   ``end_date``; markout + adverse selection bucketed by time-to-expiry; a coverage read of WHICH
   regimes the capture spans per market; and a within-short-dated-market log(TTR) gradient test.

Usage (from polymarket/research/):
    PYTHONPATH=. uv run python scripts/mm_design_inputs_run.py [--force] [--quick]

Honesty rails (brain/CODEX.md): bracketed by construction (pessimistic queue is the honest lower
bound; markout is near-queue-invariant at the touch per Task 4); CIs (block + cluster bootstrap),
never point estimates; NO profitability claim — this maps preconditions and regimes for Task 5.
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

from mm_eval import markets as mk
from mm_eval import design_inputs as di
from mm_eval.metrics import CENTS, compute_markout

RESEARCH = Path(__file__).resolve().parents[1]
L2_ROOT = Path.home() / "epsilon_l2_full"
CSV_OUT = RESEARCH / "data/analysis/csv_outputs/market_making"
PLOT_OUT = RESEARCH / "data/analysis/plots/market_making"
SCRATCH = Path("/private/tmp/claude-501/-Users-justiniturregui-Desktop-github-epsilon-quant-research/"
               "ddf227a8-bccc-4883-8070-91632609f2b0/scratchpad")
CACHE = SCRATCH / "mm_design_cache"
UNIVERSES = ("politics_negrisk", "esports")
HORIZONS = (1, 5, 30, 60)
PRIMARY_H = 30

# Observable per-token features that a live screen could act on BEFORE quoting.
SCREEN_FEATURES = ["half_spread_c", "log_tob_depth", "median_abs_imbalance", "imbalance_vol",
                   "price_extremity", "trade_intensity", "mid_vol_c"]
# Per-fill regressors: token-level (broadcast, cross-market identified) + fill-level (within-market).
# NOTE: queue_ahead is omitted — it is identically 0 under the RiskAverse pessimistic queue at 0-ms
# latency (a dead regressor), so it carries no information here.
FILL_LEVEL = ["log_trade_size", "fill_price_extremity",
              "log_abs_inv_before", "is_buy", "imbalance_at_fill", "log_depth_at_fill"]


# ──────────────────────────────────────────────────────────────────────────────
# Metadata (Gamma end_date, cached; offline-reproducible)
# ──────────────────────────────────────────────────────────────────────────────

def load_market_meta() -> dict:
    """condition_id -> {endDate, closed, label, ...} from the cached Gamma pulls."""
    meta = {}
    for fn in ("politics_market_meta.json", "esports_market_meta.json"):
        p = SCRATCH / fn
        if p.exists():
            meta.update(json.loads(p.read_text()))
    return meta


def end_ms(meta_rec: dict) -> float:
    ed = meta_rec.get("endDate")
    if not ed:
        return float("nan")
    return datetime.fromisoformat(ed.replace("Z", "+00:00")).timestamp() * 1000.0


# ──────────────────────────────────────────────────────────────────────────────
# Stage A — materialize, replay, dump per-fill records + per-token features
# ──────────────────────────────────────────────────────────────────────────────

def token_observable_features(token_dir: Path, spec: mk.MarketSpec,
                              con: duckdb.DuckDBPyConnection) -> dict:
    """Book/trade/mid observable features for one token from its materialized replay dir."""
    bookf = token_dir / "book_x.parquet"
    bbaf = token_dir / "bba_x.parquet"
    tradesf = token_dir / "trades_x.parquet"
    book_df = con.execute(
        f"SELECT timestamp_ms, bids, asks FROM read_parquet('{bookf}') ORDER BY timestamp_ms"
    ).df() if bookf.exists() else pd.DataFrame(columns=["timestamp_ms", "bids", "asks"])
    top = di.book_top_series(book_df)
    feats = di.token_book_features(top)
    bba_df = con.execute(
        f"SELECT best_bid, best_ask FROM read_parquet('{bbaf}') ORDER BY timestamp_ms"
    ).df() if bbaf.exists() else pd.DataFrame(columns=["best_bid", "best_ask"])
    feats["mid_vol_c"] = di.mid_volatility(bba_df)
    lo, hi = con.execute(
        f"SELECT min(timestamp_ms), max(timestamp_ms) FROM read_parquet('{tradesf}')"
    ).fetchone()
    active_h = (hi - lo) / 1000 / 3600 if lo is not None and hi is not None else float("nan")
    feats["active_hours"] = active_h
    feats["trade_intensity"] = spec.n_trades / active_h if active_h and active_h > 0 else float("nan")
    feats["half_spread_c"] = spec.half_spread * CENTS
    feats["avg_price"] = spec.avg_price
    feats["price_extremity"] = abs(spec.avg_price - 0.5)
    feats["n_trades"] = spec.n_trades
    feats["_book_top"] = top          # kept for the as-of attach (dropped before serialization)
    return feats


def replay_token_fills(token_dir: Path, spec: mk.MarketSpec) -> pd.DataFrame:
    """Run the engine (RiskAverse pessimistic, 0-ms) over one token; return a per-fill DataFrame
    with markout/adverse at each horizon (cents) + microstructure fill fields."""
    tele = Telemetry(fills=JsonlSink(keep=True), orders=JsonlSink(keep=False),
                     quotes=JsonlSink(keep=True))
    params = {"half_spread": spec.half_spread, "size": 100.0, "tick": 0.001}
    result = run_engine(
        replay_parquet(token_dir, gaps=[]),
        strategy=SymmetricQuoter(), queue_model=RiskAverseQueue(),
        latency_model=ConstantLatency(0.0), mode=BACKTEST, params=params,
        fee_model=FeeModel(), telemetry=tele,  # fee-free = captured truth; fills are fee-independent
    )
    if not result.fills:
        return pd.DataFrame()
    fdf = pd.DataFrame(result.fills)
    mr = compute_markout(result.fills, result.quotes, horizons=HORIZONS)
    # arrays align with result.fills order (compute_markout enumerates fills)
    for h in HORIZONS:
        fdf[f"markout_{h}s_c"] = mr.markout_to_fill[h] * CENTS
        fdf[f"adverse_drift_{h}s_c"] = mr.adverse_drift[h] * CENTS
    side_sign = np.where(fdf["side"].to_numpy() == "BUY", 1.0, -1.0)
    fdf["side_sign"] = side_sign
    fdf["is_buy"] = (side_sign > 0).astype(float)
    fdf["realized_half_spread_c"] = side_sign * (fdf["mid_at_fill"] - fdf["price"]) * CENTS
    dq = fdf["qty"].to_numpy() * side_sign
    fdf["abs_inv_before"] = np.abs(fdf["position_after"].to_numpy() - dq)
    fdf["universe"] = spec.universe
    fdf["market"] = spec.market
    keep = ["universe", "token_id", "market", "ts_exchange", "side", "side_sign", "is_buy",
            "qty", "price", "mid_at_fill", "queue_ahead", "trade_size", "position_after",
            "abs_inv_before", "realized_delta", "realized_half_spread_c",
            *[f"markout_{h}s_c" for h in HORIZONS], *[f"adverse_drift_{h}s_c" for h in HORIZONS]]
    return fdf[keep]


def build_stage_a(force: bool, quick: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize + replay all tokens; return (per_fill_df, per_token_features_df). Cached."""
    fills_cache = CACHE / "per_fill_records.parquet"
    feat_cache = CACHE / "per_token_features.parquet"
    if fills_cache.exists() and feat_cache.exists() and not force:
        print(f"[stage A] loading cached per-fill + features from {CACHE}", flush=True)
        return pd.read_parquet(fills_cache), pd.read_parquet(feat_cache)

    CACHE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    top_k = 4 if quick else 12
    all_fills, all_feats = [], []
    book_tops = {}  # token_id -> book_top_series (for as-of attach)
    for universe in UNIVERSES:
        print(f"\n[stage A] === {universe} ===", flush=True)
        specs = mk.select_markets(L2_ROOT, universe, top_k=top_k, con=con)
        print(f"  {len(specs)} quotable tokens; building compact (one full scan/table)…", flush=True)
        mk.build_compact(L2_ROOT, universe, specs, CACHE, con=con)
        for i, spec in enumerate(specs, 1):
            tdir = mk.materialize_token(spec, CACHE, con=con)
            feats = token_observable_features(tdir, spec, con)
            book_tops[spec.token_id] = feats.pop("_book_top")
            feats.update(universe=universe, token_id=spec.token_id, market=spec.market)
            all_feats.append(feats)
            fdf = replay_token_fills(tdir, spec)
            if not fdf.empty:
                fdf = di.asof_attach_book(fdf, book_tops[spec.token_id])
                all_fills.append(fdf)
            print(f"  [{i}/{len(specs)}] {spec.token_id[:10]}… fills={0 if fdf is None else len(fdf)}",
                  flush=True)
    con.close()

    per_fill = pd.concat(all_fills, ignore_index=True)
    per_token = pd.DataFrame(all_feats)
    # derived regressor columns (logs / broadcasts) on the per-fill frame
    per_fill["log_queue_ahead"] = np.log1p(per_fill["queue_ahead"].clip(lower=0))
    per_fill["log_trade_size"] = np.log1p(per_fill["trade_size"].clip(lower=0))
    per_fill["fill_price_extremity"] = (per_fill["price"] - 0.5).abs()
    per_fill["log_abs_inv_before"] = np.log1p(per_fill["abs_inv_before"].clip(lower=0))
    per_fill["log_depth_at_fill"] = np.log1p(per_fill["depth_at_fill"].clip(lower=0))
    # broadcast token-level observable features onto fills
    tok_cols = ["token_id", "half_spread_c", "median_tob_depth", "median_imbalance",
                "median_abs_imbalance", "imbalance_vol", "price_extremity", "trade_intensity",
                "mid_vol_c"]
    per_fill = per_fill.merge(per_token[tok_cols], on="token_id", how="left")
    per_fill["log_tob_depth"] = np.log1p(per_fill["median_tob_depth"].clip(lower=0))
    per_token["log_tob_depth"] = np.log1p(per_token["median_tob_depth"].clip(lower=0))

    per_fill.to_parquet(fills_cache, index=False)
    per_token.drop(columns=[c for c in per_token.columns if c.startswith("_")]).to_parquet(
        feat_cache, index=False)
    print(f"\n[stage A] cached {len(per_fill)} fills / {len(per_token)} tokens -> {CACHE}", flush=True)
    return per_fill, per_token


# ──────────────────────────────────────────────────────────────────────────────
# Stage B — analyses
# ──────────────────────────────────────────────────────────────────────────────

def attach_labels(per_token: pd.DataFrame) -> pd.DataFrame:
    """Join the Task-4 verdict (VIABLE=1/DEAD=0) + Task-4 outcome columns onto the feature table."""
    vd = pd.read_csv(CSV_OUT / "mm_validation_verdict.csv", dtype={"token_id": str})
    vd = vd[["token_id", "verdict_no_rebate", "net_edge_pess_cents", "adverse_pess_cents",
             "stability_sign"]]
    out = per_token.merge(vd, on="token_id", how="left")
    out["viable"] = (out["verdict_no_rebate"] == "VIABLE").astype(float)
    return out


def run_screen(feat_labeled: pd.DataFrame) -> dict:
    """Task 1a — per-token VIABLE-vs-DEAD separation, per-universe + pooled."""
    frames = {}
    for scope, df in [("politics_negrisk", feat_labeled[feat_labeled.universe == "politics_negrisk"]),
                      ("esports", feat_labeled[feat_labeled.universe == "esports"]),
                      ("pooled", feat_labeled)]:
        sc = di.screen_features(df, SCREEN_FEATURES, "viable", n_boot=5000, seed=0)
        sc.insert(0, "scope", scope)
        frames[scope] = sc
    return frames


def run_regression(per_fill: pd.DataFrame, target: str) -> dict:
    """Task 1b — per-fill markout regression (token-clustered bootstrap), per-universe + pooled.

    Two models with the method matched to the regressor type: 'fill' (within-market regressors) uses
    the WILD cluster bootstrap (the small-K-appropriate method for within-cluster effects); 'all'
    (fill + token-level observables) uses the PAIRS cluster bootstrap, because the wild bootstrap is
    known to be anti-conservative for **cluster-constant** regressors (the token-level features are
    constant within a token) — pairs correctly reflects their between-market sampling variability.
    """
    out = {}
    for scope, df in [("politics_negrisk", per_fill[per_fill.universe == "politics_negrisk"]),
                      ("esports", per_fill[per_fill.universe == "esports"]),
                      ("pooled", per_fill)]:
        for model, cols, meth in [("fill", FILL_LEVEL, "wild"),
                                  ("all", FILL_LEVEL + SCREEN_FEATURES, "pairs")]:
            res = di.cluster_bootstrap_ols(df, target, cols, "token_id", n_boot=2000, seed=0,
                                           method=meth)
            res.insert(0, "scope", scope)
            res.insert(1, "model", model)
            res.insert(2, "method", res.attrs["method"])
            res.insert(3, "n_fills", res.attrs["n_fills"])
            res.insert(4, "n_clusters", res.attrs["n_clusters"])
            out[f"{scope}:{model}"] = res
    return out


def run_ttr(per_fill: pd.DataFrame, meta: dict) -> dict:
    """Task 2 — politics time-to-resolution regime + coverage + within-short-dated gradient."""
    pol = per_fill[per_fill.universe == "politics_negrisk"].copy()
    end_map = {cid: end_ms(rec) for cid, rec in meta.items()}
    pol["end_ms"] = pol["market"].map(end_map)
    pol["ttr_hours"] = (pol["end_ms"] - pol["ts_exchange"]) / 1000.0 / 3600.0
    pol["ttr_days"] = pol["ttr_hours"] / 24.0
    lab_map = {cid: rec.get("label", cid[:10]) for cid, rec in meta.items()}
    pol["mkt_label"] = pol["market"].map(lab_map)

    # per-token TTR span (coverage) — what regime does each market's fills sit in?
    span = (pol.groupby(["token_id", "market", "mkt_label"])
            .agg(n_fills=("ttr_hours", "size"),
                 ttr_min_h=("ttr_hours", "min"), ttr_max_h=("ttr_hours", "max"),
                 markout_30=(f"markout_{PRIMARY_H}s_c", lambda s: np.average(s.dropna()) if s.notna().any() else np.nan))
            .reset_index())

    # bucketed edge across all politics fills
    idx = di.ttr_bucketize(pol["ttr_hours"].to_numpy(), di.DEFAULT_TTR_EDGES_H)
    be = di.bucketed_edge(pol[f"markout_{PRIMARY_H}s_c"].to_numpy(),
                          pol[f"adverse_drift_{PRIMARY_H}s_c"].to_numpy(),
                          pol["qty"].to_numpy(), idx, di.DEFAULT_TTR_LABELS,
                          market_ids=pol["market"].to_numpy(), n_boot=2000, seed=0)
    bucket_df = pd.DataFrame([{
        "bucket": b.label, "n_fills": b.n_fills, "n_markets": b.n_markets,
        "markout_30s_c": b.markout_cents.point, "markout_lo": b.markout_cents.lo,
        "markout_hi": b.markout_cents.hi, "adverse_30s_c": b.adverse_cents.point,
        "adverse_lo": b.adverse_cents.lo, "adverse_hi": b.adverse_cents.hi,
        "adverse_rate": b.adverse_rate,
    } for b in be])

    # within short-dated (in-window resolver) markets: does markout rise with log(TTR)?
    short = pol[pol["ttr_hours"] < 24 * 20]  # resolves within ~20 days of the fill -> in-window set
    short = short[np.isfinite(short["ttr_hours"]) & (short["ttr_hours"] > 0)].copy()
    grad = None
    fe = None
    fe_pairs = None
    if len(short) > 50 and short["market"].nunique() >= 2:
        short["log_ttr_h"] = np.log1p(short["ttr_hours"])
        # (a) pooled slope, wild cluster bootstrap (log_ttr varies within market -> wild appropriate)
        grad = di.cluster_bootstrap_ols(short, f"markout_{PRIMARY_H}s_c", ["log_ttr_h"],
                                        "market", n_boot=2000, seed=0, method="wild")
        grad.attrs["scope"] = "politics_short_dated"
        # (b) MARKET FIXED-EFFECTS within-slope, both methods (wild primary, pairs for transparency)
        fe = di.within_market_slope_ci(short, f"markout_{PRIMARY_H}s_c", "log_ttr_h", "market",
                                       n_boot=3000, seed=0, method="wild")
        fe_pairs = di.within_market_slope_ci(short, f"markout_{PRIMARY_H}s_c", "log_ttr_h", "market",
                                             n_boot=3000, seed=0, method="pairs")

    return {"span": span, "bucket": bucket_df, "gradient": grad, "gradient_fe": fe,
            "gradient_fe_pairs": fe_pairs,
            "n_short_fills": int(len(short)), "n_short_markets": int(short["market"].nunique())}


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_screen_auc(screen_pooled: pd.DataFrame, path: Path) -> None:
    df = screen_pooled.sort_values("auc")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(df))
    ax.errorbar(df["auc"], y, xerr=[df["auc"] - df["auc_lo"], df["auc_hi"] - df["auc"]],
                fmt="o", color="#4878d0", ecolor="#999", capsize=3)
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(df["feature"])
    ax.set_xlabel("rank-AUC (P[feature ranks VIABLE > DEAD]); 0.5 = no separation")
    ax.set_title("Observable-feature separation of VIABLE vs DEAD (pooled 24 tokens, bootstrapped)")
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_regression(reg_pooled_all: pd.DataFrame, path: Path) -> None:
    df = reg_pooled_all[reg_pooled_all["term"] != "intercept"].copy()
    df = df.sort_values("coef_cents_per_sd")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(df))
    colors = ["#d65f5f" if r else "#bbb" for r in df["sign_robust"]]
    ax.errorbar(df["coef_cents_per_sd"], y, xerr=[df["coef_cents_per_sd"] - df["lo"],
                df["hi"] - df["coef_cents_per_sd"]], fmt="none", ecolor="#999", capsize=3)
    ax.scatter(df["coef_cents_per_sd"], y, color=colors, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(df["term"])
    ax.set_xlabel("Δ per-fill markout(30s) ¢/contract per +1 SD (token-clustered CI; red = CI excl. 0)")
    ax.set_title("Per-fill markout drivers (pooled, token-clustered bootstrap)")
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_ttr(bucket_df: pd.DataFrame, path: Path) -> None:
    df = bucket_df[bucket_df["n_fills"] > 0]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(df))
    ax.errorbar(x - 0.08, df["markout_30s_c"],
                yerr=[df["markout_30s_c"] - df["markout_lo"], df["markout_hi"] - df["markout_30s_c"]],
                fmt="o-", color="#4878d0", capsize=3, label="net markout(30s) ¢")
    ax.errorbar(x + 0.08, df["adverse_30s_c"],
                yerr=[df["adverse_30s_c"] - df["adverse_lo"], df["adverse_hi"] - df["adverse_30s_c"]],
                fmt="s--", color="#d65f5f", capsize=3, label="adverse selection ¢")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.bucket}\nn={r.n_fills}\n({r.n_markets} mkt)" for r in df.itertuples()],
                       fontsize=8)
    ax.set_xlabel("time-to-resolution bucket (near expiry → mid-life)")
    ax.set_ylabel("¢/contract")
    ax.set_title("Politics-NegRisk: per-fill markout & adverse selection vs time-to-resolution")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild stage-A cache (materialize+replay)")
    ap.add_argument("--quick", action="store_true", help="top_k=4 smoke")
    args = ap.parse_args()

    CSV_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    meta = load_market_meta()

    per_fill, per_token = build_stage_a(args.force, args.quick)
    feat_labeled = attach_labels(per_token)

    # Task 1a — screen
    screens = run_screen(feat_labeled)
    screen_all = pd.concat(screens.values(), ignore_index=True)
    screen_all.to_csv(CSV_OUT / "mm_design_screen.csv", index=False)
    feat_labeled.drop(columns=[c for c in feat_labeled.columns if c.startswith("_")]).to_csv(
        CSV_OUT / "mm_design_token_features.csv", index=False)

    # Task 1b — regression
    regs = run_regression(per_fill, f"markout_{PRIMARY_H}s_c")
    reg_all = pd.concat(regs.values(), ignore_index=True)
    reg_all.to_csv(CSV_OUT / "mm_design_fill_regression.csv", index=False)

    # Task 2 — TTR regime
    ttr = run_ttr(per_fill, meta)
    ttr["span"].to_csv(CSV_OUT / "mm_design_ttr_span.csv", index=False)
    ttr["bucket"].to_csv(CSV_OUT / "mm_design_ttr_buckets.csv", index=False)
    grad_rows = []
    if ttr["gradient"] is not None:
        g = ttr["gradient"].set_index("term").loc["log_ttr_h"]
        grad_rows.append({"spec": "pooled", "method": "wild", "term": "log_ttr_h",
                          "coef_cents_per_logh": g["coef_cents_per_sd"], "lo": g["lo"], "hi": g["hi"],
                          "sign_robust": g["sign_robust"], "n_fills": ttr["n_short_fills"],
                          "n_markets": ttr["n_short_markets"]})
    for key, meth in [("gradient_fe", "wild"), ("gradient_fe_pairs", "pairs")]:
        fe = ttr.get(key)
        if fe is not None:
            grad_rows.append({"spec": "market_fixed_effects", "method": meth, "term": "log_ttr_h",
                              "coef_cents_per_logh": fe["slope"], "lo": fe["lo"], "hi": fe["hi"],
                              "sign_robust": fe["sign_robust"], "n_fills": fe["n"],
                              "n_markets": fe["n_markets"]})
    grad_df = pd.DataFrame(grad_rows)
    if not grad_df.empty:
        grad_df.to_csv(CSV_OUT / "mm_design_ttr_gradient.csv", index=False)
    # per-market within-slopes (makes the "N of M markets have positive slopes" claim reproducible)
    if ttr["gradient_fe"] is not None and ttr["gradient_fe"]["per_market_slopes"]:
        lab_map = {cid: rec.get("label", cid[:10]) for cid, rec in meta.items()}
        pm = pd.DataFrame([{"market": m, "label": lab_map.get(m, m[:10]), "within_slope_cents_per_logh": s}
                           for m, s in ttr["gradient_fe"]["per_market_slopes"].items()])
        pm.to_csv(CSV_OUT / "mm_design_ttr_per_market_slope.csv", index=False)

    # plots
    plot_screen_auc(screens["pooled"], PLOT_OUT / "mm_design_screen_auc.png")
    plot_regression(regs["pooled:all"], PLOT_OUT / "mm_design_fill_regression.png")
    plot_ttr(ttr["bucket"], PLOT_OUT / "mm_design_ttr_regime.png")

    # console summary (feeds the note)
    print("\n================ TASK 1a — MARKET SCREEN (VIABLE vs DEAD) ================")
    for scope in ("politics_negrisk", "esports", "pooled"):
        print(f"\n--- {scope} ---")
        print(screens[scope][["feature", "auc", "auc_lo", "auc_hi", "median_viable",
                              "median_dead", "diff", "diff_lo", "diff_hi"]].to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n================ TASK 1b — PER-FILL MARKOUT REGRESSION (pooled) ================")
    for m in ("fill", "all"):
        r = regs[f"pooled:{m}"]
        print(f"\n--- pooled : model={m} (n_fills={r.attrs['n_fills']}, clusters={r.attrs['n_clusters']}) ---")
        print(r[["term", "coef_cents_per_sd", "lo", "hi", "sign_robust"]].to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n================ TASK 2 — TTR REGIME (politics) ================")
    print("\n-- per-market TTR span (coverage) --")
    sp = ttr["span"].sort_values("ttr_max_h")
    print(sp.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("\n-- bucketed markout & adverse vs TTR --")
    print(ttr["bucket"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if not grad_df.empty:
        print(f"\n-- within short-dated markets: markout ~ log(TTR)  "
              f"(n_fills={ttr['n_short_fills']}, markets={ttr['n_short_markets']}) --")
        print(grad_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        if ttr["gradient_fe"] is not None:
            print("   per-market within slopes:", {k: round(v, 3)
                  for k, v in ttr["gradient_fe"]["per_market_slopes"].items()})

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_fills": int(len(per_fill)), "n_tokens": int(len(per_token)),
        "queue": "RiskAverse(pessimistic)", "primary_horizon_s": PRIMARY_H,
        "screen_pooled": screens["pooled"].to_dict(orient="records"),
        "regression_pooled_all": regs["pooled:all"][["term", "coef_cents_per_sd", "lo", "hi",
                                                      "sign_robust"]].to_dict(orient="records"),
        "ttr_buckets": ttr["bucket"].to_dict(orient="records"),
        "ttr_gradient": grad_df.to_dict(orient="records") if not grad_df.empty else None,
    }
    (SCRATCH / "mm_design_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nCSVs -> {CSV_OUT}\nPlots -> {PLOT_OUT}\nSummary -> {SCRATCH/'mm_design_summary.json'}")


if __name__ == "__main__":
    main()
