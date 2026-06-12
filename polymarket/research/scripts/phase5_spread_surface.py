#!/usr/bin/env python3
"""Block SPREAD-2 — Phase 5 mid-as-data + category-gated surface fallback.

Wires the SPREAD-1/1b trade-anchored spread surface into the copy-execution
evaluators, WITHOUT re-running any expensive backtest SQL. Everything reads
cached artifacts:
  * leader audits  -> data/analysis/**/<label>_audit_fragments.parquet
  * stage15 cohorts-> data/backtests/stage15/*__stage15.parquet
  * the frozen surface + bounce levels -> csv_outputs/copytrade/spread_surface_v1_*.csv

Subcommands (PYTHONPATH=. uv run python scripts/phase5_spread_surface.py <cmd>):
  fetch-mids   Prefetch /prices-history mids for a leader's (token, day) anchors
               at 1-min fidelity (SPREAD-1 cache keying; idempotent). Default
               leader = Domah (the canonical smoke leader). Used by the
               drift-vs-spread diagnostic only.
  fetch-mtm    Prefetch coarse-fidelity mid histories for the stage15 held-token
               universe (one span call per token; for the MTM marks).
  regate       Re-price the FALLBACK rows of every leader audit + every stage15
               cohort under {flat-3c, gated surface_fallback}; emit the
               fallback-shrink + gate/verdict-flip tables + (Domah) the
               mid-anchored drift-vs-spread split.
  mtm          Lookahead-free daily MTM equity per stage15 cohort.
  charts       MTM equity-per-cohort chart + drift-vs-spread chart.

The surface is FROZEN (SPREAD-1). surface_fallback fires ONLY on fallback rows
(no observed next-fill) and ONLY for SURFACE_VALIDATED_CATEGORIES; politics_negrisk
and any non-validated category keep flat-3c. See lib/copy_slippage.py.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_infra.views import latest_markets_path  # noqa: E402
from lib.copy_slippage import (  # noqa: E402
    SURFACE_VALIDATED_CATEGORIES,
    asof_mid_before,
    decompose_next_fill_slippage,
    k5_category,
    leader_vs_mid_cents,
    load_bounce_lookup,
    mtm_equity_curve,
    reprice_fallback_rows,
)
from lib.spread_surface import SpreadSurface, ttr_bucket  # noqa: E402
from scripts.spread_surface_build import MID_CACHE, fetch_mid_history  # noqa: E402

DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
CSV_OUT = ANALYSIS / "csv_outputs" / "copytrade"
PLOTS_OUT = ANALYSIS / "plots" / "copytrade"
STAGE15_GLOB = str(DATA / "backtests" / "stage15" / "*__stage15.parquet")
SURFACE_CSV = CSV_OUT / "spread_surface_v1_surface.csv"
BREAKS_CSV = CSV_OUT / "spread_surface_v1_activity_breaks.csv"
XCHECK_CSV = CSV_OUT / "spread_surface_v1_diag_crosschecks.csv"
MTM_MID_CACHE = ANALYSIS / "spread_surface" / "mtm_mid_history"
DOMAH = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344"

# leader label -> cached fragments parquet (re-run universe; all cached)
LEADER_FRAGMENTS: dict[str, str] = {
    "domah": str(ANALYSIS / "domah_audit_fragments.parquet"),
    "high_conviction": str(ANALYSIS / "leader_high_conviction"
                           / "leader_high_conviction_audit_fragments.parquet"),
    "ultra_maker": str(ANALYSIS / "leader_ultra_maker"
                       / "leader_ultra_maker_audit_fragments.parquet"),
    "top_leaderboard": str(ANALYSIS / "leader_top_leaderboard"
                           / "leader_top_leaderboard_audit_fragments.parquet"),
    "negrisk_directional_1": str(ANALYSIS / "leader_negrisk_directional_1"
                                 / "leader_negrisk_directional_1_audit_fragments.parquet"),
    "negrisk_directional_2": str(ANALYSIS / "leader_negrisk_directional_2"
                                 / "leader_negrisk_directional_2_audit_fragments.parquet"),
    "ee00ba": str(ANALYSIS / "domah_followups" / "leader_ee00ba_audit_fragments.parquet"),
}
LEADER_POSITIONS = {k: v.replace("_fragments.parquet", "_positions.parquet")
                    for k, v in LEADER_FRAGMENTS.items()}

API_SLEEP_S = 0.25


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def ensure_dirs() -> None:
    for d in (CSV_OUT, PLOTS_OUT, MTM_MID_CACHE):
        d.mkdir(parents=True, exist_ok=True)


def _day_window(day) -> tuple[int, int]:
    """SPREAD-1 keying: [day0 - 1h, day0 + 24h] so caches are shared/reused."""
    day0 = pd.Timestamp(day).tz_localize("UTC") if pd.Timestamp(day).tzinfo is None \
        else pd.Timestamp(day)
    return int(day0.timestamp()) - 3600, int(day0.timestamp()) + 86400


def _load_surface() -> tuple[SpreadSurface, dict]:
    surf = SpreadSurface.load(SURFACE_CSV, BREAKS_CSV)
    bounce = load_bounce_lookup(XCHECK_CSV)
    return surf, bounce


# ============================================================================
# fetch-mids  (leader anchors, 1-min per-day; for the drift/spread diagnostic)
# ============================================================================
def cmd_fetch_mids(args: argparse.Namespace) -> int:
    ensure_dirs()
    label = args.leader
    frag_path = LEADER_FRAGMENTS[label]
    f = pd.read_parquet(frag_path, columns=["outcome_token_id", "fill_ts"])
    f = f.dropna(subset=["outcome_token_id"])
    f["day"] = pd.to_datetime(f["fill_ts"]).dt.date
    td = (f.groupby(["outcome_token_id", "day"]).size().reset_index(name="n"))
    log(f"{label}: {len(td):,} distinct (token, day) anchors to ensure cached")

    client = httpx.Client(timeout=30.0)
    n_hit = n_cached = n_empty = 0
    t0 = time.time()
    for i, r in enumerate(td.itertuples(index=False)):
        start_s, end_s = _day_window(r.day)
        cache = MID_CACHE / f"midhist_{str(r.outcome_token_id)[:16]}_{start_s}_{end_s}_f1.parquet"
        if cache.exists():
            n_cached += 1
            continue
        df = fetch_mid_history(str(r.outcome_token_id), start_s, end_s, fidelity_min=1, client=client)
        n_hit += 1
        n_empty += 0 if len(df) else 1
        if (i + 1) % 250 == 0:
            rate = (i + 1) / max(1e-9, time.time() - t0)
            log(f"  {i+1:,}/{len(td):,}  (fetched {n_hit:,}, cached-skip {n_cached:,}, "
                f"empty {n_empty:,})  {rate:.1f}/s")
    client.close()
    log(f"{label} done: fetched {n_hit:,}, already-cached {n_cached:,}, empty {n_empty:,} "
        f"in {time.time()-t0:,.0f}s")
    return 0


# ============================================================================
# mid_at_trade attach (Domah drift/spread diagnostic)
# ============================================================================
def _attach_mid_at_trade(frag: pd.DataFrame) -> pd.DataFrame:
    """As-of join each fragment to the cached mid strictly before its fill."""
    f = frag.copy()
    f["fill_ts"] = pd.to_datetime(f["fill_ts"])
    f["day"] = f["fill_ts"].dt.date
    # unit-robust epoch seconds: fragment fill_ts is datetime64[us], so a raw
    # .astype('int64')/1e9 would be 1000x too small. total_seconds() is unit-agnostic.
    f["fill_epoch"] = (f["fill_ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
    cache: dict = {}
    mids = np.full(len(f), np.nan)
    ages = np.full(len(f), np.nan)
    for idx, r in enumerate(f.itertuples(index=False)):
        tok = str(r.outcome_token_id)
        key = (tok, r.day)
        if key not in cache:
            start_s, end_s = _day_window(r.day)
            cpath = MID_CACHE / f"midhist_{tok[:16]}_{start_s}_{end_s}_f1.parquet"
            if cpath.exists():
                h = pd.read_parquet(cpath).sort_values("t")
                cache[key] = (h["t"].to_numpy(), h["p"].to_numpy())
            else:
                cache[key] = (np.array([]), np.array([]))
        ts_arr, p_arr = cache[key]
        m, age = asof_mid_before(ts_arr, p_arr, r.fill_epoch, max_age_s=1800.0)
        if m is not None:
            mids[idx], ages[idx] = m, age
    f["mid_at_trade"] = mids
    f["mid_age_s"] = ages
    f["leader_vs_mid_cents"] = [
        leader_vs_mid_cents(p, (m if np.isfinite(m) else None), d)
        for p, m, d in zip(f["price"], f["mid_at_trade"], f["direction"], strict=True)
    ]
    return f


# ============================================================================
# K5 category + surface-fallback re-price for a fragment / audit frame
# ============================================================================
def _attach_k5_category(df: pd.DataFrame, market_lookup: pd.DataFrame,
                        slug_col: str = "slug") -> pd.DataFrame:
    """Join market metadata (question, neg_risk) by market_id, then map to K5.
    neg_risk is REQUIRED for politics_negrisk to resolve (else it would fall to
    'other' and wrongly become validated)."""
    out = df.merge(market_lookup, on="market_id", how="left", suffixes=("", "_mkt"))
    if slug_col not in out.columns and "slug_mkt" in out.columns:
        out[slug_col] = out["slug_mkt"]
    out["k5_category"] = k5_category(out, slug_col=slug_col,
                                     question_col="question", neg_risk_col="neg_risk")
    return out


def _market_lookup(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    mp = latest_markets_path()
    return con.execute(f"""
        SELECT CAST(id AS VARCHAR) AS market_id,
               coalesce(slug, '')      AS slug,
               coalesce(question, '')  AS question,
               coalesce(neg_risk, false) AS neg_risk
        FROM read_parquet('{mp}')
    """).df()


def _surface_repricing(df: pd.DataFrame, surf: SpreadSurface, bounce: dict,
                       *, price_col: str, ttr_h_col: str, dir_col: str,
                       maker_col: pd.Series, leader_price_col: str,
                       is_fallback: pd.Series) -> pd.DataFrame:
    """Thin wrapper over the shared lib.copy_slippage.reprice_fallback_rows (the
    same core the three evaluator entry points call)."""
    return reprice_fallback_rows(
        df, surf, bounce, price_col=price_col, ttr_h_col=ttr_h_col, dir_col=dir_col,
        maker_col=maker_col, leader_price_col=leader_price_col, is_fallback=is_fallback)


# ============================================================================
# regate
# ============================================================================
def _regate_leaders(surf, bounce, con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-leader fallback-shrink + Branch-B PnL under {flat-3c, surface_fallback}.

    Branch B (pure taker) is the branch maximally exposed to fallback slippage:
    every fill crosses, so fallback rows pay the full modeled cost. PnL is
    recomputed against each position's cached resolution mark_price."""
    mkts = _market_lookup(con)
    summary_rows: list[dict] = []
    for label, frag_path in LEADER_FRAGMENTS.items():
        if not Path(frag_path).exists():
            log(f"  {label}: fragments missing — skipped")
            continue
        cols = ["fill_ts", "market_id", "outcome_token_id", "role", "direction",
                "price", "token_amount", "slug", "hours_to_resolution", "family",
                "position_id", "nf1_price", "taker_source"]
        f = pd.read_parquet(frag_path, columns=cols)
        f = _attach_k5_category(f, mkts)
        is_fallback = (f["taker_source"] == "fallback")
        f = _surface_repricing(
            f, surf, bounce, price_col="price", ttr_h_col="hours_to_resolution",
            dir_col="direction", maker_col=(f["role"] == "maker"),
            leader_price_col="price", is_fallback=is_fallback)

        # taker copy price per regime: next_fill where present, else the regime fallback
        has_nf = f["nf1_price"].astype(float).notna() & np.isfinite(f["nf1_price"].astype(float))
        f["taker_price_flat3c"] = np.where(has_nf, f["nf1_price"].astype(float),
                                           f["flat3c_copy_price"])
        f["taker_price_surface"] = np.where(has_nf, f["nf1_price"].astype(float),
                                            f["sf_copy_price"])

        # Branch-B PnL per position against cached mark_price
        ppath = LEADER_POSITIONS.get(label)
        mark = None
        if ppath and Path(ppath).exists():
            mark = pd.read_parquet(ppath, columns=["position_id", "mark_price"])
        pnl = _branch_b_pnl(f, mark)

        n_fb = int(is_fallback.sum())
        n_fb_validated = int((is_fallback & f["k5_category"].isin(SURFACE_VALIDATED_CATEGORIES)).sum())
        n_fb_bounce = int(f["sf_used_bounce"].sum())
        fb_cents_surface = f.loc[is_fallback & f["sf_source"].str.startswith("surface"),
                                 "sf_fallback_cents"]
        summary_rows.append({
            "leader": label, "n_fills": len(f),
            "n_fallback": n_fb,
            "fallback_share": n_fb / len(f) if len(f) else 0.0,
            "fallback_share_gt40pct": (n_fb / len(f) > 0.40) if len(f) else False,
            "n_fallback_validated": n_fb_validated,
            "fallback_shrink_share": n_fb_validated / n_fb if n_fb else 0.0,
            "n_fallback_bounce_swapped": n_fb_bounce,
            "mean_surface_fallback_cents": float(fb_cents_surface.mean()) if len(fb_cents_surface) else float("nan"),
            "median_surface_fallback_cents": float(fb_cents_surface.median()) if len(fb_cents_surface) else float("nan"),
            "branchB_pnl_flat3c": pnl["flat3c"],
            "branchB_pnl_surface": pnl["surface"],
            "branchB_pnl_delta": pnl["surface"] - pnl["flat3c"],
            "branchB_verdict_flat3c": "profitable" if pnl["flat3c"] > 0 else "unprofitable",
            "branchB_verdict_surface": "profitable" if pnl["surface"] > 0 else "unprofitable",
            "branchB_verdict_flip": (pnl["flat3c"] > 0) != (pnl["surface"] > 0),
        })
    return pd.DataFrame(summary_rows)


def _branch_b_pnl(f: pd.DataFrame, mark: pd.DataFrame | None) -> dict:
    """Pure-taker PnL per regime: every fill crosses at taker_price_*; mark final
    inventory to cached resolution mark_price (unresolved -> last fragment price)."""
    g = f.copy()
    is_buy = (g["direction"].str.upper() == "BUY").to_numpy()
    tok = g["token_amount"].to_numpy(dtype=float)
    cash_sign = np.where(is_buy, -1.0, 1.0)
    token_sign = np.where(is_buy, 1.0, -1.0)
    out = {}
    contrib = pd.DataFrame({"position_id": g["position_id"].to_numpy(),
                            "token": token_sign * tok,
                            "last_price": g["price"].to_numpy(dtype=float)})
    for regime, pxcol in (("flat3c", "taker_price_flat3c"), ("surface", "taker_price_surface")):
        contrib[f"cash_{regime}"] = cash_sign * tok * g[pxcol].to_numpy(dtype=float)
    agg = contrib.groupby("position_id", sort=False).agg(
        token=("token", "sum"), last_price=("last_price", "last"),
        cash_flat3c=("cash_flat3c", "sum"), cash_surface=("cash_surface", "sum")).reset_index()
    if mark is not None:
        agg = agg.merge(mark.drop_duplicates("position_id"), on="position_id", how="left")
        agg["mark_price"] = agg["mark_price"].fillna(agg["last_price"])
    else:
        agg["mark_price"] = agg["last_price"]
    for regime in ("flat3c", "surface"):
        out[regime] = float((agg[f"cash_{regime}"] + agg["token"] * agg["mark_price"]).sum())
    return out


def _regate_stage15(surf, bounce, con) -> pd.DataFrame:
    """Recompute fallback-row copy_price + copy_pnl under surface_fallback from
    the cached stage15 audit logs (no backtest re-run), then re-derive the §8
    success-criteria-relevant headline (PnL sign + monthly Sharpe) per run and
    flag verdict flips. stage15 leaders are makers by construction (signals are
    pulled on maker IN cohort), so leader_is_maker=True -> full spread."""
    mkts = _market_lookup(con)
    rows = []
    for path in sorted(glob.glob(STAGE15_GLOB)):
        a = pd.read_parquet(path)
        if a.empty:
            continue
        a = _attach_k5_category(a, mkts)  # audit has market_id; slug/question via join
        a["_ttr_h"] = a["days_to_resolution"].astype(float) * 24.0
        is_fallback = (a["slippage_source"] == "fallback")
        a = _surface_repricing(
            a, surf, bounce, price_col="leader_price", ttr_h_col="_ttr_h",
            dir_col="leader_direction", maker_col=pd.Series(True, index=a.index),
            leader_price_col="leader_price", is_fallback=is_fallback)

        # recompute copy_pnl on fallback rows under surface; keep next_fill rows
        res = a["position_resolution"].astype(float).to_numpy()
        size = a["copy_size_usd"].astype(float).to_numpy()
        is_buy = (a["leader_direction"].str.upper() == "BUY").to_numpy()

        def pnl_for(copy_price: np.ndarray) -> float:
            tok = size / copy_price
            return float(np.where(is_buy, (res - copy_price) * tok,
                                  (copy_price - res) * tok).sum())

        cp_flat = a["copy_price"].astype(float).to_numpy().copy()      # as-logged (flat-3c)
        cp_surf = cp_flat.copy()
        fbm = is_fallback.to_numpy()
        cp_surf[fbm] = a["sf_copy_price"].to_numpy(dtype=float)[fbm]
        # rows the surface declined (non-validated cat) keep flat-3c -> sf_copy_price
        # already equals the flat fallback there? No: _surface_repricing sets
        # sf_copy_price via surface_fallback_cents which returns flat3c for
        # non-validated cats, so sf_copy_price == flat fallback there. Safe.
        valid = np.isfinite(cp_surf) & (cp_surf > 0) & (cp_surf < 1)
        cp_surf = np.where(valid, cp_surf, cp_flat)

        pnl_flat = pnl_for(cp_flat)
        pnl_surf = pnl_for(cp_surf)
        a2 = a.assign(_cp_flat=cp_flat, _cp_surf=cp_surf, _pnl_flat=None)
        sharpe_flat = _monthly_sharpe(a, cp_flat, is_buy, res, size)
        sharpe_surf = _monthly_sharpe(a, cp_surf, is_buy, res, size)

        n_fb = int(fbm.sum())
        n_fb_val = int((is_fallback & a["k5_category"].isin(SURFACE_VALIDATED_CATEGORIES)).sum())
        rid = Path(path).stem
        parts = rid.split("__")
        rows.append({
            "run_id": rid,
            "cohort": parts[0], "bucket": parts[1] if len(parts) > 1 else "",
            "sizing": parts[2] if len(parts) > 2 else "",
            "window": parts[3] if len(parts) > 3 else "",
            "n_signals": len(a), "n_fallback": n_fb,
            "fallback_share": n_fb / len(a) if len(a) else 0.0,
            "fallback_share_gt40pct": (n_fb / len(a) > 0.40) if len(a) else False,
            "fallback_shrink_share": n_fb_val / n_fb if n_fb else 0.0,
            "n_fallback_bounce_swapped": int(a["sf_used_bounce"].sum()),
            "pnl_flat3c": pnl_flat, "pnl_surface": pnl_surf,
            "pnl_delta": pnl_surf - pnl_flat,
            "sharpe_flat3c": sharpe_flat, "sharpe_surface": sharpe_surf,
            "pnl_sign_flip": (pnl_flat > 0) != (pnl_surf > 0),
            "sharpe_cross_1.0_flip": (sharpe_flat >= 1.0) != (sharpe_surf >= 1.0),
        })
    return pd.DataFrame(rows)


def _monthly_sharpe(a: pd.DataFrame, copy_price: np.ndarray, is_buy, res, size) -> float:
    tok = size / copy_price
    pnl = np.where(is_buy, (res - copy_price) * tok, (copy_price - res) * tok)
    df = pd.DataFrame({"month": pd.to_datetime(a["trade_timestamp"]).dt.to_period("M"),
                       "pnl": pnl})
    m = df.groupby("month")["pnl"].sum()
    sd = float(m.std(ddof=0))
    return float(m.mean() / sd * (12 ** 0.5)) if sd > 0 else 0.0


def _domah_drift_spread(surf, bounce, con) -> pd.DataFrame:
    """Domah next-fill rows: mid-anchored drift-vs-spread split. Needs the
    Domah mid cache (cmd_fetch_mids --leader domah)."""
    mkts = _market_lookup(con)
    f = pd.read_parquet(LEADER_FRAGMENTS["domah"])
    f = _attach_mid_at_trade(f)
    f = _attach_k5_category(f, mkts)
    nf = f[(f["taker_source"] == "next_fill") & f["mid_at_trade"].notna()].copy()
    rows = []
    for r in nf.itertuples(index=False):
        ttr_i = None if not np.isfinite(r.hours_to_resolution) else float(r.hours_to_resolution)
        pred = surf.predict(float(r.mid_at_trade), ttr_i, 0.0, r.k5_category)
        split = decompose_next_fill_slippage(float(r.nf1_price), float(r.mid_at_trade),
                                             pred.half_spread_cents, r.direction)
        if split is None:
            continue
        rows.append({"family": r.family, "k5_category": r.k5_category,
                     "leader_vs_mid_cents": r.leader_vs_mid_cents,
                     "copy_vs_mid_c": split.copy_vs_mid_c, "spread_c": split.spread_c,
                     "drift_c": split.drift_c, "is_drift": split.is_drift})
    d = pd.DataFrame(rows)
    return d


def cmd_regate(args: argparse.Namespace) -> int:
    ensure_dirs()
    surf, bounce = _load_surface()
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    log("regate: leaders (fallback-shrink + Branch-B verdict flip)…")
    leader_summ = _regate_leaders(surf, bounce, con)
    leader_summ.to_csv(CSV_OUT / "spread_surface_phase5_leader_regate.csv", index=False)
    print(leader_summ.to_string(index=False))

    log("regate: stage15 cohorts (gate-flip from cached audit logs)…")
    stage_summ = _regate_stage15(surf, bounce, con)
    stage_summ.to_csv(CSV_OUT / "spread_surface_phase5_stage15_regate.csv", index=False)
    # cohort rollup
    roll = (stage_summ.groupby("cohort")
            .agg(n_runs=("run_id", "size"),
                 mean_fallback_share=("fallback_share", "mean"),
                 mean_fallback_shrink=("fallback_shrink_share", "mean"),
                 n_pnl_sign_flip=("pnl_sign_flip", "sum"),
                 n_sharpe_flip=("sharpe_cross_1.0_flip", "sum"),
                 mean_pnl_delta=("pnl_delta", "mean")).reset_index())
    roll.to_csv(CSV_OUT / "spread_surface_phase5_stage15_cohort_rollup.csv", index=False)
    print("\nstage15 cohort rollup:")
    print(roll.to_string(index=False))

    if args.with_domah_drift:
        log("regate: Domah drift-vs-spread split (mid-anchored)…")
        d = _domah_drift_spread(surf, bounce, con)
        if len(d):
            d.to_csv(CSV_OUT / "spread_surface_phase5_domah_drift_spread.csv", index=False)
            by = (d.groupby("k5_category")
                  .agg(n=("drift_c", "size"), mean_spread_c=("spread_c", "mean"),
                       mean_drift_c=("drift_c", "mean"), drift_share=("is_drift", "mean"),
                       mean_leader_vs_mid_c=("leader_vs_mid_cents", "mean")).reset_index())
            by.to_csv(CSV_OUT / "spread_surface_phase5_domah_drift_by_cat.csv", index=False)
            print("\nDomah drift-vs-spread by K5 category:")
            print(by.to_string(index=False))
        else:
            log("  no Domah mid-anchored next-fill rows — run fetch-mids --leader domah first")
    return 0


# ============================================================================
# fetch-mtm  +  mtm  (cohort MTM equity from cached stage15 logs)
# ============================================================================
def _stage15_positions(con) -> pd.DataFrame:
    """Stage15 audit positions enriched with outcome_token_id (markets join)."""
    mp = latest_markets_path()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mtok AS
        SELECT CAST(m.id AS VARCHAR) AS market_id, r.i AS outcome_index,
               m.clob_token_ids[r.i] AS outcome_token_id
        FROM read_parquet('{mp}') m, range(1, len(m.clob_token_ids)+1) AS r(i)
        WHERE len(m.clob_token_ids) > 0
    """)
    a = con.execute(f"""
        SELECT s.cohort, s.market_id, s.outcome_index, t.outcome_token_id,
               s.trade_timestamp, s.resolution_date, s.leader_direction,
               s.copy_price, s.copy_token_amount, s.copy_size_usd, s.position_resolution
        FROM read_parquet('{STAGE15_GLOB}') s
        LEFT JOIN mtok t ON t.market_id = s.market_id AND t.outcome_index = s.outcome_index
    """).df()
    return a


def _fetch_history(client: httpx.Client, token: str, start_s: int, end_s: int,
                   fidelity: int, retries: int = 3) -> tuple[list, bool]:
    """Return (history_points, ok). ok=True iff the API returned HTTP 200 (an
    empty history from a 200 is a LEGIT 'no quotes' answer and is cacheable; a
    4xx/5xx/timeout is transient and must NOT be cached). Retries transient
    failures with linear backoff."""
    for attempt in range(retries):
        try:
            rr = client.get("https://clob.polymarket.com/prices-history",
                            params={"market": token, "startTs": start_s,
                                    "endTs": end_s, "fidelity": fidelity})
            rr.raise_for_status()
            return rr.json().get("history", []), True
        except Exception:  # noqa: BLE001 — transient; back off and retry
            time.sleep(0.5 * (attempt + 1))
    return [], False


MTM_CHUNK_DAYS = 10   # /prices-history silently returns 0 points when span x
#                       fidelity exceeds its cap; 10d @ f60 = 240 pts is safe.


def cmd_fetch_mtm(args: argparse.Namespace) -> int:
    ensure_dirs()
    con = duckdb.connect()
    pos = _stage15_positions(con)
    pos = pos.dropna(subset=["outcome_token_id"]).drop_duplicates("outcome_token_id")
    log(f"MTM held-token universe: {len(pos):,} distinct tokens "
        f"(chunked {MTM_CHUNK_DAYS}d @ f60 — endpoint caps points per call)")
    client = httpx.Client(timeout=30.0)
    chunk_s = MTM_CHUNK_DAYS * 86400
    n_done = n_cached = n_fail = n_empty = 0
    for i, r in enumerate(pos.itertuples(index=False)):
        ent = pd.to_datetime(r.trade_timestamp, utc=True)
        res = pd.to_datetime(r.resolution_date, utc=True)
        start_s = int(ent.timestamp()) - 86400
        end_s = int(res.timestamp()) + 86400
        cache = MTM_MID_CACHE / f"mtm_{str(r.outcome_token_id)[:16]}_{start_s}_{end_s}_f60.parquet"
        if cache.exists():
            n_cached += 1
            continue
        pts: list[tuple[float, float]] = []
        ok_all = True
        cs = start_s
        while cs < end_s:
            ce = min(cs + chunk_s, end_s)
            hist, ok = _fetch_history(client, str(r.outcome_token_id), cs, ce, 60)
            if not ok:
                ok_all = False
                break
            pts.extend((float(h["t"]), float(h["p"])) for h in hist)
            time.sleep(API_SLEEP_S)
            cs = ce
        if not ok_all:
            n_fail += 1            # transient — leave uncached so a re-run retries it
            continue
        # dedup + sort (chunk edges can repeat a boundary point)
        if pts:
            dfp = (pd.DataFrame(pts, columns=["t", "p"]).drop_duplicates("t")
                   .sort_values("t"))
        else:
            dfp = pd.DataFrame({"t": [], "p": []})
        n_empty += 0 if len(dfp) else 1
        dfp.to_parquet(cache, compression="zstd", index=False)
        n_done += 1
        if (i + 1) % 100 == 0:
            log(f"  {i+1:,}/{len(pos):,} (fetched {n_done:,}, cached {n_cached:,}, "
                f"legit-empty {n_empty:,}, transient-fail {n_fail:,})")
    client.close()
    log(f"MTM fetch done: fetched {n_done:,} (legit-empty {n_empty:,}), "
        f"already-cached {n_cached:,}, transient-fail-left-uncached {n_fail:,}")
    return 0


def cmd_fetch_mtm_leader(args: argparse.Namespace) -> int:
    """Daily-fidelity full-life mid history per held token for a leader's audit
    ledger — ONE `interval=max&fidelity=1440` call per token (the only form the
    endpoint serves coarse fidelity in; startTs/endTs spans are capped ~15d).
    Cache-on-200-only with retries (a failure must never poison the cache)."""
    ensure_dirs()
    label = args.leader
    f = pd.read_parquet(LEADER_FRAGMENTS[label], columns=["outcome_token_id"])
    toks = sorted(set(f["outcome_token_id"].dropna().astype(str)))
    log(f"{label}: {len(toks):,} held tokens (interval=max f1440, 1 call each)")
    client = httpx.Client(timeout=30.0)
    n_hit = n_cached = n_fail = n_empty = 0
    for i, tok in enumerate(toks):
        cache = MTM_MID_CACHE / f"leadermax_{tok[:16]}_f1440.parquet"
        if cache.exists():
            n_cached += 1
            continue
        hist, ok = [], False
        for attempt in range(3):
            try:
                rr = client.get("https://clob.polymarket.com/prices-history",
                                params={"market": tok, "interval": "max", "fidelity": 1440})
                rr.raise_for_status()
                hist = rr.json().get("history", [])
                ok = True
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5 * (attempt + 1))
        if not ok:
            n_fail += 1
            continue
        n_empty += 0 if hist else 1
        pd.DataFrame({"t": [float(h["t"]) for h in hist],
                      "p": [float(h["p"]) for h in hist]}).sort_values("t").to_parquet(
            cache, compression="zstd", index=False)
        n_hit += 1
        time.sleep(API_SLEEP_S)
        if (i + 1) % 250 == 0:
            log(f"  {i+1:,}/{len(toks):,} (fetched {n_hit:,}, cached {n_cached:,}, "
                f"legit-empty {n_empty:,}, fail {n_fail:,})")
    client.close()
    log(f"{label} MTM-mid fetch done: fetched {n_hit:,} (legit-empty {n_empty:,}), "
        f"cached {n_cached:,}, transient-fail-left-uncached {n_fail:,}")
    return 0


def cmd_mtm_leader(args: argparse.Namespace) -> int:
    """Lookahead-free daily MTM equity for a leader's copy-audit ledger, two
    curves from the same fills: the LEADER's own book (his fill prices) and the
    pure-taker COPY (Branch B: next-fill price, flat-3c fallback — the same
    branch as the §3.1 re-gate). Positions realize at end_ts with the cached
    closed_positions resolution price; unresolved positions stay open, marked
    to the forward-filled daily mid through the grid end. Marks come from the
    `fetch-mtm-leader` interval=max daily-mid caches."""
    ensure_dirs()
    label = args.leader
    from lib.copy_slippage import mtm_equity_curve_fast

    f = pd.read_parquet(LEADER_FRAGMENTS[label],
                        columns=["fill_ts", "outcome_token_id", "direction", "price",
                                 "token_amount", "B_price", "position_id", "end_ts",
                                 "usd_amount"])
    pos_pq = pd.read_parquet(LEADER_POSITIONS[label],
                             columns=["position_id", "resolution_price"])
    f = f.merge(pos_pq.drop_duplicates("position_id"), on="position_id", how="left")
    f = f.dropna(subset=["outcome_token_id"])
    n_unres = int(f["resolution_price"].isna().sum())
    log(f"{label}: {len(f):,} fills | unresolved-position fills (stay open, marked): "
        f"{n_unres:,} ({n_unres/len(f):.1%})")

    qty = f["token_amount"].astype(float).to_numpy() * np.where(
        f["direction"].str.upper() == "BUY", 1.0, -1.0)
    res_ts = pd.to_datetime(f["end_ts"]).where(f["resolution_price"].notna(), pd.NaT)
    base = pd.DataFrame({
        "entry_ts": pd.to_datetime(f["fill_ts"]),
        "res_ts": res_ts,
        "token": f["outcome_token_id"].astype(str),
        "qty": qty,
        "res_price": f["resolution_price"].astype(float),
    })

    mids: dict[str, list[tuple[float, float]]] = {}
    n_no_quotes = 0
    for tok in base["token"].unique():
        cache = MTM_MID_CACHE / f"leadermax_{tok[:16]}_f1440.parquet"
        if cache.exists():
            h = pd.read_parquet(cache)
            if len(h):
                mids[tok] = list(zip(h["t"].tolist(), h["p"].tolist()))
                continue
        n_no_quotes += 1
    log(f"  daily-mid histories loaded: {len(mids):,} tokens "
        f"({n_no_quotes:,} without quotes -> avg-entry-price mark)")

    rows, curves = [], []
    gross_volume = float(f["usd_amount"].astype(float).sum())
    for ledger, price_col in (("leader_own", "price"), ("copy_taker_B", "B_price")):
        led = base.assign(cash=(-qty * f[price_col].astype(float).to_numpy()))
        res = mtm_equity_curve_fast(led, mids)
        curves.append(res.equity.assign(ledger=ledger))
        rows.append({
            "leader": label, "ledger": ledger, "n_fills": len(led),
            "gross_volume_usd": gross_volume,
            "final_mtm_equity": float(res.equity["mtm_equity"].iloc[-1]),
            "final_realized": float(res.equity["realized"].iloc[-1]),
            "mtm_sharpe_daily_ann": res.sharpe_daily_ann,
            "mtm_max_drawdown_usd": res.max_drawdown_usd,
            "mtm_max_drawdown_pct_of_gross_volume": (res.max_drawdown_usd / gross_volume
                                                     if gross_volume else float("nan")),
            "unresolved_fill_share": n_unres / len(f) if len(f) else 0.0,
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(CSV_OUT / f"spread_surface_phase5_{label}_mtm_summary.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(
        CSV_OUT / f"spread_surface_phase5_{label}_mtm_curves.csv", index=False)
    print(summ.to_string(index=False))
    return 0


def _load_mtm_mids(tokens: set[str], spans: dict) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for tok in tokens:
        s = spans.get(tok)
        if not s:
            continue
        cache = MTM_MID_CACHE / f"mtm_{tok[:16]}_{s[0]}_{s[1]}_f60.parquet"
        if cache.exists():
            h = pd.read_parquet(cache)
            out[tok] = list(zip(h["t"].tolist(), h["p"].tolist()))
    return out


def cmd_mtm(args: argparse.Namespace) -> int:
    ensure_dirs()
    con = duckdb.connect()
    pos = _stage15_positions(con).dropna(subset=["outcome_token_id"])
    pos["outcome_token_id"] = pos["outcome_token_id"].astype(str)
    spans = {}
    for r in pos.drop_duplicates("outcome_token_id").itertuples(index=False):
        ent = pd.to_datetime(r.trade_timestamp, utc=True)
        res = pd.to_datetime(r.resolution_date, utc=True)
        spans[str(r.outcome_token_id)] = (int(ent.timestamp()) - 86400, int(res.timestamp()) + 86400)
    mids = _load_mtm_mids(set(pos["outcome_token_id"]), spans)
    log(f"loaded {len(mids):,} token mid-histories for MTM")

    rows = []
    curves = []
    for cohort, sub in pos.groupby("cohort"):
        res = mtm_equity_curve(sub, mids)
        eq = res.equity.assign(cohort=cohort)
        curves.append(eq)
        # MTM equity is a cumulative-PnL curve (starts ~0), so a peak-relative
        # drawdown fraction is degenerate; report drawdown as % of gross capital
        # deployed (sum of copy notionals), which is always a positive base.
        deployed = float(sub["copy_size_usd"].astype(float).sum())
        rows.append({"cohort": cohort, "n_positions": len(sub),
                     "deployed_capital_usd": deployed,
                     "final_mtm_equity": float(res.equity["mtm_equity"].iloc[-1]) if len(res.equity) else 0.0,
                     "mtm_sharpe_daily_ann": res.sharpe_daily_ann,
                     "mtm_max_drawdown_usd": res.max_drawdown_usd,
                     "mtm_max_drawdown_pct_of_deployed": (res.max_drawdown_usd / deployed
                                                          if deployed else float("nan"))})
    summ = pd.DataFrame(rows)
    summ.to_csv(CSV_OUT / "spread_surface_phase5_mtm_summary.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(
            CSV_OUT / "spread_surface_phase5_mtm_equity_curves.csv", index=False)
    print(summ.to_string(index=False))
    return 0


# ============================================================================
# charts
# ============================================================================
def cmd_charts(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure_dirs()

    cpath = CSV_OUT / "spread_surface_phase5_mtm_equity_curves.csv"
    if cpath.exists():
        cur = pd.read_csv(cpath, parse_dates=["date"])
        fig, ax = plt.subplots(figsize=(8, 5))
        for cohort, sub in cur.groupby("cohort"):
            ax.plot(sub["date"], sub["mtm_equity"], label=cohort, lw=1.4)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("date (daily grid)")
        ax.set_ylabel("MTM equity (USD, realized + unrealized)")
        ax.set_title("Phase-5 lookahead-free MTM equity per cohort\n"
                     "(open positions marked to forward-filled token mid; "
                     "resolution-only PnL = endpoints)")
        ax.legend(fontsize=8)
        p1 = PLOTS_OUT / "spread_surface_phase5_mtm_equity.png"
        fig.tight_layout(); fig.savefig(p1, dpi=150); plt.close(fig)
        print(f"wrote {p1}")

    lpath = CSV_OUT / "spread_surface_phase5_domah_mtm_curves.csv"
    if lpath.exists():
        cur = pd.read_csv(lpath, parse_dates=["date"])
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = {"leader_own": "Domah's own book (his fill prices)",
                  "copy_taker_B": "pure-taker copy (Branch B: next-fill / flat-3c)"}
        for ledger, sub in cur.groupby("ledger"):
            ax.plot(sub["date"], sub["mtm_equity"], label=labels.get(ledger, ledger), lw=1.4)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("date (daily grid)")
        ax.set_ylabel("MTM equity (USD, realized + unrealized)")
        ax.set_title("Domah copy-audit ledger: lookahead-free MTM equity\n"
                     "(open positions marked to forward-filled daily mid; "
                     "resolved at end_date)")
        ax.legend(fontsize=8)
        p3 = PLOTS_OUT / "spread_surface_phase5_domah_mtm_equity.png"
        fig.tight_layout(); fig.savefig(p3, dpi=150); plt.close(fig)
        print(f"wrote {p3}")

    dpath = CSV_OUT / "spread_surface_phase5_domah_drift_by_cat.csv"
    if dpath.exists():
        by = pd.read_csv(dpath)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(by))
        ax.bar(x - 0.2, by["mean_spread_c"], 0.4, label="spread component (predicted half-spread)")
        ax.bar(x + 0.2, by["mean_drift_c"], 0.4, label="drift component (beyond mid+half)")
        ax.set_xticks(x, by["k5_category"], rotation=30, ha="right")
        ax.set_ylabel("mean cents")
        ax.set_title("Domah next-fill copy cost: spread vs drift by category\n"
                     "(drift = how far the copy landed beyond mid + predicted half-spread)")
        ax.legend(fontsize=8); ax.axhline(0, color="k", lw=0.6)
        p2 = PLOTS_OUT / "spread_surface_phase5_domah_drift_spread.png"
        fig.tight_layout(); fig.savefig(p2, dpi=150); plt.close(fig)
        print(f"wrote {p2}")
    return 0


# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_fm = sub.add_parser("fetch-mids")
    p_fm.add_argument("--leader", default="domah", choices=list(LEADER_FRAGMENTS))
    sub.add_parser("fetch-mtm")
    p_fml = sub.add_parser("fetch-mtm-leader")
    p_fml.add_argument("--leader", default="domah", choices=list(LEADER_FRAGMENTS))
    p_ml = sub.add_parser("mtm-leader")
    p_ml.add_argument("--leader", default="domah", choices=list(LEADER_FRAGMENTS))
    p_rg = sub.add_parser("regate")
    p_rg.add_argument("--with-domah-drift", action="store_true",
                      help="also compute the Domah mid-anchored drift/spread split "
                           "(needs fetch-mids --leader domah first)")
    sub.add_parser("mtm")
    sub.add_parser("charts")
    args = ap.parse_args(argv)
    return {"fetch-mids": cmd_fetch_mids, "fetch-mtm": cmd_fetch_mtm,
            "fetch-mtm-leader": cmd_fetch_mtm_leader, "mtm-leader": cmd_mtm_leader,
            "regate": cmd_regate, "mtm": cmd_mtm, "charts": cmd_charts}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
