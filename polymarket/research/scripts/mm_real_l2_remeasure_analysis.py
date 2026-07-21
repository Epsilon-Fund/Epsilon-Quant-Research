#!/usr/bin/env python3
"""Aggregate analyses over the gated real-L2 features: replication, signal, cost floor.

Three pre-registered analyses (design frozen before results were seen; splits below),
all through the ONE owned metric module (``lib.microstructure_metrics``) and
book-measured state (never estimated spread):

**A. Descriptive TOB replication (new markets).** The corrected dali claim
([[pm_prealvaro_pipeline_trust_audit_findings]] Finding 1): at extreme TOB imbalance,
conditional on the mid moving within 5s, it moves toward the imbalance ~63-75%.
Re-measured here on politics_negrisk + esports: per-asset |TOB| q90 thresholds fitted
on TRAIN days, evaluated on TEST days as non-overlapping 5s episodes; hit rate reported
under BOTH zero-move settings plus directional bps and the zero-move share. Also
evaluated at dali's global 0.937422 threshold for continuity.

**B. Feature-signal reading under a clean split (A17's reopen-eligible claim).**
A17's calibration table was condemned (regime confound); the clean re-test design per
the audit is a whole-capture/whole-market split. Here: markets are split by hash into
disjoint TRAIN/TEST halves AND the days are split TRAIN/TEST (doubly disjoint).
Feature-decile thresholds (TOB, OFI-5s, micro_dev normalized by spread) are fitted on
train-markets x train-days; conditional forward-move hit rates + directional bps are
evaluated on test-markets x test-days episodes.

**C. Taker cost floor per category (class-A new-market measurement).** The dali taker
kills rest on a spread cost floor measured on crypto/geopolitics. These are different
markets: measure the executable-touch round-trip floor here — spread in cents and in
bps of mid AT TRADE TIMES (executability-weighted), plus touch depth, per universe and
price bucket. Fees are 0 (captured, documentary), so the floor IS the spread.

Splits (pre-registered):
    TRAIN days: 2026-06-19 .. 2026-07-04     TEST days: 2026-07-05 .. 2026-07-21
    market-hash split: md5(market) even/odd -> train/test halves (analysis B)

CIs: percentile cluster bootstrap over MARKETS (the dali A18 convention), 1000 draws,
seeded. Episodes are non-overlapping (5s) via the shared metric module.

Outputs -> data/analysis/csv_outputs/market_making/real_l2_remeasure/*.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.microstructure_metrics import (  # noqa: E402
    directional_return_bps,
    hit_rate,
    non_overlap_episode_mask,
    zero_move_share,
)

OUT = ROOT / "data" / "analysis" / "csv_outputs" / "market_making" / "real_l2_remeasure"
TRAIN_END = "2026-07-04"
DALI_THRESHOLD = 0.937422
HORIZON_MS = 5_000
BOOT_N = 1000
SEED = 20260721
MIN_TRAIN_ROWS = 500


def _dates(real_l2: Path) -> list[str]:
    return sorted(p.name for p in real_l2.iterdir() if p.is_dir())


def _feature_files(real_l2: Path, universe: str, dates: list[str]) -> list[str]:
    out = []
    for d in dates:
        f = real_l2 / d / universe / "features.parquet"
        if f.exists():
            out.append(str(f))
    return out


def cluster_boot_ci(values: np.ndarray, clusters: np.ndarray, *, stat=np.mean,
                    n_boot: int = BOOT_N, seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap of ``stat`` resampling whole clusters (markets)."""
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return float("nan"), float("nan")
    idx_by_c = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_c[c] for c in draw])
        if len(idx):
            stats.append(stat(values[idx]))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def _episodes(df: pd.DataFrame, signal_col: str, thr_by_asset: dict[str, float] | float,
              fwd_col: str = "fwd_mid_5s") -> pd.DataFrame:
    """Non-overlapping 5s episodes at extreme |signal| with directional-mid returns."""
    sub = df[np.isfinite(df[signal_col]) & np.isfinite(df[fwd_col]) & (df["mid"] > 0)].copy()
    if isinstance(thr_by_asset, dict):
        sub["thr"] = sub["asset_id"].map(thr_by_asset)
        sub = sub[np.isfinite(sub["thr"]) & (sub[signal_col].abs() >= sub["thr"])]
    else:
        sub = sub[sub[signal_col].abs() >= float(thr_by_asset)]
    if sub.empty:
        return sub
    parts = []
    for _, g in sub.groupby("asset_id", sort=False):
        keep = non_overlap_episode_mask(
            g["timestamp_ms"].to_numpy(np.int64) * 1_000_000,  # ms -> ns for the shared rule
            HORIZON_MS * 1_000_000,
            abs_signal=g[signal_col].abs().to_numpy(float),
        )
        parts.append(g[keep])
    eps = pd.concat(parts, ignore_index=True)
    eps["ret_bps"] = (eps[fwd_col] - eps["mid"]) / eps["mid"] * 10_000.0
    return eps


def _metric_row(eps: pd.DataFrame, signal_col: str, label: dict) -> dict:
    sig = eps[signal_col].to_numpy(float)
    ret = eps["ret_bps"].to_numpy(float)
    hr_miss, n_miss = hit_rate(sig, ret, zero_move="miss")
    hr_cond, n_cond = hit_rate(sig, ret, zero_move="exclude")
    dr, _ = directional_return_bps(sig, ret)
    signed = np.sign(sig) * ret
    clusters = eps["market"].to_numpy(str)
    lo, hi = cluster_boot_ci(signed, clusters)
    hit_vals = (np.sign(sig) * ret > 0).astype(float)
    hlo, hhi = cluster_boot_ci(hit_vals, clusters)
    return {
        **label,
        "episodes": len(eps),
        "markets": eps["market"].nunique(),
        "hit_zero_as_miss": hr_miss,
        "hit_zero_as_miss_ci_lo": hlo, "hit_zero_as_miss_ci_hi": hhi,
        "hit_conditional": hr_cond, "n_conditional": n_cond,
        "zero_move_share": zero_move_share(ret),
        "directional_return_bps": dr,
        "directional_return_bps_ci_lo": lo, "directional_return_bps_ci_hi": hi,
    }


def analysis_a_replication(real_l2: Path, universes: list[str]) -> pd.DataFrame:
    rows = []
    for uni in universes:
        dates = _dates(real_l2)
        train_f = _feature_files(real_l2, uni, [d for d in dates if d <= TRAIN_END])
        test_f = _feature_files(real_l2, uni, [d for d in dates if d > TRAIN_END])
        if not train_f or not test_f:
            continue
        con = duckdb.connect()
        thr_df = con.execute(
            "SELECT asset_id, quantile_cont(abs(tob_imbalance), 0.90) q90, count(*) n "
            "FROM read_parquet($f) WHERE isfinite(tob_imbalance) GROUP BY 1",
            {"f": train_f},
        ).df()
        thr = {str(r.asset_id): float(r.q90) for r in thr_df.itertuples()
               if r.n >= MIN_TRAIN_ROWS and np.isfinite(r.q90) and r.q90 > 0}
        test = con.execute(
            "SELECT timestamp_ms, asset_id, market, mid, tob_imbalance, fwd_mid_5s "
            "FROM read_parquet($f)", {"f": test_f},
        ).df()
        con.close()
        eps = _episodes(test, "tob_imbalance", thr)
        if len(eps):
            rows.append(_metric_row(eps, "tob_imbalance",
                                    {"universe": uni, "threshold": "per_asset_q90_train"}))
        eps_dali = _episodes(test, "tob_imbalance", DALI_THRESHOLD)
        if len(eps_dali):
            rows.append(_metric_row(eps_dali, "tob_imbalance",
                                    {"universe": uni, "threshold": f"dali_{DALI_THRESHOLD}"}))
    return pd.DataFrame(rows)


def analysis_b_clean_split(real_l2: Path, universes: list[str]) -> pd.DataFrame:
    """A17-flavor feature-signal reading: doubly-disjoint (markets x days) split."""
    rows = []
    feat_cols = {"tob_imbalance": "tob_imbalance", "ofi_5s": "ofi_5s",
                 "micro_dev_over_spread": "micro_dev / nullif(spread, 0)"}
    for uni in universes:
        dates = _dates(real_l2)
        train_f = _feature_files(real_l2, uni, [d for d in dates if d <= TRAIN_END])
        test_f = _feature_files(real_l2, uni, [d for d in dates if d > TRAIN_END])
        if not train_f or not test_f:
            continue
        con = duckdb.connect()
        sel = ", ".join(f"{expr} AS {name}" for name, expr in feat_cols.items())
        train = con.execute(
            f"SELECT asset_id, market, {sel} FROM read_parquet($f)", {"f": train_f}).df()
        test = con.execute(
            f"SELECT timestamp_ms, asset_id, market, mid, fwd_mid_5s, {sel} "
            "FROM read_parquet($f)", {"f": test_f}).df()
        con.close()

        def _half(m: str) -> int:
            return int(hashlib.md5(str(m).encode()).hexdigest(), 16) % 2

        train = train[train["market"].map(_half) == 0]        # train half
        test = test[test["market"].map(_half) == 1]           # disjoint test half
        for feat in feat_cols:
            vals = train[feat].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) < MIN_TRAIN_ROWS:
                continue
            q90 = float(vals.abs().quantile(0.90))
            if not np.isfinite(q90) or q90 <= 0:
                continue
            eps = _episodes(test, feat, q90)
            if len(eps) < 50:
                continue
            rows.append(_metric_row(eps, feat, {
                "universe": uni, "feature": feat,
                "threshold": "global_q90_train_markets_train_days", "q90": q90,
            }))
    return pd.DataFrame(rows)


def analysis_c_cost_floor(real_l2: Path, raw: Path, universes: list[str]) -> pd.DataFrame:
    """Executable-touch spread/depth AT TRADE TIMES, per universe x price bucket."""
    rows = []
    for uni in universes:
        dates = _dates(real_l2)
        files = _feature_files(real_l2, uni, dates)
        trade_files = [str(raw / d / uni / "trades_*.parquet") for d in dates
                       if (raw / d / uni).exists()]
        if not files or not trade_files:
            continue
        con = duckdb.connect()
        # spread/depth as-of each trade: join trades to the last fresh L1 state <= trade ts
        q = """
        WITH t AS (
            SELECT timestamp_ms AS tts, asset_id, price
            FROM read_parquet($t)
        ), s AS (
            SELECT timestamp_ms, asset_id, spread, mid, touch_depth
            FROM read_parquet($f)
        )
        SELECT t.asset_id, t.price, s.spread, s.mid, s.touch_depth
        FROM t ASOF JOIN s
          ON t.asset_id = s.asset_id AND s.timestamp_ms <= t.tts
        WHERE t.tts - s.timestamp_ms <= 5000
        """
        df = con.execute(q, {"t": trade_files, "f": files}).df()
        con.close()
        df = df[np.isfinite(df["spread"]) & np.isfinite(df["mid"]) & (df["mid"] > 0)]
        df["price_bucket"] = pd.cut(df["price"], [0, 0.05, 0.2, 0.8, 0.95, 1.0],
                                    labels=["p<5c", "5-20c", "20-80c", "80-95c", "p>95c"])
        df["spread_bps_mid"] = df["spread"] / df["mid"] * 10_000.0
        for bucket, g in df.groupby("price_bucket", observed=True):
            if len(g) < 100:
                continue
            rows.append({
                "universe": uni, "price_bucket": str(bucket), "trades_costed": len(g),
                "median_spread_cents": float(g["spread"].median() * 100),
                "mean_spread_cents": float(g["spread"].mean() * 100),
                "p90_spread_cents": float(g["spread"].quantile(0.9) * 100),
                "median_spread_bps_of_mid": float(g["spread_bps_mid"].median()),
                "median_touch_depth": float(g["touch_depth"].median()),
                "roundtrip_taker_floor_cents": float(g["spread"].median() * 100),  # fee=0 captured
            })
        # coverage stat: how many trades got a fresh book within 5s
        rows.append({"universe": uni, "price_bucket": "ALL", "trades_costed": len(df),
                     "median_spread_cents": float(df["spread"].median() * 100),
                     "mean_spread_cents": float(df["spread"].mean() * 100),
                     "p90_spread_cents": float(df["spread"].quantile(0.9) * 100),
                     "median_spread_bps_of_mid": float(df["spread_bps_mid"].median()),
                     "median_touch_depth": float(df["touch_depth"].median()),
                     "roundtrip_taker_floor_cents": float(df["spread"].median() * 100)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-l2-dir", type=Path, default=ROOT / "data" / "analysis" / "real_l2")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "l2_parquet_full")
    ap.add_argument("--universes", nargs="*", default=["politics_negrisk", "esports"])
    ap.add_argument("--only", choices=["a", "b", "c"], default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "a"):
        a = analysis_a_replication(args.real_l2_dir, args.universes)
        a.to_csv(OUT / "tob_descriptive_replication.csv", index=False)
        print(a.to_string(), flush=True)
    if args.only in (None, "b"):
        b = analysis_b_clean_split(args.real_l2_dir, args.universes)
        b.to_csv(OUT / "feature_signal_clean_split.csv", index=False)
        print(b.to_string(), flush=True)
    if args.only in (None, "c"):
        c = analysis_c_cost_floor(args.real_l2_dir, args.raw_dir, args.universes)
        c.to_csv(OUT / "taker_cost_floor_new_markets.csv", index=False)
        print(c.to_string(), flush=True)


if __name__ == "__main__":
    main()
