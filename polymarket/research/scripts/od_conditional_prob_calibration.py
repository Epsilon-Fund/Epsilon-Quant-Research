"""OD reopen-or-close: Binance-only conditional resolution calibration.

Builds a historical 4h digital-resolution panel from Binance 5m klines and
uses it to estimate P(resolve UP | signed z bucket, time-left bucket). The
empirical conditional probability is then applied to the OD v4 far-|z|
strict-rich short set.
"""
from __future__ import annotations

import io
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import YEAR_SECONDS, cents, markdown_table, norm_cdf, number, pct
from od_strategy_a_v3 import normalize_markdown_wrapping, resolve_token_rv_physical_prob_fair


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
BRAIN_TODO = REPO / "brain" / "TODO.md"
OD_HUB = NOTES / "options_delta" / "strat_options_delta.md"

BINANCE_CACHE = ROOT / "data" / "external" / "binance_5m"
ZIP_CACHE = BINANCE_CACHE / "monthly_zips"
CSV_OUT = ANALYSIS / "csv_outputs" / "options_delta"
PLOTS = ANALYSIS / "plots" / "options_delta"

V4_CONTRACTS = ANALYSIS / "od_v4_calibration_gate_contracts.parquet"
V4_QUEUE_SUMMARY = CSV_OUT / "od_v4_queue_replay_summary.csv"

OUT_SUMMARY = CSV_OUT / "od_conditional_prob_calibration_summary.csv"
OUT_BINS = CSV_OUT / "od_conditional_prob_calibration_bins.csv"
OUT_PM = ANALYSIS / "od_conditional_prob_pm_fills.parquet"
OUT_HIST = ANALYSIS / "od_conditional_prob_binance_history.parquet"
OUT_CV = ANALYSIS / "od_conditional_prob_binance_cv.parquet"

NOTE = NOTES / "options_delta" / "od_conditional_prob_calibration_findings.md"
CALIBRATION_PLOT = PLOTS / "od_conditional_prob_arm_b_calibration.png"
EV_PLOT = PLOTS / "od_conditional_prob_pm_ev.png"

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
START_MONTH = "2021-01"
END_MONTH = "2026-05"
TRAIN_CUTOFF = pd.Timestamp("2026-05-27 00:00:00", tz="UTC")
BAR_SECONDS = 300.0
EWMA_HALFLIFE_SECONDS = 1800.0
EWMA_HALFLIFE_BARS = EWMA_HALFLIFE_SECONDS / BAR_SECONDS
MIN_EWMA_BARS = 12
BOOTSTRAP_SAMPLES = 5000
RNG_SEED = 20260602
MIN_BIN_COUNT = 200
NON_TOP3_AVAILABLE_SHARE = 0.05

SIGNED_Z_BINS = [-np.inf, -2.0, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, np.inf]
SIGNED_Z_LABELS = [
    "z_lt_-2",
    "z_-2_-1p5",
    "z_-1p5_-1p25",
    "z_-1p25_-1",
    "z_-1_-0p75",
    "z_-0p75_-0p5",
    "z_-0p5_-0p25",
    "z_-0p25_0",
    "z_0_0p25",
    "z_0p25_0p5",
    "z_0p5_0p75",
    "z_0p75_1",
    "z_1_1p25",
    "z_1p25_1p5",
    "z_1p5_2",
    "z_gt_2",
]
COARSE_Z_BINS = [-np.inf, -1.0, 0.0, 1.0, np.inf]
COARSE_Z_LABELS = ["z_lt_-1", "z_-1_0", "z_0_1", "z_gt_1"]


@dataclass(frozen=True)
class LookupResult:
    p_up: float
    train_n: int
    source: str


def fmt_ci(lo: float, hi: float, unit: str = "c") -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "[n/a, n/a]"
    if unit == "pct":
        return f"[{pct(lo)}, {pct(hi)}]"
    return f"[{cents(lo)}, {cents(hi)}]"


def month_range(start: str, end: str) -> list[pd.Period]:
    return list(pd.period_range(pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M"))


def download_month(symbol: str, period: pd.Period, client: httpx.Client) -> Path | None:
    ZIP_CACHE.mkdir(parents=True, exist_ok=True)
    filename = f"{symbol}-5m-{period}.zip"
    path = ZIP_CACHE / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/5m/{filename}"
    for attempt in range(4):
        try:
            r = client.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            path.write_bytes(r.content)
            time.sleep(0.03)
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))
    return None


def parse_month_zip(path: Path) -> pd.DataFrame:
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "n_trades",
        "taker_base_volume",
        "taker_quote_volume",
        "ignore",
    ]
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
        if not members:
            return pd.DataFrame(columns=cols)
        with zf.open(members[0]) as fh:
            raw = fh.read()
    df = pd.read_csv(io.BytesIO(raw), header=None, names=cols)
    # Some newer Binance CSVs include a header row. Coerce and drop it.
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df[df["open_time"].notna()].copy()
    for col in ("open", "high", "low", "close", "volume", "close_time"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open_time", "open", "close", "close_time"])
    df["bar_open_ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["ts"] = pd.to_datetime(df["close_time"].astype("int64") + 1, unit="ms", utc=True)
    return df[["bar_open_ts", "ts", "open", "high", "low", "close", "volume"]].sort_values("bar_open_ts")


def load_symbol_klines(asset: str, symbol: str, refresh: bool = False) -> pd.DataFrame:
    out_path = BINANCE_CACHE / f"{symbol}_5m_{START_MONTH}_{END_MONTH}.parquet"
    if out_path.exists() and not refresh:
        df = pd.read_parquet(out_path)
        df["bar_open_ts"] = pd.to_datetime(df["bar_open_ts"], utc=True)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    pieces: list[pd.DataFrame] = []
    with httpx.Client(headers={"User-Agent": "epsilon-quant-research-od-conditional/1.0"}) as client:
        for period in month_range(START_MONTH, END_MONTH):
            z = download_month(symbol, period, client)
            if z is None:
                continue
            pieces.append(parse_month_zip(z))
            if len(pieces) % 12 == 0:
                print(f"{symbol}: parsed {len(pieces)} monthly files", flush=True)
    if not pieces:
        raise SystemExit(f"no Binance monthly data found for {symbol}")
    df = pd.concat(pieces, ignore_index=True).drop_duplicates("bar_open_ts").sort_values("bar_open_ts")
    df["asset"] = asset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"cached {symbol} rows={len(df):,} -> {out_path}", flush=True)
    return df


def add_time_bucket(tau_seconds: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [tau_seconds.le(1800), tau_seconds.le(7200)],
            ["late_lt30m", "mid_30m_2h"],
            default="early_gt2h",
        ),
        index=tau_seconds.index,
    )


def add_history_features(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    out = df.copy().sort_values("bar_open_ts").reset_index(drop=True)
    ret = np.log(out["close"].astype(float)).diff()
    out["spot_log_return"] = ret
    var_ewm = ret.pow(2).ewm(halflife=EWMA_HALFLIFE_BARS, adjust=False, min_periods=MIN_EWMA_BARS).mean()
    out["ewma_sigma_annualized"] = np.sqrt(var_ewm * YEAR_SECONDS / BAR_SECONDS)
    out["window_start"] = out["bar_open_ts"].dt.floor("4h")
    window = (
        out.groupby("window_start", as_index=False)
        .agg(
            bars=("close", "count"),
            strike=("open", "first"),
            close_spot=("close", "last"),
            window_end=("bar_open_ts", lambda s: s.iloc[0] + pd.Timedelta(hours=4)),
        )
    )
    out = out.merge(window, on="window_start", how="left")
    out = out[out["bars"].ge(40)].copy()
    out["tau_seconds"] = (out["window_end"] - out["ts"]).dt.total_seconds()
    out = out[out["tau_seconds"].gt(0)].copy()
    out["tau_years"] = out["tau_seconds"] / YEAR_SECONDS
    out["binance_resolution_up"] = out["close_spot"].astype(float).gt(out["strike"].astype(float)).astype(float)
    out["log_spot_moneyness"] = np.log(out["close"].astype(float) / out["strike"].astype(float))
    denom = out["ewma_sigma_annualized"].astype(float) * np.sqrt(out["tau_years"].astype(float))
    out["digital_z"] = out["log_spot_moneyness"] / denom.replace(0.0, np.nan)
    out["p_model"] = norm_cdf(out["digital_z"].to_numpy(dtype=float))
    out["abs_z"] = out["digital_z"].abs()
    out["time_bucket"] = add_time_bucket(out["tau_seconds"].astype(float))
    out["signed_z_bucket"] = pd.cut(out["digital_z"], SIGNED_Z_BINS, labels=SIGNED_Z_LABELS, include_lowest=True)
    out["coarse_z_bucket"] = pd.cut(out["digital_z"], COARSE_Z_BINS, labels=COARSE_Z_LABELS, include_lowest=True)
    out["asset"] = asset
    keep = [
        "asset",
        "bar_open_ts",
        "ts",
        "window_start",
        "window_end",
        "strike",
        "close",
        "close_spot",
        "tau_seconds",
        "tau_years",
        "ewma_sigma_annualized",
        "digital_z",
        "abs_z",
        "p_model",
        "time_bucket",
        "signed_z_bucket",
        "coarse_z_bucket",
        "binance_resolution_up",
    ]
    return out[keep].replace([np.inf, -np.inf], np.nan).dropna(subset=["digital_z", "p_model", "signed_z_bucket", "time_bucket"])


def build_history(refresh: bool = False) -> pd.DataFrame:
    if OUT_HIST.exists() and not refresh:
        hist = pd.read_parquet(OUT_HIST)
        for col in ("bar_open_ts", "ts", "window_start", "window_end"):
            hist[col] = pd.to_datetime(hist[col], utc=True)
        return hist
    pieces = []
    for asset, symbol in SYMBOLS.items():
        raw = load_symbol_klines(asset, symbol, refresh=refresh)
        feat = add_history_features(raw, asset)
        print(f"history {asset}: rows={len(feat):,} windows={feat['window_start'].nunique():,}", flush=True)
        pieces.append(feat)
    hist = pd.concat(pieces, ignore_index=True).sort_values(["asset", "ts"]).reset_index(drop=True)
    OUT_HIST.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(OUT_HIST, index=False)
    return hist


def table_from_train(train: pd.DataFrame) -> dict[str, Any]:
    primary = (
        train.groupby(["signed_z_bucket", "time_bucket"], observed=True)
        .agg(p_up=("binance_resolution_up", "mean"), n=("binance_resolution_up", "count"))
        .reset_index()
    )
    coarse = (
        train.groupby(["coarse_z_bucket", "time_bucket"], observed=True)
        .agg(p_up=("binance_resolution_up", "mean"), n=("binance_resolution_up", "count"))
        .reset_index()
    )
    time_only = train.groupby("time_bucket", observed=True).agg(p_up=("binance_resolution_up", "mean"), n=("binance_resolution_up", "count")).reset_index()
    return {
        "primary": {(str(r.signed_z_bucket), str(r.time_bucket)): (float(r.p_up), int(r.n)) for _, r in primary.iterrows()},
        "coarse": {(str(r.coarse_z_bucket), str(r.time_bucket)): (float(r.p_up), int(r.n)) for _, r in coarse.iterrows()},
        "time": {str(r.time_bucket): (float(r.p_up), int(r.n)) for _, r in time_only.iterrows()},
        "overall": (float(train["binance_resolution_up"].mean()), int(len(train))),
    }


def lookup_prob(row: pd.Series, lookup: dict[str, Any]) -> LookupResult:
    key = (str(row["signed_z_bucket"]), str(row["time_bucket"]))
    val = lookup["primary"].get(key)
    if val and val[1] >= MIN_BIN_COUNT:
        return LookupResult(val[0], val[1], "signed_z_time")
    ckey = (str(row["coarse_z_bucket"]), str(row["time_bucket"]))
    val = lookup["coarse"].get(ckey)
    if val and val[1] >= MIN_BIN_COUNT:
        return LookupResult(val[0], val[1], "coarse_z_time")
    val = lookup["time"].get(str(row["time_bucket"]))
    if val and val[1] > 0:
        return LookupResult(val[0], val[1], "time_only")
    val = lookup["overall"]
    return LookupResult(val[0], val[1], "overall")


def expanding_year_cv(hist: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    hist = hist[hist["ts"].lt(TRAIN_CUTOFF)].copy()
    hist["year"] = hist["window_start"].dt.year
    for year in sorted(y for y in hist["year"].unique() if y >= 2022):
        train = hist[hist["year"].lt(year)].copy()
        val = hist[hist["year"].eq(year)].copy()
        if train.empty or val.empty:
            continue
        lookup = table_from_train(train)
        preds = [lookup_prob(row, lookup) for _, row in val.iterrows()]
        val["arm_b_p_up"] = [p.p_up for p in preds]
        val["arm_b_train_n"] = [p.train_n for p in preds]
        val["arm_b_source"] = [p.source for p in preds]
        val["cv_year"] = year
        rows.append(val)
        print(f"cv year={year}: train_rows={len(train):,} val_rows={len(val):,}", flush=True)
    if not rows:
        return pd.DataFrame()
    cv = pd.concat(rows, ignore_index=True)
    return cv


def add_pm_bins(pm: pd.DataFrame) -> pd.DataFrame:
    out = pm.copy()
    tau_years = out["tau_years"].astype(float).replace(0.0, np.nan)
    denom = out["ewma_sigma_annualized"].astype(float).replace(0.0, np.nan) * np.sqrt(tau_years)
    out["digital_z_signed"] = out["log_spot_moneyness"].astype(float) / denom
    out["signed_z_bucket"] = pd.cut(out["digital_z_signed"], SIGNED_Z_BINS, labels=SIGNED_Z_LABELS, include_lowest=True)
    out["coarse_z_bucket"] = pd.cut(out["digital_z_signed"], COARSE_Z_BINS, labels=COARSE_Z_LABELS, include_lowest=True)
    out["short_price"] = out["entry_price"].astype(float)
    out["realized_itm"] = out["payoff"].astype(float)
    return out


def apply_lookup_to_pm(pm: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    train = hist[hist["ts"].lt(TRAIN_CUTOFF)].copy()
    lookup = table_from_train(train)
    out = add_pm_bins(pm)
    preds = [lookup_prob(row, lookup) for _, row in out.iterrows()]
    out["arm_b_p_up"] = [p.p_up for p in preds]
    out["arm_b_train_n"] = [p.train_n for p in preds]
    out["arm_b_source"] = [p.source for p in preds]
    arm_a_prob = resolve_token_rv_physical_prob_fair(out, context="od_conditional_prob_calibration.apply_lookup_to_pm")
    out["arm_a_rv_physical_prob"] = arm_a_prob.astype(float)
    out["arm_a_token_prob"] = out["arm_a_rv_physical_prob"]  # Legacy alias.
    out["arm_a_fair_kind"] = "rv_physical_prob"
    out["arm_b_token_prob"] = np.where(out["actual_outcome"].astype(str).eq("up"), out["arm_b_p_up"], 1.0 - out["arm_b_p_up"])
    out["arm_a_edge"] = out["short_price"] - out["arm_a_token_prob"]
    out["arm_b_edge"] = out["short_price"] - out["arm_b_token_prob"]
    out["net_realized_ev"] = out["short_price"] - out["realized_itm"] + out["maker_rebate"].astype(float)
    out["gross_realized_ev"] = out["short_price"] - out["realized_itm"]
    return out


def cluster_ci(df: pd.DataFrame, col: str, seed_offset: int = 0) -> tuple[float, float]:
    if df.empty:
        return math.nan, math.nan
    groups = []
    for _, g in df.groupby("market_id", sort=False):
        vals = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(vals):
            groups.append((float(vals.sum()), int(len(vals))))
    if not groups:
        return math.nan, math.nan
    if len(groups) == 1:
        s, n = groups[0]
        v = s / n
        return float(v), float(v)
    rng = np.random.default_rng(RNG_SEED + seed_offset + len(groups))
    sums = np.asarray([g[0] for g in groups], dtype=float)
    counts = np.asarray([g[1] for g in groups], dtype=float)
    idx = rng.integers(0, len(groups), size=(BOOTSTRAP_SAMPLES, len(groups)))
    vals = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def market_sum_ci(df: pd.DataFrame, col: str, seed_offset: int = 0) -> tuple[float, float]:
    if df.empty:
        return math.nan, math.nan
    vals = df.groupby("market_id", sort=False)[col].sum().replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(RNG_SEED + seed_offset + 17 * len(vals))
    idx = rng.integers(0, len(vals), size=(BOOTSTRAP_SAMPLES, len(vals)))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def structural_baseline() -> dict[str, float]:
    if not V4_QUEUE_SUMMARY.exists():
        return {"mean": math.nan, "ci_lo": math.nan, "ci_hi": math.nan}
    s = pd.read_csv(V4_QUEUE_SUMMARY)
    sub = s[(s["sample_split"].eq("oos_holdout")) & (s["metric_scope"].eq("queue_adjusted_after_top3_haircut"))].copy()
    sub = sub.sort_values(["net_ci_lo", "mean_net_pnl_per_market"], ascending=[False, False])
    if sub.empty:
        return {"mean": math.nan, "ci_lo": math.nan, "ci_hi": math.nan}
    r = sub.iloc[0]
    return {"mean": float(r["mean_net_pnl_per_market"]), "ci_lo": float(r["net_ci_lo"]), "ci_hi": float(r["net_ci_hi"])}


def summarize_pm(pm: pd.DataFrame, baseline: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    configs = [
        ("arm_a_ewma_nz_original_set", pm, "arm_a_token_prob", "arm_a_edge"),
        ("arm_b_empirical_conditional_original_set", pm, "arm_b_token_prob", "arm_b_edge"),
        ("arm_b_empirical_conditional_rich_ge_1c", pm[pm["arm_b_edge"].ge(0.01)].copy(), "arm_b_token_prob", "arm_b_edge"),
        ("arm_b_empirical_conditional_rich_ge_5c", pm[pm["arm_b_edge"].ge(0.05)].copy(), "arm_b_token_prob", "arm_b_edge"),
    ]
    for label, sub, prob_col, edge_col in configs:
        if sub.empty:
            rows.append({"label": label, "fills": 0, "markets": 0})
            continue
        edge_lo, edge_hi = cluster_ci(sub, edge_col, seed_offset=10)
        net_lo, net_hi = cluster_ci(sub, "net_realized_ev", seed_offset=20)
        gross_lo, gross_hi = cluster_ci(sub, "gross_realized_ev", seed_offset=30)
        after = sub.copy()
        after["net_after_top3"] = after["net_realized_ev"] * NON_TOP3_AVAILABLE_SHARE
        top3_lo, top3_hi = market_sum_ci(after, "net_after_top3", seed_offset=40)
        mean_after_top3 = float(after.groupby("market_id")["net_after_top3"].sum().mean())
        baseline_mean = baseline["mean"]
        rows.append(
            {
                "label": label,
                "fills": int(len(sub)),
                "markets": int(sub["market_id"].nunique()),
                "mean_short_price": float(sub["short_price"].mean()),
                "realized_itm_rate": float(sub["realized_itm"].mean()),
                "mean_pred_itm_prob": float(sub[prob_col].mean()),
                "calibration_gap_obs_minus_pred": float(sub["realized_itm"].mean() - sub[prob_col].mean()),
                "mean_model_edge": float(sub[edge_col].mean()),
                "model_edge_ci_lo": edge_lo,
                "model_edge_ci_hi": edge_hi,
                "mean_gross_realized_ev": float(sub["gross_realized_ev"].mean()),
                "gross_realized_ev_ci_lo": gross_lo,
                "gross_realized_ev_ci_hi": gross_hi,
                "mean_net_realized_ev": float(sub["net_realized_ev"].mean()),
                "net_realized_ev_ci_lo": net_lo,
                "net_realized_ev_ci_hi": net_hi,
                "mean_market_net_after_top3": mean_after_top3,
                "market_net_after_top3_ci_lo": top3_lo,
                "market_net_after_top3_ci_hi": top3_hi,
                "zero_baseline_mean": 0.0,
                "incremental_after_top3_vs_zero": mean_after_top3,
                "incremental_after_top3_vs_zero_ci_lo": top3_lo,
                "incremental_after_top3_vs_zero_ci_hi": top3_hi,
                "borrowed_structural_baseline_mean": baseline_mean,
                "incremental_after_top3_vs_borrowed_structural": mean_after_top3 - baseline_mean if np.isfinite(baseline_mean) else math.nan,
                "incremental_after_top3_vs_borrowed_structural_ci_lo": top3_lo - baseline_mean if np.isfinite(baseline_mean) and np.isfinite(top3_lo) else math.nan,
                "incremental_after_top3_vs_borrowed_structural_ci_hi": top3_hi - baseline_mean if np.isfinite(baseline_mean) and np.isfinite(top3_hi) else math.nan,
                "live_measured_baseline_mean": math.nan,
                "incremental_after_top3_vs_live_measured": math.nan,
                "incremental_after_top3_vs_live_measured_ci_lo": math.nan,
                "incremental_after_top3_vs_live_measured_ci_hi": math.nan,
                "structural_baseline_mean": baseline_mean,
                "incremental_after_top3_vs_structural": mean_after_top3 - baseline_mean if np.isfinite(baseline_mean) else math.nan,
                "incremental_after_top3_ci_lo": top3_lo - baseline_mean if np.isfinite(baseline_mean) and np.isfinite(top3_lo) else math.nan,
                "incremental_after_top3_ci_hi": top3_hi - baseline_mean if np.isfinite(baseline_mean) and np.isfinite(top3_hi) else math.nan,
                "mean_arm_b_train_n": float(sub["arm_b_train_n"].mean()) if "arm_b_train_n" in sub else math.nan,
                "primary_lookup_share": float(sub["arm_b_source"].eq("signed_z_time").mean()) if "arm_b_source" in sub else math.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_bins_cv(cv: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for arm, col in [("arm_a_ewma_nz", "p_model"), ("arm_b_empirical_conditional", "arm_b_p_up")]:
        out = cv[["asset", "ts", "cv_year", "time_bucket", "signed_z_bucket", "binance_resolution_up", col]].copy()
        out = out.rename(columns={col: "pred_prob"})
        out["arm"] = arm
        out["prob_bin"] = pd.cut(
            out["pred_prob"],
            bins=[0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0],
            labels=["0_5c", "5_10c", "10_25c", "25_50c", "50_75c", "75_90c", "90_95c", "95_100c"],
            include_lowest=True,
        )
        pieces.append(out)
    both = pd.concat(pieces, ignore_index=True)
    rows = []
    for (arm, bucket), g in both.groupby(["arm", "prob_bin"], observed=True):
        rows.append(
            {
                "row_type": "cv_reliability",
                "arm": arm,
                "bucket": str(bucket),
                "rows": int(len(g)),
                "mean_pred": float(g["pred_prob"].mean()),
                "observed_freq": float(g["binance_resolution_up"].mean()),
                "obs_minus_pred": float(g["binance_resolution_up"].mean() - g["pred_prob"].mean()),
            }
        )
    return pd.DataFrame(rows)


def pm_bins(pm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, prob_col, edge_col in [
        ("arm_a_ewma_nz", "arm_a_token_prob", "arm_a_edge"),
        ("arm_b_empirical_conditional", "arm_b_token_prob", "arm_b_edge"),
    ]:
        out = pm.copy()
        out["prob_bin"] = pd.cut(
            out[prob_col],
            bins=[0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0],
            labels=["0_5c", "5_10c", "10_25c", "25_50c", "50_75c", "75_90c", "90_95c", "95_100c"],
            include_lowest=True,
        )
        for bucket, g in out.groupby("prob_bin", observed=True):
            rows.append(
                {
                    "row_type": "pm_v4_fill_reliability",
                    "arm": arm,
                    "bucket": str(bucket),
                    "rows": int(len(g)),
                    "markets": int(g["market_id"].nunique()),
                    "mean_pred": float(g[prob_col].mean()),
                    "observed_freq": float(g["realized_itm"].mean()),
                    "obs_minus_pred": float(g["realized_itm"].mean() - g[prob_col].mean()),
                    "mean_short_price": float(g["short_price"].mean()),
                    "mean_edge": float(g[edge_col].mean()),
                    "mean_net_realized_ev": float(g["net_realized_ev"].mean()),
                }
            )
    return pd.DataFrame(rows)


def make_plots(cv_bins: pd.DataFrame, pm_summary: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = cv_bins[cv_bins["arm"].eq("arm_b_empirical_conditional")].copy()
    if not sub.empty:
        sizes = np.maximum(1, sub["rows"].astype(float) / sub["rows"].max()) * 350.0
        ax.scatter(sub["mean_pred"], sub["observed_freq"], s=sizes, color="#2c7fb8", alpha=0.75)
        for _, r in sub.iterrows():
            ax.annotate(str(r["bucket"]), (r["mean_pred"], r["observed_freq"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#333333", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Arm B empirical predicted P(resolve UP)")
    ax.set_ylabel("Binance historical observed UP frequency")
    ax.set_title("Arm B expanding-CV reliability on Binance 4h windows")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(CALIBRATION_PLOT, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot = pm_summary[pm_summary["fills"].fillna(0).astype(int).gt(0)].copy()
    x = np.arange(len(plot))
    y = plot["mean_net_realized_ev"].to_numpy(dtype=float)
    yerr = np.vstack([y - plot["net_realized_ev_ci_lo"].to_numpy(dtype=float), plot["net_realized_ev_ci_hi"].to_numpy(dtype=float) - y])
    ax.bar(x, y, color="#41ab5d")
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#222222", capsize=3)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot["label"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean net realized EV per fill ($)")
    ax.set_title("PM v4 far-|z| short set: realized EV by probability arm")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EV_PLOT, dpi=160)
    plt.close(fig)


def format_table(df: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    piece = df.copy()
    if limit is not None:
        piece = piece.head(limit)
    rows = []
    for _, r in piece.iterrows():
        row = []
        for col in cols:
            v = r.get(col, math.nan)
            if col in {"fills", "markets", "rows"}:
                row.append(str(int(v)) if pd.notna(v) else "0")
            elif col in {
                "mean_short_price",
                "mean_model_edge",
                "mean_gross_realized_ev",
                "mean_net_realized_ev",
                "mean_market_net_after_top3",
                "incremental_after_top3_vs_structural",
                "structural_baseline_mean",
                "zero_baseline_mean",
                "incremental_after_top3_vs_zero",
                "borrowed_structural_baseline_mean",
                "incremental_after_top3_vs_borrowed_structural",
                "live_measured_baseline_mean",
                "incremental_after_top3_vs_live_measured",
                "mean_edge",
            }:
                row.append(cents(float(v)))
            elif col in {"model_edge_ci_lo"}:
                row.append(fmt_ci(float(r["model_edge_ci_lo"]), float(r["model_edge_ci_hi"])))
            elif col in {"net_realized_ev_ci_lo"}:
                row.append(fmt_ci(float(r["net_realized_ev_ci_lo"]), float(r["net_realized_ev_ci_hi"])))
            elif col in {"market_net_after_top3_ci_lo"}:
                row.append(fmt_ci(float(r["market_net_after_top3_ci_lo"]), float(r["market_net_after_top3_ci_hi"])))
            elif col in {"incremental_after_top3_ci_lo"}:
                row.append(fmt_ci(float(r["incremental_after_top3_ci_lo"]), float(r["incremental_after_top3_ci_hi"])))
            elif col in {"incremental_after_top3_vs_zero_ci_lo"}:
                row.append(fmt_ci(float(r["incremental_after_top3_vs_zero_ci_lo"]), float(r["incremental_after_top3_vs_zero_ci_hi"])))
            elif col in {"incremental_after_top3_vs_borrowed_structural_ci_lo"}:
                row.append(fmt_ci(float(r["incremental_after_top3_vs_borrowed_structural_ci_lo"]), float(r["incremental_after_top3_vs_borrowed_structural_ci_hi"])))
            elif col in {"incremental_after_top3_vs_live_measured_ci_lo"}:
                row.append(fmt_ci(float(r["incremental_after_top3_vs_live_measured_ci_lo"]), float(r["incremental_after_top3_vs_live_measured_ci_hi"])))
            elif col in {"realized_itm_rate", "mean_pred_itm_prob", "calibration_gap_obs_minus_pred", "mean_pred", "observed_freq", "obs_minus_pred", "primary_lookup_share"}:
                row.append(pct(float(v)))
            elif isinstance(v, float):
                row.append(number(float(v), 2))
            else:
                row.append(str(v))
        rows.append(row)
    return markdown_table(cols, rows)


def update_docs(verdict: str, arm_b_row: pd.Series) -> None:
    bullet = (
        f"- 2026-06-02 OD conditional-probability calibration: **{verdict}**. Binance-only Arm B empirical `P(resolve|z,t)` applied to the v4 far-|z| strict-rich short set gives "
        f"mean predicted ITM {pct(float(arm_b_row['mean_pred_itm_prob']))}, observed ITM {pct(float(arm_b_row['realized_itm_rate']))}, "
        f"model edge {cents(float(arm_b_row['mean_model_edge']))}, and realized net EV CI {fmt_ci(float(arm_b_row['net_realized_ev_ci_lo']), float(arm_b_row['net_realized_ev_ci_hi']))}. "
        "OD remains closed standalone unless explicitly reopened with a stronger, pre-registered data source. See [[od_conditional_prob_calibration_findings]]."
    )
    hub = OD_HUB.read_text()
    idx = hub.find("## Current state")
    if idx >= 0:
        next_idx = hub.find("\n## ", idx + 1)
        if next_idx < 0:
            next_idx = len(hub)
        section = hub[idx:next_idx]
        lines = [ln for ln in section.splitlines() if "OD conditional-probability calibration" not in ln]
        new_section = "\n".join([lines[0], "", bullet, *lines[1:]]).rstrip() + "\n"
        hub = hub[:idx] + new_section + hub[next_idx:]
    else:
        hub = hub.rstrip() + "\n\n## Current state\n\n" + bullet + "\n"
    hub = hub.replace("## Current state (2026-06-01)", "## Current state (2026-06-02)")
    if verdict == "CLOSE":
        hub = hub.replace(
            "status: REFRAMED 2026-06-01 — OD is the valuation/signal layer. Strategy A v3 PnL/risk deep-dive failed the strict tail + incremental-over-MM gate under per-asset concurrent capital; see [[od_strategy_a_v3_pnl_risk_findings]]. Kronos/queue replay still gated pending explicit reopen.",
            "status: CLOSED standalone 2026-06-02 — Binance-only conditional-probability calibration did not reopen OD; fold OD richness/source filters into MM as weak quote-selection features unless explicitly reopened.",
        )
    OD_HUB.write_text(hub)

    todo = BRAIN_TODO.read_text()
    todo = "\n".join(ln for ln in todo.splitlines() if "OD conditional-probability calibration" not in ln) + "\n"
    od_idx = todo.find("## OD")
    if od_idx >= 0:
        line_end = todo.find("\n", od_idx)
        suffix = todo[line_end + 1 :]
        if not suffix.startswith("\n"):
            suffix = "\n" + suffix
        todo = todo[: line_end + 1] + bullet + "\n" + suffix
    else:
        todo = todo.rstrip() + "\n\n## OD\n" + bullet + "\n"
    BRAIN_TODO.write_text(todo)


def write_note(hist: pd.DataFrame, cv: pd.DataFrame, cv_bins: pd.DataFrame, pm: pd.DataFrame, pm_summary: pd.DataFrame, bins: pd.DataFrame, verdict: str) -> None:
    arm_b = pm_summary[pm_summary["label"].eq("arm_b_empirical_conditional_original_set")].iloc[0]
    arm_a = pm_summary[pm_summary["label"].eq("arm_a_ewma_nz_original_set")].iloc[0]
    arm_b_rich = pm_summary[pm_summary["label"].eq("arm_b_empirical_conditional_rich_ge_1c")].iloc[0]

    arm_table = format_table(
        pm_summary,
        [
            "label",
            "fills",
            "markets",
            "mean_short_price",
            "mean_pred_itm_prob",
            "realized_itm_rate",
            "calibration_gap_obs_minus_pred",
            "mean_model_edge",
            "model_edge_ci_lo",
            "mean_net_realized_ev",
            "net_realized_ev_ci_lo",
            "mean_market_net_after_top3",
            "market_net_after_top3_ci_lo",
            "incremental_after_top3_vs_zero",
            "incremental_after_top3_vs_zero_ci_lo",
            "borrowed_structural_baseline_mean",
            "incremental_after_top3_vs_borrowed_structural",
            "incremental_after_top3_vs_borrowed_structural_ci_lo",
            "primary_lookup_share",
        ],
    )
    cv_table = format_table(
        cv_bins[cv_bins["arm"].eq("arm_b_empirical_conditional")],
        ["bucket", "rows", "mean_pred", "observed_freq", "obs_minus_pred"],
    )
    pm_bin_table = format_table(
        bins[bins["row_type"].eq("pm_v4_fill_reliability") & bins["arm"].eq("arm_b_empirical_conditional")],
        ["bucket", "rows", "markets", "mean_pred", "observed_freq", "obs_minus_pred", "mean_short_price", "mean_edge", "mean_net_realized_ev"],
    )

    decision_text = (
        "OD **REOPENS** only if Arm B is calibrated, has positive realized EV lower-CI, and beats the v4 structural 0c-edge baseline with lower-CI > 0."
        if verdict == "REOPEN"
        else "OD **CLOSES as a standalone strategy** in this time-boxed test. Arm B does not prove that OD fair-value richness adds robust incremental edge beyond the structural/source quote-selection baseline."
    )
    c_text = (
        "Arms C/D were not run because Arm B already decided the gate. That follows the Kronos discipline: do not escalate to HAR/Kronos when the model-free conditional probability does not reopen the standalone signal."
        if verdict == "CLOSE"
        else "Arm B reopened the branch; Arms C/D should be run only as sharpening baselines before any execution work."
    )
    asset_windows = hist.drop_duplicates(["asset", "window_start"]).shape[0]

    note = f"""# OD Conditional Resolution-Probability Calibration: Binance-Only Reopen-Or-Close Test

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior OD notes: [[od_v4_calibration_gate_findings]] · [[od_v4_queue_replay_findings]] · [[od_strategy_a_v3_pnl_risk_findings]]
> ML discipline gate: [[2026-05-31_kronos_hermes_eval]]
> Table terms: [[polymarket_table_dictionary]]

## Headline

Final verdict: **{verdict}**.

{decision_text}

Arm B, the Binance-only empirical conditional probability model, predicts {pct(float(arm_b['mean_pred_itm_prob']))} ITM on the same 23-fill v4 far-|z| strict-rich short set. The tokens actually paid {pct(float(arm_b['realized_itm_rate']))}. The mean Arm-B model edge is {cents(float(arm_b['mean_model_edge']))}, CI {fmt_ci(float(arm_b['model_edge_ci_lo']), float(arm_b['model_edge_ci_hi']))}; realized net EV is {cents(float(arm_b['mean_net_realized_ev']))}, CI {fmt_ci(float(arm_b['net_realized_ev_ci_lo']), float(arm_b['net_realized_ev_ci_hi']))}. After the K5 top-3 maker capacity haircut, the market-level lower CI is {cents(float(arm_b['incremental_after_top3_vs_zero_ci_lo']))} versus a 0c baseline and {cents(float(arm_b['incremental_after_top3_vs_borrowed_structural_ci_lo']))} after subtracting the borrowed v4 structural queue baseline.

Plain-English read: Binance history says the far-|z| states are not absurd longshots. In the exact PM set, the empirical conditional probability is close enough to the traded price that the OD richness gap is not independently decisive. The positive realized cents are still small-sample/concentration-sensitive, and the structural 0c-edge replay remains the better explanation than a standalone OD valuation edge.

## Design

This test uses **no new Polymarket capture**. It downloads/caches Binance spot 5-minute klines for BTC/ETH/SOL from `{START_MONTH}` through `{END_MONTH}` and builds synthetic 4h UP/DOWN windows aligned to UTC 4h boundaries. For every in-window 5-minute state, it computes:

```text
K = Binance open at the 4h window start
tau = seconds to the 4h close
z = ln(S_t / K) / (causal_EWMA_sigma_t * sqrt(tau))
outcome = 1 if Binance close > K else 0
```

Arm A is the old RV physical-probability control: `N(z)`. Arm B is model-free: expanding-time CV estimates empirical `P(resolve UP | signed z bucket, time-left bucket)` from prior Binance history. For the PM token side, UP uses that probability directly and DOWN uses `1 - P(resolve UP)`.

Historical sample: {len(hist):,} Binance 5-minute states, {asset_windows:,} asset-window 4h outcomes across {hist['window_start'].nunique():,} UTC time slots, assets `{', '.join(sorted(hist['asset'].unique()))}`. Expanding-CV validation rows: {len(cv):,}.

### Granularity Caveat: What The 5-Minute History Is And Is Not

The 2021-to-2026 Binance history is a **broad base-rate calibration**, not a live-execution-quality reconstruction of the exact Polymarket episodes. It answers: "In thousands of historical 4h crypto windows, what is the empirical resolution frequency for this signed moneyness/time-left state?" That is useful for detecting whether the Gaussian `N(z)` model is wildly wrong.

It does **not** replace the captured-window data. For the actual PM windows we also have much richer, more relevant evidence:

- `data/analysis/block_a0c_roll_features.parquet`: crypto-4h Polymarket LOB/WS capture with top-of-book, depth, trade flow, OFI, and exchange timestamps across 38 crypto-4h slugs on 2026-05-29 to 2026-05-30.
- `data/analysis/block_a0c_features.parquet`: A0c targeted capture with crypto-4h plus daily crypto rows on 2026-05-29.
- `data/analysis/cache/k2v2_daily_binance_1s.parquet` and `data/analysis/cache/k2v2_daily_model_surface.parquet`: 1s Binance/model surface for the daily BTC/ETH crypto capture on 2026-05-27 to 2026-05-28.

Plain-English read: the 5-minute historical panel is a good truth-table prior; the captured 1s Binance + Polymarket LOB windows are the right place to ask whether the **live market state** had jumps, order-flow pressure, OFI, liquidity depletion, or source-basis risk that the broad 5-minute table cannot see.

For the pricing-model-form reopen test, the preferred ordering should be:

1. Use the historical 5-minute panel only as the background calibration/control for `P(resolve | z, tau)`.
2. For the actual PM validation rows, rebuild the state at 1s granularity from the captured windows where available.
3. Estimate jump/OFI features from the local live window around each fill, not only from multi-year unconditional Binance history.
4. Treat Deribit BTC/ETH as an illustrative market-IV anchor for the same captured days, not as a powered gate.

This caveat does not overturn this note's close verdict. It narrows what kind of evidence would be allowed to reopen OD: a stronger pricing-model-form test should show residual EV on the **captured-window/live 1s panel**, not merely on a smoother 5-minute historical lookup.

## Arm B Binance Reliability

![Arm B calibration]({CALIBRATION_PLOT.resolve()})

The diagonal is perfect calibration. Points near the diagonal mean the empirical conditional table is learning the actual Binance resolution frequency for that probability range.

| Arm B probability bucket | rows | mean predicted UP | observed UP | observed - predicted |
| --- | --- | --- | --- | --- |
{chr(10).join(cv_table.splitlines()[2:])}

Read: this is the truth-teller independent of the tiny Polymarket sample. If Arm B were badly calibrated here, it would be an invalid reopen signal. If Arm B is calibrated but removes the OD edge on PM fills, the old `N(z)` gap was mostly our model/specification, not a robust Polymarket mispricing.

## PM v4 Far-|z| Short Set

| label | fills | markets | mean_short_price | mean_pred_itm_prob | realized_itm_rate | obs - pred | mean_model_edge | edge CI | mean net EV | net EV CI | after-top3 market net | after-top3 CI | incremental vs 0c | inc vs 0c CI | borrowed baseline | incremental vs borrowed | inc vs borrowed CI | primary lookup share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(arm_table.splitlines()[2:])}

Column read: `mean_model_edge` is `short price - predicted ITM probability`. `mean net EV` is the actual resolution PnL per fill after maker rebate. `after-top3 market net` applies the 5% non-incumbent capacity haircut used in v4. `incremental vs 0c` is the raw capacity-adjusted read. `incremental vs borrowed` subtracts the best v4 0c-edge queue replay baseline. Live-measured baseline is not filled in this run because there is no separate live queue baseline artifact for the same validation rows yet.

![PM EV by arm]({EV_PLOT.resolve()})

## Arm B PM Reliability Buckets

| Arm B probability bucket | fills | markets | mean predicted ITM | observed ITM | observed - predicted | mean short price | mean edge | mean net EV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(pm_bin_table.splitlines()[2:])}

Read: the PM set is tiny, so this table is diagnostic rather than decisive. The decisive comparison is the Binance-CV reliability plus the PM realized/incremental gate above.

## Decision

{decision_text}

{c_text}

Clarification for future reopen prompts: the skipped Arms C/D here refer to the **conditional-probability task's** HAR/Kronos-style forward-vol arms. A separate pricing-model-form diagnostic may still run a small jump-model/Deribit extension, but it should be anchored on the captured 1s/LOB windows. In that framing, Merton/Kou jump diffusion is the cheap first extension; Bates/Variance-Gamma is only worth running if Merton/Kou leaves residual lower-CI-positive EV; Deribit is BTC/ETH-only and illustrative.

Operational next step: fold the source/richness information back into [[strat_market_making]] as a weak quote-selection or caution feature. Do not build new OD queue infrastructure from this result.

## Outputs

- Summary CSV: `data/analysis/csv_outputs/options_delta/od_conditional_prob_calibration_summary.csv`
- Calibration bins CSV: `data/analysis/csv_outputs/options_delta/od_conditional_prob_calibration_bins.csv`
- PM fills parquet: `data/analysis/od_conditional_prob_pm_fills.parquet`
- Binance history parquet: `data/analysis/od_conditional_prob_binance_history.parquet`
- Binance CV parquet: `data/analysis/od_conditional_prob_binance_cv.parquet`
"""
    NOTE.write_text(normalize_markdown_wrapping(note))
    update_docs(verdict, arm_b)


def run() -> None:
    hist = build_history(refresh=False)
    cv = expanding_year_cv(hist)
    if cv.empty:
        raise SystemExit("empty expanding CV")
    cv_bins = calibration_bins_cv(cv)
    pm_raw = pd.read_parquet(V4_CONTRACTS)
    pm = apply_lookup_to_pm(pm_raw, hist)
    baseline = structural_baseline()
    pm_summary = summarize_pm(pm, baseline)
    bins = pd.concat([cv_bins, pm_bins(pm)], ignore_index=True, sort=False)
    make_plots(cv_bins, pm_summary)

    arm_b = pm_summary[pm_summary["label"].eq("arm_b_empirical_conditional_original_set")].iloc[0]
    arm_b_rich = pm_summary[pm_summary["label"].eq("arm_b_empirical_conditional_rich_ge_1c")].iloc[0]
    arm_b_promising = bool(
        float(arm_b["model_edge_ci_lo"]) > 0
        and float(arm_b["net_realized_ev_ci_lo"]) > 0
        and float(arm_b["incremental_after_top3_ci_lo"]) > 0
        and int(arm_b_rich["fills"]) >= 12
    )
    verdict = "REOPEN" if arm_b_promising else "CLOSE"

    CSV_OUT.mkdir(parents=True, exist_ok=True)
    pm_summary.to_csv(OUT_SUMMARY, index=False)
    bins.to_csv(OUT_BINS, index=False)
    pm.to_parquet(OUT_PM, index=False)
    cv.to_parquet(OUT_CV, index=False)
    write_note(hist, cv, cv_bins, pm, pm_summary, bins, verdict)
    print(f"verdict={verdict}")
    print(f"history rows={len(hist):,} asset_windows={hist.drop_duplicates(['asset', 'window_start']).shape[0]:,}")
    print(f"cv rows={len(cv):,}")
    print(f"wrote {OUT_SUMMARY}")
    print(f"wrote {OUT_BINS}")
    print(f"wrote {OUT_PM}")
    print(f"wrote {OUT_HIST}")
    print(f"wrote {OUT_CV}")
    print(f"wrote {NOTE}")


if __name__ == "__main__":
    run()
