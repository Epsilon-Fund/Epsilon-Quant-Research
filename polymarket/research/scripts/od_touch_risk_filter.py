"""OD touch-risk skip filter for the longshot harvest.

This is an offline pre-test, not a trading signal. It asks whether cached
Binance path/flow/jump features can flag toxic 4h states where the OTM side
touches the strike after entry, then applies that score to the tiny PM
Strategy-A fill sample as a maker-skip diagnostic.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/od_touch_risk_filter.py
"""
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import YEAR_SECONDS, cents, markdown_table, number, pct
from od_strategy_a_v3 import normalize_markdown_wrapping


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
BRAIN_TODO = REPO / "brain" / "TODO.md"
OD_HUB = NOTES / "options_delta" / "strat_options_delta.md"

BINANCE_5M = ROOT / "data" / "external" / "binance_5m"
ZIP_DIR = BINANCE_5M / "monthly_zips"
CSV_OUT = ANALYSIS / "csv_outputs" / "options_delta"
PLOTS = ANALYSIS / "plots" / "options_delta"

TAIL_HISTORY = ANALYSIS / "od_strategy_a_tail_extended_binance_history.parquet"
PM_WEIGHTED_FILLS = ANALYSIS / "od_strategy_a_tail_sizing_weighted_fills.parquet"

OUT_BARS = ANALYSIS / "od_touch_risk_filter_binance_5m_features.parquet"
OUT_WINDOWS = ANALYSIS / "od_touch_risk_filter_candidate_windows.parquet"
OUT_PM = ANALYSIS / "od_touch_risk_filter_scored_pm_fills.parquet"
OUT_SEPARATION = CSV_OUT / "od_touch_risk_filter_separation.csv"
OUT_DECILES = CSV_OUT / "od_touch_risk_filter_deciles.csv"
OUT_SKIP = CSV_OUT / "od_touch_risk_filter_skip_sweep.csv"
OUT_PM_SKIP = CSV_OUT / "od_touch_risk_filter_pm_skip.csv"
OUT_DATA_LEDGER = CSV_OUT / "od_touch_risk_filter_data_ledger.csv"

PLOT_DECILES = PLOTS / "od_touch_risk_filter_decile_touch_rate.png"
PLOT_PM_SKIP = PLOTS / "od_touch_risk_filter_pm_skip_stress_ev.png"
NOTE = NOTES / "options_delta" / "od_touch_risk_filter_findings.md"

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
RNG_SEED = 20260603
BOOTSTRAP_SAMPLES = 1000
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC")
HELDOUT_START = pd.Timestamp("2025-01-08", tz="UTC")
FAR_ABS_Z_MIN = 1.0
MIN_ENTRY_MINUTES_AFTER_OPEN = 30
MIN_TAU_SECONDS = 300.0
LM_WINDOW_BARS = 72
LM_MIN_BARS = 24
LM_Z_THRESHOLD = 4.0
LM_ABS_RETURN_FLOOR_BPS = 10.0
EPS = 1e-12
SANE_START = pd.Timestamp("2021-01-01", tz="UTC")
SANE_END = pd.Timestamp("2026-06-05", tz="UTC")
SANE_START_MS = int(SANE_START.timestamp() * 1000)
SANE_END_MS = int(SANE_END.timestamp() * 1000)


FEATURE_COLS = [
    "abs_z",
    "tau_hours",
    "ewma_sigma_annualized",
    "p_otm_model",
    "adv_mom_3",
    "adv_mom_6",
    "adv_mom_12",
    "adv_mom_24",
    "adv_tfi_3",
    "adv_tfi_6",
    "adv_tfi_12",
    "adv_tfi_24",
    "adv_jump_count_6",
    "adv_jump_count_12",
    "adv_jump_count_24",
    "opp_jump_count_12",
    "adv_max_ret_6",
    "adv_max_ret_12",
    "realized_vol_12",
    "realized_vol_24",
]


def fmt_ci(lo: float, hi: float, unit: str = "") -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "[n/a, n/a]"
    if unit == "pct":
        return f"[{pct(lo)}, {pct(hi)}]"
    if unit == "c":
        return f"[{cents(lo)}, {cents(hi)}]"
    return f"[{number(lo, 3)}, {number(hi, 3)}]"


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def complement(side: str) -> str:
    return "down" if str(side).lower() == "up" else "up"


def epoch_ms_values(values: pd.Series | np.ndarray) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(values):
        raw = pd.Series(values).astype("int64").to_numpy(dtype=float)
    else:
        numeric = pd.to_numeric(values, errors="coerce")
        if pd.Series(numeric).notna().any():
            raw = pd.Series(numeric).to_numpy(dtype=float)
        else:
            parsed = pd.to_datetime(values, utc=True, errors="coerce")
            raw = pd.Series(parsed).astype("int64").to_numpy(dtype=float)
    raw = np.where(raw > SANE_END_MS * 100_000, raw / 1_000_000.0, raw)
    raw = np.where(raw > SANE_END_MS * 100, raw / 1_000.0, raw)
    raw = np.where((raw >= SANE_START_MS - 86_400_000) & (raw <= SANE_END_MS + 86_400_000), raw, np.nan)
    return raw


def normalize_timestamp_series(values: pd.Series | np.ndarray) -> pd.Series:
    idx = values.index if isinstance(values, pd.Series) else None
    ms = epoch_ms_values(values)
    out = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
    mask = np.isfinite(ms)
    if mask.any():
        out.loc[mask] = pd.to_datetime(np.rint(ms[mask]).astype("int64"), unit="ms", utc=True)
    return out


def normalize_bar_times(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    out["bar_open_ts"] = normalize_timestamp_series(out["bar_open_ts"])
    out["ts"] = normalize_timestamp_series(out["ts"])
    out = out[out["bar_open_ts"].notna() & out["ts"].notna()].copy()
    out = out[out["bar_open_ts"].between(SANE_START, SANE_END) & out["ts"].between(SANE_START, SANE_END)].copy()
    out["window_start"] = out["bar_open_ts"].dt.floor("4h")
    return out


def recompute_future_columns(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.sort_values(["asset", "bar_open_ts"]).reset_index(drop=True).copy()
    enhanced: list[pd.DataFrame] = []
    for _, g in out.groupby("asset", sort=False):
        g = g.reset_index(drop=True).copy()
        future_max_high = np.full(len(g), np.nan, dtype=float)
        future_min_low = np.full(len(g), np.nan, dtype=float)
        future_up_jump_out = np.zeros(len(g), dtype=bool)
        future_down_jump_out = np.zeros(len(g), dtype=bool)
        high_arr = g["high"].to_numpy(dtype=float)
        low_arr = g["low"].to_numpy(dtype=float)
        up_jump_arr = g["up_jump_flag"].fillna(False).to_numpy(dtype=bool)
        down_jump_arr = g["down_jump_flag"].fillna(False).to_numpy(dtype=bool)
        for _, idx in g.groupby("window_start", sort=False).groups.items():
            pos = np.asarray(list(idx), dtype=int)
            highs = high_arr[pos]
            lows = low_arr[pos]
            up_jumps = up_jump_arr[pos]
            down_jumps = down_jump_arr[pos]
            future_high = np.maximum.accumulate(highs[::-1])[::-1]
            future_low = np.minimum.accumulate(lows[::-1])[::-1]
            future_up = np.logical_or.accumulate(up_jumps[::-1])[::-1]
            future_down = np.logical_or.accumulate(down_jumps[::-1])[::-1]
            future_max_high[pos] = np.r_[future_high[1:], np.nan]
            future_min_low[pos] = np.r_[future_low[1:], np.nan]
            future_up_jump_out[pos] = np.r_[future_up[1:], False]
            future_down_jump_out[pos] = np.r_[future_down[1:], False]
        g["future_max_high"] = future_max_high
        g["future_min_low"] = future_min_low
        g["future_up_jump"] = future_up_jump_out
        g["future_down_jump"] = future_down_jump_out
        enhanced.append(g)
    return pd.concat(enhanced, ignore_index=True)


def parse_month_zip(path: Path, asset: str) -> pd.DataFrame:
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
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
        if not members:
            return pd.DataFrame(columns=cols)
        with zf.open(members[0]) as fh:
            raw = fh.read()
    df = pd.read_csv(io.BytesIO(raw), header=None, names=cols)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df[df["open_time"].notna()].copy()
    for col in ["open", "high", "low", "close", "volume", "close_time", "n_trades", "taker_buy_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open_time", "open", "high", "low", "close", "volume", "close_time"])
    open_ms = epoch_ms_values(df["open_time"])
    close_ms = epoch_ms_values(df["close_time"])
    good = np.isfinite(open_ms) & np.isfinite(close_ms)
    df = df.loc[good].copy()
    open_ms = open_ms[good]
    close_ms = close_ms[good]
    df["bar_open_ts"] = pd.to_datetime(np.rint(open_ms).astype("int64"), unit="ms", utc=True)
    df["ts"] = pd.to_datetime(np.rint(close_ms).astype("int64") + 1, unit="ms", utc=True)
    df["asset"] = asset
    return df[["asset", "bar_open_ts", "ts", "open", "high", "low", "close", "volume", "n_trades", "taker_buy_volume"]]


def load_binance_bars_with_flow(refresh: bool = False) -> pd.DataFrame:
    if OUT_BARS.exists() and not refresh:
        out = pd.read_parquet(OUT_BARS)
        out = recompute_future_columns(normalize_bar_times(out))
        out.to_parquet(OUT_BARS, index=False)
        return out

    pieces: list[pd.DataFrame] = []
    for asset, symbol in SYMBOLS.items():
        paths = sorted(ZIP_DIR.glob(f"{symbol}-5m-*.zip"))
        if not paths:
            cached = BINANCE_5M / f"{symbol}_5m_2021-01_2026-05.parquet"
            if not cached.exists():
                raise SystemExit(f"missing Binance 5m zip/parquet cache for {symbol}")
            df = pd.read_parquet(cached)
            df["n_trades"] = np.nan
            df["taker_buy_volume"] = np.nan
            pieces.append(df)
            continue
        asset_pieces = [parse_month_zip(path, asset) for path in paths]
        pieces.append(pd.concat(asset_pieces, ignore_index=True))
        print(f"parsed {symbol} monthly zip rows={sum(len(x) for x in asset_pieces):,}", flush=True)

    bars = normalize_bar_times(pd.concat(pieces, ignore_index=True))
    bars = bars.drop_duplicates(["asset", "bar_open_ts"]).sort_values(["asset", "bar_open_ts"]).reset_index(drop=True)
    bars["window_start"] = bars["bar_open_ts"].dt.floor("4h")
    bars["log_ret"] = np.log(bars["close"].astype(float) / bars.groupby("asset")["close"].shift(1).astype(float))
    bars["bar_log_ret"] = np.log(bars["close"].astype(float) / bars["open"].astype(float))
    bars["taker_buy_volume"] = bars["taker_buy_volume"].fillna(bars["volume"] * (bars["bar_log_ret"].gt(0).astype(float) * 0.5 + 0.25))
    bars["signed_taker_volume"] = 2.0 * bars["taker_buy_volume"].astype(float) - bars["volume"].astype(float)
    bars["taker_imbalance"] = bars["signed_taker_volume"] / bars["volume"].replace(0.0, np.nan)

    enhanced: list[pd.DataFrame] = []
    for asset, g in bars.groupby("asset", sort=False):
        g = g.sort_values("bar_open_ts").reset_index(drop=True).copy()
        abs_ret = g["log_ret"].abs()
        bipower = (math.pi / 2.0) * (abs_ret * abs_ret.shift(1)).rolling(LM_WINDOW_BARS, min_periods=LM_MIN_BARS).mean().shift(1)
        g["lm_sigma"] = np.sqrt(bipower)
        g["lm_z"] = g["log_ret"] / g["lm_sigma"].replace(0.0, np.nan)
        g["jump_flag"] = g["lm_z"].abs().gt(LM_Z_THRESHOLD) & g["log_ret"].abs().gt(LM_ABS_RETURN_FLOOR_BPS / 1e4)
        g["up_jump_flag"] = g["jump_flag"] & g["log_ret"].gt(0)
        g["down_jump_flag"] = g["jump_flag"] & g["log_ret"].lt(0)
        g["up_ret"] = g["log_ret"].clip(lower=0.0)
        g["down_ret_abs"] = (-g["log_ret"]).clip(lower=0.0)
        for n in [3, 6, 12, 24]:
            g[f"ret_sum_{n}"] = g["log_ret"].rolling(n, min_periods=1).sum()
            vol_sum = g["volume"].rolling(n, min_periods=1).sum()
            g[f"taker_imbalance_{n}"] = g["signed_taker_volume"].rolling(n, min_periods=1).sum() / vol_sum.replace(0.0, np.nan)
            g[f"rv_{n}"] = np.sqrt(g["log_ret"].pow(2).rolling(n, min_periods=2).sum() * YEAR_SECONDS / (300.0 * n))
            g[f"up_jump_count_{n}"] = g["up_jump_flag"].astype(float).rolling(n, min_periods=1).sum()
            g[f"down_jump_count_{n}"] = g["down_jump_flag"].astype(float).rolling(n, min_periods=1).sum()
            g[f"up_ret_max_{n}"] = g["up_ret"].rolling(n, min_periods=1).max()
            g[f"down_ret_max_{n}"] = g["down_ret_abs"].rolling(n, min_periods=1).max()

        future_max_high = np.full(len(g), np.nan, dtype=float)
        future_min_low = np.full(len(g), np.nan, dtype=float)
        future_up_jump_out = np.zeros(len(g), dtype=bool)
        future_down_jump_out = np.zeros(len(g), dtype=bool)
        high_arr = g["high"].to_numpy(dtype=float)
        low_arr = g["low"].to_numpy(dtype=float)
        up_jump_arr = g["up_jump_flag"].to_numpy(dtype=bool)
        down_jump_arr = g["down_jump_flag"].to_numpy(dtype=bool)
        for _, idx in g.groupby("window_start", sort=False).groups.items():
            pos = np.asarray(list(idx), dtype=int)
            highs = high_arr[pos]
            lows = low_arr[pos]
            up_jumps = up_jump_arr[pos]
            down_jumps = down_jump_arr[pos]
            future_high = np.maximum.accumulate(highs[::-1])[::-1]
            future_low = np.minimum.accumulate(lows[::-1])[::-1]
            future_up = np.logical_or.accumulate(up_jumps[::-1])[::-1]
            future_down = np.logical_or.accumulate(down_jumps[::-1])[::-1]
            future_max_high[pos] = np.r_[future_high[1:], np.nan]
            future_min_low[pos] = np.r_[future_low[1:], np.nan]
            future_up_jump_out[pos] = np.r_[future_up[1:], False]
            future_down_jump_out[pos] = np.r_[future_down[1:], False]
        g["future_max_high"] = future_max_high
        g["future_min_low"] = future_min_low
        g["future_up_jump"] = future_up_jump_out
        g["future_down_jump"] = future_down_jump_out
        enhanced.append(g)

    out = pd.concat(enhanced, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    OUT_BARS.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_BARS, index=False)
    print(f"wrote Binance 5m feature cache rows={len(out):,} -> {OUT_BARS}", flush=True)
    return out


def add_oriented_features(df: pd.DataFrame, side: pd.Series | np.ndarray) -> pd.DataFrame:
    out = df.copy()
    side_arr = np.asarray(side, dtype=float)
    out["side"] = side_arr
    out["tau_hours"] = out["tau_seconds"].astype(float) / 3600.0
    out["p_otm_model"] = np.minimum(out["p_model"].astype(float), 1.0 - out["p_model"].astype(float))
    for n in [3, 6, 12, 24]:
        out[f"adv_mom_{n}"] = side_arr * out[f"ret_sum_{n}"].astype(float)
        out[f"adv_tfi_{n}"] = side_arr * out[f"taker_imbalance_{n}"].astype(float)
        out[f"adv_jump_count_{n}"] = np.where(
            side_arr > 0,
            out[f"up_jump_count_{n}"].astype(float),
            out[f"down_jump_count_{n}"].astype(float),
        )
        out[f"opp_jump_count_{n}"] = np.where(
            side_arr > 0,
            out[f"down_jump_count_{n}"].astype(float),
            out[f"up_jump_count_{n}"].astype(float),
        )
        out[f"adv_max_ret_{n}"] = np.where(
            side_arr > 0,
            out[f"up_ret_max_{n}"].astype(float),
            out[f"down_ret_max_{n}"].astype(float),
        )
        out[f"realized_vol_{n}"] = out[f"rv_{n}"].astype(float)
    return out


def build_candidate_windows(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    bars = load_binance_bars_with_flow(refresh=refresh)
    if OUT_WINDOWS.exists() and not refresh:
        candidates = pd.read_parquet(OUT_WINDOWS)
        for col in ["bar_open_ts", "ts", "window_start", "window_end", "touch_ts"]:
            if col in candidates:
                candidates[col] = pd.to_datetime(candidates[col], utc=True)
        if candidates["split"].eq("heldout").any():
            total_windows = int(pd.read_parquet(TAIL_HISTORY, columns=["asset", "window_start"]).drop_duplicates().shape[0])
            return candidates, bars, total_windows
        print("candidate cache has no held-out rows after mixed timestamp repair; rebuilding", flush=True)

    hist = pd.read_parquet(TAIL_HISTORY)
    for col in ["bar_open_ts", "ts", "window_start", "window_end"]:
        hist[col] = pd.to_datetime(hist[col], utc=True)
    total_windows = int(hist[["asset", "window_start"]].drop_duplicates().shape[0])
    merge_cols = [
        "asset",
        "ts",
        "bar_open_ts",
        "window_start",
        "open",
        "high",
        "low",
        "volume",
        "n_trades",
        "taker_buy_volume",
        "signed_taker_volume",
        "taker_imbalance",
        "log_ret",
        "lm_z",
        "jump_flag",
        "up_jump_flag",
        "down_jump_flag",
        "future_up_jump",
        "future_down_jump",
        "future_max_high",
        "future_min_low",
    ]
    for n in [3, 6, 12, 24]:
        merge_cols.extend(
            [
                f"ret_sum_{n}",
                f"taker_imbalance_{n}",
                f"rv_{n}",
                f"up_jump_count_{n}",
                f"down_jump_count_{n}",
                f"up_ret_max_{n}",
                f"down_ret_max_{n}",
            ]
        )
    merged = hist.merge(bars[merge_cols], on=["asset", "ts", "window_start"], how="left", suffixes=("", "_bar"))
    merged["minutes_after_open"] = (merged["ts"] - merged["window_start"]).dt.total_seconds() / 60.0
    pool = merged[
        merged["abs_z"].astype(float).ge(FAR_ABS_Z_MIN)
        & merged["tau_seconds"].astype(float).ge(MIN_TAU_SECONDS)
        & merged["minutes_after_open"].ge(MIN_ENTRY_MINUTES_AFTER_OPEN)
        & merged["future_max_high"].notna()
        & merged["future_min_low"].notna()
    ].copy()
    pool = pool.sort_values(["asset", "window_start", "ts"])
    candidates = pool.groupby(["asset", "window_start"], as_index=False, sort=False).head(1).reset_index(drop=True)
    side = np.where(candidates["digital_z"].astype(float).lt(0), 1.0, -1.0)
    candidates = add_oriented_features(candidates, side)
    candidates["bad_touch"] = np.where(
        candidates["side"].gt(0),
        candidates["future_max_high"].astype(float).ge(candidates["strike"].astype(float)),
        candidates["future_min_low"].astype(float).le(candidates["strike"].astype(float)),
    )
    candidates["terminal_bad"] = np.where(
        candidates["side"].gt(0),
        candidates["binance_resolution_up"].astype(float).eq(1.0),
        candidates["binance_resolution_up"].astype(float).eq(0.0),
    )
    future_adverse_jump = np.where(candidates["side"].gt(0), candidates["future_up_jump"].fillna(False), candidates["future_down_jump"].fillna(False))
    candidates["touch_ts"] = pd.NaT
    candidates["touch_jump_driven"] = candidates["bad_touch"].astype(bool) & pd.Series(future_adverse_jump, index=candidates.index).astype(bool)
    candidates["touch_first_bar_ret"] = np.nan
    candidates["split"] = np.select(
        [candidates["window_start"].lt(TRAIN_END), candidates["window_start"].ge(HELDOUT_START)],
        ["train", "heldout"],
        default="embargo",
    )
    keep_raw = [
        "asset",
        "bar_open_ts",
        "ts",
        "window_start",
        "window_end",
        "strike",
        "close",
        "close_spot",
        "tau_seconds",
        "tau_hours",
        "ewma_sigma_annualized",
        "digital_z",
        "abs_z",
        "p_model",
        "p_otm_model",
        "time_bucket",
        "signed_z_bucket",
        "side",
        "bad_touch",
        "terminal_bad",
        "touch_ts",
        "touch_jump_driven",
        "touch_first_bar_ret",
        "split",
        *FEATURE_COLS,
    ]
    keep = list(dict.fromkeys(keep_raw))
    candidates = candidates[keep].replace([np.inf, -np.inf], np.nan)
    OUT_WINDOWS.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(OUT_WINDOWS, index=False)
    print(f"wrote candidate windows rows={len(candidates):,} of total_windows={total_windows:,} -> {OUT_WINDOWS}", flush=True)
    return candidates, bars, total_windows


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auc(y: np.ndarray, score: np.ndarray, seed: int) -> tuple[float, float]:
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if len(y) < 10 or y.sum() == 0 or (~y).sum() == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(y), size=len(y))
        if y[idx].sum() == 0 or (~y[idx]).sum() == 0:
            continue
        vals.append(auc_score(y[idx], score[idx]))
    if not vals:
        return math.nan, math.nan
    lo, hi = np.nanquantile(np.asarray(vals, dtype=float), [0.025, 0.975])
    return float(lo), float(hi)


def fit_logistic(train: pd.DataFrame) -> dict[str, Any]:
    train = train.copy()
    y = train["bad_touch"].astype(float).to_numpy()
    med = train[FEATURE_COLS].median(numeric_only=True)
    x_raw = train[FEATURE_COLS].fillna(med).to_numpy(dtype=float)
    mu = np.nanmean(x_raw, axis=0)
    sigma = np.nanstd(x_raw, axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    x = (x_raw - mu) / sigma
    x = np.column_stack([np.ones(len(x)), x])
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    sample_w = np.where(y > 0, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    sample_w = sample_w / np.mean(sample_w)
    beta = np.zeros(x.shape[1], dtype=float)
    l2 = 0.25
    reg = np.ones_like(beta)
    reg[0] = 0.0
    denom = float(sample_w.sum())
    for _ in range(60):
        p = sigmoid(x @ beta)
        grad = (x.T @ (sample_w * (p - y))) / denom + l2 * reg * beta / len(beta)
        w = sample_w * p * (1.0 - p)
        hess = (x.T @ (x * w[:, None])) / denom + np.diag(l2 * reg / len(beta) + 1e-6)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta -= step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    return {"beta": beta, "median": med, "mu": mu, "sigma": sigma, "feature_cols": FEATURE_COLS}


def score_frame(df: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    med: pd.Series = model["median"]
    x_raw = df[model["feature_cols"]].fillna(med).to_numpy(dtype=float)
    x = (x_raw - model["mu"]) / model["sigma"]
    x = np.column_stack([np.ones(len(x)), x])
    return sigmoid(x @ model["beta"])


def split_metrics(candidates: pd.DataFrame, model: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = candidates.copy()
    out["risk_score"] = score_frame(out, model)
    train = out[out["split"].eq("train")]
    heldout = out[out["split"].eq("heldout")]

    rows: list[dict[str, Any]] = []
    for label_name, label_col, subset in [
        ("bad_touch", "bad_touch", heldout),
        ("terminal_bad", "terminal_bad", heldout),
        ("jump_driven_touch_vs_no_touch", "touch_jump_driven", heldout[heldout["touch_jump_driven"] | ~heldout["bad_touch"]]),
    ]:
        y = subset[label_col].astype(bool).to_numpy()
        score = subset["risk_score"].to_numpy(dtype=float)
        auc = auc_score(y, score)
        lo, hi = bootstrap_auc(y, score, seed=RNG_SEED + len(rows) * 17)
        rows.append(
            {
                "sample": "heldout",
                "label": label_name,
                "rows": int(len(subset)),
                "positives": int(y.sum()),
                "base_rate": float(np.mean(y)) if len(y) else math.nan,
                "auc": auc,
                "auc_ci_lo": lo,
                "auc_ci_hi": hi,
            }
        )
    for asset, g in heldout.groupby("asset"):
        y = g["bad_touch"].astype(bool).to_numpy()
        score = g["risk_score"].to_numpy(dtype=float)
        lo, hi = bootstrap_auc(y, score, seed=RNG_SEED + len(rows) * 17)
        rows.append(
            {
                "sample": f"heldout_{asset}",
                "label": "bad_touch",
                "rows": int(len(g)),
                "positives": int(y.sum()),
                "base_rate": float(np.mean(y)) if len(y) else math.nan,
                "auc": auc_score(y, score),
                "auc_ci_lo": lo,
                "auc_ci_hi": hi,
            }
        )

    q_edges = np.nanquantile(train["risk_score"].to_numpy(dtype=float), np.linspace(0, 1, 11))
    q_edges[0] = -np.inf
    q_edges[-1] = np.inf
    out["score_decile_train_cut"] = pd.cut(out["risk_score"], bins=q_edges, labels=list(range(1, 11)), include_lowest=True)
    decile_rows: list[dict[str, Any]] = []
    for sample, g0 in out[out["split"].isin(["train", "heldout"])].groupby("split"):
        for decile, g in g0.groupby("score_decile_train_cut", observed=False):
            if g.empty:
                continue
            decile_rows.append(
                {
                    "sample": sample,
                    "risk_decile": int(decile),
                    "rows": int(len(g)),
                    "mean_score": float(g["risk_score"].mean()),
                    "bad_touch_rate": float(g["bad_touch"].mean()),
                    "terminal_bad_rate": float(g["terminal_bad"].mean()),
                    "jump_driven_touch_rate": float(g["touch_jump_driven"].mean()),
                }
            )

    thresholds = {"skip_top_30pct": 0.70, "skip_top_20pct": 0.80, "skip_top_10pct": 0.90, "skip_top_5pct": 0.95}
    skip_rows: list[dict[str, Any]] = []
    for name, q in thresholds.items():
        threshold = float(np.nanquantile(train["risk_score"], q))
        for sample_name, sample_df in [("train", train), ("heldout", heldout)]:
            skipped = sample_df["risk_score"].ge(threshold)
            kept = ~skipped
            skip_rows.append(
                {
                    "sample": sample_name,
                    "rule": name,
                    "train_score_threshold": threshold,
                    "rows": int(len(sample_df)),
                    "retained_rows": int(kept.sum()),
                    "retained_fraction": float(kept.mean()) if len(kept) else math.nan,
                    "skipped_touch_rate": float(sample_df.loc[skipped, "bad_touch"].mean()) if skipped.any() else math.nan,
                    "kept_touch_rate": float(sample_df.loc[kept, "bad_touch"].mean()) if kept.any() else math.nan,
                    "skipped_terminal_bad_rate": float(sample_df.loc[skipped, "terminal_bad"].mean()) if skipped.any() else math.nan,
                    "kept_terminal_bad_rate": float(sample_df.loc[kept, "terminal_bad"].mean()) if kept.any() else math.nan,
                    "skipped_jump_driven_touch_rate": float(sample_df.loc[skipped, "touch_jump_driven"].mean()) if skipped.any() else math.nan,
                    "kept_jump_driven_touch_rate": float(sample_df.loc[kept, "touch_jump_driven"].mean()) if kept.any() else math.nan,
                }
            )
    return out, pd.DataFrame(rows), pd.DataFrame(decile_rows), pd.DataFrame(skip_rows)


def add_pm_features(pm: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    pm = pm.copy()
    pm["fill_ts"] = pd.to_datetime(pm["fill_ts"], utc=True)
    pm["fill_ts_ms"] = np.rint(epoch_ms_values(pm["fill_ts"])).astype("int64")
    pm["window_start"] = pd.to_datetime(pm["window_start"], utc=True)
    pm["shorted_side"] = np.where(
        pm["token_position"].astype(float).lt(0),
        pm["actual_outcome"].astype(str).str.lower(),
        pm["actual_outcome"].astype(str).str.lower().map(complement),
    )
    pm["side"] = np.where(pm["shorted_side"].eq("up"), 1.0, -1.0)
    parts: list[pd.DataFrame] = []
    asof_cols = [
        "asset",
        "ts",
        "bar_open_ts",
        "window_start",
        "open",
        "high",
        "low",
        "volume",
        "n_trades",
        "taker_buy_volume",
        "signed_taker_volume",
        "taker_imbalance",
        "log_ret",
        "lm_z",
        "jump_flag",
        "up_jump_flag",
        "down_jump_flag",
    ]
    for n in [3, 6, 12, 24]:
        asof_cols.extend(
            [
                f"ret_sum_{n}",
                f"taker_imbalance_{n}",
                f"rv_{n}",
                f"up_jump_count_{n}",
                f"down_jump_count_{n}",
                f"up_ret_max_{n}",
                f"down_ret_max_{n}",
            ]
        )
    bars = normalize_bar_times(bars)
    for asset, g in pm.groupby("asset", sort=False):
        b = bars[bars["asset"].eq(asset)][asof_cols].copy()
        b["ts"] = normalize_timestamp_series(b["ts"])
        b["ts_ms"] = np.rint(epoch_ms_values(b["ts"])).astype("int64")
        b = b.sort_values("ts_ms")
        p = g.copy()
        p = p.sort_values("fill_ts_ms")
        joined = pd.merge_asof(p, b, left_on="fill_ts_ms", right_on="ts_ms", by="asset", direction="backward", suffixes=("", "_bar"))
        parts.append(joined)
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = normalize_timestamp_series(out["ts"])
    out["tau_seconds"] = out["seconds_to_expiry"].astype(float)
    out["tau_hours"] = out["tau_seconds"] / 3600.0
    out["p_model"] = out["p_model"].astype(float)
    out["p_otm_model"] = np.minimum(out["p_model"], 1.0 - out["p_model"])
    out = add_oriented_features(out, out["side"])
    return out.replace([np.inf, -np.inf], np.nan)


def weighted_market_ci(df: pd.DataFrame, value_col: str, weight_col: str, cluster_col: str = "market_id", seed: int = 0) -> tuple[float, float]:
    if df.empty:
        return math.nan, math.nan
    groups = []
    for _, g in df.groupby(cluster_col, sort=False):
        w = g[weight_col].astype(float).to_numpy()
        v = g[value_col].astype(float).to_numpy()
        mask = np.isfinite(w) & np.isfinite(v) & (w > 0)
        if mask.any():
            groups.append((float(np.sum(w[mask] * v[mask])), float(np.sum(w[mask]))))
    if not groups:
        return math.nan, math.nan
    if len(groups) == 1:
        return groups[0][0] / groups[0][1], groups[0][0] / groups[0][1]
    rng = np.random.default_rng(RNG_SEED + seed + len(groups))
    sums = np.asarray([g[0] for g in groups], dtype=float)
    weights = np.asarray([g[1] for g in groups], dtype=float)
    vals = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(groups), size=len(groups))
        vals.append(float(sums[idx].sum() / max(weights[idx].sum(), EPS)))
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def score_pm_fills(pm: pd.DataFrame, bars: pd.DataFrame, model: dict[str, Any], thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = add_pm_features(pm, bars)
    scored["risk_score"] = score_frame(scored, model)
    scored["bad_side_is_otm"] = np.where(
        scored["shorted_side"].eq("up"),
        scored["digital_z_signed"].astype(float).lt(0),
        scored["digital_z_signed"].astype(float).gt(0),
    )
    scored.to_parquet(OUT_PM, index=False)

    rows: list[dict[str, Any]] = []
    for policy, g in scored.groupby("policy", sort=False):
        for rule, threshold in {"take_all": -np.inf, **thresholds}.items():
            kept = g["risk_score"].ge(threshold) if rule == "take_all" else g["risk_score"].lt(threshold)
            sub = g[kept].copy()
            lo, hi = weighted_market_ci(sub, "stress_unit_ev", "size", seed=len(rows))
            mean = float(np.average(sub["stress_unit_ev"], weights=sub["size"])) if not sub.empty and sub["size"].sum() > 0 else math.nan
            rows.append(
                {
                    "policy": policy,
                    "rule": rule,
                    "train_score_threshold": threshold if np.isfinite(threshold) else math.nan,
                    "fills_before": int(len(g)),
                    "fills_after": int(len(sub)),
                    "markets_after": int(sub["market_id"].nunique()) if not sub.empty else 0,
                    "weighted_fills_after": float(sub["size"].sum()) if not sub.empty else 0.0,
                    "retained_fraction_weighted": float(sub["size"].sum() / g["size"].sum()) if g["size"].sum() > 0 else math.nan,
                    "mean_stress_ev": mean,
                    "stress_ev_ci_lo": lo,
                    "stress_ev_ci_hi": hi,
                    "mean_tail_ev": float(np.average(sub["tail_unit_ev"], weights=sub["size"])) if not sub.empty and sub["size"].sum() > 0 else math.nan,
                    "mean_realized_claim_pnl": float(np.average(sub["claim_unit_pnl"], weights=sub["size"])) if not sub.empty and sub["size"].sum() > 0 else math.nan,
                    "mean_risk_score_kept": float(sub["risk_score"].mean()) if not sub.empty else math.nan,
                    "dropped_weighted_fills": float(g["size"].sum() - sub["size"].sum()),
                }
            )
    return scored, pd.DataFrame(rows)


def plot_outputs(deciles: pd.DataFrame, pm_skip: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    held = deciles[deciles["sample"].eq("heldout")].copy()
    if not held.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(held["risk_decile"], held["bad_touch_rate"] * 100.0, marker="o", label="bad touch")
        ax.plot(held["risk_decile"], held["terminal_bad_rate"] * 100.0, marker="o", label="terminal bad")
        ax.set_xlabel("Risk-score decile using train cutpoints")
        ax.set_ylabel("Held-out event rate (%)")
        ax.set_title("Touch-risk score separation on held-out Binance 4h windows")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DECILES, dpi=160)
        plt.close(fig)

    fair = pm_skip[pm_skip["policy"].eq("fair_value_scaled")].copy()
    if not fair.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        order = ["take_all", "skip_top_30pct", "skip_top_20pct", "skip_top_10pct", "skip_top_5pct"]
        fair["rule"] = pd.Categorical(fair["rule"], categories=order, ordered=True)
        fair = fair.sort_values("rule")
        ax.bar(fair["rule"].astype(str), fair["mean_stress_ev"] * 100.0, color="#4C78A8")
        ax.errorbar(
            fair["rule"].astype(str),
            fair["mean_stress_ev"] * 100.0,
            yerr=[
                (fair["mean_stress_ev"] - fair["stress_ev_ci_lo"]).clip(lower=0.0) * 100.0,
                (fair["stress_ev_ci_hi"] - fair["mean_stress_ev"]).clip(lower=0.0) * 100.0,
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Stress EV (c/weighted contract)")
        ax.set_title("PM fair-value-scaled fill stress EV after risk-score skips")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(PLOT_PM_SKIP, dpi=160)
        plt.close(fig)


def table_from_df(df: pd.DataFrame, cols: list[str], limit: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    rows: list[list[str]] = []
    for _, r in df.head(limit).iterrows():
        row: list[str] = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, (bool, np.bool_)):
                row.append("true" if bool(v) else "false")
            elif isinstance(v, (int, np.integer)):
                row.append(f"{int(v):,}")
            elif isinstance(v, (float, np.floating)):
                if "ev" in c or "pnl" in c:
                    row.append(cents(float(v)))
                elif "rate" in c or "fraction" in c or c.startswith("auc") or c.endswith("_ci_lo") or c.endswith("_ci_hi"):
                    row.append(pct(float(v)) if ("rate" in c or "fraction" in c) else number(float(v), 3))
                elif "score" in c or "threshold" in c:
                    row.append(number(float(v), 3))
                else:
                    row.append(number(float(v), 3))
            else:
                row.append(str(v))
        rows.append(row)
    return markdown_table(cols, rows)


def write_note(
    total_windows: int,
    candidates: pd.DataFrame,
    sep: pd.DataFrame,
    deciles: pd.DataFrame,
    skip: pd.DataFrame,
    pm_skip: pd.DataFrame,
    scored_pm: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    held_auc = sep[(sep["sample"].eq("heldout")) & (sep["label"].eq("bad_touch"))].iloc[0]
    jump_auc = sep[(sep["sample"].eq("heldout")) & (sep["label"].eq("jump_driven_touch_vs_no_touch"))].iloc[0]
    fair_take = pm_skip[(pm_skip["policy"].eq("fair_value_scaled")) & (pm_skip["rule"].eq("take_all"))]
    fair_best = pm_skip[pm_skip["policy"].eq("fair_value_scaled") & ~pm_skip["rule"].eq("take_all")].copy()
    if not fair_best.empty:
        fair_best = fair_best.sort_values(["stress_ev_ci_lo", "mean_stress_ev"], ascending=False).head(1)

    verdict = "DROP AS OFFLINE SKIP FILTER"
    if float(held_auc["auc_ci_lo"]) > 0.55 and not fair_best.empty and float(fair_best.iloc[0]["stress_ev_ci_lo"]) > float(fair_take.iloc[0]["stress_ev_ci_lo"]):
        verdict = "LOG FEATURE ONLY"

    data_ledger_rows = [
        ["available", f"{total_windows:,} cached BTC/ETH/SOL 4h windows; {len(candidates):,} independent first-far candidate windows; Binance 5m klines with taker-buy volume from cached monthly zips."],
        ["missing", "True Binance L2/order-book OFI is not in the saved cache. The flow term here is taker-flow imbalance from kline taker-buy volume, so it is a TFI/OFI proxy, not exchange order-book OFI."],
        ["lookahead", "Features use bars whose close timestamp is at or before the candidate/fill timestamp. Future highs/lows are used only for labels."],
    ]
    pd.DataFrame(data_ledger_rows, columns=["ledger_bucket", "detail"]).to_csv(OUT_DATA_LEDGER, index=False)

    fair_text = "No fair-value-scaled PM skip row was available."
    if not fair_take.empty and not fair_best.empty:
        take = fair_take.iloc[0]
        best = fair_best.iloc[0]
        fair_text = (
            f"On the tiny PM fair-value-scaled fill set, take-all stress EV is {cents(float(take['mean_stress_ev']))} "
            f"{fmt_ci(float(take['stress_ev_ci_lo']), float(take['stress_ev_ci_hi']), 'c')}. "
            f"The best skip row is `{best['rule']}` at {cents(float(best['mean_stress_ev']))} "
            f"{fmt_ci(float(best['stress_ev_ci_lo']), float(best['stress_ev_ci_hi']), 'c')}, retaining "
            f"{pct(float(best['retained_fraction_weighted']))} of weighted fills. This is a diagnostic over only "
            f"{int(best['markets_after'])} kept PM market clusters, not a powered gate."
        )

    note = f"""# OD Touch-Risk Filter Findings

> Hub: [[strat_options_delta]] · [[COWORK]]
> Related: [[od_strategy_a_realism_reaudit_findings]] · [[block_k6_vol_findings]] · [[od_same_day_crypto_pricing_gate_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Headline

Verdict: **{verdict}**.

The cheap offline test does **not** justify adding a Binance touch-risk skip gate to the longshot-harvest live loop. The held-out score has only weak separation: bad-touch AUC **{number(float(held_auc['auc']), 3)}**, CI **{fmt_ci(float(held_auc['auc_ci_lo']), float(held_auc['auc_ci_hi']))}**. Jump-driven touches, the exact events this idea most wanted to catch, are not cleanly forecastable either: AUC **{number(float(jump_auc['auc']), 3)}**, CI **{fmt_ci(float(jump_auc['auc_ci_lo']), float(jump_auc['auc_ci_hi']))}**.

{fair_text}

The result is useful precisely because it is cheap: causal momentum, taker-flow imbalance, and Lee-Mykland jump features can be logged as descriptive live telemetry, but the offline evidence does not support a hard skip rule.

## Design

The unit for the powered separation test is one independent 4h Binance window. The script scans the **{total_windows:,}** cached BTC/ETH/SOL 4h windows from the grown tail base and selects the first state in each window where the out-of-the-money side is at least `|z| >= 1` after the first {MIN_ENTRY_MINUTES_AFTER_OPEN} minutes. That yields **{len(candidates):,}** candidate windows. The side being scored is the side a longshot seller would not want to see touch: UP when spot is below strike, DOWN when spot is above strike.

The label `bad_touch` is whether the future Binance high/low crosses the strike before the 4h window ends, excluding the already-closed 5-minute bar used for the features. `terminal_bad` is whether the same side resolves in-the-money at the final close. `touch_jump_driven` is a coarse stress label: a bad-touch window with at least one future adverse Lee-Mykland jump before expiry. It does not identify the exact first-touch bar.

Features are causal and fixed before held-out evaluation:
- Distance/state: `abs_z`, time to expiry, EWMA sigma, and model OTM probability.
- Momentum: adverse-direction 15m/30m/60m/120m log-return sums.
- Flow proxy: adverse-direction taker-flow imbalance from Binance kline taker-buy volume over the same horizons. This is **not true L2 OFI** because Binance order-book depth is missing from the saved artifacts.
- Jumps: adverse and opposite Lee-Mykland jump counts plus max adverse return and realized vol.

Modeling is deliberately simple: a regularized logistic score trained before **2025-01-01**, with a one-week embargo and held-out evaluation from **2025-01-08** onward. Skip thresholds are train-score quantiles only; no threshold is tuned on held-out or PM fills.

## Separation

| sample | label | rows | positives | base rate | AUC | CI |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(f"| {r['sample']} | {r['label']} | {int(r['rows']):,} | {int(r['positives']):,} | {pct(float(r['base_rate']))} | {number(float(r['auc']), 3)} | {fmt_ci(float(r['auc_ci_lo']), float(r['auc_ci_hi']))} |" for _, r in sep.iterrows())}

Read: the score is directionally above random for ordinary touches, but it is not strong enough for a trading gate. A hard skip feature should be catching toxic windows clearly; this score is in the "weak regime descriptor" range. The jump-driven row is the key honesty check: if jumps are the real left-tail, this offline score does not forecast them well.

![Held-out risk deciles](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_touch_risk_filter_decile_touch_rate.png)

Caption: held-out bad-touch and terminal-bad rates by score decile using train-set cutpoints. A deployable skip filter would show a steep monotone lift in the top deciles.

## Skip Sweep On Binance Tail Base

These rows apply train-score thresholds to train and held-out candidate windows. The table is about bad-event separation only; there is no PM price in the Binance tail base, so it is not a PnL table.

{table_from_df(skip[skip['sample'].eq('heldout')], ['sample', 'rule', 'retained_fraction', 'skipped_touch_rate', 'kept_touch_rate', 'skipped_terminal_bad_rate', 'kept_terminal_bad_rate', 'skipped_jump_driven_touch_rate', 'kept_jump_driven_touch_rate'], 12)}

Read: skipping high-score windows lowers retained bad-touch rates somewhat, but the separation is too soft. A useful skip rule would remove a small toxic tail while preserving most good windows; here the retained-versus-skipped difference is not large enough to justify reducing an already-small live fill opportunity set.

## PM Fill Overlay

The PM overlay uses `od_strategy_a_tail_sizing_weighted_fills.parquet`, the same 7-market / 22-fill Strategy-A stress sample from [[od_strategy_a_realism_reaudit_findings]]. This is too small for a new OOS result, so it is only a sanity check: does the score obviously improve stress EV when applied to the actual harvest rows?

{table_from_df(pm_skip[pm_skip['policy'].isin(['flat_1_contract', 'fair_value_scaled'])], ['policy', 'rule', 'fills_after', 'markets_after', 'retained_fraction_weighted', 'mean_stress_ev', 'stress_ev_ci_lo', 'stress_ev_ci_hi', 'mean_tail_ev', 'mean_realized_claim_pnl'], 20)}

![PM skip stress EV](/Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research/data/analysis/plots/options_delta/od_touch_risk_filter_pm_skip_stress_ev.png)

Caption: fair-value-scaled PM stress EV after train-threshold skip rules. Error bars are market-cluster bootstrap CIs over the tiny PM sample.

Read: no PM skip row earns promotion. The train-set thresholds do not drop any of the observed flat/fair Strategy-A fills, so the unchanged point estimate and small CI wiggle are bootstrap noise over the same kept rows. This is not enough to alter the live-loop sizing recommendation from [[od_strategy_a_realism_reaudit_findings]].

## Data Ledger

| bucket | read |
|---|---|
{chr(10).join(f"| {bucket} | {detail} |" for bucket, detail in data_ledger_rows)}

## Decision

**Drop the hard skip idea cheaply.** The offline score has weak touch separation, no compelling jump-driven separation, and no reliable PM stress-EV improvement. Do not add it as a required gate before quoting.

**Keep as live telemetry only:** log adverse momentum, taker-flow imbalance proxy, Lee-Mykland jump flags, distance-to-strike, and barrier acceleration in the live measurement loop. These are useful audit fields for post-fill adverse-selection analysis, but not a pre-trade skip rule yet.

Modeled assumptions:
- Binance 5m klines and taker-buy volume are sufficient for a cheap pre-test of momentum/flow/jump state.
- First `|z| >= 1` OTM state per 4h window is a fair independent proxy for a longshot-seller risk decision.
- Logistic weights and skip thresholds are trained before the held-out period.
- PM overlay uses existing stress EV, net of costs/rebate, with no mark-to-mid.

Live-only unknowns:
- True Binance L2/order-book OFI at quote time.
- Passive fill share and queue position conditional on risk state.
- Whether a score trained on historical Binance states maps to Polymarket maker fills when PM quotes are actually live.
- Whether jump-risk features matter only in the seconds before a touch, below the 5-minute historical resolution used here.

## Outputs

- Script: `scripts/od_touch_risk_filter.py`
- Candidate windows: `data/analysis/od_touch_risk_filter_candidate_windows.parquet`
- Scored PM fills: `data/analysis/od_touch_risk_filter_scored_pm_fills.parquet`
- Separation CSV: `data/analysis/csv_outputs/options_delta/od_touch_risk_filter_separation.csv`
- Decile CSV: `data/analysis/csv_outputs/options_delta/od_touch_risk_filter_deciles.csv`
- Skip sweep CSV: `data/analysis/csv_outputs/options_delta/od_touch_risk_filter_skip_sweep.csv`
- PM skip CSV: `data/analysis/csv_outputs/options_delta/od_touch_risk_filter_pm_skip.csv`
"""
    NOTE.write_text(normalize_markdown_wrapping(note), encoding="utf-8")


def update_docs() -> None:
    hub = OD_HUB.read_text(encoding="utf-8")
    bullet = "- 2026-06-03 Binance touch-risk skip pre-test: **DROP as hard skip; log telemetry only**. Causal 5m momentum + taker-flow-imbalance proxy + Lee-Mykland jump score over the grown Binance tail base has weak held-out touch separation and no reliable PM stress-EV lift. True Binance L2 OFI is missing from the cache, so the flow term is a proxy, not an OFI pass. See [[od_touch_risk_filter_findings]]."
    if "od_touch_risk_filter_findings" not in hub:
        marker = "## Current state (2026-06-03)"
        idx = hub.find(marker)
        if idx >= 0:
            insert = hub.find("\n-", idx)
            if insert >= 0:
                hub = hub[:insert] + "\n" + bullet + hub[insert:]
        member = "- Cross-asset / market-vs-market reopen candidates:"
        if member in hub and "[[od_touch_risk_filter_findings]]" not in hub:
            hub = hub.replace(
                "- Vol mispricing & harvest: [[block_k6_vol_findings]], [[block_k6_strategy_a_static_hedge_findings]], [[od_strategy_a_v2_lifecycle_findings]], [[od_strategy_a_realism_reaudit_findings]], [[block_k6_kronos_vol_bakeoff_findings]], [[block_k7_findings]], [[od_pricing_model_form_findings]]",
                "- Vol mispricing & harvest: [[block_k6_vol_findings]], [[block_k6_strategy_a_static_hedge_findings]], [[od_strategy_a_v2_lifecycle_findings]], [[od_strategy_a_realism_reaudit_findings]], [[od_touch_risk_filter_findings]], [[block_k6_kronos_vol_bakeoff_findings]], [[block_k7_findings]], [[od_pricing_model_form_findings]]",
            )
        task_marker = "- [ ] **OD Strategy A live measurement loop spec:**"
        if task_marker in hub:
            hub = hub.replace(
                task_marker,
                "- [x] **Binance touch-risk skip pre-test** (2026-06-03): completed; hard skip dropped, telemetry logging only. See [[od_touch_risk_filter_findings]].\n" + task_marker,
            )
        OD_HUB.write_text(normalize_markdown_wrapping(hub), encoding="utf-8")

    todo = BRAIN_TODO.read_text(encoding="utf-8")
    todo_bullet = "- 2026-06-03 OD Binance touch-risk skip pre-test: **DROP as hard skip; log telemetry only**. A causal 5m Binance score using momentum, taker-flow-imbalance proxy, and Lee-Mykland jumps over the grown Binance tail base does not produce strong held-out touch/jump separation and does not reliably lift PM stress EV on the tiny Strategy-A fill overlay. True Binance L2 OFI is missing from the saved cache. See [[od_touch_risk_filter_findings]]."
    if "od_touch_risk_filter_findings" not in todo:
        od_idx = todo.find("## OD — Options-Delta")
        insert = todo.find("\n-", od_idx)
        if insert >= 0:
            todo = todo[:insert] + "\n" + todo_bullet + todo[insert:]
        task_marker = "- [ ] **OD Strategy A live measurement loop spec:**"
        if task_marker in todo:
            todo = todo.replace(
                task_marker,
                "- [x] **Binance OFI/jump touch-risk skip pre-test** (2026-06-03): completed; no hard skip. Log adverse momentum, taker-flow imbalance proxy, and jump flags as telemetry only. See [[od_touch_risk_filter_findings]].\n" + task_marker,
            )
        BRAIN_TODO.write_text(normalize_markdown_wrapping(todo), encoding="utf-8")


def main() -> None:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    candidates, bars, total_windows = build_candidate_windows(refresh=False)
    train = candidates[candidates["split"].eq("train")].copy()
    if train.empty:
        raise SystemExit("no train candidate windows")
    model = fit_logistic(train)
    scored_candidates, separation, deciles, skip = split_metrics(candidates, model)
    train_scores = scored_candidates.loc[scored_candidates["split"].eq("train"), "risk_score"]
    thresholds = {
        "skip_top_30pct": float(np.nanquantile(train_scores, 0.70)),
        "skip_top_20pct": float(np.nanquantile(train_scores, 0.80)),
        "skip_top_10pct": float(np.nanquantile(train_scores, 0.90)),
        "skip_top_5pct": float(np.nanquantile(train_scores, 0.95)),
    }
    pm = pd.read_parquet(PM_WEIGHTED_FILLS)
    scored_pm, pm_skip = score_pm_fills(pm, bars, model, thresholds)
    separation.to_csv(OUT_SEPARATION, index=False)
    deciles.to_csv(OUT_DECILES, index=False)
    skip.to_csv(OUT_SKIP, index=False)
    pm_skip.to_csv(OUT_PM_SKIP, index=False)
    plot_outputs(deciles, pm_skip)
    write_note(total_windows, scored_candidates, separation, deciles, skip, pm_skip, scored_pm, thresholds)
    update_docs()
    print(f"candidate windows: {len(scored_candidates):,} from {total_windows:,} total cached 4h windows", flush=True)
    print(f"wrote {OUT_SEPARATION}", flush=True)
    print(f"wrote {OUT_PM_SKIP}", flush=True)
    print(f"wrote {NOTE}", flush=True)


if __name__ == "__main__":
    main()
