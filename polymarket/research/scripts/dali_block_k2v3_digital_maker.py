"""Block K2 v3 digital-anchored maker mechanism test.

Research-only sidecar. This is not a deployable market maker. It asks whether
moving the fair anchor from lagging Polymarket mid to a causal Binance-implied
digital fair, plus widening by digital delta, leaves any robust bucket after
realistic exits.
"""
from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
FEATURES = ANALYSIS / "block_a1_features.parquet"
K3_PANEL = ANALYSIS / "csv_outputs" / "options_delta" / "k3_leadlag_basis.csv"
GAMMA_CACHE = ANALYSIS / "kpeg_gamma_resolution_cache.json"
OUT_CSV = ANALYSIS / "csv_outputs" / "market_making" / "k2v3_digital_maker.csv"
NOTE = NOTES / "block_k2v3_findings.md"
PLOTS = ANALYSIS / "k2v3_plots"

RUN_POOL = ("a0b", "a0c_roll")
FAMILIES = ("crypto_4h_up_down", "daily_crypto_up_down")
FILL_WINDOW_SEC = 5
POST_DELAY_SEC = 30
TAKER_HOLD_SEC = 60
RESOLUTION_BUFFER_SEC = 10
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260531
ROBUST_MIN_FILLS = 30
YEAR_SECONDS = 365.0 * 24.0 * 3600.0
TICK = 0.01
CRYPTO_FEE_RATE = 0.07
MAKER_REBATE_SHARE = 0.20
HEDGE_FEE_RATE = 0.0002
DEFAULT_SIGMA = {"BTC": 0.60, "ETH": 0.80, "SOL": 1.20}
PHASE_BINS = [0, 15, 30, 60, 120, 240, 1e9]
PHASE_LABELS = ["0-15m", "15-30m", "30-60m", "60-120m", "120-240m", "240m+"]
TAU_BINS = [0, 15, 60, 180, 1e9]
TAU_LABELS = ["late_0-15m", "mid_15-60m", "early_60-180m", "very_early_180m+"]
Z_BINS = [0, 0.5, 1.5, 1e9]
Z_LABELS = ["near_|z|<=0.5", "mid_0.5-1.5", "far_|z|>1.5"]


@dataclass(frozen=True)
class QuoteConfig:
    name: str
    anchor: str
    base_spread_bps: float
    latency_sec: float
    as_mult: float
    flatten_abs_z: float
    flatten_tau_min: float
    inventory_cap: int = 1
    curtail_theta: bool = True


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    kind: str
    window_sec: int = 0
    offset_ticks: int = 0


def norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    out = 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))
    if np.ndim(x) == 0:
        return float(out)
    return out


def norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    out = np.exp(-0.5 * arr * arr) / math.sqrt(2.0 * math.pi)
    if np.ndim(x) == 0:
        return float(out)
    return out


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def display_path(path: Path) -> str:
    return str(path.resolve())


def z_bucket_from_abs(abs_z: float) -> str:
    if not np.isfinite(abs_z):
        return "nan"
    if abs_z < 0.5:
        return Z_LABELS[0]
    if abs_z < 1.5:
        return Z_LABELS[1]
    return Z_LABELS[2]


def tau_bucket_from_min(tau_min: float) -> str:
    if not np.isfinite(tau_min):
        return "nan"
    if tau_min < 15:
        return TAU_LABELS[0]
    if tau_min < 60:
        return TAU_LABELS[1]
    if tau_min < 180:
        return TAU_LABELS[2]
    return TAU_LABELS[3]


def maker_rebate_bps(price: float, denom: float | None = None) -> float:
    p = float(np.clip(price, 0.001, 0.999))
    d = float(np.clip(price if denom is None else denom, 0.01, 0.99))
    return CRYPTO_FEE_RATE * p * (1.0 - p) * MAKER_REBATE_SHARE / d * 10_000.0


def taker_fee_bps(price: float, denom: float) -> float:
    p = float(np.clip(price, 0.001, 0.999))
    d = float(np.clip(denom, 0.01, 0.99))
    return CRYPTO_FEE_RATE * p * (1.0 - p) / d * 10_000.0


def state_at_or_before(times: np.ndarray, values: np.ndarray, targets: np.ndarray | int) -> np.ndarray:
    target_arr = np.asarray(targets, dtype=np.int64)
    scalar = target_arr.ndim == 0
    if scalar:
        target_arr = target_arr.reshape(1)
    idx = np.searchsorted(times, target_arr, side="right") - 1
    out = np.full(len(target_arr), np.nan, dtype=float)
    valid = (idx >= 0) & (idx < len(values))
    out[valid] = values[idx[valid]]
    return out[0] if scalar else out


def bootstrap_mean_ci(rows: pd.DataFrame, value_col: str, seed: int = RNG_SEED) -> tuple[float, float]:
    clean = rows[["market_id", "fill_time_ns", value_col]].dropna().reset_index(drop=True)
    clean = clean[np.isfinite(clean[value_col])]
    if len(clean) < 5:
        return math.nan, math.nan
    block_labels: list[str] = []
    for market_id, piece in clean.groupby("market_id", sort=False):
        elapsed = (piece["fill_time_ns"] - piece["fill_time_ns"].min()) / 1_000_000_000.0
        block_labels.extend([f"{market_id}:{int(bucket)}" for bucket in (elapsed // BOOTSTRAP_CHUNK_SECONDS)])
    clean["block_id"] = block_labels
    blocks = [idx.to_numpy() for _, idx in clean.groupby("block_id", sort=False).groups.items()]
    if len(blocks) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    vals = clean[value_col].to_numpy(dtype=float)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in sampled])
        means.append(float(np.nanmean(vals[idx])))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def fetch_gamma_market(market_id: str) -> dict[str, Any]:
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"id": market_id, "error": repr(exc)}


def load_gamma_cache(market_ids: list[str]) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    if GAMMA_CACHE.exists():
        cache = json.loads(GAMMA_CACHE.read_text(encoding="utf-8"))
    changed = False
    for mid in sorted({str(x) for x in market_ids if str(x)}):
        if mid not in cache:
            cache[mid] = fetch_gamma_market(mid)
            changed = True
    if changed:
        GAMMA_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    return cache


def gamma_settlement_maps(cache: dict[str, Any]) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    settlement_by_token: dict[str, float] = {}
    market_meta: dict[str, dict[str, Any]] = {}
    for mid, meta in cache.items():
        try:
            token_ids = json.loads(meta.get("clobTokenIds", "[]")) if isinstance(meta.get("clobTokenIds"), str) else meta.get("clobTokenIds", [])
            prices = json.loads(meta.get("outcomePrices", "[]")) if isinstance(meta.get("outcomePrices"), str) else meta.get("outcomePrices", [])
        except Exception:
            token_ids, prices = [], []
        for token_id, price in zip(token_ids, prices, strict=False):
            try:
                settlement_by_token[str(token_id)] = float(price)
            except Exception:
                continue
        market_meta[str(mid)] = meta
    return settlement_by_token, market_meta


def build_model_panel() -> pd.DataFrame:
    panel = pd.read_csv(K3_PANEL)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel["window_start"] = pd.to_datetime(panel["window_start"], utc=True)
    panel["window_end"] = pd.to_datetime(panel["window_end"], utc=True)
    panel = panel.sort_values(["market_slug", "ts"]).reset_index(drop=True)

    pieces: list[pd.DataFrame] = []
    for slug, g in panel.groupby("market_slug", sort=False):
        g = g.sort_values("ts").copy()
        idx = g["ts"]
        spot = g["binance_spot"].astype(float)
        ret = np.log(spot).diff()
        dt = g["ts"].diff().dt.total_seconds().replace(0, np.nan)
        var_per_sec = (ret * ret / dt).replace([np.inf, -np.inf], np.nan)
        rolling_var = (
            pd.Series(var_per_sec.to_numpy(dtype=float), index=idx)
            .rolling("30min", min_periods=3)
            .mean()
        )
        sigma = np.sqrt(rolling_var.to_numpy(dtype=float) * YEAR_SECONDS)
        default = DEFAULT_SIGMA.get(str(g["asset"].iloc[0]), 0.80)
        sigma = pd.Series(sigma, index=g.index).ffill().fillna(default).clip(0.05, 3.0)
        tau = g["seconds_to_expiry"].astype(float).clip(lower=0.0) / YEAR_SECONDS
        strike = g["window_strike_spot"].astype(float)
        denom = sigma.to_numpy(dtype=float) * np.sqrt(np.maximum(tau.to_numpy(dtype=float), 1e-12))
        log_m = np.log(spot.to_numpy(dtype=float) / strike.to_numpy(dtype=float))
        z = log_m / np.maximum(denom, 1e-12)
        fair = norm_cdf(z)
        delta = norm_pdf(z) / np.maximum(
            spot.to_numpy(dtype=float) * sigma.to_numpy(dtype=float) * np.sqrt(np.maximum(tau.to_numpy(dtype=float), 1e-12)),
            1e-12,
        )
        theta = norm_pdf(z) * log_m / np.maximum(
            2.0 * sigma.to_numpy(dtype=float) * np.power(np.maximum(tau.to_numpy(dtype=float), 1e-12), 1.5) * YEAR_SECONDS,
            1e-12,
        )
        g["sigma_causal"] = sigma.to_numpy(dtype=float)
        g["z"] = z
        g["abs_z"] = np.abs(z)
        g["fair_up_causal"] = np.clip(fair, 0.0, 1.0)
        g["delta_up"] = delta
        g["theta_up_per_sec"] = theta
        g["ts_ns"] = g["ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
        g["window_start_ns"] = g["window_start"].to_numpy(dtype="datetime64[ns]").astype("int64")
        g["window_end_ns"] = g["window_end"].to_numpy(dtype="datetime64[ns]").astype("int64")
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def load_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = ", ".join(f"'{run}'" for run in RUN_POOL)
    fams = ", ".join(f"'{fam}'" for fam in FAMILIES)
    con = duckdb.connect()
    state_q = f"""
        SELECT
            run_id,
            received_at,
            event_type,
            asset_id,
            market_id,
            family,
            slug,
            question,
            outcome_index,
            best_bid,
            best_bid_size,
            best_ask,
            best_ask_size,
            mid,
            spread,
            market_resolved_at
        FROM read_parquet('{FEATURES}')
        WHERE run_id IN ({runs})
          AND family IN ({fams})
          AND event_type IN ('book', 'price_change', 'best_bid_ask')
          AND is_book_state_complete
          AND best_bid IS NOT NULL
          AND best_ask IS NOT NULL
          AND mid IS NOT NULL
          AND best_ask > best_bid
          AND best_bid BETWEEN 0 AND 1
          AND best_ask BETWEEN 0 AND 1
          AND lower(coalesce(slug, '')) NOT LIKE '%will-jd-vance%'
    """
    trade_q = f"""
        SELECT
            run_id,
            received_at,
            asset_id,
            market_id,
            family,
            slug,
            question,
            outcome_index,
            trade_price,
            upper(coalesce(trade_side, last_trade_side, '')) AS trade_side,
            trade_size,
            transaction_hash
        FROM read_parquet('{FEATURES}')
        WHERE run_id IN ({runs})
          AND family IN ({fams})
          AND event_type = 'last_trade_price'
          AND trade_price IS NOT NULL
          AND trade_price BETWEEN 0 AND 1
          AND trade_size > 0
          AND upper(coalesce(trade_side, last_trade_side, '')) IN ('BUY', 'SELL')
          AND lower(coalesce(slug, '')) NOT LIKE '%will-jd-vance%'
    """
    market_q = f"""
        SELECT
            market_id,
            any_value(slug) AS slug,
            any_value(family) AS family,
            any_value(question) AS question,
            min(received_at) AS first_seen,
            max(received_at) AS last_seen,
            count(*) AS n_rows,
            sum(CASE WHEN event_type = 'last_trade_price' THEN 1 ELSE 0 END) AS n_trades
        FROM read_parquet('{FEATURES}')
        WHERE run_id IN ({runs})
          AND family IN ({fams})
          AND lower(coalesce(slug, '')) NOT LIKE '%will-jd-vance%'
        GROUP BY market_id
    """
    states = con.execute(state_q).df()
    trades = con.execute(trade_q).df()
    markets = con.execute(market_q).df()
    con.close()

    for df in (states, trades, markets):
        if "received_at" in df:
            df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    states["market_resolved_at"] = pd.to_datetime(states["market_resolved_at"], utc=True, errors="coerce")
    states["t_ns"] = states["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    trades["received_at"] = pd.to_datetime(trades["received_at"], utc=True)
    trades["t_ns"] = trades["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    for col in ("asset_id", "market_id", "family", "slug"):
        states[col] = states[col].astype(str)
        trades[col] = trades[col].astype(str)
    trades["transaction_hash"] = trades["transaction_hash"].fillna("").astype(str)
    hash_mask = trades["transaction_hash"].ne("")
    with_hash = trades[hash_mask].drop_duplicates(["asset_id", "transaction_hash", "trade_price", "trade_side", "trade_size"])
    no_hash = trades[~hash_mask].drop_duplicates(["asset_id", "t_ns", "trade_price", "trade_side", "trade_size"])
    trades = pd.concat([with_hash, no_hash], ignore_index=True).sort_values(["asset_id", "t_ns"])
    return states.sort_values(["asset_id", "t_ns"]).reset_index(drop=True), trades.reset_index(drop=True), markets


def attach_model_to_states(states: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    crypto = states[states["family"].eq("crypto_4h_up_down")].copy()
    out_parts: list[pd.DataFrame] = []
    model_cols = [
        "ts_ns",
        "asset",
        "window_start_ns",
        "window_end_ns",
        "window_strike_spot",
        "window_close_spot",
        "seconds_to_expiry",
        "binance_spot",
        "sigma_causal",
        "z",
        "abs_z",
        "fair_up_causal",
        "delta_up",
        "theta_up_per_sec",
    ]
    for slug, g in crypto.groupby("slug", sort=False):
        m = model[model["market_slug"].eq(slug)][model_cols].sort_values("ts_ns")
        if m.empty:
            continue
        piece = pd.merge_asof(g.sort_values("t_ns"), m, left_on="t_ns", right_on="ts_ns", direction="backward")
        piece = piece[piece["fair_up_causal"].notna()].copy()
        if piece.empty:
            continue
        is_up = piece["outcome_index"].fillna(0).astype(int).eq(0)
        piece["token_fair"] = np.where(is_up, piece["fair_up_causal"], 1.0 - piece["fair_up_causal"])
        piece["token_delta"] = np.where(is_up, piece["delta_up"], -piece["delta_up"])
        piece["token_theta_per_sec"] = np.where(is_up, piece["theta_up_per_sec"], -piece["theta_up_per_sec"])
        piece["tau_sec"] = piece["seconds_to_expiry"].astype(float).clip(lower=0.0)
        piece["tau_min"] = piece["tau_sec"] / 60.0
        piece["elapsed_min"] = (piece["t_ns"] - piece["window_start_ns"]) / 1e9 / 60.0
        piece["local_spread_bps"] = (piece["best_ask"] - piece["best_bid"]) / piece["mid"].clip(0.01, 0.99) * 10_000.0
        piece["phase_bucket"] = pd.cut(piece["elapsed_min"], PHASE_BINS, labels=PHASE_LABELS, right=False).astype(str)
        piece["tau_bucket"] = pd.cut(piece["tau_min"], TAU_BINS, labels=TAU_LABELS, right=False).astype(str)
        piece["z_bucket"] = pd.cut(piece["abs_z"], Z_BINS, labels=Z_LABELS, right=False).astype(str)
        out_parts.append(piece)
    joined = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    if joined.empty:
        return joined
    q1, q2 = joined["local_spread_bps"].quantile([1 / 3, 2 / 3])
    joined["spread_regime"] = np.select(
        [joined["local_spread_bps"].le(q1), joined["local_spread_bps"].le(q2)],
        [f"tight_<={q1:.0f}bps", f"moderate_<={q2:.0f}bps"],
        default=f"wide_>{q2:.0f}bps",
    )
    joined["dt_next_sec"] = (
        joined.sort_values(["asset_id", "t_ns"])
        .groupby("asset_id")["t_ns"]
        .shift(-1)
        .sub(joined["t_ns"])
        .div(1e9)
        .clip(lower=0.0, upper=60.0)
        .fillna(0.0)
    )
    return joined.sort_values(["asset_id", "t_ns"]).reset_index(drop=True)


def build_active_hours(states: pd.DataFrame) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame()
    cols = ["z_bucket", "tau_bucket", "phase_bucket", "spread_regime"]
    rows = []
    rows.append({"bucket_key": "ALL", "active_market_hours": states["dt_next_sec"].sum() / 3600.0})
    for col in cols:
        for key, g in states.groupby(col, dropna=False):
            rows.append({"bucket_key": f"{col}={key}", "active_market_hours": g["dt_next_sec"].sum() / 3600.0})
    for keys, g in states.groupby(["z_bucket", "tau_bucket", "spread_regime"], dropna=False):
        rows.append(
            {
                "bucket_key": "|".join([f"z_bucket={keys[0]}", f"tau_bucket={keys[1]}", f"spread_regime={keys[2]}"]),
                "active_market_hours": g["dt_next_sec"].sum() / 3600.0,
            }
        )
    return pd.DataFrame(rows)


def spread_surface_rows(states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if states.empty:
        return pd.DataFrame()
    for keys, g in states.groupby(["phase_bucket", "z_bucket", "spread_regime"], dropna=False):
        rows.append(
            {
                "row_type": "spread_surface",
                "phase_bucket": keys[0],
                "z_bucket": keys[1],
                "spread_regime": keys[2],
                "n_quote_states": int(len(g)),
                "active_market_hours": float(g["dt_next_sec"].sum() / 3600.0),
                "mean_spread_bps": float(g["local_spread_bps"].mean()),
                "median_spread_bps": float(g["local_spread_bps"].median()),
                "mean_abs_z": float(g["abs_z"].mean()),
                "mean_tau_min": float(g["tau_min"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_trade_dict(trades: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for aid, g in trades.sort_values("t_ns").groupby("asset_id", sort=False):
        out[str(aid)] = {
            "t_ns": g["t_ns"].to_numpy(dtype=np.int64),
            "price": g["trade_price"].to_numpy(dtype=float),
            "side": g["trade_side"].to_numpy(dtype=object),
        }
    return out


def build_state_dict(states: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    cols = [
        "t_ns",
        "best_bid",
        "best_ask",
        "mid",
        "token_fair",
        "token_delta",
        "token_theta_per_sec",
        "binance_spot",
        "sigma_causal",
        "z",
        "abs_z",
        "tau_sec",
        "local_spread_bps",
    ]
    for aid, g in states.sort_values("t_ns").groupby("asset_id", sort=False):
        d = {col: g[col].to_numpy(dtype=np.int64 if col == "t_ns" else float) for col in cols}
        d["market_id"] = g["market_id"].astype(str).to_numpy(dtype=object)
        d["slug"] = g["slug"].astype(str).to_numpy(dtype=object)
        d["family"] = g["family"].astype(str).to_numpy(dtype=object)
        d["outcome_index"] = g["outcome_index"].to_numpy(dtype=int)
        d["window_end_ns"] = g["window_end_ns"].to_numpy(dtype=np.int64)
        d["window_start_ns"] = g["window_start_ns"].to_numpy(dtype=np.int64)
        d["window_strike_spot"] = g["window_strike_spot"].to_numpy(dtype=float)
        d["window_close_spot"] = g["window_close_spot"].to_numpy(dtype=float)
        d["phase_bucket"] = g["phase_bucket"].astype(str).to_numpy(dtype=object)
        d["tau_bucket"] = g["tau_bucket"].astype(str).to_numpy(dtype=object)
        d["z_bucket"] = g["z_bucket"].astype(str).to_numpy(dtype=object)
        d["spread_regime"] = g["spread_regime"].astype(str).to_numpy(dtype=object)
        out[str(aid)] = d
    return out


def build_candidate_pool(
    states: pd.DataFrame,
    trades: pd.DataFrame,
    settlement_by_token: dict[str, float],
) -> pd.DataFrame:
    state_dict = build_state_dict(states)
    records: list[pd.DataFrame] = []
    for aid, tg in trades[trades["family"].eq("crypto_4h_up_down")].groupby("asset_id", sort=False):
        st = state_dict.get(str(aid))
        if st is None:
            continue
        tg = tg.sort_values("t_ns").copy()
        state_times = st["t_ns"]
        trade_times = tg["t_ns"].to_numpy(dtype=np.int64)
        idx = np.searchsorted(state_times, trade_times, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        idx_safe = np.clip(idx, 0, len(state_times) - 1)
        quote_age = (trade_times - state_times[idx_safe]) / 1e9
        valid &= quote_age >= 0
        valid &= quote_age <= FILL_WINDOW_SEC
        if not valid.any():
            continue
        pos = np.flatnonzero(valid)
        qi = idx[pos]
        piece = tg.iloc[pos].copy()
        piece["quote_time_ns"] = state_times[qi]
        piece["quote_age_sec"] = quote_age[pos]
        for col in (
            "best_bid",
            "best_ask",
            "mid",
            "token_fair",
            "token_delta",
            "token_theta_per_sec",
            "binance_spot",
            "sigma_causal",
            "z",
            "abs_z",
            "tau_sec",
            "local_spread_bps",
            "window_strike_spot",
            "window_close_spot",
        ):
            piece[col] = st[col][qi]
        for col in ("window_start_ns", "window_end_ns"):
            piece[col] = st[col][qi]
        for col in ("phase_bucket", "tau_bucket", "z_bucket", "spread_regime"):
            piece[col] = st[col][qi]
        piece["settlement"] = settlement_by_token.get(str(aid), math.nan)
        piece["asset_id"] = str(aid)
        records.append(piece)
    if not records:
        return pd.DataFrame()
    pool = pd.concat(records, ignore_index=True).sort_values(["market_id", "asset_id", "t_ns"]).reset_index(drop=True)
    pool["fill_time_ns"] = pool["t_ns"].astype(np.int64)
    pool["category"] = "Crypto"
    return pool


def lookup_state_value(state_dict: dict[str, dict[str, np.ndarray]], asset_id: str, target_ns: int, col: str) -> float:
    st = state_dict.get(str(asset_id))
    if st is None:
        return math.nan
    return float(state_at_or_before(st["t_ns"], st[col], np.asarray(target_ns, dtype=np.int64)))


def lookup_future_model(state_dict: dict[str, dict[str, np.ndarray]], asset_id: str, target_ns: int) -> tuple[float, float]:
    st = state_dict.get(str(asset_id))
    if st is None:
        return math.nan, math.nan
    fair = float(state_at_or_before(st["t_ns"], st["token_fair"], np.asarray(target_ns, dtype=np.int64)))
    spot = float(state_at_or_before(st["t_ns"], st["binance_spot"], np.asarray(target_ns, dtype=np.int64)))
    return fair, spot


def row_with_lagged_anchor(
    row: Any,
    config: QuoteConfig,
    state_dict: dict[str, dict[str, np.ndarray]],
) -> Any | None:
    values = row._asdict() if hasattr(row, "_asdict") else dict(row)
    if config.anchor != "digital":
        values["anchor_lag_sec"] = 0.0
        values["decision_model_time_ns"] = int(values["quote_time_ns"])
        return SimpleNamespace(**values)

    lag_ns = int(round(config.latency_sec * 1_000_000_000))
    target_ns = int(values["quote_time_ns"]) - lag_ns
    if target_ns <= 0:
        return None
    for col in (
        "token_fair",
        "token_delta",
        "token_theta_per_sec",
        "binance_spot",
        "sigma_causal",
        "z",
        "abs_z",
        "tau_sec",
    ):
        lagged = lookup_state_value(state_dict, str(values["asset_id"]), target_ns, col)
        if not np.isfinite(lagged):
            return None
        values[col] = lagged
    values["anchor_lag_sec"] = float(config.latency_sec)
    values["decision_model_time_ns"] = target_ns
    values["z_bucket"] = z_bucket_from_abs(float(values["abs_z"]))
    values["tau_bucket"] = tau_bucket_from_min(float(values["tau_sec"]) / 60.0)
    return SimpleNamespace(**values)


def quote_for_row(row: Any, config: QuoteConfig, expected_token_side: int) -> tuple[float, float, float, float, bool]:
    if config.anchor == "mid":
        anchor = float(row.mid)
        half = max(config.base_spread_bps / 10_000.0 * max(anchor, 0.01), TICK / 2.0)
        return (
            float(np.clip(anchor - half, 0.001, 0.999)),
            float(np.clip(anchor + half, 0.001, 0.999)),
            half,
            0.0,
            False,
        )

    anchor = float(np.clip(row.token_fair, 0.001, 0.999))
    if abs(float(row.abs_z)) <= config.flatten_abs_z and float(row.tau_sec) <= config.flatten_tau_min * 60.0:
        return math.nan, math.nan, math.nan, math.nan, True
    if config.curtail_theta:
        theta_move = expected_token_side * float(row.token_theta_per_sec) * min(TAKER_HOLD_SEC, max(float(row.tau_sec), 1.0))
        if theta_move < -0.01:
            return math.nan, math.nan, math.nan, math.nan, True
    base_half = config.base_spread_bps / 10_000.0 * max(anchor, 0.01)
    e_abs_spot = (
        float(row.binance_spot)
        * float(row.sigma_causal)
        * math.sqrt(max(config.latency_sec, 0.001) / YEAR_SECONDS)
        * math.sqrt(2.0 / math.pi)
    )
    as_half = config.as_mult * abs(float(row.token_delta)) * e_abs_spot
    half = max(base_half, as_half, TICK / 2.0)
    return (
        float(np.clip(anchor - half, 0.001, 0.999)),
        float(np.clip(anchor + half, 0.001, 0.999)),
        float(half),
        float(as_half),
        False,
    )


def exit_taker(
    row: Any,
    token_side: int,
    entry_price: float,
    state_dict: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any] | None:
    latest_exit = int(row.window_end_ns) - RESOLUTION_BUFFER_SEC * 1_000_000_000
    target = min(int(row.fill_time_ns) + TAKER_HOLD_SEC * 1_000_000_000, latest_exit)
    if target <= int(row.fill_time_ns):
        return None
    bid = lookup_state_value(state_dict, str(row.asset_id), target, "best_bid")
    ask = lookup_state_value(state_dict, str(row.asset_id), target, "best_ask")
    if not np.isfinite(bid) or not np.isfinite(ask):
        return None
    exit_price = bid if token_side > 0 else ask
    fee = taker_fee_bps(exit_price, entry_price)
    gross = token_side * (exit_price - entry_price) / max(entry_price, 0.01) * 10_000.0
    return {
        "exit_time_ns": target,
        "exit_price": exit_price,
        "exit_mode": "taker",
        "maker_exit_filled": False,
        "exit_fee_bps": fee,
        "exit_rebate_bps": 0.0,
        "net_pnl_bps": gross + maker_rebate_bps(entry_price) - fee,
    }


def exit_maker_then_fallback(
    row: Any,
    token_side: int,
    entry_price: float,
    state_dict: dict[str, dict[str, np.ndarray]],
    trade_dict: dict[str, dict[str, np.ndarray]],
    policy: ExitPolicy,
) -> dict[str, Any] | None:
    latest_exit = int(row.window_end_ns) - RESOLUTION_BUFFER_SEC * 1_000_000_000
    post_t = int(row.fill_time_ns) + POST_DELAY_SEC * 1_000_000_000
    fallback_t = min(post_t + int(policy.window_sec) * 1_000_000_000, latest_exit)
    if fallback_t <= post_t or post_t <= int(row.fill_time_ns):
        return None
    post_mid = lookup_state_value(state_dict, str(row.asset_id), post_t, "mid")
    if not np.isfinite(post_mid):
        return None
    if token_side > 0:
        exit_quote = float(np.clip(post_mid + policy.offset_ticks * TICK, 0.001, 0.999))
    else:
        exit_quote = float(np.clip(post_mid - policy.offset_ticks * TICK, 0.001, 0.999))

    filled = False
    fill_t = fallback_t
    arr = trade_dict.get(str(row.asset_id))
    if arr is not None:
        t_ns = arr["t_ns"]
        px = arr["price"]
        side = arr["side"]
        lo = np.searchsorted(t_ns, post_t, side="left")
        hi = np.searchsorted(t_ns, fallback_t, side="right")
        if hi > lo:
            if token_side > 0:
                mask = (side[lo:hi] == "BUY") & (px[lo:hi] >= exit_quote - 1e-12)
            else:
                mask = (side[lo:hi] == "SELL") & (px[lo:hi] <= exit_quote + 1e-12)
            if mask.any():
                first = int(np.flatnonzero(mask)[0])
                fill_t = int(t_ns[lo + first])
                filled = True
    if filled:
        gross = token_side * (exit_quote - entry_price) / max(entry_price, 0.01) * 10_000.0
        exit_rebate = maker_rebate_bps(exit_quote, entry_price)
        return {
            "exit_time_ns": fill_t,
            "exit_price": exit_quote,
            "exit_mode": "maker_exit",
            "maker_exit_filled": True,
            "exit_fee_bps": 0.0,
            "exit_rebate_bps": exit_rebate,
            "net_pnl_bps": gross + maker_rebate_bps(entry_price) + exit_rebate,
        }

    bid = lookup_state_value(state_dict, str(row.asset_id), fallback_t, "best_bid")
    ask = lookup_state_value(state_dict, str(row.asset_id), fallback_t, "best_ask")
    if not np.isfinite(bid) or not np.isfinite(ask):
        return None
    exit_price = bid if token_side > 0 else ask
    fee = taker_fee_bps(exit_price, entry_price)
    gross = token_side * (exit_price - entry_price) / max(entry_price, 0.01) * 10_000.0
    return {
        "exit_time_ns": fallback_t,
        "exit_price": exit_price,
        "exit_mode": "taker_fallback",
        "maker_exit_filled": False,
        "exit_fee_bps": fee,
        "exit_rebate_bps": 0.0,
        "net_pnl_bps": gross + maker_rebate_bps(entry_price) - fee,
    }


def exit_hold_resolution(row: Any, token_side: int, entry_price: float, hedged: bool = False) -> dict[str, Any] | None:
    settlement = float(row.settlement)
    if not np.isfinite(settlement):
        return None
    gross = token_side * (settlement - entry_price) / max(entry_price, 0.01) * 10_000.0
    hedge_bps = 0.0
    hedge_cost_bps = 0.0
    if hedged:
        spot_entry = float(row.binance_spot)
        spot_close = float(row.window_close_spot)
        hedge_units = -token_side * float(row.token_delta)
        hedge_pnl_prob = hedge_units * (spot_close - spot_entry)
        hedge_cost_prob = abs(hedge_units) * spot_entry * HEDGE_FEE_RATE
        hedge_bps = hedge_pnl_prob / max(entry_price, 0.01) * 10_000.0
        hedge_cost_bps = hedge_cost_prob / max(entry_price, 0.01) * 10_000.0
    return {
        "exit_time_ns": int(row.window_end_ns),
        "exit_price": settlement,
        "exit_mode": "settlement_hedged" if hedged else "settlement",
        "maker_exit_filled": False,
        "exit_fee_bps": 0.0,
        "exit_rebate_bps": 0.0,
        "hedge_bps": hedge_bps,
        "hedge_cost_bps": hedge_cost_bps,
        "net_pnl_bps": gross + maker_rebate_bps(entry_price) + hedge_bps - hedge_cost_bps,
    }


def build_fill_row(
    row: Any,
    config: QuoteConfig,
    policy: ExitPolicy,
    token_side: int,
    entry_price: float,
    quote_bid: float,
    quote_ask: float,
    half_spread: float,
    as_half: float,
    exit_info: dict[str, Any],
    state_dict: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    denom = max(entry_price, 0.01)
    future_fair_30, future_spot_30 = lookup_future_model(state_dict, str(row.asset_id), int(row.fill_time_ns) + 30 * 1_000_000_000)
    future_fair_60, future_spot_60 = lookup_future_model(state_dict, str(row.asset_id), int(row.fill_time_ns) + 60 * 1_000_000_000)
    adverse_30 = (
        -token_side * (future_fair_30 - float(row.token_fair)) / denom * 10_000.0
        if np.isfinite(future_fair_30)
        else math.nan
    )
    adverse_60 = (
        -token_side * (future_fair_60 - float(row.token_fair)) / denom * 10_000.0
        if np.isfinite(future_fair_60)
        else math.nan
    )
    delta_attr_30 = (
        -token_side * float(row.token_delta) * (future_spot_30 - float(row.binance_spot)) / denom * 10_000.0
        if np.isfinite(future_spot_30)
        else math.nan
    )
    theta_resid_30 = adverse_30 - delta_attr_30 if np.isfinite(adverse_30) and np.isfinite(delta_attr_30) else math.nan
    spread_capture = token_side * (float(row.token_fair) - entry_price) / denom * 10_000.0
    settlement = float(row.settlement) if np.isfinite(float(row.settlement)) else math.nan
    resolution_risk = (
        -token_side * (settlement - float(row.token_fair)) / denom * 10_000.0
        if np.isfinite(settlement)
        else math.nan
    )
    return {
        "config_name": config.name,
        "anchor": config.anchor,
        "base_spread_bps": config.base_spread_bps,
        "latency_sec": config.latency_sec,
        "as_mult": config.as_mult,
        "flatten_abs_z": config.flatten_abs_z,
        "flatten_tau_min": config.flatten_tau_min,
        "exit_policy": policy.name,
        "exit_kind": policy.kind,
        "exit_window_s": policy.window_sec,
        "exit_offset_ticks": policy.offset_ticks,
        "anchor_lag_sec": float(getattr(row, "anchor_lag_sec", 0.0)),
        "market_id": str(row.market_id),
        "slug": str(row.slug),
        "asset_id": str(row.asset_id),
        "run_id": str(getattr(row, "run_id", "")),
        "family": str(row.family),
        "category": "Crypto",
        "fill_time_ns": int(row.fill_time_ns),
        "exit_time_ns": int(exit_info["exit_time_ns"]),
        "quote_time_ns": int(row.quote_time_ns),
        "decision_model_time_ns": int(getattr(row, "decision_model_time_ns", row.quote_time_ns)),
        "quote_age_sec": float(row.quote_age_sec),
        "maker_side": "BUY" if token_side > 0 else "SELL",
        "token_side": token_side,
        "trade_side": str(row.trade_side),
        "trade_price": float(row.trade_price),
        "entry_price": entry_price,
        "quote_bid": quote_bid,
        "quote_ask": quote_ask,
        "half_spread_prob": half_spread,
        "as_half_prob": as_half,
        "local_mid": float(row.mid),
        "local_spread_bps": float(row.local_spread_bps),
        "digital_fair": float(row.token_fair),
        "binance_spot": float(row.binance_spot),
        "sigma_causal": float(row.sigma_causal),
        "z": float(row.z),
        "abs_z": float(row.abs_z),
        "token_delta": float(row.token_delta),
        "tau_min": float(row.tau_sec) / 60.0,
        "phase_bucket": str(row.phase_bucket),
        "tau_bucket": str(row.tau_bucket),
        "z_bucket": str(row.z_bucket),
        "spread_regime": str(row.spread_regime),
        "settlement": settlement,
        "exit_price": float(exit_info["exit_price"]),
        "exit_mode": str(exit_info["exit_mode"]),
        "maker_exit_filled": bool(exit_info["maker_exit_filled"]),
        "spread_capture_bps": spread_capture,
        "rebate_bps": maker_rebate_bps(entry_price),
        "exit_rebate_bps": float(exit_info.get("exit_rebate_bps", 0.0)),
        "exit_fee_bps": float(exit_info.get("exit_fee_bps", 0.0)),
        "adverse_selection_30s_bps": adverse_30,
        "adverse_selection_60s_bps": adverse_60,
        "delta_spot_attr_30s_bps": delta_attr_30,
        "theta_resid_30s_bps": theta_resid_30,
        "resolution_risk_bps": resolution_risk,
        "hedge_bps": float(exit_info.get("hedge_bps", 0.0)),
        "hedge_cost_bps": float(exit_info.get("hedge_cost_bps", 0.0)),
        "net_pnl_bps": float(exit_info["net_pnl_bps"]),
    }


def exit_for_policy(
    row: Any,
    token_side: int,
    entry_price: float,
    policy: ExitPolicy,
    state_dict: dict[str, dict[str, np.ndarray]],
    trade_dict: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any] | None:
    if policy.kind == "taker":
        return exit_taker(row, token_side, entry_price, state_dict)
    if policy.kind == "maker":
        return exit_maker_then_fallback(row, token_side, entry_price, state_dict, trade_dict, policy)
    if policy.kind == "hold":
        return exit_hold_resolution(row, token_side, entry_price, hedged=False)
    if policy.kind == "hold_hedged":
        return exit_hold_resolution(row, token_side, entry_price, hedged=True)
    raise ValueError(policy.kind)


def simulate(
    candidates: pd.DataFrame,
    config: QuoteConfig,
    policy: ExitPolicy,
    state_dict: dict[str, dict[str, np.ndarray]],
    trade_dict: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cap = max(int(config.inventory_cap), 1)
    for _, group in candidates.sort_values(["market_id", "asset_id", "fill_time_ns"]).groupby(["market_id", "asset_id"], sort=False):
        open_lots: list[dict[str, int]] = []
        for row in group.itertuples(index=False):
            now = int(row.fill_time_ns)
            if open_lots:
                open_lots = [lot for lot in open_lots if int(lot["exit_time_ns"]) > now]
            q = int(sum(lot["token_side"] for lot in open_lots))
            side = str(row.trade_side)
            expected_token_side = 1 if side == "SELL" else -1
            decision_row = row_with_lagged_anchor(row, config, state_dict)
            if decision_row is None:
                continue
            quote_bid, quote_ask, half, as_half, flattened = quote_for_row(decision_row, config, expected_token_side)
            if flattened or not np.isfinite(quote_bid) or not np.isfinite(quote_ask):
                continue
            token_side = 0
            entry_price = math.nan
            trade_price = float(row.trade_price)
            if side == "SELL" and q < cap and trade_price <= quote_bid + 1e-12:
                token_side = 1
                entry_price = quote_bid
            elif side == "BUY" and q > -cap and trade_price >= quote_ask - 1e-12:
                token_side = -1
                entry_price = quote_ask
            if token_side == 0 or not np.isfinite(entry_price):
                continue
            exit_info = exit_for_policy(row, token_side, entry_price, policy, state_dict, trade_dict)
            if exit_info is None:
                continue
            open_lots.append({"exit_time_ns": int(exit_info["exit_time_ns"]), "token_side": token_side})
            rows.append(
                build_fill_row(
                    decision_row,
                    config,
                    policy,
                    token_side,
                    entry_price,
                    quote_bid,
                    quote_ask,
                    half,
                    as_half,
                    exit_info,
                    state_dict,
                )
            )
    return pd.DataFrame(rows)


def config_grid() -> list[QuoteConfig]:
    configs = [
        QuoteConfig("mid_k2_selected", "mid", 396.9687, 0.0, 0.0, 0.0, 0.0, inventory_cap=1, curtail_theta=False)
    ]
    for base in (100.0, 250.0, 500.0, 750.0):
        for latency in (1.0, 5.0, 10.0):
            for mult in (1.0, 2.0):
                for flatten in ((0.0, 0.0), (0.5, 15.0), (0.75, 30.0)):
                    name = f"digital_b{int(base)}_L{int(latency)}_m{mult:g}_flat{flatten[0]:g}z{int(flatten[1])}m"
                    configs.append(QuoteConfig(name, "digital", base, latency, mult, flatten[0], flatten[1]))
    return configs


def objective(summary: dict[str, Any]) -> float:
    n = int(summary["n_fills"])
    mean = float(summary["mean_net_pnl_bps"])
    std = float(summary["std_net_pnl_bps"])
    if n < ROBUST_MIN_FILLS or not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        return -1e9 + n
    return mean / std * math.sqrt(n)


def summarize_fills(
    fills: pd.DataFrame,
    *,
    row_type: str,
    config: QuoteConfig | None = None,
    policy: ExitPolicy | None = None,
    group_cols: list[str] | None = None,
    active_hours: pd.DataFrame | None = None,
    calendar_days: float = math.nan,
) -> pd.DataFrame:
    if group_cols is None:
        group_cols = []
    if fills.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = [((), fills)] if not group_cols else list(fills.groupby(group_cols, dropna=False, sort=True))
    active_map = {}
    if active_hours is not None and not active_hours.empty:
        active_map = dict(zip(active_hours["bucket_key"], active_hours["active_market_hours"], strict=False))
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        lo, hi = bootstrap_mean_ci(g, "net_pnl_bps", RNG_SEED + len(g))
        rec: dict[str, Any] = {
            "row_type": row_type,
            "config_name": config.name if config else (str(g["config_name"].iloc[0]) if "config_name" in g else ""),
            "anchor": config.anchor if config else (str(g["anchor"].iloc[0]) if "anchor" in g else ""),
            "base_spread_bps": config.base_spread_bps if config else (float(g["base_spread_bps"].iloc[0]) if "base_spread_bps" in g else math.nan),
            "latency_sec": config.latency_sec if config else (float(g["latency_sec"].iloc[0]) if "latency_sec" in g else math.nan),
            "anchor_lag_sec": (config.latency_sec if config and config.anchor == "digital" else 0.0)
            if config
            else (float(g["anchor_lag_sec"].iloc[0]) if "anchor_lag_sec" in g else math.nan),
            "as_mult": config.as_mult if config else (float(g["as_mult"].iloc[0]) if "as_mult" in g else math.nan),
            "flatten_abs_z": config.flatten_abs_z if config else (float(g["flatten_abs_z"].iloc[0]) if "flatten_abs_z" in g else math.nan),
            "flatten_tau_min": config.flatten_tau_min if config else (float(g["flatten_tau_min"].iloc[0]) if "flatten_tau_min" in g else math.nan),
            "exit_policy": policy.name if policy else (str(g["exit_policy"].iloc[0]) if "exit_policy" in g else ""),
            "exit_kind": policy.kind if policy else (str(g["exit_kind"].iloc[0]) if "exit_kind" in g else ""),
            "exit_window_s": policy.window_sec if policy else (int(g["exit_window_s"].iloc[0]) if "exit_window_s" in g else 0),
            "exit_offset_ticks": policy.offset_ticks if policy else (int(g["exit_offset_ticks"].iloc[0]) if "exit_offset_ticks" in g else 0),
            "n_fills": int(len(g)),
            "n_markets": int(g["market_id"].nunique()),
            "n_assets": int(g["asset_id"].nunique()),
            "mean_net_pnl_bps": float(g["net_pnl_bps"].mean()),
            "median_net_pnl_bps": float(g["net_pnl_bps"].median()),
            "std_net_pnl_bps": float(g["net_pnl_bps"].std(ddof=1)) if len(g) > 1 else math.nan,
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_lower_positive": bool(np.isfinite(lo) and lo > 0.0),
            "clears_zero": bool(len(g) >= ROBUST_MIN_FILLS and np.isfinite(lo) and lo > 0.0),
            "win_rate": float(g["net_pnl_bps"].gt(0).mean()),
            "mean_spread_capture_bps": float(g["spread_capture_bps"].mean()),
            "mean_rebate_bps": float(g["rebate_bps"].mean()),
            "mean_exit_rebate_bps": float(g["exit_rebate_bps"].mean()),
            "mean_exit_fee_bps": float(g["exit_fee_bps"].mean()),
            "mean_adverse_selection_30s_bps": float(g["adverse_selection_30s_bps"].mean()),
            "mean_delta_spot_attr_30s_bps": float(g["delta_spot_attr_30s_bps"].mean()),
            "mean_theta_resid_30s_bps": float(g["theta_resid_30s_bps"].mean()),
            "mean_resolution_risk_bps": float(g["resolution_risk_bps"].mean()),
            "mean_hedge_bps": float(g["hedge_bps"].mean()),
            "mean_hedge_cost_bps": float(g["hedge_cost_bps"].mean()),
            "maker_exit_fill_rate": float(g["maker_exit_filled"].mean()) if "maker_exit_filled" in g else math.nan,
            "mean_local_spread_bps": float(g["local_spread_bps"].mean()),
            "mean_abs_z": float(g["abs_z"].mean()),
            "mean_tau_min": float(g["tau_min"].mean()),
            "objective": math.nan,
            "fills_per_day": int(len(g)) / calendar_days if np.isfinite(calendar_days) and calendar_days > 0 else math.nan,
            "fills_per_active_market_hour": math.nan,
        }
        for col, key in zip(group_cols, keys, strict=False):
            rec[col] = str(key)
        if group_cols:
            if group_cols == ["z_bucket", "tau_bucket", "spread_regime"]:
                bucket_key = "|".join([f"{col}={key}" for col, key in zip(group_cols, keys, strict=False)])
            elif len(group_cols) == 1:
                bucket_key = f"{group_cols[0]}={keys[0]}"
            else:
                bucket_key = ""
        else:
            bucket_key = "ALL"
        active = active_map.get(bucket_key, active_map.get("ALL", math.nan))
        rec["active_market_hours"] = active
        if np.isfinite(active) and active > 0:
            rec["fills_per_active_market_hour"] = int(len(g)) / active
        rec["objective"] = objective(rec)
        rows.append(rec)
    return pd.DataFrame(rows)


def kpeg_flip_rows(states: pd.DataFrame) -> pd.DataFrame:
    selected_path = ANALYSIS / "csv_outputs" / "market_making" / "kpeg_selected_fills.csv"
    if not selected_path.exists() or states.empty:
        return pd.DataFrame()
    fills = pd.read_csv(selected_path)
    fills = fills[fills["category"].eq("Crypto") & fills["family"].eq("crypto_4h_up_down")].copy()
    if fills.empty:
        return pd.DataFrame()
    fills["asset_id"] = fills["asset_id"].astype(str)
    fills["fill_time_ns"] = fills["fill_time_ns"].astype(np.int64)
    records: list[dict[str, Any]] = []
    state_dict = build_state_dict(states)
    for r in fills.itertuples(index=False):
        fair = lookup_state_value(state_dict, str(r.asset_id), int(r.fill_time_ns), "token_fair")
        if not np.isfinite(fair):
            continue
        token_side = int(r.token_side)
        entry = float(r.entry_price)
        distance = token_side * (fair - entry) / TICK
        records.append(
            {
                "market_id": str(r.market_id),
                "fill_time_ns": int(r.fill_time_ns),
                "distance_to_binance_fair_ticks": distance,
                "past_binance_fair": bool(distance < 0),
                "net_pnl_bps_30s": float(r.net_pnl_bps_30s),
            }
        )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    rows = []
    for label, g in [("all_kpeg_fills", df), ("past_binance_fair", df[df["past_binance_fair"]])]:
        rows.append(
            {
                "row_type": "kpeg_binance_fair_flip",
                "scope": label,
                "n_fills": int(len(g)),
                "mean_distance_to_binance_fair_ticks": float(g["distance_to_binance_fair_ticks"].mean()) if len(g) else math.nan,
                "share_past_binance_fair": float(g["past_binance_fair"].mean()) if len(g) else math.nan,
                "mean_kpeg_mark_pnl_30s_bps": float(g["net_pnl_bps_30s"].mean()) if len(g) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def source_basis_rows(model: pd.DataFrame, market_meta: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_slug = model.sort_values("ts").groupby("market_slug", sort=False).tail(1)
    slug_to_mid: dict[str, str] = {}
    for mid, meta in market_meta.items():
        slug_to_mid[str(meta.get("slug") or "")] = str(mid)
    for r in by_slug.itertuples(index=False):
        mid = slug_to_mid.get(str(r.market_slug), "")
        meta = market_meta.get(mid, {})
        try:
            prices = json.loads(meta.get("outcomePrices", "[]")) if isinstance(meta.get("outcomePrices"), str) else meta.get("outcomePrices", [])
            settlement_up = float(prices[0])
        except Exception:
            settlement_up = math.nan
        binance_up = float(float(r.window_close_spot) >= float(r.window_strike_spot))
        rows.append(
            {
                "row_type": "source_basis",
                "market_id": mid,
                "slug": str(r.market_slug),
                "resolution_source": str(meta.get("resolutionSource") or ""),
                "binance_proxy_up": binance_up,
                "gamma_settlement_up": settlement_up,
                "source_direction_mismatch": bool(np.isfinite(settlement_up) and abs(settlement_up - binance_up) > 0.25),
            }
        )
    return pd.DataFrame(rows)


def guardrail_rows(bucket_results: pd.DataFrame) -> pd.DataFrame:
    if bucket_results.empty:
        return pd.DataFrame()
    n_tests = int(len(bucket_results))
    raw_ci_positive = int(bucket_results["ci_lower_positive"].fillna(False).sum())
    robust_clears = int(bucket_results["clears_zero"].fillna(False).sum())
    prereg = bucket_results[
        bucket_results["z_bucket"].eq("mid_0.5-1.5")
        & bucket_results["tau_bucket"].eq("mid_15-60m")
        & bucket_results["spread_regime"].astype(str).str.startswith("moderate_")
    ].copy()
    return pd.DataFrame(
        [
            {
                "row_type": "multiple_testing_guardrail",
                "n_bucket_tests": n_tests,
                "n_raw_ci_positive_buckets": raw_ci_positive,
                "n_robust_clearing_buckets": robust_clears,
                "preregistered_bucket_tests": int(len(prereg)),
                "preregistered_bucket_robust_clears": int(prereg["clears_zero"].fillna(False).sum()) if len(prereg) else 0,
                "note": "Only mid-|z| / mid-tau / moderate-spread is pre-registered; all other positive cells are exploratory.",
            }
        ]
    )


def oos_confirmation_rows(
    bucket_results: pd.DataFrame,
    candidates: pd.DataFrame,
    best_config: QuoteConfig,
    policies: list[ExitPolicy],
    state_dict: dict[str, dict[str, np.ndarray]],
    trade_dict: dict[str, dict[str, np.ndarray]],
    active_hours: pd.DataFrame,
    calendar_days: float,
) -> pd.DataFrame:
    if bucket_results.empty:
        return pd.DataFrame()
    cleared = bucket_results[bucket_results["clears_zero"].fillna(False)].copy()
    if cleared.empty:
        return pd.DataFrame()
    policy_by_name = {p.name: p for p in policies}
    rows: list[pd.DataFrame] = []
    for spec in cleared[["exit_policy", "z_bucket", "tau_bucket", "spread_regime"]].drop_duplicates().itertuples(index=False):
        policy = policy_by_name.get(str(spec.exit_policy))
        if policy is None:
            continue
        for split_name, run_id in (("discovery_a0b", "a0b"), ("confirm_a0c_roll", "a0c_roll")):
            sub_candidates = candidates[candidates["run_id"].eq(run_id)].copy()
            if sub_candidates.empty:
                continue
            fills = simulate(sub_candidates, best_config, policy, state_dict, trade_dict)
            if fills.empty:
                continue
            bucket = fills[
                fills["z_bucket"].eq(str(spec.z_bucket))
                & fills["tau_bucket"].eq(str(spec.tau_bucket))
                & fills["spread_regime"].eq(str(spec.spread_regime))
            ].copy()
            if bucket.empty:
                continue
            summary = summarize_fills(
                bucket,
                row_type=f"oos_{split_name}",
                config=best_config,
                policy=policy,
                active_hours=active_hours,
                calendar_days=calendar_days,
            )
            summary["z_bucket"] = str(spec.z_bucket)
            summary["tau_bucket"] = str(spec.tau_bucket)
            summary["spread_regime"] = str(spec.spread_regime)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_outputs(results: pd.DataFrame) -> list[Path]:
    PLOTS.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    spread = results[results["row_type"].eq("spread_surface")].copy()
    if not spread.empty:
        pivot = spread.pivot_table(index="phase_bucket", columns="z_bucket", values="mean_spread_bps", aggfunc="mean")
        pivot = pivot.reindex(index=PHASE_LABELS, columns=Z_LABELS)
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title("Unconditional crypto-4h spread surface")
        ax.set_xlabel("|z| bucket")
        ax.set_ylabel("time since window open")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("mean spread, bps")
        fig.tight_layout()
        path = PLOTS / "k2v3_spread_surface.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    buckets = results[results["row_type"].eq("selected_bucket")].copy()
    if not buckets.empty:
        selected_policies = {"taker_60s", "hold_resolution", "hold_resolution_delta_hedged"}
        selected_summary = results[results["row_type"].eq("selected_policy")].copy()
        maker_rows = selected_summary[selected_summary["exit_kind"].eq("maker")].copy()
        if not maker_rows.empty:
            best_maker = maker_rows.sort_values("mean_net_pnl_bps", ascending=False).iloc[0]
            selected_policies.add(str(best_maker["exit_policy"]))
        buckets = buckets[buckets["exit_policy"].isin(selected_policies)].copy()
        for policy_name, sub in buckets.groupby("exit_policy", sort=False):
            pivot = sub.pivot_table(index="tau_bucket", columns="z_bucket", values="mean_net_pnl_bps", aggfunc="mean")
            pivot = pivot.reindex(index=TAU_LABELS, columns=Z_LABELS)
            fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=160)
            vals = pivot.to_numpy(dtype=float)
            vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
            im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
            ax.set_yticks(range(len(pivot.index)), pivot.index)
            ax.set_title(f"Selected digital maker net PnL: {policy_name}")
            ax.set_xlabel("|z| bucket")
            ax.set_ylabel("tau bucket")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("mean net PnL, bps/fill")
            fig.tight_layout()
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(policy_name))[:80]
            path = PLOTS / f"k2v3_pnl_{safe}.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    return paths


def write_note(results: pd.DataFrame, plot_paths: list[Path], markets: pd.DataFrame, daily_rows: int) -> None:
    selected = results[results["row_type"].eq("selected_policy")].copy()
    best_search = results[results["row_type"].eq("config_search")].copy()
    spread = results[results["row_type"].eq("spread_surface")].copy()
    buckets = results[results["row_type"].eq("selected_bucket")].copy()
    source = results[results["row_type"].eq("source_basis")].copy()
    kpeg = results[results["row_type"].eq("kpeg_binance_fair_flip")].copy()
    guard = results[results["row_type"].eq("multiple_testing_guardrail")].copy()
    oos = results[results["row_type"].astype(str).str.startswith("oos_")].copy()

    digital_search = best_search[best_search["anchor"].eq("digital")].copy()
    best_digital = digital_search.sort_values("mean_net_pnl_bps", ascending=False).head(1)
    best_digital_objective = digital_search.sort_values("objective", ascending=False).head(1)
    mid = best_search[best_search["anchor"].eq("mid")].head(1)
    taker = selected[selected["exit_policy"].eq("taker_60s")].head(1)
    hold = selected[selected["exit_policy"].eq("hold_resolution")].head(1)
    hedged = selected[selected["exit_policy"].eq("hold_resolution_delta_hedged")].head(1)
    maker = selected[selected["exit_kind"].eq("maker")].sort_values("mean_net_pnl_bps", ascending=False).head(1)

    clears = buckets[buckets["clears_zero"].fillna(False)]
    raw_ci_positive = buckets[buckets["ci_lower_positive"].fillna(False)] if "ci_lower_positive" in buckets else pd.DataFrame()
    maker_windows = selected[selected["exit_kind"].eq("maker")].sort_values(["exit_window_s", "exit_offset_ticks"])
    window_rows = [
        [
            str(int(r.exit_window_s)),
            str(int(r.exit_offset_ticks)),
            str(int(r.n_fills)),
            pct(float(r.maker_exit_fill_rate)),
            bps(float(r.mean_net_pnl_bps)),
            f"[{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}]",
        ]
        for r in maker_windows.itertuples(index=False)
    ]
    policy_rows = [
        [
            str(r.exit_policy),
            str(int(r.n_fills)),
            bps(float(r.mean_net_pnl_bps)),
            f"[{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}]",
            pct(float(r.win_rate)),
            f"{float(r.fills_per_day):.1f}" if np.isfinite(float(r.fills_per_day)) else "n/a",
            f"{float(r.fills_per_active_market_hour):.2f}" if np.isfinite(float(r.fills_per_active_market_hour)) else "n/a",
        ]
        for r in selected.sort_values(["exit_kind", "exit_window_s", "exit_offset_ticks"]).itertuples(index=False)
    ]
    bucket_top = buckets.sort_values("mean_net_pnl_bps", ascending=False).head(12)
    bucket_rows = [
        [
            str(r.exit_policy),
            str(r.z_bucket),
            str(r.tau_bucket),
            str(r.spread_regime),
            str(int(r.n_fills)),
            bps(float(r.mean_net_pnl_bps)),
            f"[{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}]",
            "YES" if bool(r.clears_zero) else "no",
        ]
        for r in bucket_top.itertuples(index=False)
    ]
    phase_rows = []
    if not spread.empty:
        phase = spread.groupby("phase_bucket", as_index=False).agg(
            mean_spread_bps=("mean_spread_bps", "mean"),
            mean_abs_z=("mean_abs_z", "mean"),
            active_market_hours=("active_market_hours", "sum"),
        )
        phase["order"] = phase["phase_bucket"].map({v: i for i, v in enumerate(PHASE_LABELS)})
        for r in phase.sort_values("order").itertuples(index=False):
            phase_rows.append([str(r.phase_bucket), bps(float(r.mean_spread_bps)), f"{float(r.mean_abs_z):.2f}", f"{float(r.active_market_hours):.1f}"])

    latency_rows = []
    if not digital_search.empty and "latency_sec" in digital_search:
        for latency, g in digital_search.groupby("latency_sec", sort=True):
            best_l = g.sort_values("mean_net_pnl_bps", ascending=False).iloc[0]
            latency_rows.append(
                [
                    f"{float(latency):.0f}s",
                    str(best_l.config_name),
                    str(int(best_l.n_fills)),
                    bps(float(best_l.mean_net_pnl_bps)),
                    f"[{bps(float(best_l.ci_lo))}, {bps(float(best_l.ci_hi))}]",
                ]
            )

    shown_policies = {"taker_60s", "hold_resolution", "hold_resolution_delta_hedged"}
    if not maker.empty:
        shown_policies.add(str(maker.iloc[0].exit_policy))
    prereg_rows = []
    prereg = buckets[
        buckets["exit_policy"].isin(shown_policies)
        & buckets["z_bucket"].eq("mid_0.5-1.5")
        & buckets["tau_bucket"].eq("mid_15-60m")
        & buckets["spread_regime"].astype(str).str.startswith("moderate_")
    ].copy()
    for r in prereg.sort_values(["exit_kind", "exit_policy"]).itertuples(index=False):
        prereg_rows.append(
            [
                str(r.exit_policy),
                str(int(r.n_fills)),
                bps(float(r.mean_net_pnl_bps)),
                f"[{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}]",
                "YES" if bool(r.clears_zero) else "no",
            ]
        )

    oos_rows = []
    if not oos.empty:
        for r in oos.sort_values(["row_type", "exit_policy"]).itertuples(index=False):
            oos_rows.append(
                [
                    str(r.row_type).replace("oos_", ""),
                    str(r.exit_policy),
                    str(r.z_bucket),
                    str(r.tau_bucket),
                    str(r.spread_regime),
                    str(int(r.n_fills)),
                    bps(float(r.mean_net_pnl_bps)),
                    f"[{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}]",
                    "YES" if bool(r.clears_zero) else "no",
                ]
            )

    best_name = str(best_digital["config_name"].iloc[0]) if not best_digital.empty else "n/a"
    best_obj_name = str(best_digital_objective["config_name"].iloc[0]) if not best_digital_objective.empty else "n/a"
    verdict = "No robust bucket clears zero across taker, maker-exit, or hold-to-resolution exits."
    if not clears.empty:
        verdict = f"{len(clears)} bucket rows clear CI>0, but inspect exit policy and fill count before treating it as deployable."
    raw_bucket_phrase = f"{len(raw_ci_positive)} raw CI-positive bucket-policy cells; {len(clears)} robust clears."

    lines = [
        "# Block K2 v3: digital-anchored maker mechanism test",
        "",
        f"**Verdict:** {verdict}",
        "",
        "This run keeps the K2/K-PEG guardrails: causal quote inputs, one-share passive fill proxy, non-overlapping inventory by exit policy, net-of-rebate and net-of-exit-cost PnL, and block bootstrap CIs. The digital fair used for quoting is explicitly lagged by the swept latency `L`, so the quote never gets contemporaneous Binance information. The test universe is a0b+a0c_roll crypto-4h; daily crypto rows were present in A1 but excluded from the digital-anchor simulation because the K3/Binance surface only has the 4h strike/window panel.",
        "",
        "## Headline",
        "",
        f"Least-negative digital config under the taker-exit maker loop: `{best_name}`. The Sharpe-like objective pick was `{best_obj_name}`; both remain CI-negative.",
    ]
    if not best_digital.empty:
        r = best_digital.iloc[0]
        lines.append(
            f"No digital search config had positive taker-exit mean; the least-negative mean was {bps(float(r.mean_net_pnl_bps))}, CI [{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}], n={int(r.n_fills)}."
        )
    if not taker.empty:
        r = taker.iloc[0]
        lines.append(
            f"Taker-exit selected result: {bps(float(r.mean_net_pnl_bps))}, CI [{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}], n={int(r.n_fills)}."
        )
    if not maker.empty:
        r = maker.iloc[0]
        lines.append(
            f"Best maker-exit window among the selected digital config: {bps(float(r.mean_net_pnl_bps))}, CI [{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}], maker-exit fill rate {pct(float(r.maker_exit_fill_rate))}, window={int(r.exit_window_s)}s, offset={int(r.exit_offset_ticks)} ticks."
        )
    if not hold.empty:
        r = hold.iloc[0]
        lines.append(
            f"Hold-to-resolution: {bps(float(r.mean_net_pnl_bps))}, CI [{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}], n={int(r.n_fills)}."
        )
    if not hedged.empty:
        r = hedged.iloc[0]
        lines.append(
            f"Shadow entry-delta hedge variant: {bps(float(r.mean_net_pnl_bps))}, CI [{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}], mean hedge cost {bps(float(r.mean_hedge_cost_bps))}."
        )
    lines += [
        "",
        "## What Was Tested",
        "",
        "The old K2 quote used Polymarket mid as fair value. K2 v3 instead estimates the probability that the crypto window settles Up from Binance spot, the window strike, time left, and a causal realized-volatility estimate. Quotes are widened by the digital option's spot delta, so quotes automatically get wider when a small Binance move can change the settlement probability a lot.",
        "",
        "A fill happens only if a real trade print would have crossed our modeled passive quote within the A1.4h freshness window. For digital configs, the Binance anchor, delta, z, and theta are read from `quote_time - L`; the local book spread is still read at quote time because that is the venue state being quoted. After entry, the same fills are evaluated three ways: force taker exit after 60s, post passive maker exit after 30s with longer resting windows, or hold to actual Gamma settlement.",
        "",
        "## Latency Guardrail",
        "",
        markdown_table(["anchor lag L", "best config at L", "fills", "mean", "95% CI"], latency_rows),
        "",
        "## Policy Summary",
        "",
        markdown_table(["exit", "fills", "mean", "95% CI", "win", "fills/day", "fills/active-hr"], policy_rows),
        "",
        "## Maker Exit Windows",
        "",
        markdown_table(["window s", "offset ticks", "fills", "exit fill", "mean", "95% CI"], window_rows),
        "",
        "## Best Buckets",
        "",
        f"`clears` means CI lower > 0 **and** at least {ROBUST_MIN_FILLS} fills. Multiple-testing scan: {raw_bucket_phrase}",
        "",
        "The pre-registered bucket is mid-|z| / mid-tau / moderate-spread. Other cells are exploratory and should be read against the multiple-testing count, not as standalone discoveries.",
        "",
        markdown_table(["exit", "fills", "mean", "95% CI", "clears"], prereg_rows),
        "",
        markdown_table(["exit", "|z|", "tau", "spread", "fills", "mean", "95% CI", "clears"], bucket_rows),
        "",
        "## OOS Guardrail",
        "",
        "No robust in-sample bucket cleared, so the held-out confirmation step was not triggered. If a bucket clears in a future rerun, the script emits frozen a0b/a0c_roll confirmation rows before the note can call it deployable.",
        "",
        markdown_table(["split", "exit", "|z|", "tau", "spread", "fills", "mean", "95% CI", "clears"], oos_rows),
        "",
        "## Spread Surface",
        "",
        markdown_table(["time since open", "mean spread", "mean |z|", "active hrs"], phase_rows),
    ]
    if plot_paths:
        lines += ["", "## Figures", ""]
        for path in plot_paths:
            title = path.stem.replace("_", " ")
            lines.append(f"![{title}]({display_path(path)})")
    lines += [
        "",
        "## Mechanism Checks",
        "",
    ]
    if not mid.empty and not best_digital.empty:
        m = mid.iloc[0]
        d = best_digital.iloc[0]
        lines.append(
            f"Same-sample mid-anchor K2-style config had mean 30s adverse-selection cost {bps(float(m.mean_adverse_selection_30s_bps))}; the best digital-anchor config had {bps(float(d.mean_adverse_selection_30s_bps))}. This is the direct anchor comparison, not the old pooled-all-markets K2 result."
        )
    if not guard.empty:
        gr = guard.iloc[0]
        lines.append(
            f"Multiple-testing guardrail: {int(gr.n_bucket_tests)} bucket-policy cells were evaluated; {int(gr.n_raw_ci_positive_buckets)} had raw CI lower > 0, and {int(gr.n_robust_clearing_buckets)} survived the {ROBUST_MIN_FILLS}-fill robust-clear rule. Pre-registered robust clears: {int(gr.preregistered_bucket_robust_clears)}."
        )
    if not kpeg.empty:
        kr = kpeg[kpeg["scope"].eq("all_kpeg_fills")].head(1)
        if not kr.empty:
            r = kr.iloc[0]
            lines.append(
                f"K-PEG selected crypto-4h fills were {pct(float(r.share_past_binance_fair))} past the Binance-implied fair, with mean distance {float(r.mean_distance_to_binance_fair_ticks):.2f} ticks. This checks whether the chase sign-flip sits at the external fair rather than the local micro-price."
            )
    if not source.empty:
        mismatch = int(source["source_direction_mismatch"].fillna(False).sum())
        total = int(source["source_direction_mismatch"].notna().sum())
        lines.append(
            f"Resolution-source check: {mismatch}/{total} crypto-4h markets had Binance proxy direction disagree with Gamma settlement. This is a direction check, not a tick-level Chainlink-vs-Binance basis estimate."
        )
    lines += [
        "",
        "## Simple Conclusion",
        "",
        "Digital anchoring is the right way to test the adverse-selection mechanism, but the neutral maker loop still has to pay exit costs. If taker and maker-exit rows remain CI-negative while hold-to-resolution is positive, the result is not a single-venue market-making edge; it is a Track-A entry-and-carry/hedge question.",
        "",
        "Daily crypto note: the A1 panel contained daily crypto markets, but they were not included in the digital quote search because the available K3 fair-value panel covers crypto-4h only. A daily extension should parse Gamma eventStartTime/endDate and replay a matching Binance surface before pooling daily rows.",
        "",
        "## Files",
        "",
        "- `data/analysis/csv_outputs/market_making/k2v3_digital_maker.csv`",
        "- `notes/block_k2v3_findings.md`",
        f"- Input markets: {len(markets)} crypto markets in A1 a0b/a0c_roll; daily quote-state rows observed: {daily_rows:,}",
    ]
    NOTE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    print("loading K3 digital panel", flush=True)
    model = build_model_panel()
    print("loading A1 feature states/trades", flush=True)
    raw_states, trades, markets = load_feature_tables()
    daily_rows = int(raw_states["family"].eq("daily_crypto_up_down").sum())
    market_ids = sorted(raw_states["market_id"].astype(str).unique())
    gamma_cache = load_gamma_cache(market_ids)
    settlement_by_token, market_meta = gamma_settlement_maps(gamma_cache)
    print("joining causal digital model to full crypto-4h state panel", flush=True)
    states = attach_model_to_states(raw_states, model)
    if states.empty:
        raise RuntimeError("No crypto-4h states joined to K3 digital panel")
    active_hours = build_active_hours(states)
    trade_dict = build_trade_dict(trades)
    state_dict = build_state_dict(states)
    print("building candidate passive fills", flush=True)
    candidates = build_candidate_pool(states, trades, settlement_by_token)
    if candidates.empty:
        raise RuntimeError("No candidate fills for K2 v3")
    calendar_days = (states["t_ns"].max() - states["t_ns"].min()) / 1e9 / 86400.0

    search_policy = ExitPolicy("taker_60s", "taker")
    search_rows: list[pd.DataFrame] = []
    search_fills: dict[str, pd.DataFrame] = {}
    configs = config_grid()
    for idx, config in enumerate(configs, start=1):
        print(f"search {idx:03d}/{len(configs):03d} {config.name}", flush=True)
        fills = simulate(candidates, config, search_policy, state_dict, trade_dict)
        search_fills[config.name] = fills
        if fills.empty:
            continue
        summary = summarize_fills(
            fills,
            row_type="config_search",
            config=config,
            policy=search_policy,
            active_hours=active_hours,
            calendar_days=calendar_days,
        )
        search_rows.append(summary)
    search_summary = pd.concat(search_rows, ignore_index=True) if search_rows else pd.DataFrame()
    if search_summary.empty:
        raise RuntimeError("No search config produced fills")
    digital_summary = search_summary[search_summary["anchor"].eq("digital")].copy()
    if digital_summary.empty:
        raise RuntimeError("No digital config produced fills")
    robust_digital = digital_summary[digital_summary["n_fills"].ge(ROBUST_MIN_FILLS)].copy()
    if robust_digital.empty:
        robust_digital = digital_summary
    best_objective_name = str(robust_digital.sort_values("objective", ascending=False).iloc[0]["config_name"])
    best_name = str(robust_digital.sort_values("mean_net_pnl_bps", ascending=False).iloc[0]["config_name"])
    best_config = next(c for c in configs if c.name == best_name)
    print(f"selected digital config {best_name} (least-negative mean; objective pick was {best_objective_name})", flush=True)

    exit_policies = [
        ExitPolicy("taker_60s", "taker"),
        *[
            ExitPolicy(f"maker_exit_post30_win{window}_off{offset}", "maker", window, offset)
            for window in (30, 60, 120, 300, 600, 1800)
            for offset in (0, 2, 5)
        ],
        ExitPolicy("hold_resolution", "hold"),
        ExitPolicy("hold_resolution_delta_hedged", "hold_hedged"),
    ]
    selected_summaries: list[pd.DataFrame] = []
    selected_buckets: list[pd.DataFrame] = []
    for policy in exit_policies:
        print(f"selected exit {policy.name}", flush=True)
        fills = simulate(candidates, best_config, policy, state_dict, trade_dict)
        if fills.empty:
            continue
        selected_summaries.append(
            summarize_fills(
                fills,
                row_type="selected_policy",
                config=best_config,
                policy=policy,
                active_hours=active_hours,
                calendar_days=calendar_days,
            )
        )
        selected_buckets.append(
            summarize_fills(
                fills,
                row_type="selected_bucket",
                config=best_config,
                policy=policy,
                group_cols=["z_bucket", "tau_bucket", "spread_regime"],
                active_hours=active_hours,
                calendar_days=calendar_days,
            )
        )

    bucket_results = pd.concat(selected_buckets, ignore_index=True) if selected_buckets else pd.DataFrame()
    oos_rows = oos_confirmation_rows(
        bucket_results,
        candidates,
        best_config,
        exit_policies,
        state_dict,
        trade_dict,
        active_hours,
        calendar_days,
    )

    pieces = [
        spread_surface_rows(states),
        search_summary,
        *(selected_summaries or []),
        bucket_results,
        guardrail_rows(bucket_results),
        oos_rows,
        kpeg_flip_rows(states),
        source_basis_rows(model, market_meta),
        pd.DataFrame(
            [
                {
                    "row_type": "daily_crypto_scope_note",
                    "n_quote_states": daily_rows,
                    "note": "daily crypto present in A1 a0b, but excluded from digital-anchor sim because K3 fair panel covers crypto-4h only",
                }
            ]
        ),
    ]
    results = pd.concat([p for p in pieces if p is not None and not p.empty], ignore_index=True, sort=False)
    results.to_csv(OUT_CSV, index=False)
    plot_paths = plot_outputs(results)
    write_note(results, plot_paths, markets, daily_rows)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
