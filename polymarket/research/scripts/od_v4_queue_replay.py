"""OD v4 exploratory queue-aware replay.

This runs even when the Phase 0 calibration gate failed, but the outputs are
labelled exploratory. It uses captured quote states and subsequent same-token
trade flow to estimate passive ask fills for the rich-token short side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import cents, number, pct
from od_strategy_a_v3 import markdown_table, normalize_markdown_wrapping
from od_v4_calibration_gate import BOOTSTRAP_SAMPLES, RNG_SEED, ci_text


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
BRAIN_TODO = REPO / "brain" / "TODO.md"
OD_HUB = NOTES / "options_delta" / "strat_options_delta.md"

PANEL = ANALYSIS / "k6_vol_gap_panel.parquet"
A1_FEATURES = ANALYSIS / "block_a1_features.parquet"
PHASE0_SUMMARY = ANALYSIS / "csv_outputs" / "options_delta" / "od_v4_calibration_gate_summary.csv"
PHASE0_NOTE = NOTES / "options_delta" / "od_v4_calibration_gate_findings.md"

OUT_SUMMARY = ANALYSIS / "csv_outputs" / "options_delta" / "od_v4_queue_replay_summary.csv"
OUT_TRADES = ANALYSIS / "od_v4_queue_replay_trades.parquet"
OUT_CANDIDATES = ANALYSIS / "od_v4_queue_replay_candidates.parquet"
PLOTS = ANALYSIS / "plots" / "options_delta"
PLOT = PLOTS / "od_v4_queue_replay_summary.png"
NOTE = NOTES / "options_delta" / "od_v4_queue_replay_findings.md"

TICK = 0.01
QUOTE_SIZE = 1.0
NON_TOP3_AVAILABLE_SHARE = 0.05
FEE_RATE_CRYPTO = 0.07
MAKER_REBATE_PCT = 0.20
MM_CRYPTO_STRUCTURAL_NON_TOP3_LOWER_BPS = 21.8


@dataclass(frozen=True)
class ReplayConfig:
    required_edge: float
    improve_ticks: int
    wait_sec: int
    toxicity: str
    dollar_delta_cap: float

    @property
    def config_id(self) -> str:
        cap = "inf" if not np.isfinite(self.dollar_delta_cap) else str(int(self.dollar_delta_cap))
        return f"edge_{int(round(self.required_edge * 100)):02d}c_imp_{self.improve_ticks}t_wait_{self.wait_sec}s_tox_{self.toxicity}_cap_{cap}"


DECISION_STEP_SEC = 15

CONFIGS = [
    ReplayConfig(edge, improve, wait, tox, cap)
    for edge in (0.00, 0.01, 0.02, 0.05)
    for improve in (0, 1)
    for wait in (30, 300)
    for tox in ("none", "basic")
    for cap in (math.inf, 50.0)
]


def fmt_usd(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"${value:,.2f}"


def maker_rebate(price: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(price, 0.01, 0.99)
    return MAKER_REBATE_PCT * FEE_RATE_CRYPTO * p * (1.0 - p)


def bootstrap_ci_by_market(df: pd.DataFrame, col: str, seed_offset: int = 0) -> tuple[float, float]:
    if df.empty:
        return math.nan, math.nan
    vals = df.groupby("market_id", sort=False)[col].sum().replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(RNG_SEED + seed_offset + len(vals))
    idx = rng.integers(0, len(vals), size=(BOOTSTRAP_SAMPLES, len(vals)))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def market_equal_mean(df: pd.DataFrame, col: str) -> float:
    if df.empty:
        return math.nan
    vals = df.groupby("market_id", sort=False)[col].sum().to_numpy(dtype=float)
    return float(np.nanmean(vals)) if len(vals) else math.nan


def load_base_candidates() -> pd.DataFrame:
    con = duckdb.connect()
    query = f"""
    WITH panel AS (
        SELECT
            CAST(market_id AS VARCHAR) AS market_id,
            market_slug,
            asset,
            ts,
            source_run,
            CASE WHEN source_run = 'a0c_roll' THEN 'oos_holdout' ELSE 'is_discovery' END AS sample_split,
            window_start,
            window_end,
            seconds_to_expiry,
            abs_z,
            moneyness_bucket,
            time_bucket,
            state_bucket,
            source_ok_strict,
            toxic_near_expiry,
            p_model,
            digital_delta,
            binance_spot,
            up_ask,
            down_ask,
            chainlink_resolution_up
        FROM read_parquet('{PANEL.as_posix()}')
        WHERE source_ok_strict
          AND abs_z >= 1.0
          AND NOT toxic_near_expiry
    ),
    token_map AS (
        SELECT
            CAST(market_id AS VARCHAR) AS market_id,
            CAST(outcome_index AS INTEGER) AS outcome_index,
            any_value(asset_id) AS asset_id
        FROM read_parquet('{A1_FEATURES.as_posix()}')
        WHERE family = 'crypto_4h_up_down'
          AND asset_id IS NOT NULL
          AND asset_id <> ''
          AND outcome_index IN (0, 1)
        GROUP BY 1, 2
    ),
    tokens AS (
        SELECT
            p.*,
            0 AS outcome_index,
            'up' AS token_side,
            p.p_model AS token_fair,
            p.up_ask AS panel_token_ask,
            CASE WHEN p.chainlink_resolution_up THEN 1.0 ELSE 0.0 END AS token_payoff
        FROM panel p
        UNION ALL
        SELECT
            p.*,
            1 AS outcome_index,
            'down' AS token_side,
            1.0 - p.p_model AS token_fair,
            p.down_ask AS panel_token_ask,
            CASE WHEN p.chainlink_resolution_up THEN 0.0 ELSE 1.0 END AS token_payoff
        FROM panel p
    ),
    mapped AS (
        SELECT t.*, m.asset_id
        FROM tokens t
        JOIN token_map m
          ON t.market_id = m.market_id
         AND t.outcome_index = m.outcome_index
    )
    SELECT
        mapped.*,
        b.received_at AS book_ts,
        b.best_bid,
        b.best_bid_size,
        b.best_ask,
        b.best_ask_size,
        b.spread,
        b.mid AS book_mid,
        b.tob_imbalance,
        b.book_imbalance_top_n,
        b.bid_top5_shares,
        b.ask_top5_shares,
        b.signed_trade_size_60s,
        b.signed_trade_size_300s,
        b.ofi_bid_60s,
        b.ofi_ask_60s,
        b.ofi_combined_60s,
        b.mid_change_60s,
        b.is_book_state_complete
    FROM mapped
    ASOF LEFT JOIN read_parquet('{A1_FEATURES.as_posix()}') b
      ON mapped.asset_id = b.asset_id
     AND mapped.ts >= b.received_at
    ORDER BY mapped.market_id, mapped.outcome_index, mapped.ts
    """
    df = con.execute(query).fetchdf()
    con.close()
    if df.empty:
        raise SystemExit("no OD v4 queue candidates after base join")
    for col in ("ts", "book_ts", "window_start", "window_end"):
        df[col] = pd.to_datetime(df[col], utc=True)
    numeric = [
        "seconds_to_expiry",
        "abs_z",
        "p_model",
        "digital_delta",
        "binance_spot",
        "token_fair",
        "panel_token_ask",
        "token_payoff",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "book_mid",
        "bid_top5_shares",
        "ask_top5_shares",
        "signed_trade_size_60s",
        "signed_trade_size_300s",
        "ofi_bid_60s",
        "ofi_ask_60s",
        "ofi_combined_60s",
        "mid_change_60s",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["state_id"] = np.arange(len(df), dtype=np.int64)
    df["ts_ns"] = df["ts"].map(lambda x: pd.Timestamp(x).value).astype(np.int64)
    df["sample_split"] = df["sample_split"].astype(str)
    return df


def load_buy_trades(asset_ids: set[str]) -> dict[str, pd.DataFrame]:
    con = duckdb.connect()
    ids = pd.DataFrame({"asset_id": sorted(asset_ids)})
    con.register("ids", ids)
    trades = con.execute(
        f"""
        SELECT
            CAST(f.asset_id AS VARCHAR) AS asset_id,
            f.received_at,
            f.trade_price,
            f.trade_size
        FROM read_parquet('{A1_FEATURES.as_posix()}') f
        JOIN ids i ON CAST(f.asset_id AS VARCHAR) = i.asset_id
        WHERE f.is_trade = 1
          AND upper(coalesce(f.trade_side, f.last_trade_side, '')) = 'BUY'
          AND f.trade_size IS NOT NULL
          AND f.trade_price IS NOT NULL
        ORDER BY f.asset_id, f.received_at
        """
    ).fetchdf()
    con.close()
    if trades.empty:
        return {}
    trades["received_at"] = pd.to_datetime(trades["received_at"], utc=True)
    trades["ts_ns"] = trades["received_at"].map(lambda x: pd.Timestamp(x).value).astype(np.int64)
    trades["trade_price"] = pd.to_numeric(trades["trade_price"], errors="coerce")
    trades["trade_size"] = pd.to_numeric(trades["trade_size"], errors="coerce").fillna(0.0)
    return {asset: g.sort_values("ts_ns").reset_index(drop=True) for asset, g in trades.groupby("asset_id", sort=False)}


def future_buy_flow(row: pd.Series, trades_by_asset: dict[str, pd.DataFrame], wait_sec: int, quote_price: float) -> float:
    trades = trades_by_asset.get(str(row["asset_id"]))
    if trades is None or trades.empty:
        return 0.0
    times = trades["ts_ns"].to_numpy(dtype=np.int64)
    start = int(row["ts_ns"])
    end = start + wait_sec * 1_000_000_000
    lo = np.searchsorted(times, start, side="right")
    hi = np.searchsorted(times, end, side="right")
    if hi <= lo:
        return 0.0
    sl = trades.iloc[lo:hi]
    crossed = sl["trade_price"].to_numpy(dtype=float) >= quote_price - 1e-12
    if not crossed.any():
        return 0.0
    return float(sl.loc[crossed, "trade_size"].sum())


def add_future_buy_flow(df: pd.DataFrame, trades_by_asset: dict[str, pd.DataFrame], wait_sec: int) -> pd.Series:
    flows = np.zeros(len(df), dtype=float)
    if df.empty:
        return pd.Series(flows, index=df.index)
    positions = {idx: pos for pos, idx in enumerate(df.index)}
    for asset_id, idx in df.groupby("asset_id", sort=False).groups.items():
        trades = trades_by_asset.get(str(asset_id))
        if trades is None or trades.empty:
            continue
        trade_times = trades["ts_ns"].to_numpy(dtype=np.int64)
        trade_prices = trades["trade_price"].to_numpy(dtype=float)
        trade_sizes = trades["trade_size"].to_numpy(dtype=float)
        sub = df.loc[idx]
        cand_times = sub["ts_ns"].to_numpy(dtype=np.int64)
        quote_prices = sub["quote_price"].to_numpy(dtype=float)
        for local_pos, (start, quote_price) in enumerate(zip(cand_times, quote_prices, strict=True)):
            end = int(start) + wait_sec * 1_000_000_000
            lo = np.searchsorted(trade_times, int(start), side="right")
            hi = np.searchsorted(trade_times, end, side="right")
            if hi <= lo:
                continue
            crossed = trade_prices[lo:hi] >= quote_price - 1e-12
            if crossed.any():
                flows[positions[sub.index[local_pos]]] = float(trade_sizes[lo:hi][crossed].sum())
    return pd.Series(flows, index=df.index)


def valid_toxicity(row: pd.Series, toxicity: str) -> bool:
    if toxicity == "none":
        return True
    if not bool(row.get("is_book_state_complete", False)):
        return False
    best_bid = float(row.get("best_bid", math.nan))
    best_ask = float(row.get("best_ask", math.nan))
    spread = float(row.get("spread", math.nan))
    bid_top5 = float(row.get("bid_top5_shares", math.nan))
    ask_top5 = float(row.get("ask_top5_shares", math.nan))
    signed_60 = float(row.get("signed_trade_size_60s", 0.0) or 0.0)
    ofi_ask_60 = float(row.get("ofi_ask_60s", 0.0) or 0.0)
    if not (np.isfinite(best_bid) and np.isfinite(best_ask) and best_ask > best_bid):
        return False
    if np.isfinite(spread) and spread > 0.25:
        return False
    if not (np.isfinite(bid_top5) and np.isfinite(ask_top5) and bid_top5 >= 1.0 and ask_top5 >= 1.0):
        return False
    if signed_60 > 3.0 * max(1.0, ask_top5):
        return False
    if ofi_ask_60 < -3.0 * max(1.0, ask_top5):
        return False
    return True


def toxicity_mask(df: pd.DataFrame, toxicity: str) -> pd.Series:
    if toxicity == "none":
        return pd.Series(True, index=df.index)
    best_bid = df["best_bid"].astype(float)
    best_ask = df["best_ask"].astype(float)
    spread = df["spread"].astype(float)
    bid_top5 = df["bid_top5_shares"].astype(float)
    ask_top5 = df["ask_top5_shares"].astype(float)
    signed_60 = df["signed_trade_size_60s"].fillna(0.0).astype(float)
    ofi_ask_60 = df["ofi_ask_60s"].fillna(0.0).astype(float)
    complete = df["is_book_state_complete"].fillna(False).astype(bool)
    return (
        complete
        & best_bid.notna()
        & best_ask.notna()
        & best_ask.gt(best_bid)
        & (spread.isna() | spread.le(0.25))
        & bid_top5.ge(1.0)
        & ask_top5.ge(1.0)
        & signed_60.le(3.0 * np.maximum(1.0, ask_top5))
        & ofi_ask_60.ge(-3.0 * np.maximum(1.0, ask_top5))
    )


def prepare_config_candidates(base: pd.DataFrame, cfg: ReplayConfig, trades_by_asset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = base.copy()
    df = df[df["panel_token_ask"].between(0.01, 0.99) & df["best_bid"].notna() & df["best_ask"].notna()].copy()
    if df.empty:
        return df
    base_quote = df["best_ask"].astype(float) - cfg.improve_ticks * TICK
    min_passive = df["best_bid"].astype(float) + TICK
    fair_floor = df["token_fair"].astype(float) + cfg.required_edge
    df["quote_price"] = np.maximum.reduce([base_quote.to_numpy(dtype=float), min_passive.to_numpy(dtype=float), fair_floor.to_numpy(dtype=float)])
    df["quote_price"] = np.round(df["quote_price"] / TICK) * TICK
    df["quote_edge"] = df["quote_price"] - df["token_fair"].astype(float)
    df = df[
        df["quote_price"].between(0.01, 0.99)
        & df["quote_edge"].ge(cfg.required_edge - 1e-12)
        & df["quote_price"].gt(df["best_bid"].astype(float) + 1e-12)
    ].copy()
    if df.empty:
        return df
    df["toxicity_ok"] = toxicity_mask(df, cfg.toxicity)
    df = df[df["toxicity_ok"]].copy()
    if df.empty:
        return df

    touch = np.isclose(df["quote_price"], df["best_ask"].astype(float), atol=0.00001)
    inside = df["quote_price"].astype(float) < df["best_ask"].astype(float) - 1e-12
    df["queue_ahead"] = np.where(inside, 0.0, np.where(touch, df["best_ask_size"].fillna(0.0), df["ask_top5_shares"].fillna(df["best_ask_size"]).fillna(0.0)))
    df["future_buy_flow"] = add_future_buy_flow(df, trades_by_asset, cfg.wait_sec)
    df["raw_fill_units"] = np.minimum(QUOTE_SIZE, np.maximum(0.0, df["future_buy_flow"].astype(float) - df["queue_ahead"].astype(float)))
    df = df[df["raw_fill_units"].gt(1e-9)].copy()
    if df.empty:
        return df
    df["maker_rebate"] = maker_rebate(df["quote_price"].to_numpy(dtype=float))
    df["unit_resolution_pnl"] = df["quote_price"].astype(float) - df["token_payoff"].astype(float) + df["maker_rebate"].astype(float)
    df["unit_model_ev"] = df["quote_price"].astype(float) - df["token_fair"].astype(float) + df["maker_rebate"].astype(float)
    sign = np.where(df["outcome_index"].astype(int).eq(0), -1.0, 1.0)
    df["unit_signed_dollar_delta"] = sign * df["digital_delta"].astype(float) * df["binance_spot"].astype(float)
    df["unit_capital"] = np.maximum(0.01, 1.0 - df["quote_price"].astype(float))
    return df.sort_values(["market_id", "outcome_index", "ts"]).reset_index(drop=True)


def select_sequential_fills(candidates: pd.DataFrame, cfg: ReplayConfig) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    rows: list[pd.Series] = []
    for (_, outcome_index), group in candidates.groupby(["market_id", "outcome_index"], sort=False):
        next_allowed = -9_223_372_036_854_775_000
        running_delta_by_market: dict[str, float] = {}
        for _, row in group.sort_values("ts_ns").iterrows():
            ts_ns = int(row["ts_ns"])
            if ts_ns < next_allowed:
                continue
            market_id = str(row["market_id"])
            units = float(row["raw_fill_units"])
            if np.isfinite(cfg.dollar_delta_cap):
                current = running_delta_by_market.get(market_id, 0.0)
                unit_delta = float(row["unit_signed_dollar_delta"])
                proposed = current + unit_delta * units
                if abs(proposed) > cfg.dollar_delta_cap:
                    remaining = max(0.0, cfg.dollar_delta_cap - abs(current))
                    units = 0.0 if abs(unit_delta) <= 1e-12 else min(units, remaining / abs(unit_delta))
                if units <= 1e-9:
                    next_allowed = ts_ns + cfg.wait_sec * 1_000_000_000
                    continue
                running_delta_by_market[market_id] = current + unit_delta * units
            out = row.copy()
            out["filled_units"] = units
            rows.append(out)
            next_allowed = ts_ns + cfg.wait_sec * 1_000_000_000
    if not rows:
        return candidates.iloc[0:0].copy()
    out = pd.DataFrame(rows)
    out["config_id"] = cfg.config_id
    out["required_edge"] = cfg.required_edge
    out["improve_ticks"] = cfg.improve_ticks
    out["wait_sec"] = cfg.wait_sec
    out["toxicity"] = cfg.toxicity
    out["dollar_delta_cap"] = cfg.dollar_delta_cap
    out["net_pnl"] = out["filled_units"].astype(float) * out["unit_resolution_pnl"].astype(float)
    out["model_ev"] = out["filled_units"].astype(float) * out["unit_model_ev"].astype(float)
    out["capital_at_risk"] = out["filled_units"].astype(float) * out["unit_capital"].astype(float)
    out["realistic_units_after_top3"] = out["filled_units"].astype(float) * NON_TOP3_AVAILABLE_SHARE
    out["realistic_net_pnl_after_top3"] = out["net_pnl"].astype(float) * NON_TOP3_AVAILABLE_SHARE
    out["realistic_model_ev_after_top3"] = out["model_ev"].astype(float) * NON_TOP3_AVAILABLE_SHARE
    out["realistic_capital_after_top3"] = out["capital_at_risk"].astype(float) * NON_TOP3_AVAILABLE_SHARE
    return out.reset_index(drop=True)


def run_replay(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["decision_slot"] = (base["ts_ns"] // (DECISION_STEP_SEC * 1_000_000_000)).astype(np.int64)
    base = base.drop_duplicates(["market_id", "outcome_index", "decision_slot"], keep="first").reset_index(drop=True)
    print(
        f"decision-cadence candidates={len(base):,} at {DECISION_STEP_SEC}s cadence",
        flush=True,
    )
    trades_by_asset = load_buy_trades(set(base["asset_id"].dropna().astype(str)))
    all_rows: list[pd.DataFrame] = []
    for i, cfg in enumerate(CONFIGS, start=1):
        cand = prepare_config_candidates(base, cfg, trades_by_asset)
        fills = select_sequential_fills(cand, cfg)
        if not fills.empty:
            all_rows.append(fills)
        if i % 8 == 0:
            print(f"processed {i}/{len(CONFIGS)} configs", flush=True)
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True, sort=False)


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame(rows)
    for keys, group in trades.groupby(["config_id", "sample_split"], sort=False):
        config_id, sample_split = keys
        for label, metric_col, cap_col, units_col in [
            ("queue_adjusted", "net_pnl", "capital_at_risk", "filled_units"),
            ("queue_adjusted_after_top3_haircut", "realistic_net_pnl_after_top3", "realistic_capital_after_top3", "realistic_units_after_top3"),
        ]:
            g = group.copy()
            market = (
                g.groupby("market_id", as_index=False)
                .agg(
                    net_pnl=(metric_col, "sum"),
                    model_ev=("model_ev", "sum"),
                    capital=(cap_col, "sum"),
                    filled_units=(units_col, "sum"),
                    window_start=("window_start", "first"),
                    window_end=("window_end", "first"),
                    asset=("asset", "first"),
                )
            )
            if market.empty:
                continue
            lo, hi = bootstrap_ci_by_market(g, metric_col, seed_offset=hash(config_id + sample_split + label) % 10000)
            roc = market["net_pnl"].to_numpy(dtype=float) / np.maximum(1e-9, market["capital"].to_numpy(dtype=float))
            roc_lo, roc_hi = (math.nan, math.nan)
            if len(roc):
                rng = np.random.default_rng(RNG_SEED + 915 + len(roc))
                idx = rng.integers(0, len(roc), size=(BOOTSTRAP_SAMPLES, len(roc)))
                boot = roc[idx].mean(axis=1)
                roc_lo, roc_hi = np.nanquantile(boot, [0.025, 0.975])
            span_days = max((pd.to_datetime(market["window_end"]).max() - pd.to_datetime(market["window_start"]).min()).total_seconds() / 86400.0, 1 / 86400.0)
            mean_net = float(market["net_pnl"].mean())
            rows.append(
                {
                    "config_id": config_id,
                    "sample_split": sample_split,
                    "metric_scope": label,
                    "n_markets": int(market["market_id"].nunique()),
                    "n_quote_fills": int(len(g)),
                    "filled_units": float(market["filled_units"].sum()),
                    "mean_net_pnl_per_market": mean_net,
                    "median_net_pnl_per_market": float(market["net_pnl"].median()),
                    "net_ci_lo": lo,
                    "net_ci_hi": hi,
                    "win_rate_market": float((market["net_pnl"] > 0).mean()),
                    "mean_model_ev_per_market": float(market["model_ev"].mean()),
                    "total_net_pnl": float(market["net_pnl"].sum()),
                    "total_capital": float(market["capital"].sum()),
                    "mean_roc": float(np.nanmean(roc)) if len(roc) else math.nan,
                    "roc_ci_lo": float(roc_lo),
                    "roc_ci_hi": float(roc_hi),
                    "daily_pnl": float(market["net_pnl"].sum() / span_days),
                    "lower_roc_bps_minus_mm_structural": float(roc_lo * 10_000.0 - MM_CRYPTO_STRUCTURAL_NON_TOP3_LOWER_BPS) if np.isfinite(roc_lo) else math.nan,
                    "required_edge": float(group["required_edge"].iloc[0]),
                    "improve_ticks": int(group["improve_ticks"].iloc[0]),
                    "wait_sec": int(group["wait_sec"].iloc[0]),
                    "toxicity": str(group["toxicity"].iloc[0]),
                    "dollar_delta_cap": float(group["dollar_delta_cap"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def add_incremental(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    baseline = out[out["required_edge"].eq(0.0)].copy()
    if baseline.empty:
        out["same_config_lift_vs_edge0"] = math.nan
        return out
    lift = []
    for _, row in out.iterrows():
        b = baseline[
            baseline["sample_split"].eq(row["sample_split"])
            & baseline["metric_scope"].eq(row["metric_scope"])
            & baseline["improve_ticks"].eq(row["improve_ticks"])
            & baseline["wait_sec"].eq(row["wait_sec"])
            & baseline["toxicity"].eq(row["toxicity"])
            & baseline["dollar_delta_cap"].eq(row["dollar_delta_cap"])
        ]
        if b.empty:
            lift.append(math.nan)
        else:
            lift.append(float(row["mean_net_pnl_per_market"] - b.iloc[0]["mean_net_pnl_per_market"]))
    out["same_config_lift_vs_edge0"] = lift
    return out


def make_plot(summary: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    sub = summary[
        summary["sample_split"].eq("oos_holdout")
        & summary["metric_scope"].eq("queue_adjusted_after_top3_haircut")
    ].copy()
    sub = sub.sort_values(["net_ci_lo", "mean_net_pnl_per_market"], ascending=[False, False]).head(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    if not sub.empty:
        labels = [
            f"{int(r.required_edge*100)}c/{int(r.improve_ticks)}t/{int(r.wait_sec)}s/{r.toxicity}/cap{'inf' if not np.isfinite(r.dollar_delta_cap) else int(r.dollar_delta_cap)}"
            for _, r in sub.iterrows()
        ]
        x = np.arange(len(sub))
        y = sub["mean_net_pnl_per_market"].to_numpy(dtype=float)
        yerr = np.vstack([y - sub["net_ci_lo"].to_numpy(dtype=float), sub["net_ci_hi"].to_numpy(dtype=float) - y])
        ax.bar(x, y, color="#2c7fb8")
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_ylabel("Mean net PnL per market after top-3 haircut ($)")
    ax.set_title("OD v4 exploratory queue replay: best OOS configurations")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=160)
    plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str], limit: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    rows = []
    piece = df.head(limit).copy()
    for _, r in piece.iterrows():
        row = []
        for col in cols:
            v = r.get(col, math.nan)
            if col in {"required_edge", "mean_net_pnl_per_market", "median_net_pnl_per_market", "net_ci_lo", "net_ci_hi", "mean_model_ev_per_market", "same_config_lift_vs_edge0"}:
                if col == "net_ci_lo":
                    row.append(ci_text(float(r["net_ci_lo"]), float(r["net_ci_hi"])))
                elif col == "required_edge":
                    row.append(cents(float(v)))
                else:
                    row.append(cents(float(v)))
            elif col in {"win_rate_market"}:
                row.append(pct(float(v)))
            elif col in {"mean_roc", "roc_ci_lo"}:
                if col == "roc_ci_lo":
                    row.append(ci_text(float(r["roc_ci_lo"]), float(r["roc_ci_hi"]), unit="pct"))
                else:
                    row.append(pct(float(v)))
            elif col in {"daily_pnl", "total_net_pnl", "total_capital"}:
                row.append(fmt_usd(float(v)))
            elif col in {"n_markets", "n_quote_fills", "wait_sec", "improve_ticks"}:
                row.append(str(int(v)))
            elif col == "filled_units":
                row.append(number(float(v), 2))
            elif col == "dollar_delta_cap":
                row.append("none" if not np.isfinite(float(v)) else f"${int(v)}")
            else:
                row.append(str(v))
        rows.append(row)
    return markdown_table(cols, rows)


def write_note(base: pd.DataFrame, trades: pd.DataFrame, summary: pd.DataFrame) -> None:
    phase0_text = "Phase 0 summary unavailable."
    if PHASE0_SUMMARY.exists():
        phase0 = pd.read_csv(PHASE0_SUMMARY)
        row = phase0[phase0["label"].eq("full_far_strict_rich_short")]
        if not row.empty:
            r = row.iloc[0]
            phase0_text = (
                f"Phase 0 failed: {int(r['fills'])} fills / {int(r['markets'])} markets, gross EV {cents(float(r['mean_gross_ev']))}, "
                f"CI {ci_text(float(r['gross_ev_ci_lo']), float(r['gross_ev_ci_hi']))}, realized ITM {pct(float(r['realized_itm_rate']))}."
            )

    best = summary[
        summary["sample_split"].eq("oos_holdout")
        & summary["metric_scope"].eq("queue_adjusted_after_top3_haircut")
    ].sort_values(["net_ci_lo", "mean_net_pnl_per_market"], ascending=[False, False])
    best_row = best.iloc[0] if not best.empty else None
    headline = "No OOS queue-adjusted fills survived." if best_row is None else (
        f"Best exploratory OOS row after the incumbent haircut: {cents(float(best_row['mean_net_pnl_per_market']))} mean per market, "
        f"CI {ci_text(float(best_row['net_ci_lo']), float(best_row['net_ci_hi']))}, "
        f"{int(best_row['n_markets'])} markets, {number(float(best_row['filled_units']), 2)} expected filled contracts."
    )
    pass_text = "FAIL"
    if best_row is not None and float(best_row["net_ci_lo"]) > 0 and float(best_row["lower_roc_bps_minus_mm_structural"]) > 0:
        pass_text = "EXPLORATORY PASS"

    top_oos = best.copy().head(12)
    top_queue = summary[
        summary["sample_split"].eq("oos_holdout")
        & summary["metric_scope"].eq("queue_adjusted")
    ].sort_values(["net_ci_lo", "mean_net_pnl_per_market"], ascending=[False, False]).head(8)
    pooled = summary[
        summary["sample_split"].eq("is_discovery")
        & summary["metric_scope"].eq("queue_adjusted_after_top3_haircut")
    ].sort_values(["net_ci_lo", "mean_net_pnl_per_market"], ascending=[False, False]).head(8)

    note = f"""# OD v4 Exploratory Queue Replay: Can Passive Rich-Short Quotes Actually Fill?

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Gate note: [[od_v4_calibration_gate_findings]]
> MM benchmark: [[mm_deployable_cells_findings]] · [[block_k5_stress_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Headline

Official gate context: {phase0_text}

User override: this queue replay was run anyway as an exploratory falsifier. It is **not** a reversal of the Phase 0 gate.

Exploratory Phase 1 verdict: **{pass_text}**.

{headline}

Crucial read: the best deployable row uses `required_edge = 0c`. The 1c OD-richness variants are still positive in cents, but their `lift vs edge0` is negative. In other words, queue-aware execution found a tiny source/structure/capacity result, not evidence that the OD fair-value richness filter adds independent edge.

Plain-English read: queue realism does not create a new edge by itself. It changes which rich-short quotes plausibly fill, how much capacity survives the queue, and how much survives the incumbent-maker haircut. Because Phase 0 did not prove the primitive `price - realized ITM` edge, this note should be read as "what would execution look like if we pursued it anyway?"

## What This Replay Does

For every captured crypto-4h quote state in the K6/K3 panel, the replay asks whether the token at the ask is rich versus OD fair value. If it is rich enough, we try to passively **sell the rich token** at an ask quote that obeys the fair bound:

```text
quote price >= OD token fair + required edge
quote price > current best bid
```

Then the replay looks forward inside the captured LOB stream. A fill is counted only if later same-token BUY trade flow reaches our quote after consuming queue ahead:

```text
filled units = min(order size, max(0, future BUY flow at/above quote - queue ahead))
```

This is stricter than the v3 capacity proxy because it uses future trade flow to consume visible queue ahead. It is still a proxy: queue identity, hidden cancels, self queue rank, and true Polymarket matching-engine priority are not observable.

Sample construction: the base join produced {len(base):,} far/source quote states. The replay then uses one decision per market/token every {DECISION_STEP_SEC} seconds, leaving {trades[['market_id', 'outcome_index', 'config_id']].drop_duplicates().shape[0] if not trades.empty else 0:,} market-token-config combinations with at least one queue-adjusted fill. The cadence prevents the same future taker flow from being counted against many near-identical one-second quote states.

## Config Columns

`required_edge` is the minimum cents above OD fair demanded before selling the token. `improve_ticks` says how many 1c ticks we improve from the existing ask while staying passive and inside the fair bound. `wait_sec` is how long the quote rests before cancel/requote. `toxicity=basic` skips incomplete books, one-sided thin books, very wide spreads, and recent buy-flow / ask-depletion pressure that would be adverse for a seller. `dollar_delta_cap` clips expected fills when the market episode's running dollar-delta exposure gets too large.

`queue_adjusted` is the raw queue model. `queue_adjusted_after_top3_haircut` scales units and PnL by 5%, matching the K5 reality that top-3 maker wallets took about 95% of positive crypto-4h maker profit.

![Queue replay OOS summary]({PLOT.resolve()})

## Best OOS Rows After Incumbent Haircut

Unit of observation: one market episode. PnL is resolution PnL, not mark-to-mid. CI is bootstrapped by market.

| config_id | markets | quote fills | filled units | edge | improve ticks | wait sec | toxicity | cap | mean net | median net | CI | win | mean ROC | ROC CI | daily PnL | lift vs edge0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(md_table(top_oos, ['config_id','n_markets','n_quote_fills','filled_units','required_edge','improve_ticks','wait_sec','toxicity','dollar_delta_cap','mean_net_pnl_per_market','median_net_pnl_per_market','net_ci_lo','win_rate_market','mean_roc','roc_ci_lo','daily_pnl','same_config_lift_vs_edge0']).splitlines()[2:])}

Read: this is the deployability view. A positive row here means the replay found plausible queue fills and then applied the non-incumbent capacity haircut. A lower CI above zero is encouraging only as an execution proxy; it cannot override Phase 0 calibration failure.

## Raw Queue-Adjusted OOS Rows Before Incumbent Haircut

| config_id | markets | quote fills | filled units | edge | improve ticks | wait sec | toxicity | cap | mean net | median net | CI | win | total capital |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(md_table(top_queue, ['config_id','n_markets','n_quote_fills','filled_units','required_edge','improve_ticks','wait_sec','toxicity','dollar_delta_cap','mean_net_pnl_per_market','median_net_pnl_per_market','net_ci_lo','win_rate_market','total_capital']).splitlines()[2:])}

Read: this is what the same queue model says before applying the K5 top-maker haircut. The gap between this and the deployable table is the capacity/moat issue, not a pricing issue.

## Discovery Sample Sanity Check

| config_id | markets | quote fills | filled units | edge | improve ticks | wait sec | toxicity | cap | mean net | CI | win | lift vs edge0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(md_table(pooled, ['config_id','n_markets','n_quote_fills','filled_units','required_edge','improve_ticks','wait_sec','toxicity','dollar_delta_cap','mean_net_pnl_per_market','net_ci_lo','win_rate_market','same_config_lift_vs_edge0']).splitlines()[2:])}

Discovery rows are included only to catch obvious shape breaks. The OOS rows decide whether the exploratory queue replay is worth reopening.

## Decision

This run was useful, but it remains subordinate to the calibration gate. The queue proxy does **not** revive OD as a standalone strategy: the best row is a 0c-edge structural quote, OD richness lowers the result versus that row, and the ROC lower bound is still negative after comparing against the MM structural benchmark. The next real unlock is still fair-value calibration: HAR-RV/Kronos or another causal forward-vol model has to make the primitive rich-short EV lower-CI positive on enough independent markets. Queue replay is worth refining only after that, or if we explicitly fold OD richness into the MM execution layer as a weak quote-selection feature.

## Outputs

- Summary CSV: `data/analysis/csv_outputs/options_delta/od_v4_queue_replay_summary.csv`
- Trade parquet: `data/analysis/od_v4_queue_replay_trades.parquet`
- Candidate parquet: `data/analysis/od_v4_queue_replay_candidates.parquet`
"""
    NOTE.write_text(normalize_markdown_wrapping(note))


def update_docs(summary: pd.DataFrame) -> None:
    best = summary[
        summary["sample_split"].eq("oos_holdout")
        & summary["metric_scope"].eq("queue_adjusted_after_top3_haircut")
    ].sort_values(["net_ci_lo", "mean_net_pnl_per_market"], ascending=[False, False])
    if best.empty:
        bullet = "- 2026-06-01 OD v4 exploratory queue replay: ran by user override after Phase 0 failed; no OOS queue-adjusted fills survived. See [[od_v4_queue_replay_findings]]."
    else:
        r = best.iloc[0]
        bullet = (
            f"- 2026-06-01 OD v4 exploratory queue replay: ran by user override after Phase 0 failed. Best OOS queue-adjusted-after-top3 row: "
            f"{int(r['n_markets'])} markets, mean {cents(float(r['mean_net_pnl_per_market']))}, CI {ci_text(float(r['net_ci_lo']), float(r['net_ci_hi']))}, "
            f"{number(float(r['filled_units']), 2)} expected contracts. Not an official gate pass because Phase 0 calibration failed. See [[od_v4_queue_replay_findings]]."
        )

    hub = OD_HUB.read_text()
    idx = hub.find("## Current state")
    if idx >= 0:
        next_idx = hub.find("\n## ", idx + 1)
        if next_idx < 0:
            next_idx = len(hub)
        section = hub[idx:next_idx]
        lines = [ln for ln in section.splitlines() if "OD v4 exploratory queue replay" not in ln]
        new_section = "\n".join([lines[0], "", bullet, *lines[1:]]).rstrip() + "\n"
        hub = hub[:idx] + new_section + hub[next_idx:]
    else:
        hub = hub.rstrip() + "\n\n## Current state\n\n" + bullet + "\n"
    OD_HUB.write_text(hub)

    todo = BRAIN_TODO.read_text()
    todo = "\n".join(ln for ln in todo.splitlines() if "OD v4 exploratory queue replay" not in ln) + "\n"
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

    if PHASE0_NOTE.exists():
        text = PHASE0_NOTE.read_text()
        addon = (
            "\n## User-Override Exploratory Queue Replay\n\n"
            "After the official Phase 0 fail, an exploratory Phase 1 queue replay was run by user override. "
            "That run is documented in [[od_v4_queue_replay_findings]]. It does not change the Phase 0 gate verdict; it only shows what execution would look like if the branch were pursued anyway.\n"
        )
        if "## User-Override Exploratory Queue Replay" not in text:
            PHASE0_NOTE.write_text(normalize_markdown_wrapping(text.rstrip() + "\n" + addon))


def run() -> None:
    base = load_base_candidates()
    base.to_parquet(OUT_CANDIDATES, index=False)
    print(f"base candidates={len(base):,} markets={base['market_id'].nunique():,} assets={base['asset_id'].nunique():,}", flush=True)
    trades = run_replay(base)
    if trades.empty:
        summary = pd.DataFrame()
    else:
        trades.to_parquet(OUT_TRADES, index=False)
        summary = add_incremental(summarize_trades(trades))
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    make_plot(summary)
    write_note(base, trades, summary)
    update_docs(summary)
    print(f"wrote {OUT_SUMMARY}", flush=True)
    print(f"wrote {OUT_TRADES}", flush=True)
    print(f"wrote {OUT_CANDIDATES}", flush=True)
    print(f"wrote {NOTE}", flush=True)


if __name__ == "__main__":
    run()
