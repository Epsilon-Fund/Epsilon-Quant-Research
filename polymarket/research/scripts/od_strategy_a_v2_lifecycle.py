"""OD Strategy A v2: reproduce K5 winner lifecycle, hedge as overlay.

Phase order:
0. Verify actual token outcome from K-PEG asset_id/outcome_index.
1. Replay passive K-PEG fills as market episodes with multi-fill inventory,
   carried to resolution, spike-zone avoided.
2. Hedge overlay is run as a diagnostic even when Gate 1 misses, so we can see
   the variance/cost frontier without treating the hedge as the edge.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import cents, markdown_table as base_markdown_table, number, pct


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
BRAIN_TODO = REPO / "brain" / "TODO.md"
OD_HUB = NOTES / "options_delta" / "strat_options_delta.md"

K6_PANEL = ANALYSIS / "k6_vol_gap_panel.parquet"
KPEG_FILLS = ANALYSIS / "kpeg_robustness_fills.parquet"
A1_FEATURES = ANALYSIS / "block_a1_features.parquet"

OUT_SUMMARY = ANALYSIS / "csv_outputs" / "options_delta" / "od_strategy_a_v2_lifecycle.csv"
OUT_TRADES = ANALYSIS / "od_strategy_a_v2_lifecycle_trades.parquet"
OUT_FILLS = ANALYSIS / "od_strategy_a_v2_lifecycle_fills.parquet"
NOTE = NOTES / "options_delta" / "od_strategy_a_v2_lifecycle_findings.md"
FIGURES = NOTES / "options_delta" / "figures"

CRYPTO_FEE_RATE = 0.07
MAKER_REBATE_SHARE = 0.20
SPIKE_TAU_SECONDS = 15 * 60
SPIKE_MIN_PRICE = 0.40
SPIKE_MAX_PRICE = 0.60
BOOTSTRAP_SAMPLES = 1000
RNG_SEED = 20260601
GATE_SOURCE_FILTER = "all"
GATE_BUCKET = "far_absz_ge1_all_tau"
GATE_SPLIT = "oos_holdout"
ROBUST_MIN_MARKETS = 3
BINANCE_HEDGE_COST_BPS = 6.0
HEDGE_RATIOS = [0.0, 0.25, 0.50, 0.75, 1.0]
REBALANCE_BANDS_NOTIONAL = [math.inf, 25.0, 100.0, 250.0]
HEDGE_POLICIES = [
    "static_fraction",
    "vol_dependent",
    "z_dependent",
    "iv_rv_spread_dependent",
]
EXAMPLE_FILL_IDS = [327, 336, 337, 338, 339, 340, 341]

PANEL_COLS = [
    "market_id",
    "market_slug",
    "asset",
    "ts",
    "source_runs",
    "window_start",
    "window_end",
    "seconds_to_expiry",
    "up_bid",
    "up_ask",
    "down_bid",
    "down_ask",
    "polymarket_mid",
    "binance_spot",
    "binance_strike_spot",
    "binance_close_spot",
    "binance_resolution_up",
    "chainlink_resolution_up",
    "chainlink_binance_resolution_disagree",
    "ewma_sigma_annualized",
    "trailing_sigma_annualized",
    "digital_delta",
    "pm_iv_annualized",
    "pm_iv_valid",
    "iv_minus_ewma",
    "log_spot_moneyness",
    "abs_z",
    "moneyness_bucket",
    "time_bucket",
    "state_bucket",
    "source_ok_strict",
    "toxic_near_expiry",
]


def fee_rebate(price: float) -> float:
    p = float(np.clip(price, 0.0, 1.0))
    return MAKER_REBATE_SHARE * CRYPTO_FEE_RATE * p * (1.0 - p)


def ci_text(lo: float, hi: float) -> str:
    return f"[{cents(lo)}, {cents(hi)}]"


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    safe_headers = [md_cell(x) for x in headers]
    safe_rows = [[md_cell(x) for x in row] for row in rows]
    return base_markdown_table(safe_headers, safe_rows)


def read_panel_and_fills() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(K6_PANEL, columns=PANEL_COLS)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel["window_end"] = pd.to_datetime(panel["window_end"], utc=True)
    panel["market_id"] = panel["market_id"].astype(str)
    panel["ts_key"] = panel["ts"].map(lambda x: pd.Timestamp(x).value).astype("int64")
    panel = panel.sort_values(["market_id", "ts_key"]).reset_index(drop=True)

    fills = pd.read_parquet(KPEG_FILLS)
    fills = fills[fills["category"].eq("Crypto")].copy()
    fills["market_id"] = fills["market_id"].astype(str)
    fills["asset_id"] = fills["asset_id"].astype(str)
    fills["fill_ts"] = pd.to_datetime(fills["fill_time_ns"].astype("int64"), unit="ns", utc=True)
    fills["fill_ts_key"] = fills["fill_ts"].map(lambda x: pd.Timestamp(x).value).astype("int64")
    fills = fills[fills["market_id"].isin(set(panel["market_id"]))].copy()
    fills = fills.sort_values(["market_id", "fill_ts_key"]).reset_index(drop=True)
    return panel, fills


def token_outcome_map(fills: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("fill_assets", fills[["market_id", "asset_id"]].drop_duplicates())
    out = con.execute(
        f"""
        SELECT DISTINCT
            CAST(f.market_id AS VARCHAR) AS market_id,
            CAST(f.asset_id AS VARCHAR) AS asset_id,
            CAST(f.outcome_index AS INTEGER) AS outcome_index
        FROM read_parquet('{A1_FEATURES}') f
        JOIN fill_assets a
          ON CAST(f.market_id AS VARCHAR) = a.market_id
         AND CAST(f.asset_id AS VARCHAR) = a.asset_id
        WHERE f.asset_id IS NOT NULL
          AND f.asset_id <> ''
          AND f.outcome_index IS NOT NULL
        """
    ).df()
    con.close()
    out["actual_outcome"] = np.where(out["outcome_index"].eq(0), "up", "down")
    return out


def asof_join(panel: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for market_id, g in fills.groupby("market_id", sort=False):
        p = panel[panel["market_id"].eq(market_id)].sort_values("ts_key").copy()
        if p.empty:
            continue
        joined = pd.merge_asof(
            g.sort_values("fill_ts_key"),
            p,
            left_on="fill_ts_key",
            right_on="ts_key",
            direction="backward",
            suffixes=("", "_panel"),
        )
        if "market_id_panel" in joined.columns:
            joined = joined.drop(columns=["market_id_panel"])
        pieces.append(joined)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True).sort_values(["market_id", "fill_ts_key"]).reset_index(drop=True)


def build_fill_ledger(refresh: bool = True) -> pd.DataFrame:
    if OUT_FILLS.exists() and not refresh:
        return pd.read_parquet(OUT_FILLS)
    panel, fills = read_panel_and_fills()
    print(f"panel rows={len(panel):,} markets={panel['market_id'].nunique()}", flush=True)
    print(f"overlapping K-PEG crypto fills={len(fills):,} markets={fills['market_id'].nunique()}", flush=True)
    tok = token_outcome_map(fills)
    fills = fills.merge(tok[["market_id", "asset_id", "outcome_index", "actual_outcome"]], on=["market_id", "asset_id"], how="left")
    df = asof_join(panel, fills)

    up_mid = (df["up_bid"].astype(float) + df["up_ask"].astype(float)) / 2.0
    down_mid = (df["down_bid"].astype(float) + df["down_ask"].astype(float)) / 2.0
    df["heuristic_outcome"] = np.where(
        (df["current_mid"].astype(float) - up_mid).abs() <= (df["current_mid"].astype(float) - down_mid).abs(),
        "up",
        "down",
    )
    df["outcome_mismatch_old_heuristic"] = df["actual_outcome"].ne(df["heuristic_outcome"])

    chain = df["chainlink_resolution_up"]
    binance = df["binance_resolution_up"]
    resolution_up = np.where(chain.notna(), chain.astype(float), binance.astype(float)).astype(bool)
    df["resolution_up_used"] = resolution_up
    df["payoff"] = np.where(df["actual_outcome"].eq("up"), resolution_up.astype(float), 1.0 - resolution_up.astype(float))
    df["token_position"] = df["token_side"].astype(float)
    df["maker_rebate"] = df["entry_price"].map(fee_rebate)
    df["fill_pnl"] = df["token_position"] * (df["payoff"] - df["entry_price"].astype(float)) + df["maker_rebate"]
    df["cash_flow"] = -df["token_position"] * df["entry_price"].astype(float) + df["maker_rebate"]
    df["signed_delta_exposure"] = df["token_position"] * np.where(
        df["actual_outcome"].eq("up"), df["digital_delta"].astype(float), -df["digital_delta"].astype(float)
    )

    df["two_sided_book_eligible"] = (
        df[["up_bid", "up_ask", "down_bid", "down_ask"]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & df["up_ask"].astype(float).gt(df["up_bid"].astype(float))
        & df["down_ask"].astype(float).gt(df["down_bid"].astype(float))
    )
    df["late_near50_spike_zone"] = (
        df["seconds_to_expiry"].astype(float).le(SPIKE_TAU_SECONDS)
        & df["entry_price"].astype(float).between(SPIKE_MIN_PRICE, SPIKE_MAX_PRICE, inclusive="both")
    ) | df["toxic_near_expiry"].fillna(False).astype(bool)
    df["eligible"] = df["two_sided_book_eligible"] & ~df["late_near50_spike_zone"] & df["actual_outcome"].notna()
    df["strict_source_eligible"] = df["eligible"] & df["source_ok_strict"].fillna(False).astype(bool)
    df["entry_split"] = np.where(df["run_group"].eq("holdout"), "oos_holdout", "is_discovery")
    df["route"] = np.where(
        df["token_position"].gt(0),
        "maker_buy_" + df["actual_outcome"].astype(str),
        "maker_sell_" + df["actual_outcome"].astype(str),
    )
    df["fill_id"] = np.arange(len(df), dtype=int)
    OUT_FILLS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILLS, index=False)
    print(f"wrote {OUT_FILLS}", flush=True)
    return df


def bucket_mask(fills: pd.DataFrame, bucket: str) -> pd.Series:
    if bucket == "all_buckets":
        return pd.Series(True, index=fills.index)
    if bucket == "far_absz_ge1_all_tau":
        return fills["moneyness_bucket"].eq("far_absz_ge1")
    if bucket == "mid_absz_0.25_1_all_tau":
        return fills["moneyness_bucket"].eq("mid_absz_0.25_1")
    if bucket == "near_absz_lt0.25_all_tau":
        return fills["moneyness_bucket"].eq("near_absz_lt0.25")
    return fills["state_bucket"].eq(bucket)


def build_episode_for_market(g: pd.DataFrame, bucket: str, source_filter: str, split: str) -> dict[str, Any]:
    up = g[g["actual_outcome"].eq("up")]
    down = g[g["actual_outcome"].eq("down")]
    up_inv = float(up["token_position"].sum())
    down_inv = float(down["token_position"].sum())
    settlement_pnl = float((g["token_position"] * g["payoff"]).sum())
    cash = float(g["cash_flow"].sum())
    net = float(g["fill_pnl"].sum())
    first = g.iloc[0]
    last = g.iloc[-1]
    return {
        "row_type": "phase1_unhedged_lifecycle",
        "source_filter": source_filter,
        "sample_split": split,
        "bucket": bucket,
        "market_id": str(first["market_id"]),
        "market_slug": str(first["market_slug"]),
        "asset": str(first["asset"]),
        "start_ts": pd.Timestamp(g["fill_ts"].min()),
        "end_ts": pd.Timestamp(first["window_end"]),
        "median_fill_ts": pd.Timestamp(g["fill_ts"].median()),
        "n_fills": int(len(g)),
        "n_up_fills": int(len(up)),
        "n_down_fills": int(len(down)),
        "two_sided": bool(g["actual_outcome"].nunique() >= 2),
        "buy_fill_share": float(g["token_position"].gt(0).mean()),
        "sell_fill_share": float(g["token_position"].lt(0).mean()),
        "up_net_inventory": up_inv,
        "down_net_inventory": down_inv,
        "gross_abs_inventory": float(g["token_position"].abs().sum()),
        "residual_abs_inventory": float(abs(up_inv) + abs(down_inv)),
        "carry_share": float((abs(up_inv) + abs(down_inv)) / g["token_position"].abs().sum()) if len(g) else math.nan,
        "avg_entry_up": float(up["entry_price"].mean()) if len(up) else math.nan,
        "avg_entry_down": float(down["entry_price"].mean()) if len(down) else math.nan,
        "net_pnl": net,
        "cash_pnl": cash,
        "settlement_pnl": settlement_pnl,
        "maker_rebate": float(g["maker_rebate"].sum()),
        "mean_entry_price": float(g["entry_price"].mean()),
        "entry_abs_z_first": float(first["abs_z"]),
        "entry_seconds_to_expiry_first": float(first["seconds_to_expiry"]),
        "median_abs_z": float(g["abs_z"].median()),
        "median_seconds_to_expiry": float(g["seconds_to_expiry"].median()),
        "mean_signed_delta_exposure": float(g["signed_delta_exposure"].mean()),
        "final_signed_delta_exposure": float(g["signed_delta_exposure"].sum()),
        "hold_seconds_median_fill": float((pd.Timestamp(first["window_end"]) - pd.Timestamp(g["fill_ts"].median())).total_seconds()),
        "source_disagree": bool(first["chainlink_binance_resolution_disagree"]),
        "source_ok_strict": bool(first["source_ok_strict"]),
        "first_state_bucket": str(first["state_bucket"]),
        "last_state_bucket": str(last["state_bucket"]),
        "fill_ids": ",".join(str(int(x)) for x in g["fill_id"].to_numpy()),
    }


def select_nonoverlap(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return episodes
    selected: list[pd.Series] = []
    last_end = pd.Timestamp.min.tz_localize("UTC")
    for _, row in episodes.sort_values(["start_ts", "end_ts", "market_id"]).iterrows():
        start = pd.Timestamp(row["start_ts"])
        end = pd.Timestamp(row["end_ts"])
        if start >= last_end:
            selected.append(row)
            last_end = end
    if not selected:
        return episodes.iloc[0:0].copy()
    return pd.DataFrame(selected).reset_index(drop=True)


def build_episodes(fills: pd.DataFrame) -> pd.DataFrame:
    bucket_values = [
        "all_buckets",
        "far_absz_ge1_all_tau",
        "mid_absz_0.25_1_all_tau",
        "near_absz_lt0.25_all_tau",
        *sorted(fills["state_bucket"].dropna().astype(str).unique()),
    ]
    rows: list[dict[str, Any]] = []
    for source_filter in ("all", "strict"):
        source_mask = fills["eligible"] if source_filter == "all" else fills["strict_source_eligible"]
        for split in ("pooled", "is_discovery", "oos_holdout"):
            split_mask = pd.Series(True, index=fills.index) if split == "pooled" else fills["entry_split"].eq(split)
            for bucket in bucket_values:
                sub = fills[source_mask & split_mask & bucket_mask(fills, bucket)].copy()
                if sub.empty:
                    continue
                eps = [
                    build_episode_for_market(g.sort_values("fill_ts_key"), bucket, source_filter, split)
                    for _, g in sub.groupby("market_id", sort=False)
                ]
                ep = pd.DataFrame(eps)
                ep["episode_count_before_embargo"] = len(ep)
                ep["embargo_mode"] = "global_time_nonoverlap"
                ep = select_nonoverlap(ep)
                ep["episode_selected"] = True
                rows.extend(ep.to_dict("records"))
    out = pd.DataFrame(rows)
    return out


def read_hedge_panel() -> pd.DataFrame:
    cols = list(dict.fromkeys(PANEL_COLS))
    panel = pd.read_parquet(K6_PANEL, columns=cols)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel["window_end"] = pd.to_datetime(panel["window_end"], utc=True)
    panel["market_id"] = panel["market_id"].astype(str)
    panel = panel.sort_values(["market_id", "ts"]).reset_index(drop=True)
    return panel


def hedge_context(fills: pd.DataFrame) -> dict[str, float]:
    eligible = fills[fills["eligible"].fillna(False).astype(bool)]
    vol = eligible["ewma_sigma_annualized"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vol) >= 2:
        vol_q25 = float(vol.quantile(0.25))
        vol_q75 = float(vol.quantile(0.75))
    else:
        vol_q25, vol_q75 = 0.5, 1.5
    pos_iv = eligible["iv_minus_ewma"].replace([np.inf, -np.inf], np.nan)
    pos_iv = pos_iv[pos_iv.gt(0)]
    iv_scale = float(pos_iv.quantile(0.75)) if len(pos_iv) else 0.25
    if not np.isfinite(iv_scale) or iv_scale <= 0:
        iv_scale = 0.25
    return {"vol_q25": vol_q25, "vol_q75": vol_q75, "iv_scale": iv_scale}


def parse_fill_ids(fill_ids: str) -> list[int]:
    if not isinstance(fill_ids, str) or not fill_ids:
        return []
    return [int(x) for x in fill_ids.split(",") if x]


def hedge_ratio(policy: str, h_param: float, row: pd.Series, ctx: dict[str, float]) -> float:
    if h_param <= 0:
        return 0.0
    if policy == "static_fraction":
        mult = 1.0
    elif policy == "vol_dependent":
        vol = float(row.get("ewma_sigma_annualized", math.nan))
        denom = ctx["vol_q75"] - ctx["vol_q25"]
        mult = 0.5 if denom <= 1e-12 or not np.isfinite(vol) else (vol - ctx["vol_q25"]) / denom
    elif policy == "z_dependent":
        abs_z = float(row.get("abs_z", math.nan))
        mult = 1.0 / (1.0 + max(abs_z, 0.0)) if np.isfinite(abs_z) else 0.0
    elif policy == "iv_rv_spread_dependent":
        gap = float(row.get("iv_minus_ewma", math.nan))
        mult = gap / ctx["iv_scale"] if np.isfinite(gap) and gap > 0 else 0.0
    else:
        raise ValueError(f"unknown hedge policy {policy}")
    return float(np.clip(h_param * mult, 0.0, 1.0))


def hedge_ratio_values(policy: str, h_param: float, vol: float, iv_gap: float, abs_z: float, ctx: dict[str, float]) -> float:
    if h_param <= 0:
        return 0.0
    if policy == "static_fraction":
        mult = 1.0
    elif policy == "vol_dependent":
        denom = ctx["vol_q75"] - ctx["vol_q25"]
        mult = 0.5 if denom <= 1e-12 or not np.isfinite(vol) else (vol - ctx["vol_q25"]) / denom
    elif policy == "z_dependent":
        mult = 1.0 / (1.0 + max(abs_z, 0.0)) if np.isfinite(abs_z) else 0.0
    elif policy == "iv_rv_spread_dependent":
        mult = iv_gap / ctx["iv_scale"] if np.isfinite(iv_gap) and iv_gap > 0 else 0.0
    else:
        raise ValueError(f"unknown hedge policy {policy}")
    return float(np.clip(h_param * mult, 0.0, 1.0))


def make_episode_events(ep_fills: pd.DataFrame, market_panel: pd.DataFrame, end_ts: pd.Timestamp) -> pd.DataFrame:
    start_ts = pd.Timestamp(ep_fills["fill_ts"].min())
    panel_events = market_panel[
        market_panel["ts"].ge(start_ts) & market_panel["ts"].le(end_ts)
    ].copy()
    panel_events["event_type"] = "panel"
    panel_events["event_priority"] = 1
    panel_events["fill_up_position"] = 0.0
    panel_events["fill_down_position"] = 0.0

    fill_events = ep_fills.copy()
    fill_events["ts"] = pd.to_datetime(fill_events["fill_ts"], utc=True)
    fill_events["event_type"] = "fill"
    fill_events["event_priority"] = 0
    fill_events["fill_up_position"] = np.where(fill_events["actual_outcome"].eq("up"), fill_events["token_position"], 0.0)
    fill_events["fill_down_position"] = np.where(fill_events["actual_outcome"].eq("down"), fill_events["token_position"], 0.0)

    cols = [
        "ts",
        "event_type",
        "event_priority",
        "binance_spot",
        "binance_close_spot",
        "digital_delta",
        "ewma_sigma_annualized",
        "iv_minus_ewma",
        "abs_z",
        "fill_up_position",
        "fill_down_position",
    ]
    events = pd.concat([panel_events[cols], fill_events[cols]], ignore_index=True, sort=False)
    events = events.replace([np.inf, -np.inf], np.nan).dropna(subset=["ts", "binance_spot", "digital_delta"])
    return events.sort_values(["ts", "event_priority"]).reset_index(drop=True)


def simulate_hedge_for_episode(
    ep: pd.Series,
    fills_by_id: pd.DataFrame,
    market_panel: pd.DataFrame,
    *,
    policy: str,
    h_param: float,
    band_notional: float,
    ctx: dict[str, float],
    events: pd.DataFrame | None = None,
) -> dict[str, Any]:
    ep_fills = fills_by_id.loc[parse_fill_ids(str(ep["fill_ids"]))].sort_values("fill_ts")
    if events is None:
        end_ts = pd.Timestamp(ep["end_ts"])
        events = make_episode_events(ep_fills, market_panel, end_ts)

    current_units = 0.0
    hedge_pnl = 0.0
    hedge_turnover = 0.0
    hedge_cost = 0.0
    rebalances = 0
    opened = False
    last_spot = math.nan
    up_inventory = 0.0
    down_inventory = 0.0
    h_values: list[float] = []
    max_abs_notional = 0.0
    spots = events["binance_spot"].to_numpy(dtype=float)
    deltas = events["digital_delta"].to_numpy(dtype=float)
    vols = events["ewma_sigma_annualized"].to_numpy(dtype=float)
    iv_gaps = events["iv_minus_ewma"].to_numpy(dtype=float)
    abs_zs = events["abs_z"].to_numpy(dtype=float)
    fill_ups = events["fill_up_position"].to_numpy(dtype=float)
    fill_downs = events["fill_down_position"].to_numpy(dtype=float)

    for i in range(len(events)):
        spot = float(spots[i])
        if np.isfinite(last_spot):
            hedge_pnl += current_units * (spot - last_spot)
        up_inventory += float(fill_ups[i])
        down_inventory += float(fill_downs[i])
        signed_delta = (up_inventory - down_inventory) * float(deltas[i])
        h_eff = hedge_ratio_values(policy, h_param, float(vols[i]), float(iv_gaps[i]), float(abs_zs[i]), ctx)
        target_units = -h_eff * signed_delta
        h_values.append(h_eff)
        max_abs_notional = max(max_abs_notional, abs(target_units) * spot)

        should_trade = False
        if not opened and abs(target_units) > 1e-15:
            should_trade = True
            opened = True
        elif np.isfinite(band_notional):
            should_trade = abs(target_units - current_units) * spot >= band_notional
        if should_trade:
            trade_units = target_units - current_units
            trade_notional = abs(trade_units) * spot
            hedge_turnover += trade_notional
            hedge_cost += trade_notional * BINANCE_HEDGE_COST_BPS / 10_000.0
            if abs(trade_units) > 1e-15 and opened:
                rebalances += 1
            current_units = target_units
        last_spot = spot

    close_spot = float(ep_fills["binance_close_spot"].dropna().iloc[0]) if ep_fills["binance_close_spot"].notna().any() else last_spot
    if np.isfinite(last_spot) and np.isfinite(close_spot):
        hedge_pnl += current_units * (close_spot - last_spot)
        close_notional = abs(current_units) * close_spot
        hedge_turnover += close_notional
        hedge_cost += close_notional * BINANCE_HEDGE_COST_BPS / 10_000.0

    unhedged = float(ep["net_pnl"])
    net = unhedged + hedge_pnl - hedge_cost
    row = ep.to_dict()
    row.update(
        {
            "row_type": "phase2_hedge_overlay",
            "hedge_policy": policy,
            "h_param": float(h_param),
            "rebalance_band_notional": float(band_notional) if np.isfinite(band_notional) else math.inf,
            "unhedged_net_pnl": unhedged,
            "net_pnl": float(net),
            "hedge_pnl": float(hedge_pnl),
            "hedge_cost": float(hedge_cost),
            "hedge_turnover_notional": float(hedge_turnover),
            "hedge_rebalances": int(max(rebalances - 1, 0)),
            "avg_effective_h": float(np.mean(h_values)) if h_values else 0.0,
            "max_abs_hedge_notional": float(max_abs_notional),
        }
    )
    return row


def build_phase2_hedge_overlay(episodes: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    episodes = episodes[episodes["sample_split"].eq("oos_holdout")].reset_index(drop=True)
    if episodes.empty:
        return pd.DataFrame()
    panel = read_hedge_panel()
    panel_by_market = {m: g.copy() for m, g in panel.groupby("market_id", sort=False)}
    fills_by_id = fills.set_index("fill_id", drop=False)
    ctx = hedge_context(fills)
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, float, float], dict[str, Any]] = {}
    event_cache: dict[str, pd.DataFrame] = {}
    combos = [
        (policy, h_param, band)
        for policy in HEDGE_POLICIES
        for h_param in HEDGE_RATIOS
        for band in REBALANCE_BANDS_NOTIONAL
    ]
    for idx, ep in episodes.iterrows():
        market_panel = panel_by_market.get(str(ep["market_id"]))
        if market_panel is None or market_panel.empty:
            continue
        fill_key = str(ep["fill_ids"])
        if fill_key not in event_cache:
            ep_fills = fills_by_id.loc[parse_fill_ids(fill_key)].sort_values("fill_ts")
            event_cache[fill_key] = make_episode_events(ep_fills, market_panel, pd.Timestamp(ep["end_ts"]))
        for policy, h_param, band in combos:
            cache_key = (fill_key, policy, float(h_param), float(band))
            if cache_key not in cache:
                cache[cache_key] = simulate_hedge_for_episode(
                    ep,
                    fills_by_id,
                    market_panel,
                    policy=policy,
                    h_param=h_param,
                    band_notional=band,
                    ctx=ctx,
                    events=event_cache[fill_key],
                )
            base = cache[cache_key].copy()
            for col in [
                "source_filter",
                "sample_split",
                "bucket",
                "embargo_mode",
                "episode_count_before_embargo",
                "episode_selected",
            ]:
                base[col] = ep[col]
            rows.append(base)
        if (idx + 1) % 10 == 0:
            print(f"phase2 hedge overlay episodes {idx + 1}/{len(episodes)}", flush=True)
    return pd.DataFrame(rows)


def bootstrap_ci(episodes: pd.DataFrame, col: str) -> tuple[float, float]:
    vals = episodes[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(RNG_SEED + len(vals) + len(col))
    draws = rng.integers(0, len(vals), size=(BOOTSTRAP_SAMPLES, len(vals)))
    boot = vals[draws].mean(axis=1)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    episodes = episodes[episodes["row_type"].eq("phase1_unhedged_lifecycle")].copy()
    keys = ["source_filter", "sample_split", "bucket", "embargo_mode"]
    for key, g in episodes.groupby(keys, sort=True):
        lo, hi = bootstrap_ci(g, "net_pnl")
        rows.append(
            {
                "row_type": "phase1_unhedged_lifecycle",
                "source_filter": key[0],
                "sample_split": key[1],
                "bucket": key[2],
                "embargo_mode": key[3],
                "n_markets": int(g["market_id"].nunique()),
                "n_fills": int(g["n_fills"].sum()),
                "mean_net_pnl": float(g["net_pnl"].mean()),
                "net_ci_lo": lo,
                "net_ci_hi": hi,
                "median_net_pnl": float(g["net_pnl"].median()),
                "win_rate": float(g["net_pnl"].gt(0).mean()),
                "pnl_std": float(g["net_pnl"].std(ddof=1)) if len(g) > 1 else 0.0,
                "pnl_variance": float(g["net_pnl"].var(ddof=1)) if len(g) > 1 else 0.0,
                "mean_settlement_pnl": float(g["settlement_pnl"].mean()),
                "settlement_pnl_std": float(g["settlement_pnl"].std(ddof=1)) if len(g) > 1 else 0.0,
                "mean_cash_pnl": float(g["cash_pnl"].mean()),
                "mean_maker_rebate": float(g["maker_rebate"].mean()),
                "two_sided_market_share": float(g["two_sided"].mean()),
                "carry_share": float(g["carry_share"].mean()),
                "mean_fills_per_market": float(g["n_fills"].mean()),
                "median_hold_seconds": float(g["hold_seconds_median_fill"].median()),
                "source_disagree_share": float(g["source_disagree"].mean()),
                "tail_market_share": float(g["market_id"].value_counts(normalize=True).iloc[0]) if len(g) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_phase2(overlay: pd.DataFrame, phase1_summary: pd.DataFrame) -> pd.DataFrame:
    if overlay.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    unhedged_lookup = {
        (r["source_filter"], r["sample_split"], r["bucket"], r["embargo_mode"]): r
        for _, r in phase1_summary.iterrows()
    }
    keys = [
        "source_filter",
        "sample_split",
        "bucket",
        "embargo_mode",
        "hedge_policy",
        "h_param",
        "rebalance_band_notional",
    ]
    for key, g in overlay.groupby(keys, sort=True):
        lo, hi = bootstrap_ci(g, "net_pnl")
        base = unhedged_lookup.get(key[:4])
        unhedged_mean = float(g["unhedged_net_pnl"].mean())
        unhedged_std = float(g["unhedged_net_pnl"].std(ddof=1)) if len(g) > 1 else 0.0
        pnl_std = float(g["net_pnl"].std(ddof=1)) if len(g) > 1 else 0.0
        premium_retained = float(g["net_pnl"].mean() / unhedged_mean) if abs(unhedged_mean) > 1e-12 else math.nan
        variance_reduction = 1.0 - (pnl_std**2 / unhedged_std**2) if unhedged_std > 1e-12 else math.nan
        std_reduction = 1.0 - (pnl_std / unhedged_std) if unhedged_std > 1e-12 else math.nan
        rows.append(
            {
                "row_type": "phase2_hedge_overlay",
                "source_filter": key[0],
                "sample_split": key[1],
                "bucket": key[2],
                "embargo_mode": key[3],
                "hedge_policy": key[4],
                "h_param": float(key[5]),
                "rebalance_band_notional": float(key[6]),
                "n_markets": int(g["market_id"].nunique()),
                "n_fills": int(g["n_fills"].sum()),
                "mean_net_pnl": float(g["net_pnl"].mean()),
                "net_ci_lo": lo,
                "net_ci_hi": hi,
                "median_net_pnl": float(g["net_pnl"].median()),
                "win_rate": float(g["net_pnl"].gt(0).mean()),
                "pnl_std": pnl_std,
                "pnl_variance": float(g["net_pnl"].var(ddof=1)) if len(g) > 1 else 0.0,
                "mean_unhedged_pnl": unhedged_mean,
                "unhedged_pnl_std": unhedged_std,
                "mean_phase1_bucket_pnl": float(base["mean_net_pnl"]) if base is not None else math.nan,
                "phase1_bucket_ci_lo": float(base["net_ci_lo"]) if base is not None else math.nan,
                "mean_hedge_pnl": float(g["hedge_pnl"].mean()),
                "mean_hedge_cost": float(g["hedge_cost"].mean()),
                "mean_hedge_turnover_notional": float(g["hedge_turnover_notional"].mean()),
                "mean_hedge_rebalances": float(g["hedge_rebalances"].mean()),
                "mean_avg_effective_h": float(g["avg_effective_h"].mean()),
                "mean_max_abs_hedge_notional": float(g["max_abs_hedge_notional"].mean()),
                "premium_retained": premium_retained,
                "std_reduction": std_reduction,
                "variance_reduction": variance_reduction,
                "two_sided_market_share": float(g["two_sided"].mean()),
                "carry_share": float(g["carry_share"].mean()),
                "mean_fills_per_market": float(g["n_fills"].mean()),
                "median_hold_seconds": float(g["hold_seconds_median_fill"].median()),
                "source_disagree_share": float(g["source_disagree"].mean()),
                "tail_market_share": float(g["market_id"].value_counts(normalize=True).iloc[0]) if len(g) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def phase0_stats(fills: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_fills": int(len(fills)),
        "missing_actual_outcome": int(fills["actual_outcome"].isna().sum()),
        "mismatch_count": int(fills["outcome_mismatch_old_heuristic"].sum()),
        "mismatch_rate": float(fills["outcome_mismatch_old_heuristic"].mean()) if len(fills) else math.nan,
        "eligible_fills": int(fills["eligible"].sum()),
        "strict_eligible_fills": int(fills["strict_source_eligible"].sum()),
    }


def example_equity_path(fills: pd.DataFrame, start_equity: float = 1.0) -> pd.DataFrame:
    ff = fills[fills["fill_id"].isin(EXAMPLE_FILL_IDS)].sort_values("fill_ts").copy()
    if ff.empty:
        return pd.DataFrame()
    up_payoff = float(ff.loc[ff["actual_outcome"].eq("up"), "payoff"].iloc[0]) if ff["actual_outcome"].eq("up").any() else 0.0
    down_payoff = float(ff.loc[ff["actual_outcome"].eq("down"), "payoff"].iloc[0]) if ff["actual_outcome"].eq("down").any() else 0.0
    cash = 0.0
    up_inv = 0.0
    down_inv = 0.0
    rows: list[dict[str, Any]] = [
        {
            "step": 0,
            "label": "Start",
            "trade": "start",
            "cash_change": 0.0,
            "cash_after": 0.0,
            "up_inv": 0.0,
            "down_inv": 0.0,
            "settlement_value": 0.0,
            "equity_if_resolved": start_equity,
        }
    ]
    for step, (_, r) in enumerate(ff.iterrows(), start=1):
        pos = float(r["token_position"])
        price = float(r["entry_price"])
        rebate = float(r["maker_rebate"])
        cash_change = -pos * price + rebate
        cash += cash_change
        if str(r["actual_outcome"]) == "up":
            up_inv += pos
            token = "UP"
        else:
            down_inv += pos
            token = "DOWN"
        side = "long" if pos > 0 else "short"
        settlement_value = up_inv * up_payoff + down_inv * down_payoff
        rows.append(
            {
                "step": step,
                "label": str(step),
                "trade": f"{side} {token} @ {price:.2f}",
                "cash_change": cash_change,
                "cash_after": cash,
                "up_inv": up_inv,
                "down_inv": down_inv,
                "settlement_value": settlement_value,
                "equity_if_resolved": start_equity + cash + settlement_value,
            }
        )
    return pd.DataFrame(rows)


def write_practical_charts(fills: pd.DataFrame, summary: pd.DataFrame) -> dict[str, str]:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    colors = {
        "green": "#2E7D32",
        "blue": "#1565C0",
        "orange": "#EF6C00",
        "red": "#C62828",
        "gray": "#616161",
        "light": "#E8EEF7",
    }

    charts: dict[str, str] = {}
    path_df = example_equity_path(fills)
    if not path_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
        x = np.arange(len(path_df))
        equity = path_df["equity_if_resolved"].to_numpy(dtype=float)
        cash_plus_start = 1.0 + path_df["cash_after"].to_numpy(dtype=float)
        ax.plot(x, equity, marker="o", linewidth=2.5, color=colors["green"], label="Equity if resolved now")
        ax.plot(x, cash_plus_start, marker="s", linewidth=1.8, linestyle="--", color=colors["blue"], label="Start equity + cash before settlement")
        ax.axhline(1.0, color=colors["gray"], linewidth=1, alpha=0.6)
        ax.fill_between(x, 1.0, equity, where=equity >= 1.0, color=colors["green"], alpha=0.08)
        ax.set_title("ETH Example: Equity Path From $1 Starting Equity", loc="left", fontsize=13, fontweight="bold")
        ax.set_ylabel("Dollars")
        ax.set_xlabel("Fill step")
        ax.set_xticks(x)
        ax.set_xticklabels(path_df["label"].tolist())
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", frameon=False)
        for xi, yi in zip(x, equity, strict=True):
            ax.text(xi, yi + 0.035, f"${yi:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        out = FIGURES / "od_strategy_a_v2_eth_equity_path.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        charts["equity_path"] = f"figures/{out.name}"

        final_cash = float(path_df["cash_after"].iloc[-1])
        final_settlement = float(path_df["settlement_value"].iloc[-1])
        final_equity = float(path_df["equity_if_resolved"].iloc[-1])
        steps = ["Start", "Cash from fills", "Settlement", "Final equity"]
        vals = [1.0, final_cash, final_settlement, final_equity]
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
        running = 0.0
        max_y = max(1.0, 1.0 + final_cash, 1.0 + final_cash + final_settlement, final_equity) + 0.45
        for i, (label, val) in enumerate(zip(steps, vals, strict=True)):
            if label == "Final equity":
                ax.bar(i, val, bottom=0, color=colors["blue"], width=0.58)
                ax.text(i, val + 0.04, f"${val:.2f}", ha="center", fontsize=9, fontweight="bold")
                continue
            bottom = running if label != "Start" else 0.0
            color = colors["green"] if val >= 0 else colors["red"]
            ax.bar(i, val, bottom=bottom, color=color, width=0.58)
            y = bottom + val
            ax.text(i, y + (0.04 if val >= 0 else -0.08), f"{val:+.2f}" if label != "Start" else f"${val:.2f}", ha="center", fontsize=9)
            running = bottom + val
        ax.axhline(0, color=colors["gray"], linewidth=1)
        ax.set_title("ETH Example: Why the Episode Ends Positive", loc="left", fontsize=13, fontweight="bold")
        ax.set_ylabel("Dollars")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps, rotation=0)
        ax.set_ylim(0, max_y)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out = FIGURES / "od_strategy_a_v2_eth_waterfall.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        charts["waterfall"] = f"figures/{out.name}"

    if OUT_TRADES.exists():
        ledger = pd.read_parquet(OUT_TRADES)
        p1 = ledger[
            ledger["row_type"].eq("phase1_unhedged_lifecycle")
            & ledger["source_filter"].eq("all")
            & ledger["sample_split"].eq("oos_holdout")
            & ledger["bucket"].eq(GATE_BUCKET)
        ].copy()
        p2 = ledger[
            ledger["row_type"].eq("phase2_hedge_overlay")
            & ledger["source_filter"].eq("all")
            & ledger["sample_split"].eq("oos_holdout")
            & ledger["bucket"].eq(GATE_BUCKET)
            & ledger["hedge_policy"].eq("static_fraction")
            & ledger["h_param"].astype(float).eq(1.0)
            & np.isinf(ledger["rebalance_band_notional"].astype(float))
        ].copy()
        if not p1.empty and not p2.empty:
            merged = p1[["market_id", "asset", "market_slug", "net_pnl"]].merge(
                p2[["market_id", "net_pnl"]],
                on="market_id",
                suffixes=("_phase1", "_phase2"),
            )
            merged = merged.sort_values("market_slug")
            labels = [f"{a}\n{i+1}" for i, a in enumerate(merged["asset"].astype(str))]
            x = np.arange(len(merged))
            width = 0.38
            fig, ax = plt.subplots(figsize=(9.5, 5), dpi=180)
            p1_vals = 100.0 * merged["net_pnl_phase1"].to_numpy(dtype=float)
            p2_vals = 100.0 * merged["net_pnl_phase2"].to_numpy(dtype=float)
            ax.bar(x - width / 2, p1_vals, width, label="Phase 1 unhedged", color=colors["orange"])
            ax.bar(x + width / 2, p2_vals, width, label="Phase 2 static hedge", color=colors["green"])
            ax.axhline(0, color=colors["gray"], linewidth=1)
            ax.axhline(float(summary.loc[
                summary["row_type"].eq("phase1_unhedged_lifecycle")
                & summary["source_filter"].eq("all")
                & summary["sample_split"].eq("oos_holdout")
                & summary["bucket"].eq(GATE_BUCKET),
                "mean_net_pnl",
            ].iloc[0]) * 100.0, color=colors["orange"], linestyle="--", linewidth=1.2, alpha=0.8)
            ax.axhline(float(summary.loc[
                summary["row_type"].eq("phase2_hedge_overlay")
                & summary["source_filter"].eq("all")
                & summary["sample_split"].eq("oos_holdout")
                & summary["bucket"].eq(GATE_BUCKET)
                & summary["hedge_policy"].eq("static_fraction")
                & summary["h_param"].astype(float).eq(1.0)
                & np.isinf(summary["rebalance_band_notional"].astype(float)),
                "mean_net_pnl",
            ].iloc[0]) * 100.0, color=colors["green"], linestyle="--", linewidth=1.2, alpha=0.8)
            ax.set_title("Primary OOS Far-|z| Episodes: Unhedged vs Static Hedge", loc="left", fontsize=13, fontweight="bold")
            ax.set_ylabel("Episode PnL (cents)")
            ax.set_xlabel("Selected non-overlapping market episode")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(loc="upper left", frameon=False)
            fig.tight_layout()
            out = FIGURES / "od_strategy_a_v2_primary_episode_pnl.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            charts["episode_pnl"] = f"figures/{out.name}"
    return charts


def fmt_row(r: pd.Series) -> list[str]:
    return [
        str(r["bucket"]),
        str(int(r["n_markets"])),
        str(int(r["n_fills"])),
        cents(float(r["mean_net_pnl"])),
        ci_text(float(r["net_ci_lo"]), float(r["net_ci_hi"])),
        pct(float(r["win_rate"])),
        cents(float(r["pnl_std"])),
        cents(float(r["settlement_pnl_std"])),
        pct(float(r["two_sided_market_share"])),
        pct(float(r["carry_share"])),
        number(float(r["median_hold_seconds"]) / 60.0, 1),
    ]


def table_for(summary: pd.DataFrame, source_filter: str, split: str) -> list[list[str]]:
    sub = summary[
        summary["row_type"].eq("phase1_unhedged_lifecycle")
        & summary["source_filter"].eq(source_filter)
        & summary["sample_split"].eq(split)
        & summary["bucket"].ne("all_buckets")
    ].copy()
    if sub.empty:
        return []
    order = {
        "far_absz_ge1_all_tau": 0,
        "far_absz_ge1|late_lt30m": 1,
        "far_absz_ge1|mid_30m_2h": 2,
        "far_absz_ge1|early_gt2h": 3,
    }
    sub["rank"] = sub["bucket"].map(order).fillna(99)
    sub = sub.sort_values(["rank", "net_ci_lo", "mean_net_pnl"], ascending=[True, False, False])
    return [fmt_row(r) for _, r in sub.iterrows()]


def fmt_band(value: float) -> str:
    return "static" if not np.isfinite(value) else f"${value:.0f}"


def fmt_phase2_row(r: pd.Series) -> list[str]:
    return [
        str(r["bucket"]),
        str(r["hedge_policy"]),
        number(float(r["h_param"]), 2),
        fmt_band(float(r["rebalance_band_notional"])),
        str(int(r["n_markets"])),
        cents(float(r["mean_net_pnl"])),
        ci_text(float(r["net_ci_lo"]), float(r["net_ci_hi"])),
        pct(float(r["win_rate"])),
        cents(float(r["mean_hedge_cost"])),
        number(float(r["premium_retained"]), 2),
        pct(float(r["variance_reduction"])),
        number(float(r["mean_hedge_rebalances"]), 1),
    ]


def top_phase2_rows(
    summary: pd.DataFrame,
    *,
    source_filter: str = "all",
    split: str = "oos_holdout",
    bucket: str | None = None,
    limit: int = 12,
) -> list[list[str]]:
    sub = summary[
        summary["row_type"].eq("phase2_hedge_overlay")
        & summary["source_filter"].eq(source_filter)
        & summary["sample_split"].eq(split)
    ].copy()
    if bucket is None:
        sub = sub[sub["bucket"].ne("all_buckets")].copy()
    else:
        sub = sub[sub["bucket"].eq(bucket)].copy()
    sub = sub[sub["h_param"].astype(float).gt(0)].copy()
    if sub.empty:
        return []
    sub = sub.sort_values(["net_ci_lo", "mean_net_pnl", "variance_reduction"], ascending=[False, False, False])
    return [fmt_phase2_row(r) for _, r in sub.head(limit).iterrows()]


def minimum_hedge_rows(summary: pd.DataFrame, source_filter: str = "all", split: str = "oos_holdout") -> list[list[str]]:
    sub = summary[
        summary["row_type"].eq("phase2_hedge_overlay")
        & summary["source_filter"].eq(source_filter)
        & summary["sample_split"].eq(split)
        & summary["bucket"].ne("all_buckets")
        & summary["net_ci_lo"].gt(0)
    ].copy()
    if sub.empty:
        return []
    sub["band_rank"] = np.where(np.isfinite(sub["rebalance_band_notional"].astype(float)), sub["rebalance_band_notional"].astype(float), 1e12)
    sub = sub.sort_values(["bucket", "h_param", "band_rank", "net_ci_lo"], ascending=[True, True, True, False])
    rows: list[list[str]] = []
    for bucket, g in sub.groupby("bucket", sort=True):
        r = g.iloc[0]
        rows.append(
            [
                str(bucket),
                str(r["hedge_policy"]),
                number(float(r["h_param"]), 2),
                fmt_band(float(r["rebalance_band_notional"])),
                cents(float(r["mean_net_pnl"])),
                ci_text(float(r["net_ci_lo"]), float(r["net_ci_hi"])),
                pct(float(r["variance_reduction"])),
                number(float(r["premium_retained"]), 2),
            ]
        )
    return rows


def gate_row(summary: pd.DataFrame, source_filter: str = GATE_SOURCE_FILTER) -> pd.Series | None:
    g = summary[
        summary["row_type"].eq("phase1_unhedged_lifecycle")
        & summary["source_filter"].eq(source_filter)
        & summary["sample_split"].eq(GATE_SPLIT)
        & summary["bucket"].eq(GATE_BUCKET)
    ]
    return None if g.empty else g.iloc[0]


def write_note(fills: pd.DataFrame, episodes: pd.DataFrame, summary: pd.DataFrame) -> bool:
    stats = phase0_stats(fills)
    charts = write_practical_charts(fills, summary)
    equity_chart = (
        f"![ETH example equity path]({charts['equity_path']})"
        if "equity_path" in charts
        else "_Equity path chart unavailable._"
    )
    waterfall_chart = (
        f"![ETH example waterfall]({charts['waterfall']})"
        if "waterfall" in charts
        else "_Waterfall chart unavailable._"
    )
    episode_chart = (
        f"![Primary OOS episode PnL bars]({charts['episode_pnl']})"
        if "episode_pnl" in charts
        else "_Primary episode PnL chart unavailable._"
    )
    gate = gate_row(summary)
    gate_strict = gate_row(summary, "strict")
    gate_pass = False
    if gate is not None:
        gate_pass = bool(float(gate["net_ci_lo"]) > 0 and int(gate["n_markets"]) >= ROBUST_MIN_MARKETS)

    if gate is None:
        headline = "Gate 1 fails: no OOS far-|z| episode survived the lifecycle filters."
        gate_text = "No primary gate row."
    else:
        headline = (
            "Gate 1 clears; Phase 2 hedge overlay should run next."
            if gate_pass
            else "Gate 1 fails: unhedged K5 winner lifecycle does not clear OOS lower-CI on this replay."
        )
        gate_text = (
            f"Primary OOS far-|z| family (`{GATE_SOURCE_FILTER}` source): n={int(gate['n_markets'])} markets, "
            f"{int(gate['n_fills'])} fills, mean {cents(float(gate['mean_net_pnl']))}, "
            f"CI {ci_text(float(gate['net_ci_lo']), float(gate['net_ci_hi']))}."
        )
    strict_text = (
        "Strict-source gate row absent."
        if gate_strict is None
        else (
            f"Strict-source OOS far-|z| diagnostic: n={int(gate_strict['n_markets'])}, "
            f"mean {cents(float(gate_strict['mean_net_pnl']))}, "
            f"CI {ci_text(float(gate_strict['net_ci_lo']), float(gate_strict['net_ci_hi']))}."
        )
    )
    if gate is not None and np.isfinite(float(gate["net_ci_lo"])):
        hurdle_gap = max(0.0, -float(gate["net_ci_lo"]))
        phase1_miss = f"Phase 1 missed the lower-CI>0 gate by {cents(hurdle_gap)}."
    else:
        phase1_miss = "Phase 1 had no measurable primary lower-CI gate row."

    phase2_gate = summary[
        summary["row_type"].eq("phase2_hedge_overlay")
        & summary["source_filter"].eq(GATE_SOURCE_FILTER)
        & summary["sample_split"].eq(GATE_SPLIT)
        & summary["bucket"].eq(GATE_BUCKET)
    ].copy()
    if phase2_gate.empty:
        phase2_text = "Phase 2 hedge overlay rows are absent."
    else:
        best = phase2_gate.sort_values(["net_ci_lo", "mean_net_pnl"], ascending=[False, False]).iloc[0]
        phase2_text = (
            "Best primary Phase 2 overlay: "
            f"`{best['hedge_policy']}` h={number(float(best['h_param']), 2)}, "
            f"B={fmt_band(float(best['rebalance_band_notional']))}, "
            f"mean {cents(float(best['mean_net_pnl']))}, "
            f"CI {ci_text(float(best['net_ci_lo']), float(best['net_ci_hi']))}, "
            f"variance reduction {pct(float(best['variance_reduction']))}, "
            f"premium retained {number(float(best['premium_retained']), 2)}."
        )

    note = f"""# OD Strategy A v2 Lifecycle

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]

## Headline

{headline}

{gate_text}

{strict_text}

{phase1_miss}

Phase 2 hedge overlay was run anyway per steer, but it remains labeled diagnostic because the unhedged lifecycle is still the edge gate.

{phase2_text}

## How to Read This Note

This is an Options-Delta Strategy A replay, not a Block K taker race. The edge being tested is the K5-style maker lifecycle: passive K-PEG fills, two-sided inventory when both UP and DOWN fills arrive in the same market, no Polymarket exit, and settlement at resolution. The Binance hedge in Phase 2 is only a variance overlay on that lifecycle; it is not allowed to create the edge by itself.

Unit of analysis is a **market episode**. Within a market episode, all eligible fills are aggregated into one inventory path. The global time embargo then keeps only non-overlapping market windows for each restricted bucket strategy so capital is not counted twice across overlapping 4h contracts.

Samples:

- `is_discovery` is in-sample and shown only as context.
- `oos_holdout` is the decision sample.
- `all` source means the normal K5/K-PEG replay universe.
- `strict` source additionally requires the Chainlink/Binance settlement-source filters; it is a diagnostic because it changes the universe.

Decision gate: OOS `far_absz_ge1_all_tau`, source `all`, global market-episode embargo, lower 95% CI > 0. Phase 1 missed that gate by 17.14c, so the official lifecycle verdict remains fail even though the mean is strongly positive.

## Bucket Glossary

Moneyness buckets use `abs_z = abs(log(S/K) / (sigma * sqrt(tau)))`, where `S` is Binance spot, `K` is the window-open strike, `sigma` is causal EWMA/trailing vol from the K6 surface, and `tau` is time to expiry.

- `near_absz_lt0.25`: close to the strike; high binary jump risk.
- `mid_absz_0.25_1`: moderate distance from strike; delta is usually meaningful.
- `far_absz_ge1`: far from strike; longshot/pinned regime.
- `late_lt30m`: less than 30 minutes to expiry.
- `mid_30m_2h`: 30 minutes to 2 hours to expiry.
- `early_gt2h`: more than 2 hours to expiry.
- `far_absz_ge1|late_lt30m` means the intersection of far moneyness and late time. `far_absz_ge1_all_tau` pools far moneyness across all time buckets.

## Table Column Glossary

All PnL columns are per market episode and displayed in cents. All CIs are 95% bootstrap confidence intervals over market episodes.

Phase 1 columns:

- `markets`: number of non-overlapping market episodes.
- `fills`: total K-PEG fills inside those selected episodes.
- `mean net`: settlement PnL plus maker rebate, with no Polymarket exit and no hedge.
- `CI`: bootstrap CI for `mean net`.
- `win`: share of selected market episodes with positive net PnL.
- `PnL std`: standard deviation of episode net PnL.
- `settlement std`: standard deviation of the binary settlement leg; this is the jump variance a hedge is trying to tame.
- `two-sided`: share of episodes containing both UP and DOWN token fills.
- `carry`: residual inventory carried to resolution divided by gross filled inventory.
- `median hold min`: median time from episode median fill to resolution.

Phase 2 columns:

- `policy`: hedge-ratio rule.
- `h`: base hedge ratio.
- `B`: rebalance band in Binance notional; `static` means set the first nonzero hedge and do not rebalance until settlement.
- `net`: Phase 1 episode PnL plus Binance hedge PnL minus Binance costs.
- `win`: share of selected market episodes with positive hedged net PnL.
- `hedge cost`: average Binance cost at 6 bp per hedge trade/settlement notional.
- `prem retained`: hedged mean net divided by unhedged Phase 1 mean net. This can be unstable or negative when the unhedged mean is near zero or negative.
- `var reduced`: `1 - hedged variance / unhedged variance`; negative means the hedge increased variance.
- `rebal`: average count of intrawindow hedge rebalances, excluding final settlement flatten.

## Practical Money Example

The table means are **dollars per selected market episode**, displayed as cents. They are not percentage returns and not per-fill averages. One selected OOS primary-gate episode is `eth-updown-4h-1780113600`, with 7 eligible K-PEG fills in the `far_absz_ge1_all_tau` bucket.

Position convention:

- `token_position = +1`: we are long one Polymarket token.
- `token_position = -1`: we are short one Polymarket token.
- `payoff = 1` if that token resolves correctly, else `0`.
- maker rebate per fill is `0.20 * 0.07 * p * (1 - p)`.

Phase 1 Polymarket-only PnL per fill:

```text
long token:  payoff - entry_price + rebate
short token: entry_price - payoff + rebate
```

Actual fills in that ETH episode:

```text
short UP   at 0.23, UP lost   -> 0.23 - 0 + rebate = +0.2325
short DOWN at 0.91, DOWN won  -> 0.91 - 1 + rebate = -0.0889
long DOWN  at 0.71, DOWN won  -> 1 - 0.71 + rebate = +0.2929
short UP   at 0.17, UP lost   -> 0.17 - 0 + rebate = +0.1720
short UP   at 0.21, UP lost   -> 0.21 - 0 + rebate = +0.2123
short UP   at 0.21, UP lost   -> 0.21 - 0 + rebate = +0.2123
short DOWN at 0.99, DOWN won  -> 0.99 - 1 + rebate = -0.0099
```

Episode totals:

```text
Polymarket cash/rebate leg = +$2.0233
binary settlement leg      = -$1.0000
Phase 1 net                = +$1.0233 = +102.33c
gross filled contracts     = 7
rough PnL per gross fill   = +$1.0233 / 7 = +14.62c
```

That `+102.33c` is one of the 6 primary OOS far-|z| market episodes. The Phase 1 headline mean is the simple episode average:

```text
episodes: [-76.63c, +618.73c, +48.98c, +17.81c, -2.76c, +102.33c]
mean     = +118.08c
win      = 4 / 6 = 66.67%
```

{episode_chart}

Phase 2 adds a Binance hedge. This is not another Polymarket bet; it is a spot/perp position in the underlying coin. The hedge target is:

```text
signed_delta = net_UP_inventory * digital_delta - net_DOWN_inventory * digital_delta
binance_units = -h * signed_delta
```

So long UP or short DOWN has positive coin delta and gets hedged by shorting Binance. Short UP or long DOWN has negative coin delta and gets hedged by going long Binance.

For the best primary Phase 2 row (`static_fraction`, `h=1.00`, `B=static`), the first nonzero hedge in the ETH example is set after the first fill:

```text
first fill: short 1 UP
ETH spot: $2006.40
digital delta: 0.02295993
signed delta: -0.02295993
Binance hedge: +0.02295993 ETH long
ETH close: $2016.32

hedge PnL = 0.02295993 * (2016.32 - 2006.40)
          = +$0.2278
```

Binance costs are 6 bp on hedge entry/settlement turnover:

```text
entry notional ~= 0.02295993 * 2006.40 = $46.06
exit notional  ~= 0.02295993 * 2016.32 = $46.30
turnover       ~= $92.36
hedge cost     = 92.36 * 0.0006 = $0.0554
```

So the Phase 2 net for the same episode is:

```text
Phase 1 net   = +$1.0233
hedge PnL     = +$0.2278
hedge cost    = -$0.0554
Phase 2 net   = +$1.1956 = +119.56c
```

The Phase 2 headline mean uses the same 6-episode average:

```text
mean unhedged Phase 1 = +$1.1808
mean hedge PnL        = +$0.0519
mean hedge cost       = -$0.0407
mean Phase 2 net      = +$1.1920 = +119.20c
```

The hedge therefore barely changes the mean. Its purpose is variance reduction:

```text
Phase 1 std = 252.28c
Phase 2 std = 239.65c
variance reduction = 9.76%
```

That is why the primary CI improves from `[-17.14c, 323.46c]` to `[-10.56c, 313.41c]`, but still does not clear zero.

### Exposure Path From $1 Starting Equity

The replay is **not** "post one bid on UP and one bid on DOWN, then cancel the other after the first fill." Phase 1 intentionally takes every eligible K-PEG maker fill within the selected market episode. That is the K5 lifecycle being reproduced: inventory can become two-sided, matched legs can cancel exposure, and remaining inventory is carried to resolution.

Also, `short UP` / `short DOWN` is economic notation. On Polymarket this must be collateralized through existing inventory or complete-set mechanics; it is not an uncollateralized naked short. Economically, `short DOWN at 0.91` is the same payoff as `long UP at 0.09`, but it is recorded as `short DOWN` because the actual maker fill happened on the DOWN token book.

For the same ETH episode, assume we start with `$1.00` of equity and track cash plus final settlement value. In this market DOWN won, so each UP token settles to `$0` and each DOWN token settles to `$1`.

Column meanings:

- `cash change`: immediate cash from this fill, including rebate. Shorts add cash; longs spend cash.
- `cash after fills`: cumulative cash collected/spent so far, before final token settlement.
- `UP inv` / `DOWN inv`: remaining inventory in contract units. Positive means long, negative means short.
- `settlement value if resolved now`: value or liability of current inventory using this example's final outcome. Since DOWN won, UP inventory is worth `$0`, and each short DOWN is a `$1` liability.
- `equity if resolved now`: `starting equity + cash after fills + settlement value`. This is a hindsight settlement-value path for explanation, not a live mark-to-market or margin calculation.

{equity_chart}

{waterfall_chart}

| step | trade | cash change | cash after fills | UP inv | DOWN inv | settlement value if resolved now | equity if resolved now |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | short UP @ 0.23 | +$0.2325 | +$0.2325 | -1 | 0 | $0.0000 | $1.2325 |
| 2 | short DOWN @ 0.91 | +$0.9111 | +$1.1436 | -1 | -1 | -$1.0000 | $1.1436 |
| 3 | long DOWN @ 0.71 | -$0.7071 | +$0.4365 | -1 | 0 | $0.0000 | $1.4365 |
| 4 | short UP @ 0.17 | +$0.1720 | +$0.6085 | -2 | 0 | $0.0000 | $1.6085 |
| 5 | short UP @ 0.21 | +$0.2123 | +$0.8208 | -3 | 0 | $0.0000 | $1.8208 |
| 6 | short UP @ 0.21 | +$0.2123 | +$1.0331 | -4 | 0 | $0.0000 | $2.0331 |
| 7 | short DOWN @ 0.99 | +$0.9901 | +$2.0233 | -4 | -1 | -$1.0000 | $2.0233 |

Final accounting:

```text
starting equity = $1.0000
final equity    = $2.0233
net PnL         = $1.0233 = +102.33c
```

The key exposure after the final fill is `short 4 UP` and `short 1 DOWN`. Because DOWN won, the short UP inventory expired worthless in our favor, while the short DOWN inventory cost `$1` at settlement. The episode still made money because it collected `$2.0233` of net cash/rebate before paying that `$1` settlement.

## Phase 0 — Token Side Correctness

Actual outcome side is now derived from the K-PEG fill `asset_id` joined to the stored `outcome_index` in `block_a1_features.parquet` (`0=up`, `1=down`). The old heuristic inferred side by whichever K6 UP/DOWN mid was closer to the asset mid.

- fills checked: `{stats['n_fills']:,}`
- missing actual outcome map: `{stats['missing_actual_outcome']:,}`
- old-heuristic mismatches: `{stats['mismatch_count']:,}` ({pct(stats['mismatch_rate'])})
- eligible fills after two-sided/spike filters: `{stats['eligible_fills']:,}`
- strict-source eligible fills: `{stats['strict_eligible_fills']:,}`

Read: the old heuristic was mostly right but not exact. The nonzero mismatch rate means payoff and delta sign should use the actual token map going forward.

## Phase 1 — OOS Lifecycle Buckets

Primary source scope is all source-validity rows, matching the K5 real-maker lifecycle reproduction. Strict-source rows are reported as a settlement-basis diagnostic. Episodes are market-level, multi-fill, carried to resolution, and globally time-embargoed so overlapping 4h windows are not double-counted within each restricted bucket strategy.

{markdown_table(
    ["bucket", "markets", "fills", "mean net", "CI", "win", "PnL std", "settlement std", "two-sided", "carry", "median hold min"],
    table_for(summary, "all", "oos_holdout"),
)}

## Strict-Source OOS Diagnostic

{markdown_table(
    ["bucket", "markets", "fills", "mean net", "CI", "win", "PnL std", "settlement std", "two-sided", "carry", "median hold min"],
    table_for(summary, "strict", "oos_holdout"),
)}

## Phase 2 — Hedge Overlay Frontier

Diagnostic only: Phase 1 failed by {cents(max(0.0, -float(gate['net_ci_lo']))) if gate is not None else 'n/a'}, but the CI was close enough to inspect the variance/cost frontier. Hedge rows use the same market-episode inventory path, Binance spot hedge at `{BINANCE_HEDGE_COST_BPS:.1f}bp` per hedge trade/settlement notional, and no Polymarket exit. `B=static` means set the first nonzero hedge and never rebalance; finite B rebalances when target hedge notional drifts by at least that many dollars.

Primary all-source OOS far-|z| family:

{markdown_table(
    ["bucket", "policy", "h", "B", "markets", "net", "net CI", "win", "hedge cost", "prem retained", "var reduced", "rebal"],
    top_phase2_rows(summary, source_filter="all", split="oos_holdout", bucket=GATE_BUCKET, limit=12),
)}

Best all-source OOS overlays across buckets:

{markdown_table(
    ["bucket", "policy", "h", "B", "markets", "net", "net CI", "win", "hedge cost", "prem retained", "var reduced", "rebal"],
    top_phase2_rows(summary, source_filter="all", split="oos_holdout", bucket=None, limit=12),
)}

Minimum hedge rows with lower-CI > 0, where any exist:

{markdown_table(
    ["bucket", "policy", "h", "B", "net", "net CI", "var reduced", "prem retained"],
    minimum_hedge_rows(summary, source_filter="all", split="oos_holdout"),
)}

## IS Lead Table

Discovery rows are shown only as a lead.

{markdown_table(
    ["bucket", "markets", "fills", "mean net", "CI", "win", "PnL std", "settlement std", "two-sided", "carry", "median hold min"],
    table_for(summary, "all", "is_discovery"),
)}

## Gate 1 Decision

Pre-registered decision gate: OOS `far_absz_ge1_all_tau`, global market-episode embargo, lower CI > 0.

{gate_text}

Decision: **{'PASS' if gate_pass else 'FAIL'}**.

Phase 2 was run as requested despite the Phase 1 miss. Treat its frontier as a hedge-design diagnostic, not a rescue of the unhedged edge gate.

Outputs:

- `data/analysis/csv_outputs/options_delta/od_strategy_a_v2_lifecycle.csv`
- `data/analysis/od_strategy_a_v2_lifecycle_trades.parquet`
- `data/analysis/od_strategy_a_v2_lifecycle_fills.parquet`
"""
    NOTE.write_text(note, encoding="utf-8")
    print(f"wrote {NOTE}", flush=True)
    return gate_pass


def replace_section(path: Path, marker: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + replacement.strip() + "\n", encoding="utf-8")
        return
    before, after = text.split(marker, 1)
    # Preserve text after next top-level section if present.
    next_idx = after.find("\n## ", 1)
    tail = "" if next_idx == -1 else after[next_idx:]
    suffix = "" if not tail else "\n\n" + tail.lstrip("\n")
    path.write_text(before.rstrip() + "\n\n" + replacement.strip() + suffix, encoding="utf-8")


def update_docs(gate_pass: bool, summary: pd.DataFrame) -> None:
    gate = gate_row(summary)
    gate_phrase = "cleared" if gate_pass else "failed"
    if gate is not None:
        gate_line = (
            f"OD Strategy A v2 lifecycle {gate_phrase}: OOS far-|z| family n={int(gate['n_markets'])} markets / "
            f"{int(gate['n_fills'])} fills, mean {cents(float(gate['mean_net_pnl']))}, "
            f"CI {ci_text(float(gate['net_ci_lo']), float(gate['net_ci_hi']))}."
        )
    else:
        gate_line = "OD Strategy A v2 lifecycle failed: no OOS far-|z| family row survived filters."
    phase2_gate = summary[
        summary["row_type"].eq("phase2_hedge_overlay")
        & summary["source_filter"].eq(GATE_SOURCE_FILTER)
        & summary["sample_split"].eq(GATE_SPLIT)
        & summary["bucket"].eq(GATE_BUCKET)
    ].copy()
    if phase2_gate.empty:
        phase2_line = "Phase 2 hedge overlay diagnostic was requested but no primary rows survived."
    else:
        best = phase2_gate.sort_values(["net_ci_lo", "mean_net_pnl"], ascending=[False, False]).iloc[0]
        phase2_line = (
            f"Phase 2 diagnostic ran anyway; best primary overlay is {best['hedge_policy']} "
            f"h={number(float(best['h_param']), 2)}, B={fmt_band(float(best['rebalance_band_notional']))}, "
            f"mean {cents(float(best['mean_net_pnl']))}, CI {ci_text(float(best['net_ci_lo']), float(best['net_ci_hi']))}, "
            f"variance reduction {pct(float(best['variance_reduction']))}."
        )

    if OD_HUB.exists():
        hub_text = OD_HUB.read_text(encoding="utf-8")
        for old_status in [
            "status: continuous-hedge gamma scalp closed; STATIC-hedge Strategy A is a validated-mechanism near-miss (underpowered); Kronos forward-vol gated off",
            "status: OD Strategy A v2 lifecycle gate failed OOS under primary global-embargo replay; hedge overlay and Kronos remain gated off",
        ]:
            hub_text = hub_text.replace(
                old_status,
                "status: OD Strategy A v2 Phase 1 gate failed; Phase 2 hedge frontier diagnostic ran but Kronos remains gated off",
            )
        insert = f"""## Current state (2026-06-01)

- **OD Strategy A v2 lifecycle:** {gate_line} Phase 0 actual token mapping found 5/370 old-heuristic mismatches (1.35%), so future OD accounting uses `asset_id/outcome_index`, not mid-distance inference. {phase2_line} Because Gate 1 still did not clear, Kronos/HAR forward-vol bake-off remains gated off.
- **Prior K6 static-hedge single-fill gate:** superseded by Strategy A v2; it remains useful only as a single-fill diagnostic.

"""
        OD_HUB.write_text(hub_text, encoding="utf-8")
        replace_section(OD_HUB, "## Current state (2026-06-01)", insert)
        text = OD_HUB.read_text(encoding="utf-8")
        old_task = """- [ ] **Strategy A v2** (next critical-path job): (a) relax non-overlap from one-fill-per-market to
  **embargoed multi-fill** (K5 winners take many — biggest power lever); (b) replace raw spot-delta hedge
  with a **call-spread / capped-delta** hedge (cuts the 9–17c hedge cost in non-far buckets; test
  reduced/zero hedge for far buckets where the hedge currently detracts); (c) **pre-register the far-|z|
  family** as the decision gate for this strategy's geometry, re-run OOS (don't inherit far/late from the
  gamma scalp); (d) **verify the outcome-labeling heuristic** (it infers token side from asset_mid — a
  mislabel flips both payoff and delta). Codex."""
        new_task = """- [x] **Strategy A v2:** lifecycle replay plus Phase 2 hedge diagnostic complete; Gate 1 failed under the primary global time embargo
  (`far_absz_ge1_all_tau` OOS lower CI < 0), and the overlay did not lift the primary lower CI above zero."""
        if old_task in text:
            text = text.replace(old_task, new_task, 1)
        text = text.replace(
            "- [x] **Strategy A v2:** lifecycle replay complete; Gate 1 failed under the primary global time embargo\n  (`far_absz_ge1_all_tau` OOS lower CI < 0). Phase 2 hedge overlay stays gated off.",
            new_task,
            1,
        )
        text = text.replace(
            "- [x] **Strategy A v2** (2026-06-01): lifecycle replay complete; Gate 1 failed under global time embargo. Original task:",
            "- [x] **Strategy A v2** (2026-06-01): lifecycle replay plus Phase 2 hedge diagnostic complete; Gate 1 failed under global time embargo and the overlay did not lift the primary lower CI above zero. Original task:",
            1,
        )
        OD_HUB.write_text(text, encoding="utf-8")

    if BRAIN_TODO.exists():
        text = BRAIN_TODO.read_text(encoding="utf-8")
        old = "- [ ] **Strategy A v2** (next critical-path Codex job): (a) relax non-overlap → **embargoed multi-fill**"
        if old in text:
            text = text.replace(old, "- [x] **Strategy A v2** (2026-06-01): lifecycle replay complete; Gate 1 failed under global time embargo. Original task: (a) relax non-overlap → **embargoed multi-fill**", 1)
        text = text.replace(
            "- [x] **Strategy A v2** (2026-06-01): lifecycle replay complete; Gate 1 failed under global time embargo. Original task:",
            "- [x] **Strategy A v2** (2026-06-01): lifecycle replay plus Phase 2 hedge diagnostic complete; Gate 1 failed under global time embargo and the overlay did not lift the primary lower CI above zero. Original task:",
            1,
        )
        marker = "## OD — Options-Delta (state 2026-06-01)"
        if marker in text:
            text = text.replace(
                "**Continuous-hedge approaches closed; static-hedge Strategy A is a validated-mechanism near-miss:**",
                f"**Continuous-hedge approaches closed; OD Strategy A v2 lifecycle gate failed under the primary global-embargo replay:**\n- {gate_line}\n- {phase2_line}\n- Kronos remains gated off unless the unhedged lifecycle gate/capital assumption is explicitly reopened.\n\nHistorical context:",
                1,
            )
            text = text.replace(
                "- Phase 2 hedge overlay and Kronos remain gated off unless the embargo/capital assumption is explicitly reopened.",
                f"- {phase2_line}\n- Kronos remains gated off unless the unhedged lifecycle gate/capital assumption is explicitly reopened.",
                1,
            )
        BRAIN_TODO.write_text(text, encoding="utf-8")


def main() -> None:
    fills = build_fill_ledger(refresh=True)
    episodes = build_episodes(fills)
    phase1_summary = summarize_episodes(episodes)
    overlay = build_phase2_hedge_overlay(episodes, fills)
    ledger = pd.concat([episodes, overlay], ignore_index=True, sort=False)
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(OUT_TRADES, index=False)
    print(f"wrote {OUT_TRADES}", flush=True)
    phase2_summary = summarize_phase2(overlay, phase1_summary)
    summary = pd.concat([phase1_summary, phase2_summary], ignore_index=True, sort=False)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"wrote {OUT_SUMMARY}", flush=True)
    gate_pass = write_note(fills, episodes, summary)
    update_docs(gate_pass, summary)


if __name__ == "__main__":
    main()
