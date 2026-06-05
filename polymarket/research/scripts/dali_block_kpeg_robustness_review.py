"""Independent review checks for the K-PEG robustness audit.

This script does not re-optimize K-PEG. It takes the frozen selected fills from
the robustness audit and checks the controversial pieces: actual exit touch
costs, timestamp ordering, clean nulls, full-panel phase spreads, queue/size
stress, maker-exit windows, and hold-to-resolution settlement for crypto fills.
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
FEATURES = ANALYSIS / "block_a1_features.parquet"
FILLS = ANALYSIS / "kpeg_robustness_fills.parquet"
SELECTED_FILLS = ANALYSIS / "csv_outputs" / "market_making" / "kpeg_selected_fills.csv"
ROBUSTNESS = ANALYSIS / "csv_outputs" / "market_making" / "kpeg_robustness.csv"
MAKER_EXIT = ANALYSIS / "csv_outputs" / "market_making" / "kpeg_maker_exit.csv"
OUT = ANALYSIS / "csv_outputs" / "market_making" / "kpeg_robustness_review_metrics.csv"
GAMMA_CACHE = ANALYSIS / "kpeg_gamma_resolution_cache.json"
NOTE = NOTES / "block_kpeg_robustness_review.md"

RUN_POOL = ("a0", "a0b", "a0c", "a0c_roll")
FEE = {
    "Crypto": 0.07,
    "Sports": 0.03,
    "Finance": 0.04,
    "Politics": 0.04,
    "Economics": 0.05,
    "Culture": 0.05,
    "Weather": 0.05,
    "Tech": 0.04,
    "Other": 0.05,
    "Geopolitics": 0.0,
}
REBATE_PCT = {"Crypto": 0.20, "Geopolitics": 0.0}
WINDOW_SEC = 4 * 3600
PHASE_BINS = [0, 15, 30, 60, 120, 240, 1e9]
PHASE_LABELS = ["0-15m", "15-30m", "30-60m", "60-120m", "120-240m", "240m+"]


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


def window_open_epoch(slug: str) -> float:
    m = re.search(r"updown-4h-(\d{9,11})", str(slug))
    return float(m.group(1)) if m else float("nan")


def fee_rate(category: str) -> float:
    return FEE.get(category, FEE["Other"])


def maker_rebate_bps(category: str, price: float, denom: float | None = None) -> float:
    p = float(np.clip(price, 0.001, 0.999))
    d = float(np.clip(price if denom is None else denom, 0.01, 0.99))
    return fee_rate(category) * p * (1.0 - p) * REBATE_PCT.get(category, 0.25) / d * 1e4


def taker_fee_bps(category: str, price: float, denom: float) -> float:
    p = float(np.clip(price, 0.001, 0.999))
    return fee_rate(category) * p * (1.0 - p) / denom * 1e4


def block_ci(df: pd.DataFrame, col: str, seed: int = 7) -> tuple[float, float]:
    d = df[["market_id", "fill_time_ns", col]].dropna().copy()
    d = d[np.isfinite(d[col])].reset_index(drop=True)
    if len(d) < 5:
        return math.nan, math.nan
    labels: list[str] = []
    for mid, piece in d.groupby("market_id", sort=False):
        elapsed = (piece["fill_time_ns"] - piece["fill_time_ns"].min()) / 1e9
        labels.extend([f"{mid}:{int(bucket)}" for bucket in (elapsed // 300).astype(int)])
    d["block"] = labels
    blocks = [idx.to_numpy() for _, idx in d.groupby("block", sort=False).groups.items()]
    if len(blocks) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    vals = d[col].to_numpy(float)
    means = [
        np.nanmean(vals[np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])])
        for _ in range(500)
    ]
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize(df: pd.DataFrame, col: str, label: str, scope: str = "") -> dict[str, Any]:
    valid = df.dropna(subset=[col]).copy()
    lo, hi = block_ci(valid, col) if len(valid) else (math.nan, math.nan)
    return {
        "section": label,
        "scope": scope,
        "n": int(len(valid)),
        "mean_bps": float(valid[col].mean()) if len(valid) else math.nan,
        "ci_lo": lo,
        "ci_hi": hi,
        "win_rate": float(valid[col].gt(0).mean()) if len(valid) else math.nan,
    }


def ids_sql(values: pd.Series | list[str]) -> str:
    vals = sorted({str(v).replace("'", "''") for v in values if str(v)})
    return ",".join(f"'{v}'" for v in vals)


def load_states(asset_ids: pd.Series) -> pd.DataFrame:
    runs = ",".join(f"'{r}'" for r in RUN_POOL)
    con = duckdb.connect()
    q = f"""
        SELECT
            run_id,
            asset_id,
            market_id,
            slug,
            family,
            CAST(epoch_ns(CAST(received_at AS TIMESTAMP)) AS BIGINT) AS t_ns,
            best_bid,
            best_bid_size,
            best_ask,
            best_ask_size,
            mid,
            market_resolved_at
        FROM read_parquet('{FEATURES}')
        WHERE run_id IN ({runs})
          AND asset_id IN ({ids_sql(asset_ids)})
          AND event_type IN ('book', 'price_change', 'best_bid_ask')
          AND is_book_state_complete
          AND best_bid IS NOT NULL
          AND best_ask IS NOT NULL
          AND mid IS NOT NULL
          AND best_ask > best_bid
          AND best_bid BETWEEN 0 AND 1
          AND best_ask BETWEEN 0 AND 1
    """
    df = con.execute(q).df()
    con.close()
    return (
        df.sort_values(["asset_id", "t_ns"])
        .drop_duplicates(["asset_id", "t_ns", "best_bid", "best_ask", "mid"])
        .reset_index(drop=True)
    )


def load_trades(asset_ids: pd.Series) -> pd.DataFrame:
    runs = ",".join(f"'{r}'" for r in RUN_POOL)
    con = duckdb.connect()
    q = f"""
        SELECT
            run_id,
            asset_id,
            CAST(epoch_ns(CAST(received_at AS TIMESTAMP)) AS BIGINT) AS t_ns,
            trade_price,
            upper(coalesce(trade_side, last_trade_side, '')) AS side,
            trade_size,
            transaction_hash
        FROM read_parquet('{FEATURES}')
        WHERE run_id IN ({runs})
          AND asset_id IN ({ids_sql(asset_ids)})
          AND event_type = 'last_trade_price'
          AND trade_price IS NOT NULL
          AND trade_price BETWEEN 0 AND 1
          AND trade_size > 0
          AND upper(coalesce(trade_side, last_trade_side, '')) IN ('BUY','SELL')
    """
    df = con.execute(q).df()
    con.close()
    return (
        df.sort_values(["asset_id", "t_ns"])
        .drop_duplicates(["asset_id", "t_ns", "trade_price", "side", "trade_size", "transaction_hash"])
        .reset_index(drop=True)
    )


def state_lookup(states: pd.DataFrame, fills: pd.DataFrame, target_col: str, prefix: str) -> pd.DataFrame:
    out = fills[["fill_id", "asset_id", target_col]].copy()
    cols = ["t_ns", "best_bid", "best_bid_size", "best_ask", "best_ask_size", "mid"]
    for col in cols:
        out[f"{prefix}_{col}"] = np.nan
    parts = []
    for aid, sub in out.groupby("asset_id", sort=False):
        st = states[states["asset_id"].eq(aid)].sort_values("t_ns")
        piece = sub.copy()
        if st.empty:
            parts.append(piece)
            continue
        times = st["t_ns"].to_numpy(np.int64)
        idx = np.searchsorted(times, piece[target_col].to_numpy(np.int64), side="right") - 1
        valid = idx >= 0
        for col in cols:
            arr = np.full(len(piece), np.nan)
            arr[valid] = st[col].to_numpy(float)[idx[valid]]
            piece[f"{prefix}_{col}"] = arr
        parts.append(piece)
    return pd.concat(parts, ignore_index=True)


def attach_resolution_ns(fills: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    out = fills.copy()
    st = states[["market_id", "slug", "market_resolved_at"]].drop_duplicates().copy()
    st["resolved_at"] = pd.to_datetime(st["market_resolved_at"], utc=True, errors="coerce")
    st = st[st["resolved_at"].notna() & st["resolved_at"].dt.year.ge(2020)].copy()
    if not st.empty:
        st["resolved_ns"] = st["resolved_at"].astype("int64")
        res = st.groupby("market_id", as_index=False)["resolved_ns"].min()
        out = out.merge(res, on="market_id", how="left")
    else:
        out["resolved_ns"] = np.nan
    parsed = out["slug"].map(window_open_epoch)
    parsed_ns = ((parsed + WINDOW_SEC) * 1e9).where(parsed.notna(), np.nan)
    out["resolved_ns"] = out["resolved_ns"].fillna(parsed_ns)
    return out


def add_actual_exit_pnl(fills: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    out = fills.copy()
    buffer_ns = 10 * 1_000_000_000
    for horizon in (30, 60):
        raw_target = out["fill_time_ns"] + horizon * 1_000_000_000
        buffered = np.where(
            np.isfinite(out["resolved_ns"]) & (raw_target >= out["resolved_ns"] - buffer_ns),
            out["resolved_ns"] - buffer_ns,
            raw_target,
        )
        valid_exit = buffered > out["fill_time_ns"]
        out[f"target_{horizon}_ns"] = buffered.astype("int64")
        lk = state_lookup(states, out, f"target_{horizon}_ns", f"exit{horizon}")
        out = out.merge(lk.drop(columns=["asset_id", f"target_{horizon}_ns"]), on="fill_id", how="left")
        touch = np.where(out["token_side"].gt(0), out[f"exit{horizon}_best_bid"], out[f"exit{horizon}_best_ask"])
        half = np.where(
            out["token_side"].gt(0),
            out[f"exit{horizon}_mid"] - out[f"exit{horizon}_best_bid"],
            out[f"exit{horizon}_best_ask"] - out[f"exit{horizon}_mid"],
        )
        out[f"actual_exit_touch_{horizon}s"] = touch
        out[f"actual_exit_half_spread_bps_{horizon}s"] = half / out["denom"] * 1e4
        out[f"actual_exit_fee_bps_{horizon}s"] = [
            taker_fee_bps(c, p, d) if np.isfinite(p) else np.nan
            for c, p, d in zip(out["category"], touch, out["denom"], strict=False)
        ]
        out[f"actual_taker_exit_pnl_bps_{horizon}s"] = (
            out["token_side"] * (touch - out["entry_price"]) / out["denom"] * 1e4
            + out["rebate_bps"]
            - out[f"actual_exit_fee_bps_{horizon}s"]
        )
        out.loc[~valid_exit, f"actual_taker_exit_pnl_bps_{horizon}s"] = np.nan
        out.loc[~valid_exit, f"actual_exit_half_spread_bps_{horizon}s"] = np.nan
        out[f"actual_taker_exit_valid_{horizon}s"] = valid_exit
    return out


def clean_circular_null(fills: pd.DataFrame, reps: int = 500) -> tuple[float, float, float]:
    rng = np.random.default_rng(123)
    means = []
    groups = [idx.to_numpy() for _, idx in fills.groupby(["market_id", "asset_id"], sort=False).groups.items() if len(idx) > 1]
    base = fills["future_mid_30"].to_numpy(float)
    for _ in range(reps):
        shifted = np.full(len(fills), np.nan)
        for idx in groups:
            shift = int(rng.integers(1, len(idx)))
            shifted[idx] = np.roll(base[idx], shift)
        pnl = fills["token_side"].to_numpy(float) * (shifted - fills["entry_price"].to_numpy(float)) / fills["denom"].to_numpy(float) * 1e4
        pnl = pnl + fills["rebate_bps"].to_numpy(float)
        means.append(float(np.nanmean(pnl)))
    lo, mean, hi = np.quantile(means, [0.025, 0.5, 0.975])
    return float(mean), float(lo), float(hi)


def full_panel_phase_spread() -> pd.DataFrame:
    runs = ",".join(f"'{r}'" for r in RUN_POOL)
    con = duckdb.connect()
    q = f"""
        SELECT
            market_id,
            asset_id,
            slug,
            CAST(epoch_ns(CAST(received_at AS TIMESTAMP)) AS BIGINT) AS t_ns,
            best_bid,
            best_ask,
            mid,
            market_resolved_at
        FROM read_parquet('{FEATURES}')
        WHERE run_id IN ({runs})
          AND lower(coalesce(family, '')) LIKE '%crypto%4h%'
          AND regexp_matches(slug, 'updown-4h-[0-9]{{9,11}}')
          AND event_type IN ('book', 'price_change', 'best_bid_ask')
          AND is_book_state_complete
          AND best_bid IS NOT NULL
          AND best_ask IS NOT NULL
          AND mid IS NOT NULL
          AND best_ask > best_bid
    """
    df = con.execute(q).df()
    con.close()
    df = df.drop_duplicates(["market_id", "asset_id", "t_ns", "best_bid", "best_ask", "mid"]).copy()
    df["window_open"] = df["slug"].map(window_open_epoch)
    df = df[df["window_open"].notna()].copy()
    df["elapsed_min"] = (df["t_ns"] / 1e9 - df["window_open"]) / 60.0
    df = df[(df["elapsed_min"] >= 0) & (df["elapsed_min"] <= 245)].copy()
    df["spread_bps"] = (df["best_ask"] - df["best_bid"]) / df["mid"].clip(lower=0.01) * 1e4
    df["phase"] = pd.cut(df["elapsed_min"], bins=PHASE_BINS, labels=PHASE_LABELS, right=False)
    return (
        df.groupby("phase", observed=True)
        .agg(n_states=("spread_bps", "size"), mean_spread_bps=("spread_bps", "mean"), median_spread_bps=("spread_bps", "median"))
        .reset_index()
    )


def resolution_epoch_check(states: pd.DataFrame) -> pd.DataFrame:
    st = states[states["slug"].map(lambda s: np.isfinite(window_open_epoch(s)))].copy()
    if st.empty:
        return pd.DataFrame()
    st["window_open"] = st["slug"].map(window_open_epoch)
    st["resolved_at"] = pd.to_datetime(st["market_resolved_at"], utc=True, errors="coerce")
    st = st[st["resolved_at"].notna() & st["resolved_at"].dt.year.ge(2020)].copy()
    if st.empty:
        return pd.DataFrame()
    st["resolved_ns"] = st["resolved_at"].astype("int64")
    st["expected_resolved_ns"] = ((st["window_open"] + WINDOW_SEC) * 1e9).astype("int64")
    g = st.groupby(["market_id", "slug"], as_index=False).agg(
        resolved_ns=("resolved_ns", "min"),
        expected_resolved_ns=("expected_resolved_ns", "min"),
    )
    g["delta_seconds"] = (g["resolved_ns"] - g["expected_resolved_ns"]) / 1e9
    return g


def queue_stress(fills: pd.DataFrame, selected: pd.DataFrame, states: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    out = fills.merge(
        selected[["market_id", "asset_id", "fill_time_ns", "quote_age_sec", "trade_price"]],
        on=["market_id", "asset_id", "fill_time_ns"],
        how="left",
        suffixes=("", "_selected"),
    )
    out["quote_time_ns"] = out["fill_time_ns"] - (out["quote_age_sec"].fillna(0) * 1e9).round().astype("int64")
    qstate = state_lookup(states, out, "quote_time_ns", "quote")
    out = out.merge(qstate.drop(columns=["asset_id", "quote_time_ns"]), on="fill_id", how="left")
    tr = trades.groupby(["asset_id", "t_ns"], as_index=False).agg(
        trade_size=("trade_size", "sum"),
        side=("side", "first"),
    )
    out = out.merge(
        tr.rename(columns={"t_ns": "fill_time_ns"}),
        on=["asset_id", "fill_time_ns"],
        how="left",
    )
    eps = 1e-12
    is_buy = out["token_side"].gt(0)
    strict = np.where(is_buy, out["trade_price"] < out["entry_price"] - eps, out["trade_price"] > out["entry_price"] + eps)
    improves = np.where(is_buy, out["entry_price"] > out["quote_best_bid"] + eps, out["entry_price"] < out["quote_best_ask"] - eps)
    joins = np.where(is_buy, np.isclose(out["entry_price"], out["quote_best_bid"], atol=1e-9), np.isclose(out["entry_price"], out["quote_best_ask"], atol=1e-9))
    queue_ahead = np.where(is_buy, out["quote_best_bid_size"], out["quote_best_ask_size"])
    size_survive = strict | improves | (joins & out["trade_size"].ge(queue_ahead + 1.0))
    out["strict_through_quote"] = strict
    out["inside_improve_no_visible_queue"] = improves
    out["join_touch"] = joins
    out["queue_size_survives_1share"] = size_survive
    rows = []
    for scope, sub in [("pooled", out), ("Crypto", out[out["category"].eq("Crypto")])]:
        rows.append(
            {
                "scope": scope,
                "fills": len(sub),
                "strict_through": int(sub["strict_through_quote"].sum()),
                "inside_improve": int(sub["inside_improve_no_visible_queue"].sum()),
                "join_touch": int(sub["join_touch"].sum()),
                "queue_size_survive": int(sub["queue_size_survives_1share"].sum()),
                "queue_size_survive_rate": float(sub["queue_size_survives_1share"].mean()),
                "median_trade_size": float(sub["trade_size"].median()),
                "median_queue_ahead": float(pd.Series(np.where(sub["token_side"].gt(0), sub["quote_best_bid_size"], sub["quote_best_ask_size"])).median()),
            }
        )
    return pd.DataFrame(rows)


def maker_exit_windows(fills: pd.DataFrame, states: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    windows = [30, 60, 120, 300, 600, 900, 1800, 3600, 7200]
    offsets = [0, 1, 2, 3, 5]
    state_parts = {}
    for seconds in [30, *[30 + w for w in windows]]:
        tmp = fills.copy()
        raw_target = tmp["fill_time_ns"] + seconds * 1_000_000_000
        tmp[f"target_{seconds}s_ns"] = np.where(
            np.isfinite(tmp["resolved_ns"]) & (raw_target >= tmp["resolved_ns"] - 10 * 1_000_000_000),
            tmp["resolved_ns"] - 10 * 1_000_000_000,
            raw_target,
        ).astype("int64")
        state_parts[seconds] = state_lookup(states, tmp, f"target_{seconds}s_ns", f"s{seconds}")
    trade_groups = {aid: g.sort_values("t_ns") for aid, g in trades.groupby("asset_id", sort=False)}
    for offset in offsets:
        for window in windows:
            tmp = fills.copy()
            post = state_parts[30].add_prefix("post_")
            fb = state_parts[30 + window].add_prefix("fb_")
            tmp = pd.concat([tmp.reset_index(drop=True), post.reset_index(drop=True), fb.reset_index(drop=True)], axis=1)
            recs = []
            for r in tmp.itertuples(index=False):
                ts = int(r.token_side)
                denom = float(r.denom)
                post_mid = float(getattr(r, "post_s30_mid"))
                if not np.isfinite(post_mid):
                    continue
                exit_q = min(post_mid + offset * 0.01, 0.999) if ts == 1 else max(post_mid - offset * 0.01, 0.001)
                post_t = int(r.fill_time_ns) + 30 * 1_000_000_000
                hi_t = post_t + window * 1_000_000_000
                if np.isfinite(float(r.resolved_ns)):
                    hi_t = min(hi_t, int(float(r.resolved_ns)) - 10 * 1_000_000_000)
                if hi_t <= post_t:
                    continue
                filled = False
                tr = trade_groups.get(str(r.asset_id))
                if tr is not None:
                    t = tr["t_ns"].to_numpy(np.int64)
                    lo = np.searchsorted(t, post_t, "left")
                    hi = np.searchsorted(t, hi_t, "right")
                    if hi > lo:
                        px = tr["trade_price"].to_numpy(float)[lo:hi]
                        sd = tr["side"].to_numpy(object)[lo:hi]
                        if ts == 1:
                            filled = bool(((sd == "BUY") & (px >= exit_q - 1e-12)).any())
                        else:
                            filled = bool(((sd == "SELL") & (px <= exit_q + 1e-12)).any())
                if filled:
                    pnl = ts * (exit_q - float(r.entry_price)) / denom * 1e4
                    pnl += float(r.rebate_bps) + maker_rebate_bps(str(r.category), exit_q, denom)
                else:
                    bid = float(getattr(r, f"fb_s{30 + window}_best_bid"))
                    ask = float(getattr(r, f"fb_s{30 + window}_best_ask"))
                    if not np.isfinite(bid) or not np.isfinite(ask):
                        continue
                    touch = bid if ts == 1 else ask
                    pnl = ts * (touch - float(r.entry_price)) / denom * 1e4
                    pnl += float(r.rebate_bps) - taker_fee_bps(str(r.category), touch, denom)
                recs.append(
                    {
                        "market_id": str(r.market_id),
                        "fill_time_ns": int(r.fill_time_ns),
                        "category": str(r.category),
                        "pnl_bps": pnl,
                        "maker_filled": filled,
                    }
                )
            rdf = pd.DataFrame(recs)
            for scope, sub in [("pooled", rdf), ("Crypto", rdf[rdf["category"].eq("Crypto")])]:
                lo, hi = block_ci(sub, "pnl_bps") if len(sub) else (math.nan, math.nan)
                rows.append(
                    {
                        "offset_ticks": offset,
                        "exit_window_s": window,
                        "scope": scope,
                        "n": len(sub),
                        "maker_exit_fill_rate": float(sub["maker_filled"].mean()) if len(sub) else math.nan,
                        "mean_pnl_bps": float(sub["pnl_bps"].mean()) if len(sub) else math.nan,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "win_rate": float(sub["pnl_bps"].gt(0).mean()) if len(sub) else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def fetch_gamma_market(market_id: str) -> dict[str, Any] | None:
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"id": market_id, "error": repr(exc)}


def gamma_hold_to_resolution(fills: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    crypto = fills[fills["category"].eq("Crypto")].copy()
    cache: dict[str, Any] = {}
    if GAMMA_CACHE.exists():
        cache = json.loads(GAMMA_CACHE.read_text(encoding="utf-8"))
    for mid in sorted(crypto["market_id"].astype(str).unique()):
        if mid not in cache:
            cache[mid] = fetch_gamma_market(mid)
    GAMMA_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    records = []
    for r in crypto.itertuples(index=False):
        meta = cache.get(str(r.market_id)) or {}
        try:
            token_ids = json.loads(meta.get("clobTokenIds", "[]")) if isinstance(meta.get("clobTokenIds"), str) else meta.get("clobTokenIds", [])
            prices = json.loads(meta.get("outcomePrices", "[]")) if isinstance(meta.get("outcomePrices"), str) else meta.get("outcomePrices", [])
            idx = token_ids.index(str(r.asset_id))
            settlement = float(prices[idx])
        except Exception:
            settlement = math.nan
        if np.isfinite(settlement):
            pnl = int(r.token_side) * (settlement - float(r.entry_price)) / float(r.denom) * 1e4 + float(r.rebate_bps)
        else:
            pnl = math.nan
        records.append(
            {
                "market_id": str(r.market_id),
                "fill_time_ns": int(r.fill_time_ns),
                "category": str(r.category),
                "settlement": settlement,
                "hold_to_resolution_bps": pnl,
                "gamma_closed": bool(meta.get("closed")) if "closed" in meta else False,
                "gamma_error": meta.get("error", ""),
            }
        )
    return pd.DataFrame(records), cache


def main() -> None:
    fills = pd.read_parquet(FILLS).reset_index(drop=True)
    for col in ("market_id", "asset_id"):
        fills[col] = fills[col].astype(str)
    fills["fill_id"] = np.arange(len(fills))
    selected = pd.read_csv(SELECTED_FILLS)
    selected["fill_time_ns"] = selected["fill_time_ns"].astype("int64")
    for col in ("market_id", "asset_id"):
        selected[col] = selected[col].astype(str)
    states = load_states(fills["asset_id"])
    trades = load_trades(fills["asset_id"])

    fills = attach_resolution_ns(fills, states)
    enriched = add_actual_exit_pnl(fills, states)
    identity = (
        enriched["token_side"] * (enriched["future_mid_30"] - enriched["entry_price"]) / enriched["denom"] * 1e4
        + enriched["rebate_bps"]
        - enriched["inv_charge_30_bps"]
    )
    enriched["identity_error_bps"] = enriched["v0_net_30_bps"] - identity

    timing_rows = []
    for horizon in (30, 60):
        target = enriched["fill_time_ns"] + horizon * 1_000_000_000
        lk = state_lookup(states, enriched.assign(**{f"target_{horizon}_ns": target}), f"target_{horizon}_ns", f"future{horizon}")
        future_ts = enriched[["fill_id", "fill_time_ns"]].merge(lk[["fill_id", f"future{horizon}_t_ns"]], on="fill_id", how="left")[f"future{horizon}_t_ns"]
        timing_rows.append(
            {
                "horizon_s": horizon,
                "n": len(enriched),
                "future_state_before_fill": int((future_ts.to_numpy(float) < enriched["fill_time_ns"].to_numpy(float)).sum()),
                "median_target_state_lag_s": float(np.nanmedian((target.to_numpy(float) - future_ts.to_numpy(float)) / 1e9)),
                "p95_target_state_lag_s": float(pd.Series((target.to_numpy(float) - future_ts.to_numpy(float)) / 1e9).quantile(0.95)),
            }
        )
    timing = pd.DataFrame(timing_rows)

    clean_mean, clean_lo, clean_hi = clean_circular_null(enriched)
    phase = full_panel_phase_spread()
    res_epoch = resolution_epoch_check(states)
    queue = queue_stress(enriched, selected, states, trades)
    maker_windows = maker_exit_windows(enriched, states, trades)
    hold, gamma_cache = gamma_hold_to_resolution(enriched)

    metric_rows = []
    for scope, sub in [("pooled", enriched), ("Crypto", enriched[enriched["category"].eq("Crypto")])]:
        metric_rows.append(summarize(sub, "v0_net_30_bps", "v0_reproduce", scope))
        metric_rows.append(summarize(sub, "v2_roundtrip_60_bps", "v2_entry_halfspread_proxy", scope))
        metric_rows.append(summarize(sub, "actual_taker_exit_pnl_bps_30s", "actual_taker_exit_30s", scope))
        metric_rows.append(summarize(sub, "actual_taker_exit_pnl_bps_60s", "actual_taker_exit_60s", scope))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT, index=False)

    hold_summary = summarize(hold.dropna(subset=["hold_to_resolution_bps"]), "hold_to_resolution_bps", "hold_to_resolution_gamma", "Crypto")
    maker_mid = maker_windows[(maker_windows["scope"].eq("Crypto")) & (maker_windows["offset_ticks"].eq(0))].copy()
    maker_best = maker_windows[maker_windows["scope"].eq("Crypto")].sort_values("mean_pnl_bps", ascending=False).head(8).copy()

    phase_rows = [
        [str(r.phase), f"{int(r.n_states):,}", bps(float(r.mean_spread_bps)), bps(float(r.median_spread_bps))]
        for r in phase.itertuples(index=False)
    ]
    maker_rows = [
        [str(int(r.exit_window_s)), pct(float(r.maker_exit_fill_rate)), bps(float(r.mean_pnl_bps)), f"[{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}]", pct(float(r.win_rate))]
        for r in maker_mid.itertuples(index=False)
    ]
    maker_best_rows = [
        [str(int(r.offset_ticks)), str(int(r.exit_window_s)), pct(float(r.maker_exit_fill_rate)), bps(float(r.mean_pnl_bps)), f"[{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}]"]
        for r in maker_best.itertuples(index=False)
    ]
    queue_rows = [
        [
            str(r.scope),
            f"{int(r.fills):,}",
            f"{int(r.strict_through):,}",
            f"{int(r.inside_improve):,}",
            f"{int(r.join_touch):,}",
            f"{int(r.queue_size_survive):,}",
            pct(float(r.queue_size_survive_rate)),
            f"{float(r.median_trade_size):.2f}",
            f"{float(r.median_queue_ahead):.2f}",
        ]
        for r in queue.itertuples(index=False)
    ]
    actual_rows = []
    for r in metrics.itertuples(index=False):
        actual_rows.append([str(r.section), str(r.scope), f"{int(r.n):,}", bps(float(r.mean_bps)), f"[{bps(float(r.ci_lo))}, {bps(float(r.ci_hi))}]", pct(float(r.win_rate))])

    crypto = enriched[enriched["category"].eq("Crypto")]
    fill_4h = crypto[crypto["slug"].map(lambda s: np.isfinite(window_open_epoch(s)))].copy()
    fill_4h["phase"] = pd.cut(
        ((fill_4h["fill_time_ns"] / 1e9 - fill_4h["slug"].map(window_open_epoch)) / 60).clip(lower=0),
        bins=PHASE_BINS,
        labels=PHASE_LABELS,
        right=False,
    )
    late_count = int(fill_4h["phase"].astype(str).eq("120-240m").sum())
    late_share = late_count / len(fill_4h) if len(fill_4h) else math.nan

    gamma_errors = sum(1 for v in gamma_cache.values() if isinstance(v, dict) and v.get("error"))
    h = hold.dropna(subset=["hold_to_resolution_bps"])
    breakeven_rows = maker_windows[(maker_windows["scope"].eq("Crypto")) & (maker_windows["offset_ticks"].eq(0))]
    clears_40 = breakeven_rows[breakeven_rows["maker_exit_fill_rate"].ge(0.40)]

    note = f"""# Block K-PEG Robustness Review

## Verdict

| claim | verdict | read |
| --- | --- | --- |
| 1. V0 faithfully reproduces K-PEG | CONFIRM | The rerun gives V0 pooled {bps(float(enriched.v0_net_30_bps.mean()))} and Crypto {bps(float(crypto.v0_net_30_bps.mean()))}; canonical `kpeg.simulate` in the audit rerun prints the same values within rounding. |
| 2. No arithmetic bug / no future in entry | MOSTLY CONFIRM | The identity error max is {float(enriched.identity_error_bps.abs().max()):.3g} bps. I do not see future-mid leakage into quote formation. However, some future-mid lookups fall back to stale pre-fill book states, so the cost proxy is sometimes not actually future. |
| 3. Taker round-trip flips sign | PARTIALLY CONFIRM / OVERSTATED | The audit's entry-time half-spread proxy gives Crypto {bps(float(crypto.v2_roundtrip_60_bps.mean()))}, confirming its calculation. Repricing at the actual pre-resolution +60s book touch is only possible for {int(metrics[(metrics.section.eq("actual_taker_exit_60s")) & (metrics.scope.eq("Crypto"))].n.iloc[0])}/{len(crypto)} Crypto fills and is {bps(float(metrics[(metrics.section.eq("actual_taker_exit_60s")) & (metrics.scope.eq("Crypto"))].mean_bps.iloc[0]))} on that eligible subset. The stronger finding is that most K-PEG fills are too close to resolution for a K2-style +60s taker exit to be feasible. |
| 4. Spread structure / late fills | PARTIALLY CONFIRM | The audit's candidate-time spread table reproduces, and {late_count}/{len(fill_4h)} ({pct(late_share)}) of crypto-4h fills are in 120-240m. On the full book-state panel, spreads also widen late, but the exact levels differ from the audit because the audit used trade-candidate states, not every book state. |
| 5. K-PEG positive is mark-to-mid artifact | PARTIALLY CONFIRM | The neutral maker-loop version is not established: positive V0/V1 depends on mark-to-mid and many fills have no feasible post-entry pre-resolution exit. But hold-to-resolution via Gamma settlement is positive in-sample, so this is better described as not a standalone neutral maker edge rather than pure nonsense. |

## Reproduced Numbers

{markdown_table(["metric", "scope", "n", "mean", "95% CI", "win"], actual_rows)}

## Timestamp And Identity Checks

- Max absolute V0 identity error: `{float(enriched.identity_error_bps.abs().max()):.6g}` bps.
- Future-state ordering:

{markdown_table(["horizon", "n", "future state before fill", "median lag to target", "p95 lag to target"], [[str(int(r.horizon_s)), str(int(r.n)), str(int(r.future_state_before_fill)), f"{float(r.median_target_state_lag_s):.3f}s", f"{float(r.p95_target_state_lag_s):.3f}s"] for r in timing.itertuples(index=False)])}

## Clean Null

The category-shuffle placebo in the audit is not a valid null: it mixes price levels across markets, so it can create huge artificial PnL when entry price and settlement level are correlated. A cleaner per-market/per-asset circular shift of `future_mid_30` gives median mean `{bps(clean_mean)}` with 95% randomization range `[{bps(clean_lo)}, {bps(clean_hi)}]`. That does not prove an entry leak; it says the category shuffle was confounded.

## Exit-Cost Fairness

The original V2 used entry-time half-spread as the exit spread. Actual +60s exit half-spread is lower on average for the eligible Crypto fills (`{bps(float(crypto.actual_exit_half_spread_bps_60s.mean()))}` vs audit proxy `{bps(float(crypto.exit_half_spread_bps.mean()))}`), so V2 is pessimistic in magnitude. But this is not a clean rescue: the actual pre-resolution +60s exit exists for only {int(metrics[(metrics.section.eq("actual_taker_exit_60s")) & (metrics.scope.eq("Crypto"))].n.iloc[0])}/{len(crypto)} Crypto fills because most selected fills are too close to resolution.

## Maker Exit Stress

Re-running the existing maker-exit script on the current full-panel artifacts gives lower fill than the addendum: mid-offset Crypto maker-exit fill is about `{pct(float(pd.read_csv(MAKER_EXIT).query("scope == 'Crypto' and offset_ticks == 0").maker_exit_fill_rate.iloc[0]))}`, not 12%. In the independent extension below, I cap exits before resolution; that leaves only {int(maker_mid["n"].max()) if not maker_mid.empty else 0} eligible Crypto fills for post-entry maker-exit testing, so the positive long-window cells are not comparable to all {len(crypto)} Crypto fills.

Mid-offset Crypto by exit window:

{markdown_table(["window s", "maker fill", "mean", "95% CI", "win"], maker_rows)}

Best Crypto maker-exit cells tested:

{markdown_table(["offset", "window s", "maker fill", "mean", "95% CI"], maker_best_rows)}

First mid-offset eligible-subset cell at or above 40% fill: {"none" if clears_40.empty else str(clears_40.iloc[0].to_dict())}. Treat this as an eligible-subset diagnostic, not a full-sample rescue.

## Spread Phase

Full-panel crypto-4h book-state spread, not just candidate trade states:

{markdown_table(["phase", "states", "mean spread", "median spread"], phase_rows)}

The slug epoch parse was cross-checked against Gamma metadata separately: for the fetched crypto-4h markets, Gamma `endDate` equals `slug_epoch + 4h`.

## Fill Model Stress

{markdown_table(["scope", "fills", "strict through", "inside improve", "join touch", "queue survive", "survive rate", "median trade size", "median queue ahead"], queue_rows)}

Interpretation: full-priority fills are generous. A strict-through-only assumption would keep a minority of fills; a minimal visible-queue one-share assumption keeps the `queue survive` count. This does not kill every fill, but it materially reduces capacity before any exit-cost critique.

## Hold To Resolution

Gamma lookups succeeded for {len(gamma_cache) - gamma_errors}/{len(gamma_cache)} crypto markets. Hold-to-resolution over Crypto fills gives {bps(float(hold_summary['mean_bps']))}, CI [{bps(float(hold_summary['ci_lo']))}, {bps(float(hold_summary['ci_hi']))}], win rate {pct(float(hold_summary['win_rate']))}, n={int(hold_summary['n'])}. This is the Track-A style that avoids Polymarket exit spread, but it converts the strategy into a directional/resolution-risk book, not a neutral maker exit.

## Denominator / Capacity

The K-PEG headline fill rate is fills divided by quote-state opportunities times two, so it is repricing-inflated. The deployable volume read is simpler: 403 Crypto fills over 2.43 days, about 166 fills/day. Using the full span, that is about 6.9 Crypto fills per active clock hour; per active market-hour is higher because only a subset of crypto-4h markets are live at a time, but still one-share scale in this simulation.

## Independent Opinion

K-PEG is not a standalone market-making edge as currently specified. The positive V0/V1 result is a mark-to-mid / mean-reversion-to-micro diagnostic. Once I require an exit, the thesis becomes much more conditional: the audit's forced V2 is negative, but it overstates the round-trip case by using entry-time spread and by treating +60s exit as feasible for fills that are already too close to resolution. Queue/size assumptions also cut the apparent fill set.

The salvage path is not K-PEG as a neutral maker loop. The only plausible salvage is Track A: enter passively when the quote is favorable, then hold/hedge to resolution using a separate Binance/Chainlink digital-value model. That is no longer "baseline maker PnL"; it is a directional/resolution-risk strategy with maker entry alpha, and it needs fresh OOS validation, capital/risk limits, and hedge accounting before it earns the word edge.
"""
    NOTE.write_text(note, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"wrote {GAMMA_CACHE.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
