"""OD-RV scoping: PM daily crypto digitals vs Deribit 1-day options.

This is a feasibility/data-gate script, not a strategy backtest. It checks
whether the single captured PM daily BTC/ETH window can be compared against
historical Deribit option-chain prices and, where possible, builds a degraded
call-spread digital basis from public Deribit chart OHLC.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import cents, markdown_table, number, pct
from od_strategy_a_v3 import normalize_markdown_wrapping


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
BRAIN_TODO = REPO / "brain" / "TODO.md"
OD_HUB = NOTES / "options_delta" / "strat_options_delta.md"

DAILY_SURFACE = ANALYSIS / "cache" / "k2v2_daily_model_surface.parquet"
PM_LOB = ANALYSIS / "block_a1_features.parquet"
CSV_OUT = ANALYSIS / "csv_outputs" / "options_delta"
PLOTS = ANALYSIS / "plots" / "options_delta"

NOTE = NOTES / "options_delta" / "od_rv_deribit_daily_scoping_findings.md"
OUT_GATE = CSV_OUT / "od_rv_deribit_daily_data_gate.csv"
OUT_BASIS = CSV_OUT / "od_rv_deribit_daily_basis.csv"
OUT_SUMMARY = CSV_OUT / "od_rv_deribit_daily_summary.csv"
OUT_SETTLEMENT = CSV_OUT / "od_rv_deribit_daily_settlement_mismatch.csv"
OUT_CHARTS = ANALYSIS / "od_rv_deribit_daily_deribit_charts.parquet"
BASIS_PLOT = PLOTS / "od_rv_deribit_daily_basis.png"

DERIBIT_BASE = "https://www.deribit.com/api/v2"
TARGET_EXPIRY = "28MAY26"
PM_DAILY_SLUGS = {
    "BTC": "bitcoin-up-or-down-on-may-28-2026",
    "ETH": "ethereum-up-or-down-on-may-28-2026",
}

# Tight brackets around the PM window-open strike, plus wider brackets for a
# replication-error proxy. Strikes are chosen from public Deribit instruments
# that returned chart data in Phase 0 probes.
SPREADS = {
    "BTC": {"tight": (75_000, 75_500), "wide": (74_500, 75_500)},
    "ETH": {"tight": (2_050, 2_075), "wide": (2_050, 2_100)},
}
DERIBIT_FEE_PER_CONTRACT = 0.0003
PM_TAKER_FEE_RATE_CRYPTO = 0.07


def fmt_ci_value(value: float) -> str:
    return "n/a" if not np.isfinite(value) else cents(value)


def local_deribit_artifacts() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((ANALYSIS).rglob("*")):
        if not path.is_file() or "deribit" not in path.name.lower() and "dvol" not in path.name.lower():
            continue
        if path.name.startswith("od_rv_deribit_daily"):
            continue
        if path.suffix not in {".parquet", ".csv"}:
            continue
        row: dict[str, Any] = {
            "source": "local_artifact",
            "path": str(path.relative_to(ROOT)),
            "exists": True,
            "kind": "unknown",
            "rows": math.nan,
            "has_per_instrument_mark_or_iv": False,
            "notes": "",
        }
        if path.suffix == ".parquet":
            try:
                df = pd.read_parquet(path)
                cols = {str(c).lower() for c in df.columns}
                row["rows"] = int(len(df))
                if {"instrument_name", "mark_iv"}.issubset(cols) or {"instrument_name", "mark_price"}.issubset(cols):
                    row["kind"] = "per_instrument_option_history"
                    row["has_per_instrument_mark_or_iv"] = True
                elif "dvol" in " ".join(cols) or "dvol_sigma_ann" in cols:
                    row["kind"] = "dvol_index"
                    row["notes"] = "DVOL only; not a per-option surface."
            except Exception as exc:  # pragma: no cover - diagnostic row
                row["notes"] = repr(exc)
        elif path.suffix == ".csv":
            row["kind"] = "csv_report"
            row["notes"] = "CSV report, not reusable per-option history."
        rows.append(row)
    return pd.DataFrame(rows)


def deribit_get(client: httpx.Client, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    resp = client.get(f"{DERIBIT_BASE}/{endpoint}", params=params, timeout=20)
    try:
        return resp.json()
    except Exception:
        return {"error": {"message": f"http {resp.status_code}: {resp.text[:200]}"}}


def chart_instrument(asset: str, strike: int, option_type: str = "C") -> str:
    return f"{asset}-{TARGET_EXPIRY}-{strike}-{option_type}"


def fetch_deribit_chart(client: httpx.Client, instrument: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    js = deribit_get(
        client,
        "public/get_tradingview_chart_data",
        {
            "instrument_name": instrument,
            "start_timestamp": int(start.timestamp() * 1000),
            "end_timestamp": int(end.timestamp() * 1000),
            "resolution": "60",
        },
    )
    if "error" in js:
        return pd.DataFrame({"instrument": [instrument], "error": [str(js["error"])]})
    res = js.get("result", {}) or {}
    ticks = res.get("ticks", []) or []
    out = pd.DataFrame(
        {
            "instrument": instrument,
            "ts": pd.to_datetime(ticks, unit="ms", utc=True),
            "open": res.get("open", []) or [],
            "high": res.get("high", []) or [],
            "low": res.get("low", []) or [],
            "close": res.get("close", []) or [],
            "volume": res.get("volume", []) or [],
            "cost": res.get("cost", []) or [],
            "status": res.get("status", ""),
        }
    )
    out["error"] = ""
    return out


def probe_deribit(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    charts: list[pd.DataFrame] = []
    with httpx.Client(headers={"User-Agent": "epsilon-quant-research-od-rv/1.0"}) as client:
        for asset in ("BTC", "ETH"):
            inst = deribit_get(client, "public/get_instruments", {"currency": asset, "kind": "option", "expired": "true"})
            result = inst.get("result", []) if "error" not in inst else []
            near = []
            for item in result:
                exp = pd.to_datetime(item.get("expiration_timestamp"), unit="ms", utc=True)
                if start - pd.Timedelta(days=1) <= exp <= end + pd.Timedelta(days=1):
                    near.append(item.get("instrument_name"))
            rows.append(
                {
                    "source": "deribit_api",
                    "check": f"{asset} get_instruments expired=true",
                    "status": "pass_empty_target_date" if "error" not in inst else "fail",
                    "detail": f"returned {len(result)} expired option instruments; {len(near)} near PM date",
                    "endpoint": f"{DERIBIT_BASE}/public/get_instruments",
                }
            )
        for asset, spec in SPREADS.items():
            for strike in sorted(set(spec["tight"] + spec["wide"])):
                instrument = chart_instrument(asset, strike, "C")
                for endpoint, params in [
                    ("public/get_book_summary_by_instrument", {"instrument_name": instrument}),
                    (
                        "public/get_mark_price_history",
                        {
                            "instrument_name": instrument,
                            "start_timestamp": int(start.timestamp() * 1000),
                            "end_timestamp": int(end.timestamp() * 1000),
                        },
                    ),
                ]:
                    js = deribit_get(client, endpoint, params)
                    if "error" in js:
                        status = "fail_expired_or_unavailable"
                        detail = str(js["error"])
                    else:
                        status = "pass"
                        detail = str(js.get("result", ""))[:200]
                    rows.append(
                        {
                            "source": "deribit_api",
                            "check": f"{instrument} {endpoint.split('/')[-1]}",
                            "status": status,
                            "detail": detail,
                            "endpoint": f"{DERIBIT_BASE}/{endpoint}",
                        }
                    )
                chart = fetch_deribit_chart(client, instrument, start, end)
                if not chart.empty and chart["error"].eq("").any():
                    ok = chart[chart["error"].eq("")]
                    status = "pass_chart_ohlc"
                    detail = f"{len(ok)} hourly bars, total volume {ok['volume'].astype(float).sum():.2f}"
                    charts.append(chart)
                else:
                    status = "fail"
                    detail = str(chart["error"].iloc[0]) if not chart.empty and "error" in chart else "no chart rows"
                rows.append(
                    {
                        "source": "deribit_api",
                        "check": f"{instrument} tradingview_chart_data",
                        "status": status,
                        "detail": detail,
                        "endpoint": f"{DERIBIT_BASE}/public/get_tradingview_chart_data",
                    }
                )
                time.sleep(0.03)
    charts_df = pd.concat(charts, ignore_index=True) if charts else pd.DataFrame()
    return pd.DataFrame(rows), charts_df


def load_daily_surface() -> pd.DataFrame:
    if not DAILY_SURFACE.exists():
        raise SystemExit(f"missing {DAILY_SURFACE}")
    surface = pd.read_parquet(DAILY_SURFACE)
    surface["ts"] = pd.to_datetime(surface["ts"], utc=True)
    return surface[surface["asset"].isin(["BTC", "ETH"])].copy()


def load_pm_hourly(surface: pd.DataFrame) -> pd.DataFrame:
    if not PM_LOB.exists():
        raise SystemExit(f"missing {PM_LOB}")
    slugs = set(surface["slug"].dropna().astype(str).unique())
    cols = ["exchange_ts", "market_id", "slug", "asset_id", "outcome_index", "best_bid", "best_ask", "mid"]
    pm = pd.read_parquet(PM_LOB, columns=cols)
    pm["exchange_ts"] = pd.to_datetime(pm["exchange_ts"], utc=True)
    pm = pm[pm["slug"].astype(str).isin(slugs) & pm["outcome_index"].astype("Int64").eq(0)].copy()
    pm["asset"] = pm["slug"].map({v: k for k, v in PM_DAILY_SLUGS.items()})
    pm = pm[pm["asset"].isin(["BTC", "ETH"])].copy()
    pm["hour"] = pm["exchange_ts"].dt.floor("60min")
    hourly = (
        pm.groupby(["asset", "slug", "market_id", "hour"], as_index=False)
        .agg(
            pm_up_bid=("best_bid", "median"),
            pm_up_ask=("best_ask", "median"),
            pm_up_mid=("mid", "median"),
            pm_rows=("mid", "size"),
            pm_first_ts=("exchange_ts", "min"),
            pm_last_ts=("exchange_ts", "max"),
        )
        .sort_values(["asset", "hour"])
    )
    return hourly


def surface_hourly(surface: pd.DataFrame) -> pd.DataFrame:
    surface = surface.copy()
    surface["hour"] = surface["ts"].dt.floor("60min")
    return (
        surface.groupby(["asset", "hour"], as_index=False)
        .agg(
            binance_spot=("binance_spot", "median"),
            window_strike_spot=("window_strike_spot", "first"),
            window_close_spot=("window_close_spot", "first"),
            window_start=("window_start", "first"),
            window_end=("window_end", "first"),
            seconds_to_expiry=("seconds_to_expiry", "median"),
            ewma_fair_up=("fair_up_causal", "median"),
        )
        .sort_values(["asset", "hour"])
    )


def spread_digital(charts: pd.DataFrame, asset: str, low: int, high: int, spot: pd.DataFrame) -> pd.DataFrame:
    low_name = chart_instrument(asset, low, "C")
    high_name = chart_instrument(asset, high, "C")
    low_df = charts[charts["instrument"].eq(low_name)].copy()
    high_df = charts[charts["instrument"].eq(high_name)].copy()
    if low_df.empty or high_df.empty:
        return pd.DataFrame()
    cols = ["ts", "close", "volume", "cost"]
    merged = low_df[cols].rename(columns={"close": "call_low_close", "volume": "call_low_volume", "cost": "call_low_cost"}).merge(
        high_df[cols].rename(columns={"close": "call_high_close", "volume": "call_high_volume", "cost": "call_high_cost"}),
        on="ts",
        how="inner",
    )
    merged = merged.merge(spot[spot["asset"].eq(asset)], left_on="ts", right_on="hour", how="left")
    merged["asset"] = asset
    merged["strike_low"] = float(low)
    merged["strike_high"] = float(high)
    merged["call_spread_width_usd"] = float(high - low)
    merged["call_low_usd"] = merged["call_low_close"].astype(float) * merged["binance_spot"].astype(float)
    merged["call_high_usd"] = merged["call_high_close"].astype(float) * merged["binance_spot"].astype(float)
    raw = (merged["call_low_usd"] - merged["call_high_usd"]) / (high - low)
    merged["deribit_up_digital_raw"] = raw
    merged["deribit_up_digital"] = raw.clip(0.0, 1.0)
    merged["deribit_clipped"] = raw.ne(merged["deribit_up_digital"])
    merged["deribit_total_volume"] = merged["call_low_volume"].astype(float) + merged["call_high_volume"].astype(float)
    merged["deribit_total_cost"] = merged["call_low_cost"].astype(float) + merged["call_high_cost"].astype(float)
    merged["floor_stale_bar"] = (
        merged["call_low_close"].astype(float).le(0.0001)
        & merged["call_high_close"].astype(float).le(0.0001)
        & merged["deribit_total_volume"].astype(float).eq(0.0)
    )
    return merged


def build_basis(surface: pd.DataFrame, charts: pd.DataFrame, pm_hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spot = surface_hourly(surface)
    basis_rows: list[pd.DataFrame] = []
    rep_rows: list[pd.DataFrame] = []
    for asset, spec in SPREADS.items():
        tight = spread_digital(charts, asset, *spec["tight"], spot)
        wide = spread_digital(charts, asset, *spec["wide"], spot)
        if tight.empty:
            continue
        tight = tight.rename(columns={"deribit_up_digital": "deribit_up_digital_tight", "deribit_up_digital_raw": "deribit_up_digital_raw_tight"})
        tight["hour"] = tight["ts"]
        basis = tight.merge(
            pm_hourly[pm_hourly["asset"].eq(asset)],
            on=["asset", "hour"],
            how="inner",
        )
        if not wide.empty:
            wide = wide[["ts", "asset", "deribit_up_digital", "deribit_up_digital_raw"]].rename(
                columns={"deribit_up_digital": "deribit_up_digital_wide", "deribit_up_digital_raw": "deribit_up_digital_raw_wide"}
            )
            basis = basis.merge(wide, on=["asset", "ts"], how="left")
        else:
            basis["deribit_up_digital_wide"] = np.nan
            basis["deribit_up_digital_raw_wide"] = np.nan
        basis["pm_minus_deribit_basis"] = basis["pm_up_mid"] - basis["deribit_up_digital_tight"]
        basis["basis_abs"] = basis["pm_minus_deribit_basis"].abs()
        basis["replication_error_proxy"] = (basis["deribit_up_digital_tight"] - basis["deribit_up_digital_wide"]).abs()
        basis["pm_taker_fee_at_mid"] = PM_TAKER_FEE_RATE_CRYPTO * basis["pm_up_mid"] * (1.0 - basis["pm_up_mid"])
        basis["deribit_fee_proxy"] = 2.0 * DERIBIT_FEE_PER_CONTRACT
        basis["known_cost_floor"] = basis["pm_taker_fee_at_mid"] + basis["deribit_fee_proxy"] + basis["replication_error_proxy"].fillna(0.0)
        # Use only bars before the post-expiry/floor-stale region. The chart
        # endpoint keeps returning 0.0001 bars after the option is effectively
        # dead; those are not usable market-vs-market observations.
        basis["usable_basis_row"] = ~basis["floor_stale_bar"].fillna(True)
        basis_rows.append(basis)
        if not wide.empty:
            rep_rows.append(
                basis[
                    [
                        "asset",
                        "ts",
                        "call_spread_width_usd",
                        "deribit_up_digital_tight",
                        "deribit_up_digital_wide",
                        "replication_error_proxy",
                    ]
                ]
            )
    out = pd.concat(basis_rows, ignore_index=True) if basis_rows else pd.DataFrame()
    rep = pd.concat(rep_rows, ignore_index=True) if rep_rows else pd.DataFrame()
    return out.sort_values(["asset", "ts"]).reset_index(drop=True), rep


def settlement_mismatch(surface: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for asset, g in surface.groupby("asset", sort=False):
        g = g.sort_values("ts")
        strike = float(g["window_strike_spot"].iloc[0])
        pm_close = float(g["window_close_spot"].iloc[0])
        pm_end = pd.to_datetime(g["window_end"].iloc[0], utc=True)
        deribit_proxy_ts = pd.Timestamp("2026-05-28 08:00:00", tz="UTC")
        spot_at_deribit = float(g.iloc[(g["ts"] - deribit_proxy_ts).abs().argmin()]["binance_spot"])
        pm_res_up = pm_close > strike
        deribit_res_up_proxy = spot_at_deribit > strike
        rows.append(
            {
                "asset": asset,
                "pm_strike": strike,
                "deribit_proxy_expiry_ts": deribit_proxy_ts,
                "pm_resolution_ts": pm_end,
                "spot_at_deribit_proxy_expiry": spot_at_deribit,
                "spot_at_pm_resolution": pm_close,
                "pm_resolves_up": bool(pm_res_up),
                "deribit_proxy_resolves_up": bool(deribit_res_up_proxy),
                "direction_mismatch": bool(pm_res_up != deribit_res_up_proxy),
                "spot_move_deribit_to_pm_bps": math.log(pm_close / spot_at_deribit) * 10_000.0,
                "binary_slippage_if_mismatch_c": 1.0 if pm_res_up != deribit_res_up_proxy else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_basis(basis: pd.DataFrame, settlement: pd.DataFrame, pm_hourly: pd.DataFrame, charts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if basis.empty:
        return pd.DataFrame()
    for asset, g0 in basis.groupby("asset", sort=False):
        g = g0[g0["usable_basis_row"]].copy()
        if g.empty:
            continue
        width = float(g["call_spread_width_usd"].iloc[0])
        strike = float(g["window_strike_spot"].iloc[0])
        width_bp = width / strike * 10_000.0
        killer_floor = g["known_cost_floor"]
        rows.append(
            {
                "asset": asset,
                "basis_rows": int(len(g)),
                "pm_capture_first": g["pm_first_ts"].min(),
                "pm_capture_last": g["pm_last_ts"].max(),
                "deribit_first": g["ts"].min(),
                "deribit_last_usable": g["ts"].max(),
                "pm_strike": strike,
                "spread_low": float(g["strike_low"].iloc[0]),
                "spread_high": float(g["strike_high"].iloc[0]),
                "call_spread_width_usd": width,
                "call_spread_width_bp_of_strike": width_bp,
                "mean_pm_up_mid": float(g["pm_up_mid"].mean()),
                "mean_deribit_up_digital": float(g["deribit_up_digital_tight"].mean()),
                "mean_basis_pm_minus_deribit": float(g["pm_minus_deribit_basis"].mean()),
                "median_basis_pm_minus_deribit": float(g["pm_minus_deribit_basis"].median()),
                "p95_abs_basis": float(g["pm_minus_deribit_basis"].abs().quantile(0.95)),
                "positive_basis_share": float(g["pm_minus_deribit_basis"].gt(0).mean()),
                "mean_replication_error_proxy": float(g["replication_error_proxy"].mean()),
                "p95_replication_error_proxy": float(g["replication_error_proxy"].quantile(0.95)),
                "mean_pm_taker_fee": float(g["pm_taker_fee_at_mid"].mean()),
                "mean_known_cost_floor": float(killer_floor.mean()),
                "basis_minus_known_cost_floor_abs_mean": float(g["basis_abs"].mean() - killer_floor.mean()),
                "usable_deribit_chart_rows": int(len(g)),
                "deribit_zero_volume_share": float(g["deribit_total_volume"].eq(0).mean()),
                "deribit_clipped_share": float(g["deribit_clipped"].mean()),
                "settlement_direction_mismatch": bool(settlement[settlement["asset"].eq(asset)]["direction_mismatch"].iloc[0]),
                "spot_move_deribit_to_pm_bps": float(settlement[settlement["asset"].eq(asset)]["spot_move_deribit_to_pm_bps"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def make_plot(basis: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    used = basis[basis["usable_basis_row"]].copy()
    if not used.empty:
        for asset, g in used.groupby("asset"):
            ax.plot(g["ts"], g["pm_minus_deribit_basis"], marker="o", linewidth=1.8, label=f"{asset} PM - Deribit")
        ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_title("PM daily UP mid minus Deribit call-spread digital")
    ax.set_ylabel("Basis, probability points")
    ax.set_xlabel("UTC hour")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(BASIS_PLOT, dpi=160)
    plt.close(fig)


def format_gate_table(gate: pd.DataFrame) -> str:
    rows = []
    for _, r in gate.iterrows():
        check = r.get("check", "")
        if pd.isna(check) or str(check) == "":
            check = r.get("path", "")
        status = r.get("status", "")
        if pd.isna(status) or str(status) == "":
            status = r.get("kind", "")
        detail = r.get("detail", "")
        if pd.isna(detail) or str(detail) == "":
            detail = r.get("notes", "")
        rows.append([str(r["source"]), str(check), str(status), str(detail)[:180]])
    return markdown_table(["source", "check / artifact", "status", "detail"], rows)


def format_summary_table(summary: pd.DataFrame) -> str:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            [
                str(r["asset"]),
                str(int(r["basis_rows"])),
                f"{pd.to_datetime(r['pm_capture_first']).strftime('%m-%d %H:%M')} -> {pd.to_datetime(r['pm_capture_last']).strftime('%m-%d %H:%M')}",
                number(float(r["call_spread_width_bp_of_strike"]), 1),
                cents(float(r["mean_pm_up_mid"])),
                cents(float(r["mean_deribit_up_digital"])),
                cents(float(r["mean_basis_pm_minus_deribit"])),
                cents(float(r["median_basis_pm_minus_deribit"])),
                cents(float(r["p95_abs_basis"])),
                pct(float(r["positive_basis_share"])),
                cents(float(r["mean_replication_error_proxy"])),
                cents(float(r["mean_known_cost_floor"])),
                cents(float(r["basis_minus_known_cost_floor_abs_mean"])),
            ]
        )
    return markdown_table(
        [
            "asset",
            "rows",
            "PM coverage UTC",
            "spread width bp",
            "PM UP mid",
            "Deribit digital",
            "mean basis",
            "median basis",
            "p95 abs basis",
            "basis > 0",
            "rep error",
            "known cost floor",
            "abs basis minus cost",
        ],
        rows,
    )


def format_settlement_table(settlement: pd.DataFrame) -> str:
    rows = []
    for _, r in settlement.iterrows():
        rows.append(
            [
                str(r["asset"]),
                number(float(r["pm_strike"]), 2),
                number(float(r["spot_at_deribit_proxy_expiry"]), 2),
                number(float(r["spot_at_pm_resolution"]), 2),
                str(bool(r["deribit_proxy_resolves_up"])),
                str(bool(r["pm_resolves_up"])),
                str(bool(r["direction_mismatch"])),
                number(float(r["spot_move_deribit_to_pm_bps"]), 1),
                cents(float(r["binary_slippage_if_mismatch_c"])),
            ]
        )
    return markdown_table(
        ["asset", "strike", "spot at Deribit proxy", "spot at PM close", "Deribit UP?", "PM UP?", "mismatch?", "8h drift bp", "binary slip if mismatch"],
        rows,
    )


def write_note(gate: pd.DataFrame, summary: pd.DataFrame, settlement: pd.DataFrame) -> None:
    partial_pass = "pass_chart_ohlc" in set(gate["status"].astype(str))
    clean_pass = False
    best_abs_minus_cost = float(summary["basis_minus_known_cost_floor_abs_mean"].max()) if not summary.empty else math.nan
    headline = "PARTIAL PASS: degraded Phase 1 ran, but no deployable verdict"
    if not partial_pass:
        headline = "FAIL: only DVOL/live-style data available; start dual capture"
    if clean_pass:
        headline = "PASS: historical mark/IV chain available"

    note = f"""# OD-RV Scoping: PM Daily Crypto Digitals Versus Deribit 1-Day Options

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Prior notes: [[od_pricing_model_form_findings]] · [[od_v4_calibration_gate_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Headline

Phase 0 data verdict: **{headline}**.

This is a market-vs-market scoping run, not another attempt to out-price Polymarket with our own Gaussian or jump model. The trade idea is: compare the Polymarket daily crypto UP token to a Deribit option-market-implied digital on the same asset and date, then only care about the basis after two-venue costs and settlement mismatch.

The result is useful but not green-lighting. Free Deribit public data does expose historical option OHLC through `public/get_tradingview_chart_data` for the relevant expired BTC/ETH strikes, so a degraded single-window basis pass can be constructed. But the endpoints that would make this execution-grade - historical mark/IV and expired-instrument book summary - are not available for those expired instruments through the checked public endpoints. The PM daily LOB capture also covers only about 12 hours of the 24-hour market, not the full path into PM resolution.

Plain-English verdict: **start forward dual-capture before believing this RV trade.** The observed one-window basis is sometimes larger than the known cost floor, but the Deribit leg here is OHLC-close based, not bid/ask or mark-IV based, and the PM/Deribit settlement-time mismatch is still the main risk.

CI note: this scoping run has only one independent PM daily resolution for BTC and one for ETH. Hourly rows inside that day describe basis persistence, but they are not independent PnL samples, so this note deliberately avoids a fake confidence interval and specifies the forward sample needed for CI.

## Phase 0 Data Availability

The gate asked whether we can reconstruct the Deribit 1-day option chain for the captured PM daily window. The answer is a partial yes:

- Local artifacts contain the PM daily surface and old DVOL anchor, but no reusable Deribit per-instrument mark/IV history.
- Deribit's `get_mark_price_history` and `get_book_summary_by_instrument` checks fail for the expired target instruments.
- Deribit's `get_tradingview_chart_data` does return hourly OHLC for strikes bracketing the PM BTC/ETH strike, including volume/cost fields. This is enough for a *scoping* call-spread basis, not enough for executable bid/ask PnL.

{format_gate_table(gate)}

Read: this is why the run proceeds to a degraded Phase 1 instead of stopping entirely. It is not a clean Phase 0 pass because the Deribit data is chart OHLC, not a historical option book/mark surface.

## Phase 1 Degraded Single-Window Basis

The captured PM daily markets are:

- BTC: `bitcoin-up-or-down-on-may-28-2026`, strike from Binance at PM window open = about `$75,326`, PM window `2026-05-27 16:00` to `2026-05-28 16:00` UTC.
- ETH: `ethereum-up-or-down-on-may-28-2026`, strike about `$2,071`, same PM window.

Practical replication example: for BTC, the PM strike sits between Deribit `BTC-28MAY26-75000-C` and `BTC-28MAY26-75500-C`. A cash digital is approximated as:

```text
Deribit digital ≈ spot * (call_price_75000 - call_price_75500) / 500
basis = PM UP mid - Deribit digital
```

The multiplication by spot converts Deribit option premium from coin units into a USD call value before taking the call-spread slope. This is still an approximation: the call spread is `$500` wide for BTC and `$25` wide for ETH, so it measures the average digital over the bracket, not an infinitesimal binary at the exact PM strike.

{format_summary_table(summary)}

Column read: `p95 abs basis` is the 95th percentile absolute PM-minus-Deribit gap during the overlapped hourly observations. `rep error` is a proxy from comparing the tight call spread with a wider neighboring spread. `known cost floor` includes PM taker fee at the PM mid, a small Deribit fee proxy, and the replication-error proxy; it does **not** include Deribit bid/ask spread because historical expired books were unavailable.

![PM vs Deribit daily basis]({BASIS_PLOT})

Read: this is visually worth forward-capturing, but not tradable from this artifact. The basis is not a powered result: it is one PM daily resolution, hourly Deribit chart observations, partial PM LOB coverage, and no Deribit bid/ask.

## Killer 1: Call-Spread Replication Error

The replicated digital is fragile near the strike because a binary payoff is a slope, and the slope estimate depends on the call-spread width. The BTC tight spread is `$500` wide, about `66bp` of strike. The ETH tight spread is `$25` wide, about `121bp` of strike. That is not tiny for a 1-day digital.

In this pass, `rep error` compares the tight bracket with a wider neighboring bracket. It is a proxy, not a complete error model. A real capture should store multiple strikes around the PM strike so the digital can be estimated from a local slope and a curvature/error band, not just one call spread.

## Killer 2: Settlement-Timing / Reference Mismatch

PM daily resolves at the PM window close. The Deribit date option behaves like an exchange-expiry instrument and the chart becomes floor-stale after the morning expiry region. That creates an approximately eight-hour mismatch versus the PM 16:00 UTC resolution.

{format_settlement_table(settlement)}

Read: there was no direction mismatch in this one BTC/ETH sample, but the 8-hour spot drift is large enough that this cannot be assumed away. If the Deribit proxy and PM close land on opposite sides of the strike, the "hedged" basis can still lose a full binary unit.

## Feasibility Verdict

The best one-window `|basis|-known-cost-floor` read is {fmt_ci_value(best_abs_minus_cost)}. That is **not** an edge proof. It only says the market-vs-market RV idea is worth a forward data capture because the observed basis is not obviously dominated by the pieces we can measure. The unmeasured pieces - Deribit bid/ask, queue/leg execution, and settlement-reference mismatch - are exactly the pieces that decide whether this is real.

Do **not** trade this from the historical chart artifact. The right next step is a forward dual-capture that stores both venues' executable state at the same timestamps.

## Phase 2 Forward Dual-Capture Spec

Capture BTC and ETH only. SOL has no Deribit analogue.

Minimum cadence:

- Every 30 seconds during each PM daily crypto window; 5 seconds in the final hour and whenever PM UP is between 40c and 60c.
- PM: UP/DOWN top-of-book bid/ask/size, top-5 depth, trades, market slug, token IDs, outcome index, Chainlink/Pyth reference fields if obtainable, and final PM resolution timestamp/reference.
- Deribit: option book summaries or order books for the nearest expiry and at least five strikes around PM strike on each side; mark price, mark IV, bid/ask IV, underlying index, futures/perp index, and volume/open interest. Store raw instrument names because strike grids roll.
- Binance/Pyth/Chainlink: spot/index snapshots for settlement-reference diagnosis. This is required; otherwise the RV hedge may not actually cancel binary resolution risk.

Validation design:

- Treat one asset-day as one independent window. Hourly rows inside a day are basis observations, not independent PnL samples.
- Need at least about 60-100 independent asset-days before calling a 1-2c net basis positive with a useful 95% CI. BTC and ETH together produce at most two windows per calendar day, so this is roughly 30-50 calendar days of clean dual capture.
- Evaluate both directions: short PM / long Deribit-digital when PM is rich, and long PM / short Deribit-digital when PM is cheap. Include PM taker/maker route, Deribit bid/ask and fees, call-spread replication width, leg latency, and settlement mismatch PnL.
- Execution capital is two-venue: PM collateral plus Deribit option margin/premium. This gives up the PM-only simplicity, but it is the first version of OD that has a real liquid external fair-value instrument.

## Decision

OD-RV daily is **not closed** and **not validated**. It is a scoping partial-pass: the one captured PM daily window plus public Deribit chart OHLC can show a basis, but not an executable market-vs-market edge. Start forward dual-capture if we want to test it seriously; do not spend more time on historical DVOL/model-only proxies.

## Outputs

- Data gate CSV: `data/analysis/csv_outputs/options_delta/od_rv_deribit_daily_data_gate.csv`
- Basis CSV: `data/analysis/csv_outputs/options_delta/od_rv_deribit_daily_basis.csv`
- Summary CSV: `data/analysis/csv_outputs/options_delta/od_rv_deribit_daily_summary.csv`
- Settlement mismatch CSV: `data/analysis/csv_outputs/options_delta/od_rv_deribit_daily_settlement_mismatch.csv`
- Deribit chart parquet: `data/analysis/od_rv_deribit_daily_deribit_charts.parquet`
"""
    NOTE.write_text(normalize_markdown_wrapping(note), encoding="utf-8")


def replace_or_insert_bullet(path: Path, marker: str, bullet: str, after_header: str) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if marker in line:
            lines[i] = bullet
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    out: list[str] = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.strip() == after_header:
            out.append(bullet)
            inserted = True
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def update_docs(summary: pd.DataFrame) -> None:
    best = float(summary["basis_minus_known_cost_floor_abs_mean"].max()) if not summary.empty else math.nan
    bullet = (
        f"- 2026-06-02 OD-RV Deribit daily scoping: **PARTIAL PASS / CAPTURE NEEDED**. "
        f"One PM daily BTC/ETH window can be compared to Deribit expired option OHLC via call-spread digitals, "
        f"but free public checks did not expose historical expired-instrument mark/IV or bid/ask books. "
        f"Best one-window `|basis|-known-cost-floor` read is {fmt_ci_value(best)} before unmeasured Deribit spread and settlement-reference risk. "
        f"Do not trade from DVOL/model proxies; start forward PM+Deribit dual capture. See [[od_rv_deribit_daily_scoping_findings]]."
    )
    replace_or_insert_bullet(OD_HUB, "OD-RV Deribit daily scoping", bullet, "## Current state (2026-06-02)")
    replace_or_insert_bullet(BRAIN_TODO, "OD-RV Deribit daily scoping", bullet, "## OD — Options-Delta (state 2026-06-01) — REFRAMED as the valuation/signal layer")


def main() -> None:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    surface = load_daily_surface()
    start = surface["window_start"].min()
    end = surface["window_end"].max()
    local_gate = local_deribit_artifacts()
    api_gate, charts = probe_deribit(start, end)
    gate = pd.concat([local_gate, api_gate], ignore_index=True, sort=False)
    pm_hourly = load_pm_hourly(surface)
    basis, _rep = build_basis(surface, charts, pm_hourly)
    settlement = settlement_mismatch(surface)
    summary = summarize_basis(basis, settlement, pm_hourly, charts)

    gate.to_csv(OUT_GATE, index=False)
    basis.to_csv(OUT_BASIS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    settlement.to_csv(OUT_SETTLEMENT, index=False)
    if not charts.empty:
        charts.to_parquet(OUT_CHARTS, index=False)
    make_plot(basis)
    write_note(gate, summary, settlement)
    update_docs(summary)
    print(f"wrote {NOTE}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
