"""Phase 5 Stage 1.5 — analysis & diff vs Stage 1."""
import json
from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE15_DIR = ROOT / "data" / "backtests" / "stage15"
STAGE1_DIR = ROOT / "data" / "backtests"

WINDOWS_PRIMARY = {"2025", "2026"}


def flatten_15(s: dict) -> dict:
    return {
        "cohort": s["cohort"],
        "bucket": s["resolution_bucket"],
        "sizing": s["sizing_rule"],
        "window": s["test_window_start"][:4],
        "n_signals": s["n_signals"],
        "total_pnl": s["total_pnl_usd"],
        "sharpe": s["sharpe_monthly"],
        "win_rate": s.get("win_rate") or 0.0,
        "profit_factor": s.get("profit_factor") or 0.0,
        "deploy_ratio": s["deployment_ratio"],
        "sig_per_wk": s["signals_per_week"],
        "fallback_pct": s.get("slippage_fallback_pct", 0.0),
        "avg_slip_c": s.get("slippage_avg_cents", 0.0),
        "avg_size": s.get("avg_signal_size_usd", 0.0),
        "n_leaders": s.get("n_distinct_leaders", 0),
        "max_dd": s.get("max_drawdown_usd", 0.0),
    }


def flatten_1(s: dict) -> dict:
    return {
        "cohort": s["cohort"],
        "bucket": s["resolution_bucket"],
        "sizing": s["sizing_rule"],
        "window": s["test_window_start"][:4],
        "n_signals_s1": s["n_signals"],
        "total_pnl_s1": s["total_pnl_usd"],
        "sharpe_s1": s["sharpe_monthly"],
        "deploy_ratio_s1": s["deployment_ratio"],
    }


def main() -> None:
    s15 = [json.loads(p.read_text())
           for p in sorted(STAGE15_DIR.glob("*_summary.json"))]
    s1 = [json.loads(p.read_text())
          for p in sorted(STAGE1_DIR.glob("*_summary.json"))
          if "stage15" not in p.name]

    df15 = pd.DataFrame([flatten_15(s) for s in s15])
    df1 = pd.DataFrame([flatten_1(s) for s in s1])

    print(f"\n=========== STAGE 1.5 — {len(df15)} runs ===========\n")

    # ---------- PRIMARY ----------
    primary = df15[df15["window"].isin(WINDOWS_PRIMARY)].copy()
    print("=== PRIMARY (2025, 2026) — master comparison ===\n")
    primary = primary.sort_values(["cohort", "bucket", "sizing", "window"])
    print(primary[["cohort", "bucket", "sizing", "window",
                   "n_signals", "total_pnl", "sharpe", "deploy_ratio",
                   "fallback_pct", "avg_slip_c", "avg_size"]].to_string(
        index=False,
        formatters={"sharpe": "{:7.2f}".format,
                    "total_pnl": "{:>12,.0f}".format,
                    "deploy_ratio": "{:.0%}".format,
                    "fallback_pct": "{:.0%}".format,
                    "avg_slip_c": "{:.2f}".format,
                    "avg_size": "{:>7,.0f}".format,
                    "n_signals": "{:>7,}".format}))

    # ---------- SIDEBAR ----------
    print("\n\n=== SIDEBAR (2024 — low confidence, data-sparse era) ===\n")
    sb = df15[df15["window"] == "2024"].copy().sort_values(["cohort", "bucket", "sizing"])
    print(sb[["cohort", "bucket", "sizing", "window",
              "n_signals", "total_pnl", "sharpe", "deploy_ratio",
              "fallback_pct", "avg_slip_c"]].to_string(
        index=False,
        formatters={"sharpe": "{:7.2f}".format,
                    "total_pnl": "{:>10,.0f}".format,
                    "deploy_ratio": "{:.0%}".format,
                    "fallback_pct": "{:.0%}".format,
                    "avg_slip_c": "{:.2f}".format,
                    "n_signals": "{:>7,}".format}))

    # ---------- DIFF vs STAGE 1 ----------
    print("\n\n=== DIFF vs STAGE 1 (matched cohort/bucket/sizing/window) ===\n")
    merged = df15.merge(df1, on=["cohort", "bucket", "sizing", "window"], how="left")
    merged["n_sig_reduction_x"] = merged["n_signals_s1"] / merged["n_signals"].clip(1, None)
    merged["sharpe_shift"] = merged["sharpe"] - merged["sharpe_s1"]
    merged["pnl_shift_abs"] = merged["total_pnl"] - merged["total_pnl_s1"]
    primary_diff = merged[merged["window"].isin(WINDOWS_PRIMARY)].sort_values(
        ["cohort", "bucket", "sizing", "window"])
    print(primary_diff[["cohort", "bucket", "sizing", "window",
                        "n_signals_s1", "n_signals", "n_sig_reduction_x",
                        "sharpe_s1", "sharpe", "sharpe_shift",
                        "total_pnl_s1", "total_pnl", "deploy_ratio_s1",
                        "deploy_ratio"]].to_string(
        index=False,
        formatters={"sharpe_s1": "{:6.2f}".format,
                    "sharpe": "{:6.2f}".format,
                    "sharpe_shift": "{:+6.2f}".format,
                    "n_sig_reduction_x": "{:5.1f}x".format,
                    "n_signals_s1": "{:>7,}".format,
                    "n_signals": "{:>5,}".format,
                    "total_pnl_s1": "{:>12,.0f}".format,
                    "total_pnl": "{:>10,.0f}".format,
                    "deploy_ratio_s1": "{:.0%}".format,
                    "deploy_ratio": "{:.0%}".format}))

    # ---------- TOP 5 BY SHARPE (primary only) ----------
    print("\n\n=== TOP 5 BY SHARPE — PRIMARY windows ===\n")
    print(primary.sort_values("sharpe", ascending=False).head(5)[
        ["cohort", "bucket", "sizing", "window", "sharpe", "total_pnl",
         "deploy_ratio", "n_signals", "fallback_pct", "avg_slip_c"]].to_string(
        index=False,
        formatters={"sharpe": "{:7.2f}".format,
                    "total_pnl": "{:>10,.0f}".format,
                    "deploy_ratio": "{:.0%}".format,
                    "fallback_pct": "{:.0%}".format,
                    "avg_slip_c": "{:.2f}".format,
                    "n_signals": "{:>5,}".format}))

    print("\n\n=== TOP 5 BY TOTAL_PNL — PRIMARY windows ===\n")
    print(primary.sort_values("total_pnl", ascending=False).head(5)[
        ["cohort", "bucket", "sizing", "window", "total_pnl", "sharpe",
         "deploy_ratio", "n_signals", "fallback_pct"]].to_string(
        index=False,
        formatters={"sharpe": "{:7.2f}".format,
                    "total_pnl": "{:>10,.0f}".format,
                    "deploy_ratio": "{:.0%}".format,
                    "fallback_pct": "{:.0%}".format,
                    "n_signals": "{:>5,}".format}))

    primary["joint"] = primary["sharpe"] * primary["deploy_ratio"]
    print("\n\n=== TOP 5 BY SHARPE × DEPLOY — PRIMARY windows ===\n")
    print(primary.sort_values("joint", ascending=False).head(5)[
        ["cohort", "bucket", "sizing", "window", "sharpe", "deploy_ratio",
         "joint", "total_pnl", "n_signals"]].to_string(
        index=False,
        formatters={"sharpe": "{:6.2f}".format,
                    "deploy_ratio": "{:.0%}".format,
                    "joint": "{:6.3f}".format,
                    "total_pnl": "{:>10,.0f}".format,
                    "n_signals": "{:>5,}".format}))

    # ---------- CROSS-WINDOW ROBUSTNESS (primary windows only) ----------
    rob = primary.groupby(["cohort", "bucket", "sizing"]).agg(
        sharpe_mean=("sharpe", "mean"),
        sharpe_min=("sharpe", "min"),
        sharpe_max=("sharpe", "max"),
        pnl_total=("total_pnl", "sum"),
        n_pos_windows=("sharpe", lambda x: (x > 0).sum()),
        avg_deploy=("deploy_ratio", "mean"),
        avg_n_signals=("n_signals", "mean"),
        avg_fallback=("fallback_pct", "mean"),
    ).reset_index()
    rob["joint"] = rob["sharpe_mean"] * rob["avg_deploy"]
    print("\n\n=== CROSS-WINDOW ROBUSTNESS — PRIMARY windows only ===\n")
    print(rob.sort_values(["n_pos_windows", "joint"], ascending=[False, False]).to_string(
        index=False,
        formatters={"sharpe_mean": "{:6.2f}".format,
                    "sharpe_min": "{:6.2f}".format,
                    "sharpe_max": "{:6.2f}".format,
                    "pnl_total": "{:>11,.0f}".format,
                    "avg_deploy": "{:.0%}".format,
                    "avg_n_signals": "{:>6,.0f}".format,
                    "avg_fallback": "{:.0%}".format,
                    "joint": "{:6.3f}".format}))

    # ---------- per-cohort summary (the SQL the user asked for) ----------
    print("\n\n=== PER-COHORT AGGREGATE (per user SQL) — PRIMARY windows only ===\n")
    audits = []
    for p in sorted(STAGE15_DIR.glob("*.parquet")):
        name = p.name.replace(".parquet", "")
        window = name.split("__")[3]
        if window in WINDOWS_PRIMARY:
            audits.append(str(p))
    con = duckdb.connect()
    if audits:
        paths_sql = ", ".join(f"'{p}'" for p in audits)
        print(con.sql(f"""
            SELECT cohort,
                   COUNT(*) AS n_signals,
                   ROUND(AVG(slippage_cents), 3) AS avg_slip_c,
                   ROUND(STDDEV(slippage_cents), 3) AS slip_std,
                   ROUND(SUM(CASE WHEN slippage_source='fallback' THEN 1 ELSE 0 END)::DOUBLE
                         / COUNT(*), 4) AS fallback_pct,
                   ROUND(AVG(copy_size_usd), 2) AS avg_size,
                   ROUND(SUM(copy_pnl_usd), 0) AS total_pnl
            FROM read_parquet([{paths_sql}])
            GROUP BY cohort
        """).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
