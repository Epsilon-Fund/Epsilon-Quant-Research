"""Phase 5 Stage 1 — stdout summary across all 72 runs."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BACKTESTS_DIR = ROOT / "data" / "backtests"


def flatten(s: dict) -> dict:
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
        "max_dd": s["max_drawdown_usd"],
        "sig_per_wk": s["signals_per_week"],
        "n_leaders": s.get("n_distinct_leaders", 0),
        "negrisk_share": s.get("negrisk_signal_share", 0),
    }


def main() -> None:
    summaries = [json.loads(p.read_text())
                 for p in sorted(BACKTESTS_DIR.glob("*_summary.json"))]
    df = pd.DataFrame([flatten(s) for s in summaries])

    print(f"\n=========== STAGE 1 — {len(df)} runs ===========\n")

    print("=== ALL 72 RUNS (one line each) ===")
    cols = ["cohort", "bucket", "sizing", "window", "sharpe", "total_pnl",
            "deploy_ratio", "n_signals"]
    print(df[cols].sort_values(["cohort", "bucket", "sizing", "window"])
          .to_string(index=False, formatters={
              "sharpe": lambda v: f"{v:7.2f}",
              "total_pnl": lambda v: f"{v:>14,.0f}",
              "deploy_ratio": lambda v: f"{v:.2%}",
              "n_signals": lambda v: f"{v:>8,}",
          }))

    print("\n\n=== TOP 5 BY SHARPE ===")
    print(df.sort_values("sharpe", ascending=False).head(5)[cols]
          .to_string(index=False))

    print("\n\n=== TOP 5 BY TOTAL_PNL ===")
    print(df.sort_values("total_pnl", ascending=False).head(5)[cols]
          .to_string(index=False))

    df["joint"] = df["sharpe"] * df["deploy_ratio"]
    print("\n\n=== TOP 5 BY SHARPE × DEPLOY_RATIO ===")
    print(df.sort_values("joint", ascending=False).head(5)[cols + ["joint"]]
          .to_string(index=False))

    print("\n\n=== CROSS-WINDOW ROBUSTNESS (cohort × bucket × sizing) ===")
    rob = df.groupby(["cohort", "bucket", "sizing"]).agg(
        sharpe_mean=("sharpe", "mean"),
        sharpe_min=("sharpe", "min"),
        sharpe_max=("sharpe", "max"),
        pnl_total=("total_pnl", "sum"),
        n_pos_windows=("sharpe", lambda x: (x > 0).sum()),
        n_above_1=("sharpe", lambda x: (x > 1.0).sum()),
        avg_deploy=("deploy_ratio", "mean"),
        avg_n_signals=("n_signals", "mean"),
    ).reset_index()
    rob["joint_avg"] = rob["sharpe_mean"] * rob["avg_deploy"]
    print(rob.sort_values(["n_above_1", "sharpe_mean"], ascending=[False, False])
          .to_string(index=False,
                     formatters={"sharpe_mean": "{:.2f}".format,
                                 "sharpe_min":  "{:.2f}".format,
                                 "sharpe_max":  "{:.2f}".format,
                                 "pnl_total":   "{:,.0f}".format,
                                 "avg_deploy":  "{:.2%}".format,
                                 "avg_n_signals": "{:,.0f}".format,
                                 "joint_avg":   "{:.3f}".format}))

    print("\n\n=== POSITIVE-SHARPE RUNS ONLY ===")
    pos = df[df["sharpe"] > 0].sort_values("sharpe", ascending=False)
    if len(pos):
        print(pos[cols].to_string(index=False))
    else:
        print("  (none — every run negative-Sharpe at default slippage)")

    print(f"\n  positive-Sharpe runs: {len(pos)} / {len(df)}")
    print(f"  positive-PnL runs:    {(df['total_pnl'] > 0).sum()} / {len(df)}")
    print(f"  runs with Sharpe > 1: {(df['sharpe'] > 1).sum()} / {len(df)}")


if __name__ == "__main__":
    main()
