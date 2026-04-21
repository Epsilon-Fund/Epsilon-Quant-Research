"""
Verification script for build_equity_curve, build_coin_equity_curves,
and build_capital_deployment. Run from the live_trading/ directory:

    cd live_trading && python verify_equity.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.data_loader import (
    build_equity_curve,
    build_coin_equity_curves,
    build_capital_deployment,
    build_trade_pairs,
)

DATA_DIR = "dashboards/momentum"

# ── Equity curve ──────────────────────────────────────────────────────────────
curve = build_equity_curve(DATA_DIR)
print("Equity curve shape:", curve.shape)
print("Date range:", curve["date"].min(), "→", curve["date"].max())
print("Final actual cumulative P&L:", curve["actual_cumulative"].iloc[-1])
print("Non-zero days:", (curve["actual_pnl"] != 0).sum())
print()

# ── Coin equity curves ────────────────────────────────────────────────────────
coin_curves = build_coin_equity_curves(DATA_DIR)
for symbol, df in coin_curves.items():
    print(f"{symbol}: {len(df)} days, "
          f"final P&L: {df['actual_cumulative'].iloc[-1]:.2f}")
print()

# ── Capital deployment ────────────────────────────────────────────────────────
deployment = build_capital_deployment(DATA_DIR)
print("Deployment shape:", deployment.shape)
print("Max deployment %:", deployment["deployment_pct"].max())
print("Avg deployment %:", deployment["deployment_pct"].mean())
print()

# ── Cross-check: final cumulative == sum of pnl_usd across all closed pairs ──
pairs      = build_trade_pairs(DATA_DIR)
closed     = pairs.get("closed", [])
total_pnl  = sum(p["pnl_usd"] for p in closed)
curve_final = curve["actual_cumulative"].iloc[-1] if not curve.empty else 0.0
discrepancy = abs(total_pnl - curve_final)

print(f"Sum of pnl_usd from build_trade_pairs : {total_pnl:.4f}")
print(f"Final actual_cumulative from curve    : {curve_final:.4f}")
print(f"Discrepancy                           : {discrepancy:.6f}")
if discrepancy > 0.01:
    print("  WARNING: discrepancy exceeds $0.01 — investigate!")
else:
    print("  OK: totals match within rounding tolerance.")
