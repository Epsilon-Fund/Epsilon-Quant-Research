"""
Verification script for portfolio chart functions.
Run from live_trading/:

    cd live_trading && python3 verify_charts.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plotly.io as pio

from shared.data_loader import (
    build_equity_curve,
    build_coin_equity_curves,
    build_capital_deployment,
)
from shared.charts import (
    equity_chart,
    drawdown_chart,
    capital_deployment_chart,
    coin_equity_chart,
)

DATA_DIR = "dashboards/momentum"

curve       = build_equity_curve(DATA_DIR)
coin_curves = build_coin_equity_curves(DATA_DIR)
deployment  = build_capital_deployment(DATA_DIR)

fig1 = equity_chart(curve, show_theoretical=True)
fig2 = drawdown_chart(curve)
fig3 = capital_deployment_chart(deployment)
fig4 = coin_equity_chart(coin_curves, normalised=False)
fig5 = coin_equity_chart(coin_curves, normalised=True)

pio.write_html(fig1, "test_equity.html")
pio.write_html(fig2, "test_drawdown.html")
pio.write_html(fig3, "test_deployment.html")
pio.write_html(fig4, "test_coins.html")
pio.write_html(fig5, "test_coins_norm.html")

print("Charts written to test_*.html — open to verify visually")
print()
print("Sanity checks:")
print(f"  equity_chart traces   : {len(fig1.data)}  (expected 2 — actual + theoretical)")
print(f"  drawdown traces       : {len(fig2.data)}  (expected 1)")
print(f"  deployment traces     : {len(fig3.data)}  (expected 1)")
print(f"  coin_equity traces    : {len(fig4.data)}  (expected {len(coin_curves)} — one per coin)")
print(f"  coin_equity_norm traces: {len(fig5.data)} (expected {len(coin_curves)})")
print(f"  coins in fund chart   : {list(coin_curves.keys())}")
