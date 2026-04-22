"""
Portfolio page — equity curves, drawdown, and deployment, one tab per strategy.
"""
import os
import sys

import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# ── Path setup ────────────────────────────────────────────────────────────────
_PAGES_DIR = os.path.dirname(os.path.abspath(__file__))   # live_trading/pages/
_LT_DIR    = os.path.dirname(_PAGES_DIR)                   # live_trading/
if _LT_DIR not in sys.path:
    sys.path.insert(0, _LT_DIR)

from shared.styles import apply_styles
from shared.portfolio_components import render_strategy_portfolio, render_fund_portfolio

apply_styles()

# ── Registered dashboard directories (absolute paths) ────────────────────────
DASHBOARD_DIRS = {
    "Momentum":    os.path.join(_LT_DIR, "dashboards", "momentum"),
    "Stat Arb":    os.path.join(_LT_DIR, "dashboards", "statarb"),
    "BB Breakout": os.path.join(_LT_DIR, "dashboards", "bbbreakout"),
}

# ── Sidebar: single refresh button for the whole page ────────────────────────
with st.sidebar:
    st.markdown("### Controls")
    if st.button("↻ Refresh data", key="port_refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Strategy tabs ─────────────────────────────────────────────────────────────
tab_fund, tab_momentum, tab_statarb, tab_bb = st.tabs(
    ["Fund", "Momentum", "Stat Arb", "BB Breakout"]
)

with tab_fund:
    render_fund_portfolio(DASHBOARD_DIRS, prefix="fund")

with tab_momentum:
    render_strategy_portfolio(
        data_dir=DASHBOARD_DIRS["Momentum"],
        prefix="momentum",
        strategy_name="Momentum Swing",
    )

with tab_statarb:
    st.info("No trades yet")

with tab_bb:
    st.info("No trades yet")
