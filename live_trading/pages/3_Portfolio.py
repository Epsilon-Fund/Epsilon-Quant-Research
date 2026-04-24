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

# Order matters — fund tab sees this dict in insertion order
FUND_DIRS = DASHBOARD_DIRS   # same mapping, explicit alias for clarity

# ── Sidebar: single refresh button for the whole page ────────────────────────
with st.sidebar:
    st.markdown("### Controls")
    if st.button("↻ Refresh data", key="port_refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:35px;font-weight:700;letter-spacing:-0.01em;margin-bottom:10px">
  Portfolio
</h1>
""", unsafe_allow_html=True)

# ── Strategy tabs ─────────────────────────────────────────────────────────────
tab_fund, tab_momentum, tab_statarb, tab_bb = st.tabs(
    ["Fund", "Momentum", "Stat Arb", "BB Breakout"]
)

with tab_fund:
    render_fund_portfolio(FUND_DIRS, prefix="fund")

with tab_momentum:
    render_strategy_portfolio(
        data_dir=DASHBOARD_DIRS["Momentum"],
        prefix="momentum",
        strategy_name="Momentum Swing",
    )

with tab_statarb:
    render_strategy_portfolio(
        data_dir=DASHBOARD_DIRS["Stat Arb"],
        prefix="statarb",
        strategy_name="Stat Arb",
    )

with tab_bb:
    st.info("No trades yet")
