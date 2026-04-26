"""
Trade Log page — one tab per strategy, all rendering via shared components.

Tab layout
----------
Fund        → render_fund_tab()   aggregated cross-strategy flat log
Momentum    → render_strategy_tab() momentum-specific coin/performance view
Stat Arb    → stub
BB Breakout → stub

There is exactly ONE path to momentum trade data on this page: the Momentum
tab. The Fund tab shows the aggregated cross-strategy view only.
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
from shared.trade_log_components import render_fund_tab, render_strategy_tab

apply_styles()

# ── Registered dashboard directories (absolute paths) ────────────────────────
DASHBOARD_DIRS = {
    "Momentum":   os.path.join(_LT_DIR, "dashboards", "momentum"),
    "Stat Arb":   os.path.join(_LT_DIR, "dashboards", "statarb"),
    "BB Breakout": os.path.join(_LT_DIR, "dashboards", "bbbreakout"),
}

# ── Sidebar: single refresh button for the whole page ────────────────────────
with st.sidebar:
    st.markdown("### Controls")
    if st.button("↻ Refresh data", key="tl_refresh"):
        st.cache_data.clear()
        st.rerun()
    st.caption("MAE is cached in mae_cache.json after first fetch.")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:32px;font-weight:700;letter-spacing:-0.01em;margin-bottom:6px">
  Trade Journal
</h1>
<div class="dash-meta">
  Epsilon Fund &nbsp;·&nbsp; all strategies &nbsp;·&nbsp;
  use filters below to scope by date and coin
</div>
""", unsafe_allow_html=True)

# ── Strategy tabs ─────────────────────────────────────────────────────────────
tab_fund, tab_momentum, tab_statarb, tab_bb = st.tabs(
    ["Fund", "Momentum", "Stat Arb", "BB Breakout"]
)

with tab_fund:
    render_fund_tab(DASHBOARD_DIRS, prefix="fund")

with tab_momentum:
    render_strategy_tab(
        data_dir=DASHBOARD_DIRS["Momentum"],
        prefix="momentum",
        strategy_keys=["momentum_swing", "momentum_no_vol"],
        display_name="Momentum",
        show_strategy_col=True,
    )

with tab_statarb:
    render_strategy_tab(
        data_dir=DASHBOARD_DIRS["Stat Arb"],
        prefix="statarb",
        strategy_keys=["stat_arb_spread"],
        display_name="Stat Arb",
        show_strategy_col=True,
    )

with tab_bb:
    render_strategy_tab(
        data_dir=DASHBOARD_DIRS["BB Breakout"],
        prefix="bb",
        strategy_keys=["bb_breakout"],
        display_name="BB Breakout",
        show_strategy_col=True,
    )
