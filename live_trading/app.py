"""
Epsilon Fund — landing page.

Displays a live fund summary across all active strategy dashboards:
  - Total capital deployed
  - Open positions count
  - Closed trades count
  - Total realised P&L
  - Last signal date

Run:
    streamlit run live_trading/app.py
"""
import os
import sys
from datetime import datetime

import streamlit as st

st.set_page_config(
    page_title="Epsilon Fund",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Path setup ────────────────────────────────────────────────────────────────
_LT_DIR = os.path.dirname(os.path.abspath(__file__))
if _LT_DIR not in sys.path:
    sys.path.insert(0, _LT_DIR)

from shared.styles import apply_styles
from shared.data_loader import load_trades, build_trade_pairs, load_config

apply_styles()

# ── One-time cache warm-up on server startup ──────────────────────────────────
# @st.cache_resource runs once per `streamlit run` process and is shared across
# all browser sessions.  On a warm cache this completes in < 1 second.
# On a cold/stale cache it fetches only the missing bars (never a full re-pull).
@st.cache_resource(show_spinner="Updating market data cache…")
def _warm_cache():
    try:
        from shared.cache_manager import update_all_caches
        from dashboards.momentum.config import ACTIVE_ASSETS
        update_all_caches(ACTIVE_ASSETS)
    except Exception as e:
        st.warning(f"Cache update skipped: {e}")
    return True

_warm_cache()

# ── Registered strategy dashboards ───────────────────────────────────────────
DASHBOARD_DIRS = {
    "Momentum":    os.path.join(_LT_DIR, "dashboards", "momentum"),
    "Stat Arb":    os.path.join(_LT_DIR, "dashboards", "statarb"),
    "BB Breakout": os.path.join(_LT_DIR, "dashboards", "bbbreakout"),
}


@st.cache_data(ttl=120, show_spinner=False)
def _load_summary(dirs_tuple: tuple) -> dict:
    """Aggregate key metrics across all dashboard directories."""
    total_capital   = 0.0
    total_open      = 0
    total_closed    = 0
    total_pnl       = 0.0
    last_signal     = None
    any_trades      = False

    for name, data_dir in dirs_tuple:
        trades_json = os.path.join(data_dir, "trades.json")
        if not os.path.exists(trades_json):
            continue

        try:
            trades = load_trades(data_dir)
            pairs  = build_trade_pairs(data_dir)
            cfg    = load_config(data_dir)
        except Exception:
            continue

        if not trades:
            continue

        any_trades = True

        # Capital: from config, only count if strategy has trade history
        total_capital += cfg.get("capital", 0.0)

        # Open / closed counts
        total_open   += len(pairs.get("open", []))
        total_closed += len(pairs.get("closed", []))

        # Realised P&L from closed pairs
        total_pnl += sum(p["actual_pnl_usd"] for p in pairs.get("closed", []))

        # Last signal date: most recent trade date
        dates = [t["date"] for t in trades if t.get("date")]
        if dates:
            latest = max(dates)
            if last_signal is None or latest > last_signal:
                last_signal = latest

    return {
        "any_trades":    any_trades,
        "total_capital": total_capital,
        "total_open":    total_open,
        "total_closed":  total_closed,
        "total_pnl":     total_pnl,
        "last_signal":   last_signal,
    }


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:35px;font-weight:700;letter-spacing:-0.01em;margin-bottom:10px">
  Epsilon Fund
</h1>
""", unsafe_allow_html=True)

# ── Load summary ──────────────────────────────────────────────────────────────
summary = _load_summary(tuple(sorted(DASHBOARD_DIRS.items())))

if not summary["any_trades"]:
    st.info(
        "No trade history yet — run optimise.py and start trading."
    )
else:
    # ── Four metric columns ───────────────────────────────────────────────────
    pnl = summary["total_pnl"]
    pnl_display = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total capital",     f"${summary['total_capital']:,.0f}")
    m2.metric("Open positions",    summary["total_open"])
    m3.metric("Closed trades",     summary["total_closed"])
    m4.metric("Realised P&L",      pnl_display)

    # ── Last signal date ──────────────────────────────────────────────────────
    last_sig = summary["last_signal"]
    sig_str  = last_sig.isoformat() if last_sig else "—"
    st.markdown(
        f'<div class="dash-meta" style="margin-top:18px">'
        f'<strong>Last signal date:</strong> {sig_str}'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<p style="color:#888780;font-size:12px;margin-top:32px">'
    'Select a page from the sidebar to view dashboards, trade logs, or portfolio analytics.'
    '</p>',
    unsafe_allow_html=True,
)
