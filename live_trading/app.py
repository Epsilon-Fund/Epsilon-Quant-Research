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
from shared.data_loader import load_trades, build_trade_pairs, load_realised_capital, load_positions
from shared.websocket_manager import get_shared_ws

apply_styles()

# ── One-time cache warm-up on server startup ──────────────────────────────────
# @st.cache_resource runs once per `streamlit run` process and is shared across
# all browser sessions.  On a warm cache this completes in < 1 second.
# On a cold/stale cache it fetches only the missing bars (never a full re-pull).
@st.cache_resource(show_spinner="Updating market data cache…")
def _warm_cache():
    try:
        from shared.cache_manager import update_all_caches
        from dashboards.momentum.config   import ACTIVE_ASSETS as _mom
        from dashboards.bbbreakout.config import ACTIVE_ASSETS as _bb
        from dashboards.statarb.optimise  import ASSET_CONFIG  as _sa_cfg
        _sa = [sym for a in _sa_cfg for sym in (a['symbol_y'], a['symbol_x'])]
        update_all_caches(sorted(set(_mom) | set(_sa) | set(_bb)))
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


# ── One-time in-process Streamlit-cache warm-up ──────────────────────────────
# `_warm_cache` above fills the on-disk parquet cache.  This second pass primes
# Streamlit's in-memory equity-curve / deployment caches used by pages 2 & 3.
# Live prices come from the WebSocket on the dashboard pages — no REST prewarm
# is needed.  Per-dashboard `@st.cache_data` on fetch_ohlcv etc. doesn't
# persist across `_exec_page` re-imports, so it isn't worth pre-warming here.
@st.cache_resource(show_spinner=False)
def _prewarm_session():
    try:
        from shared.data_loader import build_equity_curve, build_capital_deployment
        for _data_dir in DASHBOARD_DIRS.values():
            try:
                build_equity_curve(_data_dir)
                build_capital_deployment(_data_dir)
            except Exception as e:
                print(f"prewarm: {os.path.basename(_data_dir)} skipped — {e}")
    except Exception as e:
        print(f"prewarm: equity-curve module skipped — {e}")
    return True

_prewarm_session()


@st.cache_data(ttl=120, show_spinner=False)
def _load_summary(dirs_tuple: tuple) -> dict:
    """Aggregate key metrics across all dashboard directories."""
    total_capital   = 0.0
    total_open      = 0
    total_closed    = 0
    total_pnl       = 0.0
    last_signal     = None
    any_trades      = False

    open_positions_mtm: list = []   # [{symbol, entry_price, size_usd, direction}]

    for name, data_dir in dirs_tuple:
        trades_json = os.path.join(data_dir, "trades.json")
        if not os.path.exists(trades_json):
            continue

        try:
            trades    = load_trades(data_dir)
            pairs     = build_trade_pairs(data_dir)
            positions = load_positions(data_dir)
        except Exception:
            continue

        if not trades:
            continue

        any_trades = True

        # Live equity per strategy (= config CAPITAL + realised P&L from
        # closed trades).  Reading from load_realised_capital instead of
        # cfg.get('capital') means the metric tracks actual realised equity
        # rather than the static config allocation, so it converges to
        # what's really in the book as trades close.
        total_capital += load_realised_capital(data_dir)

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

        # Stash everything the unrealised-PnL helper needs to mark each
        # open position to the WS live price — see _compute_unrealised().
        # Done here so we only have to walk positions.json once per render.
        for pid, pos in positions.items():
            if not pos.get("in_position"):
                continue
            sym = pos.get("symbol", pid)
            ep  = pos.get("entry_price")
            sz  = pos.get("size_usd")
            if not sym or not ep or not sz:
                continue
            open_positions_mtm.append({
                "symbol":      sym,
                "entry_price": float(ep),
                "size_usd":    float(sz),
                "direction":   pos.get("direction", "long"),
            })

    return {
        "any_trades":         any_trades,
        "total_capital":      total_capital,
        "total_open":         total_open,
        "total_closed":       total_closed,
        "total_pnl":          total_pnl,
        "last_signal":        last_signal,
        "open_positions_mtm": open_positions_mtm,
    }


def _compute_unrealised(open_positions: list) -> tuple[float, bool]:
    """Mark every open position to market against the shared WS prices.

    Lives OUTSIDE _load_summary's 120-second cache so the figure stays
    fresh on every render — the WS push cost is sub-millisecond.

    Returns ``(unrealised_pnl_usd, all_priced)``.  ``all_priced`` is False
    if the WS isn't connected or any symbol is missing a tick, so the
    caller can flag the number as partial.
    """
    if not open_positions:
        return 0.0, True
    symbols = sorted({p["symbol"] for p in open_positions})
    ws      = get_shared_ws(symbols)
    if not ws.is_connected:
        return 0.0, False
    prices = ws.get_prices(symbols)
    total       = 0.0
    all_priced  = True
    for p in open_positions:
        lp = prices.get(p["symbol"])
        if lp is None or p["entry_price"] <= 0:
            all_priced = False
            continue
        sign = 1 if p["direction"].lower() == "long" else -1
        total += sign * (lp - p["entry_price"]) / p["entry_price"] * p["size_usd"]
    return total, all_priced


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
    # ── Mark every open position to market via the shared WS ─────────────────
    # Cheap, so done on every render rather than cached.
    unrealised, _all_priced = _compute_unrealised(summary["open_positions_mtm"])

    realised_pnl = summary["total_pnl"]
    live_equity  = summary["total_capital"] + unrealised

    def _signed(v: float) -> str:
        return f"+${v:,.3f}" if v >= 0 else f"-${abs(v):,.3f}"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "Total capital",
        f"${live_equity:,.0f}",
        # Delta = unrealised contribution; Streamlit colours it green/red.
        delta=_signed(unrealised) if summary["open_positions_mtm"] else None,
    )
    m2.metric("Open positions",  summary["total_open"])
    m3.metric("Closed trades",   summary["total_closed"])
    m4.metric("Realised P&L",    _signed(realised_pnl))
    m5.metric("Unrealised P&L",  _signed(unrealised))

    # Tiny status line: WS health + (if partial) flag missing ticks
    _ws_label = "🟢 Live" if _all_priced and summary["open_positions_mtm"] else (
        "🟡 Partial" if summary["open_positions_mtm"] else "—"
    )

    # ── Last signal date ──────────────────────────────────────────────────────
    last_sig = summary["last_signal"]
    sig_str  = last_sig.isoformat() if last_sig else "—"
    st.markdown(
        f'<div class="dash-meta" style="margin-top:18px">'
        f'<strong>Last signal date:</strong> {sig_str}'
        f' &nbsp;&nbsp; <strong>Mark-to-market:</strong> {_ws_label}'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<p style="color:#888780;font-size:12px;margin-top:32px">'
    'Select a page from the sidebar to view dashboards, trade logs, or portfolio analytics.'
    '</p>',
    unsafe_allow_html=True,
)
