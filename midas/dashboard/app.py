from __future__ import annotations

import math
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import dotenv_values

from dashboard.db import get_heartbeat, get_kill_switch, get_open_orders, open_db, set_kill_switch
from dashboard.env_editor import EDITABLE_KEYS, read_env, restart_bot, write_env
from dashboard.log_reader import filter_logs, tail_log

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ROOT    = Path(__file__).parent.parent
_ENV_PATH = _ROOT / ".env"
_env     = dotenv_values(_ENV_PATH)

DB_PATH  = _env.get("DB_PATH", "").strip()
LOG_FILE = _env.get("LOG_FILE", "").strip()
WALLET   = _env.get("PM_FUNDER", "").strip()
DATA_API = "https://data-api.polymarket.com"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Midas Monitor", layout="wide", page_icon="⚡")

# ---------------------------------------------------------------------------
# CSS — Hyperliquid/dYdX dark aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800&display=swap');

/* ── Reset & base ──────────────────────────────────────── */
#MainMenu, footer { visibility: hidden; }
.stDeployButton  { display: none; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── Tabs ──────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 24px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
    background: transparent;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #f0f6fc !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #3b82f6 !important;
    height: 2px !important;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: #21262d !important;
    height: 1px !important;
}

/* ── Headings ──────────────────────────────────────────── */
h2, h3 {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: #484f58 !important;
    margin-top: 28px !important;
    margin-bottom: 12px !important;
}

/* ── Divider ───────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #21262d !important;
    margin: 28px 0 !important;
}

/* ── Radio → pill buttons ──────────────────────────────── */
div[data-testid="stRadio"] > div {
    display: flex;
    gap: 6px;
    flex-direction: row;
    flex-wrap: wrap;
}
div[data-testid="stRadio"] label {
    background: #161b27;
    border: 1px solid #21262d;
    border-radius: 20px;
    padding: 5px 14px;
    cursor: pointer;
    transition: all 0.15s;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(59, 130, 246, 0.1);
    border-color: #3b82f6;
}
div[data-testid="stRadio"] label p {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #8b949e;
    margin: 0 !important;
}
div[data-testid="stRadio"] label:has(input:checked) p {
    color: #60a5fa !important;
}

/* ── Buttons ───────────────────────────────────────────── */
.stButton > button {
    background: #161b27 !important;
    border: 1px solid #21262d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: all 0.15s !important;
    padding: 8px 20px !important;
}
.stButton > button:hover {
    border-color: #3b82f6 !important;
    color: #60a5fa !important;
    background: rgba(59, 130, 246, 0.06) !important;
}
.stButton > button[kind="primary"] {
    background: rgba(59, 130, 246, 0.12) !important;
    border-color: #3b82f6 !important;
    color: #60a5fa !important;
}

/* ── Form inputs ───────────────────────────────────────── */
.stNumberInput input, .stTextInput input {
    background: #161b27 !important;
    border-color: #21262d !important;
    color: #f0f6fc !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
    font-family: 'Inter', sans-serif !important;
    font-variant-numeric: tabular-nums;
}
.stSelectbox > div > div {
    background: #161b27 !important;
    border-color: #21262d !important;
    border-radius: 8px !important;
}

/* ── Dataframe ─────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
.stDataFrame iframe {
    border-radius: 10px !important;
}

/* ── Alerts ────────────────────────────────────────────── */
div[data-testid="stAlert"] {
    background: rgba(22, 27, 39, 0.8) !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    color: #8b949e !important;
    font-size: 0.85rem !important;
}
div[data-testid="stAlert"] p { color: #8b949e !important; }

/* ── Caption ───────────────────────────────────────────── */
.stCaption, small { color: #484f58 !important; font-size: 0.68rem !important; }

/* ── KPI cards ─────────────────────────────────────────── */
.kpi-card {
    background: #161b27;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px 22px 16px;
    transition: border-color 0.2s, box-shadow 0.2s;
    height: 100%;
    box-sizing: border-box;
}
.kpi-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 0 0 1px rgba(59,130,246,0.15), 0 4px 24px rgba(0,0,0,0.4);
}
.kpi-label {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #8b949e;
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: #f0f6fc;
    line-height: 1;
    letter-spacing: -0.02em;
}
.kpi-sub {
    font-size: 0.68rem;
    color: #8b949e;
    margin-top: 7px;
    font-weight: 400;
}

/* ── Status chips ──────────────────────────────────────── */
.chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    white-space: nowrap;
}
.chip-green  { background:rgba(16,185,129,.10); color:#10b981; border:1px solid rgba(16,185,129,.25); }
.chip-red    { background:rgba(239,68,68,.10);  color:#ef4444; border:1px solid rgba(239,68,68,.25); }
.chip-yellow { background:rgba(245,158,11,.10); color:#f59e0b; border:1px solid rgba(245,158,11,.25); }
.chip-blue   { background:rgba(59,130,246,.10); color:#60a5fa; border:1px solid rgba(59,130,246,.25); }
.chip-gray   { background:rgba(139,148,158,.08);color:#8b949e; border:1px solid rgba(139,148,158,.2); }

/* ── Pulse dot animation ───────────────────────────────── */
@keyframes midas-pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.45; transform:scale(1.3); }
}
.pulse-dot {
    display:inline-block; width:6px; height:6px;
    border-radius:50%; background:#10b981;
    animation:midas-pulse 2s ease-in-out infinite;
    vertical-align:middle; margin-right:2px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_TIMEFRAME_S: dict[str, int] = {
    "1d": 86_400, "3d": 259_200, "7d": 604_800, "30d": 2_592_000, "All": 0,
}

_LEVEL_COLOR = {
    "DEBUG":   "#484f58",
    "INFO":    "#8b949e",
    "WARNING": "#f59e0b",
    "ERROR":   "#ef4444",
}


def _cutoff(tf: str) -> float:
    secs = _TIMEFRAME_S[tf]
    return time.time() - secs if secs else 0.0


def _db() -> sqlite3.Connection | None:
    if not DB_PATH:
        return None
    try:
        return open_db(DB_PATH)
    except Exception:
        return None


def _safe_float(d: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


def _activity_ts(a: dict) -> float:
    raw = a.get("timestamp", a.get("createdAt", 0))
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _health_flag(drawdown_pct: float) -> str:
    if drawdown_pct <= -15.0:
        return "🔴"
    if drawdown_pct <= -5.0:
        return "🟡"
    return "🟢"


def _open_positions(positions: list) -> list:
    return [
        p for p in positions
        if not (p.get("redeemable") and _safe_float(p, "curPrice") == 0.0)
    ]


@st.cache_data(ttl=30)
def _fetch_data_api(wallet: str) -> tuple[list, list]:
    if not wallet:
        return [], []
    try:
        pos = requests.get(
            f"{DATA_API}/positions", params={"user": wallet}, timeout=10
        ).json()
        act = requests.get(
            f"{DATA_API}/activity", params={"user": wallet, "limit": 500}, timeout=10
        ).json()
        def _to_list(obj: object) -> list:
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return obj.get("data") or obj.get("results") or obj.get("positions") or []
            return []

        return (_to_list(pos), _to_list(act))
    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _kpi(
    label: str,
    value: str,
    sub: str = "",
    color: str = "neutral",
) -> str:
    colors = {
        "pos":     "#10b981",
        "neg":     "#ef4444",
        "blue":    "#60a5fa",
        "neutral": "#f0f6fc",
        "muted":   "#8b949e",
    }
    val_color = colors.get(color, colors["neutral"])
    sub_html  = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{val_color}">{value}</div>'
        f'{sub_html}</div>'
    )


def _kpi_row(cards: list[str]) -> None:
    cols = st.columns(len(cards))
    for col, html in zip(cols, cards):
        col.markdown(html, unsafe_allow_html=True)


def _chip(text: str, cls: str = "chip-gray") -> str:
    return f'<span class="chip {cls}">{text}</span>'


def _section(title: str) -> None:
    st.markdown(f"### {title}")


# ---------------------------------------------------------------------------
# Plotly chart helpers
# ---------------------------------------------------------------------------

_CHART_BASE = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    margin=dict(l=4, r=4, t=12, b=4),
    height=260,
    showlegend=False,
    font=dict(family="Inter, sans-serif", color="#8b949e", size=10),
    xaxis=dict(
        gridcolor="#1a2035", gridwidth=1,
        color="#8b949e", showline=False, zeroline=False,
        tickfont=dict(size=10, family="Inter"),
    ),
    yaxis=dict(
        gridcolor="#1a2035", gridwidth=1,
        color="#8b949e", showline=False, zeroline=False,
        tickfont=dict(size=10, family="Inter"), tickprefix="$",
    ),
    hoverlabel=dict(
        bgcolor="#161b27", bordercolor="#30363d",
        font=dict(size=12, family="Inter", color="#f0f6fc"),
    ),
)


def _plot_equity(df: pd.DataFrame) -> go.Figure:
    y = df.iloc[:, 0]
    positive  = float(y.iloc[-1]) >= 0 if len(y) else True
    line_clr  = "#10b981" if positive else "#ef4444"
    fill_clr  = "rgba(16,185,129,0.07)" if positive else "rgba(239,68,68,0.07)"
    fig = go.Figure(go.Scatter(
        x=df.index, y=y,
        mode="lines",
        line=dict(color=line_clr, width=2),
        fill="tozeroy", fillcolor=fill_clr,
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#21262d", width=1))
    layout = dict(_CHART_BASE)
    layout["title"] = dict(
        text="Equity Curve",
        font=dict(size=11, color="#8b949e", family="Inter"),
        x=0.01, y=0.97,
    )
    fig.update_layout(**layout)
    return fig


def _plot_drawdown(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df.index, y=df.iloc[:, 0],
        mode="lines",
        line=dict(color="#ef4444", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.10)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    layout = dict(_CHART_BASE)
    layout["title"] = dict(
        text="Drawdown",
        font=dict(size=11, color="#8b949e", family="Inter"),
        x=0.01, y=0.97,
    )
    fig.update_layout(**layout)
    return fig


def _fmt_log_line(rec: dict) -> str:
    level  = rec.get("level", "INFO")
    color  = _LEVEL_COLOR.get(level, "#8b949e")
    ts     = rec.get("ts", "")[:23]
    event  = rec.get("event", "")
    extras = {k: v for k, v in rec.items() if k not in ("ts", "level", "event")}
    extra  = "".join(
        f'  <span style="color:#484f58">{k}</span>=<span style="color:#6e7681">{v}</span>'
        for k, v in extras.items()
    )
    return (
        f'<div style="padding:3px 0;border-bottom:1px solid #161b27;'
        f'font-family:\'JetBrains Mono\',\'Fira Code\',monospace;font-size:11px;">'
        f'<span style="color:#484f58">{ts}</span> '
        f'<span style="color:{color};font-weight:700;min-width:56px;display:inline-block">{level}</span> '
        f'<span style="color:#c9d1d9">{event}</span>{extra}</div>'
    )


# ---------------------------------------------------------------------------
# Analytics engine  (logic unchanged — only UI layer changed)
# ---------------------------------------------------------------------------

def _compute_analytics(activity: list, positions: list, cutoff: float) -> dict:
    # Step 1 — find conditionIds that have a BUY in the selected period
    period_act = [a for a in activity if _activity_ts(a) >= cutoff]
    period_buy_cids: set[str] = set()
    period_buy_first: dict[str, float] = {}
    for a in period_act:
        cid = a.get("conditionId", "")
        if not cid:
            continue
        if a.get("type", "").upper() == "TRADE" and a.get("side", "").upper() == "BUY":
            period_buy_cids.add(cid)
            ts = _activity_ts(a)
            if cid not in period_buy_first:
                period_buy_first[cid] = ts

    open_cids = {
        p.get("conditionId", "")
        for p in positions
        if _safe_float(p, "curPrice") > 0.0
    }

    # Step 2 — for those conditionIds use ALL-TIME costs & receipts so that
    # positions straddling the period boundary aren't mis-priced.
    by_cid: dict[str, dict] = {}
    for a in sorted(activity, key=_activity_ts):
        cid = a.get("conditionId", "")
        if not cid or cid not in period_buy_cids:
            continue
        t, s = a.get("type", "").upper(), a.get("side", "").upper()
        usdc = _safe_float(a, "usdcSize", "amount")
        ts   = _activity_ts(a)
        d    = by_cid.setdefault(cid, {
            "cost": 0.0, "recv": 0.0,
            "settle_ts": None, "buy_ts": period_buy_first.get(cid),
        })
        if t == "TRADE" and s == "BUY":
            d["cost"] += usdc
        elif t == "REDEEM" or (t == "TRADE" and s == "SELL") or t == "MERGE":
            d["recv"] += usdc
            d["settle_ts"] = ts

    trades: list[dict] = []
    for cid, d in by_cid.items():
        if d["cost"] == 0 or cid in open_cids:
            continue
        settle_ts = d["settle_ts"] or d["buy_ts"] or 0.0
        pnl = d["recv"] - d["cost"]
        trades.append({"ts": settle_ts, "pnl": pnl, "cost": d["cost"], "won": pnl > 0})
    trades.sort(key=lambda x: x["ts"])

    if not trades:
        return {
            "trades": [], "equity_df": pd.DataFrame(), "dd_df": pd.DataFrame(),
            "n_total": 0, "n_wins": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "total_pnl": 0.0, "max_dd": 0.0, "sharpe": 0.0, "calmar": 0.0,
            "total_in": 0.0, "exposure": 0.0,
        }

    # USDC actually deployed in the period (cash out)
    total_in = sum(
        _safe_float(a, "usdcSize", "amount") for a in period_act
        if a.get("type", "").upper() == "TRADE" and a.get("side", "").upper() == "BUY"
    )
    # All-time cost of positions that are still open (at risk)
    exposure = sum(
        d["cost"] for cid, d in by_cid.items()
        if cid in open_cids and d["cost"] > 0
    )

    daily_pnl: dict = {}
    for tr in trades:
        day = datetime.fromtimestamp(tr["ts"], tz=timezone.utc).date()
        daily_pnl[day] = daily_pnl.get(day, 0.0) + tr["pnl"]

    dates = sorted(daily_pnl)
    cum, eq_vals, dd_vals, peak = 0.0, [], [], 0.0
    for d in dates:
        cum  += daily_pnl[d]
        peak  = max(peak, cum)
        eq_vals.append(round(cum, 4))
        dd_vals.append(round(cum - peak, 4))

    baseline   = dates[0] - timedelta(days=1)
    chart_dates = [baseline] + dates
    equity_df  = pd.DataFrame({"Equity": [0.0] + eq_vals}, index=pd.to_datetime(chart_dates))
    dd_df      = pd.DataFrame({"Drawdown": [0.0] + dd_vals}, index=pd.to_datetime(chart_dates))

    n_total  = len(trades)
    n_wins   = sum(1 for t in trades if t["won"])
    n_losses = n_total - n_wins
    win_rate = n_wins / n_total if n_total else 0.0

    wins_usdc   = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    losses_usdc = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = wins_usdc / losses_usdc if losses_usdc > 0 else float("inf")
    avg_win  = wins_usdc / n_wins    if n_wins   else 0.0
    avg_loss = losses_usdc / n_losses if n_losses else 0.0
    total_pnl = sum(t["pnl"] for t in trades)
    max_dd    = abs(min(dd_vals)) if dd_vals else 0.0

    if len(daily_pnl) > 1:
        vals   = list(daily_pnl.values())
        mean_d = sum(vals) / len(vals)
        std_d  = math.sqrt(sum((v - mean_d) ** 2 for v in vals) / len(vals))
        sharpe = (mean_d / std_d * math.sqrt(252)) if std_d > 0 else 0.0
    else:
        sharpe = 0.0

    if trades and max_dd > 0:
        span_days  = max(1.0, (trades[-1]["ts"] - trades[0]["ts"]) / 86400)
        annualised = total_pnl / span_days * 365
        calmar     = annualised / max_dd
    else:
        calmar = 0.0

    return {
        "trades": trades, "equity_df": equity_df, "dd_df": dd_df,
        "n_total": n_total, "n_wins": n_wins, "win_rate": win_rate,
        "profit_factor": profit_factor, "avg_win": avg_win, "avg_loss": avg_loss,
        "total_pnl": total_pnl, "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar,
        "total_in": total_in, "exposure": exposure,
    }


# ===========================================================================
# TAB 1 — OVERVIEW
# ===========================================================================

@st.fragment(run_every=10)
def render_overview() -> None:
    conn = _db()
    hb   = get_heartbeat(conn) if conn else None

    # ── Header banner ────────────────────────────────────────────────────────
    if hb:
        try:
            ts    = datetime.fromisoformat(hb["ts"].replace("Z", "+00:00"))
            age_s = (datetime.now(timezone.utc) - ts).total_seconds()
        except Exception:
            age_s = 9999.0
        alive = age_s < 45

        dot       = '<span class="pulse-dot"></span>' if alive else "●"
        s_cls     = "chip-green" if alive else "chip-red"
        s_label   = "Live" if alive else "Dead"
        ws_m_cls  = "chip-green" if hb["ws_market_connected"] else "chip-red"
        ws_u_cls  = "chip-green" if hb["ws_user_connected"]   else "chip-red"
        last_msg  = f"{hb['last_market_msg_age_s']:.0f}s ago" if hb["last_market_msg_age_s"] else "—"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#161b27 0%,#0d1320 100%);
                    border:1px solid #21262d; border-radius:14px;
                    padding:16px 26px; display:flex; align-items:center;
                    justify-content:space-between; margin-bottom:8px; gap:16px; flex-wrap:wrap;">
            <div style="display:flex; align-items:center; gap:14px;">
                <span style="font-size:1.45rem; font-weight:800; color:#f0f6fc;
                             letter-spacing:-0.03em; line-height:1;">⚡ MIDAS</span>
                <span style="font-size:0.62rem; font-weight:700; color:#484f58;
                             text-transform:uppercase; letter-spacing:0.12em;">Tail Harvester</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                <span class="chip {s_cls}">{dot} {s_label}</span>
                <span class="chip {ws_m_cls}">Mkt WS</span>
                <span class="chip {ws_u_cls}">User WS</span>
                <span class="chip chip-blue">{hb['active_markets']} Markets</span>
                <span class="chip chip-gray">{hb['open_orders']} Orders</span>
                <span class="chip chip-gray">Last msg {last_msg}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#130e0e; border:1px solid #3d1515; border-radius:14px;
                    padding:16px 26px; display:flex; align-items:center;
                    justify-content:space-between; margin-bottom:8px;">
            <span style="font-size:1.45rem; font-weight:800; color:#f0f6fc;">⚡ MIDAS</span>
            <span class="chip chip-red">● No heartbeat — DB not configured or bot not running</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Kill switch ──────────────────────────────────────────────────────────
    kill_armed = st.session_state.get("kill_armed", False)
    kill_active = conn is not None and get_kill_switch(conn)

    if kill_active:
        st.markdown(
            '<div style="background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.35); '
            'border-radius:10px; padding:10px 18px; margin-top:8px; '
            'font-size:0.8rem; color:#ef4444; font-weight:600;">'
            '⛔ KILL SIGNAL ACTIVE — bot is shutting down. Clear it before restarting.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear Kill Switch", key="kill_clear"):
            if conn:
                set_kill_switch(conn, False)
            st.session_state.kill_armed = False
            st.rerun()
    elif kill_armed:
        st.markdown(
            '<div style="background:rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.25); '
            'border-radius:10px; padding:10px 18px; margin-top:8px; '
            'font-size:0.8rem; color:#ef4444; font-weight:600;">'
            '⚠ Confirm: this will halt all trading. Bot will cancel orders and stop within ~15s.</div>',
            unsafe_allow_html=True,
        )
        col_yes, col_no, _ = st.columns([1, 1, 6])
        if col_yes.button("✓ Stop Bot", key="kill_confirm"):
            if conn:
                set_kill_switch(conn, True)
            st.session_state.kill_armed = False
            st.rerun()
        if col_no.button("✗ Cancel", key="kill_cancel"):
            st.session_state.kill_armed = False
            st.rerun()
    else:
        col_kill, _ = st.columns([1, 7])
        if col_kill.button("⛔ Emergency Stop", key="kill_arm"):
            st.session_state.kill_armed = True
            st.rerun()

    st.divider()

    # ── Debug — raw API response (remove once positions work) ─────────────────
    with st.expander("🔍 API Debug", expanded=True):
        st.caption(f"Wallet: `{WALLET[:10]}…{WALLET[-6:]}` (from PM_FUNDER)")
        st.caption(f"URL: `{DATA_API}/positions?user={WALLET}`")
        try:
            import json as _json
            _raw = requests.get(
                f"{DATA_API}/positions", params={"user": WALLET}, timeout=10
            )
            st.caption(f"HTTP status: {_raw.status_code}")
            _body = _raw.json()
            st.caption(f"Response type: `{type(_body).__name__}`, length: {len(_body) if isinstance(_body, (list, dict)) else 'n/a'}")
            if isinstance(_body, list):
                st.caption(f"First item keys: `{list(_body[0].keys()) if _body else 'empty list'}`")
                st.json(_body[:2])
            else:
                st.json(_body)
        except Exception as _e:
            st.error(f"Raw API call failed: {_e}")

    # ── Filled Positions ─────────────────────────────────────────────────────
    _section("Filled Positions")
    positions, activity = _fetch_data_api(WALLET)
    live = _open_positions(positions)

    if live:
        rows = []
        for p in live:
            avg = _safe_float(p, "avgPrice")
            cur = _safe_float(p, "curPrice")
            pnl = _safe_float(p, "cashPnl", "pnl")
            dd  = (cur - avg) / avg * 100 if avg > 0 else 0.0
            rows.append({
                "":        _health_flag(dd),
                "Market":  str(p.get("title", p.get("market", "")))[:48],
                "Outcome": p.get("outcome", ""),
                "Size":    _safe_float(p, "size"),
                "Entry":   f"{avg * 100:.1f}¢",
                "Current": f"{cur * 100:.1f}¢",
                "Drawdown":f"{dd:+.1f}%",
                "P&L":     f"${pnl:.2f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    elif WALLET:
        st.info("No open positions")
    else:
        st.info("Set PM_FUNDER in .env to see filled positions")

    st.divider()

    # ── Resting Orders ───────────────────────────────────────────────────────
    _section("Resting Orders")
    orders = get_open_orders(conn) if conn else []
    if orders:
        rows = []
        for o in orders:
            price_pct = round(o["price_ticks"] * o["tick_size"] * 100, 1)
            rows.append({
                "Slug":   o["event_slug"],
                "Side":   "YES" if o["is_yes"] else "NO",
                "Bid":    f"{price_pct}¢",
                "Qty":    o["qty"],
                "Status": o["status"],
                "Placed": o["placed_at"][:19].replace("T", " "),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No resting orders")

    st.divider()

    # ── Session Summary ──────────────────────────────────────────────────────
    _section("Session Summary")
    tf_col, _ = st.columns([5, 7])
    tf  = tf_col.radio("Period", list(_TIMEFRAME_S), horizontal=True,
                       label_visibility="collapsed", key="overview_tf")
    cut = _cutoff(tf)

    if positions or activity:
        period_act = [a for a in activity if _activity_ts(a) >= cut]
        buys = [
            a for a in period_act
            if a.get("type", "").upper() == "TRADE" and a.get("side", "").upper() == "BUY"
        ]
        total_in = sum(_safe_float(a, "usdcSize", "amount") for a in buys)

        open_cids = {p.get("conditionId", "") for p in positions if _safe_float(p, "curPrice") > 0.0}

        # Which conditionIds had a BUY in this period?
        period_buy_cids: set[str] = {
            a.get("conditionId", "") for a in period_act
            if a.get("type", "").upper() == "TRADE" and a.get("side", "").upper() == "BUY"
            and a.get("conditionId", "")
        }

        # Use ALL-TIME activity for those conditionIds to avoid partial-cost errors
        m_cost: dict[str, float] = {}
        m_recv: dict[str, float] = {}
        for a in activity:
            cid = a.get("conditionId", "")
            if not cid or cid not in period_buy_cids:
                continue
            t, s = a.get("type", "").upper(), a.get("side", "").upper()
            if t == "TRADE" and s == "BUY":
                m_cost[cid] = m_cost.get(cid, 0.0) + _safe_float(a, "usdcSize", "amount")
            elif t == "REDEEM" or (t == "TRADE" and s == "SELL") or t == "MERGE":
                m_recv[cid] = m_recv.get(cid, 0.0) + _safe_float(a, "usdcSize", "amount")

        period_pnl = exposure = 0.0
        n_wins = n_losses = 0
        for cid, cost in m_cost.items():
            recv = m_recv.get(cid, 0.0)
            if cid in open_cids:
                exposure += cost
            elif recv > 0 or cid in m_recv:
                period_pnl += recv - cost
                if recv > cost:
                    n_wins += 1
                else:
                    n_losses += 1
            else:
                period_pnl -= cost
                n_losses += 1

        n_settled = n_wins + n_losses
        wr_label  = f"Win Rate ({n_wins}/{n_settled})" if n_settled else "Win Rate"
        wr_val    = f"{n_wins / n_settled * 100:.0f}%" if n_settled else "—"

        # Unrealized P&L on open positions (mark-to-market)
        unrealized_pnl = sum(
            (_safe_float(p, "curPrice") - _safe_float(p, "avgPrice")) * _safe_float(p, "size")
            for p in positions
            if _safe_float(p, "curPrice") > 0.0 and _safe_float(p, "avgPrice") > 0.0
        )
        total_pnl = period_pnl + unrealized_pnl
        pnl_color = "pos" if total_pnl >= 0 else "neg"

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        _kpi_row([
            _kpi("Fills",           str(len(buys)),         color="neutral"),
            _kpi("USDC Deployed",   f"${total_in:.2f}",     color="neutral"),
            _kpi("Exposure",        f"${exposure:.2f}",     color="blue",
                 sub="At-risk in open markets"),
            _kpi("Realized P&L",    f"${period_pnl:.2f}",  color="pos" if period_pnl >= 0 else "neg",
                 sub="Settled trades only"),
            _kpi("Total P&L",       f"${total_pnl:.2f}",   color=pnl_color,
                 sub=f"Incl. ${unrealized_pnl:+.2f} unrealized"),
        ])
        _kpi_row([
            _kpi(wr_label, wr_val, color="neutral", sub="Settled trades only"),
        ])
    elif WALLET:
        st.warning("Could not reach Polymarket Data API")
    else:
        st.info("Set PM_FUNDER in .env to enable session summary")

    st.divider()

    # ── Recent Logs ──────────────────────────────────────────────────────────
    _section("Recent Logs")
    if LOG_FILE:
        lines = tail_log(LOG_FILE, n=15)
        if lines:
            st.markdown(
                '<div style="background:#0d1117; border:1px solid #21262d; border-radius:10px; '
                'padding:14px 18px; overflow-x:auto;">'
                + "".join(_fmt_log_line(r) for r in lines)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Log file is empty")
    else:
        st.info("Set LOG_FILE in .env to enable log tail")


# ===========================================================================
# TAB 2 — LOGS
# ===========================================================================

def render_logs() -> None:
    _section("Log Viewer")
    if not LOG_FILE:
        st.error("LOG_FILE not set in .env")
        return
    col1, col2, col3 = st.columns([1, 2, 1])
    min_level = col1.selectbox("Min level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
    keyword   = col2.text_input("Keyword filter")
    n_lines   = col3.number_input("Lines to load", min_value=50, max_value=2000, value=200, step=50)
    lines    = tail_log(LOG_FILE, n=int(n_lines))
    filtered = filter_logs(lines, level=min_level, keyword=keyword or None)
    st.caption(f"{len(filtered)} / {len(lines)} lines shown")
    if filtered:
        st.markdown(
            '<div style="background:#0d1117; border:1px solid #21262d; border-radius:10px; '
            'padding:14px 18px; overflow-x:auto; max-height:70vh; overflow-y:auto;">'
            + "".join(_fmt_log_line(r) for r in filtered)
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No matching log lines")


# ===========================================================================
# TAB 3 — P&L
# ===========================================================================

def render_pnl() -> None:
    if not WALLET:
        st.error("PM_FUNDER not set in .env — required for Polymarket Data API")
        return

    positions, activity = _fetch_data_api(WALLET)

    tf_col, _ = st.columns([5, 7])
    tf  = tf_col.radio("Period", list(_TIMEFRAME_S), horizontal=True,
                       label_visibility="collapsed", key="pnl_tf")
    cut = _cutoff(tf)

    an = _compute_analytics(activity, positions, cut)

    # ── Metrics ─────────────────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if an["n_total"] > 0:
        pf_str  = f"{an['profit_factor']:.2f}" if an["profit_factor"] != float("inf") else "∞"
        cal_str = f"{an['calmar']:.2f}"         if an["calmar"] != 0                  else "—"
        wr_lbl  = f"Win Rate ({an['n_wins']}/{an['n_total']})"
        pnl_clr = "pos" if an["total_pnl"] >= 0 else "neg"

        _kpi_row([
            _kpi("Net P&L",       f"${an['total_pnl']:.2f}", color=pnl_clr),
            _kpi(wr_lbl,          f"{an['win_rate']*100:.0f}%"),
            _kpi("Profit Factor", pf_str,   color="blue",    sub="Gross wins ÷ gross losses"),
            _kpi("Avg Win",       f"${an['avg_win']:.2f}",   color="pos"),
        ])
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        _kpi_row([
            _kpi("Avg Loss",     f"${an['avg_loss']:.2f}",  color="neg"),
            _kpi("Sharpe",       f"{an['sharpe']:.2f}",     color="blue", sub="Annualised · daily $"),
            _kpi("Max Drawdown", f"${an['max_dd']:.2f}",    color="neg"),
            _kpi("Calmar",       cal_str,                   color="blue", sub="Ann. return ÷ max DD"),
        ])
    else:
        st.info("No settled trades in this period — metrics unavailable.")

    # ── Charts ───────────────────────────────────────────────────────────────
    if not an["equity_df"].empty:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        col_eq, col_dd = st.columns(2)
        with col_eq:
            st.plotly_chart(_plot_equity(an["equity_df"]),
                            use_container_width=True, config={"displayModeBar": False})
        with col_dd:
            st.plotly_chart(_plot_drawdown(an["dd_df"]),
                            use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── Open Positions ───────────────────────────────────────────────────────
    _section("Open Positions")
    live = _open_positions(positions)
    if live:
        rows = []
        for p in live:
            avg = _safe_float(p, "avgPrice")
            cur = _safe_float(p, "curPrice")
            pnl = _safe_float(p, "cashPnl", "pnl")
            dd  = (cur - avg) / avg * 100 if avg > 0 else 0.0
            rows.append({
                "":          _health_flag(dd),
                "Market":    str(p.get("title", p.get("market", "")))[:55],
                "Outcome":   p.get("outcome", ""),
                "Size":      _safe_float(p, "size"),
                "Avg Entry": f"{avg * 100:.1f}¢",
                "Current":   f"{cur * 100:.1f}¢",
                "Drawdown":  f"{dd:+.1f}%",
                "Value":     f"${_safe_float(p, 'currentValue'):.2f}",
                "P&L":       f"${pnl:.2f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")

    st.divider()

    # ── Recent Trades ────────────────────────────────────────────────────────
    _section("Recent Trades")
    period_trades = [
        a for a in activity
        if _activity_ts(a) >= cut
        and a.get("type", "").upper() == "TRADE"
        and a.get("side", "").upper() == "BUY"
    ]
    if period_trades:
        rows = []
        for a in period_trades[:200]:
            ts_raw = a.get("timestamp", 0)
            ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            rows.append({
                "Time":    ts,
                "Market":  str(a.get("title", a.get("market", "")))[:52],
                "Outcome": a.get("outcome", ""),
                "Size":    _safe_float(a, "size"),
                "Price":   f"{_safe_float(a, 'price') * 100:.1f}¢",
                "USDC":    f"${_safe_float(a, 'usdcSize', 'amount'):.2f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No trades in this period")

    st.caption(
        f"Data API · 30s cache · {WALLET[:10]}…  ·  "
        f"Sharpe & Calmar annualised from daily $ P&L"
    )


# ===========================================================================
# TAB 4 — PARAMETERS
# ===========================================================================

def render_parameters() -> None:
    _section("Strategy Parameters")
    env_path = str(_ENV_PATH)
    current  = read_env(env_path)

    st.markdown(
        '<div style="background:rgba(245,158,11,.06); border:1px solid rgba(245,158,11,.2); '
        'border-radius:10px; padding:12px 18px; margin-bottom:20px; '
        'font-size:0.8rem; color:#f59e0b; font-weight:500;">'
        '⚠ Changes take effect after restarting the bot.</div>',
        unsafe_allow_html=True,
    )

    updates: dict[str, str] = {}
    for key, typ, label in EDITABLE_KEYS:
        val = current.get(key, "")
        if typ is float:
            new_val = st.number_input(label, value=float(val) if val else 0.0,
                                      key=key, format="%.4f")
            updates[key] = str(new_val)
        elif typ is int:
            new_val = st.number_input(label, value=int(float(val)) if val else 0,
                                      key=key, step=1)
            updates[key] = str(int(new_val))
        else:
            new_val = st.text_input(label, value=val, key=key)
            updates[key] = str(new_val)

    st.divider()
    col1, col2, _ = st.columns([1, 1, 4])
    if col1.button("Save .env", type="primary"):
        try:
            write_env(env_path, updates)
            st.success(".env updated — restart the bot to apply changes")
        except Exception as exc:
            st.error(f"Failed to save: {exc}")
    if col2.button("Restart Bot"):
        ok, msg = restart_bot()
        (st.success if ok else st.error)(msg)


# ===========================================================================
# Wire tabs
# ===========================================================================

tab_overview, tab_pnl, tab_logs, tab_params = st.tabs(
    ["Overview", "P&L", "Logs", "Parameters"]
)

with tab_overview:
    render_overview()

with tab_pnl:
    render_pnl()

with tab_logs:
    render_logs()

with tab_params:
    render_parameters()
