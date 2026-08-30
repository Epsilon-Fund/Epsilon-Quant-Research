"""Epsilon Polymarket research terminal — a dashboard built ONLY on epsilon_data.

Two jobs: audit (is the data sound?) and explore (where might a strategy be?). Search-first
navigation, a persistent header naming the market on screen, dark theme, consistent colours.
Run:  streamlit run dashboard/app.py   (from polymarket/research/)
"""
from __future__ import annotations
import sys
import pathlib

# make epsilon_data importable however streamlit is launched
_root = next((p for p in pathlib.Path(__file__).resolve().parents if (p / "epsilon_data").is_dir()), None)
if _root and str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import epsilon_data as ed

# ---- fixed colour language (colourblind-safe; used everywhere, meaning only) ----
C_A = "#4c9be8"      # YES / first outcome / side A
C_B = "#e8983c"      # NO / second outcome / side B
C_WON = "#4cae72"    # won / up
C_LOST = "#d1495b"   # lost / down
C_MUTE = "#8a93a3"
plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117", "savefig.facecolor": "#0e1117",
    "axes.edgecolor": "#39404d", "axes.labelcolor": "#d6dae1", "text.color": "#d6dae1",
    "xtick.color": "#8a93a3", "ytick.color": "#8a93a3", "grid.color": "#232936",
    "axes.grid": True, "font.size": 9, "figure.autolayout": True,
})

st.set_page_config(page_title="Epsilon · Polymarket research", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>table, .stDataFrame, code, .mono {font-family: 'DejaVu Sans Mono', monospace !important;}"
            "[data-testid='stMetricValue']{font-family:monospace;}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner=False)
def cat_all():
    return ed.catalog(apply_exclusions=False)

@st.cache_data(show_spinner=False)
def events_df(universe=None):
    return ed.events(universe=universe, apply_exclusions=False)

@st.cache_data(show_spinner=False)
def search_df(text):
    return ed.search(text, limit=100)

@st.cache_data(show_spinner=False)
def coverage_df():
    return ed.coverage()

@st.cache_data(show_spinner=False)
def recon_df():
    return ed.reconciliation()

@st.cache_data(show_spinner=False)
def activity_df(universe):
    return ed.activity_by_time(universe)

@st.cache_data(show_spinner=True)
def l1_df(asset_id, start=None, end=None):
    return ed.load_l1(asset_id, start, end)

@st.cache_data(show_spinner=True)
def trades_df(asset_id, start=None, end=None):
    return ed.load_trades(asset_id, start, end)

@st.cache_data(show_spinner=True)
def pair_df(condition_id):
    p = ed.load_pair(condition_id)
    return p, p.attrs.get("labels", {})

@st.cache_data(show_spinner=True)
def event_wide(event_slug):
    w = ed.load_event(event_slug)
    return w, w.attrs.get("tokens", {})

@st.cache_data(show_spinner=True)
def markout_df(asset_id):
    return ed.markout(asset_id)

@st.cache_data(show_spinner=True)
def negrisk_df(event_slug):
    ns = ed.negrisk_sum(event_slug)
    return ns, ns.attrs.get("n_captured", 0)

PLOT_BG = "#0e1117"
def _style(fig, height):
    fig.update_layout(template="plotly_dark", paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      height=height, margin=dict(l=50, r=20, t=30, b=20), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", font=dict(size=10)),
                      font=dict(family="monospace", size=11))
    fig.update_xaxes(gridcolor="#232936", showspikes=True, spikemode="across", spikethickness=1)
    fig.update_yaxes(gridcolor="#232936")
    return fig

def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))

_LOGIT_TICKS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]


def mono(df: pd.DataFrame):
    return df.style.set_properties(**{"font-family": "monospace", "text-align": "right"})


# ---------------------------------------------------------------- state
if "cid" not in st.session_state:
    st.session_state.cid = None      # selected condition_id (a market = 2 tokens)

def select(cid):
    st.session_state.cid = str(cid)


# ---------------------------------------------------------------- sidebar: search-first
cat = cat_all()
st.sidebar.title("Epsilon · Polymarket")
st.sidebar.caption(f"{len(cat):,} tokens · {cat.condition_id.nunique():,} markets · 2 universes")
mode = st.sidebar.radio("Mode", ["Explore", "Audit"], horizontal=True)

st.sidebar.markdown("### Find a market")
query = st.sidebar.text_input("Search (question or slug)", placeholder="e.g. fed july", key="q")

if query:
    res = search_df(query)
    if res.empty:
        st.sidebar.info("No match.")
    else:
        # one row per market (condition), label with question + outcome context
        res_m = res.drop_duplicates("condition_id").head(40)
        opts = {f"{r.question[:60]}  ·  {r.universe[:3]}  ·  {int(r.n_trades)}t": r.condition_id
                for r in res_m.itertuples()}
        pick = st.sidebar.radio(f"{len(opts)} match(es)", list(opts.keys()), index=0, key="pick")
        if st.sidebar.button("Open →", width="stretch"):
            select(opts[pick])
else:
    st.sidebar.caption("Or browse the busiest:")
    busy = cat[cat.n_trades > 0].sort_values("n_trades", ascending=False).drop_duplicates("condition_id").head(15)
    for r in busy.itertuples():
        if st.sidebar.button(f"{r.question[:42]} · {int(r.n_trades)}t", key=f"busy{r.condition_id}", width="stretch"):
            select(r.condition_id)

with st.sidebar.expander("Browse the tree"):
    uni = st.selectbox("Universe", ["politics_negrisk", "esports"])
    evs = events_df(uni)
    evs = evs[evs.n_markets > 0]
    ev_pick = st.selectbox("Event", evs.event_slug.tolist()[:500])
    ev_markets = cat[(cat.event_slug == ev_pick)].drop_duplicates("condition_id")
    m_pick = st.selectbox("Market", ev_markets.question.tolist())
    if st.button("Open market"):
        select(ev_markets[ev_markets.question == m_pick].condition_id.iloc[0])


# ---------------------------------------------------------------- helpers
def market_rows(cid):
    return cat[cat.condition_id == str(cid)].sort_values("outcome_index")

def header(cid):
    m = market_rows(cid)
    if m.empty:
        return
    r0 = m.iloc[0]
    st.markdown(f"### {r0.event_title or '(no event)'}")
    st.markdown(f"**{r0.question}**  —  {r0.universe}  ·  neg_risk={bool(r0.neg_risk)}  ·  "
                f"{'RESOLVED' if r0.closed else 'open'}")
    cols = st.columns(len(m))
    for c, (_, row) in zip(cols, m.iterrows()):
        colour = C_A if row.outcome_index == 0 else C_B
        won = (row.resolved_outcome == row.outcome_label) if pd.notna(row.resolved_outcome) else None
        tag = "  🏆" if won else ("  ✖" if won is False else "")
        c.markdown(f"<span style='color:{colour}'>■</span> **{row.outcome_label}**{tag}", unsafe_allow_html=True)
        c.caption(f"path: {row.path}")
        c.caption(f"trades {int(row.n_trades):,} · l1 {int(row.n_l1_events):,} · "
                  f"spread {row.median_spread_cents:.1f}¢ · mid {row.median_mid:.3f}")
    checks = {k: r0[k] for k in ["check1_roundtrip", "check2_pairing", "check3_negrisk", "check4_status", "check5_indep"]}
    st.caption("checks · " + " · ".join(f"{k.replace('check','c').split('_')[0]}={v}" for k, v in checks.items())
               + (f"  ·  EXCLUDED" if bool(r0.get("excluded")) else ""))
    st.divider()


# ================================================================ EXPLORE
_RANGEBUTTONS = dict(buttons=[
    dict(count=1, label="1h", step="hour", stepmode="backward"),
    dict(count=6, label="6h", step="hour", stepmode="backward"),
    dict(count=1, label="1d", step="day", stepmode="backward"),
    dict(count=7, label="1w", step="day", stepmode="backward"),
    dict(step="all", label="all"),
], bgcolor="#161a23", activecolor="#4c9be8", font=dict(size=10))

def _sg(n):  # WebGL above ~20k points
    return go.Scattergl if n > 20000 else go.Scatter

def _bucket(span_s):
    return "1min" if span_s <= 2*3600 else "5min" if span_s <= 26*3600 else "30min" if span_s <= 8*86400 else "1h"

def _window(ts_max, choice):
    return {"1h": pd.Timedelta("1h"), "6h": pd.Timedelta("6h"), "1d": pd.Timedelta("1D"),
            "1w": pd.Timedelta("7D")}.get(choice)

def market_panel(l1, pair, trades, labels, scale):
    """I2 — stacked, shared-x plotly: A price+band+trades, B volume, C imbalance, D spread, E vol."""
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.42, 0.14, 0.16, 0.14, 0.14],
                        subplot_titles=("price · bid–ask band · trades", "volume (buy↑ / sell↓)",
                                        "order-flow imbalance & cumulative signed", "spread", "realised vol (mid returns)"))
    logit = (scale == "logit")
    ytx = (_logit([b/100 for b in [0.5]]) if logit else None)
    def yv(p_dollars):
        return _logit(p_dollars) if logit else np.asarray(p_dollars, dtype=float) * 100
    n = len(l1)
    Sg = _sg(n)
    # A: band (ask top, bid fill), mid, complement mid, trades
    fig.add_trace(Sg(x=l1.ts, y=yv(l1.best_ask), name="ask", line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(Sg(x=l1.ts, y=yv(l1.best_bid), name="bid–ask band", fill="tonexty",
                     fillcolor="rgba(76,155,232,0.12)", line=dict(width=0), hoverinfo="skip"), 1, 1)
    fig.add_trace(Sg(x=l1.ts, y=yv(l1.mid), name=f"{labels.get(0,'A')} mid", line=dict(color=C_A, width=1.3)), 1, 1)
    if pair is not None and not pair.empty and 1 in pair.columns:
        fig.add_trace(_sg(len(pair))(x=pair.index, y=yv(pair[1]), name=f"{labels.get(1,'B')} mid",
                                     line=dict(color=C_B, width=1.0)), 1, 1)
    if trades is not None and not trades.empty:
        smax = max(float(trades["size"].max()), 1.0)
        for side, col in (("BUY", C_WON), ("SELL", C_LOST)):
            t = trades[trades.side == side]
            if t.empty: continue
            fig.add_trace(_sg(len(t))(x=t.ts, y=yv(t.price), mode="markers", name=f"{side}",
                          marker=dict(color=col, size=5 + 16 * (t["size"] / smax), opacity=0.55, line=dict(width=0)),
                          customdata=np.c_[t["size"], t.price * 100],
                          hovertemplate=side + " %{customdata[0]:.0f}@%{customdata[1]:.1f}¢<extra></extra>"), 1, 1)
    if logit:
        fig.update_yaxes(tickvals=_logit(_LOGIT_TICKS), ticktext=[f"{p:g}" for p in _LOGIT_TICKS], row=1, col=1, title="prob (logit)")
    else:
        fig.update_yaxes(title="¢", row=1, col=1)
    # B/C: volume + imbalance from trades in bucket
    if trades is not None and not trades.empty:
        span_s = (trades.ts.max() - trades.ts.min()).total_seconds() or 1
        b = _bucket(span_s)
        t = trades.set_index("ts")
        buy = t[t.side == "BUY"]["size"].resample(b).sum()
        sell = t[t.side == "SELL"]["size"].resample(b).sum()
        idx = buy.index.union(sell.index)
        buy = buy.reindex(idx, fill_value=0); sell = sell.reindex(idx, fill_value=0)
        fig.add_trace(go.Bar(x=idx, y=buy, name="buy vol", marker_color=C_WON, showlegend=False), 2, 1)
        fig.add_trace(go.Bar(x=idx, y=-sell, name="sell vol", marker_color=C_LOST, showlegend=False), 2, 1)
        net = buy - sell
        fig.add_trace(go.Bar(x=idx, y=net, name="net (buy-sell)", marker_color=C_MUTE, showlegend=False), 3, 1)
        signed = t["size"] * np.where(t.side == "BUY", 1.0, -1.0)
        cum = signed.cumsum()
        fig.add_trace(_sg(len(cum))(x=cum.index, y=cum.values, name="cum signed", line=dict(color="#c98bdb", width=1.2)), 3, 1)
        fig.update_yaxes(title="contracts", row=2, col=1); fig.update_yaxes(title="net / cum", row=3, col=1)
        fig.update_layout(barmode="relative")
    # D: spread step
    fig.add_trace(_sg(n)(x=l1.ts, y=l1.spread_c, name="spread", line=dict(color=C_MUTE, width=1, shape="hv"), showlegend=False), 4, 1)
    fig.update_yaxes(title="¢", row=4, col=1)
    # E: realised vol (rolling std of 1-min mid returns)
    mid1 = l1.set_index("ts")["mid"].resample("1min").last().ffill()
    rv = mid1.pct_change().rolling(st.session_state.get("rvwin", 5), min_periods=2).std() * 100
    fig.add_trace(_sg(len(rv))(x=rv.index, y=rv.values, name="rv", line=dict(color="#e8c14c", width=1), showlegend=False), 5, 1)
    fig.update_yaxes(title="%·", row=5, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), rangeselector=_RANGEBUTTONS, row=5, col=1)
    return _style(fig, 820)


def explore(cid):
    m = market_rows(cid)
    labels = {int(r.outcome_index): r.outcome_label for r in m.itertuples()}
    a_row = m.iloc[0]

    # controls
    cc = st.columns([1.4, 1, 1, 3])
    win = cc[0].radio("window", ["all", "1w", "1d", "6h", "1h"], horizontal=True, index=0)
    scale = cc[1].radio("scale", ["linear", "logit"], horizontal=True)
    st.session_state["rvwin"] = cc[2].selectbox("vol win (min)", [1, 5, 15], index=1)

    l1_full = l1_df(a_row.asset_id)
    if l1_full.empty:
        st.info("No L1 events for this market's side A — a quiet or unobserved token. "
                "Nothing to plot; this is a real state, not an error."); return
    tr_full = trades_df(a_row.asset_id)
    try:
        pair_full, _ = pair_df(cid)
    except Exception:
        pair_full = None

    end = l1_full.ts.max(); delta = _window(end, win)
    start = end - delta if delta is not None else l1_full.ts.min()
    def cut(df, col="ts"):
        if df is None or df.empty: return df
        return df[(df[col] >= start) & (df[col] <= end)]
    l1 = cut(l1_full); tr = cut(tr_full)
    pair = pair_full[(pair_full.index >= start) & (pair_full.index <= end)] if pair_full is not None and not pair_full.empty else pair_full

    st.caption(f"**Window:** {win} · {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} UTC · "
               f"{len(l1):,} L1 events, {len(tr):,} trades (stats below cover this window)")

    tabs = st.tabs(["Market", "Markout (adverse selection)", "NegRisk"])

    with tabs[0]:
        # window stats
        s = st.columns(5)
        s[0].metric("median spread", f"{l1.spread_c.median():.1f}¢")
        s[1].metric("mid range", f"{l1.mid.min()*100:.1f}–{l1.mid.max()*100:.1f}¢")
        s[2].metric("trades", f"{len(tr):,}")
        s[3].metric("buy vol", f"{tr[tr.side=='BUY']['size'].sum():,.0f}" if len(tr) else "0")
        s[4].metric("sell vol", f"{tr[tr.side=='SELL']['size'].sum():,.0f}" if len(tr) else "0")
        st.plotly_chart(market_panel(l1, pair, tr, labels, scale), width="stretch", theme=None)
        st.caption("Zoom/pan any subplot — the x-axis is shared. Band = tradeable bid–ask spread; "
                   "C (imbalance) is the 'are we being run over' panel: net one-sided volume with price following it "
                   "is adverse selection. Logit scale makes 0.99 vs 0.999 visible.")

    with tabs[1]:
        markout_tab(a_row.asset_id, start, end)

    with tabs[2]:
        negrisk_tab(a_row)


def markout_tab(asset_id, start, end):
    st.caption("`trades.side` is the **taker** side (verified: BUY lifts the ask, SELL hits the bid). "
               "Markout is shown from the **liquidity provider (maker)** view: **negative = the resting "
               "quote was adversely selected** (price moved against the maker). Positive = benign flow.")
    mo = markout_df(asset_id)
    mo = mo[(mo.ts >= start) & (mo.ts <= end)] if not mo.empty else mo
    if mo.empty:
        st.info("No trades in this window — nothing to mark out. (A quiet market, not an error.)"); return
    h = st.selectbox("horizon", [10, 30, 60], index=1, format_func=lambda x: f"{x}s")
    col = f"markout_{h}"
    mo = mo.dropna(subset=[col])
    if mo.empty:
        st.info("Trades exist but no post-trade mid within the horizon (near end of life)."); return
    s = st.columns(3)
    s[0].metric("mean markout (all)", f"{mo[col].mean()*100:+.3f}¢")
    s[1].metric("mean · BUY (maker sold)", f"{mo[mo.side=='BUY'][col].mean()*100:+.3f}¢")
    s[2].metric("mean · SELL (maker bought)", f"{mo[mo.side=='SELL'][col].mean()*100:+.3f}¢")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for side, colr in (("BUY", C_WON), ("SELL", C_LOST)):
            x = mo[mo.side == side][col] * 100
            if len(x):
                fig.add_trace(go.Histogram(x=x.clip(-5, 5), name=side, marker_color=colr, opacity=0.6, nbinsx=60))
        fig.add_vline(x=0, line=dict(color="#fff", width=1, dash="dot"))
        fig.update_layout(barmode="overlay", title=f"markout distribution @ {h}s (¢)")
        st.plotly_chart(_style(fig, 320), width="stretch", theme=None)
    with c2:
        fig = go.Figure()
        fig.add_trace(_sg(len(mo))(x=mo.ts, y=mo[col] * 100, mode="markers",
                      marker=dict(color=np.where(mo.side == "BUY", C_WON, C_LOST), size=4, opacity=0.5),
                      name="per-trade"))
        roll = (mo.set_index("ts")[col] * 100).rolling(50, min_periods=5).mean()
        fig.add_trace(_sg(len(roll))(x=roll.index, y=roll.values, line=dict(color="#e8c14c", width=1.5), name="rolling mean(50)"))
        fig.add_hline(y=0, line=dict(color="#fff", width=1, dash="dot"))
        fig.update_layout(title=f"markout over time @ {h}s (¢) — spikes below 0 = toxic flow")
        st.plotly_chart(_style(fig, 320), width="stretch", theme=None)
    neg = (mo[col] < 0).mean()
    st.caption(f"{neg:.0%} of fills had negative maker markout at {h}s in this window. "
               "Consistently negative (especially on one side) = flow you'd have been run over by.")


def negrisk_tab(a_row):
    if not bool(a_row.neg_risk):
        st.info("This market is not NegRisk — the sum-to-1 view applies to NegRisk events (politics)."); return
    slug = a_row.event_slug
    ev_markets = cat[cat.event_slug == slug].condition_id.nunique()
    if ev_markets < 2:
        st.info(f"Only {ev_markets} market captured for this event — nothing to sum."); return
    try:
        wide, meta = event_wide(slug)
        ns, ncap = negrisk_df(slug)
    except Exception as e:
        st.warning(f"event unavailable: {e}"); return
    yes_cols = [a for a in wide.columns if meta.get(a, {}).get("outcome") == "YES"]
    if not yes_cols:
        yes_cols = [a for a in wide.columns if meta.get(a, {}).get("outcome_index") == 0]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.04,
                        subplot_titles=("every candidate's YES mid", "YES sum (instantaneous, common timestamp)"))
    for a in yes_cols:
        fig.add_trace(_sg(len(wide))(x=wide.index, y=wide[a] * 100, line=dict(width=0.7),
                      opacity=0.5, showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(_sg(len(ns))(x=ns.index, y=ns.yes_sum * 100, line=dict(color="#e8c14c", width=1.6), name="YES sum"), 2, 1)
    fig.add_hline(y=100, line=dict(color=C_LOST, width=1, dash="dash"), row=2, col=1)
    fig.update_yaxes(title="¢", row=1, col=1); fig.update_yaxes(title="sum ¢", row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=2, col=1)
    st.plotly_chart(_style(fig, 560), width="stretch", theme=None)
    s = st.columns(3)
    s[0].metric("candidates captured", f"{ncap}")
    s[1].metric("YES sum · median", f"{ns.yes_sum.median():.3f}")
    s[2].metric("YES sum · latest", f"{ns.yes_sum.dropna().iloc[-1]:.3f}" if ns.yes_sum.notna().any() else "—")
    st.caption("Summed **at each timestamp** (never a sum of medians). A sum below 1 is explained by missing "
               "candidates (below Gamma's volume floor); a sum meaningfully above 1 with all candidates present "
               "would be a finding. Note: v1 does not store Gamma's full listed-candidate count, only what we captured.")


# ================================================================ AUDIT
def audit():
    st.header("Audit — is the data sound?")
    rec = recon_df()
    cols = st.columns(2)
    for c, (_, r) in zip(cols, rec.iterrows()):
        c.metric(r["identity"], f"{r['table_rows']:,}", "MATCH" if r["match"] else "MISMATCH",
                 delta_color=("normal" if r["match"] else "inverse"))
    st.caption("Reconciliation: catalog sums vs table row counts. If either stops matching, something drifted.")
    st.divider()

    st.subheader("Coverage calendar (active / quiet / gap)")
    cov = coverage_df()
    uni_c = st.radio("universe", ["politics_negrisk", "esports"], horizontal=True, key="covu")
    cc = cov[cov.universe == uni_c].copy()
    # 24-hour strip per date: pc>0 active(2), book-only quiet(1), no-book gap(0)
    import datetime as _dt
    grid = []
    for _, row in cc.iterrows():
        pcs = set(row.pc_hours); bks = set(row.book_hours)
        strip = [(2 if f"{h:02d}" in pcs else 1 if f"{h:02d}" in bks else 0) for h in range(24)]
        grid.append(strip)
    if grid:
        gdf = pd.DataFrame(grid, index=cc.date.tolist(), columns=[f"{h:02d}" for h in range(24)])
        fig, ax = plt.subplots(figsize=(11, max(3, len(gdf) * 0.12)))
        ax.imshow(gdf.values, aspect="auto", cmap=matplotlib.colors.ListedColormap([C_LOST, C_MUTE, C_A]),
                  vmin=0, vmax=2, interpolation="nearest")
        ax.set_yticks(range(len(gdf))); ax.set_yticklabels(gdf.index, fontsize=5)
        ax.set_xticks(range(24)); ax.set_xticklabels(gdf.columns, fontsize=6); ax.set_xlabel("hour UTC")
        ax.set_title("blue=active (price_change)  ·  grey=quiet (book only)  ·  red=gap (no book)")
        st.pyplot(fig); plt.close(fig)
        st.caption("Red is a true capture gap (the 2026-06-22→23 outage; capture start; reboot tail). "
                   "Grey is a quiet market, NOT a gap — the distinction the data preserves.")
    st.divider()

    st.subheader("Distributions")
    d = cat_all()
    def hist(series, bins, lo=None, hi=None):
        s = series.dropna().astype(float)
        if hi is not None:
            s = s.clip(upper=hi)
        counts, edges = np.histogram(s, bins=bins)
        idx = [f"{edges[i]:.2f}" for i in range(len(counts))]
        return pd.DataFrame({"count": counts}, index=idx)
    c1, c2, c3, c4 = st.columns(4)
    c1.bar_chart(hist(d.median_spread_cents, 25, hi=50)); c1.caption("median spread ¢ (≤50)")
    c2.bar_chart(hist(d[d.n_trades > 0].n_trades, 25, hi=500)); c2.caption("trades/token (≤500)")
    c3.bar_chart(hist(d.n_days, 20)); c3.caption("days alive")
    c4.bar_chart(hist(d.median_mid, 25)); c4.caption("median mid ($)")
    st.divider()

    st.subheader("Outliers — click any row's path into Explore via search")
    tabs = st.tabs(["widest spreads", "zero-trade", "40 empty", "14 inverted", "70 unresolved", "top trade counts"])
    def show(df, cols):
        st.dataframe(mono(df[cols].reset_index(drop=True)), width="stretch", height=280)
    with tabs[0]:
        show(d[d.n_l1_events > 100].sort_values("median_spread_cents", ascending=False).head(50),
             ["universe", "question", "outcome_label", "median_spread_cents", "n_trades", "path"])
    with tabs[1]:
        show(d[(d.n_trades == 0)].head(200), ["universe", "question", "outcome_label", "n_l1_events", "path"])
    with tabs[2]:
        show(d[(d.n_l1_events == 0) & (d.n_trades == 0)], ["universe", "question", "outcome_label", "identity_status", "path"])
    with tabs[3]:
        show(d[d.check4_status == "inverted"], ["universe", "question", "outcome_label", "resolved_outcome", "last_mid", "path"])
    with tabs[4]:
        show(d[d.identity_status == "unresolved"], ["universe", "condition_id", "n_trades", "n_l1_events", "path"])
    with tabs[5]:
        show(d.sort_values("n_trades", ascending=False).head(50), ["universe", "question", "outcome_label", "n_trades", "median_mid", "path"])
    st.divider()

    st.subheader("Activity by hour-of-day & weekday (when is there flow?)")
    ca, cb = st.columns(2)
    for c, u in ((ca, "politics_negrisk"), (cb, "esports")):
        a = activity_df(u)
        piv = a.pivot_table(index="weekday", columns="hour_of_day", values="n_trades", aggfunc="sum").fillna(0)
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.imshow(piv.values, aspect="auto", cmap="magma")
        ax.set_title(f"{u} — trades by weekday×hour", fontsize=8)
        ax.set_xlabel("hour UTC"); ax.set_ylabel("weekday (0=Sun)")
        c.pyplot(fig); plt.close(fig)
    st.caption("Esports and politics flow look nothing alike — that contrast is itself a finding.")
    st.divider()

    st.subheader("The stale-book cohort")
    stale = d[(d.universe == "esports") & (d.check4_status == "near_half")]
    h = stale["hours_from_last_seen_to_close"]
    within = int((h.abs() <= 1).sum()); early = int((h.abs() > 1).sum()); noclose = int(h.isna().sum())
    pol_nh = int(((d.universe == "politics_negrisk") & (d.check4_status == "near_half")).sum())
    s = st.columns(4)
    s[0].metric("esports near_half", f"{len(stale):,}")
    s[1].metric("last obs within ±1h of settle", f"{within:,}")
    s[2].metric("stopped early (>1h)", f"{early:,}")
    s[3].metric("no closed_time", f"{noclose:,}")
    st.caption(f"esports near_half = {len(stale):,} (politics = {pol_nh}; the 21,522 in an earlier draft was BOTH "
               f"universes combined — corrected). Of the esports ones, {within:,} were watched to within ±1h of "
               "settlement yet still quoting ~0.5 on a decided outcome — a genuine stale-book phenomenon, not "
               "'stopped watching'. Artefact, or money left on the table? Sample below — open one in Explore.")
    show(stale.sort_values("n_trades", ascending=False).head(50),
         ["question", "outcome_label", "last_mid", "n_trades", "hours_from_last_seen_to_close", "path"])


# ---------------------------------------------------------------- main
if st.session_state.cid is None:
    st.title("Epsilon · Polymarket research terminal")
    st.markdown("**Search a market** in the sidebar (try `fed july`), or pick one of the busiest. "
                "Then switch between **Explore** and **Audit**.")
    if mode == "Audit":
        audit()
else:
    header(st.session_state.cid)
    if mode == "Explore":
        explore(st.session_state.cid)
    else:
        audit()
