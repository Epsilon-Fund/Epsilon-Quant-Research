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
def explore(cid):
    m = market_rows(cid)
    labels = {int(r.outcome_index): r.outcome_label for r in m.itertuples()}
    try:
        pair, _ = pair_df(cid)
    except Exception as e:
        st.warning(f"pair unavailable: {e}"); pair = None

    st.subheader("Market — both sides, mid & spread, trades")
    a_row = m.iloc[0]
    l1 = l1_df(a_row.asset_id)
    if l1.empty:
        st.info("No L1 events for side A in this market (a quiet or unobserved token).")
    else:
        tr = trades_df(a_row.asset_id)
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, height_ratios=[3, 1])
        if pair is not None and not pair.empty:
            for idx in sorted(pair.columns):
                ax0.plot(pair.index, pair[idx] * 100, lw=0.9, color=(C_A if idx == 0 else C_B),
                         label=f"{labels.get(idx, idx)}")
        else:
            ax0.plot(l1.ts, l1.mid * 100, lw=0.9, color=C_A, label=labels.get(0, "A"))
        if not tr.empty:
            buy = tr[tr.side == "BUY"]; sell = tr[tr.side == "SELL"]
            smax = max(tr["size"].max(), 1)
            ax0.scatter(buy.ts, buy.price * 100, s=8 + 60 * (buy["size"] / smax), color=C_WON, alpha=0.5, label="BUY", edgecolors="none")
            ax0.scatter(sell.ts, sell.price * 100, s=8 + 60 * (sell["size"] / smax), color=C_LOST, alpha=0.5, label="SELL", edgecolors="none")
        ax0.set_ylabel("price (¢)"); ax0.legend(fontsize=7, loc="upper left"); ax0.set_ylim(-3, 103)
        ax1.plot(l1.ts, l1.spread_c, lw=0.6, color=C_MUTE); ax1.set_ylabel("spread (¢)"); ax1.set_xlabel("UTC")
        st.pyplot(fig); plt.close(fig)
        st.caption(f"side A: {len(l1):,} L1 events, {len(tr):,} trades. "
                   "Blue/orange = the two outcomes' mid; green/red marks = BUY/SELL prints sized by volume.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Spread through the market's life")
        if not l1.empty:
            g = l1.copy()
            t0 = g.ts.min(); span = (g.ts.max() - t0).total_seconds() or 1.0
            g["frac"] = (g.ts - t0).dt.total_seconds() / span
            binned = g.groupby(pd.cut(g.frac, 20), observed=True)["spread_c"].median()
            st.bar_chart(pd.DataFrame({"median spread ¢": binned.values}, index=[f"{int(i.right*100)}%" for i in binned.index]))
            st.caption("Median spread across 20 slices of the market's life (0%→100%). Does quoting tighten toward resolution?")
    with c2:
        st.subheader("Resolution path")
        if pair is not None and not pair.empty and bool(a_row.closed):
            st.line_chart((pair * 100).rename(columns=lambda i: labels.get(i, i)))
            st.caption("Both sides' mid over time. If mapping is right, winner → 100¢, loser → 0¢.")
        else:
            st.info("Open market (no settlement path yet)." if not bool(a_row.closed) else "No aligned pair data.")


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
    st.metric("esports tokens settled but book never moved (near_half)", f"{len(stale):,}")
    st.caption("Watched to settlement yet last quote still ~0.5 on a decided outcome. Artefact, or money left "
               "on the table? Sample below — open one in Explore to inspect.")
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
