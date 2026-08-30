"""Epsilon Polymarket research terminal — thin shell.

Sidebar, search, header, the audit button, and routing. Panels live in `panels/` and register
themselves; this file names none of them, so adding a panel is adding one file there. Everything
reads through `epsilon_data`. Run:  streamlit run dashboard/app.py   (from polymarket/research/)
"""
from __future__ import annotations
import sys
import pathlib

_root = next((p for p in pathlib.Path(__file__).resolve().parents if (p / "epsilon_data").is_dir()), None)
if _root and str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st
import epsilon_data as ed

from panels import panels_for, Ctx
from panels import _data as D

st.set_page_config(page_title="Epsilon · Polymarket research", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>table,.stDataFrame,code{font-family:'DejaVu Sans Mono',monospace!important;}"
            "[data-testid='stMetricValue']{font-family:monospace;}</style>", unsafe_allow_html=True)

cat = D.cat_all()
if "cid" not in st.session_state:
    st.session_state.cid = None

def select(cid):
    st.session_state.cid = str(cid); st.session_state.pop("audit", None)

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Epsilon · Polymarket")
st.sidebar.caption(f"{len(cat):,} tokens · {cat.condition_id.nunique():,} markets · 2 universes")
mode = st.sidebar.radio("Mode", ["Explore", "Audit"], horizontal=True)
st.sidebar.markdown("### Find a market")
query = st.sidebar.text_input("Search (question or slug)", placeholder="e.g. fed july", key="q")
if query:
    res = D.search_df(query)
    if res.empty:
        st.sidebar.info("No match.")
    else:
        res_m = res.drop_duplicates("condition_id").head(40)
        opts = {f"{r.question[:60]} · {r.universe[:3]} · {int(r.n_trades)}t": r.condition_id for r in res_m.itertuples()}
        pick = st.sidebar.radio(f"{len(opts)} match(es)", list(opts), index=0, key="pick")
        if st.sidebar.button("Open →", width="stretch"):
            select(opts[pick])
else:
    st.sidebar.caption("Or browse the busiest:")
    for r in cat[cat.n_trades > 0].sort_values("n_trades", ascending=False).drop_duplicates("condition_id").head(15).itertuples():
        if st.sidebar.button(f"{r.question[:42]} · {int(r.n_trades)}t", key=f"b{r.condition_id}", width="stretch"):
            select(r.condition_id)
with st.sidebar.expander("Browse the tree"):
    u = st.selectbox("Universe", ["politics_negrisk", "esports"])
    evs = D.events_df(u); evs = evs[evs.n_markets > 0]
    ev = st.selectbox("Event", evs.event_slug.tolist()[:500])
    em = cat[cat.event_slug == ev].drop_duplicates("condition_id")
    mp = st.selectbox("Market", em.question.tolist())
    if st.button("Open market"):
        select(em[em.question == mp].condition_id.iloc[0])

# ---------------------------------------------------------------- header + audit button
def market_rows(cid):
    return cat[cat.condition_id == str(cid)].sort_values("outcome_index")

def render_header(m):
    r0 = m.iloc[0]
    st.markdown(f"### {r0.event_title or '(no event)'}")
    st.markdown(f"**{r0.question}** — {r0.universe} · neg_risk={bool(r0.neg_risk)} · "
                f"{'RESOLVED' if r0.closed else 'open'}")
    cols = st.columns(len(m) + 1)
    for c, (_, row) in zip(cols, m.iterrows()):
        colr = D.C_A if row.outcome_index == 0 else D.C_B
        won = (row.resolved_outcome == row.outcome_label) if pd.notna(row.resolved_outcome) else None
        tag = " 🏆" if won else (" ✖" if won is False else "")
        c.markdown(f"<span style='color:{colr}'>■</span> **{row.outcome_label}**{tag}", unsafe_allow_html=True)
        c.caption(f"{row.path}")
        c.caption(f"tr {int(row.n_trades):,} · l1 {int(row.n_l1_events):,} · sp {row.median_spread_cents:.1f}¢ · mid {row.median_mid:.3f}")
    if cols[-1].button("🔍 Audit this market", width="stretch"):
        st.session_state["audit"] = str(r0.condition_id)
    ch = {k: r0[k] for k in ["check1_roundtrip", "check2_pairing", "check3_negrisk", "check4_status", "check5_indep"]}
    st.caption("checks · " + " · ".join(f"{k.replace('check','c').split('_')[0]}={v}" for k, v in ch.items())
               + ("  ·  EXCLUDED" if bool(r0.get("excluded")) else ""))

def render_audit(cid):
    r = D.audit_result(cid)
    icon = {"looks fine": "🟢", "worth a look": "🟡", "recommend excluding": "🔴"}.get(r.verdict, "•")
    with st.container(border=True):
        st.markdown(f"#### {icon} Audit — **{r.verdict}**")
        st.caption(r.reason)
        st.dataframe(pd.DataFrame([{"check": c.name, "level": c.level, "detail": c.detail} for c in r.checks]),
                     width="stretch", hide_index=True)
        if r.exclusion_lines:
            st.markdown(f"**Recommended:** {r.recommend_reason}")
            st.code("\n".join(r.exclusion_lines), language="text")
            st.caption("Human-decided: nothing is written until you click below. The market keeps showing, marked excluded.")
            if r.recommend_scope == "market" and st.button("Apply market-scope exclusion to exclusions.csv"):
                m = market_rows(cid)
                for a in m.asset_id:
                    ed.write_exclusion(a, "market", r.reason, "operator")
                D.cat_all.clear()
                st.success("Appended. Remove the lines from exclusions.csv to restore. Re-run to refresh.")
        else:
            st.caption("No exclusion recommended.")
        if st.button("Close audit"):
            st.session_state.pop("audit", None)

# ---------------------------------------------------------------- routing
if st.session_state.cid is None:
    st.title("Epsilon · Polymarket research terminal")
    st.markdown("**Search a market** in the sidebar (try `fed july`) or pick a busy one. Then switch **Explore** / **Audit**.")
    if mode == "Audit":
        for i, p in enumerate(panels_for("audit", has_market=False)):
            if i: st.divider()
            p.render(Ctx())
else:
    m = market_rows(st.session_state.cid)
    if m.empty:
        st.warning("Market not found."); st.stop()
    render_header(m)
    if st.session_state.get("audit") == str(st.session_state.cid):
        render_audit(st.session_state.cid)
    st.divider()
    if mode == "Explore":
        a = m.iloc[0]
        l1 = D.l1_df(a.asset_id)
        cc = st.columns([1.4, 1, 1, 3])
        win = cc[0].radio("window", ["all", "1w", "1d", "6h", "1h"], horizontal=True, index=0)
        scale = cc[1].radio("scale", ["linear", "logit"], horizontal=True)
        rvwin = cc[2].selectbox("vol win (min)", [1, 5, 15], index=1)
        end = l1.ts.max() if not l1.empty else pd.Timestamp.utcnow()
        delta = D.window_delta(win)
        start = (end - delta) if delta is not None else (l1.ts.min() if not l1.empty else end)
        ctx = Ctx(cid=str(st.session_state.cid), market=m, a_row=a, universe=a.universe,
                  neg_risk=bool(a.neg_risk), closed=bool(a.closed), event_slug=a.event_slug,
                  start=start, end=end, window=win, scale=scale, rv_win=rvwin)
        st.caption(f"**Window:** {win} · {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} UTC (stats & panels cover this window)")
        ps = panels_for("explore", has_market=True)
        for tab, p in zip(st.tabs([p.name for p in ps]), ps):
            with tab:
                p.render(ctx)
    else:
        for i, p in enumerate(panels_for("audit", has_market=False)):
            if i: st.divider()
            p.render(Ctx())
