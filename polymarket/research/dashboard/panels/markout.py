"""Explore · Markout — adverse selection from the liquidity-provider (maker) view."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ._base import panel, Ctx
from . import _data as D


@panel("Markout (adverse selection)", section="explore", needs="market", order=20)
def render(ctx: Ctx):
    st.caption("`trades.side` is the **taker** side (verified: BUY lifts the ask, SELL hits the bid). "
               "Markout is shown from the **maker** view: **negative = the resting quote was adversely "
               "selected** (price moved against the maker). Positive = benign flow.")
    mo = D.markout_df(ctx.a_row.asset_id)
    mo = mo[(mo.ts >= ctx.start) & (mo.ts <= ctx.end)] if not mo.empty else mo
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
        for side, colr in (("BUY", D.C_WON), ("SELL", D.C_LOST)):
            x = mo[mo.side == side][col] * 100
            if len(x):
                fig.add_trace(go.Histogram(x=x.clip(-5, 5), name=side, marker_color=colr, opacity=0.6, nbinsx=60))
        fig.add_vline(x=0, line=dict(color="#fff", width=1, dash="dot"))
        fig.update_layout(barmode="overlay", title=f"markout distribution @ {h}s (¢)")
        st.plotly_chart(D.style(fig, 320), width="stretch", theme=None)
    with c2:
        fig = go.Figure()
        fig.add_trace(D.sg(len(mo))(x=mo.ts, y=mo[col] * 100, mode="markers",
                      marker=dict(color=np.where(mo.side == "BUY", D.C_WON, D.C_LOST), size=4, opacity=0.5), name="per-trade"))
        roll = (mo.set_index("ts")[col] * 100).rolling(50, min_periods=5).mean()
        fig.add_trace(D.sg(len(roll))(x=roll.index, y=roll.values, line=dict(color="#e8c14c", width=1.5), name="rolling mean(50)"))
        fig.add_hline(y=0, line=dict(color="#fff", width=1, dash="dot"))
        fig.update_layout(title=f"markout over time @ {h}s (¢) — spikes below 0 = toxic flow")
        st.plotly_chart(D.style(fig, 320), width="stretch", theme=None)
    st.caption(f"{(mo[col] < 0).mean():.0%} of fills had negative maker markout at {h}s in this window. "
               "Consistently negative (especially one side) = flow you'd have been run over by.")
