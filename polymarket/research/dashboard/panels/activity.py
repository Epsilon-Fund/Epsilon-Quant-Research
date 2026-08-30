"""Audit · Activity — trade flow by UTC hour-of-day × weekday, per universe."""
from __future__ import annotations
import plotly.graph_objects as go
import streamlit as st
from ._base import panel, Ctx
from . import _data as D


@panel("Activity by time", section="audit", order=50)
def render(ctx: Ctx):
    st.subheader("Activity by hour-of-day & weekday (when is there flow?)")
    ca, cb = st.columns(2)
    for c, u in ((ca, "politics_negrisk"), (cb, "esports")):
        a = D.activity_df(u)
        piv = a.pivot_table(index="weekday", columns="hour_of_day", values="n_trades", aggfunc="sum").fillna(0)
        fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(), colorscale="Magma",
                        hovertemplate="wd %{y} h%{x}: %{z:,.0f}<extra></extra>"))
        fig.update_layout(title=f"{u} — trades by weekday×hour")
        fig.update_xaxes(title="hour UTC"); fig.update_yaxes(title="weekday (0=Sun)")
        c.plotly_chart(D.style(fig, 300), width="stretch", theme=None)
    st.caption("Esports and politics flow look nothing alike — that contrast is itself a finding.")
