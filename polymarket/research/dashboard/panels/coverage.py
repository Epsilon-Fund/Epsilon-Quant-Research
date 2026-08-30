"""Audit · Coverage calendar — active / quiet / gap, per universe × date."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from ._base import panel, Ctx
from . import _data as D


@panel("Coverage calendar", section="audit", order=20)
def render(ctx: Ctx):
    st.subheader("Coverage calendar (active / quiet / gap)")
    cov = D.coverage_df()
    uni = st.radio("universe", ["politics_negrisk", "esports"], horizontal=True, key="cov_uni")
    cc = cov[cov.universe == uni].reset_index(drop=True)
    if cc.empty:
        st.info("No coverage rows."); return
    grid = []
    for _, row in cc.iterrows():
        pcs, bks = set(row.pc_hours), set(row.book_hours)
        grid.append([(2 if f"{h:02d}" in pcs else 1 if f"{h:02d}" in bks else 0) for h in range(24)])
    z = np.array(grid)
    fig = go.Figure(go.Heatmap(z=z, x=[f"{h:02d}" for h in range(24)], y=cc.date.tolist(),
                    colorscale=[[0, D.C_LOST], [0.5, D.C_MUTE], [1, D.C_A]], zmin=0, zmax=2,
                    showscale=False, hovertemplate="%{y} %{x}h<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=8))
    fig.update_xaxes(title="hour UTC", side="top")
    st.plotly_chart(D.style(fig, max(300, len(cc) * 16)), width="stretch", theme=None)
    st.caption("blue = active (price_change) · grey = quiet (book only, NOT a gap) · red = true gap (no book). "
               "Red is the 2026-06-22→23 outage, capture start (06-19), and the reboot tail (08-21).")
