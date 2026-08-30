"""Audit · Distributions — spread, trades/token, days alive, median mid."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from ._base import panel, Ctx
from . import _data as D


def _hist(series, title, hi=None):
    s = series.dropna().astype(float)
    if hi is not None:
        s = s.clip(upper=hi)
    fig = go.Figure(go.Histogram(x=s, nbinsx=30, marker_color=D.C_A))
    fig.update_layout(title=title, bargap=0.02)
    return D.style(fig, 260)


@panel("Distributions", section="audit", order=30)
def render(ctx: Ctx):
    st.subheader("Distributions (counts, not just shapes)")
    d = D.cat_all()
    c1, c2, c3, c4 = st.columns(4)
    c1.plotly_chart(_hist(d.median_spread_cents, "median spread ¢ (≤50)", hi=50), width="stretch", theme=None)
    c2.plotly_chart(_hist(d[d.n_trades > 0].n_trades, "trades/token (≤500)", hi=500), width="stretch", theme=None)
    c3.plotly_chart(_hist(d.n_days, "days alive"), width="stretch", theme=None)
    c4.plotly_chart(_hist(d.median_mid, "median mid ($)"), width="stretch", theme=None)
