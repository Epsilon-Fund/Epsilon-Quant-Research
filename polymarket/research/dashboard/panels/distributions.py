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
    # Without this radio the panel plotted all 30,772 tokens under a universe-agnostic heading —
    # and esports is 91% of them, so every "dataset-wide" shape here was an esports shape and
    # politics was invisible (audit 2026-09, 01_digest.md §8.5). Same control coverage.py has.
    uni = st.radio("universe", ["politics_negrisk", "esports"], horizontal=True, key="dist_uni")
    d = D.cat_all()
    d = d[d.universe == uni]
    if d.empty:
        st.info(f"No tokens for {uni}."); return
    c1, c2, c3, c4 = st.columns(4)
    c1.plotly_chart(_hist(d.median_spread_cents, "median spread ¢ (≤50)", hi=50), width="stretch", theme=None)
    c2.plotly_chart(_hist(d[d.n_trades > 0].n_trades, "trades/token (≤500)", hi=500), width="stretch", theme=None)
    c3.plotly_chart(_hist(d.n_days, "days alive"), width="stretch", theme=None)
    c4.plotly_chart(_hist(d.median_mid, "median mid ($)"), width="stretch", theme=None)
    st.caption(f"{len(d):,} {uni} tokens. The first two charts are **clipped** (spread at 50¢, trades "
               "at 500), so their rightmost bar is a pile-up of everything beyond the clip, not a real "
               "bin. Trades/token counts only tokens with ≥1 trade.")
