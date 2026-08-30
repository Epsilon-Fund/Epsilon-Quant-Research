"""Shared, cached loaders, the colour language, and chart helpers — used by every panel and the
shell. All data access goes through epsilon_data (rule 2). No panel reads a raw parquet path."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import epsilon_data as ed

# ---- fixed colour language (colourblind-safe; meaning only) ----
C_A = "#4c9be8"      # YES / first outcome / side A
C_B = "#e8983c"      # NO / second outcome / side B
C_WON = "#4cae72"    # won / up / BUY
C_LOST = "#d1495b"   # lost / down / SELL
C_MUTE = "#8a93a3"
PLOT_BG = "#0e1117"
_LOGIT_TICKS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
RANGEBUTTONS = dict(buttons=[
    dict(count=1, label="1h", step="hour", stepmode="backward"),
    dict(count=6, label="6h", step="hour", stepmode="backward"),
    dict(count=1, label="1d", step="day", stepmode="backward"),
    dict(count=7, label="1w", step="day", stepmode="backward"),
    dict(step="all", label="all"),
], bgcolor="#161a23", activecolor="#4c9be8", font=dict(size=10))


def style(fig, height):
    fig.update_layout(template="plotly_dark", paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      height=height, margin=dict(l=50, r=20, t=30, b=20), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", font=dict(size=10)),
                      font=dict(family="monospace", size=11))
    fig.update_xaxes(gridcolor="#232936", showspikes=True, spikemode="across", spikethickness=1)
    fig.update_yaxes(gridcolor="#232936")
    return fig


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sg(n):
    return go.Scattergl if n > 20000 else go.Scatter


def bucket(span_s):
    return "1min" if span_s <= 2*3600 else "5min" if span_s <= 26*3600 else "30min" if span_s <= 8*86400 else "1h"


def window_delta(choice):
    return {"1h": pd.Timedelta("1h"), "6h": pd.Timedelta("6h"), "1d": pd.Timedelta("1D"),
            "1w": pd.Timedelta("7D")}.get(choice)


def mono(df):
    return df.style.set_properties(**{"font-family": "monospace", "text-align": "right"})


# ---- cached loaders (thin wrappers on epsilon_data) ----
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
def l1_df(asset_id):
    return ed.load_l1(asset_id)

@st.cache_data(show_spinner=True)
def trades_df(asset_id):
    return ed.load_trades(asset_id)

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

@st.cache_data(show_spinner=True)
def audit_result(ref):
    return ed.audit_market(ref)
