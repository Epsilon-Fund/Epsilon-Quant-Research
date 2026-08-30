"""Audit · Outliers — tables of the tails, every row carrying its path (open it via search)."""
from __future__ import annotations
import streamlit as st
from ._base import panel, Ctx
from . import _data as D


def _show(df, cols):
    st.dataframe(D.mono(df[cols].reset_index(drop=True)), width="stretch", height=280)


@panel("Outliers", section="audit", order=40)
def render(ctx: Ctx):
    st.subheader("Outliers — each row's `path` opens it via the search box")
    d = D.cat_all()
    tabs = st.tabs(["widest spreads", "zero-trade", "fully empty", "inverted", "unresolved", "top trade counts"])
    with tabs[0]:
        _show(d[d.n_l1_events > 100].sort_values("median_spread_cents", ascending=False).head(50),
              ["universe", "question", "outcome_label", "median_spread_cents", "n_trades", "path"])
    with tabs[1]:
        _show(d[d.n_trades == 0].head(200), ["universe", "question", "outcome_label", "n_l1_events", "path"])
    with tabs[2]:
        _show(d[(d.n_l1_events == 0) & (d.n_trades == 0)], ["universe", "question", "outcome_label", "identity_status", "path"])
    with tabs[3]:
        _show(d[d.check4_status == "inverted"], ["universe", "question", "outcome_label", "resolved_outcome", "last_mid", "path"])
    with tabs[4]:
        _show(d[d.identity_status == "unresolved"], ["universe", "condition_id", "n_trades", "n_l1_events", "path"])
    with tabs[5]:
        _show(d.sort_values("n_trades", ascending=False).head(50), ["universe", "question", "outcome_label", "n_trades", "median_mid", "path"])
