"""Audit · Reconciliation — the identities that must hold, shown holding."""
from __future__ import annotations
import streamlit as st
from ._base import panel, Ctx
from . import _data as D


@panel("Reconciliation", section="audit", order=10)
def render(ctx: Ctx):
    st.subheader("Reconciliation")
    rec = D.recon_df()
    cols = st.columns(len(rec))
    for c, (_, r) in zip(cols, rec.iterrows()):
        c.metric(r["identity"], f"{r['table_rows']:,}", "MATCH" if r["match"] else "MISMATCH",
                 delta_color=("normal" if r["match"] else "inverse"))
    st.caption("catalog sums vs table row counts. If either stops matching, something downstream drifted.")
