"""Audit · Stale-book cohort — esports tokens settled 1/0 but book never left ~0.5."""
from __future__ import annotations
import streamlit as st
from ._base import panel, Ctx
from . import _data as D


@panel("Stale-book cohort", section="audit", order=60)
def render(ctx: Ctx):
    st.subheader("The stale-book cohort")
    d = D.cat_all()
    stale = d[(d.universe == "esports") & (d.check4_status == "near_half")]
    h = stale["hours_from_last_seen_to_close"]
    within = int((h.abs() <= 1).sum()); early = int((h.abs() > 1).sum()); noclose = int(h.isna().sum())
    pol_nh = int(((d.universe == "politics_negrisk") & (d.check4_status == "near_half")).sum())
    s = st.columns(4)
    s[0].metric("esports near_half", f"{len(stale):,}")
    s[1].metric("last obs within ±1h of settle", f"{within:,}")
    s[2].metric("stopped early (>1h)", f"{early:,}")
    s[3].metric("no closed_time", f"{noclose:,}")
    st.caption(f"esports near_half = {len(stale):,} (politics = {pol_nh}; the '21,522' in an early draft was BOTH "
               f"universes). {within:,} were watched to within ±1h of settlement yet still quoting ~0.5 on a "
               "decided outcome — a genuine stale-book phenomenon, not 'stopped watching'. Open one in Explore.")
    st.dataframe(D.mono(stale.sort_values("n_trades", ascending=False).head(50)[
        ["question", "outcome_label", "last_mid", "n_trades", "hours_from_last_seen_to_close", "path"]].reset_index(drop=True)),
        width="stretch", height=280)
