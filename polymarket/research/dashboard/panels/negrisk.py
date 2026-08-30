"""Explore · NegRisk — every candidate's YES mid and the instantaneous YES-sum vs 1.0."""
from __future__ import annotations
from plotly.subplots import make_subplots
import streamlit as st

from ._base import panel, Ctx
from . import _data as D


@panel("NegRisk", section="explore", needs="market", order=30)
def render(ctx: Ctx):
    if not ctx.neg_risk:
        st.info("This market is not NegRisk — the sum-to-1 view applies to NegRisk events (politics)."); return
    n_markets = D.cat_all().query("event_slug == @ctx.event_slug").condition_id.nunique()
    if n_markets < 2:
        st.info(f"Only {n_markets} market captured for this event — nothing to sum."); return
    try:
        wide, meta = D.event_wide(ctx.event_slug)
        ns, ncap = D.negrisk_df(ctx.event_slug)
    except Exception as e:
        st.warning(f"event unavailable: {e}"); return
    yes_cols = [a for a in wide.columns if meta.get(a, {}).get("outcome") == "YES"] \
        or [a for a in wide.columns if meta.get(a, {}).get("outcome_index") == 0]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.04,
                        subplot_titles=("every candidate's YES mid", "YES sum (instantaneous, common timestamp)"))
    for a in yes_cols:
        fig.add_trace(D.sg(len(wide))(x=wide.index, y=wide[a] * 100, line=dict(width=0.7), opacity=0.5,
                      showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(D.sg(len(ns))(x=ns.index, y=ns.yes_sum * 100, line=dict(color="#e8c14c", width=1.6), name="YES sum"), 2, 1)
    fig.add_hline(y=100, line=dict(color=D.C_LOST, width=1, dash="dash"), row=2, col=1)
    fig.update_yaxes(title="¢", row=1, col=1); fig.update_yaxes(title="sum ¢", row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=2, col=1)
    st.plotly_chart(D.style(fig, 560), width="stretch", theme=None)
    s = st.columns(3)
    s[0].metric("candidates captured", f"{ncap}")
    s[1].metric("YES sum · median", f"{ns.yes_sum.median():.3f}")
    s[2].metric("YES sum · latest", f"{ns.yes_sum.dropna().iloc[-1]:.3f}" if ns.yes_sum.notna().any() else "—")
    st.caption("Summed **at each timestamp** (never a sum of medians). Below 1 is explained by missing "
               "candidates; meaningfully above 1 with all present is a finding. Caveat: the sum uses "
               "mid=(bid+ask)/2 and forward-fills quiet candidates, so it can overstate for many-candidate "
               "range events (e.g. Elon-tweet ranges) — read it as a diagnostic, not a clean arb signal. "
               "v1 does not store Gamma's full listed-candidate count, only what we captured.")
