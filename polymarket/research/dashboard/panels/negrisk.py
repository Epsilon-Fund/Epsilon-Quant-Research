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
                        subplot_titles=("every candidate's YES mid",
                                        "YES sum of LAST-KNOWN mids (ffilled — NOT instantaneous)"))
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
    st.caption("⚠️ **This is a stale composite, not an instantaneous sum.** Candidates are aligned on a "
               "1-second index and **forward-filled with no limit**, so each leg contributes its last "
               "known mid: on a 25-leg Elon-tweet-range event only ~1.3 legs actually update per second "
               "and the median summed quote is **~20 hours old** (audit 2026-09). Below 1 can be missing "
               "candidates; above 1 is at least as likely to be the ffill keeping dead candidates alive. "
               "The sum uses mid=(bid+ask)/2. Read it as a diagnostic of shape and of the right tail — "
               "never as a live arb signal. v1 does not store Gamma's full listed-candidate count, only "
               "what we captured.")
