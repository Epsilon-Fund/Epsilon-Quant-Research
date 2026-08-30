"""Explore · Market — price+touch band, trade prints, volume, order-flow imbalance, spread,
realised vol. One stacked figure on a shared, linked time axis."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ._base import panel, Ctx
from . import _data as D


def _fig(l1, pair, trades, labels, scale, rv_win):
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.42, 0.14, 0.16, 0.14, 0.14],
                        subplot_titles=("price · bid–ask band · trades", "volume (buy↑ / sell↓)",
                                        "order-flow imbalance & cumulative signed", "spread",
                                        "realised vol (mid returns)"))
    logit = (scale == "logit")
    yv = (lambda p: D.logit(p)) if logit else (lambda p: np.asarray(p, dtype=float) * 100)
    n = len(l1); Sg = D.sg(n)
    fig.add_trace(Sg(x=l1.ts, y=yv(l1.best_ask), line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(Sg(x=l1.ts, y=yv(l1.best_bid), name="bid–ask band", fill="tonexty",
                     fillcolor="rgba(76,155,232,0.12)", line=dict(width=0), hoverinfo="skip"), 1, 1)
    fig.add_trace(Sg(x=l1.ts, y=yv(l1.mid), name=f"{labels.get(0,'A')} mid", line=dict(color=D.C_A, width=1.3)), 1, 1)
    if pair is not None and not pair.empty and 1 in pair.columns:
        fig.add_trace(D.sg(len(pair))(x=pair.index, y=yv(pair[1]), name=f"{labels.get(1,'B')} mid",
                                      line=dict(color=D.C_B, width=1.0)), 1, 1)
    if trades is not None and not trades.empty:
        smax = max(float(trades["size"].max()), 1.0)
        for side, col in (("BUY", D.C_WON), ("SELL", D.C_LOST)):
            t = trades[trades.side == side]
            if t.empty:
                continue
            fig.add_trace(D.sg(len(t))(x=t.ts, y=yv(t.price), mode="markers", name=side,
                          marker=dict(color=col, size=5 + 16 * (t["size"] / smax), opacity=0.55, line=dict(width=0)),
                          customdata=np.c_[t["size"], t.price * 100],
                          hovertemplate=side + " %{customdata[0]:.0f}@%{customdata[1]:.1f}¢<extra></extra>"), 1, 1)
    if logit:
        fig.update_yaxes(tickvals=D.logit(D._LOGIT_TICKS), ticktext=[f"{p:g}" for p in D._LOGIT_TICKS],
                         row=1, col=1, title="prob (logit)")
    else:
        fig.update_yaxes(title="¢", row=1, col=1)
    if trades is not None and not trades.empty:
        span_s = (trades.ts.max() - trades.ts.min()).total_seconds() or 1
        b = D.bucket(span_s)
        t = trades.set_index("ts")
        buy = t[t.side == "BUY"]["size"].resample(b).sum()
        sell = t[t.side == "SELL"]["size"].resample(b).sum()
        idx = buy.index.union(sell.index)
        buy = buy.reindex(idx, fill_value=0); sell = sell.reindex(idx, fill_value=0)
        fig.add_trace(go.Bar(x=idx, y=buy, marker_color=D.C_WON, showlegend=False), 2, 1)
        fig.add_trace(go.Bar(x=idx, y=-sell, marker_color=D.C_LOST, showlegend=False), 2, 1)
        fig.add_trace(go.Bar(x=idx, y=(buy - sell), marker_color=D.C_MUTE, showlegend=False), 3, 1)
        cum = (t["size"] * np.where(t.side == "BUY", 1.0, -1.0)).cumsum()
        fig.add_trace(D.sg(len(cum))(x=cum.index, y=cum.values, name="cum signed", line=dict(color="#c98bdb", width=1.2)), 3, 1)
        fig.update_yaxes(title="contracts", row=2, col=1); fig.update_yaxes(title="net / cum", row=3, col=1)
        fig.update_layout(barmode="relative")
    fig.add_trace(D.sg(n)(x=l1.ts, y=l1.spread_c, line=dict(color=D.C_MUTE, width=1, shape="hv"), showlegend=False), 4, 1)
    fig.update_yaxes(title="¢", row=4, col=1)
    mid1 = l1.set_index("ts")["mid"].resample("1min").last().ffill()
    rv = mid1.pct_change().rolling(int(rv_win), min_periods=2).std() * 100
    fig.add_trace(D.sg(len(rv))(x=rv.index, y=rv.values, line=dict(color="#e8c14c", width=1), showlegend=False), 5, 1)
    fig.update_yaxes(title="%", row=5, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), rangeselector=D.RANGEBUTTONS, row=5, col=1)
    return D.style(fig, 820)


@panel("Market", section="explore", needs="market", order=10)
def render(ctx: Ctx):
    a = ctx.a_row
    labels = {int(r.outcome_index): r.outcome_label for r in ctx.market.itertuples()}
    l1 = D.l1_df(a.asset_id)
    if l1.empty:
        st.info("No L1 events for this market's side A — a quiet or unobserved token. "
                "Nothing to plot; this is a real state, not an error."); return
    tr = D.trades_df(a.asset_id)
    try:
        pair, _ = D.pair_df(ctx.cid)
    except Exception:
        pair = None
    l1 = l1[(l1.ts >= ctx.start) & (l1.ts <= ctx.end)]
    tr = tr[(tr.ts >= ctx.start) & (tr.ts <= ctx.end)] if not tr.empty else tr
    if pair is not None and not pair.empty:
        pair = pair[(pair.index >= ctx.start) & (pair.index <= ctx.end)]
    s = st.columns(5)
    s[0].metric("median spread", f"{l1.spread_c.median():.1f}¢" if len(l1) else "—")
    s[1].metric("mid range", f"{l1.mid.min()*100:.1f}–{l1.mid.max()*100:.1f}¢" if len(l1) else "—")
    s[2].metric("trades", f"{len(tr):,}")
    s[3].metric("buy vol", f"{tr[tr.side=='BUY']['size'].sum():,.0f}" if len(tr) else "0")
    s[4].metric("sell vol", f"{tr[tr.side=='SELL']['size'].sum():,.0f}" if len(tr) else "0")
    if l1.empty:
        st.info("No L1 events in the selected window."); return
    st.plotly_chart(_fig(l1, pair, tr, labels, ctx.scale, ctx.rv_win), width="stretch", theme=None)
    st.caption("Zoom/pan any subplot — x-axis is shared. Band = tradeable bid–ask spread; the imbalance "
               "panel is the 'are we being run over' view: one-sided net volume with price following it is "
               "adverse selection. Logit scale makes 0.99 vs 0.999 visible.")
