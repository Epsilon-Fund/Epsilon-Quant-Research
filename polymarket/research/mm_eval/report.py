"""Assemble the standardized validation report — tidy DataFrames + markdown tables.

Consumes the per-market :class:`~mm_eval.runner.MarketEval`s (one arm) and emits:

* ``scorecard_df``     — one row per (token × queue model): the full scorecard, three-way PnL,
  ND-PnL, PnLMAP, fill rate, max inventory, quote uptime, etc.
* ``markout_df``       — one row per (token × queue × horizon): markout-to-fill + adverse-drift
  with CIs, and the adverse-selection rate.
* ``breakeven_df``     — one row per (token × queue × fee-mode) at the primary horizon: half-spread,
  rebate, breakeven adverse selection A\\*, measured adverse selection, net edge CI, clears?.
* ``verdict_df``       — one row per token: the bracketed viable/fragile/dead verdict (both fee
  modes) + the optimistic/pessimistic net-edge bracket + fill-rate bracket.

Every PnL/edge column is reported bracketed by queue model; CIs are carried as ``*_lo/_hi``
columns. ND-PnL / PnLMAP definitions are documented in :mod:`mm_eval.metrics` (operational
definitions — the terms were undefined in the build plan).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mm_eval.ab import ArmResult
from mm_eval.runner import MarketEval, OPTIMISTIC, PESSIMISTIC


def scorecard_df(arm: ArmResult) -> pd.DataFrame:
    rows = []
    for tok, ev in arm.evals.items():
        for qname, run in ev.runs.items():
            s = run.scorecard
            rows.append({
                "universe": ev.spec.universe, "token_id": tok, "market": ev.spec.market,
                "queue": qname, "avg_price": ev.spec.avg_price,
                "median_spread_cents": ev.spec.median_spread * 100,
                "n_trades": ev.spec.n_trades,
                "fills": s.fills, "placed": s.placed, "fill_rate": s.fill_rate,
                "filled_qty": s.filled_qty,
                "realized": s.realized, "unrealized": s.unrealized, "rebates": s.rebates,
                "gross_pnl": s.gross_pnl, "net_ex_rebate": s.net_ex_rebate,
                "net_with_rebate": s.net_with_rebate,
                "nd_pnl_ex_rebate": s.nd_pnl_ex_rebate, "nd_pnl_with_rebate": s.nd_pnl_with_rebate,
                "pnlmap_ex_rebate": s.pnlmap_ex_rebate, "pnlmap_with_rebate": s.pnlmap_with_rebate,
                "mean_abs_inventory": s.mean_abs_inventory, "max_inventory": s.max_inventory,
                "net_position": s.net_position, "profit_ratio": s.profit_ratio,
                "max_drawdown_usd": s.max_drawdown_usd,
                "interval_sharpe": s.interval_sharpe, "sharpe_buckets": s.interval_sharpe_buckets,
                "quote_uptime_two_sided": s.quote_uptime_two_sided,
                "quote_uptime_any": s.quote_uptime_any, "stale_share": s.stale_share,
                "rebate_per_contract": s.rebate_per_contract,
                "l1_both_match_frac": s.l1_both_match_frac,
            })
    return pd.DataFrame(rows)


def markout_df(arm: ArmResult) -> pd.DataFrame:
    rows = []
    for tok, ev in arm.evals.items():
        for qname, run in ev.runs.items():
            for cp in run.markout:
                rows.append({
                    "universe": ev.spec.universe, "token_id": tok, "queue": qname,
                    "horizon_s": cp.horizon_s,
                    "markout_to_fill_cents": cp.markout_to_fill_cents.point,
                    "markout_lo": cp.markout_to_fill_cents.lo, "markout_hi": cp.markout_to_fill_cents.hi,
                    "adverse_drift_cents": cp.adverse_drift_cents.point,
                    "adverse_drift_lo": cp.adverse_drift_cents.lo, "adverse_drift_hi": cp.adverse_drift_cents.hi,
                    "adverse_rate": cp.adverse_rate, "n_fills": cp.n_fills, "n_censored": cp.n_censored,
                })
    return pd.DataFrame(rows)


def breakeven_df(arm: ArmResult) -> pd.DataFrame:
    rows = []
    for tok, ev in arm.evals.items():
        for qname, run in ev.runs.items():
            for be in (run.breakeven_no_rebate, run.breakeven_representative):
                rows.append({
                    "universe": ev.spec.universe, "token_id": tok, "queue": qname,
                    "fee_mode": be.fee_mode, "horizon_s": be.horizon_s,
                    "half_spread_cents": be.half_spread_cents, "rebate_cents": be.rebate_cents,
                    "breakeven_adverse_cents": be.breakeven_adverse_cents,
                    "measured_adverse_cents": be.measured_adverse_cents.point,
                    "measured_adverse_lo": be.measured_adverse_cents.lo,
                    "measured_adverse_hi": be.measured_adverse_cents.hi,
                    "net_edge_cents": be.net_edge_cents.point,
                    "net_edge_lo": be.net_edge_cents.lo, "net_edge_hi": be.net_edge_cents.hi,
                    "clears": be.clears,
                })
    return pd.DataFrame(rows)


def stability_df(arm: ArmResult) -> pd.DataFrame:
    rows = []
    for tok, ev in arm.evals.items():
        for qname, run in ev.runs.items():
            st = run.stability
            rows.append({
                "universe": ev.spec.universe, "token_id": tok, "queue": qname,
                "horizon_s": st.horizon_s, "n_blocks": st.n_blocks,
                "pooled_edge_cents": st.pooled_edge_cents,
                "n_nonempty_blocks": st.n_nonempty_blocks, "sign_stability": st.sign_stability,
                "lobo_min_edge_cents": st.lobo_min_edge_cents,
                "lobo_max_edge_cents": st.lobo_max_edge_cents,
                "lobo_sign_flips": st.lobo_sign_flips, "note": st.concentration_note,
            })
    return pd.DataFrame(rows)


def verdict_df(arm: ArmResult) -> pd.DataFrame:
    rows = []
    for tok, ev in arm.evals.items():
        opt = ev.runs.get(OPTIMISTIC)
        pess = ev.runs.get(PESSIMISTIC)
        if opt is None or pess is None:
            continue
        rows.append({
            "universe": ev.spec.universe, "token_id": tok, "market": ev.spec.market,
            "avg_price": ev.spec.avg_price, "median_spread_cents": ev.spec.median_spread * 100,
            "half_spread_cents": opt.breakeven_no_rebate.half_spread_cents,
            "horizon_s": ev.primary_horizon_s,
            "fill_rate_optimistic": opt.scorecard.fill_rate,
            "fill_rate_pessimistic": pess.scorecard.fill_rate,
            "net_edge_opt_cents": opt.breakeven_no_rebate.net_edge_cents.point,
            "net_edge_opt_lo": opt.breakeven_no_rebate.net_edge_cents.lo,
            "net_edge_pess_cents": pess.breakeven_no_rebate.net_edge_cents.point,
            "net_edge_pess_lo": pess.breakeven_no_rebate.net_edge_cents.lo,
            "adverse_pess_cents": pess.breakeven_no_rebate.measured_adverse_cents.point,
            "verdict_no_rebate": ev.verdict_no_rebate,
            "verdict_representative": ev.verdict_representative,
            # the naive-quoter reality alongside the per-contract precondition: a "VIABLE"
            # spread-vs-adverse read does NOT imply the inventory-blind symmetric quoter profits.
            "naive_net_ex_rebate_usd": pess.scorecard.net_ex_rebate,
            "naive_unrealized_usd": pess.scorecard.unrealized,
            "naive_max_inventory": pess.scorecard.max_inventory,
            "naive_mean_abs_inventory": pess.scorecard.mean_abs_inventory,
            "stability_note": pess.stability.concentration_note,
            "stability_sign": pess.stability.sign_stability,
            "stability_lobo_flip": pess.stability.lobo_sign_flips,
        })
    return pd.DataFrame(rows)


def _fmt_ci(point, lo, hi, unit="¢"):
    if not np.isfinite(lo) or not np.isfinite(hi):
        return f"{point:+.3f}{unit} (CI n/a)"
    return f"{point:+.3f}{unit} [{lo:+.3f}, {hi:+.3f}]"


def render_verdict_markdown(arm: ArmResult, universe: str) -> str:
    """Compact markdown verdict table for one universe (the note's headline table)."""
    df = verdict_df(arm)
    df = df[df["universe"] == universe].sort_values("net_edge_pess_cents", ascending=False)
    lines = [
        f"#### {universe} — per-market bracketed verdict (primary horizon "
        f"{int(df['horizon_s'].iloc[0]) if len(df) else '?'}s, no-rebate)",
        "",
        "| token (short) | avg px | half-spread ¢ | fill rate Opt→RA | net edge Opt (lo) | "
        "net edge RA (lo) | adverse RA ¢ | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {str(r['token_id'])[:10]}… | {r['avg_price']:.2f} | {r['half_spread_cents']:.2f} | "
            f"{r['fill_rate_optimistic']:.2f}→{r['fill_rate_pessimistic']:.2f} | "
            f"{r['net_edge_opt_cents']:+.2f} ({r['net_edge_opt_lo']:+.2f}) | "
            f"{r['net_edge_pess_cents']:+.2f} ({r['net_edge_pess_lo']:+.2f}) | "
            f"{r['adverse_pess_cents']:+.2f} | **{r['verdict_no_rebate']}** |"
        )
    return "\n".join(lines)
