"""Cohort filter definitions for Phase 5 walk-forward backtesting.

Each filter takes a DuckDB connection and a refresh date T and returns a list
of qualifying trader addresses. Metrics are computed from data with
resolution_ts < T to enforce lookahead-free selection.

Stage 1 approximation: `style_role_balance` and `style_avg_holding_hours`
for Cohort 3 use LIFETIME values from `traders.parquet` rather than
point-in-time recomputation. These are slow-changing behavioural classifiers
(a maker-heavy trader stays maker-heavy across years), so the leak from
lifetime-vs-point-in-time is small for cohort gating. The audit log records
this approximation.
"""
from pathlib import Path
from datetime import date, datetime

from data_infra.operator_denylist import OPERATOR_ADDRESSES

ROOT = Path(__file__).resolve().parents[2]
CLOSED_POS = str(ROOT / "data" / "closed_positions.parquet")
TRADERS = str(ROOT / "data" / "traders.parquet")

OPERATOR_LIST_SQL = ", ".join(f"'{a}'" for a in OPERATOR_ADDRESSES)


def _fmt_ts(T) -> str:
    """Format date/datetime for SQL TIMESTAMP literal."""
    if isinstance(T, datetime):
        return T.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(T, date):
        return T.strftime("%Y-%m-%d 00:00:00")
    return str(T)


def cohort_b_high_pf_with_size(con, T) -> list[str]:
    """Cohort 1 — B_high_pf_with_size: Phase 4 Pool B as-is."""
    sql = f"""
    WITH per_market AS (
        SELECT address, market_id,
               sum(realised_pnl) AS market_pnl,
               any_value(resolution_ts) AS resolution_ts
        FROM read_parquet('{CLOSED_POS}')
        WHERE resolution_ts < TIMESTAMP '{_fmt_ts(T)}'
          AND resolution_ts >= first_fill_ts
        GROUP BY address, market_id
    ),
    agg AS (
        SELECT address,
               sum(market_pnl) AS mkt_total_pnl,
               sum(CASE WHEN market_pnl > 0 THEN market_pnl ELSE 0 END) AS sum_win,
               sum(CASE WHEN market_pnl < 0 THEN -market_pnl ELSE 0 END) AS sum_loss_abs,
               count(*) AS n_closed_positions,
               date_diff('day', min(resolution_ts), max(resolution_ts)) AS active_days
        FROM per_market
        GROUP BY address
    )
    SELECT address FROM agg
    WHERE sum_loss_abs > 0
      AND sum_win / sum_loss_abs > 2.0
      AND mkt_total_pnl > 50000
      AND n_closed_positions > 100
      AND active_days > 90
      AND address NOT IN ({OPERATOR_LIST_SQL})
    """
    return con.sql(sql).fetchdf()["address"].tolist()


def cohort_bc_directional_negrisk(con, T) -> list[str]:
    """Cohort 2 — BC_directional_negrisk: Pool B ∩ Pool C, non-arb.

    Selection now uses fill-shape (pct_markets_balanced_and_offsetting_vw)
    rather than phantom_position_score. The old phantom filter conflated
    genuine arb with two-sided directional betting (Domah-style hedging
    without closing) and merge/split artifacts — see
    notes/api_reconciliation_v1.md and
    data/directionality_classification/contamination_crosstabs.md.

    Spec implementation note: the directional-pool spec says
    primary_style IN ('pure_directional', 'two_sided_directional'). The
    SQL form below applies the arb-exclusion half of that rule
    (NOT arb_like, equivalent to vw_arb < 0.30) but does not separately
    exclude `mixed` because that would require computing the volume-weighted
    fc_p50 PIT, which the CTE doesn't (yet) carry. In practice `mixed`
    traders fall in the residual after both gates and tend to be marginal;
    a stricter pass is left for a follow-up.
    """
    sql = f"""
    WITH per_market AS (
        SELECT address, market_id,
               sum(realised_pnl) AS market_pnl,
               sum(abs(realised_pnl)) AS market_abs_pnl,
               any_value(neg_risk) AS neg_risk,
               sum(gross_usd_volume) AS market_volume,
               max(gross_usd_volume) AS top_outcome_volume,
               sum(abs(final_token_position)) AS gross_signed_tokens,
               abs(sum(final_token_position)) AS net_signed_tokens,
               count(*) AS n_outcomes_traded,
               any_value(resolution_ts) AS resolution_ts
        FROM read_parquet('{CLOSED_POS}')
        WHERE resolution_ts < TIMESTAMP '{_fmt_ts(T)}'
          AND resolution_ts >= first_fill_ts
        GROUP BY address, market_id
    ),
    scored AS (
        SELECT *,
               -- arb shape: balanced fills AND offsetting positions.
               -- Thresholds mirror scripts/build_traders_directionality.py
               -- (FC_BALANCED=0.60, NTG_OFFSETTING=0.20).
               CASE WHEN market_volume > 0
                     AND gross_signed_tokens > 0
                     AND (top_outcome_volume / market_volume) < 0.60
                     AND (net_signed_tokens / gross_signed_tokens) < 0.20
                    THEN 1 ELSE 0 END AS is_balanced_offsetting,
               -- (retain phantom for the diagnostic column in the output,
               --  no longer used to filter)
               CASE WHEN abs(market_pnl) > 0.01
                    THEN LEAST(market_abs_pnl / abs(market_pnl), 1000.0)
                    ELSE NULL END AS phantom_ratio
        FROM per_market
    ),
    agg AS (
        SELECT address,
               sum(market_pnl) AS mkt_total_pnl,
               sum(CASE WHEN market_pnl > 0 THEN market_pnl ELSE 0 END) AS sum_win,
               sum(CASE WHEN market_pnl < 0 THEN -market_pnl ELSE 0 END) AS sum_loss_abs,
               count(*) AS n_closed_positions,
               date_diff('day', min(resolution_ts), max(resolution_ts)) AS active_days,
               sum(market_volume) AS total_volume,
               sum(CASE WHEN neg_risk THEN market_volume ELSE 0 END) AS negrisk_volume,
               -- vol-weighted share of arb-shape markets (PIT-safe).
               SUM(is_balanced_offsetting * market_volume)
                 / NULLIF(sum(market_volume), 0)
                   AS pct_markets_balanced_and_offsetting_vw
        FROM scored
        GROUP BY address
    )
    SELECT address FROM agg
    WHERE sum_loss_abs > 0
      AND sum_win / sum_loss_abs > 2.0
      AND mkt_total_pnl > 50000
      AND n_closed_positions > 200
      AND active_days > 90
      AND total_volume > 0
      AND negrisk_volume / total_volume > 0.7
      -- Exclude arb_like (NOT arb-shaped, i.e., genuinely directional).
      AND COALESCE(pct_markets_balanced_and_offsetting_vw, 0) < 0.30
      AND address NOT IN ({OPERATOR_LIST_SQL})
    """
    return con.sql(sql).fetchdf()["address"].tolist()


def cohort_e_patient_accumulators(con, T) -> list[str]:
    """Cohort 3 — E_patient_accumulators: Pool E as-is.

    Note: style_role_balance is from traders.parquet (lifetime value);
    style_avg_holding_hours is computed point-in-time from closed_positions.
    The style_role_balance approximation is documented in audit log.
    """
    sql = f"""
    WITH per_market AS (
        SELECT address, market_id,
               sum(realised_pnl) AS market_pnl,
               any_value(resolution_ts) AS resolution_ts
        FROM read_parquet('{CLOSED_POS}')
        WHERE resolution_ts < TIMESTAMP '{_fmt_ts(T)}'
          AND resolution_ts >= first_fill_ts
        GROUP BY address, market_id
    ),
    pos_agg AS (
        SELECT address,
               sum(market_pnl) AS mkt_total_pnl,
               count(*) AS n_closed_positions,
               date_diff('day', min(resolution_ts), max(resolution_ts)) AS active_days
        FROM per_market
        GROUP BY address
    ),
    style_pit AS (
        SELECT address,
               avg(holding_duration_hours) AS style_avg_holding_hours
        FROM read_parquet('{CLOSED_POS}')
        WHERE resolution_ts < TIMESTAMP '{_fmt_ts(T)}'
          AND resolution_ts >= first_fill_ts
          AND holding_duration_hours >= 0
        GROUP BY address
    )
    SELECT a.address FROM pos_agg a
    JOIN style_pit pp USING (address)
    JOIN read_parquet('{TRADERS}') t USING (address)
    WHERE a.mkt_total_pnl > 100000
      AND a.n_closed_positions > 100
      AND a.active_days > 180
      AND pp.style_avg_holding_hours > 168
      AND t.style_role_balance > 0.7
      AND a.address NOT IN ({OPERATOR_LIST_SQL})
    """
    return con.sql(sql).fetchdf()["address"].tolist()


COHORTS = {
    "B_high_pf_with_size":    cohort_b_high_pf_with_size,
    "BC_directional_negrisk": cohort_bc_directional_negrisk,
    "E_patient_accumulators": cohort_e_patient_accumulators,
}
