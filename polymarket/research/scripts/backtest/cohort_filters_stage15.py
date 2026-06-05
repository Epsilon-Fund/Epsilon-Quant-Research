"""Stage 1.5 cohort filters — percentile-based with absolute floors.

Each function returns a pandas DataFrame with one row per qualifying
address and the columns needed for TopK ranking:

    address, mkt_total_pnl, mkt_profit_factor, n_closed_positions,
    active_days, score

Active-trader gate: `first_activity_ts < T AND last_activity_ts >= T-90d`.
Percentile gates within active set:
    mkt_profit_factor in top 10%
    n_closed_positions in top 25%
Absolute floors: mkt_total_pnl > 0, active_days > 90, not operator.

Cohort-specific overlays:
    Cohort 1 (B_high_pf): no overlay
    Cohort 2 (BC_directional_negrisk): negrisk_share > 0.7 AND phantom < 3.0
    Cohort 3 (E_patient): style_role_balance > 0.7 AND
                          style_avg_holding_hours > 168

Style metrics (style_role_balance) use lifetime values from
traders.parquet — slow-changing trait, documented Stage 1.5 limitation
inherited from Stage 1.
"""
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd

from data_infra.operator_denylist import OPERATOR_ADDRESSES

ROOT = Path(__file__).resolve().parents[2]
CLOSED_POS = str(ROOT / "data" / "closed_positions.parquet")
TRADERS = str(ROOT / "data" / "traders.parquet")

OPERATOR_LIST_SQL = ", ".join(f"'{a}'" for a in OPERATOR_ADDRESSES)


def _fmt_ts(T) -> str:
    if isinstance(T, datetime):
        return T.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(T, date):
        return T.strftime("%Y-%m-%d 00:00:00")
    return str(T)


def _common_base_sql(T_dt) -> str:
    """The base CTE chain shared by all three cohorts.

    Produces a `qualified` table containing every address that passes:
      - active gate (first_activity < T AND last_activity >= T-90d)
      - absolute floors (mkt_total_pnl > 0, active_days > 90, not operator)
      - percentile gates (top 10% PF, top 25% n_closed_positions)
    """
    T_str = _fmt_ts(T_dt)
    T_minus_90 = _fmt_ts(T_dt - timedelta(days=90))
    return f"""
    WITH per_market AS (
        SELECT address, market_id,
               sum(realised_pnl) AS market_pnl,
               sum(abs(realised_pnl)) AS market_abs_pnl,
               sum(gross_usd_volume) AS market_volume,
               any_value(neg_risk) AS neg_risk,
               any_value(resolution_ts) AS resolution_ts
        FROM read_parquet('{CLOSED_POS}')
        WHERE resolution_ts < TIMESTAMP '{T_str}'
          AND resolution_ts >= first_fill_ts
        GROUP BY address, market_id
    ),
    scored AS (
        SELECT *,
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
               min(resolution_ts) AS first_ts,
               max(resolution_ts) AS last_ts,
               date_diff('day', min(resolution_ts), max(resolution_ts)) AS active_days,
               sum(market_volume) AS total_volume,
               sum(CASE WHEN neg_risk THEN market_volume ELSE 0 END) AS negrisk_volume,
               sum(CASE WHEN phantom_ratio IS NOT NULL THEN phantom_ratio * market_volume ELSE 0 END)
                   / NULLIF(sum(CASE WHEN phantom_ratio IS NOT NULL THEN market_volume ELSE 0 END), 0)
                   AS phantom_position_score
        FROM scored
        GROUP BY address
    ),
    active AS (
        SELECT *,
               CASE WHEN sum_loss_abs > 0 THEN sum_win / sum_loss_abs ELSE NULL END
                   AS mkt_profit_factor,
               CASE WHEN total_volume > 0 THEN negrisk_volume / total_volume ELSE 0 END
                   AS negrisk_volume_share
        FROM agg
        WHERE mkt_total_pnl > 0
          AND active_days > 90
          AND first_ts < TIMESTAMP '{T_str}'
          AND last_ts >= TIMESTAMP '{T_minus_90}'
          AND address NOT IN ({OPERATOR_LIST_SQL})
    ),
    ranked AS (
        SELECT *,
               percent_rank() OVER (ORDER BY mkt_profit_factor) AS pf_pct,
               percent_rank() OVER (ORDER BY n_closed_positions) AS npos_pct
        FROM active
        WHERE mkt_profit_factor IS NOT NULL
    ),
    qualified AS (
        SELECT * FROM ranked
        WHERE pf_pct >= 0.90
          AND npos_pct >= 0.75
    )
    """


def cohort_b_stage15(con, T) -> pd.DataFrame:
    """Cohort 1 — B_high_pf. No cohort-specific overlay beyond common gates."""
    sql = _common_base_sql(T) + """
    SELECT address, mkt_total_pnl, mkt_profit_factor, n_closed_positions,
           active_days, negrisk_volume_share, phantom_position_score
    FROM qualified
    """
    return con.sql(sql).fetchdf()


def cohort_bc_stage15(con, T) -> pd.DataFrame:
    """Cohort 2 — BC_directional_negrisk: negrisk-heavy directional traders."""
    sql = _common_base_sql(T) + """
    SELECT address, mkt_total_pnl, mkt_profit_factor, n_closed_positions,
           active_days, negrisk_volume_share, phantom_position_score
    FROM qualified
    WHERE negrisk_volume_share > 0.7
      AND phantom_position_score < 3.0
    """
    return con.sql(sql).fetchdf()


def cohort_e_stage15(con, T) -> pd.DataFrame:
    """Cohort 3 — E_patient_accumulators: maker-heavy long-hold traders.

    Style metrics use lifetime values from traders.parquet (approximation).
    """
    sql = _common_base_sql(T) + f"""
    , style_pit AS (
        SELECT address,
               avg(holding_duration_hours) AS style_avg_holding_hours
        FROM read_parquet('{CLOSED_POS}')
        WHERE resolution_ts < TIMESTAMP '{_fmt_ts(T)}'
          AND resolution_ts >= first_fill_ts
          AND holding_duration_hours >= 0
        GROUP BY address
    )
    SELECT q.address, q.mkt_total_pnl, q.mkt_profit_factor, q.n_closed_positions,
           q.active_days, q.negrisk_volume_share, q.phantom_position_score
    FROM qualified q
    JOIN style_pit sp USING (address)
    JOIN read_parquet('{TRADERS}') t USING (address)
    WHERE sp.style_avg_holding_hours > 168
      AND t.style_role_balance > 0.7
    """
    return con.sql(sql).fetchdf()


COHORTS_STAGE15 = {
    "B_high_pf_with_size":    cohort_b_stage15,
    "BC_directional_negrisk": cohort_bc_stage15,
    "E_patient_accumulators": cohort_e_stage15,
}


def select_top_k(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Rank by mkt_total_pnl × mkt_profit_factor; return top-k (or all)."""
    if df.empty:
        return df.assign(score=pd.Series(dtype=float), rank=pd.Series(dtype=int))
    df = df.copy()
    df["score"] = df["mkt_total_pnl"] * df["mkt_profit_factor"]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df.head(k)
