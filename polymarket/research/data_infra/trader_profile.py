"""Per-trader research dossier — Phase 4.

Usage (interactive):
    from data_infra.trader_profile import profile_trader
    p = profile_trader('0x9d84ce0306f8551e02efef1680475fc0f1dc1344')
    # returns a dict; keys: header, headline_metrics, style, capital,
    # pnl_monthly (DataFrame), market_mix (dict of DataFrames),
    # holding_distribution (DataFrame), activity_cadence (DataFrame),
    # cohort_positioning (dict)

Usage (file):
    profile_trader(addr, output=Path('notes/trader_dossier_X.md'))
    # writes a markdown file; embeds metrics tables inline.

Caveats baked into output:
  - mkt_sharpe is labelled DIAGNOSTIC, not primary
  - est_bankroll_usd_30d_max_approx labelled "Lifetime peak deployed
    capital. Descriptive only — NOT to be used for forward-looking sizing"
  - Footnote on PnL reconciliation always present
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parents[1]
TRADERS_PARQUET = ROOT / "data" / "traders.parquet"
CLOSED_POS_PARQUET = ROOT / "data" / "closed_positions.parquet"
TRADES_GLOB = str(ROOT / "data" / "trades" / "trades_delta_shard*.parquet")
TRADES_SEED = str(ROOT / "data" / "trades" / "trades_seed.parquet")
COHORT_DIR = ROOT / "data" / "cohorts"

POOL_NAMES = [
    "high_sharpe_directional",
    "high_profit_factor_with_size",
    "negrisk_specialists",
    "sports_directional_fast",
    "patient_accumulators",
    "high_kelly_edge",
]

FOOTNOTE = (
    "PnL computed from on-chain trade data + market resolution. "
    "Polymarket's UI may show different numbers due to fee accounting, "
    "mark-to-market on open positions, and merge/split events not "
    "captured here. Treat differences <20% as internally normal."
)


def _latest_markets_path() -> str:
    cands = sorted((ROOT / "data" / "markets").glob("markets_*.parquet"))
    if not cands:
        raise SystemExit("no markets_*.parquet found")
    return str(cands[-1])


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("SET preserve_insertion_order=false")
    return con


def _pools_qualified(con: duckdb.DuckDBPyConnection, address: str) -> list[str]:
    qualified = []
    for pool in POOL_NAMES:
        path = COHORT_DIR / f"{pool}.parquet"
        if not path.exists():
            continue
        n = con.sql(
            f"SELECT count(*) FROM read_parquet('{path}') WHERE address = '{address}'"
        ).fetchone()[0]
        if n > 0:
            qualified.append(pool)
    return qualified


def _verdict(row: dict, pools: list[str]) -> str:
    if row.get("is_operator_like"):
        return "operator-flagged (excluded from cohort selection)"
    if len(pools) >= 3:
        return f"robust multi-pool candidate ({len(pools)} pools)"
    if len(pools) == 2:
        return f"two-pool candidate ({', '.join(pools)})"
    if len(pools) == 1:
        return f"single-pool candidate ({pools[0]})"
    return "below cohort thresholds"


def profile_trader(
    address: str,
    output: Path | None = None,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """Generate one-trader research dossier."""
    address = address.lower()
    own_con = con is None
    if own_con:
        con = _connect()

    try:
        return _profile_impl(con, address, output)
    finally:
        if own_con:
            con.close()


def _profile_impl(
    con: duckdb.DuckDBPyConnection,
    address: str,
    output: Path | None,
) -> dict[str, Any]:
    # Header row from traders.parquet
    df = con.sql(f"""
        SELECT * FROM read_parquet('{TRADERS_PARQUET}')
        WHERE address = '{address}'
    """).fetchdf()
    if df.empty:
        result = {"header": {"address": address, "in_traders_table": False}}
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(f"# Trader profile — {address}\n\n**Not in traders.parquet** (no closed-market activity).\n")
        return result
    row = df.iloc[0].to_dict()
    pools = _pools_qualified(con, address)
    verdict = _verdict(row, pools)

    # PnL by month from closed_positions
    pnl_monthly = con.sql(f"""
        WITH cp AS (
            SELECT * FROM read_parquet('{CLOSED_POS_PARQUET}')
            WHERE address = '{address}'
              AND holding_duration_hours >= 0
              AND resolution_ts IS NOT NULL
        ),
        per_market AS (
            SELECT market_id, sum(realised_pnl) AS market_pnl,
                   any_value(resolution_ts) AS resolution_ts
            FROM cp GROUP BY market_id
        )
        SELECT date_trunc('month', resolution_ts) AS month,
               sum(market_pnl) AS month_pnl,
               count(*) AS n_markets_resolved
        FROM per_market
        GROUP BY month ORDER BY month
    """).fetchdf()
    if not pnl_monthly.empty:
        pnl_monthly["cum_pnl"] = pnl_monthly["month_pnl"].cumsum()

    # Market mix — top 10 winners / losers
    markets_path = _latest_markets_path()
    top_winners = con.sql(f"""
        WITH cp AS (
            SELECT market_id, sum(realised_pnl) AS market_pnl,
                   sum(n_fills) AS n_fills,
                   any_value(holding_duration_hours) AS holding_hours,
                   any_value(neg_risk) AS neg_risk
            FROM read_parquet('{CLOSED_POS_PARQUET}')
            WHERE address = '{address}'
              AND holding_duration_hours >= 0
            GROUP BY market_id
        )
        SELECT cp.market_id, m.question,
               round(cp.market_pnl, 2) AS realised_pnl,
               cp.n_fills, round(cp.holding_hours, 1) AS holding_hours,
               cp.neg_risk
        FROM cp LEFT JOIN read_parquet('{markets_path}') m
          ON CAST(m.id AS VARCHAR) = cp.market_id
        ORDER BY realised_pnl DESC LIMIT 10
    """).fetchdf()

    top_losers = con.sql(f"""
        WITH cp AS (
            SELECT market_id, sum(realised_pnl) AS market_pnl,
                   sum(n_fills) AS n_fills,
                   any_value(holding_duration_hours) AS holding_hours,
                   any_value(neg_risk) AS neg_risk
            FROM read_parquet('{CLOSED_POS_PARQUET}')
            WHERE address = '{address}'
              AND holding_duration_hours >= 0
            GROUP BY market_id
        )
        SELECT cp.market_id, m.question,
               round(cp.market_pnl, 2) AS realised_pnl,
               cp.n_fills, round(cp.holding_hours, 1) AS holding_hours,
               cp.neg_risk
        FROM cp LEFT JOIN read_parquet('{markets_path}') m
          ON CAST(m.id AS VARCHAR) = cp.market_id
        ORDER BY realised_pnl ASC LIMIT 10
    """).fetchdf()

    # NegRisk vs regular PnL split
    pnl_by_negrisk = con.sql(f"""
        SELECT
            COALESCE(neg_risk, FALSE) AS neg_risk,
            round(sum(realised_pnl), 2) AS pnl,
            count(DISTINCT market_id) AS n_markets
        FROM read_parquet('{CLOSED_POS_PARQUET}')
        WHERE address = '{address}'
          AND holding_duration_hours >= 0
        GROUP BY 1
    """).fetchdf()

    # Holding duration distribution
    holding_stats = con.sql(f"""
        SELECT
            count(*) AS n,
            round(quantile_cont(holding_duration_hours, 0.10), 2) AS p10,
            round(median(holding_duration_hours), 2) AS p50,
            round(quantile_cont(holding_duration_hours, 0.90), 2) AS p90,
            round(quantile_cont(holding_duration_hours, 0.99), 2) AS p99,
            round(max(holding_duration_hours), 2) AS max_h
        FROM read_parquet('{CLOSED_POS_PARQUET}')
        WHERE address = '{address}'
          AND holding_duration_hours >= 0
    """).fetchdf()

    # Activity cadence (per-day fill counts) from raw trades — scoped to this address
    activity = con.sql(f"""
        WITH t AS (
            SELECT timestamp FROM read_parquet('{TRADES_GLOB}')
            WHERE maker = '{address}' OR taker = '{address}'
            UNION ALL
            SELECT timestamp FROM read_parquet('{TRADES_SEED}')
            WHERE maker = '{address}' OR taker = '{address}'
        )
        SELECT date_trunc('day', timestamp) AS day, count(*) AS n_fills
        FROM t GROUP BY day ORDER BY day
    """).fetchdf()

    # Cohort positioning — for each pool the trader qualified for, show their
    # rank percentile vs the pool on key metrics
    cohort_positioning: dict[str, Any] = {}
    for pool in pools:
        path = COHORT_DIR / f"{pool}.parquet"
        rank_df = con.sql(f"""
            WITH p AS (
                SELECT * FROM read_parquet('{path}')
            )
            SELECT
                round(median(mkt_total_pnl), 0) AS pool_med_pnl,
                round(median(mkt_profit_factor), 2) AS pool_med_pf,
                round(median(mkt_dollar_win_rate), 3) AS pool_med_dwr,
                round(median(mkt_sharpe), 2) AS pool_med_sharpe,
                100 * sum(CASE WHEN mkt_total_pnl <= {row.get('mkt_total_pnl', 0) or 0} THEN 1 ELSE 0 END)::DOUBLE
                    / count(*) AS pnl_percentile,
                100 * sum(CASE WHEN mkt_profit_factor <= {row.get('mkt_profit_factor', 0) or 0} THEN 1 ELSE 0 END)::DOUBLE
                    / count(*) AS pf_percentile
            FROM p
        """).fetchdf()
        if not rank_df.empty:
            cohort_positioning[pool] = rank_df.iloc[0].to_dict()

    result: dict[str, Any] = {
        "header": {
            "address": address,
            "pools_qualified": pools,
            "verdict": verdict,
            "in_traders_table": True,
        },
        "headline_metrics": {
            "n_closed_positions": int(row["n_closed_positions"] or 0),
            "n_distinct_markets": int(row["n_distinct_markets"] or 0),
            "active_days": int(row.get("active_days") or 0),
            "n_fills_total": int(row.get("n_fills_total") or 0),
            "mkt_total_pnl_PRIMARY": row["mkt_total_pnl"],
            "pos_total_pnl": row["pos_total_pnl"],
            "mkt_profit_factor": row["mkt_profit_factor"],
            "mkt_dollar_win_rate": row["mkt_dollar_win_rate"],
            "mkt_sharpe_DIAGNOSTIC_only": row["mkt_sharpe"],
            "mkt_kelly_fraction": row["mkt_kelly_fraction"],
            "phantom_position_score": row["phantom_position_score"],
            "negrisk_volume_share": row["negrisk_volume_share"],
        },
        "style": {
            "style_role_balance_1eqMaker_0eqTaker": row.get("style_role_balance"),
            "style_avg_holding_hours": row.get("style_avg_holding_hours"),
            "style_median_holding_hours": row.get("style_median_holding_hours"),
            "style_pct_sub_second_within_1s_of_same_addr_fill": row.get("style_pct_sub_second"),
            "style_avg_fill_size_usd": row.get("style_avg_fill_size_usd"),
            "style_max_fill_size_usd": row.get("style_max_fill_size_usd"),
            "style_buy_sell_symmetry_0eqDirectional_1eqMM": row.get("style_buy_sell_symmetry"),
        },
        "capital_footprint": {
            "est_bankroll_usd_30d_max_approx": row.get("est_bankroll_usd_30d_max_approx"),
            "label": "Lifetime peak deployed capital. Descriptive only — NOT to be used for forward-looking sizing decisions.",
        },
        "pnl_monthly": pnl_monthly,
        "market_mix": {
            "top_winners": top_winners,
            "top_losers": top_losers,
            "pnl_by_negrisk": pnl_by_negrisk,
        },
        "holding_distribution": holding_stats,
        "activity_cadence": activity,
        "cohort_positioning": cohort_positioning,
        "footnote": FOOTNOTE,
    }

    if output is not None:
        _write_markdown(result, output)

    return result


def _write_markdown(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    h = result["header"]
    lines: list[str] = []
    lines.append(f"# Trader profile — `{h['address']}`\n")
    lines.append(f"**Verdict:** {h['verdict']}\n")
    pools = h.get("pools_qualified") or []
    lines.append(f"**Pools qualified ({len(pools)}):** "
                 f"{', '.join(pools) if pools else '_none_'}\n")

    lines.append("\n## Headline metrics\n")
    lines.append("| metric | value | note |\n|---|---:|---|\n")
    hm = result["headline_metrics"]
    n_pos = hm["n_closed_positions"]
    sharpe_warning = " ⚠️ low n_pos" if n_pos < 200 else ""
    lines.append(f"| n_closed_positions | {n_pos:,} |  |\n")
    lines.append(f"| n_distinct_markets | {hm['n_distinct_markets']:,} |  |\n")
    lines.append(f"| active_days | {hm['active_days']:,} |  |\n")
    lines.append(f"| n_fills_total | {hm['n_fills_total']:,} |  |\n")
    lines.append(f"| **mkt_total_pnl** | ${hm['mkt_total_pnl_PRIMARY']:,.0f} | **PRIMARY — trust this** |\n")
    lines.append(f"| pos_total_pnl | ${hm['pos_total_pnl']:,.0f} | inflated for NegRisk arb |\n")
    lines.append(f"| mkt_profit_factor | {hm['mkt_profit_factor']:.2f} |  |\n")
    lines.append(f"| mkt_dollar_win_rate | {hm['mkt_dollar_win_rate']:.3f} |  |\n")
    lines.append(f"| mkt_sharpe | _{hm['mkt_sharpe_DIAGNOSTIC_only']:.2f}_ | _DIAGNOSTIC{sharpe_warning}_ |\n")
    lines.append(f"| mkt_kelly_fraction | {hm['mkt_kelly_fraction']:.3f} |  |\n")
    lines.append(f"| phantom_position_score | {hm['phantom_position_score']:.2f} | >1 ⇒ NegRisk arb |\n")
    lines.append(f"| negrisk_volume_share | {hm['negrisk_volume_share']:.2f} |  |\n")

    lines.append("\n## Style profile\n")
    lines.append("| metric | value | note |\n|---|---:|---|\n")
    s = result["style"]
    lines.append(f"| style_role_balance | {s['style_role_balance_1eqMaker_0eqTaker']:.2f} | 1.0 = pure maker, 0.0 = pure taker |\n")
    avgh = s.get("style_avg_holding_hours")
    medh = s.get("style_median_holding_hours")
    if avgh is not None:
        lines.append(f"| style_avg_holding_hours | {avgh:.1f} |  |\n")
    if medh is not None:
        lines.append(f"| style_median_holding_hours | {medh:.1f} |  |\n")
    lines.append(f"| style_pct_sub_second | {s['style_pct_sub_second_within_1s_of_same_addr_fill']:.1f}% | % fills within 1s of another from same address |\n")
    lines.append(f"| style_avg_fill_size_usd | ${s['style_avg_fill_size_usd']:,.2f} |  |\n")
    lines.append(f"| style_max_fill_size_usd | ${s['style_max_fill_size_usd']:,.0f} |  |\n")
    lines.append(f"| style_buy_sell_symmetry | {s['style_buy_sell_symmetry_0eqDirectional_1eqMM']:.2f} | 0 = directional, 1 = MM-shaped |\n")

    lines.append("\n## Capital footprint\n")
    cap = result["capital_footprint"]
    lines.append(f"- **est_bankroll_usd_30d_max_approx**: ${cap['est_bankroll_usd_30d_max_approx']:,.0f}\n")
    lines.append(f"  - {cap['label']}\n")

    lines.append("\n## Monthly cumulative PnL\n")
    pm = result["pnl_monthly"]
    if not pm.empty:
        lines.append("| month | month_pnl | cum_pnl | n_markets_resolved |\n|---|---:|---:|---:|\n")
        for _, r in pm.iterrows():
            lines.append(f"| {r['month']} | ${r['month_pnl']:,.0f} | ${r['cum_pnl']:,.0f} | {int(r['n_markets_resolved'])} |\n")
    else:
        lines.append("_(no closed positions with valid resolution_ts)_\n")

    lines.append("\n## Market mix\n")
    lines.append("\n### Top 10 winning markets\n")
    lines.append("| market_id | question | realised_pnl | n_fills | holding_h | neg_risk |\n|---|---|---:|---:|---:|---|\n")
    for _, r in result["market_mix"]["top_winners"].iterrows():
        q = (r["question"] or "—")[:60]
        lines.append(f"| {r['market_id']} | {q} | ${r['realised_pnl']:,.0f} | {int(r['n_fills'])} | {r['holding_hours']} | {r['neg_risk']} |\n")

    lines.append("\n### Top 10 losing markets\n")
    lines.append("| market_id | question | realised_pnl | n_fills | holding_h | neg_risk |\n|---|---|---:|---:|---:|---|\n")
    for _, r in result["market_mix"]["top_losers"].iterrows():
        q = (r["question"] or "—")[:60]
        lines.append(f"| {r['market_id']} | {q} | ${r['realised_pnl']:,.0f} | {int(r['n_fills'])} | {r['holding_hours']} | {r['neg_risk']} |\n")

    lines.append("\n### NegRisk vs regular split\n")
    lines.append("| neg_risk | pnl | n_markets |\n|---|---:|---:|\n")
    for _, r in result["market_mix"]["pnl_by_negrisk"].iterrows():
        lines.append(f"| {r['neg_risk']} | ${r['pnl']:,.0f} | {int(r['n_markets'])} |\n")

    lines.append("\n## Holding-duration distribution (hours)\n")
    hd = result["holding_distribution"]
    if not hd.empty:
        r = hd.iloc[0]
        lines.append(f"- n: {int(r['n']):,}, p10: {r['p10']}, p50: {r['p50']}, "
                     f"p90: {r['p90']}, p99: {r['p99']}, max: {r['max_h']}\n")

    lines.append("\n## Activity cadence\n")
    ac = result["activity_cadence"]
    if not ac.empty:
        n_active_days = len(ac)
        avg_fills_per_day = ac["n_fills"].mean()
        max_fills_per_day = ac["n_fills"].max()
        lines.append(f"- {n_active_days} days with at least one fill\n")
        lines.append(f"- Avg fills/active-day: {avg_fills_per_day:.1f}\n")
        lines.append(f"- Max fills on a single day: {int(max_fills_per_day):,}\n")

    lines.append("\n## Cohort positioning\n")
    cp = result["cohort_positioning"]
    if cp:
        lines.append("| pool | trader's pnl_percentile | trader's pf_percentile | pool median PnL | pool median PF |\n|---|---:|---:|---:|---:|\n")
        for pool, vals in cp.items():
            lines.append(
                f"| {pool} | {vals['pnl_percentile']:.0f}% | {vals['pf_percentile']:.0f}% | "
                f"${vals['pool_med_pnl']:,.0f} | {vals['pool_med_pf']:.2f} |\n"
            )
    else:
        lines.append("_(no cohort qualifications)_\n")

    lines.append(f"\n---\n\n_{result['footnote']}_\n")
    output.write_text("".join(lines))
