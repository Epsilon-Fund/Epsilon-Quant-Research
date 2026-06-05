"""Build data/copyability_candidates/traders_copyability_metrics.parquet.

Per-trader sidecar that surfaces the metrics needed to *prefilter* traders
for the copy-execution audit. NO scoring, NO PASS/FAIL, NO threshold gates —
just raw metrics keyed on `address`, intended for humans to eyeball.

Universe:
  Same as scripts/build_traders_directionality.py — addresses with
  n_closed_positions > 50, operators excluded.

Inputs (read-only):
  - data/closed_positions.parquet                          (canonical)
  - data/traders.parquet                                   (canonical)
  - data/directionality_classification/traders_directionality.parquet
  - data/trades/*.parquet + data/trades/trades_seed.parquet (raw fills, via view)
  - data/markets/markets_*.parquet                         (slug → family)
  - data/analysis/*_audit_report.md                        (audit cross-ref)

Output:
  - data/copyability_candidates/traders_copyability_metrics.parquet

Schema is documented in data/copyability_candidates/metric_distributions.md.

Run from polymarket/research/:
    PYTHONPATH=. python3 scripts/build_copyability_metrics.py
"""
from __future__ import annotations

import re
import time
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from data_infra.operator_denylist import OPERATOR_ADDRESSES
from data_infra.views import latest_markets_path, load_views

ROOT = Path(__file__).resolve().parents[1]
CLOSED_POS = ROOT / "data" / "closed_positions.parquet"
TRADERS = ROOT / "data" / "traders.parquet"
DIRECTIONALITY = ROOT / "data" / "directionality_classification" / "traders_directionality.parquet"
ANALYSIS_DIR = ROOT / "data" / "analysis"
OUT_DIR = ROOT / "data" / "copyability_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PARQUET = OUT_DIR / "traders_copyability_metrics.parquet"

MIN_CLOSED_POSITIONS = 50
LOOKBACK_30D = "30 days"
LOOKBACK_90D = "90 days"
HOLD_TO_RES_THRESHOLD = 0.05    # last fill within final 5% of market lifetime

# Family heuristic — same set used in scripts/domah_copy_audit.py.
FAMILY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("sports", (
        "-mlb-", "-mlb", "mlb-", "-nba-", "-nba", "nba-", "-nfl-", "-nfl", "nfl-",
        "-nhl-", "-nhl", "nhl-", "-cfb-", "-cfb", "cfb-", "-ncaa-", "ncaa-",
        "-ufc-", "ufc-", "-soccer-", "soccer-",
        "premier-league", "champions-league", "-ucl-", "ucl-",
        "world-cup", "tennis", "atp-", "wta-", "grand-slam",
        "-boxing-", "boxing-", "-mma-", "mma-",
        "formula-1", "-f1-", "nascar",
        "yankees", "dodgers", "lakers", "warriors", "chiefs", "eagles",
        "broncos", "knicks", "celtics", "heat", "patriots",
        "saquon", "shedeur", "wrestlemania",
        "kentucky-derby", "preakness", "belmont", "open-championship",
        "masters-tournament", "ryder-cup", "stanley-cup", "world-series",
        "super-bowl", "us-open", "nba-mvp", "nfl-mvp", "heisman",
        "-pga-", "pga-", "-pga", "championship",
        "jannik-sinner", "novak-djokovic", "carlos-alcaraz", "iga-swiatek",
        "ja-morant", "jokic", "lebron",
    )),
    ("crypto", (
        "bitcoin", "ethereum", "solana", "doge", "dogecoin", "litecoin",
        "ripple", "xrp", "cardano", "polkadot", "avalanche",
        "binance", "coinbase", "kraken", "tether", "usdc",
        "crypto", "blockchain", "defi", "stablecoin",
        "-btc-", "btc-", "-eth-", "eth-", "-sol-", "sol-",
        "vitalik", "satoshi", "-cz-", "cz-",
        "sec-vs", "sec-coinbase",
        "spot-etf", "bitcoin-etf", "ether-etf",
    )),
    ("macro", (
        "-fed-", "fed-", "federal-reserve", "fomc",
        "jerome-powell", "jpow",
        "-cpi-", "cpi-", "inflation", "deflation",
        "-ppi-", "ppi-",
        "-gdp-", "gdp-",
        "jobs-report", "nonfarm", "-nfp-", "nfp-",
        "unemployment", "recession",
        "interest-rates", "rate-cut", "rate-hike", "rate-decision",
        "treasury-yield", "yield-curve", "10-year-yield",
        "sp-500", "sp500", "-qqq-", "qqq-",
        "tariff", "trade-deal",
    )),
    ("weather", (
        "temperature", "high-temp", "hottest", "coldest",
        "hurricane", "tropical-storm", "typhoon", "cyclone",
        "tornado", "snowfall", "snow-in", "rainfall",
        "atmospheric-river", "heatwave", "blizzard",
        "wildfire", "noaa",
    )),
    ("politics", (
        "trump", "biden", "harris", "vance", "kamala", "obama",
        "election", "primary", "caucus",
        "senate", "house-of-rep", "congress", "congressional",
        "mayoral", "governor", "presidential",
        "democrat", "republican", "-gop-", "gop-", "-dnc-", "dnc-",
        "-rnc-", "rnc-",
        "supreme-court", "-scotus-", "scotus-",
        "putin", "zelensky", "ukraine", "russia",
        "israel", "palestine", "gaza", "hamas", "hezbollah", "houthi",
        "iran", "netanyahu", "khamenei",
        "nato", "european-union", "-eu-", "brexit",
        "macron", "starmer", "merz", "scholz", "meloni",
        "modi", "xi-jinping", "china-", "taiwan", "north-korea", "kim-jong",
        "pope", "vatican",
        "white-house", "executive-order", "impeach", "indict",
        "fbi-", "doj-", "pentagon",
        "syria", "lebanon", "venezuela", "maduro",
        "epstein", "ghislaine",
        "shutdown", "debt-ceiling",
        "us-x-", "us-iran", "us-russia", "us-china", "us-strikes",
        "rodrigo", "milei", "musk", "vivek",
        "tiktok",
    )),
]

# Hard-coded list of addresses explicitly characterised as structurally
# uncopyable via fill-mirroring (e.g., split-position-construction traders).
# See data/analysis/leader_dthreed8b71_investigation/dthreed8b71_strategy_profile.md.
FLAGGED_UNCOPYABLE = {
    "0xd38b71f3e8ed1af71983e5c309eac3dfa9b35029",
}


def family_of(slug: str | None) -> str:
    if not slug:
        return "other"
    s = "-" + slug.lower() + "-"
    for fam, kws in FAMILY_KEYWORDS:
        for kw in kws:
            if kw in s:
                return fam
    return "other"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Audit cross-reference
# ----------------------------------------------------------------------------
def scan_audit_artifacts() -> dict[str, dict]:
    """Walk data/analysis/ for *_audit_report.md, extract the trader address,
    and locate the matching *_audit_family.parquet.

    Returns a dict: {address_lower: {"family_parquet": Path|None,
                                     "report_md": Path}}.
    """
    out: dict[str, dict] = {}
    for report in ANALYSIS_DIR.rglob("*_audit_report.md"):
        text = report.read_text(errors="ignore")
        m = re.search(r"\*\*address\*\*[^\n]*?(0x[a-f0-9]{40})", text)
        if not m:
            continue
        addr = m.group(1).lower()
        # Sibling family parquet uses the same stem (replace report.md -> family.parquet)
        family_parquet = report.with_name(report.name.replace("_audit_report.md", "_audit_family.parquet"))
        if not family_parquet.exists():
            family_parquet = None
        out[addr] = {"family_parquet": family_parquet, "report_md": report}
    return out


def compute_n_deployable_cells(family_parquet: Path) -> int:
    """Count audit family rows meeting deployable thresholds.

    Spec: A_real_capture > 0.30 AND adverse_select_ratio > 0.85 AND leader_pnl > 0.
    Family-level proxy is used because (family × role × hour) cell-level
    parquet wasn't materialised by the existing audits.
    """
    df = pd.read_parquet(family_parquet)
    mask = (
        (df["A_real_capture"] > 0.30)
        & (df["adverse_select_ratio"] > 0.85)
        & (df["leader_pnl"] > 0)
    )
    return int(mask.sum())


# ----------------------------------------------------------------------------
# Build pipeline
# ----------------------------------------------------------------------------
def build_universe(con: duckdb.DuckDBPyConnection) -> int:
    op_sql = ", ".join(f"'{a}'" for a in OPERATOR_ADDRESSES)
    con.execute(f"""
        CREATE OR REPLACE TABLE qualifying_addrs AS
        SELECT address
        FROM read_parquet('{CLOSED_POS}')
        WHERE address NOT IN ({op_sql})
        GROUP BY address
        HAVING count(*) > {MIN_CLOSED_POSITIONS}
    """)
    return con.sql("SELECT count(*) FROM qualifying_addrs").fetchone()[0]


def build_position_table(con: duckdb.DuckDBPyConnection) -> None:
    """Materialise per-position rows for qualifying addresses, with bucket label
    and hold-to-resolution flag.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE per_position AS
        SELECT
            cp.address, cp.market_id, cp.outcome_index, cp.n_fills,
            cp.realised_pnl, cp.total_bought_usd, cp.total_sold_usd,
            cp.final_token_position, cp.first_fill_ts, cp.last_fill_ts,
            cp.resolution_ts,
            CASE
                WHEN cp.total_bought_usd < 1.0 AND cp.total_sold_usd >= 1.0 THEN 'sold_no_buy'
                WHEN cp.final_token_position < -1e-6 THEN 'net_short_position'
                WHEN cp.total_bought_usd >= 1.0 AND cp.total_sold_usd >= 1.0 THEN 'bought_and_sold'
                WHEN cp.total_bought_usd >= 1.0 AND cp.total_sold_usd < 1.0 THEN 'bought_only'
                ELSE 'tiny_dust'
            END AS bucket,
            CASE
                WHEN cp.resolution_ts IS NULL OR cp.last_fill_ts IS NULL THEN NULL
                WHEN cp.resolution_ts <= cp.first_fill_ts THEN NULL
                ELSE epoch(cp.resolution_ts - cp.last_fill_ts)
                     / NULLIF(epoch(cp.resolution_ts - cp.first_fill_ts), 0)
            END AS time_to_res_share_post_last_fill
        FROM read_parquet('{CLOSED_POS}') cp
        JOIN qualifying_addrs USING (address)
    """)
    n = con.sql("SELECT count(*) FROM per_position").fetchone()[0]
    log(f"  per_position rows: {n:,}")


def build_position_aggregates(con: duckdb.DuckDBPyConnection) -> None:
    """fragmentation_index, hold_to_resolution_share, split_position_signature,
    median_per_position_pnl — all from per_position only.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE m_pos_aggs AS
        WITH frag AS (
            SELECT address, MEDIAN(n_fills) AS fragmentation_index
            FROM per_position GROUP BY address
        ),
        htr AS (
            SELECT address,
                   SUM(CASE WHEN time_to_res_share_post_last_fill <= {HOLD_TO_RES_THRESHOLD}
                            THEN 1 ELSE 0 END)::DOUBLE
                       / NULLIF(SUM(CASE WHEN time_to_res_share_post_last_fill IS NOT NULL
                                          THEN 1 ELSE 0 END), 0)
                   AS hold_to_resolution_share
            FROM per_position GROUP BY address
        ),
        binary_markets AS (
            SELECT address, market_id,
                   count(*) AS n_outcomes,
                   SUM(CASE WHEN bucket = 'bought_only' THEN 1 ELSE 0 END) AS n_bought_only,
                   SUM(CASE WHEN bucket = 'sold_no_buy' THEN 1 ELSE 0 END) AS n_sold_no_buy
            FROM per_position GROUP BY address, market_id
        ),
        split AS (
            SELECT address,
                   SUM(CASE WHEN n_outcomes = 2 AND n_bought_only = 1 AND n_sold_no_buy = 1
                            THEN 1 ELSE 0 END)::DOUBLE
                       / NULLIF(SUM(CASE WHEN n_outcomes = 2 THEN 1 ELSE 0 END), 0)
                   AS split_position_signature
            FROM binary_markets GROUP BY address
        ),
        med AS (
            SELECT address, MEDIAN(realised_pnl) AS median_per_position_pnl
            FROM per_position GROUP BY address
        )
        SELECT q.address,
               frag.fragmentation_index,
               htr.hold_to_resolution_share,
               split.split_position_signature,
               med.median_per_position_pnl
        FROM qualifying_addrs q
        LEFT JOIN frag USING (address)
        LEFT JOIN htr USING (address)
        LEFT JOIN split USING (address)
        LEFT JOIN med USING (address)
    """)


def build_market_aggregates(con: duckdb.DuckDBPyConnection) -> None:
    """pct_markets_winning, win_loss_size_ratio — from per-market PnL."""
    con.execute("""
        CREATE OR REPLACE TABLE m_mkt AS
        WITH pmkt AS (
            SELECT address, market_id, sum(realised_pnl) AS pnl
            FROM per_position GROUP BY address, market_id
        ),
        agg AS (
            SELECT address,
                   count(*) AS n_markets_eval,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::DOUBLE / count(*)
                       AS pct_markets_winning,
                   avg(CASE WHEN pnl > 0 THEN pnl END) AS avg_pnl_win_mkt,
                   avg(CASE WHEN pnl < 0 THEN pnl END) AS avg_pnl_lose_mkt
            FROM pmkt GROUP BY address
        )
        SELECT address, pct_markets_winning,
               CASE WHEN avg_pnl_lose_mkt IS NULL OR avg_pnl_lose_mkt = 0
                    THEN NULL
                    ELSE avg_pnl_win_mkt / abs(avg_pnl_lose_mkt) END
                 AS win_loss_size_ratio
        FROM agg
    """)


def build_family_aggregates(con: duckdb.DuckDBPyConnection) -> None:
    """market_family_concentration (HHI on |pnl| share) + dominant_family."""
    # Build market → family lookup via Python (slug regex matching is awkward in SQL).
    mkts_path = latest_markets_path()
    log("  building market → family lookup")
    mkt_df = con.sql(f"""
        SELECT CAST(id AS VARCHAR) AS market_id, slug
        FROM read_parquet('{mkts_path}')
    """).fetchdf()
    mkt_df["family"] = mkt_df["slug"].map(family_of)
    con.execute("CREATE OR REPLACE TABLE mkt_family AS SELECT * FROM mkt_df")

    con.execute("""
        CREATE OR REPLACE TABLE m_fam AS
        WITH pmkt AS (
            SELECT address, market_id, sum(realised_pnl) AS market_pnl
            FROM per_position GROUP BY address, market_id
        ),
        joined AS (
            SELECT pm.address, pm.market_pnl,
                   COALESCE(mf.family, 'other') AS family
            FROM pmkt pm LEFT JOIN mkt_family mf USING (market_id)
        ),
        per_fam AS (
            SELECT address, family,
                   sum(abs(market_pnl)) AS abs_pnl_in_family,
                   sum(market_pnl) AS net_pnl_in_family
            FROM joined GROUP BY address, family
        ),
        totals AS (
            SELECT address, sum(abs_pnl_in_family) AS total_abs_pnl
            FROM per_fam GROUP BY address
        ),
        shares AS (
            SELECT pf.address, pf.family,
                   pf.abs_pnl_in_family,
                   CASE WHEN t.total_abs_pnl > 0
                        THEN pf.abs_pnl_in_family / t.total_abs_pnl ELSE 0 END
                       AS family_share
            FROM per_fam pf JOIN totals t USING (address)
        )
        SELECT address,
               SUM(family_share * family_share) AS market_family_concentration,
               arg_max(family, abs_pnl_in_family) AS dominant_family
        FROM shares GROUP BY address
    """)


def build_recent_activity(con: duckdb.DuckDBPyConnection) -> None:
    """From raw fills: active_days_last_90d, volume_30d_to_lifetime_ratio.

    Computed by scanning the trades view once. Snapshot date is the max
    timestamp present in the data, not a wallclock date — keeps the metric
    stable across reruns and aligned with the data freshness, which lags
    Goldsky by ~9 days.
    """
    snap_ts = con.sql("SELECT max(timestamp) AS m FROM raw_trades").fetchone()[0]
    log(f"  snapshot timestamp = {snap_ts}")
    con.execute(f"""
        CREATE OR REPLACE TABLE m_recent AS
        WITH ex AS (
            SELECT rt.timestamp, s.role,
                   CASE WHEN s.role = 'maker' THEN rt.maker ELSE rt.taker END AS address,
                   rt.usd_amount
            FROM raw_trades rt
            CROSS JOIN (VALUES ('maker'), ('taker')) AS s(role)
            WHERE (CASE WHEN s.role = 'maker' THEN rt.maker ELSE rt.taker END) IS NOT NULL
              AND rt.maker IS NOT NULL AND rt.taker IS NOT NULL
              AND rt.maker <> rt.taker
        ),
        ex_q AS (
            SELECT e.timestamp, e.address, e.usd_amount
            FROM ex e JOIN qualifying_addrs q USING (address)
        )
        SELECT
            address,
            sum(usd_amount) AS lifetime_volume_usd,
            sum(CASE WHEN timestamp >= TIMESTAMP '{snap_ts}' - INTERVAL '{LOOKBACK_30D}'
                     THEN usd_amount ELSE 0 END) AS vol_last_30d,
            count(DISTINCT CASE WHEN timestamp >= TIMESTAMP '{snap_ts}' - INTERVAL '{LOOKBACK_90D}'
                                THEN date_trunc('day', timestamp) END) AS active_days_last_90d
        FROM ex_q GROUP BY address
    """)

    con.execute("""
        CREATE OR REPLACE TABLE m_recent_final AS
        SELECT address,
               active_days_last_90d,
               CASE WHEN lifetime_volume_usd > 0
                    THEN vol_last_30d / lifetime_volume_usd ELSE NULL END
                 AS volume_30d_to_lifetime_ratio,
               lifetime_volume_usd
        FROM m_recent
    """)


def join_existing(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Join in traders.parquet + directionality + computed metrics + audit info."""
    audits = scan_audit_artifacts()
    audited_addrs = set(audits)

    df = con.sql(f"""
        SELECT q.address,
               -- traders.parquet existing metrics (lifetime)
               t.mkt_total_pnl, t.mkt_profit_factor, t.mkt_dollar_win_rate,
               t.mkt_sharpe,
               t.n_closed_positions, t.n_distinct_markets, t.active_days,
               t.n_fills_total,
               t.phantom_position_score, t.negrisk_volume_share,
               t.style_role_balance, t.style_pct_sub_second,
               t.style_avg_holding_hours, t.style_buy_sell_symmetry,
               t.est_bankroll_lifetime_peak_deprecated AS est_bankroll_usd_30d_max_approx,
               -- directionality
               d.primary_style,
               d.pct_markets_balanced_and_offsetting_vw,
               d.pct_markets_two_sided_directional_vw,
               d.fill_concentration_p50,
               d.net_to_gross_exposure,
               -- new position-level aggregates
               p.fragmentation_index, p.hold_to_resolution_share,
               p.split_position_signature, p.median_per_position_pnl,
               -- new market-level aggregates
               mm.pct_markets_winning, mm.win_loss_size_ratio,
               -- family aggregates
               mf.market_family_concentration, mf.dominant_family,
               -- recency
               r.active_days_last_90d, r.volume_30d_to_lifetime_ratio,
               r.lifetime_volume_usd
        FROM qualifying_addrs q
        LEFT JOIN read_parquet('{TRADERS}') t USING (address)
        LEFT JOIN read_parquet('{DIRECTIONALITY}') d USING (address)
        LEFT JOIN m_pos_aggs p USING (address)
        LEFT JOIN m_mkt mm USING (address)
        LEFT JOIN m_fam mf USING (address)
        LEFT JOIN m_recent_final r USING (address)
    """).fetchdf()

    # Audit columns — assigned per-address from the Python-side scan.
    def status(addr: str) -> str:
        if addr in FLAGGED_UNCOPYABLE:
            return "flagged_uncopyable"
        if addr in audited_addrs:
            return "audited"
        return "unrun"

    df["audit_status"] = df["address"].apply(status)
    df["n_deployable_cells"] = df["address"].apply(
        lambda a: compute_n_deployable_cells(audits[a]["family_parquet"])
        if a in audits and audits[a]["family_parquet"] else np.nan
    )
    return df


def main() -> None:
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_copyability'")
    con.execute("SET preserve_insertion_order=false")

    log("loading views ...")
    load_views(con)

    log("building universe (qualifying addresses)")
    n_addr = build_universe(con)
    log(f"  qualifying addresses: {n_addr:,}")

    log("building per_position table")
    build_position_table(con)

    log("computing position aggregates (fragmentation, hold-to-res, split, median pnl)")
    build_position_aggregates(con)

    log("computing market aggregates (pct_winning, win/loss ratio)")
    build_market_aggregates(con)

    log("computing family aggregates (HHI concentration, dominant family)")
    build_family_aggregates(con)

    log("computing recent-activity aggregates (active_days_last_90d, vol_30d_ratio)")
    build_recent_activity(con)

    log("joining everything for final output")
    df = join_existing(con)
    log(f"  output rows: {len(df):,}")
    log(f"  audit_status: {df['audit_status'].value_counts().to_dict()}")

    df.to_parquet(OUT_PARQUET, index=False, compression="zstd")
    log(f"wrote {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
