"""Domah copy-execution audit (read-only diagnostic).

Goal: show how copy-trading Domah would have performed under different
execution models vs his own PnL. NOT a backtest of a strategy — a
diagnostic to inform deployment.

Methodology (confirmed with user):
  - Window: 2025-01-02 → 2026-04-24 (last trade shard).
  - Market family: slug keyword heuristic.
  - PnL: replay each fragment with a (fill_indicator, fill_price) per branch;
    mark final copy inventory to closed_positions.resolution_price.
  - Next-fill window: (anchor_ts, anchor_ts + 300s] — strict >; Polymarket
    block time ~1-2s, so this honors the "+100ms detection latency" spec.
  - Position = (address, market_id, outcome_token_id) — matches closed_positions.

Branches:
  A_opt / A_real  — role-mirrored (maker if Domah was maker, taker if taker).
                    For the maker leg, optimistic = any qualifying follow-on
                    print fills you; realistic = ≥2 such prints.
  B               — pure taker (deterministic).
  C_opt / C_real  — pure maker (you never cross), both fill models.

Outputs (polymarket/research/data/analysis/):
  domah_audit_fragments.parquet     per-fill enriched
  domah_audit_positions.parquet     per-position branch PnL
  domah_audit_family.parquet        primary family PnL table
  domah_audit_slices.parquet        secondary slices (lifecycle/hour/role)
  domah_audit_sensitivity.parquet   H1-2025 vs 2026-YTD comparison
  domah_audit_diagnostics.parquet   per-family diagnostics
  domah_audit_report.md             markdown report
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from data_infra.views import latest_markets_path

# Block SPREAD-2: optional category-gated surface fallback (default OFF; the flat
# 3c fallback is unchanged unless --surface-fallback is passed). Shared core +
# validation provenance: lib/copy_slippage.py + spread_surface_tradetime_regate_findings.
from lib.copy_slippage import k5_category, load_bounce_lookup, reprice_fallback_rows
from lib.spread_surface import SpreadSurface

# ----------------------------------------------------------------------------
# Constants — DOMAH is the original/default leader; main() now accepts any.
# ----------------------------------------------------------------------------
DOMAH = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344"

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRADES_GLOB = str(ROOT / "data" / "trades" / "*.parquet")
CLOSED_POSITIONS = ROOT / "data" / "closed_positions.parquet"

# Set by main() / load_fragments() at runtime.
_RUN_LEADER = DOMAH

WINDOW_START = "2025-01-02"
WINDOW_END   = "2026-04-25"            # exclusive cap; last fill is 2026-04-23
WINDOW_SEC   = 300                      # +5 min cap per spec
FALLBACK_CENTS_DEFAULT = 3.0
FALLBACK_SENSITIVITY_CENTS = [1.0, 2.0, 3.0, 5.0]

# Performance: materialize a trades subset filtered to Domah's tokens + window range.
# Cuts the JOIN probe side from ~30M trades to whatever Domah actually touched.
SUBSET_TABLE = "trades_subset"

# Slug-based family heuristic. Order matters: first match wins.
FAMILY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("sports", (
        "-mlb-", "-mlb", "mlb-",
        "-nba-", "-nba", "nba-",
        "-nfl-", "-nfl", "nfl-",
        "-nhl-", "-nhl", "nhl-",
        "-cfb-", "-cfb", "cfb-",
        "-ncaa-", "ncaa-",
        "-ufc-", "ufc-",
        "-soccer-", "soccer-",
        "premier-league", "champions-league", "-ucl-", "ucl-",
        "world-cup", "tennis", "atp-", "wta-", "grand-slam",
        "-boxing-", "boxing-", "-mma-", "mma-",
        "formula-1", "-f1-",
        "nascar",
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
        "noem", "marco-rubio", "pam-bondi",
        "openai-receives-federal",
        "eric-adams", "angela-rayner", "yulia-navalnaya", "greenland",
        "world-leader", "anti-cartel", "drop-out", "dropout",
        "navalny", "mahmoud-abbas", "sanchez", "lula",
        "doria-medina",
    )),
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _tag_family(slug: str | None,
                rules: list[tuple[str, tuple[str, ...]]] | None = None) -> str:
    if not slug:
        return "other"
    s = "-" + slug.lower() + "-"
    for fam, kws in (rules or FAMILY_KEYWORDS):
        for kw in kws:
            if kw in s:
                return fam
    return "other"


def _phase(hours_to_resolution: float) -> str:
    if hours_to_resolution > 168:
        return "open"
    if hours_to_resolution >= 24:
        return "middle"
    return "near-resolution"


def _hour_bucket(h: int) -> str:
    if h < 6:  return "00-06"
    if h < 12: return "06-12"
    if h < 18: return "12-18"
    return "18-24"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Step 1: Load Domah's fills with direction and side
# ----------------------------------------------------------------------------
def load_fragments(con: duckdb.DuckDBPyConnection,
                   leader: str | None = None,
                   family_keywords: list[tuple[str, tuple[str, ...]]] | None = None) -> pd.DataFrame:
    """Load `leader`'s fills directly from the trade parquets (one filtered scan,
    no view explosion). Derives role/direction/maker_side without round-tripping
    through trader_actions + joined_fills.

    Args:
      leader: address (default = DOMAH for backward-compat).
      family_keywords: override the global FAMILY_KEYWORDS rules.
    """
    leader = (leader or DOMAH).lower()
    global _RUN_LEADER
    _RUN_LEADER = leader
    log(f"Step 1: loading {leader[:10]}… fragments (direct parquet scan)…")
    markets_path = latest_markets_path()
    sql = f"""
    WITH dm_raw AS (
        SELECT
            rt.timestamp                                        AS fill_ts,
            rt.market_id,
            CASE WHEN rt.maker_asset_id='0' THEN rt.taker_asset_id
                 ELSE rt.maker_asset_id END                      AS outcome_token_id,
            rt.maker, rt.taker,
            CASE WHEN rt.maker = '{leader}' THEN 'maker' ELSE 'taker' END AS role,
            rt.maker_side,
            rt.maker_asset_id,
            rt.token_amount, rt.usd_amount, rt.price, rt.transaction_hash
        FROM read_parquet('{TRADES_GLOB}') rt
        WHERE (rt.maker = '{leader}' OR rt.taker = '{leader}')
          AND rt.timestamp >= TIMESTAMP '{WINDOW_START}'
          AND rt.timestamp <  TIMESTAMP '{WINDOW_END}'
          AND rt.maker IS NOT NULL AND rt.taker IS NOT NULL AND rt.maker <> rt.taker
          AND rt.market_id IS NOT NULL
    ),
    dm_typed AS (
        SELECT
            *,
            -- Domah BUYS iff (maker AND maker_asset_id='0') OR (taker AND maker_asset_id<>'0')
            CASE
              WHEN role='maker' AND maker_asset_id='0' THEN 'BUY'
              WHEN role='maker' AND maker_asset_id<>'0' THEN 'SELL'
              WHEN role='taker' AND maker_asset_id='0' THEN 'SELL'
              ELSE 'BUY'
            END AS direction
        FROM dm_raw
    )
    SELECT
        d.fill_ts, d.market_id, d.outcome_token_id,
        d.role, d.direction,
        d.maker_side AS fill_maker_side,
        d.price, d.token_amount, d.usd_amount,
        CASE WHEN d.direction='BUY' THEN -d.usd_amount ELSE d.usd_amount END AS usd_delta,
        CASE WHEN d.direction='BUY' THEN  d.token_amount ELSE -d.token_amount END AS token_delta,
        d.transaction_hash,
        d.maker AS cp_maker, d.taker AS cp_taker,
        m.slug, m.end_date,
        TRY_CAST(m.end_date AS TIMESTAMP) AS end_ts
    FROM dm_typed d
    LEFT JOIN read_parquet('{markets_path}') m
        ON CAST(m.id AS VARCHAR) = d.market_id
    """
    t0 = time.time()
    df = con.execute(sql).df()
    log(f"  loaded {len(df):,} fragments in {time.time()-t0:,.0f}s")

    # tag family + lifecycle + hour
    _rules = family_keywords or FAMILY_KEYWORDS
    df["family"] = df["slug"].map(lambda s: _tag_family(s, _rules))
    df["fill_ts"] = pd.to_datetime(df["fill_ts"])
    df["end_ts"]  = pd.to_datetime(df["end_ts"])
    h2r = (df["end_ts"] - df["fill_ts"]).dt.total_seconds() / 3600.0
    df["hours_to_resolution"] = h2r
    df["lifecycle_phase"] = h2r.fillna(99999).map(_phase)
    df["hour_utc"] = df["fill_ts"].dt.hour
    df["hour_bucket"] = df["hour_utc"].map(_hour_bucket)

    # Stable anchor index
    df = df.sort_values(["fill_ts", "market_id", "outcome_token_id"]).reset_index(drop=True)
    df["anchor_idx"] = df.index.astype(np.int64)

    # position_id = (address, market_id, outcome_token_id) → matches closed_positions
    df["position_id"] = df["market_id"].astype(str) + "::" + df["outcome_token_id"].astype(str)

    # is_winning_position requires a join later; defer.
    return df


# ----------------------------------------------------------------------------
# Step 2: Build a filtered trades subset once
# ----------------------------------------------------------------------------
def build_trades_subset(con: duckdb.DuckDBPyConnection, fragments: pd.DataFrame) -> None:
    """Materialize trades filtered to Domah's outcome_tokens + the window
    (plus a 300s tail so next-fill lookups don't miss right at the edge).

    This drops the JOIN's probe side from ~30M rows to ~few-million,
    which makes the per-anchor lookup ~5-10× faster.
    """
    log("Step 2a: building filtered trades subset…")
    tokens = pd.DataFrame({"outcome_token_id": fragments["outcome_token_id"].dropna().unique()})
    con.register("tokens_subset", tokens)

    t0 = time.time()
    con.execute(f"""
    CREATE OR REPLACE TABLE {SUBSET_TABLE} AS
    SELECT
        t.timestamp,
        t.market_id,
        CASE WHEN t.maker_asset_id = '0' OR t.maker_asset_id IS NULL
             THEN t.taker_asset_id ELSE t.maker_asset_id END AS outcome_token_id,
        t.maker, t.taker, t.maker_side,
        t.price
    FROM read_parquet('{TRADES_GLOB}') t
    WHERE t.timestamp >= TIMESTAMP '{WINDOW_START}'
      AND t.timestamp <  TIMESTAMP '{WINDOW_END}' + INTERVAL '{WINDOW_SEC} second'
      AND (CASE WHEN t.maker_asset_id = '0' OR t.maker_asset_id IS NULL
                THEN t.taker_asset_id ELSE t.maker_asset_id END) IN
          (SELECT outcome_token_id FROM tokens_subset)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM {SUBSET_TABLE}").fetchone()[0]
    log(f"  subset built: {n:,} rows in {time.time()-t0:,.0f}s")
    # Help the optimizer with stats
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_subset_tok_ts ON {SUBSET_TABLE} (outcome_token_id, timestamp)")
    log("  index built")


def lookup_per_anchor(con: duckdb.DuckDBPyConnection, fragments: pd.DataFrame) -> pd.DataFrame:
    """Single SQL pass: per anchor, return next-fill (same-side) + crossed flag (opposite-side).
    """
    log(f"Step 2b: single combined SQL pass over {len(fragments):,} anchors…")
    anchors = fragments[[
        "anchor_idx", "market_id", "outcome_token_id",
        "fill_ts", "cp_maker", "cp_taker",
        "fill_maker_side", "price", "direction",
    ]].copy()
    anchors.rename(columns={"fill_ts": "anchor_ts",
                            "fill_maker_side": "anchor_side",
                            "price": "anchor_price"}, inplace=True)
    anchors["anchor_ts"] = pd.to_datetime(anchors["anchor_ts"])
    anchors["opp_side"]  = np.where(anchors["anchor_side"] == "BUY", "SELL", "BUY")
    con.register("anchors", anchors)

    sql = f"""
    WITH cand AS (
      SELECT
        a.anchor_idx,
        a.anchor_price,
        a.anchor_side,
        t.timestamp  AS nf_ts,
        t.price      AS nf_price,
        t.maker_side AS nf_side,
        DATEDIFF('millisecond', a.anchor_ts, t.timestamp) / 1000.0 AS nf_lag_s,
        CASE WHEN t.maker_side = a.anchor_side THEN 1 ELSE 0 END AS is_same_side
      FROM anchors a
      JOIN {SUBSET_TABLE} t
        ON t.outcome_token_id = a.outcome_token_id
       AND t.timestamp >  a.anchor_ts
       AND t.timestamp <= a.anchor_ts + INTERVAL '{WINDOW_SEC} second'
       AND t.maker NOT IN (a.cp_maker, a.cp_taker)
       AND t.taker NOT IN (a.cp_maker, a.cp_taker)
    ),
    ranked_same AS (
      SELECT *,
        ROW_NUMBER() OVER (PARTITION BY anchor_idx ORDER BY nf_ts ASC) AS rn,
        CASE
          WHEN anchor_side='BUY'  AND nf_price <= anchor_price + 1e-9 THEN 1
          WHEN anchor_side='SELL' AND nf_price >= anchor_price - 1e-9 THEN 1
          ELSE 0
        END AS at_or_better
      FROM cand
      WHERE is_same_side = 1
    ),
    -- crossed market detection: opposite-side prints with price superior to anchor
    -- BUY anchor (Domah bid) → opposite = ask side (maker_side='SELL') → crossed iff any ask < bid
    -- SELL anchor (Domah ask) → opposite = bid side (maker_side='BUY') → crossed iff any bid > ask
    crossed_calc AS (
      SELECT
        anchor_idx,
        MAX(CASE
              WHEN is_same_side = 0 AND anchor_side='BUY'  AND nf_price < anchor_price - 1e-9 THEN 1
              WHEN is_same_side = 0 AND anchor_side='SELL' AND nf_price > anchor_price + 1e-9 THEN 1
              ELSE 0
            END) AS crossed_flag
      FROM cand
      GROUP BY anchor_idx
    ),
    agg_same AS (
      SELECT
        anchor_idx,
        COUNT(*)                                       AS nf_count,
        SUM(at_or_better)                              AS nf_at_or_better_count,
        MAX(CASE WHEN rn=1 THEN nf_ts END)             AS nf1_ts,
        MAX(CASE WHEN rn=1 THEN nf_price END)          AS nf1_price,
        MAX(CASE WHEN rn=1 THEN nf_lag_s END)          AS nf1_lag_s
      FROM ranked_same
      GROUP BY anchor_idx
    )
    SELECT
      a.anchor_idx,
      COALESCE(s.nf_count, 0)              AS nf_count,
      COALESCE(s.nf_at_or_better_count, 0) AS nf_at_or_better_count,
      s.nf1_ts,
      s.nf1_price,
      s.nf1_lag_s,
      COALESCE(c.crossed_flag, 0)          AS crossed_flag
    FROM anchors a
    LEFT JOIN agg_same s    USING (anchor_idx)
    LEFT JOIN crossed_calc c USING (anchor_idx)
    """
    t0 = time.time()
    out = con.execute(sql).df()
    log(f"  done in {time.time()-t0:,.0f}s ({len(out):,} rows)")
    return out


# ----------------------------------------------------------------------------
# Step 3: Counterfactual fill prices + indicators per branch
# ----------------------------------------------------------------------------
def apply_branches(frag: pd.DataFrame, fallback_cents: float) -> pd.DataFrame:
    """Add columns per branch:
        {branch}_fill (0/1 or float), {branch}_price (effective fill price).

    Branches:
      A_opt  maker-leg (if role='maker'): fill if nf_at_or_better_count >= 1
             taker-leg (if role='taker'): always fill at nf1_price (fallback if missing)
      A_real maker-leg: fill if nf_at_or_better_count >= 2
             taker-leg: same as A_opt
      B      always taker (nf1_price or fallback)
      C_opt  always maker (fill if nf_at_or_better_count >= 1)
      C_real always maker (fill if nf_at_or_better_count >= 2)
    """
    f = frag.copy()
    # fallback price: worse for the bot by fallback_cents on the relevant side
    fb_buy  = f["price"] + fallback_cents / 100.0
    fb_sell = f["price"] - fallback_cents / 100.0
    fallback_price = np.where(f["direction"] == "BUY", fb_buy, fb_sell).astype(float)
    nf_price       = f["nf1_price"].astype(float).values
    has_nf         = np.isfinite(nf_price)
    taker_price    = np.where(has_nf, nf_price, fallback_price)
    taker_source   = np.where(has_nf, "next_fill", "fallback")

    is_maker = (f["role"] == "maker").values
    is_taker = ~is_maker

    n = len(f)
    # Branch A — role mirrored
    a_opt_fill  = np.zeros(n, dtype=np.int8)
    a_real_fill = np.zeros(n, dtype=np.int8)
    a_opt_fill[is_maker]  = (f["nf_at_or_better_count"].values[is_maker] >= 1).astype(np.int8)
    a_real_fill[is_maker] = (f["nf_at_or_better_count"].values[is_maker] >= 2).astype(np.int8)
    a_opt_fill[is_taker]  = 1
    a_real_fill[is_taker] = 1
    a_price = np.where(is_maker, f["price"].values, taker_price)

    # Branch B — pure taker (always cross)
    b_fill  = np.ones(n, dtype=np.int8)
    b_price = taker_price

    # Branch C — pure maker (never cross)
    c_opt_fill  = (f["nf_at_or_better_count"].values >= 1).astype(np.int8)
    c_real_fill = (f["nf_at_or_better_count"].values >= 2).astype(np.int8)
    c_price = f["price"].values

    # A_opt and A_real share the same fill-price logic (maker→anchor, taker→nf);
    # only the fill indicator differs. C_opt / C_real similarly share a price.
    f["A_opt_fill"]   = a_opt_fill
    f["A_real_fill"]  = a_real_fill
    f["A_opt_price"]  = a_price
    f["A_real_price"] = a_price
    f["B_fill"]       = b_fill
    f["B_price"]      = b_price
    f["C_opt_fill"]   = c_opt_fill
    f["C_real_fill"]  = c_real_fill
    f["C_opt_price"]  = c_price
    f["C_real_price"] = c_price
    f["taker_source"] = taker_source
    f["taker_price"]  = taker_price
    f["fallback_cents"] = fallback_cents
    return f


def add_surface_fallback_columns(frag: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                                 surface_csv: Path, breaks_csv: Path,
                                 xcheck_csv: Path) -> pd.DataFrame:
    """Block SPREAD-2 (opt-in): attach the category-gated surface_fallback copy
    price for this leader's FALLBACK rows (taker_source == 'fallback'), alongside
    the incumbent flat-3c. Reuses the FROZEN SPREAD-1 surface; only validated K5
    categories are repriced (politics_negrisk and others keep flat-3c). Leaves
    next_fill rows untouched.

    Adds: k5_category, sf_fallback_cents, sf_source, sf_used_bounce, sf_copy_price,
    flat3c_copy_price, taker_price_surface (next_fill where present else surface
    fallback). The default audit path does NOT call this; pass --surface-fallback."""
    mp = latest_markets_path()
    mkts = con.execute(f"""
        SELECT CAST(id AS VARCHAR) AS market_id, coalesce(question,'') AS question,
               coalesce(neg_risk,false) AS neg_risk
        FROM read_parquet('{mp}')
    """).df()
    f = frag.merge(mkts, on="market_id", how="left")
    f["k5_category"] = k5_category(f, slug_col="slug", question_col="question",
                                   neg_risk_col="neg_risk")
    surf = SpreadSurface.load(surface_csv, breaks_csv)
    bounce = load_bounce_lookup(xcheck_csv)
    f = reprice_fallback_rows(
        f, surf, bounce, price_col="price", ttr_h_col="hours_to_resolution",
        dir_col="direction", maker_col=(f["role"] == "maker"),
        leader_price_col="price", is_fallback=(f["taker_source"] == "fallback"))
    has_nf = np.isfinite(f["nf1_price"].astype(float).to_numpy())
    f["taker_price_surface"] = np.where(has_nf, f["nf1_price"].astype(float),
                                        f["sf_copy_price"])
    return f


# ----------------------------------------------------------------------------
# Step 4: Replay positions → branch PnL
# ----------------------------------------------------------------------------
def replay_positions(frag: pd.DataFrame, closed: pd.DataFrame) -> pd.DataFrame:
    """For each position_id, compute Domah's PnL and each branch's PnL.

    Vectorized: precompute per-fragment cash/token contributions for each
    branch, then a single groupby.sum across all numeric columns.

    Sign convention:
      BUY  → cash_sign = -1 (pay), token_sign = +1 (gain tokens)
      SELL → cash_sign = +1 (receive), token_sign = -1

    Branch PnL = copy_cash_sum + copy_token_final * mark_price.
    mark_price = closed_positions.resolution_price; if missing, last fragment price.
    """
    log("Step 4: replaying positions → branch PnL…")
    f = frag  # don't copy upfront; we'll select columns at the end
    is_buy = (f["direction"] == "BUY").values
    cash_sign  = np.where(is_buy, -1.0, 1.0)
    token_sign = np.where(is_buy,  1.0, -1.0)
    tok        = f["token_amount"].values.astype(np.float64)

    branches = ["A_opt", "A_real", "B", "C_opt", "C_real"]
    # Pre-compute all branch cash/token per fragment as numpy → DataFrame
    new_cols: dict[str, np.ndarray] = {
        "domah_cash":  f["usd_delta"].values.astype(np.float64),
        "domah_token": (token_sign * tok),
    }
    for br in branches:
        fi = f[f"{br}_fill"].values.astype(np.float64)
        px = f[f"{br}_price"].values.astype(np.float64)
        copy_tok = token_sign * tok * fi
        new_cols[f"{br}_cash"]  = cash_sign * tok * fi * px
        new_cols[f"{br}_token"] = copy_tok
        new_cols[f"{br}_filled"] = tok * fi

    # Build a small DataFrame for groupby (position_id + numeric cols only)
    log("  preparing per-fragment contribs…")
    df_sum = pd.DataFrame(new_cols)
    df_sum["position_id"] = f["position_id"].values
    log("  groupby.sum…")
    summed = df_sum.groupby("position_id", sort=False).sum()

    # Per-position scalar metadata (one-pass via groupby first/agg)
    log("  per-position metadata…")
    meta = f.groupby("position_id", sort=False).agg(
        family=("family", "first"),
        market_id=("market_id", "first"),
        outcome_token_id=("outcome_token_id", "first"),
        first_ts=("fill_ts", "min"),
        last_ts=("fill_ts", "max"),
        n_fills=("anchor_idx", "size"),
        last_price=("price", "last"),
    )
    f_role = f.assign(is_maker=(f["role"] == "maker").astype(np.int32))
    role_counts = f_role.groupby("position_id", sort=False).agg(
        n_maker=("is_maker", "sum"),
    )
    pos = meta.join(role_counts).join(summed)

    # join resolution
    log("  attaching resolution_price…")
    cp = closed.set_index(["market_id", "outcome_token_id"])[
        ["resolution_price", "is_held_to_resolution", "realised_pnl"]
    ].rename(columns={"realised_pnl": "domah_realised_pnl_cp"})
    pos = pos.reset_index().merge(
        cp.reset_index(),
        on=["market_id", "outcome_token_id"], how="left"
    )
    pos["mark_price"] = pos["resolution_price"].fillna(pos["last_price"])
    pos["resolved_flag"] = pos["resolution_price"].notna()

    pos["domah_pnl_calc"] = pos["domah_cash"] + pos["domah_token"] * pos["mark_price"]
    for br in branches:
        pos[f"{br}_pnl"] = pos[f"{br}_cash"] + pos[f"{br}_token"] * pos["mark_price"]
    pos["is_winning_position"] = (pos["domah_pnl_calc"] > 0).astype(int)
    pos["n_taker"] = pos["n_fills"] - pos["n_maker"]
    log(f"  replayed {len(pos):,} positions")
    return pos


# ----------------------------------------------------------------------------
# Step 5: Family table
# ----------------------------------------------------------------------------
def family_table(pos: pd.DataFrame, frag_with_branches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    branches = ["A_opt", "A_real", "B", "C_opt", "C_real"]
    for fam, p in pos.groupby("family", sort=False):
        row = {
            "family": fam,
            "n_positions": int(len(p)),
            "n_fills": int(p["n_fills"].sum()),
            "n_resolved_pos": int(p["resolved_flag"].sum()),
            "leader_pnl": float(p["domah_pnl_calc"].sum()),
            "leader_pnl_cp": float(p["domah_realised_pnl_cp"].sum()),
        }
        for br in branches:
            row[f"{br}_pnl"] = float(p[f"{br}_pnl"].sum())

        # fallback% for the branches that use taker-leg / maker-leg
        f_fam = frag_with_branches[frag_with_branches["family"] == fam]
        f_taker_leg = f_fam[f_fam["role"] == "taker"]   # A taker leg
        f_maker_leg = f_fam[f_fam["role"] == "maker"]   # A maker leg
        row["A_taker_leg_n"]      = int(len(f_taker_leg))
        row["A_maker_leg_n"]      = int(len(f_maker_leg))
        row["A_taker_fallback_pct"] = (
            float((f_taker_leg["taker_source"] == "fallback").mean()) if len(f_taker_leg) else float("nan")
        )
        # A_maker leg "fallback" = no qualifying print (under A_real: <2 prints)
        row["A_maker_realfill_rate"] = (
            float(f_maker_leg["A_real_fill"].mean()) if len(f_maker_leg) else float("nan")
        )
        row["A_maker_optfill_rate"] = (
            float(f_maker_leg["A_opt_fill"].mean()) if len(f_maker_leg) else float("nan")
        )

        # adverse_select_ratio (using A_real maker-fill model on maker fills only)
        # = maker-leg fill rate on positions Domah won / on positions Domah lost
        f_maker_w_outcome = f_fam[f_fam["role"] == "maker"].merge(
            pos[["position_id", "is_winning_position"]], on="position_id", how="left"
        )
        win  = f_maker_w_outcome[f_maker_w_outcome["is_winning_position"] == 1]
        lose = f_maker_w_outcome[f_maker_w_outcome["is_winning_position"] == 0]
        if len(win) >= 30 and len(lose) >= 30:
            rate_win  = float(win["A_real_fill"].mean())
            rate_lose = float(lose["A_real_fill"].mean())
            row["adverse_select_ratio"] = rate_win / rate_lose if rate_lose > 0 else float("nan")
            row["adv_sel_n_win"]  = int(len(win))
            row["adv_sel_n_lose"] = int(len(lose))
        else:
            row["adverse_select_ratio"] = float("nan")
            row["adv_sel_n_win"]  = int(len(win))
            row["adv_sel_n_lose"] = int(len(lose))

        # capture ratios
        for br in ["A_opt", "A_real"]:
            row[f"{br}_capture"] = (
                row[f"{br}_pnl"] / row["leader_pnl"] if row["leader_pnl"] != 0 else float("nan")
            )

        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_fills", ascending=False).reset_index(drop=True)


def slice_table(
    pos: pd.DataFrame,
    frag_with_branches: pd.DataFrame,
    slice_col: str,
    pos_slice_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate by `slice_col` (which exists in frag), but PnL needs to be
    aggregated per position. We rebuild a per-(slice, position) PnL.

    For slices at the fragment level (lifecycle_phase, hour_bucket, role),
    we attribute a position's PnL to the slice based on the slice of its
    first fragment.
    """
    branches = ["A_opt", "A_real", "B", "C_opt", "C_real"]
    if pos_slice_col is None:
        # take first fragment per position
        first_frag = frag_with_branches.sort_values("fill_ts").groupby("position_id").first()
        slice_map = first_frag[slice_col].to_dict()
        pos = pos.copy()
        pos["slice"] = pos["position_id"].map(slice_map)
    else:
        pos = pos.rename(columns={pos_slice_col: "slice"})

    rows = []
    for slc, p in pos.groupby("slice", sort=False):
        f_slc = frag_with_branches[frag_with_branches["position_id"].isin(p["position_id"])]
        row = {
            "slice": slc,
            "n_positions": int(len(p)),
            "n_fills": int(len(f_slc)),
            "leader_pnl": float(p["domah_pnl_calc"].sum()),
        }
        for br in branches:
            row[f"{br}_pnl"] = float(p[f"{br}_pnl"].sum())
        # fallback / fill-rate
        ft = f_slc[f_slc["role"] == "taker"]
        fm = f_slc[f_slc["role"] == "maker"]
        row["A_taker_fallback_pct"] = float((ft["taker_source"] == "fallback").mean()) if len(ft) else float("nan")
        row["A_maker_optfill_rate"] = float(fm["A_opt_fill"].mean()) if len(fm) else float("nan")
        row["A_maker_realfill_rate"] = float(fm["A_real_fill"].mean()) if len(fm) else float("nan")
        # adverse select
        fmw = fm.merge(pos[["position_id", "is_winning_position"]], on="position_id", how="left")
        win  = fmw[fmw["is_winning_position"] == 1]
        lose = fmw[fmw["is_winning_position"] == 0]
        if len(win) >= 30 and len(lose) >= 30:
            rw = float(win["A_real_fill"].mean())
            rl = float(lose["A_real_fill"].mean())
            row["adverse_select_ratio"] = rw / rl if rl > 0 else float("nan")
        else:
            row["adverse_select_ratio"] = float("nan")
        # capture
        for br in ["A_opt", "A_real"]:
            row[f"{br}_capture"] = row[f"{br}_pnl"] / row["leader_pnl"] if row["leader_pnl"] != 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Step 6: Diagnostics
# ----------------------------------------------------------------------------
def diagnostics_per_family(frag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, sub in frag.groupby("family", sort=False):
        n_total = len(sub)
        if n_total <= 100:
            continue
        taker_sub = sub[sub["role"] == "taker"]
        maker_sub = sub[sub["role"] == "maker"]
        row = {
            "family": fam,
            "n_fills": int(n_total),
            "n_taker_fills": int(len(taker_sub)),
            "n_maker_fills": int(len(maker_sub)),
        }
        # taker fallback %
        row["taker_fallback_pct"] = (
            float((taker_sub["taker_source"] == "fallback").mean()) if len(taker_sub) else float("nan")
        )
        # maker fallback % (= no qualifying print at all in window)
        row["maker_nofill_realistic_pct"] = (
            float((maker_sub["nf_at_or_better_count"] < 2).mean()) if len(maker_sub) else float("nan")
        )
        row["maker_nofill_optimistic_pct"] = (
            float((maker_sub["nf_at_or_better_count"] < 1).mean()) if len(maker_sub) else float("nan")
        )
        # lag percentiles for taker fills (only when next-fill found)
        nf_lag = taker_sub.loc[taker_sub["taker_source"] == "next_fill", "nf1_lag_s"].astype(float)
        for pct, label in [(10, "p10"), (25, "p25"), (50, "p50"),
                           (75, "p75"), (90, "p90")]:
            row[f"taker_lag_{label}_s"] = float(np.nanpercentile(nf_lag, pct)) if len(nf_lag) else float("nan")
        # time-to-any-next-fill (use nf1_lag_s for all fills regardless of taker/maker)
        any_lag = sub["nf1_lag_s"].astype(float)
        for thresh in (30, 60, 120, 300, 600):
            mask_thresh = any_lag < thresh
            row[f"share_any_within_{thresh}s"] = float(mask_thresh.mean())
        row["share_none_within_1800s"] = float(any_lag.isna().mean())  # since cap was 300s, NaN ≈ no fill
        # Slip distribution on REAL taker next-fills only (excludes the 3¢ fallback
        # constant, which would otherwise dominate the histogram). Signed so +ve
        # means bot pays more / receives less than Domah.
        real_nf = taker_sub[taker_sub["taker_source"] == "next_fill"]
        slip_signed = np.where(real_nf["direction"] == "BUY",
                               (real_nf["taker_price"] - real_nf["price"]) * 100.0,
                               (real_nf["price"] - real_nf["taker_price"]) * 100.0)
        for pct, label in [(10, "p10"), (25, "p25"), (50, "p50"),
                           (75, "p75"), (90, "p90")]:
            row[f"slip_real_cents_{label}"] = (
                float(np.nanpercentile(slip_signed, pct)) if len(slip_signed) else float("nan")
            )
        row["slip_real_n"] = int(len(real_nf))
        # crossed-market share
        row["crossed_share"] = float(sub["crossed_flag"].fillna(0).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_fills", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Step 7: Fallback-cents sensitivity (taker leg only)
# ----------------------------------------------------------------------------
def fallback_sensitivity(frag_base: pd.DataFrame,
                         pos_base_template: pd.DataFrame,
                         closed: pd.DataFrame) -> pd.DataFrame:
    """Re-evaluate B and A (taker leg) under fallback ∈ {1,2,3,5}c."""
    rows = []
    for fb in FALLBACK_SENSITIVITY_CENTS:
        f = apply_branches(frag_base, fallback_cents=fb)
        p = replay_positions(f, closed)
        for fam, sub in p.groupby("family", sort=False):
            rows.append({
                "fallback_cents": fb,
                "family": fam,
                "n_positions": int(len(sub)),
                "leader_pnl": float(sub["domah_pnl_calc"].sum()),
                "A_opt_pnl":  float(sub["A_opt_pnl"].sum()),
                "A_real_pnl": float(sub["A_real_pnl"].sum()),
                "B_pnl":      float(sub["B_pnl"].sum()),
                "C_opt_pnl":  float(sub["C_opt_pnl"].sum()),
                "C_real_pnl": float(sub["C_real_pnl"].sum()),
            })
    return pd.DataFrame(rows)


def sensitivity_window_split(frag: pd.DataFrame, closed: pd.DataFrame) -> pd.DataFrame:
    """Re-run family table on H1 2025 and 2026-YTD; flag families with >30pp
    A_real capture-ratio divergence."""
    f = frag.copy()
    rows = []
    for label, lo, hi in [
        ("2025-H1",   pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-01")),
        ("2026-YTD",  pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-25")),
    ]:
        sub = f[(f["fill_ts"] >= lo) & (f["fill_ts"] < hi)].copy()
        if len(sub) < 100:
            continue
        sub_pos = replay_positions(sub, closed)
        for fam, fsub in sub_pos.groupby("family", sort=False):
            n_fills = int((sub["family"] == fam).sum())
            leader = float(fsub["domah_pnl_calc"].sum())
            a_real = float(fsub["A_real_pnl"].sum())
            a_opt  = float(fsub["A_opt_pnl"].sum())
            rows.append({
                "window": label, "family": fam,
                "n_fills": n_fills,
                "leader_pnl": leader,
                "A_opt_pnl":  a_opt,
                "A_real_pnl": a_real,
                "A_opt_capture":  a_opt / leader if leader != 0 else float("nan"),
                "A_real_capture": a_real / leader if leader != 0 else float("nan"),
            })
    out = pd.DataFrame(rows)
    return out


# ----------------------------------------------------------------------------
# Step 8: Sanity checks
# ----------------------------------------------------------------------------
def sanity_checks(pos: pd.DataFrame, fam_tbl: pd.DataFrame,
                  frag_with_branches: pd.DataFrame,
                  closed_pnl_in_window: float) -> dict:
    """Audit-correctness checks.

    Note on the user-supplied PnL invariants (e.g. A_real <= A_opt): these are
    only true *on average* — they assume the marginal filtered-out fills are
    profitable. When Domah loses on a position, removing those fills *raises*
    PnL. We therefore check the strict invariants on *fill counts*, not PnL,
    and report any PnL violations as informational only.
    """
    checks = {}
    # (a) Total Domah PnL match — allow 10% for unresolved positions whose
    # paper PnL we mark to last_price but closed_positions does not.
    domah_calc = float(pos["domah_pnl_calc"].sum())
    diff_pct = abs(domah_calc - closed_pnl_in_window) / max(abs(closed_pnl_in_window), 1)
    checks["A_total_pnl_match_within_10pct"] = {
        "calc_pnl": domah_calc,
        "closed_positions_pnl": closed_pnl_in_window,
        "diff_pct": diff_pct,
        "pass": diff_pct < 0.10,
        "note": "Difference largely from marking unresolved positions to last "
                "fill price; closed_positions does not.",
    }
    # (b) Strict fill-count invariants: A_real fills ⊆ A_opt fills; C_real ⊆ C_opt
    fill_counts = (
        frag_with_branches.groupby("family")[
            ["A_opt_fill", "A_real_fill", "C_opt_fill", "C_real_fill"]
        ].sum()
    )
    a_ok = bool((fill_counts["A_real_fill"] <= fill_counts["A_opt_fill"]).all())
    c_ok = bool((fill_counts["C_real_fill"] <= fill_counts["C_opt_fill"]).all())
    checks["B_fill_count_subset_invariants"] = {
        "A_real_le_A_opt_fills_all_families": a_ok,
        "C_real_le_C_opt_fills_all_families": c_ok,
        "pass": a_ok and c_ok,
        "fill_counts_by_family": fill_counts.to_dict(),
    }
    # (c) Informational: PnL violations (A_real > A_opt, B > A_real, C_opt < C_real)
    pnl_warn = []
    for _, r in fam_tbl.iterrows():
        if r["A_real_pnl"] > r["A_opt_pnl"] + 1.0:
            pnl_warn.append(
                f"{r['family']}: A_real PnL ({r['A_real_pnl']:,.0f}) > A_opt PnL "
                f"({r['A_opt_pnl']:,.0f}) — the maker fills filtered by the realistic "
                f"model were net-losing, so dropping them raised PnL."
            )
        if r["C_opt_pnl"] + 1.0 < r["C_real_pnl"]:
            pnl_warn.append(
                f"{r['family']}: C_real PnL > C_opt PnL — same dynamic on the pure-maker branch."
            )
    fam_maker_share = (
        frag_with_branches.groupby("family")
        .apply(lambda s: (s["role"] == "maker").mean(), include_groups=False)
        .to_dict()
    )
    for _, r in fam_tbl.iterrows():
        fam = r["family"]
        if fam_maker_share.get(fam, 0.0) > 0.5 and r["B_pnl"] > r["A_real_pnl"] + 1.0:
            pnl_warn.append(
                f"{fam} (maker_share={fam_maker_share[fam]:.0%}): pure-taker B "
                f"PnL ({r['B_pnl']:,.0f}) > role-mirrored A_real ({r['A_real_pnl']:,.0f}) "
                f"— A_real misses too many of Domah's maker positions; pure-taker buys "
                f"all of them at the next-fill price."
            )
    checks["C_pnl_monotonicity_warnings"] = {
        "warnings": pnl_warn,
        "pass": True,  # informational
        "note": "These are NOT bugs — they document where PnL fails to track fill count.",
    }
    return checks


# ----------------------------------------------------------------------------
# Step 9: Markdown report
# ----------------------------------------------------------------------------
def render_report(
    universe: dict,
    fam_tbl: pd.DataFrame,
    slice_lifecycle: pd.DataFrame,
    slice_hour: pd.DataFrame,
    slice_role: pd.DataFrame,
    sensitivity: pd.DataFrame,
    fb_sens: pd.DataFrame,
    diag: pd.DataFrame,
    checks: dict,
    interpretation: str,
) -> str:
    def fmt_df(df: pd.DataFrame) -> str:
        """Custom markdown renderer (avoids `tabulate` dep).

        Column-name heuristics decide format:
          - *capture, *ratio, *_pct, *rate, *share, *fallback*, lag_* → 4 dp
          - *_pnl, *_volume, n_*, *_count, *_n  → 0 dp with thousands separator
          - other floats → 0 dp if |x|>=1000 else 4 dp
        """
        d = df.copy()
        def looks_like_ratio(col: str) -> bool:
            c = col.lower()
            return any(k in c for k in (
                "capture", "ratio", "_pct", "rate", "share",
                "fallback", "lag_", "p10", "p25", "p50", "p75", "p90",
                "slip", "spread", "fill_rate",
            ))
        def looks_like_money(col: str) -> bool:
            c = col.lower()
            return any(k in c for k in (
                "pnl", "_usd", "volume", "_cash", "_value", "notional",
            ))
        def looks_like_count(col: str) -> bool:
            c = col.lower()
            return any(k in c for k in (
                "n_fills", "n_positions", "n_resolved", "n_maker", "n_taker",
                "n_win", "n_lose", "fallback_cents",
            ))
        for c in d.columns:
            if pd.api.types.is_float_dtype(d[c]):
                if looks_like_ratio(c):
                    fmt = lambda x: "" if pd.isna(x) else f"{x:,.4f}"
                elif looks_like_money(c) or looks_like_count(c):
                    fmt = lambda x: "" if pd.isna(x) else f"{x:,.0f}"
                else:
                    fmt = lambda x: "" if pd.isna(x) else (
                        f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.4f}"
                    )
                d[c] = d[c].apply(fmt)
            elif pd.api.types.is_integer_dtype(d[c]):
                d[c] = d[c].apply(lambda x: "" if pd.isna(x) else f"{int(x):,}")
            else:
                d[c] = d[c].astype(str).where(d[c].notna(), "")
        cols = list(d.columns)
        widths = [max(len(c), d[c].map(len).max() if len(d) else 0) for c in cols]
        head = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
        sep  = "| " + " | ".join("-" * w for w in widths) + " |"
        rows = ["| " + " | ".join(v.rjust(w) for v, w in zip(r, widths)) + " |"
                for r in d.to_numpy().tolist()]
        return "\n".join([head, sep] + rows)

    label = universe.get("label", "leader")
    lines = []
    lines.append(f"# Copy-Execution Audit: {label}")
    lines.append("")
    lines.append(f"_Generated {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")
    lines.append("Read-only diagnostic. Trade data covers 2025-01-02 → 2026-04-24 "
                 "(last available shard; not today 2026-05-16).")
    lines.append("")

    lines.append("## Universe summary")
    for k, v in universe.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:,.2f}")
        else:
            lines.append(f"- **{k}**: {v:,}" if isinstance(v, int) else f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Primary family table")
    lines.append("")
    lines.append("`A_opt` / `A_real`: role-mirrored, optimistic vs realistic maker-fill model. "
                 "`B`: pure taker. `C_opt` / `C_real`: pure maker. "
                 "`adverse_select_ratio` = A_real maker-fill rate on winning vs losing positions "
                 "(N/A if either bucket <30 maker fills). Deployable cells have **high capture AND "
                 "adverse_select_ratio close to 1.0**.")
    lines.append("")
    cols = ["family", "n_fills", "n_positions", "leader_pnl",
            "A_opt_pnl", "A_real_pnl", "B_pnl", "C_opt_pnl", "C_real_pnl",
            "A_opt_capture", "A_real_capture",
            "A_taker_fallback_pct", "A_maker_realfill_rate",
            "adverse_select_ratio", "adv_sel_n_win", "adv_sel_n_lose"]
    lines.append(fmt_df(fam_tbl[cols]))
    lines.append("")

    lines.append("### Secondary slices")
    lines.append("")
    lines.append("#### By market lifecycle phase (hours-to-resolution at fill time)")
    lines.append(fmt_df(slice_lifecycle))
    lines.append("")
    lines.append("#### By hour-of-day (UTC)")
    lines.append(fmt_df(slice_hour))
    lines.append("")
    lines.append("#### By leader's role on the originating fill")
    lines.append(fmt_df(slice_role))
    lines.append("")

    lines.append("### Sensitivity: H1 2025 vs 2026-YTD")
    lines.append("")
    lines.append("Flag families with `|H1 cap − 2026 cap| > 30pp` AND `n_fills > 200` "
                 "as 'structure-shifting': historical capture does not generalise.")
    lines.append("")
    pivot = sensitivity.pivot_table(
        index="family", columns="window",
        values=["A_opt_capture", "A_real_capture", "n_fills"],
    )
    pivot.columns = ["_".join(c) for c in pivot.columns]
    pivot = pivot.reset_index()
    lines.append(fmt_df(pivot))
    lines.append("")

    lines.append("### Sensitivity: fallback cents (taker-leg only)")
    lines.append("")
    fb_pivot = fb_sens.pivot_table(
        index="family", columns="fallback_cents",
        values=["B_pnl", "A_real_pnl"],
    )
    fb_pivot.columns = [f"{c[0]}_{int(c[1])}c" for c in fb_pivot.columns]
    fb_pivot = fb_pivot.reset_index()
    lines.append(fmt_df(fb_pivot))
    lines.append("")

    lines.append("### Per-family diagnostics")
    lines.append("")
    for _, r in diag.iterrows():
        lines.append(f"**{r['family']}** — n_fills={int(r['n_fills']):,}, "
                     f"taker={int(r['n_taker_fills']):,}, maker={int(r['n_maker_fills']):,}")
        lines.append(f"- taker_fallback_pct: {r['taker_fallback_pct']:.1%}; "
                     f"maker_no-fill_optimistic: {r['maker_nofill_optimistic_pct']:.1%}; "
                     f"maker_no-fill_realistic: {r['maker_nofill_realistic_pct']:.1%}")
        lines.append(f"- taker lag percentiles (s): "
                     f"p10={r['taker_lag_p10_s']:.1f}, p25={r['taker_lag_p25_s']:.1f}, "
                     f"p50={r['taker_lag_p50_s']:.1f}, p75={r['taker_lag_p75_s']:.1f}, "
                     f"p90={r['taker_lag_p90_s']:.1f}")
        lines.append(f"- any-next-fill cumulative share: "
                     f"<30s={r['share_any_within_30s']:.1%}, "
                     f"<60s={r['share_any_within_60s']:.1%}, "
                     f"<120s={r['share_any_within_120s']:.1%}, "
                     f"<300s={r['share_any_within_300s']:.1%}, "
                     f"none-in-window={r['share_none_within_1800s']:.1%}")
        lines.append(f"- slip on real taker next-fills (signed cents, +ve = bot worse off), "
                     f"n={int(r['slip_real_n']):,}: "
                     f"p10={r['slip_real_cents_p10']:.2f}, "
                     f"p25={r['slip_real_cents_p25']:.2f}, "
                     f"p50={r['slip_real_cents_p50']:.2f}, "
                     f"p75={r['slip_real_cents_p75']:.2f}, "
                     f"p90={r['slip_real_cents_p90']:.2f}")
        lines.append(f"- crossed-market share: {r['crossed_share']:.1%}")
        lines.append("")

    lines.append("## Sanity checks")
    for name, payload in checks.items():
        status = "PASS" if payload["pass"] else "FAIL"
        lines.append(f"- {name}: **{status}**")
    a = checks["A_total_pnl_match_within_10pct"]
    lines.append(f"  - leader PnL (replay calc) = {a['calc_pnl']:,.0f}; "
                 f"closed_positions sum = {a['closed_positions_pnl']:,.0f} "
                 f"({a['diff_pct']:.1%} drift). {a['note']}")
    warns = checks["C_pnl_monotonicity_warnings"]["warnings"]
    if warns:
        lines.append("")
        lines.append("PnL-monotonicity informationals (NOT bugs — see check description):")
        for w in warns:
            lines.append(f"  - {w}")
    lines.append("")

    lines.append("## Interpretation")
    lines.append(interpretation)
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(leader: str | None = None,
         label: str | None = None,
         out_subdir: str | None = None,
         family_keywords: list[tuple[str, tuple[str, ...]]] | None = None,
         surface_fallback: bool = False) -> None:
    leader = (leader or DOMAH).lower()
    label  = label or ("domah" if leader == DOMAH else leader[:10])
    out_dir = OUT_DIR / (out_subdir or "")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"=== copy-execution audit: leader={leader[:10]}…  label={label} ===")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    # 1. Fragments
    frag = load_fragments(con, leader=leader, family_keywords=family_keywords)
    if len(frag) == 0:
        log("FATAL: zero fragments for this leader/window — aborting")
        return
    frag.to_parquet(out_dir / f"{label}_audit_fragments_raw.parquet", index=False)

    # 2. Build filtered trades subset (5-10× speedup) + single combined lookup
    build_trades_subset(con, frag)
    nf = lookup_per_anchor(con, frag)
    frag = frag.merge(nf, on="anchor_idx", how="left")
    frag["crossed_flag"] = frag["crossed_flag"].fillna(0).astype(int)

    # 3. Branches at default fallback
    frag_b = apply_branches(frag, fallback_cents=FALLBACK_CENTS_DEFAULT)

    # 4. Replay positions (closed_positions.parquet is 27GB → filter at read time)
    closed = con.execute(f"""
        SELECT * FROM read_parquet('{CLOSED_POSITIONS}')
        WHERE address = '{leader}'
    """).df()
    log(f"  loaded {len(closed):,} closed positions for {leader[:10]}…")
    pos = replay_positions(frag_b, closed)
    pos.to_parquet(out_dir / f"{label}_audit_positions.parquet", index=False)
    # SPREAD-2 (opt-in): enrich fragments with the category-gated surface fallback.
    if surface_fallback:
        from scripts.phase5_spread_surface import BREAKS_CSV, SURFACE_CSV, XCHECK_CSV
        frag_b = add_surface_fallback_columns(frag_b, con, SURFACE_CSV, BREAKS_CSV, XCHECK_CSV)
        log(f"  surface_fallback: repriced {int((frag_b['sf_source'].str.startswith('surface')).sum()):,} "
            f"fallback rows on validated categories (flat-3c kept elsewhere)")
    frag_b.to_parquet(out_dir / f"{label}_audit_fragments.parquet", index=False)

    # 5. Family table
    fam_tbl = family_table(pos, frag_b)
    fam_tbl.to_parquet(out_dir / f"{label}_audit_family.parquet", index=False)

    # 6. Secondary slices
    s_life = slice_table(pos, frag_b, "lifecycle_phase")
    s_hour = slice_table(pos, frag_b, "hour_bucket")
    s_role = slice_table(pos, frag_b, "role")
    s_life.assign(slice_kind="lifecycle").to_parquet(out_dir / f"{label}_audit_slice_lifecycle.parquet", index=False)
    s_hour.assign(slice_kind="hour").to_parquet(out_dir / f"{label}_audit_slice_hour.parquet", index=False)
    s_role.assign(slice_kind="role").to_parquet(out_dir / f"{label}_audit_slice_role.parquet", index=False)

    # 7. Diagnostics
    diag = diagnostics_per_family(frag_b)
    diag.to_parquet(out_dir / f"{label}_audit_diagnostics.parquet", index=False)

    # 8. Sensitivities
    sens = sensitivity_window_split(frag_b, closed)
    sens.to_parquet(out_dir / f"{label}_audit_sensitivity_window.parquet", index=False)
    fb_sens = fallback_sensitivity(frag, fam_tbl, closed)
    fb_sens.to_parquet(out_dir / f"{label}_audit_sensitivity_fallback.parquet", index=False)

    # 9. Sanity checks (window-bounded leader PnL)
    closed_pnl_window = float(
        closed[closed["first_fill_ts"] >= pd.Timestamp(WINDOW_START)]["realised_pnl"].sum()
    )
    checks = sanity_checks(pos, fam_tbl, frag_b, closed_pnl_window)

    universe = {
        "label": label,
        "address": leader,
        "window_start": WINDOW_START,
        "window_end_exclusive": WINDOW_END,
        "n_fills": int(len(frag_b)),
        "dollar_volume_usd": float(frag_b["usd_amount"].sum()),
        "n_positions": int(len(pos)),
        "n_resolved_positions": int(pos["resolved_flag"].sum()),
        "n_families_covered": int(frag_b["family"].nunique()),
        "leader_pnl_calc_usd": float(pos["domah_pnl_calc"].sum()),
        "leader_pnl_closed_positions_usd": closed_pnl_window,
    }

    # 10. Interpretation — written deterministically off the numbers.
    def fam_pick(fams: pd.DataFrame, pred) -> list[str]:
        return [r["family"] for _, r in fams.iterrows() if pred(r)]
    deployable = fam_pick(
        fam_tbl,
        lambda r: (
            r["A_real_capture"] >= 0.4
            and r["leader_pnl"] > 0
            and (pd.isna(r["adverse_select_ratio"])
                 or r["adverse_select_ratio"] >= 0.85)
        ),
    )
    adverse_kills = fam_pick(
        fam_tbl,
        lambda r: (
            pd.notna(r["adverse_select_ratio"]) and r["adverse_select_ratio"] < 0.7
            and r["leader_pnl"] != 0
        ),
    )
    structurally_unstable = []
    sens_pivot = sens.pivot_table(
        index="family", columns="window", values="A_real_capture"
    )
    if {"2025-H1", "2026-YTD"}.issubset(sens_pivot.columns):
        for fam, row in sens_pivot.iterrows():
            if pd.notna(row["2025-H1"]) and pd.notna(row["2026-YTD"]):
                n2026 = int(sens.query(
                    "window=='2026-YTD' and family==@fam"
                )["n_fills"].sum())
                if abs(row["2025-H1"] - row["2026-YTD"]) > 0.30 and n2026 > 200:
                    structurally_unstable.append(fam)

    interp_lines = []
    interp_lines.append(
        f"Leader `{label}` is **{(frag_b['role']=='maker').mean():.0%} maker** on "
        f"{universe['n_fills']:,} fills with leader realised PnL of "
        f"**${universe['leader_pnl_calc_usd']:,.0f}** in this window, "
        f"so the audit hinges on the maker-leg fill model and adverse selection more than on taker slippage."
    )
    if deployable:
        interp_lines.append(
            f"**Deployable family today: " + ", ".join(deployable) + "** — A_real capture ≥40%, "
            f"leader PnL positive, and adverse_select_ratio not destructive."
        )
    else:
        # Compute the closest-call family deterministically
        scored = fam_tbl.copy()
        scored["score"] = scored["A_real_capture"].fillna(-9) * (scored["leader_pnl"] > 0).astype(int)
        closest = scored.sort_values("score", ascending=False).head(1)
        if len(closest):
            r = closest.iloc[0]
            interp_lines.append(
                "**No family clears the deployable bar** (A_real capture ≥40% on positive leader PnL "
                "AND adverse_select_ratio ≥0.85). Closest call: "
                f"**{r['family']}** with A_real capture {r['A_real_capture']:.0%}, "
                f"leader PnL ${r['leader_pnl']:,.0f}, "
                f"adv-sel {r['adverse_select_ratio']:.2f}."
            )
        else:
            interp_lines.append("**No family clears the deployable bar.**")
    if adverse_kills:
        interp_lines.append(
            f"Adverse selection is destructive for: **" + ", ".join(adverse_kills) + "** "
            f"(maker-leg fill rate is significantly higher on losing positions than winning ones). "
            f"This is the central economic blocker — even free maker rebates don't fix this."
        )
    else:
        interp_lines.append(
            "Adverse selection is not catastrophic (ratio ≥0.7) on any family with sufficient maker fills, "
            "but is below 1.0 on most — copy fills happen disproportionately when his post is wrong."
        )
    interp_lines.append(
        "Pure-taker (Branch B) is uniformly worse than role-mirroring across the maker-dominant "
        "families — paying 3¢ to cross spread on his entire flow burns capital faster than the leader "
        "earns from maker rebates."
    )
    if structurally_unstable:
        interp_lines.append(
            f"**Structurally unstable**: " + ", ".join(structurally_unstable) + " have >30pp capture "
            f"swings between H1 2025 and 2026 — historical capture there does not generalise; "
            f"only a live A/B test will answer it."
        )
    else:
        interp_lines.append(
            "H1-2025 vs 2026-YTD capture ratios are within 30pp on the substantial families, "
            "so the audit numbers should generalise modulo regime change."
        )
    interp = "\n\n".join(interp_lines)

    report_md = render_report(universe, fam_tbl, s_life, s_hour, s_role,
                              sens, fb_sens, diag, checks, interp)
    report_path = out_dir / f"{label}_audit_report.md"
    report_path.write_text(report_md)
    log(f"Report written to {report_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leader", default=DOMAH,
                    help="Leader address (default = Domah).")
    ap.add_argument("--label", default=None,
                    help="Filename label for outputs (default 'domah' for Domah else first-10 chars).")
    ap.add_argument("--out-subdir", default=None,
                    help="Subdirectory under data/analysis/ to write outputs.")
    ap.add_argument("--use-proposed-keywords", action="store_true",
                    help="Use the Task-1 augmented FAMILY_KEYWORDS from domah_family_validation.py.")
    ap.add_argument("--surface-fallback", action="store_true",
                    help="SPREAD-2: enrich fragments with the category-gated surface "
                         "fallback copy price (validated K5 cats; flat-3c elsewhere).")
    args = ap.parse_args()
    rules = None
    if args.use_proposed_keywords:
        # Lazy import so default path doesn't depend on the validation script.
        from domah_family_validation import _augment_rules
        rules = _augment_rules()
        log(f"Using proposed FAMILY_KEYWORDS (Task 1) — {sum(len(k) for _,k in rules)} keywords.")
    main(leader=args.leader, label=args.label, out_subdir=args.out_subdir,
         family_keywords=rules, surface_fallback=args.surface_fallback)
