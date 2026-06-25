"""Trade-anchored effective-spread surface — loader + predict API (Block SPREAD-1).

The surface estimates the HALF-spread (in cents, price units are 0-1 dollars so
1c = 0.01) a rational taker pays crossing the book, as a function of market
state. It is built from historical OrderFilled fills joined to lookahead-free
CLOB ``/prices-history`` midpoints by ``scripts/spread_surface_build.py`` and
validated against true L1 from ``data/live_clob/`` captures.

What this module owns (pure, importable, unit-testable):
  * bucket definitions + assignment helpers (price level, time-to-resolution,
    trailing trade-rate quartile),
  * the market-category taxonomy (verbatim port of the Block K5 slug/question
    CASE so tape-side SQL and capture-side Python agree),
  * the aggressor-sign convention for historical fills,
  * ``SpreadSurface`` — loads the built surface CSV and predicts a half-spread
    for a market state with an explicit fallback chain.

What it does NOT claim: depth, queue position, adverse selection. The surface
is an L1-touch cost model only (see the findings note's realism ledger).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# Constants — pre-registered design (Block SPREAD-1)
# ----------------------------------------------------------------------------
# Half-spread predictions never go below half a tick; PM books mostly quote
# 1c ticks at mid-range prices (tick floor pre-registered at 0.5c).
TICK_FLOOR_CENTS: float = 0.5

# Price-level buckets (pre-registered). Edges in probability units.
PRICE_BUCKET_EDGES: list[float] = [0.0, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1.0]
PRICE_BUCKET_LABELS: list[str] = [
    "p_lt_05", "p_05_15", "p_15_35", "p_35_65", "p_65_85", "p_85_95", "p_gt_95",
]

# Time-to-resolution buckets (hours). Chosen to split the natural market
# families: 4h crypto (<6h), daily (<24h), game lines (<7d), monthlies (<30d),
# outrights (30d+). "unknown" when end_date is missing.
TTR_BUCKET_EDGES_H: list[float] = [0.0, 6.0, 24.0, 168.0, 720.0, math.inf]
TTR_BUCKET_LABELS: list[str] = ["ttr_lt_6h", "ttr_6_24h", "ttr_1_7d", "ttr_7_30d", "ttr_gt_30d"]
TTR_UNKNOWN: str = "ttr_unknown"

# Activity quartile labels. Breakpoints are DATA-DERIVED at surface-build time
# (per category, on the build sample) and stored alongside the surface.
ACTIVITY_LABELS: list[str] = ["act_q1", "act_q2", "act_q3", "act_q4"]

# Trailing window for the trade-rate feature. Defined identically on the tape
# (distinct transaction_hash per market in the prior 60 min) and in live_clob
# captures (distinct last_trade_price transaction_hash per market, prior 60
# min) so build-side and validation-side features are comparable.
TRAILING_RATE_WINDOW_S: int = 3600

# The 4 CTF-Exchange internal-leg contracts. A fill row whose ``taker`` is one
# of these is the ``_matchOrders`` re-emit of the whole aggressive order — a
# DUPLICATE of its sibling maker legs (same-token bundles re-emit the order on
# the same token; cross-token mint/merge bundles are mirrored by the sibling
# legs on the complementary token with the identical half-spread content under
# bid/ask consistency). The estimator therefore uses ONLY non-internal rows.
# Source of truth: data_infra/operator_denylist.py EXCHANGE_INTERNAL_LEG.
EXCHANGE_INTERNAL_LEG: tuple[str, ...] = (
    "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",  # CTF Exchange v1 standard
    "0xc5d563a36ae78145c45a50134d48a1215220f80a",  # CTF Exchange v1 neg-risk
    "0xe111180000d2663c0091e4f400237545b87b996b",  # CTF Exchange v2 standard
    "0xe2222d279d744050d28e00520010520000310f59",  # CTF Exchange v2 neg-risk
)

CATEGORIES: list[str] = [
    "crypto_4h", "daily_crypto", "geopolitics", "sports", "politics_negrisk",
    "tech", "other",
]

# Verbatim port of the Block K5 market-category CASE
# (scripts/dali_block_k5_real_maker_pnl.py) parameterised on the source table.
# Expects columns: slug_l, question_l, neg_risk. Used by the build script over
# the markets parquet AND over capture-manifest rows so both sides of the
# validation use one taxonomy.
CATEGORY_CASE_SQL: str = """
    CASE
        WHEN regexp_matches(slug_l, '^(btc|eth|sol)-updown-4h-[0-9]+')
            THEN 'crypto_4h'
        WHEN (
            slug_l LIKE '%bitcoin%' OR slug_l LIKE '%ethereum%' OR slug_l LIKE '%solana%'
            OR slug_l LIKE '%btc%' OR slug_l LIKE '%eth%' OR slug_l LIKE '%sol-%'
            OR question_l LIKE '%bitcoin%' OR question_l LIKE '%ethereum%' OR question_l LIKE '%solana%'
        )
        AND (
            slug_l LIKE '%up-or-down%' OR slug_l LIKE '%above%' OR slug_l LIKE '%below%'
            OR question_l LIKE '%up or down%' OR question_l LIKE '%above%' OR question_l LIKE '%below%'
        )
            THEN 'daily_crypto'
        WHEN (
            slug_l LIKE '%iran%' OR slug_l LIKE '%israel%' OR slug_l LIKE '%ukraine%'
            OR slug_l LIKE '%russia%' OR slug_l LIKE '%gaza%' OR slug_l LIKE '%ceasefire%'
            OR slug_l LIKE '%nuclear%' OR slug_l LIKE '%taiwan%' OR slug_l LIKE '%china%'
            OR slug_l LIKE '%war%' OR slug_l LIKE '%hormuz%' OR question_l LIKE '%iran%'
            OR question_l LIKE '%israel%' OR question_l LIKE '%ukraine%' OR question_l LIKE '%russia%'
            OR question_l LIKE '%gaza%' OR question_l LIKE '%ceasefire%' OR question_l LIKE '%nuclear%'
            OR question_l LIKE '%taiwan%' OR question_l LIKE '%china%' OR question_l LIKE '%war%'
        )
            THEN 'geopolitics'
        WHEN (
            slug_l LIKE '%nba%' OR slug_l LIKE '%nfl%' OR slug_l LIKE '%nhl%' OR slug_l LIKE '%mlb%'
            OR slug_l LIKE '%ufc%' OR slug_l LIKE '%soccer%' OR slug_l LIKE '%champions-league%'
            OR slug_l LIKE '%premier-league%' OR question_l LIKE '% win the game%'
            OR question_l LIKE '%beat the%' OR question_l LIKE '%score%'
            OR question_l LIKE '%points%' OR question_l LIKE '%goals%'
        )
            THEN 'sports'
        WHEN neg_risk AND (
            slug_l LIKE '%election%' OR slug_l LIKE '%president%' OR slug_l LIKE '%senate%'
            OR slug_l LIKE '%congress%' OR slug_l LIKE '%trump%' OR slug_l LIKE '%biden%'
            OR slug_l LIKE '%democrat%' OR slug_l LIKE '%republican%'
            OR question_l LIKE '%election%' OR question_l LIKE '%president%' OR question_l LIKE '%senate%'
            OR question_l LIKE '%congress%' OR question_l LIKE '%trump%' OR question_l LIKE '%biden%'
        )
            THEN 'politics_negrisk'
        WHEN (
            slug_l LIKE '%openai%' OR slug_l LIKE '%ai%' OR slug_l LIKE '%nvidia%'
            OR slug_l LIKE '%tesla%' OR slug_l LIKE '%spacex%' OR slug_l LIKE '%apple%'
            OR slug_l LIKE '%iphone%' OR slug_l LIKE '%google%' OR slug_l LIKE '%meta%'
            OR question_l LIKE '%openai%' OR question_l LIKE '%nvidia%' OR question_l LIKE '%tesla%'
            OR question_l LIKE '%spacex%' OR question_l LIKE '%iphone%'
        )
            THEN 'tech'
        ELSE 'other'
    END
"""


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------
def aggressor_dir(maker_side: str | None) -> int:
    """Aggressor sign s_dir for a NON-internal-leg historical fill.

    ``maker_side`` is the passive maker's token side (see
    lib/trade_sign_normalization.historical_to_aggressor): maker sold -> the
    taker BOUGHT (+1); maker bought -> the taker SOLD (-1). 0 when unknown.

    Internal-leg rows (taker in EXCHANGE_INTERNAL_LEG) carry the AGGRESSOR's
    own side in maker_side — this function must not be applied to them; the
    estimator drops those rows as duplicates instead.
    """
    raw = (maker_side or "").strip().upper()
    if raw == "SELL":
        return 1
    if raw == "BUY":
        return -1
    return 0


def price_bucket(price: float) -> str:
    """Price-level bucket label for a probability-unit price in (0, 1)."""
    for hi, label in zip(PRICE_BUCKET_EDGES[1:], PRICE_BUCKET_LABELS, strict=True):
        if price < hi:
            return label
    return PRICE_BUCKET_LABELS[-1]  # price == 1.0


def ttr_bucket(ttr_hours: float | None) -> str:
    """Time-to-resolution bucket; negative TTR (post-end fill) maps to the
    shortest bucket; None/NaN to 'ttr_unknown'."""
    if ttr_hours is None or (isinstance(ttr_hours, float) and math.isnan(ttr_hours)):
        return TTR_UNKNOWN
    h = max(float(ttr_hours), 0.0)
    for hi, label in zip(TTR_BUCKET_EDGES_H[1:], TTR_BUCKET_LABELS, strict=True):
        if h < hi:
            return label
    return TTR_BUCKET_LABELS[-1]


def activity_bucket(trade_rate: float, breakpoints: tuple[float, float, float]) -> str:
    """Activity quartile from a trailing trade rate (distinct tx / 60 min) and
    the build-sample [q25, q50, q75] breakpoints."""
    q25, q50, q75 = breakpoints
    if trade_rate <= q25:
        return ACTIVITY_LABELS[0]
    if trade_rate <= q50:
        return ACTIVITY_LABELS[1]
    if trade_rate <= q75:
        return ACTIVITY_LABELS[2]
    return ACTIVITY_LABELS[3]


# ----------------------------------------------------------------------------
# Surface loader + predict
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SpreadPrediction:
    half_spread_cents: float       # tick-floored prediction
    source_level: str              # which fallback level supplied it
    cell_n_fills: int              # sample size behind the estimate
    raw_median_cents: float | None # pre-floor median, None for floor-only
    # frac_negative of the cell that supplied the prediction (None for
    # floor-only). Metadata passthrough so consumers can refuse contamination-
    # flagged cells (SPREAD-1b hybrid arm); the prediction itself is unchanged.
    cell_frac_negative: float | None = None


# Fallback chain, most-specific first. Each level is the subset of grouping
# keys to match on; the build script writes one aggregate row-set per level.
FALLBACK_LEVELS: list[tuple[str, ...]] = [
    ("category", "price_bucket", "ttr_bucket", "activity_bucket"),
    ("category", "price_bucket", "activity_bucket"),
    ("category", "price_bucket"),
    ("price_bucket",),
]


class SpreadSurface:
    """Loads the built surface (long-format CSV: one row per level x cell) and
    answers ``predict(market_state)`` queries through the fallback chain.

    CSV schema (written by scripts/spread_surface_build.py `surface`):
        level (int index into FALLBACK_LEVELS), category, price_bucket,
        ttr_bucket, activity_bucket (empty string where not part of the level),
        n_fills, n_markets, median_cents, p25_cents, p75_cents, frac_negative
    plus a sidecar meta CSV with per-category activity breakpoints.
    """

    def __init__(self, table: pd.DataFrame, activity_breaks: dict[str, tuple[float, float, float]],
                 min_cell_fills: int = 50):
        self.min_cell_fills = min_cell_fills
        self.activity_breaks = activity_breaks
        self._lookup: dict[tuple, pd.Series] = {}
        for _, row in table.iterrows():
            level = int(row["level"])
            keys = FALLBACK_LEVELS[level]
            key = (level,) + tuple(str(row[k]) for k in keys)
            self._lookup[key] = row

    @classmethod
    def load(cls, surface_csv: Path, meta_csv: Path, min_cell_fills: int = 50) -> "SpreadSurface":
        table = pd.read_csv(surface_csv, keep_default_na=False)
        meta = pd.read_csv(meta_csv)
        breaks = {
            str(r["category"]): (float(r["act_q25"]), float(r["act_q50"]), float(r["act_q75"]))
            for _, r in meta.iterrows()
        }
        return cls(table, breaks, min_cell_fills=min_cell_fills)

    def predict(self, price: float, ttr_hours: float | None, trade_rate: float,
                category: str) -> SpreadPrediction:
        """Half-spread (cents) for a market state, walking the fallback chain.

        ``trade_rate`` is the trailing 60-min distinct-transaction count for
        the market. Categories outside the build taxonomy fall back to
        'other'. The returned value is always >= TICK_FLOOR_CENTS.
        """
        cat = category if category in CATEGORIES else "other"
        breaks = self.activity_breaks.get(cat) or self.activity_breaks.get("other")
        state = {
            "category": cat,
            "price_bucket": price_bucket(price),
            "ttr_bucket": ttr_bucket(ttr_hours),
            "activity_bucket": activity_bucket(trade_rate, breaks) if breaks else ACTIVITY_LABELS[0],
        }
        for level, keys in enumerate(FALLBACK_LEVELS):
            key = (level,) + tuple(state[k] for k in keys)
            row = self._lookup.get(key)
            if row is None or int(row["n_fills"]) < self.min_cell_fills:
                continue
            raw = float(row["median_cents"])
            return SpreadPrediction(
                half_spread_cents=max(raw, TICK_FLOOR_CENTS),
                source_level="level%d:%s" % (level, "x".join(keys)),
                cell_n_fills=int(row["n_fills"]),
                raw_median_cents=raw,
                cell_frac_negative=(float(row["frac_negative"])
                                    if "frac_negative" in row else None),
            )
        return SpreadPrediction(
            half_spread_cents=TICK_FLOOR_CENTS,
            source_level="tick_floor_only",
            cell_n_fills=0,
            raw_median_cents=None,
        )
