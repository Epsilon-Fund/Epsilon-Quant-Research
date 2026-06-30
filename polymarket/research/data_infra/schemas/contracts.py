"""
contracts.py — POLYMARKET dataset contracts.

Project-specific declarations for the Polymarket research project. The generic
engine lives in core.py (vendored, byte-identical to the crypto copy). This
file must NOT import anything from the crypto project.

Datasets covered:
  * pm_trades            — the raw fills family (trades_seed + trades_delta_shard*)
  * pm_closed_positions  — resolved (address, market, outcome) positions
  * pm_traders           — per-wallet metrics panel
  * pm_l2_book/_trades/_price_change/_bba — live CLOB L2 capture (MM Path B)

Value sets / 0x columns / monotone columns were calibrated against the real
parquet on disk (see data_contract_validation_layer_findings.md).
"""
from __future__ import annotations

import pandera.polars as pa
import polars as pl

from .core import AppendOnlyRule, Contract, LookaheadRule, RowRule, TimestampRule

# Polymarket outcome-token prices live in [0, 1] (probabilities).
_PRICE01 = pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True)
_SIDE = pa.Column(str, pa.Check.isin(["BUY", "SELL"]), nullable=True)

# ════════════════════════════════════════════════════════════════════════════
# 1) Raw fills family — trades_seed.parquet + trades_delta_shard*.parquet
# ════════════════════════════════════════════════════════════════════════════
_TRADES_SCHEMA = pa.DataFrameSchema(
    {
        "timestamp": pa.Column(nullable=False),       # datetime[us]; ordering via TimestampRule
        "market_id": pa.Column(str, nullable=True),    # numeric id string (NOT 0x)
        "condition_id": pa.Column(str, nullable=True), # 0x condition hash
        "neg_risk": pa.Column(bool, nullable=True),    # has legitimate nulls
        "maker": pa.Column(str, nullable=True),        # 0x address
        "taker": pa.Column(str, nullable=True),        # 0x address
        "maker_asset_id": pa.Column(str, nullable=True),
        "taker_asset_id": pa.Column(str, nullable=True),
        "usd_amount": pa.Column(float, pa.Check.ge(0), nullable=False),
        "token_amount": pa.Column(float, pa.Check.ge(0), nullable=False),
        "price": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
        "maker_side": pa.Column(str, pa.Check.isin(["BUY", "SELL"]), nullable=True),
        "transaction_hash": pa.Column(str, nullable=True),  # 0x tx hash
    },
    strict=False,
    coerce=False,
)

PM_TRADES = Contract(
    name="pm_trades",
    description="Raw Polymarket fills — trades_seed + append-only trades_delta_shard* (the `raw_trades` view).",
    schema=_TRADES_SCHEMA,
    # NO monotonic-timestamp clause: PM fills are stored in coarse chronological /
    # block-log order with microsecond-scale reordering BY DESIGN (≈3.3k/6.1M
    # out-of-order in a sampled shard, 66/5M even in the seed). A strict/non-decreasing
    # invariant would be theatrically strict (realism rule #2 — don't impose an invariant
    # the instrument lacks). Ordering IS enforced where it is real (crypto OHLCV, L2 received_ns).
    timestamp=None,
    lookahead=LookaheadRule("timestamp"),  # but no FUTURE-dated fills — that is still real
    append_only=AppendOnlyRule(mode="shard"),  # delta shards never mutate in place (the headline invariant here)
    address_columns=("condition_id", "maker", "taker", "transaction_hash"),
    finite_columns=("usd_amount", "token_amount", "price"),
    drift_columns=("price", "token_amount"),
    scan_strategy="recent_shards",   # seed is 151M rows: schema+append-only all, row-scan recent
    recent_shards=2,
    max_scan_rows=3_000_000,
    notes=("Append-only family: the 151M-row seed plus dated delta shards. Schema + append-only "
           "are checked on ALL shards (cheap); row-level invariants (addresses, finite, price∈[0,1], "
           "no future fills) scan the most-recent shards only (coverage logged). No monotonic-ts "
           "clause — fills are not stored strictly time-sorted (see above). 0x lowercase: "
           "condition_id/maker/taker/transaction_hash (market_id and *_asset_id are decimal ids)."),
)

# ════════════════════════════════════════════════════════════════════════════
# 2) Closed positions — resolved (address, market, outcome) panel
# ════════════════════════════════════════════════════════════════════════════
_CLOSED_SCHEMA = pa.DataFrameSchema(
    {
        "address": pa.Column(str, nullable=True),       # 0x wallet
        "market_id": pa.Column(str, nullable=True),
        "condition_id": pa.Column(str, nullable=True),  # 0x
        "neg_risk": pa.Column(bool, nullable=True),
        "outcome_token_id": pa.Column(str, nullable=True),
        "outcome_index": pa.Column(int, pa.Check.ge(0), nullable=True),
        "n_fills": pa.Column(int, pa.Check.ge(0), nullable=True),
        "first_fill_ts": pa.Column(nullable=True),
        "last_fill_ts": pa.Column(nullable=True),
        "resolution_ts": pa.Column(nullable=True),
        "realised_pnl": pa.Column(float, nullable=True),
        "resolution_price": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True),
        "is_held_to_resolution": pa.Column(bool, nullable=True),
    },
    strict=False,
    coerce=False,
)

PM_CLOSED_POSITIONS = Contract(
    name="pm_closed_positions",
    description="Resolved (address, market, outcome) positions with synthetic redemption (closed_positions.parquet).",
    schema=_CLOSED_SCHEMA,
    timestamp=None,  # not a single ordered series; keyed by (address, market, outcome)
    lookahead=LookaheadRule("last_fill_ts"),  # no fills stamped in the future
    address_columns=("address", "condition_id"),
    finite_columns=("realised_pnl", "resolution_price", "gross_usd_volume"),
    row_rules=(
        RowRule("fill_order",
                lambda: (pl.col("first_fill_ts") <= pl.col("last_fill_ts")) | pl.col("first_fill_ts").is_null(),
                "first_fill_ts must be <= last_fill_ts"),
        RowRule("n_fills_nonneg", lambda: pl.col("n_fills") >= 0, "n_fills must be >= 0"),
    ),
    drift_columns=("realised_pnl", "holding_duration_hours"),
    scan_strategy="all",
    max_scan_rows=3_000_000,   # 270M-row file: lazy tail keeps memory bounded
    notes=("Single large file (~270M rows). Row invariants scan the most-recent 3M stored rows "
           "(coverage logged); schema/presence covers the whole file via metadata."),
)

# ════════════════════════════════════════════════════════════════════════════
# 3) Trader panel
# ════════════════════════════════════════════════════════════════════════════
_TRADERS_SCHEMA = pa.DataFrameSchema(
    {
        "address": pa.Column(str, nullable=False),  # 0x wallet
        "n_closed_positions": pa.Column(int, pa.Check.ge(0), nullable=True),
        "n_distinct_markets": pa.Column(int, pa.Check.ge(0), nullable=True),
        "total_volume_usd": pa.Column(float, pa.Check.ge(0), nullable=True),
        "first_activity_ts": pa.Column(nullable=True),
        "last_activity_ts": pa.Column(nullable=True),
        "pos_win_rate": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True),
    },
    strict=False,
    coerce=False,
)

PM_TRADERS = Contract(
    name="pm_traders",
    description="Per-wallet metrics, style, bankroll, operator flag (traders.parquet).",
    schema=_TRADERS_SCHEMA,
    lookahead=LookaheadRule("last_activity_ts"),
    address_columns=("address",),
    finite_columns=("total_volume_usd", "pos_win_rate"),
    drift_columns=("total_volume_usd", "pos_win_rate"),
    scan_strategy="all",
    max_scan_rows=3_000_000,
    notes="Derived per-wallet panel; one row per address. win-rate must stay in [0,1].",
)

# ════════════════════════════════════════════════════════════════════════════
# 4) Live CLOB L2 capture (MM Path B). Ordering is on `received_ns` (monotonic
#    receive clock); `timestamp_ms` is PM SERVER time and is NOT stored-order
#    monotone, so lookahead/no-future uses timestamp_ms while ORDER uses
#    received_ns. Calibrated against real capture: 0 received_ns inversions,
#    0 crossed bba, 0 server-ahead-of-receive.
# ════════════════════════════════════════════════════════════════════════════
_L2_COMMON = {
    "timestamp_ms": pa.Column(pl.Int64, pa.Check.gt(0), nullable=False),
    "received_at": pa.Column(str, nullable=False),
    "received_ns": pa.Column(pl.Int64, pa.Check.gt(0), nullable=False),
    "universe": pa.Column(str, nullable=False),
    "asset_id": pa.Column(str, nullable=False),
    "market": pa.Column(str, nullable=False),  # 0x condition hash
}


def _l2_ts() -> TimestampRule:
    return TimestampRule("received_ns", order="non_decreasing", epoch_unit="ns", per_file=True)


def _l2_lookahead() -> LookaheadRule:
    # server timestamp must be in the past and not ahead of local receive (300s clock-skew grace)
    return LookaheadRule("timestamp_ms", epoch_unit="ms", not_after_column="received_at",
                         grace_seconds=300.0)


PM_L2_BOOK = Contract(
    name="pm_l2_book",
    description="CLOB L2 full-book snapshots (l2_data/{date}/{universe}/book_*.parquet).",
    schema=pa.DataFrameSchema({**_L2_COMMON,
                               "bids": pa.Column(str, nullable=True),
                               "asks": pa.Column(str, nullable=True)}, strict=False),
    timestamp=_l2_ts(),
    lookahead=_l2_lookahead(),
    append_only=AppendOnlyRule(mode="shard"),  # hourly capture files never mutate
    address_columns=("market",),
    finite_columns=("timestamp_ms", "received_ns"),
    scan_strategy="all",
    max_scan_rows=5_000_000,
    notes="Append-only hourly capture. JSON bid/ask ladders kept as strings; depth parsing is downstream.",
)

PM_L2_TRADES = Contract(
    name="pm_l2_trades",
    description="CLOB trade prints (l2_data/{date}/{universe}/trades_*.parquet).",
    schema=pa.DataFrameSchema({**_L2_COMMON,
                               "price": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
                               "size": pa.Column(float, pa.Check.ge(0), nullable=False),
                               "side": _SIDE,
                               "fee_rate_bps": pa.Column(float, pa.Check.ge(0), nullable=True),
                               "transaction_hash": pa.Column(str, nullable=True)}, strict=False),
    timestamp=_l2_ts(),
    lookahead=_l2_lookahead(),
    append_only=AppendOnlyRule(mode="shard"),  # hourly capture files never mutate
    address_columns=("market", "transaction_hash"),
    finite_columns=("price", "size", "fee_rate_bps"),
    drift_columns=("price", "size"),
    scan_strategy="all",
    max_scan_rows=5_000_000,
)

PM_L2_PRICE_CHANGE = Contract(
    name="pm_l2_price_change",
    description="CLOB incremental L2 deltas (l2_data/{date}/{universe}/price_change_*.parquet).",
    schema=pa.DataFrameSchema({**_L2_COMMON,
                               "price": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
                               "side": _SIDE,
                               "size": pa.Column(float, pa.Check.ge(0), nullable=False),
                               "best_bid": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True),
                               "best_ask": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True)},
                              strict=False),
    timestamp=_l2_ts(),
    lookahead=_l2_lookahead(),
    append_only=AppendOnlyRule(mode="shard"),  # hourly capture files never mutate
    address_columns=("market",),
    finite_columns=("price", "size"),
    row_rules=(
        RowRule("ask_ge_bid",
                lambda: (pl.col("best_ask") >= pl.col("best_bid")) | (pl.col("best_bid") <= 0)
                | (pl.col("best_ask") <= 0) | pl.col("best_bid").is_null() | pl.col("best_ask").is_null(),
                "best_ask must be >= best_bid when both sides are present"),
    ),
    drift_columns=("price", "best_bid", "best_ask"),
    scan_strategy="all",
    max_scan_rows=5_000_000,
    notes="High-volume table (~1M rows/hour). best_bid=0 / best_ask=1 mean an empty side, not a crossed book.",
)

PM_L2_BBA = Contract(
    name="pm_l2_bba",
    description="CLOB top-of-book updates (l2_data/{date}/{universe}/bba_*.parquet).",
    schema=pa.DataFrameSchema({**_L2_COMMON,
                               "best_bid": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
                               "best_ask": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
                               "spread": pa.Column(float, pa.Check.ge(0), nullable=False)}, strict=False),
    timestamp=_l2_ts(),
    lookahead=_l2_lookahead(),
    append_only=AppendOnlyRule(mode="shard"),  # hourly capture files never mutate
    address_columns=("market",),
    finite_columns=("best_bid", "best_ask", "spread"),
    row_rules=(
        RowRule("ask_ge_bid",
                lambda: (pl.col("best_ask") >= pl.col("best_bid")) | (pl.col("best_bid") <= 0)
                | (pl.col("best_ask") <= 0),
                "best_ask must be >= best_bid when both sides present"),
        RowRule("spread_consistent",
                lambda: ((pl.col("spread") - (pl.col("best_ask") - pl.col("best_bid"))).abs() <= 1e-6)
                | (pl.col("best_bid") <= 0) | (pl.col("best_ask") <= 0),
                "spread must equal best_ask - best_bid when both sides present"),
    ),
    drift_columns=("best_bid", "best_ask", "spread"),
    scan_strategy="all",
    max_scan_rows=5_000_000,
)

CONTRACTS: dict[str, Contract] = {
    c.name: c for c in (
        PM_TRADES, PM_CLOSED_POSITIONS, PM_TRADERS,
        PM_L2_BOOK, PM_L2_TRADES, PM_L2_PRICE_CHANGE, PM_L2_BBA,
    )
}
