"""
contracts.py — CRYPTO dataset contracts (Binance OHLCV).

Project-specific declarations for the crypto live-trading project. The generic
engine lives in core.py (vendored, identical to the Polymarket copy). This file
must NOT import anything from the Polymarket project.
"""
from __future__ import annotations

import pandera.polars as pa
import polars as pl

from .core import Contract, LookaheadRule, RowRule, TimestampRule

# Live momentum universe (brain/CODEX.md § Active universe) + researched extras.
LIVE_UNIVERSE = ["ADAUSDT", "AVAXUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

# ── shared OHLCV schema ──────────────────────────────────────────────────────
_PRICE = pa.Column(float, pa.Check.gt(0), nullable=False)
_OHLCV_SCHEMA = pa.DataFrameSchema(
    {
        "Open": _PRICE,
        "High": _PRICE,
        "Low": _PRICE,
        "Close": _PRICE,
        "Volume": pa.Column(float, pa.Check.ge(0), nullable=False),
        "Time": pa.Column(nullable=False),  # dtype/ordering handled by TimestampRule
    },
    strict=False,  # extra columns (e.g. adjusted) are tolerated
    coerce=False,
)

_OHLC_ROW_RULES = (
    RowRule("high_ge_low", lambda: pl.col("High") >= pl.col("Low"),
            "High must be >= Low"),
    RowRule("high_ge_open_close",
            lambda: (pl.col("High") >= pl.col("Open")) & (pl.col("High") >= pl.col("Close")),
            "High must be >= both Open and Close"),
    RowRule("low_le_open_close",
            lambda: (pl.col("Low") <= pl.col("Open")) & (pl.col("Low") <= pl.col("Close")),
            "Low must be <= both Open and Close"),
)

_OHLCV_NOTES = (
    "The OHLCV cache is a REBUILDABLE refetch cache (brain data manifest), not an "
    "append-only family, so there is deliberately no append-only clause here "
    "(realism calibration — don't impose an invariant the instrument lacks). Drift "
    "monitors Volume regime; raw price level is non-stationary and drifts by design, "
    "so it is not monitored as a data-quality signal."
)

CRYPTO_OHLCV_DAILY = Contract(
    name="crypto_ohlcv_daily",
    description="Binance daily OHLCV bars per symbol (live_trading/cache/daily/*_daily.parquet).",
    schema=_OHLCV_SCHEMA,
    timestamp=TimestampRule("Time", order="strict_increasing", cadence="1d", per_file=True),
    lookahead=LookaheadRule("Time"),
    finite_columns=("Open", "High", "Low", "Close", "Volume"),
    row_rules=_OHLC_ROW_RULES,
    drift_columns=("Volume",),
    scan_strategy="all",
    notes=_OHLCV_NOTES,
)

CRYPTO_OHLCV_HOURLY = Contract(
    name="crypto_ohlcv_hourly",
    description="Binance hourly OHLCV bars per symbol (live_trading/cache/hourly/*_hourly.parquet).",
    schema=_OHLCV_SCHEMA,
    timestamp=TimestampRule("Time", order="strict_increasing", cadence="1h", per_file=True),
    lookahead=LookaheadRule("Time"),
    finite_columns=("Open", "High", "Low", "Close", "Volume"),
    row_rules=_OHLC_ROW_RULES,
    drift_columns=("Volume",),
    scan_strategy="all",
    notes=_OHLCV_NOTES,
)

CONTRACTS: dict[str, Contract] = {
    c.name: c for c in (CRYPTO_OHLCV_DAILY, CRYPTO_OHLCV_HOURLY)
}
