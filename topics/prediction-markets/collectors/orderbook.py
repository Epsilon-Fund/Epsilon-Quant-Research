"""
On-demand orderbook snapshot fetcher for price impact analysis.
NOT part of the main collection pipeline.
Call this manually during analysis to assess copy-trade slippage.

Usage:
    from collectors.orderbook import fetch_orderbook_around_trade
    snapshots = fetch_orderbook_around_trade(token_id, trade_timestamp, window_minutes=30)
"""

import json
from datetime import datetime, timedelta

from loguru import logger

import config
from collectors.falcon_client import FalconClient


def fetch_orderbook_around_trade(
    token_id: str,
    trade_timestamp_iso: str,
    window_minutes: int = 30,
) -> list[dict]:
    """
    Fetches orderbook snapshots in a window around a specific trade.

    Args:
        token_id: The token ID of the market outcome.
        trade_timestamp_iso: ISO timestamp string of the whale's trade
            (e.g. "2026-04-01T14:32:00").
        window_minutes: How many minutes before and after to fetch (default 30).

    Returns:
        List of orderbook snapshots with bids and asks parsed from JSON strings.
        Each snapshot dict includes the original fields plus parsed 'bids_parsed'
        and 'asks_parsed' lists if the raw fields were JSON strings.
    """
    client = FalconClient()

    trade_time = datetime.fromisoformat(trade_timestamp_iso)
    start_time = trade_time - timedelta(minutes=window_minutes)
    end_time = trade_time + timedelta(minutes=window_minutes)

    # Orderbook endpoint expects Unix millisecond timestamps as strings
    start_ms = str(int(start_time.timestamp() * 1000))
    end_ms = str(int(end_time.timestamp() * 1000))

    logger.info(
        "Fetching orderbook for token {} around {} (±{}min)",
        token_id, trade_timestamp_iso, window_minutes,
    )

    results = client.query(
        config.AGENT_ORDERBOOK,
        params={
            "token_id": token_id,
            "start_time": start_ms,
            "end_time": end_ms,
        },
        paginate=False,
    )

    # Parse bids/asks from JSON strings if present
    snapshots = []
    for snap in results:
        snapshot = dict(snap)

        for field in ("bids", "asks"):
            raw = snapshot.get(field)
            if isinstance(raw, str):
                try:
                    snapshot[f"{field}_parsed"] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    snapshot[f"{field}_parsed"] = []
            elif isinstance(raw, list):
                snapshot[f"{field}_parsed"] = raw
            else:
                snapshot[f"{field}_parsed"] = []

        snapshots.append(snapshot)

    logger.info("Retrieved {} orderbook snapshots", len(snapshots))
    return snapshots
