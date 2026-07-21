"""
Candlestick collector — fetches daily (1d) and hourly (1h) OHLCV candle data
for every market token in the database.
Fully resumable: checks collection_log before fetching each token.
"""

import sqlite3
import time
from datetime import datetime, timedelta

from loguru import logger

import config
from collectors.falcon_client import FalconClient


def collect_candlesticks(db_conn: sqlite3.Connection) -> tuple[int, int]:
    """
    Collect daily and hourly candlestick data for every token across all markets.

    For each market, both the YES token (side_a_token_id) and NO token (side_b_token_id)
    are fetched. Daily candles go back 360 days; hourly candles go back 90 days.

    Args:
        db_conn: Active SQLite connection.

    Returns:
        Tuple of (total_daily_candles, total_hourly_candles) inserted.
    """
    client = FalconClient()

    # Get all markets with their token IDs
    rows = db_conn.execute(
        """
        SELECT condition_id, slug, side_a_outcome, side_b_outcome,
               side_a_token_id, side_b_token_id
        FROM markets ORDER BY slug
        """
    ).fetchall()

    if not rows:
        print("\n--- No markets in database. Run --markets first. ---")
        return 0, 0

    # Build list of (condition_id, token_id, outcome, slug) to process
    tokens = []
    for r in rows:
        cid = r["condition_id"]
        slug = r["slug"]
        if r["side_a_token_id"]:
            tokens.append((cid, r["side_a_token_id"], r["side_a_outcome"] or "Yes", slug))
        if r["side_b_token_id"]:
            tokens.append((cid, r["side_b_token_id"], r["side_b_outcome"] or "No", slug))

    # Get already-completed tokens from collection_log
    done_1d = {
        r[0] for r in db_conn.execute(
            "SELECT target FROM collection_log WHERE collection_type = 'candles_1d' AND status = 'done'"
        ).fetchall()
    }
    done_1h = {
        r[0] for r in db_conn.execute(
            "SELECT target FROM collection_log WHERE collection_type = 'candles_1h' AND status = 'done'"
        ).fetchall()
    }

    total_tokens = len(tokens)
    print(f"\n--- Collecting candlesticks ---")
    print(f"  Total tokens: {total_tokens}, 1d done: {len(done_1d)}, 1h done: {len(done_1h)}")

    now = datetime.utcnow()
    end_ts = str(int(now.timestamp()))
    start_1d_ts = str(int((now - timedelta(days=360)).timestamp()))
    start_1h_ts = str(int((now - timedelta(days=90)).timestamp()))

    total_daily = 0
    total_hourly = 0

    for i, (condition_id, token_id, outcome, slug) in enumerate(tokens, 1):
        daily_count = 0
        hourly_count = 0

        # --- Daily candles ---
        if token_id not in done_1d:
            try:
                results = client.query(
                    config.AGENT_CANDLESTICKS,
                    params={
                        "token_id": token_id,
                        "interval": "1d",
                        "start_time": start_1d_ts,
                        "end_time": end_ts,
                    },
                    paginate=False,
                )

                for c in results:
                    candle_time = c.get("candle_time", c.get("time", ""))
                    composite_id = f"{token_id}_{candle_time}"
                    db_conn.execute(
                        """
                        INSERT OR IGNORE INTO candles_1d
                            (id, condition_id, token_id, outcome, candle_time,
                             open, high, low, close, mean, volume, trade_count,
                             bid_open, bid_high, bid_low, bid_close,
                             ask_open, ask_high, ask_low, ask_close)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            composite_id, condition_id, token_id, outcome, candle_time,
                            c.get("open"), c.get("high"), c.get("low"), c.get("close"),
                            c.get("mean"), c.get("volume"), c.get("trade_count"),
                            c.get("bid_open"), c.get("bid_high"), c.get("bid_low"), c.get("bid_close"),
                            c.get("ask_open"), c.get("ask_high"), c.get("ask_low"), c.get("ask_close"),
                        ),
                    )
                    daily_count += 1

                db_conn.execute(
                    """
                    INSERT INTO collection_log (collection_type, target, status, records_fetched, started_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("candles_1d", token_id, "done", daily_count,
                     datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
                )
                db_conn.commit()
                total_daily += daily_count

            except Exception as e:
                db_conn.execute(
                    """
                    INSERT INTO collection_log (collection_type, target, status, error_message, started_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("candles_1d", token_id, "error", str(e)[:500],
                     datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
                )
                db_conn.commit()
                logger.error("Error fetching 1d candles for {}: {}", token_id, e)

        # --- Hourly candles ---
        if token_id not in done_1h:
            try:
                results = client.query(
                    config.AGENT_CANDLESTICKS,
                    params={
                        "token_id": token_id,
                        "interval": "1h",
                        "start_time": start_1h_ts,
                        "end_time": end_ts,
                    },
                    paginate=False,
                )

                for c in results:
                    candle_time = c.get("candle_time", c.get("time", ""))
                    composite_id = f"{token_id}_{candle_time}"
                    db_conn.execute(
                        """
                        INSERT OR IGNORE INTO candles_1h
                            (id, condition_id, token_id, outcome, candle_time,
                             open, high, low, close, mean, volume, trade_count,
                             bid_open, bid_high, bid_low, bid_close,
                             ask_open, ask_high, ask_low, ask_close)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            composite_id, condition_id, token_id, outcome, candle_time,
                            c.get("open"), c.get("high"), c.get("low"), c.get("close"),
                            c.get("mean"), c.get("volume"), c.get("trade_count"),
                            c.get("bid_open"), c.get("bid_high"), c.get("bid_low"), c.get("bid_close"),
                            c.get("ask_open"), c.get("ask_high"), c.get("ask_low"), c.get("ask_close"),
                        ),
                    )
                    hourly_count += 1

                db_conn.execute(
                    """
                    INSERT INTO collection_log (collection_type, target, status, records_fetched, started_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("candles_1h", token_id, "done", hourly_count,
                     datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
                )
                db_conn.commit()
                total_hourly += hourly_count

            except Exception as e:
                db_conn.execute(
                    """
                    INSERT INTO collection_log (collection_type, target, status, error_message, started_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("candles_1h", token_id, "error", str(e)[:500],
                     datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
                )
                db_conn.commit()
                logger.error("Error fetching 1h candles for {}: {}", token_id, e)

        print(f"  [{i}/{total_tokens}] Candles — {slug} {outcome} token — {daily_count} daily, {hourly_count} hourly")

    print(f"  Total daily candles:  {total_daily:,}")
    print(f"  Total hourly candles: {total_hourly:,}")
    logger.info("Candlestick collection complete: {} daily, {} hourly", total_daily, total_hourly)
    return total_daily, total_hourly
