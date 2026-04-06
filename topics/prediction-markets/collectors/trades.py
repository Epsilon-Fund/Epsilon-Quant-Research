"""
Trades collector — fetches all trades for each market in the database.
Fully resumable: checks collection_log before fetching each market's trades.
"""

import sqlite3
from datetime import datetime

from loguru import logger

import config
from collectors.falcon_client import FalconClient


def collect_trades(db_conn: sqlite3.Connection) -> int:
    """
    Collect trades for every market in the database.
    Skips markets that already have status 'done' in collection_log.

    Args:
        db_conn: Active SQLite connection.

    Returns:
        Total number of new trades inserted.
    """
    client = FalconClient()

    # Get all market slugs
    rows = db_conn.execute("SELECT slug, condition_id FROM markets ORDER BY slug").fetchall()
    total_markets = len(rows)

    if total_markets == 0:
        print("\n--- No markets in database. Run --markets first. ---")
        return 0

    # Get already-completed slugs from collection_log
    done_rows = db_conn.execute(
        "SELECT target FROM collection_log WHERE collection_type = 'trades' AND status = 'done'"
    ).fetchall()
    done_slugs = {r[0] for r in done_rows}

    pending = [(slug, cid) for slug, cid in rows if slug not in done_slugs]

    print(f"\n--- Collecting trades ---")
    print(f"  Total markets: {total_markets}, already done: {len(done_slugs)}, pending: {len(pending)}")

    total_trades = 0

    for i, (slug, condition_id) in enumerate(pending, 1):
        started_at = datetime.utcnow().isoformat()

        # Insert partial log entry to mark in-progress
        cursor = db_conn.execute(
            """
            INSERT INTO collection_log (collection_type, target, status, started_at)
            VALUES (?, ?, ?, ?)
            """,
            ("trades", slug, "partial", started_at),
        )
        log_id = cursor.lastrowid
        db_conn.commit()

        try:
            results = client.query(
                config.AGENT_TRADES,
                params={
                    "market_slug": slug,
                    "proxy_wallet": "ALL",
                    "condition_id": "ALL",
                },
            )

            trade_count = 0
            for t in results:
                trade_id = t.get("id")
                if not trade_id:
                    continue

                # Look up event_slug from condition_id
                trade_cid = t.get("condition_id", condition_id)
                event_row = db_conn.execute(
                    "SELECT event_slug FROM markets WHERE condition_id = ?",
                    (trade_cid,),
                ).fetchone()
                event_slug = event_row[0] if event_row else ""

                db_conn.execute(
                    """
                    INSERT OR IGNORE INTO trades
                        (id, condition_id, event_slug, slug, proxy_wallet, side,
                         outcome, price, size, timestamp, transaction_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_id,
                        trade_cid,
                        event_slug,
                        t.get("slug", slug),
                        t.get("proxy_wallet", ""),
                        t.get("side"),
                        t.get("outcome"),
                        t.get("price"),
                        t.get("size"),
                        t.get("timestamp"),
                        t.get("transaction_hash"),
                    ),
                )
                trade_count += 1

            # Update log to done
            db_conn.execute(
                """
                UPDATE collection_log
                SET status = 'done', records_fetched = ?, completed_at = ?
                WHERE id = ?
                """,
                (trade_count, datetime.utcnow().isoformat(), log_id),
            )
            db_conn.commit()

            total_trades += trade_count
            done_count = len(done_slugs) + i
            print(f"  [{done_count}/{total_markets}] {slug} — {trade_count} trades")

        except Exception as e:
            # Update log to error, continue to next market
            db_conn.execute(
                """
                UPDATE collection_log
                SET status = 'error', error_message = ?, completed_at = ?
                WHERE id = ?
                """,
                (str(e)[:500], datetime.utcnow().isoformat(), log_id),
            )
            db_conn.commit()
            logger.error("Error collecting trades for {}: {}", slug, e)
            print(f"  [{len(done_slugs) + i}/{total_markets}] {slug} — ERROR: {e}")

    print(f"  Total new trades inserted: {total_trades:,}")
    logger.info("Trades collection complete: {} new trades", total_trades)
    return total_trades
