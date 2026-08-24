"""
Markets collector — fetches events and individual binary markets from Falcon.
Handles both open and resolved markets across all weather subcategories,
deduplicates, and upserts into the database.
"""

import sqlite3
from datetime import datetime

from loguru import logger

import config
from collectors.falcon_client import FalconClient


def _fetch_keyword(client: FalconClient, keyword: str, subcategory: str) -> list[dict]:
    """Fetch all resolved + open markets for a single keyword."""
    print(f"\n  [{subcategory}] keyword: '{keyword}'")

    resolved = client.query(
        config.AGENT_MARKETS,
        params={"market_slug": keyword, "closed": "True"},
    )
    print(f"    Resolved: {len(resolved)}")

    open_markets = client.query(
        config.AGENT_MARKETS,
        params={"market_slug": keyword, "closed": "False"},
    )
    print(f"    Open:     {len(open_markets)}")

    return resolved + open_markets


def collect_markets(db_conn: sqlite3.Connection) -> int:
    """
    Collect all weather markets across all subcategories defined in config.WEATHER_KEYWORDS.
    Deduplicates by condition_id across all keywords.

    Args:
        db_conn: Active SQLite connection.

    Returns:
        Total number of markets upserted.
    """
    client = FalconClient()
    started_at = datetime.utcnow().isoformat()

    print("\n--- Collecting all weather markets ---")

    # Fetch from all keywords, deduplicate globally
    all_raw = []
    for keyword, subcategory in config.WEATHER_KEYWORDS.items():
        results = _fetch_keyword(client, keyword, subcategory)
        all_raw.extend(results)

    # Deduplicate by condition_id
    seen = set()
    combined = []
    for m in all_raw:
        cid = m.get("condition_id")
        if cid and cid not in seen:
            seen.add(cid)
            combined.append(m)

    print(f"\n  Total raw results:         {len(all_raw):,}")
    print(f"  Unique markets after dedup: {len(combined):,}")

    # Upsert into database
    events_seen = set()
    markets_upserted = 0

    for m in combined:
        event_slug = m.get("event_slug", "")
        condition_id = m.get("condition_id")

        if not condition_id:
            logger.warning("Skipping market with no condition_id: {}", m.get("slug"))
            continue

        # Determine subcategory from slug
        slug = m.get("slug", "")
        subcategory = config.DEFAULT_CATEGORY
        for kw, subcat in config.WEATHER_KEYWORDS.items():
            if kw in slug:
                subcategory = subcat
                break

        # Upsert event (INSERT OR IGNORE — don't overwrite existing)
        if event_slug and event_slug not in events_seen:
            db_conn.execute(
                """
                INSERT OR IGNORE INTO events (event_slug, title, category, end_date, total_volume, num_markets)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_slug,
                    m.get("event_title", m.get("title", "")),
                    subcategory,
                    m.get("end_date"),
                    m.get("event_volume", m.get("volume_total")),
                    m.get("num_markets"),
                ),
            )
            events_seen.add(event_slug)

        # Upsert market (INSERT OR REPLACE — always update with latest data)
        closed_val = 1 if m.get("closed") in (True, "True", "true", 1) else 0
        db_conn.execute(
            """
            INSERT OR REPLACE INTO markets
                (condition_id, event_slug, slug, question, category, start_date, end_date,
                 closed, winning_outcome, volume_total,
                 side_a_outcome, side_b_outcome, side_a_token_id, side_b_token_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                condition_id,
                event_slug,
                slug,
                m.get("question", m.get("title", "")),
                subcategory,
                m.get("start_date"),
                m.get("end_date"),
                closed_val,
                m.get("winning_outcome"),
                m.get("volume_total", m.get("volume")),
                m.get("side_a_outcome", m.get("outcome_a")),
                m.get("side_b_outcome", m.get("outcome_b")),
                m.get("side_a_token_id", m.get("token_id_a")),
                m.get("side_b_token_id", m.get("token_id_b")),
            ),
        )
        markets_upserted += 1

    db_conn.commit()

    # Count events in DB
    event_count = db_conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    # Log to collection_log
    db_conn.execute(
        """
        INSERT INTO collection_log (collection_type, target, status, records_fetched, started_at, completed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("markets", "all_weather", "done", markets_upserted, started_at, datetime.utcnow().isoformat()),
    )
    db_conn.commit()

    print(f"\n  Events in DB:     {event_count:,}")
    print(f"  Markets upserted: {markets_upserted:,}")
    logger.info("Markets collection complete: {} markets, {} events", markets_upserted, event_count)

    return markets_upserted
