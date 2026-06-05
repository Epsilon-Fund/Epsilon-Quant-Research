from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


_SCHEMA = """
CREATE TABLE IF NOT EXISTS heartbeat (
    id INTEGER PRIMARY KEY DEFAULT 1,
    ts TEXT NOT NULL,
    ws_market_connected INTEGER NOT NULL,
    ws_user_connected INTEGER NOT NULL,
    last_market_msg_age_s REAL,
    active_markets INTEGER NOT NULL,
    open_orders INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS open_orders (
    event_slug TEXT PRIMARY KEY,
    token_id TEXT NOT NULL,
    is_yes INTEGER NOT NULL,
    price_ticks INTEGER NOT NULL,
    tick_size REAL NOT NULL,
    qty INTEGER NOT NULL,
    status TEXT NOT NULL,
    venue_order_id TEXT,
    placed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kill_switch (
    id INTEGER PRIMARY KEY DEFAULT 1,
    requested INTEGER NOT NULL DEFAULT 0
);
"""


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def write_heartbeat(
    conn: sqlite3.Connection,
    *,
    ws_market_connected: bool,
    ws_user_connected: bool,
    last_market_msg_age_s: float | None,
    active_markets: int,
    open_orders: int,
) -> None:
    conn.execute(
        """
        INSERT INTO heartbeat
            (id, ts, ws_market_connected, ws_user_connected,
             last_market_msg_age_s, active_markets, open_orders)
        VALUES (1, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            ts = excluded.ts,
            ws_market_connected = excluded.ws_market_connected,
            ws_user_connected = excluded.ws_user_connected,
            last_market_msg_age_s = excluded.last_market_msg_age_s,
            active_markets = excluded.active_markets,
            open_orders = excluded.open_orders
        """,
        (
            _now_iso(),
            int(ws_market_connected),
            int(ws_user_connected),
            last_market_msg_age_s,
            active_markets,
            open_orders,
        ),
    )
    conn.commit()


def upsert_order(
    conn: sqlite3.Connection,
    *,
    event_slug: str,
    token_id: str,
    is_yes: bool,
    price_ticks: int,
    tick_size: float,
    qty: int,
    status: str,
    venue_order_id: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO open_orders
            (event_slug, token_id, is_yes, price_ticks, tick_size,
             qty, status, venue_order_id, placed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_slug) DO UPDATE SET
            token_id = excluded.token_id,
            is_yes = excluded.is_yes,
            price_ticks = excluded.price_ticks,
            tick_size = excluded.tick_size,
            qty = excluded.qty,
            status = excluded.status,
            venue_order_id = excluded.venue_order_id
        """,
        (
            event_slug,
            token_id,
            int(is_yes),
            price_ticks,
            tick_size,
            qty,
            status,
            venue_order_id,
            _now_iso(),
        ),
    )
    conn.commit()


def delete_order(conn: sqlite3.Connection, event_slug: str) -> None:
    conn.execute("DELETE FROM open_orders WHERE event_slug = ?", (event_slug,))
    conn.commit()


def delete_all_orders(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM open_orders")
    conn.commit()


def check_kill_switch(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute("SELECT requested FROM kill_switch WHERE id = 1").fetchone()
        return bool(row[0]) if row else False
    except sqlite3.OperationalError:
        return False


def set_kill_switch(conn: sqlite3.Connection, requested: bool) -> None:
    conn.execute(
        """
        INSERT INTO kill_switch (id, requested)
        VALUES (1, ?)
        ON CONFLICT(id) DO UPDATE SET requested = excluded.requested
        """,
        (int(requested),),
    )
    conn.commit()
