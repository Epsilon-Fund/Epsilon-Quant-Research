from __future__ import annotations

import sqlite3


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_heartbeat(conn: sqlite3.Connection) -> dict | None:
    try:
        row = conn.execute("SELECT * FROM heartbeat WHERE id = 1").fetchone()
        return dict(row) if row else None
    except sqlite3.OperationalError:
        return None  # tables not yet created — bot hasn't run yet


def get_open_orders(conn: sqlite3.Connection) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT * FROM open_orders WHERE status = 'OPEN' ORDER BY placed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []  # tables not yet created — bot hasn't run yet


def get_kill_switch(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute("SELECT requested FROM kill_switch WHERE id = 1").fetchone()
        return bool(row[0]) if row else False
    except sqlite3.OperationalError:
        return False


def set_kill_switch(conn: sqlite3.Connection, requested: bool) -> None:
    try:
        conn.execute(
            """
            INSERT INTO kill_switch (id, requested)
            VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET requested = excluded.requested
            """,
            (int(requested),),
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass
