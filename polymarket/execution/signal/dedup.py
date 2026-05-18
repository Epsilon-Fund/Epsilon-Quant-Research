"""(transaction_hash, log_index) tracking to drop duplicate fills.

Per PLAN.md decision 8, dedup is journal-backed: the journal is the
single source of truth, the in-memory set is only an acceleration.
On startup the set is rebuilt from today's journal; if today's
journal is empty, yesterday's is also consulted to handle
just-past-midnight restarts.
"""
from __future__ import annotations

import threading
from datetime import date, datetime, timedelta, timezone

from polymarket.execution.journal import JsonlWriter


class Deduplicator:
    def __init__(
        self, journal: JsonlWriter, today_utc: date | None = None
    ) -> None:
        self._seen: set[str] = set()
        self._lock: threading.Lock = threading.Lock()
        self._rebuild_from_journal(journal, today_utc)

    def _rebuild_from_journal(
        self, journal: JsonlWriter, today_utc: date | None
    ) -> None:
        today = today_utc if today_utc is not None else datetime.now(timezone.utc).date()
        today_count = self._absorb_day(journal, today)
        if today_count == 0:
            self._absorb_day(journal, today - timedelta(days=1))

    def _absorb_day(self, journal: JsonlWriter, day: date) -> int:
        absorbed = 0
        for event in journal.read_today(today_utc=day):
            if event.get("event_type") != "LEADER_FILL_OBSERVED":
                continue
            tx = event.get("transaction_hash")
            if isinstance(tx, str):
                self._seen.add(tx)
                absorbed += 1
        return absorbed

    def is_duplicate(self, transaction_hash: str) -> bool:
        with self._lock:
            return transaction_hash in self._seen

    def mark_seen(self, transaction_hash: str) -> None:
        with self._lock:
            self._seen.add(transaction_hash)

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)
