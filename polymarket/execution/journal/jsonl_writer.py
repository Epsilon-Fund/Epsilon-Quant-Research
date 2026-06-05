"""Thread-safe JSONL writer with daily rotation.

Datetime convention: this codebase uses datetime.now(timezone.utc),
never datetime.utcnow(). Naive datetimes are rejected at event
construction; this writer only sees pre-validated ts_utc values.

File-handle caching: handles are keyed by date, not held as a single
rolling handle. Writing an event whose ts_utc.date() differs from any
cached handle's date opens a new handle and keeps prior ones open
until close(). This is correct for the PoC's low write volume; do not
optimise to a single rolling handle.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import date, datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from .events import _BaseEvent


class JsonlWriter:
    """Append-only JSONL event log with per-day file rotation.

    Files are named ``{service_name}-{YYYY-MM-DD}.jsonl`` based on the
    *event's* ``ts_utc.date()``. A late write whose ts_utc is yesterday
    opens (or reopens) yesterday's file rather than today's.

    Thread-safety: ``write`` is guarded by an internal lock so multiple
    threads (watcher, mirror) can write concurrently. ``read_today``
    does NOT take the lock — readers see flushed bytes and tolerate a
    partial trailing line by dropping it, never raising.
    """

    def __init__(self, directory: Path, service_name: str) -> None:
        self._directory: Path = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._service_name: str = service_name
        self._handles: dict[date, IO[str]] = {}
        self._lock: threading.Lock = threading.Lock()
        self._closed: bool = False

    def write(self, event: _BaseEvent) -> None:
        if self._closed:
            raise RuntimeError("JsonlWriter is closed")
        record = event.to_dict()
        line = json.dumps(record) + "\n"
        d = event.ts_utc.date()
        with self._lock:
            handle = self._handles.get(d)
            if handle is None:
                handle = open(self._path_for(d), "a", encoding="utf-8")
                self._handles[d] = handle
            handle.write(line)
            handle.flush()

    def read_today(self, today_utc: date | None = None) -> Iterator[dict[str, Any]]:
        target = today_utc if today_utc is not None else datetime.now(timezone.utc).date()
        path = self._path_for(target)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate a partial trailing line during concurrent writes.
                    continue

    def close(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                try:
                    handle.close()
                except Exception:
                    pass
            self._handles.clear()
            self._closed = True

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _path_for(self, d: date) -> Path:
        return self._directory / f"{self._service_name}-{d.isoformat()}.jsonl"
