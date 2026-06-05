from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import IO


_LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}


class StructuredJSONLogger:
    """Writes one JSON line per call to stdout and optionally to a log file.

    Implements the executor.venue.StructuredLogger Protocol — drop-in replacement
    for NullLogger anywhere a StructuredLogger is accepted.

    Thread-safe on CPython: print() holds the GIL for the duration of the write,
    so concurrent calls from asyncio.to_thread workers produce whole lines.
    """

    __slots__ = ("_min_level", "_log_file")

    def __init__(self, level: str = "INFO", log_file: str | None = None) -> None:
        upper = level.upper()
        if upper not in _LEVELS:
            raise ValueError(
                f"Unknown log level {level!r}. Valid levels: {sorted(_LEVELS)}"
            )
        self._min_level = _LEVELS[upper]
        self._log_file: IO[str] | None = (
            open(log_file, "a", encoding="utf-8") if log_file else None  # noqa: WPS515
        )

    def debug(self, message: str, **fields: object) -> None:
        if self._min_level <= _LEVELS["DEBUG"]:
            self._emit("DEBUG", message, fields)

    def info(self, message: str, **fields: object) -> None:
        if self._min_level <= _LEVELS["INFO"]:
            self._emit("INFO", message, fields)

    def warning(self, message: str, **fields: object) -> None:
        if self._min_level <= _LEVELS["WARNING"]:
            self._emit("WARNING", message, fields)

    def error(self, message: str, **fields: object) -> None:
        if self._min_level <= _LEVELS["ERROR"]:
            self._emit("ERROR", message, fields)

    def _emit(self, level: str, event: str, fields: dict[str, object]) -> None:
        now = datetime.now(timezone.utc)
        ts = f"{now.strftime('%Y-%m-%dT%H:%M:%S')}.{now.microsecond // 1000:03d}Z"
        record: dict[str, object] = {"ts": ts, "level": level, "event": event}
        record.update(fields)
        line = json.dumps(record, default=str)
        print(line, flush=True)
        if self._log_file is not None:
            print(line, file=self._log_file, flush=True)


def build_logger(default_level: str = "INFO") -> StructuredJSONLogger:
    """Build a logger whose level is controlled by LOG_LEVEL; file output via LOG_FILE."""
    level = os.getenv("LOG_LEVEL", default_level).strip().upper() or default_level
    log_file = os.getenv("LOG_FILE", "").strip() or None
    return StructuredJSONLogger(level=level, log_file=log_file)
