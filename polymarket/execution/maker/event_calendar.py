"""Static scheduled-event calendar for maker telemetry.

The calendar is intentionally manual. It accepts JSON or a tiny YAML
subset shaped as:

    events:
      - name: Example
        timestamp_utc: "2026-06-17T18:00:00Z"
        category: fomc

No PyYAML dependency is used; this execution module stays stdlib-only
unless the project explicitly adds dependencies through uv.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    name: str
    timestamp_utc: datetime
    category: str = "politics"
    source: str = "manual"

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        if self.timestamp_utc.utcoffset() != timedelta(0):
            raise ValueError("timestamp_utc must be UTC")


class EventCalendar:
    """In-memory scheduled-event proximity checks."""

    def __init__(self, events: list[ScheduledEvent] | None = None) -> None:
        self._events: tuple[ScheduledEvent, ...] = tuple(events or [])

    @classmethod
    def from_file(cls, path: Path | str) -> EventCalendar:
        file_path = Path(path)
        if not file_path.exists():
            return cls([])
        text = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            payload = _parse_simple_yaml(text)
        rows = payload.get("events", payload) if isinstance(payload, dict) else payload
        events = [_event_from_row(row) for row in rows if isinstance(row, dict)]
        return cls(events)

    @classmethod
    def default(cls) -> EventCalendar:
        return cls.from_file(Path(__file__).with_name("politics_events.yaml"))

    @property
    def events(self) -> tuple[ScheduledEvent, ...]:
        return self._events

    def is_event_proximate(
        self, timestamp: datetime, window_minutes: int = 30
    ) -> bool:
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        ts_utc = timestamp.astimezone(timezone.utc)
        window = timedelta(minutes=window_minutes)
        return any(abs(event.timestamp_utc - ts_utc) <= window for event in self._events)


def _event_from_row(row: dict[str, Any]) -> ScheduledEvent:
    name = str(row.get("name", "")).strip()
    if not name:
        raise ValueError("calendar event missing name")
    raw_ts = row.get("timestamp_utc") or row.get("timestamp")
    if raw_ts is None:
        raise ValueError(f"calendar event {name!r} missing timestamp_utc")
    return ScheduledEvent(
        name=name,
        timestamp_utc=_parse_datetime(str(raw_ts)),
        category=str(row.get("category", "politics")),
        source=str(row.get("source", "manual")),
    )


def _parse_datetime(value: str) -> datetime:
    cleaned = value.strip().strip('"').strip("'")
    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_simple_yaml(text: str) -> dict[str, list[dict[str, str]]]:
    events: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_events = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped == "events:":
            in_events = True
            continue
        if not in_events:
            continue
        if stripped.startswith("- "):
            if current:
                events.append(current)
            current = {}
            remainder = stripped[2:].strip()
            if remainder:
                key, value = _split_yaml_pair(remainder)
                current[key] = value
            continue
        if current is not None and ":" in stripped:
            key, value = _split_yaml_pair(stripped)
            current[key] = value
    if current:
        events.append(current)
    return {"events": events}


def _split_yaml_pair(line: str) -> tuple[str, str]:
    key, value = line.split(":", 1)
    return key.strip(), value.strip().strip('"').strip("'")
