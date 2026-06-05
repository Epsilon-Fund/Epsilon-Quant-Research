from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from polymarket.execution.maker.event_calendar import EventCalendar


def test_yaml_calendar_loads_and_detects_proximity(tmp_path: Path) -> None:
    path = tmp_path / "events.yaml"
    path.write_text(
        "\n".join([
            "events:",
            "  - name: FOMC",
            "    timestamp_utc: \"2026-06-17T18:00:00Z\"",
            "    category: fomc",
        ]),
        encoding="utf-8",
    )
    calendar = EventCalendar.from_file(path)
    assert len(calendar.events) == 1
    assert calendar.is_event_proximate(
        datetime(2026, 6, 17, 18, 4, tzinfo=timezone.utc),
        window_minutes=5,
    )
    assert not calendar.is_event_proximate(
        datetime(2026, 6, 17, 18, 6, tzinfo=timezone.utc),
        window_minutes=5,
    )


def test_missing_calendar_is_empty(tmp_path: Path) -> None:
    calendar = EventCalendar.from_file(tmp_path / "missing.yaml")
    assert calendar.events == ()
