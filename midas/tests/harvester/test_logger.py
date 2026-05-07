from __future__ import annotations

import json

import pytest

from harvester.logger import StructuredJSONLogger, build_logger

pytestmark = [pytest.mark.unit, pytest.mark.harvester]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_unknown_level_raises() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        StructuredJSONLogger(level="TRACE")


def test_case_insensitive_level() -> None:
    logger = StructuredJSONLogger(level="info")
    assert logger._min_level == 1  # INFO == 1


# ---------------------------------------------------------------------------
# Emit format
# ---------------------------------------------------------------------------


def test_info_emits_valid_json(capsys) -> None:
    logger = StructuredJSONLogger("DEBUG")
    logger.info("test_event", foo="bar", count=42)
    out = capsys.readouterr().out
    record = json.loads(out.strip())
    assert record["level"] == "INFO"
    assert record["event"] == "test_event"
    assert record["foo"] == "bar"
    assert record["count"] == 42


def test_timestamp_format(capsys) -> None:
    logger = StructuredJSONLogger("DEBUG")
    logger.info("ts_check")
    out = capsys.readouterr().out
    record = json.loads(out.strip())
    ts = record["ts"]
    # ISO-8601 format: 2024-01-15T12:34:56.789Z
    assert "T" in ts
    assert ts.endswith("Z")
    assert len(ts) == 24  # YYYY-MM-DDTHH:MM:SS.mmmZ


def test_each_call_emits_one_line(capsys) -> None:
    logger = StructuredJSONLogger("DEBUG")
    logger.info("first")
    logger.info("second")
    out = capsys.readouterr().out
    lines = [l for l in out.strip().splitlines() if l]
    assert len(lines) == 2


def test_non_serializable_field_uses_str(capsys) -> None:
    logger = StructuredJSONLogger("DEBUG")
    logger.info("obj_field", val=object())
    out = capsys.readouterr().out
    record = json.loads(out.strip())
    assert "val" in record  # converted via default=str, not raised


# ---------------------------------------------------------------------------
# Level filtering
# ---------------------------------------------------------------------------


def test_debug_not_emitted_at_info_level(capsys) -> None:
    logger = StructuredJSONLogger("INFO")
    logger.debug("should_be_filtered")
    out = capsys.readouterr().out
    assert out == ""


def test_info_emitted_at_info_level(capsys) -> None:
    logger = StructuredJSONLogger("INFO")
    logger.info("should_appear")
    assert capsys.readouterr().out != ""


def test_warning_emitted_at_info_level(capsys) -> None:
    logger = StructuredJSONLogger("INFO")
    logger.warning("should_appear")
    record = json.loads(capsys.readouterr().out.strip())
    assert record["level"] == "WARNING"


def test_error_emitted_at_warning_level(capsys) -> None:
    logger = StructuredJSONLogger("WARNING")
    logger.error("should_appear")
    record = json.loads(capsys.readouterr().out.strip())
    assert record["level"] == "ERROR"


def test_info_not_emitted_at_warning_level(capsys) -> None:
    logger = StructuredJSONLogger("WARNING")
    logger.info("should_be_filtered")
    assert capsys.readouterr().out == ""


def test_debug_emitted_at_debug_level(capsys) -> None:
    logger = StructuredJSONLogger("DEBUG")
    logger.debug("should_appear")
    record = json.loads(capsys.readouterr().out.strip())
    assert record["level"] == "DEBUG"


# ---------------------------------------------------------------------------
# build_logger
# ---------------------------------------------------------------------------


def test_build_logger_default_is_info(monkeypatch) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    logger = build_logger()
    assert logger._min_level == 1  # INFO


def test_build_logger_respects_env_var(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    logger = build_logger()
    assert logger._min_level == 0  # DEBUG


def test_build_logger_case_insensitive_env(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "warning")
    logger = build_logger()
    assert logger._min_level == 2  # WARNING


def test_build_logger_default_level_param(monkeypatch) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    logger = build_logger(default_level="ERROR")
    assert logger._min_level == 3  # ERROR
