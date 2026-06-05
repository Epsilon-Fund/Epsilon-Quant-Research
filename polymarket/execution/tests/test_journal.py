"""Tests for polymarket/execution/journal/."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from polymarket.execution.journal import (
    LEADER_FILL_DROP_REASONS,
    WATCHER_MALFORMED_REASONS,
    AmbiguousSubmit,
    FillRecorded,
    JsonlWriter,
    KillSwitchTripped,
    LeaderFillDropped,
    LeaderFillObserved,
    MirrorSignalEmitted,
    OrderSubmitted,
    RiskHalt,
    SignalReceived,
    WatcherConnectFailed,
    WatcherMalformedMessage,
    WatcherReconnected,
    WatcherStarted,
    WatcherStopped,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_happy_path_round_trip(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "exec-test")
    ts = _now()
    writer.write(SignalReceived(ts_utc=ts, signal_kind="ENTRY",
                                condition_id="0xcond", asset_id="42"))
    writer.write(OrderSubmitted(ts_utc=ts, client_order_id="coid-1",
                                condition_id="0xcond", asset_id="42",
                                side="BUY", size=1.0, price=0.5,
                                order_type="FOK"))
    writer.write(FillRecorded(ts_utc=ts, transaction_hash="0xtx",
                              condition_id="0xcond", asset_id="42",
                              side="BUY", size=1.0, price=0.5,
                              proxy_wallet="0xleader"))
    writer.close()

    events = list(writer.read_today(today_utc=ts.date()))
    assert len(events) == 3
    assert events[0]["event_type"] == "SIGNAL_RECEIVED"
    assert events[1]["event_type"] == "ORDER_SUBMITTED"
    assert events[2]["event_type"] == "FILL_RECORDED"
    parsed_ts = datetime.fromisoformat(events[0]["ts_utc"])
    assert parsed_ts.tzinfo is not None
    assert parsed_ts.utcoffset() == timedelta(0)
    assert events[1]["client_order_id"] == "coid-1"
    assert events[2]["proxy_wallet"] == "0xleader"


def test_daily_rotation(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "rot")
    day_n = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    day_n1 = datetime(2026, 5, 8, 0, 30, tzinfo=timezone.utc)
    writer.write(KillSwitchTripped(ts_utc=day_n, path="/tmp/k"))
    writer.write(KillSwitchTripped(ts_utc=day_n1, path="/tmp/k"))
    writer.close()

    file_n = tmp_path / "rot-2026-05-07.jsonl"
    file_n1 = tmp_path / "rot-2026-05-08.jsonl"
    assert file_n.exists()
    assert file_n1.exists()

    events_n1 = list(writer.read_today(today_utc=day_n1.date()))
    assert len(events_n1) == 1


def test_concurrent_writes_no_corruption(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "concurrent")
    ts = _now()

    def worker(thread_id: int) -> None:
        for i in range(50):
            writer.write(WatcherReconnected(ts_utc=ts,
                                            gap_seconds=float(thread_id * 100 + i)))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    writer.close()

    file_path = tmp_path / f"concurrent-{ts.date().isoformat()}.jsonl"
    raw_lines = file_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 200
    for line in raw_lines:
        record = json.loads(line)
        assert record["event_type"] == "WATCHER_RECONNECTED"
        assert isinstance(record["gap_seconds"], float)


def test_naive_datetime_rejected() -> None:
    naive = datetime(2026, 5, 7, 12, 0)
    with pytest.raises(ValueError, match="tz-aware"):
        SignalReceived(ts_utc=naive, signal_kind="ENTRY",
                       condition_id="0xc", asset_id="42")


def test_non_utc_datetime_rejected() -> None:
    eastern = timezone(timedelta(hours=-5))
    aware = datetime(2026, 5, 7, 12, 0, tzinfo=eastern)
    with pytest.raises(ValueError, match="UTC"):
        RiskHalt(ts_utc=aware, reason="x", detail="y")


def test_leader_fill_dropped_valid_reasons() -> None:
    ts = _now()
    LeaderFillDropped(ts_utc=ts, transaction_hash="0xtx", reason="no_position")
    LeaderFillDropped(ts_utc=ts, transaction_hash="0xtx", reason="duplicate")


def test_leader_fill_dropped_invalid_reason_raises() -> None:
    ts = _now()
    with pytest.raises(ValueError, match="LEADER_FILL_DROP_REASONS"):
        LeaderFillDropped(ts_utc=ts, transaction_hash="0xtx", reason="explosion")


def test_leader_fill_drop_reasons_constant() -> None:
    assert LEADER_FILL_DROP_REASONS == frozenset({
        "no_position", "duplicate", "leader_no_position",
    })


def test_leader_fill_dropped_leader_no_position_round_trip(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "lfdrop")
    ts = _now()
    writer.write(LeaderFillDropped(
        ts_utc=ts, transaction_hash="0xtx", reason="leader_no_position",
    ))
    writer.close()
    events = list(writer.read_today(today_utc=ts.date()))
    assert len(events) == 1
    assert events[0]["reason"] == "leader_no_position"


def test_mirror_signal_emitted_round_trip(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "mse")
    ts = _now()
    writer.write(MirrorSignalEmitted(
        ts_utc=ts,
        signal_id="abc123",
        kind="ENTRY",
        condition_id="0xcond",
        asset_id="42",
        side="BUY",
        target_size_shares=166.6667,
        leader_fill_price=0.30,
    ))
    writer.close()
    events = list(writer.read_today(today_utc=ts.date()))
    assert len(events) == 1
    assert events[0]["event_type"] == "MIRROR_SIGNAL_EMITTED"
    assert events[0]["signal_id"] == "abc123"
    assert events[0]["kind"] == "ENTRY"
    assert events[0]["target_size_shares"] == pytest.approx(166.6667)


def test_read_today_tolerates_partial_trailing_line(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "partial")
    ts = _now()
    writer.write(SignalReceived(ts_utc=ts, signal_kind="ENTRY",
                                condition_id="0xc", asset_id="42"))
    writer.close()

    file_path = tmp_path / f"partial-{ts.date().isoformat()}.jsonl"
    with open(file_path, "a", encoding="utf-8") as f:
        f.write('{"event_type": "TRUNCATED",')  # malformed, no newline

    events = list(writer.read_today(today_utc=ts.date()))
    assert len(events) == 1
    assert events[0]["event_type"] == "SIGNAL_RECEIVED"


def test_read_today_missing_file_returns_empty(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "missing")
    events = list(writer.read_today())
    assert events == []


def test_ambiguous_submit_round_trip(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "amb")
    ts = _now()
    writer.write(AmbiguousSubmit(ts_utc=ts, client_order_id="coid-x",
                                 detail="timeout"))
    writer.close()
    events = list(writer.read_today(today_utc=ts.date()))
    assert len(events) == 1
    assert events[0]["event_type"] == "AMBIGUOUS_SUBMIT"
    assert events[0]["client_order_id"] == "coid-x"


def test_watcher_started_round_trip(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "ws-events")
    ts = _now()
    writer.write(WatcherStarted(ts_utc=ts, ws_url="wss://x"))
    writer.close()
    events = list(writer.read_today(today_utc=ts.date()))
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_STARTED"
    assert events[0]["ws_url"] == "wss://x"


def test_watcher_stopped_round_trip(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "ws-events-2")
    ts = _now()
    writer.write(WatcherStopped(ts_utc=ts))
    writer.close()
    events = list(writer.read_today(today_utc=ts.date()))
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_STOPPED"


def test_watcher_malformed_round_trip(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "ws-mal")
    ts = _now()
    writer.write(WatcherMalformedMessage(ts_utc=ts, raw="garbage", reason="json_parse"))
    writer.close()
    events = list(writer.read_today(today_utc=ts.date()))
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_MALFORMED_MESSAGE"
    assert events[0]["reason"] == "json_parse"
    assert events[0]["raw"] == "garbage"


def test_watcher_malformed_invalid_reason_raises() -> None:
    ts = _now()
    with pytest.raises(ValueError, match="WATCHER_MALFORMED_REASONS"):
        WatcherMalformedMessage(ts_utc=ts, raw="x", reason="explosion")


def test_watcher_malformed_reasons_constant() -> None:
    assert WATCHER_MALFORMED_REASONS == frozenset({
        "json_parse", "missing_field", "wrong_type", "wrong_topic",
        "payload_parse",
    })


def test_watcher_connect_failed_round_trip(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "ws-cf")
    ts = _now()
    writer.write(WatcherConnectFailed(
        ts_utc=ts,
        ws_url="wss://ws-live-data.polymarket.com",
        attempt_number=2,
        error="InvalidStatus(404)",
    ))
    writer.close()
    events = list(writer.read_today(today_utc=ts.date()))
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_CONNECT_FAILED"
    assert events[0]["ws_url"] == "wss://ws-live-data.polymarket.com"
    assert events[0]["attempt_number"] == 2
    assert events[0]["error"] == "InvalidStatus(404)"


def test_watcher_malformed_payload_parse_round_trip(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "ws-mal-2")
    ts = _now()
    writer.write(WatcherMalformedMessage(
        ts_utc=ts, raw="garbage", reason="payload_parse"
    ))
    writer.close()
    events = list(writer.read_today(today_utc=ts.date()))
    assert len(events) == 1
    assert events[0]["reason"] == "payload_parse"


def test_leader_fill_observed_round_trip(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "lfo")
    ts = _now()
    obs = _now()
    writer.write(LeaderFillObserved(
        ts_utc=ts,
        transaction_hash="0xtx",
        condition_id="0xcond",
        asset_id="42",
        side="BUY",
        size=10.0,
        price=0.5,
        proxy_wallet="0xleader",
        observed_at_utc=obs,
    ))
    writer.close()
    events = list(writer.read_today(today_utc=ts.date()))
    assert len(events) == 1
    assert events[0]["event_type"] == "LEADER_FILL_OBSERVED"
    assert events[0]["transaction_hash"] == "0xtx"
    parsed_obs = datetime.fromisoformat(events[0]["observed_at_utc"])
    assert parsed_obs.tzinfo is not None
    assert parsed_obs.utcoffset() == timedelta(0)


def test_leader_fill_observed_naive_observed_at_raises() -> None:
    ts = _now()
    naive = datetime(2026, 5, 7, 12, 0)
    with pytest.raises(ValueError, match="tz-aware"):
        LeaderFillObserved(
            ts_utc=ts, transaction_hash="0xtx", condition_id="0xc",
            asset_id="42", side="BUY", size=1.0, price=0.5,
            proxy_wallet="0xl", observed_at_utc=naive,
        )


def test_context_manager_closes(tmp_path: Path) -> None:
    ts = _now()
    with JsonlWriter(tmp_path, "ctx") as writer:
        writer.write(KillSwitchTripped(ts_utc=ts, path="/tmp/k"))
    with pytest.raises(RuntimeError, match="closed"):
        writer.write(KillSwitchTripped(ts_utc=ts, path="/tmp/k"))
