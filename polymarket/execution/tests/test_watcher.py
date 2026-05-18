"""Tests for watcher/ — schema check, leader filter, queueing, reconnect logging."""
from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import (
    JsonlWriter,
    LeaderFillObserved,
)
from polymarket.execution.watcher.leader_watcher import LeaderWatcher

_LEADER = "0x" + "a" * 40
_FUNDER = "0x" + "b" * 40

_BASE_CONFIG_FIELDS: dict[str, Any] = {
    "leader_address": _LEADER,
    "private_key": "0xpk",
    "api_key": "k",
    "api_secret": "s",
    "passphrase": "p",
    "funder": _FUNDER,
    "chain_id": 137,
    "signature_type": 1,
    "clob_url": "https://clob.polymarket.com",
    "gamma_url": "https://gamma-api.polymarket.com",
    "data_url": "https://data-api.polymarket.com",
    "ws_url": "wss://ws-live-data.polymarket.com",
    "max_capital_usd": 100.0,
    "per_trade_cap_usd": 20.0,
    "per_market_cap_usd": 50.0,
    "sizing_usd": 50.0,
    "max_open_positions": 3,
    "default_order_type": "FOK",
    "pricing_mode": "leader_fill",
    "price_deviation_pct": 2.0,
    "daily_loss_halt_usd": 200.0,
    "killswitch_path": Path("/tmp/polymarket_killswitch"),
    "journal_dir": Path("./journal_logs"),
    "log_level": "INFO",
    "max_real_orders": 5,
    "require_operator_confirm": False,
}


def _config(**overrides: Any) -> ExecutionConfig:
    return ExecutionConfig(**{**_BASE_CONFIG_FIELDS, **overrides})


def _trade_msg(
    proxy_wallet: str = _LEADER,
    *,
    transaction_hash: str = "0xtx",
    condition_id: str = "0xcond",
    asset: str = "42",
    side: str = "BUY",
    size: float | str = 10.0,
    price: float | str = 0.5,
    timestamp: int = 1730000000,
    payload_as_string: bool = True,
    **payload_overrides: Any,
) -> str:
    payload = {
        "transactionHash": transaction_hash,
        "conditionId": condition_id,
        "asset": asset,
        "side": side,
        "size": size,
        "price": price,
        "proxyWallet": proxy_wallet,
        "timestamp": timestamp,
        **payload_overrides,
    }
    # Real RTDS has been observed sending payload as a JSON-encoded
    # string; default reflects that. Pass payload_as_string=False to
    # exercise the defensive dict path.
    payload_field = json.dumps(payload) if payload_as_string else payload
    return json.dumps({
        "topic": "activity",
        "type": "trades",
        "payload": payload_field,
        "connection_id": "conn-1",
        "timestamp": 1730000000,
    })


class _FakeRtdsClient:
    """Stand-in for RtdsClient — records calls, never opens sockets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.started = False
        self.stopped = False
        self.on_message = kwargs.get("on_message")
        self.on_reconnect = kwargs.get("on_reconnect")
        self.on_connect_failed = kwargs.get("on_connect_failed")

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def _watcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    leader_address: str = _LEADER,
    queue_size: int = 100,
) -> tuple[LeaderWatcher, JsonlWriter, "queue.Queue[LeaderFillObserved]", _FakeRtdsClient]:
    monkeypatch.setattr(
        "polymarket.execution.watcher.leader_watcher.RtdsClient", _FakeRtdsClient
    )
    journal = JsonlWriter(tmp_path, "watcher-test")
    q: queue.Queue[LeaderFillObserved] = queue.Queue(maxsize=queue_size)
    cfg = _config(leader_address=leader_address)
    watcher = LeaderWatcher(config=cfg, journal=journal, fill_queue=q)
    fake = watcher._client  # type: ignore[attr-defined]
    assert isinstance(fake, _FakeRtdsClient)
    return watcher, journal, q, fake


def _today_events(journal: JsonlWriter) -> list[dict[str, Any]]:
    return list(journal.read_today())


# 1. Happy path
def test_leader_fill_observed_journaled_and_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    watcher._handle_message(_trade_msg())
    journal.close()
    events = _today_events(journal)
    fill_events = [e for e in events if e["event_type"] == "LEADER_FILL_OBSERVED"]
    assert len(fill_events) == 1
    assert fill_events[0]["transaction_hash"] == "0xtx"
    assert fill_events[0]["proxy_wallet"] == _LEADER
    assert q.qsize() == 1
    queued = q.get_nowait()
    assert isinstance(queued, LeaderFillObserved)
    assert queued.transaction_hash == "0xtx"


# 2. Non-leader fill silently dropped
def test_non_leader_fill_silently_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    watcher._handle_message(_trade_msg(proxy_wallet="0x" + "c" * 40))
    journal.close()
    assert _today_events(journal) == []
    assert q.qsize() == 0


# 3. Address case-insensitive match
def test_address_case_insensitive_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    upper_leader = "0x" + "A" * 40  # uppercase 0xAAAA...
    watcher._handle_message(_trade_msg(proxy_wallet=upper_leader))
    journal.close()
    fill_events = [e for e in _today_events(journal)
                   if e["event_type"] == "LEADER_FILL_OBSERVED"]
    assert len(fill_events) == 1
    assert q.qsize() == 1


# 4. Malformed JSON
def test_malformed_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    watcher._handle_message("not json")
    journal.close()
    events = _today_events(journal)
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_MALFORMED_MESSAGE"
    assert events[0]["reason"] == "json_parse"
    assert q.qsize() == 0


# 4b. Empty / whitespace-only frames are silently dropped (RTDS sends
# one such frame after subscribe; we don't want it polluting the journal).
@pytest.mark.parametrize("raw", ["", "   ", "\n", "\t\n  "])
def test_empty_frame_silently_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    watcher._handle_message(raw)
    journal.close()
    assert _today_events(journal) == []
    assert q.qsize() == 0


# 5. Missing top-level "type" field
def test_missing_type_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    raw = json.dumps({"topic": "activity", "payload": {"proxyWallet": _LEADER}})
    watcher._handle_message(raw)
    journal.close()
    events = _today_events(journal)
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_MALFORMED_MESSAGE"
    assert events[0]["reason"] == "missing_field"
    assert q.qsize() == 0


# 6. Wrong topic — topic right, type wrong (or topic wrong) → wrong_topic
def test_wrong_topic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    raw = json.dumps({
        "topic": "activity",
        "type": "orders_matched",
        "payload": {"proxyWallet": _LEADER},
    })
    watcher._handle_message(raw)
    journal.close()
    events = _today_events(journal)
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_MALFORMED_MESSAGE"
    assert events[0]["reason"] == "wrong_topic"
    assert q.qsize() == 0


# 7. Missing proxyWallet
def test_missing_proxy_wallet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    raw = json.dumps({
        "topic": "activity",
        "type": "trades",
        "payload": {"transactionHash": "0xtx"},
    })
    watcher._handle_message(raw)
    journal.close()
    events = _today_events(journal)
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_MALFORMED_MESSAGE"
    assert events[0]["reason"] == "missing_field"
    assert q.qsize() == 0


# 8. Leader-matched but missing required trade field
def test_leader_match_missing_trade_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    raw = json.dumps({
        "topic": "activity",
        "type": "trades",
        "payload": {
            "proxyWallet": _LEADER,
            # transactionHash deliberately missing
            "conditionId": "0xcond",
            "asset": "42",
            "side": "BUY",
            "size": 10.0,
            "price": 0.5,
            "timestamp": 1730000000,
        },
    })
    watcher._handle_message(raw)
    journal.close()
    events = _today_events(journal)
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_MALFORMED_MESSAGE"
    assert events[0]["reason"] == "missing_field"
    assert q.qsize() == 0


# 9. Long raw input truncated to 500 chars
def test_long_raw_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    huge = "{" + ("x" * 10_000)  # invalid JSON, very long
    watcher._handle_message(huge)
    journal.close()
    events = _today_events(journal)
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_MALFORMED_MESSAGE"
    assert len(events[0]["raw"]) == 500


# 10. Queue blocking under backpressure
def test_queue_blocks_when_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch, queue_size=2)
    # Two leader fills land — queue full.
    watcher._handle_message(_trade_msg(transaction_hash="0xtx1"))
    watcher._handle_message(_trade_msg(transaction_hash="0xtx2"))
    assert q.qsize() == 2

    started = threading.Event()
    finished = threading.Event()

    def push_third() -> None:
        started.set()
        watcher._handle_message(_trade_msg(transaction_hash="0xtx3"))
        finished.set()

    t = threading.Thread(target=push_third, daemon=True)
    t.start()
    assert started.wait(timeout=1.0)
    assert not finished.wait(timeout=0.2)  # should still be blocked

    # Drain one — the third call should now complete.
    q.get_nowait()
    assert finished.wait(timeout=1.0)
    t.join(timeout=1.0)
    assert q.qsize() == 2  # one left over from the pre-block state, plus the unblocked one


# 11. Reconnect handler
def test_reconnect_journals_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, _q, _ = _watcher(tmp_path, monkeypatch)
    watcher._handle_reconnect(12.5)
    journal.close()
    events = _today_events(journal)
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_RECONNECTED"
    assert events[0]["gap_seconds"] == 12.5


# 11b. Connect-failed handler
def test_connect_failed_journals_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, _q, _ = _watcher(tmp_path, monkeypatch)
    watcher._handle_connect_failed(
        "wss://ws-live-data.polymarket.com", 3, "InvalidStatus(404)"
    )
    journal.close()
    events = _today_events(journal)
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_CONNECT_FAILED"
    assert events[0]["ws_url"] == "wss://ws-live-data.polymarket.com"
    assert events[0]["attempt_number"] == 3
    assert events[0]["error"] == "InvalidStatus(404)"


# 12a. Happy path with dict payload (defensive against future protocol changes)
def test_dict_payload_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    watcher._handle_message(_trade_msg(payload_as_string=False))
    journal.close()
    fill_events = [e for e in _today_events(journal)
                   if e["event_type"] == "LEADER_FILL_OBSERVED"]
    assert len(fill_events) == 1
    assert q.qsize() == 1


# 12b. Malformed inner payload string → payload_parse
def test_malformed_inner_payload_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    raw = json.dumps({
        "topic": "activity",
        "type": "trades",
        "payload": "not valid json{{{",
        "connection_id": "c",
        "timestamp": 1,
    })
    watcher._handle_message(raw)
    journal.close()
    events = _today_events(journal)
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_MALFORMED_MESSAGE"
    assert events[0]["reason"] == "payload_parse"
    assert q.qsize() == 0


# 12c. Wrong payload type (list) → wrong_type
def test_payload_wrong_type_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    raw = json.dumps({
        "topic": "activity",
        "type": "trades",
        "payload": [1, 2, 3],
        "connection_id": "c",
        "timestamp": 1,
    })
    watcher._handle_message(raw)
    journal.close()
    events = _today_events(journal)
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_MALFORMED_MESSAGE"
    assert events[0]["reason"] == "wrong_type"
    assert q.qsize() == 0


# 12d. Wrong payload type (string that decodes to a list) → wrong_type
def test_payload_string_decodes_to_non_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, q, _ = _watcher(tmp_path, monkeypatch)
    raw = json.dumps({
        "topic": "activity",
        "type": "trades",
        "payload": "[1, 2, 3]",
        "connection_id": "c",
        "timestamp": 1,
    })
    watcher._handle_message(raw)
    journal.close()
    events = _today_events(journal)
    assert len(events) == 1
    assert events[0]["event_type"] == "WATCHER_MALFORMED_MESSAGE"
    assert events[0]["reason"] == "wrong_type"
    assert q.qsize() == 0


# 13. start/stop emits events; mocked client receives start/stop
def test_start_stop_emits_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher, journal, _q, fake = _watcher(tmp_path, monkeypatch)
    watcher.start()
    watcher.stop()
    journal.close()
    events = _today_events(journal)
    types = [e["event_type"] for e in events]
    assert "WATCHER_STARTED" in types
    assert "WATCHER_STOPPED" in types
    started = next(e for e in events if e["event_type"] == "WATCHER_STARTED")
    assert started["ws_url"] == _BASE_CONFIG_FIELDS["ws_url"]
    assert fake.started is True
    assert fake.stopped is True
