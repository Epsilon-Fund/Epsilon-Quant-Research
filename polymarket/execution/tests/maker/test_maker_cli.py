from __future__ import annotations

import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polymarket.execution.journal import JsonlWriter
from polymarket.execution.maker import cli as maker_cli


def _env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        "POLYMARKET_LEADER_ADDRESS": "0x" + "1" * 40,
        "POLYMARKET_PRIVATE_KEY": "0x" + "1" * 64,
        "POLYMARKET_API_KEY": "api-key",
        "POLYMARKET_API_SECRET": "api-secret",
        "POLYMARKET_PASSPHRASE": "passphrase",
        "POLYMARKET_FUNDER": "0x" + "2" * 40,
        "POLYMARKET_JOURNAL_DIR": str(tmp_path),
        "POLYMARKET_MAKER_CONDITION_ID": "0x" + "a" * 40,
        "POLYMARKET_VENUE": "fake",
        "POLYMARKET_MAKER_MAX_RUNTIME_SECONDS": "0.3",
    }
    env.update(overrides)
    return env


def _read_maker_events(tmp_path: Path) -> list[dict]:
    journal = JsonlWriter(tmp_path, "maker")
    return list(journal.read_today(today_utc=datetime.now(timezone.utc).date()))


def test_cli_missing_condition_id_returns_config_error(tmp_path: Path) -> None:
    env = _env(tmp_path)
    del env["POLYMARKET_MAKER_CONDITION_ID"]
    assert maker_cli.main(env) == 2


def test_cli_runs_and_logs_session_lifecycle(tmp_path: Path, monkeypatch) -> None:
    # No real network: every poll/lookup fails gracefully and the loop still
    # boots all three threads and shuts down cleanly.
    def _no_network(*_args, **_kwargs):
        raise urllib.error.URLError("no network in test")

    monkeypatch.setattr(urllib.request, "urlopen", _no_network)

    exit_code = maker_cli.main(_env(tmp_path))
    assert exit_code == 0

    events = _read_maker_events(tmp_path)
    started = [e for e in events if e["event_type"] == "MAKER_SESSION_STARTED"]
    stopped = [e for e in events if e["event_type"] == "MAKER_SESSION_STOPPED"]
    assert len(started) == 1
    assert started[0]["condition_id"] == "0x" + "a" * 40
    assert started[0]["venue"] == "fake"
    assert len(stopped) == 1
    assert stopped[0]["reason"] == "max_runtime_reached"
    # Market lookup failed (no network), so the engine logged skips rather
    # than submitting any quotes — proves the run loop executed.
    assert any(e["event_type"] == "MAKER_QUOTE_SKIPPED" for e in events)


def test_check_auth_reads_open_orders_without_submit(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    class _AuthResult:
        venue_open_client_order_ids = ("coid-1", "coid-2")

    class _AuthVenue:
        def __init__(self) -> None:
            self.read_calls = 0
            self.submit_calls = 0
            self.stop_calls = 0

        def reconcile_open_orders(self, expected: set[str]) -> _AuthResult:
            assert expected == set()
            self.read_calls += 1
            return _AuthResult()

        def submit_order(self, **kwargs: Any) -> None:  # noqa: ARG002
            self.submit_calls += 1
            raise AssertionError("auth check must not submit")

        def stop(self) -> None:
            self.stop_calls += 1

    venue = _AuthVenue()

    def _build_venue(mode: str, *_args: Any, **_kwargs: Any) -> _AuthVenue:
        assert mode == "real"
        return venue

    monkeypatch.setattr(maker_cli, "build_venue_adapter", _build_venue)

    assert maker_cli.main(_env(tmp_path), check_auth=True) == 0

    captured = capsys.readouterr()
    assert "open_orders=2" in captured.out
    assert venue.read_calls == 1
    assert venue.submit_calls == 0
    assert venue.stop_calls == 1
