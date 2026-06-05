from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pytest

from polymarket.execution.journal import JsonlWriter
from polymarket.execution.maker.resolution_handler import (
    ResolutionHandler,
    _redeem_amounts,
)


def _response(payload) -> io.BytesIO:
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


class _Redeemer:
    def __init__(self, tx_hash: str = "0xredeemed") -> None:
        self.calls: list[dict] = []
        self.tx_hash = tx_hash

    def redeem(self, row: dict) -> str:
        self.calls.append(row)
        return self.tx_hash


class _FailingRedeemer:
    def redeem(self, row: dict) -> str:  # noqa: ARG002
        raise RuntimeError("boom")


def test_resolution_and_redeemable_poll_logs_and_redeems(tmp_path: Path) -> None:
    journal = JsonlWriter(tmp_path, "resolution")
    redeemer = _Redeemer()

    def urlopen(url: str, **kwargs):  # noqa: ARG001
        url = getattr(url, "full_url", url)
        path = urlparse(url).path
        if path.endswith("/markets"):
            return _response([
                {"conditionId": "0xABC", "closed": True, "active": False}
            ])
        if path.endswith("/positions"):
            return _response([
                {
                    "conditionId": "0xABC",
                    "asset": "asset-yes",
                    "outcomeIndex": 0,
                    "size": "12.5",
                    "redeemable": True,
                    "negativeRisk": True,
                }
            ])
        raise AssertionError(url)

    handler = ResolutionHandler(
        condition_ids_provider=lambda: {"0xabc"},
        funder="0x" + "1" * 40,
        gamma_url="https://gamma-api.polymarket.com",
        data_url="https://data-api.polymarket.com",
        journal=journal,
        redeemer=redeemer,
        urlopen_fn=urlopen,
    )
    handler.poll_once()

    events = list(journal.read_today(today_utc=datetime.now(timezone.utc).date()))
    assert any(e["event_type"] == "MARKET_RESOLVED" for e in events)
    assert any(e["event_type"] == "POSITION_REDEEMABLE" for e in events)
    assert any(e["event_type"] == "POSITION_REDEEMED" for e in events)
    assert handler.is_resolved("0xabc")
    assert len(redeemer.calls) == 1


def test_redemption_failure_is_non_blocking_and_logged(tmp_path: Path) -> None:
    journal = JsonlWriter(tmp_path, "resolution-fail")

    def urlopen(url: str, **kwargs):  # noqa: ARG001
        url = getattr(url, "full_url", url)
        path = urlparse(url).path
        if path.endswith("/markets"):
            return _response([])
        return _response([
            {
                "conditionId": "0xABC",
                "asset": "asset-yes",
                "outcomeIndex": 1,
                "size": "1",
                "redeemable": True,
                "negativeRisk": True,
            }
        ])

    handler = ResolutionHandler(
        condition_ids_provider=lambda: set(),
        funder="0x" + "1" * 40,
        gamma_url="https://gamma-api.polymarket.com",
        data_url="https://data-api.polymarket.com",
        journal=journal,
        redeemer=_FailingRedeemer(),
        urlopen_fn=urlopen,
    )
    handler.poll_once()

    events = list(journal.read_today(today_utc=datetime.now(timezone.utc).date()))
    failures = [e for e in events if e["event_type"] == "REDEMPTION_FAILED"]
    assert len(failures) == 1
    assert failures[0]["reason"] == "redeem_positions_failed"


def test_redeem_amounts_uses_outcome_index_and_six_decimals() -> None:
    assert _redeem_amounts({"outcomeIndex": 1, "size": "1.25"}) == [0, 1_250_000]


def test_redeem_amounts_supports_three_plus_outcome_markets() -> None:
    # NegRisk events routinely have 3+ candidates; outcome_index 3 must size
    # the amounts vector to length 4 instead of raising.
    assert _redeem_amounts({"outcomeIndex": 3, "size": "2"}) == [
        0, 0, 0, 2_000_000,
    ]
    assert _redeem_amounts({"outcomeIndex": 0, "size": "0.5"}) == [500_000]


def test_redeem_amounts_rejects_negative_outcome_index() -> None:
    with pytest.raises(ValueError, match="outcomeIndex must be >= 0"):
        _redeem_amounts({"outcomeIndex": -1, "size": "1"})
