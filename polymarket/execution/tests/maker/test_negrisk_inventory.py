from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

from polymarket.execution.journal import JsonlWriter
from polymarket.execution.maker.negrisk_inventory import NegRiskInventoryTracker


def _response(payload) -> io.BytesIO:
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


def test_activity_poll_applies_merge_delta_and_persists(tmp_path: Path) -> None:
    payload = {
        "data": [
            {
                "type": "SPLIT",
                "conditionId": "0xABC",
                "size": "5",
                "transactionHash": "0xsplit",
                "timestamp": 1,
            },
            {
                "type": "MERGE",
                "conditionId": "0xABC",
                "size": "2",
                "transactionHash": "0xmerge",
                "timestamp": 2,
            },
        ]
    }
    journal = JsonlWriter(tmp_path / "journal", "maker-test")
    tracker = NegRiskInventoryTracker(
        funder="0x" + "1" * 40,
        data_url="https://data-api.polymarket.com",
        state_path=tmp_path / "inventory.jsonl",
        journal=journal,
        urlopen_fn=lambda *a, **k: _response(payload),
    )

    assert tracker.poll_once() == 2
    assert tracker.get_basket_exposure("0xabc") == 3.0
    assert tracker.poll_once() == 0

    reloaded = NegRiskInventoryTracker(
        funder="0x" + "1" * 40,
        data_url="https://data-api.polymarket.com",
        state_path=tmp_path / "inventory.jsonl",
        urlopen_fn=lambda *a, **k: _response({"data": []}),
    )
    assert reloaded.get_basket_exposure("0xabc") == 3.0

    events = list(journal.read_today(today_utc=datetime.now(timezone.utc).date()))
    updates = [e for e in events if e["event_type"] == "BASKET_INVENTORY_UPDATED"]
    assert [e["delta"] for e in updates] == [5.0, -2.0]
