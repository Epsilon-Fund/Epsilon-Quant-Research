"""Subscribes to a leader address's fills, emits LeaderFillObserved.

Pure stateless filtering: schema check → leader-address filter →
enqueue. NO dedup at this layer (per PLAN.md decision 8 — dedup
belongs in signal/, fed by the journal).

Field-name resilience: RTDS payloads documented in WS_PROBE_FINDINGS.md
use camelCase (`transactionHash`, `conditionId`, `proxyWallet`,
`asset`). This parser accepts a small set of equivalents
(`asset_id`, `condition_id`, `market`) so that an upstream rename
or Data-API-style payload doesn't silently fail.

Top-level message shape, confirmed against live RTDS on 2026-05-07:
two separate fields, ``topic`` (e.g. "activity") and ``type``
(e.g. "trades"), plus ``payload``, ``connection_id``, and a
server-side ``timestamp``. We require ``topic == "activity"`` and
``type == "trades"`` jointly; anything else is reason="wrong_topic".

Payload encoding: RTDS has been observed sending ``payload`` as
either a JSON-encoded string OR a nested dict. Step 3a normalises
both to a dict (``inner``) before downstream processing. Field
names inside ``inner`` are RTDS camelCase (``transactionHash``,
``conditionId``, ``asset``, ``proxyWallet``, ``side``, ``size``,
``price``, ``timestamp``). The probe at tests/probes/ws_leader_fills_probe.py
uses the same names; LeaderFillObserved on our side translates to
snake_case per PLAN.md convention.
"""
from __future__ import annotations

import json
import queue
from datetime import datetime, timezone
from typing import Any

from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import (
    JsonlWriter,
    LeaderFillObserved,
    WatcherConnectFailed,
    WatcherMalformedMessage,
    WatcherReconnected,
    WatcherStarted,
    WatcherStopped,
)

from .rtds_client import RtdsClient

_RAW_TRUNCATE_CHARS: int = 500
_EXPECTED_TOPIC: str = "activity"
_EXPECTED_TYPE: str = "trades"


def _truncate(raw: str) -> str:
    if len(raw) <= _RAW_TRUNCATE_CHARS:
        return raw
    return raw[:_RAW_TRUNCATE_CHARS]


def _first_str(d: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = d.get(name)
        if isinstance(value, str):
            return value
    return None


def _first_number(d: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = d.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


class LeaderWatcher:
    def __init__(
        self,
        config: ExecutionConfig,
        journal: JsonlWriter,
        fill_queue: "queue.Queue[LeaderFillObserved]",
    ) -> None:
        self._config: ExecutionConfig = config
        self._journal: JsonlWriter = journal
        self._fill_queue: queue.Queue[LeaderFillObserved] = fill_queue
        self._client: RtdsClient = RtdsClient(
            ws_url=config.ws_url,
            on_message=self._handle_message,
            on_reconnect=self._handle_reconnect,
            on_connect_failed=self._handle_connect_failed,
        )

    def start(self) -> None:
        self._journal.write(
            WatcherStarted(
                ts_utc=datetime.now(timezone.utc), ws_url=self._config.ws_url
            )
        )
        self._client.start()

    def stop(self) -> None:
        self._client.stop()
        self._journal.write(
            WatcherStopped(ts_utc=datetime.now(timezone.utc))
        )

    def _handle_reconnect(self, gap_seconds: float) -> None:
        self._journal.write(
            WatcherReconnected(
                ts_utc=datetime.now(timezone.utc), gap_seconds=gap_seconds
            )
        )

    def _handle_connect_failed(
        self, ws_url: str, attempt_number: int, error: str
    ) -> None:
        self._journal.write(
            WatcherConnectFailed(
                ts_utc=datetime.now(timezone.utc),
                ws_url=ws_url,
                attempt_number=attempt_number,
                error=error,
            )
        )

    def _handle_message(self, raw: str) -> None:
        # 0. Drop empty/whitespace frames silently. RTDS sends a blank
        # text frame as part of normal traffic (observed once per
        # connect, immediately after subscribe); journaling these as
        # malformed pollutes the operator's signal-vs-noise.
        if not raw.strip():
            return

        # 1. Parse JSON
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self._journal.write(
                WatcherMalformedMessage(
                    ts_utc=datetime.now(timezone.utc),
                    raw=_truncate(raw),
                    reason="json_parse",
                )
            )
            return

        if not isinstance(msg, dict):
            self._journal.write(
                WatcherMalformedMessage(
                    ts_utc=datetime.now(timezone.utc),
                    raw=_truncate(raw),
                    reason="missing_field",
                )
            )
            return

        # 2. Top-level fields (RTDS sends separate ``topic`` and ``type``).
        if "topic" not in msg or "type" not in msg or "payload" not in msg:
            self._journal.write(
                WatcherMalformedMessage(
                    ts_utc=datetime.now(timezone.utc),
                    raw=_truncate(raw),
                    reason="missing_field",
                )
            )
            return
        topic = msg.get("topic")
        msg_type = msg.get("type")
        if not isinstance(topic, str) or not isinstance(msg_type, str):
            self._journal.write(
                WatcherMalformedMessage(
                    ts_utc=datetime.now(timezone.utc),
                    raw=_truncate(raw),
                    reason="missing_field",
                )
            )
            return

        # 3. Topic + type must both match (joint check).
        if topic != _EXPECTED_TOPIC or msg_type != _EXPECTED_TYPE:
            self._journal.write(
                WatcherMalformedMessage(
                    ts_utc=datetime.now(timezone.utc),
                    raw=_truncate(raw),
                    reason="wrong_topic",
                )
            )
            return

        # 3a. Normalise payload to a dict. RTDS has been seen to send
        # payload as either a JSON-encoded string or a nested dict;
        # accept both. Anything else is malformed.
        payload_raw = msg["payload"]
        if isinstance(payload_raw, str):
            try:
                inner = json.loads(payload_raw)
            except json.JSONDecodeError:
                self._journal.write(
                    WatcherMalformedMessage(
                        ts_utc=datetime.now(timezone.utc),
                        raw=_truncate(raw),
                        reason="payload_parse",
                    )
                )
                return
            if not isinstance(inner, dict):
                self._journal.write(
                    WatcherMalformedMessage(
                        ts_utc=datetime.now(timezone.utc),
                        raw=_truncate(raw),
                        reason="wrong_type",
                    )
                )
                return
        elif isinstance(payload_raw, dict):
            inner = payload_raw
        else:
            self._journal.write(
                WatcherMalformedMessage(
                    ts_utc=datetime.now(timezone.utc),
                    raw=_truncate(raw),
                    reason="wrong_type",
                )
            )
            return

        # 4. proxyWallet
        proxy_wallet_raw = inner.get("proxyWallet")
        if not isinstance(proxy_wallet_raw, str):
            self._journal.write(
                WatcherMalformedMessage(
                    ts_utc=datetime.now(timezone.utc),
                    raw=_truncate(raw),
                    reason="missing_field",
                )
            )
            return

        # 5. Address filter (silent drop on miss — high-volume path)
        proxy_wallet = proxy_wallet_raw.lower()
        if proxy_wallet != self._config.leader_address:
            return

        # 6. Required trade fields. RTDS canonical inner keys (verbatim
        # from a live sample): transactionHash, conditionId, asset, side,
        # size, price, timestamp. snake_case fallbacks accepted defensively.
        transaction_hash = _first_str(inner, "transactionHash", "transaction_hash")
        condition_id = _first_str(inner, "conditionId", "condition_id")
        asset_id = _first_str(inner, "asset", "asset_id")
        side = _first_str(inner, "side")
        size = _first_number(inner, "size")
        price = _first_number(inner, "price")
        timestamp_present = "timestamp" in inner  # required to be present
        if (
            transaction_hash is None
            or condition_id is None
            or asset_id is None
            or side is None
            or size is None
            or price is None
            or not timestamp_present
        ):
            self._journal.write(
                WatcherMalformedMessage(
                    ts_utc=datetime.now(timezone.utc),
                    raw=_truncate(raw),
                    reason="missing_field",
                )
            )
            return

        # 7. Build event, journal it.
        now = datetime.now(timezone.utc)
        event = LeaderFillObserved(
            ts_utc=now,
            transaction_hash=transaction_hash,
            condition_id=condition_id,
            asset_id=asset_id,
            side=side,
            size=size,
            price=price,
            proxy_wallet=proxy_wallet,
            observed_at_utc=now,
        )
        self._journal.write(event)

        # 8. Enqueue. block=True so a slow consumer applies backpressure.
        self._fill_queue.put(event, block=True)
