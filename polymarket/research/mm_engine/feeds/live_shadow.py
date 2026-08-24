"""Live-shadow feed adapter: read-only public WS -> identical ``MarketEvent`` stream.

Connects to the public Polymarket market channel
(``wss://ws-subscriptions-clob.polymarket.com/ws/market``) read-only — **no auth, no order
placement** — and emits the *same* :class:`MarketEvent` objects the replay adapter does.

The "same code path" is literal: raw WS frames are turned into capture envelopes by the
exact same ``scripts.dali_live_clob_capture.envelope()`` used to record data, then into
events by the exact same :func:`mm_engine.events.envelope_to_events` the replay adapter
uses. The only difference between the two feeds is the transport (a socket vs. a file) and
that live stamps fresh local receive clocks (``ts_local_iso`` / ``ts_monotonic_ns``) while
replay carries the recorded ones — the executable ``ts_exchange`` / ``payload`` are identical,
so strategy decisions are identical.

The transport is injectable (the ``connect`` arg), which lets a test replay a recorded
frame sequence through this exact adapter with no network — the basis of the Phase 0 smoke
test (build-plan item 4 + 5).
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from mm_engine.events import GapMarker, envelope_to_events
from mm_engine.interfaces import MarketEvent
from scripts.dali_live_clob_capture import TokenMeta, envelope


DEFAULT_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class StopFeed(Exception):
    """Transport signals a clean end of stream (no gap)."""


class FeedDisconnected(Exception):
    """Transport signals an unexpected disconnect (a real data gap)."""


class Transport(Protocol):
    def send(self, data: str) -> None: ...
    def recv(self) -> str | bytes: ...
    def close(self) -> None: ...


@dataclass
class FrameTransport:
    """Test/replay transport: serves a fixed list of raw WS frames, then stops cleanly.

    ``recv()`` returns the next frame and raises :class:`StopFeed` when exhausted; ``send``
    and ``close`` are no-ops. This drives the live-shadow adapter through the identical
    parsing code with zero network, so a recorded frame sequence proves same-code-path.
    """

    frames: list[str | bytes]
    _i: int = 0
    sent: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sent is None:
            self.sent = []

    def send(self, data: str) -> None:
        self.sent.append(data)

    def recv(self) -> str | bytes:
        if self._i >= len(self.frames):
            raise StopFeed
        frame = self.frames[self._i]
        self._i += 1
        return frame

    def close(self) -> None:
        return None


class _WebsocketTransport:
    """Adapter over ``websocket-client`` that hides timeouts and maps close -> disconnect."""

    def __init__(self, ws_url: str, timeout: float) -> None:
        import websocket  # lazy import: tests using FrameTransport need no network/dep

        self._mod = websocket
        self._conn = websocket.create_connection(ws_url, timeout=timeout)

    def send(self, data: str) -> None:
        self._conn.send(data)

    def recv(self) -> str | bytes:
        while True:
            try:
                return self._conn.recv()
            except self._mod.WebSocketTimeoutException:
                continue
            except self._mod.WebSocketConnectionClosedException as exc:
                raise FeedDisconnected from exc

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


def _default_connect(ws_url: str, timeout: float) -> Transport:
    return _WebsocketTransport(ws_url, timeout)


def live_shadow_feed(
    token_ids: list[str],
    *,
    ws_url: str = DEFAULT_WS_URL,
    metadata: dict[str, TokenMeta] | None = None,
    connect: Callable[[str, float], Transport] | None = None,
    custom_feature_enabled: bool = True,
    timeout: float = 5.0,
    max_events: int | None = None,
    record_to: Path | str | None = None,
) -> Iterator[MarketEvent | GapMarker]:
    """Yield ``MarketEvent``s from a read-only market-channel connection.

    ``connect`` defaults to a real websocket; inject a factory returning a
    :class:`FrameTransport` to replay recorded frames offline. ``max_events`` bounds the
    stream (handy for a short live smoke). On a real disconnect a
    :class:`GapMarker` is emitted and the feed stops (no reconnect in Phase 0).

    ``record_to``: if given, every capture envelope this session produces is appended to that
    JSONL file (the exact ``envelope()`` records, with the session's own receive clocks). The
    file is therefore a replayable shard — feeding it back through ``replay_feed`` reproduces
    this session's events byte-for-byte, which is what makes the record→replay reconciliation
    a real same-code-path / divergence test (not two replays of pre-captured data).
    """
    if metadata is None:
        metadata = {t: TokenMeta(token_id=t) for t in token_ids}

    factory = connect or _default_connect
    transport = factory(ws_url, timeout)
    subscription = {
        "assets_ids": list(token_ids),
        "type": "market",
        "custom_feature_enabled": bool(custom_feature_enabled),
    }
    rec_fh = None
    if record_to is not None:
        rec_path = Path(record_to)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec_fh = rec_path.open("a", encoding="utf-8")
    emitted = 0
    try:
        transport.send(json.dumps(subscription))
        while True:
            try:
                raw = transport.recv()
            except StopFeed:
                break
            except FeedDisconnected:
                yield GapMarker(reason="disconnect")
                break
            for rec in envelope(raw, metadata):
                if rec_fh is not None:
                    rec_fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
                for ev in envelope_to_events(rec):
                    yield ev
                    emitted += 1
                    if max_events is not None and emitted >= max_events:
                        return
    finally:
        if rec_fh is not None:
            rec_fh.close()
        transport.close()
