"""CLOB WebSocket connection with reconnect/heartbeat.

Talks to Polymarket's RTDS (Real-Time Data Service) at
``wss://ws-live-data.polymarket.com``. Network/protocol layer only —
no parsing, no filtering, no journaling. Pongs are recognised at the
protocol level and not forwarded to the on_message callback. Every
other text frame is forwarded raw.

Threading model: a single background thread runs the connect-and-read
loop. A second thread sends "ping" every 5 s for the duration of each
connection. websockets.sync.client connections are safe for one
sender + one receiver concurrently, which is exactly what we use.

Reconnect: exponential backoff 1, 2, 4, 8, 16, capped at 30 seconds.
Backoff index resets on a successful connection. ``on_reconnect`` only
fires after a successful connect that follows at least one prior
disconnect.

Connect-failure observability: ``on_connect_failed`` fires once per
*failed* connect attempt, with attempt_number 1-indexed within the
current outage streak. The counter resets to 1 after each successful
connect. Mid-stream read errors do NOT trigger on_connect_failed —
they are disconnects, and the next connect attempt that fails will
fire on_connect_failed with attempt_number=1.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from websockets.sync.client import connect

_SUBSCRIBE_ENVELOPE: str = (
    '{"action":"subscribe","subscriptions":'
    '[{"topic":"activity","type":"trades","filters":""}]}'
)
_PING_INTERVAL_S: float = 5.0
_BACKOFF_SCHEDULE_S: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
_STOP_TIMEOUT_S: float = 5.0

_log = logging.getLogger(__name__)


class RtdsClient:
    def __init__(
        self,
        ws_url: str,
        on_message: Callable[[str], None],
        on_reconnect: Callable[[float], None],
        on_connect_failed: Callable[[str, int, str], None],
    ) -> None:
        self._ws_url: str = ws_url
        self._on_message: Callable[[str], None] = on_message
        self._on_reconnect: Callable[[float], None] = on_reconnect
        self._on_connect_failed: Callable[[str, int, str], None] = on_connect_failed
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="rtds-client", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_TIMEOUT_S)
            self._thread = None

    def _run(self) -> None:
        backoff_idx = 0
        attempt_number = 1
        had_disconnect = False
        last_disconnect_ts: float | None = None

        while not self._stop_event.is_set():
            # 1. Try to connect.
            try:
                ws = connect(self._ws_url, open_timeout=10)
            except Exception as exc:
                try:
                    self._on_connect_failed(self._ws_url, attempt_number, repr(exc))
                except Exception:
                    _log.exception("on_connect_failed callback raised")
                last_disconnect_ts = time.monotonic()
                had_disconnect = True
                if self._stop_event.is_set():
                    break
                delay = _BACKOFF_SCHEDULE_S[
                    min(backoff_idx, len(_BACKOFF_SCHEDULE_S) - 1)
                ]
                backoff_idx += 1
                attempt_number += 1
                self._stop_event.wait(timeout=delay)
                continue

            # 2. Connect succeeded.
            try:
                if had_disconnect and last_disconnect_ts is not None:
                    gap = time.monotonic() - last_disconnect_ts
                    try:
                        self._on_reconnect(gap)
                    except Exception:
                        _log.exception("on_reconnect callback raised")
                backoff_idx = 0
                attempt_number = 1
                ws.send(_SUBSCRIBE_ENVELOPE)
                pinger_stop = threading.Event()
                pinger = threading.Thread(
                    target=self._pinger, args=(ws, pinger_stop),
                    name="rtds-pinger", daemon=True,
                )
                pinger.start()
                try:
                    for raw in ws:
                        if self._stop_event.is_set():
                            break
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8", errors="replace")
                        if raw.strip().lower() == "pong":
                            continue  # heartbeat, not data
                        try:
                            self._on_message(raw)
                        except Exception:
                            _log.exception("on_message callback raised")
                finally:
                    pinger_stop.set()
                    pinger.join(timeout=1.0)
            except Exception as exc:
                _log.warning("RTDS read loop error: %r", exc)
            finally:
                try:
                    ws.close()
                except Exception:
                    pass

            # 3. Connection ended (clean or mid-stream error). Loop and reconnect.
            had_disconnect = True
            last_disconnect_ts = time.monotonic()
            if self._stop_event.is_set():
                break
            delay = _BACKOFF_SCHEDULE_S[
                min(backoff_idx, len(_BACKOFF_SCHEDULE_S) - 1)
            ]
            backoff_idx += 1
            self._stop_event.wait(timeout=delay)

    def _pinger(self, ws: object, stop: threading.Event) -> None:
        while not stop.is_set():
            if stop.wait(timeout=_PING_INTERVAL_S):
                return
            try:
                ws.send("ping")  # type: ignore[attr-defined]
            except Exception:
                return
