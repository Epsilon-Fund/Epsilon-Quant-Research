from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import websockets
import websockets.exceptions


# ---------------------------------------------------------------------------
# Public event types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BookUpdateEvent:
    """Best bid/ask changed for a token."""

    token_id: str
    best_bid: float  # 0.0 when no bids exist
    best_ask: float  # 1.0 when no asks exist
    ts_ns: int


@dataclass(frozen=True, slots=True)
class UserOrderEvent:
    """Our own order changed state (fill, cancel, reject)."""

    client_order_id: str
    status: str  # "MATCHED", "CANCELED", "REJECTED", etc.
    fill_qty: float | None
    fill_price: float | None
    ts_ns: int


MarketDataEvent = BookUpdateEvent | UserOrderEvent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarketDataConfig:
    market_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    user_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    ping_interval_s: float = 10.0
    reconnect_delay_s: float = 2.0
    queue_max_size: int = 1_000
    subscription_chunk_size: int = 200  # max tokens per subscription message


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MarketDataClient:
    """Maintains live WebSocket connections to Polymarket and emits normalized events.

    Usage::

        async with MarketDataClient(token_ids=[...], condition_ids=[...], api_key=...) as client:
            async for event in client.events():
                ...

    The market WS sends a full book snapshot on subscribe, so best_bid is
    available within the first message — no separate REST bootstrap needed.
    """

    __slots__ = (
        "_token_ids",
        "_condition_ids",
        "_config",
        "_api_key",
        "_api_secret",
        "_passphrase",
        "_queue",
        "_bid_levels",
        "_ask_levels",
        "_market_task",
        "_user_task",
        "_running",
        "_logger",
    )

    def __init__(
        self,
        *,
        token_ids: list[str],
        condition_ids: list[str],
        config: MarketDataConfig | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        passphrase: str | None = None,
        logger: Any | None = None,
    ) -> None:
        self._token_ids = list(token_ids)
        self._condition_ids = list(condition_ids)
        self._config = config or MarketDataConfig()
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self._logger = logger
        self._queue: asyncio.Queue[MarketDataEvent] = asyncio.Queue(
            maxsize=self._config.queue_max_size
        )
        # token_id -> {price_str -> size}
        self._bid_levels: dict[str, dict[str, float]] = {}
        self._ask_levels: dict[str, dict[str, float]] = {}
        self._market_task: asyncio.Task[None] | None = None
        self._user_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        if self._token_ids:
            self._market_task = asyncio.create_task(self._market_ws_loop())
        if self._condition_ids and self._api_key:
            self._user_task = asyncio.create_task(self._user_ws_loop())

    async def stop(self) -> None:
        self._running = False
        tasks = [t for t in (self._market_task, self._user_task) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._market_task = None
        self._user_task = None

    async def add_token_ids(self, new_ids: list[str]) -> None:
        """Register new token IDs and reconnect the market WS to subscribe to them.

        Safe to call from the event-loop thread while the client is running.
        The WS sends a full book snapshot on reconnect, so new tokens are priced
        within seconds of this call.
        """
        existing = set(self._token_ids)
        added = [tid for tid in new_ids if tid not in existing]
        if not added:
            return
        self._token_ids.extend(added)
        if self._running:
            if self._market_task is not None and not self._market_task.done():
                self._market_task.cancel()
                await asyncio.gather(self._market_task, return_exceptions=True)
            self._market_task = asyncio.create_task(self._market_ws_loop())

    async def add_condition_ids(self, new_ids: list[str]) -> None:
        """Register new condition IDs and reconnect the user WS to receive fills."""
        existing = set(self._condition_ids)
        added = [cid for cid in new_ids if cid not in existing]
        if not added:
            return
        self._condition_ids.extend(added)
        if self._running and self._api_key:
            if self._user_task is not None and not self._user_task.done():
                self._user_task.cancel()
                await asyncio.gather(self._user_task, return_exceptions=True)
            self._user_task = asyncio.create_task(self._user_ws_loop())

    async def __aenter__(self) -> MarketDataClient:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    async def events(self) -> AsyncIterator[MarketDataEvent]:
        """Yields events as they arrive. Returns when stop() is called."""
        while self._running:
            try:
                yield await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

    # ------------------------------------------------------------------
    # WebSocket loops — each reconnects automatically on disconnect
    # ------------------------------------------------------------------

    def _log(self, level: str, event: str, **kw: Any) -> None:
        if self._logger is None:
            return
        getattr(self._logger, level, self._logger.info)(event, **kw)

    async def _market_ws_loop(self) -> None:
        while self._running:
            token_ids = list(self._token_ids)
            self._log("info", "market_ws_connecting", token_count=len(token_ids))
            try:
                async with websockets.connect(self._config.market_ws_url) as ws:
                    await ws.send(json.dumps({
                        "assets_ids": token_ids,
                        "type": "Market",
                        "custom_feature_enabled": True,
                    }))
                    self._log(
                        "info", "market_ws_subscribed",
                        token_count=len(token_ids),
                        chunks=1,
                    )
                    ping = asyncio.create_task(self._ping_loop(ws))
                    try:
                        async for raw in ws:
                            self._handle_market_message(raw)
                    finally:
                        ping.cancel()
                        await asyncio.gather(ping, return_exceptions=True)
                self._log("warning", "market_ws_closed", token_count=len(token_ids))
            except (websockets.exceptions.WebSocketException, OSError) as exc:
                self._log("warning", "market_ws_error", error=str(exc), token_count=len(token_ids))
            except asyncio.CancelledError:
                return
            if self._running:
                await asyncio.sleep(self._config.reconnect_delay_s)

    async def _user_ws_loop(self) -> None:
        while self._running:
            try:
                async with websockets.connect(self._config.user_ws_url) as ws:
                    await ws.send(json.dumps({
                        "markets": self._condition_ids,
                        "type": "User",
                        "auth": {
                            "apiKey": self._api_key,
                            "secret": self._api_secret,
                            "passphrase": self._passphrase,
                        },
                    }))
                    ping = asyncio.create_task(self._ping_loop(ws))
                    try:
                        async for raw in ws:
                            self._handle_user_message(raw)
                    finally:
                        ping.cancel()
                        await asyncio.gather(ping, return_exceptions=True)
            except (websockets.exceptions.WebSocketException, OSError):
                pass
            except asyncio.CancelledError:
                return
            if self._running:
                await asyncio.sleep(self._config.reconnect_delay_s)

    async def _ping_loop(self, ws: Any) -> None:
        try:
            while True:
                await asyncio.sleep(self._config.ping_interval_s)
                await ws.send(json.dumps({"type": "PING"}))
        except (websockets.exceptions.WebSocketException, asyncio.CancelledError):
            pass

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------

    def _handle_market_message(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return
        if isinstance(msg, list):
            for item in msg:
                if isinstance(item, dict):
                    self._process_market_event(item)
        elif isinstance(msg, dict):
            self._process_market_event(msg)

    def _process_market_event(self, msg: dict[str, Any]) -> None:
        ts_ns = _ts_ns(msg)

        # Real-time price change — primary live format.
        # Each change carries best_bid/best_ask for the affected token.
        if "price_changes" in msg:
            for change in (msg.get("price_changes") or []):
                if not isinstance(change, dict):
                    continue
                token_id = str(change.get("asset_id", ""))
                if not token_id:
                    continue
                best_bid = _to_float(change.get("best_bid"))
                best_ask = _to_float(change.get("best_ask"))
                if best_bid is not None or best_ask is not None:
                    self._put(BookUpdateEvent(
                        token_id=token_id,
                        best_bid=best_bid if best_bid is not None else 0.0,
                        best_ask=best_ask if best_ask is not None else 1.0,
                        ts_ns=ts_ns,
                    ))
                else:
                    # No best_bid/ask in message — update levels and compute
                    price = str(change.get("price", ""))
                    size = _to_float(change.get("size"))
                    side = str(change.get("side", "")).upper()
                    if not price:
                        continue
                    bids = self._bid_levels.setdefault(token_id, {})
                    asks = self._ask_levels.setdefault(token_id, {})
                    levels = bids if side == "BUY" else asks
                    if size and size > 0:
                        levels[price] = size
                    else:
                        levels.pop(price, None)
                    self._emit_book(token_id, ts_ns)
            return

        # Full book snapshot sent on subscription — fields: asset_id, bids, asks.
        token_id = str(msg.get("asset_id", ""))
        if token_id and ("bids" in msg or "asks" in msg):
            self._bid_levels[token_id] = _parse_levels(msg.get("bids") or [])
            self._ask_levels[token_id] = _parse_levels(msg.get("asks") or [])
            self._emit_book(token_id, ts_ns)
            return

        # Legacy format kept for backwards compatibility (event_type based).
        event_type = str(msg.get("event_type", ""))
        if not token_id or not event_type:
            return

        if event_type == "book":
            self._bid_levels[token_id] = _parse_levels(
                msg.get("buys") or msg.get("bids") or []
            )
            self._ask_levels[token_id] = _parse_levels(
                msg.get("sells") or msg.get("asks") or []
            )
            self._emit_book(token_id, ts_ns)
        elif event_type == "price_change":
            bids = self._bid_levels.setdefault(token_id, {})
            asks = self._ask_levels.setdefault(token_id, {})
            for change in (msg.get("changes") or []):
                if not isinstance(change, dict):
                    continue
                price = str(change.get("price", ""))
                size = _to_float(change.get("size"))
                side = str(change.get("side", "")).upper()
                if not price:
                    continue
                levels = bids if side == "BUY" else asks
                if size and size > 0:
                    levels[price] = size
                else:
                    levels.pop(price, None)
            self._emit_book(token_id, ts_ns)

    def _emit_book(self, token_id: str, ts_ns: int) -> None:
        bids = self._bid_levels.get(token_id, {})
        asks = self._ask_levels.get(token_id, {})
        best_bid = max((float(p) for p, s in bids.items() if s > 0), default=0.0)
        best_ask = min((float(p) for p, s in asks.items() if s > 0), default=1.0)
        self._put(BookUpdateEvent(
            token_id=token_id, best_bid=best_bid, best_ask=best_ask, ts_ns=ts_ns,
        ))

    def _handle_user_message(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return
        if isinstance(msg, list):
            for item in msg:
                if isinstance(item, dict):
                    self._process_user_event(item)
        elif isinstance(msg, dict):
            self._process_user_event(msg)

    def _process_user_event(self, msg: dict[str, Any]) -> None:
        client_order_id = str(
            msg.get("clientOrderId") or msg.get("client_order_id") or ""
        )
        # For trade/fill messages Polymarket uses maker_order_id (our passive GTC bid) and
        # taker_order_id (the counterparty). Try maker first — we are always the maker.
        venue_order_id = str(
            msg.get("maker_order_id") or msg.get("taker_order_id")
            or msg.get("order_id") or msg.get("id") or ""
        )
        if not client_order_id and not venue_order_id:
            return
        status = str(msg.get("status") or msg.get("type") or msg.get("event_type") or "UNKNOWN").upper()
        ts_ns = _ts_ns(msg)
        fill_qty: float | None = None
        fill_price: float | None = None
        if status in {"MATCHED", "TRADE"}:
            fill_qty = _to_float(msg.get("size") or msg.get("matched_amount"))
            fill_price = _to_float(msg.get("price"))

        # When a taker hits multiple resting orders, Polymarket delivers a single TRADE event
        # with a "maker_orders" array listing each maker's portion. Emit individual fill events
        # per maker so the OMS can attribute the correct amount to our specific order.
        maker_orders = msg.get("maker_orders")
        if isinstance(maker_orders, list) and maker_orders and status in {"MATCHED", "TRADE"}:
            for mo in maker_orders:
                mo_id = str(mo.get("order_id") or "")
                mo_amount = _to_float(mo.get("matched_amount"))
                mo_price = _to_float(mo.get("price")) or fill_price
                if mo_id and mo_amount:
                    self._put(UserOrderEvent(
                        client_order_id=mo_id,
                        status="MATCHED",
                        fill_qty=mo_amount,
                        fill_price=mo_price or 0.0,
                        ts_ns=ts_ns,
                    ))
            return

        self._put(UserOrderEvent(
            client_order_id=client_order_id or venue_order_id,
            status=status,
            fill_qty=fill_qty,
            fill_price=fill_price,
            ts_ns=ts_ns,
        ))

    def _put(self, event: MarketDataEvent) -> None:
        """Drop oldest event if queue is full to avoid blocking the WS loop."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_levels(raw: list[Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in raw:
        if isinstance(item, dict) and "price" in item:
            size = _to_float(item.get("size"))
            if size and size > 0:
                result[str(item["price"])] = size
    return result


def _ts_ns(msg: dict[str, Any]) -> int:
    for key in ("ts_ns", "timestamp", "ts", "ts_ms"):
        val = msg.get(key)
        if val is not None:
            try:
                v = float(str(val))
                if v > 1e16:
                    return int(v)              # nanoseconds  (~1.7e18)
                if v > 1e13:
                    return int(v) * 1_000      # microseconds (~1.7e15) → ns
                if v > 1e10:
                    return int(v) * 1_000_000  # milliseconds (~1.7e12) → ns
                return int(v * 1e9)            # seconds      (~1.7e9)  → ns
            except (ValueError, TypeError):
                pass
    return time.time_ns()


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return None
