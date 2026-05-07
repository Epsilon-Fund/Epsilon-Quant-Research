from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from executor.venue import (
    NormalizedOrderEvent,
    Side,
    SubmitOrderStatus,
    TimeInForce,
    VenueAdapter,
    VenueCancelEvent,
    VenueFillEvent,
    VenueOrderAck,
    VenueOrderIntent,
    VenueRejectEvent,
)

from .market_data import UserOrderEvent


class OrderStatus(str, Enum):
    IDLE = "IDLE"            # no open order for this event
    OPEN = "OPEN"            # order submitted and acknowledged
    AMBIGUOUS = "AMBIGUOUS"  # submit/cancel timed out — outcome unknown


@dataclass(slots=True)
class EventOrderState:
    """Live state of our position in one event."""

    event_slug: str
    status: OrderStatus = OrderStatus.IDLE
    client_order_id: str | None = None
    token_id: str | None = None
    price_ticks: int | None = None
    qty: int = 0
    filled_qty: float = 0.0    # shares; accumulates from both WS (float) and poll (int)
    filled_usdc: float = 0.0   # USDC cost basis; used by risk gate for notional cap


class OrderManagementSystem:
    """Enforces one-open-order-per-event and tracks cumulative fills per event.

    Synchronous — intended to be called from the async main loop via
    asyncio.to_thread so it never blocks the event loop directly.

    Fill tracking aggregates across all tokens within an event: if we fill
    3 units on YES-18C then 4 units on NO-13C, filled_qty for the event is 7.
    This is intentional — see design notes in oms.py.
    """

    __slots__ = ("_adapter", "_package_id", "_order_qty", "_states", "_coid_to_slug")

    def __init__(
        self,
        adapter: VenueAdapter,
        *,
        package_id: str,
        order_qty: int,
    ) -> None:
        if order_qty <= 0:
            raise ValueError("order_qty must be > 0")
        self._adapter = adapter
        self._package_id = package_id
        self._order_qty = order_qty
        self._states: dict[str, EventOrderState] = {}
        # Maps client_order_id → event_slug so fills from any token
        # on the event accumulate to the same counter.
        self._coid_to_slug: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Primary operations — called by execution.py
    # ------------------------------------------------------------------

    def place_or_replace(
        self,
        *,
        event_slug: str,
        token_id: str,
        condition_id: str,
        price_ticks: int,
        now_ns: int,
        tick_size: float = 0.01,
    ) -> None:
        """Submit a passive GTC BUY on token_id at price_ticks.

        - Same token + same price as current open order → no-op.
        - Different token or price → cancel existing, then submit new.
        - Ambiguous state → skip until reconciliation resolves it.
        """
        state = self._get_or_create(event_slug)

        if state.status == OrderStatus.AMBIGUOUS:
            return

        if state.status == OrderStatus.OPEN:
            if state.token_id == token_id and state.price_ticks == price_ticks:
                return
            self._cancel_open(state, now_ns)
            if state.status == OrderStatus.AMBIGUOUS:
                return

        self._submit(
            state,
            token_id=token_id,
            condition_id=condition_id,
            price_ticks=price_ticks,
            now_ns=now_ns,
            tick_size=tick_size,
        )

    def cancel(self, *, event_slug: str, now_ns: int) -> None:
        """Cancel the open order for this event, if any."""
        state = self._states.get(event_slug)
        if state is None or state.status != OrderStatus.OPEN:
            return
        self._cancel_open(state, now_ns)

    def cancel_all(self, *, now_ns: int) -> None:
        """Cancel all open orders. Call on shutdown."""
        for slug in list(self._states):
            self.cancel(event_slug=slug, now_ns=now_ns)

    # ------------------------------------------------------------------
    # Fill / cancel notification paths
    # ------------------------------------------------------------------

    def on_user_order_event(self, event: UserOrderEvent) -> None:
        """Process a fill or status change arriving from the user WebSocket."""
        state = self._state_for_coid(event.client_order_id)
        if state is None:
            return

        if event.status in {"MATCHED", "TRADE"}:
            if event.fill_qty is not None and event.fill_qty > 0:
                state.filled_qty += event.fill_qty
                if event.fill_price is not None:
                    state.filled_usdc += event.fill_qty * event.fill_price

        elif event.status in {"CANCELED", "CANCELLED", "REJECTED"}:
            if state.client_order_id == event.client_order_id:
                self._clear_open(state)

    def on_venue_events(self, events: tuple[NormalizedOrderEvent, ...]) -> None:
        """Process events arriving from the executor's HTTP poll."""
        for event in events:
            state = self._state_for_coid(event.client_order_id)
            if state is None:
                continue

            if isinstance(event, VenueFillEvent):
                state.filled_qty += float(event.fill_qty)
                state.filled_usdc += float(event.fill_qty) * event.fill_price
                if state.status == OrderStatus.AMBIGUOUS:
                    state.status = OrderStatus.OPEN

            elif isinstance(event, (VenueCancelEvent, VenueRejectEvent)):
                if state.client_order_id == event.client_order_id:
                    self._clear_open(state)

            elif isinstance(event, VenueOrderAck):
                if (
                    state.client_order_id == event.client_order_id
                    and state.status == OrderStatus.AMBIGUOUS
                ):
                    state.status = OrderStatus.OPEN

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def order_state(self, event_slug: str) -> EventOrderState | None:
        return self._states.get(event_slug)

    def all_open_client_order_ids(self) -> set[str]:
        return {
            s.client_order_id
            for s in self._states.values()
            if s.status == OrderStatus.OPEN and s.client_order_id is not None
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, event_slug: str) -> EventOrderState:
        if event_slug not in self._states:
            self._states[event_slug] = EventOrderState(event_slug=event_slug)
        return self._states[event_slug]

    def _state_for_coid(self, client_order_id: str) -> EventOrderState | None:
        slug = self._coid_to_slug.get(client_order_id)
        return self._states.get(slug) if slug else None

    def _submit(
        self,
        state: EventOrderState,
        *,
        token_id: str,
        condition_id: str,
        price_ticks: int,
        now_ns: int,
        tick_size: float = 0.01,
    ) -> None:
        intent = VenueOrderIntent(
            package_id=self._package_id,
            leg_id=token_id,
            market_id=condition_id,
            token_id=token_id,
            side=Side.BUY,
            quantity=self._order_qty,
            limit_price_ticks=price_ticks,
            tif=TimeInForce.GTC,
            ts_ns=now_ns,
            tick_size=tick_size,
        )

        try:
            result = self._adapter.submit_order(intent)
        except Exception:
            state.status = OrderStatus.AMBIGUOUS
            return

        if result.client_order_id:
            self._coid_to_slug[result.client_order_id] = state.event_slug
        # Register venue_order_id too — WS fill messages carry the venue hash, not our client ID,
        # because we don't embed client_order_id in the EIP-712 struct that Polymarket signs.
        if result.venue_order_id:
            self._coid_to_slug[result.venue_order_id] = state.event_slug

        if result.status == SubmitOrderStatus.ACKNOWLEDGED:
            state.status = OrderStatus.OPEN
            state.client_order_id = result.client_order_id
            state.token_id = token_id
            state.price_ticks = price_ticks
            state.qty = self._order_qty

        elif result.status == SubmitOrderStatus.AMBIGUOUS:
            state.status = OrderStatus.AMBIGUOUS
            state.client_order_id = result.client_order_id

        elif result.status == SubmitOrderStatus.ALREADY_SUBMITTED:
            # Idempotent hit — order is already live
            state.status = OrderStatus.OPEN
            state.client_order_id = result.client_order_id
            state.token_id = token_id
            state.price_ticks = price_ticks
            state.qty = self._order_qty

        # REJECTED: leave as IDLE — strategy will retry on next tick

    def _cancel_open(self, state: EventOrderState, now_ns: int) -> None:
        if state.client_order_id is None:
            self._clear_open(state)
            return

        try:
            self._adapter.cancel_order(
                client_order_id=state.client_order_id,
                reason="oms_replace",
                now_ns=now_ns,
            )
        except Exception:
            state.status = OrderStatus.AMBIGUOUS
            return

        self._clear_open(state)

    def _clear_open(self, state: EventOrderState) -> None:
        state.status = OrderStatus.IDLE
        state.client_order_id = None
        state.token_id = None
        state.price_ticks = None
