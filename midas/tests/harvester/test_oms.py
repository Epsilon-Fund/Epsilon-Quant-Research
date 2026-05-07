from __future__ import annotations

import pytest

from executor.venue import (
    CancelOrderResult,
    CancelOrderStatus,
    ReconciliationResult,
    SubmitOrderResult,
    SubmitOrderStatus,
    VenueCancelEvent,
    VenueFillEvent,
    VenueOrderAck,
)
from harvester.market_data import UserOrderEvent
from harvester.oms import EventOrderState, OrderManagementSystem, OrderStatus

pytestmark = [pytest.mark.unit, pytest.mark.harvester]

SLUG = "london-temp-april-30"
SLUG2 = "london-temp-may-1"
TOKEN_A = "token-yes-18c"
TOKEN_B = "token-no-13c"
COND = "0xabcd"
PRICE = 91
NOW = 1_700_000_000_000_000_000


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------


class _Adapter:
    def __init__(self) -> None:
        self.submit_calls = 0
        self.cancel_calls = 0
        self._order_ctr = 0
        self._raise_on_submit: Exception | None = None
        self._ambiguous_on_submit: bool = False

    def raise_on_next_submit(self, exc: Exception) -> None:
        self._raise_on_submit = exc

    def return_ambiguous_on_next_submit(self) -> None:
        self._ambiguous_on_submit = True

    def submit_order(self, intent):
        self.submit_calls += 1
        self._order_ctr += 1
        if self._raise_on_submit:
            exc, self._raise_on_submit = self._raise_on_submit, None
            raise exc
        if self._ambiguous_on_submit:
            self._ambiguous_on_submit = False
            return SubmitOrderResult(
                status=SubmitOrderStatus.AMBIGUOUS,
                client_order_id=f"coid-{self._order_ctr}",
                venue_order_id=None,
                events=tuple(),
                ambiguous=True,
            )
        return SubmitOrderResult(
            status=SubmitOrderStatus.ACKNOWLEDGED,
            client_order_id=f"coid-{self._order_ctr}",
            venue_order_id=f"void-{self._order_ctr}",
            events=tuple(),
            ambiguous=False,
        )

    def cancel_order(self, *, client_order_id=None, venue_order_id=None, reason="", now_ns=None):
        self.cancel_calls += 1
        return CancelOrderResult(
            status=CancelOrderStatus.CANCELED,
            client_order_id=client_order_id,
            venue_order_id=None,
            events=tuple(),
            ambiguous=False,
        )

    def poll_or_process_order_updates(self, raw_updates=None):
        return tuple()

    def reconcile_open_orders(self, expected):
        return ReconciliationResult(
            venue_open_client_order_ids=tuple(),
            unknown_open_client_order_ids=tuple(),
            missing_expected_client_order_ids=tuple(),
            generated_events=tuple(),
        )


def _oms(adapter: _Adapter | None = None) -> tuple[OrderManagementSystem, _Adapter]:
    if adapter is None:
        adapter = _Adapter()
    return OrderManagementSystem(adapter, package_id="test-pkg", order_qty=10), adapter


def _place(oms: OrderManagementSystem, *, slug=SLUG, token=TOKEN_A, price=PRICE) -> None:
    oms.place_or_replace(event_slug=slug, token_id=token, condition_id=COND, price_ticks=price, now_ns=NOW)


def _fill_ws(oms: OrderManagementSystem, coid: str, qty: float, price: float = 0.91) -> None:
    oms.on_user_order_event(UserOrderEvent(
        client_order_id=coid, status="MATCHED", fill_qty=qty, fill_price=price, ts_ns=NOW,
    ))


def _cancel_ws(oms: OrderManagementSystem, coid: str) -> None:
    oms.on_user_order_event(UserOrderEvent(
        client_order_id=coid, status="CANCELED", fill_qty=None, fill_price=None, ts_ns=NOW,
    ))


# ---------------------------------------------------------------------------
# Basic placement
# ---------------------------------------------------------------------------


def test_first_submit_sets_open_status() -> None:
    oms, adapter = _oms()
    _place(oms)
    state = oms.order_state(SLUG)
    assert state.status == OrderStatus.OPEN
    assert adapter.submit_calls == 1


def test_same_token_and_price_is_noop() -> None:
    oms, adapter = _oms()
    _place(oms)
    _place(oms)
    assert adapter.submit_calls == 1


def test_price_change_cancels_and_resubmits() -> None:
    oms, adapter = _oms()
    _place(oms, price=91)
    _place(oms, price=94)
    assert adapter.cancel_calls == 1
    assert adapter.submit_calls == 2
    assert oms.order_state(SLUG).price_ticks == 94


def test_token_switch_cancels_and_resubmits() -> None:
    oms, adapter = _oms()
    _place(oms, token=TOKEN_A)
    _place(oms, token=TOKEN_B)
    assert adapter.cancel_calls == 1
    assert adapter.submit_calls == 2
    assert oms.order_state(SLUG).token_id == TOKEN_B


# ---------------------------------------------------------------------------
# Fill tracking — WS path
# ---------------------------------------------------------------------------


def test_ws_fill_accumulates_filled_qty() -> None:
    oms, _ = _oms()
    _place(oms)
    coid = oms.order_state(SLUG).client_order_id
    _fill_ws(oms, coid, qty=5.0, price=0.91)
    assert oms.order_state(SLUG).filled_qty == 5.0


def test_ws_fill_accumulates_filled_usdc() -> None:
    oms, _ = _oms()
    _place(oms)
    coid = oms.order_state(SLUG).client_order_id
    _fill_ws(oms, coid, qty=5.0, price=0.91)
    assert abs(oms.order_state(SLUG).filled_usdc - 5.0 * 0.91) < 1e-9


def test_ws_fills_from_two_orders_aggregate() -> None:
    oms, _ = _oms()
    _place(oms, token=TOKEN_A, price=91)
    coid1 = oms.order_state(SLUG).client_order_id
    _fill_ws(oms, coid1, qty=3.0, price=0.91)
    _cancel_ws(oms, coid1)
    _place(oms, token=TOKEN_B, price=93)
    coid2 = oms.order_state(SLUG).client_order_id
    _fill_ws(oms, coid2, qty=4.0, price=0.93)
    assert oms.order_state(SLUG).filled_qty == 7.0


def test_ws_fill_for_unknown_coid_is_ignored() -> None:
    oms, _ = _oms()
    _fill_ws(oms, "unknown-coid", qty=5.0)  # should not raise


def test_ws_cancel_clears_open_order() -> None:
    oms, _ = _oms()
    _place(oms)
    coid = oms.order_state(SLUG).client_order_id
    _cancel_ws(oms, coid)
    state = oms.order_state(SLUG)
    assert state.status == OrderStatus.IDLE
    assert state.client_order_id is None


# ---------------------------------------------------------------------------
# Fill tracking — venue poll path
# ---------------------------------------------------------------------------


def test_venue_fill_event_accumulates_qty_and_usdc() -> None:
    oms, _ = _oms()
    _place(oms)
    coid = oms.order_state(SLUG).client_order_id
    fill = VenueFillEvent(
        package_id="test-pkg", leg_id=TOKEN_A, client_order_id=coid,
        fill_qty=5, fill_price=0.91, ts_ns=NOW,
    )
    oms.on_venue_events((fill,))
    state = oms.order_state(SLUG)
    assert state.filled_qty == 5.0
    assert abs(state.filled_usdc - 5 * 0.91) < 1e-9


def test_venue_cancel_event_clears_open_order() -> None:
    oms, _ = _oms()
    _place(oms)
    coid = oms.order_state(SLUG).client_order_id
    cancel = VenueCancelEvent(
        package_id="test-pkg", leg_id=TOKEN_A, client_order_id=coid,
        canceled_qty=10, reason="expired", ts_ns=NOW,
    )
    oms.on_venue_events((cancel,))
    assert oms.order_state(SLUG).status == OrderStatus.IDLE


def test_venue_ack_resolves_ambiguous_to_open() -> None:
    # AMBIGUOUS via status (not exception) stores the COID, so a later ack can resolve it.
    oms, adapter = _oms()
    adapter.return_ambiguous_on_next_submit()
    _place(oms)
    assert oms.order_state(SLUG).status == OrderStatus.AMBIGUOUS
    coid = oms.order_state(SLUG).client_order_id
    assert coid is not None  # AMBIGUOUS-via-status stores the coid
    ack = VenueOrderAck(
        package_id="test-pkg", leg_id=TOKEN_A, client_order_id=coid,
        venue_order_id="void-1", ts_ns=NOW,
    )
    oms.on_venue_events((ack,))
    assert oms.order_state(SLUG).status == OrderStatus.OPEN


# ---------------------------------------------------------------------------
# AMBIGUOUS state
# ---------------------------------------------------------------------------


def test_submit_timeout_sets_ambiguous() -> None:
    oms, adapter = _oms()
    adapter.raise_on_next_submit(TimeoutError("timeout"))
    _place(oms)
    assert oms.order_state(SLUG).status == OrderStatus.AMBIGUOUS


def test_ambiguous_blocks_further_placement() -> None:
    oms, adapter = _oms()
    adapter.raise_on_next_submit(TimeoutError("timeout"))
    _place(oms)
    _place(oms, price=94)  # should be blocked
    assert adapter.submit_calls == 1


# ---------------------------------------------------------------------------
# cancel() / cancel_all()
# ---------------------------------------------------------------------------


def test_cancel_clears_open_order() -> None:
    oms, adapter = _oms()
    _place(oms)
    oms.cancel(event_slug=SLUG, now_ns=NOW)
    assert oms.order_state(SLUG).status == OrderStatus.IDLE
    assert adapter.cancel_calls == 1


def test_cancel_noop_when_no_open_order() -> None:
    oms, adapter = _oms()
    oms.cancel(event_slug=SLUG, now_ns=NOW)
    assert adapter.cancel_calls == 0


def test_cancel_all_clears_all_events() -> None:
    oms, adapter = _oms()
    _place(oms, slug=SLUG)
    _place(oms, slug=SLUG2)
    oms.cancel_all(now_ns=NOW)
    assert oms.order_state(SLUG).status == OrderStatus.IDLE
    assert oms.order_state(SLUG2).status == OrderStatus.IDLE
    assert adapter.cancel_calls == 2


# ---------------------------------------------------------------------------
# all_open_client_order_ids()
# ---------------------------------------------------------------------------


def test_all_open_ids_returns_open_coids() -> None:
    oms, _ = _oms()
    _place(oms, slug=SLUG)
    _place(oms, slug=SLUG2)
    coids = oms.all_open_client_order_ids()
    assert len(coids) == 2


def test_all_open_ids_excludes_idle() -> None:
    oms, _ = _oms()
    _place(oms, slug=SLUG)
    _place(oms, slug=SLUG2)
    oms.cancel(event_slug=SLUG, now_ns=NOW)
    coids = oms.all_open_client_order_ids()
    assert len(coids) == 1
