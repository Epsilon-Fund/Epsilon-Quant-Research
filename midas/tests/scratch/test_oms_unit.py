"""
Scratch unit tests for harvester/oms.py — delete when done.
"""
from __future__ import annotations

import pytest

from executor.venue import (
    CancelOrderResult,
    CancelOrderStatus,
    ReconciliationResult,
    SubmitOrderResult,
    SubmitOrderStatus,
)
from harvester.market_data import UserOrderEvent
from harvester.oms import OrderManagementSystem, OrderStatus

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SLUG = "london-temp-april-30"
SLUG2 = "london-temp-may-1"
TOKEN_A = "token-yes-18c"
TOKEN_B = "token-no-13c"
COND = "0xabcd"
PRICE = 91
NOW = 1_700_000_000_000_000_000


# ---------------------------------------------------------------------------
# Controllable stub adapter — simpler than FakeVenueAdapter for unit tests
# ---------------------------------------------------------------------------

class _Adapter:
    """Tracks calls and returns configurable responses."""

    def __init__(self, submit_raises: Exception | None = None) -> None:
        self._submit_raises = submit_raises
        self._counter = 0
        self.submit_calls = 0
        self.cancel_calls = 0

    def submit_order(self, intent):
        self.submit_calls += 1
        if self._submit_raises:
            raise self._submit_raises
        self._counter += 1
        return SubmitOrderResult(
            status=SubmitOrderStatus.ACKNOWLEDGED,
            client_order_id=f"coid-{self._counter}",
            venue_order_id=f"void-{self._counter}",
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


def _oms(adapter: _Adapter | None = None) -> OrderManagementSystem:
    return OrderManagementSystem(
        adapter or _Adapter(),
        package_id="test-pkg",
        order_qty=10,
    )


def _place(oms, *, slug=SLUG, token=TOKEN_A, price=PRICE):
    oms.place_or_replace(
        event_slug=slug, token_id=token, condition_id=COND,
        price_ticks=price, now_ns=NOW,
    )


def _fill(oms, coid: str, qty: float) -> None:
    oms.on_user_order_event(
        UserOrderEvent(client_order_id=coid, status="MATCHED",
                       fill_qty=qty, fill_price=0.91, ts_ns=NOW)
    )


def _cancel_ws(oms, coid: str) -> None:
    oms.on_user_order_event(
        UserOrderEvent(client_order_id=coid, status="CANCELED",
                       fill_qty=None, fill_price=None, ts_ns=NOW)
    )


# ---------------------------------------------------------------------------
# 1. First submit places order
# ---------------------------------------------------------------------------

def test_first_submit_places_order() -> None:
    oms = _oms()
    _place(oms)
    state = oms.order_state(SLUG)
    assert state.status == OrderStatus.OPEN
    assert state.token_id == TOKEN_A
    assert state.price_ticks == PRICE


# ---------------------------------------------------------------------------
# 2. Same token + same price is a no-op
# ---------------------------------------------------------------------------

def test_same_token_and_price_is_noop() -> None:
    adapter = _Adapter()
    oms = _oms(adapter)
    _place(oms)
    _place(oms)  # identical call
    assert adapter.submit_calls == 1  # second call did nothing


# ---------------------------------------------------------------------------
# 3. Price change cancels and resubmits
# ---------------------------------------------------------------------------

def test_price_change_cancels_and_resubmits() -> None:
    adapter = _Adapter()
    oms = _oms(adapter)
    _place(oms, price=91)
    first_coid = oms.order_state(SLUG).client_order_id

    _place(oms, price=92)

    state = oms.order_state(SLUG)
    assert state.status == OrderStatus.OPEN
    assert state.price_ticks == 92
    assert state.client_order_id != first_coid
    assert adapter.submit_calls == 2
    assert adapter.cancel_calls == 1


# ---------------------------------------------------------------------------
# 4. Token switch cancels and resubmits
# ---------------------------------------------------------------------------

def test_token_switch_cancels_and_resubmits() -> None:
    adapter = _Adapter()
    oms = _oms(adapter)
    _place(oms, token=TOKEN_A)
    _place(oms, token=TOKEN_B)

    state = oms.order_state(SLUG)
    assert state.status == OrderStatus.OPEN
    assert state.token_id == TOKEN_B
    assert adapter.submit_calls == 2
    assert adapter.cancel_calls == 1


# ---------------------------------------------------------------------------
# 5. Fill via WS accumulates to event
# ---------------------------------------------------------------------------

def test_fill_via_ws_accumulates_to_event() -> None:
    oms = _oms()
    _place(oms)
    coid = oms.order_state(SLUG).client_order_id

    _fill(oms, coid, 4.0)

    assert oms.order_state(SLUG).filled_qty == 4.0


# ---------------------------------------------------------------------------
# 6. Fills from two different tokens aggregate to the same event (key test)
# ---------------------------------------------------------------------------

def test_fills_from_two_tokens_aggregate_to_same_event() -> None:
    adapter = _Adapter()
    oms = _oms(adapter)

    # Order on TOKEN_A, partial fill of 3
    _place(oms, token=TOKEN_A, price=91)
    coid_a = oms.order_state(SLUG).client_order_id
    _fill(oms, coid_a, 3.0)

    # WS tells us order was canceled, switch to TOKEN_B
    _cancel_ws(oms, coid_a)
    _place(oms, token=TOKEN_B, price=92)
    coid_b = oms.order_state(SLUG).client_order_id
    _fill(oms, coid_b, 5.0)

    # Total should be 3 + 5 = 8, not just 5
    assert oms.order_state(SLUG).filled_qty == 8.0


# ---------------------------------------------------------------------------
# 7. WS cancel clears open order
# ---------------------------------------------------------------------------

def test_ws_cancel_clears_open_order() -> None:
    oms = _oms()
    _place(oms)
    coid = oms.order_state(SLUG).client_order_id

    _cancel_ws(oms, coid)

    state = oms.order_state(SLUG)
    assert state.status == OrderStatus.IDLE
    assert state.client_order_id is None
    assert state.token_id is None


# ---------------------------------------------------------------------------
# 8. Submit timeout → AMBIGUOUS
# ---------------------------------------------------------------------------

def test_submit_timeout_marks_ambiguous() -> None:
    oms = _oms(_Adapter(submit_raises=TimeoutError("submit timeout")))
    _place(oms)
    assert oms.order_state(SLUG).status == OrderStatus.AMBIGUOUS


# ---------------------------------------------------------------------------
# 9. AMBIGUOUS blocks further action
# ---------------------------------------------------------------------------

def test_ambiguous_blocks_further_action() -> None:
    adapter = _Adapter(submit_raises=TimeoutError("timeout"))
    oms = _oms(adapter)
    _place(oms)
    assert oms.order_state(SLUG).status == OrderStatus.AMBIGUOUS

    _place(oms, price=92)  # should be skipped entirely

    assert adapter.submit_calls == 1  # no second attempt


# ---------------------------------------------------------------------------
# 10. cancel_all clears all open orders
# ---------------------------------------------------------------------------

def test_cancel_all_clears_all_open_orders() -> None:
    adapter = _Adapter()
    oms = _oms(adapter)
    _place(oms, slug=SLUG)
    _place(oms, slug=SLUG2)

    oms.cancel_all(now_ns=NOW)

    assert oms.order_state(SLUG).status == OrderStatus.IDLE
    assert oms.order_state(SLUG2).status == OrderStatus.IDLE
    assert oms.all_open_client_order_ids() == set()
    assert adapter.cancel_calls == 2


# ---------------------------------------------------------------------------
# 11. Fill for unknown order is silently ignored
# ---------------------------------------------------------------------------

def test_fill_for_unknown_order_is_ignored() -> None:
    oms = _oms()
    oms.on_user_order_event(
        UserOrderEvent(client_order_id="ghost-coid", status="MATCHED",
                       fill_qty=5.0, fill_price=0.91, ts_ns=NOW)
    )
    assert oms.order_state(SLUG) is None  # no state created
