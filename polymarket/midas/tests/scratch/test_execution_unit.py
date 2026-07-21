from __future__ import annotations

import time as _time
import pytest

from executor.venue import (
    CancelOrderResult,
    CancelOrderStatus,
    ReconciliationResult,
    SubmitOrderResult,
    SubmitOrderStatus,
)
from harvester.execution import ExecutionConfig, ExecutionEngine
from harvester.market_data import BookUpdateEvent, UserOrderEvent
from harvester.oms import OrderManagementSystem, OrderStatus
from harvester.registry import TokenRecord, TokenRegistry
from harvester.strategy import StrategyConfig, TailHarvesterStrategy

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

ONE_HOUR_NS = 3_600_000_000_000
SLUG = "london-temp-april-30"
TOKEN = "tok-yes"
COND = "0xabcd"
BID_THRESHOLD = 0.90

# These must be anchored to real wall-clock time because execution.py calls
# time.time_ns() internally to evaluate the time window — a fixed 2023 timestamp
# would look expired from 2026 and the strategy would never PLACE.
_NOW_NS = _time.time_ns()
END_NS = _NOW_NS + ONE_HOUR_NS  # 1 hour from now


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeClient:
    """Emits a fixed sequence of events then ends — engine completes naturally."""

    def __init__(self, *events: BookUpdateEvent | UserOrderEvent) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    async def events(self):  # type: ignore[override]
        for e in self._events:
            yield e


class _Adapter:
    """Controllable stub — records all calls."""

    def __init__(self) -> None:
        self.submit_calls = 0
        self.cancel_calls = 0
        self.poll_calls = 0
        self._order_counter = 0

    def submit_order(self, intent):
        self.submit_calls += 1
        self._order_counter += 1
        return SubmitOrderResult(
            status=SubmitOrderStatus.ACKNOWLEDGED,
            client_order_id=f"coid-{self._order_counter}",
            venue_order_id=f"void-{self._order_counter}",
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
        self.poll_calls += 1
        return tuple()

    def reconcile_open_orders(self, expected):
        return ReconciliationResult(
            venue_open_client_order_ids=tuple(),
            unknown_open_client_order_ids=tuple(),
            missing_expected_client_order_ids=tuple(),
            generated_events=tuple(),
        )


def _build(
    *events: BookUpdateEvent | UserOrderEvent,
    adapter: _Adapter | None = None,
) -> tuple[ExecutionEngine, _Adapter, OrderManagementSystem]:
    if adapter is None:
        adapter = _Adapter()
    record = TokenRecord(token_id=TOKEN, event_slug=SLUG, condition_id=COND, end_date_ns=END_NS)
    registry = TokenRegistry([record], bid_threshold=BID_THRESHOLD)
    strategy = TailHarvesterStrategy(
        registry,
        StrategyConfig(max_hours_before_close=2.0, min_reprice_ticks=2),
    )
    oms = OrderManagementSystem(adapter, package_id="test-pkg", order_qty=10)
    engine = ExecutionEngine(
        client=_FakeClient(*events),
        registry=registry,
        strategy=strategy,
        oms=oms,
        adapter=adapter,
        # Long poll interval so the background poll never fires during tests
        config=ExecutionConfig(poll_interval_s=3600.0, shutdown_timeout_s=5.0),
    )
    return engine, adapter, oms


# ---------------------------------------------------------------------------
# 1. Qualifying price update → order submitted
# ---------------------------------------------------------------------------


async def test_qualifying_book_update_submits_order() -> None:
    event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    engine, adapter, oms = _build(event)
    await engine.run()
    assert adapter.submit_calls == 1
    assert oms.order_state(SLUG).status == OrderStatus.IDLE  # IDLE because shutdown cancelled it


# ---------------------------------------------------------------------------
# 2. Price below threshold → no order submitted
# ---------------------------------------------------------------------------


async def test_subthreshold_book_update_does_not_submit() -> None:
    event = BookUpdateEvent(token_id=TOKEN, best_bid=0.50, best_ask=0.50, ts_ns=_NOW_NS)
    engine, adapter, _ = _build(event)
    await engine.run()
    assert adapter.submit_calls == 0


# ---------------------------------------------------------------------------
# 3. Fill event forwarded to OMS and accumulated
# ---------------------------------------------------------------------------


async def test_fill_event_accumulated_by_oms() -> None:
    book = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    # The adapter returns coid-1 for the first submit, so the fill must use that id
    fill = UserOrderEvent(
        client_order_id="coid-1",
        status="MATCHED",
        fill_qty=5.0,
        fill_price=0.93,
        ts_ns=_NOW_NS + 1,
    )
    engine, _, oms = _build(book, fill)
    await engine.run()
    # filled_qty survives shutdown (shutdown only clears open state, not fill accumulator)
    state = oms.order_state(SLUG)
    assert state is not None
    assert state.filled_qty == 5.0


# ---------------------------------------------------------------------------
# 4. Shutdown sequence cancels the open order
# ---------------------------------------------------------------------------


async def test_shutdown_cancels_open_order() -> None:
    event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    engine, adapter, oms = _build(event)
    await engine.run()
    # Order was placed (submit_calls=1) then cancelled during shutdown (cancel_calls=1)
    assert adapter.submit_calls == 1
    assert adapter.cancel_calls == 1
    assert oms.order_state(SLUG).status == OrderStatus.IDLE


# ---------------------------------------------------------------------------
# 5. No order placed when outside time window → no cancel on shutdown
# ---------------------------------------------------------------------------


async def test_outside_time_window_no_order_no_cancel() -> None:
    from harvester.registry import TokenRecord, TokenRegistry
    from harvester.strategy import StrategyConfig, TailHarvesterStrategy

    far_end = _time.time_ns() + 10 * ONE_HOUR_NS  # 10h away, max_hours=2
    adapter = _Adapter()
    record = TokenRecord(token_id=TOKEN, event_slug=SLUG, condition_id=COND, end_date_ns=far_end)
    registry = TokenRegistry([record], bid_threshold=BID_THRESHOLD)
    strategy = TailHarvesterStrategy(
        registry, StrategyConfig(max_hours_before_close=2.0, min_reprice_ticks=2)
    )
    oms = OrderManagementSystem(adapter, package_id="test-pkg", order_qty=10)
    engine = ExecutionEngine(
        client=_FakeClient(
            BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
        ),
        registry=registry,
        strategy=strategy,
        oms=oms,
        adapter=adapter,
        config=ExecutionConfig(poll_interval_s=3600.0, shutdown_timeout_s=5.0),
    )
    await engine.run()
    assert adapter.submit_calls == 0
    assert adapter.cancel_calls == 0
