from __future__ import annotations

import time as _time

import pytest

from executor.risk import ExecutionRiskManager, RiskManagerConfig
from executor.venue import (
    CancelOrderResult,
    CancelOrderStatus,
    ReconciliationResult,
    SubmitOrderResult,
    SubmitOrderStatus,
    VenueCancelEvent,
)
from harvester.execution import ExecutionConfig, ExecutionEngine
from harvester.market_data import BookUpdateEvent, UserOrderEvent
from harvester.oms import OrderManagementSystem, OrderStatus
from harvester.registry import TokenRecord, TokenRegistry
from harvester.strategy import StrategyConfig, TailHarvesterStrategy

pytestmark = [pytest.mark.unit, pytest.mark.harvester]

SLUG = "london-temp-april-30"
TOKEN = "tok-yes"
COND = "0xabcd"
BID_THRESHOLD = 0.90

_NOW_NS = _time.time_ns()


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeClient:
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
    def __init__(self) -> None:
        self.submit_calls = 0
        self.cancel_calls = 0
        self.poll_calls = 0
        self._order_ctr = 0
        self._reconcile_events: tuple = tuple()

    def submit_order(self, intent):
        self.submit_calls += 1
        self._order_ctr += 1
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
        self.poll_calls += 1
        return tuple()

    def get_positions(self):
        return []

    def reconcile_open_orders(self, expected):
        return ReconciliationResult(
            venue_open_client_order_ids=tuple(),
            unknown_open_client_order_ids=tuple(),
            missing_expected_client_order_ids=tuple(),
            generated_events=self._reconcile_events,
        )


def _build(
    *events: BookUpdateEvent | UserOrderEvent,
    adapter: _Adapter | None = None,
    risk: ExecutionRiskManager | None = None,
    registry: TokenRegistry | None = None,
) -> tuple[ExecutionEngine, _Adapter, OrderManagementSystem]:
    if adapter is None:
        adapter = _Adapter()
    if registry is None:
        record = TokenRecord(token_id=TOKEN, event_slug=SLUG, condition_id=COND)
        registry = TokenRegistry([record], bid_threshold=BID_THRESHOLD)
    strategy = TailHarvesterStrategy(registry, StrategyConfig(min_reprice_ticks=2))
    oms = OrderManagementSystem(adapter, package_id="test-pkg", order_qty=10)
    engine = ExecutionEngine(
        client=_FakeClient(*events),
        registry=registry,
        strategy=strategy,
        oms=oms,
        adapter=adapter,
        config=ExecutionConfig(poll_interval_s=3600.0, shutdown_timeout_s=5.0),
        risk=risk,
    )
    return engine, adapter, oms


# ---------------------------------------------------------------------------
# 1. Qualifying price → order placed
# ---------------------------------------------------------------------------


async def test_qualifying_book_update_submits_order() -> None:
    event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    engine, adapter, _ = _build(event)
    await engine.run()
    assert adapter.submit_calls == 1


# ---------------------------------------------------------------------------
# 2. Below threshold → no order
# ---------------------------------------------------------------------------


async def test_subthreshold_book_update_does_not_submit() -> None:
    event = BookUpdateEvent(token_id=TOKEN, best_bid=0.50, best_ask=0.50, ts_ns=_NOW_NS)
    engine, adapter, _ = _build(event)
    await engine.run()
    assert adapter.submit_calls == 0


# ---------------------------------------------------------------------------
# 3. Fill forwarded to OMS
# ---------------------------------------------------------------------------


async def test_fill_event_accumulated_by_oms() -> None:
    book = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    fill = UserOrderEvent(
        client_order_id="coid-1", status="MATCHED",
        fill_qty=5.0, fill_price=0.93, ts_ns=_NOW_NS + 1,
    )
    engine, _, oms = _build(book, fill)
    await engine.run()
    state = oms.order_state(SLUG)
    assert state is not None
    assert state.filled_qty == 5.0
    assert abs(state.filled_usdc - 5.0 * 0.93) < 1e-9


# ---------------------------------------------------------------------------
# 4. Shutdown cancels open order
# ---------------------------------------------------------------------------


async def test_shutdown_cancels_open_order() -> None:
    event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    engine, adapter, oms = _build(event)
    await engine.run()
    assert adapter.submit_calls == 1
    assert adapter.cancel_calls == 1
    assert oms.order_state(SLUG).status == OrderStatus.IDLE


# ---------------------------------------------------------------------------
# 5. Closed market → no order placed, no cancel on shutdown
# ---------------------------------------------------------------------------


async def test_closed_market_does_not_submit() -> None:
    record = TokenRecord(token_id=TOKEN, event_slug=SLUG, condition_id=COND)
    registry = TokenRegistry([record], bid_threshold=BID_THRESHOLD)
    registry.mark_closed(SLUG)
    event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    engine, adapter, _ = _build(event, registry=registry)
    await engine.run()
    assert adapter.submit_calls == 0
    assert adapter.cancel_calls == 0


# ---------------------------------------------------------------------------
# 6. Risk gate — notional cap blocks PLACE
# ---------------------------------------------------------------------------


async def test_risk_notional_cap_blocks_order() -> None:
    risk = ExecutionRiskManager(
        RiskManagerConfig(max_notional_per_event_usdc=0.0),
    )
    event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    engine, adapter, _ = _build(event, risk=risk)
    await engine.run()
    assert adapter.submit_calls == 0


# ---------------------------------------------------------------------------
# 7. Risk gate — manual kill switch blocks PLACE
# ---------------------------------------------------------------------------


async def test_risk_kill_switch_blocks_order() -> None:
    risk = ExecutionRiskManager(RiskManagerConfig())
    risk.activate_manual_kill_switch("test halt", _NOW_NS)
    event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    engine, adapter, _ = _build(event, risk=risk)
    await engine.run()
    assert adapter.submit_calls == 0


# ---------------------------------------------------------------------------
# 8. Risk gate allows order when cap not reached
# ---------------------------------------------------------------------------


async def test_risk_gate_allows_when_under_cap() -> None:
    risk = ExecutionRiskManager(
        RiskManagerConfig(max_notional_per_event_usdc=100.0),
    )
    event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    engine, adapter, _ = _build(event, risk=risk)
    await engine.run()
    assert adapter.submit_calls == 1


# ---------------------------------------------------------------------------
# 9. Reconcile on startup feeds generated events into OMS
# ---------------------------------------------------------------------------


async def test_reconcile_on_startup_processes_generated_events() -> None:
    adapter = _Adapter()
    record = TokenRecord(token_id=TOKEN, event_slug=SLUG, condition_id=COND)
    registry = TokenRegistry([record], bid_threshold=BID_THRESHOLD)
    strategy = TailHarvesterStrategy(registry, StrategyConfig(min_reprice_ticks=2))
    oms = OrderManagementSystem(adapter, package_id="test-pkg", order_qty=10)

    # Pre-populate OMS with an OPEN order as if from a prior run
    oms.place_or_replace(
        event_slug=SLUG, token_id=TOKEN, condition_id=COND,
        price_ticks=91, now_ns=_NOW_NS,
    )
    assert oms.order_state(SLUG).status == OrderStatus.OPEN
    coid = oms.order_state(SLUG).client_order_id

    # Reconcile will return a VenueCancelEvent for that order (venue already closed it)
    adapter._reconcile_events = (
        VenueCancelEvent(
            package_id="test-pkg", leg_id=TOKEN, client_order_id=coid,
            canceled_qty=10, reason="expired", ts_ns=_NOW_NS,
        ),
    )

    engine = ExecutionEngine(
        client=_FakeClient(),  # no market events
        registry=registry,
        strategy=strategy,
        oms=oms,
        adapter=adapter,
        config=ExecutionConfig(poll_interval_s=3600.0, shutdown_timeout_s=5.0),
    )
    await engine.run()

    # Reconcile should have cleared the stale OPEN order
    assert oms.order_state(SLUG).status == OrderStatus.IDLE
