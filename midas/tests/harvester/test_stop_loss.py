from __future__ import annotations

import time as _time

import pytest

from executor.venue import (
    CancelOrderResult,
    CancelOrderStatus,
    ReconciliationResult,
    SubmitOrderResult,
    SubmitOrderStatus,
    VenueCancelEvent,
    VenueOrderIntent,
)
from harvester.execution import ExecutionConfig, ExecutionEngine
from harvester.market_data import BookUpdateEvent, UserOrderEvent
from harvester.oms import EventOrderState, OrderManagementSystem, OrderStatus
from harvester.registry import TokenRecord, TokenRegistry
from harvester.strategy import StrategyConfig, TailHarvesterStrategy

pytestmark = [pytest.mark.unit, pytest.mark.harvester]

SLUG = "london-temp-april-30"
TOKEN = "tok-yes"
COND = "0xabcd"
BID_THRESHOLD = 0.90
STOP_LOSS = 0.05   # 5¢  — prices are always in [0, 1]; NEVER write 5 or 60
_NOW_NS = _time.time_ns()


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _RecordingAdapter:
    """Adapter that records all submitted intents for assertions."""

    def __init__(self) -> None:
        self.intents: list[VenueOrderIntent] = []
        self._order_ctr = 0

    def submit_order(self, intent: VenueOrderIntent) -> SubmitOrderResult:
        self.intents.append(intent)
        self._order_ctr += 1
        return SubmitOrderResult(
            status=SubmitOrderStatus.ACKNOWLEDGED,
            client_order_id=f"coid-{self._order_ctr}",
            venue_order_id=f"void-{self._order_ctr}",
            events=tuple(),
            ambiguous=False,
        )

    def cancel_order(self, *, client_order_id=None, venue_order_id=None, reason="", now_ns=None):
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

    def get_positions(self):
        return []

    def buy_count(self) -> int:
        from executor.venue import Side
        return sum(1 for i in self.intents if i.side == Side.BUY)

    def sell_count(self) -> int:
        from executor.venue import Side
        return sum(1 for i in self.intents if i.side == Side.SELL)

    def sell_intents(self) -> list[VenueOrderIntent]:
        from executor.venue import Side
        return [i for i in self.intents if i.side == Side.SELL]


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


def _build(
    *events: BookUpdateEvent | UserOrderEvent,
    stop_loss_price: float = STOP_LOSS,
    adapter: _RecordingAdapter | None = None,
) -> tuple[ExecutionEngine, _RecordingAdapter, OrderManagementSystem, TokenRegistry]:
    if adapter is None:
        adapter = _RecordingAdapter()
    record = TokenRecord(token_id=TOKEN, event_slug=SLUG, condition_id=COND)
    registry = TokenRegistry([record], bid_threshold=BID_THRESHOLD)
    strategy = TailHarvesterStrategy(registry, StrategyConfig(min_reprice_ticks=2))
    oms = OrderManagementSystem(adapter, package_id="test-pkg", order_qty=5)
    engine = ExecutionEngine(
        client=_FakeClient(*events),
        registry=registry,
        strategy=strategy,
        oms=oms,
        adapter=adapter,
        config=ExecutionConfig(
            poll_interval_s=3600.0,
            shutdown_timeout_s=5.0,
            stop_loss_price=stop_loss_price,
        ),
    )
    return engine, adapter, oms, registry


# ---------------------------------------------------------------------------
# 1. Stop-loss triggers and submits a SELL when bid drops below threshold
# ---------------------------------------------------------------------------


async def test_stop_loss_triggers_sell_on_price_breach() -> None:
    buy_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    fill_event = UserOrderEvent(
        client_order_id="coid-1", status="MATCHED",
        fill_qty=5.0, fill_price=0.93, ts_ns=_NOW_NS + 1,
    )
    crash_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.03, best_ask=0.97, ts_ns=_NOW_NS + 2)

    engine, adapter, oms, _ = _build(buy_event, fill_event, crash_event)
    await engine.run()

    assert adapter.buy_count() == 1
    assert adapter.sell_count() == 1

    sell = adapter.sell_intents()[0]
    from executor.venue import Side, TimeInForce
    assert sell.side == Side.SELL
    assert sell.token_id == TOKEN
    # bid=0.03, tick_size=0.01 → floor(0.03 * 100) = 3 ticks
    assert sell.limit_price_ticks == 3
    assert sell.tif == TimeInForce.GTC
    assert sell.quantity == 5  # round(filled_qty=5.0)


# ---------------------------------------------------------------------------
# 2. No fill → stop-loss never fires
# ---------------------------------------------------------------------------


async def test_stop_loss_does_not_trigger_without_fill() -> None:
    buy_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    crash_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.03, best_ask=0.97, ts_ns=_NOW_NS + 1)

    engine, adapter, _, _ = _build(buy_event, crash_event)
    await engine.run()

    assert adapter.sell_count() == 0


# ---------------------------------------------------------------------------
# 3. Stop-loss fires exactly once even when multiple low-price ticks arrive
# ---------------------------------------------------------------------------


async def test_stop_loss_fires_only_once() -> None:
    buy_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    fill_event = UserOrderEvent(
        client_order_id="coid-1", status="MATCHED",
        fill_qty=5.0, fill_price=0.93, ts_ns=_NOW_NS + 1,
    )
    crash1 = BookUpdateEvent(token_id=TOKEN, best_bid=0.03, best_ask=0.97, ts_ns=_NOW_NS + 2)
    crash2 = BookUpdateEvent(token_id=TOKEN, best_bid=0.02, best_ask=0.98, ts_ns=_NOW_NS + 3)
    crash3 = BookUpdateEvent(token_id=TOKEN, best_bid=0.01, best_ask=0.99, ts_ns=_NOW_NS + 4)

    engine, adapter, _, _ = _build(buy_event, fill_event, crash1, crash2, crash3)
    await engine.run()

    assert adapter.sell_count() == 1


# ---------------------------------------------------------------------------
# 4. Disabled when stop_loss_price = 0
# ---------------------------------------------------------------------------


async def test_stop_loss_disabled_when_zero() -> None:
    buy_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    fill_event = UserOrderEvent(
        client_order_id="coid-1", status="MATCHED",
        fill_qty=5.0, fill_price=0.93, ts_ns=_NOW_NS + 1,
    )
    crash_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.01, best_ask=0.99, ts_ns=_NOW_NS + 2)

    engine, adapter, _, _ = _build(buy_event, fill_event, crash_event, stop_loss_price=0.0)
    await engine.run()

    assert adapter.sell_count() == 0


# ---------------------------------------------------------------------------
# 5. Above threshold — no sell even with a filled position
# ---------------------------------------------------------------------------


async def test_stop_loss_no_trigger_above_threshold() -> None:
    buy_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    fill_event = UserOrderEvent(
        client_order_id="coid-1", status="MATCHED",
        fill_qty=5.0, fill_price=0.93, ts_ns=_NOW_NS + 1,
    )
    # Bid is above STOP_LOSS (0.05), so no sell
    safe_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.91, best_ask=0.09, ts_ns=_NOW_NS + 2)

    engine, adapter, _, _ = _build(buy_event, fill_event, safe_event)
    await engine.run()

    assert adapter.sell_count() == 0


# ---------------------------------------------------------------------------
# 6. After stop-loss fires, slug is permanently blacklisted
# ---------------------------------------------------------------------------


async def test_stop_loss_blacklists_slug() -> None:
    buy_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    fill_event = UserOrderEvent(
        client_order_id="coid-1", status="MATCHED",
        fill_qty=5.0, fill_price=0.93, ts_ns=_NOW_NS + 1,
    )
    crash_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.03, best_ask=0.97, ts_ns=_NOW_NS + 2)

    engine, _, _, registry = _build(buy_event, fill_event, crash_event)
    await engine.run()

    assert SLUG in registry._closed_slugs


# ---------------------------------------------------------------------------
# 7. OMS.sell_position unit test — correct intent with fractional fill rounding
# ---------------------------------------------------------------------------


def test_sell_position_submits_correct_intent() -> None:
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    state = oms._get_or_create(SLUG)
    state.token_id = TOKEN       # must use token_id, not position_token_id
    state.condition_id = COND
    state.tick_size = 0.01
    state.filled_qty = 3.7  # fractional — should round to 4

    oms.sell_position(event_slug=SLUG, now_ns=_NOW_NS, bid_price=0.01)

    assert adapter.sell_count() == 1
    sell = adapter.sell_intents()[0]
    from executor.venue import Side, TimeInForce
    assert sell.side == Side.SELL
    assert sell.token_id == TOKEN
    assert sell.market_id == COND
    # bid=0.01, tick_size=0.01 → floor(0.01 * 100) = 1 tick
    assert sell.limit_price_ticks == 1
    assert sell.tif == TimeInForce.GTC
    assert sell.quantity == 4  # round(3.7)


# ---------------------------------------------------------------------------
# 8. OMS.sell_position no-ops when nothing to sell
# ---------------------------------------------------------------------------


def test_sell_position_noop_when_no_fill() -> None:
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    # No state for this slug
    oms.sell_position(event_slug=SLUG, now_ns=_NOW_NS, bid_price=0.05)
    assert adapter.sell_count() == 0

    # State exists but filled_qty is zero
    state = oms._get_or_create(SLUG)
    state.token_id = TOKEN       # must use token_id, not position_token_id
    state.condition_id = COND
    state.filled_qty = 0.0

    oms.sell_position(event_slug=SLUG, now_ns=_NOW_NS, bid_price=0.05)
    assert adapter.sell_count() == 0


# ---------------------------------------------------------------------------
# 9. CRITICAL: stop-loss fires even after the order slot is cleared by cancel
#
# Real Polymarket flow: GTC BUY gets fully filled → exchange sends CANCELLED
# with reason "fully_matched" which calls _clear_open.  Before the fix,
# _clear_open set token_id=None so the stop-loss check silently failed.
# ---------------------------------------------------------------------------


async def test_stop_loss_fires_after_slot_cleared_by_cancel() -> None:
    """
    Buy → WS fill → WS cancel (CANCELLED) → price crash.
    The cancel clears the OMS slot, but we still hold a position.
    Stop-loss MUST fire and submit a SELL.
    """
    buy_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    fill_event = UserOrderEvent(
        client_order_id="coid-1", status="MATCHED",
        fill_qty=5.0, fill_price=0.93, ts_ns=_NOW_NS + 1,
    )
    # Slot cleared by cancel (Polymarket sends this after a fully-filled GTC order)
    cancel_event = UserOrderEvent(
        client_order_id="coid-1", status="CANCELLED",
        fill_qty=None, fill_price=None, ts_ns=_NOW_NS + 2,
    )
    crash_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.03, best_ask=0.97, ts_ns=_NOW_NS + 3)

    engine, adapter, oms, _ = _build(buy_event, fill_event, cancel_event, crash_event)
    await engine.run()

    # Slot was cleared but position still exists — stop-loss must fire
    assert adapter.sell_count() == 1, (
        "stop-loss must fire after slot cleared by cancel: "
        f"filled_qty={oms.order_state(SLUG).filled_qty if oms.order_state(SLUG) else 'N/A'}"
    )
    sell = adapter.sell_intents()[0]
    assert sell.token_id == TOKEN


# ---------------------------------------------------------------------------
# 10. CRITICAL: _clear_open preserves token_id when filled_qty > 0
# ---------------------------------------------------------------------------


def test_clear_open_preserves_token_id_when_position_exists() -> None:
    """_clear_open must not wipe token_id/condition_id when a position still exists."""
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    state = oms._get_or_create(SLUG)
    state.token_id = TOKEN
    state.condition_id = COND
    state.client_order_id = "coid-1"
    state.status = OrderStatus.OPEN
    state.filled_qty = 5.0  # position exists

    # Simulate _clear_open (as triggered by a cancel event)
    oms._clear_open(state)

    assert state.status == OrderStatus.IDLE
    assert state.client_order_id is None
    # token_id and condition_id must be preserved because filled_qty > 0
    assert state.token_id == TOKEN, "_clear_open must preserve token_id when position exists"
    assert state.condition_id == COND, "_clear_open must preserve condition_id when position exists"


def test_clear_open_wipes_token_id_when_no_position() -> None:
    """_clear_open must clear token_id when there is no fill at all."""
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    state = oms._get_or_create(SLUG)
    state.token_id = TOKEN
    state.condition_id = COND
    state.client_order_id = "coid-1"
    state.status = OrderStatus.OPEN
    state.filled_qty = 0.0
    state.confirmed_notional = 0.0  # no evidence of any fill

    oms._clear_open(state)

    assert state.status == OrderStatus.IDLE
    assert state.token_id is None, "_clear_open must wipe token_id when no position"
    assert state.condition_id is None


def test_clear_open_preserves_token_id_when_confirmed_notional_set() -> None:
    """confirmed_notional > 0 means we bought but WS fills were missed — still hold a position."""
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    state = oms._get_or_create(SLUG)
    state.token_id = TOKEN
    state.condition_id = COND
    state.client_order_id = "coid-1"
    state.status = OrderStatus.OPEN
    state.filled_qty = 0.0          # WS fill events missed
    state.confirmed_notional = 4.65 # but poll-confirmed this order was matched

    oms._clear_open(state)

    assert state.token_id == TOKEN, "confirmed_notional > 0 implies position — must preserve token_id"
    assert state.condition_id == COND


# ---------------------------------------------------------------------------
# 11. Boundary: bid exactly at stop-loss price does NOT trigger (strict <)
# ---------------------------------------------------------------------------


async def test_stop_loss_exact_boundary_does_not_trigger() -> None:
    """bid == stop_loss_price must NOT trigger: condition is strictly less-than."""
    buy_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    fill_event = UserOrderEvent(
        client_order_id="coid-1", status="MATCHED",
        fill_qty=5.0, fill_price=0.93, ts_ns=_NOW_NS + 1,
    )
    # Bid sits exactly on the threshold — must not trigger
    exact_event = BookUpdateEvent(token_id=TOKEN, best_bid=STOP_LOSS, best_ask=0.95, ts_ns=_NOW_NS + 2)

    engine, adapter, _, _ = _build(buy_event, fill_event, exact_event)
    await engine.run()

    assert adapter.sell_count() == 0, "bid == stop_loss_price must not trigger (condition is strict <)"


# ---------------------------------------------------------------------------
# 12. Sell price tick calculation — correct for both tick-size families
# ---------------------------------------------------------------------------


def test_sell_position_tick_price_standard_market() -> None:
    """sell limit price in ticks is correct for a 0.01 tick-size market."""
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    state = oms._get_or_create(SLUG)
    state.token_id = TOKEN
    state.condition_id = COND
    state.tick_size = 0.01
    state.filled_qty = 10.0

    # bid=0.03, tick_size=0.01 → floor(0.03 * 100) = 3 ticks
    oms.sell_position(event_slug=SLUG, now_ns=_NOW_NS, bid_price=0.03)
    sell = adapter.sell_intents()[0]
    assert sell.limit_price_ticks == 3

    # bid=0.60, tick_size=0.01 → floor(0.60 * 100) = 60 ticks
    adapter2 = _RecordingAdapter()
    oms2 = OrderManagementSystem(adapter2, package_id="pkg", order_qty=5)
    s2 = oms2._get_or_create(SLUG)
    s2.token_id = TOKEN
    s2.condition_id = COND
    s2.tick_size = 0.01
    s2.filled_qty = 10.0
    oms2.sell_position(event_slug=SLUG, now_ns=_NOW_NS, bid_price=0.60)
    sell2 = adapter2.sell_intents()[0]
    assert sell2.limit_price_ticks == 60


def test_sell_position_tick_price_fine_market() -> None:
    """sell limit price is correct for a 0.001 tick-size market."""
    TOKEN_FINE = "tok-fine"
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    state = oms._get_or_create(SLUG)
    state.token_id = TOKEN_FINE
    state.condition_id = COND
    state.tick_size = 0.001
    state.filled_qty = 5.0

    # bid=0.030, tick_size=0.001 → floor(0.030 * 1000) = 30 ticks
    oms.sell_position(event_slug=SLUG, now_ns=_NOW_NS, bid_price=0.030)
    sell = adapter.sell_intents()[0]
    assert sell.limit_price_ticks == 30

    # bid=0.600, tick_size=0.001 → floor(0.600 * 1000) = 600 ticks
    adapter2 = _RecordingAdapter()
    oms2 = OrderManagementSystem(adapter2, package_id="pkg", order_qty=5)
    s2 = oms2._get_or_create(SLUG)
    s2.token_id = TOKEN_FINE
    s2.condition_id = COND
    s2.tick_size = 0.001
    s2.filled_qty = 5.0
    oms2.sell_position(event_slug=SLUG, now_ns=_NOW_NS, bid_price=0.600)
    sell2 = adapter2.sell_intents()[0]
    assert sell2.limit_price_ticks == 600


def test_sell_position_minimum_one_tick() -> None:
    """sell limit is clamped to at least 1 tick, never 0."""
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    state = oms._get_or_create(SLUG)
    state.token_id = TOKEN
    state.condition_id = COND
    state.tick_size = 0.01
    state.filled_qty = 10.0

    # bid=0.005 < 1 tick (0.01) → should clamp to 1
    oms.sell_position(event_slug=SLUG, now_ns=_NOW_NS, bid_price=0.005)
    sell = adapter.sell_intents()[0]
    assert sell.limit_price_ticks == 1, "sell limit must be clamped to >= 1 tick"


# ---------------------------------------------------------------------------
# 13. stop_loss_price parameter scale sanity — 0.6 vs 60
# ---------------------------------------------------------------------------


async def test_stop_loss_price_scale_0_6_fires_at_60_cents() -> None:
    """
    Polymarket prices are always in [0, 1].
    RISK_MIN_TOKEN_PRICE=0.6 means 60 cents.
    This test verifies the comparison is against the [0,1] float, not cents.
    """
    buy_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    fill_event = UserOrderEvent(
        client_order_id="coid-1", status="MATCHED",
        fill_qty=5.0, fill_price=0.93, ts_ns=_NOW_NS + 1,
    )
    # Bid at 55 cents — below 0.6 → must trigger
    crash_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.55, best_ask=0.45, ts_ns=_NOW_NS + 2)
    # Bid at 65 cents — above 0.6 → must NOT trigger
    safe_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.65, best_ask=0.35, ts_ns=_NOW_NS + 3)

    # With stop_loss=0.6: crash at 0.55 fires, safe at 0.65 does not
    engine, adapter, _, _ = _build(
        buy_event, fill_event, safe_event, stop_loss_price=0.6,
    )
    await engine.run()
    assert adapter.sell_count() == 0, "0.65 bid is above 0.6 stop-loss, must NOT trigger"

    # Now with crash below 0.6
    engine2, adapter2, _, _ = _build(
        buy_event, fill_event, crash_event, stop_loss_price=0.6,
    )
    await engine2.run()
    assert adapter2.sell_count() == 1, "0.55 bid is below 0.6 stop-loss, must trigger"


async def test_stop_loss_price_60_would_always_fire() -> None:
    """
    Setting stop_loss_price=60 (instead of 0.6) is broken: best_bid is always
    in [0,1] so the condition best_bid < 60 is ALWAYS true, firing on every tick
    even before any fill occurs — a sanity check that 0.6 is the correct value.
    """
    # No fill at all — but stop_loss_price=60 means any tick would pass the price check
    buy_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)
    fill_event = UserOrderEvent(
        client_order_id="coid-1", status="MATCHED",
        fill_qty=5.0, fill_price=0.93, ts_ns=_NOW_NS + 1,
    )
    # Even a healthy 93-cent bid
    healthy_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS + 2)

    engine, adapter, _, _ = _build(
        buy_event, fill_event, healthy_event, stop_loss_price=60.0,
    )
    await engine.run()
    # stop_loss=60 fires even at 93 cents — proves 60 is the wrong scale
    assert adapter.sell_count() == 1, (
        "stop_loss_price=60 always fires because best_bid is in [0,1] — "
        "this confirms 0.6 is the correct value, not 60"
    )


# ===========================================================================
# EDGE CASE TESTS — the three remaining failure paths after the _clear_open fix
# ===========================================================================

# ---------------------------------------------------------------------------
# PATH A: Out-of-order WS events (CANCELLED arrives before MATCHED)
#
# Polymarket WS can deliver events out of order on reconnect.
# CANCELLED arrives while filled_qty=0 → _clear_open wipes token_id.
# Then MATCHED fill arrives late → filled_qty=5, but token_id would stay None
# without the _coid_to_token_data restoration introduced in _submit.
# ---------------------------------------------------------------------------


def test_late_fill_after_early_cancel_restores_token_id() -> None:
    """
    Out-of-order WS: CANCELLED arrives before MATCHED.
    The late MATCHED fill must restore token_id from _coid_to_token_data.
    """
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    # Place a BUY order — _submit stores (TOKEN, COND, tick_size) per coid
    oms.place_or_replace(
        event_slug=SLUG, token_id=TOKEN, condition_id=COND,
        price_ticks=93, now_ns=_NOW_NS,
    )
    state = oms.order_state(SLUG)
    assert state.status == OrderStatus.OPEN
    coid = state.client_order_id  # "coid-1" from fake adapter

    # CANCELLED arrives first — filled_qty=0, so _clear_open wipes token_id
    oms.on_user_order_event(UserOrderEvent(
        client_order_id=coid, status="CANCELLED",
        fill_qty=None, fill_price=None, ts_ns=_NOW_NS + 1,
    ))
    state = oms.order_state(SLUG)
    assert state.status == OrderStatus.IDLE
    assert state.token_id is None, "confirm token_id was wiped (pre-condition for this test)"

    # MATCHED fill arrives late — must restore token_id from _coid_to_token_data
    oms.on_user_order_event(UserOrderEvent(
        client_order_id=coid, status="MATCHED",
        fill_qty=5.0, fill_price=0.93, ts_ns=_NOW_NS + 2,
    ))

    state = oms.order_state(SLUG)
    assert state.filled_qty == 5.0
    assert state.token_id == TOKEN, (
        "late MATCHED fill must restore token_id — "
        "without this, stop-loss is blind after out-of-order WS events"
    )
    assert state.condition_id == COND


async def test_stop_loss_fires_after_out_of_order_ws_cancel_then_fill() -> None:
    """
    Full integration: out-of-order WS (cancel first, fill second) then price crash.
    Stop-loss must fire because the late fill restored token_id.
    """
    buy_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.93, best_ask=0.07, ts_ns=_NOW_NS)

    # Out-of-order: cancel before fill
    cancel_first = UserOrderEvent(
        client_order_id="coid-1", status="CANCELLED",
        fill_qty=None, fill_price=None, ts_ns=_NOW_NS + 1,
    )
    fill_late = UserOrderEvent(
        client_order_id="coid-1", status="MATCHED",
        fill_qty=5.0, fill_price=0.93, ts_ns=_NOW_NS + 2,
    )
    crash_event = BookUpdateEvent(token_id=TOKEN, best_bid=0.03, best_ask=0.97, ts_ns=_NOW_NS + 3)

    engine, adapter, oms, _ = _build(buy_event, cancel_first, fill_late, crash_event)
    await engine.run()

    assert adapter.sell_count() == 1, (
        "stop-loss must fire even when WS delivered CANCELLED before MATCHED"
    )


# ---------------------------------------------------------------------------
# PATH B & C: update_filled_from_position restores token identity
#
# Covers two scenarios:
#   B — WS cancel fires, poll's fully_matched guard fails (client_order_id already None),
#       so confirmed_notional never gets set. Only position reconcile can save us.
#   C — Bot restart: fresh OMS, position reconcile is the first source of truth.
#
# In both cases update_filled_from_position must restore token_id/condition_id.
# ---------------------------------------------------------------------------


def test_update_filled_from_position_restores_token_id_when_none() -> None:
    """
    Position reconcile (every 2 min) must restore token_id when it was wiped,
    so stop-loss and sell_position can work even after all fill tracking failed.
    """
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    # State with wiped token (simulate worst-case: all tracking failed)
    state = oms._get_or_create(SLUG)
    state.token_id = None
    state.condition_id = None
    state.tick_size = 0.01
    state.filled_qty = 0.0

    corrected = oms.update_filled_from_position(
        SLUG,
        actual_qty=5.0,
        actual_cost_usdc=4.65,
        token_id=TOKEN,
        condition_id=COND,
        tick_size=0.01,
    )

    assert corrected is True
    state = oms.order_state(SLUG)
    assert state.token_id == TOKEN, "position reconcile must restore token_id"
    assert state.condition_id == COND
    assert state.filled_qty == 5.0


def test_update_filled_from_position_does_not_overwrite_active_token_id() -> None:
    """
    If an order is already live (token_id is set), position reconcile must NOT
    overwrite it — the live order's identity takes precedence.
    """
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    state = oms._get_or_create(SLUG)
    state.token_id = "tok-active"   # live order on this token
    state.condition_id = COND
    state.filled_qty = 3.0

    oms.update_filled_from_position(
        SLUG,
        actual_qty=3.0,
        actual_cost_usdc=2.79,
        token_id=TOKEN,   # different token from reconcile data
        condition_id=COND,
    )

    assert state.token_id == "tok-active", (
        "must not overwrite active order token_id with reconcile data"
    )


def test_update_filled_from_position_restores_fine_tick_size() -> None:
    """Tick size must be restored so sell_position uses the correct price granularity."""
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    state = oms._get_or_create(SLUG)
    state.token_id = None
    state.condition_id = None
    state.tick_size = 0.01  # wrong default for a 0.001 market

    oms.update_filled_from_position(
        SLUG,
        actual_qty=5.0,
        actual_cost_usdc=4.65,
        token_id=TOKEN,
        condition_id=COND,
        tick_size=0.001,  # correct tick size from registry
    )

    state = oms.order_state(SLUG)
    assert state.tick_size == 0.001, "tick_size must be restored from registry data"

    # Verify sell uses the correct tick calculation
    oms.sell_position(event_slug=SLUG, now_ns=_NOW_NS, bid_price=0.030)
    sell = adapter.sell_intents()[0]
    assert sell.limit_price_ticks == 30, "0.030 / 0.001 = 30 ticks (not 3)"


def test_stop_loss_works_after_bot_restart_via_position_reconcile() -> None:
    """
    Simulate a bot restart: fresh OMS (token_id=None, filled_qty=0).
    Position reconcile restores both. sell_position must then work correctly.
    This is PATH C: the startup _reconcile_positions call.
    """
    adapter = _RecordingAdapter()
    oms = OrderManagementSystem(adapter, package_id="pkg", order_qty=5)

    # Fresh OMS, no history — position reconcile discovers we hold shares
    oms.update_filled_from_position(
        SLUG,
        actual_qty=5.0,
        actual_cost_usdc=4.65,
        token_id=TOKEN,
        condition_id=COND,
        tick_size=0.01,
    )

    state = oms.order_state(SLUG)
    assert state.token_id == TOKEN
    assert state.filled_qty == 5.0

    # Stop-loss sell must now work
    oms.sell_position(event_slug=SLUG, now_ns=_NOW_NS, bid_price=0.03)
    assert adapter.sell_count() == 1, "sell_position must work after restart + position reconcile"
    sell = adapter.sell_intents()[0]
    assert sell.token_id == TOKEN
    assert sell.limit_price_ticks == 3
