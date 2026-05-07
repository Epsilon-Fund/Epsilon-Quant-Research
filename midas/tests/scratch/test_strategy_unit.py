from __future__ import annotations

import pytest

from harvester.market_data import BookUpdateEvent
from harvester.registry import TokenRecord, TokenRegistry
from harvester.strategy import StrategyConfig, TailHarvesterStrategy, floor_to_tick

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

NOW_NS = 1_700_000_000_000_000_000
ONE_HOUR_NS = 3_600_000_000_000
THRESHOLD = 0.90
SLUG = "london-temp-april-30"
TOKEN = "tok-yes"
COND = "0xabcd"
WITHIN_END_NS = NOW_NS + ONE_HOUR_NS  # 1h away — inside a 2h window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    token_id: str = TOKEN,
    slug: str = SLUG,
    cond: str = COND,
    end_date_ns: int | None = WITHIN_END_NS,
) -> TokenRecord:
    return TokenRecord(token_id=token_id, event_slug=slug, condition_id=cond, end_date_ns=end_date_ns)


def _registry(*records: TokenRecord) -> TokenRegistry:
    return TokenRegistry(list(records), bid_threshold=THRESHOLD)


def _bid(registry: TokenRegistry, token_id: str, bid: float) -> None:
    registry.update(BookUpdateEvent(token_id=token_id, best_bid=bid, best_ask=1.0 - bid, ts_ns=NOW_NS))


def _strategy(
    registry: TokenRegistry,
    max_hours: float = 2.0,
    min_reprice: int = 2,
) -> TailHarvesterStrategy:
    return TailHarvesterStrategy(
        registry,
        StrategyConfig(max_hours_before_close=max_hours, min_reprice_ticks=min_reprice),
    )


# ---------------------------------------------------------------------------
# floor_to_tick
# ---------------------------------------------------------------------------


def test_floor_to_tick_standard() -> None:
    assert floor_to_tick(0.91) == 91


def test_floor_to_tick_floors_not_rounds() -> None:
    # 0.919 in IEEE 754 → 91.8999... → floor = 91, not 92
    assert floor_to_tick(0.919) == 91
    assert floor_to_tick(0.915) == 91


def test_floor_to_tick_clamps_at_99() -> None:
    assert floor_to_tick(1.0) == 99
    assert floor_to_tick(0.999) == 99


def test_floor_to_tick_zero() -> None:
    assert floor_to_tick(0.0) == 0


# ---------------------------------------------------------------------------
# evaluate — bid threshold gate
# ---------------------------------------------------------------------------


def test_evaluate_below_threshold_no_open_order_is_noop() -> None:
    reg = _registry(_record())
    _bid(reg, TOKEN, 0.85)
    sig = _strategy(reg).evaluate(SLUG, NOW_NS)
    assert sig.action == "NO_OP"


def test_evaluate_below_threshold_with_open_order_cancels() -> None:
    reg = _registry(_record())
    _bid(reg, TOKEN, 0.85)
    sig = _strategy(reg).evaluate(SLUG, NOW_NS, has_open_order=True)
    assert sig.action == "CANCEL"


# ---------------------------------------------------------------------------
# evaluate — time window gate
# ---------------------------------------------------------------------------


def test_evaluate_outside_time_window_cancels() -> None:
    far_end = NOW_NS + 5 * ONE_HOUR_NS  # 5h away, max_hours=2
    reg = _registry(_record(end_date_ns=far_end))
    _bid(reg, TOKEN, 0.93)
    sig = _strategy(reg, max_hours=2.0).evaluate(SLUG, NOW_NS)
    assert sig.action == "CANCEL"


def test_evaluate_expired_market_cancels() -> None:
    reg = _registry(_record(end_date_ns=NOW_NS - ONE_HOUR_NS))
    _bid(reg, TOKEN, 0.93)
    sig = _strategy(reg).evaluate(SLUG, NOW_NS)
    assert sig.action == "CANCEL"


def test_evaluate_unknown_end_date_refuses_to_trade() -> None:
    reg = _registry(_record(end_date_ns=None))
    _bid(reg, TOKEN, 0.93)
    sig = _strategy(reg).evaluate(SLUG, NOW_NS)
    assert sig.action == "CANCEL"


# ---------------------------------------------------------------------------
# evaluate — PLACE path
# ---------------------------------------------------------------------------


def test_evaluate_places_when_all_conditions_met() -> None:
    reg = _registry(_record())
    _bid(reg, TOKEN, 0.93)
    sig = _strategy(reg).evaluate(SLUG, NOW_NS)
    assert sig.action == "PLACE"
    assert sig.token_id == TOKEN
    assert sig.condition_id == COND
    assert sig.price_ticks == 93


def test_evaluate_targets_cheapest_qualified_token() -> None:
    tok_expensive = _record("tok-expensive")
    tok_cheap = _record("tok-cheap")
    reg = _registry(tok_expensive, tok_cheap)
    _bid(reg, "tok-expensive", 0.95)
    _bid(reg, "tok-cheap", 0.91)
    sig = _strategy(reg).evaluate(SLUG, NOW_NS)
    assert sig.action == "PLACE"
    assert sig.token_id == "tok-cheap"
    assert sig.price_ticks == 91


def test_evaluate_price_ticks_uses_floor() -> None:
    reg = _registry(_record())
    _bid(reg, TOKEN, 0.935)  # floor(93.5) = 93, not 94
    sig = _strategy(reg).evaluate(SLUG, NOW_NS)
    assert sig.price_ticks == 93


def test_evaluate_price_ticks_clamps_at_99_when_bid_is_one() -> None:
    reg = _registry(_record())
    _bid(reg, TOKEN, 1.0)
    sig = _strategy(reg).evaluate(SLUG, NOW_NS)
    assert sig.action == "PLACE"
    assert sig.price_ticks == 99


# ---------------------------------------------------------------------------
# evaluate — reprice suppression
# ---------------------------------------------------------------------------


def test_evaluate_noop_when_price_move_below_reprice_threshold() -> None:
    # move of 1¢ < min_reprice_ticks=2 → suppress cancel/replace
    reg = _registry(_record())
    _bid(reg, TOKEN, 0.92)
    sig = _strategy(reg, min_reprice=2).evaluate(
        SLUG, NOW_NS, has_open_order=True, current_price_ticks=93
    )
    assert sig.action == "NO_OP"


def test_evaluate_reprices_when_move_meets_threshold() -> None:
    # move of 3¢ >= min_reprice_ticks=2 → allow
    reg = _registry(_record())
    _bid(reg, TOKEN, 0.96)
    sig = _strategy(reg, min_reprice=2).evaluate(
        SLUG, NOW_NS, has_open_order=True, current_price_ticks=93
    )
    assert sig.action == "PLACE"
    assert sig.price_ticks == 96


def test_evaluate_places_when_open_but_current_price_unknown() -> None:
    # has_open_order=True but no current price known → treat as fresh placement
    reg = _registry(_record())
    _bid(reg, TOKEN, 0.93)
    sig = _strategy(reg).evaluate(
        SLUG, NOW_NS, has_open_order=True, current_price_ticks=None
    )
    assert sig.action == "PLACE"


# ---------------------------------------------------------------------------
# evaluate — earliest end_date_ns across sub-markets
# ---------------------------------------------------------------------------


def test_evaluate_uses_earliest_end_date_within_window() -> None:
    # tok-a closes in 1h (inside 2h window), tok-b closes in 5h (outside)
    # min(1h, 5h) = 1h → within window → PLACE
    tok_a = _record("tok-a", end_date_ns=NOW_NS + 1 * ONE_HOUR_NS)
    tok_b = _record("tok-b", end_date_ns=NOW_NS + 5 * ONE_HOUR_NS)
    reg = _registry(tok_a, tok_b)
    _bid(reg, "tok-a", 0.93)
    _bid(reg, "tok-b", 0.91)
    sig = _strategy(reg, max_hours=2.0).evaluate(SLUG, NOW_NS)
    assert sig.action == "PLACE"


def test_evaluate_cancels_when_any_sub_market_has_no_end_date() -> None:
    tok_a = _record("tok-a", end_date_ns=NOW_NS + 1 * ONE_HOUR_NS)
    tok_b = _record("tok-b", end_date_ns=None)
    reg = _registry(tok_a, tok_b)
    _bid(reg, "tok-a", 0.93)
    _bid(reg, "tok-b", 0.91)
    sig = _strategy(reg, max_hours=2.0).evaluate(SLUG, NOW_NS)
    assert sig.action == "CANCEL"
