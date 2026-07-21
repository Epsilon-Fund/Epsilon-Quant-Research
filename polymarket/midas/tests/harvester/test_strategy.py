from __future__ import annotations

import pytest

from harvester.market_data import BookUpdateEvent
from harvester.registry import TokenRecord, TokenRegistry
from harvester.strategy import (
    StrategyConfig,
    TailHarvesterStrategy,
    floor_to_tick,
)

pytestmark = [pytest.mark.unit, pytest.mark.harvester]

NOW_NS = 1_700_000_000_000_000_000
THRESHOLD = 0.90
SLUG = "london-temp-april-30"
SLUG2 = "london-temp-may-1"
TOKEN = "tok-yes"
TOKEN2 = "tok-yes-2"
COND = "0xabcd"


def _rec(token_id: str = TOKEN, slug: str = SLUG) -> TokenRecord:
    return TokenRecord(token_id=token_id, event_slug=slug, condition_id=COND)


def _reg(*records: TokenRecord) -> TokenRegistry:
    return TokenRegistry(list(records), bid_threshold=THRESHOLD)


def _bid(registry: TokenRegistry, token_id: str, bid: float) -> None:
    registry.update(BookUpdateEvent(token_id=token_id, best_bid=bid, best_ask=1.0 - bid, ts_ns=NOW_NS))


def _strat(registry: TokenRegistry, min_reprice: int = 2) -> TailHarvesterStrategy:
    return TailHarvesterStrategy(registry, StrategyConfig(min_reprice_ticks=min_reprice))


# ---------------------------------------------------------------------------
# floor_to_tick
# ---------------------------------------------------------------------------


def test_floor_to_tick_standard() -> None:
    assert floor_to_tick(0.91) == 91


def test_floor_to_tick_floors_not_rounds() -> None:
    assert floor_to_tick(0.919) == 91


def test_floor_to_tick_clamps_at_99() -> None:
    assert floor_to_tick(1.0) == 99


def test_floor_to_tick_zero() -> None:
    assert floor_to_tick(0.0) == 0


def test_floor_to_tick_exact_boundary() -> None:
    assert floor_to_tick(0.99) == 99


def test_floor_to_tick_tenth_cent_market() -> None:
    assert floor_to_tick(0.942, 0.001) == 942


def test_floor_to_tick_tenth_cent_floors() -> None:
    assert floor_to_tick(0.9429, 0.001) == 942


def test_floor_to_tick_tenth_cent_clamps_at_999() -> None:
    assert floor_to_tick(1.0, 0.001) == 999


# ---------------------------------------------------------------------------
# Bid threshold gate
# ---------------------------------------------------------------------------


def test_evaluate_below_threshold_no_open_order_is_noop() -> None:
    reg = _reg(_rec())
    _bid(reg, TOKEN, 0.85)
    sig = _strat(reg).evaluate(SLUG)
    assert sig.action == "NO_OP"


def test_evaluate_below_threshold_with_open_order_cancels() -> None:
    reg = _reg(_rec())
    _bid(reg, TOKEN, 0.85)
    sig = _strat(reg).evaluate(SLUG, has_open_order=True, current_price_ticks=85)
    assert sig.action == "CANCEL"


def test_evaluate_no_updates_no_open_order_is_noop() -> None:
    reg = _reg(_rec())
    sig = _strat(reg).evaluate(SLUG)
    assert sig.action == "NO_OP"


# ---------------------------------------------------------------------------
# Market-closed gate (accepting_orders poller)
# ---------------------------------------------------------------------------


def test_evaluate_closed_market_no_open_order_is_noop() -> None:
    reg = _reg(_rec())
    _bid(reg, TOKEN, 0.95)
    reg.mark_closed(SLUG)
    sig = _strat(reg).evaluate(SLUG)
    assert sig.action == "NO_OP"


def test_evaluate_closed_market_with_open_order_cancels() -> None:
    reg = _reg(_rec())
    _bid(reg, TOKEN, 0.95)
    reg.mark_closed(SLUG)
    sig = _strat(reg).evaluate(SLUG, has_open_order=True, current_price_ticks=95)
    assert sig.action == "CANCEL"


def test_evaluate_closed_overrides_qualifying_bid() -> None:
    # Even if price is above threshold, closed market must not place
    reg = _reg(_rec())
    _bid(reg, TOKEN, 0.99)
    reg.mark_closed(SLUG)
    sig = _strat(reg).evaluate(SLUG)
    assert sig.action == "NO_OP"


def test_evaluate_closed_one_slug_does_not_affect_another() -> None:
    rec_a = _rec(TOKEN, slug=SLUG)
    rec_b = _rec(TOKEN2, slug=SLUG2)
    reg = _reg(rec_a, rec_b)
    _bid(reg, TOKEN, 0.93)
    _bid(reg, TOKEN2, 0.93)
    reg.mark_closed(SLUG)
    # SLUG is closed → NO_OP; SLUG2 is still open → PLACE
    assert _strat(reg).evaluate(SLUG).action == "NO_OP"
    assert _strat(reg).evaluate(SLUG2).action == "PLACE"


# ---------------------------------------------------------------------------
# PLACE path
# ---------------------------------------------------------------------------


def test_evaluate_places_when_all_conditions_met() -> None:
    reg = _reg(_rec())
    _bid(reg, TOKEN, 0.93)
    sig = _strat(reg).evaluate(SLUG)
    assert sig.action == "PLACE"
    assert sig.token_id == TOKEN
    assert sig.condition_id == COND
    assert sig.price_ticks == 93


def test_evaluate_targets_cheapest_qualified_token() -> None:
    rec_a = _rec(TOKEN, slug=SLUG)
    rec_b = _rec(TOKEN2, slug=SLUG)
    reg = _reg(rec_a, rec_b)
    _bid(reg, TOKEN, 0.95)
    _bid(reg, TOKEN2, 0.91)
    sig = _strat(reg).evaluate(SLUG)
    assert sig.action == "PLACE"
    assert sig.token_id == TOKEN2


def test_evaluate_price_ticks_uses_floor() -> None:
    reg = _reg(_rec())
    _bid(reg, TOKEN, 0.935)
    sig = _strat(reg).evaluate(SLUG)
    assert sig.price_ticks == 93


def test_evaluate_price_ticks_clamped_at_99() -> None:
    reg = _reg(_rec())
    _bid(reg, TOKEN, 1.0)
    sig = _strat(reg).evaluate(SLUG)
    assert sig.price_ticks == 99


# ---------------------------------------------------------------------------
# Reprice suppression
# ---------------------------------------------------------------------------


def test_evaluate_noop_when_price_move_below_min_reprice() -> None:
    reg = _reg(_rec())
    _bid(reg, TOKEN, 0.93)
    sig = _strat(reg, min_reprice=2).evaluate(SLUG, has_open_order=True, current_price_ticks=93)
    assert sig.action == "NO_OP"


def test_evaluate_noop_when_move_is_one_tick() -> None:
    reg = _reg(_rec())
    _bid(reg, TOKEN, 0.94)
    sig = _strat(reg, min_reprice=2).evaluate(SLUG, has_open_order=True, current_price_ticks=93)
    assert sig.action == "NO_OP"


def test_evaluate_places_when_move_meets_min_reprice() -> None:
    reg = _reg(_rec())
    _bid(reg, TOKEN, 0.96)
    sig = _strat(reg, min_reprice=2).evaluate(SLUG, has_open_order=True, current_price_ticks=93)
    assert sig.action == "PLACE"


def test_evaluate_places_when_open_but_current_price_unknown() -> None:
    reg = _reg(_rec())
    _bid(reg, TOKEN, 0.93)
    sig = _strat(reg).evaluate(SLUG, has_open_order=True, current_price_ticks=None)
    assert sig.action == "PLACE"


# ---------------------------------------------------------------------------
# Token selection — always targets the cheapest qualified token (most margin)
# ---------------------------------------------------------------------------


def test_always_switches_to_cheapest_token_even_when_current_price_stable() -> None:
    # TOKEN2@91 has more margin than TOKEN@93 — strategy must switch even though TOKEN price unchanged
    rec_a = _rec(TOKEN, slug=SLUG)
    rec_b = _rec(TOKEN2, slug=SLUG)
    reg = _reg(rec_a, rec_b)
    _bid(reg, TOKEN, 0.93)
    _bid(reg, TOKEN2, 0.91)
    sig = _strat(reg, min_reprice=2).evaluate(
        SLUG, has_open_order=True, current_price_ticks=93, current_token_id=TOKEN
    )
    assert sig.action == "PLACE"
    assert sig.token_id == TOKEN2   # switched to cheapest (most margin)
    assert sig.price_ticks == 91


def test_switches_to_cheapest_token_regardless_of_current_price_move() -> None:
    # TOKEN moved up to 96 but TOKEN2@91 is still cheaper — switch to TOKEN2
    rec_a = _rec(TOKEN, slug=SLUG)
    rec_b = _rec(TOKEN2, slug=SLUG)
    reg = _reg(rec_a, rec_b)
    _bid(reg, TOKEN, 0.96)
    _bid(reg, TOKEN2, 0.91)
    sig = _strat(reg, min_reprice=2).evaluate(
        SLUG, has_open_order=True, current_price_ticks=93, current_token_id=TOKEN
    )
    assert sig.action == "PLACE"
    assert sig.token_id == TOKEN2   # cheapest qualified, not the token we were on
    assert sig.price_ticks == 91


def test_stickiness_switches_when_current_token_drops_below_threshold() -> None:
    rec_a = _rec(TOKEN, slug=SLUG)
    rec_b = _rec(TOKEN2, slug=SLUG)
    reg = _reg(rec_a, rec_b)
    _bid(reg, TOKEN, 0.85)   # fell below threshold
    _bid(reg, TOKEN2, 0.91)
    sig = _strat(reg, min_reprice=2).evaluate(
        SLUG, has_open_order=True, current_price_ticks=93, current_token_id=TOKEN
    )
    assert sig.action == "PLACE"
    assert sig.token_id == TOKEN2  # switched to the only remaining qualified token
