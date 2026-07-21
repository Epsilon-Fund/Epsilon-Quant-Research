from __future__ import annotations

import pytest

from harvester.market_data import BookUpdateEvent
from harvester.registry import TokenRecord, TokenRegistry

pytestmark = [pytest.mark.unit, pytest.mark.harvester]

SLUG = "london-temp-april-30"
SLUG2 = "london-temp-may-1"
TOKEN_A = "tok-yes"
TOKEN_B = "tok-no"
TOKEN_C = "tok-yes-2"
THRESHOLD = 0.90
NOW_NS = 1_700_000_000_000_000_000


def _rec(token_id: str, slug: str = SLUG, end_ns: int | None = None) -> TokenRecord:
    return TokenRecord(token_id=token_id, event_slug=slug, condition_id="0xabcd", end_date_ns=end_ns)


def _reg(*records: TokenRecord) -> TokenRegistry:
    return TokenRegistry(list(records), bid_threshold=THRESHOLD)


def _update(reg: TokenRegistry, token_id: str, bid: float) -> None:
    reg.update(BookUpdateEvent(token_id=token_id, best_bid=bid, best_ask=1.0 - bid, ts_ns=NOW_NS))


# ---------------------------------------------------------------------------
# qualified()
# ---------------------------------------------------------------------------


def test_qualified_above_threshold() -> None:
    reg = _reg(_rec(TOKEN_A))
    _update(reg, TOKEN_A, 0.93)
    assert len(reg.qualified()) == 1


def test_qualified_exactly_at_threshold() -> None:
    reg = _reg(_rec(TOKEN_A))
    _update(reg, TOKEN_A, THRESHOLD)
    assert len(reg.qualified()) == 1


def test_qualified_below_threshold() -> None:
    reg = _reg(_rec(TOKEN_A))
    _update(reg, TOKEN_A, 0.89)
    assert reg.qualified() == ()


def test_qualified_no_updates_is_empty() -> None:
    reg = _reg(_rec(TOKEN_A))
    assert reg.qualified() == ()


def test_qualified_sorted_cheapest_first() -> None:
    reg = _reg(_rec(TOKEN_A), _rec(TOKEN_B))
    _update(reg, TOKEN_A, 0.95)
    _update(reg, TOKEN_B, 0.91)
    q = reg.qualified()
    assert q[0].record.token_id == TOKEN_B
    assert q[1].record.token_id == TOKEN_A


def test_qualified_only_above_threshold_tokens_included() -> None:
    reg = _reg(_rec(TOKEN_A), _rec(TOKEN_B))
    _update(reg, TOKEN_A, 0.93)
    _update(reg, TOKEN_B, 0.80)
    q = reg.qualified()
    assert len(q) == 1
    assert q[0].record.token_id == TOKEN_A


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


def test_update_sets_best_bid() -> None:
    reg = _reg(_rec(TOKEN_A))
    _update(reg, TOKEN_A, 0.93)
    assert reg.state_for(TOKEN_A).best_bid == 0.93


def test_update_overwrites_prior_bid() -> None:
    reg = _reg(_rec(TOKEN_A))
    _update(reg, TOKEN_A, 0.85)
    _update(reg, TOKEN_A, 0.93)
    assert reg.state_for(TOKEN_A).best_bid == 0.93


def test_update_unknown_token_is_silently_ignored() -> None:
    reg = _reg(_rec(TOKEN_A))
    reg.update(BookUpdateEvent(token_id="unknown", best_bid=0.95, best_ask=0.05, ts_ns=NOW_NS))
    assert reg.state_for("unknown") is None


# ---------------------------------------------------------------------------
# all_token_ids() / token_ids_for_event()
# ---------------------------------------------------------------------------


def test_all_token_ids_returns_all_registered() -> None:
    reg = _reg(_rec(TOKEN_A), _rec(TOKEN_B))
    assert set(reg.all_token_ids()) == {TOKEN_A, TOKEN_B}


def test_token_ids_for_event_groups_by_slug() -> None:
    reg = _reg(_rec(TOKEN_A, SLUG), _rec(TOKEN_B, SLUG), _rec(TOKEN_C, SLUG2))
    assert set(reg.token_ids_for_event(SLUG)) == {TOKEN_A, TOKEN_B}
    assert set(reg.token_ids_for_event(SLUG2)) == {TOKEN_C}


def test_token_ids_for_event_unknown_slug_returns_empty() -> None:
    reg = _reg(_rec(TOKEN_A))
    assert reg.token_ids_for_event("nonexistent") == ()


# ---------------------------------------------------------------------------
# state_for()
# ---------------------------------------------------------------------------


def test_state_for_returns_record() -> None:
    reg = _reg(_rec(TOKEN_A))
    state = reg.state_for(TOKEN_A)
    assert state is not None
    assert state.record.token_id == TOKEN_A


def test_state_for_unknown_returns_none() -> None:
    reg = _reg(_rec(TOKEN_A))
    assert reg.state_for("unknown") is None
