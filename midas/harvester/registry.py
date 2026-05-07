from __future__ import annotations

from dataclasses import dataclass, field

from .market_data import BookUpdateEvent


# ---------------------------------------------------------------------------
# Static token metadata — loaded once at startup from polymarket_discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenRecord:
    token_id: str
    event_slug: str
    condition_id: str
    end_date_ns: int | None = None  # resolution timestamp; None means unknown
    tick_size: float = 0.01         # minimum price increment (0.01 or 0.001)
    is_yes: bool = True             # True = YES token, False = NO token


# ---------------------------------------------------------------------------
# Live token state — updated on every BookUpdateEvent
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TokenState:
    record: TokenRecord
    best_bid: float = 0.0
    best_ask: float = 1.0
    last_updated_ns: int = 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TokenRegistry:
    """In-memory view of current best bid/ask for every watched token.

    Populated with static metadata at startup, then kept live via update().
    The strategy calls qualified() on every price tick to find actionable tokens.
    mark_closed() is called by the accepting_orders poller when a market stops
    accepting new orders — the strategy then cancels resting orders for that slug.
    """

    __slots__ = ("_states", "_threshold", "_closed_slugs")

    def __init__(self, records: list[TokenRecord], bid_threshold: float) -> None:
        if not 0.0 < bid_threshold <= 1.0:
            raise ValueError(f"bid_threshold must be in (0, 1], got {bid_threshold}")
        self._threshold = bid_threshold
        self._states: dict[str, TokenState] = {
            r.token_id: TokenState(record=r) for r in records
        }
        self._closed_slugs: set[str] = set()

    # ------------------------------------------------------------------
    # Write path — called by the main loop on every BookUpdateEvent
    # ------------------------------------------------------------------

    def update(self, event: BookUpdateEvent) -> None:
        state = self._states.get(event.token_id)
        if state is None:
            return  # token not registered — ignore
        state.best_bid = event.best_bid
        state.best_ask = event.best_ask
        state.last_updated_ns = event.ts_ns

    def add_records(self, records: list[TokenRecord]) -> None:
        """Register new token records at runtime (e.g. when rolling to the next day).

        Existing records are never overwritten. Returns silently for duplicates.
        """
        for r in records:
            if r.token_id not in self._states:
                self._states[r.token_id] = TokenState(record=r)

    # ------------------------------------------------------------------
    # Read paths — called by strategy and OMS
    # ------------------------------------------------------------------

    def qualified(self) -> tuple[TokenState, ...]:
        """Tokens eligible for bidding, sorted cheapest first (most spread to capture).

        Entry gate: a slug only qualifies if at least one of its YES tokens has
        best_bid >= threshold. This confirms the market has a near-certain outcome.

        Token selection: once the gate is open, ALL tokens in that slug (YES or NO)
        with best_bid >= threshold are returned. A NO token at 90¢ is equally valid —
        it resolves to $1 if the NO outcome is correct, same spread mechanics as YES.
        """
        qualifying_slugs = {
            s.record.event_slug
            for s in self._states.values()
            if s.record.is_yes and s.best_bid >= self._threshold
        }
        return tuple(
            sorted(
                (
                    s for s in self._states.values()
                    if s.record.event_slug in qualifying_slugs
                    and s.best_bid >= self._threshold
                ),
                key=lambda s: s.best_bid,
            )
        )

    def state_for(self, token_id: str) -> TokenState | None:
        return self._states.get(token_id)

    def all_token_ids(self) -> tuple[str, ...]:
        """All registered token IDs — used to build the WS subscription list."""
        return tuple(self._states.keys())

    def token_ids_for_event(self, event_slug: str) -> tuple[str, ...]:
        """All token IDs belonging to a specific event."""
        return tuple(
            s.record.token_id
            for s in self._states.values()
            if s.record.event_slug == event_slug
        )

    # ------------------------------------------------------------------
    # Market-close tracking — driven by the accepting_orders poller
    # ------------------------------------------------------------------

    def mark_closed(self, event_slug: str) -> None:
        """Signal that a market is no longer accepting orders."""
        self._closed_slugs.add(event_slug)

    def is_closed(self, event_slug: str) -> bool:
        return event_slug in self._closed_slugs
