"""Normalize Polymarket trade-side conventions for Dali research."""
from __future__ import annotations

from typing import Literal


AggressorSide = Literal["BUY", "SELL", "UNKNOWN"]
LiveSideSemantics = Literal["aggressor", "maker", "unknown"]

UNKNOWN: AggressorSide = "UNKNOWN"


def _clean_side(side: str | None) -> AggressorSide:
    raw = (side or "").strip().upper()
    if raw in {"BUY", "SELL"}:
        return raw  # type: ignore[return-value]
    return UNKNOWN


def invert_side(side: str | None) -> AggressorSide:
    clean = _clean_side(side)
    if clean == "BUY":
        return "SELL"
    if clean == "SELL":
        return "BUY"
    return UNKNOWN


def historical_to_aggressor(maker_side: str | None) -> AggressorSide:
    """Convert historical fill ``maker_side`` to token-side aggressor.

    In the local historical fills, ``maker_side`` is the passive maker's token
    side. If the maker was buying the token, the active taker/aggressor sold
    into that bid. If the maker was selling, the aggressor bought from the ask.
    """

    return invert_side(maker_side)


def live_to_aggressor(
    last_trade_price_side: str | None,
    *,
    semantics: LiveSideSemantics = "unknown",
) -> AggressorSide:
    """Convert live ``last_trade_price.side`` to token-side aggressor.

    The default is deliberately conservative. The current live sample has too
    few classifiable trades to prove whether Polymarket reports aggressor side
    or maker side on the market WebSocket. Use ``semantics='aggressor'`` only
    after the sign-convention audit establishes it.
    """

    if semantics == "aggressor":
        return _clean_side(last_trade_price_side)
    if semantics == "maker":
        return invert_side(last_trade_price_side)
    return UNKNOWN
