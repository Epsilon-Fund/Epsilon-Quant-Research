"""Maintained Polymarket CLOB book state and OFI helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class TopOfBook:
    bid_price: float | None
    bid_size: float | None
    ask_price: float | None
    ask_size: float | None


@dataclass(frozen=True)
class OfiContribution:
    bid: float = 0.0
    ask: float = 0.0

    @property
    def combined(self) -> float:
        return self.bid + self.ask


@dataclass(frozen=True)
class MultiLevelOfiContribution:
    bid: tuple[float, ...]
    ask: tuple[float, ...]

    @property
    def combined(self) -> tuple[float, ...]:
        return tuple(bid + ask for bid, ask in zip(self.bid, self.ask, strict=True))


def _clean_levels(levels: Iterable[tuple[float, float]]) -> dict[float, float]:
    out: dict[float, float] = {}
    for price, size in levels:
        if size > 0:
            out[float(price)] = float(size)
    return out


def bid_ofi(
    prev_price: float | None,
    prev_size: float | None,
    new_price: float | None,
    new_size: float | None,
) -> float:
    prev_size = 0.0 if prev_size is None else prev_size
    new_size = 0.0 if new_size is None else new_size
    if prev_price is None and new_price is None:
        return 0.0
    if prev_price is None:
        return new_size
    if new_price is None:
        return -prev_size
    if new_price > prev_price:
        return new_size
    if new_price < prev_price:
        return -prev_size
    return new_size - prev_size


def ask_ofi(
    prev_price: float | None,
    prev_size: float | None,
    new_price: float | None,
    new_size: float | None,
) -> float:
    prev_size = 0.0 if prev_size is None else prev_size
    new_size = 0.0 if new_size is None else new_size
    if prev_price is None and new_price is None:
        return 0.0
    if prev_price is None:
        return -new_size
    if new_price is None:
        return prev_size
    if new_price < prev_price:
        return -new_size
    if new_price > prev_price:
        return prev_size
    return prev_size - new_size


def _level_at(levels: list[tuple[float, float]], idx: int) -> tuple[float | None, float | None]:
    if idx >= len(levels):
        return None, None
    return levels[idx]


def multi_level_ofi(
    prev_bids: list[tuple[float, float]],
    prev_asks: list[tuple[float, float]],
    new_bids: list[tuple[float, float]],
    new_asks: list[tuple[float, float]],
    depth: int,
) -> MultiLevelOfiContribution:
    bid_parts: list[float] = []
    ask_parts: list[float] = []
    for idx in range(depth):
        prev_bid_price, prev_bid_size = _level_at(prev_bids, idx)
        new_bid_price, new_bid_size = _level_at(new_bids, idx)
        prev_ask_price, prev_ask_size = _level_at(prev_asks, idx)
        new_ask_price, new_ask_size = _level_at(new_asks, idx)
        bid_parts.append(bid_ofi(prev_bid_price, prev_bid_size, new_bid_price, new_bid_size))
        ask_parts.append(ask_ofi(prev_ask_price, prev_ask_size, new_ask_price, new_ask_size))
    return MultiLevelOfiContribution(bid=tuple(bid_parts), ask=tuple(ask_parts))


@dataclass
class ClobBook:
    """Price-level book for one CLOB token.

    ``is_complete`` means the state has been anchored by a full ``book`` event.
    ``price_change`` messages before the first full book can be recorded for
    telemetry, but they are not enough to produce reliable depth or OFI.
    """

    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    is_complete: bool = False

    def top(self) -> TopOfBook:
        bid_price = max(self.bids) if self.bids else None
        ask_price = min(self.asks) if self.asks else None
        return TopOfBook(
            bid_price=bid_price,
            bid_size=self.bids.get(bid_price) if bid_price is not None else None,
            ask_price=ask_price,
            ask_size=self.asks.get(ask_price) if ask_price is not None else None,
        )

    def levels(self, side: str) -> list[tuple[float, float]]:
        if side == "bid":
            return sorted(self.bids.items(), key=lambda item: item[0], reverse=True)
        if side == "ask":
            return sorted(self.asks.items(), key=lambda item: item[0])
        raise ValueError(f"unknown side: {side}")

    def multi_level_replace(
        self,
        bids: Iterable[tuple[float, float]],
        asks: Iterable[tuple[float, float]],
        depth: int = 10,
    ) -> MultiLevelOfiContribution:
        prev_bids = self.levels("bid")[:depth]
        prev_asks = self.levels("ask")[:depth]
        had_anchor = self.is_complete
        self.bids = _clean_levels(bids)
        self.asks = _clean_levels(asks)
        self.is_complete = True
        if not had_anchor:
            zeros = tuple(0.0 for _ in range(depth))
            return MultiLevelOfiContribution(bid=zeros, ask=zeros)
        return multi_level_ofi(
            prev_bids,
            prev_asks,
            self.levels("bid")[:depth],
            self.levels("ask")[:depth],
            depth,
        )

    def multi_level_update_level(
        self,
        side: str,
        price: float,
        size: float,
        depth: int = 10,
    ) -> MultiLevelOfiContribution:
        prev_bids = self.levels("bid")[:depth]
        prev_asks = self.levels("ask")[:depth]
        levels = self.bids if side == "BUY" else self.asks if side == "SELL" else None
        if levels is None:
            raise ValueError(f"unknown price_change side: {side}")
        if size <= 0:
            levels.pop(float(price), None)
        else:
            levels[float(price)] = float(size)

        if not self.is_complete:
            zeros = tuple(0.0 for _ in range(depth))
            return MultiLevelOfiContribution(bid=zeros, ask=zeros)
        return multi_level_ofi(
            prev_bids,
            prev_asks,
            self.levels("bid")[:depth],
            self.levels("ask")[:depth],
            depth,
        )

    def replace(
        self,
        bids: Iterable[tuple[float, float]],
        asks: Iterable[tuple[float, float]],
    ) -> OfiContribution:
        prev = self.top()
        had_anchor = self.is_complete
        self.bids = _clean_levels(bids)
        self.asks = _clean_levels(asks)
        self.is_complete = True
        if not had_anchor:
            return OfiContribution()
        new = self.top()
        return OfiContribution(
            bid=bid_ofi(prev.bid_price, prev.bid_size, new.bid_price, new.bid_size),
            ask=ask_ofi(prev.ask_price, prev.ask_size, new.ask_price, new.ask_size),
        )

    def update_level(self, side: str, price: float, size: float) -> OfiContribution:
        prev = self.top()
        levels = self.bids if side == "BUY" else self.asks if side == "SELL" else None
        if levels is None:
            raise ValueError(f"unknown price_change side: {side}")
        if size <= 0:
            levels.pop(float(price), None)
        else:
            levels[float(price)] = float(size)

        if not self.is_complete:
            return OfiContribution()
        new = self.top()
        return OfiContribution(
            bid=bid_ofi(prev.bid_price, prev.bid_size, new.bid_price, new.bid_size),
            ask=ask_ofi(prev.ask_price, prev.ask_size, new.ask_price, new.ask_size),
        )

    def mid(self) -> float | None:
        top = self.top()
        if top.bid_price is None or top.ask_price is None:
            return None
        return (top.bid_price + top.ask_price) / 2

    def spread(self) -> float | None:
        top = self.top()
        if top.bid_price is None or top.ask_price is None:
            return None
        return top.ask_price - top.bid_price

    def walk(self, side: str, size: float) -> tuple[float | None, float, float]:
        """Return average executable price, filled size, and notional.

        ``side`` is the taker's action: BUY walks asks; SELL walks bids.
        """
        if size <= 0:
            return None, 0.0, 0.0
        levels = self.levels("ask" if side == "BUY" else "bid")
        remaining = float(size)
        filled = 0.0
        notional = 0.0
        for price, available in levels:
            take = min(remaining, available)
            if take <= 0:
                continue
            filled += take
            notional += take * price
            remaining -= take
            if remaining <= 1e-12:
                break
        if filled <= 0:
            return None, 0.0, 0.0
        return notional / filled, filled, notional
