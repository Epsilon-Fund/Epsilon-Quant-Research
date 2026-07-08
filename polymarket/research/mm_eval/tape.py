"""Causal public-trade tape injection for strategies (Task 5.1).

The frozen :class:`~mm_engine.interfaces.Strategy` protocol receives only
``(book, inventory, params)`` — it cannot see ``last_trade`` events directly, yet a live
quoter subscribed to the market websocket *does* see every public trade print. The VPIN /
sweep-score toxicity lens (Task 5.1's Lens 1) needs exactly that stream.

This module restores the live-observable information **without touching the engine or the
frozen interface**, via the same injection pattern as the Task-5 ``end_date_ms`` τ anchor:

* :class:`TradeTape` — an append-only list of ``(ts_ms, price, size, side)`` tuples.
* :func:`tape_feed` — wraps any engine feed; every ``last_trade`` event is appended to the
  tape **at the moment it is yielded**, i.e. before the engine processes it. When the
  engine then calls ``strategy.quote(...)`` for that event, the tape already contains the
  print — the same instant a live strategy would have seen it. Strictly causal: the tape
  never contains a trade with ``ts_exchange`` beyond the event being processed.

The runner creates one ``TradeTape`` per engine run and passes it through
``params["trade_tape"]`` (params is handed to the strategy by reference; nothing is
serialized). Strategies read it duck-typed (``params.get("trade_tape")``), so ``mm_engine``
gains no dependency on this module.

``side`` is the taker/aggressor side from the capture's ``trades`` table (``BUY`` = buyer
was aggressor). It can be missing on some prints; consumers should fall back to a tick
rule in that case.
"""
from __future__ import annotations

from mm_engine.interfaces import MarketEvent


class TradeTape:
    """Append-only public-trade log: rows of ``(ts_ms, price, size, side)``."""

    __slots__ = ("rows",)

    def __init__(self) -> None:
        self.rows: list[tuple[int, float, float, str | None]] = []

    def append(self, ts_ms: int, price: float, size: float, side: str | None) -> None:
        self.rows.append((ts_ms, price, size, side))

    def __len__(self) -> int:
        return len(self.rows)


def tape_feed(feed, tape: TradeTape):
    """Yield ``feed`` items unchanged, recording each ``last_trade`` onto ``tape`` first.

    GapMarkers and non-trade events pass straight through. Malformed trade payloads
    (missing/None price or size) are NOT recorded — the engine's fill simulator skips
    them the same way, so tape and fills stay consistent.
    """
    for item in feed:
        if isinstance(item, MarketEvent) and item.type == "last_trade":
            p = item.payload.get("price")
            s = item.payload.get("size")
            if p is not None and s is not None:
                try:
                    tape.append(item.ts_exchange, float(p), float(s),
                                item.payload.get("side"))
                except (TypeError, ValueError):
                    pass
        yield item
