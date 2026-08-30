"""The panel contract + registry. Adding a visualisation is adding ONE file in this package that
calls @panel(...) — the shell discovers it, names nothing explicitly, and needs no edit."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

REGISTRY: list["Panel"] = []


@dataclass
class Panel:
    name: str          # tab / section title
    section: str       # "explore" (needs a market) or "audit" (dataset-wide)
    needs: str         # "market" | "event" | "none"
    order: int         # sort order within its section
    render: Callable   # render(ctx) -> None, draws via streamlit


def panel(name: str, section: str, needs: str = "none", order: int = 100):
    """Decorator: register a render(ctx) function as a dashboard panel."""
    def deco(fn: Callable):
        REGISTRY.append(Panel(name=name, section=section, needs=needs, order=order, render=fn))
        return fn
    return deco


@dataclass
class Ctx:
    """Everything a panel might need, assembled by the shell. A panel uses only what it declares."""
    cid: str | None = None       # selected condition_id (a market)
    market: object = None        # DataFrame: the market's token rows (sorted by outcome_index)
    a_row: object = None         # the outcome_index-0 row
    universe: str | None = None
    neg_risk: bool = False
    closed: bool = False
    event_slug: str | None = None
    start: object = None         # window bounds (UTC ts) for explore panels
    end: object = None
    window: str = "all"
    scale: str = "linear"        # "linear" | "logit"
    rv_win: int = 5              # realised-vol window (minutes)


def panels_for(section: str, has_market: bool) -> list[Panel]:
    out = [p for p in REGISTRY if p.section == section and not (p.needs == "market" and not has_market)]
    return sorted(out, key=lambda p: p.order)
