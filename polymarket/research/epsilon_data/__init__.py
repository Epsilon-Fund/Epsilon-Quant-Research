"""epsilon_data — the public loader for the Polymarket research library (research/v1).

Everything a dashboard, notebook or analysis should use lives here. No side effects on import,
no printing. See README.md for the full manual, tables, units and traps.

    import epsilon_data as ed
    ed.search("fed")                       # find a market
    df = ed.load_l1("politics/…/yes")      # its L1 tape
"""
from __future__ import annotations

from .config import __version__, data_root
from .catalog import (
    catalog, events, search, resolve,
    coverage, reconciliation, activity_by_time, negrisk_sum,
)
from .tape import load_l1, load_trades, load_pair, load_event, markout

__all__ = [
    "catalog", "events", "search", "resolve",
    "load_l1", "load_trades", "load_pair", "load_event", "markout",
    "coverage", "reconciliation", "activity_by_time", "negrisk_sum",
    "data_root", "__version__",
]
