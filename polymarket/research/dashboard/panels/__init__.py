"""Panel registry package. Importing it auto-discovers every panel module (any file not starting
with '_') so each self-registers via @panel(...). The shell imports `panels_for` and never names a
panel — so adding a panel is adding a file here, with zero edits elsewhere."""
from __future__ import annotations
import importlib
import pkgutil

from ._base import Ctx, Panel, panel, panels_for, REGISTRY

__all__ = ["Ctx", "Panel", "panel", "panels_for", "REGISTRY"]


def _discover():
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")


_discover()
