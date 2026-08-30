"""Where the data lives. Overridable by the EPSILON_DATA_ROOT environment variable so the
same code reads a local `research_v1/` today and an R2-mirrored copy later, with no edits."""
from __future__ import annotations
import os
from pathlib import Path

__version__ = "1.0"

# Default: the local built library, two levels up from this file (polymarket/research/data/research_v1).
_DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent / "data" / "research_v1")


def data_root() -> str:
    """The data root as a STRING — a local directory OR an `s3://…` bucket path. Returned as a
    string (not Path) because Path() mangles `s3://` on Windows (→ `s3:\\`). Override with the
    EPSILON_DATA_ROOT env var. Nothing here hardcodes an absolute path into a caller."""
    return os.environ.get("EPSILON_DATA_ROOT", _DEFAULT_ROOT)
