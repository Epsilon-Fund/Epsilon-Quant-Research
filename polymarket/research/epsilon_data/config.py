"""Where the data lives. Overridable by the EPSILON_DATA_ROOT environment variable so the
same code reads a local `research_v1/` today and an R2-mirrored copy later, with no edits."""
from __future__ import annotations
import os
from pathlib import Path

__version__ = "1.0"

# Default: the local built library, two levels up from this file (polymarket/research/data/research_v1).
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "research_v1"


def data_root() -> Path:
    """Directory holding tokens.parquet, l1/, trades/, obs_stats.parquet, exclusions.csv.

    Override with the EPSILON_DATA_ROOT env var (or pass root= to the loader functions that
    accept it). Nothing here hardcodes an absolute path into a caller.
    """
    return Path(os.environ.get("EPSILON_DATA_ROOT", str(_DEFAULT_ROOT)))
