"""Where the data lives. Overridable by the EPSILON_DATA_ROOT environment variable so the
same code reads a local `research_v1/` today and an R2-mirrored copy later, with no edits.

Importing this module loads `polymarket/research/.env` if one exists, because every doc tells
you to put EPSILON_DATA_ROOT and the R2 credentials there. **Variables already set in the shell
always win** — the file only fills in what is missing, so an explicit `export` still overrides it.
Values are never printed.
"""
from __future__ import annotations
import os
from pathlib import Path

__version__ = "1.0"

# The project root (polymarket/research) — this file is at <project>/epsilon_data/config.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default: the local built library, two levels up from this file (polymarket/research/data/research_v1).
_DEFAULT_ROOT = str(_PROJECT_ROOT / "data" / "research_v1")

# Where a .env is looked for. The project root only — this is not a search up the filesystem.
ENV_FILE = _PROJECT_ROOT / ".env"


def load_env(path: str | os.PathLike | None = None) -> bool:
    """Load `polymarket/research/.env` into os.environ WITHOUT overriding anything already set.

    Returns True if a file was found and read. Uses python-dotenv (shipped in requirements.txt)
    when available and falls back to a minimal KEY=VALUE parser so a missing optional dependency
    can never be the reason the credentials don't load. Never prints or logs a value.
    """
    p = Path(path) if path is not None else ENV_FILE
    if not p.is_file():
        return False
    try:
        from dotenv import load_dotenv
        load_dotenv(p, override=False)          # shell env wins
        return True
    except ImportError:
        pass
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            os.environ.setdefault(key, val)     # shell env wins
    return True


load_env()


def data_root() -> str:
    """The data root as a STRING — a local directory OR an `s3://…` bucket path. Returned as a
    string (not Path) because Path() mangles `s3://` on Windows (→ `s3:\\`). Override with the
    EPSILON_DATA_ROOT env var (shell first, then `polymarket/research/.env`). Nothing here
    hardcodes an absolute path into a caller."""
    return os.environ.get("EPSILON_DATA_ROOT", _DEFAULT_ROOT)
