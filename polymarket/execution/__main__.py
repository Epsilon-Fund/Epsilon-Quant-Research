"""Module entry point so `python -m polymarket.execution` works."""
from __future__ import annotations

import sys

from polymarket.execution.cli import main

if __name__ == "__main__":
    sys.exit(main())
