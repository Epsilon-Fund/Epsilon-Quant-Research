"""
cli.py — command-line entry for the calibration scoring layer (epsilon shim).

SHIM since 2026-07-05: parsing/formatting live in lemma-calibrate
(`library/calibrate/`); this wrapper injects the epsilon book -> ledger-path
mapping so the historical invocation lines keep working unchanged:

  crypto:      PYTHONPATH=. uv run python -m infrastructure.calibration.cli --book crypto score
  polymarket:  cd polymarket/research && PYTHONPATH=. uv run python -m lib.calibration.cli --book polymarket score

The book selects which forked-ledger to read ($SF_BOOK / $SF_LEDGER_DIR also
work). This file is byte-identical across projects.
"""

from __future__ import annotations

import sys

from lemma.calibrate.cli import main as _main

from .core import _books


def main(argv: list[str] | None = None) -> int:
    return _main(argv, books=_books())


if __name__ == "__main__":
    sys.exit(main())
