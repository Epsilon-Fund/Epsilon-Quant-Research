"""Module entry point so `python -m polymarket.execution` works.

Modes:
    --mode copytrade   (default) — leader copy-trading bot (cli.main)
    --mode maker       — politics NegRisk passive maker loop (maker.cli.main)

The mode flag is parsed here and stripped before delegating, so each
mode's main() still sees a clean environment-driven config.
"""
from __future__ import annotations

import sys


def _select_mode(argv: list[str]) -> tuple[str, list[str]]:
    """Extract --mode VALUE (or --mode=VALUE) from argv; default copytrade."""
    mode = "copytrade"
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--mode":
            if i + 1 < len(argv):
                mode = argv[i + 1]
                i += 2
                continue
            i += 1
            continue
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
            i += 1
            continue
        remaining.append(arg)
        i += 1
    return mode, remaining


def main() -> int:
    mode, rest = _select_mode(sys.argv[1:])
    check_auth = "--check-auth" in rest
    if mode in ("maker", "mm"):
        from polymarket.execution.maker.cli import main as maker_main

        return maker_main(check_auth=check_auth)
    if mode in ("copytrade", "copy", "mirror"):
        from polymarket.execution.cli import main as copytrade_main

        return copytrade_main()
    print(
        f"[startup] Unknown --mode {mode!r} (expected 'copytrade' or 'maker')",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
