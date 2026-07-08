"""Module entry point so `python -m polymarket.execution` works.

Modes:
    --mode copytrade   (default) — leader copy-trading bot (cli.main)
    --mode maker       — politics NegRisk passive maker loop (maker.cli.main)
    --mode mm_bridge   — MM-engine → maker live bridge, DRY-RUN (maker.mm_bridge_cli.main)
    --mode mm_latency  — Join-2b latency probe harness, DRY-RUN (maker.mm_latency_harness.main)
    --mode mm_calibrate — Join-2d queue/latency calibration on recorded logs (maker.mm_calibration.main)

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
    if mode in ("mm_bridge", "bridge"):
        from polymarket.execution.maker.mm_bridge_cli import main as bridge_main

        return bridge_main()
    if mode in ("mm_latency", "latency"):
        from polymarket.execution.maker.mm_latency_harness import main as latency_main

        return latency_main()
    if mode in ("mm_calibrate", "calibrate"):
        from polymarket.execution.maker.mm_calibration import main as calibrate_main

        return calibrate_main()
    if mode in ("maker", "mm"):
        from polymarket.execution.maker.cli import main as maker_main

        return maker_main(check_auth=check_auth)
    if mode in ("copytrade", "copy", "mirror"):
        from polymarket.execution.cli import main as copytrade_main

        return copytrade_main()
    print(
        f"[startup] Unknown --mode {mode!r} "
        "(expected 'copytrade', 'maker', 'mm_bridge', 'mm_latency', or 'mm_calibrate')",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
