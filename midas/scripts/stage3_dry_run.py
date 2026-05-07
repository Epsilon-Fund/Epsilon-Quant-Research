"""
Stage 3 — Dry-run validation.

Runs the full harvester stack against live Polymarket data but with
FakeVenueAdapter — no orders reach the exchange.  WebSocket, strategy
signals, OMS, risk gate, and the accepting_orders poller all run exactly
as they would in production.

Run from the midas/ directory:
    python scripts/stage3_dry_run.py

Stop with Ctrl+C when satisfied.

Pass criteria:
  - "dry_run_active" warning appears before startup (confirms FakeVenueAdapter)
  - "startup_complete" log appears with dry_run=true
  - Book update events flow in (confirms WebSocket is live)
  - "signal.place" logs appear when a YES token bid crosses the threshold
  - "signal.cancel" logs appear as prices move
  - No errors or unhandled exceptions
  - Clean "shutdown_complete" on Ctrl+C

What to watch for:
  - signal.place  → strategy saw a qualifying bid, fake order placed
  - signal.cancel → bid dropped below threshold or market marked closed
  - market_closed → accepting_orders poller detected market closure
  - risk.blocked  → risk gate fired (notional cap or kill switch)
"""

from __future__ import annotations

import os
import sys

# Force DRY_RUN before anything else is imported so config sees it
os.environ["DRY_RUN"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("  Stage 3 — Dry-run (FakeVenueAdapter, no real orders)")
print("  Press Ctrl+C to stop cleanly.")
print("=" * 60)
print()

from harvester.main import main  # noqa: E402

main()
