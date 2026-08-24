"""
Stage 1 — Discovery validation.

Calls fetch_markets_for_slug for every slug in PM_SLUGS and prints what
Polymarket returns.  No CLOB connection, no WebSocket, no orders — Gamma API
only (public, no auth required).

Run from the midas/ directory:
    python scripts/stage1_discover.py

Pass criteria:
  - Every slug returns at least one market
  - accepting_orders=True
  - end_date_ns is not None
  - token_ids are present
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv(".env", override=False)

# Make sure harvester/ and executor/ are importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executor.polymarket_discovery import SlugResolutionConfig, fetch_markets_for_slug


def main() -> None:
    raw = os.getenv("PM_SLUGS", "").strip()
    if not raw:
        print("ERROR: PM_SLUGS is empty — add at least one slug to .env", file=sys.stderr)
        sys.exit(1)

    slugs = [s.strip() for s in raw.split(",") if s.strip()]
    gamma_url = os.getenv("GAMMA_API_URL", "https://gamma-api.polymarket.com")
    config = SlugResolutionConfig(gamma_api_url=gamma_url)

    failures: list[str] = []

    for slug in slugs:
        print(f"\n{'─' * 60}")
        print(f"  Slug: {slug}")

        try:
            markets = fetch_markets_for_slug(slug, config)
        except Exception as e:
            print(f"  ERROR fetching: {e}")
            failures.append(slug)
            continue

        active = [m for m in markets if m.token_ids]
        if not active:
            print("  FAIL — no active markets returned (wrong slug or already closed)")
            failures.append(slug)
            continue

        for m in active:
            print(f"\n  Question:         {m.question}")
            print(f"  Condition ID:     {m.market_id}")
            print(f"  Token IDs:        {', '.join(m.token_ids)}")

            # Accepting orders
            status = "YES" if m.accepting_orders else "NO ← PROBLEM"
            print(f"  Accepting orders: {status}")
            if not m.accepting_orders:
                failures.append(slug)

            # End date
            if m.end_date_ns is None:
                print("  End date:         None ← PROBLEM (strategy will refuse to trade)")
                failures.append(slug)
            else:
                dt = datetime.fromtimestamp(m.end_date_ns / 1e9, tz=timezone.utc)
                now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
                ttm_h = (m.end_date_ns - now_ns) / 3_600_000_000_000
                print(f"  End date (UTC):   {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print(f"  Time to close:    {ttm_h:.1f}h from now")

            # Prices
            yes = f"{m.yes_price_cents}¢" if m.yes_price_cents is not None else "n/a"
            no  = f"{m.no_price_cents}¢"  if m.no_price_cents  is not None else "n/a"
            print(f"  Current prices:   YES {yes}  /  NO {no}")

    print(f"\n{'═' * 60}")
    if not failures:
        print("  PASS — all slugs returned valid tradable markets")
        print("  Ready for Stage 2.")
    else:
        unique = list(dict.fromkeys(failures))
        print(f"  FAIL — problems with: {unique}")
        print("  Fix the slugs above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
