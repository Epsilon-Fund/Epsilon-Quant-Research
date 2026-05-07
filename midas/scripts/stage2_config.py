"""
Stage 2 — Config validation.

Loads HarvesterConfig.from_env() and prints every parsed value so you can
visually confirm the .env was read correctly.  No network calls at all.

Run from the midas/ directory:
    python scripts/stage2_config.py

Pass criteria:
  - No ValueError on load
  - Slugs, credentials, and risk caps show expected values
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harvester.config import HarvesterConfig


def _yn(value: bool) -> str:
    return "YES" if value else "NO"


def _masked(value: str | None) -> str:
    if not value:
        return "NOT SET ← required"
    return f"{value[:4]}{'*' * (len(value) - 4)}"


def main() -> None:
    print("Loading config from .env ...\n")

    try:
        config = HarvesterConfig.from_env()
    except ValueError as e:
        print(f"FAIL — {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FAIL — unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

    print("─" * 60)
    print("  CREDENTIALS")
    print(f"  API key:              {_masked(config.api_key)}")
    print(f"  API secret:           {_masked(config.api_secret)}")
    print(f"  Passphrase:           {_masked(config.passphrase)}")
    print(f"  Private key:          {'SET' if config.private_key else 'NOT SET  (dry-run — no real orders)'}")

    print("\n  MARKETS")
    print(f"  Slugs ({len(config.slugs)}):")
    for slug in config.slugs:
        print(f"    - {slug}")

    print("\n  STRATEGY")
    print(f"  Bid threshold:        {config.bid_threshold:.2f}  ({config.bid_threshold * 100:.0f}¢ minimum)")
    print(f"  Min reprice ticks:    {config.strategy.min_reprice_ticks}¢")

    print("\n  ORDER MANAGEMENT")
    print(f"  Order qty:            {config.oms_order_qty} shares per order")
    print(f"  Package ID:           {config.oms_package_id}")

    print("\n  RISK")
    print(f"  Daily loss cap:       ${config.risk.daily_loss_cap_usdc:.2f} USDC")
    print(f"  Notional cap/event:   ${config.risk.max_notional_per_event_usdc:.2f} USDC")
    print(f"  Auto kill switch:     {_yn(config.risk.enable_auto_kill_switch)}")

    print("\n  EXECUTION")
    print(f"  Poll interval:        {config.execution.poll_interval_s}s")
    print(f"  Shutdown timeout:     {config.execution.shutdown_timeout_s}s")

    print("\n  INFRASTRUCTURE")
    print(f"  CLOB API URL:         {config.adapter.api_url}")
    print(f"  Gamma API URL:        {config.discovery.gamma_api_url}")

    print("\n" + "─" * 60)

    # Warn about anything that looks wrong
    warnings: list[str] = []
    if not config.private_key:
        warnings.append("PM_PRIVATE_KEY not set — orders will not be signed (fine for Stage 3, not for Stage 4)")
    if not config.slugs:
        warnings.append("PM_SLUGS is empty — nothing to trade")
    if config.oms_order_qty > 5:
        warnings.append(f"OMS_ORDER_QTY={config.oms_order_qty} — consider keeping at 1 for canary runs")

    if warnings:
        print("  WARNINGS:")
        for w in warnings:
            print(f"  ⚠  {w}")
        print()

    print("  PASS — config loaded successfully")
    print("  Ready for Stage 3.")


if __name__ == "__main__":
    main()
