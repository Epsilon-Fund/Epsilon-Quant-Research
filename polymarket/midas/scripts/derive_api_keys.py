"""
One-time script to derive Polymarket CLOB API credentials from your private key.

Run:
    PM_PRIVATE_KEY=0x... python scripts/derive_api_keys.py

Or run without the env var and it will prompt you.

The three values it prints go into your .env file as:
    PM_API_KEY=...
    PM_API_SECRET=...
    PM_PASSPHRASE=...
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    private_key = os.getenv("PM_PRIVATE_KEY", "").strip()
    if not private_key:
        private_key = input("Private key (0x...): ").strip()
    if not private_key:
        print("Error: private key is required", file=sys.stderr)
        sys.exit(1)

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
    except ImportError:
        print("Error: py-clob-client not installed.", file=sys.stderr)
        print("Run: pip install py-clob-client", file=sys.stderr)
        sys.exit(1)

    host = "https://clob.polymarket.com"
    print(f"Connecting to {host} ...", file=sys.stderr)

    client = ClobClient(host=host, key=private_key, chain_id=POLYGON)

    try:
        creds = client.derive_api_key()
        source = "derive_api_key"
    except Exception as e:
        print(f"derive_api_key failed ({e}), trying create_api_key ...", file=sys.stderr)
        try:
            creds = client.create_api_key()
            source = "create_api_key"
        except Exception as e2:
            print(f"create_api_key also failed: {e2}", file=sys.stderr)
            sys.exit(1)

    # Support both attribute naming conventions across py-clob-client versions
    api_key = getattr(creds, "api_key", None) or getattr(creds, "key", None)
    api_secret = getattr(creds, "api_secret", None) or getattr(creds, "secret", None)
    passphrase = (
        getattr(creds, "api_passphrase", None)
        or getattr(creds, "passphrase", None)
    )

    if not all([api_key, api_secret, passphrase]):
        print(f"Unexpected response shape: {creds}", file=sys.stderr)
        sys.exit(1)

    print(f"\n# Derived via {source} — paste into your .env", file=sys.stderr)
    print(f"PM_API_KEY={api_key}")
    print(f"PM_API_SECRET={api_secret}")
    print(f"PM_PASSPHRASE={passphrase}")


if __name__ == "__main__":
    main()
