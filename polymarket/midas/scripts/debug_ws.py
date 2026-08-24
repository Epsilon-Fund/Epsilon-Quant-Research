"""
Quick WebSocket diagnostic — connects directly and prints raw messages for 20s.
Run from the midas/ directory:
    python scripts/debug_ws.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(".env", override=False)

from executor.polymarket_discovery import SlugResolutionConfig, fetch_markets_for_slug

import websockets


async def main() -> None:
    raw_slugs = os.getenv("PM_SLUGS", "").strip()
    slugs = [s.strip() for s in raw_slugs.split(",") if s.strip()]
    gamma_url = os.getenv("GAMMA_API_URL", "https://gamma-api.polymarket.com")
    config = SlugResolutionConfig(gamma_api_url=gamma_url)

    token_ids: list[str] = []
    for slug in slugs:
        markets = fetch_markets_for_slug(slug, config)
        for m in markets:
            token_ids.extend(m.token_ids)

    print(f"Subscribing to {len(token_ids)} tokens across {len(slugs)} slugs")
    print(f"First 3 token IDs: {token_ids[:3]}")
    print()

    url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    print(f"Connecting to {url} ...")

    try:
        async with websockets.connect(url, open_timeout=10) as ws:
            print("Connected. Sending subscription...")
            sub = {"assets_ids": token_ids, "type": "Market"}
            await ws.send(json.dumps(sub))
            print(f"Subscription sent. Waiting for messages (20s timeout)...\n")

            count = 0
            async def read_messages():
                nonlocal count
                async for raw in ws:
                    count += 1
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        msg = raw
                    print(f"[msg #{count}] {json.dumps(msg)[:300]}")
                    if count >= 10:
                        break

            try:
                await asyncio.wait_for(read_messages(), timeout=20.0)
            except asyncio.TimeoutError:
                print(f"\nTimeout after 20s. Received {count} messages total.")

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


asyncio.run(main())
