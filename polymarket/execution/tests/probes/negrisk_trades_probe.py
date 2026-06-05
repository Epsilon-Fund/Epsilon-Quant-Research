"""Probe RTDS for trades on a specific NegRisk event.

Read-only. No auth. No order submission. Runs ~95 s and exits.

Target event: elc-sot-mid-2026-05-12 (active NegRisk event with
recent on-window activity).
  - negRisk=True. 3 sub-markets (esports series outcome).
  - ~$2.2M 24h volume at probe time, observed live trades on the
    first probe sweep.
Earlier candidate `who-will-be-confirmed-as-fed-chair` had $4M
24h volume but was idle for the 90-s window; switched after
empirical confirmation that this slug is actively trading.

Strategy: subscribe to RTDS topic=activity, type=trades with empty
filters (the documented filter parameters don't actually work — see
WS_PROBE_FINDINGS.md). Filter client-side on the inner payload's
`eventSlug` field. Log every matched message verbatim. After the
window, print summary (counts + top traders by fills).

Usage:
    pip install 'websockets>=12'
    python polymarket/execution/tests/probes/negrisk_trades_probe.py
"""
from __future__ import annotations

import asyncio
import collections
import json
import signal
import sys
import time
from contextlib import suppress

import websockets

RTDS_URL = "wss://ws-live-data.polymarket.com"
TARGET_EVENT_SLUG = "elc-sot-mid-2026-05-12"
SUBSCRIBE_MSG = json.dumps(
    {
        "action": "subscribe",
        "subscriptions": [{"topic": "activity", "type": "trades", "filters": ""}],
    }
)
PING_INTERVAL_S = 5.0
RUN_SECONDS = 90.0
CONNECT_TIMEOUT_S = 10.0


async def heartbeat(ws) -> None:
    try:
        while True:
            await asyncio.sleep(PING_INTERVAL_S)
            await ws.send("ping")
    except (asyncio.CancelledError, websockets.ConnectionClosed):
        return


async def receive_loop(ws, deadline: float, stats: dict, matched: list) -> None:
    while time.monotonic() < deadline:
        try:
            raw = await asyncio.wait_for(
                ws.recv(), timeout=max(0.0, deadline - time.monotonic())
            )
        except asyncio.TimeoutError:
            return
        except websockets.ConnectionClosed as exc:
            print(f"[ws] closed: {exc!r}", flush=True)
            return

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if raw.strip().lower() == "pong":
            stats["pongs"] += 1
            continue
        if "payload" not in raw:
            stats["non_payload"] += 1
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            stats["bad_json"] += 1
            continue

        topic = msg.get("topic")
        msg_type = msg.get("type")
        stats["total_msgs"] += 1
        if topic != "activity" or msg_type != "trades":
            stats["other_topics"][f"{topic}/{msg_type}"] += 1
            continue

        # Payload arrives as a string in some streams, dict in others.
        payload = msg["payload"]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                stats["payload_parse_fail"] += 1
                continue
        if not isinstance(payload, dict):
            stats["payload_wrong_type"] += 1
            continue

        stats["all_trades"] += 1
        event_slug = payload.get("eventSlug")
        if event_slug == TARGET_EVENT_SLUG:
            stats["target_trades"] += 1
            stats["traders"][payload.get("proxyWallet", "?").lower()] += 1
            matched.append({**msg, "payload": payload})  # inner dict normalised
            wallet = payload.get("proxyWallet", "?")
            outcome = payload.get("outcome", "?")
            side = payload.get("side", "?")
            size = payload.get("size", "?")
            price = payload.get("price", "?")
            tx = (payload.get("transactionHash") or "")[:14]
            print(
                f"[TARGET] {wallet} {side:<4} {size:>9} @ {price:>7}  "
                f"outcome={outcome:<25} tx={tx}",
                flush=True,
            )


async def run() -> None:
    print(f"[ws] connecting {RTDS_URL}", flush=True)
    print(f"[ws] filtering client-side on eventSlug={TARGET_EVENT_SLUG!r}", flush=True)
    stats: dict = {
        "total_msgs": 0,
        "all_trades": 0,
        "target_trades": 0,
        "non_payload": 0,
        "bad_json": 0,
        "pongs": 0,
        "payload_parse_fail": 0,
        "payload_wrong_type": 0,
        "other_topics": collections.Counter(),
        "traders": collections.Counter(),
    }
    matched: list = []
    deadline = time.monotonic() + RUN_SECONDS
    try:
        async with websockets.connect(
            RTDS_URL, open_timeout=CONNECT_TIMEOUT_S, close_timeout=5
        ) as ws:
            await ws.send(SUBSCRIBE_MSG)
            print("[ws] subscribed", flush=True)
            hb = asyncio.create_task(heartbeat(ws))
            try:
                await receive_loop(ws, deadline, stats, matched)
            finally:
                hb.cancel()
                with suppress(asyncio.CancelledError):
                    await hb
    except Exception as exc:
        print(f"[ws] error: {exc!r}", flush=True)

    print("\n---- summary ----", flush=True)
    print(f"  total RTDS messages    : {stats['total_msgs']}", flush=True)
    print(f"  all activity/trades    : {stats['all_trades']}", flush=True)
    print(f"  ↳ matching target slug : {stats['target_trades']}", flush=True)
    print(f"  other topics seen      : {dict(stats['other_topics']) or '(none)'}", flush=True)
    print(f"  non-payload frames     : {stats['non_payload']}", flush=True)
    print(f"  payload parse failures : {stats['payload_parse_fail']}", flush=True)
    print(f"  pongs                  : {stats['pongs']}", flush=True)
    print("\n---- top traders on target event (by fill count) ----", flush=True)
    for wallet, n in stats["traders"].most_common(10):
        print(f"  {n:>4}  {wallet}", flush=True)
    if matched:
        print("\n---- first matched trade (verbatim) ----", flush=True)
        print(json.dumps(matched[0], indent=2)[:1500], flush=True)


def main() -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[exit] cancelled by user", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
