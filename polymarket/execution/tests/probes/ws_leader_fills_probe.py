"""Probe: can we watch one leader's fills across all Polymarket markets via WebSocket?

Read-only. No auth. No order submission. Runs ~75 s and exits.

Strategy (chosen from WS_PROBE_FINDINGS.md, answer "B-via-RTDS"):

  1. GET https://data-api.polymarket.com/trades?user=<LEADER>&takerOnly=false&limit=5
     to capture a baseline of the leader's most recent trades and the
     response shape from the polling fallback.

  2. Connect to wss://ws-live-data.polymarket.com (Polymarket's RTDS),
     subscribe to {topic:"activity", type:"trades", filters:""} (no
     wallet filter — RTDS doesn't expose one), and listen for ~60 s.

  3. For every received message, log topic/type and (if a trade) the
     proxyWallet, side, size, price, slug, transactionHash. Highlight
     any payload whose proxyWallet matches LEADER.

  4. Maintain the documented 5 s "ping" heartbeat.

  5. Print a summary of message counts and exit.

Run:
    pip install 'websockets>=12.0'
    python polymarket/execution/tests/probes/ws_leader_fills_probe.py
"""
from __future__ import annotations

import asyncio
import json
import signal
import sys
import time
import urllib.parse
import urllib.request
from contextlib import suppress

import websockets

LEADER = "0x9d84ce0306f8551e02efef1680475fc0f1dc1344".lower()
RTDS_URL = "wss://ws-live-data.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com/trades"
SUBSCRIBE_MSG = json.dumps(
    {
        "action": "subscribe",
        "subscriptions": [{"topic": "activity", "type": "trades", "filters": ""}],
    }
)
PING_INTERVAL_S = 5.0
RUN_SECONDS = 60.0
CONNECT_TIMEOUT_S = 10.0


def fetch_data_api_baseline() -> None:
    qs = urllib.parse.urlencode({"user": LEADER, "takerOnly": "false", "limit": "5"})
    url = f"{DATA_API_URL}?{qs}"
    print(f"[baseline] GET {url}", flush=True)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read()
            elapsed_ms = (time.monotonic() - t0) * 1000
            data = json.loads(body)
            print(
                f"[baseline] HTTP {resp.status} in {elapsed_ms:.0f}ms, "
                f"{len(data) if isinstance(data, list) else '?'} trades returned",
                flush=True,
            )
            if isinstance(data, list) and data:
                first = data[0]
                print(f"[baseline] sample fields: {sorted(first.keys())}", flush=True)
                print(
                    f"[baseline] most-recent: tx={first.get('transactionHash')} "
                    f"ts={first.get('timestamp')} side={first.get('side')} "
                    f"size={first.get('size')} price={first.get('price')} "
                    f"slug={first.get('slug')}",
                    flush=True,
                )
    except Exception as exc:
        print(f"[baseline] FAILED: {exc!r}", flush=True)


async def heartbeat(ws) -> None:
    try:
        while True:
            await asyncio.sleep(PING_INTERVAL_S)
            await ws.send("ping")
    except (asyncio.CancelledError, websockets.ConnectionClosed):
        return


async def receive_loop(ws, deadline: float, stats: dict) -> None:
    while time.monotonic() < deadline:
        timeout = max(0.0, deadline - time.monotonic())
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return
        except websockets.ConnectionClosed as exc:
            print(f"[ws] connection closed: {exc!r}", flush=True)
            return

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")

        # RTDS sends bare "pong" text frames in response to "ping".
        if raw.strip().lower() == "pong":
            stats["pongs"] += 1
            continue

        # Official client drops anything without "payload".
        if "payload" not in raw:
            stats["non_payload"] += 1
            print(f"[non-payload] {raw[:200]}", flush=True)
            continue

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            stats["bad_json"] += 1
            print(f"[bad-json] {raw[:200]}", flush=True)
            continue

        topic = msg.get("topic")
        msg_type = msg.get("type")
        payload = msg.get("payload") or {}
        stats["messages"] += 1

        if topic == "activity" and msg_type == "trades":
            stats["trades"] += 1
            wallet = (payload.get("proxyWallet") or "").lower()
            tag = "LEADER-MATCH" if wallet == LEADER else "trade"
            print(
                f"[{tag}] wallet={wallet} side={payload.get('side')} "
                f"size={payload.get('size')} price={payload.get('price')} "
                f"slug={payload.get('slug')} tx={payload.get('transactionHash')}",
                flush=True,
            )
            if wallet == LEADER:
                stats["leader_hits"] += 1
                if not stats["leader_samples"]:
                    stats["leader_samples"].append(payload)
        else:
            stats["other"] += 1
            print(
                f"[other] topic={topic} type={msg_type} "
                f"payload_keys={list(payload.keys())[:8]}",
                flush=True,
            )


async def run() -> None:
    fetch_data_api_baseline()
    print(f"[ws] connecting {RTDS_URL}", flush=True)
    stats = {
        "messages": 0,
        "trades": 0,
        "leader_hits": 0,
        "other": 0,
        "non_payload": 0,
        "bad_json": 0,
        "pongs": 0,
        "leader_samples": [],
    }
    deadline = time.monotonic() + RUN_SECONDS
    try:
        async with websockets.connect(
            RTDS_URL, open_timeout=CONNECT_TIMEOUT_S, close_timeout=5
        ) as ws:
            await ws.send(SUBSCRIBE_MSG)
            print(f"[ws] subscribed: {SUBSCRIBE_MSG}", flush=True)
            hb = asyncio.create_task(heartbeat(ws))
            try:
                await receive_loop(ws, deadline, stats)
            finally:
                hb.cancel()
                with suppress(asyncio.CancelledError):
                    await hb
    except Exception as exc:
        print(f"[ws] error: {exc!r}", flush=True)

    print("---- summary ----", flush=True)
    for key in (
        "messages",
        "trades",
        "leader_hits",
        "other",
        "non_payload",
        "bad_json",
        "pongs",
    ):
        print(f"  {key}: {stats[key]}", flush=True)
    if stats["leader_samples"]:
        sample = stats["leader_samples"][0]
        print(f"  leader sample fields: {sorted(sample.keys())}", flush=True)
        print(f"  leader sample: {json.dumps(sample, indent=2)[:800]}", flush=True)
    else:
        print("  (no leader trades observed in this window — expected if leader was idle)", flush=True)


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[exit] cancelled by user", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    # Make Ctrl-C exit even mid-await on platforms where asyncio.run swallows it.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
