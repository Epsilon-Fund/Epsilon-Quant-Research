# WebSocket leader-fills probe — findings

Investigation date: 2026-05-07. Read-only desk research + one probe script
([ws_leader_fills_probe.py](ws_leader_fills_probe.py)). No auth used. No
orders submitted.

---

## 1. Which answer is correct

**Answer is between B and C, and it's effectively a new option B′ — there is
a usable WebSocket, but it is *not* the CLOB WS that the original prompt
pointed at.**

### Evidence

- **CLOB user channel** (`wss://ws-subscriptions-clob.polymarket.com/ws/user`)
  is **authenticated and filtered by API key**. It only ever emits the
  authenticating wallet's own orders/trades. There is no "watch arbitrary
  address" parameter. → rules out **answer A**. *(Source:
  [docs.polymarket.com/developers/CLOB/websocket/user-channel](https://docs.polymarket.com/developers/CLOB/websocket/user-channel))*

- **CLOB market channel** (`wss://ws-subscriptions-clob.polymarket.com/ws/market`)
  is unauthenticated and per-`asset_id`, but the only trade-like event is
  `last_trade_price`, whose payload contains `asset_id`, `event_type`,
  `fee_rate_bps`, `market`, `price`, `side`, `size`, `timestamp` — and
  **no maker/taker/proxyWallet field**. So even if we subscribed to
  every market the leader trades on, we couldn't tell whose fill it was.
  → rules out the naive form of **answer B**. *(Source:
  [docs.polymarket.com/developers/CLOB/websocket/market-channel](https://docs.polymarket.com/developers/CLOB/websocket/market-channel),
  cross-checked against
  [Polymarket/clob-client/examples/socketConnection.ts](https://github.com/Polymarket/clob-client/blob/main/examples/socketConnection.ts))*

- **Real-Time Data Service (RTDS)** at `wss://ws-live-data.polymarket.com`
  is a separate Polymarket-operated WS (official client:
  [Polymarket/real-time-data-client](https://github.com/Polymarket/real-time-data-client)).
  Topic `activity`, type `trades` is **unauthenticated** and streams
  every public trade across every market. Each `payload` includes
  `proxyWallet` (the trader's wallet) and `transactionHash`. Filtering
  client-side on `proxyWallet == leader` is straightforward.

  Subscribe envelope:
  ```json
  {"action":"subscribe","subscriptions":[{"topic":"activity","type":"trades","filters":""}]}
  ```
  Heartbeat: send the literal text frame `"ping"` every 5 s; server
  responds with `"pong"`. Keep `filters` empty — the documented
  `event_slug` / `market_slug` filters are reportedly broken
  ([issue #34](https://github.com/Polymarket/real-time-data-client/issues/34)).
  There is **no** `user`/`address` filter exposed.

- **Data API fallback** at `GET https://data-api.polymarket.com/trades?user=<addr>&takerOnly=false`
  is unauthenticated and returns the same payload shape as RTDS.
  Use it as a baseline at probe start and as reconnect-gap backfill.
  *(Source: [docs.polymarket.com/api-reference/core/get-trades-for-a-user-or-markets](https://docs.polymarket.com/api-reference/core/get-trades-for-a-user-or-markets))*

So the operative answer is **B′: a non-CLOB Polymarket WS (RTDS) gives us
every trade unfiltered, with the trader address present, so we filter
client-side.** Pure C (Data-API polling only) is the documented fallback,
not the primary path.

---

## 2. Recommended watcher architecture

Connect once to `wss://ws-live-data.polymarket.com`, subscribe to
`{topic:"activity", type:"trades", filters:""}` with no auth, send a
literal `"ping"` text frame every 5 s, and discard messages whose
`payload.proxyWallet.lower()` does not equal the configured
`POLYMARKET_LEADER_ADDRESS`. Dedup on `payload.transactionHash`
(composite-key with `asset/side/size/price/timestamp` only if a single
tx contains multiple fills). Use the Data API trades endpoint
(`?user=<leader>&takerOnly=false`) as the [fallback_poller.py](../../watcher/fallback_poller.py)
data source, run it on a slow timer (e.g. 30 s) for reconcile/backfill
on reconnect — not as the primary path. The watcher should treat
RTDS-disconnect-with-bounded-gap as routine and only escalate to the
poller as a redundancy / gap-fill.

---

## 3. Follow-up questions the probe should empirically answer

1. **Maker vs taker coverage.** The Data API has a `takerOnly=true`
   default, which suggests asymmetric coverage. Does RTDS emit one
   message per fill (one side) or one per leg (both sides)? Probe by
   running `ws_leader_fills_probe.py` for a longer window against a
   high-volume address, then cross-checking against
   `?user=<addr>&takerOnly=false` over the same wall-clock window.
2. **Snapshot-on-connect.** RTDS messages carry a `connection_id` that
   hints at session resumption, but the docs are silent. Assume no
   snapshot on reconnect; rely on the Data-API backfill on every
   reconnect for safety.
3. **Filter-string format.** Issue #34 says slug filters are broken.
   Confirm empirically that `filters: ""` actually delivers all trades
   (vs. e.g. delivering nothing). The probe will surface this — if the
   60 s window logs zero `activity/trades` messages on a busy day,
   that's the symptom.
4. **Address normalization.** Polymarket's `proxyWallet` is the
   *proxy* address, not the EOA. Confirm
   `0x9d84ce0306f8551e02efef1680475fc0f1dc1344` (domah) is the proxy
   address, not the EOA. The probe's baseline GET will return the
   proxy that the Data API actually has on file — eyeball that match.
5. **Latency.** Measure the gap between trade-occurred-at and
   message-received-at. Need < 15 s end-to-end for the latency
   target. RTDS doesn't include a server-emit timestamp distinct from
   the trade timestamp, so we'll have to estimate from local clock.
6. **Cross-leakage.** Does RTDS emit anything other than
   `topic=activity / type=trades` on the same connection (e.g.
   `orders_matched`, market events) that we'd silently drop? The
   probe logs `[other]` for any unexpected topic/type and we should
   review the tally.

---

## 4. Rate-limit constraints to design around

- **RTDS:** No published rate limits or per-IP connection caps. The
  official client uses a single long-lived connection; the watcher
  should do the same. Reconnect with exponential backoff (e.g.
  1s → 2s → 4s → … capped at 30s) to avoid hammering on outage.
- **Data API `/trades`:** No published rate limits, but the endpoint
  is heavier than WS. Cap polling cadence at the slower of:
  (a) once every 20–30 s as the live fallback, (b) once per
  RTDS-reconnect for backfill. Use `limit=500` and walk back via
  `offset` only if backfill spans >500 trades (extremely unlikely
  for a single wallet within a normal reconnect gap).
- **Gamma API** (`https://gamma-api.polymarket.com/markets`): not used
  by the watcher in the recommended design; only relevant if we ever
  fall back to per-market CLOB WS subscriptions, which we won't.
- **Heartbeat cost:** 1 ping/5 s on a single connection. Negligible.
- **Dedup memory:** keep a rolling LRU of `~10_000` recent
  `transactionHash` strings to absorb WS↔poller overlap during
  backfill; that's <1 MB.

---

## Notes on the probe script

- Uses only stdlib + `websockets>=12.0`. `httpx` is not needed —
  baseline HTTP fetch uses `urllib`. A probe-local
  [requirements.txt](requirements.txt) declares the single dep.
- Authenticated calls were not required; the probe is fully read-only
  on public surfaces.
- 60 s WS window + heartbeat + baseline GET stays comfortably under
  the 90 s ceiling. Ctrl-C exits cleanly via `KeyboardInterrupt`.
- The probe does **not** populate [watcher/](../../watcher/). That is
  deliberate per the constraint; this script informs the watcher
  design and is meant to be run by a human operator, not imported.
