# polymarket/execution/ — Plan

This is the rolling plan for the Polymarket copy-trading PoC. Read
this and CLAUDE.md before starting any work in this module.

## Goal

Mirror ONE hardcoded leader's fills on Polymarket with acceptable
latency, correctness, and capital safety. PoC scope: real money,
small ($50–$100). No paper-trading layer. Hardcoded leader address
via POLYMARKET_LEADER_ADDRESS env var.

## Architecture

Data flow:

  RTDS WebSocket → watcher → signal → risk → mirror_engine → venue
                                                                ├─ FAKE: _PrintVenueAdapter (cli.py)
                                                                └─ REAL: RealVenueAdapter (mirror/) →
                                                                         _kernel/polymarket_adapter →
                                                                         mirror/clob_http_client (substitute) →
                                                                         _kernel/polymarket_sdk_signer →
                                                                         py-clob-client → Polymarket CLOB

Real-venue path: the bot's kwargs-style `VenueAdapter` (defined in
`mirror/mirror_engine.py`) is implemented by `RealVenueAdapter`, which
translates to the kernel's intent-style API and runs a background
polling thread that journals fills as they arrive. The kernel's HTTP
client is **not** used (broken wire encoding — see decision 15);
`mirror/clob_http_client.py` is the substitute.

The `_kernel/` folder is vendored from midas/executor/ on 2026-05-06
and is treated as frozen. Do not edit it. Do not import from midas/.
See CLAUDE.md "Kernel vendoring status" for context.

## Key decisions made

1. **Detection: Polymarket RTDS WebSocket** at
   wss://ws-live-data.polymarket.com, topic "activity" type "trades",
   filter client-side on payload.proxyWallet. Confirmed via probe at
   tests/probes/ws_leader_fills_probe.py and findings at
   tests/probes/WS_PROBE_FINDINGS.md. Data API trades endpoint is
   the reconnect-gap fallback only, not the primary path.

2. **Sizing: fixed USD constant** (POLYMARKET_SIZING_USD=50) for the
   PoC. Leader-proportional and bankroll-aware sizing are deferred
   until research-side ships leader_rankings.parquet with
   estimated_bankroll_usd. Sizing is not a module — it's one config
   value read in mirror_engine.

3. **Exit logic: mirror exits only.** No independent stops or
   take-profits. When leader sells, bot sells the proportional
   amount of its position.

4. **Risk: per-trade cap, per-market cap, total-deployed cap,
   price-deviation cap, daily-loss halt, file-presence kill switch,
   max open positions.** Leader-health checking was considered and
   dropped — leader quality is a research-side concern, not
   execution.

5. **Order type: FOK (Fill-Or-Kill)** by default. Avoids partial
   fills and stale orders sitting on the book; if the price moved,
   prefer not to enter.

6. **Logging: append-only JSONL** via journal/ module. Daily-rotated
   files. Consumed by Dimitris's planned dashboard later.

7. **Address convention: lowercase 0x-prefixed proxy addresses
   throughout.** RTDS emits proxy addresses; config holds proxy
   addresses; comparisons are proxy-vs-proxy.

8. **Dedup is journal-backed.** signal/dedup.py rebuilds its
   seen-txhash set from the day's journal on startup. The journal
   is the single source of truth for "have I seen this fill." The
   in-memory LRU is an acceleration, not the source of truth.
   Implication: journal/jsonl_writer.py needs a read method
   (e.g. read_today()) in addition to write.

9. **Ambiguous submits halt the bot.** When
   _kernel/polymarket_adapter returns ambiguous_submit=True,
   mirror_engine writes a RiskHalt event with reason
   "ambiguous_submit" and trips the kill switch. Operator checks
   Polymarket UI to determine whether the order landed, manually
   reconciles, removes the kill switch. Auto-reconcile is a future
   enhancement.

10. **Bot tracks open positions via the journal.** Position state
    is reconstructed from FillRecorded events on startup. Signal
    classifier's EXIT signals are filtered against current
    positions: if the bot has no position in the market, the EXIT
    is logged (LeaderFillDropped, reason "no_position") and dropped
    silently. Expected behaviour, not an error. Implication:
    signal/classifier.py is not a stateless pure function — it
    needs read access to current position state.

11. **POLYMARKET_PRICE_DEVIATION_PCT governs the bot's slippage
    tolerance, not a comparison to leader's fill price.** When
    mirror_engine submits, it queries current best price and
    rejects if current price has moved >N% from leader's fill
    price. Orders that pass the check are submitted FOK at current
    price (not leader's). Failed FOK is logged and not retried —
    by retry time the situation has further changed. If FOK rejects
    too often empirically, IOC is the fallback (decide after
    observing).

12. **Empty/keepalive frames from RTDS are silently dropped.**
    RTDS sends empty text frames as part of normal traffic
    (observed once per smoke run, immediately after subscribe).
    watcher/leader_watcher.py step 1 strips and discards empty
    raw input before the JSON parse, with no journal entry. This
    preserves the "any WATCHER_MALFORMED_MESSAGE event means real
    schema violation" invariant for operator alerting.

13. **WatcherConnectFailed event for upstream connection
    failures.** rtds_client.py's on_reconnect callback only fires
    after a successful reconnect. Initial connection failures (or
    multi-attempt outages where no connect ever succeeds)
    previously had no journal trail, so the operator could read
    WATCHER_STARTED → WATCHER_STOPPED and incorrectly conclude
    the watcher was healthy. Now: every failed connect attempt
    fires on_connect_failed with (ws_url, attempt_number,
    error_repr), emitting a WATCHER_CONNECT_FAILED event.
    attempt_number is 1-indexed and resets to 1 on every
    successful connect.

14. **WS URL is RTDS, not CLOB.** POLYMARKET_WS_URL points to
    wss://ws-live-data.polymarket.com (RTDS) per
    WS_PROBE_FINDINGS.md. The CLOB WebSocket
    (wss://ws-subscriptions-clob.polymarket.com/ws/) is only
    relevant for authenticated user-channel subscriptions, which
    this module does not use.

15. **Substitute HTTP client at `mirror/clob_http_client.py`.** The
    vendored `_kernel/polymarket_clob_client.py` has a wire-format
    bug in `_build_request`: it `str()`s the kernel's `int`-typed
    quantity and price-ticks fields directly, sending e.g.
    `price="42"` to py-clob-client when the wire expects `"0.42"`.
    Off by 100×; would be rejected by Polymarket as price > 1.
    Fix (Strategy A from REAL_VENUE_PLAN.md): substitute the HTTP
    client only — keep the kernel's adapter for idempotency,
    ambiguous-submit handling, and event normalization. Substitute
    decodes `int(quantity) / quantity_scale` → fractional shares
    (default scale 10_000 = 4 decimal places) and
    `int(price_ticks) * tick_size` → dollar price. Tick size flows
    in via `set_tick_size(token_id, tick_size)` side channel
    populated by the wrapper before each submit (the kernel's
    `PolymarketCLOBClient` Protocol can't take it as an argument
    and we won't modify the kernel).

16. **Fill model: kernel returns ACK only; fills come async via
    polling.** The kernel's `submit_order` returns `ACKNOWLEDGED`
    when the venue accepts the order — *not* when it fills. Fills
    arrive via `kernel.poll_or_process_order_updates(None)` which
    must be polled on a separate thread. `RealVenueAdapter` runs
    a background poller (~700 ms cadence) that journals
    `FillRecorded` events as they arrive. Mirror_engine's
    "immediate fill info" path remains for the fake venue (which
    fills synchronously) but is bypassed for real (no fill data on
    `SubmitResult`). Bot's in-process `_bot_positions` map is
    therefore best-effort during a run; the journal-rebuild on
    next startup is the source of truth.

17. **Real-venue safety harness: `max_real_orders` +
    `operator_confirm`.** Two new env vars
    (`POLYMARKET_MAX_REAL_ORDERS=5`, default;
    `POLYMARKET_REQUIRE_OPERATOR_CONFIRM=false`, default) gate
    real-venue submits in mirror_engine. Both are gated on
    `venue.is_real_venue()` (duck-typed; absence treats as fake) so
    they never affect fake-venue runs. `max_real_orders` halts the
    bot after N submit *attempts* (accepted or rejected — even
    rejections consume the budget); writes `RiskHalt` reason
    `"max_real_orders"`. `require_operator_confirm` blocks on
    stdin per order; declining writes `RiskHalt` reason
    `"operator_aborted"` but does NOT halt — operator can decline
    individual orders without killing the bot. Both reasons added
    to `risk.types.VETO_REASONS`. Counter is per-process; restart
    resets.

18. **Refuse-to-start credential validation in real mode.**
    `cli._validate_real_credentials` rejects placeholders
    (empty / `dummy` / `placeholder` / all-zeros / etc.) before
    any kernel construction. Soak commands using all-zero
    `POLYMARKET_PRIVATE_KEY` now exit 4 (was 3 via
    `NotImplementedError`) when accidentally promoted to
    `POLYMARKET_VENUE=real`. Hard-prevents the most common
    "I forgot to swap in real creds" failure mode.

19. **FOK → IOC reduction.** Polymarket's `TimeInForce` enum on
    the kernel side exposes only `IOC` and `GTC`. Our
    `POLYMARKET_DEFAULT_ORDER_TYPE=FOK` maps to `IOC` with
    `expires_at_ns = ts_ns + 1` (immediate cancel of unfilled
    portion) — closest behavioural reduction. True FOK
    (all-or-nothing) isn't directly representable without
    bypassing the kernel; partial fills are accepted under IOC.
    For a $5 PoC where typical fills are 5–100 shares this is a
    non-issue; revisit if partial-fill behaviour becomes a
    problem.

20. **Synthetic `transaction_hash` for real-venue fills.** The
    kernel's `VenueFillEvent` has no `transaction_hash` field —
    only `package_id`, `leg_id`, `client_order_id`, `fill_qty`,
    `fill_price_ticks`, `ts_ns`. `RealVenueAdapter` synthesises
    `transaction_hash = f"{client_order_id}:fill:{ts_ns}"` for
    `FillRecorded` events, which is unique per fill and stable
    across restarts (lets state-rebuild parse the coid back out
    of the hash to clean up `_coid_to_fields` after a fill).
    Cross-referencing journal fills to PolygonScan therefore
    needs a manual `venue_order_id` lookup. Tracked as v2
    enhancement; not blocking PoC.

## Done

- Skeleton structure created (36 files, all empty docstrings).
- _kernel/ vendored from midas (5 files: venue, state_machine,
  polymarket_adapter, polymarket_clob_client,
  polymarket_sdk_signer).
- WebSocket probe completed; RTDS confirmed as detection path.
- CLAUDE.md, .env.example, README.md written.
- Cleanup pass: sizing/ deleted, risk/leader_health.py deleted,
  POLYMARKET_BANKROLL_USD removed, POLYMARKET_SIZING_USD,
  POLYMARKET_MAX_OPEN_POSITIONS, POLYMARKET_DEFAULT_ORDER_TYPE
  added, daily loss raised to 200.
- Confirmed _kernel/ does not read environment variables directly;
  all env reads will be in config.py.
- Smoke test verified watcher end-to-end against real RTDS: 40
  leader fills observed in 60 seconds, 0 reconnects, 0 connect
  failures, 0 unexpected malformed messages. Position rebuild
  logic verified independently against the smoke journal output
  (verify_position_rebuild.py script).
- Empty-frame filter added to watcher/leader_watcher.py.
- WatcherConnectFailed event added to journal vocabulary.
- .env.example WS URL corrected from CLOB to RTDS.
- Pricing mode added: `leader_fill` (default) | `current_book`.
  `leader_fill` mode bypasses the orderbook fetch entirely and
  submits at the leader's exact fill price. Justification: leaders
  who hammer one side of a thin book leave it empty by the time
  the bot looks up best-price; `current_book` mode rejected 100%
  of signals against gravia-terminal in soak. `leader_fill` works
  for information-driven leaders where the price level is the
  signal. `current_book` remains available for speed-driven
  leaders. Per-leader selection is research-side; for now it's a
  global config knob.
- per_trade_cap_usd raised from 20 → 50 in `.env.example` and
  `config._DEFAULTS` to be ≥ sizing_usd. Earlier value was an
  ordering accident: every ENTRY tripped size_cap and never
  reached the venue. Cap is still meaningful — it remains the
  only USD bound on EXITs (which are share-sized upstream).
- End-to-end pipeline validated against domah on real RTDS with
  `leader_fill` mode and the fake venue: 2 fills observed → 2
  signals → 2 risk-passes → 2 venue submits → 2 acks → 2 fills
  recorded. Math correct: `target_shares = sizing_usd /
  leader_fill_price` (e.g. 50/0.964 = 51.85, 50/0.37 = 135.13).
  Position state survives restart via journal rebuild (verified
  by `mirror/mirror_engine.py::_rebuild_state_from_journal`).
- Cross-checked the bot's journal against Polymarket's Data API
  for domah on 2026-05-08: bot's two journaled fills match the
  Data API exactly on `transactionHash`, size, price, timestamp.
  polymarketanalytics.com dropped one of the trades (the $2,890
  Powell BUY) but it appeared correctly in our journal AND in
  the Data API. **Polymarket Data API is the authoritative
  cross-check; bot matches it.**
- Discovered the kernel's wire-encoding bug in
  `_kernel/polymarket_clob_client.py::_build_request`: stringifies
  `int` quantity and `int` price-ticks directly without decoding
  to py-clob-client's expected decimal-string format. Sending
  `price="42"` for a $0.42 order would be rejected by Polymarket
  (price > 1). Confirmed by reading the wire path through to
  `OrderArgs(price=…, size=…)`. See decision 15.
- Implemented substitute HTTP client at
  `mirror/clob_http_client.py` (Prompt 1 of real-venue wiring).
  Drop-in replacement for `_kernel/polymarket_clob_client.py`
  that satisfies the kernel's `PolymarketCLOBClient` Protocol
  unchanged. Decodes `quantity` via `int(size) / quantity_scale`
  and `price` via `int(ticks) * tick_size` before passing to the
  signer. 30 unit tests covering decoding correctness, headers,
  HTTP error mapping (TimeoutError on socket timeout, OSError on
  URLError → kernel maps to `VenueTimeoutError`/`VenueTransportError`),
  and round-trip preservation for fills observed in production
  (5.0, 11.11, 12.96, 0.05 shares; 0.013, 0.42, 0.99 prices).
  Adapter-construction smoke confirmed the substitute satisfies
  the Protocol via `PolymarketVenueAdapter(client=substitute, …)`.
- Implemented `RealVenueAdapter` at
  `mirror/real_venue_adapter.py` (Prompt 2 of real-venue wiring).
  Wraps the kernel adapter to expose the bot's kwargs-style
  `VenueAdapter` Protocol; runs a background polling thread that
  journals fills via `FillRecorded` as `VenueFillEvent`s arrive
  from `kernel.poll_or_process_order_updates()`. State maps
  (`_coid_to_fields`, `_venue_to_coid`) are journal-backed —
  rebuilt from today's + yesterday's `OrderSubmitted` /
  `OrderAcknowledged` / `FillRecorded` / `OrderRejected` events
  on construction, so in-flight orders survive restart. Mock-kernel
  test suite (25 tests) covers translation, result-status mapping,
  exception → ambiguous, polling-thread fill journaling,
  unknown-coid handling, kernel-poll-exception resilience, and all
  three state-rebuild paths.
- Real-venue safety harness wired into `MirrorEngine.handle_signal`
  (decision 17). `max_real_orders` halts after the budget is hit;
  `require_operator_confirm` blocks on stdin per order and skips
  individual orders without halting. Two new VETO_REASONS values
  (`max_real_orders`, `operator_aborted`). Five mirror_engine
  tests lock in the behaviour, including the must-not-fire-on-fake-
  venue gate.
- Credential placeholder validation in `cli._validate_real_credentials`
  + 9-case parametrised `_looks_like_placeholder` test (decision 18).
- `cli.build_venue_adapter("real", config, journal=…)` constructs
  the full chain: `PyClobClientOrderSigner` → `ClobHttpClient`
  (substitute) → `PolymarketVenueAdapter` (kernel) →
  `RealVenueAdapter` (wrapper) → `wrapper.start()` (launches
  polling thread). cli shutdown calls `venue.stop()` if present
  (duck-typed; fake adapter has no `stop`).
- Operator runbook at `scripts/SMOKE_REAL.md` for the first real-
  money test. Step-by-step: prerequisites (funded proxy wallet,
  geo-unblocked region, real `.env`), paranoid first run
  (`MAX_REAL_ORDERS=1` + `REQUIRE_OPERATOR_CONFIRM=true`), then
  moderate-paranoia second run, with a failure-mode matrix mapping
  journal events to common causes.
- Test count: **132 → 214** across all changes since last sync
  (+30 ClobHttpClient, +25 RealVenueAdapter, +7 config, +5
  mirror_engine harness, +8 cli, +2 risk, plus parametrised
  expansions). Full suite passes in <1.5 s.
- Installed `py-clob-client` to user site-packages (transitive dep
  of the kernel's signer; was previously absent because no test
  exercised the import path until adapter-construction smoke).

## Next (in order)

1. ✅ Implement config.py + journal/ (foundations; everything
   depends on these).
2. ✅ Implement watcher/ (RTDS client, fill stream, dedup).
3. ✅ Implement risk/ breakers (one per file, pure functions).
4. ✅ Implement signal/ (classifier: entry/exit/scale).
5. ✅ Implement mirror/ (orchestration loop, fixed-USD sizing
   inline, kernel call).
6. ✅ Wire it all together in cli.py; ran dry against fake venue
   end-to-end on real RTDS.
7. ✅ Real-venue wiring: substitute HTTP client (Prompt 1) +
   `RealVenueAdapter` wrapper, polling thread, safety harness,
   credential validation, operator runbook (Prompt 2). 214 tests
   green; fake-venue smoke unchanged.
8. VPS provisioning + API key creation (region must allow
   Polymarket order submission; UK dev machine is geo-blocked
   for the Data API and presumably for order submit too —
   confirmed empirically during smoke runs). Pre-VPS: confirm
   geo-block for order submit specifically (Data API confirmed
   blocked from UK; order submit assumed blocked but not yet
   directly tested).
9. Funding a fresh Polymarket account (separate from any
   research-side wallet; bounded by POLYMARKET_MAX_CAPITAL_USD
   = $100 per PoC scope). $5 minimum for the first smoke per
   `scripts/SMOKE_REAL.md`. ← **next**
10. First real-money run per `scripts/SMOKE_REAL.md`:
    `MAX_REAL_ORDERS=1`, `REQUIRE_OPERATOR_CONFIRM=true`, $5
    sizing, against a known-active leader. Verify `FILL_RECORDED`
    in journal lines up with the order on the Polymarket UI;
    increase budget on subsequent runs.

## Deferred

- Leader-proportional sizing (waits for research-side parquet).
- API-key derivation at startup (currently manual paste in .env).
  Reference: midas/scripts/derive_api_keys.py and the QuickNode
  pattern (deriveApiKey() then createApiKey() fallback).
- Repo-wide reorganisation merging polymarket-copy/ and
  polymarket/execution/ under a shared parent. Wait until PoC is
  running.
- Re-vendoring _kernel/ against current midas (state_machine.py is
  gone upstream, would require non-trivial migration).

## Open questions to resolve before real money

- The +100 LOC growth in midas/executor/polymarket_sdk_signer.py
  since vendoring: bug fix, feature, or refactor? **PARTIALLY
  RESOLVED:** the vendored signer is sufficient for our use —
  cli wiring uses `PyClobClientOrderSigner` from `_kernel/`
  unchanged, and 25 mock-based tests of the wrapper pass.
  Whether the diff against current upstream contains anything
  load-bearing for live orders is still untested empirically.
  Will surface naturally on the first real-money smoke if it
  matters; not blocking.
- RTDS message latency in practice (probe estimated; needs
  real-world measurement).
- Maker vs taker coverage in RTDS (whether single fills emit one
  message or two). **RESOLVED:** RTDS does not emit market
  resolution events at all. Resolved positions vanish silently
  from a leader's RTDS-visible activity — the bot will not see
  the redemption/settlement of the leader's holdings, only their
  on-book fills. Not a blocker for the PoC: long-duration
  markets (the actual copy targets) have explicit on-book
  exits before resolution; high-frequency markets that resolve
  fast aren't good copy targets anyway. Implication: bot's
  in-process leader-position ledger may stay non-zero indefinitely
  for a market that resolves with the leader still holding;
  harmless because no further EXIT signal arrives.
- Confirm POLYMARKET_LEADER_ADDRESS for domah is the proxy
  address, not EOA. **RESOLVED:** RTDS emits proxyWallet field
  as the proxy address. Watcher compares lowercased proxyWallet
  against config.leader_address. Smoke test confirmed
  empirically that the matching works. Same comparison logic
  applies to all addresses in the system.
- Daily-loss halt: realised-only or mark-to-market? Decide
  when implementing risk/daily_loss.py. Default to realised-only
  for safety; mark-to-market can flicker on single bad prints.
- Geo-restriction: bot's deployment location must be in a region
  where Polymarket allows order submission. From the UK dev
  machine, Data API returns 403 Forbidden and CLOB order
  submission is presumably also blocked. RTDS WebSocket is NOT
  geo-blocked (smoke test connected and streamed fills
  successfully from the UK). Plan: VPS deployment in a
  non-blocked region (e.g. AWS Tokyo, Frankfurt) before first
  real-money submission. Until then, mirror_engine can be
  developed and tested against the fake_venue from _kernel/, but
  live order submission is not testable from the dev machine.
- Maker vs taker coverage in RTDS: smoke test produced 40 fills
  for one leader, all side=BUY. Whether RTDS emits both sides of
  every fill (one event per leg) or one event per fill is not
  yet empirically confirmed. Will surface naturally once signal/
  runs against a leader with mixed BUY/SELL activity.
  **RESOLVED:** see resolution note above — RTDS emits per-fill
  events keyed by `proxyWallet`; soak runs against domah,
  gravia, elPolloLoco, and the crypto-unwinder all confirmed
  the bot sees BUYs and SELLs identically. Open subtlety:
  resolution events are not emitted (see above).
- RTDS payload includes a server-side `timestamp` field
  (Unix-seconds, the on-chain block timestamp) distinct from the
  bot's local receive time. Currently the watcher requires the
  field to be present (step 6 of the validation pipeline) but
  does NOT store it on `LeaderFillObserved`; only the bot's
  local clock makes it onto `ts_utc` and `observed_at_utc`.
  Adding the RTDS-emit timestamp as a second field would let us
  reconcile bot's journal 1:1 with Polymarket's Data API on a
  single timestamp instead of having to add ~1 s for receive lag.
  ~30 min fix; deferred but worth doing before any forensics
  work that reconciles to chain order.
- Real-venue wiring requires translating the bot's local
  kwargs-style VenueAdapter interface (defined in
  `mirror/mirror_engine.py`) to `_kernel/polymarket_adapter.py`'s
  intent-style API (`VenueOrderIntent` with `quantity: int` and
  `limit_price_ticks: int`). **RESOLVED:** implemented in
  `RealVenueAdapter` per decision 16. share→int via
  `quantity_scale=10_000`, price→tick via `set_tick_size` side
  channel + per-asset cache, `Side`/`TimeInForce` mapped 1:1
  (FOK→IOC per decision 19), `package_id=client_order_id` and
  `leg_id="leg-0"` synthesised since copy-trading is single-leg.
- **Tick-size lookup robustness.** `RealVenueAdapter._get_tick_size`
  fetches from `{clob_url}/book?token_id=…` on first use per
  asset. If the fetch fails (network, 404, malformed response)
  the wrapper falls back to `tick_size_default=0.01`. This is
  silently wrong for sub-penny markets (gravia's $0.013 trades
  are on tick-0.001 markets). On a tick-0.001 market with our
  fallback, a $0.013 price would round to ticks=1, which the
  substitute decodes as 1 × 0.01 = $0.01 — a 23% mis-price.
  First real-money smoke must be on a market with confirmed
  tick=0.01 (most political/sports markets) until we add
  explicit-tick configuration or harden the fetch path.
- **Synthetic `transaction_hash` on real-venue `FillRecorded`.**
  The kernel's `VenueFillEvent` has no `transaction_hash` field;
  the wrapper synthesises `f"{client_order_id}:fill:{ts_ns}"`.
  Cross-checking journal fills against PolygonScan or the
  Polymarket UI requires a manual `venue_order_id` lookup
  instead. Document this in the operator runbook.
- **Geo-block confirmation for order submit specifically.**
  Data API confirmed 403 from UK. WebSocket (RTDS) works from
  UK. Order submission assumed blocked but **never directly
  tested** — could be 200, could be a different error. Worth
  one tiny attempt from UK before VPS provisioning to know
  exactly what the symptom is, even though the bot will run
  from a non-blocked region either way.
- polymarket-apis adoption deferred: research-side investigation
  confirmed it as actively maintained, Pydantic-v2 typed, covers
  Data API + CLOB read paths. Two flags: requires Python 3.12
  (project is 3.11+, must bump first), and poly-eip712-structs
  is loosely pinned. Plan: adopt for read-only clients
  (get_positions, get_trades, get_order_book_midpoint) when
  starting mirror/. Keep py-clob-client in _kernel/ for order
  signing — that's independently audited.

## Conventions

- **Addresses:** lowercase 0x-prefixed proxy wallet addresses
  everywhere. Compare proxy-to-proxy. Never mix EOA and proxy in the
  same comparison.
- **Timestamps:** UTC throughout. RTDS payload `timestamp` is
  Unix seconds; convert to UTC `datetime` at the watcher boundary
  and pass datetimes inward.
- **Market key:** `condition_id` is canonical. Token-level state
  uses `(condition_id, outcome_index)` or the explicit `asset_id`
  when interacting with the CLOB.
- **Trade key:** `(transaction_hash, log_index)` is the unique
  identifier for a fill — used for dedup and reconcile against the
  Data API. RTDS exposes `transactionHash` but not `log_index`;
  compose with `(asset, side, size, price, timestamp)` if a single
  tx contains multiple fills.
- **Module boundaries:** each subfolder has a single responsibility
  (watcher detects, signal classifies, risk vetoes, mirror submits,
  journal records). No `mirror/` import inside `risk/`. Risk
  functions are pure of `(config, state, candidate_order)`.
- **Code style:** Python 3.11+, `uv` for packaging,
  `from __future__ import annotations` at the top of every file,
  frozen dataclasses (`@dataclass(frozen=True, slots=True)`) for
  value types, type hints on all public functions, snake_case
  filenames, PascalCase classes.
- **Logging discipline:** every order submitted, every signal
  received, every risk halt, every reconnection emits a JSONL event
  via `journal/jsonl_writer.py`. No bare `print()` in the hot path.
- **Capital gate:** `POLYMARKET_MAX_CAPITAL_USD` is the hard
  ceiling. Never raise it in code; never bypass it in tests against
  the real venue (see Testing rule: no tests hit the real
  Polymarket API, ever).
- **Kill switch:** the file at `POLYMARKET_KILLSWITCH_PATH` is
  checked before every order submission. Existence ⇒ no new orders.
  Cancellations of existing orders are still permitted.
