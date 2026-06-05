# polymarket/execution/ — Plan

> Hub: [[COWORK]] · [[POLYMARKET_BRAIN]]

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
    real-venue submits in mirror_engine and maker_engine. Both are gated on
    `venue.is_real_venue()` (duck-typed; absence treats as fake) so
    they never affect fake-venue runs. `POLYMARKET_MAX_REAL_ORDERS=0`
    is valid and means observe-only: real venue can be wired for
    auth/market/book reads, but no submit attempts are permitted.
    In mirror_engine, `max_real_orders` halts the bot after N submit
    *attempts* (accepted or rejected — even rejections consume the
    budget); writes `RiskHalt` reason `"max_real_orders"`. In
    maker_engine, the same limit writes `RiskHalt` and
    `MakerQuoteSkipped` but the loop keeps running so smoke runs can
    continue observing market lookup/book/pricing without placing
    orders. `require_operator_confirm` blocks on
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

21. **Politics NegRisk maker Phase-1 is measurement-first.** New
    `maker/` code is a one-condition scaffold driven by
    `POLYMARKET_MAKER_CONDITION_ID`. It places one passive GTC bid
    and one passive GTC ask on the YES token, both sized at
    `MAKER_SIZE_CONTRACTS=1` by default. It joins the displayed best
    bid / best ask; it does not improve, chase, peg dynamically, use
    OD pricing, or orchestrate multiple markets.

22. **Composite basket inventory comes from Data API activity polling.**
    `maker/negrisk_inventory.py` polls
    `/activity?user=<funder>&type=SPLIT,MERGE,REDEEM,CONVERSION`,
    applies split/merge/redeem/convert deltas to an in-memory
    `basket_inventory[condition_id]`, and persists replayable JSONL.
    RTDS is not used for these operations because it is silent for
    split/merge/redeem/convert events. The maker engine consults
    `get_basket_exposure(condition_id)` and skips the bid side once
    exposure is at the Phase-2 hard cap of 10 contracts.

23. **Resolution and redemption are polling-driven and non-blocking.**
    `maker/resolution_handler.py` polls Gamma market rows for the
    bot's open condition_ids and logs `MarketResolved` when a market
    is closed/inactive. It also polls
    `/positions?user=<funder>&redeemable=true`, logs
    `PositionRedeemable`, and attempts a best-effort
    `NegRiskAdapter.redeemPositions(bytes32,uint256[])` call via
    lazy web3 if `POLYMARKET_RPC_URL` and a usable private key are
    configured. Redemption failures log `RedemptionFailed` and do
    not halt the loop.

24. **Maker telemetry is the product.** `maker/maker_engine.py` logs
    quote placement, cancel, fill telemetry, missed fills, and
    resolution cancels. Fill telemetry includes
    `top_maker_rank_at_fill`, `post_fill_price_drift_60s`,
    `news_proximate`, and `fill_share_this_market`. Values may be
    `None` when the needed calendar or post-fill market data is not
    available; the field is still journaled so Phase 2 dashboards can
    distinguish missing telemetry from a false value.

25. **Scheduled events are manual ops input.** `maker/event_calendar.py`
    loads `maker/politics_events.yaml` (stdlib-only JSON or a tiny YAML
    subset) and answers event-proximity checks. No scraper, X/Twitter
    feed, or LLM news reader is part of Phase 2.

26. **Maker fill detection is Data-API-polled, not journal-matched.**
    `FillRecorded` carries no `client_order_id`, so the maker engine
    cannot match journal fills to live quotes. `process_fills_once(market)`
    instead polls `data-api/trades`, keeps rows where
    `proxyWallet == funder` and `ts >= session_start`, and matches a fill
    to a resting quote within half a tick of price. Unmatched funder fills
    (taker fills on orders that already left quote state) are still logged
    as `MakerFillTelemetry` with `top_maker_rank_at_fill=None`.

27. **NegRisk redemption supports 3+ outcome markets.**
    `_redeem_amounts` sizes the `uint256[]` vector to `outcome_index + 1`
    (zero-padded) and only rejects `outcome_index < 0`, instead of the old
    0-or-1-only guard that broke every NegRisk event with 3+ candidates.

28. **Maker has its own CLI entry point.** `maker/cli.py` wires
    inventory + resolution + engine, runs their three threads, and shuts
    down by cancelling quotes then stopping components in reverse start
    order. It reuses `cli.build_venue_adapter` (so `MAX_REAL_ORDERS` /
    `REQUIRE_OPERATOR_CONFIRM` apply unchanged) and is reachable via
    `python -m polymarket.execution --mode maker`. `MakerSessionStarted` /
    `MakerSessionStopped` journal the session lifecycle.

29. **Maker supports real-venue observe-only mode.**
    `POLYMARKET_MAX_REAL_ORDERS=0` is accepted by `ExecutionConfig`.
    On a real-like maker venue the first submit opportunity is blocked
    by the existing `>= max_real_orders` gate, journals `RiskHalt`
    reason `"max_real_orders"` plus `MakerQuoteSkipped`, and leaves the
    loop alive. This is the Phase-2 smoke mode for validating auth,
    market lookup, book fetch, and quote pricing without submitting.

30. **Maker first smoke is tick-allow-listed.**
    `MakerEngineConfig.allowed_tick_sizes` defaults to `{0.01}` and can
    be overridden with comma-separated `POLYMARKET_MAKER_ALLOWED_TICKS`.
    If Gamma reports a tick outside the allow-set, maker_engine cancels
    any open quotes for the market and skips both sides with
    `MakerQuoteSkipped reason="tick_size_not_allowed"` including the
    observed tick in `detail`.

31. **Auth-only verification is a read path, not a quote path.**
    `python -m polymarket.execution --mode maker --check-auth` builds
    the real venue adapter, calls the kernel open-order reconciliation
    read, prints the open-order count, stops the adapter, and exits. It
    submits nothing. Exit code 0 means the read succeeded; 5 means the
    auth/open-order read failed.

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
- **NegRisk handling complete.** `mirror/market_metadata.py`
  (Gamma-backed cache for `negRisk` + `tick_size` per
  condition_id). `mirror/clob_signer.py` (replaces kernel
  signer; passes `PartialCreateOrderOptions(neg_risk=…,
  tick_size=…)` to py-clob-client's `ClobClient.create_order`).
  `set_neg_risk(token_id, value)` side-channel on the HTTP
  client, symmetric with `set_tick_size`. Refuses to submit
  on Gamma fetch failure with `OrderRejected` reason
  `"cannot_classify_market"`. Tested end-to-end with both
  flag values; **233 tests total**.

- **Kernel signer dead-code in our path.**
  `mirror/clob_signer.py` fully replaces
  `_kernel/polymarket_sdk_signer.py` for our execution chain.
  Verify with:
  `grep -rn "PyClobClientOrderSigner" polymarket/execution/ --include="*.py"`
  Should show only the kernel file itself (untouched) and
  doc references in `mirror/clob_signer.py` /
  `mirror/clob_http_client.py` / `cli.py` (which describe
  the replacement, not import it). The live code path goes
  `ClobHttpClient → ClobSigner → py-clob-client.ClobClient`,
  no kernel signer involved.

- NegRisk wire-signing support landed: new
  `mirror/market_metadata.py` (Gamma-backed `MarketMetadataCache`
  with permanent in-process caching of `negRisk` + `tick_size`),
  `set_neg_risk(token_id, …)` side-channel on `ClobHttpClient`
  symmetric with `set_tick_size`, new `mirror/clob_signer.py`
  (`ClobSigner` — direct py-clob-client wrapper that passes
  `PartialCreateOrderOptions(neg_risk=…, tick_size=…)` to
  `ClobClient.create_order`, replacing the kernel's
  `PyClobClientOrderSigner` in the real-venue path). cli wiring
  updated: `build_venue_adapter("real", …)` now constructs the
  full chain `ClobSigner → ClobHttpClient → kernel adapter →
  RealVenueAdapter` plus `MarketMetadataCache` from
  `config.gamma_url`. Gamma fetch failure ⇒ `OrderRejected`
  reason `"cannot_classify_market"`; refuse-to-submit semantics
  match the rest of the credential-validation paranoia. **19 new
  tests** (11 metadata + 4 http_client NegRisk + 4 wrapper NegRisk);
  full suite **214 → 233 green**. Reference: NegRisk findings doc
  at `tests/probes/NEGRISK_FINDINGS.md`.

- **Politics NegRisk maker Phase-1 prerequisites landed.**
  `maker/event_calendar.py`, `maker/negrisk_inventory.py`,
  `maker/resolution_handler.py`, `maker/maker_engine.py`,
  `maker/politics_events.yaml`, and `maker/README.md` implement the
  Phase-2 measurement scaffold. Tests in `tests/maker/` cover the
  NegRisk verifying-contract acceptance path, event calendar,
  inventory replay/dedup, resolution/redemption polling, two-sided
  passive quoting, cancel/replace, resolution cancel, basket exposure
  cap, and required telemetry fields. Full execution suite:
  **248 green** via repo-root `PYTHONPATH=. uv run --no-project --with
  pytest --with py-clob-client python -m pytest
  polymarket/execution/tests`.

- **Politics NegRisk maker Phase-1 review pass — three targeted fixes
  (decisions 26–28).** Resolved the blockers found in the Phase-1
  review handoff (`brain/handoffs/2026-06-03_politics_negrisk_phase1_review.md`)
  before the Phase-2 smoke:
  - **Fix A — fill detection (decision 26).** `MakerEngine.process_fills_once`
    no longer reads `FILL_RECORDED` journal events (which carry no
    `client_order_id`, so the match always returned `None` and
    `MakerFillTelemetry` was never logged). It now takes the active
    `MakerMarket` and polls `data-api/trades`, filtering to rows where
    `proxyWallet == config.funder` and `ts >= session_start`, matching a
    fill to a resting quote when `abs(quote.price - fill_price) <=
    tick_size/2`. Matched quotes are popped from `_quotes`; an unmatched
    funder fill is still ours and is logged with
    `top_maker_rank_at_fill=None`. `_log_fill_telemetry` now takes the raw
    trade row plus a `matched` flag; dead `_coid_from_fill_event` /
    `_parse_event_ts` helpers deleted.
  - **Fix B — `_redeem_amounts` for 3+ outcome markets (decision 27).**
    `resolution_handler._redeem_amounts` previously raised for any
    `outcomeIndex not in (0, 1)`, breaking redemption on every NegRisk
    event with 3+ candidates. Now it accepts `outcome_index >= 0` and
    sizes the amounts vector to `outcome_index + 1` (zero-padded).
  - **Fix C — maker CLI entry point (decision 28).** New
    `maker/cli.py` wires `NegRiskInventoryTracker` + `ResolutionHandler`
    + `MakerEngine`, starts all three background threads, installs
    SIGINT/SIGTERM handlers, and on shutdown cancels all open quotes then
    stops the three components in reverse start order (plus the
    duck-typed venue `stop()`). It reuses `cli.build_venue_adapter` so the
    `MAX_REAL_ORDERS` / `REQUIRE_OPERATOR_CONFIRM` safety harness is
    respected identically. `__main__.py` now dispatches `--mode maker`
    (default `copytrade`). Two additive journal events —
    `MakerSessionStarted` / `MakerSessionStopped` — record session
    lifecycle. An optional `POLYMARKET_MAKER_MAX_RUNTIME_SECONDS`
    auto-stop supports bounded dry smokes.
  - **Nice-to-fixes.** `DataApiTradeClient.price_after_60s` switched from
    a symmetric ±10s window to "first trade at or after `fill_ts + 60s`,
    no upper bound" (illiquid politics markets were yielding too many
    `None`s); `MakerEngine._top_maker_rank_at_fill` gained a docstring
    noting it is a fill-order proxy, not a true queue rank.
  - **Pre-smoke hardening (decisions 29–31).** `MAX_REAL_ORDERS=0`
    is now an observe-only real-venue mode, maker_engine blocks
    disallowed ticks via `POLYMARKET_MAKER_ALLOWED_TICKS` (default
    `{0.01}`), and `--mode maker --check-auth` performs a read-only
    open-orders auth check with no submit path.
  - **Tests.** `tests/maker/` updated for the Data-API fill-detection path
    (matched + unmatched-taker cases), the 3+ outcome and negative-index
    redemption cases, maker CLI lifecycle, observe-only
    `MAX_REAL_ORDERS=0`, disallowed-tick skips, and read-only auth check.
    Maker suite **22 green**; full execution suite **256 green** with
    `PYTHONPATH=. uv run --no-project --with pytest --with py-clob-client
    --with websockets python -m pytest polymarket/execution/tests`.

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
8. ✅ NegRisk handling: `MarketMetadataCache` (Gamma-backed),
   `ClobSigner` (passes `PartialCreateOrderOptions(neg_risk=…)`
   to py-clob-client), refuse-to-submit on metadata cache miss.
   233 tests green.
9. ✅ Politics NegRisk maker Phase-1: resolution handler,
   composite inventory tracker, manual event calendar, and minimal
   one-market passive maker measurement scaffold. 248 tests green.
10. PLAN.md sync (this update). ← **just completed**
11. Snapshot commit + tag
    (`execution-wiring-complete-negrisk` or similar). ← **next**
12. Slack message to colleague about the kernel encoding bug
    (`mirror/clob_http_client.py` exists because the kernel's
    `_build_request` stringifies `int` ticks / integer-shares
    without the dollar/decimal conversion py-clob-client
    expects; ours fixes that downstream and also bypasses the
    kernel signer entirely for NegRisk reasons).
13. Polymarket account credentials into `.env` (private key
    from wallet; API key/secret/passphrase via
    `midas/scripts/derive_api_keys.py` or Polymarket UI;
    `POLYMARKET_FUNDER` from the Account page).
14. Auth-only verification path implemented:
    `python -m polymarket.execution --mode maker --check-auth` hits
    the CLOB open-orders reconciliation/read and submits nothing.
    Operator still needs to run it with real credentials before the
    first live smoke.
15. First real-money smoke per `scripts/SMOKE_REAL.md`:
    `MAX_REAL_ORDERS=1`, `REQUIRE_OPERATOR_CONFIRM=true`, $5
    sizing, against a known-active leader. Verify
    `FILL_RECORDED` in journal lines up with the order on the
    Polymarket UI; increase budget on subsequent runs.
16. Post-smoke: medium robustness pass per the
    "Post-PoC robustness roadmap" section below.

## Known limitations

- **Kernel idempotency-via-client_order_id is bot-side, not
  venue-side.** py-clob-client generates its own salt per
  `create_order` call. Resubmitting the same `client_order_id`
  produces a different on-wire order with a different
  `venue_order_id`. The kernel adapter maps
  `client_order_id ↔ venue_order_id` synchronously within
  each submit call (so a single submit is internally
  consistent), but this means:
    * The "idempotency" advertised by the kernel adapter
      only protects against bot-side double-submission via
      in-memory tracking — not against the venue receiving
      two distinct orders if the bot retried after a network
      hiccup.
    * Ambiguous-submit handling must remain conservative
      (halt the bot, don't auto-retry). Auto-retry would
      risk creating duplicate venue orders.
    * Acceptable for the PoC (`max_real_orders` cap is small,
      operator-confirm is on for the first runs) but a real
      concern for unattended operation with many orders/day.

- **NegRisk per-market cap is structurally correct, but total-deployed
  cap is over-conservative on diversified NegRisk positions.** Each
  binary sub-market within a NegRisk event has its own `condition_id`,
  so the per-market cap (`per_market_cap_usd`) correctly treats positions
  on different outcomes as separate markets. The over-constraint is on
  `max_capital_usd`: if a leader spreads $30 across 3 mutually-exclusive
  outcomes of a NegRisk event, the bot's total-deployed cap counts the
  full $90 as exposure even though the economic loss is bounded by
  $1 × shares (NegRisk's convert-floor). Conservative for the PoC; if
  cap-vetoes start firing surprisingly often on NegRisk leaders,
  consider a NegRisk-aware deployed-cap that nets across same-event
  outcomes.

- **No mirroring of leader split / merge / redeem / convert actions.**
  These are silent on RTDS; the bot would have to poll
  `data-api.polymarket.com/activity?type=SPLIT,MERGE,REDEEM,CONVERSION`.
  Skipped for PoC — they're inventory-management moves, not directional
  signals. See NEGRISK_FINDINGS.md §4 for rationale.

- **Self-redemption is best-effort, not a capital-safety guarantee.**
  `maker/resolution_handler.py` can call
  `NegRiskAdapter.redeemPositions` via web3.py, but only when
  `POLYMARKET_RPC_URL` and a usable private key are configured.
  Failures are intentionally non-blocking and journaled. Manual UI
  redemption remains the fallback if the proxy/safe path or gas setup
  is not ready.

- **Synthetic `transaction_hash` on real-venue `FillRecorded` events.**
  Format: `f"{client_order_id}:fill:{ts_ns}"`. The kernel's
  `VenueFillEvent` carries no on-chain tx hash. Cross-referencing
  journal fills to PolygonScan therefore requires a manual venue
  order id lookup. v2 enhancement.

## Post-PoC robustness roadmap

First real-money smoke happens *before* any robustness work;
that smoke produces real-world data about which assumptions
hold. Colleague's parallel bot had two specific failure modes
worth designing against — max-size-per-market didn't fire when
expected, and a separate bug submitted the wrong order. These
inform what to build next, but only after the smoke. Medium-
scope investments planned (none of these block PoC):

1. **State verification tools.** Standalone script that reads
   the journal and reports current bot state — bot positions,
   leader positions, deployed_usd by market, in-flight
   orders, daily realised PnL. Disagreement with the
   Polymarket UI ⇒ state drift detected before it costs
   money. Lives in `scripts/`.

2. **Pre-submission invariant assertions.** Before any order
   is submitted, assert that fields are internally consistent:
    * `size > 0` and within configured cap.
    * `price ∈ [0, 1]` (Polymarket binary range).
    * `side ∈ {"BUY", "SELL"}`.
    * `asset_id` matches a market the leader actually traded
      (deduce from `signal.source_transaction_hash`).
    * `condition_id` from signal matches `asset_id`'s parent
      market in our metadata cache.
   Cheap assertions; loud failures. Defends against the
   "wrong order submitted" class of bug.

3. **Cap verification under fault injection.** Tests that
   deliberately seed corrupted in-memory state (bot positions
   disagree with journal-derived state) and verify caps still
   fire correctly. Current tests assume state is correct.

4. **Periodic state snapshots in the journal.** Every N
   minutes, write a `StateSnapshot` event with the bot's full
   position state, deployed_usd, halted status, etc. Lets you
   reconstruct "what was the bot's state at 14:23 yesterday"
   without piecing together individual fills.

5. **Replay tooling.** Given a journal slice, reconstruct
   what the bot saw and what it did, second-by-second.
   Useful for post-mortem. May overlap with the position-
   rebuild logic in `mirror_engine` and `signal/classifier`;
   could be a thin layer on top.

Not in scope for PoC; planned for the medium robustness pass
after the first real-money smoke succeeds.

## Deferred

- Leader-proportional sizing (waits for research-side parquet).
- API-key derivation at startup (currently manual paste in .env).
  Reference: midas/scripts/derive_api_keys.py and the QuickNode
  pattern (deriveApiKey() then createApiKey() fallback).
- ~~Repo-wide reorganisation merging polymarket-copy/ and
  polymarket/execution/ under a shared parent.~~ Done — research
  moved to polymarket/research/.
- Re-vendoring _kernel/ against current midas (state_machine.py is
  gone upstream, would require non-trivial migration).

## Open questions to resolve before real money

- The +100 LOC growth in midas/executor/polymarket_sdk_signer.py
  since vendoring: bug fix, feature, or refactor? **RESOLVED:
  bypassed entirely.** `mirror/clob_signer.py` replaces the
  kernel signer in the real-venue path (the kernel signer
  ignored `PartialCreateOrderOptions`, which made NegRisk
  signing impossible anyway). The kernel signer file is now
  dead code in our execution chain — still vendored, still
  untouched, but never imported from `polymarket/execution/`.
  Verified by `grep -rn "PyClobClientOrderSigner"
  polymarket/execution/ --include="*.py"`: only docstring
  references in the new files and the kernel file itself
  appear; no live imports. If midas's upstream signer ever
  gets improvements we want, we'd need to actively pull them
  into `mirror/clob_signer.py` — they don't propagate
  automatically.
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
- **Tick-size lookup robustness.** Tick size now comes through the
  Gamma-backed metadata path, but missing/malformed Gamma tick fields
  still fall back to `0.01`. This is silently wrong for sub-penny
  markets (gravia's $0.013 trades are on tick-0.001 markets). On a
  tick-0.001 market with a 0.01 fallback, a $0.013 price would round
  to ticks=1, which the substitute decodes as 1 × 0.01 = $0.01 — a
  23% mis-price. For the maker Phase-2 smoke, `MakerEngine` now
  enforces an allowed-tick set (default `{0.01}`) and skips both sides
  on disallowed ticks. Broader sub-penny support still requires
  hardening the metadata/fetch path before allowing `0.001`.
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
