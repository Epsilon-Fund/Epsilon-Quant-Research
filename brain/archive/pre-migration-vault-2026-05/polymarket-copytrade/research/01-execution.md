---
title: "Copytrade 01 — execution layer state dump (2026-05-09)"
created: 2026-05-09
archived: 2026-09-09
status: archived — pre-migration vault (May 2026); historical, not current
owner: justin
project: copytrade
para: archive
hubs:
  - pre-migration-vault-2026-05
tags:
  - archive
  - pre-migration-vault
  - copytrade
---

> **ARCHIVED — PRE-MIGRATION VAULT (May 2026).** Historical record, imported 2026-09-09 from the standalone Cowork vault that predates this repo becoming the brain. Index: [[pre-migration-vault-2026-05]].
> **DEPRIORITISED (2026-08-25).** The copy-trading thread is not the active research thread (that is [[strat_market_making]]). It is not archived either — its execution/signing infrastructure is live and shared with the market-making machinery. Pick this thread back up only with Justin.

# 01 — execution layer (live copy-trading bot)

*Source: state dump from `copytrade-exec` chat, 2026-05-09. Repo: `Epsilon-Quant-Research/polymarket/execution/`. Sibling: `polymarket/research/` — see `02-data.md`.*

---

## TL;DR

Single-leader copy-trading PoC. **Engineering complete, fake-venue verified, never run real money.** 214 unit tests passing. End-to-end test: real RTDS feed observed a leader's $2,890 buy → classifier → risk → fake venue mirror, math correct. Three operational tasks (PLAN.md sync + snapshot, credentials in `.env`, smoke run) stand between today and first real-money fill. UK is geo-blocked for order submission; needs VPN/VPS in a non-blocked region. Strategy thesis: profitable Polymarket traders on multi-day-or-longer markets have a real edge; proportionally-sized mirror orders capture part of it. Multi-leader cohort mode is the long-term form, contingent on research-side `leader_rankings.parquet`.

---

## Goal / what the PoC is doing today

The bot watches one specific Polymarket trader's wallet in real time and mirrors their trades on a separate Polymarket account, with safety controls and full auditability.

Single-leader PoC; cohort copy-trading (multi-leader, ranking-weighted) is the long-term form.

---

## Architecture

Seven modules + a vendored kernel. Single Python process, threading-based concurrency (no asyncio).

```
RTDS WebSocket (wss://ws-live-data.polymarket.com — Polymarket's live trade firehose)
        │
        ▼
   watcher/         Subscribes to all Polymarket trades, filters by leader's
                    proxy wallet address. Pushes LeaderFillObserved events
                    to fill_queue. Runs in a background thread.
        │
        ▼ fill_queue (queue.Queue, maxsize=1000, blocking on full — backpressure)
        │
   signal/          Reads fills, deduplicates (journal-backed), classifies
                    as ENTRY / EXIT, computes target_size_shares. Maintains
                    in-memory state for both bot and leader positions.
                    Emits MirrorSignal events.
        │
        ▼ signal_queue (queue.Queue, maxsize=1000)
        │
   risk/            Seven independent breakers, pure functions:
                    (config, state, candidate_order) → Veto | None.
                    Per-trade cap, per-market cap, total-deployed cap,
                    price deviation, daily loss, max open positions, kill switch.
                    First veto wins; veto = skip signal, halt only on kill switch.
        │
        ▼
   mirror/          Builds CandidateOrder. Applies safety harness
                    (max_real_orders limit, optional operator_confirm prompt).
                    Submits via venue adapter. Handles ACK / REJECT / AMBIGUOUS
                    distinctly. Real venue: substitute HTTP client → kernel's
                    PolymarketVenueAdapter → py-clob-client → Polymarket CLOB.
                    Polling thread asynchronously catches fills and writes
                    FillRecorded events.
        │
        ▼
   _kernel/         Vendored from midas/executor/. Treated as frozen.
                    PolymarketVenueAdapter (order lifecycle, idempotency,
                    ambiguous-submit detection), polymarket_sdk_signer,
                    venue.py (Protocol definitions), state_machine (event types).
   journal/         Append-only daily-rotated JSONL files. 13 event types.
                    Source of truth for: dedup state, position state,
                    daily PnL, in-flight order tracking. Re-read on every
                    restart by Classifier, MirrorEngine, RealVenueAdapter
                    to rebuild in-memory state.
   config.py        Frozen dataclass. Reads ~25 env vars, validates types
                    and constraints. Refuses to start in real mode if
                    credentials look like placeholders.
   cli.py           Entry point: python -m polymarket.execution.cli.
                    Wires everything, manages thread lifecycles, signal
                    handlers (SIGINT/SIGTERM), shutdown drain.
```

### Where positions are held

Nowhere persistent. Position state is *reconstructed from the journal on every startup* — today's + yesterday's `FillRecorded` events are replayed through average-cost accounting in `signal/classifier.py` and `mirror/mirror_engine.py`. Same pattern for the bot's *and* the leader's positions. After startup, in-memory dicts are updated as new fills arrive. Bot crashes → restart → state is rebuilt. The journal is the database; there is no other database.

This is also true for dedup (which `transaction_hash`es have we seen?) and in-flight orders (which `client_order_id`s are pending?). Same journal-replay pattern. It's the bot's universal recovery mechanism.

---

## What's actually working vs hardcoded

### Verified end-to-end

- RTDS WebSocket connection. 60-second smoke produced 32 fills with correct latency, no malformed messages, zero dropped reconnects.
- Watcher → journal pipeline: bot's journal verified more accurate than third-party Polymarket analytics sites, cross-checked against Polymarket's own Data API. All transaction hashes match.
- Classifier handles all branches correctly. Domah test produced 2 ENTRYs + 2 `no_position` drops + 3 `leader_no_position` drops, every fill accounted for.
- Risk pipeline: all 7 breakers tested, fail-fast ordering verified. Kill switch correctly triggers halt; other vetoes skip-and-continue.
- Fake-venue end-to-end: 2 fills observed → 2 signals → 2 risk-passes → 2 venue submits → 2 acknowledgements → 2 fills recorded in same session. Math correct (`50 USD / leader_price = target_shares`).
- HTTP client substitute: replaces a real wire-encoding bug in the vendored kernel that would have caused 100% rejection on real venue. 30 unit tests covering decoding, transport, and round-trip preservation.
- Polling thread: tested with mock kernel events. Handles missing-mapping cases, kernel exceptions, journal-replay state recovery.

### Hardcoded for PoC

- **Leader**: one address, set via `POLYMARKET_LEADER_ADDRESS`. No dynamic switching, no multi-leader.
- **Sizing**: fixed $50 per trade (configurable via `POLYMARKET_SIZING_USD`; tuned down to $5–10 for first runs). Same dollar amount regardless of leader's conviction or current price.
- **Exit logic**: mirror exits only. No independent stops, no take-profit. When leader sells X% of their position, bot sells X% of its position.
- **Pricing mode**: `leader_fill` by default (submit at the leader's exact fill price). Alternative `current_book` mode exists (best ask/bid + slippage) but isn't default.
- **Order type**: FOK requested, IOC on the wire (kernel doesn't expose FOK; IOC with immediate-expiry is functionally equivalent).
- **Bankroll estimation**: not done. Sizing is dollar-fixed.

### Not yet exercised in real life

- Real-money submission. Wired but never run. Manual smoke runbook exists in `scripts/SMOKE_REAL.md`.
- Operator-confirmation prompt. Code path exists, never tested with a human at the keyboard.
- Polling-thread fill recovery. Tested with mock kernel events; never seen a real Polymarket fill come back through the polling cycle.

---

## What's in flight

Three small operational tasks, no engineering.

### Account setup

- Polymarket account exists, $113 USDC.e funded, separate from colleague's account.
- Need: EOA private key (from "Private Key" tab in Polymarket UI), proxy wallet address (from Account page), CLOB API credentials (probably via `midas/scripts/derive_api_keys.py` rather than the Relayer page in the UI — those are different credential systems and the Relayer page is *not* the one the bot needs).

### Documentation hygiene

- PLAN.md sync (decisions accumulated since last update).
- Snapshot commit + tag (mark "engineering complete" state).
- Slack message to colleague about the kernel encoding bug we worked around so he knows before midas's executor goes live.

### First smoke run

- VPN to non-blocked region (UK is blocked for order submission; RTDS reads work fine).
- `.env` populated with real credentials.
- Auth-only test (read open orders) before any submission.
- Runbook: `MAX_REAL_ORDERS=1`, `REQUIRE_OPERATOR_CONFIRM=true`, `SIZING_USD=10`. Active leader. Type "yes" at the prompt. Verify on Polymarket UI.

---

## Open questions / decisions

### Blocking real-money smoke (operational, not engineering)

- Polymarket UI's CLOB API credential generation flow is unclear — probably solved via deterministic derivation from private key (the colleague's `derive_api_keys.py`).
- Geo-restriction confirmed: UK blocks Data API + presumably order submission. RTDS reads work. VPN routes (US East / Frankfurt / Tokyo) bypass; long-term VPS deployment in a non-blocked region needed.

### Empirical questions, will surface during real run

- **Latency budget**: assumed 1–2 seconds end-to-end (RTDS detection ~200ms, decision ~50ms, submission ~500ms, polling-thread fill catch ~700ms). Project brief tolerance was "15 seconds acceptable." Real-world numbers come from first run.
- **Tick size handling**: per-asset, fetched from CLOB orderbook on first encounter, defaults to $0.01 if fetch fails. Sub-penny markets ($0.001 ticks) covered correctly via cache. The default fallback is a known weakness for penny markets if the orderbook fetch fails.
- **Synthetic transaction_hash**: the kernel's `VenueFillEvent` doesn't expose the on-chain transaction hash. Bot synthesises one as `<client_order_id>:fill:<ts_ns>` for the FillRecorded event. Cross-referencing journal fills to PolygonScan needs manual `venue_order_id` lookup. Acceptable for $5-10 PoC; flag for v2.
- **+100 LOC growth in midas's `polymarket_sdk_signer.py`**: midas updated since vendoring; unknown whether the change is feature, fix, or refactor. Trust the snapshot for first run; investigate if signing fails for unexplained reasons.

### Deferred enhancements (not blocking)

- API key derivation at startup (vs manual paste in `.env`).
- RTDS-emit timestamp stored alongside receive timestamp for cleaner Data API reconciliation.
- Polymarket-apis library adoption (Python 3.12+ floor; currently using py-clob-client via the kernel).
- Multi-leader support (research-side ranking integration).

---

## Immediate next 3 TODOs

1. **PLAN.md sync + snapshot commit + Slack message to colleague.** ~10 minutes total. Marks engineering complete.
2. **Get Polymarket credentials into `.env`.** Private key from UI → `derive_api_keys.py` → fill in `.env`. Run a read-only auth check (fetch open orders, expect empty list). Confirm credentials work before any submission.
3. **Run the first real-money smoke.** Per `scripts/SMOKE_REAL.md`. VPN on, `MAX_REAL_ORDERS=1`, `REQUIRE_OPERATOR_CONFIRM=true`, `SIZING_USD=10`. Pick an active leader. Type "yes" at the prompt. Cross-check on Polymarket UI.

If step 3 succeeds, the PoC is operationally complete. Path forward: remove safety harness, increase capital, deploy to VPS for unattended operation, then multi-leader cohort copying.

---

## Specific call-outs

### Latency budget

| Step | Time |
|---|---|
| RTDS event detection (Polymarket → bot) | ~200ms |
| Classification + risk + sizing | ~50ms |
| Submission (bot → CLOB ACK) | ~500ms typical, 5s timeout |
| Fill notification (CLOB → bot via polling thread) | up to 700ms |
| **End-to-end leader-fill-to-bot-fill** | **~1–2s typical, 5–7s worst case** |

Project brief tolerance: 15 seconds. Bot is well under that, even with conservative settings. Latency is *not* the binding constraint for the PoC's leader profile (multi-day markets, infrequent fills). It would become binding only for high-frequency scalping leaders, which the bot deliberately doesn't target.

### Sizing rule

**Today**: fixed dollar amount per trade, set via `POLYMARKET_SIZING_USD`. Default $50, tuned to $5–10 for first runs. Doesn't preserve leader's conviction signal, doesn't react to bankroll, doesn't change with current price.

**Why this and not bankroll-proportional**: research-side decision. Sizing rules are a research problem. Execution's job is "given a sizing decision, place the order correctly, log it, reconcile it." Conflating the two makes copy-trading systems hard to debug, and a fixed-USD baseline lets you isolate execution issues from sizing issues.

**Future**: when research-side ships `leader_rankings.parquet` with bankroll estimates, sizing graduates to per-leader proportional. Math:

```
leader_fraction = leader_trade_usd / research.estimated_bankroll_usd
my_bet_usd = leader_fraction × my_strategy_capital
my_bet_usd = min(my_bet_usd, max_per_trade_cap, available_balance)
```

The caps matter — leader making an unusually huge trade (e.g. cashing out their whole book) shouldn't translate to bot betting its whole account.

### NegRisk handling

**Status: not handled. Real gap.**

NegRisk markets on Polymarket are multi-outcome markets where outcomes are mutually exclusive (e.g. "Who will win the 2028 election?" with 5 candidates, exactly one resolves to YES). Polymarket has special contract logic for these — the position structure differs from binary markets, redemption flow is different, splitting/merging via the relayer is meaningful.

The bot currently treats every position as binary YES/NO and uses `(condition_id, asset_id)` as the position key. For NegRisk markets:
- Position keying might need adjustment — multi-outcome markets have multiple `asset_id`s under one `condition_id`.
- The leader's exit logic could be ambiguous: "sell 30% of NO position on Candidate B" while also "buy 30% YES on Candidate C" might be a related rebalance, not two independent trades.
- Resolution is different — RTDS won't emit anything when the `condition_id` resolves; the position just becomes worthless or claimable.

For the PoC against directional traders on binary markets (sports, simple yes/no events), this isn't blocking. For any leader who routinely trades multi-outcome event markets (election, tournaments, multi-candidate prediction), NegRisk would need explicit handling.

**Action**: flag in PLAN.md as a known gap. Address only when a leader being mirrored makes meaningful NegRisk trades. Likely surfaces as a "the bot did something weird with this market" finding during operation, then becomes a real ticket.

### Future `leader_rankings.parquet` — fields the bot needs

The interface is file-based (research writes, execution reads, never code-imported). The bot reads on startup and refreshes periodically (e.g. every 4 hours). If file is missing or stale, falls back to hardcoded `POLYMARKET_LEADER_ADDRESS` from env.

**For sizing**:
- `proxy_address` (lowercase 0x, 42 chars) — canonical leader ID. Match against RTDS `proxyWallet`.
- `estimated_bankroll_usd` — for proportional sizing math. Method opaque to execution; research decides.
- `bankroll_method` (e.g. `"rolling_max_30d_v1"`) — execution detects when research's algorithm changes and can react if needed.

**For ranking / selection**:
- `rank_score` and/or `rank_position` — opaque scalar(s); just used to pick "trader N from the list" or "all traders above threshold X."
- `last_updated_utc` — execution refuses to act on stale rankings (e.g. >24h old).

**For risk-side input** (lets execution be smarter without research-side calls):
- `typical_position_count` — sanity check unusual activity ("leader normally has 5 positions, suddenly has 50").
- `typical_hold_duration_hours` — fallback for exit-tracker timeout.
- `30d_winrate`, `30d_pnl_usd` — journal context only, NOT for halting trades (research has already filtered for these).

**For pricing-mode override** (per-leader, when adopted):
- `pricing_mode_recommended` (`leader_fill` | `current_book`) — based on trader edge type (information vs speed). Execution respects this if the file provides it; defaults to global config otherwise.
- `maker_taker_ratio_30d` — supporting evidence for the recommendation. High-taker traders are speed-based (use current_book); high-maker traders are conviction-based (use leader_fill).

**Format**: Parquet (or JSON), atomically written (write to `.tmp`, rename) so execution doesn't read partial files. Path: `research/output/leader_rankings.parquet`.

---

## Cross-refs / open items

- **Interface contract** — captured in `decisions/0001-leader-rankings-schema.md`. Mapping from data-side `traders.parquet` columns to exec-side requirements is mostly clean except for time windows: exec wants 30d-rolling, data currently ships lifetime. Real gap — see ADR.
- **NegRisk** — gap on both sides. Data flags it via `phantom_position_score` for ranking; exec doesn't yet handle position keying or rebalance detection. Composite design needed before NegRisk-active leaders are mirrored. Tracked in `polymarket-copytrade/TODO.md` v2 work.
- **Bankroll mismatch** — exec's interface needs point-in-time bankroll for honest sizing, data's `est_bankroll_usd_30d_max_approx` is lifetime peak. Phase 5 plan already calls for point-in-time bankroll computation; integration depends on it.
- **Sizing rule division of labour** — explicitly documented above: research decides bankroll, execution applies fraction × strategy capital with caps. Belongs in `03-system-design.md`.
- **End-to-end design** — see `03-system-design.md`.
