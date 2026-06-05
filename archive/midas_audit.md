# Polymarket Code Audit — `midas/` and Adjacent
> Hub: [[COWORK]]


Read-only audit conducted 2026-05-06. No files modified, no code executed beyond `find` / `grep` / `cat`.

---

## 1. Directory tree of `midas/` (depth 3, files only)

```
midas/
├── .env.example
├── README.md
├── pytest.ini
├── requirements.txt
├── examples/
│   ├── __init__.py
│   ├── minimal_executor_wiring.py        (137 lines)
│   └── run_executor_from_env.py          (2,081 lines)
├── executor/
│   ├── __init__.py                       (230 lines, public re-exports)
│   ├── executor_service.py               (1,116)
│   ├── fake_venue_adapter.py             (210)
│   ├── journal.py                        (1,192)
│   ├── planner.py                        (708)
│   ├── polymarket_adapter.py             (869)
│   ├── polymarket_clob_client.py         (309)
│   ├── polymarket_sdk_signer.py          (212)
│   ├── risk.py                           (1,219)
│   ├── slug_structural_arb.py            (958)
│   ├── state_machine.py                  (860)
│   └── venue.py                          (217)
├── harvester/
│   └── __init__.py                       (empty — stub package)
└── tests/
    ├── __init__.py
    ├── executor/                         (failure_modes/, helpers/, integration/,
    │                                      performance/, property/, replay/, unit/
    │                                      — ~22 test modules)
    ├── harvester/__init__.py             (empty)
    └── helpers/{factories,stubs}.py
```

Total in-scope source (excluding tests): ~10,300 LOC across `executor/` + `examples/`. `harvester/` is a placeholder.

---

## 2. Dependencies — per file

Throughout: **no file in `midas/executor/` directly imports `web3` or `eth_account`**. All EVM signing is delegated to `py_clob_client`, and only [polymarket_sdk_signer.py](midas/executor/polymarket_sdk_signer.py) imports it (lazily, inside a function).

### `executor/executor_service.py`
- stdlib: `collections`, `dataclasses`, `enum`, `time.perf_counter`, `typing`
- third-party: none
- local: `.journal`, `.planner`, `.risk`, `.state_machine`, `.venue`

### `executor/polymarket_adapter.py`
- stdlib: `dataclasses`, `enum`, `hashlib.blake2b`, `time.perf_counter`, `typing` (incl. `Protocol`)
- third-party: none
- local: `.state_machine`, `.venue`
- **Polymarket SDK: defines its own `PolymarketCLOBClient` Protocol — does not import `py_clob_client`.**

### `executor/polymarket_clob_client.py`
- stdlib: `dataclasses`, `json`, `random`, `socket`, `time`, `typing`, `urllib.error`, `urllib.parse`, `urllib.request`
- third-party: none (uses bare `urllib` — no `requests`/`httpx`)
- local: `.polymarket_adapter`

### `executor/polymarket_sdk_signer.py`
- stdlib: `dataclasses`, `inspect`, `typing`
- third-party: **`py_clob_client.client.ClobClient`, `py_clob_client.clob_types.OrderArgs`** — imported lazily inside builder function (~lines 80–94).
- local: `.polymarket_adapter`

### `executor/state_machine.py`
- stdlib: `dataclasses`, `enum`, `typing`. No third-party, no Polymarket coupling.

### `executor/venue.py`
- stdlib: `dataclasses`, `enum`, `hashlib.blake2b`, `typing`. local: `.state_machine`. Generic.

### `executor/planner.py`
- stdlib: `dataclasses`, `enum`, `typing`. local: `.state_machine`. Generic.

### `executor/journal.py`
- stdlib: `collections`, `dataclasses`, `datetime`, `enum`, `itertools`, `json`, `pathlib`, `queue`, `threading`, `uuid`. local: `.planner`, `.state_machine`, `.venue`. Generic.

### `executor/risk.py`
- stdlib: `collections`, `dataclasses`, `enum`, `typing`. local: `.planner`, `.state_machine`, `.venue`. Generic.

### `executor/slug_structural_arb.py`
- stdlib: `concurrent.futures`, `dataclasses`, `datetime`, `json`, `re`, `typing`, `urllib.parse`, `urllib.request`
- third-party: none
- local: none (only references `executor` types via duck typing)
- **Talks to Polymarket REST (`gamma-api`, `clob`) directly via `urllib`.**

### `executor/fake_venue_adapter.py`
- stdlib: `dataclasses`, `typing`. local: `.state_machine`, `.venue`. Generic test double.

### `examples/run_executor_from_env.py`
- stdlib: `concurrent.futures`, `csv`, `dataclasses`, `datetime`, `json`, `os`, `pathlib`, `re`, `sys`, `time`, `typing`, `urllib`
- third-party: optional `dotenv` (per `requirements.txt`)
- local: re-exports from `executor/`

### `examples/minimal_executor_wiring.py`
- stdlib only. local: `executor/`.

### `requirements.txt` (verbatim)
```
aiohttp>=3.9.0
websockets>=12.0
python-dotenv>=1.0.0
py-clob-client>=0.18.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```
**Note:** `aiohttp` and `websockets` are declared but **not actually imported** anywhere in `midas/executor/` or `midas/examples/`. They appear to be reserved for the unimplemented harvester. The executor's transport is bare `urllib`.

---

## 3. Entry points

| File | Line | What it does |
|------|------|-------------|
| [examples/run_executor_from_env.py](midas/examples/run_executor_from_env.py#L2080) | 2080 | Loads `.env` → builds `PolymarketCLOBHttpClient` + `PyClobClientOrderSigner` → wires `ExecutorService` + journal + risk → runs `slug_structural_arb` strategy loop. Live/paper toggle via `EXECUTOR_LIVE_TRADING_ENABLED`. |
| [examples/minimal_executor_wiring.py](midas/examples/minimal_executor_wiring.py#L136) | 136 | Educational harness: wires `ExecutorService` against `FakeVenueAdapter`. No network. |
| `tests/executor/test_*.py` | various | Several test files have `if __name__ == "__main__": pytest.main(...)` style runners. |

**No FastAPI / Flask / argparse-CLI / typer scripts.** No HTTP server. No background daemon. The only "service" is the `run_executor_from_env.py` loop.

---

## 4. Polymarket touch points (per file)

### [executor/polymarket_adapter.py](midas/executor/polymarket_adapter.py) — **ORDER_SUBMIT** (+ light DATA_API for polling)
Defines `PolymarketAdapterConfig` (holds private_key + api_key/secret/passphrase), `PolymarketOrderRequest`, `PolymarketSubmitResponse`, `PolymarketOrderUpdate`, the `PolymarketCLOBClient` `Protocol` ([line 127](midas/executor/polymarket_adapter.py#L127)), and the concrete `PolymarketVenueAdapter` ([line 156](midas/executor/polymarket_adapter.py#L156)) that implements the executor's `VenueAdapter` interface. Handles submit/cancel/poll, idempotency keys (blake2b client-order-id factory), ambiguous-submit detection, and event normalization. **Does not talk HTTP itself** — delegates to a `PolymarketCLOBClient`.

### [executor/polymarket_clob_client.py](midas/executor/polymarket_clob_client.py) — **ORDER_SUBMIT** + **DATA_API**
Concrete HTTP transport (`PolymarketCLOBHttpClient`) using stdlib `urllib`. Adds `X-API-KEY` / `X-API-SECRET` / `X-API-PASSPHRASE` headers. Calls `submit_path`, `cancel_path`, `updates_path`, `open_orders_path` (all configurable). Order signing is delegated to an injected `OrderSigner` callable.

### [executor/polymarket_sdk_signer.py](midas/executor/polymarket_sdk_signer.py) — **WALLET**
`PyClobClientOrderSigner` wraps `py_clob_client.client.ClobClient` (lazy-imported). `PyClobClientOrderSignerConfig` holds `api_url`, `chain_id` (default 137), `signature_type` (default 1), `funder`, `maker`. `build_py_clob_client_signer()` is the factory used by `run_executor_from_env`. This is the only place the SDK is touched.

### [executor/slug_structural_arb.py](midas/executor/slug_structural_arb.py) — **PRICE_FETCH** + **DATA_API**
Strategy logic. Fetches markets from `gamma-api.polymarket.com` and orderbooks from CLOB via `urllib`. Detects YES/NO mispricing across paired slugs and emits `Opportunity` records that the executor consumes. Read-only against Polymarket; no signing here.

### [executor/executor_service.py](midas/executor/executor_service.py), [planner.py](midas/executor/planner.py), [risk.py](midas/executor/risk.py), [journal.py](midas/executor/journal.py), [state_machine.py](midas/executor/state_machine.py), [venue.py](midas/executor/venue.py), [fake_venue_adapter.py](midas/executor/fake_venue_adapter.py) — **OTHER (venue-agnostic execution framework)**
No Polymarket-specific code. They are the generic execution engine: lifecycle state machine, order/package planner, risk caps & kill switches, event-sourced journal with replay/recovery, abstract `VenueAdapter` protocol, and a deterministic in-memory fake venue for tests.

### [examples/run_executor_from_env.py](midas/examples/run_executor_from_env.py) — **WALLET** + **ORDER_SUBMIT** + **DATA_API**
Single biggest file (2,081 LOC). Reads all `POLYMARKET_*` env vars, builds the SDK signer + HTTP client + adapter + executor + journal + risk, then runs the strategy loop. Has its own market-data polling helpers and CSV/JSONL log emitters.

### [examples/minimal_executor_wiring.py](midas/examples/minimal_executor_wiring.py) — **OTHER**
No Polymarket calls. Wires fake adapter only.

### **No WEBSOCKET file anywhere in midas/.** All Polymarket interaction is REST/HTTP polling.

---

## 5. Wallet/auth setup — present vs. missing

### Present
- **EOA private key handling** via `POLYMARKET_PRIVATE_KEY` env → passed into `py_clob_client.ClobClient(key=...)`. ([polymarket_sdk_signer.py](midas/executor/polymarket_sdk_signer.py) lazy import @ ~line 80–94.)
- **Proxy-wallet (funder) support** via `POLYMARKET_FUNDER` env + `signature_type=1` — both are passed through to `ClobClient`. The Polymarket UI proxy wallet (signature type 1 / 2) is supported insofar as `py_clob_client` supports it.
- **CLOB L2 HTTP auth** — `X-API-KEY`, `X-API-SECRET`, `X-API-PASSPHRASE` headers added by `PolymarketCLOBHttpClient._headers()` ([polymarket_clob_client.py](midas/executor/polymarket_clob_client.py)).
- **`.env.example`** documents all eight env vars (`POLYMARKET_API_URL`, `POLYMARKET_GAMMA_API_URL`, `POLYMARKET_PRIVATE_KEY`, `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_PASSPHRASE`, `POLYMARKET_CHAIN_ID`, `POLYMARKET_SIGNATURE_TYPE`, `POLYMARKET_FUNDER`).

### Missing
- **No `create_api_key` / `derive_api_key` flow.** The user is expected to manually generate API key+secret+passphrase in the Polymarket UI and paste them into `.env`. There is no L1-auth → L2-key bootstrapping in code.
- **No direct `eth_account.Account` usage.** All signing is hidden behind `py_clob_client`. Fine if the SDK is enough; a pain if you ever need a custom signature path (e.g. Safe wallet, hardware wallet, off-machine signer).
- **No proxy-wallet derivation from EOA.** The funder address must be supplied via env; nothing computes the deterministic Polymarket proxy address from the private key.
- **No API-key expiry/rotation handling.** Polymarket API keys can be revoked/rotated; if they go stale the executor will just receive 401s and treat them as transient errors. (`polymarket_clob_client.py` retries on transient errors generically.)
- **No nonce / rate-limit / replay-protection logic** beyond what the SDK provides.
- **No `.env` loader** is wired in `run_executor_from_env.py` itself — it relies on env being already set (or `python-dotenv` invoked externally). `dotenv` is in `requirements.txt` but I see no `load_dotenv()` call in scope.

---

## 6. What's reusable for a new `polymarket/execution/` module

### REUSE_AS_IS

- **[executor/venue.py](midas/executor/venue.py)** — clean `VenueAdapter` Protocol, `SubmitOrderResult`, `CancelOrderResult`, `ClientOrderIdFactory`. Zero Polymarket coupling. Copy verbatim.
- **[executor/state_machine.py](midas/executor/state_machine.py)** — order/package state machine, event types. Pure domain logic, fully generic.
- **[executor/fake_venue_adapter.py](midas/executor/fake_venue_adapter.py)** — useful as a test double in the new module's test suite.
- **[executor/polymarket_clob_client.py](midas/executor/polymarket_clob_client.py)** — only ~310 lines, well-isolated HTTP client with retry/backoff. Reuse as the transport layer of the new module. Watch the bare-`urllib` choice — you may want to swap to `httpx` for async, but the surface area is small.
- **[executor/polymarket_sdk_signer.py](midas/executor/polymarket_sdk_signer.py)** — concise `py_clob_client` wrapper. Reuse as the signer.

### REUSE_WITH_REFACTOR

- **[executor/polymarket_adapter.py](midas/executor/polymarket_adapter.py)** — most of the value is here. Submit/cancel/poll plumbing is reusable, but it's bound to the executor's `VenueAdapter` interface. If your new `polymarket/execution/` module has a different surface (e.g. async, or a thinner "place_limit / cancel / get_book" API), the adapter needs an interface adjustment but the body of submit/cancel/normalization transfers cleanly. The `PolymarketCLOBClient` Protocol ([line 127](midas/executor/polymarket_adapter.py#L127)) is the right seam to lift out.
- **[executor/journal.py](midas/executor/journal.py)** — event-sourced journal is high-quality but tightly typed to the executor's event vocabulary (`OrderIntent`, `VenueOrderAck`, etc.). If the new module reuses those events, take it. If you redesign events, port the storage/replay infrastructure but rewrite the schema layer.
- **[executor/risk.py](midas/executor/risk.py)** — generic in shape, but threshold semantics and kill-switch wiring assume the executor's package model. Worth reusing the structure; reset the configuration.
- **[executor/planner.py](midas/executor/planner.py)** — structurally generic, but the multi-leg `ExecutionPlan` model is shaped around structural-arb opportunities. If the new module is single-leg market-making/HFT, the planner is overweight — borrow ideas, write a thinner version.

### DON'T_REUSE

- **[executor/executor_service.py](midas/executor/executor_service.py)** — 1,100 lines of orchestration glue specific to the executor's package/leg/journal model. Patterns are good (single-writer hot path, latency tagging) but lifting the file wholesale will pull in all of `planner.py`, `journal.py`, `risk.py`, `state_machine.py` whether you want them or not. Read for inspiration; don't import.
- **[executor/slug_structural_arb.py](midas/executor/slug_structural_arb.py)** — a strategy, not an execution module. Belongs alongside the strategy code, not in `polymarket/execution/`.
- **[examples/run_executor_from_env.py](midas/examples/run_executor_from_env.py)** — a 2,000-line wiring script. Useful as a reference for env-var loading and HTTP+SDK assembly; not a building block.

**Net assessment:** if you want a clean "thin Polymarket execution module" (submit/cancel/book/sign/wallet), the kernel you actually need is `polymarket_adapter.py` + `polymarket_clob_client.py` + `polymarket_sdk_signer.py` + `venue.py` + `state_machine.py` — roughly **2,500 LOC of useful, well-isolated code**. Everything else is either generic plumbing you may already have, or strategy/orchestration above the execution layer.

---

## 7. Files outside `midas/` referencing Polymarket

**No `from midas` / `import midas` anywhere outside `midas/`** (verified with `grep -rn`). The package is fully isolated.

But there *are* two other Polymarket-related code locations in the repo, which the user should be aware of since they may overlap with the new module:

### `polymarket-copy/` (top-level)
A separate, parallel Polymarket research project with its own `pyproject.toml`, `uv.lock`, and a `.venv/`. Notable files:
- `polymarket-copy/data_infra/gamma.py` — Gamma API client (`GAMMA_BASE = "https://gamma-api.polymarket.com"`).
- `polymarket-copy/data_infra/duck.py` — DuckDB integration.
- `polymarket-copy/scripts/build_markets_table.py` — references `condition_id`, builds a markets table.
- `polymarket-copy/scripts/inspect_seed.py`.

This appears to be a separate research workstream (data harvesting / DuckDB), **not** an execution module. No `py_clob_client` usage spotted. **Out of scope for the new execution module**, but worth knowing it exists — the new module shouldn't duplicate `gamma.py`.

### `topics/prediction-markets/` (visible via worktree at `.claude/worktrees/sweet-mccarthy-18618b/`)
A research data pipeline:
- `config.py` — `polymarket.db` SQLite path, Falcon agent IDs.
- `run_collection.py` — argparse CLI, "Polymarket research data pipeline".
- `run_whale_discovery.py`.
- `collectors/{markets,trades,candlesticks}.py` — fetches markets, trades, candles; writes to SQLite. Heavy use of `condition_id`, `proxy_wallet` field on trade rows.

Again: data collection, not execution. **Out of scope for the new module**, but a likely consumer of any market-data API the new module exposes.

Neither of these locations imports from `midas/`, and `midas/` does not import from either of them.

---

## 8. Open questions

1. **Is the executor expecting a proxy wallet or an EOA at runtime?** `POLYMARKET_FUNDER` is documented as required in `.env.example`, and `signature_type=1` is the default — both imply proxy wallet. But `funder` is `Optional` in `PyClobClientOrderSignerConfig`. If you launch with funder unset, behaviour depends on `py_clob_client`'s defaults. Worth confirming against the SDK before live trading.

2. **Where do API keys come from at runtime?** The `.env` flow assumes the user has manually run the Polymarket UI's "create API key" step. There is no `create_api_key` call in the codebase. If the new module is meant to be self-bootstrapping, this gap needs to be closed.

3. **Why are `aiohttp` and `websockets` in `requirements.txt` but unused in code?** Strongly suggests the harvester was *intended* to be async/WS-based and was never written. The executor itself is pure synchronous `urllib`. The new module should decide upfront: sync HTTP polling like the executor, or async + WS like the harvester was meant to be?

4. **Does `run_executor_from_env.py` actually call `dotenv.load_dotenv()`?** I didn't find one. Either the launch wrapper does it externally, or `.env` is sourced via shell. Worth verifying — silent fallback to empty env would log misleading errors.

5. **Order-update polling cadence and freshness model.** The adapter polls `updates_path` with a sequence cursor. What's the default poll interval, and is it tight enough for the strategies' edge windows? Not obvious from a code skim.

6. **Ambiguous-submit recovery.** `polymarket_adapter.py` flags `ambiguous_submit=True` on timeout. Reading the executor service, the kill-switch path looks manual rather than automated. Confirm whether the new module should auto-cancel-on-ambiguity or halt and await operator.

7. **Relationship to `polymarket-copy/` and `topics/prediction-markets/`.** Are these abandoned, parallel, or feeders into the new module? If `polymarket-copy/data_infra/gamma.py` is canonical, the new execution module's market-data layer should import from it rather than re-implementing in `slug_structural_arb`-style `urllib` calls.

8. **Harvester intent.** `harvester/` is empty but `.env.example` has six `HARVESTER_*` config vars (slugs, entry-window hours, price threshold, max notional, live toggle, log paths). Is the harvester something the new `polymarket/execution/` module is expected to cover, or a separate strategy that will plug into it?
