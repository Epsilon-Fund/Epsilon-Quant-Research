# Polymarket Execution

The live/paper execution stack for the Polymarket research programme. Its first job is a **copy-trading mirror**: watch one leader wallet's fills in real time over Polymarket's RTDS WebSocket and mirror them with fixed USD sizing and mirror-exit logic. Around that core sit a measurement-grade **maker engine** (passive quoting with NegRisk-aware inventory and resolution handling), layered **risk controls**, and an append-only **JSONL journal** of every signal, order, halt, and reconnection.

## How an order happens

```
RTDS WebSocket → watcher → signal → risk → mirror_engine → _kernel/polymarket_adapter → CLOB
```

Each stage has a single responsibility and the boundaries are enforced: risk is pure functions of `(config, state, candidate_order)` — it cannot reach into the mirror; the venue adapter is the only thing that talks to the exchange.

## Safety model

- **Capital is gated by environment variable** (`POLYMARKET_MAX_CAPITAL_USD`), never raised in code.
- **Kill switch**: a file at `POLYMARKET_KILLSWITCH_PATH` is checked before every order submission — if it exists, no new orders go out (cancels still can).
- **Risk stack**: per-order caps, daily-loss limit, max open positions, and price-deviation checks, each its own module under `risk/`.
- **Everything is journaled**: every order submitted, signal received, risk halt, and reconnection writes a JSONL event via `journal/jsonl_writer.py`. No exceptions.
- **First-real-money runs are operator-confirmed**: the smoke runbook caps the session at one real order with explicit confirmation required.

## Module map

| Module | Responsibility |
|---|---|
| `watcher/` | RTDS WebSocket subscription to the leader's fills; reconnect handling |
| `signal/` | Turns observed leader fills into candidate mirror orders |
| `risk/` | Pure-function gate: caps, daily loss, kill switch, max positions, price deviation |
| `mirror/` | Mirror engine, CLOB HTTP client + signer, order book, market metadata, real venue adapter |
| `maker/` | Passive maker engine: event calendar, NegRisk inventory tracker, resolution/redemption handler, maker CLI ([maker/README.md](maker/README.md)) |
| `journal/` | Append-only JSONL event log (orders, signals, halts, reconnects) |
| `_kernel/` | Vendored execution kernel (venue adapters, order state) — **frozen**, never edited here |
| `scripts/` | Runbooks + utilities: [SMOKE_REAL.md](scripts/SMOKE_REAL.md) (first real-money smoke), [SMOKE_MAKER.md](scripts/SMOKE_MAKER.md), position-rebuild verifier |
| `tests/` | Failure-mode-first test suite (happy path / edge / failure per module) + read-only API probes with written findings ([WS](tests/probes/WS_PROBE_FINDINGS.md), [NegRisk](tests/probes/NEGRISK_FINDINGS.md)) |
| `cli.py`, `config.py` | Entry point and env-driven configuration |

## Why it can be trusted

The full execution test suite is **256 tests green**, with the maker + NegRisk dependency slice at 67. Tests never hit the real Polymarket API — order submission is exercised against the kernel's fake venue adapter. Where exchange behaviour was unclear, it was settled by read-only probe scripts whose findings are committed next to the tests. Conventions match the research side byte-for-byte (lowercase `0x` addresses, UTC timestamps, `condition_id` as market key, `(transaction_hash, log_index)` as trade key) so research artifacts plug in without translation.

## Run it

```bash
python -m polymarket.execution.cli        # main entry point
pytest polymarket/execution/tests -q      # full suite, no network
```

Development rules live in [CLAUDE.md](CLAUDE.md); current architecture and milestones in [PLAN.md](PLAN.md); the first-fill runbook in [scripts/SMOKE_REAL.md](scripts/SMOKE_REAL.md).

Vault hub: brain/COWORK.md · brain/POLYMARKET_BRAIN.md
