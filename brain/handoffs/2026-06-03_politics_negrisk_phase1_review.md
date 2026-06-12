---
title: "Handoff — Politics NegRisk Phase-1 review + second Codex pass"
tags: [handoff, mm, politics, negrisk, exec, phase1, codex]
created: 2026-06-03
status: ready for Opus review + second Codex pass
---

# Handoff — Politics NegRisk Phase-1 review

> Hub: [[strat_market_making]] · [[COWORK]] · [[2026-06-03_politics_negrisk_live_loop]]

## Read first (in order)

1. `brain/CODEX.md`
2. `brain/TODO.md` § MM
3. `brain/COWORK.md`
4. `polymarket/research/notes/market_making/mm_politics_negrisk_live_loop_design.md` — design decisions + Phase-2 pre-registered success criteria
5. `polymarket/research/notes/market_making/mm_politics_negrisk_accounting_findings.md` — the 2,290 bps edge anchor
6. `polymarket/execution/PLAN.md`
7. `polymarket/execution/CLAUDE.md`

## What was built (Phase 1)

All work is in `polymarket/execution/maker/`. Five modules landed:

- `maker_engine.py` — passive two-sided quoting loop (757 LOC)
- `resolution_handler.py` — Gamma poll + Data API redeemable poll + web3.py self-redemption
- `negrisk_inventory.py` — polls `/activity?type=SPLIT,MERGE,REDEEM,CONVERSION`, JSONL persistence
- `event_calendar.py` — stdlib YAML parser + proximity check
- `politics_events.yaml` — real upcoming events (June–July 2026 elections, SCOTUS days, FOMC)

Plus NegRisk signature fix landed in `mirror/real_venue_adapter.py` (Gamma cache) and `mirror/clob_signer.py` (`_neg_risk` → `PartialCreateOrderOptions`). Tests exist in `tests/maker/`.

## What's solid (do not rewrite)

- **NegRisk signature fix** — correct, tested, unblock lifted
- **`resolution_handler.py`** — correct: Gamma poll, Data API positions, web3.py redemption, non-blocking failure, proper journal events
- **`negrisk_inventory.py`** — correct: polls the right endpoint, JSONL persistence, restart replay, thread-safe
- **`event_calendar.py` + `politics_events.yaml`** — correct, stdlib-only, real events populated

## What needs fixing before Phase-2 smoke (three items)

### Fix 1 — Fill detection in `maker_engine.py` (CRITICAL)

`process_fills_once` reads `FILL_RECORDED` journal events and calls `_coid_from_fill_event` to match them to live quotes. But `FillRecorded` (in `journal/events.py`) has **no `client_order_id` field** — only `transaction_hash, condition_id, asset_id, side, size, price, proxy_wallet`. So `_coid_from_fill_event` always returns `None`. `MakerFillTelemetry` is never logged, and filled quotes are never removed from `_quotes`.

**Fix:** replace the journal-based fill detection with a poll of `data-api/trades?user=<funder>&conditionId=<market>`. Match fills as: asset_id matches our market, `proxy_wallet == config.funder`, `ts >= session_start`, price matches one of our live quote prices. When matched, log `MakerFillTelemetry` and remove the quote from `_quotes`. The `DataApiTradeClient.get_trades()` already exists in `maker_engine.py` — just change what `process_fills_once` does with it.

Also remove or fix `_coid_from_fill_event` — the `:fill:` pattern is dead code.

### Fix 2 — `_redeem_amounts` for 3+ outcome markets (in `resolution_handler.py`)

```python
if outcome_index not in (0, 1):
    raise ValueError(...)
amounts = [0, 0]
```

Raises for any NegRisk market with 3+ outcomes. Fix:

```python
if outcome_index < 0:
    raise ValueError(f"outcomeIndex must be >= 0 (got {outcome_index})")
amounts = [0] * (outcome_index + 1)
amounts[outcome_index] = raw
```

### Fix 3 — Wire up a CLI entry point

No way to actually run the maker loop. Need a minimal `maker/cli.py` (or extend `cli.py` at the execution root) that wires: `NegRiskInventoryTracker` + `ResolutionHandler` + `MakerEngine`, starts all three background threads, handles KeyboardInterrupt cleanly (stops threads, cancels open quotes), and respects the same `MAX_REAL_ORDERS` / `REQUIRE_OPERATOR_CONFIRM` env vars as the copytrade bot.

## Nice-to-fix (not blocking Phase 2)

- `price_after_60s` window is ±10s. Politics markets are illiquid; widen to "first trade at or after fill_ts + 60s" rather than "nearest trade to the 60s mark."
- `_top_maker_rank_at_fill` is documented as an approximation (order of fills in a 5s window ≠ queue position) — add a comment so it's clear this is a proxy metric.

## Codex prompt for the second pass

Paste this into a Codex session. Read the five files above first.

```markdown
Before doing anything else, read:
1. `brain/CODEX.md`
2. `brain/TODO.md`
3. `brain/COWORK.md`
4. `brain/POLYMARKET_BRAIN.md`
5. `polymarket/execution/PLAN.md`
6. `polymarket/execution/CLAUDE.md`
7. `polymarket/execution/maker/maker_engine.py`
8. `polymarket/execution/maker/resolution_handler.py`
9. `polymarket/execution/journal/events.py`

---

## Task: Politics NegRisk Phase-1 — three targeted fixes

### Fix A — Fill detection (CRITICAL, in `maker_engine.py`)

`process_fills_once` tries to match `FILL_RECORDED` journal events to live quotes using
`_coid_from_fill_event`, but `FillRecorded` in `journal/events.py` has no `client_order_id`
field — so the match always returns `None` and `MakerFillTelemetry` is never logged.

Replace `process_fills_once` with a poll of `data-api/trades` for our funder wallet:

1. Call `self._data_client.get_trades(condition_id)` — already exists.
2. Filter to rows where `proxy_wallet == config.funder` (lowercase match), `ts >= session_start`,
   and `asset_id == market.asset_id`.
3. For each unseen row (dedup on `transactionHash`), check if `price` matches any live quote
   within half a tick (`abs(quote.price - fill_price) <= tick_size / 2`).
4. If matched: add tx to `_seen_fill_txs`, call `_log_fill_telemetry` with the row data,
   pop the matched side from `_quotes`.
5. If no live quote matches price, it is still our fill (just a taker fill on a resting order
   that already left our quote state) — log `MakerFillTelemetry` with `top_maker_rank_at_fill=None`.

Also delete `_coid_from_fill_event` — it is dead code.

`_log_fill_telemetry` currently takes a journal `event` dict. Change its signature to accept
the raw trade row dict from the Data API directly. Update the call sites.

---

### Fix B — `_redeem_amounts` for 3+ outcome markets (in `resolution_handler.py`)

Current code:
```python
if outcome_index not in (0, 1):
    raise ValueError(f"outcomeIndex must be 0 or 1 (got {outcome_index})")
amounts = [0, 0]
amounts[outcome_index] = raw
```

This raises for NegRisk markets with 3+ outcomes (any market with >2 candidates).

Replace with:
```python
if outcome_index < 0:
    raise ValueError(f"outcomeIndex must be >= 0 (got {outcome_index})")
amounts = [0] * (outcome_index + 1)
amounts[outcome_index] = raw
```

---

### Fix C — CLI entry point (new file `polymarket/execution/maker/cli.py`)

A minimal `__main__`-style entry point that:

1. Reads env vars: `POLYMARKET_MAKER_CONDITION_ID` (required), plus all vars in
   `MakerEngineConfig.from_env()` and `ExecutionConfig`.
2. Instantiates: `NegRiskInventoryTracker`, `ResolutionHandler`, `MakerEngine`
   (wiring `ResolutionHandler` as the `resolution_state` and `NegRiskInventoryTracker`
   as the `inventory`).
3. Starts all three background threads.
4. Waits for `KeyboardInterrupt` or `SIGTERM`.
5. On shutdown: calls `maker_engine.cancel_all(reason="shutdown")`, then `stop()` on all
   three components in reverse order.
6. Respects the same safety env vars as the copytrade `cli.py`:
   `MAX_REAL_ORDERS`, `REQUIRE_OPERATOR_CONFIRM`.
7. Logs startup and shutdown to the journal.

Wire it into `polymarket/execution/__main__.py` with a `--mode maker` flag (or equivalent)
so it can be launched as `uv run python -m polymarket.execution --mode maker`.

---

### Nice-to-fix (do these if time allows)

- In `DataApiTradeClient.price_after_60s`: change from "nearest trade within ±10s of the 60s
  mark" to "first trade at or after `fill_ts + 60s`, no upper bound." Politics markets are
  illiquid; the ±10s window produces too many `None` values.

- In `MakerEngine._top_maker_rank_at_fill`: add a docstring comment noting this is a proxy
  (fill order in a 5s window, not true queue position).

---

### What NOT to change

- `resolution_handler.py` except Fix B above.
- `negrisk_inventory.py` — correct as-is.
- `event_calendar.py` / `politics_events.yaml` — correct as-is.
- `mirror/clob_signer.py` and `mirror/real_venue_adapter.py` NegRisk path — correct as-is.
- `_kernel/` — frozen, never edit.

---

### Output

- Updated `maker_engine.py` with Fix A.
- Updated `resolution_handler.py` with Fix B.
- New `maker/cli.py` for Fix C.
- Updated `__main__.py` if wiring there.
- Updated tests in `tests/maker/` to cover the new fill-detection path and the 3+ outcome
  redemption case.
- Update `polymarket/execution/PLAN.md` to record the fix decisions.
- Do NOT write handoff notes — Cowork handles that.
```

## After second pass

- Run a smoke test: `MAX_REAL_ORDERS=0 REQUIRE_OPERATOR_CONFIRM=false` on a live politics
  NegRisk market (no orders submitted). Confirm journal shows `MakerQuotePlaced`,
  `MakerQuoteSkipped`, and no errors.
- If fills happen (unlikely with `MAX_REAL_ORDERS=0`), confirm `MakerFillTelemetry` appears.
- If smoke passes → Phase 2 is ready to deploy at `MAX_REAL_ORDERS=2`, `SIZING=1`.
