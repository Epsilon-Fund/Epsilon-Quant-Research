# polymarket/execution/ — Claude Code rules

> Hub: [[COWORK]] · [[POLYMARKET_BRAIN]]

This module is a Polymarket copy-trading execution bot. It mirrors
ONE hardcoded trader's fills with fixed USD sizing and
mirror-exit logic. The trader address is set via env var
`POLYMARKET_LEADER_ADDRESS`.

## Architecture

Data flow:

  RTDS WebSocket → watcher → signal → risk → mirror_engine → _kernel/polymarket_adapter → CLOB

Each subfolder has a single responsibility. Do not blur boundaries.
If you find yourself wanting to import from `mirror/` inside `risk/`,
something is wrong — risk should be pure functions of (config, state,
candidate_order).

## Kernel vendoring status

_kernel/ was vendored from midas/executor/ on 2026-05-06.
Upstream midas has since changed structurally
(state_machine.py merged into venue.py, several files grew
significantly). Do not attempt to re-vendor without a planned
migration — it is not a clean diff. The vendored kernel is
the canonical version for this module. Treat it as frozen.

## Hard rules

1. Never edit anything inside `_kernel/`. It is vendored from
   midas/executor/. If you think it needs a change, stop and ask.

2. Never write code that touches `midas/`. Different module, different
   owner.

3. Lowercase 0x-prefixed addresses everywhere. UTC timestamps everywhere.
   `condition_id` is the canonical market key. `(transaction_hash, log_index)`
   is the unique trade key. These match the research repo for future
   interop.

4. Real money trading is gated by `POLYMARKET_MAX_CAPITAL_USD` env var.
   Default in `.env.example` is 100. Never raise this in code.

5. Every order submitted, every signal received, every risk halt, every
   reconnection — log to JSONL via `journal/jsonl_writer.py`. No
   exceptions.

6. The kill-switch file path is `POLYMARKET_KILLSWITCH_PATH`.
   `risk/kill_switch.py` is checked before every order submission. If
   the file exists, no new orders go out (existing orders may still
   be cancelled).

7. Minimal targeted changes. No refactors. No renames. No "while I'm
   here" cleanup.

8. UK-based manual operation is acceptable. Daily timeframe assumptions
   from research-side do NOT apply here — this is event-driven on the
   leader's fills.

9. Sizing for the PoC is a fixed USD constant
   (POLYMARKET_SIZING_USD). Do not implement leader-proportional,
   Kelly, or any other sizing logic. When research ships
   leader_rankings.parquet, sizing graduates to its own module
   — but not before.

## Testing

- Every new module gets a corresponding test file. Coverage isn't
  the goal; *failure-mode coverage* is. Each test file should
  exercise: happy path, one edge case, one failure mode.

- Use the kernel's `fake_venue_adapter` (in `_kernel/`) for any
  test that needs to exercise order submission.

- No tests should hit the real Polymarket API. Ever.

## Conventions

- Python 3.11+. uv for packaging.
- `from __future__ import annotations` at the top of every file.
- Dataclasses with `frozen=True` for value types.
- Type hints on all public functions.
- snake_case files, PascalCase classes.

## What to do when stuck

1. If a kernel file looks wrong: stop, document in this file under
   "Open kernel issues", continue with a workaround.
2. If a Polymarket API behaviour is unclear: stop, write a small
   read-only probe script in `tests/probes/`, run it, capture
   output, then implement.
3. If a design choice is ambiguous: stop and ask. Do not guess.

## Open kernel issues

(none yet)
