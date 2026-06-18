---
title: "MM Maker Infrastructure Audit Findings"
tags: [market-making, execution, maker-engine, negrisk, telemetry]
created: 2026-06-04
status: measurement-grade-not-production-grade
---

# MM Maker Infrastructure Audit Findings

> Hub: [[strat_market_making]] · [[COWORK]]

## Plain-English Summary

This note audits the shared maker infrastructure in `polymarket/execution/maker/` without wiring any market list. The important update is that the Phase-1 maker prerequisites are already built and tested, despite stale TODO wording that previously said "Phase-1 exec build" was next. The code is measurement-grade: good enough to run a one-condition, telemetry-heavy smoke loop, but not a production market-making system.

## Test Result

Commands run from the repo root:

```bash
PYTHONPATH=. uv run --no-project --with pytest --with py-clob-client --with websockets python -m pytest polymarket/execution/tests/maker polymarket/execution/tests/test_market_metadata.py polymarket/execution/tests/test_clob_http_client.py -q
PYTHONPATH=. uv run --no-project --with pytest --with py-clob-client --with websockets python -m pytest polymarket/execution/tests -q
```

Results:

- Maker plus NegRisk dependency slice: **67 passed in 0.64s**.
- Full execution suite: **256 passed in 1.55s**.

## What Exists

The current maker scaffold includes:

- `maker/event_calendar.py`: manual scheduled-event calendar and event-proximity checks.
- `maker/negrisk_inventory.py`: Data API activity polling for `SPLIT`, `MERGE`, `REDEEM`, and `CONVERSION`, with replayable JSONL inventory state.
- `maker/resolution_handler.py`: Gamma closure polling, redeemable-position polling, and best-effort `NegRiskAdapter.redeemPositions` via lazy web3.
- `maker/maker_engine.py`: one-condition passive quote loop that joins displayed best bid and best ask on the YES token, refreshes when price moves by more than one tick, cancels on resolution/shutdown, and logs quote/fill/missed-fill telemetry.
- `maker/cli.py`: `python -m polymarket.execution --mode maker`, plus `--check-auth` read-only real-venue auth verification.
- Shared NegRisk signing path: `mirror/market_metadata.py`, `mirror/clob_http_client.py` with `set_neg_risk`, and `mirror/clob_signer.py` passing `PartialCreateOrderOptions(neg_risk=..., tick_size=...)`.

## Telemetry Emitted

The journal vocabulary already covers:

- Quote lifecycle: `OrderSubmitted`, `OrderAcknowledged`, `OrderRejected`, `MakerQuotePlaced`, `MakerQuoteCanceled`, `MakerQuoteSkipped`.
- Fill diagnostics: `MakerFillTelemetry` with `top_maker_rank_at_fill`, `post_fill_price_drift_60s`, `news_proximate`, and `fill_share_this_market`.
- Missed fills: `MakerMissedFill` when another wallet trades at our resting quote price while our quote remains unfilled.
- NegRisk inventory and resolution: `BasketInventoryUpdated`, `MarketResolved`, `PositionRedeemable`, `PositionRedeemed`, `RedemptionFailed`.
- Session lifecycle: `MakerSessionStarted`, `MakerSessionStopped`.

## Measurement-Grade vs Production-Grade

The engine is measurement-grade because it can answer the first live questions: did we place/cancel quotes, did we get filled, did prints happen at our price without us, what was the rough post-fill drift, and was the fill near a scheduled event?

It is not production-grade:

- Queue position is a coarse fill-order proxy inside a +/-5 second same-price window, not true CLOB queue position.
- `post_fill_price_drift_60s` is a raw price difference, not yet side-adjusted adverse-selection bps.
- Fill share is our session fill count divided by Data API trade count, not volume-weighted maker share or eligible same-side quote share.
- Book depth, depth ahead of our quote, quote age, cancel latency, and missed-fill depth are not journaled.
- It quotes one condition and the YES token only; it does not orchestrate multiple markets, rank candidates, optimize tick placement, or manage cross-market exposure.
- News telemetry is a manual scheduled-event flag only; no breaking-news feed or LLM scanner is included.
- No lane-specific market list should be wired here. Each lane must choose its own markets after its own measurement gate.

## Stale Docs / TODO Lines

`polymarket/execution/PLAN.md` is current and records the landed Phase-1 work. `brain/TODO.md` was stale before this audit in three places and has now been corrected:

- The MM header still said "NEXT: Phase-1 exec build"; that is now complete.
- The open task still listed the Phase-1 build as unchecked; it is now checked and linked to this audit.
- The SPX close-style live collector and `other:misc_other` characterization tasks needed updating after [[mm_structural_maker_directional_decomposition_findings]] and [[mm_nonpolitics_target_screen_findings]]; both are now synced.

## Decision

Treat the maker scaffold as ready for lane-owned live measurement smokes once a sleeve survives its research gate. Do not treat it as a production bot. The next shared infra work is telemetry hardening only if a lane has a live measurement candidate: side-adjusted drift bps, volume-weighted fill share, top-of-book depth at quote/fill, quote age, cancel latency, and a clearer queue proxy.
