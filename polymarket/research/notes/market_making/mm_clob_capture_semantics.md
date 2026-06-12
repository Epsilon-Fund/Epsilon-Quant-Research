---
title: "PM CLOB Capture Semantics"
created: 2026-06-07
status: active
owner: justin
project: polymarket
para: project
hubs:
  - strat_market_making
  - COWORK
tags:
  - research
  - market-making
---
# PM CLOB Capture Semantics
> Hub: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · [[COWORK]]

## Summary

- Scope: PM CLOB Capture Semantics in the MM/market-making area.
- Existing takeaway/status: This note records what the public Polymarket CLOB market websocket can and cannot prove for Stage-1 market-making measurement. It exists so future agents do not over-interpret the live `data/live_clob/` JSONL captures as wallet-level order truth.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
## Purpose

This note records what the public Polymarket CLOB market websocket can and cannot prove for Stage-1 market-making measurement. It exists so future agents do not over-interpret the live `data/live_clob/` JSONL captures as wallet-level order truth.

## Public Market Websocket Shape

The public market channel is `wss://ws-subscriptions-clob.polymarket.com/ws/market`. It is read-only and unauthenticated. Official docs describe it as public level-2 market data: orderbook snapshots, price-level updates, trade executions, and market lifecycle events.

Useful event types:

- `book`: full L2 snapshot for a token, with aggregate bid/ask price levels and sizes. Docs say it is emitted on subscription and when a trade affects the book.
- `price_change`: price-level update. Docs say it is emitted when a new order is placed or an order is canceled.
- `last_trade_price`: trade execution print, with token, price, side, size, fee bps, timestamp, and sometimes transaction hash.
- `best_bid_ask`: top-of-book update. This is logically derivable from L2, but is useful as a checksum for L2 reconstruction.

Repo entry points:

- Recorder: `polymarket/research/scripts/dali_block_a0_capture.py`
- Raw websocket envelope/parser: `polymarket/research/scripts/dali_live_clob_capture.py`
- Stage-1 analyzer: `polymarket/research/scripts/mm_stage1_analyze_capture.py`
- Durable runbook: [[block_a0_runbook]]

## What We Cannot See

The public L2 feed is anonymous. `book` and `price_change` expose aggregate price levels, not individual order IDs, maker wallets, or queue position. Therefore:

- We cannot see which wallet owns a bid or ask in live public L2.
- We cannot exactly reconstruct queue position or our future fill share from public data alone.
- We cannot prove exact cancel-vs-fill causality at order ID level.
- Historical on-chain/order-filled data can identify wallets after fills, but that is different from knowing who owns the current resting book.

The authenticated user channel is for our own orders/trades only; it does not reveal other makers' private order owners.

## What We Can Infer

We can reconstruct a useful price-level book and classify liquidity events probabilistically:

- Start from `book`.
- Apply `price_change` by token, side, price, and size.
- Maintain top-N levels, e.g. first four levels.
- Cross-check reconstructed L1 against `best_bid_ask`.
- Align `last_trade_price` prints with nearby L2 depletion/moves.

Trade-vs-cancel interpretation:

- If `last_trade_price` occurs and same-neighborhood L2 size disappears at/through the touch, classify as likely liquidity-taking / book consumption.
- If size decreases with no nearby `last_trade_price`, classify as likely cancel/pull.
- If size increases, classify as new passive liquidity.
- If several changes share the same exchange timestamp or arrive in a burst, mark the interval ambiguous rather than forcing a false precise label.

This is good enough for Stage-1 book-shape, fill-opportunity, and toxicity measurement. It is not good enough for production-grade queue simulation without either our own order telemetry or an additional private/order-level source.

## Timestamp Semantics

The recorder preserves both exchange-provided and local timing:

- Raw PM payload timestamp: `message.timestamp`, usually millisecond epoch when the event type provides it.
- Local wall-clock receive time: `received_at`, UTC ISO string.
- Local monotonic receive clock: `received_monotonic_ns`, from `time.monotonic_ns()`.

The envelope is built in `dali_live_clob_capture.py`; the durable runner writes those envelopes unchanged into hourly JSONL shards and records heartbeats/gaps in `capture_gaps.jsonl`.

Practical rules:

- Prefer PM `message.timestamp` for event-time ordering when present.
- Use local monotonic receive time to preserve local arrival order and measure gaps.
- Treat absolute exchange-vs-local latency cautiously because local clock skew can make apparent latency negative.
- Initial `book` snapshots can have stale timestamps; separate initial snapshots from live updates in latency audits.

## Required Reconstruction Audit

Before using the data for any stronger backtest, run an explicit reconstruction audit:

1. Rebuild top-N L2 from `book + price_change`.
2. Compare reconstructed L1 to native `best_bid_ask` by token and timestamp neighborhood.
3. Measure the share of `last_trade_price` events with clean book depletion at the trade price/touch.
4. Measure the share of price-level decreases with no nearby trade print.
5. Report clean vs ambiguous intervals per market/category.

Only the clean subset should drive quoteability/toxicity claims. Ambiguous intervals are still useful for health/coverage, but should not be promoted into exact maker/taker attribution.

## Current Stage-1 Framing

The current Stage-1 capture is a live measurement loop, not a deployability verdict. It answers:

- Is the market quoteable: spread, depth, top-N book shape?
- Is there flow: `last_trade_price` arrivals per hour?
- Is the flow toxic: post-trade mid drift in the trade direction?
- Is the first-mover / slow-market idea worth a Stage-2 active quote test?

It does not answer:

- Which wallet owns the book?
- What exact queue position we would have?
- What fill share we would capture if we quoted?
- Whether the strategy is production-grade.
