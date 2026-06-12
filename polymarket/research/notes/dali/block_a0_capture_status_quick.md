---
title: "Block A0 Capture Status"
created: 2026-06-05
status: generated
owner: justin
project: polymarket
para: project
hubs:
  - COWORK
tags:
  - research
  - dali
---
# Block A0 Capture Status
> Hub: [[COWORK]]


## Summary

- Scope: Block A0 Capture Status in the Dali research lineage area.
- Existing takeaway/status: Quick Block A0 capture-status snapshot covering event totals, gap logs, shard coverage, and per-market counts. It is a lightweight coverage check, not a strategy conclusion.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
Run dir: `data/live_clob/block_a0/block_a0_20260528_morning`

## Total Event Counts

```json
{
  "best_bid_ask": 23840,
  "book": 2232,
  "last_trade_price": 1062,
  "new_market": 3693,
  "price_change": 502637
}
```

## Gap Log Counts

```json
{
  "capture_start": 1,
  "connect_attempt": 4,
  "connected": 4,
  "disconnect_or_error": 3,
  "heartbeat": 613
}
```

## Shards

shard,first_received_at,last_received_at,book,price_change,new_market,best_bid_ask,last_trade_price,inter_shard_gap_seconds
block_a0_20260528_morning_20260527T100728Z.jsonl,2026-05-27T10:07:28.281Z,2026-05-27T11:07:28.112Z,128,45606,139,1712,47,0.0
block_a0_20260528_morning_20260527T110728Z.jsonl,2026-05-27T11:07:28.454Z,2026-05-27T12:07:28.280Z,150,54347,309,2126,75,0.342
block_a0_20260528_morning_20260527T120728Z.jsonl,2026-05-27T12:07:28.602Z,2026-05-27T13:07:28.520Z,266,69616,242,3276,133,0.322
block_a0_20260528_morning_20260527T130728Z.jsonl,2026-05-27T13:07:28.820Z,2026-05-27T14:07:28.642Z,256,42761,293,3610,128,0.3
block_a0_20260528_morning_20260527T140728Z.jsonl,2026-05-27T14:07:28.712Z,2026-05-27T15:07:28.965Z,272,36510,183,1556,135,0.07
block_a0_20260528_morning_20260527T150728Z.jsonl,2026-05-27T15:07:28.983Z,2026-05-27T16:07:29.012Z,304,60809,372,1284,128,0.018
block_a0_20260528_morning_20260527T160729Z.jsonl,2026-05-27T16:07:29.108Z,2026-05-27T17:07:28.993Z,180,40739,513,1704,90,0.096
block_a0_20260528_morning_20260527T170728Z.jsonl,2026-05-27T17:07:29.033Z,2026-05-27T18:07:29.732Z,244,36124,292,1699,122,0.04
block_a0_20260528_morning_20260527T180729Z.jsonl,2026-05-27T18:07:29.736Z,2026-05-27T19:07:29.820Z,226,61470,201,5293,101,0.004
block_a0_20260528_morning_20260527T190729Z.jsonl,2026-05-27T19:07:29.830Z,2026-05-27T20:07:30.113Z,156,41503,890,1140,78,0.01
block_a0_20260528_morning_20260527T200730Z.jsonl,2026-05-27T20:07:30.137Z,2026-05-27T20:23:33.130Z,50,13152,259,440,25,0.024

## Per-Market Counts

market,book,price_change,best_bid_ask,last_trade_price,total
will-china-invade-taiwan-by-december-31-2027,14,257772,6,3.0,257795
nato-x-russia-military-clash-by-december-31-2026-244,60,170642,442,26.0,171170
us-iran-nuclear-deal-before-2027,298,158176,1918,145.0,160537
will-openai-announce-earbuds-or-headphones-in-2026,8,102626,9386,0.0,112020
will-anthropic-have-the-best-ai-model-at-the-end-of-june-2026,128,101848,4446,59.0,106481
will-mojtaba-khamenei-be-head-of-state-in-iran-end-of-2026,92,65682,108,42.0,65924
will-spain-win-the-2026-fifa-world-cup-963,818,54034,812,402.0,56066
will-france-win-the-2026-fifa-world-cup-924,668,38798,660,330.0,40456
will-google-have-the-best-ai-model-at-the-end-of-june-2026,116,25234,112,52.0,25514
will-gpt-6-be-released,10,16384,1830,1.0,18225
metamask-fdv-above-700m-one-day-after-launch-696-977-652-246-632,10,12578,4118,1.0,16707
will-the-sp-500-have-the-best-performance-in-2026-545,10,1500,2,1.0,1513
