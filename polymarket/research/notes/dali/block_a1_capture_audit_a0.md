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
- Existing takeaway/status: Block A1 capture audit for the A0 data slice, covering event totals, gap logs, shards, and per-market counts. It verifies the input capture before interpreting A1 OFI/TFI results.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
Run dir: `data/live_clob/block_a0/block_a0_20260528_morning`

## Total Event Counts

```json
{
  "best_bid_ask": 50630,
  "book": 4326,
  "last_trade_price": 2095,
  "new_market": 8427,
  "price_change": 1253183
}
```

## Gap Log Counts

```json
{
  "capture_end": 1,
  "capture_start": 1,
  "connect_attempt": 5,
  "connected": 5,
  "disconnect_or_error": 4,
  "heartbeat": 1435
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
block_a0_20260528_morning_20260527T200730Z.jsonl,2026-05-27T20:07:30.137Z,2026-05-27T21:07:30.221Z,260,50962,969,1744,128,0.024
block_a0_20260528_morning_20260527T210730Z.jsonl,2026-05-27T21:07:30.289Z,2026-05-27T22:07:30.332Z,190,50348,485,1378,95,0.068
block_a0_20260528_morning_20260527T220730Z.jsonl,2026-05-27T22:07:30.520Z,2026-05-27T23:07:30.525Z,120,55532,369,1166,60,0.188
block_a0_20260528_morning_20260527T230730Z.jsonl,2026-05-27T23:07:30.530Z,2026-05-28T00:07:30.641Z,88,68503,189,1996,44,0.005
block_a0_20260528_morning_20260528T000730Z.jsonl,2026-05-28T00:07:30.685Z,2026-05-28T01:07:30.735Z,124,51227,160,1352,50,0.044
block_a0_20260528_morning_20260528T010730Z.jsonl,2026-05-28T01:07:30.757Z,2026-05-28T02:07:30.756Z,86,50593,166,1394,43,0.022
block_a0_20260528_morning_20260528T020730Z.jsonl,2026-05-28T02:07:30.803Z,2026-05-28T03:07:31.163Z,140,45210,417,1716,70,0.047
block_a0_20260528_morning_20260528T030731Z.jsonl,2026-05-28T03:07:31.176Z,2026-05-28T04:07:31.362Z,144,38189,249,3046,72,0.013
block_a0_20260528_morning_20260528T040731Z.jsonl,2026-05-28T04:07:31.370Z,2026-05-28T05:07:31.507Z,76,66785,921,2330,38,0.008
block_a0_20260528_morning_20260528T050731Z.jsonl,2026-05-28T05:07:31.513Z,2026-05-28T06:07:31.538Z,116,61161,142,1866,58,0.006
block_a0_20260528_morning_20260528T060731Z.jsonl,2026-05-28T06:07:31.555Z,2026-05-28T07:07:31.504Z,156,57812,140,1620,78,0.017
block_a0_20260528_morning_20260528T070731Z.jsonl,2026-05-28T07:07:31.589Z,2026-05-28T08:07:31.604Z,244,58016,192,2192,122,0.085
block_a0_20260528_morning_20260528T080731Z.jsonl,2026-05-28T08:07:31.673Z,2026-05-28T09:07:31.687Z,182,55717,265,2184,91,0.069
block_a0_20260528_morning_20260528T090731Z.jsonl,2026-05-28T09:07:31.696Z,2026-05-28T10:07:27.948Z,218,53643,329,3246,109,0.009

## Per-Market Counts

market,book,price_change,best_bid_ask,last_trade_price,total
will-anthropic-have-the-best-ai-model-at-the-end-of-june-2026,320,696502,9650,153,706625
will-china-invade-taiwan-by-december-31-2027,38,614446,28,14,614526
us-iran-nuclear-deal-before-2027,422,325460,4876,206,330964
will-openai-announce-earbuds-or-headphones-in-2026,12,262220,25434,1,287667
nato-x-russia-military-clash-by-december-31-2026-244,74,211640,570,32,212316
will-mojtaba-khamenei-be-head-of-state-in-iran-end-of-2026,248,116844,282,119,117493
will-france-win-the-2026-fifa-world-cup-924,1420,78140,1410,705,81675
will-spain-win-the-2026-fifa-world-cup-963,1550,77710,1542,767,81569
will-google-have-the-best-ai-model-at-the-end-of-june-2026,200,70688,222,92,71202
will-gpt-6-be-released,14,33516,1832,2,35364
metamask-fdv-above-700m-one-day-after-launch-696-977-652-246-632,12,15028,4774,1,19815
will-the-sp-500-have-the-best-performance-in-2026-545,16,4172,10,3,4201
