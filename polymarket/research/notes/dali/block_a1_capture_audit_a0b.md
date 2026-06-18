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
- Existing takeaway/status: Block A1 capture audit for the A0b data slice, covering event totals, gap logs, shards, and per-market counts. It verifies input coverage before downstream A1 analyses.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
Run dir: `data/live_clob/block_a0b/block_a0b_replacements_v2_20260527`

## Total Event Counts

```json
{
  "best_bid_ask": 58660,
  "book": 12212,
  "last_trade_price": 6063,
  "market_resolved": 3,
  "new_market": 3570,
  "price_change": 842267,
  "tick_size_change": 20
}
```

## Gap Log Counts

```json
{
  "capture_end": 1,
  "capture_start": 1,
  "connect_attempt": 2,
  "connected": 2,
  "disconnect_or_error": 1,
  "heartbeat": 717
}
```

## Shards

shard,first_received_at,last_received_at,book,price_change,best_bid_ask,last_trade_price,new_market,tick_size_change,market_resolved,inter_shard_gap_seconds
block_a0b_replacements_v2_20260527_20260527T211912Z.jsonl,2026-05-27T21:19:13.270Z,2026-05-27T22:19:13.010Z,1212,86948,3918,592,455,0.0,0.0,0.0
block_a0b_replacements_v2_20260527_20260527T221913Z.jsonl,2026-05-27T22:19:13.016Z,2026-05-27T23:19:12.981Z,470,47366,1320,235,311,4.0,0.0,0.006
block_a0b_replacements_v2_20260527_20260527T231912Z.jsonl,2026-05-27T23:19:13.688Z,2026-05-28T00:19:13.284Z,512,48418,1236,255,156,0.0,1.0,0.707
block_a0b_replacements_v2_20260527_20260528T001913Z.jsonl,2026-05-28T00:19:13.650Z,2026-05-28T01:19:13.365Z,998,70712,2386,488,166,0.0,0.0,0.366
block_a0b_replacements_v2_20260527_20260528T011913Z.jsonl,2026-05-28T01:19:13.418Z,2026-05-28T02:19:13.355Z,1054,90667,5974,524,161,0.0,0.0,0.053
block_a0b_replacements_v2_20260527_20260528T021913Z.jsonl,2026-05-28T02:19:13.361Z,2026-05-28T03:19:13.518Z,1604,70831,6480,797,408,0.0,0.0,0.006
block_a0b_replacements_v2_20260527_20260528T031913Z.jsonl,2026-05-28T03:19:13.526Z,2026-05-28T04:19:13.678Z,1076,91687,7314,537,400,12.0,1.0,0.008
block_a0b_replacements_v2_20260527_20260528T041913Z.jsonl,2026-05-28T04:19:13.707Z,2026-05-28T05:19:13.815Z,680,83371,7644,338,772,0.0,0.0,0.029
block_a0b_replacements_v2_20260527_20260528T051913Z.jsonl,2026-05-28T05:19:13.822Z,2026-05-28T06:19:13.997Z,866,70128,6084,431,141,0.0,0.0,0.007
block_a0b_replacements_v2_20260527_20260528T061914Z.jsonl,2026-05-28T06:19:14.015Z,2026-05-28T07:19:13.961Z,1494,76476,6152,746,139,0.0,0.0,0.018
block_a0b_replacements_v2_20260527_20260528T071913Z.jsonl,2026-05-28T07:19:14.076Z,2026-05-28T08:19:14.018Z,1664,62737,5164,830,247,4.0,1.0,0.115
block_a0b_replacements_v2_20260527_20260528T081914Z.jsonl,2026-05-28T08:19:14.074Z,2026-05-28T09:19:13.574Z,582,42926,4988,290,214,0.0,0.0,0.056

## Per-Market Counts

market,book,price_change,best_bid_ask,last_trade_price,total,tick_size_change,market_resolved
bitcoin-up-or-down-on-may-28-2026,2212,464216,9544,1096,477072,4.0,0.0
ethereum-up-or-down-on-may-28-2026,500,320380,16896,248,338028,4.0,0.0
btc-updown-4h-1779926400,1988,180404,12774,990,196162,4.0,2.0
btc-updown-4h-1779940800,2858,178698,12950,1424,195936,4.0,2.0
strait-of-hormuz-traffic-returns-to-normal-by-end-of-june,798,193922,818,396,195934,0.0,0.0
strait-of-hormuz-traffic-returns-to-normal-by-july-31,274,143124,296,131,143825,0.0,0.0
nba-okc-sas-2026-05-28,2556,101118,2552,1269,107495,0.0,0.0
btc-updown-4h-1779912000,622,68700,2430,309,72067,4.0,2.0
will-psg-win-the-202526-champions-league,404,33972,400,200,34976,0.0,0.0
