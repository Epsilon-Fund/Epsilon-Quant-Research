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
- Existing takeaway/status: Latest Block A0c capture-status snapshot covering event totals, gap logs, shards, and per-market counts. Use it as a data-availability checkpoint.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
Run dir: `data/live_clob/block_a0c/block_a0c_targeted_20260529_morning`

## Total Event Counts

```json
{
  "best_bid_ask": 176,
  "book": 78,
  "last_trade_price": 22,
  "new_market": 14,
  "price_change": 2622
}
```

## Gap Log Counts

```json
{
  "capture_start": 1,
  "connect_attempt": 1,
  "connected": 1,
  "heartbeat": 1
}
```

## Shards

shard,first_received_at,last_received_at,book,price_change,best_bid_ask,last_trade_price,new_market,inter_shard_gap_seconds
block_a0c_targeted_20260529_morning_20260529T090130Z.jsonl,2026-05-29T09:01:30.515Z,2026-05-29T09:03:08.113Z,78,2622,176,22,14,0.0

## Per-Market Counts

market,book,price_change,total,best_bid_ask,last_trade_price
will-jd-vance-win-the-2028-us-presidential-election,2,1340,1342,0.0,0.0
btc-updown-4h-1780041600,2,892,938,44.0,0.0
sol-updown-4h-1780041600,2,548,568,18.0,0.0
strait-of-hormuz-traffic-returns-to-normal-by-july-31,2,548,552,2.0,0.0
eth-updown-4h-1780041600,2,504,552,46.0,0.0
bitcoin-up-or-down-on-may-29-2026,4,402,409,2.0,1.0
nhl-mon-car-2026-05-29-spread-home-1pt5,2,292,294,0.0,0.0
us-x-iran-permanent-peace-deal-by-june-30-2026-837-641-896-877-363-892-537-597,30,214,286,28.0,14.0
ethereum-up-or-down-on-may-29-2026,4,232,253,16.0,1.0
solana-up-or-down-on-may-29-2026,2,98,108,8.0,0.0
ucl-psg-ars-2026-05-30-psg,14,64,96,12.0,6.0
will-spain-win-the-2026-fifa-world-cup-963,2,54,56,0.0,0.0
will-the-oklahoma-city-thunder-win-the-2026-nba-finals,2,28,30,0.0,0.0
btc-updown-4h-1780056000,2,12,14,0.0,0.0
nba-sas-okc-2026-05-30-total-211pt5,2,8,10,0.0,0.0
sol-updown-4h-1780056000,2,4,6,0.0,0.0
eth-updown-4h-1780056000,2,4,6,0.0,0.0
