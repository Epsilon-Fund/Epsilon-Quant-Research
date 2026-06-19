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
- Existing takeaway/status: Final Block A0c capture-status snapshot covering event totals, gap logs, shards, and per-market counts. It records capture completeness for the A0c branch.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
Run dir: `data/live_clob/block_a0c/block_a0c_targeted_20260529_morning`

## Total Event Counts

```json
{
  "best_bid_ask": 197022,
  "book": 26289,
  "last_trade_price": 12788,
  "market_resolved": 10,
  "new_market": 7704,
  "price_change": 2150145,
  "tick_size_change": 32
}
```

## Gap Log Counts

```json
{
  "capture_end": 1,
  "capture_start": 1,
  "connect_attempt": 17,
  "connected": 17,
  "disconnect_or_error": 16,
  "heartbeat": 1427
}
```

## Shards

shard,first_received_at,last_received_at,book,price_change,best_bid_ask,last_trade_price,new_market,tick_size_change,market_resolved,inter_shard_gap_seconds
block_a0c_targeted_20260529_morning_20260529T090130Z.jsonl,2026-05-29T09:01:30.515Z,2026-05-29T10:01:30.343Z,1198,94552,6853,575,173,0.0,0.0,0.0
block_a0c_targeted_20260529_morning_20260529T100130Z.jsonl,2026-05-29T10:01:30.353Z,2026-05-29T11:01:30.305Z,1008,116096,9079,499,187,0.0,0.0,0.01
block_a0c_targeted_20260529_morning_20260529T110130Z.jsonl,2026-05-29T11:01:30.420Z,2026-05-29T11:59:24.929Z,1743,138045,13516,865,191,4.0,0.0,0.115
block_a0c_targeted_20260529_morning_20260529T120130Z.jsonl,2026-05-29T12:01:31.277Z,2026-05-29T13:00:26.538Z,648,30588,3654,233,267,0.0,3.0,126.348
block_a0c_targeted_20260529_morning_20260529T130131Z.jsonl,2026-05-29T13:02:32.892Z,2026-05-29T14:01:31.586Z,1132,123154,18840,452,183,0.0,0.0,126.354
block_a0c_targeted_20260529_morning_20260529T140131Z.jsonl,2026-05-29T14:01:31.602Z,2026-05-29T15:01:31.601Z,2720,316180,61832,1351,306,8.0,0.0,0.016
block_a0c_targeted_20260529_morning_20260529T150131Z.jsonl,2026-05-29T15:01:31.616Z,2026-05-29T16:01:31.565Z,3092,223132,54414,1532,322,16.0,0.0,0.015
block_a0c_targeted_20260529_morning_20260529T160131Z.jsonl,2026-05-29T16:01:31.648Z,2026-05-29T17:01:31.733Z,1308,46515,1686,649,507,0.0,6.0,0.083
block_a0c_targeted_20260529_morning_20260529T170131Z.jsonl,2026-05-29T17:01:31.771Z,2026-05-29T18:01:31.729Z,1276,42640,1470,627,291,0.0,0.0,0.038
block_a0c_targeted_20260529_morning_20260529T180131Z.jsonl,2026-05-29T18:01:31.811Z,2026-05-29T19:01:31.809Z,766,39670,880,381,231,0.0,0.0,0.082
block_a0c_targeted_20260529_morning_20260529T190131Z.jsonl,2026-05-29T19:01:31.897Z,2026-05-29T20:01:32.026Z,686,45001,786,335,194,0.0,0.0,0.088
block_a0c_targeted_20260529_morning_20260529T200132Z.jsonl,2026-05-29T20:01:32.106Z,2026-05-29T21:01:32.181Z,648,40146,766,321,322,0.0,0.0,0.08
block_a0c_targeted_20260529_morning_20260529T210132Z.jsonl,2026-05-29T21:01:32.297Z,2026-05-29T22:01:32.243Z,918,71762,978,456,303,0.0,0.0,0.116
block_a0c_targeted_20260529_morning_20260529T220132Z.jsonl,2026-05-29T22:01:32.301Z,2026-05-29T23:01:32.240Z,784,50019,874,389,1028,0.0,0.0,0.058
block_a0c_targeted_20260529_morning_20260529T230132Z.jsonl,2026-05-29T23:01:32.250Z,2026-05-30T00:01:32.225Z,808,47515,1008,394,434,0.0,0.0,0.01
block_a0c_targeted_20260529_morning_20260530T000132Z.jsonl,2026-05-30T00:01:32.349Z,2026-05-30T01:01:32.321Z,548,149871,6290,272,406,0.0,0.0,0.124
block_a0c_targeted_20260529_morning_20260530T010132Z.jsonl,2026-05-30T01:01:32.415Z,2026-05-30T02:01:32.387Z,696,137523,6798,346,197,4.0,0.0,0.094
block_a0c_targeted_20260529_morning_20260530T020132Z.jsonl,2026-05-30T02:01:32.397Z,2026-05-30T03:01:32.444Z,684,80517,1166,330,213,0.0,0.0,0.01
block_a0c_targeted_20260529_morning_20260530T030132Z.jsonl,2026-05-30T03:01:32.590Z,2026-05-30T04:01:32.389Z,1488,64388,1518,740,167,0.0,1.0,0.146
block_a0c_targeted_20260529_morning_20260530T040132Z.jsonl,2026-05-30T04:01:32.398Z,2026-05-30T05:01:32.936Z,1274,57547,1340,631,901,0.0,0.0,0.009
block_a0c_targeted_20260529_morning_20260530T050132Z.jsonl,2026-05-30T05:01:33.141Z,2026-05-30T06:01:33.141Z,672,60753,724,330,295,0.0,0.0,0.205
block_a0c_targeted_20260529_morning_20260530T060133Z.jsonl,2026-05-30T06:01:33.188Z,2026-05-30T07:01:33.253Z,698,57534,770,344,162,0.0,0.0,0.047
block_a0c_targeted_20260529_morning_20260530T070133Z.jsonl,2026-05-30T07:01:33.261Z,2026-05-30T08:01:33.255Z,658,63744,890,328,227,0.0,0.0,0.008
block_a0c_targeted_20260529_morning_20260530T080133Z.jsonl,2026-05-30T08:01:33.304Z,2026-05-30T09:01:30.176Z,836,53253,890,408,197,0.0,0.0,0.049

## Per-Market Counts

market,book,price_change,best_bid_ask,last_trade_price,tick_size_change,market_resolved,total
will-jd-vance-win-the-2028-us-presidential-election,310,1354258,276,138,0.0,0.0,1354982
nhl-mon-car-2026-05-29-spread-home-1pt5,2020,475140,13940,984,4.0,2.0,492090
strait-of-hormuz-traffic-returns-to-normal-by-july-31,3064,434214,5984,1463,0.0,0.0,444725
bitcoin-up-or-down-on-may-29-2026,1950,296470,12238,952,4.0,2.0,311616
eth-updown-4h-1780056000,572,241750,44024,271,4.0,2.0,286623
ethereum-up-or-down-on-may-29-2026,600,234110,16944,284,4.0,2.0,251944
us-x-iran-permanent-peace-deal-by-june-30-2026-837-641-896-877-363-892-537-597,2594,240646,3876,1266,0.0,0.0,248382
ucl-psg-ars-2026-05-30-psg,9032,208418,9000,4471,0.0,0.0,230921
btc-updown-4h-1780056000,1210,204458,16312,586,4.0,2.0,222572
sol-updown-4h-1780056000,268,114496,27052,119,4.0,2.0,141941
solana-up-or-down-on-may-29-2026,184,102028,21008,75,4.0,2.0,123301
btc-updown-4h-1780041600,1347,95108,7204,669,0.0,2.0,104330
eth-updown-4h-1780041600,272,74402,8502,134,0.0,2.0,83312
sol-updown-4h-1780041600,158,60770,8054,77,4.0,2.0,69065
nba-sas-okc-2026-05-30-total-211pt5,248,60900,214,106,0.0,0.0,61468
will-spain-win-the-2026-fifa-world-cup-963,1380,54032,1346,671,0.0,0.0,57429
will-the-oklahoma-city-thunder-win-the-2026-nba-finals,1080,49090,1048,522,0.0,0.0,51740
