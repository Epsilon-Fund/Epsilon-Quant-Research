---
title: Block A0 Runbook
created: 2026-05-27
status: archived
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
tags:
  - dali
  - block-a0
  - runbook
  - capture
---

# Block A0 Runbook

> Hub: [[COWORK]]


## Summary

This runbook lays out the original Block A0 24-hour gap-tolerant OFI capture plan for the morning of 2026-05-28. It documents prepared configs/scripts, local laptop and VPS/tmux modes, output paths, and restart behavior. The note is historical operating context for the A0 capture lineage rather than an active run instruction.

Generated: 2026-05-27

Purpose: start a 24h gap-tolerant OFI capture on the morning of 2026-05-28.
The capture is robust to WebSocket disconnects and process restarts, but a
laptop that sleeps or powers off cannot receive live WebSocket data during that
time. Those gaps are acceptable for this phase and will be visible in the audit.

## Prepared Files

- Base config: `configs/block_a0_capture.yaml`
- Generated fee-aware config: `configs/block_a0_capture.generated.yaml`
- Shortlist builder: `scripts/dali_block_a0_prepare.py`
- Durable runner: `scripts/dali_block_a0_capture.py`
- Audit utility: `scripts/dali_block_a0_capture_audit.py`

Current generated shortlist contains 12 markets:

- 4 true geopolitics/world-event markets, fee-free per CLOB metadata where
  available.
- 4 AI/tech markets.
- 2 sports markets.
- 1 finance/equity-index market.
- 1 contrasting crypto/finance-style market.

## Tomorrow Morning Start

From the repo:

```bash
cd /Users/justiniturregui/Desktop/github/epsilon-quant-research/polymarket/research

PYTHONPATH=. uv run python scripts/dali_block_a0_prepare.py

PYTHONPATH=. uv run python scripts/dali_block_a0_capture.py \
  --config configs/block_a0_capture.generated.yaml \
  --run-id block_a0_20260528_morning \
  --duration-hours 24
```

Output will land under:

```text
data/live_clob/block_a0/block_a0_20260528_morning/
```

The runner rotates JSONL shards hourly and writes:

- `capture_config.yaml`
- `capture_gaps.jsonl`
- hourly `*.jsonl` shards
- one `*.manifest.json` per shard

## Laptop Mode

If running on your laptop, this is gap-tolerant, not continuous. If the laptop
sleeps, shuts down, loses Wi-Fi, or the process dies, no live WS events are
captured during that interval. Restart with the same `--run-id`; a new shard
will be appended to the same run directory, and the audit will show the gap.

Optional if you want to keep a Mac awake while the process runs:

```bash
caffeinate -dimsu PYTHONPATH=. uv run python scripts/dali_block_a0_capture.py \
  --config configs/block_a0_capture.generated.yaml \
  --run-id block_a0_20260528_morning \
  --duration-hours 24
```

If you want to sleep and close the laptop, prefer VPS mode.

## VPS Or tmux Mode

On the VPS, run inside `tmux`:

```bash
cd /path/to/epsilon-quant-research/polymarket/research
tmux new -s dali_block_a0
PYTHONPATH=. uv run python scripts/dali_block_a0_prepare.py
PYTHONPATH=. uv run python scripts/dali_block_a0_capture.py \
  --config configs/block_a0_capture.generated.yaml \
  --run-id block_a0_20260528_morning \
  --duration-hours 24
```

Detach with `Ctrl-b d`. Reattach:

```bash
tmux attach -t dali_block_a0
```

## Status Check

During or after capture:

```bash
PYTHONPATH=. uv run python scripts/dali_block_a0_capture_audit.py \
  --run-dir data/live_clob/block_a0/block_a0_20260528_morning \
  --out notes/dali/block_a0_capture_status.md
```

Review:

- total `book`, `price_change`, `best_bid_ask`, `last_trade_price` counts,
- per-market counts,
- `capture_gaps.jsonl`,
- `inter_shard_gap_seconds` in the shard table.

## After 24h

If the capture has at least:

- enough `last_trade_price` events for sign audit progress,
- useful event counts across at least several markets,
- no catastrophic gaps,

then replay features and start Block A analysis. If counts are thin, extend to
48h with the same command and same run id:

```bash
PYTHONPATH=. uv run python scripts/dali_block_a0_capture.py \
  --config configs/block_a0_capture.generated.yaml \
  --run-id block_a0_20260528_morning \
  --duration-hours 24
```
