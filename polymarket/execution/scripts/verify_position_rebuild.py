from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

# Read the most recent smoke journal file
journal_dir = Path("journal_logs")
files = sorted(journal_dir.glob("smoke-*.jsonl"))
if not files:
    print("No smoke journal files found. Run smoke_watcher.py first.")
    sys.exit(1)

path = files[-1]
print(f"Reading {path}\n")

# Reconstruct leader positions using the exact algorithm signal/
# will use. If this script crashes or produces nonsense, the
# algorithm is wrong.
positions: dict[tuple[str, str], dict] = {}

fill_count = 0
for line in path.read_text().splitlines():
    if not line.strip():
        continue
    event = json.loads(line)
    if event.get("event_type") != "LEADER_FILL_OBSERVED":
        continue
    fill_count += 1

    key = (event["condition_id"].lower(), event["asset_id"])
    side = event["side"]
    shares = float(event["size"])
    price = float(event["price"])

    if side == "BUY":
        existing = positions.get(key)
        if existing is None:
            positions[key] = {
                "condition_id": key[0],
                "asset_id": key[1],
                "side": side,
                "shares": shares,
                "avg_entry_price": price,
                "total_entry_usd": shares * price,
            }
        else:
            new_shares = existing["shares"] + shares
            new_total = existing["total_entry_usd"] + (shares * price)
            positions[key] = {
                **existing,
                "shares": new_shares,
                "avg_entry_price": new_total / new_shares,
                "total_entry_usd": new_total,
            }
    elif side == "SELL":
        existing = positions.get(key)
        if existing is None:
            continue  # leader exited a position we never saw entered
        new_shares = max(0.0, existing["shares"] - shares)
        if new_shares == 0.0:
            del positions[key]
            continue
        new_total = existing["total_entry_usd"] * (new_shares / existing["shares"])
        positions[key] = {**existing, "shares": new_shares,
                          "total_entry_usd": new_total}

print(f"Processed {fill_count} fills")
print(f"Resulting open positions: {len(positions)}\n")
for key, pos in positions.items():
    print(f"  {key[0][:18]}.../{key[1][-12:]}: "
          f"side={pos['side']} shares={pos['shares']:.2f} "
          f"avg=${pos['avg_entry_price']:.4f} "
          f"total=${pos['total_entry_usd']:.2f}")
