from __future__ import annotations
import os
import queue
import time
from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import JsonlWriter
from polymarket.execution.watcher import LeaderWatcher

# Dimitris's proxy wallet (sourced from midas/.env PM_FUNDER).
os.environ.setdefault("POLYMARKET_LEADER_ADDRESS",
    "0x52409dc9f10e9662db70b435e57a6d3e90184607")
# Required env vars set to dummy values — this is read-only.
os.environ.setdefault("POLYMARKET_PRIVATE_KEY", "0x" + "00"*32)
os.environ.setdefault("POLYMARKET_API_KEY", "dummy")
os.environ.setdefault("POLYMARKET_API_SECRET", "dummy")
os.environ.setdefault("POLYMARKET_PASSPHRASE", "dummy")
os.environ.setdefault("POLYMARKET_FUNDER", "0x" + "0"*40)

config = ExecutionConfig.from_env()
journal = JsonlWriter(config.journal_dir, "smoke")
fill_queue: queue.Queue = queue.Queue(maxsize=1000)
watcher = LeaderWatcher(config, journal, fill_queue)

watcher.start()
print(f"Watching {config.leader_address} for 5 minutes...", flush=True)
try:
    for i in range(300):
        time.sleep(1)
        if (i + 1) % 30 == 0:
            print(f"  {i+1}s: queue has {fill_queue.qsize()} fills", flush=True)
finally:
    watcher.stop()
    journal.close()

print(f"\nDone. Final queue size: {fill_queue.qsize()}")
print(f"Journal dir: {config.journal_dir}")
