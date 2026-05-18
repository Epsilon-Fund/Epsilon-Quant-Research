from __future__ import annotations
import os
import queue
import threading
import time
from polymarket.execution.config import ExecutionConfig
from polymarket.execution.journal import JsonlWriter
from polymarket.execution.watcher import LeaderWatcher
from polymarket.execution.signal import Classifier, Deduplicator

# Use the same leader you've been smoke-testing with. Update
# this default if you want a different leader.
os.environ.setdefault("POLYMARKET_LEADER_ADDRESS",
    "0x8068e018bbc4cd013f611490460aaea05d601900")
os.environ.setdefault("POLYMARKET_PRIVATE_KEY", "0x" + "00" * 32)
os.environ.setdefault("POLYMARKET_API_KEY", "dummy")
os.environ.setdefault("POLYMARKET_API_SECRET", "dummy")
os.environ.setdefault("POLYMARKET_PASSPHRASE", "dummy")
os.environ.setdefault("POLYMARKET_FUNDER", "0x" + "0" * 40)

config = ExecutionConfig.from_env()
journal = JsonlWriter(config.journal_dir, "dry_run")

fill_queue: queue.Queue = queue.Queue(maxsize=1000)
signal_queue: queue.Queue = queue.Queue(maxsize=1000)

dedup = Deduplicator(journal)
classifier = Classifier(config, journal, dedup, signal_queue)
watcher = LeaderWatcher(config, journal, fill_queue)

print(f"Bot positions on startup: {len(classifier._bot_positions)}")
print(f"Leader positions on startup: {len(classifier._leader_positions)}")
print(f"Dedup seed size: {len(dedup)}")

stop_event = threading.Event()

def consume_fills() -> None:
    """Pull from fill_queue, run classifier, signal_queue gets the result."""
    while not stop_event.is_set():
        try:
            fill = fill_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            classifier.process_fill(fill)
        except Exception as e:
            print(f"  CLASSIFIER ERROR: {e}")
        fill_queue.task_done()

consumer = threading.Thread(target=consume_fills, daemon=True)
consumer.start()
watcher.start()
print(f"Watching {config.leader_address} for 1 minute...\n")

signals_seen = 0
try:
    for i in range(60):
        time.sleep(1)
        # Drain signal_queue if anything's there
        while not signal_queue.empty():
            signal = signal_queue.get_nowait()
            signals_seen += 1
            print(
                f"  [{i+1:3d}s] SIGNAL #{signals_seen}: "
                f"kind={signal.kind.value} "
                f"side={signal.side} "
                f"shares={signal.target_size_shares:.4f} @ "
                f"${signal.leader_fill_price:.4f} "
                f"cond={signal.condition_id[:18]}..."
            )
        if (i + 1) % 15 == 0:
            print(
                f"  [{i+1:3d}s] tick: "
                f"fill_queue={fill_queue.qsize()} "
                f"signal_queue={signal_queue.qsize()} "
                f"signals_total={signals_seen} "
                f"bot_positions={len(classifier._bot_positions)} "
                f"leader_positions={len(classifier._leader_positions)}"
            )
finally:
    stop_event.set()
    consumer.join(timeout=2)
    watcher.stop()
    journal.close()

print(f"\nDone.")
print(f"  Total signals emitted: {signals_seen}")
print(f"  Final bot positions: {len(classifier._bot_positions)}")
print(f"  Final leader positions: {len(classifier._leader_positions)}")
print(f"  Journal: {config.journal_dir}")
