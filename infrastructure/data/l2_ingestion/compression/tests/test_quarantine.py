"""
Quarantine-and-continue test for the L2 compression pipeline.

Regression guard for the 2026-08-21 outage: a single unreadable shard (a gzip
truncated by an unclean reboot) used to raise out of process_files() and abort the
whole run, silently blocking every pending shard behind it for three days. The fix
catches the parse failure per shard, quarantines the bad path, and keeps going.

This test writes one VALID shard and one deliberately TRUNCATED shard into a temp
directory, runs process_files(), and asserts:
  * the valid shard produced Parquet and landed in _processed.txt,
  * the truncated shard landed in the quarantine list and _quarantine.txt,
  * the truncated shard did NOT land in _processed.txt.

Run from the repo root (needs pandas + pyarrow, the pipeline's own deps):
    python -m pytest infrastructure/data/l2_ingestion/compression/tests -q
"""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

# Load the pipeline module directly from its file path — l2_ingestion is not an
# importable package (no __init__.py), and this is exactly the module the VPS runs.
_PIPELINE_PATH = Path(__file__).resolve().parents[1] / "pipeline.py"
_spec = importlib.util.spec_from_file_location("l2_pipeline_under_test", _PIPELINE_PATH)
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


def _write_valid_shard(path: Path, n_rows: int = 5) -> None:
    """One book envelope per line — exactly the shape rows_from_envelope() expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for i in range(n_rows):
            rec = {
                "received_at": "2026-08-21T00:00:00.000Z",
                "received_monotonic_ns": 1000 + i,
                "event_type": "book",
                "assets": [{"universe": "testuni"}],
                "message": {
                    "timestamp": str(1_700_000_000_000 + i),
                    "market": "0xmarket",
                    "asset_id": "0xasset",
                    "bids": [["0.50", "10"]],
                    "asks": [["0.60", "12"]],
                },
            }
            fh.write(json.dumps(rec) + "\n")


def _write_truncated_shard(path: Path) -> None:
    """Write a valid multi-line gzip, then chop the tail so decompression raises
    part-way through — mimicking a shard cut off by an unclean shutdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b""
    for i in range(2000):
        payload += (json.dumps({
            "received_at": "2026-08-21T00:00:00.000Z",
            "received_monotonic_ns": i,
            "event_type": "book",
            "assets": [{"universe": "testuni"}],
            "message": {"timestamp": str(i), "market": "m", "asset_id": "a",
                        "bids": [["0.5", "1"]], "asks": [["0.6", "1"]]},
        }) + "\n").encode("utf-8")
    blob = gzip.compress(payload)
    truncated = blob[: len(blob) // 2]  # drop the tail + trailer -> corrupt stream
    path.write_bytes(truncated)


def test_quarantine_and_continue(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "2026-08-21"
    good = raw_dir / "esports_00.jsonl.gz"
    bad = raw_dir / "esports_01.jsonl.gz"
    _write_valid_shard(good, n_rows=5)
    _write_truncated_shard(bad)

    parquet_root = tmp_path / "parquet"
    processed_state = parquet_root / "_processed.txt"
    quarantine_state = parquet_root / "_quarantine.txt"

    # Sanity: the bad shard really is unreadable (else the test proves nothing).
    with pytest.raises(Exception):
        with gzip.open(bad, "rt", encoding="utf-8") as fh:
            for _ in fh:
                pass

    summary = pipeline.process_files(
        [good, bad],
        parquet_root,
        processed_state=processed_state,
        quarantine_state=quarantine_state,
    )

    good_rp = str(good.resolve())
    bad_rp = str(bad.resolve())

    # The valid shard produced Parquet (book table) and was recorded processed.
    produced = list((parquet_root / "2026-08-21" / "testuni").glob("book_*.parquet"))
    assert produced, "valid shard should have produced a book Parquet file"
    assert summary["total_rows"] == 5
    processed_lines = processed_state.read_text(encoding="utf-8").split()
    assert good_rp in processed_lines

    # The truncated shard was quarantined, not processed.
    assert bad_rp in summary["quarantined"]
    assert quarantine_state.exists()
    quarantine_lines = quarantine_state.read_text(encoding="utf-8").split()
    assert bad_rp in quarantine_lines
    assert bad_rp not in processed_lines, "a quarantined shard must never be marked processed"

    # A second run keeps the bad path deduped in the quarantine file but still
    # reports it (so the run keeps exiting non-zero until the shard is fixed).
    summary2 = pipeline.process_files(
        [bad],
        parquet_root,
        processed_state=processed_state,
        quarantine_state=quarantine_state,
    )
    assert bad_rp in summary2["quarantined"]
    assert quarantine_state.read_text(encoding="utf-8").count(bad_rp) == 1
