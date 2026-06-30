"""Replay feed adapter: captured CLOB JSONL -> ordered ``MarketEvent`` stream.

Reads the JSONL envelope shards written by ``scripts/dali_live_clob_capture.py`` /
``scripts/dali_block_a0_capture.py`` (format per [[mm_clob_capture_semantics]]), converts
each record with the shared :func:`mm_engine.events.envelope_to_events`, and yields the
events in ``ts_exchange`` order — lookahead-free, exactly as the repo invariants require.

Capture gaps from ``capture_gaps.jsonl`` are honored: a :class:`GapMarker` is interleaved
at each disconnect, so the book tracker marks the book stale from that point until the
next full ``book`` snapshot re-anchors it (build-plan Phase 0, item 2).

Source may be a single ``.jsonl``/``.jsonl.gz`` file, a list of files, or a run directory
(its ``*.jsonl`` shards, excluding ``capture_gaps.jsonl`` and ``*.manifest.json``). When a
run directory or a file with a sibling ``capture_gaps.jsonl`` is given, gaps load
automatically unless an explicit gap list is passed.
"""
from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from mm_engine.events import GapMarker, envelope_to_events, _iso_to_epoch_ms
from mm_engine.feeds._merge import order_and_interleave
from mm_engine.interfaces import MarketEvent


# capture_gaps.jsonl control event_types that constitute a real data gap.
GAP_EVENT_TYPES = frozenset({"disconnect_or_error"})

# The gaps log ships as capture_gaps.jsonl OR capture_gaps.jsonl.gz (real R2 captures gzip it);
# either way it is a control log, never a data shard.
_GAP_LOG_STEM = "capture_gaps"


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_lines(path: Path) -> Iterator[dict]:
    with _open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _resolve_shards(source) -> list[Path]:
    """Resolve ``source`` to an ordered list of JSONL shard paths."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            shards = [
                p
                for p in path.iterdir()
                if not p.name.startswith(_GAP_LOG_STEM)   # excludes capture_gaps.jsonl[.gz]
                and not p.name.endswith(".manifest.json")
                and (p.suffix == ".jsonl" or p.name.endswith(".jsonl.gz"))
            ]
            return sorted(shards)
        return [path]
    if isinstance(source, Iterable):
        return [Path(p) for p in source]
    raise TypeError(f"unsupported replay source: {source!r}")


def _gap_log_for(source) -> Path | None:
    """Find a sibling/contained ``capture_gaps.jsonl`` (or ``.jsonl.gz``) for ``source``."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        base = path if path.is_dir() else path.parent
        for name in ("capture_gaps.jsonl", "capture_gaps.jsonl.gz"):
            candidate = base / name
            if candidate.exists():
                return candidate
    return None


def load_capture_gaps(gap_log: Path) -> list[int]:
    """Return sorted disconnect timestamps (ms epoch) from a ``capture_gaps.jsonl``."""
    times: list[int] = []
    for rec in _iter_lines(Path(gap_log)):
        if str(rec.get("event_type")) not in GAP_EVENT_TYPES:
            continue
        ms = _iso_to_epoch_ms(rec.get("ts"))
        if ms is not None:
            times.append(ms)
    return sorted(times)


def replay_feed(
    source,
    *,
    gaps: list[int] | None = None,
) -> Iterator[MarketEvent | GapMarker]:
    """Yield ``MarketEvent``s in ``ts_exchange`` order, interleaving capture-gap markers.

    ``source``: a file, list of files, or run directory. ``gaps``: explicit list of
    disconnect timestamps (ms epoch); if ``None``, loaded from a sibling/contained
    ``capture_gaps.jsonl`` when present.
    """
    shards = _resolve_shards(source)
    events: list[MarketEvent] = [
        ev for shard in shards for rec in _iter_lines(shard) for ev in envelope_to_events(rec)
    ]

    if gaps is None:
        gap_log = _gap_log_for(source)
        gaps = load_capture_gaps(gap_log) if gap_log is not None else []

    # Shared ordering + gap interleaving (identical to the Parquet adapter — see _merge).
    yield from order_and_interleave(events, gaps)
