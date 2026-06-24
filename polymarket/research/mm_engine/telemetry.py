"""Raw, append-only JSONL telemetry — NOT pre-aggregated.

The engine emits three raw event streams so Alvaro's validation layer can compute ND-PnL /
PnLMAP / markout / adverse-selection downstream (we deliberately log the primitives, not
summaries):

* **fills** — one row per realized fill: ts, token, side, price, qty, ``queue_ahead`` (from
  :class:`~mm_engine.interfaces.FillResult`), mid-at-fill, position/cash after, and the
  triggering trade — enough to derive markout at +1/+5/+30/+60s by joining to the quotes/mid
  trajectory.
* **orders** — one row per place/cancel/replace/throttled op (with ts + client id).
* **quotes** — one row per strategy evaluation (every event): ts, token, event_type, stale,
  best_bid/ask, mid, and per resting order a ``get_queue_ahead`` snapshot. This stream
  doubles as the per-token mid trajectory markout needs.

Each :class:`JsonlSink` writes append-only JSONL when given a path and (by default) keeps the
records in memory so the engine can return them and the reconciliation harness can diff them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JsonlSink:
    path: Path | None = None
    keep: bool = True
    records: list[dict] = field(default_factory=list)
    _fh: Any = None

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, rec: dict) -> None:
        if self.keep:
            self.records.append(rec)
        if self._fh is not None:
            self._fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


@dataclass
class Telemetry:
    fills: JsonlSink = field(default_factory=JsonlSink)
    orders: JsonlSink = field(default_factory=JsonlSink)
    quotes: JsonlSink = field(default_factory=JsonlSink)

    @classmethod
    def in_memory(cls) -> "Telemetry":
        return cls(JsonlSink(), JsonlSink(), JsonlSink())

    @classmethod
    def to_dir(cls, out_dir: Path | str, *, prefix: str = "", keep: bool = True) -> "Telemetry":
        out = Path(out_dir)
        tag = f"{prefix}_" if prefix else ""
        return cls(
            fills=JsonlSink(out / f"{tag}fills.jsonl", keep=keep),
            orders=JsonlSink(out / f"{tag}orders.jsonl", keep=keep),
            quotes=JsonlSink(out / f"{tag}quotes.jsonl", keep=keep),
        )

    def close(self) -> None:
        self.fills.close()
        self.orders.close()
        self.quotes.close()
