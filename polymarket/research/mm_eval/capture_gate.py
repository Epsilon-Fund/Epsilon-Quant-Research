"""Capture-quality gate — REQUIRED pre-analysis check for captured L2.

Why this exists
---------------
[[pm_dali_workflow_revision_decision]] Tier 2 item 2: the dali era's "A1 gate" encoded
no thresholds; nothing gets a feature panel until the capture passes an explicit,
thresholded reconstruction gate. This module ports the checks from
``scripts/mm_reconstruction_audit.py`` (Alvaro's lane) + ``mm_engine.book.BookTracker``
into a library gate any analysis can (and must) call before consuming a captured slice:

1. **BBA-checksum classification, lead-lag-aware.** Replay ``book`` + ``price_change``
   through :class:`~mm_engine.book.BookTracker`; at every ``best_bid_ask`` checkpoint
   classify:

   * ``clean``        — fresh book, reconstructed L1 == native L1 at arrival.
   * ``clean_lagged`` — fresh book, mismatch at arrival, but the reconstruction matches
     once ALL same-millisecond ``book``/``price_change`` updates for the token are
     applied. This is the intra-ms ordering artifact diagnosed in
     [[mm_reconstruction_audit_findings]] (the BBA frame's receive clock precedes its
     own triggering ``price_change``); counting it as a mismatch understates
     reconstruction quality ~3x on fast books.
   * ``stale``        — the tracker flags the book unusable (no anchor / staleness
     window / pending gap / incomplete); we would not quote off it, so it is excluded
     from the clean/mismatch denominator but capped by ``max_stale_pct``.
   * ``mismatch``     — fresh book, and the disagreement survives the same-ms flush:
     genuine silent reconstruction error.

2. **Staleness gating**: the tracker's ≤5s rule (``DEFAULT_STALENESS_MS``) is applied
   in-stream; every consumer downstream sees ``BookState.stale``.

3. **Gap handling**: explicit ``capture_gaps.parquet`` sidecars are honored when present
   (interleaved as ``GapMarker`` -> ``note_gap``). The rolling VPS cloud archive does NOT
   ship gap sidecars (verified 2026-07-21: no ``capture_gaps`` and no ``metadata/`` prefix
   in ``r2:epsilon-polymarket-data``), so the gate ALSO infers gaps from the receive-clock
   heartbeat: a silence longer than ``gap_infer_ms`` across the whole universe stream
   marks every book suspect until its next snapshot. Inferred gaps are reported as
   ``gaps_inferred`` and labelled as inference, not ground truth.

4. **Trade-in-spread coherence**: fraction of ``last_trade`` prints falling inside the
   reconstructed [best_bid, best_ask] when the book is fresh and two-sided.

5. **Exchange-ts ordering**: events are merged in the engine's canonical lookahead-free
   order (``mm_engine.feeds._merge.order_and_interleave`` — ``ts_exchange`` primary);
   malformed timestamps (``ts_exchange <= 0``) are counted and capped.

Pass thresholds (explicit, defaults below)
------------------------------------------
* ``min_fresh_clean_pct`` (default **95.0**): ``(clean + clean_lagged) / (clean +
  clean_lagged + mismatch) * 100`` — reconstruction accuracy over decisive (fresh)
  checkpoints, the corrected metric from [[mm_join1_reconciliation_findings]]. 85–95 is
  MARGINAL, <85 FAIL (same ladder as the reconstruction audit's verdict).
* ``min_trade_in_spread_pct`` (default **95.0**).
* ``max_stale_pct`` (default **20.0**) of all checkpoints.
* ``min_checks`` (default **1000**) BBA checkpoints, else the slice is UNDERPOWERED
  (cannot pass; can only be waived explicitly).

Usage::

    from mm_eval.capture_gate import CaptureQualityGate, require_pass
    gate = CaptureQualityGate()
    result = gate.run_slice(Path("data/l2_parquet_full/2026-06-20/politics_negrisk"))
    require_pass(result)            # raises CaptureQualityError unless PASS/MARGINAL
    # analysis may proceed; optionally pass on_state= to record book states in-pass

Memory note: slices are replayed **hour-chunk by hour-chunk** (shard suffix = capture
hour) with one persistent tracker, so a 50M-event day never materializes at once.
Cross-chunk ordering is preserved to shard granularity (shards are cut on the capture
wall clock; the canonical sort is re-imposed within each chunk).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb

from mm_engine.book import DEFAULT_STALENESS_MS, BookTracker
from mm_engine.events import GapMarker, bba_event, book_event, price_change_event, trade_event
from mm_engine.feeds._merge import order_and_interleave
from mm_engine.feeds.replay_parquet import SCHEMA, _maybe_json_levels, _read_rows, load_parquet_gaps
from mm_engine.interfaces import BookState, MarketEvent

BBO_TOL = 1e-9
_SHARD_RE = re.compile(r"_(\d+)\.parquet$")


# --------------------------------------------------------------------------------------
# thresholds + result containers
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GateThresholds:
    """Explicit pass thresholds — see module docstring for definitions."""

    min_fresh_clean_pct: float = 95.0     # PASS floor on lead-lag-aware fresh-clean %
    marginal_fresh_clean_pct: float = 85.0  # MARGINAL floor (below -> FAIL)
    min_trade_in_spread_pct: float = 95.0
    max_stale_pct: float = 20.0
    min_checks: int = 1000


@dataclass
class TokenStats:
    checks: int = 0
    clean: int = 0
    clean_lagged: int = 0
    mismatch: int = 0
    stale: int = 0
    stale_no_anchor: int = 0
    stale_window: int = 0
    stale_incomplete: int = 0
    trades: int = 0
    trades_in_spread: int = 0
    trades_outside: int = 0

    def add(self, other: "TokenStats") -> None:
        for f_ in self.__dataclass_fields__:
            setattr(self, f_, getattr(self, f_) + getattr(other, f_))

    @property
    def fresh(self) -> int:
        return self.clean + self.clean_lagged + self.mismatch

    @property
    def fresh_clean_pct(self) -> float:
        return 100.0 * (self.clean + self.clean_lagged) / self.fresh if self.fresh else 0.0

    @property
    def raw_clean_pct(self) -> float:
        return 100.0 * self.clean / self.fresh if self.fresh else 0.0

    @property
    def stale_pct(self) -> float:
        return 100.0 * self.stale / self.checks if self.checks else 0.0

    @property
    def trade_in_spread_pct(self) -> float:
        return 100.0 * self.trades_in_spread / self.trades if self.trades else 0.0


@dataclass
class SliceGateResult:
    """Gate outcome for one or more (day, universe) directories replayed as a stream."""

    slice_id: str
    thresholds: GateThresholds
    overall: TokenStats
    per_token: dict[str, TokenStats]
    event_counts: dict[str, int]
    gaps_sidecar: int = 0          # explicit capture_gaps entries interleaved
    gaps_inferred: int = 0         # heartbeat-silence inferences (labelled, not ground truth)
    bad_ts_events: int = 0         # ts_exchange <= 0 (malformed; replayed at stream head)

    @property
    def verdict(self) -> str:
        t, s = self.thresholds, self.overall
        if s.checks < t.min_checks:
            return "UNDERPOWERED"
        if s.stale_pct > t.max_stale_pct:
            return "FAIL"
        if s.fresh_clean_pct >= t.min_fresh_clean_pct and s.trade_in_spread_pct >= t.min_trade_in_spread_pct:
            return "PASS"
        if s.fresh_clean_pct >= t.marginal_fresh_clean_pct:
            return "MARGINAL"
        return "FAIL"

    def to_dict(self) -> dict:
        return {
            "slice_id": self.slice_id,
            "verdict": self.verdict,
            "thresholds": asdict(self.thresholds),
            "overall": asdict(self.overall),
            "derived": {
                "fresh_clean_pct": self.overall.fresh_clean_pct,
                "raw_clean_pct": self.overall.raw_clean_pct,
                "stale_pct": self.overall.stale_pct,
                "trade_in_spread_pct": self.overall.trade_in_spread_pct,
            },
            "event_counts": self.event_counts,
            "gaps_sidecar": self.gaps_sidecar,
            "gaps_inferred": self.gaps_inferred,
            "bad_ts_events": self.bad_ts_events,
            "n_tokens": len(self.per_token),
        }

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))


class CaptureQualityError(RuntimeError):
    """Raised by :func:`require_pass` when a slice fails the capture-quality gate."""


def require_pass(result: SliceGateResult, *, allow_marginal: bool = True) -> None:
    """Fail-closed guard: analysis must call this before consuming a slice."""
    ok = {"PASS", "MARGINAL"} if allow_marginal else {"PASS"}
    if result.verdict not in ok:
        raise CaptureQualityError(
            f"capture slice {result.slice_id} gate verdict={result.verdict} "
            f"(fresh_clean={result.overall.fresh_clean_pct:.2f}%, "
            f"stale={result.overall.stale_pct:.2f}%, "
            f"trade_in_spread={result.overall.trade_in_spread_pct:.2f}%, "
            f"checks={result.overall.checks})"
        )


# --------------------------------------------------------------------------------------
# chunked event loading (hourly shards -> canonical order within each chunk)
# --------------------------------------------------------------------------------------
def _shard_hours(directory: Path) -> list[str]:
    hours: set[str] = set()
    for p in directory.glob("*.parquet"):
        m = _SHARD_RE.search(p.name)
        if m:
            hours.add(m.group(1))
    return sorted(hours)


def _chunk_events(con: duckdb.DuckDBPyConnection, directory: Path, hour: str) -> list[MarketEvent]:
    events: list[MarketEvent] = []

    def files(table: str) -> list[str]:
        return sorted(str(p) for p in directory.glob(f"{table}_*_{hour}.parquet"))

    fb = files("book")
    if fb:
        for ts, rcv_at, rcv_ns, aid, mkt, bids, asks in _read_rows(con, fb, SCHEMA["book"]):
            events.append(book_event(
                asset_id=aid, market=mkt,
                bids=_maybe_json_levels(bids), asks=_maybe_json_levels(asks),
                ts_exchange=ts, ts_local_iso=rcv_at, ts_monotonic_ns=rcv_ns,
            ))
    ft = files("trades")
    if ft:
        for ts, rcv_at, rcv_ns, aid, mkt, price, size, side in _read_rows(con, ft, SCHEMA["trades"]):
            events.append(trade_event(
                asset_id=aid, market=mkt, price=price, side=side, size=size,
                ts_exchange=ts, ts_local_iso=rcv_at, ts_monotonic_ns=rcv_ns,
            ))
    fp = files("price_change")
    if fp:
        for ts, rcv_at, rcv_ns, aid, mkt, price, side, size in _read_rows(con, fp, SCHEMA["price_change"]):
            events.append(price_change_event(
                asset_id=aid, market=mkt, price=price, side=side, size=size,
                ts_exchange=ts, ts_local_iso=rcv_at, ts_monotonic_ns=rcv_ns,
            ))
    fa = files("bba")
    if fa:
        for ts, rcv_at, rcv_ns, aid, mkt, bbid, bask, bsz, asz in _read_rows(con, fa, SCHEMA["bba"]):
            events.append(bba_event(
                asset_id=aid, market=mkt, best_bid=bbid, best_ask=bask,
                bid_size=bsz, ask_size=asz,
                ts_exchange=ts, ts_local_iso=rcv_at, ts_monotonic_ns=rcv_ns,
            ))
    return events


def iter_slice_events(
    directories: list[Path],
    *,
    gaps: list[int] | None = None,
) -> Iterator[MarketEvent | GapMarker]:
    """Canonically-ordered event stream over slice dirs, one hour-chunk at a time.

    Same event builders and same ``order_and_interleave`` as the engine's Parquet
    replay adapter, so the gate audits exactly the stream analyses consume — chunked
    hourly so multi-day slices never materialize in memory at once.
    """
    if gaps is None:
        gaps = load_parquet_gaps(directories)
    remaining = sorted(set(gaps))
    # cap DuckDB's thread pool: this iterator runs inside worker processes, and each
    # worker grabbing every core oversubscribes the box (worker right-sizing rule)
    con = duckdb.connect(config={"threads": 2})
    try:
        for d in directories:
            for hour in _shard_hours(d):
                events = _chunk_events(con, d, hour)
                if not events:
                    continue
                # feed only the gaps that can fire in this chunk; keep the rest pending
                yield from order_and_interleave(events, remaining)
                # order_and_interleave consumes gaps positionally; recompute what's left
                if remaining:
                    last_recv = max(ev.ts_exchange for ev in events)
                    remaining = [g for g in remaining if g > last_recv]
    finally:
        con.close()


# --------------------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------------------
@dataclass
class _PendingBBA:
    ts_exchange: int
    claimed_bid: float | None
    claimed_ask: float | None
    raw_match: bool


def _f(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out


def _match(recon: float | None, native: float | None) -> bool:
    if recon is None and native is None:
        return True
    if recon is None or native is None:
        return False
    return abs(recon - native) <= BBO_TOL


@dataclass
class CaptureQualityGate:
    thresholds: GateThresholds = field(default_factory=GateThresholds)
    staleness_ms: int = DEFAULT_STALENESS_MS
    gap_infer_ms: int = 30_000   # universe-wide receive-clock silence -> inferred gap

    def run_slice(
        self,
        source: Path | list[Path],
        *,
        slice_id: str | None = None,
        on_state: Callable[[MarketEvent, BookState], None] | None = None,
    ) -> SliceGateResult:
        """Replay one slice (dir or dir list) and classify every checkpoint.

        ``on_state``, when given, is called with ``(event, post-event BookState)`` for
        every market event — the hook analyses use to record book states in the SAME
        gated pass instead of replaying twice.
        """
        dirs = [source] if isinstance(source, Path) else [Path(p) for p in source]
        sid = slice_id or ";".join(str(d) for d in dirs)
        return self.run_events(iter_slice_events(dirs), slice_id=sid, on_state=on_state)

    def run_events(
        self,
        stream: Iterator[MarketEvent | GapMarker],
        *,
        slice_id: str = "<events>",
        on_state: Callable[[MarketEvent, BookState], None] | None = None,
    ) -> SliceGateResult:
        """Classify an already-ordered event stream (the core loop; testable directly)."""
        tracker = BookTracker(staleness_ms=self.staleness_ms)
        per_token: dict[str, TokenStats] = defaultdict(TokenStats)
        counts: dict[str, int] = defaultdict(int)
        pending: dict[str, list[_PendingBBA]] = defaultdict(list)
        gaps_sidecar = 0
        gaps_inferred = 0
        bad_ts = 0
        last_recv_ms: int | None = None

        def _resolve(token_id: str) -> None:
            """Final same-ms-flushed classification for parked checkpoints."""
            for p in pending.pop(token_id, []):
                state = tracker.snapshot(token_id, p.ts_exchange)
                s = per_token[token_id]
                recon_bid = state.bids[0][0] if state.bids else None
                recon_ask = state.asks[0][0] if state.asks else None
                lag_match = _match(recon_bid, p.claimed_bid) and _match(recon_ask, p.claimed_ask)
                if p.raw_match:
                    s.clean += 1
                elif lag_match:
                    s.clean_lagged += 1
                else:
                    s.mismatch += 1

        for ev in stream:
            if isinstance(ev, GapMarker):
                tracker.note_gap(ev)
                gaps_sidecar += 1
                continue

            counts[ev.type] += 1
            if ev.ts_exchange <= 0:
                bad_ts += 1

            # heartbeat gap inference (universe-wide receive-clock silence)
            recv_ms = ev.ts_exchange
            if last_recv_ms is not None and recv_ms - last_recv_ms > self.gap_infer_ms:
                tracker.note_gap()
                gaps_inferred += 1
            if last_recv_ms is None or recv_ms > last_recv_ms:
                last_recv_ms = recv_ms

            # a later-ts event closes the same-ms window -> resolve parked checkpoints
            if pending.get(ev.token_id) and ev.ts_exchange > pending[ev.token_id][0].ts_exchange:
                _resolve(ev.token_id)

            state = tracker.apply(ev)

            if ev.type == "best_bid_ask":
                s = per_token[ev.token_id]
                s.checks += 1
                if state.stale:
                    s.stale += 1
                    reason = self._stale_reason(tracker, ev.token_id, ev.ts_exchange)
                    if reason == "no_anchor":
                        s.stale_no_anchor += 1
                    elif reason == "incomplete":
                        s.stale_incomplete += 1
                    else:
                        s.stale_window += 1
                else:
                    recon_bid = state.bids[0][0] if state.bids else None
                    recon_ask = state.asks[0][0] if state.asks else None
                    raw = _match(recon_bid, _f(ev.payload.get("best_bid"))) and _match(
                        recon_ask, _f(ev.payload.get("best_ask"))
                    )
                    pending[ev.token_id].append(_PendingBBA(
                        ts_exchange=ev.ts_exchange,
                        claimed_bid=_f(ev.payload.get("best_bid")),
                        claimed_ask=_f(ev.payload.get("best_ask")),
                        raw_match=raw,
                    ))

            elif ev.type == "last_trade":
                if not state.stale and state.bids and state.asks:
                    price = _f(ev.payload.get("price"))
                    if price is not None:
                        s = per_token[ev.token_id]
                        s.trades += 1
                        if state.bids[0][0] - BBO_TOL <= price <= state.asks[0][0] + BBO_TOL:
                            s.trades_in_spread += 1
                        else:
                            s.trades_outside += 1

            if on_state is not None:
                on_state(ev, state)

        for token_id in list(pending):
            _resolve(token_id)

        overall = TokenStats()
        for s in per_token.values():
            overall.add(s)
        return SliceGateResult(
            slice_id=slice_id,
            thresholds=self.thresholds,
            overall=overall,
            per_token=dict(per_token),
            event_counts=dict(counts),
            gaps_sidecar=gaps_sidecar,
            gaps_inferred=gaps_inferred,
            bad_ts_events=bad_ts,
        )

    @staticmethod
    def _stale_reason(tracker: BookTracker, token_id: str, ts: int) -> str:
        tb = tracker._books.get(token_id)
        if tb is None or not tb.anchored or tb.last_depth_ts is None:
            return "no_anchor"
        if not tb.book.is_complete:
            return "incomplete"
        return "window"
