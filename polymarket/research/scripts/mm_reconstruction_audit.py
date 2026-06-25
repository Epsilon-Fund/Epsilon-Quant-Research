#!/usr/bin/env python3
"""MM L2 Reconstruction Audit — the go/no-go gate for the MM backtester.

Question this answers: *does our captured public L2 (``book`` + ``price_change``)
let us reconstruct the order book cleanly enough to trust a fill simulation?* If the
reconstructed top-of-book does not agree with the exchange's native ``best_bid_ask``
checksum, then queue position and fills built on top of that book are fiction, and the
whole backtester is unsafe to build. So this runs FIRST (Phase 1, Task 1 — see
[[mm_engine_phase01_buildplan]]).

Method (spec: [[mm_clob_capture_semantics]] § Required Reconstruction Audit):

1.  Per day, per universe, load ``book_*`` / ``price_change_*`` / ``bba_*`` / ``trades_*``
    Parquet shards via DuckDB and merge into ONE event stream ordered by
    ``timestamp_ms`` (PM exchange time), tie-broken by ``received_ns`` then arrival —
    the same lookahead-free ordering the replay adapter uses.
2.  Replay the stream through ``mm_engine.book.BookTracker`` *as-is* — we are testing
    Justin's book builder, not rewriting it. ``book`` events anchor a fresh snapshot;
    ``price_change`` events apply deltas; ``best_bid_ask`` is telemetry-only.
3.  At every ``best_bid_ask`` checkpoint, compare the reconstructed L1 (top bid / top
    ask) against the native ``best_bid`` / ``best_ask`` and classify the interval:
      * **clean**     — book is fresh (not stale) AND reconstructed L1 == native L1.
      * **stale**     — ``BookState.stale`` is True (no anchor yet, beyond the staleness
                        window, or a pending gap) — we would not quote off it.
      * **mismatch**  — fresh book, but reconstructed L1 disagrees with native L1. These
                        are the concerning ones: silent reconstruction error.
4.  At every ``last_trade`` print, check trade-book coherence: does the trade price fall
    inside the reconstructed [best_bid, best_ask]? Trades printing outside the spread
    suggest the book is lagging the tape.
5.  Aggregate per market (token), per universe, and overall; surface the top-10 worst
    markets by mismatch rate; write a findings note.

Diagnostic only: we REPORT mismatches, we never patch them.

Run (from ``polymarket/research``):
    PYTHONPATH=. uv run python scripts/mm_reconstruction_audit.py --data-dir ./l2_data/

Repo invariants honored: DuckDB over Parquet; events ordered by ``ts_exchange`` before
any aggregation (lookahead-free); the captured Parquet is read-only.

NOTE on gaps: ``capture_gaps.jsonl`` is NOT part of the Parquet pull, so this standalone
audit cannot interleave gap markers. "Stale" intervals here therefore come from
*no-anchor* or *staleness-window* only, never from an explicitly-signalled capture gap.
The breakdown reported under "stale reason" makes that explicit.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

# Make `mm_engine` importable when run as `python scripts/mm_reconstruction_audit.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mm_engine.book import DEFAULT_STALENESS_MS, BookTracker  # noqa: E402
from mm_engine.interfaces import MarketEvent  # noqa: E402

BBO_TOL = 1e-9  # float tolerance for "reconstructed L1 == native L1"
WORST_MIN_CHECKS = 50  # min bba checkpoints before a market is eligible for "worst" lists
EVENT_KINDS = ("book", "price_change", "bba", "trades")


# --------------------------------------------------------------------------------------
# Parquet -> MarketEvent
# --------------------------------------------------------------------------------------
def _f(value) -> float | None:
    """Coerce a Parquet cell to float, mapping NULL / NaN to None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _load_kind(con: duckdb.DuckDBPyConnection, files: list[Path]) -> list[tuple]:
    """Read all shards of one event kind ordered by (timestamp_ms, received_ns).

    Returns raw DuckDB rows; column handling is done per-kind in :func:`_build_events`.
    """
    if not files:
        return []
    globs = [str(p).replace("\\", "/") for p in files]
    rel = con.sql(
        "SELECT * FROM read_parquet($globs) ORDER BY timestamp_ms, received_ns",
        params={"globs": globs},
    )
    return rel.fetchall(), [c for c in rel.columns]  # type: ignore[return-value]


def _rows_to_dicts(rows_and_cols) -> list[dict]:
    rows, cols = rows_and_cols
    return [dict(zip(cols, r)) for r in rows]


def _build_events(day_dir: Path, universe: str, con: duckdb.DuckDBPyConnection):
    """Build the merged, time-ordered MarketEvent stream for one (day, universe).

    Returns ``(events, counts)`` where ``counts`` is per-kind raw row counts.
    """
    counts: dict[str, int] = {}
    tagged: list[tuple[int, int, int, MarketEvent]] = []
    seq = 0

    def _shards(prefix: str) -> list[Path]:
        return sorted(day_dir.glob(f"{prefix}_{universe}_*.parquet"))

    # book ---------------------------------------------------------------------------
    book_rows = _rows_to_dicts(_load_kind(con, _shards("book")))
    counts["book"] = len(book_rows)
    for r in book_rows:
        try:
            bids = json.loads(r.get("bids") or "[]")
            asks = json.loads(r.get("asks") or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            bids, asks = [], []
        payload = {"bids": bids, "asks": asks, "market": r.get("market")}
        ev = _mk("book", r, payload)
        tagged.append((ev.ts_exchange, ev.ts_monotonic_ns, seq, ev))
        seq += 1

    # price_change -------------------------------------------------------------------
    pc_rows = _rows_to_dicts(_load_kind(con, _shards("price_change")))
    counts["price_change"] = len(pc_rows)
    for r in pc_rows:
        payload = {
            "side": r.get("side"),
            "price": r.get("price"),
            "size": r.get("size"),
            "best_bid": r.get("best_bid"),
            "best_ask": r.get("best_ask"),
            "market": r.get("market"),
        }
        ev = _mk("price_change", r, payload)
        tagged.append((ev.ts_exchange, ev.ts_monotonic_ns, seq, ev))
        seq += 1

    # best_bid_ask (telemetry checkpoint) --------------------------------------------
    bba_rows = _rows_to_dicts(_load_kind(con, _shards("bba")))
    counts["bba"] = len(bba_rows)
    for r in bba_rows:
        payload = {"best_bid": r.get("best_bid"), "best_ask": r.get("best_ask")}
        ev = _mk("best_bid_ask", r, payload)
        tagged.append((ev.ts_exchange, ev.ts_monotonic_ns, seq, ev))
        seq += 1

    # trades (last_trade) ------------------------------------------------------------
    tr_rows = _rows_to_dicts(_load_kind(con, _shards("trades")))
    counts["trades"] = len(tr_rows)
    for r in tr_rows:
        payload = {"price": r.get("price"), "size": r.get("size"), "side": r.get("side")}
        ev = _mk("last_trade", r, payload)
        tagged.append((ev.ts_exchange, ev.ts_monotonic_ns, seq, ev))
        seq += 1

    # Lookahead-free ordering: exchange time, then local monotonic clock, then arrival.
    tagged.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in tagged], counts


def _mk(etype: str, row: dict, payload: dict) -> MarketEvent:
    """Map a Parquet row to a MarketEvent using the agreed field mapping."""
    ts = row.get("timestamp_ms")
    try:
        ts_exchange = int(ts)
    except (TypeError, ValueError):
        ts_exchange = 0
    ns = row.get("received_ns")
    try:
        ts_ns = int(ns)
    except (TypeError, ValueError):
        ts_ns = 0
    return MarketEvent(
        type=etype,
        token_id=str(row.get("asset_id") or ""),
        ts_exchange=ts_exchange,
        ts_local_iso=str(row.get("received_at") or ""),
        ts_monotonic_ns=ts_ns,
        payload=payload,
    )


# --------------------------------------------------------------------------------------
# Aggregation containers
# --------------------------------------------------------------------------------------
@dataclass
class Stats:
    checks: int = 0          # bba checkpoints seen
    clean: int = 0
    stale: int = 0
    mismatch: int = 0
    # stale sub-reasons (why BookState.stale was True at the checkpoint)
    stale_no_anchor: int = 0
    stale_window: int = 0
    stale_incomplete: int = 0
    stale_but_correct: int = 0   # stale checkpoints whose recon L1 *would* have matched
    # trade-book coherence
    trades: int = 0              # last_trade prints with a fresh two-sided book
    trades_in_spread: int = 0
    trades_outside: int = 0

    def add(self, other: "Stats") -> None:
        for f_ in self.__dataclass_fields__:
            setattr(self, f_, getattr(self, f_) + getattr(other, f_))

    @property
    def clean_pct(self) -> float:
        return 100.0 * self.clean / self.checks if self.checks else 0.0

    @property
    def stale_pct(self) -> float:
        return 100.0 * self.stale / self.checks if self.checks else 0.0

    @property
    def mismatch_pct(self) -> float:
        return 100.0 * self.mismatch / self.checks if self.checks else 0.0

    @property
    def trade_in_spread_pct(self) -> float:
        return 100.0 * self.trades_in_spread / self.trades if self.trades else 0.0


def _match(recon: float | None, native: float | None) -> bool:
    if recon is None and native is None:
        return True
    if recon is None or native is None:
        return False
    return abs(recon - native) <= BBO_TOL


def _stale_reason(tracker: BookTracker, token_id: str, ts: int) -> str:
    """Inspect tracker internals to label WHY a checkpoint was stale (read-only)."""
    tb = tracker._books.get(token_id)
    if tb is None or not tb.anchored or tb.last_depth_ts is None:
        return "no_anchor"
    if not tb.book.is_complete:
        return "incomplete"
    if (ts - tb.last_depth_ts) > tracker.staleness_ms:
        return "window"
    return "window"  # default bucket if stale for some other transient reason


# --------------------------------------------------------------------------------------
# Core audit per (day, universe)
# --------------------------------------------------------------------------------------
def audit_day_universe(events: list[MarketEvent], staleness_ms: int):
    """Replay events through BookTracker; return per-token Stats for this slice."""
    tracker = BookTracker(staleness_ms=staleness_ms)
    per_token: dict[str, Stats] = defaultdict(Stats)

    for ev in events:
        state = tracker.apply(ev)

        if ev.type == "best_bid_ask":
            s = per_token[ev.token_id]
            s.checks += 1
            recon_bid = state.bids[0][0] if state.bids else None
            recon_ask = state.asks[0][0] if state.asks else None
            native_bid = _f(ev.payload.get("best_bid"))
            native_ask = _f(ev.payload.get("best_ask"))
            matched = _match(recon_bid, native_bid) and _match(recon_ask, native_ask)

            if state.stale:
                s.stale += 1
                reason = _stale_reason(tracker, ev.token_id, ev.ts_exchange)
                if reason == "no_anchor":
                    s.stale_no_anchor += 1
                elif reason == "incomplete":
                    s.stale_incomplete += 1
                else:
                    s.stale_window += 1
                if matched:
                    s.stale_but_correct += 1
            elif matched:
                s.clean += 1
            else:
                s.mismatch += 1

        elif ev.type == "last_trade":
            if state.stale or not state.bids or not state.asks:
                continue
            price = _f(ev.payload.get("price"))
            if price is None:
                continue
            s = per_token[ev.token_id]
            s.trades += 1
            best_bid = state.bids[0][0]
            best_ask = state.asks[0][0]
            if best_bid - BBO_TOL <= price <= best_ask + BBO_TOL:
                s.trades_in_spread += 1
            else:
                s.trades_outside += 1

    return per_token


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------
def discover(data_dir: Path) -> list[tuple[str, str, Path]]:
    """Find (date, universe, dir) slices under ``data_dir/{date}/{universe}/``."""
    slices: list[tuple[str, str, Path]] = []
    for date_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for uni_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            if any(uni_dir.glob("book_*.parquet")) or any(uni_dir.glob("bba_*.parquet")):
                slices.append((date_dir.name, uni_dir.name, uni_dir))
    return slices


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def _fmt_row(label: str, s: Stats, width: int = 28) -> str:
    return (
        f"{label[:width]:<{width}} "
        f"{s.checks:>10} "
        f"{s.clean_pct:>9.2f} "
        f"{s.stale_pct:>9.2f} "
        f"{s.mismatch_pct:>10.2f} "
        f"{s.trade_in_spread_pct:>11.2f}"
    )


def _header(width: int = 28) -> str:
    return (
        f"{'slice':<{width}} "
        f"{'checks':>10} "
        f"{'clean%':>9} "
        f"{'stale%':>9} "
        f"{'mismatch%':>10} "
        f"{'trade_in%':>11}"
    )


def verdict(clean_pct: float) -> str:
    if clean_pct >= 95.0:
        return "PASS (>=95% clean) — proceed to the queue model (Task 2)."
    if clean_pct >= 85.0:
        return "MARGINAL (85-95% clean) — investigate stale/mismatch drivers before proceeding."
    return "FAIL (<85% clean) — data-quality problem that blocks the backtester."


def build_markdown(
    *,
    overall: Stats,
    per_universe: dict[str, Stats],
    per_token: dict[str, tuple[str, Stats]],
    counts_total: dict[str, int],
    date_range: tuple[str, str],
    total_events: int,
    slices: list[tuple[str, str, Path]],
    staleness_ms: int,
    data_dir: Path,
) -> str:
    worst = sorted(
        (
            (tok, uni, st)
            for tok, (uni, st) in per_token.items()
            if st.checks >= WORST_MIN_CHECKS
        ),
        key=lambda x: x[2].mismatch_pct,
        reverse=True,
    )[:10]
    stale_worst = sorted(
        (
            (tok, uni, st)
            for tok, (uni, st) in per_token.items()
            if st.checks >= WORST_MIN_CHECKS
        ),
        key=lambda x: x[2].stale_pct,
        reverse=True,
    )[:10]

    d0, d1 = date_range
    lines: list[str] = []
    lines.append("---")
    lines.append('title: "MM L2 Reconstruction Audit — can we rebuild the book cleanly?"')
    lines.append("status: active")
    lines.append("owner: alvaro")
    lines.append("project: mm")
    lines.append("para: project")
    lines.append("hubs:")
    lines.append("  - strat_market_making")
    lines.append("tags:")
    lines.append("  - market-making")
    lines.append("  - data-quality")
    lines.append("  - backtesting")
    lines.append("---")
    lines.append("")
    lines.append("# MM L2 Reconstruction Audit — Can We Rebuild the Book Cleanly?")
    lines.append("")
    lines.append(
        "> Hub: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · "
        "Method spec: [[mm_clob_capture_semantics]] § Required Reconstruction Audit · "
        "Build plan: [[mm_engine_phase01_buildplan]]"
    )
    lines.append("")
    lines.append("## Plain-English Summary")
    lines.append("")
    lines.append(
        "- **What this is.** The foundational go/no-go gate for the market-making "
        "backtester. We replay captured public L2 (`book` snapshots + `price_change` "
        "deltas) through the engine's `BookTracker` and check the reconstructed "
        "top-of-book against the exchange's own `best_bid_ask` checksum."
    )
    lines.append(
        "- **Why it was written.** If the book doesn't reconstruct cleanly, every "
        "downstream realism model (queue position, fill simulation) is built on a "
        "fictional book. So this must pass *before* the queue model (Task 2) is built."
    )
    lines.append(
        f"- **What it covers.** All Parquet shards under `{data_dir.as_posix()}` — "
        f"dates {d0} → {d1}, universes {sorted(per_universe)}, "
        f"{total_events:,} merged events."
    )
    lines.append(
        f"- **Headline result.** Across {overall.checks:,} `best_bid_ask` checkpoints: "
        f"**{overall.clean_pct:.2f}% clean**, {overall.stale_pct:.2f}% stale, "
        f"**{overall.mismatch_pct:.2f}% mismatch**. "
        f"Trade-in-spread coherence: {overall.trade_in_spread_pct:.2f}% of "
        f"{overall.trades:,} prints landed inside the reconstructed spread."
    )
    lines.append(f"- **Verdict.** {verdict(overall.clean_pct)}")
    lines.append("")

    lines.append("## How to read the three buckets")
    lines.append("")
    lines.append(
        "Every `best_bid_ask` (BBA) event is a checkpoint: the exchange tells us its own "
        "best bid and best ask, and we compare that against the L1 we reconstructed from "
        "`book`+`price_change`. Each checkpoint lands in exactly one bucket:"
    )
    lines.append("")
    lines.append(
        "- **clean** — the book was fresh (not stale) and our reconstructed best "
        "bid/ask matched the native best bid/ask within `1e-9`. This is the only subset "
        "that should drive fill/queue claims."
    )
    lines.append(
        "- **stale** — `BookState.stale` was `True`: either no full `book` snapshot has "
        "anchored the token yet (*no_anchor*), the book sat longer than the "
        f"{staleness_ms / 1000:.0f}s staleness window since its last update (*window*), "
        "or the book is one-sided/incomplete (*incomplete*). A stale book is one we "
        "would refuse to quote off — it is not a reconstruction *error*, it is the "
        "engine correctly flagging uncertainty."
    )
    lines.append(
        "- **mismatch** — the book was fresh but our L1 disagreed with the native L1. "
        "These are the concerning ones: silent reconstruction error that a fill "
        "simulator would never notice."
    )
    lines.append("")
    lines.append(
        "**Worked example.** A token gets a `book` snapshot at `t=0` (best bid 0.40 / "
        "ask 0.42). At `t=1.2s` a `price_change` lifts the bid to 0.41. At `t=1.3s` a "
        "`best_bid_ask` says best_bid=0.41, best_ask=0.42 → we reconstructed 0.41/0.42 → "
        "**clean**. If instead we had missed the `price_change`, we'd still show 0.40 → "
        "**mismatch**. If 8s passed with no update before the BBA, the book is past the "
        "5s window → **stale**."
    )
    lines.append("")
    lines.append(
        "**`stale_but_correct`** counts stale checkpoints whose reconstructed L1 *would* "
        "have matched the native L1 anyway. A high value means staleness is conservative "
        "book-keeping (quiet market, sparse updates) rather than genuine data loss — an "
        "important realism distinction (see [[mm_engine_phase01_buildplan]] and the "
        "realism-calibration rules in `brain/CODEX.md`)."
    )
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| bba checkpoints | {overall.checks:,} |")
    lines.append(f"| clean % | {overall.clean_pct:.2f} |")
    lines.append(f"| stale % | {overall.stale_pct:.2f} |")
    lines.append(f"| mismatch % | {overall.mismatch_pct:.2f} |")
    lines.append(
        f"| stale breakdown | no_anchor={overall.stale_no_anchor:,}, "
        f"window={overall.stale_window:,}, incomplete={overall.stale_incomplete:,} |"
    )
    lines.append(
        f"| stale_but_correct | {overall.stale_but_correct:,} "
        f"({100.0 * overall.stale_but_correct / overall.stale if overall.stale else 0:.1f}% of stale) |"
    )
    lines.append(
        f"| trades checked / in-spread % | {overall.trades:,} / "
        f"{overall.trade_in_spread_pct:.2f} |"
    )
    lines.append(f"| total events processed | {total_events:,} |")
    lines.append(
        "| raw rows | "
        + ", ".join(f"{k}={v:,}" for k, v in counts_total.items())
        + " |"
    )
    lines.append("")
    lines.append(
        "*Unit of observation: one `best_bid_ask` checkpoint (for the three buckets) or "
        "one `last_trade` print with a fresh two-sided book (for trade coherence). "
        "Percentages are of `bba checkpoints` / `trades checked` respectively.*"
    )
    lines.append("")

    lines.append("## Per-universe")
    lines.append("")
    lines.append(
        "| universe | checks | clean % | stale % | mismatch % | stale_but_correct % | trades | trade_in_spread % |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for uni in sorted(per_universe):
        s = per_universe[uni]
        sbc = 100.0 * s.stale_but_correct / s.stale if s.stale else 0.0
        lines.append(
            f"| {uni} | {s.checks:,} | {s.clean_pct:.2f} | {s.stale_pct:.2f} | "
            f"{s.mismatch_pct:.2f} | {sbc:.1f} | {s.trades:,} | {s.trade_in_spread_pct:.2f} |"
        )
    lines.append("")
    lines.append(
        "*Read: a universe with high clean% and near-zero mismatch% reconstructs "
        "reliably. High stale% with high stale_but_correct% is usually a quiet market "
        "(sparse updates tripping the staleness window), not broken data. High "
        "mismatch% is the real red flag.*"
    )
    lines.append("")

    lines.append("## Top-10 worst markets by mismatch rate")
    lines.append("")
    lines.append(
        f"*Only markets with >= {WORST_MIN_CHECKS} checkpoints are eligible (small "
        "samples are noise). `token_id` is the CLOB token. These are where "
        "reconstruction is weakest and where to look first for capture/ordering bugs.*"
    )
    lines.append("")
    if worst:
        lines.append("| token_id (truncated) | universe | checks | mismatch % | stale % | clean % |")
        lines.append("|---|---|---|---|---|---|")
        for tok, uni, st in worst:
            lines.append(
                f"| `{tok[:18]}…` | {uni} | {st.checks:,} | {st.mismatch_pct:.2f} | "
                f"{st.stale_pct:.2f} | {st.clean_pct:.2f} |"
            )
    else:
        lines.append("_No markets met the minimum-checkpoint threshold._")
    lines.append("")

    lines.append("## Top-10 markets by stale rate")
    lines.append("")
    lines.append(
        "*High stale% is usually benign (quiet market) — cross-read with "
        "stale_but_correct%. Listed for coverage diagnostics, not as a failure list.*"
    )
    lines.append("")
    if stale_worst:
        lines.append("| token_id (truncated) | universe | checks | stale % | stale_but_correct % | mismatch % |")
        lines.append("|---|---|---|---|---|---|")
        for tok, uni, st in stale_worst:
            sbc = 100.0 * st.stale_but_correct / st.stale if st.stale else 0.0
            lines.append(
                f"| `{tok[:18]}…` | {uni} | {st.checks:,} | {st.stale_pct:.2f} | "
                f"{sbc:.1f} | {st.mismatch_pct:.2f} |"
            )
    else:
        lines.append("_No markets met the minimum-checkpoint threshold._")
    lines.append("")

    lines.append("## Decision and next step")
    lines.append("")
    lines.append(f"**Gate:** {verdict(overall.clean_pct)}")
    lines.append("")
    lines.append(
        "- **If clean% >= 95 and mismatch% is small:** the book reconstructs faithfully; "
        "proceed to the queue model (Task 2 — `OptimisticQueue` / `RiskAverseQueue` / "
        "`ProbQueue`). The clean subset is what drives quoteability/fill claims."
    )
    lines.append(
        "- **If mismatch% is non-trivial (> ~2-3%):** do NOT proceed. Investigate the "
        "worst markets above for missed `price_change` events, same-timestamp ordering "
        "ambiguity, or capture races, before any fill simulation is trusted."
    )
    lines.append(
        "- **If stale% is high but stale_but_correct% is also high:** this is mostly "
        "quiet markets tripping the staleness window, not data loss — confirm by "
        "checking update cadence, and consider whether the 5s window is right per "
        "universe. It is not, by itself, a blocker."
    )
    lines.append("")
    lines.append("### Caveats / assumption ledger")
    lines.append("")
    lines.append(
        "- **No capture-gap signal.** `capture_gaps.jsonl` is not part of the Parquet "
        "pull, so `BookTracker.note_gap` is never called here. Stale intervals are "
        "*no_anchor* / *window* / *incomplete* only — not explicitly-signalled WS "
        "disconnects. A full reconciliation run (with the gap log) may reclassify some "
        "intervals as gap-stale. Re-run with the gap log when available."
    )
    lines.append(
        "- **BBO-only checksum.** We validate L1 (top-of-book) against `best_bid_ask`. "
        "Deeper levels are reconstructed but not independently checksummed by the feed, "
        "so depth accuracy is assumed-correct-if-L1-is-correct, not proven."
    )
    lines.append(
        "- **Same-timestamp ordering.** Events sharing a `timestamp_ms` are ordered by "
        "`received_ns` then arrival — the same rule the replay adapter uses. A residual "
        "mismatch rate at sub-ms bursts is expected and is itself a finding."
    )
    lines.append("")
    lines.append(
        f"Generated by `scripts/mm_reconstruction_audit.py` over `{data_dir.as_posix()}` "
        f"(staleness window {staleness_ms} ms; {len(slices)} day×universe slices)."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MM L2 reconstruction audit (go/no-go gate).")
    ap.add_argument("--data-dir", required=True, help="Dir of pulled Parquet: {date}/{universe}/*.parquet")
    ap.add_argument(
        "--out",
        default="notes/overview/data_quality/mm_reconstruction_audit_findings.md",
        help="Markdown findings output path (relative to polymarket/research).",
    )
    ap.add_argument("--staleness-ms", type=int, default=DEFAULT_STALENESS_MS)
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        print(f"ERROR: --data-dir not found: {data_dir}", file=sys.stderr)
        return 2

    slices = discover(data_dir)
    if not slices:
        print(f"ERROR: no {{date}}/{{universe}}/*.parquet slices under {data_dir}", file=sys.stderr)
        return 2

    con = duckdb.connect()
    per_universe: dict[str, Stats] = defaultdict(Stats)
    per_token: dict[str, tuple[str, Stats]] = {}
    counts_total: dict[str, int] = defaultdict(int)
    total_events = 0
    dates: set[str] = set()

    print(f"Reconstruction audit over {len(slices)} day×universe slices "
          f"(staleness={args.staleness_ms} ms)\n")
    for date, universe, day_dir in slices:
        events, counts = _build_events(day_dir, universe, con)
        for k, v in counts.items():
            counts_total[k] += v
        total_events += len(events)
        dates.add(date)

        token_stats = audit_day_universe(events, args.staleness_ms)
        slice_stat = Stats()
        for tok, st in token_stats.items():
            slice_stat.add(st)
            per_universe[universe].add(st)
            if tok in per_token:
                per_token[tok][1].add(st)
            else:
                merged = Stats()
                merged.add(st)
                per_token[tok] = (universe, merged)

        print(
            f"  {date} / {universe:<18} events={len(events):>9,}  "
            f"checks={slice_stat.checks:>7,}  clean={slice_stat.clean_pct:6.2f}%  "
            f"stale={slice_stat.stale_pct:6.2f}%  mismatch={slice_stat.mismatch_pct:6.2f}%"
        )

    overall = Stats()
    for s in per_universe.values():
        overall.add(s)

    # --- stdout summary table -------------------------------------------------------
    print("\n" + "=" * 92)
    print("SUMMARY  (unit = best_bid_ask checkpoint; trade_in% = last_trade prints inside recon spread)")
    print("=" * 92)
    print(_header())
    print("-" * 92)
    for uni in sorted(per_universe):
        print(_fmt_row(uni, per_universe[uni]))
    print("-" * 92)
    print(_fmt_row("OVERALL", overall))
    print("=" * 92)
    print(f"\nVerdict: {verdict(overall.clean_pct)}\n")

    # --- markdown -------------------------------------------------------------------
    date_range = (min(dates), max(dates)) if dates else ("?", "?")
    md = build_markdown(
        overall=overall,
        per_universe=dict(per_universe),
        per_token=per_token,
        counts_total=dict(counts_total),
        date_range=date_range,
        total_events=total_events,
        slices=slices,
        staleness_ms=args.staleness_ms,
        data_dir=data_dir,
    )
    out_path = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Findings written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
