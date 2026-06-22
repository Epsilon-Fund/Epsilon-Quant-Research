"""WebSocket capture daemon: subscribes to the Polymarket CLOB WS for every
asset in ``live_universe.json`` (hot-reloaded), stamps each message with arrival
timestamps, and appends raw events to hourly-rotated ``*.jsonl.gz`` files with
auto-reconnect and gap logging.

This is the critical 24/7 component. It captures raw L2 data only — no scoring,
no analysis. It talks to the discovery daemon ONLY through ``live_universe.json``
on disk (never imports it). Output:

    data/raw/{YYYY-MM-DD}/{universe}_{HH}.jsonl.gz   one file per universe per hour
    data/capture_gaps.jsonl                          health/event log

Run:
    python capture/daemon.py                       # 24/7 (production)
    python capture/daemon.py --duration-seconds 60 # bounded run (testing)

Both SIGINT and SIGTERM trigger the same graceful shutdown (flush + close all
shard files, write final stats). The bounded `--duration-seconds` path exits
through that same shutdown, so testing exercises the real teardown.

NOTE (capacity): this build uses a SINGLE WS connection. Polymarket's market
channel supports ~500 asset IDs per connection; the discovery volume floors keep
us near ~700, so the planned next step is sharding assets across 2 connections
(e.g. a select() loop over multiple sockets). Flagged, not yet implemented.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import signal
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# --- Package-relative paths (no imports from discovery — files are the API) ---
HERE = Path(__file__).resolve().parent          # .../l2_ingestion/capture
PKG_ROOT = HERE.parent                          # .../l2_ingestion
DEFAULT_UNIVERSE = PKG_ROOT / "data" / "live_universe.json"
DEFAULT_RAW_ROOT = PKG_ROOT / "data" / "raw"
DEFAULT_GAP_LOG = PKG_ROOT / "data" / "capture_gaps.jsonl"
DEFAULT_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Reconnect backoff schedule (seconds): 1 → 2 → 4 → ... → capped at 60.
RECONNECT_BACKOFF = [1, 2, 4, 8, 16, 30, 60]

logger = logging.getLogger("l2.capture")


# ---------------------------------------------------------------------------
# Per-asset metadata, loaded from live_universe.json. Carries the universe so
# every captured event can be routed to the right per-universe shard file and
# is self-describing on replay.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AssetMeta:
    asset_id: str
    universe: str
    condition_id: str = ""
    neg_risk: bool = False


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Envelope — adapted from dali_live_clob_capture.envelope. The timestamps are
# stamped FIRST, before any JSON parsing, so received_at/received_monotonic_ns
# reflect true arrival time (not parse time).
# ---------------------------------------------------------------------------
def event_type_of(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("event_type") or message.get("type") or "dict")
    if isinstance(message, list):
        return "list"
    return type(message).__name__


def asset_ids_of(message: Any) -> list[str]:
    ids: list[str] = []
    if not isinstance(message, dict):
        return ids
    for key in ("asset_id", "assetId"):
        value = message.get(key)
        if value:
            ids.append(str(value))
    for key in ("assets_ids", "asset_ids", "clob_token_ids", "clobTokenIds"):
        value = message.get(key)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
    for change in message.get("price_changes") or []:
        if isinstance(change, dict) and change.get("asset_id"):
            ids.append(str(change["asset_id"]))
    return sorted(set(ids))


def envelope(raw: str | bytes, metadata: dict[str, AssetMeta]) -> list[dict[str, Any]]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    received_at = utc_now_iso()
    received_monotonic_ns = time.monotonic_ns()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw

    messages = parsed if isinstance(parsed, list) else [parsed]
    out: list[dict[str, Any]] = []
    for message in messages:
        ids = asset_ids_of(message)
        out.append(
            {
                "received_at": received_at,
                "received_monotonic_ns": received_monotonic_ns,
                "event_type": event_type_of(message),
                "asset_id": ids[0] if ids else None,
                "asset_ids": ids,
                "assets": [asdict(metadata[i]) for i in ids if i in metadata],
                "message": message,
            }
        )
    return out


def route_universe(record: dict[str, Any], metadata: dict[str, AssetMeta]) -> str:
    """Which universe shard this event belongs to — first known asset wins."""
    for aid in record.get("asset_ids") or []:
        meta = metadata.get(aid)
        if meta is not None:
            return meta.universe
    return "unknown"


# ---------------------------------------------------------------------------
# Universe file loading. Returns (ws_url, metadata, generated_at, mtime).
# ---------------------------------------------------------------------------
def load_universe(path: Path) -> tuple[str, dict[str, AssetMeta], str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ws_url = data.get("ws_url") or DEFAULT_WS_URL
    meta: dict[str, AssetMeta] = {}
    for asset in data.get("assets", []):
        aid = str(asset.get("asset_id") or "")
        if not aid:
            continue
        meta[aid] = AssetMeta(
            asset_id=aid,
            universe=str(asset.get("universe") or "unknown"),
            condition_id=str(asset.get("condition_id") or ""),
            neg_risk=bool(asset.get("neg_risk", False)),
        )
    return ws_url, meta, str(data.get("generated_at") or ""), path.stat().st_mtime


# ---------------------------------------------------------------------------
# Graceful shutdown flag — same pattern as dali_block_a0_capture.StopFlag.
# ---------------------------------------------------------------------------
class StopFlag:
    def __init__(self) -> None:
        self.stop = False

    def install(self) -> None:
        def handler(signum: int, _frame: object) -> None:
            self.stop = True
            logger.info("received signal %s — shutting down after current loop", signum)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


# ---------------------------------------------------------------------------
# ShardManager — one open gzip file per universe for the current hour. Adapts
# the ShardWriter rotation pattern, but writes gzip directly (append-mode) so
# files are always {universe}_{HH}.jsonl.gz. Reopening within the same hour
# appends a new gzip member (transparent to gunzip), which handles restarts.
# ---------------------------------------------------------------------------
class ShardManager:
    def __init__(self, raw_root: Path) -> None:
        self.raw_root = raw_root
        self.handles: dict[str, Any] = {}
        self.paths: dict[str, Path] = {}
        self.hour_key: tuple[str, str] | None = None
        self.universe_counts: dict[str, Counter[str]] = defaultdict(Counter)

    @staticmethod
    def _current_key() -> tuple[str, str]:
        now = datetime.now(UTC)
        return now.strftime("%Y-%m-%d"), now.strftime("%H")

    def maybe_rotate(self) -> None:
        key = self._current_key()
        if self.hour_key is None:
            self.hour_key = key
        elif key != self.hour_key:
            self.close_all("rotated")
            self.hour_key = key

    def _handle(self, universe: str) -> Any:
        if universe in self.handles:
            return self.handles[universe]
        assert self.hour_key is not None
        date_str, hour = self.hour_key
        day_dir = self.raw_root / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{universe}_{hour}.jsonl.gz"
        fh = gzip.open(path, "at", encoding="utf-8")
        self.handles[universe] = fh
        self.paths[universe] = path
        logger.info("opened shard %s", path.name)
        return fh

    def write_line(self, universe: str, line: str) -> None:
        self._handle(universe).write(line + "\n")

    def flush(self) -> None:
        for fh in self.handles.values():
            try:
                fh.flush()
            except Exception:
                pass

    def close_all(self, status: str = "closed") -> None:
        for universe, fh in list(self.handles.items()):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass
            logger.info("closed shard %s (%s)", self.paths.get(universe, Path(universe)).name, status)
        self.handles.clear()
        self.paths.clear()


# ---------------------------------------------------------------------------
# Health log — every disconnect/reconnect/subscription-change/stale event.
# ---------------------------------------------------------------------------
def append_gap(path: Path, event_type: str, payload: dict[str, Any]) -> None:
    record = {"ts": utc_now_iso(), "event_type": event_type, **payload}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def make_subscription(asset_ids: list[str]) -> str:
    return json.dumps(
        {"assets_ids": asset_ids, "type": "market", "custom_feature_enabled": True}
    )


# ---------------------------------------------------------------------------
# Hot-reload — re-read the universe file, diff, subscribe additions on the live
# socket, return True if a reconnect is needed (removals: the market channel has
# no clean per-asset unsubscribe, so we reconnect with the new full list).
# metadata is mutated IN PLACE so the recv loop and reconnect see the new set.
# ---------------------------------------------------------------------------
def reload_universe(
    universe_path: Path,
    metadata: dict[str, AssetMeta],
    ws: Any,
    gap_log: Path,
) -> bool:
    try:
        _ws_url, new_meta, generated_at, _mtime = load_universe(universe_path)
    except Exception as exc:
        append_gap(gap_log, "reload_failed", {"error": repr(exc)})
        logger.warning("universe reload failed: %s", exc)
        return False

    old_ids = set(metadata)
    new_ids = set(new_meta)
    added = new_ids - old_ids
    removed = old_ids - new_ids

    # Always refresh metadata content (universe/neg_risk may change even if ids don't).
    metadata.clear()
    metadata.update(new_meta)

    if not added and not removed:
        return False

    logger.info("hot-reload: +%d / -%d assets (now %d, generated_at=%s)",
                len(added), len(removed), len(new_meta), generated_at)
    append_gap(gap_log, "subscription_change", {
        "added": len(added), "removed": len(removed),
        "total_now": len(new_meta), "generated_at": generated_at,
    })

    if added:
        try:
            ws.send(make_subscription(sorted(added)))
            append_gap(gap_log, "subscribe_delta", {"added": len(added)})
        except Exception as exc:
            append_gap(gap_log, "subscribe_delta_failed", {"error": repr(exc)})
            return True  # couldn't add live → reconnect with full list

    return bool(removed)  # removals require a reconnect


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------
def run(
    universe_path: Path,
    raw_root: Path,
    gap_log: Path,
    ws_url_override: str | None = None,
    duration_seconds: float | None = None,
    reload_seconds: float = 20.0,
    heartbeat_seconds: float = 60.0,
    stale_warning_seconds: float = 60.0,
) -> int:
    try:
        import websocket
        from websocket import WebSocketTimeoutException
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing websocket-client. Install it in this environment.") from exc

    if not universe_path.exists():
        raise SystemExit(
            f"universe file not found: {universe_path}\n"
            "Run discovery first: python discovery/daemon.py --once"
        )

    ws_url, metadata, generated_at, uni_mtime = load_universe(universe_path)
    if ws_url_override:
        ws_url = ws_url_override
    if not metadata:
        raise SystemExit("universe file has no assets to subscribe to")

    gap_log.parent.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    stop = StopFlag()
    stop.install()
    shards = ShardManager(raw_root)
    total_counts: Counter[str] = Counter()
    started_mono = time.monotonic()
    last_event_mono = time.monotonic()
    reconnect_attempt = 0
    disconnect_mono: float | None = None

    def expired() -> bool:
        return duration_seconds is not None and (time.monotonic() - started_mono) >= duration_seconds

    append_gap(gap_log, "capture_start", {
        "universe_file": str(universe_path), "generated_at": generated_at,
        "asset_count": len(metadata), "ws_url": ws_url,
    })
    logger.info("capture starting: %d assets, ws=%s", len(metadata), ws_url)
    if len(metadata) > 500:
        logger.warning("subscribing %d assets on a SINGLE connection (>~500 WS limit) — "
                       "shard across connections for production.", len(metadata))

    while not stop.stop and not expired():
        shards.maybe_rotate()
        try:
            asset_ids = list(metadata)
            append_gap(gap_log, "connect_attempt", {"attempt": reconnect_attempt + 1, "asset_count": len(asset_ids)})
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.send(make_subscription(asset_ids))

            downtime = None if disconnect_mono is None else round(time.monotonic() - disconnect_mono, 1)
            append_gap(gap_log, "connected", {
                "attempt": reconnect_attempt + 1, "asset_count": len(asset_ids),
                "downtime_seconds": downtime,
            })
            logger.info("connected (%d assets)%s", len(asset_ids),
                        f", downtime {downtime}s" if downtime else "")
            reconnect_attempt = 0
            disconnect_mono = None
            last_heartbeat = time.monotonic()
            hb_base = sum(total_counts.values())
            last_reload = time.monotonic()

            try:
                while not stop.stop and not expired():
                    now = time.monotonic()
                    shards.maybe_rotate()

                    # --- heartbeat: rate, subs, staleness ---
                    if now - last_heartbeat >= heartbeat_seconds:
                        total = sum(total_counts.values())
                        rate = (total - hb_base) / (now - last_heartbeat) if now > last_heartbeat else 0.0
                        stale = now - last_event_mono
                        logger.info("heartbeat: %d events, %.1f msg/s, %d subs, %.0fs since last | %s",
                                    total, rate, len(metadata), stale, dict(total_counts))
                        append_gap(gap_log, "heartbeat", {
                            "total_counts": dict(total_counts), "msgs_per_sec": round(rate, 2),
                            "subscription_count": len(metadata), "seconds_since_last_event": round(stale, 1),
                        })
                        if stale >= stale_warning_seconds:
                            append_gap(gap_log, "stale_warning", {"seconds_since_last_event": round(stale, 1)})
                            logger.warning("STALE: no messages for %.0fs", stale)
                        last_heartbeat = now
                        hb_base = total

                    # --- hot-reload: pick up discovery's new universe file ---
                    if now - last_reload >= reload_seconds:
                        last_reload = now
                        try:
                            new_mtime = universe_path.stat().st_mtime
                        except FileNotFoundError:
                            new_mtime = uni_mtime
                        if new_mtime != uni_mtime:
                            uni_mtime = new_mtime
                            if reload_universe(universe_path, metadata, ws, gap_log):
                                logger.info("reconnecting to apply universe changes")
                                break  # outer loop reconnects with updated metadata

                    # --- receive + write ---
                    try:
                        raw = ws.recv()
                    except WebSocketTimeoutException:
                        continue
                    records = envelope(raw, metadata)
                    for rec in records:
                        universe = route_universe(rec, metadata)
                        line = json.dumps(rec, sort_keys=True, separators=(",", ":"))
                        shards.write_line(universe, line)
                        etype = str(rec.get("event_type") or "unknown")
                        total_counts[etype] += 1
                        shards.universe_counts[universe][etype] += 1
                    shards.flush()
                    last_event_mono = time.monotonic()
            finally:
                try:
                    ws.close()
                except Exception:
                    pass
        except Exception as exc:
            if stop.stop or expired():
                break
            disconnect_mono = time.monotonic()
            reconnect_attempt += 1
            delay = RECONNECT_BACKOFF[min(reconnect_attempt - 1, len(RECONNECT_BACKOFF) - 1)]
            append_gap(gap_log, "disconnect_or_error", {
                "attempt": reconnect_attempt, "error": repr(exc), "backoff_seconds": delay,
            })
            logger.warning("disconnect/error: %s — reconnecting in %ds (attempt %d)",
                           exc, delay, reconnect_attempt)
            time.sleep(delay)

    # --- graceful shutdown ---
    shards.close_all("stopped" if stop.stop else "complete")
    append_gap(gap_log, "capture_end", {
        "total_counts": dict(total_counts),
        "stopped_by_signal": stop.stop,
        "per_universe": {u: dict(c) for u, c in shards.universe_counts.items()},
    })
    logger.info("shutdown complete: %d events total %s",
                sum(total_counts.values()), dict(total_counts))
    logger.info("per-universe: %s", {u: dict(c) for u, c in shards.universe_counts.items()})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Polymarket L2 capture daemon")
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--gap-log", type=Path, default=DEFAULT_GAP_LOG)
    parser.add_argument("--ws-url", default=None, help="override ws_url from the universe file")
    parser.add_argument("--duration-seconds", type=float, default=None,
                        help="bounded run for testing; omit for 24/7")
    parser.add_argument("--reload-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    return run(
        universe_path=args.universe_file,
        raw_root=args.raw_root,
        gap_log=args.gap_log,
        ws_url_override=args.ws_url,
        duration_seconds=args.duration_seconds,
        reload_seconds=args.reload_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
