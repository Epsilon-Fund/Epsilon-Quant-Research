"""Durable Block A0 live CLOB capture runner.

This runner is meant for 24-48h captures. It keeps the same JSONL envelope
shape as ``dali_live_clob_capture.py`` while adding reconnects, hourly shard
rotation, heartbeat/gap logs, and fee-aware manifests.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.dali_live_clob_capture import TokenMeta, envelope


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "block_a0_capture.generated.yaml"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


class StopFlag:
    def __init__(self) -> None:
        self.stop = False

    def install(self) -> None:
        def handler(signum: int, _frame: object) -> None:
            self.stop = True
            print(f"{utc_now()} received signal {signum}; closing after current loop")

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


class ShardWriter:
    def __init__(self, run_dir: Path, run_id: str, config: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.config = config
        self.path: Path | None = None
        self.fh: Any | None = None
        self.started_at = ""
        self.counts: Counter[str] = Counter()

    def open_new(self) -> None:
        self.close("rotated")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.run_dir / f"{self.run_id}_{stamp}.jsonl"
        self.started_at = utc_now()
        self.counts = Counter()
        self.fh = self.path.open("a", encoding="utf-8")
        self.write_manifest(status="open")
        print(f"{utc_now()} opened shard {self.path.relative_to(ROOT)}")

    def write_manifest(self, status: str) -> None:
        if self.path is None:
            return
        manifest = {
            "run_id": self.run_id,
            "status": status,
            "started_at": self.started_at,
            "updated_at": utc_now(),
            "jsonl_path": str(self.path.relative_to(ROOT)),
            "counts": dict(self.counts),
            "markets": self.config.get("markets", []),
            "capture": self.config.get("capture", {}),
            "run": self.config.get("run", {}),
        }
        self.path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_records(self, records: list[dict[str, Any]]) -> None:
        if self.fh is None:
            self.open_new()
        assert self.fh is not None
        for rec in records:
            self.counts[str(rec.get("event_type") or "unknown")] += 1
            self.fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        self.fh.flush()

    def close(self, status: str = "closed") -> None:
        if self.fh is not None:
            self.fh.flush()
            self.fh.close()
            self.fh = None
            self.write_manifest(status=status)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text()) or {}
    if not config.get("markets"):
        raise SystemExit(
            f"{path} has no markets. Run scripts/dali_block_a0_prepare.py first."
        )
    return config


def build_tokens(config: dict[str, Any]) -> tuple[list[str], dict[str, TokenMeta]]:
    metadata: dict[str, TokenMeta] = {}
    for market in config["markets"]:
        for outcome_index, token_id in enumerate(market.get("clob_token_ids") or []):
            metadata[str(token_id)] = TokenMeta(
                token_id=str(token_id),
                market_id=str(market.get("id") or ""),
                question=str(market.get("question") or ""),
                slug=str(market.get("slug") or ""),
                family=str(market.get("family") or ""),
                outcome_index=outcome_index,
            )
    return list(metadata), metadata


def append_control(log_path: Path, event_type: str, payload: dict[str, Any]) -> None:
    record = {"ts": utc_now(), "event_type": event_type, **payload}
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def run(config: dict[str, Any], config_path: Path) -> Path:
    try:
        import websocket
        from websocket import WebSocketTimeoutException
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing websocket-client in this environment.") from exc

    run_cfg = config["run"]
    cap_cfg = config["capture"]
    run_id = str(run_cfg["run_id"])
    out_dir = rel(cap_cfg["out_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config_copy = out_dir / "capture_config.yaml"
    run_config_copy.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    token_ids, metadata = build_tokens(config)
    if not token_ids:
        raise SystemExit("no token ids in generated config")

    total_duration_s = float(run_cfg.get("duration_hours", 24)) * 3600
    rotate_s = float(run_cfg.get("rotate_minutes", 60)) * 60
    heartbeat_s = float(run_cfg.get("heartbeat_seconds", 60))
    stale_warning_s = float(run_cfg.get("stale_warning_seconds", 900))
    print_every = int(run_cfg.get("print_every_events", 500))
    backoff = [float(x) for x in run_cfg.get("reconnect_backoff_seconds", [1, 2, 4, 8, 16, 30])]
    ws_url = str(cap_cfg["ws_url"])
    subscription = {
        "assets_ids": token_ids,
        "type": "market",
        "custom_feature_enabled": bool(cap_cfg.get("custom_feature_enabled", True)),
    }

    stop = StopFlag()
    stop.install()
    writer = ShardWriter(out_dir, run_id, config)
    gap_log = out_dir / "capture_gaps.jsonl"
    total_counts: Counter[str] = Counter()
    started_monotonic = time.monotonic()
    next_rotate = started_monotonic
    last_event_monotonic = time.monotonic()
    reconnect_attempt = 0

    append_control(
        gap_log,
        "capture_start",
        {
            "config": display_path(config_path),
            "run_dir": str(out_dir.relative_to(ROOT)),
            "token_count": len(token_ids),
            "market_count": len(config["markets"]),
            "mode": "gap-tolerant" if run_cfg.get("tolerate_gaps", True) else "continuous",
        },
    )

    while not stop.stop and (time.monotonic() - started_monotonic) < total_duration_s:
        now = time.monotonic()
        if writer.fh is None or now >= next_rotate:
            writer.open_new()
            next_rotate = now + rotate_s

        try:
            append_control(gap_log, "connect_attempt", {"attempt": reconnect_attempt + 1})
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.send(json.dumps(subscription))
            append_control(gap_log, "connected", {"attempt": reconnect_attempt + 1})
            reconnect_attempt = 0
            last_heartbeat = time.monotonic()

            try:
                while not stop.stop and (time.monotonic() - started_monotonic) < total_duration_s:
                    now = time.monotonic()
                    if now >= next_rotate:
                        writer.open_new()
                        next_rotate = now + rotate_s
                    if now - last_heartbeat >= heartbeat_s:
                        stale = now - last_event_monotonic
                        writer.write_manifest(status="open")
                        append_control(
                            gap_log,
                            "heartbeat",
                            {
                                "total_counts": dict(total_counts),
                                "seconds_since_last_event": round(stale, 3),
                                "current_shard": (
                                    str(writer.path.relative_to(ROOT)) if writer.path else None
                                ),
                            },
                        )
                        if stale >= stale_warning_s:
                            append_control(
                                gap_log,
                                "stale_warning",
                                {"seconds_since_last_event": round(stale, 3)},
                            )
                        last_heartbeat = now
                    try:
                        raw = ws.recv()
                    except WebSocketTimeoutException:
                        continue
                    records = envelope(raw, metadata)
                    writer.write_records(records)
                    last_event_monotonic = time.monotonic()
                    for rec in records:
                        total_counts[str(rec.get("event_type") or "unknown")] += 1
                    total = sum(total_counts.values())
                    if total and total % print_every == 0:
                        print(f"{utc_now()} total events {total} {dict(total_counts)}")
            finally:
                try:
                    ws.close()
                except Exception:
                    pass
        except Exception as exc:
            reconnect_attempt += 1
            delay = backoff[min(reconnect_attempt - 1, len(backoff) - 1)]
            append_control(
                gap_log,
                "disconnect_or_error",
                {"attempt": reconnect_attempt, "error": repr(exc), "backoff_seconds": delay},
            )
            time.sleep(delay)

    writer.close("complete" if not stop.stop else "stopped")
    append_control(
        gap_log,
        "capture_end",
        {"total_counts": dict(total_counts), "stopped_by_signal": stop.stop},
    )
    print(f"{utc_now()} done {dict(total_counts)}")
    print(f"run dir: {out_dir.relative_to(ROOT)}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration-hours", type=float, help="override config duration")
    parser.add_argument("--run-id", help="override config run_id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.duration_hours is not None:
        config.setdefault("run", {})["duration_hours"] = args.duration_hours
    if args.run_id:
        config.setdefault("run", {})["run_id"] = args.run_id
    run(config, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
