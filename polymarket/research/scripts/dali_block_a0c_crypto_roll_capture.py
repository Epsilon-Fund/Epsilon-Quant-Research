"""Rolling BTC/ETH/SOL 4h up/down capture for Block A0c.

The durable A0 capture runner subscribes to a static token list. This supervisor
adds lightweight rollover coverage by periodically rediscovering listed 4h
crypto windows, writing a chunk config, and running the existing capture runner
for a short interval. Chunking creates tiny reconnect gaps, but avoids mutating
the main A0c universe capture and picks up newly listed future windows.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import yaml

from scripts.dali_block_a0c_prepare import CandidateSpec, event_markets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT_RUN_ID = "block_a0c_crypto_roll_20260529_morning"
ASSET_SLUG_PREFIXES = {
    "BTC": "btc-updown-4h",
    "ETH": "eth-updown-4h",
    "SOL": "sol-updown-4h",
}


@dataclass
class StopState:
    stop: bool = False


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": utc_now(), **payload}, sort_keys=True) + "\n")


def align_4h_floor(ts: datetime) -> datetime:
    ts = ts.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    return ts.replace(hour=(ts.hour // 4) * 4)


def window_starts(now: datetime, lookahead_hours: float) -> list[datetime]:
    start = align_4h_floor(now)
    end = now + timedelta(hours=lookahead_hours)
    out: list[datetime] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(hours=4)
    return out


def make_spec(asset: str, start: datetime) -> CandidateSpec:
    slug = f"{ASSET_SLUG_PREFIXES[asset]}-{int(start.timestamp())}"
    end = start + timedelta(hours=4)
    return CandidateSpec(
        slug=slug,
        family="crypto_4h_up_down",
        category_hint="Crypto",
        rationale=(
            f"crypto_4h_roll: {asset} 4h window "
            f"{start.isoformat(timespec='seconds').replace('+00:00', 'Z')} to "
            f"{end.isoformat(timespec='seconds').replace('+00:00', 'Z')}"
        ),
        force_include=True,
    )


def discover_markets(
    *,
    assets: list[str],
    lookahead_hours: float,
    log_path: Path,
) -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    with httpx.Client(timeout=20) as client:
        for start in window_starts(now, lookahead_hours):
            for asset in assets:
                spec = make_spec(asset, start)
                try:
                    rows = event_markets(client, spec)
                except Exception as exc:
                    append_jsonl(
                        log_path,
                        {
                            "event_type": "discovery_miss",
                            "slug": spec.slug,
                            "error": repr(exc),
                        },
                    )
                    continue
                if not rows:
                    append_jsonl(log_path, {"event_type": "discovery_empty", "slug": spec.slug})
                    continue
                for row in rows[:1]:
                    key = str(row.get("condition_id") or row.get("id") or row.get("slug"))
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    row["selection_rationale"] = spec.rationale
                    metrics = row.setdefault("selection_metrics", {})
                    metrics["roll_window_start"] = start.isoformat(timespec="seconds").replace(
                        "+00:00", "Z"
                    )
                    metrics["roll_window_end"] = (
                        start + timedelta(hours=4)
                    ).isoformat(timespec="seconds").replace("+00:00", "Z")
                    selected.append(row)
    selected.sort(key=lambda row: (row.get("end_date") or "", row.get("slug") or ""))
    append_jsonl(
        log_path,
        {
            "event_type": "discovery_complete",
            "market_count": len(selected),
            "token_count": sum(len(m.get("clob_token_ids") or []) for m in selected),
            "slugs": [m.get("slug") for m in selected],
        },
    )
    return selected


def build_config(
    *,
    chunk_run_id: str,
    parent_run_id: str,
    duration_hours: float,
    markets: list[dict[str, Any]],
    refresh_minutes: float,
    lookahead_hours: float,
) -> dict[str, Any]:
    return {
        "run": {
            "run_id": chunk_run_id,
            "label": "block_a0c_crypto_roll",
            "duration_hours": duration_hours,
            "rotate_minutes": 60,
            "print_every_events": 1000,
            "heartbeat_seconds": 60,
            "stale_warning_seconds": 900,
            "reconnect_backoff_seconds": [1, 2, 4, 8, 16, 30],
            "tolerate_gaps": True,
        },
        "capture": {
            "ws_url": "wss://ws-subscriptions-clob.polymarket.com/ws/market",
            "custom_feature_enabled": True,
            "out_dir": f"data/live_clob/block_a0c_crypto_roll/{parent_run_id}",
        },
        "prepared_at": utc_now(),
        "notes": [
            "Rolling A0c companion capture; keep separate from the main A0c targeted panel.",
            "The supervisor refreshes discovery between chunks to pick up newly listed 4h windows.",
            f"Refresh interval minutes: {refresh_minutes}; lookahead hours: {lookahead_hours}.",
        ],
        "markets": markets,
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, width=100), encoding="utf-8")


def install_signal_handlers(state: StopState, child_holder: dict[str, subprocess.Popen[Any] | None]) -> None:
    def handler(signum: int, _frame: object) -> None:
        state.stop = True
        child = child_holder.get("child")
        if child and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def run_chunk(config_path: Path, chunk_run_id: str, duration_hours: float, child_holder: dict[str, Any]) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "dali_block_a0_capture.py"),
        "--config",
        str(config_path),
        "--run-id",
        chunk_run_id,
        "--duration-hours",
        str(duration_hours),
    ]
    child = subprocess.Popen(cmd, cwd=ROOT, env=env)
    child_holder["child"] = child
    try:
        return child.wait()
    finally:
        child_holder["child"] = None


def parse_assets(raw: str) -> list[str]:
    assets = [part.strip().upper() for part in raw.split(",") if part.strip()]
    unknown = [asset for asset in assets if asset not in ASSET_SLUG_PREFIXES]
    if unknown:
        raise SystemExit(f"unknown assets: {unknown}; allowed: {sorted(ASSET_SLUG_PREFIXES)}")
    return assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run-id", default=DEFAULT_PARENT_RUN_ID)
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--refresh-minutes", type=float, default=60.0)
    parser.add_argument("--lookahead-hours", type=float, default=28.0)
    parser.add_argument("--assets", default="BTC,ETH,SOL")
    parser.add_argument("--once", action="store_true", help="discover once and run a single chunk")
    parser.add_argument("--dry-run", action="store_true", help="write config without launching capture")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets = parse_assets(args.assets)
    parent_root = ROOT / "data" / "live_clob" / "block_a0c_crypto_roll" / args.parent_run_id
    config_root = ROOT / "configs" / "block_a0c_crypto_roll" / args.parent_run_id
    log_path = parent_root / "roll_supervisor.jsonl"
    latest_config = ROOT / "configs" / "block_a0c_crypto_roll.generated.yaml"
    parent_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    stop_state = StopState()
    child_holder: dict[str, subprocess.Popen[Any] | None] = {"child": None}
    install_signal_handlers(stop_state, child_holder)

    deadline = time.monotonic() + args.duration_hours * 3600
    chunk_idx = 0
    append_jsonl(
        log_path,
        {
            "event_type": "roll_start",
            "parent_run_id": args.parent_run_id,
            "duration_hours": args.duration_hours,
            "refresh_minutes": args.refresh_minutes,
            "lookahead_hours": args.lookahead_hours,
            "assets": assets,
        },
    )

    while not stop_state.stop and time.monotonic() < deadline:
        remaining_hours = max((deadline - time.monotonic()) / 3600, 0)
        chunk_hours = min(args.refresh_minutes / 60, remaining_hours)
        if chunk_hours <= 0:
            break
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        chunk_run_id = f"{args.parent_run_id}_chunk_{chunk_idx:03d}_{stamp}"
        markets = discover_markets(assets=assets, lookahead_hours=args.lookahead_hours, log_path=log_path)
        if not markets:
            append_jsonl(log_path, {"event_type": "no_markets_sleep", "sleep_seconds": 60})
            time.sleep(60)
            continue
        config = build_config(
            chunk_run_id=chunk_run_id,
            parent_run_id=args.parent_run_id,
            duration_hours=chunk_hours,
            markets=markets,
            refresh_minutes=args.refresh_minutes,
            lookahead_hours=args.lookahead_hours,
        )
        config_path = config_root / f"{chunk_run_id}.yaml"
        write_config(config_path, config)
        write_config(latest_config, config)
        append_jsonl(
            log_path,
            {
                "event_type": "chunk_prepared",
                "chunk_index": chunk_idx,
                "chunk_run_id": chunk_run_id,
                "config": str(config_path.relative_to(ROOT)),
                "duration_hours": chunk_hours,
                "market_count": len(markets),
                "token_count": sum(len(m.get("clob_token_ids") or []) for m in markets),
            },
        )
        if args.dry_run:
            print(f"wrote {config_path.relative_to(ROOT)}")
            return 0
        code = run_chunk(config_path, chunk_run_id, chunk_hours, child_holder)
        append_jsonl(
            log_path,
            {
                "event_type": "chunk_complete",
                "chunk_index": chunk_idx,
                "chunk_run_id": chunk_run_id,
                "return_code": code,
            },
        )
        if args.once:
            break
        chunk_idx += 1

    append_jsonl(log_path, {"event_type": "roll_end", "stopped": stop_state.stop})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
