#!/usr/bin/env python3
"""Cooperative edit locks for shared Markdown brain files.

This guard is intentionally simple: agents acquire a short-lived lock before
editing canonical Markdown, then release it when done. Lock files are Markdown so
Relay can sync them between collaborators; Git ignores the volatile lock files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys


ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = ROOT / "brain" / "agents" / "locks"
VALID_AGENTS = {"codex", "cowork", "justin"}
AGENT_ALIASES = {
    "code": "codex",
    "claude": "cowork",
    "claude-code": "cowork",
    "coworker": "cowork",
}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def parse_iso(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def rel_path(raw: str) -> str:
    path = (ROOT / raw).resolve(strict=False)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        raise SystemExit(f"Path is outside repo: {raw}")


def lock_path_for(rel: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", rel).strip("_")
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    if len(slug) > 90:
        slug = slug[:90].rstrip("._-")
    return LOCK_DIR / f"{digest}_{slug}.lock.md"


def file_sha(rel: str) -> str:
    path = ROOT / rel
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frontmatter_value(value: str) -> object:
    value = value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def read_lock(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    data: dict[str, object] = {}
    for line in text[4:end].splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = frontmatter_value(value)
    return data


def write_lock(path: Path, data: dict[str, object]) -> None:
    lines = ["---"]
    for key, value in data.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=True)}")
    lines.extend(
        [
            "---",
            "",
            f"# Edit lock: {data['path']}",
            "",
            "This file is a temporary cooperative lock for agents editing shared Markdown.",
            "Release it with `python3 tools/brain_edit_guard.py release --agent <agent> --path <path>`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def expires_at(minutes: int) -> str:
    return (now_utc() + dt.timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")


def is_expired(lock: dict[str, object]) -> bool:
    expiry = parse_iso(str(lock.get("expires_at", "")))
    return bool(expiry and expiry <= now_utc())


def lane_owner(rel: str) -> str | None:
    parts = rel.split("/")
    if len(parts) >= 3 and parts[:2] == ["brain", "agents"]:
        if parts[2] in {"codex", "cowork"}:
            return parts[2]
    return None


def is_generated_or_lock(rel: str) -> bool:
    return rel.startswith("brain/generated/") or rel.startswith("brain/agents/locks/")


def normalize_agent(agent: str) -> str:
    normalized = AGENT_ALIASES.get(agent.strip().lower(), agent.strip().lower())
    if normalized not in VALID_AGENTS:
        names = sorted(VALID_AGENTS | set(AGENT_ALIASES))
        raise SystemExit(f"Unknown agent '{agent}'. Use one of: {', '.join(names)}.")
    return normalized


def validate_target(rel: str, agent: str, allow_cross_lane: bool = False) -> None:
    if is_generated_or_lock(rel):
        raise SystemExit(f"Do not edit generated/lock files directly: {rel}")
    owner = lane_owner(rel)
    if owner and owner != agent and not allow_cross_lane:
        raise SystemExit(
            f"STOP: {rel} belongs to the {owner} lane, but agent is {agent}. "
            "Use your own scratch lane, or rerun with --allow-cross-lane only when Justin explicitly asked for this cross-lane edit."
        )


def acquire(args: argparse.Namespace) -> int:
    args.agent = normalize_agent(args.agent)
    rel = rel_path(args.path)
    validate_target(rel, args.agent, args.allow_cross_lane)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path_for(rel)
    data = {
        "path": rel,
        "agent": args.agent,
        "intent": args.intent,
        "started_at": now_utc().isoformat().replace("+00:00", "Z"),
        "expires_at": expires_at(args.ttl_minutes),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "pid": os.getpid(),
        "sha256_at_acquire": file_sha(rel),
    }

    if lock_file.exists():
        existing = read_lock(lock_file)
        existing_agent = str(existing.get("agent", "unknown"))
        expired = is_expired(existing)
        if existing_agent != args.agent and not expired:
            print(
                "STOP: active edit lock exists\n"
                f"  path   : {existing.get('path', rel)}\n"
                f"  agent  : {existing_agent}\n"
                f"  intent : {existing.get('intent', '')}\n"
                f"  expires: {existing.get('expires_at', '')}\n"
                "Ask Justin which agent owns this edit, wait, or release with --force if this is stale.",
                file=sys.stderr,
            )
            return 2
        write_lock(lock_file, data)
        print(("replaced stale" if expired else "refreshed") + f" lock: {rel}")
        return 0

    try:
        with lock_file.open("x", encoding="utf-8") as f:
            f.write("")
    except FileExistsError:
        print(f"STOP: lock appeared while acquiring: {lock_file}", file=sys.stderr)
        return 2
    write_lock(lock_file, data)
    print(f"acquired lock: {rel}")
    return 0


def release(args: argparse.Namespace) -> int:
    args.agent = normalize_agent(args.agent)
    rel = rel_path(args.path)
    lock_file = lock_path_for(rel)
    if not lock_file.exists():
        print(f"no lock found: {rel}")
        return 0
    lock = read_lock(lock_file)
    existing_agent = str(lock.get("agent", "unknown"))
    if existing_agent != args.agent and not args.force:
        print(
            f"STOP: {rel} is locked by {existing_agent}, not {args.agent}. "
            "Use --force only after confirming the lock is stale.",
            file=sys.stderr,
        )
        return 2
    lock_file.unlink()
    print(f"released lock: {rel}")
    return 0


def check(args: argparse.Namespace) -> int:
    args.agent = normalize_agent(args.agent)
    rel = rel_path(args.path)
    validate_target(rel, args.agent, args.allow_cross_lane)
    lock_file = lock_path_for(rel)
    if not lock_file.exists():
        if args.require_lock:
            print(f"STOP: no active lock for {rel}. Acquire one before editing.", file=sys.stderr)
            return 2
        print(f"ok: no active lock for {rel}")
        return 0
    lock = read_lock(lock_file)
    existing_agent = str(lock.get("agent", "unknown"))
    expired = is_expired(lock)
    if existing_agent != args.agent and not expired:
        print(
            f"STOP: {rel} is locked by {existing_agent} until {lock.get('expires_at', '')}.",
            file=sys.stderr,
        )
        return 2
    if expired:
        print(f"ok: only stale lock exists for {rel}")
        return 0
    print(f"ok: {rel} is locked by {args.agent}")
    return 0


def status(_: argparse.Namespace) -> int:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    locks = sorted(LOCK_DIR.glob("*.lock.md"))
    if not locks:
        print("no active lock files")
        return 0
    for lock_file in locks:
        lock = read_lock(lock_file)
        state = "expired" if is_expired(lock) else "active"
        print(
            f"{state:7} {lock.get('agent', 'unknown'):8} "
            f"{lock.get('path', lock_file.name)} "
            f"(expires {lock.get('expires_at', '')}; intent: {lock.get('intent', '')})"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cooperative edit lock guard for brain Markdown.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    acquire_p = sub.add_parser("acquire", help="Acquire or refresh a lock for a file.")
    acquire_p.add_argument("--agent", required=True)
    acquire_p.add_argument("--path", required=True)
    acquire_p.add_argument("--intent", required=True)
    acquire_p.add_argument("--ttl-minutes", type=int, default=120)
    acquire_p.add_argument("--allow-cross-lane", action="store_true")
    acquire_p.set_defaults(func=acquire)

    release_p = sub.add_parser("release", help="Release a lock for a file.")
    release_p.add_argument("--agent", required=True)
    release_p.add_argument("--path", required=True)
    release_p.add_argument("--force", action="store_true")
    release_p.set_defaults(func=release)

    check_p = sub.add_parser("check", help="Check lane ownership and active locks.")
    check_p.add_argument("--agent", required=True)
    check_p.add_argument("--path", required=True)
    check_p.add_argument("--require-lock", action="store_true")
    check_p.add_argument("--allow-cross-lane", action="store_true")
    check_p.set_defaults(func=check)

    status_p = sub.add_parser("status", help="List active lock files.")
    status_p.set_defaults(func=status)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
