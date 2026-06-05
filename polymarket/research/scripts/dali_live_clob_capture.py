"""Capture live Polymarket CLOB market-channel events for Dali.

This is read-only: no auth, no order placement. It subscribes to CLOB token
IDs and writes JSONL envelopes containing the raw market-channel message plus
local receive timestamps for replay, sign validation, latency checks, and OFI
feature construction.

Examples:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/dali_live_clob_capture.py \
        --family ai_product --max-markets 5 --duration-seconds 300

    PYTHONPATH=. uv run python scripts/dali_live_clob_capture.py \
        --token-id 123 --token-id 456 --duration-seconds 60
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "data" / "analysis" / "dali_gamma_current_future_candidate_markets.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "live_clob"
DEFAULT_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass(frozen=True)
class TokenMeta:
    token_id: str
    market_id: str = ""
    question: str = ""
    slug: str = ""
    family: str = ""
    outcome_index: int | None = None


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_token_ids(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = raw.strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [part.strip().strip("\"'") for part in text.strip("[]").split(",")]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def load_candidate_tokens(
    csv_path: Path,
    *,
    family: str | None,
    slug_contains: str | None,
    question_contains: str | None,
    max_markets: int,
    max_tokens: int,
) -> list[TokenMeta]:
    if not csv_path.exists():
        raise SystemExit(
            f"candidate CSV not found: {csv_path}. "
            "Pass --token-id directly or rerun the Dali market-universe screen."
        )

    tokens: list[TokenMeta] = []
    seen_markets = 0
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_family = (row.get("family") or "").strip()
            slug = (row.get("slug") or "").strip()
            question = (row.get("question") or "").strip()
            if family and row_family != family:
                continue
            if slug_contains and slug_contains.lower() not in slug.lower():
                continue
            if question_contains and question_contains.lower() not in question.lower():
                continue

            ids = parse_token_ids(row.get("clobTokenIds") or row.get("clob_token_ids"))
            if not ids:
                continue
            seen_markets += 1
            for outcome_index, token_id in enumerate(ids):
                tokens.append(
                    TokenMeta(
                        token_id=token_id,
                        market_id=str(row.get("id") or row.get("market_id") or ""),
                        question=question,
                        slug=slug,
                        family=row_family,
                        outcome_index=outcome_index,
                    )
                )
                if len(tokens) >= max_tokens:
                    return tokens
            if seen_markets >= max_markets:
                return tokens
    return tokens


def build_subscription_tokens(args: argparse.Namespace) -> list[TokenMeta]:
    direct = [TokenMeta(token_id=t) for t in args.token_id]
    if direct:
        return direct[: args.max_tokens]
    return load_candidate_tokens(
        args.candidates_csv,
        family=args.family,
        slug_contains=args.slug_contains,
        question_contains=args.question_contains,
        max_markets=args.max_markets,
        max_tokens=args.max_tokens,
    )


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


def envelope(raw: str | bytes, metadata: dict[str, TokenMeta]) -> list[dict[str, Any]]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    received_at = utc_now()
    received_monotonic_ns = time.monotonic_ns()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw

    messages = parsed if isinstance(parsed, list) else [parsed]
    out: list[dict[str, Any]] = []
    for message in messages:
        asset_ids = asset_ids_of(message)
        out.append(
            {
                "received_at": received_at,
                "received_monotonic_ns": received_monotonic_ns,
                "event_type": event_type_of(message),
                "asset_ids": asset_ids,
                "assets": [
                    asdict(metadata[token_id])
                    for token_id in asset_ids
                    if token_id in metadata
                ],
                "message": message,
            }
        )
    return out


def default_output_path(out_dir: Path, label: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)[:80]
    return out_dir / f"dali_clob_{safe}_{stamp}.jsonl"


def write_manifest(out_path: Path, args: argparse.Namespace, tokens: list[TokenMeta]) -> None:
    manifest = {
        "created_at": utc_now(),
        "ws_url": args.ws_url,
        "duration_seconds": args.duration_seconds,
        "custom_feature_enabled": not args.no_custom_feature,
        "token_count": len(tokens),
        "tokens": [asdict(token) for token in tokens],
    }
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def capture(args: argparse.Namespace, tokens: list[TokenMeta], out_path: Path) -> Counter[str]:
    try:
        import websocket
        from websocket import WebSocketTimeoutException
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing websocket-client. Run from polymarket/research with `uv run`, "
            "or add websocket-client to the active environment."
        ) from exc

    token_ids = [token.token_id for token in tokens]
    metadata = {token.token_id: token for token in tokens}
    subscription = {
        "assets_ids": token_ids,
        "type": "market",
        "custom_feature_enabled": not args.no_custom_feature,
    }
    counts: Counter[str] = Counter()
    deadline = time.monotonic() + args.duration_seconds

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(out_path, args, tokens)
    print(f"connecting to {args.ws_url}")
    print(f"subscribing to {len(token_ids)} token ids")
    print(f"writing {out_path.relative_to(ROOT)}")

    ws = websocket.create_connection(args.ws_url, timeout=5)
    try:
        ws.send(json.dumps(subscription))
        with out_path.open("a") as fh:
            while time.monotonic() < deadline:
                try:
                    raw = ws.recv()
                except WebSocketTimeoutException:
                    continue
                for rec in envelope(raw, metadata):
                    counts[rec["event_type"]] += 1
                    fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
                total = sum(counts.values())
                if total and total % args.print_every == 0:
                    print(f"{utc_now()} captured {total} events {dict(counts)}")
    finally:
        ws.close()
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--token-id", action="append", default=[])
    parser.add_argument("--family")
    parser.add_argument("--slug-contains")
    parser.add_argument("--question-contains")
    parser.add_argument("--max-markets", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--no-custom-feature", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokens = build_subscription_tokens(args)
    if not tokens:
        raise SystemExit("no token ids selected")

    label = args.family or args.slug_contains or args.question_contains or "manual"
    out_path = args.out or default_output_path(args.out_dir, label)
    counts = capture(args, tokens, out_path)
    print(f"done: {sum(counts.values())} events {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
