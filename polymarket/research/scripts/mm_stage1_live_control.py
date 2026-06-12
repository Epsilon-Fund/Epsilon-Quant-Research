"""Control Stage-1 Polymarket live CLOB capture runs.

This is a thin operator wrapper around ``dali_block_a0_capture.py``. It creates
fresh live Gamma/CLOB configs, launches the three established Stage-1 capture
lanes in detached screen sessions, starts caffeinate, and can run a local
periodic health logger.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data" / "live_clob"
STATE_PATH = DATA_DIR / "mm_stage1_current.json"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Origin": "https://polymarket.com",
    "Referer": "https://polymarket.com/",
}

LANES = [
    {
        "name": "A first-mover",
        "label": "mm_stage1_first_mover",
        "screen": "mm_stage1_awake",
        "config_stem": "mm_stage1_first_mover_capture",
        "out_dir": "data/live_clob/mm_stage1_first_mover",
        "purpose": (
            "Stage-1 continuation live CLOB capture for first-mover/sports-style "
            "markets; observation only; no orders/auth"
        ),
        "plan": [
            ("sports_recurring", 14),
            ("other_residual", 2),
            ("geopolitics_diagnostic", 2),
        ],
    },
    {
        "name": "B broad diagnostics",
        "label": "mm_stage1_broad_live",
        "screen": "mm_stage1_broad",
        "config_stem": "mm_stage1_broad_live_capture",
        "out_dir": "data/live_clob/mm_stage1_broad_live",
        "purpose": (
            "Stage-1 continuation live CLOB capture for broad diagnostics; "
            "observation only; no orders/auth"
        ),
        "plan": [
            ("crypto_fast_5m", 12),
            ("crypto_fast_15m", 6),
            ("equity_index_open", 1),
            ("equity_index_close", 2),
            ("sports_recurring", 8),
            ("other_residual", 4),
            ("geopolitics_diagnostic", 3),
        ],
    },
    {
        "name": "C slow crypto + finance",
        "label": "mm_stage1_slow_crypto_finance",
        "screen": "mm_stage1_finance",
        "config_stem": "mm_stage1_slow_crypto_finance_capture",
        "out_dir": "data/live_clob/mm_stage1_slow_crypto_finance",
        "purpose": (
            "Stage-1 continuation live CLOB capture for slow crypto + finance "
            "diagnostics; observation only; no orders/auth"
        ),
        "plan": [
            ("crypto_4h", 8),
            ("crypto_daily", 8),
            ("equity_index_open", 1),
            ("equity_index_close", 2),
            ("crypto_fast_5m", 8),
            ("sports_recurring", 4),
        ],
    },
]

CONTROL_SCREENS = [lane["screen"] for lane in LANES] + [
    "mm_stage1_caffeinate",
    "mm_stage1_health",
]


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_stamp(dt: datetime | None = None) -> str:
    return (dt or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def boolish(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def token_ids(row: dict[str, Any]) -> list[str]:
    raw = row.get("clobTokenIds") or row.get("clob_token_ids") or row.get("clobTokenIDs")
    if isinstance(raw, str):
        try:
            values = json.loads(raw)
        except Exception:
            values = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        values = raw
    else:
        values = []
    return [str(value) for value in values if str(value)]


def slugish(row: dict[str, Any]) -> str:
    return str(row.get("slug") or row.get("eventSlug") or row.get("marketSlug") or "")


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=check)


def screen_ls() -> str:
    proc = run_cmd(["screen", "-ls"], check=False)
    return (proc.stdout or "") + (proc.stderr or "")


def active_screen_names() -> set[str]:
    names: set[str] = set()
    for line in screen_ls().splitlines():
        match = re.search(r"\d+\.([^\s]+)\s+\(", line)
        if match:
            names.add(match.group(1))
    return names


def ps_lines() -> list[str]:
    proc = run_cmd(["ps", "-Ao", "pid,ppid,stat,etime,command"], check=False)
    return proc.stdout.splitlines()


def recorder_pids() -> list[int]:
    pids: list[int] = []
    pattern = re.compile(
        r"^\s*(\d+).*/Python .*scripts/dali_block_a0_capture.py --config "
        r"configs/mm_stage1_.*\.yaml"
    )
    for line in ps_lines():
        match = pattern.search(line)
        if match:
            pids.append(int(match.group(1)))
    return pids


def fetch_json(client: httpx.Client, path: str, params: dict[str, Any] | None = None) -> Any:
    url = path if path.startswith("http") else GAMMA + path
    response = client.get(url, params=params or {})
    response.raise_for_status()
    return response.json()


def fetch_gamma_pages(client: httpx.Client, limit_pages: int = 12, page_limit: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    base_params = [
        {
            "active": "true",
            "closed": "false",
            "limit": page_limit,
            "offset": 0,
            "order": "volume24hr",
            "ascending": "false",
        },
        {
            "active": "true",
            "closed": "false",
            "limit": page_limit,
            "offset": 0,
            "order": "createdAt",
            "ascending": "false",
        },
        {
            "active": "true",
            "closed": "false",
            "limit": page_limit,
            "offset": 0,
            "order": "endDate",
            "ascending": "true",
        },
    ]
    for base in base_params:
        for page in range(limit_pages):
            params = dict(base, offset=page * page_limit)
            try:
                data = fetch_json(client, "/markets", params)
            except Exception as exc:
                print(f"warning: gamma page failed {params}: {exc}", file=sys.stderr)
                break
            batch = (data.get("data") or data.get("markets") or []) if isinstance(data, dict) else (data or [])
            if not batch:
                break
            for row in batch:
                if not isinstance(row, dict):
                    continue
                key = slugish(row) or str(row.get("id") or row.get("conditionId") or "")
                if key and key not in seen:
                    seen.add(key)
                    rows.append(row)
            if len(batch) < page_limit:
                break
    return rows


def search_gamma(client: httpx.Client, term: str, limit: int = 80) -> list[dict[str, Any]]:
    try:
        data = fetch_json(client, "/public-search", {"q": term, "limit": limit})
    except Exception:
        return []
    items: list[dict[str, Any]] = []
    if isinstance(data, dict):
        items.extend(item for item in data.get("markets") or [] if isinstance(item, dict))
        for event in data.get("events") or []:
            if isinstance(event, dict):
                items.extend(item for item in event.get("markets") or [] if isinstance(item, dict))
    elif isinstance(data, list):
        items = [item for item in data if isinstance(item, dict)]
    return items


def markets_from_event(client: httpx.Client, slug: str) -> list[dict[str, Any]]:
    try:
        data = fetch_json(client, f"/events/slug/{slug}")
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    return [market for market in data.get("markets") or [] if isinstance(market, dict)]


def active_market(row: dict[str, Any], now: datetime) -> bool:
    if boolish(row.get("closed"), False):
        return False
    if row.get("active") is not None and not boolish(row.get("active"), True):
        return False
    if row.get("enableOrderBook") is not None and not boolish(row.get("enableOrderBook"), True):
        return False
    if row.get("acceptingOrders") is not None and not boolish(row.get("acceptingOrders"), True):
        return False
    if len(token_ids(row)) < 2:
        return False
    end = parse_dt(row.get("endDate") or row.get("end_date"))
    return not (end and end < now - timedelta(minutes=2))


def exact_interval(slug: str, question: str, interval: str) -> bool:
    if re.search(rf"(^|-){re.escape(interval)}($|-)", slug):
        return True
    human = {"5m": "5 minute", "15m": "15 minute", "4h": "4 hour"}[interval]
    return human in question or human + "s" in question


def classify(row: dict[str, Any]) -> str | None:
    slug = slugish(row).lower()
    question = str(row.get("question") or "").lower()
    category = str(row.get("category") or row.get("categoryName") or "").lower()
    text = " ".join([slug, question, category])
    cryptoish = re.search(r"(^|-)(btc|eth|sol|xrp)(-|$)|bitcoin|ethereum|solana|xrp", text)
    if cryptoish:
        if exact_interval(slug, question, "15m"):
            return "crypto_fast_15m"
        if exact_interval(slug, question, "5m"):
            return "crypto_fast_5m"
        if exact_interval(slug, question, "4h"):
            return "crypto_4h"
        if "up or down" in question or "above" in question or "daily" in slug:
            return "crypto_daily"
    if any(x in text for x in ["s&p", "spx", "nasdaq", "qqq", "dow jones", "russell"]) and "up or down" in question:
        return "equity_index_open" if "open" in text else "equity_index_close"
    if any(x in category for x in ["sports"]) or any(
        x in text
        for x in [
            "fifa",
            "world cup",
            "nba",
            "wnba",
            "mlb",
            "nhl",
            "ufc",
            "soccer",
            "tennis",
            "champions league",
        ]
    ):
        return "sports_recurring"
    if any(x in text for x in ["israel", "iran", "ukraine", "russia", "china", "taiwan", "gaza", "war", "ceasefire", "nato"]):
        return "geopolitics_diagnostic"
    if any(x in category for x in ["crypto", "sports", "elections", "politics"]):
        return None
    return "other_residual"


def book_for_token(client: httpx.Client, token_id: str) -> dict[str, Any] | None:
    try:
        response = client.get(CLOB + "/book", params={"token_id": token_id})
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def price_level_stats(levels: Any, side: str) -> tuple[int, float | None, float | None]:
    parsed: list[tuple[float, float | None]] = []
    for level in levels or []:
        if not isinstance(level, dict):
            continue
        price = fnum(level.get("price"), math.nan)
        size = fnum(level.get("size"), math.nan)
        if math.isfinite(price):
            parsed.append((price, size if math.isfinite(size) else None))
    if not parsed:
        return 0, None, None
    best = max(parsed, key=lambda item: item[0]) if side == "bid" else min(parsed, key=lambda item: item[0])
    return len(parsed), best[0], best[1]


def enrich_books(client: httpx.Client, ids: list[str]) -> tuple[float | None, float | None, list[dict[str, Any]]]:
    tokens: list[dict[str, Any]] = []
    best_bid: float | None = None
    best_ask: float | None = None
    for token_id in ids[:2]:
        book = book_for_token(client, token_id)
        time.sleep(0.015)
        bids = (book or {}).get("bids") or (book or {}).get("buys") or []
        asks = (book or {}).get("asks") or (book or {}).get("sells") or []
        bid_count, bid, bid_size = price_level_stats(bids, "bid")
        ask_count, ask, ask_size = price_level_stats(asks, "ask")
        if bid is not None and (best_bid is None or bid > best_bid):
            best_bid = bid
        if ask is not None and (best_ask is None or ask < best_ask):
            best_ask = ask
        mid = ((bid + ask) / 2.0) if bid is not None and ask is not None else None
        spread = (ask - bid) if bid is not None and ask is not None else None
        spread_bps = (spread / mid * 10000.0) if mid and spread is not None else None
        tokens.append(
            {
                "token_id": token_id,
                "book_bids": bid_count,
                "book_asks": ask_count,
                "book_best_bid": bid,
                "book_bid_size": bid_size,
                "book_best_ask": ask,
                "book_ask_size": ask_size,
                "book_spread": spread,
                "book_spread_bps": spread_bps,
            }
        )
    return best_bid, best_ask, tokens


def market_record(client: httpx.Client, row: dict[str, Any], family: str, stamp: str) -> dict[str, Any] | None:
    ids = token_ids(row)
    if len(ids) < 2:
        return None
    book_bid, book_ask, books = enrich_books(client, ids)
    gamma_bid = fnum(row.get("bestBid"), math.nan)
    gamma_ask = fnum(row.get("bestAsk"), math.nan)
    best_bid = book_bid if book_bid is not None else (gamma_bid if math.isfinite(gamma_bid) else None)
    best_ask = book_ask if book_ask is not None else (gamma_ask if math.isfinite(gamma_ask) else None)
    mid = ((best_bid + best_ask) / 2.0) if best_bid is not None and best_ask is not None else None
    spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
    volume24h = fnum(row.get("volume24hr") or row.get("volume24h") or row.get("volume24Hour"), 0.0)
    liquidity = fnum(row.get("liquidity") or row.get("liquidityClob"), 0.0)
    return {
        "id": str(row.get("id") or ""),
        "condition_id": str(row.get("conditionId") or row.get("condition_id") or ""),
        "question": str(row.get("question") or row.get("title") or ""),
        "group_item_title": row.get("groupItemTitle"),
        "event_slug": str(row.get("eventSlug") or row.get("event_slug") or slugish(row)),
        "slug": slugish(row),
        "family": family,
        "category": str(row.get("category") or row.get("categoryName") or ""),
        "created_at": str(row.get("createdAt") or row.get("created_at") or ""),
        "end_date": str(row.get("endDate") or row.get("end_date") or ""),
        "volume24hr": volume24h,
        "volume": fnum(row.get("volume"), 0.0),
        "liquidity": liquidity,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
        "neg_risk": boolish(row.get("negRisk") or row.get("neg_risk"), False),
        "clob_token_ids": ids[:2],
        "fee": {"fees_enabled": boolish(row.get("feesEnabled"), True), "peak_effective_rate_estimate": 0.0125},
        "book_tokens": books,
        "selection_metrics": {"gamma_volume24hr": volume24h, "gamma_liquidity": liquidity},
        "selection_rationale": (
            "Stage-1 continuation live CLOB capture; observation only; no orders/auth. "
            f"Refreshed from current Gamma/CLOB on {stamp}; live measurement loop only."
        ),
    }


def build_universe(client: httpx.Client, now: datetime) -> dict[str, list[dict[str, Any]]]:
    raw = fetch_gamma_pages(client)
    search_terms = [
        "bitcoin up or down 5m",
        "ethereum up or down 5m",
        "bitcoin up or down 15m",
        "ethereum up or down 15m",
        "bitcoin up or down 4h",
        "ethereum up or down 4h",
        "solana up or down 4h",
        "xrp up or down 4h",
        "bitcoin above",
        "ethereum above",
        "spx up or down",
        "s&p 500 up or down",
        "nasdaq up or down",
        "world cup",
        "nba",
        "mlb",
        "israel iran",
        "ukraine russia",
        "gaza ceasefire",
        "AI",
        "fed rates",
    ]
    event_slugs = [
        "bitcoin-up-or-down",
        "ethereum-up-or-down",
        "solana-up-or-down",
        "xrp-up-or-down",
        "bitcoin-up-or-down-5m",
        "ethereum-up-or-down-5m",
        "bitcoin-up-or-down-15m",
        "ethereum-up-or-down-15m",
        "bitcoin-up-or-down-4h",
        "ethereum-up-or-down-4h",
        "solana-up-or-down-4h",
        "xrp-up-or-down-4h",
        "spx-open-daily-up-or-down",
        "spx-daily-up-or-down",
        "nasdaq-daily-up-or-down",
        "world-cup-winner",
    ]
    for term in search_terms:
        raw.extend(search_gamma(client, term))
    for slug in event_slugs:
        raw.extend(markets_from_event(client, slug))

    unique: dict[str, dict[str, Any]] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        key = slugish(row) or str(row.get("id") or row.get("conditionId") or "")
        if key and key not in unique:
            unique[key] = row

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique.values():
        if not active_market(row, now):
            continue
        family = classify(row)
        if family:
            buckets[family].append(row)
    for family in buckets:
        buckets[family].sort(
            key=lambda row: (
                fnum(row.get("volume24hr") or row.get("volume24h"), 0.0),
                fnum(row.get("liquidity"), 0.0),
                parse_dt(row.get("createdAt") or row.get("created_at")) or datetime(1970, 1, 1, tzinfo=UTC),
            ),
            reverse=True,
        )
    return buckets


def select_markets(client: httpx.Client, buckets: dict[str, list[dict[str, Any]]], plan: list[tuple[str, int]], stamp: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for family, target in plan:
        got = 0
        for row in buckets.get(family, []):
            if got >= target:
                break
            slug = slugish(row)
            if not slug or slug in seen:
                continue
            record = market_record(client, row, family, stamp)
            if record is None:
                continue
            seen.add(slug)
            out.append(record)
            got += 1
    return out


def write_configs(duration_hours: float) -> dict[str, Any]:
    now = utc_now()
    stamp = utc_stamp(now)
    client = httpx.Client(timeout=12, headers=HEADERS, follow_redirects=True)
    buckets = build_universe(client, now)
    state: dict[str, Any] = {
        "stamp": stamp,
        "prepared_at": iso(now),
        "lanes": [],
        "bucket_raw_counts": {family: len(rows) for family, rows in sorted(buckets.items())},
    }
    common_run = {
        "duration_hours": duration_hours,
        "rotate_minutes": 30,
        "print_every_events": 1000,
        "heartbeat_seconds": 60,
        "stale_warning_seconds": 300,
        "reconnect_backoff_seconds": [1, 2, 4, 8, 16, 30],
        "tolerate_gaps": True,
    }
    for lane in LANES:
        markets = select_markets(client, buckets, lane["plan"], stamp)
        config_path = CONFIG_DIR / f"{lane['config_stem']}.awake_{stamp}.yaml"
        config = {
            "prepared_at": iso(now),
            "purpose": f"{lane['purpose']}. Refreshed on {stamp}.",
            "run": {"run_id": f"{lane['label']}_awake_{stamp}", "label": lane["label"], **common_run},
            "capture": {
                "ws_url": "wss://ws-subscriptions-clob.polymarket.com/ws/market",
                "custom_feature_enabled": True,
                "out_dir": lane["out_dir"],
            },
            "selection": {
                "source": (
                    "fresh current Gamma/CLOB lean refresh; current same-day finance, sports, "
                    "crypto controls, and diagnostics as available"
                ),
                "lane_boundary": "Stage-1 measurement only; no orders/auth; politics deployment separate",
            },
            "markets": markets,
        }
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_dir = ROOT / lane["out_dir"] / config["run"]["run_id"]
        family_counts: dict[str, int] = defaultdict(int)
        for market in markets:
            family_counts[str(market.get("family"))] += 1
        state["lanes"].append(
            {
                "name": lane["name"],
                "label": lane["label"],
                "screen": lane["screen"],
                "run_id": config["run"]["run_id"],
                "config": str(config_path.relative_to(ROOT)),
                "run_dir": str(run_dir.relative_to(ROOT)),
                "out_dir": lane["out_dir"],
                "markets": len(markets),
                "tokens": sum(len(market.get("clob_token_ids") or []) for market in markets),
                "families": dict(family_counts),
                "sample_slugs": [market["slug"] for market in markets[:5]],
            }
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state


def launch_lane(lane_state: dict[str, Any]) -> None:
    log_dir = ROOT / str(lane_state["out_dir"]) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(lane_state["run_id"])
    log_path = log_dir / f"{run_id}.stdout.log"
    config = str(lane_state["config"])
    command = (
        f"cd {ROOT} && exec env PYTHONPATH=. uv run python "
        f"scripts/dali_block_a0_capture.py --config {config} > {log_path} 2>&1"
    )
    run_cmd(["screen", "-S", str(lane_state["screen"]), "-dm", "sh", "-lc", command])


def launch_health_loop(interval_hours: float, stamp: str) -> Path:
    health_dir = DATA_DIR / "mm_stage1_health"
    health_dir.mkdir(parents=True, exist_ok=True)
    log_path = health_dir / f"mm_stage1_health_{stamp}.jsonl"
    sleep_seconds = max(60, int(interval_hours * 3600))
    command = (
        f"cd {ROOT} && while true; do "
        f"uv run python scripts/mm_stage1_live_control.py status --json >> {log_path} 2>&1; "
        f"sleep {sleep_seconds}; "
        "done"
    )
    run_cmd(["screen", "-S", "mm_stage1_health", "-dm", "sh", "-lc", command])
    return log_path


def start(args: argparse.Namespace) -> int:
    existing = active_screen_names().intersection(CONTROL_SCREENS)
    if existing and not args.takeover:
        print(
            "stage-1 screens already exist: "
            + ", ".join(sorted(existing))
            + ". Use --takeover to stop them before starting fresh.",
            file=sys.stderr,
        )
        return 2
    if existing and args.takeover:
        stop(argparse.Namespace(grace_seconds=args.grace_seconds))

    state = write_configs(args.duration_hours)
    for lane_state in state["lanes"]:
        launch_lane(lane_state)
    run_cmd(["screen", "-S", "mm_stage1_caffeinate", "-dm", "/usr/bin/caffeinate", "-d", "-i"])
    health_log = launch_health_loop(args.health_interval_hours, state["stamp"])
    time.sleep(2)
    summary = {
        "status": "started",
        "stamp": state["stamp"],
        "health_interval_hours": args.health_interval_hours,
        "health_log": str(health_log),
        "lanes": state["lanes"],
        "screens": sorted(active_screen_names().intersection(CONTROL_SCREENS)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def stop(args: argparse.Namespace) -> int:
    pids = recorder_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if pids:
        time.sleep(float(args.grace_seconds))
    for pid in recorder_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    run_cmd(["killall", "caffeinate"], check=False)
    for screen in CONTROL_SCREENS:
        run_cmd(["screen", "-S", screen, "-X", "quit"], check=False)
    run_cmd(["screen", "-wipe"], check=False)
    print(json.dumps({"status": "stopped", "terminated_recorder_pids": pids}, sort_keys=True))
    return 0


def latest_run_dir(out_dir: str) -> Path | None:
    base = ROOT / out_dir
    if not base.exists():
        return None
    dirs = [path for path in base.iterdir() if path.is_dir() and path.name.startswith("mm_stage1_")]
    if not dirs:
        return None
    return max(dirs, key=lambda path: path.stat().st_mtime)


def read_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    lanes = []
    for lane in LANES:
        run_dir = latest_run_dir(lane["out_dir"])
        if run_dir is None:
            continue
        lanes.append(
            {
                "name": lane["name"],
                "label": lane["label"],
                "screen": lane["screen"],
                "run_id": run_dir.name,
                "config": None,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "out_dir": lane["out_dir"],
            }
        )
    return {"stamp": None, "lanes": lanes}


def load_gap_events(run_dir: Path) -> list[dict[str, Any]]:
    gap_path = run_dir / "capture_gaps.jsonl"
    if not gap_path.exists():
        return []
    events: list[dict[str, Any]] = []
    with gap_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def status(args: argparse.Namespace) -> int:
    screens = active_screen_names()
    ps_text = "\n".join(ps_lines())
    state = read_state()
    results: list[dict[str, Any]] = []
    for lane_state in state.get("lanes", []):
        run_dir = ROOT / str(lane_state["run_dir"])
        events = load_gap_events(run_dir)
        heartbeat = next((event for event in reversed(events) if event.get("event_type") == "heartbeat"), None)
        ended = next(
            (
                event
                for event in reversed(events)
                if event.get("event_type") in {"capture_end", "capture_stop", "ended"}
            ),
            None,
        )
        warnings = [
            event
            for event in events
            if event.get("event_type") in {"stale_warning", "disconnect_or_error"}
        ]
        shard_rel = heartbeat.get("current_shard") if heartbeat else None
        shard_path = ROOT / shard_rel if shard_rel else None
        size = shard_path.stat().st_size if shard_path and shard_path.exists() else 0
        screen = str(lane_state["screen"])
        config = str(lane_state.get("config") or "")
        python_alive = bool(config and config in ps_text)
        if not python_alive:
            python_alive = str(lane_state.get("run_id") or "") in ps_text
        results.append(
            {
                "run": lane_state["name"],
                "run_id": lane_state.get("run_id"),
                "screen": screen,
                "screen_alive": screen in screens,
                "python_alive": python_alive,
                "latest_heartbeat_ts": heartbeat.get("ts") if heartbeat else None,
                "seconds_since_last_event": heartbeat.get("seconds_since_last_event") if heartbeat else None,
                "total_counts": heartbeat.get("total_counts") if heartbeat else {},
                "current_shard": shard_rel,
                "current_shard_size_bytes": size,
                "warning_count_total": len(warnings),
                "latest_warning": warnings[-1] if warnings else None,
                "ended": ended is not None,
                "ended_event": ended,
            }
        )
    payload = {
        "checked_at": iso(),
        "read_only_observation": True,
        "not_deployability_verdict": True,
        "caffeinate_alive": "mm_stage1_caffeinate" in screens or "caffeinate -d -i" in ps_text,
        "health_loop_alive": "mm_stage1_health" in screens,
        "runs": results,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    start_p = sub.add_parser("start", help="Generate fresh configs and start capture screens.")
    start_p.add_argument("--takeover", action="store_true", help="Stop existing mm_stage1 screens first.")
    start_p.add_argument("--duration-hours", type=float, default=16.0)
    start_p.add_argument("--health-interval-hours", type=float, default=4.0)
    start_p.add_argument("--grace-seconds", type=float, default=8.0)
    start_p.set_defaults(func=start)

    stop_p = sub.add_parser("stop", help="Gracefully stop capture screens and kill caffeinate.")
    stop_p.add_argument("--grace-seconds", type=float, default=8.0)
    stop_p.set_defaults(func=stop)

    status_p = sub.add_parser("status", help="Report current capture health.")
    status_p.add_argument("--json", action="store_true", help="Emit compact JSON for health logs.")
    status_p.set_defaults(func=status)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
