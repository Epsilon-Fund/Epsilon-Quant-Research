"""Prepare a hand-picked Block A0b capture config from Polymarket event slugs.

This is for exploratory replacement/parallel captures where we want a small
set of currently active markets without touching the main A0 shortlist.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from scripts.dali_live_clob_capture import parse_token_ids


ROOT = Path(__file__).resolve().parents[1]
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DEFAULT_OUT = ROOT / "configs" / "block_a0b_capture.generated.yaml"


MARKET_REQUESTS = [
    {
        "slug": "strait-of-hormuz-traffic-returns-to-normal-by-end-of-june",
        "family": "geopolitics_policy",
        "category_hint": "Geopolitics",
        "selection_rationale": "High trade density, tight spread, clean binary geopolitics market.",
    },
    {
        "slug": "strait-of-hormuz-traffic-returns-to-normal-by-july-31",
        "family": "geopolitics_policy",
        "category_hint": "Geopolitics",
        "selection_rationale": "High trade density, tight spread, clean binary geopolitics market.",
    },
    {
        "slug": "bitcoin-up-or-down-on-may-28-2026",
        "family": "daily_crypto_up_down",
        "category_hint": "Crypto",
        "selection_rationale": "Active daily crypto up/down baseline with tight visible spread.",
    },
    {
        "slug": "ethereum-up-or-down-on-may-28-2026",
        "family": "daily_crypto_up_down",
        "category_hint": "Crypto",
        "selection_rationale": "Paired daily crypto up/down baseline; less active than BTC but still useful.",
    },
    {
        "slug": "btc-updown-4h-1779912000",
        "family": "crypto_4h_up_down",
        "category_hint": "Crypto",
        "selection_rationale": "BTC 4h up/down window 1 of 3 for the 12h A0b crypto lane.",
    },
    {
        "slug": "btc-updown-4h-1779926400",
        "family": "crypto_4h_up_down",
        "category_hint": "Crypto",
        "selection_rationale": "BTC 4h up/down window 2 of 3 for the 12h A0b crypto lane.",
    },
    {
        "slug": "btc-updown-4h-1779940800",
        "family": "crypto_4h_up_down",
        "category_hint": "Crypto",
        "selection_rationale": "BTC 4h up/down window 3 of 3 for the 12h A0b crypto lane.",
    },
    {
        "slug": "nba-okc-sas-2026-05-28",
        "market_id": "2327929",
        "family": "sports_game_lines",
        "category_hint": "Sports",
        "selection_rationale": "Main Thunder vs. Spurs moneyline; very active same-day sports flow.",
    },
    {
        "slug": "uefa-champions-league-winner",
        "market_id": "566136",
        "family": "sports_neg_risk_outright",
        "category_hint": "Sports",
        "selection_rationale": "Most active UCL winner negative-risk leg, PSG.",
    },
]


def peak_fee_rate(category: str) -> float:
    return {
        "Geopolitics": 0.0,
        "Sports": 0.0075,
        "Crypto": 0.0175,
        "Finance": 0.0100,
        "Politics": 0.0100,
        "Tech": 0.0100,
        "Culture": 0.0125,
        "Weather": 0.0125,
        "Economics": 0.0125,
        "Other / General": 0.0125,
    }.get(category, 0.0125)


def parse_list(raw: Any) -> list[Any]:
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []
    return raw if isinstance(raw, list) else []


def fetch_json(client: httpx.Client, url: str) -> dict[str, Any]:
    r = client.get(url)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, dict) else {}


def select_market(event: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    markets = event.get("markets") or []
    if not markets:
        raise ValueError(f"{request['slug']} has no markets")
    market_id = request.get("market_id")
    if market_id:
        for market in markets:
            if str(market.get("id")) == str(market_id):
                return market
        raise ValueError(f"{request['slug']} missing market id {market_id}")
    return markets[0]


def enrich_market(client: httpx.Client, request: dict[str, Any]) -> dict[str, Any]:
    event = fetch_json(client, f"{GAMMA_BASE}/events/slug/{request['slug']}")
    market = select_market(event, request)
    condition_id = str(market.get("conditionId") or market.get("condition_id") or "")
    fee_info: dict[str, Any] = {}
    if condition_id:
        try:
            fee_info = fetch_json(client, f"{CLOB_BASE}/clob-markets/{condition_id}")
        except Exception as exc:
            fee_info = {"fetch_error": repr(exc)}
    fee_details = fee_info.get("fd") or {}
    fee_rate = fee_details.get("r")
    category = str(event.get("category") or market.get("category") or request["category_hint"])
    if fee_rate == 0:
        category = "Geopolitics"
    elif fee_rate == 0.03:
        category = "Sports"
    elif fee_rate == 0.07:
        category = "Crypto"

    token_ids = parse_token_ids(str(market.get("clobTokenIds") or ""))
    if not token_ids:
        token_ids = [str(x) for x in parse_list(market.get("clobTokenIds"))]
    best_bid = market.get("bestBid")
    best_ask = market.get("bestAsk")
    try:
        mid = (float(best_bid) + float(best_ask)) / 2
        spread = float(best_ask) - float(best_bid)
    except (TypeError, ValueError):
        mid = None
        spread = None

    title = str(market.get("groupItemTitle") or market.get("question") or event.get("title") or "")
    return {
        "id": str(market.get("id") or ""),
        "condition_id": condition_id,
        "question": str(market.get("question") or event.get("title") or ""),
        "group_item_title": title,
        "event_slug": str(event.get("slug") or request["slug"]),
        "slug": str(market.get("slug") or request["slug"]),
        "family": str(request["family"]),
        "category": category,
        "end_date": market.get("endDate") or event.get("endDate"),
        "volume": float(market.get("volume") or event.get("volume") or 0),
        "volume24hr": float(market.get("volume24hr") or event.get("volume24hr") or 0),
        "liquidity": float(market.get("liquidity") or event.get("liquidity") or 0),
        "best_bid": float(best_bid) if best_bid is not None else None,
        "best_ask": float(best_ask) if best_ask is not None else None,
        "mid": mid,
        "spread": spread,
        "neg_risk": bool(market.get("negRisk") or event.get("enableNegRisk")),
        "clob_token_ids": token_ids,
        "fee": {
            "fees_enabled": bool(market.get("feesEnabled", fee_details.get("to", False))),
            "fee_rate": fee_details.get("r"),
            "fee_exponent": fee_details.get("e"),
            "taker_only": fee_details.get("to"),
            "maker_base_fee_bps": market.get("makerBaseFee"),
            "taker_base_fee_bps": market.get("takerBaseFee"),
            "fee_schedule": market.get("feeSchedule"),
            "peak_effective_rate_estimate": peak_fee_rate(category),
            "clob_market_info": fee_info,
        },
        "selection_rationale": request["selection_rationale"],
    }


def build_config(run_id: str, duration_hours: float) -> dict[str, Any]:
    with httpx.Client(timeout=20) as client:
        markets = [enrich_market(client, request) for request in MARKET_REQUESTS]
    return {
        "run": {
            "run_id": run_id,
            "label": "block_a0b",
            "duration_hours": duration_hours,
            "rotate_minutes": 60,
            "print_every_events": 500,
            "heartbeat_seconds": 60,
            "stale_warning_seconds": 900,
            "reconnect_backoff_seconds": [1, 2, 4, 8, 16, 30],
            "tolerate_gaps": True,
        },
        "capture": {
            "ws_url": "wss://ws-subscriptions-clob.polymarket.com/ws/market",
            "custom_feature_enabled": True,
            "out_dir": "data/live_clob/block_a0b",
        },
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "notes": [
            "A0b is a parallel replacement capture; do not merge it with A0 as one balanced panel.",
            "BTC 4h includes three explicit listed windows: current plus the next two 4h windows.",
            "UCL winner is a negative-risk event; only the PSG binary leg is included.",
        ],
        "markets": markets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-id", default="block_a0b_replacements_20260527")
    parser.add_argument("--duration-hours", type=float, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(args.run_id, args.duration_hours)
    args.out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")
    for market in config["markets"]:
        print(
            f"- {market['family']}: {market['group_item_title']} "
            f"mid={market['mid']} spread={market['spread']} neg_risk={market['neg_risk']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
