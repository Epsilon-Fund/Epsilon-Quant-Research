"""Prepare a targeted Block A0c capture config from live Polymarket data.

A0c is a validation capture for the A1 universe-selection criteria. It uses
current Gamma metadata, CLOB book snapshots, and recent public trade prints to
avoid padding the run with stale or quote-only markets.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from scripts.dali_live_clob_capture import parse_token_ids


ROOT = Path(__file__).resolve().parents[1]
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"
DEFAULT_OUT = ROOT / "configs" / "block_a0c_capture.generated.yaml"


@dataclass(frozen=True)
class CandidateSpec:
    slug: str
    family: str
    category_hint: str
    rationale: str
    pick_n: int = 1
    force_include: bool = False
    target_titles: tuple[str, ...] = ()


CANDIDATES = [
    # Crypto daily baselines.
    CandidateSpec(
        "bitcoin-up-or-down-on-may-29-2026",
        "daily_crypto_up_down",
        "Crypto",
        "crypto_daily: BTC daily, current high trade-rate up/down baseline",
        force_include=True,
    ),
    CandidateSpec(
        "ethereum-up-or-down-on-may-29-2026",
        "daily_crypto_up_down",
        "Crypto",
        "crypto_daily: ETH daily, paired high trade-rate up/down baseline",
        force_include=True,
    ),
    CandidateSpec(
        "solana-up-or-down-on-may-29-2026",
        "daily_crypto_up_down",
        "Crypto",
        "crypto_daily: SOL daily, included for cross-asset crypto comparison",
        force_include=True,
    ),
    # Explicit 4h windows. These are force-included even if future windows have
    # low launch-time prints because the whole point is to span window rollovers.
    CandidateSpec(
        "btc-updown-4h-1780041600",
        "crypto_4h_up_down",
        "Crypto",
        "crypto_4h: BTC current 4h window, tight spread/depth and live trade-rate",
        force_include=True,
    ),
    CandidateSpec(
        "btc-updown-4h-1780056000",
        "crypto_4h_up_down",
        "Crypto",
        "crypto_4h: BTC next 4h window, explicit rollover coverage",
        force_include=True,
    ),
    CandidateSpec(
        "eth-updown-4h-1780041600",
        "crypto_4h_up_down",
        "Crypto",
        "crypto_4h: ETH current 4h window, paired crypto rollover coverage",
        force_include=True,
    ),
    CandidateSpec(
        "eth-updown-4h-1780056000",
        "crypto_4h_up_down",
        "Crypto",
        "crypto_4h: ETH next 4h window, explicit rollover coverage",
        force_include=True,
    ),
    CandidateSpec(
        "sol-updown-4h-1780041600",
        "crypto_4h_up_down",
        "Crypto",
        "crypto_4h: SOL current 4h window, third crypto asset for comparison",
        force_include=True,
    ),
    CandidateSpec(
        "sol-updown-4h-1780056000",
        "crypto_4h_up_down",
        "Crypto",
        "crypto_4h: SOL next 4h window, explicit rollover coverage",
        force_include=True,
    ),
    # Sports/event-clock flow.
    CandidateSpec(
        "nhl-mon-car-2026-05-29",
        "sports_game_lines",
        "Sports",
        "sports_in_game: NHL game scheduled inside the capture window; A0b showed sports trade density",
        pick_n=1,
    ),
    CandidateSpec(
        "ucl-psg-ars-2026-05-30",
        "sports_game_lines",
        "Sports",
        "sports_event_clock: Champions League match market near capture window, high volume/depth",
        pick_n=1,
    ),
    CandidateSpec(
        "nba-sas-okc-2026-05-30",
        "sports_game_lines",
        "Sports",
        "sports_event_clock: NBA game market, included as A0b sports follow-up",
        pick_n=1,
    ),
    # Explicit neg-risk outrights, one liquid leg each.
    CandidateSpec(
        "world-cup-winner",
        "sports_neg_risk_outright",
        "Sports",
        "neg_risk: World Cup outright, top liquid leg selected",
        pick_n=1,
    ),
    CandidateSpec(
        "2026-nba-champion",
        "sports_neg_risk_outright",
        "Sports",
        "neg_risk: NBA champion outright, top liquid leg selected",
        pick_n=1,
    ),
    CandidateSpec(
        "presidential-election-winner-2028",
        "politics_neg_risk_outright",
        "Politics",
        "neg_risk: election outright, explicit non-sports negative-risk comparison",
        pick_n=1,
    ),
    # Geopolitics / fee-free.
    CandidateSpec(
        "us-x-iran-permanent-peace-deal-by",
        "geopolitics_policy",
        "Geopolitics",
        "geopolitics: high live trade-rate Iran policy family, fee-free if CLOB fee schedule is disabled",
        pick_n=1,
    ),
    CandidateSpec(
        "us-iran-nuclear-deal-by-june-30",
        "geopolitics_policy",
        "Geopolitics",
        "geopolitics: clean US-Iran nuclear deal binary, high live trade-rate and mid-band price",
        pick_n=1,
    ),
    CandidateSpec(
        "strait-of-hormuz-traffic-returns-to-normal-by-july-31",
        "geopolitics_policy",
        "Geopolitics",
        "geopolitics: A0b-proven Hormuz-style clean binary with spread/depth",
        force_include=True,
    ),
]


def parse_list(raw: Any) -> list[Any]:
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []
    return raw if isinstance(raw, list) else []


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fetch_json(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> Any:
    r = client.get(url, params=params)
    r.raise_for_status()
    return r.json()


def fee_peak_rate(category: str) -> float:
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


def book_stats(client: httpx.Client, token_id: str) -> dict[str, float | int | None]:
    try:
        raw = fetch_json(client, f"{CLOB_BASE}/book", {"token_id": token_id})
    except Exception as exc:
        return {"book_error": repr(exc)}
    bids = raw.get("bids") or []
    asks = raw.get("asks") or []
    bid_prices = [(as_float(x.get("price")), as_float(x.get("size"))) for x in bids]
    ask_prices = [(as_float(x.get("price")), as_float(x.get("size"))) for x in asks]
    bid_prices = [(p, s) for p, s in bid_prices if p is not None and s is not None]
    ask_prices = [(p, s) for p, s in ask_prices if p is not None and s is not None]
    best_bid = max((p for p, _ in bid_prices), default=None)
    best_ask = min((p for p, _ in ask_prices), default=None)
    bid_size = sum(s for p, s in bid_prices if best_bid is not None and p == best_bid)
    ask_size = sum(s for p, s in ask_prices if best_ask is not None and p == best_ask)
    mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
    spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
    touch_notional = (
        best_bid * bid_size + best_ask * ask_size
        if best_bid is not None and best_ask is not None
        else None
    )
    spread_bps = spread / mid * 10_000 if mid and spread is not None else None
    return {
        "book_bids": len(bid_prices),
        "book_asks": len(ask_prices),
        "book_best_bid": best_bid,
        "book_best_ask": best_ask,
        "book_mid": mid,
        "book_spread": spread,
        "book_spread_bps": spread_bps,
        "book_top_bid_size": bid_size,
        "book_top_ask_size": ask_size,
        "book_touch_notional": touch_notional,
    }


def trade_stats(client: httpx.Client, condition_id: str) -> dict[str, Any]:
    now = int(time.time())
    try:
        trades = fetch_json(
            client,
            f"{DATA_API_BASE}/trades",
            {"market": condition_id, "limit": 1000, "takerOnly": "true"},
        )
    except Exception as exc:
        return {"trade_error": repr(exc)}
    if not isinstance(trades, list):
        return {"trade_error": "unexpected trades response"}
    n_24h = n_48h = 0
    usd_24h = usd_48h = 0.0
    latest_ts = None
    sides: dict[str, int] = {}
    for trade in trades:
        ts = trade.get("timestamp")
        try:
            ts_int = int(ts)
        except (TypeError, ValueError):
            continue
        age = now - ts_int
        price = as_float(trade.get("price")) or 0.0
        size = as_float(trade.get("size")) or 0.0
        usd = price * size
        side = str(trade.get("side") or "")
        sides[side] = sides.get(side, 0) + 1
        latest_ts = max(latest_ts or ts_int, ts_int)
        if age <= 24 * 3600:
            n_24h += 1
            usd_24h += usd
        if age <= 48 * 3600:
            n_48h += 1
            usd_48h += usd
    return {
        "trades_24h": n_24h,
        "trades_48h": n_48h,
        "usd_24h": round(usd_24h, 2),
        "usd_48h": round(usd_48h, 2),
        "last_trade_ts": latest_ts,
        "last_trade_minutes_ago": round((now - latest_ts) / 60, 1) if latest_ts else None,
        "trade_sides": sides,
        "trade_rate_source": "data-api/trades?takerOnly=true",
    }


def market_title(market: dict[str, Any], event: dict[str, Any]) -> str:
    return str(
        market.get("groupItemTitle")
        or market.get("question")
        or event.get("title")
        or market.get("slug")
        or ""
    )


def market_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    trade_count = float(row.get("trades_24h") or 0)
    touch = float(row.get("book_touch_notional") or 0)
    spread_bps = float(row.get("book_spread_bps") or 99_999)
    mid = float(row.get("book_mid") or 0)
    spread_penalty = max(spread_bps - 200, 0) * 0.5
    boundary_penalty = 200 if mid < 0.05 or mid > 0.95 else 0
    return (trade_count * 10 + min(touch, 20_000) / 50 - spread_penalty - boundary_penalty, trade_count, touch, -spread_bps)


def event_markets(client: httpx.Client, spec: CandidateSpec) -> list[dict[str, Any]]:
    event = fetch_json(client, f"{GAMMA_BASE}/events/slug/{spec.slug}")
    if not isinstance(event, dict) or not event.get("markets"):
        return []
    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for market in event.get("markets") or []:
        if market.get("closed") or not market.get("active", True):
            continue
        end_raw = market.get("endDate") or event.get("endDate")
        if end_raw:
            try:
                end_dt = datetime.fromisoformat(str(end_raw).replace("Z", "+00:00"))
            except ValueError:
                end_dt = None
            if end_dt is not None and end_dt < now:
                continue
        title = market_title(market, event)
        if spec.target_titles and not any(t.lower() in title.lower() for t in spec.target_titles):
            continue
        token_ids = parse_token_ids(str(market.get("clobTokenIds") or ""))
        if not token_ids:
            token_ids = [str(x) for x in parse_list(market.get("clobTokenIds"))]
        if len(token_ids) < 2:
            continue
        condition_id = str(market.get("conditionId") or "")
        stats = {
            **book_stats(client, token_ids[0]),
            **trade_stats(client, condition_id),
        }
        best_bid = stats.get("book_best_bid")
        best_ask = stats.get("book_best_ask")
        if best_bid is None:
            best_bid = as_float(market.get("bestBid"))
        if best_ask is None:
            best_ask = as_float(market.get("bestAsk"))
        mid = (
            (float(best_bid) + float(best_ask)) / 2
            if best_bid is not None and best_ask is not None
            else as_float(market.get("lastTradePrice"))
        )
        spread = (
            float(best_ask) - float(best_bid)
            if best_bid is not None and best_ask is not None
            else None
        )
        fee_info: dict[str, Any] = {}
        if condition_id:
            try:
                fee_info = fetch_json(client, f"{CLOB_BASE}/clob-markets/{condition_id}")
            except Exception as exc:
                fee_info = {"fetch_error": repr(exc)}
        fee_details = fee_info.get("fd") if isinstance(fee_info, dict) else {}
        fee_details = fee_details or {}
        category = spec.category_hint
        fee_rate = fee_details.get("r")
        if fee_rate == 0:
            category = "Geopolitics"
        elif fee_rate == 0.03:
            category = "Sports"
        elif fee_rate == 0.07:
            category = "Crypto"
        row = {
            "id": str(market.get("id") or ""),
            "condition_id": condition_id,
            "question": str(market.get("question") or event.get("title") or ""),
            "group_item_title": title,
            "event_slug": str(event.get("slug") or spec.slug),
            "slug": str(market.get("slug") or spec.slug),
            "family": spec.family,
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
                "peak_effective_rate_estimate": fee_peak_rate(category),
                "clob_market_info": fee_info,
            },
            "selection_rationale": spec.rationale,
            "selection_metrics": stats,
            "selection_flags": {
                "force_include": spec.force_include,
                "trades_24h_ge_50": (stats.get("trades_24h") or 0) >= 50,
                "spread_bps_le_200": (
                    stats.get("book_spread_bps") is not None
                    and float(stats["book_spread_bps"]) <= 200
                ),
                "touch_notional_ge_500": (
                    stats.get("book_touch_notional") is not None
                    and float(stats["book_touch_notional"]) >= 500
                ),
            },
        }
        rows.append(row)
    rows.sort(key=market_score, reverse=True)
    return rows


def dedupe_markets(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for market in markets:
        key = str(market.get("id") or market.get("condition_id"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(market)
    return out


def build_config(run_id: str, duration_hours: float, max_markets: int) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    with httpx.Client(timeout=20) as client:
        for spec in CANDIDATES:
            rows = event_markets(client, spec)
            if not rows:
                print(f"skip {spec.slug}: no active subscribable markets")
                continue
            eligible = []
            for row in rows:
                metrics = row.get("selection_metrics") or {}
                trades_24h = float(metrics.get("trades_24h") or 0)
                spread_bps = float(metrics.get("book_spread_bps") or 99_999)
                touch = float(metrics.get("book_touch_notional") or 0)
                mid = float(metrics.get("book_mid") or row.get("mid") or 0)
                if spec.force_include or (
                    trades_24h >= 50
                    and spread_bps <= 400
                    and touch >= 500
                    and 0.05 <= mid <= 0.95
                ):
                    eligible.append(row)
            if not eligible:
                print(f"skip {spec.slug}: no rows passed live trade/spread/depth filter")
                continue
            chosen = eligible[: spec.pick_n]
            selected.extend(chosen)
    selected = dedupe_markets(selected)[:max_markets]
    return {
        "run": {
            "run_id": run_id,
            "label": "block_a0c",
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
            "out_dir": "data/live_clob/block_a0c",
        },
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "notes": [
            "A0c is additive and should not be merged with A0/A0b as one balanced panel.",
            "Selection uses live Gamma metadata, CLOB touch depth/spread, and Data API recent trades.",
            "Future crypto 4h windows are force-included to validate rollover capture even when launch-time trade count is low.",
            "No equity_index or single_stock families are included.",
        ],
        "markets": selected,
    }


def comment_yaml_markets(yaml_text: str, markets: list[dict[str, Any]]) -> str:
    lines = yaml_text.splitlines()
    out: list[str] = []
    market_idx = 0
    in_markets = False
    for line in lines:
        if line == "markets:":
            in_markets = True
            out.append(line)
            continue
        if in_markets and line.startswith("- id:") and market_idx < len(markets):
            market = markets[market_idx]
            metrics = market.get("selection_metrics") or {}
            flags = market.get("selection_flags") or {}
            title = market.get("group_item_title") or market.get("question") or market.get("slug")
            out.append(
                f"# {title} | {market.get('family')} | {market.get('selection_rationale')}"
            )
            out.append(
                "# gates: "
                f"trades24h={metrics.get('trades_24h')} "
                f"spread_bps={metrics.get('book_spread_bps')} "
                f"touch_notional={metrics.get('book_touch_notional')} "
                f"neg_risk={market.get('neg_risk')} "
                f"force={flags.get('force_include')}"
            )
            market_idx += 1
        out.append(line)
    return "\n".join(out) + "\n"


def validate_config(config: dict[str, Any]) -> None:
    market_count = len(config.get("markets") or [])
    token_count = sum(len(m.get("clob_token_ids") or []) for m in config.get("markets") or [])
    if market_count < 15:
        print(f"warning: only {market_count} markets selected; below 15 target")
    if token_count == 0:
        raise SystemExit("no token ids selected")
    families: dict[str, int] = {}
    neg = 0
    for market in config.get("markets") or []:
        families[str(market.get("family"))] = families.get(str(market.get("family")), 0) + 1
        neg += int(bool(market.get("neg_risk")))
    print(f"markets: {market_count}; tokens: {token_count}; neg-risk: {neg}; families: {families}")
    for market in config.get("markets") or []:
        metrics = market.get("selection_metrics") or {}
        print(
            f"- {market.get('family')}: {market.get('group_item_title')} "
            f"trades24h={metrics.get('trades_24h')} "
            f"spread_bps={metrics.get('book_spread_bps')} "
            f"touch=${metrics.get('book_touch_notional')} "
            f"neg={market.get('neg_risk')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-id", default="block_a0c_targeted_20260529_morning")
    parser.add_argument("--duration-hours", type=float, default=24)
    parser.add_argument("--max-markets", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(args.run_id, args.duration_hours, args.max_markets)
    validate_config(config)
    raw = yaml.safe_dump(config, sort_keys=False, width=100)
    text = comment_yaml_markets(raw, config["markets"])
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
