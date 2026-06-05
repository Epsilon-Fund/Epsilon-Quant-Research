"""Forward capture for PM crypto windows aligned to Deribit 08:00 UTC expiry.

This is live data-gathering infrastructure, not a backtest and not an
execution bot. It stores executable state from Polymarket, Deribit, and
reference venues at the same wall-clock snapshots so the OD-RV branch can be
validated later with bid/ask-aware costs and settlement-source diagnostics.

Example:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/od_rv_deribit_aligned_capture.py --once

    PYTHONPATH=. uv run python scripts/od_rv_deribit_aligned_capture.py \
        --expiry-date 2026-06-03 --families 4h,hourly --assets BTC,ETH
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "live_clob" / "od_rv_deribit_aligned"

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"
DERIBIT_BASE = "https://www.deribit.com/api/v2"
BINANCE_BASE = "https://api.binance.com"
PYTH_BASE = "https://hermes.pyth.network"

NY = ZoneInfo("America/New_York")

ASSETS = {
    "BTC": {
        "pm_hourly_prefix": "bitcoin",
        "pm_4h_prefix": "btc",
        "binance_symbol": "BTCUSDT",
        "deribit_currency": "BTC",
        "deribit_index": "btc_usd",
        "pyth_price_id": "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
        "chainlink_stream": "https://data.chain.link/streams/btc-usd",
    },
    "ETH": {
        "pm_hourly_prefix": "ethereum",
        "pm_4h_prefix": "eth",
        "binance_symbol": "ETHUSDT",
        "deribit_currency": "ETH",
        "deribit_index": "eth_usd",
        "pyth_price_id": "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
        "chainlink_stream": "https://data.chain.link/streams/eth-usd",
    },
}


@dataclass(frozen=True)
class PmSpec:
    asset: str
    family: str
    slug: str
    window_start: datetime
    window_end: datetime
    expected_resolution_source: str


@dataclass(frozen=True)
class DeribitExpiry:
    asset: str
    expiry: datetime
    settlement_period: str
    instrument_count: int


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_dt(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        out = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return out.astimezone(UTC)


def parse_json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fetch_json(client: httpx.Client, url: str, params: Any | None = None) -> Any:
    resp = client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def month_slug(dt: datetime) -> str:
    return dt.strftime("%B").lower()


def hour_slug(dt: datetime) -> str:
    hour = dt.hour
    suffix = "am" if hour < 12 else "pm"
    hour12 = hour % 12 or 12
    return f"{hour12}{suffix}"


def pm_specs_for_expiry(asset: str, expiry: datetime, families: list[str]) -> list[PmSpec]:
    meta = ASSETS[asset]
    specs: list[PmSpec] = []
    if "4h" in families:
        start = expiry - timedelta(hours=4)
        slug = f"{meta['pm_4h_prefix']}-updown-4h-{int(start.timestamp())}"
        specs.append(
            PmSpec(
                asset=asset,
                family="4h",
                slug=slug,
                window_start=start,
                window_end=expiry,
                expected_resolution_source=str(meta["chainlink_stream"]),
            )
        )
    if "hourly" in families:
        start = expiry - timedelta(hours=1)
        local = start.astimezone(NY)
        slug = (
            f"{meta['pm_hourly_prefix']}-up-or-down-"
            f"{month_slug(local)}-{local.day}-{local.year}-{hour_slug(local)}-et"
        )
        specs.append(
            PmSpec(
                asset=asset,
                family="hourly",
                slug=slug,
                window_start=start,
                window_end=expiry,
                expected_resolution_source=f"https://www.binance.com/en/trade/{asset}_USDT",
            )
        )
    return specs


def discover_deribit_expiries(client: httpx.Client, asset: str) -> list[DeribitExpiry]:
    ccy = str(ASSETS[asset]["deribit_currency"])
    data = fetch_json(
        client,
        f"{DERIBIT_BASE}/public/get_instruments",
        {"currency": ccy, "kind": "option", "expired": "false"},
    )
    instruments = data.get("result", []) if isinstance(data, dict) else []
    groups: dict[tuple[int, str], int] = {}
    for instrument in instruments:
        exp_ms = instrument.get("expiration_timestamp")
        if exp_ms is None:
            continue
        key = (int(exp_ms), str(instrument.get("settlement_period") or ""))
        groups[key] = groups.get(key, 0) + 1
    out = [
        DeribitExpiry(
            asset=asset,
            expiry=datetime.fromtimestamp(exp_ms / 1000, tz=UTC),
            settlement_period=period,
            instrument_count=count,
        )
        for (exp_ms, period), count in groups.items()
    ]
    return sorted(out, key=lambda row: row.expiry)


def choose_expiry(expiries: list[DeribitExpiry], expiry_date: str | None) -> DeribitExpiry:
    if not expiries:
        raise SystemExit("Deribit returned no active option expiries")
    if expiry_date:
        wanted = datetime.fromisoformat(expiry_date).date()
        matches = [row for row in expiries if row.expiry.date() == wanted]
        if not matches:
            raise SystemExit(f"no Deribit expiry found for {expiry_date}")
        return matches[0]
    now = utc_now()
    future = [row for row in expiries if row.expiry >= now]
    return future[0] if future else expiries[-1]


def discover_pm_market(client: httpx.Client, spec: PmSpec) -> dict[str, Any] | None:
    rows = fetch_json(client, f"{GAMMA_BASE}/markets", {"slug": spec.slug})
    if not isinstance(rows, list) or not rows:
        return None
    market = rows[0]
    outcomes = [str(x) for x in parse_json_list(market.get("outcomes"))]
    token_ids = [str(x) for x in parse_json_list(market.get("clobTokenIds"))]
    return {
        "asset": spec.asset,
        "family": spec.family,
        "slug": spec.slug,
        "question": market.get("question"),
        "condition_id": market.get("conditionId"),
        "market_id": market.get("id"),
        "active": market.get("active"),
        "closed": market.get("closed"),
        "accepting_orders": market.get("acceptingOrders"),
        "outcomes": outcomes,
        "clob_token_ids": token_ids,
        "window_start": iso(spec.window_start),
        "window_end": iso(spec.window_end),
        "event_start_time": market.get("eventStartTime")
        or ((market.get("events") or [{}])[0].get("startTime") if market.get("events") else None),
        "end_date": market.get("endDate"),
        "resolution_source": market.get("resolutionSource"),
        "expected_resolution_source": spec.expected_resolution_source,
        "description": market.get("description"),
        "fee_schedule": market.get("feeSchedule"),
        "best_bid": safe_float(market.get("bestBid")),
        "best_ask": safe_float(market.get("bestAsk")),
        "last_trade_price": safe_float(market.get("lastTradePrice")),
        "volume": safe_float(market.get("volume")),
        "liquidity": safe_float(market.get("liquidity")),
        "raw_gamma_market": market,
    }


def normalize_book_levels(raw_levels: Any, side: str, top_n: int) -> list[dict[str, float]]:
    levels: list[tuple[float, float]] = []
    for level in raw_levels or []:
        if isinstance(level, dict):
            price = safe_float(level.get("price"))
            size = safe_float(level.get("size"))
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price = safe_float(level[0])
            size = safe_float(level[1])
        else:
            continue
        if price is not None and size is not None:
            levels.append((price, size))
    levels.sort(key=lambda item: item[0], reverse=(side == "bid"))
    return [{"price": price, "size": size} for price, size in levels[:top_n]]


def summarize_book(raw: dict[str, Any], top_n: int) -> dict[str, Any]:
    bids = normalize_book_levels(raw.get("bids"), "bid", top_n)
    asks = normalize_book_levels(raw.get("asks"), "ask", top_n)
    best_bid = bids[0]["price"] if bids else None
    best_ask = asks[0]["price"] if asks else None
    mid = (
        (best_bid + best_ask) / 2
        if best_bid is not None and best_ask is not None
        else None
    )
    spread = (
        best_ask - best_bid
        if best_bid is not None and best_ask is not None
        else None
    )
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_size": bids[0]["size"] if bids else None,
        "ask_size": asks[0]["size"] if asks else None,
        "mid": mid,
        "spread": spread,
        "top_bids": bids,
        "top_asks": asks,
    }


def fetch_pm_books(client: httpx.Client, market: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes = market.get("outcomes") or []
    for idx, token_id in enumerate(market.get("clob_token_ids") or []):
        row: dict[str, Any] = {
            "asset": market.get("asset"),
            "family": market.get("family"),
            "slug": market.get("slug"),
            "condition_id": market.get("condition_id"),
            "outcome_index": idx,
            "outcome": outcomes[idx] if idx < len(outcomes) else "",
            "token_id": token_id,
        }
        try:
            raw = fetch_json(client, f"{CLOB_BASE}/book", {"token_id": token_id})
            row.update(summarize_book(raw if isinstance(raw, dict) else {}, top_n))
            row["raw_book"] = raw
        except Exception as exc:
            row["error"] = repr(exc)
        rows.append(row)
    return rows


def fetch_pm_trades(client: httpx.Client, market: dict[str, Any], limit: int) -> dict[str, Any]:
    condition_id = market.get("condition_id")
    if not condition_id:
        return {"error": "missing condition_id", "trades": []}
    try:
        trades = fetch_json(
            client,
            f"{DATA_API_BASE}/trades",
            {"market": condition_id, "limit": limit, "takerOnly": "true"},
        )
        return {"source": f"{DATA_API_BASE}/trades", "trades": trades if isinstance(trades, list) else []}
    except Exception as exc:
        return {"error": repr(exc), "trades": []}


def fetch_binance_ticker(client: httpx.Client, asset: str) -> dict[str, Any]:
    symbol = str(ASSETS[asset]["binance_symbol"])
    try:
        raw = fetch_json(client, f"{BINANCE_BASE}/api/v3/ticker/price", {"symbol": symbol})
        return {
            "source": "binance_ticker_price",
            "symbol": symbol,
            "price": safe_float(raw.get("price") if isinstance(raw, dict) else None),
            "raw": raw,
        }
    except Exception as exc:
        return {"source": "binance_ticker_price", "symbol": symbol, "error": repr(exc)}


def fetch_binance_open(
    client: httpx.Client,
    asset: str,
    start: datetime,
    interval: str,
) -> dict[str, Any]:
    symbol = str(ASSETS[asset]["binance_symbol"])
    try:
        raw = fetch_json(
            client,
            f"{BINANCE_BASE}/api/v3/klines",
            {"symbol": symbol, "interval": interval, "startTime": utc_ms(start), "limit": 1},
        )
        if isinstance(raw, list) and raw:
            kline = raw[0]
            return {
                "source": f"binance_{interval}_open",
                "symbol": symbol,
                "interval": interval,
                "open_time": kline[0],
                "open": safe_float(kline[1]),
                "close": safe_float(kline[4]),
                "raw": kline,
            }
        return {"source": f"binance_{interval}_open", "symbol": symbol, "interval": interval, "raw": raw}
    except Exception as exc:
        return {"source": f"binance_{interval}_open", "symbol": symbol, "interval": interval, "error": repr(exc)}


def fetch_deribit_index(client: httpx.Client, asset: str) -> dict[str, Any]:
    index_name = str(ASSETS[asset]["deribit_index"])
    try:
        raw = fetch_json(client, f"{DERIBIT_BASE}/public/get_index_price", {"index_name": index_name})
        result = raw.get("result", {}) if isinstance(raw, dict) else {}
        return {
            "source": "deribit_index_price",
            "index_name": index_name,
            "price": safe_float(result.get("index_price")),
            "raw": raw,
        }
    except Exception as exc:
        return {"source": "deribit_index_price", "index_name": index_name, "error": repr(exc)}


def parse_pyth_price(item: dict[str, Any]) -> dict[str, Any]:
    price = item.get("price") or {}
    expo = safe_float(price.get("expo"))
    raw_price = safe_float(price.get("price"))
    raw_conf = safe_float(price.get("conf"))
    scale = 10 ** expo if expo is not None else None
    return {
        "id": item.get("id"),
        "price": raw_price * scale if raw_price is not None and scale is not None else None,
        "conf": raw_conf * scale if raw_conf is not None and scale is not None else None,
        "expo": expo,
        "publish_time": price.get("publish_time"),
        "raw": item,
    }


def fetch_pyth_prices(client: httpx.Client, assets: list[str]) -> dict[str, Any]:
    params = [("ids[]", str(ASSETS[asset]["pyth_price_id"])) for asset in assets]
    id_to_asset = {str(ASSETS[asset]["pyth_price_id"]): asset for asset in assets}
    try:
        raw = fetch_json(client, f"{PYTH_BASE}/v2/updates/price/latest", params)
        parsed = raw.get("parsed", []) if isinstance(raw, dict) else []
        prices: dict[str, Any] = {}
        for item in parsed:
            price_id = str(item.get("id") or "")
            asset = id_to_asset.get(price_id, price_id)
            prices[asset] = parse_pyth_price(item)
        return {"source": "pyth_hermes_latest", "prices": prices, "raw": raw}
    except Exception as exc:
        return {"source": "pyth_hermes_latest", "error": repr(exc)}


def reference_snapshots(client: httpx.Client, assets: list[str]) -> dict[str, Any]:
    refs: dict[str, Any] = {"by_asset": {}, "pyth": fetch_pyth_prices(client, assets)}
    for asset in assets:
        refs["by_asset"][asset] = {
            "binance": fetch_binance_ticker(client, asset),
            "deribit": fetch_deribit_index(client, asset),
            "chainlink": {
                "source": "chainlink_data_stream",
                "url": ASSETS[asset]["chainlink_stream"],
                "status": "metadata_only_no_unauth_price_parser",
            },
        }
    return refs


def pm_strike_probe(client: httpx.Client, market: dict[str, Any]) -> dict[str, Any]:
    asset = str(market["asset"])
    start = parse_dt(market.get("event_start_time")) or parse_dt(market.get("window_start"))
    if start is None:
        return {"strike": None, "source": "missing_window_start"}
    family = str(market.get("family"))
    if family == "hourly":
        probe = fetch_binance_open(client, asset, start, "1h")
        return {
            "strike": probe.get("open"),
            "source": "official_pm_binance_1h_open",
            "official_resolution_source": market.get("resolution_source"),
            "probe": probe,
        }
    probe = fetch_binance_open(client, asset, start, "4h")
    return {
        "strike": probe.get("open"),
        "source": "binance_4h_open_fallback_for_chainlink_pm",
        "official_resolution_source": market.get("resolution_source"),
        "note": "PM 4h official source is Chainlink Data Streams; unauthenticated raw stream price was not parsed here.",
        "probe": probe,
    }


def load_deribit_instruments(client: httpx.Client, asset: str, expiry: datetime) -> list[dict[str, Any]]:
    ccy = str(ASSETS[asset]["deribit_currency"])
    data = fetch_json(
        client,
        f"{DERIBIT_BASE}/public/get_instruments",
        {"currency": ccy, "kind": "option", "expired": "false"},
    )
    instruments = data.get("result", []) if isinstance(data, dict) else []
    target_ms = utc_ms(expiry)
    return [inst for inst in instruments if int(inst.get("expiration_timestamp") or -1) == target_ms]


def choose_strikes(strikes: list[float], center: float, each_side: int) -> list[float]:
    unique = sorted(set(float(s) for s in strikes if math.isfinite(float(s))))
    below = [strike for strike in unique if strike <= center]
    above = [strike for strike in unique if strike > center]
    selected = below[-each_side:] + above[:each_side]
    if len(selected) < each_side * 2:
        selected_set = set(selected)
        for strike in sorted(unique, key=lambda s: abs(s - center)):
            if strike not in selected_set:
                selected.append(strike)
                selected_set.add(strike)
            if len(selected) >= min(len(unique), each_side * 2):
                break
    return sorted(selected)


def instrument_index(instruments: list[dict[str, Any]]) -> dict[tuple[float, str], dict[str, Any]]:
    out: dict[tuple[float, str], dict[str, Any]] = {}
    for inst in instruments:
        strike = safe_float(inst.get("strike"))
        option_type = str(inst.get("option_type") or "").lower()
        if strike is None or option_type not in {"call", "put"}:
            continue
        out[(strike, "C" if option_type == "call" else "P")] = inst
    return out


def fetch_deribit_books(
    client: httpx.Client,
    asset: str,
    expiry: datetime,
    center: float,
    each_side: int,
    depth: int,
) -> dict[str, Any]:
    instruments = load_deribit_instruments(client, asset, expiry)
    strikes = [float(inst["strike"]) for inst in instruments if inst.get("strike") is not None]
    selected_strikes = choose_strikes(strikes, center, each_side)
    by_key = instrument_index(instruments)
    rows: list[dict[str, Any]] = []
    for strike in selected_strikes:
        for option_type in ("C", "P"):
            inst = by_key.get((strike, option_type))
            if not inst:
                rows.append({"strike": strike, "option_type": option_type, "error": "instrument_not_found"})
                continue
            name = str(inst.get("instrument_name"))
            row: dict[str, Any] = {
                "asset": asset,
                "instrument_name": name,
                "strike": strike,
                "option_type": option_type,
                "expiry": iso(expiry),
                "instrument_metadata": inst,
            }
            try:
                raw = fetch_json(
                    client,
                    f"{DERIBIT_BASE}/public/get_order_book",
                    {"instrument_name": name, "depth": depth},
                )
                result = raw.get("result", {}) if isinstance(raw, dict) else {}
                row.update(
                    {
                        "state": result.get("state"),
                        "index_price": safe_float(result.get("index_price")),
                        "underlying_price": safe_float(result.get("underlying_price")),
                        "estimated_delivery_price": safe_float(result.get("estimated_delivery_price")),
                        "underlying_index": result.get("underlying_index"),
                        "mark_price": safe_float(result.get("mark_price")),
                        "mark_iv": safe_float(result.get("mark_iv")),
                        "bid_iv": safe_float(result.get("bid_iv")),
                        "ask_iv": safe_float(result.get("ask_iv")),
                        "best_bid_price": safe_float(result.get("best_bid_price")),
                        "best_ask_price": safe_float(result.get("best_ask_price")),
                        "best_bid_amount": safe_float(result.get("best_bid_amount")),
                        "best_ask_amount": safe_float(result.get("best_ask_amount")),
                        "open_interest": safe_float(result.get("open_interest")),
                        "stats": result.get("stats"),
                        "greeks": result.get("greeks"),
                        "top_bids": normalize_book_levels(result.get("bids"), "bid", depth),
                        "top_asks": normalize_book_levels(result.get("asks"), "ask", depth),
                        "raw_order_book": raw,
                    }
                )
            except Exception as exc:
                row["error"] = repr(exc)
            rows.append(row)
    return {
        "asset": asset,
        "expiry": iso(expiry),
        "center": center,
        "strikes_each_side": each_side,
        "selected_strikes": selected_strikes,
        "instrument_count_for_expiry": len(instruments),
        "books": rows,
    }


def up_mid(pm_books: list[dict[str, Any]]) -> float | None:
    for book in pm_books:
        if str(book.get("outcome")).lower() == "up" or book.get("outcome_index") == 0:
            return safe_float(book.get("mid"))
    return None


def next_cadence_seconds(markets: list[dict[str, Any]], pm_books_by_slug: dict[str, list[dict[str, Any]]]) -> int:
    now = utc_now()
    final_hour = False
    midband = False
    for market in markets:
        end = parse_dt(market.get("end_date")) or parse_dt(market.get("window_end"))
        if end is not None and timedelta(0) <= end - now <= timedelta(hours=1):
            final_hour = True
        mid = up_mid(pm_books_by_slug.get(str(market.get("slug")), []))
        if mid is not None and 0.40 <= mid <= 0.60:
            midband = True
    return 5 if final_hour or midband else 30


def build_snapshot(
    client: httpx.Client,
    *,
    run_id: str,
    markets: list[dict[str, Any]],
    expiry_by_asset: dict[str, datetime],
    top_n: int,
    trade_limit: int,
    strikes_each_side: int,
    deribit_depth: int,
) -> dict[str, Any]:
    assets = sorted({str(market["asset"]) for market in markets})
    captured_at = utc_now()
    pm_books_by_slug: dict[str, list[dict[str, Any]]] = {}
    pm_trades_by_slug: dict[str, Any] = {}
    strike_by_slug: dict[str, Any] = {}

    for market in markets:
        slug = str(market["slug"])
        pm_books_by_slug[slug] = fetch_pm_books(client, market, top_n)
        pm_trades_by_slug[slug] = fetch_pm_trades(client, market, trade_limit)
        probe = pm_strike_probe(client, market)
        strike_by_slug[slug] = {**probe, "asset": market.get("asset"), "family": market.get("family")}

    refs = reference_snapshots(client, assets)
    deribit_by_slug: dict[str, Any] = {}
    for market in markets:
        slug = str(market["slug"])
        asset = str(market["asset"])
        strike_probe = strike_by_slug.get(slug, {})
        center = safe_float(strike_probe.get("strike"))
        if center is None:
            center = safe_float(((refs.get("by_asset") or {}).get(asset) or {}).get("deribit", {}).get("price"))
        if center is None:
            center = safe_float(((refs.get("by_asset") or {}).get(asset) or {}).get("binance", {}).get("price"))
        if center is None:
            deribit_by_slug[slug] = {"asset": asset, "slug": slug, "error": "no center price available"}
            continue
        deribit_row = fetch_deribit_books(
            client,
            asset,
            expiry_by_asset[asset],
            center,
            strikes_each_side,
            deribit_depth,
        )
        deribit_row["slug"] = slug
        deribit_row["family"] = market.get("family")
        deribit_row["pm_strike_probe"] = strike_probe
        deribit_by_slug[slug] = deribit_row

    cadence = next_cadence_seconds(markets, pm_books_by_slug)
    return {
        "schema": "od_rv_deribit_aligned_snapshot_v1",
        "run_id": run_id,
        "captured_at": iso(captured_at),
        "cadence_seconds_next": cadence,
        "markets": markets,
        "pm_books_by_slug": pm_books_by_slug,
        "pm_recent_trades_by_slug": pm_trades_by_slug,
        "pm_strike_probes_by_slug": strike_by_slug,
        "deribit_by_market_slug": deribit_by_slug,
        "reference_snapshots": refs,
    }


def default_run_id(expiry: datetime) -> str:
    return f"od_rv_deribit_aligned_{expiry.strftime('%Y%m%dT%H%M%SZ')}"


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return iso(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default) + "\n")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def parse_assets(raw: str) -> list[str]:
    assets = [part.strip().upper() for part in raw.split(",") if part.strip()]
    unknown = [asset for asset in assets if asset not in ASSETS]
    if unknown:
        raise SystemExit(f"unknown assets {unknown}; allowed: {sorted(ASSETS)}")
    return assets


def parse_families(raw: str) -> list[str]:
    families = [part.strip().lower() for part in raw.split(",") if part.strip()]
    allowed = {"4h", "hourly"}
    unknown = [family for family in families if family not in allowed]
    if unknown:
        raise SystemExit(f"unknown families {unknown}; allowed: {sorted(allowed)}")
    return families


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default="BTC,ETH")
    parser.add_argument("--families", default="4h,hourly")
    parser.add_argument("--expiry-date", help="Deribit expiry date in YYYY-MM-DD; expiry time is discovered from Deribit")
    parser.add_argument("--run-id")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--duration-hours", type=float)
    parser.add_argument("--post-expiry-minutes", type=float, default=10.0)
    parser.add_argument("--top-n-pm-depth", type=int, default=5)
    parser.add_argument("--trade-limit", type=int, default=100)
    parser.add_argument("--strikes-each-side", type=int, default=5)
    parser.add_argument("--deribit-depth", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets = parse_assets(args.assets)
    families = parse_families(args.families)

    with httpx.Client(timeout=args.timeout_seconds, headers={"User-Agent": "epsilon-od-rv-capture/1.0"}) as client:
        expiry_rows = {asset: choose_expiry(discover_deribit_expiries(client, asset), args.expiry_date) for asset in assets}
        expiry_by_asset = {asset: row.expiry for asset, row in expiry_rows.items()}
        unique_expiries = sorted(set(expiry_by_asset.values()))
        if len(unique_expiries) != 1:
            raise SystemExit(f"assets resolved to different Deribit expiries: {expiry_by_asset}")
        expiry = unique_expiries[0]
        run_id = args.run_id or default_run_id(expiry)
        run_dir = args.out_dir / run_id
        out_path = run_dir / f"{run_id}.jsonl"
        manifest_path = run_dir / "manifest.json"

        specs = [spec for asset in assets for spec in pm_specs_for_expiry(asset, expiry, families)]
        markets = [market for spec in specs if (market := discover_pm_market(client, spec)) is not None]
        missing = [asdict(spec) for spec in specs if not any(market.get("slug") == spec.slug for market in markets)]
        if not markets:
            raise SystemExit("no aligned Polymarket markets discovered for the selected expiry")

        manifest = {
            "schema": "od_rv_deribit_aligned_manifest_v1",
            "created_at": iso(utc_now()),
            "run_id": run_id,
            "out_path": str(out_path.relative_to(ROOT)),
            "assets": assets,
            "families": families,
            "deribit_expiries": {asset: asdict(row) for asset, row in expiry_rows.items()},
            "markets": markets,
            "missing_pm_specs": missing,
            "cadence_rule": "30s base; 5s in final hour or whenever PM UP mid is in [0.40, 0.60]",
            "notes": [
                "Read-only forward capture; no order placement.",
                "PM daily crypto is intentionally not discovered here because it resolves 16:00 UTC, not Deribit 08:00 UTC.",
                "PM hourly uses Binance 1H open/close; PM 4h uses Chainlink Data Streams and stores Binance/Pyth/Deribit as reference diagnostics.",
            ],
        }
        write_manifest(manifest_path, manifest)

        print(f"run_id={run_id}")
        print(f"deribit_expiry={iso(expiry)}")
        print(f"markets={', '.join(str(m['slug']) for m in markets)}")
        print(f"writing={out_path.relative_to(ROOT)}")
        if missing:
            print(f"missing_pm_specs={len(missing)}")
        if args.dry_run:
            return 0

        deadline = (
            utc_now() + timedelta(hours=args.duration_hours)
            if args.duration_hours is not None
            else expiry + timedelta(minutes=args.post_expiry_minutes)
        )
        if args.once:
            deadline = utc_now()

        while True:
            started = time.monotonic()
            snapshot = build_snapshot(
                client,
                run_id=run_id,
                markets=markets,
                expiry_by_asset=expiry_by_asset,
                top_n=args.top_n_pm_depth,
                trade_limit=args.trade_limit,
                strikes_each_side=args.strikes_each_side,
                deribit_depth=args.deribit_depth,
            )
            write_jsonl(out_path, snapshot)
            print(
                f"{snapshot['captured_at']} wrote snapshot; "
                f"next_cadence={snapshot['cadence_seconds_next']}s"
            )
            if args.once or utc_now() >= deadline:
                break
            elapsed = time.monotonic() - started
            sleep_s = max(0.0, float(snapshot["cadence_seconds_next"]) - elapsed)
            time.sleep(sleep_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
