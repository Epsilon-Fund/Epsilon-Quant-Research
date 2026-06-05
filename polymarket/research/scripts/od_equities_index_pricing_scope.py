"""Equity financial-binary scope for OD pricing reopen.

This is the cheap pass requested before any listed-options build. It refreshes
the Polymarket equity binary universe, measures persistence and small-capital
capacity, and declares the external option-data path for a later pricing gate.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/od_equities_index_pricing_scope.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import httpx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
CSV_OUT = ANALYSIS / "csv_outputs" / "options_delta"
TRADES_DIR = DATA / "trades"

OUT_MARKETS = CSV_OUT / "od_equities_index_pricing_scope_market_detail.csv"
OUT_TEMPLATES = CSV_OUT / "od_equities_index_pricing_scope_template_summary.csv"
OUT_REFERENCES = CSV_OUT / "od_equities_index_pricing_scope_reference_paths.csv"

sys.path.insert(0, str(ROOT))
from data_infra.operator_denylist import EXCHANGE_INTERNAL_LEG  # noqa: E402


AS_OF = pd.Timestamp(datetime.now(tz=UTC)).floor("s")
RECENT_DAYS = 90
RECENT_CUTOFF = AS_OF - pd.Timedelta(days=RECENT_DAYS)
APPROX_BUSINESS_DAYS = len(pd.bdate_range(RECENT_CUTOFF.date(), AS_OF.date()))
SMALL_TICKET_USD = 100.0
SMALL_CAP_HEADROOM_MULTIPLE = 10.0
SMALL_CAP_MIN_HEADROOM_USD = SMALL_TICKET_USD * SMALL_CAP_HEADROOM_MULTIPLE
MAX_SMALL_CAP_SPREAD_CENTS = 3.0
HTTP_HEADERS = {"User-Agent": "epsilon-quant-research-od-equities-scope/1.0"}


@dataclass(frozen=True)
class SeriesSpec:
    family: str
    slug: str
    label: str
    subtype: str
    underlying: str


SERIES: list[SeriesSpec] = [
    SeriesSpec("index_up_down", "spx-daily-up-or-down", "SPX daily up/down", "terminal_direction", "SPX"),
    SeriesSpec("index_up_down", "ndx-daily-up-or-down", "NDX daily up/down", "terminal_direction", "NDX"),
    SeriesSpec("single_stock_up_down", "nvda-daily-up-down", "NVDA daily up/down", "terminal_direction", "NVDA"),
    SeriesSpec("single_stock_up_down", "tsla-daily-up-down", "TSLA daily up/down", "terminal_direction", "TSLA"),
    SeriesSpec("single_stock_up_down", "meta-daily-up-down", "META daily up/down", "terminal_direction", "META"),
    SeriesSpec("single_stock_up_down", "googl-daily-up-down", "GOOGL daily up/down", "terminal_direction", "GOOGL"),
    SeriesSpec("single_stock_up_down", "msft-daily-up-down", "MSFT daily up/down", "terminal_direction", "MSFT"),
    SeriesSpec("single_stock_up_down", "amzn-daily-up-down", "AMZN daily up/down", "terminal_direction", "AMZN"),
    SeriesSpec("single_stock_up_down", "aapl-daily-up-down", "AAPL daily up/down", "terminal_direction", "AAPL"),
    SeriesSpec("close_above_ladder", "sp-500-monthly-ou", "SPX monthly close-above", "terminal_price_ladder", "SPX"),
    SeriesSpec("close_above_ladder", "spx-multi-strikes-weekly", "SPX weekly close-above", "terminal_price_ladder", "SPX"),
    SeriesSpec("close_above_ladder", "ndx-multi-strikes-weekly", "NDX weekly close-above", "terminal_price_ladder", "NDX"),
    SeriesSpec("close_above_ladder", "nvidia-multi-strikes-monthly", "NVDA monthly close-above", "terminal_price_ladder", "NVDA"),
    SeriesSpec("close_above_ladder", "google-multi-strikes-monthly", "GOOGL monthly close-above", "terminal_price_ladder", "GOOGL"),
    SeriesSpec("close_above_ladder", "tesla-multi-strikes-monthly", "TSLA monthly close-above", "terminal_price_ladder", "TSLA"),
    SeriesSpec("close_above_ladder", "tsla-multi-strikes-weekly", "TSLA weekly close-above", "terminal_price_ladder", "TSLA"),
]

SEARCH_TERMS = [
    "spx up or down",
    "s&p 500 up or down",
    "ndx up or down",
    "nasdaq 100 up or down",
    "nvda up or down",
    "nvidia up or down",
    "tsla up or down",
    "tesla up or down",
    "meta up or down",
    "googl up or down",
    "google up or down",
    "msft up or down",
    "microsoft up or down",
    "amzn up or down",
    "amazon up or down",
    "aapl up or down",
    "apple up or down",
    "spx above",
    "s&p 500 above",
    "ndx above",
    "nasdaq above",
    "nvidia above",
    "tesla above",
    "google above",
]

REFERENCE_ROWS = [
    {
        "scope": "index_up_down",
        "primary_path": "Cboe DataShop/LiveVol SPX/SPXW and NDX/XND option snapshots at Polymarket quote timestamps; official index close as settlement.",
        "pricing_method": "Compute a cash-or-nothing digital at the PM threshold using a tight vertical call-spread slope around strike, with Black-Scholes N(d2) from interpolated 0DTE IV as a cross-check.",
        "why_clean": "PM settles on official equity-index closes, and listed index options provide a real same-underlying volatility surface.",
        "blocking_detail": "Historical intraday option snapshots are vendor/API data, not present locally; build only after the capacity/persistence gate clears.",
    },
    {
        "scope": "single_stock_up_down",
        "primary_path": "OCC/OPRA equity option chain snapshots, or Alpha Vantage historical options if the required symbols/dates are covered; official equity close as settlement.",
        "pricing_method": "Same digital/call-spread method as index, with corporate-action/halt handling and symbol-specific dividend/borrow assumptions.",
        "why_clean": "Listed equity options exist for the liquid mega-cap names, but the PM markets are thinner and wider.",
        "blocking_detail": "Current PM spread/capacity must improve before spending on the single-name option surface.",
    },
    {
        "scope": "close_above_ladder",
        "primary_path": "Only terminal close-above ladders can use listed-option call spreads; path-dependent hit/barrier markets are out of scope.",
        "pricing_method": "For terminal ladders, compare each strike to a listed call-spread digital around that strike; do not use this for hit/dip markets.",
        "why_clean": "A terminal close-above strike has a listed-option analogue; a hit/dip barrier does not.",
        "blocking_detail": "Observed equity ladders are sparse/one-off relative to daily up/down templates.",
    },
]

EQUITY_RE = re.compile(
    r"\b(spx|s&p|s&p 500|sp 500|nasdaq|nasdaq 100|ndx|qqq|nvidia|nvda|tesla|tsla|meta|google|googl|microsoft|msft|amazon|amzn|apple|aapl)\b",
    re.IGNORECASE,
)


def num(x: Any) -> float:
    if x is None:
        return math.nan
    try:
        return float(x)
    except Exception:
        return math.nan


def parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            out = json.loads(value)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []


def ts(value: Any) -> pd.Timestamp | pd.NaT:
    if not value:
        return pd.NaT
    try:
        out = pd.Timestamp(value)
        if out.tzinfo is None:
            return out.tz_localize("UTC")
        return out.tz_convert("UTC")
    except Exception:
        return pd.NaT


def get_json(client: httpx.Client, base: str, path: str, params: dict[str, Any] | None = None) -> Any:
    response = client.get(f"{base}{path}", params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def infer_underlying(text: str) -> str:
    text_l = text.lower()
    pairs = [
        ("nasdaq", "NDX"),
        ("ndx", "NDX"),
        ("qqq", "QQQ"),
        ("s&p 500", "SPX"),
        ("s&p", "SPX"),
        ("spx", "SPX"),
        ("sp 500", "SPX"),
        ("nvidia", "NVDA"),
        ("nvda", "NVDA"),
        ("tesla", "TSLA"),
        ("tsla", "TSLA"),
        ("meta", "META"),
        ("google", "GOOGL"),
        ("googl", "GOOGL"),
        ("microsoft", "MSFT"),
        ("msft", "MSFT"),
        ("amazon", "AMZN"),
        ("amzn", "AMZN"),
        ("apple", "AAPL"),
        ("aapl", "AAPL"),
    ]
    for key, value in pairs:
        if key in text_l:
            return value
    return "unknown"


def classify_event(event: dict[str, Any]) -> tuple[str, str, str] | None:
    text = " ".join(
        str(event.get(k) or "")
        for k in ["slug", "title", "seriesSlug", "description"]
    ).lower()
    if not EQUITY_RE.search(text):
        return None
    if event.get("negRisk") or event.get("enableNegRisk"):
        return None
    underlying = infer_underlying(text)
    is_index = underlying in {"SPX", "NDX", "QQQ"}
    updown = "up-or-down" in text or "up or down" in text or "up-down" in text
    close_above = "above" in text or "close" in text or "multi-strikes" in text
    if updown and is_index:
        return "index_up_down", "terminal_direction", underlying
    if updown:
        return "single_stock_up_down", "terminal_direction", underlying
    if close_above:
        return "close_above_ladder", "terminal_price_ladder", underlying
    return None


def resolve_series(client: httpx.Client, slug: str) -> dict[str, Any] | None:
    rows = get_json(client, "https://gamma-api.polymarket.com", "/series", {"slug": slug, "limit": 3})
    for row in rows:
        if row.get("slug") == slug:
            return row
    return rows[0] if rows else None


def fetch_series_events(client: httpx.Client, spec: SeriesSpec) -> list[dict[str, Any]]:
    series = resolve_series(client, spec.slug)
    if not series:
        print(f"[warn] missing series {spec.slug}", flush=True)
        return []
    events: dict[str, dict[str, Any]] = {}
    series_id = series.get("id")
    for closed in (False, True):
        for offset in range(0, 2000, 100):
            page = get_json(
                client,
                "https://gamma-api.polymarket.com",
                "/events",
                {
                    "series_id": series_id,
                    "closed": str(closed).lower(),
                    "limit": 100,
                    "offset": offset,
                    "order": "closedTime" if closed else "endDate",
                    "ascending": "false",
                },
            )
            if not page:
                break
            old_closed_seen = 0
            for event in page:
                event_ts = ts(event.get("closedTime")) if event.get("closedTime") else ts(event.get("endDate"))
                if closed and pd.notna(event_ts) and event_ts < RECENT_CUTOFF:
                    old_closed_seen += 1
                    continue
                event["_family"] = spec.family
                event["_series_slug"] = spec.slug
                event["_series_label"] = spec.label
                event["_subtype"] = spec.subtype
                event["_underlying"] = spec.underlying
                events[str(event.get("id") or event.get("slug"))] = event
            if len(page) < 100 or (closed and old_closed_seen >= 90):
                break
            time.sleep(0.03)
    return list(events.values())


def fetch_search_events(client: httpx.Client) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for term in SEARCH_TERMS:
        page = get_json(client, "https://gamma-api.polymarket.com", "/public-search", {"q": term, "limit": 30})
        for stub in page.get("events", []) or []:
            classified = classify_event(stub)
            if not classified:
                continue
            event_ts = ts(stub.get("closedTime")) if stub.get("closedTime") else ts(stub.get("endDate"))
            active = bool(stub.get("active")) and not bool(stub.get("closed"))
            recent = bool(stub.get("closed")) and pd.notna(event_ts) and event_ts >= RECENT_CUTOFF
            if not (active or recent):
                continue
            slug = stub.get("slug")
            event = stub
            if slug:
                try:
                    event = get_json(client, "https://gamma-api.polymarket.com", f"/events/slug/{slug}")
                except Exception:
                    event = stub
            family, subtype, underlying = classified
            event["_family"] = family
            event["_series_slug"] = event.get("seriesSlug") or "search"
            event["_series_label"] = f"{underlying} search-discovered"
            event["_subtype"] = subtype
            event["_underlying"] = underlying
            events[str(event.get("id") or slug)] = event
        time.sleep(0.03)
    return list(events.values())


def clob_book_stats(client: httpx.Client, token_ids: list[Any]) -> dict[str, float]:
    spreads: list[float] = []
    ask_depths: list[float] = []
    bid_depths: list[float] = []
    for token_id in token_ids:
        if not token_id:
            continue
        try:
            book = get_json(client, "https://clob.polymarket.com", "/book", {"token_id": str(token_id)})
        except Exception:
            continue
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if not bids or not asks:
            continue
        best_bid = max((num(b.get("price")) for b in bids), default=math.nan)
        best_ask = min((num(a.get("price")) for a in asks), default=math.nan)
        if not np.isfinite(best_bid) or not np.isfinite(best_ask):
            continue
        bid_row = max(bids, key=lambda b: num(b.get("price")))
        ask_row = min(asks, key=lambda a: num(a.get("price")))
        bid_size = num(bid_row.get("size"))
        ask_size = num(ask_row.get("size"))
        spread = best_ask - best_bid
        if np.isfinite(spread) and spread >= 0:
            spreads.append(spread)
        if np.isfinite(ask_size):
            ask_depths.append(best_ask * ask_size)
        if np.isfinite(bid_size):
            bid_depths.append(best_bid * bid_size)
        time.sleep(0.01)
    return {
        "clob_spread": min(spreads) if spreads else math.nan,
        "clob_best_ask_depth_usd": max(ask_depths) if ask_depths else math.nan,
        "clob_best_bid_depth_usd": max(bid_depths) if bid_depths else math.nan,
    }


def flatten_events(events: list[dict[str, Any]], client: httpx.Client) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        family = event.get("_family")
        subtype = event.get("_subtype")
        underlying = event.get("_underlying")
        if not family:
            classified = classify_event(event)
            if not classified:
                continue
            family, subtype, underlying = classified
        markets = event.get("markets") or []
        for market in markets:
            condition_id = str(market.get("conditionId") or market.get("condition_id") or "").lower()
            if not condition_id or condition_id == "none":
                continue
            if market.get("negRisk") or event.get("negRisk") or event.get("enableNegRisk"):
                continue
            outcomes = parse_json_list(market.get("outcomes"))
            outcome_prices = parse_json_list(market.get("outcomePrices"))
            clob_token_ids = parse_json_list(market.get("clobTokenIds"))
            market_ts = ts(market.get("closedTime")) if market.get("closedTime") else ts(market.get("endDate"))
            event_ts = ts(event.get("closedTime")) if event.get("closedTime") else ts(event.get("endDate"))
            settlement_ts = market_ts if pd.notna(market_ts) else event_ts
            active = bool(market.get("active", event.get("active"))) and not bool(market.get("closed", event.get("closed")))
            closed = bool(market.get("closed", event.get("closed")))
            is_recent = closed and pd.notna(settlement_ts) and settlement_ts >= RECENT_CUTOFF
            is_live = active and (pd.isna(settlement_ts) or settlement_ts >= AS_OF - pd.Timedelta(hours=24))
            if not is_live and not is_recent:
                continue
            volume = num(market.get("volume"))
            volume_24h = num(market.get("volume24hr"))
            if not np.isfinite(volume):
                volume = num(event.get("volume"))
            if not np.isfinite(volume_24h):
                volume_24h = 0.0 if closed else num(event.get("volume24hr"))
            if (not np.isfinite(volume) or volume <= 0) and (not np.isfinite(volume_24h) or volume_24h <= 0):
                continue
            best_bid = num(market.get("bestBid"))
            best_ask = num(market.get("bestAsk"))
            gamma_spread = num(market.get("spread"))
            if not np.isfinite(gamma_spread) and np.isfinite(best_bid) and np.isfinite(best_ask):
                gamma_spread = best_ask - best_bid
            clob_stats = clob_book_stats(client, clob_token_ids) if is_live else {}
            spread = clob_stats.get("clob_spread", math.nan)
            if not np.isfinite(spread):
                spread = gamma_spread
            rows.append(
                {
                    "as_of_utc": AS_OF.isoformat(),
                    "recent_cutoff_utc": RECENT_CUTOFF.isoformat(),
                    "family": family,
                    "series_slug": event.get("_series_slug") or event.get("seriesSlug") or "search",
                    "series_label": event.get("_series_label") or "",
                    "subtype": subtype,
                    "underlying": underlying or infer_underlying(f"{event.get('title')} {market.get('question')}"),
                    "event_id": str(event.get("id") or ""),
                    "event_slug": event.get("slug") or "",
                    "event_title": event.get("title") or "",
                    "market_id": str(market.get("id") or ""),
                    "condition_id": condition_id,
                    "market_slug": market.get("slug") or "",
                    "market_question": market.get("question") or event.get("title") or "",
                    "active": is_live,
                    "closed": closed,
                    "settlement_ts_utc": settlement_ts.isoformat() if pd.notna(settlement_ts) else "",
                    "settlement_date_utc": settlement_ts.date().isoformat() if pd.notna(settlement_ts) else "",
                    "resolution_source": market.get("resolutionSource") or event.get("resolutionSource") or "",
                    "description_snippet": (market.get("description") or event.get("description") or "")[:500].replace("\n", " "),
                    "volume_usd": volume if np.isfinite(volume) else 0.0,
                    "volume_24h_usd": volume_24h if np.isfinite(volume_24h) else 0.0,
                    "liquidity_usd": num(market.get("liquidity")),
                    "gamma_best_bid": best_bid,
                    "gamma_best_ask": best_ask,
                    "gamma_spread": gamma_spread,
                    "clob_spread": clob_stats.get("clob_spread", math.nan),
                    "spread": spread,
                    "clob_best_ask_depth_usd": clob_stats.get("clob_best_ask_depth_usd", math.nan),
                    "clob_best_bid_depth_usd": clob_stats.get("clob_best_bid_depth_usd", math.nan),
                    "order_price_min_tick_size": num(market.get("orderPriceMinTickSize")),
                    "order_min_size": num(market.get("orderMinSize")),
                    "maker_base_fee": num(market.get("makerBaseFee")),
                    "taker_base_fee": num(market.get("takerBaseFee")),
                    "fees_enabled": bool(market.get("feesEnabled")) if market.get("feesEnabled") is not None else None,
                    "rewards_max_spread": num(market.get("rewardsMaxSpread")),
                    "rewards_min_size": num(market.get("rewardsMinSize")),
                    "n_outcomes": len(outcomes),
                    "n_clob_tokens": len(clob_token_ids),
                    "outcomes": json.dumps(outcomes),
                    "outcome_prices": json.dumps(outcome_prices),
                    "clob_token_ids": json.dumps(clob_token_ids),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(["condition_id", "market_slug"])
    return df.sort_values(["family", "underlying", "active", "volume_24h_usd", "volume_usd"], ascending=[True, True, False, False, False])


def sql_list(values: list[str] | set[str]) -> str:
    vals = sorted({str(v).lower().replace("'", "''") for v in values if str(v)})
    return ", ".join(f"'{v}'" for v in vals) if vals else "''"


def recent_trade_paths() -> list[str]:
    date_re = re.compile(r"(20\d{2})-?(\d{2})-?(\d{2})")
    paths: list[str] = []
    for path in sorted(TRADES_DIR.glob("*.parquet")):
        dates = []
        for y, m, d in date_re.findall(path.name):
            try:
                dates.append(pd.Timestamp(f"{y}-{m}-{d}", tz="UTC"))
            except Exception:
                continue
        if dates and max(dates) >= RECENT_CUTOFF - pd.Timedelta(days=2):
            paths.append(str(path))
    if not paths:
        raise SystemExit(f"no recent trade shards found in {TRADES_DIR}")
    return paths


def parquet_list(paths: list[str]) -> str:
    return "[" + ", ".join(f"'{p.replace(chr(39), chr(39) + chr(39))}'" for p in paths) + "]"


def raw_trade_concentration(condition_ids: list[str]) -> pd.DataFrame:
    if not condition_ids:
        return pd.DataFrame()
    con = duckdb.connect()
    temp_dir = ANALYSIS / ".duckdb_tmp_od_equities_scope"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    ids = sql_list(condition_ids)
    internals = sql_list(EXCHANGE_INTERNAL_LEG)
    paths = parquet_list(recent_trade_paths())
    return con.execute(
        f"""
        WITH scoped AS (
            SELECT
                lower(CAST(condition_id AS VARCHAR)) AS condition_id,
                lower(CAST(maker AS VARCHAR)) AS maker,
                sum(usd_amount) AS maker_usd,
                count(*) AS maker_fills
            FROM read_parquet({paths})
            WHERE lower(CAST(condition_id AS VARCHAR)) IN ({ids})
              AND maker IS NOT NULL
              AND taker IS NOT NULL
              AND lower(CAST(maker AS VARCHAR)) NOT IN ({internals})
              AND lower(CAST(taker AS VARCHAR)) NOT IN ({internals})
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (PARTITION BY condition_id ORDER BY maker_usd DESC) AS maker_rank,
                sum(maker_usd) OVER (PARTITION BY condition_id) AS total_maker_usd
            FROM scoped
        )
        SELECT
            condition_id,
            max(total_maker_usd) AS raw_trade_maker_usd,
            sum(maker_usd) FILTER (WHERE maker_rank <= 3) AS top3_maker_usd,
            sum(maker_usd) FILTER (WHERE maker_rank BETWEEN 4 AND 23) AS next20_maker_usd,
            count(*) AS maker_wallets,
            sum(maker_fills) AS maker_fills,
            sum(maker_usd) FILTER (WHERE maker_rank <= 3) / nullif(max(total_maker_usd), 0) AS top3_maker_share,
            sum(maker_usd) FILTER (WHERE maker_rank BETWEEN 4 AND 23) / nullif(max(total_maker_usd), 0) AS next20_maker_share,
            1.0 - coalesce(sum(maker_usd) FILTER (WHERE maker_rank <= 3) / nullif(max(total_maker_usd), 0), 0.0) AS non_top3_maker_share
        FROM ranked
        GROUP BY 1
        """
    ).df()


def apply_concentration(markets: pd.DataFrame, conc: pd.DataFrame) -> pd.DataFrame:
    if markets.empty:
        return markets
    markets = markets.copy()
    if conc.empty:
        markets["has_exact_concentration"] = False
        markets["top3_maker_share_effective"] = 0.70
        markets["next20_maker_share_effective"] = 0.20
        markets["non_top3_maker_share_effective"] = 0.30
        markets["concentration_source"] = "default_proxy_no_raw_trade_match"
        return markets
    markets = markets.merge(conc, on="condition_id", how="left")
    markets["has_exact_concentration"] = markets["top3_maker_share"].notna()
    proxy = (
        markets[markets["has_exact_concentration"]]
        .groupby(["family", "subtype"], as_index=False)
        .agg(
            proxy_top3=("top3_maker_share", "median"),
            proxy_next20=("next20_maker_share", "median"),
            proxy_non_top3=("non_top3_maker_share", "median"),
        )
    )
    markets = markets.merge(proxy, on=["family", "subtype"], how="left")
    markets["proxy_top3"] = markets["proxy_top3"].fillna(0.70)
    markets["proxy_next20"] = markets["proxy_next20"].fillna(0.20)
    markets["proxy_non_top3"] = markets["proxy_non_top3"].fillna(0.30)
    markets["top3_maker_share_effective"] = markets["top3_maker_share"].fillna(markets["proxy_top3"])
    markets["next20_maker_share_effective"] = markets["next20_maker_share"].fillna(markets["proxy_next20"])
    markets["non_top3_maker_share_effective"] = markets["non_top3_maker_share"].fillna(markets["proxy_non_top3"])
    markets["concentration_source"] = np.where(
        markets["has_exact_concentration"],
        "raw_recent_trades_condition_join",
        "equity_subset_family_proxy_from_raw_recent_trades",
    )
    return markets


def persistence_class(row: pd.Series) -> str:
    if row["subtype"] == "terminal_direction":
        if row["recent_event_dates"] >= 20 and row["median_event_gap_days"] <= 4.0:
            return "recurring_daily"
        if row["recent_event_dates"] >= 8:
            return "intermittent_daily"
        return "one_off_or_new"
    if row["recent_event_dates"] >= 6 and row["median_event_gap_days"] <= 10.0:
        return "recurring_weekly_ladder"
    if row["recent_event_dates"] >= 2:
        return "intermittent_ladder"
    return "one_off_or_new"


def summarize_templates(markets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if markets.empty:
        return pd.DataFrame()
    for keys, g in markets.groupby(["family", "subtype", "underlying", "series_slug"], dropna=False):
        family, subtype, underlying, series_slug = keys
        live = g[g["active"]]
        recent = g[g["closed"]]
        event_daily = (
            recent.groupby(["event_slug", "settlement_date_utc"], as_index=False)
            .agg(event_volume_usd=("volume_usd", "sum"))
            .query("event_volume_usd > 0")
        )
        event_vols = event_daily["event_volume_usd"].to_numpy(float)
        event_dates = pd.to_datetime(event_daily["settlement_date_utc"], errors="coerce").dropna().sort_values()
        gaps = event_dates.diff().dt.days.dropna().to_numpy(float)
        live_run_rate = float(live["volume_24h_usd"].sum())
        recent_run_rate = float(recent["volume_usd"].sum() / RECENT_DAYS)
        run_rate = live_run_rate if live_run_rate > 0 else recent_run_rate
        weights = np.maximum(g["volume_usd"].fillna(0.0), 1.0)
        non_top3_share = float(np.average(g["non_top3_maker_share_effective"].fillna(0.30), weights=weights))
        top3_share = float(np.average(g["top3_maker_share_effective"].fillna(0.70), weights=weights))
        next20_share = float(np.average(g["next20_maker_share_effective"].fillna(0.20), weights=weights))
        headroom = run_rate * non_top3_share
        row = {
            "as_of_utc": AS_OF.isoformat(),
            "recent_window_days": RECENT_DAYS,
            "approx_recent_business_days": APPROX_BUSINESS_DAYS,
            "family": family,
            "subtype": subtype,
            "underlying": underlying,
            "series_slug": series_slug,
            "live_market_count": int(live["market_id"].nunique()),
            "live_event_count": int(live["event_slug"].nunique()),
            "recent_resolved_market_count": int(recent["market_id"].nunique()),
            "recent_event_dates": int(event_dates.dt.date.nunique()) if len(event_dates) else 0,
            "business_day_persistence_share": float(event_dates.dt.date.nunique() / APPROX_BUSINESS_DAYS)
            if APPROX_BUSINESS_DAYS
            else math.nan,
            "median_event_gap_days": float(np.nanmedian(gaps)) if len(gaps) else math.nan,
            "live_24h_volume_usd": live_run_rate,
            "recent_90d_volume_usd": float(recent["volume_usd"].sum()),
            "recent_volume_usd_per_day": recent_run_rate,
            "run_rate_volume_usd_per_day": run_rate,
            "median_recent_event_volume_usd": float(np.nanmedian(event_vols)) if len(event_vols) else math.nan,
            "max_recent_event_volume_usd": float(np.nanmax(event_vols)) if len(event_vols) else math.nan,
            "max_to_median_event_volume": float(np.nanmax(event_vols) / np.nanmedian(event_vols))
            if len(event_vols) and np.nanmedian(event_vols) > 0
            else math.nan,
            "top3_event_volume_share": float(np.sort(event_vols)[-3:].sum() / event_vols.sum())
            if len(event_vols) and event_vols.sum() > 0
            else math.nan,
            "median_live_spread_cents": float(live["spread"].dropna().median() * 100.0)
            if not live["spread"].dropna().empty
            else math.nan,
            "median_live_clob_spread_cents": float(live["clob_spread"].dropna().median() * 100.0)
            if not live["clob_spread"].dropna().empty
            else math.nan,
            "median_live_best_ask_depth_usd": float(live["clob_best_ask_depth_usd"].dropna().median())
            if not live["clob_best_ask_depth_usd"].dropna().empty
            else math.nan,
            "weighted_top3_maker_share": top3_share,
            "weighted_next20_maker_share": next20_share,
            "weighted_non_top3_maker_share": non_top3_share,
            "non_top3_headroom_usd_per_day": headroom,
            "markets_with_exact_concentration": int(g["has_exact_concentration"].sum()),
            "exact_concentration_volume_share": float(
                g.loc[g["has_exact_concentration"], "volume_usd"].sum() / g["volume_usd"].sum()
            )
            if g["volume_usd"].sum() > 0
            else math.nan,
            "sample_event": str(g.sort_values(["active", "volume_24h_usd", "volume_usd"], ascending=[False, False, False]).iloc[0]["event_slug"]),
            "sample_settlement_ts_utc": str(g.sort_values(["active", "volume_24h_usd", "volume_usd"], ascending=[False, False, False]).iloc[0]["settlement_ts_utc"]),
            "sample_resolution_source": str(g.sort_values(["active", "volume_24h_usd", "volume_usd"], ascending=[False, False, False]).iloc[0]["resolution_source"]),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["persistence_class"] = summary.apply(persistence_class, axis=1)
    summary["spiky_recent_volume"] = (
        (summary["top3_event_volume_share"].fillna(1.0) >= 0.50)
        | (summary["max_to_median_event_volume"].fillna(99.0) >= 8.0)
    )
    summary["clears_small_cap_headroom"] = summary["non_top3_headroom_usd_per_day"] >= SMALL_CAP_MIN_HEADROOM_USD
    summary["clears_small_cap_spread"] = summary["median_live_spread_cents"].fillna(99.0) <= MAX_SMALL_CAP_SPREAD_CENTS
    summary["clears_persistence"] = summary["persistence_class"].isin(["recurring_daily", "intermittent_daily"])
    summary["clears_small_cap_capacity"] = (
        summary["clears_small_cap_headroom"]
        & summary["clears_small_cap_spread"]
        & summary["clears_persistence"]
        & summary["live_market_count"].gt(0)
    )
    summary["pricing_build_candidate"] = summary["clears_small_cap_capacity"] & summary["family"].eq("index_up_down")
    return summary.sort_values(
        ["pricing_build_candidate", "clears_small_cap_capacity", "non_top3_headroom_usd_per_day"],
        ascending=[False, False, False],
    )


def main() -> int:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HTTP_HEADERS) as client:
        events: list[dict[str, Any]] = []
        for spec in SERIES:
            events.extend(fetch_series_events(client, spec))
        events.extend(fetch_search_events(client))
        markets = flatten_events(events, client)
    if markets.empty:
        raise SystemExit("no equity candidate markets fetched")
    conc = raw_trade_concentration(markets["condition_id"].dropna().astype(str).tolist())
    markets = apply_concentration(markets, conc)
    templates = summarize_templates(markets)
    refs = pd.DataFrame(REFERENCE_ROWS)

    markets.to_csv(OUT_MARKETS, index=False)
    templates.to_csv(OUT_TEMPLATES, index=False)
    refs.to_csv(OUT_REFERENCES, index=False)

    print(f"as_of={AS_OF.isoformat()} recent_cutoff={RECENT_CUTOFF.isoformat()} approx_bdays={APPROX_BUSINESS_DAYS}")
    print(f"wrote {OUT_MARKETS.relative_to(ROOT)} ({len(markets)} rows)")
    print(f"wrote {OUT_TEMPLATES.relative_to(ROOT)} ({len(templates)} rows)")
    print(f"wrote {OUT_REFERENCES.relative_to(ROOT)} ({len(refs)} rows)")
    cols = [
        "family",
        "underlying",
        "series_slug",
        "persistence_class",
        "live_24h_volume_usd",
        "run_rate_volume_usd_per_day",
        "non_top3_headroom_usd_per_day",
        "median_live_spread_cents",
        "clears_small_cap_capacity",
        "pricing_build_candidate",
    ]
    print(templates[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
