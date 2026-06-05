"""OD/MM Gate 0 universe and capacity map for PM financial binaries.

This is a cheap screening script. It deliberately stops at market family,
reference, spread/tick/fee, and historical maker-concentration capacity
proxies. It does not build fair-value models or test residual pricing edge.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/od_cross_asset_gate0_universe_map.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import httpx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
CSV_OUT = ANALYSIS / "csv_outputs" / "options_delta"
NOTE = ROOT / "notes" / "options_delta" / "od_cross_asset_gate0_universe_map_findings.md"

OUT_FAMILY = CSV_OUT / "od_cross_asset_gate0_universe_map.csv"
OUT_MARKETS = CSV_OUT / "od_cross_asset_gate0_market_detail.csv"
OUT_REFERENCES = CSV_OUT / "od_cross_asset_gate0_reference_checks.csv"

TRADES_DIR = DATA / "trades"
K5_WALLET_CACHE = ANALYSIS / "k5_stress_wallet_market_full.parquet"
LOCAL_MARKETS = sorted((DATA / "markets").glob("markets_*.parquet"))[-1]

sys.path.insert(0, str(ROOT))
from data_infra.operator_denylist import EXCHANGE_INTERNAL_LEG  # noqa: E402


AS_OF = pd.Timestamp("2026-06-02T12:30:00Z")
RECENT_DAYS = 90
RECENT_CUTOFF = AS_OF - pd.Timedelta(days=RECENT_DAYS)
MIN_VOLUME_DAY_USD = 50_000.0
MIN_HEADROOM_DAY_USD = 25_000.0
HTTP_HEADERS = {"User-Agent": "epsilon-quant-research-od-gate0/1.0"}


@dataclass(frozen=True)
class SeriesSpec:
    family: str
    slug: str
    label: str
    subtype: str
    underlying: str


SERIES: list[SeriesSpec] = [
    SeriesSpec("crypto_daily", "btc-up-or-down-daily", "BTC daily up/down", "terminal_direction", "BTC"),
    SeriesSpec("crypto_daily", "eth-up-or-down-daily", "ETH daily up/down", "terminal_direction", "ETH"),
    SeriesSpec("crypto_daily", "solana-up-or-down-daily", "SOL daily up/down", "terminal_direction", "SOL"),
    SeriesSpec("index_up_down", "spx-daily-up-or-down", "SPX daily up/down", "terminal_direction", "SPX"),
    SeriesSpec("index_up_down", "ndx-daily-up-or-down", "NDX daily up/down", "terminal_direction", "NDX"),
    SeriesSpec("single_stock_up_down", "nvda-daily-up-down", "NVDA daily up/down", "terminal_direction", "NVDA"),
    SeriesSpec("single_stock_up_down", "tsla-daily-up-down", "TSLA daily up/down", "terminal_direction", "TSLA"),
    SeriesSpec("single_stock_up_down", "meta-daily-up-down", "META daily up/down", "terminal_direction", "META"),
    SeriesSpec("single_stock_up_down", "googl-daily-up-down", "GOOGL daily up/down", "terminal_direction", "GOOGL"),
    SeriesSpec("single_stock_up_down", "msft-daily-up-down", "MSFT daily up/down", "terminal_direction", "MSFT"),
    SeriesSpec("single_stock_up_down", "amzn-daily-up-down", "AMZN daily up/down", "terminal_direction", "AMZN"),
    SeriesSpec("single_stock_up_down", "aapl-daily-up-down", "AAPL daily up/down", "terminal_direction", "AAPL"),
    SeriesSpec("close_above_price_band", "btc-multi-strikes-weekly", "BTC weekly multi-strike", "barrier_threshold", "BTC"),
    SeriesSpec("close_above_price_band", "ethereum-multi-strikes-weekly", "ETH weekly multi-strike", "barrier_threshold", "ETH"),
    SeriesSpec("close_above_price_band", "solana-multi-strikes-weekly", "SOL weekly multi-strike", "barrier_threshold", "SOL"),
    SeriesSpec("close_above_price_band", "bitcoin-hit-price-daily", "BTC daily hit-price", "barrier_threshold", "BTC"),
    SeriesSpec("close_above_price_band", "ethereum-hit-price-daily", "ETH daily hit-price", "barrier_threshold", "ETH"),
    SeriesSpec("close_above_price_band", "solana-hit-price-daily", "SOL daily hit-price", "barrier_threshold", "SOL"),
    SeriesSpec("close_above_price_band", "bitcoin-hit-price-monthly", "BTC monthly hit-price", "barrier_threshold", "BTC"),
    SeriesSpec("close_above_price_band", "ethereum-hit-price-monthly", "ETH monthly hit-price", "barrier_threshold", "ETH"),
    SeriesSpec("close_above_price_band", "solana-hit-price-monthly", "SOL monthly hit-price", "barrier_threshold", "SOL"),
    SeriesSpec("close_above_price_band", "sp-500-monthly-ou", "SPX monthly close-above", "terminal_price_band", "SPX"),
    SeriesSpec("close_above_price_band", "spx-multi-strikes-weekly", "SPX weekly close-above", "terminal_price_band", "SPX"),
    SeriesSpec("close_above_price_band", "ndx-multi-strikes-weekly", "NDX weekly close-above", "terminal_price_band", "NDX"),
    SeriesSpec("close_above_price_band", "nvidia-multi-strikes-monthly", "NVDA monthly close-above", "terminal_price_band", "NVDA"),
    SeriesSpec("close_above_price_band", "google-multi-strikes-monthly", "GOOGL monthly close-above", "terminal_price_band", "GOOGL"),
    SeriesSpec("close_above_price_band", "tesla-multi-strikes-monthly", "TSLA monthly close-above", "terminal_price_band", "TSLA"),
    SeriesSpec("close_above_price_band", "tsla-multi-strikes-weekly", "TSLA weekly close-above", "terminal_price_band", "TSLA"),
]

NEG_RISK_SEARCH_TERMS = [
    "what will sp 500 close at",
    "what will s&p 500 close at",
    "what will spx open at",
    "what will sp 500 open at end",
    "what will nasdaq 100 close at",
    "what will ndx close at",
    "what will tesla close at",
    "what will nvidia close at",
    "what will bitcoin close at",
    "what will ethereum close at",
]


REFERENCE_ROWS = [
    {
        "family": "crypto_daily",
        "clean_external_reference": True,
        "reference_read": "Binance settlement source on current PM daily templates; BTC/ETH have Deribit listed options for forward capture, SOL has no Deribit analogue.",
        "free_api_or_surface": "Binance public candles; Deribit public live instruments/books for BTC/ETH.",
        "caveat": "Gate 0 clears volume/capacity, but the parallel OD-RV settlement check parks literal 16:00 UTC PM daily vs 08:00 UTC Deribit daily. Use settlement-aligned BTC/ETH windows or find a clean 16:00 external surface.",
    },
    {
        "family": "index_up_down",
        "clean_external_reference": True,
        "reference_read": "SPX/NDX PM templates settle on official closes; SPX/NDX or ETF/futures options exist for external digital/call-spread checks.",
        "free_api_or_surface": "Official close pages are public; live option/futures surfaces are listed, but clean historical API access is usually delayed/premium.",
        "caveat": "Future test needs a declared data path for option/futures quotes; the PM settlement timestamp is clean.",
    },
    {
        "family": "single_stock_up_down",
        "clean_external_reference": True,
        "reference_read": "Liquid single names have official closes and listed OCC equity options.",
        "free_api_or_surface": "Daily close APIs exist; historical option-chain APIs generally require keys/premium tiers.",
        "caveat": "Corporate actions, halts, and official-close handling must be encoded; current PM flow is the main blocker.",
    },
    {
        "family": "close_above_price_band",
        "clean_external_reference": False,
        "reference_read": "The terminal close-above subset has listed-option analogues, but most volume comes from crypto hit/threshold baskets that are path-dependent barrier claims.",
        "free_api_or_surface": "Spot/futures feeds are available; no clean free listed barrier-option surface was identified.",
        "caveat": "Do not let high crypto threshold volume reopen a homemade barrier model under Gate 0.",
    },
    {
        "family": "neg_risk_baskets",
        "clean_external_reference": False,
        "reference_read": "True financial neg-risk range baskets surfaced as one-offs, not recurring liquid templates.",
        "free_api_or_surface": "External price reference can exist, but fair value depends on PM internal range/basket consistency and merge/split accounting.",
        "caveat": "Treat as PM-internal consistency research only if a liquid recurring financial basket appears.",
    },
]


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
        return pd.Timestamp(value, tz="UTC")
    except Exception:
        try:
            return pd.Timestamp(value).tz_localize("UTC")
        except Exception:
            return pd.NaT


def dollars(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"${value:,.0f}"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def cents(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}c"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def get_json(client: httpx.Client, path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"https://gamma-api.polymarket.com{path}"
    resp = client.get(url, params=params or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def resolve_series(client: httpx.Client, slug: str) -> dict[str, Any] | None:
    rows = get_json(client, "/series", {"slug": slug, "limit": 3})
    for row in rows:
        if row.get("slug") == slug:
            return row
    return rows[0] if rows else None


def fetch_series_events(client: httpx.Client, spec: SeriesSpec) -> list[dict[str, Any]]:
    series = resolve_series(client, spec.slug)
    if not series:
        print(f"warning: missing series {spec.slug}")
        return []
    series_id = series.get("id")
    events: dict[str, dict[str, Any]] = {}
    for closed in (False, True):
        for offset in range(0, 1200, 100):
            params = {
                "series_id": series_id,
                "closed": str(closed).lower(),
                "limit": 100,
                "offset": offset,
                "order": "closedTime" if closed else "endDate",
                "ascending": "false",
            }
            page = get_json(client, "/events", params)
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


def fetch_neg_risk_events(client: httpx.Client) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    financial_words = re.compile(
        r"\b(spx|s&p|s&p 500|nasdaq|ndx|qqq|tesla|tsla|nvidia|nvda|bitcoin|btc|ethereum|eth|solana|sol)\b",
        re.IGNORECASE,
    )
    for term in NEG_RISK_SEARCH_TERMS:
        js = get_json(client, "/public-search", {"q": term, "limit": 20})
        for event in js.get("events", []) or []:
            title = f"{event.get('title') or ''} {event.get('slug') or ''}"
            event_ts = ts(event.get("closedTime")) if event.get("closedTime") else ts(event.get("endDate"))
            is_live = bool(event.get("active")) and not bool(event.get("closed"))
            is_recent = bool(event.get("closed")) and pd.notna(event_ts) and event_ts >= RECENT_CUTOFF
            if not (is_live or is_recent):
                continue
            if not (event.get("negRisk") or event.get("enableNegRisk")):
                continue
            if not financial_words.search(title):
                continue
            slug = event.get("slug")
            try:
                event = get_json(client, f"/events/slug/{slug}") if slug else event
            except Exception:
                pass
            event["_family"] = "neg_risk_baskets"
            event["_series_slug"] = event.get("seriesSlug") or "direct_search"
            event["_series_label"] = "financial neg-risk basket"
            event["_subtype"] = "neg_risk_range"
            event["_underlying"] = infer_underlying(title)
            events[str(event.get("id") or slug)] = event
        time.sleep(0.03)
    return list(events.values())


def infer_underlying(text: str) -> str:
    text_l = text.lower()
    for key, value in [
        ("bitcoin", "BTC"),
        ("btc", "BTC"),
        ("ethereum", "ETH"),
        ("eth", "ETH"),
        ("solana", "SOL"),
        ("sol", "SOL"),
        ("s&p", "SPX"),
        ("spx", "SPX"),
        ("nasdaq", "NDX"),
        ("ndx", "NDX"),
        ("qqq", "QQQ"),
        ("tesla", "TSLA"),
        ("tsla", "TSLA"),
        ("nvidia", "NVDA"),
        ("nvda", "NVDA"),
        ("google", "GOOGL"),
        ("googl", "GOOGL"),
        ("meta", "META"),
        ("microsoft", "MSFT"),
        ("msft", "MSFT"),
        ("amazon", "AMZN"),
        ("amzn", "AMZN"),
        ("apple", "AAPL"),
        ("aapl", "AAPL"),
    ]:
        if key in text_l:
            return value
    return "unknown"


def flatten_events(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        markets = event.get("markets") or []
        if not markets and event.get("id"):
            markets = [event]
        for market in markets:
            outcomes = parse_json_list(market.get("outcomes"))
            outcome_prices = parse_json_list(market.get("outcomePrices"))
            clob_token_ids = parse_json_list(market.get("clobTokenIds"))
            market_ts = ts(market.get("closedTime")) if market.get("closedTime") else ts(market.get("endDate"))
            event_ts = ts(event.get("closedTime")) if event.get("closedTime") else ts(event.get("endDate"))
            settlement_ts = market_ts if pd.notna(market_ts) else event_ts
            active_raw = bool(market.get("active", event.get("active"))) and not bool(
                market.get("closed", event.get("closed"))
            )
            closed = bool(market.get("closed", event.get("closed")))
            volume = num(market.get("volume"))
            volume_24h = num(market.get("volume24hr"))
            if not np.isfinite(volume):
                volume = num(event.get("volume"))
            if not np.isfinite(volume_24h):
                volume_24h = 0.0 if closed else num(event.get("volume24hr"))
            is_recent = closed and pd.notna(settlement_ts) and settlement_ts >= RECENT_CUTOFF
            is_live = active_raw and (pd.isna(settlement_ts) or settlement_ts >= AS_OF or volume_24h > 0)
            if not is_live and not is_recent:
                continue
            if (not np.isfinite(volume) or volume <= 0) and (not np.isfinite(volume_24h) or volume_24h <= 0):
                continue
            best_bid = num(market.get("bestBid"))
            best_ask = num(market.get("bestAsk"))
            spread = num(market.get("spread"))
            if not np.isfinite(spread) and np.isfinite(best_bid) and np.isfinite(best_ask):
                spread = best_ask - best_bid
            rows.append(
                {
                    "family": event.get("_family"),
                    "series_slug": event.get("_series_slug") or event.get("seriesSlug"),
                    "series_label": event.get("_series_label") or "",
                    "subtype": event.get("_subtype") or "",
                    "underlying": event.get("_underlying") or infer_underlying(event.get("title") or market.get("question") or ""),
                    "event_id": str(event.get("id") or ""),
                    "event_slug": event.get("slug") or "",
                    "event_title": event.get("title") or "",
                    "market_id": str(market.get("id") or ""),
                    "condition_id": str(market.get("conditionId") or market.get("condition_id") or ""),
                    "market_slug": market.get("slug") or "",
                    "market_question": market.get("question") or event.get("title") or "",
                    "active": is_live,
                    "closed": closed,
                    "settlement_ts_utc": settlement_ts.isoformat() if pd.notna(settlement_ts) else "",
                    "resolution_source": market.get("resolutionSource") or event.get("resolutionSource") or "",
                    "description_snippet": (market.get("description") or event.get("description") or "")[:500].replace("\n", " "),
                    "volume_usd": volume if np.isfinite(volume) else 0.0,
                    "volume_24h_usd": volume_24h if np.isfinite(volume_24h) else 0.0,
                    "liquidity_usd": num(market.get("liquidity")),
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "order_price_min_tick_size": num(market.get("orderPriceMinTickSize")),
                    "order_min_size": num(market.get("orderMinSize")),
                    "maker_base_fee": num(market.get("makerBaseFee")),
                    "taker_base_fee": num(market.get("takerBaseFee")),
                    "fees_enabled": bool(market.get("feesEnabled")) if market.get("feesEnabled") is not None else None,
                    "rewards_max_spread": num(market.get("rewardsMaxSpread")),
                    "rewards_min_size": num(market.get("rewardsMinSize")),
                    "neg_risk": bool(market.get("negRisk", event.get("negRisk", False))),
                    "enable_neg_risk_event": bool(event.get("enableNegRisk", False)),
                    "n_outcomes": len(outcomes),
                    "n_clob_tokens": len(clob_token_ids),
                    "outcomes": json.dumps(outcomes),
                    "outcome_prices": json.dumps(outcome_prices),
                    "clob_token_ids": json.dumps(clob_token_ids),
                    "concentration_source": "local_orderfilled_exact_pending",
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(["family", "market_id"])
    if df.empty:
        return df
    return df.sort_values(["family", "active", "volume_24h_usd", "volume_usd"], ascending=[True, False, False, False])


def sql_list(values: list[str]) -> str:
    vals = sorted({str(v).lower().replace("'", "''") for v in values if str(v)})
    return ", ".join(f"'{v}'" for v in vals) if vals else "''"


def recent_trade_paths() -> list[str]:
    """Return raw trade shards that can contain the 90-day Gate 0 window.

    The local raw archive is large. Candidate markets here are live or recently
    resolved, so older shards cannot contribute exact concentration rows.
    """
    paths: list[str] = []
    date_re = re.compile(r"(20\d{2})-?(\d{2})-?(\d{2})")
    for path in sorted(TRADES_DIR.glob("*.parquet")):
        matches = date_re.findall(path.name)
        if not matches:
            continue
        dates = []
        for y, m, d in matches:
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
    quoted = []
    for path in paths:
        escaped = path.replace("'", "''")
        quoted.append(f"'{escaped}'")
    return "[" + ", ".join(quoted) + "]"


def local_concentration(condition_ids: list[str]) -> pd.DataFrame:
    if not condition_ids:
        return pd.DataFrame()
    con = duckdb.connect()
    temp_dir = ANALYSIS / ".duckdb_tmp_od_gate0"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    ids = sql_list(condition_ids)
    query = f"""
        WITH local_market_map AS (
            SELECT
                CAST(id AS VARCHAR) AS market_id,
                lower(CAST(condition_id AS VARCHAR)) AS condition_id
            FROM read_parquet('{LOCAL_MARKETS}')
            WHERE lower(CAST(condition_id AS VARCHAR)) IN ({ids})
        ),
        scoped AS (
            SELECT
                lmm.condition_id,
                lower(k.address) AS maker,
                max(k.global_maker_usd) AS maker_usd,
                max(k.global_maker_fills) AS maker_fills,
                min(k.global_market_maker_rank) AS maker_rank
            FROM read_parquet('{K5_WALLET_CACHE}') k
            JOIN local_market_map lmm ON CAST(k.market_id AS VARCHAR) = lmm.market_id
            WHERE k.global_maker_usd IS NOT NULL
              AND k.global_market_maker_rank IS NOT NULL
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT
                *,
                sum(maker_usd) OVER (PARTITION BY condition_id) AS total_maker_usd
            FROM scoped
        )
        SELECT
            condition_id,
            sum(total_maker_usd) / nullif(count(*), 0) AS local_maker_usd,
            sum(maker_usd) FILTER (WHERE maker_rank <= 3) AS top3_maker_usd,
            sum(maker_usd) FILTER (WHERE maker_rank BETWEEN 4 AND 23) AS next20_maker_usd,
            count(*) AS maker_wallets,
            sum(maker_fills) AS local_maker_fills,
            sum(maker_usd) FILTER (WHERE maker_rank <= 3) / nullif(max(total_maker_usd), 0) AS top3_maker_share,
            sum(maker_usd) FILTER (WHERE maker_rank BETWEEN 4 AND 23) / nullif(max(total_maker_usd), 0) AS next20_maker_share,
            1.0 - coalesce(sum(maker_usd) FILTER (WHERE maker_rank <= 3) / nullif(max(total_maker_usd), 0), 0.0)
                AS non_top3_maker_share
        FROM ranked
        GROUP BY 1
    """
    out = con.execute(query).df()
    return out


def apply_concentration(markets: pd.DataFrame, conc: pd.DataFrame) -> pd.DataFrame:
    if markets.empty:
        return markets
    markets = markets.copy()
    markets["condition_id"] = markets["condition_id"].astype(str).str.lower()
    merged = markets.merge(conc, on="condition_id", how="left")
    merged["has_local_concentration"] = merged["top3_maker_share"].notna()
    proxy = (
        merged[merged["has_local_concentration"]]
        .groupby("family", as_index=False)
        .agg(
            family_proxy_top3_share=("top3_maker_share", "median"),
            family_proxy_next20_share=("next20_maker_share", "median"),
            family_proxy_non_top3_share=("non_top3_maker_share", "median"),
        )
    )
    merged = merged.merge(proxy, on="family", how="left")
    for col, default in [
        ("family_proxy_top3_share", 0.70),
        ("family_proxy_next20_share", 0.20),
        ("family_proxy_non_top3_share", 0.30),
    ]:
        merged[col] = merged[col].fillna(default)
    merged["top3_maker_share_effective"] = merged["top3_maker_share"].fillna(merged["family_proxy_top3_share"])
    merged["next20_maker_share_effective"] = merged["next20_maker_share"].fillna(merged["family_proxy_next20_share"])
    merged["non_top3_maker_share_effective"] = merged["non_top3_maker_share"].fillna(
        merged["family_proxy_non_top3_share"]
    )
    merged["concentration_source"] = np.where(
        merged["has_local_concentration"],
        "k5_stress_cache_condition_join",
        "family_proxy_from_k5_stress_cache",
    )
    return merged


def summarize(markets: pd.DataFrame) -> pd.DataFrame:
    refs = pd.DataFrame(REFERENCE_ROWS)
    if markets.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for family, g in markets.groupby("family"):
        live = g[g["active"]]
        recent = g[g["closed"]]
        run_rate = live["volume_24h_usd"].sum()
        if run_rate <= 0:
            run_rate = recent["volume_usd"].sum() / RECENT_DAYS
        non_top3_weighted = np.average(
            g["non_top3_maker_share_effective"].fillna(0.30),
            weights=np.maximum(g["volume_usd"].fillna(0.0), 1.0),
        )
        top3_weighted = np.average(
            g["top3_maker_share_effective"].fillna(0.70),
            weights=np.maximum(g["volume_usd"].fillna(0.0), 1.0),
        )
        next20_weighted = np.average(
            g["next20_maker_share_effective"].fillna(0.20),
            weights=np.maximum(g["volume_usd"].fillna(0.0), 1.0),
        )
        headroom_day = run_rate * non_top3_weighted
        sample = g.sort_values(["active", "volume_24h_usd", "volume_usd"], ascending=[False, False, False]).iloc[0]
        rows.append(
            {
                "family": family,
                "live_market_count": int(live["market_id"].nunique()),
                "recent_resolved_market_count": int(recent["market_id"].nunique()),
                "total_market_count": int(g["market_id"].nunique()),
                "live_24h_volume_usd": float(live["volume_24h_usd"].sum()),
                "recent_90d_volume_usd": float(recent["volume_usd"].sum()),
                "run_rate_volume_usd_per_day": float(run_rate),
                "median_live_spread_cents": float(live["spread"].dropna().median() * 100.0)
                if not live["spread"].dropna().empty
                else math.nan,
                "median_tick_size_cents": float(g["order_price_min_tick_size"].dropna().median() * 100.0)
                if not g["order_price_min_tick_size"].dropna().empty
                else math.nan,
                "median_order_min_size_usd": float(g["order_min_size"].dropna().median())
                if not g["order_min_size"].dropna().empty
                else math.nan,
                "median_taker_base_fee": float(g["taker_base_fee"].dropna().median())
                if not g["taker_base_fee"].dropna().empty
                else math.nan,
                "median_maker_base_fee": float(g["maker_base_fee"].dropna().median())
                if not g["maker_base_fee"].dropna().empty
                else math.nan,
                "median_rewards_max_spread_cents": float(g["rewards_max_spread"].dropna().median())
                if not g["rewards_max_spread"].dropna().empty
                else math.nan,
                "weighted_top3_maker_share": float(top3_weighted),
                "weighted_next20_maker_share": float(next20_weighted),
                "weighted_non_top3_maker_share": float(non_top3_weighted),
                "non_top3_headroom_usd_per_day": float(headroom_day),
                "markets_with_exact_local_concentration": int(g["has_local_concentration"].sum()),
                "sample_resolution_source": sample.get("resolution_source", ""),
                "sample_settlement_ts_utc": sample.get("settlement_ts_utc", ""),
                "sample_event": sample.get("event_slug", ""),
                "sample_market": sample.get("market_slug", ""),
            }
        )
    summary = pd.DataFrame(rows).merge(refs, on="family", how="left")
    summary["meaningful_volume"] = summary["run_rate_volume_usd_per_day"] >= MIN_VOLUME_DAY_USD
    summary["clears_headroom_min"] = summary["non_top3_headroom_usd_per_day"] >= MIN_HEADROOM_DAY_USD
    summary["merits_fair_value_test"] = (
        summary["meaningful_volume"]
        & summary["clean_external_reference"].fillna(False)
        & summary["clears_headroom_min"]
    )
    summary["rank_score"] = np.where(
        summary["clean_external_reference"].fillna(False),
        summary["non_top3_headroom_usd_per_day"],
        0.0,
    )
    summary["decision"] = np.where(
        summary["merits_fair_value_test"],
        "MERITS-FAIR-VALUE-TEST",
        np.where(summary["meaningful_volume"], "DO-NOT-PURSUE: reference/capacity gate failed", "DO-NOT-PURSUE: thin"),
    )
    return summary.sort_values(["merits_fair_value_test", "rank_score"], ascending=[False, False])


def write_note(summary: pd.DataFrame, markets: pd.DataFrame, references: pd.DataFrame) -> None:
    ranked_rows = []
    for i, row in summary.reset_index(drop=True).iterrows():
        ranked_rows.append(
            [
                str(i + 1),
                str(row["family"]),
                str(row["decision"]),
                dollars(row["run_rate_volume_usd_per_day"]),
                dollars(row["non_top3_headroom_usd_per_day"]),
                pct(row["weighted_top3_maker_share"]),
                cents(row["median_live_spread_cents"] / 100.0)
                if np.isfinite(row["median_live_spread_cents"])
                else "n/a",
                "yes" if row.get("clean_external_reference") else "no",
            ]
        )
    sample_rows = []
    samples = (
        markets.sort_values(["family", "active", "volume_24h_usd", "volume_usd"], ascending=[True, False, False, False])
        .groupby("family")
        .head(3)
    )
    for _, row in samples.iterrows():
        sample_volume = row["volume_24h_usd"] if row["active"] and row["volume_24h_usd"] > 0 else row["volume_usd"]
        sample_rows.append(
            [
                row["family"],
                row["event_slug"] or row["market_slug"],
                "live" if row["active"] else "recent",
                dollars(sample_volume),
                row["settlement_ts_utc"][:19],
                (row["resolution_source"] or "description-only")[:55],
                row["concentration_source"].replace("_", " "),
            ]
        )
    note = f"""# PM Financial-Binary Gate 0 Universe And Capacity Map

> Hub: [[strat_options_delta]] · [[POLYMARKET_BRAIN]]
> Related: [[od_cross_asset_updown_scoping]] · [[od_pricing_model_form_findings]] · [[mm_deployable_cells_findings]]
> CSV: `data/analysis/csv_outputs/options_delta/od_cross_asset_gate0_universe_map.csv`

## Headline

Ranked by non-incumbent headroom times clean-reference availability, **only crypto-daily clears Gate 0 for a future external-fair-value test**. That is a universe/capacity pass, not permission to ignore settlement: the parallel OD-RV settlement check parks literal PM daily BTC/ETH vs Deribit daily because PM settles at 16:00 UTC and Deribit expires at 08:00 UTC. Index up/down and single-stock up/down are clean but sub-scale under the stated volume/headroom minimum. Close-above/price-band has a lot of volume, but the current volume is mostly crypto hit/barrier style threshold markets rather than clean terminal digitals, so it should not reopen a homemade pricing-model branch. True financial negative-risk baskets were sparse one-offs in this scrape.

Decision rule was pre-registered before the script ran: a family merits a future fair-value test only if daily run-rate volume is at least `{dollars(MIN_VOLUME_DAY_USD)}`, non-top3 headroom is at least `{dollars(MIN_HEADROOM_DAY_USD)}` per day, and a clean external reference exists. The 90-day recent window is `{RECENT_CUTOFF.date()}` through `{AS_OF.date()}`. This note is only Gate 0; it does not claim any fair-value edge.

## Ranked Families

Unit of observation is a Polymarket binary market row. `run-rate volume` uses current live 24h volume when live markets exist; otherwise it uses recent 90-day resolved volume divided by 90. `non-top3 headroom` is run-rate volume multiplied by the K5-style non-top3 maker-share proxy. Exact concentration comes from the K5 stress wallet-market cache, joined through local Gamma condition IDs; newer markets that are not in the local metadata inherit the family median from matched cached markets.

{markdown_table(["rank", "family", "decision", "run-rate volume/day", "non-top3 headroom/day", "top3 share", "median live spread", "clean ref"], ranked_rows)}

Read: the cheapest decisive filter does not say crypto-daily has edge. It says crypto-daily has enough volume, a plausible external reference, and enough non-incumbent fill flow to justify one later fair-value/capture gate. Everything else stops here unless the live universe changes.

## Sample Markets And Settlement Sources

The exact settlement timestamp used here is Gamma `endDate`/`closedTime`, stored per market in the detail CSV. The source string is the Gamma `resolutionSource` field when populated, otherwise the market description is the only source text.

{markdown_table(["family", "sample", "state", "volume field", "settlement UTC", "resolution source", "concentration"], sample_rows)}

## External Reference Read

{markdown_table(["family", "clean?", "reference read", "caveat"], [[r["family"], "yes" if r["clean_external_reference"] else "no", r["reference_read"], r["caveat"]] for _, r in references.iterrows()])}

## Public Sources Checked

- [Polymarket market data overview](https://docs.polymarket.com/market-data/overview) for Gamma, CLOB, and Data API surface area.
- [Polymarket fees](https://docs.polymarket.com/trading/fees) for fee formula, category taker rates, and maker-rebate framing.
- [Deribit public instruments API](https://docs.deribit.com/api-reference/market-data/public-get_instruments) for BTC/ETH listed option availability and the live/expired instrument split.
- [Alpha Vantage documentation](https://www.alphavantage.co/documentation/) for equity daily data and historical options API availability/premium caveats.
- [Cboe SPX options specifications](https://www.cboe.com/tradable_products/sp_500/spx_options/specifications/) for listed SPX option surface availability.
- [OCC equity options product specifications](https://www.theocc.com/Clearance-and-Settlement/Clearing/Equity-Options-Product-Specifications) for listed single-stock option structure.
- [Yahoo Finance historical prices help](https://help.yahoo.com/kb/finance/historical-prices-sln2311.html) for settlement-source availability and licensing caveats on downloadable historical prices.

## Family Decisions

**crypto-daily:** **MERITS-FAIR-VALUE-TEST on Gate 0 only**, but the later test must obey settlement matching. BTC/ETH have a Deribit options analogue and clean Binance settlement source on current daily PM templates, yet [[od_rv_deribit_daily_capture_findings]] parks literal 16:00 UTC PM daily vs 08:00 UTC Deribit daily. The actionable route is settlement-aligned BTC/ETH capture or a clean 16:00 external surface; SOL should be excluded or treated as spot/futures-only until a real options surface exists. This is not permission to reopen crypto-4h absolute pricing.

**index up/down:** **DO-NOT-PURSUE for now: thin.** SPX/NDX settlement is clean and listed options/futures exist, but current live run-rate and non-top3 headroom both miss the minimum. Recheck if the daily SPX/NDX templates grow above the headroom threshold; no Gate 1 spend now.

**single-stock up/down:** **DO-NOT-PURSUE for now: thin.** The external reference story is clean for liquid names, but live flow and non-top3 headroom are below the minimum. Recheck only if the daily stock templates grow by an order of magnitude.

**close-above/price-band:** **DO-NOT-PURSUE as a family despite volume.** The high-volume rows are mostly crypto threshold/hit markets, which are path-dependent barrier claims. Terminal close-above subsets can be clean, but their current flow is not what drives the family totals. Do not use this as a backdoor into another handcrafted model-form test.

**neg-risk baskets:** **DO-NOT-PURSUE: thin and not clean enough.** The true financial neg-risk rows found were sparse one-offs. If a recurring, liquid financial range basket appears, it should first be an internal-consistency/merge-split accounting task, not OD fair value.

## Guardrails

- No pricing model, fair-value residual, or PnL test was run here.
- Maker concentration is a capacity proxy, not proof that a faster or better-priced entrant cannot steal share.
- Markets missing from the local Gamma metadata/K5 cache are not assigned exact maker concentration; those rows use family proxies.
- The close-above/price-band volume is not a clean terminal digital volume estimate because barrier/hit templates dominate the scrape.

## Outputs

- Family CSV: `data/analysis/csv_outputs/options_delta/od_cross_asset_gate0_universe_map.csv`
- Market detail CSV: `data/analysis/csv_outputs/options_delta/od_cross_asset_gate0_market_detail.csv`
- Reference checks CSV: `data/analysis/csv_outputs/options_delta/od_cross_asset_gate0_reference_checks.csv`
- Script: `scripts/od_cross_asset_gate0_universe_map.py`
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> int:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HTTP_HEADERS) as client:
        events: list[dict[str, Any]] = []
        for spec in SERIES:
            events.extend(fetch_series_events(client, spec))
        events.extend(fetch_neg_risk_events(client))
    markets = flatten_events(events)
    if markets.empty:
        raise SystemExit("no candidate markets fetched")
    conc = local_concentration(markets["condition_id"].dropna().astype(str).tolist())
    markets = apply_concentration(markets, conc)
    refs = pd.DataFrame(REFERENCE_ROWS)
    summary = summarize(markets)

    summary.to_csv(OUT_FAMILY, index=False)
    markets.to_csv(OUT_MARKETS, index=False)
    refs.to_csv(OUT_REFERENCES, index=False)
    write_note(summary, markets, refs)

    print(f"wrote {OUT_FAMILY.relative_to(ROOT)} ({len(summary)} rows)")
    print(f"wrote {OUT_MARKETS.relative_to(ROOT)} ({len(markets)} rows)")
    print(f"wrote {OUT_REFERENCES.relative_to(ROOT)} ({len(refs)} rows)")
    print(f"wrote {NOTE.relative_to(ROOT)}")
    print(summary[["family", "decision", "run_rate_volume_usd_per_day", "non_top3_headroom_usd_per_day"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
