"""SPX opens up/down current-data scope.

This is an additive pass for the equity-index pricing note. The prior SPX
daily up/down work scoped close-direction markets; this checks the separate
SPX Open Daily Up or Down series with current Gamma/CLOB data plus local
candidate metadata. It intentionally does not turn fills or post-open books
into a historical executable pricing backtest.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/od_equities_spx_open_updown_scope.py
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time
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
LIVE_CLOB_DIR = DATA / "live_clob"

OUT_MARKETS = CSV_OUT / "od_equities_spx_open_updown_scope_market_detail.csv"
OUT_SUMMARY = CSV_OUT / "od_equities_spx_open_updown_scope_summary.csv"
OUT_CLOB_SCAN = CSV_OUT / "od_equities_spx_open_updown_scope_local_clob_scan.csv"

AS_OF = pd.Timestamp(datetime.now(tz=UTC)).floor("s")
RECENT_DAYS = 90
RECENT_CUTOFF = AS_OF - pd.Timedelta(days=RECENT_DAYS)
SMALL_TICKET_USD = 100.0
SMALL_CAP_HEADROOM_MULTIPLE = 10.0
SMALL_CAP_MIN_HEADROOM_USD = SMALL_TICKET_USD * SMALL_CAP_HEADROOM_MULTIPLE
HTTP_HEADERS = {"User-Agent": "epsilon-quant-research-spx-open-updown-scope/1.0"}

SERIES_SLUG = "spx-open-daily-up-or-down"
SEARCH_TERMS = [
    "spx opens up or down",
    "spx open up or down",
    "s&p 500 opens up or down",
    "s&p 500 open up or down",
    "spx opens higher",
    "spx opens",
]
OPEN_RE = re.compile(r"\b(spx|s&p 500|s&p)\b.*\b(open|opens|opening)\b.*\b(up|down)\b", re.I)
DATE_RE = re.compile(r"(20\d{2})-?(\d{2})-?(\d{2})")


def num(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


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


def get_json(client: httpx.Client, base: str, path: str, params: dict[str, Any] | None = None) -> Any:
    response = client.get(f"{base}{path}", params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def resolve_series(client: httpx.Client) -> dict[str, Any]:
    rows = get_json(client, "https://gamma-api.polymarket.com", "/series", {"slug": SERIES_SLUG, "limit": 3})
    for row in rows:
        if row.get("slug") == SERIES_SLUG:
            return row
    return rows[0] if rows else {}


def fetch_series_events(client: httpx.Client) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    series = resolve_series(client)
    series_id = series.get("id")
    if series_id:
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
                old_seen = 0
                for event in page:
                    event_ts = ts(event.get("closedTime")) if event.get("closedTime") else ts(event.get("endDate"))
                    if closed and pd.notna(event_ts) and event_ts < RECENT_CUTOFF:
                        old_seen += 1
                        continue
                    events[str(event.get("id") or event.get("slug"))] = event
                if len(page) < 100 or old_seen >= 90:
                    break
                time.sleep(0.03)
    for term in SEARCH_TERMS:
        page = get_json(client, "https://gamma-api.polymarket.com", "/public-search", {"q": term, "limit": 30})
        for stub in page.get("events", []) or []:
            text = " ".join(str(stub.get(k) or "") for k in ["slug", "title", "description"])
            if not OPEN_RE.search(text):
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
            events[str(event.get("id") or slug)] = event
        time.sleep(0.03)
    return list(events.values())


def book_stats(client: httpx.Client, token_ids: list[Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx, token_id in enumerate(token_ids):
        if not token_id:
            continue
        try:
            book = get_json(client, "https://clob.polymarket.com", "/book", {"token_id": str(token_id)})
        except Exception:
            continue
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        best_bid = max((num(b.get("price")) for b in bids), default=math.nan)
        best_ask = min((num(a.get("price")) for a in asks), default=math.nan)
        best_bid_size = math.nan
        best_ask_size = math.nan
        if bids and np.isfinite(best_bid):
            best_bid_size = num(max(bids, key=lambda b: num(b.get("price"))).get("size"))
        if asks and np.isfinite(best_ask):
            best_ask_size = num(min(asks, key=lambda a: num(a.get("price"))).get("size"))
        rows.append(
            {
                "token_index": idx,
                "token_id": str(token_id),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "best_bid_size": best_bid_size,
                "best_ask_size": best_ask_size,
                "bid_depth_usd": best_bid * best_bid_size if np.isfinite(best_bid * best_bid_size) else math.nan,
                "ask_depth_usd": best_ask * best_ask_size if np.isfinite(best_ask * best_ask_size) else math.nan,
                "n_bids": len(bids),
                "n_asks": len(asks),
                "book_hash": book.get("hash") or "",
            }
        )
        time.sleep(0.01)
    if not rows:
        return {
            "current_clob_token_books": "[]",
            "current_clob_any_book": False,
            "current_clob_two_sided_spread": math.nan,
            "current_clob_min_tick_spread": math.nan,
            "current_clob_best_ask_depth_usd": math.nan,
            "current_clob_best_bid_depth_usd": math.nan,
        }
    books = pd.DataFrame(rows)
    spread_rows = books[books["best_bid"].notna() & books["best_ask"].notna()].copy()
    min_tick_spread = math.nan
    if not spread_rows.empty:
        min_tick_spread = float((spread_rows["best_ask"] - spread_rows["best_bid"]).min())
    return {
        "current_clob_token_books": json.dumps(rows),
        "current_clob_any_book": True,
        "current_clob_two_sided_spread": min_tick_spread,
        "current_clob_min_tick_spread": min_tick_spread,
        "current_clob_best_ask_depth_usd": float(books["ask_depth_usd"].max(skipna=True))
        if books["ask_depth_usd"].notna().any()
        else math.nan,
        "current_clob_best_bid_depth_usd": float(books["bid_depth_usd"].max(skipna=True))
        if books["bid_depth_usd"].notna().any()
        else math.nan,
    }


def flatten_events(events: list[dict[str, Any]], client: httpx.Client) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        markets = event.get("markets") or []
        for market in markets:
            text = " ".join(
                str(x or "")
                for x in [
                    event.get("slug"),
                    event.get("title"),
                    market.get("slug"),
                    market.get("question"),
                    market.get("description") or event.get("description"),
                ]
            )
            if not OPEN_RE.search(text):
                continue
            if event.get("negRisk") or event.get("enableNegRisk") or market.get("negRisk"):
                continue
            condition_id = str(market.get("conditionId") or market.get("condition_id") or "").lower()
            if not condition_id or condition_id == "none":
                continue
            event_ts = ts(event.get("closedTime")) if event.get("closedTime") else ts(event.get("endDate"))
            market_ts = ts(market.get("closedTime")) if market.get("closedTime") else ts(market.get("endDate"))
            settlement_ts = market_ts if pd.notna(market_ts) else event_ts
            active = bool(market.get("active", event.get("active"))) and not bool(market.get("closed", event.get("closed")))
            closed = bool(market.get("closed", event.get("closed")))
            recent = closed and pd.notna(settlement_ts) and settlement_ts >= RECENT_CUTOFF
            if not active and not recent:
                continue
            token_ids = parse_json_list(market.get("clobTokenIds"))
            outcomes = parse_json_list(market.get("outcomes"))
            outcome_prices = parse_json_list(market.get("outcomePrices"))
            current_stats = book_stats(client, token_ids) if active else {}
            start_ts = ts(event.get("startTime") or market.get("eventStartTime") or market.get("gameStartTime"))
            current_phase = "unknown"
            if pd.notna(start_ts):
                current_phase = "pre_open" if AS_OF < start_ts else "post_open_or_after_open_signal"
            rows.append(
                {
                    "as_of_utc": AS_OF.isoformat(),
                    "recent_cutoff_utc": RECENT_CUTOFF.isoformat(),
                    "series_slug": event.get("seriesSlug") or SERIES_SLUG,
                    "event_id": str(event.get("id") or ""),
                    "event_slug": event.get("slug") or "",
                    "event_title": event.get("title") or "",
                    "market_id": str(market.get("id") or ""),
                    "condition_id": condition_id,
                    "market_slug": market.get("slug") or "",
                    "market_question": market.get("question") or event.get("title") or "",
                    "active": active,
                    "closed": closed,
                    "start_ts_utc": start_ts.isoformat() if pd.notna(start_ts) else "",
                    "current_book_phase": current_phase,
                    "settlement_ts_utc": settlement_ts.isoformat() if pd.notna(settlement_ts) else "",
                    "settlement_date_utc": settlement_ts.date().isoformat() if pd.notna(settlement_ts) else "",
                    "resolution_source": market.get("resolutionSource") or event.get("resolutionSource") or "",
                    "description_snippet": (market.get("description") or event.get("description") or "")[:700].replace("\n", " "),
                    "gamma_volume_usd": num(market.get("volume") if market.get("volume") is not None else event.get("volume")),
                    "gamma_volume_24h_usd": num(
                        market.get("volume24hr") if market.get("volume24hr") is not None else event.get("volume24hr")
                    ),
                    "gamma_volume_1wk_usd": num(
                        market.get("volume1wk") if market.get("volume1wk") is not None else event.get("volume1wk")
                    ),
                    "gamma_volume_1mo_usd": num(
                        market.get("volume1mo") if market.get("volume1mo") is not None else event.get("volume1mo")
                    ),
                    "liquidity_usd": num(market.get("liquidity") if market.get("liquidity") is not None else event.get("liquidity")),
                    "gamma_best_bid": num(market.get("bestBid")),
                    "gamma_best_ask": num(market.get("bestAsk")),
                    "gamma_spread": num(market.get("spread")),
                    "outcomes": json.dumps(outcomes),
                    "outcome_prices": json.dumps(outcome_prices),
                    "clob_token_ids": json.dumps([str(x) for x in token_ids]),
                    "order_price_min_tick_size": num(market.get("orderPriceMinTickSize")),
                    "order_min_size": num(market.get("orderMinSize")),
                    "maker_base_fee": num(market.get("makerBaseFee")),
                    "taker_base_fee": num(market.get("takerBaseFee")),
                    "fees_enabled": bool(market.get("feesEnabled")) if market.get("feesEnabled") is not None else None,
                    **current_stats,
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(["condition_id", "market_slug"])
    return df.sort_values(["active", "settlement_ts_utc"], ascending=[False, False])


def local_dali_candidates() -> pd.DataFrame:
    path = ANALYSIS / "csv_outputs" / "dali" / "dali_tfi_equity_index_100_candidates.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    out = df[df["slug"].astype(str).str.contains(r"spx-opens-up-or-down", case=False, na=False)].copy()
    if out.empty:
        return out
    out["condition_id"] = out["condition_id"].astype(str).str.lower()
    out["local_end_ts"] = pd.to_datetime(out["end_ts"], utc=True, errors="coerce")
    return out


def recent_trade_paths() -> list[str]:
    paths: list[str] = []
    for path in sorted(TRADES_DIR.glob("*.parquet")):
        dates = []
        for y, m, d in DATE_RE.findall(path.name):
            try:
                dates.append(pd.Timestamp(f"{y}-{m}-{d}", tz="UTC"))
            except Exception:
                continue
        if dates and max(dates) >= RECENT_CUTOFF - pd.Timedelta(days=2):
            paths.append(str(path))
    return paths


def parquet_list(paths: list[str]) -> str:
    return "[" + ", ".join(f"'{p.replace(chr(39), chr(39) + chr(39))}'" for p in paths) + "]"


def sql_list(values: list[str] | set[str]) -> str:
    vals = sorted({str(v).lower().replace("'", "''") for v in values if str(v)})
    return ", ".join(f"'{v}'" for v in vals) if vals else "''"


def raw_trade_concentration(condition_ids: list[str]) -> pd.DataFrame:
    paths = recent_trade_paths()
    if not condition_ids or not paths:
        return pd.DataFrame()
    con = duckdb.connect()
    temp_dir = ANALYSIS / ".duckdb_tmp_od_spx_open_scope"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    ids = sql_list(condition_ids)
    return con.execute(
        f"""
        WITH scoped AS (
            SELECT
                lower(CAST(condition_id AS VARCHAR)) AS condition_id,
                lower(CAST(maker AS VARCHAR)) AS maker,
                sum(usd_amount) AS maker_usd,
                count(*) AS maker_fills
            FROM read_parquet({parquet_list(paths)})
            WHERE lower(CAST(condition_id AS VARCHAR)) IN ({ids})
              AND maker IS NOT NULL
              AND taker IS NOT NULL
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


def merge_local(markets: pd.DataFrame, local: pd.DataFrame, conc: pd.DataFrame) -> pd.DataFrame:
    if markets.empty:
        markets = pd.DataFrame(columns=["condition_id"])
    out = markets.copy()
    if not local.empty:
        local_cols = [
            "condition_id",
            "n_fills",
            "usd_volume",
            "local_end_ts",
            "first_fill_ts",
            "last_fill_ts",
            "active_fill_days",
            "midband_pct",
            "gamma_volume",
        ]
        out = out.merge(local[local_cols], on="condition_id", how="outer", suffixes=("", "_local"))
        for col in ["event_slug", "market_slug"]:
            if col not in out:
                out[col] = ""
        slug_map = local.set_index("condition_id")["slug"].to_dict()
        question_map = local.set_index("condition_id")["question"].to_dict()
        out["market_slug"] = out["market_slug"].fillna(out["condition_id"].map(slug_map))
        out["event_slug"] = out["event_slug"].fillna(out["condition_id"].map(slug_map))
        out["market_question"] = out.get("market_question", pd.Series(index=out.index, dtype=object)).fillna(
            out["condition_id"].map(question_map)
        )
    else:
        for col in ["n_fills", "usd_volume", "first_fill_ts", "last_fill_ts", "active_fill_days", "midband_pct", "gamma_volume"]:
            out[col] = np.nan
    if not conc.empty:
        out = out.merge(conc, on="condition_id", how="left")
    return out


def scan_local_clob(markets: pd.DataFrame) -> pd.DataFrame:
    cond_to_slug: dict[str, str] = {}
    token_to_cond: dict[str, str] = {}
    slug_to_cond: dict[str, str] = {}
    for _, row in markets.iterrows():
        cond = str(row.get("condition_id") or "").lower()
        if not cond or cond == "nan":
            continue
        slug = str(row.get("market_slug") or row.get("event_slug") or "")
        cond_to_slug[cond] = slug
        if slug:
            slug_to_cond[slug.lower()] = cond
        for token in parse_json_list(row.get("clob_token_ids")):
            if token:
                token_to_cond[str(token)] = cond
    rows: dict[str, dict[str, Any]] = {
        cond: {
            "condition_id": cond,
            "local_clob_matching_lines": 0,
            "local_clob_book_lines": 0,
            "local_clob_new_market_lines": 0,
            "local_clob_price_change_lines": 0,
            "local_clob_first_seen_utc": "",
            "local_clob_last_seen_utc": "",
            "local_clob_example_file": "",
        }
        for cond in cond_to_slug
    }
    if not cond_to_slug or not LIVE_CLOB_DIR.exists():
        return pd.DataFrame(rows.values())

    try:
        rg = subprocess.run(
            ["rg", "--files-with-matches", "-i", "spx-opens-up-or-down", str(LIVE_CLOB_DIR)],
            check=False,
            capture_output=True,
            text=True,
        )
        files = [Path(p) for p in rg.stdout.splitlines() if p.endswith(".jsonl")]
    except Exception:
        files = sorted(LIVE_CLOB_DIR.rglob("*.jsonl"))

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    lower = line.lower()
                    if (
                        "spx-opens-up-or-down" not in lower
                        and not any(cond in lower for cond in cond_to_slug)
                        and not any(token in line for token in token_to_cond)
                    ):
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        obj = {}
                    event_type = str(obj.get("event_type") or obj.get("message", {}).get("event_type") or "").lower()
                    received_at = str(obj.get("received_at") or "")
                    message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
                    conds: set[str] = set()
                    for key in ["condition_id", "market"]:
                        value = str(message.get(key) or "").lower()
                        if value in cond_to_slug:
                            conds.add(value)
                    for token in (obj.get("asset_ids") or []) + ([message.get("asset_id")] if message.get("asset_id") else []):
                        cond = token_to_cond.get(str(token))
                        if cond:
                            conds.add(cond)
                    slug_values = [
                        message.get("slug"),
                        message.get("event_message", {}).get("slug") if isinstance(message.get("event_message"), dict) else None,
                    ]
                    for slug in slug_values:
                        cond = slug_to_cond.get(str(slug or "").lower())
                        if cond:
                            conds.add(cond)
                    if event_type == "new_market":
                        cond = str(message.get("condition_id") or message.get("market") or "").lower()
                        if cond in cond_to_slug:
                            conds.add(cond)
                            for token in message.get("clob_token_ids") or message.get("assets_ids") or []:
                                token_to_cond[str(token)] = cond
                    if not conds and "spx-opens-up-or-down" in lower:
                        for slug, cond in slug_to_cond.items():
                            if slug and slug in lower:
                                conds.add(cond)
                    for cond in conds:
                        row = rows[cond]
                        row["local_clob_matching_lines"] += 1
                        if event_type == "book":
                            row["local_clob_book_lines"] += 1
                        elif event_type == "new_market":
                            row["local_clob_new_market_lines"] += 1
                        elif event_type == "price_change":
                            row["local_clob_price_change_lines"] += 1
                        if received_at:
                            if not row["local_clob_first_seen_utc"] or received_at < row["local_clob_first_seen_utc"]:
                                row["local_clob_first_seen_utc"] = received_at
                            if not row["local_clob_last_seen_utc"] or received_at > row["local_clob_last_seen_utc"]:
                                row["local_clob_last_seen_utc"] = received_at
                        if not row["local_clob_example_file"]:
                            row["local_clob_example_file"] = str(path.relative_to(ROOT))
        except UnicodeDecodeError:
            continue
    return pd.DataFrame(rows.values())


def summarize(markets: pd.DataFrame, clob_scan: pd.DataFrame, series: dict[str, Any]) -> pd.DataFrame:
    local = markets[markets["n_fills"].notna()].copy() if "n_fills" in markets else pd.DataFrame()
    live = markets[markets.get("active", False).fillna(False)] if "active" in markets else pd.DataFrame()
    event_dates = pd.to_datetime(markets.get("settlement_date_utc", pd.Series(dtype=str)), errors="coerce").dropna()
    local_dates = pd.to_datetime(local.get("local_end_ts", pd.Series(dtype=str)), errors="coerce").dropna()
    if event_dates.empty and not local_dates.empty:
        event_dates = local_dates
    gaps = event_dates.sort_values().diff().dt.days.dropna().to_numpy(float)
    weights = markets["raw_trade_maker_usd"].fillna(markets["usd_volume"]).fillna(markets["gamma_volume_usd"]).fillna(1.0)
    non_top3 = markets["non_top3_maker_share"].fillna(0.30)
    top3 = markets["top3_maker_share"].fillna(0.70)
    non_top3_share = float(np.average(non_top3, weights=np.maximum(weights.to_numpy(float), 1.0))) if len(markets) else math.nan
    top3_share = float(np.average(top3, weights=np.maximum(weights.to_numpy(float), 1.0))) if len(markets) else math.nan
    live_24h = float(live["gamma_volume_24h_usd"].fillna(0).sum()) if not live.empty and "gamma_volume_24h_usd" in live else 0.0
    recent_volume = float(markets["gamma_volume_usd"].fillna(markets["usd_volume"]).fillna(0).sum()) if len(markets) else 0.0
    local_volume = float(local["usd_volume"].fillna(0).sum()) if not local.empty else 0.0
    run_rate = live_24h if live_24h > 0 else recent_volume / RECENT_DAYS
    headroom = run_rate * non_top3_share if np.isfinite(non_top3_share) else math.nan
    book_lines = int(clob_scan["local_clob_book_lines"].sum()) if not clob_scan.empty else 0
    new_market_lines = int(clob_scan["local_clob_new_market_lines"].sum()) if not clob_scan.empty else 0
    current_spread = (
        float(live["current_clob_min_tick_spread"].dropna().min())
        if not live.empty and "current_clob_min_tick_spread" in live and live["current_clob_min_tick_spread"].dropna().any()
        else math.nan
    )
    current_ask_depth = (
        float(live["current_clob_best_ask_depth_usd"].dropna().max())
        if not live.empty and "current_clob_best_ask_depth_usd" in live and live["current_clob_best_ask_depth_usd"].dropna().any()
        else math.nan
    )
    current_bid_depth = (
        float(live["current_clob_best_bid_depth_usd"].dropna().max())
        if not live.empty and "current_clob_best_bid_depth_usd" in live and live["current_clob_best_bid_depth_usd"].dropna().any()
        else math.nan
    )
    return pd.DataFrame(
        [
            {
                "as_of_utc": AS_OF.isoformat(),
                "recent_cutoff_utc": RECENT_CUTOFF.isoformat(),
                "series_slug": SERIES_SLUG,
                "series_id": series.get("id", ""),
                "series_recurrence": series.get("recurrence", ""),
                "series_active": bool(series.get("active")) if series else None,
                "gamma_market_rows": int(markets["condition_id"].nunique()) if len(markets) else 0,
                "gamma_live_market_rows": int(live["condition_id"].nunique()) if not live.empty else 0,
                "local_candidate_rows": int(local["condition_id"].nunique()) if not local.empty else 0,
                "recent_event_dates": int(event_dates.dt.date.nunique()) if len(event_dates) else 0,
                "median_event_gap_days": float(np.nanmedian(gaps)) if len(gaps) else math.nan,
                "live_24h_volume_usd": live_24h,
                "recent_gamma_or_local_volume_usd": recent_volume,
                "local_candidate_volume_usd": local_volume,
                "local_candidate_fills": int(local["n_fills"].fillna(0).sum()) if not local.empty else 0,
                "median_local_event_volume_usd": float(local["usd_volume"].median()) if not local.empty else math.nan,
                "max_local_event_volume_usd": float(local["usd_volume"].max()) if not local.empty else math.nan,
                "weighted_top3_maker_share": top3_share,
                "weighted_non_top3_maker_share": non_top3_share,
                "non_top3_headroom_usd_per_day": headroom,
                "current_clob_min_tick_spread_cents": current_spread * 100 if np.isfinite(current_spread) else math.nan,
                "current_clob_best_ask_depth_usd": current_ask_depth,
                "current_clob_best_bid_depth_usd": current_bid_depth,
                "current_book_phase": ";".join(sorted(set(live["current_book_phase"].dropna().astype(str)))) if not live.empty else "",
                "local_clob_book_lines": book_lines,
                "local_clob_new_market_lines": new_market_lines,
                "has_replayable_historical_clob_books": book_lines > 0,
                "current_scope_verdict": "CLEARS_CURRENT_SCOPE_NEEDS_LIVE_PREOPEN_REPLAY"
                if headroom >= SMALL_CAP_MIN_HEADROOM_USD
                else "CURRENT_SCOPE_THIN_OR_UNPROVEN",
                "pricing_gate_status": "BLOCKED_FOR_MARKET_DATE_CI_WITHOUT_PREOPEN_CLOB_AND_ES_SNAPSHOTS",
            }
        ]
    )


def main() -> int:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HTTP_HEADERS) as client:
        series = resolve_series(client)
        events = fetch_series_events(client)
        markets = flatten_events(events, client)
    local = local_dali_candidates()
    if not local.empty:
        local["local_end_ts"] = pd.to_datetime(local["end_ts"], utc=True, errors="coerce")
    condition_ids = sorted(
        set(markets.get("condition_id", pd.Series(dtype=str)).dropna().astype(str).str.lower().tolist())
        | set(local.get("condition_id", pd.Series(dtype=str)).dropna().astype(str).str.lower().tolist())
    )
    conc = raw_trade_concentration(condition_ids)
    markets = merge_local(markets, local, conc)
    clob_scan = scan_local_clob(markets)
    markets = markets.merge(clob_scan, on="condition_id", how="left")
    summary = summarize(markets, clob_scan, series)

    markets.to_csv(OUT_MARKETS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    clob_scan.to_csv(OUT_CLOB_SCAN, index=False)

    print(f"as_of={AS_OF.isoformat()} recent_cutoff={RECENT_CUTOFF.isoformat()}")
    print(f"wrote {OUT_MARKETS.relative_to(ROOT)} ({len(markets)} rows)")
    print(f"wrote {OUT_SUMMARY.relative_to(ROOT)} ({len(summary)} rows)")
    print(f"wrote {OUT_CLOB_SCAN.relative_to(ROOT)} ({len(clob_scan)} rows)")
    cols = [
        "series_slug",
        "gamma_market_rows",
        "gamma_live_market_rows",
        "local_candidate_rows",
        "recent_event_dates",
        "live_24h_volume_usd",
        "local_candidate_volume_usd",
        "weighted_top3_maker_share",
        "non_top3_headroom_usd_per_day",
        "current_clob_min_tick_spread_cents",
        "local_clob_book_lines",
        "current_scope_verdict",
        "pricing_gate_status",
    ]
    print(summary[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
