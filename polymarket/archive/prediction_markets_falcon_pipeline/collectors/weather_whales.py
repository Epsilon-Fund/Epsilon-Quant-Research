"""
Weather whale discovery — scans the global Falcon leaderboard and identifies
wallets that primarily trade weather markets (< 100 non-weather trades).
Stores their full profiles, lifetime stats, and PnL series.
"""

import base64
import json
import sqlite3
from datetime import date, datetime

from loguru import logger

import config
from collectors.falcon_client import FalconClient


def _decode_perf_by_category(raw_value) -> list[dict]:
    """Decode the performance_by_category field (base64 JSON or plain JSON)."""
    if not raw_value:
        return []

    # Try base64 decode first
    if isinstance(raw_value, str):
        try:
            decoded = base64.b64decode(raw_value).decode("utf-8")
            return json.loads(decoded)
        except Exception:
            pass
        # Try plain JSON
        try:
            return json.loads(raw_value)
        except Exception:
            return []

    if isinstance(raw_value, list):
        return raw_value

    return []


def _count_weather_trades(perf_categories: list[dict]) -> tuple[int, int, float]:
    """
    Count weather and non-weather trades from the category breakdown.

    Returns:
        (weather_trades, non_weather_trades, weather_pnl)
    """
    weather_trades = 0
    weather_pnl = 0.0
    total_trades = 0

    for cat in perf_categories:
        trades = cat.get("trades", 0) or 0
        pnl = cat.get("pnl", 0.0) or 0.0
        category = cat.get("category", "")

        total_trades += trades
        if category == "Weather":
            weather_trades = trades
            weather_pnl = pnl

    non_weather = total_trades - weather_trades
    return weather_trades, non_weather, weather_pnl


def discover_weather_whales(
    db_conn: sqlite3.Connection,
    max_non_weather_trades: int = 100,
) -> int:
    """
    Scan the global leaderboard and filter for weather-focused wallets.

    For each wallet on the leaderboard:
    1. Fetch Wallet 360 profile
    2. Decode performance_by_category
    3. If non-weather trades < max_non_weather_trades, keep it
    4. Store in falcon_leaderboard + wallet_profiles

    Args:
        db_conn: Active SQLite connection.
        max_non_weather_trades: Maximum non-weather trades to qualify (default 100).

    Returns:
        Number of weather whales found.
    """
    client = FalconClient()
    snapshot_date = date.today().isoformat()

    print(f"\n--- Discovering weather whales (max {max_non_weather_trades} non-weather trades) ---")

    # Step 1: Fetch full leaderboard
    print("  Fetching global leaderboard...")
    leaderboard = client.query(
        config.AGENT_LEADERBOARD,
        params={
            "min_win_rate_15d": "0.45",
            "max_win_rate_15d": "0.95",
            "min_roi_15d": "0",
            "min_total_trades_15d": "20",
            "max_total_trades_15d": "100000",
            "min_pnl_15d": "100",
            "sort_by": "h_score",
        },
        paginate=True,
    )
    print(f"  Leaderboard size: {len(leaderboard):,} wallets")

    # Store leaderboard snapshot
    for r in leaderboard:
        wallet = r.get("wallet", r.get("proxy_wallet", ""))
        if not wallet:
            continue
        composite_id = f"{wallet}_{snapshot_date}"
        db_conn.execute(
            """
            INSERT OR IGNORE INTO falcon_leaderboard
                (id, wallet, snapshot_date, leaderboard_rank, tier, h_score,
                 roi_pct_15d, win_rate_pct_15d, sharpe_ratio_15d,
                 total_trades_15d, markets_traded_15d, total_pnl_15d,
                 total_volume_15d, trajectory)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                composite_id, wallet, snapshot_date,
                r.get("leaderboard_rank", r.get("rank")),
                r.get("tier"), r.get("h_score"),
                r.get("roi_pct_15d", r.get("roi_pct")),
                r.get("win_rate_pct_15d", r.get("win_rate_pct")),
                r.get("sharpe_ratio_15d", r.get("sharpe_ratio")),
                r.get("total_trades_15d", r.get("total_trades")),
                r.get("markets_traded_15d", r.get("markets_traded")),
                r.get("total_pnl_15d", r.get("total_pnl")),
                r.get("total_volume_15d", r.get("total_volume")),
                r.get("trajectory"),
            ),
        )
    db_conn.commit()
    print(f"  Leaderboard snapshot saved")

    # Step 2: Check already-screened wallets to avoid re-fetching
    already_screened = {
        r[0] for r in db_conn.execute(
            "SELECT target FROM collection_log WHERE collection_type = 'whale_screen'"
        ).fetchall()
    }

    wallets_to_screen = [
        r.get("wallet", r.get("proxy_wallet", ""))
        for r in leaderboard
        if r.get("wallet", r.get("proxy_wallet", ""))
        and r.get("wallet", r.get("proxy_wallet", "")) not in already_screened
    ]

    print(f"  Already screened: {len(already_screened):,}, to screen: {len(wallets_to_screen):,}")

    # Step 3: Screen each wallet via Wallet 360
    whales_found = 0
    screened = 0

    for i, wallet in enumerate(wallets_to_screen, 1):
        try:
            profiles = client.query(
                config.AGENT_WALLET_360,
                params={"proxy_wallet": wallet, "window_days": config.WALLET_WINDOW_DAYS},
                paginate=False,
            )

            if not profiles:
                # Log as screened even if no data
                db_conn.execute(
                    "INSERT INTO collection_log (collection_type, target, status, records_fetched, started_at, completed_at) VALUES (?, ?, ?, ?, ?, ?)",
                    ("whale_screen", wallet, "done", 0, datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
                )
                if i % 100 == 0:
                    db_conn.commit()
                continue

            p = profiles[0]
            perf_raw = p.get("performance_by_category", "")
            perf_cats = _decode_perf_by_category(perf_raw)
            weather_trades, non_weather, weather_pnl = _count_weather_trades(perf_cats)

            # Apply filter
            is_weather_whale = non_weather < max_non_weather_trades and weather_trades > 0

            if is_weather_whale:
                whales_found += 1

                # Encode perf_by_category as storable JSON
                perf_by_cat_str = json.dumps(perf_cats) if perf_cats else perf_raw
                perf_trend = p.get("performance_trend")
                if isinstance(perf_trend, (dict, list)):
                    perf_trend = json.dumps(perf_trend)

                # Store full Wallet 360 profile
                db_conn.execute(
                    """
                    INSERT OR REPLACE INTO wallet_profiles
                        (proxy_wallet, win_rate, total_pnl, roi, sharpe_ratio, sortino_ratio,
                         max_drawdown, edge_decay, performance_trend, performance_by_category,
                         sybil_risk_flag, suspicious_win_rate_flag, weather_pnl,
                         window_days, last_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        wallet,
                        p.get("win_rate"), p.get("total_pnl"), p.get("roi"),
                        p.get("sharpe_ratio"), p.get("sortino_ratio"),
                        p.get("max_drawdown"), p.get("edge_decay"),
                        perf_trend, perf_by_cat_str,
                        1 if p.get("sybil_risk_flag") else 0,
                        1 if p.get("suspicious_win_rate_flag") else 0,
                        weather_pnl,
                        int(config.WALLET_WINDOW_DAYS),
                        p.get("last_active"),
                    ),
                )

                print(
                    f"  WHALE #{whales_found}: {wallet[:12]}... "
                    f"weather={weather_trades} trades, pnl=${weather_pnl:,.0f}, "
                    f"non-weather={non_weather}"
                )

            # Log as screened
            db_conn.execute(
                "INSERT INTO collection_log (collection_type, target, status, records_fetched, started_at, completed_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("whale_screen", wallet, "done", 1 if is_weather_whale else 0,
                 datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
            )

            screened += 1
            if i % 25 == 0:
                db_conn.commit()
                pct = i / len(wallets_to_screen) * 100
                print(f"  Screened: {i:,}/{len(wallets_to_screen):,} ({pct:.1f}%) — {whales_found} whales found so far")

        except Exception as e:
            logger.error("Error screening wallet {}: {}", wallet, e)

    db_conn.commit()

    print(f"\n  === Screening Complete ===")
    print(f"  Wallets screened:     {screened:,}")
    print(f"  Weather whales found: {whales_found}")

    # Log completion
    db_conn.execute(
        "INSERT INTO collection_log (collection_type, target, status, records_fetched, started_at, completed_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("weather_whales", snapshot_date, "done", whales_found,
         datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
    )
    db_conn.commit()

    return whales_found


def enrich_weather_whales(db_conn: sqlite3.Connection) -> None:
    """
    Fetch wallet_lifetime and wallet_pnl_series for all discovered weather whales.
    Only processes wallets already in wallet_profiles (i.e., confirmed weather whales).
    """
    client = FalconClient()

    whales = [
        r[0] for r in db_conn.execute(
            "SELECT proxy_wallet FROM wallet_profiles WHERE weather_pnl > 0"
        ).fetchall()
    ]

    print(f"\n--- Enriching {len(whales)} weather whales ---")

    # Wallet lifetime
    existing_lifetime = {
        r[0] for r in db_conn.execute("SELECT proxy_wallet FROM wallet_lifetime").fetchall()
    }
    pending_lt = [w for w in whales if w not in existing_lifetime]

    if pending_lt:
        print(f"\n  Fetching lifetime stats ({len(pending_lt)} pending)...")
        for i, wallet in enumerate(pending_lt, 1):
            try:
                results = client.query(
                    config.AGENT_WALLET_LIFETIME,
                    params={"wallet_address": wallet},
                    paginate=False,
                )
                if results:
                    p = results[0]
                    db_conn.execute(
                        """
                        INSERT OR REPLACE INTO wallet_lifetime
                            (proxy_wallet, total_trades, avg_trade_size, avg_pnl_per_trade,
                             total_invested, total_pnl, roi_pct, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (wallet, p.get("total_trades"), p.get("avg_trade_size"),
                         p.get("avg_pnl_per_trade"), p.get("total_invested"),
                         p.get("total_pnl"), p.get("roi_pct"), p.get("last_updated")),
                    )
                if i % 25 == 0:
                    db_conn.commit()
                    print(f"    Lifetime: {i}/{len(pending_lt)}")
            except Exception as e:
                logger.error("Error fetching lifetime for {}: {}", wallet, e)
        db_conn.commit()

    # Wallet PnL series
    existing_pnl = {
        r[0] for r in db_conn.execute("SELECT DISTINCT proxy_wallet FROM wallet_pnl_series").fetchall()
    }
    pending_pnl = [w for w in whales if w not in existing_pnl]

    if pending_pnl:
        today_str = date.today().isoformat()
        print(f"\n  Fetching PnL series ({len(pending_pnl)} pending)...")
        for i, wallet in enumerate(pending_pnl, 1):
            try:
                results = client.query(
                    config.AGENT_WALLET_PNL_SERIES,
                    params={
                        "granularity": "1d",
                        "wallet": wallet,
                        "start_time": "2024-01-01",
                        "end_time": today_str,
                    },
                    paginate=False,
                )
                for p in results:
                    d = p.get("date", p.get("time", ""))
                    db_conn.execute(
                        """
                        INSERT OR IGNORE INTO wallet_pnl_series
                            (id, proxy_wallet, date, invested, pnl, trades, wins, losses, win_rate)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (f"{wallet}_{d}", wallet, d, p.get("invested"), p.get("pnl"),
                         p.get("trades"), p.get("wins"), p.get("losses"), p.get("win_rate")),
                    )
                if i % 25 == 0:
                    db_conn.commit()
                    print(f"    PnL series: {i}/{len(pending_pnl)}")
            except Exception as e:
                logger.error("Error fetching PnL series for {}: {}", wallet, e)
        db_conn.commit()

    print("  Enrichment complete.")
