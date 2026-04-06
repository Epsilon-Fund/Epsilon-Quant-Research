"""
Database health check — prints a complete overview of the research database.
Shows table counts, collection status, top whales, and strategy previews.
"""

import os
import sys

# Add project root to path so imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import config


def main():
    if not os.path.exists(config.DB_PATH):
        print("Database not found. Run 'python run_collection.py' first.")
        return

    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row

    # === Table counts ===
    events = conn.execute("SELECT COUNT(*) as c FROM events").fetchone()["c"]
    markets = conn.execute("SELECT COUNT(*) as c FROM markets").fetchone()["c"]
    resolved = conn.execute("SELECT COUNT(*) as c FROM markets WHERE closed = 1").fetchone()["c"]
    open_markets = conn.execute("SELECT COUNT(*) as c FROM markets WHERE closed = 0").fetchone()["c"]
    trades = conn.execute("SELECT COUNT(*) as c FROM trades").fetchone()["c"]
    wallets = conn.execute("SELECT COUNT(*) as c FROM wallet_profiles").fetchone()["c"]

    candles_1d = conn.execute("SELECT COUNT(*) as c FROM candles_1d").fetchone()["c"]
    candles_1d_markets = conn.execute("SELECT COUNT(DISTINCT condition_id) as c FROM candles_1d").fetchone()["c"]
    candles_1h = conn.execute("SELECT COUNT(*) as c FROM candles_1h").fetchone()["c"]
    candles_1h_markets = conn.execute("SELECT COUNT(DISTINCT condition_id) as c FROM candles_1h").fetchone()["c"]

    leaderboard = conn.execute("SELECT COUNT(*) as c FROM falcon_leaderboard").fetchone()["c"]
    last_snapshot = conn.execute(
        "SELECT MAX(snapshot_date) as d FROM falcon_leaderboard"
    ).fetchone()["d"]
    wallet_lifetime = conn.execute("SELECT COUNT(*) as c FROM wallet_lifetime").fetchone()["c"]
    pnl_series = conn.execute("SELECT COUNT(*) as c FROM wallet_pnl_series").fetchone()["c"]
    pnl_wallets = conn.execute("SELECT COUNT(DISTINCT proxy_wallet) as c FROM wallet_pnl_series").fetchone()["c"]

    # Get dominant category
    cat_row = conn.execute(
        "SELECT category, COUNT(*) as c FROM events GROUP BY category ORDER BY c DESC LIMIT 1"
    ).fetchone()
    category = cat_row["category"] if cat_row else "none"

    print("\n=== Polymarket Research Database ===\n")
    print(f"Events:          {events:>6}   ({category})")
    print(f"Markets:         {markets:>6}   ({resolved} resolved, {open_markets} open)")
    print(f"Trades:          {trades:>6,}")
    print(f"Wallets (360):   {wallets:>6}")
    print()
    print(f"Candles (daily):   {candles_1d:>8,}  rows across {candles_1d_markets} markets")
    print(f"Candles (hourly):  {candles_1h:>8,}  rows across {candles_1h_markets} markets")
    print()
    snapshot_str = f" (last snapshot: {last_snapshot})" if last_snapshot else ""
    print(f"Leaderboard:       {leaderboard:>8,}  whale candidates{snapshot_str}")
    print(f"Wallet lifetime:   {wallet_lifetime:>8,}  wallets profiled")
    print(f"Wallet PnL series: {pnl_series:>8,}  daily data points across {pnl_wallets} wallets")

    # === Collection log ===
    print("\n=== Collection Log ===")

    # Markets
    markets_log = conn.execute(
        "SELECT status, completed_at FROM collection_log WHERE collection_type = 'markets' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if markets_log:
        print(f"Markets:        {markets_log['status']}  ({markets_log['completed_at']})")
    else:
        print("Markets:        not started")

    # Candles
    candles_1d_done = conn.execute(
        "SELECT COUNT(*) as c FROM collection_log WHERE collection_type = 'candles_1d' AND status = 'done'"
    ).fetchone()["c"]
    candles_1h_done = conn.execute(
        "SELECT COUNT(*) as c FROM collection_log WHERE collection_type = 'candles_1h' AND status = 'done'"
    ).fetchone()["c"]
    total_tokens = conn.execute(
        """SELECT COUNT(*) as c FROM (
            SELECT side_a_token_id FROM markets WHERE side_a_token_id IS NOT NULL
            UNION ALL
            SELECT side_b_token_id FROM markets WHERE side_b_token_id IS NOT NULL
        )"""
    ).fetchone()["c"]
    print(f"Candles (1d):   done  {candles_1d_done}/{total_tokens} tokens")
    print(f"Candles (1h):   done  {candles_1h_done}/{total_tokens} tokens")

    # Trades
    trades_done = conn.execute(
        "SELECT COUNT(*) as c FROM collection_log WHERE collection_type = 'trades' AND status = 'done'"
    ).fetchone()["c"]
    trades_error = conn.execute(
        "SELECT COUNT(*) as c FROM collection_log WHERE collection_type = 'trades' AND status = 'error'"
    ).fetchone()["c"]
    trades_partial = conn.execute(
        "SELECT COUNT(*) as c FROM collection_log WHERE collection_type = 'trades' AND status = 'partial'"
    ).fetchone()["c"]
    print(f"Trades:         done  {trades_done}/{markets} markets complete", end="")
    if trades_error:
        print(f", {trades_error} errors", end="")
    if trades_partial:
        print(f", {trades_partial} in-progress", end="")
    print()

    # Leaderboard
    lb_log = conn.execute(
        "SELECT status, completed_at FROM collection_log WHERE collection_type = 'leaderboard' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if lb_log:
        print(f"Leaderboard:    {lb_log['status']}  ({lb_log['completed_at']})")
    else:
        print("Leaderboard:    not started")

    # Wallet lifetime
    wl_log = conn.execute(
        "SELECT status, records_fetched FROM collection_log WHERE collection_type = 'wallet_lifetime' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    lb_wallets = conn.execute("SELECT COUNT(DISTINCT wallet) as c FROM falcon_leaderboard").fetchone()["c"]
    if wl_log:
        print(f"Wallet lifetime: {wl_log['status']}  {wallet_lifetime}/{lb_wallets} profiled")
    else:
        print(f"Wallet lifetime: not started ({lb_wallets} wallets on leaderboard)")

    # Wallet PnL series
    ps_log = conn.execute(
        "SELECT status, records_fetched FROM collection_log WHERE collection_type = 'wallet_pnl_series' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if ps_log:
        print(f"PnL series:     {ps_log['status']}  {pnl_wallets}/{lb_wallets} wallets")
    else:
        print(f"PnL series:     not started")

    # Wallets 360
    wallets_log = conn.execute(
        "SELECT status, records_fetched, completed_at FROM collection_log WHERE collection_type = 'wallets' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    distinct_wallets = conn.execute(
        "SELECT COUNT(DISTINCT proxy_wallet) as c FROM trades WHERE proxy_wallet != ''"
    ).fetchone()["c"]
    if wallets_log:
        print(f"Wallets (360):  {wallets_log['status']}  {wallets}/{distinct_wallets} profiled")
    else:
        print(f"Wallets (360):  not started ({distinct_wallets} unique wallets in trades)")

    # === Strategy B — Tail Candidates ===
    print("\n=== Strategy B — Tail Candidates (price < 5%, volume > $5,000) ===")
    tails = conn.execute(
        """
        SELECT m.slug, t.price as last_price, m.volume_total
        FROM markets m
        JOIN trades t ON t.slug = m.slug
        WHERE t.price < 0.05
          AND t.price > 0
          AND m.volume_total > 5000
          AND m.closed = 0
        GROUP BY m.slug
        HAVING last_price = MIN(t.price)
        ORDER BY m.volume_total DESC
        LIMIT 10
        """
    ).fetchall()

    if tails:
        for t in tails:
            price_str = f"{t['last_price']:.3f}"
            vol = f"${t['volume_total']:,.0f}" if t["volume_total"] else "N/A"
            print(f"  {t['slug']:<50} last_price={price_str:<8} vol={vol}")
    else:
        print("  No matching markets found (need trades data for open markets).")

    # === Strategy A — Top Whale Candidates (H-Score) ===
    print("\n=== Strategy A — Top Whale Candidates (H-Score ranked) ===")
    whales_hscore = conn.execute(
        """
        SELECT wallet, h_score, total_pnl_15d, win_rate_pct_15d, trajectory
        FROM falcon_leaderboard
        WHERE h_score IS NOT NULL
        ORDER BY h_score DESC
        LIMIT 10
        """
    ).fetchall()

    if whales_hscore:
        print(f"  {'Rank':<6}{'Wallet':<18}{'H-Score':>8}{'15d PnL':>12}{'Win Rate':>10}{'Trajectory':<14}")
        for i, w in enumerate(whales_hscore, 1):
            addr = w["wallet"]
            short_addr = f"{addr[:5]}...{addr[-3:]}" if len(addr) > 10 else addr
            pnl = f"${w['total_pnl_15d']:,.0f}" if w["total_pnl_15d"] else "N/A"
            wr = f"{w['win_rate_pct_15d']:.1f}%" if w["win_rate_pct_15d"] else "N/A"
            traj = w["trajectory"] or "N/A"
            print(f"  {i:<6}{short_addr:<18}{w['h_score']:>8.1f}{pnl:>12}{wr:>10}  {traj}")
    else:
        print("  No leaderboard data yet. Run --leaderboard first.")

    # === Top 10 Weather Whales (Wallet 360) ===
    print("\n=== Top 10 Weather Whales (Wallet 360) ===")
    whales = conn.execute(
        """
        SELECT proxy_wallet, weather_pnl, win_rate, sharpe_ratio, edge_decay, sybil_risk_flag
        FROM wallet_profiles
        WHERE weather_pnl IS NOT NULL AND weather_pnl > 0
        ORDER BY weather_pnl DESC
        LIMIT 10
        """
    ).fetchall()

    if whales:
        print(f"  {'Rank':<6}{'Wallet':<18}{'Weather PnL':>12}{'Win Rate':>10}{'Sharpe':>8}{'Edge Decay':>12}{'Sybil':>7}")
        for i, w in enumerate(whales, 1):
            addr = w["proxy_wallet"]
            short_addr = f"{addr[:5]}...{addr[-3:]}" if len(addr) > 10 else addr
            wr = f"{w['win_rate']*100:.1f}%" if w["win_rate"] else "N/A"
            sharpe = f"{w['sharpe_ratio']:.2f}" if w["sharpe_ratio"] else "N/A"
            ed = f"{w['edge_decay']:.2f}" if w["edge_decay"] else "N/A"
            sybil = "Yes" if w["sybil_risk_flag"] else "No"
            print(f"  {i:<6}{short_addr:<18}${w['weather_pnl']:>10,.0f}{wr:>10}{sharpe:>8}{ed:>12}{sybil:>7}")
    else:
        print("  No wallet profiles with weather PnL yet.")

    print()
    conn.close()


if __name__ == "__main__":
    main()
