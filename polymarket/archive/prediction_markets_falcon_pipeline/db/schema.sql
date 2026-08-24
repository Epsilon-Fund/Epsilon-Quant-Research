-- Polymarket Research Database Schema
-- Parent events group multiple binary markets together
-- e.g. "Highest temperature in Shanghai on April 2?" contains 11 YES/NO markets

CREATE TABLE IF NOT EXISTS events (
    event_slug TEXT PRIMARY KEY,
    title TEXT,
    category TEXT DEFAULT 'weather',
    end_date TEXT,
    total_volume REAL,
    num_markets INTEGER,
    fetched_at TEXT DEFAULT (datetime('now'))
);

-- Each individual binary YES/NO market
-- Multiple markets belong to one parent event
CREATE TABLE IF NOT EXISTS markets (
    condition_id TEXT PRIMARY KEY,
    event_slug TEXT REFERENCES events(event_slug),
    slug TEXT UNIQUE NOT NULL,
    question TEXT,
    category TEXT DEFAULT 'weather',
    start_date TEXT,
    end_date TEXT,
    closed INTEGER DEFAULT 0,
    winning_outcome TEXT,
    volume_total REAL,
    side_a_outcome TEXT,
    side_b_outcome TEXT,
    side_a_token_id TEXT,
    side_b_token_id TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

-- Every individual trade on any market
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    condition_id TEXT REFERENCES markets(condition_id),
    event_slug TEXT,
    slug TEXT NOT NULL,
    proxy_wallet TEXT NOT NULL,
    side TEXT,
    outcome TEXT,
    price REAL,
    size REAL,
    timestamp TEXT,
    transaction_hash TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

-- Wallet intelligence profiles from Falcon Wallet 360
CREATE TABLE IF NOT EXISTS wallet_profiles (
    proxy_wallet TEXT PRIMARY KEY,
    win_rate REAL,
    total_pnl REAL,
    roi REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    edge_decay REAL,
    performance_trend TEXT,
    performance_by_category TEXT,
    sybil_risk_flag INTEGER DEFAULT 0,
    suspicious_win_rate_flag INTEGER DEFAULT 0,
    weather_pnl REAL,
    window_days INTEGER,
    last_active TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

-- Daily candlestick data per market token (up to 360 days back)
CREATE TABLE IF NOT EXISTS candles_1d (
    id TEXT PRIMARY KEY,           -- composite: token_id + candle_time
    condition_id TEXT REFERENCES markets(condition_id),
    token_id TEXT NOT NULL,
    outcome TEXT,                  -- "Yes" or "No"
    candle_time TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    mean REAL,
    volume REAL,
    trade_count INTEGER,
    bid_open REAL,
    bid_high REAL,
    bid_low REAL,
    bid_close REAL,
    ask_open REAL,
    ask_high REAL,
    ask_low REAL,
    ask_close REAL,
    fetched_at TEXT DEFAULT (datetime('now'))
);

-- Hourly candlestick data per market token (up to 90 days back)
CREATE TABLE IF NOT EXISTS candles_1h (
    id TEXT PRIMARY KEY,           -- composite: token_id + candle_time
    condition_id TEXT REFERENCES markets(condition_id),
    token_id TEXT NOT NULL,
    outcome TEXT,
    candle_time TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    mean REAL,
    volume REAL,
    trade_count INTEGER,
    bid_open REAL,
    bid_high REAL,
    bid_low REAL,
    bid_close REAL,
    ask_open REAL,
    ask_high REAL,
    ask_low REAL,
    ask_close REAL,
    fetched_at TEXT DEFAULT (datetime('now'))
);

-- Falcon Leaderboard snapshots (H-Score ranked, taken periodically)
CREATE TABLE IF NOT EXISTS falcon_leaderboard (
    id TEXT PRIMARY KEY,           -- composite: wallet + snapshot_date
    wallet TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,   -- date this snapshot was taken
    leaderboard_rank INTEGER,
    tier TEXT,
    h_score REAL,
    roi_pct_15d REAL,
    win_rate_pct_15d REAL,
    sharpe_ratio_15d REAL,
    total_trades_15d INTEGER,
    markets_traded_15d INTEGER,
    total_pnl_15d REAL,
    total_volume_15d REAL,
    trajectory TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

-- Wallet lifetime performance summary (all-time stats, ~9 months available)
CREATE TABLE IF NOT EXISTS wallet_lifetime (
    proxy_wallet TEXT PRIMARY KEY,
    total_trades INTEGER,
    avg_trade_size REAL,
    avg_pnl_per_trade REAL,
    total_invested REAL,
    total_pnl REAL,
    roi_pct REAL,
    last_updated TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

-- Daily PnL time series per wallet (equity curve over time)
CREATE TABLE IF NOT EXISTS wallet_pnl_series (
    id TEXT PRIMARY KEY,           -- composite: proxy_wallet + date
    proxy_wallet TEXT NOT NULL,
    date TEXT NOT NULL,
    invested REAL,
    pnl REAL,
    trades INTEGER,
    wins INTEGER,
    losses INTEGER,
    win_rate REAL,
    fetched_at TEXT DEFAULT (datetime('now'))
);

-- Pipeline checkpoint log — makes collection fully resumable
CREATE TABLE IF NOT EXISTS collection_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_type TEXT NOT NULL,
    target TEXT,
    status TEXT NOT NULL,
    records_fetched INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

-- Indexes for fast backtesting queries
CREATE INDEX IF NOT EXISTS idx_markets_event ON markets(event_slug);
CREATE INDEX IF NOT EXISTS idx_markets_category ON markets(category);
CREATE INDEX IF NOT EXISTS idx_markets_volume ON markets(volume_total);
CREATE INDEX IF NOT EXISTS idx_markets_closed ON markets(closed);
CREATE INDEX IF NOT EXISTS idx_trades_slug ON trades(slug);
CREATE INDEX IF NOT EXISTS idx_trades_wallet ON trades(proxy_wallet);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_price ON trades(price);
CREATE INDEX IF NOT EXISTS idx_trades_event ON trades(event_slug);
CREATE INDEX IF NOT EXISTS idx_wallet_pnl ON wallet_profiles(total_pnl);
CREATE INDEX IF NOT EXISTS idx_wallet_weather ON wallet_profiles(weather_pnl);
CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);
CREATE INDEX IF NOT EXISTS idx_events_end_date ON events(end_date);
CREATE INDEX IF NOT EXISTS idx_candles_1d_condition ON candles_1d(condition_id);
CREATE INDEX IF NOT EXISTS idx_candles_1d_token ON candles_1d(token_id);
CREATE INDEX IF NOT EXISTS idx_candles_1d_time ON candles_1d(candle_time);
CREATE INDEX IF NOT EXISTS idx_candles_1h_condition ON candles_1h(condition_id);
CREATE INDEX IF NOT EXISTS idx_candles_1h_token ON candles_1h(token_id);
CREATE INDEX IF NOT EXISTS idx_candles_1h_time ON candles_1h(candle_time);
CREATE INDEX IF NOT EXISTS idx_leaderboard_wallet ON falcon_leaderboard(wallet);
CREATE INDEX IF NOT EXISTS idx_leaderboard_score ON falcon_leaderboard(h_score);
CREATE INDEX IF NOT EXISTS idx_leaderboard_date ON falcon_leaderboard(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_wallet_lifetime_pnl ON wallet_lifetime(total_pnl);
CREATE INDEX IF NOT EXISTS idx_pnl_series_wallet ON wallet_pnl_series(proxy_wallet);
CREATE INDEX IF NOT EXISTS idx_pnl_series_date ON wallet_pnl_series(date);
