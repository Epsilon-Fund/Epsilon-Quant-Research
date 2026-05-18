

# Used by optimise.py only — not referenced during live trading
# # ── optimise.py only ─────────────────────────────────────────────────
WF_CONFIG = {
    "train_bars": 1050,
    "test_bars":  137,
    "burnin":     100,
    "n_trials":   400,
    "cost":       0.001,
}
# ── dashboard.py only ────────────────────────────────────────────────
CAPITAL          = 100_000  # total portfolio capital in USD
EXECUTION_HOUR   = 8        # UTC hour for theoretical execution price
INDICATOR_WARMUP = 100      # bars fetched to warm up indicators ie burn in bars

# Round-trip trading cost as a fraction of notional position size.
# Entry cost + exit cost = size_usd * TRADING_COST_PCT * 2
# Set once you know the platform fee schedule (e.g. 0.001 = 0.1% per leg).
TRADING_COST_PCT = 0.001

# Bar frequency the strategy is designed for — drives which OHLC cache the
# theoretical-curve backtest feeds it (see shared/theoretical_curve.py).
DATA_FREQUENCY = 'daily'

# Coin weights — must sum to 1.0
# Edit here to allocate unequally across coins.
# Coins in ACTIVE_ASSETS but absent from this dict share the remaining weight equally.
COIN_WEIGHTS = {
}

# ── shared ───────────────────────────────────────────────────────────
ACTIVE_ASSETS = [
    "ADAUSDT",
    "AVAXUSDT",
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
]

# Add coins here after running optimise.py --asset <symbol>
# Remove coins here to stop the dashboard processing them
# live_params.json entries are preserved even when a coin is removed here
