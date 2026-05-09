# ── optimise.py only ─────────────────────────────────────────────────────────
WF_CONFIG = {
    "train_bars": 25_200,   # 1H bars (~3.5 years) — from research notebooks
    "test_bars":   6_600,   # 1H bars (~9 months)
    "burnin":        200,   # 1H bars dropped for indicator warm-up
    "n_trials":      400,
    "cost":          0.001, # round-trip fraction
}

# ── dashboard.py + streamlit_app.py ──────────────────────────────────────────
CAPITAL          = 100_000   # total portfolio capital in USD
EXECUTION_HOUR   = 8         # UTC hour for theoretical execution price
INDICATOR_WARMUP = 200       # hourly bars dropped as indicator burn-in
TRADING_COST_PCT = 0.001     # one-way cost fraction (used for unrealised P&L display)

# Coin weights — must sum to ≤ 1.0
# Coins in ACTIVE_ASSETS but absent from this dict share the remaining weight equally.
COIN_WEIGHTS = {}

# ── shared ────────────────────────────────────────────────────────────────────
ACTIVE_ASSETS = ["BTCUSDT", "ETHUSDT", "AVAXUSDT", "LINKUSDT", "POLUSDT"]
# Add coins here after running:  python optimise.py --asset <SYMBOL>
# Remove coins here to pause the dashboard without losing live_params.json data
