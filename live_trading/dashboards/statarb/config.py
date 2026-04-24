
# Used by optimise.py only — not referenced during live trading
WF_CONFIG = {
    "train_bars": 1050,
    "test_bars":  137,
    "burnin":     100,
    "n_trials":   400,
    "cost":       0.001,
}

# ── dashboard.py / streamlit_app.py ─────────────────────────────────────────
CAPITAL          = 100_000  # total strategy capital in USD
EXECUTION_HOUR   = 8        # UTC hour for theoretical execution price
INDICATOR_WARMUP = 300      # bars fetched — needs max_lookback + max_z_lookback warm-up
TRADING_COST_PCT = 0.001    # 10bps per leg (two legs per spread trade)

# Coin weights — must sum ≤ 1.0
# Empty = equal weight across all ACTIVE_ASSETS
COIN_WEIGHTS = {}

# ── shared ───────────────────────────────────────────────────────────────────
ACTIVE_ASSETS = [
    'FILSNX',
    'ATOMARB',
    'LINKTRX',
    'LTCAPT',
]
