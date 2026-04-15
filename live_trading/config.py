

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

# Coin weights — must sum to 1.0
# Edit here to allocate unequally across coins.
# Coins in ACTIVE_ASSETS but absent from this dict share the remaining weight equally.
COIN_WEIGHTS = {
    "ETHUSDT": 0.25,
    "SOLUSDT": 0.25,
    "XRPUSDT": 0.25,
    "BTCUSDT": 0.25,
}

# ── shared ───────────────────────────────────────────────────────────
ACTIVE_ASSETS = [
    "ETHUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "BTCUSDT",
]

# Add coins here after running optimise.py --asset <symbol>
# Remove coins here to stop the dashboard processing them
# live_params.json entries are preserved even when a coin is removed here
