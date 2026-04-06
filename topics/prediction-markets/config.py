"""
Central configuration for the Polymarket research pipeline.
All settings, API keys, and tunable parameters live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Falcon API
FALCON_API_KEY = os.getenv("FALCON_API_KEY")
FALCON_BASE_URL = "https://narrative.agent.heisenberg.so/api/v2/semantic/retrieve/parameterized"

# Agent IDs
AGENT_MARKETS = 574
AGENT_TRADES = 556
AGENT_CANDLESTICKS = 568
AGENT_ORDERBOOK = 572
AGENT_TOP_TRADERS = 579
AGENT_WALLET_360 = 581
AGENT_LEADERBOARD = 584
AGENT_WALLET_LIFETIME = 586
AGENT_WALLET_PNL_SERIES = 569

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "polymarket.db")

# Collection behaviour
PAGINATION_LIMIT = 100
REQUEST_DELAY_SECONDS = 0.05
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

# Market collection — all weather subcategories
WEATHER_KEYWORDS = {
    "highest-temp": "temperature",       # ~21K markets (the big one)
    "global-temperature": "global",      # ~100 markets
    "precipitation": "precipitation",    # ~120 markets
    "tornado": "tornadoes",              # ~50 markets
    "hurricane": "hurricanes",           # ~80 markets
    "earthquake": "earthquakes",         # ~140 markets
    "volcano": "volcanoes",              # ~2 markets
    "pandemic": "pandemics",             # ~22 markets
}
DEFAULT_CATEGORY = "weather"

# Wallet 360
WALLET_WINDOW_DAYS = "15"

# Leaderboard filters for whale candidates
MIN_WIN_RATE = "0.55"
MIN_TOTAL_PNL = "500"
LEADERBOARD_PERIOD = "30d"
