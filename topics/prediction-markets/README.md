# Polymarket Prediction Market Research Pipeline

A data pipeline for collecting and analyzing Polymarket prediction market data via the Falcon API. Built to support two trading strategies: whale copy-trading and tail harvesting on weather markets, with a schema general enough to expand to any market category.

## Prerequisites

- Python 3.10+
- A Falcon API key (from Heisenberg / Narrative)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your .env file
cp .env.example .env
# Edit .env and add your FALCON_API_KEY

# 3. Run the pipeline
python run_collection.py
```

## Usage

### Full pipeline (markets → candles → trades → leaderboard → wallets)

```bash
python run_collection.py
```

### Individual steps

```bash
python run_collection.py --markets      # Collect events + markets only
python run_collection.py --candles      # Collect candlesticks only (requires markets)
python run_collection.py --trades       # Collect trades only (requires markets)
python run_collection.py --leaderboard  # Leaderboard + wallet lifetime + wallet PnL series
python run_collection.py --wallets      # Collect Wallet 360 profiles only (requires trades)
```

### Reset database (drops and recreates all tables)

```bash
python run_collection.py --reset
```

### Check progress

```bash
python scripts/check_db.py
```

### Export data to CSV

```bash
python scripts/export_csv.py markets
python scripts/export_csv.py trades
python scripts/export_csv.py wallet_profiles
python scripts/export_csv.py events
```

## Resuming after interruption

The pipeline is fully resumable. Every collection step logs its progress to a `collection_log` table. When you re-run the pipeline, it automatically skips markets whose trades have already been collected and wallets that have already been profiled. Just run the same command again and it picks up where it left off.

## Schema overview

The database has nine main tables connected in a hierarchy:

- **events** — Parent groupings (e.g. "Highest temperature in Shanghai on April 2?"). Each event contains multiple binary markets.
- **markets** — Individual YES/NO binary markets (e.g. "Will the temperature be 23°C?"). Each market belongs to one event and has its own price, volume, and outcome.
- **candles_1d** / **candles_1h** — Daily and hourly OHLCV candlestick data per market token. Used for price history, backtesting, and chart analysis.
- **trades** — Every individual trade placed on any market. Links to both the market (via `condition_id`) and the event (via `event_slug`).
- **falcon_leaderboard** — Periodic snapshots of Falcon's H-Score ranked leaderboard. Used to identify whale candidates.
- **wallet_lifetime** — All-time performance stats per leaderboard wallet (total trades, ROI, invested).
- **wallet_pnl_series** — Daily PnL time series per leaderboard wallet (equity curves over time).
- **wallet_profiles** — Wallet 360 intelligence profiles with win rate, PnL, Sharpe ratio, and per-category performance.

A tenth table, **collection_log**, tracks pipeline progress and makes collection resumable.

## Strategy data requirements

### Strategy A — Whale copy-trading

Needs: `wallet_profiles` (to identify skilled traders by weather PnL, win rate, Sharpe), `trades` (to see what they bought/sold and when), `markets` (to find currently open markets to trade).

### Strategy B — Tail harvesting

Needs: `markets` (to find multi-outcome events where some outcomes have very low prices), `events` (to group markets by parent event and compare probabilities), `trades` (to check volume and liquidity on low-probability outcomes).

## On-demand analysis tools

The `collectors/orderbook.py` module contains functions to fetch orderbook snapshots around specific trades for slippage and price impact analysis. This is **not** part of the main collection pipeline — it is meant to be called manually during the analysis phase (e.g. from a notebook) when you want to evaluate how much a copy-trade would move the market.

```python
from collectors.orderbook import fetch_orderbook_around_trade
snapshots = fetch_orderbook_around_trade(token_id, "2026-04-01T14:32:00", window_minutes=30)
```
