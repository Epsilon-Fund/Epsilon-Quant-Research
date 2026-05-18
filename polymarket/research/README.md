# polymarket-copy

Historical trade & trader data infrastructure for cohort-based copy trading on
Polymarket. Offline-only: historical data ingestion + trader ranking. No live
trading, no WebSockets, no execution.

See `CLAUDE.md` for goals, architecture, and conventions.

## Setup

```bash
uv sync
cp .env.example .env
```

## Layout

- `data_infra/` — Goldsky + Gamma clients, parquet I/O helpers
- `data/` — parquet shards (gitignored)
- `scripts/` — backfill / refresh entry points
- `notebooks/` — exploration
