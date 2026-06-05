# Midas
> Hub: [[COWORK]]


Epsilon Fund quantitative trading system for Polymarket.

## Structure

```
midas/
├── executor/       — Structural arb executor (from colleague's repo, unchanged)
├── harvester/      — Tail risk harvester strategy (new)
├── examples/       — Runnable example scripts
├── tests/
│   ├── executor/   — Tests for the executor package
│   └── harvester/  — Tests for the harvester package
└── data/
    ├── journal/    — Order event logs (gitignored)
    └── research/   — Strategy research logs (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Fill in your credentials in .env
```

## Running tests

```bash
pytest tests/executor/ -x -q   # executor tests
pytest tests/harvester/ -x -q  # harvester tests (once built)
```

## Safety

Both strategies default to paper mode (`LIVE_TRADING_ENABLED=false`).
Never set live mode without first validating in paper mode.
