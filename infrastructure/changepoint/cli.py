"""
CLI shim — the engine lives in the extracted library package (library/changepoint/,
`lemma-changepoint`); this keeps the documented epsilon invocation working:

    PYTHONPATH=. .venv/bin/python -m infrastructure.changepoint.cli detect \
        live_trading/cache/daily/BTCUSDT_daily.parquet --column Close --returns \
        --detector bocpd --out infrastructure/changepoint/changepoints/btc_daily_bocpd.parquet

    PYTHONPATH=. .venv/bin/python -m infrastructure.changepoint.cli benchmark
    PYTHONPATH=. .venv/bin/python -m infrastructure.changepoint.cli kappa-demo
"""
from __future__ import annotations

from lemma.changepoint.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
