"""Run the minimal Dali paper backtest on a live CLOB JSONL capture."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from lib.backtest_engine import BacktestEngine, load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_DIR = ROOT / "data" / "live_clob"
DEFAULT_CONFIG = ROOT / "configs" / "backtest_default.yaml"
DEFAULT_OUT_DIR = ROOT / "data" / "backtests"


def default_out(path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUT_DIR / f"{path.stem}_paper_journal_{stamp}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="?", type=Path)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.latest:
        captures = sorted(DEFAULT_IN_DIR.glob("*.jsonl"))
        if not captures:
            raise SystemExit(f"no captures found in {DEFAULT_IN_DIR}")
        path = captures[-1]
    elif args.jsonl:
        path = args.jsonl
    else:
        raise SystemExit("pass a capture JSONL or --latest")

    config = load_config(args.config)
    engine = BacktestEngine(config)
    journal = engine.run_jsonl(path)
    out = args.out or default_out(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    journal.to_csv(out, index=False)
    print(f"input: {path.relative_to(ROOT)}")
    print(f"config: {args.config.relative_to(ROOT)}")
    print(f"journal: {out.relative_to(ROOT)}")
    print(f"closed trades: {len(journal):,}")
    print(f"open positions: {len(engine.positions):,}")
    print(f"skipped orders: {len(engine.skipped_orders):,}")
    if len(journal):
        print(f"net pnl: {journal['net_pnl'].sum():.6f}")
        print(f"exit reasons: {journal['exit_reason'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
