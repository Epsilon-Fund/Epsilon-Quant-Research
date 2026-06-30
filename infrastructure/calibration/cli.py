"""
cli.py — command-line entry for the calibration scoring layer.

Run per project (NEVER cross-import the two books):
  crypto:      PYTHONPATH=. uv run python -m infrastructure.calibration.cli score --book crypto
  polymarket:  cd polymarket/research && PYTHONPATH=. uv run python -m lib.calibration.cli score --book polymarket

The book selects which forked-ledger to read ($SF_BOOK / $SF_LEDGER_DIR also
work). This file is byte-identical across projects.
"""

from __future__ import annotations

import argparse
import json
import sys

from .core import (
    calibration_table,
    load_scored_forecasts,
    reliability_diagram,
    score_ledger,
)


def _fmt_scorecard(s: dict) -> str:
    m = s["murphy"]
    sp = s["spiegelhalter"]
    c = s["calibration_in_the_large"]
    lines = [
        f"n scored forecasts : {s['n']}",
        f"Brier              : {s['brier']:.4f}",
        f"  reliability      : {m['reliability']:.4f}  (lower = better calibrated)",
        f"  resolution       : {m['resolution']:.4f}  (higher = more discriminating)",
        f"  uncertainty      : {m['uncertainty']:.4f}  (base rate {m['base_rate']:.3f})",
        f"  REL-RES+UNC      : {m['reliability'] - m['resolution'] + m['uncertainty']:.4f}"
        f"  (+ residual {m['residual']:.2e})",
        f"log-loss           : {s['log_loss']:.4f}",
        f"ECE / MCE          : {s['ece']:.4f} / {s['mce']:.4f}",
        f"Spiegelhalter Z    : {sp['z']:.3f}  (p={sp['p_value']:.3f}; |Z|>1.96 rejects calibration)",
        f"calibration-in-large: mean_pred {c['mean_pred']:.3f} vs base {c['base_rate']:.3f}"
        f"  (bias {c['bias']:+.3f})",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="calibrate",
        description="Score forecast/market calibration on the forecast ledger.",
    )
    p.add_argument("--book", choices=["polymarket", "crypto"], default=None,
                   help="which book's ledger to read (else $SF_BOOK / $SF_LEDGER_DIR)")
    p.add_argument("--bins", type=int, default=10, help="reliability bins (default 10)")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("score", help="print a calibration scorecard for the ledger")
    sp.add_argument("--json", action="store_true", help="emit JSON instead of text")

    sub.add_parser("table", help="print the calibration table (pred prob vs observed freq)")

    sp = sub.add_parser("report", help="write a reliability-diagram PNG + print the scorecard")
    sp.add_argument("--out", required=True, help="output PNG path")

    args = p.parse_args(argv)

    if args.command == "score":
        s = score_ledger(book=args.book, n_bins=args.bins)
        if args.json:
            s = {k: (v if k != "table" else v.to_dict(orient="records")) for k, v in s.items()}
            print(json.dumps(s, ensure_ascii=False, indent=2, default=str))
        else:
            print(_fmt_scorecard(s))
        return 0

    if args.command == "table":
        df = load_scored_forecasts(book=args.book)
        if df.empty:
            print("(no scored forecasts in the ledger yet)")
            return 0
        print(calibration_table(df["prob"].to_numpy(float), df["label"].to_numpy(float),
                                n_bins=args.bins).to_string(index=False))
        return 0

    if args.command == "report":
        df = load_scored_forecasts(book=args.book)
        if df.empty:
            print("(no scored forecasts in the ledger yet)")
            return 0
        p_arr = df["prob"].to_numpy(float)
        y_arr = df["label"].to_numpy(float)
        out = reliability_diagram({"ledger": (p_arr, y_arr)}, args.out, n_bins=args.bins,
                                  title="Forecast ledger calibration")
        print(_fmt_scorecard(score_ledger(book=args.book, n_bins=args.bins)))
        print(f"\nwrote {out}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
