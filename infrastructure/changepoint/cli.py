"""
CLI for the causal changepoint detector. Run from the repo root with the crypto venv:

    PYTHONPATH=. .venv/bin/python -m infrastructure.changepoint.cli detect \
        live_trading/cache/daily/BTCUSDT_daily.parquet --column Close --returns \
        --detector bocpd --out infrastructure/changepoint/changepoints/btc_daily_bocpd.parquet

    PYTHONPATH=. .venv/bin/python -m infrastructure.changepoint.cli benchmark
    PYTHONPATH=. .venv/bin/python -m infrastructure.changepoint.cli kappa-demo
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .stream import append_changepoints, breaks_from_stream, causal_standardize, run_detector


def _load_series(path: str, column: str, ts_col: str | None):
    df = pd.read_parquet(path)
    if ts_col and ts_col in df.columns:
        df = df.set_index(ts_col)
    if column not in df.columns:
        raise SystemExit(f"column '{column}' not in {path}; have {list(df.columns)}")
    s = df[column].astype(float)
    return s


def _cmd_detect(args) -> int:
    s = _load_series(args.path, args.column, args.ts_col)
    vals = s.to_numpy()
    if args.returns:
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = np.diff(np.log(vals), prepend=np.log(vals[0]))
        vals[~np.isfinite(vals)] = 0.0
    if args.standardize:
        vals = causal_standardize(vals)
    stream = run_detector(vals, name=args.detector, timestamps=s.index)
    breaks = breaks_from_stream(stream)
    print(f"[changepoint] {args.detector} on {args.path}::{args.column}"
          f"{' (log-returns)' if args.returns else ''}: "
          f"{len(breaks)} break(s) over {len(stream)} bars")
    for b in breaks[-15:]:
        row = stream.loc[b]
        print(f"    {b}  change_prob={row['change_prob']:.3f}  "
              f"run_length={int(row['run_length_mode'])}  stat={row['statistic']:.3f}")
    if len(breaks) > 15:
        print(f"    ... ({len(breaks) - 15} earlier breaks omitted)")
    if args.out:
        p = append_changepoints(stream, args.out)
        print(f"    appended -> {p}")
    return 0


def _cmd_benchmark(args) -> int:
    from .detectors import make_detector
    from .evaluate import benchmark_detector
    names = [args.detector] if args.detector else ["cusum", "page_hinkley", "bocpd"]
    print(f"{'detector':14s} {'scenario':11s} {'recall':>7s} {'med_lag':>8s} "
          f"{'far/1k':>8s} {'n_det':>6s}")
    print("-" * 60)
    for name in names:
        res = benchmark_detector(lambda n=name: make_detector(n), tolerance=args.tolerance)
        for scen in ("mean_shift", "var_shift", "noise"):
            m = res[scen]
            rec = "  n/a" if m["recall"] != m["recall"] else f"{m['recall']:.2f}"
            lag = "   n/a" if m["median_lag"] != m["median_lag"] else f"{m['median_lag']:.1f}"
            print(f"{name:14s} {scen:11s} {rec:>7s} {lag:>8s} "
                  f"{m['far_per_1000']:>8.2f} {m['n_detected']:>6.1f}")
    return 0


def _cmd_kappa_demo(args) -> int:
    from . import offline
    from .detectors import make_detector
    from .evaluate import kappa_vs_transitions
    x, labels, _ = offline.make_markov_switching(3000, seed=args.seed)
    print(f"{'detector':14s} {'kappa':>7s} {'tr_recall':>10s} {'fl_prec':>8s} "
          f"{'n_trans':>8s} {'n_flags':>8s}  (tolerance={args.tolerance})")
    print("-" * 64)
    for name in ("cusum", "page_hinkley", "bocpd"):
        s = run_detector(x, name=name)
        k = kappa_vs_transitions(labels, s["cp_flag"].to_numpy(), tolerance=args.tolerance)
        print(f"{name:14s} {k['kappa']:>7.3f} {k['transition_recall']:>10.2f} "
              f"{k['flag_precision']:>8.2f} {k['n_transitions']:>8d} {k['n_flags']:>8d}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="changepoint")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="run a detector over a parquet series")
    d.add_argument("path")
    d.add_argument("--column", default="Close")
    d.add_argument("--ts-col", default="Time")
    d.add_argument("--detector", default="bocpd", choices=["cusum", "page_hinkley", "bocpd"])
    d.add_argument("--returns", action="store_true", help="use log returns of the column")
    d.add_argument("--standardize", action="store_true", help="causal expanding z-score (lookahead-free)")
    d.add_argument("--out", default=None, help="append-only parquet output path")
    d.set_defaults(func=_cmd_detect)

    b = sub.add_parser("benchmark", help="detection-lag / FPR on synthetic series")
    b.add_argument("--detector", default=None, choices=["cusum", "page_hinkley", "bocpd"])
    b.add_argument("--tolerance", type=int, default=25)
    b.set_defaults(func=_cmd_benchmark)

    k = sub.add_parser("kappa-demo", help="Cohen's kappa vs Markov-switching transitions")
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--tolerance", type=int, default=5)
    k.set_defaults(func=_cmd_kappa_demo)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
