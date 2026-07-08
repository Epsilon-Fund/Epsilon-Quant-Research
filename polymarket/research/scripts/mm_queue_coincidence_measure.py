"""Measure trade / price_change timestamp coincidence on captured L2 — motivates FIX 3.

FIX 3 (order-invariant netting in ``mm_engine/queue_models.py``) only matters when a
``last_trade`` and the ``price_change`` reporting its depletion land on the **same**
``ts_exchange`` (``timestamp_ms``): then the arrival order within that millisecond would,
under naive immediate attribution, decide whether the trade is correctly netted out or
double-counted as a cancel. This script measures how often that coincidence actually
happens on the real capture in ``./l2_data/`` so the findings note can state the impact.

For each trade we map the aggressor side to the resting book side it consumes (a SELL
aggressor lifts resting BIDs -> a ``price_change`` with ``side='BUY'``; a BUY aggressor
lifts resting ASKs -> ``side='SELL'``), exactly as the queue model does, and ask:

* **exact coincidence** — is there a ``price_change`` at the same ``(timestamp_ms, asset_id,
  price, resting_side)``? This is the precise event FIX 3 nets.
* **loose coincidence** — is there *any* ``price_change`` at the same ``(timestamp_ms,
  asset_id)``? (an upper bound on how often the trade's millisecond is "busy").

Run:  PYTHONPATH=. uv run python scripts/mm_queue_coincidence_measure.py [--l2-dir l2_data]

Read-only aggregation (DuckDB over Parquet); deterministic; no writes.
"""
from __future__ import annotations

import argparse
import glob
import os

import duckdb


def _universes(l2_dir: str) -> list[str]:
    found = set()
    for path in glob.glob(os.path.join(l2_dir, "*", "*")):
        if os.path.isdir(path):
            found.add(os.path.basename(path))
    return sorted(found)


_METRIC_SQL = """
WITH tr AS (
    SELECT
        timestamp_ms,
        asset_id,
        round(price, 3) AS px,
        CASE WHEN upper(side) = 'SELL' THEN 'BUY'
             WHEN upper(side) = 'BUY'  THEN 'SELL' END AS resting_side
    FROM read_parquet({trades!r})
),
pck AS (  -- distinct so the LEFT JOINs match at most one row per trade
    SELECT DISTINCT timestamp_ms, asset_id, round(price, 3) AS px, upper(side) AS side
    FROM read_parquet({pcs!r})
),
pca AS (
    SELECT DISTINCT timestamp_ms, asset_id FROM read_parquet({pcs!r})
),
j AS (
    SELECT
        tr.*,
        pck.timestamp_ms IS NOT NULL AS has_exact,
        pca.timestamp_ms IS NOT NULL AS has_any
    FROM tr
    LEFT JOIN pck
      ON pck.timestamp_ms = tr.timestamp_ms AND pck.asset_id = tr.asset_id
     AND pck.px = tr.px AND pck.side = tr.resting_side
    LEFT JOIN pca
      ON pca.timestamp_ms = tr.timestamp_ms AND pca.asset_id = tr.asset_id
)
SELECT
    count(*)                          AS n_trades,
    count(*) FILTER (WHERE has_exact) AS n_exact,
    count(*) FILTER (WHERE has_any)   AS n_any
FROM j
"""


def _measure(con: duckdb.DuckDBPyConnection, trades_glob: str, pcs_glob: str) -> dict:
    if not glob.glob(trades_glob) or not glob.glob(pcs_glob):
        return {"n_trades": 0, "n_exact": 0, "n_any": 0}
    sql = _METRIC_SQL.format(trades=trades_glob, pcs=pcs_glob)
    row = con.execute(sql).fetchone()
    return {"n_trades": row[0], "n_exact": row[1], "n_any": row[2]}


def _pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.2f}%" if den else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l2-dir", default="l2_data", help="root of the captured L2 tree")
    args = ap.parse_args()

    con = duckdb.connect()
    universes = _universes(args.l2_dir)
    if not universes:
        print(f"No universes found under {args.l2_dir!r}.")
        return

    print(f"L2 trade / price_change timestamp-coincidence ({args.l2_dir})\n")
    header = f"{'universe':<20}{'trades':>12}{'exact-coincident':>20}{'any-coincident':>18}"
    print(header)
    print("-" * len(header))

    totals = {"n_trades": 0, "n_exact": 0, "n_any": 0}
    for uni in universes:
        trades_glob = os.path.join(args.l2_dir, "*", uni, "trades_*.parquet")
        pcs_glob = os.path.join(args.l2_dir, "*", uni, "price_change_*.parquet")
        m = _measure(con, trades_glob, pcs_glob)
        for k in totals:
            totals[k] += m[k]
        print(f"{uni:<20}{m['n_trades']:>12,}"
              f"{m['n_exact']:>12,} ({_pct(m['n_exact'], m['n_trades']):>5})"
              f"{m['n_any']:>10,} ({_pct(m['n_any'], m['n_trades']):>5})")

    print("-" * len(header))
    print(f"{'ALL':<20}{totals['n_trades']:>12,}"
          f"{totals['n_exact']:>12,} ({_pct(totals['n_exact'], totals['n_trades']):>5})"
          f"{totals['n_any']:>10,} ({_pct(totals['n_any'], totals['n_trades']):>5})")
    print(
        "\nRead: 'exact-coincident' = fraction of trades that share a timestamp with a "
        "price_change\nat the same (token, price, resting side) — the case FIX 3 nets. "
        "'any-coincident' =\nfraction whose millisecond also carries some other "
        "price_change (busy-ms upper bound)."
    )


if __name__ == "__main__":
    main()
