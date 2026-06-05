"""Repair Gamma market metadata on direct Polygon OrderFilled shards.

This is a metadata-only repair for shards produced by
``sync_polygon_order_fills.py``. It does not refetch logs; it fills null
``market_id``/``condition_id`` values by looking up the traded CLOB token ID in
Gamma, including closed markets.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/repair_polygon_market_ids.py
    PYTHONPATH=. uv run python scripts/repair_polygon_market_ids.py --write
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from data_infra.duck import connect
from sync_polygon_order_fills import (
    INPROG_DIR,
    fetch_gamma_token_mappings,
    load_market_lookup,
)


DEFAULT_GLOB = "data/trades/trades_delta_shardpolygon_*.parquet"


def sql_path(path: str | Path) -> str:
    return str(path).replace("'", "''")


def token_expr(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return (
        f"CASE WHEN {prefix}maker_asset_id = '0' "
        f"THEN {prefix}taker_asset_id ELSE {prefix}maker_asset_id END"
    )


def lookup_arrow(
    lookup: dict[str, tuple[str | None, str | None, bool | None]],
) -> pa.Table:
    rows = [
        (token, market_id, condition_id, neg_risk)
        for token, (market_id, condition_id, neg_risk) in lookup.items()
        if market_id is not None
    ]
    return pa.table(
        {
            "token_id": [row[0] for row in rows],
            "market_id": [row[1] for row in rows],
            "condition_id": [row[2] for row in rows],
            "neg_risk": [row[3] for row in rows],
        }
    )


def missing_tokens(con: Any, parquet_glob: str) -> list[str]:
    rows = con.sql(
        f"""
        SELECT DISTINCT {token_expr()} AS token_id
        FROM read_parquet('{sql_path(parquet_glob)}')
        WHERE market_id IS NULL
          AND {token_expr()} IS NOT NULL
          AND {token_expr()} <> '0'
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def fetch_missing_mappings(
    tokens: list[str],
    lookup: dict[str, tuple[str | None, str | None, bool | None]],
    *,
    batch_size: int,
    timeout_s: float,
    max_attempts: int,
    progress_tokens: int,
) -> None:
    for start in range(0, len(tokens), progress_tokens):
        chunk = tokens[start : start + progress_tokens]
        fetch_gamma_token_mappings(
            chunk,
            lookup,
        batch_size=batch_size,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
        closed_first=True,
    )
        print(
            f"gamma_lookup={min(start + progress_tokens, len(tokens)):,}/"
            f"{len(tokens):,} lookup_size={len(lookup):,}",
            flush=True,
        )


def shard_stats(con: Any, path: Path) -> tuple[int, int, int]:
    row = con.sql(
        f"""
        SELECT
            count() AS rows,
            count() FILTER (WHERE p.market_id IS NULL) AS null_before,
            count() FILTER (
                WHERE p.market_id IS NULL AND m.market_id IS NOT NULL
            ) AS repairable
        FROM read_parquet('{sql_path(path)}') AS p
        LEFT JOIN token_lookup AS m
          ON {token_expr('p')} = m.token_id
        """
    ).fetchone()
    return int(row[0]), int(row[1]), int(row[2])


def repair_shard(
    con: Any,
    path: Path,
    *,
    write: bool,
    replace_inprogress: bool,
) -> dict[str, Any]:
    rows, null_before, repairable = shard_stats(con, path)
    summary = {
        "path": str(path),
        "rows": rows,
        "null_market_id_before": null_before,
        "repairable_rows": repairable,
        "status": "dry_run" if not write else "unchanged",
    }
    if not write or repairable == 0:
        return summary

    INPROG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INPROG_DIR / f"{path.name}.repair.inprogress"
    if tmp.exists():
        if not replace_inprogress:
            raise SystemExit(f"in-progress repair file exists: {tmp}")
        tmp.unlink()

    con.sql(
        f"""
        COPY (
            SELECT
                p.timestamp,
                CASE
                    WHEN p.market_id IS NULL AND m.market_id IS NOT NULL
                    THEN m.market_id
                    ELSE p.market_id
                END AS market_id,
                CASE
                    WHEN p.market_id IS NULL AND m.market_id IS NOT NULL
                    THEN m.condition_id
                    ELSE p.condition_id
                END AS condition_id,
                CASE
                    WHEN p.market_id IS NULL AND m.market_id IS NOT NULL
                    THEN COALESCE(m.neg_risk, p.neg_risk)
                    ELSE p.neg_risk
                END AS neg_risk,
                p.maker,
                p.taker,
                p.maker_asset_id,
                p.taker_asset_id,
                p.usd_amount,
                p.token_amount,
                p.price,
                p.maker_side,
                p.transaction_hash
            FROM read_parquet('{sql_path(path)}') AS p
            LEFT JOIN token_lookup AS m
              ON {token_expr('p')} = m.token_id
        ) TO '{sql_path(tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    tmp.replace(path)
    summary["status"] = "written"
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default=DEFAULT_GLOB)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--replace-inprogress", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--gamma-batch-size", type=int, default=40)
    parser.add_argument("--progress-tokens", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [Path(path) for path in sorted(glob.glob(args.glob))]
    if not paths:
        raise SystemExit(f"no parquet shards matched {args.glob}")

    con = connect()
    tokens = missing_tokens(con, args.glob)
    lookup = load_market_lookup()
    print(
        json.dumps(
            {
                "write": args.write,
                "shards": len(paths),
                "missing_tokens": len(tokens),
                "lookup_loaded": len(lookup),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    fetch_missing_mappings(
        tokens,
        lookup,
        batch_size=args.gamma_batch_size,
        timeout_s=args.timeout_s,
        max_attempts=args.max_attempts,
        progress_tokens=args.progress_tokens,
    )

    con.register("token_lookup", lookup_arrow(lookup))
    summaries = [
        repair_shard(
            con,
            path,
            write=args.write,
            replace_inprogress=args.replace_inprogress,
        )
        for path in paths
    ]
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
