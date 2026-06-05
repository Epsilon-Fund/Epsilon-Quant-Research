"""Batch replay Block A live CLOB capture shards into one feature Parquet.

This is intentionally a thin wrapper around ``dali_clob_replay_features``:
the underlying one-shard replay logic remains unchanged. The wrapper adds
run/shard metadata, enriches asset-level rows from the raw message metadata,
and trims rows after ``market_resolved`` lifecycle events.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.dali_clob_replay_features import replay


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "analysis" / "block_a1_features.parquet"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    run_dir: Path


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_run_spec(raw: str) -> RunSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--run must look like run_id=/path/to/run_dir")
    run_id, path = raw.split("=", 1)
    run_id = run_id.strip()
    if not run_id:
        raise argparse.ArgumentTypeError("run_id cannot be empty")
    run_dir = Path(path).expanduser()
    if not run_dir.exists() or not run_dir.is_dir():
        raise argparse.ArgumentTypeError(f"run directory not found: {run_dir}")
    return RunSpec(run_id=run_id, run_dir=run_dir)


def jsonl_shards(run_dir: Path) -> list[Path]:
    return sorted(
        path for path in run_dir.glob("*.jsonl")
        if path.name != "capture_gaps.jsonl"
    )


def as_ts(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)) or str(value).isdigit():
            numeric = float(value)
            unit = "ms" if numeric > 10_000_000_000 else "s"
            return pd.to_datetime(numeric, unit=unit, utc=True)
        return pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return None


def metadata_from_record(rec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for asset in rec.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        asset_id = str(asset.get("token_id") or asset.get("asset_id") or "")
        if not asset_id:
            continue
        out[asset_id] = {
            "asset_id": asset_id,
            "market_id": str(asset.get("market_id") or ""),
            "family": str(asset.get("family") or ""),
            "slug": str(asset.get("slug") or ""),
            "question": str(asset.get("question") or ""),
            "outcome_index": asset.get("outcome_index"),
        }
    return out


def scan_shard_metadata(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, pd.Timestamp]]:
    """Return asset metadata and first resolution timestamp by asset_id."""
    metadata: dict[str, dict[str, Any]] = {}
    resolved_at: dict[str, pd.Timestamp] = {}
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            metadata.update(metadata_from_record(rec))
            event_type = str(rec.get("event_type") or "")
            if event_type != "market_resolved":
                continue
            ts = as_ts(rec.get("received_at") or (rec.get("message") or {}).get("timestamp"))
            if ts is None:
                continue
            asset_ids = rec.get("asset_ids") or []
            msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
            asset_ids = asset_ids or msg.get("asset_ids") or msg.get("clob_token_ids") or []
            for asset_id in asset_ids:
                key = str(asset_id)
                prev = resolved_at.get(key)
                if prev is None or ts < prev:
                    resolved_at[key] = ts
    return metadata, resolved_at


def enrich_features(
    df: pd.DataFrame,
    *,
    run_id: str,
    shard: Path,
    metadata: dict[str, dict[str, Any]],
    resolved_at: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "run_id", run_id)
    out.insert(1, "shard", shard.name)
    out["asset_id"] = out["asset_id"].astype(str)

    meta = pd.DataFrame(metadata.values())
    if not meta.empty:
        out = out.merge(meta, on="asset_id", how="left", suffixes=("", "_raw"))
        for col in ("market_id", "family", "slug", "question", "outcome_index"):
            raw_col = f"{col}_raw"
            if raw_col in out.columns:
                if col in out.columns:
                    out[col] = out[col].where(out[col].notna() & out[col].astype(str).ne(""), out[raw_col])
                else:
                    out[col] = out[raw_col]
                out = out.drop(columns=[raw_col])

    if resolved_at:
        res = pd.DataFrame(
            {"asset_id": list(resolved_at.keys()), "market_resolved_at": list(resolved_at.values())}
        )
        out = out.merge(res, on="asset_id", how="left")
        keep = out["market_resolved_at"].isna() | (out["received_at"] <= out["market_resolved_at"])
        out = out.loc[keep].copy()
    else:
        out["market_resolved_at"] = pd.NaT

    return out


def replay_run(spec: RunSpec, top_n: int) -> pd.DataFrame:
    shards = jsonl_shards(spec.run_dir)
    if not shards:
        raise SystemExit(f"no JSONL shards found under {spec.run_dir}")

    pieces: list[pd.DataFrame] = []
    for idx, shard in enumerate(shards, start=1):
        metadata, resolved_at = scan_shard_metadata(shard)
        df = replay(shard, top_n=top_n)
        df = enrich_features(
            df,
            run_id=spec.run_id,
            shard=shard,
            metadata=metadata,
            resolved_at=resolved_at,
        )
        pieces.append(df)
        print(
            f"[{spec.run_id}] {idx:02d}/{len(shards):02d} "
            f"{display_path(shard)} -> {len(df):,} rows",
            flush=True,
        )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def atomic_write_parquet(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(out)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=parse_run_spec,
        help="Run spec as run_id=/path/to/run_dir. Repeat for joined output.",
    )
    parser.add_argument("--run-dir", type=Path, help="Single run directory.")
    parser.add_argument("--run-id", help="Single run id used with --run-dir.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs: list[RunSpec] = list(args.runs or [])
    if args.run_dir:
        runs.append(RunSpec(run_id=args.run_id or args.run_dir.name, run_dir=args.run_dir))
    if not runs:
        raise SystemExit("pass --run run_id=/path or --run-dir with --run-id")

    pieces = [replay_run(spec, args.top_n) for spec in runs]
    out_df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    out_df = out_df.sort_values(["run_id", "asset_id", "received_at"]).reset_index(drop=True)
    atomic_write_parquet(out_df, args.out)
    print(f"output: {display_path(args.out)}")
    print(f"rows: {len(out_df):,}")
    if len(out_df):
        print(f"runs: {out_df['run_id'].value_counts().to_dict()}")
        print(f"event types: {out_df['event_type'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
