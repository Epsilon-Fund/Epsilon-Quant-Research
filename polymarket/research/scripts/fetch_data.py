"""fetch_data.py — download the research_v1 library from R2 to local disk. No rclone.

    python scripts/fetch_data.py                    # -> ./data/research_v1  (the loader default)
    python scripts/fetch_data.py --dest D:/epsilon/research_v1
    python scripts/fetch_data.py --dry-run          # list what it would fetch, transfer nothing

Credentials: the SAME three env vars the loader uses — EPSILON_R2_KEY_ID / EPSILON_R2_SECRET /
EPSILON_R2_ENDPOINT — read via epsilon_data's own credential path (with the rclone.conf [r2]
fallback). Ask the operator for them; put them in a gitignored .env. They are read/write today,
so treat them accordingly. Secrets are never printed or written anywhere.

READ-ONLY against R2: this script only lists and gets objects. It contains no put/delete/copy —
the key can delete and the 71 GB raw archive has no second copy.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# import epsilon_data (for the shared credential loader) however we're launched
_root = next((p for p in Path(__file__).resolve().parents if (p / "epsilon_data").is_dir()), None)
if _root and str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

BUCKET = "epsilon-polymarket-data"
PREFIX = "research/v1/"
EXPECT_FILES = 792
EXPECT_TOKENS = 30772


def _client():
    """R2 S3 client using the loader's credentials (same env vars / rclone fallback)."""
    from epsilon_data._internal import _r2_creds  # reuse — do not write a second credential loader
    creds = _r2_creds()
    if not creds:
        sys.exit("No R2 credentials. Set EPSILON_R2_KEY_ID / EPSILON_R2_SECRET / EPSILON_R2_ENDPOINT "
                 "(or an rclone [r2] remote). Ask the operator; put them in a gitignored .env.")
    kid, sec, ep = creds
    import boto3
    from botocore.config import Config
    # R2 requires PATH-style addressing (the default virtual-hosted style builds an unresolvable
    # bucket.<account>.r2.cloudflarestorage.com host and hangs). Explicit timeouts fail fast.
    return boto3.client("s3", endpoint_url=f"https://{ep}", aws_access_key_id=kid,
                        aws_secret_access_key=sec, region_name="auto",
                        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4",
                                      retries={"max_attempts": 6, "mode": "standard"},
                                      connect_timeout=15, read_timeout=60, max_pool_connections=16))


def _list(s3):
    """[(relative_path, size)] for every object under the prefix. List-only."""
    out = []
    tok = None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=PREFIX)
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            key = o["Key"]
            if key.endswith("/"):
                continue
            out.append((key[len(PREFIX):], o["Size"], key))
        if r.get("IsTruncated"):
            tok = r.get("NextContinuationToken")
        else:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description="Fetch research_v1 from R2 (no rclone).")
    ap.add_argument("--dest", default=str((_root or Path(".")) / "data" / "research_v1"),
                    help="destination dir (default: ./data/research_v1, the loader default)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched; transfer nothing")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dest = Path(args.dest)

    s3 = _client()
    print(f"Listing s3://{BUCKET}/{PREFIX} …", flush=True)
    objs = _list(s3)
    total_files = len(objs)
    total_bytes = sum(sz for _, sz, _ in objs)
    print(f"R2 holds {total_files} files, {total_bytes/1e9:.2f} GB under {PREFIX}", flush=True)

    if args.dry_run:
        # what we'd fetch vs skip (by same-size), transferring nothing
        to_fetch = skip = 0
        for rel, sz, _ in objs:
            p = dest / rel
            if p.exists() and p.stat().st_size == sz:
                skip += 1
            else:
                to_fetch += 1
        print(f"[dry-run] dest={dest}")
        print(f"[dry-run] would fetch {to_fetch}, skip {skip} (already present, same size). No bytes transferred.")
        print("\nSample of what would be fetched:")
        shown = 0
        for rel, sz, _ in objs:
            p = dest / rel
            if not (p.exists() and p.stat().st_size == sz):
                print(f"    {rel}  ({sz/1e6:.2f} MB)"); shown += 1
            if shown >= 5:
                break
        return 0

    # plan
    todo = []
    skipped = 0
    for rel, sz, key in objs:
        p = dest / rel
        if p.exists() and p.stat().st_size == sz:
            skipped += 1
        else:
            todo.append((rel, sz, key))
    fetch_bytes = sum(sz for _, sz, _ in todo)
    print(f"dest={dest}\n{skipped} already present (same size, skipped); {len(todo)} to fetch "
          f"({fetch_bytes/1e9:.2f} GB).", flush=True)

    done_files = 0
    done_bytes = 0
    lock = threading.Lock()
    t0 = time.time()

    def fetch(item):
        rel, sz, key = item
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        s3.download_file(BUCKET, key, str(tmp))   # GetObject only (read-only)
        os.replace(tmp, p)
        return sz

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch, it): it for it in todo}
            for fut in as_completed(futs):
                sz = fut.result()
                with lock:
                    done_files += 1
                    done_bytes += sz
                    if done_files % 20 == 0 or done_files == len(todo):
                        el = time.time() - t0
                        rate = done_bytes / el / 1e6 if el else 0
                        print(f"  {done_files}/{len(todo)} files · {done_bytes/1e9:.2f}/{fetch_bytes/1e9:.2f} GB · {rate:.1f} MB/s",
                              flush=True)
    print(f"fetch done in {time.time()-t0:.0f}s ({skipped} skipped, {len(todo)} fetched).", flush=True)

    # ---- verify ----
    checks = {}
    # exclude our own local-only artifacts (never in R2's listing) so a re-run still verifies clean
    _local_only = {"_fetch_receipt.json"}
    local_files = [p for p in dest.rglob("*")
                   if p.is_file() and not p.name.endswith(".part") and p.name not in _local_only]
    checks["file_count"] = {"expect": total_files, "got": len(local_files), "ok": len(local_files) == total_files}
    local_bytes = sum(p.stat().st_size for p in local_files)
    checks["total_bytes"] = {"expect": total_bytes, "got": local_bytes, "ok": local_bytes == total_bytes}
    import duckdb
    def q(sql):
        c = duckdb.connect()
        try:
            return c.execute(sql).fetchone()[0]
        finally:
            c.close()
    def uq(*parts):
        return str(dest.joinpath(*parts)).replace("\\", "/")
    try:
        ntok = int(q(f"SELECT COUNT(*) FROM read_parquet('{uq('tokens.parquet')}')"))
    except Exception as e:
        ntok = -1
    checks["tokens_rows"] = {"expect": EXPECT_TOKENS, "got": ntok, "ok": ntok == EXPECT_TOKENS}
    try:
        nl1 = int(q(f"SELECT COUNT(*) FROM read_parquet('{uq('l1','*','*','*.parquet')}')"))
        ntr = int(q(f"SELECT COUNT(*) FROM read_parquet('{uq('trades','*','*','*.parquet')}')"))
    except Exception:
        nl1 = ntr = 0
    checks["l1_trades_nonempty"] = {"l1_rows": nl1, "trades_rows": ntr, "ok": nl1 > 0 and ntr > 0}

    receipt = {"fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "source": f"s3://{BUCKET}/{PREFIX}", "file_count": len(local_files),
               "total_bytes": local_bytes, "checks": checks}
    (dest / "_fetch_receipt.json").write_text(json.dumps(receipt, indent=2))

    print("\n=== verification ===")
    all_ok = True
    for name, c in checks.items():
        ok = c.get("ok"); all_ok &= bool(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: {c}")
    if not all_ok:
        bad = [n for n, c in checks.items() if not c.get("ok")]
        print(f"\nFETCH INCOMPLETE — failing check(s): {', '.join(bad)}. Re-run the same command to finish.")
        return 1
    print("\nAll checks passed. Wrote _fetch_receipt.json. Next:")
    print(f"    set EPSILON_DATA_ROOT={dest}")
    print(f"    python scripts/check_setup.py")
    print(f"    streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
