#!/usr/bin/env bash
#
# expire_r2_raw.sh — expire raw shards from the R2 archive as soon as they are
# VERIFIED parsed, so cloud storage does not grow without bound. Parquet is the
# keeper (kept forever); raw is deleted the moment its parquet is confirmed.
#
# A raw object  r2:<bucket>/raw/{date}/{shard}.jsonl.gz  is deleted when EITHER:
#   A. its parquet is confirmed in R2 — a non-empty *_{shard}.parquet under the same
#      date (the normal "verified parsed" path); OR
#   B. it is an unknown_* shard AND older than UNKNOWN_R2_RETENTION_DAYS. unknown_* is
#      the WS's global new_market announcement firehose — no L2 content, so it never
#      produces a parquet and path (A) would keep it forever (R2 grows ~4 MiB/day
#      unbounded). (B) is age-gated PAST sync_cloud.sh's local raw retention so the
#      local copy is already gone and the next sync won't re-upload what we delete
#      (the 2026-06-24 re-upload churn bug).
# Anything else that fails verification is KEPT and logged. This NEVER touches parquet,
# NEVER touches research-live-clob/, and never deletes parsed raw lacking a parquet.
#
# "Verified parsed" leans on the compression pipeline's own guarantees: it writes a
# shard's parquet only after passing row-count and column checks, and names it
# {table}_{shard}.parquet — so a non-empty parquet for the shard means the parse
# succeeded. (It does NOT re-validate parquet content here; that was checked at
# parse time. A subtle parser bug that still produced a non-empty parquet is an
# accepted risk of deleting raw — see the data-retention decision.)
#
# Usage:
#   expire_r2_raw.sh            # real run (deletes eligible raw from R2)
#   expire_r2_raw.sh --dry-run  # preview only; deletes nothing
#
# Linux / GNU coreutils assumed. Run on the VPS via systemd timer (every 6h).

set -euo pipefail

REMOTE="r2"
BUCKET="epsilon-polymarket-data"
# unknown_* shards never get a parquet (WS new_market firehose). Age them out of R2
# after this many days. MUST be >= sync_cloud.sh's RAW_RETENTION_DAYS (~4d kept locally)
# so the local copy is gone first and the next sync won't re-upload what we delete here.
UNKNOWN_R2_RETENTION_DAYS="${UNKNOWN_R2_RETENTION_DAYS:-4}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/data/logs"
LOG_FILE="$LOG_DIR/expire_r2_raw.log"
mkdir -p "$LOG_DIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"; }

command -v rclone >/dev/null 2>&1 || { log "ERROR: rclone not found in PATH"; exit 2; }

unknown_cutoff="$(date -u -d "${UNKNOWN_R2_RETENTION_DAYS} days ago" +%Y-%m-%d)"
label=""; [ "$DRY_RUN" -eq 1 ] && label="(DRY RUN) "
log "=== ${label}expire_r2_raw start (delete raw with confirmed parquet; age out unknown_* older than ${UNKNOWN_R2_RETENTION_DAYS}d, i.e. before ${unknown_cutoff}) ==="

raw_list="$(mktemp)"; pq_list="$(mktemp)"
trap 'rm -f "$raw_list" "$pq_list"' EXIT

# "<relpath>\t<size>"  — fail safe: a listing error aborts with nothing deleted.
if ! rclone lsf -R --files-only --format "ps" --separator $'\t' "$REMOTE:$BUCKET/raw" > "$raw_list" 2>>"$LOG_FILE"; then
    log "ERROR: could not list R2 raw — aborting, nothing deleted"; exit 1
fi
if ! rclone lsf -R --files-only --format "ps" --separator $'\t' "$REMOTE:$BUCKET/parquet" > "$pq_list" 2>>"$LOG_FILE"; then
    log "ERROR: could not list R2 parquet — aborting, nothing deleted"; exit 1
fi

deleted=0 kept_unverified=0 freed=0 unknown_aged=0
while IFS=$'\t' read -r rawpath rawsize; do
    [ -n "$rawpath" ] || continue
    case "$rawpath" in *.jsonl.gz) ;; *) continue ;; esac
    [[ "$rawsize" =~ ^[0-9]+$ ]] || rawsize=0

    date="${rawpath%%/*}"
    [[ "$date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || continue   # ignore non-date-partition paths

    base="${rawpath##*/}"; stem="${base%.jsonl.gz}"

    # unknown_* never gets a parquet (new_market firehose) — age it out instead of
    # keeping it forever. Gated past local raw retention so sync won't re-upload it.
    case "$stem" in
      unknown_*)
        if [[ ! "$date" < "$unknown_cutoff" ]]; then       # too recent -> keep for now
            kept_unverified=$((kept_unverified + 1)); continue
        fi
        if [ "$DRY_RUN" -eq 1 ]; then
            log "  WOULD DELETE (unknown, aged): raw/$rawpath"
            deleted=$((deleted + 1)); unknown_aged=$((unknown_aged + 1)); freed=$((freed + rawsize))
        elif rclone deletefile "$REMOTE:$BUCKET/raw/$rawpath" 2>>"$LOG_FILE"; then
            deleted=$((deleted + 1)); unknown_aged=$((unknown_aged + 1)); freed=$((freed + rawsize))
        else
            log "  WARN: failed to delete raw/$rawpath"
        fi
        continue
        ;;
    esac

    # verify: a NON-EMPTY parquet exists for this shard ({table}_{stem}.parquet under same date)
    pqmatch="$(awk -F'\t' -v d="$date" -v s="$stem" \
        '$1 ~ ("^" d "/.*_" s "\\.parquet$") && ($2 + 0) > 0 { print $1; exit }' "$pq_list")"
    if [ -z "$pqmatch" ]; then
        kept_unverified=$((kept_unverified + 1))
        log "  KEEP (no non-empty parquet found): $rawpath"
        continue
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        log "  WOULD DELETE: raw/$rawpath  (verified by $pqmatch)"
        deleted=$((deleted + 1)); freed=$((freed + rawsize))
    elif rclone deletefile "$REMOTE:$BUCKET/raw/$rawpath" 2>>"$LOG_FILE"; then
        deleted=$((deleted + 1)); freed=$((freed + rawsize))
    else
        log "  WARN: failed to delete raw/$rawpath"
    fi
done < "$raw_list"

log "${label}summary: ${deleted} raw shard(s) $([ "$DRY_RUN" -eq 1 ] && echo 'eligible' || echo 'deleted') (~$((freed / 1024 / 1024)) MiB; ${unknown_aged} were aged-out unknown_), kept_unverified=${kept_unverified}"
log "=== ${label}expire_r2_raw done ==="
exit 0
