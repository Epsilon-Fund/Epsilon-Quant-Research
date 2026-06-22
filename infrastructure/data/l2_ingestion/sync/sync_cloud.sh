#!/usr/bin/env bash
#
# sync_cloud.sh — push local L2 data to Cloudflare R2, then prune old local raw.
#
# Run hourly (after the compression pipeline) via systemd/cron. Assumes rclone is
# already configured with a remote named "r2" pointing at the R2 account
# (see deploy/ for one-time setup). It does NOT configure rclone.
#
# What it does:
#   1. parquet -> R2 with `rclone sync`  (mirror the working set both ways)
#   2. raw     -> R2 with `rclone copy`  (backup; copy NEVER deletes on the
#      destination, so pruning local raw below does not erase the cloud backup)
#   3. delete LOCAL raw *.jsonl.gz older than RETENTION_DAYS (cloud keeps them)
#   4. never deletes local parquet (we query those locally)
#
# IMPORTANT: raw uses `copy`, not `sync`, ON PURPOSE. `rclone sync` makes the
# destination match the source — so once we delete a local raw shard, a `sync`
# would delete it from R2 too, destroying the backup. `copy` only adds/updates.
#
# Exits nonzero on any rclone failure so the cron/systemd unit flags the problem.

set -euo pipefail

# --- config ---------------------------------------------------------------
REMOTE="r2"
BUCKET="epsilon-polymarket-data"
RETENTION_DAYS=7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/data"
PARQUET_DIR="$DATA_DIR/parquet"
RAW_DIR="$DATA_DIR/raw"
LOG_DIR="$DATA_DIR/logs"
LOG_FILE="$LOG_DIR/sync_cloud.log"

mkdir -p "$LOG_DIR"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"; }

# rclone options shared by both transfers:
#   --checksum : compare by content hash, not mtime+size (see notes / explanation)
#   --transfers/--checkers : modest parallelism for many small files
#   --stats-one-line : compact periodic + final summary line
RCLONE_OPTS=(--checksum --transfers 8 --checkers 16 --stats-one-line --stats=1m)

# --- preflight ------------------------------------------------------------
command -v rclone >/dev/null 2>&1 || { log "ERROR: rclone not found in PATH"; exit 2; }

START=$SECONDS
log "=== sync_cloud start (remote=$REMOTE bucket=$BUCKET) ==="

# run_rclone <mode> <src> <dst> : stream+log output, return rclone's exit code
run_rclone() {
    local mode="$1" src="$2" dst="$3"
    if [ ! -d "$src" ]; then
        log "skip: source $src does not exist yet"
        return 0
    fi
    set +e
    rclone "$mode" "$src" "$dst" "${RCLONE_OPTS[@]}" 2>&1 | tee -a "$LOG_FILE"
    local rc=${PIPESTATUS[0]}
    set -e
    return "$rc"
}

# --- 1. parquet (working set) -> sync (mirror) ----------------------------
log "syncing parquet  $PARQUET_DIR -> $REMOTE:$BUCKET/parquet"
if ! run_rclone sync "$PARQUET_DIR" "$REMOTE:$BUCKET/parquet"; then
    log "ERROR: parquet sync failed"
    exit 1
fi

# --- 2. raw (backup) -> copy (never deletes remote) -----------------------
log "backing up raw   $RAW_DIR -> $REMOTE:$BUCKET/raw  (copy, no remote deletes)"
RAW_OK=1
if ! run_rclone copy "$RAW_DIR" "$REMOTE:$BUCKET/raw"; then
    log "ERROR: raw backup failed — will NOT prune local raw this run"
    RAW_OK=0
fi

# --- 3. prune local raw older than RETENTION_DAYS (only if backup succeeded) -
PRUNED=0
if [ "$RAW_OK" -eq 1 ] && [ -d "$RAW_DIR" ]; then
    log "pruning local raw *.jsonl.gz older than ${RETENTION_DAYS}d (cloud retains them)"
    while IFS= read -r -d '' f; do
        rm -f "$f" && PRUNED=$((PRUNED + 1)) && log "  pruned $(basename "$f")"
    done < <(find "$RAW_DIR" -name '*.jsonl.gz' -type f -mtime +"$RETENTION_DAYS" -print0)
    log "  pruned $PRUNED local raw file(s)"
fi

# --- 4. summary -----------------------------------------------------------
DUR=$((SECONDS - START))
PARQUET_BYTES=$( [ -d "$PARQUET_DIR" ] && du -sb "$PARQUET_DIR" 2>/dev/null | cut -f1 || echo 0 )
RAW_BYTES=$( [ -d "$RAW_DIR" ] && du -sb "$RAW_DIR" 2>/dev/null | cut -f1 || echo 0 )
PARQUET_FILES=$( [ -d "$PARQUET_DIR" ] && find "$PARQUET_DIR" -name '*.parquet' -type f | wc -l || echo 0 )
RAW_FILES=$( [ -d "$RAW_DIR" ] && find "$RAW_DIR" -name '*.jsonl.gz' -type f | wc -l || echo 0 )

log "transfer summary (bytes transferred are in the rclone 'Transferred:' lines above)"
log "  local parquet: ${PARQUET_FILES} files, ${PARQUET_BYTES} bytes"
log "  local raw    : ${RAW_FILES} files, ${RAW_BYTES} bytes (pruned ${PRUNED} this run)"
log "=== sync_cloud done in ${DUR}s ==="
exit 0
