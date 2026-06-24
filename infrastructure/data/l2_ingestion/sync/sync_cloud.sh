#!/usr/bin/env bash
#
# sync_cloud.sh — push local L2 data to Cloudflare R2, then prune old local
# raw AND parquet that are confirmed safe in R2.
#
# Run every 6h (after the compression pipeline) via systemd timer. Assumes rclone
# is already configured with a remote named "r2" pointing at the R2 account
# (see deploy/ for one-time setup; the remote must have no_check_bucket=true so an
# Object-Read&Write token is not asked to CreateBucket). It does NOT configure rclone.
# Linux / GNU coreutils assumed (stat -c, du -sb, find -mindepth, df --output).
#
# What it does (ORDER MATTERS — raw is pruned BEFORE it is re-uploaded):
#   1. parquet -> R2 with `rclone copy`  (backup; copy NEVER deletes on the remote)
#   2. prune LOCAL raw whose PARQUET is confirmed in R2 (any age) — stops re-upload churn
#   3. raw (only the remaining UNPARSED shards) -> R2 with `rclone copy`
#   4. BACKSTOP: prune LOCAL raw older than RAW_RETENTION_DAYS that is verified in R2
#      (bounds disk against parquet-less unknown_*/failed-parse shards)
#   5. prune LOCAL parquet *.parquet older than PARQUET_RETENTION_DAYS (7 -> ~8d kept)
#   6. alarm (nonzero exit) if the disk is still above DISK_ALERT_PCT afterward
#
# WHY BOTH USE `copy`, NEVER `sync`:
#   `rclone sync` makes the destination mirror the source — so once we delete a
#   local file, a `sync` would delete it from R2 too, destroying the archive.
#   `copy` only ever adds/updates on the remote, never deletes. Parquet USED to
#   use `sync`; it was switched to `copy` when local parquet pruning was added
#   (2026-06-23), otherwise the new pruning would wipe the R2 parquet archive.
#
# WHY RAW IS PRUNED BEFORE IT IS UPLOADED (re-upload churn fix, 2026-06-24):
#   Raw's only job in R2 is a safety net for shards not yet parsed. expire_r2_raw.sh
#   deletes an R2 raw shard once its parquet is confirmed; previously the next sync
#   RE-UPLOADED that same raw from the local copy (kept ~3 days via the old age-based
#   prune), so R2 raw never actually shrank on parse — it churned for ~3-4 days until
#   the local copy aged out. Fix: as soon as a shard's non-empty parquet is confirmed
#   in R2, delete the LOCAL raw (prune_raw_if_parsed) and do it BEFORE the raw upload,
#   so parsed raw is never sent again. Net: both local and R2 raw hold only unparsed
#   shards; parsed raw is gone within one cycle. expire_r2_raw.sh stays as the R2-side
#   backstop for raw uploaded in a prior run before its parquet existed.
#
# PRUNE SAFETY MODEL — PER-FILE VERIFIED (changed 2026-06-23):
#   Pruning is NOT gated on a global "did the whole upload succeed" flag. A local
#   file is deleted ONLY if R2 already holds an object at the SAME relative path
#   with the SAME byte size. This is critical: on a 24/7 capture there is ALWAYS a
#   current-hour shard being appended, and rclone fails to copy that one file
#   ("source file is being updated"). The previous global RAW_OK gate therefore
#   evaluated false on EVERY run, so pruning never happened and the disk filled to
#   99%. Per-file verification prunes everything provably backed up while leaving
#   in-progress / not-yet-uploaded files untouched.
#
#   WHY SIZE IS A SOUND PROXY FOR CONTENT HERE (not a hash): the producers are
#   write-once-per-path. Raw shards are append-only then closed (a file >retention
#   age is no longer being written). compression/pipeline.py writes each shard to
#   a UNIQUE path "{table}_{shard}.parquet" exactly once (no read-concat, no
#   in-place rewrite) and records the shard as processed so it is never reprocessed.
#   So a given relative path never changes content, and a same-path same-size R2
#   object is necessarily the same bytes. This invariant is what makes the
#   size-only check safe and also why `copy` never produces R2 duplicates/orphans
#   (paths are never renamed or superseded). If that producer invariant ever
#   changes (in-place parquet rewrites / recompaction), switch this check to a
#   content hash (rclone lsf --format ph --hash md5) before trusting it.
#
# DISK MATH / CADENCE: raw is now pruned on parse-confirm (not by age), so local raw
# holds only the unparsed window plus a <=RAW_RETENTION_DAYS verified backstop of
# parquet-less unknown_*/failed shards — typically ~1-2 GB. Parquet keeps ~8 days
# (~9 GB). With the OS that lands the disk around ~50% (it was ~77% under the old
# age-based raw retention). The DISK_ALERT_PCT guard below is the backstop: if a
# persistent R2 outage (the only path where pruning legitimately can't run, because
# we keep everything we cannot verify) lets the disk creep back up, the run exits
# nonzero so the systemd unit / log monitoring surfaces it.

set -euo pipefail

# --- config ---------------------------------------------------------------
REMOTE="r2"
BUCKET="epsilon-polymarket-data"
# RAW_RETENTION_DAYS is now ONLY a backstop for UNPARSED raw (unknown_*/failed parses)
# that is already safe in R2 — parsed raw is deleted immediately on parse-confirm,
# regardless of age (see prune_raw_if_parsed).
RAW_RETENTION_DAYS=3
PARQUET_RETENTION_DAYS=7
DISK_ALERT_PCT=90

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/data"
PARQUET_DIR="$DATA_DIR/parquet"
RAW_DIR="$DATA_DIR/raw"
LOG_DIR="$DATA_DIR/logs"
LOG_FILE="$LOG_DIR/sync_cloud.log"

mkdir -p "$LOG_DIR"

# keep the (append-only) sync log bounded so it never contributes to disk fill
if [ -f "$LOG_FILE" ] && [ "$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)" -gt 20000 ]; then
    tail -n 5000 "$LOG_FILE" > "$LOG_FILE.tmp" 2>/dev/null && mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"; }

# rclone options shared by both transfers:
#   --checksum : compare by content hash, not mtime+size
#   --transfers/--checkers : modest parallelism for many small files
#   --stats-one-line : compact periodic + final summary line
RCLONE_OPTS=(--checksum --transfers 8 --checkers 16 --stats-one-line --stats=1m)

# --- preflight ------------------------------------------------------------
command -v rclone >/dev/null 2>&1 || { log "ERROR: rclone not found in PATH"; exit 2; }

START=$SECONDS
UPLOAD_ERRORS=0
PRUNED_TOTAL=0
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

# prune_verified <local_dir> <remote_subpath> <glob> <retention_days>
#   Deletes local files older than <retention_days> ONLY when R2 already holds an
#   object at the same relative path with an identical byte size. Files not yet on
#   R2 (the in-progress current shard, or an upload that just failed) are kept and
#   logged. Never deletes anything from R2. Cleans up emptied date directories.
#   Fails safe in every direction: any uncertainty -> keep the local file.
prune_verified() {
    local local_dir="$1" remote_sub="$2" pattern="$3" days="$4"
    local_dir="${local_dir%/}"   # guard against a trailing slash breaking the prefix strip
    if [ ! -d "$local_dir" ]; then
        log "  prune skip: $local_dir does not exist"
        return 0
    fi

    # One R2 listing for this subtree. --format "ps" => col1=path, col2=size (tab-sep).
    # Do NOT reorder without updating the awk below.
    local manifest; manifest="$(mktemp)"
    if ! rclone lsf -R --files-only --format "ps" --separator $'\t' \
            "$REMOTE:$BUCKET/$remote_sub" > "$manifest" 2>>"$LOG_FILE"; then
        log "  prune ABORT ($remote_sub): could not list R2 — keeping ALL local files this run"
        rm -f "$manifest"
        return 0
    fi
    local mlines; mlines="$(wc -l < "$manifest" 2>/dev/null || echo 0)"

    local pruned=0 kept=0 rel lsize rsize
    while IFS= read -r -d '' f; do
        rel="${f#"$local_dir"/}"
        lsize="$(stat -c %s "$f" 2>/dev/null || echo -1)"
        rsize="$(awk -F'\t' -v p="$rel" '$1 == p { print $2; exit }' "$manifest")"
        if [ -n "$rsize" ] && [ "$rsize" = "$lsize" ]; then
            if rm -f "$f"; then pruned=$((pruned + 1)); else log "  WARN: failed to rm $rel"; fi
        else
            kept=$((kept + 1))
            log "  keep (not verified in R2): $remote_sub/$rel"
        fi
    done < <(find "$local_dir" -name "$pattern" -type f -mtime +"$days" -print0)

    # If R2 listed nothing but we have an old-file backlog, that is either a true
    # first run or an R2 read regression — flag it (otherwise it looks healthy).
    if [ "$mlines" -eq 0 ] && [ "$kept" -gt 0 ]; then
        log "  WARN ($remote_sub): R2 listing EMPTY but $kept local file(s) older than ${days}d exist — first run, or an R2 read problem. Investigate if it repeats."
    fi

    find "$local_dir" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    log "  $remote_sub: pruned $pruned verified file(s), kept $kept unverified (older than ${days}d)"
    PRUNED_TOTAL=$((PRUNED_TOTAL + pruned))
    rm -f "$manifest"
}

# prune_raw_if_parsed
#   Delete a LOCAL raw *.jsonl.gz the moment a NON-EMPTY parquet for that shard is
#   confirmed in R2 — no age requirement. This is the local mirror of the
#   expire_r2_raw.sh gate, and running it BEFORE the raw upload is what stops the
#   re-upload churn (parsed raw is gone locally, so `rclone copy` cannot re-send it).
#   Kept: any shard with no non-empty parquet in R2 — the live current-hour shard,
#   genuinely failed parses, and the parquet-less `unknown_*` universe (the latter two
#   are then bounded locally by the age-based backstop). Fails safe: if R2 parquet
#   cannot be listed, keep ALL local raw this run.
prune_raw_if_parsed() {
    if [ ! -d "$RAW_DIR" ]; then
        log "  raw(parse) prune skip: $RAW_DIR does not exist"
        return 0
    fi
    local pq_list; pq_list="$(mktemp)"
    if ! rclone lsf -R --files-only --format "ps" --separator $'\t' \
            "$REMOTE:$BUCKET/parquet" > "$pq_list" 2>>"$LOG_FILE"; then
        log "  raw(parse) prune ABORT: could not list R2 parquet — keeping ALL local raw this run"
        rm -f "$pq_list"
        return 0
    fi

    local pruned=0 kept=0 rel date base stem pqmatch
    while IFS= read -r -d '' f; do
        rel="${f#"$RAW_DIR"/}"                        # e.g. 2026-06-23/esports_08.jsonl.gz
        date="${rel%%/*}"
        case "$date" in
            [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
            *) kept=$((kept + 1)); continue ;;        # non-date-partitioned -> leave alone
        esac
        base="${rel##*/}"; stem="${base%.jsonl.gz}"   # e.g. esports_08
        # a NON-EMPTY parquet for this shard: parquet/{date}/.../*_{stem}.parquet
        pqmatch="$(awk -F'\t' -v d="$date" -v s="$stem" \
            '$1 ~ ("^" d "/.*_" s "\\.parquet$") && ($2 + 0) > 0 { print $1; exit }' "$pq_list")"
        if [ -n "$pqmatch" ]; then
            if rm -f "$f"; then pruned=$((pruned + 1)); else log "  WARN: failed to rm raw/$rel"; fi
        else
            kept=$((kept + 1))
        fi
    done < <(find "$RAW_DIR" -name '*.jsonl.gz' -type f -print0)

    find "$RAW_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    log "  raw(parse): pruned $pruned parsed file(s), kept $kept unparsed/unknown"
    PRUNED_TOTAL=$((PRUNED_TOTAL + pruned))
    rm -f "$pq_list"
}

# --- 1. parquet (working set) -> copy (never deletes remote) --------------
#   Upload parquet FIRST so step 2 can safely delete any local raw it now covers.
log "backing up parquet $PARQUET_DIR -> $REMOTE:$BUCKET/parquet  (copy, no remote deletes)"
if ! run_rclone copy "$PARQUET_DIR" "$REMOTE:$BUCKET/parquet"; then
    log "WARN: parquet upload had errors — raw prune keeps any shard whose parquet is not yet in R2"
    UPLOAD_ERRORS=$((UPLOAD_ERRORS + 1))
fi

# --- 2. prune LOCAL raw whose parquet is confirmed in R2 (any age) --------
#   MUST run before the raw upload so parsed raw is never (re-)uploaded — this is the
#   fix for the expire<->sync re-upload churn (2026-06-24).
log "pruning local raw whose parquet is confirmed in R2 (any age)"
prune_raw_if_parsed

# --- 3. raw (only the remaining UNPARSED shards) -> copy ------------------
log "backing up remaining unparsed raw $RAW_DIR -> $REMOTE:$BUCKET/raw  (copy, no remote deletes)"
if ! run_rclone copy "$RAW_DIR" "$REMOTE:$BUCKET/raw"; then
    log "WARN: raw upload had errors (expected: the live current-hour shard) — safe to retry next run"
    UPLOAD_ERRORS=$((UPLOAD_ERRORS + 1))
fi

# --- 4. BACKSTOP: prune local UNPARSED raw older than RAW_RETENTION_DAYS ---
#   Bounds local disk against unknown_*/failed-parse raw that will never get a parquet.
#   Per-file verified: only deletes raw already same-path/same-size in R2, so a failed
#   parse is preserved in R2 even after its local copy ages out.
log "backstop: pruning local raw older than ${RAW_RETENTION_DAYS}d that is verified in R2"
prune_verified "$RAW_DIR" "raw" '*.jsonl.gz' "$RAW_RETENTION_DAYS"

# --- 5. prune local parquet older than PARQUET_RETENTION_DAYS (verified) --
log "pruning local parquet *.parquet older than ${PARQUET_RETENTION_DAYS}d (only files verified in R2)"
prune_verified "$PARQUET_DIR" "parquet" '*.parquet' "$PARQUET_RETENTION_DAYS"

# --- 6. summary -----------------------------------------------------------
DUR=$((SECONDS - START))
PARQUET_BYTES=$( [ -d "$PARQUET_DIR" ] && du -sb "$PARQUET_DIR" 2>/dev/null | cut -f1 || echo 0 )
RAW_BYTES=$( [ -d "$RAW_DIR" ] && du -sb "$RAW_DIR" 2>/dev/null | cut -f1 || echo 0 )
PARQUET_FILES=$( [ -d "$PARQUET_DIR" ] && find "$PARQUET_DIR" -name '*.parquet' -type f | wc -l || echo 0 )
RAW_FILES=$( [ -d "$RAW_DIR" ] && find "$RAW_DIR" -name '*.jsonl.gz' -type f | wc -l || echo 0 )

log "transfer + prune summary (bytes transferred are in the rclone 'Transferred:' lines above)"
log "  local parquet: ${PARQUET_FILES} files, ${PARQUET_BYTES} bytes"
log "  local raw    : ${RAW_FILES} files, ${RAW_BYTES} bytes"
log "  pruned this run: ${PRUNED_TOTAL} local file(s); upload-error groups: ${UPLOAD_ERRORS}"
log "=== sync_cloud done in ${DUR}s ==="

# --- 7. disk high-water-mark guard (monitoring backstop) ------------------
DISK_PCT="$(df --output=pcent "$DATA_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ -n "$DISK_PCT" ] && [ "$DISK_PCT" -ge "$DISK_ALERT_PCT" ]; then
    log "CRITICAL: disk at ${DISK_PCT}% (>= ${DISK_ALERT_PCT}%) after sync+prune — backup/prune may be failing. Check R2 reachability and this log; do NOT delete data/raw or data/parquet by hand."
    exit 1
fi
exit 0
