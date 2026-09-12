#!/bin/bash
#SBATCH --job-name=storage-archive-0912
#SBATCH --partition=default_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --exclude=scaglione-compute-01
#SBATCH --output=/home/nc437/ladder-lite/storage_cleanup_20260912_archive/logs/%x_%j.out
#SBATCH --error=/home/nc437/ladder-lite/storage_cleanup_20260912_archive/logs/%x_%j.err

set -u -o pipefail
export LC_ALL=C

ROOT=/home/nc437/ladder-lite/storage_cleanup_20260912_archive
GROUP=${SLURM_ARRAY_TASK_ID:?missing array index}
MANIFEST=$ROOT/manifest_${GROUP}.tsv
STATUS=$ROOT/status_${GROUP}.tsv
QUEUE_AUDIT=$ROOT/queue_audit_$SLURM_JOB_ID.txt
REF_AUDIT=$ROOT/reference_audit_$SLURM_JOB_ID.txt
JOB_IDS=$ROOT/job_ids_$SLURM_JOB_ID.txt
SLURM_BIN=/usr/local/slurm/slurm-25.05.5/bin

mkdir -p "$ROOT/logs"
cd "$ROOT" || exit 2
sha256sum -c CODE_SHA256SUMS || exit 2
test -s "$MANIFEST" || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock unavailable" >&2; exit 3; }
exec 9>"$ROOT/archive_worker_${GROUP}.lock"
flock -n 9 || { echo "another archive worker holds the lock" >&2; exit 8; }

fsync_file() {
    python3 -c 'import os,sys; fd=os.open(sys.argv[1], os.O_RDONLY); os.fsync(fd); os.close(fd)' "$1"
}
fsync_dir() {
    python3 -c 'import os,sys; fd=os.open(sys.argv[1], os.O_RDONLY); os.fsync(fd); os.close(fd)' "$1"
}
if [ ! -s "$STATUS" ]; then
    printf 'source_path\tphase\tsource_sha256\tsource_size_bytes\tarchive_path\tarchive_sha256\tarchive_size_bytes\tevent_utc\n' > "$STATUS"
    fsync_file "$STATUS" || exit 3
    fsync_dir "$ROOT" || exit 3
fi

# Fail closed on dependency-check errors; the four archive workers are
# recognized explicitly by the read-only auditor and excluded from self-hits.
python3 "$ROOT/audit_remote.py" > "$ROOT/runtime_audit_${SLURM_ARRAY_JOB_ID}_${GROUP}.json" || exit 4
python3 - "$ROOT/runtime_audit_${SLURM_ARRAY_JOB_ID}_${GROUP}.json" <<'PY_AUDIT'
import json,sys
x=json.load(open(sys.argv[1]))
if x['errors'] or x['references']:
    print(json.dumps({'errors':x['errors'],'references':x['references']}))
    raise SystemExit(9)
PY_AUDIT
[ "$?" = 0 ] || exit 9

append_event() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >> "$STATUS" || return 1
    fsync_file "$STATUS"
}
latest_row() {
    awk -F '\t' -v p="$1" 'NR > 1 && $1 == p && ($2 == "prepared" || $2 == "success") {line=$0} END {print line}' "$STATUS"
}
verify_archive() {
    local arc="$1" source_sha="$2" archive_sha="$3" got decomp
    [ -f "$arc" ] || return 1
    got=$(sha256sum "$arc" | cut -d' ' -f1) || return 1
    [ "$got" = "$archive_sha" ] || return 1
    gzip -t -- "$arc" || return 1
    decomp=$(gzip -cd -- "$arc" | sha256sum | cut -d' ' -f1) || return 1
    [ "$decomp" = "$source_sha" ]
}
record_failure() {
    append_event "$1" failed "" "$2" "$3" "" "" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

archive_one() {
    local src="$1" expected_size="$2" expected_mtime="$3" arc="$4"
    local row phase source_sha source_size status_arc archive_sha archive_size event
    local tmp="$arc.part.$SLURM_JOB_ID"
    local before sha decomp after arcsha arcstat
    case "$src" in
      /home/nc437/ladder-lite/phys240kw/*.columns.jsonl|/home/nc437/ladder-lite/cg_acceleration_20260903/cg/*/*.columns.jsonl|/home/nc437/ladder-lite/cg_acceleration_20260903/network_cache/*.pkl) ;;
      *) echo "source_outside_allowlist=$src" >&2; return 1 ;;
    esac
    [ "$arc" = "$ROOT/${src#/home/nc437/ladder-lite/}.gz" ] || return 1
    mkdir -p "$(dirname "$arc")" || return 1

    row=$(latest_row "$src")
    if [ -n "$row" ]; then
        [ "$(printf '%s\n' "$row" | awk -F '\t' '{print NF}')" = 8 ] || return 1
        IFS=$'\t' read -r status_src phase source_sha source_size status_arc archive_sha archive_size event <<< "$row"
        [ "$status_src" = "$src" ] && [ "$status_arc" = "$arc" ] || return 1
        if [ "$phase" = success ]; then
            [ ! -e "$src" ] && [ ! -L "$src" ] || { echo "source_reappeared_after_success=$src" >&2; return 1; }
            verify_archive "$status_arc" "$source_sha" "$archive_sha"
            return $?
        fi
        if [ "$phase" = prepared ]; then
            verify_archive "$status_arc" "$source_sha" "$archive_sha" || return 1
            if [ -e "$src" ]; then
                [ -f "$src" ] && [ ! -L "$src" ] || return 1
                [ "$(stat -c %h "$src")" = 1 ] || return 1
                [ "$(stat -c '%s' "$src")" = "$source_size" ] || return 1
                [ "$(sha256sum "$src" | cut -d' ' -f1)" = "$source_sha" ] || return 1
                rm -- "$src" || return 1
                fsync_dir "$(dirname "$src")" || return 1
            fi
            [ ! -e "$src" ] || return 1
            append_event "$src" success "$source_sha" "$source_size" "$status_arc" "$archive_sha" "$archive_size" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" || return 1
            return 0
        fi
    fi

    [ -e "$src" ] || { echo "source_missing_without_durable_record=$src" >&2; return 1; }
    [ -f "$src" ] && [ ! -L "$src" ] || return 1
    [ "$(stat -c %h "$src")" = 1 ] || { echo "hardlinked_source=$src" >&2; return 1; }
    [ ! -e "$arc" ] || { echo "archive_collision_without_durable_record=$arc" >&2; return 1; }
    rm -f -- "$tmp"
    before=$(stat -c '%s|%Y|%i|%d' "$src") || return 1
    case "$before" in
        "$expected_size|$expected_mtime|"*) ;;
        *) echo "manifest_stat_mismatch=$src before=$before expected=$expected_size|$expected_mtime" >&2; return 1 ;;
    esac
    sha=$(sha256sum "$src" | cut -d' ' -f1) || return 1
    gzip -1 -c -- "$src" > "$tmp" || { rm -f -- "$tmp"; return 1; }
    fsync_file "$tmp" || { rm -f -- "$tmp"; return 1; }
    gzip -t -- "$tmp" || { rm -f -- "$tmp"; return 1; }
    decomp=$(gzip -cd -- "$tmp" | sha256sum | cut -d' ' -f1) || { rm -f -- "$tmp"; return 1; }
    [ "$sha" = "$decomp" ] || { echo "decompressed_hash_mismatch=$src" >&2; rm -f -- "$tmp"; return 1; }
    after=$(stat -c '%s|%Y|%i|%d' "$src") || { rm -f -- "$tmp"; return 1; }
    [ "$before" = "$after" ] || { echo "source_changed=$src" >&2; rm -f -- "$tmp"; return 1; }
    mv -- "$tmp" "$arc" || { rm -f -- "$tmp"; return 1; }
    fsync_dir "$(dirname "$arc")" || return 1
    arcsha=$(sha256sum "$arc" | cut -d' ' -f1) || return 1
    arcstat=$(stat -c '%s' "$arc") || return 1
    # Durable prepared record precedes source unlink. Resume reconciles it.
    append_event "$src" prepared "$sha" "$(printf '%s' "$before" | cut -d'|' -f1)" "$arc" "$arcsha" "$arcstat" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" || return 1
    [ "$(sha256sum "$src" | cut -d' ' -f1)" = "$sha" ] || return 1
    [ "$(stat -c '%s|%Y|%i|%d' "$src")" = "$before" ] || return 1
    rm -- "$src" || return 1
    fsync_dir "$(dirname "$src")" || return 1
    [ ! -e "$src" ] || return 1
    append_event "$src" success "$sha" "$(printf '%s' "$before" | cut -d'|' -f1)" "$arc" "$arcsha" "$arcstat" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" || return 1
    return 0
}

success_count=0
failure_count=0
while IFS=$'\t' read -r src expected_size expected_mtime arc; do
    [ "$src" = source_path ] && continue
    if archive_one "$src" "$expected_size" "$expected_mtime" "$arc"; then
        success_count=$((success_count + 1))
    else
        failure_count=$((failure_count + 1))
        record_failure "$src" "$expected_size" "$arc"
    fi
done < "$MANIFEST"

printf 'archive_job=%s\tsuccess_count=%s\tfailure_count=%s\tstatus=%s\n' "$SLURM_JOB_ID" "$success_count" "$failure_count" "$STATUS"
[ "$failure_count" -eq 0 ]
