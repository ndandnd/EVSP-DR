#!/bin/bash
#SBATCH --job-name=storage-archive
#SBATCH --partition=default_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --exclude=scaglione-compute-01
#SBATCH --output=/home/nc437/ladder-lite/storage_cleanup_20260911_archive/logs/%x_%j.out
#SBATCH --error=/home/nc437/ladder-lite/storage_cleanup_20260911_archive/logs/%x_%j.err

set -u -o pipefail
export LC_ALL=C

ROOT=/home/nc437/ladder-lite/storage_cleanup_20260911_archive
MANIFEST=$ROOT/remaining_manifest.tsv
STATUS=$ROOT/ARCHIVED_POOLS_STATUS.tsv
QUEUE_AUDIT=$ROOT/queue_audit_$SLURM_JOB_ID.txt
REF_AUDIT=$ROOT/reference_audit_$SLURM_JOB_ID.txt
JOB_IDS=$ROOT/job_ids_$SLURM_JOB_ID.txt
SLURM_BIN=/usr/local/slurm/slurm-25.05.5/bin

mkdir -p "$ROOT/logs" "$ROOT/finetime" "$ROOT/factfill"
test -s "$MANIFEST" || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
touch "$STATUS"

# Abort if any queued job currently names either source root in its command,
# working directory, or output paths. This protects a newly submitted job too.
"$SLURM_BIN/squeue" --me -h -o '%A|%j|%Z' > "$QUEUE_AUDIT" || exit 3
: > "$REF_AUDIT"
"$SLURM_BIN/squeue" --me -h -o '%A' | sort -u > "$JOB_IDS" || exit 3
while IFS= read -r job_id; do
    [ -n "$job_id" ] || continue
    "$SLURM_BIN/scontrol" show job "$job_id" 2>/dev/null |
        tr ' ' '\n' |
        grep -E '^(Command|WorkDir|StdOut|StdErr)=.*(finetime|factfill)' >> "$REF_AUDIT" || true
done < "$JOB_IDS"
if [ -s "$REF_AUDIT" ]; then
    echo "active/pending/held reference detected; refusing archive" >&2
    cat "$REF_AUDIT" >&2
    exit 9
fi

record_failure() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tfailed\n' "$1" "$2" "$3" "$4" "$5" "$6" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
}

status_success() {
    awk -F '\t' -v p="$1" '$1 == p && $8 == "success" {ok=1} END {exit ok ? 0 : 1}' "$STATUS"
}

archive_one() {
    local src="$1" expected_size="$2" expected_mtime="$3" arc="$4"
    local tmp="$arc.part.$SLURM_JOB_ID"
    local before sha decomp after arcsha arcstat

    status_success "$src" && return 0
    [ -e "$src" ] || { echo "source_missing=$src" >&2; return 1; }
    [ ! -e "$arc" ] || { echo "archive_collision=$arc" >&2; return 1; }
    rm -f -- "$tmp"
    before=$(stat -c '%s|%Y|%i|%d' "$src") || return 1
    case "$before" in
        "$expected_size|$expected_mtime|"*) ;;
        *) echo "manifest_stat_mismatch=$src before=$before expected=$expected_size|$expected_mtime" >&2; return 1 ;;
    esac
    sha=$(sha256sum "$src" | cut -d' ' -f1) || return 1
    gzip -1 -c -- "$src" > "$tmp" || { rm -f -- "$tmp"; return 1; }
    python3 -c 'import os,sys; fd=os.open(sys.argv[1], os.O_RDONLY); os.fsync(fd); os.close(fd)' "$tmp" || { rm -f -- "$tmp"; return 1; }
    gzip -t -- "$tmp" || { rm -f -- "$tmp"; return 1; }
    decomp=$(gzip -cd -- "$tmp" | sha256sum | cut -d' ' -f1) || { rm -f -- "$tmp"; return 1; }
    [ "$sha" = "$decomp" ] || { echo "decompressed_hash_mismatch=$src" >&2; rm -f -- "$tmp"; return 1; }
    after=$(stat -c '%s|%Y|%i|%d' "$src") || { rm -f -- "$tmp"; return 1; }
    [ "$before" = "$after" ] || { echo "source_changed=$src" >&2; rm -f -- "$tmp"; return 1; }
    mv -- "$tmp" "$arc" || { rm -f -- "$tmp"; return 1; }
    python3 -c 'import os,sys; fd=os.open(sys.argv[1], os.O_RDONLY); os.fsync(fd); os.close(fd)' "$(dirname "$arc")" || return 1
    arcsha=$(sha256sum "$arc" | cut -d' ' -f1) || return 1
    arcstat=$(stat -c '%s' "$arc") || return 1
    rm -- "$src" || return 1
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tsuccess\n' "$src" "$sha" "$(printf '%s' "$before" | cut -d'|' -f1)" "$arc" "$arcsha" "$arcstat" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
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
        record_failure "$src" "" "$expected_size" "$arc" "" ""
    fi
done < "$MANIFEST"

printf 'archive_job=%s\tsuccess_count=%s\tfailure_count=%s\tstatus=%s\n' "$SLURM_JOB_ID" "$success_count" "$failure_count" "$STATUS"
[ "$failure_count" -eq 0 ]
