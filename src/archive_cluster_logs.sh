#!/bin/bash
# Move completed legacy Slurm stdout/stderr files into one recoverable archive.
# The script refuses to run while this user's jobs are active.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if command -v squeue >/dev/null 2>&1; then
    if ! active_jobs=$(squeue -h -u "$USER" -o '%A'); then
        echo "[STOP] Could not verify Slurm job state; no logs were moved." >&2
        exit 2
    fi
    if [[ -n "$active_jobs" ]]; then
        echo "[STOP] Slurm jobs are still active; no logs were moved." >&2
        exit 2
    fi
fi

STAMP=$(date +%Y%m%d_%H%M%S)_$$
ARCHIVE=${1:-"$HOME/evspdr-slurm-logs-$STAMP"}
if [[ -e "$ARCHIVE" || -e "${ARCHIVE}_SHA256SUMS" ]]; then
    echo "[STOP] Archive target already exists; no logs were moved: $ARCHIVE" >&2
    exit 2
fi
mkdir -p "$ARCHIVE/root" "$ARCHIVE/src"

find "$ROOT" -maxdepth 1 -type f \
    \( -name '*.out' -o -name '*.err' \) \
    -exec mv -n -- {} "$ARCHIVE/root/" \;
find "$ROOT/src" -maxdepth 1 -type f \
    \( -name '*.out' -o -name '*.err' \) \
    -exec mv -n -- {} "$ARCHIVE/src/" \;

find "$ARCHIVE" -type f -print0 | sort -z | xargs -0 -r sha256sum \
    > "${ARCHIVE}_SHA256SUMS"
echo "[archived] $ARCHIVE"
echo "[checksums] ${ARCHIVE}_SHA256SUMS"
