#!/bin/bash
# Move completed legacy Slurm stdout/stderr files into one recoverable archive.
# The script refuses to run while this user's jobs are active.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if command -v squeue >/dev/null 2>&1 &&
   [[ -n "$(squeue -h -u "$USER" -o '%A' 2>/dev/null)" ]]; then
    echo "[STOP] Slurm jobs are still active; no logs were moved." >&2
    exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE=${1:-"$HOME/evspdr-slurm-logs-$STAMP"}
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
