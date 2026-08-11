#!/bin/bash
# Archive the evidence after jobs finish.  This is intentionally
# non-destructive and intentionally refuses to run on a login node; submit
# submit_overnight_correctness_collect.sub instead.

set -euo pipefail
ROOT="${EVSP_DR_ROOT:-$HOME/EVSP-DR}"
cd "$ROOT"
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "refuse to hash/compress on a login node" >&2
  echo "use: sbatch src/submit_overnight_correctness_collect.sub" >&2
  exit 2
fi
ARCHIVE_ID="${EVSP_ARCHIVE_ID:-job${SLURM_JOB_ID}}"
DEST="${EVSP_ARCHIVE_DIR:-$HOME}/evsp_overnight_correctness_${ARCHIVE_ID}.tar.gz"

paths=()
for path in \
  src/results/master_audit \
  src/results/stopping_mip \
  src/results/stopping_controls \
  src/results/campaign_manifests \
  src/cluster_logs/overnight_correctness; do
  if [ -e "$path" ]; then paths+=("$path"); fi
done
if [ "${#paths[@]}" -eq 0 ]; then
  echo "no overnight correctness artifacts found" >&2
  exit 2
fi

partial="${DEST}.partial.${SLURM_JOB_ID}"
checksum_partial="${DEST}.sha256.partial.${SLURM_JOB_ID}"
tar -czf "$partial" "${paths[@]}"
mv -f "$partial" "$DEST"
sha256sum "$DEST" > "$checksum_partial"
mv -f "$checksum_partial" "${DEST}.sha256"
cat "${DEST}.sha256"
echo "archive: $DEST"
echo "copy this archive and .sha256 to the authenticated Mac for the GitHub release"
