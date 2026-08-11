#!/bin/bash
# Lightweight login-node wrapper. It resolves the exact campaign commit and
# submits one small prep/orchestrator job; journal validation, SHA256, tar, and
# all downstream submissions happen on that allocated compute node.

set -euo pipefail

ROOT="${EVSP_DR_ROOT:-$HOME/EVSP-DR}"
cd "$ROOT"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "[LAUNCH] this wrapper belongs on the login node, not inside a job" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "[LAUNCH] sbatch is unavailable; run this on Unicorn" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[LAUNCH] tracked worktree changes exist; pull/resolve before launching" >&2
  exit 2
fi
commit=$(git rev-parse HEAD)
job_id=$(sbatch --parsable src/submit_overnight_correctness_prep.sub "$commit")
job_id=${job_id%%;*}
echo "[LAUNCH] OCprep commit=${commit} job=${job_id}"
