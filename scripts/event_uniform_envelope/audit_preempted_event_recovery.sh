#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 2 ]] || evsp_die "usage: $0 [MEDIUM_ROOT] [EXTENSION_ROOT]"

MEDIUM_ROOT="${1:-$HOME/ladder-lite/medium_event_legacy_20260830_44b6d5}"
EXTENSION_ROOT="${2:-$HOME/ladder-lite/event_extension_20260830_44b6d5}"
MEDIUM_RESUME="$(cd "$MEDIUM_ROOT/preempt_recovery_i12_20260830" && pwd)"
EXTENSION_RESUME="$(cd "$EXTENSION_ROOT/preempt_recovery_i12_20260830" && pwd)"
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"

audit_one() {
  local root="$1" label="$2" job_list sacct_file
  local -a job_ids
  mapfile -t job_ids < <(
    awk -F'\t' 'FNR > 1 {print $2}' "$root"/jobs_*.tsv | sort -u
  )
  [[ ${#job_ids[@]} -ge 1 ]] || evsp_die "no recovery jobs for $label"
  job_list=$(IFS=,; echo "${job_ids[*]}")
  if squeue --me -h -j "$job_list" | grep -q .; then
    evsp_die "$label recovery is still active"
  fi
  sacct_file="$root/slurm_accounting.psv"
  sacct -j "$job_list" -n -P \
    -o JobID%48,JobIDRaw,JobName%40,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,NodeList \
    > "$sacct_file"
  "$PYTHON_BIN" "$SCRIPT_DIR/audit_cg_resume.py" \
    --resume-root "$root" --sacct "$sacct_file"
  sha256sum "$root/execution_plan.json" "$root/matrix.tsv" \
    "$sacct_file" "$root/resume_summary.csv" \
    > "$root/AUDIT_SHA256SUMS"
}

audit_one "$MEDIUM_RESUME" medium_k20
audit_one "$EXTENSION_RESUME" extension_k40

echo "Medium recovery CSV: $MEDIUM_RESUME/resume_summary.csv"
echo "Extension recovery CSV: $EXTENSION_RESUME/resume_summary.csv"
