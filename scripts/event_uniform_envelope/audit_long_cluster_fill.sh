#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 2 ]] || evsp_die "usage: $0 PANEL_A_ROOT PANEL_B_ROOT"
A_ROOT=$(cd "$1" && pwd)
B_ROOT=$(cd "$2" && pwd)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
A_RESUME_ROOT="$A_ROOT/cg_resume24h_2dd2b4c"
B_RESUME_ROOT="$B_ROOT/cg_certification6h_13596d0"

collect_accounting() {
  local root="$1"
  local resume_root="$2"
  local output="$3"
  local files=()
  local path
  while IFS= read -r path; do files+=("$path"); done < <(
    find "$root" -maxdepth 1 -type f -name 'highs_native*_jobs.tsv' -print | sort
  )
  while IFS= read -r path; do files+=("$path"); done < <(
    find "$resume_root" -maxdepth 1 -type f -name 'jobs_*.tsv' -print | sort
  )
  [[ ${#files[@]} -gt 0 ]] || evsp_die "no long-fill job records under $root"
  mapfile -t ids < <(awk -F'\t' 'FNR > 1 && $2 != "" {print $2}' "${files[@]}" | sort -u)
  [[ ${#ids[@]} -gt 0 ]] || evsp_die "long-fill job records contain no IDs"
  local job_list
  job_list=$(IFS=,; echo "${ids[*]}")
  sacct -j "$job_list" -n -P \
    -o JobIDRaw,JobName%40,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,ReqMem,NodeList \
    > "$output"
}

collect_accounting "$A_ROOT" "$A_RESUME_ROOT" "$A_ROOT/long_fill_slurm_accounting.psv"
collect_accounting "$B_ROOT" "$B_RESUME_ROOT" "$B_ROOT/long_fill_slurm_accounting.psv"

"$PYTHON_BIN" "$SCRIPT_DIR/audit_cg_resume.py" \
  --resume-root "$A_RESUME_ROOT" --sacct "$A_ROOT/long_fill_slurm_accounting.psv"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_cg_resume.py" \
  --resume-root "$B_RESUME_ROOT" --sacct "$B_ROOT/long_fill_slurm_accounting.psv"
if [[ -f "$A_ROOT/panel_a_highs_inputs.tsv" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/audit_backend_reproduction.py" \
    --root "$A_ROOT" --panel A --manifest "$A_ROOT/panel_a_highs_inputs.tsv" \
    --sacct "$A_ROOT/long_fill_slurm_accounting.psv" \
    --out "$A_ROOT/backend_reproduction.csv"
else
  echo "native-HiGHS Panel A manifest absent; skipping backend CSV" >&2
fi
if [[ -f "$B_ROOT/panel_b_highs_inputs.tsv" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/audit_backend_reproduction.py" \
    --root "$B_ROOT" --panel B --manifest "$B_ROOT/panel_b_highs_inputs.tsv" \
    --sacct "$B_ROOT/long_fill_slurm_accounting.psv" \
    --out "$B_ROOT/backend_reproduction.csv"
else
  echo "native-HiGHS Panel B manifest absent; skipping backend CSV" >&2
fi

A_SUMMARY_FILES=(
  "$A_RESUME_ROOT/resume_summary.csv"
  "$A_ROOT/long_fill_slurm_accounting.psv"
)
B_SUMMARY_FILES=(
  "$B_RESUME_ROOT/resume_summary.csv"
  "$B_ROOT/long_fill_slurm_accounting.psv"
)
[[ ! -f "$A_ROOT/backend_reproduction.csv" ]] \
  || A_SUMMARY_FILES+=("$A_ROOT/backend_reproduction.csv")
[[ ! -f "$B_ROOT/backend_reproduction.csv" ]] \
  || B_SUMMARY_FILES+=("$B_ROOT/backend_reproduction.csv")
sha256sum "${A_SUMMARY_FILES[@]}" > "$A_ROOT/LONG_FILL_SUMMARY_SHA256SUMS"
sha256sum "${B_SUMMARY_FILES[@]}" > "$B_ROOT/LONG_FILL_SUMMARY_SHA256SUMS"

echo "Panel A CSVs: $A_RESUME_ROOT/resume_summary.csv ; $A_ROOT/backend_reproduction.csv"
echo "Panel B CSVs: $B_RESUME_ROOT/resume_summary.csv ; $B_ROOT/backend_reproduction.csv"
