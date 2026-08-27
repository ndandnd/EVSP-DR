#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 1 ]] || evsp_die "usage: $0 PANEL_B_ROOT"
B_ROOT=$(cd "$1" && pwd)
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
SOLVER_COMMIT="13596d0f03c70b9caf406db06e5a27c8ad4fbe8f"
AGENT_BRANCH="cursor/event-based-pricer-2969"
REMOTE=$(git -C "$REPO" ls-remote --heads origin "${AGENT_BRANCH}*")
printf '%s\n' "$REMOTE" >&2
[[ "$(printf '%s\n' "$REMOTE" | awk 'NF {n++} END {print n+0}')" == 1 ]] \
  || evsp_die "expected exactly one Agent E branch"
[[ "$(printf '%s\n' "$REMOTE" | awk '{print $2}')" == "refs/heads/$AGENT_BRANCH" ]] \
  || evsp_die "unexpected Agent E ref"
AGENT_SHA=$(printf '%s\n' "$REMOTE" | awk '{print $1}')
git -C "$REPO" cat-file -e "$SOLVER_COMMIT^{commit}" \
  || evsp_die "historical Panel B solver commit is unavailable"
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
PARENT="$B_ROOT/cg_certification6h_13596d0"
RESUME="$B_ROOT/cg_certification24h_13596d0"
PARENT_CHECK=$(mktemp)
trap 'rm -f "$PARENT_CHECK" "${INDEX_FILE:-}"' EXIT
"$PYTHON_BIN" "$SCRIPT_DIR/select_cg_resume_indices.py" \
  --resume-root "$PARENT" --expected-panel B \
  --expected-commit "$SOLVER_COMMIT" --expected-wall-limit-s 21600 \
  > "$PARENT_CHECK"
[[ ! -s "$PARENT_CHECK" ]] \
  || evsp_die "six-hour parent still contains incomplete, uncapped rows"
if [[ ! -e "$RESUME" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_cg_reresume.py" \
    --source-resume-root "$PARENT" --out-root "$RESUME" --panel B \
    --representation uniform_2_1 --expected-cells 2 \
    --wall-limit-s 86400 --max-iters 50000 \
    --solver-commit "$SOLVER_COMMIT"
fi
"$PYTHON_BIN" "$SCRIPT_DIR/repair_cg_resume_telemetry.py" \
  --resume-root "$RESUME"
INDEX_FILE=$(mktemp)
"$PYTHON_BIN" "$SCRIPT_DIR/select_cg_resume_indices.py" \
  --resume-root "$RESUME" --expected-panel B \
  --expected-commit "$SOLVER_COMMIT" --expected-wall-limit-s 86400 \
  > "$INDEX_FILE"
mapfile -t INDICES < "$INDEX_FILE"
if [[ ${#INDICES[@]} == 0 ]]; then
  echo "Panel B 24h CG tail has no pending cells"
  exit 0
fi
ACTIVE=$(squeue --me -h -o '%A|%j' | awk -F'|' '$2 == "eub27_r24" {print $1}' | sort -u)
[[ "$(printf '%s\n' "$ACTIVE" | awk 'NF {n++} END {print n+0}')" -le 1 ]] \
  || evsp_die "multiple active eub27_r24 jobs"
if [[ -n "$ACTIVE" ]]; then
  echo "Panel B 24h CG tail already active: $ACTIVE"
  exit 0
fi
ARRAY=$(IFS=,; echo "${INDICES[*]}")
EXPORTS="ALL,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$RESUME,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_CUMULATIVE_WALL_LIMIT_S=86400,EVSP_MAX_ITERS=50000,EVSP_PYTHON=$PYTHON_BIN"
JOB=$(evsp_submit_and_resolve eub27_r24 \
  --array="$ARRAY%2" -p default_partition -c 2 --mem=24G \
  -t 18:15:00 --signal=B:USR1@180 --no-requeue \
  --export="$EXPORTS" \
  -o "$RESUME/logs/resume_%A_%a.out" \
  -e "$RESUME/logs/resume_%A_%a.err" \
  "$SCRIPT_DIR/cg_resume_extended.sub")
{
  echo -e "stage\tarray_job_id\ttasks\twrapper_commit\tsolver_commit\tagent_tip_observed\tcumulative_wall_limit_s\tcategory\tpartition"
  echo -e "panel_b_certification24h\t$JOB\t${#INDICES[@]}\t$WRAPPER_COMMIT\t$SOLVER_COMMIT\t$AGENT_SHA\t86400\textended_cg\tdefault_partition"
} > "$RESUME/jobs_${JOB}.tsv"
sha256sum "$RESUME/execution_plan.json" "$RESUME/matrix.tsv" \
  > "$RESUME/SUBMISSION_INPUT_SHA256SUMS"
echo "Panel B 24h CG tail: $JOB (${#INDICES[@]} tasks)"
