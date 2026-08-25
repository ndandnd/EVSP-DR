#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 2 ]] || evsp_die "usage: $0 PANEL_A_ROOT PANEL_B_ROOT"

A_ROOT=$(cd "$1" && pwd)
B_ROOT=$(cd "$2" && pwd)
REPO=$(evsp_repo_root)
MANAGER_BRANCH=$(git -C "$REPO" branch --show-current)
[[ -n "$MANAGER_BRANCH" ]] || evsp_die "manager checkout must be on a named branch"
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$MANAGER_BRANCH" | tail -1)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"

A_RESUME_COMMIT="2dd2b4cd81fb15da137f6d443f5a495e22fd0255"
B_RESUME_COMMIT="13596d0f03c70b9caf406db06e5a27c8ad4fbe8f"
HIGHS_COMMIT="44b6d5030a78ddca9c74f582d70ad87572e61794"
AGENT_BRANCH="cursor/event-based-pricer-2969"

REMOTE=$(git -C "$REPO" ls-remote --heads origin "${AGENT_BRANCH}*") \
  || evsp_die "could not resolve Agent E branch"
printf '%s\n' "$REMOTE" >&2
[[ "$(printf '%s\n' "$REMOTE" | awk 'NF {n++} END {print n+0}')" == 1 ]] \
  || evsp_die "expected exactly one Agent E branch"
[[ "$(printf '%s\n' "$REMOTE" | awk '{print $2}')" == "refs/heads/$AGENT_BRANCH" ]] \
  || evsp_die "unexpected Agent E ref"
AGENT_SHA=$(printf '%s\n' "$REMOTE" | awk '{print $1}')
git -C "$REPO" fetch origin \
  "refs/heads/$AGENT_BRANCH:refs/remotes/origin/$AGENT_BRANCH"
git -C "$REPO" merge-base --is-ancestor "$HIGHS_COMMIT" "$AGENT_SHA" \
  || evsp_die "reviewed HiGHS commit is not an ancestor of Agent E tip"
for commit in "$A_RESUME_COMMIT" "$B_RESUME_COMMIT" "$HIGHS_COMMIT"; do
  git -C "$REPO" cat-file -e "$commit^{commit}" \
    || evsp_die "required commit is unavailable: $commit"
done

A_EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$A_RESUME_COMMIT")
B_EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$B_RESUME_COMMIT")
HIGHS_EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$HIGHS_COMMIT")

queued_job_id() {
  local name="$1"
  local ids
  ids=$(squeue --me -h -o '%A|%j' | awk -F'|' -v name="$name" '$2 == name {print $1}' | sort -u)
  local count
  count=$(printf '%s\n' "$ids" | awk 'NF {n++} END {print n+0}')
  [[ "$count" -le 1 ]] || evsp_die "multiple active jobs named $name"
  printf '%s\n' "$ids"
}

A_RESUME_ROOT="$A_ROOT/cg_resume24h_2dd2b4c"
B_RESUME_ROOT="$B_ROOT/cg_certification6h_13596d0"
if [[ ! -e "$A_RESUME_ROOT" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_cg_resume.py" \
    --root "$A_ROOT" --out-root "$A_RESUME_ROOT" --panel A \
    --representation uniform_2_1 --expected-cells 2 \
    --wall-limit-s 86400 --solver-commit "$A_RESUME_COMMIT"
fi
if [[ ! -e "$B_RESUME_ROOT" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_cg_resume.py" \
    --root "$B_ROOT" --out-root "$B_RESUME_ROOT" --panel B \
    --expected-cells 18 --wall-limit-s 21600 \
    --solver-commit "$B_RESUME_COMMIT"
fi

A_RESUME_INDEX_FILE=$(mktemp)
B_RESUME_INDEX_FILE=$(mktemp)
trap 'rm -f "$A_RESUME_INDEX_FILE" "$B_RESUME_INDEX_FILE" "${A_HIGHS_INDEX_FILE:-}" "${B_HIGHS_INDEX_FILE:-}"' EXIT
"$PYTHON_BIN" "$SCRIPT_DIR/select_cg_resume_indices.py" \
  --resume-root "$A_RESUME_ROOT" --expected-panel A \
  --expected-commit "$A_RESUME_COMMIT" --expected-wall-limit-s 86400 \
  > "$A_RESUME_INDEX_FILE"
"$PYTHON_BIN" "$SCRIPT_DIR/select_cg_resume_indices.py" \
  --resume-root "$B_RESUME_ROOT" --expected-panel B \
  --expected-commit "$B_RESUME_COMMIT" --expected-wall-limit-s 21600 \
  > "$B_RESUME_INDEX_FILE"
mapfile -t A_RESUME_INDICES < "$A_RESUME_INDEX_FILE"
mapfile -t B_RESUME_INDICES < "$B_RESUME_INDEX_FILE"

A_RESUME_JOB=""
A_ACTIVE_JOB=$(queued_job_id eua25_r24)
if [[ -n "$A_ACTIVE_JOB" ]]; then
  A_RESUME_JOB="$A_ACTIVE_JOB (already active)"
elif [[ ${#A_RESUME_INDICES[@]} -gt 0 ]]; then
  A_ARRAY=$(IFS=,; echo "${A_RESUME_INDICES[*]}")
  A_EXPORTS="ALL,EVSP_EXECUTION_REPO=$A_EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$A_RESUME_ROOT,EVSP_EXPECTED_COMMIT=$A_RESUME_COMMIT,EVSP_CUMULATIVE_WALL_LIMIT_S=86400,EVSP_PYTHON=$PYTHON_BIN"
  A_RESUME_JOB=$(evsp_submit_and_resolve eua25_r24 \
    --array="${A_ARRAY}%2" -p default_partition -c 2 --mem=24G \
    -t 12:15:00 --signal=B:USR1@180 --no-requeue \
    --export="$A_EXPORTS" \
    -o "$A_RESUME_ROOT/logs/resume_%A_%a.out" \
    -e "$A_RESUME_ROOT/logs/resume_%A_%a.err" \
    "$SCRIPT_DIR/cg_resume_extended.sub")
  {
    echo -e "stage\tarray_job_id\ttasks\tsolver_commit\tcumulative_wall_limit_s\tcategory"
    echo -e "panel_a_resume24h\t$A_RESUME_JOB\t${#A_RESUME_INDICES[@]}\t$A_RESUME_COMMIT\t86400\textended_cg"
  } > "$A_RESUME_ROOT/jobs_${A_RESUME_JOB}.tsv"
fi

B_RESUME_JOB=""
B_ACTIVE_JOB=$(queued_job_id eub25_r6)
if [[ -n "$B_ACTIVE_JOB" ]]; then
  B_RESUME_JOB="$B_ACTIVE_JOB (already active)"
elif [[ ${#B_RESUME_INDICES[@]} -gt 0 ]]; then
  B_ARRAY=$(IFS=,; echo "${B_RESUME_INDICES[*]}")
  B_EXPORTS="ALL,EVSP_EXECUTION_REPO=$B_EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$B_RESUME_ROOT,EVSP_EXPECTED_COMMIT=$B_RESUME_COMMIT,EVSP_CUMULATIVE_WALL_LIMIT_S=21600,EVSP_PYTHON=$PYTHON_BIN"
  B_RESUME_JOB=$(evsp_submit_and_resolve eub25_r6 \
    --array="${B_ARRAY}%18" -p default_partition -c 2 --mem=24G \
    -t 06:15:00 --signal=B:USR1@180 --no-requeue \
    --export="$B_EXPORTS" \
    -o "$B_RESUME_ROOT/logs/resume_%A_%a.out" \
    -e "$B_RESUME_ROOT/logs/resume_%A_%a.err" \
    "$SCRIPT_DIR/cg_resume_extended.sub")
  {
    echo -e "stage\tarray_job_id\ttasks\tsolver_commit\tcumulative_wall_limit_s\tcategory"
    echo -e "panel_b_certification6h\t$B_RESUME_JOB\t${#B_RESUME_INDICES[@]}\t$B_RESUME_COMMIT\t21600\textended_cg"
  } > "$B_RESUME_ROOT/jobs_${B_RESUME_JOB}.tsv"
fi

PYTHON_TAG=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
HIGHS_VENDOR_ROOT="$HOME/ladder-lite/vendor"
HIGHS_TARGET="$HIGHS_VENDOR_ROOT/highspy-1.15.1-py${PYTHON_TAG}"
mkdir -p "$HIGHS_VENDOR_ROOT"
set +e
HIGHS_VERSION=$(PYTHONPATH="$HIGHS_TARGET" "$PYTHON_BIN" -c \
  'import highspy; print(highspy.Highs().version()); print(highspy.__file__)' 2>&1)
HIGHS_RC=$?
set -e
if [[ "$HIGHS_RC" != 0 ]]; then
  [[ ! -e "$HIGHS_TARGET" ]] \
    || evsp_die "existing native-HiGHS vendor directory failed validation: $HIGHS_TARGET"
  HIGHS_TMP=$(mktemp -d "$HIGHS_VENDOR_ROOT/.highspy-1.15.1.XXXXXX")
  "$PYTHON_BIN" -m pip install --disable-pip-version-check --no-deps \
    --target "$HIGHS_TMP" highspy==1.15.1
  HIGHS_VERSION=$(PYTHONPATH="$HIGHS_TMP" "$PYTHON_BIN" -c \
    'import highspy; assert highspy.Highs().version() == "1.15.1"; print(highspy.Highs().version()); print(highspy.__file__)')
  mv "$HIGHS_TMP" "$HIGHS_TARGET"
  HIGHS_RC=0
fi
printf 'return_code=%s\n%s\n' "$HIGHS_RC" "$HIGHS_VERSION" \
  | tee "$A_ROOT/highs_native_preflight.txt"
cp "$A_ROOT/highs_native_preflight.txt" "$B_ROOT/highs_native_preflight.txt"
find "$HIGHS_TARGET" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$A_ROOT/highs_native_vendor_sha256s.txt"
cp "$A_ROOT/highs_native_vendor_sha256s.txt" \
  "$B_ROOT/highs_native_vendor_sha256s.txt"

A_HIGHS_JOB=""
B_HIGHS_JOB=""
if [[ "$HIGHS_RC" == 0 ]]; then
  A_MANIFEST="$A_ROOT/panel_a_highs_inputs.tsv"
  B_MANIFEST="$B_ROOT/panel_b_highs_inputs.tsv"
  if [[ ! -e "$A_MANIFEST" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_integer_manifest.py" \
      --root "$A_ROOT" --panel A --source-dir cg --out "$A_MANIFEST" \
      --provenance "$A_ROOT/highs_native_provenance.json" \
      --wrapper-commit "$WRAPPER_COMMIT" --solver-commit "$HIGHS_COMMIT"
  fi
  if [[ ! -e "$B_MANIFEST" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_integer_manifest.py" \
      --root "$B_ROOT" --panel B --source-dir frozen_v2 --out "$B_MANIFEST" \
      --provenance "$B_ROOT/highs_native_provenance.json" \
      --wrapper-commit "$WRAPPER_COMMIT" --solver-commit "$HIGHS_COMMIT"
  fi
  mkdir -p "$A_ROOT/mip_highs_native" "$B_ROOT/mip_highs_native"
  A_HIGHS_INDEX_FILE=$(mktemp)
  B_HIGHS_INDEX_FILE=$(mktemp)
  "$PYTHON_BIN" "$SCRIPT_DIR/select_missing_integer_indices.py" \
    --manifest "$A_MANIFEST" --root "$A_ROOT" --panel A --stage mip \
    --artifact-dir "$A_ROOT/mip_highs_native" > "$A_HIGHS_INDEX_FILE"
  "$PYTHON_BIN" "$SCRIPT_DIR/select_missing_integer_indices.py" \
    --manifest "$B_MANIFEST" --root "$B_ROOT" --panel B --stage mip \
    --artifact-dir "$B_ROOT/mip_highs_native" > "$B_HIGHS_INDEX_FILE"
  mapfile -t A_HIGHS_INDICES < "$A_HIGHS_INDEX_FILE"
  mapfile -t B_HIGHS_INDICES < "$B_HIGHS_INDEX_FILE"

  A_HIGHS_ACTIVE=$(queued_job_id eua25_hgh)
  if [[ -n "$A_HIGHS_ACTIVE" ]]; then
    A_HIGHS_JOB="$A_HIGHS_ACTIVE (already active)"
  elif [[ ${#A_HIGHS_INDICES[@]} -gt 0 ]]; then
    A_HIGHS_ARRAY=$(IFS=,; echo "${A_HIGHS_INDICES[*]}")
    A_HIGHS_EXPORTS="ALL,EVSP_EXECUTION_REPO=$HIGHS_EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$HIGHS_COMMIT,EVSP_PANEL=A,EVSP_INTEGER_MANIFEST=$A_MANIFEST,EVSP_HIGHS_OUTPUT_DIR=$A_ROOT/mip_highs_native,EVSP_HIGHS_PYTHONPATH=$HIGHS_TARGET,EVSP_PYTHON=$PYTHON_BIN"
    A_HIGHS_JOB=$(evsp_submit_and_resolve eua25_hgh \
      --array="${A_HIGHS_ARRAY}%54" -p default_partition -c 8 --mem=24G \
      -t 01:15:00 --no-requeue --export="$A_HIGHS_EXPORTS" \
      -o "$A_ROOT/logs/highs_%A_%a.out" -e "$A_ROOT/logs/highs_%A_%a.err" \
      "$SCRIPT_DIR/pool_mip_highs_native.sub")
    {
      echo -e "stage\tarray_job_id\ttasks\tsolver_commit\tbackend\ttimelimit_s\tthreads"
      echo -e "mip_highs_native\t$A_HIGHS_JOB\t${#A_HIGHS_INDICES[@]}\t$HIGHS_COMMIT\thighspy_native\t1800\t8"
    } > "$A_ROOT/highs_native_${A_HIGHS_JOB}_jobs.tsv"
  fi
  B_HIGHS_ACTIVE=$(queued_job_id eub25_hgh)
  if [[ -n "$B_HIGHS_ACTIVE" ]]; then
    B_HIGHS_JOB="$B_HIGHS_ACTIVE (already active)"
  elif [[ ${#B_HIGHS_INDICES[@]} -gt 0 ]]; then
    B_HIGHS_ARRAY=$(IFS=,; echo "${B_HIGHS_INDICES[*]}")
    B_HIGHS_EXPORTS="ALL,EVSP_EXECUTION_REPO=$HIGHS_EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$HIGHS_COMMIT,EVSP_PANEL=B,EVSP_INTEGER_MANIFEST=$B_MANIFEST,EVSP_HIGHS_OUTPUT_DIR=$B_ROOT/mip_highs_native,EVSP_HIGHS_PYTHONPATH=$HIGHS_TARGET,EVSP_PYTHON=$PYTHON_BIN"
    B_HIGHS_JOB=$(evsp_submit_and_resolve eub25_hgh \
      --array="${B_HIGHS_ARRAY}%45" -p default_partition -c 8 --mem=24G \
      -t 01:15:00 --no-requeue --export="$B_HIGHS_EXPORTS" \
      -o "$B_ROOT/logs/highs_%A_%a.out" -e "$B_ROOT/logs/highs_%A_%a.err" \
      "$SCRIPT_DIR/pool_mip_highs_native.sub")
    {
      echo -e "stage\tarray_job_id\ttasks\tsolver_commit\tbackend\ttimelimit_s\tthreads"
      echo -e "mip_highs_native\t$B_HIGHS_JOB\t${#B_HIGHS_INDICES[@]}\t$HIGHS_COMMIT\thighspy_native\t1800\t8"
    } > "$B_ROOT/highs_native_${B_HIGHS_JOB}_jobs.tsv"
  fi
else
  echo "WARNING: native HiGHS unavailable; submitted CG extensions only" >&2
fi

for root in "$A_RESUME_ROOT" "$B_RESUME_ROOT"; do
  sha256sum "$root/execution_plan.json" "$root/matrix.tsv" \
    > "$root/SUBMISSION_INPUT_SHA256SUMS"
done
echo "Panel A 24h resume: ${A_RESUME_JOB:-skipped} (${#A_RESUME_INDICES[@]} tasks)"
echo "Panel B 6h certification: ${B_RESUME_JOB:-skipped} (${#B_RESUME_INDICES[@]} tasks)"
echo "Panel A native HiGHS: ${A_HIGHS_JOB:-skipped}"
echo "Panel B native HiGHS: ${B_HIGHS_JOB:-skipped}"
