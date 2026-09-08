#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [THRESHOLD_ROOT]"

BASE_ROOT="${1:-$HOME/ladder-lite/threshold_9_15_event_20260904_9bdbb17}"
BASE_ROOT=$(cd "$BASE_ROOT" && pwd)
RESUME_ROOT="${EVSP_RESUME_ROOT:-$BASE_ROOT/cg_resume48h_20260906}"
RESUME_ROOT=$(cd "$RESUME_ROOT" && pwd)
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
[[ -n "$BRANCH" ]] || evsp_die "manager checkout must be on a named branch"
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$WRAPPER_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
[[ -x "$PYTHON_BIN" ]] || evsp_die "missing Python: $PYTHON_BIN"
[[ -r /share/apps/software/gurobi/gurobi.lic ]] || evsp_die "shared Gurobi license unreadable"

SHORT=${WRAPPER_COMMIT:0:10}
MIP_ROOT="${EVSP_POOL_MIP_ROOT:-$BASE_ROOT/pool_mip8h_20260908_$SHORT}"
SOURCE_JOB="${EVSP_SOURCE_DEPENDENCY_JOB:-481176_48}"
MAX_CONCURRENT="${EVSP_MIP_MAX_CONCURRENT:-12}"
MIP_MEM="${EVSP_MIP_MEM:-48G}"
[[ "$SOURCE_JOB" =~ ^[0-9]+_[0-9]+$ ]] || evsp_die "source dependency must be one Slurm array task id"
RESUME_TASK=${SOURCE_JOB##*_}
DERIVED_ACTIVE_INDEX=$(
  "$PYTHON_BIN" - "$RESUME_ROOT/matrix.tsv" "$BASE_ROOT/matrix.tsv" "$RESUME_TASK" <<'PY'
import csv,sys
resume,base,local=sys.argv[1:]
with open(resume,newline="") as f:
    matches=[r for r in csv.DictReader(f,delimiter="\t") if r["local_index"]==local]
assert len(matches)==1
cell=matches[0]["cell"]
with open(base,newline="") as f:
    rows=list(csv.reader(f,delimiter="\t"))
matches=[r[0] for r in rows if len(r)==11 and r[1]==cell]
assert len(matches)==1
print(matches[0])
PY
) || evsp_die "could not map live continuation task to base matrix"
ACTIVE_INDEX="${EVSP_ACTIVE_SOURCE_INDEX:-$DERIVED_ACTIVE_INDEX}"
[[ "$ACTIVE_INDEX" == "$DERIVED_ACTIVE_INDEX" ]] || evsp_die "active index override disagrees with continuation matrix ($DERIVED_ACTIVE_INDEX)"
[[ "$ACTIVE_INDEX" =~ ^[0-9]+$ && "$ACTIVE_INDEX" -ge 0 && "$ACTIVE_INDEX" -lt 70 ]] || evsp_die "active source index must be 0..69"
[[ "$MAX_CONCURRENT" =~ ^[0-9]+$ && "$MAX_CONCURRENT" -ge 2 && "$MAX_CONCURRENT" -le 24 ]] || evsp_die "total concurrency must be 2..24"
[[ "$MIP_MEM" =~ ^[0-9]+[GM]$ ]] || evsp_die "EVSP_MIP_MEM must look like 48G"
READY_CONCURRENT=$((MAX_CONCURRENT-1))

mkdir -p "$(dirname "$MIP_ROOT")"
exec 9>"$MIP_ROOT.launch.lock"
flock -n 9 || evsp_die "another launcher owns $MIP_ROOT"

PREPARER="$EXECUTION_REPO/scripts/event_uniform_envelope/prepare_threshold_pool_mip8h.py"
if [[ ! -e "$MIP_ROOT" ]]; then
  "$PYTHON_BIN" "$PREPARER" --base-root "$BASE_ROOT" --resume-root "$RESUME_ROOT" \
    --execution-repo "$EXECUTION_REPO" --execution-commit "$WRAPPER_COMMIT" \
    --source-dependency-job "$SOURCE_JOB" --active-source-index "$ACTIVE_INDEX" \
    --max-concurrent-total "$MAX_CONCURRENT" --mip-memory "$MIP_MEM" \
    --output-root "$MIP_ROOT"
fi
PLAN="$MIP_ROOT/execution_plan.json"; MATRIX="$MIP_ROOT/matrix.tsv"
[[ -s "$MIP_ROOT/PREPARATION_COMPLETE" && -s "$PLAN" && -s "$MATRIX" ]] || evsp_die "incomplete campaign preparation"
read -r PLAN_SHA MATRIX_SHA < <(
  "$PYTHON_BIN" - "$PLAN" "$MATRIX" "$WRAPPER_COMMIT" "$SOURCE_JOB" <<'PY'
import hashlib,json,sys
p,m,commit,job=sys.argv[1:]
d=json.load(open(p)); assert d["schema"]=="evsp-dr-threshold-raw-pool-mip8h-v1"
assert d["execution_commit"]==commit and d["source_dependency_job"]==job
def h(x): return hashlib.sha256(open(x,"rb").read()).hexdigest()
assert d["matrix_sha256"]==h(m) and d["cells"]==70
print(h(p),h(m))
PY
) || evsp_die "campaign identity validation failed"
"$PYTHON_BIN" - "$PLAN" "$ACTIVE_INDEX" "$MAX_CONCURRENT" "$MIP_MEM" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["active_source_index"]==int(sys.argv[2])
assert d["max_concurrent_total"]==int(sys.argv[3])
assert d["mip"]["memory"]==sys.argv[4]
PY

FREEZE_WORKER="$EXECUTION_REPO/scripts/event_uniform_envelope/threshold_pool_freeze.sub"
MIP_WORKER="$EXECUTION_REPO/scripts/event_uniform_envelope/threshold_pool_mip8h.sub"
FREEZER="$EXECUTION_REPO/src/freeze_terminal_exact_cg_pool.py"
RUNNER="$EXECUTION_REPO/src/run_exact_pool_mip.py"
file_sha() { local value; value=$(sha256sum "$1"); printf '%s\n' "${value%% *}"; }
FREEZE_WORKER_SHA=$(file_sha "$FREEZE_WORKER"); MIP_WORKER_SHA=$(file_sha "$MIP_WORKER")
FREEZER_SHA=$(file_sha "$FREEZER"); RUNNER_SHA=$(file_sha "$RUNNER")
"$PYTHON_BIN" - "$PLAN" "$FREEZE_WORKER_SHA" "$MIP_WORKER_SHA" "$FREEZER_SHA" "$RUNNER_SHA" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))["code_sha256"]
keys=("scripts/event_uniform_envelope/threshold_pool_freeze.sub","scripts/event_uniform_envelope/threshold_pool_mip8h.sub","src/freeze_terminal_exact_cg_pool.py","src/run_exact_pool_mip.py")
assert all(d[k]==v for k,v in zip(keys,sys.argv[2:]))
PY

READY_INDICES="0-$((ACTIVE_INDEX-1)),$((ACTIVE_INDEX+1))-69"
[[ "$ACTIVE_INDEX" == 0 ]] && READY_INDICES="1-69"
[[ "$ACTIVE_INDEX" == 69 ]] && READY_INDICES="0-68"
JOBS="$MIP_ROOT/jobs.tsv"
if [[ ! -e "$JOBS" ]]; then
  printf 'stage\tjob_id\tarray\tdependency\tcommit\tplan_sha256\tmatrix_sha256\tmem\tcpus\tpartition\ttime\trequeue\n' > "$JOBS"
fi

record_job() {
  "$PYTHON_BIN" - "$JOBS" "$@" <<'PY'
import os,sys
p,*values=sys.argv[1:]
with open(p,"a",encoding="utf-8") as f:
    f.write("\t".join(values)+"\n"); f.flush(); os.fsync(f.fileno())
PY
}
job_for() { awk -F'\t' -v stage="$1" 'NR>1 && $1==stage {print $2}' "$JOBS"; }
ROOT_TAG=$(printf '%s' "$MIP_ROOT" | shasum | cut -c1-4)
refuse_unrecorded() {
  local name="$1" seen
  seen=$(squeue --me -h -o '%j' | awk -v n="$name" '$0==n {c++} END{print c+0}')
  [[ "$seen" == 0 ]] || evsp_die "unrecorded active job named $name; refusing duplicate"
  seen=$(sacct -X -n -P -S "$(date +%F)" --name "$name" -o JobIDRaw 2>/dev/null | awk -F'|' 'NF&&$1!~/\./{c++} END{print c+0}')
  [[ "$seen" == 0 ]] || evsp_die "unrecorded job named $name exists in today's accounting; refusing duplicate"
}

COMMON_EXPORTS="HOME,PATH,USER,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$WRAPPER_COMMIT,EVSP_POOL_MIP_ROOT=$MIP_ROOT,EVSP_POOL_PLAN_SHA256=$PLAN_SHA,EVSP_POOL_MATRIX_SHA256=$MATRIX_SHA,EVSP_PYTHON=$PYTHON_BIN"
FREEZE_EXPORTS="$COMMON_EXPORTS,EVSP_FREEZE_WORKER_SHA256=$FREEZE_WORKER_SHA,EVSP_FREEZER_SHA256=$FREEZER_SHA"
MIP_EXPORTS="$COMMON_EXPORTS,EVSP_MIP_WORKER_SHA256=$MIP_WORKER_SHA,EVSP_MIP_RUNNER_SHA256=$RUNNER_SHA"
mkdir -p "$MIP_ROOT/logs/freeze" "$MIP_ROOT/logs/mip"

FR_READY=$(job_for freeze_ready)
if [[ -z "$FR_READY" ]]; then
  NAME="tpfR${SHORT:0:5}$ROOT_TAG"; refuse_unrecorded "$NAME"
  FR_READY=$(evsp_submit_and_resolve "$NAME" --array="$READY_INDICES%$READY_CONCURRENT" \
    -p scaglione -c 1 --mem=16G -t 04:00:00 --no-requeue --open-mode=append \
    --export="$FREEZE_EXPORTS" -o "$MIP_ROOT/logs/freeze/%x_%A_%a.out" \
    -e "$MIP_ROOT/logs/freeze/%x_%A_%a.err" "$FREEZE_WORKER")
  record_job freeze_ready "$FR_READY" "$READY_INDICES" none "$WRAPPER_COMMIT" "$PLAN_SHA" "$MATRIX_SHA" 16G 1 scaglione 04:00:00 false
fi
FR_ACTIVE=$(job_for freeze_active)
if [[ -z "$FR_ACTIVE" ]]; then
  NAME="tpfD${SHORT:0:5}$ROOT_TAG"; refuse_unrecorded "$NAME"
  FR_ACTIVE=$(evsp_submit_and_resolve "$NAME" --array="$ACTIVE_INDEX" \
    -p scaglione -c 1 --mem=16G -t 04:00:00 --no-requeue --open-mode=append \
    --dependency="afterany:$SOURCE_JOB" --kill-on-invalid-dep=yes \
    --export="$FREEZE_EXPORTS" -o "$MIP_ROOT/logs/freeze/%x_%A_%a.out" \
    -e "$MIP_ROOT/logs/freeze/%x_%A_%a.err" "$FREEZE_WORKER")
  record_job freeze_active "$FR_ACTIVE" "$ACTIVE_INDEX" "afterany:$SOURCE_JOB" "$WRAPPER_COMMIT" "$PLAN_SHA" "$MATRIX_SHA" 16G 1 scaglione 04:00:00 false
fi
MIP_READY=$(job_for mip_ready)
if [[ -z "$MIP_READY" ]]; then
  NAME="tpmR${SHORT:0:5}$ROOT_TAG"; refuse_unrecorded "$NAME"
  MIP_READY=$(evsp_submit_and_resolve "$NAME" --array="$READY_INDICES%$READY_CONCURRENT" \
    -p scaglione -c 8 --mem="$MIP_MEM" -t 10:30:00 --no-requeue --open-mode=append \
    --dependency="aftercorr:$FR_READY" --kill-on-invalid-dep=yes --export="$MIP_EXPORTS" \
    -o "$MIP_ROOT/logs/mip/%x_%A_%a.out" -e "$MIP_ROOT/logs/mip/%x_%A_%a.err" "$MIP_WORKER")
  record_job mip_ready "$MIP_READY" "$READY_INDICES" "aftercorr:$FR_READY" "$WRAPPER_COMMIT" "$PLAN_SHA" "$MATRIX_SHA" "$MIP_MEM" 8 scaglione 10:30:00 false
fi
MIP_ACTIVE=$(job_for mip_active)
if [[ -z "$MIP_ACTIVE" ]]; then
  NAME="tpmD${SHORT:0:5}$ROOT_TAG"; refuse_unrecorded "$NAME"
  MIP_ACTIVE=$(evsp_submit_and_resolve "$NAME" --array="$ACTIVE_INDEX" \
    -p scaglione -c 8 --mem="$MIP_MEM" -t 10:30:00 --no-requeue --open-mode=append \
    --dependency="afterok:$FR_ACTIVE" --kill-on-invalid-dep=yes --export="$MIP_EXPORTS" \
    -o "$MIP_ROOT/logs/mip/%x_%A_%a.out" -e "$MIP_ROOT/logs/mip/%x_%A_%a.err" "$MIP_WORKER")
  record_job mip_active "$MIP_ACTIVE" "$ACTIVE_INDEX" "afterok:$FR_ACTIVE" "$WRAPPER_COMMIT" "$PLAN_SHA" "$MATRIX_SHA" "$MIP_MEM" 8 scaglione 10:30:00 false
fi

echo "Prepared immutable campaign: $MIP_ROOT"
echo "Ready freeze/MIP: $FR_READY -> $MIP_READY (69 cells, max $MAX_CONCURRENT)"
echo "Deferred cell $ACTIVE_INDEX: $SOURCE_JOB -> $FR_ACTIVE -> $MIP_ACTIVE"
echo "Each MIP gets 28800 total solver seconds across its two stages; Slurm wall 10:30:00."
