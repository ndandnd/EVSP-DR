#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd); source "$SCRIPT_DIR/common.sh"; evsp_require_unicorn
ROOT="${1:-$HOME/ladder-lite/legacy_selected10_current2_20260908_bead344}"
REPO=$(evsp_repo_root); BRANCH=$(git -C "$REPO" branch --show-current); [[ -n "$BRANCH" ]] || evsp_die "named branch required"
COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH"|tail -1); EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"; [[ -x "$PYTHON_BIN" ]] || evsp_die "missing Python"
[[ -r /share/apps/software/gurobi/gurobi.lic ]] || evsp_die "Gurobi license unreadable"
[[ -s "$ROOT/jobs.tsv" && -s "$ROOT/execution_plan.json" && -s "$ROOT/pool_matrix.tsv" ]] || evsp_die "source campaign controls missing"
SOURCE_COMMIT=bead34452aa422ec6b2c7799d0c9c9698208aa32
SOURCE_CG_JOB=$(awk -F'\t' '$1=="cg"{print $2}' "$ROOT/jobs.tsv")
SOURCE_FREEZE_JOB=$(awk -F'\t' '$1=="freeze"{print $2}' "$ROOT/jobs.tsv")
SOURCE_MIP_JOB=$(awk -F'\t' '$1=="mip"{print $2}' "$ROOT/jobs.tsv")
[[ "$SOURCE_CG_JOB" == 583344 && "$SOURCE_FREEZE_JOB" == 583346 && "$SOURCE_MIP_JOB" == 583348 ]] || evsp_die "source job ledger mismatch"
if squeue -h -j "$SOURCE_FREEZE_JOB,$SOURCE_MIP_JOB" | grep -q .; then evsp_die "obsolete downstream jobs remain active; cancel 583346 and 583348"; fi
exec 9>"$ROOT.replacement.lock"; flock -n 9 || evsp_die "another replacement launcher owns $ROOT"
"$PYTHON_BIN" "$EXECUTION_REPO/scripts/event_uniform_envelope/prepare_legacy_selected10_freeze_replacement.py" --execution-repo "$EXECUTION_REPO" --execution-commit "$COMMIT" --campaign-root "$ROOT"
REPLACEMENT_PLAN="$ROOT/replacement_freeze_plan.json"; REPLACEMENT_SHA=$(sha256sum "$REPLACEMENT_PLAN"|awk '{print $1}')
read -r SOURCE_PLAN_SHA POOL_SHA < <("$PYTHON_BIN" - "$ROOT/execution_plan.json" "$ROOT/pool_matrix.tsv" <<'PY'
import hashlib,json,sys
p,m=sys.argv[1:]; d=json.load(open(p)); h=lambda x:hashlib.sha256(open(x,'rb').read()).hexdigest()
assert d['schema']=='evsp-dr-legacy-selected10-current-sensitivity-v1' and d['execution_commit']=='bead34452aa422ec6b2c7799d0c9c9698208aa32'
assert d['pool_matrix_sha256']==h(m)
print(h(p),h(m))
PY
) || evsp_die "source control identity mismatch"
JOBS="$ROOT/replacement_jobs.tsv"; [[ ! -e "$JOBS" ]] || evsp_die "replacement job ledger already exists"
printf 'stage\tjob_id\tarray\tdependency\texecution_commit\treplacement_plan_sha256\n' > "$JOBS"
record(){ "$PYTHON_BIN" - "$JOBS" "$@" <<'PY'
import os,sys
with open(sys.argv[1],'a') as f:f.write('\t'.join(sys.argv[2:])+'\n');f.flush();os.fsync(f.fileno())
PY
}
tag=$(printf '%s' "$ROOT"|shasum|cut -c1-4)
FREEZE_COMMON="HOME,PATH,USER,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$COMMIT,EVSP_SOURCE_SOLVER_COMMIT=$SOURCE_COMMIT,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_REPLACEMENT_PLAN=$REPLACEMENT_PLAN,EVSP_REPLACEMENT_PLAN_SHA256=$REPLACEMENT_SHA,EVSP_POOL_MATRIX_SHA256=$POOL_SHA,EVSP_PYTHON=$PYTHON_BIN"
FREEZE=$(evsp_submit_and_resolve "l2rf${COMMIT:0:4}$tag" --array="0-1%2" -p default_partition -c 1 --mem=16G -t 02:00:00 --no-requeue --open-mode=append --dependency="aftercorr:$SOURCE_CG_JOB" --kill-on-invalid-dep=yes --export="$FREEZE_COMMON" -o "$ROOT/logs/freeze/%x_%A_%a.out" -e "$ROOT/logs/freeze/%x_%A_%a.err" "$EXECUTION_REPO/scripts/event_uniform_envelope/legacy_selected10_freeze_physics.sub")
record freeze_replacement "$FREEZE" 0-1 "aftercorr:$SOURCE_CG_JOB" "$COMMIT" "$REPLACEMENT_SHA"
MIP_COMMON="HOME,PATH,USER,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$COMMIT,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_PLAN_SHA256=$SOURCE_PLAN_SHA,EVSP_POOL_MATRIX_SHA256=$POOL_SHA,EVSP_PYTHON=$PYTHON_BIN"
MIP=$(evsp_submit_and_resolve "l2rm${COMMIT:0:4}$tag" --array="0-1%2" -p default_partition -c 8 --mem=48G -t 10:30:00 --no-requeue --signal=B:TERM@180 --open-mode=append --dependency="aftercorr:$FREEZE" --kill-on-invalid-dep=yes --export="$MIP_COMMON" -o "$ROOT/logs/mip/%x_%A_%a.out" -e "$ROOT/logs/mip/%x_%A_%a.err" "$EXECUTION_REPO/scripts/event_uniform_envelope/nested_threshold_mip8h.sub")
record mip_replacement "$MIP" 0-1 "aftercorr:$FREEZE" "$COMMIT" "$REPLACEMENT_SHA"
echo "replacement freeze $FREEZE -> MIP $MIP; source CG $SOURCE_CG_JOB preserved"
