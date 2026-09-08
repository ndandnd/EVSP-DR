#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd); source "$SCRIPT_DIR/common.sh"; evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [CAMPAIGN_ROOT]"
REPO=$(evsp_repo_root); BRANCH=$(git -C "$REPO" branch --show-current); [[ -n "$BRANCH" ]] || evsp_die "named branch required"
COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH"|tail -1); EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"; [[ -x "$PYTHON_BIN" ]] || evsp_die "missing Python"
[[ -r /share/apps/software/gurobi/gurobi.lic ]] || evsp_die "Gurobi license unreadable"
ROOT="${1:-$HOME/ladder-lite/legacy_selected10_current2_20260908_${COMMIT:0:7}}"
CG_CONCURRENCY=2; MIP_CONCURRENCY=2
mkdir -p "$(dirname "$ROOT")"; exec 9>"$ROOT.launch.lock"; flock -n 9 || evsp_die "another launcher owns $ROOT"
if [[ ! -e "$ROOT" ]]; then
  "$PYTHON_BIN" "$EXECUTION_REPO/scripts/event_uniform_envelope/prepare_legacy_selected10_current.py" --execution-repo "$EXECUTION_REPO" --execution-commit "$COMMIT" --output-root "$ROOT"
fi
PLAN="$ROOT/execution_plan.json"; MATRIX="$ROOT/matrix.tsv"; POOL_MATRIX="$ROOT/pool_matrix.tsv"
[[ -s "$ROOT/PREPARATION_COMPLETE" && -s "$PLAN" && -s "$MATRIX" && -s "$POOL_MATRIX" ]] || evsp_die "incomplete preparation"
read -r PLAN_SHA MATRIX_SHA POOL_SHA < <("$PYTHON_BIN" - "$PLAN" "$MATRIX" "$POOL_MATRIX" "$COMMIT" <<'PY'
import hashlib,json,sys
p,m,pm,c=sys.argv[1:]; d=json.load(open(p)); h=lambda x:hashlib.sha256(open(x,'rb').read()).hexdigest()
assert d['schema']=='evsp-dr-legacy-selected10-current-sensitivity-v1' and d['execution_commit']==c and d['cells']==2
assert d['matrix_sha256']==h(m) and d['pool_matrix_sha256']==h(pm)
print(h(p),h(m),h(pm))
PY
) || evsp_die "control identity mismatch"
JOBS="$ROOT/jobs.tsv"; [[ -e "$JOBS" ]] || printf 'stage\tjob_id\tarray\tdependency\tpartition\tcpus\tmem\ttime\trequeue\tcommit\tplan_sha256\n' > "$JOBS"
record(){ "$PYTHON_BIN" - "$JOBS" "$@" <<'PY'
import os,sys
with open(sys.argv[1],'a') as f:f.write('\t'.join(sys.argv[2:])+'\n');f.flush();os.fsync(f.fileno())
PY
}
job(){ awk -F'\t' -v s="$1" 'NR>1&&$1==s{print $2}' "$JOBS"; }
tag=$(printf '%s' "$ROOT"|shasum|cut -c1-4)
clear_name(){ local n="$1" c; c=$(squeue --me -h -o '%j'|awk -v n="$n" '$0==n{x++}END{print x+0}'); [[ "$c" == 0 ]]||evsp_die "unrecorded active $n"; c=$(sacct -X -n -P -S "$(date +%F)" --name "$n" -o JobIDRaw 2>/dev/null|awk -F'|' 'NF&&$1!~/\./{x++}END{print x+0}'); [[ "$c" == 0 ]]||evsp_die "unrecorded accounted $n"; }
COMMON="HOME,PATH,USER,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$COMMIT,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_PLAN_SHA256=$PLAN_SHA,EVSP_MATRIX_SHA256=$MATRIX_SHA,EVSP_POOL_MATRIX_SHA256=$POOL_SHA,EVSP_PYTHON=$PYTHON_BIN"
mkdir -p "$ROOT/logs/cache" "$ROOT/logs/cg" "$ROOT/logs/freeze" "$ROOT/logs/mip"
CACHE=$(job cache)
if [[ -z "$CACHE" ]]; then N="l2n${COMMIT:0:4}$tag"; clear_name "$N"; CACHE=$(evsp_submit_and_resolve "$N" --array="0-1%$CG_CONCURRENCY" -p default_partition -c 1 --mem=96G -t 24:30:00 --requeue --signal=B:TERM@180 --open-mode=append --export="$COMMON" -o "$ROOT/logs/cache/%x_%A_%a.out" -e "$ROOT/logs/cache/%x_%A_%a.err" "$EXECUTION_REPO/scripts/event_uniform_envelope/legacy_selected10_cache.sub"); record cache "$CACHE" 0-1 none default_partition 1 96G 24:30:00 true "$COMMIT" "$PLAN_SHA"; fi
CG=$(job cg)
if [[ -z "$CG" ]]; then N="l2c${COMMIT:0:4}$tag"; clear_name "$N"; CG=$(evsp_submit_and_resolve "$N" --array="0-1%$CG_CONCURRENCY" -p default_partition -c 1 --mem=96G -t 08:15:00 --requeue --signal=B:TERM@180 --open-mode=append --dependency="aftercorr:$CACHE" --kill-on-invalid-dep=yes --export="$COMMON" -o "$ROOT/logs/cg/%x_%A_%a.out" -e "$ROOT/logs/cg/%x_%A_%a.err" "$EXECUTION_REPO/scripts/event_uniform_envelope/legacy_selected10_cg.sub"); record cg "$CG" 0-1 "aftercorr:$CACHE" default_partition 1 96G 08:15:00 true "$COMMIT" "$PLAN_SHA"; fi
FREEZE=$(job freeze)
if [[ -z "$FREEZE" ]]; then N="l2f${COMMIT:0:4}$tag"; clear_name "$N"; FREEZE=$(evsp_submit_and_resolve "$N" --array="0-1%$CG_CONCURRENCY" -p default_partition -c 1 --mem=16G -t 02:00:00 --no-requeue --open-mode=append --dependency="aftercorr:$CG" --kill-on-invalid-dep=yes --export="$COMMON" -o "$ROOT/logs/freeze/%x_%A_%a.out" -e "$ROOT/logs/freeze/%x_%A_%a.err" "$EXECUTION_REPO/scripts/event_uniform_envelope/nested_threshold_freeze.sub"); record freeze "$FREEZE" 0-1 "aftercorr:$CG" default_partition 1 16G 02:00:00 false "$COMMIT" "$PLAN_SHA"; fi
MIP=$(job mip)
if [[ -z "$MIP" ]]; then N="l2m${COMMIT:0:4}$tag"; clear_name "$N"; MIP=$(evsp_submit_and_resolve "$N" --array="0-1%$MIP_CONCURRENCY" -p default_partition -c 8 --mem=48G -t 10:30:00 --no-requeue --signal=B:TERM@180 --open-mode=append --dependency="aftercorr:$FREEZE" --kill-on-invalid-dep=yes --export="$COMMON" -o "$ROOT/logs/mip/%x_%A_%a.out" -e "$ROOT/logs/mip/%x_%A_%a.err" "$EXECUTION_REPO/scripts/event_uniform_envelope/nested_threshold_mip8h.sub"); record mip "$MIP" 0-1 "aftercorr:$FREEZE" default_partition 8 48G 10:30:00 false "$COMMIT" "$PLAN_SHA"; fi
echo "Legacy Selected-10 current-algorithm two-cell sensitivity pipeline: $ROOT"; echo "cache $CACHE -> CG $CG -> snapshot $FREEZE -> MIP $MIP"
echo "CG has 28800 seconds excluding network construction; MIP has 28800 total seconds across two stages."
echo "MIP preemption is censored; a retry starts a fresh Gurobi tree."
