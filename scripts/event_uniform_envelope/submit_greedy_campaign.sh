#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd); source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [CAMPAIGN_ROOT]"
REPO=$(evsp_repo_root); BRANCH=$(git -C "$REPO" branch --show-current)
[[ -n "$BRANCH" ]] || evsp_die "named branch required"
COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
ROOT="${1:-$HOME/ladder-lite/greedy_event4_${COMMIT:0:7}}"
mkdir -p "$(dirname "$ROOT")"; exec 9>"$ROOT.launch.lock"
flock -n 9 || evsp_die "another launcher owns $ROOT"
if [[ ! -e "$ROOT" ]]; then
  "$PYTHON_BIN" "$EXECUTION_REPO/scripts/event_uniform_envelope/greedy_campaign.py" prepare --root "$ROOT" --commit "$COMMIT"
fi
PLAN_SHA=$("$PYTHON_BIN" - "$ROOT/plan.json" "$COMMIT" <<'PY'
import hashlib,json,sys
raw=open(sys.argv[1],'rb').read();p=json.loads(raw)
assert p['schema']=='evsp-dr-greedy-event-campaign-v1' and p['execution_commit']==sys.argv[2]
assert [c['cell'] for c in p['cells']]==['k03_p2','k05_p2','k06_p1','easy_k10']
assert p['column_pool_treatment']=='GREEDY' and p['mip_concurrency']==2
assert [c['mip_seconds'] for c in p['cells']]==[28800,28800,3600,3600]
assert all(c['cg_seconds']==28800 for c in p['cells'])
print(hashlib.sha256(raw).hexdigest())
PY
)
WORKER="$EXECUTION_REPO/scripts/event_uniform_envelope/greedy_campaign.sub"
WORKER_SHA=$(sha256sum "$WORKER" | awk '{print $1}')
COMMON="HOME,PATH,USER,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$COMMIT,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_PLAN_SHA256=$PLAN_SHA,EVSP_PYTHON=$PYTHON_BIN,EVSP_WORKER_SHA256=$WORKER_SHA"
tag=$(printf '%s' "$ROOT" | shasum | cut -c1-6)
job(){
  [[ -e "$ROOT/submission.$1.json" ]] || return 0
  "$PYTHON_BIN" - "$ROOT/submission.$1.json" "$COMMIT" "$PLAN_SHA" "$1" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));assert p['commit']==sys.argv[2] and p['plan_sha256']==sys.argv[3] and p['stage']==sys.argv[4]
assert str(p['job_id']).isdigit();print(p['job_id'])
PY
}
clear_name(){
  local n="$1" active accounted
  active=$(squeue --me -h -o '%j' | awk -v n="$n" '$0==n{x++}END{print x+0}')
  accounted=$(sacct -X -n -P -S "$(date +%F)" --name "$n" -o JobIDRaw | awk -F'|' 'NF&&$1!~/\./{x++}END{print x+0}')
  [[ "$active" == 0 && "$accounted" == 0 ]] || evsp_die "unrecorded matching $n; inspect before retry"
}
record(){
  "$PYTHON_BIN" - "$ROOT/submission.$1.json" "$@" "$COMMIT" "$PLAN_SHA" <<'PY'
import json,os,sys
path,stage,job,dep,partition,cpus,memory,allocation,array,commit,digest=sys.argv[1:]
with open(path,'x') as f:
 json.dump(dict(stage=stage,job_id=job,dependency=dep,partition=partition,cpus=int(cpus),memory=memory,allocation=allocation,commit=commit,plan_sha256=digest,array=array,requeue=False),f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
PY
}
CACHE=$(job cache)
if [[ -z "$CACHE" ]]; then
  NAME="g4n${COMMIT:0:4}$tag"; clear_name "$NAME"
  CACHE=$(evsp_submit_and_resolve "$NAME" --array=0-3%4 -p default_partition -c 1 --mem=96G -t 04:00:00 --no-requeue --open-mode=append --export="$COMMON,EVSP_GREEDY_STAGE=cache" -o "$ROOT/logs/%x_%A_%a.out" -e "$ROOT/logs/%x_%A_%a.err" "$WORKER")
  record cache "$CACHE" none default_partition 1 96G 04:00:00 0-3%4
fi
CG=$(job cg)
if [[ -z "$CG" ]]; then
  NAME="g4c${COMMIT:0:4}$tag"; clear_name "$NAME"
  CG=$(evsp_submit_and_resolve "$NAME" --array=0-3%4 -p default_partition -c 1 --mem=96G -t 08:30:00 --no-requeue --open-mode=append --dependency="aftercorr:$CACHE" --kill-on-invalid-dep=yes --export="$COMMON,EVSP_GREEDY_STAGE=cg" -o "$ROOT/logs/%x_%A_%a.out" -e "$ROOT/logs/%x_%A_%a.err" "$WORKER")
  record cg "$CG" "aftercorr:$CACHE" default_partition 1 96G 08:30:00 0-3%4
fi
MIP=$(job mip)
if [[ -z "$MIP" ]]; then
  NAME="g4m${COMMIT:0:4}$tag"; clear_name "$NAME"
  MIP=$(evsp_submit_and_resolve "$NAME" --array=0-3%2 -p scaglione --exclude=scaglione-compute-01,scaglione-cpu-04 -c 8 --mem=48G -t 10:00:00 --no-requeue --open-mode=append --dependency="aftercorr:$CG" --kill-on-invalid-dep=yes --export="$COMMON,EVSP_GREEDY_STAGE=mip" -o "$ROOT/logs/%x_%A_%a.out" -e "$ROOT/logs/%x_%A_%a.err" "$WORKER")
  record mip "$MIP" "aftercorr:$CG" scaglione 8 48G 10:00:00 0-3%2
fi
echo "GREEDY event4: cache/seed $CACHE -> CG/freeze $CG -> MIP $MIP at $ROOT"
