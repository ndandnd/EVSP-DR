#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [CAMPAIGN_ROOT]"
REPO=$(evsp_repo_root); BRANCH=$(git -C "$REPO" branch --show-current)
[[ -n "$BRANCH" ]] || evsp_die "named branch required"
COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
ROOT="${1:-$HOME/ladder-lite/matched_tariff_peak12_peak18_${COMMIT:0:7}}"
mkdir -p "$(dirname "$ROOT")"; exec 9>"$ROOT.launch.lock"
flock -n 9 || evsp_die "another launcher owns $ROOT"
if [[ ! -e "$ROOT" ]]; then
  "$PYTHON_BIN" "$EXECUTION_REPO/scripts/event_uniform_envelope/matched_tariff_pilot.py" prepare --root "$ROOT" --commit "$COMMIT" --cg-seconds 14400 --mip-seconds 3600 --tariffs peak12 peak18 --split-stages
fi
PLAN_SHA=$("$PYTHON_BIN" - "$ROOT/plan.json" "$COMMIT" <<'PY'
import hashlib,json,sys
raw=open(sys.argv[1],'rb').read();p=json.loads(raw)
assert p['schema']=='evsp-dr-matched-tariff-pilot-v1' and p['commit']==sys.argv[2]
assert p['layout']=='split_cg_mip' and p['tariff_ids']==['peak12','peak18']
assert len(p['cells'])==8 and p['cg_seconds']==14400 and p['mip_seconds']==3600
assert all(c['fleet']==c['peak_concurrency'] for c in p['cells'])
print(hashlib.sha256(raw).hexdigest())
PY
)
WORKER="$EXECUTION_REPO/scripts/event_uniform_envelope/matched_tariff_split.sub"
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
path,stage,job,dep,partition,cpus,memory,allocation,commit,digest=sys.argv[1:]
with open(path,'x') as f:
 json.dump(dict(stage=stage,job_id=job,dependency=dep,partition=partition,cpus=int(cpus),memory=memory,allocation=allocation,commit=commit,plan_sha256=digest,array='0-7%4',requeue=False),f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
PY
}
CG=$(job cg)
if [[ -z "$CG" ]]; then
  NAME="mtxg${COMMIT:0:4}$tag"; clear_name "$NAME"
  CG=$(evsp_submit_and_resolve "$NAME" --array=0-7%4 -p default_partition -c 1 --mem=16G -t 05:00:00 --no-requeue --open-mode=append --export="$COMMON,EVSP_TARIFF_STAGE=cg" -o "$ROOT/logs/%x_%A_%a.out" -e "$ROOT/logs/%x_%A_%a.err" "$WORKER")
  record cg "$CG" none default_partition 1 16G 05:00:00
fi
MIP=$(job mip)
if [[ -z "$MIP" ]]; then
  NAME="mtxm${COMMIT:0:4}$tag"; clear_name "$NAME"
  MIP=$(evsp_submit_and_resolve "$NAME" --array=0-7%4 -p scaglione --exclude=scaglione-compute-01,scaglione-cpu-04 -c 8 --mem=48G -t 02:00:00 --no-requeue --open-mode=append --dependency="aftercorr:$CG" --kill-on-invalid-dep=yes --export="$COMMON,EVSP_TARIFF_STAGE=mip" -o "$ROOT/logs/%x_%A_%a.out" -e "$ROOT/logs/%x_%A_%a.err" "$WORKER")
  record mip "$MIP" "aftercorr:$CG" scaglione 8 48G 02:00:00
fi
echo "Tariff peak12/peak18: CG $CG -> Scaglione MIP $MIP at $ROOT"
