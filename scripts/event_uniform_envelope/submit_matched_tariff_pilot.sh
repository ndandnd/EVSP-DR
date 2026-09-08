#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [CAMPAIGN_ROOT]"
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
[[ -n "$BRANCH" ]] || evsp_die "named branch required"
COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
ROOT="${1:-$HOME/ladder-lite/matched_tariff8_20260908_${COMMIT:0:7}}"
mkdir -p "$(dirname "$ROOT")"
exec 9>"$ROOT.launch.lock"
flock -n 9 || evsp_die "another launcher owns $ROOT"
if [[ ! -e "$ROOT" ]]; then
  "$PYTHON_BIN" "$EXECUTION_REPO/scripts/event_uniform_envelope/matched_tariff_pilot.py" prepare --root "$ROOT" --commit "$COMMIT" --cg-seconds 14400 --mip-seconds 14400
fi
PLAN_SHA=$("$PYTHON_BIN" - "$ROOT/plan.json" "$COMMIT" <<'PY'
import hashlib,json,sys
raw=open(sys.argv[1],'rb').read(); p=json.loads(raw)
assert p['schema']=='evsp-dr-matched-tariff-pilot-v1' and p['commit']==sys.argv[2]
assert len(p['cells'])==8 and p['cg_seconds']==14400 and p['mip_seconds']==14400
assert all(c['fleet']==c['peak_concurrency'] for c in p['cells'])
print(hashlib.sha256(raw).hexdigest())
PY
)
if [[ -s "$ROOT/submission.json" ]]; then
  "$PYTHON_BIN" - "$ROOT/submission.json" "$COMMIT" "$PLAN_SHA" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));assert p['commit']==sys.argv[2] and p['plan_sha256']==sys.argv[3]
print('Already submitted matched tariff array',p['job_id'])
PY
  exit 0
fi
tag=$(printf '%s' "$ROOT" | shasum | cut -c1-6)
NAME="mt8${COMMIT:0:4}$tag"
active=$(squeue --me -h -o '%j' | awk -v n="$NAME" '$0==n{x++}END{print x+0}')
accounted=$(sacct -X -n -P -S "$(date +%F)" --name "$NAME" -o JobIDRaw | awk -F'|' 'NF&&$1!~/\./{x++}END{print x+0}')
[[ "$active" == 0 && "$accounted" == 0 ]] || evsp_die "unrecorded submission with matching name; inspect scheduler before retry"
COMMON="HOME,PATH,USER,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$COMMIT,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_PLAN_SHA256=$PLAN_SHA,EVSP_PYTHON=$PYTHON_BIN"
JOB=$(evsp_submit_and_resolve "$NAME" --array=0-7%8 -p default_partition -c 8 --mem=96G -t 12:00:00 --no-requeue --open-mode=append --export="$COMMON" -o "$ROOT/logs/%x_%A_%a.out" -e "$ROOT/logs/%x_%A_%a.err" "$EXECUTION_REPO/scripts/event_uniform_envelope/matched_tariff_pilot.sub")
"$PYTHON_BIN" - "$ROOT/submission.json" "$JOB" "$COMMIT" "$PLAN_SHA" <<'PY'
import json,os,sys
with open(sys.argv[1],'x') as f:
 json.dump(dict(job_id=sys.argv[2],commit=sys.argv[3],plan_sha256=sys.argv[4],array='0-7%8',cpus=8,memory='96G',allocation='12:00:00',requeue=False),f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
PY
echo "Matched tariff pilot array $JOB: $ROOT"
