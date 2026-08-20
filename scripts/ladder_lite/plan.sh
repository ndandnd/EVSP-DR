#!/bin/bash

main() {
  REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) || return 1
  LL_ROOT=${LL_ROOT:-"$HOME/ladder-lite"}
  LL_PYTHON=${LL_PYTHON:-/home/nc437/evsp_env/bin/python3.12}
  LL_CAMPAIGN=${LL_CAMPAIGN:-"ll_$(date -u +%Y%m%d)"}
  CAMPAIGN_DIR="$LL_ROOT/campaign/$LL_CAMPAIGN"; PLAN="$CAMPAIGN_DIR/approved-plan.json"
  MATRIX="$CAMPAIGN_DIR/task_matrix.csv"; PLAN_LOG="$CAMPAIGN_DIR/plan.log"
  COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || return 1
  if git -C "$REPO" symbolic-ref -q HEAD >/dev/null 2>&1; then
    echo "plan.sh requires a detached checkout" >&2; return 1
  fi
  if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    echo "plan.sh requires a tracked-clean checkout" >&2; return 1
  fi
  [ -x "$LL_PYTHON" ] || {
    echo "LL_PYTHON is not executable: $LL_PYTHON" >&2; return 1;
  }
  if [ -e "$PLAN" ] || [ -e "$MATRIX" ]; then
    echo "plan or matrix already exists: $CAMPAIGN_DIR" >&2; return 1
  fi
  mkdir -p "$CAMPAIGN_DIR" || return 1
  (
    cd "$REPO" || exit 1
    "$LL_PYTHON" -B src/launch_scale_ladder.py \
      --campaign "$LL_CAMPAIGN" --python "$LL_PYTHON" \
      --reservation-root "$LL_ROOT" --plan-out "$PLAN" \
      --matrix-out "$MATRIX"
  ) >"$PLAN_LOG" || return 1
  echo "[ll] staging scientific inputs for $LL_CAMPAIGN"; (
    cd "$REPO" || exit 1
    "$LL_PYTHON" -B -c \
      'import json,sys;sys.path.insert(0,"src");import launch_scale_ladder as L;L._stage_scientific_inputs(json.load(open(sys.argv[1])))' \
      "$PLAN"
  ) || return 1
  "$LL_PYTHON" -B - "$PLAN" "$CAMPAIGN_DIR/campaign.json" "$COMMIT" <<'PY' || return 1
import datetime,hashlib,json,sys
praw=open(sys.argv[1],"rb").read(); p=json.loads(praw)
counts={k:len(v) for k,v in sorted(p["task_groups"].items())}
assert sum(counts.values())==200, counts
out={"approval_sha256":hashlib.sha256(praw).hexdigest(),
     "execution_mode":"ladder_lite_direct_array","campaign":p["campaign"],
     "commit":sys.argv[3],"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "submitted":False}
with open(sys.argv[2],"x") as h: json.dump(out,h,indent=2,sort_keys=True);h.write("\n")
print("commit       :",sys.argv[3]);print("plan         :",sys.argv[1])
print("plan sha256  :",out["approval_sha256"]);print("groups       :",counts)
print("total tasks  :",sum(counts.values()))
PY
}
main "$@"
